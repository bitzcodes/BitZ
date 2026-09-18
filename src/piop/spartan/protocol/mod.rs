//! The one BitZ protocol every relation runs.
//!
//! A relation describes itself once — its committed bit tensor, its
//! constraint matrices, how its assignment blocks map to bit slots, the
//! domain strings it binds under — and this module runs the protocol of
//! paper §2.1 for it:
//!
//! 1. bind the statement (the relation's assignment binding and, when the
//!    relation runs them, the opener policy and Round 0 of the opening),
//! 2. grind and draw the Step-2 prime, project the relation,
//! 3. run the Spartan PIOP over the runtime field, every drawn challenge
//!    behind one grinding boundary at the profile difficulty,
//! 4. bitify the terminal claim into one functional over the committed bit
//!    tensor, bind it, grind the terminal boundary,
//! 5. discharge that functional: directly through the runtime-q exponent
//!    fold ([`prove`]), through a virtual map onto a derived grid
//!    ([`prove_virtual`]), or — Strategy 2 — after an exact integer lift
//!    and a grinded fresh-prime reduction ([`prove_reduced`]).
//!
//! Transcripts are byte-for-byte those of the protocols this module
//! replaced; the pins in `tests/transcript_state_pins.rs` hold them.

use crate::piop::spartan::SpartanField as _;
use field::{RingOps, Uint};
pub mod binding;
pub mod bitify;
pub mod linear;

use std::{borrow::Cow, sync::OnceLock};

use flock_core::pcs::{
    commit::Commitment,
    ligerito::{
        LigeritoProfile, ProverConfig as LigProverConfig, VerifierConfig as LigVerifierConfig,
    },
};

use thiserror::Error;

use {
    crate::{
        ext_proj::PrimeSamplingError,
        ligerito::{LOG_PACKING, packed_vars},
        ligerito_flock::{
            FlockCommitHint, FlockRsError, IntEvalRsLigModQProof, IntEvalRsLigVirtProof,
            LigeritoSelection, ModQOpeningKind, OodRound, ProverOod, ResolvedLigerito, VerifierOod,
            bind_prover_ood, bind_verifier_ood, commit_rs_ligerito_rows,
            prove_mle_eval_mod_q_ligerito_virtual_runtime,
            prove_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_runtime,
            prove_mle_eval_mod_q_ligerito_with_weight_chunks,
            verify_mle_eval_mod_q_ligerito_virtual_runtime,
            verify_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_runtime,
            verify_mle_eval_mod_q_ligerito_with_weight_chunks_runtime,
        },
        pcs::IntegerMatrixLayout,
        poly::{mle::DenseMultilinearExtension, univariate::binary_gf128::Gf128},
        transcript::traits::Transcript,
    },
    circuit::linear_map::binary::VirtualMap,
};

use super::{
    ConstraintMatricesSkeleton, PreparedConstraintMatrices, SpartanField, absorb_spartan_message,
    grinding::{
        GrindingError, ProverGrindingTranscript, VerifierGrindingTranscript,
        grind_and_absorb_in_domain, verify_and_absorb_in_domain,
    },
    matrix::{self, ScaledMleEvaluationClaim, SpartanMatrixCoefficient},
    multiswap::reduce::{step50_accepts_lift, step50_integer_lift, step50_reduce},
    piop::{
        SpartanError, SpartanPiopProof, prove_spartan_piop_field,
        prove_spartan_piop_native_u64_borrowed,
        prove_spartan_piop_native_u64_with_univariate_skip_borrowed,
        prove_spartan_piop_raw_products_native_assignment,
        prove_spartan_piop_raw_products_raw_witness, verify_spartan_proof,
        verify_spartan_univariate_skip_proof,
    },
    profile::{IopInstanceFacts, IopSecurityParams, IopSecurityProfile, Lambda100, ProfileError},
    raw_monty::{NativeProducts, RawMontyCoefficient, RawWitness},
    sumcheck::R1csProductMles,
    univariate_skip::UnivariateSkipSpartanPiopProof,
};

pub use binding::BindingHasher;
pub use bitify::{BitifiedClaim, BitifiedRows, BlockTable, ScaleSide, SlotRange};

/// Runtime-configured Spartan field used by every BitZ relation.
pub type SpartanBitzField = field::Fp<2>;

/// The runtime field configuration.
pub type FieldConfig = <SpartanBitzField as crate::piop::spartan::SpartanField>::Config;

/// Embedded, validator-gated Ligerito profiles begin at a 22-variable
/// committed bit MLE: seven slot variables plus fifteen gate variables.
pub const MIN_PRODUCTION_GATE_VARS: usize = 15;

/// Failures in layout validation, claim translation, or either proof system.
#[derive(Debug, Error)]
pub enum ProtocolError {
    /// A relation-level failure (layout, witness or matrix construction).
    #[error(transparent)]
    Relation(Box<dyn std::error::Error + Send + Sync + 'static>),

    #[error(transparent)]
    Spartan(#[from] SpartanError),

    #[error("failed to derive a Ligerito configuration: {0}")]
    LigeritoConfig(String),

    #[error("the BitZ opening rejected: {0:?}")]
    Bitz(FlockRsError),

    #[error("the sampled modulus is not supported by the runtime field")]
    UnsupportedFieldModulus,

    #[error("the prepared relation does not match the witness layout")]
    RelationWitnessLayoutMismatch,

    #[error("the compact bit rows do not match the relation layout")]
    InvalidBitRows,

    #[error("the BitZ parameters are invalid for the relation layout")]
    InvalidBitzParameters,

    #[error("the combined Spartan/BitZ proof requires at least 2^15 gate slots")]
    UnauditedBitzParameters,

    #[error("the commitment parameters do not match the derived BitZ configuration")]
    CommitmentConfigMismatch,

    #[error("the commitment does not match the prepared terminal-opening statement")]
    PreparedOpeningCommitmentMismatch,

    #[error("the terminal Spartan claim has the wrong point shape")]
    InvalidClaimPoint,

    #[error("a terminal Spartan claim element does not use the sampled runtime field")]
    ClaimFieldMismatch,

    #[error("a constant-only terminal claim has a nonzero adjusted value")]
    InvalidConstantOnlyClaim,

    #[error("a host length does not fit the canonical transcript encoding")]
    BindingEncodingOverflow,

    #[error("the block table is not a complete little-endian block map")]
    InvalidBlockTable,

    /// The security profile could not be instantiated at this shape.
    #[error(transparent)]
    Profile(#[from] ProfileError),

    /// Runtime-prime sampling failed.
    #[error(transparent)]
    PrimeSampling(#[from] PrimeSamplingError),

    /// The full-width fingerprint search found no prime.
    #[error("no full-width fingerprint prime was found within {attempts} draws")]
    FingerprintSearchExhausted { attempts: usize },

    /// A Fiat--Shamir grinding nonce could not be produced or checked.
    #[error(transparent)]
    Grinding(#[from] GrindingError),

    /// The relation supports the other prime strategy only.
    #[error("the security profile's prime strategy does not match the relation")]
    UnsupportedProfile,

    /// A proof selected a different univariate-prefix width than the one
    /// whose degree was included in the prepared security profile.
    #[error("the proof uses univariate skip K={actual}; expected K={expected}")]
    UnexpectedUnivariateSkipVariables { expected: u8, actual: u8 },

    /// The derived prime interval must keep the row weights to one
    /// exponent-fold chunk (`q_bits <= c_w`); the profile guarantees this,
    /// so a violation is an internal error.
    #[error("the runtime prime produced a multi-chunk row functional")]
    MultiChunkRuntimeWeights,

    /// The relation's PIOP witness representation does not fit its kernel.
    #[error("the relation's PIOP witness representation does not fit its kernel")]
    UnsupportedKernel,

    /// The relation offers no matrices for this prime.
    #[error("the relation does not project its matrices to the runtime prime")]
    MatrixSourceUnavailable,

    /// The prepared relation carries no opener configuration for this side.
    #[error("the prepared relation carries no opener configuration for this side")]
    OpenerConfigUnavailable,

    /// The relation offers no discharge of this kind.
    #[error("the relation does not support this discharge")]
    UnsupportedDischarge,

    /// The Step 5.0 integer lift fails the mod-`Q` or magnitude check.
    #[error("the Step 5.0 integer lift is inconsistent with the mod-Q claim")]
    InvalidIntegerLift,

    /// A linear relation, its map, witness and grids disagree.
    #[error("the relation or witness geometry is inconsistent")]
    InvalidGeometry,

    /// The packed native prefix kernel supports only K = 0, ..., 4.
    #[error("the inner prefix must be in 0..={max}, got {actual}")]
    InvalidInnerPrefix { actual: usize, max: usize },

    /// The legacy prover's final inner-sumcheck claim did not equal
    /// `V(r_h) H(r_h)`.
    #[error("the inner sumcheck has an inconsistent terminal product")]
    InvalidInnerTerminalClaim,

    /// The committed source assignment does not contain its shared leading one.
    #[error("the source assignment has a malformed shared constant cell")]
    InvalidSharedConstant,

    /// The public statement must carry one entry per committed instance.
    #[error("public statement length mismatch: expected {expected}, got {actual}")]
    InvalidPublicStatementLength { expected: usize, actual: usize },

    /// A chain statement's blocks or digest disagree with the witness the
    /// prover was handed.
    #[error("the chain statement does not match the witness")]
    ChainStatementMismatch,

    /// An explicit Ligerito configuration disagrees with the prepared
    /// security profile.
    #[error("the Ligerito configuration does not match the prepared profile")]
    MismatchedLigeritoConfig,

    /// A SHA-256 runtime-prime profile or draw failed.
    #[error(transparent)]
    Prime(#[from] super::sha256::Sha256PrimeError),

    /// A SHA-256 relation projection failed.
    #[error(transparent)]
    Constraint(#[from] super::sha256::Sha256ConstraintError),
}

impl ProtocolError {
    /// Wraps a relation-level error.
    pub fn relation(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Relation(Box::new(error))
    }
}

/// The transcript domain strings and profiler scope labels of one relation.
#[derive(Clone, Copy, Debug)]
pub struct Domains {
    /// Tag of the statement frame carrying the assignment binding.
    pub statement_tag: &'static [u8],
    /// Domain bound before the Step-2 prime draw.
    pub prime_sampling: &'static [u8],
    /// Grinding domains of the initial, per-draw PIOP and terminal boundaries.
    pub initial_grinding: &'static [u8],
    pub piop_grinding: &'static [u8],
    pub terminal_grinding: &'static [u8],
    /// Domain of the bridge digest bound between Spartan and BitZ (the direct
    /// discharge).
    pub bitified_claim: &'static [u8],
    /// The direct opener's statement domain.
    pub opening: ModQOpeningKind,
    /// Tag of the frame binding the bitified claim before a virtual or
    /// reduced discharge.
    pub claim_tag: &'static [u8],
    /// Strategy 2: the grinding domain before the reduction draw and the
    /// reduction prime's sampling domain.
    pub reduction_grinding: &'static [u8],
    pub reduction_prime: &'static [u8],
    /// Profiler scope labels.
    pub scopes: Scopes,
}

/// Per-relation span constructors keep tracing callsites and labels static.
#[derive(Clone, Copy, Debug)]
pub struct Scopes {
    pub relation_projection_prove: fn() -> tracing::Span,
    pub relation_projection_verify: fn() -> tracing::Span,
    pub witness_projection_prove: fn() -> tracing::Span,
    pub spartan_prove: fn() -> tracing::Span,
    pub spartan_verify: fn() -> tracing::Span,
    pub bitify_prover: fn() -> tracing::Span,
    pub bitify_verifier: fn() -> tracing::Span,
    pub bitz_prove: fn() -> tracing::Span,
    pub bitz_verify: fn() -> tracing::Span,
    pub bitz_prepare_prover: fn() -> tracing::Span,
    pub bitz_prepare_verifier: fn() -> tracing::Span,
}

/// Builds the [`Scopes`] of a relation from its scope prefix.
#[macro_export]
macro_rules! protocol_scopes {
    ($prefix:literal) => {
        $crate::piop::spartan::protocol::Scopes {
            relation_projection_prove: || {
                tracing::info_span!(concat!($prefix, ":relation_projection_prove"))
            },
            relation_projection_verify: || {
                tracing::info_span!(concat!($prefix, ":relation_projection_verify"))
            },
            witness_projection_prove: || {
                tracing::info_span!(concat!($prefix, ":witness_projection_prove"))
            },
            spartan_prove: || tracing::info_span!(concat!($prefix, ":spartan_prove")),
            spartan_verify: || tracing::info_span!(concat!($prefix, ":spartan_verify")),
            bitify_prover: || tracing::info_span!(concat!($prefix, ":bitify_prover")),
            bitify_verifier: || tracing::info_span!(concat!($prefix, ":bitify_verifier")),
            bitz_prove: || tracing::info_span!(concat!($prefix, ":bitz_prove")),
            bitz_verify: || tracing::info_span!(concat!($prefix, ":bitz_verify")),
            bitz_prepare_prover: || tracing::info_span!(concat!($prefix, ":bitz_prepare_prover")),
            bitz_prepare_verifier: || tracing::info_span!(concat!($prefix, ":bitz_prepare_verifier")),
        }
    };
}

/// Which Spartan outer reduction the relation runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernel {
    /// The cubic outer sumcheck over every row variable.
    Plain,
    /// A known-zero univariate prefix skip over the `skip_vars` low row
    /// variables, then the cubic tail.
    UnivariateSkip { skip_vars: usize },
}

/// Which prime policy the relation's profile must carry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimeStrategy {
    /// One derived-width Step-2 prime.
    Single,
    /// Strategy 2: a full-width fingerprint prime and a Step-5.0 reduction
    /// prime.
    TwoPrime,
}

/// The optional steps of the statement phase and the PIOP.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    /// Bind the opener policy digest after the statement.
    pub policy_bind: bool,
    /// Run Round 0 (the out-of-domain sample) before the prime draw.
    pub ood_round: bool,
    /// Wrap every PIOP draw in a grinding boundary at the profile difficulty.
    pub piop_grinding: bool,
    /// Where a nonzero Spartan scale goes in the bitified claim.
    pub scale_side: ScaleSide,
    pub strategy: PrimeStrategy,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            policy_bind: true,
            ood_round: true,
            piop_grinding: true,
            scale_side: ScaleSide::Rows,
            strategy: PrimeStrategy::Single,
        }
    }
}

/// Instantiates a prime-independent skeleton at a runtime prime.
type SkeletonProjection<C> =
    fn(
        &ConstraintMatricesSkeleton<SpartanBitzField, C>,
        &FieldConfig,
    ) -> Result<PreparedConstraintMatrices<SpartanBitzField, C>, matrix::SpartanMatrixError>;

/// Where a relation's constraint matrices come from at the runtime prime.
pub enum MatrixSource<C> {
    /// Prime-independent coefficients: prepared once, instantiated per draw
    /// in `O(log nnz)` with a bit-identical digest (build with
    /// [`MatrixSource::skeleton`]).
    Skeleton {
        skeleton: ConstraintMatricesSkeleton<SpartanBitzField, C>,
        project: SkeletonProjection<C>,
    },
    /// Matrices at a fixed prime (the relation never draws one).
    Fixed(PreparedConstraintMatrices<SpartanBitzField, C>),
    /// Projected by the relation at every draw
    /// ([`RelationSpec::project_matrices`]).
    PerPrime,
}

impl<C> MatrixSource<C>
where
    C: SpartanMatrixCoefficient<SpartanBitzField> + RawMontyCoefficient,
{
    /// Prepares prime-independent matrices once.
    pub fn skeleton(matrices: super::ConstraintMatrices<C>) -> Result<Self, ProtocolError>
    where
        C: matrix::ModulusIndependentCoefficient<SpartanBitzField>,
    {
        let skeleton = ConstraintMatricesSkeleton::new(matrices).map_err(SpartanError::from)?;
        Ok(Self::Skeleton {
            skeleton,
            project: PreparedConstraintMatrices::from_skeleton,
        })
    }

    fn project<S: RelationSpec<Coefficient = C>>(
        &self,
        spec: &S,
        config: &FieldConfig,
    ) -> Result<Cow<'_, PreparedConstraintMatrices<SpartanBitzField, C>>, ProtocolError> {
        match self {
            Self::Skeleton { skeleton, project } => Ok(Cow::Owned(
                project(skeleton, config).map_err(SpartanError::from)?,
            )),
            Self::Fixed(matrices) => {
                if matrices.field_modulus_encoding()
                    != SpartanBitzField::canonical_modulus_encoding(config)
                {
                    return Err(ProtocolError::UnsupportedFieldModulus);
                }
                Ok(Cow::Borrowed(matrices))
            }
            Self::PerPrime => Ok(Cow::Owned(spec.project_matrices(config)?)),
        }
    }
}

/// The prover's view of a witness at the runtime prime: whichever table
/// representation the relation's Spartan kernel consumes.
pub enum PiopWitness<'w> {
    /// Exact `u64` products and a borrowed native assignment (the leading
    /// entries of the padded column domain; the rest is zero).
    Native {
        az: Cow<'w, [u64]>,
        bz: Cow<'w, [u64]>,
        cz: Cow<'w, [u64]>,
        assignment: &'w [u64],
        /// Public constant block established by a typed witness constructor.
        constant_prefix: Option<super::raw_monty::NativeConstantPrefix>,
    },
    Mul32(&'w super::MulWitness<u32>),
    Mul64(&'w super::MulWitness<u64>),
    Mul128(&'w super::MulWitness<u128>),
    CmAnd(&'w super::CmAndWitness),
    /// Exact 4096-bit row operands with a borrowed declared-width assignment.
    IntegerProducts {
        products: crate::sumcheck::outer::OuterInputs<field::Uint<64>>,
        witness: RawWitness<'w>,
    },
    /// Complete padded assignment in canonical Montgomery representation for
    /// the sampled prime, matching `RawWitness::Field`. The generic driver
    /// derives row products from its already-projected public matrices.
    FieldAssignment {
        assignment: Vec<u128>,
    },
    /// Field-valued tables for the delayed reduction kernel.
    Field {
        products: R1csProductMles<SpartanBitzField>,
        assignment: DenseMultilinearExtension<SpartanBitzField>,
    },
}

/// What a relation sees when it binds its bitified claim before a virtual
/// or reduced discharge.
pub struct ClaimFrame<'a> {
    pub field: &'a FieldConfig,
    pub binding: &'a [u8; 32],
    pub matrices_digest: &'a [u8; 32],
    pub terminal_claim: &'a ScaledMleEvaluationClaim<SpartanBitzField>,
    pub opening: &'a BitifiedClaim,
    /// Dense canonical row weights over the committed tensor.
    pub row_weights: &'a [u128],
    /// Canonical clear column weights.
    pub col_weights: &'a [u128],
    /// Strategy 2: the exact integer lift of the claim.
    pub mu_prime: Option<&'a field::Uint<5>>,
}

/// The static description of one relation: everything the protocol needs
/// besides a witness.
pub trait RelationSpec: Sync {
    /// The constraint-matrix coefficient type.
    type Coefficient: SpartanMatrixCoefficient<SpartanBitzField> + RawMontyCoefficient + Send + Sync;
    /// The prover's witness.
    type Witness: ?Sized + Sync;
    /// The virtual map of a virtual discharge (any map type for relations
    /// that discharge directly).
    type Map: VirtualMap;

    fn domains(&self) -> &'static Domains;

    fn schedule(&self) -> Schedule {
        Schedule::default()
    }

    /// Geometry of the committed bit tensor.
    fn committed_layout(&self) -> IntegerMatrixLayout;
    /// Geometry after the public virtual map; ordinary relations use the source.
    fn opening_layout(&self) -> IntegerMatrixLayout {
        self.committed_layout()
    }
    /// Integer magnitude bound; a smaller width requires a padding map.
    fn opening_word_bits(&self) -> usize {
        self.opening_layout().word_bits
    }

    /// Number of gate coordinates of the assignment MLE (the assignment has
    /// `gate_vars + selector_vars` variables).
    fn gate_vars(&self) -> usize;

    /// The public statement facts the security-profile derivation consumes.
    fn instance_facts(&self) -> IopInstanceFacts;

    /// The constraint matrices.
    fn matrices(&self) -> Result<MatrixSource<Self::Coefficient>, ProtocolError>;

    /// Projects the matrices to a runtime prime
    /// ([`MatrixSource::PerPrime`] relations).
    fn project_matrices(
        &self,
        config: &FieldConfig,
    ) -> Result<PreparedConstraintMatrices<SpartanBitzField, Self::Coefficient>, ProtocolError> {
        let _ = config;
        Err(ProtocolError::MatrixSourceUnavailable)
    }

    /// Rejects layouts the protocol cannot run.
    fn validate_geometry(&self) -> Result<(), ProtocolError>;

    /// How the assignment blocks map to bit slots.
    fn block_table(&self) -> BlockTable;

    fn kernel(&self) -> Kernel;

    /// Uniform difficulty for the opener's wrapped challenges.
    fn opener_grinding_bits(&self, security: &IopSecurityParams) -> u32 {
        security.forest_round_grinding_bits
    }

    fn check_witness(&self, witness: &Self::Witness) -> Result<(), ProtocolError>;

    /// The digest binding layout, commitment and the profile's public
    /// parameters — everything fixed BEFORE the prime draw.
    fn assignment_binding(
        &self,
        commitment: &Commitment,
        security: &IopSecurityParams,
        ligerito: &LigProverConfig,
    ) -> Result<[u8; 32], ProtocolError>;

    /// Writes the relation's constants section of the bridge digest (between
    /// the runtime prime and the terminal claim; direct discharge only).
    fn hash_bridge_constants(&self, hasher: &mut BindingHasher) -> Result<(), ProtocolError> {
        let _ = hasher;
        Ok(())
    }

    /// Draws (or fixes) the runtime prime. Default: the profile interval
    /// under the relation's sampling domain.
    fn runtime_prime<T: Transcript>(
        &self,
        transcript: &mut T,
        security: &IopSecurityParams,
    ) -> Result<field::FpCtx<2>, ProtocolError> {
        sample_mod_q(
            transcript,
            self.domains().prime_sampling,
            security.projection_min,
            security.projection_max,
        )
    }

    /// The witness tables the Spartan kernel consumes at the runtime prime.
    fn piop_witness<'w>(
        &self,
        witness: &'w Self::Witness,
        config: &FieldConfig,
    ) -> Result<PiopWitness<'w>, ProtocolError>;

    /// The virtual map of a virtual discharge.
    fn map(&self) -> Option<&Self::Map> {
        None
    }

    /// The derived rows the virtual opening runs against (`None`: the
    /// committed rows themselves).
    fn derived_rows(&self, witness: &Self::Witness) -> Option<Vec<Vec<u64>>> {
        let _ = witness;
        None
    }

    /// The digest binding the bitified claim before a virtual or reduced
    /// discharge (absorbed under the relation's `claim_tag`).
    fn claim_digest(&self, frame: ClaimFrame<'_>) -> Result<[u8; 32], ProtocolError> {
        let _ = frame;
        Err(ProtocolError::UnsupportedDischarge)
    }
}

/// The opener configuration of a prepared relation.
#[derive(Clone)]
pub enum Opener {
    /// A resolved Ligerito selection (with its policy digest and Round-0
    /// accounting).
    Resolved(ResolvedLigerito),
    /// Explicit prover/verifier configurations (either side may be absent
    /// on a one-sided context).
    Custom {
        prover: Option<LigProverConfig>,
        verifier: Option<LigVerifierConfig>,
    },
}

impl Opener {
    pub fn prover(&self) -> Result<&LigProverConfig, ProtocolError> {
        match self {
            Self::Resolved(resolved) => Ok(resolved.prover()),
            Self::Custom { prover, .. } => prover
                .as_ref()
                .ok_or(ProtocolError::OpenerConfigUnavailable),
        }
    }

    pub fn verifier(&self) -> Result<&LigVerifierConfig, ProtocolError> {
        match self {
            Self::Resolved(resolved) => Ok(resolved.verifier()),
            Self::Custom { verifier, .. } => verifier
                .as_ref()
                .ok_or(ProtocolError::OpenerConfigUnavailable),
        }
    }

    /// The prover configuration for statement bindings (the verifier's copy
    /// carries the same public fields).
    fn binding_config(&self) -> Result<Cow<'_, LigProverConfig>, ProtocolError> {
        match self {
            Self::Resolved(resolved) => Ok(Cow::Borrowed(resolved.prover())),
            Self::Custom {
                prover: Some(prover),
                ..
            } => Ok(Cow::Borrowed(prover)),
            Self::Custom {
                prover: None,
                verifier: Some(verifier),
            } => Ok(Cow::Owned(LigProverConfig {
                recursive_steps: verifier.recursive_steps,
                initial_log_msg_cols: verifier.initial_log_msg_cols,
                initial_log_num_interleaved: verifier.initial_log_num_interleaved,
                initial_k: verifier.initial_k,
                log_inv_rates: verifier.log_inv_rates.clone(),
                recursive_log_msg_cols: verifier.recursive_log_msg_cols.clone(),
                recursive_ks: verifier.recursive_ks.clone(),
                queries: verifier.queries.clone(),
                grinding_bits: verifier.grinding_bits.clone(),
                fold_grinding_bits: verifier.fold_grinding_bits.clone(),
                ood_samples: verifier.ood_samples.clone(),
                merkle_hash: verifier.merkle_hash,
            })),
            Self::Custom {
                prover: None,
                verifier: None,
            } => Err(ProtocolError::OpenerConfigUnavailable),
        }
    }

    fn resolved(&self) -> Option<&ResolvedLigerito> {
        match self {
            Self::Resolved(resolved) => Some(resolved),
            Self::Custom { .. } => None,
        }
    }
}

/// The prime-independent prefix of a relation: the constraint matrices
/// with their prime-independent preparation, the relation description and
/// the instantiated security profile — everything the Spartan PIOP and
/// bitification consume, and nothing of the opener. Compositions that
/// discharge the bitified claim through their own opener (the hybrid of
/// [`crate::hybrid`]) prepare this directly.
pub struct PreparedRelationPrefix<S: RelationSpec> {
    spec: S,
    matrices: MatrixSource<S::Coefficient>,
    security: IopSecurityParams,
}

impl<S: RelationSpec> PreparedRelationPrefix<S> {
    /// Prepares the prefix under an explicit profile.
    pub fn new<P: IopSecurityProfile>(spec: S) -> Result<Self, ProtocolError> {
        spec.validate_geometry()?;
        let security = instantiate_profile::<P, S>(&spec)?;
        Self::with_security(spec, security)
    }

    /// Prepares the prefix under already-instantiated security parameters.
    pub fn with_security(spec: S, security: IopSecurityParams) -> Result<Self, ProtocolError> {
        spec.validate_geometry()?;
        let matrices = spec.matrices()?;
        Ok(Self {
            spec,
            matrices,
            security,
        })
    }

    /// The relation description this prefix was prepared from.
    pub const fn layout(&self) -> &S {
        &self.spec
    }

    /// BitZ geometry of the committed bit tensor.
    pub fn params(&self) -> IntegerMatrixLayout {
        self.spec.committed_layout()
    }

    /// The instantiated security parameters and their accounting.
    pub const fn security(&self) -> &IopSecurityParams {
        &self.security
    }

    pub const fn matrices(&self) -> &MatrixSource<S::Coefficient> {
        &self.matrices
    }

    /// The prime-independent skeleton, for relations prepared from one.
    pub fn skeleton(&self) -> Option<&ConstraintMatricesSkeleton<SpartanBitzField, S::Coefficient>> {
        match &self.matrices {
            MatrixSource::Skeleton { skeleton, .. } => Some(skeleton),
            _ => None,
        }
    }
}

/// The setup-once, prime-independent bundle of a relation: its
/// [`PreparedRelationPrefix`] plus the opener configuration.
pub struct PreparedRelation<S: RelationSpec> {
    prefix: PreparedRelationPrefix<S>,
    selection: Option<LigeritoSelection>,
    opener: Opener,
}

impl<S: RelationSpec> PreparedRelation<S> {
    /// Prepares the relation at the default [`Lambda100`] profile.
    pub fn new(spec: S) -> Result<Self, ProtocolError> {
        Self::new_with_profile::<Lambda100>(spec)
    }

    /// Prepares the relation under an explicit profile with the profile's
    /// default opener (Johnson+OOD at 100 bits).
    pub fn new_with_profile<P: IopSecurityProfile>(spec: S) -> Result<Self, ProtocolError> {
        Self::new_with_profile_and_ligerito::<P>(
            spec,
            LigeritoSelection::for_target(P::LIGERITO_TARGET_BITS),
        )
    }

    /// Prepares the relation under an explicit profile and an explicit
    /// Ligerito opener geometry.
    pub fn new_with_profile_and_ligerito<P: IopSecurityProfile>(
        spec: S,
        selection: LigeritoSelection,
    ) -> Result<Self, ProtocolError> {
        let prefix = PreparedRelationPrefix::new::<P>(spec)?;
        Self::with_ligerito(prefix, selection)
    }

    /// Completes a prefix with a resolved Ligerito selection at the
    /// profile's target.
    pub fn with_ligerito(
        mut prefix: PreparedRelationPrefix<S>,
        selection: LigeritoSelection,
    ) -> Result<Self, ProtocolError> {
        if prefix.spec.gate_vars() < MIN_PRODUCTION_GATE_VARS {
            return Err(ProtocolError::UnauditedBitzParameters);
        }
        let p = prefix.params();
        let ligerito = selection
            .resolve(packed_variables(&p)?, prefix.security.ligerito_target_bits)
            .map_err(ProtocolError::LigeritoConfig)?;
        prefix.security.adopt_ood_round(ligerito.ood_bits())?;
        validate_config_pair(&p, ligerito.prover(), ligerito.verifier())?;
        Ok(Self {
            prefix,
            selection: Some(selection),
            opener: Opener::Resolved(ligerito),
        })
    }

    /// Completes a prefix with explicit opener configurations (no policy
    /// digest, no Round 0).
    pub fn with_opener_configs(
        prefix: PreparedRelationPrefix<S>,
        prover: Option<LigProverConfig>,
        verifier: Option<LigVerifierConfig>,
    ) -> Result<Self, ProtocolError> {
        if let (Some(pc), Some(vc)) = (&prover, &verifier) {
            validate_config_pair(&prefix.params(), pc, vc)?;
        }
        Ok(Self {
            prefix,
            selection: None,
            opener: Opener::Custom { prover, verifier },
        })
    }

    /// The opener-independent prefix.
    pub const fn prefix(&self) -> &PreparedRelationPrefix<S> {
        &self.prefix
    }

    /// Drops the opener and returns the prefix.
    pub fn into_prefix(self) -> PreparedRelationPrefix<S> {
        self.prefix
    }

    /// The relation description this bundle was prepared from.
    pub const fn layout(&self) -> &S {
        &self.prefix.spec
    }

    /// BitZ geometry of the committed bit tensor.
    pub fn params(&self) -> IntegerMatrixLayout {
        self.prefix.params()
    }

    /// The instantiated security parameters and their accounting.
    pub const fn security(&self) -> &IopSecurityParams {
        &self.prefix.security
    }

    /// The opener geometry this relation was prepared with, if it came
    /// from a Ligerito selection.
    pub const fn ligerito(&self) -> Option<LigeritoSelection> {
        self.selection
    }

    /// The resolved opener of a relation prepared through a Ligerito
    /// selection.
    ///
    /// # Panics
    ///
    /// On a relation prepared from explicit opener configurations; use
    /// [`Self::opener`] there.
    pub fn ligerito_configuration(&self) -> &ResolvedLigerito {
        self.opener
            .resolved()
            .expect("the relation was prepared through a Ligerito selection")
    }

    pub const fn opener(&self) -> &Opener {
        &self.opener
    }

    /// The prime-independent skeleton, for relations prepared from one.
    pub fn skeleton(&self) -> Option<&ConstraintMatricesSkeleton<SpartanBitzField, S::Coefficient>> {
        self.prefix.skeleton()
    }

    /// The digest binding layout, commitment and the profile's public
    /// parameters — everything fixed BEFORE the prime draw.
    pub fn assignment_binding(&self, commitment: &Commitment) -> Result<[u8; 32], ProtocolError> {
        assignment_binding(&self.prefix, &self.opener, commitment)
    }
}

/// The statement binding of a prefix under an opener configuration.
fn assignment_binding<S: RelationSpec>(
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    commitment: &Commitment,
) -> Result<[u8; 32], ProtocolError> {
    let ligerito = opener.binding_config()?;
    prefix
        .spec
        .assignment_binding(commitment, &prefix.security, &ligerito)
}

/// Binds the statement: the assignment binding frame, then (per the
/// relation's schedule) the opener policy digest and Round 0.
fn bind_prover_statement<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    hint: &FlockCommitHint,
) -> Result<([u8; 32], ProverOod), ProtocolError> {
    let spec = &prefix.spec;
    let schedule = spec.schedule();
    let binding = assignment_binding(prefix, opener, &hint.commitment)?;
    absorb_spartan_message(transcript, spec.domains().statement_tag, &binding);
    if schedule.policy_bind {
        if let Some(resolved) = opener.resolved() {
            resolved.bind(transcript);
        }
    }
    let ood = if schedule.ood_round {
        bind_prover_ood(transcript, hint, prefix.security.ood)
    } else {
        ProverOod::from(prefix.security.ood)
    };
    Ok((binding, ood))
}

fn bind_verifier_statement<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    commitment: &Commitment,
    round: Option<&OodRound>,
) -> Result<([u8; 32], VerifierOod), ProtocolError> {
    let spec = &prefix.spec;
    let schedule = spec.schedule();
    let binding = assignment_binding(prefix, opener, commitment)?;
    absorb_spartan_message(transcript, spec.domains().statement_tag, &binding);
    if schedule.policy_bind {
        if let Some(resolved) = opener.resolved() {
            resolved.bind(transcript);
        }
    }
    let ood = if schedule.ood_round {
        bind_verifier_ood(
            transcript,
            packed_variables(&prefix.params())?,
            prefix.security.ood,
            round,
        )
        .map_err(ProtocolError::Bitz)?
    } else {
        VerifierOod::from(prefix.security.ood)
    };
    Ok((binding, ood))
}

/// Instantiates the profile `P` at the relation's instance facts and checks
/// its prime strategy against the relation's schedule.
pub fn instantiate_profile<P: IopSecurityProfile, S: RelationSpec>(
    spec: &S,
) -> Result<IopSecurityParams, ProtocolError> {
    let security = P::instantiate(&spec.instance_facts())?;
    let two_prime = security.projection_full_width || security.reduction.is_some();
    let expected = spec.schedule().strategy == PrimeStrategy::TwoPrime;
    if two_prime != expected {
        return Err(ProtocolError::UnsupportedProfile);
    }
    Ok(security)
}

/// The Spartan part of a proof, in the shape the relation's kernel produces.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpartanProof {
    Plain(SpartanPiopProof<SpartanBitzField>),
    UnivariateSkip(UnivariateSkipSpartanPiopProof<SpartanBitzField>),
}

impl SpartanProof {
    /// Field elements in the payload, for analytic size accounting.
    pub fn payload_elements(&self) -> usize {
        match self {
            Self::Plain(proof) => {
                4 * proof.outer.sumcheck.round_polynomials.len()
                    + 3
                    + 3 * proof.inner.round_polynomials.len()
            }
            Self::UnivariateSkip(proof) => {
                proof.outer.skip.finite_q_evaluations.len()
                    + 1
                    + 4 * proof.outer.tail.sumcheck.round_polynomials.len()
                    + 3
                    + 3 * proof.inner.round_polynomials.len()
            }
        }
    }

    /// The plain cubic-outer proof, if that is the kernel's shape.
    pub const fn plain(&self) -> Option<&SpartanPiopProof<SpartanBitzField>> {
        match self {
            Self::Plain(proof) => Some(proof),
            Self::UnivariateSkip(_) => None,
        }
    }

    /// The univariate-skip proof, if that is the kernel's shape.
    pub const fn univariate_skip(
        &self,
    ) -> Option<&UnivariateSkipSpartanPiopProof<SpartanBitzField>> {
        match self {
            Self::UnivariateSkip(proof) => Some(proof),
            Self::Plain(_) => None,
        }
    }
}

/// The transcript messages of the protocol prefix (steps 2–4): the grinding
/// nonces around the prime draw and the opening, and the Spartan reduction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanPrefixProof {
    pub initial_nonce: u64,
    pub piop_nonces: Vec<u64>,
    pub spartan: SpartanProof,
    pub terminal_nonce: u64,
}

/// Strategy 2: the exact integer lift of the bitified claim and the nonce
/// before the reduction-prime draw.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReductionProof {
    pub mu_prime: field::Uint<5>,
    pub nonce: u64,
}

/// An opening proof of either discharge kind.
pub trait OpeningProof: Clone {
    fn to_bytes(&self) -> Vec<u8>;
    fn grinding_nonces(&self) -> &[u64];
    fn ood(&self) -> Option<&OodRound>;
}

impl OpeningProof for IntEvalRsLigModQProof {
    fn to_bytes(&self) -> Vec<u8> {
        IntEvalRsLigModQProof::to_bytes(self)
    }

    fn grinding_nonces(&self) -> &[u64] {
        &self.grinding_nonces
    }

    fn ood(&self) -> Option<&OodRound> {
        self.ood.as_ref()
    }
}

impl OpeningProof for IntEvalRsLigVirtProof {
    fn to_bytes(&self) -> Vec<u8> {
        IntEvalRsLigVirtProof::to_bytes(self)
    }

    fn grinding_nonces(&self) -> &[u64] {
        &self.grinding_nonces
    }

    fn ood(&self) -> Option<&OodRound> {
        self.ood.as_ref()
    }
}

/// A proof over a transcript-selected prime: the Spartan reduction, the
/// optional Step-5.0 lift, the BitZ opening and any profile-selected
/// grinding nonces.
/// The prepared relation determines which opening proof is accepted.
#[derive(Clone)]
pub enum Opening {
    Direct(IntEvalRsLigModQProof),
    Virtual(IntEvalRsLigVirtProof),
}
impl Opening {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Direct(p) => p.to_bytes(),
            Self::Virtual(p) => p.to_bytes(),
        }
    }
    pub fn ood(&self) -> Option<&OodRound> {
        match self {
            Self::Direct(p) => p.ood.as_ref(),
            Self::Virtual(p) => p.ood.as_ref(),
        }
    }
    pub fn direct(&self) -> Option<&IntEvalRsLigModQProof> {
        match self {
            Self::Direct(p) => Some(p),
            _ => None,
        }
    }
    pub fn direct_mut(&mut self) -> Option<&mut IntEvalRsLigModQProof> {
        match self {
            Self::Direct(p) => Some(p),
            _ => None,
        }
    }
}
impl OpeningProof for Opening {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
    fn grinding_nonces(&self) -> &[u64] {
        match self {
            Self::Direct(p) => &p.grinding_nonces,
            Self::Virtual(p) => &p.grinding_nonces,
        }
    }
    fn ood(&self) -> Option<&OodRound> {
        self.ood()
    }
}

#[derive(Clone)]
pub struct Proof<O: OpeningProof = Opening> {
    prefix: SpartanPrefixProof,
    reduction: Option<ReductionProof>,
    bitz: O,
}

impl<O: OpeningProof> Proof<O> {
    fn map_opening<P: OpeningProof>(self, map: impl FnOnce(O) -> P) -> Proof<P> {
        Proof {
            prefix: self.prefix,
            reduction: self.reduction,
            bitz: map(self.bitz),
        }
    }
    pub fn opening_grinding_nonces(&self) -> &[u64] {
        self.bitz.grinding_nonces()
    }

    /// Spartan outer and inner sumcheck proofs.
    pub const fn spartan(&self) -> &SpartanProof {
        &self.prefix.spartan
    }

    /// The BitZ opening proof.
    pub const fn bitz(&self) -> &O {
        &self.bitz
    }

    /// The prefix messages.
    pub const fn prefix(&self) -> &SpartanPrefixProof {
        &self.prefix
    }

    /// The Step-5.0 lift, on reduced discharges.
    pub const fn reduction(&self) -> Option<&ReductionProof> {
        self.reduction.as_ref()
    }

    pub const fn initial_nonce(&self) -> u64 {
        self.prefix.initial_nonce
    }

    pub const fn terminal_nonce(&self) -> u64 {
        self.prefix.terminal_nonce
    }

    pub fn piop_nonces(&self) -> &[u64] {
        &self.prefix.piop_nonces
    }

    /// Field elements in the Spartan payload, for analytic size accounting.
    pub fn spartan_payload_elements(&self) -> usize {
        self.prefix.spartan.payload_elements()
    }

    /// Transmitted grinding nonces (initial/terminal boundaries when armed,
    /// the per-draw PIOP nonces, and the forest section in the BitZ stream).
    pub fn grinding_nonce_count(&self, security: &IopSecurityParams) -> usize {
        usize::from(security.initial_grinding_bits > 0)
            + usize::from(security.terminal_grinding_bits > 0)
            + self.prefix.piop_nonces.len()
            + self.bitz.grinding_nonces().len()
    }

    /// Serialized size in bytes of the proof: the Spartan payload as 16-byte
    /// field elements, the nonces as 8-byte words, and the BitZ opening's
    /// exact codec bytes.
    pub fn size_bytes(&self, security: &IopSecurityParams) -> usize {
        self.spartan_payload_elements() * 16
            + (self.grinding_nonce_count(security) - self.bitz.grinding_nonces().len()) * 8
            + self.bitz.to_bytes().len()
    }

    /// Splits the proof into its parts (tests and codecs).
    pub fn into_parts(self) -> (SpartanPrefixProof, Option<ReductionProof>, O) {
        (self.prefix, self.reduction, self.bitz)
    }

    /// Assembles a proof from its parts (tests and codecs).
    pub const fn from_parts(
        prefix: SpartanPrefixProof,
        reduction: Option<ReductionProof>,
        bitz: O,
    ) -> Self {
        Self {
            prefix,
            reduction,
            bitz,
        }
    }

    /// Step 5.0 exact integer lift of the bitified terminal claim (reduced
    /// discharges).
    pub fn mu_prime(&self) -> Option<&field::Uint<5>> {
        self.reduction.as_ref().map(|reduction| &reduction.mu_prime)
    }

    /// Grinding nonce immediately before the reduction-prime draw (reduced
    /// discharges).
    pub fn reduction_nonce(&self) -> Option<u64> {
        self.reduction.as_ref().map(|reduction| reduction.nonce)
    }

    /// Serialized byte length of the Step 5.0 integer lift (0 without one).
    pub fn mu_prime_bytes(&self) -> usize {
        self.mu_prime().map_or(0, |_| 40)
    }

    /// Mutable access to the opening proof (tests).
    pub fn bitz_mut(&mut self) -> &mut O {
        &mut self.bitz
    }

    /// Mutable access to the prefix messages (tests).
    pub fn prefix_mut(&mut self) -> &mut SpartanPrefixProof {
        &mut self.prefix
    }
}

/// Commits prebuilt compact bit rows under the prepared relation's opener
/// configuration.
///
/// Accepting ownership of `rows` lets benchmarks time bitification separately
/// and move the packed store into the commitment without retaining a duplicate.
pub fn commit<S: RelationSpec>(
    prepared: &PreparedRelation<S>,
    rows: Vec<Vec<u64>>,
) -> Result<FlockCommitHint, ProtocolError> {
    let p = prepared.params();
    prepared.prefix.spec.validate_geometry()?;
    validate_bit_rows(&p, &rows)?;
    let pc = prepared.opener.prover()?;
    if let Ok(vc) = prepared.opener.verifier() {
        validate_config_pair(&p, pc, vc)?;
    }

    // All assertion-bearing shape requirements of the low-level commit have
    // been checked above.
    let hint = commit_rs_ligerito_rows(&p, rows, pc);
    validate_commitment(&p, &hint.commitment, pc)?;
    Ok(hint)
}

/// Proves a one-prime relation, selecting direct or virtual discharge from
/// the prepared relation's public map. Strategy 2 uses `prove_reduced`.
pub fn prove<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prepared: &PreparedRelation<S>,
    witness: &S::Witness,
    hint: &FlockCommitHint,
) -> Result<Proof, ProtocolError> {
    prove_with_opener(
        transcript,
        &prepared.prefix,
        &prepared.opener,
        witness,
        hint,
    )
}

/// [`prove`] over a prefix and an explicit opener.
pub fn prove_with_opener<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    witness: &S::Witness,
    hint: &FlockCommitHint,
) -> Result<Proof, ProtocolError> {
    if prefix.spec.map().is_some() {
        return prove_virtual_with_opener(transcript, prefix, opener, witness, hint)
            .map(|p| p.map_opening(Opening::Virtual));
    }
    let spec = &prefix.spec;
    spec.check_witness(witness)?;
    let p = prefix.params();
    let pc = opener.prover()?;
    validate_bit_rows(&p, hint.rows())?;
    validate_commitment(&p, &hint.commitment, pc)?;
    let security = &prefix.security;
    let domains = spec.domains();
    let scopes = &domains.scopes;

    let (binding, ood) = bind_prover_statement(transcript, prefix, opener, hint)?;
    let proved = prove_piop(transcript, prefix, witness, &binding)?;
    let prime = &proved.prime;

    // Steps 5.1–5.3: the runtime-q BitZ opening (one chunk by construction).
    let bitz = {
        let _step5 = tracing::info_span!("step5:open_prove").entered();
        let _scope = (scopes.bitz_prove)().entered();
        let chunks = {
            let _scope = (scopes.bitz_prepare_prover)().entered();
            bitify::prepare_chunks(&proved.opening, &proved.table, prime.modulus_bits(), &prime)?
        };
        if chunks.len() != 1 {
            return Err(ProtocolError::MultiChunkRuntimeWeights);
        }
        prove_mle_eval_mod_q_ligerito_with_weight_chunks(
            transcript,
            domains.opening,
            hint,
            &p,
            &chunks,
            &proved.bridge_digest,
            prime.modulus_bits(),
            bitz_generator(),
            spec.opener_grinding_bits(security),
            ood,
            pc,
        )
        .map_err(ProtocolError::Bitz)?
    };

    Ok(Proof {
        prefix: proved.messages,
        reduction: None,
        bitz: Opening::Direct(bitz),
    })
}

/// Verifies the relation-selected discharge, re-deriving the prime from the
/// bound transcript and rejecting an incompatible opening variant.
pub fn verify<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prepared: &PreparedRelation<S>,
    commitment: &Commitment,
    proof: &Proof,
) -> Result<(), ProtocolError> {
    verify_with_opener(
        transcript,
        &prepared.prefix,
        &prepared.opener,
        commitment,
        proof,
    )
}

/// [`verify`] over a prefix and an explicit opener.
pub fn verify_with_opener<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    commitment: &Commitment,
    proof: &Proof,
) -> Result<(), ProtocolError> {
    let bitz = match (&proof.bitz, prefix.spec.map().is_some()) {
        (Opening::Direct(p), false) => p,
        (Opening::Virtual(p), true) => {
            return verify_virtual_parts(transcript, prefix, opener, commitment, &proof.prefix, p);
        }
        _ => return Err(ProtocolError::UnsupportedDischarge),
    };
    let spec = &prefix.spec;
    let p = prefix.params();
    let vc = opener.verifier()?;
    let binding_config = opener.binding_config()?;
    validate_commitment(&p, commitment, &binding_config)?;
    let security = &prefix.security;
    let domains = spec.domains();
    let scopes = &domains.scopes;
    check_proof_kernel(spec.kernel(), &proof.prefix.spartan)?;

    let (binding, ood) =
        bind_verifier_statement(transcript, prefix, opener, commitment, bitz.ood.as_ref())?;
    let verified = verify_piop(transcript, prefix, &binding, &proof.prefix)?;
    let prime = &verified.prime;

    let _step5 = tracing::info_span!("step5:open_verify").entered();
    let _scope = (scopes.bitz_verify)().entered();
    let (chunks, col_weights_q) = {
        let _scope = (scopes.bitz_prepare_verifier)().entered();
        let chunks = bitify::prepare_chunks(
            &verified.opening,
            &verified.table,
            prime.modulus_bits(),
            &prime,
        )?;
        let col_weights_q: Vec<u128> = bitify::column_weights(&verified.opening, &prime)?;
        (chunks, col_weights_q)
    };
    if chunks.len() != 1 {
        return Err(ProtocolError::MultiChunkRuntimeWeights);
    }
    verify_mle_eval_mod_q_ligerito_with_weight_chunks_runtime(
        transcript,
        domains.opening,
        commitment,
        bitz,
        &p,
        &chunks,
        &col_weights_q,
        &verified.bridge_digest,
        bitz_generator(),
        verified.opening.claimed,
        prime.modulus_u128(),
        prime.modulus_bits(),
        spec.opener_grinding_bits(security),
        ood,
        vc,
    )
    .map_err(ProtocolError::Bitz)
}

/// Binds the bitified claim frame of a virtual or reduced discharge.
fn bind_claim_frame<T: Transcript, S: RelationSpec>(
    transcript: &mut T,
    spec: &S,
    frame: ClaimFrame<'_>,
) -> Result<(), ProtocolError> {
    let digest = spec.claim_digest(frame)?;
    absorb_spartan_message(transcript, spec.domains().claim_tag, &digest);
    Ok(())
}

fn virtual_weight_chunks<S: RelationSpec>(
    spec: &S,
    weights: &[u128],
    q_bits: usize,
) -> Result<crate::pcs::ModQWeightChunks, ProtocolError> {
    let p = spec.opening_layout();
    let width = spec.opening_word_bits();
    let result = if width < p.word_bits {
        crate::pcs::ModQWeightChunks::from_dense_padded(
            &p,
            weights,
            q_bits,
            width,
            spec.map().ok_or(ProtocolError::UnsupportedDischarge)?,
        )
    } else if width == p.word_bits {
        crate::pcs::ModQWeightChunks::from_dense(&p, weights, q_bits)
    } else {
        Err(())
    };
    result.map_err(|_| ProtocolError::InvalidBitzParameters)
}

/// Proves the relation, discharging the bitified claim through the
/// relation's virtual map onto its derived grid at the runtime prime.
pub fn prove_virtual<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prepared: &PreparedRelation<S>,
    witness: &S::Witness,
    hint: &FlockCommitHint,
) -> Result<Proof<IntEvalRsLigVirtProof>, ProtocolError> {
    prove_virtual_with_opener(
        transcript,
        &prepared.prefix,
        &prepared.opener,
        witness,
        hint,
    )
}

/// [`prove_virtual`] over a prefix and an explicit opener.
pub fn prove_virtual_with_opener<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    witness: &S::Witness,
    hint: &FlockCommitHint,
) -> Result<Proof<IntEvalRsLigVirtProof>, ProtocolError> {
    let spec = &prefix.spec;
    spec.check_witness(witness)?;
    let p = prefix.params();
    let pc = opener.prover()?;
    validate_bit_rows(&p, hint.rows())?;
    validate_commitment(&p, &hint.commitment, pc)?;
    let security = &prefix.security;
    let domains = spec.domains();
    let scopes = &domains.scopes;
    let map = spec.map().ok_or(ProtocolError::UnsupportedDischarge)?;

    let (binding, ood) = bind_prover_statement(transcript, prefix, opener, hint)?;
    let proved = prove_piop(transcript, prefix, witness, &binding)?;
    let prime = &proved.prime;

    let row_weights = bitify::dense_row_weights(&proved.opening, &proved.table, &prime)?;
    let col_weights: Vec<u128> = bitify::column_weights(&proved.opening, &prime)?;
    bind_claim_frame(
        transcript,
        spec,
        ClaimFrame {
            field: prime,
            binding: &binding,
            matrices_digest: &proved.matrices_digest,
            terminal_claim: &proved.terminal_claim,
            opening: &proved.opening,
            row_weights: &row_weights,
            col_weights: &col_weights,
            mu_prime: None,
        },
    )?;

    let bitz = {
        let _step5 = tracing::info_span!("step5:open_prove").entered();
        let _scope = (scopes.bitz_prove)().entered();
        let derived = spec.derived_rows(witness);
        let h_rows = derived.as_deref().unwrap_or(hint.rows());
        let chunks = virtual_weight_chunks(spec, &row_weights, prime.modulus_bits())?;
        prove_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_runtime(
            transcript,
            hint,
            h_rows,
            &spec.opening_layout(),
            &p,
            map,
            &chunks,
            prime.modulus_u128(),
            prime.modulus_bits(),
            bitz_generator(),
            spec.opener_grinding_bits(security),
            ood,
            pc,
        )
        .map_err(ProtocolError::Bitz)?
    };

    Ok(Proof {
        prefix: proved.messages,
        reduction: None,
        bitz,
    })
}

/// Verifies a virtual-discharge proof.
pub fn verify_virtual<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prepared: &PreparedRelation<S>,
    commitment: &Commitment,
    proof: &Proof<IntEvalRsLigVirtProof>,
) -> Result<(), ProtocolError> {
    verify_virtual_with_opener(
        transcript,
        &prepared.prefix,
        &prepared.opener,
        commitment,
        proof,
    )
}

/// [`verify_virtual`] over a prefix and an explicit opener.
pub fn verify_virtual_with_opener<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    commitment: &Commitment,
    proof: &Proof<IntEvalRsLigVirtProof>,
) -> Result<(), ProtocolError> {
    verify_virtual_parts(
        transcript,
        prefix,
        opener,
        commitment,
        &proof.prefix,
        &proof.bitz,
    )
}

fn verify_virtual_parts<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    opener: &Opener,
    commitment: &Commitment,
    messages: &SpartanPrefixProof,
    bitz: &IntEvalRsLigVirtProof,
) -> Result<(), ProtocolError> {
    let spec = &prefix.spec;
    let p = prefix.params();
    let vc = opener.verifier()?;
    let binding_config = opener.binding_config()?;
    validate_commitment(&p, commitment, &binding_config)?;
    let security = &prefix.security;
    let domains = spec.domains();
    let scopes = &domains.scopes;
    let map = spec.map().ok_or(ProtocolError::UnsupportedDischarge)?;
    check_proof_kernel(spec.kernel(), &messages.spartan)?;

    let (binding, ood) =
        bind_verifier_statement(transcript, prefix, opener, commitment, bitz.ood.as_ref())?;
    let verified = verify_piop(transcript, prefix, &binding, messages)?;
    let prime = &verified.prime;

    let row_weights = bitify::dense_row_weights(&verified.opening, &verified.table, &prime)?;
    let col_weights: Vec<u128> = bitify::column_weights(&verified.opening, &prime)?;
    bind_claim_frame(
        transcript,
        spec,
        ClaimFrame {
            field: prime,
            binding: &binding,
            matrices_digest: &verified.matrices_digest,
            terminal_claim: &verified.terminal_claim,
            opening: &verified.opening,
            row_weights: &row_weights,
            col_weights: &col_weights,
            mu_prime: None,
        },
    )?;

    let _step5 = tracing::info_span!("step5:open_verify").entered();
    let _scope = (scopes.bitz_verify)().entered();
    let chunks = virtual_weight_chunks(spec, &row_weights, prime.modulus_bits())?;
    verify_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_runtime(
        transcript,
        commitment,
        bitz,
        &spec.opening_layout(),
        &p,
        map,
        &chunks,
        &col_weights,
        bitz_generator(),
        verified.opening.claimed,
        prime.modulus_u128(),
        prime.modulus_bits(),
        spec.opener_grinding_bits(security),
        ood,
        vc,
    )
    .map_err(ProtocolError::Bitz)
}

/// Proves the relation under Strategy 2: after bitification the prover
/// sends the exact integer lift `mu'` of the mod-`Q` tensor claim; a
/// grinded fresh reduction prime `q'` re-projects the claim below the
/// exponent-fold no-wrap boundary, and the runtime-`q'` virtual opening of
/// the committed rows (under the relation's identity map) discharges it.
pub fn prove_reduced<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prepared: &PreparedRelation<S>,
    witness: &S::Witness,
    hint: &FlockCommitHint,
) -> Result<Proof<IntEvalRsLigVirtProof>, ProtocolError> {
    let prefix = &prepared.prefix;
    let opener = &prepared.opener;
    let spec = &prefix.spec;
    spec.check_witness(witness)?;
    let p = prepared.params();
    let pc = opener.prover()?;
    validate_bit_rows(&p, hint.rows())?;
    validate_commitment(&p, &hint.commitment, pc)?;
    let security = &prefix.security;
    let reduction = security
        .reduction
        .ok_or(ProtocolError::UnsupportedProfile)?;
    let domains = spec.domains();
    let scopes = &domains.scopes;
    let map = spec.map().ok_or(ProtocolError::UnsupportedDischarge)?;

    let (binding, ood) = bind_prover_statement(transcript, prefix, opener, hint)?;
    let proved = prove_piop(transcript, prefix, witness, &binding)?;
    let prime = &proved.prime;
    let row_weights = bitify::dense_row_weights(&proved.opening, &proved.table, &prime)?;
    let col_weights: Vec<u128> = bitify::column_weights(&proved.opening, &prime)?;

    // Step 5.0: exact integer lift, grinded fresh-prime draw, re-projection.
    let step5_0_scope = tracing::info_span!("step5_0:reduce_prove").entered();
    let mu_prime = step50_integer_lift(hint.rows(), &row_weights, &col_weights);
    if !step50_accepts_lift(
        &mu_prime,
        proved.opening.claimed,
        prime.modulus_u128(),
        p.cells(),
    ) {
        return Err(ProtocolError::InvalidIntegerLift);
    }
    bind_claim_frame(
        transcript,
        spec,
        ClaimFrame {
            field: prime,
            binding: &binding,
            matrices_digest: &proved.matrices_digest,
            terminal_claim: &proved.terminal_claim,
            opening: &proved.opening,
            row_weights: &row_weights,
            col_weights: &col_weights,
            mu_prime: Some(&mu_prime),
        },
    )?;
    let nonce = grind_and_absorb_in_domain(
        transcript,
        domains.reduction_grinding,
        0,
        reduction.grinding_bits,
    )?;
    let reduced = sample_mod_q(
        transcript,
        domains.reduction_prime,
        reduction.min,
        reduction.max,
    )?;
    let (row_weights_reduced, _, _) = step50_reduce(
        &row_weights,
        &col_weights,
        &mu_prime,
        reduced.modulus_u128(),
    );
    drop(step5_0_scope);

    // Steps 5.1–5.3: the BitZ opening at the reduced prime.
    let bitz = {
        let _step5 = tracing::info_span!("step5:open_prove").entered();
        let _scope = (scopes.bitz_prove)().entered();
        prove_mle_eval_mod_q_ligerito_virtual_runtime(
            transcript,
            hint,
            hint.rows(),
            &p,
            &p,
            map,
            &row_weights_reduced,
            reduced.modulus_u128(),
            reduced.modulus_bits(),
            bitz_generator(),
            spec.opener_grinding_bits(security),
            ood,
            pc,
        )
        .map_err(ProtocolError::Bitz)?
    };

    Ok(Proof {
        prefix: proved.messages,
        reduction: Some(ReductionProof { mu_prime, nonce }),
        bitz,
    })
}

/// Verifies a Strategy-2 proof, re-deriving both primes from the bound
/// transcript.
pub fn verify_reduced<T: Transcript + Send, S: RelationSpec>(
    transcript: &mut T,
    prepared: &PreparedRelation<S>,
    commitment: &Commitment,
    proof: &Proof<IntEvalRsLigVirtProof>,
) -> Result<(), ProtocolError> {
    let prefix = &prepared.prefix;
    let opener = &prepared.opener;
    let spec = &prefix.spec;
    let p = prepared.params();
    let vc = opener.verifier()?;
    let binding_config = opener.binding_config()?;
    validate_commitment(&p, commitment, &binding_config)?;
    let security = &prefix.security;
    let reduction = security
        .reduction
        .ok_or(ProtocolError::UnsupportedProfile)?;
    let domains = spec.domains();
    let scopes = &domains.scopes;
    let map = spec.map().ok_or(ProtocolError::UnsupportedDischarge)?;
    let lift = proof
        .reduction
        .as_ref()
        .ok_or(ProtocolError::InvalidIntegerLift)?;
    check_proof_kernel(spec.kernel(), &proof.prefix.spartan)?;

    let (binding, ood) = bind_verifier_statement(
        transcript,
        prefix,
        opener,
        commitment,
        proof.bitz.ood.as_ref(),
    )?;
    let verified = verify_piop(transcript, prefix, &binding, &proof.prefix)?;
    let prime = &verified.prime;
    let row_weights = bitify::dense_row_weights(&verified.opening, &verified.table, &prime)?;
    let col_weights: Vec<u128> = bitify::column_weights(&verified.opening, &prime)?;

    // Step 5.0: the claimed integer lift must land in the derived mod-Q
    // class and inside the d * Q^2 magnitude bound.
    let step5_0_scope = tracing::info_span!("step5_0:reduce_verify").entered();
    if !step50_accepts_lift(
        &lift.mu_prime,
        verified.opening.claimed,
        prime.modulus_u128(),
        p.cells(),
    ) {
        return Err(ProtocolError::InvalidIntegerLift);
    }
    bind_claim_frame(
        transcript,
        spec,
        ClaimFrame {
            field: prime,
            binding: &binding,
            matrices_digest: &verified.matrices_digest,
            terminal_claim: &verified.terminal_claim,
            opening: &verified.opening,
            row_weights: &row_weights,
            col_weights: &col_weights,
            mu_prime: Some(&lift.mu_prime),
        },
    )?;
    verify_and_absorb_in_domain(
        transcript,
        domains.reduction_grinding,
        0,
        reduction.grinding_bits,
        lift.nonce,
    )?;
    let reduced = sample_mod_q(
        transcript,
        domains.reduction_prime,
        reduction.min,
        reduction.max,
    )?;
    let (row_weights_reduced, col_weights_reduced, claimed_reduced) = step50_reduce(
        &row_weights,
        &col_weights,
        &lift.mu_prime,
        reduced.modulus_u128(),
    );
    drop(step5_0_scope);

    let _step5 = tracing::info_span!("step5:open_verify").entered();
    let _scope = (scopes.bitz_verify)().entered();
    verify_mle_eval_mod_q_ligerito_virtual_runtime(
        transcript,
        commitment,
        &proof.bitz,
        &p,
        &p,
        map,
        &row_weights_reduced,
        &col_weights_reduced,
        bitz_generator(),
        claimed_reduced,
        reduced.modulus_u128(),
        reduced.modulus_bits(),
        spec.opener_grinding_bits(security),
        ood,
        vc,
    )
    .map_err(ProtocolError::Bitz)
}

/// The prover's output of the protocol prefix: the transcript messages and
/// the bitified claim the discharge consumes.
pub struct ProvedPrefix {
    pub messages: SpartanPrefixProof,
    pub terminal_claim: ScaledMleEvaluationClaim<SpartanBitzField>,
    pub matrices_digest: [u8; 32],
    pub opening: BitifiedClaim,
    pub bridge_digest: [u8; 32],
    pub prime: field::FpCtx<2>,
    pub table: BlockTable,
}

/// The verifier's output of the protocol prefix.
pub struct VerifiedPrefix {
    pub terminal_claim: ScaledMleEvaluationClaim<SpartanBitzField>,
    pub matrices_digest: [u8; 32],
    pub opening: BitifiedClaim,
    pub bridge_digest: [u8; 32],
    pub prime: field::FpCtx<2>,
    pub table: BlockTable,
}

/// Steps 2–4 of the protocol after the statement has been bound: the initial
/// grinding boundary, the Step-2 prime draw and relation projection, the
/// Spartan PIOP under per-draw grinding, bitification and the terminal
/// boundary.
pub fn prove_piop<T: Transcript, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    witness: &S::Witness,
    binding: &[u8; 32],
) -> Result<ProvedPrefix, ProtocolError> {
    let spec = &prefix.spec;
    spec.check_witness(witness)?;
    let security = &prefix.security;
    let domains = spec.domains();
    let scopes = &domains.scopes;
    let schedule = spec.schedule();

    // Step 2: pre-draw grinding, prime draw, and relation projection into
    // the runtime field.
    let step2_scope = tracing::info_span!("step2:project_prove").entered();
    let initial_nonce = grind_boundary(
        transcript,
        domains.initial_grinding,
        security.initial_grinding_bits,
    )?;
    let prime = spec.runtime_prime(transcript, security)?;
    let matrices = {
        let _scope = (scopes.relation_projection_prove)().entered();
        prefix.matrices.project(spec, &prime)?
    };
    let piop_witness = {
        let _scope = (scopes.witness_projection_prove)().entered();
        spec.piop_witness(witness, &prime)?
    };
    drop(step2_scope);

    // Step 3: the Spartan PIOP over F_q, every drawn challenge preceded by
    // one PIOP grinding boundary at the profile's difficulty (a transparent
    // pass-through at λ = 100).
    let (spartan, terminal_claim, piop_nonces) = {
        let _step3 = tracing::info_span!("step3:piop_prove").entered();
        let _scope = (scopes.spartan_prove)().entered();
        if schedule.piop_grinding {
            let mut grinder = ProverGrindingTranscript::<_>::new_in_domain(
                transcript,
                piop_wrap_bits(security),
                domains.piop_grinding,
            );
            let (spartan, terminal_claim) = run_kernel(
                &mut grinder,
                spec.kernel(),
                &matrices,
                binding,
                piop_witness,
            )?;
            (spartan, terminal_claim, grinder.finish())
        } else {
            let (spartan, terminal_claim) =
                run_kernel(transcript, spec.kernel(), &matrices, binding, piop_witness)?;
            (spartan, terminal_claim, Vec::new())
        }
    };

    // Step 4: bitification at the runtime prime, plus the terminal
    // boundary protecting the opening challenges.
    let step4_scope = tracing::info_span!("step4:bitify_prove").entered();
    let table = spec.block_table();
    let (opening, bridge_digest) = {
        let _scope = (scopes.bitify_prover)().entered();
        bitify_and_bind(
            spec,
            &matrices,
            binding,
            &terminal_claim,
            &table,
            schedule.scale_side,
            &prime,
        )?
    };
    let terminal_nonce = grind_boundary(
        transcript,
        domains.terminal_grinding,
        security.terminal_grinding_bits,
    )?;
    drop(step4_scope);

    Ok(ProvedPrefix {
        messages: SpartanPrefixProof {
            initial_nonce,
            piop_nonces,
            spartan,
            terminal_nonce,
        },
        terminal_claim,
        matrices_digest: *matrices.digest(),
        opening,
        bridge_digest,
        prime,
        table,
    })
}

/// Verifier twin of [`prove_piop`].
pub fn verify_piop<T: Transcript, S: RelationSpec>(
    transcript: &mut T,
    prefix: &PreparedRelationPrefix<S>,
    binding: &[u8; 32],
    messages: &SpartanPrefixProof,
) -> Result<VerifiedPrefix, ProtocolError> {
    let spec = &prefix.spec;
    let security = &prefix.security;
    let domains = spec.domains();
    let scopes = &domains.scopes;
    let schedule = spec.schedule();
    check_proof_kernel(spec.kernel(), &messages.spartan)?;

    let step2_scope = tracing::info_span!("step2:project_verify").entered();
    check_boundary(
        transcript,
        domains.initial_grinding,
        security.initial_grinding_bits,
        messages.initial_nonce,
    )?;
    let prime = spec.runtime_prime(transcript, security)?;
    let matrices = {
        let _scope = (scopes.relation_projection_verify)().entered();
        prefix.matrices.project(spec, &prime)?
    };
    drop(step2_scope);

    let terminal_claim = {
        let _step3 = tracing::info_span!("step3:piop_verify").entered();
        let _scope = (scopes.spartan_verify)().entered();
        if schedule.piop_grinding {
            let mut grinder = VerifierGrindingTranscript::<_>::new_in_domain(
                transcript,
                piop_wrap_bits(security),
                &messages.piop_nonces,
                domains.piop_grinding,
            );
            let terminal_claim =
                verify_kernel(&mut grinder, &matrices, binding, &messages.spartan)?;
            grinder.finish()?;
            terminal_claim
        } else {
            if !messages.piop_nonces.is_empty() {
                return Err(GrindingError::InvalidNonce {
                    nonce: messages.piop_nonces[0],
                    bits: 0,
                }
                .into());
            }
            verify_kernel(transcript, &matrices, binding, &messages.spartan)?
        }
    };

    let step4_scope = tracing::info_span!("step4:bitify_verify").entered();
    let table = spec.block_table();
    let (opening, bridge_digest) = {
        let _scope = (scopes.bitify_verifier)().entered();
        bitify_and_bind(
            spec,
            &matrices,
            binding,
            &terminal_claim,
            &table,
            schedule.scale_side,
            &prime,
        )?
    };
    check_boundary(
        transcript,
        domains.terminal_grinding,
        security.terminal_grinding_bits,
        messages.terminal_nonce,
    )?;
    drop(step4_scope);

    Ok(VerifiedPrefix {
        terminal_claim,
        matrices_digest: *matrices.digest(),
        opening,
        bridge_digest,
        prime,
        table,
    })
}

/// Runs the relation's Spartan kernel on the witness tables.
fn run_kernel<C, T: Transcript>(
    transcript: &mut T,
    kernel: Kernel,
    matrices: &PreparedConstraintMatrices<SpartanBitzField, C>,
    binding: &[u8; 32],
    witness: PiopWitness<'_>,
) -> Result<(SpartanProof, ScaledMleEvaluationClaim<SpartanBitzField>), ProtocolError>
where
    C: SpartanMatrixCoefficient<SpartanBitzField> + RawMontyCoefficient,
{
    let (spartan, claim) = match (kernel, witness) {
        (
            Kernel::UnivariateSkip { skip_vars },
            PiopWitness::Native {
                az,
                bz,
                cz,
                assignment,
                constant_prefix,
            },
        ) => {
            let products = NativeProducts {
                az: &az,
                bz: &bz,
                cz: &cz,
            };
            let (proof, claim) = prove_spartan_piop_native_u64_with_univariate_skip_borrowed(
                transcript,
                matrices,
                binding,
                products,
                assignment,
                skip_vars,
                constant_prefix,
            )?;
            (SpartanProof::UnivariateSkip(proof), claim)
        }
        (Kernel::UnivariateSkip { skip_vars }, PiopWitness::Mul32(witness)) => {
            let (proof, claim) = super::piop::prove_spartan_piop_raw_native_u64_with_skip_core(
                transcript,
                matrices,
                binding,
                witness,
                witness.inner_witness(),
                skip_vars,
            )?;
            (SpartanProof::UnivariateSkip(proof), claim)
        }
        (Kernel::UnivariateSkip { .. }, _) => return Err(ProtocolError::UnsupportedKernel),
        (
            Kernel::Plain,
            PiopWitness::Native {
                az,
                bz,
                cz,
                assignment,
                constant_prefix,
            },
        ) => {
            let products = NativeProducts {
                az: &az,
                bz: &bz,
                cz: &cz,
            };
            let (proof, claim) = prove_spartan_piop_native_u64_borrowed(
                transcript,
                matrices,
                binding,
                products,
                assignment,
                constant_prefix,
            )?;
            (SpartanProof::Plain(proof), claim)
        }
        (Kernel::Plain, PiopWitness::Mul32(witness)) => {
            let (proof, claim) = prove_spartan_piop_raw_products_raw_witness(
                transcript,
                matrices,
                binding,
                witness,
                witness.inner_witness(),
            )?;
            (SpartanProof::Plain(proof), claim)
        }
        (Kernel::Plain, PiopWitness::Mul64(witness)) => {
            let (proof, claim) = prove_spartan_piop_raw_products_raw_witness(
                transcript,
                matrices,
                binding,
                witness.native_products(),
                witness.inner_witness(),
            )?;
            (SpartanProof::Plain(proof), claim)
        }
        (Kernel::Plain, PiopWitness::Mul128(witness)) => {
            let (proof, claim) = prove_spartan_piop_raw_products_raw_witness(
                transcript,
                matrices,
                binding,
                witness.native_products(),
                witness.inner_witness(),
            )?;
            (SpartanProof::Plain(proof), claim)
        }
        (Kernel::Plain, PiopWitness::CmAnd(witness)) => {
            let (proof, claim) = prove_spartan_piop_raw_products_native_assignment(
                transcript,
                matrices,
                binding,
                witness,
                witness.assignment(),
                None,
            )?;
            (SpartanProof::Plain(proof), claim)
        }
        (Kernel::Plain, PiopWitness::FieldAssignment { assignment }) => {
            let products =
                super::matrix::products_from_montgomery_assignment(matrices, &assignment)?;
            let (proof, claim) = prove_spartan_piop_raw_products_raw_witness(
                transcript,
                matrices,
                binding,
                products,
                RawWitness::Field(assignment),
            )?;
            (SpartanProof::Plain(proof), claim)
        }
        (Kernel::Plain, PiopWitness::IntegerProducts { products, witness }) => {
            let (proof, claim) = prove_spartan_piop_raw_products_raw_witness(
                transcript, matrices, binding, products, witness,
            )?;
            (SpartanProof::Plain(proof), claim)
        }
        (
            Kernel::Plain,
            PiopWitness::Field {
                products,
                assignment,
            },
        ) => {
            let (proof, claim) =
                prove_spartan_piop_field(transcript, matrices, binding, products, assignment)?;
            (SpartanProof::Plain(proof), claim)
        }
    };
    Ok((spartan, claim))
}

/// Verifies the Spartan proof of either kernel shape.
fn verify_kernel<C, T: Transcript>(
    transcript: &mut T,
    matrices: &PreparedConstraintMatrices<SpartanBitzField, C>,
    binding: &[u8; 32],
    spartan: &SpartanProof,
) -> Result<ScaledMleEvaluationClaim<SpartanBitzField>, ProtocolError>
where
    C: SpartanMatrixCoefficient<SpartanBitzField>,
{
    Ok(match spartan {
        SpartanProof::Plain(spartan) => {
            verify_spartan_proof(transcript, matrices, binding, spartan)?
        }
        SpartanProof::UnivariateSkip(spartan) => {
            verify_spartan_univariate_skip_proof(transcript, matrices, binding, spartan)?
        }
    })
}

/// The proof's Spartan shape must be the one the relation's kernel produces
/// (checked before any transcript operation).
fn check_proof_kernel(kernel: Kernel, spartan: &SpartanProof) -> Result<(), ProtocolError> {
    let expected = match kernel {
        Kernel::Plain => 0,
        Kernel::UnivariateSkip { skip_vars } => skip_vars as u8,
    };
    let actual = match spartan {
        SpartanProof::Plain(_) => 0,
        SpartanProof::UnivariateSkip(proof) => proof.outer.skip.skip_vars,
    };
    if expected != actual {
        return Err(ProtocolError::UnexpectedUnivariateSkipVariables { expected, actual });
    }
    Ok(())
}

/// Bitifies the terminal claim and computes the bridge digest.
#[allow(clippy::too_many_arguments)]
fn bitify_and_bind<S: RelationSpec>(
    spec: &S,
    matrices: &PreparedConstraintMatrices<SpartanBitzField, S::Coefficient>,
    binding: &[u8; 32],
    terminal_claim: &ScaledMleEvaluationClaim<SpartanBitzField>,
    table: &BlockTable,
    scale_side: ScaleSide,
    prime: &field::FpCtx<2>,
) -> Result<(BitifiedClaim, [u8; 32]), ProtocolError> {
    let opening = bitify::bitify(
        terminal_claim,
        spec.opening_layout(),
        spec.gate_vars(),
        table,
        scale_side,
        &prime,
    )?;
    let digest = bitify::bridge_digest(
        spec.domains().bitified_claim,
        binding,
        matrices.field_modulus_encoding(),
        matrices.digest(),
        prime.modulus_u128(),
        |hasher| spec.hash_bridge_constants(hasher),
        terminal_claim,
        &opening,
        matrices.config(),
    )?;
    Ok((opening, digest))
}

/// Uniform per-draw PIOP grinding difficulty: the maximum requirement of
/// any single drawn challenge.
pub fn piop_wrap_bits(security: &IopSecurityParams) -> u32 {
    security
        .initial_grinding_bits
        .max(security.piop_round_grinding_bits)
}

/// One grinding boundary at round index 0 (no bytes at difficulty 0).
pub fn grind_boundary<T: Transcript>(
    transcript: &mut T,
    domain: &[u8],
    bits: u32,
) -> Result<u64, GrindingError> {
    if bits == 0 {
        return Ok(0);
    }
    grind_and_absorb_in_domain(transcript, domain, 0, bits)
}

/// Verifier twin of [`grind_boundary`].
pub fn check_boundary<T: Transcript>(
    transcript: &mut T,
    domain: &[u8],
    bits: u32,
    nonce: u64,
) -> Result<(), GrindingError> {
    if bits == 0 {
        if nonce != 0 {
            return Err(GrindingError::InvalidNonce { nonce, bits });
        }
        return Ok(());
    }
    verify_and_absorb_in_domain(transcript, domain, 0, bits, nonce)
}

/// Samples a prime from `[min, max]` under `domain` (the prime-domain,
/// prime-min, prime-max frames, the transcript-driven draw, the prime-q
/// frame) and builds its runtime field.
pub fn sample_mod_q(
    transcript: &mut impl Transcript,
    domain: &[u8],
    min: u128,
    max: u128,
) -> Result<field::FpCtx<2>, ProtocolError> {
    absorb_spartan_message(transcript, b"prime-domain", domain);
    absorb_spartan_message(transcript, b"prime-min", &min.to_le_bytes());
    absorb_spartan_message(transcript, b"prime-max", &max.to_le_bytes());
    let field = crate::ext_proj::sample_prime_context(transcript, min, max, 128)?;
    let q = u128::from(*field.modulus());
    absorb_spartan_message(transcript, b"prime-q", &q.to_le_bytes());
    if field.modulus_bits() < super::SPARTAN_MIN_MODULUS_BITS as usize {
        return Err(ProtocolError::UnsupportedFieldModulus);
    }
    Ok(field)
}

/// Samples a full-width prime under the same bounded policy as smaller
/// intervals. Prover and verifier replay the identical advancing stream.
pub fn sample_full_width_prime(
    transcript: &mut impl Transcript,
    domain: &[u8],
    min: u128,
    max: u128,
) -> Result<field::FpCtx<2>, ProtocolError> {
    absorb_spartan_message(transcript, b"prime-domain", domain);
    let field = crate::ext_proj::sample_prime_context(transcript, min, max, 128)?;
    let q = u128::from(*field.modulus());
    absorb_spartan_message(transcript, b"prime-q", &q.to_le_bytes());
    if field.modulus_bits() < super::SPARTAN_MIN_MODULUS_BITS as usize {
        return Err(ProtocolError::UnsupportedFieldModulus);
    }
    Ok(field)
}

/// Validates a public protocol modulus and prepares its shared arithmetic once.
pub fn runtime_field(q: u128) -> Result<field::FpCtx<2>, ProtocolError> {
    let config = SpartanBitzField::make_cfg(&Uint::from(q))
        .map_err(|_| ProtocolError::UnsupportedFieldModulus)?;
    SpartanBitzField::validate_config(&config)
        .map_err(|_| ProtocolError::UnsupportedFieldModulus)?;
    Ok(config)
}

/// The committed bit rows must have exactly the layout's shape.
pub fn validate_bit_rows(p: &IntegerMatrixLayout, rows: &[Vec<u64>]) -> Result<(), ProtocolError> {
    let row_count = checked_pow2(p.row_vars)?;
    let col_count = checked_pow2(p.col_vars)?;
    let row_bits = row_count
        .checked_mul(p.word_bits)
        .ok_or(ProtocolError::InvalidBitRows)?;
    if row_bits % u64::BITS as usize != 0 || rows.len() != col_count {
        return Err(ProtocolError::InvalidBitRows);
    }
    let words_per_col = row_bits / u64::BITS as usize;
    if rows.iter().any(|row| row.len() != words_per_col) {
        return Err(ProtocolError::InvalidBitRows);
    }
    Ok(())
}

/// The prover and verifier Ligerito configurations must agree and fit the
/// layout.
pub fn validate_config_pair(
    p: &IntegerMatrixLayout,
    pc: &LigProverConfig,
    vc: &LigVerifierConfig,
) -> Result<(), ProtocolError> {
    let m_p = packed_variables(p)?;
    let Some(&log_inv_rate) = pc.log_inv_rates.first() else {
        return Err(ProtocolError::InvalidBitzParameters);
    };
    if log_inv_rate == 0
        || pc.initial_k >= m_p
        || pc.initial_k != vc.initial_k
        || pc.log_inv_rates != vc.log_inv_rates
        || pc.recursive_steps != vc.recursive_steps
        || pc.initial_log_msg_cols != vc.initial_log_msg_cols
        || pc.initial_log_num_interleaved != vc.initial_log_num_interleaved
        || pc.recursive_log_msg_cols != vc.recursive_log_msg_cols
        || pc.recursive_ks != vc.recursive_ks
        || pc.queries != vc.queries
        || pc.grinding_bits != vc.grinding_bits
        || pc.fold_grinding_bits != vc.fold_grinding_bits
        || pc.ood_samples != vc.ood_samples
        || pc.merkle_hash != vc.merkle_hash
    {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    Ok(())
}

/// The commitment must have been produced under the prepared configuration.
pub fn validate_commitment(
    p: &IntegerMatrixLayout,
    commitment: &Commitment,
    pc: &LigProverConfig,
) -> Result<(), ProtocolError> {
    let m_p = packed_variables(p)?;
    let Some(&log_inv_rate) = pc.log_inv_rates.first() else {
        return Err(ProtocolError::InvalidBitzParameters);
    };
    let params = &commitment.params;
    if params.m != m_p + LOG_PACKING
        || params.log_inv_rate != log_inv_rate
        || params.log_batch_size != pc.initial_k
        || params.profile != LigeritoProfile::default()
        || params.merkle_hash != pc.merkle_hash
    {
        return Err(ProtocolError::CommitmentConfigMismatch);
    }
    Ok(())
}

/// The packed-word variable count of the committed tensor, checked against
/// the shared packing rule.
pub fn packed_variables(p: &IntegerMatrixLayout) -> Result<usize, ProtocolError> {
    if !p.word_bits.is_power_of_two() || p.word_bits > u128::BITS as usize {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    let row_bit_vars = p
        .row_vars
        .checked_add(p.word_bits.trailing_zeros() as usize)
        .ok_or(ProtocolError::InvalidBitzParameters)?;
    let expected = row_bit_vars
        .checked_sub(LOG_PACKING)
        .and_then(|folded| folded.checked_add(p.col_vars))
        .ok_or(ProtocolError::InvalidBitzParameters)?;
    if packed_vars(p) != expected {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    Ok(expected)
}

pub fn checked_pow2(exponent: usize) -> Result<usize, ProtocolError> {
    let exponent = u32::try_from(exponent).map_err(|_| ProtocolError::InvalidBitzParameters)?;
    1_usize
        .checked_shl(exponent)
        .ok_or(ProtocolError::InvalidBitzParameters)
}

/// The GF(2^128) generator every opening runs against.
pub fn bitz_generator() -> Gf128 {
    static GENERATOR: OnceLock<Gf128> = OnceLock::new();
    *GENERATOR.get_or_init(crate::pcs::smallest_generator)
}

/// The PCS-only terminal-opening benchmark path: the BitZ opening of an
/// externally supplied terminal claim at the fixed comparison field, with
/// relation projection, profile derivation and statement binding excluded
/// from every timer.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod terminal {
    use super::*;
    use crate::pcs::{FQ_BITS, FQ_MOD};

    /// How the terminal-opening statement binds the relation and commitment.
    #[derive(Clone, Copy)]
    pub enum Binding<S> {
        /// The relation's paper assignment binding (profile parameters and
        /// Ligerito configuration included).
        Paper,
        /// A relation-provided digest of the layout and commitment alone.
        Custom(fn(&S, &Commitment) -> Result<[u8; 32], ProtocolError>),
    }

    /// What the terminal-opening statement frame carries under its tag.
    #[derive(Clone, Copy)]
    pub enum StatementPayload {
        /// The bridge digest of the bitified claim (the u32 path).
        BridgeDigest,
        /// The assignment binding, followed by a `relation_tag` frame with
        /// the relation digest (the BabyBear path).
        AssignmentBinding { relation_tag: &'static [u8] },
    }

    /// Setup-once context retaining only the public digests and configs.
    pub struct PreparedTerminalOpening<S: RelationSpec> {
        spec: S,
        params: IntegerMatrixLayout,
        security: IopSecurityParams,
        ligerito: ResolvedLigerito,
        binding: Binding<S>,
        assignment_binding: [u8; 32],
        relation_modulus_encoding: Box<[u8]>,
        relation_digest: [u8; 32],
        statement_tag: &'static [u8],
        payload: StatementPayload,
    }

    impl<S: RelationSpec> PreparedTerminalOpening<S> {
        pub const fn ligerito_configuration(&self) -> &ResolvedLigerito {
            &self.ligerito
        }

        pub const fn security(&self) -> &IopSecurityParams {
            &self.security
        }

        pub const fn params(&self) -> &IntegerMatrixLayout {
            &self.params
        }

        fn statement_binding(&self, commitment: &Commitment) -> Result<[u8; 32], ProtocolError> {
            match self.binding {
                Binding::Paper => {
                    self.spec
                        .assignment_binding(commitment, &self.security, self.ligerito.prover())
                }
                Binding::Custom(binding) => binding(&self.spec, commitment),
            }
        }

        fn validate_commitment(&self, commitment: &Commitment) -> Result<(), ProtocolError> {
            validate_commitment(&self.params, commitment, self.ligerito.prover())?;
            if self.statement_binding(commitment)? != self.assignment_binding {
                return Err(ProtocolError::PreparedOpeningCommitmentMismatch);
            }
            Ok(())
        }

        fn bind_claim(
            &self,
            transcript: &mut impl Transcript,
            terminal_claim: &ScaledMleEvaluationClaim<SpartanBitzField>,
        ) -> Result<(BitifiedClaim, [u8; 32], BlockTable, field::FpCtx<2>), ProtocolError> {
            let prime = runtime_field(FQ_MOD)?;
            let table = self.spec.block_table();
            let opening = bitify::bitify(
                terminal_claim,
                self.params,
                self.spec.gate_vars(),
                &table,
                self.spec.schedule().scale_side,
                &prime,
            )?;
            let bridge_digest = bitify::bridge_digest(
                self.spec.domains().bitified_claim,
                &self.assignment_binding,
                &self.relation_modulus_encoding,
                &self.relation_digest,
                FQ_MOD,
                |hasher| self.spec.hash_bridge_constants(hasher),
                terminal_claim,
                &opening,
                &prime,
            )?;
            match self.payload {
                StatementPayload::BridgeDigest => {
                    absorb_spartan_message(transcript, self.statement_tag, &bridge_digest);
                }
                StatementPayload::AssignmentBinding { relation_tag } => {
                    absorb_spartan_message(
                        transcript,
                        self.statement_tag,
                        &self.assignment_binding,
                    );
                    absorb_spartan_message(transcript, relation_tag, &self.relation_digest);
                }
            }
            self.ligerito.bind(transcript);
            Ok((opening, bridge_digest, table, prime))
        }
    }

    /// Prepares the fixed-q, PCS-only terminal-opening context.
    pub fn prepare<S: RelationSpec + Clone>(
        prepared: &PreparedRelation<S>,
        commitment: &Commitment,
        statement_tag: &'static [u8],
        payload: StatementPayload,
        binding: Binding<S>,
    ) -> Result<PreparedTerminalOpening<S>, ProtocolError> {
        let prime = runtime_field(FQ_MOD)?;
        let matrices = prepared
            .prefix
            .matrices
            .project(prepared.layout(), &prime)?;
        let params = prepared.params();
        let ligerito = prepared.ligerito_configuration().clone();
        validate_commitment(&params, commitment, ligerito.prover())?;
        let mut terminal = PreparedTerminalOpening {
            spec: prepared.layout().clone(),
            params,
            security: prepared.security().clone(),
            ligerito,
            binding,
            assignment_binding: [0; 32],
            relation_modulus_encoding: matrices.field_modulus_encoding().into(),
            relation_digest: *matrices.digest(),
            statement_tag,
            payload,
        };
        terminal.assignment_binding = terminal.statement_binding(commitment)?;
        Ok(terminal)
    }

    /// Commits already-materialized bit rows with the exact configuration
    /// retained by the PCS-only context.
    pub fn commit<S: RelationSpec>(
        prepared: &PreparedTerminalOpening<S>,
        rows: Vec<Vec<u64>>,
    ) -> Result<FlockCommitHint, ProtocolError> {
        validate_bit_rows(&prepared.params, &rows)?;
        let hint = commit_rs_ligerito_rows(&prepared.params, rows, prepared.ligerito.prover());
        prepared.validate_commitment(&hint.commitment)?;
        Ok(hint)
    }

    /// Proves one already-derived terminal assignment-MLE claim, with all
    /// Spartan work deliberately outside the benchmark boundary.
    pub fn prove<T: Transcript + Send, S: RelationSpec>(
        transcript: &mut T,
        prepared: &PreparedTerminalOpening<S>,
        hint: &FlockCommitHint,
        terminal_claim: &ScaledMleEvaluationClaim<SpartanBitzField>,
    ) -> Result<IntEvalRsLigModQProof, ProtocolError> {
        prepared.validate_commitment(&hint.commitment)?;
        validate_bit_rows(&prepared.params, hint.rows())?;
        let (opening, bridge_digest, table, prime) =
            prepared.bind_claim(transcript, terminal_claim)?;
        let ood = bind_prover_ood(transcript, hint, prepared.security.ood);
        let chunks = bitify::prepare_chunks(&opening, &table, FQ_BITS, &prime)?;
        if chunks.len() != 1 {
            return Err(ProtocolError::MultiChunkRuntimeWeights);
        }
        prove_mle_eval_mod_q_ligerito_with_weight_chunks(
            transcript,
            prepared.spec.domains().opening,
            hint,
            &prepared.params,
            &chunks,
            &bridge_digest,
            FQ_BITS,
            bitz_generator(),
            prepared.spec.opener_grinding_bits(&prepared.security),
            ood,
            prepared.ligerito.prover(),
        )
        .map_err(ProtocolError::Bitz)
    }

    /// Verifies the PCS-only terminal opening from public data alone.
    pub fn verify<T: Transcript + Send, S: RelationSpec>(
        transcript: &mut T,
        prepared: &PreparedTerminalOpening<S>,
        commitment: &Commitment,
        terminal_claim: &ScaledMleEvaluationClaim<SpartanBitzField>,
        proof: &IntEvalRsLigModQProof,
    ) -> Result<(), ProtocolError> {
        prepared.validate_commitment(commitment)?;
        let (opening, bridge_digest, table, prime) =
            prepared.bind_claim(transcript, terminal_claim)?;
        let ood = bind_verifier_ood(
            transcript,
            packed_variables(&prepared.params)?,
            prepared.security.ood,
            proof.ood.as_ref(),
        )
        .map_err(ProtocolError::Bitz)?;
        let chunks = bitify::prepare_chunks(&opening, &table, FQ_BITS, &prime)?;
        if chunks.len() != 1 {
            return Err(ProtocolError::MultiChunkRuntimeWeights);
        }
        let col_weights = bitify::column_weights(&opening, &prime)?;
        verify_mle_eval_mod_q_ligerito_with_weight_chunks_runtime(
            transcript,
            prepared.spec.domains().opening,
            commitment,
            proof,
            &prepared.params,
            &chunks,
            &col_weights,
            &bridge_digest,
            bitz_generator(),
            opening.claimed,
            FQ_MOD,
            FQ_BITS,
            prepared.spec.opener_grinding_bits(&prepared.security),
            ood,
            prepared.ligerito.verifier(),
        )
        .map_err(ProtocolError::Bitz)
    }
}

impl From<super::sumcheck::SumcheckError> for ProtocolError {
    fn from(value: super::sumcheck::SumcheckError) -> Self {
        Self::Spartan(value.into())
    }
}
