//! The MultiSwap integer Mod-R1CS over BitZ (Limber's RSA-accumulator
//! benchmark), as a description of the shared protocol of
//! [`super::super::protocol`] under the paper's Strategy 2 instantiation
//! (large PIOP field, grinding concentrated at the Step 5.0 reduction draw):
//!
//! 1. Bind the complete public statement: the integer circuit digest, the
//!    block layout, and the BitZ commitment to the witness/quotient bits.
//! 2. Draw the 128-bit fingerprint prime `Q` (the commit-before-prime
//!    order is the Zaratan fingerprint; the full-width interval makes the
//!    draw `<= 2^-114` sound without grinding).
//! 3. Project the integer matrices, assignment, and products modulo `Q`
//!    and run the stock Spartan PIOP (cubic outer + batched quadratic
//!    inner) over `F_Q`; every round message carries error `<= 3/Q`.
//! 4. Translate the terminal scaled assignment-MLE claim through the
//!    public bitification adjoint into one mod-`Q` tensor functional over
//!    the committed bit tensor.
//! 5. Step 5.0 ([`super::reduce`]): the prover sends the exact integer
//!    lift `mu'` of that tensor claim; after checking `mu' = mu (mod Q)`
//!    and the `d * Q^2` bound, a 10-bit grind and a fresh 113-bit
//!    reduction prime `q'` re-project the claim below the exponent-fold
//!    no-wrap boundary, and the runtime-`q'` BitZ opening discharges it.
//!
//! The opening runs through the virtual-map entry points with the exact
//! identity map, which the library recognizes and routes to the direct
//! base opening.
//!
//! Soundness floors are documented in [`super::prime`]: every step is at
//! or below `2^-114` — the fingerprint draw `2^-114.0`, Spartan rounds
//! `2^-125.4`, the grinded reduction draw `2^-114.2` — matching the floors
//! Limber's own implementation accepts, at a total grinding cost of
//! `2^10` hashes.

use crate::ligerito_flock::IntEvalRsLigVirtProof;
use crate::piop::spartan::protocol::Proof;
use crate::piop::spartan::protocol::ProtocolError;

use crate::piop::spartan::SpartanField as _;
use blake3::Hasher;
use circuit::linear_map::CscMatrix;
use field::RingOps;
use flock_core::pcs::{
    commit::Commitment,
    ligerito::{ProverConfig as LigProverConfig, VerifierConfig as LigVerifierConfig},
};

use {
    crate::{
        f2map::cell_count,
        ligerito::packed_vars,
        ligerito_flock::{
            FlockCommitHint, FlockRsError, LigeritoStatementConfig, ModQOpeningKind,
            validated_udr_lig_configs_with,
        },
        pcs::IntegerMatrixLayout,
        transcript::traits::Transcript,
    },
    circuit::linear_map::binary::{
        PreparedVirtualMap, PreparedVirtualMapError, RepeatedVirtualMap,
    },
};

use super::super::{
    PreparedConstraintMatrices, SpartanBitzField,
    profile::{IopInstanceFacts, IopSecurityParams, IopSecurityProfile, Limber114},
    protocol::{
        self, BindingHasher, BlockTable, ClaimFrame, Domains, FieldConfig, Kernel, MatrixSource,
        PiopWitness, PreparedRelation, PreparedRelationPrefix, PrimeStrategy, RelationSpec,
        ScaleSide, Schedule, SlotRange,
    },
};
use super::{
    circuit::{MULTISWAP_VALUE_BITS, MultiswapCircuit, MultiswapCircuitError},
    prime::{
        FINGERPRINT_SAMPLING_DOMAIN, MultiswapPrimeError, MultiswapPrimeProfile,
        REDUCTION_SAMPLING_DOMAIN, multiswap_instance_facts,
    },
    relation::{
        MULTISWAP_QUOS_SLOT_START, MULTISWAP_SLOTS, MULTISWAP_W_SLOT_START, MultiswapAssignment,
        MultiswapIntegerRelation, MultiswapLayout, MultiswapLayoutError,
    },
};

const MULTISWAP_STATEMENT_DOMAIN: &[u8] = b"bitz/spartan-multiswap/statement/v2";
const MULTISWAP_BINDING_DOMAIN: &[u8] = b"bitz/spartan-multiswap/assignment-binding/v2";
const MULTISWAP_OPENING_CLAIM_DOMAIN: &[u8] = b"bitz/spartan-multiswap/opening-claim/v3";
/// Local identity block repeated across gates; any power-of-two factor of
/// the cell count works, and the slot count keeps the local map small.
const IDENTITY_LOCAL_ROWS: usize = MULTISWAP_SLOTS;

static MULTISWAP_DOMAINS: Domains = Domains {
    statement_tag: b"multiswap-statement",
    prime_sampling: FINGERPRINT_SAMPLING_DOMAIN,
    // No initial, per-draw or terminal boundary under Strategy 2.
    initial_grinding: b"",
    piop_grinding: b"",
    terminal_grinding: b"",
    bitified_claim: b"",
    opening: ModQOpeningKind::U32Mul,
    claim_tag: b"multiswap-opening-claim",
    reduction_grinding: b"bitz/spartan-multiswap/grinding/reduction/v2",
    reduction_prime: REDUCTION_SAMPLING_DOMAIN,
    scopes: crate::protocol_scopes!("multiswap"),
};

impl From<MultiswapCircuitError> for ProtocolError {
    fn from(error: MultiswapCircuitError) -> Self {
        Self::relation(error)
    }
}

impl From<MultiswapLayoutError> for ProtocolError {
    fn from(error: MultiswapLayoutError) -> Self {
        Self::relation(error)
    }
}

impl From<MultiswapPrimeError> for ProtocolError {
    fn from(error: MultiswapPrimeError) -> Self {
        Self::relation(error)
    }
}

impl From<PreparedVirtualMapError> for ProtocolError {
    fn from(error: PreparedVirtualMapError) -> Self {
        Self::relation(error)
    }
}

/// The MultiSwap relation as the shared protocol sees it: the integer
/// relation, the identity opening map and the statement digests.
pub struct MultiswapSpec {
    relation: MultiswapIntegerRelation,
    map: RepeatedVirtualMap,
    statement_digest: [u8; 32],
    comparison_statement_digest: [u8; 32],
    batch_count: usize,
}

impl MultiswapSpec {
    fn new(circuit: &MultiswapCircuit) -> Result<Self, ProtocolError> {
        let relation = MultiswapIntegerRelation::new(circuit)?;
        let params = relation.layout().bitz_params();
        let cells = cell_count(&params);
        if cells % IDENTITY_LOCAL_ROWS != 0 {
            return Err(ProtocolError::InvalidBitzParameters);
        }
        let identity_columns = (0..IDENTITY_LOCAL_ROWS)
            .map(|index| vec![(index, true)])
            .collect::<Vec<_>>();
        let local = PreparedVirtualMap::new(
            CscMatrix::try_from_columns(IDENTITY_LOCAL_ROWS, identity_columns)
                .expect("the identity block is a valid CSC matrix"),
        )?;
        let cells_per_block = cells / IDENTITY_LOCAL_ROWS;
        let map = RepeatedVirtualMap::new(local, cells_per_block)?;
        debug_assert!(circuit::linear_map::binary::VirtualMap::is_identity(&map));
        Ok(Self {
            relation,
            map,
            statement_digest: circuit.statement_digest(),
            comparison_statement_digest: circuit.comparison_statement_digest(),
            batch_count: circuit.batch_count(),
        })
    }

    pub const fn relation(&self) -> &MultiswapIntegerRelation {
        &self.relation
    }

    pub const fn layout(&self) -> &MultiswapLayout {
        self.relation.layout()
    }

    pub const fn map(&self) -> &RepeatedVirtualMap {
        &self.map
    }

    pub const fn statement_digest(&self) -> &[u8; 32] {
        &self.statement_digest
    }
}

impl RelationSpec for MultiswapSpec {
    type Coefficient = SpartanBitzField;
    type Witness = MultiswapAssignment;
    type Map = RepeatedVirtualMap;

    fn domains(&self) -> &'static Domains {
        &MULTISWAP_DOMAINS
    }

    fn schedule(&self) -> Schedule {
        Schedule {
            policy_bind: false,
            ood_round: false,
            piop_grinding: false,
            scale_side: ScaleSide::Rows,
            strategy: PrimeStrategy::TwoPrime,
        }
    }

    fn committed_layout(&self) -> IntegerMatrixLayout {
        self.layout().bitz_params()
    }

    fn gate_vars(&self) -> usize {
        self.layout().gate_vars()
    }

    fn instance_facts(&self) -> IopInstanceFacts {
        let params = self.committed_layout();
        multiswap_instance_facts(
            u32::try_from(params.row_vars).expect("row variables fit u32"),
            u32::try_from(params.word_bits).expect("word bits fit u32"),
            u32::try_from(self.layout().gate_vars() + 2).expect("assignment variables fit u32"),
        )
    }

    fn matrices(&self) -> Result<MatrixSource<SpartanBitzField>, ProtocolError> {
        Ok(MatrixSource::PerPrime)
    }

    /// The integer matrices modulo the fingerprint field (zero residues
    /// dropped identically on both sides).
    fn project_matrices(
        &self,
        config: &FieldConfig,
    ) -> Result<PreparedConstraintMatrices<SpartanBitzField, SpartanBitzField>, ProtocolError> {
        Ok(self.relation.project::<SpartanBitzField>(config)?)
    }

    fn validate_geometry(&self) -> Result<(), ProtocolError> {
        let params = self.committed_layout();
        if params.word_bits != 1 || cell_count(&params) % IDENTITY_LOCAL_ROWS != 0 {
            return Err(ProtocolError::InvalidBitzParameters);
        }
        Ok(())
    }

    /// Block order is 00=constant, 01=witness, 10=quotient, 11=the empty
    /// zero block.
    fn block_table(&self) -> BlockTable {
        BlockTable::new(
            2,
            vec![
                None,
                Some(SlotRange {
                    bit_slot_start: MULTISWAP_W_SLOT_START,
                    bit_count: MULTISWAP_VALUE_BITS,
                }),
                Some(SlotRange {
                    bit_slot_start: MULTISWAP_QUOS_SLOT_START,
                    bit_count: MULTISWAP_VALUE_BITS,
                }),
                None,
            ],
        )
        .expect("the MultiSwap block table is complete")
    }

    fn kernel(&self) -> Kernel {
        Kernel::Plain
    }

    fn check_witness(&self, assignment: &MultiswapAssignment) -> Result<(), ProtocolError> {
        if assignment.layout() != self.layout() {
            return Err(ProtocolError::RelationWitnessLayoutMismatch);
        }
        Ok(())
    }

    /// Digest binding the integer statement, layout, profile, and commitment.
    fn assignment_binding(
        &self,
        commitment: &Commitment,
        security: &IopSecurityParams,
        ligerito: &LigProverConfig,
    ) -> Result<[u8; 32], ProtocolError> {
        let layout = self.layout();
        let p = self.committed_layout();
        let reduction = security
            .reduction
            .ok_or(ProtocolError::UnsupportedProfile)?;
        let mut hasher = BindingHasher::new();
        // Keep the historical unbatched 114-bit transcript pins. New profiles
        // and batches bind the full comparison contract and opening
        // configuration.
        if security.lambda != 114 || self.batch_count != 1 {
            hasher
                .bytes(b"bitz/spartan-multiswap/configuration/v3")
                .bytes(&security.lambda.to_le_bytes())
                .bytes(security.profile_name.as_bytes())
                .bytes(&config_digest(ligerito))
                .bytes(&self.comparison_statement_digest);
        }
        hasher
            .bytes(MULTISWAP_BINDING_DOMAIN)
            .bytes(MULTISWAP_STATEMENT_DOMAIN)
            .bytes(&self.statement_digest)
            .bytes(&commitment.root);
        hasher.usizes(&[
            commitment.params.m,
            commitment.params.log_inv_rate,
            commitment.params.log_batch_size,
            layout.capacity(),
            layout.gate_vars(),
            layout.column_vars(),
            layout.high_gate_vars(),
            layout.assignment_len(),
            p.row_vars,
            p.col_vars,
            p.word_bits,
            MULTISWAP_VALUE_BITS,
            reduction.grinding_bits as usize,
        ])?;
        hasher
            .u128_le(security.projection_min)
            .u128_le(security.projection_max)
            .u128_le(reduction.min)
            .u128_le(reduction.max)
            .bytes(&circuit::linear_map::binary::VirtualMap::digest(&self.map));
        Ok(hasher.finalize())
    }

    /// The full-width fingerprint prime `Q ∈ [2^127, 2^128)`.
    fn runtime_prime<T: Transcript>(
        &self,
        transcript: &mut T,
        security: &IopSecurityParams,
    ) -> Result<field::FpCtx<2>, ProtocolError> {
        protocol::sample_full_width_prime(
            transcript,
            FINGERPRINT_SAMPLING_DOMAIN,
            security.projection_min,
            security.projection_max,
        )
    }

    fn piop_witness<'w>(
        &self,
        assignment: &'w MultiswapAssignment,
        config: &FieldConfig,
    ) -> Result<PiopWitness<'w>, ProtocolError> {
        Ok(PiopWitness::FieldAssignment {
            assignment: assignment.projected_assignment(config),
        })
    }

    fn map(&self) -> Option<&RepeatedVirtualMap> {
        Some(&self.map)
    }

    /// Rebinds the derived terminal claim, its bitified image, and the Step
    /// 5.0 integer lift before the grinded reduction draw.
    fn claim_digest(&self, frame: ClaimFrame<'_>) -> Result<[u8; 32], ProtocolError> {
        let mu_prime = frame.mu_prime.ok_or(ProtocolError::InvalidIntegerLift)?;
        let mut hasher = BindingHasher::new();
        hasher
            .bytes(MULTISWAP_OPENING_CLAIM_DOMAIN)
            .bytes(frame.binding);
        hasher.usize(frame.terminal_claim.point().len())?;
        for coordinate in frame.terminal_claim.point() {
            hasher.element(coordinate, frame.field);
        }
        hasher.element(frame.terminal_claim.scale(), frame.field);
        hasher.element(frame.terminal_claim.value(), frame.field);
        hasher.u128_le(frame.opening.claimed);
        for weight in frame.col_weights {
            hasher.u128_le(*weight);
        }
        hasher.prefixed(&super::reduce::encode_integer_lift(mu_prime))?;
        Ok(hasher.finalize())
    }
}

/// Setup-once, prime-independent bundle: the integer relation, the identity
/// opening map, the BitZ shape, the instantiated security profile, and the
/// statement digest.
pub struct PreparedMultiswapRelation {
    inner: PreparedRelation<MultiswapSpec>,
    params: IntegerMatrixLayout,
    profile: MultiswapPrimeProfile,
    opening_configs: (LigProverConfig, LigVerifierConfig),
    opening_config_digest: [u8; 32],
}

impl PreparedMultiswapRelation {
    /// Prepares the relation, layout, and identity map from a built circuit
    /// at the pinned [`Limber114`] comparison profile.
    pub fn new(circuit: &MultiswapCircuit) -> Result<Self, ProtocolError> {
        Self::new_with_profile::<Limber114>(circuit)
    }

    /// Prepares the relation under an explicit security profile. The
    /// profile must be a two-prime Strategy-2 configuration (the MultiSwap
    /// defect bound rules out a single derived-width fingerprint).
    pub fn new_with_profile<P: IopSecurityProfile>(
        circuit: &MultiswapCircuit,
    ) -> Result<Self, ProtocolError> {
        let spec = MultiswapSpec::new(circuit)?;
        let params = spec.committed_layout();
        let prefix = PreparedRelationPrefix::new::<P>(spec)?;
        let profile = MultiswapPrimeProfile::from_security(prefix.security())?;
        let opening_configs = validated_udr_lig_configs_with(
            packed_vars(&params),
            3,
            4,
            prefix.security().ligerito_target_bits,
        )
        .map_err(ProtocolError::LigeritoConfig)?;
        let opening_config_digest = config_digest(&opening_configs.0);
        let inner = PreparedRelation::with_opener_configs(
            prefix,
            Some(opening_configs.0.clone()),
            Some(opening_configs.1.clone()),
        )?;
        Ok(Self {
            inner,
            params,
            profile,
            opening_configs,
            opening_config_digest,
        })
    }

    /// Opener settings derived and validated under this relation's profile.
    pub fn ligerito_configs(&self) -> (LigProverConfig, LigVerifierConfig) {
        self.opening_configs.clone()
    }

    pub const fn opening_config_digest(&self) -> &[u8; 32] {
        &self.opening_config_digest
    }

    fn validate_config(&self, config: &impl LigeritoStatementConfig) -> Result<(), ProtocolError> {
        if config_digest(config) != self.opening_config_digest {
            return Err(ProtocolError::Bitz(FlockRsError::CommitmentConfig));
        }
        Ok(())
    }

    /// The instantiated security parameters and their accounting.
    pub const fn security(&self) -> &IopSecurityParams {
        self.inner.security()
    }

    /// The prime-independent integer relation.
    pub const fn relation(&self) -> &MultiswapIntegerRelation {
        self.inner.layout().relation()
    }

    /// Shared block/bit layout.
    pub const fn layout(&self) -> &MultiswapLayout {
        self.inner.layout().layout()
    }

    /// BitZ shape of the committed bit tensor.
    pub const fn params(&self) -> &IntegerMatrixLayout {
        &self.params
    }

    /// Runtime-prime profile.
    pub const fn profile(&self) -> MultiswapPrimeProfile {
        self.profile
    }

    /// Canonical digest of the integer circuit statement.
    pub const fn statement_digest(&self) -> &[u8; 32] {
        self.inner.layout().statement_digest()
    }

    /// Identity opening map.
    pub const fn map(&self) -> &RepeatedVirtualMap {
        self.inner.layout().map()
    }
}

/// Derives the production Ligerito configuration for the MultiSwap shape.
pub fn multiswap_lig_configs(
    p: &IntegerMatrixLayout,
) -> Result<(LigProverConfig, LigVerifierConfig), ProtocolError> {
    // `udrg:3:4:114`: UDR geometry at rate 1/8 with fold arity 4, fold
    // grinding, BLAKE3, validator-gated at the row's 114-bit target. Chosen
    // 2026-09-08 over the audited `udrg:1:4:128` (rate 1/2) for proof size:
    // at 2^25 committed bits the Ligerito part drops from 245 KiB to 179 KiB
    // (each query buys 0.83 instead of 0.41 bits) while the opener stays at
    // ~10 ms; the Johnson openers reach 114 bits only through a ~17 s
    // Round-0 grind at this shape.
    validated_udr_lig_configs_with(packed_vars(p), 3, 4, 114).map_err(ProtocolError::LigeritoConfig)
}

/// Commits prebuilt packed witness/quotient bit rows.
pub fn commit_multiswap_witness(
    p: &IntegerMatrixLayout,
    rows: Vec<Vec<u64>>,
    pc: &LigProverConfig,
) -> Result<FlockCommitHint, ProtocolError> {
    if p.word_bits != 1 {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    protocol::validate_bit_rows(p, &rows)?;
    let hint = crate::ligerito_flock::commit_rs_ligerito_rows(p, rows, pc);
    crate::ligerito_flock::validate_ligerito_commitment(&hint.commitment, pc)
        .map_err(ProtocolError::Bitz)?;
    Ok(hint)
}

/// Proves the MultiSwap Mod-R1CS against a committed bit witness.
pub fn prove_multiswap_mod_r1cs<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedMultiswapRelation,
    assignment: &MultiswapAssignment,
    hint: &FlockCommitHint,
    pc: &LigProverConfig,
) -> Result<Proof<IntEvalRsLigVirtProof>, ProtocolError> {
    prepared.validate_config(pc)?;
    protocol::prove_reduced(transcript, &prepared.inner, assignment, hint)
}

/// Verifies the MultiSwap proof, re-deriving both primes from the bound
/// transcript.
pub fn verify_multiswap_mod_r1cs<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedMultiswapRelation,
    commitment: &Commitment,
    proof: &Proof<IntEvalRsLigVirtProof>,
    vc: &LigVerifierConfig,
) -> Result<(), ProtocolError> {
    prepared.validate_config(vc)?;
    protocol::verify_reduced(transcript, &prepared.inner, commitment, proof)
}

fn config_digest(config: &impl LigeritoStatementConfig) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(b"bitz/multiswap/ligerito-config/v1");
    for v in [
        config.recursive_steps(),
        config.initial_log_msg_cols(),
        config.initial_log_num_interleaved(),
        config.initial_k(),
    ] {
        h.update(&(v as u64).to_le_bytes());
    }
    for vs in [
        config.log_inv_rates(),
        config.recursive_log_msg_cols(),
        config.recursive_ks(),
        config.queries(),
        config.grinding_bits(),
        config.fold_grinding_bits(),
        config.ood_samples(),
    ] {
        h.update(&(vs.len() as u64).to_le_bytes());
        for v in vs {
            h.update(&(*v as u64).to_le_bytes());
        }
    }
    h.update(&[match config.merkle_hash() {
        flock_core::merkle::HashKind::Sha256 => 0,
        flock_core::merkle::HashKind::Blake3 => 1,
    }]);
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::super::circuit::MultiswapDims;
    use super::*;
    use crate::{piop::spartan::profile::Lambda100, transcript::Blake3Transcript};

    fn mini_setup() -> (
        PreparedMultiswapRelation,
        MultiswapAssignment,
        FlockCommitHint,
        LigProverConfig,
        LigVerifierConfig,
    ) {
        let circuit = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
        circuit.is_sat_integer().unwrap();
        let prepared = PreparedMultiswapRelation::new(&circuit).unwrap();
        let assignment = MultiswapAssignment::new(&circuit).unwrap();
        let (pc, vc) = prepared.ligerito_configs();
        let hint =
            commit_multiswap_witness(prepared.params(), assignment.bitz_bit_rows(), &pc).unwrap();
        (prepared, assignment, hint, pc, vc)
    }

    #[test]
    fn mini_multiswap_roundtrips_and_is_deterministic() {
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (prepared, assignment, hint, pc, vc) = mini_setup();
        assert_eq!(prepared.security().lambda, 114);
        assert!(prepared.security().projection_full_width);
        assert_eq!(prepared.profile().reduction_grinding_bits(), 10);

        let mut pt = Blake3Transcript::new();
        let proof = prove_multiswap_mod_r1cs(&mut pt, &prepared, &assignment, &hint, &pc).unwrap();
        assert!(proof.piop_nonces().is_empty());
        assert!(proof.mu_prime().is_some());
        let mut vt = Blake3Transcript::new();
        verify_multiswap_mod_r1cs(&mut vt, &prepared, &hint.commitment, &proof, &vc).unwrap();

        let mut second = Blake3Transcript::new();
        let again =
            prove_multiswap_mod_r1cs(&mut second, &prepared, &assignment, &hint, &pc).unwrap();
        assert_eq!(again.bitz().to_bytes(), proof.bitz().to_bytes());
        assert_eq!(again.mu_prime(), proof.mu_prime());
        assert_eq!(second.state_digest(), pt.state_digest());

        // A wrong integer lift is rejected before the reduction draw.
        let (prefix, reduction, bitz) = proof.clone().into_parts();
        let mut reduction = reduction.unwrap();
        reduction.mu_prime = reduction.mu_prime.wrapping_add(&field::Uint::ONE);
        let tampered = Proof::<IntEvalRsLigVirtProof>::from_parts(prefix, Some(reduction), bitz);
        assert!(matches!(
            verify_multiswap_mod_r1cs(
                &mut Blake3Transcript::new(),
                &prepared,
                &hint.commitment,
                &tampered,
                &vc
            ),
            Err(ProtocolError::InvalidIntegerLift)
        ));

        // A foreign opener configuration is rejected up front.
        let foreign =
            validated_udr_lig_configs_with(packed_vars(prepared.params()), 1, 4, 114).unwrap();
        assert!(matches!(
            prove_multiswap_mod_r1cs(
                &mut Blake3Transcript::new(),
                &prepared,
                &assignment,
                &hint,
                &foreign.0
            ),
            Err(ProtocolError::Bitz(FlockRsError::CommitmentConfig))
        ));
    }

    #[test]
    fn mini_batches_preserve_verified_codec_roundtrips() {
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for batch in [1, 2, 4] {
            let circuit = MultiswapCircuit::build_batch(MultiswapDims::mini(), batch).unwrap();
            let prepared = PreparedMultiswapRelation::new(&circuit).unwrap();
            let assignment = MultiswapAssignment::new(&circuit).unwrap();
            let (pc, vc) = prepared.ligerito_configs();
            let hint = commit_multiswap_witness(prepared.params(), assignment.bitz_bit_rows(), &pc)
                .unwrap();
            let mut prover = Blake3Transcript::new();
            let proof =
                prove_multiswap_mod_r1cs(&mut prover, &prepared, &assignment, &hint, &pc).unwrap();
            let original = proof.clone();
            let (prefix, reduction, opening) = proof.into_parts();
            let bytes = opening.to_bytes();
            let decoded = IntEvalRsLigVirtProof::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.to_bytes(), bytes);
            let decoded = Proof::from_parts(prefix, reduction, decoded);
            let mut verifier = Blake3Transcript::new();
            verify_multiswap_mod_r1cs(&mut verifier, &prepared, &hint.commitment, &decoded, &vc)
                .unwrap();
            let mut original_verifier = Blake3Transcript::new();
            verify_multiswap_mod_r1cs(
                &mut original_verifier,
                &prepared,
                &hint.commitment,
                &original,
                &vc,
            )
            .unwrap();
            assert_eq!(original_verifier.state_digest(), verifier.state_digest());
        }
    }

    #[test]
    fn single_prime_profiles_are_rejected() {
        let circuit = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
        assert!(matches!(
            PreparedMultiswapRelation::new_with_profile::<Lambda100>(&circuit),
            Err(ProtocolError::UnsupportedProfile)
        ));
    }
}
