//! The linear-batching PIOP: relations whose constraints are all linear in
//! a derived bit grid `h = M·f` (the SHA-256 compression and chain
//! relations). There is no nonlinear outer sumcheck: the verifier draws a
//! local-row point `ξ` and an instance point `η`, batches the public
//! input/output equalities and the shared constant behind two more
//! challenges, and the collapsed relation is one rank-one functional over
//! the assignment grid. A power-of-two batch in the product layout opens
//! that functional directly; otherwise a quadratic inner sumcheck over the
//! assignment domain first reduces it to a point (the legacy path).
//!
//! The steps, on one transcript:
//!
//! 1. the relation's statement frames, the opener policy digest, Round 0;
//! 2. the initial grinding boundary, the runtime prime, the runtime
//!    relation frames;
//! 3. `ξ`, `η`, the terminal boundary, the constant-one batch challenge,
//!    the public-IO batch challenges; the relation's batching;
//! 4. (legacy) the inner sumcheck; the opening claim frame;
//! 5. the virtual opening of the derived grid.

use crate::sumcheck::boundary::ProverGrindingRoundBoundary;
use crate::sumcheck::inner::{
    packed::{PackedInput, Sha256InnerGrinding},
    prove_inner_sumcheck,
};
use field::RingOps;
use flock_core::pcs::{
    commit::Commitment,
    ligerito::{ProverConfig as LigProverConfig, VerifierConfig as LigVerifierConfig},
};

use {
    crate::{
        ligerito::packed_vars,
        ligerito_flock::{
            FlockCommitHint, IntEvalRsLigVirtProof, LigeritoStatementConfig, ResolvedLigerito,
            bind_prover_ood, bind_verifier_ood,
            prove_mle_eval_mod_q_ligerito_virtual_with_weight_source_runtime,
            verify_mle_eval_mod_q_ligerito_virtual_with_weight_source_runtime,
        },
        pcs::{GeneratedModQWeightSource, IntegerMatrixLayout, ModQWeightSource},
        transcript::traits::Transcript,
    },
    circuit::linear_map::binary::VirtualMap,
};

use super::{
    super::{
        absorb_spartan_message,
        sha256::inner_sumcheck::{SHA256_INNER_PREFIX_MAX_VARS, Sha256InnerBitSource},
        squeeze_field,
        sumcheck::SumcheckProof,
    },
    BindingHasher, FieldConfig, ProtocolError, SpartanBitzField, check_boundary, bitz_generator,
    grind_boundary,
};
use crate::piop::spartan::profile::IopSecurityParams;
use crate::poly::mle::FactoredMultilinearExtension;

/// The transcript domain strings of one linear relation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearDomains {
    pub initial_grinding: &'static [u8],
    /// The terminal boundary before the batching challenges.
    pub terminal_grinding: &'static [u8],
    pub local_point: &'static [u8],
    pub instance_point: &'static [u8],
    pub constant_one: &'static [u8],
    pub public_io: &'static [u8],
    /// Tag and digest domain of the opening-claim frame.
    pub claim_tag: &'static [u8],
    pub claim_domain: &'static [u8],
    /// Extra bytes after the digest domain on the product path.
    pub claim_variant: Option<&'static [u8]>,
}

/// The relation's batching of its collapsed constraints: the initial claim
/// `μ`, the factored coefficient table `V` of the legacy inner sumcheck,
/// and its closed-form evaluation.
pub(crate) trait LinearBatching {
    fn initial_claim(&self) -> &SpartanBitzField;

    fn factored_matrix_mle(
        &self,
        config: &FieldConfig,
    ) -> Result<FactoredMultilinearExtension<'_, SpartanBitzField>, ProtocolError>;

    fn evaluate(
        &self,
        point: &[SpartanBitzField],
        config: &FieldConfig,
    ) -> Result<SpartanBitzField, ProtocolError>;
}

/// The rank-one functional the virtual opening discharges: the canonical
/// row weight of every opened row on demand, the clear column weights and
/// the claimed value.
pub(crate) struct OpeningClaim<'a> {
    pub rows: Box<dyn Fn(usize) -> Option<u128> + Sync + 'a>,
    pub cols: Vec<u128>,
    pub claimed: u128,
}

/// The claim's rows as the opener's weight source.
fn weight_source<'a>(
    layout: &IntegerMatrixLayout,
    q_bits: usize,
    claim: &'a OpeningClaim<'a>,
) -> Result<GeneratedModQWeightSource<&'a (dyn Fn(usize) -> Option<u128> + Sync + 'a)>, ProtocolError>
{
    GeneratedModQWeightSource::new(layout, q_bits, &*claim.rows)
        .map_err(|()| ProtocolError::InvalidGeometry)
}

/// The static description of one linear relation.
pub(crate) trait LinearRelationSpec: Sync {
    type Statement: ?Sized + Sync;
    type Witness: ?Sized + Sync;
    type Map: VirtualMap;
    type Batching: LinearBatching;

    fn domains(&self) -> &'static LinearDomains;
    fn security(&self) -> &IopSecurityParams;
    fn ligerito(&self) -> Result<&ResolvedLigerito, ProtocolError>;

    /// The committed source tensor.
    fn source_layout(&self) -> &IntegerMatrixLayout;
    /// The derived grid the opening runs against.
    fn opened_layout(&self) -> &IntegerMatrixLayout;
    fn map(&self) -> &Self::Map;
    /// Whether the batch opens its rank-one claim directly (no inner sumcheck).
    fn product_layout(&self) -> bool;

    fn local_vars(&self) -> usize;
    fn instance_vars(&self) -> usize;
    fn public_io_count(&self) -> usize;

    fn validate_statement(&self, statement: &Self::Statement) -> Result<(), ProtocolError>;
    fn validate_opener_config(
        &self,
        config: &dyn LigeritoStatementConfig,
    ) -> Result<(), ProtocolError>;
    fn validate_witness(
        &self,
        statement: &Self::Statement,
        witness: &Self::Witness,
        hint: &FlockCommitHint,
    ) -> Result<(), ProtocolError>;

    /// The statement frames in absorption order, and the assignment binding
    /// the opening claim later cites.
    fn statement_frames(
        &self,
        statement: &Self::Statement,
        commitment: &Commitment,
        config: &dyn LigeritoStatementConfig,
    ) -> Result<(Vec<(&'static [u8], Vec<u8>)>, [u8; 32]), ProtocolError>;

    fn runtime_prime<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> Result<field::FpCtx<2>, ProtocolError>;

    /// The frames bound after the prime draw.
    fn runtime_relation_frames(&self, config: &FieldConfig) -> Vec<(&'static [u8], Vec<u8>)>;

    #[allow(clippy::too_many_arguments)]
    fn batching(
        &self,
        statement: &Self::Statement,
        local_point: &[SpartanBitzField],
        instance_point: &[SpartanBitzField],
        slot_weights: Vec<SpartanBitzField>,
        public_io_batch: SpartanBitzField,
        constant_one: SpartanBitzField,
        reducer: &field::FpCtx<2>,
        config: &FieldConfig,
    ) -> Result<Self::Batching, ProtocolError>;

    /// The product path's rank-one claim.
    fn product_claim<'a>(
        &'a self,
        batching: &'a Self::Batching,
        prime: &'a field::FpCtx<2>,
    ) -> Result<OpeningClaim<'a>, ProtocolError>;

    /// The legacy path's claim from the inner sumcheck's terminal point.
    fn inner_claim<'a>(
        &'a self,
        point: &[SpartanBitzField],
        coefficient_evaluation: &SpartanBitzField,
        final_claim: SpartanBitzField,
        prime: &'a field::FpCtx<2>,
    ) -> Result<OpeningClaim<'a>, ProtocolError>;

    /// The assignment grid bits the legacy inner sumcheck multiplies by.
    fn inner_bits<'a>(
        &'a self,
        witness: &'a Self::Witness,
    ) -> Result<Box<dyn Sha256InnerBitSource + 'a>, ProtocolError>;

    /// The derived rows the opening runs against.
    fn opened_rows<'a>(&self, witness: &'a Self::Witness) -> Result<&'a [Vec<u64>], ProtocolError>;
}

/// A linear-relation proof: the boundary nonces, the legacy inner sumcheck
/// (empty on the product path) and the virtual opening.
#[derive(Clone)]
pub struct LinearProof {
    initial_nonce: u64,
    inner: SumcheckProof<SpartanBitzField, 3>,
    inner_nonces: Vec<u64>,
    terminal_nonce: u64,
    bitz: IntEvalRsLigVirtProof,
}

impl LinearProof {
    pub const fn initial_nonce(&self) -> u64 {
        self.initial_nonce
    }

    pub const fn inner(&self) -> &SumcheckProof<SpartanBitzField, 3> {
        &self.inner
    }

    pub fn inner_nonces(&self) -> &[u64] {
        &self.inner_nonces
    }

    pub const fn terminal_nonce(&self) -> u64 {
        self.terminal_nonce
    }

    pub const fn bitz(&self) -> &IntEvalRsLigVirtProof {
        &self.bitz
    }

    /// PIOP payload bytes: the two boundary nonces, the inner round
    /// coefficients and their nonces.
    pub fn piop_bytes(&self) -> usize {
        2 * 8 + 3 * 16 * self.inner.round_polynomials.len() + 8 * self.inner_nonces.len()
    }

    pub fn initial_nonce_mut(&mut self) -> &mut u64 {
        &mut self.initial_nonce
    }

    pub fn inner_mut(&mut self) -> &mut SumcheckProof<SpartanBitzField, 3> {
        &mut self.inner
    }

    pub fn inner_nonces_mut(&mut self) -> &mut Vec<u64> {
        &mut self.inner_nonces
    }

    pub fn terminal_nonce_mut(&mut self) -> &mut u64 {
        &mut self.terminal_nonce
    }

    pub fn bitz_mut(&mut self) -> &mut IntEvalRsLigVirtProof {
        &mut self.bitz
    }
}

/// Prover-side options that do not move the transcript.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LinearProveOptions {
    /// The legacy inner prover's packed-prefix width.
    pub prefix_vars: usize,
}

/// One statement frame: its tag and payload.
pub(crate) fn frame(tag: &'static [u8], bytes: impl AsRef<[u8]>) -> (&'static [u8], Vec<u8>) {
    (tag, bytes.as_ref().to_vec())
}

fn absorb_frames<T: Transcript>(transcript: &mut T, frames: &[(&'static [u8], Vec<u8>)]) {
    for (tag, bytes) in frames {
        absorb_spartan_message(transcript, tag, bytes);
    }
}

fn challenge_point<T: Transcript>(
    transcript: &mut T,
    domain: &[u8],
    vars: usize,
    config: &FieldConfig,
) -> Result<Vec<SpartanBitzField>, ProtocolError> {
    absorb_spartan_message(transcript, b"challenge-domain", domain);
    Ok((0..vars)
        .map(|_| squeeze_field(transcript, config))
        .collect::<Result<Vec<_>, _>>()?)
}

/// The batching challenges after the terminal boundary.
struct BatchChallenges {
    constant_one: SpartanBitzField,
    slot_weights: Vec<SpartanBitzField>,
    public_io_batch: SpartanBitzField,
}

fn batch_challenges<T: Transcript, S: LinearRelationSpec>(
    transcript: &mut T,
    spec: &S,
    config: &FieldConfig,
) -> Result<BatchChallenges, ProtocolError> {
    let domains = spec.domains();
    let constant_one = challenge_point(transcript, domains.constant_one, 1, config)?
        .pop()
        .expect("one challenge");
    let mut public = challenge_point(
        transcript,
        domains.public_io,
        spec.public_io_count() + 1,
        config,
    )?;
    let public_io_batch = public.pop().expect("the batch weight");
    Ok(BatchChallenges {
        constant_one,
        slot_weights: public,
        public_io_batch,
    })
}

/// The opening-claim digest:
///
/// `domain ‖ [variant] ‖ binding ‖ |ξ| ξ ‖ |η| η ‖ [|r| r ‖ V(r)] ‖ α₀ ‖ |λ| λ ‖ α_pub ‖ |rows| rows ‖ |cols| cols ‖ claimed`
#[allow(clippy::too_many_arguments)]
fn claim_digest(
    domains: &LinearDomains,
    binding: &[u8; 32],
    local_point: &[SpartanBitzField],
    instance_point: &[SpartanBitzField],
    assignment: Option<(&[SpartanBitzField], &SpartanBitzField)>,
    challenges: &BatchChallenges,
    claim: &OpeningClaim<'_>,
    rows: &impl ModQWeightSource,
    field_config: &FieldConfig,
) -> Result<[u8; 32], ProtocolError> {
    let mut hasher = BindingHasher::new();
    hasher.bytes(domains.claim_domain);
    if assignment.is_none() {
        if let Some(variant) = domains.claim_variant {
            hasher.bytes(variant);
        }
    }
    hasher.bytes(binding);
    for point in [local_point, instance_point] {
        hasher.usize(point.len())?;
        for coordinate in point {
            hasher.element(coordinate, field_config);
        }
    }
    if let Some((point, evaluation)) = assignment {
        hasher.usize(point.len())?;
        for coordinate in point {
            hasher.element(coordinate, field_config);
        }
        hasher.element(evaluation, field_config);
    }
    hasher.element(&challenges.constant_one, field_config);
    hasher.usize(challenges.slot_weights.len())?;
    for coordinate in &challenges.slot_weights {
        hasher.element(coordinate, field_config);
    }
    hasher.element(&challenges.public_io_batch, field_config);
    hasher.usize(rows.row_count())?;
    for row in 0..rows.row_count() {
        let weight = rows
            .canonical_weight(row)
            .ok_or(ProtocolError::InvalidGeometry)?;
        hasher.u128_le(weight);
    }
    hasher.usize(claim.cols.len())?;
    for weight in &claim.cols {
        hasher.u128_le(*weight);
    }
    hasher.u128_le(claim.claimed);
    Ok(hasher.finalize())
}

/// Proves a linear relation against its committed source rows.
pub(crate) fn prove_linear<T: Transcript + Send, S: LinearRelationSpec>(
    transcript: &mut T,
    spec: &S,
    statement: &S::Statement,
    witness: &S::Witness,
    hint: &FlockCommitHint,
    pc: &LigProverConfig,
    options: LinearProveOptions,
) -> Result<LinearProof, ProtocolError> {
    spec.validate_statement(statement)?;
    spec.validate_opener_config(pc)?;
    spec.validate_witness(statement, witness, hint)?;
    if options.prefix_vars > SHA256_INNER_PREFIX_MAX_VARS {
        return Err(ProtocolError::InvalidInnerPrefix {
            actual: options.prefix_vars,
            max: SHA256_INNER_PREFIX_MAX_VARS,
        });
    }
    crate::ligerito_flock::validate_ligerito_commitment(&hint.commitment, pc)
        .map_err(ProtocolError::Bitz)?;
    let f_layout = spec.source_layout();
    if hint.commitment.params.m != f_layout.row_vars + f_layout.col_vars {
        return Err(ProtocolError::InvalidGeometry);
    }
    let security = spec.security();
    let domains = spec.domains();

    let binding = {
        let _scope = tracing::info_span!("sha256:statement_bind_prover").entered();
        let (frames, binding) = spec.statement_frames(statement, &hint.commitment, pc)?;
        absorb_frames(transcript, &frames);
        binding
    };
    spec.ligerito()?.bind(transcript);
    let ood = bind_prover_ood(transcript, hint, security.ood);

    // Commit first, then derive the one runtime prime. The exact signed
    // relation stays q-independent; only its one local-row collapse is
    // performed in the sampled field.
    let step2_scope = tracing::info_span!("step2:project_prove").entered();
    let initial_nonce = {
        let _scope = tracing::info_span!("sha256:initial_grinding_prove").entered();
        grind_boundary(
            transcript,
            domains.initial_grinding,
            security.initial_grinding_bits,
        )?
    };
    let prime = {
        let _scope = tracing::info_span!("sha256:runtime_prime_sample_prover").entered();
        spec.runtime_prime(transcript)?
    };
    absorb_frames(transcript, &spec.runtime_relation_frames(&prime));
    drop(step2_scope);

    // Collapse the local rows and form the product-structured residual.
    // Power-of-two batches open it directly; partial batches use the legacy
    // assignment-domain sumcheck below. There is no nonlinear outer sumcheck.
    let step3_scope = tracing::info_span!("step3:piop_prove").entered();
    let config = &prime;
    let local_point = challenge_point(transcript, domains.local_point, spec.local_vars(), config)?;
    let instance_point = challenge_point(
        transcript,
        domains.instance_point,
        spec.instance_vars(),
        config,
    )?;
    let terminal_nonce = {
        let _scope = tracing::info_span!("sha256:public_batch_grinding_prove").entered();
        grind_boundary(
            transcript,
            domains.terminal_grinding,
            security.terminal_grinding_bits,
        )?
    };
    let challenges = batch_challenges(transcript, spec, config)?;
    let reducer = {
        let _scope = tracing::info_span!("sha256:reducer_init_prover").entered();
        crate::utils::delayed_reduction::prepare_field(config)
            .map_err(super::super::sumcheck::SumcheckError::from)
            .map_err(super::super::piop::SpartanError::from)?
    };
    let batching = {
        let _scope = tracing::info_span!("sha256:product_batch_prepare_prover").entered();
        spec.batching(
            statement,
            &local_point,
            &instance_point,
            challenges.slot_weights.clone(),
            challenges.public_io_batch.clone(),
            challenges.constant_one.clone(),
            &reducer,
            config,
        )?
    };

    let (inner, inner_nonces, claim, assignment) = if spec.product_layout() {
        drop(step3_scope);
        let step4_scope = tracing::info_span!("step4:bitify_prove").entered();
        let claim = {
            let _scope = tracing::info_span!("sha256:direct_opening_prepare_prover").entered();
            spec.product_claim(&batching, &prime)?
        };
        drop(step4_scope);
        (
            SumcheckProof {
                round_polynomials: Vec::new(),
            },
            Vec::new(),
            claim,
            None,
        )
    } else {
        let (inner, inner_nonces) = {
            let _scope = tracing::info_span!("sha256:spartan_inner_prove").entered();
            let factored = batching.factored_matrix_mle(config)?;
            let bits = spec.inner_bits(witness)?;
            {
                let coefficients = &factored;
                let mut boundary =
                    ProverGrindingRoundBoundary::<Sha256InnerGrinding>::with_round_offset(
                        security.piop_round_grinding_bits,
                        0,
                    );
                prove_inner_sumcheck(
                    config,
                    transcript,
                    batching.initial_claim().clone(),
                    PackedInput::new(
                        coefficients,
                        &*bits,
                        spec.opened_layout().row_vars + spec.opened_layout().col_vars,
                        coefficients.live_len(),
                        options.prefix_vars,
                    ),
                    (),
                    &mut boundary,
                )
                .map(|out| (out, boundary.into_nonces()))
            }
            .map_err(super::super::piop::SpartanError::from)?
        };
        if inner.final_claim
            != config.mul(
                &(inner.terminal_evaluations[0].clone()),
                &(&inner.terminal_evaluations[1]),
            )
        {
            return Err(ProtocolError::InvalidInnerTerminalClaim);
        }
        drop(step3_scope);
        let step4_scope = tracing::info_span!("step4:bitify_prove").entered();
        let claim = {
            let _scope = tracing::info_span!("sha256:opening_prepare_prover").entered();
            spec.inner_claim(
                &inner.point,
                &inner.terminal_evaluations[0],
                inner.final_claim.clone(),
                &prime,
            )?
        };
        drop(step4_scope);
        (
            inner.proof,
            inner_nonces,
            claim,
            Some((inner.point, inner.terminal_evaluations[0])),
        )
    };
    let rows = weight_source(spec.opened_layout(), prime.modulus_bits(), &claim)?;
    {
        let _step4 = tracing::info_span!("step4:bitify_prove").entered();
        let _scope = tracing::info_span!("sha256:opening_claim_absorb_prover").entered();
        let digest = claim_digest(
            domains,
            &binding,
            &local_point,
            &instance_point,
            assignment
                .as_ref()
                .map(|(point, evaluation)| (point.as_slice(), evaluation)),
            &challenges,
            &claim,
            &rows,
            &config,
        )?;
        absorb_spartan_message(transcript, domains.claim_tag, &digest);
    }

    let bitz = {
        let _step5 = tracing::info_span!("step5:open_prove").entered();
        let _scope = tracing::info_span!("sha256:bitz_prove").entered();
        prove_mle_eval_mod_q_ligerito_virtual_with_weight_source_runtime(
            transcript,
            hint,
            spec.opened_rows(witness)?,
            spec.opened_layout(),
            f_layout,
            spec.map(),
            &rows,
            prime.modulus_u128(),
            prime.modulus_bits(),
            bitz_generator(),
            security.forest_round_grinding_bits,
            ood,
            pc,
        )
        .map_err(ProtocolError::Bitz)?
    };

    Ok(LinearProof {
        initial_nonce,
        inner,
        inner_nonces,
        terminal_nonce,
        bitz,
    })
}

/// Verifies a linear-relation proof, re-deriving the prime from the bound
/// transcript.
pub(crate) fn verify_linear<T: Transcript + Send, S: LinearRelationSpec>(
    transcript: &mut T,
    spec: &S,
    statement: &S::Statement,
    commitment: &Commitment,
    proof: &LinearProof,
    vc: &LigVerifierConfig,
) -> Result<(), ProtocolError> {
    spec.validate_statement(statement)?;
    spec.validate_opener_config(vc)?;
    crate::ligerito_flock::validate_ligerito_commitment(commitment, vc)
        .map_err(ProtocolError::Bitz)?;
    let f_layout = spec.source_layout();
    let opened = spec.opened_layout();
    let product_layout = spec.product_layout();
    let expected_inner_rounds = if product_layout {
        0
    } else {
        opened.row_vars + opened.col_vars
    };
    if commitment.params.m != f_layout.row_vars + f_layout.col_vars
        || proof.inner.round_polynomials.len() != expected_inner_rounds
        || (product_layout && !proof.inner_nonces.is_empty())
    {
        return Err(ProtocolError::InvalidGeometry);
    }
    let security = spec.security();
    let domains = spec.domains();

    let binding = {
        let _scope = tracing::info_span!("sha256:statement_bind_verifier").entered();
        let (frames, binding) = spec.statement_frames(statement, commitment, vc)?;
        absorb_frames(transcript, &frames);
        binding
    };
    spec.ligerito()?.bind(transcript);
    let ood = bind_verifier_ood(
        transcript,
        packed_vars(f_layout),
        security.ood,
        proof.bitz.ood.as_ref(),
    )
    .map_err(ProtocolError::Bitz)?;

    let step2_scope = tracing::info_span!("step2:project_verify").entered();
    {
        let _scope = tracing::info_span!("sha256:initial_grinding_verify").entered();
        check_boundary(
            transcript,
            domains.initial_grinding,
            security.initial_grinding_bits,
            proof.initial_nonce,
        )?;
    }
    let prime = {
        let _scope = tracing::info_span!("sha256:runtime_prime_sample_verifier").entered();
        spec.runtime_prime(transcript)?
    };
    absorb_frames(transcript, &spec.runtime_relation_frames(&prime));
    drop(step2_scope);

    let step3_scope = tracing::info_span!("step3:piop_verify").entered();
    let config = &prime;
    let local_point = challenge_point(transcript, domains.local_point, spec.local_vars(), config)?;
    let instance_point = challenge_point(
        transcript,
        domains.instance_point,
        spec.instance_vars(),
        config,
    )?;
    {
        let _scope = tracing::info_span!("sha256:public_batch_grinding_verify").entered();
        check_boundary(
            transcript,
            domains.terminal_grinding,
            security.terminal_grinding_bits,
            proof.terminal_nonce,
        )?;
    }
    let challenges = batch_challenges(transcript, spec, config)?;
    let reducer = {
        let _scope = tracing::info_span!("sha256:reducer_init_verifier").entered();
        crate::utils::delayed_reduction::prepare_field(config)
            .map_err(super::super::sumcheck::SumcheckError::from)
            .map_err(super::super::piop::SpartanError::from)?
    };
    let batching = {
        let _scope = tracing::info_span!("sha256:product_batch_prepare_verifier").entered();
        spec.batching(
            statement,
            &local_point,
            &instance_point,
            challenges.slot_weights.clone(),
            challenges.public_io_batch.clone(),
            challenges.constant_one.clone(),
            &reducer,
            config,
        )?
    };

    let (claim, assignment) = if product_layout {
        drop(step3_scope);
        let _step4 = tracing::info_span!("step4:bitify_verify").entered();
        let _scope = tracing::info_span!("sha256:direct_opening_prepare_verifier").entered();
        (spec.product_claim(&batching, &prime)?, None)
    } else {
        let (assignment_point, inner_claim) = {
            let _scope = tracing::info_span!("sha256:spartan_inner_verify").entered();
            proof
                .inner
                .verify_grinded::<Sha256InnerGrinding>(
                    transcript,
                    batching.initial_claim().clone(),
                    opened.row_vars + opened.col_vars,
                    config,
                    &proof.inner_nonces,
                    security.piop_round_grinding_bits,
                )
                .map_err(super::super::piop::SpartanError::from)?
        };
        drop(step3_scope);
        let _step4 = tracing::info_span!("step4:bitify_verify").entered();
        let collapsed_evaluation = batching.evaluate(&assignment_point, config)?;
        let claim = {
            let _scope = tracing::info_span!("sha256:opening_prepare_verifier").entered();
            spec.inner_claim(
                &assignment_point,
                &collapsed_evaluation,
                inner_claim,
                &prime,
            )?
        };
        (claim, Some((assignment_point, collapsed_evaluation)))
    };
    let rows = weight_source(opened, prime.modulus_bits(), &claim)?;
    {
        let _step4 = tracing::info_span!("step4:bitify_verify").entered();
        let _scope = tracing::info_span!("sha256:opening_claim_absorb_verifier").entered();
        let digest = claim_digest(
            domains,
            &binding,
            &local_point,
            &instance_point,
            assignment
                .as_ref()
                .map(|(point, evaluation)| (point.as_slice(), evaluation)),
            &challenges,
            &claim,
            &rows,
            &config,
        )?;
        absorb_spartan_message(transcript, domains.claim_tag, &digest);
    }

    let _step5 = tracing::info_span!("step5:open_verify").entered();
    let _scope = tracing::info_span!("sha256:bitz_verify").entered();
    verify_mle_eval_mod_q_ligerito_virtual_with_weight_source_runtime(
        transcript,
        commitment,
        &proof.bitz,
        opened,
        f_layout,
        spec.map(),
        &rows,
        &claim.cols,
        bitz_generator(),
        claim.claimed,
        prime.modulus_u128(),
        prime.modulus_bits(),
        security.forest_round_grinding_bits,
        ood,
        vc,
    )
    .map_err(ProtocolError::Bitz)
}
