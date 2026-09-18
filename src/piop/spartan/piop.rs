//! Composition of Spartan's outer and inner sumchecks.
#[cfg(test)]
use crate::piop::spartan::mul::MulWitness;
use crate::sumcheck::{
    UngrindedRoundBoundary,
    bridge::{
        PreparedBinding,
        native::{NativeBinding, RowFunctional},
    },
    inner::prove_inner_sumcheck,
    outer::{
        self, EqualityFactors, OuterArithmetic, OuterClaim, OuterRows,
        arithmetic::{NativeProducts, factors_from_raw},
    },
    proof::OuterSumcheckOutput,
};
#[cfg(test)]
use circuit::linear_map::CscMatrix;

use crate::piop::spartan::raw_monty::RawFieldStorage;
use blake3::Hasher;
#[cfg(test)]
use field::Uint;
use field::{Fp, RingOps};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;

use crate::{poly::mle::DenseMultilinearExtension, transcript::traits::Transcript};

use super::{
    SpartanField, absorb_spartan_message,
    matrix::{
        MleClaimError, PreparedConstraintMatrices, ProductRowFunctional, ScaledMleEvaluationClaim,
        SpartanMatrixCoefficient, SpartanMatrixError, make_equality_factors,
    },
    raw_monty::{NativeConstantPrefix, RawMontyCoefficient, RawWitness, make_equality_factors_raw},
    squeeze_field,
    sumcheck::{OuterSumcheckProof, R1csProductMles, SumcheckError, SumcheckProof},
    univariate_skip::UnivariateSkipSpartanPiopProof,
};

/// Domain separator for the native Spartan PIOP transcript.
///
/// Version two identifies the target-native transcript fork: it uses BLAKE3
/// statement digests, runtime field configurations, exact challenge sampling,
/// and an explicit assignment-oracle binding.
pub const SPARTAN_PIOP_DOMAIN: &[u8] = b"bitz/spartan/piop/v2";

/// Domain separator for the known-zero univariate-skip PIOP.
///
/// The standard `v2` schedule deliberately retains its original domain and
/// transcript bytes.  A distinct domain prevents either proof shape from
/// being replayed as the other.
pub const SPARTAN_UNIVARIATE_SKIP_PIOP_DOMAIN: &[u8] = b"bitz/spartan/piop/univariate-skip/v1";

/// Domain separator for the assignment-oracle commitment in the PIOP
/// statement.
pub const SPARTAN_ASSIGNMENT_ORACLE_DOMAIN: &[u8] = b"bitz/spartan/assignment-oracle/v1";

const NONSUCCINCT_ASSIGNMENT_DIGEST_DOMAIN: &[u8] = b"bitz/spartan/full-assignment-digest/v1";

/// The cubic outer and quadratic inner sumcheck proofs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanPiopProof<F> {
    pub outer: OuterSumcheckProof<F>,
    pub inner: SumcheckProof<F, 3>,
}

/// Failures while composing or checking the Spartan PIOP.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SpartanError {
    #[error(transparent)]
    Matrix(#[from] SpartanMatrixError),

    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    #[error(transparent)]
    MleClaim(#[from] MleClaimError),

    #[error("Az, Bz, and Cz do not use the prepared row domain")]
    InvalidProductDimensions,

    #[error("the assignment does not use the prepared column domain")]
    InvalidAssignmentDimensions,

    #[error("the assignment has nonzero values outside the logical column range")]
    InvalidAssignmentPadding,

    #[error("a dense multilinear-extension table has invalid shape")]
    InvalidMleShape,

    #[error("a proof or witness value uses a different field configuration")]
    FieldConfigurationMismatch,

    #[error("the supplied opening claim is not the claim derived from the proof")]
    InvalidMleClaim,

    #[error("univariate skip supports K in 1..=4, got {skip_vars}")]
    InvalidUnivariateSkipVariables { skip_vars: usize },

    #[error("cannot skip {skip_vars} variables from an outer sumcheck with {row_vars} variables")]
    UnivariateSkipExceedsRowVariables { skip_vars: usize, row_vars: usize },

    #[error("univariate-skip proof has {actual} finite evaluations, expected {expected}")]
    InvalidUnivariateSkipMessageLength { expected: usize, actual: usize },
}

/// Runs the complete outer and inner Spartan reductions.
///
/// The caller supplies field-valued `A/B/C` products and the complete R1CS
/// assignment. Constraint generation and the eventual succinct opening of the
/// returned assignment-MLE claim are deliberately outside this native PIOP.
/// `assignment_oracle_binding` must canonically commit to that eventual
/// opening oracle (normally a PCS commitment) and is absorbed before any
/// Fiat--Shamir challenge. Use [`prove_spartan_nonsuccinct`] while no PCS is
/// connected; it binds a canonical digest of the complete assignment table.
pub fn prove_spartan_piop<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<F>,
    assignment: DenseMultilinearExtension<F>,
) -> Result<(SpartanPiopProof<F>, ScaledMleEvaluationClaim<F>), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    prove_spartan_piop_field_tables(
        transcript,
        matrices,
        assignment_oracle_binding,
        products,
        assignment,
    )
}

/// Runs Spartan with an explicit known-zero univariate skip for the first
/// `skip_vars` little-endian row variables.
///
/// This is a distinct proof protocol and transcript domain. The existing
/// [`prove_spartan_piop`] entry point always retains the standard cubic outer
/// sumcheck.
pub fn prove_spartan_piop_with_univariate_skip<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<F>,
    assignment: DenseMultilinearExtension<F>,
    skip_vars: usize,
) -> Result<
    (
        UnivariateSkipSpartanPiopProof<F>,
        ScaledMleEvaluationClaim<F>,
    ),
    SpartanError,
>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    prove_spartan_piop_skipped_field_tables(
        transcript,
        matrices,
        assignment_oracle_binding,
        products,
        assignment,
        skip_vars,
    )
}

/// Runs the two-limb runtime-field prover with delayed Barrett reduction.
// The raw coefficient bound is crate-internal: every coefficient type this
// crate proves with implements it, and the bound names no public API.
#[allow(private_bounds)]
pub fn prove_spartan_piop_field<C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<Fp<2>>,
    assignment: DenseMultilinearExtension<Fp<2>>,
) -> Result<(SpartanPiopProof<Fp<2>>, ScaledMleEvaluationClaim<Fp<2>>), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
{
    prove_spartan_piop_raw_field(
        transcript,
        matrices,
        assignment_oracle_binding,
        products,
        assignment,
    )
}

/// Runs the standard-outer baseline used by the controlled skip benchmark,
/// retaining exact `u64` products and assignment through the first round of
/// each sumcheck.
/// Native coefficients, native witness folding, and every later field
/// coefficient sum use delayed Barrett reduction. Field-MLE folding remains
/// immediate.
pub fn prove_spartan_piop_u32_native(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, bool>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<u64>,
    assignment: DenseMultilinearExtension<u64>,
) -> Result<(SpartanPiopProof<Fp<2>>, ScaledMleEvaluationClaim<Fp<2>>), SpartanError> {
    prove_spartan_piop_native_u64(
        transcript,
        matrices,
        assignment_oracle_binding,
        products,
        assignment,
    )
}

/// Runs the native-u32 Spartan prover with an explicit known-zero univariate
/// prefix skip. The canonical high-level U32 adapter fixes this argument to
/// `K=3`; the generic width remains public for controlled PIOP benchmarks.
///
/// Native `Az` and `Bz` interpolation remains exact in signed `i64`, while
/// `Cz` and the residual use signed `i128`. After the skip challenge all three
/// tables are folded into the configured field and the existing cubic tail is
/// reused unchanged.
pub fn prove_spartan_piop_u32_native_with_univariate_skip(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, bool>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<u64>,
    assignment: DenseMultilinearExtension<u64>,
    skip_vars: usize,
) -> Result<
    (
        UnivariateSkipSpartanPiopProof<Fp<2>>,
        ScaledMleEvaluationClaim<Fp<2>>,
    ),
    SpartanError,
> {
    validate_native_u32_prover_inputs(matrices, &products, &assignment)?;
    let domain = assignment.evaluations.len();
    prove_spartan_piop_raw_native_u64_with_skip_core(
        transcript,
        matrices,
        assignment_oracle_binding,
        NativeProducts::from_mles(&products),
        RawWitness::native_borrowed(&assignment.evaluations, domain),
        skip_vars,
    )
}

/// Native-u64 prefix skip over borrowed exact product and assignment tables.
/// Assignment entries beyond the logical column count are implicitly zero,
/// so proving does not allocate a padded witness copy.
pub(crate) fn prove_spartan_piop_native_u64_with_univariate_skip_borrowed<C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: NativeProducts<'_>,
    assignment: &[u64],
    skip_vars: usize,
    constant_prefix: Option<NativeConstantPrefix>,
) -> Result<
    (
        UnivariateSkipSpartanPiopProof<Fp<2>>,
        ScaledMleEvaluationClaim<Fp<2>>,
    ),
    SpartanError,
>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
{
    validate_native_u32_prover_slices(matrices, products, assignment)?;
    let domain = 1usize << matrices.num_column_vars();
    prove_spartan_piop_raw_native_u64_with_skip_core(
        transcript,
        matrices,
        assignment_oracle_binding,
        products,
        RawWitness::native_borrowed_with_constant_prefix(assignment, domain, constant_prefix),
        skip_vars,
    )
}

pub(crate) fn prove_spartan_piop_native_u64_borrowed<C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: NativeProducts<'_>,
    assignment: &[u64],
    constant_prefix: Option<NativeConstantPrefix>,
) -> Result<(SpartanPiopProof<Fp<2>>, ScaledMleEvaluationClaim<Fp<2>>), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
{
    validate_native_u32_prover_slices(matrices, products, assignment)?;
    let domain = 1usize << matrices.num_column_vars();
    prove_spartan_piop_raw_products_raw_witness(
        transcript,
        matrices,
        assignment_oracle_binding,
        products,
        RawWitness::native_borrowed_with_constant_prefix(assignment, domain, constant_prefix),
    )
}

/// The raw-table Spartan prover for relations whose assignment holds exact
/// `u64` values but whose per-row products do not fit `u64` (the u64
/// multiplication relation): the outer sumcheck consumes borrowed integer
/// products and the inner sumcheck consumes the native assignment. The first
/// outer fold materializes only half-sized field tables. Statement,
/// transcript, and proof are identical to [`prove_spartan_piop_field`]
/// under delayed Barrett reduction on the projected tables.
pub(crate) fn prove_spartan_piop_raw_products_native_assignment<C, I: OuterRows>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: I,
    assignment: &[u64],
    constant_prefix: Option<NativeConstantPrefix>,
) -> Result<(SpartanPiopProof<Fp<2>>, ScaledMleEvaluationClaim<Fp<2>>), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
    field::FpCtx<2>: OuterArithmetic<I::AB, I::C>,
{
    let domain = 1usize << matrices.num_column_vars();
    let column_count = matrices.matrices().column_count();
    if assignment.len() != column_count && assignment.len() != domain {
        return Err(SpartanError::InvalidAssignmentDimensions);
    }
    if assignment.first() != Some(&1) {
        return Err(SpartanMatrixError::InvalidAssignmentConstant.into());
    }
    if assignment.len() > column_count && assignment[column_count..].iter().any(|&value| value != 0)
    {
        return Err(SpartanError::InvalidAssignmentPadding);
    }
    prove_spartan_piop_raw_products_raw_witness(
        transcript,
        matrices,
        assignment_oracle_binding,
        products,
        RawWitness::native_borrowed_with_constant_prefix(assignment, domain, constant_prefix),
    )
}

/// Shared Spartan composition for borrowed exact integer products and a
/// caller-built inner-sumcheck witness (native integers or field residues). The witness table must span the padded column domain; for
/// raw residues the caller guarantees canonical entries and the constant
/// one at index zero. Statement, transcript, and proof are identical to
/// [`prove_spartan_piop_field`] under
/// delayed Barrett reduction on the projected tables.
pub(crate) fn prove_spartan_piop_raw_products_raw_witness<C, I: OuterRows>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: I,
    witness: RawWitness<'_>,
) -> Result<(SpartanPiopProof<Fp<2>>, ScaledMleEvaluationClaim<Fp<2>>), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
    field::FpCtx<2>: OuterArithmetic<I::AB, I::C>,
{
    let rows = 1usize << matrices.num_row_vars();
    if products.validate()? != matrices.num_row_vars() || products.dimensions().0 != rows {
        return Err(SpartanError::InvalidProductDimensions);
    }
    absorb_statement(transcript, matrices, assignment_oracle_binding);

    let field_config = matrices.config();
    let ctx = crate::piop::spartan::raw_monty::field_context(field_config);
    let tau = (0..matrices.num_row_vars())
        .map(|_| squeeze_field(transcript, field_config))
        .collect::<Result<Vec<_>, _>>()?;
    let (eq_low, eq_high) = {
        let _g = tracing::info_span!("sp:eq").entered();
        make_equality_factors_raw(&ctx, &tau)
    };
    let outer = {
        let _scope = tracing::info_span!("spartan:outer_sumcheck").entered();
        outer::prove_outer_sumcheck(
            &ctx,
            transcript,
            OuterClaim::RowwiseZero,
            &tau,
            products,
            Some(factors_from_raw(&ctx, eq_low, eq_high)),
            &mut UngrindedRoundBoundary,
        )
        .map(OuterSumcheckOutput::from)?
    };

    let rho = squeeze_field(transcript, field_config)?;
    let inner_initial_claim = batched_product_claim(
        &outer.proof.az_mle_claim,
        &outer.proof.bz_mle_claim,
        &outer.proof.cz_mle_claim,
        &rho,
        &field_config,
    );
    let inner = {
        let weights = NativeBinding::new(&ctx, matrices, rho)
            .bind_structured_rows(&RowFunctional::Point(&outer.eval_points))?;
        let values = witness;
        let _scope = tracing::info_span!("spartan:inner_sumcheck").entered();
        prove_inner_sumcheck(
            &ctx,
            transcript,
            inner_initial_claim,
            values,
            weights,
            &mut UngrindedRoundBoundary,
        )?
    };

    let claim = ScaledMleEvaluationClaim::new(
        inner.point.into_boxed_slice(),
        inner.terminal_evaluations[0],
        inner.final_claim,
    );
    let proof = SpartanPiopProof {
        outer: outer.proof,
        inner: inner.proof,
    };
    Ok((proof, claim))
}

/// Runs the native-u64 Spartan prover with sparse coefficients supplied by the
/// relation. Multiplicands remain bounded to 32 bits so the existing native
/// sumcheck accumulation bounds continue to apply.
///
/// This crate-private entry point lets relations such as BabyBear reuse the
/// optimized u32-native kernels without widening the public u32 API.
// The raw coefficient bound is crate-internal: every coefficient type this
// crate proves with implements it, and the bound names no public API.
#[allow(private_bounds)]
pub(crate) fn prove_spartan_piop_native_u64<C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<u64>,
    assignment: DenseMultilinearExtension<u64>,
) -> Result<(SpartanPiopProof<Fp<2>>, ScaledMleEvaluationClaim<Fp<2>>), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
{
    validate_native_u32_prover_inputs(matrices, &products, &assignment)?;
    let domain = assignment.evaluations.len();
    prove_spartan_piop_raw_products_raw_witness(
        transcript,
        matrices,
        assignment_oracle_binding,
        NativeProducts::from_mles(&products),
        RawWitness::native_borrowed(&assignment.evaluations, domain),
    )
}

fn prove_spartan_piop_field_tables<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<F>,
    assignment: DenseMultilinearExtension<F>,
) -> Result<(SpartanPiopProof<F>, ScaledMleEvaluationClaim<F>), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    {
        let _g = tracing::info_span!("sp:validate").entered();
        validate_prover_inputs(matrices, &products, &assignment)?;
    }
    absorb_statement(transcript, matrices, assignment_oracle_binding);

    let field_config = matrices.config();
    let tau = (0..matrices.num_row_vars())
        .map(|_| squeeze_field(transcript, field_config))
        .collect::<Result<Vec<_>, _>>()?;
    let equality_factors = {
        let _g = tracing::info_span!("sp:eq").entered();
        make_equality_factors(&tau, field_config)?
    };
    let outer = {
        let _scope = tracing::info_span!("spartan:outer_sumcheck").entered();
        let factors = EqualityFactors::from_mles(equality_factors, field_config)?;
        outer::prove_outer_sumcheck(
            field_config,
            transcript,
            OuterClaim::Sum(F::zero_with_cfg(field_config)),
            &tau,
            products,
            Some(factors),
            &mut UngrindedRoundBoundary,
        )
        .map(OuterSumcheckOutput::from)?
    };

    // The outer prover absorbed [Az(r_x), Bz(r_x), Cz(r_x)] before returning.
    let rho = squeeze_field(transcript, field_config)?;
    let inner_initial_claim = batched_product_claim(
        &outer.proof.az_mle_claim,
        &outer.proof.bz_mle_claim,
        &outer.proof.cz_mle_claim,
        &rho,
        &field_config,
    );
    let batched_matrix = {
        let _scope = tracing::info_span!("spartan:bind_and_batch").entered();
        crate::piop::spartan::matrix::eq_table_prover(&outer.eval_points, matrices.config())
            .map_err(crate::sumcheck::SumcheckError::from)
            .and_then(|weights| matrices.binding(&rho).bind_rows(&weights))?
    };
    let inner = {
        let _scope = tracing::info_span!("spartan:inner_sumcheck").entered();
        prove_inner_sumcheck(
            field_config,
            transcript,
            inner_initial_claim,
            assignment.evaluations,
            batched_matrix.evaluations,
            &mut UngrindedRoundBoundary,
        )?
    };

    let claim = ScaledMleEvaluationClaim::new(
        inner.point.into_boxed_slice(),
        inner.terminal_evaluations[0],
        inner.final_claim,
    );
    let proof = SpartanPiopProof {
        outer: outer.proof,
        inner: inner.proof,
    };

    Ok((proof, claim))
}

fn prove_spartan_piop_skipped_field_tables<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<F>,
    assignment: DenseMultilinearExtension<F>,
    skip_vars: usize,
) -> Result<
    (
        UnivariateSkipSpartanPiopProof<F>,
        ScaledMleEvaluationClaim<F>,
    ),
    SpartanError,
>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    validate_prover_inputs(matrices, &products, &assignment)?;
    let skip_vars = validate_univariate_skip_variables(skip_vars, matrices.num_row_vars())?;
    absorb_univariate_skip_statement(transcript, matrices, assignment_oracle_binding, skip_vars);

    let field_config = matrices.config();
    let tail_vars = matrices.num_row_vars() - usize::from(skip_vars);
    let tau_tail = (0..tail_vars)
        .map(|_| squeeze_field(transcript, field_config))
        .collect::<Result<Vec<_>, _>>()?;
    let equality_factors = make_equality_factors(&tau_tail, field_config)?;
    let outer = {
        let _scope = tracing::info_span!("spartan:outer_univariate_skip").entered();
        let factors = EqualityFactors::from_mles(equality_factors, field_config)?;
        let prepared = outer::prepare_univariate_skip(field_config, skip_vars)?;
        outer::prove_outer_zerocheck_with_skip(
            field_config,
            transcript,
            &prepared,
            &tau_tail,
            products,
            Some(factors),
            &mut UngrindedRoundBoundary,
        )
        .map(outer::univariate::UnivariateSkipOuterSumcheckOutput::from)?
    };

    // The reused cubic tail absorbed [Az(r), Bz(r), Cz(r)] before returning.
    let rho = squeeze_field(transcript, field_config)?;
    let inner_initial_claim = batched_product_claim(
        &outer.proof.tail.az_mle_claim,
        &outer.proof.tail.bz_mle_claim,
        &outer.proof.tail.cz_mle_claim,
        &rho,
        &field_config,
    );
    let batched_matrix = {
        let _scope = tracing::info_span!("spartan:bind_and_batch").entered();
        let row_factors = outer
            .row_binding
            .row_factors(matrices.num_row_vars(), field_config)?;
        matrices.structured().bind_prefix(&row_factors, &rho)?
    };
    let inner = {
        let _scope = tracing::info_span!("spartan:inner_sumcheck").entered();
        prove_inner_sumcheck(
            field_config,
            transcript,
            inner_initial_claim,
            assignment.evaluations,
            batched_matrix.evaluations,
            &mut UngrindedRoundBoundary,
        )?
    };

    let claim = ScaledMleEvaluationClaim::new(
        inner.point.into_boxed_slice(),
        inner.terminal_evaluations[0],
        inner.final_claim,
    );
    let proof = UnivariateSkipSpartanPiopProof {
        outer: outer.proof,
        inner: inner.proof,
    };
    Ok((proof, claim))
}

/// The raw-table Spartan prover for field-valued products: identical
/// statement, transcript, and proof to [`prove_spartan_piop_field_tables`]
/// with the delayed-Barrett reducer, on 16-byte residue tables.
fn prove_spartan_piop_raw_field<C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: R1csProductMles<Fp<2>>,
    assignment: DenseMultilinearExtension<Fp<2>>,
) -> Result<(SpartanPiopProof<Fp<2>>, ScaledMleEvaluationClaim<Fp<2>>), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
{
    {
        let _g = tracing::info_span!("sp:validate").entered();
        validate_prover_inputs(matrices, &products, &assignment)?;
    }
    absorb_statement(transcript, matrices, assignment_oracle_binding);

    let field_config = matrices.config();
    let ctx = crate::piop::spartan::raw_monty::field_context(field_config);
    let tau = (0..matrices.num_row_vars())
        .map(|_| squeeze_field(transcript, field_config))
        .collect::<Result<Vec<_>, _>>()?;
    let (eq_low, eq_high) = {
        let _g = tracing::info_span!("sp:eq").entered();
        make_equality_factors(&tau, field_config)?
    };
    let outer = {
        let _scope = tracing::info_span!("spartan:outer_sumcheck").entered();
        let factors = EqualityFactors::from_mles((eq_low, eq_high), field_config)?;
        outer::prove_outer_sumcheck(
            field_config,
            transcript,
            OuterClaim::Sum(field_config.zero()),
            &tau,
            products,
            Some(factors),
            &mut UngrindedRoundBoundary,
        )
        .map(OuterSumcheckOutput::from)?
    };

    // The outer prover absorbed [Az(r_x), Bz(r_x), Cz(r_x)] before returning.
    let rho = squeeze_field(transcript, field_config)?;
    let inner_initial_claim = batched_product_claim(
        &outer.proof.az_mle_claim,
        &outer.proof.bz_mle_claim,
        &outer.proof.cz_mle_claim,
        &rho,
        &field_config,
    );
    let inner = {
        let witness = RawWitness::Field(ctx.raw_vec(&assignment.evaluations));
        drop(assignment);
        let weights = NativeBinding::new(&ctx, matrices, rho)
            .bind_structured_rows(&RowFunctional::Point(&outer.eval_points))?;
        let values = witness;
        let _scope = tracing::info_span!("spartan:inner_sumcheck").entered();
        prove_inner_sumcheck(
            &ctx,
            transcript,
            inner_initial_claim,
            values,
            weights,
            &mut UngrindedRoundBoundary,
        )?
    };

    let claim = ScaledMleEvaluationClaim::new(
        inner.point.into_boxed_slice(),
        inner.terminal_evaluations[0],
        inner.final_claim,
    );
    let proof = SpartanPiopProof {
        outer: outer.proof,
        inner: inner.proof,
    };
    Ok((proof, claim))
}

/// The raw-table native-u64 univariate-skip Spartan prover: identical
/// statement, transcript, and proof to the delayed-Barrett generic driver.
/// The raw-table native univariate-skip prover core. Inputs must already have
/// passed [`validate_native_u32_prover_inputs`] or
/// [`validate_native_u32_prover_slices`].
pub(crate) fn prove_spartan_piop_raw_native_u64_with_skip_core<C, I: OuterRows>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    assignment_oracle_binding: &[u8; 32],
    products: I,
    witness: RawWitness<'_>,
    skip_vars: usize,
) -> Result<
    (
        UnivariateSkipSpartanPiopProof<Fp<2>>,
        ScaledMleEvaluationClaim<Fp<2>>,
    ),
    SpartanError,
>
where
    C: SpartanMatrixCoefficient<Fp<2>> + RawMontyCoefficient,
    field::FpCtx<2>: OuterArithmetic<I::AB, I::C>,
{
    let skip_vars = validate_univariate_skip_variables(skip_vars, matrices.num_row_vars())?;
    absorb_univariate_skip_statement(transcript, matrices, assignment_oracle_binding, skip_vars);

    let field_config = matrices.config();
    let ctx = crate::piop::spartan::raw_monty::field_context(field_config);
    let tail_vars = matrices.num_row_vars() - usize::from(skip_vars);
    let tau_tail = (0..tail_vars)
        .map(|_| squeeze_field(transcript, field_config))
        .collect::<Result<Vec<_>, _>>()?;
    let (eq_low, eq_high) = make_equality_factors_raw(&ctx, &tau_tail);

    let outer = {
        let _scope = tracing::info_span!("spartan:outer_univariate_skip").entered();
        let prepared = outer::prepare_univariate_skip(&ctx, skip_vars)?;
        outer::prove_outer_zerocheck_with_skip(
            &ctx,
            transcript,
            &prepared,
            &tau_tail,
            products,
            Some(factors_from_raw(&ctx, eq_low, eq_high)),
            &mut UngrindedRoundBoundary,
        )
        .map(outer::univariate::UnivariateSkipOuterSumcheckOutput::from)?
    };
    let (outer_proof, row_binding) = (outer.proof, outer.row_binding);

    let rho = squeeze_field(transcript, field_config)?;
    let inner_initial_claim = batched_product_claim(
        &outer_proof.tail.az_mle_claim,
        &outer_proof.tail.bz_mle_claim,
        &outer_proof.tail.cz_mle_claim,
        &rho,
        &field_config,
    );
    let inner = {
        let row_factors = row_binding.row_factors(matrices.num_row_vars(), field_config)?;
        let weights = NativeBinding::new(&ctx, matrices, rho)
            .bind_structured_rows(&RowFunctional::Prefix(&row_factors))?;
        let values = witness;
        let _scope = tracing::info_span!("spartan:inner_sumcheck").entered();
        prove_inner_sumcheck(
            &ctx,
            transcript,
            inner_initial_claim,
            values,
            weights,
            &mut UngrindedRoundBoundary,
        )?
    };

    let claim = ScaledMleEvaluationClaim::new(
        inner.point.into_boxed_slice(),
        inner.terminal_evaluations[0],
        inner.final_claim,
    );
    let proof = UnivariateSkipSpartanPiopProof {
        outer: outer_proof,
        inner: inner.proof,
    };
    Ok((proof, claim))
}

/// Verifies both sumchecks and returns the terminal scaled assignment claim
/// `D(r_y) * h(r_y) = final_claim`.
pub fn verify_spartan_proof<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
    proof: &SpartanPiopProof<F>,
) -> Result<ScaledMleEvaluationClaim<F>, SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    validate_proof(matrices, proof)?;
    absorb_statement(transcript, matrices, assignment_oracle_binding);

    let field_config = matrices.config();
    let tau = (0..matrices.num_row_vars())
        .map(|_| squeeze_field(transcript, field_config))
        .collect::<Result<Vec<_>, _>>()?;
    let outer = proof.outer.verify(
        transcript,
        F::zero_with_cfg(field_config),
        &tau,
        field_config,
    )?;

    // The outer verifier absorbed [Az(r_x), Bz(r_x), Cz(r_x)] before returning.
    let rho = squeeze_field(transcript, field_config)?;
    let inner_initial_claim = batched_product_claim(
        &outer.az_mle_claim,
        &outer.bz_mle_claim,
        &outer.cz_mle_claim,
        &rho,
        &field_config,
    );
    let (column_point, final_claim) = proof.inner.verify(
        transcript,
        inner_initial_claim,
        matrices.num_column_vars(),
        field_config,
    )?;
    let matrix_evaluation =
        matrices
            .structured()
            .evaluate_equality(&outer.eval_points, &rho, &column_point)?;

    Ok(ScaledMleEvaluationClaim::new(
        column_point.into_boxed_slice(),
        matrix_evaluation,
        final_claim,
    ))
}

/// Verifies the univariate-skip Spartan reduction and returns the
/// terminal scaled assignment claim `D(r_y) * h(r_y) = final_claim`.
///
/// The outer proof verifies only the known-zero prefix reduction and its
/// ordinary cubic tail. This function completes the matrix binding and inner
/// sumcheck exactly as the standard Spartan verifier does.
pub fn verify_spartan_univariate_skip_proof<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
    proof: &UnivariateSkipSpartanPiopProof<F>,
) -> Result<ScaledMleEvaluationClaim<F>, SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    validate_univariate_skip_proof(matrices, proof)?;
    let skip_vars = proof.outer.skip.skip_vars;
    absorb_univariate_skip_statement(transcript, matrices, assignment_oracle_binding, skip_vars);

    let field_config = matrices.config();
    let tail_vars = matrices.num_row_vars() - usize::from(skip_vars);
    let tau_tail = (0..tail_vars)
        .map(|_| squeeze_field(transcript, field_config))
        .collect::<Result<Vec<_>, _>>()?;
    let outer = proof
        .outer
        .verify(transcript, &tau_tail, matrices.num_row_vars(), field_config)?;

    let rho = squeeze_field(transcript, field_config)?;
    let inner_initial_claim = batched_product_claim(
        &outer.az_mle_claim,
        &outer.bz_mle_claim,
        &outer.cz_mle_claim,
        &rho,
        &field_config,
    );
    let (column_point, final_claim) = proof.inner.verify(
        transcript,
        inner_initial_claim,
        matrices.num_column_vars(),
        field_config,
    )?;
    // The matrices under the outer row functional at the inner point: the
    // succinct closed form on the block-selector relations, the sparse
    // evaluation elsewhere (one value either way).
    let prefix = outer
        .row_binding
        .prefix_weights(matrices.num_row_vars(), field_config)?;
    let functional = ProductRowFunctional {
        skip_vars: usize::from(outer.row_binding.skip_vars),
        prefix: &prefix,
        tail_point: &outer.row_binding.tail_point,
    };
    let matrix_evaluation =
        matrices
            .structured()
            .evaluate_product(&functional, &rho, &column_point)?;

    Ok(ScaledMleEvaluationClaim::new(
        column_point.into_boxed_slice(),
        matrix_evaluation,
        final_claim,
    ))
}

/// Proves the PIOP while binding the complete assignment table itself.
///
/// This is the sound native path until a hiding PCS commitment is available.
/// The digest is binding, not hiding; callers who need witness privacy should
/// use [`prove_spartan_piop`] with a canonical 32-byte binding of the PCS
/// commitment.
pub fn prove_spartan_nonsuccinct<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    products: R1csProductMles<F>,
    assignment: DenseMultilinearExtension<F>,
) -> Result<(SpartanPiopProof<F>, ScaledMleEvaluationClaim<F>), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    validate_prover_inputs(matrices, &products, &assignment)?;
    let assignment_binding =
        nonsuccinct_assignment_digest(matrices, &assignment, matrices.config())?;
    prove_spartan_piop(
        transcript,
        matrices,
        &assignment_binding,
        products,
        assignment,
    )
}

/// Verifies the PIOP, checks that `mle_claim` is the transcript-derived claim,
/// and discharges it against a complete assignment table.
///
/// This is intentionally nonsuccinct. It is the native integration seam to be
/// replaced by the BitZ PCS opening protocol later.
pub fn verify_spartan_with_mle_claim<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    proof: &SpartanPiopProof<F>,
    mle_claim: &ScaledMleEvaluationClaim<F>,
    assignment: &DenseMultilinearExtension<F>,
) -> Result<(), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    validate_assignment(matrices, assignment)?;
    let assignment_binding =
        nonsuccinct_assignment_digest(matrices, assignment, matrices.config())?;
    let expected_claim = verify_spartan_proof(transcript, matrices, &assignment_binding, proof)?;
    if mle_claim != &expected_claim {
        return Err(SpartanError::InvalidMleClaim);
    }
    mle_claim.nonsuccinct_verify(assignment, matrices.config())?;
    Ok(())
}

fn absorb_statement<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
) where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    absorb_spartan_message(transcript, b"protocol", SPARTAN_PIOP_DOMAIN);
    absorb_spartan_message(
        transcript,
        b"field-modulus",
        matrices.field_modulus_encoding(),
    );
    absorb_spartan_message(transcript, b"matrix-statement", matrices.digest());
    absorb_spartan_message(
        transcript,
        SPARTAN_ASSIGNMENT_ORACLE_DOMAIN,
        assignment_oracle_binding,
    );
}

fn validate_univariate_skip_variables(
    skip_vars: usize,
    row_vars: usize,
) -> Result<u8, SpartanError> {
    if !(1..=4).contains(&skip_vars) {
        return Err(SpartanError::InvalidUnivariateSkipVariables { skip_vars });
    }
    if skip_vars > row_vars {
        return Err(SpartanError::UnivariateSkipExceedsRowVariables {
            skip_vars,
            row_vars,
        });
    }
    Ok(skip_vars as u8)
}

fn absorb_univariate_skip_statement<F, C>(
    transcript: &mut impl Transcript,
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment_oracle_binding: &[u8; 32],
    skip_vars: u8,
) where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    absorb_spartan_message(transcript, b"protocol", SPARTAN_UNIVARIATE_SKIP_PIOP_DOMAIN);
    absorb_spartan_message(
        transcript,
        b"field-modulus",
        matrices.field_modulus_encoding(),
    );
    absorb_spartan_message(transcript, b"matrix-statement", matrices.digest());
    absorb_spartan_message(
        transcript,
        SPARTAN_ASSIGNMENT_ORACLE_DOMAIN,
        assignment_oracle_binding,
    );
    absorb_spartan_message(transcript, b"univariate-skip-vars", &[skip_vars]);
}

fn validate_prover_inputs<F, C>(
    matrices: &PreparedConstraintMatrices<F, C>,
    products: &R1csProductMles<F>,
    assignment: &DenseMultilinearExtension<F>,
) -> Result<(), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    let row_vars = matrices.num_row_vars();
    if products.az.num_vars != row_vars
        || products.bz.num_vars != row_vars
        || products.cz.num_vars != row_vars
    {
        return Err(SpartanError::InvalidProductDimensions);
    }

    for mle in [&products.az, &products.bz, &products.cz] {
        validate_mle_shape(mle)?;
        validate_elements_field(&mle.evaluations, matrices)?;
    }
    validate_assignment(matrices, assignment)
}

fn validate_native_u32_prover_inputs<C>(
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    products: &R1csProductMles<u64>,
    assignment: &DenseMultilinearExtension<u64>,
) -> Result<(), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>>,
{
    let row_vars = matrices.num_row_vars();
    if products.az.num_vars != row_vars
        || products.bz.num_vars != row_vars
        || products.cz.num_vars != row_vars
    {
        return Err(SpartanError::InvalidProductDimensions);
    }
    for mle in [&products.az, &products.bz, &products.cz] {
        validate_mle_shape(mle)?;
    }
    if products
        .az
        .evaluations
        .iter()
        .chain(&products.bz.evaluations)
        .any(|&value| value > u64::from(u32::MAX))
    {
        return Err(SumcheckError::NativeMultiplicandOutOfRange.into());
    }

    if assignment.num_vars != matrices.num_column_vars() {
        return Err(SpartanError::InvalidAssignmentDimensions);
    }
    validate_mle_shape(assignment)?;
    if assignment.evaluations.first() != Some(&1) {
        return Err(SpartanMatrixError::InvalidAssignmentConstant.into());
    }
    if assignment.evaluations[matrices.matrices().column_count()..]
        .iter()
        .any(|&value| value != 0)
    {
        return Err(SpartanError::InvalidAssignmentPadding);
    }
    Ok(())
}

/// [`validate_native_u32_prover_inputs`] for borrowed tables: the products
/// span the padded row domain, `Az`/`Bz` are 32-bit wide, and the assignment
/// is either the complete padded column table or exactly the logical
/// columns (its padding then being implicitly zero).
fn validate_native_u32_prover_slices<C>(
    matrices: &PreparedConstraintMatrices<Fp<2>, C>,
    products: NativeProducts<'_>,
    assignment: &[u64],
) -> Result<(), SpartanError>
where
    C: SpartanMatrixCoefficient<Fp<2>>,
{
    let rows = 1usize << matrices.num_row_vars();
    if products.az.len() != rows || products.bz.len() != rows || products.cz.len() != rows {
        return Err(SpartanError::InvalidProductDimensions);
    }
    if products
        .az
        .iter()
        .chain(products.bz)
        .any(|&value| value > u64::from(u32::MAX))
    {
        return Err(SumcheckError::NativeMultiplicandOutOfRange.into());
    }

    let domain = 1usize << matrices.num_column_vars();
    let column_count = matrices.matrices().column_count();
    if assignment.len() != column_count && assignment.len() != domain {
        return Err(SpartanError::InvalidAssignmentDimensions);
    }
    if assignment.first() != Some(&1) {
        return Err(SpartanMatrixError::InvalidAssignmentConstant.into());
    }
    if assignment.len() > column_count && assignment[column_count..].iter().any(|&value| value != 0)
    {
        return Err(SpartanError::InvalidAssignmentPadding);
    }
    Ok(())
}

#[cfg(test)]
fn project_native_mle(
    mle: DenseMultilinearExtension<u64>,
    field_config: &field::FpCtx<2>,
) -> DenseMultilinearExtension<Fp<2>> {
    DenseMultilinearExtension {
        evaluations: mle
            .evaluations
            .into_iter()
            .map(|value| Fp::<2>::from_with_cfg(value, field_config))
            .collect(),
        num_vars: mle.num_vars,
    }
}

#[cfg(test)]
fn project_native_products(
    products: R1csProductMles<u64>,
    field_config: &field::FpCtx<2>,
) -> R1csProductMles<Fp<2>> {
    R1csProductMles {
        az: project_native_mle(products.az, field_config),
        bz: project_native_mle(products.bz, field_config),
        cz: project_native_mle(products.cz, field_config),
    }
}

fn validate_assignment<F, C>(
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment: &DenseMultilinearExtension<F>,
) -> Result<(), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    if assignment.num_vars != matrices.num_column_vars() {
        return Err(SpartanError::InvalidAssignmentDimensions);
    }
    validate_mle_shape(assignment)?;
    validate_elements_field(&assignment.evaluations, matrices)?;

    let one = F::one_with_cfg(matrices.config());
    if assignment.evaluations.first() != Some(&one) {
        return Err(SpartanMatrixError::InvalidAssignmentConstant.into());
    }
    let zero = F::zero_with_cfg(matrices.config());
    let padding = &assignment.evaluations[matrices.matrices().column_count()..];
    #[cfg(feature = "parallel")]
    let has_nonzero = if padding.len() >= (1 << 14) && rayon::current_num_threads() > 1 {
        padding.par_iter().any(|value| value != &zero)
    } else {
        padding.iter().any(|value| value != &zero)
    };
    #[cfg(not(feature = "parallel"))]
    let has_nonzero = padding.iter().any(|value| value != &zero);
    if has_nonzero {
        return Err(SpartanError::InvalidAssignmentPadding);
    }
    Ok(())
}

fn nonsuccinct_assignment_digest<F, C>(
    matrices: &PreparedConstraintMatrices<F, C>,
    assignment: &DenseMultilinearExtension<F>,
    field_config: &F::Config,
) -> Result<[u8; 32], SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    // Keep this helper independently defensive because it is the binding used
    // by both prover and verifier before the first challenge.
    validate_assignment(matrices, assignment)?;

    let mut hasher = Hasher::new();
    hasher.update(NONSUCCINCT_ASSIGNMENT_DIGEST_DOMAIN);
    hash_binding_bytes(&mut hasher, matrices.field_modulus_encoding())?;
    hash_binding_usize(&mut hasher, matrices.matrices().column_count())?;
    hash_binding_usize(&mut hasher, assignment.num_vars)?;
    hash_binding_usize(&mut hasher, assignment.evaluations.len())?;
    for value in &assignment.evaluations {
        hash_binding_bytes(&mut hasher, &value.canonical_element_encoding(field_config))?;
    }
    Ok(*hasher.finalize().as_bytes())
}

fn hash_binding_bytes(hasher: &mut Hasher, bytes: &[u8]) -> Result<(), SpartanError> {
    hash_binding_usize(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}

fn hash_binding_usize(hasher: &mut Hasher, value: usize) -> Result<(), SpartanError> {
    let value = u64::try_from(value).map_err(|_| SpartanMatrixError::DomainTooLarge)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

fn validate_proof<F, C>(
    matrices: &PreparedConstraintMatrices<F, C>,
    proof: &SpartanPiopProof<F>,
) -> Result<(), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    if proof.outer.sumcheck.round_polynomials.len() != matrices.num_row_vars() {
        return Err(SumcheckError::InvalidRoundCount {
            expected: matrices.num_row_vars(),
            actual: proof.outer.sumcheck.round_polynomials.len(),
        }
        .into());
    }
    if proof.inner.round_polynomials.len() != matrices.num_column_vars() {
        return Err(SumcheckError::InvalidRoundCount {
            expected: matrices.num_column_vars(),
            actual: proof.inner.round_polynomials.len(),
        }
        .into());
    }

    for round in &proof.outer.sumcheck.round_polynomials {
        validate_elements_field(round, matrices)?;
    }
    validate_elements_field(
        &[
            proof.outer.az_mle_claim.clone(),
            proof.outer.bz_mle_claim.clone(),
            proof.outer.cz_mle_claim.clone(),
        ],
        matrices,
    )?;
    for round in &proof.inner.round_polynomials {
        validate_elements_field(round, matrices)?;
    }
    Ok(())
}

fn validate_univariate_skip_proof<F, C>(
    matrices: &PreparedConstraintMatrices<F, C>,
    proof: &UnivariateSkipSpartanPiopProof<F>,
) -> Result<(), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    let skip_vars = usize::from(proof.outer.skip.skip_vars);
    validate_univariate_skip_variables(skip_vars, matrices.num_row_vars())?;

    let expected_finite = (1usize << skip_vars) - 2;
    let actual_finite = proof.outer.skip.finite_q_evaluations.len();
    if actual_finite != expected_finite {
        return Err(SpartanError::InvalidUnivariateSkipMessageLength {
            expected: expected_finite,
            actual: actual_finite,
        });
    }

    let expected_tail_rounds = matrices.num_row_vars() - skip_vars;
    let actual_tail_rounds = proof.outer.tail.sumcheck.round_polynomials.len();
    if actual_tail_rounds != expected_tail_rounds {
        return Err(SumcheckError::InvalidRoundCount {
            expected: expected_tail_rounds,
            actual: actual_tail_rounds,
        }
        .into());
    }
    if proof.inner.round_polynomials.len() != matrices.num_column_vars() {
        return Err(SumcheckError::InvalidRoundCount {
            expected: matrices.num_column_vars(),
            actual: proof.inner.round_polynomials.len(),
        }
        .into());
    }

    validate_elements_field(&proof.outer.skip.finite_q_evaluations, matrices)?;
    validate_elements_field(
        std::slice::from_ref(&proof.outer.skip.q_at_infinity),
        matrices,
    )?;
    for round in &proof.outer.tail.sumcheck.round_polynomials {
        validate_elements_field(round, matrices)?;
    }
    validate_elements_field(
        &[
            proof.outer.tail.az_mle_claim.clone(),
            proof.outer.tail.bz_mle_claim.clone(),
            proof.outer.tail.cz_mle_claim.clone(),
        ],
        matrices,
    )?;
    for round in &proof.inner.round_polynomials {
        validate_elements_field(round, matrices)?;
    }
    Ok(())
}

fn validate_mle_shape<F>(mle: &DenseMultilinearExtension<F>) -> Result<(), SpartanError> {
    let expected = 1usize
        .checked_shl(u32::try_from(mle.num_vars).map_err(|_| SpartanError::InvalidMleShape)?)
        .ok_or(SpartanError::InvalidMleShape)?;
    if mle.evaluations.len() != expected {
        return Err(SpartanError::InvalidMleShape);
    }
    Ok(())
}

fn validate_elements_field<F, C>(
    values: &[F],
    matrices: &PreparedConstraintMatrices<F, C>,
) -> Result<(), SpartanError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    let sweep = |values: &[F]| -> bool {
        values.iter().all(|value| {
            value
                .validate_element(matrices.field_modulus_encoding())
                .is_ok()
        })
    };
    #[cfg(feature = "parallel")]
    let all_valid = if values.len() >= (1 << 14) && rayon::current_num_threads() > 1 {
        values.par_chunks(1 << 12).all(|chunk| sweep(chunk))
    } else {
        sweep(values)
    };
    #[cfg(not(feature = "parallel"))]
    let all_valid = sweep(values);
    if !all_valid {
        // Sequential re-scan for the canonical first error.
        for value in values {
            value
                .validate_element(matrices.field_modulus_encoding())
                .map_err(SpartanMatrixError::from)?;
        }
        unreachable!("parallel validation rejected but the canonical scan found no error");
    }
    Ok(())
}

fn batched_product_claim<F>(az: &F, bz: &F, cz: &F, rho: &F, field_config: &F::Config) -> F
where
    F: SpartanField,
{
    let rho_squared = field_config.mul(&(rho.clone()), &(rho));
    let mut claim = az.clone();
    claim = field_config.add(&(claim), &(&(field_config.mul(&(rho.clone()), &(bz)))));
    claim = field_config.add(&(claim), &(&(field_config.mul(&(rho_squared), &(cz)))));
    claim
}

#[cfg(test)]
mod tests {

    use crate::transcript::{Blake3Transcript, traits::Transcript};

    use super::*;
    use crate::piop::spartan::matrix::{
        ConstraintMatrices, build_assignment_mle, build_product_mles,
    };
    use crate::piop::spartan::u32_mul::{prepare_u32_mul_relation, project_u32_mul_native_witness};

    const Q100: u128 = (1_u128 << 100) - 15;
    const ROWS: usize = 5;
    const COLUMNS: usize = 7;

    fn config(modulus: u128) -> <Fp<2> as crate::piop::spartan::SpartanField>::Config {
        Fp::<2>::make_cfg(&Uint::from(modulus)).expect("odd prime test modulus")
    }

    fn field(value: u128, config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config) -> Fp<2> {
        Fp::<2>::from_with_cfg(value, config)
    }

    fn multiply(
        matrix: &CscMatrix<Box<[Fp<2>]>>,
        assignment: &[Fp<2>],
        config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> Vec<Fp<2>> {
        assert_eq!(assignment.len(), matrix.column_count());
        let mut products = vec![Fp::<2>::zero_with_cfg(config); matrix.row_count()];
        for (column, entries) in matrix.columns().enumerate() {
            for (row, coefficient) in entries {
                products[row] = config.add(
                    &(products[row]),
                    &(&(config.mul(&(coefficient.clone()), &(&assignment[column])))),
                );
            }
        }
        products
    }

    fn fixture(
        config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> (
        PreparedConstraintMatrices<Fp<2>>,
        R1csProductMles<Fp<2>>,
        DenseMultilinearExtension<Fp<2>>,
    ) {
        let assignment_values: Vec<_> = (0..COLUMNS)
            .map(|column| field(if column == 0 { 1 } else { (column + 1) as u128 }, config))
            .collect();
        let a = CscMatrix::try_from_rows(
            COLUMNS,
            (0..ROWS)
                .map(|row| {
                    vec![
                        (0, field((row + 2) as u128, config)),
                        (row + 1, field((2 * row + 3) as u128, config)),
                    ]
                })
                .collect(),
        )
        .unwrap();
        let b = CscMatrix::try_from_rows(
            COLUMNS,
            (0..ROWS)
                .map(|row| {
                    vec![
                        (0, field((row + 5) as u128, config)),
                        (row + 2, field((3 * row + 7) as u128, config)),
                    ]
                })
                .collect(),
        )
        .unwrap();
        let az = multiply(&a, &assignment_values, config);
        let bz = multiply(&b, &assignment_values, config);
        let c = CscMatrix::try_from_rows(
            COLUMNS,
            az.iter()
                .zip(&bz)
                .map(|(az, bz)| vec![(0, config.mul(&(az.clone()), &(bz)))])
                .collect(),
        )
        .unwrap();
        let cz = multiply(&c, &assignment_values, config);
        assert!(
            az.iter()
                .zip(&bz)
                .zip(&cz)
                .all(|((az, bz), cz)| config.mul(&(az.clone()), &(bz)) == *cz)
        );

        let prepared =
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), config)
                .unwrap();
        let products = build_product_mles(&az, &bz, &cz, ROWS, config).unwrap();
        let assignment = build_assignment_mle(&assignment_values, COLUMNS, config).unwrap();
        (prepared, products, assignment)
    }

    fn round_trip(modulus: u128) {
        let config = config(modulus);
        let (matrices, products, assignment) = fixture(&config);
        let assignment_binding =
            nonsuccinct_assignment_digest(&matrices, &assignment, &config).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        let (proof, claim) = prove_spartan_nonsuccinct(
            &mut prover_transcript,
            &matrices,
            products,
            assignment.clone(),
        )
        .unwrap();
        assert_eq!(proof.outer.sumcheck.round_polynomials.len(), 3);
        assert_eq!(proof.inner.round_polynomials.len(), 3);

        let mut verifier_transcript = Blake3Transcript::new();
        let expected_claim = verify_spartan_proof(
            &mut verifier_transcript,
            &matrices,
            &assignment_binding,
            &proof,
        )
        .unwrap();
        assert_eq!(claim, expected_claim);
        claim.nonsuccinct_verify(&assignment, &config).unwrap();

        // Equal continuation proves that every challenge and absorption stayed
        // in lockstep across the prover and verifier implementations.
        let prover_continuation: u128 = prover_transcript.get_challenge();
        let verifier_continuation: u128 = verifier_transcript.get_challenge();
        assert_eq!(prover_continuation, verifier_continuation);

        let mut complete_transcript = Blake3Transcript::new();
        verify_spartan_with_mle_claim(
            &mut complete_transcript,
            &matrices,
            &proof,
            &claim,
            &assignment,
        )
        .unwrap();
    }

    fn delayed_reduction_matches_reference(modulus: u128) {
        let config = config(modulus);
        let (matrices, products, assignment) = fixture(&config);
        let assignment_binding =
            nonsuccinct_assignment_digest(&matrices, &assignment, &config).unwrap();
        let mut reference = None;

        for delayed in [false, true] {
            let mut transcript = Blake3Transcript::new();
            let (proof, claim) = if delayed {
                prove_spartan_piop_field(
                    &mut transcript,
                    &matrices,
                    &assignment_binding,
                    products.clone(),
                    assignment.clone(),
                )
            } else {
                prove_spartan_piop(
                    &mut transcript,
                    &matrices,
                    &assignment_binding,
                    products.clone(),
                    assignment.clone(),
                )
            }
            .unwrap();
            let continuation: u128 = transcript.get_challenge();

            if let Some((reference_proof, reference_claim, reference_continuation)) = &reference {
                assert_eq!(&proof, reference_proof);
                assert_eq!(&claim, reference_claim);
                assert_eq!(&continuation, reference_continuation);
            } else {
                reference = Some((proof.clone(), claim.clone(), continuation));
            }

            let mut verifier_transcript = Blake3Transcript::new();
            let verified = verify_spartan_proof(
                &mut verifier_transcript,
                &matrices,
                &assignment_binding,
                &proof,
            )
            .unwrap();
            assert_eq!(claim, verified);
        }
    }

    fn native_u32_delayed_reduction_matches_reference(modulus: u128) {
        let config = config(modulus);
        let inputs = [
            (0, u32::MAX),
            (1, 1),
            (u32::MAX, u32::MAX),
            (0x8000_0000, 2),
            (17, 19),
        ];
        let witness = MulWitness::<u32>::from_inputs(&inputs).unwrap();
        let matrices = prepare_u32_mul_relation(*witness.layout(), &config).unwrap();
        let native = project_u32_mul_native_witness(&witness);
        let (assignment, products) = native.into_parts();
        let assignment_binding = [0xA5; 32];
        let mut reference = None;

        for delayed in [false, true] {
            let mut transcript = Blake3Transcript::new();
            let (proof, claim) = if delayed {
                prove_spartan_piop_u32_native(
                    &mut transcript,
                    &matrices,
                    &assignment_binding,
                    products.clone(),
                    assignment.clone(),
                )
            } else {
                prove_spartan_piop(
                    &mut transcript,
                    &matrices,
                    &assignment_binding,
                    project_native_products(products.clone(), &config),
                    project_native_mle(assignment.clone(), &config),
                )
            }
            .unwrap();
            let continuation: u128 = transcript.get_challenge();

            if let Some((reference_proof, reference_claim, reference_continuation)) = &reference {
                assert_eq!(&proof, reference_proof);
                assert_eq!(&claim, reference_claim);
                assert_eq!(&continuation, reference_continuation);
            } else {
                reference = Some((proof.clone(), claim.clone(), continuation));
            }

            let mut verifier_transcript = Blake3Transcript::new();
            let verified = verify_spartan_proof(
                &mut verifier_transcript,
                &matrices,
                &assignment_binding,
                &proof,
            )
            .unwrap();
            assert_eq!(claim, verified);
        }
    }

    fn univariate_skip_round_trip(modulus: u128) {
        let field_config = config(modulus);
        let (matrices, products, assignment) = fixture(&field_config);
        let assignment_binding =
            nonsuccinct_assignment_digest(&matrices, &assignment, &field_config).unwrap();

        for skip_vars in 1..=matrices.num_row_vars() {
            let mut prover_transcript = Blake3Transcript::new();
            let (proof, claim) = prove_spartan_piop_with_univariate_skip(
                &mut prover_transcript,
                &matrices,
                &assignment_binding,
                products.clone(),
                assignment.clone(),
                skip_vars,
            )
            .unwrap();

            assert_eq!(usize::from(proof.outer.skip.skip_vars), skip_vars);
            assert_eq!(
                proof.outer.skip.finite_q_evaluations.len(),
                (1usize << skip_vars) - 2
            );
            assert_eq!(
                proof.outer.tail.sumcheck.round_polynomials.len(),
                matrices.num_row_vars() - skip_vars
            );
            let outer_fields = proof.outer.skip.finite_q_evaluations.len()
                + 1
                + 4 * proof.outer.tail.sumcheck.round_polynomials.len()
                + 3;
            assert_eq!(
                outer_fields,
                4 * matrices.num_row_vars() - 4 * skip_vars + (1usize << skip_vars) + 2
            );

            let mut verifier_transcript = Blake3Transcript::new();
            let verified = verify_spartan_univariate_skip_proof(
                &mut verifier_transcript,
                &matrices,
                &assignment_binding,
                &proof,
            )
            .unwrap();
            assert_eq!(claim, verified);
            claim
                .nonsuccinct_verify(&assignment, &field_config)
                .unwrap();
            assert_eq!(
                prover_transcript.get_challenge::<u128>(),
                verifier_transcript.get_challenge::<u128>()
            );
        }
    }

    fn native_univariate_skip_round_trip(modulus: u128) {
        let field_config = config(modulus);
        let inputs = (0..16)
            .map(|index| match index % 4 {
                0 => (0, u32::MAX),
                1 => (1, 1),
                2 => (u32::MAX, u32::MAX),
                _ => (0x8000_0000 + index as u32, 17 + index as u32),
            })
            .collect::<Vec<_>>();
        let witness = MulWitness::<u32>::from_inputs(&inputs).unwrap();
        let matrices = prepare_u32_mul_relation(*witness.layout(), &field_config).unwrap();
        assert!(matrices.num_row_vars() >= 4);
        let native = project_u32_mul_native_witness(&witness);
        let (assignment, products) = native.into_parts();
        let assignment_binding = [0x6D; 32];

        for skip_vars in 1..=4 {
            let mut generic_transcript = Blake3Transcript::new();
            let (generic_proof, generic_claim) = prove_spartan_piop_with_univariate_skip(
                &mut generic_transcript,
                &matrices,
                &assignment_binding,
                project_native_products(products.clone(), &field_config),
                project_native_mle(assignment.clone(), &field_config),
                skip_vars,
            )
            .unwrap();

            let mut prover_transcript = Blake3Transcript::new();
            let (proof, claim) = prove_spartan_piop_u32_native_with_univariate_skip(
                &mut prover_transcript,
                &matrices,
                &assignment_binding,
                products.clone(),
                assignment.clone(),
                skip_vars,
            )
            .unwrap();
            assert_eq!(proof, generic_proof);
            assert_eq!(claim, generic_claim);
            if skip_vars == matrices.num_row_vars() {
                assert!(proof.outer.tail.sumcheck.round_polynomials.is_empty());
            }

            let mut verifier_transcript = Blake3Transcript::new();
            let verified = verify_spartan_univariate_skip_proof(
                &mut verifier_transcript,
                &matrices,
                &assignment_binding,
                &proof,
            )
            .unwrap();
            assert_eq!(claim, verified);
            let generic_continuation = generic_transcript.get_challenge::<u128>();
            let native_continuation = prover_transcript.get_challenge::<u128>();
            let verifier_continuation = verifier_transcript.get_challenge::<u128>();
            assert_eq!(native_continuation, verifier_continuation);
            assert_eq!(generic_continuation, verifier_continuation);
        }
    }

    #[test]
    fn piop_is_generic_across_runtime_prime_configurations() {
        round_trip(Q100);
        round_trip((1_u128 << 127) - 1);
    }

    #[test]
    fn delayed_reduction_is_proof_and_transcript_exact() {
        delayed_reduction_matches_reference(Q100);
        delayed_reduction_matches_reference((1_u128 << 127) - 1);
    }

    #[test]
    fn native_u32_first_round_is_proof_and_transcript_exact_at_boundaries() {
        native_u32_delayed_reduction_matches_reference(Q100);
        native_u32_delayed_reduction_matches_reference((1_u128 << 127) - 1);
    }

    #[test]
    fn univariate_skip_piop_is_generic_and_transcript_exact() {
        univariate_skip_round_trip(Q100);
        univariate_skip_round_trip((1_u128 << 127) - 1);
    }

    #[test]
    fn native_u32_univariate_skip_piop_supports_every_k() {
        native_univariate_skip_round_trip(Q100);
        native_univariate_skip_round_trip((1_u128 << 127) - 1);
    }

    #[test]
    fn univariate_skip_rejects_malformed_and_tampered_proofs() {
        let field_config = config(Q100);
        let (matrices, products, assignment) = fixture(&field_config);
        let assignment_binding =
            nonsuccinct_assignment_digest(&matrices, &assignment, &field_config).unwrap();
        let mut prover_transcript = Blake3Transcript::new();
        let (proof, _) = prove_spartan_piop_with_univariate_skip(
            &mut prover_transcript,
            &matrices,
            &assignment_binding,
            products.clone(),
            assignment.clone(),
            2,
        )
        .unwrap();
        let one = field(1, &field_config);

        for coordinate in 0..proof.outer.skip.finite_q_evaluations.len() {
            let mut tampered = proof.clone();
            tampered.outer.skip.finite_q_evaluations[coordinate] = field_config.add(
                &(tampered.outer.skip.finite_q_evaluations[coordinate]),
                &(&one),
            );
            assert!(
                verify_spartan_univariate_skip_proof(
                    &mut Blake3Transcript::new(),
                    &matrices,
                    &assignment_binding,
                    &tampered,
                )
                .is_err(),
                "finite coordinate {coordinate} must be binding"
            );
        }

        let mut tampered_infinity = proof.clone();
        tampered_infinity.outer.skip.q_at_infinity =
            field_config.add(&(tampered_infinity.outer.skip.q_at_infinity), &(&one));
        assert!(
            verify_spartan_univariate_skip_proof(
                &mut Blake3Transcript::new(),
                &matrices,
                &assignment_binding,
                &tampered_infinity,
            )
            .is_err()
        );

        let mut tampered_tail = proof.clone();
        tampered_tail.outer.tail.sumcheck.round_polynomials[0][0] = field_config.add(
            &(tampered_tail.outer.tail.sumcheck.round_polynomials[0][0]),
            &(&one),
        );
        assert!(
            verify_spartan_univariate_skip_proof(
                &mut Blake3Transcript::new(),
                &matrices,
                &assignment_binding,
                &tampered_tail,
            )
            .is_err()
        );

        for terminal in 0..3 {
            let mut tampered_terminal = proof.clone();
            match terminal {
                0 => {
                    tampered_terminal.outer.tail.az_mle_claim =
                        field_config.add(&(tampered_terminal.outer.tail.az_mle_claim), &(&one))
                }
                1 => {
                    tampered_terminal.outer.tail.bz_mle_claim =
                        field_config.add(&(tampered_terminal.outer.tail.bz_mle_claim), &(&one))
                }
                2 => {
                    tampered_terminal.outer.tail.cz_mle_claim =
                        field_config.add(&(tampered_terminal.outer.tail.cz_mle_claim), &(&one))
                }
                _ => unreachable!(),
            }
            assert!(
                verify_spartan_univariate_skip_proof(
                    &mut Blake3Transcript::new(),
                    &matrices,
                    &assignment_binding,
                    &tampered_terminal,
                )
                .is_err(),
                "terminal claim {terminal} must be binding"
            );
        }

        let mut malformed_k = proof.clone();
        malformed_k.outer.skip.skip_vars = 0;
        let mut malformed_transcript = Blake3Transcript::new();
        let mut untouched_transcript = malformed_transcript.clone();
        assert_eq!(
            verify_spartan_univariate_skip_proof(
                &mut malformed_transcript,
                &matrices,
                &assignment_binding,
                &malformed_k,
            ),
            Err(SpartanError::InvalidUnivariateSkipVariables { skip_vars: 0 })
        );
        assert_eq!(
            malformed_transcript.get_challenge::<u128>(),
            untouched_transcript.get_challenge::<u128>()
        );

        let mut malformed_message = proof.clone();
        malformed_message.outer.skip.finite_q_evaluations = malformed_message
            .outer
            .skip
            .finite_q_evaluations
            .iter()
            .take(1)
            .cloned()
            .collect();
        let mut malformed_transcript = Blake3Transcript::new();
        let mut untouched_transcript = malformed_transcript.clone();
        assert_eq!(
            verify_spartan_univariate_skip_proof(
                &mut malformed_transcript,
                &matrices,
                &assignment_binding,
                &malformed_message,
            ),
            Err(SpartanError::InvalidUnivariateSkipMessageLength {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            malformed_transcript.get_challenge::<u128>(),
            untouched_transcript.get_challenge::<u128>()
        );

        let mut malformed_tail = proof.clone();
        malformed_tail.outer.tail.sumcheck.round_polynomials.pop();
        let mut malformed_transcript = Blake3Transcript::new();
        let mut untouched_transcript = malformed_transcript.clone();
        assert_eq!(
            verify_spartan_univariate_skip_proof(
                &mut malformed_transcript,
                &matrices,
                &assignment_binding,
                &malformed_tail,
            ),
            Err(SpartanError::Sumcheck(SumcheckError::InvalidRoundCount {
                expected: 1,
                actual: 0,
            }))
        );
        assert_eq!(
            malformed_transcript.get_challenge::<u128>(),
            untouched_transcript.get_challenge::<u128>()
        );

        let mut malformed_inner = proof.clone();
        malformed_inner.inner.round_polynomials.pop();
        let mut malformed_transcript = Blake3Transcript::new();
        let mut untouched_transcript = malformed_transcript.clone();
        assert_eq!(
            verify_spartan_univariate_skip_proof(
                &mut malformed_transcript,
                &matrices,
                &assignment_binding,
                &malformed_inner,
            ),
            Err(SpartanError::Sumcheck(SumcheckError::InvalidRoundCount {
                expected: matrices.num_column_vars(),
                actual: matrices.num_column_vars() - 1,
            }))
        );
        assert_eq!(
            malformed_transcript.get_challenge::<u128>(),
            untouched_transcript.get_challenge::<u128>()
        );

        let mut cross_k = proof.clone();
        cross_k.outer.skip.skip_vars = 3;
        let mut malformed_transcript = Blake3Transcript::new();
        let mut untouched_transcript = malformed_transcript.clone();
        assert_eq!(
            verify_spartan_univariate_skip_proof(
                &mut malformed_transcript,
                &matrices,
                &assignment_binding,
                &cross_k,
            ),
            Err(SpartanError::InvalidUnivariateSkipMessageLength {
                expected: 6,
                actual: 2,
            })
        );
        assert_eq!(
            malformed_transcript.get_challenge::<u128>(),
            untouched_transcript.get_challenge::<u128>()
        );

        let other_config = config((1_u128 << 127) - 1);
        let mut foreign_field = proof.clone();
        foreign_field.outer.skip.finite_q_evaluations[0] =
            crate::piop::spartan::noncanonical_test_value(&matrices.config());
        let mut malformed_transcript = Blake3Transcript::new();
        let mut untouched_transcript = malformed_transcript.clone();
        assert_eq!(
            verify_spartan_univariate_skip_proof(
                &mut malformed_transcript,
                &matrices,
                &assignment_binding,
                &foreign_field,
            ),
            Err(SpartanError::Matrix(
                SpartanMatrixError::InvalidFieldConfiguration(
                    crate::piop::spartan::SpartanFieldError::NonCanonicalElement
                )
            ))
        );
        assert_eq!(
            malformed_transcript.get_challenge::<u128>(),
            untouched_transcript.get_challenge::<u128>()
        );

        for skip_vars in [0, 4, 5] {
            let mut actual = Blake3Transcript::new();
            let mut untouched = actual.clone();
            assert!(
                prove_spartan_piop_with_univariate_skip(
                    &mut actual,
                    &matrices,
                    &assignment_binding,
                    products.clone(),
                    assignment.clone(),
                    skip_vars,
                )
                .is_err()
            );
            assert_eq!(
                actual.get_challenge::<u128>(),
                untouched.get_challenge::<u128>()
            );
        }
    }

    #[test]
    fn univariate_skip_domain_and_k_are_bound_before_the_first_challenge() {
        let field_config = config(Q100);
        let (matrices, _, _) = fixture(&field_config);
        let assignment_binding = [0xA7; 32];

        let mut standard = Blake3Transcript::new();
        absorb_statement(&mut standard, &matrices, &assignment_binding);
        let standard_challenge = squeeze_field::<Fp<2>, _>(&mut standard, &field_config).unwrap();

        let mut skip_k2 = Blake3Transcript::new();
        absorb_univariate_skip_statement(&mut skip_k2, &matrices, &assignment_binding, 2);
        let skip_k2_challenge = squeeze_field::<Fp<2>, _>(&mut skip_k2, &field_config).unwrap();

        let mut skip_k3 = Blake3Transcript::new();
        absorb_univariate_skip_statement(&mut skip_k3, &matrices, &assignment_binding, 3);
        let skip_k3_challenge = squeeze_field::<Fp<2>, _>(&mut skip_k3, &field_config).unwrap();

        assert_ne!(standard_challenge, skip_k2_challenge);
        assert_ne!(skip_k2_challenge, skip_k3_challenge);
    }

    #[test]
    fn native_u32_zero_variable_outer_sumcheck_is_exact() {
        let config = config(Q100);
        let witness = MulWitness::<u32>::from_inputs(&[(u32::MAX, u32::MAX)]).unwrap();
        let matrices = prepare_u32_mul_relation(*witness.layout(), &config).unwrap();
        let (assignment, products) = project_u32_mul_native_witness(&witness).into_parts();
        let assignment_binding = [0x3C; 32];
        let mut reference = None;

        for delayed in [false, true] {
            let mut transcript = Blake3Transcript::new();
            let (proof, claim) = if delayed {
                prove_spartan_piop_u32_native(
                    &mut transcript,
                    &matrices,
                    &assignment_binding,
                    products.clone(),
                    assignment.clone(),
                )
            } else {
                prove_spartan_piop(
                    &mut transcript,
                    &matrices,
                    &assignment_binding,
                    project_native_products(products.clone(), &config),
                    project_native_mle(assignment.clone(), &config),
                )
            }
            .unwrap();
            assert!(proof.outer.sumcheck.round_polynomials.is_empty());
            let continuation = transcript.get_challenge::<u128>();

            if let Some((reference_proof, reference_claim, reference_continuation)) = &reference {
                assert_eq!(&proof, reference_proof);
                assert_eq!(&claim, reference_claim);
                assert_eq!(&continuation, reference_continuation);
            } else {
                reference = Some((proof.clone(), claim.clone(), continuation));
            }

            let mut verifier_transcript = Blake3Transcript::new();
            let verified = verify_spartan_proof(
                &mut verifier_transcript,
                &matrices,
                &assignment_binding,
                &proof,
            )
            .unwrap();
            assert_eq!(claim, verified);
        }
    }

    #[test]
    fn native_u32_first_round_rejects_wide_multiplicands_before_absorption() {
        let config = config(Q100);
        let witness = MulWitness::<u32>::from_inputs(&[(2, 3)]).unwrap();
        let matrices = prepare_u32_mul_relation(*witness.layout(), &config).unwrap();
        let native = project_u32_mul_native_witness(&witness);
        let (assignment, mut products) = native.into_parts();
        products.az.evaluations[0] = u64::from(u32::MAX) + 1;
        let mut rejected_transcript = Blake3Transcript::new();

        assert_eq!(
            prove_spartan_piop_u32_native(
                &mut rejected_transcript,
                &matrices,
                &[0x5A; 32],
                products,
                assignment,
            ),
            Err(SpartanError::Sumcheck(
                SumcheckError::NativeMultiplicandOutOfRange
            ))
        );

        let mut fresh_transcript = Blake3Transcript::new();
        assert_eq!(
            rejected_transcript.get_challenge::<u128>(),
            fresh_transcript.get_challenge::<u128>()
        );
    }

    #[test]
    fn piop_is_generic_across_montgomery_limb_widths_and_zero_variable_domains() {
        let modulus = Uint::<2>::from(Q100).zero_extend::<3>();
        let config = Fp::<3>::make_cfg(&modulus).unwrap();
        let one = Fp::<3>::one_with_cfg(&config);
        let a = CscMatrix::try_from_rows(1, vec![vec![(0, one.clone())]]).unwrap();
        let b = a.clone();
        let c = a.clone();
        let matrices =
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), &config)
                .unwrap();
        let products = build_product_mles(
            std::slice::from_ref(&one),
            std::slice::from_ref(&one),
            std::slice::from_ref(&one),
            1,
            &config,
        )
        .unwrap();
        let assignment = build_assignment_mle(std::slice::from_ref(&one), 1, &config).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        let (proof, claim) = prove_spartan_nonsuccinct(
            &mut prover_transcript,
            &matrices,
            products,
            assignment.clone(),
        )
        .unwrap();
        assert!(proof.outer.sumcheck.round_polynomials.is_empty());
        assert!(proof.inner.round_polynomials.is_empty());

        let mut verifier_transcript = Blake3Transcript::new();
        verify_spartan_with_mle_claim(
            &mut verifier_transcript,
            &matrices,
            &proof,
            &claim,
            &assignment,
        )
        .unwrap();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_field_validation_preserves_first_error_and_pre_absorption() {
        let config = config(Q100);
        let foreign_config = self::config((1_u128 << 127) - 1);
        let one = Fp::<2>::one_with_cfg(&config);
        let zero = Fp::<2>::zero_with_cfg(&config);
        let logical_rows = (1 << 13) + 1;
        let matrix = || {
            CscMatrix::try_from_rows(
                1,
                (0..logical_rows)
                    .map(|row| {
                        if row == 0 {
                            vec![(0, one.clone())]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect(),
            )
            .unwrap()
        };
        let matrices = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(matrix(), matrix(), matrix()).unwrap(),
            &config,
        )
        .unwrap();
        assert_eq!(matrices.num_row_vars(), 14);

        let sequential_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let foreign = crate::piop::spartan::noncanonical_test_value(&matrices.config());
        let malformed = field::FpCtx::from_prime_u128(u128::MAX - 158)
            .from_montgomery_integer(*config.modulus());

        let mut foreign_first = vec![zero.clone(); 1 << 14];
        foreign_first[17] = foreign.clone();
        foreign_first[(1 << 13) + 3] = malformed.clone();
        let sequential_error = sequential_pool
            .install(|| validate_elements_field(&foreign_first, &matrices))
            .unwrap_err();
        let parallel_error = parallel_pool
            .install(|| validate_elements_field(&foreign_first, &matrices))
            .unwrap_err();
        assert_eq!(
            sequential_error,
            SpartanError::Matrix(SpartanMatrixError::InvalidFieldConfiguration(
                crate::piop::spartan::SpartanFieldError::NonCanonicalElement
            ))
        );
        assert_eq!(parallel_error, sequential_error);

        let mut malformed_first = vec![zero.clone(); 1 << 14];
        malformed_first[17] = malformed;
        malformed_first[(1 << 13) + 3] = foreign;
        let sequential_error = sequential_pool
            .install(|| validate_elements_field(&malformed_first, &matrices))
            .unwrap_err();
        let parallel_error = parallel_pool
            .install(|| validate_elements_field(&malformed_first, &matrices))
            .unwrap_err();
        assert_eq!(parallel_error, sequential_error);
        assert_eq!(
            sequential_error,
            SpartanError::Matrix(SpartanMatrixError::InvalidFieldConfiguration(
                crate::piop::spartan::SpartanFieldError::NonCanonicalElement,
            ))
        );

        let products = R1csProductMles {
            az: DenseMultilinearExtension {
                evaluations: foreign_first,
                num_vars: 14,
            },
            bz: DenseMultilinearExtension {
                evaluations: vec![zero.clone(); 1 << 14],
                num_vars: 14,
            },
            cz: DenseMultilinearExtension {
                evaluations: vec![zero; 1 << 14],
                num_vars: 14,
            },
        };
        let assignment = DenseMultilinearExtension {
            evaluations: vec![one.clone()],
            num_vars: 0,
        };
        let mut rejected_transcript = Blake3Transcript::new();
        let result = parallel_pool.install(|| {
            prove_spartan_piop(
                &mut rejected_transcript,
                &matrices,
                &[0x5a; 32],
                products,
                assignment,
            )
        });
        assert_eq!(
            result,
            Err(SpartanError::Matrix(
                SpartanMatrixError::InvalidFieldConfiguration(
                    crate::piop::spartan::SpartanFieldError::NonCanonicalElement
                )
            ))
        );

        let mut fresh_transcript = Blake3Transcript::new();
        assert_eq!(
            rejected_transcript.get_challenge::<u128>(),
            fresh_transcript.get_challenge::<u128>()
        );

        let logical_columns = (1 << 15) + 1;
        let wide_matrix =
            || CscMatrix::try_from_rows(logical_columns, vec![vec![(0, one.clone())]]).unwrap();
        let wide_matrices = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(wide_matrix(), wide_matrix(), wide_matrix()).unwrap(),
            &config,
        )
        .unwrap();
        assert_eq!(wide_matrices.num_column_vars(), 16);
        let mut padded_assignment = DenseMultilinearExtension {
            evaluations: vec![Fp::<2>::zero_with_cfg(&config); 1 << 16],
            num_vars: 16,
        };
        padded_assignment.evaluations[0] = one.clone();
        *padded_assignment.evaluations.last_mut().unwrap() = one.clone();
        let sequential_error = sequential_pool
            .install(|| validate_assignment(&wide_matrices, &padded_assignment))
            .unwrap_err();
        let parallel_error = parallel_pool
            .install(|| validate_assignment(&wide_matrices, &padded_assignment))
            .unwrap_err();
        assert_eq!(sequential_error, SpartanError::InvalidAssignmentPadding);
        assert_eq!(parallel_error, sequential_error);

        let one_value = DenseMultilinearExtension {
            evaluations: vec![one],
            num_vars: 0,
        };
        let products = R1csProductMles {
            az: one_value.clone(),
            bz: one_value.clone(),
            cz: one_value,
        };
        let mut rejected_transcript = Blake3Transcript::new();
        let result = parallel_pool.install(|| {
            prove_spartan_piop(
                &mut rejected_transcript,
                &wide_matrices,
                &[0xa5; 32],
                products,
                padded_assignment,
            )
        });
        assert_eq!(result, Err(SpartanError::InvalidAssignmentPadding));

        let mut fresh_transcript = Blake3Transcript::new();
        assert_eq!(
            rejected_transcript.get_challenge::<u128>(),
            fresh_transcript.get_challenge::<u128>()
        );
    }

    #[test]
    fn tampered_round_and_assignment_are_rejected() {
        let config = config(Q100);
        let (matrices, products, assignment) = fixture(&config);
        let assignment_binding =
            nonsuccinct_assignment_digest(&matrices, &assignment, &config).unwrap();
        let mut prover_transcript = Blake3Transcript::new();
        let (proof, claim) = prove_spartan_nonsuccinct(
            &mut prover_transcript,
            &matrices,
            products,
            assignment.clone(),
        )
        .unwrap();

        let mut tampered_proof = proof.clone();
        tampered_proof.outer.sumcheck.round_polynomials[0][0] = config.add(
            &(tampered_proof.outer.sumcheck.round_polynomials[0][0]),
            &(&Fp::<2>::one_with_cfg(&config)),
        );
        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            verify_spartan_proof(
                &mut verifier_transcript,
                &matrices,
                &assignment_binding,
                &tampered_proof,
            )
            .is_err()
        );

        let mut tampered_terminal = proof.clone();
        tampered_terminal.outer.az_mle_claim = config.add(
            &(tampered_terminal.outer.az_mle_claim),
            &(&Fp::<2>::one_with_cfg(&config)),
        );
        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            verify_spartan_proof(
                &mut verifier_transcript,
                &matrices,
                &assignment_binding,
                &tampered_terminal,
            )
            .is_err()
        );

        let foreign_config = self::config((1_u128 << 127) - 1);
        let mut foreign_proof = proof.clone();
        foreign_proof.outer.sumcheck.round_polynomials[0][0] =
            crate::piop::spartan::noncanonical_test_value(&matrices.config());
        let mut rejected_transcript = Blake3Transcript::new();
        assert_eq!(
            verify_spartan_proof(
                &mut rejected_transcript,
                &matrices,
                &assignment_binding,
                &foreign_proof,
            ),
            Err(SpartanError::Matrix(
                SpartanMatrixError::InvalidFieldConfiguration(
                    crate::piop::spartan::SpartanFieldError::NonCanonicalElement
                )
            ))
        );
        let mut untouched_transcript = Blake3Transcript::new();
        assert_eq!(
            rejected_transcript.get_challenge::<u128>(),
            untouched_transcript.get_challenge::<u128>(),
            "foreign-config proofs must fail before transcript mutation"
        );

        let mut tampered_assignment = assignment;
        tampered_assignment.evaluations[1] = config.add(
            &(tampered_assignment.evaluations[1]),
            &(&Fp::<2>::one_with_cfg(&config)),
        );
        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            verify_spartan_with_mle_claim(
                &mut verifier_transcript,
                &matrices,
                &proof,
                &claim,
                &tampered_assignment,
            )
            .is_err()
        );
    }

    #[test]
    fn nonsuccinct_verifier_rejects_the_zero_assignment_forgery() {
        let config = config(Q100);
        let one = Fp::<2>::one_with_cfg(&config);
        let zero = Fp::<2>::zero_with_cfg(&config);
        let a = CscMatrix::try_from_rows(1, vec![vec![(0, one.clone())]]).unwrap();
        let b = a.clone();
        let c = CscMatrix::try_from_rows(1, vec![Vec::new()]).unwrap();
        let matrices =
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), &config)
                .unwrap();

        // Without validating and pre-challenge binding of the assignment,
        // these all-zero terminal claims would discharge against h = [0].
        let forged_proof = SpartanPiopProof {
            outer: OuterSumcheckProof {
                sumcheck: SumcheckProof {
                    round_polynomials: Vec::new(),
                },
                az_mle_claim: zero.clone(),
                bz_mle_claim: zero.clone(),
                cz_mle_claim: zero.clone(),
            },
            inner: SumcheckProof {
                round_polynomials: Vec::new(),
            },
        };
        let forged_claim = ScaledMleEvaluationClaim::new(
            Vec::new().into_boxed_slice(),
            zero.clone(),
            zero.clone(),
        );
        let forged_assignment = DenseMultilinearExtension {
            evaluations: vec![zero],
            num_vars: 0,
        };
        let mut transcript = Blake3Transcript::new();

        assert_eq!(
            verify_spartan_with_mle_claim(
                &mut transcript,
                &matrices,
                &forged_proof,
                &forged_claim,
                &forged_assignment,
            ),
            Err(SpartanError::Matrix(
                SpartanMatrixError::InvalidAssignmentConstant
            ))
        );
    }
}
