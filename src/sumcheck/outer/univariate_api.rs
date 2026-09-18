//! Known-zero univariate prefix followed by the shared ordinary engine.
use super::{
    OuterArithmetic,
    api::*,
    engine::RoundState,
    inputs::OuterRows,
    ordinary::*,
    traversal::{PreparedFold, fold_and_message, integer_buckets},
    univariate::{UnivariateSkipProof, lagrange_weights_at},
};
use crate::piop::spartan::{SpartanField, matrix::make_equality_factors};
use crate::sumcheck::{RoundBoundaryPolicy, SumcheckError, SumcheckProof};
use crate::transcript::traits::Transcript;
use field::FieldOps;
use field::{BatchMulAcc, MergeAccumulator, Reduce};

/// Public field constants for 1..=4 little-endian prefix bits.
#[derive(Clone, Debug)]
pub struct PreparedUnivariateSkip<E> {
    skip_vars: u8,
    modulus: Vec<u8>,
    inverse_factorial_squared: E,
}
impl<E> PreparedUnivariateSkip<E> {
    pub fn skip_vars(&self) -> u8 {
        self.skip_vars
    }
}
pub fn prepare_univariate_skip<F: FieldOps>(
    field: &F,
    skip_vars: u8,
) -> Result<PreparedUnivariateSkip<F::Elem>, SumcheckError>
where
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    if !(1..=4).contains(&skip_vars) {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    let width = 1usize << skip_vars;
    let mut value = field.zero();
    // Base and exterior nodes must remain pairwise distinct.
    for _ in 1..=(2 * width - 3) {
        value = field.add(&value, &field.one());
        if <F::Elem as SpartanField>::is_zero(&value) {
            return Err(SumcheckError::InvalidSkipField);
        }
    }
    let mut factorial = field.one();
    value = field.one();
    for _ in 1..width {
        factorial = field.mul(&factorial, &value);
        value = field.add(&value, &field.one());
    }
    let inverse = *field.inverse_ct(&factorial).value();
    Ok(PreparedUnivariateSkip {
        skip_vars,
        modulus: F::Elem::canonical_modulus_encoding(field),
        inverse_factorial_squared: field.mul(&inverse, &inverse),
    })
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkippedOuterOutput<E> {
    pub prefix: UnivariateSkipProof<E>,
    pub prefix_challenge: E,
    pub tail: OuterOutput<E>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkippedOuterVerifierOutput<E> {
    pub prefix_challenge: E,
    pub tail: OuterVerifierOutput<E>,
}

impl<E> From<SkippedOuterOutput<E>> for super::univariate::UnivariateSkipOuterSumcheckOutput<E> {
    fn from(out: SkippedOuterOutput<E>) -> Self {
        let tail: crate::sumcheck::proof::OuterSumcheckOutput<E> = out.tail.into();
        let skip_vars = out.prefix.skip_vars;
        Self {
            proof: super::univariate::UnivariateSkipOuterSumcheckProof {
                skip: out.prefix,
                tail: tail.proof,
            },
            row_binding: super::univariate::PrefixUnivariateRowBinding {
                skip_vars,
                z: out.prefix_challenge,
                tail_point: tail.eval_points,
            },
            final_claim: tail.final_claim,
        }
    }
}

/// Interpolates a rowwise-zero prefix directly from row storage, then proves
/// the remaining Boolean tail. K=n is supported and produces an empty tail.
/// Owned inputs are released after prefix folding. Supplied factors must be
/// fresh equality tables for `tau_tail`; `None` prepares them here.
pub fn prove_outer_zerocheck_with_skip<F, I: OuterRows>(
    field: &F,
    transcript: &mut impl Transcript,
    prepared: &PreparedUnivariateSkip<F::Elem>,
    tau_tail: &[F::Elem],
    rows: I,
    factors: Option<EqualityFactors<F::Elem>>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<SkippedOuterOutput<F::Elem>, SumcheckError>
where
    F: OuterArithmetic<I::AB, I::C>,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    let prefix = prepare_prefix(
        field, transcript, prepared, tau_tail, &rows, factors, boundary,
    )?;
    drop(rows);
    finish_prefix(field, transcript, tau_tail, prefix, boundary)
}
struct PreparedPrefix<E> {
    proof: UnivariateSkipProof<E>,
    z: E,
    state: RoundState<E>,
    factors: EqualityFactors<E>,
    products: R1csProductTableBuffers<E>,
    pending: Option<[E; 3]>,
    inverses: Vec<E>,
}
fn finish_prefix<F>(
    field: &F,
    transcript: &mut impl Transcript,
    tau: &[F::Elem],
    prefix: PreparedPrefix<F::Elem>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<SkippedOuterOutput<F::Elem>, SumcheckError>
where
    F: FieldOps,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    #[cfg(feature = "bench-internals")]
    let _phase = super::measure::Phase::start(3);
    let _scope = tracing::info_span!("spartan:univariate_skip_tail").entered();
    let tail = continue_field_with_inverses(
        transcript,
        field,
        field,
        tau,
        prefix.factors,
        prefix.products,
        prefix.state,
        prefix.pending,
        Some(prefix.inverses),
        boundary,
    )?
    .into();
    Ok(SkippedOuterOutput {
        prefix: prefix.proof,
        prefix_challenge: prefix.z,
        tail,
    })
}
fn prepare_prefix<F, I: OuterRows>(
    field: &F,
    transcript: &mut impl Transcript,
    prepared: &PreparedUnivariateSkip<F::Elem>,
    tau: &[F::Elem],
    rows: &I,
    factors: Option<EqualityFactors<F::Elem>>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<PreparedPrefix<F::Elem>, SumcheckError>
where
    F: OuterArithmetic<I::AB, I::C>,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    #[cfg(feature = "bench-internals")]
    let setup = super::measure::Phase::start(0);
    let n = rows.validate()?;
    let a = rows.dimensions().0;
    let k = usize::from(prepared.skip_vars);
    if n < k || tau.len() != n - k {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    if prepared.modulus != F::Elem::canonical_modulus_encoding(field) {
        return Err(SumcheckError::FieldConfigurationMismatch);
    }
    boundary.validate(n - k)?;
    let factors = if let Some(factors) = factors {
        if !factors.matches_rows(a >> k) {
            return Err(SumcheckError::InvalidEqualityDimensions);
        }
        factors
    } else {
        let (low, high) = make_equality_factors(tau, field)
            .map_err(|_| SumcheckError::InvalidEqualityDimensions)?;
        EqualityFactors::new(low.evaluations, high.evaluations, field)
    };
    #[cfg(feature = "bench-internals")]
    drop(setup);
    match k {
        1 => prefix::<F, I, 2, 1, 0>(field, transcript, prepared, tau, rows, factors),
        2 => prefix::<F, I, 4, 3, 1>(field, transcript, prepared, tau, rows, factors),
        3 => prefix::<F, I, 8, 7, 3>(field, transcript, prepared, tau, rows, factors),
        4 => prefix::<F, I, 16, 15, 7>(field, transcript, prepared, tau, rows, factors),
        _ => Err(SumcheckError::InvalidProductDimensions),
    }
}

/// Forward differences at both ends of a base block.
pub(super) struct Differences<T, const M: usize> {
    pub left: [T; M],
    pub right: [T; M],
}
impl<T: Copy, const M: usize> Differences<T, M> {
    #[inline(always)]
    pub fn new(mut work: [T; M], sub: impl Fn(T, T) -> T) -> Self {
        let mut left = [work[0]; M];
        let mut right = [work[M - 1]; M];
        // Literal public orders keep LLVM from vectorizing the tiny shrinking
        // triangle with stack gathers/scatters for wide integer operands.
        macro_rules! order {
            ($order:literal) => {
                if M > $order {
                    for i in 0..M - $order {
                        work[i] = sub(work[i + 1], work[i]);
                    }
                    left[$order] = work[0];
                    right[$order] = work[M - $order - 1];
                }
            };
        }
        const {
            assert!(M > 0 && M <= 16);
        }
        order!(1);
        order!(2);
        order!(3);
        order!(4);
        order!(5);
        order!(6);
        order!(7);
        order!(8);
        order!(9);
        order!(10);
        order!(11);
        order!(12);
        order!(13);
        order!(14);
        order!(15);
        Self { left, right }
    }
}

/// Transpose the extrapolation traversal: each difference is read once and
/// advances all requested exterior nodes, retaining only STEPS running values.
#[inline(always)]
fn extrapolate_side<T: Copy, const M: usize, const STEPS: usize>(
    differences: &[T; M],
    op: impl Fn(T, T) -> T,
) -> [T; STEPS] {
    let mut running = [differences[M - 1]; STEPS];
    for j in (0..M - 1).rev() {
        let mut value = differences[j];
        for result in &mut running {
            *result = op(value, *result);
            value = *result;
        }
    }
    running
}

// Keep K=4 interpolation separate from padded reads. LLVM 22's SCCP pass
// can spend minutes propagating their ranges through the unrolled difference
// triangle, including during LTO. Smaller prefixes and one-word operands
// retain their inlining.
#[inline(always)]
pub(super) fn exterior_values<T: Copy, const M: usize, const LANES: usize, const STEPS: usize>(
    values: [T; M],
    add: impl Fn(T, T) -> T,
    sub: impl Fn(T, T) -> T,
) -> [T; LANES] {
    if M == 16 && core::mem::size_of::<T>() > core::mem::size_of::<u64>() {
        exterior_values_large::<T, M, LANES, STEPS>(values, add, sub)
    } else {
        exterior_values_inline::<T, M, LANES, STEPS>(values, add, sub)
    }
}

#[inline(never)]
fn exterior_values_large<T: Copy, const M: usize, const LANES: usize, const STEPS: usize>(
    values: [T; M],
    add: impl Fn(T, T) -> T,
    sub: impl Fn(T, T) -> T,
) -> [T; LANES] {
    exterior_values_inline::<T, M, LANES, STEPS>(values, add, sub)
}

#[inline(always)]
fn exterior_values_inline<T: Copy, const M: usize, const LANES: usize, const STEPS: usize>(
    values: [T; M],
    add: impl Fn(T, T) -> T,
    sub: impl Fn(T, T) -> T,
) -> [T; LANES] {
    const {
        assert!(LANES == M - 1 && 2 * STEPS + 1 == LANES);
    }
    let differences = Differences::new(values, &sub);
    let mut nodes = [differences.left[M - 1]; LANES];
    let left = extrapolate_side::<_, M, STEPS>(&differences.left, sub);
    for step in 0..STEPS {
        nodes[2 * step] = left[step];
    }
    let right = extrapolate_side::<_, M, STEPS>(&differences.right, add);
    for step in 0..STEPS {
        nodes[2 * step + 1] = right[step];
    }
    nodes
}

fn prefix<F, I: OuterRows, const M: usize, const LANES: usize, const STEPS: usize>(
    field: &F,
    transcript: &mut impl Transcript,
    prepared: &PreparedUnivariateSkip<F::Elem>,
    tau: &[F::Elem],
    rows: &I,
    mut factors: EqualityFactors<F::Elem>,
) -> Result<PreparedPrefix<F::Elem>, SumcheckError>
where
    F: OuterArithmetic<I::AB, I::C>,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    debug_assert_eq!(M - 1, LANES);
    #[cfg(feature = "bench-internals")]
    let coefficients_phase = super::measure::Phase::start(1);
    let message_scope = tracing::info_span!("spartan:univariate_skip_message").entered();
    let mut message = integer_buckets::<F, I::AB, I::C, LANES>(
        field,
        factors.weights(),
        |acc, weights, block, index| {
            // Finish one difference workspace at a time. Only the exterior values
            // and leading differences remain live during residual multiplication.
            let a = exterior_values::<_, M, LANES, STEPS>(
                core::array::from_fn(|j| field.lift_ab(rows.a(M * block + j))),
                |a, b| field.add_ab(a, b),
                |a, b| field.sub_ab(a, b),
            );
            let b = exterior_values::<_, M, LANES, STEPS>(
                core::array::from_fn(|j| field.lift_ab(rows.b(M * block + j))),
                |a, b| field.add_ab(a, b),
                |a, b| field.sub_ab(a, b),
            );
            let c = exterior_values::<_, M, LANES, STEPS>(
                core::array::from_fn(|j| field.lift_c(rows.c(M * block + j))),
                |a, b| field.add_c(a, b),
                |a, b| field.sub_c(a, b),
            );
            for node in 0..LANES - 1 {
                field.accumulate::<true>(
                    &mut acc[node],
                    weights,
                    index,
                    field.residual(field.product(a[node], b[node]), c[node]),
                );
            }
            field.accumulate::<true>(
                &mut acc[LANES - 1],
                weights,
                index,
                field.product(a[LANES - 1], b[LANES - 1]),
            );
        },
    )?;
    message[LANES - 1] = field.mul(&message[LANES - 1], &prepared.inverse_factorial_squared);
    let proof = UnivariateSkipProof::from_ordered_message(
        usize::from(prepared.skip_vars),
        message.to_vec(),
    )?;
    drop(message_scope);
    let reduction = proof.verify_reduction(transcript, field)?;
    #[cfg(feature = "bench-internals")]
    drop(coefficients_phase);
    #[cfg(feature = "bench-internals")]
    let _fold = super::measure::Phase::start(2);
    let _scope = tracing::info_span!("spartan:univariate_skip_prefix_fold").entered();
    let state = RoundState::new(reduction.q_at_z, tau.len(), field);
    let weights: [F::Elem; M] = lagrange_weights_at(&reduction.z, M, field)
        .try_into()
        .map_err(|_| SumcheckError::InvalidProductDimensions)?;
    let fold = PreparedFold::new(field, weights, rows);
    let mut products = R1csProductTableBuffers::zeroed(rows.dimensions().0 / M, field);
    let endpoint = tau
        .first()
        .map(FactoredEndpoint::for_tau)
        .unwrap_or(FactoredEndpoint::Zero);
    let inverses = batch_invert_nonzero(tau, field);
    let next = fold_and_message(field, &mut products, &mut factors, endpoint, fold)?;
    let pending = next.map(|values| {
        reconstruct_eq_factored_cubic_without_linear(
            &state.claim,
            &tau[0],
            &inverses[0],
            endpoint,
            values,
            &field.one(),
            &field.one(),
            field,
        )
    });
    Ok(PreparedPrefix {
        proof,
        z: reduction.z,
        state,
        factors,
        products,
        pending,
        inverses,
    })
}

pub fn verify_outer_zerocheck_with_skip<F: FieldOps>(
    field: &F,
    transcript: &mut impl Transcript,
    prepared: &PreparedUnivariateSkip<F::Elem>,
    tau_tail: &[F::Elem],
    prefix: &UnivariateSkipProof<F::Elem>,
    tail: &SumcheckProof<F::Elem, 4>,
    evaluations: OuterEvaluations<F::Elem>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<SkippedOuterVerifierOutput<F::Elem>, SumcheckError>
where
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    if prepared.modulus != F::Elem::canonical_modulus_encoding(field) {
        return Err(SumcheckError::FieldConfigurationMismatch);
    }
    if prefix.skip_vars != prepared.skip_vars {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    prefix.validate_shape(usize::from(prefix.skip_vars) + tau_tail.len())?;
    if tail.round_polynomials.len() != tau_tail.len() {
        return Err(SumcheckError::InvalidRoundCount {
            expected: tau_tail.len(),
            actual: tail.round_polynomials.len(),
        });
    }
    boundary.validate(tau_tail.len())?;
    crate::sumcheck::proof::validate_field_elements(tau_tail, field)?;
    crate::sumcheck::proof::validate_field_elements(
        &[evaluations.ax, evaluations.bx, evaluations.cx],
        field,
    )?;
    for round in &tail.round_polynomials {
        crate::sumcheck::proof::validate_field_elements(round, field)?;
    }
    let reduction = prefix.verify_reduction(transcript, field)?;
    let tail = verify_outer_sumcheck(
        field,
        transcript,
        reduction.q_at_z,
        tau_tail,
        tail,
        evaluations,
        boundary,
    )?;
    Ok(SkippedOuterVerifierOutput {
        prefix_challenge: reduction.z,
        tail,
    })
}
