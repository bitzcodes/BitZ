//! Equality-factored arithmetic and ordinary outer reduction.
use super::super::{SumcheckError, boundary::*, proof::*};
use super::engine::RoundState;
use crate::piop::spartan::SpartanField;
#[cfg(test)]
use crate::piop::spartan::absorb_field_elements;
#[cfg(test)]
use crate::piop::spartan::sumcheck::R1csProductMles;
use crate::poly::mle::DenseMultilinearExtension;
use crate::sumcheck::arithmetic::*;
use crate::transcript::traits::Transcript;
#[cfg(test)]
use field::Fp;
use field::RingOps;
use field::{BatchMulAcc, MergeAccumulator, Reduce};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
/// Independent direct-cubic field-table oracle under the requested boundary policy.
#[cfg(test)]
pub(crate) fn prove_field_with_boundary_reference<F, R, P>(
    transcript: &mut impl Transcript,
    initial_claim: F,
    tau: &[F],
    (eq_low, eq_high): (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
    products: R1csProductMles<F>,
    field_cfg: &F::Config,
    reducer: &R,
    round_boundary: &mut P,
) -> Result<OuterSumcheckOutput<F>, SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
    P: RoundBoundaryPolicy,
{
    let num_vars = products.az.num_vars;
    if !has_dense_shape(&products.az)
        || !has_dense_shape(&products.bz)
        || !has_dense_shape(&products.cz)
        || products.bz.num_vars != num_vars
        || products.cz.num_vars != num_vars
    {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    if tau.len() != num_vars {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    if !has_dense_shape(&eq_low)
        || !has_dense_shape(&eq_high)
        || eq_low
            .num_vars
            .checked_add(eq_high.num_vars)
            .is_none_or(|eq_vars| eq_vars != num_vars)
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    round_boundary.validate(num_vars)?;

    let _ = reducer;
    prove_outer_sumcheck_direct_reference_with_boundary(
        transcript,
        initial_claim,
        tau,
        products,
        field_cfg,
        round_boundary,
    )
}

/// Independent direct-cubic oracle for transcript-compatibility tests. This is
/// intentionally scalar and materializes the full equality table so it cannot
/// accidentally share the optimized factor-stripping arithmetic.
#[cfg(test)]
pub(crate) fn prove_outer_sumcheck_direct_reference<F: SpartanField>(
    transcript: &mut impl Transcript,
    initial_claim: F,
    tau: &[F],
    products: R1csProductMles<F>,
    field_cfg: &F::Config,
) -> Result<OuterSumcheckOutput<F>, SumcheckError> {
    prove_outer_sumcheck_direct_reference_with_boundary(
        transcript,
        initial_claim,
        tau,
        products,
        field_cfg,
        &mut UngrindedRoundBoundary,
    )
}

#[cfg(test)]
pub(crate) fn prove_outer_sumcheck_direct_reference_with_boundary<F, P>(
    transcript: &mut impl Transcript,
    initial_claim: F,
    tau: &[F],
    products: R1csProductMles<F>,
    field_cfg: &F::Config,
    round_boundary: &mut P,
) -> Result<OuterSumcheckOutput<F>, SumcheckError>
where
    F: SpartanField,
    P: RoundBoundaryPolicy,
{
    let num_vars = products.az.num_vars;
    if !has_dense_shape(&products.az)
        || !has_dense_shape(&products.bz)
        || !has_dense_shape(&products.cz)
        || products.bz.num_vars != num_vars
        || products.cz.num_vars != num_vars
    {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    if tau.len() != num_vars {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }

    round_boundary.validate(num_vars)?;
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let two = (field_cfg).add(&one, &one);
    let three = (field_cfg).add(&two, &one);
    let interpolation = CubicInterpolation::new(field_cfg);
    let mut equality = crate::piop::spartan::matrix::eq_table(tau, field_cfg)
        .expect("test reference receives valid equality coordinates");
    let mut products = R1csProductTableBuffers::from_mles(products);
    let mut product_scratch = R1csProductTableBuffers::filled(products.len() / 2, &zero);
    let mut equality_scratch = vec![zero.clone(); equality.len() / 2];
    let mut current_claim = initial_claim;
    let mut eval_points = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);

    while products.len() > 1 {
        let mut evaluations = std::array::from_fn(|_| zero.clone());
        for pair in 0..products.len() / 2 {
            let index = 2 * pair;
            for (evaluation, point) in
                evaluations
                    .iter_mut()
                    .zip([zero.clone(), two.clone(), three.clone()])
            {
                let equality_at =
                    interpolate_pair(&equality[index], &equality[index + 1], &point, &field_cfg);
                let az_at = interpolate_pair(
                    &products.az[index],
                    &products.az[index + 1],
                    &point,
                    &field_cfg,
                );
                let bz_at = interpolate_pair(
                    &products.bz[index],
                    &products.bz[index + 1],
                    &point,
                    &field_cfg,
                );
                let cz_at = interpolate_pair(
                    &products.cz[index],
                    &products.cz[index + 1],
                    &point,
                    &field_cfg,
                );
                *evaluation = field_cfg.add(
                    &(*evaluation),
                    &(&(field_cfg).mul(
                        &equality_at,
                        &(field_cfg).sub(&(field_cfg).mul(&az_at, &bz_at), &cz_at),
                    )),
                );
            }
        }

        let coefficients_without_linear =
            interpolation.coefficients_without_linear(&current_claim, evaluations, &field_cfg);
        let challenge = recover_full_round_polynomial_and_sample_next_challenge_with_boundary(
            transcript,
            &mut current_claim,
            &coefficients_without_linear,
            &mut round_polynomials,
            &mut eval_points,
            &zero,
            field_cfg,
            round_boundary,
        )?;

        let next_len = products.len() / 2;
        product_scratch.truncate(next_len);
        equality_scratch.truncate(next_len);
        fold_product_tables(&products, &mut product_scratch, &challenge, &field_cfg);
        fold_table(&equality, &mut equality_scratch, &challenge, &field_cfg);
        products.swap(&mut product_scratch);
        std::mem::swap(&mut equality, &mut equality_scratch);
    }

    let az_mle_claim = products.az[0].clone();
    let bz_mle_claim = products.bz[0].clone();
    let cz_mle_claim = products.cz[0].clone();
    let terminal = field_cfg.mul(
        &equality[0],
        &field_cfg.sub(&field_cfg.mul(&az_mle_claim, &bz_mle_claim), &cz_mle_claim),
    );
    if current_claim != terminal {
        return Err(SumcheckError::InvalidTerminalClaim);
    }
    absorb_field_elements(
        transcript,
        &[
            az_mle_claim.clone(),
            bz_mle_claim.clone(),
            cz_mle_claim.clone(),
        ],
        &field_cfg,
    );

    Ok(OuterSumcheckOutput {
        proof: OuterSumcheckProof {
            sumcheck: SumcheckProof { round_polynomials },
            az_mle_claim,
            bz_mle_claim,
            cz_mle_claim,
        },
        eval_points,
        final_claim: current_claim,
    })
}

/// Proves the first outer round from exact native u32 relation products, then
/// continues with the ordinary field-valued prover.
///
/// The native tables are consumed at the first Fiat--Shamir challenge. Every
/// folded entry is reduced to the field before it is stored or used in a later
/// multiplication.
#[cfg(test)]
pub(crate) fn prove_u32_first_round<R>(
    transcript: &mut impl Transcript,
    initial_claim: Fp<2>,
    tau: &[Fp<2>],
    (eq_low, eq_high): (
        DenseMultilinearExtension<Fp<2>>,
        DenseMultilinearExtension<Fp<2>>,
    ),
    products: R1csProductMles<u64>,
    field_cfg: &field::FpCtx<2>,
    reducer: &R,
) -> Result<OuterSumcheckOutput<Fp<2>>, SumcheckError>
where
    R: BatchMulAcc<Fp<2>>
        + Reduce<<R as BatchMulAcc<Fp<2>>>::Accumulator, Output = Fp<2>>
        + Sync
        + SumcheckLinearReducer,
{
    let num_vars = products.az.num_vars;
    if !has_dense_shape(&products.az)
        || !has_dense_shape(&products.bz)
        || !has_dense_shape(&products.cz)
        || products.bz.num_vars != num_vars
        || products.cz.num_vars != num_vars
    {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    if tau.len() != num_vars {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    if products
        .az
        .evaluations
        .iter()
        .chain(&products.bz.evaluations)
        .any(|&value| value > u64::from(u32::MAX))
    {
        return Err(SumcheckError::NativeMultiplicandOutOfRange);
    }
    if !has_dense_shape(&eq_low)
        || !has_dense_shape(&eq_high)
        || eq_low
            .num_vars
            .checked_add(eq_high.num_vars)
            .is_none_or(|eq_vars| eq_vars != num_vars)
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }

    let zero = field_cfg.zero();
    let one = field_cfg.one();
    let native = R1csProductTableBuffers::from_mles(products);
    let mut state = RoundState::new(initial_claim, num_vars, field_cfg);
    if num_vars == 0 {
        return state.finish(
            field_cfg,
            transcript,
            [
                native_to_field(native.az[0], field_cfg),
                native_to_field(native.bz[0], field_cfg),
                native_to_field(native.cz[0], field_cfg),
            ],
        );
    }
    let mut factors = EqualityFactors::new(eq_low.evaluations, eq_high.evaluations, field_cfg);
    factors.strip(field_cfg);
    let inverses = batch_invert_nonzero(tau, field_cfg);
    let coefficients = compute_u32_native_eq_factored_coefficients_without_linear(
        &native,
        factors.weights(),
        &state.claim,
        &tau[0],
        &inverses[0],
        &state.equality_scale,
        &one,
        &zero,
        reducer,
        field_cfg,
    )?;
    let challenge = state.sample(
        field_cfg,
        transcript,
        tau,
        &coefficients,
        &mut UngrindedRoundBoundary,
    )?;
    let folded = fold_u64_product_tables_to_field(&native, &challenge, &zero, field_cfg, reducer)?;
    continue_field(
        transcript,
        field_cfg,
        reducer,
        tau,
        factors,
        folded,
        state,
        None,
        &mut UngrindedRoundBoundary,
    )
}

/// Owned evaluation tables used by the fused outer-sumcheck kernels.
pub(crate) struct R1csProductTableBuffers<F> {
    pub(super) az: Vec<F>,
    pub(super) bz: Vec<F>,
    pub(super) cz: Vec<F>,
}

impl<F: Copy> R1csProductTableBuffers<F> {
    #[cfg(test)]
    fn from_mles(products: R1csProductMles<F>) -> Self {
        Self {
            az: products.az.evaluations,
            bz: products.bz.evaluations,
            cz: products.cz.evaluations,
        }
    }

    #[inline(always)]
    #[cfg(test)]
    pub(super) fn filled(len: usize, value: &F) -> Self {
        Self {
            az: vec![*value; len],
            bz: vec![*value; len],
            cz: vec![*value; len],
        }
    }

    pub(super) fn zeroed(len: usize, field: &F::Config) -> Self
    where
        F: SpartanField,
    {
        Self {
            az: field.zero_vec(len),
            bz: field.zero_vec(len),
            cz: field.zero_vec(len),
        }
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.az.len(), self.bz.len());
        debug_assert_eq!(self.az.len(), self.cz.len());
        self.az.len()
    }

    fn truncate(&mut self, len: usize) {
        debug_assert!(self.az.len() >= len);
        debug_assert!(self.bz.len() >= len);
        debug_assert!(self.cz.len() >= len);
        self.az.truncate(len);
        self.bz.truncate(len);
        self.cz.truncate(len);
    }

    fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.az, &mut other.az);
        std::mem::swap(&mut self.bz, &mut other.bz);
        std::mem::swap(&mut self.cz, &mut other.cz);
    }
}

/// Accumulates an exact native pair for the differential test oracle.
#[inline]
#[cfg(test)]
pub(crate) fn accumulate_u32_native_outer_pair<R>(
    accumulators: &mut [<R as SumcheckLinearReducer>::Accumulator; 2],
    products: &R1csProductTableBuffers<u64>,
    index: usize,
    weight: &Fp<2>,
    negative_weight: &Fp<2>,
    endpoint: FactoredEndpoint,
    reducer: &R,
) where
    R: SumcheckLinearReducer,
{
    let az_zero = products.az[index];
    let az_one = products.az[index + 1];
    let bz_zero = products.bz[index];
    let bz_one = products.bz[index + 1];
    let cz_zero = products.cz[index];
    let cz_one = products.cz[index + 1];
    let (az_endpoint, bz_endpoint, cz_endpoint) = match endpoint {
        FactoredEndpoint::Zero => (az_zero, bz_zero, cz_zero),
        FactoredEndpoint::One => (az_one, bz_one, cz_one),
        FactoredEndpoint::KnownZero => (0, 0, 0),
    };
    let endpoint_residual =
        i128::from(native_u32_product(az_endpoint, bz_endpoint)) - i128::from(cz_endpoint);
    let az_delta = i128::from(az_one) - i128::from(az_zero);
    let bz_delta = i128::from(bz_one) - i128::from(bz_zero);
    let infinity = az_delta * bz_delta;

    // `A_endpoint * B_endpoint` and `C_endpoint` are u64, so their signed
    // difference has magnitude at most `u64::MAX`. Each delta magnitude is at
    // most `u32::MAX`, hence the infinity product also fits u64.
    multiply_accumulate_signed_linear(
        &mut accumulators[0],
        weight,
        negative_weight,
        endpoint_residual,
        reducer,
    );
    multiply_accumulate_signed_linear(
        &mut accumulators[1],
        weight,
        negative_weight,
        infinity,
        reducer,
    );
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn compute_u32_native_eq_factored_coefficients_without_linear<R>(
    products: &R1csProductTableBuffers<u64>,
    equality_weights: StrippedEqualityWeights<'_, Fp<2>>,
    current_claim: &Fp<2>,
    tau: &Fp<2>,
    tau_inverse_or_zero: &Fp<2>,
    bound_equality: &Fp<2>,
    one: &Fp<2>,
    zero: &Fp<2>,
    reducer: &R,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) -> Result<[Fp<2>; 3], SumcheckError>
where
    R: SumcheckLinearReducer
        + BatchMulAcc<Fp<2>>
        + Reduce<<R as BatchMulAcc<Fp<2>>>::Accumulator, Output = Fp<2>>
        + Sync,
{
    let pair_count = products.len() / 2;
    let endpoint = FactoredEndpoint::for_tau(tau);

    // The first round uses the same eq_out * (sum eq_in * residual)
    // decomposition, with exact native u32 products feeding field×u64 inner
    // accumulators and field×field outer accumulators.
    if equality_weights
        .low_weights()
        .is_some_and(|low| low.len() >= TWO_LEVEL_EQUALITY_MIN_LOW_PAIRS)
    {
        let low_weights = equality_weights
            .low_weights()
            .expect("two-level branch requires stripped low weights");
        let negative_low_weights = low_weights
            .iter()
            .map(|weight| (field_config).sub(zero, weight))
            .collect::<Vec<_>>();
        let low_pair_count = low_weights.len();
        let accumulate_high_bucket = |mut outer: [<R as BatchMulAcc<Fp<2>>>::Accumulator; 2],
                                      high_index: usize|
         -> Result<_, SumcheckError> {
            let mut inner =
                std::array::from_fn(|_| <R as SumcheckLinearReducer>::accumulator_zero(reducer));
            let product_pair_start = high_index * low_pair_count;
            for (low_pair_index, (weight, negative_weight)) in
                low_weights.iter().zip(&negative_low_weights).enumerate()
            {
                let index = 2 * (product_pair_start + low_pair_index);
                accumulate_u32_native_outer_pair(
                    &mut inner,
                    products,
                    index,
                    weight,
                    negative_weight,
                    endpoint,
                    reducer,
                );
            }

            let inner = reduce_two_linear_accumulators(inner, reducer, &field_config)?;
            let high_weight = &equality_weights.high[high_index];
            for (outer, inner) in outer.iter_mut().zip(&inner) {
                <R as BatchMulAcc<Fp<2>>>::mul_acc(reducer, outer, high_weight, inner);
            }
            Ok(outer)
        };

        #[cfg(feature = "parallel")]
        let accumulators = if should_parallelize(pair_count) {
            (0..equality_weights.high.len())
                .into_par_iter()
                .try_fold(
                    || std::array::from_fn(|_| <R as BatchMulAcc<Fp<2>>>::Accumulator::zero()),
                    accumulate_high_bucket,
                )
                .try_reduce(
                    || std::array::from_fn(|_| <R as BatchMulAcc<Fp<2>>>::Accumulator::zero()),
                    |left, right| Ok(merge_accumulators(left, right)),
                )?
        } else {
            let mut accumulators =
                std::array::from_fn(|_| <R as BatchMulAcc<Fp<2>>>::Accumulator::zero());
            for high_index in 0..equality_weights.high.len() {
                accumulators = accumulate_high_bucket(accumulators, high_index)?;
            }
            accumulators
        };

        #[cfg(not(feature = "parallel"))]
        let accumulators = {
            let mut accumulators =
                std::array::from_fn(|_| <R as BatchMulAcc<Fp<2>>>::Accumulator::zero());
            for high_index in 0..equality_weights.high.len() {
                accumulators = accumulate_high_bucket(accumulators, high_index)?;
            }
            accumulators
        };

        let evaluations = reduce_two_accumulators(accumulators, reducer, &field_config)?;
        return Ok(reconstruct_eq_factored_cubic_without_linear(
            current_claim,
            tau,
            tau_inverse_or_zero,
            endpoint,
            evaluations,
            bound_equality,
            one,
            &field_config,
        ));
    }

    let accumulators = sum_linear_accumulators(
        pair_count,
        |accumulators, pair| {
            let index = 2 * pair;
            let weight = equality_weights.pair_weight(pair, &field_config);
            let negative_weight = (field_config).sub(zero, &weight);
            accumulate_u32_native_outer_pair(
                accumulators,
                products,
                index,
                &weight,
                &negative_weight,
                endpoint,
                reducer,
            );
        },
        reducer,
    );
    let evaluations = reduce_two_linear_accumulators(accumulators, reducer, &field_config)?;
    Ok(reconstruct_eq_factored_cubic_without_linear(
        current_claim,
        tau,
        tau_inverse_or_zero,
        endpoint,
        evaluations,
        bound_equality,
        one,
        &field_config,
    ))
}
#[cfg(test)]
pub(crate) fn fold_u64_product_tables_to_field<R>(
    input: &R1csProductTableBuffers<u64>,
    challenge: &Fp<2>,
    zero: &Fp<2>,
    field_cfg: &field::FpCtx<2>,
    reducer: &R,
) -> Result<R1csProductTableBuffers<Fp<2>>, SumcheckError>
where
    R: SumcheckLinearReducer,
{
    let mut output = R1csProductTableBuffers::filled(input.len() / 2, zero);
    fold_u64_table_to_field(
        &input.az,
        &mut output.az,
        challenge,
        zero,
        reducer,
        &field_cfg,
    )?;
    fold_u64_table_to_field(
        &input.bz,
        &mut output.bz,
        challenge,
        zero,
        reducer,
        &field_cfg,
    )?;
    fold_u64_table_to_field(
        &input.cz,
        &mut output.cz,
        challenge,
        zero,
        reducer,
        &field_cfg,
    )?;
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FactoredEndpoint {
    Zero,
    One,
    /// Rowwise zero residual: accumulate only the quadratic leading coefficient.
    #[cfg(test)]
    KnownZero,
}

impl FactoredEndpoint {
    #[inline]
    pub(crate) fn for_tau<F>(tau: &F) -> Self
    where
        F: SpartanField,
    {
        if F::is_zero(tau) {
            Self::One
        } else {
            Self::Zero
        }
    }
}

/// Inverts every nonzero entry with one field inversion. Zero entries remain
/// zero and use the endpoint-one recovery path below.
pub(crate) fn batch_invert_nonzero<F>(values: &[F], field_cfg: &F::Config) -> Vec<F>
where
    F: SpartanField,
{
    field::BatchFieldOps::batch_invert_or_zero_ct(field_cfg, values)
}

#[inline]
pub(crate) fn equality_coordinate_evaluation<F>(
    tau: &F,
    point: &F,
    one: &F,
    field_config: &F::Config,
) -> F
where
    F: SpartanField,
{
    let at_zero = (field_config).sub(one, tau);
    let slope = (field_config).sub(tau, &at_zero);
    (field_config).add(&at_zero, &(field_config).mul(point, &slope))
}

/// Converts one endpoint and the leading coefficient of the quadratic
/// cofactor into `[c0, c2, c3]` for the original equality-weighted cubic.
/// The existing round helper restores `c1`, so the proof and transcript remain
/// exactly `[F; 4]`.
pub(crate) fn reconstruct_eq_factored_cubic_without_linear<F>(
    current_claim: &F,
    tau: &F,
    tau_inverse_or_zero: &F,
    endpoint: FactoredEndpoint,
    endpoint_and_infinity: [F; 2],
    bound_equality: &F,
    one: &F,
    field_config: &F::Config,
) -> [F; 3]
where
    F: SpartanField,
{
    let [endpoint_evaluation, infinity] = endpoint_and_infinity;
    let endpoint_evaluation = (field_config).mul(bound_equality, &endpoint_evaluation);
    let infinity = (field_config).mul(bound_equality, &infinity);
    let at_zero = (field_config).sub(one, tau);
    let slope = (field_config).sub(tau, &at_zero);

    let (cofactor_zero, cofactor_one) = match endpoint {
        FactoredEndpoint::Zero => {
            let weighted_zero = (field_config).mul(&at_zero, &endpoint_evaluation);
            let cofactor_one = (field_config).mul(
                &(field_config).sub(current_claim, &weighted_zero),
                tau_inverse_or_zero,
            );
            (endpoint_evaluation, cofactor_one)
        }
        FactoredEndpoint::One => (*current_claim, endpoint_evaluation),
        #[cfg(test)]
        FactoredEndpoint::KnownZero => (field_config.zero(), field_config.zero()),
    };

    let linear = (field_config).sub(
        &(field_config).sub(&cofactor_one, &cofactor_zero),
        &infinity,
    );
    let c0 = (field_config).mul(&at_zero, &cofactor_zero);
    let c2 = (field_config).add(
        &(field_config).mul(&at_zero, &infinity),
        &(field_config).mul(&slope, &linear),
    );
    let c3 = (field_config).mul(&slope, &infinity);
    [c0, c2, c3]
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_eq_factored_cofactor_evaluations<F, R>(
    accumulators: &mut [<R as BatchMulAcc<F>>::Accumulator; 2],
    weight: &F,
    endpoint: FactoredEndpoint,
    az_zero: &F,
    az_one: &F,
    bz_zero: &F,
    bz_one: &F,
    cz_zero: &F,
    cz_one: &F,
    reducer: &R,
    field_config: &F::Config,
) where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    let endpoint_residual = match endpoint {
        FactoredEndpoint::Zero => {
            (field_config).sub(&(field_config).mul(az_zero, bz_zero), cz_zero)
        }
        FactoredEndpoint::One => (field_config).sub(&(field_config).mul(az_one, bz_one), cz_one),
        #[cfg(test)]
        FactoredEndpoint::KnownZero => field_config.zero(),
    };
    reducer.mul_acc(&mut accumulators[0], weight, &endpoint_residual);

    let az_delta = (field_config).sub(az_one, az_zero);
    let bz_delta = (field_config).sub(bz_one, bz_zero);
    let infinity = (field_config).mul(&az_delta, &bz_delta);
    reducer.mul_acc(&mut accumulators[1], weight, &infinity);
}

/// Public-modulus constants used to interpolate a cubic from evaluations at
/// 0, 1, 2, and 3. They are prepared once per outer sumcheck, outside every
/// product loop.
#[cfg(test)]
pub(crate) struct CubicInterpolation<F> {
    half: F,
    sixth: F,
}

#[cfg(test)]
impl<F> CubicInterpolation<F>
where
    F: SpartanField,
{
    fn new(field_cfg: &F::Config) -> Self {
        let one = F::one_with_cfg(field_cfg);
        let two = (field_cfg).add(&one, &one);
        let three = (field_cfg).add(&two, &one);
        let six = (field_cfg).add(&three, &three);
        Self {
            half: *field::FieldOps::inverse_ct(&field_cfg, &two).value(),
            sixth: *field::FieldOps::inverse_ct(field_cfg, &six).value(),
        }
    }

    /// Converts `[g(0), g(2), g(3)]` to `[c0, c2, c3]`, deriving
    /// `g(1) = current_claim - g(0)` from the sumcheck relation.
    fn coefficients_without_linear(
        &self,
        current_claim: &F,
        evaluations: [F; 3],
        field_config: &F::Config,
    ) -> [F; 3] {
        let [at_zero, at_two, at_three] = evaluations;
        let at_one = (field_config).sub(current_claim, &at_zero);
        let three_at_one = (field_config).add(&(field_config).add(&at_one, &at_one), &at_one);
        let three_at_two = (field_config).add(&(field_config).add(&at_two, &at_two), &at_two);
        let third_difference = (field_config).sub(
            &(field_config).add(&(field_config).sub(&at_three, &three_at_two), &three_at_one),
            &at_zero,
        );
        let c3 = (field_config).mul(&third_difference, &self.sixth);

        let second_difference = (field_config).add(
            &(field_config).sub(&at_two, &(field_config).add(&at_one, &at_one)),
            &at_zero,
        );
        let three_c3 = (field_config).add(&(field_config).add(&c3, &c3), &c3);
        let c2 = (field_config).sub(
            &(field_config).mul(&second_difference, &self.half),
            &three_c3,
        );
        [at_zero, c2, c3]
    }
}

/// Equality weights with the current coordinate stripped.
pub(crate) struct StrippedEqualityWeights<'a, F> {
    low: Option<&'a [F]>,
    high: &'a [F],
}

impl<'a, F> StrippedEqualityWeights<'a, F>
where
    F: SpartanField,
{
    fn low(low: &'a [F], high: &'a [F]) -> Self {
        debug_assert!(!low.is_empty());
        debug_assert!(low.len().is_power_of_two());
        debug_assert!(high.len().is_power_of_two());
        Self {
            low: Some(low),
            high,
        }
    }

    fn high(high: &'a [F]) -> Self {
        debug_assert!(!high.is_empty());
        debug_assert!(high.len().is_power_of_two());
        Self { low: None, high }
    }

    pub(super) fn buckets(&self, one: &'a [F; 1]) -> (&'a [F], &'a [F]) {
        match self.low {
            Some(low) => (low, self.high),
            None => (self.high, one),
        }
    }

    /// Returns the suffix equality weight after stripping the active
    /// coordinate. Bound-coordinate equality factors are tracked separately.
    #[inline]
    pub(super) fn pair_weight(&self, pair: usize, field_config: &F::Config) -> F {
        if let Some(low) = self.low {
            return (field_config).mul(&low[pair % low.len()], &self.high[pair / low.len()]);
        }

        self.high[pair]
    }

    #[inline]
    fn low_weights(&self) -> Option<&'a [F]> {
        self.low
    }
}

/// The Spartan2-style two-level decomposition evaluates the four-factor term
///
/// `eq_out * eq_in * A * B`
///
/// as `eq_out * (sum(eq_in * (A * B - C)))`: first reduce one inner subtotal
/// per `eq_out` bucket, then delayed-MAC that subtotal into the outer sum. This
/// removes the N-scaling immediate `eq_out * eq_in` multiplication. Very short
/// buckets do not amortize the inner reduction, so their tail rounds retain
/// the direct product.
pub(crate) const TWO_LEVEL_EQUALITY_MIN_LOW_PAIRS: usize = 8;

/// Computes the cofactor endpoint and leading coefficient in two
/// delayed-reduction levels.
///
/// For each `eq_out` (high-factor) bucket, the first level accumulates
/// `eq_in * (A * B - C)` over all low pairs and reduces that subtotal. The
/// second level accumulates `eq_out * subtotal`. At no point is
/// `eq_out * eq_in` formed with an immediate field multiplication.
pub(crate) fn compute_two_level_cofactor_evaluations<F, R>(
    products: &R1csProductTableBuffers<F>,
    equality_weights: &StrippedEqualityWeights<'_, F>,
    endpoint: FactoredEndpoint,
    reducer: &R,
    field_config: &F::Config,
) -> Result<[F; 2], SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    let low_weights = equality_weights
        .low_weights()
        .expect("two-level accumulation requires stripped low weights");
    let low_pair_count = low_weights.len();
    debug_assert_eq!(
        products.len() / 2,
        low_pair_count * equality_weights.high.len()
    );

    let accumulate_high_bucket = |mut outer: [<R as BatchMulAcc<F>>::Accumulator; 2],
                                  high_index: usize|
     -> Result<_, SumcheckError> {
        let mut inner = std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero());
        let product_pair_start = high_index * low_pair_count;

        for (low_pair_index, weight) in low_weights.iter().enumerate() {
            let product_index = 2 * (product_pair_start + low_pair_index);
            accumulate_eq_factored_cofactor_evaluations(
                &mut inner,
                weight,
                endpoint,
                &products.az[product_index],
                &products.az[product_index + 1],
                &products.bz[product_index],
                &products.bz[product_index + 1],
                &products.cz[product_index],
                &products.cz[product_index + 1],
                reducer,
                &field_config,
            );
        }

        let inner = reduce_two_accumulators_bounded(inner, reducer, low_pair_count, &field_config)?;
        let high_weight = &equality_weights.high[high_index];
        for (outer, inner) in outer.iter_mut().zip(&inner) {
            reducer.mul_acc(outer, high_weight, inner);
        }
        Ok(outer)
    };

    #[cfg(feature = "parallel")]
    if should_parallelize(products.len() / 2) {
        let accumulators = (0..equality_weights.high.len())
            .into_par_iter()
            .try_fold(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                accumulate_high_bucket,
            )
            .try_reduce(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |left, right| Ok(merge_accumulators(left, right)),
            )?;
        return reduce_two_accumulators_bounded(
            accumulators,
            reducer,
            equality_weights.high.len(),
            &field_config,
        );
    }

    let mut accumulators = std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero());
    for high_index in 0..equality_weights.high.len() {
        accumulators = accumulate_high_bucket(accumulators, high_index)?;
    }
    reduce_two_accumulators_bounded(
        accumulators,
        reducer,
        equality_weights.high.len(),
        &field_config,
    )
}

/// Computes `[c0, c2, c3]` from one endpoint and the leading coefficient of
/// the quadratic cofactor over adjacent pairs in the product tables.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_eq_factored_coefficients_without_linear<F, R>(
    products: &R1csProductTableBuffers<F>,
    equality_weights: StrippedEqualityWeights<'_, F>,
    current_claim: &F,
    tau: &F,
    tau_inverse_or_zero: &F,
    bound_equality: &F,
    one: &F,
    reducer: &R,
    field_config: &F::Config,
) -> Result<[F; 3], SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    let pair_count = products.len() / 2;
    let endpoint = FactoredEndpoint::for_tau(tau);
    let evaluations = if equality_weights
        .low_weights()
        .is_some_and(|low| low.len() >= TWO_LEVEL_EQUALITY_MIN_LOW_PAIRS)
    {
        compute_two_level_cofactor_evaluations(
            products,
            &equality_weights,
            endpoint,
            reducer,
            &field_config,
        )?
    } else {
        let accumulators = sum_product_accumulators(
            pair_count,
            |accumulators, pair| {
                let index = 2 * pair;
                let weight = equality_weights.pair_weight(pair, &field_config);
                accumulate_eq_factored_cofactor_evaluations(
                    accumulators,
                    &weight,
                    endpoint,
                    &products.az[index],
                    &products.az[index + 1],
                    &products.bz[index],
                    &products.bz[index + 1],
                    &products.cz[index],
                    &products.cz[index + 1],
                    reducer,
                    &field_config,
                );
            },
            reducer,
        );
        reduce_two_accumulators_bounded(accumulators, reducer, pair_count, &field_config)?
    };
    Ok(reconstruct_eq_factored_cubic_without_linear(
        current_claim,
        tau,
        tau_inverse_or_zero,
        endpoint,
        evaluations,
        bound_equality,
        one,
        &field_config,
    ))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_product_chunk<F>(
    az: &[F],
    bz: &[F],
    cz: &[F],
    az_output: &mut [F],
    bz_output: &mut [F],
    cz_output: &mut [F],
    challenge: &F,
    field_config: &F::Config,
) -> [[F; 2]; 3]
where
    F: SpartanField,
{
    debug_assert_eq!(az_output.len(), 2);
    debug_assert_eq!(bz_output.len(), 2);
    debug_assert_eq!(cz_output.len(), 2);

    let folded = [
        fold_two_pairs(az, challenge, &field_config),
        fold_two_pairs(bz, challenge, &field_config),
        fold_two_pairs(cz, challenge, &field_config),
    ];
    az_output.clone_from_slice(&folded[0]);
    bz_output.clone_from_slice(&folded[1]);
    cz_output.clone_from_slice(&folded[2]);
    folded
}

/// Folds one evaluation table into preallocated storage.
/// Removes the active equality coordinate while leaving all sampled-coordinate
/// factors in the prover's single `bound_equality` scalar.
pub(crate) fn strip_equality_coordinate<F>(input: &[F], output: &mut [F], field_config: &F::Config)
where
    F: SpartanField,
{
    debug_assert_eq!(input.len(), 2 * output.len());

    #[cfg(feature = "parallel")]
    if should_parallelize(output.len()) {
        input
            .par_chunks_exact(2)
            .zip(output.par_iter_mut())
            .for_each(|(pair, value)| {
                *value = (field_config).add(&pair[0], &pair[1]);
            });
        return;
    }

    input
        .chunks_exact(2)
        .zip(output.iter_mut())
        .for_each(|(pair, value)| {
            *value = (field_config).add(&pair[0], &pair[1]);
        });
}

pub(crate) fn fold_product_tables<F>(
    input: &R1csProductTableBuffers<F>,
    output: &mut R1csProductTableBuffers<F>,
    challenge: &F,
    field_config: &F::Config,
) where
    F: SpartanField,
{
    debug_assert_eq!(input.len(), 2 * output.len());
    fold_table(&input.az, &mut output.az, challenge, &field_config);
    fold_table(&input.bz, &mut output.bz, challenge, &field_config);
    fold_table(&input.cz, &mut output.cz, challenge, &field_config);
}

/// Fused product-table fold and two-level equality accumulation for an active
/// low equality table. Each high bucket owns contiguous input/output ranges,
/// which keeps the parallel path allocation-free and race-free.
pub(crate) fn fold_products_and_compute_next_two_level<F, R>(
    input: &R1csProductTableBuffers<F>,
    output: &mut R1csProductTableBuffers<F>,
    challenge: &F,
    equality_weights: &StrippedEqualityWeights<'_, F>,
    endpoint: FactoredEndpoint,
    reducer: &R,
    field_config: &F::Config,
) -> Result<[F; 2], SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    let low_weights = equality_weights
        .low_weights()
        .expect("two-level accumulation requires stripped low weights");
    let low_pair_count = low_weights.len();
    let inner_reduction = reducer.prepare_reduce(low_pair_count);
    debug_assert_eq!(
        output.len() / 2,
        low_pair_count * equality_weights.high.len()
    );
    let input_values_per_high = 4 * low_pair_count;
    let output_values_per_high = 2 * low_pair_count;

    let accumulate_high_bucket = |mut outer: [<R as BatchMulAcc<F>>::Accumulator; 2],
                                  high_index: usize,
                                  az: &[F],
                                  bz: &[F],
                                  cz: &[F],
                                  az_output: &mut [F],
                                  bz_output: &mut [F],
                                  cz_output: &mut [F]|
     -> Result<_, SumcheckError> {
        let mut inner = std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero());
        for (low_pair_index, weight) in low_weights.iter().enumerate() {
            let input_start = 4 * low_pair_index;
            let output_start = 2 * low_pair_index;
            let [az, bz, cz] = fold_product_chunk(
                &az[input_start..input_start + 4],
                &bz[input_start..input_start + 4],
                &cz[input_start..input_start + 4],
                &mut az_output[output_start..output_start + 2],
                &mut bz_output[output_start..output_start + 2],
                &mut cz_output[output_start..output_start + 2],
                challenge,
                &field_config,
            );
            accumulate_eq_factored_cofactor_evaluations(
                &mut inner,
                weight,
                endpoint,
                &az[0],
                &az[1],
                &bz[0],
                &bz[1],
                &cz[0],
                &cz[1],
                reducer,
                &field_config,
            );
        }

        let inner = reduce_two_prepared(inner, &inner_reduction);
        let high_weight = &equality_weights.high[high_index];
        for (outer, inner) in outer.iter_mut().zip(&inner) {
            reducer.mul_acc(outer, high_weight, inner);
        }
        Ok(outer)
    };

    #[cfg(feature = "parallel")]
    if parallel_outer_fold(output.len() / 2) {
        let min_buckets = outer_fold_grain(output.len() / 2).div_ceil(low_pair_count);
        let accumulators = (
            input.az.par_chunks_exact(input_values_per_high),
            input.bz.par_chunks_exact(input_values_per_high),
            input.cz.par_chunks_exact(input_values_per_high),
            output.az.par_chunks_exact_mut(output_values_per_high),
            output.bz.par_chunks_exact_mut(output_values_per_high),
            output.cz.par_chunks_exact_mut(output_values_per_high),
        )
            .into_par_iter()
            .with_min_len(min_buckets)
            .enumerate()
            .try_fold(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |outer, (high_index, (az, bz, cz, az_output, bz_output, cz_output))| {
                    accumulate_high_bucket(
                        outer, high_index, az, bz, cz, az_output, bz_output, cz_output,
                    )
                },
            )
            .try_reduce(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |left, right| Ok(merge_accumulators(left, right)),
            )?;
        return reduce_two_accumulators_bounded(
            accumulators,
            reducer,
            equality_weights.high.len(),
            &field_config,
        );
    }

    let mut accumulators = std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero());
    for high_index in 0..equality_weights.high.len() {
        let input_start = high_index * input_values_per_high;
        let output_start = high_index * output_values_per_high;
        accumulators = accumulate_high_bucket(
            accumulators,
            high_index,
            &input.az[input_start..input_start + input_values_per_high],
            &input.bz[input_start..input_start + input_values_per_high],
            &input.cz[input_start..input_start + input_values_per_high],
            &mut output.az[output_start..output_start + output_values_per_high],
            &mut output.bz[output_start..output_start + output_values_per_high],
            &mut output.cz[output_start..output_start + output_values_per_high],
        )?;
    }
    reduce_two_accumulators_bounded(
        accumulators,
        reducer,
        equality_weights.high.len(),
        &field_config,
    )
}

/// Folds all product tables and accumulates the next round polynomial.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_products_and_compute_next<F, R>(
    input: &R1csProductTableBuffers<F>,
    output: &mut R1csProductTableBuffers<F>,
    challenge: &F,
    equality_weights: StrippedEqualityWeights<'_, F>,
    current_claim: &F,
    tau: &F,
    tau_inverse_or_zero: &F,
    bound_equality: &F,
    one: &F,
    reducer: &R,
    field_config: &F::Config,
) -> Result<[F; 3], SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    debug_assert_eq!(input.len(), 2 * output.len());
    let endpoint = FactoredEndpoint::for_tau(tau);

    if equality_weights
        .low_weights()
        .is_some_and(|low| low.len() >= TWO_LEVEL_EQUALITY_MIN_LOW_PAIRS)
    {
        let evaluations = fold_products_and_compute_next_two_level(
            input,
            output,
            challenge,
            &equality_weights,
            endpoint,
            reducer,
            &field_config,
        )?;
        return Ok(reconstruct_eq_factored_cubic_without_linear(
            current_claim,
            tau,
            tau_inverse_or_zero,
            endpoint,
            evaluations,
            bound_equality,
            one,
            &field_config,
        ));
    }

    let accumulate = |mut accumulators: [<R as BatchMulAcc<F>>::Accumulator; 2],
                      chunk: usize,
                      az: &[F],
                      bz: &[F],
                      cz: &[F],
                      az_output: &mut [F],
                      bz_output: &mut [F],
                      cz_output: &mut [F]| {
        let [az, bz, cz] = fold_product_chunk(
            az,
            bz,
            cz,
            az_output,
            bz_output,
            cz_output,
            challenge,
            &field_config,
        );
        let weight = equality_weights.pair_weight(chunk, &field_config);
        accumulate_eq_factored_cofactor_evaluations(
            &mut accumulators,
            &weight,
            endpoint,
            &az[0],
            &az[1],
            &bz[0],
            &bz[1],
            &cz[0],
            &cz[1],
            reducer,
            &field_config,
        );
        accumulators
    };

    let chunk_count = output.len() / 2;

    #[cfg(feature = "parallel")]
    if should_parallelize(chunk_count) {
        let accumulators = (
            input.az.par_chunks_exact(4),
            input.bz.par_chunks_exact(4),
            input.cz.par_chunks_exact(4),
            output.az.par_chunks_exact_mut(2),
            output.bz.par_chunks_exact_mut(2),
            output.cz.par_chunks_exact_mut(2),
        )
            .into_par_iter()
            .enumerate()
            .fold(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |accumulators, (chunk, (az, bz, cz, az_output, bz_output, cz_output))| {
                    accumulate(
                        accumulators,
                        chunk,
                        az,
                        bz,
                        cz,
                        az_output,
                        bz_output,
                        cz_output,
                    )
                },
            )
            .reduce(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |left, right| merge_accumulators(left, right),
            );
        let evaluations =
            reduce_two_accumulators_bounded(accumulators, reducer, chunk_count, &field_config)?;
        return Ok(reconstruct_eq_factored_cubic_without_linear(
            current_claim,
            tau,
            tau_inverse_or_zero,
            endpoint,
            evaluations,
            bound_equality,
            one,
            &field_config,
        ));
    }

    let mut accumulators = std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero());
    for chunk in 0..chunk_count {
        let input_start = 4 * chunk;
        let output_start = 2 * chunk;
        accumulators = accumulate(
            accumulators,
            chunk,
            &input.az[input_start..input_start + 4],
            &input.bz[input_start..input_start + 4],
            &input.cz[input_start..input_start + 4],
            &mut output.az[output_start..output_start + 2],
            &mut output.bz[output_start..output_start + 2],
            &mut output.cz[output_start..output_start + 2],
        );
    }
    let evaluations =
        reduce_two_accumulators_bounded(accumulators, reducer, chunk_count, &field_config)?;
    Ok(reconstruct_eq_factored_cubic_without_linear(
        current_claim,
        tau,
        tau_inverse_or_zero,
        endpoint,
        evaluations,
        bound_equality,
        one,
        &field_config,
    ))
}

/// Two-level equality tables and their reusable folding scratch.
/// Tables supplied by a caller must encode the challenge point in the same
/// coordinate order as the proof; only their dimensions can be validated here.
pub struct EqualityFactors<E> {
    low: Vec<E>,
    high: Vec<E>,
    scratch_low: Vec<E>,
    scratch_high: Vec<E>,
}
impl<E: SpartanField> EqualityFactors<E> {
    /// Validates dense-table metadata before taking ownership of their values.
    pub fn from_mles(
        (low, high): (DenseMultilinearExtension<E>, DenseMultilinearExtension<E>),
        field: &E::Config,
    ) -> Result<Self, SumcheckError> {
        if !has_dense_shape(&low) || !has_dense_shape(&high) {
            return Err(SumcheckError::InvalidEqualityDimensions);
        }
        Ok(Self::new(low.evaluations, high.evaluations, field))
    }
    pub fn new(low: Vec<E>, high: Vec<E>, field: &E::Config) -> Self {
        Self {
            scratch_low: vec![field.zero(); low.len() / 2],
            scratch_high: vec![field.zero(); high.len() / 2],
            low,
            high,
        }
    }
    pub(super) fn matches_rows(&self, rows: usize) -> bool {
        self.low.len().is_power_of_two()
            && self.high.len().is_power_of_two()
            && self.low.len().checked_mul(self.high.len()) == Some(rows)
    }
    pub(super) fn strip(&mut self, field: &E::Config) {
        let (active, scratch) = if self.low.len() > 1 {
            (&mut self.low, &mut self.scratch_low)
        } else {
            (&mut self.high, &mut self.scratch_high)
        };
        scratch.truncate(active.len() / 2);
        strip_equality_coordinate(active, scratch, field);
        core::mem::swap(active, scratch);
    }
    pub(super) fn weights(&self) -> StrippedEqualityWeights<'_, E> {
        if self.low.len() > 1 {
            StrippedEqualityWeights::low(&self.low, &self.high)
        } else {
            StrippedEqualityWeights::high(&self.high)
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn continue_field<
    E: SpartanField,
    R: BatchMulAcc<E> + Reduce<<R as BatchMulAcc<E>>::Accumulator, Output = E> + Sync,
>(
    transcript: &mut impl Transcript,
    field: &E::Config,
    reducer: &R,
    tau: &[E],
    factors: EqualityFactors<E>,
    products: R1csProductTableBuffers<E>,
    state: RoundState<E>,
    pending: Option<[E; 3]>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<OuterSumcheckOutput<E>, SumcheckError> {
    continue_field_with_inverses(
        transcript, field, reducer, tau, factors, products, state, pending, None, boundary,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn continue_field_with_inverses<
    E: SpartanField,
    R: BatchMulAcc<E> + Reduce<<R as BatchMulAcc<E>>::Accumulator, Output = E> + Sync,
>(
    transcript: &mut impl Transcript,
    field: &E::Config,
    reducer: &R,
    tau: &[E],
    mut factors: EqualityFactors<E>,
    mut products: R1csProductTableBuffers<E>,
    mut state: RoundState<E>,
    pending: Option<[E; 3]>,
    precomputed_inverses: Option<Vec<E>>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<OuterSumcheckOutput<E>, SumcheckError> {
    if products.len() > 1 {
        let inverses = precomputed_inverses.unwrap_or_else(|| batch_invert_nonzero(tau, field));
        let round = state.point.len();
        let coefficients = if let Some(pending) = pending {
            pending
        } else {
            factors.strip(field);
            compute_eq_factored_coefficients_without_linear(
                &products,
                factors.weights(),
                &state.claim,
                &tau[round],
                &inverses[round],
                &state.equality_scale,
                &field.one(),
                reducer,
                field,
            )?
        };
        let mut scratch = R1csProductTableBuffers::zeroed(products.len() / 2, field);
        state.continue_with(
            field,
            transcript,
            tau,
            coefficients,
            boundary,
            |round, challenge, state| {
                let len = products.len() / 2;
                scratch.truncate(len);
                let next = if len > 1 {
                    factors.strip(field);
                    fold_products_and_compute_next(
                        &products,
                        &mut scratch,
                        &challenge,
                        factors.weights(),
                        &state.claim,
                        &tau[round + 1],
                        &inverses[round + 1],
                        &state.equality_scale,
                        &field.one(),
                        reducer,
                        field,
                    )?
                } else {
                    fold_product_tables(&products, &mut scratch, &challenge, field);
                    [field.zero(); 3]
                };
                products.swap(&mut scratch);
                Ok(next)
            },
        )?;
    }
    state.finish(
        field,
        transcript,
        [products.az[0], products.bz[0], products.cz[0]],
    )
}
