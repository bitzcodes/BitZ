use super::super::univariate::{
    PrefixSkipK1, PrefixSkipK2, PrefixSkipK3, PrefixSkipK4, PrefixSkipSpec,
};
use field::{BatchMulAcc, MergeAccumulator, Reduce};
// Exact native-u32 arithmetic for the univariate-prefix outer-sumcheck skip.
//
// The native R1CS product tables keep `Az` and `Bz` as exact `u32` values and
// `Cz` as exact `u64` values.  For `K <= 4`, every interpolation of `Az` or
// `Bz` at the fixed exterior nodes fits in `i64`, while the interpolated `Cz`
// value and residual fit in `i128`.  Residuals are not projected into the
// field one at a time: their signed two-limb magnitudes feed the existing
// delayed linear accumulator and are reduced only at equality-factor
// boundaries.

use crate::piop::spartan::SpartanField as _;
use crate::piop::spartan::raw_monty::RawFieldStorage;
use crate::utils::delayed_reduction::EncodedMac;
#[cfg(test)]
use field::Uint;
use field::{Fp, RingOps};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use field::{CtMask, CtSelect};

use crate::poly::mle::DenseMultilinearExtension;

use crate::piop::spartan::{
    raw_monty::{NativeProducts, Raw, RawProducts},
    sumcheck::{R1csProductMles, SumcheckError, SumcheckLinearReducer},
};

type Field = Fp<2>;
type FieldConfig = field::FpCtx<2>;
type LinearAccumulator<R> = <R as SumcheckLinearReducer>::Accumulator;
type ProductAccumulator<R> = <R as BatchMulAcc<Field>>::Accumulator;

/// Checked standalone entry point for native skip-message arithmetic.
///
/// The complete native Spartan prover validates the same product tables once
/// at its public boundary and therefore calls
/// [`native_message_validated`] directly.
#[allow(dead_code)]
#[cfg(test)]
pub(crate) fn compute_native_message<R>(
    skip_vars: usize,
    equality_factors: &(
        DenseMultilinearExtension<Field>,
        DenseMultilinearExtension<Field>,
    ),
    products: &R1csProductMles<u64>,
    field_cfg: &FieldConfig,
    reducer: &R,
) -> Result<Vec<Field>, SumcheckError>
where
    R: SumcheckLinearReducer
        + BatchMulAcc<Field>
        + Reduce<<R as BatchMulAcc<Field>>::Accumulator, Output = Field>
        + Sync,
{
    validate_native_inputs_for_skip(skip_vars, products)?;
    validate_equality_factors(equality_factors, products.az.num_vars - skip_vars)?;
    native_message_validated(skip_vars, equality_factors, products, field_cfg, reducer)
}

/// Computes the native skip message in transcript order:
///
/// `Q(-1), Q(M), Q(-2), Q(M + 1), ..., Q(infinity)`.
///
/// The two equality factors are borrowed because the caller reuses them for
/// the ordinary cubic tail after the prefix has been folded. The native PIOP
/// must have validated product shape and `Az`/`Bz` width before calling this
/// hot-path entry point. Standalone callers use
/// [`native_message`].
#[cfg(test)]
pub(crate) fn native_message_validated<R>(
    skip_vars: usize,
    equality_factors: &(
        DenseMultilinearExtension<Field>,
        DenseMultilinearExtension<Field>,
    ),
    products: &R1csProductMles<u64>,
    field_cfg: &FieldConfig,
    reducer: &R,
) -> Result<Vec<Field>, SumcheckError>
where
    R: SumcheckLinearReducer
        + BatchMulAcc<Field>
        + Reduce<<R as BatchMulAcc<Field>>::Accumulator, Output = Field>
        + Sync,
{
    match skip_vars {
        1 => native_message_for::<PrefixSkipK1, 2, 1, R>(
            equality_factors,
            products,
            field_cfg,
            reducer,
            &FINITE_LAGRANGE_K1,
            &TOP_DIFFERENCE_K1,
        ),
        2 => native_message_for::<PrefixSkipK2, 4, 3, R>(
            equality_factors,
            products,
            field_cfg,
            reducer,
            &FINITE_LAGRANGE_K2,
            &TOP_DIFFERENCE_K2,
        ),
        3 => native_message_for::<PrefixSkipK3, 8, 7, R>(
            equality_factors,
            products,
            field_cfg,
            reducer,
            &FINITE_LAGRANGE_K3,
            &TOP_DIFFERENCE_K3,
        ),
        4 => native_message_for::<PrefixSkipK4, 16, 15, R>(
            equality_factors,
            products,
            field_cfg,
            reducer,
            &FINITE_LAGRANGE_K4,
            &TOP_DIFFERENCE_K4,
        ),
        _ => Err(SumcheckError::InvalidProductDimensions),
    }
}

/// Checked standalone entry point for native prefix folding.
///
/// The complete native Spartan prover has already validated these tables and
/// calls [`fold_native_lagrange_validated`] to avoid a second full
/// multiplicand scan.
#[allow(dead_code)]
#[cfg(test)]
pub(crate) fn fold_native_lagrange<R>(
    skip_vars: usize,
    products: R1csProductMles<u64>,
    challenge: &Field,
    field_cfg: &FieldConfig,
    reducer: &R,
) -> Result<R1csProductMles<Field>, SumcheckError>
where
    R: SumcheckLinearReducer,
{
    validate_native_inputs_for_skip(skip_vars, &products)?;
    fold_native_lagrange_validated(skip_vars, products, challenge, field_cfg, reducer)
}

/// Folds each contiguous native block `s + M * x` at the skip challenge.
///
/// The returned tables have exactly the remaining `n - K` Bit variables
/// and can be passed directly to the existing field-valued outer sumcheck.
/// Product shape and native multiplicand width must already be validated; use
/// [`fold_native_lagrange`] outside the complete PIOP path.
#[cfg(test)]
pub(crate) fn fold_native_lagrange_validated<R>(
    skip_vars: usize,
    products: R1csProductMles<u64>,
    challenge: &Field,
    field_cfg: &FieldConfig,
    reducer: &R,
) -> Result<R1csProductMles<Field>, SumcheckError>
where
    R: SumcheckLinearReducer,
{
    match skip_vars {
        1 => fold_native_lagrange_for::<PrefixSkipK1, 2, R>(
            products,
            challenge,
            field_cfg,
            reducer,
            &TOP_DIFFERENCE_K1,
        ),
        2 => fold_native_lagrange_for::<PrefixSkipK2, 4, R>(
            products,
            challenge,
            field_cfg,
            reducer,
            &TOP_DIFFERENCE_K2,
        ),
        3 => fold_native_lagrange_for::<PrefixSkipK3, 8, R>(
            products,
            challenge,
            field_cfg,
            reducer,
            &TOP_DIFFERENCE_K3,
        ),
        4 => fold_native_lagrange_for::<PrefixSkipK4, 16, R>(
            products,
            challenge,
            field_cfg,
            reducer,
            &TOP_DIFFERENCE_K4,
        ),
        _ => Err(SumcheckError::InvalidProductDimensions),
    }
}

/// The raw-table twin of [`fold_native_lagrange_validated`]: the same
/// per-block linear accumulation and reduction, with every folded entry stored
/// as a canonical residue for the raw outer tail.
pub(crate) fn fold_encoded_lagrange(
    skip_vars: usize,
    products: NativeProducts<'_>,
    challenge: &Field,
    ctx: &field::FpCtx<2>,
) -> Result<RawProducts, SumcheckError> {
    match skip_vars {
        1 => fold_encoded_lagrange_for::<PrefixSkipK1, 2>(
            products,
            challenge,
            ctx,
            &TOP_DIFFERENCE_K1,
        ),
        2 => fold_encoded_lagrange_for::<PrefixSkipK2, 4>(
            products,
            challenge,
            ctx,
            &TOP_DIFFERENCE_K2,
        ),
        3 => fold_encoded_lagrange_for::<PrefixSkipK3, 8>(
            products,
            challenge,
            ctx,
            &TOP_DIFFERENCE_K3,
        ),
        4 => fold_encoded_lagrange_for::<PrefixSkipK4, 16>(
            products,
            challenge,
            ctx,
            &TOP_DIFFERENCE_K4,
        ),
        _ => Err(SumcheckError::InvalidProductDimensions),
    }
}

fn fold_encoded_lagrange_for<S, const M: usize>(
    products: NativeProducts<'_>,
    challenge: &Field,
    ctx: &field::FpCtx<2>,
    top_difference: &[i64; M],
) -> Result<RawProducts, SumcheckError>
where
    S: PrefixSkipSpec<InterpolatedAB = i64, Residual = i128>,
{
    debug_assert_eq!(S::BLOCK_LEN, M);
    debug_assert!(products.len().is_multiple_of(M));
    debug_assert_eq!(products.bz.len(), products.len());
    debug_assert_eq!(products.cz.len(), products.len());
    let weights: Vec<Raw> = base_lagrange_weights(challenge, &ctx, top_difference)
        .iter()
        .map(|weight| ctx.raw(weight))
        .collect();

    let (az, bz, cz) = (products.az, products.bz, products.cz);
    let block_count = az.len() / M;
    let mut output = RawProducts::zeros(block_count);
    // Each block sum has at most 16 raw × u64 products, far below q · R, so
    // one Montgomery reduction plus a conversion multiply replaces the
    // Barrett remainder (same canonical residue).
    let fold_block = |az: &[u64], bz: &[u64], cz: &[u64]| -> [Raw; 3] {
        let mut accumulators = [field::FpLinearAcc::<2, 1>::default(); 3];
        for (((&weight, &az), &bz), &cz) in weights.iter().zip(az).zip(bz).zip(cz) {
            accumulators[0].accumulate_encoded(ctx, weight, az);
            accumulators[1].accumulate_encoded(ctx, weight, bz);
            accumulators[2].accumulate_encoded(ctx, weight, cz);
        }
        let [az, bz, cz] = accumulators;
        [
            ctx.plain_to_raw(ctx.redc_linear(&az)),
            ctx.plain_to_raw(ctx.redc_linear(&bz)),
            ctx.plain_to_raw(ctx.redc_linear(&cz)),
        ]
    };

    #[cfg(feature = "parallel")]
    if should_parallelize(block_count) {
        (
            output.az.par_iter_mut(),
            output.bz.par_iter_mut(),
            output.cz.par_iter_mut(),
            az.par_chunks_exact(M),
            bz.par_chunks_exact(M),
            cz.par_chunks_exact(M),
        )
            .into_par_iter()
            .for_each(|(az_out, bz_out, cz_out, az, bz, cz)| {
                let [folded_az, folded_bz, folded_cz] = fold_block(az, bz, cz);
                *az_out = folded_az;
                *bz_out = folded_bz;
                *cz_out = folded_cz;
            });
        return Ok(output);
    }

    for (index, ((az, bz), cz)) in az
        .chunks_exact(M)
        .zip(bz.chunks_exact(M))
        .zip(cz.chunks_exact(M))
        .enumerate()
    {
        let [folded_az, folded_bz, folded_cz] = fold_block(az, bz, cz);
        output.az[index] = folded_az;
        output.bz[index] = folded_bz;
        output.cz[index] = folded_cz;
    }
    Ok(output)
}

/// The raw twin of [`native_message_validated`]: the same
/// per-suffix exact interpolations and two-level `eq_out · Σ eq_in · residual`
/// accumulation on raw equality weights, in transcript order
/// `Q(-1), Q(M), Q(-2), Q(M + 1), ..., Q(infinity)`.
pub(crate) fn encoded_native_message(
    cfg: &FieldConfig,
    skip_vars: usize,
    eq_low: &[Raw],
    eq_high: &[Raw],
    products: NativeProducts<'_>,
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
) -> Result<Vec<Field>, SumcheckError> {
    match skip_vars {
        1 => skip_message_raw_for::<2, 1>(
            cfg,
            eq_low,
            eq_high,
            products,
            ctx,
            reducer,
            &FINITE_LAGRANGE_K1,
            &TOP_DIFFERENCE_K1,
        ),
        2 => skip_message_raw_for::<4, 3>(
            cfg,
            eq_low,
            eq_high,
            products,
            ctx,
            reducer,
            &FINITE_LAGRANGE_K2,
            &TOP_DIFFERENCE_K2,
        ),
        3 => skip_message_raw_for::<8, 7>(
            cfg,
            eq_low,
            eq_high,
            products,
            ctx,
            reducer,
            &FINITE_LAGRANGE_K3,
            &TOP_DIFFERENCE_K3,
        ),
        4 => skip_message_raw_for::<16, 15>(
            cfg,
            eq_low,
            eq_high,
            products,
            ctx,
            reducer,
            &FINITE_LAGRANGE_K4,
            &TOP_DIFFERENCE_K4,
        ),
        _ => Err(SumcheckError::InvalidProductDimensions),
    }
}

/// Branch-free `accumulators += (±weight) · |value|` with the magnitude split
/// into its two `u64` limbs (`[low, high]`), the raw twin of
/// [`accumulate_signed_i128`].
#[inline(always)]
fn accumulate_signed_i128_raw(
    ctx: &field::FpCtx<2>,
    accumulators: &mut [field::FpLinearAcc<2, 1>; 2],
    weight: Raw,
    negative_weight: Raw,
    value: i128,
) {
    let mask = (value >> 127) as u128;
    let magnitude = ((value as u128) ^ mask).wrapping_sub(mask);
    let selected = (weight & !mask) | (negative_weight & mask);
    accumulators[0].accumulate_encoded(ctx, selected, magnitude as u64);
    accumulators[1].accumulate_encoded(ctx, selected, (magnitude >> 64) as u64);
}

#[allow(clippy::too_many_arguments)]
fn skip_message_raw_for<const M: usize, const LANES: usize>(
    cfg: &FieldConfig,
    eq_low: &[Raw],
    eq_high: &[Raw],
    products: NativeProducts<'_>,
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    finite_lagrange: &[[i64; M]],
    top_difference: &[i64; M],
) -> Result<Vec<Field>, SumcheckError> {
    debug_assert_eq!(LANES, finite_lagrange.len() + 1);
    debug_assert_eq!(LANES, M - 1);
    let (az, bz, cz) = (products.az, products.bz, products.cz);
    let suffixes = eq_low
        .len()
        .checked_mul(eq_high.len())
        .ok_or(SumcheckError::InvalidEqualityDimensions)?;
    if az.len() != bz.len()
        || az.len() != cz.len()
        || az.len()
            != M.checked_mul(suffixes)
                .ok_or(SumcheckError::InvalidEqualityDimensions)?
        || eq_low.is_empty()
        || eq_high.is_empty()
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    let two_to_64 = ctx.native_residue_u128(1_u128 << 64);
    let negative_low: Vec<Raw> = eq_low.iter().map(|&weight| ctx.neg_raw(weight)).collect();
    let low_len = eq_low.len();

    let bucket = |high_index: usize| -> [field::FpProductAcc<2>; LANES] {
        let mut inner: [[field::FpLinearAcc<2, 1>; 2]; LANES] =
            std::array::from_fn(|_| [field::FpLinearAcc::<2, 1>::default(); 2]);
        let suffix_start = high_index * low_len;
        for (low_index, (&weight, &negative_weight)) in eq_low.iter().zip(&negative_low).enumerate()
        {
            let block_start = M * (suffix_start + low_index);
            let block_end = block_start + M;
            let az = &az[block_start..block_end];
            let bz = &bz[block_start..block_end];
            let cz = &cz[block_start..block_end];
            for (lane, coefficients) in finite_lagrange.iter().enumerate() {
                let az_at = interpolate_u32(az, coefficients);
                let bz_at = interpolate_u32(bz, coefficients);
                let cz_at = interpolate_u64(cz, coefficients);
                let residual = i128::from(az_at) * i128::from(bz_at) - cz_at;
                accumulate_signed_i128_raw(
                    ctx,
                    &mut inner[lane],
                    weight,
                    negative_weight,
                    residual,
                );
            }
            let az_top = interpolate_u32(az, top_difference);
            let bz_top = interpolate_u32(bz, top_difference);
            accumulate_signed_i128_raw(
                ctx,
                &mut inner[LANES - 1],
                weight,
                negative_weight,
                i128::from(az_top) * i128::from(bz_top),
            );
        }
        let high_weight = eq_high[high_index];
        let mut outer: [field::FpProductAcc<2>; LANES] =
            std::array::from_fn(|_| field::FpProductAcc::<2>::default());
        for (outer, [low, high]) in outer.iter_mut().zip(inner) {
            let low = low.reduce_encoded(reducer);
            let high = high.reduce_encoded(reducer);
            let value = ctx.add_raw(low, ctx.mul_raw(high, two_to_64));
            outer.accumulate_encoded(ctx, high_weight, value);
        }
        outer
    };
    let merge = |mut left: [field::FpProductAcc<2>; LANES],
                 right: [field::FpProductAcc<2>; LANES]| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    let zero = || std::array::from_fn(|_| field::FpProductAcc::<2>::default());

    #[cfg(feature = "parallel")]
    let outer = if should_parallelize(az.len() / M) {
        (0..eq_high.len())
            .into_par_iter()
            .map(bucket)
            .reduce(zero, merge)
    } else {
        (0..eq_high.len()).map(bucket).fold(zero(), merge)
    };
    #[cfg(not(feature = "parallel"))]
    let outer = (0..eq_high.len()).map(bucket).fold(zero(), merge);

    let mut message: Vec<Field> = outer
        .into_iter()
        .map(|accumulator| {
            crate::utils::delayed_reduction::element(&cfg, accumulator.reduce_encoded(reducer))
        })
        .collect();
    // Q(infinity) = (leading lane) / ((M - 1)!)², exactly as the generic kernel.
    let factorial = Field::from_with_cfg(factorial(M - 1), cfg);
    let inverse_factorial = *field::FieldOps::inverse_ct(cfg, &factorial).value();
    let infinity_scale = (cfg).mul(&inverse_factorial, &inverse_factorial);
    let infinity = message
        .last_mut()
        .expect("every supported prefix skip has an infinity lane");
    *infinity = cfg.mul(&(*infinity), &(&infinity_scale));
    Ok(message)
}

#[cfg(test)]
fn native_message_for<S, const M: usize, const LANES: usize, R>(
    equality_factors: &(
        DenseMultilinearExtension<Field>,
        DenseMultilinearExtension<Field>,
    ),
    products: &R1csProductMles<u64>,
    field_cfg: &FieldConfig,
    reducer: &R,
    finite_lagrange: &[[i64; M]],
    top_difference: &[i64; M],
) -> Result<Vec<Field>, SumcheckError>
where
    S: PrefixSkipSpec<InterpolatedAB = i64, Residual = i128>,
    R: SumcheckLinearReducer
        + BatchMulAcc<Field>
        + Reduce<<R as BatchMulAcc<Field>>::Accumulator, Output = Field>
        + Sync,
{
    debug_assert_eq!(S::BLOCK_LEN, M);
    debug_assert!(validate_native_inputs::<M>(products).is_ok());
    debug_assert!(
        validate_equality_factors(equality_factors, products.az.num_vars - S::SKIP_VARS).is_ok()
    );

    debug_assert_eq!(LANES, finite_lagrange.len() + 1);
    debug_assert_eq!(LANES, M - 1);
    let (eq_low, eq_high) = equality_factors;
    let low_weights = &eq_low.evaluations;
    let high_weights = &eq_high.evaluations;
    let zero = Field::zero_with_cfg(field_cfg);
    let two_to_64 = Field::from_with_cfg(1_u128 << 64, field_cfg);
    let negative_low_weights = low_weights
        .iter()
        .map(|weight| (field_cfg).sub(&zero, weight))
        .collect::<Vec<_>>();

    let accumulate_high = |outer: &mut [ProductAccumulator<R>; LANES], high_index: usize| {
        accumulate_high_bucket::<M, LANES, R>(
            outer,
            high_index,
            low_weights,
            &negative_low_weights,
            high_weights,
            products,
            finite_lagrange,
            top_difference,
            &two_to_64,
            reducer,
            &field_cfg,
        )
    };

    #[cfg(feature = "parallel")]
    let outer = if should_parallelize(products.az.evaluations.len() / M) {
        (0..high_weights.len())
            .into_par_iter()
            .try_fold(
                || product_accumulators::<LANES, R>(reducer),
                |mut outer, high_index| {
                    accumulate_high(&mut outer, high_index)?;
                    Ok::<_, SumcheckError>(outer)
                },
            )
            .try_reduce(
                || product_accumulators::<LANES, R>(reducer),
                |mut left, right| {
                    merge_product_accumulators(&mut left, right, reducer);
                    Ok::<_, SumcheckError>(left)
                },
            )?
    } else {
        let mut outer = product_accumulators::<LANES, R>(reducer);
        for high_index in 0..high_weights.len() {
            accumulate_high(&mut outer, high_index)?;
        }
        outer
    };

    #[cfg(not(feature = "parallel"))]
    let outer = {
        let mut outer = product_accumulators::<LANES, R>(reducer);
        for high_index in 0..high_weights.len() {
            accumulate_high(&mut outer, high_index)?;
        }
        outer
    };

    let mut message = outer
        .into_iter()
        .map(|accumulator| Reduce::reduce(reducer, accumulator))
        .collect::<Vec<_>>();

    // The last lane accumulated Delta^(M-1) A * Delta^(M-1) B.  Dividing by
    // ((M - 1)!)^2 converts it to the coefficient of Y^(2M - 2), i.e.
    // Q(infinity).  The denominator is nonzero for every supported field.
    let factorial = Field::from_with_cfg(factorial(M - 1), field_cfg);
    let inverse_factorial = *field::FieldOps::inverse_ct(field_cfg, &factorial).value();
    let infinity_scale = (field_cfg).mul(&inverse_factorial, &inverse_factorial);
    let infinity = message
        .last_mut()
        .expect("every supported prefix skip has an infinity lane");
    *infinity = field_cfg.mul(&(*infinity), &(&infinity_scale));
    Ok(message)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn accumulate_high_bucket<const M: usize, const LANES: usize, R>(
    outer: &mut [ProductAccumulator<R>; LANES],
    high_index: usize,
    low_weights: &[Field],
    negative_low_weights: &[Field],
    high_weights: &[Field],
    products: &R1csProductMles<u64>,
    finite_lagrange: &[[i64; M]],
    top_difference: &[i64; M],
    two_to_64: &Field,
    reducer: &R,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) -> Result<(), SumcheckError>
where
    R: SumcheckLinearReducer
        + BatchMulAcc<Field>
        + Reduce<<R as BatchMulAcc<Field>>::Accumulator, Output = Field>
        + Sync,
{
    debug_assert_eq!(LANES, finite_lagrange.len() + 1);
    let mut inner = linear_limb_accumulators::<LANES, R>(reducer);
    let low_len = low_weights.len();
    let suffix_start = high_index * low_len;

    for low_index in 0..low_len {
        let suffix = suffix_start + low_index;
        let block_start = M * suffix;
        let block_end = block_start + M;
        let az = &products.az.evaluations[block_start..block_end];
        let bz = &products.bz.evaluations[block_start..block_end];
        let cz = &products.cz.evaluations[block_start..block_end];
        let weight = &low_weights[low_index];
        let negative_weight = &negative_low_weights[low_index];

        for (lane, coefficients) in finite_lagrange.iter().enumerate() {
            let az_at = interpolate_u32(az, coefficients);
            let bz_at = interpolate_u32(bz, coefficients);
            let cz_at = interpolate_u64(cz, coefficients);
            let residual = i128::from(az_at) * i128::from(bz_at) - cz_at;
            accumulate_signed_i128(&mut inner[lane], weight, negative_weight, residual, reducer);
        }

        let az_top = interpolate_u32(az, top_difference);
        let bz_top = interpolate_u32(bz, top_difference);
        let infinity_numerator = i128::from(az_top) * i128::from(bz_top);
        accumulate_signed_i128(
            &mut inner[LANES - 1],
            weight,
            negative_weight,
            infinity_numerator,
            reducer,
        );
    }

    let high_weight = &high_weights[high_index];
    for (outer, limbs) in outer.iter_mut().zip(inner) {
        let [low, high] = limbs;
        let low = <R as SumcheckLinearReducer>::reduce(reducer, low, &field_config)?;
        let high = <R as SumcheckLinearReducer>::reduce(reducer, high, &field_config)?;
        let value = (field_config).add(&low, &(field_config).mul(&high, two_to_64));
        <R as BatchMulAcc<Field>>::mul_acc(reducer, outer, high_weight, &value);
    }
    Ok(())
}

#[cfg(test)]
fn fold_native_lagrange_for<S, const M: usize, R>(
    products: R1csProductMles<u64>,
    challenge: &Field,
    field_cfg: &FieldConfig,
    reducer: &R,
    top_difference: &[i64; M],
) -> Result<R1csProductMles<Field>, SumcheckError>
where
    S: PrefixSkipSpec<InterpolatedAB = i64, Residual = i128>,
    R: SumcheckLinearReducer,
{
    debug_assert_eq!(S::BLOCK_LEN, M);
    debug_assert!(validate_native_inputs::<M>(&products).is_ok());
    let remaining_vars = products.az.num_vars - S::SKIP_VARS;
    let weights = base_lagrange_weights(challenge, field_cfg, top_difference);

    let R1csProductMles { az, bz, cz } = products;
    let block_count = az.evaluations.len() / M;
    let zero = Field::zero_with_cfg(field_cfg);
    let mut folded_az = vec![zero; block_count];
    let mut folded_bz = vec![zero; block_count];
    let mut folded_cz = vec![zero; block_count];

    #[cfg(feature = "parallel")]
    if should_parallelize(block_count) {
        folded_az
            .par_iter_mut()
            .zip(folded_bz.par_iter_mut())
            .zip(folded_cz.par_iter_mut())
            .zip(
                az.evaluations
                    .par_chunks_exact(M)
                    .zip(bz.evaluations.par_chunks_exact(M))
                    .zip(cz.evaluations.par_chunks_exact(M)),
            )
            .try_for_each(|(((az_output, bz_output), cz_output), ((az, bz), cz))| {
                let [folded_az, folded_bz, folded_cz] =
                    fold_native_block(az, bz, cz, &weights, reducer, &field_cfg)?;
                *az_output = folded_az;
                *bz_output = folded_bz;
                *cz_output = folded_cz;
                Ok::<(), SumcheckError>(())
            })?;
    } else {
        for (index, ((az, bz), cz)) in az
            .evaluations
            .chunks_exact(M)
            .zip(bz.evaluations.chunks_exact(M))
            .zip(cz.evaluations.chunks_exact(M))
            .enumerate()
        {
            let [az, bz, cz] = fold_native_block(az, bz, cz, &weights, reducer, &field_cfg)?;
            folded_az[index] = az;
            folded_bz[index] = bz;
            folded_cz[index] = cz;
        }
    }

    #[cfg(not(feature = "parallel"))]
    for (index, ((az, bz), cz)) in az
        .evaluations
        .chunks_exact(M)
        .zip(bz.evaluations.chunks_exact(M))
        .zip(cz.evaluations.chunks_exact(M))
        .enumerate()
    {
        let [az, bz, cz] = fold_native_block(az, bz, cz, &weights, reducer, &field_cfg)?;
        folded_az[index] = az;
        folded_bz[index] = bz;
        folded_cz[index] = cz;
    }

    Ok(R1csProductMles {
        az: DenseMultilinearExtension {
            evaluations: folded_az,
            num_vars: remaining_vars,
        },
        bz: DenseMultilinearExtension {
            evaluations: folded_bz,
            num_vars: remaining_vars,
        },
        cz: DenseMultilinearExtension {
            evaluations: folded_cz,
            num_vars: remaining_vars,
        },
    })
}

#[cfg(test)]
fn fold_native_block<R>(
    az: &[u64],
    bz: &[u64],
    cz: &[u64],
    weights: &[Field],
    reducer: &R,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) -> Result<[Field; 3], SumcheckError>
where
    R: SumcheckLinearReducer,
{
    debug_assert_eq!(az.len(), weights.len());
    debug_assert_eq!(bz.len(), weights.len());
    debug_assert_eq!(cz.len(), weights.len());
    let mut accumulators: [LinearAccumulator<R>; 3] =
        std::array::from_fn(|_| <R as SumcheckLinearReducer>::accumulator_zero(reducer));

    for (((weight, az), bz), cz) in weights.iter().zip(az).zip(bz).zip(cz) {
        <R as SumcheckLinearReducer>::multiply_accumulate(
            reducer,
            &mut accumulators[0],
            weight,
            az,
        );
        <R as SumcheckLinearReducer>::multiply_accumulate(
            reducer,
            &mut accumulators[1],
            weight,
            bz,
        );
        <R as SumcheckLinearReducer>::multiply_accumulate(
            reducer,
            &mut accumulators[2],
            weight,
            cz,
        );
    }

    let [az, bz, cz] = accumulators;
    Ok([
        <R as SumcheckLinearReducer>::reduce(reducer, az, &field_config)?,
        <R as SumcheckLinearReducer>::reduce(reducer, bz, &field_config)?,
        <R as SumcheckLinearReducer>::reduce(reducer, cz, &field_config)?,
    ])
}

fn base_lagrange_weights<const M: usize>(
    challenge: &Field,
    field_cfg: &FieldConfig,
    top_difference: &[i64; M],
) -> Vec<Field> {
    let one = Field::one_with_cfg(field_cfg);
    let mut prefix = Vec::with_capacity(M + 1);
    prefix.push(one);
    for point in 0..M {
        let factor = (field_cfg).sub(challenge, &Field::from_with_cfg(point as u64, field_cfg));
        prefix.push((field_cfg).mul(
            prefix.last().expect("prefix always starts with one"),
            &factor,
        ));
    }

    let mut suffix = vec![one; M + 1];
    for point in (0..M).rev() {
        let factor = (field_cfg).sub(challenge, &Field::from_with_cfg(point as u64, field_cfg));
        suffix[point] = (field_cfg).mul(&factor, &suffix[point + 1]);
    }

    let factorial = Field::from_with_cfg(factorial(M - 1), field_cfg);
    let inverse_factorial = *field::FieldOps::inverse_ct(field_cfg, &factorial).value();
    (0..M)
        .map(|point| {
            // 1 / prod_{t != point}(point - t)
            //   = (-1)^(M - 1 - point) binom(M - 1, point) / (M - 1)!.
            let coefficient = Field::from_with_cfg(top_difference[point], field_cfg);
            let numerator = (field_cfg).mul(&prefix[point], &suffix[point + 1]);
            (field_cfg).mul(
                &(field_cfg).mul(&numerator, &coefficient),
                &inverse_factorial,
            )
        })
        .collect()
}

#[cfg(test)]
fn validate_native_inputs<const M: usize>(
    products: &R1csProductMles<u64>,
) -> Result<(), SumcheckError> {
    let num_vars = products.az.num_vars;
    let valid_shape = |mle: &DenseMultilinearExtension<u64>| {
        1usize
            .checked_shl(u32::try_from(mle.num_vars).unwrap_or(u32::MAX))
            .is_some_and(|expected| expected == mle.evaluations.len())
    };
    if !valid_shape(&products.az)
        || !valid_shape(&products.bz)
        || !valid_shape(&products.cz)
        || products.bz.num_vars != num_vars
        || products.cz.num_vars != num_vars
        || num_vars < M.ilog2() as usize
    {
        return Err(SumcheckError::InvalidProductDimensions);
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
    Ok(())
}

#[allow(dead_code)]
#[cfg(test)]
fn validate_native_inputs_for_skip(
    skip_vars: usize,
    products: &R1csProductMles<u64>,
) -> Result<(), SumcheckError> {
    match skip_vars {
        1 => validate_native_inputs::<2>(products),
        2 => validate_native_inputs::<4>(products),
        3 => validate_native_inputs::<8>(products),
        4 => validate_native_inputs::<16>(products),
        _ => Err(SumcheckError::InvalidProductDimensions),
    }
}

#[cfg(test)]
fn validate_equality_factors(
    equality_factors: &(
        DenseMultilinearExtension<Field>,
        DenseMultilinearExtension<Field>,
    ),
    expected_vars: usize,
) -> Result<(), SumcheckError> {
    let valid_shape = |mle: &DenseMultilinearExtension<Field>| {
        1usize
            .checked_shl(u32::try_from(mle.num_vars).unwrap_or(u32::MAX))
            .is_some_and(|expected| expected == mle.evaluations.len())
    };
    if !valid_shape(&equality_factors.0)
        || !valid_shape(&equality_factors.1)
        || equality_factors
            .0
            .num_vars
            .checked_add(equality_factors.1.num_vars)
            .is_none_or(|actual| actual != expected_vars)
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    Ok(())
}

#[inline]
fn interpolate_u32<const M: usize>(values: &[u64], coefficients: &[i64; M]) -> i64 {
    debug_assert_eq!(values.len(), M);
    values
        .iter()
        .zip(coefficients)
        .fold(0_i64, |sum, (&value, &coefficient)| {
            debug_assert!(value <= u64::from(u32::MAX));
            sum + coefficient * value as i64
        })
}

#[inline]
fn interpolate_u64<const M: usize>(values: &[u64], coefficients: &[i64; M]) -> i128 {
    debug_assert_eq!(values.len(), M);
    values
        .iter()
        .zip(coefficients)
        .fold(0_i128, |sum, (&value, &coefficient)| {
            sum + i128::from(coefficient) * i128::from(value)
        })
}

#[inline]
#[cfg(test)]
fn accumulate_signed_i128<R>(
    accumulators: &mut [LinearAccumulator<R>; 2],
    weight: &Field,
    negative_weight: &Field,
    value: i128,
    reducer: &R,
) where
    R: SumcheckLinearReducer,
{
    // Branch-free signed magnitude.  The exact K <= 4 interpolation bounds
    // keep every residual strictly above i128::MIN, but this also handles that
    // endpoint without overflow.
    let sign_mask = (value >> 127) as u128;
    let magnitude = ((value as u128) ^ sign_mask).wrapping_sub(sign_mask);
    let selected_weight =
        CtSelect::ct_select(weight, &negative_weight, CtMask::from_lsb(sign_mask as u64));
    let low = magnitude as u64;
    let high = (magnitude >> 64) as u64;
    <R as SumcheckLinearReducer>::multiply_accumulate(
        reducer,
        &mut accumulators[0],
        &selected_weight,
        &low,
    );
    <R as SumcheckLinearReducer>::multiply_accumulate(
        reducer,
        &mut accumulators[1],
        &selected_weight,
        &high,
    );
}

#[cfg(test)]
fn linear_limb_accumulators<const LANES: usize, R>(
    reducer: &R,
) -> [[LinearAccumulator<R>; 2]; LANES]
where
    R: SumcheckLinearReducer,
{
    std::array::from_fn(|_| {
        std::array::from_fn(|_| <R as SumcheckLinearReducer>::accumulator_zero(reducer))
    })
}

#[cfg(test)]
fn product_accumulators<const LANES: usize, R>(reducer: &R) -> [ProductAccumulator<R>; LANES]
where
    R: BatchMulAcc<Field> + Reduce<<R as BatchMulAcc<Field>>::Accumulator, Output = Field> + Sync,
{
    std::array::from_fn(|_| <R as BatchMulAcc<Field>>::Accumulator::zero())
}

#[cfg(feature = "parallel")]
#[cfg(test)]
fn merge_product_accumulators<const LANES: usize, R>(
    left: &mut [ProductAccumulator<R>; LANES],
    right: [ProductAccumulator<R>; LANES],
    reducer: &R,
) where
    R: BatchMulAcc<Field> + Reduce<<R as BatchMulAcc<Field>>::Accumulator, Output = Field> + Sync,
{
    debug_assert_eq!(left.len(), right.len());
    for (left, right) in left.iter_mut().zip(right) {
        left.merge_assign(&right);
    }
}

#[cfg(feature = "parallel")]
#[inline]
fn should_parallelize(work_items: usize) -> bool {
    work_items >= (1 << 12) && rayon::current_num_threads() > 1
}

#[inline]
const fn factorial(value: usize) -> u64 {
    let mut result = 1_u64;
    let mut factor = 2_usize;
    while factor <= value {
        result *= factor as u64;
        factor += 1;
    }
    result
}

// L_s(lambda) for D = {0, ..., M - 1}.  Rows follow the balanced exterior
// order used by the transcript.  Keeping these exact signed tables static
// avoids coefficient generation and division in the prover hot path.
const FINITE_LAGRANGE_K1: [[i64; 2]; 0] = [];
const FINITE_LAGRANGE_K2: [[i64; 4]; 2] = [[4, -6, 4, -1], [-1, 4, -6, 4]];
const FINITE_LAGRANGE_K3: [[i64; 8]; 6] = [
    [8, -28, 56, -70, 56, -28, 8, -1],
    [-1, 8, -28, 56, -70, 56, -28, 8],
    [36, -168, 378, -504, 420, -216, 63, -8],
    [-8, 63, -216, 420, -504, 378, -168, 36],
    [120, -630, 1512, -2100, 1800, -945, 280, -36],
    [-36, 280, -945, 1800, -2100, 1512, -630, 120],
];
const FINITE_LAGRANGE_K4: [[i64; 16]; 14] = [
    [
        16, -120, 560, -1820, 4368, -8008, 11440, -12870, 11440, -8008, 4368, -1820, 560, -120, 16,
        -1,
    ],
    [
        -1, 16, -120, 560, -1820, 4368, -8008, 11440, -12870, 11440, -8008, 4368, -1820, 560, -120,
        16,
    ],
    [
        136, -1360, 7140, -24752, 61880, -116688, 170170, -194480, 175032, -123760, 68068, -28560,
        8840, -1904, 255, -16,
    ],
    [
        -16, 255, -1904, 8840, -28560, 68068, -123760, 175032, -194480, 170170, -116688, 61880,
        -24752, 7140, -1360, 136,
    ],
    [
        816, -9180, 51408, -185640, 477360, -918918, 1361360, -1575288, 1432080, -1021020, 565488,
        -238680, 74256, -16065, 2160, -136,
    ],
    [
        -136, 2160, -16065, 74256, -238680, 565488, -1021020, 1432080, -1575288, 1361360, -918918,
        477360, -185640, 51408, -9180, 816,
    ],
    [
        3876, -46512, 271320, -1007760, 2645370, -5173168, 7759752, -9069840, 8314020, -5969040,
        3325608, -1410864, 440895, -95760, 12920, -816,
    ],
    [
        -816, 12920, -95760, 440895, -1410864, 3325608, -5969040, 8314020, -9069840, 7759752,
        -5173168, 2645370, -1007760, 271320, -46512, 3876,
    ],
    [
        15504, -193800, 1162800, -4408950, 11757200, -23279256, 35271600, -41570100, 38372400,
        -27713400, 15519504, -6613425, 2074800, -452200, 61200, -3876,
    ],
    [
        -3876, 61200, -452200, 2074800, -6613425, 15519504, -27713400, 38372400, -41570100,
        35271600, -23279256, 11757200, -4408950, 1162800, -193800, 15504,
    ],
    [
        54264, -697680, 4273290, -16460080, 44442216, -88884432, 135795660, -161164080, 149652360,
        -108636528, 61108047, -26142480, 8230040, -1799280, 244188, -15504,
    ],
    [
        -15504, 244188, -1799280, 8230040, -26142480, 61108047, -108636528, 149652360, -161164080,
        135795660, -88884432, 44442216, -16460080, 4273290, -697680, 54264,
    ],
    [
        170544, -2238390, 13927760, -54318264, 148140720, -298750452, 459616080, -548725320,
        512143632, -373438065, 210882672, -90530440, 28588560, -6267492, 852720, -54264,
    ],
    [
        -54264, 852720, -6267492, 28588560, -90530440, 210882672, -373438065, 512143632,
        -548725320, 459616080, -298750452, 148140720, -54318264, 13927760, -2238390, 170544,
    ],
];

const TOP_DIFFERENCE_K1: [i64; 2] = [-1, 1];
const TOP_DIFFERENCE_K2: [i64; 4] = [-1, 3, -3, 1];
const TOP_DIFFERENCE_K3: [i64; 8] = [-1, 7, -21, 35, -35, 21, -7, 1];
const TOP_DIFFERENCE_K4: [i64; 16] = [
    -1, 15, -105, 455, -1365, 3003, -5005, 6435, -6435, 5005, -3003, 1365, -455, 105, -15, 1,
];

#[cfg(test)]
mod tests {

    use super::*;
    use crate::piop::spartan::{
        make_equality_factors,
        sumcheck::BigUintSumcheckOracle,
        univariate_skip::{field_message, fold_field_lagrange},
    };

    const TEST_MODULUS: u128 = (1_u128 << 100) - 15;
    const OTHER_TEST_MODULUS: u128 = (1_u128 << 127) - 1;

    #[test]
    fn fixed_lagrange_tables_interpolate_all_basis_monomials() {
        check_finite_lagrange::<2>(&FINITE_LAGRANGE_K1, &[]);
        check_finite_lagrange::<4>(&FINITE_LAGRANGE_K2, &[-1, 4]);
        check_finite_lagrange::<8>(&FINITE_LAGRANGE_K3, &[-1, 8, -2, 9, -3, 10]);
        check_finite_lagrange::<16>(
            &FINITE_LAGRANGE_K4,
            &[-1, 16, -2, 17, -3, 18, -4, 19, -5, 20, -6, 21, -7, 22],
        );
    }

    #[test]
    fn top_difference_tables_select_the_leading_coefficient() {
        check_top_difference(&TOP_DIFFERENCE_K1);
        check_top_difference(&TOP_DIFFERENCE_K2);
        check_top_difference(&TOP_DIFFERENCE_K3);
        check_top_difference(&TOP_DIFFERENCE_K4);
    }

    #[test]
    fn k4_u32_interpolations_fit_the_declared_i64_layout() {
        let maximum = u64::from(u32::MAX);
        for coefficients in FINITE_LAGRANGE_K4
            .iter()
            .chain(std::iter::once(&TOP_DIFFERENCE_K4))
        {
            let positive = coefficients
                .iter()
                .map(|&coefficient| coefficient.max(0) as i128 * i128::from(maximum))
                .sum::<i128>();
            let negative = coefficients
                .iter()
                .map(|&coefficient| coefficient.min(0) as i128 * i128::from(maximum))
                .sum::<i128>();
            assert!(positive <= i128::from(i64::MAX));
            assert!(negative >= i128::from(i64::MIN));
        }
    }

    #[test]
    fn native_messages_and_folds_match_the_field_oracle_for_every_k() {
        for modulus in [TEST_MODULUS, OTHER_TEST_MODULUS] {
            check_native_messages_and_folds(modulus);
        }
    }

    #[test]
    fn checked_entry_points_reject_wide_native_multiplicands() {
        let field_cfg = Fp::<2>::make_cfg(&Uint::from(TEST_MODULUS)).unwrap();
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let tau_tail = [
            Fp::<2>::from_with_cfg(17_u64, &field_cfg),
            Fp::<2>::from_with_cfg(29_u64, &field_cfg),
        ];
        let equality_factors = make_equality_factors(&tau_tail, &field_cfg).unwrap();
        let challenge = Fp::<2>::from_with_cfg(43_u64, &field_cfg);
        let mut products = patterned_native_products(3);
        products.az.evaluations[0] = u64::from(u32::MAX) + 1;

        assert_eq!(
            compute_native_message(1, &equality_factors, &products, &field_cfg, &reducer,),
            Err(SumcheckError::NativeMultiplicandOutOfRange)
        );
        assert_eq!(
            fold_native_lagrange(1, products, &challenge, &field_cfg, &reducer),
            Err(SumcheckError::NativeMultiplicandOutOfRange)
        );
    }

    fn check_native_messages_and_folds(modulus: u128) {
        let field_cfg = Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let reference_reducer = BigUintSumcheckOracle::new(&field_cfg).unwrap();
        let tau_tail = [
            Fp::<2>::from_with_cfg(17_u64, &field_cfg),
            Fp::<2>::from_with_cfg(29_u64, &field_cfg),
        ];
        let equality_factors = make_equality_factors(&tau_tail, &field_cfg).unwrap();
        let challenge = Fp::<2>::from_with_cfg(43_u64, &field_cfg);

        for skip_vars in 1..=4 {
            let native_products = patterned_native_products(skip_vars + tau_tail.len());
            let field_products = project_products(&native_products, &field_cfg);
            let native_message = compute_native_message(
                skip_vars,
                &equality_factors,
                &native_products,
                &field_cfg,
                &reducer,
            )
            .unwrap();
            let reference_message = compute_native_message(
                skip_vars,
                &equality_factors,
                &native_products,
                &field_cfg,
                &reference_reducer,
            )
            .unwrap();
            let field_message = field_message(
                skip_vars,
                (&equality_factors.0, &equality_factors.1),
                &field_products,
                &field_cfg,
            )
            .unwrap();
            assert_eq!(
                native_message, field_message,
                "message mismatch for K={skip_vars}"
            );
            assert_eq!(
                native_message, reference_message,
                "reducer mismatch for K={skip_vars}"
            );

            let native_fold = fold_native_lagrange(
                skip_vars,
                native_products.clone(),
                &challenge,
                &field_cfg,
                &reducer,
            )
            .unwrap();
            let reference_fold = fold_native_lagrange(
                skip_vars,
                native_products,
                &challenge,
                &field_cfg,
                &reference_reducer,
            )
            .unwrap();
            let field_fold =
                fold_field_lagrange(field_products, skip_vars, &challenge, &field_cfg).unwrap();
            assert_eq!(native_fold, field_fold, "fold mismatch for K={skip_vars}");
            assert_eq!(
                native_fold, reference_fold,
                "fold reducer mismatch for K={skip_vars}"
            );
        }
    }

    fn check_finite_lagrange<const M: usize>(tables: &[[i64; M]], nodes: &[i128]) {
        assert_eq!(tables.len(), nodes.len());
        for (coefficients, &node) in tables.iter().zip(nodes) {
            for degree in 0..M {
                let interpolated = coefficients
                    .iter()
                    .enumerate()
                    .map(|(point, &coefficient)| {
                        i128::from(coefficient) * (point as i128).pow(degree as u32)
                    })
                    .sum::<i128>();
                assert_eq!(interpolated, node.pow(degree as u32));
            }
        }
    }

    fn check_top_difference<const M: usize>(coefficients: &[i64; M]) {
        for degree in 0..M {
            let selected = coefficients
                .iter()
                .enumerate()
                .map(|(point, &coefficient)| {
                    i128::from(coefficient) * (point as i128).pow(degree as u32)
                })
                .sum::<i128>();
            let expected = if degree == M - 1 {
                i128::from(factorial(M - 1))
            } else {
                0
            };
            assert_eq!(selected, expected);
        }
    }

    fn patterned_native_products(num_vars: usize) -> R1csProductMles<u64> {
        let len = 1usize << num_vars;
        let az = (0..len)
            .map(|index| match index % 4 {
                0 => 0,
                1 => 1,
                2 => u64::from(u32::MAX),
                _ => u64::from((index as u32).wrapping_mul(0x9e37_79b9)),
            })
            .collect::<Vec<_>>();
        let bz = (0..len)
            .map(|index| match index % 4 {
                0 => u64::from(u32::MAX),
                1 => 7,
                2 => 1,
                _ => u64::from((index as u32).rotate_left(13) ^ 0xa5a5_5a5a),
            })
            .collect::<Vec<_>>();
        let cz = az
            .iter()
            .zip(&bz)
            .enumerate()
            .map(|(index, (&az, &bz))| match index % 3 {
                0 => az * bz,
                1 => u64::MAX,
                _ => 0,
            })
            .collect::<Vec<_>>();
        R1csProductMles {
            az: dense(az, num_vars),
            bz: dense(bz, num_vars),
            cz: dense(cz, num_vars),
        }
    }

    fn project_products(
        products: &R1csProductMles<u64>,
        field_cfg: &FieldConfig,
    ) -> R1csProductMles<Field> {
        let project = |mle: &DenseMultilinearExtension<u64>| DenseMultilinearExtension {
            evaluations: mle
                .evaluations
                .iter()
                .map(|&value| Field::from_with_cfg(value, field_cfg))
                .collect(),
            num_vars: mle.num_vars,
        };
        R1csProductMles {
            az: project(&products.az),
            bz: project(&products.bz),
            cz: project(&products.cz),
        }
    }

    fn dense<T>(evaluations: Vec<T>, num_vars: usize) -> DenseMultilinearExtension<T> {
        DenseMultilinearExtension {
            evaluations,
            num_vars,
        }
    }
}
