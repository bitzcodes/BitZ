//! Native first folds and fused two-limb arithmetic for the shared outer engine.
#[cfg(test)]
use super::engine::RoundState;
#[cfg(test)]
use crate::piop::spartan::SpartanField as _;
use crate::piop::spartan::raw_monty::*;
use crate::piop::spartan::sumcheck::R1csProductMles;
#[cfg(test)]
use crate::sumcheck::SumcheckError;
use crate::sumcheck::outer::ordinary::*;
#[cfg(test)]
use crate::sumcheck::{boundary::*, proof::*};
#[cfg(test)]
use crate::transcript::traits::Transcript;
#[cfg(test)]
use crate::utils::delayed_reduction::EncodedMac;
#[cfg(test)]
use field::RingOps;

#[cfg(all(test, feature = "parallel"))]
use rayon::prelude::*;
mod native;
#[cfg(test)]
pub(crate) use native::NativeInput;
pub use native::NativeWideProducts;
#[cfg(test)]
pub(super) use native::legacy_dispatch as prove_native_reference;
/// Removes the current Boolean coordinate by adding adjacent equality weights.
#[cfg(test)]
pub(crate) fn strip_coordinate_raw(ctx: &field::FpCtx<2>, input: &[Raw]) -> Vec<Raw> {
    debug_assert!(input.len() >= 2 && input.len().is_power_of_two());
    #[cfg(feature = "parallel")]
    if parallel(input.len() / 2) {
        return input
            .par_chunks_exact(2)
            .map(|pair| ctx.add_raw(pair[0], pair[1]))
            .collect();
    }
    input
        .chunks_exact(2)
        .map(|pair| ctx.add_raw(pair[0], pair[1]))
        .collect()
}

/// Equality weights with the active coordinate stripped: the pair weight is
/// `low[pair % low.len()] · high[pair / low.len()]`, or `high[pair]` once the
/// low coordinates are exhausted.
#[derive(Clone, Copy)]
#[cfg(test)]
pub(crate) struct RawEqWeights<'a> {
    low: Option<&'a [Raw]>,
    high: &'a [Raw],
}

#[cfg(test)]
impl<'a> RawEqWeights<'a> {
    #[inline(always)]
    fn pair_weight(&self, ctx: &field::FpCtx<2>, pair: usize) -> Raw {
        match self.low {
            Some(low) => ctx.mul_raw(low[pair % low.len()], self.high[pair / low.len()]),
            None => self.high[pair],
        }
    }

    /// The low table when the two-level bucket decomposition applies.
    #[inline]
    fn two_level_low(&self) -> Option<&'a [Raw]> {
        self.low
            .filter(|low| low.len() >= TWO_LEVEL_EQUALITY_MIN_LOW_PAIRS)
    }

    fn pair_count(&self) -> usize {
        self.low.map_or(1, <[Raw]>::len) * self.high.len()
    }
}

// ---------------------------------------------------------------------------
// Outer sumcheck
// ---------------------------------------------------------------------------

/// Borrowed exact native `Az`, `Bz`, `Cz` tables of one common power-of-two
/// length (the row domain), read by the first outer round and the prefix
/// skip without copying the relation's witness.
#[derive(Clone, Copy)]
pub(crate) struct NativeProducts<'a> {
    pub az: &'a [u64],
    pub bz: &'a [u64],
    pub cz: &'a [u64],
}

impl<'a> NativeProducts<'a> {
    pub(crate) fn from_mles(products: &'a R1csProductMles<u64>) -> Self {
        Self {
            az: &products.az.evaluations,
            bz: &products.bz.evaluations,
            cz: &products.cz.evaluations,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.az.len()
    }
}

/// Owned raw `Az`, `Bz`, `Cz` tables.
#[cfg(test)]
pub struct RawProducts {
    pub az: Vec<Raw>,
    pub bz: Vec<Raw>,
    pub cz: Vec<Raw>,
}

#[cfg(test)]
impl RawProducts {
    pub(crate) fn zeros(len: usize) -> Self {
        Self {
            az: vec![0; len],
            bz: vec![0; len],
            cz: vec![0; len],
        }
    }

    pub(crate) fn from_field(ctx: &field::FpCtx<2>, products: &R1csProductMles<Field>) -> Self {
        Self {
            az: ctx.raw_vec(&products.az.evaluations),
            bz: ctx.raw_vec(&products.bz.evaluations),
            cz: ctx.raw_vec(&products.cz.evaluations),
        }
    }

    pub(crate) fn len(&self) -> usize {
        debug_assert_eq!(self.az.len(), self.bz.len());
        debug_assert_eq!(self.az.len(), self.cz.len());
        self.az.len()
    }

    fn truncate(&mut self, len: usize) {
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

#[inline(always)]
#[cfg(test)]
pub(crate) fn accumulate_cofactor_raw(
    ctx: &field::FpCtx<2>,
    accumulators: &mut ProductPair,
    weight: Raw,
    endpoint: FactoredEndpoint,
    az0: Raw,
    az1: Raw,
    bz0: Raw,
    bz1: Raw,
    cz0: Raw,
    cz1: Raw,
) {
    let residual = match endpoint {
        FactoredEndpoint::Zero => ctx.sub_raw(ctx.mul_raw(az0, bz0), cz0),
        FactoredEndpoint::One => ctx.sub_raw(ctx.mul_raw(az1, bz1), cz1),
        FactoredEndpoint::KnownZero => 0,
    };
    if endpoint != FactoredEndpoint::KnownZero {
        accumulators[0].accumulate_encoded(ctx, weight, residual);
    }
    let infinity = ctx.mul_raw(ctx.sub_raw(az1, az0), ctx.sub_raw(bz1, bz0));
    accumulators[1].accumulate_encoded(ctx, weight, infinity);
}

/// Branch-free `accumulator += (±weight) · |value|` for `|value| ≤ u64::MAX`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
#[cfg(test)]
pub(crate) fn accumulate_native_cofactor_raw(
    ctx: &field::FpCtx<2>,
    accumulators: &mut LinearPair,
    weight: Raw,
    negative_weight: Raw,
    endpoint: FactoredEndpoint,
    az0: u64,
    az1: u64,
    bz0: u64,
    bz1: u64,
    cz0: u64,
    cz1: u64,
) {
    let (az_e, bz_e, cz_e) = match endpoint {
        FactoredEndpoint::Zero => (az0, bz0, cz0),
        FactoredEndpoint::One => (az1, bz1, cz1),
        FactoredEndpoint::KnownZero => (0, 0, 0),
    };
    // Multiplicands are validated 32-bit wide, so the product is a u64 and the
    // residual magnitude is at most u64::MAX; each delta is below 2^32.
    let residual = i128::from(az_e * bz_e) - i128::from(cz_e);
    let infinity = (i128::from(az1) - i128::from(az0)) * (i128::from(bz1) - i128::from(bz0));
    if endpoint != FactoredEndpoint::KnownZero {
        accumulate_signed_raw(ctx, &mut accumulators[0], weight, negative_weight, residual);
    }
    accumulate_signed_raw(ctx, &mut accumulators[1], weight, negative_weight, infinity);
}

/// `[endpoint evaluation, leading coefficient]` of the cofactor over the
/// current raw product tables.
#[cfg(test)]
pub(crate) fn cofactor_evaluations_raw(
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    products: &RawProducts,
    weights: RawEqWeights<'_>,
    endpoint: FactoredEndpoint,
) -> [Raw; 2] {
    let pair_count = products.len() / 2;
    debug_assert_eq!(pair_count, weights.pair_count());

    if let Some(low) = weights.two_level_low() {
        let low_pairs = low.len();
        let bucket = |high_index: usize| -> ProductPair {
            let mut inner = product_pair();
            let start = 2 * high_index * low_pairs;
            for (low_index, &weight) in low.iter().enumerate() {
                let index = start + 2 * low_index;
                accumulate_cofactor_raw(
                    ctx,
                    &mut inner,
                    weight,
                    endpoint,
                    products.az[index],
                    products.az[index + 1],
                    products.bz[index],
                    products.bz[index + 1],
                    products.cz[index],
                    products.cz[index + 1],
                );
            }
            let inner = reduce_product_pair(inner, reducer);
            let high_weight = weights.high[high_index];
            let mut outer = product_pair();
            outer[0].accumulate_encoded(ctx, high_weight, inner[0]);
            outer[1].accumulate_encoded(ctx, high_weight, inner[1]);
            outer
        };
        #[cfg(feature = "parallel")]
        if parallel(pair_count) {
            let total = (0..weights.high.len())
                .into_par_iter()
                .map(bucket)
                .reduce(product_pair, merge_product_pair);
            return reduce_product_pair(total, reducer);
        }
        let total = (0..weights.high.len())
            .map(bucket)
            .fold(product_pair(), merge_product_pair);
        return reduce_product_pair(total, reducer);
    }

    let block = |start: usize, end: usize| -> ProductPair {
        let mut accumulators = product_pair();
        for pair in start..end {
            let index = 2 * pair;
            let weight = weights.pair_weight(ctx, pair);
            accumulate_cofactor_raw(
                ctx,
                &mut accumulators,
                weight,
                endpoint,
                products.az[index],
                products.az[index + 1],
                products.bz[index],
                products.bz[index + 1],
                products.cz[index],
                products.cz[index + 1],
            );
        }
        accumulators
    };
    #[cfg(feature = "parallel")]
    if parallel(pair_count) {
        let blocks = pair_count.div_ceil(FOLD_BLOCK);
        let total = (0..blocks)
            .into_par_iter()
            .map(|b| block(b * FOLD_BLOCK, ((b + 1) * FOLD_BLOCK).min(pair_count)))
            .reduce(product_pair, merge_product_pair);
        return reduce_product_pair(total, reducer);
    }
    reduce_product_pair(block(0, pair_count), reducer)
}

/// The native (exact `u64`) twin of [`cofactor_evaluations_raw`] for the first
/// outer round: field × `u64` accumulation, one reduction per bucket.
#[cfg(test)]
pub(crate) fn native_cofactor_evaluations_raw(
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    products: NativeProducts<'_>,
    weights: RawEqWeights<'_>,
    endpoint: FactoredEndpoint,
) -> [Raw; 2] {
    let (az, bz, cz) = (products.az, products.bz, products.cz);
    let pair_count = az.len() / 2;
    debug_assert_eq!(pair_count, weights.pair_count());

    if let Some(low) = weights.two_level_low() {
        let negative_low: Vec<Raw> = low.iter().map(|&weight| ctx.neg_raw(weight)).collect();
        let low_pairs = low.len();
        let bucket = |high_index: usize| -> ProductPair {
            let mut inner = linear_pair();
            let start = 2 * high_index * low_pairs;
            for (low_index, (&weight, &negative_weight)) in
                low.iter().zip(&negative_low).enumerate()
            {
                let index = start + 2 * low_index;
                accumulate_native_cofactor_raw(
                    ctx,
                    &mut inner,
                    weight,
                    negative_weight,
                    endpoint,
                    az[index],
                    az[index + 1],
                    bz[index],
                    bz[index + 1],
                    cz[index],
                    cz[index + 1],
                );
            }
            let inner = reduce_linear_pair(inner, reducer);
            let high_weight = weights.high[high_index];
            let mut outer = product_pair();
            outer[0].accumulate_encoded(ctx, high_weight, inner[0]);
            outer[1].accumulate_encoded(ctx, high_weight, inner[1]);
            outer
        };
        #[cfg(feature = "parallel")]
        if parallel(pair_count) {
            let total = (0..weights.high.len())
                .into_par_iter()
                .map(bucket)
                .reduce(product_pair, merge_product_pair);
            return reduce_product_pair(total, reducer);
        }
        let total = (0..weights.high.len())
            .map(bucket)
            .fold(product_pair(), merge_product_pair);
        return reduce_product_pair(total, reducer);
    }

    let block = |start: usize, end: usize| -> LinearPair {
        let mut accumulators = linear_pair();
        for pair in start..end {
            let index = 2 * pair;
            let weight = weights.pair_weight(ctx, pair);
            accumulate_native_cofactor_raw(
                ctx,
                &mut accumulators,
                weight,
                ctx.neg_raw(weight),
                endpoint,
                az[index],
                az[index + 1],
                bz[index],
                bz[index + 1],
                cz[index],
                cz[index + 1],
            );
        }
        accumulators
    };
    let merge = |mut left: LinearPair, right: LinearPair| -> LinearPair {
        left[0] += right[0];
        left[1] += right[1];
        left
    };
    #[cfg(feature = "parallel")]
    if parallel(pair_count) {
        let blocks = pair_count.div_ceil(FOLD_BLOCK);
        let total = (0..blocks)
            .into_par_iter()
            .map(|b| block(b * FOLD_BLOCK, ((b + 1) * FOLD_BLOCK).min(pair_count)))
            .reduce(linear_pair, merge);
        return reduce_linear_pair(total, reducer);
    }
    reduce_linear_pair(block(0, pair_count), reducer)
}

/// Raw slices of one product table triple, for disjoint parallel blocks.
#[cfg(test)]
pub(crate) struct ProductSlices<'a> {
    az: &'a [Raw],
    bz: &'a [Raw],
    cz: &'a [Raw],
}

#[cfg(test)]
pub(crate) struct ProductSlicesMut<'a> {
    az: &'a mut [Raw],
    bz: &'a mut [Raw],
    cz: &'a mut [Raw],
}

/// Folds every table at `challenge` (no accumulation): the last round.
#[cfg(test)]
pub(crate) fn fold_products_raw(
    ctx: &field::FpCtx<2>,
    input: &RawProducts,
    output: &mut RawProducts,
    challenge: Raw,
) {
    debug_assert_eq!(input.len(), 2 * output.len());
    let fold = |table_in: &[Raw], table_out: &mut [Raw]| {
        #[cfg(feature = "parallel")]
        if parallel(table_out.len()) {
            table_in
                .par_chunks_exact(2)
                .zip(table_out.par_iter_mut())
                .for_each(|(pair, value)| *value = ctx.interpolate(pair[0], pair[1], challenge));
            return;
        }
        for (pair, value) in table_in.chunks_exact(2).zip(table_out.iter_mut()) {
            *value = ctx.interpolate(pair[0], pair[1], challenge);
        }
    };
    fold(&input.az, &mut output.az);
    fold(&input.bz, &mut output.bz);
    fold(&input.cz, &mut output.cz);
}

/// Folds the tables at `challenge` and accumulates the next round's cofactor
/// evaluations from the folded pairs in the same pass.
#[cfg(test)]
pub(crate) fn fold_products_and_cofactor_evaluations_raw(
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    input: &RawProducts,
    output: &mut RawProducts,
    challenge: Raw,
    weights: RawEqWeights<'_>,
    endpoint: FactoredEndpoint,
) -> [Raw; 2] {
    debug_assert_eq!(input.len(), 2 * output.len());
    let out_pairs = output.len() / 2;
    debug_assert_eq!(out_pairs, weights.pair_count());

    // One contiguous output pair range → its input window is four times as
    // long. `first_pair` indexes the equality weights.
    let process = |first_pair: usize,
                   input: ProductSlices<'_>,
                   output: ProductSlicesMut<'_>|
     -> ProductPair {
        let pairs = output.az.len() / 2;
        let mut accumulators = product_pair();
        if let Some(low) = weights.two_level_low() {
            // The block is a whole number of high buckets (see the callers).
            let low_pairs = low.len();
            debug_assert_eq!(pairs % low_pairs, 0);
            debug_assert_eq!(first_pair % low_pairs, 0);
            let mut high_index = first_pair / low_pairs;
            let mut base = 0;
            while base < pairs {
                let mut inner = product_pair();
                for (low_index, &weight) in low.iter().enumerate() {
                    let pair = base + low_index;
                    let (i, o) = (4 * pair, 2 * pair);
                    let az = [
                        ctx.interpolate(input.az[i], input.az[i + 1], challenge),
                        ctx.interpolate(input.az[i + 2], input.az[i + 3], challenge),
                    ];
                    let bz = [
                        ctx.interpolate(input.bz[i], input.bz[i + 1], challenge),
                        ctx.interpolate(input.bz[i + 2], input.bz[i + 3], challenge),
                    ];
                    let cz = [
                        ctx.interpolate(input.cz[i], input.cz[i + 1], challenge),
                        ctx.interpolate(input.cz[i + 2], input.cz[i + 3], challenge),
                    ];
                    output.az[o] = az[0];
                    output.az[o + 1] = az[1];
                    output.bz[o] = bz[0];
                    output.bz[o + 1] = bz[1];
                    output.cz[o] = cz[0];
                    output.cz[o + 1] = cz[1];
                    accumulate_cofactor_raw(
                        ctx, &mut inner, weight, endpoint, az[0], az[1], bz[0], bz[1], cz[0], cz[1],
                    );
                }
                let inner = reduce_product_pair(inner, reducer);
                let high_weight = weights.high[high_index];
                accumulators[0].accumulate_encoded(ctx, high_weight, inner[0]);
                accumulators[1].accumulate_encoded(ctx, high_weight, inner[1]);
                high_index += 1;
                base += low_pairs;
            }
        } else {
            for pair in 0..pairs {
                let (i, o) = (4 * pair, 2 * pair);
                let az = [
                    ctx.interpolate(input.az[i], input.az[i + 1], challenge),
                    ctx.interpolate(input.az[i + 2], input.az[i + 3], challenge),
                ];
                let bz = [
                    ctx.interpolate(input.bz[i], input.bz[i + 1], challenge),
                    ctx.interpolate(input.bz[i + 2], input.bz[i + 3], challenge),
                ];
                let cz = [
                    ctx.interpolate(input.cz[i], input.cz[i + 1], challenge),
                    ctx.interpolate(input.cz[i + 2], input.cz[i + 3], challenge),
                ];
                output.az[o] = az[0];
                output.az[o + 1] = az[1];
                output.bz[o] = bz[0];
                output.bz[o + 1] = bz[1];
                output.cz[o] = cz[0];
                output.cz[o + 1] = cz[1];
                let weight = weights.pair_weight(ctx, first_pair + pair);
                accumulate_cofactor_raw(
                    ctx,
                    &mut accumulators,
                    weight,
                    endpoint,
                    az[0],
                    az[1],
                    bz[0],
                    bz[1],
                    cz[0],
                    cz[1],
                );
            }
        }
        accumulators
    };

    // Block size in output pairs: a multiple of the low pair count so every
    // block covers whole high buckets.
    let block_pairs = match weights.two_level_low() {
        Some(low) => {
            let low_pairs = low.len();
            (FOLD_BLOCK / 2).div_ceil(low_pairs).max(1) * low_pairs
        }
        None => FOLD_BLOCK / 2,
    };

    #[cfg(feature = "parallel")]
    if parallel(out_pairs) {
        let out_block = 2 * block_pairs;
        let in_block = 4 * block_pairs;
        let total = (
            input.az.par_chunks(in_block),
            input.bz.par_chunks(in_block),
            input.cz.par_chunks(in_block),
            output.az.par_chunks_mut(out_block),
            output.bz.par_chunks_mut(out_block),
            output.cz.par_chunks_mut(out_block),
        )
            .into_par_iter()
            .enumerate()
            .map(|(block, (az, bz, cz, az_out, bz_out, cz_out))| {
                process(
                    block * block_pairs,
                    ProductSlices { az, bz, cz },
                    ProductSlicesMut {
                        az: az_out,
                        bz: bz_out,
                        cz: cz_out,
                    },
                )
            })
            .reduce(product_pair, merge_product_pair);
        return reduce_product_pair(total, reducer);
    }

    let total = process(
        0,
        ProductSlices {
            az: &input.az,
            bz: &input.bz,
            cz: &input.cz,
        },
        ProductSlicesMut {
            az: &mut output.az,
            bz: &mut output.bz,
            cz: &mut output.cz,
        },
    );
    reduce_product_pair(total, reducer)
}

/// Folds the exact native tables into the field at `challenge` — every output
/// is `(1 - challenge) · v0 + challenge · v1`, Montgomery-reduced once and
/// converted to raw form — and accumulates the next round's cofactor
/// evaluations in the same pass.
#[cfg(test)]
pub(crate) fn fold_native_products_and_cofactor_evaluations_raw(
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    products: NativeProducts<'_>,
    output: &mut RawProducts,
    challenge: Raw,
    weights: Option<(RawEqWeights<'_>, FactoredEndpoint)>,
) -> [Raw; 2] {
    let (az_in, bz_in, cz_in) = (products.az, products.bz, products.cz);
    debug_assert_eq!(az_in.len(), 2 * output.len());
    let one_minus_challenge = ctx.sub_raw(ctx.one_raw(), challenge);
    // Plain Montgomery reduction plus one conversion multiply: cheaper than
    // the Barrett remainder of the R-scaled sum, same canonical residue.
    let fold = |v0: u64, v1: u64| -> Raw {
        ctx.plain_to_raw(fold_native_pair_plain(
            ctx,
            one_minus_challenge,
            challenge,
            v0,
            v1,
        ))
    };

    let Some((weights, endpoint)) = weights else {
        // Final fold: a single output entry per table, no next round.
        debug_assert_eq!(output.len(), 1);
        output.az[0] = fold(az_in[0], az_in[1]);
        output.bz[0] = fold(bz_in[0], bz_in[1]);
        output.cz[0] = fold(cz_in[0], cz_in[1]);
        return [0, 0];
    };

    let out_pairs = output.len() / 2;
    debug_assert_eq!(out_pairs, weights.pair_count());
    let process = |first_pair: usize,
                   az_in: &[u64],
                   bz_in: &[u64],
                   cz_in: &[u64],
                   az_out: &mut [Raw],
                   bz_out: &mut [Raw],
                   cz_out: &mut [Raw]|
     -> ProductPair {
        let pairs = az_out.len() / 2;
        let mut accumulators = product_pair();
        let mut emit = |pair: usize| -> [[Raw; 2]; 3] {
            let (i, o) = (4 * pair, 2 * pair);
            let az = [
                fold(az_in[i], az_in[i + 1]),
                fold(az_in[i + 2], az_in[i + 3]),
            ];
            let bz = [
                fold(bz_in[i], bz_in[i + 1]),
                fold(bz_in[i + 2], bz_in[i + 3]),
            ];
            let cz = [
                fold(cz_in[i], cz_in[i + 1]),
                fold(cz_in[i + 2], cz_in[i + 3]),
            ];
            az_out[o] = az[0];
            az_out[o + 1] = az[1];
            bz_out[o] = bz[0];
            bz_out[o + 1] = bz[1];
            cz_out[o] = cz[0];
            cz_out[o + 1] = cz[1];
            [az, bz, cz]
        };
        if let Some(low) = weights.two_level_low() {
            let low_pairs = low.len();
            debug_assert_eq!(pairs % low_pairs, 0);
            debug_assert_eq!(first_pair % low_pairs, 0);
            let mut high_index = first_pair / low_pairs;
            let mut base = 0;
            while base < pairs {
                let mut inner = product_pair();
                for (low_index, &weight) in low.iter().enumerate() {
                    let [az, bz, cz] = emit(base + low_index);
                    accumulate_cofactor_raw(
                        ctx, &mut inner, weight, endpoint, az[0], az[1], bz[0], bz[1], cz[0], cz[1],
                    );
                }
                let inner = reduce_product_pair(inner, reducer);
                let high_weight = weights.high[high_index];
                accumulators[0].accumulate_encoded(ctx, high_weight, inner[0]);
                accumulators[1].accumulate_encoded(ctx, high_weight, inner[1]);
                high_index += 1;
                base += low_pairs;
            }
        } else {
            for pair in 0..pairs {
                let [az, bz, cz] = emit(pair);
                let weight = weights.pair_weight(ctx, first_pair + pair);
                accumulate_cofactor_raw(
                    ctx,
                    &mut accumulators,
                    weight,
                    endpoint,
                    az[0],
                    az[1],
                    bz[0],
                    bz[1],
                    cz[0],
                    cz[1],
                );
            }
        }
        accumulators
    };

    let block_pairs = match weights.two_level_low() {
        Some(low) => {
            let low_pairs = low.len();
            (FOLD_BLOCK / 2).div_ceil(low_pairs).max(1) * low_pairs
        }
        None => FOLD_BLOCK / 2,
    };

    #[cfg(feature = "parallel")]
    if parallel(out_pairs) {
        let out_block = 2 * block_pairs;
        let in_block = 4 * block_pairs;
        let total = (
            az_in.par_chunks(in_block),
            bz_in.par_chunks(in_block),
            cz_in.par_chunks(in_block),
            output.az.par_chunks_mut(out_block),
            output.bz.par_chunks_mut(out_block),
            output.cz.par_chunks_mut(out_block),
        )
            .into_par_iter()
            .enumerate()
            .map(|(block, (az, bz, cz, az_out, bz_out, cz_out))| {
                process(block * block_pairs, az, bz, cz, az_out, bz_out, cz_out)
            })
            .reduce(product_pair, merge_product_pair);
        return reduce_product_pair(total, reducer);
    }

    let total = process(
        0,
        az_in,
        bz_in,
        cz_in,
        &mut output.az,
        &mut output.bz,
        &mut output.cz,
    );
    reduce_product_pair(total, reducer)
}

/// Shared prover-side scalars of one outer sumcheck.
#[cfg(test)]
pub(crate) struct EncodedScalars<'a> {
    ctx: &'a field::FpCtx<2>,
    reducer: &'a field::FpCtx<2>,
    tau: &'a [Field],
    tau_inverses: Vec<Field>,
    zero: Field,
    one: Field,
}

#[cfg(test)]
impl EncodedScalars<'_> {
    fn coefficients(
        &self,
        round: usize,
        current_claim: &Field,
        endpoint: FactoredEndpoint,
        evaluations: [Raw; 2],
        bound_equality: &Field,
    ) -> [Field; 3] {
        reconstruct_eq_factored_cubic_without_linear(
            current_claim,
            &self.tau[round],
            &self.tau_inverses[round],
            endpoint,
            [
                crate::utils::delayed_reduction::element(self.ctx, evaluations[0]),
                crate::utils::delayed_reduction::element(self.ctx, evaluations[1]),
            ],
            bound_equality,
            &self.one,
            self.ctx,
        )
    }
}

/// Removes the active coordinate from the equality factors: the low table
/// while it has more than one entry, then the high table.
#[cfg(test)]
pub(crate) fn strip_active_coordinate(
    ctx: &field::FpCtx<2>,
    eq_low: &mut Vec<Raw>,
    eq_high: &mut Vec<Raw>,
) {
    if eq_low.len() > 1 {
        *eq_low = strip_coordinate_raw(ctx, eq_low);
    } else {
        *eq_high = strip_coordinate_raw(ctx, eq_high);
    }
}

#[cfg(test)]
pub(crate) fn weights_of<'a>(eq_low: &'a [Raw], eq_high: &'a [Raw]) -> RawEqWeights<'a> {
    RawEqWeights {
        low: (!eq_low.is_empty() && eq_low.len() > 1).then_some(eq_low),
        high: eq_high,
    }
}

/// Runs the remaining field-valued outer rounds from a prepared state: the
/// equality factors already stripped of the current coordinate (they are the
/// current round's weights) and the current round's coefficients computed.
/// `round_boundary` acts between each absorbed round polynomial and its
/// challenge ([`UngrindedRoundBoundary`] adds no transcript bytes).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn continue_encoded<T: Transcript, P: RoundBoundaryPolicy>(
    transcript: &mut T,
    scalars: &EncodedScalars<'_>,
    state: &mut RoundState<Field>,
    mut eq_low: Vec<Raw>,
    mut eq_high: Vec<Raw>,
    mut products: RawProducts,
    coefficients: [Field; 3],
    round_boundary: &mut P,
) -> Result<RawProducts, SumcheckError> {
    let ctx = scalars.ctx;
    let mut scratch = RawProducts::zeros(products.len() / 2);
    state.continue_with(
        ctx,
        transcript,
        scalars.tau,
        coefficients,
        round_boundary,
        |round, challenge, state| {
            let next_len = products.len() / 2;
            scratch.truncate(next_len);
            let coefficients = if next_len == 1 {
                fold_products_raw(ctx, &products, &mut scratch, ctx.raw(&challenge));
                [scalars.zero; 3]
            } else {
                strip_active_coordinate(ctx, &mut eq_low, &mut eq_high);
                let endpoint = FactoredEndpoint::for_tau(&scalars.tau[round + 1]);
                let evaluations = fold_products_and_cofactor_evaluations_raw(
                    ctx,
                    scalars.reducer,
                    &products,
                    &mut scratch,
                    ctx.raw(&challenge),
                    weights_of(&eq_low, &eq_high),
                    endpoint,
                );
                scalars.coefficients(
                    round + 1,
                    &state.claim,
                    endpoint,
                    evaluations,
                    &state.equality_scale,
                )
            };
            products.swap(&mut scratch);
            Ok(coefficients)
        },
    )?;
    Ok(products)
}

/// Only the three terminal scalars cross back from raw storage.
#[cfg(test)]
fn finish_encoded(
    transcript: &mut impl Transcript,
    ctx: &field::FpCtx<2>,
    state: RoundState<Field>,
    products: &RawProducts,
) -> Result<OuterSumcheckOutput<Field>, SumcheckError> {
    state.finish(
        ctx,
        transcript,
        [
            shared_raw(ctx, products.az[0]),
            shared_raw(ctx, products.bz[0]),
            shared_raw(ctx, products.cz[0]),
        ],
    )
}

#[cfg(test)]
fn encoded_scalars<'a>(
    ctx: &'a field::FpCtx<2>,
    reducer: &'a field::FpCtx<2>,
    tau: &'a [Field],
) -> EncodedScalars<'a> {
    EncodedScalars {
        ctx,
        reducer,
        tau,
        tau_inverses: batch_invert_nonzero(tau, ctx),
        zero: Field::zero_with_cfg(ctx),
        one: Field::one_with_cfg(ctx),
    }
}

/// Proves the cubic outer sumcheck from field-valued raw product tables: the
/// encoded storage test driver with the delayed-Barrett
/// reducer. `eq_low`/`eq_high` are the full equality factors of `tau`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn prove_encoded<T: Transcript>(
    transcript: &mut T,
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    initial_claim: Field,
    tau: &[Field],
    eq_low: Vec<Raw>,
    eq_high: Vec<Raw>,
    products: RawProducts,
) -> Result<OuterSumcheckOutput<Field>, SumcheckError> {
    prepare_encoded(
        transcript,
        ctx,
        reducer,
        initial_claim,
        tau,
        eq_low,
        eq_high,
        products,
        &mut UngrindedRoundBoundary,
        false,
    )
}

/// [`prove_encoded`] under an explicit message/challenge round
/// boundary policy: the raw twin of
/// the ordinary generic engine, transcript-identical to it
/// under the same policy.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn prepare_encoded<T: Transcript, P: RoundBoundaryPolicy>(
    transcript: &mut T,
    ctx: &field::FpCtx<2>,
    _reducer: &field::FpCtx<2>,
    initial_claim: Field,
    tau: &[Field],
    eq_low: Vec<Raw>,
    eq_high: Vec<Raw>,
    products: RawProducts,
    round_boundary: &mut P,
    known_zero: bool,
) -> Result<OuterSumcheckOutput<Field>, SumcheckError> {
    super::prove_outer_sumcheck(
        ctx,
        transcript,
        if known_zero {
            super::OuterClaim::RowwiseZero
        } else {
            super::OuterClaim::Sum(initial_claim)
        },
        tau,
        &native::ResidueRows {
            field: ctx,
            products: &products,
        },
        Some(factors_from_raw(ctx, eq_low, eq_high)),
        round_boundary,
    )
    .map(Into::into)
}

pub(crate) fn factors_from_raw(
    ctx: &field::FpCtx<2>,
    low: Vec<Raw>,
    high: Vec<Raw>,
) -> EqualityFactors<Field> {
    EqualityFactors::new(
        low.into_iter().map(|x| shared_raw(ctx, x)).collect(),
        high.into_iter().map(|x| shared_raw(ctx, x)).collect(),
        ctx,
    )
}

/// Independent retained field arithmetic reference for differential tests.
#[cfg(test)]
pub(super) fn prepare_encoded_reference<T: Transcript, P: RoundBoundaryPolicy>(
    transcript: &mut T,
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    initial_claim: Field,
    tau: &[Field],
    mut eq_low: Vec<Raw>,
    mut eq_high: Vec<Raw>,
    products: RawProducts,
    round_boundary: &mut P,
    known_zero: bool,
) -> Result<OuterSumcheckOutput<Field>, SumcheckError> {
    let num_vars = tau.len();
    if !products.len().is_power_of_two()
        || products.az.len() != products.bz.len()
        || products.az.len() != products.cz.len()
        || tau.len() != products.len().ilog2() as usize
        || !eq_low.len().is_power_of_two()
        || !eq_high.len().is_power_of_two()
        || eq_low.len().checked_mul(eq_high.len()) != Some(products.len())
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    round_boundary.validate(num_vars)?;
    let scalars = encoded_scalars(ctx, reducer, tau);
    let mut state = RoundState::new(initial_claim, num_vars, ctx);
    if num_vars == 0 {
        return finish_encoded(transcript, ctx, state, &products);
    }
    strip_active_coordinate(ctx, &mut eq_low, &mut eq_high);
    let endpoint = if known_zero {
        FactoredEndpoint::KnownZero
    } else {
        FactoredEndpoint::for_tau(&tau[0])
    };
    let evaluations = {
        let _scope = tracing::info_span!("raw:outer_round0").entered();
        cofactor_evaluations_raw(
            ctx,
            reducer,
            &products,
            weights_of(&eq_low, &eq_high),
            endpoint,
        )
    };
    let coefficients = scalars.coefficients(
        0,
        &state.claim,
        endpoint,
        evaluations,
        &state.equality_scale,
    );
    let _rounds_scope = tracing::info_span!("raw:outer_rounds").entered();
    let products = continue_encoded(
        transcript,
        &scalars,
        &mut state,
        eq_low,
        eq_high,
        products,
        coefficients,
        round_boundary,
    )?;
    finish_encoded(transcript, ctx, state, &products)
}

#[cfg(test)]
fn prove_native_prefix<T: Transcript>(
    transcript: &mut T,
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    initial_claim: Field,
    tau: &[Field],
    mut eq_low: Vec<Raw>,
    mut eq_high: Vec<Raw>,
    len: usize,
    known_zero: bool,
    singleton: impl FnOnce() -> [Raw; 3],
    round0: impl FnOnce(RawEqWeights<'_>, FactoredEndpoint) -> [Raw; 2],
    fold: impl FnOnce(&mut RawProducts, Raw, Option<(RawEqWeights<'_>, FactoredEndpoint)>) -> [Raw; 2],
) -> Result<OuterSumcheckOutput<Field>, SumcheckError> {
    let num_vars = tau.len();
    if !len.is_power_of_two()
        || num_vars != len.ilog2() as usize
        || !eq_low.len().is_power_of_two()
        || !eq_high.len().is_power_of_two()
        || eq_low.len().checked_mul(eq_high.len()) != Some(len)
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    let scalars = encoded_scalars(ctx, reducer, tau);
    let mut state = RoundState::new(initial_claim, num_vars, ctx);
    if num_vars == 0 {
        let [a, b, c] = singleton();
        return state.finish(
            ctx,
            transcript,
            [shared_raw(ctx, a), shared_raw(ctx, b), shared_raw(ctx, c)],
        );
    }
    strip_active_coordinate(ctx, &mut eq_low, &mut eq_high);
    let endpoint = if known_zero {
        FactoredEndpoint::KnownZero
    } else {
        FactoredEndpoint::for_tau(&tau[0])
    };
    let evaluations = {
        let _scope = tracing::info_span!("raw:outer_native_round0").entered();
        round0(weights_of(&eq_low, &eq_high), endpoint)
    };
    let coefficients = scalars.coefficients(
        0,
        &state.claim,
        endpoint,
        evaluations,
        &state.equality_scale,
    );
    let challenge = state.sample(
        ctx,
        transcript,
        tau,
        &coefficients,
        &mut UngrindedRoundBoundary,
    )?;
    let mut folded = RawProducts::zeros(len / 2);
    if len == 2 {
        fold(&mut folded, ctx.raw(&challenge), None);
        return finish_encoded(transcript, ctx, state, &folded);
    }
    strip_active_coordinate(ctx, &mut eq_low, &mut eq_high);
    let endpoint = FactoredEndpoint::for_tau(&tau[1]);
    let evaluations = {
        let _scope = tracing::info_span!("raw:outer_native_fold0").entered();
        fold(
            &mut folded,
            ctx.raw(&challenge),
            Some((weights_of(&eq_low, &eq_high), endpoint)),
        )
    };
    let coefficients = scalars.coefficients(
        1,
        &state.claim,
        endpoint,
        evaluations,
        &state.equality_scale,
    );
    let _rounds_scope = tracing::info_span!("raw:outer_rounds").entered();
    let products = continue_encoded(
        transcript,
        &scalars,
        &mut state,
        eq_low,
        eq_high,
        folded,
        coefficients,
        &mut UngrindedRoundBoundary,
    )?;
    finish_encoded(transcript, ctx, state, &products)
}

// ---------------------------------------------------------------------------

/// BitZ's validated R1CS relation promises rowwise zero residuals.
#[cfg(test)]
pub(crate) fn prove_encoded_zerocheck(
    transcript: &mut impl Transcript,
    field: &field::FpCtx<2>,
    tau: &[Field],
    low: Vec<Raw>,
    high: Vec<Raw>,
    products: RawProducts,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<OuterSumcheckOutput<Field>, SumcheckError> {
    prepare_encoded(
        transcript,
        field,
        field,
        field.zero(),
        tau,
        low,
        high,
        products,
        boundary,
        true,
    )
}
