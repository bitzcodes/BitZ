//! Encoded Montgomery storage and native kernels for the Spartan inner prover.
//! Outer arithmetic and protocol continuation live in `crate::sumcheck::outer`.
use crate::piop::spartan::SpartanField as _;
#[cfg(test)]
use crate::piop::spartan::mul::MulLayout;
#[cfg(test)]
use crate::sumcheck::bridge::PreparedBinding;
pub use crate::sumcheck::outer::arithmetic::NativeWideProducts;
#[cfg(test)]
pub use crate::sumcheck::outer::arithmetic::RawProducts;
pub(crate) use crate::sumcheck::outer::arithmetic::*;
use crate::utils::delayed_reduction::EncodedMac;
#[cfg(test)]
use circuit::linear_map::CscMatrix;
#[cfg(test)]
use field::Uint;
use field::{Fp, RingOps};
use std::borrow::Cow;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(test)]
use crate::piop::spartan::sumcheck::R1csProductMles;

#[cfg(test)]
use crate::piop::spartan::{
    baby_bear_mul::BabyBearMulCoefficient,
    matrix::{PreparedConstraintMatrices, SpartanMatrixCoefficient},
    u64_mul::U64MulCoefficient,
};

pub(crate) use crate::sumcheck::arithmetic::merge_accumulators as merge_product_pair;
use crate::sumcheck::{SumcheckError, arithmetic::merge_accumulators};

// Compatibility exports while callers migrate to the matrix bridge.
pub use crate::sumcheck::bridge::native::RawMontyCoefficient;
#[cfg(test)]
use crate::sumcheck::bridge::native::*;

mod folded;

mod native_witness;
mod state;
use folded::FoldedValue;
pub use state::NativeWeights;

pub use native_witness::{NativeBlockWitness, NativeLimbWitness};

pub(crate) type Field = Fp<2>;
pub(crate) type FieldConfig = field::FpCtx<2>;

/// One canonical Montgomery residue of the runtime field: the same two limbs
/// `Fp<2>::as_montgomery_integer` exposes, packed little-endian into a `u128`.
pub(crate) type Raw = u128;

/// Minimum work items before a kernel splits across the rayon pool. Matches
/// the generic prover's threshold; the split never changes any value.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_ITEMS: usize = 1 << 12;

/// Output elements per parallel block in the fused fold kernels (even, so a
/// block always holds whole output pairs).
pub(crate) const FOLD_BLOCK: usize = 1 << 11;

#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn parallel(work_items: usize) -> bool {
    work_items >= PARALLEL_MIN_ITEMS && rayon::current_num_threads() > 1
}

#[inline(always)]
pub(crate) const fn words_to_raw(words: &[u64; 2]) -> Raw {
    (words[0] as u128) | ((words[1] as u128) << 64)
}

#[inline(always)]
pub(crate) const fn raw_to_words(value: Raw) -> [u64; 2] {
    [value as u64, (value >> 64) as u64]
}

/// Prepares the shared arithmetic provider from the current protocol's config.
/// Configuration ownership stays with the prepared relation and round scalars.
pub(crate) fn field_context(config: &FieldConfig) -> field::FpCtx<2> {
    config.clone()
}

pub trait RawFieldStorage {
    fn one_raw(&self) -> Raw;
    fn raw(&self, value: &Field) -> Raw;
    fn native_residue(&self, value: u64) -> Raw;
    fn native_residue_u128(&self, value: u128) -> Raw;
    fn raw_vec(&self, values: &[Field]) -> Vec<Raw>;
    fn add_raw(&self, lhs: Raw, rhs: Raw) -> Raw;
    fn sub_raw(&self, lhs: Raw, rhs: Raw) -> Raw;
    fn neg_raw(&self, value: Raw) -> Raw;
    fn mul_raw(&self, lhs: Raw, rhs: Raw) -> Raw;
    fn plain_to_raw(&self, plain: Raw) -> Raw;
    fn redc_linear(&self, accumulator: &field::FpLinearAcc<2, 1>) -> Raw;
    fn redc(&self, value: [u64; 4]) -> Raw;
    fn interpolate(&self, at_zero: Raw, at_one: Raw, point: Raw) -> Raw;
}

impl RawFieldStorage for field::FpCtx<2> {
    #[inline]
    fn one_raw(&self) -> Raw {
        raw_shared(field::RingOps::one(&self))
    }
    #[inline]
    fn raw(&self, value: &Field) -> Raw {
        words_to_raw(value.as_montgomery_integer().as_words())
    }
    #[inline]
    fn native_residue(&self, value: u64) -> Raw {
        raw_shared(field::IntegerEmbedding::from_integer(&self, &value))
    }
    #[inline]
    fn native_residue_u128(&self, value: u128) -> Raw {
        raw_shared(field::IntegerEmbedding::from_integer(&self, &value))
    }
    #[inline]
    fn raw_vec(&self, values: &[Field]) -> Vec<Raw> {
        #[cfg(feature = "parallel")]
        if parallel(values.len()) {
            return values.par_iter().map(|value| self.raw(value)).collect();
        }
        values.iter().map(|value| self.raw(value)).collect()
    }
    #[inline]
    fn add_raw(&self, lhs: Raw, rhs: Raw) -> Raw {
        self.add_canonical_u128(lhs, rhs)
    }
    #[inline]
    fn sub_raw(&self, lhs: Raw, rhs: Raw) -> Raw {
        self.sub_canonical_u128(lhs, rhs)
    }
    #[inline]
    fn neg_raw(&self, value: Raw) -> Raw {
        self.sub_raw(0, value)
    }
    #[inline]
    fn mul_raw(&self, lhs: Raw, rhs: Raw) -> Raw {
        raw_shared(field::RingOps::mul(
            &self,
            &shared_raw(&self, lhs),
            &shared_raw(&self, rhs),
        ))
    }
    #[inline]
    fn plain_to_raw(&self, plain: Raw) -> Raw {
        raw_shared(self.from_canonical_integer(&field::Uint::from_words(raw_to_words(plain))))
    }
    #[inline]
    fn redc_linear(&self, accumulator: &field::FpLinearAcc<2, 1>) -> Raw {
        let (lo, hi, head) = accumulator.unreduced_integer().as_parts();
        let limbs = [lo[0], lo[1], hi[0], head, 0];
        debug_assert_eq!(limbs[4], 0);
        // Sufficient for `sum < q · R`: the top limb stays below q's top limb.
        debug_assert!(limbs[3] < (self.modulus_u128() >> 64) as u64);
        self.redc([limbs[0], limbs[1], limbs[2], limbs[3]])
    }
    #[inline]
    fn redc(&self, value: [u64; 4]) -> Raw {
        words_to_raw(
            self.reduce_montgomery_bounded(&field::Uint::from_words(value))
                .as_words(),
        )
    }
    #[inline(always)]
    fn interpolate(&self, at_zero: Raw, at_one: Raw, point: Raw) -> Raw {
        self.add_raw(at_zero, self.mul_raw(point, self.sub_raw(at_one, at_zero)))
    }
}
#[inline(always)]
pub(crate) fn shared_raw(ctx: &field::FpCtx<2>, raw: Raw) -> field::Fp<2> {
    ctx.from_montgomery_integer(field::Uint::from_words(raw_to_words(raw)))
}
#[inline(always)]
pub(crate) fn raw_shared(value: field::Fp<2>) -> Raw {
    words_to_raw(value.as_montgomery_integer().as_words())
}

// ---------------------------------------------------------------------------
// Equality tables
// ---------------------------------------------------------------------------

/// `eq(boolean_index, point)` in little-endian index order, the raw twin of
/// `matrix::eq_table`: the same doubling recurrence, entry for entry.
pub(crate) fn eq_table_raw(ctx: &field::FpCtx<2>, point: &[Raw]) -> Vec<Raw> {
    let mut table = Vec::new();
    eq_table_raw_into(ctx, point, &mut table);
    table
}
pub(crate) fn eq_table_raw_into(ctx: &field::FpCtx<2>, point: &[Raw], table: &mut Vec<Raw>) {
    table.resize(1usize << point.len(), 0);

    table[0] = ctx.one_raw();
    for (coordinate, &challenge) in point.iter().enumerate() {
        let half = 1usize << coordinate;
        let (zero_children, one_children) = table[..2 * half].split_at_mut(half);
        let expand = |zero_child: &mut Raw, one_child: &mut Raw| {
            let parent = *zero_child;
            *one_child = ctx.mul_raw(parent, challenge);
            *zero_child = ctx.sub_raw(parent, *one_child);
        };
        #[cfg(feature = "parallel")]
        if half >= (1 << 13) && rayon::current_num_threads() > 1 {
            zero_children
                .par_iter_mut()
                .zip(one_children.par_iter_mut())
                .for_each(|(zero_child, one_child)| expand(zero_child, one_child));
            continue;
        }
        for (zero_child, one_child) in zero_children.iter_mut().zip(one_children) {
            expand(zero_child, one_child);
        }
    }
}

/// The low- and high-coordinate equality factors of `matrix::make_equality_factors`,
/// as raw tables: the first `len / 2` coordinates form the low table.
pub(crate) fn make_equality_factors_raw(
    ctx: &field::FpCtx<2>,
    point: &[Field],
) -> (Vec<Raw>, Vec<Raw>) {
    let split = point.len() / 2;
    let raw_point = ctx.raw_vec(point);
    (
        eq_table_raw(ctx, &raw_point[..split]),
        eq_table_raw(ctx, &raw_point[split..]),
    )
}

/// Sums adjacent pairs: the equality table with its active (lowest) coordinate
pub(crate) type ProductPair = [field::FpProductAcc<2>; 2];
pub(crate) type LinearPair = [field::FpLinearAcc<2, 1>; 2];

#[inline(always)]
pub(crate) fn product_pair() -> ProductPair {
    [
        field::FpProductAcc::<2>::default(),
        field::FpProductAcc::<2>::default(),
    ]
}

#[inline(always)]
pub(crate) fn linear_pair() -> LinearPair {
    [
        field::FpLinearAcc::<2, 1>::default(),
        field::FpLinearAcc::<2, 1>::default(),
    ]
}

#[inline(always)]
pub(crate) fn reduce_product_pair(pair: ProductPair, reducer: &field::FpCtx<2>) -> [Raw; 2] {
    let [endpoint, infinity] = pair;
    [
        endpoint.reduce_encoded(reducer),
        infinity.reduce_encoded(reducer),
    ]
}

#[inline(always)]
pub(crate) fn reduce_linear_pair(pair: LinearPair, reducer: &field::FpCtx<2>) -> [Raw; 2] {
    let [endpoint, infinity] = pair;
    [
        endpoint.reduce_encoded(reducer),
        infinity.reduce_encoded(reducer),
    ]
}

#[inline(always)]
#[cfg(test)]
pub(crate) fn accumulate_signed_raw(
    ctx: &field::FpCtx<2>,
    accumulator: &mut field::FpLinearAcc<2, 1>,
    weight: Raw,
    negative_weight: Raw,
    value: i128,
) {
    let mask = (value >> 127) as u128;
    let magnitude = ((value as u128) ^ mask).wrapping_sub(mask);
    debug_assert!(magnitude <= u128::from(u64::MAX));
    let selected = (weight & !mask) | (negative_weight & mask);
    accumulator.accumulate_encoded(ctx, selected, magnitude as u64);
}

// Inner sumcheck
// ---------------------------------------------------------------------------

/// Constructor-owned declaration of a leading `[1, 0, ...]` assignment block.
/// Only typed relation adapters may attach this structure to native values.
/// This is trusted prover metadata, not validation: the accompanying assignment
/// must satisfy the declared prefix. Reusing it for other values can produce
/// an invalid proof; it does not alter the verifier's checks.
#[derive(Clone, Copy, Debug)]
pub struct NativeConstantPrefix(usize);

impl NativeConstantPrefix {
    pub(crate) fn new(block_len: usize) -> Self {
        assert!(block_len.is_power_of_two());
        Self(block_len)
    }
}

/// The assignment table entering the inner sumcheck.
pub enum RawWitness<'a> {
    /// Exact native values (the u32 and BabyBear relations): the first round
    /// accumulates field × `u64` products and its fold projects into the
    /// field. `values` holds the leading entries of a `domain`-length table
    /// whose remainder is zero, so a relation's logical assignment can be
    /// borrowed without padding it.
    Native {
        values: Cow<'a, [u64]>,
        domain: usize,
        constant_prefix: Option<NativeConstantPrefix>,
    },
    /// Borrowed declared-width x/y/u256 product segments.
    Wide(NativeBlockWitness<'a>),
    /// Borrowed 32-limb witness and quotient blocks.
    Limbs(NativeLimbWitness<'a>),
    /// Raw field residues.
    Field(Vec<Raw>),
}

impl<'a> RawWitness<'a> {
    /// A complete native table.
    #[cfg(test)]
    pub(crate) fn native_owned(values: Vec<u64>) -> Self {
        let domain = values.len();
        Self::Native {
            values: Cow::Owned(values),
            domain,
            constant_prefix: None,
        }
    }

    /// The leading `values` of a `domain`-length native table (the rest is
    /// zero).
    pub(crate) fn native_borrowed(values: &'a [u64], domain: usize) -> Self {
        Self::native_borrowed_with_constant_prefix(values, domain, None)
    }

    pub(crate) fn native_borrowed_with_constant_prefix(
        values: &'a [u64],
        domain: usize,
        constant_prefix: Option<NativeConstantPrefix>,
    ) -> Self {
        debug_assert!(values.len() <= domain);
        assert!(constant_prefix.is_none_or(|prefix| prefix.0 <= values.len()));
        Self::Native {
            values: Cow::Borrowed(values),
            domain,
            constant_prefix,
        }
    }

    /// The (padded) table length.
    fn len(&self) -> usize {
        match self {
            Self::Native { domain, .. } => *domain,
            Self::Field(values) => values.len(),
            Self::Wide(values) => values.len(),
            Self::Limbs(values) => values.len(),
        }
    }

    /// Pads a borrowed native table out to its domain (a copy only when the
    /// caller could not lend the whole table).
    fn materialize(self) -> Self {
        match self {
            Self::Native {
                mut values,
                domain,
                constant_prefix,
            } if values.len() < domain => {
                values.to_mut().resize(domain, 0);
                Self::Native {
                    values,
                    domain,
                    constant_prefix,
                }
            }
            other => other,
        }
    }
}

#[inline(always)]
fn fold_native_pair(
    reducer: &field::FpCtx<2>,
    one_minus_challenge: Raw,
    challenge: Raw,
    at_zero: u64,
    at_one: u64,
) -> Raw {
    let mut accumulator = field::FpLinearAcc::<2, 1>::default();
    accumulator.accumulate_encoded(reducer, one_minus_challenge, at_zero);
    accumulator.accumulate_encoded(reducer, challenge, at_one);
    accumulator.reduce_encoded(reducer)
}

/// `(1 - c) · w0 + c · w1` as a PLAIN residue (scale one) from the raw
/// challenge and exact native values: the `R`-scaled sum
/// `(1-c)_raw · w0 + c_raw · w1 < 2 · 2^64 · q` is Montgomery-reduced once,
/// which is far cheaper than the Barrett remainder the raw form needs.
#[inline(always)]
pub(crate) fn fold_native_pair_plain(
    ctx: &field::FpCtx<2>,
    one_minus_challenge: Raw,
    challenge: Raw,
    at_zero: u64,
    at_one: u64,
) -> Raw {
    let coefficients = [
        crate::utils::delayed_reduction::element(ctx, one_minus_challenge),
        crate::utils::delayed_reduction::element(ctx, challenge),
    ];
    let values = [
        field::Uint::from_words([at_zero]),
        field::Uint::from_words([at_one]),
    ];
    u128::from(ctx.weighted_pair_to_integer(&coefficients, &values))
}

/// Round-zero `[c0, c2]` over a field witness: `Σ m0·w0` and
/// `Σ (m1 − m0)(w1 − w0)` over adjacent pairs.
fn inner_coefficients_field_raw(
    ctx: &field::FpCtx<2>,
    matrix: &[Raw],
    witness: &[Raw],
) -> [Raw; 2] {
    debug_assert_eq!(matrix.len(), witness.len());
    debug_assert_eq!(matrix.len() % 2, 0);
    let block = |matrix: &[Raw], witness: &[Raw]| -> ProductPair {
        let mut accumulators = product_pair();
        for (m, w) in matrix.chunks_exact(2).zip(witness.chunks_exact(2)) {
            accumulators[0].accumulate_encoded(ctx, m[0], w[0]);
            accumulators[1].accumulate_encoded(
                ctx,
                ctx.sub_raw(m[1], m[0]),
                ctx.sub_raw(w[1], w[0]),
            );
        }
        accumulators
    };
    #[cfg(feature = "parallel")]
    if parallel(matrix.len() / 2) {
        let total = matrix
            .par_chunks(2 * FOLD_BLOCK)
            .zip(witness.par_chunks(2 * FOLD_BLOCK))
            .map(|(m, w)| block(m, w))
            .reduce(product_pair, merge_product_pair);
        return reduce_product_pair(total, ctx);
    }
    reduce_product_pair(block(matrix, witness), ctx)
}

/// Round-zero `[c0, c2]` over an exact native witness.
fn inner_coefficients_native_raw(
    ctx: &field::FpCtx<2>,
    matrix: &[Raw],
    witness: &[u64],
) -> [Raw; 2] {
    debug_assert_eq!(matrix.len(), witness.len());
    inner_coefficients_native_pairs(ctx, matrix, |start, len| {
        witness[start..start + len]
            .chunks_exact(2)
            .map(|w| [w[0], w[1]])
    })
}
fn inner_coefficients_native_map(
    ctx: &field::FpCtx<2>,
    matrix: &[Raw],
    read: impl Fn(usize) -> u64 + Sync,
) -> [Raw; 2] {
    inner_coefficients_native_pairs(ctx, matrix, |start, len| {
        (start..start + len)
            .step_by(2)
            .map(|i| [read(i), read(i + 1)])
    })
}

fn inner_coefficients_native_pairs<I: Iterator<Item = [u64; 2]>>(
    ctx: &field::FpCtx<2>,
    matrix: &[Raw],
    pairs: impl Fn(usize, usize) -> I + Sync,
) -> [Raw; 2] {
    debug_assert_eq!(matrix.len() % 2, 0);
    let block = |start: usize, matrix: &[Raw]| -> LinearPair {
        let mut accumulators = linear_pair();
        for (m, w) in matrix.chunks_exact(2).zip(pairs(start, matrix.len())) {
            accumulators[0].accumulate_encoded(ctx, m[0], w[0]);
            // (m1 - m0)(w1 - w0) as (±(m1 - m0)) · |w1 - w0|: one product,
            // congruent modulo q to the two-product form.
            let delta = ctx.sub_raw(m[1], m[0]);
            let mask = 0u64.wrapping_sub(u64::from(w[1] < w[0]));
            let magnitude = (w[1].wrapping_sub(w[0]) ^ mask).wrapping_sub(mask);
            let mask128 = (mask as u128) | ((mask as u128) << 64);
            let signed = (delta & !mask128) | (ctx.neg_raw(delta) & mask128);
            accumulators[1].accumulate_encoded(ctx, signed, magnitude);
        }
        accumulators
    };
    #[cfg(feature = "parallel")]
    if parallel(matrix.len() / 2) {
        let total = matrix
            .par_chunks(2 * FOLD_BLOCK)
            .enumerate()
            .map(|(i, m)| block(i * 2 * FOLD_BLOCK, m))
            .reduce(linear_pair, merge_accumulators);
        return reduce_linear_pair(total, ctx);
    }
    reduce_linear_pair(block(0, matrix), ctx)
}

/// First native fold with canonical integer output and typed linear MAC.
fn fold_inner_native_raw(
    ctx: &field::FpCtx<2>,
    matrix: &[Raw],
    witness: &[u64],
    matrix_out: &mut [Raw],
    out: &mut [field::Uint<2>],
    challenge: Raw,
) -> [Raw; 2] {
    debug_assert_eq!(witness.len(), 2 * out.len());
    fold_native_pairs::<true, _>(
        ctx,
        matrix,
        |start, len| {
            witness[start..start + len]
                .chunks_exact(2)
                .map(|w| [w[0], w[1]])
        },
        matrix_out,
        out,
        challenge,
    )
}

fn fold_native_map<const FOLD_WEIGHTS: bool>(
    ctx: &field::FpCtx<2>,
    weights: &[Raw],
    read: impl Fn(usize) -> u64 + Sync,
    matrix_out: &mut [Raw],
    out: &mut [field::Uint<2>],
    challenge: Raw,
) -> [Raw; 2] {
    fold_native_pairs::<FOLD_WEIGHTS, _>(
        ctx,
        weights,
        |start, len| {
            (start..start + len)
                .step_by(2)
                .map(|i| [read(i), read(i + 1)])
        },
        matrix_out,
        out,
        challenge,
    )
}

fn fold_native_pairs<const FOLD_WEIGHTS: bool, I: Iterator<Item = [u64; 2]>>(
    ctx: &field::FpCtx<2>,
    weights: &[Raw],
    pairs: impl Fn(usize, usize) -> I + Sync,
    matrix_out: &mut [Raw],
    out: &mut [field::Uint<2>],
    challenge: Raw,
) -> [Raw; 2] {
    debug_assert_eq!(weights.len(), out.len() * if FOLD_WEIGHTS { 2 } else { 1 });
    debug_assert_eq!(matrix_out.len(), if FOLD_WEIGHTS { out.len() } else { 0 });
    debug_assert_eq!(out.len() % 2, 0);
    let one_minus = ctx.sub_raw(ctx.one_raw(), challenge);
    let block = |start: usize, mout: &mut [Raw], out: &mut [field::Uint<2>]| {
        let mut acc = folded::pair::<field::Uint<2>>();
        let mut input = pairs(2 * start, 2 * out.len());
        let stride = if FOLD_WEIGHTS { 2 } else { 1 };
        let weights = &weights[start * stride..(start + out.len()) * stride];
        for (pair, (weights, z)) in weights
            .chunks_exact(2 * stride)
            .zip(out.chunks_exact_mut(2))
            .enumerate()
        {
            let mut m = [0; 2];
            for j in 0..2 {
                m[j] = if FOLD_WEIGHTS {
                    ctx.interpolate(weights[2 * j], weights[2 * j + 1], challenge)
                } else {
                    weights[j]
                };
                if FOLD_WEIGHTS {
                    mout[2 * pair + j] = m[j];
                }
                let [at_zero, at_one] = input.next().expect("native pairs cover every output");
                z[j] = field::Uint::from_words(raw_to_words(fold_native_pair_plain(
                    ctx, one_minus, challenge, at_zero, at_one,
                )));
            }
            <field::Uint<2> as FoldedValue>::accumulate(ctx, &mut acc[0], m[0], z[0]);
            <field::Uint<2> as FoldedValue>::accumulate(
                ctx,
                &mut acc[1],
                ctx.sub_raw(m[1], m[0]),
                field::Uint::from_words(raw_to_words(
                    ctx.sub_raw(u128::from(z[1]), u128::from(z[0])),
                )),
            );
        }
        acc
    };
    #[cfg(feature = "parallel")]
    if parallel(out.len() / 2) {
        let acc = if FOLD_WEIGHTS {
            matrix_out
                .par_chunks_mut(FOLD_BLOCK)
                .zip(out.par_chunks_mut(FOLD_BLOCK))
                .enumerate()
                .map(|(i, (m, z))| block(i * FOLD_BLOCK, m, z))
                .reduce(folded::pair::<field::Uint<2>>, merge_accumulators)
        } else {
            out.par_chunks_mut(FOLD_BLOCK)
                .enumerate()
                .map(|(i, z)| block(i * FOLD_BLOCK, &mut [], z))
                .reduce(folded::pair::<field::Uint<2>>, merge_accumulators)
        };
        return folded::reduce::<field::Uint<2>>(ctx, acc);
    }
    folded::reduce::<field::Uint<2>>(ctx, block(0, matrix_out, out))
}

// ---------------------------------------------------------------------------
// Structured inner sumcheck (block-selector relations)
// ---------------------------------------------------------------------------

/// The batched matrix of a block-selector relation, without materializing it:
/// `D(k · block_len + r) = scales[k] · weights[r]` for `r < rows`, zero for
/// blocks without a scale and for `r ≥ rows`.
#[derive(Clone)]
pub struct BlockScales {
    pub block_len: usize,
    pub rows: usize,
    /// Per block: the summed `f · coefficient` of the runs starting there.
    pub scales: Vec<Option<Raw>>,
}

/// Folds one table at `challenge` (no accumulation).
fn fold_table_raw(ctx: &field::FpCtx<2>, input: &[Raw], output: &mut [Raw], challenge: Raw) {
    debug_assert_eq!(input.len(), 2 * output.len());
    #[cfg(feature = "parallel")]
    if parallel(output.len()) {
        input
            .par_chunks_exact(2)
            .zip(output.par_iter_mut())
            .for_each(|(pair, value)| *value = ctx.interpolate(pair[0], pair[1], challenge));
        return;
    }
    for (pair, value) in input.chunks_exact(2).zip(output.iter_mut()) {
        *value = ctx.interpolate(pair[0], pair[1], challenge);
    }
}

/// Folds one witness block at `challenge` against the ALREADY folded weight
/// vector and accumulates the block's next-round partial sums
/// First native block fold: the
/// folded block is emitted in plain form.
fn fold_block_native_raw(
    ctx: &field::FpCtx<2>,
    weights: &[Raw],
    values: &[u64],
    out: &mut [field::Uint<2>],
    challenge: Raw,
) -> [Raw; 2] {
    debug_assert_eq!(values.len(), 2 * out.len());
    fold_native_pairs::<false, _>(
        ctx,
        weights,
        |start, len| {
            values[start..start + len]
                .chunks_exact(2)
                .map(|w| [w[0], w[1]])
        },
        &mut [],
        out,
        challenge,
    )
}

/// Terminal folds for structural constant/zero blocks. Private values never
/// determine whether a block receives the full weighted-sum kernel.
fn sparse_block_value_raw(eq_at_zero: Raw, block: BlockValues<'_>) -> Option<Raw> {
    match block {
        BlockValues::ConstantOne => Some(eq_at_zero),
        BlockValues::Zero => Some(0),
        BlockValues::Native([]) | BlockValues::Field([]) => Some(0),
        _ => None,
    }
}

/// `Σ_y eq[y] · z[y]` as a raw residue, for a block that carries no matrix
/// scale (its fold never enters a message before the block rounds).
fn weighted_block_sum_raw(ctx: &field::FpCtx<2>, eq: &[Raw], block: BlockValues<'_>) -> Raw {
    match block {
        BlockValues::U32(_)
        | BlockValues::U64FromU32Halves(_, _)
        | BlockValues::U128(_)
        | BlockValues::U256(_, _)
        | BlockValues::ConstantOne
        | BlockValues::Limbs(_)
        | BlockValues::Zero => native_witness::weighted_wide_block(ctx, eq, block),
        BlockValues::Native(values) => {
            debug_assert!(values.len() <= eq.len());
            let block = |eq: &[Raw], values: &[u64]| -> field::FpLinearAcc<2, 1> {
                let mut accumulator = field::FpLinearAcc::<2, 1>::default();
                for (&weight, &value) in eq.iter().zip(values) {
                    accumulator.accumulate_encoded(ctx, weight, value);
                }
                accumulator
            };
            #[cfg(feature = "parallel")]
            if parallel(values.len()) {
                let total = eq[..values.len()]
                    .par_chunks(2 * FOLD_BLOCK)
                    .zip(values.par_chunks(2 * FOLD_BLOCK))
                    .map(|(eq, values)| block(eq, values))
                    .reduce(field::FpLinearAcc::<2, 1>::default, |mut left, right| {
                        left += right;
                        left
                    });
                return total.reduce_encoded(ctx);
            }
            block(&eq[..values.len()], values).reduce_encoded(ctx)
        }
        BlockValues::Field(values) => {
            debug_assert!(values.len() <= eq.len());
            let block = |eq: &[Raw], values: &[Raw]| -> field::FpProductAcc<2> {
                let mut accumulator = field::FpProductAcc::<2>::default();
                for (&weight, &value) in eq.iter().zip(values) {
                    accumulator.accumulate_encoded(ctx, weight, value);
                }
                accumulator
            };
            #[cfg(feature = "parallel")]
            if parallel(values.len()) {
                let total = eq[..values.len()]
                    .par_chunks(2 * FOLD_BLOCK)
                    .zip(values.par_chunks(2 * FOLD_BLOCK))
                    .map(|(eq, values)| block(eq, values))
                    .reduce(field::FpProductAcc::<2>::default, |mut left, right| {
                        left += right;
                        left
                    });
                return total.reduce_encoded(ctx);
            }
            block(&eq[..values.len()], values).reduce_encoded(ctx)
        }
    }
}

/// One whole witness block in the caller's representation (entries past the
/// live prefix are zero).
#[derive(Clone, Copy)]
enum BlockValues<'a> {
    Native(&'a [u64]),
    U32(&'a [u32]),
    U64FromU32Halves(&'a [u32], &'a [u32]),
    Field(&'a [Raw]),
    U128(&'a [u128]),
    U256(&'a [u128], &'a [u128]),
    ConstantOne,
    Zero,
    Limbs(&'a [field::Uint<32>]),
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::piop::spartan::{
        baby_bear_mul::{BabyBearMulLayout, baby_bear_mul_constraint_matrices},
        matrix::{ConstraintMatrices, ConstraintMatricesSkeleton, eq_table, make_equality_factors},
        squeeze_field,
        sumcheck::{
            prove_inner_sumcheck_u32_native_with_reducer, prove_inner_sumcheck_with_reducer,
            prove_u32_first_round,
        },
        u32_mul::u32_mul_constraint_matrices,
        univariate_skip::PrefixUnivariateRowBinding,
        univariate_skip_native::{
            encoded_native_message, fold_encoded_lagrange, fold_native_lagrange_validated,
            native_message_validated,
        },
    };
    use crate::{poly::mle::DenseMultilinearExtension, transcript::Blake3Transcript};

    const MODULI: [u128; 3] = [(1_u128 << 100) - 15, (1_u128 << 127) - 1, u128::MAX - 158];

    fn config(modulus: u128) -> FieldConfig {
        Fp::<2>::make_cfg(&Uint::from(modulus)).expect("odd test modulus")
    }

    fn random_field(rng: &mut StdRng, cfg: &FieldConfig) -> Field {
        Field::from_with_cfg(rng.random::<u128>(), cfg)
    }

    fn random_fields(rng: &mut StdRng, cfg: &FieldConfig, len: usize) -> Vec<Field> {
        (0..len).map(|_| random_field(rng, cfg)).collect()
    }

    fn dense<T>(evaluations: Vec<T>) -> DenseMultilinearExtension<T> {
        let num_vars = evaluations.len().trailing_zeros() as usize;
        assert_eq!(evaluations.len(), 1 << num_vars);
        DenseMultilinearExtension {
            evaluations,
            num_vars,
        }
    }

    fn next_challenge(transcript: &mut Blake3Transcript, cfg: &FieldConfig) -> Field {
        squeeze_field::<Field, _>(transcript, cfg).unwrap()
    }

    /// `signed_words_residue` against a BigInt reference: two's-complement
    /// words of every width up to nine (the P-256 row products), random and
    /// extreme values of both signs, zero, over every test modulus (all
    /// above `2^64`).
    #[test]
    fn signed_words_residue_matches_bigint() {
        use num_bigint::BigInt;
        use num_traits::ToPrimitive;
        let mut rng = StdRng::seed_from_u64(11);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let shared = field::create_prime_field(field::Uint::from_words([
                modulus as u64,
                (modulus >> 64) as u64,
            ]));
            let projection = field::PreparedSignedProjection::new(shared, 9);
            let q = BigInt::from(modulus);
            for len in 0..=9usize {
                let mut cases: Vec<Vec<u64>> = (0..16)
                    .map(|_| (0..len).map(|_| rng.random::<u64>()).collect())
                    .collect();
                cases.push(vec![u64::MAX; len]);
                if len > 0 {
                    let mut minimum = vec![0; len];
                    minimum[len - 1] = 1 << 63;
                    cases.push(minimum);
                }
                for words in cases {
                    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let value = BigInt::from_signed_bytes_le(&bytes);
                    let expected = ((value % &q) + &q) % &q;
                    let expected = Field::from_with_cfg(expected.to_u128().unwrap(), &cfg);
                    assert_eq!(
                        crate::utils::delayed_reduction::element(&cfg, {
                            let value = projection.project(&words);
                            let w = value.as_montgomery_integer().as_words();
                            Raw::from(w[0]) | (Raw::from(w[1]) << 64)
                        }),
                        expected,
                        "modulus {modulus} words {words:?}"
                    );
                }
            }
        }
    }

    /// Throughput microbenchmark of the raw kernels (run with `--ignored
    /// --nocapture`): dependent and independent multiplication chains, the
    /// two reductions, and a generic `Fp` multiplication for scale.
    #[test]
    #[ignore]
    fn raw_arithmetic_microbench() {
        use std::time::Instant;
        let cfg = config(MODULI[0]);
        let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let n = 1usize << 22;
        let a: Vec<Raw> = (0..n)
            .map(|_| ctx.raw(&random_field(&mut rng, &cfg)))
            .collect();
        let b: Vec<Raw> = (0..n)
            .map(|_| ctx.raw(&random_field(&mut rng, &cfg)))
            .collect();
        let fa: Vec<Field> = a
            .iter()
            .map(|&v| crate::utils::delayed_reduction::element(&cfg, v))
            .collect();
        let fb: Vec<Field> = b
            .iter()
            .map(|&v| crate::utils::delayed_reduction::element(&cfg, v))
            .collect();
        let ns = |started: Instant| started.elapsed().as_secs_f64() * 1e9 / n as f64;

        // Dependent chain.
        let started = Instant::now();
        let mut acc = a[0];
        for &x in &b {
            acc = ctx.mul_raw(acc, x);
        }
        let dep = ns(started);
        std::hint::black_box(acc);

        // Independent products.
        let mut out = vec![0 as Raw; n];
        let started = Instant::now();
        for i in 0..n {
            out[i] = ctx.mul_raw(a[i], b[i]);
        }
        let indep = ns(started);
        std::hint::black_box(&out);

        // Interpolation.
        let c = ctx.raw(&random_field(&mut rng, &cfg));
        let started = Instant::now();
        for i in 0..n {
            out[i] = ctx.interpolate(a[i], b[i], c);
        }
        let interp = ns(started);
        std::hint::black_box(&out);

        // Product MAC + one reduction per 64 products.
        let started = Instant::now();
        let mut total = 0 as Raw;
        for chunk in a.chunks_exact(64).zip(b.chunks_exact(64)) {
            let mut acc = field::FpProductAcc::<2>::default();
            for (&x, &y) in chunk.0.iter().zip(chunk.1) {
                acc.accumulate_encoded(&ctx, x, y);
            }
            total ^= acc.reduce_encoded(&reducer);
        }
        let mac = ns(started);
        std::hint::black_box(total);

        // Linear reduce alone.
        let started = Instant::now();
        let mut total = 0 as Raw;
        for i in 0..n {
            let mut acc = field::FpLinearAcc::<2, 1>::default();
            acc.accumulate_encoded(&ctx, a[i], b[i] as u64);
            acc.accumulate_encoded(&ctx, b[i], a[i] as u64);
            total ^= acc.reduce_encoded(&reducer);
        }
        let linear = ns(started);
        std::hint::black_box(total);

        // Generic Fp multiplication.
        let mut fout = vec![Field::zero_with_cfg(&cfg); n];
        let started = Instant::now();
        for i in 0..n {
            fout[i] = cfg.mul(&(fa[i].clone()), &(&fb[i]));
        }
        let generic = ns(started);
        std::hint::black_box(&fout);

        eprintln!(
            "raw mul: dependent {dep:.2} ns, independent {indep:.2} ns, interpolate {interp:.2} ns, \
             product MAC+reduce/64 {mac:.2} ns, 2 linear MAC + reduce {linear:.2} ns; \
             Fp mul {generic:.2} ns"
        );
    }

    #[test]
    fn raw_arithmetic_matches_monty_field() {
        let mut rng = StdRng::seed_from_u64(0x5eed_a11c);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            assert_eq!(
                crate::utils::delayed_reduction::element(&cfg, ctx.one_raw()),
                Field::one_with_cfg(&cfg)
            );
            assert_eq!(
                crate::utils::delayed_reduction::element(&cfg, 0),
                Field::zero_with_cfg(&cfg)
            );
            let edge = [0_u128, 1, 2, modulus - 2, modulus - 1];
            let mut samples: Vec<(Field, Field, Field)> = edge
                .iter()
                .flat_map(|&a| edge.iter().map(move |&b| (a, b)))
                .map(|(a, b)| {
                    (
                        Field::from_with_cfg(a, &cfg),
                        Field::from_with_cfg(b, &cfg),
                        Field::from_with_cfg(a ^ b, &cfg),
                    )
                })
                .collect();
            for _ in 0..500 {
                samples.push((
                    random_field(&mut rng, &cfg),
                    random_field(&mut rng, &cfg),
                    random_field(&mut rng, &cfg),
                ));
            }
            for (a, b, c) in samples {
                let (ra, rb, rc) = (ctx.raw(&a), ctx.raw(&b), ctx.raw(&c));
                assert_eq!(crate::utils::delayed_reduction::element(&cfg, ra), a);
                assert_eq!(
                    crate::utils::delayed_reduction::element(&cfg, ctx.add_raw(ra, rb)),
                    cfg.add(&(a.clone()), &(&b))
                );
                assert_eq!(
                    crate::utils::delayed_reduction::element(&cfg, ctx.sub_raw(ra, rb)),
                    cfg.sub(&(a.clone()), &(&b))
                );
                assert_eq!(
                    crate::utils::delayed_reduction::element(&cfg, ctx.neg_raw(ra)),
                    cfg.neg(&a)
                );
                assert_eq!(
                    crate::utils::delayed_reduction::element(&cfg, ctx.mul_raw(ra, rb)),
                    cfg.mul(&(a.clone()), &(&b))
                );
                assert_eq!(
                    crate::utils::delayed_reduction::element(&cfg, ctx.interpolate(ra, rb, rc)),
                    cfg.add(
                        &(a.clone()),
                        &(&(cfg.mul(&(c.clone()), &(&(cfg.sub(&(b.clone()), &(&a)))))))
                    )
                );
            }
            for value in [0_u64, 1, 7, u32::MAX as u64, u64::MAX - 3, u64::MAX] {
                assert_eq!(
                    crate::utils::delayed_reduction::element(&cfg, ctx.native_residue(value)),
                    Field::from_with_cfg(value, &cfg)
                );
            }
        }
    }

    #[test]
    fn raw_equality_tables_match_generic() {
        let mut rng = StdRng::seed_from_u64(0xe9_7ab1e);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            for len in 0..=15 {
                let point = random_fields(&mut rng, &cfg, len);
                let expected = eq_table(&point, &cfg).unwrap();
                let raw = eq_table_raw(&ctx, &ctx.raw_vec(&point));
                assert_eq!(raw.len(), expected.len());
                assert!(
                    raw.iter()
                        .zip(&expected)
                        .all(|(r, e)| { crate::utils::delayed_reduction::element(&cfg, *r) == *e })
                );
                let (low, high) = make_equality_factors(&point, &cfg).unwrap();
                let (raw_low, raw_high) = make_equality_factors_raw(&ctx, &point);
                assert_eq!(ctx.raw_vec(&low.evaluations), raw_low);
                assert_eq!(ctx.raw_vec(&high.evaluations), raw_high);
            }
        }
    }

    fn outer_claim(cfg: &FieldConfig, tau: &[Field], products: &R1csProductMles<Field>) -> Field {
        let weights = eq_table(tau, cfg).unwrap();
        let mut claim = Field::zero_with_cfg(cfg);
        for (index, weight) in weights.iter().enumerate() {
            let residual = cfg.sub(
                &(cfg.mul(
                    &(products.az.evaluations[index].clone()),
                    &(&products.bz.evaluations[index]),
                )),
                &(&products.cz.evaluations[index]),
            );
            claim = cfg.add(&(claim), &(&(cfg.mul(&(weight.clone()), &(&residual)))));
        }
        claim
    }

    #[test]
    fn raw_field_outer_matches_generic() {
        let mut rng = StdRng::seed_from_u64(0x0a7e_0000);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            let generic_reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            for num_vars in 0..=13 {
                let len = 1usize << num_vars;
                let tau = random_fields(&mut rng, &cfg, num_vars);
                let products = R1csProductMles {
                    az: dense(random_fields(&mut rng, &cfg, len)),
                    bz: dense(random_fields(&mut rng, &cfg, len)),
                    cz: dense(random_fields(&mut rng, &cfg, len)),
                };
                let claim = outer_claim(&cfg, &tau, &products);

                let mut generic_transcript = Blake3Transcript::new();
                let expected = crate::sumcheck::outer::EqualityFactors::from_mles(
                    make_equality_factors(&tau, &cfg).unwrap(),
                    &cfg,
                )
                .and_then(|factors| {
                    crate::sumcheck::outer::prove_outer_sumcheck(
                        &cfg,
                        &mut generic_transcript,
                        crate::sumcheck::outer::OuterClaim::Sum(claim.clone()),
                        &tau,
                        products.clone(),
                        Some(factors),
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                })
                .map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                .unwrap();

                let mut raw_transcript = Blake3Transcript::new();
                let (eq_low, eq_high) = make_equality_factors_raw(&ctx, &tau);
                let actual = prove_encoded(
                    &mut raw_transcript,
                    &ctx,
                    &reducer,
                    claim,
                    &tau,
                    eq_low,
                    eq_high,
                    RawProducts::from_field(&ctx, &products),
                )
                .unwrap();
                assert_eq!(actual, expected, "num_vars={num_vars} modulus={modulus:#x}");
                assert_eq!(
                    next_challenge(&mut raw_transcript, &cfg),
                    next_challenge(&mut generic_transcript, &cfg)
                );
            }
        }
    }

    #[test]
    fn raw_native_outer_matches_generic() {
        let mut rng = StdRng::seed_from_u64(0x0a7e_4a71);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            let generic_reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            for num_vars in 0..=13 {
                let len = 1usize << num_vars;
                let tau = random_fields(&mut rng, &cfg, num_vars);
                let az: Vec<u64> = (0..len).map(|_| u64::from(rng.random::<u32>())).collect();
                let bz: Vec<u64> = (0..len).map(|_| u64::from(rng.random::<u32>())).collect();
                let cz: Vec<u64> = (0..len)
                    .map(|index| {
                        // Mostly satisfied rows, with some arbitrary residuals.
                        if rng.random::<u8>() < 200 {
                            az[index] * bz[index]
                        } else {
                            rng.random::<u64>()
                        }
                    })
                    .collect();
                let products = R1csProductMles {
                    az: dense(az),
                    bz: dense(bz),
                    cz: dense(cz),
                };
                let field_products = R1csProductMles {
                    az: dense(
                        products
                            .az
                            .evaluations
                            .iter()
                            .map(|&v| Field::from_with_cfg(v, &cfg))
                            .collect(),
                    ),
                    bz: dense(
                        products
                            .bz
                            .evaluations
                            .iter()
                            .map(|&v| Field::from_with_cfg(v, &cfg))
                            .collect(),
                    ),
                    cz: dense(
                        products
                            .cz
                            .evaluations
                            .iter()
                            .map(|&v| Field::from_with_cfg(v, &cfg))
                            .collect(),
                    ),
                };
                let claim = outer_claim(&cfg, &tau, &field_products);

                let mut generic_transcript = Blake3Transcript::new();
                let expected = prove_u32_first_round(
                    &mut generic_transcript,
                    claim.clone(),
                    &tau,
                    make_equality_factors(&tau, &cfg).unwrap(),
                    products.clone(),
                    &cfg,
                    &generic_reducer,
                )
                .unwrap();

                let mut raw_transcript = Blake3Transcript::new();
                let (eq_low, eq_high) = make_equality_factors_raw(&ctx, &tau);
                let actual = crate::sumcheck::outer::prove_outer_sumcheck(
                    &ctx,
                    &mut raw_transcript,
                    crate::sumcheck::outer::OuterClaim::Sum(claim),
                    &tau,
                    NativeProducts::from_mles(&products),
                    Some(factors_from_raw(&ctx, eq_low, eq_high)),
                    &mut crate::sumcheck::UngrindedRoundBoundary,
                )
                .map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                .unwrap();
                assert_eq!(actual, expected, "num_vars={num_vars} modulus={modulus:#x}");
                assert_eq!(
                    next_challenge(&mut raw_transcript, &cfg),
                    next_challenge(&mut generic_transcript, &cfg)
                );
            }
        }
    }

    #[test]
    fn raw_inner_matches_generic_with_live_prefix() {
        let mut rng = StdRng::seed_from_u64(0x1aa3_11fe);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            let generic_reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            for num_vars in 0..=14 {
                let len = 1usize << num_vars;
                for live in [
                    len,
                    len.div_ceil(2),
                    (len * 5).div_ceil(8),
                    rng.random_range(1..=len),
                ] {
                    let mut matrix = random_fields(&mut rng, &cfg, len);
                    let mut witness = random_fields(&mut rng, &cfg, len);
                    let mut native: Vec<u64> = (0..len).map(|_| rng.random::<u64>()).collect();
                    for index in live..len {
                        matrix[index] = Field::zero_with_cfg(&cfg);
                        witness[index] = Field::zero_with_cfg(&cfg);
                        native[index] = 0;
                    }
                    let claim = |witness: &[Field]| {
                        matrix
                            .iter()
                            .zip(witness)
                            .fold(Field::zero_with_cfg(&cfg), |sum, (m, w)| {
                                cfg.add(&(sum), &(&(cfg.mul(&(m.clone()), &(w)))))
                            })
                    };

                    // Field witness.
                    let field_claim = claim(&witness);
                    let mut generic_transcript = Blake3Transcript::new();
                    let expected = prove_inner_sumcheck_with_reducer(
                        &mut generic_transcript,
                        field_claim.clone(),
                        dense(matrix.clone()),
                        dense(witness.clone()),
                        &cfg,
                        &generic_reducer,
                    )
                    .unwrap();
                    let mut raw_transcript = Blake3Transcript::new();
                    let actual = crate::sumcheck::inner::prove_inner_sumcheck(
                        &ctx,
                        &mut raw_transcript,
                        field_claim,
                        RawWitness::Field(ctx.raw_vec(&witness)),
                        crate::sumcheck::inner::native::NativeWeights::Dense {
                            matrix: ctx.raw_vec(&matrix),
                            live: live,
                        },
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                    .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                    .unwrap();
                    assert_eq!(actual, expected, "field num_vars={num_vars} live={live}");
                    assert_eq!(
                        next_challenge(&mut raw_transcript, &cfg),
                        next_challenge(&mut generic_transcript, &cfg)
                    );

                    // Native witness.
                    let native_field: Vec<Field> = native
                        .iter()
                        .map(|&v| Field::from_with_cfg(v, &cfg))
                        .collect();
                    let native_claim = claim(&native_field);
                    let mut generic_transcript = Blake3Transcript::new();
                    let expected = prove_inner_sumcheck_u32_native_with_reducer(
                        &mut generic_transcript,
                        native_claim.clone(),
                        dense(matrix.clone()),
                        dense(native.clone()),
                        &cfg,
                        &generic_reducer,
                    )
                    .unwrap();
                    let mut raw_transcript = Blake3Transcript::new();
                    let actual = crate::sumcheck::inner::prove_inner_sumcheck(
                        &ctx,
                        &mut raw_transcript,
                        native_claim,
                        RawWitness::native_owned(native.clone()),
                        crate::sumcheck::inner::native::NativeWeights::Dense {
                            matrix: ctx.raw_vec(&matrix),
                            live: live,
                        },
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                    .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                    .unwrap();
                    assert_eq!(actual, expected, "native num_vars={num_vars} live={live}");
                    assert_eq!(
                        next_challenge(&mut raw_transcript, &cfg),
                        next_challenge(&mut generic_transcript, &cfg)
                    );
                }
            }
        }
    }

    fn random_field_matrices(
        rng: &mut StdRng,
        cfg: &FieldConfig,
        rows: usize,
        columns: usize,
    ) -> ConstraintMatrices<Field> {
        let mut matrix = || {
            let entries = (0..rows)
                .map(|_| {
                    let mut row: Vec<(usize, Field)> = Vec::new();
                    for column in 0..columns {
                        if rng.random::<u8>() < 96 {
                            let mut value = random_field(rng, cfg);
                            if <Field as crate::piop::spartan::SpartanField>::is_zero(&value) {
                                value = Field::one_with_cfg(cfg);
                            }
                            row.push((column, value));
                        }
                    }
                    row
                })
                .collect();
            CscMatrix::try_from_rows(columns, entries).unwrap()
        };
        ConstraintMatrices::new(matrix(), matrix(), matrix()).unwrap()
    }

    fn assert_binding_matches<C>(
        ctx: &field::FpCtx<2>,
        rng: &mut StdRng,
        cfg: &FieldConfig,
        matrices: &PreparedConstraintMatrices<Field, C>,
    ) where
        C: SpartanMatrixCoefficient<Field> + RawMontyCoefficient,
    {
        let row_point = random_fields(rng, cfg, matrices.num_row_vars());
        let rho = random_field(rng, cfg);
        let expected = crate::piop::spartan::matrix::eq_table_prover(&row_point, matrices.config())
            .map_err(crate::sumcheck::SumcheckError::from)
            .and_then(|weights| matrices.binding(&rho).bind_rows(&weights))
            .unwrap();
        let row_weights = eq_table_raw(ctx, &ctx.raw_vec(&row_point));
        let actual = reference_native_binding(ctx, matrices, &row_weights, ctx.raw(&rho));
        assert_eq!(actual, ctx.raw_vec(&expected.evaluations));

        for skip_vars in 1..=4usize {
            if skip_vars > matrices.num_row_vars() {
                break;
            }
            let binding = PrefixUnivariateRowBinding {
                skip_vars: skip_vars as u8,
                z: random_field(rng, cfg),
                tail_point: random_fields(rng, cfg, matrices.num_row_vars() - skip_vars),
            };
            let factors = binding.row_factors(matrices.num_row_vars(), cfg).unwrap();
            let expected = matrices.structured().bind_prefix(&factors, &rho).unwrap();
            let actual = reference_prefix_binding(ctx, matrices, &factors, ctx.raw(&rho));
            assert_eq!(
                actual,
                ctx.raw_vec(&expected.evaluations),
                "skip_vars={skip_vars}"
            );
        }
    }

    #[test]
    fn raw_binding_matches_generic() {
        let mut rng = StdRng::seed_from_u64(0xb1_4d00);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);

            for (rows, columns) in [(1, 1), (5, 9), (16, 16), (37, 70), (300, 129)] {
                let prepared = PreparedConstraintMatrices::<Field, Field>::new(
                    random_field_matrices(&mut rng, &cfg, rows, columns),
                    &cfg,
                )
                .unwrap();
                assert_binding_matches(&ctx, &mut rng, &cfg, &prepared);
            }

            for multiplications in [1usize, 3, 256, 257, 1000, 4096, 5000] {
                let layout = MulLayout::<u32>::new(multiplications).unwrap();
                let prepared = PreparedConstraintMatrices::<Field, bool>::new(
                    u32_mul_constraint_matrices(&layout, true).unwrap(),
                    &cfg,
                )
                .unwrap();
                assert!(prepared.selector_layout().is_some());
                assert_binding_matches(&ctx, &mut rng, &cfg, &prepared);

                let layout = BabyBearMulLayout::new(multiplications).unwrap();
                let prepared = PreparedConstraintMatrices::<Field, BabyBearMulCoefficient>::new(
                    baby_bear_mul_constraint_matrices(&layout).unwrap(),
                    &cfg,
                )
                .unwrap();
                assert_binding_matches(&ctx, &mut rng, &cfg, &prepared);
            }
        }
    }

    #[test]
    fn structured_inner_matches_dense() {
        let mut rng = StdRng::seed_from_u64(0x57ac_7ed0);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            for &(block_log, blocks_log) in &[
                (1usize, 0usize),
                (1, 1),
                (1, 3),
                (2, 2),
                (3, 1),
                (6, 2),
                (8, 3),
                (11, 2),
                (12, 0),
            ] {
                let block_len = 1usize << block_log;
                let blocks = 1usize << blocks_log;
                let domain = block_len * blocks;
                let num_column_vars = block_log + blocks_log;
                for trial in 0..4 {
                    let rows = if trial == 0 {
                        block_len
                    } else {
                        rng.random_range(1..=block_len)
                    };
                    let mut scales: Vec<Option<Raw>> = (0..blocks)
                        .map(|_| {
                            (rng.random::<u8>() < 160)
                                .then(|| ctx.raw(&random_field(&mut rng, &cfg)))
                        })
                        .collect();
                    if scales.iter().all(Option::is_none) {
                        scales[blocks - 1] = Some(ctx.raw(&random_field(&mut rng, &cfg)));
                    }
                    // Every scaled block's run must lie inside the live prefix, as
                    // the relation's column count guarantees.
                    let min_live = scales
                        .iter()
                        .enumerate()
                        .filter(|(_, scale)| scale.is_some())
                        .map(|(block, _)| block * block_len + rows)
                        .max()
                        .unwrap();
                    let live = match trial {
                        0 => domain,
                        1 => rng.random_range(min_live..=domain),
                        _ => (min_live + rng.random_range(0..=block_len)).min(domain),
                    };
                    let weights: Vec<Raw> = (0..block_len)
                        .map(|_| ctx.raw(&random_field(&mut rng, &cfg)))
                        .collect();
                    let mut dense = vec![0 as Raw; domain];
                    for (block, scale) in scales.iter().enumerate() {
                        if let Some(scale) = scale {
                            for (row, weight) in weights[..rows].iter().enumerate() {
                                dense[block * block_len + row] = ctx.mul_raw(*scale, *weight);
                            }
                        }
                    }
                    let mut native: Vec<u64> = (0..domain).map(|_| rng.random()).collect();
                    let mut field: Vec<Raw> = (0..domain)
                        .map(|_| ctx.raw(&random_field(&mut rng, &cfg)))
                        .collect();
                    for index in live..domain {
                        native[index] = 0;
                        field[index] = 0;
                    }
                    if trial % 2 == 0 {
                        // The constant block of the generated relations: zero
                        // beyond entry 0.
                        for index in 1..block_len.min(live) {
                            native[index] = 0;
                            field[index] = 0;
                        }
                    }
                    let scaled = BlockScales {
                        block_len,
                        rows,
                        scales: scales.clone(),
                    };
                    let field_claim = dense.iter().zip(&field).fold(0 as Raw, |sum, (&d, &w)| {
                        ctx.add_raw(sum, ctx.mul_raw(d, w))
                    });
                    let native_claim = dense.iter().zip(&native).fold(0 as Raw, |sum, (&d, &w)| {
                        ctx.add_raw(sum, ctx.mul_raw(d, ctx.native_residue(w)))
                    });

                    for kind in 0..2 {
                        let (claim, dense_witness, structured_witness) = if kind == 0 {
                            (
                                crate::utils::delayed_reduction::element(&cfg, native_claim),
                                RawWitness::native_owned(native.clone()),
                                RawWitness::native_owned(native.clone()),
                            )
                        } else {
                            (
                                crate::utils::delayed_reduction::element(&cfg, field_claim),
                                RawWitness::Field(field.clone()),
                                RawWitness::Field(field.clone()),
                            )
                        };
                        let mut dense_transcript = Blake3Transcript::new();
                        let expected = crate::sumcheck::inner::prove_inner_sumcheck(
                            &ctx,
                            &mut dense_transcript,
                            claim.clone(),
                            dense_witness,
                            crate::sumcheck::inner::native::NativeWeights::Dense {
                                matrix: dense.clone(),
                                live: live,
                            },
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                        .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                        .unwrap();
                        let mut structured_transcript = Blake3Transcript::new();
                        let actual = crate::sumcheck::inner::prove_inner_sumcheck(
                            &ctx,
                            &mut structured_transcript,
                            claim,
                            structured_witness,
                            crate::sumcheck::inner::native::NativeWeights::Blocks {
                                weights: (&weights).to_vec(),
                                scales: (&scaled).clone(),
                                live: live,
                                num_vars: num_column_vars,
                            },
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                        .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                        .unwrap();
                        assert_eq!(
                            actual, expected,
                            "kind={kind} block_log={block_log} blocks_log={blocks_log} trial={trial} rows={rows} live={live}"
                        );
                        assert_eq!(
                            next_challenge(&mut structured_transcript, &cfg),
                            next_challenge(&mut dense_transcript, &cfg)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn declared_constant_prefix_matches_dense_and_preserves_generic_blocks() {
        let mut rng = StdRng::seed_from_u64(0xc057_a17);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = field_context(&cfg);
            for (block_len, blocks, logical_blocks) in [(8, 4, 4), (64, 8, 5)] {
                let domain = block_len * blocks;
                let live = logical_blocks * block_len - 3;
                let rows = block_len - 4;
                for constant_is_scaled in [false, true] {
                    for declaration in [None, Some(block_len), Some(block_len / 2)] {
                        let mut native: Vec<u64> = (0..live).map(|_| rng.random()).collect();
                        native[0] = 1;
                        let prefix = declaration.unwrap_or(1);
                        native[1..prefix].fill(0);
                        // The generic and width-mismatch cases contain a real
                        // nonzero value after the declared constant prefix.
                        if prefix < block_len {
                            native[prefix] = 7;
                        }
                        let weights: Vec<_> = (0..block_len)
                            .map(|_| ctx.raw(&random_field(&mut rng, &cfg)))
                            .collect();
                        let scales: Vec<_> = (0..blocks)
                            .map(|block| {
                                (block < logical_blocks && (block != 0 || constant_is_scaled))
                                    .then(|| ctx.raw(&random_field(&mut rng, &cfg)))
                            })
                            .collect();
                        let mut matrix = vec![0; domain];
                        for (block, scale) in scales.iter().enumerate() {
                            if let Some(scale) = scale {
                                for row in 0..rows {
                                    matrix[block * block_len + row] =
                                        ctx.mul_raw(*scale, weights[row]);
                                }
                            }
                        }
                        let claim = matrix.iter().zip(&native).fold(0, |sum, (&m, &w)| {
                            ctx.add_raw(sum, ctx.mul_raw(m, ctx.native_residue(w)))
                        });
                        let claim = crate::utils::delayed_reduction::element(&cfg, claim);
                        let mut dense_transcript = Blake3Transcript::new();
                        let expected = crate::sumcheck::inner::prove_inner_sumcheck(
                            &ctx,
                            &mut dense_transcript,
                            claim.clone(),
                            RawWitness::native_borrowed(&native, domain),
                            crate::sumcheck::inner::native::NativeWeights::Dense {
                                matrix: matrix,
                                live: live,
                            },
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                        .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                        .unwrap();
                        let mut structured_transcript = Blake3Transcript::new();
                        let actual = crate::sumcheck::inner::prove_inner_sumcheck(
                            &ctx,
                            &mut structured_transcript,
                            claim,
                            RawWitness::native_borrowed_with_constant_prefix(
                                &native,
                                domain,
                                declaration.map(NativeConstantPrefix::new),
                            ),
                            crate::sumcheck::inner::native::NativeWeights::Blocks {
                                weights: (&weights).to_vec(),
                                scales: (&BlockScales {
                                    block_len,
                                    rows,
                                    scales,
                                })
                                    .clone(),
                                live: live,
                                num_vars: domain.ilog2() as usize,
                            },
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                        .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                        .unwrap();
                        assert_eq!(
                            actual, expected,
                            "block_len={block_len} declaration={declaration:?}"
                        );
                        assert_eq!(
                            next_challenge(&mut structured_transcript, &cfg),
                            next_challenge(&mut dense_transcript, &cfg)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn block_selector_layouts_are_detected_for_the_generated_relations() {
        let cfg = config(MODULI[0]);
        for multiplications in [1usize, 3, 255, 256, 257, 1000, 4096, 5000] {
            let layout = MulLayout::<u32>::new(multiplications).unwrap();
            let matrices = u32_mul_constraint_matrices(&layout, true).unwrap();
            let prepared =
                PreparedConstraintMatrices::<Field, bool>::new(matrices.clone(), &cfg).unwrap();
            let detected = prepared
                .block_selector()
                .expect("u32 selectors are block selectors");
            assert_eq!(detected.rows, multiplications);
            assert_eq!(detected.block_len, layout.capacity());
            assert_eq!(detected.a.len(), 1);
            assert_eq!(detected.a[0].start, layout.capacity());
            assert_eq!(detected.b[0].start, 2 * layout.capacity());
            assert_eq!(detected.c[0].start, 3 * layout.capacity());
            assert!(
                detected.a[0].coefficient && detected.b[0].coefficient && detected.c[0].coefficient
            );
            let skeleton = ConstraintMatricesSkeleton::<Field, bool>::new(matrices).unwrap();
            let from_skeleton = PreparedConstraintMatrices::from_skeleton(&skeleton, &cfg).unwrap();
            let replayed = from_skeleton.block_selector().unwrap();
            assert_eq!(replayed.rows, detected.rows);
            assert_eq!(replayed.block_len, detected.block_len);
            assert_eq!(replayed.c[0].start, detected.c[0].start);

            let layout = BabyBearMulLayout::new(multiplications).unwrap();
            let matrices = baby_bear_mul_constraint_matrices(&layout).unwrap();
            let prepared = PreparedConstraintMatrices::<Field, BabyBearMulCoefficient>::new(
                matrices.clone(),
                &cfg,
            )
            .unwrap();
            let detected = prepared
                .block_selector()
                .expect("BabyBear selectors are block selectors");
            assert_eq!(detected.rows, multiplications);
            assert_eq!(detected.block_len, layout.capacity());
            assert_eq!(detected.c.len(), 2);
            assert_eq!(detected.c[0].start, 3 * layout.capacity());
            assert_eq!(detected.c[0].coefficient, BabyBearMulCoefficient::One);
            assert_eq!(detected.c[1].start, 4 * layout.capacity());
            assert_eq!(detected.c[1].coefficient, BabyBearMulCoefficient::Modulus);
            let skeleton =
                ConstraintMatricesSkeleton::<Field, BabyBearMulCoefficient>::new(matrices).unwrap();
            let from_skeleton = PreparedConstraintMatrices::from_skeleton(&skeleton, &cfg).unwrap();
            let replayed = from_skeleton.block_selector().unwrap();
            assert_eq!(replayed.c[1].start, detected.c[1].start);
            assert_eq!(replayed.c[1].coefficient, BabyBearMulCoefficient::Modulus);
        }

        // General matrices are not block selectors.
        let mut rng = StdRng::seed_from_u64(0x0b10_c5e1);
        let prepared = PreparedConstraintMatrices::<Field, Field>::new(
            random_field_matrices(&mut rng, &cfg, 37, 70),
            &cfg,
        )
        .unwrap();
        assert!(prepared.block_selector().is_none());

        // A run starting at column zero: one block spanning the domain.
        let rows = 5;
        let identity = CscMatrix::try_from_rows(
            8,
            (0..rows)
                .map(|row| vec![(row, Field::one_with_cfg(&cfg))])
                .collect(),
        )
        .unwrap();
        let empty = CscMatrix::<Box<[Field]>>::try_from_rows(8, vec![Vec::new(); rows]).unwrap();
        let prepared = PreparedConstraintMatrices::<Field, Field>::new(
            ConstraintMatrices::new(identity, empty.clone(), empty).unwrap(),
            &cfg,
        )
        .unwrap();
        let detected = prepared.block_selector().unwrap();
        assert_eq!((detected.rows, detected.block_len), (rows, 8));
        assert_eq!(detected.a[0].start, 0);
        assert!(detected.b.is_empty() && detected.c.is_empty());
    }

    #[test]
    fn raw_skip_message_and_prefix_fold_match_generic() {
        let mut rng = StdRng::seed_from_u64(0x5c1b_0000);
        for modulus in MODULI {
            let cfg = config(modulus);
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            let generic_reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            for num_vars in 1..=13 {
                let len = 1usize << num_vars;
                let az: Vec<u64> = (0..len).map(|_| u64::from(rng.random::<u32>())).collect();
                let bz: Vec<u64> = (0..len).map(|_| u64::from(rng.random::<u32>())).collect();
                let cz: Vec<u64> = (0..len)
                    .map(|index| {
                        if rng.random::<u8>() < 200 {
                            az[index] * bz[index]
                        } else {
                            rng.random::<u64>()
                        }
                    })
                    .collect();
                let mles = R1csProductMles {
                    az: dense(az.clone()),
                    bz: dense(bz.clone()),
                    cz: dense(cz.clone()),
                };
                let products = NativeProducts {
                    az: &az,
                    bz: &bz,
                    cz: &cz,
                };
                for skip_vars in 1..=4usize.min(num_vars) {
                    let tau_tail = random_fields(&mut rng, &cfg, num_vars - skip_vars);
                    let factors = make_equality_factors(&tau_tail, &cfg).unwrap();
                    let expected = native_message_validated(
                        skip_vars,
                        &factors,
                        &mles,
                        &cfg,
                        &generic_reducer,
                    )
                    .unwrap();
                    let (eq_low, eq_high) = make_equality_factors_raw(&ctx, &tau_tail);
                    let actual = encoded_native_message(
                        &cfg, skip_vars, &eq_low, &eq_high, products, &ctx, &reducer,
                    )
                    .unwrap();
                    assert_eq!(actual, expected, "skip message K={skip_vars} n={num_vars}");

                    let z = random_field(&mut rng, &cfg);
                    let expected = fold_native_lagrange_validated(
                        skip_vars,
                        mles.clone(),
                        &z,
                        &cfg,
                        &generic_reducer,
                    )
                    .unwrap();
                    let actual = fold_encoded_lagrange(skip_vars, products, &z, &ctx).unwrap();
                    assert_eq!(ctx.raw_vec(&expected.az.evaluations), actual.az);
                    assert_eq!(ctx.raw_vec(&expected.bz.evaluations), actual.bz);
                    assert_eq!(ctx.raw_vec(&expected.cz.evaluations), actual.cz);
                }
            }
        }
    }
}
