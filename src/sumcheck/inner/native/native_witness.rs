//! Borrowed native assignment blocks and declared-width first-round kernels.
use super::*;
use field::{CtOrd, CtSelect, Fp, FpLinearAcc, MergeAccumulator, RingOps, Uint};

/// Borrowed logical blocks, including implicit constants and padding.
#[derive(Clone, Copy)]
pub struct NativeBlockWitness<'a> {
    block_len: usize,
    block_count: usize,
    words: usize,
    blocks: [BlockValues<'a>; 8],
}
impl<'a> NativeBlockWitness<'a> {
    fn from_blocks<const N: usize>(
        block_len: usize,
        words: usize,
        blocks: [BlockValues<'a>; N],
    ) -> Self {
        assert!(block_len.is_power_of_two() && N.is_power_of_two() && N <= 8);
        assert!(block_len.checked_mul(N).is_some());
        let mut storage = [BlockValues::Zero; 8];
        for (i, block) in blocks.into_iter().enumerate() {
            let len = match block {
                BlockValues::Native(x) => x.len(),
                BlockValues::U32(x) => x.len(),
                BlockValues::U128(x) => x.len(),
                BlockValues::U64FromU32Halves(lo, hi) => {
                    assert_eq!(lo.len(), hi.len());
                    lo.len()
                }
                BlockValues::U256(lo, hi) => {
                    assert_eq!(lo.len(), hi.len());
                    lo.len()
                }
                BlockValues::ConstantOne | BlockValues::Zero => 0,
                _ => panic!("expected native multiplication block"),
            };
            assert!(len <= block_len);
            storage[i] = block;
        }
        Self {
            block_len,
            block_count: N,
            words,
            blocks: storage,
        }
    }
    pub(crate) fn u32(
        block_len: usize,
        x: &'a [u32],
        y: &'a [u32],
        lo: &'a [u32],
        hi: &'a [u32],
    ) -> Self {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), lo.len());
        Self::from_blocks(
            block_len,
            1,
            [
                BlockValues::ConstantOne,
                BlockValues::U32(x),
                BlockValues::U32(y),
                BlockValues::U64FromU32Halves(lo, hi),
            ],
        )
    }
    pub(crate) fn u64(
        block_len: usize,
        x: &'a [u64],
        y: &'a [u64],
        lo: &'a [u64],
        hi: &'a [u64],
    ) -> Self {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), lo.len());
        assert_eq!(x.len(), hi.len());
        Self::from_blocks(
            block_len,
            1,
            [
                BlockValues::ConstantOne,
                BlockValues::Native(x),
                BlockValues::Native(y),
                BlockValues::Native(lo),
                BlockValues::Native(hi),
                BlockValues::Zero,
                BlockValues::Zero,
                BlockValues::Zero,
            ],
        )
    }
    pub(crate) fn new(
        block_len: usize,
        x: &'a [u128],
        y: &'a [u128],
        lo: &'a [u128],
        hi: &'a [u128],
    ) -> Self {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), lo.len());
        Self::from_blocks(
            block_len,
            4,
            [
                BlockValues::ConstantOne,
                BlockValues::U128(x),
                BlockValues::U128(y),
                BlockValues::U256(lo, hi),
            ],
        )
    }
    pub(super) fn len(&self) -> usize {
        self.block_len * self.block_count
    }
    pub(super) fn block_len(&self) -> usize {
        self.block_len
    }
    pub(super) fn words(&self) -> usize {
        self.words
    }
    pub(super) fn block(&self, i: usize) -> BlockValues<'a> {
        assert!(i < self.block_count);
        self.blocks[i]
    }
    pub(super) fn read_u64(&self, index: usize) -> u64 {
        assert!(index < self.len());
        let row = index % self.block_len;
        match self.block(index / self.block_len) {
            BlockValues::ConstantOne => (row == 0) as u64,
            BlockValues::Zero => 0,
            BlockValues::Native(v) => v.get(row).copied().unwrap_or(0),
            BlockValues::U32(v) => read_u32(v, row),
            BlockValues::U64FromU32Halves(lo, hi) => read_u64_halves(lo, hi, row),
            _ => unreachable!("declared one-word source"),
        }
    }
    pub(super) fn read(&self, index: usize) -> Uint<4> {
        assert!(index < self.len());
        let row = index % self.block_len;
        match self.block(index / self.block_len) {
            BlockValues::U128(v) => read_u128(v, row).zero_extend(),
            BlockValues::U256(lo, hi) => read_u256(lo, hi, row),
            _ => Uint::from_u64(self.read_u64(index)),
        }
    }
}
#[inline(always)]
fn read_u32(v: &[u32], i: usize) -> u64 {
    u64::from(v.get(i).copied().unwrap_or(0))
}
#[inline(always)]
fn read_u64_halves(lo: &[u32], hi: &[u32], i: usize) -> u64 {
    read_u32(lo, i) | (read_u32(hi, i) << 32)
}
/// A four-block assignment: implicit one, borrowed witness, borrowed quotient,
/// implicit zero. The source contract declares 32 limbs, independently of values.
#[derive(Clone, Copy)]
pub struct NativeLimbWitness<'a> {
    block_len: usize,
    witness: &'a [Uint<32>],
    quotient: &'a [Uint<32>],
}
impl<'a> NativeLimbWitness<'a> {
    pub(crate) fn new(block_len: usize, witness: &'a [Uint<32>], quotient: &'a [Uint<32>]) -> Self {
        assert!(block_len.is_power_of_two());
        assert!(block_len.checked_mul(4).is_some());
        assert!(witness.len() <= block_len && quotient.len() <= block_len);
        Self {
            block_len,
            witness,
            quotient,
        }
    }
    pub(super) fn len(self) -> usize {
        4 * self.block_len
    }
    pub(super) fn block(self, index: usize) -> BlockValues<'a> {
        match index {
            0 => BlockValues::ConstantOne,
            1 => BlockValues::Limbs(self.witness),
            2 => BlockValues::Limbs(self.quotient),
            3 => BlockValues::Zero,
            _ => panic!("assignment block out of range"),
        }
    }
    pub(crate) fn read(self, index: usize) -> Uint<32> {
        assert!(index < self.len());
        let row = index % self.block_len;
        match self.block(index / self.block_len) {
            BlockValues::ConstantOne => Uint::from_u64((row == 0) as u64),
            BlockValues::Limbs(values) => read_limbs(values, row),
            BlockValues::Zero => Uint::ZERO,
            _ => unreachable!(),
        }
    }
}
#[inline]
fn read_limbs(values: &[Uint<32>], index: usize) -> Uint<32> {
    values.get(index).copied().unwrap_or(Uint::ZERO)
}

#[inline]
fn read_u128(values: &[u128], i: usize) -> Uint<2> {
    Uint::from_words(raw_to_words(values.get(i).copied().unwrap_or(0)))
}
#[inline]
fn read_u256(lo: &[u128], hi: &[u128], i: usize) -> Uint<4> {
    let lo = lo.get(i).copied().unwrap_or(0);
    let hi = hi.get(i).copied().unwrap_or(0);
    Uint::from_words([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64])
}

pub(super) fn wide_coefficients<const N: usize>(
    ctx: &field::FpCtx<2>,
    weights: &[Raw],
    read: impl Fn(usize) -> Uint<N> + Sync,
) -> [Raw; 2] {
    assert_eq!(weights.len() % 2, 0);
    let f = ctx;
    let zero = || [FpLinearAcc::<2, N>::zero(); 2];
    let block = |start: usize, weights: &[Raw]| {
        let mut acc = zero();
        for (i, m) in weights.chunks_exact(2).enumerate() {
            let a = read(start + 2 * i);
            let b = read(start + 2 * i + 1);
            acc[0].accumulate(&shared_raw(f, m[0]), &a);
            let negative = b.ct_lt(&a);
            let difference = b.wrapping_sub(&a);
            let magnitude = Uint::ct_select(&difference, &difference.wrapping_neg(), negative);
            let delta = shared_raw(f, ctx.sub_raw(m[1], m[0]));
            acc[1].accumulate(&Fp::ct_select(&delta, &f.neg(&delta), negative), &magnitude);
        }
        acc
    };
    #[cfg(feature = "parallel")]
    if parallel(weights.len() / 2) {
        let acc = weights
            .par_chunks(2 * FOLD_BLOCK)
            .enumerate()
            .map(|(i, w)| block(i * 2 * FOLD_BLOCK, w))
            .reduce(zero, merge_accumulators);
        return acc.map(|a| raw_shared(field::Reduce::reduce(f, a)));
    }
    block(0, weights).map(|a| raw_shared(field::Reduce::reduce(f, a)))
}

/// Precompute the contribution of each public limb position to a first fold.
/// With at most 32 limbs per input, the unreduced sum is below
/// `64 * (q - 1) * (2^64 - 1) < q * 2^128`, so one REDC yields a canonical
/// plain residue. All declared limbs are processed, including zero limbs.
struct PreparedLimbFold<const N: usize> {
    weights: [[Fp<2>; N]; 2],
}

impl<const N: usize> PreparedLimbFold<N> {
    fn new(ctx: &field::FpCtx<2>, coefficients: [Fp<2>; 2]) -> Self {
        assert!(N <= 32, "native fold exceeds the declared limb bound");
        let radix = shared_raw(ctx, ctx.native_residue_u128(1u128 << 64));
        let weights = coefficients.map(|coefficient| {
            let mut weight = coefficient;
            core::array::from_fn(|_| {
                let current = weight;
                weight = ctx.mul(&weight, &radix);
                current
            })
        });
        Self { weights }
    }

    #[inline(always)]
    fn fold(&self, ctx: &field::FpCtx<2>, left: &Uint<N>, right: &Uint<N>) -> Uint<2> {
        let mut accumulator = FpLinearAcc::<2, 1>::zero();
        for (weights, value) in self.weights.iter().zip([left, right]) {
            for (weight, &word) in weights.iter().zip(value.as_words()) {
                accumulator.accumulate(weight, &Uint::from_u64(word));
            }
        }
        Uint::from_words(raw_to_words(ctx.redc_linear(&accumulator)))
    }
}

// FOLD_WEIGHTS is selected once by the dense/structured caller. Both forms
// read each borrowed integer once, write canonical folds, and accumulate the
// next message in the same traversal.
pub(super) fn wide_fold<const N: usize, const FOLD_WEIGHTS: bool>(
    ctx: &field::FpCtx<2>,
    weights: &[Raw],
    read: impl Fn(usize) -> Uint<N> + Sync,
    matrix_out: &mut [Raw],
    out: &mut [Uint<2>],
    challenge: Raw,
) -> [Raw; 2] {
    assert_eq!(out.len() % 2, 0);
    assert_eq!(weights.len(), out.len() * if FOLD_WEIGHTS { 2 } else { 1 });
    assert_eq!(matrix_out.len(), if FOLD_WEIGHTS { out.len() } else { 0 });
    let f = ctx;
    let challenge = shared_raw(f, challenge);
    let coefficients = [f.sub(&f.one(), &challenge), challenge];
    let prepared_fold = PreparedLimbFold::<N>::new(ctx, coefficients);
    let zero = || [FpLinearAcc::<2, 2>::zero(); 2];
    let block = |start: usize, mout: &mut [Raw], out: &mut [Uint<2>]| {
        let mut acc = zero();
        for (pair, z) in out.chunks_exact_mut(2).enumerate() {
            let mut m = [0; 2];
            for j in 0..2 {
                let i = start + 2 * pair + j;
                z[j] = prepared_fold.fold(ctx, &read(2 * i), &read(2 * i + 1));
                m[j] = if FOLD_WEIGHTS {
                    ctx.interpolate(weights[2 * i], weights[2 * i + 1], raw_shared(challenge))
                } else {
                    weights[i]
                };
                if FOLD_WEIGHTS {
                    mout[2 * pair + j] = m[j];
                }
            }
            acc[0].accumulate(&shared_raw(f, m[0]), &z[0]);
            acc[1].accumulate(
                &shared_raw(f, ctx.sub_raw(m[1], m[0])),
                &Uint::from_words(raw_to_words(
                    ctx.sub_raw(words_to_raw(z[1].as_words()), words_to_raw(z[0].as_words())),
                )),
            );
        }
        acc
    };
    let reduce = |a: [FpLinearAcc<2, 2>; 2]| a.map(|a| raw_shared(field::Reduce::reduce(f, a)));
    #[cfg(feature = "parallel")]
    if parallel(out.len() / 2) {
        let acc = if FOLD_WEIGHTS {
            matrix_out
                .par_chunks_mut(2 * FOLD_BLOCK)
                .zip(out.par_chunks_mut(2 * FOLD_BLOCK))
                .enumerate()
                .map(|(i, (m, z))| block(i * 2 * FOLD_BLOCK, m, z))
                .reduce(zero, merge_accumulators)
        } else {
            out.par_chunks_mut(2 * FOLD_BLOCK)
                .enumerate()
                .map(|(i, z)| block(i * 2 * FOLD_BLOCK, &mut [], z))
                .reduce(zero, merge_accumulators)
        };
        return reduce(acc);
    }
    reduce(block(0, matrix_out, out))
}

impl BlockValues<'_> {
    pub(super) fn coefficients(self, ctx: &field::FpCtx<2>, weights: &[Raw]) -> [Raw; 2] {
        match self {
            Self::U32(v) => inner_coefficients_native_map(ctx, weights, |i| read_u32(v, i)),
            Self::U64FromU32Halves(lo, hi) => {
                inner_coefficients_native_map(ctx, weights, |i| read_u64_halves(lo, hi, i))
            }
            Self::Native(v) if v.len() < weights.len() => {
                inner_coefficients_native_map(ctx, weights, |i| v.get(i).copied().unwrap_or(0))
            }
            Self::Native(v) => inner_coefficients_native_raw(ctx, weights, &v[..weights.len()]),
            Self::Field(v) => inner_coefficients_field_raw(ctx, weights, &v[..weights.len()]),
            Self::Zero => [0; 2],
            Self::Limbs(v) => wide_coefficients(ctx, weights, |i| read_limbs(v, i)),
            Self::U128(v) => wide_coefficients(ctx, weights, |i| read_u128(v, i)),
            Self::U256(lo, hi) => wide_coefficients(ctx, weights, |i| read_u256(lo, hi, i)),
            Self::ConstantOne => {
                wide_coefficients(ctx, weights, |i| Uint::<1>::from_u64((i == 0) as u64))
            }
        }
    }
    pub(super) fn folded_pair(self, ctx: &field::FpCtx<2>, challenge: Raw) -> Raw {
        let f = ctx;
        let c = shared_raw(f, challenge);
        let coefficients = [f.sub(&f.one(), &c), c];
        match self {
            Self::U32(v) => fold_native_pair(
                ctx,
                ctx.sub_raw(ctx.one_raw(), challenge),
                challenge,
                read_u32(v, 0),
                read_u32(v, 1),
            ),
            Self::U64FromU32Halves(lo, hi) => fold_native_pair(
                ctx,
                ctx.sub_raw(ctx.one_raw(), challenge),
                challenge,
                read_u64_halves(lo, hi, 0),
                read_u64_halves(lo, hi, 1),
            ),
            Self::Native(v) => fold_native_pair(
                ctx,
                ctx.sub_raw(ctx.one_raw(), challenge),
                challenge,
                v.first().copied().unwrap_or(0),
                v.get(1).copied().unwrap_or(0),
            ),
            Self::Field(v) => ctx.interpolate(v[0], v[1], challenge),
            Self::Zero => 0,
            Self::Limbs(v) => {
                raw_shared(f.weighted_pair(&coefficients, &[read_limbs(v, 0), read_limbs(v, 1)]))
            }
            Self::U128(v) => {
                raw_shared(f.weighted_pair(&coefficients, &[read_u128(v, 0), read_u128(v, 1)]))
            }
            Self::U256(lo, hi) => raw_shared(
                f.weighted_pair(&coefficients, &[read_u256(lo, hi, 0), read_u256(lo, hi, 1)]),
            ),
            Self::ConstantOne => raw_shared(coefficients[0]),
        }
    }
    pub(super) fn fold_integer_block(
        self,
        ctx: &field::FpCtx<2>,
        weights: &[Raw],
        out: &mut [Uint<2>],
        challenge: Raw,
    ) -> [Raw; 2] {
        match self {
            Self::U32(v) => {
                fold_native_map::<false>(ctx, weights, |i| read_u32(v, i), &mut [], out, challenge)
            }
            Self::U64FromU32Halves(lo, hi) => fold_native_map::<false>(
                ctx,
                weights,
                |i| read_u64_halves(lo, hi, i),
                &mut [],
                out,
                challenge,
            ),
            Self::Native(v) if v.len() < 2 * out.len() => fold_native_map::<false>(
                ctx,
                weights,
                |i| v.get(i).copied().unwrap_or(0),
                &mut [],
                out,
                challenge,
            ),
            Self::Native(v) => {
                fold_block_native_raw(ctx, weights, &v[..2 * out.len()], out, challenge)
            }
            Self::Field(_) => unreachable!("integer source dispatch"),
            Self::Zero => {
                out.fill(Uint::ZERO);
                [0; 2]
            }
            Self::Limbs(v) => {
                wide_fold::<32, false>(ctx, weights, |i| read_limbs(v, i), &mut [], out, challenge)
            }
            Self::U128(v) => {
                wide_fold::<2, false>(ctx, weights, |i| read_u128(v, i), &mut [], out, challenge)
            }
            Self::U256(lo, hi) => wide_fold::<4, false>(
                ctx,
                weights,
                |i| read_u256(lo, hi, i),
                &mut [],
                out,
                challenge,
            ),
            Self::ConstantOne => wide_fold::<1, false>(
                ctx,
                weights,
                |i| Uint::from_u64((i == 0) as u64),
                &mut [],
                out,
                challenge,
            ),
        }
    }
}

pub(super) fn weighted_wide<const N: usize>(
    ctx: &field::FpCtx<2>,
    eq: &[Raw],
    read: impl Fn(usize) -> Uint<N> + Sync,
) -> Raw {
    let f = ctx;
    let block = |start: usize, weights: &[Raw]| {
        let mut acc = FpLinearAcc::<2, N>::zero();
        for (i, &w) in weights.iter().enumerate() {
            acc.accumulate(&shared_raw(f, w), &read(start + i));
        }
        acc
    };
    #[cfg(feature = "parallel")]
    if parallel(eq.len()) {
        let acc = eq
            .par_chunks(2 * FOLD_BLOCK)
            .enumerate()
            .map(|(i, w)| block(i * 2 * FOLD_BLOCK, w))
            .reduce(FpLinearAcc::zero, |mut a, b| {
                a.merge_assign(&b);
                a
            });
        return raw_shared(field::Reduce::reduce(f, acc));
    }
    raw_shared(field::Reduce::reduce(f, block(0, eq)))
}
pub(super) fn weighted_wide_block(
    ctx: &field::FpCtx<2>,
    eq: &[Raw],
    block: BlockValues<'_>,
) -> Raw {
    match block {
        BlockValues::U32(v) => weighted_wide(ctx, eq, |i| Uint::<1>::from_u64(read_u32(v, i))),
        BlockValues::U64FromU32Halves(lo, hi) => {
            weighted_wide(ctx, eq, |i| Uint::<1>::from_u64(read_u64_halves(lo, hi, i)))
        }
        BlockValues::U128(v) => weighted_wide(ctx, eq, |i| read_u128(v, i)),
        BlockValues::U256(lo, hi) => weighted_wide(ctx, eq, |i| read_u256(lo, hi, i)),
        BlockValues::ConstantOne => eq[0],
        BlockValues::Zero => 0,
        BlockValues::Limbs(v) => weighted_wide(ctx, eq, |i| read_limbs(v, i)),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Blake3Transcript;

    use rand::{RngExt, SeedableRng, rngs::StdRng};

    #[test]
    fn prepared_limb_fold_matches_independent_bigint() {
        use num_bigint::BigUint;
        use num_traits::ToPrimitive;

        fn check<const N: usize>() {
            let mut rng = StdRng::seed_from_u64(0x6c69_6d62_666f_6c64 + N as u64);
            for q in [(1u128 << 65) - 49, (1u128 << 100) - 15, u128::MAX - 158] {
                let cfg = Field::make_cfg(&Uint::from_words(raw_to_words(q))).unwrap();
                let ctx = field_context(&cfg);
                let modulus = BigUint::from(q);
                for case in 0..20 {
                    let challenge = match case {
                        0 => 0,
                        1 => 1,
                        2 => q - 1,
                        _ => rng.random::<u128>() % q,
                    };
                    let left = Uint::<N>::from_words(core::array::from_fn(|_| match case {
                        0 | 2 => u64::MAX,
                        1 => 0,
                        _ => rng.random(),
                    }));
                    let right = Uint::<N>::from_words(core::array::from_fn(|_| match case {
                        0 => 0,
                        1 | 2 => u64::MAX,
                        _ => rng.random(),
                    }));
                    let c = shared_raw(&ctx, ctx.native_residue_u128(challenge));
                    let prepared = PreparedLimbFold::new(&ctx, [ctx.sub(&ctx.one(), &c), c]);
                    let integer = |value: &Uint<N>| {
                        BigUint::from_bytes_le(
                            &value
                                .as_words()
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                    };
                    let c = BigUint::from(challenge);
                    let expected = ((&modulus + BigUint::from(1u8) - &c) * integer(&left)
                        + c * integer(&right))
                        % &modulus;
                    assert_eq!(
                        u128::from(prepared.fold(&ctx, &left, &right)),
                        expected.to_u128().unwrap(),
                        "N={N} q={q} case={case}"
                    );
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<4>();
        check::<32>();
    }

    #[test]
    fn limb_witness_matches_projected_dense_and_structured_rounds() {
        let mut rng = StdRng::seed_from_u64(0x33326c696d62);
        for q in [(1u128 << 100) - 15, u128::MAX - 158] {
            let cfg = Field::make_cfg(&Uint::from_words(raw_to_words(q))).unwrap();
            let ctx = field_context(&cfg);
            for log in [0, 1, 3, 7] {
                let cap = 1usize << log;
                let live = cap.saturating_sub(1).max(1);
                let mut values = || {
                    (0..live)
                        .map(|i| {
                            Uint::<32>::from_words(core::array::from_fn(|_| match i {
                                0 => u64::MAX,
                                1 => 0,
                                _ => rng.random(),
                            }))
                        })
                        .collect::<Vec<_>>()
                };
                let witness = values();
                let quotient = values();
                let input = NativeLimbWitness::new(cap, &witness, &quotient);
                let radix = Field::from_with_cfg(1u128 << 64, &cfg);
                let projected: Vec<_> = (0..input.len())
                    .map(|i| {
                        let v = input.read(i);
                        let mut out = Field::zero_with_cfg(&cfg);
                        for &word in v.as_words().iter().rev() {
                            out = cfg.mul(&(out), &(&radix));
                            out = cfg.add(&(out), &(&Field::from_with_cfg(word, &cfg)));
                        }
                        ctx.raw(&out)
                    })
                    .collect();
                let weights: Vec<_> = (0..cap)
                    .map(|_| ctx.native_residue_u128(rng.random()))
                    .collect();
                for active in [[false, true, true, false], [true, false, true, true]] {
                    let scales: Vec<_> = active
                        .iter()
                        .map(|&v| v.then(|| ctx.native_residue_u128(rng.random())))
                        .collect();
                    let mut matrix = vec![0; 4 * cap];
                    for b in 0..4 {
                        if let Some(scale) = scales[b] {
                            for i in 0..cap {
                                matrix[b * cap + i] = ctx.mul_raw(scale, weights[i]);
                            }
                        }
                    }
                    let raw_claim = matrix
                        .iter()
                        .zip(&projected)
                        .fold(0, |a, (&m, &w)| ctx.add_raw(a, ctx.mul_raw(m, w)));
                    let claim = crate::utils::delayed_reduction::element(&cfg, raw_claim);
                    let expected = crate::sumcheck::inner::prove_inner_sumcheck(
                        &ctx,
                        &mut Blake3Transcript::new(),
                        claim.clone(),
                        RawWitness::Field(projected.clone()),
                        crate::sumcheck::inner::native::NativeWeights::Dense {
                            matrix: matrix.clone(),
                            live: 4 * cap,
                        },
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                    .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                    .unwrap();
                    let got = crate::sumcheck::inner::prove_inner_sumcheck(
                        &ctx,
                        &mut Blake3Transcript::new(),
                        claim.clone(),
                        RawWitness::Limbs(input),
                        crate::sumcheck::inner::native::NativeWeights::Dense {
                            matrix: matrix,
                            live: 4 * cap,
                        },
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                    .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                    .unwrap();
                    assert_eq!(got, expected, "dense log={log}");
                    if cap >= 2 {
                        let got = crate::sumcheck::inner::prove_inner_sumcheck(
                            &ctx,
                            &mut Blake3Transcript::new(),
                            claim,
                            RawWitness::Limbs(input),
                            crate::sumcheck::inner::native::NativeWeights::Blocks {
                                weights: (&weights).to_vec(),
                                scales: (&BlockScales {
                                    block_len: cap,
                                    rows: cap,
                                    scales,
                                })
                                    .clone(),
                                live: 4 * cap,
                                num_vars: log + 2,
                            },
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                        .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                        .unwrap();
                        assert_eq!(got, expected, "structured log={log}");
                    }
                }
            }
        }
    }

    #[test]
    fn segmented_witness_matches_dense_and_structured_projection() {
        let mut rng = StdRng::seed_from_u64(0x7769_6465_5f696e6e);
        for q in [(1u128 << 100) - 15, u128::MAX - 158] {
            let cfg = Field::make_cfg(&Uint::from_words(raw_to_words(q))).unwrap();
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let ctx = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            for log in [0, 1, 2, 6, 13] {
                let cap = 1usize << log;
                for live in [cap, cap.saturating_sub(1).max(1)] {
                    let mut values = || {
                        (0..live)
                            .map(|i| {
                                if i < 4 {
                                    [u128::MAX, 0, 1u128 << 127, 1][i]
                                } else {
                                    rng.random()
                                }
                            })
                            .collect::<Vec<u128>>()
                    };
                    let x = values();
                    let y = values();
                    let lo = values();
                    let hi = values();
                    let input = NativeBlockWitness::new(cap, &x, &y, &lo, &hi);
                    let two128 = cfg.mul(
                        &(Field::from_with_cfg(1u128 << 127, &cfg)),
                        &(&Field::from_with_cfg(2u64, &cfg)),
                    );
                    let mut projected = vec![0; 4 * cap];
                    projected[0] = ctx.one_raw();
                    for i in 0..live {
                        projected[cap + i] = ctx.raw(&Field::from_with_cfg(x[i], &cfg));
                        projected[2 * cap + i] = ctx.raw(&Field::from_with_cfg(y[i], &cfg));
                        projected[3 * cap + i] = ctx.raw(
                            &(cfg.add(
                                &(Field::from_with_cfg(lo[i], &cfg)),
                                &(&(cfg.mul(
                                    &(two128.clone()),
                                    &(&Field::from_with_cfg(hi[i], &cfg)),
                                ))),
                            )),
                        );
                    }
                    let weights: Vec<_> = (0..cap)
                        .map(|_| ctx.native_residue_u128(rng.random()))
                        .collect();
                    for active in [[false, true, true, true], [true, false, true, false]] {
                        let scales: Vec<_> = active
                            .iter()
                            .map(|&v| v.then(|| ctx.native_residue_u128(rng.random())))
                            .collect();
                        let mut matrix = vec![0; 4 * cap];
                        for b in 0..4 {
                            if let Some(scale) = scales[b] {
                                for i in 0..cap {
                                    matrix[b * cap + i] = ctx.mul_raw(scale, weights[i]);
                                }
                            }
                        }
                        let claim = crate::utils::delayed_reduction::element(
                            &cfg,
                            matrix
                                .iter()
                                .zip(&projected)
                                .fold(0, |a, (&m, &w)| ctx.add_raw(a, ctx.mul_raw(m, w))),
                        );
                        let expected = crate::sumcheck::inner::prove_inner_sumcheck(
                            &ctx,
                            &mut Blake3Transcript::new(),
                            claim.clone(),
                            RawWitness::Field(projected.clone()),
                            crate::sumcheck::inner::native::NativeWeights::Dense {
                                matrix: matrix.clone(),
                                live: 4 * cap,
                            },
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                        .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                        .unwrap();
                        let got = crate::sumcheck::inner::prove_inner_sumcheck(
                            &ctx,
                            &mut Blake3Transcript::new(),
                            claim.clone(),
                            RawWitness::Wide(input),
                            crate::sumcheck::inner::native::NativeWeights::Dense {
                                matrix: matrix,
                                live: 4 * cap,
                            },
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                        .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                        .unwrap();
                        assert_eq!(got, expected, "dense log={log} live={live}");
                        if cap >= 2 {
                            let got = crate::sumcheck::inner::prove_inner_sumcheck(
                                &ctx,
                                &mut Blake3Transcript::new(),
                                claim,
                                RawWitness::Wide(input),
                                crate::sumcheck::inner::native::NativeWeights::Blocks {
                                    weights: (&weights).to_vec(),
                                    scales: (&BlockScales {
                                        block_len: cap,
                                        rows: cap,
                                        scales,
                                    })
                                        .clone(),
                                    live: 4 * cap,
                                    num_vars: log + 2,
                                },
                                &mut crate::sumcheck::UngrindedRoundBoundary,
                            )
                            .map(crate::piop::spartan::sumcheck::InnerSumcheckOutput::from)
                            .unwrap();
                            assert_eq!(got, expected, "structured log={log} live={live}");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn compact_u32_u64_blocks_match_dense_proofs_and_transcripts() {
        use crate::{
            sumcheck::{UngrindedRoundBoundary, inner::prove_inner_sumcheck},
            transcript::traits::Transcript,
        };
        let mut rng = StdRng::seed_from_u64(0x636f6d70616374);
        let cfg = Field::make_cfg(&Uint::from((1u128 << 100) - 15)).unwrap();
        let ctx = field_context(&cfg);
        for log in [0, 1, 2, 6, 13] {
            let cap = 1usize << log;
            for live in [cap, cap.saturating_sub(1).max(1)] {
                let values: [Vec<u64>; 4] = core::array::from_fn(|b| {
                    (0..live)
                        .map(|i| match i % 5 {
                            0 => u64::MAX,
                            1 => 0,
                            2 => 1u64 << (b * 16),
                            _ => rng.random(),
                        })
                        .collect()
                });
                let narrow = values
                    .each_ref()
                    .map(|v| v.iter().map(|&x| x as u32).collect::<Vec<_>>());
                for bits in [32, 64] {
                    let input = if bits == 32 {
                        NativeBlockWitness::u32(cap, &narrow[0], &narrow[1], &narrow[2], &narrow[3])
                    } else {
                        NativeBlockWitness::u64(cap, &values[0], &values[1], &values[2], &values[3])
                    };
                    let blocks = if bits == 32 { 4 } else { 8 };
                    let mut native = vec![0; blocks * cap];
                    native[0] = 1;
                    for i in 0..live {
                        if bits == 32 {
                            native[cap + i] = u64::from(narrow[0][i]);
                            native[2 * cap + i] = u64::from(narrow[1][i]);
                            native[3 * cap + i] =
                                u64::from(narrow[2][i]) | (u64::from(narrow[3][i]) << 32);
                        } else {
                            for b in 0..4 {
                                native[(b + 1) * cap + i] = values[b][i];
                            }
                        }
                    }
                    assert_eq!(
                        (0..native.len())
                            .map(|i| input.read_u64(i))
                            .collect::<Vec<_>>(),
                        native
                    );
                    let weights: Vec<_> = (0..cap)
                        .map(|_| ctx.native_residue_u128(rng.random()))
                        .collect();
                    for alternate in [false, true] {
                        let scales: Vec<_> = (0..blocks)
                            .map(|b| {
                                ((b % 2 == 0) == alternate)
                                    .then(|| ctx.native_residue_u128(rng.random()))
                            })
                            .collect();
                        let matrix: Vec<_> = scales
                            .iter()
                            .flat_map(|s| {
                                weights.iter().map(|&w| s.map_or(0, |s| ctx.mul_raw(s, w)))
                            })
                            .collect();
                        let projected: Vec<_> =
                            native.iter().map(|&v| ctx.native_residue(v)).collect();
                        let claim = shared_raw(
                            &ctx,
                            matrix
                                .iter()
                                .zip(&projected)
                                .fold(0, |acc, (&m, &w)| ctx.add_raw(acc, ctx.mul_raw(m, w))),
                        );
                        let mut expected_t = Blake3Transcript::new();
                        let expected = prove_inner_sumcheck(
                            &ctx,
                            &mut expected_t,
                            claim,
                            RawWitness::Field(projected),
                            NativeWeights::Dense {
                                matrix: matrix.clone(),
                                live: native.len(),
                            },
                            &mut UngrindedRoundBoundary,
                        )
                        .unwrap();
                        let expected_tail = expected_t.get_challenges::<u8>(32);
                        for structured in [false, true] {
                            if structured && cap == 1 {
                                continue;
                            }
                            let weights = if structured {
                                NativeWeights::Blocks {
                                    weights: weights.clone(),
                                    scales: BlockScales {
                                        block_len: cap,
                                        rows: cap,
                                        scales: scales.clone(),
                                    },
                                    live: native.len(),
                                    num_vars: log + blocks.ilog2() as usize,
                                }
                            } else {
                                NativeWeights::Dense {
                                    matrix: matrix.clone(),
                                    live: native.len(),
                                }
                            };
                            let mut transcript = Blake3Transcript::new();
                            let got = prove_inner_sumcheck(
                                &ctx,
                                &mut transcript,
                                claim,
                                RawWitness::Wide(input),
                                weights,
                                &mut UngrindedRoundBoundary,
                            )
                            .unwrap();
                            assert_eq!(
                                got, expected,
                                "bits={bits} log={log} live={live} structured={structured}"
                            );
                            assert_eq!(transcript.get_challenges::<u8>(32), expected_tail);
                        }
                    }
                }
            }
        }
    }
}
