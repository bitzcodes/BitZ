//! Typed folded witness storage. Its type selects the unique MAC scale.
use super::*;
use field::{Fp, FpLinearAcc, FpProductAcc, MergeAccumulator, Uint};

pub(super) trait FoldedValue: Copy + Send + Sync {
    type Acc: Copy + Send + MergeAccumulator;
    fn zero(ctx: &field::FpCtx<2>) -> Self;
    fn encoding(self) -> Raw;
    fn from_encoding(ctx: &field::FpCtx<2>, value: Raw) -> Self;
    fn final_raw(self, ctx: &field::FpCtx<2>) -> Raw;
    fn accumulate(ctx: &field::FpCtx<2>, acc: &mut Self::Acc, weight: Raw, value: Self);
    fn reduce(ctx: &field::FpCtx<2>, acc: Self::Acc) -> Raw;
    fn fold_initial(
        ctx: &field::FpCtx<2>,
        block: BlockValues<'_>,
        weights: &[Raw],
        out: &mut [Self],
        challenge: Raw,
    ) -> [Raw; 2];
}
impl FoldedValue for Uint<2> {
    type Acc = FpLinearAcc<2, 2>;
    fn zero(_: &field::FpCtx<2>) -> Self {
        Self::ZERO
    }
    #[inline(always)]
    fn encoding(self) -> Raw {
        words_to_raw(self.as_words())
    }
    #[inline(always)]
    fn from_encoding(_: &field::FpCtx<2>, value: Raw) -> Self {
        Self::from_words(raw_to_words(value))
    }
    #[inline(always)]
    fn final_raw(self, ctx: &field::FpCtx<2>) -> Raw {
        ctx.plain_to_raw(self.encoding())
    }
    #[inline(always)]
    fn accumulate(ctx: &field::FpCtx<2>, acc: &mut Self::Acc, weight: Raw, value: Self) {
        acc.accumulate(&shared_raw(ctx, weight), &value);
    }
    #[inline(always)]
    fn reduce(ctx: &field::FpCtx<2>, acc: Self::Acc) -> Raw {
        raw_shared(field::Reduce::reduce(ctx, acc))
    }
    fn fold_initial(
        ctx: &field::FpCtx<2>,
        block: BlockValues<'_>,
        weights: &[Raw],
        out: &mut [Self],
        challenge: Raw,
    ) -> [Raw; 2] {
        block.fold_integer_block(ctx, weights, out, challenge)
    }
}
impl FoldedValue for Fp<2> {
    type Acc = FpProductAcc<2>;
    fn zero(ctx: &field::FpCtx<2>) -> Self {
        shared_raw(ctx, 0)
    }
    #[inline(always)]
    fn encoding(self) -> Raw {
        raw_shared(self)
    }
    #[inline(always)]
    fn from_encoding(ctx: &field::FpCtx<2>, value: Raw) -> Self {
        shared_raw(ctx, value)
    }
    #[inline(always)]
    fn final_raw(self, _: &field::FpCtx<2>) -> Raw {
        raw_shared(self)
    }
    #[inline(always)]
    fn accumulate(ctx: &field::FpCtx<2>, acc: &mut Self::Acc, weight: Raw, value: Self) {
        acc.accumulate(&shared_raw(ctx, weight), &value);
    }
    #[inline(always)]
    fn reduce(ctx: &field::FpCtx<2>, acc: Self::Acc) -> Raw {
        raw_shared(field::Reduce::reduce(ctx, acc))
    }
    fn fold_initial(
        ctx: &field::FpCtx<2>,
        block: BlockValues<'_>,
        weights: &[Raw],
        out: &mut [Self],
        challenge: Raw,
    ) -> [Raw; 2] {
        let BlockValues::Field(values) = block else {
            unreachable!("field source dispatch")
        };
        fold_round::<Self, false, _>(
            ctx,
            weights,
            &values[..2 * out.len()],
            |value| shared_raw(ctx, value),
            &mut [],
            out,
            challenge,
        )
    }
}

pub(super) fn pair<W: FoldedValue>() -> [W::Acc; 2] {
    [W::Acc::zero(); 2]
}
pub(super) fn reduce<W: FoldedValue>(ctx: &field::FpCtx<2>, a: [W::Acc; 2]) -> [Raw; 2] {
    a.map(|a| W::reduce(ctx, a))
}

pub(super) fn fold_round<W: FoldedValue, const FOLD_WEIGHTS: bool, I: Copy + Sync>(
    ctx: &field::FpCtx<2>,
    weights: &[Raw],
    input: &[I],
    read: impl Fn(I) -> W + Sync,
    matrix_out: &mut [Raw],
    out: &mut [W],
    challenge: Raw,
) -> [Raw; 2] {
    assert_eq!(out.len() % 2, 0);
    assert_eq!(input.len(), 2 * out.len());
    assert_eq!(weights.len(), out.len() * if FOLD_WEIGHTS { 2 } else { 1 });
    assert_eq!(matrix_out.len(), if FOLD_WEIGHTS { out.len() } else { 0 });
    let block = |weights: &[Raw], input: &[I], mout: &mut [Raw], out: &mut [W]| {
        let weight_stride = if FOLD_WEIGHTS { 4 } else { 2 };
        let mut acc = pair::<W>();
        for (pair, ((weights, values), z)) in weights
            .chunks_exact(weight_stride)
            .zip(input.chunks_exact(4))
            .zip(out.chunks_exact_mut(2))
            .enumerate()
        {
            let z0 = W::from_encoding(
                ctx,
                ctx.interpolate(
                    read(values[0]).encoding(),
                    read(values[1]).encoding(),
                    challenge,
                ),
            );
            let z1 = W::from_encoding(
                ctx,
                ctx.interpolate(
                    read(values[2]).encoding(),
                    read(values[3]).encoding(),
                    challenge,
                ),
            );
            z[0] = z0;
            z[1] = z1;
            let m = if FOLD_WEIGHTS {
                [
                    ctx.interpolate(weights[0], weights[1], challenge),
                    ctx.interpolate(weights[2], weights[3], challenge),
                ]
            } else {
                [weights[0], weights[1]]
            };
            if FOLD_WEIGHTS {
                mout[2 * pair] = m[0];
                mout[2 * pair + 1] = m[1];
            }
            W::accumulate(ctx, &mut acc[0], m[0], z0);
            W::accumulate(
                ctx,
                &mut acc[1],
                ctx.sub_raw(m[1], m[0]),
                W::from_encoding(ctx, ctx.sub_raw(z1.encoding(), z0.encoding())),
            );
        }
        acc
    };
    #[cfg(feature = "parallel")]
    if parallel(out.len() / 2) {
        let acc = if FOLD_WEIGHTS {
            (
                weights.par_chunks(2 * FOLD_BLOCK),
                input.par_chunks(2 * FOLD_BLOCK),
                matrix_out.par_chunks_mut(FOLD_BLOCK),
                out.par_chunks_mut(FOLD_BLOCK),
            )
                .into_par_iter()
                .map(|(w, values, m, z)| block(w, values, m, z))
                .reduce(pair::<W>, merge_accumulators::<W::Acc, 2>)
        } else {
            (
                weights.par_chunks(FOLD_BLOCK),
                input.par_chunks(2 * FOLD_BLOCK),
                out.par_chunks_mut(FOLD_BLOCK),
            )
                .into_par_iter()
                .map(|(w, values, z)| block(w, values, &mut [], z))
                .reduce(pair::<W>, merge_accumulators::<W::Acc, 2>)
        };
        return reduce::<W>(ctx, acc);
    }
    reduce::<W>(ctx, block(weights, input, matrix_out, out))
}
