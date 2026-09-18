//! Shared bucket scheduling and fused mixed-input folding.
use super::{OuterArithmetic, inputs::OuterRows, ordinary::*};
use crate::piop::spartan::SpartanField;
#[cfg(feature = "parallel")]
use crate::sumcheck::arithmetic::{outer_fold_grain, parallel_outer_fold, should_parallelize};
use crate::sumcheck::{
    SumcheckError,
    arithmetic::{merge_accumulators, reduce_two_accumulators_bounded, reduce_two_prepared},
};
use field::PreparedLinearCombination;
use field::{BatchMulAcc, MergeAccumulator, Reduce};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A prepared mixed fold, shared by ordinary and prefix traversal. Keeping the
/// consuming operation as an inlined method avoids a large closure return
/// through temporary stack storage in the inner loop.
pub(super) struct PreparedFold<'a, F, I: OuterRows, const TERMS: usize>
where
    F: OuterArithmetic<I::AB, I::C> + 'a,
{
    rows: &'a I,
    ab: <F as PreparedLinearCombination<I::AB>>::Prepared<'a, TERMS>,
    c: <F as PreparedLinearCombination<I::C>>::Prepared<'a, TERMS>,
}
impl<'a, F, I: OuterRows, const TERMS: usize> PreparedFold<'a, F, I, TERMS>
where
    F: OuterArithmetic<I::AB, I::C>,
{
    pub(super) fn new(field: &'a F, coefficients: [F::Elem; TERMS], rows: &'a I) -> Self {
        Self {
            rows,
            ab: <F as PreparedLinearCombination<I::AB>>::prepare_linear_combination(
                field,
                coefficients,
            ),
            c: <F as PreparedLinearCombination<I::C>>::prepare_linear_combination(
                field,
                coefficients,
            ),
        }
    }
}
pub(super) trait FoldRows<E>: Sync {
    fn fold(&self, row: usize) -> [E; 3];
}
impl<F, I: OuterRows, const TERMS: usize> FoldRows<F::Elem> for PreparedFold<'_, F, I, TERMS>
where
    F: OuterArithmetic<I::AB, I::C>,
{
    #[inline(always)]
    fn fold(&self, row: usize) -> [F::Elem; 3] {
        [
            <F as PreparedLinearCombination<I::AB>>::linear_combination(&self.ab, |j| {
                self.rows.a(TERMS * row + j)
            }),
            <F as PreparedLinearCombination<I::AB>>::linear_combination(&self.ab, |j| {
                self.rows.b(TERMS * row + j)
            }),
            <F as PreparedLinearCombination<I::C>>::linear_combination(&self.c, |j| {
                self.rows.c(TERMS * row + j)
            }),
        ]
    }
}

/// Equality buckets keep exact integer residuals integral until each subtotal
/// is reduced. Outer merging uses the same exact field-product accumulator.
pub(super) fn integer_buckets<F, AB, C, const LANES: usize>(
    field: &F,
    weights: StrippedEqualityWeights<'_, F::Elem>,
    accumulate: impl Fn(
        &mut [<F as OuterArithmetic<AB, C>>::Accumulator; LANES],
        &F::Weights,
        usize,
        usize,
    ) + Sync,
) -> Result<[F::Elem; LANES], SumcheckError>
where
    F: OuterArithmetic<AB, C>,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    let one = [field.one()];
    let (low, high) = weights.buckets(&one);
    assert!(
        low.len() as u64 <= 1u64 << 32,
        "equality bucket exceeds accumulator bound"
    );
    let prepared = field.prepare_weights(low);
    let zero = || core::array::from_fn(|_| <F as BatchMulAcc<F::Elem>>::Accumulator::zero());
    let bucket = |mut outer: [<F as BatchMulAcc<F::Elem>>::Accumulator; LANES], h: usize| {
        let mut inner = core::array::from_fn(|_| field.accumulator());
        for l in 0..low.len() {
            accumulate(&mut inner, &prepared, h * low.len() + l, l);
        }
        for (out, inner) in outer.iter_mut().zip(inner) {
            field.mul_acc(out, &high[h], &field.finish_accumulator(&prepared, inner));
        }
        outer
    };
    #[cfg(feature = "parallel")]
    let total = if should_parallelize(low.len() * high.len()) {
        (0..high.len())
            .into_par_iter()
            .with_min_len(1024usize.div_ceil(low.len()))
            .fold(zero, bucket)
            .reduce(zero, |a, b| merge_accumulators(a, b))
    } else {
        (0..high.len()).fold(zero(), bucket)
    };
    #[cfg(not(feature = "parallel"))]
    let total = (0..high.len()).fold(zero(), bucket);
    let mut result = [field.zero(); LANES];
    for (r, a) in result.iter_mut().zip(total) {
        *r = field.prepare_reduce(high.len())(a);
    }
    Ok(result)
}

/// Fold directly from any borrowed input representation. When another round
/// remains, compute its cofactor while each newly folded pair is in registers.
pub(super) fn fold_and_message<F>(
    field: &F,
    products: &mut R1csProductTableBuffers<F::Elem>,
    factors: &mut EqualityFactors<F::Elem>,
    endpoint: FactoredEndpoint,
    fold: impl FoldRows<F::Elem>,
) -> Result<Option<[F::Elem; 2]>, SumcheckError>
where
    F: field::FieldOps,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    if products.az.len() == 1 {
        let [a, b, c] = fold.fold(0);
        products.az[0] = a;
        products.bz[0] = b;
        products.cz[0] = c;
        return Ok(None);
    }
    factors.strip(field);
    let weights = factors.weights();
    let one = [field.one()];
    let (low, high) = weights.buckets(&one);
    let chunk = 2 * low.len();
    fold_buckets(field, products, low, high, chunk, endpoint, fold)
}

fn fold_buckets<F>(
    field: &F,
    products: &mut R1csProductTableBuffers<F::Elem>,
    low: &[F::Elem],
    high: &[F::Elem],
    chunk: usize,
    endpoint: FactoredEndpoint,
    fold: impl FoldRows<F::Elem>,
) -> Result<Option<[F::Elem; 2]>, SumcheckError>
where
    F: field::FieldOps,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    let inner_reduction = field.prepare_reduce(low.len());
    let zero = || core::array::from_fn(|_| <F as BatchMulAcc<F::Elem>>::Accumulator::zero());
    let bucket = |h: usize, a: &mut [F::Elem], b: &mut [F::Elem], c: &mut [F::Elem]| {
        let mut inner = zero();
        for (l, w) in low.iter().enumerate() {
            let i = 2 * l;
            let [a0, b0, c0] = fold.fold(h * chunk + i);
            let [a1, b1, c1] = fold.fold(h * chunk + i + 1);
            a[i] = a0;
            a[i + 1] = a1;
            b[i] = b0;
            b[i + 1] = b1;
            c[i] = c0;
            c[i + 1] = c1;
            accumulate_eq_factored_cofactor_evaluations(
                &mut inner, w, endpoint, &a0, &a1, &b0, &b1, &c0, &c1, field, field,
            );
        }
        let values = reduce_two_prepared(inner, &inner_reduction);
        let mut out = zero();
        for (a, v) in out.iter_mut().zip(values) {
            field.mul_acc(a, &high[h], &v);
        }
        Ok::<_, SumcheckError>(out)
    };
    #[cfg(feature = "parallel")]
    if parallel_outer_fold(products.az.len() / 2) {
        let min_buckets = outer_fold_grain(products.az.len() / 2).div_ceil(low.len());
        let total = (
            products.az.par_chunks_mut(chunk),
            products.bz.par_chunks_mut(chunk),
            products.cz.par_chunks_mut(chunk),
        )
            .into_par_iter()
            .with_min_len(min_buckets)
            .enumerate()
            .try_fold(zero, |outer, (h, (a, b, c))| {
                Ok::<_, SumcheckError>(merge_accumulators(outer, bucket(h, a, b, c)?))
            })
            .try_reduce(zero, |a, b| Ok(merge_accumulators(a, b)))?;
        return Ok(Some(reduce_two_accumulators_bounded(
            total,
            field,
            high.len(),
            field,
        )?));
    }
    let mut total = zero();
    for (h, ((a, b), c)) in products
        .az
        .chunks_mut(chunk)
        .zip(products.bz.chunks_mut(chunk))
        .zip(products.cz.chunks_mut(chunk))
        .enumerate()
    {
        total = merge_accumulators(total, bucket(h, a, b, c)?);
    }
    Ok(Some(reduce_two_accumulators_bounded(
        total,
        field,
        high.len(),
        field,
    )?))
}
