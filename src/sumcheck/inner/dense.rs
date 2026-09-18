use super::{Acc, Elem};
use field::{BatchMulAcc, FieldOps, MergeAccumulator, PreparedLinearCombination, Reduce};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(super) struct Tables<T, E> {
    source: Option<Vec<T>>,
    values: Vec<E>,
    weights: Vec<E>,
    values_scratch: Vec<E>,
    weights_scratch: Vec<E>,
}
impl<T: Copy, E: Copy> Tables<T, E> {
    pub(super) fn new(values: Vec<T>, weights: Vec<E>) -> Self {
        Self {
            source: Some(values),
            values: Vec::new(),
            weights,
            values_scratch: Vec::new(),
            weights_scratch: Vec::new(),
        }
    }
    pub(super) fn terminal<F: PreparedLinearCombination<T, Elem = E>>(&self, field: &F) -> [E; 2] {
        let value = if let Some(source) = &self.source {
            let prepared = field.prepare_linear_combination([field.one()]);
            F::linear_combination(&prepared, |_| source[0])
        } else {
            self.values[0]
        };
        [self.weights[0], value]
    }
}

pub(super) fn first_round<F, T>(field: &F, tables: &Tables<T, Elem<F>>) -> [Elem<F>; 2]
where
    F: FieldOps + BatchMulAcc<Elem<F>, T> + Reduce<Acc<F, T>, Output = Elem<F>> + Sync,
    T: Copy + Send + Sync,
{
    let values = tables.source.as_ref().unwrap();
    let zero = || [Acc::<F, T>::zero(), Acc::<F, T>::zero()];
    let accumulate = |mut acc: [Acc<F, T>; 2], (weights, values): (&[Elem<F>], &[T])| {
        let delta = field.sub(&weights[1], &weights[0]);
        field.mul_acc(&mut acc[0], &weights[0], &values[0]);
        // Δw*(v1-v0) = Δw*v1 + (-Δw)*v0, with no native-width subtraction.
        field.mul_acc(&mut acc[1], &delta, &values[1]);
        field.mul_acc(&mut acc[1], &field.neg(&delta), &values[0]);
        acc
    };
    #[cfg(feature = "parallel")]
    let acc = if crate::sumcheck::arithmetic::should_parallelize(values.len() / 2) {
        tables
            .weights
            .par_chunks_exact(2)
            .zip(values.par_chunks_exact(2))
            .fold(zero, accumulate)
            .reduce(zero, crate::sumcheck::arithmetic::merge_accumulators)
    } else {
        tables
            .weights
            .chunks_exact(2)
            .zip(values.chunks_exact(2))
            .fold(zero(), accumulate)
    };
    #[cfg(not(feature = "parallel"))]
    let acc = tables
        .weights
        .chunks_exact(2)
        .zip(values.chunks_exact(2))
        .fold(zero(), accumulate);
    let reduce = field.prepare_reduce(values.len());
    acc.map(reduce)
}

/// Fold two output rows and accumulate their next-round contribution before
/// they leave registers. Input conversion is part of `read_folded`.
fn fold_into<F>(
    field: &F,
    weights: &[Elem<F>],
    weights_out: &mut [Elem<F>],
    values_out: &mut [Elem<F>],
    challenge: &Elem<F>,
    read_folded: impl Fn(usize) -> Elem<F> + Sync,
) -> [Elem<F>; 2]
where
    F: FieldOps + BatchMulAcc<Elem<F>> + Reduce<Acc<F, Elem<F>>, Output = Elem<F>> + Sync,
{
    let zero = || [Acc::<F, Elem<F>>::zero(), Acc::<F, Elem<F>>::zero()];
    let fold_weight = |i| {
        field.add(
            &weights[2 * i],
            &field.mul(challenge, &field.sub(&weights[2 * i + 1], &weights[2 * i])),
        )
    };
    if weights_out.len() == 1 {
        weights_out[0] = fold_weight(0);
        values_out[0] = read_folded(0);
        return [field.zero(); 2];
    }
    let contribution =
        |mut acc: [Acc<F, Elem<F>>; 2],
         (pair, (w, v)): (usize, (&mut [Elem<F>], &mut [Elem<F>]))| {
            let i = 2 * pair;
            w[0] = fold_weight(i);
            w[1] = fold_weight(i + 1);
            v[0] = read_folded(i);
            v[1] = read_folded(i + 1);
            field.mul_acc(&mut acc[0], &w[0], &v[0]);
            field.mul_acc(
                &mut acc[1],
                &field.sub(&w[1], &w[0]),
                &field.sub(&v[1], &v[0]),
            );
            acc
        };
    let pairs = weights_out.len() / 2;
    #[cfg(feature = "parallel")]
    let acc = if crate::sumcheck::arithmetic::should_parallelize(pairs) {
        weights_out
            .par_chunks_exact_mut(2)
            .zip(values_out.par_chunks_exact_mut(2))
            .enumerate()
            .fold(zero, contribution)
            .reduce(zero, crate::sumcheck::arithmetic::merge_accumulators)
    } else {
        weights_out
            .chunks_exact_mut(2)
            .zip(values_out.chunks_exact_mut(2))
            .enumerate()
            .fold(zero(), contribution)
    };
    #[cfg(not(feature = "parallel"))]
    let acc = weights_out
        .chunks_exact_mut(2)
        .zip(values_out.chunks_exact_mut(2))
        .enumerate()
        .fold(zero(), contribution);
    let reduce = field.prepare_reduce(pairs);
    acc.map(reduce)
}

pub(super) fn fold_round<F, T>(
    field: &F,
    tables: &mut Tables<T, Elem<F>>,
    challenge: &Elem<F>,
) -> [Elem<F>; 2]
where
    F: FieldOps
        + PreparedLinearCombination<T>
        + BatchMulAcc<Elem<F>>
        + Reduce<Acc<F, Elem<F>>, Output = Elem<F>>
        + Sync,
    T: Copy + Send + Sync,
{
    let n = tables.weights.len() / 2;
    tables.weights_scratch.resize(n, field.zero());
    tables.values_scratch.resize(n, field.zero());
    let next = if let Some(source) = tables.source.take() {
        let fold =
            field.prepare_linear_combination([field.sub(&field.one(), challenge), *challenge]);
        fold_into(
            field,
            &tables.weights,
            &mut tables.weights_scratch,
            &mut tables.values_scratch,
            challenge,
            |i| F::linear_combination(&fold, |j| source[2 * i + j]),
        )
    } else {
        fold_into(
            field,
            &tables.weights,
            &mut tables.weights_scratch,
            &mut tables.values_scratch,
            challenge,
            |i| {
                field.add(
                    &tables.values[2 * i],
                    &field.mul(
                        challenge,
                        &field.sub(&tables.values[2 * i + 1], &tables.values[2 * i]),
                    ),
                )
            },
        )
    };
    core::mem::swap(&mut tables.weights, &mut tables.weights_scratch);
    core::mem::swap(&mut tables.values, &mut tables.values_scratch);
    next
}

pub struct State<T, E> {
    tables: Tables<T, E>,
    coefficients: [E; 2],
    num_vars: usize,
}
impl<T: Copy + Send + Sync, E: Copy> State<T, E> {
    pub(super) fn new<F>(
        field: &F,
        values: Vec<T>,
        weights: Vec<E>,
    ) -> Result<Self, super::SumcheckError>
    where
        F: FieldOps<Elem = E> + BatchMulAcc<E, T> + Reduce<Acc<F, T>, Output = E> + Sync,
    {
        if !values.len().is_power_of_two() || values.len() != weights.len() {
            return Err(super::SumcheckError::InvalidProductDimensions);
        }
        let num_vars = values.len().ilog2() as usize;
        let tables = Tables::new(values, weights);
        let coefficients = if num_vars == 0 {
            [field.zero(); 2]
        } else {
            first_round(field, &tables)
        };
        Ok(Self {
            tables,
            coefficients,
            num_vars,
        })
    }
}
impl<F, T> super::input::State<F> for State<T, Elem<F>>
where
    F: FieldOps
        + PreparedLinearCombination<T>
        + BatchMulAcc<Elem<F>, T>
        + BatchMulAcc<Elem<F>>
        + Sync,
    F: Reduce<Acc<F, T>, Output = Elem<F>> + Reduce<Acc<F, Elem<F>>, Output = Elem<F>>,
    T: Copy + Send + Sync,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }
    fn coefficients(&self, _: &F) -> Result<[Elem<F>; 2], super::SumcheckError> {
        Ok(self.coefficients)
    }
    fn fold(&mut self, field: &F, r: &Elem<F>) -> Result<(), super::SumcheckError> {
        self.coefficients = fold_round(field, &mut self.tables, r);
        Ok(())
    }
    fn terminal(&self, field: &F) -> Result<[Elem<F>; 2], super::SumcheckError> {
        Ok(self.tables.terminal(field))
    }
}
