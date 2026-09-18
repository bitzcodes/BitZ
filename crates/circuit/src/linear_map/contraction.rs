//! CSC contraction scheduling and monomorphized arithmetic kernels.
use super::{
    BilinearEval, ColumnValues, CscMatrix, LeftMul, LinearMapError, SparseIndex, SparseSegment,
};
use field::{BatchMulAcc, MergeAccumulator, Reduce, RingOps};
use rayon::prelude::*;

/// Disjoint contiguous column ranges; sparse backends retain control of each
/// dot product, so implicit ones and singleton columns never require a MAC.
pub fn columns_into<E: Send>(out: &mut [E], parallel: bool, evaluate: impl Fn(usize) -> E + Sync) {
    column_chunks_into::<4096, _>(out, parallel, evaluate);
}

fn column_chunks_into<const BLOCK: usize, E: Send>(
    out: &mut [E],
    parallel: bool,
    evaluate: impl Fn(usize) -> E + Sync,
) {
    const PARALLEL_THRESHOLD: usize = 1 << 12;
    if parallel && out.len() >= PARALLEL_THRESHOLD && rayon::current_num_threads() > 1 {
        if BLOCK == 1 {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, dst)| *dst = evaluate(i));
            return;
        }
        out.par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(|(b, chunk)| {
                for (i, dst) in chunk.iter_mut().enumerate() {
                    *dst = evaluate(b * BLOCK + i);
                }
            });
    } else {
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = evaluate(i);
        }
    }
}

/// Σ_i c_i w[row_i], preserving the caller's coefficient-specific fast paths.
#[inline]
pub fn segment_dot<C, E: Copy, I: SparseIndex>(
    column: SparseSegment<'_, Box<[C]>, I>,
    zero: E,
    mut scale: impl FnMut(usize, &C) -> E,
    mut add: impl FnMut(E, E) -> E,
) -> E {
    let mut entries = column.indices().iter().zip(column.coefficients());
    let Some((&row, c)) = entries.next() else {
        return zero;
    };
    let mut sum = scale(row.to_usize(), c);
    for (&row, c) in entries {
        sum = add(sum, scale(row.to_usize(), c));
    }
    sum
}

fn dimensions(
    rows: usize,
    columns: usize,
    actual_rows: usize,
    actual_columns: usize,
) -> Result<(), LinearMapError> {
    for (kind, expected, actual) in [
        ("row weights", rows, actual_rows),
        ("column values", columns, actual_columns),
    ] {
        if expected != actual {
            return Err(LinearMapError::Length {
                kind,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

/// Signed contraction uses unsigned mixed MAC, preserving cheap singleton ±1
/// and avoiding signed overflow even for i64::MIN.
#[inline]
fn signed_segment<F, I: SparseIndex>(
    field: &F,
    segment: SparseSegment<'_, Box<[i64]>, I>,
    weights: &[F::Elem],
    reduce: &impl Fn(<F as BatchMulAcc<F::Elem, u64>>::Accumulator) -> F::Elem,
) -> F::Elem
where
    F: RingOps + BatchMulAcc<F::Elem, u64>,
{
    if let Some((row, c)) = segment.single() {
        match *c {
            1 => return weights[row],
            -1 => return field.neg(&weights[row]),
            _ => {}
        }
    }
    let mut acc = <F as BatchMulAcc<F::Elem, u64>>::Accumulator::zero();
    for (row, c) in segment {
        let weight = if *c < 0 {
            field.neg(&weights[row])
        } else {
            weights[row]
        };
        field.mul_acc(&mut acc, &weight, &c.unsigned_abs());
    }
    reduce(acc)
}
/// Keeps the qualified signed-coefficient kernel behind mathematical operations.
pub struct PreparedSignedSparse<'a, F: RingOps, I = usize> {
    field: &'a F,
    matrix: &'a CscMatrix<Box<[i64]>, I>,
    max_terms: usize,
    parallel: bool,
}
impl<'a, F: RingOps, I: SparseIndex> PreparedSignedSparse<'a, F, I> {
    pub fn new(field: &'a F, matrix: &'a CscMatrix<Box<[i64]>, I>) -> Self {
        Self {
            field,
            matrix,
            // The total nonzero count bounds each column without rescanning
            // topology when preparing the native linear-product reducer.
            max_terms: matrix.nnz().max(1),
            parallel: true,
        }
    }
    pub fn serial(mut self) -> Self {
        self.parallel = false;
        self
    }
}
impl<F, I> LeftMul<F::Elem> for PreparedSignedSparse<'_, F, I>
where
    F: RingOps
        + Sync
        + BatchMulAcc<F::Elem, u64>
        + Reduce<<F as BatchMulAcc<F::Elem, u64>>::Accumulator, Output = F::Elem>,
    I: SparseIndex,
{
    type Output = F::Elem;
    fn mul_left_into(
        &mut self,
        weights: &[F::Elem],
        out: &mut [F::Elem],
    ) -> Result<(), LinearMapError> {
        dimensions(
            self.matrix.row_count(),
            self.matrix.column_count(),
            weights.len(),
            out.len(),
        )?;
        let reduce = self.field.prepare_reduce(self.max_terms);
        // Signed SHA columns have very uneven costs; keep work stealing fine
        // grained here while dense/native contractions retain coarse ranges.
        column_chunks_into::<1, _>(out, self.parallel, |j| {
            signed_segment(self.field, self.matrix.column(j).unwrap(), weights, &reduce)
        });
        Ok(())
    }
}
impl<F, I> BilinearEval<F> for PreparedSignedSparse<'_, F, I>
where
    F: RingOps
        + Sync
        + BatchMulAcc<F::Elem, u64>
        + Reduce<<F as BatchMulAcc<F::Elem, u64>>::Accumulator, Output = F::Elem>,
    I: SparseIndex,
{
    fn evaluate_bilinear(
        &mut self,
        weights: &[F::Elem],
        columns: &impl ColumnValues<F::Elem>,
    ) -> Result<F::Elem, LinearMapError> {
        dimensions(
            self.matrix.row_count(),
            self.matrix.column_count(),
            weights.len(),
            columns.len(),
        )?;
        let reduce = self.field.prepare_reduce(self.max_terms);
        let mut total = self.field.zero();
        for (j, column) in self.matrix.columns().enumerate() {
            let value = signed_segment(self.field, column, weights, &reduce);
            total = self
                .field
                .add(&total, &self.field.mul(&value, &columns.scalar(j)));
        }
        Ok(total)
    }
}
