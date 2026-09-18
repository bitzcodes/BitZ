//! Prepared sparse execution. Storage preparation and arithmetic selection are
//! static; CSR and CSC share validation, contraction and scheduling.
use super::contraction::columns_into;
use super::{
    BilinearEval, CoefficientStore, ColumnValues, CscMatrix, CsrMatrix, LeftMul, LinearMapError,
    RightMul, SparseIndex,
};
use field::{BatchMulAcc, MergeAccumulator, Reduce, RingOps};
use std::borrow::Cow;

/// Consume native coefficients directly, without preparing a field-valued copy.
pub struct DirectCoefficients;
/// Distinct coefficients embedded once, or borrowed from field-valued storage.
pub struct PreparedCoefficients<'a, E: Clone> {
    values: Cow<'a, [E]>,
}

pub trait CoefficientArithmetic<F: RingOps, S, Src> {
    type Accumulator: MergeAccumulator;
    fn mul_acc(
        &self,
        field: &F,
        storage: &S,
        acc: &mut Self::Accumulator,
        entry: usize,
        source: &Src,
    );
}
impl<F, C> CoefficientArithmetic<F, Box<[C]>, F::Elem> for DirectCoefficients
where
    F: RingOps + BatchMulAcc<F::Elem, C>,
{
    type Accumulator = <F as BatchMulAcc<F::Elem, C>>::Accumulator;
    #[inline]
    fn mul_acc(
        &self,
        field: &F,
        storage: &Box<[C]>,
        acc: &mut Self::Accumulator,
        entry: usize,
        source: &F::Elem,
    ) {
        field.mul_acc(acc, source, &storage[entry]);
    }
}
impl<F, S, Src> CoefficientArithmetic<F, S, Src> for PreparedCoefficients<'_, F::Elem>
where
    F: RingOps + BatchMulAcc<F::Elem, Src>,
    S: CoefficientStore,
{
    type Accumulator = <F as BatchMulAcc<F::Elem, Src>>::Accumulator;
    #[inline]
    fn mul_acc(
        &self,
        field: &F,
        storage: &S,
        acc: &mut Self::Accumulator,
        entry: usize,
        source: &Src,
    ) {
        field.mul_acc(acc, &self.values[storage.coefficient_index(entry)], source);
    }
}
fn dimensions(expected: usize, actual: usize, kind: &'static str) -> Result<(), LinearMapError> {
    if expected != actual {
        Err(LinearMapError::Length {
            kind,
            expected,
            actual,
        })
    } else {
        Ok(())
    }
}
fn max_terms<I: SparseIndex>(offsets: &[I]) -> usize {
    offsets
        .windows(2)
        .map(|w| w[1].to_usize() - w[0].to_usize())
        .max()
        .unwrap_or(0)
        .max(1)
}
#[inline]
fn contract<F, S, P, Src, I>(
    field: &F,
    storage: &S,
    arithmetic: &P,
    indices: &[I],
    range: std::ops::Range<usize>,
    read: impl Fn(usize) -> Src,
    reduce: &impl Fn(P::Accumulator) -> F::Elem,
) -> F::Elem
where
    F: RingOps,
    P: CoefficientArithmetic<F, S, Src>,
    I: SparseIndex,
{
    let mut acc = P::Accumulator::zero();
    for entry in range {
        arithmetic.mul_acc(
            field,
            storage,
            &mut acc,
            entry,
            &read(indices[entry].to_usize()),
        );
    }
    reduce(acc)
}
macro_rules! prepared {
    ($name:ident,$matrix:ident,$offsets:ident,$indices:ident,$source_count:ident,$out_count:ident,$op:ident,$method:ident,$left:expr) => {
        /// Prepared execution retains the field, canonical matrix and coefficient
        /// policy. Reused multiplication writes only caller-owned output storage.
        pub struct $name<'a, F: RingOps, S, P, I = usize> {
            field: &'a F,
            matrix: &'a $matrix<S, I>,
            arithmetic: P,
            max_terms: usize,
            parallel: bool,
        }
        impl<'a, F: RingOps, S: CoefficientStore, P, I: SparseIndex> $name<'a, F, S, P, I> {
            fn with_arithmetic(field: &'a F, matrix: &'a $matrix<S, I>, arithmetic: P) -> Self {
                Self {
                    field,
                    matrix,
                    arithmetic,
                    max_terms: max_terms(matrix.$offsets()),
                    parallel: true,
                }
            }
            pub fn serial(mut self) -> Self {
                self.parallel = false;
                self
            }
            pub fn field(&self) -> &'a F {
                self.field
            }
            pub fn row_count(&self) -> usize {
                self.matrix.row_count()
            }
            pub fn column_count(&self) -> usize {
                self.matrix.column_count()
            }
        }
        impl<'a, F: RingOps, C, I: SparseIndex> $name<'a, F, Box<[C]>, DirectCoefficients, I> {
            pub fn mixed(field: &'a F, matrix: &'a $matrix<Box<[C]>, I>) -> Self {
                Self::with_arithmetic(field, matrix, DirectCoefficients)
            }
        }
        impl<'a, F: RingOps, S: CoefficientStore, I: SparseIndex>
            $name<'a, F, S, PreparedCoefficients<'a, F::Elem>, I>
        {
            /// Embed distinct slots once against this field provider. Runtime
            /// field compatibility remains the caller's contract.
            pub fn projected(
                field: &'a F,
                matrix: &'a $matrix<S, I>,
                project: impl Fn(S::Ref<'_>) -> F::Elem,
            ) -> Self {
                let values = (0..matrix.coefficients().distinct_len())
                    .map(|i| project(matrix.coefficients().distinct_coefficient(i)))
                    .collect();
                Self::with_arithmetic(
                    field,
                    matrix,
                    PreparedCoefficients {
                        values: Cow::Owned(values),
                    },
                )
            }
            pub fn prepared_coefficient_count(&self) -> usize {
                self.arithmetic.values.len()
            }
        }
        impl<'a, F: RingOps, I: SparseIndex>
            $name<'a, F, Box<[F::Elem]>, PreparedCoefficients<'a, F::Elem>, I>
        {
            pub fn borrowed(field: &'a F, matrix: &'a $matrix<Box<[F::Elem]>, I>) -> Self {
                Self::with_arithmetic(
                    field,
                    matrix,
                    PreparedCoefficients {
                        values: Cow::Borrowed(matrix.coefficients()),
                    },
                )
            }
        }
        impl<F, S, P, I, Src> $op<Src> for $name<'_, F, S, P, I>
        where
            F: RingOps + Sync + Reduce<P::Accumulator, Output = F::Elem>,
            S: CoefficientStore + Sync,
            P: CoefficientArithmetic<F, S, Src> + Sync,
            I: SparseIndex,
            Src: Copy + Sync,
        {
            type Output = F::Elem;
            fn $method(
                &mut self,
                source: &[Src],
                out: &mut [F::Elem],
            ) -> Result<(), LinearMapError> {
                dimensions(self.matrix.$source_count(), source.len(), "source")?;
                dimensions(self.matrix.$out_count(), out.len(), "output")?;
                let reduce = self.field.prepare_reduce(self.max_terms);
                columns_into(out, self.parallel, |segment| {
                    let offsets = self.matrix.$offsets();
                    contract(
                        self.field,
                        self.matrix.coefficients(),
                        &self.arithmetic,
                        self.matrix.$indices(),
                        offsets[segment].to_usize()..offsets[segment + 1].to_usize(),
                        |i| source[i],
                        &reduce,
                    )
                });
                Ok(())
            }
        }
        impl<F, S, P, I> BilinearEval<F> for $name<'_, F, S, P, I>
        where
            F: RingOps + Sync + Reduce<P::Accumulator, Output = F::Elem>,
            S: CoefficientStore + Sync,
            P: CoefficientArithmetic<F, S, F::Elem> + Sync,
            I: SparseIndex,
        {
            fn evaluate_bilinear(
                &mut self,
                weights: &[F::Elem],
                columns: &impl ColumnValues<F::Elem>,
            ) -> Result<F::Elem, LinearMapError> {
                dimensions(self.row_count(), weights.len(), "row weights")?;
                dimensions(self.column_count(), columns.len(), "column values")?;
                let reduce = self.field.prepare_reduce(self.max_terms);
                let mut total = self.field.zero();
                for (segment, bounds) in self.matrix.$offsets().windows(2).enumerate() {
                    let value = contract(
                        self.field,
                        self.matrix.coefficients(),
                        &self.arithmetic,
                        self.matrix.$indices(),
                        bounds[0].to_usize()..bounds[1].to_usize(),
                        |i| if $left { weights[i] } else { columns.scalar(i) },
                        &reduce,
                    );
                    let other = if $left {
                        columns.scalar(segment)
                    } else {
                        weights[segment]
                    };
                    total = self.field.add(&total, &self.field.mul(&value, &other));
                }
                Ok(total)
            }
        }
    };
}
prepared!(
    PreparedColumns,
    CscMatrix,
    column_offsets,
    row_indices,
    row_count,
    column_count,
    LeftMul,
    mul_left_into,
    true
);
prepared!(
    PreparedRows,
    CsrMatrix,
    row_offsets,
    column_indices,
    column_count,
    row_count,
    RightMul,
    mul_right_into,
    false
);
