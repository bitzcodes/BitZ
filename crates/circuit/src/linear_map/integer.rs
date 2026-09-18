//! Exact integer matrix products. Public bounds are proved before private evaluation.
use super::{CscMatrix, CsrMatrix, LeftMul, LinearMapError, RightMul, SparseIndex};
use crate::integer_storage::IntegerTable;
use field::{CheckedArithmetic, CtMask, CtOrd, CtSelect, IntegerOps, Uint, WideMul, Z};
use std::marker::PhantomData;

// A source type supplies its complete representable range. No witness-dependent
// bounds, narrowing decisions, or overflow checks occur during multiplication.
pub trait ExactInput<const L: usize>: Copy {
    fn magnitudes() -> Option<(Uint<L>, Uint<L>)>;
    fn product(self, coefficient: &Z<L>) -> Z<L>;
}
impl<const L: usize> ExactInput<L> for bool {
    fn magnitudes() -> Option<(Uint<L>, Uint<L>)> {
        Some((Uint::ONE, Uint::ZERO))
    }
    fn product(self, c: &Z<L>) -> Z<L> {
        Z::ct_select(&Z::ZERO, c, CtMask::from_lsb(self as u64))
    }
}
impl<const N: usize, const L: usize> ExactInput<L> for Z<N> {
    fn magnitudes() -> Option<(Uint<L>, Uint<L>)> {
        let negative = Z::<N>::MIN.unsigned_abs().checked_resize_ct::<L>();
        let positive = Z::<N>::MAX.unsigned_abs().checked_resize_ct::<L>();
        (negative.validity() & positive.validity())
            .declassify()
            .then(|| (*positive.value(), *negative.value()))
    }
    fn product(self, c: &Z<L>) -> Z<L> {
        *IntegerOps
            .mul_wide(c, &self)
            .checked_resize_ct::<L>()
            .value()
    }
}
impl<const N: usize, const L: usize> ExactInput<L> for Uint<N> {
    fn magnitudes() -> Option<(Uint<L>, Uint<L>)> {
        let positive = Uint::<N>::MAX.checked_resize_ct::<L>();
        positive
            .validity()
            .declassify()
            .then(|| (*positive.value(), Uint::ZERO))
    }
    fn product(self, c: &Z<L>) -> Z<L> {
        *IntegerOps
            .mul_wide(c, &self)
            .checked_resize_ct::<L>()
            .value()
    }
}
macro_rules! native {
    ($t:ty,$n:expr,$cast:ty) => {
        impl<const L: usize> ExactInput<L> for $t {
            fn magnitudes() -> Option<(Uint<L>, Uint<L>)> {
                let maximum = Uint::<$n>::from(<$t>::MAX as $cast).checked_resize_ct::<L>();
                maximum
                    .validity()
                    .declassify()
                    .then(|| (*maximum.value(), Uint::ZERO))
            }
            fn product(self, c: &Z<L>) -> Z<L> {
                *IntegerOps
                    .mul_wide(c, &Uint::<$n>::from(self as $cast))
                    .checked_resize_ct::<L>()
                    .value()
            }
        }
    };
}
native!(u32, 1, u64);
native!(u64, 1, u64);
native!(u128, 2, u128);

fn prepare<Src: ExactInput<L>, const L: usize, I: SparseIndex>(
    coefficients: &IntegerTable,
    offsets: &[I],
) -> Result<Vec<Z<L>>, LinearMapError> {
    if L == 0 {
        return Err(LinearMapError::IntegerWidth {
            output_limbs: L,
            segment: 0,
        });
    }
    let (input_positive, input_negative) =
        Src::magnitudes().ok_or(LinearMapError::IntegerWidth {
            output_limbs: L,
            segment: 0,
        })?;
    let view = coefficients.view();
    let mut prepared = Vec::with_capacity(view.len());
    for (segment, bounds) in offsets.windows(2).enumerate() {
        let error = || LinearMapError::IntegerWidth {
            output_limbs: L,
            segment,
        };
        let mut positive = Uint::<L>::ZERO;
        let mut negative = Uint::<L>::ZERO;
        for entry in bounds[0].to_usize()..bounds[1].to_usize() {
            let coefficient =
                field::ZRef::from_twos_complement_words(&view[entry]).checked_resize_ct::<L>();
            if !coefficient.validity().declassify() {
                return Err(error());
            }
            let coefficient = *coefficient.value();
            let magnitude = coefficient.unsigned_abs();
            let (p, n) = if coefficient.is_negative_ct().declassify() {
                (input_negative, input_positive)
            } else {
                (input_positive, input_negative)
            };
            let p = magnitude.checked_mul_ct(&p);
            let n = magnitude.checked_mul_ct(&n);
            if !(p.validity() & n.validity()).declassify() {
                return Err(error());
            }
            let p = positive.checked_add_ct(p.value());
            let n = negative.checked_add_ct(n.value());
            if !(p.validity() & n.validity()).declassify() {
                return Err(error());
            }
            positive = *p.value();
            negative = *n.value();
            prepared.push(coefficient);
        }
        if !positive.ct_le(&Z::<L>::MAX.unsigned_abs()).declassify()
            || !negative.ct_le(&Z::<L>::MIN.unsigned_abs()).declassify()
        {
            return Err(error());
        }
    }
    Ok(prepared)
}

macro_rules! prepared {
    ($name:ident,$matrix:ident,$offsets:ident,$indices:ident,$source_count:ident,$out_count:ident,$op:ident,$method:ident) => {
        /// Exact products with a caller-selected output limb width. Preparation
        /// proves every intermediate sum for the source type's entire range.
        pub struct $name<'a, Src, const L: usize, I = usize> {
            matrix: &'a $matrix<IntegerTable, I>,
            coefficients: Vec<Z<L>>,
            source: PhantomData<Src>,
        }
        impl<'a, Src: ExactInput<L>, const L: usize, I: SparseIndex> $name<'a, Src, L, I> {
            pub fn new(matrix: &'a $matrix<IntegerTable, I>) -> Result<Self, LinearMapError> {
                let coefficients = prepare::<Src, L, I>(matrix.coefficients(), matrix.$offsets())?;
                Ok(Self {
                    matrix,
                    coefficients,
                    source: PhantomData,
                })
            }
        }
        impl<Src: ExactInput<L>, const L: usize, I: SparseIndex> $op<Src> for $name<'_, Src, L, I> {
            type Output = Z<L>;
            fn $method(&mut self, values: &[Src], out: &mut [Z<L>]) -> Result<(), LinearMapError> {
                for (kind, expected, actual) in [
                    ("source", self.matrix.$source_count(), values.len()),
                    ("output", self.matrix.$out_count(), out.len()),
                ] {
                    if expected != actual {
                        return Err(LinearMapError::Length {
                            kind,
                            expected,
                            actual,
                        });
                    }
                }
                for (dst, bounds) in out.iter_mut().zip(self.matrix.$offsets().windows(2)) {
                    let mut sum = Z::ZERO;
                    for entry in bounds[0].to_usize()..bounds[1].to_usize() {
                        let product = values[self.matrix.$indices()[entry].to_usize()]
                            .product(&self.coefficients[entry]);
                        // Positive/negative subset bounds cover all partial sums.
                        sum = sum.wrapping_add(&product);
                    }
                    *dst = sum;
                }
                Ok(())
            }
        }
    };
}
prepared!(
    PreparedIntegerRows,
    CsrMatrix,
    row_offsets,
    column_indices,
    column_count,
    row_count,
    RightMul,
    mul_right_into
);
prepared!(
    PreparedIntegerColumns,
    CscMatrix,
    column_offsets,
    row_indices,
    row_count,
    column_count,
    LeftMul,
    mul_left_into
);
