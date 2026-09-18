use crate::utils::from_ref::FromRef;
use field::{Bit, CtSelect};
use num_traits::CheckedAdd;
use thiserror::Error;

/// A trait for inner product algorithms implementations.
pub trait InnerProduct<Lhs: ?Sized, Rhs, Output> {
    /// The main entry point for the inner product.
    /// `CHECK` determines whether the implementation should check for overflow.
    fn inner_product<const CHECK: bool>(
        lhs: &Lhs,
        rhs: &[Rhs],
        zero: Output,
    ) -> Result<Output, InnerProductError>;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum InnerProductError {
    #[error("The length of LHS and RHS does not match: LHS={lhs}, RHS={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("Arithmetic overflow")]
    Overflow,
}

/// The inner product for slices containing `Bit` elements.
/// Uses `add` or `checked_add` to sum the elements of the RHS that
/// correspond to `true` elements of the boolean slice.
pub struct BooleanInnerProductAdd;

impl<Rhs: Clone, Out: FromRef<Rhs> + CheckedAdd + CtSelect + Clone> InnerProduct<[Bit], Rhs, Out>
    for BooleanInnerProductAdd
{
    /// Bit inner product.
    #[allow(clippy::arithmetic_side_effects)] // Used in unchecked mode
    fn inner_product<const CHECK: bool>(
        lhs: &[Bit],
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        if lhs.len() != rhs.as_ref().len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.as_ref().len(),
            });
        }

        lhs.iter()
            .zip(rhs)
            .try_fold(zero.clone(), |acc, (bit, value)| {
                let selected = Out::ct_select(&zero, &Out::from_ref(value), bit.mask());
                if CHECK {
                    acc.checked_add(&selected)
                        .ok_or(InnerProductError::Overflow)
                } else {
                    Ok(acc + selected)
                }
            })
    }
}
