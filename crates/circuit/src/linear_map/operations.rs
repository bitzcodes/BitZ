//! Mathematical matrix operations over commutative scalars; storage and
//! reduction stay behind the operator. Implementations validate dimensions
//! before modifying output and fully overwrite valid output slices.
use super::{ColumnValues, LinearMapError};
use field::RingOps;

/// Computes `out[j] = Σ_i weights[i] M[i,j]`.
pub trait LeftMul<Src> {
    type Output;
    fn mul_left_into(
        &mut self,
        weights: &[Src],
        out: &mut [Self::Output],
    ) -> Result<(), LinearMapError>;
}
/// Computes `out[i] = Σ_j M[i,j] values[j]`.
pub trait RightMul<Src> {
    type Output;
    fn mul_right_into(
        &mut self,
        values: &[Src],
        out: &mut [Self::Output],
    ) -> Result<(), LinearMapError>;
}
/// Computes `Σ_i,j weights[i] M[i,j] columns[j]` without materializing a bound table.
pub trait BilinearEval<F: RingOps> {
    fn evaluate_bilinear(
        &mut self,
        weights: &[F::Elem],
        columns: &impl ColumnValues<F::Elem>,
    ) -> Result<F::Elem, LinearMapError>;
}
