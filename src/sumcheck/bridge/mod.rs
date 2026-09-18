//! Matrix-MLE contraction, shared by the outer/inner bridge and terminal checks.
//! These operations never touch a transcript or derive a claim from a witness.
use field::RingOps;
pub(crate) mod native;
pub(crate) mod repeated;
use crate::sumcheck::SumcheckError;

/// Prepared arithmetic and reusable workspace; every result owns its values.
pub(crate) trait PreparedBinding<F: RingOps> {
    type Bound;
    fn bind_rows(&mut self, rows: &[F::Elem]) -> Result<Self::Bound, SumcheckError>;
    fn bind_rows_into(
        &mut self,
        rows: &[F::Elem],
        out: &mut Self::Bound,
    ) -> Result<(), SumcheckError>;
    /// Direct bilinear evaluation. Must not construct the column coefficient table.
    fn evaluate_at(
        &mut self,
        rows: &[F::Elem],
        column_point: &[F::Elem],
    ) -> Result<F::Elem, SumcheckError>;
}
#[cfg(feature = "ecdsa")]
pub(crate) mod composite;
pub(crate) mod dense;
pub(crate) mod structured;
#[cfg(test)]
mod tests;
