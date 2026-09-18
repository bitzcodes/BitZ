//! Collapse a local relation once; callers keep its instance factor separate.
use crate::sumcheck::SumcheckError;
use field::RingOps;
pub(crate) fn collapse_signed_columns(
    relation: &circuit::linear_map::CscMatrix<Box<[i64]>>,
    weights: &[field::Fp<2>],
    field: &field::FpCtx<2>,
) -> Result<Vec<field::Fp<2>>, SumcheckError> {
    if weights.len() != relation.row_count().next_power_of_two() {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    let mut out = field.zero_vec(relation.column_count());
    use circuit::linear_map::LeftMul;
    let mut prepared = circuit::linear_map::contraction::PreparedSignedSparse::new(field, relation);
    if !cfg!(feature = "parallel") {
        prepared = prepared.serial();
    }
    prepared
        .mul_left_into(&weights[..relation.row_count()], &mut out)
        .map_err(|_| SumcheckError::InvalidProductDimensions)?;
    Ok(out)
}
