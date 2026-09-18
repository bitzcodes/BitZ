//! Public binary maps shared with matrix binding.
use crate::pcs::IntegerMatrixLayout;

/// Number of bit cells in an integer-evaluation shape.
pub fn cell_count(p: &IntegerMatrixLayout) -> usize {
    1usize << (p.row_vars + p.word_bits.trailing_zeros() as usize + p.col_vars)
}
/// Row-bit width of an integer-evaluation shape.
pub fn cell_row_bits(p: &IntegerMatrixLayout) -> usize {
    p.row_vars + p.word_bits.trailing_zeros() as usize
}
