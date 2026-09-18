//! Protocol boundary adapters for the shared typed Barrett accumulators.
//! Arithmetic and accumulator storage belong to `field`.

use field::{Fp, FpCtx, FpLinearAcc, FpProductAcc, Uint};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum DelayedReductionError {
    #[error("delayed reduction requires a modulus between 65 and 128 bits")]
    UnsupportedModulus,
}

pub(crate) fn prepare_field(config: &field::FpCtx<2>) -> Result<FpCtx<2>, DelayedReductionError> {
    let words = *config.modulus().as_words();
    if words[1] == 0 {
        return Err(DelayedReductionError::UnsupportedModulus);
    }
    Ok(config.clone())
}
#[inline(always)]
pub(crate) fn element(field: &FpCtx<2>, raw: u128) -> Fp<2> {
    field.from_montgomery_integer(Uint::from_words([raw as u64, (raw >> 64) as u64]))
}
#[inline(always)]
pub(crate) fn encoded(value: Fp<2>) -> u128 {
    let w = value.as_montgomery_integer().as_words();
    w[0] as u128 | ((w[1] as u128) << 64)
}
/// Encoded-table adapter; each accumulator's operand pair fixes its scale.
/// Modulus ownership and batch capacity are established by the calling kernel.
pub(crate) trait EncodedMac<Rhs> {
    fn accumulate_encoded(&mut self, field: &FpCtx<2>, lhs: u128, rhs: Rhs);
    fn reduce_encoded(self, field: &FpCtx<2>) -> u128;
}
impl EncodedMac<u128> for FpProductAcc<2> {
    #[inline(always)]
    fn accumulate_encoded(&mut self, field: &FpCtx<2>, lhs: u128, rhs: u128) {
        self.accumulate(&element(field, lhs), &element(field, rhs));
    }
    #[inline]
    fn reduce_encoded(self, field: &FpCtx<2>) -> u128 {
        encoded(field::Reduce::reduce(field, self))
    }
}
impl EncodedMac<u64> for FpLinearAcc<2, 1> {
    #[inline(always)]
    fn accumulate_encoded(&mut self, field: &FpCtx<2>, lhs: u128, rhs: u64) {
        self.accumulate(&element(field, lhs), &Uint::from_words([rhs]));
    }
    #[inline]
    fn reduce_encoded(self, field: &FpCtx<2>) -> u128 {
        encoded(field::Reduce::reduce(field, self))
    }
}
