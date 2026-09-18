//! Operator conveniences for fixed binary fields; providers own arithmetic.
use crate::{B127, B127Ops, FieldOps, Gf128, RingOps};
use core::ops::*;
impl num_traits::CheckedAdd for Gf128 {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(*self + *rhs)
    }
}
impl num_traits::CheckedSub for Gf128 {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(*self - *rhs)
    }
}
impl num_traits::CheckedMul for Gf128 {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(*self * *rhs)
    }
}
impl num_traits::CheckedNeg for Gf128 {
    fn checked_neg(&self) -> Option<Self> {
        Some(*self)
    }
}
impl Gf128 {
    /// Inverse for callers that require a nonzero value. Use the provider's
    /// `inverse_ct` to keep the validity mask private.
    pub fn invert_nonzero(self) -> Self {
        let result = crate::Gf128Ops.inverse_ct(&self);
        assert!(result.validity().declassify(), "zero has no inverse");
        *result.value()
    }
}
impl num_traits::CheckedAdd for B127 {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(*self + *rhs)
    }
}
impl num_traits::CheckedSub for B127 {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(*self - *rhs)
    }
}
impl num_traits::CheckedMul for B127 {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        Some(*self * *rhs)
    }
}
impl num_traits::CheckedNeg for B127 {
    fn checked_neg(&self) -> Option<Self> {
        Some(*self)
    }
}
impl B127 {
    /// Inverse for callers that require a nonzero value. Use the provider's
    /// `inverse_ct` to keep the validity mask private.
    pub fn invert_nonzero(self) -> Self {
        let result = B127Ops.inverse_ct(&self);
        assert!(result.validity().declassify(), "zero has no inverse");
        *result.value()
    }
}
impl Add for B127 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        B127Ops.add(&self, &rhs)
    }
}
impl Add<&Self> for B127 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self {
        B127Ops.add(&self, rhs)
    }
}
impl AddAssign for B127 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = B127Ops.add(self, &rhs);
    }
}
impl AddAssign<&Self> for B127 {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        *self = B127Ops.add(self, rhs);
    }
}
impl Sub for B127 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        B127Ops.sub(&self, &rhs)
    }
}
impl Sub<&Self> for B127 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        B127Ops.sub(&self, rhs)
    }
}
impl SubAssign for B127 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = B127Ops.sub(self, &rhs);
    }
}
impl SubAssign<&Self> for B127 {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        *self = B127Ops.sub(self, rhs);
    }
}
impl Mul for B127 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        B127Ops.mul(&self, &rhs)
    }
}
impl Mul<&Self> for B127 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        B127Ops.mul(&self, rhs)
    }
}
impl MulAssign for B127 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = B127Ops.mul(self, &rhs);
    }
}
impl MulAssign<&Self> for B127 {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = B127Ops.mul(self, rhs);
    }
}
impl Neg for B127 {
    type Output = Self;
    fn neg(self) -> Self {
        self
    }
}
impl Div for B127 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.invert_nonzero()
    }
}
impl Div<&Self> for B127 {
    type Output = Self;
    fn div(self, rhs: &Self) -> Self {
        self / *rhs
    }
}
impl DivAssign for B127 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl DivAssign<&Self> for B127 {
    fn div_assign(&mut self, rhs: &Self) {
        *self = *self / *rhs;
    }
}
impl num_traits::Zero for B127 {
    fn zero() -> Self {
        Self::ZERO
    }
    fn is_zero(&self) -> bool {
        self.as_words() == &[0, 0]
    }
}
impl num_traits::One for B127 {
    fn one() -> Self {
        Self::ONE
    }
}
impl num_traits::ConstZero for B127 {
    const ZERO: Self = Self::ZERO;
}
impl num_traits::ConstOne for B127 {
    const ONE: Self = Self::ONE;
}
impl From<bool> for B127 {
    fn from(v: bool) -> Self {
        Self::from_polynomial_words([v as u64, 0])
    }
}
impl core::fmt::Display for B127 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "0x{:016x}{:016x}",
            self.as_words()[1],
            self.as_words()[0]
        )
    }
}
impl num_traits::Inv for B127 {
    type Output = Option<Self>;
    fn inv(self) -> Self::Output {
        let r = B127Ops.inverse_ct(&self);
        r.validity().declassify().then_some(*r.value())
    }
}
impl num_traits::Pow<u32> for B127 {
    type Output = Self;
    fn pow(self, e: u32) -> Self {
        {
            let mut e = e;
            let mut base = self;
            let mut out = Self::ONE;
            while e != 0 {
                if e & 1 != 0 {
                    out *= base;
                }
                e >>= 1;
                if e != 0 {
                    base = base.square();
                }
            }
            out
        }
    }
}
impl core::iter::Sum for B127 {
    fn sum<I: Iterator<Item = Self>>(i: I) -> Self {
        i.fold(Self::ZERO, |a, b| a + b)
    }
}
impl<'a> core::iter::Sum<&'a Self> for B127 {
    fn sum<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ZERO, |a, b| a + b)
    }
}
impl core::iter::Product for B127 {
    fn product<I: Iterator<Item = Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}
impl<'a> core::iter::Product<&'a Self> for B127 {
    fn product<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}

impl Gf128 {
    pub const fn zero() -> Self {
        Self::ZERO
    }
    pub const fn one() -> Self {
        Self::ONE
    }
}
impl B127 {
    pub const fn zero() -> Self {
        Self::ZERO
    }
    pub const fn one() -> Self {
        Self::ONE
    }
}
