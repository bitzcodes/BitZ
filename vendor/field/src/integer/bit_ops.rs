//! Wrapping one-bit integer operators and explicit checked adapters.
use crate::Bit;
use core::ops::*;
impl core::fmt::Display for Bit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_u64())
    }
}
impl From<bool> for Bit {
    fn from(v: bool) -> Self {
        Self::from_bool(v)
    }
}
impl From<Bit> for u8 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for u16 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for u32 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for u64 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for u128 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for usize {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for i8 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for i16 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for i32 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for i64 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for i128 {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for isize {
    fn from(v: Bit) -> Self {
        v.as_u64() as Self
    }
}
impl From<Bit> for bool {
    fn from(v: Bit) -> Self {
        v.as_u64() != 0
    }
}
impl Add for Bit {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_lsb(self.as_u64() ^ rhs.as_u64())
    }
}
impl Add<&Self> for Bit {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        self + *rhs
    }
}
impl AddAssign for Bit {
    fn add_assign(&mut self, rhs: Self) {
        *self = (*self).add(rhs);
    }
}
impl AddAssign<&Self> for Bit {
    fn add_assign(&mut self, rhs: &Self) {
        *self = (*self).add(*rhs);
    }
}
impl num_traits::CheckedAdd for Bit {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        (self.as_u64() & rhs.as_u64() == 0).then_some((*self).add(*rhs))
    }
}
impl Sub for Bit {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_lsb(self.as_u64() ^ rhs.as_u64())
    }
}
impl Sub<&Self> for Bit {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        self - *rhs
    }
}
impl SubAssign for Bit {
    fn sub_assign(&mut self, rhs: Self) {
        *self = (*self).sub(rhs);
    }
}
impl SubAssign<&Self> for Bit {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = (*self).sub(*rhs);
    }
}
impl num_traits::CheckedSub for Bit {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        (self.as_u64() >= rhs.as_u64()).then_some((*self).sub(*rhs))
    }
}
impl Mul for Bit {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_lsb(self.as_u64() & rhs.as_u64())
    }
}
impl Mul<&Self> for Bit {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        self * *rhs
    }
}
impl MulAssign for Bit {
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self).mul(rhs);
    }
}
impl MulAssign<&Self> for Bit {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = (*self).mul(*rhs);
    }
}
impl num_traits::CheckedMul for Bit {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        (true).then_some((*self).mul(*rhs))
    }
}
impl core::iter::Sum for Bit {
    fn sum<I: Iterator<Item = Self>>(i: I) -> Self {
        i.fold(Self::ZERO, |a, b| a + b)
    }
}
impl<'a> core::iter::Sum<&'a Self> for Bit {
    fn sum<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ZERO, |a, b| a + b)
    }
}
impl core::iter::Product for Bit {
    fn product<I: Iterator<Item = Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}
impl<'a> core::iter::Product<&'a Self> for Bit {
    fn product<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}
impl num_traits::Zero for Bit {
    fn zero() -> Self {
        Self::ZERO
    }
    fn is_zero(&self) -> bool {
        self.as_u64() == 0
    }
}
impl num_traits::One for Bit {
    fn one() -> Self {
        Self::ONE
    }
}
impl num_traits::ConstZero for Bit {
    const ZERO: Self = Self::ZERO;
}
impl num_traits::ConstOne for Bit {
    const ONE: Self = Self::ONE;
}
