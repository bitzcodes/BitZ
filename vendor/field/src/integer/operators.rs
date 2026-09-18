//! Integer operators use the fixed-width wrapping ring, matching machine-word
//! arithmetic. Use `CheckedArithmetic` when overflow must be rejected, or
//! `WideMul` for an exact product. No operator grows the declared limb count.
use super::Z;
use crate::CtEq;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use num_traits::{One, Zero};

impl<const L: usize> From<u64> for Z<L> {
    fn from(value: u64) -> Self {
        // Construction is exact. Explicit bit-pattern construction remains
        // available when the caller intends a wrapping reinterpretation.
        if L == 1 {
            assert!(
                value <= i64::MAX as u64,
                "integer does not fit the signed width"
            );
        }
        Self::from_twos_complement_words(core::array::from_fn(|i| if i == 0 { value } else { 0 }))
    }
}
impl<const L: usize> From<i128> for Z<L> {
    fn from(value: i128) -> Self {
        if L == 1 {
            assert!(
                value >= i64::MIN as i128 && value <= i64::MAX as i128,
                "integer does not fit the signed width"
            );
        }
        Self::from_twos_complement_words(core::array::from_fn(|i| match i {
            0 => value as u64,
            1 => (value >> 64) as u64,
            _ => (value >> 127) as u64,
        }))
    }
}
impl<const L: usize> Zero for Z<L> {
    fn zero() -> Self {
        Self::ZERO
    }
    fn is_zero(&self) -> bool {
        self.ct_is_zero().declassify()
    }
}
impl<const L: usize> One for Z<L> {
    fn one() -> Self {
        Self::ONE
    }
}
impl<const L: usize> Add for Z<L> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(&rhs)
    }
}
impl<const L: usize> AddAssign for Z<L> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.wrapping_add(&rhs);
    }
}
impl<const L: usize> Sub for Z<L> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(&rhs)
    }
}
impl<const L: usize> SubAssign for Z<L> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.wrapping_sub(&rhs);
    }
}
impl<const L: usize> Mul for Z<L> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(&rhs)
    }
}
impl<const L: usize> Neg for Z<L> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}
impl<const L: usize> Sum for Z<L> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<const L: usize> From<u64> for super::Uint<L> {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}
impl From<u128> for super::Uint<2> {
    fn from(value: u128) -> Self {
        Self::from_words([value as u64, (value >> 64) as u64])
    }
}
impl From<super::Uint<1>> for u64 {
    fn from(value: super::Uint<1>) -> Self {
        value.as_words()[0]
    }
}
impl From<super::Uint<2>> for u128 {
    fn from(value: super::Uint<2>) -> Self {
        value.as_words()[0] as u128 | ((value.as_words()[1] as u128) << 64)
    }
}

// The Option adapters deliberately disclose validity. Private inputs should use
// CheckedArithmetic's masked results directly.
impl<const L: usize> core::fmt::Display for super::Uint<L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "0x")?;
        for w in self.as_words().iter().rev() {
            write!(f, "{w:016x}")?;
        }
        Ok(())
    }
}
impl<const L: usize> num_traits::ConstZero for super::Uint<L> {
    const ZERO: Self = Self::ZERO;
}
impl<const L: usize> num_traits::ConstOne for super::Uint<L> {
    const ONE: Self = Self::ONE;
}
impl<const L: usize> core::ops::Add for super::Uint<L> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(&rhs)
    }
}
impl<const L: usize> core::ops::Add<&Self> for super::Uint<L> {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        self.wrapping_add(rhs)
    }
}
impl<const L: usize> core::ops::AddAssign for super::Uint<L> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.wrapping_add(&rhs);
    }
}
impl<const L: usize> core::ops::AddAssign<&Self> for super::Uint<L> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_add(rhs);
    }
}
impl<const L: usize> num_traits::CheckedAdd for super::Uint<L> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        let v = crate::CheckedArithmetic::checked_add_ct(self, rhs);
        v.validity().declassify().then_some(*v.value())
    }
}
impl<const L: usize> core::ops::Sub for super::Uint<L> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(&rhs)
    }
}
impl<const L: usize> core::ops::Sub<&Self> for super::Uint<L> {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        self.wrapping_sub(rhs)
    }
}
impl<const L: usize> core::ops::SubAssign for super::Uint<L> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.wrapping_sub(&rhs);
    }
}
impl<const L: usize> core::ops::SubAssign<&Self> for super::Uint<L> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_sub(rhs);
    }
}
impl<const L: usize> num_traits::CheckedSub for super::Uint<L> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        let v = crate::CheckedArithmetic::checked_sub_ct(self, rhs);
        v.validity().declassify().then_some(*v.value())
    }
}
impl<const L: usize> core::ops::Mul for super::Uint<L> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(&rhs)
    }
}
impl<const L: usize> core::ops::Mul<&Self> for super::Uint<L> {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        self.wrapping_mul(rhs)
    }
}
impl<const L: usize> core::ops::MulAssign for super::Uint<L> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.wrapping_mul(&rhs);
    }
}
impl<const L: usize> core::ops::MulAssign<&Self> for super::Uint<L> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_mul(rhs);
    }
}
impl<const L: usize> num_traits::CheckedMul for super::Uint<L> {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        let v = crate::CheckedArithmetic::checked_mul_ct(self, rhs);
        v.validity().declassify().then_some(*v.value())
    }
}
impl<const L: usize> Zero for super::Uint<L> {
    fn zero() -> Self {
        Self::ZERO
    }
    fn is_zero(&self) -> bool {
        self.ct_is_zero().declassify()
    }
}
impl<const L: usize> One for super::Uint<L> {
    fn one() -> Self {
        Self::ONE
    }
}
impl<const L: usize> core::iter::Sum for super::Uint<L> {
    fn sum<I: Iterator<Item = Self>>(i: I) -> Self {
        i.fold(Self::ZERO, |a, b| a + b)
    }
}
impl<'a, const L: usize> core::iter::Sum<&'a Self> for super::Uint<L> {
    fn sum<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ZERO, |a, b| a + b)
    }
}
impl<const L: usize> core::iter::Product for super::Uint<L> {
    fn product<I: Iterator<Item = Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}
impl<'a, const L: usize> core::iter::Product<&'a Self> for super::Uint<L> {
    fn product<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}
impl<const L: usize> From<bool> for super::Uint<L> {
    fn from(v: bool) -> Self {
        Self::from(v as u64)
    }
}
impl<const L: usize> core::fmt::Display for super::Z<L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "0x")?;
        for w in self.as_words().iter().rev() {
            write!(f, "{w:016x}")?;
        }
        Ok(())
    }
}
impl<const L: usize> num_traits::ConstZero for super::Z<L> {
    const ZERO: Self = Self::ZERO;
}
impl<const L: usize> num_traits::ConstOne for super::Z<L> {
    const ONE: Self = Self::ONE;
}
impl<const L: usize> core::ops::Add<&Self> for super::Z<L> {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        self.wrapping_add(rhs)
    }
}
impl<const L: usize> core::ops::AddAssign<&Self> for super::Z<L> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_add(rhs);
    }
}
impl<const L: usize> num_traits::CheckedAdd for super::Z<L> {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        let v = crate::CheckedArithmetic::checked_add_ct(self, rhs);
        v.validity().declassify().then_some(*v.value())
    }
}
impl<const L: usize> core::ops::Sub<&Self> for super::Z<L> {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        self.wrapping_sub(rhs)
    }
}
impl<const L: usize> core::ops::SubAssign<&Self> for super::Z<L> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_sub(rhs);
    }
}
impl<const L: usize> num_traits::CheckedSub for super::Z<L> {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        let v = crate::CheckedArithmetic::checked_sub_ct(self, rhs);
        v.validity().declassify().then_some(*v.value())
    }
}
impl<const L: usize> core::ops::Mul<&Self> for super::Z<L> {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        self.wrapping_mul(rhs)
    }
}
impl<const L: usize> core::ops::MulAssign for super::Z<L> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.wrapping_mul(&rhs);
    }
}
impl<const L: usize> core::ops::MulAssign<&Self> for super::Z<L> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.wrapping_mul(rhs);
    }
}
impl<const L: usize> num_traits::CheckedMul for super::Z<L> {
    fn checked_mul(&self, rhs: &Self) -> Option<Self> {
        let v = crate::CheckedArithmetic::checked_mul_ct(self, rhs);
        v.validity().declassify().then_some(*v.value())
    }
}
impl<const L: usize> num_traits::CheckedNeg for super::Z<L> {
    fn checked_neg(&self) -> Option<Self> {
        let v = crate::CheckedArithmetic::checked_neg_ct(self);
        v.validity().declassify().then_some(*v.value())
    }
}
impl<'a, const L: usize> core::iter::Sum<&'a Self> for super::Z<L> {
    fn sum<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ZERO, |a, b| a + b)
    }
}
impl<const L: usize> core::iter::Product for super::Z<L> {
    fn product<I: Iterator<Item = Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}
impl<'a, const L: usize> core::iter::Product<&'a Self> for super::Z<L> {
    fn product<I: Iterator<Item = &'a Self>>(i: I) -> Self {
        i.fold(Self::ONE, |a, b| a * b)
    }
}
impl<const L: usize> From<bool> for super::Z<L> {
    fn from(v: bool) -> Self {
        Self::from(v as u64)
    }
}
impl<const L: usize> From<i8> for Z<L> {
    fn from(v: i8) -> Self {
        Self::from(v as i128)
    }
}
impl<const L: usize> From<i16> for Z<L> {
    fn from(v: i16) -> Self {
        Self::from(v as i128)
    }
}
impl<const L: usize> From<i32> for Z<L> {
    fn from(v: i32) -> Self {
        Self::from(v as i128)
    }
}
impl<const L: usize> From<i64> for Z<L> {
    fn from(v: i64) -> Self {
        Self::from(v as i128)
    }
}
