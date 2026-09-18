//! Operators only for static field identities; runtime fields need a context.
use super::*;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

macro_rules! binary_operator {
    ($trait:ident,$method:ident,$assign:ident,$assign_method:ident) => {
        impl<P: PrimeSpec<L>, const L: usize> $trait for StaticFp<P, L> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                StaticFpOps::<P, L>::new().$method(&self, &rhs)
            }
        }
        impl<P: PrimeSpec<L>, const L: usize> $assign for StaticFp<P, L> {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                *self = StaticFpOps::<P, L>::new().$method(self, &rhs);
            }
        }
    };
}
binary_operator!(Add, add, AddAssign, add_assign);
binary_operator!(Sub, sub, SubAssign, sub_assign);
binary_operator!(Mul, mul, MulAssign, mul_assign);
impl<P: PrimeSpec<L>, const L: usize> Neg for StaticFp<P, L> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        StaticFpOps::<P, L>::new().neg(&self)
    }
}

impl<P: PrimeSpec<L>, const L: usize> From<u128> for StaticFp<P, L> {
    #[inline]
    fn from(value: u128) -> Self {
        StaticFpOps::<P, L>::new().from_integer(&value)
    }
}

impl<P: PrimeSpec<2>> StaticFp<P, 2> {
    /// Embeds an ordinary unsigned integer into this static field.
    #[inline]
    pub fn from_u128(value: u128) -> Self {
        Self::from(value)
    }

    /// The canonical integer encoding, independent of Montgomery storage.
    #[inline]
    pub fn canonical_u128(&self) -> u128 {
        let value = StaticFpOps::<P, 2>::new().to_integer(self);
        value.as_words()[0] as u128 | ((value.as_words()[1] as u128) << 64)
    }
}
