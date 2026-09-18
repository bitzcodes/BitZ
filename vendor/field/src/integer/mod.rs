//! Fixed-width integers. Storage width is part of the type; arithmetic never trims it.

use crate::ct::{Bit, CtEq, CtMask, CtOrd, CtSelect, CtValue};
use crate::traits::{CheckedArithmetic, WrappingArithmetic};

mod operators;
pub(crate) mod product;
pub use product::{IntegerOps, UintAccumulator, UintProduct, ZAccumulator, ZProduct};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Uint<const L: usize>(pub(crate) [u64; L]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Z<const L: usize>(pub(crate) Uint<L>);

macro_rules! inherent_wrapping {
    ($name:ident) => {
        impl<const L: usize> $name<L> {
            #[inline]
            pub fn wrapping_add(&self, rhs: &Self) -> Self {
                <Self as WrappingArithmetic>::wrapping_add(self, rhs)
            }
            #[inline]
            pub fn wrapping_sub(&self, rhs: &Self) -> Self {
                <Self as WrappingArithmetic>::wrapping_sub(self, rhs)
            }
            #[inline]
            pub fn wrapping_mul(&self, rhs: &Self) -> Self {
                <Self as WrappingArithmetic>::wrapping_mul(self, rhs)
            }
            #[inline]
            pub fn wrapping_neg(&self) -> Self {
                <Self as WrappingArithmetic>::wrapping_neg(self)
            }
        }
    };
}
inherent_wrapping!(Uint);
inherent_wrapping!(Z);

impl<const L: usize> Uint<L> {
    pub const ZERO: Self = {
        assert!(L > 0);
        Self([0; L])
    };
    pub const ONE: Self = Self::from_u64(1);
    pub const MAX: Self = {
        assert!(L > 0);
        Self([u64::MAX; L])
    };
    pub const fn from_words(words: [u64; L]) -> Self {
        const {
            assert!(L > 0);
        }
        Self(words)
    }
    pub const fn as_words(&self) -> &[u64; L] {
        &self.0
    }
    pub const fn from_u64(value: u64) -> Self {
        let mut result = Self::ZERO;
        result.0[0] = value;
        result
    }
    pub fn zero_extend<const M: usize>(&self) -> Uint<M> {
        const {
            assert!(M >= L);
        }
        let mut out = Uint::ZERO;
        out.0[..L].copy_from_slice(&self.0);
        out
    }
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Uint<M>> {
        let mut out = Uint::ZERO;
        let mut excess = 0;
        for i in 0..L {
            if i < M {
                out.0[i] = self.0[i];
            } else {
                excess |= self.0[i];
            }
        }
        CtValue::new(out, excess.ct_is_zero())
    }
    pub fn checked_to_signed_ct(&self) -> CtValue<Z<L>> {
        CtValue::new(Z(*self), !CtMask::from_lsb(self.0[L - 1] >> 63))
    }
    pub fn bit(&self, public_index: usize) -> Bit {
        assert!(public_index < L * 64, "bit index out of range");
        Bit::from_lsb(self.0[public_index / 64] >> (public_index % 64))
    }
    pub(crate) fn adc(&self, rhs: &Self) -> (Self, u64) {
        let mut out = Self::ZERO;
        #[cfg(target_arch = "x86_64")]
        {
            let mut carry = 0u8;
            for i in 0..L {
                carry =
                    core::arch::x86_64::_addcarry_u64(carry, self.0[i], rhs.0[i], &mut out.0[i]);
            }
            (out, u64::from(carry))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut carry = 0;
            for i in 0..L {
                let sum = self.0[i] as u128 + rhs.0[i] as u128 + carry as u128;
                out.0[i] = sum as u64;
                carry = (sum >> 64) as u64;
            }
            (out, carry)
        }
    }
    pub(crate) fn sbb(&self, rhs: &Self) -> (Self, u64) {
        let mut out = Self::ZERO;
        // Express a single borrow chain on x86-64. The baseline SBB intrinsic
        // avoids expanding every limb into two subtractions and boolean merges.
        #[cfg(target_arch = "x86_64")]
        {
            let mut borrow = 0u8;
            for i in 0..L {
                borrow =
                    core::arch::x86_64::_subborrow_u64(borrow, self.0[i], rhs.0[i], &mut out.0[i]);
            }
            (out, u64::from(borrow))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut borrow = 0;
            for i in 0..L {
                let (word, b0) = self.0[i].overflowing_sub(rhs.0[i]);
                let (word, b1) = word.overflowing_sub(borrow);
                out.0[i] = word;
                borrow = (b0 | b1) as u64;
            }
            (out, borrow)
        }
    }
    /// Shift with zero fill. A public shift at least the width returns zero.
    pub fn shr(&self, public_shift: usize) -> Self {
        let mut out = Self::ZERO;
        if public_shift >= L * 64 {
            return out;
        }
        let (words, bits) = (public_shift / 64, public_shift % 64);
        for i in 0..L - words {
            out.0[i] = self.0[i + words] >> bits;
            if bits != 0 && i + words + 1 < L {
                out.0[i] |= self.0[i + words + 1] << (64 - bits);
            }
        }
        out
    }
    pub fn truncating_shl(&self, public_shift: usize) -> Self {
        let mut out = Self::ZERO;
        if public_shift >= L * 64 {
            return out;
        }
        let (words, bits) = (public_shift / 64, public_shift % 64);
        for i in words..L {
            out.0[i] = self.0[i - words] << bits;
            if bits != 0 && i > words {
                out.0[i] |= self.0[i - words - 1] >> (64 - bits);
            }
        }
        out
    }
    pub fn checked_shl_ct(&self, public_shift: usize) -> CtValue<Self> {
        let out = self.truncating_shl(public_shift);
        CtValue::new(out, out.shr(public_shift).ct_eq(self))
    }
    /// Restoring division over every dividend bit, with masked subtraction.
    pub(crate) fn div_rem_words<const D: usize>(&self, divisor: &Uint<D>) -> (Self, Uint<D>) {
        let mut quotient = Self::ZERO;
        let mut remainder = Uint::<D>::ZERO;
        for i in (0..L * 64).rev() {
            let overflow = remainder.0[D - 1] >> 63;
            remainder = remainder.truncating_shl(1);
            remainder.0[0] |= self.bit(i).as_u64();
            let (difference, borrow) = remainder.sbb(divisor);
            let take = CtMask::from_lsb(overflow | (borrow ^ 1));
            remainder = Uint::ct_select(&remainder, &difference, take);
            quotient.0[i / 64] |= (take.word() & 1) << (i % 64);
        }
        (quotient, remainder)
    }
}

impl<const L: usize> Default for Uint<L> {
    fn default() -> Self {
        Self::ZERO
    }
}
impl<const L: usize> CtEq for Uint<L> {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        let mut diff = 0;
        for i in 0..L {
            diff |= self.0[i] ^ rhs.0[i];
        }
        diff.ct_is_zero()
    }
    fn ct_is_zero(&self) -> CtMask {
        self.ct_eq(&Self::ZERO)
    }
}
impl<const L: usize> CtOrd for Uint<L> {
    fn ct_lt(&self, rhs: &Self) -> CtMask {
        CtMask::from_lsb(self.sbb(rhs).1)
    }
}
impl<const L: usize> CtSelect for Uint<L> {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self(core::array::from_fn(|i| {
            u64::ct_select(&a.0[i], &b.0[i], mask)
        }))
    }
}
impl<const L: usize> WrappingArithmetic for Uint<L> {
    fn wrapping_add(&self, rhs: &Self) -> Self {
        self.adc(rhs).0
    }
    fn wrapping_sub(&self, rhs: &Self) -> Self {
        self.sbb(rhs).0
    }
    fn wrapping_neg(&self) -> Self {
        Self::ZERO.wrapping_sub(self)
    }
    fn wrapping_mul(&self, rhs: &Self) -> Self {
        let mut out = Self::ZERO;
        for i in 0..L {
            let mut carry = 0;
            for j in 0..L - i {
                let sum = self.0[i] as u128 * rhs.0[j] as u128 + out.0[i + j] as u128 + carry;
                out.0[i + j] = sum as u64;
                carry = sum >> 64;
            }
        }
        out
    }
}
impl<const L: usize> CheckedArithmetic for Uint<L> {
    fn checked_add_ct(&self, rhs: &Self) -> CtValue<Self> {
        let (out, carry) = self.adc(rhs);
        CtValue::new(out, carry.ct_is_zero())
    }
    fn checked_sub_ct(&self, rhs: &Self) -> CtValue<Self> {
        let (out, borrow) = self.sbb(rhs);
        CtValue::new(out, borrow.ct_is_zero())
    }
    fn checked_mul_ct(&self, rhs: &Self) -> CtValue<Self> {
        use crate::traits::WideMul;
        IntegerOps.mul_wide(self, rhs).checked_resize_ct()
    }
    fn checked_neg_ct(&self) -> CtValue<Self> {
        CtValue::new(self.wrapping_neg(), self.ct_is_zero())
    }
    fn checked_div_rem_ct(&self, rhs: &Self) -> CtValue<(Self, Self)> {
        CtValue::new(self.div_rem_words(rhs), !rhs.ct_is_zero())
    }
}

impl<const L: usize> Z<L> {
    pub const ZERO: Self = Self(Uint::ZERO);
    pub const ONE: Self = Self(Uint::ONE);
    pub const MIN: Self = {
        let mut words = [0; L];
        assert!(L > 0);
        words[L - 1] = 1 << 63;
        Self(Uint(words))
    };
    pub const MAX: Self = {
        let mut words = [u64::MAX; L];
        assert!(L > 0);
        words[L - 1] >>= 1;
        Self(Uint(words))
    };
    pub const fn from_twos_complement_words(words: [u64; L]) -> Self {
        Self(Uint::from_words(words))
    }
    pub const fn as_words(&self) -> &[u64; L] {
        self.0.as_words()
    }
    pub fn sign_extend<const M: usize>(&self) -> Z<M> {
        const {
            assert!(M >= L);
        }
        let mut out = [self.is_negative_ct().word(); M];
        out[..L].copy_from_slice(self.as_words());
        Z::from_twos_complement_words(out)
    }
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Z<M>> {
        const {
            assert!(M > 0);
        }
        let sign = self.is_negative_ct().word();
        let words = core::array::from_fn(|i| if i < L { self.0.0[i] } else { sign });
        let out = Z::<M>::from_twos_complement_words(words);
        let expected = out.is_negative_ct().word();
        let mut diff = sign ^ expected;
        for i in M..L {
            diff |= self.0.0[i] ^ expected;
        }
        CtValue::new(out, diff.ct_is_zero())
    }
    pub fn checked_to_unsigned_ct(&self) -> CtValue<Uint<L>> {
        CtValue::new(self.0, !self.is_negative_ct())
    }
    pub fn is_negative_ct(&self) -> CtMask {
        CtMask::from_lsb(self.0.0[L - 1] >> 63)
    }
    pub fn unsigned_abs(&self) -> Uint<L> {
        Uint::ct_select(&self.0, &self.0.wrapping_neg(), self.is_negative_ct())
    }
    pub fn arithmetic_shr(&self, public_shift: usize) -> Self {
        let fill = self.is_negative_ct().word();
        if public_shift >= L * 64 {
            return Self::from_twos_complement_words([fill; L]);
        }
        let mut out = self.0.shr(public_shift);
        let fill_bits = Uint([fill; L]).truncating_shl(L * 64 - public_shift);
        for i in 0..L {
            out.0[i] |= fill_bits.0[i];
        }
        Self(out)
    }
    pub fn truncating_shl(&self, public_shift: usize) -> Self {
        Self(self.0.truncating_shl(public_shift))
    }
    pub fn checked_shl_ct(&self, public_shift: usize) -> CtValue<Self> {
        let out = self.truncating_shl(public_shift);
        CtValue::new(out, out.arithmetic_shr(public_shift).ct_eq(self))
    }
}
impl<const L: usize> Default for Z<L> {
    fn default() -> Self {
        Self::ZERO
    }
}
impl<const L: usize> CtEq for Z<L> {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        self.0.ct_eq(&rhs.0)
    }
    fn ct_is_zero(&self) -> CtMask {
        self.0.ct_is_zero()
    }
}
impl<const L: usize> CtOrd for Z<L> {
    fn ct_lt(&self, rhs: &Self) -> CtMask {
        let signs_differ = CtMask::from_lsb((self.0.0[L - 1] ^ rhs.0.0[L - 1]) >> 63);
        CtMask::ct_select(&self.0.ct_lt(&rhs.0), &self.is_negative_ct(), signs_differ)
    }
}
impl<const L: usize> CtSelect for Z<L> {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self(Uint::ct_select(&a.0, &b.0, mask))
    }
}
impl<const L: usize> WrappingArithmetic for Z<L> {
    fn wrapping_add(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_add(&rhs.0))
    }
    fn wrapping_sub(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_sub(&rhs.0))
    }
    fn wrapping_neg(&self) -> Self {
        Self(self.0.wrapping_neg())
    }
    fn wrapping_mul(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_mul(&rhs.0))
    }
}
impl<const L: usize> CheckedArithmetic for Z<L> {
    fn checked_add_ct(&self, rhs: &Self) -> CtValue<Self> {
        let out = self.wrapping_add(rhs);
        let overflow =
            ((self.0.0[L - 1] ^ out.0.0[L - 1]) & (rhs.0.0[L - 1] ^ out.0.0[L - 1])) >> 63;
        CtValue::new(out, !CtMask::from_lsb(overflow))
    }
    fn checked_sub_ct(&self, rhs: &Self) -> CtValue<Self> {
        let out = self.wrapping_sub(rhs);
        let overflow =
            ((self.0.0[L - 1] ^ rhs.0.0[L - 1]) & (self.0.0[L - 1] ^ out.0.0[L - 1])) >> 63;
        CtValue::new(out, !CtMask::from_lsb(overflow))
    }
    fn checked_mul_ct(&self, rhs: &Self) -> CtValue<Self> {
        use crate::traits::WideMul;
        IntegerOps.mul_wide(self, rhs).checked_resize_ct()
    }
    fn checked_neg_ct(&self) -> CtValue<Self> {
        CtValue::new(self.wrapping_neg(), !self.ct_eq(&Self::MIN))
    }
    fn checked_div_rem_ct(&self, rhs: &Self) -> CtValue<(Self, Self)> {
        let (q, r) = self.unsigned_abs().div_rem_words(&rhs.unsigned_abs());
        let qsign = CtMask::from_lsb((self.0.0[L - 1] ^ rhs.0.0[L - 1]) >> 63);
        let q = Self(Uint::ct_select(&q, &q.wrapping_neg(), qsign));
        let r = Self(Uint::ct_select(
            &r,
            &r.wrapping_neg(),
            self.is_negative_ct(),
        ));
        let overflow = self.ct_eq(&Self::MIN) & rhs.ct_eq(&Self(Uint::MAX));
        CtValue::new((q, r), !rhs.ct_is_zero() & !overflow)
    }
}

macro_rules! bitwise {
    ($ty:ident) => {
        impl<const L: usize> core::ops::BitAnd for $ty<L> {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                Self::from_bits(core::array::from_fn(|i| {
                    self.as_words()[i] & rhs.as_words()[i]
                }))
            }
        }
        impl<const L: usize> core::ops::BitOr for $ty<L> {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self::from_bits(core::array::from_fn(|i| {
                    self.as_words()[i] | rhs.as_words()[i]
                }))
            }
        }
        impl<const L: usize> core::ops::BitXor for $ty<L> {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self {
                Self::from_bits(core::array::from_fn(|i| {
                    self.as_words()[i] ^ rhs.as_words()[i]
                }))
            }
        }
        impl<const L: usize> core::ops::Not for $ty<L> {
            type Output = Self;
            fn not(self) -> Self {
                Self::from_bits(core::array::from_fn(|i| !self.as_words()[i]))
            }
        }
    };
}
impl<const L: usize> Uint<L> {
    fn from_bits(words: [u64; L]) -> Self {
        Self::from_words(words)
    }
}
impl<const L: usize> Z<L> {
    fn from_bits(words: [u64; L]) -> Self {
        Self::from_twos_complement_words(words)
    }
}
bitwise!(Uint);
bitwise!(Z);

#[derive(Clone, Copy, Debug)]
pub struct UintRef<'a>(&'a [u64]);
impl<'a> UintRef<'a> {
    pub fn new(words: &'a [u64]) -> Self {
        assert!(!words.is_empty());
        Self(words)
    }
    pub fn as_words(&self) -> &'a [u64] {
        self.0
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ZRef<'a>(&'a [u64]);
impl<'a> ZRef<'a> {
    pub fn from_twos_complement_words(words: &'a [u64]) -> Self {
        assert!(!words.is_empty());
        Self(words)
    }
    /// Copy to a declared width, preserving sign and reporting discarded bits.
    pub fn checked_resize_ct<const L: usize>(&self) -> CtValue<Z<L>> {
        product::extract_signed(self.0)
    }
    pub fn as_words(&self) -> &'a [u64] {
        self.0
    }
}

/// Explicit arithmetic modulo the integer storage width.
#[derive(Clone, Copy, Debug, Default)]
pub struct WrappingOps<T>(core::marker::PhantomData<T>);
impl<T> WrappingOps<T> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}
macro_rules! wrapping_ring {
    ($ty:ident) => {
        impl<const L: usize> crate::traits::RingOps for WrappingOps<$ty<L>> {
            type Elem = $ty<L>;
            fn zero(&self) -> Self::Elem {
                $ty::ZERO
            }
            fn one(&self) -> Self::Elem {
                $ty::ONE
            }
            fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
                a.wrapping_add(b)
            }
            fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
                a.wrapping_sub(b)
            }
            fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
                a.wrapping_mul(b)
            }
            fn neg(&self, a: &Self::Elem) -> Self::Elem {
                a.wrapping_neg()
            }
        }
    };
}
wrapping_ring!(Uint);
wrapping_ring!(Z);

mod bit_ops;
