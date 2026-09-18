//! Fixed-width masks and explicit arithmetic validity.

use core::ops::{BitAnd, BitOr, BitXor, Not};

/// Either all zeroes or all ones. Construction never accepts an arbitrary mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct CtMask(u64);

impl CtMask {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(u64::MAX);

    #[inline(always)]
    pub fn from_lsb(bit: u64) -> Self {
        Self(0u64.wrapping_sub(opaque_word(bit & 1)))
    }

    #[inline(always)]
    pub(crate) fn word(self) -> u64 {
        self.0
    }

    /// Explicitly reveal this mask. Do not branch on private validity in a kernel.
    pub fn declassify(self) -> bool {
        self.0 != 0
    }
}

/// Make a mask's source opaque to value-based branch reconstruction. On the
/// supported native targets an empty register-only asm block avoids the stack
/// traffic emitted by `black_box(u64)`. The emitted instruction stream is still
/// inspected by the retained probes; this is not a universal compiler guarantee.
#[inline(always)]
fn opaque_word(value: u64) -> u64 {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        let mut value = value;
        // SAFETY: the block contains only a comment. It leaves the input/output
        // register unchanged, touches no memory or stack, and preserves flags.
        unsafe {
            core::arch::asm!("/* {value} */",value=inout(reg) value,options(nomem,nostack,preserves_flags));
        }
        value
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        core::hint::black_box(value)
    }
}

impl BitAnd for CtMask {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl BitOr for CtMask {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl Not for CtMask {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}
impl BitXor for CtMask {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

pub trait CtEq {
    fn ct_eq(&self, rhs: &Self) -> CtMask;
    fn ct_is_zero(&self) -> CtMask;
}

pub trait CtOrd: CtEq {
    fn ct_lt(&self, rhs: &Self) -> CtMask;
    fn ct_le(&self, rhs: &Self) -> CtMask {
        !rhs.ct_lt(self)
    }
}

pub trait CtSelect: Sized {
    /// Select `if_true` when the mask is true, otherwise `if_false`.
    fn ct_select(if_false: &Self, if_true: &Self, mask: CtMask) -> Self;
    fn ct_assign(&mut self, value: &Self, mask: CtMask) {
        *self = Self::ct_select(self, value, mask);
    }
}

impl CtEq for u64 {
    #[inline(always)]
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        (self ^ rhs).ct_is_zero()
    }
    #[inline(always)]
    fn ct_is_zero(&self) -> CtMask {
        CtMask::from_lsb(((self | self.wrapping_neg()) >> 63) ^ 1)
    }
}
impl CtOrd for u64 {
    #[inline(always)]
    fn ct_lt(&self, rhs: &Self) -> CtMask {
        CtMask::from_lsb(self.overflowing_sub(*rhs).1 as u64)
    }
}
impl CtSelect for u64 {
    #[inline(always)]
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let selected;
            // SAFETY: scalar comparison/select only. Both operands are loaded
            // before entering asm; the mask cannot select a memory address.
            // Flags are intentionally not declared preserved.
            unsafe {
                core::arch::asm!(
                    "cmp {mask}, #0", "csel {selected}, {a}, {b}, eq",
                    selected=lateout(reg) selected,a=in(reg) *a,b=in(reg) *b,
                    mask=in(reg) mask.word(),options(pure,nomem,nostack),
                );
            }
            selected
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            (a & !mask.word()) | (b & mask.word())
        }
    }
}
impl CtSelect for CtMask {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self(u64::ct_select(&a.0, &b.0, mask))
    }
}
impl<T: CtSelect, const N: usize> CtSelect for [T; N] {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        core::array::from_fn(|i| T::ct_select(&a[i], &b[i], mask))
    }
}

/// An integer-valued bit, distinct from a polynomial-basis byte.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bit(u8);
impl Bit {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const fn from_bool(bit: bool) -> Self {
        Self(bit as u8)
    }
    pub const fn from_lsb(word: u64) -> Self {
        Self((word & 1) as u8)
    }
    pub const fn as_u64(self) -> u64 {
        self.0 as u64
    }
    pub fn mask(self) -> CtMask {
        CtMask::from_lsb(self.as_u64())
    }
}
impl CtEq for Bit {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        self.as_u64().ct_eq(&rhs.as_u64())
    }
    fn ct_is_zero(&self) -> CtMask {
        self.as_u64().ct_is_zero()
    }
}
impl CtSelect for Bit {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self::from_lsb(u64::ct_select(&a.as_u64(), &b.as_u64(), mask))
    }
}

/// A defined value and an explicit validity mask; invalidity never short-circuits.
#[derive(Clone, Copy, Debug)]
pub struct CtValue<T> {
    value: T,
    valid: CtMask,
}
impl<T> CtValue<T> {
    pub const fn new(value: T, valid: CtMask) -> Self {
        Self { value, valid }
    }
    pub const fn value(&self) -> &T {
        &self.value
    }
    pub const fn validity(&self) -> CtMask {
        self.valid
    }
    pub fn into_parts(self) -> (T, CtMask) {
        (self.value, self.valid)
    }
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> CtValue<U> {
        CtValue::new(f(self.value), self.valid)
    }
    pub fn and_then<U>(self, f: impl FnOnce(T) -> CtValue<U>) -> CtValue<U> {
        let next = f(self.value);
        CtValue::new(next.value, self.valid & next.valid)
    }
}
