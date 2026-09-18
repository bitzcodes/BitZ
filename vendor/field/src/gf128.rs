//! `GF(2^128) = F_2[X] / <X^128 + X^7 + X^2 + X + 1>` — the GHASH field.
//!
//! Bit `i` of the little-endian value `hi:lo` is the coefficient of `X^i`:
//! `lo` carries `X^0..X^63`, `hi` carries `X^64..X^127`. This is *not* the
//! reflected byte order NIST SP 800-38D uses, so published GHASH vectors apply
//! only through a bit-reversal.
//!
//! Layout matches flock's `Gf128`, so converting between the two is a field
//! copy.

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::{ConstOne, ConstZero, Inv, One, Zero};

// Always compiled: the active kernel where no carryless-multiply instruction
// exists, and the oracle the SIMD kernels are tested against. On aarch64 only
// tests call it, hence the allow.
#[cfg_attr(all(target_arch = "aarch64", target_feature = "aes"), allow(dead_code))]
mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(crate) mod aarch64;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "sse4.1"
))]
mod x86_64;

pub mod kernels;

mod pow;
mod sumcheck;
mod wide;

pub use pow::{MULT_ORDER, ORDER_PRIME_FACTORS, is_generator, smallest_generator};
pub use wide::Wide256;

// The multiply and square in use on this target. The gate is `aes`, not `neon`:
// `pmull` is a crypto extension, on by default for `aarch64-apple-darwin` but
// not for `aarch64-unknown-linux-gnu`, which silently gets `portable`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use aarch64 as kernel;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(
        target_arch = "x86_64",
        target_feature = "pclmulqdq",
        target_feature = "sse4.1"
    )
)))]
use portable as kernel;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "sse4.1"
))]
use x86_64 as kernel;

/// Which kernel this build selected. A timing means nothing without it.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub const KERNEL: &str = "neon";
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "sse4.1"
))]
pub const KERNEL: &str = "pclmul-karatsuba-barrett";
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(
        target_arch = "x86_64",
        target_feature = "pclmulqdq",
        target_feature = "sse4.1"
    )
)))]
pub const KERNEL: &str = "portable";

/// Low bits of the reduction polynomial: `X^128 = X^7 + X^2 + X + 1`.
pub const REDUCTION: u64 = 0x87;

/// An element of `GF(2^128)`.
///
/// Two `u64` words rather than one `u128`: the words map onto the kernels'
/// 64-bit SIMD lanes (`pmull` multiplies 64x64), scalar `u128` arithmetic
/// lowers to the same word ops anyway, and on wasm32 `u128` multiplies
/// become libcalls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(C, align(16))]
pub struct Gf128 {
    pub lo: u64,
    pub hi: u64,
}

impl Gf128 {
    pub const fn from_polynomial_words(words: [u64; 2]) -> Self {
        Self::new(words[0], words[1])
    }
    pub const fn as_integer(&self) -> &crate::Uint<2> {
        // SAFETY: Uint<2> is transparent over [u64;2]; this repr(C) value
        // contains exactly two adjacent words with at least that alignment.
        unsafe { &*(self.as_words() as *const [u64; 2]).cast::<crate::Uint<2>>() }
    }
    pub fn reduce_wide(words: [u64; 4]) -> Self {
        Self::from(kernels::reduce_256_to_128(words))
    }

    /// Polynomial-basis words in little-endian order.
    pub const fn as_words(&self) -> &[u64; 2] {
        // SAFETY: repr(C) contains exactly two u64s with no padding.
        unsafe { &*(core::ptr::addr_of!(self.lo).cast::<[u64; 2]>()) }
    }

    pub const ZERO: Self = Self::new(0, 0);
    pub const ONE: Self = Self::new(1, 0);

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    /// Fixed-schedule inverse, mapping zero to zero.
    #[inline]
    pub fn inverse_or_zero(self) -> Self {
        use crate::FieldOps;
        *crate::Gf128Ops.inverse_ct(&self).value()
    }

    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> crate::Gf128Product {
        use crate::WideMul;
        crate::Gf128Ops.mul_wide(&self, &rhs)
    }

    /// Interpret bits as polynomial coefficients, rather than numeric integer
    /// embedding (which is parity in characteristic two).
    pub const fn from_polynomial_bits(value: u128) -> Self {
        Self::new(value as u64, (value >> 64) as u64)
    }

    /// `X`, whose multiplicative order is the full `2^128 - 1` — checked by
    /// [`is_generator`], not assumed. `X` in AES's `GF(2^8)` has order 51 of
    /// 255.
    pub const GENERATOR: Self = Self::new(2, 0);

    pub const fn new(lo: u64, hi: u64) -> Self {
        Self { lo, hi }
    }

    #[inline]
    pub fn square(self) -> Self {
        kernel::square(self.words()).into()
    }

    /// Multiply by `X`: a shift and a conditional fold, cheaper than the
    /// general multiply.
    pub const fn mul_x(self) -> Self {
        let [lo, hi] = portable::mul_x(self.words());
        Self { lo, hi }
    }

    /// Canonical encoding: 16 bytes little-endian, `lo` first — the encoding
    /// flock's transcript absorbs.
    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.lo.to_le_bytes());
        out[8..].copy_from_slice(&self.hi.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        let mut lo = [0u8; 8];
        let mut hi = [0u8; 8];
        lo.copy_from_slice(&bytes[..8]);
        hi.copy_from_slice(&bytes[8..]);
        Self::new(u64::from_le_bytes(lo), u64::from_le_bytes(hi))
    }

    const fn words(self) -> [u64; 2] {
        [self.lo, self.hi]
    }
}

/// The bit pattern as 32 hex digits, `hi` first — how the polynomial reads on
/// paper, high coefficients leftmost.
impl Display for Gf128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:#034x}", (self.hi as u128) << 64 | self.lo as u128)
    }
}

impl Zero for Gf128 {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.lo == 0 && self.hi == 0
    }
}

impl One for Gf128 {
    fn one() -> Self {
        Self::ONE
    }
}

impl ConstZero for Gf128 {
    const ZERO: Self = Self::new(0, 0);
}

impl ConstOne for Gf128 {
    const ONE: Self = Self::new(1, 0);
}

impl From<[u64; 2]> for Gf128 {
    #[inline]
    fn from(words: [u64; 2]) -> Self {
        Self::new(words[0], words[1])
    }
}

impl From<bool> for Gf128 {
    fn from(bit: bool) -> Self {
        Self::new(bit as u64, 0)
    }
}

impl Add for Gf128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.lo ^ rhs.lo, self.hi ^ rhs.hi)
    }
}

impl Add<&Gf128> for Gf128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self {
        self.add(*rhs)
    }
}

impl AddAssign for Gf128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

impl AddAssign<&Gf128> for Gf128 {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        *self += *rhs;
    }
}

/// Characteristic 2: subtraction is addition.
impl Sub for Gf128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.lo ^ rhs.lo, self.hi ^ rhs.hi)
    }
}

impl Sub<&Gf128> for Gf128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        self.sub(*rhs)
    }
}

impl SubAssign for Gf128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

impl SubAssign<&Gf128> for Gf128 {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        *self -= *rhs;
    }
}

impl Neg for Gf128 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl Mul for Gf128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        kernel::mul(self.words(), rhs.words()).into()
    }
}

impl Mul<&Gf128> for Gf128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        self.mul(*rhs)
    }
}

impl MulAssign for Gf128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<&Gf128> for Gf128 {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl Sum for Gf128 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<'a> Sum<&'a Gf128> for Gf128 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl Product for Gf128 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}

impl<'a> Product<&'a Gf128> for Gf128 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}

impl Div for Gf128 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)] // Division is multiplication by the inverse
    fn div(self, rhs: Self) -> Self {
        self * rhs.inv().expect("Division by zero")
    }
}

impl Div<&Gf128> for Gf128 {
    type Output = Self;
    fn div(self, rhs: &Self) -> Self {
        self.div(*rhs)
    }
}

impl DivAssign for Gf128 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl DivAssign<&Gf128> for Gf128 {
    fn div_assign(&mut self, rhs: &Self) {
        *self = *self / *rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    /// A random field element; the tests only need uniform 128-bit words.
    fn f128(rng: &mut Pcg64) -> Gf128 {
        Gf128::new(rng.next_u64(), rng.next_u64())
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn div_by_zero_panics() {
        let _ = Gf128::ONE / Gf128::ZERO;
    }

    #[test]
    fn layout_matches_flock() {
        assert_eq!(size_of::<Gf128>(), 16);
        assert_eq!(align_of::<Gf128>(), 16);
    }

    #[test]
    fn zero_and_one_are_identities() {
        let mut rng = Pcg64::seed_from_u64(1);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(a + Gf128::ZERO, a);
            assert_eq!(a * Gf128::ONE, a);
            assert_eq!(a * Gf128::ZERO, Gf128::ZERO);
        }
    }

    #[test]
    fn addition_is_xor_and_self_inverse() {
        let mut rng = Pcg64::seed_from_u64(2);
        for _ in 0..256 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            assert_eq!(a + b, Gf128::new(a.lo ^ b.lo, a.hi ^ b.hi));
            assert_eq!(a + a, Gf128::ZERO);
            assert_eq!(a - b, a + b);
            assert_eq!(-a, a);
        }
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        let mut rng = Pcg64::seed_from_u64(3);
        for _ in 0..256 {
            let (a, b, c) = (f128(&mut rng), f128(&mut rng), f128(&mut rng));
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
        }
    }

    #[test]
    fn multiplication_distributes_over_addition() {
        let mut rng = Pcg64::seed_from_u64(4);
        for _ in 0..256 {
            let (a, b, c) = (f128(&mut rng), f128(&mut rng), f128(&mut rng));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn square_matches_self_multiply() {
        let mut rng = Pcg64::seed_from_u64(5);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn mul_x_matches_multiply_by_generator() {
        let mut rng = Pcg64::seed_from_u64(6);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(a.mul_x(), a * Gf128::GENERATOR);
        }
    }

    /// These pin down the reduction polynomial *and* the word order at once:
    /// get either wrong and at least one fails.
    #[test]
    fn reduction_boundary_products() {
        let x = Gf128::GENERATOR;
        let x_63 = Gf128::new(1 << 63, 0);
        let x_64 = Gf128::new(0, 1);
        let x_127 = Gf128::new(0, 1 << 63);

        assert_eq!(x * x_63, x_64); // crosses the word boundary
        assert_eq!(x * x_127, Gf128::new(REDUCTION, 0)); // crosses X^128
        assert_eq!(x_64 * x_64, Gf128::new(REDUCTION, 0)); // the same, from both halves
    }

    /// Squaring is the Frobenius endomorphism, hence `F_2`-linear.
    #[test]
    fn squaring_is_additive() {
        let mut rng = Pcg64::seed_from_u64(7);
        for _ in 0..256 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            assert_eq!((a + b).square(), a.square() + b.square());
        }
    }

    #[test]
    fn byte_encoding_round_trips() {
        let mut rng = Pcg64::seed_from_u64(8);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(Gf128::from_bytes(a.to_bytes()), a);
        }
        assert_eq!(Gf128::ONE.to_bytes()[0], 1);
        assert_eq!(Gf128::new(0, 1).to_bytes()[8], 1);
    }

    /// The SIMD kernels must agree with the portable pipeline bit for bit, on
    /// the boundary cases as well as on random input.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_matches_portable() {
        let edges = [
            Gf128::ZERO,
            Gf128::ONE,
            Gf128::GENERATOR,
            Gf128::new(REDUCTION, 0),
            Gf128::new(1 << 63, 0),
            Gf128::new(0, 1),
            Gf128::new(0, 1 << 63),
            Gf128::new(u64::MAX, u64::MAX),
        ];

        let mut rng = Pcg64::seed_from_u64(9);
        let mut cases: Vec<(Gf128, Gf128)> = Vec::new();
        for &a in &edges {
            for &b in &edges {
                cases.push((a, b));
            }
            for _ in 0..64 {
                cases.push((a, f128(&mut rng)));
                cases.push((f128(&mut rng), a));
            }
        }
        for _ in 0..2048 {
            cases.push((f128(&mut rng), f128(&mut rng)));
        }

        for (a, b) in cases {
            assert_eq!(
                aarch64::mul(a.words(), b.words()),
                portable::mul(a.words(), b.words()),
                "multiply disagrees on {a:?} * {b:?}"
            );
            assert_eq!(
                aarch64::square(a.words()),
                portable::square(a.words()),
                "square disagrees on {a:?}"
            );
            // The squaring runs are separate loops on each side, so they are
            // checked against each other rather than only against `square`.
            for k in [0, 1, 6, 24, 48, 127] {
                assert_eq!(
                    aarch64::square_n(a.words(), k),
                    portable::square_n(a.words(), k),
                    "square_n({k}) disagrees on {a:?}"
                );
            }
        }
    }

    #[test]
    fn u128_conversion_is_the_bit_pattern() {
        assert_eq!(Gf128::from_polynomial_bits(1u128), Gf128::ONE);
        assert_eq!(Gf128::from_polynomial_bits(1u128 << 64), Gf128::new(0, 1));
        assert_eq!(Gf128::from(true), Gf128::ONE);
        assert_eq!(Gf128::from(false), Gf128::ZERO);
    }
}
