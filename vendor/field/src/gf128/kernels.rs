//! Architecture kernels used by fused consumers and NTT schedules.
//!
//! These functions retain the accepted Flock schedules. Consumers must gate the
//! complete instruction set before calling an architecture entry point.

use crate::{Gf128, Gf128Product};

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub mod aarch64;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "sse4.1"
))]
pub mod x86_64;

#[inline]
pub const fn mul_by_x(value: Gf128) -> Gf128 {
    value.mul_x()
}

#[inline]
pub fn ghash_reduce(r0: u64, r1: u64, r2: u64, r3: u64) -> Gf128 {
    super::portable::reduce([r0, r1, r2, r3]).into()
}

/// Fixed-schedule portable arithmetic, also used as a differential reference.
pub mod software {
    pub fn clmul_64x64(a: u64, b: u64) -> [u64; 2] {
        let (lo, hi) = super::super::portable::clmul64(a, b);
        [lo, hi]
    }

    use super::{Gf128, Gf128Product};
    #[inline]
    pub fn ghash_mul(a: Gf128, b: Gf128) -> Gf128 {
        super::super::portable::mul([a.lo, a.hi], [b.lo, b.hi]).into()
    }
    #[inline]
    pub fn ghash_square(a: Gf128) -> Gf128 {
        super::super::portable::square([a.lo, a.hi]).into()
    }
    #[inline]
    pub fn ghash_mul_unreduced(a: Gf128, b: Gf128) -> Gf128Product {
        Gf128Product::from_polynomial_words(super::super::portable::wide_mul(
            [a.lo, a.hi],
            [b.lo, b.hi],
        ))
    }
}

/// Exact product of two 64-bit polynomial-basis words. Instruction gates
/// cover PMULL's AES extension and x86 PCLMUL separately.
#[inline]
pub fn clmul_64x64(a: u64, b: u64) -> [u64; 2] {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        let p: u128 = core::mem::transmute(core::arch::aarch64::vmull_p64(a, b));
        return [p as u64, (p >> 64) as u64];
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::{_mm_clmulepi64_si128, _mm_set_epi64x};
        return core::mem::transmute(_mm_clmulepi64_si128::<0>(
            _mm_set_epi64x(0, a as i64),
            _mm_set_epi64x(0, b as i64),
        ));
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    )))]
    {
        software::clmul_64x64(a, b)
    }
}

/// Karatsuba product, retaining the BitZ word kernel for callers consuming
/// exact polynomial words rather than an architecture accumulator.
#[inline]
pub fn clmul_128x128(a: &[u64; 2], b: &[u64; 2]) -> [u64; 4] {
    let lo = clmul_64x64(a[0], b[0]);
    let hi = clmul_64x64(a[1], b[1]);
    let cross = clmul_64x64(a[0] ^ a[1], b[0] ^ b[1]);
    [
        lo[0],
        lo[1] ^ cross[0] ^ lo[0] ^ hi[0],
        hi[0] ^ cross[1] ^ lo[1] ^ hi[1],
        hi[1],
    ]
}

/// Canonical GHASH remainder of an exact four-word polynomial.
#[inline]
pub fn reduce_256_to_128(words: [u64; 4]) -> [u64; 2] {
    super::portable::reduce(words)
}
