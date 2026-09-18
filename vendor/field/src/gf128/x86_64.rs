// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Adapted from the local Flock PCLMUL kernels; see VENDORED.md.

//! PCLMUL arithmetic. This module requires both pclmulqdq and sse4.1.
//! Products remain in vector registers across XOR accumulation.

use core::arch::x86_64::*;

pub type Wide = (__m128i, __m128i);

#[inline(always)]
fn load(words: [u64; 2]) -> __m128i {
    // SAFETY: the module's compile-time gate includes the required instructions;
    // the unaligned load reads exactly the supplied two-word array.
    unsafe { _mm_loadu_si128(words.as_ptr().cast()) }
}
#[inline(always)]
fn store(value: __m128i) -> [u64; 2] {
    let mut words = [0; 2];
    // SAFETY: the array has room for the complete unaligned 128-bit store.
    unsafe { _mm_storeu_si128(words.as_mut_ptr().cast(), value) };
    words
}
#[inline(always)]
fn reduce(low: __m128i, high: __m128i) -> __m128i {
    // SAFETY: pclmulqdq, sse4.1 and baseline SSE2 are enabled for this module.
    unsafe {
        let polynomial = _mm_set_epi64x(0, 0x87);
        let p0 = _mm_clmulepi64_si128::<0x00>(high, polynomial);
        let p1 = _mm_clmulepi64_si128::<0x01>(high, polynomial);
        let spill = _mm_extract_epi64::<1>(p1) as u64;
        let correction = spill ^ (spill << 1) ^ (spill << 2) ^ (spill << 7);
        _mm_xor_si128(
            _mm_xor_si128(low, p0),
            _mm_xor_si128(
                _mm_slli_si128::<8>(p1),
                _mm_set_epi64x(0, correction as i64),
            ),
        )
    }
}
#[inline]
pub fn mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    // SAFETY: all required features are checked at the module boundary.
    unsafe {
        let a = load(a);
        let b = load(b);
        let p0 = _mm_clmulepi64_si128::<0x00>(a, b);
        let p2 = _mm_clmulepi64_si128::<0x11>(a, b);
        let ax = _mm_xor_si128(a, _mm_shuffle_epi32::<0x4e>(a));
        let bx = _mm_xor_si128(b, _mm_shuffle_epi32::<0x4e>(b));
        let cross = _mm_xor_si128(_mm_clmulepi64_si128::<0x00>(ax, bx), _mm_xor_si128(p0, p2));
        store(reduce(
            _mm_xor_si128(p0, _mm_slli_si128::<8>(cross)),
            _mm_xor_si128(p2, _mm_srli_si128::<8>(cross)),
        ))
    }
}
#[inline(always)]
fn square_inner(a: __m128i) -> __m128i {
    // SAFETY: pclmulqdq is enabled; cross terms cancel in characteristic two.
    unsafe {
        reduce(
            _mm_clmulepi64_si128::<0x00>(a, a),
            _mm_clmulepi64_si128::<0x11>(a, a),
        )
    }
}
#[inline]
pub fn square(a: [u64; 2]) -> [u64; 2] {
    store(square_inner(load(a)))
}
#[inline]
pub fn square_n(a: [u64; 2], n: u32) -> [u64; 2] {
    let mut value = load(a);
    for _ in 0..n {
        value = square_inner(value);
    }
    store(value)
}
#[inline]
pub fn wide_zero() -> Wide {
    // SAFETY: SSE2 is baseline for x86_64.
    unsafe { (_mm_setzero_si128(), _mm_setzero_si128()) }
}
#[inline]
pub fn wide_of(a: [u64; 2]) -> Wide {
    // SAFETY: SSE2 is baseline for x86_64.
    unsafe { (load(a), _mm_setzero_si128()) }
}
#[inline]
pub fn wide_mul(a: [u64; 2], b: [u64; 2]) -> Wide {
    // SAFETY: pclmulqdq is enabled at the module boundary.
    unsafe {
        let a = load(a);
        let b = load(b);
        let low = _mm_clmulepi64_si128::<0x00>(a, b);
        let high = _mm_clmulepi64_si128::<0x11>(a, b);
        let cross = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x01>(a, b),
            _mm_clmulepi64_si128::<0x10>(a, b),
        );
        (
            _mm_xor_si128(low, _mm_slli_si128::<8>(cross)),
            _mm_xor_si128(high, _mm_srli_si128::<8>(cross)),
        )
    }
}
#[inline]
pub fn wide_add(a: Wide, b: Wide) -> Wide {
    // SAFETY: SSE2 is baseline for x86_64.
    unsafe { (_mm_xor_si128(a.0, b.0), _mm_xor_si128(a.1, b.1)) }
}
#[inline]
pub fn wide_add_low(a: Wide, b: [u64; 2]) -> Wide {
    // SAFETY: SSE2 is baseline for x86_64.
    unsafe { (_mm_xor_si128(a.0, load(b)), a.1) }
}
#[inline]
pub fn wide_reduce(value: Wide) -> [u64; 2] {
    store(reduce(value.0, value.1))
}
pub(crate) fn wide_words(value: Wide) -> [u64; 4] {
    let lo = store(value.0);
    let hi = store(value.1);
    [lo[0], lo[1], hi[0], hi[1]]
}
#[inline]
pub(crate) fn wide_square(a: [u64; 2]) -> Wide {
    // SAFETY: pclmulqdq is enabled for the module.
    unsafe {
        let a = load(a);
        (
            _mm_clmulepi64_si128::<0x00>(a, a),
            _mm_clmulepi64_si128::<0x11>(a, a),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf128::portable;
    #[test]
    fn pclmul_matches_portable_before_and_after_reduction() {
        let mut seed = 0x712f48267281u64;
        let mut word = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        let mut sum = wide_zero();
        let mut reference = portable::wide_zero();
        for _ in 0..2048 {
            let a = [word(), word()];
            let b = [word(), word()];
            let product = wide_mul(a, b);
            let expected = portable::wide_mul(a, b);
            assert_eq!(wide_words(product), expected);
            assert_eq!(mul(a, b), portable::mul(a, b));
            assert_eq!(square(a), portable::square(a));
            sum = wide_add(sum, product);
            reference = portable::wide_add(reference, expected);
        }
        assert_eq!(wide_words(sum), reference);
        assert_eq!(wide_reduce(sum), portable::wide_reduce(reference));
    }
}

#[inline]
pub fn wide_from_words(words: [u64; 4]) -> Wide {
    (load([words[0], words[1]]), load([words[2], words[3]]))
}
