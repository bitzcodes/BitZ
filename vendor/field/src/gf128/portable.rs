//! Scalar carryless-multiply pipeline for `GF(2^128)`.
//!
//! Two jobs: the production path on targets without a carryless-multiply
//! instruction, and the oracle the SIMD kernels are tested against.

use super::REDUCTION;

/// Carryless product of two 64-bit polynomials, as `(low, high)` halves of the
/// 128-bit result.
#[inline]
#[cfg(target_pointer_width = "64")]
pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
    const M: u64 = 0x1111_1111_1111_1111;
    const W: u128 = 0x1111_1111_1111_1111_1111_1111_1111_1111;
    // Separate coefficients into four residue classes. Integer multiplication
    // counts polynomial terms in four-bit lanes; masking keeps their parity.
    // Remove b's top nibble so each count is at most 15: a count of 16 would
    // carry into the next lane and corrupt its parity.
    let low_b = b & 0x0fff_ffff_ffff_ffff;
    let a0 = (a & M) as u128;
    let a1 = (a & (M << 1)) as u128;
    let a2 = (a & (M << 2)) as u128;
    let a3 = (a & (M << 3)) as u128;
    let b0 = (low_b & M) as u128;
    let b1 = (low_b & (M << 1)) as u128;
    let b2 = (low_b & (M << 2)) as u128;
    let b3 = (low_b & (M << 3)) as u128;
    let z0 = (a0 * b0) ^ (a1 * b3) ^ (a2 * b2) ^ (a3 * b1);
    let z1 = (a0 * b1) ^ (a1 * b0) ^ (a2 * b3) ^ (a3 * b2);
    let z2 = (a0 * b2) ^ (a1 * b1) ^ (a2 * b0) ^ (a3 * b3);
    let z3 = (a0 * b3) ^ (a1 * b2) ^ (a2 * b1) ^ (a3 * b0);
    let mut product = (z0 & W) | (z1 & (W << 1)) | (z2 & (W << 2)) | (z3 & (W << 3));
    // Restore the four omitted coefficients with a fixed schedule. The barrier
    // prevents the compiler from turning the selection into an operand branch.
    for i in 60..64 {
        let mask = 0u128.wrapping_sub(core::hint::black_box((b >> i) & 1) as u128);
        product ^= ((a as u128) << i) & mask;
    }
    (product as u64, (product >> 64) as u64)
}

// On smaller targets, wide integer multiplication may call a data-dependent
// runtime helper. Retain the fixed bit scan instead of relying on that helper.
#[cfg(not(target_pointer_width = "64"))]
pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
    let mut lo = 0;
    let mut hi = 0;
    for i in 0..64 {
        let mask = 0u64.wrapping_sub(core::hint::black_box((b >> i) & 1));
        lo ^= (a << i) & mask;
        if i > 0 {
            hi ^= (a >> (64 - i)) & mask;
        }
    }
    (lo, hi)
}

/// Karatsuba 128x128 carryless product, as four words in ascending
/// significance.
#[inline]
pub fn clmul128(a: [u64; 2], b: [u64; 2]) -> [u64; 4] {
    let (l0, l1) = clmul64(a[0], b[0]);
    let (h0, h1) = clmul64(a[1], b[1]);
    let (m0, m1) = clmul64(a[0] ^ a[1], b[0] ^ b[1]);
    [l0, l1 ^ m0 ^ l0 ^ h0, h0 ^ m1 ^ l1 ^ h1, h1]
}

/// `w * g` where `g = X^7 + X^2 + X + 1`, as `(low, high)` 64-bit halves.
fn times_g(w: u64) -> (u64, u64) {
    (
        w ^ (w << 1) ^ (w << 2) ^ (w << 7),
        (w >> 63) ^ (w >> 62) ^ (w >> 57),
    )
}

/// Reduce a 256-bit carryless product modulo `X^128 + X^7 + X^2 + X + 1`.
///
/// Writing the input as `r0 + r1*X^64 + r2*X^128 + r3*X^192` and substituting
/// `X^128 = g`, the two high words fold down as `r2*g + (r3*g)*X^64`. That fold
/// spills at most seven bits back above `X^128`, which one more substitution
/// clears.
pub fn reduce(r: [u64; 4]) -> [u64; 2] {
    let (a0, a1) = times_g(r[2]);
    let (b0, b1) = times_g(r[3]);
    let w0 = r[0] ^ a0;
    let w1 = r[1] ^ a1 ^ b0;
    // `b1` holds at most 7 bits, so `b1*g` has at most 14 and stays in word 0.
    [w0 ^ b1 ^ (b1 << 1) ^ (b1 << 2) ^ (b1 << 7), w1]
}

#[inline]
pub fn mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    reduce(clmul128(a, b))
}

/// In characteristic 2 the cross terms of `(a0 + a1*X^64)^2` cancel, leaving
/// `a0^2 + a1^2*X^128`.
#[inline]
pub fn square(a: [u64; 2]) -> [u64; 2] {
    reduce(wide_square(a))
}

#[inline]
pub fn square_n(a: [u64; 2], k: u32) -> [u64; 2] {
    let mut w = a;
    for _ in 0..k {
        w = square(w);
    }
    w
}

/// An unreduced 256-bit value. Four words, least significant first — the
/// shape [`clmul128`] returns.
pub type Wide = [u64; 4];

pub(crate) fn wide_words(value: Wide) -> [u64; 4] {
    value
}
pub(crate) fn wide_square(a: [u64; 2]) -> Wide {
    // Squaring inserts one zero between adjacent coefficients.
    fn spread(word: u32) -> u64 {
        let mut x = word as u64;
        x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
        x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
        x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        (x | (x << 1)) & 0x5555_5555_5555_5555
    }
    [
        spread(a[0] as u32),
        spread((a[0] >> 32) as u32),
        spread(a[1] as u32),
        spread((a[1] >> 32) as u32),
    ]
}

pub fn wide_zero() -> Wide {
    [0; 4]
}

pub fn wide_of(a: [u64; 2]) -> Wide {
    [a[0], a[1], 0, 0]
}

pub fn wide_mul(a: [u64; 2], b: [u64; 2]) -> Wide {
    clmul128(a, b)
}

pub fn wide_add(x: Wide, y: Wide) -> Wide {
    [x[0] ^ y[0], x[1] ^ y[1], x[2] ^ y[2], x[3] ^ y[3]]
}

/// Add a reduced element into an accumulator: only the low words are touched,
/// so no zero high half is materialized.
pub fn wide_add_low(x: Wide, a: [u64; 2]) -> Wide {
    [x[0] ^ a[0], x[1] ^ a[1], x[2], x[3]]
}

pub fn wide_reduce(w: Wide) -> [u64; 2] {
    reduce(w)
}

/// Multiply by the generator `X`: a one-bit shift of the 128-bit value, folding
/// `X^128 = g` back in when the `X^127` coefficient overflows.
pub const fn mul_x(a: [u64; 2]) -> [u64; 2] {
    let carry = (a[1] >> 63).wrapping_neg();
    [
        (a[0] << 1) ^ (carry & REDUCTION),
        (a[1] << 1) | (a[0] >> 63),
    ]
}

/// This module is the oracle the SIMD kernels are checked against, so agreeing
/// with them proves nothing. Each primitive is pinned instead to an independent
/// derivation — `clmul64` to the definition of a polynomial product, `reduce`
/// to explicit long division — each written in a different shape from the code
/// under test, so a shared bug would have to survive two unrelated
/// formulations.
#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    fn bit(words: &[u64], i: usize) -> bool {
        (words[i / 64] >> (i % 64)) & 1 == 1
    }

    fn flip(words: &mut [u64], i: usize) {
        words[i / 64] ^= 1 << (i % 64);
    }

    /// `(a*b)_k` is the parity of `{(i, j) : i + j = k, a_i = b_j = 1}` — the
    /// definition of a polynomial product, evaluated term by term.
    fn clmul64_by_definition(a: u64, b: u64) -> (u64, u64) {
        let mut out = [0u64; 2];
        for i in 0..64 {
            for j in 0..64 {
                if (a >> i) & 1 == 1 && (b >> j) & 1 == 1 {
                    flip(&mut out, i + j);
                }
            }
        }
        (out[0], out[1])
    }

    /// Long division by `f = X^128 + X^7 + X^2 + X + 1`: clear the highest set
    /// bit at or above 128 by subtracting a shifted copy of `f`, and repeat.
    /// Structurally unlike `reduce`, which folds twice in closed form.
    fn reduce_by_long_division(r: [u64; 4]) -> [u64; 2] {
        let mut r = r;
        for i in (128..256).rev() {
            if bit(&r, i) {
                flip(&mut r, i);
                for term in [7, 2, 1, 0] {
                    flip(&mut r, i - 128 + term);
                }
            }
        }
        [r[0], r[1]]
    }

    #[test]
    fn clmul64_matches_the_coefficient_definition() {
        let mut rng = Pcg64::seed_from_u64(101);
        let edges = [0, 1, 2, u64::MAX, 1 << 63, REDUCTION];
        for &a in &edges {
            for &b in &edges {
                assert_eq!(
                    clmul64(a, b),
                    clmul64_by_definition(a, b),
                    "{a:#x} * {b:#x}"
                );
            }
        }
        for _ in 0..512 {
            let (a, b) = (rng.next_u64(), rng.next_u64());
            assert_eq!(
                clmul64(a, b),
                clmul64_by_definition(a, b),
                "{a:#x} * {b:#x}"
            );
        }
    }

    #[test]
    fn dense_products_and_top_nibbles_match_the_definition() {
        // Exercise maximal lane counts and every restored top-nibble pattern.
        for nibble in 0..16u64 {
            let b = 0x0fff_ffff_ffff_ffff | (nibble << 60);
            for a in [u64::MAX, 0x1111_1111_1111_1111, 0x8888_8888_8888_8888] {
                assert_eq!(clmul64(a, b), clmul64_by_definition(a, b));
                assert_eq!(clmul64(b, a), clmul64_by_definition(b, a));
            }
        }
    }

    #[test]
    fn clmul128_matches_polynomial_convolution() {
        let mut rng = Pcg64::seed_from_u64(104);
        for _ in 0..256 {
            let a = [rng.next_u64(), rng.next_u64()];
            let b = [rng.next_u64(), rng.next_u64()];
            let mut expected = [0; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let (lo, hi) = clmul64_by_definition(a[i], b[j]);
                    expected[i + j] ^= lo;
                    expected[i + j + 1] ^= hi;
                }
            }
            assert_eq!(clmul128(a, b), expected);
        }
    }

    #[test]
    fn reduce_matches_long_division() {
        let mut rng = Pcg64::seed_from_u64(102);
        // Every single-bit input in turn, so each of the 128 foldable positions
        // is exercised alone rather than only in random combination.
        for i in 0..256 {
            let mut r = [0u64; 4];
            flip(&mut r, i);
            assert_eq!(reduce(r), reduce_by_long_division(r), "bit {i}");
        }
        for _ in 0..512 {
            let r = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];
            assert_eq!(reduce(r), reduce_by_long_division(r), "{r:?}");
        }
    }

    /// Bit spreading is checked against polynomial multiplication, independent
    /// of whichever hardware kernel is active.
    #[test]
    fn square_matches_the_general_multiply() {
        let mut rng = Pcg64::seed_from_u64(103);
        for _ in 0..512 {
            let a = [rng.next_u64(), rng.next_u64()];
            assert_eq!(square(a), mul(a, a), "{a:?}");
        }
    }

    /// Products that land exactly on the `X^128` boundary, where a wrong
    /// reduction polynomial or a swapped word order shows up immediately.
    #[test]
    fn boundary_products() {
        let x = [2, 0];
        assert_eq!(mul(x, [1 << 63, 0]), [0, 1]); // X * X^63  = X^64
        assert_eq!(mul(x, [0, 1 << 63]), [REDUCTION, 0]); // X * X^127 = g
        assert_eq!(mul([0, 1], [0, 1]), [REDUCTION, 0]); // X^64 * X^64 = g
        assert_eq!(mul_x([0, 1 << 63]), [REDUCTION, 0]);
    }
}

#[inline]
pub fn wide_from_words(words: [u64; 4]) -> Wide {
    words
}
