//! Shared binary arithmetic and Flock-owned slice execution helpers.

pub use ::field::{Gf128, Gf128Product, Gf8};
pub use ::field::binary::{AES_EMBEDDING_TABLE, embed_gf8};
pub use ::field::binary::gf8 as gf8_kernels;
pub use ::field::gf128::kernels as gf128_kernels;
pub use ::field::gf128::kernels::mul_by_x;

pub(crate) mod f128_slice;

#[cfg(test)]
mod tests {
    use super::*;
    use field::gf128::kernels::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn next_f128(&mut self) -> Gf128 {
            Gf128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    #[test]
    fn add_identities() {
        let mut rng = Rng::new(1);
        for _ in 0..64 {
            let a = rng.next_f128();
            assert_eq!(a + Gf128::ZERO, a);
            assert_eq!(a + a, Gf128::ZERO);
        }
    }

    #[test]
    fn mul_identities() {
        let mut rng = Rng::new(2);
        for _ in 0..64 {
            let a = rng.next_f128();
            assert_eq!(a * Gf128::ZERO, Gf128::ZERO);
            assert_eq!(a * Gf128::ONE, a);
        }
    }

    #[test]
    fn mul_by_x_matches_mul_by_gen() {
        let mut rng = Rng::new(3);
        for _ in 0..256 {
            let a = rng.next_f128();
            assert_eq!(mul_by_x(a), a * Gf128::GENERATOR);
        }
    }

    #[test]
    fn deferred_reduction_matches_direct() {
        let mut rng = Rng::new(4);
        for _ in 0..64 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let direct = a * b;
            let deferred = a.mul_unreduced(b).reduce();
            assert_eq!(direct, deferred);
        }
    }

    #[test]
    fn deferred_xor_commutes_with_reduction() {
        // Σ aᵢ·bᵢ in Gf128 must equal reduce(XOR-sum of unreduced products).
        let mut rng = Rng::new(5);
        let n = 16;
        let pairs: Vec<(Gf128, Gf128)> = (0..n).map(|_| (rng.next_f128(), rng.next_f128())).collect();

        let direct: Gf128 = pairs.iter().fold(Gf128::ZERO, |acc, (a, b)| acc + *a * *b);

        let mut acc = Gf128Product::zero();
        for (a, b) in &pairs {
            acc ^= a.mul_unreduced(*b);
        }
        assert_eq!(direct, acc.reduce());
    }

    #[test]
    fn inverse_roundtrip() {
        let mut rng = Rng::new(6);
        for _ in 0..16 {
            let a = rng.next_f128();
            if a.is_zero() {
                continue;
            }
            assert_eq!(a * a.inverse_or_zero(), Gf128::ONE);
        }
    }

    #[test]
    fn associativity_random() {
        let mut rng = Rng::new(7);
        for _ in 0..64 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let c = rng.next_f128();
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn mul_commutativity() {
        let mut rng = Rng::new(91);
        for _ in 0..256 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            assert_eq!(a * b, b * a);
        }
    }

    #[test]
    fn ghash_reduction_smoking_gun() {
        // The defining identity of the GHASH polynomial:
        //   x · x^127 = x^128 = x^7 + x^2 + x + 1 = 0x87.
        // If the reduction constant 0x87 is wrong (e.g. 0x86, 0x07, byte-swapped),
        // this test fails immediately and pinpoints the bug.
        let x = Gf128::GENERATOR;
        let x_127 = Gf128 {
            lo: 0,
            hi: 1u64 << 63,
        };
        assert_eq!(x * x_127, Gf128 { lo: 0x87, hi: 0 }, "x · x^127");

        // x · x^63 = x^64 — crosses the lo/hi word boundary with no reduction.
        // Catches lo/hi swaps and off-by-one in the 64-bit word split.
        let x_63 = Gf128 {
            lo: 1u64 << 63,
            hi: 0,
        };
        assert_eq!(x * x_63, Gf128 { lo: 0, hi: 1 }, "x · x^63 = x^64");

        // x^64 · x^64 = x^128 = 0x87 — reaches the reduction through a different
        // multiplication path (high·high product).
        let x_64 = Gf128 { lo: 0, hi: 1 };
        assert_eq!(x_64 * x_64, Gf128 { lo: 0x87, hi: 0 }, "x^64 · x^64");

        // x · x = x^2 (no reduction).
        assert_eq!(x * x, Gf128 { lo: 4, hi: 0 }, "x^2");
    }

    #[test]
    fn high_bit_inputs_reduce_correctly() {
        // Verify mul still satisfies a^{-1} · a = 1 when both inputs have the
        // top bit (x^127) set — exercising the most overflow-prone code path
        // of `ghash_reduce`. The inverse test naturally lands here for random
        // inputs only by luck; this makes it deterministic.
        let high = Gf128 {
            lo: 0,
            hi: 1u64 << 63,
        };
        assert_eq!(high * high.inverse_or_zero(), Gf128::ONE);
        let almost_max = Gf128 {
            lo: u64::MAX,
            hi: u64::MAX,
        };
        assert_eq!(almost_max * almost_max.inverse_or_zero(), Gf128::ONE);
        let just_top = Gf128 {
            lo: 0,
            hi: u64::MAX,
        };
        assert_eq!(just_top * just_top.inverse_or_zero(), Gf128::ONE);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_mul_vec2_matches_scalar() {
        let mut rng = Rng::new(11);
        for _ in 0..128 {
            let a0 = rng.next_f128();
            let a1 = rng.next_f128();
            let b0 = rng.next_f128();
            let b1 = rng.next_f128();
            let expected = [a0 * b0, a1 * b1];
            let result = unsafe { aarch64::ghash_mul_vec2_neon([a0, a1], [b0, b1]) };
            assert_eq!(result[0], expected[0], "lane 0");
            assert_eq!(result[1], expected[1], "lane 1");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn all_neon_variants_agree() {
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let sw = software::ghash_mul(a, b);
            let sb = unsafe { aarch64::ghash_mul_schoolbook(a, b) };
            let ka = unsafe { aarch64::ghash_mul_karatsuba(a, b) };
            let kb = unsafe { aarch64::ghash_mul_karatsuba_barrett(a, b) };
            let bi = unsafe { aarch64::ghash_mul_binius(a, b) };
            assert_eq!(sw, sb);
            assert_eq!(sw, ka);
            assert_eq!(sw, kb);
            assert_eq!(sw, bi);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    #[test]
    fn all_x86_variants_agree() {
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let sw = software::ghash_mul(a, b);
            let sb = unsafe { x86_64::ghash_mul_schoolbook(a, b) };
            let ka = unsafe { x86_64::ghash_mul_karatsuba(a, b) };
            let kb = unsafe { x86_64::ghash_mul_karatsuba_barrett(a, b) };
            let bi = unsafe { x86_64::ghash_mul_binius(a, b) };
            // Unreduced + deferred reduce must match the direct software product.
            let un = unsafe { x86_64::ghash_mul_unreduced_x86(a, b) }.reduce();
            assert_eq!(sw, sb, "schoolbook");
            assert_eq!(sw, ka, "karatsuba");
            assert_eq!(sw, kb, "karatsuba_barrett");
            assert_eq!(sw, bi, "binius");
            assert_eq!(sw, un, "unreduced");
        }
    }

    /// The 4-lane VPCLMULQDQ multiply must agree, lane for lane, with the
    /// canonical scalar `Gf128::mul` — the clmul `0x87` reduction reaches the
    /// same field element by a different route, so verify, don't assume.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "pclmulqdq",
    target_feature = "sse4.1",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn ghash_mul_x4_matches_scalar() {
        use core::arch::x86_64::*;
        let mut rng = Rng::new(0x4A4_C0DE);
        for _ in 0..256 {
            let xs = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            let ys = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            // SAFETY: vpclmulqdq+avx512f enabled at compile time (cfg gate).
            let got: [Gf128; 4] = unsafe {
                let x = _mm512_loadu_si512(xs.as_ptr() as *const __m512i);
                let y = _mm512_loadu_si512(ys.as_ptr() as *const __m512i);
                let r = x86_64::ghash_mul_x4(x, y);
                let mut out = [Gf128::ZERO; 4];
                _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, r);
                out
            };
            for lane in 0..4 {
                assert_eq!(
                    got[lane],
                    xs[lane] * ys[lane],
                    "lane {lane}: x4 != scalar mul"
                );
            }
        }
    }

    /// The 4-lane deferred-reduction accumulator must equal the scalar
    /// XOR-of-`mul_unreduced` it replaces, both before and after `reduce()`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "pclmulqdq",
    target_feature = "sse4.1",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn wide_ghash_x4_matches_scalar_deferred() {
        let mut rng = Rng::new(0xDEF_E44);
        for _ in 0..128 {
            // SAFETY: vpclmulqdq+avx512f+sse4.1 enabled at compile time.
            let mut wide = unsafe { x86_64::WideGhashX4::zero() };
            let mut scalar = Gf128Product::zero();
            for _ in 0..5 {
                let xs = [
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                ];
                let ys = [
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                ];
                // xs via contiguous load, ys via scalar set — exercises both.
                unsafe {
                    let xv = x86_64::f128x4_loadu(xs.as_ptr());
                    let yv = x86_64::f128x4_set(ys[0], ys[1], ys[2], ys[3]);
                    wide.mul_acc(xv, yv);
                }
                for i in 0..4 {
                    scalar ^= xs[i].mul_unreduced(ys[i]);
                }
            }
            let folded = unsafe { wide.fold() };
            assert_eq!(folded, scalar, "wide fold != scalar deferred accumulator");
            assert_eq!(folded.reduce(), scalar.reduce(), "reduced values differ");
        }
    }

    #[test]
    fn square_matches_self_mul() {
        let mut rng = Rng::new(0x5147);
        for _ in 0..1000 {
            let a = rng.next_f128();
            assert_eq!(a.square(), a * a);
        }
        assert_eq!(Gf128::ZERO.square(), Gf128::ZERO);
        assert_eq!(Gf128::ONE.square(), Gf128::ONE);
    }
}
