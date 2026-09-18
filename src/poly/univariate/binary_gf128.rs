//! `GF(2^128)`: degree-128 binary extension field.
//!
//! Elements are `F_2[X] / <f(X)>` where `f(X)` is the GHASH / AES-GCM
//! reduction polynomial
//!
//! ```text
//! f(X) = X^128 + X^7 + X^2 + X + 1.
//! ```
//!
//! `f` is irreducible over `F_2`. The factor ring is therefore a field
//! of order `2^128`. Its multiplicative group has order `2^128 - 1`,
//! so for every nonzero `a` we have `a^{2^128 - 1} = 1` and
//! `a^{-1} = a^{2^128 - 2}`.
//!
//! Storage layout: each element is a [`Uint<2>`] (2 × `u64` = 128 bits)
//! bit-packed polynomial of degree `< 128`. Bit `64*w + b` (LSB-first
//! within each `u64` limb) holds the coefficient of `X^{64*w + b}`.
//! Addition is XOR; multiplication is the F_2 carryless product
//! followed by reduction modulo `f`.
//!
//! Intended use: the random "projecting element" `α` in the F_2 proving
//! path. After the ideal check runs over `F_2[X]`, the protocol samples
//! `α ∈ GF(2^128)` and substitutes `X = α` in every committed cell, so
//! that the sumcheck-based phase runs over `GF(2^128)` instead of a
//! prime field. The 128-bit choice (vs the prior 192-bit field) cuts
//! the inner Karatsuba mul-count from 6 to 3 PMULL/PCLMUL ops per
//! field multiplication and the canonical-representative width from
//! `Uint<3>` to `Uint<2>`.

use crate::poly::coefficient::{
    Coefficient, FieldRepresentation, PolynomialField, SignedCoefficient,
};
use crate::utils::inner_transparent_field::InnerTransparentField;
use field::Uint;
#[cfg(test)]
use num_traits::{One, Zero};

use crate::poly::univariate::dense::DensePolynomial;

/// A `GF(2^128)[X]<D>` polynomial — degree-`<D` univariate with
/// `GF(2^128)`-valued coefficients. The natural target of the
/// `F_2[X] → GF(2^128)[X]` coefficient lift used in step 2 of the
/// F_2 proving path (see `protocol/src/f2_prove_plan.md`).
///
/// Built on top of the existing [`DensePolynomial`] machinery:
/// `Gf128` already implements `Coefficient` (via the
/// degenerate `PolynomialField` impl), so addition / negation /
/// `EvaluatablePolynomial<R, R>` (Horner at a point) all come
/// for free.
pub type GF128Poly<const D: usize> = DensePolynomial<Gf128, D>;

/// Low bits of the reduction polynomial — `g(X) = X^7 + X^2 + X + 1`,
/// stored as `0x87` (bits 0, 1, 2, 7 set). The `X^128` term is implicit
/// in the reduction routine: `X^128 ≡ g(X) mod f`.
pub const REDUCTION_LOW_GF128: u64 = 0x87;

/// `FieldRepresentation::Modulus`-shaped representation of the reduction polynomial:
/// the same low-bit pattern as [`REDUCTION_LOW_GF128`], promoted to a
/// `Uint<2>` so the byte width matches `FieldRepresentation::Inner` (a requirement
/// of `Transcript::absorb_random_field`).
pub const MODULUS_LOW_BITS_GF128: Uint<2> = Uint::<2>::from_words([REDUCTION_LOW_GF128, 0]);

pub use field::Gf128;
impl Coefficient for Gf128 {}
impl SignedCoefficient for Gf128 {}
impl FieldRepresentation for Gf128 {
    type Inner = Uint<2>;
    type Modulus = Uint<2>;
    fn inner(&self) -> &Self::Inner {
        self.as_integer()
    }
    fn set_inner(&mut self, value: Self::Inner) {
        *self = Self::from_polynomial_words(*value.as_words());
    }
    fn into_inner(self) -> Self::Inner {
        *self.as_integer()
    }
}
impl PolynomialField for Gf128 {
    type Config = ();
    fn cfg(&self) -> &() {
        &()
    }
    fn config_from_modulus(value: &Self::Modulus) -> Option<()> {
        (*value == MODULUS_LOW_BITS_GF128).then_some(())
    }
    fn is_zero(value: &Self) -> bool {
        value.as_words() == &[0, 0]
    }
    fn modulus(&self) -> Self::Modulus {
        MODULUS_LOW_BITS_GF128
    }
    fn new_with_cfg(value: Self::Inner, _: &()) -> Self {
        Self::from_polynomial_words(*value.as_words())
    }
    fn new_unchecked_with_cfg(value: Self::Inner, _: &()) -> Self {
        Self::from_polynomial_words(*value.as_words())
    }
    fn zero_with_cfg(_: &()) -> Self {
        Self::ZERO
    }
    fn one_with_cfg(_: &()) -> Self {
        Self::ONE
    }
    fn interpolation_node(index: u64, _: &()) -> Self {
        Self::from_polynomial_words([index, 0])
    }
}
impl crate::transcript::traits::GenTranscribable for Gf128 {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        field::CanonicalCodec::decode_public(&field::Gf128Ops, bytes)
            .expect("canonical binary-field encoding")
    }
    fn write_transcription_bytes_exact(&self, out: &mut [u8]) {
        field::CanonicalCodec::encode_into(&field::Gf128Ops, self, out);
    }
}
impl crate::transcript::traits::ConstTranscribable for Gf128 {
    const NUM_BYTES: usize = 16;
}

impl InnerTransparentField for Gf128 {
    #[inline]
    fn add_inner(lhs: &Self::Inner, rhs: &Self::Inner, _config: &Self::Config) -> Self::Inner {
        *lhs ^ *rhs
    }

    #[inline]
    fn sub_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner {
        // Characteristic 2: subtraction is XOR, same as addition.
        Self::add_inner(lhs, rhs, config)
    }

    #[inline]
    fn mul_assign_by_inner(&mut self, rhs: &Self::Inner) {
        // The inner representation is just the field element; lift to
        // `Gf128` and reuse the regular `MulAssign<&Self>`.
        let r = Self::from_polynomial_words(*rhs.as_words());
        *self *= &r;
    }

    #[inline]
    fn mul_by_node2(&self, node2: &Self) -> Self {
        // from(2) = X here, so the node-2 multiply is the cheap shift+reduce.
        debug_assert_eq!(
            node2,
            &Self::from_polynomial_words([2, 0]),
            "node2 must be the field generator X = from(2)"
        );
        self.mul_x()
    }
}

/// Delayed-reduction accumulate: 256-bit unreduced carryless products,
/// XOR-combined, reduced once per accumulator (see `crate::utils::wide_mul`).
/// On aarch64 the accumulator is a NEON vector pair ([`field::Gf128Product`]); the
/// word-array form remains on every other target. Same field values
/// either way (the trait laws).
impl crate::utils::wide_mul::WideMulAcc for Gf128 {
    fn eqf_inverse(&self) -> Option<Self> {
        (self.as_words() != &[0, 0]).then(|| self.inverse_or_zero())
    }

    type Wide = field::Gf128Product;
    #[inline(always)]
    fn wide_zero(_: &Self) -> Self::Wide {
        field::Gf128Product::zero()
    }
    #[inline(always)]
    fn wide_of(x: &Self) -> Self::Wide {
        field::Gf128Product::from_element(*x)
    }
    #[inline(always)]
    fn mul_wide(a: &Self, b: &Self) -> Self::Wide {
        field::WideMul::<field::Gf128>::mul_wide(&field::Gf128Ops, a, b)
    }
    #[inline(always)]
    fn wide_add_assign(acc: &mut Self::Wide, x: &Self::Wide) {
        *acc ^= *x;
    }
    #[inline(always)]
    fn wide_sub_assign(acc: &mut Self::Wide, x: &Self::Wide) {
        *acc ^= *x;
    }
    #[inline(always)]
    fn from_wide(value: Self::Wide) -> Self {
        value.reduce()
    }

    #[inline(always)]
    fn add_assign_masked(acc: &mut Self, x: &Self, mask: bool) {
        // Branchless select: XOR in `x` under an all-ones/all-zeros mask —
        // add in char 2, immune to the coin-flip mispredicts a data-bit
        // branch would cost.
        *acc += field::CtSelect::ct_select(&Self::ZERO, x, field::CtMask::from_lsb(mask as u64));
    }

    fn eqf_single_pair_round(
        l: &[Self],
        r: &[Self],
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        let [a, b, c] =
            field::SumcheckKernels::eqf_single_pair_round(&field::Gf128Ops, (l), (r), (w), half);
        Some((a, b, c))
    }
    fn eqf_two_pair_round(
        l0: &[Self],
        r0: &[Self],
        l1: &[Self],
        r1: &[Self],
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        let [a, b, c] = field::SumcheckKernels::eqf_two_pair_round(
            &field::Gf128Ops,
            (l0),
            (r0),
            (l1),
            (r1),
            (w),
            half,
        );
        Some((a, b, c))
    }
    fn eqf_fold_in_place(v: &mut [Self], rho: &Self, half: usize) -> bool {
        field::SumcheckKernels::eqf_fold_in_place(&field::Gf128Ops, (v), rho, half);
        true
    }
    fn eqf_fused_fold_round(
        l: &mut [Self],
        r: &mut [Self],
        rho: &Self,
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        let [a, b, c] = field::SumcheckKernels::eqf_fused_fold_round(
            &field::Gf128Ops,
            (l),
            (r),
            rho,
            (w),
            half,
        );
        Some((a, b, c))
    }
    fn eqf_grid_pass(
        l: &mut [Self],
        r: &mut [Self],
        p: &[Self],
        s: &[Self],
        n: usize,
    ) -> Option<[Self; 9]> {
        Some(field::SumcheckKernels::eqf_grid_pass(
            &field::Gf128Ops,
            (l),
            (r),
            (p),
            (s),
            n,
        ))
    }
}

pub(crate) use field::gf128::kernels::clmul_128x128;
#[cfg(test)]
use field::gf128::kernels::software::clmul_64x64 as clmul_64x64_scalar;

impl<const D: usize> crate::poly::univariate::F2AddAssign for GF128Poly<D> {
    #[allow(clippy::arithmetic_side_effects)]
    fn f2_add_assign(&mut self, rhs: &Self) {
        for (a, b) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *a = *a + *b;
        }
    }
}

/// Identity lift `GF(2^128) → GF(2^128)`. Completes the `FromRef` chain a
/// generic `F_2` linear-code encoder kernel needs for `GF128Poly<D>` (via
/// `FromRef<DensePolynomial<S,D>> for DensePolynomial<R,D>` with `R = S`).
impl crate::utils::from_ref::FromRef<Gf128> for Gf128 {
    #[inline(always)]
    fn from_ref(value: &Gf128) -> Self {
        *value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::wide_mul::WideMulAcc;
    use field::gf128::kernels::{clmul_64x64, reduce_256_to_128};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn rand_elt(rng: &mut StdRng) -> Gf128 {
        Gf128::from_polynomial_words([rng.random(), rng.random()])
    }

    /// The fused deferred-fold + round kernel is VALUE-EXACT vs its
    /// two-pass contract — `eqf_fold_in_place` then `eqf_single_pair_round`
    /// — on both the returned coefficients and the folded buffer prefix
    /// (odd `half` exercises the single-slot tail).
    #[test]
    fn eqf_fused_fold_round_matches_fold_then_round() {
        let mut rng = StdRng::seed_from_u64(0xF05E);
        for half in [1usize, 2, 37, 64] {
            let n = half << 2;
            let l0: Vec<_> = (0..n).map(|_| rand_elt(&mut rng)).collect();
            let r0: Vec<_> = (0..n).map(|_| rand_elt(&mut rng)).collect();
            let w: Vec<_> = (0..half).map(|_| rand_elt(&mut rng)).collect();
            let rho = rand_elt(&mut rng);

            // Two-pass reference: the pinned fold kernel, then the pinned
            // single-pair round kernel over the folded prefixes.
            let mut l_ref = l0.clone();
            let mut r_ref = r0.clone();
            assert!(Gf128::eqf_fold_in_place(&mut l_ref, &rho, n >> 1));
            assert!(Gf128::eqf_fold_in_place(&mut r_ref, &rho, n >> 1));
            l_ref.truncate(n >> 1);
            r_ref.truncate(n >> 1);
            let expect =
                Gf128::eqf_single_pair_round(&l_ref, &r_ref, &w, half).expect("single-pair kernel");

            let mut l_fused = l0.clone();
            let mut r_fused = r0.clone();
            let got = Gf128::eqf_fused_fold_round(&mut l_fused, &mut r_fused, &rho, &w, half)
                .expect("fused kernel");
            assert_eq!(got, expect, "coefficients (half = {half})");
            assert_eq!(
                &l_fused[..n >> 1],
                &l_ref[..],
                "folded L prefix (half = {half})"
            );
            assert_eq!(
                &r_fused[..n >> 1],
                &r_ref[..],
                "folded R prefix (half = {half})"
            );
        }
    }

    /// The fixed-scalar (preprocessed-ρ) kernels are BIT-IDENTICAL to the
    /// composed kernels they replace: the in-place fold and the fused
    /// fold+round pass, across degenerate and random challenges (and, via
    /// the fold identity `fold(0, a) = ρ·a`, the reduced fixed multiply
    /// itself against the general multiply).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn fixed_gf_mul_matches_composed() {
        let mut rng = StdRng::seed_from_u64(0xF16E);
        let mut scalars = vec![
            Gf128::zero(),
            Gf128::one(),
            Gf128::from_polynomial_words([u64::MAX, u64::MAX]),
        ];
        // Monomials (the dual-basis columns' shape) and randoms.
        for k in [1usize, 6, 63, 64, 65, 121, 127] {
            let mut w = [0u64; 2];
            w[k >> 6] = 1u64 << (k & 63);
            scalars.push(Gf128::from_polynomial_words(w));
        }
        for _ in 0..8 {
            scalars.push(rand_elt(&mut rng));
        }
        for s in scalars {
            let f = field::PreparedGf128Mul::new(s.into());
            for _ in 0..32 {
                let a = rand_elt(&mut rng);
                assert_eq!(
                    Gf128::from(f.mul(&a.into())),
                    s * a,
                    "scalar {:?}",
                    s.as_words()
                );
            }
            assert_eq!(Gf128::from(f.mul(&field::Gf128::ZERO)), Gf128::zero());
            assert_eq!(Gf128::from(f.mul(&field::Gf128::ONE)), s);
        }
    }

    /// The NEON-resident multiply/square pipeline is BIT-IDENTICAL to the
    /// scalar Karatsuba + shift/XOR-fold pipeline (which stays compiled on
    /// every target): dense edge cases + random pairs.
    #[test]
    fn neon_mul_matches_scalar_pipeline() {
        let mut rng = StdRng::seed_from_u64(0x6EA5);
        let mut cases: Vec<[u64; 2]> = vec![
            [0, 0],
            [1, 0],
            [0, 1],
            [u64::MAX, u64::MAX],
            [0x87, 0],
            [0, 0x8000_0000_0000_0000],
            [u64::MAX, 0],
            [0, u64::MAX],
        ];
        for _ in 0..2000 {
            cases.push([rng.random(), rng.random()]);
        }
        for (i, a) in cases.iter().enumerate() {
            let b = &cases[(i * 7 + 3) % cases.len()];
            let scalar = reduce_256_to_128(clmul_128x128(a, b));
            assert_eq!(
                *(Gf128::from_polynomial_words(*a) * Gf128::from_polynomial_words(*b)).as_words(),
                scalar,
                "mul mismatch at case {i}"
            );
            // Square: the dedicated 2-PMULL path == the general multiply
            // of the element with itself.
            let sq = Gf128::from_polynomial_words(*a).square();
            let sq_ref = Gf128::from_polynomial_words(reduce_256_to_128(clmul_128x128(a, a)));
            assert_eq!(sq, sq_ref, "square mismatch at case {i}");
        }
    }

    /// GHASH-field mul micro-baseline — the three shapes the prover cares
    /// about: LATENCY (one dependent chain — the pow/base chains),
    /// THROUGHPUT (8 independent chains — ILP ceiling), and the KERNEL
    /// shape (the eqf round body's 5-mul slot with wide accumulation).
    /// Run:
    ///   RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=<wt>/target-local \
    ///   cargo test --offline -p zinc-poly --features "parallel simd" --release \
    ///   --lib binary_gf128::tests::ghash_mul_baseline -- --ignored --nocapture --exact
    #[test]
    #[ignore = "GHASH-field mul micro-baseline — measurement"]
    #[allow(clippy::arithmetic_side_effects, clippy::cast_precision_loss)]
    fn ghash_mul_baseline() {
        use std::time::Instant;
        let mut rng = StdRng::seed_from_u64(0xF1E1D);
        const N: usize = 1 << 22;

        // LATENCY: one dependent chain x <- x*y.
        let y = rand_elt(&mut rng);
        let mut x = rand_elt(&mut rng);
        let t0 = Instant::now();
        for _ in 0..N {
            x = x * &y;
        }
        let lat = t0.elapsed().as_secs_f64() * 1e9 / N as f64;
        std::hint::black_box(x);

        // THROUGHPUT: 8 independent chains.
        let mut xs: Vec<Gf128> = (0..8).map(|_| rand_elt(&mut rng)).collect();
        let ys: Vec<Gf128> = (0..8).map(|_| rand_elt(&mut rng)).collect();
        let t0 = Instant::now();
        for _ in 0..(N / 8) {
            for k in 0..8 {
                xs[k] = xs[k] * &ys[k];
            }
        }
        let thr = t0.elapsed().as_secs_f64() * 1e9 / N as f64;
        std::hint::black_box(&xs);

        // KERNEL shape: the fused single-pair round body over arrays
        // (5 muls per slot: 2 reduced weight-folds + 3 wide products).
        let m = 1usize << 16;
        let l: Vec<Gf128> = (0..m).map(|_| rand_elt(&mut rng)).collect();
        let r: Vec<Gf128> = (0..m).map(|_| rand_elt(&mut rng)).collect();
        let w: Vec<Gf128> = (0..m / 2).map(|_| rand_elt(&mut rng)).collect();
        let reps = 64usize;
        let t0 = Instant::now();
        let mut acc = (Gf128::zero(), Gf128::zero(), Gf128::zero());
        for _ in 0..reps {
            let out = <Gf128 as crate::utils::wide_mul::WideMulAcc>::eqf_single_pair_round(
                &l,
                &r,
                &w,
                m / 2,
            )
            .expect("gf128 has the fused kernel");
            acc = out;
        }
        let slots = (m / 2) * reps;
        let kern = t0.elapsed().as_secs_f64() * 1e9 / (slots * 5) as f64;
        std::hint::black_box(acc);

        eprintln!("\nGHASH GF(2^128) mul baseline (N=2^22):");
        eprintln!("  latency (dependent chain) : {lat:>6.2} ns/mul");
        eprintln!("  throughput (8 indep chains): {thr:>6.2} ns/mul");
        eprintln!("  kernel slot rate           : {kern:>6.2} ns/mul-equivalent (5/slot)");
    }

    fn gf(lo: u64, hi: u64) -> Gf128 {
        Gf128::from_polynomial_words([lo, hi])
    }

    #[test]
    fn zero_and_one_are_identities() {
        let z = Gf128::zero();
        let o = Gf128::one();
        let a = gf(0xDEAD_BEEF_CAFE_F00D, 0xA5A5_A5A5_A5A5_A5A5);
        assert_eq!(a + z, a);
        assert_eq!(a * o, a);
        assert_eq!(a * z, z);
    }

    #[test]
    fn add_is_xor_and_self_inverse() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            assert_eq!((a + b) + b, a); // char 2: x + x = 0
            assert_eq!(a + Gf128::zero(), a);
        }
    }

    #[test]
    fn one_is_mul_identity_and_zero_annihilates() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_128);
        let one = Gf128::one();
        let zero = Gf128::zero();
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            assert_eq!(a * one, a);
            assert_eq!(a * zero, zero);
        }
    }

    #[test]
    fn square_matches_self_multiply() {
        let mut rng = StdRng::seed_from_u64(0xBEEF_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        let a = gf(0xDEAD_BEEF, 0xCAFE_F00D);
        let b = gf(0x9E37_79B1_DEAD_BEEF, 0x1234_5678);
        let c = gf(0xA5A5_5A5A_F00D_BAAD, 0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(a * b, b * a);
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn distributivity_holds() {
        let a = gf(0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A);
        let b = gf(0xDEAD_BEEF, 0xCAFE_F00D);
        let c = gf(0x1234_5678, 0x9ABC_DEF0);
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn inverse_satisfies_a_times_inv_equals_one() {
        let mut rng = StdRng::seed_from_u64(0xF00D_128);
        let one = Gf128::one();
        for _ in 0..64 {
            let mut a = rand_elt(&mut rng);
            if a.is_zero() {
                a = Gf128::one();
            }
            let inv = a.invert_nonzero();
            assert_eq!(a * inv, one);
        }
    }

    /// The Itoh–Tsujii inverse equals the naive Fermat all-ones ladder
    /// (the pre-optimisation implementation, inlined here as the
    /// reference), and `square_n` equals repeated squaring.
    #[test]
    fn itoh_tsujii_matches_fermat_ladder() {
        let mut rng = StdRng::seed_from_u64(0xF00D_129);
        for _ in 0..200 {
            let mut a = rand_elt(&mut rng);
            if a.is_zero() {
                a = Gf128::one();
            }
            let fermat = {
                let mut c = a;
                for _ in 1..127 {
                    c = c.square();
                    c *= &a;
                }
                c.square()
            };
            assert_eq!(a.invert_nonzero(), fermat, "IT vs Fermat");
            let mut s = a;
            for k in 0..9 {
                assert_eq!(a.square_n(k), s, "square_n({k})");
                s = s.square();
            }
        }
    }

    #[test]
    fn frobenius_squaring_is_linear() {
        // In characteristic 2 the Frobenius `x → x²` is additive:
        // `(x + y)² = x² + y²`.
        let mut rng = StdRng::seed_from_u64(0xBAAD_128);
        for _ in 0..256 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            assert_eq!((a + b).square(), a.square() + b.square());
        }
    }

    #[test]
    fn karatsuba_clmul_matches_scalar_clmul() {
        // The Karatsuba `clmul_128x128` composes three 64×64 clmuls;
        // cross-check against a naive 4-mul layout via the scalar path,
        // so the Karatsuba bookkeeping (the XOR mixes) is right.
        let mut rng = StdRng::seed_from_u64(0xABCD_128);
        for _ in 0..64 {
            let a: [u64; 2] = [rng.random(), rng.random()];
            let b: [u64; 2] = [rng.random(), rng.random()];
            let karatsuba = clmul_128x128(&a, &b);

            let m00 = clmul_64x64_scalar(a[0], b[0]);
            let m01 = clmul_64x64_scalar(a[0], b[1]);
            let m10 = clmul_64x64_scalar(a[1], b[0]);
            let m11 = clmul_64x64_scalar(a[1], b[1]);
            let naive = [
                m00[0],
                m00[1] ^ m01[0] ^ m10[0],
                m11[0] ^ m01[1] ^ m10[1],
                m11[1],
            ];
            assert_eq!(karatsuba, naive);
        }
    }

    /// Frobenius: `a^{2^128} = a` for every `a ∈ GF(2^128)`. Computing
    /// 128 squarings is cheap.
    #[test]
    fn frobenius_equals_2_pow_128() {
        let cases = [
            gf(0xDEAD_BEEF, 0xCAFE_F00D),
            gf(0x9E37_79B1_DEAD_BEEF, 0x1234_5678),
            Gf128::one(),
        ];
        for a in cases {
            let mut x = a;
            for _ in 0..128 {
                x = x.square();
            }
            assert_eq!(
                x, a,
                "Frobenius failed: a^{{2^128}} should be a; got {x} for a = {a}"
            );
        }
    }

    #[test]
    fn mul_x_matches_general_multiply_by_generator() {
        let x = Gf128::from_polynomial_words([2, 0]); // the generator X = from(2)
        // Fixed edge cases: top bit set (forces the 0x87 reduction), one, zero.
        let edge = [
            gf(0, 0),
            gf(1, 0),
            gf(0, 1u64 << 63), // X^127  → X^128 ≡ 0x87
            gf(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF),
            gf(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
        ];
        for a in edge {
            assert_eq!(a.mul_x(), a * &x, "mul_x must equal a·X for {a}");
        }
        // Randomised cross-check.
        let mut rng = StdRng::seed_from_u64(0x6D75_6C78); // "mulx"
        for _ in 0..10_000 {
            let a = rand_elt(&mut rng);
            assert_eq!(a.mul_x(), a * &x);
        }
        // And the trait hook the sumcheck calls dispatches to it.
        use crate::utils::inner_transparent_field::InnerTransparentField;
        let a = gf(0xDEAD_BEEF, 0xC0FF_EE00);
        assert_eq!(a.mul_by_node2(&x), a * &x);
    }

    #[test]
    fn implements_field_and_prime_field_traits() {
        fn assert_field<F: crate::poly::coefficient::FieldRepresentation>() {}
        fn assert_prime_field<F: crate::poly::coefficient::PolynomialField>() {}
        assert_field::<Gf128>();
        assert_prime_field::<Gf128>();
    }

    #[test]
    fn cfg_keyed_constructors_match_const_constructors() {
        use crate::poly::coefficient::PolynomialField;

        let cfg = ();
        assert_eq!(
            <Gf128 as PolynomialField>::zero_with_cfg(&cfg),
            Gf128::zero(),
        );
        assert_eq!(<Gf128 as PolynomialField>::one_with_cfg(&cfg), Gf128::one(),);
        let words = [0xDEAD_BEEFu64, 0xCAFEu64];
        let v: Gf128 = <Gf128 as PolynomialField>::new_with_cfg(Uint::<2>::from_words(words), &cfg);
        assert_eq!(*v.as_words(), words);
    }

    #[test]
    fn division_by_self_yields_one() {
        let a = gf(0xDEAD_BEEF, 0xCAFE_F00D);
        let b = a / a;
        assert!(b.is_one(), "a / a should be 1; got {b}");
    }
}
