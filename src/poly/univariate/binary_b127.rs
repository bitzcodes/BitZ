//! `GF(2^127)` ("b127"): degree-127 binary extension field with a
//! trinomial modulus.
//!
//! Elements are `F_2[X] / <f(X)>` where
//!
//! ```text
//! f(X) = X^127 + X + 1.
//! ```
//!
//! `f` is irreducible over `F_2` (a classical trinomial; cf. Zierler's
//! tables), so the factor ring is a field of order `2^127`. Two structural
//! facts distinguish it from the GHASH field ([`Gf128`]) and are
//! load-bearing for the exponent-fold PCS:
//!
//! * **The multiplicative group has PRIME order.** `|B^×| = 2^127 − 1` is
//!   the Mersenne prime `M_127` (Lucas, 1876). Every element outside
//!   `{0, 1}` therefore generates the full group — the generator check of
//!   the exponent binding collapses to `α ∉ {0, 1}` ([`is_generator_b127`]),
//!   with no order factorization and no resampling loop (contrast
//!   `GF(2^128)`, where only ≈49 % of `K^×` generates). The exponent map
//!   `n ↦ α^n` is injective on `[0, 2^127 − 1)`.
//! * **127 is prime**, so the only proper subfield is `F_2 = {0, 1}` —
//!   there are no intermediate towers (no `GF(2^8)`-style acceleration, and
//!   no `2^κ`-bit ring-switch packing: `[B : F_2] = 127` is not a power of
//!   two, which is exactly why this field cannot ride the flock Ligerito
//!   opener; see `docs/DESIGN.md`).
//!
//! # The speed trade (and how it measures)
//!
//! The reduction modulo the trinomial is two shifted XORs: writing a
//! carryless product `P = L + X^127·H` (`deg P ≤ 252` for canonical
//! operands, so `deg H ≤ 125`),
//!
//! ```text
//! P ≡ L ⊕ H ⊕ (H << 1)      (mod X^127 + X + 1),
//! ```
//!
//! and `deg((X+1)·H) ≤ 126 < 127` — ONE fold, exact, no carry chain. The
//! GHASH reduction by contrast costs a 3-PMULL fold. On the NEON pipeline
//! that turns a 7-PMULL reduced multiply into a 4-PMULL one (schoolbook
//! product + PMULL-free reduction), and the 5-PMULL GHASH squaring into
//! 2 PMULLs — trading PMULLs for shift/logical µops. **Measured verdict
//! (Apple M4, interleaved-rep harness): the trade LOSES there** — PMULL
//! throughput is abundant enough that b127 lands at 0.88–1.02× of the
//! GHASH pipeline across this repo's hot patterns (winning only the
//! squaring chain, 1.02×). It is the right trade on cores where carryless
//! multiply is port-constrained. Full study: `docs/b127-field.md`.
//!
//! Storage layout: each element is a [`Uint<2>`] (2 × `u64`) bit-packed
//! polynomial of degree `< 127`; bit `64·w + b` (LSB-first per limb) holds
//! the coefficient of `X^{64·w + b}`. **Canonical invariant: bit 127 (word
//! 1, bit 63) is zero.** Constructors that accept raw 128-bit patterns
//! either fold the top bit (`X^127 ≡ X + 1`: [`From<u128>`],
//! [`PolynomialField::new_with_cfg`] — a 2-to-1 covering, so uniform 128-bit
//! transcript draws yield uniform field elements) or reject it
//! ([`Self::try_from_words`], the codec-canonicality entry point).
//!
//! Algorithm provenance: the element representation, the Karatsuba product
//! and the two-XOR trinomial fold follow Reilabs'
//! [`ghash-powers-bench`](https://github.com/reilabs/ghash-powers-bench)
//! (`b127`), restructured into this repo's NEON-resident idiom (the
//! products, the vector-register wide accumulators and the fused eq-factored
//! kernels are shared with / mirrored from [`binary_gf128`]; only the
//! reduction differs).
//!
//! [`Gf128`]: crate::poly::univariate::binary_gf128::Gf128
//! [`binary_gf128`]: crate::poly::univariate::binary_gf128

use crate::poly::coefficient::{
    Coefficient, FieldRepresentation, PolynomialField, SignedCoefficient,
};
use crate::utils::inner_transparent_field::InnerTransparentField;
use field::Uint;
#[cfg(test)]
use num_traits::{One, Zero};

// Only the scalar (non-NEON) squaring pipeline calls the 64×64 base mul —
// and the NEON↔scalar parity tests, which exercise it explicitly.
#[cfg(test)]
use field::gf128::kernels::clmul_64x64;

/// Low bits of the reduction polynomial — `g(X) = X + 1`, stored as `0x3`
/// (bits 0 and 1 set). The `X^127` term is implicit in the reduction
/// routines: `X^127 ≡ g(X) mod f`.
pub const REDUCTION_LOW_B127: u64 = 0x3;

/// Word-1 mask clearing bit 127 (the canonical-invariant bit): a reduced
/// element satisfies `words()[1] & !MASK_HI_B127 == 0`.
pub const MASK_HI_B127: u64 = u64::MAX >> 1;

/// `FieldRepresentation::Modulus`-shaped representation of the reduction polynomial.
/// Unlike GHASH's (where the `X^128` term cannot be stored in 128 bits and
/// is left implicit), `f(X) = X^127 + X + 1` fits: bit 127 + bits 1, 0.
pub const B127_MODULUS: Uint<2> = Uint::<2>::from_words([REDUCTION_LOW_B127, 1u64 << 63]);

/// The order of the multiplicative group `B^× = GF(2^127)^×`:
/// `2^127 − 1 = 170141183460469231731687303715884105727`, the Mersenne
/// prime `M_127` (Lucas, 1876; the largest prime found by hand). Prime
/// group order makes EVERY non-identity element a generator, so the
/// exponent map `n ↦ α^n` is injective on `[0, 2^127 − 1)` for any
/// `α ∉ {0, 1}` — the b127 replacement for `GF128_MULT_ORDER` + the
/// 9-factor primitive-element test.
pub const B127_MULT_ORDER: u128 = u128::MAX >> 1;

/// Does `α` generate `B^×`? Prime group order (`M_127`, see
/// [`B127_MULT_ORDER`]) means the subgroup generated by any `α ∉ {0, 1}`
/// has order dividing a prime and exceeding 1 — i.e. the whole group. (And
/// `{0, 1}` is exactly the unique proper subfield `F_2`, since 127 is
/// prime.) A Fiat–Shamir `α` is thus a generator with probability
/// `1 − 2^{-126}`; no resampling loop, no order factorization.
#[inline]
pub fn is_generator_b127(alpha: &B127) -> bool {
    *alpha != B127::ZERO && *alpha != B127::ONE
}

pub use field::B127;
impl Coefficient for B127 {}
impl SignedCoefficient for B127 {}
impl FieldRepresentation for B127 {
    type Inner = Uint<2>;
    type Modulus = Uint<2>;
    fn inner(&self) -> &Self::Inner {
        self.as_integer()
    }
    fn set_inner(&mut self, value: Self::Inner) {
        *self = Self::from_canonical_words(*value.as_words());
    }
    fn into_inner(self) -> Self::Inner {
        *self.as_integer()
    }
}
impl PolynomialField for B127 {
    type Config = ();
    fn cfg(&self) -> &() {
        &()
    }
    fn config_from_modulus(value: &Self::Modulus) -> Option<()> {
        (*value == B127_MODULUS).then_some(())
    }
    fn is_zero(value: &Self) -> bool {
        value.as_words() == &[0, 0]
    }
    fn modulus(&self) -> Self::Modulus {
        B127_MODULUS
    }
    fn new_with_cfg(value: Self::Inner, _: &()) -> Self {
        Self::from_polynomial_words(*value.as_words())
    }
    fn new_unchecked_with_cfg(value: Self::Inner, _: &()) -> Self {
        Self::from_canonical_words(*value.as_words())
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
impl crate::transcript::traits::GenTranscribable for B127 {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        field::CanonicalCodec::decode_public(&field::B127Ops, bytes)
            .expect("canonical binary-field encoding")
    }
    fn write_transcription_bytes_exact(&self, out: &mut [u8]) {
        field::CanonicalCodec::encode_into(&field::B127Ops, self, out);
    }
}
impl crate::transcript::traits::ConstTranscribable for B127 {
    const NUM_BYTES: usize = 16;
    const NUM_BITS: usize = 127;
}

impl InnerTransparentField for B127 {
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
        // The inner representation is the field element (canonical by the
        // producers of `Inner` values in this crate); lift and reuse the
        // regular `MulAssign<&Self>`.
        debug_assert!(
            rhs.as_words()[1] >> 63 == 0,
            "GF(2^127): non-canonical inner (bit 127 set)"
        );
        let r = Self::from_canonical_words(*rhs.as_words());
        *self *= &r;
    }

    #[inline]
    fn mul_by_node2(&self, node2: &Self) -> Self {
        // from(2) = X here, so the node-2 multiply is the cheap shift+fold.
        debug_assert_eq!(
            node2,
            &Self::from_canonical_words([2, 0]),
            "node2 must be the field generator X = from(2)"
        );
        self.mul_x()
    }
}

/// Delayed-reduction accumulate: 256-bit unreduced carryless products,
/// XOR-combined, reduced once per accumulator (see `crate::utils::wide_mul`).
/// All trait-reachable wide values are XORs of products of canonical
/// (`deg ≤ 126`) operands, hence `deg ≤ 252` — inside the single-fold
/// reduction contract of [`reduce_256_to_127`].
impl crate::utils::wide_mul::WideMulAcc for B127 {
    type Wide = field::B127Product;
    #[inline(always)]
    fn wide_zero(_: &Self) -> Self::Wide {
        field::B127Product::zero()
    }
    #[inline(always)]
    fn wide_of(x: &Self) -> Self::Wide {
        field::B127Product::from_element(*x)
    }
    #[inline(always)]
    fn mul_wide(a: &Self, b: &Self) -> Self::Wide {
        field::WideMul::<field::B127>::mul_wide(&field::B127Ops, a, b)
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
            field::SumcheckKernels::eqf_single_pair_round(&field::B127Ops, (l), (r), (w), half);
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
            &field::B127Ops,
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
        field::SumcheckKernels::eqf_fold_in_place(&field::B127Ops, (v), rho, half);
        true
    }
    fn eqf_fused_fold_round(
        l: &mut [Self],
        r: &mut [Self],
        rho: &Self,
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        let [a, b, c] =
            field::SumcheckKernels::eqf_fused_fold_round(&field::B127Ops, (l), (r), rho, (w), half);
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
            &field::B127Ops,
            (l),
            (r),
            (p),
            (s),
            n,
        ))
    }
}

#[cfg(test)]
fn reduce_256_to_127(words: [u64; 4]) -> [u64; 2] {
    *field::B127::reduce_polynomial_words(words).as_words()
}
impl crate::utils::from_ref::FromRef<B127> for B127 {
    fn from_ref(value: &Self) -> Self {
        *value
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::super::binary_gf128::clmul_128x128;
    use super::*;
    use crate::utils::wide_mul::WideMulAcc;
    use rand::{SeedableRng, rand_core::Rng as _, rngs::StdRng};

    fn rand_elt(rng: &mut StdRng) -> B127 {
        B127::from_canonical_words([rng.next_u64(), rng.next_u64() & MASK_HI_B127])
    }

    /// Bit-by-bit shift-and-fold reference multiply modulo
    /// `X^127 + X + 1` (both operands canonical `< 2^127`). The
    /// correctness anchor for both pipelines.
    fn mul_reference(a: u128, b: u128) -> u128 {
        debug_assert!(a >> 127 == 0 && b >> 127 == 0);
        let mut acc = 0u128;
        let mut base = a;
        let mut m = b;
        while m != 0 {
            if m & 1 == 1 {
                acc ^= base;
            }
            m >>= 1;
            let carry = (base >> 126) & 1;
            base = (base << 1) & (u128::MAX >> 1);
            if carry == 1 {
                base ^= REDUCTION_LOW_B127 as u128; // X^127 ≡ X + 1
            }
        }
        acc
    }

    fn to_u128(x: &B127) -> u128 {
        let w = x.as_words();
        (w[0] as u128) | ((w[1] as u128) << 64)
    }

    #[test]
    fn pipeline_matches_bit_reference() {
        let mut rng = StdRng::seed_from_u64(0xB127_0001);
        for _ in 0..2_000 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            let prod = a * b;
            assert_eq!(to_u128(&prod), mul_reference(to_u128(&a), to_u128(&b)));
            // Canonical invariant preserved.
            assert_eq!(prod.as_words()[1] >> 63, 0);
        }
        // Edge cases: extremes of the canonical range.
        let top = B127::from_canonical_words([u64::MAX, MASK_HI_B127]);
        for (a, b) in [
            (B127::zero(), top),
            (B127::one(), top),
            (top, top),
            (B127::from_canonical_words([2, 0]), top),
        ] {
            assert_eq!(to_u128(&(a * b)), mul_reference(to_u128(&a), to_u128(&b)));
        }
    }

    /// The NEON pipeline (products + trinomial fold + wide accumulators)
    /// is bit-identical to the scalar Karatsuba + shift/XOR pipeline —
    /// the b127 analogue of GF128's `neon_mul_matches_scalar_pipeline`.
    #[test]
    fn neon_mul_matches_scalar_pipeline() {
        let mut rng = StdRng::seed_from_u64(0xB127_0002);
        for _ in 0..2_000 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            let scalar = reduce_256_to_127(clmul_128x128(a.as_words(), b.as_words()));
            assert_eq!(*(a * b).as_words(), scalar, "mul: NEON vs scalar");
            let sq_scalar = {
                let w = a.as_words();
                let lo = clmul_64x64(w[0], w[0]);
                let hi = clmul_64x64(w[1], w[1]);
                reduce_256_to_127([lo[0], lo[1], hi[0], hi[1]])
            };
            assert_eq!(*a.square().as_words(), sq_scalar, "square: NEON vs scalar");
            // Wide roundtrip: from_wide(mul_wide(a, b)) == a·b.
            let w = B127::mul_wide(&a, &b);
            assert_eq!(B127::from_wide(w), a * b, "wide roundtrip");
        }
        // The Karatsuba NEON product variant agrees with the default.
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            let mut rng = StdRng::seed_from_u64(0xB127_0003);
            for _ in 0..2_000 {
                let a = rand_elt(&mut rng);
                let b = rand_elt(&mut rng);
                assert_eq!(
                    *field::B127::from(a)
                        .mul_karatsuba(field::B127::from(b))
                        .as_words(),
                    *(a * b).as_words(),
                    "kara vs schoolbook"
                );
                assert_eq!(
                    *field::B127::from(a)
                        .mul_pfold(field::B127::from(b))
                        .as_words(),
                    *(a * b).as_words(),
                    "pfold vs schoolbook"
                );
            }
        }
    }

    #[test]
    fn zero_and_one_are_identities() {
        let mut rng = StdRng::seed_from_u64(0xB127_0004);
        for _ in 0..100 {
            let a = rand_elt(&mut rng);
            assert_eq!(a + B127::zero(), a);
            assert_eq!(a * B127::one(), a);
            assert_eq!(a * B127::zero(), B127::zero());
        }
    }

    #[test]
    fn add_is_xor_and_self_inverse() {
        let mut rng = StdRng::seed_from_u64(0xB127_0005);
        for _ in 0..100 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            let s = a + b;
            assert_eq!(s.as_words()[0], a.as_words()[0] ^ b.as_words()[0]);
            assert_eq!(s.as_words()[1], a.as_words()[1] ^ b.as_words()[1]);
            assert_eq!(a + a, B127::zero());
            assert_eq!(a - b, a + b);
            assert_eq!(-a, a);
        }
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        let mut rng = StdRng::seed_from_u64(0xB127_0006);
        for _ in 0..200 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            let c = rand_elt(&mut rng);
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
        }
    }

    #[test]
    fn distributivity_holds() {
        let mut rng = StdRng::seed_from_u64(0xB127_0007);
        for _ in 0..200 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            let c = rand_elt(&mut rng);
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn square_matches_self_multiply() {
        let mut rng = StdRng::seed_from_u64(0xB127_0008);
        for _ in 0..500 {
            let a = rand_elt(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn frobenius_squaring_is_linear() {
        let mut rng = StdRng::seed_from_u64(0xB127_0009);
        for _ in 0..200 {
            let a = rand_elt(&mut rng);
            let b = rand_elt(&mut rng);
            assert_eq!((a + b).square(), a.square() + b.square());
        }
    }

    /// `a^{2^127} = a` — 127 squarings return every element to itself.
    /// Combined with the (classical) primality of `2^127 − 1`, this is the
    /// full order certificate: `ord(a) | M_127` prime ⇒ every `a ∉ {0, 1}`
    /// has order exactly `M_127` — the [`is_generator_b127`] soundness fact.
    #[test]
    fn frobenius_equals_2_pow_127() {
        let mut rng = StdRng::seed_from_u64(0xB127_000A);
        for _ in 0..25 {
            let a = rand_elt(&mut rng);
            let mut x = a;
            for _ in 0..127 {
                x = x.square();
            }
            assert_eq!(x, a);
        }
        assert_eq!(B127_MULT_ORDER, (1u128 << 127) - 1);
    }

    #[test]
    fn inverse_satisfies_a_times_inv_equals_one() {
        let mut rng = StdRng::seed_from_u64(0xB127_000B);
        for _ in 0..50 {
            let a = rand_elt(&mut rng);
            if a.is_zero() {
                continue;
            }
            assert_eq!(a * a.invert_nonzero(), B127::one());
        }
        assert_eq!(B127::one().invert_nonzero(), B127::one());
    }

    /// The Itoh–Tsujii inverse equals the naive Fermat all-ones ladder
    /// (the pre-optimisation implementation, inlined here as the
    /// reference), and `square_n` equals repeated squaring.
    #[test]
    fn itoh_tsujii_matches_fermat_ladder() {
        let mut rng = StdRng::seed_from_u64(0xB127_000E);
        for _ in 0..200 {
            let mut a = rand_elt(&mut rng);
            if a.is_zero() {
                a = B127::one();
            }
            let fermat = {
                let mut c = a;
                for _ in 1..126 {
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
    fn division_by_self_yields_one() {
        let mut rng = StdRng::seed_from_u64(0xB127_000C);
        for _ in 0..25 {
            let a = rand_elt(&mut rng);
            if a.is_zero() {
                continue;
            }
            assert_eq!(a / a, B127::one());
        }
    }

    #[test]
    fn mul_x_matches_general_multiply_by_generator() {
        let mut rng = StdRng::seed_from_u64(0xB127_000D);
        let x = B127::from_canonical_words([2, 0]);
        for _ in 0..500 {
            let a = rand_elt(&mut rng);
            assert_eq!(a.mul_x(), a * x);
        }
        // And X^127 ≡ X + 1: 127 mul_x steps from 1 give 0x3.
        let mut p = B127::one();
        for _ in 0..127 {
            p = p.mul_x();
        }
        assert_eq!(p, B127::from_canonical_words([REDUCTION_LOW_B127, 0]));
    }

    /// `new_with_cfg` folds bit 127 exactly as `X^127 ≡ X + 1` demands:
    /// the fold of `v | 2^127` equals the fold of `v` plus `X + 1`.
    #[test]
    fn new_with_cfg_folds_bit_127() {
        let mut rng = StdRng::seed_from_u64(0xB127_000E);
        for _ in 0..200 {
            let lo = rng.next_u64();
            let hi = rng.next_u64() & MASK_HI_B127;
            let plain = B127::new_with_cfg(Uint::<2>::from_words([lo, hi]), &());
            let topped = B127::new_with_cfg(Uint::<2>::from_words([lo, hi | (1 << 63)]), &());
            assert_eq!(
                topped,
                plain + B127::from_canonical_words([REDUCTION_LOW_B127, 0])
            );
            // Both are canonical.
            assert_eq!(plain.as_words()[1] >> 63, 0);
            assert_eq!(topped.as_words()[1] >> 63, 0);
        }
        // From<u128> agrees with new_with_cfg on every pattern shape.
        let v = u128::MAX;
        assert_eq!(
            B127::from_polynomial_bits(v),
            B127::new_with_cfg(Uint::<2>::from_words([v as u64, (v >> 64) as u64]), &())
        );
    }

    #[test]
    fn try_from_words_rejects_noncanonical() {
        assert!(B127::try_from_words([0, 1 << 63]).is_none());
        assert!(B127::try_from_words([u64::MAX, MASK_HI_B127]).is_some());
    }

    #[test]
    fn is_generator_rejects_exactly_the_subfield() {
        assert!(!is_generator_b127(&B127::zero()));
        assert!(!is_generator_b127(&B127::one()));
        assert!(is_generator_b127(&B127::from_canonical_words([2, 0])));
        assert!(is_generator_b127(&B127::from_canonical_words([3, 0])));
    }

    /// The trait law: sums of products through the wide accumulator equal
    /// the same sums with reduced multiplies — bit-for-bit.
    #[test]
    fn wide_accumulate_matches_reduced_sum() {
        let mut rng = StdRng::seed_from_u64(0xB127_000F);
        for _ in 0..50 {
            let terms: Vec<(B127, B127)> = (0..17)
                .map(|_| (rand_elt(&mut rng), rand_elt(&mut rng)))
                .collect();
            let mut acc = B127::wide_zero(&B127::zero());
            let mut reduced = B127::zero();
            for (a, b) in &terms {
                B127::wide_add_assign(&mut acc, &B127::mul_wide(a, b));
                reduced += *a * *b;
            }
            assert_eq!(B127::from_wide(acc), reduced);
        }
    }

    /// The fused eq-factored kernels are value-exact against naive field
    /// arithmetic (the driver-contract identities).
    #[test]
    fn eqf_kernels_are_value_exact() {
        let mut rng = StdRng::seed_from_u64(0xB127_0010);
        for half in [1usize, 2, 3, 8, 13] {
            let l: Vec<_> = (0..2 * half).map(|_| rand_elt(&mut rng)).collect();
            let r: Vec<_> = (0..2 * half).map(|_| rand_elt(&mut rng)).collect();
            let l1: Vec<_> = (0..2 * half).map(|_| rand_elt(&mut rng)).collect();
            let r1: Vec<_> = (0..2 * half).map(|_| rand_elt(&mut rng)).collect();
            let w: Vec<_> = (0..half).map(|_| rand_elt(&mut rng)).collect();

            // Single-pair reference.
            let (mut c0, mut c1, mut c2) = (B127::zero(), B127::zero(), B127::zero());
            for b in 0..half {
                let (wl0, wl1) = (w[b] * l[2 * b], w[b] * l[2 * b + 1]);
                let (r0v, r1v) = (r[2 * b], r[2 * b + 1]);
                let t0 = wl0 * r0v;
                let t2 = (wl1 - wl0) * (r1v - r0v);
                c0 += t0;
                c2 += t2;
                c1 += wl1 * r1v - t0 - t2;
            }
            let got = B127::eqf_single_pair_round(&l, &r, &w, half).unwrap();
            assert_eq!(got, (c0, c1, c2), "single-pair, half={half}");

            // Two-pair reference: the same three sums over BOTH pairs.
            let (mut d0, mut d1, mut d2) = (c0, c1, c2);
            for b in 0..half {
                let (wl0, wl1) = (w[b] * l1[2 * b], w[b] * l1[2 * b + 1]);
                let (r0v, r1v) = (r1[2 * b], r1[2 * b + 1]);
                let t0 = wl0 * r0v;
                let t2 = (wl1 - wl0) * (r1v - r0v);
                d0 += t0;
                d2 += t2;
                d1 += wl1 * r1v - t0 - t2;
            }
            let got2 = B127::eqf_two_pair_round(&l, &r, &l1, &r1, &w, half).unwrap();
            assert_eq!(got2, (d0, d1, d2), "two-pair, half={half}");

            // Fold reference.
            let rho = rand_elt(&mut rng);
            let mut v = l.clone();
            let expected: Vec<_> = (0..half)
                .map(|b| v[2 * b] + rho * (v[2 * b + 1] - v[2 * b]))
                .collect();
            assert!(B127::eqf_fold_in_place(&mut v, &rho, half));
            assert_eq!(&v[..half], &expected[..], "fold, half={half}");
        }
    }

    #[test]
    fn implements_field_and_prime_field_traits() {
        fn assert_field<F: crate::poly::coefficient::FieldRepresentation>() {}
        fn assert_prime_field<F: crate::poly::coefficient::PolynomialField>() {}
        assert_field::<B127>();
        assert_prime_field::<B127>();
    }

    #[test]
    fn cfg_keyed_constructors_match_const_constructors() {
        assert_eq!(B127::zero_with_cfg(&()), B127::zero());
        assert_eq!(B127::one_with_cfg(&()), B127::one());
        assert!(B127::config_from_modulus(&B127_MODULUS).is_some());
        assert!(
            B127::config_from_modulus(&Uint::<2>::from_words([0x87, 0])).is_none(),
            "the GHASH modulus is not the b127 modulus"
        );
    }

    /// The general (two-fold) scalar reducer agrees with reduce-then-add
    /// linearity on arbitrary 256-bit inputs, and with the reference on
    /// in-contract products.
    #[test]
    fn scalar_reducer_is_linear_and_total() {
        let mut rng = StdRng::seed_from_u64(0xB127_0011);
        for _ in 0..500 {
            let w: [u64; 4] = core::array::from_fn(|_| rng.next_u64());
            let v: [u64; 4] = core::array::from_fn(|_| rng.next_u64());
            let xor = core::array::from_fn(|i| w[i] ^ v[i]);
            // F_2-linearity of the reduction, even on out-of-contract input.
            let rw = B127::reduce_wide(w);
            let rv = B127::reduce_wide(v);
            assert_eq!(B127::reduce_wide(xor), rw + rv);
            // Canonical output always.
            assert_eq!(rw.as_words()[1] >> 63, 0);
        }
        // Totality anchor: reducing X^255 = X^128·X^127 ≡ (X+1)·X^128 …
        // — checked against the bit reference through a product that
        // reaches the top: (X^126)² = X^252.
        let x126 = {
            let mut p = B127::one();
            for _ in 0..126 {
                p = p.mul_x();
            }
            p
        };
        let sq = x126.square();
        assert_eq!(to_u128(&sq), mul_reference(to_u128(&x126), to_u128(&x126)));
    }
}
