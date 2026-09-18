//! Exponentiation, inversion, and the primitive-element test.

use super::{Gf128, kernel};
use num_traits::{Inv, Pow};

/// The order of the multiplicative group, `2^128 - 1`.
///
/// Exactly `u128::MAX`, so for a generator `alpha` the map `n -> alpha^n` is
/// injective on `[0, 2^128 - 1)`: every `u128` but `u128::MAX` names a distinct
/// element.
pub const MULT_ORDER: u128 = u128::MAX;

/// The nine distinct primes dividing [`MULT_ORDER`], which is squarefree.
///
/// `2^128 - 1 = prod_{k=0}^{6} (2^(2^k) + 1)`: the five Fermat primes
/// `3, 5, 17, 257, 65537`, then `2^32 + 1 = 641 * 6700417` and
/// `2^64 + 1 = 274177 * 67280421310721`.
///
/// [`is_generator`] tests one condition per entry, so a composite or missing
/// entry would weaken it silently. The tests re-derive the product and the
/// primality of every entry rather than trusting this list.
pub const ORDER_PRIME_FACTORS: [u128; 9] =
    [3, 5, 17, 257, 641, 65537, 274177, 6700417, 67280421310721];

impl Gf128 {
    /// `self^(2^k)`. On NEON the value stays in one vector register, so `k`
    /// squarings cost `k` PMULL pairs and one load/store — why
    /// [`Inv::inv`] works in runs.
    pub fn square_n(self, k: u32) -> Self {
        kernel::square_n(self.words(), k).into()
    }
}

impl Pow<u128> for Gf128 {
    type Output = Self;

    /// Square-and-multiply, low bit first. `self^0` is `ONE`, including for
    /// `ZERO`.
    fn pow(self, exp: u128) -> Self {
        let mut acc = Self::ONE;
        let mut base = self;
        let mut e = exp;
        while e != 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }
}

impl Pow<&u128> for Gf128 {
    type Output = Self;
    fn pow(self, rhs: &u128) -> Self {
        self.pow(*rhs)
    }
}

impl Pow<u32> for Gf128 {
    type Output = Self;
    fn pow(self, rhs: u32) -> Self {
        self.pow(u128::from(rhs))
    }
}

impl Inv for Gf128 {
    type Output = Option<Self>;

    /// The multiplicative inverse, or `None` for zero.
    ///
    /// Itoh–Tsujii, Information and Computation 78(3):171-177, 1988: with
    /// `b_k = self^(2^k - 1)`, the identity `b_(m+n) = (b_m)^(2^n) * b_n`
    /// turns each step of the addition chain
    /// `1, 2, 3, 6, 12, 24, 48, 96, 120, 126, 127` into one squaring run and
    /// one multiply. `b_127` squared is `self^(2^128 - 2)`, the inverse.
    /// 127 squarings and 10 multiplies, against 127 and 127 for the same
    /// exponent by square-and-multiply.
    ///
    /// The paper counts only the multiplies, since a normal basis squares by
    /// cyclic shift. This basis is polynomial, so the squarings are real work
    /// and [`Gf128::square_n`] is what keeps them cheap.
    fn inv(self) -> Option<Self> {
        use crate::FieldOps;
        let inverse = crate::Gf128Ops.inverse_ct(&self);
        inverse.validity().declassify().then_some(*inverse.value())
    }
}

/// Returns `true` if `a` generates the whole multiplicative group.
///
/// The primitive-element test: `a^((2^128 - 1)/p) != 1` for every prime `p`
/// dividing the order. Zero needs its own guard rather than a fast path —
/// `0^e` is `0`, never `1`, so every factor would pass.
///
/// `prod (1 - 1/p)` is 49.9%, so a random `a` passes after two tries on
/// average.
pub fn is_generator(a: Gf128) -> bool {
    !a.is_zero()
        && ORDER_PRIME_FACTORS
            .iter()
            .all(|&p| a.pow(MULT_ORDER / p) != Gf128::ONE)
}

/// The smallest `Gf128::from_polynomial_bits(n)`, `n >= 2`, that generates the group — a
/// deterministic choice for tests and benches. The protocol accepts any `a`
/// that [`is_generator`] does.
pub fn smallest_generator() -> Gf128 {
    (2u128..)
        .map(Gf128::from_polynomial_bits)
        .find(|&a| is_generator(a))
        .expect("a finite field's multiplicative group is cyclic")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    fn f128(rng: &mut Pcg64) -> Gf128 {
        Gf128::new(rng.next_u64(), rng.next_u64())
    }

    fn is_prime(n: u128) -> bool {
        if n < 2 {
            return false;
        }
        let mut d = 2u128;
        while d * d <= n {
            if n.is_multiple_of(d) {
                return false;
            }
            d += 1;
        }
        true
    }

    /// [`is_generator`] tests one condition per entry, so a composite entry or
    /// a missing one would silently weaken it into accepting elements of a
    /// proper subgroup. Trial division rather than a probabilistic test: the
    /// largest entry is under `2^47`, so this is a proof, not evidence.
    #[test]
    fn order_factorization_is_complete() {
        let mut product = 1u128;
        for &p in &ORDER_PRIME_FACTORS {
            assert!(is_prime(p), "{p} is not prime");
            product = product.checked_mul(p).expect("product overflows u128");
        }
        assert_eq!(product, MULT_ORDER);
        assert_eq!(MULT_ORDER, u128::MAX);
    }

    #[test]
    fn square_n_is_repeated_squaring() {
        let mut rng = Pcg64::seed_from_u64(301);
        for _ in 0..64 {
            let a = f128(&mut rng);
            let mut expected = a;
            for k in 0..=130u32 {
                assert_eq!(a.square_n(k), expected, "square_n({k}) of {a:?}");
                expected = expected.square();
            }
        }
    }

    #[test]
    fn pow_matches_repeated_multiplication() {
        let mut rng = Pcg64::seed_from_u64(302);
        for _ in 0..64 {
            let a = f128(&mut rng);
            let mut expected = Gf128::ONE;
            for e in 0..64u128 {
                assert_eq!(a.pow(e), expected, "{a:?}^{e}");
                expected *= a;
            }
            // A pure power of two exercises the squaring chain with a single
            // set bit, where an off-by-one in the ladder still shows up.
            assert_eq!(a.pow(1_u128 << 100), a.square_n(100));
        }
    }

    /// Lagrange: every non-zero element satisfies `a^(2^128 - 1) = 1`, and
    /// squaring 128 times is the identity (Frobenius over the prime field).
    #[test]
    fn group_order_and_frobenius() {
        let mut rng = Pcg64::seed_from_u64(303);
        for _ in 0..64 {
            let a = f128(&mut rng);
            assert_eq!(a.square_n(128), a);
            if !a.is_zero() {
                assert_eq!(a.pow(MULT_ORDER), Gf128::ONE);
            }
        }
    }

    /// Itoh–Tsujii against the definition it is an optimisation of.
    #[test]
    fn inverse_matches_fermat() {
        let mut rng = Pcg64::seed_from_u64(304);
        assert_eq!(Gf128::ZERO.inv(), None);
        assert_eq!(Gf128::ONE.inv(), Some(Gf128::ONE));
        for _ in 0..256 {
            let a = f128(&mut rng);
            if a.is_zero() {
                continue;
            }
            let inv = a.inv().expect("non-zero");
            assert_eq!(inv, a.pow(MULT_ORDER - 1), "{a:?}");
            assert_eq!(a * inv, Gf128::ONE, "{a:?}");
        }
    }

    /// `X` is a generator, and the smallest one — see [`Gf128::GENERATOR`] for
    /// why that is not automatic.
    #[test]
    fn generator_is_x() {
        assert!(is_generator(Gf128::GENERATOR));
        assert_eq!(smallest_generator(), Gf128::GENERATOR);
    }

    /// The test must reject as well as accept, or it would be vacuous. `g^3`
    /// has order `(2^128-1)/3`; `g^2` is still a generator because the order is
    /// odd, so squaring permutes the group.
    #[test]
    fn is_generator_rejects_proper_subgroups() {
        let g = Gf128::GENERATOR;
        assert!(!is_generator(Gf128::ZERO));
        assert!(!is_generator(Gf128::ONE));
        for &p in &ORDER_PRIME_FACTORS {
            let a = g.pow(p);
            assert!(!is_generator(a), "g^{p} should have order (2^128-1)/{p}");
        }
        assert!(is_generator(g.pow(2_u32)));
    }

    #[test]
    fn comb_matches_the_general_pow() {
        let mut rng = Pcg64::seed_from_u64(305);
        let alpha = smallest_generator();
        for win in [1u32, 4, 8] {
            let comb = crate::preparation::FixedBasePow::<_, 2>::new_public(
                crate::Gf128Ops,
                alpha,
                win as usize,
            );
            assert_eq!(comb.pow_public(&crate::Uint::ZERO), Gf128::ONE);
            for _ in 0..64 {
                let e = (rng.next_u64() as u128) << 64 | rng.next_u64() as u128;
                assert_eq!(
                    comb.pow_public(&crate::Uint::from_words([e as u64, (e >> 64) as u64])),
                    alpha.pow(e),
                    "win {win}, exp {e:#x}"
                );
            }
            assert_eq!(comb.pow_public(&crate::Uint::MAX), alpha.pow(u128::MAX));
        }
    }

    #[test]
    fn byte_window_products_handle_zero_and_sparse_exponents() {
        for alpha in [Gf128::ZERO, Gf128::ONE, smallest_generator()] {
            let comb =
                crate::preparation::FixedBasePow::<_, 2>::new_public(crate::Gf128Ops, alpha, 8);
            for exponent in [0, 1, 1 << 63, 1 << 64, 1 << 127, u128::MAX] {
                let words = crate::Uint::from_words([exponent as u64, (exponent >> 64) as u64]);
                assert_eq!(comb.pow_public(&words), alpha.pow(exponent));
            }
        }
    }
}
