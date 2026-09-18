//! Deferred-reduction accumulator.
//!
//! Reduction is a fixed pattern of shifts and XORs of the input words — no step
//! depends on the data — so it is `F_2`-linear:
//! `reduce(x + y) = reduce(x) + reduce(y)`. A sum of products can therefore be
//! accumulated unreduced and reduced once, bit-identically to reducing each
//! product. That trades one reduction per term for one per accumulator, the
//! shape of the sumcheck round bodies.

use std::ops::AddAssign;

use super::{Gf128, kernel};

/// An unreduced 256-bit carryless product, or a sum of several.
#[derive(Clone, Copy)]
pub struct Wide256(kernel::Wide);

impl Wide256 {
    #[inline]
    pub(crate) fn from_words(words: [u64; 4]) -> Self {
        Self(kernel::wide_from_words(words))
    }
    /// Internal bound: XOR of canonical B127 products (degree <=252).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub(crate) fn reduce_b127_product(self) -> [u64; 2] {
        crate::binary::b127_aarch64::reduce_product(self.0.0, self.0.1)
    }
    #[inline]
    pub(crate) fn words(self) -> [u64; 4] {
        kernel::wide_words(self.0)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    pub(crate) fn square(a: Gf128) -> Self {
        Self(kernel::wide_square(a.words()))
    }
    #[inline]
    pub fn zero() -> Self {
        Self(kernel::wide_zero())
    }

    /// A reduced element seen as a wide value — the same polynomial, with
    /// nothing above `X^128`.
    #[inline]
    pub fn of(a: Gf128) -> Self {
        Self(kernel::wide_of(a.words()))
    }

    /// The 256-bit product of two elements, left unreduced.
    #[inline]
    pub fn mul(a: Gf128, b: Gf128) -> Self {
        Self(kernel::wide_mul(a.words(), b.words()))
    }

    #[inline]
    pub fn reduce(self) -> Gf128 {
        kernel::wide_reduce(self.0).into()
    }
}

impl AddAssign for Wide256 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = kernel::wide_add(self.0, rhs.0);
    }
}

/// Adding a reduced element directly, without [`Wide256::of`] first: only the
/// low half is touched, so the zero high half is never materialized.
impl AddAssign<Gf128> for Wide256 {
    #[inline]
    fn add_assign(&mut self, rhs: Gf128) {
        self.0 = kernel::wide_add_low(self.0, rhs.words());
    }
}

#[cfg(test)]
mod tests {
    use super::super::portable;
    use super::*;
    use num_traits::ConstZero;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    fn f128(rng: &mut Pcg64) -> Gf128 {
        Gf128::new(rng.next_u64(), rng.next_u64())
    }

    fn words4(rng: &mut Pcg64) -> [u64; 4] {
        [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ]
    }

    /// The claim the whole type rests on. Tested against `portable::reduce`
    /// over arbitrary 256-bit values, not only over ones reachable as products,
    /// since linearity is a property of the map and not of its inputs.
    #[test]
    fn reduction_is_linear() {
        let mut rng = Pcg64::seed_from_u64(201);
        for _ in 0..512 {
            let (x, y) = (words4(&mut rng), words4(&mut rng));
            let sum = portable::wide_add(x, y);
            let [lo, hi] = portable::reduce(sum);
            let ([xlo, xhi], [ylo, yhi]) = (portable::reduce(x), portable::reduce(y));
            assert_eq!([lo, hi], [xlo ^ ylo, xhi ^ yhi], "{x:?} + {y:?}");
        }
    }

    #[test]
    fn constructors_agree_with_the_field() {
        let mut rng = Pcg64::seed_from_u64(202);
        assert_eq!(Wide256::zero().reduce(), Gf128::ZERO);
        for _ in 0..256 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            assert_eq!(Wide256::of(a).reduce(), a);
            assert_eq!(Wide256::mul(a, b).reduce(), a * b);
        }
    }

    /// Accumulate 64 products unreduced and reduce once, against reducing each
    /// product and summing in the field.
    #[test]
    fn deferred_sum_matches_eager_sum() {
        let mut rng = Pcg64::seed_from_u64(203);
        for _ in 0..64 {
            let mut wide = Wide256::zero();
            let mut eager = Gf128::ZERO;
            for _ in 0..64 {
                let (a, b) = (f128(&mut rng), f128(&mut rng));
                wide += Wide256::mul(a, b);
                eager += a * b;
            }
            assert_eq!(wide.reduce(), eager);
        }
    }

    /// Adding a reduced element directly agrees with upcasting it first, and
    /// with the field sum — interleaved with products, so the untouched high
    /// half is checked against live data rather than zeros.
    #[test]
    fn adding_reduced_elements_matches_upcast() {
        let mut rng = Pcg64::seed_from_u64(205);
        for _ in 0..64 {
            let mut direct = Wide256::zero();
            let mut upcast = Wide256::zero();
            let mut eager = Gf128::ZERO;
            for _ in 0..16 {
                let (a, b, c) = (f128(&mut rng), f128(&mut rng), f128(&mut rng));
                direct += Wide256::mul(a, b);
                direct += c;
                upcast += Wide256::mul(a, b);
                upcast += Wide256::of(c);
                eager += a * b + c;
            }
            assert_eq!(direct.reduce(), upcast.reduce());
            assert_eq!(direct.reduce(), eager);
        }
    }

    /// As with the multiply, the vector accumulator is pinned to the portable
    /// one — here before reduction as well as after, so an error in
    /// `clmul_256` that the fold happens to cancel still shows up.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_wide_matches_portable() {
        use super::super::aarch64;

        let mut rng = Pcg64::seed_from_u64(204);
        let mut neon = aarch64::wide_zero();
        let mut port = portable::wide_zero();
        for _ in 0..512 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            let (aw, bw) = (a.words(), b.words());

            let (n, p) = (aarch64::wide_mul(aw, bw), portable::wide_mul(aw, bw));
            assert_eq!(aarch64::wide_words(n), p, "product of {a:?} and {b:?}");

            neon = aarch64::wide_add(neon, n);
            port = portable::wide_add(port, p);
            assert_eq!(aarch64::wide_words(neon), port);
        }
        assert_eq!(aarch64::wide_reduce(neon), portable::wide_reduce(port));

        let a = f128(&mut rng);
        assert_eq!(
            aarch64::wide_words(aarch64::wide_of(a.words())),
            portable::wide_of(a.words())
        );
        assert_eq!(
            aarch64::wide_words(aarch64::wide_add_low(neon, a.words())),
            portable::wide_add_low(port, a.words())
        );
    }
}
