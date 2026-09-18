//! Canonical u128 operations for consumers that retain a field context beside
//! integer storage. Prepared factors are typed Montgomery field elements.

use super::*;

#[inline(always)]
fn words(value: u128) -> Uint<2> {
    Uint::from_words([value as u64, (value >> 64) as u64])
}
#[inline(always)]
fn packed(value: Uint<2>) -> u128 {
    value.as_words()[0] as u128 | ((value.as_words()[1] as u128) << 64)
}

impl FpCtx<2> {
    /// REDC of a prepared bounded integer, with canonical integer output.
    /// The owner guarantees `value < modulus * 2^128` before entering its
    /// kernel. This is the bound used by short native folds.
    #[inline]
    pub fn reduce_montgomery_bounded(&self, value: &Uint<4>) -> Uint<2> {
        let w = value.as_words();
        debug_assert!(
            Uint::from_words([w[2], w[3]])
                .ct_lt(self.modulus())
                .declassify()
        );
        self.params().redc(UintProduct {
            low: [w[0], w[1]],
            high: [w[2], w[3]],
        })
    }
    /// Trusted odd prime; primality is checked only in debug builds.
    pub fn from_prime_u128(prime: u128) -> Self {
        create_prime_field(words(prime))
    }
    pub fn modulus_u128(&self) -> u128 {
        packed(*self.modulus())
    }
    /// Reduces arbitrary u128 input to canonical integer output.
    #[inline]
    pub fn reduce_u128(&self, value: u128) -> u128 {
        packed(self.reduce_integer(&words(value)))
    }
    /// Adds canonical inputs under this context, without re-projecting them.
    #[inline]
    pub fn add_canonical_u128(&self, a: u128, b: u128) -> u128 {
        debug_assert!(a < self.modulus_u128() && b < self.modulus_u128());
        packed(self.params().add(&words(a), &words(b)))
    }
    /// Subtracts canonical inputs under this context, without re-projecting them.
    #[inline]
    pub fn sub_canonical_u128(&self, a: u128, b: u128) -> u128 {
        debug_assert!(a < self.modulus_u128() && b < self.modulus_u128());
        packed(self.params().sub(&words(a), &words(b)))
    }
    /// Adds arbitrary u128 inputs, returning a canonical integer.
    #[inline]
    pub fn add_u128(&self, a: u128, b: u128) -> u128 {
        packed(self.params().add(
            &self.reduce_integer(&words(a)),
            &self.reduce_integer(&words(b)),
        ))
    }
    /// Subtracts arbitrary u128 inputs, returning a canonical integer.
    #[inline]
    pub fn sub_u128(&self, a: u128, b: u128) -> u128 {
        packed(self.params().sub(
            &self.reduce_integer(&words(a)),
            &self.reduce_integer(&words(b)),
        ))
    }
    #[inline]
    pub fn prepare_multiplier_u128(&self, value: u128) -> Fp<2> {
        self.from_integer(&value)
    }
    /// Multiplies arbitrary integer input by a prepared Montgomery factor.
    #[inline]
    pub fn mul_prepared_u128(&self, value: u128, factor: &Fp<2>) -> u128 {
        packed(self.mul_canonical(&self.reduce_integer(&words(value)), factor))
    }
    /// Multiplies a known canonical integer by a prepared Montgomery factor.
    /// The owner establishes `value < modulus`; no projection is performed.
    #[inline]
    pub fn mul_canonical_u128(&self, value: u128, factor: &Fp<2>) -> u128 {
        packed(self.mul_canonical(&words(value), factor))
    }
    #[inline]
    pub fn mul_u128(&self, a: u128, b: u128) -> u128 {
        self.mul_prepared_u128(a, &self.prepare_multiplier_u128(b))
    }
    pub fn powers_u128(&self, base: u128, len: usize) -> Vec<u128> {
        let factor = self.prepare_multiplier_u128(base);
        let mut current = 1;
        (0..len)
            .map(|_| {
                let value = current;
                current = self.mul_canonical_u128(current, &factor);
                value
            })
            .collect()
    }
}
