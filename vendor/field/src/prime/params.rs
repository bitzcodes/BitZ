use crate::integer::product::Words;
use crate::integer::{UintAccumulator, UintProduct};
use crate::traits::WideMul;
use crate::{CtMask, CtSelect, Uint};

/// Shared immutable constants. Preparation depends only on the public modulus.
#[derive(Clone, Debug)]
pub(crate) struct PrimeParameters<const L: usize> {
    pub modulus: Uint<L>,
    pub one: Uint<L>,
    pub r2: Uint<L>,
    pub neg_inv: u64,
    pub barrett_mu: [u64; 3],
    general_reduction: Option<crate::modular::reduction::Barrett<L>>,
}

impl<const L: usize> PrimeParameters<L> {
    pub const fn new(modulus: Uint<L>) -> Self {
        assert!(L > 0);
        debug_assert!(modulus.0[0] & 1 == 1, "Montgomery modulus must be odd");
        let mut any_high = 0;
        let mut i = 1;
        while i < L {
            any_high |= modulus.0[i];
            i += 1;
        }
        debug_assert!(
            any_high != 0 || modulus.0[0] >= 3,
            "prime modulus must be at least three"
        );
        let mut inverse = 1u64;
        i = 0;
        while i < 6 {
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(modulus.0[0].wrapping_mul(inverse)));
            i += 1;
        }
        let mut one = Uint::ONE;
        i = 0;
        while i < 64 * L {
            one = Self::double_public(one, &modulus);
            i += 1;
        }
        let mut r2 = one;
        i = 0;
        while i < 64 * L {
            r2 = Self::double_public(r2, &modulus);
            i += 1;
        }
        Self {
            modulus,
            one,
            r2,
            neg_inv: inverse.wrapping_neg(),
            barrett_mu: Self::prepare_barrett(&modulus),
            general_reduction: if L == 2 && modulus.0[1] != 0 {
                None
            } else {
                Some(crate::modular::reduction::Barrett::new(&modulus))
            },
        }
    }

    /// floor(2^256 / p), for the normalized two-limb case. This is bounded
    /// public setup, also usable in a static associated constant.
    const fn prepare_barrett(modulus: &Uint<L>) -> [u64; 3] {
        let mut quotient = [0u64; 3];
        if L != 2 || modulus.0[1] == 0 {
            return quotient;
        }
        let mut remainder = [0u64; 2];
        let mut position = 257usize;
        while position > 0 {
            position -= 1;
            let carry = remainder[1] >> 63;
            remainder[1] = (remainder[1] << 1) | (remainder[0] >> 63);
            remainder[0] = (remainder[0] << 1) | (position == 256) as u64;
            let (low, b0) = remainder[0].overflowing_sub(modulus.0[0]);
            let (high, b1) = remainder[1].overflowing_sub(modulus.0[1]);
            let (high, b2) = high.overflowing_sub(b0 as u64);
            if carry != 0 || !(b1 | b2) {
                remainder = [low, high];
                if position < 192 {
                    quotient[position / 64] |= 1 << (position % 64);
                }
            }
        }
        quotient
    }

    // Fixed-width long division for const preparation. No O(R / p) subtraction loop.
    const fn double_public(value: Uint<L>, modulus: &Uint<L>) -> Uint<L> {
        let mut shifted = [0; L];
        let mut carry = 0;
        let mut i = 0;
        while i < L {
            shifted[i] = (value.0[i] << 1) | carry;
            carry = value.0[i] >> 63;
            i += 1;
        }
        let mut difference = [0; L];
        let mut borrow = 0;
        i = 0;
        while i < L {
            let (word, b0) = shifted[i].overflowing_sub(modulus.0[i]);
            let (word, b1) = word.overflowing_sub(borrow);
            difference[i] = word;
            borrow = (b0 | b1) as u64;
            i += 1;
        }
        if carry != 0 || borrow == 0 {
            Uint(difference)
        } else {
            Uint(shifted)
        }
    }

    #[inline]
    pub fn add(&self, a: &Uint<L>, b: &Uint<L>) -> Uint<L> {
        let (sum, carry) = a.adc(b);
        let (reduced, borrow) = sum.sbb(&self.modulus);
        Uint::ct_select(&sum, &reduced, CtMask::from_lsb(carry | (borrow ^ 1)))
    }
    #[inline]
    pub fn sub(&self, a: &Uint<L>, b: &Uint<L>) -> Uint<L> {
        if L == 2 {
            let lhs = a.0[0] as u128 | ((a.0[1] as u128) << 64);
            let rhs = b.0[0] as u128 | ((b.0[1] as u128) << 64);
            let modulus = self.modulus.0[0] as u128 | ((self.modulus.0[1] as u128) << 64);
            let (difference, borrow) = lhs.overflowing_sub(rhs);
            let mask = 0u128.wrapping_sub(borrow as u128);
            let result = difference.wrapping_add(modulus & mask);
            return Uint(core::array::from_fn(|i| (result >> (64 * i)) as u64));
        }
        let (difference, borrow) = a.sbb(b);
        // Mask only the correction, rather than selecting two full results.
        // The discarded carry cancels the wrapping subtraction's borrow.
        let mask = CtMask::from_lsb(borrow).word();
        let correction = Uint(core::array::from_fn(|i| self.modulus.0[i] & mask));
        difference.adc(&correction).0
    }
    #[inline]
    pub fn neg(&self, a: &Uint<L>) -> Uint<L> {
        self.sub(&Uint::ZERO, a)
    }

    /// REDC for inputs below pR. Scalar products of canonical residues satisfy it.
    #[inline]
    pub fn redc(&self, product: UintProduct<L, L>) -> Uint<L> {
        if L == 2 {
            let result = super::montgomery128::redc(
                [
                    product.low[0],
                    product.low[1],
                    product.high[0],
                    product.high[1],
                ],
                [self.modulus.0[0], self.modulus.0[1]],
                self.neg_inv,
            );
            return Uint(core::array::from_fn(|i| result[i]));
        }
        let mut value = UintAccumulator::from(product);
        for i in 0..L {
            let factor = value.word(i).wrapping_mul(self.neg_inv);
            let mut carry = 0;
            for j in 0..L {
                let sum =
                    factor as u128 * self.modulus.0[j] as u128 + value.word(i + j) as u128 + carry;
                value.set_word(i + j, sum as u64);
                carry = sum >> 64;
            }
            for k in i + L..2 * L + 1 {
                let sum = value.word(k) as u128 + carry;
                value.set_word(k, sum as u64);
                carry = sum >> 64;
            }
            debug_assert_eq!(carry, 0);
        }
        let candidate = Uint(core::array::from_fn(|i| value.word(L + i)));
        let (reduced, borrow) = candidate.sbb(&self.modulus);
        Uint::ct_select(
            &candidate,
            &reduced,
            CtMask::from_lsb(value.head | (borrow ^ 1)),
        )
    }
    #[inline]
    pub fn mul(&self, a: &Uint<L>, b: &Uint<L>) -> Uint<L> {
        if L == 2 {
            let result = super::montgomery128::mul(
                [a.0[0], a.0[1]],
                [b.0[0], b.0[1]],
                [self.modulus.0[0], self.modulus.0[1]],
                self.neg_inv,
            );
            return Uint(core::array::from_fn(|i| result[i]));
        }
        self.redc(crate::IntegerOps.mul_wide(a, b))
    }
    pub fn from_canonical(&self, a: &Uint<L>) -> Uint<L> {
        self.mul(a, &self.r2)
    }
    pub fn to_canonical(&self, a: &Uint<L>) -> Uint<L> {
        self.mul(a, &Uint::ONE)
    }

    /// Reduce every supplied limb. The schedule depends on width, never magnitude.
    pub fn remainder(&self, words: &(impl Words + ?Sized)) -> Uint<L> {
        // Preserve the measured two-limb schedule; other public widths use
        // prepared radix division rather than a bit-by-bit execution fallback.
        if L == 2 && self.modulus.0[1] != 0 {
            return self.remainder_barrett(words);
        }
        self.general_reduction
            .as_ref()
            .expect("prepared general modulus")
            .remainder(words, &self.modulus)
    }

    fn remainder_barrett(&self, words: &(impl Words + ?Sized)) -> Uint<L> {
        use super::barrett128::*;
        let r2 = [self.r2.0[0], self.r2.0[1]];
        let modulus = [self.modulus.0[0], self.modulus.0[1]];
        let result = if words.len() <= 5 {
            let mut limbs = [0; 5];
            for (i, dst) in limbs.iter_mut().enumerate().take(words.len()) {
                *dst = words.word(i);
            }
            let (folded, carry) = fold_fifth_limb(limbs, r2);
            add_mod_masked(barrett_reduce_4(folded, self), r2, carry, modulus)
        } else {
            // Horner in radix 2^64; no dependence on the integer's magnitude.
            let mut remainder = [0; 2];
            for i in (0..words.len()).rev() {
                remainder = barrett_reduce_4([words.word(i), remainder[0], remainder[1], 0], self);
            }
            remainder
        };
        Uint(core::array::from_fn(|i| result[i]))
    }

    pub fn reduce_product_acc(&self, input: &UintAccumulator<L, L>) -> Uint<L> {
        if L == 2 && self.modulus.0[1] != 0 {
            use super::barrett128::*;
            let limbs = core::array::from_fn(|i| input.word(i));
            let (folded, carry) = fold_fifth_limb(limbs, [self.r2.0[0], self.r2.0[1]]);
            let redc = montgomery_reduce_4(folded, self);
            let reduced = barrett_reduce_4([redc[0], redc[1], redc[2], 0], self);
            let result = add_mod_masked(
                reduced,
                [self.one.0[0], self.one.0[1]],
                carry,
                [self.modulus.0[0], self.modulus.0[1]],
            );
            return Uint(core::array::from_fn(|i| result[i]));
        }
        self.to_canonical(&self.remainder(input))
    }
}
