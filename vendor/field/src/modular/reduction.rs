//! Prepared radix-2^64 Barrett division. All execution bounds are public widths.

use crate::integer::product::Words;
use crate::{CtMask, Uint, UintProduct};

#[derive(Clone, Debug)]
pub(crate) struct Barrett<const L: usize> {
    limbs: usize,
    reciprocal: UintProduct<L, 1>,
    power_of_two: Option<usize>,
}

// Stable Rust cannot express [u64; 2 * L + 2]. The split is storage only.
struct Scratch<const L: usize> {
    product: UintProduct<L, L>,
    head: [u64; 2],
}
impl<const L: usize> Scratch<L> {
    fn zero() -> Self {
        Self {
            product: UintProduct::ZERO,
            head: [0; 2],
        }
    }
    fn word(&self, i: usize) -> u64 {
        if i < 2 * L {
            self.product.word(i)
        } else {
            self.head[i - 2 * L]
        }
    }
    fn set(&mut self, i: usize, word: u64) {
        if i < 2 * L {
            self.product.set_word(i, word);
        } else {
            self.head[i - 2 * L] = word;
        }
    }
}

impl<const L: usize> Barrett<L> {
    /// Public setup, including const preparation for static prime specifications.
    pub const fn new(modulus: &Uint<L>) -> Self {
        assert!(L > 0);
        let mut limbs = L;
        while limbs > 0 && modulus.0[limbs - 1] == 0 {
            limbs -= 1;
        }
        assert!(limbs > 0, "zero divisor");
        let mut bits = 0;
        let mut power = 0;
        let mut i = 0;
        while i < limbs {
            bits += modulus.0[i].count_ones();
            if modulus.0[i] != 0 {
                power = i * 64 + modulus.0[i].trailing_zeros() as usize;
            }
            i += 1;
        }
        let mut reciprocal = UintProduct::ZERO;
        if bits == 1 {
            return Self {
                limbs,
                reciprocal,
                power_of_two: Some(power),
            };
        }
        // mu = floor(B^(2k) / m). For non-powers of two m > B^(k-1),
        // so mu occupies at most k+1 limbs. This loop only handles public data.
        let mut rem = [0u64; L];
        let mut position = 128 * limbs + 1;
        while position > 0 {
            position -= 1;
            let mut carry = (position == 128 * limbs) as u64;
            i = 0;
            while i < limbs {
                let next = rem[i] >> 63;
                rem[i] = (rem[i] << 1) | carry;
                carry = next;
                i += 1;
            }
            let mut difference = [0u64; L];
            let mut borrow = false;
            i = 0;
            while i < limbs {
                let (v, b0) = rem[i].overflowing_sub(modulus.0[i]);
                let (v, b1) = v.overflowing_sub(borrow as u64);
                difference[i] = v;
                borrow = b0 | b1;
                i += 1;
            }
            if carry != 0 || !borrow {
                rem = difference;
                let word = position / 64;
                if word < L {
                    reciprocal.low[word] |= 1 << (position % 64);
                } else {
                    reciprocal.high[word - L] |= 1 << (position % 64);
                }
            }
        }
        Self {
            limbs,
            reciprocal,
            power_of_two: None,
        }
    }

    pub fn remainder(&self, input: &(impl Words + ?Sized), modulus: &Uint<L>) -> Uint<L> {
        self.divide_into(input, modulus, |_, _| {})
    }

    pub fn divide_into(
        &self,
        input: &(impl Words + ?Sized),
        modulus: &Uint<L>,
        mut quotient_word: impl FnMut(usize, u64),
    ) -> Uint<L> {
        if let Some(bits) = self.power_of_two {
            let (whole, shift) = (bits / 64, bits % 64);
            let mut remainder = Uint::ZERO;
            for i in 0..input.len() {
                if i < whole {
                    remainder.0[i] = input.word(i);
                } else if i == whole && shift != 0 {
                    remainder.0[i] = input.word(i) & ((1u64 << shift) - 1);
                }
                let lo = if i + whole < input.len() {
                    input.word(i + whole)
                } else {
                    0
                };
                let hi = if shift != 0 && i + whole + 1 < input.len() {
                    input.word(i + whole + 1)
                } else {
                    0
                };
                quotient_word(
                    i,
                    if shift == 0 {
                        lo
                    } else {
                        (lo >> shift) | (hi << (64 - shift))
                    },
                );
            }
            return remainder;
        }
        // Full-width small divisors are common in P-256 hints. Pass their
        // type-level width through the kernel so the limb loops specialize;
        // padded divisors retain their prepared public significant width.
        if self.limbs == L && L <= 4 {
            self.divide_into_limbs(input, modulus, quotient_word, L)
        } else {
            self.divide_into_limbs(input, modulus, quotient_word, self.limbs)
        }
    }

    #[inline(always)]
    fn divide_into_limbs(
        &self,
        input: &(impl Words + ?Sized),
        modulus: &Uint<L>,
        mut quotient_word: impl FnMut(usize, u64),
        k: usize,
    ) -> Uint<L> {
        let mut remainder = Uint::ZERO;
        for block in (0..input.len().div_ceil(k)).rev() {
            let offset = block * k;
            let available = input.len() - offset;
            if available < k {
                // The public leading partial block is below B^(k-1) <= m.
                // It is already a remainder, and its quotient words are zero.
                for i in 0..available {
                    remainder.0[i] = input.word(offset + i);
                    quotient_word(offset + i, 0);
                }
                continue;
            }
            let mut value = UintProduct::<L, L>::ZERO;
            for i in 0..k {
                value.set_word(
                    i,
                    if offset + i < input.len() {
                        input.word(offset + i)
                    } else {
                        0
                    },
                );
                value.set_word(k + i, remainder.0[i]);
            }
            let (quotient, rem) = self.divide_block(&value, modulus, k);
            remainder = rem;
            for i in 0..k.min(input.len() - offset) {
                quotient_word(offset + i, quotient.0[i]);
            }
        }
        remainder
    }

    /// Divide x < m B^k, hence the quotient fits k limbs. Barrett's estimate
    /// is at most two below the exact quotient; execute both corrections.
    #[inline(always)]
    fn divide_block(&self, x: &UintProduct<L, L>, modulus: &Uint<L>, k: usize) -> (Uint<L>, Uint<L>) {
        let mut product = Scratch::<L>::zero();
        for i in 0..=k {
            let a = if i < k + 1 && k - 1 + i < 2 * k {
                x.word(k - 1 + i)
            } else {
                0
            };
            let mut carry = 0u128;
            for j in 0..=k {
                let sum = a as u128 * self.reciprocal.word(j) as u128
                    + product.word(i + j) as u128
                    + carry;
                product.set(i + j, sum as u64);
                carry = sum >> 64;
            }
            product.set(i + k + 1, carry as u64);
        }
        let mut quotient = UintProduct::<L, 1>::ZERO;
        for i in 0..=k {
            quotient.set_word(i, product.word(k + 1 + i));
        }
        let mut multiple = UintProduct::<L, 1>::ZERO;
        for i in 0..=k {
            let mut carry = 0u128;
            for j in 0..k.min(k + 1 - i) {
                let sum = quotient.word(i) as u128 * modulus.0[j] as u128
                    + multiple.word(i + j) as u128
                    + carry;
                multiple.set_word(i + j, sum as u64);
                carry = sum >> 64;
            }
            if i == 0 {
                multiple.set_word(k, carry as u64);
            }
        }
        let mut remainder = UintProduct::<L, 1>::ZERO;
        let mut borrow = false;
        for i in 0..=k {
            let (word, b0) = x.word(i).overflowing_sub(multiple.word(i));
            let (word, b1) = word.overflowing_sub(borrow as u64);
            remainder.set_word(i, word);
            borrow = b0 | b1;
        }
        for _ in 0..2 {
            let mut difference = UintProduct::<L, 1>::ZERO;
            let mut borrow = false;
            for i in 0..=k {
                let m = if i < k { modulus.0[i] } else { 0 };
                let (word, b0) = remainder.word(i).overflowing_sub(m);
                let (word, b1) = word.overflowing_sub(borrow as u64);
                difference.set_word(i, word);
                borrow = b0 | b1;
            }
            let mask = CtMask::from_lsb((borrow as u64) ^ 1).word();
            let mut carry = mask & 1;
            for i in 0..=k {
                remainder.set_word(i, (remainder.word(i) & !mask) | (difference.word(i) & mask));
                let (word, overflow) = quotient.word(i).overflowing_add(carry);
                quotient.set_word(i, word);
                carry = overflow as u64;
            }
        }
        debug_assert_eq!(remainder.word(k), 0);
        debug_assert_eq!(quotient.word(k), 0);
        let mut q = Uint::ZERO;
        let mut r = Uint::ZERO;
        for i in 0..k {
            q.0[i] = quotient.word(i);
            r.0[i] = remainder.word(i);
        }
        (q, r)
    }
}
