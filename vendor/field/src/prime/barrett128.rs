//! Fixed two-limb delayed reduction, moved from BitZ's production kernel.
//! Requires 2^64 < modulus < 2^128. Public modulus checks select the fallback.
use super::params::PrimeParameters;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Uint, UintAccumulator};
    use num_bigint::BigUint;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    fn big(words: &[u64]) -> BigUint {
        BigUint::from_bytes_le(
            &words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn arbitrary_width_remainders_and_montgomery_scales() {
        let mut rng = Pcg64::seed_from_u64(1301);
        for modulus in [
            [13, 1],
            [u64::MAX - 158, u64::MAX],
            [u64::MAX - 14, (1 << 36) - 1],
        ] {
            let params = PrimeParameters::new(Uint::from_words(modulus));
            let p = big(&modulus);
            assert_eq!(big(&params.barrett_mu), (BigUint::from(1u64) << 256) / &p);
            for n in 1..10 {
                for iteration in 0..64 {
                    let words: Vec<_> = (0..n)
                        .map(|_| {
                            if iteration == 0 {
                                u64::MAX
                            } else {
                                rng.next_u64()
                            }
                        })
                        .collect();
                    assert_eq!(
                        big(params.remainder(words.as_slice()).as_words()),
                        big(&words) % &p
                    );
                    if n == 5 {
                        let acc = UintAccumulator {
                            low: [words[0], words[1]],
                            high: [words[2], words[3]],
                            head: words[4],
                        };
                        let result = params.reduce_product_acc(&acc);
                        assert_eq!((big(result.as_words()) << 128) % &p, big(&words) % &p);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn mac(accumulator: u64, lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
    let sum = accumulator as u128 + lhs as u128 * rhs as u128 + carry as u128;
    (sum as u64, (sum >> 64) as u64)
}

/// Folds `limbs[4] * 2^256` into four limbs via `2^256 mod q = R^2 mod q`.
#[inline(always)]
pub(super) fn fold_fifth_limb(limbs: [u64; 5], r2: [u64; 2]) -> ([u64; 4], u64) {
    let high = limbs[4];
    let mut out = [limbs[0], limbs[1], limbs[2], limbs[3]];

    let (word0, carry) = mac(out[0], high, r2[0], 0);
    out[0] = word0;
    let (word1, carry) = mac(out[1], high, r2[1], carry);
    out[1] = word1;

    let sum = (out[2] as u128) + (carry as u128);
    out[2] = sum as u64;
    let sum = (out[3] as u128) + (sum >> 64);
    out[3] = sum as u64;
    (out, (sum >> 64) as u64)
}

/// Two fixed Montgomery-elimination rounds. The three-limb result is not
/// assumed canonical: for a 100-bit modulus it may be many multiples of q and
/// is routed through Barrett reduction by the caller.
#[inline(always)]
pub(super) fn montgomery_reduce_4<const L: usize>(
    input: [u64; 4],
    reducer: &PrimeParameters<L>,
) -> [u64; 3] {
    let mut value = [input[0], input[1], input[2], input[3], 0u64];
    let mut i = 0;
    while i < 2 {
        let multiplier = value[i].wrapping_mul(reducer.neg_inv);
        let (word0, carry) = mac(value[i], multiplier, reducer.modulus.0[0], 0);
        value[i] = word0;
        let (word1, carry) = mac(value[i + 1], multiplier, reducer.modulus.0[1], carry);
        value[i + 1] = word1;

        let mut propagation = carry;
        let mut k = i + 2;
        while k < 5 {
            let sum = (value[k] as u128) + (propagation as u128);
            value[k] = sum as u64;
            propagation = (sum >> 64) as u64;
            k += 1;
        }
        debug_assert_eq!(propagation, 0);
        i += 1;
    }

    [value[2], value[3], value[4]]
}

/// Fixed `k = 2` Barrett reduction for any four-limb input.
#[inline(always)]
pub(super) fn barrett_reduce_4<const L: usize>(
    input: [u64; 4],
    reducer: &PrimeParameters<L>,
) -> [u64; 2] {
    // q1 = floor(input / b), q2 = q1 * mu, q3 = floor(q2 / b^3).
    let q1 = [input[1], input[2], input[3]];
    let q2 = multiply_3x3(q1, reducer.barrett_mu);
    let q3 = [q2[3], q2[4], q2[5]];

    // Only the low k+1 limbs are needed for the textbook Barrett remainder.
    let product = multiply_3x2_low3(q3, [reducer.modulus.0[0], reducer.modulus.0[1]]);
    let mut remainder = wrapping_sub_3([input[0], input[1], input[2]], product);

    // For mu=floor(b^(2k)/q), the candidate is below 3q. Execute both
    // correction steps unconditionally and select with borrow masks.
    remainder = conditional_subtract_3(remainder, [reducer.modulus.0[0], reducer.modulus.0[1]]);
    remainder = conditional_subtract_3(remainder, [reducer.modulus.0[0], reducer.modulus.0[1]]);
    debug_assert_eq!(remainder[2], 0);
    debug_assert!(less_than_2(
        [remainder[0], remainder[1]],
        [reducer.modulus.0[0], reducer.modulus.0[1]]
    ));
    [remainder[0], remainder[1]]
}

#[inline(always)]
fn multiply_3x3(lhs: [u64; 3], rhs: [u64; 3]) -> [u64; 6] {
    let mut out = [0u64; 6];
    let mut i = 0;
    while i < 3 {
        let mut carry = 0u64;
        let mut j = 0;
        while j < 3 {
            let (word, next_carry) = mac(out[i + j], lhs[i], rhs[j], carry);
            out[i + j] = word;
            carry = next_carry;
            j += 1;
        }
        let sum = (out[i + 3] as u128) + (carry as u128);
        out[i + 3] = sum as u64;
        let mut propagation = (sum >> 64) as u64;
        let mut k = i + 4;
        while k < 6 {
            let sum = (out[k] as u128) + (propagation as u128);
            out[k] = sum as u64;
            propagation = (sum >> 64) as u64;
            k += 1;
        }
        debug_assert_eq!(propagation, 0);
        i += 1;
    }
    out
}

#[inline(always)]
fn multiply_3x2_low3(lhs: [u64; 3], rhs: [u64; 2]) -> [u64; 3] {
    let mut out = [0u64; 3];
    let mut i = 0;
    while i < 3 {
        let mut carry = 0u64;
        let mut j = 0;
        while j < 2 && i + j < 3 {
            let (word, next_carry) = mac(out[i + j], lhs[i], rhs[j], carry);
            out[i + j] = word;
            carry = next_carry;
            j += 1;
        }
        if i + j < 3 {
            let sum = (out[i + j] as u128) + (carry as u128);
            out[i + j] = sum as u64;
        }
        i += 1;
    }
    out
}

#[inline(always)]
fn wrapping_sub_3(lhs: [u64; 3], rhs: [u64; 3]) -> [u64; 3] {
    let (word0, borrow0) = lhs[0].overflowing_sub(rhs[0]);
    let (word1, borrow1a) = lhs[1].overflowing_sub(rhs[1]);
    let (word1, borrow1b) = word1.overflowing_sub(u64::from(borrow0));
    let borrow1 = borrow1a | borrow1b;
    let (word2, _) = lhs[2].overflowing_sub(rhs[2]);
    let (word2, _) = word2.overflowing_sub(u64::from(borrow1));
    [word0, word1, word2]
}

#[inline(always)]
fn conditional_subtract_3(value: [u64; 3], modulus: [u64; 2]) -> [u64; 3] {
    let (word0, borrow0) = value[0].overflowing_sub(modulus[0]);
    let (word1, borrow1a) = value[1].overflowing_sub(modulus[1]);
    let (word1, borrow1b) = word1.overflowing_sub(u64::from(borrow0));
    let borrow1 = borrow1a | borrow1b;
    let (word2, borrow2) = value[2].overflowing_sub(u64::from(borrow1));

    // Select the difference iff the complete three-limb subtraction did not
    // borrow. `0 - bit` creates an all-zero/all-one mask without a branch.
    let select_difference = crate::CtMask::from_lsb(u64::from(!borrow2)).word();
    [
        (word0 & select_difference) | (value[0] & !select_difference),
        (word1 & select_difference) | (value[1] & !select_difference),
        (word2 & select_difference) | (value[2] & !select_difference),
    ]
}

#[inline(always)]
pub(super) fn add_mod_masked(
    value: [u64; 2],
    addend: [u64; 2],
    bit: u64,
    modulus: [u64; 2],
) -> [u64; 2] {
    let mask = crate::CtMask::from_lsb(bit).word();
    let sum = (value[0] as u128) + ((addend[0] & mask) as u128);
    let low = sum as u64;
    let sum = (value[1] as u128) + ((addend[1] & mask) as u128) + (sum >> 64);
    let candidate = [low, sum as u64, (sum >> 64) as u64];
    let reduced = conditional_subtract_3(candidate, modulus);
    debug_assert_eq!(reduced[2], 0);
    [reduced[0], reduced[1]]
}

#[inline(always)]
fn less_than_2(lhs: [u64; 2], rhs: [u64; 2]) -> bool {
    lhs[1] < rhs[1] || (lhs[1] == rhs[1] && lhs[0] < rhs[0])
}
