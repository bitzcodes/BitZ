//! Step 5.0: reducing the prime modulus of a bitified tensor claim.
//!
//! Implements the paper's Remark "Description of Step 5.0" (Technical
//! Overview, \S{}2.1) for integer constraints.  The bitified terminal claim
//!
//! ```text
//! < u1 (x) u2, bits > = mu   in F_Q,     u1 in F_Q^{2^t},  u2 in F_Q^{2^s},
//! ```
//!
//! lives over the large fingerprint field `F_Q`, which violates the
//! exponent-fold no-wrap condition.  The prover therefore sends the exact
//! integer evaluation
//!
//! ```text
//! mu' = < lift(u1) (x) lift(u2), bits >   in Z,     0 <= mu' < d * Q^2,
//! ```
//!
//! where `lift` is the canonical representative in `[0, Q)` and
//! `d = 2^{t+s}` is the committed cell count.  The verifier checks
//! `mu' = mu (mod Q)` and the magnitude bound, samples a fresh prime `q'`
//! below the no-wrap boundary, and both sides continue with the tensor
//! *factors* reduced modulo `q'` — reduction is a ring homomorphism, so
//! `lift(u1) (x) lift(u2) mod q' = (lift(u1) mod q') (x) (lift(u2) mod q')`
//! entrywise, and the claim keeps the tensor shape the low-entropy grand
//! product needs.
//!
//! Lifting the factors separately (rather than the entries of the reduced
//! tensor, whose bound would be `d * Q` as in the remark's statement) is
//! what preserves the tensor decomposition; the price is the `d * Q^2`
//! magnitude bound used in the soundness accounting: a false claim leaves
//! an integer defect below `2 * d * Q^2 <= 2^282`, which has at most two
//! prime divisors in the `[2^112, 2^113)` reduction interval.

use field::RingOps;
use field::{CanonicalCodec, CtMask, CtOrd, CtSelect, IntegerOps, PreparedDivisor, Uint, WideMul};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Exact lift storage: 256 product bits plus 64 bits of public cell-count
/// headroom. A shape with fewer than 2^64 cells cannot overflow this type.
pub fn step50_integer_lift(
    rows: &[Vec<u64>],
    row_weights: &[u128],
    col_weights: &[u128],
) -> Uint<5> {
    assert_eq!(rows.len(), col_weights.len(), "one bit column per weight");
    row_weights
        .len()
        .checked_mul(col_weights.len())
        .expect("lift cell count exceeds usize");
    assert!(
        rows.iter()
            .all(|words| words.len() == row_weights.len().div_ceil(64)),
        "packed column has the wrong public length"
    );

    let _scope = tracing::info_span!("step5_0:integer_lift").entered();
    let digits: Vec<[u32; 4]> = row_weights
        .iter()
        .map(|&weight| core::array::from_fn(|i| (weight >> (32 * i)) as u32))
        .collect();
    let tile_term = |(tile, columns): (usize, &[Vec<u64>])| {
        lift_column_tile(
            columns,
            &digits,
            &col_weights[tile * LIFT_COLUMNS..][..columns.len()],
        )
    };
    #[cfg(feature = "parallel")]
    {
        rows.par_chunks(LIFT_COLUMNS)
            .enumerate()
            .map(tile_term)
            .reduce(|| Uint::ZERO, |left, right| left.wrapping_add(&right))
    }
    #[cfg(not(feature = "parallel"))]
    {
        rows.chunks(LIFT_COLUMNS)
            .enumerate()
            .map(tile_term)
            .fold(Uint::ZERO, |left, right| left.wrapping_add(&right))
    }
}

const LIFT_COLUMNS: usize = 4;
const LIFT_ROWS: usize = 4096;

/// Each digit sum is at most 4096*(2^32-1). Carry propagation therefore
/// fits in u64; only completed chunks are merged into the exact Uint<3>.
fn lift_column_tile(columns: &[Vec<u64>], weights: &[[u32; 4]], col_weights: &[u128]) -> Uint<5> {
    let mut sums = [Uint::<3>::ZERO; LIFT_COLUMNS];
    for (chunk_index, chunk) in weights.chunks(LIFT_ROWS).enumerate() {
        let mut digits = [[0u64; 4]; LIFT_COLUMNS];
        for (word_index, word_weights) in chunk.chunks(64).enumerate() {
            let index = chunk_index * (LIFT_ROWS / 64) + word_index;
            let words: [u64; LIFT_COLUMNS] = core::array::from_fn(|column| {
                // The final tile's length is public.
                columns.get(column).map_or(0, |words| words[index])
            });
            for (bit, weight) in word_weights.iter().enumerate() {
                for (sum, &word) in digits.iter_mut().zip(&words) {
                    // Preserve the mask through LLVM's branch reconstruction.
                    let mask = u64::ct_select(&0, &u64::MAX, CtMask::from_lsb(word >> bit));
                    for (acc, &digit) in sum.iter_mut().zip(weight) {
                        *acc += u64::from(digit) & mask;
                    }
                }
            }
        }
        for (sum, digit) in sums.iter_mut().zip(digits) {
            let a = digit[0];
            let b = digit[1] + (a >> 32);
            let c = digit[2] + (b >> 32);
            let d = digit[3] + (c >> 32);
            let chunk_sum = Uint::from_words([
                (a & 0xffff_ffff) | (b << 32),
                (c & 0xffff_ffff) | (d << 32),
                d >> 32,
            ]);
            *sum = sum.wrapping_add(&chunk_sum);
        }
    }
    sums.iter()
        .zip(col_weights)
        .fold(Uint::ZERO, |total, (sum, &weight)| {
            let product = IntegerOps.mul_wide(sum, &Uint::<2>::from(weight));
            total.wrapping_add(product.checked_resize_ct::<5>().value())
        })
}

/// Exclusive magnitude bound `cells * q^2`, with the same five-limb bound.
pub fn step50_mu_prime_bound(cells: usize, q: u128) -> Uint<5> {
    let square = IntegerOps.mul_wide(&Uint::<2>::from(q), &Uint::<2>::from(q));
    let square = *square.checked_resize_ct::<4>().value();
    *IntegerOps
        .mul_wide(&square, &Uint::<1>::from(cells as u64))
        .checked_resize_ct::<5>()
        .value()
}

/// Validate the now-public proof message against its magnitude and mod-Q claim.
pub fn step50_accepts_lift(mu_prime: &Uint<5>, mu: u128, q: u128, cells: usize) -> bool {
    mu_prime
        .ct_lt(&step50_mu_prime_bound(cells, q))
        .declassify()
        && lift_mod_u128(mu_prime, q) == mu
}

pub fn step50_reduce(
    row_weights: &[u128],
    col_weights: &[u128],
    mu_prime: &Uint<5>,
    q_prime: u128,
) -> (Vec<u128>, Vec<u128>, u128) {
    let divisor = PreparedDivisor::new(Uint::<2>::from(q_prime)).expect("nonzero public modulus");
    let reduce = |weights: &[u128]| {
        weights
            .iter()
            .map(|&w| {
                let (_, r) = divisor.div_rem_ct(&Uint::<2>::from(w));
                u128::from(r)
            })
            .collect()
    };
    let (_, claimed) = divisor.div_rem_ct(mu_prime);
    (
        reduce(row_weights),
        reduce(col_weights),
        u128::from(claimed),
    )
}

pub fn encode_integer_lift(value: &Uint<5>) -> [u8; 40] {
    let mut bytes = [0; 40];
    IntegerOps.encode_into(value, &mut bytes);
    bytes
}

fn lift_mod_u128(value: &Uint<5>, modulus: u128) -> u128 {
    let divisor = PreparedDivisor::new(Uint::<2>::from(modulus)).expect("nonzero public modulus");
    u128::from(divisor.div_rem_ct(value).1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use num_traits::Zero;

    fn oracle(value: &Uint<5>) -> BigUint {
        BigUint::from_bytes_le(&encode_integer_lift(value))
    }
    fn biguint_mod_u128(value: &BigUint, modulus: u128) -> u128 {
        let digits = (value % BigUint::from(modulus)).to_u64_digits();
        digits.first().copied().unwrap_or(0) as u128
            | ((digits.get(1).copied().unwrap_or(0) as u128) << 64)
    }

    fn dense_lift(bits: &[Vec<u64>], rw: &[u128], cw: &[u128]) -> BigUint {
        let mut total = BigUint::zero();
        for (column, words) in bits.iter().enumerate() {
            for row in 0..rw.len() {
                if (words[row / 64] >> (row % 64)) & 1 == 1 {
                    total += BigUint::from(rw[row]) * BigUint::from(cw[column]);
                }
            }
        }
        total
    }

    #[test]
    fn digit_lift_matches_oracle_at_all_boundaries() {
        for row_count in [0usize, 1, 63, 64, 65, 4095, 4096, 4097, 8192] {
            for column_count in [0, 1, 3, 4, 5] {
                let rw: Vec<_> = (0..row_count)
                    .map(|i| match i % 4 {
                        0 => u128::MAX,
                        1 => 0,
                        2 => 1 << 127,
                        _ => (i as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835),
                    })
                    .collect();
                let cw: Vec<_> = (0..column_count).map(|i| u128::MAX - i as u128).collect();
                for pattern in [0, u64::MAX, 0xaaaa_5555_8123_4567] {
                    let bits = vec![vec![pattern; row_count.div_ceil(64)]; column_count];
                    let expected = dense_lift(&bits, &rw, &cw);
                    assert_eq!(
                        oracle(&step50_integer_lift(&bits, &rw, &cw)),
                        expected,
                        "rows={row_count}, columns={column_count}, pattern={pattern}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn digit_lift_is_identical_across_worker_counts() {
        let rw: Vec<_> = (0..8193u128).map(|i| u128::MAX - i).collect();
        let cw = vec![u128::MAX; 9];
        let bits = vec![vec![0x1234_5678_9abc_def0; rw.len().div_ceil(64)]; cw.len()];
        let prove = || step50_integer_lift(&bits, &rw, &cw);
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(prove);
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap()
            .install(prove);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn integer_lift_matches_the_dense_reference_and_reduces_consistently() {
        let q: u128 = (1u128 << 127) + 0x2d; // any odd 128-bit modulus works here
        let rw: Vec<u128> = (0..128u128)
            .map(|index| (q - 1).wrapping_sub(index * 0x1234_5678_9abc_def1) % q)
            .collect();
        let cw: Vec<u128> = (0..4u128).map(|index| (index * 0x0fed_cba9) % q).collect();
        let rows: Vec<Vec<u64>> = (0..4)
            .map(|column| vec![0x8000_0001_0000_0111u64 << column, u64::MAX >> column])
            .collect();

        let lifted = step50_integer_lift(&rows, &rw, &cw);
        assert_eq!(oracle(&lifted), dense_lift(&rows, &rw, &cw));

        let mu = lift_mod_u128(&lifted, q);
        assert!(step50_accepts_lift(&lifted, mu, q, rw.len() * cw.len()));
        assert!(!step50_accepts_lift(
            &lifted,
            mu ^ 1,
            q,
            rw.len() * cw.len()
        ));
        assert!(!step50_accepts_lift(
            &lifted.wrapping_add(&step50_mu_prime_bound(rw.len() * cw.len(), q)),
            mu,
            q,
            rw.len() * cw.len(),
        ));

        let q_prime: u128 = (1u128 << 112) + 0x1d; // odd; primality irrelevant here
        let (rw_reduced, cw_reduced, claimed) = step50_reduce(&rw, &cw, &lifted, q_prime);
        assert!(rw_reduced.iter().all(|&w| w < q_prime));
        assert!(cw_reduced.iter().all(|&w| w < q_prime));
        assert_eq!(
            claimed,
            biguint_mod_u128(&dense_lift(&rows, &rw, &cw), q_prime)
        );
        // The reduced-factor tensor evaluates to the reduced claim mod q'.
        let reduced_eval = dense_lift(&rows, &rw_reduced, &cw_reduced);
        assert_eq!(biguint_mod_u128(&reduced_eval, q_prime), claimed);
    }
}
