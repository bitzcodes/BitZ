//! P-256 ECDSA verification for an in-circuit SHA-256 digest.

use std::array;

use crate::Circuit;
use crate::p256::{VERIFY_DIGEST_INPUT_BITS, VERIFY_DIGEST_WITNESS_BITS, verify_digest_circuit};
use crate::sha256::{SHA256_2KB_MESSAGE_BITS, SHA256_2KB_WITNESS_BITS, sha256_2kb_circuit};

/// Hashes a block-aligned message and verifies its P-256 signature.
///
/// The getter supplies message bits in stream order. `p256_inputs` contains
/// Q.x, Q.y, r, s, r^-1 and s^-1 in little-endian word order. SHA adds the
/// final padding block; thus `message_bits / 512 + 1` compressions execute.
pub fn verify_block_aligned_message_circuit<CS: Circuit>(
    circuit: &mut CS,
    message_bits: usize,
    message_bit: impl Fn(usize) -> CS::Bool,
    p256_inputs: &[CS::Bool; VERIFY_2KB_P256_INPUT_BITS],
) {
    let digest = crate::sha256::sha256_block_aligned_circuit(circuit, message_bits, message_bit);
    let inputs = array::from_fn(|index| {
        if index < 256 {
            digest[255 - index].clone()
        } else {
            p256_inputs[index - 256].clone()
        }
    });
    verify_digest_circuit(circuit, &inputs);
}

/// Number of non-digest P-256 inputs: Q.x, Q.y, r, s, r^-1, and s^-1.
pub const VERIFY_2KB_P256_INPUT_BITS: usize = VERIFY_DIGEST_INPUT_BITS - 256;

/// Number of Boolean inputs: a 2 KiB message followed by the six P-256 words.
pub const VERIFY_2KB_INPUT_BITS: usize = SHA256_2KB_MESSAGE_BITS + VERIFY_2KB_P256_INPUT_BITS;

/// Total Boolean witness size, including the message and P-256 inputs.
///
/// The SHA-256 digest is wired directly into the P-256 verifier, so it does
/// not allocate another 256 Boolean inputs.
pub const VERIFY_2KB_WITNESS_BITS: usize =
    SHA256_2KB_WITNESS_BITS + VERIFY_DIGEST_WITNESS_BITS - 256;

/// Number of packed `M * w` bits, including the implicit constant one.
pub const VERIFY_2KB_INTEGER_WITNESS_BITS: usize = 1_890_711;

/// Number of rank-1 constraints in the composed circuit.
pub const VERIFY_2KB_R1CS_ROWS: usize = 13_133;

/// Hashes a 2 KiB message and verifies its P-256 ECDSA signature.
///
/// Inputs are the message in conventional stream order, followed by Q.x,
/// Q.y, r, s, r^-1, and s^-1 as little-endian 256-bit words. SHA-256 returns
/// its digest in conventional big-endian stream order, so its bits are
/// reversed when wired into the verifier's little-endian digest word.
pub fn verify_2kb_message_circuit<CS: Circuit>(
    circuit: &mut CS,
    inputs: &[CS::Bool; VERIFY_2KB_INPUT_BITS],
) {
    let (message, p256_inputs) = inputs.split_at(SHA256_2KB_MESSAGE_BITS);
    let message: &[CS::Bool; SHA256_2KB_MESSAGE_BITS] = message
        .try_into()
        .unwrap_or_else(|_| unreachable!("message length is fixed"));
    let digest = sha256_2kb_circuit(circuit, message);

    let verifier_inputs: [CS::Bool; VERIFY_DIGEST_INPUT_BITS] = array::from_fn(|index| {
        if index < 256 {
            digest[255 - index].clone()
        } else {
            p256_inputs[index - 256].clone()
        }
    });
    verify_digest_circuit(circuit, &verifier_inputs);
}

#[cfg(test)]
mod tests {
    use num_bigint::{BigInt, BigUint, Sign};
    use num_traits::{One, Zero};

    use super::*;

    use crate::stats::{Dummy, LeanStats, Stats};
    use crate::witgen::ProductWitgen;

    fn from_hex(value: &[u8]) -> BigUint {
        BigUint::parse_bytes(value, 16).unwrap()
    }

    fn inverse(value: &BigUint, modulus: &BigUint) -> BigUint {
        let mut t = BigInt::zero();
        let mut new_t = BigInt::one();
        let mut r = BigInt::from(modulus.clone());
        let mut new_r = BigInt::from(value.clone());
        while !new_r.is_zero() {
            let quotient = &r / &new_r;
            (t, new_t) = (new_t.clone(), t - &quotient * new_t);
            (r, new_r) = (new_r.clone(), r - quotient * new_r);
        }
        assert_eq!(r, BigInt::one());
        let modulus = BigInt::from(modulus.clone());
        let mut t = t % &modulus;
        if t.sign() == Sign::Minus {
            t += modulus;
        }
        t.to_biguint().unwrap()
    }

    fn valid_input() -> Box<[bool; VERIFY_2KB_INPUT_BITS]> {
        let modulus = from_hex(b"ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
        let digest = from_hex(b"10fc3c51a152e90e5b90319b601d92ccf37290ef53c35ff92507687d8a911a08");
        let gx = from_hex(b"6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");
        let gy = from_hex(b"4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5");
        let s = (&digest + &gx) % &modulus;
        let values = [
            gx.clone(),
            gy,
            gx.clone(),
            s.clone(),
            inverse(&gx, &modulus),
            inverse(&s, &modulus),
        ];
        let bits: Box<[bool]> = (0..VERIFY_2KB_INPUT_BITS)
            .map(|index| {
                if index < SHA256_2KB_MESSAGE_BITS {
                    let byte = (index / 8) as u8;
                    byte & (1 << (7 - index % 8)) != 0
                } else {
                    let index = index - SHA256_2KB_MESSAGE_BITS;
                    values[index / 256].bit((index % 256) as u64)
                }
            })
            .collect();
        bits.try_into().unwrap()
    }

    fn stored_bigint(value: &[u64]) -> BigInt {
        let bytes = value
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        BigInt::from_signed_bytes_le(&bytes)
    }

    #[test]
    fn dimensions_are_the_sum_without_duplicate_digest_inputs() {
        let mut stats = Stats::new(VERIFY_2KB_INPUT_BITS);
        verify_2kb_message_circuit(&mut stats, &[Dummy; VERIFY_2KB_INPUT_BITS]);
        assert_eq!(
            stats.lean_stats(),
            LeanStats {
                m_rows: VERIFY_2KB_INTEGER_WITNESS_BITS,
                m_cols: VERIFY_2KB_WITNESS_BITS + 1,
                r1cs_rows: VERIFY_2KB_R1CS_ROWS,
            }
        );
    }

    #[test]
    fn valid_signature_satisfies_the_composed_circuit() {
        crate::p256::prepare();
        let inputs = valid_input();
        let mut witgen =
            ProductWitgen::with_inputs_and_capacity(inputs.as_ref(), VERIFY_2KB_WITNESS_BITS);
        verify_2kb_message_circuit(&mut witgen, &inputs);
        assert_eq!(witgen.witness().bit_len(), VERIFY_2KB_WITNESS_BITS);
        assert_eq!(
            witgen.integer_witness().bit_len(),
            VERIFY_2KB_INTEGER_WITNESS_BITS
        );
        assert_eq!(witgen.products().a_mw.len(), VERIFY_2KB_R1CS_ROWS);
        assert_eq!(witgen.products().b_mw.len(), VERIFY_2KB_R1CS_ROWS);
        assert_eq!(witgen.products().c_mw.len(), VERIFY_2KB_R1CS_ROWS);
        for ((a, b), c) in witgen
            .products()
            .a_mw
            .iter()
            .zip(witgen.products().b_mw.iter())
            .zip(witgen.products().c_mw.iter())
        {
            assert_eq!(stored_bigint(a) * stored_bigint(b), stored_bigint(c));
        }
    }
}
