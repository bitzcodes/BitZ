// Copyright 2026 The Binius Developers
//! Fixed-length SHA-256 chain with standard P-256 signature verification.
//!
//! Public inputs are `(i, Qx, Qy, r, s)`: one u64 exponent and four integers
//! encoded as four little-endian u64 limbs each. The private message has
//! 64*(2^i-1) bytes; padding
//! adds the final compression. A circuit is reusable across all instances at i.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use crate::{bignum::BigUint, p256::verify_digest, sha256::sha256_fixed};

pub const PROFILE: &str = "sha256-chain-p256/standard/v1";

/// The fixed relation and its input wires; construction never takes an instance.
pub struct Sha256Ecdsa {
	log_compressions: u8,
	exponent: Wire,
	message: Vec<Wire>,
	public: [BigUint; 4],
}

pub fn message_len(log_compressions: u8) -> Result<usize, String> {
	if !(3..=16).contains(&log_compressions) {
		return Err("log_compressions must be in 3..=16".into());
	}
	1usize
		.checked_shl(u32::from(log_compressions))
		.and_then(|n| n.checked_sub(1))
		.and_then(|n| n.checked_mul(64))
		.ok_or_else(|| "message length overflow".into())
}

/// Encodes a canonical 32-byte BE integer as four LE u64 limbs.
pub fn limbs(bytes: &[u8; 32]) -> [u64; 4] {
	std::array::from_fn(|i| {
		u64::from_be_bytes(
			bytes[24 - i * 8..32 - i * 8]
				.try_into()
				.expect("eight-byte limb"),
		)
	})
}

/// Encodes the verifier's expected public statement, independently of a witness.
pub fn public_words(i: u8, qx: &[u8; 32], qy: &[u8; 32], r: &[u8; 32], s: &[u8; 32]) -> Vec<Word> {
	std::iter::once(Word::from_u64(u64::from(i)))
		.chain(
			[qx, qy, r, s]
				.into_iter()
				.flat_map(limbs)
				.map(Word::from_u64),
		)
		.collect()
}

impl Sha256Ecdsa {
	pub fn new(b: &CircuitBuilder, log_compressions: u8) -> Result<Self, String> {
		let len = message_len(log_compressions)?;
		let exponent = b.add_inout();
		b.assert_eq("public exponent", exponent, b.add_constant_64(u64::from(log_compressions)));
		let public = std::array::from_fn(|_| BigUint::new_inout(b, 4));
		let message: Vec<_> = (0..len / 4).map(|_| b.add_witness()).collect();
		let zero = b.add_constant(Word::ZERO);
		for (i, &word) in message.iter().enumerate() {
			b.assert_eq(format!("message word {i} is u32"), b.shr(word, 32), zero);
		}
		let digest = sha256_fixed(&b.subcircuit("SHA-256 chain"), &message, len);
		let digest = BigUint {
			limbs: (0..4)
				.map(|i| b.bxor(digest[7 - 2 * i], b.shl(digest[6 - 2 * i], 32)))
				.collect(),
		};
		verify_digest(
			&b.subcircuit("P-256 verification"),
			&digest,
			&public[0],
			&public[1],
			&public[2],
			&public[3],
		);
		Ok(Self {
			log_compressions,
			exponent,
			message,
			public,
		})
	}

	/// Assigns only instance inputs. Circuit execution computes the digest and arithmetic hints.
	pub fn populate(
		&self,
		w: &mut WitnessFiller<'_>,
		message: &[u8],
		qx: &[u8; 32],
		qy: &[u8; 32],
		r: &[u8; 32],
		s: &[u8; 32],
	) -> Result<(), String> {
		if message.len() != message_len(self.log_compressions)? {
			return Err("incorrect SHA-chain message length".into());
		}
		w[self.exponent] = Word::from_u64(u64::from(self.log_compressions));
		for (input, bytes) in self.public.iter().zip([qx, qy, r, s]) {
			input.populate_limbs(w, &limbs(bytes));
		}
		for (&wire, bytes) in self.message.iter().zip(message.chunks_exact(4)) {
			w[wire] = Word::from_u64(u64::from(u32::from_be_bytes(
				bytes.try_into().expect("four-byte chunk"),
			)));
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use p256::ecdsa::{Signature, SigningKey, signature::Signer};

	use super::*;

	#[test]
	fn private_message_is_bound_and_range_constrained() {
		let builder = CircuitBuilder::new();
		let relation = Sha256Ecdsa::new(&builder, 3).unwrap();
		let circuit = builder.build();
		let key = SigningKey::from_bytes((&[17u8; 32]).into()).unwrap();
		let q = key.verifying_key().to_encoded_point(false);
		let qx = (*q.x().unwrap()).into();
		let qy = (*q.y().unwrap()).into();
		for byte in [0u8, 255] {
			let message = vec![byte; message_len(3).unwrap()];
			let sig: Signature = key.sign(&message);
			let (r, s) = sig.split_bytes();
			let (r, s) = (r.into(), s.into());
			let mut w = circuit.new_witness_filler();
			relation
				.populate(&mut w, &message, &qx, &qy, &r, &s)
				.unwrap();
			circuit.populate_wire_witness(&mut w).unwrap();
			let valid = w.into_value_vec();
			circuit.constraint_system().verify(&valid).unwrap();
			for bit in [0, 32, 63] {
				let mut bad = valid.clone();
				let index = circuit.witness_index(relation.message[0]);
				bad[index] = Word(bad[index].0 ^ (1u64 << bit));
				assert!(circuit.constraint_system().verify(&bad).is_err());

				let mut w = circuit.new_witness_filler();
				relation
					.populate(&mut w, &message, &qx, &qy, &r, &s)
					.unwrap();
				w[relation.message[0]] = Word(w[relation.message[0]].0 ^ (1u64 << bit));
				assert!(circuit.populate_wire_witness(&mut w).is_err());
			}
			assert!(
				relation
					.populate(&mut circuit.new_witness_filler(), &message[1..], &qx, &qy, &r, &s)
					.is_err()
			);
		}
	}

	#[test]
	fn checked_lengths_and_encodings() {
		assert_eq!(message_len(3).unwrap(), 448);
		assert_eq!(message_len(11).unwrap(), 131_008);
		assert_eq!(message_len(16).unwrap(), 4_194_240);
		for i in [0, 2, 17, 255] {
			assert!(message_len(i).is_err());
		}
		let bytes = std::array::from_fn(|i| i as u8);
		assert_eq!(
			limbs(&bytes),
			[
				0x18191a1b1c1d1e1f,
				0x1011121314151617,
				0x08090a0b0c0d0e0f,
				0x0001020304050607
			]
		);
		assert_eq!(public_words(3, &bytes, &bytes, &bytes, &bytes).len(), 17);
	}
}
