// Copyright 2026 The Binius Developers
//! Standard P-256 ECDSA verification over a private SHA-256 digest.
//!
//! Integers are four little-endian u64 limbs. Public coordinates and signature
//! scalars are canonical; both s and n-s are accepted. The complete affine
//! formulas use a=-3 and four-bit joint multiplication, without a curve-specific
//! endomorphism. Every division, including inactive point-addition cases, is
//! constrained. This implements the SHA-chain comparison's standard ECDSA relation.

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::bignum::{
	BigUint, PseudoMersennePrimeField, assert_eq, biguint_eq, biguint_lt, select, sub,
};

pub const MODULUS: &str = "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff";
pub const ORDER: &str = "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";
const CURVE_B: &str = "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b";
const GENERATOR_X: &str = "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296";
const GENERATOR_Y: &str = "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5";

fn integer(hex: &str) -> num_bigint::BigUint {
	num_bigint::BigUint::parse_bytes(hex.as_bytes(), 16).expect("P-256 constant")
}

fn constant(b: &CircuitBuilder, value: &num_bigint::BigUint) -> BigUint {
	BigUint::new_constant(b, value).zero_extend(b, 4)
}

fn field(b: &CircuitBuilder, modulus: &str) -> PseudoMersennePrimeField {
	let complement = (num_bigint::BigUint::from(1u8) << 256usize) - integer(modulus);
	PseudoMersennePrimeField::new(b, 256, &complement.to_u64_digits())
}

#[derive(Clone)]
struct Point {
	x: BigUint,
	y: BigUint,
	infinity: Wire,
}

struct Curve {
	fp: PseudoMersennePrimeField,
	zero: BigUint,
	one: BigUint,
	three: BigUint,
}

impl Curve {
	fn new(b: &CircuitBuilder) -> Self {
		Self {
			fp: field(b, MODULUS),
			zero: constant(b, &0u8.into()),
			one: constant(b, &1u8.into()),
			three: constant(b, &3u8.into()),
		}
	}

	fn infinity(&self, b: &CircuitBuilder) -> Point {
		Point {
			x: self.zero.clone(),
			y: self.zero.clone(),
			infinity: b.add_constant(Word::ALL_ONE),
		}
	}

	fn select(&self, b: &CircuitBuilder, cond: Wire, yes: &Point, no: &Point) -> Point {
		Point {
			x: select(b, cond, &yes.x, &no.x),
			y: select(b, cond, &yes.y, &no.y),
			infinity: b.select(cond, yes.infinity, no.infinity),
		}
	}

	fn assert_on_curve(&self, b: &CircuitBuilder, q: &Point) {
		b.assert_true("Q.x < p", biguint_lt(b, &q.x, self.fp.modulus()));
		b.assert_true("Q.y < p", biguint_lt(b, &q.y, self.fp.modulus()));
		b.assert_false("Q is finite", q.infinity);
		let x3 = self.fp.mul(b, &self.fp.square(b, &q.x), &q.x);
		let three_x = self.fp.add(b, &self.fp.add(b, &q.x, &q.x), &q.x);
		let rhs = self
			.fp
			.add(b, &self.fp.sub(b, &x3, &three_x), &constant(b, &integer(CURVE_B)));
		assert_eq(b, "P-256 curve equation", &self.fp.square(b, &q.y), &rhs);
	}

	fn add(&self, b: &CircuitBuilder, p: &Point, q: &Point) -> Point {
		let f = &self.fp;
		let same_x = biguint_eq(b, &p.x, &q.x);
		let same_y = biguint_eq(b, &p.y, &q.y);
		let doubling = b.band(same_x, same_y);
		let any_infinity = b.bor(p.infinity, q.infinity);
		let active = b.band(
			b.bnot(any_infinity),
			b.bor(b.bnot(same_x), b.band(doubling, b.bnot(p.y.is_zero(b)))),
		);
		let x2 = f.square(b, &p.x);
		let double_num = f.sub(b, &f.add(b, &f.add(b, &x2, &x2), &x2), &self.three);
		let numerator = select(b, doubling, &double_num, &f.sub(b, &q.y, &p.y));
		let denominator = select(b, doubling, &f.add(b, &p.y, &p.y), &f.sub(b, &q.x, &p.x));
		// Inactive cases use the exact division 0/1; no unbound slope or zero divisor.
		let numerator = select(b, active, &numerator, &self.zero);
		let denominator = select(b, active, &denominator, &self.one);
		let slope = f.div(b, &numerator, &denominator, b.add_constant(Word::ALL_ONE));
		let x = f.sub(b, &f.sub(b, &f.square(b, &slope), &p.x), &q.x);
		let y = f.sub(b, &f.mul(b, &slope, &f.sub(b, &p.x, &x)), &p.y);
		let sum = Point {
			x,
			y,
			infinity: b.bnot(active),
		};
		let sum = self.select(b, q.infinity, p, &sum);
		let sum = self.select(b, p.infinity, q, &sum);
		self.select(b, sum.infinity, &self.infinity(b), &sum)
	}

	fn table(&self, b: &CircuitBuilder, q: &Point) -> Vec<Point> {
		let mut table = vec![self.infinity(b), q.clone()];
		for i in 2..16 {
			table.push(self.add(&b.subcircuit(format!("Q table {i}")), &table[i - 1], q));
		}
		table
	}

	fn generator_table(&self, b: &CircuitBuilder) -> Vec<Point> {
		let p = integer(MODULUS);
		let gx = integer(GENERATOR_X);
		let gy = integer(GENERATOR_Y);
		let mut x = gx.clone();
		let mut y = gy.clone();
		let mut table = vec![self.infinity(b)];
		for i in 1..16 {
			table.push(Point {
				x: constant(b, &x),
				y: constant(b, &y),
				infinity: b.add_constant(Word::ZERO),
			});
			if i == 15 {
				break;
			}
			let (numerator, denominator) = if i == 1 {
				((3u8 * &x * &x + &p - 3u8) % &p, (2u8 * &y) % &p)
			} else {
				((&gy + &p - &y) % &p, (&gx + &p - &x) % &p)
			};
			let slope = numerator * denominator.modpow(&(&p - 2u8), &p) % &p;
			let next_x = (&slope * &slope + 2u8 * &p - &x - &gx) % &p;
			let next_y = (&slope * (&x + &p - &next_x) + &p - &y) % &p;
			x = next_x;
			y = next_y;
		}
		table
	}

	fn lookup(
		&self,
		b: &CircuitBuilder,
		table: &[Point],
		scalar: &BigUint,
		window: usize,
	) -> Point {
		let mut values = table.to_vec();
		for bit in 0..4 {
			let position = 4 * window + bit;
			let flag = b.shl(b.shr(scalar.limbs[position / 64], (position % 64) as u32), 63);
			values = values
				.chunks_exact(2)
				.map(|pair| self.select(b, flag, &pair[1], &pair[0]))
				.collect();
		}
		values.pop().expect("nonempty table")
	}

	fn joint_mul(&self, b: &CircuitBuilder, u1: &BigUint, u2: &BigUint, q: &Point) -> Point {
		let g_table = self.generator_table(b);
		let q_table = self.table(b, q);
		let mut sum = self.infinity(b);
		for window in (0..64).rev() {
			let b = b.subcircuit(format!("MSM window {window}"));
			for _ in 0..4 {
				sum = self.add(&b, &sum, &sum);
			}
			sum = self.add(&b, &sum, &self.lookup(&b, &g_table, u1, window));
			sum = self.add(&b, &sum, &self.lookup(&b, &q_table, u2, window));
		}
		sum
	}
}

/// Reduces a 256-bit integer modulo n. Since 2^256 < 2n, one subtraction suffices.
fn reduce_scalar(b: &CircuitBuilder, value: &BigUint, modulus: &BigUint) -> BigUint {
	assert_eq!(value.limbs.len(), 4);
	let subtract = b.bnot(biguint_lt(b, value, modulus));
	let result = sub(b, value, &modulus.zero_unless(b, subtract));
	b.assert_true("reduced scalar < n", biguint_lt(b, &result, modulus));
	result
}

/// Constrains standard P-256 verification; the digest and all inputs are four LE limbs.
///
/// The digest must be wired from the message hash by the caller. qx, qy, r, s
/// may be public inputs. This gadget accepts high-s signatures and rejects R=O.
pub fn verify_digest(
	b: &CircuitBuilder,
	digest: &BigUint,
	qx: &BigUint,
	qy: &BigUint,
	r: &BigUint,
	s: &BigUint,
) {
	for value in [digest, qx, qy, r, s] {
		assert_eq!(value.limbs.len(), 4);
	}
	let curve = Curve::new(b);
	let scalar = field(b, ORDER);
	let q = Point {
		x: qx.clone(),
		y: qy.clone(),
		infinity: b.add_constant(Word::ZERO),
	};
	curve.assert_on_curve(b, &q);
	for (name, value) in [("r", r), ("s", s)] {
		b.assert_true(format!("{name} < n"), biguint_lt(b, value, scalar.modulus()));
		b.assert_false(format!("{name} != 0"), value.is_zero(b));
	}
	let z = reduce_scalar(b, digest, scalar.modulus());
	let yes = b.add_constant(Word::ALL_ONE);
	let u1 = scalar.div(b, &z, s, yes);
	let u2 = scalar.div(b, r, s, yes);
	let result = curve.joint_mul(b, &u1, &u2, &q);
	b.assert_false("ECDSA result is finite", result.infinity);
	assert_eq(b, "ECDSA x mod n = r", &reduce_scalar(b, &result.x, scalar.modulus()), r);
}

#[cfg(test)]
mod tests {
	use p256::ecdsa::{
		Signature, SigningKey,
		signature::hazmat::{PrehashSigner, PrehashVerifier},
	};

	use super::*;
	use crate::sha256_ecdsa::limbs;

	fn bytes(value: &num_bigint::BigUint) -> [u8; 32] {
		let raw = value.to_bytes_be();
		let mut out = [0; 32];
		out[32 - raw.len()..].copy_from_slice(&raw);
		out
	}

	#[test]
	fn digest_boundaries_both_s_forms_and_invalid_inputs() {
		let b = CircuitBuilder::new();
		let inputs: [BigUint; 5] = std::array::from_fn(|_| BigUint::new_inout(&b, 4));
		verify_digest(&b, &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]);
		let circuit = b.build();
		let key = SigningKey::from_bytes((&[17u8; 32]).into()).unwrap();
		let q = key.verifying_key().to_encoded_point(false);
		let n = integer(ORDER);
		let digests = [
			[0; 32],
			bytes(&(&n - 1u8)),
			bytes(&n),
			bytes(&(&n + 1u8)),
			[255; 32],
		];
		for digest in digests {
			let sig: Signature = key.sign_prehash(&digest).unwrap();
			let (r, s) = sig.split_bytes();
			let alternate = bytes(&(&n - num_bigint::BigUint::from_bytes_be(&s)));
			for s in [<[u8; 32]>::from(s), alternate] {
				key.verifying_key()
					.verify_prehash(&digest, &Signature::from_scalars(r, s).unwrap())
					.unwrap();
				let values = [
					digest,
					(*q.x().unwrap()).into(),
					(*q.y().unwrap()).into(),
					r.into(),
					s,
				];
				let mut w = circuit.new_witness_filler();
				for (input, value) in inputs.iter().zip(&values) {
					input.populate_limbs(&mut w, &limbs(value));
				}
				circuit.populate_wire_witness(&mut w).unwrap();
				let valid = w.into_value_vec();
				circuit.constraint_system().verify(&valid).unwrap();
				// Mutate already-populated public values: fixture validation cannot mask these
				// failures.
				for index in 0..5 {
					let mut bad = valid.clone();
					let wire = circuit.witness_index(inputs[index].limbs[0]);
					bad[wire] = Word::from_u64(bad[wire].0 ^ 1);
					assert!(circuit.constraint_system().verify(&bad).is_err(), "input {index}");
				}
			}
		}
		for (index, value) in [
			(3, [0; 32]),
			(4, [0; 32]),
			(3, bytes(&n)),
			(4, bytes(&n)),
			(1, bytes(&integer(MODULUS))),
		] {
			let digest = [9; 32];
			let sig: Signature = key.sign_prehash(&digest).unwrap();
			let (r, s) = sig.split_bytes();
			let mut values = [
				digest,
				(*q.x().unwrap()).into(),
				(*q.y().unwrap()).into(),
				r.into(),
				s.into(),
			];
			values[index] = value;
			let mut w = circuit.new_witness_filler();
			for (input, value) in inputs.iter().zip(&values) {
				input.populate_limbs(&mut w, &limbs(value));
			}
			assert!(circuit.populate_wire_witness(&mut w).is_err());
		}
	}

	#[test]
	fn verification_rejects_infinite_result() {
		let b = CircuitBuilder::new();
		let inputs: [BigUint; 5] = std::array::from_fn(|_| BigUint::new_inout(&b, 4));
		verify_digest(&b, &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]);
		let circuit = b.build();
		// z=r=s=1 and Q=-G give R=G-G=O, which ECDSA must reject.
		let one = bytes(&1u8.into());
		let values = [
			one,
			bytes(&integer(GENERATOR_X)),
			bytes(&(integer(MODULUS) - integer(GENERATOR_Y))),
			one,
			one,
		];
		let mut w = circuit.new_witness_filler();
		for (input, value) in inputs.iter().zip(&values) {
			input.populate_limbs(&mut w, &limbs(value));
		}
		assert!(circuit.populate_wire_witness(&mut w).is_err());
	}

	#[test]
	fn complete_addition_handles_identity_doubling_and_opposites() {
		use p256::{ProjectivePoint, elliptic_curve::sec1::ToEncodedPoint};
		let b = CircuitBuilder::new();
		let curve = Curve::new(&b);
		let points: [Point; 2] = std::array::from_fn(|_| Point {
			x: BigUint::new_inout(&b, 4),
			y: BigUint::new_inout(&b, 4),
			infinity: b.add_inout(),
		});
		let sum = curve.add(&b, &points[0], &points[1]);
		for &wire in sum
			.x
			.limbs
			.iter()
			.chain(&sum.y.limbs)
			.chain(std::iter::once(&sum.infinity))
		{
			b.mark_inout(wire);
		}
		let circuit = b.build();
		let g = ProjectivePoint::GENERATOR;
		let identity = ProjectivePoint::IDENTITY;
		let coordinates = |p: ProjectivePoint| -> ([u8; 32], [u8; 32], Word) {
			if p == identity {
				([0; 32], [0; 32], Word::ALL_ONE)
			} else {
				let p = p.to_affine().to_encoded_point(false);
				((*p.x().unwrap()).into(), (*p.y().unwrap()).into(), Word::ZERO)
			}
		};
		for (p, q) in [
			(g, g),
			(g, -g),
			(identity, g),
			(g, identity),
			(identity, identity),
			(g, g + g),
		] {
			let mut w = circuit.new_witness_filler();
			for (wire, value) in points.iter().zip([p, q]) {
				let (x, y, infinity) = coordinates(value);
				wire.x.populate_limbs(&mut w, &limbs(&x));
				wire.y.populate_limbs(&mut w, &limbs(&y));
				w[wire.infinity] = infinity;
			}
			circuit.populate_wire_witness(&mut w).unwrap();
			let (x, y, infinity) = coordinates(p + q);
			assert_eq!(w[sum.infinity].0 >> 63, infinity.0 >> 63);
			for (wire, expected) in sum
				.x
				.limbs
				.iter()
				.chain(&sum.y.limbs)
				.zip(limbs(&x).into_iter().chain(limbs(&y)))
			{
				assert_eq!(w[*wire].0, expected);
			}
			let valid = w.into_value_vec();
			circuit.constraint_system().verify(&valid).unwrap();
			let mut bad = valid.clone();
			let flag = circuit.witness_index(sum.infinity);
			bad[flag] = Word(bad[flag].0 ^ (1u64 << 63));
			assert!(circuit.constraint_system().verify(&bad).is_err());
		}
	}

	#[test]
	fn scalar_reduction_and_division_hint_are_constrained() {
		let b = CircuitBuilder::new();
		let f = field(&b, ORDER);
		let input = BigUint::new_inout(&b, 4);
		let reduced = reduce_scalar(&b, &input, f.modulus());
		let quotient =
			f.div(&b, &reduced, &constant(&b, &5u8.into()), b.add_constant(Word::ALL_ONE));
		for &wire in &quotient.limbs {
			b.force_commit(wire);
		}
		let circuit = b.build();
		for input_value in [integer(ORDER) - 1u8, integer(ORDER), integer(MODULUS) - 1u8] {
			let mut w = circuit.new_witness_filler();
			input.populate_limbs(&mut w, &limbs(&bytes(&input_value)));
			circuit.populate_wire_witness(&mut w).unwrap();
			let mut values = w.into_value_vec();
			circuit.constraint_system().verify(&values).unwrap();
			let wire = circuit.witness_index(quotient.limbs[0]);
			values[wire] = Word(values[wire].0 ^ 1);
			assert!(circuit.constraint_system().verify(&values).is_err());
		}
	}
}
