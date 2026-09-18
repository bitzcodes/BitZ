//! Binius counterpart of the integer relation `x * y = z + 2^32 * w`.
use binius_frontend::{CircuitBuilder, Wire};

/// Allocate `[x, y, z, w]` and constrain all four words to 32 bits.
///
/// `z` is the multiplication result modulo `2^32`; `w` is its overflow word.
pub fn add_u32_mul_mod32(builder: &CircuitBuilder) -> [Wire; 4] {
    let words = std::array::from_fn(|_| builder.add_witness());
    for (word, name) in words.into_iter().zip(["u32_x", "u32_y", "u32_z", "u32_w"]) {
        builder.assert_zero(name, builder.shr(word, 32));
    }
    let [x, y, z, w] = words;
    // Range-constrained u32 operands have a product below 2^64.
    let (_, lo) = builder.imul(x, y);
    // The range constraints make these two bit ranges disjoint, so XOR here
    // reconstructs the integer z + 2^32 * w without introducing carry bits.
    let product = builder.bxor(z, builder.shl(w, 32));
    builder.assert_eq("u32_mod32_product", lo, product);
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_core::{constraint_system::ValueVec, word::Word};
    use binius_frontend::Circuit;
    use binius_hash::Blake3HashSuite;
    use binius_prover::{OptimalPackedB128, Prover};
    use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
    use binius_verifier::Verifier;

    fn populate(circuit: &Circuit, wires: &[Wire; 4], row: [u64; 4]) -> (ValueVec, bool) {
        let mut filler = circuit.new_witness_filler();
        for (&wire, value) in wires.iter().zip(row) {
            filler[wire] = Word(value);
        }
        // The evaluator computes all internal values before reporting assertions.
        // Keep that assignment so the compiled constraints can be tested directly.
        let populated = circuit.populate_wire_witness(&mut filler).is_ok();
        (filler.into_value_vec(), populated)
    }

    #[test]
    fn mod32_constraints_accept_boundaries_and_reject_invalid_words() {
        let builder = CircuitBuilder::new();
        let wires = add_u32_mul_mod32(&builder);
        let circuit = builder.build();
        let cs = circuit.constraint_system();
        cs.validate().unwrap();
        let max = u32::MAX as u64;
        let valid = [
            [0, 0, 0, 0],
            [0, max, 0, 0],
            [1, max, max, 0],
            [max, max, 1, max - 1],
            [1 << 16, 1 << 16, 0, 1],
            [1 << 31, 2, 0, 1],
            [max, 2, max - 1, 1],
        ];
        for row in valid {
            let (witness, populated) = populate(&circuit, &wires, row);
            assert!(populated, "valid row {row:?}");
            cs.verify(&witness).unwrap();
        }

        let invalid = [
            [max, max, 0, max - 1],         // Incorrect modular result.
            [max, max, 1, max - 2],         // Incorrect overflow.
            [1 << 32, 0, 0, 0],             // Out-of-range x; product equality still holds.
            [0, 1 << 32, 0, 0],             // Out-of-range y; product equality still holds.
            [1 << 16, 1 << 16, 1 << 32, 0], // Out-of-range z encoding the whole product.
            [0, 0, 0, 1 << 32],             // Out-of-range w lost by the left shift.
        ];
        for row in invalid {
            let (witness, populated) = populate(&circuit, &wires, row);
            assert!(!populated, "invalid row {row:?}");
            assert!(
                cs.verify(&witness).is_err(),
                "compiled constraints accepted invalid row {row:?}"
            );
        }
    }

    #[test]
    fn mod32_binius_proof_roundtrip() {
        let builder = CircuitBuilder::new();
        let wires: Vec<_> = (0..16).map(|_| add_u32_mul_mod32(&builder)).collect();
        let circuit = builder.build();
        let mut filler = circuit.new_witness_filler();
        for (i, wires) in wires.iter().enumerate() {
            let x = u32::MAX as u64 - i as u64;
            let y = u32::MAX as u64;
            let product = x * y;
            for (&wire, value) in wires
                .iter()
                .zip([x, y, product & 0xffff_ffff, product >> 32])
            {
                filler[wire] = Word(value);
            }
        }
        circuit.populate_wire_witness(&mut filler).unwrap();
        let witness = filler.into_value_vec();
        let verifier = Verifier::<Blake3HashSuite>::setup_with_security_bits(
            circuit.constraint_system().clone(),
            1,
            112,
        )
        .unwrap();
        let prover = Prover::<OptimalPackedB128, Blake3HashSuite>::setup(verifier.clone()).unwrap();
        type Challenger = HasherChallenger<blake3::Hasher>;
        let mut transcript = ProverTranscript::new(Challenger::default());
        prover.prove(&witness, &mut transcript).unwrap();
        let mut transcript = VerifierTranscript::new(Challenger::default(), transcript.finalize());
        verifier.verify(witness.inout(), &mut transcript).unwrap();
        transcript.finalize().unwrap();
    }
}
