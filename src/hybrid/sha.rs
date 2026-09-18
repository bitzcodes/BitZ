//! Sequential SHA-256 compression, with a fixed IV and one public final state.
use super::{
    BinaryClaim, Error,
    channel::{ProverChannel, VerifierChannel},
};
use crate::transcript::Blake3Transcript;
use binius_circuits::sha256::compress::{State, ref_compress, sha256_compress_2x_seq};
use binius_core::{
    constraint_system::{InoutSegment, ValueVec},
    word::Word,
};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_iop::channel::OracleSpec;
use binius_prover::{IOPProver, protocols::shift::KeyCollection};
use binius_verifier::{IOPVerifier, config::B128};
use flock_core::field::Gf128;

pub const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Each block contains sixteen SHA words, in the standard big-endian word order.
/// This is a compression chain; callers supply any desired SHA padding themselves.
pub fn chaining_value(blocks: &[[u32; 16]]) -> [u32; 8] {
    blocks
        .iter()
        .fold(SHA256_IV, |state, block| ref_compress(state, *block))
}

pub(super) struct ShaRelation {
    pub circuit: Circuit,
    pub blocks: Vec<[Wire; 16]>,
    output: [Wire; 8],
    pub verifier: IOPVerifier,
    prover: IOPProver,
    /// Recycles the IOP prover's large, short-lived working buffers across
    /// proofs, as Binius64's own `Prover` does for its lifetime. With the
    /// global allocator every proof page-faults them afresh: the SHA
    /// reductions took 264 ms against 232 ms in all-Binius at 2^14
    /// compressions, 231 ms with the pool. (A per-proof pool recovers
    /// nothing: the gain is the cross-proof reuse.) The pool keeps its
    /// blocks, so a verifier running next in the same process allocates
    /// cold pages instead of reusing the prover's — the same situation as
    /// Binius64's verifier after its pooled prover.
    pool: binius_compute::BufferPool,
}

impl ShaRelation {
    pub fn new(count: usize) -> Result<Self, Error> {
        tracing::info!(compressions = count, "building chained SHA circuit");
        let builder = CircuitBuilder::new();
        let blocks: Vec<[Wire; 16]> = (0..count)
            .map(|_| std::array::from_fn(|_| builder.add_witness()))
            .collect();
        let output = std::array::from_fn(|_| builder.add_inout());
        let mut state = State::iv(&builder);
        for pair in blocks.chunks_exact(2) {
            state = sha256_compress_2x_seq(&builder, state, [pair[0], pair[1]]);
        }
        let mask = builder.add_constant(Word(u32::MAX as u64));
        for (actual, expected) in state.0.into_iter().zip(output) {
            builder.assert_eq(
                "final_sha_chaining_value",
                builder.band(actual, mask),
                expected,
            );
        }
        tracing::info!("compiling chained SHA circuit");
        let circuit = builder.build();
        let cs = circuit.constraint_system();
        cs.validate().map_err(|e| Error::Binius(e.to_string()))?;
        if !cs.imul_constraints.is_empty() || !cs.bmul_constraints.is_empty() {
            return Err(Error::Invalid(
                "SHA circuit unexpectedly requires auxiliary oracles",
            ));
        }
        let verifier = IOPVerifier::new(cs.clone(), cs.log_public_words(InoutSegment::Public));
        tracing::info!("preparing SHA shift keys");
        let prover = IOPProver::new(
            verifier.clone(),
            KeyCollection::build(cs, InoutSegment::Public),
        );
        Ok(Self {
            circuit,
            blocks,
            output,
            verifier,
            prover,
            pool: binius_compute::BufferPool::new(),
        })
    }

    pub fn public(&self, final_state: [u32; 8]) -> Vec<Word> {
        final_state
            .into_iter()
            .map(|word| Word(u64::from(word)))
            .collect()
    }

    pub fn populate(&self, blocks: &[[u32; 16]], final_state: [u32; 8]) -> Result<ValueVec, Error> {
        if blocks.len() != self.blocks.len() {
            return Err(Error::Invalid("SHA block count"));
        }
        let mut filler = self.circuit.new_witness_filler();
        for (wires, block) in self.blocks.iter().zip(blocks) {
            for (&wire, &word) in wires.iter().zip(block) {
                filler[wire] = Word(word as u64);
            }
        }
        for (&wire, word) in self.output.iter().zip(final_state) {
            filler[wire] = Word(word as u64);
        }
        self.circuit
            .populate_wire_witness(&mut filler)
            .map_err(|e| Error::Binius(e.to_string()))?;
        Ok(filler.into_value_vec())
    }

    pub fn pack(&self, witness: &ValueVec) -> Vec<Gf128> {
        let mut packed = vec![Gf128::ZERO; 1 << self.verifier.log_witness_elems()];
        for (dst, words) in packed.iter_mut().zip(witness.non_public().chunks(2)) {
            *dst = Gf128 {
                lo: words[0].0,
                hi: words.get(1).map_or(0, |w| w.0),
            };
        }
        packed
    }

    pub fn prove(
        &self,
        t: &mut Blake3Transcript,
        witness: &ValueVec,
    ) -> Result<(Vec<u128>, BinaryClaim), Error> {
        let mut channel = ProverChannel {
            transcript: t,
            messages: Vec::new(),
            spec: vec![OracleSpec::new(self.verifier.log_witness_elems())],
        };
        let alloc = &self.pool;
        let (_, _, point, value) = self
            .prover
            .prove_to_evaluation::<_, binius_prover::OptimalPackedB128, _>(
                witness,
                &mut channel,
                &alloc,
            )
            .map_err(|e| Error::Binius(e.to_string()))?;
        if !channel.spec.is_empty() {
            return Err(Error::Invalid("unconsumed SHA oracle"));
        }
        Ok((channel.messages, evaluation_claim(&point, value)))
    }

    pub fn verify(
        &self,
        t: &mut Blake3Transcript,
        public: &[Word],
        messages: &[u128],
    ) -> Result<BinaryClaim, Error> {
        let mut channel = VerifierChannel {
            transcript: t,
            messages,
            spec: vec![OracleSpec::new(self.verifier.log_witness_elems())],
        };
        let (_, point, value) = self
            .verifier
            .verify_to_evaluation(public, &mut channel)
            .map_err(|e| Error::Binius(e.to_string()))?;
        if !channel.messages.is_empty() || !channel.spec.is_empty() {
            return Err(Error::Invalid("trailing SHA prefix data"));
        }
        Ok(evaluation_claim(&point, value))
    }
}

fn evaluation_claim(point: &[B128], value: B128) -> BinaryClaim {
    let convert = |x: B128| Gf128 {
        lo: u128::from(x) as u64,
        hi: (u128::from(x) >> 64) as u64,
    };
    let r: Vec<_> = point.iter().copied().map(convert).collect();
    BinaryClaim {
        low: super::sumcheck::eq_table(&r[..7]),
        high_point: r[7..].to_vec(),
        value: convert(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_claim_matches_private_bits_and_checks_wiring() {
        let relation = ShaRelation::new(2).unwrap();
        let blocks = [[0x01234567; 16], [0x89abcdef; 16]];
        let final_state = chaining_value(&blocks);
        let public = relation.public(final_state);
        let witness = relation.populate(&blocks, final_state).unwrap();
        relation
            .circuit
            .constraint_system()
            .verify(&witness)
            .unwrap();
        assert_eq!(public, witness.inout());
        let transcript = Blake3Transcript::new;
        let (messages, claim) = relation.prove(&mut transcript(), &witness).unwrap();
        let verified = relation
            .verify(&mut transcript(), &public, &messages)
            .unwrap();
        assert_eq!(claim.low, verified.low);
        assert_eq!(claim.high_point, verified.high_point);
        assert_eq!(claim.value, verified.value);

        // Evaluate the committed private bit-MLE independently of Binius's reduction.
        let high = super::super::sumcheck::eq_table(&claim.high_point);
        let value =
            relation
                .pack(&witness)
                .iter()
                .zip(high)
                .fold(Gf128::ZERO, |sum, (word, weight)| {
                    let mut bits = u128::from(word.lo) | (u128::from(word.hi) << 64);
                    let mut low = Gf128::ZERO;
                    while bits != 0 {
                        low += claim.low[bits.trailing_zeros() as usize];
                        bits &= bits - 1;
                    }
                    sum + low * weight
                });
        assert_eq!(value, claim.value);

        // The last prefix message is the claimed wiring evaluation. There are no later
        // Fiat-Shamir challenges to make this fail accidentally: check_native must reject it.
        let mut forged = messages.clone();
        *forged.last_mut().unwrap() ^= 1;
        assert!(
            relation
                .verify(&mut transcript(), &public, &forged)
                .is_err()
        );
        let mut changed_public = public.clone();
        changed_public[0].0 ^= 1;
        assert!(
            relation
                .verify(&mut transcript(), &changed_public, &messages)
                .is_err()
        );
        assert!(
            relation
                .verify(&mut transcript(), &public[..7], &messages)
                .is_err()
        );
    }
}
