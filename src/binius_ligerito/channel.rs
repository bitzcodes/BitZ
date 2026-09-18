//! Binius64's IOP channel on the composition's single BLAKE3 transcript.
//!
//! Every oracle the PIOP commits — the packed witness, and the IntMul
//! reduction's logup* pushforward when the circuit multiplies — is committed
//! by [`BinaryPcs`] (rate 1/2 by default, BLAKE3 Merkle tree), its root bound into the
//! transcript and Round 0 taken immediately, before the next challenge. Oracle
//! linear relations are queued, not opened: the adapter discharges each
//! oracle's relations with one Ligerito opening after the PIOP prefix ends,
//! exactly as Binius64's own BaseFold channel defers its openings.
use super::{Oracle, b128_to_f128};
use crate::{
    binary_pcs::{BinaryPcs, Round0Prover, Round0Verifier},
    ligerito_flock::OodRound,
    poly::univariate::binary_gf128::Gf128 as Gf,
    transcript::{Blake3Transcript, traits::Transcript},
};
use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{Field, PackedField};
use binius_iop::channel::{Error as IopError, IOPVerifierChannel, OracleSpec};
use binius_iop_prover::channel::IOPProverChannel;
use binius_ip::channel::{
    Error as IpError, IPVerifierChannel, WordIPVerifierChannel, pack_words_concrete, select_word,
    subset_sum_word,
};
use binius_ip_prover::channel::{IPProverChannel, WordIPProverChannel};
use binius_math::{FieldSlice, FieldVec};
use binius_verifier::config::B128;
use flock_core::{field::Gf128, merkle::Hash, pcs::commit::ProverData};

const ORACLE_DOMAIN: &[u8] = b"bitz/binius64-ligerito/oracle/v1";

/// Bind an oracle's root, in commitment order, before Round 0 draws `ζ`.
fn absorb_root(t: &mut Blake3Transcript, index: usize, root: &Hash) {
    t.absorb_slice(ORACLE_DOMAIN);
    t.absorb_slice(&(index as u64).to_le_bytes());
    t.absorb_slice(root);
}

fn observe(t: &mut Blake3Transcript, v: B128) {
    t.absorb_slice(&u128::from(v).to_le_bytes());
}

fn challenge(t: &mut Blake3Transcript) -> B128 {
    let x: Gf = t.get_field_challenge(&());
    B128::new(u128::from(x.as_words()[0]) | (u128::from(x.as_words()[1]) << 64))
}

fn observe_words(t: &mut Blake3Transcript, words: &[Word]) {
    for word in words {
        t.absorb_slice(&word.0.to_le_bytes());
    }
}

fn sample_bits(t: &mut Blake3Transcript, bits: usize) -> Word {
    assert!(bits <= Word::BITS);
    let value = u128::from(challenge(t)) as u64;
    Word(
        value
            & (u64::MAX
                .checked_shr((Word::BITS - bits) as u32)
                .unwrap_or(0)),
    )
}

/// One committed oracle on the prover side.
pub(super) struct ProverOracle {
    pub packed: Vec<Gf128>,
    pub data: ProverData,
    pub root: Hash,
    pub round0: Round0Prover,
}

/// One queued relation `⟨basis, oracle⟩ = claim`.
pub(super) struct ProverRelation {
    pub oracle: Oracle,
    pub basis: Vec<Gf128>,
    pub claim: B128,
}

pub(super) struct ProverChannel<'a> {
    pub transcript: &'a mut Blake3Transcript,
    pub messages: Vec<u128>,
    /// Oracles still to be committed, in order.
    pub specs: Vec<OracleSpec>,
    /// One solved opener per expected oracle, in commitment order.
    pub pcs: &'a [BinaryPcs],
    pub oracles: Vec<ProverOracle>,
    pub relations: Vec<ProverRelation>,
}

impl IPProverChannel<B128> for ProverChannel<'_> {
    fn send_one(&mut self, elem: B128) {
        self.messages.push(u128::from(elem));
        observe(self.transcript, elem);
    }
    fn observe_one(&mut self, elem: B128) {
        observe(self.transcript, elem);
    }
    fn sample(&mut self) -> B128 {
        challenge(self.transcript)
    }
}

impl WordIPProverChannel<B128> for ProverChannel<'_> {
    type Word = Word;
    fn observe_words(&mut self, words: &[Word]) {
        observe_words(self.transcript, words);
    }
    fn sample_bits(&mut self, bits: usize) -> Word {
        sample_bits(self.transcript, bits)
    }
}

impl<P: PackedField<Scalar = B128>, A: Allocator> IOPProverChannel<P, A> for ProverChannel<'_> {
    type Oracle = Oracle;
    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &self.specs
    }
    fn send_oracle(&mut self, buffer: FieldSlice<'_, P>) -> Oracle {
        let index = self.oracles.len();
        let expected = OracleSpec::new(buffer.log_len());
        assert!(
            self.specs.first() == Some(&expected),
            "oracle {index} has shape {expected:?}, expected {:?}",
            self.specs.first()
        );
        self.specs.remove(0);
        let pcs = &self.pcs[index];
        let (packed, commitment, data) = tracing::info_span!(
            "Commit oracle",
            component = if index == 0 {
                "binius-ligerito.witness-commit"
            } else {
                "binius-ligerito.oracle-commit"
            },
            scope_kind = "procedure",
            oracle_index = index,
            tag_proving = true,
            tag_pcs = true,
            tag_commit = true,
        )
        .in_scope(|| {
            let packed: Vec<Gf128> = buffer.iter_scalars().map(b128_to_f128).collect();
            let (commitment, data) = pcs
                .commit(&packed)
                .expect("the oracle length matches its specification");
            absorb_root(self.transcript, index, &commitment.root);
            (packed, commitment, data)
        });
        let round0 = tracing::info_span!(
            "Round 0",
            component = "binius-ligerito.round0",
            scope_kind = "procedure",
            oracle_index = index,
            tag_proving = true,
            tag_pcs = true,
            tag_opening_proof = true,
        )
        .in_scope(|| pcs.prove_round0(self.transcript, &packed));
        self.oracles.push(ProverOracle {
            packed,
            data,
            root: commitment.root,
            round0,
        });
        index
    }
    fn prove_oracle_relation(&mut self, oracle: Oracle, transparent: FieldVec<P, A>, claim: B128) {
        let packed = &self.oracles[oracle].packed;
        let basis: Vec<Gf128> = transparent.iter_scalars().map(b128_to_f128).collect();
        assert_eq!(
            basis.len(),
            packed.len(),
            "transparent basis covers the oracle"
        );
        self.relations.push(ProverRelation {
            oracle,
            basis,
            claim,
        });
    }
    fn finalize_oracle(&mut self, oracle: Oracle, buffer: FieldVec<P, A>) {
        // The committed words were copied at `send_oracle`; the handed-back
        // buffer must be that oracle.
        assert_eq!(
            1usize << buffer.log_len(),
            self.oracles[oracle].packed.len(),
            "finalized buffer is the committed oracle"
        );
    }
}

/// One received oracle on the verifier side.
pub(super) struct VerifierOracle {
    pub root: Hash,
    pub round0: Round0Verifier,
}

/// One queued relation on the verifier side: `⟨transparent, oracle⟩ = claim`
/// with the transparent basis given as an MLE evaluator.
pub(super) struct VerifierRelation {
    pub oracle: Oracle,
    pub transparent: Box<dyn Fn(&[B128]) -> B128>,
    pub claim: B128,
}

pub(super) struct VerifierChannel<'a> {
    pub transcript: &'a mut Blake3Transcript,
    pub messages: &'a [u128],
    pub specs: Vec<OracleSpec>,
    pub pcs: &'a [BinaryPcs],
    /// The proof's oracle roots and Round-0 messages, in commitment order.
    pub roots: &'a [Hash],
    pub rounds0: &'a [OodRound],
    pub oracles: Vec<VerifierOracle>,
    pub relations: Vec<VerifierRelation>,
}

impl IPVerifierChannel<B128> for VerifierChannel<'_> {
    type Elem = B128;
    fn recv_one(&mut self) -> Result<B128, IpError> {
        let (&head, tail) = self.messages.split_first().ok_or(IpError::ProofEmpty)?;
        self.messages = tail;
        let elem = B128::new(head);
        observe(self.transcript, elem);
        Ok(elem)
    }
    fn sample(&mut self) -> B128 {
        challenge(self.transcript)
    }
    fn observe_one(&mut self, elem: B128) -> B128 {
        observe(self.transcript, elem);
        elem
    }
    fn assert_zero(&mut self, elem: B128) -> Result<(), IpError> {
        if elem == B128::ZERO {
            Ok(())
        } else {
            Err(IpError::InvalidAssert)
        }
    }
}

impl WordIPVerifierChannel<B128> for VerifierChannel<'_> {
    type Word = Word;
    fn observe_words(&mut self, words: &[Word]) -> Vec<Word> {
        observe_words(self.transcript, words);
        words.to_vec()
    }
    fn sample_bits(&mut self, bits: usize) -> Word {
        sample_bits(self.transcript, bits)
    }
    fn subset_sum(&mut self, elems: &[B128], word: &Word) -> B128 {
        subset_sum_word(elems, *word)
    }
    fn select(&mut self, elems: &[B128], word: &Word) -> B128 {
        select_word(elems, *word)
    }
    fn pack_words(&mut self, words: &[Word]) -> Vec<B128> {
        pack_words_concrete::<B128, B128>(words)
    }
}

impl IOPVerifierChannel<B128> for VerifierChannel<'_> {
    type Oracle = Oracle;
    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &self.specs
    }
    fn recv_oracle(&mut self, log_msg_len: usize, dependent: bool) -> Result<Oracle, IopError> {
        let index = self.oracles.len();
        if !dependent
            || self.specs.first() != Some(&OracleSpec::new(log_msg_len))
            || index >= self.pcs.len()
        {
            return Err(IpError::InvalidAssert.into());
        }
        let (Some(root), Some(round)) = (self.roots.get(index), self.rounds0.get(index)) else {
            return Err(IopError::ProofEmpty);
        };
        self.specs.remove(0);
        absorb_root(self.transcript, index, root);
        let round0 = self.pcs[index]
            .verify_round0(self.transcript, round)
            .map_err(|_| IopError::from(IpError::InvalidAssert))?;
        self.oracles.push(VerifierOracle {
            root: *root,
            round0,
        });
        Ok(index)
    }
    fn verify_oracle_relation(
        &mut self,
        oracle: Oracle,
        transparent: Box<dyn Fn(&[B128]) -> B128>,
        claim: B128,
    ) -> Result<(), IopError> {
        if oracle >= self.oracles.len() {
            return Err(IpError::InvalidAssert.into());
        }
        self.relations.push(VerifierRelation {
            oracle,
            transparent,
            claim,
        });
        Ok(())
    }
}
