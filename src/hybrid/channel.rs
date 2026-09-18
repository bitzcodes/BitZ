//! A Binius prefix channel on the composition's single Fiat–Shamir transcript.
use super::Gf;
use crate::transcript::{Blake3Transcript, traits::Transcript};
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

pub(super) struct ProverChannel<'a> {
    pub transcript: &'a mut Blake3Transcript,
    pub messages: Vec<u128>,
    pub spec: Vec<OracleSpec>,
}

fn observe(t: &mut Blake3Transcript, v: B128) {
    t.absorb_slice(&u128::from(v).to_le_bytes());
}

fn challenge(t: &mut Blake3Transcript) -> B128 {
    let x: Gf = t.get_field_challenge(&());
    B128::new(u128::from(x.as_words()[0]) | (u128::from(x.as_words()[1]) << 64))
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

impl<P: PackedField<Scalar = B128>, A: Allocator> IOPProverChannel<P, A> for ProverChannel<'_> {
    type Oracle = ();
    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &self.spec
    }
    fn send_oracle(&mut self, buffer: FieldSlice<P>) {
        assert_eq!(self.spec, [OracleSpec::new(buffer.log_len())]);
        self.spec.clear();
        // The composition already committed this witness and bound its root.
        self.transcript
            .absorb_slice(b"hybrid/sha-precommitted-witness/v2");
    }
    fn prove_oracle_relation(&mut self, _: (), _: FieldVec<P, A>, _: B128) {
        panic!("the SHA prefix must stop before ring switching");
    }
    fn finalize_oracle(&mut self, _: (), _: FieldVec<P, A>) {
        panic!("the composition owns the precommitted SHA witness");
    }
}

pub(super) struct VerifierChannel<'a> {
    pub transcript: &'a mut Blake3Transcript,
    pub messages: &'a [u128],
    pub spec: Vec<OracleSpec>,
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

impl IOPVerifierChannel<B128> for VerifierChannel<'_> {
    type Oracle = ();
    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &self.spec
    }
    fn recv_oracle(&mut self, log_msg_len: usize, dependent: bool) -> Result<(), IopError> {
        if !dependent || self.spec != [OracleSpec::new(log_msg_len)] {
            return Err(IpError::InvalidAssert.into());
        }
        self.spec.clear();
        self.transcript
            .absorb_slice(b"hybrid/sha-precommitted-witness/v2");
        Ok(())
    }
    fn verify_oracle_relation(
        &mut self,
        _: (),
        _: Box<dyn Fn(&[B128]) -> B128>,
        _: B128,
    ) -> Result<(), IopError> {
        Err(IpError::InvalidAssert.into())
    }
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

impl WordIPProverChannel<B128> for ProverChannel<'_> {
    type Word = Word;
    fn observe_words(&mut self, words: &[Word]) {
        observe_words(self.transcript, words);
    }
    fn sample_bits(&mut self, bits: usize) -> Word {
        sample_bits(self.transcript, bits)
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
