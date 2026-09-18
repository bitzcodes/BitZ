// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module provides the trait definitions for transcript functionality in the Spartan2 library.
//! Transcripts are used for Fiat-Shamir transformations to make interactive proof systems non-interactive.
use crate::{
  errors::SpartanError,
  traits::mod_engine::{SumcheckEngine, SumcheckField},
};

/// This trait allows types to implement how they want to be added to `TranscriptEngine`.
///
/// (The previous `<G: Group>` marker was dropped — the parameter
/// was only used as a phantom for type-level domain separation between
/// protocols, which the `dom_sep` label already handles. Removing it lets
/// non-group-based PCSes implement the trait without inventing a placeholder
/// curve group.)
pub trait TranscriptReprTrait: Send + Sync {
  /// returns a byte representation of self to be added to the transcript
  fn to_transcript_bytes(&self) -> Vec<u8>;
}

/// Field-agnostic transcript operations: absorb raw bytes, squeeze raw
/// bytes, and domain separation.
///
/// This is the byte-level Fiat-Shamir chain. A single byte-chain can feed
/// multiple field reductions — e.g. the dual-field SNARK's `Z_p` sumcheck and
/// the underlying PCS's `F_q` opening both read challenges from the *same*
/// `ByteTranscript`, each reducing the raw bytes into its own field. This
/// decoupling is what lets `sumcheck_modp` (arithmetic over a dynamic
/// prime) share one transcript with a curve-based PCS opening.
pub trait ByteTranscript: Send + Sync {
  /// Squeeze 64 raw bytes of challenge material under a label.
  fn squeeze_bytes(&mut self, label: &'static [u8]) -> Result<[u8; 64], SpartanError>;

  /// Absorb raw bytes under a label.
  fn absorb_bytes(&mut self, label: &'static [u8], bytes: &[u8]);

  /// Add a domain separator.
  fn dom_sep(&mut self, bytes: &'static [u8]);

  /// Absorb anything implementing `TranscriptReprTrait`, via its bytes.
  /// Field-agnostic: absorbing only ever moves bytes into the chain.
  fn absorb<T: TranscriptReprTrait>(&mut self, label: &'static [u8], o: &T) {
    self.absorb_bytes(label, &o.to_transcript_bytes());
  }
}

/// This trait defines the behavior of a transcript engine compatible with
/// Spartan. It layers field-typed conveniences over [`ByteTranscript`].
pub trait TranscriptEngineTrait<E: SumcheckEngine>: ByteTranscript {
  /// Initialize the transcript with the field's modulus context.
  ///
  /// The `params` are needed so that `squeeze` can construct challenges in
  /// dynamic-modulus fields (where the modulus is a runtime value carried
  /// in `params`). For static-modulus fields `params` is `()`.
  fn new_with_params(label: &'static [u8], params: <E::Scalar as SumcheckField>::Params) -> Self
  where
    Self: Sized;

  /// Initialize the transcript. Convenience for fields whose `Params` is
  /// `Default` (i.e. all static-modulus fields, where `Params = ()`).
  fn new(label: &'static [u8]) -> Self
  where
    Self: Sized,
    <E::Scalar as SumcheckField>::Params: Default,
  {
    Self::new_with_params(label, Default::default())
  }

  /// returns a scalar element of the field as a challenge
  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, SpartanError>;
}

impl<T: TranscriptReprTrait> TranscriptReprTrait for &[T] {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self
      .iter()
      .flat_map(|t| t.to_transcript_bytes())
      .collect::<Vec<u8>>()
  }
}
