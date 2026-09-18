//! `F127`: the prime field mod M127 = 2^127 − 1, the q-side field of
//! the small-field / hash-commitment instantiation.
//!
//! Chosen from the four-way microbench (`field_128_candidates_microbench`
//! in `dyn_prime.rs`): Mersenne reduction gave 3.8–4.9× over t256 per
//! multiplication. The arithmetic itself is `ff_derive`'s fixed-modulus
//! 2-limb Montgomery code (audited derive, 3–4× measured); a
//! Mersenne-fold fast path can replace hot loops later if profiles ask.
//!
//! M127 has two-adicity 1 — no FFT structure — which is fine here: this
//! stack is sumcheck- and expander-code-based and never runs an FFT.
//! Curve-based commitments cannot use this field (a ~2^127 group has
//! only ~63-bit discrete-log security); it is only sound with the
//! hash-based (Brakedown) backend.
//! Challenges are drawn from the base field under the accepted
//! `LAMBDA_BOUND2 = 117` target.

// ff_derive emits an undocumented public F127Repr companion struct;
// everything hand-written here is documented.
#![allow(missing_docs)]

use crate::{
  big_num::DelayedReduction,
  traits::{PrimeFieldExt, transcript::TranscriptReprTrait},
};
use ff::{Field, PrimeField};

/// Generator 3 is a quadratic nonresidue mod M127 (p ≡ 7 mod 12).
#[derive(PrimeField)]
#[PrimeFieldModulus = "170141183460469231731687303715884105727"]
#[PrimeFieldGenerator = "3"]
#[PrimeFieldReprEndianness = "little"]
pub struct F127([u64; 2]);

impl PrimeFieldExt for F127 {
  fn from_uniform(bytes: &[u8]) -> Self {
    // Interpret `bytes` as a little-endian integer and reduce: Horner
    // over 128-bit chunks from the MOST significant down; each step is
    // acc·2^128 + chunk, and 2^128 ≡ 2 (mod 2^127 − 1). (Walking the
    // chunks low-first once scaled every wide value by 8 — caught by
    // the M127 MultiSwap run's reconstruction sumcheck.)
    let mut acc = F127::ZERO;
    for chunk in bytes.chunks(16).rev() {
      let mut le = [0u8; 16];
      le[..chunk.len()].copy_from_slice(chunk);
      acc = acc.double() + F127::from_u128(u128::from_le_bytes(le));
    }
    acc
  }
}

impl TranscriptReprTrait for F127 {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().as_ref().to_vec()
  }
}

// ---- serde: canonical little-endian repr bytes --------------------------

impl serde::Serialize for F127 {
  fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(self.to_repr().as_ref(), s)
  }
}

impl<'de> serde::Deserialize<'de> for F127 {
  fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
    let bytes: Vec<u8> = serde::Deserialize::deserialize(d)?;
    let arr: [u8; 16] = bytes
      .as_slice()
      .try_into()
      .map_err(|_| serde::de::Error::custom("F127: expected 16 bytes"))?;
    Option::<F127>::from(F127::from_repr(F127Repr(arr)))
      .ok_or_else(|| serde::de::Error::custom("F127: non-canonical repr"))
  }
}

// ---- delayed reduction: eager for now -----------------------------------

/// Eagerly-reduced accumulator: each product is reduced as it lands.
/// Correct and simple; a `WideLimbs<5>` unreduced 2-limb accumulator
/// (mirroring the 4-limb `montgomery_reduce_9` pattern) is the fast
/// path to add when the smoke test graduates to benchmarking.
#[derive(Clone, Copy)]
pub struct F127EagerAcc(F127);

impl Default for F127EagerAcc {
  fn default() -> Self {
    Self(F127::ZERO)
  }
}

impl core::ops::AddAssign for F127EagerAcc {
  fn add_assign(&mut self, rhs: Self) {
    self.0 += rhs.0;
  }
}

impl DelayedReduction<F127> for F127 {
  type Accumulator = F127EagerAcc;

  #[inline(always)]
  fn unreduced_multiply_accumulate(acc: &mut Self::Accumulator, field: &Self, value: &F127) {
    acc.0 += *field * *value;
  }

  #[inline(always)]
  fn reduce(acc: &Self::Accumulator) -> Self {
    acc.0
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;

  #[test]
  fn from_uniform_matches_horner() {
    // 64 uniform bytes reduce consistently with direct arithmetic.
    let bytes = [0xabu8; 64];
    let v = F127::from_uniform(&bytes);
    let mut expect = F127::ZERO;
    for chunk in bytes.chunks(16) {
      expect = expect.double() + F127::from_u128(u128::from_le_bytes(chunk.try_into().unwrap()));
    }
    assert_eq!(v, expect);
  }

  #[test]
  fn serde_roundtrips() {
    let x = F127::from_u128(0x1234_5678_9abc_def0_1122_3344_5566_7788u128);
    let bytes = bincode::serialize(&x).unwrap();
    let y: F127 = bincode::deserialize(&bytes).unwrap();
    assert_eq!(x, y);
  }

  crate::test_delayed_reduction!(f127_delayed_reduction, super::F127);
}
