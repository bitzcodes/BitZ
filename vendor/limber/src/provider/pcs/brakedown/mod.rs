//! Brakedown polynomial commitment scheme — group-free, based on a linear-time
//! expander code + Merkle column commitments (no curve / MSM).
//!
//! Construction, parameters, and the commit/open/verify (tensor IOPP) shape
//! follow *Brakedown: Linear-time and field-agnostic SNARKs for R1CS*
//! (Golovnev, Lee, Setty, Thaler, Wahby — eprint 2021/1043). Built
//! standalone; the Mod-PCS integration is deferred (it needs commitment
//! homomorphism a Merkle commitment lacks).

pub mod code;
pub mod commit;
pub mod eval;
pub mod merkle;

pub use code::{DEFAULT_SPEC, SPECS};
pub use commit::{
  BrakedownCommitData, BrakedownParams, commit as brakedown_commit,
  commit_plain as brakedown_commit_plain,
};
pub use eval::{
  BrakedownDirectOpen, BrakedownEvalArg, BrakedownGroupArg, open_direct as brakedown_open_direct,
  open_group as brakedown_open_group, open_with_data as brakedown_open_with_data,
  verify_direct as brakedown_verify_direct, verify_group as brakedown_verify_group,
  verify_open as brakedown_verify_open,
};

#[cfg(test)]
mod field_ab_tests {
  use super::*;
  use crate::provider::pt256::t256;
  use crate::traits::PrimeFieldExt;
  use ff::{Field, PrimeField};
  use std::time::Instant;

  /// M127 = 2^127 − 1 via ff_derive, promoted to `PrimeFieldExt` for the
  /// Brakedown field A/B (see `field_128_candidates_microbench`).
  #[derive(ff::PrimeField)]
  #[PrimeFieldModulus = "170141183460469231731687303715884105727"]
  #[PrimeFieldGenerator = "3"]
  #[PrimeFieldReprEndianness = "little"]
  struct F127([u64; 2]);

  /// Eager accumulator so the test field satisfies the
  /// `DelayedReduction` supertrait of `PrimeFieldExt`.
  #[derive(Clone, Copy, Default)]
  struct F127Acc(F127);
  impl core::ops::AddAssign for F127Acc {
    fn add_assign(&mut self, rhs: Self) {
      self.0 += rhs.0;
    }
  }
  impl crate::traits::DelayedReduction<F127> for F127 {
    type Accumulator = F127Acc;
    fn unreduced_multiply_accumulate(acc: &mut F127Acc, field: &Self, value: &F127) {
      acc.0 += *field * *value;
    }
    fn reduce(acc: &F127Acc) -> Self {
      acc.0
    }
  }

  impl PrimeFieldExt for F127 {
    fn from_uniform(bytes: &[u8]) -> Self {
      // Horner over 128-bit chunks; 2^128 ≡ 2 (mod 2^127 − 1), so each
      // step is acc·2 + chunk.
      let mut acc = F127::ZERO;
      for chunk in bytes.chunks(16).rev() {
        let mut le = [0u8; 16];
        le[..chunk.len()].copy_from_slice(chunk);
        acc = acc.double() + F127::from_u128(u128::from_le_bytes(le));
      }
      acc
    }
  }

  /// Commit-path A/B at equal element count (the real workload's chunk
  /// count is field-independent): t256 vs F127 on 16-bit chunk data.
  /// Run: RAYON_NUM_THREADS=1 cargo test --release brakedown_field_ab -- --ignored --nocapture
  #[test]
  #[ignore]
  fn brakedown_field_ab() {
    fn run<F: PrimeFieldExt>(tag: &str, log_n: usize) -> f64 {
      let n = 1usize << log_n;
      let poly: Vec<F> = (0..n)
        .map(|i| F::from((i as u64).wrapping_mul(0x9e37_79b9) & 0xffff))
        .collect();
      let params = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"ab-seed");
      // Warm once, then time.
      let _ = brakedown_commit(&params, &poly);
      let t = Instant::now();
      let (_root, _data) = brakedown_commit(&params, &poly);
      let ms = t.elapsed().as_secs_f64() * 1e3;
      println!(
        "  {tag:6} 2^{log_n}: commit {ms:8.1} ms  ({:.1} ns/elem)",
        ms * 1e6 / n as f64
      );
      ms
    }
    for log_n in [18usize, 20, 21] {
      println!("n = 2^{log_n}:");
      let a = run::<t256::Scalar>("t256", log_n);
      let b = run::<F127>("F127", log_n);
      println!("  ratio: {:.2}x", a / b);
    }
  }
}
