//! A [`CommitBackend`] over the BaseFold multilinear PCS: hash-based,
//! transparent (no trusted setup, no curve), `O(log²n)` proofs. Targets are
//! opened one BaseFold proof each (multi-target batching into a single shared
//! fold is a later optimization). Non-hiding, like the Brakedown backend.

use super::code::FoldableCode;
use super::open::{EvalProof, prove_eval, verify_eval};
use super::query::{CommittedLayer, commit_layer};
use crate::errors::SpartanError;
use crate::provider::pcs::commit_backend::{CommitBackend, OpenTarget};
use crate::traits::mod_engine::SumcheckEngine;
use crate::traits::transcript::ByteTranscript;
use core::marker::PhantomData;

/// Inverse rate of the foldable code (codeword length = `8·2^log_k`).
const BF_INV_RATE: usize = 8;
/// Column queries per opening (proximity soundness; conservative POC value).
const BF_N_QUERIES: usize = 100;
/// Public seed for the deterministic foldable-code diagonals.
const BF_SEED: &[u8] = b"limber-basefold-v1";

/// Deterministic foldable code for `2^log_k` messages, built once per
/// `(field, log_k)` and cached (the build inverts `O(2^log_k)` diagonals).
fn code_for<F: crate::traits::PrimeFieldExt>(log_k: usize) -> &'static FoldableCode<F> {
  use std::any::{Any, TypeId};
  use std::collections::HashMap;
  use std::sync::{Mutex, OnceLock};
  static CACHE: OnceLock<Mutex<HashMap<(TypeId, usize), &'static (dyn Any + Send + Sync)>>> =
    OnceLock::new();
  let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
  let mut guard = cache.lock().expect("basefold code cache poisoned");
  let key = (TypeId::of::<F>(), log_k);
  if let Some(p) = guard.get(&key) {
    return p.downcast_ref::<FoldableCode<F>>().expect("cache type");
  }
  let code: &'static FoldableCode<F> =
    Box::leak(Box::new(FoldableCode::new(log_k, BF_INV_RATE, BF_SEED)));
  guard.insert(key, code as &'static (dyn Any + Send + Sync));
  code
}

fn verr(msg: &str) -> SpartanError {
  SpartanError::InternalError {
    reason: format!("basefold: {msg}"),
  }
}

/// BaseFold commitment backend, parameterized by the sumcheck engine `SE`
/// (which fixes the commitment field `SE::Scalar`).
#[derive(Clone, Debug)]
pub struct BfBackend<SE>(PhantomData<SE>);

impl<SE> CommitBackend for BfBackend<SE>
where
  SE: SumcheckEngine + Clone + core::fmt::Debug + 'static,
  SE::Scalar: crate::traits::PrimeFieldExt
    + crate::traits::transcript::TranscriptReprTrait
    + serde::Serialize
    + serde::de::DeserializeOwned,
{
  type Scalar = SE::Scalar;
  type SE = SE;
  type Ck = ();
  type Vk = ();
  type Comm = [u8; 32];
  type Blind = ();
  /// Prover-retained: the committed codeword (`c0`) tree + entries.
  type Data = CommittedLayer<SE::Scalar>;
  type BatchOpenArg = Vec<EvalProof<SE::Scalar>>;

  fn blind(_ck: &Self::Ck, _n: usize) -> Self::Blind {}

  fn comm_transcript_bytes(comm: &Self::Comm) -> Vec<u8> {
    comm.to_vec()
  }

  fn commit(
    _ck: &Self::Ck,
    poly: &[Self::Scalar],
    _blind: &Self::Blind,
    _small: bool,
  ) -> Result<(Self::Comm, Self::Data), SpartanError> {
    let n = poly.len();
    if !n.is_power_of_two() {
      return Err(verr("polynomial length must be a power of two"));
    }
    let log_k = n.ilog2() as usize;
    let code = code_for::<Self::Scalar>(log_k);
    let c0 = commit_layer(code.encode(poly));
    Ok((c0.root, c0))
  }

  fn recommit_data(
    ck: &Self::Ck,
    comm: &Self::Comm,
    poly: &[Self::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<Self::Data, SpartanError> {
    let (root, data) = Self::commit(ck, poly, blind, small)?;
    if &root != comm {
      return Err(verr("recommit root mismatch"));
    }
    Ok(data)
  }

  fn open_targets(
    _ck: &Self::Ck,
    targets: &[OpenTarget<'_, Self>],
    sub: &mut impl ByteTranscript,
  ) -> Result<Self::BatchOpenArg, SpartanError> {
    let mut proofs = Vec::with_capacity(targets.len());
    for t in targets {
      let n = t.poly.len();
      if !n.is_power_of_two() {
        return Err(verr("polynomial length must be a power of two"));
      }
      let log_k = n.ilog2() as usize;
      if t.point.len() != log_k {
        return Err(verr("point dimension mismatch"));
      }
      let code = code_for::<Self::Scalar>(log_k);
      sub.absorb_bytes(b"bf-comm", t.comm);
      let proof = prove_eval(code, t.data, t.poly, &t.point, t.eval, BF_N_QUERIES, sub)?;
      proofs.push(proof);
    }
    Ok(proofs)
  }

  fn verify_targets(
    _vk: &Self::Vk,
    targets: &[(&Self::Comm, Vec<Self::Scalar>, Self::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError> {
    if arg.len() != targets.len() {
      return Err(verr("proof count mismatch"));
    }
    for ((comm, point, eval), proof) in targets.iter().zip(arg) {
      let log_k = point.len();
      let code = code_for::<Self::Scalar>(log_k);
      sub.absorb_bytes(b"bf-comm", *comm);
      if !verify_eval(code, comm, point, *eval, proof, BF_N_QUERIES, sub)? {
        return Err(verr("evaluation proof rejected"));
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256HyraxEngine;
  use crate::provider::keccak::Keccak256Transcript;
  use crate::provider::pcs::basefold::open::EvalProof;
  use crate::provider::pt256::t256;
  use crate::traits::transcript::TranscriptEngineTrait;
  use ff::Field;

  type B = BfBackend<T256HyraxEngine>;
  type F = t256::Scalar;
  type Tr = Keccak256Transcript<T256HyraxEngine>;

  fn build_eq_eval(m: &[F], z: &[F]) -> F {
    // MLE of m at z.
    let d = z.len();
    let mut acc = F::ZERO;
    for (idx, mi) in m.iter().enumerate() {
      let mut w = F::ONE;
      for (j, &zj) in z.iter().enumerate() {
        let bit = (idx >> j) & 1;
        w *= if bit == 1 { zj } else { F::ONE - zj };
      }
      let _ = d;
      acc += w * *mi;
    }
    acc
  }

  fn rand(n: usize, seed: u64) -> Vec<F> {
    super::super::code::rand_vec(n, seed)
  }

  #[test]
  #[ignore = "measurement; run with --ignored --nocapture"]
  fn proof_size() {
    for d in [12usize, 16, 18] {
      let poly = rand(1 << d, 1);
      let z = rand(d, 2);
      let y = build_eq_eval(&poly, &z);
      let tc = std::time::Instant::now();
      let (comm, data) = B::commit(&(), &poly, &(), false).unwrap();
      let commit_ms = tc.elapsed().as_secs_f64() * 1e3;
      let mut tp: Tr = Keccak256Transcript::new_with_params(b"bf", ());
      let targets = [OpenTarget {
        comm: &comm,
        poly: &poly,
        blind: &(),
        data: &data,
        point: z.clone(),
        eval: y,
      }];
      let to = std::time::Instant::now();
      let arg = B::open_targets(&(), &targets, &mut tp).unwrap();
      let open_ms = to.elapsed().as_secs_f64() * 1e3;
      let bytes = bincode::serialize(&arg).unwrap();
      println!(
        "BaseFold n=2^{d} rate=1/{BF_INV_RATE} {BF_N_QUERIES}q: commit {commit_ms:.1} ms, \
         open {open_ms:.1} ms, proof {} bytes ({:.1} KB)",
        bytes.len(),
        bytes.len() as f64 / 1024.0
      );
    }
  }

  #[test]
  fn commit_backend_round_trip() {
    let d = 6;
    let poly = rand(1 << d, 1);
    let z = rand(d, 2);
    let y = build_eq_eval(&poly, &z);

    let (comm, data) = B::commit(&(), &poly, &(), false).unwrap();

    // Open.
    let mut tp: Tr = Keccak256Transcript::new_with_params(b"bf-be", ());
    let targets = [OpenTarget {
      comm: &comm,
      poly: &poly,
      blind: &(),
      data: &data,
      point: z.clone(),
      eval: y,
    }];
    let arg = B::open_targets(&(), &targets, &mut tp).unwrap();

    // Verify (accept).
    let mut tv: Tr = Keccak256Transcript::new_with_params(b"bf-be", ());
    let vt = [(&comm, z.clone(), y)];
    assert!(B::verify_targets(&(), &vt, &arg, &mut tv).is_ok());

    // Wrong eval -> reject.
    let mut tv2: Tr = Keccak256Transcript::new_with_params(b"bf-be", ());
    let vt2 = [(&comm, z.clone(), y + F::ONE)];
    assert!(B::verify_targets(&(), &vt2, &arg, &mut tv2).is_err());

    // Tampered proof -> reject.
    let mut bad: Vec<EvalProof<F>> = arg.clone();
    bad[0].sumcheck[0][0] += F::ONE;
    let mut tv3: Tr = Keccak256Transcript::new_with_params(b"bf-be", ());
    let vt3 = [(&comm, z.clone(), y)];
    assert!(B::verify_targets(&(), &vt3, &bad, &mut tv3).is_err());
  }
}
