//! Full BaseFold multilinear opening: an eq-weighted sumcheck for
//! `ẑ(z) = Σ_b eq(z,b)·m[b]` **interleaved** with the codeword fold. Each
//! round's (random) sumcheck challenge is also the fold challenge, so the
//! proximity test folds with random challenges while the `eq(z,·)` weighting
//! pins the evaluation to the specific point `z`. After `d` rounds the folded
//! codeword is `base(ẑ(α))` and the sumcheck's final claim must equal
//! `eq(z,α)·ẑ(α)`, tying the recovered value back to the claimed `y`.
//!
//! Fiat–Shamir runs over the crate `ByteTranscript`, so this slots directly
//! behind `CommitBackend`. The caller absorbs the commitment before opening;
//! this routine absorbs the opening's own `(z, y)` and every fold root.

use super::super::brakedown::merkle::{Hash, MerkleTree, hash_leaf, verify_path};
use super::code::FoldableCode;
use super::query::{CommittedLayer, QueryOpening, commit_layer, leaf_bytes};
use crate::errors::SpartanError;
use crate::traits::PrimeFieldExt;
use crate::traits::transcript::ByteTranscript;
use ff::{Field, PrimeField};

/// A full multilinear-evaluation proof.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvalProof<F> {
  /// Per round: the quadratic sumcheck message as `(h(0), h(1), h(2))`.
  pub sumcheck: Vec<[F; 3]>,
  /// Merkle roots of the folded codewords `c¹, …, cᵈ`.
  pub layer_roots: Vec<Hash>,
  /// The base codeword `cᵈ`.
  pub base: Vec<F>,
  /// Fold-consistency query openings.
  pub queries: Vec<QueryOpening<F>>,
}

#[inline]
fn eqf<F: Field>(a: F, b: F) -> F {
  a * b + (F::ONE - a) * (F::ONE - b)
}

/// `eq(z, ·)` evaluations over the hypercube, `eq_table[idx] = Π_j eqf(z[j],
/// bit_j(idx))` — same index convention as the committed evaluation vector.
fn build_eq<F: Field>(z: &[F]) -> Vec<F> {
  let mut t = vec![F::ONE; 1usize << z.len()];
  for (j, &zj) in z.iter().enumerate() {
    let bit = 1usize << j;
    for (idx, ti) in t.iter_mut().enumerate() {
      let bj = if idx & bit != 0 { zj } else { F::ONE - zj };
      *ti *= bj;
    }
  }
  t
}

/// Evaluation-basis fold `(1−α)·lo + α·hi` (lo = first half = MSB 0).
fn fold_eval<F: Field>(t: &[F], alpha: F) -> Vec<F> {
  let half = t.len() / 2;
  (0..half)
    .map(|j| (F::ONE - alpha) * t[j] + alpha * t[half + j])
    .collect()
}

/// Evaluate the quadratic through `(0,h0),(1,h1),(2,h2)` at `x`.
fn quad_eval<F: Field>(h: &[F; 3], x: F) -> F {
  let inv2 = Option::<F>::from((F::ONE + F::ONE).invert()).expect("2 invertible");
  let one = F::ONE;
  let two = one + one;
  h[0] * (x - one) * (x - two) * inv2 - h[1] * x * (x - two) + h[2] * x * (x - one) * inv2
}

/// `eq(z, α)` with the sumcheck's MSB-first challenge ordering: round `j`
/// (challenge `α[j]`) bound variable `b_{d−j}`, i.e. `z`-coordinate `z[d−1−j]`.
fn eq_at_fold<F: Field>(z: &[F], alphas: &[F]) -> F {
  let d = z.len();
  alphas
    .iter()
    .enumerate()
    .fold(F::ONE, |acc, (j, &a)| acc * eqf(z[d - 1 - j], a))
}

fn absorb_f<T: ByteTranscript, F: PrimeField>(sub: &mut T, label: &'static [u8], f: &F) {
  sub.absorb_bytes(label, f.to_repr().as_ref());
}

fn squeeze_scalar<T: ByteTranscript, F: PrimeFieldExt>(
  sub: &mut T,
  label: &'static [u8],
) -> Result<F, SpartanError> {
  Ok(F::from_uniform(&sub.squeeze_bytes(label)?))
}

fn squeeze_index<T: ByteTranscript>(
  sub: &mut T,
  label: &'static [u8],
  bound: usize,
) -> Result<usize, SpartanError> {
  let b = sub.squeeze_bytes(label)?;
  let mut e = [0u8; 8];
  e.copy_from_slice(&b[..8]);
  Ok((u64::from_le_bytes(e) % bound as u64) as usize)
}

fn root_of<F: PrimeField>(word: &[F]) -> Hash {
  let leaves: Vec<Hash> = word.iter().map(|x| hash_leaf(&leaf_bytes(x))).collect();
  MerkleTree::from_leaves(leaves).root()
}

/// Prove `ẑ(z) = y` for the committed evaluations `m_evals` (`c0 =
/// commit(encode(m_evals))`). The caller must have absorbed the commitment
/// into `sub` beforehand.
pub fn prove_eval<F: PrimeFieldExt>(
  code: &FoldableCode<F>,
  c0: &CommittedLayer<F>,
  m_evals: &[F],
  z: &[F],
  y: F,
  n_queries: usize,
  sub: &mut impl ByteTranscript,
) -> Result<EvalProof<F>, SpartanError> {
  let d = code.log_k();
  assert_eq!(m_evals.len(), 1usize << d);
  assert_eq!(z.len(), d);

  for zi in z {
    absorb_f(sub, b"bf-z", zi);
  }
  absorb_f(sub, b"bf-y", &y);

  let mut m = m_evals.to_vec();
  let mut eq = build_eq(z);
  let mut cur = c0.word.clone();
  let mut folded: Vec<CommittedLayer<F>> = Vec::with_capacity(d);
  let mut sumcheck: Vec<[F; 3]> = Vec::with_capacity(d);

  for i in 0..d {
    let half = m.len() / 2;
    let (mut h0, mut h1, mut h2) = (F::ZERO, F::ZERO, F::ZERO);
    for j in 0..half {
      h0 += eq[j] * m[j];
      h1 += eq[half + j] * m[half + j];
      let e2 = eq[half + j].double() - eq[j];
      let m2 = m[half + j].double() - m[j];
      h2 += e2 * m2;
    }
    absorb_f(sub, b"bf-h", &h0);
    absorb_f(sub, b"bf-h", &h1);
    absorb_f(sub, b"bf-h", &h2);
    let alpha: F = squeeze_scalar(sub, b"bf-a")?;
    sumcheck.push([h0, h1, h2]);

    m = fold_eval(&m, alpha);
    eq = fold_eval(&eq, alpha);
    cur = code.fold(&cur, alpha, d - i);
    let committed = commit_layer(cur.clone());
    sub.absorb_bytes(b"bf-cf", &committed.root);
    folded.push(committed);
  }

  let base = folded.last().expect("d>=1").word.clone();
  let layer_roots: Vec<Hash> = folded.iter().map(|l| l.root).collect();

  let word_at = |li: usize| -> &Vec<F> {
    if li == 0 {
      &c0.word
    } else {
      &folded[li - 1].word
    }
  };
  let tree_at = |li: usize| {
    if li == 0 {
      &c0.tree
    } else {
      &folded[li - 1].tree
    }
  };

  let n0 = code.codeword_len();
  let mut queries = Vec::with_capacity(n_queries);
  for _ in 0..n_queries {
    let mut s = squeeze_index(sub, b"bf-idx", n0)?;
    let mut opened = Vec::with_capacity(d);
    let mut len = n0;
    for li in 0..d {
      let half = len / 2;
      let lo = s % half;
      let hi = lo + half;
      let w = word_at(li);
      let t = tree_at(li);
      opened.push((w[lo], t.path(lo), w[hi], t.path(hi)));
      s = lo;
      len = half;
    }
    queries.push(QueryOpening { layers: opened });
  }

  Ok(EvalProof {
    sumcheck,
    layer_roots,
    base,
    queries,
  })
}

/// Verify a `prove_eval` proof of `ẑ(z) = y` against commitment `c0_root`.
/// `Ok(true)` = accept, `Ok(false)` = reject, `Err` = transcript failure.
pub fn verify_eval<F: PrimeFieldExt>(
  code: &FoldableCode<F>,
  c0_root: &Hash,
  z: &[F],
  y: F,
  proof: &EvalProof<F>,
  n_queries: usize,
  sub: &mut impl ByteTranscript,
) -> Result<bool, SpartanError> {
  let d = code.log_k();
  if proof.sumcheck.len() != d
    || proof.layer_roots.len() != d
    || proof.queries.len() != n_queries
    || z.len() != d
  {
    return Ok(false);
  }
  // Base must be a genuine base codeword, matching its committed root.
  let Some(v) = code.base_value(&proof.base) else {
    return Ok(false);
  };
  if root_of(&proof.base) != proof.layer_roots[d - 1] {
    return Ok(false);
  }

  for zi in z {
    absorb_f(sub, b"bf-z", zi);
  }
  absorb_f(sub, b"bf-y", &y);

  let mut claim = y;
  let mut alphas: Vec<F> = Vec::with_capacity(d);
  for i in 0..d {
    let [h0, h1, h2] = proof.sumcheck[i];
    if h0 + h1 != claim {
      return Ok(false);
    }
    absorb_f(sub, b"bf-h", &h0);
    absorb_f(sub, b"bf-h", &h1);
    absorb_f(sub, b"bf-h", &h2);
    let alpha: F = squeeze_scalar(sub, b"bf-a")?;
    sub.absorb_bytes(b"bf-cf", &proof.layer_roots[i]);
    claim = quad_eval(&[h0, h1, h2], alpha);
    alphas.push(alpha);
  }

  if claim != eq_at_fold(z, &alphas) * v {
    return Ok(false);
  }

  let roots: Vec<&Hash> = std::iter::once(c0_root)
    .chain(proof.layer_roots.iter())
    .collect();
  let n0 = code.codeword_len();
  for query in &proof.queries {
    if query.layers.len() != d {
      return Ok(false);
    }
    let mut s = squeeze_index(sub, b"bf-idx", n0)?;
    let mut chained: Option<F> = None;
    let mut len = n0;
    for (li, (vlo, plo, vhi, phi)) in query.layers.iter().enumerate() {
      let half = len / 2;
      let lo = s % half;
      let hi = lo + half;
      if !verify_path(roots[li], &leaf_bytes(vlo), lo, plo)
        || !verify_path(roots[li], &leaf_bytes(vhi), hi, phi)
      {
        return Ok(false);
      }
      let entry_at_s = if s < half { *vlo } else { *vhi };
      if let Some(exp) = chained
        && exp != entry_at_s
      {
        return Ok(false);
      }
      chained = Some(code.fold_pair(*vlo, *vhi, alphas[li], d - li, lo));
      s = lo;
      len = half;
    }
    if chained != Some(proof.base[s]) {
      return Ok(false);
    }
  }
  Ok(true)
}

#[cfg(test)]
mod tests {
  use super::super::code::rand_vec;
  use super::*;
  use crate::provider::T256HyraxEngine;
  use crate::provider::keccak::Keccak256Transcript;
  use crate::provider::pt256::t256;
  use crate::traits::transcript::TranscriptEngineTrait;

  type F = t256::Scalar;
  type Tr = Keccak256Transcript<T256HyraxEngine>;

  fn mle_eval(m: &[F], z: &[F]) -> F {
    build_eq(z).iter().zip(m).map(|(e, mi)| *e * *mi).sum()
  }

  #[test]
  fn eval_open_round_trip_and_tamper() {
    let d = 6;
    let c = 4;
    let nq = 24;
    let code = FoldableCode::<F>::new(d, c, b"eval-seed");
    let m: Vec<F> = rand_vec(1 << d, 5);
    let c0 = commit_layer(code.encode(&m));
    let z: Vec<F> = rand_vec(d, 8);
    let y = mle_eval(&m, &z);

    // Prover transcript: absorb the commitment (as the backend would), open.
    let mut tp: Tr = Keccak256Transcript::new_with_params(b"bf-test", ());
    tp.absorb_bytes(b"comm", &c0.root);
    let proof = prove_eval(&code, &c0, &m, &z, y, nq, &mut tp).unwrap();

    let verify = |y: F, proof: &EvalProof<F>| {
      let mut tv: Tr = Keccak256Transcript::new_with_params(b"bf-test", ());
      tv.absorb_bytes(b"comm", &c0.root);
      verify_eval(&code, &c0.root, &z, y, proof, nq, &mut tv).unwrap()
    };

    assert!(verify(y, &proof), "honest open verifies");
    assert!(!verify(y + F::ONE, &proof), "wrong y rejected");

    let mut bad = proof.clone();
    bad.sumcheck[0][0] += F::ONE;
    assert!(!verify(y, &bad), "tampered sumcheck rejected");

    let mut bad2 = proof.clone();
    bad2.queries[0].layers[0].0 += F::ONE;
    assert!(!verify(y, &bad2), "tampered opening rejected");
  }
}
