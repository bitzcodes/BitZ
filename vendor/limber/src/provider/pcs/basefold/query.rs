//! BaseFold's FRI-style **fold-and-query proximity IOP** — the soundness
//! core, built and tested here in isolation before the sumcheck eval-binding
//! is layered on (that turns this proximity test into a full multilinear PCS
//! opening).
//!
//! Prover commits `c⁰ = encode(msg)`, then folds `c⁰ → c¹ → … → cᵈ` with
//! Fiat–Shamir challenges, Merkle-committing each layer. The verifier makes
//! `n_queries` random spot-checks: for each, it walks a query index `s` down
//! the layers, checks each fold step (`child = fold(cⁱ[lo], cⁱ[lo+half], αᵢ)`),
//! the chaining (`child` reappears as the tracked entry of the next layer),
//! and the Merkle openings — ending at the shipped base codeword. A committed
//! word that isn't a genuine fold chain fails w.h.p.
//!
//! Fiat–Shamir here is a self-contained Shake stream; it becomes the crate
//! `ByteTranscript` when wired behind `CommitBackend`.

use super::super::brakedown::merkle::{Hash, MerkleTree, hash_leaf, verify_path};
use super::code::FoldableCode;
use crate::traits::PrimeFieldExt;
use ff::PrimeField;
use sha3::Shake256;
use sha3::digest::{ExtendableOutput, Update, XofReader};

pub(crate) fn leaf_bytes<F: PrimeField>(x: &F) -> Vec<u8> {
  x.to_repr().as_ref().to_vec()
}

/// Prover-side committed codeword layer: Merkle tree + raw entries.
pub struct CommittedLayer<F> {
  /// Merkle root over the per-entry leaves.
  pub root: Hash,
  pub(crate) tree: MerkleTree,
  pub(crate) word: Vec<F>,
}

/// Commit one codeword layer (Merkle over per-entry leaves).
pub fn commit_layer<F: PrimeFieldExt>(word: Vec<F>) -> CommittedLayer<F> {
  let leaves: Vec<Hash> = word.iter().map(|x| hash_leaf(&leaf_bytes(x))).collect();
  let tree = MerkleTree::from_leaves(leaves);
  CommittedLayer {
    root: tree.root(),
    tree,
    word,
  }
}

/// One query's openings: `(v_lo, path_lo, v_hi, path_hi)` per fold layer.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryOpening<F> {
  /// Per layer `0..d`: the two folded parent entries with Merkle paths.
  pub layers: Vec<(F, Vec<Hash>, F, Vec<Hash>)>,
}

/// A proximity proof: the fold-layer roots (`c¹..cᵈ`; `c⁰`'s root is the
/// commitment), the base codeword `cᵈ`, and the per-query openings.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProximityProof<F> {
  /// Merkle roots of `c¹, …, cᵈ`.
  pub layer_roots: Vec<Hash>,
  /// The base codeword `cᵈ` (length `c`), shipped in full.
  pub base: Vec<F>,
  /// Per-query fold-consistency openings.
  pub queries: Vec<QueryOpening<F>>,
}

fn absorb(x: &mut Shake256, label: &[u8], bytes: &[u8]) {
  x.update(&(label.len() as u64).to_le_bytes());
  x.update(label);
  x.update(&(bytes.len() as u64).to_le_bytes());
  x.update(bytes);
}

fn squeeze_scalar<F: PrimeFieldExt>(x: &Shake256) -> F {
  let mut r = x.clone().finalize_xof();
  let mut b = [0u8; 64];
  r.read(&mut b);
  F::from_uniform(&b)
}

fn squeeze_index(x: &Shake256, ctr: u64, bound: usize) -> usize {
  let mut h = x.clone();
  h.update(b"idx");
  h.update(&ctr.to_le_bytes());
  let mut r = h.finalize_xof();
  let mut b = [0u8; 8];
  r.read(&mut b);
  (u64::from_le_bytes(b) % bound as u64) as usize
}

/// Prove proximity of the committed codeword `c0` to the fold chain of `code`.
pub fn prove<F: PrimeFieldExt>(
  code: &FoldableCode<F>,
  c0: &CommittedLayer<F>,
  n_queries: usize,
) -> ProximityProof<F> {
  let d = code.log_k();
  let mut fs = Shake256::default();
  absorb(&mut fs, b"c0", &c0.root);

  // Fold c0 -> c1 -> ... -> cd, committing each layer; challenges from FS.
  let mut folded: Vec<CommittedLayer<F>> = Vec::with_capacity(d);
  let mut cur: Vec<F> = c0.word.clone();
  for level in (1..=d).rev() {
    let alpha: F = squeeze_scalar(&fs);
    cur = code.fold(&cur, alpha, level);
    let committed = commit_layer(cur.clone());
    absorb(&mut fs, b"cf", &committed.root);
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
  let tree_at = |li: usize| -> &MerkleTree {
    if li == 0 {
      &c0.tree
    } else {
      &folded[li - 1].tree
    }
  };

  let n0 = code.codeword_len();
  let mut queries = Vec::with_capacity(n_queries);
  for q in 0..n_queries {
    let mut s = squeeze_index(&fs, q as u64, n0);
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

  ProximityProof {
    layer_roots,
    base,
    queries,
  }
}

/// Verify a proximity proof against the commitment root `c0_root`.
pub fn verify<F: PrimeFieldExt>(
  code: &FoldableCode<F>,
  c0_root: &Hash,
  proof: &ProximityProof<F>,
  n_queries: usize,
) -> bool {
  let d = code.log_k();
  if proof.layer_roots.len() != d || proof.queries.len() != n_queries {
    return false;
  }
  // The base must be a genuine base codeword, and match its committed root.
  if code.base_value(&proof.base).is_none() {
    return false;
  }
  let base_root = {
    let leaves: Vec<Hash> = proof
      .base
      .iter()
      .map(|x| hash_leaf(&leaf_bytes(x)))
      .collect();
    MerkleTree::from_leaves(leaves).root()
  };
  if base_root != proof.layer_roots[d - 1] {
    return false;
  }

  // Re-derive Fiat-Shamir challenges and absorb the roots in order.
  let mut fs = Shake256::default();
  absorb(&mut fs, b"c0", c0_root);
  let mut alphas: Vec<(usize, F)> = Vec::with_capacity(d);
  for (i, level) in (1..=d).rev().enumerate() {
    let alpha: F = squeeze_scalar(&fs);
    alphas.push((level, alpha));
    absorb(&mut fs, b"cf", &proof.layer_roots[i]);
  }

  let roots: Vec<&Hash> = std::iter::once(c0_root)
    .chain(proof.layer_roots.iter())
    .collect();
  let n0 = code.codeword_len();
  for (q, query) in proof.queries.iter().enumerate() {
    if query.layers.len() != d {
      return false;
    }
    let mut s = squeeze_index(&fs, q as u64, n0);
    let mut chained: Option<F> = None;
    let mut len = n0;
    for (li, (vlo, plo, vhi, phi)) in query.layers.iter().enumerate() {
      let half = len / 2;
      let lo = s % half;
      let hi = lo + half;
      if !verify_path(roots[li], &leaf_bytes(vlo), lo, plo)
        || !verify_path(roots[li], &leaf_bytes(vhi), hi, phi)
      {
        return false;
      }
      // The tracked entry is at index s == lo (if s<half) or hi.
      let entry_at_s = if s < half { *vlo } else { *vhi };
      if let Some(exp) = chained
        && exp != entry_at_s
      {
        return false;
      }
      let (level, alpha) = alphas[li];
      chained = Some(code.fold_pair(*vlo, *vhi, alpha, level, lo));
      s = lo;
      len = half;
    }
    // Final folded value must equal the base codeword at the walked index.
    if chained != Some(proof.base[s]) {
      return false;
    }
  }
  true
}

#[cfg(test)]
mod tests {
  use super::super::code::rand_vec;
  use super::*;
  use crate::provider::pt256::t256;
  use ff::Field;

  type F = t256::Scalar;

  #[test]
  fn proximity_round_trip_and_tamper() {
    let log_k = 6;
    let c = 4;
    let nq = 24;
    let code = FoldableCode::<F>::new(log_k, c, b"prox-seed");
    let msg: Vec<F> = rand_vec(1 << log_k, 3);
    let c0 = commit_layer(code.encode(&msg));

    let proof = prove(&code, &c0, nq);
    assert!(verify(&code, &c0.root, &proof, nq), "honest proof verifies");

    // Tamper the base -> reject.
    let mut bad = proof.clone();
    bad.base[0] += F::ONE;
    assert!(!verify(&code, &c0.root, &bad, nq), "tampered base rejected");

    // Tamper a queried opening value -> reject.
    let mut bad2 = proof.clone();
    bad2.queries[0].layers[0].0 += F::ONE;
    assert!(
      !verify(&code, &c0.root, &bad2, nq),
      "tampered opening rejected"
    );

    // A codeword NOT in the code (random word) must fail proximity.
    let junk: Vec<F> = rand_vec(code.codeword_len(), 999);
    let cj = commit_layer(junk);
    let pj = prove(&code, &cj, nq);
    assert!(!verify(&code, &cj.root, &pj, nq), "non-codeword rejected");
  }
}
