//! Brakedown evaluation opening — the tensor IOPP.
//!
//! The committed poly is the `n_rows × row_len` message matrix `M`. An MLE eval
//! at `r = (r_hi, r_lo)` factors as `eval = ⟨ e_row · M , e_col ⟩` with
//! `e_row = eq(r_hi)` (length `n_rows`), `e_col = eq(r_lo)` (length `row_len`).
//!
//! The prover sends two combined message rows — `w_prox = r_comb·M` for a random
//! `r_comb` (proximity / well-formedness) and `w_eval = e_row·M` (the eval) —
//! plus `t` opened columns of the encoded matrix (with Merkle paths). For each
//! opened column `c` the verifier checks, using the code's linearity
//! (`combo·EncodedMatrix = Enc(combo·M)`):
//! `Enc(w_prox)[c] == ⟨r_comb, col_c⟩` and `Enc(w_eval)[c] == ⟨e_row, col_c⟩`,
//! and finally `eval == ⟨w_eval, e_col⟩`.

use super::{
  code::{next_index, next_scalar, xof},
  commit::{BrakedownCommitData, BrakedownParams, column_to_bytes, commit},
  merkle::{Hash, hash_leaf, verify_batch_path},
};
use crate::{
  errors::SpartanError,
  traits::{PrimeFieldExt, transcript::ByteTranscript},
};
use ff::Field;
use std::collections::HashMap;

/// Evaluation argument: the proximity row, the eval row, the opened columns
/// (sorted-unique `(index, entries)`), and one batched Merkle multiproof
/// covering all of them.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
  serialize = "F: serde::Serialize + ff::PrimeField",
  deserialize = "F: serde::de::DeserializeOwned + ff::PrimeField"
))]
pub struct BrakedownEvalArg<F> {
  w_prox: Vec<F>,
  w_eval: Vec<F>,
  /// Opened columns in sorted-unique challenge order. The indices are
  /// NOT shipped: the verifier re-derives them from the transcript, so
  /// they were pure wire redundancy. Entries use the compact
  /// length-prefixed encoding (committed chunk data is mostly small).
  #[serde(with = "compact_columns")]
  columns: Vec<Vec<F>>,
  auth: Vec<Hash>,
}

/// Length-prefixed field-element wire encoding for opened columns:
/// each entry is `len (1 byte) || minimal little-endian repr bytes`.
/// Bijective with canonical field elements, so the Merkle/consistency
/// checks are unaffected; on chunk-granularity committed data this
/// shrinks the dominant proof component ~2x versus 32-byte reprs.
mod compact_columns {
  use ff::PrimeField;
  use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

  pub fn serialize<F: PrimeField, S: Serializer>(
    cols: &[Vec<F>],
    ser: S,
  ) -> Result<S::Ok, S::Error> {
    let encoded: Vec<Vec<u8>> = cols
      .iter()
      .map(|col| {
        let mut buf = Vec::with_capacity(col.len() * 8);
        for x in col {
          let repr = x.to_repr();
          let bytes = repr.as_ref();
          let len = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
          buf.push(len as u8);
          buf.extend_from_slice(&bytes[..len]);
        }
        buf
      })
      .collect();
    encoded.serialize(ser)
  }

  pub fn deserialize<'de, F: PrimeField, D: Deserializer<'de>>(
    de: D,
  ) -> Result<Vec<Vec<F>>, D::Error> {
    let encoded: Vec<Vec<u8>> = Vec::deserialize(de)?;
    encoded
      .into_iter()
      .map(|buf| {
        let mut out = Vec::new();
        let mut i = 0;
        while i < buf.len() {
          let len = buf[i] as usize;
          i += 1;
          let mut repr = F::Repr::default();
          let dst = repr.as_mut();
          if len > dst.len() || i + len > buf.len() {
            return Err(D::Error::custom("bad column entry length"));
          }
          dst[..len].copy_from_slice(&buf[i..i + len]);
          if len > 0 && dst[len - 1] == 0 {
            return Err(D::Error::custom("non-minimal column entry"));
          }
          i += len;
          let f =
            Option::<F>::from(F::from_repr(repr)).ok_or_else(|| D::Error::custom("bad entry"))?;
          out.push(f);
        }
        Ok(out)
      })
      .collect()
  }
}

impl<F: serde::Serialize> BrakedownEvalArg<F> {
  /// Serialized size of each component, for proof-size accounting:
  /// `(w_prox, w_eval, columns, auth)` in bytes.
  pub fn component_sizes(&self) -> (usize, usize, usize, usize) {
    (
      bincode::serialized_size(&self.w_prox).unwrap_or(0) as usize,
      bincode::serialized_size(&self.w_eval).unwrap_or(0) as usize,
      bincode::serialized_size(&self.columns).unwrap_or(0) as usize,
      bincode::serialized_size(&self.auth).unwrap_or(0) as usize,
    )
  }
}

/// eq-polynomial evaluations over `r`, high-bit-first: `out[k] = ∏_b (k_b?
/// r_b : 1-r_b)` with `r[0]` the most-significant index bit. Factorizes as
/// `eq(r_hi||r_lo)[i·2^|lo| + j] = eq(r_hi)[i]·eq(r_lo)[j]`.
fn eq_evals<F: Field>(r: &[F]) -> Vec<F> {
  let mut e = vec![F::ONE];
  for &ri in r {
    let mut next = Vec::with_capacity(e.len() * 2);
    for &ev in &e {
      next.push(ev * (F::ONE - ri));
      next.push(ev * ri);
    }
    e = next;
  }
  e
}

fn inner<F: Field>(a: &[F], b: &[F]) -> F {
  a.iter().zip(b).fold(F::ZERO, |acc, (x, y)| acc + *x * *y)
}

/// `combo · M` over the systematic (message) columns: `out[j] = Σ_i combo[i] ·
/// encoded[i][j]` for `j < row_len`.
fn combine_message_rows<F: Field>(encoded: &[Vec<F>], combo: &[F], row_len: usize) -> Vec<F> {
  let mut out = vec![F::ZERO; row_len];
  for (i, row) in encoded.iter().enumerate() {
    let c = combo[i];
    for (o, &m) in out.iter_mut().zip(&row[..row_len]) {
      *o += c * m;
    }
  }
  out
}

fn expand_scalars<F: PrimeFieldExt>(seed: &[u8; 64], n: usize) -> Vec<F> {
  let mut r = xof(seed, b"sc");
  (0..n).map(|_| next_scalar::<F>(&mut r)).collect()
}

fn expand_indices(seed: &[u8; 64], count: usize, bound: usize) -> Vec<usize> {
  let mut r = xof(seed, b"idx");
  (0..count).map(|_| next_index(&mut r, bound)).collect()
}

fn verr(reason: &str) -> SpartanError {
  SpartanError::ProofVerifyError {
    reason: format!("brakedown: {reason}"),
  }
}

/// Prove `poly(point) = eval`. Re-derives the commitment internally (so the
/// caller need only hold the root); returns the evaluation and its argument.
pub fn open<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  poly: &[F],
  point: &[F],
  transcript: &mut impl ByteTranscript,
) -> Result<(F, BrakedownEvalArg<F>), SpartanError> {
  let (root, data) = commit(params, poly);
  open_with_data(params, &root, &data, point, transcript)
}

/// A batch opening group: several commitments sharing one code layout
/// (`row_len`, spec, seed) and opened at points with a common suffix —
/// the last `log2(row_len)` coordinates, which determine the column
/// weights — ship ONE proximity row and ONE gamma-combined evaluation
/// row (the per-target rows were 80% of the compressed proof), plus
/// per-tree columns and authentication. Member polynomials may have
/// DIFFERENT lengths: each tree contributes its own rows and its own
/// row-weight vector; only the column side of the tensor point must
/// agree. Soundness is standard RLC batching: the proximity
/// combination spans every tree's rows, and gamma is squeezed after
/// all roots and claims are transcript-bound.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
  serialize = "F: serde::Serialize + ff::PrimeField",
  deserialize = "F: serde::de::DeserializeOwned + ff::PrimeField"
))]
pub struct BrakedownGroupArg<F> {
  w_prox: Vec<F>,
  w_eval: Vec<F>,
  /// Per tree, in group order: opened columns (sorted-unique challenge
  /// order, compact encoding) and the Merkle multiproof.
  trees: Vec<TreeOpening<F>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
  serialize = "F: serde::Serialize + ff::PrimeField",
  deserialize = "F: serde::de::DeserializeOwned + ff::PrimeField"
))]
/// One tree's share of a [`BrakedownGroupArg`]: its opened columns and
/// Merkle multiproof.
pub struct TreeOpening<F> {
  #[serde(with = "compact_columns")]
  columns: Vec<Vec<F>>,
  auth: Vec<Hash>,
}

impl<F: serde::Serialize + ff::PrimeField> BrakedownGroupArg<F> {
  /// `(rows, columns, auth)` serialized bytes, for anatomy accounting.
  pub fn component_sizes(&self) -> (usize, usize, usize) {
    let rows = bincode::serialized_size(&self.w_prox).unwrap_or(0)
      + bincode::serialized_size(&self.w_eval).unwrap_or(0);
    let mut cols = 0u64;
    let mut auth = 0u64;
    for t in &self.trees {
      cols += {
        // compact encoding size via a probe serialize of the wrapper
        bincode::serialized_size(t).unwrap_or(0) - bincode::serialized_size(&t.auth).unwrap_or(0)
      };
      auth += bincode::serialized_size(&t.auth).unwrap_or(0);
    }
    (rows as usize, cols as usize, auth as usize)
  }
}

/// A directly-shipped small polynomial. For tiny non-hiding
/// commitments the compact polynomial itself is smaller than any
/// column-opening argument (a group's row pair alone is `2·row_len`
/// dense field elements), so the verifier just recommits the shipped
/// coefficients, checks the root, and evaluates the claim directly.
/// No transcript interaction is involved.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
  serialize = "F: serde::Serialize + ff::PrimeField",
  deserialize = "F: serde::de::DeserializeOwned + ff::PrimeField"
))]
pub struct BrakedownDirectOpen<F> {
  /// The polynomial's coefficients (compact encoding; single entry).
  #[serde(with = "compact_columns")]
  poly: Vec<Vec<F>>,
}

impl<F: serde::Serialize + ff::PrimeField> BrakedownDirectOpen<F> {
  /// Serialized (compact wire-format) size in bytes, for anatomy
  /// accounting.
  pub fn size(&self) -> usize {
    bincode::serialized_size(self).unwrap_or(0) as usize
  }
}

/// Package a small polynomial for direct shipping.
pub fn open_direct<F: PrimeFieldExt>(poly: &[F]) -> BrakedownDirectOpen<F> {
  BrakedownDirectOpen {
    poly: vec![poly.to_vec()],
  }
}

/// Verify a directly-shipped polynomial: re-hash the shipped bytes
/// against the plain-hash commitment (see
/// [`commit_plain`](super::commit::commit_plain)) and evaluate the multilinear
/// claim directly.
pub fn verify_direct<F: PrimeFieldExt>(
  root: &Hash,
  point: &[F],
  eval: F,
  arg: &BrakedownDirectOpen<F>,
) -> Result<(), SpartanError> {
  if arg.poly.len() != 1 {
    return Err(verr("direct: malformed argument"));
  }
  let poly = &arg.poly[0];
  let n = 1usize << point.len();
  if poly.len() > n {
    return Err(verr("direct: polynomial length mismatch"));
  }
  let mut padded;
  let full = if poly.len() == n {
    poly.as_slice()
  } else {
    padded = poly.clone();
    padded.resize(n, F::ZERO);
    padded.as_slice()
  };
  if hash_leaf(&column_to_bytes(full)) != *root {
    return Err(verr("direct: commitment mismatch"));
  }
  if inner(&eq_evals(point), full) != eval {
    return Err(verr("direct: claimed evaluation does not match"));
  }
  Ok(())
}

/// The number of matrix rows a point of this length implies under
/// `params`' shared row length.
fn group_rows<F: PrimeFieldExt>(params: &BrakedownParams<F>, point_len: usize) -> usize {
  let log_cols = params.row_len.trailing_zeros() as usize;
  1usize << point_len.saturating_sub(log_cols)
}

/// Open a group of commitments sharing `params`' code at points with a
/// common suffix. `items` are `(root, data, full point)` in canonical
/// order; every `data` must use `params`' `row_len` (heights may
/// differ). Returns the per-item evaluations and the argument.
pub fn open_group<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  items: &[(&Hash, &BrakedownCommitData<F>, &[F])],
  transcript: &mut impl ByteTranscript,
) -> Result<(Vec<F>, BrakedownGroupArg<F>), SpartanError> {
  let log_cols = params.row_len.trailing_zeros() as usize;
  for (root, data, point) in items {
    debug_assert!(
      point.len() >= log_cols,
      "point shorter than the column part"
    );
    debug_assert_eq!(
      data.encoded.len(),
      group_rows(params, point.len()),
      "commit data height does not match the point length"
    );
    debug_assert_eq!(
      &point[point.len() - log_cols..],
      &items[0].2[items[0].2.len() - log_cols..],
      "grouped points must share their column suffix"
    );
    transcript.absorb_bytes(b"bd_root", *root);
  }
  let suffix = &items[0].2[items[0].2.len() - log_cols..];
  let e_col = eq_evals(suffix);
  let total_rows: usize = items
    .iter()
    .map(|(_, _, p)| group_rows(params, p.len()))
    .sum();

  // ONE proximity combination spanning every tree's rows.
  let seed_rc = transcript.squeeze_bytes(b"bd_rcomb")?;
  let r_all = expand_scalars::<F>(&seed_rc, total_rows);
  let mut w_prox = vec![F::ZERO; params.row_len];
  let mut off = 0usize;
  for (_, data, point) in items {
    let rows = group_rows(params, point.len());
    let part = combine_message_rows(&data.encoded, &r_all[off..off + rows], params.row_len);
    for (o, v) in w_prox.iter_mut().zip(part) {
      *o += v;
    }
    off += rows;
  }

  // Per-item evaluation rows (each with its OWN row weights, from the
  // prefix of its point), gamma-combined AFTER the claims are implied
  // (gamma depends on the transcript state binding the roots).
  let gamma = F::from_uniform(&transcript.squeeze_bytes(b"bd_gamma")?);
  let mut w_eval = vec![F::ZERO; params.row_len];
  let mut evals = Vec::with_capacity(items.len());
  let mut g = F::ONE;
  for (_, data, point) in items {
    let e_row = eq_evals(&point[..point.len() - log_cols]);
    let part = combine_message_rows(&data.encoded, &e_row, params.row_len);
    evals.push(inner(&part, &e_col));
    for (o, v) in w_eval.iter_mut().zip(part) {
      *o += g * v;
    }
    g *= gamma;
  }
  transcript.absorb_bytes(b"bd_wprox", &column_to_bytes(&w_prox));
  transcript.absorb_bytes(b"bd_weval", &column_to_bytes(&w_eval));

  let seed_cols = transcript.squeeze_bytes(b"bd_cols")?;
  let idxs = expand_indices(&seed_cols, params.n_col_opens, params.n_cols);
  let mut unique = idxs;
  unique.sort_unstable();
  unique.dedup();
  let trees = items
    .iter()
    .map(|(_, data, _)| TreeOpening {
      columns: unique
        .iter()
        .map(|&c| data.encoded.iter().map(|row| row[c]).collect())
        .collect(),
      auth: data.tree.batch_path(&unique),
    })
    .collect();

  Ok((
    evals,
    BrakedownGroupArg {
      w_prox,
      w_eval,
      trees,
    },
  ))
}

/// Verifier mirror of [`open_group`]: `items` are `(root, full point,
/// claimed eval)` in the same canonical order.
pub fn verify_group<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  items: &[(&Hash, &[F], F)],
  arg: &BrakedownGroupArg<F>,
  transcript: &mut impl ByteTranscript,
) -> Result<(), SpartanError> {
  if arg.trees.len() != items.len() {
    return Err(verr("group: wrong number of tree openings"));
  }
  let log_cols = params.row_len.trailing_zeros() as usize;
  for (root, point, _) in items {
    if point.len() < log_cols
      || point[point.len() - log_cols..] != items[0].1[items[0].1.len() - log_cols..]
    {
      return Err(verr("group: points do not share a column suffix"));
    }
    transcript.absorb_bytes(b"bd_root", *root);
  }
  let suffix = &items[0].1[items[0].1.len() - log_cols..];
  let e_col = eq_evals(suffix);
  let total_rows: usize = items
    .iter()
    .map(|(_, p, _)| group_rows(params, p.len()))
    .sum();

  let seed_rc = transcript.squeeze_bytes(b"bd_rcomb")?;
  let r_all = expand_scalars::<F>(&seed_rc, total_rows);
  let gamma = F::from_uniform(&transcript.squeeze_bytes(b"bd_gamma")?);
  if arg.w_prox.len() != params.row_len || arg.w_eval.len() != params.row_len {
    return Err(verr("group: bad combined-row length"));
  }
  transcript.absorb_bytes(b"bd_wprox", &column_to_bytes(&arg.w_prox));
  transcript.absorb_bytes(b"bd_weval", &column_to_bytes(&arg.w_eval));

  let seed_cols = transcript.squeeze_bytes(b"bd_cols")?;
  let idxs = expand_indices(&seed_cols, params.n_col_opens, params.n_cols);
  let mut unique = idxs.clone();
  unique.sort_unstable();
  unique.dedup();

  let height = params.n_cols.next_power_of_two().trailing_zeros() as usize;
  for ((root, point, _), tree) in items.iter().zip(&arg.trees) {
    let rows = group_rows(params, point.len());
    if tree.columns.len() != unique.len() {
      return Err(verr("group: wrong number of opened columns"));
    }
    for col in &tree.columns {
      if col.len() != rows {
        return Err(verr("group: opened column has wrong height"));
      }
    }
    let leaves: Vec<(usize, Hash)> = unique
      .iter()
      .zip(&tree.columns)
      .map(|(&i, entries)| (i, hash_leaf(&column_to_bytes(entries))))
      .collect();
    if !verify_batch_path(root, height, &leaves, &tree.auth) {
      return Err(verr("group: Merkle multiproof check failed"));
    }
  }

  let enc_prox = params.code.encode(&arg.w_prox);
  let enc_eval = params.code.encode(&arg.w_eval);
  let pos: std::collections::HashMap<usize, usize> =
    unique.iter().enumerate().map(|(k, &c)| (c, k)).collect();
  let gammas: Vec<F> = {
    let mut v = Vec::with_capacity(items.len());
    let mut g = F::ONE;
    for _ in items {
      v.push(g);
      g *= gamma;
    }
    v
  };
  let e_rows: Vec<Vec<F>> = items
    .iter()
    .map(|(_, point, _)| eq_evals(&point[..point.len() - log_cols]))
    .collect();
  let offsets: Vec<usize> = {
    let mut v = Vec::with_capacity(items.len());
    let mut off = 0;
    for (_, point, _) in items {
      v.push(off);
      off += group_rows(params, point.len());
    }
    v
  };
  for &c in &idxs {
    let k = *pos.get(&c).ok_or_else(|| verr("group: missing column"))?;
    let mut prox = F::ZERO;
    let mut ev = F::ZERO;
    for (t, tree) in arg.trees.iter().enumerate() {
      let col = &tree.columns[k];
      prox += inner(&r_all[offsets[t]..offsets[t] + col.len()], col);
      ev += gammas[t] * inner(&e_rows[t], col);
    }
    if prox != enc_prox[c] {
      return Err(verr("group: proximity check failed"));
    }
    if ev != enc_eval[c] {
      return Err(verr("group: eval-consistency check failed"));
    }
  }

  // The gamma-combined row must open to the gamma-combined claims.
  let combined_claim = items
    .iter()
    .zip(&gammas)
    .fold(F::ZERO, |acc, ((_, _, e), g)| acc + *g * *e);
  if inner(&arg.w_eval, &e_col) != combined_claim {
    return Err(verr("group: claimed evaluations do not match"));
  }
  Ok(())
}

/// Open a single commitment at `point` (ungrouped path; grouped
/// same-point openings should prefer [`open_group`]).
pub fn open_with_data<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  root: &Hash,
  data: &BrakedownCommitData<F>,
  point: &[F],
  transcript: &mut impl ByteTranscript,
) -> Result<(F, BrakedownEvalArg<F>), SpartanError> {
  transcript.absorb_bytes(b"bd_root", root);
  let log_rows = params.n_rows.trailing_zeros() as usize;
  let e_row = eq_evals(&point[..log_rows]);
  let e_col = eq_evals(&point[log_rows..]);

  let seed_rc = transcript.squeeze_bytes(b"bd_rcomb")?;
  let r_comb = expand_scalars::<F>(&seed_rc, params.n_rows);
  let w_prox = combine_message_rows(&data.encoded, &r_comb, params.row_len);
  let w_eval = combine_message_rows(&data.encoded, &e_row, params.row_len);
  transcript.absorb_bytes(b"bd_wprox", &column_to_bytes(&w_prox));
  transcript.absorb_bytes(b"bd_weval", &column_to_bytes(&w_eval));
  let eval = inner(&w_eval, &e_col);

  let seed_cols = transcript.squeeze_bytes(b"bd_cols")?;
  let idxs = expand_indices(&seed_cols, params.n_col_opens, params.n_cols);
  let mut unique = idxs;
  unique.sort_unstable();
  unique.dedup();
  let columns: Vec<Vec<F>> = unique
    .iter()
    .map(|&c| data.encoded.iter().map(|row| row[c]).collect())
    .collect();
  let auth = data.tree.batch_path(&unique);
  Ok((
    eval,
    BrakedownEvalArg {
      w_prox,
      w_eval,
      columns,
      auth,
    },
  ))
}

/// Verify that `point` opens `comm` (the Merkle root) to `eval`.
pub fn verify_open<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  root: &Hash,
  point: &[F],
  eval: F,
  arg: &BrakedownEvalArg<F>,
  transcript: &mut impl ByteTranscript,
) -> Result<(), SpartanError> {
  transcript.absorb_bytes(b"bd_root", root);
  let log_rows = params.n_rows.trailing_zeros() as usize;
  let e_row = eq_evals(&point[..log_rows]);
  let e_col = eq_evals(&point[log_rows..]);

  let seed_rc = transcript.squeeze_bytes(b"bd_rcomb")?;
  let r_comb = expand_scalars::<F>(&seed_rc, params.n_rows);
  if arg.w_prox.len() != params.row_len || arg.w_eval.len() != params.row_len {
    return Err(verr("bad combined-row length"));
  }
  transcript.absorb_bytes(b"bd_wprox", &column_to_bytes(&arg.w_prox));
  transcript.absorb_bytes(b"bd_weval", &column_to_bytes(&arg.w_eval));

  let seed_cols = transcript.squeeze_bytes(b"bd_cols")?;
  let idxs = expand_indices(&seed_cols, params.n_col_opens, params.n_cols);
  let mut unique = idxs.clone();
  unique.sort_unstable();
  unique.dedup();

  // The columns arrive in sorted-unique challenge order (indices are
  // re-derived, not shipped).
  if arg.columns.len() != unique.len() {
    return Err(verr("wrong number of opened columns"));
  }
  for col in &arg.columns {
    if col.len() != params.n_rows {
      return Err(verr("opened column has wrong height"));
    }
  }

  // One batched Merkle multiproof over all opened columns.
  let height = params.n_cols.next_power_of_two().trailing_zeros() as usize;
  let leaves: Vec<(usize, Hash)> = unique
    .iter()
    .zip(&arg.columns)
    .map(|(&i, entries)| (i, hash_leaf(&column_to_bytes(entries))))
    .collect();
  if !verify_batch_path(root, height, &leaves, &arg.auth) {
    return Err(verr("Merkle multiproof check failed"));
  }

  // Encode the two combined rows once (verifier-side, O(row_len)).
  let enc_prox = params.code.encode(&arg.w_prox);
  let enc_eval = params.code.encode(&arg.w_eval);

  // Per-challenge checks (a duplicate challenge index reuses its column).
  let lookup: HashMap<usize, &[F]> = unique
    .iter()
    .zip(&arg.columns)
    .map(|(&i, e)| (i, e.as_slice()))
    .collect();
  for &c in &idxs {
    let entries = lookup
      .get(&c)
      .ok_or_else(|| verr("missing opened column"))?;
    if inner(&r_comb, entries) != enc_prox[c] {
      return Err(verr("proximity check failed"));
    }
    if inner(&e_row, entries) != enc_eval[c] {
      return Err(verr("eval-consistency check failed"));
    }
  }

  if inner(&arg.w_eval, &e_col) != eval {
    return Err(verr("claimed evaluation does not match"));
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{T256HyraxEngine, keccak::Keccak256Transcript, pcs::brakedown::code::DEFAULT_SPEC},
    traits::{Engine, transcript::TranscriptEngineTrait},
  };

  type E = T256HyraxEngine;
  type F = <E as Engine>::Scalar;

  fn rand_vec(n: usize, tag: u64) -> Vec<F> {
    let mut r = xof(b"bd-eval-test", &tag.to_le_bytes());
    (0..n).map(|_| next_scalar::<F>(&mut r)).collect()
  }

  fn run(log_n: usize) {
    let n = 1usize << log_n;
    let params = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
    let poly = rand_vec(n, 1);
    let point = rand_vec(log_n, 2);

    // brute-force MLE eval (same eq convention as the reshape)
    let expected = inner(&poly, &eq_evals(&point));

    let (root, _) = commit(&params, &poly);
    let mut pt = Keccak256Transcript::<E>::new(b"bd");
    let (eval, arg) = open(&params, &poly, &point, &mut pt).unwrap();
    assert_eq!(
      eval, expected,
      "protocol eval must equal MLE eval (log_n={log_n})"
    );

    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    verify_open(&params, &root, &point, eval, &arg, &mut vt).unwrap();
  }

  #[test]
  fn roundtrip_eval_matches_mle() {
    for log_n in [4usize, 8, 12, 14] {
      run(log_n);
    }
  }

  #[test]
  fn verify_rejects_tampering() {
    let log_n = 12;
    let n = 1usize << log_n;
    let params = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
    let poly = rand_vec(n, 1);
    let point = rand_vec(log_n, 2);
    let (root, _) = commit(&params, &poly);
    let mut pt = Keccak256Transcript::<E>::new(b"bd");
    let (eval, arg) = open(&params, &poly, &point, &mut pt).unwrap();

    // wrong claimed eval
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &root, &point, eval + F::ONE, &arg, &mut vt).is_err());

    // tampered eval row
    let mut bad = arg.clone();
    bad.w_eval[0] += F::ONE;
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &root, &point, eval, &bad, &mut vt).is_err());

    // tampered opened column entry
    let mut bad = arg.clone();
    bad.columns[0][0] += F::ONE;
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &root, &point, eval, &bad, &mut vt).is_err());

    // wrong commitment root
    let mut wrong_root = root;
    wrong_root[0] ^= 1;
    let mut vt = Keccak256Transcript::<E>::new(b"bd");
    assert!(verify_open(&params, &wrong_root, &point, eval, &arg, &mut vt).is_err());
  }

  /// Open/verify time and proof size across sizes. Run with:
  ///   RAYON_NUM_THREADS=1 cargo test --release --lib -- --ignored --nocapture eval_bench
  /// and again without the env var.
  #[test]
  #[ignore = "benchmark; run with --release --ignored --nocapture"]
  fn eval_bench() {
    use std::time::Instant;
    println!(
      "\n== Brakedown open/verify/proof (threads={}) ==",
      rayon::current_num_threads()
    );
    for log_n in [12usize, 14, 16, 18] {
      let n = 1usize << log_n;
      let params = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
      let poly = rand_vec(n, 1);
      let point = rand_vec(log_n, 2);
      let (root, _) = commit(&params, &poly);

      let mut pt = Keccak256Transcript::<E>::new(b"bd");
      let t0 = Instant::now();
      let (eval, arg) = open(&params, &poly, &point, &mut pt).unwrap();
      let open_ms = t0.elapsed().as_secs_f64() * 1e3; // incl. internal re-commit

      let mut vt = Keccak256Transcript::<E>::new(b"bd");
      let t1 = Instant::now();
      verify_open(&params, &root, &point, eval, &arg, &mut vt).unwrap();
      let ver_ms = t1.elapsed().as_secs_f64() * 1e3;

      let fb = 32usize; // bytes per field element / hash
      let rows_b = (arg.w_prox.len() + arg.w_eval.len()) * fb;
      let col_b: usize = arg.columns.iter().map(|e| e.len() * fb).sum();
      let auth_b = arg.auth.len() * 32;
      let kb = (rows_b + col_b + auth_b) as f64 / 1024.0;
      println!(
        "n=2^{log_n:<2} open(+commit) {open_ms:7.1}ms  verify {ver_ms:6.2}ms  \
         proof {kb:8.1}KB  (t={} rows={} rowlen={} cols={})",
        params.n_col_opens, params.n_rows, params.row_len, params.n_cols,
      );
    }
  }
}
