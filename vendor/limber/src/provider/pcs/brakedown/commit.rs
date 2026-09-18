//! Brakedown commitment: reshape a polynomial into a row-major matrix, encode
//! each row with the linear-time code, and Merkle-hash the columns of the
//! encoded matrix. The root is the commitment.
//!
//! Layout uses power-of-two row/column message dimensions (so the MLE evaluation
//! point factors as a row⊗column tensor in the eval step). The encoded column
//! count `n_cols = ⌈R·row_len⌉` need not be a power of two — the Merkle tree
//! pads its leaves.

use super::{
  code::{BrakedownCode, CodeSpec, num_column_opens},
  merkle::{Hash, MerkleTree, hash_leaf},
};
use crate::traits::PrimeFieldExt;
use ff::PrimeField;
use rayon::prelude::*;

/// The committing layout for a fixed (power-of-two) polynomial length.
#[derive(Clone, Debug)]
pub struct BrakedownParams<F> {
  /// the linear code applied to each matrix row (message length `row_len`)
  pub code: BrakedownCode<F>,
  /// number of matrix rows (power of two)
  pub n_rows: usize,
  /// message symbols per row (power of two)
  pub row_len: usize,
  /// encoded row length = `code.codeword_len()`
  pub n_cols: usize,
  /// number of columns the IOPP opens (capped at `n_cols`)
  pub n_col_opens: usize,
}

impl<F: PrimeFieldExt> BrakedownParams<F> {
  /// Choose a layout for length-`n` polynomials (`n` a power of two) at
  /// `lambda`-bit security, sampling the code matrices from `seed`.
  ///
  /// Rows are chosen near the proof-size optimum `n_rows ≈ √(R·n / t)` (rounded
  /// to a power of two), so the codeword has `≫ t` columns when `n` is large.
  pub fn new(n: usize, spec: CodeSpec, lambda: usize, seed: &[u8]) -> Self {
    assert!(n.is_power_of_two(), "poly length must be a power of two");
    let log_n = n.trailing_zeros() as usize;
    let t = num_column_opens(&spec, lambda);
    let target_rows = (spec.r * n as f64 / t.max(1) as f64).sqrt();
    let log_rows = (target_rows.log2().round() as i64).clamp(0, log_n as i64) as usize;
    Self::new_with_row_len(n, spec, lambda, seed, n >> log_rows)
  }

  /// Choose a layout with an explicit `row_len` (power of two dividing
  /// `n`). Layouts sharing `(row_len, spec, seed)` share the code — the
  /// prerequisite for combining opening rows across commitments of
  /// different lengths.
  pub fn new_with_row_len(
    n: usize,
    spec: CodeSpec,
    lambda: usize,
    seed: &[u8],
    row_len: usize,
  ) -> Self {
    assert!(n.is_power_of_two(), "poly length must be a power of two");
    assert!(
      row_len.is_power_of_two() && row_len <= n,
      "row length must be a power of two dividing the poly length"
    );
    let t = num_column_opens(&spec, lambda);
    let code = BrakedownCode::new(row_len, spec, seed);
    let n_cols = code.codeword_len();
    Self {
      code,
      n_rows: n / row_len,
      row_len,
      n_cols,
      n_col_opens: t.min(n_cols),
    }
  }

  /// Polynomial length this layout commits.
  pub fn poly_len(&self) -> usize {
    self.n_rows * self.row_len
  }
}

/// Prover-retained data: the encoded matrix (row-major, `n_rows × n_cols`) and
/// the column Merkle tree. Needed to answer openings.
#[derive(Clone, Debug)]
pub struct BrakedownCommitData<F> {
  /// `n_rows` encoded rows, each of length `n_cols`
  pub encoded: Vec<Vec<F>>,
  /// Merkle tree over the `n_cols` columns
  pub tree: MerkleTree,
}

/// Serialize a column (the field elements at one column index) to bytes. Shared
/// by `commit` (building leaves) and the verifier (recomputing a leaf).
pub(crate) fn column_to_bytes<F: PrimeField>(col: &[F]) -> Vec<u8> {
  // Canonical length-prefixed encoding: each entry is `len (1 byte) ||
  // minimal little-endian bytes` (trailing zeros dropped). Injective on
  // sequences, so the Merkle binding is unaffected; hashed bytes drop
  // ~2x on committed chunk data (systematic entries are <= 16-bit,
  // zeros are 1 byte) while full-width parity entries pay +1 byte.
  let mut buf = Vec::with_capacity(col.len() * 8);
  for x in col {
    let repr = x.to_repr();
    let bytes = repr.as_ref();
    let len = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    buf.push(len as u8);
    buf.extend_from_slice(&bytes[..len]);
  }
  buf
}

/// Commit to a small polynomial as a plain hash of its canonical byte
/// encoding — no code, no Merkle tree. Used for polynomials below the
/// backend's direct-ship threshold, whose openings ship the polynomial
/// itself: the verifier re-hashes the shipped bytes and evaluates the
/// claim directly, so an encoded matrix would never be used. Binding
/// follows from collision resistance of the hash on the injective
/// encoding; the scheme is not hiding (as documented for this backend).
pub fn commit_plain<F: PrimeFieldExt>(poly: &[F]) -> (Hash, BrakedownCommitData<F>) {
  let root = hash_leaf(&column_to_bytes(poly));
  (
    root,
    BrakedownCommitData {
      encoded: Vec::new(),
      tree: MerkleTree::from_leaves(vec![root]),
    },
  )
}

/// Commit to `poly` (length `params.poly_len()`): returns the Merkle root and
/// the prover-retained commit data.
pub fn commit<F: PrimeFieldExt>(
  params: &BrakedownParams<F>,
  poly: &[F],
) -> (Hash, BrakedownCommitData<F>) {
  assert_eq!(poly.len(), params.poly_len(), "poly length mismatch");
  let rl = params.row_len;
  let split = std::env::var_os("BDSPLIT").is_some();
  let t0 = std::time::Instant::now();
  // Encode each row in parallel; rows tile the poly with no padding (pow2 dims).
  let encoded: Vec<Vec<F>> = (0..params.n_rows)
    .into_par_iter()
    .map(|i| params.code.encode(&poly[i * rl..(i + 1) * rl]))
    .collect();
  let t_enc = t0.elapsed();
  let t1 = std::time::Instant::now();
  // Hash each encoded column into a leaf, in parallel.
  let leaves: Vec<Hash> = (0..params.n_cols)
    .into_par_iter()
    .map(|c| {
      let col: Vec<F> = encoded.iter().map(|row| row[c]).collect();
      hash_leaf(&column_to_bytes(&col))
    })
    .collect();
  let t_hash = t1.elapsed();
  let t2 = std::time::Instant::now();
  let tree = MerkleTree::from_leaves(leaves);
  if split {
    eprintln!(
      "BDSPLIT n_rows={} row_len={} n_cols={} (sys={}): encode={:.1}ms hash={:.1}ms tree={:.1}ms",
      params.n_rows,
      rl,
      params.n_cols,
      rl,
      t_enc.as_secs_f64() * 1e3,
      t_hash.as_secs_f64() * 1e3,
      t2.elapsed().as_secs_f64() * 1e3,
    );
  }
  (tree.root(), BrakedownCommitData { encoded, tree })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{T256HyraxEngine, pcs::brakedown::code::DEFAULT_SPEC},
    traits::Engine,
  };
  use ff::Field;

  type F = <T256HyraxEngine as Engine>::Scalar;

  fn rand_poly(n: usize, tag: u64) -> Vec<F> {
    // deterministic pseudo-random field elements
    use sha3::{
      Shake256,
      digest::{ExtendableOutput, Update, XofReader},
    };
    let mut h = Shake256::default();
    h.update(b"brakedown-commit-test");
    h.update(&tag.to_le_bytes());
    let mut r = h.finalize_xof();
    (0..n)
      .map(|_| {
        let mut b = [0u8; 64];
        r.read(&mut b);
        F::from_uniform(&b)
      })
      .collect()
  }

  #[test]
  fn layout_is_power_of_two_and_covers_poly() {
    for log_n in [8usize, 12, 16] {
      let n = 1 << log_n;
      let p = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
      assert!(p.n_rows.is_power_of_two());
      assert!(p.row_len.is_power_of_two());
      assert_eq!(p.poly_len(), n);
      assert!(p.n_col_opens <= p.n_cols);
      assert_eq!(p.n_cols, p.code.codeword_len());
    }
  }

  #[test]
  fn commit_is_deterministic_and_binding() {
    let n = 1 << 14;
    let p = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
    let poly = rand_poly(n, 1);

    let (root_a, data) = commit(&p, &poly);
    let (root_b, _) = commit(&p, &poly);
    assert_eq!(root_a, root_b, "commit must be deterministic");
    assert_eq!(data.encoded.len(), p.n_rows);
    assert!(data.encoded.iter().all(|r| r.len() == p.n_cols));

    // systematic: encoded row begins with the original row
    for i in 0..p.n_rows {
      assert_eq!(
        &data.encoded[i][..p.row_len],
        &poly[i * p.row_len..(i + 1) * p.row_len]
      );
    }

    // changing one coefficient changes the root
    let mut poly2 = poly.clone();
    poly2[0] += F::ONE;
    let (root_c, _) = commit(&p, &poly2);
    assert_ne!(root_a, root_c, "commit must bind to the polynomial");
  }

  /// Commit-cost A/B: Brakedown (encode + Merkle hash) vs Hyrax (Pedersen MSM,
  /// `is_small` fast path), back-to-back, same machine. Run with:
  ///   RAYON_NUM_THREADS=1 cargo test --lib -- --ignored --nocapture commit_bench
  /// and again without the env var for the multi-thread picture.
  #[test]
  #[ignore = "benchmark; run with --ignored --nocapture"]
  fn commit_bench() {
    use crate::traits::pcs::PCSEngineTrait;
    use sha3::{
      Shake256,
      digest::{ExtendableOutput, Update, XofReader},
    };
    use std::time::Instant;
    type Hyrax = crate::provider::pcs::hyrax_pc::HyraxPCS<T256HyraxEngine>;

    // small (< 2^64) values so Hyrax's is_small fast MSM path is valid
    fn small_poly(n: usize) -> Vec<F> {
      let mut h = Shake256::default();
      h.update(b"commit-bench");
      let mut r = h.finalize_xof();
      (0..n)
        .map(|_| {
          let mut b = [0u8; 8];
          r.read(&mut b);
          F::from(u64::from_le_bytes(b))
        })
        .collect()
    }

    let threads = rayon::current_num_threads();
    println!("\n== commit A/B (threads={threads}) ==");
    for log_n in [12usize, 14, 16, 18] {
      let n = 1usize << log_n;
      let poly = small_poly(n);
      let iters = 5;

      let bp = BrakedownParams::<F>::new(n, DEFAULT_SPEC, 128, b"seed");
      let _ = commit(&bp, &poly); // warmup
      let t0 = Instant::now();
      for _ in 0..iters {
        let _ = commit(&bp, &poly);
      }
      let bd_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

      let width = 1usize << log_n.div_ceil(2);
      let (ck, _vk) = Hyrax::setup(b"hbench", n, width);
      Hyrax::precompute_ck(&ck);
      let blind = Hyrax::blind(&ck, n);
      let _ = Hyrax::commit(&ck, &poly, &blind, true).unwrap(); // warmup
      let t1 = Instant::now();
      for _ in 0..iters {
        let _ = Hyrax::commit(&ck, &poly, &blind, true).unwrap();
      }
      let hx_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

      println!(
        "n=2^{log_n:<2}  brakedown {bd_ms:7.2}ms  hyrax {hx_ms:7.2}ms  \
         (bd rows={} cols={})  bd/hyrax={:.2}x",
        bp.n_rows,
        bp.n_cols,
        bd_ms / hx_ms,
      );
    }
  }
}
