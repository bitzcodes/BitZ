//! BaseFold's random **foldable code** — a recursive butterfly with random
//! diagonals. Unlike Reed–Solomon it needs no smooth / FFT-friendly field
//! (it has `O(n log n)` encoding over *any* sufficiently large field), so it
//! works over our sampled ~256-bit commitment field where WHIR/FRI cannot.
//!
//! Construction (inverse-rate `c`, message length `2^log_k`):
//! - base (`level 0`): `a ↦ (a·g_0, …, a·g_{c-1})`
//! - `level i`: split `msg = (m_l, m_r)`; `l = enc(m_l)`, `r = enc(m_r)`;
//!   emit `(l + t_i⊙r,  l − t_i⊙r)`
//!
//! Folding a `level i` codeword `(w0, w1)` with challenge `α` recovers
//! `l = (w0+w1)/2`, `r = (w0−w1)/(2 t_i)` and returns `l + α·r` — a
//! `level (i−1)` codeword of `enc(m_l + α·m_r)`. Folding all `log_k` levels
//! with a point's coordinates therefore evaluates the multilinear extension
//! of `msg` at that point; that is the correspondence the BaseFold
//! prover/verifier ride on.

use crate::traits::PrimeFieldExt;
use ff::Field;

use super::super::brakedown::code::{next_scalar, xof};
use sha3::digest::XofReader;

#[inline]
fn inv<F: Field>(x: F) -> F {
  Option::<F>::from(x.invert()).expect("inverse of a nonzero code element")
}

/// Montgomery batch inversion: invert all of `xs` with ONE field inversion
/// plus `O(n)` multiplications (vs `n` inversions). All entries must be
/// nonzero.
fn batch_invert<F: Field>(xs: &[F]) -> Vec<F> {
  let n = xs.len();
  let mut prefix = vec![F::ONE; n];
  let mut acc = F::ONE;
  for (i, &x) in xs.iter().enumerate() {
    prefix[i] = acc;
    acc *= x;
  }
  let mut inv_acc = inv(acc);
  let mut out = vec![F::ZERO; n];
  for i in (0..n).rev() {
    out[i] = inv_acc * prefix[i];
    inv_acc *= xs[i];
  }
  out
}

/// Next nonzero field element from the deterministic stream.
fn nz<F: PrimeFieldExt>(r: &mut impl XofReader) -> F {
  loop {
    let s = next_scalar::<F>(r);
    if s != F::ZERO {
      return s;
    }
  }
}

/// A random foldable code for a fixed message length `2^log_k` and
/// inverse rate `c` (codeword length `c · 2^log_k`).
#[derive(Clone, Debug)]
pub struct FoldableCode<F> {
  log_k: usize,
  inv_rate: usize,
  /// Base generator, length `c`, all nonzero.
  base_gen: Vec<F>,
  /// `diag[i]` is the diagonal for level `i+1`, length `c · 2^i`, all nonzero.
  diag: Vec<Vec<F>>,
  /// Precomputed `diag[i][j]⁻¹` for the fold hot path.
  diag_inv: Vec<Vec<F>>,
  /// Precomputed `2⁻¹`.
  inv2: F,
}

impl<F: PrimeFieldExt> FoldableCode<F> {
  /// Deterministically sample a foldable code from `seed`. Sub-codes at
  /// `log_k' < log_k` sampled from the SAME seed share `base_gen` and the
  /// first `log_k'` diagonals (sampling order: base first, then diagonals
  /// low→high), so a `log_k−1` code is exactly the fold target of a
  /// `log_k` code.
  pub fn new(log_k: usize, inv_rate: usize, seed: &[u8]) -> Self {
    let c = inv_rate;
    let mut r = xof(seed, b"basefold-foldable-code-v1");
    let base_gen: Vec<F> = (0..c).map(|_| nz::<F>(&mut r)).collect();
    let diag: Vec<Vec<F>> = (0..log_k)
      .map(|i| (0..c << i).map(|_| nz::<F>(&mut r)).collect())
      .collect();
    // Batch-invert all diagonals at once (was O(n) inversions — the actual
    // hotspot; a length-2^21 code inverts ~2^21 elements per build).
    let lens: Vec<usize> = diag.iter().map(|r| r.len()).collect();
    let flat: Vec<F> = diag.iter().flatten().copied().collect();
    let flat_inv = batch_invert(&flat);
    let mut diag_inv: Vec<Vec<F>> = Vec::with_capacity(diag.len());
    let mut off = 0;
    for len in lens {
      diag_inv.push(flat_inv[off..off + len].to_vec());
      off += len;
    }
    let inv2 = inv(F::ONE + F::ONE);
    Self {
      log_k,
      inv_rate: c,
      base_gen,
      diag,
      diag_inv,
      inv2,
    }
  }

  /// `log₂` of the message length.
  pub fn log_k(&self) -> usize {
    self.log_k
  }
  /// Message length `2^log_k`.
  pub fn msg_len(&self) -> usize {
    1usize << self.log_k
  }
  /// Codeword length `c · 2^log_k`.
  pub fn codeword_len(&self) -> usize {
    self.inv_rate << self.log_k
  }

  /// Encode a length-`2^log_k` message to a length-`c·2^log_k` codeword.
  pub fn encode(&self, msg: &[F]) -> Vec<F> {
    assert_eq!(msg.len(), self.msg_len(), "message length");
    self.enc(msg, self.log_k)
  }

  fn enc(&self, msg: &[F], level: usize) -> Vec<F> {
    if level == 0 {
      let a = msg[0];
      return self.base_gen.iter().map(|g| a * *g).collect();
    }
    let half = 1usize << (level - 1);
    let l = self.enc(&msg[..half], level - 1);
    let r = self.enc(&msg[half..], level - 1);
    let t = &self.diag[level - 1];
    let n = l.len();
    debug_assert_eq!(n, t.len());
    let mut out = vec![F::ZERO; 2 * n];
    for j in 0..n {
      let tr = t[j] * r[j];
      out[j] = l[j] + tr;
      out[n + j] = l[j] - tr;
    }
    out
  }

  /// Fold a single index pair `(w_lo, w_hi)` — the entries at `j` and
  /// `j+half` of a `level`-codeword — with `alpha`, giving the child entry
  /// at index `j` of the `level-1` codeword. Verifier-side counterpart of
  /// [`Self::fold`].
  pub fn fold_pair(&self, w_lo: F, w_hi: F, alpha: F, level: usize, j: usize) -> F {
    let l = (w_lo + w_hi) * self.inv2;
    let r = (w_lo - w_hi) * self.inv2 * self.diag_inv[level - 1][j];
    (F::ONE - alpha) * l + alpha * r
  }

  /// If `base` is a valid base codeword `v · base_gen`, return `v`; else
  /// `None`. (`base_gen[0] != 0` by construction.)
  pub fn base_value(&self, base: &[F]) -> Option<F> {
    if base.len() != self.inv_rate {
      return None;
    }
    let v = base[0] * inv(self.base_gen[0]);
    for (b, g) in base.iter().zip(&self.base_gen) {
      if *b != v * *g {
        return None;
      }
    }
    Some(v)
  }

  /// Fold a `level`-codeword with challenge `alpha` into a `level-1`
  /// codeword. `level` must be in `1..=log_k`.
  pub fn fold(&self, w: &[F], alpha: F, level: usize) -> Vec<F> {
    assert!((1..=self.log_k).contains(&level), "fold level in range");
    let n = w.len() / 2;
    assert_eq!(n, self.diag[level - 1].len(), "codeword length for level");
    let (w0, w1) = w.split_at(n);
    let tinv = &self.diag_inv[level - 1];
    (0..n)
      .map(|j| {
        let l = (w0[j] + w1[j]) * self.inv2;
        // r = (w0 − w1) / (2 t) = (w0 − w1) · inv2 · t⁻¹
        let r = (w0[j] - w1[j]) * self.inv2 * tinv[j];
        // Evaluation-basis fold (fixing a multilinear variable to α):
        (F::ONE - alpha) * l + alpha * r
      })
      .collect()
  }
}

/// Test-only deterministic random vectors, shared across BaseFold modules.
#[cfg(test)]
pub(crate) fn rand_vec<F: PrimeFieldExt>(n: usize, seed: u64) -> Vec<F> {
  let mut r = xof(&seed.to_le_bytes(), b"test-msg");
  (0..n).map(|_| next_scalar::<F>(&mut r)).collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pt256::t256;

  type F = t256::Scalar;

  #[test]
  #[ignore = "measurement"]
  fn commit_breakdown() {
    use super::super::query::commit_layer;
    let log_k = 18usize;
    let code = FoldableCode::<F>::new(log_k, 8, b"m");
    let msg: Vec<F> = rand_vec(1 << log_k, 1);
    let t = std::time::Instant::now();
    let cw = code.encode(&msg);
    let enc_ms = t.elapsed().as_secs_f64() * 1e3;
    let n = cw.len();
    let t2 = std::time::Instant::now();
    let _c0 = commit_layer(cw);
    let merkle_ms = t2.elapsed().as_secs_f64() * 1e3;
    println!("2^{log_k} rate1/8 codeword={n}: encode {enc_ms:.0} ms, merkle {merkle_ms:.0} ms");
  }

  #[test]
  #[ignore = "measurement"]
  fn encode_cost() {
    for log_k in [16usize, 18] {
      for rate in [2usize, 4, 8] {
        let code = FoldableCode::<F>::new(log_k, rate, b"m");
        let msg: Vec<F> = rand_vec(1 << log_k, 1);
        // warm
        let _ = code.encode(&msg);
        let t = std::time::Instant::now();
        let cw = code.encode(&msg);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
          "encode log_k={log_k} rate=1/{rate}: codeword={} ({:.0} Mmul est) -> {ms:.1} ms",
          cw.len(),
          (log_k as f64) * (cw.len() as f64) / 2.0 / 1e6
        );
      }
    }
  }

  /// Defining invariant: folding an encoded message equals encoding the
  /// folded message (`fold∘encode == encode∘fold_msg`). This is the
  /// correctness property of the whole construction.
  #[test]
  fn foldability_one_level() {
    for log_k in 1..=6 {
      let c = 4;
      let code = FoldableCode::<F>::new(log_k, c, b"seed-A");
      let sub = FoldableCode::<F>::new(log_k - 1, c, b"seed-A"); // matching fold target
      let msg = rand_vec(1 << log_k, 7);
      let alpha = rand_vec(1, 99)[0];

      let lhs = code.fold(&code.encode(&msg), alpha, log_k);

      let half = 1usize << (log_k - 1);
      let msg_folded: Vec<F> = (0..half)
        .map(|j| (F::ONE - alpha) * msg[j] + alpha * msg[half + j])
        .collect();
      let rhs = sub.encode(&msg_folded);

      assert_eq!(lhs, rhs, "foldability at log_k={log_k}");
    }
  }

  /// Folding all `log_k` levels with a point's coordinates yields the base
  /// encoding of the multilinear extension of `msg` evaluated at that point.
  /// Cross-checks the codeword fold against an independent MLE evaluation.
  #[test]
  fn full_fold_equals_mle_eval() {
    let log_k = 5;
    let c = 4;
    let code = FoldableCode::<F>::new(log_k, c, b"seed-B");
    let msg = rand_vec(1 << log_k, 11);
    // Challenges: point[0] folds the top level (MSB), … point[log_k-1] the base.
    let point = rand_vec(log_k, 22);

    // Fold the codeword down to the base (length c).
    let mut w = code.encode(&msg);
    for (i, &alpha) in point.iter().enumerate() {
      w = code.fold(&w, alpha, log_k - i);
    }
    assert_eq!(w.len(), c);

    // Fold the MESSAGE the same way -> a single scalar (the MLE eval).
    let mut m = msg.clone();
    for &alpha in &point {
      let half = m.len() / 2;
      m = (0..half)
        .map(|j| (F::ONE - alpha) * m[j] + alpha * m[half + j])
        .collect();
    }
    assert_eq!(m.len(), 1);
    let eval = m[0];

    // The base-encoded folded scalar must equal the fully folded codeword.
    let expected: Vec<F> = code.base_gen.iter().map(|g| eval * *g).collect();
    assert_eq!(w, expected, "full fold == base(MLE eval)");
  }
}
