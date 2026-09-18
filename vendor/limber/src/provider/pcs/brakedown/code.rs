//! Brakedown's linear-time Spielman/expander code (GLSTW21).
//!
//! Systematic recursive code `Enc(x) = [ x | Enc(A·x) | B·Enc(A·x) ]` of length
//! `⌈R·n⌉`, with sparse random matrices `A` (precode) and `B` (postcode) sampled
//! deterministically from a public seed. Parameters and density formulas are
//! ported from the Brakedown authors' reference impl `conroi/lcpc`
//! (`lcpc-brakedown-pc`), per the paper (eprint 2021/1043).
//!
//! Soundness of the PCS built on this code rests on the code's relative minimum
//! distance `δ = β/R`; the IOPP opens `num_column_opens(spec, λ)` columns.

use crate::traits::PrimeFieldExt;
use sha3::{
  Shake256,
  digest::{ExtendableOutput, Update, XofReader},
};

/// A code specification `(α, β, R)`. Relative minimum distance is `δ = β/R`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CodeSpec {
  /// recursive (intermediate) message-length ratio: `m = ⌈α·n⌉`
  pub alpha: f64,
  /// distance parameter
  pub beta: f64,
  /// codeword expansion / rate: codeword length is `⌈R·n⌉`
  pub r: f64,
}

/// The six GLSTW parameter sets (lcpc `SdigCode1..6`), increasing distance.
pub const SPECS: [CodeSpec; 6] = [
  CodeSpec {
    alpha: 0.1195,
    beta: 0.0284,
    r: 1.42,
  },
  CodeSpec {
    alpha: 0.138,
    beta: 0.0444,
    r: 1.47,
  },
  CodeSpec {
    alpha: 0.178,
    beta: 0.061,
    r: 1.521,
  },
  CodeSpec {
    alpha: 0.2,
    beta: 0.082,
    r: 1.64,
  },
  CodeSpec {
    alpha: 0.211,
    beta: 0.097,
    r: 1.616,
  },
  CodeSpec {
    alpha: 0.238,
    beta: 0.1205,
    r: 1.72,
  },
];

/// Default spec: largest distance (`δ≈0.070`) ⇒ fewest column opens.
pub const DEFAULT_SPEC: CodeSpec = SPECS[5];

/// Recursion base case: lengths `≤ BASELEN` use a dense systematic base code.
pub const BASELEN: usize = 20;

impl CodeSpec {
  /// Relative minimum distance `δ = β/R`.
  pub fn distance(&self) -> f64 {
    self.beta / self.r
  }
}

/// Number of columns the IOPP must open for `lambda`-bit security:
/// `⌈ -λ / log2(1 - δ/3) ⌉` (lcpc `_n_col_opens`; the `δ/3` is the proximity
/// test's effective distance).
pub fn num_column_opens(spec: &CodeSpec, lambda: usize) -> usize {
  let d3 = spec.distance() / 3.0;
  (-(lambda as f64) / (1.0 - d3).log2()).ceil() as usize
}

#[inline]
fn ceil_mul(a: usize, ratio: f64) -> usize {
  (a as f64 * ratio).ceil() as usize
}

/// Binary entropy `H(x) = -x log2 x - (1-x) log2(1-x)`, with `H(0)=H(1)=0`.
fn ent(x: f64) -> f64 {
  if x <= 0.0 || x >= 1.0 {
    0.0
  } else {
    -x * x.log2() - (1.0 - x) * (1.0 - x).log2()
  }
}

/// Precode row density `cn` for input length `ni` (lcpc `matgen.rs`).
fn cn(ni: usize, s: &CodeSpec) -> usize {
  let mi = ceil_mul(ni, s.alpha);
  let a = ((ni as f64) * 32.0 * s.beta / 25.0).ceil() as usize;
  let b = 4 + ((ni as f64) * s.beta).ceil() as usize;
  let cnst1 = ent(s.beta) + s.alpha * ent(1.28 * s.beta / s.alpha);
  let cnst2 = s.beta * (s.alpha / (1.28 * s.beta)).log2();
  let c = ((110.0 / (ni as f64) + cnst1) / cnst2).ceil() as usize;
  a.max(b).min(c).min(mi).max(1)
}

/// Postcode row density `dn` for input length `ni`, field bit-size `log2p`.
fn dn(ni: usize, s: &CodeSpec, log2p: f64) -> usize {
  let mi = ceil_mul(ni, s.alpha);
  let niprime = ceil_mul(mi, s.r);
  let miprime = ceil_mul(ni, s.r).saturating_sub(ni).saturating_sub(niprime);
  let tmp1 = ((ni as f64) * 2.0 * s.beta).ceil() as usize;
  let tmp2 = (ceil_mul(ni, s.r) as f64 - ni as f64 + 110.0).max(0.0);
  let mu = s.r - 1.0 - s.r * s.alpha;
  let nu = s.beta + s.alpha * s.beta + 0.03;
  let cnst1 = s.r * s.alpha * ent(s.beta / s.r) + mu * ent(nu / mu);
  let cnst2 = s.alpha * s.beta * (mu / nu).log2();
  let d = (tmp1 + (tmp2 / log2p).ceil() as usize)
    .min(((110.0 / (ni as f64) + cnst1) / cnst2).ceil() as usize);
  d.min(miprime).max(1)
}

/// A sparse matrix with `rows.len()` rows over `cols` columns; row `i` holds a
/// list of `(column, value)` nonzeros. Computes `y = x·M` (`|x| = rows`,
/// `|y| = cols`).
#[derive(Clone, Debug)]
struct SparseMat<F> {
  cols: usize,
  rows: Vec<Vec<(usize, F)>>,
}

impl<F: PrimeFieldExt> SparseMat<F> {
  /// Sample `n_rows × n_cols` with `density` distinct random nonzeros per row,
  /// drawn deterministically from `reader`.
  fn sample(reader: &mut impl XofReader, n_rows: usize, n_cols: usize, density: usize) -> Self {
    let d = density.min(n_cols);
    let rows = (0..n_rows)
      .map(|_| {
        let mut entries: Vec<(usize, F)> = Vec::with_capacity(d);
        while entries.len() < d {
          let c = next_index(reader, n_cols);
          if entries.iter().all(|(cc, _)| *cc != c) {
            entries.push((c, next_nonzero(reader)));
          }
        }
        entries
      })
      .collect();
    Self { cols: n_cols, rows }
  }

  fn mul_vec(&self, x: &[F]) -> Vec<F> {
    debug_assert_eq!(x.len(), self.rows.len());
    // Delayed reduction: scatter UNREDUCED products into wide per-output
    // accumulators and pay one Montgomery reduction per output instead
    // of one per term (~density/rate ≈ 6 terms land on each output, so
    // the per-term modular reduction dominates the plain version).
    let mut acc =
      vec![<F as crate::traits::DelayedReduction<F>>::Accumulator::default(); self.cols];
    for (i, row) in self.rows.iter().enumerate() {
      let xi = x[i];
      // Input-sparsity guard: zero inputs contribute nothing, and the
      // committed data (limb/chunk polynomials of real witnesses) is
      // mostly structural zeros — the check is ~10x cheaper than the
      // multiply it skips.
      if xi.is_zero_vartime() {
        continue;
      }
      for (c, v) in row {
        F::unreduced_multiply_accumulate(&mut acc[*c], &xi, v);
      }
    }
    acc.iter().map(F::reduce).collect()
  }
}

/// Deterministic XOF stream seeded by `seed || tag` (Shake256). Also used by the
/// IOPP to expand transcript challenges into field elements / column indices.
pub(crate) fn xof(seed: &[u8], tag: &[u8]) -> impl XofReader + use<> {
  let mut h = Shake256::default();
  h.update(seed);
  h.update(tag);
  h.finalize_xof()
}

/// Next field element from a XOF stream (64 uniform bytes → field).
pub(crate) fn next_scalar<F: PrimeFieldExt>(reader: &mut impl XofReader) -> F {
  let mut b = [0u8; 64];
  reader.read(&mut b);
  F::from_uniform(&b)
}

fn next_nonzero<F: PrimeFieldExt>(reader: &mut impl XofReader) -> F {
  loop {
    let s = next_scalar::<F>(reader);
    if s != F::ZERO {
      return s;
    }
  }
}

/// Next index in `[0, bound)` from a XOF stream.
pub(crate) fn next_index(reader: &mut impl XofReader, bound: usize) -> usize {
  let mut b = [0u8; 8];
  reader.read(&mut b);
  (u64::from_le_bytes(b) % bound as u64) as usize
}

/// One recursion level: precode `A` (`ni → mi`) and postcode `B`
/// (`niprime → miprime`).
#[derive(Clone, Debug)]
struct Level<F> {
  a: SparseMat<F>,
  b: SparseMat<F>,
}

/// A fully-instantiated Brakedown code for a fixed message length.
#[derive(Clone, Debug)]
pub struct BrakedownCode<F> {
  spec: CodeSpec,
  msg_len: usize,
  codeword_len: usize,
  levels: Vec<Level<F>>,
  /// Dense base code: `base_in × base_parity` generator for the smallest level.
  base: SparseMat<F>,
  base_in: usize,
}

impl<F: PrimeFieldExt> BrakedownCode<F> {
  /// Instantiate the code for messages of length `msg_len`, sampling all
  /// matrices deterministically from `seed`.
  pub fn new(msg_len: usize, spec: CodeSpec, seed: &[u8]) -> Self {
    assert!(msg_len > 0, "message length must be positive");
    let log2p = F::CAPACITY as f64; // field bit-size for dn (lower bound on log2|F|)
    let mut levels = Vec::new();
    let mut ni = msg_len;
    let mut lvl = 0usize;
    while ni > BASELEN {
      let mi = ceil_mul(ni, spec.alpha);
      let niprime = ceil_mul(mi, spec.r);
      let miprime = ceil_mul(ni, spec.r)
        .saturating_sub(ni)
        .saturating_sub(niprime);
      let cn = cn(ni, &spec);
      let dn = dn(ni, &spec, log2p);
      let mut ra = xof(
        seed,
        &[b"A".as_slice(), &(lvl as u64).to_le_bytes()].concat(),
      );
      let mut rb = xof(
        seed,
        &[b"B".as_slice(), &(lvl as u64).to_le_bytes()].concat(),
      );
      let a = SparseMat::sample(&mut ra, ni, mi, cn);
      let b = SparseMat::sample(&mut rb, niprime, miprime, dn);
      levels.push(Level { a, b });
      ni = mi;
      lvl += 1;
    }
    // Base: dense systematic code, base_in -> ⌈R·base_in⌉ (parity = base_in × parity).
    let base_in = ni;
    let base_parity = ceil_mul(base_in, spec.r).saturating_sub(base_in).max(1);
    let mut rbase = xof(seed, b"base");
    let base = SparseMat::sample(&mut rbase, base_in, base_parity, base_parity);
    let codeword_len = ceil_mul(msg_len, spec.r);
    Self {
      spec,
      msg_len,
      codeword_len,
      levels,
      base,
      base_in,
    }
  }

  /// Message (input) length this code was instantiated for.
  pub fn message_len(&self) -> usize {
    self.msg_len
  }

  /// Codeword (output) length, `⌈R·message_len⌉`.
  pub fn codeword_len(&self) -> usize {
    self.codeword_len
  }

  /// The code spec `(α, β, R)` in use.
  pub fn spec(&self) -> &CodeSpec {
    &self.spec
  }

  /// Encode a message of length `message_len()` into a codeword of length
  /// `codeword_len()`. Systematic: `out[..message_len()] == msg`.
  pub fn encode(&self, msg: &[F]) -> Vec<F> {
    assert_eq!(msg.len(), self.msg_len, "message length mismatch");
    self.encode_level(0, msg)
  }

  fn encode_level(&self, level: usize, x: &[F]) -> Vec<F> {
    if level == self.levels.len() {
      debug_assert_eq!(x.len(), self.base_in);
      let parity = self.base.mul_vec(x);
      let mut out = Vec::with_capacity(x.len() + parity.len());
      out.extend_from_slice(x);
      out.extend(parity);
      return out;
    }
    let lvl = &self.levels[level];
    let z = lvl.a.mul_vec(x); // length mi
    let z_enc = self.encode_level(level + 1, &z); // length niprime
    let v = lvl.b.mul_vec(&z_enc); // length miprime
    let mut out = Vec::with_capacity(x.len() + z_enc.len() + v.len());
    out.extend_from_slice(x);
    out.extend(z_enc);
    out.extend(v);
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{provider::T256HyraxEngine, traits::Engine};
  use ff::Field;

  type F = <T256HyraxEngine as Engine>::Scalar;

  fn rand_vec(n: usize, tag: &[u8]) -> Vec<F> {
    let mut r = xof(b"test-msg", tag);
    (0..n).map(|_| next_scalar::<F>(&mut r)).collect()
  }

  #[test]
  fn encode_is_systematic_and_right_length() {
    for &n in &[1usize, 5, 20, 21, 64, 100, 1000] {
      let code = BrakedownCode::<F>::new(n, DEFAULT_SPEC, b"seed");
      let x = rand_vec(n, b"x");
      let cw = code.encode(&x);
      assert_eq!(cw.len(), code.codeword_len());
      assert!(cw.len() >= ceil_mul(n, DEFAULT_SPEC.r));
      assert_eq!(&cw[..n], &x[..], "code must be systematic (n={n})");
    }
  }

  #[test]
  fn encode_is_linear() {
    let n = 300;
    let code = BrakedownCode::<F>::new(n, DEFAULT_SPEC, b"seed");
    let x = rand_vec(n, b"x");
    let y = rand_vec(n, b"y");
    let xy: Vec<F> = x.iter().zip(&y).map(|(a, b)| *a + *b).collect();
    let ex = code.encode(&x);
    let ey = code.encode(&y);
    let exy = code.encode(&xy);
    for i in 0..exy.len() {
      assert_eq!(exy[i], ex[i] + ey[i], "encode must be F-linear at {i}");
    }
  }

  #[test]
  fn distinct_messages_have_far_codewords() {
    // Distance sanity: two messages differing in one symbol should yield
    // codewords differing in many positions (≥ a small fraction of δ·len).
    let n = 1000;
    let code = BrakedownCode::<F>::new(n, DEFAULT_SPEC, b"seed");
    let mut x = rand_vec(n, b"x");
    let cw0 = code.encode(&x);
    x[0] += F::ONE;
    let cw1 = code.encode(&x);
    let differ = cw0.iter().zip(&cw1).filter(|(a, b)| a != b).count();
    let floor = ((code.codeword_len() as f64) * DEFAULT_SPEC.distance() * 0.5) as usize;
    assert!(
      differ >= floor.max(1),
      "expected ≥{floor} differing positions, got {differ}"
    );
  }

  #[test]
  fn column_opens_is_sane() {
    let t = num_column_opens(&DEFAULT_SPEC, 128);
    assert!(t > 0 && t < 100_000, "column opens out of range: {t}");
  }
}
