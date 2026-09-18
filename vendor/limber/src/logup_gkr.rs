//! LogUp-GKR range proof.
//!
//! Proves that every value in a witness vector lies in `[0, 2^bits)` using the
//! logarithmic-derivative (LogUp) lookup identity
//!
//! ```text
//!   sum_{i=0}^{N-1} 1/(r + w[i])  =  sum_{j=0}^{2^bits - 1} m_j/(r + j)
//! ```
//!
//! where `m_j` is the multiplicity (number of witnesses equal to `j`). The
//! identity holds for a random challenge `r` iff the multiset `{w[i]}` is
//! contained in the table `[0, 2^bits)` — which is exactly the range claim.
//!
//! Each side of the identity is a sum of rational functions, evaluated by a
//! layered GKR circuit whose gates add fractions:
//!
//! ```text
//!   (p_l, q_l) + (p_r, q_r) = (p_l·q_r + p_r·q_l,  q_l·q_r)
//! ```
//!
//! A balanced binary tree of these gates reduces `2^d` leaf fractions to a
//! single root fraction `(P, Q) = (numerator, denominator)` equal to the whole
//! sum. We run one tree per side and check the cross-product
//! `P_L·Q_R == P_R·Q_L` (i.e. `P_L/Q_L == P_R/Q_R`).
//!
//! The GKR protocol reduces the root claim, layer by layer, to a single
//! evaluation claim about each side's *leaf* multilinear extension at a random
//! point. For the witness side that claim pins `w(ρ_L)`; for the table side it
//! pins `m(ρ_R)`. Those two evaluations are returned as [`RangeClaims`] so the
//! caller can discharge them with PCS openings of the committed witness and
//! multiplicity polynomials. The table-index and all-ones-numerator leaves are
//! structured, so the verifier checks them in closed form.
//!
//! This module is intentionally self-contained (no PCS dependency yet) so the
//! GKR core and the LogUp identity can be tested in isolation before wiring the
//! returned [`RangeClaims`] into a commitment scheme. **Fiat–Shamir note:** a
//! real caller must absorb the witness and multiplicity commitments into the
//! transcript *before* calling [`LogUpRangeProof::prove`]/`verify`, so the
//! challenge `r` is bound to them.

use crate::{
  errors::SpartanError,
  polys::eq::EqPolynomial,
  traits::{
    PrimeFieldExt,
    mod_engine::SumcheckEngine,
    transcript::{ByteTranscript, TranscriptEngineTrait},
  },
};
use ff::{Field, PrimeField};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Below this many index pairs, the hot loops run serially (rayon overhead
/// would dominate).
const PAR_THRESHOLD: usize = 1 << 12;

/// In-place bind of a dense MLE's top variable to `r`: replaces the table with
/// its restriction `Z(r, ·)` (length halves). Matches the big-endian
/// convention of [`crate::polys::multilinear::MultilinearPolynomial`].
#[inline]
fn bind_top<F: PrimeField>(v: &mut Vec<F>, r: F) {
  let h = v.len() / 2;
  if h >= PAR_THRESHOLD {
    let (lo, hi) = v.split_at_mut(h);
    lo.par_iter_mut().zip(hi.par_iter()).for_each(|(a, b)| {
      *a += r * (*b - *a);
    });
  } else {
    for i in 0..h {
      let diff = v[i + h] - v[i];
      v[i] += r * diff;
    }
  }
  v.truncate(h);
}

/// Precomputed constants for [`eval_cubic`] — the two field inversions
/// cost ~2.4 µs per call, and provers/verifiers evaluate thousands of
/// round polynomials per proof; build once per GKR walk.
#[derive(Clone, Copy)]
struct CubicConsts<F> {
  inv2: F,
  inv6: F,
}

impl<F: PrimeField> CubicConsts<F> {
  fn new() -> Self {
    Self {
      inv2: F::from(2).invert().expect("2 invertible"),
      inv6: F::from(6).invert().expect("6 invertible"),
    }
  }
}

/// Evaluate a degree-3 univariate at `r` from its evaluations at `0,1,2,3`
/// via Lagrange interpolation.
fn eval_cubic_with<F: PrimeField>(cc: &CubicConsts<F>, e: &[F; 4], r: F) -> F {
  let r1 = r - F::ONE;
  let r2 = r - F::from(2);
  let r3 = r - F::from(3);
  // Lagrange basis at nodes {0,1,2,3}.
  let l0 = r1 * r2 * r3 * (-cc.inv6);
  let l1 = r * r2 * r3 * cc.inv2;
  let l2 = r * r1 * r3 * (-cc.inv2);
  let l3 = r * r1 * r2 * cc.inv6;
  e[0] * l0 + e[1] * l1 + e[2] * l2 + e[3] * l3
}

/// Closed-form evaluation of the MLE of the table-index function
/// `f(j) = j` over `{0,1}^len`, at `point` (with `point[0]` the most
/// significant variable): `sum_k point[k]·2^(len-1-k)`.
fn idx_mle_eval<F: PrimeField>(point: &[F]) -> F {
  let mut acc = F::ZERO;
  let mut pow = F::ONE;
  for k in (0..point.len()).rev() {
    acc += point[k] * pow;
    pow = pow + pow;
  }
  acc
}

/// One GKR layer's reduction: the cubic-sumcheck round polynomials (each as
/// evaluations at `0,1,2,3`) plus the input layer's four evaluations
/// `(p(0,ρ'), p(1,ρ'), q(0,ρ'), q(1,ρ'))` at the sumcheck point.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct GkrLayerProof<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  round_polys: Vec<[E::Scalar; 4]>,
  p0: E::Scalar,
  p1: E::Scalar,
  q0: E::Scalar,
  q1: E::Scalar,
}

/// A fractional-sum GKR proof: one [`GkrLayerProof`] per layer, top (root) to
/// bottom (leaves).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct GkrProof<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  layers: Vec<GkrLayerProof<E>>,
}

/// Prover output for one fraction tree.
struct GkrOut<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  proof: GkrProof<E>,
  root_p: E::Scalar,
  root_q: E::Scalar,
  leaf_point: Vec<E::Scalar>,
  leaf_p: E::Scalar,
  leaf_q: E::Scalar,
}

/// One lockstep layer of a batched multi-tree GKR walk: the γ-combined
/// round polynomials — a single cubic per round shared by every active
/// tree — plus each active tree's input-layer evaluations
/// `[p0, p1, q0, q1]`, in tree order. The per-tree finals cannot be
/// batched away: they seed the next layer's per-tree claims and are
/// ultimately pinned by the leaf-level PCS openings / closed-form table
/// checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct GkrMultiLayerProof<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  /// When the layer skips its first two rounds (see [`skip_at_layer`]):
  /// the γ-combined bivariate restriction of the first two sumcheck
  /// variables, as 16 evaluations on the `{0,1,2,3}²` grid (row-major,
  /// `grid[t1·4 + t2]`, eq factors included). Replaces the first two
  /// entries of `round_polys`.
  skip: Option<Vec<E::Scalar>>,
  round_polys: Vec<[E::Scalar; 4]>,
  finals: Vec<[E::Scalar; 4]>,
}

/// The shared proof of a batched multi-tree GKR walk: one
/// [`GkrMultiLayerProof`] per layer of the DEEPEST tree, top to bottom.
/// Replaces the per-tree [`GkrProof`]s the lockstep walk used to emit —
/// round-polynomial size drops from `Σ_t Θ(d_t²)` to `Θ(max_d²)` field
/// elements while the per-tree finals stay `4·Σ_t d_t`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct GkrMultiProof<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  layers: Vec<GkrMultiLayerProof<E>>,
}

/// Per-tree prover output of a batched multi-tree walk (the shared
/// transcript artifact lives in [`GkrMultiProof`]).
struct GkrTreeOut<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  root_p: E::Scalar,
  root_q: E::Scalar,
  leaf_point: Vec<E::Scalar>,
  leaf_p: E::Scalar,
  leaf_q: E::Scalar,
}

/// Whether layer `k` of the lockstep walk replaces its first two
/// sumcheck rounds with one bivariate skip round: every active tree
/// must be at its leaf layer (`depth == k+1`), where the tables are
/// still structured input data, and there must be at least two
/// variables to skip. Depths are known to both sides, so the predicate
/// is transcript-deterministic. `GKRSKIP=0` disables (benching knob;
/// prover and verifier share the process).
fn skip_at_layer(k: usize, active_depths: &[usize]) -> bool {
  k >= 2
    && !active_depths.is_empty()
    && active_depths.iter().all(|&d| d == k + 1)
    && std::env::var("GKRSKIP").map_or(true, |v| v != "0")
}

/// Evaluate the bivariate on the `{0,1,2,3}²` grid (row-major) at
/// `(u1, u2)` by two nested Lagrange interpolations.
fn eval_grid16<F: PrimeField>(cc: &CubicConsts<F>, grid: &[F], u1: F, u2: F) -> F {
  debug_assert_eq!(grid.len(), 16);
  let rows: [F; 4] = core::array::from_fn(|i| {
    eval_cubic_with(
      cc,
      &[
        grid[i * 4],
        grid[i * 4 + 1],
        grid[i * 4 + 2],
        grid[i * 4 + 3],
      ],
      u2,
    )
  });
  eval_cubic_with(cc, &rows, u1)
}

/// `eq(i; c) = (1-c)(1-i) + c·i` at the small integer node `i`.
fn eq_at_node<F: PrimeField>(c: F, i: usize) -> F {
  let fi = F::from(i as u64);
  (F::ONE - c) * (F::ONE - fi) + c * fi
}

/// The 2×4 extension coefficients of one boolean variable to the nodes
/// `{0,1,2,3}`: `ext[i] = (1-i, i)` as small signed integers. A table
/// value extends to node `i` as `(1-i)·T(0,·) + i·T(1,·)`.
const EXT2: [[i64; 2]; 4] = [[1, 0], [0, 1], [-1, 2], [-2, 3]];

/// Build all layers of the fraction tree from the leaves up to the root.
/// `levels_p[k]` / `levels_q[k]` hold the `2^k`-entry layer; `[d]` is the
/// leaves and `[0]` the single-entry root.
fn build_levels<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  p_leaves: Vec<E::Scalar>,
  q_leaves: Vec<E::Scalar>,
  ones_numerator: bool,
  hint: LeafHint<'_, E::Scalar>,
) -> (Vec<Vec<E::Scalar>>, Vec<Vec<E::Scalar>>) {
  // With all-ones numerators the leaf `p` table is never read (the leaf
  // combine uses `q` only, and the leaf layer's sumcheck treats the
  // numerator MLE as the constant 1), so callers may pass an empty
  // `p_leaves` to skip the allocation.
  let n = q_leaves.len();
  debug_assert!(p_leaves.len() == n || (ones_numerator && p_leaves.is_empty()));
  let d = n.trailing_zeros() as usize;
  let mut levels_p = vec![Vec::new(); d + 1];
  let mut levels_q = vec![Vec::new(); d + 1];
  levels_p[d] = p_leaves;
  levels_q[d] = q_leaves;
  for k in (0..d).rev() {
    let half = 1usize << k;
    let mut op = vec![E::Scalar::ZERO; half];
    let mut oq = vec![E::Scalar::ZERO; half];
    // Structured leaf combine: with q = offset + v elementwise,
    // q0·q1 = offset² + offset·(v0+v1) + v0·v1 costs two one-fold
    // scaled multiplies instead of a full field multiplication (and the
    // table tree's numerator combine collapses the same way).
    if k + 1 == d
      && let Some(()) = {
        let off_sq_of = |r: E::Scalar| (r * r, r.scale_shift64(), E::Scalar::ONE.scale_shift64());
        match hint {
          LeafHint::OnesAffine { offset, raw } => {
            let (off2, off_s, one_s) = off_sq_of(offset);
            let in_q = &levels_q[d];
            let combine = |i: usize, q: &mut E::Scalar, p: &mut E::Scalar| {
              let (v0, v1) = (raw[i], raw[i + half]);
              *q = off2 + off_s.mul_u64_scaled(v0 + v1) + one_s.mul_u64_scaled(v0 * v1);
              *p = in_q[i] + in_q[i + half];
            };
            if half >= PAR_THRESHOLD {
              op.par_iter_mut()
                .zip(oq.par_iter_mut())
                .enumerate()
                .for_each(|(i, (p, q))| combine(i, q, p));
            } else {
              for i in 0..half {
                let (mut q, mut p) = (E::Scalar::ZERO, E::Scalar::ZERO);
                combine(i, &mut q, &mut p);
                oq[i] = q;
                op[i] = p;
              }
            }
            Some(())
          }
          LeafHint::TableAffine { offset, mult } => {
            let (off2, off_s, one_s) = off_sq_of(offset);
            let combine = |i: usize, q: &mut E::Scalar, p: &mut E::Scalar| {
              let (j0, j1) = (i as u64, (i + half) as u64);
              let (m0, m1) = (mult[i], mult[i + half]);
              *q = off2 + off_s.mul_u64_scaled(j0 + j1) + one_s.mul_u64_scaled(j0 * j1);
              // m0·(offset+j1) + m1·(offset+j0)
              *p = off_s.mul_u64_scaled(m0 + m1) + one_s.mul_u64_scaled(m0 * j1 + m1 * j0);
            };
            if half >= PAR_THRESHOLD {
              op.par_iter_mut()
                .zip(oq.par_iter_mut())
                .enumerate()
                .for_each(|(i, (p, q))| combine(i, q, p));
            } else {
              for i in 0..half {
                let (mut q, mut p) = (E::Scalar::ZERO, E::Scalar::ZERO);
                combine(i, &mut q, &mut p);
                oq[i] = q;
                op[i] = p;
              }
            }
            Some(())
          }
          LeafHint::None => None,
        }
      }
    {
      levels_p[k] = op;
      levels_q[k] = oq;
      continue;
    }
    {
      let in_p = &levels_p[k + 1];
      let in_q = &levels_q[k + 1];
      // Combine the (top=0, i) and (top=1, i) fractions. When the leaf
      // numerators are all 1 (witness trees) the first combine is just
      // p = q0 + q1.
      let leaf_ones = ones_numerator && k + 1 == d;
      if half >= PAR_THRESHOLD {
        op.par_iter_mut()
          .zip(oq.par_iter_mut())
          .enumerate()
          .for_each(|(i, (p, q))| {
            *p = if leaf_ones {
              in_q[i] + in_q[i + half]
            } else {
              in_p[i] * in_q[i + half] + in_p[i + half] * in_q[i]
            };
            *q = in_q[i] * in_q[i + half];
          });
      } else {
        for i in 0..half {
          op[i] = if leaf_ones {
            in_q[i] + in_q[i + half]
          } else {
            in_p[i] * in_q[i + half] + in_p[i + half] * in_q[i]
          };
          oq[i] = in_q[i] * in_q[i + half];
        }
      }
    }
    levels_p[k] = op;
    levels_q[k] = oq;
  }
  (levels_p, levels_q)
}

/// One sumcheck round's unweighted integrand sums `(h(0), h(∞))` over the
/// halved cube, with the Gruen/Dao–Thaler eq-split weights (`lo` block
/// table, optional `hi` per-block factors, `blk_len` block size). The
/// integrand is `g = p0·q1 + p1·q0 + λ·q0·q1`; for all-ones leaf
/// numerators (`leaf_ones`) the `a`-tables are implicit. Shared by the
/// single-tree and lockstep multi-tree provers.
#[allow(clippy::too_many_arguments)]
fn round_h_sums<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  a0: &[E::Scalar],
  a1: &[E::Scalar],
  b0: &[E::Scalar],
  b1: &[E::Scalar],
  leaf_ones: bool,
  lambda: E::Scalar,
  lo: &[E::Scalar],
  hi: &[E::Scalar],
  blk_len: usize,
) -> [E::Scalar; 2] {
  let h = b0.len() / 2;
  debug_assert_eq!(
    if hi.is_empty() {
      blk_len
    } else {
      hi.len() * blk_len
    },
    h
  );
  let g_at = |i: usize| -> [E::Scalar; 2] {
    let b0d = b0[i + h] - b0[i];
    let b1d = b1[i + h] - b1[i];
    if leaf_ones {
      let g0 = b0[i] + b1[i] + lambda * (b0[i] * b1[i]);
      let ginf = lambda * (b0d * b1d);
      return [g0, ginf];
    }
    let a0d = a0[i + h] - a0[i];
    let a1d = a1[i + h] - a1[i];
    let g0 = a0[i] * b1[i] + a1[i] * b0[i] + lambda * (b0[i] * b1[i]);
    let ginf = a0d * b1d + a1d * b0d + lambda * (b0d * b1d);
    [g0, ginf]
  };
  let add2 = |mut a: [E::Scalar; 2], b: [E::Scalar; 2]| {
    a[0] += b[0];
    a[1] += b[1];
    a
  };
  let block = |bh: usize, hi_w: E::Scalar| -> [E::Scalar; 2] {
    let base = bh * blk_len;
    let mut s = [E::Scalar::ZERO; 2];
    for (il, lw) in lo[..blk_len].iter().enumerate() {
      let g = g_at(base + il);
      s[0] += *lw * g[0];
      s[1] += *lw * g[1];
    }
    [hi_w * s[0], hi_w * s[1]]
  };
  if hi.is_empty() {
    block(0, E::Scalar::ONE)
  } else if h >= PAR_THRESHOLD {
    (0..hi.len())
      .into_par_iter()
      .map(|bh| block(bh, hi[bh]))
      .reduce(|| [E::Scalar::ZERO; 2], add2)
  } else {
    (0..hi.len()).fold([E::Scalar::ZERO; 2], |acc, bh| add2(acc, block(bh, hi[bh])))
  }
}

/// Direct evaluation of `h(1)` (the `t = 1` half of the round integrand)
/// — the negligible-probability fallback when the Gruen prefix factor
/// isn't invertible.
fn round_h1_direct<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  a0: &[E::Scalar],
  a1: &[E::Scalar],
  b0: &[E::Scalar],
  b1: &[E::Scalar],
  leaf_ones: bool,
  lambda: E::Scalar,
  w_full: &[E::Scalar],
) -> E::Scalar {
  let h = b0.len() / 2;
  (0..h).fold(E::Scalar::ZERO, |s, i| {
    let g = if leaf_ones {
      b0[i + h] + b1[i + h] + lambda * (b0[i + h] * b1[i + h])
    } else {
      a0[i + h] * b1[i + h] + a1[i + h] * b0[i + h] + lambda * (b0[i + h] * b1[i + h])
    };
    s + w_full[i] * g
  })
}

/// Prove that `(root_p, root_q)` is the sum of the leaf fractions
/// `(p_leaves[i], q_leaves[i])`, reducing the root claim to a single
/// evaluation claim on the leaf MLEs.
fn gkr_prove<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  p_leaves: Vec<E::Scalar>,
  q_leaves: Vec<E::Scalar>,
  ones_numerator: bool,
  transcript: &mut E::TE,
) -> Result<GkrOut<E>, SpartanError> {
  let n = p_leaves.len();
  assert!(n.is_power_of_two() && n == q_leaves.len() && n >= 1);
  let d = n.trailing_zeros() as usize;
  let (levels_p, levels_q) = build_levels::<E>(p_leaves, q_leaves, ones_numerator, LeafHint::None);
  let cubic = CubicConsts::<E::Scalar>::new();

  let root_p = levels_p[0][0];
  let root_q = levels_q[0][0];
  transcript.absorb(b"gkr_root_p", &root_p);
  transcript.absorb(b"gkr_root_q", &root_q);

  let mut layers = Vec::with_capacity(d);
  let mut point: Vec<E::Scalar> = Vec::new();
  let mut claim_p = root_p;
  let mut claim_q = root_q;

  for k in 0..d {
    let lambda = transcript.squeeze(b"gkr_lambda")?;
    let half = 1usize << k; // size of the x-cube for this layer's sumcheck
    // Witness trees have all-ones leaf numerators; the MLE of all-ones
    // is identically 1 under binding, so the leaf layer's a-tables are
    // implicit (g = b0 + b1 + λ·b0·b1, h(∞) = λ·b0d·b1d) and never
    // allocated or bound. The emitted round polynomials are identical.
    let leaf_ones = ones_numerator && k + 1 == d;

    let (mut a0, mut a1) = if leaf_ones {
      (Vec::new(), Vec::new())
    } else {
      (
        levels_p[k + 1][0..half].to_vec(),
        levels_p[k + 1][half..2 * half].to_vec(),
      )
    };
    let mut b0 = levels_q[k + 1][0..half].to_vec();
    let mut b1 = levels_q[k + 1][half..2 * half].to_vec();

    // Gruen eq-factoring: the integrand is eq(ρ; x)·g(x) with
    // g = p0·q1 + p1·q0 + λ·q0·q1 of degree 2 per variable, and
    // eq(ρ; (r_{<j}, t, x_rest)) = E_pref · E_j(t) · eq(ρ_{>j}, x_rest)
    // where E_j(t) = (1−t)(1−ρ_j) + t·ρ_j is linear. So the round
    // polynomial is s(t) = E_pref·E_j(t)·h(t) with h of degree 2: per
    // index we only accumulate h(0) and the quadratic coefficient h(∞)
    // (from table diffs), and recover h(1) from the running claim via
    // s(0) + s(1) = claim. The eq table is never extrapolated or bound;
    // the per-round weights are the suffix tables eq(ρ_{>j}, ·),
    // precomputed here in one backward pass (suffix[j] covers ρ_{j..};
    // round j uses suffix[j+1]). The emitted s(0..3) are exactly the
    // same field elements as the unfactored evaluation, so the
    // transcript and verifier are unchanged.
    // Dao–Thaler split: instead of materializing every suffix table
    // (Σ 2^j work, serial), factor eq(ρ_{>j}, x) = eq(ρ_{j+1..m}, x_hi)
    // · eq(ρ_{m..k}, x_lo) with m ≈ k/2. The lo table is built once per
    // layer; the hi tables shrink per round and cost O(2^{k/2}) total.
    // Since h(0)/h(∞) are linear in the weight, the hi factor is hoisted
    // out of the inner loop (one mult per block), so per-index cost is
    // unchanged and the emitted round polynomials are identical.
    // Asymmetric split: cap the lo table at 2^7 so the hi side keeps
    // h/128 parallel blocks per round (parallel grain matters more than
    // balancing the two table sizes — both builds are cheap either way).
    let m = k.saturating_sub(7);
    let lo_tbl: Vec<E::Scalar> = if m < k {
      EqPolynomial::evals_from_points(&point[m..k])
    } else {
      vec![E::Scalar::ONE]
    };
    let lo_len = lo_tbl.len();

    let mut e_pref = E::Scalar::ONE;
    let mut claim = claim_p + lambda * claim_q;
    let mut round_polys = Vec::with_capacity(k);
    let mut challenges = Vec::with_capacity(k);

    for round in 0..k {
      let h = b0.len() / 2;
      let c = point[round];
      // Per-round hi table (eq over ρ_{round+1..m}); for late rounds the
      // remaining suffix lives entirely in the lo part, so build it
      // directly (small, ≤ 2^{k-m}).
      let (hi_tbl, blk_len) = if round + 1 < m {
        (
          EqPolynomial::<E::Scalar>::evals_from_points(&point[round + 1..m]),
          lo_len,
        )
      } else {
        let direct = if round + 1 < k {
          EqPolynomial::<E::Scalar>::evals_from_points(&point[round + 1..k])
        } else {
          vec![E::Scalar::ONE]
        };
        let l = direct.len();
        (direct, l)
      };
      let lo: &[E::Scalar] = if round + 1 < m { &lo_tbl } else { &hi_tbl };
      let hi: &[E::Scalar] = if round + 1 < m { &hi_tbl } else { &[] };
      debug_assert_eq!(
        if hi.is_empty() {
          blk_len
        } else {
          hi.len() * blk_len
        },
        h
      );

      let [h0, hinf] = round_h_sums::<E>(&a0, &a1, &b0, &b1, leaf_ones, lambda, lo, hi, blk_len);

      // s(t) = E_pref·E(t)·h(t), E(t) = (1−t)(1−c) + t·c.
      let e0 = e_pref * (E::Scalar::ONE - c);
      let e1 = e_pref * c;
      let s0 = e0 * h0;
      let s1 = claim - s0;
      // h(1) from the claim; direct evaluation as a (negligible-
      // probability) fallback when E_pref·c isn't invertible.
      let h1 = match Option::<E::Scalar>::from(e1.invert()) {
        Some(e1_inv) => s1 * e1_inv,
        None => {
          let w_full = if round + 1 < k {
            EqPolynomial::<E::Scalar>::evals_from_points(&point[round + 1..k])
          } else {
            vec![E::Scalar::ONE]
          };
          round_h1_direct::<E>(&a0, &a1, &b0, &b1, leaf_ones, lambda, &w_full)
        }
      };
      // h(t) = h0 + b·t + c2·t².
      let c2 = hinf;
      let bq = h1 - h0 - c2;
      let two = E::Scalar::from(2);
      let three = E::Scalar::from(3);
      let h2 = h0 + two * bq + E::Scalar::from(4) * c2;
      let h3 = h0 + three * bq + E::Scalar::from(9) * c2;
      let e2 = e_pref * (three * c - E::Scalar::ONE);
      let e3 = e_pref * (E::Scalar::from(5) * c - two);
      let s = [s0, s1, e2 * h2, e3 * h3];

      for v in &s {
        transcript.absorb(b"gkr_rp", v);
      }
      let ri = transcript.squeeze(b"gkr_chal")?;
      if !leaf_ones {
        bind_top(&mut a0, ri);
        bind_top(&mut a1, ri);
      }
      bind_top(&mut b0, ri);
      bind_top(&mut b1, ri);
      claim = eval_cubic_with(&cubic, &s, ri);
      e_pref *= (E::Scalar::ONE - c) * (E::Scalar::ONE - ri) + c * ri;
      round_polys.push(s);
      challenges.push(ri);
    }

    let (p0, p1) = if leaf_ones {
      (E::Scalar::ONE, E::Scalar::ONE)
    } else {
      (a0[0], a1[0])
    };
    let (q0, q1) = (b0[0], b1[0]);
    transcript.absorb(b"gkr_p0", &p0);
    transcript.absorb(b"gkr_p1", &p1);
    transcript.absorb(b"gkr_q0", &q0);
    transcript.absorb(b"gkr_q1", &q1);
    let c = transcript.squeeze(b"gkr_c")?;

    // Next layer's claim point is (c, challenges) with c the top variable.
    let mut next_point = Vec::with_capacity(k + 1);
    next_point.push(c);
    next_point.extend_from_slice(&challenges);
    point = next_point;
    claim_p = (E::Scalar::ONE - c) * p0 + c * p1;
    claim_q = (E::Scalar::ONE - c) * q0 + c * q1;

    layers.push(GkrLayerProof {
      round_polys,
      p0,
      p1,
      q0,
      q1,
    });
  }

  Ok(GkrOut {
    proof: GkrProof { layers },
    root_p,
    root_q,
    leaf_point: point,
    leaf_p: claim_p,
    leaf_q: claim_q,
  })
}

/// Verify a fractional-sum GKR proof against the claimed root fraction and the
/// expected number of layers `d`. Returns the reduced leaf claim
/// `(point, p_leaf(point), q_leaf(point))`.
fn gkr_verify<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  root_p: E::Scalar,
  root_q: E::Scalar,
  d: usize,
  proof: &GkrProof<E>,
  transcript: &mut E::TE,
) -> Result<(Vec<E::Scalar>, E::Scalar, E::Scalar), SpartanError> {
  if proof.layers.len() != d {
    return Err(SpartanError::ProofVerifyError {
      reason: "logup-gkr: wrong number of layers".to_string(),
    });
  }
  transcript.absorb(b"gkr_root_p", &root_p);
  transcript.absorb(b"gkr_root_q", &root_q);

  let cubic = CubicConsts::<E::Scalar>::new();
  let mut point: Vec<E::Scalar> = Vec::new();
  let mut claim_p = root_p;
  let mut claim_q = root_q;

  for k in 0..d {
    let lp = &proof.layers[k];
    if lp.round_polys.len() != k {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: wrong round count in layer".to_string(),
      });
    }
    let lambda = transcript.squeeze(b"gkr_lambda")?;
    let mut claim = claim_p + lambda * claim_q;
    let mut challenges = Vec::with_capacity(k);

    for s in &lp.round_polys {
      // Sumcheck consistency: s(0) + s(1) == running claim.
      if s[0] + s[1] != claim {
        return Err(SpartanError::ProofVerifyError {
          reason: "logup-gkr: sumcheck round mismatch".to_string(),
        });
      }
      for v in s {
        transcript.absorb(b"gkr_rp", v);
      }
      let ri = transcript.squeeze(b"gkr_chal")?;
      claim = eval_cubic_with(&cubic, s, ri);
      challenges.push(ri);
    }

    transcript.absorb(b"gkr_p0", &lp.p0);
    transcript.absorb(b"gkr_p1", &lp.p1);
    transcript.absorb(b"gkr_q0", &lp.q0);
    transcript.absorb(b"gkr_q1", &lp.q1);

    // Final sumcheck check against the gate relation:
    //   eq(point, ρ')·[p0·q1 + p1·q0 + λ·q0·q1] == claim.
    let eq_val = EqPolynomial::new(point.clone()).evaluate(&challenges);
    let gate = lp.p0 * lp.q1 + lp.p1 * lp.q0 + lambda * (lp.q0 * lp.q1);
    if eq_val * gate != claim {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: layer gate check failed".to_string(),
      });
    }

    let c = transcript.squeeze(b"gkr_c")?;
    let mut next_point = Vec::with_capacity(k + 1);
    next_point.push(c);
    next_point.extend_from_slice(&challenges);
    point = next_point;
    claim_p = (E::Scalar::ONE - c) * lp.p0 + c * lp.p1;
    claim_q = (E::Scalar::ONE - c) * lp.q0 + c * lp.q1;
  }

  Ok((point, claim_p, claim_q))
}

/// Prove several fraction trees IN LOCKSTEP with shared challenges and a
/// γ-batched transcript: all roots are absorbed before any challenge,
/// then every layer/round advances all still-active trees together. Per
/// round the trees' cubic round polynomials are combined into ONE
/// γ-power RLC `S(X) = Σ_i γ^i s_i(X)` — only `S` is absorbed and
/// serialized — before the one shared challenge is squeezed. Soundness
/// is the standard batched-sumcheck argument: γ is squeezed at layer
/// start, AFTER the previous layer's per-tree finals (which fix every
/// tree's layer claim) are absorbed, so a cheating claim survives the
/// RLC with probability ≤ (#trees−1)/|F| per layer, on top of the usual
/// per-round Schwartz–Zippel loss. The lockstep gives (a) per-round
/// work that is independent ACROSS trees — the multithreading unlock
/// the serial per-tree loop lacked, (b) shared eq/suffix tables and
/// Gruen prefix factors per round, (c) identical leaf points for
/// equal-depth trees, whose PCS claims then share batched-open weight
/// passes, and (d) round-polynomial proof size `Θ(max_d²)` instead of
/// `Σ_t Θ(d_t²)`. Trees may differ in depth: all start at layer 0 and a
/// tree exits after its last layer with its leaf claim at the
/// then-current shared point. Per-tree input-layer finals remain in the
/// proof (they seed the next layer's claims), grouped per layer in
/// [`GkrMultiLayerProof`].
/// Structured-leaf hint for the leaf-skip integer fast path. A
/// prover-side hint only — it never affects the transcript, and the
/// skip falls back to field arithmetic without it.
#[derive(Clone, Copy)]
pub(crate) enum LeafHint<'a, F> {
  /// No structure exposed.
  None,
  /// All-ones numerators and affine denominators `q[i] = offset + raw[i]`
  /// with small `raw` (the witness trees). The i64/u64 basis
  /// accumulators need `raw < 2^24`.
  OnesAffine { offset: F, raw: &'a [u64] },
  /// The table tree: numerators are the (small) multiplicities and
  /// denominators are index-affine, `q[i] = offset + i`.
  TableAffine { offset: F, mult: &'a [u64] },
}

/// Per-active-tree working state for one lockstep layer.
struct Ls<'a, F> {
  t: usize,
  a0: Vec<F>,
  a1: Vec<F>,
  b0: Vec<F>,
  b1: Vec<F>,
  leaf_ones: bool,
  /// Leaf structure, present only at the tree's leaf layer.
  raw: LeafHint<'a, F>,
  claim: F,
}

/// One tree's 16 evaluations for the leaf-skip round:
/// `grid[t1·4+t2] = eq((t1,t2); c01) · Σ_z eq(z; c_z) · g(T̂(t1,t2,z))`,
/// where `T̂` extends each table multilinearly in the two top index bits
/// to the integer nodes `{0,1,2,3}` and `g` is the layer integrand.
/// `wz` is the eq table over the remaining point coordinates (sums
/// to 1) and `wz_scaled` its `2^64`-scaled image, so the structured
/// fast path can accumulate small-integer data with one-fold scaled
/// multiplies. Falls back to plain field arithmetic when the leaf
/// structure is unavailable.
fn skip_grid_for_tree<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  s: &Ls<'_, E::Scalar>,
  lambda: E::Scalar,
  wz: &[E::Scalar],
  wz_scaled: &[E::Scalar],
  q4: usize,
  e01: &[E::Scalar; 16],
) -> [E::Scalar; 16] {
  if s.leaf_ones
    && let LeafHint::OnesAffine { offset: r, raw } = s.raw
  {
    // Integer fast path: b = r + w elementwise, so with Σ_z eq_z = 1
    //   Σ_z eq_z·g = (2r + λr²) + (1 + λr)·Σ_z eq_z·(ŵ0+ŵ1) + λ·Σ_z eq_z·ŵ0ŵ1.
    // The bivariate restrictions of the two data sums are determined by
    // 4 bilinear and 9 symmetrized-quadratic BASIS sums, so the hot
    // loop accumulates only those (13 unsigned one-fold multiplies per
    // index — the small-integer extrapolation happens once at the end,
    // not per index):
    //   lin[y]     = Σ_z eq_z·(w0+w1)[y·q4+z]
    //   quad[a][b] = Σ_z eq_z·Σ_{y1+y1'=a, y2+y2'=b} w0[y1y2]·w1[y1'y2']
    // u64 bounds: w < 2^24 (caller-guaranteed) ⇒ products < 2^48, the
    // widest basis sum (a=b=1, four products) < 2^50.
    let h = raw.len() / 2;
    debug_assert_eq!(h, 4 * q4);
    let (w0, w1) = raw.split_at(h);
    let c0 = r + r + lambda * r * r;
    let a1c = E::Scalar::ONE + lambda * r;
    let mut lin = [E::Scalar::ZERO; 4];
    let mut quad = [[E::Scalar::ZERO; 3]; 3];
    for z in 0..q4 {
      let wsc = wz_scaled[z];
      let v0: [u64; 4] = core::array::from_fn(|y| w0[y * q4 + z]);
      let v1: [u64; 4] = core::array::from_fn(|y| w1[y * q4 + z]);
      for y in 0..4 {
        lin[y] += wsc.mul_u64_scaled(v0[y] + v1[y]);
      }
      quad[0][0] += wsc.mul_u64_scaled(v0[0] * v1[0]);
      quad[0][1] += wsc.mul_u64_scaled(v0[0] * v1[1] + v0[1] * v1[0]);
      quad[0][2] += wsc.mul_u64_scaled(v0[1] * v1[1]);
      quad[1][0] += wsc.mul_u64_scaled(v0[0] * v1[2] + v0[2] * v1[0]);
      quad[1][1] +=
        wsc.mul_u64_scaled(v0[0] * v1[3] + v0[1] * v1[2] + v0[2] * v1[1] + v0[3] * v1[0]);
      quad[1][2] += wsc.mul_u64_scaled(v0[1] * v1[3] + v0[3] * v1[1]);
      quad[2][0] += wsc.mul_u64_scaled(v0[2] * v1[2]);
      quad[2][1] += wsc.mul_u64_scaled(v0[2] * v1[3] + v0[3] * v1[2]);
      quad[2][2] += wsc.mul_u64_scaled(v0[3] * v1[3]);
    }
    // Assemble the 16 grid values from the 13 basis sums. Per-variable
    // factors at the integer nodes t ∈ {0,1,2,3}: linear eqbit values
    // (EXT2) and symmetrized quadratics ((1-t)², t(1-t), t²).
    const Q2: [[i64; 3]; 4] = [[1, 0, 0], [0, 0, 1], [1, -2, 4], [4, -6, 9]];
    let mul_small = |x: E::Scalar, k: i64| -> E::Scalar {
      let m = x * E::Scalar::from(k.unsigned_abs());
      if k < 0 { -m } else { m }
    };
    return core::array::from_fn(|g| {
      let (i1, i2) = (g / 4, g % 4);
      let mut s1 = E::Scalar::ZERO;
      for y1 in 0..2 {
        for y2 in 0..2 {
          s1 += mul_small(lin[y1 * 2 + y2], EXT2[i1][y1] * EXT2[i2][y2]);
        }
      }
      let mut s2 = E::Scalar::ZERO;
      for a in 0..3 {
        for b in 0..3 {
          s2 += mul_small(quad[a][b], Q2[i1][a] * Q2[i2][b]);
        }
      }
      e01[g] * (c0 + a1c * s1 + lambda * s2)
    });
  }

  if !s.leaf_ones
    && let LeafHint::TableAffine { offset: r, mult } = s.raw
  {
    // Table-tree fast path: a = m (small) and b = r + j with j the leaf
    // index, whose bit-multilinear extension is LINEAR in the skip
    // nodes: for the b0 half, b̂(t1,t2)[z] = r + z + J with
    // J = t1·2^{k-1} + t2·2^{k-2}; the b1 half adds 2^k. Expanding the
    // integrand in r leaves the basis sums
    //   Mj_y = Σ_z eq_z·mj[y·q4+z],  MZj_y = Σ_z eq_z·mj[y·q4+z]·z,
    //   Z1 = Σ_z eq_z·z,  Z2 = Σ_z eq_z·z²
    // (18 one-fold multiplies per index).
    let h = mult.len() / 2;
    debug_assert_eq!(h, 4 * q4);
    let (m0, m1) = mult.split_at(h);
    let mut mm = [[E::Scalar::ZERO; 4]; 2];
    let mut mz = [[E::Scalar::ZERO; 4]; 2];
    let mut z1 = E::Scalar::ZERO;
    let mut z2 = E::Scalar::ZERO;
    for z in 0..q4 {
      let wsc = wz_scaled[z];
      let zu = z as u64;
      z1 += wsc.mul_u64_scaled(zu);
      z2 += wsc.mul_u64_scaled(zu * zu);
      for y in 0..4 {
        let (a, b) = (m0[y * q4 + z], m1[y * q4 + z]);
        mm[0][y] += wsc.mul_u64_scaled(a);
        mm[1][y] += wsc.mul_u64_scaled(b);
        mz[0][y] += wsc.mul_u64_scaled(a * zu);
        mz[1][y] += wsc.mul_u64_scaled(b * zu);
      }
    }
    let mul_small = |x: E::Scalar, k: i64| -> E::Scalar {
      let m = x * E::Scalar::from(k.unsigned_abs());
      if k < 0 { -m } else { m }
    };
    let half_k = (4 * q4) as u64;
    return core::array::from_fn(|g| {
      let (i1, i2) = (g / 4, g % 4);
      let comb = |t: &[E::Scalar; 4]| -> E::Scalar {
        let mut acc = E::Scalar::ZERO;
        for y1 in 0..2 {
          for y2 in 0..2 {
            acc += mul_small(t[y1 * 2 + y2], EXT2[i1][y1] * EXT2[i2][y2]);
          }
        }
        acc
      };
      let (mh0, mh1) = (comb(&mm[0]), comb(&mm[1]));
      let (mzh0, mzh1) = (comb(&mz[0]), comb(&mz[1]));
      let j0 = (i1 as u64) * (half_k / 2) + (i2 as u64) * (half_k / 4);
      let j1 = j0 + half_k;
      // m̂0·b̂1 + m̂1·b̂0 sums to r·(M̂0+M̂1) + MẐ0 + J1·M̂0 + MẐ1 + J0·M̂1;
      // λ·b̂0·b̂1 sums to λ·(r² + r·(2Z1 + J0+J1) + Z2 + (J0+J1)Z1 + J0J1).
      let lin =
        mzh0 + E::Scalar::from(j1) * mh0 + mzh1 + E::Scalar::from(j0) * mh1 + r * (mh0 + mh1);
      let quad = z2
        + E::Scalar::from(j0 + j1) * z1
        + E::Scalar::from(j0 * j1)
        + r * (z1 + z1 + E::Scalar::from(j0 + j1))
        + r * r;
      e01[g] * (lin + lambda * quad)
    });
  }

  // Generic field-arithmetic path (any tree shape; correctness only —
  // configs that reach it are off the structured hot path).
  let ext_f: [[E::Scalar; 2]; 4] = core::array::from_fn(|i| {
    core::array::from_fn(|y| {
      let v = EXT2[i][y];
      let f = E::Scalar::from(v.unsigned_abs());
      if v < 0 { -f } else { f }
    })
  });
  let mut acc = [E::Scalar::ZERO; 16];
  for z in 0..q4 {
    let wv = wz[z];
    let load = |t: &[E::Scalar]| -> [E::Scalar; 4] { core::array::from_fn(|y| t[y * q4 + z]) };
    let b0v = load(&s.b0);
    let b1v = load(&s.b1);
    let (a0v, a1v) = if s.leaf_ones {
      ([E::Scalar::ONE; 4], [E::Scalar::ONE; 4])
    } else {
      (load(&s.a0), load(&s.a1))
    };
    let ext2d = |t: &[E::Scalar; 4], i1: usize, i2: usize| -> E::Scalar {
      ext_f[i1][0] * (ext_f[i2][0] * t[0] + ext_f[i2][1] * t[1])
        + ext_f[i1][1] * (ext_f[i2][0] * t[2] + ext_f[i2][1] * t[3])
    };
    for i1 in 0..4 {
      for i2 in 0..4 {
        let bh0 = ext2d(&b0v, i1, i2);
        let bh1 = ext2d(&b1v, i1, i2);
        let gval = if s.leaf_ones {
          bh0 + bh1 + lambda * (bh0 * bh1)
        } else {
          ext2d(&a0v, i1, i2) * bh1 + ext2d(&a1v, i1, i2) * bh0 + lambda * (bh0 * bh1)
        };
        acc[i1 * 4 + i2] += wv * gval;
      }
    }
  }
  core::array::from_fn(|g| e01[g] * acc[g])
}

/// Bind the two skip variables at once: tensor fold with
/// `κ_y = eqbit(y1; u1)·eqbit(y2; u2)`. The structured path rebuilds
/// the tables from the raw integers with one-fold scaled multiplies;
/// otherwise two sequential top binds.
fn skip_fold_tree<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  s: &mut Ls<'_, E::Scalar>,
  u1: E::Scalar,
  u2: E::Scalar,
  q4: usize,
) {
  if s.leaf_ones
    && let LeafHint::OnesAffine { offset: r, raw } = s.raw
  {
    let h = raw.len() / 2;
    let (w0, w1) = raw.split_at(h);
    let one = E::Scalar::ONE;
    let kap: [E::Scalar; 4] = [
      (one - u1) * (one - u2),
      (one - u1) * u2,
      u1 * (one - u2),
      u1 * u2,
    ];
    let kap_s = kap.map(|x| x.scale_shift64());
    let fold = |w: &[u64]| -> Vec<E::Scalar> {
      (0..q4)
        .map(|z| {
          // Σ_y κ_y = 1, so the offsets contribute exactly one `r`.
          r + kap_s[0].mul_u64_scaled(w[z])
            + kap_s[1].mul_u64_scaled(w[q4 + z])
            + kap_s[2].mul_u64_scaled(w[2 * q4 + z])
            + kap_s[3].mul_u64_scaled(w[3 * q4 + z])
        })
        .collect()
    };
    s.b0 = fold(w0);
    s.b1 = fold(w1);
    return;
  }
  if !s.leaf_ones
    && let LeafHint::TableAffine { offset: r, mult } = s.raw
  {
    let h = mult.len() / 2;
    let (m0, m1) = mult.split_at(h);
    let one = E::Scalar::ONE;
    let kap: [E::Scalar; 4] = [
      (one - u1) * (one - u2),
      (one - u1) * u2,
      u1 * (one - u2),
      u1 * u2,
    ];
    let kap_s = kap.map(|x| x.scale_shift64());
    let fold_m = |m: &[u64]| -> Vec<E::Scalar> {
      (0..q4)
        .map(|z| {
          kap_s[0].mul_u64_scaled(m[z])
            + kap_s[1].mul_u64_scaled(m[q4 + z])
            + kap_s[2].mul_u64_scaled(m[2 * q4 + z])
            + kap_s[3].mul_u64_scaled(m[3 * q4 + z])
        })
        .collect()
    };
    s.a0 = fold_m(m0);
    s.a1 = fold_m(m1);
    // b̂[z] = r + z + J(u1,u2) is affine in z: two multiplies, then adds.
    let base = r + u1 * E::Scalar::from((2 * q4) as u64) + u2 * E::Scalar::from(q4 as u64);
    let mut acc0 = base;
    s.b0 = (0..q4)
      .map(|_| {
        let v = acc0;
        acc0 += one;
        v
      })
      .collect();
    let mut acc1 = base + E::Scalar::from((4 * q4) as u64);
    s.b1 = (0..q4)
      .map(|_| {
        let v = acc1;
        acc1 += one;
        v
      })
      .collect();
    return;
  }
  if !s.leaf_ones {
    bind_top(&mut s.a0, u1);
    bind_top(&mut s.a0, u2);
    bind_top(&mut s.a1, u1);
    bind_top(&mut s.a1, u2);
  }
  bind_top(&mut s.b0, u1);
  bind_top(&mut s.b0, u2);
  bind_top(&mut s.b1, u1);
  bind_top(&mut s.b1, u2);
}

/// One tree's input to [`gkr_prove_multi`]. `raw` optionally exposes
/// the leaf denominators' structure `q[i] = offset + raw[i]` (small
/// integers), which the leaf-skip round exploits to accumulate in
/// integer space; it is a prover-side hint only and never affects the
/// transcript.
pub(crate) struct GkrTreeInput<'a, F> {
  pub p: Vec<F>,
  pub q: Vec<F>,
  pub ones: bool,
  pub raw: LeafHint<'a, F>,
}

fn gkr_prove_multi<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  inputs: Vec<GkrTreeInput<'_, E::Scalar>>,
  transcript: &mut E::TE,
) -> Result<(GkrMultiProof<E>, Vec<GkrTreeOut<E>>), SpartanError> {
  let nt = inputs.len();
  let raws: Vec<LeafHint<'_, E::Scalar>> = inputs.iter().map(|i| i.raw).collect();
  // Build every tree's levels — pure computation, parallel over trees.
  let (_bl_span, bl_t) = crate::start_span!("gkr_build_levels");
  let mut built: Vec<(Vec<Vec<E::Scalar>>, Vec<Vec<E::Scalar>>, bool)> = inputs
    .into_par_iter()
    .map(|inp| {
      let (p, q, ones) = (inp.p, inp.q, inp.ones);
      assert!(q.len().is_power_of_two() && !q.is_empty());
      #[cfg(debug_assertions)]
      match inp.raw {
        LeafHint::None => {}
        LeafHint::OnesAffine { offset, raw } => {
          assert!(inp.ones && raw.len() == q.len());
          assert!(
            raw
              .iter()
              .zip(&q)
              .all(|(&w, qv)| offset + E::Scalar::from(w) == *qv),
            "ones-affine hint disagrees with the field leaves"
          );
        }
        LeafHint::TableAffine { offset, mult } => {
          assert!(!inp.ones && mult.len() == q.len() && mult.len() == p.len());
          assert!(
            q.iter()
              .enumerate()
              .all(|(j, qv)| offset + E::Scalar::from(j as u64) == *qv)
              && mult
                .iter()
                .zip(&p)
                .all(|(&m, pv)| E::Scalar::from(m) == *pv),
            "table-affine hint disagrees with the field leaves"
          );
        }
      }
      let (lp, lq) = build_levels::<E>(p, q, ones, inp.raw);
      (lp, lq, ones)
    })
    .collect();
  tracing::info!(elapsed_ms = %bl_t.elapsed().as_millis(), "gkr_build_levels");
  drop(_bl_span);
  let depths: Vec<usize> = built.iter().map(|(lp, _, _)| lp.len() - 1).collect();
  let max_d = depths.iter().copied().max().unwrap_or(0);
  let cubic = CubicConsts::<E::Scalar>::new();

  let roots: Vec<(E::Scalar, E::Scalar)> = built
    .iter()
    .map(|(lp, lq, _)| {
      // Single-leaf all-ones trees elide the `p` table entirely.
      let rp = if lp[0].is_empty() {
        E::Scalar::ONE
      } else {
        lp[0][0]
      };
      (rp, lq[0][0])
    })
    .collect();

  // Absorb ALL roots before any challenge.
  for (rp, rq) in &roots {
    transcript.absorb(b"gkr_root_p", rp);
    transcript.absorb(b"gkr_root_q", rq);
  }

  let mut claim_p: Vec<E::Scalar> = roots.iter().map(|r| r.0).collect();
  let mut claim_q: Vec<E::Scalar> = roots.iter().map(|r| r.1).collect();
  let mut shared_layers: Vec<GkrMultiLayerProof<E>> = Vec::with_capacity(max_d);
  let mut leaf_points: Vec<Vec<E::Scalar>> = vec![Vec::new(); nt];
  let mut point: Vec<E::Scalar> = Vec::new();

  for k in 0..max_d {
    let lambda = transcript.squeeze(b"gkr_lambda")?;
    let gamma = transcript.squeeze(b"gkr_gamma")?;
    let half = 1usize << k;
    let mut st: Vec<Ls<'_, E::Scalar>> = Vec::new();
    for t in 0..nt {
      if k >= depths[t] {
        continue;
      }
      let (lp, lq, ones) = &mut built[t];
      let leaf_ones = *ones && k + 1 == depths[t];
      // Each level is read by exactly one layer, so MOVE it out of the
      // build instead of copying (`split_off` pays one half-copy; the
      // old `to_vec` pair paid two full ones).
      let (a0, a1) = if leaf_ones {
        (Vec::new(), Vec::new())
      } else {
        let mut v = core::mem::take(&mut lp[k + 1]);
        let a1 = v.split_off(half);
        (v, a1)
      };
      let mut qv = core::mem::take(&mut lq[k + 1]);
      let b1 = qv.split_off(half);
      st.push(Ls {
        t,
        a0,
        a1,
        b0: qv,
        b1,
        leaf_ones,
        raw: if k + 1 == depths[t] {
          raws[t]
        } else {
          LeafHint::None
        },
        claim: claim_p[t] + lambda * claim_q[t],
      });
    }
    let mut layer_round_polys: Vec<[E::Scalar; 4]> = Vec::with_capacity(k);

    // Shared Gruen/Dao–Thaler machinery (see `gkr_prove` for the
    // derivation) — the split tables and prefix factors depend only on
    // the shared point/challenges, so they are built once per round for
    // ALL trees.
    let m = k.saturating_sub(7);
    let lo_tbl: Vec<E::Scalar> = if m < k {
      EqPolynomial::evals_from_points(&point[m..k])
    } else {
      vec![E::Scalar::ONE]
    };
    let lo_len = lo_tbl.len();
    let mut e_pref = E::Scalar::ONE;
    let mut challenges: Vec<E::Scalar> = Vec::with_capacity(k);

    // Leaf-layer bivariate skip: when every active tree is at its leaf
    // (tables still structured input data), replace rounds 0 and 1 with
    // ONE message — the γ-combined degree-(3,3) restriction of the
    // first two variables on the {0,1,2,3}² grid — then bind both
    // variables at once.
    let active_depths: Vec<usize> = st.iter().map(|s| depths[s.t]).collect();
    let mut skip_grid: Option<Vec<E::Scalar>> = None;
    let mut skip_rounds = 0usize;
    if skip_at_layer(k, &active_depths) {
      let q4 = 1usize << (k - 2);
      let wz = EqPolynomial::<E::Scalar>::evals_from_points(&point[2..k]);
      let wz_scaled: Vec<E::Scalar> = wz.iter().map(|w| w.scale_shift64()).collect();
      let e01: [E::Scalar; 16] =
        core::array::from_fn(|g| eq_at_node(point[0], g / 4) * eq_at_node(point[1], g % 4));
      let grids: Vec<[E::Scalar; 16]> = st
        .par_iter()
        .map(|s| skip_grid_for_tree::<E>(s, lambda, &wz, &wz_scaled, q4, &e01))
        .collect();
      let mut comb = [E::Scalar::ZERO; 16];
      let mut g = E::Scalar::ONE;
      for gr in &grids {
        for (c, v) in comb.iter_mut().zip(gr.iter()) {
          *c += g * *v;
        }
        g *= gamma;
      }
      for v in &comb {
        transcript.absorb(b"gkr_skip", v);
      }
      let u1 = transcript.squeeze(b"gkr_chal")?;
      let u2 = transcript.squeeze(b"gkr_chal")?;
      st.par_iter_mut().zip(grids.par_iter()).for_each(|(s, gr)| {
        skip_fold_tree::<E>(s, u1, u2, q4);
        s.claim = eval_grid16(&cubic, gr, u1, u2);
      });
      e_pref *= ((E::Scalar::ONE - point[0]) * (E::Scalar::ONE - u1) + point[0] * u1)
        * ((E::Scalar::ONE - point[1]) * (E::Scalar::ONE - u2) + point[1] * u2);
      challenges.push(u1);
      challenges.push(u2);
      skip_grid = Some(comb.to_vec());
      skip_rounds = 2;
    }

    for round in skip_rounds..k {
      let c = point[round];
      let (hi_tbl, blk_len) = if round + 1 < m {
        (
          EqPolynomial::<E::Scalar>::evals_from_points(&point[round + 1..m]),
          lo_len,
        )
      } else {
        let direct = if round + 1 < k {
          EqPolynomial::<E::Scalar>::evals_from_points(&point[round + 1..k])
        } else {
          vec![E::Scalar::ONE]
        };
        let l = direct.len();
        (direct, l)
      };
      let lo: &[E::Scalar] = if round + 1 < m { &lo_tbl } else { &hi_tbl };
      let hi: &[E::Scalar] = if round + 1 < m { &hi_tbl } else { &[] };

      let e0c = e_pref * (E::Scalar::ONE - c);
      let e1c = e_pref * c;
      let e1_inv = Option::<E::Scalar>::from(e1c.invert());
      let two = E::Scalar::from(2);
      let three = E::Scalar::from(3);
      let e2 = e_pref * (three * c - E::Scalar::ONE);
      let e3 = e_pref * (E::Scalar::from(5) * c - two);

      // Per-tree round polynomials — independent across trees.
      let ss: Vec<[E::Scalar; 4]> = st
        .par_iter()
        .map(|s| {
          let [h0, hinf] = round_h_sums::<E>(
            &s.a0,
            &s.a1,
            &s.b0,
            &s.b1,
            s.leaf_ones,
            lambda,
            lo,
            hi,
            blk_len,
          );
          let s0 = e0c * h0;
          let s1 = s.claim - s0;
          let h1 = match e1_inv {
            Some(inv) => s1 * inv,
            None => {
              let w_full = if round + 1 < k {
                EqPolynomial::<E::Scalar>::evals_from_points(&point[round + 1..k])
              } else {
                vec![E::Scalar::ONE]
              };
              round_h1_direct::<E>(&s.a0, &s.a1, &s.b0, &s.b1, s.leaf_ones, lambda, &w_full)
            }
          };
          let c2 = hinf;
          let bq = h1 - h0 - c2;
          let h2 = h0 + two * bq + E::Scalar::from(4) * c2;
          let h3 = h0 + three * bq + E::Scalar::from(9) * c2;
          [s0, s1, e2 * h2, e3 * h3]
        })
        .collect();

      // γ-RLC across trees: one combined cubic goes on the transcript
      // (and in the proof); each tree still tracks its own claim for the
      // next round's Gruen shortcut.
      let mut comb = [E::Scalar::ZERO; 4];
      let mut g = E::Scalar::ONE;
      for s in &ss {
        for (cj, sj) in comb.iter_mut().zip(s.iter()) {
          *cj += g * sj;
        }
        g *= gamma;
      }
      for v in &comb {
        transcript.absorb(b"gkr_rp", v);
      }
      layer_round_polys.push(comb);
      let ri = transcript.squeeze(b"gkr_chal")?;
      st.par_iter_mut().zip(ss.par_iter()).for_each(|(s, sp)| {
        if !s.leaf_ones {
          bind_top(&mut s.a0, ri);
          bind_top(&mut s.a1, ri);
        }
        bind_top(&mut s.b0, ri);
        bind_top(&mut s.b1, ri);
        s.claim = eval_cubic_with(&cubic, sp, ri);
      });
      e_pref *= (E::Scalar::ONE - c) * (E::Scalar::ONE - ri) + c * ri;
      challenges.push(ri);
    }

    // Layer end: absorb every active tree's input-layer evaluations,
    // then one shared merge challenge.
    let finals: Vec<(E::Scalar, E::Scalar, E::Scalar, E::Scalar)> = st
      .iter()
      .map(|s| {
        let (p0, p1) = if s.leaf_ones {
          (E::Scalar::ONE, E::Scalar::ONE)
        } else {
          (s.a0[0], s.a1[0])
        };
        (p0, p1, s.b0[0], s.b1[0])
      })
      .collect();
    for (p0, p1, q0, q1) in &finals {
      transcript.absorb(b"gkr_p0", p0);
      transcript.absorb(b"gkr_p1", p1);
      transcript.absorb(b"gkr_q0", q0);
      transcript.absorb(b"gkr_q1", q1);
    }
    let cc = transcript.squeeze(b"gkr_c")?;
    let mut next_point = Vec::with_capacity(k + 1);
    next_point.push(cc);
    next_point.extend_from_slice(&challenges);
    point = next_point;

    for (s, (p0, p1, q0, q1)) in st.into_iter().zip(finals.iter().copied()) {
      let t = s.t;
      claim_p[t] = (E::Scalar::ONE - cc) * p0 + cc * p1;
      claim_q[t] = (E::Scalar::ONE - cc) * q0 + cc * q1;
      if k + 1 == depths[t] {
        leaf_points[t] = point.clone();
      }
    }
    shared_layers.push(GkrMultiLayerProof {
      skip: skip_grid,
      round_polys: layer_round_polys,
      finals: finals
        .into_iter()
        .map(|(p0, p1, q0, q1)| [p0, p1, q0, q1])
        .collect(),
    });
  }

  Ok((
    GkrMultiProof {
      layers: shared_layers,
    },
    (0..nt)
      .map(|t| GkrTreeOut {
        root_p: roots[t].0,
        root_q: roots[t].1,
        leaf_point: core::mem::take(&mut leaf_points[t]),
        leaf_p: claim_p[t],
        leaf_q: claim_q[t],
      })
      .collect(),
  ))
}

/// Verifier mirror of [`gkr_prove_multi`]: walk the shared γ-batched
/// proof in the same lockstep transcript order. Per round it checks the
/// ONE combined cubic against the γ-RLC of the active trees' claims;
/// per layer end it checks `eq · Σ_i γ^i gate_i` against the reduced
/// combined claim, then advances each tree's claim from its own finals.
/// Returns each tree's `(leaf point, p, q)` claims.
fn gkr_verify_multi<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
>(
  roots: &[(E::Scalar, E::Scalar)],
  depths: &[usize],
  proof: &GkrMultiProof<E>,
  transcript: &mut E::TE,
) -> Result<Vec<(Vec<E::Scalar>, E::Scalar, E::Scalar)>, SpartanError> {
  let nt = roots.len();
  if depths.len() != nt {
    return Err(SpartanError::ProofVerifyError {
      reason: "logup-gkr multi: tree count mismatch".to_string(),
    });
  }
  let max_d = depths.iter().copied().max().unwrap_or(0);
  if proof.layers.len() != max_d {
    return Err(SpartanError::ProofVerifyError {
      reason: "logup-gkr: wrong number of layers".to_string(),
    });
  }
  for root in roots {
    transcript.absorb(b"gkr_root_p", &root.0);
    transcript.absorb(b"gkr_root_q", &root.1);
  }
  let cubic = CubicConsts::<E::Scalar>::new();

  let mut claim_p: Vec<E::Scalar> = roots.iter().map(|r| r.0).collect();
  let mut claim_q: Vec<E::Scalar> = roots.iter().map(|r| r.1).collect();
  let mut leaf_points: Vec<Vec<E::Scalar>> = vec![Vec::new(); nt];
  let mut point: Vec<E::Scalar> = Vec::new();

  for (k, layer) in proof.layers.iter().enumerate() {
    let lambda = transcript.squeeze(b"gkr_lambda")?;
    let gamma = transcript.squeeze(b"gkr_gamma")?;
    let active: Vec<usize> = (0..nt).filter(|&t| k < depths[t]).collect();
    let active_depths: Vec<usize> = active.iter().map(|&t| depths[t]).collect();
    let skip_here = skip_at_layer(k, &active_depths);
    let expected_rounds = if skip_here { k - 2 } else { k };
    if layer.round_polys.len() != expected_rounds
      || layer.finals.len() != active.len()
      || layer.skip.is_some() != skip_here
    {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: layer shape mismatch".to_string(),
      });
    }
    // γ-RLC of the active trees' layer claims — the batched sumcheck's
    // starting claim.
    let mut claim = E::Scalar::ZERO;
    let mut g = E::Scalar::ONE;
    for &t in &active {
      claim += g * (claim_p[t] + lambda * claim_q[t]);
      g *= gamma;
    }
    let mut challenges: Vec<E::Scalar> = Vec::with_capacity(k);

    if let Some(grid) = &layer.skip {
      // The bivariate skip round: its boolean-image values must sum to
      // the layer claim; two challenges bind both variables at once.
      if grid.len() != 16 {
        return Err(SpartanError::ProofVerifyError {
          reason: "logup-gkr: malformed skip round".to_string(),
        });
      }
      if grid[0] + grid[1] + grid[4] + grid[5] != claim {
        return Err(SpartanError::ProofVerifyError {
          reason: "logup-gkr: skip round mismatch".to_string(),
        });
      }
      for v in grid {
        transcript.absorb(b"gkr_skip", v);
      }
      let u1 = transcript.squeeze(b"gkr_chal")?;
      let u2 = transcript.squeeze(b"gkr_chal")?;
      claim = eval_grid16(&cubic, grid, u1, u2);
      challenges.push(u1);
      challenges.push(u2);
    }

    for s in &layer.round_polys {
      if s[0] + s[1] != claim {
        return Err(SpartanError::ProofVerifyError {
          reason: "logup-gkr: sumcheck round mismatch".to_string(),
        });
      }
      for v in s {
        transcript.absorb(b"gkr_rp", v);
      }
      let ri = transcript.squeeze(b"gkr_chal")?;
      claim = eval_cubic_with(&cubic, s, ri);
      challenges.push(ri);
    }

    let eq_val = EqPolynomial::new(point.clone()).evaluate(&challenges);
    let mut gates = E::Scalar::ZERO;
    let mut g = E::Scalar::ONE;
    for f in &layer.finals {
      let [p0, p1, q0, q1] = *f;
      transcript.absorb(b"gkr_p0", &p0);
      transcript.absorb(b"gkr_p1", &p1);
      transcript.absorb(b"gkr_q0", &q0);
      transcript.absorb(b"gkr_q1", &q1);
      gates += g * (p0 * q1 + p1 * q0 + lambda * (q0 * q1));
      g *= gamma;
    }
    if eq_val * gates != claim {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: layer gate check failed".to_string(),
      });
    }

    let cc = transcript.squeeze(b"gkr_c")?;
    let mut next_point = Vec::with_capacity(k + 1);
    next_point.push(cc);
    next_point.extend_from_slice(&challenges);
    point = next_point;
    for (i, &t) in active.iter().enumerate() {
      let [p0, p1, q0, q1] = layer.finals[i];
      claim_p[t] = (E::Scalar::ONE - cc) * p0 + cc * p1;
      claim_q[t] = (E::Scalar::ONE - cc) * q0 + cc * q1;
      if k + 1 == depths[t] {
        leaf_points[t] = point.clone();
      }
    }
  }

  Ok(
    (0..nt)
      .map(|t| (core::mem::take(&mut leaf_points[t]), claim_p[t], claim_q[t]))
      .collect(),
  )
}

/// Evaluation claims a [`LogUpRangeProof`] reduces to, for the caller to
/// discharge with PCS openings.
#[derive(Clone, Debug)]
pub struct RangeClaims<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  /// The LogUp challenge `r`.
  pub r: E::Scalar,
  /// Point at which the witness MLE must be opened (`ρ_L`).
  pub wit_point: Vec<E::Scalar>,
  /// Claimed `w(ρ_L)` — open the witness commitment here and check equality.
  pub wit_eval: E::Scalar,
  /// Point at which the multiplicity MLE must be opened (`ρ_R`).
  pub mult_point: Vec<E::Scalar>,
  /// Claimed `m(ρ_R)` — open the multiplicity commitment here and check.
  pub mult_eval: E::Scalar,
}

/// A LogUp-GKR proof that a witness vector is range-bounded by `[0, 2^bits)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogUpRangeProof<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  p_lhs_root: E::Scalar,
  q_lhs_root: E::Scalar,
  p_rhs_root: E::Scalar,
  q_rhs_root: E::Scalar,
  lhs_gkr: GkrProof<E>,
  rhs_gkr: GkrProof<E>,
}

impl<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> LogUpRangeProof<E>
{
  /// Number of variables of the padded witness polynomial (`log2` of the
  /// witness leaf count). The witness commitment must be over this many
  /// variables, with the trailing slots padded with the value `0`.
  pub fn witness_num_vars(witness_len: usize) -> usize {
    witness_len.max(1).next_power_of_two().trailing_zeros() as usize
  }

  /// The multiplicity table `m_j = #{i : w[i] = j}` over `[0, 2^bits)`,
  /// including the power-of-two padding `0`s that [`Self::prove`] appends.
  /// A caller that commits the multiplicity polynomial (which it must do
  /// — and absorb — *before* the transcript point where `prove` squeezes
  /// the LogUp challenge `r`) gets the exact vector `prove` will use.
  /// Errors if any witness value is out of range.
  pub fn multiplicities(bits: usize, witness: &[u64]) -> Result<Vec<u64>, SpartanError> {
    let table = 1usize << bits;
    let n = witness.len().max(1).next_power_of_two();
    let mut mult = vec![0u64; table];
    for &w in witness {
      let idx = w as usize;
      if idx >= table {
        return Err(SpartanError::InvalidInputLength {
          reason: format!("logup-gkr: witness value {w} >= 2^{bits}"),
        });
      }
      mult[idx] += 1;
    }
    mult[0] += (n - witness.len()) as u64;
    Ok(mult)
  }

  /// Prove that every value in `witness` lies in `[0, 2^bits)`.
  ///
  /// Returns the proof and the [`RangeClaims`] (the witness and multiplicity
  /// evaluation claims to discharge via PCS). The witness is conceptually
  /// padded to the next power of two with the value `0` (the multiplicity of
  /// `0` is bumped accordingly), so the committed witness polynomial has
  /// [`Self::witness_num_vars`] variables.
  pub fn prove(
    bits: usize,
    witness: &[u64],
    transcript: &mut E::TE,
  ) -> Result<(Self, RangeClaims<E>), SpartanError> {
    if witness.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "logup-gkr: empty witness".to_string(),
      });
    }
    let table = 1usize << bits;
    let n = witness.len().next_power_of_two();

    // Multiplicities over the table, plus the padding `0`s. Callers that
    // commit the multiplicity polynomial (they must, before this point in
    // the transcript) obtain the identical vector from
    // [`Self::multiplicities`].
    let mult = Self::multiplicities(bits, witness)?;

    transcript.dom_sep(b"logup_range");
    let r = transcript.squeeze(b"logup_r")?;

    // Witness-side leaves: (1, r + w[i]); padding leaves use w = 0.
    let p_lhs = vec![E::Scalar::ONE; n];
    let mut q_lhs = vec![E::Scalar::ZERO; n];
    for (i, slot) in q_lhs.iter_mut().enumerate() {
      let w = if i < witness.len() { witness[i] } else { 0 };
      *slot = r + E::Scalar::from(w);
    }

    // Table-side leaves: (m_j, r + j).
    let mut p_rhs = vec![E::Scalar::ZERO; table];
    let mut q_rhs = vec![E::Scalar::ZERO; table];
    for j in 0..table {
      p_rhs[j] = E::Scalar::from(mult[j]);
      q_rhs[j] = r + E::Scalar::from(j as u64);
    }

    let lhs = gkr_prove::<E>(p_lhs, q_lhs, true, transcript)?;
    let rhs = gkr_prove::<E>(p_rhs, q_rhs, false, transcript)?;

    let claims = RangeClaims {
      r,
      wit_point: lhs.leaf_point.clone(),
      wit_eval: lhs.leaf_q - r, // q_leaf = r + w  ⇒  w = q_leaf - r
      mult_point: rhs.leaf_point.clone(),
      mult_eval: rhs.leaf_p, // p_leaf = m
    };

    Ok((
      LogUpRangeProof {
        p_lhs_root: lhs.root_p,
        q_lhs_root: lhs.root_q,
        p_rhs_root: rhs.root_p,
        q_rhs_root: rhs.root_q,
        lhs_gkr: lhs.proof,
        rhs_gkr: rhs.proof,
      },
      claims,
    ))
  }

  /// Verify the proof and return the [`RangeClaims`] the caller must discharge
  /// (witness opening at `wit_point == wit_eval`, multiplicity opening at
  /// `mult_point == mult_eval`).
  pub fn verify(
    &self,
    bits: usize,
    transcript: &mut E::TE,
  ) -> Result<RangeClaims<E>, SpartanError> {
    let d_lhs = self.lhs_gkr.layers.len();

    transcript.dom_sep(b"logup_range");
    let r = transcript.squeeze(b"logup_r")?;

    let (lhs_point, lhs_p, lhs_q) = gkr_verify::<E>(
      self.p_lhs_root,
      self.q_lhs_root,
      d_lhs,
      &self.lhs_gkr,
      transcript,
    )?;
    let (rhs_point, rhs_p, rhs_q) = gkr_verify::<E>(
      self.p_rhs_root,
      self.q_rhs_root,
      bits,
      &self.rhs_gkr,
      transcript,
    )?;

    // LogUp identity: sum_LHS == sum_RHS  ⇔  P_L/Q_L == P_R/Q_R.
    if self.q_lhs_root == E::Scalar::ZERO || self.q_rhs_root == E::Scalar::ZERO {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: zero denominator (challenge hit a pole)".to_string(),
      });
    }
    if self.p_lhs_root * self.q_rhs_root != self.p_rhs_root * self.q_lhs_root {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: LogUp identity failed (value out of range)".to_string(),
      });
    }

    // Structured leaf checks. Witness numerators are all 1, so the leaf
    // numerator MLE must evaluate to 1.
    if lhs_p != E::Scalar::ONE {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: witness numerator leaf != 1".to_string(),
      });
    }
    // Table denominators are r + idx(j); the verifier reconstructs idx in
    // closed form.
    if rhs_q != r + idx_mle_eval::<E::Scalar>(&rhs_point) {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr: table index leaf mismatch".to_string(),
      });
    }

    Ok(RangeClaims {
      r,
      wit_point: lhs_point,
      wit_eval: lhs_q - r,
      mult_point: rhs_point,
      mult_eval: rhs_p,
    })
  }
}

/// Evaluation claims a [`LogUpMultiRangeProof`] reduces to: one
/// `(point, eval)` pair per witness tree (in input order), plus the single
/// multiplicity claim. Each must be discharged with a PCS opening.
#[derive(Clone, Debug)]
pub struct MultiRangeClaims<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  /// The (shared) LogUp challenge `r`.
  pub r: E::Scalar,
  /// `(ρ_i, w_i(ρ_i))` per witness tree, in input order.
  pub wit_claims: Vec<(Vec<E::Scalar>, E::Scalar)>,
  /// Point at which the shared multiplicity MLE must be opened.
  pub mult_point: Vec<E::Scalar>,
  /// Claimed `m(mult_point)`.
  pub mult_eval: E::Scalar,
}

/// A LogUp-GKR proof that *several* witness vectors are all range-bounded
/// by `[0, 2^bits)` against ONE shared multiplicity table. The lookup
/// identity is additive, so each witness gets its own fraction tree
/// (reduced to an opening of its own commitment) while the table side —
/// the `2^bits`-leaf tree and the multiplicity commitment — is paid once:
///
/// ```text
///   Σ_b Σ_i 1/(r + w_b[i])  =  Σ_j m_j/(r + j)
/// ```
///
/// The verifier sums the witness root fractions and cross-multiplies
/// against the table root. Every witness vector must already be a power
/// of two long (committed polynomials are); the table counts them all.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogUpMultiRangeProof<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> {
  /// Per-witness-tree root fraction `(P_b, Q_b)`.
  wit_roots: Vec<(E::Scalar, E::Scalar)>,
  p_rhs_root: E::Scalar,
  q_rhs_root: E::Scalar,
  /// One shared γ-batched GKR walk over all witness trees + the table
  /// tree (see [`GkrMultiProof`]) — round polynomials are combined
  /// across trees, so proof size no longer scales with tree count.
  gkr: GkrMultiProof<E>,
}

impl<
  E: SumcheckEngine<Scalar: crate::traits::PrimeFieldExt + Serialize + serde::de::DeserializeOwned>,
> LogUpMultiRangeProof<E>
{
  /// The shared multiplicity table over all witness vectors:
  /// `m_j = Σ_b #{i : w_b[i] = j}`. Each witness must be power-of-two
  /// long (no implicit padding — committed polys already are). Callers
  /// must commit this table and absorb it *before* the transcript point
  /// where [`Self::prove`] squeezes the LogUp challenge `r`.
  pub fn multiplicities(bits: usize, witnesses: &[&[u64]]) -> Result<Vec<u64>, SpartanError> {
    let table = 1usize << bits;
    let mut mult = vec![0u64; table];
    for (b, witness) in witnesses.iter().enumerate() {
      if witness.is_empty() || !witness.len().is_power_of_two() {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "logup-gkr multi: witness {b} length {} is not a positive power of two",
            witness.len()
          ),
        });
      }
      for &w in witness.iter() {
        let idx = w as usize;
        if idx >= table {
          return Err(SpartanError::InvalidInputLength {
            reason: format!("logup-gkr multi: witness {b} value {w} >= 2^{bits}"),
          });
        }
        mult[idx] += 1;
      }
    }
    Ok(mult)
  }

  /// Prove that every value of every witness lies in `[0, 2^bits)`.
  pub fn prove(
    bits: usize,
    witnesses: &[&[u64]],
    transcript: &mut E::TE,
  ) -> Result<(Self, MultiRangeClaims<E>), SpartanError> {
    if witnesses.is_empty() {
      return Err(SpartanError::InvalidInputLength {
        reason: "logup-gkr multi: no witnesses".to_string(),
      });
    }
    let table = 1usize << bits;
    let mult = Self::multiplicities(bits, witnesses)?;

    transcript.dom_sep(b"logup_multi_range");
    let r = transcript.squeeze(b"logup_r")?;

    // Shared denominator table `r + j` for j < 2^bits, built with 2^bits
    // field ADDS (each `Scalar::from` is a Montgomery multiplication;
    // the witness trees would otherwise pay one per leaf — millions per
    // proof).
    let mut r_plus: Vec<E::Scalar> = Vec::with_capacity(table);
    let mut acc = r;
    for _ in 0..table {
      r_plus.push(acc);
      acc += E::Scalar::ONE;
    }

    // One fraction tree per witness — leaves (1, r + w_b[i]), the
    // all-ones numerator table elided — plus the shared table tree with
    // leaves (m_j, r + j), all proven IN LOCKSTEP with shared
    // challenges (see `gkr_prove_multi`; this is what lets the
    // per-round work parallelize across trees instead of running one
    // serial GKR per witness).
    let mut inputs: Vec<GkrTreeInput<'_, E::Scalar>> = Vec::with_capacity(witnesses.len() + 1);
    for witness in witnesses {
      let p = if witness.len() > 1 {
        Vec::new()
      } else {
        vec![E::Scalar::ONE]
      };
      let q: Vec<E::Scalar> = witness.iter().map(|&w| r_plus[w as usize]).collect();
      inputs.push(GkrTreeInput {
        p,
        q,
        ones: true,
        // Structured-leaf hint for the leaf-skip fast path (its u64
        // basis accumulators need witness values < 2^24).
        raw: if bits <= 24 {
          LeafHint::OnesAffine {
            offset: r,
            raw: witness,
          }
        } else {
          LeafHint::None
        },
      });
    }
    let p_rhs: Vec<E::Scalar> = mult.iter().map(|&m| E::Scalar::from(m)).collect();
    let q_rhs = r_plus;
    inputs.push(GkrTreeInput {
      p: p_rhs,
      q: q_rhs,
      ones: false,
      raw: if bits <= 24 {
        LeafHint::TableAffine {
          offset: r,
          mult: &mult,
        }
      } else {
        LeafHint::None
      },
    });

    let (gkr, mut outs) = gkr_prove_multi::<E>(inputs, transcript)?;
    let rhs = outs.pop().expect("table tree present");
    let mut wit_roots = Vec::with_capacity(witnesses.len());
    let mut wit_claims = Vec::with_capacity(witnesses.len());
    for out in outs {
      wit_roots.push((out.root_p, out.root_q));
      wit_claims.push((out.leaf_point, out.leaf_q - r));
    }

    let claims = MultiRangeClaims {
      r,
      wit_claims,
      mult_point: rhs.leaf_point,
      mult_eval: rhs.leaf_p,
    };

    Ok((
      LogUpMultiRangeProof {
        wit_roots,
        p_rhs_root: rhs.root_p,
        q_rhs_root: rhs.root_q,
        gkr,
      },
      claims,
    ))
  }

  /// Verify against the expected per-witness tree depths (`log2` of each
  /// committed witness polynomial's length — the caller MUST pin these,
  /// otherwise a prover could range-check smaller polynomials). Returns
  /// the [`MultiRangeClaims`] to discharge via PCS openings.
  pub fn verify(
    &self,
    bits: usize,
    expected_wit_depths: &[usize],
    transcript: &mut E::TE,
  ) -> Result<MultiRangeClaims<E>, SpartanError> {
    if self.wit_roots.len() != expected_wit_depths.len() || expected_wit_depths.is_empty() {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: witness tree count mismatch".to_string(),
      });
    }

    transcript.dom_sep(b"logup_multi_range");
    let r = transcript.squeeze(b"logup_r")?;

    // Lockstep verification mirroring `gkr_prove_multi`'s transcript
    // order: witness trees in input order, then the table tree.
    let mut roots: Vec<(E::Scalar, E::Scalar)> = self.wit_roots.clone();
    roots.push((self.p_rhs_root, self.q_rhs_root));
    let mut depths: Vec<usize> = expected_wit_depths.to_vec();
    depths.push(bits);
    let mut leaf_outs = gkr_verify_multi::<E>(&roots, &depths, &self.gkr, transcript)?;

    let (rhs_point, rhs_p, rhs_q) = leaf_outs.pop().expect("table tree present");
    let mut wit_claims = Vec::with_capacity(self.wit_roots.len());
    for (i, (point, leaf_p, leaf_q)) in leaf_outs.into_iter().enumerate() {
      // Witness numerators are all 1.
      if leaf_p != E::Scalar::ONE {
        return Err(SpartanError::ProofVerifyError {
          reason: format!("logup-gkr multi: witness {i} numerator leaf != 1"),
        });
      }
      wit_claims.push((point, leaf_q - r));
    }
    // Table denominators are r + idx(j), checked in closed form.
    if rhs_q != r + idx_mle_eval::<E::Scalar>(&rhs_point) {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: table index leaf mismatch".to_string(),
      });
    }

    // Summed LogUp identity: Σ_b P_b/Q_b == P_R/Q_R, via fraction folding
    // and one cross-product. Reject zero denominators (pole hits).
    if self.q_rhs_root == E::Scalar::ZERO {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: zero table denominator".to_string(),
      });
    }
    let mut sum_p = E::Scalar::ZERO;
    let mut sum_q = E::Scalar::ONE;
    for (i, &(p_b, q_b)) in self.wit_roots.iter().enumerate() {
      if q_b == E::Scalar::ZERO {
        return Err(SpartanError::ProofVerifyError {
          reason: format!("logup-gkr multi: zero denominator in witness {i}"),
        });
      }
      sum_p = sum_p * q_b + p_b * sum_q;
      sum_q *= q_b;
    }
    if sum_p * self.q_rhs_root != self.p_rhs_root * sum_q {
      return Err(SpartanError::ProofVerifyError {
        reason: "logup-gkr multi: LogUp identity failed (value out of range)".to_string(),
      });
    }

    Ok(MultiRangeClaims {
      r,
      wit_claims,
      mult_point: rhs_point,
      mult_eval: rhs_p,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
  use crate::traits::Engine;
  use ff::Field;
  use rand::{Rng, SeedableRng, rngs::StdRng};

  /// Direct MLE evaluation of a dense table at `point` (point[0] = MSB),
  /// matching the GKR's big-endian variable order.
  fn mle_eval<F: PrimeField>(table: &[F], point: &[F]) -> F {
    let chis = EqPolynomial::evals_from_points(point);
    table.iter().zip(chis.iter()).map(|(z, c)| *z * *c).sum()
  }

  #[test]
  fn idx_mle_matches_dense() {
    type F = <PallasHyraxEngine as Engine>::Scalar;
    for bits in 1..=6usize {
      let table: Vec<F> = (0..(1u64 << bits)).map(F::from).collect();
      let point: Vec<F> = (0..bits)
        .map(|i| F::from((7 * i as u64 + 3) % 101))
        .collect();
      assert_eq!(idx_mle_eval::<F>(&point), mle_eval(&table, &point));
    }
  }

  /// End-to-end: an in-range witness verifies, and the reduced witness /
  /// multiplicity claims match the true MLE evaluations.
  fn range_roundtrip_with<E: Engine>(bits: usize, witness: &[u64])
  where
    <E::Scalar as crate::traits::mod_engine::SumcheckField>::Params: Default,
  {
    type TEof<E> = <E as Engine>::TE;
    let mut tp = <TEof<E>>::new(b"logup_test");
    let (proof, claims_p) = LogUpRangeProof::<E>::prove(bits, witness, &mut tp).unwrap();

    let mut tv = <TEof<E>>::new(b"logup_test");
    let claims_v = proof.verify(bits, &mut tv).unwrap();

    // Prover and verifier agree on the reduced claims.
    assert_eq!(claims_p.r, claims_v.r);
    assert_eq!(claims_p.wit_point, claims_v.wit_point);
    assert_eq!(claims_p.wit_eval, claims_v.wit_eval);
    assert_eq!(claims_p.mult_point, claims_v.mult_point);
    assert_eq!(claims_p.mult_eval, claims_v.mult_eval);

    // The reduced claims match the true MLEs of the padded witness and the
    // multiplicity table (this is what a PCS opening would confirm).
    let table = 1usize << bits;
    let n = witness.len().next_power_of_two();
    let mut w_tbl = vec![E::Scalar::ZERO; n];
    for (i, slot) in w_tbl.iter_mut().enumerate() {
      let w = if i < witness.len() { witness[i] } else { 0 };
      *slot = E::Scalar::from(w);
    }
    let mut mult = vec![0u64; table];
    for &w in witness {
      mult[w as usize] += 1;
    }
    mult[0] += (n - witness.len()) as u64;
    let m_tbl: Vec<E::Scalar> = mult.iter().map(|&m| E::Scalar::from(m)).collect();

    assert_eq!(claims_v.wit_eval, mle_eval(&w_tbl, &claims_v.wit_point));
    assert_eq!(claims_v.mult_eval, mle_eval(&m_tbl, &claims_v.mult_point));
  }

  #[test]
  fn range_roundtrips_small() {
    range_roundtrip_with::<PallasHyraxEngine>(4, &[3, 7, 3, 0, 15, 1, 9, 3]);
  }

  #[test]
  fn range_roundtrips_non_power_of_two_witness() {
    // 5 witnesses → padded to 8 with value 0.
    range_roundtrip_with::<PallasHyraxEngine>(4, &[1, 2, 14, 14, 0]);
  }

  #[test]
  fn range_roundtrips_t256_bits8() {
    let mut rng = StdRng::seed_from_u64(42);
    let witness: Vec<u64> = (0..200).map(|_| rng.gen_range(0..256)).collect();
    range_roundtrip_with::<T256HyraxEngine>(8, &witness);
  }

  #[test]
  fn prove_rejects_out_of_range() {
    type E = PallasHyraxEngine;
    let mut tp = <E as Engine>::TE::new(b"logup_test");
    // 16 >= 2^4 = 16, out of range.
    assert!(LogUpRangeProof::<E>::prove(4, &[1, 2, 16], &mut tp).is_err());
  }

  #[test]
  fn verify_rejects_tampered_root() {
    type E = PallasHyraxEngine;
    let mut tp = <E as Engine>::TE::new(b"logup_test");
    let (mut proof, _) = LogUpRangeProof::<E>::prove(4, &[3, 7, 3, 0], &mut tp).unwrap();
    // Corrupt the witness-side numerator root: breaks the LogUp identity.
    proof.p_lhs_root += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_test");
    assert!(proof.verify(4, &mut tv).is_err());
  }

  #[test]
  fn verify_rejects_tampered_layer() {
    type E = PallasHyraxEngine;
    let mut tp = <E as Engine>::TE::new(b"logup_test");
    let (mut proof, _) =
      LogUpRangeProof::<E>::prove(4, &[3, 7, 3, 0, 1, 2, 5, 5], &mut tp).unwrap();
    // Corrupt a sumcheck round polynomial deep in the table tree.
    let last = proof.rhs_gkr.layers.len() - 1;
    proof.rhs_gkr.layers[last].round_polys[0][2] += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_test");
    assert!(proof.verify(4, &mut tv).is_err());
  }

  /// Multi-witness roundtrip: several trees of different sizes against one
  /// shared table; reduced claims match the true MLEs.
  #[test]
  fn multi_range_roundtrips() {
    type E = PallasHyraxEngine;
    type F = <E as Engine>::Scalar;
    let w0: Vec<u64> = vec![3, 7, 3, 0, 15, 1, 9, 3]; // len 8
    let w1: Vec<u64> = vec![14, 0]; // len 2
    let w2: Vec<u64> = vec![5, 5, 5, 5, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7]; // len 16
    let witnesses: Vec<&[u64]> = vec![&w0, &w1, &w2];

    let mut tp = <E as Engine>::TE::new(b"logup_multi");
    let (proof, claims_p) = LogUpMultiRangeProof::<E>::prove(4, &witnesses, &mut tp).unwrap();

    let depths = [3usize, 1, 4];
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    let claims_v = proof.verify(4, &depths, &mut tv).unwrap();

    assert_eq!(claims_p.r, claims_v.r);
    assert_eq!(claims_p.wit_claims.len(), 3);

    // Every reduced witness claim matches the true MLE of its vector, and
    // the multiplicity claim matches the shared table's MLE.
    for (b, witness) in witnesses.iter().enumerate() {
      let tbl: Vec<F> = witness.iter().map(|&w| F::from(w)).collect();
      let (point, eval) = &claims_v.wit_claims[b];
      assert_eq!(*eval, mle_eval(&tbl, point), "witness {b} claim mismatch");
    }
    let mult = LogUpMultiRangeProof::<E>::multiplicities(4, &witnesses).unwrap();
    let m_tbl: Vec<F> = mult.iter().map(|&m| F::from(m)).collect();
    assert_eq!(claims_v.mult_eval, mle_eval(&m_tbl, &claims_v.mult_point));
  }

  /// Deep witness trees over a shallow table (the MultiSwap shape):
  /// layers `bits..depth-1` have only the witness trees active, so the
  /// leaf layer takes the bivariate skip round on the integer fast
  /// path. The reduced claims must still match the true MLEs.
  #[test]
  fn multi_range_skip_layer_roundtrips() {
    type E = T256HyraxEngine;
    type F = <E as Engine>::Scalar;
    let n = 1usize << 10;
    let w0: Vec<u64> = (0..n as u64).map(|i| (i * 37 + 11) % 256).collect();
    let w1: Vec<u64> = (0..n as u64).map(|i| (i * i + 3) % 256).collect();
    let witnesses: Vec<&[u64]> = vec![&w0, &w1];

    let mut tp = <E as Engine>::TE::new(b"logup_multi_skip");
    let (proof, _) = LogUpMultiRangeProof::<E>::prove(8, &witnesses, &mut tp).unwrap();
    // The leaf layer of the deep trees must actually have taken the
    // skip round (this is what this test exists to pin).
    let leaf_layer = proof.gkr.layers.last().unwrap();
    assert!(leaf_layer.skip.is_some(), "skip round did not fire");
    assert_eq!(leaf_layer.round_polys.len(), 9 - 2);

    let mut tv = <E as Engine>::TE::new(b"logup_multi_skip");
    let claims = proof.verify(8, &[10, 10], &mut tv).unwrap();
    for (b, witness) in witnesses.iter().enumerate() {
      let tbl: Vec<F> = witness.iter().map(|&w| F::from(w)).collect();
      let (point, eval) = &claims.wit_claims[b];
      assert_eq!(*eval, mle_eval(&tbl, point), "witness {b} claim mismatch");
    }
    let mult = LogUpMultiRangeProof::<E>::multiplicities(8, &witnesses).unwrap();
    let m_tbl: Vec<F> = mult.iter().map(|&m| F::from(m)).collect();
    assert_eq!(claims.mult_eval, mle_eval(&m_tbl, &claims.mult_point));

    // Tampering with the skip grid is rejected: the boolean-image sum
    // check catches boolean-node edits, and off-boolean edits corrupt
    // the reduced claim so a later check fails.
    for idx in [0usize, 2, 7, 15] {
      let mut bad = proof.clone();
      let last = bad.gkr.layers.len() - 1;
      bad.gkr.layers[last].skip.as_mut().unwrap()[idx] += F::ONE;
      let mut tv = <E as Engine>::TE::new(b"logup_multi_skip");
      assert!(
        bad.verify(8, &[10, 10], &mut tv).is_err(),
        "tampered skip grid entry {idx} accepted"
      );
    }
    // A proof missing its skip round (shape mismatch) is rejected.
    let mut bad = proof.clone();
    let last = bad.gkr.layers.len() - 1;
    bad.gkr.layers[last].skip = None;
    let mut tv = <E as Engine>::TE::new(b"logup_multi_skip");
    assert!(bad.verify(8, &[10, 10], &mut tv).is_err());
  }

  /// Multi-witness: out-of-range value rejected at prove; tampered roots
  /// and wrong expected depths rejected at verify.
  #[test]
  fn multi_range_rejects_bad() {
    type E = PallasHyraxEngine;
    let w0: Vec<u64> = vec![3, 16, 0, 1]; // 16 out of range for bits=4
    assert!(
      LogUpMultiRangeProof::<E>::prove(4, &[&w0], &mut <E as Engine>::TE::new(b"m")).is_err()
    );

    let w0: Vec<u64> = vec![3, 7, 3, 0];
    let w1: Vec<u64> = vec![1, 2];
    let mut tp = <E as Engine>::TE::new(b"logup_multi");
    let (proof, _) = LogUpMultiRangeProof::<E>::prove(4, &[&w0, &w1], &mut tp).unwrap();

    // Tampered witness root breaks the summed identity.
    let mut bad = proof.clone();
    bad.wit_roots[1].0 += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(bad.verify(4, &[2, 1], &mut tv).is_err());

    // Wrong pinned depth rejected.
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(proof.verify(4, &[2, 2], &mut tv).is_err());

    // Honest proof still verifies.
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(proof.verify(4, &[2, 1], &mut tv).is_ok());
  }

  /// Tampering with the shared γ-batched GKR walk — a combined round
  /// polynomial or one tree's layer finals — is rejected.
  #[test]
  fn multi_range_rejects_tampered_gkr() {
    type E = PallasHyraxEngine;
    let w0: Vec<u64> = vec![3, 7, 3, 0, 15, 1, 9, 3];
    let w1: Vec<u64> = vec![1, 2, 0, 0];
    let mut tp = <E as Engine>::TE::new(b"logup_multi");
    let (proof, _) = LogUpMultiRangeProof::<E>::prove(4, &[&w0, &w1], &mut tp).unwrap();

    // Corrupt a combined round polynomial deep in the walk.
    let mut bad = proof.clone();
    let last = bad.gkr.layers.len() - 1;
    bad.gkr.layers[last].round_polys[0][2] += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(bad.verify(4, &[3, 2], &mut tv).is_err());

    // Corrupt one tree's input-layer finals.
    let mut bad = proof.clone();
    bad.gkr.layers[1].finals[0][3] += <E as Engine>::Scalar::ONE;
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(bad.verify(4, &[3, 2], &mut tv).is_err());

    // Honest proof still verifies.
    let mut tv = <E as Engine>::TE::new(b"logup_multi");
    assert!(proof.verify(4, &[3, 2], &mut tv).is_ok());
  }
}
