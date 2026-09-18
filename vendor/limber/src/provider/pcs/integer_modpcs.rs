//! `IntegerModPCS`: the integer Mod-PCS (the paper's IntEval protocol)
//! for the dual-field engines, wrapping a field PCS as the underlying
//! F PCS — Hyrax-over-T256 in curve mode, Brakedown in hash mode (see
//! `commit_backend`). Commits integer-valued polynomials (limb-split
//! into `T`-bounded chunks and range-checked with the shared LogUp-GKR
//! argument) and opens them at `Z_p` points.
//!
//! The PCS soundly bridges `F_q` arithmetic to `Z_p` evaluations via
//! small-prime fingerprinting:
//! the verifier samples `s` random primes `p_i ≈ 2^{log P}` and opens
//! the F-committed polynomial at `r mod p_i` for each. Because each
//! reduced point is small (`< P`), the F arithmetic stays below `q`
//! and faithfully matches the integer arithmetic, letting the verifier
//! check `to_int(F_y^{(i)}) ≡ int_y (mod p_i)`. By CRT, agreement on
//! `s` independent primes implies the integer evaluation is correct
//! with high probability.

use crate::{
  errors::SpartanError,
  polys::eq::EqPolynomial,
  provider::{
    T256DynPrimeEngine, T256HyraxEngine,
    keccak::Keccak256Transcript,
    pcs::{
      commit_backend::{BdBackend, CommitBackend, OpenTarget},
      hyrax_pc::{HyraxBlind, HyraxPCS},
    },
    pt256::t256,
  },
  start_span,
  traits::{
    PrimeFieldExt,
    mod_engine::{ModPCSEngineTrait, SmallValueBlock, SumcheckEngine, SumcheckField},
    pcs::PCSEngineTrait,
    transcript::{ByteTranscript, TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::marker::PhantomData;
use ff::{Field, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Underlying standard PCS: Hyrax over T256.
type Hyrax = HyraxPCS<T256HyraxEngine>;

/// IntEval parameters. Application-level inputs are:
///   - `log T_f`: norm bound on the polynomial `f` being committed.
///   - `log T`:   norm bound on each *limb* of the split polynomial.
///   - `k`:       per-iteration variable count for partial evaluation.
///
/// (Naming matches the paper: `\Bound[f]` and `\Bound` in the LaTeX
/// source render as `T_f` and `T` respectively — see preamble.tex's
/// `\newcommand{\Bound}[1][]{\mathsf{T}_{#1}}`. The `compute_params.py`
/// script uses the same `T` / `log_T` convention.)
///
/// In no-limb-split mode, the polynomial is committed
/// as a single limb, so `T = T_f`. With limb-splitting, `T` is
/// chosen smaller than `T_f` (typically `~32` bits) so each limb fits
/// inside F's characteristic with room for IntEval's intermediate
/// products.
///
/// Module constants: `LAMBDA = 128` (security target), `LOG_Q = 256`
/// (T256's characteristic width). Protocol parameters `(log P, s)` are
/// *derived* from `(log T, k, num_vars)` per the paper's recipe.
///
/// `derive(log_t_f, log_t, k, num_vars)` returns a valid setting;
/// `explicit(...)` lets a caller override `(k, log P, s)` and revalidates.
/// Both go through `validate(num_vars)` which checks the four bounds
/// from §4.4 — Final Eval, Partial Eval Norm, Soundness 1, Soundness 2.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntEvalParams {
  /// Serialized parameter format; old keys must be regenerated.
  pub format_version: u32,
  /// Requested prime-divisibility security, independent of challenge width.
  pub security_bits: usize,
  /// Bit-width of the q-side field's characteristic these parameters
  /// were derived for. `LOG_Q` (=256, t256) via the plain
  /// constructors; other fields use [`IntEvalParams::derive_for_q`].
  /// Every norm/soundness bound checks against this value.
  pub log_q: usize,
  /// Per-iteration variables consumed during partial evaluation.
  pub k: usize,
  /// Bit-width upper bound on the random small primes `p_i ∈ [P/2, P]`.
  pub log_p: usize,
  /// Number of random small primes sampled per evaluation.
  pub s: usize,
  /// Norm bound on each *limb* of the split polynomial, in bits.
  /// In no-limb-split mode this equals `log_t_f`.
  pub log_t: usize,
  /// Norm bound on the committed polynomial `f` itself, in bits.
  pub log_t_f: usize,
  /// Number of limbs per polynomial coefficient: `⌈log_t_f / log_t⌉`.
  /// Setup-fixed and public — both prover and verifier read this from
  /// the params they share. `1` in no-limb-split mode.
  pub numlimb: usize,
  /// Bit-width of the limb index, `⌈log_2 numlimb⌉`. `0` when
  /// `numlimb = 1` (no extra polynomial variables needed).
  pub numlimb_var: usize,
}

/// Security parameter (bits). The protocol targets `2^{-λ}` soundness.
pub const LAMBDA: usize = 128;
/// Width of the outer challenge, used in integer magnitude bounds.
pub const CHALLENGE_BITS: usize = 128;

/// Accepted challenge-soundness target (bits) for Soundness Bound 2
/// (`s·n/|F| ≤ 2^-target`). Deliberately below `LAMBDA`: the system's
/// overall soundness is already bounded by the ~2^-114 fingerprint
/// prime-sampling term, so demanding full 128-bit challenge soundness
/// would over-secure one term while another sits lower. Set to 117 so
/// a ~2^127 field (e.g. M127) passes with challenges drawn from the
/// base field; may be lowered further if a future instantiation calls
/// for it.
pub const LAMBDA_BOUND2: usize = 117;

/// Bit-width of the underlying F's characteristic `q`. Fixed at 256 for
/// T256; future engines with other widths would parameterize this.
pub const LOG_Q: usize = 256;

impl IntEvalParams {
  /// Derive a valid `(log P, s)` for the given `(log T_f, log B, k,
  /// num_vars)`. Picks the largest `log P` satisfying Final Eval +
  /// Partial Eval Norm bounds (the latter using `log T`, the limb
  /// bound), then the smallest `s` satisfying Soundness 1.
  pub fn derive(
    log_t_f: usize,
    log_t: usize,
    k: usize,
    num_vars: usize,
  ) -> Result<Self, SpartanError> {
    Self::derive_for_q(LOG_Q, log_t_f, log_t, k, num_vars)
  }

  /// [`IntEvalParams::derive`] for an arbitrary field-characteristic
  /// width (the norm/soundness bounds all scale with `log_q`).
  pub fn derive_for_q(
    log_q: usize,
    log_t_f: usize,
    log_t: usize,
    k: usize,
    num_vars: usize,
  ) -> Result<Self, SpartanError> {
    Self::derive_for_q_with_target(log_q, log_t_f, log_t, k, num_vars, LAMBDA)
  }

  /// Derive explicit per-check security without changing the challenge width.
  pub fn derive_with_target(
    log_t_f: usize,
    log_t: usize,
    k: usize,
    num_vars: usize,
    security_bits: usize,
  ) -> Result<Self, SpartanError> {
    Self::derive_for_q_with_target(LOG_Q, log_t_f, log_t, k, num_vars, security_bits)
  }

  /// Field-width and security-target aware parameter derivation.
  pub fn derive_for_q_with_target(
    log_q: usize,
    log_t_f: usize,
    log_t: usize,
    k: usize,
    num_vars: usize,
    security_bits: usize,
  ) -> Result<Self, SpartanError> {
    if !(64..=128).contains(&security_bits) {
      return Err(params_overflow(
        "unsupported security target (expected 64..=128)",
      ));
    }
    if k == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "IntEvalParams::derive: k must be positive".to_string(),
      });
    }
    let nl_pre = checked_numlimb(log_t_f, log_t)?;
    let nlv_pre = numlimb_var(nl_pre);
    let num_vars_total = num_vars
      .checked_add(nlv_pre)
      .ok_or_else(|| params_overflow("num_vars + numlimb_var"))?;

    // Find max log_p satisfying Partial Evaluation Norm Bound:
    //   k + k·log_p + max(log_t, log_p) < log_q   (uses limb bound T)
    let mut log_p = 0usize;
    for lp in 1..log_q {
      let partial = k
        .checked_mul(lp)
        .and_then(|x| x.checked_add(k))
        .and_then(|x| x.checked_add(log_t.max(lp)))
        .ok_or_else(|| params_overflow("k + k*log_p + max(log_t, log_p)"))?;
      if partial < log_q {
        log_p = lp;
      } else {
        break;
      }
    }
    if log_p <= 1 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::derive: no log P > 1 satisfies Partial Eval Norm \
           for k={k}, log T={log_t}, log q={log_q}"
        ),
      });
    }

    // Smallest s satisfying the prime-divisibility soundness bound
    //   (log_{P/2}(y) / (π(P) − π(P/2)))^s ≤ 2^{−security_bits},
    // where log2(y) = n + CHALLENGE_BITS·n + log_t bounds the integer difference between a
    // false and the true partial evaluation, and log_{P/2}(y) upper-bounds how
    // many primes ≥ P/2 can divide it. `bits_per_prime` is the soundness each
    // random small prime in (P/2, P] contributes; the prime count π(P)−π(P/2)
    // is lower-bounded (Dusart/Rosser–Schoenfeld) so s stays sound. Replaces
    // the older crude `(32 λ n / P)` union bound, which over-provisioned s.
    let bits_per_prime = soundness_bits_per_prime(log_p, num_vars_total, log_t);
    if bits_per_prime <= 0.0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::derive: prime-divisibility soundness gives ≤ 0 bits per \
           prime for k={k}, num_vars={num_vars}, derived log_p={log_p}"
        ),
      });
    }
    let s = (security_bits as f64 / bits_per_prime).ceil() as usize;

    let nl = nl_pre;
    let p = Self {
      format_version: 2,
      security_bits,
      log_q,
      k,
      log_p,
      s,
      log_t,
      log_t_f,
      numlimb: nl,
      numlimb_var: numlimb_var(nl),
    };
    p.validate(num_vars)?;
    Ok(p)
  }

  /// Conservative bits supplied by the sampled small primes at this shape.
  pub fn prime_security_bits(&self, num_vars: usize) -> f64 {
    self.s as f64 * soundness_bits_per_prime(self.log_p, num_vars + self.numlimb_var, self.log_t)
  }

  /// No-limb-split convenience: derive params with `log T = log T_f`
  /// (single-limb regime).
  pub fn derive_no_limb_split(
    log_t_f: usize,
    k: usize,
    num_vars: usize,
  ) -> Result<Self, SpartanError> {
    Self::derive(log_t_f, log_t_f, k, num_vars)
  }

  /// Use explicit `(k, log P, s, log T, log T_f)`. Validates against
  /// `num_vars` so a caller-tuned configuration can't bypass the bound
  /// checks. Errors if any of the four bounds is violated.
  pub fn explicit(
    k: usize,
    log_p: usize,
    s: usize,
    log_t: usize,
    log_t_f: usize,
    num_vars: usize,
  ) -> Result<Self, SpartanError> {
    let nl = checked_numlimb(log_t_f, log_t)?;
    let p = Self {
      format_version: 2,
      security_bits: LAMBDA,
      log_q: LOG_Q,
      k,
      log_p,
      s,
      log_t,
      log_t_f,
      numlimb: nl,
      numlimb_var: numlimb_var(nl),
    };
    p.validate(num_vars)?;
    Ok(p)
  }

  /// Check all four bounds from §4.4. Each is evaluated in log-space to
  /// avoid overflow; the comparisons match the paper's inequalities
  /// after taking `log_2` of both sides. `num_vars` is the *original*
  /// polynomial variable count — the limb-split polynomial has
  /// `num_vars + numlimb_var` variables, and that's what enters the
  /// soundness bounds.
  pub fn validate(&self, num_vars: usize) -> Result<(), SpartanError> {
    if self.format_version != 2 || !(64..=128).contains(&self.security_bits) {
      return Err(SpartanError::InvalidInputLength {
        reason: "unsupported IntEval parameter version or security target".into(),
      });
    }
    // `k = 0` would divide by zero in the iteration-layer formulas and
    // makes no protocol sense (zero variables consumed per iteration).
    if self.k == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "IntEvalParams: k must be positive".to_string(),
      });
    }
    let num_vars_total = num_vars
      .checked_add(self.numlimb_var)
      .ok_or_else(|| params_overflow("num_vars + numlimb_var"))?;

    // Limb-decomposition self-consistency: `numlimb` and `numlimb_var`
    // must match the formulas implied by `(log_t, log_t_f)`. Catches
    // hand-rolled `IntEvalParams { ... }` literals that get the
    // relation wrong (and rejects `log_t = 0` before the asserting
    // `numlimb` helper would panic).
    let expected_nl = checked_numlimb(self.log_t_f, self.log_t)?;
    if self.numlimb != expected_nl {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams: numlimb = {} does not match ⌈log_T_f / log_T⌉ = ⌈{}/{}⌉ = {}",
          self.numlimb, self.log_t_f, self.log_t, expected_nl
        ),
      });
    }
    let expected_nlv = numlimb_var(self.numlimb);
    if self.numlimb_var != expected_nlv {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams: numlimb_var = {} does not match ⌈log_2 numlimb⌉ = ⌈log_2 {}⌉ = {}",
          self.numlimb_var, self.numlimb, expected_nlv
        ),
      });
    }

    // Final Evaluation Bound: 2^k * P^(k+1) < q
    //   log: k + (k+1)·log_p < log_q
    let final_eval_lhs = self
      .k
      .checked_add(1)
      .and_then(|x| x.checked_mul(self.log_p))
      .and_then(|x| x.checked_add(self.k))
      .ok_or_else(|| params_overflow("k + (k+1)*log_p"))?;
    if final_eval_lhs >= self.log_q {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Final Evaluation Bound violated: k + (k+1)·log_p = {} >= log_q = {}",
          final_eval_lhs, self.log_q
        ),
      });
    }

    // Partial Evaluation Norm Bound: 2^k · P^k · max(T, P) <= (q-P)/2
    //   log (approximate, dropping the -P-1 below q): k + k·log_p + max(log_t, log_p) < log_q
    // Uses `log_t` (the *limb* bound), not `log_t_f`, since IntEval
    // operates on the (possibly limb-split) polynomial.
    let partial_norm_lhs = self
      .k
      .checked_mul(self.log_p)
      .and_then(|x| x.checked_add(self.k))
      .and_then(|x| x.checked_add(self.log_t.max(self.log_p)))
      .ok_or_else(|| params_overflow("k + k*log_p + max(log_t, log_p)"))?;
    if partial_norm_lhs >= self.log_q {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Partial Evaluation Norm Bound violated: k + k·log_p + max(log_B, log_p) = {} >= log_q = {}",
          partial_norm_lhs, self.log_q
        ),
      });
    }

    // Sanity: `log_t > log_t_f` doesn't make sense — the limb bound
    // can't exceed the polynomial bound.
    if self.log_t > self.log_t_f {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams: log_t ({}) must not exceed log_t_f ({})",
          self.log_t, self.log_t_f
        ),
      });
    }

    // Soundness Bound 1 (prime divisibility): (log_P(y) / (π(P) − π(P/2)))^s ≤ 2^{−λ}
    //   <=>  s · bits_per_prime ≥ λ,  bits_per_prime = log2(π(P)−π(P/2)) − log2(log_P y).
    let bits_per_prime = soundness_bits_per_prime(self.log_p, num_vars_total, self.log_t);
    if bits_per_prime <= 0.0 || (self.s as f64) * bits_per_prime < self.security_bits as f64 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Soundness Bound 1 violated: s·bits_per_prime = {:.2} < λ = {}",
          (self.s as f64) * bits_per_prime,
          self.security_bits
        ),
      });
    }

    // Soundness Bound 2: s · n / |F| <= 2^{-target}, with the accepted
    // target `LAMBDA_BOUND2` (117, not λ = 128 — see its doc comment).
    //   log: log(s·n) - log_q <= -target
    //   <=>  log_q >= target + log(s·n)
    let log_sn = ceil_log2(
      self
        .s
        .checked_mul(num_vars)
        .ok_or_else(|| params_overflow("s * num_vars"))?
        .max(1),
    );
    if self.log_q
      < LAMBDA_BOUND2
        .checked_add(log_sn)
        .ok_or_else(|| params_overflow("target + log(s*n)"))?
    {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEval Soundness Bound 2 violated: log_q = {} < target + log(s·n) = {}",
          self.log_q,
          LAMBDA_BOUND2 + log_sn
        ),
      });
    }

    Ok(())
  }

  /// Coarse prover-cost estimate for these params on a `num_vars`-variable
  /// polynomial, in "committed scalar" units (one unit ≈ the amortized
  /// per-scalar cost of a Hyrax commit plus its share of the batched
  /// open). This is a *ranking heuristic* for `derive_optimized`, not a
  /// performance guarantee — soundness is enforced independently by
  /// `validate`, so a misranked candidate costs time, never security.
  ///
  /// Terms, mirroring the prove path (committed-chunk representation:
  /// every oracle is committed ONCE, as 16-bit chunks):
  ///   - `f_limb`'s chunk commit/open: `2^{n_tot} · ⌈log T / 16⌉`;
  ///   - per iteration layer `j` (`m_j = 2^{n_tot − j·k}` slots): the
  ///     `a_j`/`b_j` chunk commitments (`s` chains, widths `log P + 1`
  ///     and `log q − log P + 1`) and the integer partial-evaluation
  ///     work (`s` chains × `m_j · 2^k` bigint mult-adds of
  ///     `⌈(k·(log P + 1) + log T)/64⌉`-word operands);
  ///   - eq-tensor weight assembly for the `s` per-chain claims of the
  ///     final batched opens: `s · 2^{n_tot}` field ops.
  pub fn estimated_prover_cost(&self, num_vars: usize) -> f64 {
    /// One 256-bit field op, in committed-scalar units.
    const FIELD_OP: f64 = 0.05;
    /// One 64-bit bigint word op, in committed-scalar units.
    const BIGINT_WORD: f64 = 0.02;

    let two = |e: usize| (e as f64).exp2();
    let n_tot = num_vars + self.numlimb_var;
    let s = self.s as f64;

    // f_limb chunk commit/open.
    let f_chunks = self.log_t.div_ceil(CHUNK_BITS) as f64;
    let mut cost = two(n_tot) * f_chunks;

    // Iteration layers (none when n_tot <= k).
    let t_layers = n_tot.saturating_sub(self.k).div_ceil(self.k);
    let ab_chunks = ((self.log_p + 1).div_ceil(CHUNK_BITS)
      + (self.log_q - self.log_p + 1).div_ceil(CHUNK_BITS)) as f64;
    let words = (self.k * (self.log_p + 1) + self.log_t).div_ceil(64) as f64;
    for j in 1..=t_layers {
      let m = two(n_tot - j * self.k);
      // Per-layer a/b chunk commitments.
      cost += m * s * ab_chunks;
      // Integer partial evaluation: input size m · 2^k per chain.
      cost += BIGINT_WORD * s * m * two(self.k) * words;
    }

    // Per-claim eq-tensor weights in the batched opens.
    cost += FIELD_OP * s * two(n_tot);
    cost
  }

  /// Derive params *optimized for the given input length*: search over
  /// the per-iteration variable count `k` and the limb bound `log T`,
  /// deriving the dependent `(log P, s)` for each candidate via
  /// [`Self::derive`], and return the candidate minimizing
  /// [`Self::estimated_prover_cost`] (ties broken by smaller `s`, i.e.
  /// cheaper verifier and smaller proofs, then smaller `numlimb`).
  ///
  /// The search space is tiny — `k ∈ [1, num_vars + numlimb_var]` per
  /// limb candidate, and limb candidates halve `log T` from `log T_f`
  /// down to the 16-bit range-check chunk width — so this is cheap
  /// enough to run at every setup. Every candidate passes through
  /// `derive`'s `validate`, so the optimizer can only affect
  /// performance, never soundness.
  pub fn derive_optimized(log_t_f: usize, num_vars: usize) -> Result<Self, SpartanError> {
    // Limb candidates: numlimb = 2^v, log T = ⌈log T_f / 2^v⌉. Splitting
    // below one range-check chunk (16 bits) cannot reduce the per-value
    // chunk count further, so stop there.
    let mut log_t_candidates: Vec<usize> = Vec::new();
    for v in 0..=24usize {
      let log_t = log_t_f.div_ceil(1usize << v).max(1);
      if log_t_candidates.last() != Some(&log_t) {
        log_t_candidates.push(log_t);
      }
      if log_t <= CHUNK_BITS {
        break;
      }
    }

    let mut best: Option<(f64, Self)> = None;
    let mut last_err: Option<SpartanError> = None;
    for &log_t in &log_t_candidates {
      let nlv = numlimb_var(checked_numlimb(log_t_f, log_t)?);
      // k > n_tot behaves like k = n_tot (zero iterations) but only
      // tightens the norm bounds, so cap the search there.
      let k_max = num_vars
        .checked_add(nlv)
        .ok_or_else(|| params_overflow("num_vars + numlimb_var"))?
        .max(1);
      for k in 1..=k_max {
        match Self::derive(log_t_f, log_t, k, num_vars) {
          Ok(p) => {
            let cost = p.estimated_prover_cost(num_vars);
            let better = match &best {
              None => true,
              Some((bc, bp)) => {
                cost < *bc || (cost == *bc && (p.s, p.numlimb) < (bp.s, bp.numlimb))
              }
            };
            if better {
              best = Some((cost, p));
            }
          }
          Err(e) => last_err = Some(e),
        }
      }
    }

    best.map(|(_, p)| p).ok_or_else(|| {
      last_err.unwrap_or(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::derive_optimized: no valid (k, log T) candidate for \
           log_t_f={log_t_f}, num_vars={num_vars}"
        ),
      })
    })
  }

  /// These params committing values of width `log_t_f` bits (a *segment*
  /// of a width-grouped witness) instead of `self.log_t_f`, keeping the
  /// shared `(log_t, log_p, log_q, s, k)` bounds so every segment reduces
  /// against the SAME range check and combined opening. Only `numlimb`
  /// and `numlimb_var` change. `log_t_f` must be a positive multiple of
  /// `log_t` no larger than `self.log_t_f`; the norm and soundness bounds
  /// then hold a fortiori — narrower values with the same `(log_t, k)`
  /// and at least as many CRT primes (derived for the wider `num_vars`).
  pub fn narrowed(&self, log_t_f: usize) -> Result<Self, SpartanError> {
    if log_t_f == 0 || !log_t_f.is_multiple_of(self.log_t) || log_t_f > self.log_t_f {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntEvalParams::narrowed: log_t_f={log_t_f} must be a positive multiple of \
           log_t={} and <= self.log_t_f={}",
          self.log_t, self.log_t_f
        ),
      });
    }
    let nl = numlimb(log_t_f, self.log_t);
    Ok(IntEvalParams {
      log_t_f,
      numlimb: nl,
      numlimb_var: numlimb_var(nl),
      ..self.clone()
    })
  }
}

/// The error every checked `IntEvalParams`/chunk-layout arithmetic path
/// maps overflow to: malformed public parameters must return
/// `SpartanError`, never panic, truncate, or overflow first.
fn params_overflow(what: &str) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: format!("IntEvalParams: {what} overflows usize"),
  }
}

/// Checked limb-count helper: rejects `log_t = 0` (the asserting
/// [`numlimb`] would panic) and overflow in the ceiling division.
fn checked_numlimb(log_t_f: usize, log_t: usize) -> Result<usize, SpartanError> {
  if log_t == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "IntEvalParams: log_t must be positive".to_string(),
    });
  }
  log_t_f
    .checked_add(log_t - 1)
    .map(|x| (x / log_t).max(1))
    .ok_or_else(|| params_overflow("ceil(log_t_f / log_t)"))
}

/// Ceiling `log_2`. `ceil_log2(0)` returns 0 (callers guard with `.max(1)`).
fn ceil_log2(x: usize) -> usize {
  if x <= 1 {
    return 0;
  }
  (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// Strict lower bound on `π(2^log2_x)` (the prime-counting function). Uses
/// Dusart's (2010) `π(x) ≥ (x/ln x)(1 + 1/ln x)` for `x ≥ 599`, and the
/// Rosser–Schoenfeld `π(x) > x/ln x` (valid `x ≥ 17`) below that. Returns a
/// lower bound so downstream prime-count soundness estimates stay conservative.
fn pi_lower_2pow(log2_x: usize) -> f64 {
  let x = (log2_x as f64).exp2();
  let lnx = (log2_x as f64) * core::f64::consts::LN_2;
  if x >= 599.0 {
    (x / lnx) * (1.0 + 1.0 / lnx)
  } else {
    x / lnx
  }
}

/// Upper bound on `π(2^log2_x)` via Dusart's `π(x) ≤ (x/ln x)(1 + 1.2762/ln x)`
/// (valid `x ≥ 2`).
fn pi_upper_2pow(log2_x: usize) -> f64 {
  let x = (log2_x as f64).exp2();
  let lnx = (log2_x as f64) * core::f64::consts::LN_2;
  (x / lnx) * (1.0 + 1.2762 / lnx)
}

/// `log2` of a lower bound on the number of primes in `(P/2, P]`, `P = 2^log_p`.
fn log2_primes_in_top_half(log_p: usize) -> f64 {
  let count = (pi_lower_2pow(log_p) - pi_upper_2pow(log_p.saturating_sub(1))).max(1.0);
  count.log2()
}

/// Soundness (in bits) each random small prime `p ∈ (P/2, P]`, `P = 2^log_p`,
/// contributes to the IntEval CRT fingerprint:
///   `log2(π(P) − π(P/2)) − log2(log_{P/2}(y))`,
/// with `log2(y) = n + CHALLENGE_BITS·n + log_t` the bound on the integer difference between
/// a false and the true partial evaluation, and `log_{P/2}(y)` an upper bound on
/// how many primes `≥ P/2` can divide it. `n` is the limb-split polynomial's
/// variable count. A larger value ⇒ fewer primes `s` needed. Primes below
/// `2^5` are too sparse for the bounds, so they return a rejecting value.
fn soundness_bits_per_prime(log_p: usize, n: usize, log_t: usize) -> f64 {
  if log_p < 5 {
    return -1.0;
  }
  let log2_y = (n as f64) * (1.0 + CHALLENGE_BITS as f64) + (log_t as f64);
  let log_p_y = (log2_y / ((log_p - 1) as f64)).max(1.0);
  log2_primes_in_top_half(log_p) - log_p_y.log2()
}

/// Mod-PCS commitment key wraps Hyrax's plus the IntEval parameters.
///
/// `eval` is a size-1 Hyrax key the IntEval protocol uses internally to
/// form the per-opening eval commitment `G^{f_y}` (which the verifier
/// reconstructs locally). It lives inside the Mod-PCS key — not on the
/// universal `ModPCSEngineTrait` surface — so the trait stays
/// PCS-agnostic (a hash/FRI Mod-PCS would carry no such key).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerModCommitmentKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) eval: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) params: IntEvalParams,
  /// Maximum committable polynomial length: the capacity the inner Hyrax
  /// key (and every `f_chunk_len` the key implies) was validated for.
  pub(crate) max_n: usize,
}

impl IntegerModCommitmentKey {
  /// The shared fallible constructor (see [`validate_key_capacity`]):
  /// every Hyrax-side integer commitment key is built through here, so a
  /// stored key always carries a validated `(params, max_n)` pair.
  fn new_checked(
    inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
    eval: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
    params: IntEvalParams,
    max_n: usize,
  ) -> Result<Self, SpartanError> {
    validate_key_capacity(&params, max_n)?;
    Ok(Self {
      inner,
      eval,
      params,
      max_n,
    })
  }
}

/// Verifier key wraps Hyrax's plus the IntEval parameters.
///
/// `eval` mirrors the commitment key's size-1 eval key: the verifier
/// reconstructs the eval commitment `G^{f_y}` locally during opening
/// verification, so it needs the same generators. Kept inside the key
/// rather than on the trait surface (PCS-agnostic).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerModVerifierKey {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  pub(crate) eval: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  pub(crate) params: IntEvalParams,
}

/// Commitment is the underlying Hyrax commitment to the polynomial's
/// base-2^16 CHUNK decomposition (limb-split, then each limb split into
/// `chunk_stride(log_t)` 16-bit slots — the same layout the range
/// check's F batch consumes, so no separate chunk commitment is ever
/// made). Limb evaluations fold to chunk evaluations at the public
/// `chunk_fold_point`. The IntEval protocol runs entirely at eval
/// time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegerModCommitment {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
}

impl TranscriptReprTrait for IntegerModCommitment {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.inner.to_transcript_bytes()
  }
}

/// Blind delegates to Hyrax's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegerModBlind {
  pub(crate) inner: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
}

/// One per-small-prime opening: the F-side evaluation `F_y^(i)` and the
/// Hyrax evaluation argument. The eval commitment `comm_eval = G^{f_y}`
/// is deterministic in `f_y` (zero-blind), so the verifier reconstructs
/// it locally — no `blind_eval` shipped.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallPrimeOpening {
  /// F_y^(i) = f_F(r mod p_i), the F-evaluation at the small-prime-
  /// reduced point. Sent in the clear; verifier checks it for the
  /// CRT congruence `to_int(F_y^(i)) ≡ int_v' (mod p_i)`.
  pub f_y: t256::Scalar,
  /// Hyrax evaluation argument for the opening at `r mod p_i`.
  pub hyrax_arg: <Hyrax as PCSEngineTrait<T256HyraxEngine>>::EvaluationArgument,
}

/// One iteration's identity-check evaluation claims. The commitments
/// live in the per-layer, per-role chunk polynomials
/// (`IntEvalArgument::ab_comms`); every claim here folds through
/// `chunk_fold_point` and is discharged by the final batched opens.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned")]
pub struct IterationOracles<F = t256::Scalar> {
  /// Claimed `a_{j-1}(γ_ext)` where `γ_ext = (γ[0..n-jk], r^(i)[n-jk..n-(j-1)k])`.
  /// A claim on the input commitment for `j=1`, else on layer `j-1`'s
  /// `a` chunk commitment at `(bits(chain), γ_ext, x_*)`.
  pub a_prev_eval: F,
  /// Claimed `a_j(γ[0..n-jk])` — a claim on layer `j`'s `a` chunk
  /// commitment at `(bits(chain), γ_prefix, x_*)`.
  pub a_curr_eval: F,
  /// Claimed `b_j(γ[0..n-jk])` — same shape on the `b` chunk commitment.
  pub b_curr_eval: F,
}

/// Per-prime chain: `t = ⌈(n-k)/k⌉` iterations plus the claimed
/// final-remainder evaluation `a_t(r^(i)[0..n-tk])` (a claim on the input
/// commitment for `t = 0`, else on the last layer's `a` chunk
/// commitment).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned")]
pub struct ChainData<F = t256::Scalar> {
  /// Per-iteration identity-check evaluation claims.
  pub iterations: Vec<IterationOracles<F>>,
  /// Claimed `a_t(r^(i)[0..n-tk])`, used by the CRT check.
  pub final_eval: F,
}

/// Evaluation argument: the prover-sent integer evaluation `int_v'`,
/// the reduction-sumcheck round polynomials, and one
/// per-prime chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IntEvalArgument<B: CommitBackend> {
  /// Per-round compressed univariate polynomial of the reduction
  /// sumcheck `sum_k limb(k) · f_limb(int_r, k) ≡_p int_y`. Each inner
  /// vector is the round poly's coefficients excluding the linear term,
  /// stored as `BigUint` (canonical representatives mod `p`) so the
  /// `IntEvalArgument` stays Serde-friendly without dragging
  /// `SumcheckProof<T256DynPrimeEngine>` through serde plumbing.
  /// `numlimb_var` entries total; empty when `numlimb_var = 0`
  /// (no-limb-split mode, the reduction sumcheck is degenerate).
  pub reduction_round_polys: Vec<Vec<BigUint>>,
  /// `int_v' = f_limb(int_r, int_r_k)` as a signed integer. Negative
  /// values come from `(1 - r_i)` factors in the multilinear chi. For
  /// `numlimb_var = 0` this equals the integer evaluation of `f` at
  /// `int_r`.
  pub int_v_prime: BigInt,
  /// One per small prime sampled from the transcript. Length matches
  /// `params.s`.
  pub chains: Vec<ChainData<B::Scalar>>,
  /// Two chunk commitments per iteration layer `j ∈ [1, t]` (`a_j` then
  /// `b_j`), each committing ALL chains' shifted values in the range-
  /// check chunk layout `((chain·m + x)·stride + c)` with chains padded
  /// to the next power of two. The layer commitment IS its range-check
  /// chunk oracle; layer evaluations fold through [`chunk_fold_point`].
  /// Empty when `t = 0`.
  pub(crate) ab_comms: Vec<B::Comm>,
  /// ONE shared LogUp-GKR range check covering all `(bound, size)`
  /// batch groups. Canonical batch order is `f_limb`, then for each
  /// iteration `j = 1..=t` the `a_j` batch (all `s` chains) and the
  /// `b_j` batch — `1 + 2t` batches (just `f_limb` when `t = 0`), all
  /// sharing one multiplicity table and one table-side GKR. EVERY
  /// batch's chunk polynomial is its target's own commitment
  /// (committed-chunk representation), so the range check carries only
  /// the multiplicity commitment and the GKR itself — no per-batch
  /// chunk commitments or reconstruction sumchecks. See
  /// [`prove_shared_range_check`].
  pub(crate) range_check: SharedRangeCheck<B>,
  /// ONE combined opening discharging every evaluation claim made
  /// anywhere in the protocol, over all commitments in canonical order:
  /// the input `f`, the stacked layers `ab_1..ab_t`, the `1+2t` chunk
  /// commitments, and the multiplicity table.
  pub(crate) combined_open: CombinedBatchOpen<B>,
}

/// Per-polynomial portion of a batched [`IntEvalBatchArgument`]: the same
/// reduction-sumcheck round polynomials, integer evaluation, chains, and
/// stacked-layer commitments a standalone [`IntEvalArgument`] carries,
/// minus the range check and combined opening (which are shared across
/// the whole batch).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IntEvalPerPolyArgument<B: CommitBackend> {
  /// See [`IntEvalArgument::reduction_round_polys`].
  pub reduction_round_polys: Vec<Vec<BigUint>>,
  /// See [`IntEvalArgument::int_v_prime`].
  pub int_v_prime: BigInt,
  /// See [`IntEvalArgument::chains`].
  pub chains: Vec<ChainData<B::Scalar>>,
  /// See [`IntEvalArgument::ab_comms`].
  pub(crate) ab_comms: Vec<B::Comm>,
}

/// Evaluation argument for a *batched* Mod-PCS open of several integer
/// polynomials at (possibly distinct) points. Each polynomial's reduction
/// sumcheck and per-prime chains run independently
/// ([`IntEvalPerPolyArgument`]), but ALL of their range-check batches are
/// discharged by ONE shared LogUp-GKR range check and ALL of their
/// evaluation claims by ONE combined inner-product opening — so the fixed
/// per-open costs (the `2^16`-table-side GKR, the merged IPA) are paid
/// once for the whole batch instead of once per polynomial.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IntEvalBatchArgument<B: CommitBackend> {
  /// One entry per opened polynomial, in the caller's input order.
  pub per_poly: Vec<IntEvalPerPolyArgument<B>>,
  /// ONE shared LogUp-GKR range check covering every batch of every poly.
  pub(crate) range_check: SharedRangeCheck<B>,
  /// ONE combined opening discharging every evaluation claim of every poly.
  pub(crate) combined_open: CombinedBatchOpen<B>,
  /// Per polynomial, per declared [`SmallValueBlock`]: the block's own
  /// MLE evaluation `e2` at the transcript point (see
  /// [`small_block_claims`]). Empty for polynomials without blocks.
  pub(crate) small_block_evals: Vec<Vec<B::Scalar>>,
}

impl<B: CommitBackend> IntEvalBatchArgument<B>
where
  IntEvalPerPolyArgument<B>: Serialize,
  SharedRangeCheck<B>: Serialize,
  CombinedBatchOpen<B>: Serialize,
{
  /// Number of dyadic chunk blocks admitted to the shared LogUp multiset.
  /// Benchmark/reporting accessor; the bitmap itself remains protocol-private.
  pub fn active_logup_block_count(&self) -> usize {
    self
      .range_check
      .active_blocks
      .iter()
      .flatten()
      .filter(|&&active| active)
      .count()
  }

  /// Total dyadic chunk blocks represented by the shared range-check bitmap.
  pub fn logup_block_count(&self) -> usize {
    self.range_check.active_blocks.iter().map(Vec::len).sum()
  }

  /// Serialized size of each top-level component, for proof-size
  /// accounting: `(per_poly, range_check, combined_open)` in bytes.
  pub fn component_sizes(&self) -> (usize, usize, usize) {
    let sz_pp: u64 = self
      .per_poly
      .iter()
      .map(|p| bincode::serialized_size(p).unwrap_or(0))
      .sum();
    let sz_rc = bincode::serialized_size(&self.range_check).unwrap_or(0);
    let sz_co = bincode::serialized_size(&self.combined_open).unwrap_or(0);
    (sz_pp as usize, sz_rc as usize, sz_co as usize)
  }
}

/// ONE combined multi-point opening for ALL commitments of a Mod-PCS
/// open. Per commitment, its claims are RLC-combined (`Σ_i λ^i·y_i =
/// Σ_x f(x)·W(x)`); the per-commitment degree-2 sumchecks then run
/// INTERLEAVED with shared, tail-aligned challenges (every round all
/// active instances absorb their round polynomials before the one shared
/// challenge is squeezed — shorter instances join late). Each
/// commitment's final point is therefore the last `n_j` shared
/// challenges, so all final points share the same column coordinates and
/// the openings collapse into a single μ-RLC'd inner-product argument
/// ([`HyraxPCS::prove_same_column_batch`]). Sub-column-width commitments
/// (test sizes) fall back to individual opens.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CombinedBatchOpen<B: crate::provider::pcs::commit_backend::CommitBackend> {
  /// Per-commitment compressed sumcheck round polynomials, tail-aligned
  /// to the shared challenge vector (entry `j` has `n_j` rounds).
  pub(crate) round_polys: Vec<Vec<crate::polys::univariate::CompressedUniPoly<B::Scalar>>>,
  /// Per-commitment claimed final evaluation `f_j(r_j)`.
  pub(crate) final_evals: Vec<B::Scalar>,
  /// The backend's discharge of the per-commitment single-point
  /// openings (Hyrax: μ-merged same-column IPA + small-size fallback
  /// opens; Brakedown: per-target tensor-IOPP arguments).
  pub(crate) backend: B::BatchOpenArg,
}

/// Hyrax's final-opening argument: the μ-merged same-column opening over
/// every column-width-or-larger commitment plus individual fallback
/// opens for the sub-column-width ones (test sizes), in canonical order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyraxBatchOpenArg {
  /// `None` iff no commitment reaches the column width.
  pub(crate) merged: Option<SmallPrimeOpening>,
  pub(crate) small_opens: Vec<SmallPrimeOpening>,
}

/// Chunk width (bits) for the LogUp range checks: values are decomposed
/// into base-`2^16` chunks, each looked up against the `[0, 2^16)` table.
pub(crate) const CHUNK_BITS: usize = 16;

/// Per-batch data of the shared range check: the batch's chunk
/// commitment, the claimed evaluations its checks rest on (discharged by
/// the final batched opens), and the value-reconstruction sumcheck tying
/// chunks to the batch's value polynomials.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RangeCheckBatchData<B: CommitBackend> {
  /// Stacked chunk-polynomial commitment (entries in `[0, 2^16)`),
  /// laid out `((p·n_values + within)·stride + c)`.
  pub(crate) chunk_comm: B::Comm,
  /// Claimed `V(r_v) = Σ_p eq(r_v_poly, p)·value_p(r_v_within)` — for an
  /// `a/b` batch this equals the stacked layer MLE at `(role, r_v)`, for
  /// the `f_limb` batch it is `f(r_v_within)`; discharged by the
  /// corresponding commitment's batched open.
  pub(crate) value_eval: B::Scalar,
  /// Claimed `chunk(r_v ++ r_b)` — the reconstruction sumcheck's final
  /// chunk evaluation, discharged by the chunk commitment's batched open.
  pub(crate) reconstr_eval: B::Scalar,
  /// Value-reconstruction sumcheck (`Σ_c 2^(16c)·chunk(r_v, c) =
  /// value(r_v)`), over the Hyrax base field.
  pub(crate) value_reconstr_sumcheck: crate::sumcheck::SumcheckProof<B::SE>,
}

/// ONE shared LogUp-GKR range check covering all `(bound, size)` batch
/// groups of a Mod-PCS opening. Every batch's 16-bit chunks (and, for
/// non-16-aligned bounds, its shifted top chunks) are witness trees of a
/// single [`crate::logup_gkr::LogUpMultiRangeProof`] against one
/// `2^16`-entry multiplicity table — the table-side GKR and the
/// multiplicity commitment are paid once per opening instead of once per
/// batch. Witness-tree order: all batches' chunk trees in canonical
/// batch order, then the shifted-top trees of the non-aligned batches in
/// the same order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SharedRangeCheck<B: CommitBackend> {
  /// Commitment to the shared `2^16`-entry multiplicity table.
  pub(crate) mult_comm: B::Comm,
  /// The multi-witness LogUp-GKR membership argument.
  pub(crate) logup: crate::logup_gkr::LogUpMultiRangeProof<B::SE>,
  /// Per-batch commitments, openings, and reconstruction sumchecks for
  /// batches that commit a FRESH chunk polynomial, in canonical batch
  /// order. In the current protocol EVERY batch is precommitted (its
  /// chunk polynomial is the target's own commitment, the chunk→value
  /// relation definitional via [`chunk_fold_point`]), so this is always
  /// empty; the machinery remains for non-precommitted batch kinds.
  pub(crate) batches: Vec<RangeCheckBatchData<B>>,
  /// Per batch: which dyadic chunk blocks (`2^RC_BLOCK_LOG` slots each)
  /// enter the LogUp multiset. Inactive blocks are proven all-zero by
  /// fresh random-point opening claims instead. Untrusted prover advice:
  /// a nonzero block marked inactive fails its zero claim w.h.p.; a zero
  /// block marked active merely wastes prover work.
  pub(crate) active_blocks: Vec<Vec<bool>>,
}

/// `BigUint → t256::Scalar` via 64-byte wide reduction. Value-preserving
/// for inputs below the scalar field, otherwise reduces uniformly. The
/// shared range check run at open time is what turns the resulting
/// commitment into a *sound* commitment to a bounded integer.
fn biguint_to_scalar<F: PrimeFieldExt>(v: &BigUint) -> F {
  // Fast path: values that fit one u64 digit (e.g. every limb at
  // T ≤ 2^64) skip the 512-bit uniform-reduction path.
  if v.bits() <= 64 {
    return F::from(v.iter_u64_digits().next().unwrap_or(0));
  }
  let mut bytes = v.to_bytes_le();
  bytes.resize(64, 0);
  F::from_uniform(&bytes)
}

/// `t256::Scalar → BigUint` via the canonical (non-Montgomery) integer
/// representation. Inverse of `biguint_to_scalar` for inputs that fit
/// in the scalar field.
fn scalar_to_biguint<F: ff::PrimeField>(s: &F) -> BigUint {
  BigUint::from_bytes_le(s.to_repr().as_ref())
}

/// `t256::Scalar → BigInt` in *balanced* representation. The canonical
/// integer in `[0, q)` is reinterpreted as a signed integer in
/// `[-q/2, q/2)`: values `≥ ⌈q/2⌉` become `value - q`. Used by the
/// IntEval CRT check: when the integer evaluation is negative, the F
/// arithmetic produces a result near `q` (because `(1 - r_i)` wraps to
/// `q + 1 - r_i`), so the verifier must lift back to a signed value.
fn scalar_to_balanced_int<F: ff::PrimeField>(s: &F) -> BigInt {
  let v = scalar_to_biguint(s);
  let q = field_q::<F>();
  let half = &q >> 1;
  if v > half {
    BigInt::from(v) - BigInt::from(q)
  } else {
    BigInt::from(v)
  }
}

/// The T256 scalar field's characteristic `q` as a `BigUint`. Computed
/// once via `(q - 1) + 1` from `-Scalar::ONE`'s representation; cheap
/// enough to recompute per call since it's just byte arithmetic.
fn field_q<F: ff::PrimeField>() -> BigUint {
  // `q` is a compile-time constant of the curve; compute it once and
  // hand out clones (this is called per `shift_b`, i.e. per range check).
  // Per-call compute: generic statics are unavailable and this is
  // cheap byte arithmetic on a short repr.
  {
    let q_minus_1 = (-F::ONE).to_repr();
    let mut bytes = q_minus_1.as_ref().to_vec();
    let mut carry = 1u8;
    for b in bytes.iter_mut() {
      let (v, c) = b.overflowing_add(carry);
      *b = v;
      carry = u8::from(c);
    }
    debug_assert_eq!(carry, 0);
    BigUint::from_bytes_le(&bytes)
  }
}

/// Canonical integer in `[0, p)` from a `DynPrime<2>` value.
fn dyn_to_biguint(d: &crate::dyn_prime::DynPrime<2>) -> BigUint {
  BigUint::from_bytes_le(&d.to_le_bytes())
}

/// Extract `p` (the dynamic prime) from a non-empty point. Uses the
/// modulus carried by the first component's `FixedMontyParams<2>`.
fn extract_p(point: &[crate::dyn_prime::DynPrime<2>]) -> Result<BigUint, SpartanError> {
  let p0 = point.first().ok_or(SpartanError::InternalError {
    reason: "IntegerModPCS: point must have at least one component to extract p".to_string(),
  })?;
  let modulus = p0.params().modulus();
  // `modulus` is `&Odd<Uint<4>>`; `.as_ref()` gives the inner `Uint<4>`.
  let bytes = modulus.as_ref().to_le_bytes();
  Ok(BigUint::from_bytes_le(bytes.as_slice()))
}

/// Number of limbs needed to represent any value bounded by `T_f`
/// using limbs each bounded by `T`: `numlimb = ⌈log_T(T_f)⌉ = ⌈log_t_f
/// / log_t⌉`. Returns `1` for the no-limb-split degenerate case
/// (`log_t == log_t_f`).
pub fn numlimb(log_t_f: usize, log_t: usize) -> usize {
  assert!(log_t > 0, "log_t must be positive");
  log_t_f.div_ceil(log_t).max(1)
}

/// Bit-width of the limb index — `⌈log_2 numlimb⌉`. `0` if
/// `numlimb == 1` (no extra polynomial variables needed).
pub fn numlimb_var(numlimb: usize) -> usize {
  ceil_log2(numlimb.max(1))
}

/// Decompose a `BigUint` value `v ∈ [0, 2^log_bound)` into base-`2^16`
/// little-endian chunks: `v = sum_c 2^(16c) · chunks[c]` with
/// `chunks[c] < 2^16` and `⌈log_bound / 16⌉` entries. Asserts
/// `v < 2^log_bound`; values that exceed the bound are caller errors.
/// Used by the LogUp range-check arguments.
fn chunk_decompose_value(v: &BigUint, log_bound: usize) -> Vec<u64> {
  let numchunks = log_bound.div_ceil(CHUNK_BITS);
  let bytes = v.to_bytes_le();
  debug_assert!(
    bit_decompose_check_no_overflow(&bytes, log_bound),
    "value 0x{:x} exceeds bound 2^{}",
    v,
    log_bound
  );
  let byte_at = |i: usize| -> u64 { if i < bytes.len() { bytes[i] as u64 } else { 0 } };
  (0..numchunks)
    .map(|c| byte_at(2 * c) | (byte_at(2 * c + 1) << 8))
    .collect()
}

/// Dyadic block granularity (log2 of slots) for dropping all-zero
/// regions from the range check's LogUp multiset. Committed chunk
/// polynomials carry large all-zero regions (padded rows, padded polys);
/// each all-zero block is proven zero by ONE random-point opening claim
/// (Schwartz-Zippel, strictly stronger than range membership) instead of
/// walking `2^RC_BLOCK_LOG` leaves through the GKR.
const RC_BLOCK_LOG: usize = 16;

/// Block split of an `n_chunks`-slot chunk polynomial:
/// `(block_log, n_blocks)` with `n_blocks · 2^block_log = n_chunks`.
fn rc_block_split(n_chunks: usize) -> (usize, usize) {
  let n_vars = ceil_log2(n_chunks.max(1));
  let block_log = RC_BLOCK_LOG.min(n_vars);
  (block_log, 1usize << (n_vars - block_log))
}

/// Pack an active-block bitmap for transcript absorption (LSB-first).
fn pack_bitmap(bits: &[bool]) -> Vec<u8> {
  let mut packed = vec![0u8; bits.len().div_ceil(8)];
  for (i, &b) in bits.iter().enumerate() {
    if b {
      packed[i / 8] |= 1 << (i % 8);
    }
  }
  packed
}

/// Chunk slots per value in a chunk-decomposed layout: `⌈log_bound/16⌉`
/// rounded up to a power of two, min 2 (the extra slots are zero-valued
/// and zero-weighted). Single source of truth shared by
/// [`BatchDims::new`] and the committed-chunk layout of [`commit`].
fn chunk_stride(log_bound: usize) -> usize {
  log_bound.div_ceil(CHUNK_BITS).next_power_of_two().max(2)
}

/// Checked variant of [`chunk_stride`]: the ceiling division uses
/// `checked_add` and the power-of-two rounding uses
/// `checked_next_power_of_two`, so a malformed `log_bound` returns
/// `SpartanError` instead of overflowing.
fn checked_chunk_stride(log_bound: usize) -> Result<usize, SpartanError> {
  let ceil = log_bound
    .checked_add(CHUNK_BITS - 1)
    .map(|x| x / CHUNK_BITS)
    .ok_or_else(|| params_overflow("ceil(log_bound / 16)"))?;
  ceil
    .checked_next_power_of_two()
    .map(|x| x.max(2))
    .ok_or_else(|| params_overflow("chunk_stride next_power_of_two"))
}

/// The single source of truth for the committed-chunk length of an
/// `n`-coefficient integer polynomial under `params`:
/// `n · 2^numlimb_var · chunk_stride(log_t)`, all arithmetic checked.
/// The limb-index shift is realized as a checked multiply (a bare
/// `checked_shl` only bounds the shift amount, not the shifted-out bits);
/// `numlimb_var` is first bounds-checked through `u32::try_from`.
pub fn f_chunk_len(params: &IntEvalParams, n: usize) -> Result<usize, SpartanError> {
  let shift =
    u32::try_from(params.numlimb_var).map_err(|_| params_overflow("numlimb_var as u32"))?;
  let limb_mult = 1usize
    .checked_shl(shift)
    .ok_or_else(|| params_overflow("2^numlimb_var"))?;
  let stride = checked_chunk_stride(params.log_t)?;
  n.checked_mul(limb_mult)
    .and_then(|x| x.checked_mul(stride))
    .ok_or_else(|| params_overflow("n * 2^numlimb_var * chunk_stride(log_t)"))
}

/// Shared fallible key-capacity validation for both integer Mod-PCS
/// commitment keys: rejects a zero capacity, validates the params at
/// `ceil_log2(max_n)` variables, and validates the inflated chunk length
/// before any key stores `max_n`.
fn validate_key_capacity(params: &IntEvalParams, max_n: usize) -> Result<(), SpartanError> {
  if max_n == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "integer Mod-PCS commitment key: capacity must be positive".to_string(),
    });
  }
  params.validate(ceil_log2(max_n))?;
  f_chunk_len(params, max_n)?;
  Ok(())
}

/// Build the stacked chunk polynomial of one `(num_polys, n_values,
/// log_bound)` batch: index `((p·n_values + within)·stride + c)` holds
/// chunk `c` of `values[p][within]`; padding polys (`p ≥ num_polys`) and
/// slots `c ≥ numchunks` stay zero. This IS the committed representation
/// of an integer polynomial (see [`ModPCSEngineTrait::commit`] for
/// `IntegerModPCS`), and also every range-check batch's witness layout.
fn build_chunk_poly(values: &[&[BigUint]], n_values: usize, log_bound: usize) -> Vec<u64> {
  let d = BatchDims::new(values.len(), n_values, log_bound);
  let num_polys = values.len();
  let mut chunk_vals: Vec<u64> = vec![0u64; d.n_chunks];
  // Values bounded below 2^64 are a single u64 digit; their chunks are
  // three shifts, no byte-buffer round-trip (the per-limb
  // `chunk_decompose_value` allocation dominated this pass otherwise).
  let single_word = log_bound <= 64;
  chunk_vals
    .par_chunks_mut(d.stride)
    .enumerate()
    .for_each(|(gv, slot)| {
      let p = gv / n_values;
      if p >= num_polys {
        return; // padding poly: all-zero
      }
      let within = gv % n_values;
      let v = &values[p][within];
      if single_word {
        let w = v.iter_u64_digits().next().unwrap_or(0);
        debug_assert!(v.bits() as usize <= log_bound);
        for (c, s) in slot.iter_mut().take(d.numchunks).enumerate() {
          *s = (w >> (CHUNK_BITS * c)) & ((1u64 << CHUNK_BITS) - 1);
        }
      } else {
        for (c, ch) in chunk_decompose_value(v, log_bound).into_iter().enumerate() {
          slot[c] = ch;
        }
      }
    });
  chunk_vals
}

/// Montgomery-form scalar for a base-2^16 chunk value, from a table
/// built once per process — `t256::Scalar::from` costs a Montgomery
/// multiplication, and the chunk pipelines (commit, layer commits, GKR
/// witness prep) convert millions of sub-2^16 values per proof.
fn scalar_from_chunk<F: PrimeFieldExt>(c: u64) -> F {
  F::from_chunk(c)
}

/// The chunk-axis folding point and scale of the committed-chunk layout:
/// coordinates `x_*` (one per chunk-axis variable, big-endian over the
/// chunk index) and `α` such that any multilinear `chunk` whose boolean
/// slices satisfy `value(z) = Σ_{c<2^log_stride} 2^{16c}·chunk(z, c)`
/// satisfies `chunk(z ++ x_*) = α·value(z)` for ALL `z` — so a
/// value-polynomial evaluation claim is ONE ordinary opening claim on
/// the chunk commitment.
///
/// Works because the weight vector `[2^{16c}]_c` is a tensor product
/// over the index bits: bit `b` contributes the factor `u_b = 2^{16·2^b}`
/// when set, so `x_b = u_b/(1+u_b)` gives `eq(x_*, c) = α·2^{16c}` with
/// `α = Π_b (1+u_b)^{-1}`. The fold weighs EVERY slot, including padding
/// slots `c ≥ numchunks` — honest layouts leave those zero, and the
/// range check pins them to zero soundness-grade with `range_zpad`
/// claims (a fresh random-point opening of each padding slot, zero by
/// Schwartz–Zippel), so any `numchunks` is supported.
fn chunk_fold_point<F: ff::PrimeField>(log_stride: usize) -> (Vec<F>, F) {
  let mut coords_lsb_first = Vec::with_capacity(log_stride);
  let mut alpha = F::ONE;
  for b in 0..log_stride {
    // u_b = 2^(16·2^b), by repeated squaring of 2^16.
    let mut u = F::from(1u64 << CHUNK_BITS);
    for _ in 0..b {
      u = u.square();
    }
    let denom_inv = Option::<F>::from(ff::Field::invert(&(F::ONE + u)))
      .expect("1 + 2^(16·2^b) is invertible in a prime field of odd order");
    coords_lsb_first.push(u * denom_inv);
    alpha *= denom_inv;
  }
  coords_lsb_first.reverse(); // big-endian: first coordinate = top chunk-index bit
  (coords_lsb_first, alpha)
}

/// Helper for `chunk_decompose_value`'s debug_assert: checks that the
/// LE `bytes` representation has zero bits above `num_bits`.
fn bit_decompose_check_no_overflow(bytes: &[u8], num_bits: usize) -> bool {
  let cutoff_byte = num_bits / 8;
  let cutoff_bit = num_bits % 8;
  for (i, b) in bytes.iter().enumerate() {
    match i.cmp(&cutoff_byte) {
      std::cmp::Ordering::Less => {}
      std::cmp::Ordering::Equal => {
        if cutoff_bit < 8 && (*b >> cutoff_bit) != 0 {
          return false;
        }
      }
      std::cmp::Ordering::Greater => {
        if *b != 0 {
          return false;
        }
      }
    }
  }
  true
}

/// Big-endian boolean MLE point for the index `idx` over `num_bits`
/// variables: `point[0]` is the most significant bit. Binding an MLE's
/// trailing variables to this point selects the slot `idx` of the
/// bottom axis.
fn bool_point_of_index<F: ff::PrimeField>(idx: usize, num_bits: usize) -> Vec<F> {
  (0..num_bits)
    .rev()
    .map(|b| if (idx >> b) & 1 == 1 { F::ONE } else { F::ZERO })
    .collect()
}

/// Split a single `BigUint` value `v ∈ [0, 2^log_t_f)` into `numlimb`
/// limbs each in `[0, 2^log_t)`, base-`T` little-endian: `v = sum_i
/// T^i · limbs[i]`. Asserts `v < 2^(numlimb · log_t)`; values that
/// exceed the declared bound `T_f` are caller errors (the committed-
/// chunk range check enforces the bound soundness-grade).
///
/// One pass over the LE byte representation — limb `i` is the bit
/// window `[i·log_t, (i+1)·log_t)` — instead of `numlimb` bignum
/// `div_rem`s (which dominated the reduction span at ~2^13 × 32
/// limbs).
fn split_value_into_limbs(v: &BigUint, log_t: usize, numlimb: usize) -> Vec<BigUint> {
  let bytes = v.to_bytes_le();
  debug_assert!(
    bit_decompose_check_no_overflow(&bytes, numlimb * log_t),
    "value 0x{:x} exceeds bound 2^{}",
    v,
    numlimb * log_t
  );
  (0..numlimb)
    .map(|i| extract_bit_window(&bytes, i * log_t, log_t))
    .collect()
}

/// Extract the `len`-bit window starting at bit `start` (LSB-first bit
/// order) of an LE byte string, as a `BigUint`.
fn extract_bit_window(bytes: &[u8], start: usize, len: usize) -> BigUint {
  let byte_at = |i: usize| -> u16 { if i < bytes.len() { bytes[i] as u16 } else { 0 } };
  let shift = start % 8;
  let first = start / 8;
  let n_out = len.div_ceil(8);
  let mut out = vec![0u8; n_out];
  for (j, o) in out.iter_mut().enumerate() {
    let lo = byte_at(first + j) >> shift;
    let hi = if shift == 0 {
      0
    } else {
      byte_at(first + j + 1) << (8 - shift)
    };
    *o = (lo | hi) as u8;
  }
  let rem = len % 8;
  if rem != 0 {
    out[n_out - 1] &= (1u8 << rem) - 1;
  }
  BigUint::from_bytes_le(&out)
}

/// Build the public limb-weight polynomial `limb` as a `DynPrime<2>`
/// MLE of size `2^numlimb_var`: `limb[k] = T^k` for `k < numlimb`, else
/// `0` (padding when `numlimb` isn't a power of two). Used by the
/// reduction sumcheck integrand
/// `sum_k limb(k) · f_limb(int_r, k)`.
fn build_limb_weight_dynprime(
  params: &IntEvalParams,
  monty: &crypto_bigint::modular::FixedMontyParams<2>,
) -> Vec<crate::dyn_prime::DynPrime<2>> {
  let stride = 1usize << params.numlimb_var;
  let t = BigUint::one() << params.log_t;
  let mut out = Vec::with_capacity(stride);
  let mut pow = BigUint::one();
  for k in 0..stride {
    if k < params.numlimb {
      out.push(
        <crate::dyn_prime::DynPrime<2> as SumcheckField>::from_bytes_reduce(
          monty,
          &pow.to_bytes_le(),
        ),
      );
      pow = &pow * &t;
    } else {
      out.push(<crate::dyn_prime::DynPrime<2> as SumcheckField>::zero(
        monty,
      ));
    }
  }
  out
}

/// Limb-split a multilinear polynomial. Input `poly` has length `2^n`;
/// output has length `2^n · 2^numlimb_var` where `numlimb_var =
/// ⌈log_2 numlimb⌉`. Layout: `f_limb[x · 2^numlimb_var + k]` is the
/// `k`-th limb of `f[x]` for `k < numlimb`, else `0`. The original
/// `n` variables occupy the top bits of the combined index, limb
/// variables the bottom bits — matches `EqPolynomial::evals_from_points`'s
/// convention so the limb-reduction sumcheck (step D3) treats the
/// limb dimension as the *last* variables.
fn limb_split_polynomial(poly: &[BigUint], log_t: usize, log_t_f: usize) -> Vec<BigUint> {
  let numlimb = numlimb(log_t_f, log_t);
  let numlimb_var = numlimb_var(numlimb);
  let stride = 1usize << numlimb_var;
  // Each coefficient expands to `stride` contiguous slots: its `numlimb`
  // limbs followed by zero padding. Order is preserved across the
  // parallel map, so the `x · stride + k` layout is identical to the
  // sequential version.
  poly
    .par_iter()
    .flat_map_iter(|v| {
      let mut limbs = split_value_into_limbs(v, log_t, numlimb);
      limbs.resize(stride, BigUint::zero());
      limbs.into_iter()
    })
    .collect()
}

/// Public shift bound for an `a_j` polynomial: under truncated divmod
/// with divisor `p_i`, `a_j(x) ∈ (-p_i, p_i)`. Using the universal
/// upper bound `P = 2^log_p` for all primes in the sample range gives
/// a constant shift per-`params`, independent of the specific `p_i`.
fn shift_a(params: &IntEvalParams) -> BigUint {
  BigUint::one() << params.log_p
}

/// Public shift bound for a `b_j` polynomial: per the paper's bound
/// `||g_j|| < (q-P)/2` and `|b_j| ≤ ||g_j||/p_i`, we have
/// `|b_j| < (q-P)/(2 p_i) < q/(2·P/2) = q/P` (using `p_i ≥ P/2`).
/// So shifting by `⌊q/P⌋` is sound. Like `shift_a`, this is a public
/// per-`params` constant.
/// Generic over the q-side field: `q` here must be the modulus of the
/// field the chain actually runs over (a LOG_Q-pass bug once pinned
/// this to t256's q, blowing the b-side budget 2^128-fold at q=127).
fn shift_b<F: ff::PrimeField>(params: &IntEvalParams) -> BigUint {
  &field_q::<F>() / (BigUint::one() << params.log_p)
}

/// Integer partial-evaluation at the *last* `k` variables. Given a
/// multilinear polynomial `poly` of `2^n_cur` evaluations and a binding
/// vector `r_lower` of length `k`, returns the `2^(n_cur - k)`
/// evaluations of `g(X) = poly(X, r_lower)`. Computed over Z (no
/// reduction); intermediate magnitudes can grow large.
fn integer_partial_evaluate_top_k(poly: &[BigInt], r_lower: &[BigUint]) -> Vec<BigInt> {
  let k = r_lower.len();
  let two_k = 1usize << k;
  assert!(poly.len().is_multiple_of(two_k));
  let new_size = poly.len() / two_k;

  let r_int: Vec<BigInt> = r_lower.iter().map(|x| BigInt::from(x.clone())).collect();
  let one = BigInt::one();

  // Precompute integer chi(r_lower, y) for y ∈ [0, 2^k). Bit-order
  // matches `EqPolynomial::evals_from_points`: variable i corresponds
  // to bit (k-1-i) of y.
  let chi_table: Vec<BigInt> = (0..two_k)
    .map(|y| {
      let mut chi = one.clone();
      for (i, ri) in r_int.iter().enumerate().take(k) {
        let bit = (y >> (k - 1 - i)) & 1;
        let factor = if bit == 1 { ri.clone() } else { &one - ri };
        chi *= factor;
      }
      chi
    })
    .collect();

  (0..new_size)
    .into_par_iter()
    .map(|x| {
      let mut slot = BigInt::zero();
      for (y, chi_y) in chi_table.iter().enumerate().take(two_k) {
        slot += &poly[x * two_k + y] * chi_y;
      }
      slot
    })
    .collect()
}

/// Signed 256-bit sign-magnitude integer (`Copy`, stack-allocated).
/// Used by the per-prime chain partial evaluations, whose every
/// intermediate is bounded by the Partial Evaluation Norm Bound
/// `2^k · P^k · max(T, P) ≤ (q−P)/2 < 2^255` (enforced by
/// [`IntEvalParams::validate`]) — so heap `BigInt` arithmetic there is
/// pure allocation overhead. Overflow is a bug, caught by
/// `debug_assert`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct I256 {
  neg: bool,
  mag: [u64; 4],
}

impl I256 {
  const ZERO: Self = I256 {
    neg: false,
    mag: [0u64; 4],
  };

  fn from_u64(v: u64) -> Self {
    I256 {
      neg: false,
      mag: [v, 0, 0, 0],
    }
  }

  /// Non-negative value below 2^256.
  fn from_biguint(v: &BigUint) -> Self {
    let mut mag = [0u64; 4];
    for (i, d) in v.iter_u64_digits().enumerate() {
      assert!(i < 4, "I256::from_biguint: value exceeds 256 bits");
      mag[i] = d;
    }
    I256 { neg: false, mag }
  }

  fn to_bigint(self) -> BigInt {
    let mut bytes = [0u8; 32];
    for (i, w) in self.mag.iter().enumerate() {
      bytes[8 * i..8 * (i + 1)].copy_from_slice(&w.to_le_bytes());
    }
    let m = BigInt::from_bytes_le(Sign::Plus, &bytes);
    if self.neg { -m } else { m }
  }

  /// Number of significant limbs (0 for zero).
  fn limbs(&self) -> usize {
    4 - self.mag.iter().rev().take_while(|&&w| w == 0).count()
  }

  fn mag_cmp(a: &[u64; 4], b: &[u64; 4]) -> core::cmp::Ordering {
    for i in (0..4).rev() {
      match a[i].cmp(&b[i]) {
        core::cmp::Ordering::Equal => continue,
        o => return o,
      }
    }
    core::cmp::Ordering::Equal
  }

  fn add(self, other: Self) -> Self {
    if self.neg == other.neg {
      let mut mag = [0u64; 4];
      let mut carry = 0u64;
      for (i, m) in mag.iter_mut().enumerate() {
        let (s1, c1) = self.mag[i].overflowing_add(other.mag[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        *m = s2;
        carry = u64::from(c1) + u64::from(c2);
      }
      debug_assert_eq!(carry, 0, "I256 add overflow");
      I256 {
        neg: self.neg && mag != [0u64; 4],
        mag,
      }
    } else {
      let (pos, neg) = if self.neg {
        (other, self)
      } else {
        (self, other)
      };
      let (big, small, out_neg) = match Self::mag_cmp(&pos.mag, &neg.mag) {
        core::cmp::Ordering::Less => (neg.mag, pos.mag, true),
        core::cmp::Ordering::Equal => return I256::ZERO,
        core::cmp::Ordering::Greater => (pos.mag, neg.mag, false),
      };
      let mut mag = [0u64; 4];
      let mut borrow = 0u64;
      for (i, m) in mag.iter_mut().enumerate() {
        let (d1, b1) = big[i].overflowing_sub(small[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        *m = d2;
        borrow = u64::from(b1) + u64::from(b2);
      }
      debug_assert_eq!(borrow, 0);
      I256 { neg: out_neg, mag }
    }
  }

  /// Schoolbook product, length-aware; overflow is a caller bug.
  fn mul(self, other: Self) -> Self {
    let la = self.limbs();
    let lb = other.limbs();
    if la == 0 || lb == 0 {
      return I256::ZERO;
    }
    let mut wide = [0u64; 8];
    for i in 0..la {
      let mut carry: u128 = 0;
      for j in 0..lb {
        let t = (self.mag[i] as u128) * (other.mag[j] as u128) + wide[i + j] as u128 + carry;
        wide[i + j] = t as u64;
        carry = t >> 64;
      }
      wide[i + lb] = (wide[i + lb] as u128 + carry) as u64;
    }
    debug_assert!(
      wide[4..].iter().all(|&w| w == 0),
      "I256 mul overflow (norm bound violated)"
    );
    let mag = [wide[0], wide[1], wide[2], wide[3]];
    I256 {
      neg: (self.neg != other.neg) && mag != [0u64; 4],
      mag,
    }
  }

  /// Truncated-toward-zero division by a positive word `d ≤ 2^63`:
  /// `(q, r)` with `q·d + r = self`, `sign(r) = sign(self)`, `|r| < d`.
  /// Matches `BigInt`'s `/`+`%` semantics.
  fn div_rem_u64(self, d: u64) -> (Self, Self) {
    debug_assert!((1..=(1u64 << 63)).contains(&d));
    let mut q = [0u64; 4];
    let mut rem: u128 = 0;
    for i in (0..4).rev() {
      let cur = (rem << 64) | self.mag[i] as u128;
      q[i] = (cur / d as u128) as u64;
      rem = cur % d as u128;
    }
    (
      I256 {
        neg: self.neg && q != [0u64; 4],
        mag: q,
      },
      I256 {
        neg: self.neg && rem != 0,
        mag: [rem as u64, 0, 0, 0],
      },
    )
  }
}

/// Fixed-width mirror of [`integer_partial_evaluate_top_k`]: bind the
/// LAST `k` variables of `poly` at `r_lower` over ℤ, with all values and
/// intermediates in signed 256-bit range (guaranteed by the Partial
/// Evaluation Norm Bound). Requires every `r_lower[i] < 2^64` (the
/// `log_p ≤ 63` fast-path gate).
fn integer_partial_evaluate_top_k_i256(poly: &[I256], r_lower: &[u64]) -> Vec<I256> {
  let k = r_lower.len();
  let two_k = 1usize << k;
  assert!(poly.len().is_multiple_of(two_k));
  let new_size = poly.len() / two_k;

  // chi(r_lower, y) over ℤ; variable i ↔ bit (k−1−i) of y, matching the
  // BigInt path. The bit-0 factor is `1 − r` (negative for r ≥ 2).
  let chi_table: Vec<I256> = (0..two_k)
    .map(|y| {
      let mut chi = I256::from_u64(1);
      for (i, &ri) in r_lower.iter().enumerate().take(k) {
        let bit = (y >> (k - 1 - i)) & 1;
        let factor = if bit == 1 {
          I256::from_u64(ri)
        } else if ri == 0 {
          I256::from_u64(1)
        } else {
          I256 {
            neg: ri > 1,
            mag: [ri - 1, 0, 0, 0],
          }
        };
        chi = chi.mul(factor);
      }
      chi
    })
    .collect();

  (0..new_size)
    .into_par_iter()
    .map(|x| {
      let mut slot = I256::ZERO;
      for (y, chi_y) in chi_table.iter().enumerate().take(two_k) {
        slot = slot.add(poly[x * two_k + y].mul(*chi_y));
      }
      slot
    })
    .collect()
}

/// Compute the signed integer MLE evaluation `sum_k chi_int(k, point) ·
/// poly[k]`, where `chi_int(k, point) = prod_i (k_i · point_i + (1-k_i) ·
/// (1-point_i))` over Z (no reduction). Returns the full integer.
///
/// Used by the IntEval prover to compute `int_v' = f(int_r)`. The result
/// can be huge — bounded by `2^n · p^n · max(|poly|)` in magnitude — and
/// can be negative when `(1 - point_i)` flips signs.
fn integer_mle_evaluate(poly: &[BigUint], point: &[BigUint]) -> BigInt {
  let n = poly.len();
  let num_vars = n.trailing_zeros() as usize;
  debug_assert_eq!(1 << num_vars, n);
  debug_assert_eq!(point.len(), num_vars);

  // Bind variables one at a time (top variable first), folding the table
  // in half per round over ℤ: out[i] = lo[i] + r·(hi[i] − lo[i]). The
  // bit-order matches `EqPolynomial::evals_from_points` / `bind_poly_var_top`:
  // variable `i` corresponds to bit `num_vars - 1 - i` of the index, so
  // `point[0]` splits the table into its two halves. Entry widths grow by
  // ~|r| bits per round while the count halves, so the total bigint work
  // is dominated by the early (narrow) rounds — far cheaper than
  // materializing the 2^n chi table at full width, and each round is
  // embarrassingly parallel.
  let mut cur: Vec<BigInt> = poly.par_iter().map(|x| BigInt::from(x.clone())).collect();
  for r in point {
    let r_int = BigInt::from(r.clone());
    let h = cur.len() / 2;
    let (lo, hi) = cur.split_at(h);
    cur = if h >= 1 << 10 {
      lo.par_iter()
        .zip(hi.par_iter())
        .map(|(a, b)| a + &r_int * (b - a))
        .collect()
    } else {
      lo.iter()
        .zip(hi.iter())
        .map(|(a, b)| a + &r_int * (b - a))
        .collect()
    };
  }
  cur.pop().expect("non-empty table")
}

/// Rejection-sample a small prime in `[2^{log_p - 1}, 2^{log_p})` from
/// the transcript via Miller-Rabin / Lucas BPSW. Squeezes 64 bytes at a
/// time, builds a `log_p`-bit candidate with the MSB and LSB forced,
/// runs `crypto_primes::is_prime`, and retries on composite. The two
/// sides (prover & verifier) drive the transcript identically, so they
/// arrive at the same prime.
fn sample_small_prime<T: ByteTranscript>(
  transcript: &mut T,
  log_p: usize,
) -> Result<BigUint, SpartanError> {
  use crypto_primes::{Flavor, is_prime};
  // `crypto_primes::is_prime` works over `Uint<L>`; we use `U256` here
  // since `log_p` is bounded by `LOG_Q = 256`.
  use crypto_bigint::U256;
  assert!(log_p > 1 && log_p <= LOG_Q);
  let bytes_needed = log_p.div_ceil(8);
  loop {
    let bytes = transcript.squeeze_bytes(b"sample_small_p")?;
    let mut buf = [0u8; 32];
    buf[..bytes_needed].copy_from_slice(&bytes[..bytes_needed]);
    // Force MSB of bit (log_p - 1) so candidate has exactly log_p bits;
    // force LSB so it's odd. Clear bits above log_p - 1 so width is exact.
    let top_byte = (log_p - 1) / 8;
    let top_bit_in_byte = (log_p - 1) % 8;
    // Clear bits above log_p - 1.
    if top_byte < 32 {
      let mask_top: u8 = (1u16 << (top_bit_in_byte + 1)).wrapping_sub(1) as u8;
      buf[top_byte] &= mask_top;
      for b in &mut buf[(top_byte + 1)..] {
        *b = 0;
      }
    }
    // Force MSB and LSB.
    buf[top_byte] |= 1u8 << top_bit_in_byte;
    buf[0] |= 0x01;
    let candidate = U256::from_le_slice(&buf);
    if is_prime(Flavor::Any, &candidate) {
      return Ok(BigUint::from_bytes_le(&buf));
    }
  }
}

/// Sound Mod-PCS for `T256DynPrimeEngine`. See module docs.
#[derive(Clone)]
pub struct IntegerModPCS {
  _phantom: PhantomData<()>,
}

/// Application-level defaults used by the trait `setup` when the caller
/// doesn't pass explicit `IntEvalParams`. These are the application-
/// level bounds and iteration knob; the rest of the protocol parameters
/// (`log P`, `s`) are derived from them. Use `setup_with_params` to
/// override.
///
/// Default polynomial norm bound used by trait `setup` (`log_2(T_f)`).
pub const DEFAULT_LOG_T_F: usize = 32;
/// Default per-iteration variable count used by trait `setup`. The paper
/// recommends `k = ⌈log λ⌉ = 7`, but a measured (size × k × log_t) sweep on
/// the msshape family (vars 2^11–2^14, 256-bit) and MultiSwap (2^13,
/// 2048-bit) found `k = 9` (with `log_t = 64`) fastest at every size, with a
/// flat basin k=8–10.
pub const DEFAULT_K: usize = 9;

impl IntegerModPCS {
  /// Explicit-params setup. Validates the params against `num_vars =
  /// log_2(n)` so caller-supplied configurations can't bypass the
  /// IntEval soundness bounds.
  pub fn setup_with_params(
    label: &'static [u8],
    n: usize,
    width: usize,
    params: IntEvalParams,
  ) -> Result<
    (
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::CommitmentKey,
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::VerifierKey,
    ),
    SpartanError,
  > {
    validate_key_capacity(&params, n)?;
    // Hyrax CK must be sized for the *committed chunk* polynomial: the
    // input poly has `n` coefficients, each limb-split into
    // `2^numlimb_var` slots and each limb chunk-decomposed into
    // `chunk_stride(log_t)` base-2^16 slots — exactly `f_chunk_len`.
    let inflated_n = f_chunk_len(&params, n)?;
    let (inner_ck, inner_vk) = Hyrax::setup(label, inflated_n, width);
    // Size-1 eval key for the internal `G^{f_y}` eval commitments (kept
    // inside the Mod-PCS key, off the PCS-agnostic trait surface).
    let (eval_ck, _) = Hyrax::setup(b"imod_modpcs_eval", 1, 1);
    // Precompute once at setup so neither prove nor verify rebuilds the
    // eval key's generator table per `G^{f_y}` (re)construction. Cloning
    // after precompute propagates it into both the commitment and verifier
    // keys (mirrors the pre-refactor `precompute_ck(&ck_s)`).
    Hyrax::precompute_ck(&eval_ck);
    Ok((
      IntegerModCommitmentKey::new_checked(inner_ck, eval_ck.clone(), params.clone(), n)?,
      IntegerModVerifierKey {
        inner: inner_vk,
        eval: eval_ck,
        params,
      },
    ))
  }

  /// Like the trait `setup`, but instead of the fixed application
  /// defaults `(DEFAULT_LOG_T_F, DEFAULT_K)` it derives params
  /// *optimized for this input length* via
  /// [`IntEvalParams::derive_optimized`]: `(k, log T, log P, s)` are
  /// chosen to minimize the estimated prover cost for an `n`-coefficient
  /// polynomial with coefficients bounded by `2^log_t_f`. The chosen
  /// params still pass the full §4.4 `validate`, so this never trades
  /// soundness for speed.
  pub fn setup_optimized(
    label: &'static [u8],
    n: usize,
    width: usize,
    log_t_f: usize,
  ) -> Result<
    (
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::CommitmentKey,
      <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::VerifierKey,
    ),
    SpartanError,
  > {
    let num_vars = ceil_log2(n.max(1));
    let params = IntEvalParams::derive_optimized(log_t_f, num_vars)?;
    Self::setup_with_params(label, n, width, params)
  }
}

impl ModPCSEngineTrait<T256DynPrimeEngine> for IntegerModPCS {
  type CommitmentKey = IntegerModCommitmentKey;
  type VerifierKey = IntegerModVerifierKey;
  type Commitment = IntegerModCommitment;
  type Blind = IntegerModBlind;
  type EvaluationArgument = IntEvalArgument<HyBackend>;
  type BatchEvaluationArgument = IntEvalBatchArgument<HyBackend>;

  /// Trait-driven setup: derive `IntEvalParams` optimized for this
  /// polynomial size via [`IntEvalParams::derive_optimized`], with the
  /// application-default norm bound `DEFAULT_LOG_T_F`. Panics if the
  /// derivation fails (which only happens for pathologically small `n`);
  /// callers that need control over the security or norm-bound
  /// parameters should use `setup_with_params`.
  fn setup(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    let num_vars = ceil_log2(n.max(1));
    let params = IntEvalParams::derive_optimized(DEFAULT_LOG_T_F, num_vars).expect(
      "default IntEvalParams derivation must satisfy the paper's bounds; \
         override with `setup_with_params` to use tighter parameters",
    );
    let inflated_n = f_chunk_len(&params, n)
      .expect("library-derived default parameters imply a valid f_chunk_len");
    let (inner_ck, inner_vk) = Hyrax::setup(label, inflated_n, width);
    let (eval_ck, _) = Hyrax::setup(b"imod_modpcs_eval", 1, 1);
    // Precompute the eval key once at setup (see `setup_with_params`).
    Hyrax::precompute_ck(&eval_ck);
    (
      // The infallible trait boundary: this constructor call only sees
      // library-derived, already-validated parameters, so the single
      // invariant-backed expect is unreachable for a valid trait call.
      IntegerModCommitmentKey::new_checked(inner_ck, eval_ck.clone(), params.clone(), n)
        .expect("library-derived default parameters must validate at this capacity"),
      IntegerModVerifierKey {
        inner: inner_vk,
        eval: eval_ck,
        params,
      },
    )
  }

  fn precompute_ck(ck: &Self::CommitmentKey) {
    Hyrax::precompute_ck(&ck.inner);
    // Also precompute the size-1 eval key (used for the internal `G^{f_y}`
    // eval commitments) so a deserialized key doesn't rebuild it lazily.
    Hyrax::precompute_ck(&ck.eval);
  }

  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind {
    // Documented caller contract for the infallible trait boundary: the
    // key was validated for `max_n`, and `f_chunk_len` is monotone in
    // `n`, so any `n <= max_n` has a valid inflated length. A caller
    // violating the capacity contract gets this deliberate assertion,
    // not an accidental overflow.
    assert!(
      n <= ck.max_n,
      "IntegerModPCS::blind: n = {n} exceeds the commitment-key capacity {}",
      ck.max_n
    );
    // `commit` limb-splits an `n`-coefficient polynomial to
    // `2^numlimb_var · n` coefficients and chunk-decomposes each limb
    // into `chunk_stride(log_t)` base-2^16 slots before reaching the
    // inner Hyrax PCS, so the blind must cover that inflated length.
    // (Size-1 eval commits skip splitting in `commit`; an over-long
    // blind is harmless there.)
    let inflated =
      f_chunk_len(&ck.params, n).expect("f_chunk_len validated for commitment-key capacity");
    IntegerModBlind {
      inner: Hyrax::blind(&ck.inner, inflated),
    }
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    // The commitment is the Hyrax commitment of the polynomial's
    // base-2^16 CHUNK decomposition: each coefficient limb-splits into
    // `numlimb` limbs in `[0, T)`, and each limb chunk-decomposes into
    // `chunk_stride(log_t)` 16-bit slots (index `limb_index·stride + c`,
    // matching the range-check batch layout). Committing the chunks —
    // not the limbs — means the range check's chunk oracle IS the input
    // commitment (no duplicate MSM), and every limb evaluation claim
    // folds to one chunk claim through `chunk_fold_point`. Chunk values
    // are `< 2^16`, so the small-scalar MSM fast path (`is_small`) is
    // always legitimate regardless of `log_t`.
    //
    // Stopgap: size-1 commits are single-value commits used internally
    // (e.g. the eval-value commitment); the value may be any F element,
    // not bounded by `T_f`, so skip splitting/chunking and the small-
    // scalar path in that case.
    //
    // Capacity check independent of `blind`, so bypassing `blind` cannot
    // bypass the key's validated capacity.
    if v.len() > ck.max_n {
      return Err(SpartanError::InvalidVectorSize {
        actual: v.len(),
        max: ck.max_n,
      });
    }
    let params = &ck.params;
    if v.len() == 1 {
      let v_fq: Vec<t256::Scalar> = v.iter().map(biguint_to_scalar).collect();
      let inner = Hyrax::commit(&ck.inner, &v_fq, &r.inner, false)?;
      return Ok(IntegerModCommitment { inner });
    }
    let v_limbs = limb_split_polynomial(v, params.log_t, params.log_t_f);
    let chunk_vals = build_chunk_poly(&[&v_limbs], v_limbs.len(), params.log_t);
    let chunk_fq: Vec<t256::Scalar> = chunk_vals
      .par_iter()
      .map(|&c| scalar_from_chunk(c))
      .collect();
    let inner = Hyrax::commit(&ck.inner, &chunk_fq, &r.inner, true)?;
    Ok(IntegerModCommitment { inner })
  }

  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError> {
    Hyrax::check_commitment(&comm.inner, n, width)
  }

  fn commitment_log_t_f(ck: &Self::CommitmentKey) -> usize {
    ck.params.log_t_f
  }

  fn verifier_log_t_f(vk: &Self::VerifierKey) -> usize {
    vk.params.log_t_f
  }

  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    poly: &[BigUint],
    blind: &Self::Blind,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    eval: &BigUint,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let (_prove_span, prove_t) = start_span!("integer_modpcs_prove");
    let mut st = prove_one_poly::<HyBackend, T256DynPrimeEngine>(
      &ck.params, ck, transcript, poly, point, eval,
    )?;
    let (range_check, combined_open, _) = finish_batch_open::<HyBackend, T256DynPrimeEngine>(
      &ck.params,
      ck,
      transcript,
      std::slice::from_mut(&mut st),
      &[&comm.inner],
      &[&blind.inner],
      &[&[]],
    )?;
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "integer_modpcs_prove");
    Ok(IntEvalArgument {
      reduction_round_polys: st.reduction_round_polys,
      int_v_prime: st.int_v_prime,
      chains: st.chains,
      ab_comms: st.ab_comms,
      range_check,
      combined_open,
    })
  }

  fn prove_batch(
    ck: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    polys: &[&[BigUint]],
    blinds: &[&Self::Blind],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError> {
    let empty: Vec<&[SmallValueBlock]> = vec![&[]; polys.len()];
    <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::prove_batch_with_blocks(
      ck, transcript, comms, polys, blinds, points, evals, &empty,
    )
  }

  fn prove_batch_with_blocks(
    ck: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    polys: &[&[BigUint]],
    blinds: &[&Self::Blind],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    blocks: &[&[SmallValueBlock]],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError> {
    let (_prove_span, prove_t) = start_span!("integer_modpcs_prove_batch");
    let n = polys.len();
    if n == 0 || comms.len() != n || blinds.len() != n || points.len() != n || evals.len() != n {
      return Err(SpartanError::InternalError {
        reason: "IntegerModPCS::prove_batch: empty or mismatched inputs".to_string(),
      });
    }
    // Per polynomial: reduction sumcheck + chains, advancing the shared
    // transcript in index order. The expensive range check and inner-
    // product opening are then run ONCE over every polynomial's batches.
    // Two-phase schedule: every polynomial's reduction + chain COMMITS
    // land on the transcript before ANY checking challenge (gammas) is
    // squeezed — the batching-friendly ordering.
    let mut ph1s: Vec<ChainPhase1<HyBackend>> = Vec::with_capacity(n);
    for i in 0..n {
      ph1s.push(prove_one_poly_phase1::<HyBackend, T256DynPrimeEngine>(
        &ck.params, ck, transcript, polys[i], points[i], evals[i],
      )?);
    }
    let mut states: Vec<PerPolyProver<HyBackend>> = Vec::with_capacity(n);
    for ph1 in ph1s {
      states.push(prove_one_poly_phase2::<HyBackend, T256DynPrimeEngine>(
        &ck.params, transcript, ph1,
      )?);
    }
    let comm_inners: Vec<_> = comms.iter().map(|c| &c.inner).collect();
    let blind_inners: Vec<_> = blinds.iter().map(|b| &b.inner).collect();
    let (range_check, combined_open, small_block_evals) =
      finish_batch_open::<HyBackend, T256DynPrimeEngine>(
        &ck.params,
        ck,
        transcript,
        &mut states,
        &comm_inners,
        &blind_inners,
        blocks,
      )?;
    let per_poly = states
      .into_iter()
      .map(|st| IntEvalPerPolyArgument {
        reduction_round_polys: st.reduction_round_polys,
        int_v_prime: st.int_v_prime,
        chains: st.chains,
        ab_comms: st.ab_comms,
      })
      .collect();
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "integer_modpcs_prove_batch");
    Ok(IntEvalBatchArgument {
      per_poly,
      range_check,
      combined_open,
      small_block_evals,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    point: &[<T256DynPrimeEngine as SumcheckEngine>::Scalar],
    eval: &BigUint,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("integer_modpcs_verify");
    let mut v = verify_one_poly::<HyBackend, T256DynPrimeEngine>(
      &vk.params,
      transcript,
      point,
      eval,
      &arg.reduction_round_polys,
      &arg.int_v_prime,
      &arg.chains,
      &arg.ab_comms,
    )?;
    finish_batch_verify::<HyBackend, T256DynPrimeEngine>(
      &vk.params,
      vk,
      transcript,
      &[&comm.inner],
      std::slice::from_mut(&mut v),
      &[arg.ab_comms.as_slice()],
      &arg.range_check,
      &arg.combined_open,
      &[&[]],
      &[],
    )?;
    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "integer_modpcs_verify");
    Ok(())
  }

  fn verify_batch(
    vk: &Self::VerifierKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    arg: &Self::BatchEvaluationArgument,
  ) -> Result<(), SpartanError> {
    let empty: Vec<&[SmallValueBlock]> = vec![&[]; comms.len()];
    <Self as ModPCSEngineTrait<T256DynPrimeEngine>>::verify_batch_with_blocks(
      vk, transcript, comms, points, evals, arg, &empty,
    )
  }

  fn verify_batch_with_blocks(
    vk: &Self::VerifierKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    arg: &Self::BatchEvaluationArgument,
    blocks: &[&[SmallValueBlock]],
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("integer_modpcs_verify_batch");
    let n = arg.per_poly.len();
    if comms.len() != n || points.len() != n || evals.len() != n {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let mut vph1s: Vec<VerifyPhase1> = Vec::with_capacity(n);
    for i in 0..n {
      let pp = &arg.per_poly[i];
      vph1s.push(verify_one_poly_phase1::<HyBackend, T256DynPrimeEngine>(
        &vk.params,
        transcript,
        points[i],
        evals[i],
        &pp.reduction_round_polys,
        &pp.int_v_prime,
        &pp.chains,
        &pp.ab_comms,
      )?);
    }
    let mut vs: Vec<PerPolyVerifier> = Vec::with_capacity(n);
    for (i, ph1) in vph1s.into_iter().enumerate() {
      let pp = &arg.per_poly[i];
      vs.push(verify_one_poly_phase2::<HyBackend, T256DynPrimeEngine>(
        &vk.params,
        transcript,
        &pp.chains,
        &pp.int_v_prime,
        ph1,
      )?);
    }
    let comm_inners: Vec<_> = comms.iter().map(|c| &c.inner).collect();
    let ab_comms_per_poly: Vec<_> = arg
      .per_poly
      .iter()
      .map(|pp| pp.ab_comms.as_slice())
      .collect();
    finish_batch_verify::<HyBackend, T256DynPrimeEngine>(
      &vk.params,
      vk,
      transcript,
      &comm_inners,
      &mut vs,
      &ab_comms_per_poly,
      &arg.range_check,
      &arg.combined_open,
      blocks,
      &arg.small_block_evals,
    )?;
    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "integer_modpcs_verify_batch");
    Ok(())
  }

  fn commit_at(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
    log_t_f: usize,
  ) -> Result<Self::Commitment, SpartanError> {
    let params = ck.params.narrowed(log_t_f)?;
    Self::commit_seg(ck, v, r, &params)
  }

  fn prove_batch_with_params(
    ck: &Self::CommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    polys: &[&[BigUint]],
    blinds: &[&Self::Blind],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    blocks: &[&[SmallValueBlock]],
    log_t_fs: &[usize],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError> {
    if log_t_fs.len() != polys.len() {
      return Err(SpartanError::InternalError {
        reason: "prove_batch_with_params: log_t_fs length mismatch".to_string(),
      });
    }
    let params_per: Vec<IntEvalParams> = log_t_fs
      .iter()
      .map(|&l| ck.params.narrowed(l))
      .collect::<Result<_, _>>()?;
    Self::prove_batch_seg(
      ck,
      transcript,
      comms,
      polys,
      blinds,
      points,
      evals,
      blocks,
      &params_per,
    )
  }

  fn verify_batch_with_params(
    vk: &Self::VerifierKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    arg: &Self::BatchEvaluationArgument,
    blocks: &[&[SmallValueBlock]],
    log_t_fs: &[usize],
  ) -> Result<(), SpartanError> {
    let params_per: Vec<IntEvalParams> = log_t_fs
      .iter()
      .map(|&l| vk.params.narrowed(l))
      .collect::<Result<_, _>>()?;
    Self::verify_batch_seg(
      vk,
      transcript,
      comms,
      points,
      evals,
      arg,
      blocks,
      &params_per,
    )
  }
}

/// Width-grouped commitment primitives: commit and batch-open with a
/// *per-polynomial* `IntEvalParams`, so a witness split into segments by
/// value width commits each segment at a matched bound. A narrow segment
/// (small `log_t_f` → few limbs) yields a shorter chunk vector and a
/// genuinely cheaper MSM / range check, while the shared range check and
/// combined opening still run once over the whole batch. `ck.params`
/// must be the *widest* segment's params (it sizes the generators and
/// carries the uniform `(log_t, log_p, log_q)` bounds every segment
/// agrees on). See `docs/imod_followups.md` for the measured basis.
#[allow(dead_code)] // exercised by tests; wired into the driver next
impl IntegerModPCS {
  pub(crate) fn commit_seg(
    ck: &IntegerModCommitmentKey,
    v: &[BigUint],
    r: &IntegerModBlind,
    params: &IntEvalParams,
  ) -> Result<IntegerModCommitment, SpartanError> {
    if v.len() == 1 {
      let v_fq: Vec<t256::Scalar> = v.iter().map(biguint_to_scalar).collect();
      let inner = Hyrax::commit(&ck.inner, &v_fq, &r.inner, false)?;
      return Ok(IntegerModCommitment { inner });
    }
    let v_limbs = limb_split_polynomial(v, params.log_t, params.log_t_f);
    let chunk_vals = build_chunk_poly(&[&v_limbs], v_limbs.len(), params.log_t);
    let chunk_fq: Vec<t256::Scalar> = chunk_vals
      .par_iter()
      .map(|&c| scalar_from_chunk(c))
      .collect();
    let inner = Hyrax::commit(&ck.inner, &chunk_fq, &r.inner, true)?;
    Ok(IntegerModCommitment { inner })
  }

  /// [`ModPCSEngineTrait::prove_batch_with_blocks`] with a per-polynomial
  /// `IntEvalParams`: poly `i`'s reduction + chains run at `params_per[i]`,
  /// while the shared range check + combined open run once at `ck.params`.
  /// Every polynomial that carries a non-empty `blocks` entry MUST have
  /// `params_per[i].numlimb_var == ck.params.numlimb_var` (the small-value
  /// gadget indexes the chunk oracle at the batch numlimb_var); segments
  /// narrower than the batch commit at `log_t_f = log_t` (numlimb 1) and
  /// range-check by commit width, so they carry no blocks.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn prove_batch_seg(
    ck: &IntegerModCommitmentKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&IntegerModCommitment],
    polys: &[&[BigUint]],
    blinds: &[&IntegerModBlind],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    blocks: &[&[SmallValueBlock]],
    params_per: &[IntEvalParams],
  ) -> Result<IntEvalBatchArgument<HyBackend>, SpartanError> {
    let n = polys.len();
    if n == 0
      || comms.len() != n
      || blinds.len() != n
      || points.len() != n
      || evals.len() != n
      || blocks.len() != n
      || params_per.len() != n
    {
      return Err(SpartanError::InternalError {
        reason: "prove_batch_with_params: empty or mismatched inputs".to_string(),
      });
    }
    let mut ph1s: Vec<ChainPhase1<HyBackend>> = Vec::with_capacity(n);
    for i in 0..n {
      ph1s.push(prove_one_poly_phase1::<HyBackend, T256DynPrimeEngine>(
        &params_per[i],
        ck,
        transcript,
        polys[i],
        points[i],
        evals[i],
      )?);
    }
    let mut states: Vec<PerPolyProver<HyBackend>> = Vec::with_capacity(n);
    for (i, ph1) in ph1s.into_iter().enumerate() {
      states.push(prove_one_poly_phase2::<HyBackend, T256DynPrimeEngine>(
        &params_per[i],
        transcript,
        ph1,
      )?);
    }
    let comm_inners: Vec<_> = comms.iter().map(|c| &c.inner).collect();
    let blind_inners: Vec<_> = blinds.iter().map(|b| &b.inner).collect();
    let (range_check, combined_open, small_block_evals) =
      finish_batch_open::<HyBackend, T256DynPrimeEngine>(
        &ck.params,
        ck,
        transcript,
        &mut states,
        &comm_inners,
        &blind_inners,
        blocks,
      )?;
    let per_poly = states
      .into_iter()
      .map(|st| IntEvalPerPolyArgument {
        reduction_round_polys: st.reduction_round_polys,
        int_v_prime: st.int_v_prime,
        chains: st.chains,
        ab_comms: st.ab_comms,
      })
      .collect();
    Ok(IntEvalBatchArgument {
      per_poly,
      range_check,
      combined_open,
      small_block_evals,
    })
  }

  /// Verifier mirror of [`IntegerModPCS::prove_batch_seg`].
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn verify_batch_seg(
    vk: &IntegerModVerifierKey,
    transcript: &mut <T256DynPrimeEngine as SumcheckEngine>::TE,
    comms: &[&IntegerModCommitment],
    points: &[&[<T256DynPrimeEngine as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    arg: &IntEvalBatchArgument<HyBackend>,
    blocks: &[&[SmallValueBlock]],
    params_per: &[IntEvalParams],
  ) -> Result<(), SpartanError> {
    let n = arg.per_poly.len();
    if comms.len() != n
      || points.len() != n
      || evals.len() != n
      || blocks.len() != n
      || params_per.len() != n
    {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let mut vph1s: Vec<VerifyPhase1> = Vec::with_capacity(n);
    for i in 0..n {
      let pp = &arg.per_poly[i];
      vph1s.push(verify_one_poly_phase1::<HyBackend, T256DynPrimeEngine>(
        &params_per[i],
        transcript,
        points[i],
        evals[i],
        &pp.reduction_round_polys,
        &pp.int_v_prime,
        &pp.chains,
        &pp.ab_comms,
      )?);
    }
    let mut vs: Vec<PerPolyVerifier> = Vec::with_capacity(n);
    for (i, ph1) in vph1s.into_iter().enumerate() {
      let pp = &arg.per_poly[i];
      vs.push(verify_one_poly_phase2::<HyBackend, T256DynPrimeEngine>(
        &params_per[i],
        transcript,
        &pp.chains,
        &pp.int_v_prime,
        ph1,
      )?);
    }
    let comm_inners: Vec<_> = comms.iter().map(|c| &c.inner).collect();
    let ab_comms_per_poly: Vec<_> = arg
      .per_poly
      .iter()
      .map(|pp| pp.ab_comms.as_slice())
      .collect();
    finish_batch_verify::<HyBackend, T256DynPrimeEngine>(
      &vk.params,
      vk,
      transcript,
      &comm_inners,
      &mut vs,
      &ab_comms_per_poly,
      &arg.range_check,
      &arg.combined_open,
      blocks,
      &arg.small_block_evals,
    )?;
    Ok(())
  }
}

/// The Brakedown-backed integer Mod-PCS: the same IntEval protocol as
/// [`IntegerModPCS`], with commitments and final openings going through
/// `BdBackend` (hash-based, non-hiding — this instantiation is NOT
/// zero-knowledge). Used for the code-commitment comparison
/// instantiation: no elliptic-curve operations anywhere in the prover,
/// at the cost of megabyte-scale proofs and slower verification.
#[derive(Clone, Debug)]
pub struct IntegerModPCSBd<SE = T256HyraxEngine>(core::marker::PhantomData<SE>);

/// Commitment key for the Brakedown Mod-PCS: the IntEval parameters plus
/// the validated capacity (Brakedown itself needs no key material — its
/// code matrices derive from a public seed).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BdModCommitmentKey {
  pub(crate) backend_config: [usize; 4],
  pub(crate) params: IntEvalParams,
  /// Maximum committable polynomial length the key was validated for.
  pub(crate) max_n: usize,
}

/// Verifier key: same content as the commitment key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BdModVerifierKey {
  pub(crate) backend_config: [usize; 4],
  pub(crate) params: IntEvalParams,
}

/// A Brakedown Mod-PCS commitment: the Merkle root of the chunk
/// polynomial's encoded-column tree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BdModCommitment {
  pub(crate) root: [u8; 32],
}

impl TranscriptReprTrait for BdModCommitment {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.root.to_vec()
  }
}

impl BdModCommitmentKey {
  /// Key from explicit IntEval params and the polynomial capacity it must
  /// serve. Rejects a zero capacity, validates the params at that
  /// capacity, and validates the implied `f_chunk_len` (see
  /// [`validate_key_capacity`]).
  pub fn new(params: IntEvalParams, max_n: usize) -> Result<Self, SpartanError> {
    validate_key_capacity(&params, max_n)?;
    Ok(Self {
      params,
      max_n,
      backend_config: super::commit_backend::bd_configuration(),
    })
  }
}

impl BdModVerifierKey {
  /// Key from explicit IntEval params.
  pub fn new(params: IntEvalParams) -> Self {
    Self {
      params,
      backend_config: super::commit_backend::bd_configuration(),
    }
  }
}

impl<ME, SE> ModPCSEngineTrait<ME> for IntegerModPCSBd<SE>
where
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
  SE: crate::traits::mod_engine::SumcheckEngine,
  SE::Scalar: crate::traits::PrimeFieldExt
    + crate::traits::transcript::TranscriptReprTrait
    + Serialize
    + serde::de::DeserializeOwned
    + crate::big_num::DelayedReduction<SE::Scalar>
    + Send
    + Sync
    + 'static,
{
  type CommitmentKey = BdModCommitmentKey;
  type VerifierKey = BdModVerifierKey;
  type Commitment = BdModCommitment;
  type Blind = ();
  type EvaluationArgument = IntEvalArgument<BdBackend<SE>>;
  type BatchEvaluationArgument = IntEvalBatchArgument<BdBackend<SE>>;

  fn setup(
    _label: &'static [u8],
    n: usize,
    _width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey) {
    let num_vars = ceil_log2(n.max(1));
    let params = IntEvalParams::derive_optimized(DEFAULT_LOG_T_F, num_vars).expect(
      "default IntEvalParams derivation must satisfy the paper's bounds; \
         override with `setup_with_params` to use tighter parameters",
    );
    (
      // Infallible trait boundary: library-derived validated params, so
      // the single invariant-backed expect is unreachable for a valid
      // trait call.
      BdModCommitmentKey::new(params.clone(), n)
        .expect("library-derived default parameters must validate at this capacity"),
      BdModVerifierKey::new(params),
    )
  }

  fn precompute_ck(_ck: &Self::CommitmentKey) {}

  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind {
    // Same documented caller contract as the Hyrax impl; the blind itself
    // is a unit for this non-hiding backend.
    assert!(
      n <= ck.max_n,
      "IntegerModPCSBd::blind: n = {n} exceeds the commitment-key capacity {}",
      ck.max_n
    );
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    _r: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError> {
    if ck.backend_config != super::commit_backend::bd_configuration() {
      return Err(SpartanError::InvalidInputLength {
        reason: "Brakedown configuration differs from key".into(),
      });
    }
    // Identical chunk layout to the Hyrax Mod-PCS (the protocol's
    // committed-chunk representation), committed with Brakedown.
    //
    // Capacity check independent of `blind`, so bypassing `blind` cannot
    // bypass the key's validated capacity.
    if v.len() > ck.max_n {
      return Err(SpartanError::InvalidVectorSize {
        actual: v.len(),
        max: ck.max_n,
      });
    }
    let params = &ck.params;
    let v_limbs = limb_split_polynomial(v, params.log_t, params.log_t_f);
    let chunk_vals = build_chunk_poly(&[&v_limbs], v_limbs.len(), params.log_t);
    let chunk_fq: Vec<SE::Scalar> = chunk_vals
      .par_iter()
      .map(|&c| scalar_from_chunk(c))
      .collect();
    let (root, _data) = BdBackend::<SE>::commit(&(), &chunk_fq, &(), true)?;
    Ok(BdModCommitment { root })
  }

  fn check_commitment(
    _comm: &Self::Commitment,
    _n: usize,
    _width: usize,
  ) -> Result<(), SpartanError> {
    Ok(())
  }

  fn commitment_log_t_f(ck: &Self::CommitmentKey) -> usize {
    ck.params.log_t_f
  }

  fn verifier_log_t_f(vk: &Self::VerifierKey) -> usize {
    vk.params.log_t_f
  }

  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut <ME as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    poly: &[BigUint],
    blind: &Self::Blind,
    point: &[<ME as SumcheckEngine>::Scalar],
    eval: &BigUint,
  ) -> Result<Self::EvaluationArgument, SpartanError> {
    let (_prove_span, prove_t) = start_span!("integer_modpcs_bd_prove");
    if ck.backend_config != super::commit_backend::bd_configuration() {
      return Err(SpartanError::InvalidInputLength {
        reason: "Brakedown configuration differs from key".into(),
      });
    }
    let mut st =
      prove_one_poly::<BdBackend<SE>, ME>(&ck.params, &(), transcript, poly, point, eval)?;
    let (range_check, combined_open, _) = finish_batch_open::<BdBackend<SE>, ME>(
      &ck.params,
      &(),
      transcript,
      std::slice::from_mut(&mut st),
      &[&comm.root],
      &[blind],
      &[&[]],
    )?;
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "integer_modpcs_bd_prove");
    Ok(IntEvalArgument {
      reduction_round_polys: st.reduction_round_polys,
      int_v_prime: st.int_v_prime,
      chains: st.chains,
      ab_comms: st.ab_comms,
      range_check,
      combined_open,
    })
  }

  fn prove_batch(
    ck: &Self::CommitmentKey,
    transcript: &mut <ME as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    polys: &[&[BigUint]],
    blinds: &[&Self::Blind],
    points: &[&[<ME as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError> {
    let empty: Vec<&[SmallValueBlock]> = vec![&[]; polys.len()];
    <Self as ModPCSEngineTrait<ME>>::prove_batch_with_blocks(
      ck, transcript, comms, polys, blinds, points, evals, &empty,
    )
  }

  fn prove_batch_with_blocks(
    ck: &Self::CommitmentKey,
    transcript: &mut <ME as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    polys: &[&[BigUint]],
    blinds: &[&Self::Blind],
    points: &[&[<ME as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    blocks: &[&[SmallValueBlock]],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError> {
    let (_prove_span, prove_t) = start_span!("integer_modpcs_bd_prove_batch");
    if ck.backend_config != super::commit_backend::bd_configuration() {
      return Err(SpartanError::InvalidInputLength {
        reason: "Brakedown configuration differs from key".into(),
      });
    }
    let n = polys.len();
    if n == 0 || comms.len() != n || blinds.len() != n || points.len() != n || evals.len() != n {
      return Err(SpartanError::InternalError {
        reason: "IntegerModPCSBd::prove_batch: empty or mismatched inputs".to_string(),
      });
    }
    // Two-phase schedule (as in the Hyrax impl): all commits before
    // all checking challenges.
    let mut ph1s: Vec<ChainPhase1<BdBackend<SE>>> = Vec::with_capacity(n);
    for i in 0..n {
      ph1s.push(prove_one_poly_phase1::<BdBackend<SE>, ME>(
        &ck.params,
        &(),
        transcript,
        polys[i],
        points[i],
        evals[i],
      )?);
    }
    let mut states: Vec<PerPolyProver<BdBackend<SE>>> = Vec::with_capacity(n);
    for ph1 in ph1s {
      states.push(prove_one_poly_phase2::<BdBackend<SE>, ME>(
        &ck.params, transcript, ph1,
      )?);
    }
    let comm_roots: Vec<_> = comms.iter().map(|c| &c.root).collect();
    let (range_check, combined_open, small_block_evals) = finish_batch_open::<BdBackend<SE>, ME>(
      &ck.params,
      &(),
      transcript,
      &mut states,
      &comm_roots,
      blinds,
      blocks,
    )?;
    let per_poly = states
      .into_iter()
      .map(|st| IntEvalPerPolyArgument {
        reduction_round_polys: st.reduction_round_polys,
        int_v_prime: st.int_v_prime,
        chains: st.chains,
        ab_comms: st.ab_comms,
      })
      .collect();
    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "integer_modpcs_bd_prove_batch");
    Ok(IntEvalBatchArgument {
      per_poly,
      range_check,
      combined_open,
      small_block_evals,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut <ME as SumcheckEngine>::TE,
    comm: &Self::Commitment,
    point: &[<ME as SumcheckEngine>::Scalar],
    eval: &BigUint,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("integer_modpcs_bd_verify");
    if vk.backend_config != super::commit_backend::bd_configuration() {
      return Err(SpartanError::InvalidInputLength {
        reason: "Brakedown configuration differs from key".into(),
      });
    }
    let mut v = verify_one_poly::<BdBackend<SE>, ME>(
      &vk.params,
      transcript,
      point,
      eval,
      &arg.reduction_round_polys,
      &arg.int_v_prime,
      &arg.chains,
      &arg.ab_comms,
    )?;
    finish_batch_verify::<BdBackend<SE>, ME>(
      &vk.params,
      &(),
      transcript,
      &[&comm.root],
      std::slice::from_mut(&mut v),
      &[arg.ab_comms.as_slice()],
      &arg.range_check,
      &arg.combined_open,
      &[&[]],
      &[],
    )?;
    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "integer_modpcs_bd_verify");
    Ok(())
  }

  fn verify_batch(
    vk: &Self::VerifierKey,
    transcript: &mut <ME as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    points: &[&[<ME as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    arg: &Self::BatchEvaluationArgument,
  ) -> Result<(), SpartanError> {
    let empty: Vec<&[SmallValueBlock]> = vec![&[]; comms.len()];
    <Self as ModPCSEngineTrait<ME>>::verify_batch_with_blocks(
      vk, transcript, comms, points, evals, arg, &empty,
    )
  }

  fn verify_batch_with_blocks(
    vk: &Self::VerifierKey,
    transcript: &mut <ME as SumcheckEngine>::TE,
    comms: &[&Self::Commitment],
    points: &[&[<ME as SumcheckEngine>::Scalar]],
    evals: &[&BigUint],
    arg: &Self::BatchEvaluationArgument,
    blocks: &[&[SmallValueBlock]],
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("integer_modpcs_bd_verify_batch");
    if vk.backend_config != super::commit_backend::bd_configuration() {
      return Err(SpartanError::InvalidInputLength {
        reason: "Brakedown configuration differs from key".into(),
      });
    }
    let n = arg.per_poly.len();
    if comms.len() != n || points.len() != n || evals.len() != n {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let mut vph1s: Vec<VerifyPhase1> = Vec::with_capacity(n);
    for i in 0..n {
      let pp = &arg.per_poly[i];
      vph1s.push(verify_one_poly_phase1::<BdBackend<SE>, ME>(
        &vk.params,
        transcript,
        points[i],
        evals[i],
        &pp.reduction_round_polys,
        &pp.int_v_prime,
        &pp.chains,
        &pp.ab_comms,
      )?);
    }
    let mut vs: Vec<PerPolyVerifier<SE::Scalar>> = Vec::with_capacity(n);
    for (i, ph1) in vph1s.into_iter().enumerate() {
      let pp = &arg.per_poly[i];
      vs.push(verify_one_poly_phase2::<BdBackend<SE>, ME>(
        &vk.params,
        transcript,
        &pp.chains,
        &pp.int_v_prime,
        ph1,
      )?);
    }
    let comm_roots: Vec<_> = comms.iter().map(|c| &c.root).collect();
    let ab_comms_per_poly: Vec<_> = arg
      .per_poly
      .iter()
      .map(|pp| pp.ab_comms.as_slice())
      .collect();
    finish_batch_verify::<BdBackend<SE>, ME>(
      &vk.params,
      &(),
      transcript,
      &comm_roots,
      &mut vs,
      &ab_comms_per_poly,
      &arg.range_check,
      &arg.combined_open,
      blocks,
      &arg.small_block_evals,
    )?;
    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "integer_modpcs_bd_verify_batch");
    Ok(())
  }

  /// Explicit traversal of every `ME::Scalar` (`DynPrime<2>`) the batch
  /// argument carries: there are none — identical reasoning to the Hyrax
  /// impl (p-side data is `BigUint`/`BigInt`; chains, range check, and
  /// combined opening are q-side static scalars and hashes).
  fn batch_arg_is_in_context(
    _arg: &Self::BatchEvaluationArgument,
    _expected: &<<ME as SumcheckEngine>::Scalar as SumcheckField>::Params,
  ) -> bool {
    true
  }
}

/// Per-polynomial prover state of a (batched) Mod-PCS open: the proof
/// outputs plus the witness polynomials the shared range check / combined
/// opening borrow from. Built by [`prove_one_poly`], consumed by
/// [`finish_batch_open`].
struct PerPolyProver<B: CommitBackend> {
  /// This poly's own `numlimb_var` (limb-axis variable count); segments of
  /// different widths differ here, so block claims must use it, not the
  /// shared batch params.
  numlimb_var: usize,
  reduction_round_polys: Vec<Vec<BigUint>>,
  int_v_prime: BigInt,
  chains: Vec<ChainData<B::Scalar>>,
  /// Per-layer chunk commitments, 2 per layer (`a_j` then `b_j`).
  ab_comms: Vec<B::Comm>,
  /// `f_limb` reduced to the Hyrax base field: F-batch value polynomial
  /// (chain-claim evaluations only — the combined open runs on chunks).
  poly_fq: Vec<B::Scalar>,
  /// `f_limb` as integers: the F-batch's range-checked values.
  f_limb: Vec<BigUint>,
  /// Per-prime chain states feeding the `a_j`/`b_j` layer batches.
  chain_states: Vec<ChainProverState<B::Scalar>>,
  /// Per-layer, per-role stacked chunk polynomials (the committed
  /// oracles) and their blinds; index `2·(j−1) + role`.
  ab_chunk_polys: Vec<Vec<B::Scalar>>,
  ab_blinds: Vec<B::Blind>,
  /// Retained opening data for each layer chunk commitment.
  ab_open_aux: Vec<B::Data>,
  /// Accumulated multi-point claims on the input commitment / the
  /// per-layer chunk commitments (already in chunk coordinates).
  f_claims: OpenClaims<B::Scalar>,
  ab_claims: Vec<OpenClaims<B::Scalar>>,
  t_layers: usize,
}

/// Everything phase 1 of a per-polynomial open produces and phase 2
/// consumes: the reduction outputs, the chain state, and the committed
/// chunk polynomials. Splitting here lets the batch prover commit ALL
/// polynomials' chains before ANY checking challenge is squeezed (the
/// two-tree batching schedule).
struct ChainPhase1<B: CommitBackend> {
  num_vars: usize,
  with_iter: bool,
  t_layers: usize,
  log_spad: usize,
  log_bound_a: usize,
  log_bound_b: usize,
  f_limb: Vec<BigUint>,
  poly_fq: Vec<B::Scalar>,
  int_v_prime: BigInt,
  reduction_round_polys: Vec<Vec<BigUint>>,
  chain_states: Vec<ChainProverState<B::Scalar>>,
  ab_chunk_polys: Vec<Vec<B::Scalar>>,
  ab_blinds: Vec<B::Blind>,
  ab_comms: Vec<B::Comm>,
  ab_open_aux: Vec<B::Data>,
}

fn prove_one_poly_phase1<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  backend_ck: &B::Ck,
  transcript: &mut Keccak256Transcript<ME>,
  poly: &[BigUint],
  point: &[crate::dyn_prime::DynPrime<2>],
  eval: &BigUint,
) -> Result<ChainPhase1<B>, SpartanError> {
  let monty = point
    .first()
    .map(|p| *p.params())
    .ok_or(SpartanError::InternalError {
      reason: "IntegerModPCS::prove: empty point".to_string(),
    })?;

  let (_red_span, red_t) = start_span!("imod_pcs_reduction_sc");
  // 0. Limb-split f → f_limb. For numlimb=1 this is a literal pass-
  //    through; for numlimb>1 f_limb has 2^numlimb_var times as many
  //    coefficients (and 2^numlimb_var slots per original coefficient,
  //    padded with zero if numlimb isn't a power of two).
  let (_ls_span, ls_t) = start_span!("imod_pcs_red_limb_split");
  let f_limb = limb_split_polynomial(poly, params.log_t, params.log_t_f);
  info!(elapsed_ms = %ls_t.elapsed().as_millis(), "imod_pcs_red_limb_split");
  drop(_ls_span);

  // 1. Reduction sumcheck: reduce the eval claim
  //    `f(int_r) ≡_p eval` to a claim about `f_limb` at a combined
  //    point `(int_r, r_k)` where `r_k` are the sumcheck challenges.
  let (_dc_span, dc_t) = start_span!("imod_pcs_red_to_dynprime");
  let f_limb_p: Vec<crate::dyn_prime::DynPrime<2>> = f_limb
    .iter()
    .map(|b| {
      <crate::dyn_prime::DynPrime<2> as SumcheckField>::from_bytes_reduce(&monty, &b.to_bytes_le())
    })
    .collect();
  info!(elapsed_ms = %dc_t.elapsed().as_millis(), "imod_pcs_red_to_dynprime");
  drop(_dc_span);
  // Partial-eval f_limb at the original `point` in Z_p, leaving the last
  // numlimb_var variables free.
  let (_pe_span, pe_t) = start_span!("imod_pcs_red_dynprime_bind");
  let mut mle = crate::polys_modp::multilinear::MultilinearPolynomial::new(f_limb_p, monty);
  for r_i in point {
    mle.bind_poly_var_top(r_i);
  }
  let f_limb_at_int_r: Vec<crate::dyn_prime::DynPrime<2>> = mle.into_vec();
  info!(elapsed_ms = %pe_t.elapsed().as_millis(), "imod_pcs_red_dynprime_bind");
  drop(_pe_span);
  debug_assert_eq!(f_limb_at_int_r.len(), 1 << params.numlimb_var);

  let limb_p = build_limb_weight_dynprime(params, &monty);
  let eval_p = <crate::dyn_prime::DynPrime<2> as SumcheckField>::from_bytes_reduce(
    &monty,
    &eval.to_bytes_le(),
  );

  let mut poly_lhs = crate::polys_modp::multilinear::MultilinearPolynomial::new(limb_p, monty);
  let mut poly_rhs =
    crate::polys_modp::multilinear::MultilinearPolynomial::new(f_limb_at_int_r, monty);
  let (red_sc, r_k, final_claims) = crate::sumcheck_modp::SumcheckProof::<ME>::prove_quad(
    &eval_p,
    params.numlimb_var,
    &mut poly_lhs,
    &mut poly_rhs,
    transcript,
  )?;
  // `final_claims = [limb(r_k), f_limb(int_r, r_k)]`.
  let f_eval_p = final_claims[1];

  // 2. Extend the integer point with r_k (canonical < p integers).
  let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
  let r_k_int: Vec<BigUint> = r_k.iter().map(dyn_to_biguint).collect();
  let int_point_ext: Vec<BigUint> = int_point.iter().chain(r_k_int.iter()).cloned().collect();

  // 3. int_v' = f_limb at the extended point, over Z.
  let (_ie_span, ie_t) = start_span!("imod_pcs_red_integer_eval");
  let int_v_prime = integer_mle_evaluate(&f_limb, &int_point_ext);
  info!(elapsed_ms = %ie_t.elapsed().as_millis(), "imod_pcs_red_integer_eval");
  drop(_ie_span);

  // 4. Sanity: f_eval_p ≡ int_v' (mod p).
  let p = extract_p(point)?;
  let int_v_mod_p_u = int_v_prime
    .mod_floor(&BigInt::from(p.clone()))
    .to_biguint()
    .expect("mod_floor by a positive divisor is non-negative");
  let f_eval_bu = BigUint::from_bytes_le(&f_eval_p.to_le_bytes());
  if int_v_mod_p_u != f_eval_bu {
    return Err(SpartanError::InternalError {
      reason: "IntegerModPCS::prove: f_limb(ext_point) ≠ int_v' mod p (prover bug)".to_string(),
    });
  }

  // 5. Bind int_v' into the transcript.
  absorb_bigint(transcript, &int_v_prime);

  let reduction_round_polys: Vec<Vec<BigUint>> = red_sc
    .compressed_polys
    .iter()
    .map(|cp| {
      cp.coeffs_except_linear_term
        .iter()
        .map(dyn_to_biguint)
        .collect()
    })
    .collect();

  // From here on the chain prover operates on `f_limb` over the extended
  // point. For numlimb_var=0 these match the pre-D3 `poly` / `point`.
  let int_point = int_point_ext;
  let num_vars = point.len() + params.numlimb_var;
  let with_iter = num_vars > params.k;
  let poly = f_limb.as_slice();
  let (_fq_span, fq_t) = start_span!("imod_pcs_red_to_fq");
  let poly_fq: Vec<B::Scalar> = poly.iter().map(biguint_to_scalar::<B::Scalar>).collect();
  info!(elapsed_ms = %fq_t.elapsed().as_millis(), "imod_pcs_red_to_fq");
  drop(_fq_span);
  info!(elapsed_ms = %red_t.elapsed().as_millis(), "imod_pcs_reduction");
  drop(_red_span);

  // Phase 1: per prime, sample p_i, run all t iterations (if any).
  let (_p1_span, p1_t) = start_span!("imod_pcs_chain_phase1");
  // Fast path: with `log_p ≤ 63` every chain intermediate fits signed
  // 256 bits (the Partial Eval Norm bound), the prime and the reduced
  // point coordinates fit u64, and the whole partial-eval/divmod loop
  // runs on stack-allocated `I256` instead of heap `BigInt`. The
  // shared read-only `poly_i256` also replaces the per-chain
  // `poly_bigint.clone()`.
  let fast = params.log_p <= 63;
  let poly_i256: Vec<I256> = if with_iter && fast {
    poly.par_iter().map(I256::from_biguint).collect()
  } else {
    Vec::new()
  };
  let poly_bigint: Vec<BigInt> = if with_iter && !fast {
    poly.par_iter().map(|x| BigInt::from(x.clone())).collect()
  } else {
    Vec::new()
  };

  let primes: Vec<BigUint> = (0..params.s)
    .map(|_| sample_small_prime(transcript, params.log_p))
    .collect::<Result<Vec<_>, SpartanError>>()?;

  let (_cb_span, cb_t) = start_span!("imod_pcs_chain_build");
  let chain_states: Vec<ChainProverState<B::Scalar>> = primes
    .par_iter()
    .map(|p_i| -> Result<ChainProverState<B::Scalar>, SpartanError> {
      let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % p_i).collect();
      let mut iters = Vec::new();

      if with_iter {
        let t = num_vars.saturating_sub(params.k).div_ceil(params.k);
        let n = num_vars;
        let k = params.k;
        let d_big = BigInt::from(p_i.clone());
        let p_u64 = if fast {
          p_i.iter_u64_digits().next().expect("prime is nonzero")
        } else {
          0
        };
        let s_a = BigInt::from(shift_a(params));
        let s_b = BigInt::from(shift_b::<B::Scalar>(params));

        let mut a_prev_int: Vec<BigInt> = Vec::new();
        let mut a_prev_i256: Vec<I256> = Vec::new();

        for j in 1..=t {
          let lo = n - j * k;
          let hi = n - (j - 1) * k;
          let r_lower = &r_i_int[lo..hi];

          let (b_j_int, a_j_int): (Vec<BigInt>, Vec<BigInt>) = if fast {
            let src: &[I256] = if j == 1 { &poly_i256 } else { &a_prev_i256 };
            let r_lower_u64: Vec<u64> = r_lower
              .iter()
              .map(|x| x.iter_u64_digits().next().unwrap_or(0))
              .collect();
            let g_j = integer_partial_evaluate_top_k_i256(src, &r_lower_u64);
            let mut b_j = Vec::with_capacity(g_j.len());
            let mut a_j = Vec::with_capacity(g_j.len());
            let mut a_next = Vec::with_capacity(g_j.len());
            for g in g_j {
              let (q, r) = g.div_rem_u64(p_u64);
              b_j.push(q.to_bigint());
              a_j.push(r.to_bigint());
              a_next.push(r);
            }
            a_prev_i256 = a_next;
            (b_j, a_j)
          } else {
            let src: &[BigInt] = if j == 1 { &poly_bigint } else { &a_prev_int };
            let g_j_int = integer_partial_evaluate_top_k(src, r_lower);
            let (b_j, a_j): (Vec<BigInt>, Vec<BigInt>) = g_j_int
              .iter()
              .map(|g| {
                let q = g / &d_big;
                let r = g - &q * &d_big;
                (q, r)
              })
              .unzip();
            a_prev_int = a_j.clone();
            (b_j, a_j)
          };

          let a_j_shifted: Vec<BigUint> = a_j_int
            .iter()
            .map(|x| (x + &s_a).to_biguint().expect("shift makes non-negative"))
            .collect();
          let b_j_shifted: Vec<BigUint> = b_j_int
            .iter()
            .map(|x| (x + &s_b).to_biguint().expect("shift makes non-negative"))
            .collect();

          let a_j_shifted_fq: Vec<B::Scalar> = a_j_shifted
            .iter()
            .map(biguint_to_scalar::<B::Scalar>)
            .collect();
          let b_j_shifted_fq: Vec<B::Scalar> = b_j_shifted
            .iter()
            .map(biguint_to_scalar::<B::Scalar>)
            .collect();

          if std::env::var_os("CHAIN_BITS").is_some() {
            let max_a = a_j_shifted.iter().map(|v| v.bits()).max().unwrap_or(0);
            let max_b = b_j_shifted.iter().map(|v| v.bits()).max().unwrap_or(0);
            eprintln!(
              "CHAIN_BITS layer j={j}: max|a_shifted|={max_a} bits (budget {}), max|b_shifted|={max_b} bits (budget {})",
              params.log_p + 1,
              params.log_q - params.log_p + 1
            );
          }
          iters.push(IterationProverState {
            a_shifted: a_j_shifted,
            a_shifted_fq: a_j_shifted_fq,
            b_shifted: b_j_shifted,
            b_shifted_fq: b_j_shifted_fq,
          });
        }
      }

      Ok(ChainProverState {
        r_i_int,
        iters,
      })
    })
    .collect::<Result<Vec<_>, SpartanError>>()?;
  info!(elapsed_ms = %cb_t.elapsed().as_millis(), "imod_pcs_chain_build");
  drop(_cb_span);

  let t_layers = if with_iter {
    num_vars.saturating_sub(params.k).div_ceil(params.k)
  } else {
    0
  };
  let s_pad = params.s.next_power_of_two();
  let log_spad = s_pad.trailing_zeros() as usize;
  let log_bound_a = params.log_p + 1;
  let log_bound_b = params.log_q - params.log_p + 1;
  // Per layer and role, the committed oracle is the 16-bit CHUNK
  // decomposition of the shifted values in the shared range-check
  // layout (`index = (chain·m + x)·stride + c`, chains padded to
  // s_pad) — the same representation the range check consumes, so the
  // layer commitment doubles as its chunk oracle (no duplicate MSM,
  // and the small-scalar fast path always applies). Order: per layer
  // `j`, the `a_j` chunk commitment then the `b_j` one (2t total).
  let mut ab_chunk_polys: Vec<Vec<B::Scalar>> = Vec::with_capacity(2 * t_layers);
  let mut ab_blinds: Vec<B::Blind> = Vec::with_capacity(2 * t_layers);
  let mut ab_open_aux: Vec<B::Data> = Vec::with_capacity(2 * t_layers);
  let mut ab_comms: Vec<B::Comm> = Vec::with_capacity(2 * t_layers);
  let (_ab_span, ab_t) = start_span!("imod_pcs_ab_commit");
  for jm1 in 0..t_layers {
    let m = 1usize << (num_vars - (jm1 + 1) * params.k);
    for role in 0..2u8 {
      let values: Vec<&[BigUint]> = chain_states
        .iter()
        .map(|cs| {
          if role == 0 {
            cs.iters[jm1].a_shifted.as_slice()
          } else {
            cs.iters[jm1].b_shifted.as_slice()
          }
        })
        .collect();
      let log_bound = if role == 0 { log_bound_a } else { log_bound_b };
      let chunk_vals = build_chunk_poly(&values, m, log_bound);
      let chunk_fq: Vec<B::Scalar> = chunk_vals
        .par_iter()
        .map(|&c| scalar_from_chunk::<B::Scalar>(c))
        .collect();
      let blind = B::blind(backend_ck, chunk_fq.len());
      let (comm, data) = B::commit(backend_ck, &chunk_fq, &blind, true)?;
      transcript.absorb_bytes(b"ab_chunk", &B::comm_transcript_bytes(&comm));
      ab_chunk_polys.push(chunk_fq);
      ab_blinds.push(blind);
      ab_open_aux.push(data);
      ab_comms.push(comm);
    }
  }
  info!(elapsed_ms = %ab_t.elapsed().as_millis(), "imod_pcs_ab_commit");
  drop(_ab_span);
  info!(elapsed_ms = %p1_t.elapsed().as_millis(), "imod_pcs_chain_phase1");
  drop(_p1_span);

  Ok(ChainPhase1 {
    num_vars,
    with_iter,
    t_layers,
    log_spad,
    log_bound_a,
    log_bound_b,
    f_limb,
    poly_fq,
    int_v_prime,
    reduction_round_polys,
    chain_states,
    ab_chunk_polys,
    ab_blinds,
    ab_comms,
    ab_open_aux,
  })
}

/// Phase 2: squeeze the gammas and build every evaluation claim from
/// the phase-1 state. Consumes the state; returns the assembled
/// [`PerPolyProver`].
fn prove_one_poly_phase2<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  transcript: &mut Keccak256Transcript<ME>,
  ph1: ChainPhase1<B>,
) -> Result<PerPolyProver<B>, SpartanError> {
  let ChainPhase1 {
    num_vars,
    with_iter,
    t_layers,
    log_spad,
    log_bound_a,
    log_bound_b,
    f_limb,
    poly_fq,
    int_v_prime,
    reduction_round_polys,
    chain_states,
    ab_chunk_polys,
    ab_blinds,
    ab_comms,
    ab_open_aux,
  } = ph1;

  // Sample γ ∈ F^{n-k} after all phase-1 commits are absorbed.
  let gamma_fq: Vec<B::Scalar> = if with_iter {
    (0..(num_vars - params.k))
      .map(|i| {
        let bytes = transcript.squeeze_bytes(b"gamma")?;
        let label = (i as u64).to_le_bytes();
        transcript.absorb_bytes(b"gamma_idx", &label);
        Ok(<B::Scalar as PrimeFieldExt>::from_uniform(&bytes))
      })
      .collect::<Result<Vec<_>, SpartanError>>()?
  } else {
    Vec::new()
  };

  // Identity-check and final-remainder evaluations become multi-point
  // claims on `f` / the per-layer chunk commitments. Layer-value claims
  // fold to chunk claims immediately (`value(z)·α = chunk(z ++ x_*)`);
  // `ab_claims[2·(j−1) + role]` targets layer j's role commitment.
  let (_open_span, open_t) = start_span!("imod_pcs_chain_claims");
  let mut f_claims = OpenClaims::<B::Scalar>::default();
  let mut ab_claims: Vec<OpenClaims<B::Scalar>> = (0..2 * t_layers)
    .map(|_| OpenClaims::<B::Scalar>::default())
    .collect();
  let (fold_a, alpha_a) =
    chunk_fold_point::<B::Scalar>(chunk_stride(log_bound_a).trailing_zeros() as usize);
  let (fold_b, alpha_b) =
    chunk_fold_point::<B::Scalar>(chunk_stride(log_bound_b).trailing_zeros() as usize);
  let ab_chunk_point = |role: u8, chain: usize, sub: &[B::Scalar]| -> Vec<B::Scalar> {
    let fold = if role == 0 { &fold_a } else { &fold_b };
    let mut pt = Vec::with_capacity(log_spad + sub.len() + fold.len());
    pt.extend(bool_point_of_index::<B::Scalar>(chain, log_spad));
    pt.extend_from_slice(sub);
    pt.extend_from_slice(fold);
    pt
  };

  let poly_at_gamma: Vec<B::Scalar> = if with_iter {
    let mut m = crate::polys::multilinear::MultilinearPolynomial::new(poly_fq.clone());
    for r in &gamma_fq[..(num_vars - params.k)] {
      m.bind_poly_var_top(r);
    }
    m.into_vec()
  } else {
    Vec::new()
  };

  let mut chains: Vec<ChainData<B::Scalar>> = Vec::with_capacity(params.s);
  for (ci, state) in chain_states.iter().enumerate() {
    let r_i_int = &state.r_i_int;
    let iters = &state.iters;
    let n = num_vars;
    let k = params.k;

    let mut iter_oracles = Vec::with_capacity(iters.len());
    for (jm1, iter_state) in iters.iter().enumerate() {
      let j = jm1 + 1;
      let prefix_len = n - j * k;
      let lo = n - j * k;
      let hi = n - (j - 1) * k;
      let r_lower_fq: Vec<B::Scalar> = r_i_int[lo..hi]
        .iter()
        .map(biguint_to_scalar::<B::Scalar>)
        .collect();
      let gamma_prefix: Vec<B::Scalar> = gamma_fq[..prefix_len].to_vec();
      let gamma_extended: Vec<B::Scalar> = gamma_prefix
        .iter()
        .chain(r_lower_fq.iter())
        .copied()
        .collect();

      let a_prev_eval = if j == 1 {
        let v = mle_evaluate_fq(&poly_at_gamma, &r_lower_fq);
        f_claims.push(gamma_extended.clone(), v);
        v
      } else {
        let v = mle_evaluate_fq(&iters[jm1 - 1].a_shifted_fq, &gamma_extended);
        ab_claims[2 * (jm1 - 1)].push(ab_chunk_point(0, ci, &gamma_extended), alpha_a * v);
        v
      };
      let a_curr_eval = mle_evaluate_fq(&iter_state.a_shifted_fq, &gamma_prefix);
      let b_curr_eval = mle_evaluate_fq(&iter_state.b_shifted_fq, &gamma_prefix);
      ab_claims[2 * jm1].push(ab_chunk_point(0, ci, &gamma_prefix), alpha_a * a_curr_eval);
      ab_claims[2 * jm1 + 1].push(ab_chunk_point(1, ci, &gamma_prefix), alpha_b * b_curr_eval);
      for ev in [&a_prev_eval, &a_curr_eval, &b_curr_eval] {
        transcript.absorb_bytes(b"claim_ev", ev.to_repr().as_ref());
      }
      iter_oracles.push(IterationOracles {
        a_prev_eval,
        a_curr_eval,
        b_curr_eval,
      });
    }

    let t = iters.len();
    let final_point_fq: Vec<B::Scalar> = r_i_int[..(num_vars - t * params.k)]
      .iter()
      .map(biguint_to_scalar::<B::Scalar>)
      .collect();
    let final_eval = if t == 0 {
      let v = mle_evaluate_fq(&poly_fq, &final_point_fq);
      f_claims.push(final_point_fq, v);
      v
    } else {
      let last = &iters[t - 1];
      let v = mle_evaluate_fq(&last.a_shifted_fq, &final_point_fq);
      ab_claims[2 * (t - 1)].push(ab_chunk_point(0, ci, &final_point_fq), alpha_a * v);
      v
    };
    transcript.absorb_bytes(b"claim_ev", final_eval.to_repr().as_ref());

    chains.push(ChainData {
      iterations: iter_oracles,
      final_eval,
    });
  }
  info!(elapsed_ms = %open_t.elapsed().as_millis(), "imod_pcs_chain_claims");
  drop(_open_span);

  Ok(PerPolyProver {
    numlimb_var: params.numlimb_var,
    reduction_round_polys,
    int_v_prime,
    chains,
    ab_comms,
    poly_fq,
    f_limb,
    chain_states,
    ab_chunk_polys,
    ab_blinds,
    ab_open_aux,
    f_claims,
    ab_claims,
    t_layers,
  })
}

/// Single-polynomial open: phase 1 then phase 2 back to back (the
/// transcript order matches the pre-split protocol for one poly).
fn prove_one_poly<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  backend_ck: &B::Ck,
  transcript: &mut Keccak256Transcript<ME>,
  poly: &[BigUint],
  point: &[crate::dyn_prime::DynPrime<2>],
  eval: &BigUint,
) -> Result<PerPolyProver<B>, SpartanError> {
  let ph1 = prove_one_poly_phase1::<B, ME>(params, backend_ck, transcript, poly, point, eval)?;
  prove_one_poly_phase2::<B, ME>(params, transcript, ph1)
}

/// Shared finish of a (batched) Mod-PCS open: ONE LogUp-GKR range check
/// over every batch of every polynomial, then ONE combined inner-product
/// opening discharging every evaluation claim. `comms[p]` / `blinds[p]`
/// are polynomial `p`'s input commitment / blind. The range check's value
/// claims are routed back into each state's claim sets before the combined
/// opening consumes them. Canonical batch / target order: for each
/// polynomial in turn (its `f_limb` batch then its `a_j`/`b_j` layer
/// batches), then all chunk commitments in that batch order, then the
/// shared multiplicity table.
/// The two zero-subcube claims of one [`SmallValueBlock`] on a
/// polynomial's chunk oracle `C(x, k, c)` (index `(x·2^nlv + k)·stride +
/// c`). With every chunk already range-checked below `2^16`, `w[x] <
/// 2^16` for all `x` in the block iff `C(x, k, c) = 0` for `(k, c) ≠
/// (0, 0)`, i.e. iff
///   `C(prefix, r_x, r_kc) = eq(r_kc, 0) · C(prefix, r_x, 0)`
/// as polynomials in `(r_x, r_kc)` — the difference is multilinear with
/// hypercube values exactly the block's non-lowest chunks. Both sides
/// squeeze `r_x`, `r_kc` here; the prover computes `e2 = C(prefix, r_x,
/// 0)` from the chunk polynomial, the verifier takes it from the proof;
/// `e1 = eq(r_kc, 0)·e2` is derived. The batched open then binds both
/// evaluations to the commitment. `n` is the polynomial's variable
/// count, `nlv`/`log_stride` the limb/chunk index bits.
fn small_block_claims<B: CommitBackend, T: ByteTranscript>(
  transcript: &mut T,
  n: usize,
  nlv: usize,
  log_stride: usize,
  block: &SmallValueBlock,
  chunk_fq: Option<&[B::Scalar]>,
  e2_in: Option<B::Scalar>,
) -> Result<(OpenClaims<B::Scalar>, B::Scalar), SpartanError> {
  block.validate(n)?;
  let m = block.log_len;
  let kc_bits = nlv + log_stride;
  let mut squeeze = |label: &'static [u8], i: usize| -> Result<B::Scalar, SpartanError> {
    let bytes = transcript.squeeze_bytes(label)?;
    transcript.absorb_bytes(b"blk_idx", &(i as u64).to_le_bytes());
    Ok(<B::Scalar as PrimeFieldExt>::from_uniform(&bytes))
  };
  let r_x: Vec<B::Scalar> = (0..m)
    .map(|i| squeeze(b"blk_rx", i))
    .collect::<Result<_, _>>()?;
  let r_kc: Vec<B::Scalar> = (0..kc_bits)
    .map(|i| squeeze(b"blk_rkc", i))
    .collect::<Result<_, _>>()?;
  let prefix = bool_point_of_index::<B::Scalar>(block.start >> m, n - m);
  let mut p1 = prefix.clone();
  p1.extend_from_slice(&r_x);
  p1.extend_from_slice(&r_kc);
  let mut p2 = prefix;
  p2.extend_from_slice(&r_x);
  p2.extend(core::iter::repeat_n(B::Scalar::ZERO, kc_bits));
  let eq0 = r_kc
    .iter()
    .fold(B::Scalar::ONE, |acc, r| acc * (B::Scalar::ONE - *r));
  let e2 = match (chunk_fq, e2_in) {
    (Some(c), _) => mle_evaluate_fq(c, &p2),
    (None, Some(e)) => e,
    (None, None) => {
      return Err(SpartanError::InternalError {
        reason: "small_block_claims: neither chunk data nor a proof value".to_string(),
      });
    }
  };
  transcript.absorb_bytes(b"blk_ev", e2.to_repr().as_ref());
  let mut cl = OpenClaims::<B::Scalar>::default();
  cl.push(p1, eq0 * e2);
  cl.push(p2, e2);
  Ok((cl, e2))
}

fn finish_batch_open<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  backend_ck: &B::Ck,
  transcript: &mut Keccak256Transcript<ME>,
  states: &mut [PerPolyProver<B>],
  comms: &[&B::Comm],
  blinds: &[&B::Blind],
  blocks: &[&[SmallValueBlock]],
) -> Result<
  (
    SharedRangeCheck<B>,
    CombinedBatchOpen<B>,
    Vec<Vec<B::Scalar>>,
  ),
  SpartanError,
>
where
  B::Scalar: crate::big_num::DelayedReduction<B::Scalar>,
{
  let log_bound_a = params.log_p + 1;
  let log_bound_b = params.log_q - params.log_p + 1;

  // ONE shared LogUp-GKR range check across every polynomial's batches.
  // Each poly's F batch is PRECOMMITTED: the input commitment already
  // is its chunk polynomial, so the range check reuses it instead of
  // committing again. `f_batch_idx[p]` is poly `p`'s canonical batch
  // index.
  let mut f_batch_idx: Vec<usize> = Vec::with_capacity(states.len());
  let (range_check, rc_art) = {
    let mut rc_batches: Vec<RangeBatchInputs<'_, B>> = Vec::new();
    for (p, st) in states.iter().enumerate() {
      f_batch_idx.push(rc_batches.len());
      rc_batches.push(RangeBatchInputs {
        target: RcTarget::F { poly: p },
        value_polys_fq: vec![st.poly_fq.as_slice()],
        values: vec![st.f_limb.as_slice()],
        n_values: st.f_limb.len(),
        log_bound: params.log_t,
        precommitted: Some((comms[p], blinds[p])),
      });
      for j in 0..st.t_layers {
        for (role, log_bound) in [(0u8, log_bound_a), (1u8, log_bound_b)] {
          let value_polys_fq = st
            .chain_states
            .iter()
            .map(|cs| {
              if role == 0 {
                cs.iters[j].a_shifted_fq.as_slice()
              } else {
                cs.iters[j].b_shifted_fq.as_slice()
              }
            })
            .collect::<Vec<_>>();
          let values = st
            .chain_states
            .iter()
            .map(|cs| {
              if role == 0 {
                cs.iters[j].a_shifted.as_slice()
              } else {
                cs.iters[j].b_shifted.as_slice()
              }
            })
            .collect::<Vec<_>>();
          let n_values = values[0].len();
          rc_batches.push(RangeBatchInputs {
            target: RcTarget::Ab {
              poly: p,
              layer: j,
              role,
            },
            value_polys_fq,
            values,
            n_values,
            log_bound,
            precommitted: Some((
              &st.ab_comms[2 * j + role as usize],
              &st.ab_blinds[2 * j + role as usize],
            )),
          });
        }
      }
    }
    let (_rc_span, rc_t) = start_span!("imod_pcs_rc_shared");
    let out = prove_shared_range_check::<B, ME>(backend_ck, &rc_batches, transcript)?;
    info!(elapsed_ms = %rc_t.elapsed().as_millis(), "imod_pcs_rc_shared");
    drop(_rc_span);
    out
  };

  // Every batch is precommitted (its chunk polynomial IS the target's
  // commitment), so the range check emits no value claims.
  debug_assert!(rc_art.value_claims.is_empty());

  // Per poly: fold every f_limb claim into a chunk claim on the input
  // commitment (`f_limb(z) · α = chunk(z ++ x_*)`), then append each
  // target's GKR/top/zero-pad chunk claims from the range check — they
  // are claims on the same commitments. `ab_claims` are already in
  // chunk coordinates. Canonical batch index of poly `p` layer `j` role
  // `r` is `f_batch_idx[p] + 1 + 2j + r`.
  let mut f_target_claims: Vec<OpenClaims<B::Scalar>> = Vec::with_capacity(states.len());
  let mut ab_target_claims: Vec<Vec<OpenClaims<B::Scalar>>> = Vec::with_capacity(states.len());
  let mut small_block_evals: Vec<Vec<B::Scalar>> = Vec::with_capacity(states.len());
  for (p, st) in states.iter().enumerate() {
    let d = BatchDims::new(1, st.f_limb.len(), params.log_t);
    let (fold_pt, alpha) = chunk_fold_point::<B::Scalar>(d.log_stride);
    let mut cl = OpenClaims::<B::Scalar>::default();
    for (z, y) in st.f_claims.points.iter().zip(st.f_claims.evals.iter()) {
      let mut zc = Vec::with_capacity(z.len() + fold_pt.len());
      zc.extend_from_slice(z);
      zc.extend_from_slice(&fold_pt);
      cl.push(zc, alpha * *y);
    }
    let (_, _, rc_claims) = &rc_art.chunk_data[f_batch_idx[p]];
    for (z, y) in rc_claims.points.iter().zip(rc_claims.evals.iter()) {
      cl.push(z.clone(), *y);
    }
    // Small-value block claims (zero-subcube gadget), transcript-ordered
    // after the range check, per poly, in declaration order.
    let poly_n = (st.f_limb.len().trailing_zeros() as usize) - st.numlimb_var;
    let blk_list: &[SmallValueBlock] = blocks.get(p).copied().unwrap_or(&[]);
    let mut blk_evals = Vec::with_capacity(blk_list.len());
    for blk in blk_list {
      let (bcl, e2) = small_block_claims::<B, _>(
        transcript,
        poly_n,
        st.numlimb_var,
        d.log_stride,
        blk,
        Some(rc_art.chunk_data[f_batch_idx[p]].0.as_slice()),
        None,
      )?;
      for (z, y) in bcl.points.into_iter().zip(bcl.evals) {
        cl.push(z, y);
      }
      blk_evals.push(e2);
    }
    small_block_evals.push(blk_evals);
    f_target_claims.push(cl);

    let mut per_layer = Vec::with_capacity(2 * st.t_layers);
    for (idx, ab_cl) in st.ab_claims.iter().enumerate() {
      let mut cl = ab_cl.clone();
      let (_, _, rc_claims) = &rc_art.chunk_data[f_batch_idx[p] + 1 + idx];
      for (z, y) in rc_claims.points.iter().zip(rc_claims.evals.iter()) {
        cl.push(z.clone(), *y);
      }
      per_layer.push(cl);
    }
    ab_target_claims.push(per_layer);
  }

  // ONE combined opening over all commitments — every target opens as
  // its chunk polynomial.
  let (_bo_span, bo_t) = start_span!("imod_pcs_batched_opens");
  let mut bsub = spawn_batch_subtranscript::<B, _>(transcript)?;
  let mut bo_targets: Vec<(
    &B::Comm,
    &[B::Scalar],
    &B::Blind,
    &B::Data,
    &OpenClaims<B::Scalar>,
  )> = Vec::new();
  // The input commitments were made through the ModPCS surface, which
  // returns no retained opening data — regenerate it (free for Hyrax,
  // a re-encode for Brakedown).
  let f_open_aux: Vec<B::Data> = states
    .iter()
    .enumerate()
    .map(|(p, _)| {
      B::recommit_data(
        backend_ck,
        comms[p],
        &rc_art.chunk_data[f_batch_idx[p]].0,
        blinds[p],
        true,
      )
    })
    .collect::<Result<Vec<_>, _>>()?;
  for (p, st) in states.iter().enumerate() {
    bo_targets.push((
      comms[p],
      rc_art.chunk_data[f_batch_idx[p]].0.as_slice(),
      blinds[p],
      &f_open_aux[p],
      &f_target_claims[p],
    ));
    for ((((comm, poly), blind), data), claims) in st
      .ab_comms
      .iter()
      .zip(st.ab_chunk_polys.iter())
      .zip(st.ab_blinds.iter())
      .zip(st.ab_open_aux.iter())
      .zip(ab_target_claims[p].iter())
    {
      bo_targets.push((comm, poly.as_slice(), blind, data, claims));
    }
  }
  bo_targets.push((
    &range_check.mult_comm,
    rc_art.mult_fq.as_slice(),
    &rc_art.mult_blind,
    &rc_art.mult_data,
    &rc_art.mult_claims,
  ));
  let combined_open = prove_combined_batch_open::<B>(backend_ck, &mut bsub, &bo_targets)?;
  info!(elapsed_ms = %bo_t.elapsed().as_millis(), "imod_pcs_batched_opens");
  drop(_bo_span);

  Ok((range_check, combined_open, small_block_evals))
}

/// Per-polynomial verifier state: the accumulated open claims plus the
/// public dimensions [`finish_batch_verify`] needs to pin batch shapes
/// and combined-open point lengths.
struct PerPolyVerifier<F = t256::Scalar> {
  /// This poly's own `numlimb_var` (see [`PerPolyProver::numlimb_var`]).
  numlimb_var: usize,
  f_claims: OpenClaims<F>,
  ab_claims: Vec<OpenClaims<F>>,
  num_vars: usize,
  t: usize,
  log_spad: usize,
}

/// Verifier mirror of [`prove_one_poly`]: reconstruct and verify the
/// reduction sumcheck and the per-prime chains (identity + CRT checks),
/// advancing `transcript` identically, and return the open claims and
/// dimensions for [`finish_batch_verify`].
#[allow(clippy::too_many_arguments)]
/// Verifier phase-1 state: everything derived while replaying the
/// commit phase (reduction checks, prime derivation, ab-comm absorbs)
/// that the claims phase consumes.
struct VerifyPhase1 {
  num_vars: usize,
  with_iter: bool,
  log_spad: usize,
  chain_primes: Vec<(BigUint, Vec<BigUint>)>,
}

fn verify_one_poly_phase1<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  transcript: &mut Keccak256Transcript<ME>,
  point: &[crate::dyn_prime::DynPrime<2>],
  eval: &BigUint,
  reduction_round_polys: &[Vec<BigUint>],
  int_v_prime: &BigInt,
  chains: &[ChainData<B::Scalar>],
  ab_comms: &[B::Comm],
) -> Result<VerifyPhase1, SpartanError> {
  let monty = point
    .first()
    .map(|p| *p.params())
    .ok_or(SpartanError::InternalError {
      reason: "IntegerModPCS::verify: empty point".to_string(),
    })?;

  let (_vred_span, vred_t) = start_span!("imod_pcs_verify_reduction_sc");
  if chains.len() != params.s {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  if reduction_round_polys.len() != params.numlimb_var {
    return Err(SpartanError::InvalidSumcheckProof);
  }

  let eval_p = <crate::dyn_prime::DynPrime<2> as SumcheckField>::from_bytes_reduce(
    &monty,
    &eval.to_bytes_le(),
  );
  let red_sc_polys: Vec<
    crate::polys_modp::univariate::CompressedUniPoly<crate::dyn_prime::DynPrime<2>>,
  > = reduction_round_polys
    .iter()
    .map(|coeffs| crate::polys_modp::univariate::CompressedUniPoly {
      coeffs_except_linear_term: coeffs
        .iter()
        .map(|b| {
          <crate::dyn_prime::DynPrime<2> as SumcheckField>::from_bytes_reduce(
            &monty,
            &b.to_bytes_le(),
          )
        })
        .collect(),
    })
    .collect();
  let red_sc = crate::sumcheck_modp::SumcheckProof::<ME> {
    compressed_polys: red_sc_polys,
  };
  let (red_final_claim, r_k) = red_sc.verify(eval_p, params.numlimb_var, 2, &monty, transcript)?;

  let limb_p = build_limb_weight_dynprime(params, &monty);
  let mut limb_mle = crate::polys_modp::multilinear::MultilinearPolynomial::new(limb_p, monty);
  for r in &r_k {
    limb_mle.bind_poly_var_top(r);
  }
  let limb_at_r_k = limb_mle.into_vec()[0];
  let limb_inv = <crate::dyn_prime::DynPrime<2> as SumcheckField>::invert(&limb_at_r_k)
    .ok_or(SpartanError::InvalidSumcheckProof)?;
  let f_eval_p = red_final_claim * limb_inv;

  let p = extract_p(point)?;
  let int_v_mod_p_u = int_v_prime
    .mod_floor(&BigInt::from(p.clone()))
    .to_biguint()
    .ok_or(SpartanError::InvalidSumcheckProof)?;
  let f_eval_bu = BigUint::from_bytes_le(&f_eval_p.to_le_bytes());
  if int_v_mod_p_u != f_eval_bu {
    return Err(SpartanError::InvalidSumcheckProof);
  }

  absorb_bigint(transcript, int_v_prime);

  let int_point_orig: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
  let r_k_int: Vec<BigUint> = r_k.iter().map(dyn_to_biguint).collect();
  let int_point: Vec<BigUint> = int_point_orig
    .iter()
    .chain(r_k_int.iter())
    .cloned()
    .collect();

  let num_vars = point.len() + params.numlimb_var;
  let with_iter = num_vars > params.k;
  let n = num_vars;
  let k = params.k;
  let t = if with_iter { (n - k).div_ceil(k) } else { 0 };
  info!(elapsed_ms = %vred_t.elapsed().as_millis(), "imod_pcs_verify_reduction_sc");
  drop(_vred_span);

  let _vchain_span = start_span!("imod_pcs_verify_chains").0;
  let primes: Vec<BigUint> = (0..params.s)
    .map(|_| sample_small_prime(transcript, params.log_p))
    .collect::<Result<Vec<_>, SpartanError>>()?;
  let mut chain_primes: Vec<(BigUint, Vec<BigUint>)> = Vec::with_capacity(params.s);
  for (chain, p_i) in chains.iter().zip(primes.iter()) {
    if chain.iterations.len() != t {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    let r_i_int: Vec<BigUint> = int_point.iter().map(|x| x % p_i).collect();
    chain_primes.push((p_i.clone(), r_i_int));
  }
  if ab_comms.len() != 2 * t {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  let s_pad = params.s.next_power_of_two();
  let log_spad = s_pad.trailing_zeros() as usize;
  for c in ab_comms {
    transcript.absorb_bytes(b"ab_chunk", &B::comm_transcript_bytes(c));
  }
  drop(_vchain_span);

  Ok(VerifyPhase1 {
    num_vars,
    with_iter,
    log_spad,
    chain_primes,
  })
}

/// Verifier phase 2: squeeze gammas, rebuild every claim, run the
/// per-layer identity and CRT checks.
fn verify_one_poly_phase2<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  transcript: &mut Keccak256Transcript<ME>,
  chains: &[ChainData<B::Scalar>],
  int_v_prime: &BigInt,
  ph1: VerifyPhase1,
) -> Result<PerPolyVerifier<B::Scalar>, SpartanError> {
  let VerifyPhase1 {
    num_vars,
    with_iter,
    log_spad,
    chain_primes,
  } = ph1;
  let n = num_vars;
  let k = params.k;
  let t = if with_iter { (n - k).div_ceil(k) } else { 0 };
  let (_vclaims_span, vchain_t) = start_span!("imod_pcs_verify_claims");

  let gamma_fq: Vec<B::Scalar> = if with_iter {
    (0..(n - k))
      .map(|i| {
        let bytes = transcript.squeeze_bytes(b"gamma")?;
        let label = (i as u64).to_le_bytes();
        transcript.absorb_bytes(b"gamma_idx", &label);
        Ok(<B::Scalar as PrimeFieldExt>::from_uniform(&bytes))
      })
      .collect::<Result<Vec<_>, SpartanError>>()?
  } else {
    Vec::new()
  };

  let shift_a_fq = biguint_to_scalar::<B::Scalar>(&shift_a(params));
  let shift_b_fq = biguint_to_scalar::<B::Scalar>(&shift_b::<B::Scalar>(params));

  let log_bound_a = params.log_p + 1;
  let log_bound_b = params.log_q - params.log_p + 1;
  let (fold_a, alpha_a) =
    chunk_fold_point::<B::Scalar>(chunk_stride(log_bound_a).trailing_zeros() as usize);
  let (fold_b, alpha_b) =
    chunk_fold_point::<B::Scalar>(chunk_stride(log_bound_b).trailing_zeros() as usize);
  let ab_chunk_point = |role: u8, chain: usize, sub: &[B::Scalar]| -> Vec<B::Scalar> {
    let fold = if role == 0 { &fold_a } else { &fold_b };
    let mut pt = Vec::with_capacity(log_spad + sub.len() + fold.len());
    pt.extend(bool_point_of_index::<B::Scalar>(chain, log_spad));
    pt.extend_from_slice(sub);
    pt.extend_from_slice(fold);
    pt
  };

  let mut f_claims = OpenClaims::<B::Scalar>::default();
  let mut ab_claims: Vec<OpenClaims<B::Scalar>> = (0..2 * t)
    .map(|_| OpenClaims::<B::Scalar>::default())
    .collect();
  for (chain_idx, chain) in chains.iter().enumerate() {
    let (p_i, r_i_int) = &chain_primes[chain_idx];
    let p_i_fq = biguint_to_scalar::<B::Scalar>(p_i);

    for (jm1, iter) in chain.iterations.iter().enumerate() {
      let j = jm1 + 1;
      let prefix_len = n - j * k;
      let lo = n - j * k;
      let hi = n - (j - 1) * k;
      let r_lower_fq: Vec<B::Scalar> = r_i_int[lo..hi]
        .iter()
        .map(biguint_to_scalar::<B::Scalar>)
        .collect();
      let gamma_prefix: Vec<B::Scalar> = gamma_fq[..prefix_len].to_vec();
      let gamma_extended: Vec<B::Scalar> = gamma_prefix
        .iter()
        .chain(r_lower_fq.iter())
        .copied()
        .collect();

      if j == 1 {
        f_claims.push(gamma_extended.clone(), iter.a_prev_eval);
      } else {
        ab_claims[2 * (jm1 - 1)].push(
          ab_chunk_point(0, chain_idx, &gamma_extended),
          alpha_a * iter.a_prev_eval,
        );
      }
      ab_claims[2 * jm1].push(
        ab_chunk_point(0, chain_idx, &gamma_prefix),
        alpha_a * iter.a_curr_eval,
      );
      ab_claims[2 * jm1 + 1].push(
        ab_chunk_point(1, chain_idx, &gamma_prefix),
        alpha_b * iter.b_curr_eval,
      );
      for ev in [&iter.a_prev_eval, &iter.a_curr_eval, &iter.b_curr_eval] {
        transcript.absorb_bytes(b"claim_ev", ev.to_repr().as_ref());
      }

      let lhs_a = iter.a_curr_eval - shift_a_fq;
      let lhs_b = iter.b_curr_eval - shift_b_fq;
      let lhs = lhs_a + p_i_fq * lhs_b;
      let rhs = if j == 1 {
        iter.a_prev_eval
      } else {
        iter.a_prev_eval - shift_a_fq
      };
      if lhs != rhs {
        return Err(SpartanError::InvalidSumcheckProof);
      }
    }

    let final_point_fq: Vec<B::Scalar> = r_i_int[..(n - t * k)]
      .iter()
      .map(biguint_to_scalar::<B::Scalar>)
      .collect();
    if t == 0 {
      f_claims.push(final_point_fq, chain.final_eval);
    } else {
      ab_claims[2 * (t - 1)].push(
        ab_chunk_point(0, chain_idx, &final_point_fq),
        alpha_a * chain.final_eval,
      );
    }
    transcript.absorb_bytes(b"claim_ev", chain.final_eval.to_repr().as_ref());

    let final_f = if t == 0 {
      chain.final_eval
    } else {
      chain.final_eval - shift_a_fq
    };
    let lhs = scalar_to_balanced_int(&final_f)
      .mod_floor(&BigInt::from(p_i.clone()))
      .to_biguint()
      .ok_or(SpartanError::InvalidSumcheckProof)?;
    let rhs = int_v_prime
      .mod_floor(&BigInt::from(p_i.clone()))
      .to_biguint()
      .ok_or(SpartanError::InvalidSumcheckProof)?;
    if lhs != rhs {
      return Err(SpartanError::InvalidSumcheckProof);
    }
  }
  info!(elapsed_ms = %vchain_t.elapsed().as_millis(), "imod_pcs_verify_chains");
  drop(_vclaims_span);

  Ok(PerPolyVerifier {
    numlimb_var: params.numlimb_var,
    f_claims,
    ab_claims,
    num_vars,
    t,
    log_spad,
  })
}

/// Single-polynomial verify: phase 1 then phase 2 back to back.
fn verify_one_poly<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  transcript: &mut Keccak256Transcript<ME>,
  point: &[crate::dyn_prime::DynPrime<2>],
  eval: &BigUint,
  reduction_round_polys: &[Vec<BigUint>],
  int_v_prime: &BigInt,
  chains: &[ChainData<B::Scalar>],
  ab_comms: &[B::Comm],
) -> Result<PerPolyVerifier<B::Scalar>, SpartanError> {
  let ph1 = verify_one_poly_phase1::<B, ME>(
    params,
    transcript,
    point,
    eval,
    reduction_round_polys,
    int_v_prime,
    chains,
    ab_comms,
  )?;
  verify_one_poly_phase2::<B, ME>(params, transcript, chains, int_v_prime, ph1)
}

fn finish_batch_verify<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  params: &IntEvalParams,
  backend_vk: &B::Vk,
  transcript: &mut Keccak256Transcript<ME>,
  comms: &[&B::Comm],
  verifiers: &mut [PerPolyVerifier<B::Scalar>],
  ab_comms_per_poly: &[&[B::Comm]],
  range_check: &SharedRangeCheck<B>,
  combined_open: &CombinedBatchOpen<B>,
  blocks: &[&[SmallValueBlock]],
  small_block_evals: &[Vec<B::Scalar>],
) -> Result<(), SpartanError>
where
  B::Scalar: crate::big_num::DelayedReduction<B::Scalar>,
{
  let log_bound_a = params.log_p + 1;
  let log_bound_b = params.log_q - params.log_p + 1;

  let (_vrc_span, vrc_t) = start_span!("imod_pcs_verify_rc");
  let mut rc_metas: Vec<RangeBatchMeta<'_, B>> = Vec::new();
  let mut f_batch_idx: Vec<usize> = Vec::with_capacity(verifiers.len());
  for (p, v) in verifiers.iter().enumerate() {
    f_batch_idx.push(rc_metas.len());
    rc_metas.push(RangeBatchMeta {
      target: RcTarget::F { poly: p },
      num_polys: 1,
      n_values: 1usize << v.num_vars,
      log_bound: params.log_t,
      precommitted_comm: Some(comms[p]),
    });
    for j in 0..v.t {
      let n_values = 1usize << (v.num_vars - (j + 1) * params.k);
      for (role, log_bound) in [(0u8, log_bound_a), (1u8, log_bound_b)] {
        rc_metas.push(RangeBatchMeta {
          target: RcTarget::Ab {
            poly: p,
            layer: j,
            role,
          },
          num_polys: params.s,
          n_values,
          log_bound,
          precommitted_comm: Some(&ab_comms_per_poly[p][2 * j + role as usize]),
        });
      }
    }
  }
  let rc_claims = verify_shared_range_check(&rc_metas, range_check, transcript)?;
  // Every batch is precommitted, so no value claims to route.
  debug_assert!(rc_claims.value_claims.is_empty());
  info!(elapsed_ms = %vrc_t.elapsed().as_millis(), "imod_pcs_verify_rc");
  drop(_vrc_span);

  // Per poly: fold every f_limb claim into a chunk claim on the input
  // commitment and append each target's GKR/top/zero-pad chunk claims
  // (mirror of the prover's claim assembly). `ab_claims` are already in
  // chunk coordinates.
  let log_stride_a = chunk_stride(log_bound_a).trailing_zeros() as usize;
  let log_stride_b = chunk_stride(log_bound_b).trailing_zeros() as usize;
  let mut f_target_claims: Vec<OpenClaims<B::Scalar>> = Vec::with_capacity(verifiers.len());
  let mut ab_target_claims: Vec<Vec<OpenClaims<B::Scalar>>> = Vec::with_capacity(verifiers.len());
  let mut f_log_stride: Vec<usize> = Vec::with_capacity(verifiers.len());
  for (p, v) in verifiers.iter().enumerate() {
    let d = BatchDims::new(1, 1usize << v.num_vars, params.log_t);
    let (fold_pt, alpha) = chunk_fold_point::<B::Scalar>(d.log_stride);
    let mut cl = OpenClaims::<B::Scalar>::default();
    for (z, y) in v.f_claims.points.iter().zip(v.f_claims.evals.iter()) {
      let mut zc = Vec::with_capacity(z.len() + fold_pt.len());
      zc.extend_from_slice(z);
      zc.extend_from_slice(&fold_pt);
      cl.push(zc, alpha * *y);
    }
    let rc_cl = &rc_claims.chunk_claims[f_batch_idx[p]];
    for (z, y) in rc_cl.points.iter().zip(rc_cl.evals.iter()) {
      cl.push(z.clone(), *y);
    }
    // Small-value block claims: mirror of the prover's assembly, with
    // `e2` read from the proof.
    let poly_n = v.num_vars - v.numlimb_var;
    let blk_list: &[SmallValueBlock] = blocks.get(p).copied().unwrap_or(&[]);
    for (bi, blk) in blk_list.iter().enumerate() {
      let e2 = small_block_evals
        .get(p)
        .and_then(|e| e.get(bi))
        .copied()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      let (bcl, _) = small_block_claims::<B, _>(
        transcript,
        poly_n,
        v.numlimb_var,
        d.log_stride,
        blk,
        None,
        Some(e2),
      )?;
      for (z, y) in bcl.points.into_iter().zip(bcl.evals) {
        cl.push(z, y);
      }
    }
    f_target_claims.push(cl);
    f_log_stride.push(d.log_stride);

    let mut per_layer = Vec::with_capacity(2 * v.t);
    for (idx, ab_cl) in v.ab_claims.iter().enumerate() {
      let mut cl = ab_cl.clone();
      let rc_cl = &rc_claims.chunk_claims[f_batch_idx[p] + 1 + idx];
      for (z, y) in rc_cl.points.iter().zip(rc_cl.evals.iter()) {
        cl.push(z.clone(), *y);
      }
      per_layer.push(cl);
    }
    ab_target_claims.push(per_layer);
  }

  let (_vbo_span, vbo_t) = start_span!("imod_pcs_verify_batched_opens");
  let mut bsub = spawn_batch_subtranscript::<B, _>(transcript)?;
  let mut bo_targets: Vec<(&B::Comm, usize, &OpenClaims<B::Scalar>)> = Vec::new();
  for (p, v) in verifiers.iter().enumerate() {
    bo_targets.push((comms[p], v.num_vars + f_log_stride[p], &f_target_claims[p]));
    for (idx, ab_comm) in ab_comms_per_poly[p].iter().enumerate() {
      let j = idx / 2;
      let log_stride = if idx % 2 == 0 {
        log_stride_a
      } else {
        log_stride_b
      };
      let m_vars = v.num_vars - (j + 1) * params.k;
      bo_targets.push((
        ab_comm,
        v.log_spad + m_vars + log_stride,
        &ab_target_claims[p][idx],
      ));
    }
  }
  bo_targets.push((&range_check.mult_comm, CHUNK_BITS, &rc_claims.mult_claims));
  verify_combined_batch_open::<B>(backend_vk, &mut bsub, &bo_targets, combined_open)?;
  info!(elapsed_ms = %vbo_t.elapsed().as_millis(), "imod_pcs_verify_batched_opens");
  drop(_vbo_span);
  Ok(())
}

/// Multilinear evaluation of `poly_fq` at point `r` over F. Mirrors the
/// dot-product form `sum_k chi(r, k) · poly[k]` used elsewhere.
fn mle_evaluate_fq<F: ff::PrimeField>(poly_fq: &[F], r: &[F]) -> F {
  let chis = EqPolynomial::evals_from_points(r);
  debug_assert_eq!(chis.len(), poly_fq.len());
  let mut acc = F::ZERO;
  for (c, v) in chis.iter().zip(poly_fq.iter()) {
    acc += *c * *v;
  }
  acc
}

/// Prover-side per-iteration state. Lives only during prove, never
/// serialized — holds the underlying F polynomial / blind / commitment
/// for both `a_j_shifted` and `b_j_shifted` so phase 2 can produce
/// openings at γ.
struct IterationProverState<F = t256::Scalar> {
  /// `a_j_shifted` as integers; kept so the range check can re-chunk
  /// without re-shifting / re-casting from F.
  a_shifted: Vec<BigUint>,
  a_shifted_fq: Vec<F>,
  /// `b_j_shifted` as integers (same reason).
  b_shifted: Vec<BigUint>,
  b_shifted_fq: Vec<F>,
}

/// Prover-side per-chain state collected in phase 1 and consumed in
/// phase 2.
struct ChainProverState<F = t256::Scalar> {
  r_i_int: Vec<BigUint>,
  iters: Vec<IterationProverState<F>>,
}

/// The Hyrax/Pedersen instantiation of the F-side backend seam:
/// preserves the pre-seam protocol byte-for-byte — the μ challenge, the
/// width split between the merged same-column IPA and individual
/// fallback opens, and every absorb happen in the exact order the
/// monolithic implementation used.
#[derive(Clone, Debug)]
pub struct HyBackend;

impl CommitBackend for HyBackend {
  type Scalar = t256::Scalar;
  type SE = T256HyraxEngine;
  type Ck = IntegerModCommitmentKey;
  type Vk = IntegerModVerifierKey;
  type Comm = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment;
  type Blind = <Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind;
  type Data = ();
  type BatchOpenArg = HyraxBatchOpenArg;

  fn blind(ck: &Self::Ck, n: usize) -> Self::Blind {
    Hyrax::blind(&ck.inner, n)
  }

  fn comm_transcript_bytes(comm: &Self::Comm) -> Vec<u8> {
    comm.to_transcript_bytes()
  }

  fn commit(
    ck: &Self::Ck,
    poly: &[t256::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<(Self::Comm, Self::Data), SpartanError> {
    Ok((Hyrax::commit(&ck.inner, poly, blind, small)?, ()))
  }

  fn recommit_data(
    _ck: &Self::Ck,
    _comm: &Self::Comm,
    _poly: &[t256::Scalar],
    _blind: &Self::Blind,
    _small: bool,
  ) -> Result<Self::Data, SpartanError> {
    Ok(())
  }

  fn open_targets(
    ck: &Self::Ck,
    targets: &[OpenTarget<'_, Self>],
    sub: &mut impl ByteTranscript,
  ) -> Result<Self::BatchOpenArg, SpartanError> {
    let mu_bytes = sub.squeeze_bytes(b"cbo_mu")?;
    let mu = <t256::Scalar as PrimeFieldExt>::from_uniform(&mu_bytes);

    // Merge every column-width-or-larger commitment into ONE same-column
    // IPA; open the rest (test sizes) individually.
    let width = Hyrax::key_num_cols(&ck.inner);
    let mut small_opens = Vec::new();
    let mut big_items: Vec<(
      &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
      &[t256::Scalar],
      &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
      Vec<t256::Scalar>,
    )> = Vec::new();
    let mut y_star = t256::Scalar::ZERO;
    let mut mu_pow = t256::Scalar::ONE;
    for t in targets {
      if t.poly.len() >= width {
        y_star += mu_pow * t.eval;
        mu_pow *= mu;
        big_items.push((t.comm, t.poly, t.blind, t.point.clone()));
      } else {
        small_opens.push(hyrax_open_at(
          &ck.inner, &ck.eval, sub, t.comm, t.poly, t.blind, &t.point,
        )?);
      }
    }
    let merged = if big_items.is_empty() {
      None
    } else {
      // `y_star` is recomputed by the verifier from the in-clear final
      // evaluations, so the eval commitment needs no hiding: use the
      // deterministic zero-blind `G^{y_star}`.
      let blind_eval = HyraxBlind::<T256HyraxEngine>::zero(&ck.eval, 1);
      let comm_eval = Hyrax::commit(&ck.eval, &[y_star], &blind_eval, false)?;
      let arg = Hyrax::prove_same_column_batch(
        &ck.inner,
        &ck.eval,
        sub,
        &big_items,
        mu,
        &comm_eval,
        &blind_eval,
      )?;
      Some(SmallPrimeOpening {
        f_y: y_star,
        hyrax_arg: arg,
      })
    };
    Ok(HyraxBatchOpenArg {
      merged,
      small_opens,
    })
  }

  fn verify_targets(
    vk: &Self::Vk,
    targets: &[(&Self::Comm, Vec<t256::Scalar>, t256::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError> {
    let mu_bytes = sub.squeeze_bytes(b"cbo_mu")?;
    let mu = <t256::Scalar as PrimeFieldExt>::from_uniform(&mu_bytes);

    let width = Hyrax::vk_num_cols(&vk.inner);
    let mut small_iter = arg.small_opens.iter();
    let mut big_items: Vec<(
      &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
      Vec<t256::Scalar>,
    )> = Vec::new();
    let mut y_star = t256::Scalar::ZERO;
    let mut mu_pow = t256::Scalar::ONE;
    for (comm, r_j, eval) in targets {
      if (1usize << r_j.len()) >= width {
        y_star += mu_pow * eval;
        mu_pow *= mu;
        big_items.push((comm, r_j.clone()));
      } else {
        let open = small_iter
          .next()
          .ok_or(SpartanError::InvalidSumcheckProof)?;
        hyrax_verify_open(&vk.inner, &vk.eval, sub, comm, r_j, open)?;
        if open.f_y != *eval {
          return Err(SpartanError::InvalidSumcheckProof);
        }
      }
    }
    if small_iter.next().is_some() {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    if big_items.is_empty() {
      if arg.merged.is_some() {
        return Err(SpartanError::InvalidSumcheckProof);
      }
    } else {
      let merged = arg
        .merged
        .as_ref()
        .ok_or(SpartanError::InvalidSumcheckProof)?;
      if merged.f_y != y_star {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      let zero_blind = HyraxBlind::<T256HyraxEngine>::zero(&vk.eval, 1);
      let comm_eval = Hyrax::commit(&vk.eval, &[merged.f_y], &zero_blind, false)?;
      Hyrax::verify_same_column_batch(
        &vk.inner,
        &vk.eval,
        sub,
        &big_items,
        mu,
        &comm_eval,
        &merged.hyrax_arg,
      )?;
    }
    Ok(())
  }
}

/// Helper: open the Hyrax commitment `comm` at `point` to produce a
/// `SmallPrimeOpening` (eval value + blind + Hyrax eval-argument). The
/// underlying polynomial `poly_fq` and its `blind` are inputs.
fn hyrax_open_at<T: ByteTranscript>(
  ck: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  transcript: &mut T,
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  poly_fq: &[t256::Scalar],
  blind: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Blind,
  point: &[t256::Scalar],
) -> Result<SmallPrimeOpening, SpartanError> {
  let f_y = mle_evaluate_fq(poly_fq, point);
  // f_y is sent in the clear in the proof, so the IPA's eval commitment
  // gains nothing from hiding. Use a zero blind and a deterministic
  // `comm_eval = G^{f_y}` that the verifier reconstructs locally; this
  // removes the per-open `Hyrax::blind` + 1-elem MSM and lets us drop
  // `blind_eval` from `SmallPrimeOpening`.
  let blind_eval = HyraxBlind::<T256HyraxEngine>::zero(ck_eval, 1);
  let comm_eval = Hyrax::commit(ck_eval, &[f_y], &blind_eval, false)?;
  let arg = Hyrax::prove(
    ck,
    ck_eval,
    transcript,
    comm,
    poly_fq,
    blind,
    point,
    &comm_eval,
    &blind_eval,
  )?;
  Ok(SmallPrimeOpening {
    f_y,
    hyrax_arg: arg,
  })
}

/// Mirror of `hyrax_open_at` on the verifier side: reconstruct
/// `comm_eval = G^{f_y}` (zero blind) and verify the Hyrax argument
/// against the polynomial commitment `comm` at `point`.
fn hyrax_verify_open<T: ByteTranscript>(
  vk: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::VerifierKey,
  ck_eval: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::CommitmentKey,
  transcript: &mut T,
  comm: &<Hyrax as PCSEngineTrait<T256HyraxEngine>>::Commitment,
  point: &[t256::Scalar],
  opening: &SmallPrimeOpening,
) -> Result<(), SpartanError> {
  let zero_blind = HyraxBlind::<T256HyraxEngine>::zero(ck_eval, 1);
  let comm_eval = Hyrax::commit(ck_eval, &[opening.f_y], &zero_blind, false)?;
  Hyrax::verify(
    vk,
    ck_eval,
    transcript,
    comm,
    point,
    &comm_eval,
    &opening.hyrax_arg,
  )
}

/// Which commitment a range-check batch's value claim targets. Routing
/// keys off the variant only; the payload identifies the batch in
/// `Debug` output and diagnostics.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum RcTarget {
  /// The input polynomial `f` of batch member `poly` (its `f_limb` batch).
  F { poly: usize },
  /// The stacked `(role, chain, x)` commitment of iteration layer `layer`
  /// (0-indexed) of batch member `poly`, restricted to `role` (0 = a,
  /// 1 = b).
  Ab { poly: usize, layer: usize, role: u8 },
}

/// Accumulated multi-point evaluation claims against one commitment.
#[derive(Default, Clone, Debug)]
struct OpenClaims<F = t256::Scalar> {
  points: Vec<Vec<F>>,
  evals: Vec<F>,
}

impl<F> OpenClaims<F> {
  fn push(&mut self, point: Vec<F>, eval: F) {
    self.points.push(point);
    self.evals.push(eval);
  }
}

/// Inputs for one homogeneous batch of the shared range check: `N` value
/// polynomials, all of length `n_values` (a power of two) and the same
/// bound `2^log_bound`. The value polynomials are NOT separately
/// committed — their `V(r_v)` evaluation becomes a claim on `target`.
struct RangeBatchInputs<'a, B: CommitBackend> {
  target: RcTarget,
  value_polys_fq: Vec<&'a [B::Scalar]>,
  values: Vec<&'a [BigUint]>,
  /// Coefficients per polynomial (same for all; a power of two).
  n_values: usize,
  /// Bit-width of the shared bound (each value `< 2^log_bound`).
  log_bound: usize,
  /// `Some((comm, blind))` when the batch's chunk polynomial is ALREADY
  /// committed as the target's own commitment (F batches: the Mod-PCS
  /// commitment IS the chunk polynomial). No fresh chunk commitment is
  /// made and no value-reconstruction sumcheck runs — the chunk→value
  /// relation is definitional via [`chunk_fold_point`] — and the GKR's
  /// chunk claims are discharged against this commitment.
  precommitted: Option<(&'a B::Comm, &'a B::Blind)>,
}

/// Verifier-side metadata for one batch of the shared range check.
struct RangeBatchMeta<'a, B: CommitBackend> {
  target: RcTarget,
  num_polys: usize,
  n_values: usize,
  log_bound: usize,
  /// Mirror of [`RangeBatchInputs::precommitted`]: the target's own
  /// commitment, for batches whose chunk polynomial is the committed
  /// representation itself (F batches).
  precommitted_comm: Option<&'a B::Comm>,
}

/// Sizes derived from a batch's public parameters, shared by prover and
/// verifier.
#[derive(Clone, Copy)]
struct BatchDims {
  log_np: usize,
  log_nv: usize,
  numchunks: usize,
  stride: usize,
  log_stride: usize,
  n_chunks: usize,
  /// Bit-width of the top chunk, `log_bound − 16·(numchunks−1)` ∈ [1, 16].
  rem: usize,
}

impl BatchDims {
  fn new(num_polys: usize, n_values: usize, log_bound: usize) -> Self {
    let n_pad = num_polys.next_power_of_two();
    let log_np = n_pad.trailing_zeros() as usize;
    let log_nv = ceil_log2(n_values.max(1));
    let numchunks = log_bound.div_ceil(CHUNK_BITS);
    // Min stride 2 keeps the reconstruction sumcheck non-degenerate when a
    // bound fits in a single chunk (the extra slot is zero-valued and
    // zero-weighted).
    let stride = chunk_stride(log_bound);
    Self {
      log_np,
      log_nv,
      numchunks,
      stride,
      log_stride: stride.trailing_zeros() as usize,
      n_chunks: n_pad * n_values * stride,
      rem: log_bound - CHUNK_BITS * (numchunks - 1),
    }
  }

  /// Whether the top chunk needs the shifted-lookup tightening.
  fn top_needed(&self) -> bool {
    self.rem < CHUNK_BITS
  }

  /// The public shift `2^16 − 2^rem` applied to top chunks so the same
  /// `2^16` table enforces `top < 2^rem`.
  fn top_shift(&self) -> u64 {
    (1u64 << CHUNK_BITS) - (1u64 << self.rem)
  }
}

/// Masked base-`2^16` weight vector for the value-reconstruction
/// sumcheck: `w[c] = 2^(16c)` for `c < ⌈log_bound/16⌉`, else `0`. Length
/// `stride` (the padded per-value chunk count). Chunk slots at
/// `c ≥ numchunks` carry zero weight, so the prover can't inflate a
/// value past its bound regardless of those (still range-checked) slots.
fn chunk_weight_vector<F: ff::PrimeField>(log_bound: usize, stride: usize) -> Vec<F> {
  let numchunks = log_bound.div_ceil(CHUNK_BITS);
  let base = F::from(1u64 << CHUNK_BITS);
  let mut weight = Vec::with_capacity(stride);
  let mut pow = F::ONE;
  for c in 0..stride {
    if c < numchunks {
      weight.push(pow);
      pow *= base;
    } else {
      weight.push(F::ZERO);
    }
  }
  weight
}

/// Spawn the F-side sub-transcript of the shared range check, seeded
/// from the parent and binding every batch's chunk commitment plus the
/// shared multiplicity commitment — all before any challenge (in
/// particular the LogUp `r`) is squeezed. Both prover and verifier
/// reconstruct it identically.
fn spawn_shared_range_subtranscript<
  'a,
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  parent: &mut Keccak256Transcript<ME>,
  chunk_comms: impl Iterator<Item = &'a B::Comm>,
  mult_comm: &B::Comm,
) -> Result<<B::SE as SumcheckEngine>::TE, SpartanError> {
  let seed = parent.squeeze_bytes(b"range_seed")?;
  let mut sub =
    <<B::SE as SumcheckEngine>::TE as TranscriptEngineTrait<B::SE>>::new(b"range_check");
  sub.absorb_bytes(b"seed", &seed);
  for cc in chunk_comms {
    sub.absorb_bytes(b"range_chunk_comm", &B::comm_transcript_bytes(cc));
  }
  sub.absorb_bytes(b"range_mult_comm", &B::comm_transcript_bytes(mult_comm));
  Ok(sub)
}

/// Spawn the F-side sub-transcript of the final batched-open phase.
/// Each commitment's claims are absorbed (and its λ squeezed) inside
/// [`prove_batched_open`] / [`verify_batched_open`].
fn spawn_batch_subtranscript<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  parent: &mut Keccak256Transcript<ME>,
) -> Result<<B::SE as SumcheckEngine>::TE, SpartanError> {
  let seed = parent.squeeze_bytes(b"batch_seed")?;
  let mut sub =
    <<B::SE as SumcheckEngine>::TE as TranscriptEngineTrait<B::SE>>::new(b"batched_opens");
  sub.absorb_bytes(b"seed", &seed);
  Ok(sub)
}

/// Absorb a commitment and its claims into the batch sub-transcript,
/// binding them before the RLC challenge λ is squeezed.
fn absorb_batch_claims<B: CommitBackend>(
  sub: &mut impl ByteTranscript,
  comm: &B::Comm,
  claims: &OpenClaims<B::Scalar>,
) {
  sub.absorb_bytes(b"bo_comm", &B::comm_transcript_bytes(comm));
  for (z, y) in claims.points.iter().zip(claims.evals.iter()) {
    for c in z {
      sub.absorb_bytes(b"bo_pt", c.to_repr().as_ref());
    }
    sub.absorb_bytes(b"bo_ev", y.to_repr().as_ref());
  }
}

/// Prove the combined multi-point opening (see [`CombinedBatchOpen`]).
/// `targets` are `(commitment, poly, blind, claims)` in canonical order.
/// Round polynomial evals `(e0, e2)` for one instance of the interleaved
/// combined open, via delayed reduction: accumulate the product sums
/// `e0 = Σ f0·w0` and `e2 = Σ (2f1-f0)(2w1-w0)` UNREDUCED and Montgomery-
/// reduce once per round instead of once per term (the BDDT round-0
/// pattern — heaviest in the large early rounds). Reduction is a ring
/// homomorphism so the result is identical to the per-term form. `e1` is
/// recovered by the caller as `run - e0`. Interior sparsity (all-zero
/// strided f-slots) is skipped as before.
fn interleaved_round_evals<B: CommitBackend>(
  fz: &[B::Scalar],
  wz: &[B::Scalar],
  h: usize,
) -> (B::Scalar, B::Scalar) {
  use crate::big_num::DelayedReduction;
  let zero = <B::Scalar as DelayedReduction<B::Scalar>>::Accumulator::default;
  let step = |acc: &mut (
    <B::Scalar as DelayedReduction<B::Scalar>>::Accumulator,
    <B::Scalar as DelayedReduction<B::Scalar>>::Accumulator,
  ),
              i: usize| {
    let (f0, f1) = (fz[i], fz[i + h]);
    let f0z = f0.is_zero_vartime();
    if f0z && f1.is_zero_vartime() {
      return;
    }
    let (w0, w1) = (wz[i], wz[i + h]);
    if !f0z {
      B::Scalar::unreduced_multiply_accumulate(&mut acc.0, &f0, &w0);
    }
    let a = f1 + f1 - f0;
    let b = w1 + w1 - w0;
    B::Scalar::unreduced_multiply_accumulate(&mut acc.1, &a, &b);
  };
  let (a0, a2) = if h >= 1 << 12 {
    (0..h)
      .into_par_iter()
      .fold(
        || (zero(), zero()),
        |mut acc, i| {
          step(&mut acc, i);
          acc
        },
      )
      .reduce(
        || (zero(), zero()),
        |mut x, y| {
          x.0 += y.0;
          x.1 += y.1;
          x
        },
      )
  } else {
    let mut acc = (zero(), zero());
    for i in 0..h {
      step(&mut acc, i);
    }
    acc
  };
  (
    <B::Scalar as DelayedReduction<B::Scalar>>::reduce(&a0),
    <B::Scalar as DelayedReduction<B::Scalar>>::reduce(&a2),
  )
}

fn prove_combined_batch_open<B: CommitBackend>(
  ck: &B::Ck,
  sub: &mut impl ByteTranscript,
  targets: &[(
    &B::Comm,
    &[B::Scalar],
    &B::Blind,
    &B::Data,
    &OpenClaims<B::Scalar>,
  )],
) -> Result<CombinedBatchOpen<B>, SpartanError> {
  use crate::polys::univariate::UniPoly;
  let m = targets.len();

  // 1. Per commitment: bind claims, squeeze λ, build W and the combined
  //    running claim.
  let (_bw_span, bw_t) = start_span!("bo_w_build");
  let mut eq_cache = EqTableCache::new();
  let mut fs = Vec::with_capacity(m);
  let mut ws = Vec::with_capacity(m);
  let mut run = Vec::with_capacity(m);
  let mut nv = Vec::with_capacity(m);
  for (comm, poly, _, _, claims) in targets {
    debug_assert!(!claims.points.is_empty());
    absorb_batch_claims::<B>(sub, comm, claims);
    let lambda = B::Scalar::from_uniform(&sub.squeeze_bytes(b"bo_lambda")?);
    let (w, c) = batch_weight(
      &claims.points,
      &claims.evals,
      lambda,
      poly.len(),
      &mut eq_cache,
    );
    fs.push(crate::polys::multilinear::MultilinearPolynomial::new(
      poly.to_vec(),
    ));
    ws.push(crate::polys::multilinear::MultilinearPolynomial::new(w));
    run.push(c);
    nv.push(poly.len().trailing_zeros() as usize);
  }
  let n_max = *nv.iter().max().expect("non-empty targets");
  info!(elapsed_ms = %bw_t.elapsed().as_millis(), m = %m, eq_builds = %eq_cache.entries.len(), eq_hits = %eq_cache.hits, "bo_w_build");
  drop(_bw_span);
  let (_bs_span, bs_t) = start_span!("bo_interleaved_sc");

  // 2. Interleaved, tail-aligned sumcheck rounds: per global round, all
  //    active instances absorb their round polynomial, then ONE shared
  //    challenge is squeezed and binds them all.
  let mut round_polys: Vec<Vec<crate::polys::univariate::CompressedUniPoly<B::Scalar>>> =
    vec![Vec::new(); m];
  let mut challenges: Vec<B::Scalar> = Vec::with_capacity(n_max);
  for g in 0..n_max {
    let mut staged: Vec<(usize, UniPoly<B::Scalar>)> = Vec::new();
    for j in 0..m {
      if g < n_max - nv[j] {
        continue; // joins later
      }
      let len = fs[j].Z.len();
      let h = len / 2;
      // Interior sparsity: skip zero-f terms (chunk-layout polynomials
      // have many strided zero slots — values narrower than their limb
      // budget, bit values, dropped regions).
      let (e0, e2) = interleaved_round_evals::<B>(&fs[j].Z, &ws[j].Z, h);
      let uni = UniPoly::from_evals(&[e0, run[j] - e0, e2])?;
      sub.absorb(b"cbo_p", &uni);
      staged.push((j, uni));
    }
    let r_g = B::Scalar::from_uniform(&sub.squeeze_bytes(b"cbo_c")?);
    for (j, uni) in staged {
      run[j] = uni.evaluate(&r_g);
      round_polys[j].push(uni.compress());
      fs[j].bind_poly_var_top(&r_g);
      ws[j].bind_poly_var_top(&r_g);
    }
    challenges.push(r_g);
  }

  info!(elapsed_ms = %bs_t.elapsed().as_millis(), "bo_interleaved_sc");
  drop(_bs_span);

  // 3. Send the per-commitment final evaluations; the backend
  //    discharges the resulting one-evaluation-per-commitment claims.
  let final_evals: Vec<B::Scalar> = fs.iter().map(|f| f[0]).collect();
  for y in &final_evals {
    sub.absorb_bytes(b"cbo_fe", y.to_repr().as_ref());
  }

  let (_bm_span, bm_t) = start_span!("bo_backend_open");
  let open_targets: Vec<OpenTarget<'_, B>> = targets
    .iter()
    .enumerate()
    .map(|(j, &(comm, poly, blind, data, _))| OpenTarget {
      comm,
      poly,
      blind,
      data,
      point: challenges[n_max - nv[j]..].to_vec(),
      eval: final_evals[j],
    })
    .collect();
  let backend = B::open_targets(ck, &open_targets, sub)?;
  info!(elapsed_ms = %bm_t.elapsed().as_millis(), "bo_backend_open");
  drop(_bm_span);

  Ok(CombinedBatchOpen {
    round_polys,
    final_evals,
    backend,
  })
}

/// Verifier mirror of [`prove_combined_batch_open`]. `targets` are
/// `(commitment, num_vars, claims)` in canonical order; every claim's
/// point length is pinned to its commitment's variable count.
fn verify_combined_batch_open<B: CommitBackend>(
  vk: &B::Vk,
  sub: &mut impl ByteTranscript,
  targets: &[(&B::Comm, usize, &OpenClaims<B::Scalar>)],
  arg: &CombinedBatchOpen<B>,
) -> Result<(), SpartanError> {
  let m = targets.len();
  if arg.round_polys.len() != m || arg.final_evals.len() != m {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  let mut lambdas = Vec::with_capacity(m);
  let mut run = Vec::with_capacity(m);
  let mut nv = Vec::with_capacity(m);
  for (j, (comm, num_vars, claims)) in targets.iter().enumerate() {
    if claims.points.is_empty() || claims.points.iter().any(|z| z.len() != *num_vars) {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    if arg.round_polys[j].len() != *num_vars {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    absorb_batch_claims::<B>(sub, comm, claims);
    let lambda = B::Scalar::from_uniform(&sub.squeeze_bytes(b"bo_lambda")?);
    let mut c = B::Scalar::ZERO;
    let mut lam_pow = B::Scalar::ONE;
    for y in &claims.evals {
      c += lam_pow * y;
      lam_pow *= lambda;
    }
    lambdas.push(lambda);
    run.push(c);
    nv.push(*num_vars);
  }
  let n_max = *nv.iter().max().expect("non-empty targets");

  let mut challenges: Vec<B::Scalar> = Vec::with_capacity(n_max);
  for g in 0..n_max {
    let mut staged: Vec<(usize, crate::polys::univariate::UniPoly<B::Scalar>)> = Vec::new();
    for j in 0..m {
      if g < n_max - nv[j] {
        continue;
      }
      let uni = arg.round_polys[j][g - (n_max - nv[j])].decompress(&run[j]);
      if uni.degree() != 2 {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      sub.absorb(b"cbo_p", &uni);
      staged.push((j, uni));
    }
    let r_g = B::Scalar::from_uniform(&sub.squeeze_bytes(b"cbo_c")?);
    for (j, uni) in staged {
      run[j] = uni.evaluate(&r_g);
    }
    challenges.push(r_g);
  }

  // Final sumcheck checks: run_j == f_j(r_j)·W_j(r_j), with W_j(r_j)
  // recomputed in closed form.
  for (j, (_, _, claims)) in targets.iter().enumerate() {
    let r_j = &challenges[n_max - nv[j]..];
    let mut w_at_r = B::Scalar::ZERO;
    let mut lam_pow = B::Scalar::ONE;
    for z in &claims.points {
      w_at_r += lam_pow * EqPolynomial::<B::Scalar>::new(z.clone()).evaluate(r_j);
      lam_pow *= lambdas[j];
    }
    if run[j] != arg.final_evals[j] * w_at_r {
      return Err(SpartanError::InvalidSumcheckProof);
    }
  }

  for y in &arg.final_evals {
    sub.absorb_bytes(b"cbo_fe", y.to_repr().as_ref());
  }

  let vt: Vec<(&B::Comm, Vec<B::Scalar>, B::Scalar)> = targets
    .iter()
    .enumerate()
    .map(|(j, (comm, _, _))| {
      (
        *comm,
        challenges[n_max - nv[j]..].to_vec(),
        arg.final_evals[j],
      )
    })
    .collect();
  B::verify_targets(vk, &vt, &arg.backend, sub)
}

/// Build `W = Σ_i λ^i·eq(z_i, ·)` (length `n`) and the combined claim
/// `Σ_i λ^i·y_i` for a batched open, exploiting claim-point structure so
/// the cost is far below the naive `#claims · n`:
///
/// - **Boolean head**: a claim whose leading coordinates are exactly
///   `0/1` (stacked-layer points `(role, bits(chain), ·)`, role-prefixed
///   range-check points) has `eq(z, ·)` supported on a single block —
///   write `λ^i·eq(tail)` into that block, cost `2^(n_vars − h)`.
/// - **Shared random prefix**: consecutive claims sharing a common
///   prefix (the `j=1` a_prev points `(γ_prefix, r^(c))`) are
///   tensor-combined: `eq(prefix) ⊗ Σ_c λ^c·eq(tail_c)` — one full-size
///   pass per *group* instead of per claim.
///
/// The output is bit-identical to the naive construction (the structure
/// only changes how the same table is assembled), so the transcript and
/// verifier are unaffected.
/// Memo for eq-evaluation tables keyed by their (tail) point. The
/// batched-open claims repeat a handful of distinct points many times —
/// every same-depth range-check claim tails off with the SAME shared
/// GKR leaf point, across lanes — so a linear-scan cache turns most
/// `2^k`-multiplication table builds into lookups.
struct EqTableCache<F> {
  entries: Vec<(Vec<F>, std::sync::Arc<Vec<F>>)>,
  hits: usize,
}

impl<F: ff::PrimeField> EqTableCache<F> {
  fn new() -> Self {
    Self {
      entries: Vec::new(),
      hits: 0,
    }
  }

  fn get(&mut self, point: &[F]) -> std::sync::Arc<Vec<F>> {
    if let Some((_, tbl)) = self.entries.iter().find(|(p, _)| p.as_slice() == point) {
      self.hits += 1;
      return tbl.clone();
    }
    let tbl = std::sync::Arc::new(EqPolynomial::<F>::evals_from_points(point));
    self.entries.push((point.to_vec(), tbl.clone()));
    tbl
  }
}

fn batch_weight<F: ff::PrimeField>(
  points: &[Vec<F>],
  evals: &[F],
  lambda: F,
  n: usize,
  cache: &mut EqTableCache<F>,
) -> (Vec<F>, F) {
  let n_vars = n.trailing_zeros() as usize;
  debug_assert_eq!(1usize << n_vars, n);
  let mut w = vec![F::ZERO; n];
  let mut claim = F::ZERO;
  let mut lam_pow = F::ONE;

  // Length of the leading run of exactly-boolean coordinates.
  let bool_head = |z: &[F]| -> usize {
    z.iter()
      .take_while(|c| **c == F::ZERO || **c == F::ONE)
      .count()
  };
  // Longest common prefix of two points.
  let lcp =
    |a: &[F], b: &[F]| -> usize { a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count() };

  let mut i = 0;
  while i < points.len() {
    let z = &points[i];
    debug_assert_eq!(z.len(), n_vars);
    let h = bool_head(z);
    if h > 0 {
      // Block write: λ^i·eq(z[h..]) into the block selected by z[..h].
      let mut block = 0usize;
      for c in &z[..h] {
        block = (block << 1) | usize::from(*c == F::ONE);
      }
      let tail = cache.get(&z[h..]);
      let bs = tail.len();
      let lam = lam_pow;
      w[block * bs..(block + 1) * bs]
        .par_iter_mut()
        .zip(tail.par_iter())
        .for_each(|(wj, e)| *wj += lam * e);
      claim += lam_pow * evals[i];
      lam_pow *= lambda;
      i += 1;
      continue;
    }

    // Group consecutive claims sharing a (random) prefix with z. Distinct
    // protocol points never collide by accident over a 256-bit field, so
    // any LCP > 0 identifies genuinely shared structure.
    let mut j = i + 1;
    let mut p = n_vars;
    while j < points.len() && lcp(z, &points[j]) > 0 {
      p = p.min(lcp(z, &points[j]));
      j += 1;
    }
    if j > i + 1 && p > 0 && p < n_vars {
      // Tensor: eq(z[..p]) ⊗ Σ_c λ^c·eq(z_c[p..]).
      let tail_len = 1usize << (n_vars - p);
      let mut small = vec![F::ZERO; tail_len];
      for c in i..j {
        debug_assert_eq!(&points[c][..p], &z[..p]);
        let tail = cache.get(&points[c][p..]);
        let lam = lam_pow;
        for (sj, e) in small.iter_mut().zip(tail.iter()) {
          *sj += lam * e;
        }
        claim += lam_pow * evals[c];
        lam_pow *= lambda;
      }
      let pref = EqPolynomial::<F>::evals_from_points(&z[..p]);
      w.par_chunks_mut(tail_len)
        .zip(pref.par_iter())
        .for_each(|(chunk, pe)| {
          for (wj, sj) in chunk.iter_mut().zip(small.iter()) {
            *wj += *pe * sj;
          }
        });
      i = j;
      continue;
    }

    // Singleton: one full-size accumulate.
    let eq_c = cache.get(z);
    let lam = lam_pow;
    w.par_iter_mut()
      .zip(eq_c.par_iter())
      .for_each(|(wj, e)| *wj += lam * e);
    claim += lam_pow * evals[i];
    lam_pow *= lambda;
    i += 1;
  }
  (w, claim)
}

/// Prover-side artifacts of the shared range check that feed the final
/// batched opens: value claims routed to `f` / stacked-layer targets,
/// per-batch chunk polynomials + blinds + claims, and the multiplicity
/// table's polynomial + blind + claim.
struct RcProverArtifacts<B: CommitBackend> {
  value_claims: Vec<(RcTarget, Vec<B::Scalar>, B::Scalar)>,
  chunk_data: Vec<(Vec<B::Scalar>, B::Blind, OpenClaims<B::Scalar>)>,
  mult_fq: Vec<B::Scalar>,
  mult_blind: B::Blind,
  /// Retained opening data for the multiplicity commitment.
  mult_data: B::Data,
  mult_claims: OpenClaims<B::Scalar>,
}

/// Claims the verifier of the shared range check hands back for the
/// batched-open verification.
struct RcVerifyClaims<F = t256::Scalar> {
  value_claims: Vec<(RcTarget, Vec<F>, F)>,
  chunk_claims: Vec<OpenClaims<F>>,
  mult_claims: OpenClaims<F>,
}

/// Prover side of the shared LogUp-GKR range check covering all batches
/// of one Mod-PCS opening. Per batch: build and commit the stacked chunk
/// polynomial. Shared: one multiplicity table and one multi-witness
/// LogUp whose witness trees are all batches' chunk polys plus the
/// shifted-top-chunk sub-polys of non-16-aligned batches. All
/// evaluation obligations (LogUp witness/table claims, top claims, the
/// per-batch `V(r_v)` value claims, and the reconstruction sumchecks'
/// final chunk evaluations) are returned as CLAIMS to be discharged by
/// the caller's batched opens — this function performs no Hyrax opens.
fn prove_shared_range_check<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  backend_ck: &B::Ck,
  batches: &[RangeBatchInputs<'_, B>],
  parent: &mut Keccak256Transcript<ME>,
) -> Result<(SharedRangeCheck<B>, RcProverArtifacts<B>), SpartanError>
where
  B::Scalar: crate::big_num::DelayedReduction<B::Scalar>,
{
  debug_assert!(!batches.is_empty());

  let dims: Vec<BatchDims> = batches
    .iter()
    .map(|b| BatchDims::new(b.value_polys_fq.len(), b.n_values, b.log_bound))
    .collect();

  // 1. Per batch: stacked chunk polynomial (u64 entries, each < 2^16).
  //    Index `((p·n_values + within)·stride + c)`. Padding polys
  //    (`p ≥ num_polys`) and slots `c ≥ numchunks` stay zero (zero is in
  //    the table, and those slots carry zero weight). Precommitted
  //    batches (F: the input commitment IS the chunk polynomial) build
  //    the chunk values for the GKR but make no fresh commitment.
  let (_rcc_span, rcc_t) = start_span!("rc_chunk_commit");
  let mut chunk_vals_all: Vec<Vec<u64>> = Vec::with_capacity(batches.len());
  let mut chunk_fq_all: Vec<Vec<B::Scalar>> = Vec::with_capacity(batches.len());
  let mut chunk_blinds: Vec<B::Blind> = Vec::with_capacity(batches.len());
  // `Some` for batches that committed a fresh chunk polynomial here
  // (these land in `SharedRangeCheck::batches`); `None` for
  // precommitted batches.
  let mut created_comms: Vec<Option<B::Comm>> = Vec::with_capacity(batches.len());
  for (b, d) in batches.iter().zip(dims.iter()) {
    let num_polys = b.value_polys_fq.len();
    debug_assert!(num_polys >= 1);
    debug_assert!(b.n_values.is_power_of_two());
    debug_assert!(b.values.iter().all(|v| v.len() == b.n_values));
    info!(
      num_polys = num_polys,
      n_values = b.n_values,
      log_bound = b.log_bound,
      stride = d.stride,
      n_chunks = d.n_chunks,
      "imod_pcs_range_batch"
    );
    let chunk_vals = build_chunk_poly(&b.values, b.n_values, b.log_bound);
    debug_assert_eq!(chunk_vals.len(), d.n_chunks);
    let chunk_fq: Vec<B::Scalar> = chunk_vals
      .par_iter()
      .map(|&c| scalar_from_chunk::<B::Scalar>(c))
      .collect();
    if let Some((_, blind)) = b.precommitted {
      chunk_blinds.push((*blind).clone());
      created_comms.push(None);
    } else {
      let blind = B::blind(backend_ck, d.n_chunks);
      let (comm, _data) = B::commit(backend_ck, &chunk_fq, &blind, true)?;
      chunk_blinds.push(blind);
      created_comms.push(Some(comm));
    }
    chunk_vals_all.push(chunk_vals);
    chunk_fq_all.push(chunk_fq);
  }
  // Canonical per-batch commitment references: the input commitment for
  // precommitted batches, the fresh chunk commitment otherwise.
  let chunk_comm_refs: Vec<&B::Comm> = batches
    .iter()
    .zip(created_comms.iter())
    .map(|(b, created)| match b.precommitted {
      Some((comm, _)) => comm,
      None => created.as_ref().expect("created for non-precommitted"),
    })
    .collect();

  info!(elapsed_ms = %rcc_t.elapsed().as_millis(), "rc_chunk_commit");
  drop(_rcc_span);

  // 2. Shifted top chunks of the non-16-aligned batches: `top + (2^16 −
  //    2^rem)` is in the 2^16 table iff `top < 2^rem`. These become extra
  //    LogUp witness trees; no extra commitment (their MLE is the chunk
  //    MLE at a boolean-extended point, plus the public shift).
  let mut top_vals_all: Vec<(usize, Vec<u64>)> = Vec::new(); // (batch idx, shifted tops)
  for (bi, d) in dims.iter().enumerate() {
    if d.top_needed() {
      let shift = d.top_shift();
      let stride = d.stride;
      let tops: Vec<u64> = (0..d.n_chunks / stride)
        .map(|gv| chunk_vals_all[bi][gv * stride + (d.numchunks - 1)] + shift)
        .collect();
      top_vals_all.push((bi, tops));
    }
  }

  // 2b. Active-block maps: all-zero dyadic blocks leave the multiset
  //     and are pinned to zero by direct opening claims below. Padded
  //     rows/polys make these regions large in practice.
  let mut active_blocks: Vec<Vec<bool>> = chunk_vals_all
    .iter()
    .map(|cv| {
      let (block_log, n_blocks) = rc_block_split(cv.len());
      (0..n_blocks)
        .map(|b| {
          cv[(b << block_log)..((b + 1) << block_log)]
            .iter()
            .any(|&v| v != 0)
        })
        .collect()
    })
    .collect();
  // Degenerate all-zero input: keep one live tree so the LogUp argument
  // is well-formed (a zero block is valid multiset input).
  if top_vals_all.is_empty() && active_blocks.iter().all(|a| a.iter().all(|&x| !x)) {
    active_blocks[0][0] = true;
  }

  // 3. The shared multiplicity table over the ACTIVE witness trees,
  //    committed before the LogUp challenge `r` is squeezed
  //    (multiplicities chosen after `r` would break the lookup identity).
  let witness_refs: Vec<&[u64]> = chunk_vals_all
    .iter()
    .zip(active_blocks.iter())
    .flat_map(|(cv, act)| {
      let (block_log, _) = rc_block_split(cv.len());
      act
        .iter()
        .enumerate()
        .filter(|(_, a)| **a)
        .map(move |(b, _)| &cv[(b << block_log)..((b + 1) << block_log)])
    })
    .chain(top_vals_all.iter().map(|(_, v)| v.as_slice()))
    .collect();
  let (_rcm_span, rcm_t) = start_span!("rc_mult_commit");
  let mult =
    crate::logup_gkr::LogUpMultiRangeProof::<B::SE>::multiplicities(CHUNK_BITS, &witness_refs)?;
  let mult_fq: Vec<B::Scalar> = mult.iter().map(|&m| B::Scalar::from(m)).collect();
  let mult_blind = B::blind(backend_ck, mult_fq.len());
  let (mult_comm, mult_data) = B::commit(backend_ck, &mult_fq, &mult_blind, true)?;
  info!(elapsed_ms = %rcm_t.elapsed().as_millis(), "rc_mult_commit");
  drop(_rcm_span);

  // 4. Sub-transcript bound to (parent, chunk comms, mult comm), plus
  //    the active-block maps (prover advice fixed before any challenge).
  let mut sub =
    spawn_shared_range_subtranscript::<B, ME>(parent, chunk_comm_refs.iter().copied(), &mult_comm)?;
  for act in &active_blocks {
    sub.absorb_bytes(b"rc_active", &pack_bitmap(act));
  }

  // 5. ONE multi-witness LogUp-GKR: every entry of every ACTIVE tree is in
  //    [0, 2^16). Its reduced claims become batched-open claims.
  let (_rcl_span, rcl_t) = start_span!("rc_logup_gkr");
  let (logup, claims) =
    crate::logup_gkr::LogUpMultiRangeProof::<B::SE>::prove(CHUNK_BITS, &witness_refs, &mut sub)?;
  info!(elapsed_ms = %rcl_t.elapsed().as_millis(), "rc_logup_gkr");
  drop(_rcl_span);
  let mut chunk_claims: Vec<OpenClaims<B::Scalar>> =
    vec![OpenClaims::<B::Scalar>::default(); batches.len()];
  let mut wc = claims.wit_claims.iter();
  for (bi, act) in active_blocks.iter().enumerate() {
    let n_chunk_vars = ceil_log2(chunk_vals_all[bi].len().max(1));
    let (block_log, _) = rc_block_split(chunk_vals_all[bi].len());
    for (blk, &a) in act.iter().enumerate() {
      if !a {
        continue;
      }
      let (point, eval) = wc.next().expect("one claim per active block");
      let full: Vec<B::Scalar> = bool_point_of_index::<B::Scalar>(blk, n_chunk_vars - block_log)
        .into_iter()
        .chain(point.iter().copied())
        .collect();
      chunk_claims[bi].push(full, *eval);
    }
  }
  for (bi, _) in top_vals_all.iter() {
    let d = &dims[*bi];
    let (point, eval) = wc.next().expect("one claim per top tree");
    let ext: Vec<B::Scalar> = point
      .iter()
      .copied()
      .chain(bool_point_of_index::<B::Scalar>(
        d.numchunks - 1,
        d.log_stride,
      ))
      .collect();
    chunk_claims[*bi].push(ext, *eval - B::Scalar::from(d.top_shift()));
  }
  // Inactive blocks: pin each to zero at ONE shared random point per
  // batch (distinct boolean block prefixes keep the claims independent;
  // per-block Schwartz–Zippel is unaffected by sharing the challenge).
  // Sharing lets the batch-open eq-table cache serve every inactive
  // block of a batch from one build.
  for (bi, act) in active_blocks.iter().enumerate() {
    let n_chunk_vars = ceil_log2(chunk_vals_all[bi].len().max(1));
    let (block_log, _) = rc_block_split(chunk_vals_all[bi].len());
    let mut r_blk: Option<Vec<B::Scalar>> = None;
    for (blk, &a) in act.iter().enumerate() {
      if a {
        continue;
      }
      if r_blk.is_none() {
        r_blk = Some(
          (0..block_log)
            .map(|_| sub.squeeze(b"range_zblk"))
            .collect::<Result<Vec<_>, _>>()?,
        );
      }
      let full: Vec<B::Scalar> = bool_point_of_index::<B::Scalar>(blk, n_chunk_vars - block_log)
        .into_iter()
        .chain(r_blk.as_ref().expect("just set").iter().copied())
        .collect();
      chunk_claims[bi].push(full, B::Scalar::ZERO);
    }
  }
  let mut mult_claims = OpenClaims::<B::Scalar>::default();
  mult_claims.push(claims.mult_point.clone(), claims.mult_eval);

  // 6. Per non-precommitted batch: value-reconstruction sumcheck tying
  //    the chunks to the batch's value polynomials. The folded value
  //    `V(r_v) = Σ_p eq(r_v_poly, p)·value_p(r_v_within)` is sent as a
  //    claim on the batch's target commitment, and the sumcheck's final
  //    chunk evaluation is a claim on the chunk commitment.
  //    Precommitted batches skip this entirely: their chunk polynomial
  //    IS the committed representation, so the chunk→value relation is
  //    definitional (`chunk_fold_point`) — nothing to tie.
  let (_rcr_span, rcr_t) = start_span!("rc_reconstr");
  let mut value_claims: Vec<(RcTarget, Vec<B::Scalar>, B::Scalar)> =
    Vec::with_capacity(batches.len());
  let mut batch_data: Vec<RangeCheckBatchData<B>> = Vec::with_capacity(batches.len());
  for (bi, (b, d)) in batches.iter().zip(dims.iter()).enumerate() {
    if b.precommitted.is_some() {
      // Zero-pad claims: the fold (`chunk_fold_point`) weighs EVERY
      // slot, so padding slots `c ≥ numchunks` must be pinned to zero.
      // A fresh random-point opening claim per padding slot does it
      // (Schwartz–Zippel: a nonzero multilinear restriction survives a
      // random evaluation with negligible probability).
      if d.stride > d.numchunks {
        let r_pad: Vec<B::Scalar> = (0..(d.log_np + d.log_nv))
          .map(|_| sub.squeeze(b"range_zpad"))
          .collect::<Result<Vec<_>, _>>()?;
        for c in d.numchunks..d.stride {
          let pt: Vec<B::Scalar> = r_pad
            .iter()
            .copied()
            .chain(bool_point_of_index::<B::Scalar>(c, d.log_stride))
            .collect();
          chunk_claims[bi].push(pt, B::Scalar::ZERO);
        }
      }
      continue;
    }
    let num_polys = b.value_polys_fq.len();
    let r_v: Vec<B::Scalar> = (0..(d.log_np + d.log_nv))
      .map(|_| sub.squeeze(b"range_rv"))
      .collect::<Result<Vec<_>, _>>()?;
    let r_v_poly = &r_v[..d.log_np];
    let r_v_within = &r_v[d.log_np..];

    let eq_weights = EqPolynomial::<B::Scalar>::new(r_v_poly.to_vec()).evals();
    let w = &eq_weights[..num_polys];
    let mut combined_poly = vec![B::Scalar::ZERO; b.n_values];
    for (p, poly) in b.value_polys_fq.iter().enumerate() {
      for (o, &v) in combined_poly.iter_mut().zip(poly.iter()) {
        *o += w[p] * v;
      }
    }
    let value_eval = mle_evaluate_fq(&combined_poly, r_v_within);
    sub.absorb_bytes(b"rc_value_ev", value_eval.to_repr().as_ref());
    value_claims.push((b.target, r_v.clone(), value_eval));

    // Partial-eval chunk poly at r_v, leaving the chunk axis.
    let mut chunk_mle =
      crate::polys::multilinear::MultilinearPolynomial::new(chunk_fq_all[bi].clone());
    for r in &r_v {
      chunk_mle.bind_poly_var_top(r);
    }
    let chunk_at_rv: Vec<B::Scalar> = chunk_mle.into_vec();
    debug_assert_eq!(chunk_at_rv.len(), d.stride);

    let weight = chunk_weight_vector::<B::Scalar>(b.log_bound, d.stride);
    let mut poly_w = crate::polys::multilinear::MultilinearPolynomial::new(weight);
    let mut poly_c = crate::polys::multilinear::MultilinearPolynomial::new(chunk_at_rv);
    let (value_reconstr_sumcheck, r_b, sc_claims) =
      crate::sumcheck::SumcheckProof::<B::SE>::prove_quad(
        &value_eval,
        d.log_stride,
        &mut poly_w,
        &mut poly_c,
        &mut sub,
      )?;
    let reconstr_eval = sc_claims[1];
    sub.absorb_bytes(b"rc_reconstr_ev", reconstr_eval.to_repr().as_ref());
    let combined: Vec<B::Scalar> = r_v.iter().chain(r_b.iter()).copied().collect();
    chunk_claims[bi].push(combined, reconstr_eval);

    batch_data.push(RangeCheckBatchData {
      chunk_comm: created_comms[bi]
        .clone()
        .expect("non-precommitted batch has a fresh chunk commitment"),
      value_eval,
      reconstr_eval,
      value_reconstr_sumcheck,
    });
  }

  info!(elapsed_ms = %rcr_t.elapsed().as_millis(), "rc_reconstr");
  drop(_rcr_span);

  let chunk_data: Vec<(Vec<B::Scalar>, B::Blind, OpenClaims<B::Scalar>)> = chunk_fq_all
    .into_iter()
    .zip(chunk_blinds)
    .zip(chunk_claims)
    .map(|((fq, blind), claims)| (fq, blind, claims))
    .collect();

  Ok((
    SharedRangeCheck {
      mult_comm,
      logup,
      batches: batch_data,
      active_blocks,
    },
    RcProverArtifacts {
      value_claims,
      chunk_data,
      mult_fq,
      mult_blind,
      mult_data,
      mult_claims,
    },
  ))
}

/// Verifier-side mirror of `prove_shared_range_check`. Re-derives the
/// transcript, verifies the multi-witness LogUp (with all tree depths
/// pinned to the public batch shapes), re-runs each batch's
/// reconstruction sumcheck against the claimed evaluations, and returns
/// the claims for the batched-open verification.
fn verify_shared_range_check<
  B: CommitBackend,
  ME: crate::traits::mod_engine::ModEngine<
      Scalar = crate::dyn_prime::DynPrime<2>,
      TE = Keccak256Transcript<ME>,
    >,
>(
  metas: &[RangeBatchMeta<'_, B>],
  arg: &SharedRangeCheck<B>,
  parent: &mut Keccak256Transcript<ME>,
) -> Result<RcVerifyClaims<B::Scalar>, SpartanError>
where
  B::Scalar: crate::big_num::DelayedReduction<B::Scalar>,
{
  // The proof carries per-batch data only for batches that committed a
  // fresh chunk polynomial; precommitted batches (F) contribute their
  // target's own commitment and skip reconstruction.
  let n_fresh = metas
    .iter()
    .filter(|m| m.precommitted_comm.is_none())
    .count();
  if metas.is_empty() || arg.batches.len() != n_fresh {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  let dims: Vec<BatchDims> = metas
    .iter()
    .map(|m| BatchDims::new(m.num_polys, m.n_values, m.log_bound))
    .collect();
  // Canonical per-batch commitment references (input comm for
  // precommitted batches, proof-carried chunk comm otherwise).
  let chunk_comm_refs: Vec<&B::Comm> = {
    let mut fresh = arg.batches.iter();
    metas
      .iter()
      .map(|m| match m.precommitted_comm {
        Some(comm) => comm,
        None => &fresh.next().expect("length checked above").chunk_comm,
      })
      .collect()
  };

  // Tree-depth pinning: one tree per ACTIVE chunk block (block size and
  // count fixed by the public batch shape; the active map is prover
  // advice policed by the zero claims below), then shifted-top trees of
  // the non-aligned batches.
  let active_blocks = &arg.active_blocks;
  if active_blocks.len() != metas.len() {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  let mut expected_depths: Vec<usize> = Vec::new();
  for (d, act) in dims.iter().zip(active_blocks.iter()) {
    let (block_log, n_blocks) = rc_block_split(d.n_chunks);
    if act.len() != n_blocks {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    for &a in act.iter() {
      if a {
        expected_depths.push(block_log);
      }
    }
  }
  let mut top_batches: Vec<usize> = Vec::new();
  for (bi, d) in dims.iter().enumerate() {
    if d.top_needed() {
      expected_depths.push(d.log_np + d.log_nv);
      top_batches.push(bi);
    }
  }

  // 1. Spawn the same sub-transcript the prover used, absorbing the
  //    active-block maps at the same point.
  let mut sub = spawn_shared_range_subtranscript::<B, ME>(
    parent,
    chunk_comm_refs.iter().copied(),
    &arg.mult_comm,
  )?;
  for act in active_blocks.iter() {
    sub.absorb_bytes(b"rc_active", &pack_bitmap(act));
  }

  // 2. Multi-witness LogUp membership: every chunk (and shifted top) in
  //    [0, 2^16). Its reduced claims become batched-open claims.
  let claims = arg.logup.verify(CHUNK_BITS, &expected_depths, &mut sub)?;
  let mut chunk_claims: Vec<OpenClaims<B::Scalar>> =
    vec![OpenClaims::<B::Scalar>::default(); metas.len()];
  let mut wc = claims.wit_claims.iter();
  for (bi, (d, act)) in dims.iter().zip(active_blocks.iter()).enumerate() {
    let n_chunk_vars = ceil_log2(d.n_chunks.max(1));
    let (block_log, _) = rc_block_split(d.n_chunks);
    for (blk, &a) in act.iter().enumerate() {
      if !a {
        continue;
      }
      let (point, eval) = wc.next().ok_or(SpartanError::InvalidSumcheckProof)?;
      let full: Vec<B::Scalar> = bool_point_of_index::<B::Scalar>(blk, n_chunk_vars - block_log)
        .into_iter()
        .chain(point.iter().copied())
        .collect();
      chunk_claims[bi].push(full, *eval);
    }
  }
  for &bi in top_batches.iter() {
    let d = &dims[bi];
    let (point, eval) = wc.next().ok_or(SpartanError::InvalidSumcheckProof)?;
    let ext: Vec<B::Scalar> = point
      .iter()
      .copied()
      .chain(bool_point_of_index::<B::Scalar>(
        d.numchunks - 1,
        d.log_stride,
      ))
      .collect();
    chunk_claims[bi].push(ext, *eval - B::Scalar::from(d.top_shift()));
  }
  // Inactive blocks: same shared-per-batch random-point zero claims as
  // the prover.
  for (bi, (d, act)) in dims.iter().zip(active_blocks.iter()).enumerate() {
    let n_chunk_vars = ceil_log2(d.n_chunks.max(1));
    let (block_log, _) = rc_block_split(d.n_chunks);
    let mut r_blk: Option<Vec<B::Scalar>> = None;
    for (blk, &a) in act.iter().enumerate() {
      if a {
        continue;
      }
      if r_blk.is_none() {
        r_blk = Some(
          (0..block_log)
            .map(|_| sub.squeeze(b"range_zblk"))
            .collect::<Result<Vec<_>, _>>()?,
        );
      }
      let full: Vec<B::Scalar> = bool_point_of_index::<B::Scalar>(blk, n_chunk_vars - block_log)
        .into_iter()
        .chain(r_blk.as_ref().expect("just set").iter().copied())
        .collect();
      chunk_claims[bi].push(full, B::Scalar::ZERO);
    }
  }
  let mut mult_claims = OpenClaims::<B::Scalar>::default();
  mult_claims.push(claims.mult_point.clone(), claims.mult_eval);

  // 3. Per non-precommitted batch: r_v squeeze, claimed `V(r_v)`
  //    absorbed and recorded as a claim on the batch's target,
  //    reconstruction sumcheck verified against it, and the final
  //    integrand check `w(r_b)·chunk(r_v, r_b) == final claim` using the
  //    claimed chunk evaluation (itself a claim on the chunk
  //    commitment). Precommitted batches (F) skip reconstruction — the
  //    chunk→value relation is definitional via `chunk_fold_point`.
  let mut value_claims: Vec<(RcTarget, Vec<B::Scalar>, B::Scalar)> =
    Vec::with_capacity(metas.len());
  let mut fresh_i = 0usize;
  for (bi, (m, d)) in metas.iter().zip(dims.iter()).enumerate() {
    if m.precommitted_comm.is_some() {
      // Mirror of the prover's zero-pad claims (see
      // `prove_shared_range_check`).
      if d.stride > d.numchunks {
        let r_pad: Vec<B::Scalar> = (0..(d.log_np + d.log_nv))
          .map(|_| sub.squeeze(b"range_zpad"))
          .collect::<Result<Vec<_>, _>>()?;
        for c in d.numchunks..d.stride {
          let pt: Vec<B::Scalar> = r_pad
            .iter()
            .copied()
            .chain(bool_point_of_index::<B::Scalar>(c, d.log_stride))
            .collect();
          chunk_claims[bi].push(pt, B::Scalar::ZERO);
        }
      }
      continue;
    }
    let b = &arg.batches[fresh_i];
    fresh_i += 1;
    let r_v: Vec<B::Scalar> = (0..(d.log_np + d.log_nv))
      .map(|_| sub.squeeze(b"range_rv"))
      .collect::<Result<Vec<_>, _>>()?;
    sub.absorb_bytes(b"rc_value_ev", b.value_eval.to_repr().as_ref());
    value_claims.push((m.target, r_v.clone(), b.value_eval));

    let (vr_final_claim, r_b) =
      b.value_reconstr_sumcheck
        .verify(b.value_eval, d.log_stride, 2, &mut sub)?;
    sub.absorb_bytes(b"rc_reconstr_ev", b.reconstr_eval.to_repr().as_ref());

    let mut w_poly = crate::polys::multilinear::MultilinearPolynomial::new(chunk_weight_vector(
      m.log_bound,
      d.stride,
    ));
    for r in &r_b {
      w_poly.bind_poly_var_top(r);
    }
    let w_at_rb = w_poly.into_vec()[0];
    if vr_final_claim != w_at_rb * b.reconstr_eval {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    let combined: Vec<B::Scalar> = r_v.iter().chain(r_b.iter()).copied().collect();
    chunk_claims[bi].push(combined, b.reconstr_eval);
  }

  Ok(RcVerifyClaims {
    value_claims,
    chunk_claims,
    mult_claims,
  })
}

/// Absorb a `BigInt` into a `ByteTranscript` as `(sign_byte, LE
/// magnitude bytes)`. Sign byte is `0` for non-negative, `1` for
/// negative. Length-prefixed by usize → 8 bytes LE so re-derivation is
/// unambiguous.
fn absorb_bigint<T: ByteTranscript>(transcript: &mut T, x: &BigInt) {
  let sign_byte: u8 = match x.sign() {
    Sign::Minus => 1,
    _ => 0,
  };
  let mag = x.magnitude().to_bytes_le();
  let mut buf = Vec::with_capacity(1 + 8 + mag.len());
  buf.push(sign_byte);
  buf.extend_from_slice(&(mag.len() as u64).to_le_bytes());
  buf.extend_from_slice(&mag);
  transcript.absorb_bytes(b"int_v_prime", &buf);
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    dyn_prime::DynPrime,
    traits::{mod_engine::SumcheckField, transcript::TranscriptEngineTrait},
  };

  type ME = T256DynPrimeEngine;
  type MP = IntegerModPCS;
  type DP = DynPrime<2>;

  /// Setup + commit round-trip: an IntEval-committed polynomial commits
  /// to the same Hyrax handle as a direct Hyrax commit of its
  /// limb-split, base-2^16-chunked representation (the committed-chunk
  /// layout). Sanity check that the wrapper isn't mangling the
  /// underlying commitment.
  #[test]
  fn commit_delegates_to_hyrax() {
    let n = 16usize;
    let (ck, _vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-test", n, 256);
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(7u32 * i as u32 + 3)).collect();
    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    // Re-commit the chunk layout directly via Hyrax and confirm equality.
    let limbs = limb_split_polynomial(&poly, ck.params.log_t, ck.params.log_t_f);
    let chunks = build_chunk_poly(&[&limbs], limbs.len(), ck.params.log_t);
    let chunk_fq: Vec<t256::Scalar> = chunks.iter().map(|&c| scalar_from_chunk(c)).collect();
    let direct = Hyrax::commit(&ck.inner, &chunk_fq, &blind.inner, true).unwrap();
    assert_eq!(comm.inner, direct);
  }

  /// The chunk-axis folding identity behind the committed-chunk
  /// representation: `chunk_mle(z ++ x_*) = α · limb_mle(z)` at
  /// arbitrary (non-boolean) `z`, across limb widths including the
  /// single-chunk (dead-bit) and multi-chunk tensor cases.
  #[test]
  fn chunk_fold_point_matches_limb_evaluation() {
    for (log_t, log_t_f) in [(8usize, 8usize), (16, 64), (32, 64), (64, 256), (128, 512)] {
      let n = 8usize;
      let mask = (BigUint::one() << log_t_f) - BigUint::one();
      let poly: Vec<BigUint> = (0..n)
        .map(|i| ((BigUint::from(0x9e3779b97f4a7c15u64) << (7 * i)) + BigUint::from(i)) & &mask)
        .collect();
      let limbs = limb_split_polynomial(&poly, log_t, log_t_f);
      let chunks = build_chunk_poly(&[&limbs], limbs.len(), log_t);
      let limbs_fq: Vec<t256::Scalar> = limbs.iter().map(biguint_to_scalar).collect();
      let chunk_fq: Vec<t256::Scalar> = chunks.iter().map(|&c| scalar_from_chunk(c)).collect();
      let d = BatchDims::new(1, limbs.len(), log_t);
      let (fold_pt, alpha) = chunk_fold_point(d.log_stride);
      let nv = limbs.len().trailing_zeros() as usize;
      let z: Vec<t256::Scalar> = (0..nv)
        .map(|i| t256::Scalar::from(0xACE1u64 + 77 * i as u64))
        .collect();
      let zc: Vec<t256::Scalar> = z.iter().copied().chain(fold_pt.iter().copied()).collect();
      assert_eq!(
        mle_evaluate_fq(&chunk_fq, &zc),
        alpha * mle_evaluate_fq(&limbs_fq, &z),
        "fold identity failed for log_t={log_t}, log_t_f={log_t_f}"
      );
    }
  }

  /// The fixed-width chain arithmetic (`I256` partial eval + divmod)
  /// agrees with the `BigInt` path on mixed-sign, wide (≈190-bit)
  /// values — the layer-1 (non-negative limbs) and layer-≥2 (signed
  /// remainders) shapes both included.
  #[test]
  fn i256_partial_eval_matches_bigint_path() {
    let k = 3usize;
    let n = 1usize << 7;
    let p: u64 = 1_048_573; // 20-bit prime
    let poly_int: Vec<BigInt> = (0..n)
      .map(|i| {
        let base = (BigInt::from(0x9e3779b97f4a7c15u64) << (i % 120)) + BigInt::from(i);
        if i % 3 == 0 { -base } else { base }
      })
      .collect();
    let r: Vec<BigUint> = (0..k)
      .map(|i| BigUint::from((0xACE1u64 * (i as u64 + 7)) % p))
      .collect();

    let expect = integer_partial_evaluate_top_k(&poly_int, &r);

    let poly_i256: Vec<I256> = poly_int
      .iter()
      .map(|v| {
        let mut x = I256::from_biguint(v.magnitude());
        x.neg = v.sign() == Sign::Minus && !x.mag.iter().all(|&w| w == 0);
        x
      })
      .collect();
    let r_u64: Vec<u64> = r
      .iter()
      .map(|x| x.iter_u64_digits().next().unwrap_or(0))
      .collect();
    let got = integer_partial_evaluate_top_k_i256(&poly_i256, &r_u64);

    assert_eq!(expect.len(), got.len());
    let d_big = BigInt::from(p);
    for (e, g) in expect.iter().zip(got.iter()) {
      assert_eq!(*e, g.to_bigint(), "partial eval mismatch");
      // Divmod agreement (truncated toward zero, sign(r) = sign(g)).
      let (q, rem) = g.div_rem_u64(p);
      let q_big = e / &d_big;
      let rem_big = e - &q_big * &d_big;
      assert_eq!(q.to_bigint(), q_big, "quotient mismatch");
      assert_eq!(rem.to_bigint(), rem_big, "remainder mismatch");
    }
  }

  /// `derive` produces params that pass `validate` for the variable
  /// counts our test SNARKs use.
  #[test]
  fn derive_default_params_valid() {
    for num_vars in [1usize, 2, 4, 8, 16, 25] {
      let p = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, DEFAULT_K, num_vars).unwrap();
      p.validate(num_vars).unwrap();
      assert!(p.k >= 1 && p.k <= 12);
      assert!(p.log_p > 5 + ceil_log2(LAMBDA) + ceil_log2(num_vars.max(1)));
      assert!(p.s >= 1);
    }
  }

  /// Print derived params for the variable counts our tests use, as a
  /// human-readable record. Failing the test isn't the goal; the
  /// printed values document what `derive` actually picks.
  #[test]
  fn derive_picks_reasonable_params() {
    for num_vars in [4usize, 8, 16, 25] {
      let p = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, DEFAULT_K, num_vars).unwrap();
      eprintln!(
        "derive(log_T_f={}, k={}, n={}) → log_P={}, s={}",
        p.log_t_f, p.k, num_vars, p.log_p, p.s
      );
    }
  }

  /// `derive_optimized` returns valid params for every input length and
  /// norm bound we care about, and never ranks worse than the fixed
  /// defaults (which are inside its search space whenever they derive).
  #[test]
  fn derive_optimized_valid_and_never_worse_than_default() {
    for log_t_f in [32usize, 64, 256, 2048] {
      for num_vars in [1usize, 2, 4, 8, 12, 16, 20, 25] {
        let opt = IntEvalParams::derive_optimized(log_t_f, num_vars).unwrap();
        opt.validate(num_vars).unwrap();
        if let Ok(default) = IntEvalParams::derive_no_limb_split(log_t_f, DEFAULT_K, num_vars) {
          assert!(
            opt.estimated_prover_cost(num_vars) <= default.estimated_prover_cost(num_vars),
            "optimized params cost more than defaults for log_t_f={log_t_f}, n={num_vars}"
          );
        }
      }
    }
  }

  /// Human-readable record of what `derive_optimized` picks versus the
  /// fixed defaults, with the cost model's ratio. Documents the
  /// optimizer's behavior; the printed values are the point.
  #[test]
  fn derive_optimized_prints_table() {
    for log_t_f in [32usize, 256, 2048] {
      for num_vars in [4usize, 8, 12, 16, 20, 25] {
        let opt = IntEvalParams::derive_optimized(log_t_f, num_vars).unwrap();
        let opt_cost = opt.estimated_prover_cost(num_vars);
        let default_str = match IntEvalParams::derive_no_limb_split(log_t_f, DEFAULT_K, num_vars) {
          Ok(d) => format!(
            "default(k={}, log_P={}, s={}) cost={:.0}, ratio={:.2}",
            d.k,
            d.log_p,
            d.s,
            d.estimated_prover_cost(num_vars),
            d.estimated_prover_cost(num_vars) / opt_cost
          ),
          Err(_) => "default: invalid".to_string(),
        };
        eprintln!(
          "log_T_f={log_t_f} n={num_vars}: opt(k={}, log_P={}, s={}, log_T={}, numlimb={}) cost={opt_cost:.0} | {default_str}",
          opt.k, opt.log_p, opt.s, opt.log_t, opt.numlimb
        );
      }
    }
  }

  /// End-to-end prove/verify with `setup_optimized`-chosen params, both
  /// at a size where the optimizer can skip iterations and at one where
  /// the chain/iteration machinery runs.
  #[test]
  fn prove_verify_roundtrips_optimized_params() {
    for num_vars in [4usize, 8] {
      let n = 1usize << num_vars;
      let (ck, vk) =
        IntegerModPCS::setup_optimized(b"inteval-opt", n, 256, DEFAULT_LOG_T_F).unwrap();

      let dyn_params = small_dyn_params();
      let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
      let point: Vec<DP> = (0..num_vars)
        .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
        .collect();

      let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
      let int_v = integer_mle_evaluate(&poly, &int_point);
      let p: BigUint = BigUint::from(37u32);
      let eval = int_v
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap();

      let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
      let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

      let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
      let arg =
        <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
          .unwrap();

      let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
    }
  }

  /// Wide limbs (`log_t = 128 > 64`) with large coefficients must round-trip
  /// and verify. Regression for the truncation bug: `commit` previously
  /// hardcoded `is_small = true`, so 128-bit limbs were silently truncated to
  /// their low 64 bits, producing a non-binding commitment whose opening
  /// failed the IPA verify ("first equation failed"). With the fix, wide
  /// limbs take the checked full-MSM path and the opening verifies.
  #[test]
  fn wide_limb_commit_roundtrips_and_verifies() {
    let num_vars = 6usize;
    let n = 1usize << num_vars;
    // Force log_t = 128 (numlimb = 2): the > 64-bit limb regime.
    let params = (2..16usize)
      .find_map(|k| IntEvalParams::derive(256, 128, k, num_vars).ok())
      .expect("a valid k exists for log_t = 128");
    assert!(params.log_t > 64, "test must exercise > 64-bit limbs");
    let (ck, vk) = IntegerModPCS::setup_with_params(b"wide-limb", n, 256, params).unwrap();

    let dyn_params = small_dyn_params();
    // Coefficients ~2^200 so each splits into limbs that exceed 2^64 — exactly
    // what the buggy is_small=true path truncated.
    let big = BigUint::from(2u32).pow(200);
    let poly: Vec<BigUint> = (0..n).map(|i| &big + BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p: BigUint = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// N≥2 batch path: a two-poly `prove_batch` round-trips, and tampering
  /// *either* poly's claimed eval makes `verify_batch` reject. Guards the
  /// batch-specific logic (per-target λ routing, μ-RLC, the `RcTarget` poly
  /// index) that the SNARK driver tests only hit on the happy path.
  #[test]
  fn batch_open_rejects_tampered_eval() {
    let num_vars = 6usize;
    let n = 1usize << num_vars;
    let (ck, vk) =
      IntegerModPCS::setup_optimized(b"inteval-batch", n, 256, DEFAULT_LOG_T_F).unwrap();

    let dyn_params = small_dyn_params();
    let p: BigUint = BigUint::from(37u32);

    // Two distinct polynomials opened at two distinct points.
    let poly0: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let poly1: Vec<BigUint> = (0..n).map(|i| BigUint::from((i as u32) * 3 + 5)).collect();
    let point0: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();
    let point1: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 5 + 1) % 37))
      .collect();

    let eval_of = |poly: &[BigUint], point: &[DP]| -> BigUint {
      let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
      integer_mle_evaluate(poly, &int_point)
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap()
    };
    let eval0 = eval_of(&poly0, &point0);
    let eval1 = eval_of(&poly1, &point1);

    let blind0 = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let blind1 = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm0 = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly0, &blind0).unwrap();
    let comm1 = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly1, &blind1).unwrap();

    let comms = [&comm0, &comm1];
    let polys: [&[BigUint]; 2] = [&poly0, &poly1];
    let blinds = [&blind0, &blind1];
    let points: [&[DP]; 2] = [&point0, &point1];
    let evals = [&eval0, &eval1];

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-batch", dyn_params);
    let arg = <MP as ModPCSEngineTrait<ME>>::prove_batch(
      &ck, &mut pt, &comms, &polys, &blinds, &points, &evals,
    )
    .unwrap();

    // Positive: the untampered 2-poly batch verifies.
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-batch", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify_batch(&vk, &mut vt, &comms, &points, &evals, &arg)
      .unwrap();

    // Negative: tampering *either* poly's claimed eval must reject — confirms
    // each claim is checked against the correct commitment.
    let bad0 = (&eval0 + BigUint::from(1u32)) % &p;
    let bad1 = (&eval1 + BigUint::from(1u32)) % &p;
    for (i, bad) in [(0usize, &bad0), (1usize, &bad1)] {
      let evals_t = if i == 0 { [bad, &eval1] } else { [&eval0, bad] };
      let mut vtt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-batch", dyn_params);
      assert!(
        <MP as ModPCSEngineTrait<ME>>::verify_batch(&vk, &mut vtt, &comms, &points, &evals_t, &arg)
          .is_err(),
        "verify_batch must reject a tampered eval for poly {i}"
      );
    }
  }

  /// Width-grouped commitment: a heterogeneous batch where one poly is
  /// committed and opened at a WIDE `IntEvalParams` (many limbs) and the
  /// other at a NARROW one (few limbs) round-trips through
  /// `prove_batch_with_params` / `verify_batch_with_params`, and a
  /// tampered narrow-segment eval still rejects. This is the enabling
  /// primitive for segmenting a mixed-width witness so its narrow part
  /// commits cheaply — the shared range check + combined open run once at
  /// the wide `ck.params` while each poly reduces at its own bound.
  /// The width-grouped `W(point)` decomposition: for an aligned dyadic
  /// tiling, `W(point) = sum_seg selector_seg(point_hi) · Seg(point_lo)`,
  /// where a segment `[start, start+2^L)` uses the LAST `L` coords for its
  /// local evaluation and the leading coords as an eq-selector on
  /// `start >> L` (MSB-first, matching `integer_mle_evaluate`). Checked
  /// over the integers so no modulus hides an ordering bug.
  /// Real-layout end-to-end payoff (--ignored, release): commit + open the
  /// actual full-statement witness uniformly (@2048, one poly) vs width-
  /// grouped over its real segments (each at its own bound), through one
  /// shared range check + combined open. Large field (M127) so the
  /// reduction's limb division never hits its toy-field zero.
  #[test]
  #[ignore = "measurement; run with --release --ignored --nocapture"]
  fn segmented_open_real_layout_measurement() {
    use crate::multiswap::{
      poseidon::PoseidonParams,
      statement::{Config, build},
    };
    use std::time::Instant;

    let dp = crypto_bigint::modular::FixedMontyParams::<2>::new(
      crypto_bigint::Odd::new(crypto_bigint::U128::MAX >> 1).unwrap(),
    );
    let pmod: BigUint = (BigUint::from(1u32) << 127u32) - BigUint::from(1u32);

    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<ME>(&Config::Full { swaps: 1 }, &poseidon, true).unwrap();
    let w = &st.built.w;
    let segs = st.built.shape.width_segments().to_vec();
    let num_vars = w.len().trailing_zeros() as usize;
    let wide = IntEvalParams::derive(2048, 64, DEFAULT_K, num_vars).unwrap();
    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"seg-real", w.len(), 256, wide.clone()).unwrap();

    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dp, ((i as u64) * 7 + 3) % 101))
      .collect();
    let eval_bu = |poly: &[BigUint], pt: &[DP]| -> BigUint {
      let ip: Vec<BigUint> = pt.iter().map(dyn_to_biguint).collect();
      integer_mle_evaluate(poly, &ip)
        .mod_floor(&BigInt::from(pmod.clone()))
        .to_biguint()
        .unwrap()
    };

    // Uniform: whole w @2048, one poly.
    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, w.len());
    let t = Instant::now();
    let comm = IntegerModPCS::commit_seg(&ck, w, &blind, &wide).unwrap();
    let u_commit = t.elapsed().as_secs_f64() * 1e3;
    let eval = eval_bu(w, &point);
    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"u", dp);
    let t = Instant::now();
    let arg = IntegerModPCS::prove_batch_seg(
      &ck,
      &mut pt,
      &[&comm],
      &[w.as_slice()],
      &[&blind],
      &[point.as_slice()],
      &[&eval],
      &[&[]],
      std::slice::from_ref(&wide),
    )
    .unwrap();
    let u_open = t.elapsed().as_secs_f64() * 1e3;
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"u", dp);
    IntegerModPCS::verify_batch_seg(
      &vk,
      &mut vt,
      &[&comm],
      &[point.as_slice()],
      &[&eval],
      &arg,
      &[&[]],
      std::slice::from_ref(&wide),
    )
    .unwrap();

    // Segmented: one poly per width segment.
    let mut params_per = Vec::new();
    let mut slices: Vec<Vec<BigUint>> = Vec::new();
    let mut locals: Vec<Vec<DP>> = Vec::new();
    let mut evals: Vec<BigUint> = Vec::new();
    for s in &segs {
      let sl = w[s.start..s.start + s.size()].to_vec();
      let hi = num_vars - s.log_len;
      let local: Vec<DP> = point[hi..].to_vec();
      evals.push(eval_bu(&sl, &local));
      params_per.push(wide.narrowed(s.log_t_f).unwrap());
      slices.push(sl);
      locals.push(local);
    }
    let blinds_s: Vec<_> = segs
      .iter()
      .map(|s| <MP as ModPCSEngineTrait<ME>>::blind(&ck, s.size()))
      .collect();
    let t = Instant::now();
    let comms_s: Vec<_> = (0..segs.len())
      .map(|i| IntegerModPCS::commit_seg(&ck, &slices[i], &blinds_s[i], &params_per[i]).unwrap())
      .collect();
    let s_commit = t.elapsed().as_secs_f64() * 1e3;
    let cr: Vec<_> = comms_s.iter().collect();
    let pr: Vec<&[BigUint]> = slices.iter().map(|v| v.as_slice()).collect();
    let br: Vec<_> = blinds_s.iter().collect();
    let ptr: Vec<&[DP]> = locals.iter().map(|v| v.as_slice()).collect();
    let er: Vec<&BigUint> = evals.iter().collect();
    let nb: Vec<&[SmallValueBlock]> = vec![&[]; segs.len()];
    let mut pt2 = <ME as SumcheckEngine>::TE::new_with_params(b"s", dp);
    let t = Instant::now();
    let arg2 =
      IntegerModPCS::prove_batch_seg(&ck, &mut pt2, &cr, &pr, &br, &ptr, &er, &nb, &params_per)
        .unwrap();
    let s_open = t.elapsed().as_secs_f64() * 1e3;
    let mut vt2 = <ME as SumcheckEngine>::TE::new_with_params(b"s", dp);
    IntegerModPCS::verify_batch_seg(&vk, &mut vt2, &cr, &ptr, &er, &arg2, &nb, &params_per)
      .unwrap();

    println!("REAL layout: num_vars={num_vars}, {} segments", segs.len());
    println!(
      "UNIFORM  @2048: commit {u_commit:.1} ms  open {u_open:.1} ms  total {:.1} ms",
      u_commit + u_open
    );
    println!(
      "SEGMENTED     : commit {s_commit:.1} ms  open {s_open:.1} ms  total {:.1} ms",
      s_commit + s_open
    );
    println!(
      "speedup vs uniform: SEG total {:.2}x",
      (u_commit + u_open) / (s_commit + s_open)
    );
  }

  #[test]
  fn width_segment_selector_sum_reconstructs_mle() {
    let num_vars = 5usize;
    let n = 1usize << num_vars;
    let w: Vec<BigUint> = (0..n).map(|i| BigUint::from((i as u64) * 7 + 1)).collect();
    // point coords (arbitrary small integers, MSB-first)
    let point: Vec<BigUint> = (0..num_vars)
      .map(|i| BigUint::from((i as u64) * 3 + 2))
      .collect();
    let full = integer_mle_evaluate(&w, &point);

    // An aligned dyadic tiling of [0, 32): 16 | 8 | 4 | 4.
    let segs: [(usize, usize); 4] = [(0, 4), (16, 3), (24, 2), (28, 2)];
    let mut acc = BigInt::from(0u32);
    for (start, log_len) in segs {
      let hi_vars = num_vars - log_len;
      // selector = eq(start >> log_len, point[0..hi_vars]) as an integer product.
      let h = start >> log_len;
      let mut sel = BigInt::from(1u32);
      for (i, pt) in point.iter().enumerate().take(hi_vars) {
        // point[i] corresponds to bit (hi_vars-1-i) of h.
        let bit = (h >> (hi_vars - 1 - i)) & 1;
        let pi = BigInt::from(pt.clone());
        sel *= if bit == 1 {
          pi
        } else {
          BigInt::from(1u32) - pi
        };
      }
      let seg = &w[start..start + (1 << log_len)];
      let local = &point[hi_vars..];
      acc += sel * integer_mle_evaluate(seg, local);
    }
    assert_eq!(acc, full, "selector-sum must reconstruct the full MLE");
  }

  /// Batch of two polys differing in BOTH size and params (the width-
  /// grouped case): poly0 has 2^7 values at the wide bound, poly1 has 2^5
  /// at a narrow bound. Isolates heterogeneous-size + heterogeneous-param
  /// batching.
  /// Four polys of DIFFERENT sizes at the SAME params in one batch — the
  /// width-grouped shape. Uses a LARGE field on purpose: the reduction
  /// verifier recovers `f_eval = red_final_claim / limb(r_k)`, and
  /// `limb(r_k)` (the limb-recombination weight at the reduction
  /// challenge) is zero at one field point. Over the toy field 37 that
  /// hits with prob ~1/37, so a heterogeneous batch whose transcript lands
  /// on it fails verification; over a ~124-bit prime (as the real prover
  /// samples) it is ~2^-124. Guards that segmentation opens are sound at
  /// production field sizes.
  #[test]
  fn four_diff_size_same_param_batch() {
    // Large field (M127 = 2^127-1) so limb(r_k)=0 is ~2^-127, not ~1/37.
    let dp = crypto_bigint::modular::FixedMontyParams::<2>::new(
      crypto_bigint::Odd::new(crypto_bigint::U128::MAX >> 1).unwrap(),
    );
    let p: BigUint = (BigUint::from(1u32) << 127u32) - BigUint::from(1u32);
    let wide = IntEvalParams::derive_optimized(256, 7).unwrap();
    let (ck, vk) = IntegerModPCS::setup_with_params(b"four", 128, 256, wide.clone()).unwrap();
    let sizes = [7usize, 6, 6, 5];
    let ev = |poly: &[BigUint], pt: &[DP]| -> BigUint {
      let ip: Vec<BigUint> = pt.iter().map(dyn_to_biguint).collect();
      integer_mle_evaluate(poly, &ip)
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap()
    };
    let mut polys = Vec::new();
    let mut pts = Vec::new();
    let mut blinds = Vec::new();
    let mut evals = Vec::new();
    for (k, &lv) in sizes.iter().enumerate() {
      let n = 1usize << lv;
      let poly: Vec<BigUint> = (0..n)
        .map(|i| BigUint::from((i as u32 + 1) * (k as u32 + 1)) << 100)
        .collect();
      let pt: Vec<DP> = (0..lv)
        .map(|i| DP::from_u64(&dp, ((i as u64) * 7 + 3 + k as u64) % 37))
        .collect();
      evals.push(ev(&poly, &pt));
      blinds.push(<MP as ModPCSEngineTrait<ME>>::blind(&ck, n));
      polys.push(poly);
      pts.push(pt);
    }
    let comms: Vec<_> = (0..4)
      .map(|k| IntegerModPCS::commit_seg(&ck, &polys[k], &blinds[k], &wide).unwrap())
      .collect();
    let cr: Vec<_> = comms.iter().collect();
    let pr: Vec<&[BigUint]> = polys.iter().map(|v| v.as_slice()).collect();
    let br: Vec<_> = blinds.iter().collect();
    let ptr: Vec<&[DP]> = pts.iter().map(|v| v.as_slice()).collect();
    let er: Vec<&BigUint> = evals.iter().collect();
    let nb: Vec<&[SmallValueBlock]> = vec![&[]; 4];
    let pp = vec![wide.clone(); 4];
    let mut tp = <ME as SumcheckEngine>::TE::new_with_params(b"four", dp);
    let a =
      IntegerModPCS::prove_batch_seg(&ck, &mut tp, &cr, &pr, &br, &ptr, &er, &nb, &pp).unwrap();
    let mut tv = <ME as SumcheckEngine>::TE::new_with_params(b"four", dp);
    IntegerModPCS::verify_batch_seg(&vk, &mut tv, &cr, &ptr, &er, &a, &nb, &pp).unwrap();
  }

  #[test]
  fn diff_size_and_param_batch_round_trips() {
    let dyn_params = small_dyn_params();
    let p: BigUint = BigUint::from(37u32);
    let wide = IntEvalParams::derive_optimized(256, 7).unwrap();
    let narrow = wide.narrowed(wide.log_t).unwrap(); // = log_t (numlimb 1)
    let (ck, vk) = IntegerModPCS::setup_with_params(b"ds", 128, 256, wide.clone()).unwrap();

    let n0 = 1usize << 7;
    let n1 = 1usize << 5;
    let poly0: Vec<BigUint> = (0..n0)
      .map(|i| BigUint::from(i as u32 + 1) << 100)
      .collect();
    let poly1: Vec<BigUint> = (0..n1).map(|i| BigUint::from(i as u32 * 3 + 2)).collect();
    let pt0: Vec<DP> = (0..7)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();
    let pt1: Vec<DP> = (0..5)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 5 + 1) % 37))
      .collect();
    let ev = |poly: &[BigUint], pt: &[DP]| -> BigUint {
      let ip: Vec<BigUint> = pt.iter().map(dyn_to_biguint).collect();
      integer_mle_evaluate(poly, &ip)
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap()
    };
    let e0 = ev(&poly0, &pt0);
    let e1 = ev(&poly1, &pt1);
    let b0 = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n0);
    let b1 = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n1);
    let c0 = IntegerModPCS::commit_seg(&ck, &poly0, &b0, &wide).unwrap();
    let c1 = IntegerModPCS::commit_seg(&ck, &poly1, &b1, &narrow).unwrap();
    let nb: [&[SmallValueBlock]; 2] = [&[], &[]];
    let pp = [wide.clone(), narrow.clone()];
    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"ds", dyn_params);
    let arg = IntegerModPCS::prove_batch_seg(
      &ck,
      &mut pt,
      &[&c0, &c1],
      &[&poly0, &poly1],
      &[&b0, &b1],
      &[&pt0, &pt1],
      &[&e0, &e1],
      &nb,
      &pp,
    )
    .unwrap();
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"ds", dyn_params);
    IntegerModPCS::verify_batch_seg(
      &vk,
      &mut vt,
      &[&c0, &c1],
      &[&pt0, &pt1],
      &[&e0, &e1],
      &arg,
      &nb,
      &pp,
    )
    .unwrap();
  }

  /// The small-value block, on a NARROW segment (numlimb_var=0), actually
  /// enforces `< 2^16`: an honest all-16-bit poly verifies, and tampering
  /// one value to 2^20 (still a valid narrow commitment, chunks < 2^16, so
  /// the plain range check passes) is rejected by the block claim. Guards
  /// the per-segment numlimb_var wiring that makes the block sound in
  /// width-grouped mode. Large field so the reduction division is safe.
  #[test]
  fn narrow_segment_block_rejects_out_of_range() {
    let dp = crypto_bigint::modular::FixedMontyParams::<2>::new(
      crypto_bigint::Odd::new(crypto_bigint::U128::MAX >> 1).unwrap(),
    );
    let p: BigUint = (BigUint::from(1u32) << 127u32) - BigUint::from(1u32);
    let num_vars = 6usize;
    let n = 1usize << num_vars;
    let wide = IntEvalParams::derive_optimized(256, num_vars).unwrap();
    let narrow = wide.narrowed(wide.log_t).unwrap(); // numlimb 1 -> numlimb_var 0
    assert_eq!(narrow.numlimb_var, 0);
    let (ck, vk) = IntegerModPCS::setup_with_params(b"nb", n, 256, wide.clone()).unwrap();

    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dp, ((i as u64) * 7 + 3) % 101))
      .collect();
    let ev = |poly: &[BigUint], pt: &[DP]| -> BigUint {
      let ip: Vec<BigUint> = pt.iter().map(dyn_to_biguint).collect();
      integer_mle_evaluate(poly, &ip)
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap()
    };
    let block = SmallValueBlock {
      start: 0,
      log_len: num_vars,
    }; // covers all n
    let blks: [&[SmallValueBlock]; 1] = [std::slice::from_ref(&block)];

    // Honest: all values < 2^16.
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 * 3 + 1)).collect();
    let bl = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = IntegerModPCS::commit_seg(&ck, &poly, &bl, &narrow).unwrap();
    let e = ev(&poly, &point);
    let mut tp = <ME as SumcheckEngine>::TE::new_with_params(b"nb", dp);
    let arg = IntegerModPCS::prove_batch_seg(
      &ck,
      &mut tp,
      &[&comm],
      &[&poly],
      &[&bl],
      &[&point],
      &[&e],
      &blks,
      std::slice::from_ref(&narrow),
    )
    .unwrap();
    let mut tv = <ME as SumcheckEngine>::TE::new_with_params(b"nb", dp);
    IntegerModPCS::verify_batch_seg(
      &vk,
      &mut tv,
      &[&comm],
      &[&point],
      &[&e],
      &arg,
      &blks,
      std::slice::from_ref(&narrow),
    )
    .unwrap();

    // Tamper: value 2^20 (>= 2^16). Chunks still < 2^16, so the plain range
    // check is fine; the block must reject.
    let mut bad = poly.clone();
    bad[3] = BigUint::from(1u32) << 20u32;
    let comm2 = IntegerModPCS::commit_seg(&ck, &bad, &bl, &narrow).unwrap();
    let e2 = ev(&bad, &point);
    let mut tp2 = <ME as SumcheckEngine>::TE::new_with_params(b"nb", dp);
    let arg2 = IntegerModPCS::prove_batch_seg(
      &ck,
      &mut tp2,
      &[&comm2],
      &[&bad],
      &[&bl],
      &[&point],
      &[&e2],
      &blks,
      std::slice::from_ref(&narrow),
    )
    .unwrap();
    let mut tv2 = <ME as SumcheckEngine>::TE::new_with_params(b"nb", dp);
    assert!(
      IntegerModPCS::verify_batch_seg(
        &vk,
        &mut tv2,
        &[&comm2],
        &[&point],
        &[&e2],
        &arg2,
        &blks,
        std::slice::from_ref(&narrow),
      )
      .is_err(),
      "block must reject a value >= 2^16 on a narrow segment"
    );
  }

  #[test]
  fn mixed_width_batch_round_trips() {
    let num_vars = 6usize;
    let n = 1usize << num_vars;
    let dyn_params = small_dyn_params();
    let p: BigUint = BigUint::from(37u32);

    let wide_params = IntEvalParams::derive_optimized(256, num_vars).unwrap();
    let narrow_params = wide_params.narrowed(128).unwrap();
    // Distinct limb counts are the whole point of the exercise.
    assert!(narrow_params.numlimb < wide_params.numlimb);

    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"seg-batch", n, 256, wide_params.clone()).unwrap();

    // poly0 wide (~2^180 values), poly1 narrow (~2^38 values).
    let poly0: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1) << 180).collect();
    let poly1: Vec<BigUint> = (0..n)
      .map(|i| BigUint::from((i as u32) * 3 + 5) << 30)
      .collect();

    let point0: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();
    let point1: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 5 + 1) % 37))
      .collect();

    let eval_of = |poly: &[BigUint], point: &[DP]| -> BigUint {
      let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
      integer_mle_evaluate(poly, &int_point)
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap()
    };
    let eval0 = eval_of(&poly0, &point0);
    let eval1 = eval_of(&poly1, &point1);

    let blind0 = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let blind1 = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm0 = IntegerModPCS::commit_seg(&ck, &poly0, &blind0, &wide_params).unwrap();
    let comm1 = IntegerModPCS::commit_seg(&ck, &poly1, &blind1, &narrow_params).unwrap();

    let comms = [&comm0, &comm1];
    let polys: [&[BigUint]; 2] = [&poly0, &poly1];
    let blinds = [&blind0, &blind1];
    let points: [&[DP]; 2] = [&point0, &point1];
    let evals = [&eval0, &eval1];
    let no_blocks: [&[SmallValueBlock]; 2] = [&[], &[]];
    let params_per = [wide_params.clone(), narrow_params.clone()];

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"seg-batch", dyn_params);
    let arg = IntegerModPCS::prove_batch_seg(
      &ck,
      &mut pt,
      &comms,
      &polys,
      &blinds,
      &points,
      &evals,
      &no_blocks,
      &params_per,
    )
    .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"seg-batch", dyn_params);
    IntegerModPCS::verify_batch_seg(
      &vk,
      &mut vt,
      &comms,
      &points,
      &evals,
      &arg,
      &no_blocks,
      &params_per,
    )
    .unwrap();

    // Negative control: tamper the narrow segment's claimed eval.
    let bad1 = (&eval1 + BigUint::from(1u32)) % &p;
    let evals_bad = [&eval0, &bad1];
    let mut vtb = <ME as SumcheckEngine>::TE::new_with_params(b"seg-batch", dyn_params);
    assert!(
      IntegerModPCS::verify_batch_seg(
        &vk,
        &mut vtb,
        &comms,
        &points,
        &evals_bad,
        &arg,
        &no_blocks,
        &params_per,
      )
      .is_err(),
      "verify must reject a tampered narrow-segment eval"
    );
  }

  /// End-to-end payoff of width grouping on a realistic *mixed*-width
  /// witness (`--ignored`, single-thread). Baseline: commit the whole
  /// witness at the wide 2048-bit bound and open it as one poly. Grouped:
  /// commit the wide half at 2048 and the narrow half at 256 (an 8x-fewer-
  /// limb segment via `narrowed`), open both through one shared range
  /// check + combined open. Prints commit + open wall time for each — the
  /// commit+open is ~99% of the real Spartan prove (measured), so this is
  /// the mechanism's achievable prover speedup on this width mix.
  #[test]
  #[ignore = "measurement; run explicitly with --ignored --nocapture"]
  fn width_grouping_endtoend_measurement() {
    use std::time::Instant;
    let num_vars = 13usize; // 8192 values, multiswap-scale
    let n = 1usize << num_vars;
    let half = n / 2;
    let dyn_params = small_dyn_params();
    let p: BigUint = BigUint::from(37u32);

    let wide = IntEvalParams::derive_optimized(2048, num_vars).unwrap();
    let narrow = wide.narrowed(256).unwrap();
    println!(
      "wide: log_t={} numlimb={}  narrow: numlimb={} ({}x fewer limbs)",
      wide.log_t,
      wide.numlimb,
      narrow.numlimb,
      wide.numlimb / narrow.numlimb
    );

    // Wide half: full 2048-bit values. Narrow half: ~254-bit values.
    let wide_val = (BigUint::from(1u32) << 2000usize) + BigUint::from(12345u32);
    let narrow_val = (BigUint::from(1u32) << 250usize) + BigUint::from(678u32);
    let whole: Vec<BigUint> = (0..n)
      .map(|i| {
        if i < half {
          wide_val.clone()
        } else {
          narrow_val.clone()
        }
      })
      .collect();
    let seg_wide: Vec<BigUint> = whole[..half].to_vec();
    let seg_narrow: Vec<BigUint> = whole[half..].to_vec();

    let (ck, vk) = IntegerModPCS::setup_with_params(b"wg-e2e", n, 256, wide.clone()).unwrap();

    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();
    let pt_lo: Vec<DP> = point[1..].to_vec(); // segment point (top bit selects half)
    let eval_of = |poly: &[BigUint], point: &[DP]| -> BigUint {
      let ip: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
      integer_mle_evaluate(poly, &ip)
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap()
    };

    // ---- Baseline: whole witness at the wide bound, one poly ----
    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let t = Instant::now();
    let comm = IntegerModPCS::commit_seg(&ck, &whole, &blind, &wide).unwrap();
    let base_commit = t.elapsed().as_secs_f64() * 1e3;
    let eval = eval_of(&whole, &point);
    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"wg", dyn_params);
    let t = Instant::now();
    let arg = IntegerModPCS::prove_batch_seg(
      &ck,
      &mut pt,
      &[&comm],
      &[&whole],
      &[&blind],
      &[&point],
      &[&eval],
      &[&[]],
      std::slice::from_ref(&wide),
    )
    .unwrap();
    let base_open = t.elapsed().as_secs_f64() * 1e3;
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"wg", dyn_params);
    IntegerModPCS::verify_batch_seg(
      &vk,
      &mut vt,
      &[&comm],
      &[&point],
      &[&eval],
      &arg,
      &[&[]],
      std::slice::from_ref(&wide),
    )
    .unwrap();

    // ---- Grouped: wide half @2048 + narrow half @256, one batch ----
    let bw = <MP as ModPCSEngineTrait<ME>>::blind(&ck, half);
    let bn = <MP as ModPCSEngineTrait<ME>>::blind(&ck, half);
    let t = Instant::now();
    let cw = IntegerModPCS::commit_seg(&ck, &seg_wide, &bw, &wide).unwrap();
    let cn = IntegerModPCS::commit_seg(&ck, &seg_narrow, &bn, &narrow).unwrap();
    let grp_commit = t.elapsed().as_secs_f64() * 1e3;
    let ew = eval_of(&seg_wide, &pt_lo);
    let en = eval_of(&seg_narrow, &pt_lo);
    let mut pt2 = <ME as SumcheckEngine>::TE::new_with_params(b"wg", dyn_params);
    let t = Instant::now();
    let arg2 = IntegerModPCS::prove_batch_seg(
      &ck,
      &mut pt2,
      &[&cw, &cn],
      &[&seg_wide, &seg_narrow],
      &[&bw, &bn],
      &[&pt_lo, &pt_lo],
      &[&ew, &en],
      &[&[], &[]],
      &[wide.clone(), narrow.clone()],
    )
    .unwrap();
    let grp_open = t.elapsed().as_secs_f64() * 1e3;
    let mut vt2 = <ME as SumcheckEngine>::TE::new_with_params(b"wg", dyn_params);
    IntegerModPCS::verify_batch_seg(
      &vk,
      &mut vt2,
      &[&cw, &cn],
      &[&pt_lo, &pt_lo],
      &[&ew, &en],
      &arg2,
      &[&[], &[]],
      &[wide.clone(), narrow.clone()],
    )
    .unwrap();

    println!(
      "BASELINE (whole @2048):  commit {base_commit:.1} ms  open {base_open:.1} ms  total {:.1} ms",
      base_commit + base_open
    );
    println!(
      "GROUPED  (2048 | 256):   commit {grp_commit:.1} ms  open {grp_open:.1} ms  total {:.1} ms",
      grp_commit + grp_open
    );
    println!(
      "speedup: commit {:.2}x  open {:.2}x  total {:.2}x",
      base_commit / grp_commit,
      base_open / grp_open,
      (base_commit + base_open) / (grp_commit + grp_open)
    );
  }

  /// `validate` catches a hand-rolled `IntEvalParams` literal where
  /// `numlimb` is inconsistent with `(log_t, log_t_f)`.
  #[test]
  fn validate_rejects_bad_numlimb() {
    let bad = IntEvalParams {
      format_version: 2,
      security_bits: LAMBDA,
      log_q: LOG_Q,
      k: 7,
      log_p: 27,
      s: 10,
      log_t: 16,
      log_t_f: 32, // ⌈32/16⌉ = 2
      numlimb: 1,  // mismatched
      numlimb_var: 0,
    };
    let err = bad.validate(8).unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// `numlimb` / `numlimb_var` sanity. Standard cases plus boundary.
  #[test]
  fn numlimb_basic() {
    assert_eq!(numlimb(32, 32), 1);
    assert_eq!(numlimb_var(1), 0);
    assert_eq!(numlimb(32, 16), 2);
    assert_eq!(numlimb_var(2), 1);
    assert_eq!(numlimb(32, 8), 4);
    assert_eq!(numlimb_var(4), 2);
    assert_eq!(numlimb(32, 12), 3); // ceil(32/12) = 3
    assert_eq!(numlimb_var(3), 2); // ceil(log_2 3) = 2 → pad to 4 slots
    assert_eq!(numlimb(33, 16), 3); // log_t_f not divisible by log_t
  }

  /// `chunk_decompose_value` is invertible by `sum_c 2^(16c) · chunk[c]`
  /// and every chunk lies in `[0, 2^16)`.
  #[test]
  fn chunk_decompose_round_trips() {
    for (v, log_bound) in [
      (BigUint::from(0u32), 8),
      (BigUint::from(1u32), 1),
      (BigUint::from(0xffu32), 8),
      (BigUint::from(0xabcdu32), 16),
      (BigUint::from(0xdeadbeefu32), 32),
      (BigUint::from(0xffff_ffff_ffff_ffffu64), 64),
      (BigUint::from(0x7fff_ffffu32), 31), // odd bit count, top-bit-zero
      ((BigUint::one() << 227) - BigUint::one(), 227), // b_j-style width
    ] {
      let chunks = chunk_decompose_value(&v, log_bound);
      assert_eq!(chunks.len(), log_bound.div_ceil(CHUNK_BITS));
      let rem = log_bound - CHUNK_BITS * (chunks.len() - 1);
      for (c, ch) in chunks.iter().enumerate() {
        assert!(*ch < 1u64 << CHUNK_BITS);
        if c == chunks.len() - 1 {
          assert!(*ch < 1u64 << rem, "top chunk exceeds 2^{rem}");
        }
      }
      let mut acc = BigUint::zero();
      for (c, ch) in chunks.iter().enumerate() {
        acc += BigUint::from(*ch) << (CHUNK_BITS * c);
      }
      assert_eq!(acc, v, "decomp of 0x{v:x} doesn't round-trip");
    }
  }

  /// `batch_weight`'s structured assembly (boolean-head blocks, shared-
  /// prefix tensor groups, singletons) matches the naive Σ λ^i·eq(z_i,·).
  #[test]
  fn batch_weight_matches_naive() {
    let n_vars = 6usize;
    let n = 1usize << n_vars;
    let rnd = |s: u64| t256::Scalar::from(s * s + 3);
    // Mixed structure: two boolean-head claims, a shared-prefix group of
    // three, and an unrelated singleton.
    let shared: Vec<t256::Scalar> = (0..3).map(|i| rnd(40 + i)).collect();
    let mut points: Vec<Vec<t256::Scalar>> = vec![
      bool_point_of_index(5, 3)
        .into_iter()
        .chain((0..3).map(|i| rnd(7 + i)))
        .collect(),
      bool_point_of_index(2, 2)
        .into_iter()
        .chain((0..4).map(|i| rnd(11 + i)))
        .collect(),
    ];
    for c in 0..3u64 {
      points.push(
        shared
          .iter()
          .copied()
          .chain((0..3).map(|i| rnd(100 * (c + 1) + i)))
          .collect(),
      );
    }
    points.push((0..6).map(|i| rnd(900 + i)).collect());
    let evals: Vec<t256::Scalar> = (0..points.len() as u64).map(|i| rnd(70 + i)).collect();
    let lambda = rnd(31337);

    let (w, claim) = batch_weight(&points, &evals, lambda, n, &mut EqTableCache::new());

    let mut w_naive = vec![t256::Scalar::ZERO; n];
    let mut claim_naive = t256::Scalar::ZERO;
    let mut lam = t256::Scalar::ONE;
    for (z, y) in points.iter().zip(evals.iter()) {
      let eq = EqPolynomial::<t256::Scalar>::evals_from_points(z);
      for (wj, e) in w_naive.iter_mut().zip(eq.iter()) {
        *wj += lam * e;
      }
      claim_naive += lam * y;
      lam *= lambda;
    }
    assert_eq!(w, w_naive);
    assert_eq!(claim, claim_naive);
  }

  /// `bool_point_of_index` selects the right slot: binding an MLE's
  /// variables to `bits(idx)` evaluates the dense table at `idx`.
  #[test]
  fn bool_point_selects_index() {
    let table: Vec<t256::Scalar> = (0..8u64).map(|i| t256::Scalar::from(100 + i)).collect();
    for idx in 0..8usize {
      let pt = bool_point_of_index(idx, 3);
      assert_eq!(mle_evaluate_fq(&table, &pt), table[idx]);
    }
  }

  /// `split_value_into_limbs` is invertible: reconstruct from limbs.
  #[test]
  fn split_value_round_trips() {
    let log_t = 8usize;
    let t = BigUint::one() << log_t;
    for (v, log_t_f) in [
      (BigUint::from(0u32), 32),
      (BigUint::from(1u32), 32),
      (BigUint::from(0xdeadbeefu32), 32),
      (BigUint::from(0xffffu32), 16),
      (BigUint::from(0xffu32), 8),
      (BigUint::from(0xffff_ffff_ffff_ffffu64), 64),
    ] {
      let nl = numlimb(log_t_f, log_t);
      let limbs = split_value_into_limbs(&v, log_t, nl);
      assert_eq!(limbs.len(), nl);
      for limb in &limbs {
        assert!(limb < &t, "limb 0x{:x} exceeds 2^{}", limb, log_t);
      }
      // Reconstruct: sum_i T^i · limbs[i].
      let mut acc = BigUint::zero();
      for limb in limbs.iter().rev() {
        acc = &acc * &t + limb;
      }
      assert_eq!(acc, v);
    }
  }

  /// `limb_split_polynomial` no-op when `log_t == log_t_f` (numlimb=1).
  /// In that case the output equals the input (only one limb, no
  /// padding since `numlimb_var = 0` → stride = 1).
  #[test]
  fn limb_split_no_op_when_log_t_eq_log_t_f() {
    let poly: Vec<BigUint> = (0..8u32).map(BigUint::from).collect();
    let out = limb_split_polynomial(&poly, 32, 32);
    assert_eq!(out, poly);
  }

  /// `limb_split_polynomial` with `numlimb = 2` (T = 2^8, T_f = 2^16):
  /// each coefficient becomes two limbs `(low, high)`, laid out in
  /// adjacent slots. Recoverable by `low + 256 · high == original`.
  #[test]
  fn limb_split_pairs_of_limbs() {
    let poly = vec![
      BigUint::from(0x0000u32),
      BigUint::from(0x00ffu32),
      BigUint::from(0xff00u32),
      BigUint::from(0xabcdu32),
    ];
    let out = limb_split_polynomial(&poly, 8, 16);
    assert_eq!(out.len(), 8); // 4 · 2 slots

    for (x, orig) in poly.iter().enumerate() {
      let lo = &out[x * 2];
      let hi = &out[x * 2 + 1];
      let reconstructed = lo + BigUint::from(256u32) * hi;
      assert_eq!(&reconstructed, orig, "slot {x}");
    }
  }

  /// `limb_split_polynomial` with non-power-of-two `numlimb`: pad
  /// the missing slots with zero.
  #[test]
  fn limb_split_pads_to_power_of_two() {
    let poly = vec![BigUint::from(0x0afbcu32)]; // 20 bits
    let out = limb_split_polynomial(&poly, 8, 20); // numlimb = 3, stride = 4
    assert_eq!(out.len(), 4);
    // 0x0afbc = 0xbc + 0xaf · 256 + 0x00 · 65536.
    assert_eq!(out[0], BigUint::from(0xbcu32));
    assert_eq!(out[1], BigUint::from(0xafu32));
    assert_eq!(out[2], BigUint::from(0x00u32)); // top limb (within numlimb)
    assert_eq!(out[3], BigUint::from(0u32)); // padding slot
  }

  /// Partial-eval at the last variable should match a 2-step direct
  /// evaluation: poly is 8 evals (3 vars), partial-eval the last var,
  /// then evaluate the remaining 2-var poly at a 2-component point.
  #[test]
  fn integer_partial_evaluate_matches_full_eval() {
    // poly[x_0, x_1, x_2] = 100·x_0 + 10·x_1 + x_2 (over Z).
    // The evaluation table walks (x_0, x_1, x_2) in big-endian bit order,
    // so poly[(b2 b1 b0)] = 100·b2 + 10·b1 + b0.
    let poly: Vec<BigInt> = (0..8u32)
      .map(|k| BigInt::from(100 * ((k >> 2) & 1) + 10 * ((k >> 1) & 1) + (k & 1)))
      .collect();
    // Partial-eval at last variable to value 3.
    let r_last = vec![BigUint::from(3u32)];
    let g = integer_partial_evaluate_top_k(&poly, &r_last);
    assert_eq!(g.len(), 4);
    // g[(b2 b1)] = poly(b2, b1, 3) = 100·b2 + 10·b1 + 3.
    for k in 0..4u32 {
      let expected = BigInt::from(100 * ((k >> 1) & 1) + 10 * (k & 1) + 3);
      assert_eq!(g[k as usize], expected);
    }
  }

  /// `integer_mle_evaluate`'s fold matches the naive chi-product sum
  /// (the pre-optimization definition), pinning the variable bit-order.
  #[test]
  fn integer_mle_evaluate_matches_naive() {
    let num_vars = 5usize;
    let n = 1usize << num_vars;
    let poly: Vec<BigUint> = (0..n).map(|k| BigUint::from((k * k + 7) as u64)).collect();
    let point: Vec<BigUint> = (0..num_vars)
      .map(|i| BigUint::from((1u64 << 40) + 31 * i as u64 + 5))
      .collect();

    // Naive: sum_k chi(point, k) · poly[k], variable i ↔ bit num_vars−1−i.
    let point_int: Vec<BigInt> = point.iter().map(|x| BigInt::from(x.clone())).collect();
    let one = BigInt::one();
    let mut naive = BigInt::zero();
    for (k, poly_k) in poly.iter().enumerate() {
      let mut chi = one.clone();
      for (i, pi) in point_int.iter().enumerate() {
        let bit = (k >> (num_vars - 1 - i)) & 1;
        chi *= if bit == 1 { pi.clone() } else { &one - pi };
      }
      naive += chi * BigInt::from(poly_k.clone());
    }

    assert_eq!(integer_mle_evaluate(&poly, &point), naive);
  }

  #[test]
  #[ignore]
  fn small_q_param_grid() {
    // Exploratory: what (log_p, s, layers) does each candidate log_q
    // admit, per the derive()/validate() formulas, ignoring Soundness
    // Bound 2 (which pins log_q >= LAMBDA + log2(s*n) and would need
    // extension-field challenges to relax)?
    let num_vars = 13usize; // multiswap 2^13
    let log_t_f = 2048usize;
    for log_q in [64usize, 96, 128, 160, 192, 256] {
      println!("--- log_q = {log_q} ---");
      for log_t in [16usize, 32, 64] {
        let nl = numlimb(log_t_f, log_t);
        let nlv = numlimb_var(nl);
        let n_total = num_vars + nlv;
        let mut best: Option<(usize, usize, usize, usize)> = None;
        for k in 1..=16usize {
          let mut log_p = 0usize;
          for lp in 2..log_q {
            let partial = k + k * lp + log_t.max(lp);
            let final_eval = k + (k + 1) * lp;
            if partial < log_q && final_eval < log_q {
              log_p = lp;
            }
          }
          if log_p < 5 {
            continue;
          }
          let bpp = soundness_bits_per_prime(log_p, n_total, log_t);
          if bpp <= 0.0 {
            continue;
          }
          let s = (LAMBDA as f64 / bpp).ceil() as usize;
          let layers = n_total.div_ceil(k);
          let better = match best {
            None => true,
            Some((bl, _, _, bs)) => layers < bl || (layers == bl && s < bs),
          };
          if better {
            best = Some((layers, k, log_p, s));
          }
          println!(
            "  log_t={log_t:2} k={k:2}: log_p={log_p:2} s={s:3} layers={layers:2}              bound2_needs_log_q>={}",
            LAMBDA_BOUND2 + ceil_log2((s * num_vars).max(1))
          );
        }
        if let Some((layers, k, log_p, s)) = best {
          println!("  => best log_t={log_t}: k={k} log_p={log_p} s={s} layers={layers}");
        } else {
          println!("  => log_t={log_t}: INFEASIBLE");
        }
      }
    }
  }

  /// `explicit` rejects a config whose Soundness Bound 1 fails: a small
  /// `log_p` paired with a small `s` gives a too-large soundness error.
  #[test]
  fn explicit_rejects_bad_soundness() {
    let err = IntEvalParams::explicit(
      /* k */ 7, /* log_p */ 12, // way too small: soundness_1 = (32·128·n/2^12)^s
      /* s */ 1, /* log_t */ 32, /* log_t_f */ 32, /* num_vars */ 8,
    )
    .unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// `explicit` rejects a config whose Partial Evaluation Norm Bound
  /// fails: large `k` × large `log_p` overflows the field.
  #[test]
  fn explicit_rejects_partial_norm_overflow() {
    let err = IntEvalParams::explicit(
      /* k */ 12, /* log_p */ 40, /* s */ 5, /* log_t */ 32,
      /* log_t_f */ 32, /* num_vars */ 8,
    )
    .unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// `setup_with_params` accepts a valid override, rejects a bad one.
  #[test]
  fn setup_with_params_round_trips_overrides() {
    let n = 16usize;
    let p = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, DEFAULT_K, ceil_log2(n)).unwrap();
    let (_ck, _vk) = IntegerModPCS::setup_with_params(b"override", n, 256, p).unwrap();

    // Bad params: zero `s` makes soundness_1 fail trivially.
    let bad = IntEvalParams {
      format_version: 2,
      security_bits: LAMBDA,
      log_q: LOG_Q,
      k: 7,
      log_p: 20,
      s: 0,
      log_t: 32,
      log_t_f: 32,
      numlimb: 1,
      numlimb_var: 0,
    };
    let err = IntegerModPCS::setup_with_params(b"override", n, 256, bad).unwrap_err();
    assert!(matches!(err, SpartanError::InvalidInputLength { .. }));
  }

  /// Helper: build params for a small dynamic prime so we can
  /// deterministically evaluate the polynomial at a known Z_p point.
  fn small_dyn_params() -> crypto_bigint::modular::FixedMontyParams<2> {
    use crypto_bigint::{Odd, U128};
    // A small prime (37) so the integer evaluation is human-verifiable.
    crypto_bigint::modular::FixedMontyParams::new(Odd::new(U128::from(37u32)).unwrap())
  }

  /// End-to-end IntEval prove/verify for the `n ≤ k` regime.
  #[test]
  fn prove_verify_roundtrips_small_witness() {
    let num_vars = 4usize;
    let n = 1usize << num_vars; // 16
    let (ck, vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-rt", n, 256);

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();

    // Oracle Z_p eval: take the integer evaluation reduced mod p.
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p: BigUint = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// A large witness with an all-zero tail exercises the active-block
  /// split: zero blocks leave the LogUp multiset (pinned by fresh
  /// zero claims instead), the roundtrip verifies, and a forged
  /// active-block map is rejected in both directions.
  #[test]
  fn rc_zero_blocks_dropped_and_bitmap_pinned() {
    let num_vars = 12usize;
    let n = 1usize << num_vars; // 4096 values × 128 chunk slots -> 8 blocks
    let params = IntEvalParams::derive(2048, 64, DEFAULT_K, num_vars).unwrap();
    let (ck, vk) = MP::setup_with_params(b"inteval-zb", n, 256, params).unwrap();

    let dyn_params = small_dyn_params();
    // Nonzero head, all-zero tail (the padded-row shape).
    let poly: Vec<BigUint> = (0..n)
      .map(|i| {
        if i < 100 {
          (BigUint::from(i as u32 + 1) << 1000) + BigUint::from(7u32)
        } else {
          BigUint::from(0u32)
        }
      })
      .collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();
    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    // The f batch's chunk polynomial spans multiple 2^16-slot blocks;
    // the zero tail must have deactivated at least one, and the nonzero
    // head must have kept the first active.
    assert!(
      arg.range_check.active_blocks[0].iter().any(|&a| !a),
      "expected an inactive (all-zero) block in the f batch"
    );
    assert!(
      arg.range_check.active_blocks[0][0],
      "head block must stay active"
    );

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();

    // Forgery 1: claim an active (nonzero) block is all-zero.
    let mut forged = arg.clone();
    forged.range_check.active_blocks[0][0] = false;
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &forged).is_err(),
      "nonzero block forged inactive must fail"
    );

    // Forgery 2: resurrect a zero block as active.
    let mut forged = arg.clone();
    let zi = forged.range_check.active_blocks[0]
      .iter()
      .position(|&a| !a)
      .unwrap();
    forged.range_check.active_blocks[0][zi] = true;
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &forged).is_err(),
      "zero block forged active must fail"
    );
  }

  /// Verifier rejects a tampered claimed Z_p eval.
  #[test]
  fn verify_rejects_wrong_eval() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;
    let (ck, vk) = <MP as ModPCSEngineTrait<ME>>::setup(b"inteval-rt", n, 256);

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 3) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p = BigUint::from(37u32);
    let real_eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();
    // Tamper: add 1 mod 37.
    let bad_eval = (real_eval.clone() + BigUint::from(1u32)) % &p;

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &real_eval)
        .unwrap();

    // Verifier with the bad eval claim must reject.
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev", dyn_params);
    let err = <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &bad_eval, &arg)
      .unwrap_err();
    assert!(matches!(err, SpartanError::InvalidSumcheckProof));
  }

  /// Step C end-to-end: `n > k` triggers the partial-eval iteration
  /// path. Uses explicit small-k params (k=2) so a 4-var poly hits
  /// `t = ⌈(4-2)/2⌉ = 1` iteration.
  #[test]
  fn prove_verify_roundtrips_with_iteration() {
    let num_vars = 4usize;
    let n = 1usize << num_vars; // 16
    // Small-k config so the partial-eval iteration path triggers.
    // `derive` picks the largest valid log_p and the smallest s.
    let small_params =
      IntEvalParams::derive_no_limb_split(8, 2, num_vars).expect("valid derived params");
    let (ck, vk) = IntegerModPCS::setup_with_params(b"inteval-iter", n, 256, small_params).unwrap();

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 3 + 5) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// Two-iteration roundtrip (`k=2`, `num_vars=6` → `t=2`). Exercises the
  /// a_prev batch (j=1, batched across chains) *and* the per-iteration
  /// individual a_prev opens (j=2, chain-specific commitments) together.
  #[test]
  fn prove_verify_roundtrips_with_two_iterations() {
    let num_vars = 6usize;
    let n = 1usize << num_vars; // 64
    let small_params =
      IntEvalParams::derive_no_limb_split(8, 2, num_vars).expect("valid derived params");
    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"inteval-iter2", n, 256, small_params).unwrap();

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 3 + 5) % 37))
      .collect();
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let eval = integer_mle_evaluate(&poly, &int_point)
      .mod_floor(&BigInt::from(37u32))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter2", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    // Confirm we actually exercised t=2 (two layers, each with an `a`
    // and a `b` chunk commitment).
    assert_eq!(arg.chains[0].iterations.len(), 2, "expected t=2");
    assert_eq!(
      arg.ab_comms.len(),
      4,
      "expected 2 layers × (a, b) chunk comms"
    );

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-iter2", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// Step D5 (stacked rbatchrange): tampering *any* range-check group's
  /// committed bit evaluation must make the verifier reject. The
  /// iteration config (k=2, num_vars=4 → t=1) yields three groups:
  /// `f_limb`, the `a_1` batch, and the `b_1` batch — so this exercises
  /// every segment type. A passing roundtrip with all groups present is
  /// not enough; we must confirm each group is actually checked.
  #[test]
  fn verify_rejects_tampered_range_check() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;
    let small_params =
      IntEvalParams::derive_no_limb_split(8, 2, num_vars).expect("valid derived params");
    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"inteval-rc-tamper", n, 256, small_params).unwrap();

    let dyn_params = small_dyn_params();
    let poly: Vec<BigUint> = (0..n).map(|i| BigUint::from(i as u32 + 1)).collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 3 + 5) % 37))
      .collect();
    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let eval = integer_mle_evaluate(&poly, &int_point)
      .mod_floor(&BigInt::from(37u32))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();

    // Every batch's chunk polynomial IS its target's commitment now, so
    // the range check carries no per-batch proof data at all.
    assert_eq!(
      arg.range_check.batches.len(),
      0,
      "all batches are precommitted"
    );

    // Tampering a layer chunk commitment (the a_1 range-check oracle)
    // must be rejected: the GKR claims and identity-check claims no
    // longer match the substituted commitment.
    let mut bad = arg.clone();
    bad.ab_comms[0] = bad.ab_comms[1].clone();
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &bad,).is_err(),
      "ab chunk-comm tamper not rejected"
    );

    // Tampering the multiplicity commitment must be rejected (the LogUp
    // table side no longer matches).
    let mut bad = arg.clone();
    bad.range_check.mult_comm = arg.ab_comms[0].clone();
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &bad,).is_err(),
      "mult-comm tamper not rejected"
    );

    // Tampering a batched open's final evaluation must be rejected.
    let mut bad = arg.clone();
    bad.combined_open.final_evals[0] += t256::Scalar::ONE;
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &bad,).is_err()
    );

    // Dropping a layer chunk commitment (count mismatch) must be
    // rejected.
    let mut short = arg.clone();
    short.ab_comms.pop();
    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"intev-rc", dyn_params);
    assert!(
      <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &short,).is_err()
    );
  }

  /// Step D4 end-to-end: real limb-splitting (`log_T < log_T_f` → `numlimb
  /// = 2`, `numlimb_var = 1`). Each polynomial coefficient is split into
  /// two 4-bit limbs; the F-PCS commits a polynomial of `2 · n = 32`
  /// slots. The reduction sumcheck runs one round and binds `r_k` of
  /// length 1; the IntEval body operates on `f_limb` at the extended
  /// point `(int_r, int_r_k)`.
  #[test]
  fn prove_verify_roundtrips_with_limb_split() {
    let num_vars = 4usize;
    let n = 1usize << num_vars;

    // log_T = 4 < log_T_f = 8 → numlimb = 2, numlimb_var = 1.
    // k = 2 keeps soundness derivation feasible at this small λ-style
    // setup. Coefficients < 2^8 fit in two 4-bit limbs.
    let limb_params = IntEvalParams::derive(8, 4, 2, num_vars).expect("valid derived params");
    assert_eq!(limb_params.numlimb, 2);
    assert_eq!(limb_params.numlimb_var, 1);

    let (ck, vk) =
      IntegerModPCS::setup_with_params(b"limb-split-test", n, 256, limb_params).unwrap();

    let dyn_params = small_dyn_params();
    // Coefficients in [0, 2^8). The integer eval can grow large but
    // mod p reduces to a clean Z_p value.
    let poly: Vec<BigUint> = (0..n)
      .map(|i| BigUint::from((i * 13 + 1) as u32 & 0xff))
      .collect();
    let point: Vec<DP> = (0..num_vars)
      .map(|i| DP::from_u64(&dyn_params, ((i as u64) * 7 + 2) % 37))
      .collect();

    let int_point: Vec<BigUint> = point.iter().map(dyn_to_biguint).collect();
    let int_v = integer_mle_evaluate(&poly, &int_point);
    let p = BigUint::from(37u32);
    let eval = int_v
      .mod_floor(&BigInt::from(p.clone()))
      .to_biguint()
      .unwrap();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"limb-split", dyn_params);
    let arg =
      <MP as ModPCSEngineTrait<ME>>::prove(&ck, &mut pt, &comm, &poly, &blind, &point, &eval)
        .unwrap();
    // The reduction sumcheck ran one round → one entry in
    // reduction_round_polys.
    assert_eq!(arg.reduction_round_polys.len(), 1);

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"limb-split", dyn_params);
    <MP as ModPCSEngineTrait<ME>>::verify(&vk, &mut vt, &comm, &point, &eval, &arg).unwrap();
  }

  /// Regression: limb-split commit when the *inflated* polynomial spans
  /// multiple Hyrax rows (`width < 2^numlimb_var · n`). `blind` must cover
  /// the inflated length, not the input length — otherwise `commit`
  /// indexes past the blind. The masked case from
  /// `prove_verify_roundtrips_with_limb_split` (where `n < width`, so
  /// everything fit in one row) did not exercise this.
  #[test]
  fn limb_split_commit_spans_multiple_hyrax_rows() {
    let num_vars = 4usize;
    let n = 1usize << num_vars; // 16
    // numlimb = 2, numlimb_var = 1 → inflated length 32.
    let limb_params = IntEvalParams::derive(8, 4, 2, num_vars).expect("valid derived params");
    assert_eq!(limb_params.numlimb_var, 1);

    // width = 4 < inflated length 32 → div_ceil(32, 4) = 8 Hyrax rows,
    // versus div_ceil(16, 4) = 4 rows for the un-inflated blind.
    let (ck, _vk) =
      IntegerModPCS::setup_with_params(b"limb-split-rows", n, 4, limb_params).unwrap();

    let poly: Vec<BigUint> = (0..n)
      .map(|i| BigUint::from((i * 13 + 1) as u32 & 0xff))
      .collect();

    let blind = <MP as ModPCSEngineTrait<ME>>::blind(&ck, n);
    // The bug manifested as an index-out-of-bounds panic here.
    let comm = <MP as ModPCSEngineTrait<ME>>::commit(&ck, &poly, &blind).unwrap();

    // Commit matches a direct Hyrax commit of the limb-split, chunked
    // polynomial (the committed-chunk layout).
    let limbs = limb_split_polynomial(&poly, 4, 8);
    let chunks = build_chunk_poly(&[&limbs], limbs.len(), 4);
    let chunk_fq: Vec<t256::Scalar> = chunks.iter().map(|&c| scalar_from_chunk(c)).collect();
    let direct = Hyrax::commit(&ck.inner, &chunk_fq, &blind.inner, true).unwrap();
    assert_eq!(comm.inner, direct);
  }

  /// `checked_chunk_stride` / `f_chunk_len` boundary behavior: normal
  /// table values, checked ceiling addition, the `usize -> u32` shift
  /// conversion, shift overflow, and multiply overflow all return
  /// `SpartanError` instead of panicking or wrapping.
  #[test]
  fn f_chunk_len_boundaries() {
    // Normal values: chunk_stride(64) = 4; the benchmark memory table
    // (log_t_f = 256, log_t = 64 → numlimb = 4, numlimb_var = 2).
    assert_eq!(checked_chunk_stride(64).unwrap(), 4);
    assert_eq!(checked_chunk_stride(64).unwrap(), chunk_stride(64));
    let params = IntEvalParams::derive(256, 64, 9, 13).unwrap();
    assert_eq!(params.numlimb, 4);
    assert_eq!(params.numlimb_var, 2);
    assert_eq!(f_chunk_len(&params, 1 << 13).unwrap(), 1 << 17);
    assert_eq!(f_chunk_len(&params, 1 << 17).unwrap(), 1 << 21);
    assert_eq!(f_chunk_len(&params, 1 << 21).unwrap(), 1 << 25);

    // Checked stride: the ceiling addition overflows.
    assert!(checked_chunk_stride(usize::MAX).is_err());

    // Multiply overflow in the final product.
    assert!(f_chunk_len(&params, usize::MAX).is_err());

    // Shift-amount overflow (2^numlimb_var no longer fits usize) via a
    // hand-rolled params literal; the u32 conversion path needs a value
    // above u32::MAX, which usize accommodates on 64-bit targets.
    let mut absurd = params.clone();
    absurd.numlimb_var = usize::BITS as usize;
    assert!(f_chunk_len(&absurd, 4).is_err());
    absurd.numlimb_var = u32::MAX as usize + 1;
    assert!(f_chunk_len(&absurd, 4).is_err());
  }

  /// Malformed public parameters return errors, never panic: `log_t = 0`
  /// (the asserting `numlimb` path) and `k = 0` (the `div_ceil(k)` path).
  #[test]
  fn params_reject_zero_log_t_and_zero_k() {
    assert!(IntEvalParams::derive(256, 0, 9, 13).is_err());
    assert!(IntEvalParams::derive(256, 64, 0, 13).is_err());
    assert!(IntEvalParams::explicit(0, 20, 15, 64, 256, 13).is_err());
    let mut params = IntEvalParams::derive(256, 64, 9, 13).unwrap();
    params.k = 0;
    assert!(params.validate(13).is_err());
    let mut params = IntEvalParams::derive(256, 64, 9, 13).unwrap();
    params.log_t = 0;
    assert!(params.validate(13).is_err());
  }

  /// Key-capacity contract: zero capacity is rejected at construction for
  /// both backends; an over-capacity vector is rejected by `commit` with
  /// `InvalidVectorSize` (independent of `blind`).
  #[test]
  fn key_capacity_is_enforced() {
    let params = IntEvalParams::derive(32, 16, 9, 4).unwrap();
    // Zero capacity rejected.
    assert!(IntegerModPCS::setup_with_params(b"cap-test", 0, 4, params.clone()).is_err());
    assert!(BdModCommitmentKey::new(params.clone(), 0).is_err());

    // Hyrax: commit rejects an over-capacity vector.
    let n = 16usize;
    let (ck, _vk) = IntegerModPCS::setup_with_params(b"cap-test", n, 4, params.clone()).unwrap();
    assert_eq!(ck.max_n, n);
    let blind = <IntegerModPCS as ModPCSEngineTrait<ME>>::blind(&ck, n);
    let long: Vec<BigUint> = (0..n + 1).map(|i| BigUint::from(i as u32)).collect();
    match <IntegerModPCS as ModPCSEngineTrait<ME>>::commit(&ck, &long, &blind) {
      Err(SpartanError::InvalidVectorSize { actual, max }) => {
        assert_eq!((actual, max), (n + 1, n));
      }
      other => panic!("expected InvalidVectorSize, got {other:?}"),
    }

    // Brakedown: same check through its own commit.
    let bd_ck = BdModCommitmentKey::new(params, n).unwrap();
    assert_eq!(bd_ck.max_n, n);
    type BdMP = IntegerModPCSBd;
    type BE = crate::provider::T256DynPrimeBdEngine;
    match <BdMP as ModPCSEngineTrait<BE>>::commit(&bd_ck, &long, &()) {
      Err(SpartanError::InvalidVectorSize { actual, max }) => {
        assert_eq!((actual, max), (n + 1, n));
      }
      other => panic!("expected InvalidVectorSize, got {other:?}"),
    }
  }

  /// `blind` documents `n <= max_n` as a caller contract and enforces it
  /// with a deliberate assertion, not an accidental overflow.
  #[test]
  #[should_panic(expected = "exceeds the commitment-key capacity")]
  fn blind_asserts_the_capacity_contract() {
    let params = IntEvalParams::derive(32, 16, 9, 4).unwrap();
    let (ck, _vk) = IntegerModPCS::setup_with_params(b"cap-test", 16, 4, params).unwrap();
    let _ = <IntegerModPCS as ModPCSEngineTrait<ME>>::blind(&ck, 17);
  }
}

#[cfg(test)]
mod matched_security_tests {
  use super::*;

  #[test]
  fn native_128_target_validates_all_114_comparison_batches() {
    assert_eq!(LAMBDA_BOUND2, 117);
    assert_eq!(CHALLENGE_BITS, 128);
    for log_n in 13..=17 {
      for k in [9, 11] {
        let p = IntEvalParams::derive_with_target(2048, 64, k, log_n, 128).unwrap();
        p.validate(log_n).unwrap();
        assert_eq!(p.security_bits, LAMBDA);
        assert!(p.prime_security_bits(log_n) >= 128.0);
        let native = IntEvalParams::derive(2048, 64, k, log_n).unwrap();
        assert_eq!(bincode::serialize(&p).unwrap(), bincode::serialize(&native).unwrap());
        let mut fewer = p;
        fewer.s -= 1;
        assert!(fewer.validate(log_n).is_err());
      }
    }
  }

  #[test]
  fn target_112_derives_and_validates_all_reference_batches() {
    for log_n in 13..=17 {
      for k in [9, 11] {
        let p = IntEvalParams::derive_with_target(2048, 64, k, log_n, 112).unwrap();
        p.validate(log_n).unwrap();
        assert!(p.prime_security_bits(log_n) >= 112.0);
        let mut fewer = p.clone();
        fewer.s -= 1;
        assert!(fewer.validate(log_n).is_err());
        let legacy = IntEvalParams::derive(2048, 64, k, log_n).unwrap();
        assert_eq!(legacy.security_bits, 128);
        assert_eq!(
          (p.log_p, p.log_t, p.log_t_f),
          (legacy.log_p, legacy.log_t, legacy.log_t_f)
        );
        assert!(p.s <= legacy.s);
        if log_n == 13 {
          assert_eq!(p.s, if k == 9 { 14 } else { 27 });
        }
      }
    }
  }

  #[test]
  fn prime_count_is_conservative_and_uses_lower_interval_endpoint() {
    for bits in 5..=16 {
      let hi = 1usize << bits;
      let is_prime = |n: usize| n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0);
      let actual = ((hi / 2 + 1)..=hi).filter(|n| is_prime(*n)).count();
      assert!(log2_primes_in_top_half(bits).exp2() <= actual as f64);
      let log_y = 18.0 * (1.0 + 128.0) + 64.0;
      let conservative = log2_primes_in_top_half(bits) - (log_y / (bits - 1) as f64).log2();
      assert!((soundness_bits_per_prime(bits, 18, 64) - conservative).abs() < 1e-9);
    }
    assert!(IntEvalParams::derive_with_target(2048, 64, 9, 13, 0).is_err());
    assert!(IntEvalParams::derive_with_target(2048, 64, 9, 13, 129).is_err());
    let mut p = IntEvalParams::derive_with_target(2048, 64, 9, 13, 112).unwrap();
    p.format_version = 1;
    assert!(p.validate(13).is_err());
  }
}
