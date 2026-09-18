//! Trait scaffolding for running IntMod-Spartan over a
//! runtime-determined prime field (the dual-field driver).
//!
//! The existing `Engine` trait (in `traits/mod.rs`) bakes the sumcheck field
//! into the curve via `Engine::Scalar: PrimeField + PrimeFieldBits + …`. Those
//! `ff` traits require a compile-time-known modulus (`MODULUS: &'static str`),
//! which excludes dynamic-modulus arithmetic. The dual-field driver needs the SNARK
//! arithmetic to happen modulo a verifier-sampled ~128-bit prime, so this file
//! introduces a parallel trait hierarchy:
//!
//!   - [`SumcheckField`]: a field-like interface that does **not** require a
//!     static modulus. Implemented by both static-modulus curve scalars
//!     (via a blanket impl) and by the runtime-modulus
//!     `DynPrime` type (backed by `crypto-bigint::MontyForm`).
//!
//!   - [`SumcheckEngine`]: the minimum surface that `sumcheck.rs` needs. Both
//!     `Engine` and [`ModEngine`] implement it (via blanket impls).
//!     Loosening `impl<E: Engine> SumcheckProof<E>` to `impl<E: SumcheckEngine>`
//!     is what lets the sumcheck code run over either static or dynamic fields
//!     without algorithmic changes.
//!
//!   - [`ModEngine`]: the engine for `IntModSpartanModpSNARK`. Bundles a
//!     `SumcheckField` (the dynamic prime), a [`ModPCSEngineTrait`] (the
//!     Mod-PCS), and a transcript. **Deliberately PCS-agnostic** — does not
//!     assume the Mod-PCS sits on an elliptic-curve / Pedersen commitment;
//!     FRI-based, hash-based, or other underlying structures are also fine.
//!
//!   - [`ModPCSEngineTrait`]: PCS interface for committing polynomials over
//!     `Self::Scalar` and opening them at `Self::Scalar` points. The Mod-PCS
//!     analog of `PCSEngineTrait`. Concrete impls hide their underlying
//!     machinery (curve+Pedersen, FRI, Merkle, etc.) entirely.
//!
//! This file only defines the trait shapes. `sumcheck_modp.rs` and
//! `imod_spartan_modp.rs` consume them; `provider/mod.rs` supplies the
//! concrete `ModEngine` impls (`T256DynPrimeEngine` and the Brakedown-
//! backed variants).

use crate::{
  errors::SpartanError,
  traits::{
    Engine, PrimeFieldExt,
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use ff::{Field, PrimeField};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

/// An aligned block of witness indices `[start, start + 2^log_len)` whose
/// committed integer values are asserted to be below `2^16` — one chunk
/// of the Mod-PCS's committed representation. A sound Mod-PCS discharges
/// the assertion with zero-subcube opening claims on its chunk oracle
/// (see `provider::pcs::integer_modpcs`), so it is the SNARK-level
/// analogue of a 16-bit range lookup and costs no constraint rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmallValueBlock {
  /// First witness index; must be a multiple of `2^log_len`.
  pub start: usize,
  /// Log2 of the block length.
  pub log_len: usize,
}

impl SmallValueBlock {
  /// Block length `2^log_len`.
  pub fn size(&self) -> usize {
    1usize << self.log_len
  }

  /// Check alignment and containment in a polynomial of `2^num_vars`
  /// coefficients.
  pub fn validate(&self, num_vars: usize) -> Result<(), SpartanError> {
    let ok = self.log_len <= num_vars
      && self.start.is_multiple_of(self.size())
      && self.start + self.size() <= (1usize << num_vars);
    if ok {
      Ok(())
    } else {
      Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SmallValueBlock {{ start: {}, log_len: {} }} is misaligned or out of range for 2^{num_vars} coefficients",
          self.start, self.log_len
        ),
      })
    }
  }
}

/// A field-like interface suitable for sumcheck arithmetic.
///
/// Unlike `ff::PrimeField`, does **not** require a compile-time-known modulus.
/// This lets the trait be implemented by both static-modulus curve scalars
/// and by `DynPrime`-style dynamic-modulus types.
///
/// The trait provides only what the sumcheck and MLE code requires:
/// arithmetic, additive/multiplicative identities, `u64` casting, inversion,
/// and a way to reduce raw bytes (e.g. from a transcript squeeze) into the
/// field. It deliberately omits anything that assumes a static modulus
/// (`MODULUS`, `NUM_BITS`, `S`, etc.).
pub trait SumcheckField:
  Sized
  + Copy
  + Clone
  + Send
  + Sync
  + Debug
  + PartialEq
  + Eq
  + Add<Output = Self>
  + Sub<Output = Self>
  + Mul<Output = Self>
  + Neg<Output = Self>
  + AddAssign
  + SubAssign
  + MulAssign
  + TranscriptReprTrait
  + 'static
{
  /// Per-modulus context. For static-modulus fields this is `()` (the
  /// modulus is baked into the type at compile time). For dynamic-modulus
  /// fields like `DynPrime`, this carries the runtime modulus and the
  /// Montgomery constants derived from it.
  ///
  /// Constructors (`zero`, `one`, `from_u64`) take `&Self::Params` because
  /// for runtime moduli they need the modulus to materialize a value.
  /// Arithmetic ops between two `Self` values don't need params because
  /// dynamic-field implementors (e.g. `crypto-bigint::MontyForm`) carry
  /// the params per element internally.
  type Params: Clone + Debug + Send + Sync + 'static;

  /// The additive identity. Replaces `ff::Field::ZERO`; method form so it
  /// works for runtime-modulus fields where the const form would not be
  /// const-evaluable.
  fn zero(params: &Self::Params) -> Self;

  /// The multiplicative identity. Replaces `ff::Field::ONE`; method form
  /// for the same reason as `zero()`.
  fn one(params: &Self::Params) -> Self;

  /// Cast a `u64` into the field, reducing modulo the field's modulus.
  fn from_u64(params: &Self::Params, v: u64) -> Self;

  /// Multiplicative inverse. Returns `None` if `*self` is the additive
  /// identity (which has no inverse).
  fn invert(&self) -> Option<Self>;

  /// Serialize the field element to little-endian bytes. Used by
  /// `polys_modp` when implementing `TranscriptReprTrait` for round
  /// polynomials.
  fn to_le_bytes(&self) -> Vec<u8>;

  /// Reduce raw bytes (from a transcript squeeze) into a field element.
  /// Used to derive Fiat-Shamir challenges. For static-modulus fields this
  /// forwards to `PrimeFieldExt::from_uniform`; for dynamic-modulus fields
  /// it builds an integer from the bytes and reduces mod the modulus.
  ///
  /// NOTE: the dynamic-field implementation currently consumes
  /// at most `LIMBS * 8` bytes, so for a modulus near that width the
  /// challenge is slightly biased. Fine for the prototype; a
  /// soundness-grade version needs wide reduction (≥ modulus_bits + 128
  /// input bits).
  fn from_bytes_reduce(params: &Self::Params, bytes: &[u8]) -> Self;

  /// Whether this value belongs to the modulus context `expected`. Static
  /// fields return `true` (the modulus is baked into the type); dynamic
  /// fields compare the modulus carried by the value with
  /// `expected`'s. Verifiers use this to reject proof-carried values from
  /// a foreign context *before* any mixed-context arithmetic can panic.
  fn is_in_context(&self, expected: &Self::Params) -> bool;
}

/// Blanket impl: any static-modulus prime field that already implements
/// `PrimeField + PrimeFieldExt` (i.e. every `Engine::Scalar` in this codebase)
/// automatically satisfies `SumcheckField`. This is the backward-compat
/// path for the static-field code — it keeps running without any per-type impls.
impl<F> SumcheckField for F
where
  F: PrimeField + PrimeFieldExt + TranscriptReprTrait + Copy + Send + Sync + 'static,
{
  type Params = ();

  fn zero(_: &()) -> Self {
    <F as Field>::ZERO
  }
  fn one(_: &()) -> Self {
    <F as Field>::ONE
  }
  fn from_u64(_: &(), v: u64) -> Self {
    Self::from(v)
  }
  fn invert(&self) -> Option<Self> {
    <F as Field>::invert(self).into()
  }
  fn to_le_bytes(&self) -> Vec<u8> {
    self.to_repr().as_ref().to_vec()
  }
  fn from_bytes_reduce(_: &(), bytes: &[u8]) -> Self {
    <F as PrimeFieldExt>::from_uniform(bytes)
  }
  fn is_in_context(&self, _: &()) -> bool {
    true
  }
}

/// The minimum engine surface that `sumcheck.rs` and the MLE polynomial
/// operations need.
///
/// Both `Engine` and [`ModEngine`] satisfy this (via blanket
/// impls). Loosening the bound on `SumcheckProof<E>` from
/// `E: Engine` to `E: SumcheckEngine` is what lets the same sumcheck code
/// run over the curve scalar or over a dynamic prime field.
pub trait SumcheckEngine:
  Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq + 'static
{
  /// The field over which the sumcheck arithmetic runs. Round polynomials,
  /// challenges, sumcheck claims, and MLE evaluations all live in `Scalar`.
  type Scalar: SumcheckField;

  /// Transcript engine that supports Fiat-Shamir absorption and
  /// challenge-squeeze of `Self::Scalar` values.
  type TE: TranscriptEngineTrait<Self>;
}

/// Blanket impl: any `Engine` is a `SumcheckEngine` (its `Scalar` is the
/// curve scalar, which satisfies `SumcheckField` via the blanket impl
/// above; its `TE` is just the engine's own transcript). This is the
/// backward-compat path — static-field SNARK code can switch its trait
/// bound from `E: Engine` to `E: SumcheckEngine` without changing any
/// concrete impl.
impl<E: Engine> SumcheckEngine for E {
  type Scalar = E::Scalar;
  type TE = E::TE;
}

/// The engine for the dual-field `IntModSpartanModpSNARK`.
///
/// Pairs a `SumcheckField` (the runtime-sampled prime field) with a Mod-PCS
/// that commits integer-valued polynomials and opens them at points in
/// `Self::Scalar^n`. The underlying curve and its static-modulus PCS are
/// hidden inside [`ModEngine::ModPCS`] — the SNARK code never sees the
/// commitment field directly.
pub trait ModEngine: SumcheckEngine {
  /// The Mod-PCS. Commits polynomials whose evaluations are in
  /// `Self::Scalar` and opens them at `Self::Scalar^n` points.
  ///
  /// Deliberately PCS-agnostic: this engine does **not** expose the
  /// underlying commitment machinery (curve+Pedersen, FRI, Merkle,
  /// lattice, etc.). Each `ModPCSEngineTrait` impl knows its own
  /// internals; the SNARK code that calls it does not.
  type ModPCS: ModPCSEngineTrait<Self>;

  /// Placeholder `Params` used to construct the transcript *before* the
  /// real `p` is sampled. The transcript's typed `squeeze<F>` is never
  /// called against this placeholder — only the byte-level absorb/squeeze
  /// operations run pre-`sample_params`. After sampling, the driver calls
  /// `Keccak256Transcript::set_params` with the real `params`.
  ///
  /// For static-modulus fields, return the `Params` default (`()`). For
  /// dynamic-modulus fields, return any valid params (e.g. the smallest
  /// valid odd modulus). No default impl: requiring every `ModEngine`
  /// impl to spell this out avoids the `Params: Default` bound rippling
  /// into every driver method that touches a transcript.
  fn bootstrap_params() -> <Self::Scalar as SumcheckField>::Params;

  /// Sample the runtime modulus context for `Self::Scalar` from a
  /// `ByteTranscript`. For static-modulus fields, return the `Params`
  /// default (`()`) and absorb/squeeze nothing. For dynamic-modulus
  /// fields, run the actual rejection-sampling / primality-testing loop
  /// that drives the verifier-sampled prime `p`.
  fn sample_params<T: crate::traits::transcript::ByteTranscript>(
    transcript: &mut T,
  ) -> <Self::Scalar as SumcheckField>::Params;
}

/// PCS interface for committing polynomials whose evaluations come from a
/// dynamic-prime field, and opening them at points in that field.
///
/// Mod-PCS analog of `PCSEngineTrait`. Concrete implementations handle the
/// `Self::Scalar` ↔ underlying-field conversion internally (limb-splitting
/// plus the IntEval small-prime fingerprinting in `IntegerModPCS`).
///
/// The interface deliberately mirrors `PCSEngineTrait` so the SNARK code that
/// calls it (`IntModSpartanSNARK::prove`/`verify`) has the same shape as
/// today, just with `Self::Scalar` substituted for `E::Scalar`.
pub trait ModPCSEngineTrait<E: ModEngine>: Clone + Send + Sync {
  /// Commitment key.
  type CommitmentKey: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Verifier key.
  type VerifierKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Commitment value, absorbable into the transcript.
  ///
  /// `TranscriptReprTrait` is marker-free (no `<G: Group>`) since the
  /// transcript trait split — non-group-based PCSes can implement
  /// it without inventing a placeholder curve group.
  type Commitment: Clone
    + Debug
    + Send
    + Sync
    + PartialEq
    + Eq
    + TranscriptReprTrait
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Blind / commitment randomizer.
  type Blind: Clone + Debug + Send + Sync + PartialEq + Eq + Serialize + for<'de> Deserialize<'de>;

  /// Evaluation-argument data sent in the proof.
  type EvaluationArgument: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Evaluation-argument data for a *batched* multi-polynomial opening
  /// ([`prove_batch`](Self::prove_batch)). Sound impls share the
  /// expensive per-open machinery (range checks, inner-product argument)
  /// across all polynomials.
  type BatchEvaluationArgument: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Sample commitment keys for vectors of length up to `n`.
  fn setup(
    label: &'static [u8],
    n: usize,
    width: usize,
  ) -> (Self::CommitmentKey, Self::VerifierKey);

  /// Eagerly initialize any lazily-computed tables in the commitment key.
  /// Default no-op; override to match `PCSEngineTrait::precompute_ck`.
  fn precompute_ck(_ck: &Self::CommitmentKey) {}

  /// Sample a fresh blind suitable for committing a polynomial of length `n`.
  ///
  /// Mirrors `PCSEngineTrait::blind` for engines that wrap a field
  /// PCS. For future hash-based / non-hiding Mod-PCS impls, this can
  /// return a unit-typed sentinel.
  fn blind(ck: &Self::CommitmentKey, n: usize) -> Self::Blind;

  /// Commit to an **integer-valued** polynomial. Each entry is a
  /// non-negative bounded integer in `BigUint` form — `p`-independent and
  /// chosen before the runtime prime `p` is sampled, so the commitment
  /// binds the integers themselves.
  ///
  /// The impl chooses any commitment-internal representation (small-scalar
  /// fast paths, limb-splitting, etc.) itself — the universal surface does
  /// not expose group/Pedersen-specific hints.
  fn commit(
    ck: &Self::CommitmentKey,
    v: &[BigUint],
    r: &Self::Blind,
  ) -> Result<Self::Commitment, SpartanError>;

  /// Length / shape sanity check on a commitment.
  fn check_commitment(comm: &Self::Commitment, n: usize, width: usize) -> Result<(), SpartanError>;

  /// The commitment key's native value-width bound in bits (the width a
  /// plain [`commit`](Self::commit) uses). Width-grouped segments commit at
  /// `<= ` this via [`commit_at`](Self::commit_at).
  fn commitment_log_t_f(ck: &Self::CommitmentKey) -> usize;

  /// Verifier-key mirror of [`commitment_log_t_f`](Self::commitment_log_t_f).
  fn verifier_log_t_f(vk: &Self::VerifierKey) -> usize;

  /// Prove that the integer-valued polynomial `poly` evaluates at the
  /// `Z_p` point `point` to `eval` (the canonical integer in `[0, p)`
  /// representing the `Z_p` evaluation).
  ///
  /// PCS-agnostic: the surface carries no group/Pedersen "commitment to the
  /// evaluation" — a sound Mod-PCS impl binds `eval` however its own
  /// machinery requires (e.g. a Pedersen-backed impl manages its own
  /// eval-commitment key internally; a hash/FRI impl needs none).
  fn prove(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[BigUint],
    blind: &Self::Blind,
    point: &[E::Scalar],
    eval: &BigUint,
  ) -> Result<Self::EvaluationArgument, SpartanError>;

  /// Verify a polynomial opening. `eval` is the canonical integer in
  /// `[0, p)` representing the claimed `Z_p` evaluation; the IntEval
  /// protocol uses it to check `int_v' ≡ eval (mod p)`.
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &BigUint,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError>;

  /// Prove a *batch* of openings — polynomial `polys[i]` (committed by
  /// `comms[i]` with blind `blinds[i]`) evaluates at `points[i]` to
  /// `evals[i]` — in ONE argument. The `i`-th opening uses the same
  /// arguments it would as a standalone [`prove`](Self::prove) call; the
  /// distinction is that sound impls discharge all openings through a
  /// single shared range check and inner-product argument rather than
  /// repeating that fixed per-open work `polys.len()` times. All slices
  /// have equal length; `points[i]` may differ in length (the
  /// polynomials need not share a number of variables). The transcript is
  /// threaded through all openings in index order.
  fn prove_batch(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comms: &[&Self::Commitment],
    polys: &[&[BigUint]],
    blinds: &[&Self::Blind],
    points: &[&[E::Scalar]],
    evals: &[&BigUint],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError>;

  /// Verify a batched opening produced by [`prove_batch`](Self::prove_batch).
  /// `comms`, `points`, and `evals` mirror the prover's inputs in the
  /// same index order.
  fn verify_batch(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comms: &[&Self::Commitment],
    points: &[&[E::Scalar]],
    evals: &[&BigUint],
    arg: &Self::BatchEvaluationArgument,
  ) -> Result<(), SpartanError>;

  /// [`prove_batch`](Self::prove_batch) plus per-polynomial
  /// [`SmallValueBlock`] assertions: `blocks[i]` lists the blocks of
  /// `polys[i]` whose values are asserted `< 2^16`. The default only
  /// supports the trivial (all-empty) case.
  #[allow(clippy::too_many_arguments)]
  fn prove_batch_with_blocks(
    ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comms: &[&Self::Commitment],
    polys: &[&[BigUint]],
    blinds: &[&Self::Blind],
    points: &[&[E::Scalar]],
    evals: &[&BigUint],
    blocks: &[&[SmallValueBlock]],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError> {
    if blocks.iter().any(|b| !b.is_empty()) {
      return Err(SpartanError::InternalError {
        reason: "this Mod-PCS does not support small-value blocks".to_string(),
      });
    }
    Self::prove_batch(ck, transcript, comms, polys, blinds, points, evals)
  }

  /// Verify a [`prove_batch_with_blocks`](Self::prove_batch_with_blocks)
  /// argument; `blocks` mirrors the prover's declaration.
  fn verify_batch_with_blocks(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comms: &[&Self::Commitment],
    points: &[&[E::Scalar]],
    evals: &[&BigUint],
    arg: &Self::BatchEvaluationArgument,
    blocks: &[&[SmallValueBlock]],
  ) -> Result<(), SpartanError> {
    if blocks.iter().any(|b| !b.is_empty()) {
      return Err(SpartanError::InternalError {
        reason: "this Mod-PCS does not support small-value blocks".to_string(),
      });
    }
    Self::verify_batch(vk, transcript, comms, points, evals, arg)
  }

  /// Commit `v` as an integer polynomial whose values are bounded by
  /// `2^log_t_f` bits — a width-grouped commitment *segment*. A narrower
  /// bound lets the impl commit at fewer internal limbs (cheaper MSM /
  /// range check). The default only supports the key's native width.
  fn commit_at(
    _ck: &Self::CommitmentKey,
    _v: &[BigUint],
    _r: &Self::Blind,
    _log_t_f: usize,
  ) -> Result<Self::Commitment, SpartanError> {
    Err(SpartanError::InternalError {
      reason: "this Mod-PCS does not support width-grouped commitment".to_string(),
    })
  }

  /// [`prove_batch_with_blocks`](Self::prove_batch_with_blocks) where
  /// polynomial `i` was committed at width `log_t_fs[i]` bits (its
  /// width-grouped segment bound). Every poly shares one range check and
  /// combined opening; the per-poly width only changes its own limb count.
  /// The default is unsupported.
  #[allow(clippy::too_many_arguments)]
  fn prove_batch_with_params(
    _ck: &Self::CommitmentKey,
    _transcript: &mut E::TE,
    _comms: &[&Self::Commitment],
    _polys: &[&[BigUint]],
    _blinds: &[&Self::Blind],
    _points: &[&[E::Scalar]],
    _evals: &[&BigUint],
    _blocks: &[&[SmallValueBlock]],
    _log_t_fs: &[usize],
  ) -> Result<Self::BatchEvaluationArgument, SpartanError> {
    Err(SpartanError::InternalError {
      reason: "this Mod-PCS does not support width-grouped commitment".to_string(),
    })
  }

  /// Verify a [`prove_batch_with_params`](Self::prove_batch_with_params)
  /// argument; `log_t_fs` mirrors the prover's per-poly segment widths.
  #[allow(clippy::too_many_arguments)]
  fn verify_batch_with_params(
    _vk: &Self::VerifierKey,
    _transcript: &mut E::TE,
    _comms: &[&Self::Commitment],
    _points: &[&[E::Scalar]],
    _evals: &[&BigUint],
    _arg: &Self::BatchEvaluationArgument,
    _blocks: &[&[SmallValueBlock]],
    _log_t_fs: &[usize],
  ) -> Result<(), SpartanError> {
    Err(SpartanError::InternalError {
      reason: "this Mod-PCS does not support width-grouped commitment".to_string(),
    })
  }

  /// Whether the batch evaluation argument carries no sampled-prime
  /// (`E::Scalar`) data outside `expected` — the Mod-PCS counterpart of
  /// [`SumcheckField::is_in_context`], checked by verify to reject a proof
  /// built under different params. Defaults to `true`: a Mod-PCS whose
  /// argument is pure q-side data (static scalars and hashes, independent
  /// of the sampled prime) is always in context. Backends whose argument
  /// embeds `E::Scalar` data override this.
  fn batch_arg_is_in_context(
    _arg: &Self::BatchEvaluationArgument,
    _expected: &<E::Scalar as SumcheckField>::Params,
  ) -> bool {
    true
  }
}
