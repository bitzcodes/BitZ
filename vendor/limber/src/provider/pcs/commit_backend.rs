//! The interface separating the integer Mod-PCS *protocol* from the
//! *commitment scheme* it runs on.
//!
//! The protocol — sumchecks, the per-prime chains, the range check —
//! is ordinary field arithmetic and never manipulates commitments
//! algebraically. It needs a commitment scheme for exactly two things:
//!
//! 1. committing to polynomials over the Hyrax scalar field, and
//! 2. proving, at the very end, what each committed polynomial
//!    evaluates to at one point (the protocol has already reduced all
//!    of its claims down to one evaluation per commitment).
//!
//! [`CommitBackend`] captures those two operations. Two
//! implementations exist: the Pedersen/Hyrax one (`HyBackend`, in
//! `integer_modpcs.rs`), which keeps the existing protocol
//! byte-for-byte — including its trick of merging all the final
//! evaluation proofs into a single inner-product argument, which works
//! because Pedersen commitments can be added together — and the
//! hash-based Brakedown one ([`BdBackend`]), which cannot merge
//! commitments and instead proves each evaluation separately with
//! Merkle-tree column openings. Brakedown commitments have no
//! randomness, so that instantiation is not zero-knowledge (the same
//! trade-off Zinc+ makes).

use crate::{
  errors::SpartanError,
  provider::pcs::brakedown::{
    BrakedownCommitData, BrakedownDirectOpen, BrakedownGroupArg, BrakedownParams, brakedown_commit,
    brakedown_commit_plain, brakedown_open_direct, brakedown_open_group, brakedown_verify_direct,
    brakedown_verify_group,
  },
  traits::{PrimeFieldExt, transcript::ByteTranscript},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{
  collections::HashMap,
  sync::{Mutex, OnceLock},
};

/// Security level for Brakedown layouts (column-open count derivation).
/// Aligned with the honest system floor: the 2^-114 fingerprint
/// prime-sampling term is the weakest link, so provisioning the column
/// count above 114 over-secures this one term (cf. LAMBDA_BOUND2 = 117
/// for the challenge bound). Swept: each ~10 bits is ~0.5 MB of proof.
const BD_LAMBDA_DEFAULT: usize = 114;
/// Column-open security parameter, `BDLAMBDA` overrides (benching knob).
/// Default 114; the honest system floor is ~114 (the 2^-114 fingerprint
/// term), so values in [114,117] are free, below 114 lowers real security.
fn bd_lambda() -> usize {
  std::env::var("BDLAMBDA")
    .ok()
    .map(|v| v.parse::<usize>().expect("invalid BDLAMBDA"))
    .unwrap_or(BD_LAMBDA_DEFAULT)
}
/// All runtime Brakedown knobs affecting the committed code or opening.
pub(crate) fn bd_configuration() -> [usize; 4] {
  let read = |name: &str, default| {
    std::env::var(name)
      .map(|v| v.parse::<usize>().expect("invalid Brakedown parameter"))
      .unwrap_or(default)
  };
  let cfg = [
    bd_lambda(),
    read("BDSPEC", 4),
    read("BDROWLEN", 1 << 15),
    bd_direct_max(),
  ];
  assert!(
    (64..=128).contains(&cfg[0])
      && cfg[1] < crate::provider::pcs::brakedown::SPECS.len()
      && cfg[2].is_power_of_two(),
    "invalid Brakedown configuration"
  );
  cfg
}
/// Public seed for the deterministic expander-code matrices; both prover
/// and verifier derive identical layouts from (length, spec, seed).
const BD_SEED: &[u8] = b"imod-modpcs-brakedown-v1";
/// Targets of at most this many coefficients ship their (compact)
/// polynomial directly instead of a column-opening argument: below this
/// size the plaintext is smaller than the argument, and the verifier
/// just recommits and evaluates. BDDIRECT overrides (benching knob).
const BD_DIRECT_MAX: usize = 1 << 16;

fn bd_direct_max() -> usize {
  std::env::var("BDDIRECT")
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(BD_DIRECT_MAX)
}

/// The Brakedown backend's batch-opening argument: column-opening
/// groups for the large targets (in group order) and directly-shipped
/// polynomials for the small ones (in canonical target order). The
/// partition and grouping are deterministic functions of the target
/// list, so the verifier reconstructs them without transport.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
  serialize = "F: serde::Serialize + ff::PrimeField",
  deserialize = "F: serde::de::DeserializeOwned + ff::PrimeField"
))]
pub struct BdBatchOpenArg<F> {
  /// Grouped column-opening arguments for targets above the direct-ship
  /// threshold.
  pub groups: Vec<BrakedownGroupArg<F>>,
  /// Directly-shipped small polynomials.
  pub direct: Vec<BrakedownDirectOpen<F>>,
}

/// Per-length Brakedown layout cache (code sampling is deterministic in
/// the public seed, so prover and verifier agree without transport).
pub(crate) fn bd_params<F: crate::traits::PrimeFieldExt>(n: usize) -> &'static BrakedownParams<F> {
  use std::any::{Any, TypeId};
  // One cache across all field instantiations, keyed by (field, length);
  // generic statics are unavailable, so entries go through `dyn Any`.
  static CACHE: OnceLock<
    Mutex<HashMap<(TypeId, usize, [usize; 4]), &'static (dyn Any + Send + Sync)>>,
  > = OnceLock::new();
  let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
  let mut guard = cache.lock().expect("bd params cache poisoned");
  let config = bd_configuration();
  let key = (TypeId::of::<F>(), n, config);
  if let Some(p) = guard.get(&key) {
    return p.downcast_ref::<BrakedownParams<F>>().expect("cache type");
  }
  // BDSPEC=<0..5> overrides the code spec (benching knob; prover and
  // verifier share the process, so layouts agree).
  let spec = crate::provider::pcs::brakedown::SPECS[config[1]];
  let row_len = config[2].min(n);
  let params: &'static BrakedownParams<F> = Box::leak(Box::new(BrakedownParams::new_with_row_len(
    n, spec, config[0], BD_SEED, row_len,
  )));
  guard.insert(key, params as &'static (dyn Any + Send + Sync));
  params
}

/// Bounded cache of commit-time opening data, keyed by Merkle root, so
/// `prove` does not re-encode witness polynomials committed moments
/// earlier through the ModPCS surface. Purely a prover-side
/// memoization: on a miss the data is recomputed and checked against
/// the expected root.
///
/// Carries fixed audit instrumentation (per-field-type hit / miss /
/// re-encode / wholesale-clear counters), updated under the cache's own
/// mutex — no extra atomic or lock — and enabled for every Brakedown
/// run, so an audited run never measures a different binary from an
/// unaudited one. Instance-based so unit tests can exercise a private
/// instance deterministically; production uses one process-global
/// instance ([`bd_retained_cache`]).
struct BdRetainedCache {
  state: Mutex<BdRetainedCacheState>,
}

#[derive(Default)]
struct BdRetainedCacheState {
  map: HashMap<(std::any::TypeId, [u8; 32]), Box<dyn std::any::Any + Send + Sync>>,
  stats: HashMap<std::any::TypeId, super::BdRetainedCacheStats>,
}

impl BdRetainedCache {
  fn new() -> Self {
    Self {
      state: Mutex::new(BdRetainedCacheState::default()),
    }
  }

  fn put<F: 'static>(&self, root: [u8; 32], data: Box<dyn std::any::Any + Send + Sync>) {
    let mut guard = self.state.lock().expect("bd data cache poisoned");
    if guard.map.len() >= 8 {
      // Wholesale clear, not LRU eviction; attributed to the inserting
      // field type.
      guard.map.clear();
      guard
        .stats
        .entry(std::any::TypeId::of::<F>())
        .or_default()
        .wholesale_clears += 1;
    }
    guard.map.insert((std::any::TypeId::of::<F>(), root), data);
  }

  fn get<F: 'static, D: Clone + 'static>(&self, root: &[u8; 32]) -> Option<D> {
    let mut guard = self.state.lock().expect("bd data cache poisoned");
    let found = guard
      .map
      .get(&(std::any::TypeId::of::<F>(), *root))
      .and_then(|b| b.downcast_ref::<D>().cloned());
    let stats = guard.stats.entry(std::any::TypeId::of::<F>()).or_default();
    if found.is_some() {
      stats.hits += 1;
    } else {
      stats.misses += 1;
    }
    found
  }

  /// A cache miss forced a full re-encode on the recommit path.
  fn note_reencode<F: 'static>(&self) {
    let mut guard = self.state.lock().expect("bd data cache poisoned");
    guard
      .stats
      .entry(std::any::TypeId::of::<F>())
      .or_default()
      .reencodes += 1;
  }

  /// Clear every retained entry (all field types) and zero `F`'s
  /// counters, giving a deterministic empty-cache starting state.
  fn reset<F: 'static>(&self) {
    let mut guard = self.state.lock().expect("bd data cache poisoned");
    guard.map.clear();
    guard.stats.insert(
      std::any::TypeId::of::<F>(),
      super::BdRetainedCacheStats::default(),
    );
  }

  fn stats<F: 'static>(&self) -> super::BdRetainedCacheStats {
    self
      .state
      .lock()
      .expect("bd data cache poisoned")
      .stats
      .get(&std::any::TypeId::of::<F>())
      .copied()
      .unwrap_or_default()
  }
}

fn bd_data_cache_put<F: crate::traits::PrimeFieldExt>(
  root: [u8; 32],
  data: BrakedownCommitData<F>,
) {
  bd_retained_cache().put::<F>(root, Box::new(data));
}

fn bd_data_cache_get<F: crate::traits::PrimeFieldExt>(
  root: &[u8; 32],
) -> Option<BrakedownCommitData<F>> {
  bd_retained_cache().get::<F, BrakedownCommitData<F>>(root)
}

fn bd_retained_cache() -> &'static BdRetainedCache {
  static CACHE: OnceLock<BdRetainedCache> = OnceLock::new();
  CACHE.get_or_init(BdRetainedCache::new)
}

/// Clear the process-global retained cache and zero `F`'s audit
/// counters. `pub(super)` — Rust privacy is visible-to-self-and-
/// descendants, so the parent module's public `#[doc(hidden)]` wrappers
/// delegate here; the cache statics and their storage stay private.
pub(super) fn bd_retained_cache_reset_for<F: 'static>() {
  bd_retained_cache().reset::<F>();
}

/// Snapshot `F`'s retained-cache audit counters (see the reset
/// counterpart for the visibility rationale).
pub(super) fn bd_retained_cache_stats_for<F: 'static>() -> super::BdRetainedCacheStats {
  bd_retained_cache().stats::<F>()
}

/// One final-opening target after the interleaved claim reduction: the
/// commitment, its polynomial (and any retained commit data), the
/// reduced point, and the claimed evaluation (already transcript-bound
/// by the claim reduction).
pub struct OpenTarget<'a, B: CommitBackend> {
  pub comm: &'a B::Comm,
  pub poly: &'a [B::Scalar],
  pub blind: &'a B::Blind,
  pub data: &'a B::Data,
  pub point: Vec<B::Scalar>,
  pub eval: B::Scalar,
}

/// The F-side commitment backend seam. `Data` is whatever the prover
/// retains alongside a commitment to answer openings (Hyrax: the blind;
/// Brakedown: the encoded matrix + Merkle tree).
pub trait CommitBackend: Sized + Send + Sync + 'static {
  /// The prime field the committed polynomials live over (the q-side
  /// field). Today both backends set this to `t256::Scalar`; making it
  /// an associated type is what lets the same protocol instantiate
  /// over a smaller field (e.g. M127) without touching protocol code.
  type Scalar: PrimeFieldExt
    + crate::traits::transcript::TranscriptReprTrait
    + Serialize
    + DeserializeOwned
    + Send
    + Sync
    + 'static;
  /// A curve-free engine naming this backend's (Scalar, transcript)
  /// pair, for the sub-protocols (LogUp-GKR) that are generic over
  /// `SumcheckEngine`. Full `Engine`s satisfy this via the blanket
  /// impl; a small-field backend mints a ~20-line struct.
  type SE: crate::traits::mod_engine::SumcheckEngine<Scalar = Self::Scalar>;
  type Ck: Send + Sync + Clone;
  type Vk: Send + Sync + Clone;
  type Comm: Clone + core::fmt::Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync;
  type Blind: Clone + core::fmt::Debug + Serialize + DeserializeOwned + Send + Sync;
  type Data: Send + Sync;
  type BatchOpenArg: Clone + core::fmt::Debug + Serialize + DeserializeOwned + Send + Sync;

  /// Fresh commitment randomness for an `n`-coefficient polynomial
  /// (`()` for non-hiding backends).
  fn blind(ck: &Self::Ck, n: usize) -> Self::Blind;

  /// Transcript representation of a commitment (`absorb`-equivalent).
  fn comm_transcript_bytes(comm: &Self::Comm) -> Vec<u8>;

  /// Regenerate the retained opening data for a polynomial that was
  /// committed elsewhere (the ModPCS commit surface returns only the
  /// commitment). `comm` is the expected commitment. Free for Hyrax
  /// (`()`); Brakedown serves it from a small cache filled at commit
  /// time, re-encoding only on a miss.
  fn recommit_data(
    ck: &Self::Ck,
    comm: &Self::Comm,
    poly: &[Self::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<Self::Data, SpartanError>;

  /// Commit to an F-polynomial under `blind`. `small` hints that every
  /// coefficient is < 2^16 (Hyrax small-scalar MSM path; Brakedown
  /// ignores it). Deterministic given `(poly, blind)` — callers
  /// recommit to check commitment equality.
  fn commit(
    ck: &Self::Ck,
    poly: &[Self::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<(Self::Comm, Self::Data), SpartanError>;

  /// Discharge every target's single-point evaluation claim against a
  /// shared sub-transcript (the claims and evals are already bound to it
  /// by the claim-reduction phase).
  fn open_targets(
    ck: &Self::Ck,
    targets: &[OpenTarget<'_, Self>],
    sub: &mut impl ByteTranscript,
  ) -> Result<Self::BatchOpenArg, SpartanError>;

  /// Verifier mirror of [`Self::open_targets`].
  fn verify_targets(
    vk: &Self::Vk,
    targets: &[(&Self::Comm, Vec<Self::Scalar>, Self::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError>;
}

/// Brakedown (hash-based) backend: non-hiding, MSM-free, per-target
/// tensor-IOPP openings. The comparison instantiation.
#[derive(Clone, Debug)]
pub struct BdBackend<SE = crate::provider::T256HyraxEngine>(core::marker::PhantomData<SE>);

impl<SE> CommitBackend for BdBackend<SE>
where
  SE: crate::traits::mod_engine::SumcheckEngine,
  SE::Scalar: PrimeFieldExt
    + crate::traits::transcript::TranscriptReprTrait
    + Serialize
    + DeserializeOwned
    + Send
    + Sync
    + 'static,
{
  type Scalar = SE::Scalar;
  type SE = SE;
  type Ck = ();
  type Vk = ();
  type Comm = [u8; 32];
  type Blind = ();
  type Data = BrakedownCommitData<SE::Scalar>;
  type BatchOpenArg = BdBatchOpenArg<SE::Scalar>;

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
    let params = bd_params(poly.len().next_power_of_two());
    let mut padded;
    let poly = if poly.len() == params.poly_len() {
      poly
    } else {
      padded = poly.to_vec();
      padded.resize(params.poly_len(), Self::Scalar::from(0u64));
      &padded[..]
    };
    // Below the direct-ship threshold the opening ships the polynomial
    // itself, so the commitment is just a plain hash of its canonical
    // bytes — no encoding, no Merkle tree (see `open_targets`).
    let (root, data) = if poly.len() <= bd_direct_max() {
      brakedown_commit_plain(poly)
    } else {
      brakedown_commit(params, poly)
    };
    bd_data_cache_put(root, data.clone());
    Ok((root, data))
  }

  fn recommit_data(
    ck: &Self::Ck,
    comm: &Self::Comm,
    poly: &[Self::Scalar],
    blind: &Self::Blind,
    small: bool,
  ) -> Result<Self::Data, SpartanError> {
    if let Some(data) = bd_data_cache_get(comm) {
      return Ok(data);
    }
    // Miss: this recommit is a full re-encode (audited).
    bd_retained_cache().note_reencode::<SE::Scalar>();
    let (root, data) = Self::commit(ck, poly, blind, small)?;
    if &root != comm {
      return Err(SpartanError::InternalError {
        reason: "brakedown recommit: root mismatch with the input commitment".to_string(),
      });
    }
    Ok(data)
  }

  fn open_targets(
    _ck: &Self::Ck,
    targets: &[OpenTarget<'_, Self>],
    sub: &mut impl ByteTranscript,
  ) -> Result<Self::BatchOpenArg, SpartanError> {
    // Small targets ship their polynomial directly. The rest: targets
    // whose layouts share a code (uniform row length) and whose points
    // share their column suffix form ONE group with ONE proximity row
    // and ONE gamma-combined evaluation row (per-target rows dominated
    // the proof); with the uniform-row-length policy and the claim
    // reduction's shared challenge tail, that is normally every large
    // target. Both the partition and the grouping are by first
    // appearance in the canonical target order, so the verifier
    // reconstructs them deterministically.
    let direct_max = bd_direct_max();
    let mut direct = Vec::new();
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, t) in targets.iter().enumerate() {
      if 1usize << t.point.len() <= direct_max {
        direct.push(brakedown_open_direct(t.poly));
        continue;
      }
      let params = bd_params::<SE::Scalar>(1usize << t.point.len());
      let lc = params.row_len.trailing_zeros() as usize;
      match groups.iter_mut().find(|(rep, _)| {
        let rp = &targets[*rep].point;
        bd_params::<SE::Scalar>(1usize << rp.len()).row_len == params.row_len
          && rp[rp.len() - lc..] == t.point[t.point.len() - lc..]
      }) {
        Some((_, members)) => members.push(i),
        None => groups.push((i, vec![i])),
      }
    }
    let mut args = Vec::with_capacity(groups.len());
    for (rep, members) in &groups {
      let params = bd_params(1usize << targets[*rep].point.len());
      let items: Vec<(&[u8; 32], &BrakedownCommitData<SE::Scalar>, &[SE::Scalar])> = members
        .iter()
        .map(|&i| {
          (
            targets[i].comm,
            targets[i].data,
            targets[i].point.as_slice(),
          )
        })
        .collect();
      let (evals, arg) = brakedown_open_group(params, &items, sub)?;
      for (k, &i) in members.iter().enumerate() {
        debug_assert_eq!(
          evals[k], targets[i].eval,
          "claim reduction / opening eval mismatch"
        );
        let _ = (k, i);
      }
      args.push(arg);
    }
    Ok(BdBatchOpenArg {
      groups: args,
      direct,
    })
  }

  fn verify_targets(
    _vk: &Self::Vk,
    targets: &[(&Self::Comm, Vec<Self::Scalar>, Self::Scalar)],
    arg: &Self::BatchOpenArg,
    sub: &mut impl ByteTranscript,
  ) -> Result<(), SpartanError> {
    // Reconstruct the prover's direct/grouped partition and grouping
    // (shared code + shared column suffix) from the canonical target
    // order.
    let direct_max = bd_direct_max();
    let mut n_direct = 0usize;
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, (comm, point, eval)) in targets.iter().enumerate() {
      if 1usize << point.len() <= direct_max {
        let a = arg
          .direct
          .get(n_direct)
          .ok_or_else(|| SpartanError::ProofVerifyError {
            reason: "brakedown backend: missing direct opening".to_string(),
          })?;
        brakedown_verify_direct(comm, point, *eval, a)?;
        n_direct += 1;
        continue;
      }
      let params = bd_params::<SE::Scalar>(1usize << point.len());
      let lc = params.row_len.trailing_zeros() as usize;
      match groups.iter_mut().find(|(rep, _)| {
        let rp = &targets[*rep].1;
        bd_params::<SE::Scalar>(1usize << rp.len()).row_len == params.row_len
          && rp[rp.len() - lc..] == point[point.len() - lc..]
      }) {
        Some((_, members)) => members.push(i),
        None => groups.push((i, vec![i])),
      }
    }
    if arg.groups.len() != groups.len() || arg.direct.len() != n_direct {
      return Err(SpartanError::ProofVerifyError {
        reason: "brakedown backend: wrong number of opening arguments".to_string(),
      });
    }
    for ((rep, members), a) in groups.iter().zip(arg.groups.iter()) {
      let params = bd_params(1usize << targets[*rep].1.len());
      let items: Vec<(&[u8; 32], &[SE::Scalar], SE::Scalar)> = members
        .iter()
        .map(|&i| (targets[i].0, targets[i].1.as_slice(), targets[i].2))
        .collect();
      brakedown_verify_group(params, &items, a, sub)?;
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Test-private marker types standing in for field types: a local
  /// `BdRetainedCache` instance plus private `TypeId`s make the pattern
  /// fully deterministic — no other test in the process can touch them.
  struct FieldA;
  struct FieldB;

  fn root(i: u8) -> [u8; 32] {
    [i; 32]
  }

  /// Deterministic pattern: put → hit, absent → miss, reset zeroes both
  /// the map and the counters.
  #[test]
  fn retained_cache_hit_miss_and_reset_pattern() {
    let cache = BdRetainedCache::new();
    cache.put::<FieldA>(root(1), Box::new(7usize));
    assert_eq!(cache.get::<FieldA, usize>(&root(1)), Some(7));
    assert_eq!(cache.get::<FieldA, usize>(&root(2)), None);
    // A different field type never sees another type's entries.
    assert_eq!(cache.get::<FieldB, usize>(&root(1)), None);
    let a = cache.stats::<FieldA>();
    assert_eq!(
      (a.hits, a.misses, a.reencodes, a.wholesale_clears),
      (1, 1, 0, 0)
    );
    let b = cache.stats::<FieldB>();
    assert_eq!((b.hits, b.misses), (0, 1));

    cache.reset::<FieldA>();
    assert_eq!(cache.get::<FieldA, usize>(&root(1)), None);
    let a = cache.stats::<FieldA>();
    // The post-reset miss is the only event on the zeroed counters.
    assert_eq!(
      (a.hits, a.misses, a.reencodes, a.wholesale_clears),
      (0, 1, 0, 0)
    );
  }

  /// The 8-entry insertion bound clears the whole map (not LRU) and is
  /// counted against the inserting type.
  #[test]
  fn retained_cache_wholesale_clear_pattern() {
    let cache = BdRetainedCache::new();
    for i in 0..8u8 {
      cache.put::<FieldA>(root(i), Box::new(i as usize));
    }
    assert_eq!(cache.stats::<FieldA>().wholesale_clears, 0);
    // The 9th insertion finds len == 8 and clears everything first.
    cache.put::<FieldA>(root(8), Box::new(8usize));
    assert_eq!(cache.stats::<FieldA>().wholesale_clears, 1);
    assert_eq!(cache.get::<FieldA, usize>(&root(0)), None); // wiped
    assert_eq!(cache.get::<FieldA, usize>(&root(8)), Some(8)); // fresh
  }

  /// The recommit path's re-encode counter through the process-global
  /// cache: a reset (empty cache) makes the next `recommit_data` a
  /// deterministic miss + re-encode. Interference-immune: only this test
  /// commits with the F127 engine's scalar under an empty cache, and a
  /// miss cannot be turned back into a hit by concurrent evictions.
  #[test]
  fn retained_cache_reencode_is_audited() {
    type SE = crate::provider::F127Engine;
    type F = <SE as crate::traits::mod_engine::SumcheckEngine>::Scalar;
    let poly: Vec<F> = (0..64u64).map(F::from).collect();
    let (comm, _data) = BdBackend::<SE>::commit(&(), &poly, &(), true).unwrap();

    bd_retained_cache_reset_for::<F>();
    let before = bd_retained_cache_stats_for::<F>();
    assert_eq!(before, super::super::BdRetainedCacheStats::default());
    let _ = BdBackend::<SE>::recommit_data(&(), &comm, &poly, &(), true).unwrap();
    let after = bd_retained_cache_stats_for::<F>();
    assert_eq!(after.misses, 1);
    assert_eq!(after.reencodes, 1);
    assert_eq!(after.hits, 0);
  }
}
