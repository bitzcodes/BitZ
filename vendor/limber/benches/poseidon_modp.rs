//! benches/poseidon_modp.rs
//! Thirty Poseidon2 compressions proven with `imod_r1cs_modp` in ONE
//! mixed-modulus circuit — three independent ten-compression chains, one
//! per non-native prime field (BN254-Fr, BLS12-381-Fr, secp256k1-Fr, in
//! the fixed `FIELD_ORDER`) — under Hyrax and Brakedown Mod-PCS backends.
//! Benchmark groups have no field dimension: every group contains exactly
//! one combined case per backend.
//!
//! Flags (strict grammar, parsed by the shared crate-side
//! `limber::poseidon_bench::RunConfig` — see plan §12):
//!   BDPCS=1                  Brakedown instead of Hyrax
//!   IMOD_K / BDK             backend-specific k override in 7..=13
//!   KSWEEP=1                 only the §6 combined prove_e2e sweep
//!   PSIZE=1                  one combined four-line block, no Criterion
//!   HASHES=<n>               hashes PER FIELD; default 10, 1..=256
//!   POSEIDON_ALLOW_LARGE=1   lift the per-field cap (memory estimate)
//!   POSEIDON_ALLOW_KNOBS=1   permit the seven repo knobs (recorded)
//!   POSEIDON_ALLOW_DIRTY=1   permit a non-publishable exploratory run
//!
//! Published runs go through `scripts/run_poseidon_bench.sh`, which
//! passes the immutable run-config path/hash via the handshake variables;
//! a bare `cargo bench --bench poseidon_modp` runs unmanaged (no
//! manifest/sidecars, Criterion default output directory).
//!
//! No zero-knowledge claim is made for this driver (plan §12): Hyrax
//! commitments are hiding, Brakedown commitments are not, and the
//! sumcheck transcript carries unmasked witness-dependent data
//! regardless of backend.
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion};
use limber::{
  errors::SpartanError,
  imod_r1cs_modp::{IntModR1CSInstanceModp, IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::{
    IntModSpartanModpProverKey, IntModSpartanModpSNARK, IntModSpartanModpVerifierKey,
  },
  poseidon_bench::{
    BenchBackend, BenchMode, ENV_CONFIG_PATH, ENV_CONFIG_SHA256, ENV_RUN_DIR, K_ORDER,
    POSEIDON_LOG_T, POSEIDON_LOG_T_F, RunConfig, protocol_bytes_from_full_config,
  },
  poseidon2::{
    FIELD_ORDER, Layout, Poseidon2ParamsSet, PoseidonVerifierKey, build_all_params, build_inputs,
    build_shape, compute_advice, expected_chain, validate_advice, verify_poseidon_chain,
  },
  provider::{
    T256DynPrimeBdEngine, T256DynPrimeEngine,
    keccak::Keccak256Transcript,
    pcs::{
      bd_retained_cache_reset, bd_retained_cache_stats, f_chunk_len, integer_modpcs::IntEvalParams,
      prewarm_brakedown_params,
    },
  },
  traits::mod_engine::ModEngine,
};
use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, time::Duration};

/// The one thing with no generic form: the per-engine inherent
/// `setup_with_params`. Bench-local; no public library item is bounded
/// by it.
trait PoseidonBackend: ModEngine<TE = Keccak256Transcript<Self>> + Sized {
  /// Backend name for benchmark IDs.
  const NAME: &'static str;
  /// Whether this backend uses the Brakedown retained cache (and thus
  /// the deterministic empty-cache reset/audit policy).
  const USES_RETAINED_CACHE: bool;
  fn setup_with_params(
    shape: IntModR1CSShapeModp<Self>,
    params: IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<Self>,
      IntModSpartanModpVerifierKey<Self>,
    ),
    SpartanError,
  >;
}

impl PoseidonBackend for T256DynPrimeEngine {
  const NAME: &'static str = "hyrax";
  const USES_RETAINED_CACHE: bool = false;
  fn setup_with_params(
    shape: IntModR1CSShapeModp<Self>,
    params: IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<Self>,
      IntModSpartanModpVerifierKey<Self>,
    ),
    SpartanError,
  > {
    IntModSpartanModpSNARK::<Self>::setup_with_params(shape, params)
  }
}

impl PoseidonBackend for T256DynPrimeBdEngine {
  const NAME: &'static str = "brakedown";
  const USES_RETAINED_CACHE: bool = true;
  fn setup_with_params(
    shape: IntModR1CSShapeModp<Self>,
    params: IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<Self>,
      IntModSpartanModpVerifierKey<Self>,
    ),
    SpartanError,
  > {
    IntModSpartanModpSNARK::<Self>::setup_with_params(shape, params)
  }
}

/// Everything one `(backend, H, k)` combined-circuit configuration needs,
/// built once in the single combined preflight. Immutable fixtures live
/// here; anything a timed operation consumes is cloned in an untimed
/// setup closure.
struct Fixture<B: PoseidonBackend> {
  set: Poseidon2ParamsSet,
  shape: IntModR1CSShapeModp<B>,
  layout: Layout,
  ie_params: IntEvalParams,
  messages: Vec<BigUint>,
  w: Vec<BigUint>,
  q: Vec<BigUint>,
  digests: [BigUint; 3],
  pk: IntModSpartanModpProverKey<B>,
  pvk: PoseidonVerifierKey<B>,
  instance: IntModR1CSInstanceModp<B>,
  proof: IntModSpartanModpSNARK<B>,
}

/// The §11 four-line proof-size accounting for the ONE combined proof
/// (never summed into one figure; the three public digests are excluded
/// by convention).
struct ProofSizeLines {
  input_commitments: usize,
  eval_arg: usize,
  sumcheck_remainder: usize,
}

fn proof_size_lines<B: PoseidonBackend>(fx: &Fixture<B>) -> Result<ProofSizeLines, SpartanError> {
  let lc = fx.layout.num_cons().ilog2() as usize;
  let lv = fx.layout.num_vars().ilog2() as usize;
  Ok(ProofSizeLines {
    input_commitments: fx.instance.commitment_bytes()?.len(),
    eval_arg: fx.proof.eval_arg_bytes()?.len(),
    // Cubic outer rounds (3 coefficients), quadratic inner rounds (2),
    // plus 6 claimed evaluations, at 16 B per 2-limb scalar. Analytical
    // payload, no framing (`DynPrime` lacks serde).
    sumcheck_remainder: 16 * (3 * lc + 2 * (lv + 1) + 6),
  })
}

fn print_proof_size_block<B: PoseidonBackend>(fx: &Fixture<B>, lines: &ProofSizeLines) {
  println!(
    "{}/mixed3/Hpf{}-total{}: proof size\n  input commitments   : {} B\n  \
     eval_arg            : {} B\n  sumcheck remainder  : {} B (analytical payload, no framing)\n  \
     public digests (3)  : public statement — excluded by convention",
    B::NAME,
    fx.layout.hashes_per_field(),
    fx.layout.total_hashes(),
    lines.input_commitments,
    lines.eval_arg,
    lines.sumcheck_remainder,
  );
}

/// Build the combined fixture: the fixed-order parameter set, the single
/// mixed-modulus shape, combined advice, keys, and one complete untimed
/// proof + verification. The untimed proof serves the preflight reference
/// check (each field's `expected_chain` against its block digest, plus
/// one `validate_advice` over the combined advice), the Brakedown
/// internal-layout warm-up, and the proof-size lines; after this, no
/// reference comparison happens anywhere in the bench.
fn build_fixture<B: PoseidonBackend>(hashes_per_field: usize, k: usize) -> Fixture<B> {
  let set = build_all_params().expect("params build");
  let (shape, layout) = build_shape::<B>(&set, hashes_per_field).expect("shape build");
  let messages = build_inputs(hashes_per_field).expect("inputs build");
  let (w, q, digests) = compute_advice(&set, &layout, &messages).expect("advice");
  validate_advice(&set, &layout, &w, &q, &digests).expect("advice bounds");
  for (f, field) in FIELD_ORDER.iter().enumerate() {
    let chain = expected_chain(set.get(*field), &messages).expect("reference chain");
    assert_eq!(
      &digests[f],
      chain.last().expect("nonempty chain"),
      "preflight digest mismatch for block {}",
      field.name()
    );
  }

  let ie_params = IntEvalParams::derive(POSEIDON_LOG_T_F, POSEIDON_LOG_T, k, layout.log_n())
    .expect("IntEval params satisfy bounds");
  if B::USES_RETAINED_CACHE {
    // Pre-build the deterministic code layouts for the input chunk
    // length (w and q share it); the untimed proof below warms the
    // internal layer/table layouts. Published Brakedown results are
    // layout-warm steady state; no cold number is reported.
    let n = layout.num_vars().max(layout.num_cons());
    let len = f_chunk_len(&ie_params, n).expect("validated params");
    let _ = prewarm_brakedown_params(len);
  }
  let (pk, vk) = B::setup_with_params(shape.clone(), ie_params.clone()).expect("backend setup");
  let pvk = PoseidonVerifierKey::new(vk, &set, &layout).expect("verifier-key predicates");
  let (witness, instance) =
    IntModR1CSWitnessModp::<B>::new(&shape, pk.ck(), w.clone(), q.clone(), digests.to_vec())
      .expect("witness commit");
  let proof = IntModSpartanModpSNARK::<B>::prove(&pk, &instance, &witness).expect("prove");
  verify_poseidon_chain(&pvk, &instance, &proof).expect("preflight verification");

  Fixture {
    set,
    shape,
    layout,
    ie_params,
    messages,
    w,
    q,
    digests,
    pk,
    pvk,
    instance,
    proof,
  }
}

/// Managed-run state from the runner handshake, if any.
struct Managed {
  run_dir: PathBuf,
  config12: String,
}

/// Atomic JSON write: temp file in the target directory, then rename.
fn write_json_atomic(path: &PathBuf, value: &serde_json::Value) {
  let bytes = limber::poseidon_bench::canonical_json_bytes(value);
  let tmp = path.with_extension("json.tmp");
  std::fs::write(&tmp, &bytes).expect("sidecar write");
  std::fs::rename(&tmp, path).expect("sidecar rename");
}

/// Verify the runner handshake: recompute the config file's SHA-256,
/// compare with the passed full hash, and byte-compare the canonical
/// protocol subsection against this process's own parse of the
/// environment. Returns `None` (unmanaged) when no run directory is set.
fn handshake(config: &RunConfig) -> Option<Managed> {
  let run_dir = std::env::var_os(ENV_RUN_DIR)?;
  let config_path = std::env::var(ENV_CONFIG_PATH).expect("run dir set but no config path");
  let passed_hash = std::env::var(ENV_CONFIG_SHA256).expect("run dir set but no config hash");
  let file = std::fs::read(&config_path).expect("run-config readable");
  let mut h = Sha256::new();
  h.update(&file);
  let actual: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
  assert_eq!(
    actual, passed_hash,
    "run-config file hash does not match the hash the runner passed"
  );
  let file_protocol =
    protocol_bytes_from_full_config(std::str::from_utf8(&file).expect("run-config is UTF-8"))
      .expect("run-config protocol subsection");
  assert_eq!(
    file_protocol,
    config.protocol_canonical_bytes(),
    "run-config protocol subsection does not match this process's environment parse"
  );
  Some(Managed {
    run_dir: PathBuf::from(run_dir),
    config12: passed_hash[..12].to_string(),
  })
}

/// One untimed audited iteration from the empty-cache state, recorded
/// under the full group ID into `cache-audit.json` (atomically updated
/// after each group's audit).
fn audit_group<B: PoseidonBackend>(
  managed: &Option<Managed>,
  audit: &mut serde_json::Map<String, serde_json::Value>,
  group_id: &str,
  op: impl FnOnce(),
) {
  if !B::USES_RETAINED_CACHE {
    return;
  }
  bd_retained_cache_reset();
  op();
  let stats = bd_retained_cache_stats();
  audit.insert(
    group_id.to_string(),
    serde_json::to_value(stats).expect("stats serialize"),
  );
  if let Some(m) = managed {
    write_json_atomic(
      &m.run_dir.join("cache-audit.json"),
      &serde_json::Value::Object(audit.clone()),
    );
  }
}

/// Reset the retained cache in an untimed setup closure (Brakedown only;
/// a no-op for Hyrax).
fn maybe_reset<B: PoseidonBackend>() {
  if B::USES_RETAINED_CACHE {
    bd_retained_cache_reset();
  }
}

/// Benchmark ID:
/// `{backend}/mixed3/Hpf{H}-total{3H}/c2^{lc}v2^{lv}/k{k}/cfg-{12}`;
/// `advice/` passes `backend = None` (backend-independent).
fn bench_id(backend: Option<&str>, layout: &Layout, k: usize, config12: &str) -> String {
  let lc = layout.num_cons().ilog2();
  let lv = layout.num_vars().ilog2();
  let tail = format!(
    "mixed3/Hpf{}-total{}/c2^{lc}v2^{lv}/k{k}/cfg-{config12}",
    layout.hashes_per_field(),
    layout.total_hashes()
  );
  match backend {
    Some(b) => format!("{b}/{tail}"),
    None => tail,
  }
}

/// The normal-mode groups for one backend, in the pinned literal order
/// `[setup, advice (Hyrax only), commit_witness, prove_after_input_commit,
/// prove_e2e, verify]`; each group contains exactly one combined case.
fn run_normal<B: PoseidonBackend>(
  c: &mut Criterion,
  config: &RunConfig,
  managed: &Option<Managed>,
  config12: &str,
) {
  let fx = build_fixture::<B>(config.hashes, config.k);
  let mut audit = serde_json::Map::new();

  // setup/: raw setup_with_params + checked PoseidonVerifierKey::new.
  {
    let mut g = c.benchmark_group("setup");
    let id = bench_id(Some(B::NAME), &fx.layout, config.k, config12);
    g.bench_function(&id, |b| {
      b.iter_batched(
        || (fx.shape.clone(), fx.ie_params.clone()),
        |(shape, ie)| {
          let (_pk, vk) = B::setup_with_params(shape, ie).expect("setup");
          let _pvk = PoseidonVerifierKey::new(vk, &fx.set, &fx.layout).expect("vk predicates");
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // advice/: all three blocks' compute_advice only — backend-independent,
  // so only the default Hyrax normal run registers it (its ID omits the
  // backend segment).
  if !B::USES_RETAINED_CACHE {
    let mut g = c.benchmark_group("advice");
    let id = bench_id(None, &fx.layout, config.k, config12);
    g.bench_function(&id, |b| {
      b.iter(|| {
        let _ = compute_advice(&fx.set, &fx.layout, &fx.messages).expect("advice");
      });
    });
    g.finish();
  }

  // commit_witness/: IntModR1CSWitnessModp::new for the combined W/Q
  // vectors, from a reset cache.
  {
    let mut g = c.benchmark_group("commit_witness");
    let id = bench_id(Some(B::NAME), &fx.layout, config.k, config12);
    audit_group::<B>(managed, &mut audit, &format!("commit_witness/{id}"), || {
      let _ = IntModR1CSWitnessModp::<B>::new(
        &fx.shape,
        fx.pk.ck(),
        fx.w.clone(),
        fx.q.clone(),
        fx.digests.to_vec(),
      )
      .expect("commit");
    });
    g.bench_function(&id, |b| {
      b.iter_batched(
        || {
          maybe_reset::<B>();
          (fx.w.clone(), fx.q.clone(), fx.digests.to_vec())
        },
        |(w, q, x)| {
          let _ = IntModR1CSWitnessModp::<B>::new(&fx.shape, fx.pk.ck(), w, q, x).expect("commit");
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // prove_after_input_commit/: untimed reset + W/Q commit, timed prove —
  // including any deterministic W/Q re-encoding caused by internal
  // commitment eviction. Never described as "commit-free".
  {
    let mut g = c.benchmark_group("prove_after_input_commit");
    let id = bench_id(Some(B::NAME), &fx.layout, config.k, config12);
    audit_group::<B>(
      managed,
      &mut audit,
      &format!("prove_after_input_commit/{id}"),
      || {
        let (witness, instance) = IntModR1CSWitnessModp::<B>::new(
          &fx.shape,
          fx.pk.ck(),
          fx.w.clone(),
          fx.q.clone(),
          fx.digests.to_vec(),
        )
        .expect("commit");
        let _ = IntModSpartanModpSNARK::<B>::prove(&fx.pk, &instance, &witness).expect("prove");
      },
    );
    g.bench_function(&id, |b| {
      b.iter_batched(
        || {
          maybe_reset::<B>();
          IntModR1CSWitnessModp::<B>::new(
            &fx.shape,
            fx.pk.ck(),
            fx.w.clone(),
            fx.q.clone(),
            fx.digests.to_vec(),
          )
          .expect("commit")
        },
        |(witness, instance)| {
          let _ = IntModSpartanModpSNARK::<B>::prove(&fx.pk, &instance, &witness).expect("prove");
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // prove_e2e/: combined advice + commit + prove — quote this as
  // "prover time".
  {
    let mut g = c.benchmark_group("prove_e2e");
    let id = bench_id(Some(B::NAME), &fx.layout, config.k, config12);
    audit_group::<B>(managed, &mut audit, &format!("prove_e2e/{id}"), || {
      prove_e2e_once(&fx);
    });
    g.bench_function(&id, |b| {
      b.iter_batched(
        maybe_reset::<B>,
        |()| prove_e2e_once(&fx),
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // verify/: one verify_poseidon_chain, including all three canonicality
  // checks. No retained-cache access, nothing consumed.
  {
    let mut g = c.benchmark_group("verify");
    let id = bench_id(Some(B::NAME), &fx.layout, config.k, config12);
    g.bench_function(&id, |b| {
      b.iter(|| verify_poseidon_chain(&fx.pvk, &fx.instance, &fx.proof).expect("verify"));
    });
    g.finish();
  }

  // For Hyrax the audit map is legitimately empty; write it anyway so
  // the runner can require the sidecar's presence uniformly.
  if let Some(m) = managed {
    write_json_atomic(
      &m.run_dir.join("cache-audit.json"),
      &serde_json::Value::Object(audit),
    );
  }
}

/// The timed `prove_e2e` region: combined advice + W/Q commit + prove.
fn prove_e2e_once<B: PoseidonBackend>(fx: &Fixture<B>) {
  let (w, q, digests) = compute_advice(&fx.set, &fx.layout, &fx.messages).expect("advice");
  let (witness, instance) =
    IntModR1CSWitnessModp::<B>::new(&fx.shape, fx.pk.ck(), w, q, digests.to_vec()).expect("commit");
  let _ = IntModSpartanModpSNARK::<B>::prove(&fx.pk, &instance, &witness).expect("prove");
}

/// KSWEEP mode: Criterion-only combined `prove_e2e` sweep at fixed
/// `H = 10` per field, one combined case per `k` in the pinned `K_ORDER`.
/// Every preflight completes (and `ksweep-metadata.json` is written)
/// before any group registers; a `k` registers only if its single
/// combined preflight passed, but failed candidates stay in the metadata.
fn run_ksweep<B: PoseidonBackend>(
  c: &mut Criterion,
  config: &RunConfig,
  managed: &Option<Managed>,
  config12: &str,
) {
  let mut metadata = serde_json::Map::new();
  let mut admissible: Vec<(usize, Fixture<B>)> = Vec::new();

  for &k in K_ORDER.iter() {
    let entry = match IntEvalParams::derive(POSEIDON_LOG_T_F, POSEIDON_LOG_T, k, config.log_n) {
      Err(e) => {
        // Failed derive: (log_p, s) and all four proof-size lines null;
        // the exact error string is always present.
        serde_json::json!({
          "admissible": false, "error": e.to_string(),
          "log_p": null, "s": null,
          "proof_size": { "input_commitments": null, "eval_arg": null,
                          "sumcheck_remainder": null, "public_digests": "excluded" },
        })
      }
      Ok(ie) => {
        // One untimed combined proof + verify_poseidon_chain gate: a k is
        // inadmissible if setup, proving, or verification of any of its
        // three field blocks/digests fails.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
          build_fixture::<B>(config.hashes, k)
        })) {
          Err(_) => serde_json::json!({
            "admissible": false,
            "error": "untimed combined proof or verification failed",
            "log_p": ie.log_p, "s": ie.s,
            "proof_size": { "input_commitments": null, "eval_arg": null,
                            "sumcheck_remainder": null, "public_digests": "excluded" },
          }),
          Ok(fx) => {
            let lines = proof_size_lines(&fx).expect("proof-size serialization");
            let v = serde_json::json!({
              "admissible": true, "error": null,
              "log_p": ie.log_p, "s": ie.s,
              "proof_size": {
                "input_commitments": lines.input_commitments,
                "eval_arg": lines.eval_arg,
                "sumcheck_remainder": lines.sumcheck_remainder,
                "public_digests": "excluded",
              },
            });
            admissible.push((k, fx));
            v
          }
        }
      }
    };
    let ok = entry["admissible"].as_bool() == Some(true);
    metadata.insert(format!("k{k}"), entry);
    if !ok {
      println!("ksweep: k = {k} is inadmissible (see metadata)");
    }
  }

  let meta_doc = serde_json::json!({
    "backend": B::NAME,
    "circuit": "mixed3",
    "hashes_per_field": config.hashes,
    "total_hashes": config.total_hashes,
    "k_order": K_ORDER.to_vec(),
    "candidates": serde_json::Value::Object(metadata),
  });
  if let Some(m) = managed {
    write_json_atomic(&m.run_dir.join("ksweep-metadata.json"), &meta_doc);
  } else {
    println!(
      "ksweep metadata (unmanaged run):\n{}",
      serde_json::to_string_pretty(&meta_doc).expect("metadata serializes")
    );
  }

  // Register the sweep group: one combined case per admissible k, in the
  // pinned K_ORDER, with the same empty-cache policy as normal prove_e2e
  // samples.
  let mut audit = serde_json::Map::new();
  let mut g = c.benchmark_group("prove_e2e");
  for (k, fx) in &admissible {
    let id = bench_id(Some(B::NAME), &fx.layout, *k, config12);
    audit_group::<B>(managed, &mut audit, &format!("prove_e2e/{id}"), || {
      prove_e2e_once(fx);
    });
    g.bench_function(&id, |b| {
      b.iter_batched(
        maybe_reset::<B>,
        |()| prove_e2e_once(fx),
        BatchSize::PerIteration,
      );
    });
  }
  g.finish();
  if let Some(m) = managed {
    write_json_atomic(
      &m.run_dir.join("cache-audit.json"),
      &serde_json::Value::Object(audit),
    );
  }
}

/// PSIZE mode: ONE combined-circuit four-line block (it must not report
/// or sum three fictitious per-field proofs), plus the `proof-size.json`
/// sidecar; exits without Criterion groups.
fn run_psize<B: PoseidonBackend>(config: &RunConfig, managed: &Option<Managed>) {
  let fx = build_fixture::<B>(config.hashes, config.k);
  let lines = proof_size_lines(&fx).expect("proof-size serialization");
  print_proof_size_block(&fx, &lines);
  let doc = serde_json::json!({
    "backend": B::NAME,
    "circuit": "mixed3",
    "hashes_per_field": config.hashes,
    "total_hashes": config.total_hashes,
    "k": config.k,
    "combined": {
      "input_commitments": lines.input_commitments,
      "eval_arg": lines.eval_arg,
      "sumcheck_remainder": lines.sumcheck_remainder,
      "public_digests": "excluded",
    },
  });
  if let Some(m) = managed {
    write_json_atomic(&m.run_dir.join("proof-size.json"), &doc);
  }
}

fn dispatch<B: PoseidonBackend>(config: &RunConfig, managed: &Option<Managed>) {
  // Combined memory-estimate print before anything else on a lifted-cap
  // run.
  if config.allow_large && config.hashes > 256 {
    let n = config.num_vars.max(config.num_cons);
    let ie = IntEvalParams::derive(POSEIDON_LOG_T_F, POSEIDON_LOG_T, config.k, config.log_n)
      .expect("IntEval params");
    let len = f_chunk_len(&ie, n).expect("validated params");
    println!(
      "POSEIDON_ALLOW_LARGE: H = {} per field ({} total) -> f_chunk_len = {len} \
       (~{} MiB as u64, ~{} MiB as 32-byte scalars) per polynomial",
      config.hashes,
      config.total_hashes,
      (len * 8) >> 20,
      (len * 32) >> 20,
    );
  }

  let config12 = match managed {
    Some(m) => m.config12.clone(),
    None => {
      // Unmanaged runs still isolate Criterion baselines by hashing the
      // canonical protocol bytes.
      let mut h = Sha256::new();
      h.update(config.protocol_canonical_bytes());
      h.finalize()[..6]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
    }
  };

  match config.mode {
    BenchMode::ProofSize => {
      run_psize::<B>(config, managed);
    }
    BenchMode::KSweep | BenchMode::Normal => {
      let mut c = Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(20));
      if let Some(m) = managed {
        c = c.output_directory(&m.run_dir.join("criterion"));
      }
      if config.mode == BenchMode::KSweep {
        run_ksweep::<B>(&mut c, config, managed, &config12);
      } else {
        run_normal::<B>(&mut c, config, managed, &config12);
      }
      c.final_summary();
    }
  }
}

fn main() {
  let env_map: BTreeMap<std::ffi::OsString, std::ffi::OsString> = std::env::vars_os().collect();
  let config = match RunConfig::parse(&env_map) {
    Ok(c) => c,
    Err(e) => {
      eprintln!("poseidon_modp bench: {e}");
      std::process::exit(1);
    }
  };
  let managed = handshake(&config);
  if managed.is_none() {
    eprintln!(
      "poseidon_modp bench: unmanaged run (no {ENV_RUN_DIR}); published results must go \
       through scripts/run_poseidon_bench.sh"
    );
  }
  match config.backend {
    BenchBackend::Hyrax => dispatch::<T256DynPrimeEngine>(&config, &managed),
    BenchBackend::Brakedown => dispatch::<T256DynPrimeBdEngine>(&config, &managed),
  }
}
