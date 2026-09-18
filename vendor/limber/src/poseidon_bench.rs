//! Shared benchmark-run configuration for the Poseidon2 bench: the pure
//! environment-flag parser, the pinned registration orders, and the
//! canonical protocol-JSON serialization.
//!
//! Both `benches/poseidon_modp.rs` and the `poseidon_bench_config` helper
//! binary call [`RunConfig::parse`], so the shell runner never
//! independently interprets benchmark flags or backend defaults. The
//! parser is pure: it reads only the map it is given, never the process
//! environment, so its complete conflict table is unit-testable.

use crate::{errors::SpartanError, provider::pcs::integer_modpcs::IntEvalParams};
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};

/// `log_2(T_f)` for the Poseidon2 workload: committed values are
/// canonical residues below the ~256-bit target moduli.
pub const POSEIDON_LOG_T_F: usize = 256;
/// Limb bound (bits) for the IntEval range checks.
pub const POSEIDON_LOG_T: usize = 64;
/// Persisted default IntEval `k` for the Hyrax backend on the combined
/// mixed-modulus circuit (log_n = 14). Selected by the 2026-09-02
/// combined-circuit tuning `KSWEEP` (Apple M2, `RAYON_NUM_THREADS=1`,
/// `target-cpu=native`): combined median 785 ms at `k = 9`, tie band
/// {9, 11}, winner = smallest in band. Work-item-15 agreement from a
/// clean revision is still required before publication.
pub const DEFAULT_HYRAX_K: usize = 9;
/// Persisted default IntEval `k` for the Brakedown backend on the
/// combined mixed-modulus circuit. Same sweep: combined median 644 ms at
/// the tie band {9, 10, 11}, winner = smallest in band.
pub const DEFAULT_BD_K: usize = 9;
/// Default chain length PER FIELD (the combined circuit contains `3H`).
pub const DEFAULT_HASHES: usize = 10;
/// Per-field chain-length cap without `POSEIDON_ALLOW_LARGE=1`.
pub const MAX_HASHES: usize = 256;

/// Pinned `KSWEEP` candidate registration order: one combined case per
/// `k`; `FIELD_ORDER` below is the internal block/IO order, not a
/// registration dimension.
pub const K_ORDER: [usize; 7] = [10, 7, 12, 9, 13, 8, 11];
/// The circuit's semantic field-block/public-IO order (layout metadata;
/// benchmark groups have no field dimension). Must match
/// `poseidon2::FIELD_ORDER` — a unit test binds the two.
pub const FIELD_ORDER: [&str; 3] = ["bn254", "bls12_381", "secp256k1"];
/// Literal normal-mode group order (`advice` is registered only by the
/// default Hyrax normal run).
pub const GROUP_ORDER: [&str; 6] = [
  "setup",
  "advice",
  "commit_witness",
  "prove_after_input_commit",
  "prove_e2e",
  "verify",
];

/// Runner-handshake variable: absolute run directory (Criterion writes to
/// `<run-dir>/criterion/`; sidecars land beside it).
pub const ENV_RUN_DIR: &str = "POSEIDON_RUN_DIR";
/// Runner-handshake variable: absolute path of the immutable
/// `run-config.json`.
pub const ENV_CONFIG_PATH: &str = "POSEIDON_CONFIG_PATH";
/// Runner-handshake variable: full SHA-256 (lowercase hex) of the
/// config's exact bytes.
pub const ENV_CONFIG_SHA256: &str = "POSEIDON_CONFIG_SHA256";

/// The seven repository knobs that silently change Brakedown layout or
/// prover work. A canonical run requires all seven unset;
/// `POSEIDON_ALLOW_KNOBS=1` permits a clearly labelled nonstandard
/// configuration whose raw values and effective interpretations enter
/// the config hash.
pub const HIDDEN_KNOBS: [&str; 7] = [
  "BDDIRECT",
  "BDSPEC",
  "BDROWLEN",
  "BDSPLIT",
  "CHAIN_BITS",
  "GKRSKIP",
  "RUST_LOG",
];

/// Benchmark mode, resolved from `KSWEEP`/`PSIZE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchMode {
  /// The ordinary Criterion groups.
  Normal,
  /// Only the §6 `prove_e2e` sweep groups at fixed `H = 10`, then exit.
  KSweep,
  /// The four-line proof-size blocks for all fields, then exit.
  ProofSize,
}

impl BenchMode {
  /// Stable lowercase name for JSON/IDs.
  pub fn name(&self) -> &'static str {
    match self {
      BenchMode::Normal => "normal",
      BenchMode::KSweep => "ksweep",
      BenchMode::ProofSize => "proof_size",
    }
  }
}

/// Commitment backend, resolved from `BDPCS`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchBackend {
  /// Hyrax (curve) Mod-PCS — the default.
  Hyrax,
  /// Brakedown (hash) Mod-PCS.
  Brakedown,
}

impl BenchBackend {
  /// Stable lowercase name for JSON/IDs.
  pub fn name(&self) -> &'static str {
    match self {
      BenchBackend::Hyrax => "hyrax",
      BenchBackend::Brakedown => "brakedown",
    }
  }
}

/// Raw and effective state of one permitted hidden knob.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnobState {
  /// The raw environment value (present knobs only).
  pub raw: String,
  /// The effective interpretation the backend code will apply.
  pub effective: String,
}

/// A fully resolved benchmark-run configuration.
#[derive(Clone, Debug)]
pub struct RunConfig {
  /// Resolved mode.
  pub mode: BenchMode,
  /// Resolved backend.
  pub backend: BenchBackend,
  /// Resolved chain length `H` PER FIELD.
  pub hashes: usize,
  /// Total compressions in the combined circuit: `3H`.
  pub total_hashes: usize,
  /// Resolved IntEval `k` (default-or-override; ignored in `KSweep`).
  pub k: usize,
  /// Whether `k` came from an explicit override (`IMOD_K`/`BDK`).
  pub k_overridden: bool,
  /// `POSEIDON_ALLOW_LARGE=1`.
  pub allow_large: bool,
  /// `POSEIDON_ALLOW_KNOBS=1`.
  pub allow_knobs: bool,
  /// `POSEIDON_ALLOW_DIRTY=1`.
  pub allow_dirty: bool,
  /// Present hidden knobs with raw and effective values (empty for a
  /// canonical run).
  pub knobs: BTreeMap<String, KnobState>,
  /// Real (unpadded) combined rows for the resolved `H`: `1299H`.
  pub real_rows: usize,
  /// Real (unpadded) combined columns for the resolved `H`: `1302H − 3`.
  pub real_cols: usize,
  /// Padded constraint rows.
  pub num_cons: usize,
  /// Padded witness columns.
  pub num_vars: usize,
  /// `log₂(max(num_cons, num_vars))`.
  pub log_n: usize,
}

fn cfg_err(reason: impl Into<String>) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: format!("poseidon bench config: {}", reason.into()),
  }
}

/// Fetch a recognized, result-affecting environment value. Present
/// non-Unicode values are rejected rather than platform-dependently
/// encoded.
fn get_unicode(
  env: &BTreeMap<OsString, OsString>,
  key: &str,
) -> Result<Option<String>, SpartanError> {
  match env.get(OsStr::new(key)) {
    None => Ok(None),
    Some(v) => v
      .to_str()
      .map(|s| Some(s.to_string()))
      .ok_or_else(|| cfg_err(format!("{key} is not valid Unicode"))),
  }
}

/// Strict boolean flag: exactly `"0"` or `"1"`; anything else errors.
fn parse_bool(env: &BTreeMap<OsString, OsString>, key: &str) -> Result<bool, SpartanError> {
  match get_unicode(env, key)?.as_deref() {
    None => Ok(false),
    Some("0") => Ok(false),
    Some("1") => Ok(true),
    Some(other) => Err(cfg_err(format!(
      "{key} must be exactly 0 or 1, got {other:?}"
    ))),
  }
}

/// Numeric flag: `usize`, present-or-absent.
fn parse_usize(
  env: &BTreeMap<OsString, OsString>,
  key: &str,
) -> Result<Option<usize>, SpartanError> {
  match get_unicode(env, key)? {
    None => Ok(None),
    Some(s) => s
      .parse::<usize>()
      .map(Some)
      .map_err(|_| cfg_err(format!("{key} must be a usize, got {s:?}"))),
  }
}

fn is_present(env: &BTreeMap<OsString, OsString>, key: &str) -> bool {
  env.contains_key(OsStr::new(key))
}

impl RunConfig {
  /// Parse a benchmark-run configuration from an environment map. Pure:
  /// consults nothing but `env`. Implements the complete flag grammar and
  /// conflict table of plan §12; conflicts are errors, never silently
  /// ignored.
  pub fn parse(env: &BTreeMap<OsString, OsString>) -> Result<Self, SpartanError> {
    // Boolean flags first (strict 0/1).
    let bdpcs = parse_bool(env, "BDPCS")?;
    let ksweep = parse_bool(env, "KSWEEP")?;
    let psize = parse_bool(env, "PSIZE")?;
    let allow_large = parse_bool(env, "POSEIDON_ALLOW_LARGE")?;
    let allow_knobs = parse_bool(env, "POSEIDON_ALLOW_KNOBS")?;
    let allow_dirty = parse_bool(env, "POSEIDON_ALLOW_DIRTY")?;

    let backend = if bdpcs {
      BenchBackend::Brakedown
    } else {
      BenchBackend::Hyrax
    };

    // Mode resolution and mode-level conflicts. In KSweep the PRESENCE of
    // these variables is an error, even when a value equals a default.
    let mode = if ksweep {
      for key in ["PSIZE", "HASHES", "IMOD_K", "BDK", "POSEIDON_ALLOW_LARGE"] {
        if is_present(env, key) {
          return Err(cfg_err(format!("KSWEEP=1 rejects the presence of {key}")));
        }
      }
      BenchMode::KSweep
    } else if psize {
      BenchMode::ProofSize
    } else {
      BenchMode::Normal
    };

    // Backend-specific k overrides.
    if is_present(env, "IMOD_K") && is_present(env, "BDK") {
      return Err(cfg_err("setting both IMOD_K and BDK is always an error"));
    }
    match backend {
      BenchBackend::Hyrax => {
        if is_present(env, "BDK") {
          return Err(cfg_err("the Hyrax backend rejects BDK (use IMOD_K)"));
        }
      }
      BenchBackend::Brakedown => {
        if is_present(env, "IMOD_K") {
          return Err(cfg_err("the Brakedown backend rejects IMOD_K (use BDK)"));
        }
      }
    }

    // Chain length PER FIELD: default 10, 1..=256, POSEIDON_ALLOW_LARGE
    // lifts the cap (still <= u32::MAX via the shared per-field
    // chain-count validator; combined-dimension arithmetic additionally
    // checks 3H below).
    let hashes = parse_usize(env, "HASHES")?.unwrap_or(DEFAULT_HASHES);
    if hashes == 0 {
      return Err(cfg_err("HASHES must be at least 1"));
    }
    if hashes > MAX_HASHES && !allow_large {
      return Err(cfg_err(format!(
        "HASHES = {hashes} exceeds {MAX_HASHES} per field; set POSEIDON_ALLOW_LARGE=1 to lift the cap"
      )));
    }
    if hashes > u32::MAX as usize {
      return Err(cfg_err("HASHES exceeds u32::MAX"));
    }

    // Combined-circuit dimensions for the resolved per-field H (checked
    // arithmetic, including every multiplication by three).
    let total_hashes = hashes
      .checked_mul(3)
      .ok_or_else(|| cfg_err("3H overflows usize"))?;
    let real_rows = 433usize
      .checked_mul(hashes)
      .and_then(|r| r.checked_mul(3))
      .ok_or_else(|| cfg_err("row arithmetic overflow"))?;
    let real_cols = 434usize
      .checked_mul(hashes)
      .and_then(|c| c.checked_sub(1))
      .and_then(|c| c.checked_mul(3))
      .ok_or_else(|| cfg_err("column arithmetic overflow"))?;
    let num_cons = real_rows
      .checked_next_power_of_two()
      .ok_or_else(|| cfg_err("padded row overflow"))?;
    let num_vars = real_cols
      .checked_next_power_of_two()
      .ok_or_else(|| cfg_err("padded column overflow"))?;
    let log_n = num_cons.max(num_vars).ilog2() as usize;

    // k resolution: override in 7..=13 whose derived params validate, or
    // the persisted per-backend default.
    let override_key = match backend {
      BenchBackend::Hyrax => "IMOD_K",
      BenchBackend::Brakedown => "BDK",
    };
    let k_override = parse_usize(env, override_key)?;
    if let Some(k) = k_override {
      if !(7..=13).contains(&k) {
        return Err(cfg_err(format!("{override_key} = {k} is outside 7..=13")));
      }
      IntEvalParams::derive(POSEIDON_LOG_T_F, POSEIDON_LOG_T, k, log_n).map_err(|e| {
        cfg_err(format!(
          "{override_key} = {k} fails IntEval derivation: {e}"
        ))
      })?;
    }
    let k = k_override.unwrap_or(match backend {
      BenchBackend::Hyrax => DEFAULT_HYRAX_K,
      BenchBackend::Brakedown => DEFAULT_BD_K,
    });

    // Hidden knobs: presence is an error without POSEIDON_ALLOW_KNOBS=1
    // (which waives only this prohibition, never syntax, range, mode, or
    // backend-specific conflicts). With the override, validate before any
    // backend code inherits a silent default or an out-of-range panic.
    let mut knobs = BTreeMap::new();
    for &knob in HIDDEN_KNOBS.iter() {
      if !is_present(env, knob) {
        continue;
      }
      if !allow_knobs {
        return Err(cfg_err(format!(
          "{knob} is set; a canonical run requires all seven repo knobs unset \
           (POSEIDON_ALLOW_KNOBS=1 permits a labelled nonstandard run)"
        )));
      }
      let raw = get_unicode(env, knob)?.expect("presence checked");
      let effective = match knob {
        "BDDIRECT" => {
          let v = raw
            .parse::<usize>()
            .map_err(|_| cfg_err(format!("BDDIRECT must be a usize, got {raw:?}")))?;
          format!("direct-ship threshold {v}")
        }
        "BDSPEC" => {
          let v = raw
            .parse::<usize>()
            .ok()
            .filter(|v| *v <= 5)
            .ok_or_else(|| cfg_err(format!("BDSPEC must be an integer in 0..=5, got {raw:?}")))?;
          format!("code spec {v}")
        }
        "BDROWLEN" => {
          let v = raw
            .parse::<usize>()
            .ok()
            .filter(|v| *v > 0)
            .ok_or_else(|| cfg_err(format!("BDROWLEN must be a positive usize, got {raw:?}")))?;
          // Per-input effective row length is `.min(n)`; report it for
          // the run's w/q input chunk length, not just the request.
          let f_chunk = num_vars
            .max(num_cons)
            .checked_mul(4 * 4)
            .ok_or_else(|| cfg_err("f_chunk arithmetic overflow"))?;
          format!(
            "requested {v}, effective {} for input length {f_chunk}",
            v.min(f_chunk)
          )
        }
        // Presence-sensitive booleans: the string "0" is still present.
        "BDSPLIT" => "present (enabled regardless of value)".to_string(),
        "CHAIN_BITS" => "present (enabled regardless of value)".to_string(),
        // Inverted sense: any value other than "0" enables the skip.
        "GKRSKIP" => {
          if raw == "0" {
            "skip DISABLED (inverted sense: 0 disables)".to_string()
          } else {
            "skip enabled".to_string()
          }
        }
        "RUST_LOG" => {
          tracing_subscriber::EnvFilter::builder()
            .parse(&raw)
            .map_err(|e| cfg_err(format!("RUST_LOG rejected by EnvFilter: {e}")))?;
          format!("EnvFilter {raw:?}")
        }
        _ => unreachable!("knob list is fixed"),
      };
      knobs.insert(knob.to_string(), KnobState { raw, effective });
    }

    Ok(Self {
      mode,
      backend,
      hashes,
      total_hashes,
      k,
      k_overridden: k_override.is_some(),
      allow_large,
      allow_knobs,
      allow_dirty,
      knobs,
      real_rows,
      real_cols,
      num_cons,
      num_vars,
      log_n,
    })
  }

  /// The canonical protocol subsection: every immutable, result-affecting
  /// resolved input. `serde_json`'s default `Map` is BTree-backed, so
  /// serialization is key-sorted and deterministic.
  pub fn protocol_json(&self) -> serde_json::Value {
    let mut knobs = serde_json::Map::new();
    for (name, state) in &self.knobs {
      knobs.insert(
        name.clone(),
        serde_json::json!({ "raw": state.raw, "effective": state.effective }),
      );
    }
    serde_json::json!({
      "workload": "limber-poseidon2-v1",
      "circuit": "mixed3",
      "mode": self.mode.name(),
      "backend": self.backend.name(),
      "hashes_per_field": self.hashes,
      "total_hashes": self.total_hashes,
      "num_io": 3,
      "field_blocks": FIELD_ORDER.to_vec(),
      "k": self.k,
      "k_overridden": self.k_overridden,
      "k_defaults": { "hyrax": DEFAULT_HYRAX_K, "brakedown": DEFAULT_BD_K },
      "log_t_f": POSEIDON_LOG_T_F,
      "log_t": POSEIDON_LOG_T,
      "dims": {
        "real_rows": self.real_rows,
        "real_cols": self.real_cols,
        "num_cons": self.num_cons,
        "num_vars": self.num_vars,
        "log_n": self.log_n,
      },
      "k_order": K_ORDER.to_vec(),
      "field_order": FIELD_ORDER.to_vec(),
      "group_order": GROUP_ORDER.to_vec(),
      "allow_large": self.allow_large,
      "allow_knobs": self.allow_knobs,
      "allow_dirty": self.allow_dirty,
      "knobs": serde_json::Value::Object(knobs),
      "criterion": {
        "sample_size": 10,
        "warm_up_time_s": 1,
        "measurement_time_s": 20,
      },
      "brakedown_cache_policy":
        "layout-warm steady state; empty retained cache before every measured commit/prove sample",
    })
  }

  /// Canonical bytes of the protocol subsection (pretty, sorted keys,
  /// trailing newline).
  pub fn protocol_canonical_bytes(&self) -> Vec<u8> {
    canonical_json_bytes(&self.protocol_json())
  }
}

/// Canonical JSON bytes: two-space pretty-printing over BTree-backed maps
/// (sorted keys) plus one trailing newline.
pub fn canonical_json_bytes(value: &serde_json::Value) -> Vec<u8> {
  let mut bytes =
    serde_json::to_vec_pretty(value).expect("serde_json::Value serialization cannot fail");
  bytes.push(b'\n');
  bytes
}

/// Extract and canonicalize the `protocol` subsection of a full
/// run-config JSON document (the benchmark byte-compares this against its
/// own re-parse of the environment).
pub fn protocol_bytes_from_full_config(full: &str) -> Result<Vec<u8>, SpartanError> {
  let doc: serde_json::Value = serde_json::from_str(full)
    .map_err(|e| cfg_err(format!("run-config is not valid JSON: {e}")))?;
  let protocol = doc
    .get("protocol")
    .ok_or_else(|| cfg_err("run-config has no protocol subsection"))?;
  Ok(canonical_json_bytes(protocol))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn env(pairs: &[(&str, &str)]) -> BTreeMap<OsString, OsString> {
    pairs
      .iter()
      .map(|(k, v)| (OsString::from(k), OsString::from(v)))
      .collect()
  }

  #[test]
  fn k_order_is_a_permutation_of_7_to_13() {
    let mut sorted = K_ORDER.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (7..=13).collect::<Vec<_>>());
  }

  #[test]
  fn defaults_resolve_cleanly() {
    let cfg = RunConfig::parse(&env(&[])).unwrap();
    assert_eq!(cfg.mode, BenchMode::Normal);
    assert_eq!(cfg.backend, BenchBackend::Hyrax);
    assert_eq!(cfg.hashes, 10);
    assert_eq!(cfg.total_hashes, 30);
    assert_eq!(cfg.k, DEFAULT_HYRAX_K);
    assert!(!cfg.k_overridden);
    // Combined-circuit dimensions: 1299H rows, 1302H − 3 columns, one
    // 2^14 × 2^14 padded domain at the default H = 10 per field.
    assert_eq!(
      (
        cfg.real_rows,
        cfg.real_cols,
        cfg.num_cons,
        cfg.num_vars,
        cfg.log_n
      ),
      (12990, 13017, 16384, 16384, 14)
    );
    let bd = RunConfig::parse(&env(&[("BDPCS", "1")])).unwrap();
    assert_eq!(bd.backend, BenchBackend::Brakedown);
    assert_eq!(bd.k, DEFAULT_BD_K);
  }

  #[test]
  fn field_order_matches_the_circuit_module() {
    // The bench-side string order is layout metadata for the SAME
    // semantic order the circuit module pins.
    let circuit: Vec<&str> = crate::poseidon2::FIELD_ORDER
      .iter()
      .map(|f| f.name())
      .collect();
    assert_eq!(FIELD_ORDER.to_vec(), circuit);
  }

  #[test]
  fn boolean_flags_are_strict() {
    // BDPCS=0 must NOT enable Brakedown (the old presence semantics).
    let cfg = RunConfig::parse(&env(&[("BDPCS", "0")])).unwrap();
    assert_eq!(cfg.backend, BenchBackend::Hyrax);
    for bad in ["true", "yes", "2", "", " 1"] {
      assert!(
        RunConfig::parse(&env(&[("BDPCS", bad)])).is_err(),
        "{bad:?}"
      );
      assert!(
        RunConfig::parse(&env(&[("PSIZE", bad)])).is_err(),
        "{bad:?}"
      );
      assert!(
        RunConfig::parse(&env(&[("KSWEEP", bad)])).is_err(),
        "{bad:?}"
      );
      assert!(
        RunConfig::parse(&env(&[("POSEIDON_ALLOW_LARGE", bad)])).is_err(),
        "{bad:?}"
      );
      assert!(
        RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", bad)])).is_err(),
        "{bad:?}"
      );
      assert!(
        RunConfig::parse(&env(&[("POSEIDON_ALLOW_DIRTY", bad)])).is_err(),
        "{bad:?}"
      );
    }
  }

  #[test]
  fn hashes_bounds_both_ends() {
    assert!(RunConfig::parse(&env(&[("HASHES", "0")])).is_err());
    assert_eq!(
      RunConfig::parse(&env(&[("HASHES", "1")])).unwrap().hashes,
      1
    );
    assert_eq!(
      RunConfig::parse(&env(&[("HASHES", "256")])).unwrap().hashes,
      256
    );
    assert!(RunConfig::parse(&env(&[("HASHES", "257")])).is_err());
    let large = RunConfig::parse(&env(&[("HASHES", "257"), ("POSEIDON_ALLOW_LARGE", "1")]));
    assert_eq!(large.unwrap().hashes, 257);
    assert!(RunConfig::parse(&env(&[("HASHES", "abc")])).is_err());
    assert!(
      RunConfig::parse(&env(&[
        ("HASHES", "4294967296"),
        ("POSEIDON_ALLOW_LARGE", "1")
      ]))
      .is_err()
    );
  }

  #[test]
  fn k_override_bounds_and_backend_conflicts() {
    for (key, backend_env) in [("IMOD_K", vec![]), ("BDK", vec![("BDPCS", "1")])] {
      for k in ["6", "14"] {
        let mut e = backend_env.clone();
        e.push((key, k));
        assert!(RunConfig::parse(&env(&e)).is_err(), "{key}={k}");
      }
      for k in ["7", "13"] {
        let mut e = backend_env.clone();
        e.push((key, k));
        let cfg = RunConfig::parse(&env(&e)).unwrap();
        assert_eq!(cfg.k, k.parse::<usize>().unwrap());
        assert!(cfg.k_overridden);
      }
    }
    // Both k variables: always an error.
    assert!(RunConfig::parse(&env(&[("IMOD_K", "9"), ("BDK", "9")])).is_err());
    // Irrelevant-backend k variables.
    assert!(RunConfig::parse(&env(&[("BDK", "9")])).is_err()); // Hyrax rejects BDK
    assert!(RunConfig::parse(&env(&[("BDPCS", "1"), ("IMOD_K", "9")])).is_err());
  }

  #[test]
  fn ksweep_rejects_conflicting_presence() {
    assert_eq!(
      RunConfig::parse(&env(&[("KSWEEP", "1")])).unwrap().mode,
      BenchMode::KSweep
    );
    // Presence is the conflict, even when the value equals a default.
    for (key, val) in [
      ("PSIZE", "0"),
      ("HASHES", "10"),
      ("IMOD_K", "9"),
      ("BDK", "11"),
      ("POSEIDON_ALLOW_LARGE", "0"),
    ] {
      assert!(
        RunConfig::parse(&env(&[("KSWEEP", "1"), (key, val)])).is_err(),
        "KSWEEP with {key}={val}"
      );
    }
    // POSEIDON_ALLOW_KNOBS does not waive mode conflicts.
    assert!(
      RunConfig::parse(&env(&[
        ("KSWEEP", "1"),
        ("HASHES", "10"),
        ("POSEIDON_ALLOW_KNOBS", "1")
      ]))
      .is_err()
    );
  }

  #[test]
  fn psize_mode_resolves() {
    let cfg = RunConfig::parse(&env(&[("PSIZE", "1")])).unwrap();
    assert_eq!(cfg.mode, BenchMode::ProofSize);
  }

  #[test]
  fn hidden_knobs_require_the_override_and_validate() {
    for knob in HIDDEN_KNOBS {
      let e = env(&[(knob, "1")]);
      assert!(RunConfig::parse(&e).is_err(), "{knob} without override");
    }
    // With the override: values are validated, not inherited blindly.
    let ok = RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("BDSPEC", "5")])).unwrap();
    assert!(ok.knobs.contains_key("BDSPEC"));
    assert!(RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("BDSPEC", "6")])).is_err());
    assert!(RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("BDROWLEN", "0")])).is_err());
    assert!(RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("BDDIRECT", "x")])).is_err());
    // Presence-sensitive knobs: "0" is still present (and permitted only
    // under the override).
    let cfg = RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("BDSPLIT", "0")])).unwrap();
    assert!(cfg.knobs["BDSPLIT"].effective.contains("present"));
    // GKRSKIP inverted sense.
    let cfg = RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("GKRSKIP", "0")])).unwrap();
    assert!(cfg.knobs["GKRSKIP"].effective.contains("DISABLED"));
    let cfg = RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("GKRSKIP", "1")])).unwrap();
    assert!(cfg.knobs["GKRSKIP"].effective.contains("enabled"));
    // RUST_LOG must be EnvFilter-valid.
    assert!(RunConfig::parse(&env(&[("POSEIDON_ALLOW_KNOBS", "1"), ("RUST_LOG", "info")])).is_ok());
    assert!(
      RunConfig::parse(&env(&[
        ("POSEIDON_ALLOW_KNOBS", "1"),
        ("RUST_LOG", "info[=]/")
      ]))
      .is_err()
    );
  }

  #[test]
  fn protocol_bytes_are_deterministic_and_extractable() {
    let cfg = RunConfig::parse(&env(&[])).unwrap();
    let a = cfg.protocol_canonical_bytes();
    let b = cfg.protocol_canonical_bytes();
    assert_eq!(a, b);
    let full = serde_json::json!({ "protocol": cfg.protocol_json(), "environment": {} });
    let extracted =
      protocol_bytes_from_full_config(&serde_json::to_string(&full).unwrap()).unwrap();
    assert_eq!(a, extracted);
    // A config difference changes the protocol bytes.
    let other = RunConfig::parse(&env(&[("BDPCS", "1")])).unwrap();
    assert_ne!(a, other.protocol_canonical_bytes());
  }

  #[test]
  fn non_unicode_values_are_rejected() {
    #[cfg(unix)]
    {
      use std::os::unix::ffi::OsStringExt;
      let mut e = BTreeMap::new();
      e.insert(
        OsString::from("HASHES"),
        OsString::from_vec(vec![0x66, 0xff, 0xfe]),
      );
      assert!(RunConfig::parse(&e).is_err());
      // Presence-only knobs are covered too.
      let mut e = BTreeMap::new();
      e.insert(
        OsString::from("BDSPLIT"),
        OsString::from_vec(vec![0xff, 0xfe]),
      );
      e.insert(OsString::from("POSEIDON_ALLOW_KNOBS"), OsString::from("1"));
      assert!(RunConfig::parse(&e).is_err());
    }
  }
}
