//! Immutable run-config emitter for the Poseidon2 benchmark runner.
//!
//! Resolves the benchmark flags through the shared crate-side parser
//! (`limber::poseidon_bench::RunConfig::parse`) — the shell runner never
//! interprets flags itself — then gathers the source/toolchain/host
//! fields and emits one canonical sorted-key JSON document on stdout.
//!
//! `--check <path>` regenerates the document in memory and byte-compares
//! it against the file, exiting nonzero on drift (used immediately before
//! benchmark launch and again after a run to catch source/toolchain
//! changes during the measurement).
//!
//! Only immutable, result-affecting inputs belong here; timestamps, exit
//! status, audit counters, logs, and artifact hashes belong to the run
//! manifest and are excluded so the config hash is stable. Host-only:
//! the CI WASM job builds the library alone.

#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms
)]
#![allow(non_snake_case)]

use limber::poseidon_bench::{RunConfig, canonical_json_bytes};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::process::{Command, ExitCode};

fn sha256_hex(bytes: &[u8]) -> String {
  let mut h = Sha256::new();
  h.update(bytes);
  hex_string(&h.finalize())
}

fn hex_string(bytes: &[u8]) -> String {
  bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
  let out = Command::new(cmd)
    .args(args)
    .output()
    .map_err(|e| format!("{cmd} {args:?}: {e}"))?;
  if !out.status.success() {
    return Err(format!("{cmd} {args:?} exited with {}", out.status));
  }
  String::from_utf8(out.stdout).map_err(|e| format!("{cmd} output not UTF-8: {e}"))
}

/// A recognized, result-affecting environment value must be valid
/// Unicode; absence is fine.
fn env_unicode(key: &str) -> Result<Option<String>, String> {
  match std::env::var_os(key) {
    None => Ok(None),
    Some(v) => v
      .to_str()
      .map(|s| Some(s.to_string()))
      .ok_or_else(|| format!("{key} is not valid Unicode")),
  }
}

/// Version of a package pinned in Cargo.lock (exact resolved version).
fn locked_version(lock: &str, package: &str) -> Option<String> {
  let needle = format!("name = \"{package}\"");
  let mut lines = lock.lines();
  while let Some(line) = lines.next() {
    if line.trim() == needle
      && let Some(v) = lines.next()
    {
      return v
        .trim()
        .strip_prefix("version = \"")
        .and_then(|s| s.strip_suffix('"'))
        .map(str::to_string);
    }
  }
  None
}

fn gather(config: &RunConfig) -> Result<serde_json::Value, String> {
  // Source state.
  let git_sha = run("git", &["rev-parse", "HEAD"])?.trim().to_string();
  let porcelain = run("git", &["status", "--porcelain"])?;
  let dirty = !porcelain.trim().is_empty();
  if dirty && !config.allow_dirty {
    return Err(
      "working tree is dirty; canonical runs require a clean tree \
       (POSEIDON_ALLOW_DIRTY=1 permits a non-publishable exploratory run)"
        .to_string(),
    );
  }
  let dirty_hash = if dirty {
    // Hash the tracked diff plus every untracked (non-ignored) file's
    // path and contents, so the exact exploratory source state is pinned.
    let diff = Command::new("git")
      .args(["diff", "--binary", "HEAD"])
      .output()
      .map_err(|e| format!("git diff: {e}"))?;
    if !diff.status.success() {
      return Err("git diff --binary HEAD failed".to_string());
    }
    let mut h = Sha256::new();
    h.update(&diff.stdout);
    let untracked = run("git", &["ls-files", "--others", "--exclude-standard"])?;
    for path in untracked.lines().filter(|l| !l.is_empty()) {
      h.update(path.as_bytes());
      h.update([0u8]);
      let contents = std::fs::read(path).map_err(|e| format!("read untracked file {path}: {e}"))?;
      h.update((contents.len() as u64).to_le_bytes());
      h.update(&contents);
    }
    Some(hex_string(&h.finalize()))
  } else {
    None
  };
  let lock = std::fs::read_to_string("Cargo.lock").map_err(|e| format!("Cargo.lock: {e}"))?;
  let lock_sha = sha256_hex(lock.as_bytes());

  // Toolchain.
  let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
  let rustc_vv = run(&rustc, &["-vV"])?;
  let cargo_version = run("cargo", &["--version"])?.trim().to_string();
  let criterion_version = locked_version(&lock, "criterion");
  let jem = cfg!(feature = "jem");
  let allocator = if jem {
    locked_version(&lock, "tikv-jemallocator")
      .map(|v| format!("tikv-jemallocator {v}"))
      .unwrap_or_else(|| "tikv-jemallocator (unlocked)".to_string())
  } else {
    "system".to_string()
  };

  // Host.
  let os = std::env::consts::OS;
  let arch = std::env::consts::ARCH;
  let kernel = run("uname", &["-a"]).unwrap_or_else(|_| "unknown".to_string());
  let cpu_model = if cfg!(target_os = "macos") {
    run("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_else(|_| "unknown".to_string())
  } else {
    std::fs::read_to_string("/proc/cpuinfo")
      .ok()
      .and_then(|s| {
        s.lines()
          .find(|l| l.starts_with("model name"))
          .and_then(|l| l.split(':').nth(1))
          .map(|m| m.trim().to_string())
      })
      .unwrap_or_else(|| "unknown".to_string())
  };
  let logical_cores = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(0);
  let physical_cores = if cfg!(target_os = "macos") {
    run("sysctl", &["-n", "hw.physicalcpu"])
      .ok()
      .and_then(|s| s.trim().parse::<usize>().ok())
  } else {
    None
  };

  // Result-affecting environment.
  let rayon_threads = env_unicode("RAYON_NUM_THREADS")?;
  let rustflags = env_unicode("RUSTFLAGS")?;
  let target_features = {
    let mut feats: Vec<&str> = Vec::new();
    if cfg!(target_feature = "avx2") {
      feats.push("avx2");
    }
    if cfg!(target_feature = "aes") {
      feats.push("aes");
    }
    if cfg!(target_feature = "neon") {
      feats.push("neon");
    }
    feats
  };

  Ok(serde_json::json!({
    "protocol": config.protocol_json(),
    "environment": {
      "git_sha": git_sha,
      "git_dirty": dirty,
      "dirty_source_hash": dirty_hash,
      "publishable": !dirty,
      "cargo_lock_sha256": lock_sha,
      "os": os,
      "kernel": kernel.trim(),
      "arch": arch,
      "cpu_model": cpu_model.trim(),
      "logical_cores": logical_cores,
      "physical_cores": physical_cores,
      "rustc_vV": rustc_vv.trim(),
      "cargo_version": cargo_version,
      "criterion_version": criterion_version,
      "allocator": allocator,
      "jem_feature": jem,
      "rayon_num_threads": rayon_threads,
      "rustflags": rustflags,
      "target_features": target_features,
    },
  }))
}

fn main() -> ExitCode {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let env_map: BTreeMap<OsString, OsString> = std::env::vars_os().collect();

  let config = match RunConfig::parse(&env_map) {
    Ok(c) => c,
    Err(e) => {
      eprintln!("poseidon_bench_config: {e}");
      return ExitCode::FAILURE;
    }
  };
  let doc = match gather(&config) {
    Ok(d) => d,
    Err(e) => {
      eprintln!("poseidon_bench_config: {e}");
      return ExitCode::FAILURE;
    }
  };
  let bytes = canonical_json_bytes(&doc);

  match args.as_slice() {
    [] => {
      use std::io::Write;
      std::io::stdout()
        .write_all(&bytes)
        .expect("stdout write failed");
      ExitCode::SUCCESS
    }
    [flag, path] if flag == "--check" => {
      let on_disk = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
          eprintln!("poseidon_bench_config: read {path}: {e}");
          return ExitCode::FAILURE;
        }
      };
      if on_disk == bytes {
        eprintln!("poseidon_bench_config: {path} is up to date");
        ExitCode::SUCCESS
      } else {
        eprintln!(
          "poseidon_bench_config: {path} DRIFTS from the regenerated config \
           (source, toolchain, host, or flags changed)"
        );
        ExitCode::FAILURE
      }
    }
    other => {
      eprintln!("usage: poseidon_bench_config [--check <run-config.json>], got {other:?}");
      ExitCode::FAILURE
    }
  }
}
