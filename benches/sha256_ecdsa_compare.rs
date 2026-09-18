//! One method/configuration per process; fixture construction is outside timers.
mod common;
#[path = "support/sha256_ecdsa_fixture.rs"]
mod shared_fixture;

use bincode::Options;
use bitz::{piop::spartan::ecdsa_sha256::*, transcript::Blake3Transcript};
use flock_core::pcs::commit::Commitment;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::error::Error;
use std::{cell::RefCell, collections::HashMap, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Timing {
    Perfetto,
    WallClock,
}

#[derive(clap::Parser)]
#[command(args_override_self = true)]
struct Args {
    #[command(flatten)]
    cargo: common::cli::CargoArgs,
    #[arg(long, value_parser = ["bitz-split", "bitz-all", "binius64", "binius64-ligerito"])]
    method: String,
    #[arg(long)]
    r: usize,
    #[arg(long)]
    c: usize,
    #[arg(long, default_value = "100", value_parser = common::cli::ecdsa_target)]
    target: u32,
    #[arg(long, default_value = "1", value_parser = common::cli::positive)]
    threads: usize,
    #[arg(long, default_value = "3", value_parser = common::cli::positive)]
    reps: usize,
    #[arg(long, default_value = "0")]
    seed: u64,
    #[arg(long)]
    fixture: Option<std::path::PathBuf>,
    #[arg(long)]
    export_fixture: Option<std::path::PathBuf>,
    #[arg(long)]
    binius64_worker: Option<std::path::PathBuf>,
    #[arg(long = "log-inv-rate", alias = "binius-log-inv-rate", default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=3))]
    log_inv_rate: u8,
    /// Wall-clock reports top-level timings without recording internal spans.
    #[arg(long, value_enum, default_value = "perfetto")]
    timing: Timing,
}

impl Args {
    fn exponent(&self) -> usize {
        self.r + self.c
    }
}

type Fixture = shared_fixture::SignedFixture;

fn fixture(args: &Args) -> Result<Fixture> {
    let fixture = match &args.fixture {
        Some(path) => Fixture::read(path)?,
        None => Fixture::generate(args.exponent() as u8, args.seed)?,
    };
    if fixture.log_compressions as usize != args.exponent() || fixture.seed != args.seed {
        return Err("fixture configuration mismatch".into());
    }
    Ok(fixture)
}

fn statement(fixture: &Fixture) -> Sha256EcdsaStatement {
    Sha256EcdsaStatement {
        log_compressions: fixture.log_compressions,
        qx: fixture.qx,
        qy: fixture.qy,
        r: fixture.r,
        s: fixture.s,
    }
}

fn dispatch_binius(args: &Args) -> Result<()> {
    let worker = args
        .binius64_worker
        .as_ref()
        .ok_or("binius64 requires --binius64-worker PATH")?;
    let mut command = std::process::Command::new(worker);
    command.args([
        "--method",
        &args.method,
        "--r",
        &args.r.to_string(),
        "--c",
        &args.c.to_string(),
        "--target",
        &args.target.to_string(),
        "--log-inv-rate",
        &args.log_inv_rate.to_string(),
        "--threads",
        &args.threads.to_string(),
        "--reps",
        &args.reps.to_string(),
        "--seed",
        &args.seed.to_string(),
    ]);
    if let Some(path) = &args.fixture {
        command.arg("--fixture").arg(path);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(command.exec().into())
    }
    #[cfg(not(unix))]
    {
        std::process::exit(command.status()?.code().unwrap_or(1));
    }
}

fn setup<T, E: Into<Box<dyn Error>>>(
    timing: Timing,
    f: impl FnOnce() -> std::result::Result<T, E>,
) -> Result<(T, f64)> {
    if timing == Timing::WallClock {
        let start = Instant::now();
        let value = f();
        let ms = start.elapsed().as_secs_f64() * 1000.;
        return Ok((value.map_err(Into::into)?, ms));
    }
    let (value, duration) = bitz::observability::measure(tracing::info_span!("benchmark:setup"), f)?;
    Ok((value.map_err(Into::into)?, duration.as_secs_f64() * 1000.))
}

/// Both backends use the same operation boundaries. Only Perfetto captures
/// nested protocol phases; wall-clock measurements stay in this harness.
struct TrialTiming {
    recording: Option<bitz::observability::Recording<Vec<u8>>>,
    wall_ms: RefCell<HashMap<&'static str, f64>>,
}

impl TrialTiming {
    fn start(timing: Timing) -> Result<Self> {
        Ok(Self {
            recording: if timing == Timing::Perfetto {
                Some(bitz::observability::Recording::start(Vec::new())?)
            } else {
                None
            },
            wall_ms: RefCell::new(HashMap::new()),
        })
    }

    fn enter(&self, name: &'static str, span: tracing::Span) -> TimedScope<'_> {
        TimedScope {
            timer: self,
            name,
            _span: span.entered(),
            start: self.recording.is_none().then(Instant::now),
        }
    }

    fn finish(self) -> Result<TrialMeasurements> {
        Ok(TrialMeasurements {
            intervals: self.recording.map(|r| r.intervals()).transpose()?,
            wall_ms: self.wall_ms.into_inner(),
        })
    }
}

struct TimedScope<'a> {
    timer: &'a TrialTiming,
    name: &'static str,
    _span: tracing::span::EnteredSpan,
    start: Option<Instant>,
}

impl TimedScope<'_> {
    fn in_scope<T>(self, f: impl FnOnce() -> T) -> T {
        f()
    }
}

impl Drop for TimedScope<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let ms = start.elapsed().as_secs_f64() * 1000.;
            self.timer.wall_ms.borrow_mut().insert(self.name, ms);
        }
    }
}

struct TrialMeasurements {
    intervals: Option<Vec<bitz::observability::Interval>>,
    wall_ms: HashMap<&'static str, f64>,
}

impl TrialMeasurements {
    fn ms(&self, name: &str) -> f64 {
        match &self.intervals {
            Some(intervals) => common::span_ms(intervals, name),
            None => self.wall_ms[name],
        }
    }

    fn phases(&self, name: &str) -> Result<Vec<(String, f64)>> {
        match &self.intervals {
            Some(intervals) => Ok(bitz::observability::phase_totals(intervals, name)?),
            None => Ok(Vec::new()),
        }
    }
}

macro_rules! timed {
    ($timer:expr, $name:literal) => {
        $timer.enter($name, tracing::info_span!($name))
    };
}

#[derive(Serialize, Deserialize)]
struct BitzWire {
    commitment: Commitment,
    proof: Vec<u8>,
}

fn bitz_revision() -> Option<&'static str> {
    Some(env!("BITZ_REVISION"))
}

#[derive(Serialize)]
struct Measurements<D> {
    setup_ms: f64,
    witness_ms: f64,
    commit_ms: f64,
    e2e_prover_ms: f64,
    protocol_ms: f64,
    verify_ms: f64,
    codec_ms: f64,
    proof_object_bytes: usize,
    proof_material_bytes: usize,
    outer_ms: Option<f64>,
    inner_ms: Option<f64>,
    opening_ms: Option<f64>,
    folding_ms: Option<f64>,
    #[serde(flatten)]
    details: D,
}

#[derive(Serialize)]
struct BitzDetails {
    proof_digest: String,
    prover_transcript: String,
    verifier_transcript: String,
    ligerito_profile: String,
    phases_seconds: Vec<(String, f64)>,
    verify_phases_seconds: Vec<(String, f64)>,
    security: Value,
    circuit: Value,
}

#[derive(Serialize)]
struct ResultRecord<'a, D> {
    schema: &'static str,
    timing: Timing,
    method: &'a str,
    zk: bool,
    fixture_profile: &'static str,
    trial: &'static str,
    sample: usize,
    log_compressions: usize,
    compressions: usize,
    message_bytes: usize,
    signatures: usize,
    r: Option<usize>,
    c: Option<usize>,
    security_target: Option<u32>,
    threads: usize,
    seed: u64,
    fixture_id: &'a str,
    statement_bytes: usize,
    statement: &'static str,
    bitz_revision: Option<&'static str>,
    verified: bool,
    prove_ms: f64,
    witness_to_proof_ms: f64,
    #[serde(flatten)]
    measurements: Measurements<D>,
}

fn result_record<'a, D>(
    args: &'a Args,
    fixture: &'a Fixture,
    trial: usize,
    row: Measurements<D>,
) -> ResultRecord<'a, D> {
    ResultRecord {
        schema: "bitz/sha256-ecdsa-compare/v1",
        timing: args.timing,
        method: &args.method,
        zk: false,
        fixture_profile: shared_fixture::SCHEMA,
        trial: if trial == 0 { "warmup" } else { "sample" },
        sample: trial,
        log_compressions: args.exponent(),
        compressions: 1usize << args.exponent(),
        message_bytes: fixture.message.len(),
        signatures: 1,
        r: None,
        c: None,
        security_target: Some(args.target),
        threads: args.threads,
        seed: args.seed,
        fixture_id: &fixture.id,
        statement_bytes: 129,
        statement: "public-key-signature; witness-message",
        bitz_revision: bitz_revision(),
        verified: true,
        prove_ms: row.commit_ms + row.protocol_ms,
        witness_to_proof_ms: row.witness_ms + row.commit_ms + row.protocol_ms,
        measurements: row,
    }
}

fn emit<D: Serialize>(args: &Args, fixture: &Fixture, trial: usize, row: Measurements<D>) {
    println!(
        "{}",
        serde_json::to_string(&result_record(args, fixture, trial, row))
            .expect("serialize SHA/ECDSA result")
    );
}
fn bitz(args: &Args, fixture: &Fixture, mode: OuterMode) -> Result<()> {
    let statement = statement(fixture);
    // The campaign runner selects the opener rate per case through
    // `BITZ_LIG_PROFILE`; record the request verbatim on every row.
    let ligerito_profile =
        std::env::var("BITZ_LIG_PROFILE").unwrap_or_else(|_| "default-by-target".into());
    let (prepared, setup_ms) = setup(args.timing, || {
        prepare_sha256_ecdsa(args.exponent(), args.target, mode)
            .and_then(|p| p.with_ligerito(common::ligerito_selection(args.target as usize)))
    })?;
    let security = prepared.security()?;
    for trial in 0..=args.reps {
        let recording = TrialTiming::start(args.timing)?;
        let e2e = timed!(recording, "benchmark:e2e");
        let witness = timed!(recording, "benchmark:witness")
            .in_scope(|| generate_sha256_ecdsa_witness(&prepared, &statement, &fixture.message))?;
        let hint = timed!(recording, "benchmark:commit")
            .in_scope(|| commit_sha256_ecdsa(&prepared, &witness))?;
        let mut prover_transcript = Blake3Transcript::new();
        let proof = timed!(recording, "benchmark:protocol").in_scope(|| {
            prove_sha256_ecdsa(
                &mut prover_transcript,
                &prepared,
                &statement,
                &witness,
                &hint,
                4,
            )
        })?;
        drop(e2e);
        let codec = timed!(recording, "benchmark:codec");
        let proof_bytes = proof.to_bytes();
        let object_bytes = proof_bytes.len();
        let proof_digest = blake3::hash(&proof_bytes).to_hex().to_string();
        let wire = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize(&BitzWire {
                commitment: hint.commitment.clone(),
                proof: proof_bytes,
            })?;
        let decoded: BitzWire = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(wire.len() as u64)
            .reject_trailing_bytes()
            .deserialize(&wire)?;
        let decoded_proof = Sha256EcdsaProof::from_bytes(&decoded.proof)?;
        drop(codec);
        let mut verifier_transcript = Blake3Transcript::new();
        timed!(recording, "benchmark:verification").in_scope(|| {
            fixture.validate_statement()?;
            verify_sha256_ecdsa(
                &mut verifier_transcript,
                &prepared,
                &statement,
                &decoded.commitment,
                &decoded_proof,
            )
            .map_err(|e| -> Box<dyn Error> { e.into() })
        })?;
        let timings = recording.finish()?;
        let witness_ms = timings.ms("benchmark:witness");
        let commit_ms = timings.ms("benchmark:commit");
        let protocol_ms = timings.ms("benchmark:protocol");
        let e2e_prover_ms = timings.ms("benchmark:e2e");
        let codec_ms = timings.ms("benchmark:codec");
        let verify_ms = timings.ms("benchmark:verification");
        let phases = timings.phases("benchmark:e2e")?;
        let verify_phases = timings.phases("benchmark:verification")?;
        let phase = |name: &str| {
            phases
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value * 1000.)
        };
        emit(
            args,
            fixture,
            trial,
            Measurements {
                setup_ms,
                witness_ms,
                commit_ms,
                e2e_prover_ms,
                protocol_ms,
                verify_ms,
                codec_ms,
                proof_object_bytes: object_bytes,
                proof_material_bytes: wire.len(),
                outer_ms: phase("ecdsa:outer_prove"),
                inner_ms: phase("ecdsa:shared_inner_prove"),
                opening_ms: phase("ecdsa:bitz_prove"),
                folding_ms: None,
                details: BitzDetails {
                    proof_digest,
                    prover_transcript: blake3::Hash::from(prover_transcript.state_digest()).to_hex().to_string(),
                    verifier_transcript: blake3::Hash::from(verifier_transcript.state_digest()).to_hex().to_string(),
                    ligerito_profile: ligerito_profile.clone(),
                    phases_seconds: phases,
                    verify_phases_seconds: verify_phases,
                    security: json!({"model": "round-by-round-economic", "economic_bits": security.compute_economic_security_bits(),
                    "statistical_bits_lower_bound": security.compute_statistical_security_bits(), "projection_bits": 113,
                    "ligerito": common::ligerito_report(prepared.ligerito_configuration(), prepared.ligerito_configuration().round0(args.target)?) }),
                    circuit: json!({"nonlinear_rows": prepared.nonlinear_rows(), "linear_rows": prepared.linear_rows(),
                    "outer_active_rows": prepared.outer_rows(), "outer_slots": prepared.outer_domain_size(),
                    "source_bits": prepared.live_source_bits(), "assignment_bits": prepared.live_assignment_bits()}),
                },
            },
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = <Args as clap::Parser>::parse();
    let exponent = args.r.checked_add(args.c).ok_or("exponent overflow")?;
    if !(3..=16).contains(&exponent) {
        return Err("require 3 <= r+c <= 16 and target 100/128".into());
    }
    if args.method == "binius64-ligerito" && args.target != 100 {
        return Err("the BitZ opener gate is fixed at 100 bits".into());
    }
    if let Some(path) = &args.export_fixture {
        return shared_fixture::SignedFixture::generate(args.exponent() as u8, args.seed)?
            .write(path);
    }
    if args.method.starts_with("binius64") {
        if args.timing == Timing::WallClock {
            return Err("--timing wall-clock supports bitz-split, bitz-all".into());
        }
        return dispatch_binius(&args);
    }
    if args.timing == Timing::Perfetto {
        bitz::observability::install()?;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;

    let fixture = fixture(&args)?;
    common::start_gkr_recording();
    let result = match args.method.as_str() {
        "bitz-split" => bitz(&args, &fixture, OuterMode::Split),
        "bitz-all" => bitz(&args, &fixture, OuterMode::AllRows),
        _ => unreachable!(),
    };
    common::print_gkr_schedules();
    result
}

#[cfg(test)]
mod reporting_tests {
    use super::*;

    #[test]
    fn result_envelope_keeps_totals_nulls_and_trial_numbering() {
        let mut args = Args {
            cargo: Default::default(),
            method: "bitz-split".into(),
            r: 1,
            c: 2,
            target: 100,
            threads: 1,
            reps: 1,
            seed: 0,
            fixture: None,
            export_fixture: None,
            binius64_worker: None,
            log_inv_rate: 1,
            timing: Timing::Perfetto,
        };
        let fixture = Fixture::generate(3, 0).unwrap();
        let row = || Measurements {
            setup_ms: 1.0,
            witness_ms: 2.0,
            commit_ms: 3.0,
            e2e_prover_ms: 15.0,
            protocol_ms: 4.0,
            verify_ms: 5.0,
            codec_ms: 6.0,
            proof_object_bytes: 7,
            proof_material_bytes: 8,
            outer_ms: None,
            inner_ms: None,
            opening_ms: None,
            folding_ms: None,
            details: BitzDetails {
                proof_digest: "test-proof".into(),
                prover_transcript: "test-prover".into(),
                verifier_transcript: "test-verifier".into(),
                ligerito_profile: "custom:1:4".into(),
                phases_seconds: vec![("commit".into(), 0.003)],
                verify_phases_seconds: vec![],
                security: json!({"bits":100}),
                circuit: json!({}),
            },
        };
        let warmup = serde_json::to_value(result_record(&args, &fixture, 0, row())).unwrap();
        assert_eq!(warmup["trial"], "warmup");
        assert_eq!(warmup["sample"], 0);
        assert_eq!(warmup["prove_ms"], 7.0);
        assert_eq!(warmup["witness_to_proof_ms"], 9.0);
        for key in ["r", "c", "outer_ms", "inner_ms", "opening_ms", "folding_ms"] {
            assert!(warmup.get(key).unwrap().is_null(), "{key}");
        }
        assert_eq!(warmup["phases_seconds"], json!([["commit", 0.003]]));
        assert!(warmup.get("phases_ms").is_none());
        assert!(warmup["proof_object_bytes"].is_u64());
        args.method = "bitz-all".into();
        let sample = serde_json::to_value(result_record(&args, &fixture, 1, row())).unwrap();
        assert_eq!(sample["trial"], "sample");
        assert_eq!(sample["sample"], 1);
        assert!(sample["r"].is_null());
        assert!(sample["c"].is_null());
        assert_eq!(sample["security_target"], 100);
    }
}

#[cfg(test)]
mod cli_tests {
    use super::Args;
    use clap::{CommandFactory, Parser, error::ErrorKind};

    fn parse(extra: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(["ecdsa", "--method", "bitz-split", "--r", "1", "--c", "2"]
            .into_iter().chain(extra.iter().copied()))
    }

    #[test]
    fn defaults_script_options_and_last_value_wins() {
        Args::command().debug_assert();
        let defaults = parse(&["--bench"]).unwrap();
        assert_eq!((defaults.exponent(), defaults.target, defaults.threads, defaults.reps, defaults.seed),
            (3, 100, 1, 3, 0));
        let args = parse(&["--method", "bitz-all", "--r", "14", "--c", "2", "--target", "128",
            "--threads", "8", "--reps", "5", "--seed", "42", "--fixture", "fixture.json",
            "--export-fixture", "export.json", "--binius64-worker", "worker"]).unwrap();
        assert_eq!((args.method.as_str(), args.exponent(), args.target, args.threads, args.reps, args.seed),
            ("bitz-all", 16, 128, 8, 5, 42));
        assert_eq!(args.fixture.as_deref(), Some(std::path::Path::new("fixture.json")));
        assert_eq!(args.export_fixture.as_deref(), Some(std::path::Path::new("export.json")));
        assert_eq!(args.binius64_worker.as_deref(), Some(std::path::Path::new("worker")));
        for method in ["bitz-all", "binius64"] {
            assert_eq!(parse(&["--method", method]).unwrap().method, method);
        }
    }

    #[test]
    fn rejects_unknown_missing_and_invalid_scalar_options() {
        for value in ["100", "0100", "+100"] {
            assert_eq!(parse(&["--target", value]).unwrap().target, 100);
        }
        for extra in [
            &["--method", "unknown"][..], &["--target", "114"], &["--target", "129"],
            &["--threads", "0"], &["--reps", "0"], &["--r", "nope"],
            &["--seed", "-1"], &["--unknown"], &["--fixture"],
        ] {
            assert!(parse(extra).is_err(), "accepted {extra:?}");
        }
        for argv in [&["ecdsa"][..], &["ecdsa", "--method", "bitz-split", "--r", "1"]] {
            assert!(Args::try_parse_from(argv).is_err());
        }
        assert_eq!(Args::try_parse_from(["ecdsa", "--help"]).err().unwrap().kind(), ErrorKind::DisplayHelp);
    }
}
