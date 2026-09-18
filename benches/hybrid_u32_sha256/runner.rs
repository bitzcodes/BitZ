//! Shared benchmark and proof CLI runner. Run each mode in its own process for
//! meaningful peak-memory comparisons; every mode uses BLAKE3 Merkle hashing.
use ::bitz::piop::spartan::protocol::{self, PreparedRelation};
use bitz::piop::spartan::MulRow;
use bitz::piop::spartan::mul::{MulLayout, MulWitness};

use binius_circuits::sha256::compress::{State, sha256_compress_2x_seq};
use binius_core::{constraint_system::ValueVec, word::Word};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_hash::Blake3HashSuite;
use binius_prover::{OptimalPackedB128, Prover};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
use binius_verifier::Verifier;
use bitz::{
    binius_ligerito::{Accounting, Prepared as BiniusLigerito},
    hybrid::{CompositionProfile, Parameters, PreparedHybrid, chaining_value},
    observability::{self, Interval, Recording},
    transcript::Blake3Transcript,
};
use std::error::Error;

#[path = "../common/cli.rs"]
mod cli;
#[path = "../common/output.rs"]
mod output;
#[path = "report.rs"]
mod report;
#[path = "sweep.rs"]
mod sweep;
use output::{BenchmarkOutput, FileMode, JsonStyle};
use report::{HybridRow, NativeRow};

type Challenger = HasherChallenger<blake3::Hasher>;
type AnyError = Box<dyn Error>;

fn select_ligerito(
    cli: Option<&str>,
    target: usize,
) -> Result<bitz::ligerito_flock::LigeritoSelection, AnyError> {
    match cli {
        Some(request) => Ok(bitz::ligerito_flock::LigeritoSelection::parse(
            request, target,
        )?),
        None => Ok(bitz::ligerito_flock::LigeritoSelection::JOHNSON),
    }
}

/// How the native Binius64 circuit is proved: Binius64's own ring switch +
/// BaseFold/FRI, or its PIOP prefix with every oracle committed and opened by
/// the BitZ opener (Johnson regime, grinding, Round 0; rate and accounting
/// from `BITZ_BINIUS_LOG_INV_RATE` / `BITZ_BINIUS_LIGERITO_ACCOUNTING`, gated
/// at 100 bits).
enum NativeBackend {
    Binius {
        prover: Prover<OptimalPackedB128, Blake3HashSuite>,
        verifier: Verifier<Blake3HashSuite>,
        config: BiniusConfig,
    },
    Ligerito(BiniusLigerito),
}

struct Native {
    circuit: Circuit,
    blocks: Vec<[Wire; 16]>,
    output: [Wire; 8],
    multiplications: Vec<[Wire; 4]>,
    backend: NativeBackend,
}

impl Native {
    fn new(multiplications: usize, compressions: usize, ligerito: bool, binius: Option<BiniusConfig>) -> Result<Self, AnyError> {
        let builder = CircuitBuilder::new();
        let multiplications = (0..multiplications)
            .map(|_| bitz::hybrid::mod32_binius::add_u32_mul_mod32(&builder))
            .collect();
        let blocks: Vec<[Wire; 16]> = (0..compressions)
            .map(|_| std::array::from_fn(|_| builder.add_witness()))
            .collect();
        let output = std::array::from_fn(|_| builder.add_inout());
        let mut state = State::iv(&builder);
        for pair in blocks.chunks_exact(2) {
            state = sha256_compress_2x_seq(&builder, state, [pair[0], pair[1]]);
        }
        let mask = builder.add_constant(Word(u32::MAX as u64));
        for (actual, expected) in state.0.into_iter().zip(output) {
            builder.assert_eq(
                "final_sha_chaining_value",
                builder.band(actual, mask),
                expected,
            );
        }
        let circuit = builder.build();
        let backend = if ligerito {
            NativeBackend::Ligerito(BiniusLigerito::with_options(
                circuit.constraint_system(),
                binius_ligerito_log_inv_rate(),
                binius_ligerito_accounting(),
            )?)
        } else {
            // 112 bits for the FRI component leaves slack for the binary PIOPs
            // and the second proof in the separate mode. No default 96-bit preset.
            let config = binius.expect("Binius configuration");
            let verifier = Verifier::<Blake3HashSuite>::setup_with_security_bits(
                circuit.constraint_system().clone(),
                config.log_inv_rate,
                config.security_bits,
            )?;
            let prover = Prover::setup(verifier.clone())?;
            NativeBackend::Binius { prover, verifier, config }
        };
        Ok(Self {
            circuit,
            blocks,
            output,
            multiplications,
            backend,
        })
    }
    fn populate(&self, inputs: &[MulRow<u32>], blocks: &[[u32; 16]]) -> Result<ValueVec, AnyError> {
        if inputs.len() != self.multiplications.len() || blocks.len() != self.blocks.len() {
            return Err("native witness shape does not match the circuit".into());
        }
        let mut filler = self.circuit.new_witness_filler();
        for (wires, row) in self.multiplications.iter().zip(inputs) {
            for (&wire, value) in wires.iter().zip([row.x, row.y, row.lo, row.hi]) {
                filler[wire] = Word(value as u64);
            }
        }
        for (wires, block) in self.blocks.iter().zip(blocks) {
            for (&wire, &word) in wires.iter().zip(block) {
                filler[wire] = Word(word as u64);
            }
        }
        for (&wire, word) in self.output.iter().zip(chaining_value(blocks)) {
            filler[wire] = Word(word as u64);
        }
        self.circuit.populate_wire_witness(&mut filler)?;
        Ok(filler.into_value_vec())
    }
    fn prove(&self, witness: &ValueVec) -> Result<Vec<u8>, AnyError> {
        match &self.backend {
            NativeBackend::Binius { prover, .. } => {
                let mut t = ProverTranscript::new(Challenger::default());
                prover.prove(witness, &mut t)?;
                Ok(t.finalize())
            }
            NativeBackend::Ligerito(prepared) => {
                let proof = prepared.prove(witness)?;
                Ok(proof.to_bytes())
            }
        }
    }
    fn verify(&self, witness: &ValueVec, bytes: Vec<u8>) -> Result<(), AnyError> {
        match &self.backend {
            NativeBackend::Binius { verifier, .. } => {
                let mut t = VerifierTranscript::new(Challenger::default(), bytes);
                verifier.verify(witness.inout(), &mut t)?;
                t.finalize()?;
                Ok(())
            }
            NativeBackend::Ligerito(prepared) => {
                let decoded = prepared.proof_from_bytes(&bytes)?;
                prepared.verify(witness.inout(), &decoded)?;
                Ok(())
            }
        }
    }
    fn setup_line(&self) -> String {
        match &self.backend {
            NativeBackend::Binius { config, .. } => format!(
                "binius_fri_component_bits={} binius_log_inv_rate={}",
                config.security_bits,
                config.log_inv_rate
            ),
            NativeBackend::Ligerito(prepared) => {
                let security = prepared.security();
                let witness = prepared.opener(0);
                // Per oracle, per level: (fold-challenge grinding bits, query
                // grinding bits, queries) — the proof-of-work the opener pays.
                let ladders: Vec<String> = (0..prepared.oracle_specs().len())
                    .map(|i| {
                        let levels: Vec<String> = prepared
                            .opener(i)
                            .config()
                            .levels
                            .iter()
                            .map(|l| {
                                format!(
                                    "(k={},fold_grind={},query_grind={},queries={})",
                                    l.k_recursive, l.fold_grinding_bits, l.grinding_bits, l.queries
                                )
                            })
                            .collect();
                        format!("oracle{i}[{}]", levels.join(","))
                    })
                    .collect();
                format!(
                    "ligerito_component_bits={} accounting={} algebraic_security_bits={:.3} union_bound_bits={:.3} round_by_round_bits={:.3} level0_queries={} level0_fold_grinding_bits={} ood_grinding_bits={} log_inv_rate={} oracle_logs={:?} binding_term={} ladders={}",
                    prepared.component_bits(),
                    security.accounting.name(),
                    security.algebraic_bits,
                    security.union_bound_bits,
                    security.round_by_round_bits,
                    witness.level0_queries(),
                    witness.level0_fold_grinding_bits(),
                    witness.ood_grinding_bits(),
                    prepared.log_inv_rate(),
                    prepared
                        .oracle_specs()
                        .iter()
                        .map(|s| s.log_msg_len)
                        .collect::<Vec<_>>(),
                    security
                        .binding_term()
                        .map(|t| format!("{}:{:.2}", t.name, -t.error_bound.log2()))
                        .unwrap_or_default(),
                    ladders.join(" ")
                )
            }
        }
    }
}

fn millis(intervals: &[Interval], label: &str) -> f64 {
    observability::duration(intervals, label).expect("required hybrid interval").as_secs_f64() * 1000.0
}
fn peak_kib() -> u64 {
    // Linux only: this target is also built as a [[bin]], which does not get
    // dev-dependencies, so no libc/getrusage path is available here. On other
    // platforms the sweep's per-case child processes are sampled externally.
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}

#[derive(clap::Parser)]
#[command(args_override_self = true, name = "hybrid-u32-sha256", about = "Non-ZK SHA chain and multiplication benchmark (100-bit composition target)", after_help = "Set RAYON_NUM_THREADS to control threads. Single runs default to 2^20 products and 2^16 compressions. Sweeps default to equal packed witnesses (15:7 through 20:12). Results directories must not already exist.")]
struct Args {
    #[command(flatten)]
    cargo: cli::CargoArgs,
    #[arg(long, default_value = "hybrid", requires_if("all", "sweep"), value_parser = ["hybrid", "separate", "all-binius", "binius-ligerito", "all"])]
    mode: String,
    #[arg(long, env = "BITZ_LIG_PROFILE")]
    profile: Option<String>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(9..=22))]
    mul_log: Option<u32>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=16))]
    sha_log: Option<u32>,
    /// Measured iterations after one warmup.
    #[arg(long, default_value = "5", value_parser = cli::positive)]
    iterations: usize,
    #[arg(long, conflicts_with_all = ["mul_log", "sha_log", "output", "verify"])]
    sweep: bool,
    #[arg(long, requires = "sweep", value_parser = |value: &str| sweep::parse_shapes(value).map_err(|error| error.to_string()))]
    shapes: Option<cli::List<sweep::Shape>>,
    #[arg(long, requires = "sweep")]
    results_dir: Option<std::path::PathBuf>,
    #[arg(long, conflicts_with = "verify")]
    output: Option<String>,
    #[arg(long)]
    verify: Option<String>,
}

#[derive(clap::Parser, Clone, Copy)]
struct BiniusConfig {
    #[arg(long, env = "BITZ_HYBRID_BINIUS_LOG_INV_RATE", default_value = "1")]
    log_inv_rate: usize,
    #[arg(long, env = "BITZ_HYBRID_BINIUS_SECURITY_BITS", default_value = "112")]
    security_bits: usize,
}

/// BitZ-opener rate for `binius-ligerito` mode: `BITZ_BINIUS_LOG_INV_RATE`
/// (1 = rate 1/2, 3 = rate 1/8), the same knob the mul benches read.
fn binius_ligerito_log_inv_rate() -> usize {
    std::env::var("BITZ_BINIUS_LOG_INV_RATE")
        .ok()
        .map(|v| {
            v.parse()
                .expect("BITZ_BINIUS_LOG_INV_RATE must be an integer")
        })
        .unwrap_or(bitz::binary_pcs::LOG_INV_RATE)
}

/// `BITZ_BINIUS_LIGERITO_ACCOUNTING`: `union` (default) or `rbr`, exactly as
/// `benches/mul_compare.rs` reads it.
fn binius_ligerito_accounting() -> Accounting {
    match std::env::var("BITZ_BINIUS_LIGERITO_ACCOUNTING").as_deref() {
        Err(_) | Ok("union") | Ok("union-bound") => Accounting::UnionBound,
        Ok("rbr") | Ok("round-by-round") => Accounting::RoundByRound,
        Ok(other) => panic!("BITZ_BINIUS_LIGERITO_ACCOUNTING must be union or rbr, not {other:?}"),
    }
}

pub fn run() -> Result<(), AnyError> {
    let Args { mode, profile, mul_log, sha_log, iterations, sweep, shapes, results_dir, output, verify: verify_file, .. } = <Args as clap::Parser>::parse();
    if (output.is_some() || verify_file.is_some()) && mode != "hybrid" {
        return Err("--output and --verify require --mode hybrid".into());
    }
    if sweep {
        return sweep::run(shapes.unwrap_or_else(sweep::equal_witness_shapes), &mode, iterations, results_dir, profile.as_deref());
    }
    let mut parameters = Parameters::default();
    if let Some(log) = mul_log { parameters.multiplications = 1 << log; }
    if let Some(log) = sha_log { parameters.sha_compressions = 1 << log; }
    let binius = if mode == "separate" || mode == "all-binius" {
        Some(cli::environment::<BiniusConfig>())
    } else { None };
    let ligerito = match mode.as_str() {
        "hybrid" => select_ligerito(profile.as_deref(), 106)?,
        "separate" => select_ligerito(profile.as_deref(), 112)?,
        _ => bitz::ligerito_flock::LigeritoSelection::JOHNSON,
    };
    use tracing_subscriber::prelude::*;
    tracing_subscriber::registry()
        .with(observability::layer())
        .with(tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_target(false)
            .with_filter(tracing_subscriber::filter::LevelFilter::INFO))
        .try_init()?;
    if let Some(path) = verify_file {
        use bincode::Options;
        if std::fs::metadata(&path)?.len() > 64 << 20 {
            return Err("proof exceeds 64 MiB limit".into());
        }
        let statement_path = format!("{path}.statement.bin");
        if std::fs::metadata(&statement_path)?.len() > 1024 {
            return Err("statement exceeds 1 KiB limit".into());
        }
        let statement_bytes = std::fs::read(statement_path)?;
        let statement: bitz::hybrid::Statement = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(1024)
            .reject_trailing_bytes()
            .deserialize(&statement_bytes)?;
        let prepared = PreparedHybrid::new_with_ligerito(
            statement.parameters,
            ligerito,
        )?;
        let proof = prepared.proof_from_bytes(&statement, &std::fs::read(path)?)?;
        prepared.verify(&statement, &proof)?;
        println!(
            "verified {} multiplications modulo 2^32 (xy=z+2^32*w, four u32 limbs) and {} chained SHA-256 compressions",
            statement.parameters.multiplications, statement.parameters.sha_compressions
        );
        return Ok(());
    }
    let inputs: Vec<_> = (0..parameters.multiplications as u32)
        .map(|i| (i.wrapping_mul(0x9e3779b9), u32::MAX - i))
        .collect();
    let blocks: Vec<[u32; 16]> = (0..parameters.sha_compressions as u32)
        .map(|i| std::array::from_fn(|j| i.wrapping_mul(0x85ebca6b).wrapping_add(j as u32)))
        .collect();
    eprintln!(
        "mode={mode} multiplication_relation=u32_mod_2_32 multiplications={} chained_compressions={} merkle=blake3 non_zk=true threads={}",
        parameters.multiplications,
        parameters.sha_compressions,
        binius_utils::rayon::current_num_threads()
    );
    let setup_recording = Recording::start(Vec::new())?;
    let setup = tracing::info_span!("benchmark:setup").entered();
    if mode == "hybrid" {

        let prepared = PreparedHybrid::new_with_ligerito(
            parameters,
            ligerito,
        )?;
        let request = profile
            .clone()
            .unwrap_or_else(|| "custom:1:4".into());
        let report = prepared
            .ligerito_configuration()
            .report(&request, prepared.ood_round());
        eprintln!("LIGERITO_CONFIG {report}");
        if let Some(path) = &output {
            BenchmarkOutput::new("").write_json(
                format!("{path}.ligerito.json"),
                &report,
                FileMode::Replace,
                JsonStyle::Pretty,
            )?;
        }
        drop(setup);
        let setup_ms = millis(&setup_recording.intervals()?, "benchmark:setup");
        let binding = prepared
            .security()
            .binding_term()
            .map(|term| format!("{}:{:.2}", term.name, -term.error_bound.log2()))
            .unwrap_or_default();
        eprintln!(
            "setup_ms={setup_ms:.3} packed_logs={:?} algebraic_security_bits={:.3} ligerito_component_bits={} log_inv_rate={} ood_grinding_bits={} binding_term={binding}",
            prepared.packed_witness_logs(),
            prepared.security().algebraic_bits,
            prepared.ligerito_configuration().security().target_security_bits,
            prepared.log_inv_rate(),
            prepared.ood_round().map_or(0, |p| p.grinding_bits)
        );
        let mut csv = output::csv_writer(std::io::stdout().lock());
        csv.write_record(HybridRow::HEADER)?;
        csv.flush()?;
        for iteration in 0..=iterations {
            let recording = Recording::start(Vec::new())?;
            let start = tracing::info_span!("benchmark:proving").entered();
            let witness_commit = tracing::info_span!("benchmark:witness_commit").entered();
            let witness = tracing::info_span!("benchmark:witness").entered();
            let rows: Vec<_> = inputs
                .iter()
                .map(|&(x, y)| MulRow::<u32>::new(x, y))
                .collect();
            // Hybrid fuses assignment synthesis into commit_mod32, so this is
            // the native row construction only; witness_commit_ms below is
            // that plus the commitment.
            drop(witness);
            let committed = prepared.commit_mod32(&rows, &blocks)?;
            drop(witness_commit);
            let continuation = tracing::info_span!("benchmark:continuation").entered();
            let proof = prepared.prove(&committed)?;
            drop(continuation);
            drop(start);
            let bytes = proof.to_bytes();
            let verify = tracing::info_span!("benchmark:verification").entered();
            let decoded = prepared.proof_from_bytes(committed.statement(), &bytes)?;
            prepared.verify(committed.statement(), &decoded)?;
            drop(verify);
            let peak_rss_kib = peak_kib(); // Snapshot before trace extraction/processing.
            let intervals = recording.intervals()?;
            let witness_ms = millis(&intervals, "benchmark:witness");
            let witness_commit_ms = millis(&intervals, "benchmark:witness_commit");
            let continuation_ms = millis(&intervals, "benchmark:continuation");
            let total_ms = millis(&intervals, "benchmark:proving");
            let verify_ms = millis(&intervals, "benchmark:verification");
            let phases = observability::phase_totals(&intervals, "benchmark:proving")?;
            let phase_ms = |name: &str| -> Result<f64, AnyError> {
                phases
                    .iter()
                    .find(|(label, _)| *label == name)
                    .map(|(_, seconds)| seconds * 1000.0)
                    .ok_or_else(|| format!("missing hybrid prover timing: {name}").into())
            };
            // These are disjoint control-thread wall-clock scopes, including
            // their parallel work. Never add their nested profiling records.
            let mul_piop_ms = phase_ms("hybrid:mul_piop")?;
            let sha_piop_ms = phase_ms("hybrid:sha_piop")?;
            let mul_opening_ms = phase_ms("hybrid:mul_opening")?;
            let joint_sumcheck_ms = phase_ms("hybrid:joint_sumcheck")?;
            // Round 0 (the out-of-domain sample, run right after the
            // statement) is part of the shared opening protocol; it is
            // reported on its own and counted in shared_opening_ms.
            let ood_round_ms = phase_ms("hybrid:ood_round")?;
            let shared_opening_ms = phase_ms("hybrid:opening_iop")? + ood_round_ms;
            let piop_ms = mul_piop_ms + sha_piop_ms;
            let iop_ms = mul_opening_ms + joint_sumcheck_ms + shared_opening_ms;
            if iteration == 0 {
                continue;
            }
            csv.serialize(HybridRow {
                mode: "hybrid",
                iteration: iteration - 1,
                setup_ms,
                witness_ms,
                witness_commit_ms,
                continuation_ms,
                total_prover_ms: total_ms,
                verify_ms,
                proof_bytes: bytes.len(),
                peak_rss_kib,
                piop_ms,
                iop_ms,
                mul_piop_ms,
                sha_piop_ms,
                mul_opening_ms,
                joint_sumcheck_ms,
                shared_opening_ms,
                ood_round_ms,
            })?;
            csv.flush()?;
            if let Some(path) = &output {
                let artifacts = BenchmarkOutput::new("");
                artifacts.write_bytes(path, &bytes, FileMode::Replace)?;
                artifacts.write_bytes(
                    format!("{path}.statement.bin"),
                    &bincode::serialize(committed.statement())?,
                    FileMode::Replace,
                )?;
                let statement = format!(
                    "protocol=hybrid-u32-mod32-sha256-v5\nmultiplication_relation=xy=z+2^32*w (x,y,z,w are u32)\nparameters={:?}\nroots={:02x?}\nfinal_sha_state={:08x?}\n",
                    committed.statement().parameters,
                    committed.statement().roots,
                    committed.statement().final_sha_state
                );
                artifacts.write_text(
                    format!("{path}.statement.txt"),
                    &statement,
                    FileMode::Replace,
                )?;
            }
        }
    } else {
        let native = Native::new(
            if mode == "separate" { 0 } else { inputs.len() },
            blocks.len(),
            mode == "binius-ligerito",
            binius,
        )?;
        let separate = if mode == "separate" {
            Some(
                PreparedRelation::<MulLayout<u32>>::new_with_profile_and_ligerito::<
                    CompositionProfile,
                >(MulLayout::<u32>::new(inputs.len())?, ligerito)?,
            )
        } else {
            None
        };
        if let Some(p) = &separate {
            let request = profile
                .clone()
                    .unwrap_or_else(|| "custom:1:4".into());
            eprintln!(
                "LIGERITO_CONFIG {}",
                p.ligerito_configuration()
                    .report(&request, p.security().ood)
            );
        }
        drop(setup);
        let setup_ms = millis(&setup_recording.intervals()?, "benchmark:setup");
        if let NativeBackend::Ligerito(prepared) = &native.backend {
            let identity = report::BiniusLigeritoIdentity::new(prepared)?;
            eprintln!("LIGERITO_CONFIG {}", serde_json::to_string(&identity)?);
        }
        eprintln!("setup_ms={setup_ms:.3} {}", native.setup_line());
        let mut csv = output::csv_writer(std::io::stdout().lock());
        csv.write_record(NativeRow::header(&mode))?;
        csv.flush()?;
        for iteration in 0..=iterations {
            let recording = Recording::start(Vec::new())?;
            let start = tracing::info_span!("benchmark:proving").entered();
            let witness_scope = tracing::info_span!("benchmark:witness").entered();
            let rows: Vec<_> = inputs
                .iter()
                .map(|&(x, y)| MulRow::<u32>::new(x, y))
                .collect();
            let witness = native.populate(if mode == "separate" { &[] } else { &rows }, &blocks)?;
            // Native witness generation: row construction plus the circuit's
            // own witness filling, before any proving work.
            drop(witness_scope);
            let bytes = native.prove(&witness)?;
            let mul = if let Some(relation) = &separate {
                let witness = MulWitness::<u32>::from_rows(&rows)?;
                let hint = protocol::commit(relation, witness.bitz_bit_rows())?;
                let proof =
                    protocol::prove(&mut Blake3Transcript::new(), relation, &witness, &hint)?;
                Some((hint, proof))
            } else {
                None
            };
            drop(start);
            // The standalone u32 API has no enclosing wire codec. Its size
            // here is the exact BitZ wire section plus raw Spartan/nonces;
            // report this as a payload estimate rather than invent framing.
            let mut proof_bytes = bytes.len();
            if let Some((hint, proof)) = &mul {
                let relation = separate.as_ref().expect("separate mode");
                proof_bytes += hint.commitment.root.len()
                    + proof.bitz().to_bytes().len()
                    + proof.spartan_payload_elements() * 16
                    + (proof.grinding_nonce_count(relation.security())
                        - proof.opening_grinding_nonces().len())
                        * 8;
            }
            let verify = tracing::info_span!("benchmark:verification").entered();
            native.verify(&witness, bytes)?;
            if let Some((hint, proof)) = &mul {
                protocol::verify(
                    &mut Blake3Transcript::new(),
                    separate.as_ref().expect("separate mode"),
                    &hint.commitment,
                    proof,
                )?;
            }
            drop(verify);
            let peak_rss_kib = peak_kib();
            let intervals = recording.intervals()?;
            let witness_ms = millis(&intervals, "benchmark:witness");
            let total_ms = millis(&intervals, "benchmark:proving");
            let verify_ms = millis(&intervals, "benchmark:verification");
            if iteration == 0 {
                continue;
            }
            csv.serialize(NativeRow {
                mode: &mode,
                iteration: iteration - 1,
                setup_ms,
                witness_ms,
                total_prover_ms: total_ms,
                verify_ms,
                proof_bytes,
                peak_rss_kib,
            })?;
            csv.flush()?;
        }
        if mode == "separate" {
            eprintln!(
                "separate proof_bytes is a payload estimate (no enclosing standalone-u32 framing)"
            );
        }
    }
    Ok(())
}


#[cfg(test)]
mod cli_tests {
    use super::{Args, BiniusConfig};
    use clap::{CommandFactory, FromArgMatches, Parser, error::ErrorKind};

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        let matches = Args::command().mut_args(|arg| arg.env(None::<&str>))
            .try_get_matches_from(std::iter::once("hybrid").chain(args.iter().copied()))?;
        Args::from_arg_matches(&matches)
    }

    #[test]
    fn defaults_and_script_forms() {
        Args::command().debug_assert();
        BiniusConfig::command().debug_assert();
        let defaults = parse(&[]).unwrap();
        assert_eq!((defaults.mode.as_str(), defaults.iterations), ("hybrid", 5));
        assert!(
            defaults.profile.is_none() && defaults.mul_log.is_none() && defaults.sha_log.is_none()
        );
        let sweep = parse(&[
            "--sweep",
            "--mode",
            "all",
            "--shapes",
            "15:7,16:8",
            "--iterations",
            "3",
            "--results-dir",
            "campaign",
            "--bench",
        ])
        .unwrap();
        assert!(sweep.sweep);
        assert_eq!((sweep.mode.as_str(), sweep.iterations), ("all", 3));
        assert_eq!(
            sweep.shapes.unwrap(),
            super::sweep::parse_shapes("15:7,16:8").unwrap()
        );
        assert_eq!(
            sweep.results_dir.as_deref(),
            Some(std::path::Path::new("campaign"))
        );
        assert_eq!(
            parse(&["--verify", "proof"]).unwrap().verify.as_deref(),
            Some("proof")
        );
        let single = parse(&[
            "--mul-log",
            "9",
            "--sha-log",
            "1",
            "--mul-log",
            "22",
            "--sha-log",
            "16",
            "--iterations",
            "1",
            "--iterations",
            "2",
        ])
        .unwrap();
        assert_eq!(
            (single.mul_log, single.sha_log, single.iterations),
            (Some(22), Some(16), 2)
        );
    }

    #[test]
    fn rejects_invalid_values_and_conflicting_options() {
        for args in [
            &["--iterations", "0"][..], &["--mul-log", "8"], &["--mul-log", "23"],
            &["--sha-log", "0"], &["--sha-log", "17"], &["--mode", "unknown"],
            &["--iterations"], &["--unknown"], &["--shapes", "15:7"], &["--mode", "all"],
            &["--sweep", "--shapes", "8:7"], &["--sweep", "--shapes", "15:7,15:7"],
            &["--results-dir", "campaign"], &["--output", "proof", "--verify", "proof"],
            &["--sweep", "--mul-log", "15"], &["--sweep", "--sha-log", "7"],
            &["--sweep", "--output", "proof"], &["--sweep", "--verify", "proof"],
        ] {
            assert!(parse(args).is_err(), "accepted {args:?}");
        }
        assert_eq!(parse(&["--help"]).err().unwrap().kind(), ErrorKind::DisplayHelp);
    }

    #[test]
    fn profile_environment_probe() {
        let Ok(mode) = std::env::var("BITZ_CLI_PROFILE_PROBE") else {
            return;
        };
        let argv = if mode == "override" {
            vec!["hybrid", "--profile", "custom:1:4"]
        } else {
            vec!["hybrid"]
        };
        let args = Args::try_parse_from(argv).unwrap();
        println!("CLI_PROFILE {}", args.profile.as_deref().unwrap());
    }

    #[test]
    fn profile_environment_fallback_and_cli_override() {
        let test = concat!(module_path!(), "::profile_environment_probe");
        let test = test.split_once("::").unwrap().1;
        for (mode, expected) in [("fallback", "udrg:1:4"), ("override", "custom:1:4")] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", test, "--nocapture"]).env_clear()
                .env("BITZ_CLI_PROFILE_PROBE", mode).env("BITZ_LIG_PROFILE", "udrg:1:4")
                .output().unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("CLI_PROFILE {expected}")));
        }
    }
}
