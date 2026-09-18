//! Isolated non-ZK Binius64 worker for the common SHA-chain/P-256 relation.
//!
//! Two openers over the identical circuit and witness: Binius64's own ring
//! switch + BaseFold (`--method binius64`, the upstream prover), and the BitZ
//! opener (`--method binius64-ligerito` / `--opener bitz`): the same PIOP up to
//! the witness evaluation claim, every oracle committed and opened by BitZ's
//! binary PCS under the round-by-round 100-bit gate
//! (`bitz::binius_ligerito::Prepared`).
#![recursion_limit = "256"]

#[path = "../../../benches/support/sha256_ecdsa_fixture.rs"]
mod fixture;
use bitz::observability;
#[path = "../../../benches/common/trace_capture.rs"]
mod trace_capture;

use binius_circuits::sha256_ecdsa::{PROFILE, Sha256Ecdsa, public_words};
use binius_frontend::{Circuit, CircuitBuilder};
use binius_hash::sha256::Sha256HashSuite;
use binius_prover::{OptimalPackedB128, Prover};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{Verifier, config::StdChallenger};
use bitz::binius_ligerito::{Accounting, Prepared};
use fixture::{Result, SignedFixture};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const PHASES: [&str; 4] = [
    "Prepare witness",
    "Commit witness",
    "[phase] PCS Opening",
    "[phase] Finish PCS",
];

/// Keep the worker's existing human-readable report keys. The dependency also
/// exports component IDs, which the shared collector correctly prefers as labels.
fn phase_timings(intervals: &[observability::Interval]) -> Result<BTreeMap<String, f64>> {
    let prove = observability::span(intervals, "worker:prove")?;
    let phases: Vec<_> = intervals
        .iter()
        .filter(|s| {
            PHASES.contains(&s.name.as_str())
                && s.start_ns >= prove.start_ns
                && s.end_ns <= prove.end_ns
        })
        .cloned()
        .map(|mut span| {
            span.component = None;
            span
        })
        .collect();
    Ok(observability::totals(&phases)
        .into_iter()
        .map(|(name, seconds)| (name, seconds * 1000.))
        .collect())
}

struct Args {
    /// `binius64` (ring switch + BaseFold) or `binius64-ligerito` (BitZ opener).
    method: String,
    exponent: u8,
    threads: usize,
    target: usize,
    /// Level-0 inverse-rate exponent for either opener: 1 = rate 1/2, 3 = 1/8.
    log_inv_rate: usize,
    reps: usize,
    seed: u64,
    fixture: Option<PathBuf>,
    self_test: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let (mut r, mut c) = (None::<u8>, None::<u8>);
        let mut opener: Option<String> = None;
        let mut out = Self {
            method: "binius64".into(),
            exponent: 0,
            threads: 1,
            target: 100,
            log_inv_rate: 1,
            reps: 3,
            seed: 0,
            fixture: None,
            self_test: false,
        };
        while let Some(flag) = args.next() {
            if flag == "--self-test" {
                out.self_test = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--method" if ["binius64", "binius64-ligerito"].contains(&value.as_str()) => {
                    out.method = value;
                }
                "--opener" if ["basefold", "bitz"].contains(&value.as_str()) => {
                    opener = Some(value);
                }
                "--r" => r = Some(value.parse()?),
                "--c" => c = Some(value.parse()?),
                "--target" => out.target = value.parse()?,
                "--log-inv-rate" => out.log_inv_rate = value.parse()?,
                "--threads" => out.threads = value.parse()?,
                "--reps" => out.reps = value.parse()?,
                "--seed" => out.seed = value.parse()?,
                "--fixture" => out.fixture = Some(value.into()),
                _ => return Err(format!("unknown argument {flag} {value}").into()),
            }
        }
        if let Some(opener) = opener {
            let implied = if opener == "bitz" {
                "binius64-ligerito"
            } else {
                "binius64"
            };
            if out.method != implied && std::env::args().any(|a| a == "--method") {
                return Err("--opener disagrees with --method".into());
            }
            out.method = implied.into();
        }
        out.exponent = r
            .ok_or("--r required")?
            .checked_add(c.ok_or("--c required")?)
            .ok_or("exponent overflow")?;
        binius_circuits::sha256_ecdsa::message_len(out.exponent)?;
        if ![100, 128].contains(&out.target) || out.threads == 0 || out.reps == 0 {
            return Err("require target 100/128 and positive threads/reps".into());
        }
        if !(1..=3).contains(&out.log_inv_rate) {
            return Err("--log-inv-rate must be 1, 2, or 3".into());
        }
        if out.method == "binius64-ligerito" && out.target != 100 {
            return Err("the BitZ opener's whole-protocol gate is fixed at 100 bits".into());
        }
        Ok(out)
    }
}

fn build_info() -> serde_json::Value {
    json!({"binius_revision":env!("BINIUS_REVISION"), "lock_sha256":env!("LOCK_SHA256"),
        "source_sha256":env!("SOURCE_SHA256"), "rustc":env!("BUILD_RUSTC"), "rustflags":env!("BUILD_RUSTFLAGS"),
        "bitz_revision":env!("BITZ_REVISION"), "bitz_dirty":env!("BITZ_DIRTY") == "dirty",
        "circuit_profile":PROFILE, "fixture_profile":fixture::SCHEMA, "zk":false})
}

fn expected_words(fixture: &SignedFixture) -> Vec<binius_core::word::Word> {
    public_words(
        fixture.log_compressions,
        &fixture.qx,
        &fixture.qy,
        &fixture.r,
        &fixture.s,
    )
}

fn verify(
    verifier: &Verifier<Sha256HashSuite>,
    fixture: &SignedFixture,
    proof: Vec<u8>,
) -> Result<()> {
    fixture.validate_statement()?;
    let expected = expected_words(fixture);
    let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
    verifier.verify(&expected, &mut transcript)?;
    transcript.finalize()?;
    Ok(())
}

/// Statement validation + decode + BitZ-opener verification, the counterpart
/// of [`verify`] for `binius64-ligerito` rows.
fn verify_bitz(prepared: &Prepared, fixture: &SignedFixture, proof: &[u8]) -> Result<()> {
    fixture.validate_statement()?;
    let expected = expected_words(fixture);
    let decoded = prepared.proof_from_bytes(proof)?;
    prepared.verify(&expected, &decoded)?;
    Ok(())
}

fn base_row(args: &Args, fixture: &SignedFixture, trial: usize, circuit_id: &str) -> serde_json::Value {
    json!({
        "schema":"bitz/sha256-ecdsa-compare/v1", "method":args.method, "zk":false,
        "trial":if trial == 0 {"warmup"} else {"sample"}, "sample":trial,
        "log_compressions":args.exponent, "compressions":1usize << args.exponent, "message_bytes":fixture.message.len(),
        "signatures":1, "r":null, "c":null, "security_target":args.target, "log_inv_rate":args.log_inv_rate,
        "threads":args.threads, "seed":args.seed,
        "fixture_id":fixture.id, "fixture_profile":fixture::SCHEMA, "statement_bytes":129,
        "statement":"public-key-signature; witness-message", "binius_revision":env!("BINIUS_REVISION"),
        "bitz_revision":env!("BITZ_REVISION"),
        "circuit_profile":PROFILE, "circuit_id":circuit_id, "verified":true,
    })
}

fn circuit_json(circuit: &Circuit) -> serde_json::Value {
    json!({"gates":circuit.n_gates(), "bitand":circuit.constraint_system().and_constraints.len(),
        "intmul":circuit.constraint_system().imul_constraints.len(), "public_words":17})
}

/// The upstream prover: ring switch + BaseFold at `--log-inv-rate`.
fn run_basefold(args: &Args, fixture: &SignedFixture) -> Result<()> {
    let recording = observability::Recording::start(Vec::new())?;
    let setup = tracing::info_span!("worker:setup").entered();
    let builder = CircuitBuilder::new();
    let relation = Sha256Ecdsa::new(&builder, args.exponent)?;
    let circuit = builder.build();
    let verifier = Verifier::<Sha256HashSuite>::setup_with_security_bits(
        circuit.constraint_system().clone(),
        args.log_inv_rate,
        args.target,
    )?;
    let prover = Prover::<OptimalPackedB128, Sha256HashSuite>::setup(verifier.clone())?;
    drop(setup);
    let setup_ms =
        observability::duration(&recording.intervals()?, "worker:setup")?.as_secs_f64() * 1000.;
    let circuit_id =
        blake3::hash(format!("{PROFILE}:{}:{}", args.exponent, env!("SOURCE_SHA256")).as_bytes())
            .to_hex()
            .to_string();
    for trial in 0..=args.reps {
        let recording = observability::Recording::start(Vec::new())?;
        let e2e = tracing::info_span!("worker:e2e", trial, warmup = trial == 0).entered();
        let witness_start = tracing::info_span!("worker:assignment").entered();
        let mut filler = circuit.new_witness_filler();
        relation.populate(
            &mut filler,
            &fixture.message,
            &fixture.qx,
            &fixture.qy,
            &fixture.r,
            &fixture.s,
        )?;
        circuit.populate_wire_witness(&mut filler)?;
        let witness = filler.into_value_vec();
        drop(witness_start);
        let start = tracing::info_span!("worker:prove").entered();
        let mut transcript = ProverTranscript::new(StdChallenger::default());
        prover.prove(&witness, &mut transcript)?;
        let bytes = transcript.finalize();
        drop(start);
        drop(e2e);
        // Transcript bytes already are the proof wire format. Copying accounts for transport decoding.
        let codec = tracing::info_span!("worker:codec").entered();
        let proof_bytes = bytes.len();
        let decoded = bytes.clone();
        drop(codec);
        drop(witness);
        tracing::info_span!("worker:verification")
            .in_scope(|| verify(&verifier, &fixture, decoded))?;
        let intervals = recording.intervals()?;
        let millis =
            |name| observability::duration(&intervals, name).map(|d| d.as_secs_f64() * 1000.);
        let assignment_ms = millis("worker:assignment")?;
        let prove_call_ms = millis("worker:prove")?;
        let e2e_prover_ms = millis("worker:e2e")?;
        let codec_ms = millis("worker:codec")?;
        let verify_ms = millis("worker:verification")?;
        let phases = phase_timings(&intervals)?;
        let phase = |key: &str| {
            phases
                .get(key)
                .copied()
                .ok_or_else(|| format!("missing timing phase {key}"))
        };
        let packing = phase(PHASES[0])?;
        let commit_ms = phase(PHASES[1])?;
        let opening_ms = phase(PHASES[2])? + phase(PHASES[3])?;
        let witness_ms = assignment_ms + packing;
        let protocol_ms = prove_call_ms - packing - commit_ms;
        if !protocol_ms.is_finite() || protocol_ms < opening_ms {
            return Err("overlapping or invalid phase measurements".into());
        }
        if args.self_test && trial == 0 {
            for which in 0..5 {
                let mut other = fixture.clone();
                match which {
                    0 => other.qx[31] ^= 1,
                    1 => other.qy[31] ^= 1,
                    2 => other.r[31] ^= 1,
                    3 => other.s[31] ^= 1,
                    _ => other.log_compressions += 1,
                }
                if verify(&verifier, &other, bytes.clone()).is_ok() {
                    return Err("accepted altered public statement".into());
                }
            }
            let mut corrupt = bytes.clone();
            corrupt[0] ^= 1;
            if verify(&verifier, &fixture, corrupt).is_ok() {
                return Err("accepted corrupt proof".into());
            }
            let mut extra = bytes.clone();
            extra.push(0);
            if verify(&verifier, &fixture, extra).is_ok() {
                return Err("accepted trailing proof byte".into());
            }
            if verify(&verifier, &fixture, bytes[..bytes.len() - 1].to_vec()).is_ok() {
                return Err("accepted truncated proof".into());
            }
        }
        let mut row = base_row(args, fixture, trial, &circuit_id);
        row.as_object_mut().unwrap().extend(
            json!({
                "setup_ms":setup_ms, "witness_ms":witness_ms, "commit_ms":commit_ms, "protocol_ms":protocol_ms,
                "prove_ms":commit_ms+protocol_ms, "witness_to_proof_ms":witness_ms+commit_ms+protocol_ms, "e2e_prover_ms":e2e_prover_ms,
                "verify_ms":verify_ms, "codec_ms":codec_ms, "opening_ms":opening_ms,
                "outer_ms":null, "inner_ms":null, "folding_ms":null,
                "proof_object_bytes":proof_bytes, "proof_material_bytes":proof_bytes, "phases_ms":phases,
                "security":{"model":"Binius IOP and BaseFold FRI; query target only", "pcs":"BaseFold", "fri_query_target_bits":args.target,
                "fri_queries":verifier.fri_params().n_test_queries(), "log_inv_rate":verifier.fri_params().rs_code().log_inv_rate(), "merkle_hash":"SHA-256", "transcript":"StdChallenger",
                "fri_fold_arities":verifier.fri_params().fold_arities(), "fri_log_message_len":verifier.fri_params().log_msg_len(),
                "fri_final_challenges":verifier.fri_params().n_final_challenges(), "extension_field_bits":128,
                    "statistical_bits_lower_bound":null},
                "circuit":circuit_json(&circuit)
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        println!("{row}");
    }
    Ok(())
}

/// The BitZ opener over the same circuit and witness: Binius64's PIOP up to the
/// witness evaluation claim, every oracle committed and opened by BitZ's binary
/// PCS, gated at 100 bits round-by-round. Witness packing happens inside the
/// prover here, so `witness_ms` is the wire assignment alone.
fn run_bitz_opener(args: &Args, fixture: &SignedFixture) -> Result<()> {
    let recording = observability::Recording::start(Vec::new())?;
    let setup = tracing::info_span!("worker:setup").entered();
    let builder = CircuitBuilder::new();
    let relation = Sha256Ecdsa::new(&builder, args.exponent)?;
    let circuit = builder.build();
    let prepared = Prepared::with_options(
        circuit.constraint_system(),
        args.log_inv_rate,
        Accounting::RoundByRound,
    )?;
    drop(setup);
    let setup_ms = observability::duration(&recording.intervals()?, "worker:setup")?.as_secs_f64() * 1000.;
    let security = prepared.security().clone();
    if security.algebraic_bits < f64::from(security.target_bits) {
        return Err("BitZ opener gate unsatisfied".into());
    }
    let expected = expected_words(fixture);
    let circuit_id =
        blake3::hash(format!("{PROFILE}:{}:{}", args.exponent, env!("SOURCE_SHA256")).as_bytes())
            .to_hex()
            .to_string();
    for trial in 0..=args.reps {
        let recording = observability::Recording::start(Vec::new())?;
        let e2e = tracing::info_span!("worker:e2e").entered();
        let witness_start = tracing::info_span!("worker:assignment").entered();
        let mut filler = circuit.new_witness_filler();
        relation.populate(
            &mut filler,
            &fixture.message,
            &fixture.qx,
            &fixture.qy,
            &fixture.r,
            &fixture.s,
        )?;
        circuit.populate_wire_witness(&mut filler)?;
        let witness = filler.into_value_vec();
        drop(witness_start);
        let proving = tracing::info_span!("worker:prove").entered();
        let proof = prepared.prove(&witness)?;
        drop(proving);
        drop(e2e);
        if witness
            .inout()
            .iter()
            .map(|w| w.0)
            .ne(expected.iter().map(|w| w.0))
        {
            return Err("witness public words disagree with the statement".into());
        }
        let codec = tracing::info_span!("worker:codec").entered();
        let bytes = proof.to_bytes();
        let proof_bytes = bytes.len();
        let decoded = prepared.proof_from_bytes(&bytes)?;
        drop(codec);
        drop(witness);
        let verification = tracing::info_span!("worker:verification").entered();
        fixture.validate_statement()?;
        prepared.verify(&expected, &decoded)?;
        drop(verification);
        let intervals = recording.intervals()?;
        let ms = |label| observability::duration(&intervals, label).map(|d| d.as_secs_f64() * 1000.);
        let witness_ms = ms("worker:assignment")?;
        let prove_call_ms = ms("worker:prove")?;
        let e2e_prover_ms = ms("worker:e2e")?;
        let codec_ms = ms("worker:codec")?;
        let verify_ms = ms("worker:verification")?;
        let phases = trace_capture::BiniusLigeritoPhases::from_spans(&intervals);
        let commit_ms = (phases.commit.1 - phases.commit.0) as f64 / 1e6;
        let union_ms = |ranges: &[(u64, u64)]| {
            let mut end = 0;
            let mut total = 0;
            for &(start, stop) in ranges {
                total += stop.saturating_sub(start.max(end));
                end = end.max(stop);
            }
            total as f64 / 1e6
        };
        let piop_ms = union_ms(&phases.piop);
        let opening_ms = union_ms(&phases.opening);
        let protocol_ms = prove_call_ms - commit_ms;
        if !protocol_ms.is_finite() || protocol_ms < opening_ms {
            return Err("overlapping or invalid phase measurements".into());
        }
        if args.self_test && trial == 0 {
            for which in 0..5 {
                let mut other = fixture.clone();
                match which {
                    0 => other.qx[31] ^= 1,
                    1 => other.qy[31] ^= 1,
                    2 => other.r[31] ^= 1,
                    3 => other.s[31] ^= 1,
                    _ => other.log_compressions += 1,
                }
                if verify_bitz(&prepared, &other, &bytes).is_ok() {
                    return Err("accepted altered public statement".into());
                }
            }
            let mut corrupt = bytes.clone();
            corrupt[0] ^= 1;
            if verify_bitz(&prepared, &fixture, &corrupt).is_ok() {
                return Err("accepted corrupt proof".into());
            }
            let mut extra = bytes.clone();
            extra.push(0);
            if verify_bitz(&prepared, &fixture, &extra).is_ok() {
                return Err("accepted trailing proof byte".into());
            }
            if verify_bitz(&prepared, &fixture, &bytes[..bytes.len() - 1]).is_ok() {
                return Err("accepted truncated proof".into());
            }
        }
        let mut row = base_row(args, fixture, trial, &circuit_id);
        row.as_object_mut().unwrap().extend(
            json!({
                "setup_ms":setup_ms, "witness_ms":witness_ms, "commit_ms":commit_ms, "protocol_ms":protocol_ms,
                "prove_ms":commit_ms+protocol_ms, "witness_to_proof_ms":witness_ms+commit_ms+protocol_ms, "e2e_prover_ms":e2e_prover_ms,
                "verify_ms":verify_ms, "codec_ms":codec_ms, "opening_ms":opening_ms,
                "outer_ms":null, "inner_ms":null, "folding_ms":null,
                "proof_object_bytes":proof_bytes, "proof_material_bytes":proof_bytes,
                "phases_ms":{"Commit oracles":commit_ms, "PIOP prefix":piop_ms, "Opening":opening_ms},
                "security":{"model":"Binius64 PIOP with the BitZ opener; whole-protocol algebraic accounting",
                    "pcs":"BitZ-Ligerito", "accounting":security.accounting.name(),
                    "target_bits":security.target_bits, "algebraic_bits":security.algebraic_bits,
                    "union_bound_bits":security.union_bound_bits, "round_by_round_bits":security.round_by_round_bits,
                    "component_bits":prepared.component_bits(), "log_inv_rate":args.log_inv_rate,
                    "oracles":prepared.oracle_specs().len(),
                    "binding_term":security.binding_term().map(|t| t.name),
                    "terms":security.terms.iter().map(|t| json!({"name":t.name, "bits":-t.error_bound.log2()})).collect::<Vec<_>>(),
                    "merkle_hash":"BLAKE3", "transcript":"BLAKE3", "extension_field_bits":128,
                    "statistical_bits_lower_bound":null},
                "circuit":circuit_json(&circuit)
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        println!("{row}");
    }
    Ok(())
}

fn main() -> Result<()> {
    if matches!(std::env::args().nth(1).as_deref(), Some("--help" | "-h")) {
        println!(
            "binius64-sha256-ecdsa --r R --c C [--method binius64|binius64-ligerito] [--opener basefold|bitz]\n\
            \x20   [--target 100|128] [--log-inv-rate 1|2|3] [--threads N] [--reps N] [--seed N] [--fixture PATH] [--self-test]\n\
            Standard P-256, non-ZK; 3 <= R+C <= 16. --log-inv-rate selects rate 1/2 (1), 1/4 (2), or 1/8 (3) for either opener.\n\
            --opener bitz (= --method binius64-ligerito) proves through the BitZ opener, round-by-round 100-bit gate.\n\
            --build-info prints pinned source/build metadata."
        );
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("--build-info") {
        println!("{}", build_info());
        return Ok(());
    }
    let args = Args::parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    observability::install()?;
    let fixture = match &args.fixture {
        Some(path) => SignedFixture::read(path)?,
        None => SignedFixture::generate(args.exponent, args.seed)?,
    };
    if fixture.log_compressions != args.exponent || fixture.seed != args.seed {
        return Err("fixture configuration mismatch".into());
    }
    if args.method == "binius64-ligerito" {
        run_bitz_opener(&args, &fixture)
    } else {
        run_basefold(&args, &fixture)
    }
}

#[cfg(test)]
#[path = "../../../benches/support/sha256_ecdsa_test_vectors.rs"]
mod vectors;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_keys_preserve_names_and_union_repeated_component_spans() {
        let span = |id, name: &str, component: &str, start_ns, end_ns| observability::Interval {
            id,
            parent: None,
            track_id: id,
            depth: 0,
            name: name.into(),
            component: Some(component.into()),
            start_ns,
            end_ns,
        };
        let intervals = vec![
            span(0, "worker:prove", "worker:prove", 100, 1000),
            span(1, PHASES[0], "prepare_witness", 110, 210),
            span(2, PHASES[0], "prepare_witness", 160, 260),
            span(3, PHASES[1], "commit_witness", 260, 360),
            span(4, PHASES[2], "ring_switching", 600, 800),
            span(5, PHASES[3], "finish_pcs", 800, 900),
            span(6, PHASES[0], "prepare_witness", 1100, 1200),
        ];
        let phases = phase_timings(&intervals).unwrap();
        assert_eq!(phases.len(), 4);
        assert!((phases[PHASES[0]] - 150. / 1e6).abs() < 1e-12);
        assert!((phases[PHASES[2]] - 200. / 1e6).abs() < 1e-12);
    }

    #[test]
    fn message_larger_than_u16_satisfies_the_circuit() {
        let fixture = SignedFixture::generate(11, 1).unwrap();
        assert_eq!(fixture.message.len(), 131_008);
        let builder = CircuitBuilder::new();
        let relation = Sha256Ecdsa::new(&builder, 11).unwrap();
        let circuit = builder.build();
        let mut filler = circuit.new_witness_filler();
        relation
            .populate(
                &mut filler,
                &fixture.message,
                &fixture.qx,
                &fixture.qy,
                &fixture.r,
                &fixture.s,
            )
            .unwrap();
        circuit.populate_wire_witness(&mut filler).unwrap();
        circuit
            .constraint_system()
            .verify(&filler.into_value_vec())
            .unwrap();
    }

    #[test]
    fn explicit_fri_targets_and_invalid_parameters() {
        let builder = CircuitBuilder::new();
        let input = builder.add_witness();
        let output = builder.add_inout();
        builder.assert_eq("product", builder.band(input, input), output);
        let cs = builder.build().constraint_system().clone();
        for (rate, target) in [(0, 100), (1, 0), (1, 129)] {
            assert!(
                Verifier::<Sha256HashSuite>::setup_with_security_bits(cs.clone(), rate, target)
                    .is_err()
            );
        }
        for rate in 1..=3 {
            for target in [100, 128] {
                let verifier =
                    Verifier::<Sha256HashSuite>::setup_with_security_bits(cs.clone(), rate, target)
                        .unwrap();
                assert_eq!(
                    verifier.fri_params().n_test_queries(),
                    binius_verifier::fri::calculate_n_test_queries(target, rate)
                );
            }
        }
        // The paper's second Binius rate: the query count follows the rate.
        let eighth = Verifier::<Sha256HashSuite>::setup_with_security_bits(cs.clone(), 3, 100)
            .unwrap();
        assert_eq!(
            eighth.fri_params().n_test_queries(),
            binius_verifier::fri::calculate_n_test_queries(100, 3)
        );
        let default = Verifier::<Sha256HashSuite>::setup(cs, 1).unwrap();
        assert_eq!(
            default.fri_params().n_test_queries(),
            binius_verifier::fri::calculate_n_test_queries(96, 1)
        );
    }

    #[test]
    fn changed_message_and_signature_fail_the_composed_circuit() {
        let fixture = SignedFixture::generate(3, 0).unwrap();
        let builder = CircuitBuilder::new();
        let relation = Sha256Ecdsa::new(&builder, 3).unwrap();
        let circuit = builder.build();
        for field in ["message", "r", "s"] {
            let mut changed = fixture.clone();
            match field {
                "message" => changed.message[0] ^= 1,
                "r" => changed.r[31] ^= 1,
                "s" => changed.s[31] ^= 1,
                _ => unreachable!(),
            }
            let mut filler = circuit.new_witness_filler();
            // Do not run fixture validation: the circuit must reject the
            // changed instance even when all arithmetic hints are recomputed.
            relation
                .populate(
                    &mut filler,
                    &changed.message,
                    &changed.qx,
                    &changed.qy,
                    &changed.r,
                    &changed.s,
                )
                .unwrap();
            if circuit.populate_wire_witness(&mut filler).is_ok() {
                assert!(
                    circuit
                        .constraint_system()
                        .verify(&filler.into_value_vec())
                        .is_err(),
                    "accepted changed {field}"
                );
            }
        }
        assert_eq!(binius_circuits::sha256_ecdsa::message_len(6).unwrap(), 4032);
        let builder = CircuitBuilder::new();
        let relation = Sha256Ecdsa::new(&builder, 6).unwrap();
        let circuit = builder.build();
        assert!(
            relation
                .populate(
                    &mut circuit.new_witness_filler(),
                    &vec![0; 4096],
                    &fixture.qx,
                    &fixture.qy,
                    &fixture.r,
                    &fixture.s,
                )
                .is_err()
        );
    }

    #[test]
    fn shared_standard_and_exceptional_fixtures_prove_and_bind_public_inputs() {
        let builder = CircuitBuilder::new();
        let relation = Sha256Ecdsa::new(&builder, 3).unwrap();
        let circuit = builder.build();
        let verifier = Verifier::<Sha256HashSuite>::setup_with_security_bits(
            circuit.constraint_system().clone(),
            1,
            100,
        )
        .unwrap();
        let prover = Prover::<OptimalPackedB128, Sha256HashSuite>::setup(verifier.clone()).unwrap();
        for fixture in vectors::vectors() {
            let mut filler = circuit.new_witness_filler();
            relation
                .populate(
                    &mut filler,
                    &fixture.message,
                    &fixture.qx,
                    &fixture.qy,
                    &fixture.r,
                    &fixture.s,
                )
                .unwrap();
            circuit.populate_wire_witness(&mut filler).unwrap();
            let witness = filler.into_value_vec();
            circuit.constraint_system().verify(&witness).unwrap();
            let mut transcript = ProverTranscript::new(StdChallenger::default());
            prover.prove(&witness, &mut transcript).unwrap();
            let bytes = transcript.finalize();
            drop(witness);
            verify(&verifier, &fixture, bytes.clone()).unwrap();
            // Bypass fixture validation to exercise the native verifier's
            // binding of every public word, including the exponent and key.
            let expected = public_words(
                fixture.log_compressions,
                &fixture.qx,
                &fixture.qy,
                &fixture.r,
                &fixture.s,
            );
            for index in 0..expected.len() {
                let mut changed = expected.clone();
                changed[index].0 ^= 1;
                let mut transcript =
                    VerifierTranscript::new(StdChallenger::default(), bytes.clone());
                assert!(
                    verifier.verify(&changed, &mut transcript).is_err(),
                    "accepted changed public word {index}"
                );
            }
            let mut other = fixture.clone();
            other.s = (-p256::ecdsa::Signature::from_scalars(other.r, other.s)
                .unwrap()
                .s()
                .as_ref())
            .to_bytes()
            .into();
            assert!(verify(&verifier, &other, bytes.clone()).is_err());
            other = fixture.clone();
            other.log_compressions = 4;
            assert!(verify(&verifier, &other, bytes.clone()).is_err());
            let mut extra = bytes.clone();
            extra.push(0);
            assert!(verify(&verifier, &fixture, extra).is_err());
            assert!(verify(&verifier, &fixture, bytes[..bytes.len() - 1].to_vec()).is_err());
        }
    }

    /// The standard P-256 gadget lowers its select gates to BMUL constraints
    /// (fork `frontend/src/gates/select.rs`; 45k of them at exponent 3), so
    /// this round trip exercises the adapter's full IntMul + BinMul + AND
    /// reduction mix, and its security report must carry the BinMul term.
    #[test]
    fn bitz_opener_round_trips_and_binds_the_statement() {
        let fixture = SignedFixture::generate(3, 7).unwrap();
        let builder = CircuitBuilder::new();
        let relation = Sha256Ecdsa::new(&builder, 3).unwrap();
        let circuit = builder.build();
        assert!(!circuit.constraint_system().bmul_constraints.is_empty());
        let prepared =
            Prepared::with_options(circuit.constraint_system(), 1, Accounting::RoundByRound)
                .unwrap();
        assert!(prepared.security().round_by_round_bits >= 100.0);
        assert!(
            prepared
                .security()
                .terms
                .iter()
                .any(|t| t.name.contains("BinMul")),
            "missing BinMul security term"
        );
        let mut filler = circuit.new_witness_filler();
        relation
            .populate(
                &mut filler,
                &fixture.message,
                &fixture.qx,
                &fixture.qy,
                &fixture.r,
                &fixture.s,
            )
            .unwrap();
        circuit.populate_wire_witness(&mut filler).unwrap();
        let witness = filler.into_value_vec();
        let proof = prepared.prove(&witness).unwrap();
        let bytes = proof.to_bytes();
        drop(witness);
        verify_bitz(&prepared, &fixture, &bytes).unwrap();
        let mut other = fixture.clone();
        other.qx[31] ^= 1;
        assert!(verify_bitz(&prepared, &other, &bytes).is_err());
        assert!(verify_bitz(&prepared, &fixture, &bytes[..bytes.len() - 1]).is_err());
        let mut corrupt = bytes;
        corrupt[12] ^= 1;
        assert!(verify_bitz(&prepared, &fixture, &corrupt).is_err());
    }
}
