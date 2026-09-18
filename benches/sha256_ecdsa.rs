//! One worker process per (compression count, outer mode, security, threads).
//! Signing is fixture preparation; inversion hints belong to timed witness generation.
mod common;
#[cfg(feature = "bench-peak-memory")]
#[global_allocator]
static HEAP_ALLOCATOR: common::peak_memory::PeakAlloc = common::peak_memory::PeakAlloc;


use bitz::{piop::spartan::ecdsa_sha256::*, transcript::Blake3Transcript};
use p256::ecdsa::{
    Signature, SigningKey,
    signature::{Signer, Verifier},
};
use serde_json::json;
use std::error::Error;

#[derive(clap::Parser)]
struct Args {
    #[command(flatten)]
    cargo: common::cli::CargoArgs,
    #[arg(value_parser = clap::value_parser!(u32).range(3..=16))]
    exponent: u32,
    #[arg(value_parser = ["split", "all"])]
    mode: String,
    #[arg(value_parser = common::cli::ecdsa_target)]
    security: u32,
    #[arg(value_parser = common::cli::positive)]
    reps: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "bench-peak-memory")]
    let _heap_report = common::heap_run::Report::start();

    let args = <Args as clap::Parser>::parse();
    let Args { exponent, security: lambda, reps, .. } = args;
    let exponent = exponent as usize;
    let mode = if args.mode == "split" { OuterMode::Split } else { OuterMode::AllRows };
    bitz::observability::install().expect("install Perfetto subscriber");
    let threads = common::init();
    let (prepared, setup) = bitz::observability::measure(tracing::info_span!("ecdsa:setup"), || {
        prepare_sha256_ecdsa(exponent, lambda, mode)
            .and_then(|p| p.with_ligerito(common::ligerito_selection(lambda as usize)))
    })?;
    let prepared = prepared?;
    let setup_ms = setup.as_secs_f64() * 1000.;
    let security = prepared.security()?;
    println!("LIGERITO_CONFIG {}", common::ligerito_report(prepared.ligerito_configuration(), prepared.ligerito_configuration().round0(lambda)?));
    let message: Vec<_> = (0..prepared.message_bytes()).map(|i| i as u8).collect();
    let key = SigningKey::from_bytes((&[7u8; 32]).into())?;
    let signature: Signature = key.sign(&message);
    key.verifying_key().verify(&message, &signature)?;
    let point = key.verifying_key().to_encoded_point(false);
    let (r, s) = signature.split_bytes();
    let statement = Sha256EcdsaStatement {
        log_compressions: exponent as u8,
        qx: point.x().unwrap().as_slice().try_into()?,
        qy: point.y().unwrap().as_slice().try_into()?,
        r: r.into(),
        s: s.into(),
    };
    for trial in 0..=reps {
        let recording = bitz::observability::Recording::start(Vec::new())?;
        let start = tracing::info_span!("benchmark:witness").entered();
        let witness = generate_sha256_ecdsa_witness(&prepared, &statement, &message)?;
        drop(start);
        let start = tracing::info_span!("benchmark:commit").entered();
        let hint = commit_sha256_ecdsa(&prepared, &witness)?;
        drop(start);
        let start = tracing::info_span!("benchmark:protocol").entered();
        let proof = prove_sha256_ecdsa(
            &mut Blake3Transcript::new(),
            &prepared,
            &statement,
            &witness,
            &hint,
            4,
        )?;
        drop(start);
        // Serialization and decoding are outside proving and verifying timers.
        let bytes = proof.to_bytes();
        let decoded = Sha256EcdsaProof::from_bytes(&bytes)?;
        assert_eq!(decoded.to_bytes(), bytes);
        let start = tracing::info_span!("benchmark:verification").entered();
        verify_sha256_ecdsa(
            &mut Blake3Transcript::new(),
            &prepared,
            &statement,
            &hint.commitment,
            &decoded,
        )?;
        drop(start);
        let intervals = recording.intervals()?;
        let witness_ms = common::span_ms(&intervals, "benchmark:witness");
        let commit_ms = common::span_ms(&intervals, "benchmark:commit");
        let protocol_ms = common::span_ms(&intervals, "benchmark:protocol");
        let verify_ms = common::span_ms(&intervals, "benchmark:verification");
        let verification = bitz::observability::span(&intervals, "benchmark:verification")?;
        let prove_phases = bitz::observability::totals(intervals.iter().filter(|s| s.end_ns <= verification.start_ns));
        let verify_phases = bitz::observability::phase_totals(&intervals, "benchmark:verification")?;
        println!(
            "{}",
            json!({
                "schema": "bitz/sha256-ecdsa/v2",
                "ligerito": common::ligerito_report(prepared.ligerito_configuration(), prepared.ligerito_configuration().round0(lambda)?), "trial": if trial==0 {"warmup"} else {"sample"}, "sample": trial,
                "log_compressions": exponent, "compressions": prepared.compressions(), "message_bytes": message.len(),
                "mode": args.mode, "security_target": lambda, "threads": threads,
                "security_model": "round-by-round-economic", "economic_bits": security.compute_economic_security_bits(), "statistical_bits_lower_bound": security.compute_statistical_security_bits(),
                "setup_ms": setup_ms, "witness_ms": witness_ms, "commit_ms": commit_ms,
                "prove_ms": commit_ms+protocol_ms, "protocol_ms": protocol_ms, "verify_ms": verify_ms,
                "proof_bytes": bytes.len(), "source_bits": prepared.live_source_bits(), "assignment_bits": prepared.live_assignment_bits(),
                "nonlinear_rows": prepared.nonlinear_rows(), "linear_rows": prepared.linear_rows(), "verified": true,
                "prove_phases_seconds": prove_phases, "verify_phases_seconds": verify_phases,
            })
        );
    }
    Ok(())
}
