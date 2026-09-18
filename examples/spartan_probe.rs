//! Full Perfetto scope tree of ONE Spartan-path prove (the u32 or the
//! BabyBear paper path) at a chosen exponent: a step-3 (PIOP) microscope for
//! the raw-residue prover kernels. One excluded warm-up prove, then
//! `PROBE_REPS` profiled proves, each dumped separately.
//!
//! ```text
//! PROBE_KIND=u32 PROBE_EXP=20 RUSTFLAGS="-C target-cpu=native" \
//!   cargo run --release --example spartan_probe --features unchecked,span-metrics
//! ```

use ::bitz::piop::spartan::baby_bear_mul::BabyBearMulLayout;
use ::bitz::piop::spartan::protocol;
use ::bitz::piop::spartan::protocol::PreparedRelation;
use bitz::piop::spartan::mul::{MulLayout, MulWitness};

use bitz::piop::spartan::{BabyBearMulWitness, sample_baby_bear_operand_with};
use bitz::transcript::Blake3Transcript;
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.parse().expect("integer env value"))
        .unwrap_or(default)
}

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    let _ = flock_core::init_perf_thread_pool();
    let exponent = env_usize("PROBE_EXP", 20);
    let reps = env_usize("PROBE_REPS", 1);
    let kind = std::env::var("PROBE_KIND").unwrap_or_else(|_| "u32".to_string());
    let count = 1usize << exponent;
    let mut rng = StdRng::seed_from_u64(0x5044_5250_4f42_4500 ^ exponent as u64);
    eprintln!(
        "spartan_probe kind={kind} exponent={exponent} threads={}",
        rayon::current_num_threads()
    );

    match kind.as_str() {
        "u32" => {
            let witness =
                MulWitness::<u32>::from_fn(count, |_| (rng.random::<u32>(), rng.random::<u32>()))
                    .expect("witness");
            let relation =
                PreparedRelation::<MulLayout<u32>>::new(*witness.layout()).expect("relation");
            for rep in 0..=reps {
                let profile =
                    bitz::observability::Recording::start(Vec::new()).expect("capture profile");
                let (hint, started) =
                    bitz::observability::measure(tracing::info_span!("spartan_probe:hint"), || {
                        protocol::commit(&relation, witness.bitz_bit_rows()).expect("commit")
                    })
                    .expect("measure completed operation");
                let commit_ms = started.as_secs_f64() * 1e3;
                let mut transcript = Blake3Transcript::new();
                let (proof, started) =
                    bitz::observability::measure(tracing::info_span!("spartan_probe:proof"), || {
                        protocol::prove(&mut transcript, &relation, &witness, &hint).expect("prove")
                    })
                    .expect("measure completed operation");
                let prove_ms = started.as_secs_f64() * 1e3;
                let mut verifier = Blake3Transcript::new();
                protocol::verify(&mut verifier, &relation, &hint.commitment, &proof)
                    .expect("verify");
                let header = if rep == 0 {
                    format!(
                        "u32 2^{exponent} WARM-UP (commit {commit_ms:.1} ms, prove {prove_ms:.1} ms)"
                    )
                } else {
                    format!(
                        "u32 2^{exponent} prove #{rep} (commit {commit_ms:.1} ms, prove {prove_ms:.1} ms)"
                    )
                };
                bitz::observability::write_profile(std::io::stderr().lock(), &header, &profile.intervals().expect("profile intervals"), None).expect("write profile");
            }
        }
        "bb" => {
            let witness = BabyBearMulWitness::from_fn(count, |_| {
                (
                    sample_baby_bear_operand_with(|| rng.random::<u32>()),
                    sample_baby_bear_operand_with(|| rng.random::<u32>()),
                )
            })
            .expect("witness");
            let layout = *witness.layout();
            let prepared = PreparedRelation::<BabyBearMulLayout>::new(layout).expect("relation");
            for rep in 0..=reps {
                let profile =
                    bitz::observability::Recording::start(Vec::new()).expect("capture profile");
                let (hint, started) =
                    bitz::observability::measure(tracing::info_span!("spartan_probe:hint"), || {
                        protocol::commit(&prepared, witness.bitz_bit_rows()).expect("commit")
                    })
                    .expect("measure completed operation");
                let commit_ms = started.as_secs_f64() * 1e3;
                let mut transcript = Blake3Transcript::new();
                let (proof, started) =
                    bitz::observability::measure(tracing::info_span!("spartan_probe:proof"), || {
                        protocol::prove(&mut transcript, &prepared, &witness, &hint).expect("prove")
                    })
                    .expect("measure completed operation");
                let prove_ms = started.as_secs_f64() * 1e3;
                let mut verifier = Blake3Transcript::new();
                protocol::verify(&mut verifier, &prepared, &hint.commitment, &proof)
                    .expect("verify");
                let header = if rep == 0 {
                    format!(
                        "bb 2^{exponent} WARM-UP (commit {commit_ms:.1} ms, prove {prove_ms:.1} ms)"
                    )
                } else {
                    format!(
                        "bb 2^{exponent} prove #{rep} (commit {commit_ms:.1} ms, prove {prove_ms:.1} ms)"
                    )
                };
                bitz::observability::write_profile(std::io::stderr().lock(), &header, &profile.intervals().expect("profile intervals"), None).expect("write profile");
            }
        }
        other => panic!("PROBE_KIND must be u32 or bb, got {other}"),
    }
}
