//! End-to-end benchmark for the CM-AND relation with the F₂-VIRTUAL
//! `w = x ⊕ y` block: Spartan proves `x + y − w − 2z = 0` per gate; the
//! commitment carries only the `x`/`y`/`z` bits; the virtual BitZ opening
//! derives the `w` bits structurally and opens the terminal claim
//! against the compact commitment. Every measured proof is verified.
//!
//! Defaults to the production sweep `2^15, 2^16` gates. Override with
//! `BITZ_CM_EXPONENTS`, e.g.:
//!
//! ```text
//! BITZ_CM_EXPONENTS="15" BITZ_BENCH_REPS=3 \
//!   cargo bench --bench cm_and --features unchecked
//! ```

mod common;
use clap::builder::TypedValueParser;

use std::hint::black_box;

use bitz::piop::spartan::{
    CM_AND_F_LIVE_SLOTS, CM_AND_H_SLOTS, CmAndWitness, SpartanBitzField, prepare_cm_and_relation,
    prove_cm_and_bitz, spartan_bitz_field_config, verify_cm_and_bitz,
};
use bitz::transcript::Blake3Transcript;
use rand::{RngExt, SeedableRng, rngs::StdRng};

#[global_allocator]
static ALLOCATOR: common::peak_memory::PeakAlloc = common::peak_memory::PeakAlloc;

fn reset_peak() {
    common::peak_memory::reset_peak();
}

fn live_mib() -> f64 {
    common::peak_memory::live_bytes() as f64 / (1024.0 * 1024.0)
}

fn peak_mib() -> f64 {
    common::peak_memory::peak_bytes() as f64 / (1024.0 * 1024.0)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|left, right| left.total_cmp(right));
    samples[samples.len() / 2]
}

fn phase_ms(phases: &[(String, f64)], label: &str) -> Option<f64> {
    phases
        .iter()
        .find(|(phase, _)| *phase == label)
        .map(|(_, seconds)| seconds * 1e3)
}

#[derive(clap::Parser)]
struct Env {
    #[arg(long, env = "BITZ_CM_EXPONENTS", default_value = "15 16",
        value_parser = common::cli::list::<usize>.try_map(|values| {
            if values.iter().all(|&n| n >= 15) { Ok(values) } else { Err("expected exponents >=15") }
        }))]
    exponents: common::cli::List<usize>,
    #[arg(long, env = "BITZ_CM_SEED", default_value_t = 0x0043_4d5f_414e_4400)]
    seed: u64,
}

fn bench_exponent(exponent: usize, reps: usize, root_seed: u64) {
    let gates = 1usize << exponent;
    let shape_seed = root_seed ^ (exponent as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut rng = StdRng::seed_from_u64(shape_seed);
    let field_config = spartan_bitz_field_config();

    let (witness, started) =
        bitz::observability::measure(tracing::info_span!("cm_and:witness"), || {
            CmAndWitness::from_fn(gates, |_| (rng.random::<u32>(), rng.random::<u32>())).unwrap()
        })
        .expect("measure completed operation");
    let witness_ms = started.as_secs_f64() * 1e3;
    let layout = *witness.layout();

    let started_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
    let started = tracing::info_span!("cm_and:started").entered();
    let relation = prepare_cm_and_relation(layout, &field_config)
        .and_then(|p| p.with_ligerito(common::ligerito_selection(100)))
        .expect("valid CM-AND relation");
    let resolved = relation.ligerito_configuration().unwrap();
    println!(
        "LIGERITO_CONFIG {}",
        common::ligerito_report(resolved, resolved.round0(100).unwrap())
    );
    let relation_ms = {
        drop(started);
        bitz::observability::duration(
            &started_recording
                .intervals()
                .expect("complete operation capture"),
            "cm_and:started",
        )
        .expect("query completed operation")
    }
    .as_secs_f64()
        * 1e3;

    let (bit_rows, started) =
        bitz::observability::measure(tracing::info_span!("cm_and:bit_rows"), || {
            witness.f_bit_rows()
        })
        .expect("measure completed operation");
    let bit_rows_ms = started.as_secs_f64() * 1e3;

    let (hint, started) = bitz::observability::measure(tracing::info_span!("cm_and:hint"), || {
        bitz::piop::spartan::cm::commit_cm_and_witness_with_config(
            &layout,
            bit_rows,
            relation.ligerito_configuration().unwrap().prover(),
        )
        .expect("BitZ commitment succeeds")
    })
    .expect("measure completed operation");
    let commit_ms = started.as_secs_f64() * 1e3;

    // Excluded warm-up; also the first end-to-end correctness check.
    let mut pt = Blake3Transcript::new();
    let warm_proof = prove_cm_and_bitz(&mut pt, &relation, &witness, &hint).expect("warm-up prove");
    let mut vt = Blake3Transcript::new();
    verify_cm_and_bitz(&mut vt, &relation, &hint.commitment, &warm_proof).expect("warm-up verify");
    drop(warm_proof);


    let mut prove_ms = Vec::with_capacity(reps);
    let mut verify_ms = Vec::with_capacity(reps);
    let mut prove_splits: Vec<[f64; 3]> = Vec::with_capacity(reps);
    let mut verify_splits: Vec<[f64; 3]> = Vec::with_capacity(reps);
    let mut last_proof = None;
    for _ in 0..reps {
        let mut pt = Blake3Transcript::new();
        let recording = bitz::observability::Recording::start(Vec::new()).expect("start CM prover");
        let proving = tracing::info_span!("benchmark:proving").entered();
        let proof = prove_cm_and_bitz(&mut pt, &relation, &witness, &hint).expect("prove");
        drop(proving);
        let intervals = recording.intervals().expect("query CM prover");
        prove_ms.push(common::span_ms(&intervals, "benchmark:proving"));
        let phases = bitz::observability::phase_totals(&intervals, "benchmark:proving").unwrap();
        if let (Some(a), Some(b), Some(c)) = (
            phase_ms(&phases, "cm-bitz:spartan_prove"),
            phase_ms(&phases, "cm-bitz:bitify_prover"),
            phase_ms(&phases, "cm-bitz:bitz_prove"),
        ) {
            prove_splits.push([a, b, c]);
        }

        let mut vt = Blake3Transcript::new();
        let recording = bitz::observability::Recording::start(Vec::new()).expect("start CM verifier");
        let verification = tracing::info_span!("benchmark:verification").entered();
        verify_cm_and_bitz(&mut vt, &relation, &hint.commitment, &proof).expect("verify");
        drop(verification);
        let intervals = recording.intervals().expect("query CM verifier");
        verify_ms.push(common::span_ms(&intervals, "benchmark:verification"));
        let phases = bitz::observability::phase_totals(&intervals, "benchmark:verification").unwrap();
        if let (Some(a), Some(b), Some(c)) = (
            phase_ms(&phases, "cm-bitz:spartan_verify"),
            phase_ms(&phases, "cm-bitz:bitify_verifier"),
            phase_ms(&phases, "cm-bitz:bitz_verify"),
        ) {
            verify_splits.push([a, b, c]);
        }
        black_box(&proof);
        last_proof = Some(proof);
    }

    let last_proof = last_proof.expect("at least one repetition");
    let bitz_bytes = last_proof.bitz().to_bytes().len();
    let spartan = last_proof.spartan().plain().expect("CM-AND runs the plain kernel");
    let spartan_elements = 4 * spartan.outer.sumcheck.round_polynomials.len()
        + 3
        + 3 * spartan.inner.round_polynomials.len();
    drop(last_proof);

    // One extra proof for the peak-heap measurement.

    let live_before = live_mib();
    reset_peak();
    let mut pt = Blake3Transcript::new();
    let peak_proof = prove_cm_and_bitz(&mut pt, &relation, &witness, &hint).expect("peak prove");
    black_box(&peak_proof);
    let peak = peak_mib();
    let mut vt = Blake3Transcript::new();
    verify_cm_and_bitz(&mut vt, &relation, &hint.commitment, &peak_proof).expect("peak verify");


    let prove_median = median(prove_ms);
    let verify_median = median(verify_ms);
    let split = |samples: &Vec<[f64; 3]>, k: usize| {
        (samples.len() == reps).then(|| median(samples.iter().map(|s| s[k]).collect()))
    };

    let derived_bits = CM_AND_H_SLOTS * layout.capacity();
    let committed_live_bits = CM_AND_F_LIVE_SLOTS * layout.capacity();
    println!();
    println!("cm_and gates=2^{exponent} ({gates}) [production]  seed={shape_seed:#018x}");
    println!(
        "  R1CS: rows={} cols={} nnz={} (A=B=0, one linear C row per gate)",
        gates,
        layout.assignment_len(),
        4 * gates,
    );
    println!(
        "  bits: derived h = {} (x|y|z|w) | committed live = {} (x|y|z; w VIRTUAL) | map nnz = {}",
        derived_bits,
        committed_live_bits,
        relation.map().nnz(),
    );
    println!("  setup: witness {witness_ms:8.2} ms | relation+map {relation_ms:8.2} ms | bit-pack {bit_rows_ms:8.2} ms | commit {commit_ms:8.2} ms");
    println!("  prove:  {prove_median:9.2} ms   (median of {reps})");
    println!("  verify: {verify_median:9.2} ms");
    if let (Some(a), Some(b), Some(c)) = (
        split(&prove_splits, 0),
        split(&prove_splits, 1),
        split(&prove_splits, 2),
    ) {
        println!(
            "  prove split: Spartan {a:9.2} ms | bitify {b:8.2} ms | virtual BitZ {c:9.2} ms | residual {:7.2} ms",
            (prove_median - a - b - c).max(0.0),
        );
    }
    if let (Some(a), Some(b), Some(c)) = (
        split(&verify_splits, 0),
        split(&verify_splits, 1),
        split(&verify_splits, 2),
    ) {
        println!(
            "  verify split: Spartan {a:8.2} ms | bitify {b:8.2} ms | virtual BitZ {c:9.2} ms | residual {:7.2} ms",
            (verify_median - a - b - c).max(0.0),
        );
    }
    println!(
        "  proof:  virtual BitZ {bitz_bytes:9} B | Spartan canonical field payload {spartan_elements:6} elements / {:6} B",
        spartan_elements * 16,
    );
    println!(
        "  heap:   live before prove {live_before:8.2} MiB | peak {peak:8.2} MiB | delta {:8.2} MiB",
        (peak - live_before).max(0.0),
    );
    if let (Some(pa), Some(pb), Some(pc), Some(va), Some(vb), Some(vc2)) = (
        split(&prove_splits, 0),
        split(&prove_splits, 1),
        split(&prove_splits, 2),
        split(&verify_splits, 0),
        split(&verify_splits, 1),
        split(&verify_splits, 2),
    ) {
        println!(
            "  RESULT exponent={exponent} gates={gates} derived_bits={derived_bits} committed_live_bits={committed_live_bits} map_nnz={} prove_ms={prove_median:.4} spartan_prove_ms={pa:.4} bitify_prove_ms={pb:.4} bitz_prove_ms={pc:.4} verify_ms={verify_median:.4} spartan_verify_ms={va:.4} bitify_verify_ms={vb:.4} bitz_verify_ms={vc2:.4} spartan_bytes={} bitz_bytes={bitz_bytes} peak_mib={peak:.4}",
            relation.map().nnz(),
            spartan_elements * 16,
        );
    }
}

fn main() {
    common::cli::EnvironmentCli::parse();
    let Env { exponents, seed } = common::cli::environment();
    let reps = common::reps(None, 5);


    bitz::observability::install().expect("install Perfetto subscriber");
    common::enforce_known_env();
    let _ = flock_core::init_perf_thread_pool();

    println!("CM-AND: Spartan (A=B=0) + virtual BitZ opening (w = x XOR y derived, not committed)");
    #[cfg(feature = "parallel")]
    println!("rayon threads: {}", rayon::current_num_threads());
    println!("repetitions: {reps}; root seed: {seed:#018x}");

    for exponent in exponents {
        flock_core::scratch::clear();
        bench_exponent(exponent, reps, seed);
    }
    flock_core::scratch::clear();
}
