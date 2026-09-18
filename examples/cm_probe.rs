//! Full phase-tree dump of one CM-AND prove + verify (the virtual BitZ
//! pipeline), for hot-path diagnosis:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" \
//!   cargo run --release --features unchecked,span-metrics --example cm_probe -- [log2_gates]
//! ```
//!
//! Prints Perfetto's scope tree (stderr) once for the prove and once
//! for the verify, plus wall-clock totals and the proof size.


use bitz::piop::spartan::{
    CmAndWitness, SpartanBitzField, commit_cm_and_witness, prepare_cm_and_relation,
    prove_cm_and_bitz, spartan_bitz_field_config, verify_cm_and_bitz,
};
use bitz::transcript::Blake3Transcript;

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    let _ = flock_core::init_perf_thread_pool();
    let log2_gates: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(15);
    let gates = 1usize << log2_gates;
    let config = spartan_bitz_field_config();

    let witness = CmAndWitness::from_fn(gates, |i| {
        let r = splitmix(0xCAFE ^ i as u64);
        (r as u32, (r >> 32) as u32)
    })
    .unwrap();
    let layout = *witness.layout();
    let relation = prepare_cm_and_relation(layout, &config).unwrap();
    let hint = commit_cm_and_witness(&layout, witness.f_bit_rows()).unwrap();

    // Warm-up (excluded), also the correctness check.
    let profile = bitz::observability::Recording::start(Vec::new()).expect("capture warmup");
    let mut pt = Blake3Transcript::new();
    let proof = prove_cm_and_bitz(&mut pt, &relation, &witness, &hint).unwrap();
    let mut vt = Blake3Transcript::new();
    verify_cm_and_bitz(&mut vt, &relation, &hint.commitment, &proof).unwrap();
    bitz::observability::write_profile(std::io::stderr().lock(), "cm_probe warmup", &profile.intervals().expect("warmup intervals"), None).expect("write profile");
    drop(proof);

    let profile = bitz::observability::Recording::start(Vec::new()).expect("capture prove");
    let mut pt = Blake3Transcript::new();
    let started_recording =
        bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
    let started = tracing::info_span!("cm_probe:started").entered();
    let proof = prove_cm_and_bitz(&mut pt, &relation, &witness, &hint).unwrap();
    eprintln!(
        "== PROVE 2^{log2_gates} gates: {:.2} ms ==",
        { drop(started); bitz::observability::duration(&started_recording.intervals().expect("complete operation capture"), "cm_probe:started").expect("query completed operation") }.as_secs_f64() * 1e3
    );
    bitz::observability::write_profile(std::io::stderr().lock(), "cm_probe prove", &profile.intervals().expect("prover intervals"), None).expect("write profile");

    let profile = bitz::observability::Recording::start(Vec::new()).expect("capture verify");
    let mut vt = Blake3Transcript::new();
    let started_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
    let started = tracing::info_span!("cm_probe:started").entered();
    verify_cm_and_bitz(&mut vt, &relation, &hint.commitment, &proof).unwrap();
    eprintln!(
        "== VERIFY: {:.2} ms ==",
        { drop(started); bitz::observability::duration(&started_recording.intervals().expect("complete operation capture"), "cm_probe:started").expect("query completed operation") }.as_secs_f64() * 1e3
    );
    bitz::observability::write_profile(std::io::stderr().lock(), "cm_probe verify", &profile.intervals().expect("verifier intervals"), None).expect("write profile");

    eprintln!("proof: virtual BitZ {} B", proof.bitz().to_bytes().len());
    let digest = blake3::hash(&proof.bitz().to_bytes());
    eprintln!("bitz proof digest: {}", digest.to_hex());
}
