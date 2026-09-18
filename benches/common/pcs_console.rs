//! Compact terminal output for the shared-witness PCS comparisons.

use bitz::observability::Interval;

/// Explain the console's timing boundaries once before the size sweep.
pub fn print_timing_definitions() {
    eprintln!("Timing definitions (wall-clock milliseconds):");
    eprintln!(
        "  shared_witness_generation_ms: generate the canonical integer witness once per size, before backend conversion."
    );
    eprintln!(
        "  backend_setup_ms: prepare the backend once per size; excluded from trial timings."
    );
    eprintln!(
        "  commitment_generation_ms: commit the converted witness, including serialization/transcript work inside the commitment phase."
    );
    eprintln!(
        "  opening_proof_ms: generate the terminal evaluation opening proof; excludes commitment, claim derivation, and verification."
    );
    eprintln!(
        "    BitZ: terminal BitZ opening; WHIR: WHIR opening; Binius64: ring-switch reduction + BaseFold opening."
    );
    eprintln!(
        "  commit_and_open_ms: sum of those two disjoint phases; excludes setup, witness generation/conversion, claim derivation, and verification."
    );
    eprintln!(
        "  Trial lines are individual durations, not medians; warmup lines are excluded from measured-sample summaries."
    );
    eprintln!(
        "  Full multiplication PIOP and end-to-end multiplication proving: N/A (PCS-only benchmark)."
    );
}

/// Print outer phase durations after the verified trial has finished.
pub fn print_trial(
    trial: &str,
    intervals: &[Interval],
    proof_bytes: usize,
    wire_bytes: usize,
) {
    let phase = |label| {
        intervals
            .iter()
            .find(|interval| interval.label() == label)
            .expect("PCS trial records its outer commit and opening phases")
    };
    let commit = phase("pcs-compare:commit");
    let opening = phase("pcs-compare:opening");
    // These outer phases are sequential; child spans must not be added again.
    assert!(commit.end_ns <= opening.start_ns);
    let milliseconds = |interval: &Interval| {
        std::time::Duration::from_nanos(interval.end_ns - interval.start_ns).as_secs_f64() * 1e3
    };
    let commitment_generation_ms = milliseconds(commit);
    let opening_proof_ms = milliseconds(opening);
    let commit_and_open_ms = commitment_generation_ms + opening_proof_ms;
    eprintln!(
        "    {trial}: commitment_generation_ms={commitment_generation_ms:.3}, \
         opening_proof_ms={opening_proof_ms:.3}, \
         commit_and_open_ms={commit_and_open_ms:.3}, proof={proof_bytes} B, wire={wire_bytes} B"
    );
}
