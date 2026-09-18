//! End-to-end benchmark for CHAINED SHA-256 compressions — the
//! Merkle–Damgård chain `H_{i+1} = Compress(H_i, M_i)` over `2^k` blocks
//! (a `64·2^k`-byte message) proved as one relation whose intermediate
//! chaining values are witness. The counterpart of `sha256_compressions`
//! (independent compressions with every input public): same circuit, same
//! 184 linear constraints per compression, same runtime-prime protocol and
//! direct product opening, but instance `i`'s chaining state is read from
//! instance `i - 1`'s committed output cells through the chained virtual
//! map, so the committed source shrinks by the 256 state bits per
//! compression and the public statement is the block sequence plus the
//! final digest (the standard initial state is baked into the relation).
//!
//! Output follows the unified schema (`docs/bench-schema.md`): the
//! end-to-end prover (`prove_ms`) covers bit packing, commitment, the prime
//! draw + grinding, the linear PIOP, bitification, and the PCS opening.
//! Witness synthesis (the native chain plus per-compression circuit
//! replay) and public relation construction are excluded and reported
//! separately. Every measured proof is verified.
//!
//! Defaults to the complete supported range `2^7, ..., 2^16` with three
//! measured repetitions after one warm-up. Override with, for example:
//!
//! ```text
//! BITZ_BENCH_SHAPES="10 12" BITZ_BENCH_REPS=1 \
//!   cargo bench --bench sha256_chain --features unchecked
//! ```
//!
//! `BITZ_BENCH_LAMBDA=100|128|sha128-reference-schedule` selects the security
//! profile (default `Lambda100`; the two-prime `Limber114` profile is
//! MultiSwap-only and is rejected here). Blocks are pseudo-random from
//! `BITZ_BENCH_SEED`; a real message is the same bench with its parsed,
//! padded blocks.

mod common;
#[cfg(feature = "bench-peak-memory")]
#[global_allocator]
static HEAP_ALLOCATOR: common::peak_memory::PeakAlloc = common::peak_memory::PeakAlloc;

use std::hint::black_box;

use {
    circuit::linear_map::binary::VirtualMap,
    bitz::{
        piop::spartan::{
            IopSecurityProfile, PreparedSha256ChainBatch, PrimePolicy,
            SHA256_CHAIN_F_INSTANCE_BITS, SHA256_CHAIN_H_BAR_LIVE_BITS, SHA256_CONSTRAINTS,
            SHA256_MAX_LOG_COMPRESSIONS, Sha256ConstraintError,
            commit_sha256_chain_witness_with_config, generate_sha256_chain_witnesses,
            prepare_sha256_chain_batch_with_profile, prove_sha256_chain_with_config,
            sha256_chain_configs, verify_sha256_chain_with_config,
        },
        transcript::Blake3Transcript,
    },
};

/// One rep's raw measurements; step extraction happens in `common`.
struct RepTiming {
    e2e_ms: f64,
    witness_ms: f64,
    commit_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    prove_phases: Vec<(String, f64)>,
    verify_phases: Vec<(String, f64)>,
    piop_bytes: usize,
    bitz_bytes: usize,
    forests: usize,
}

impl RepTiming {
    fn emit_trial(&self, trial: &str) {
        if std::env::var("BITZ_BENCH_PHASE_SAMPLES").is_ok_and(|v| v == "1") {
            let gkr = self
                .prove_phases
                .iter()
                .find(|(n, _)| n == "mc:forest")
                .map_or(0.0, |(_, v)| 1000.0 * v);
            println!(
                "PROVER_TRIAL {}",
                serde_json::json!({"trial":trial,"verified":true,
                "e2e_ms":self.e2e_ms,"prove_ms":self.prove_ms,"witness_ms":self.witness_ms,
                "verify_ms":self.verify_ms,"gkr_ms":gkr,"proof_bytes":self.piop_bytes+self.bitz_bytes})
            );
        }
    }
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

fn make_blocks(compressions: usize, seed: u64) -> Vec<[u32; 16]> {
    let mut rng = SplitMix64(seed);
    (0..compressions)
        .map(|_| std::array::from_fn(|_| rng.next_u32()))
        .collect()
}

fn shapes() -> Vec<usize> {
    common::shape_values(
        None,
        clap::builder::RangedU64ValueParser::<usize>::new()
            .range(7..=SHA256_MAX_LOG_COMPRESSIONS as u64),
    )
    .unwrap_or_else(|| (7..=16).collect())
}

fn fmt_ms(milliseconds: f64) -> String {
    if milliseconds < 1.0 {
        format!("{:8.2} us", milliseconds * 1e3)
    } else if milliseconds < 1_000.0 {
        format!("{milliseconds:8.2} ms")
    } else {
        format!("{:8.2} s ", milliseconds / 1e3)
    }
}

fn run_once(
    blocks: &[[u32; 16]],
    prepared: &PreparedSha256ChainBatch,
    pc: &flock_core::pcs::ligerito::ProverConfig,
    vc: &flock_core::pcs::ligerito::VerifierConfig,
) -> RepTiming {
    let recording =
        bitz::observability::Recording::start(Vec::new()).expect("start SHA chain trial");

    let e2e = tracing::info_span!("chain:witness_to_proof").entered();
    // Witness synthesis (the native chain, the per-compression circuit
    // replay, and packing) is excluded from the prover boundary
    // (docs/bench-schema.md).
    let witness_scope = tracing::info_span!("chain:witness").entered();
    let witness = generate_sha256_chain_witnesses(prepared, blocks)
        .expect("SHA chain witness synthesis succeeds");
    let statement = witness.statement();
    black_box(&statement);
    drop(witness_scope);

    // End-to-end prove: Step 1 commitment plus the runtime-prime proof.
    let proving = tracing::info_span!("chain:proving").entered();
    let commit = tracing::info_span!("chain:commit").entered();
    let hint = commit_sha256_chain_witness_with_config(prepared, &witness, pc)
        .expect("SHA chain source commitment succeeds");
    drop(commit);

    let mut prover_transcript = Blake3Transcript::new();
    let proof = prove_sha256_chain_with_config(
        &mut prover_transcript,
        prepared,
        &statement,
        &witness,
        &hint,
        pc,
    )
    .expect("SHA chain proof succeeds");
    drop(proving);
    drop(e2e);
    let forests = proof.bitz().mfs.len();
    assert_eq!(
        forests, 1,
        "every chain proof uses exactly one merged forest"
    );

    let mut verifier_transcript = Blake3Transcript::new();
    let verification = tracing::info_span!("chain:verification").entered();
    verify_sha256_chain_with_config(
        &mut verifier_transcript,
        prepared,
        &statement,
        &hint.commitment,
        &proof,
        vc,
    )
    .expect("SHA chain proof verifies");
    drop(verification);
    common::proof_fingerprint::linear(&proof, &hint.commitment.root, &prover_transcript);
    let intervals = recording.intervals().expect("query SHA chain trial");
    let witness_ms = common::span_ms(&intervals, "chain:witness");
    let commit_ms = common::span_ms(&intervals, "chain:commit");
    let prove_ms = common::span_ms(&intervals, "chain:proving");
    let verify_ms = common::span_ms(&intervals, "chain:verification");
    let prove_phases = bitz::observability::phase_totals(&intervals, "chain:proving").unwrap();
    let verify_phases = bitz::observability::phase_totals(&intervals, "chain:verification").unwrap();
    black_box(&proof);

    RepTiming {
        e2e_ms: common::span_ms(&intervals, "chain:witness_to_proof"),
        witness_ms,
        commit_ms,
        prove_ms,
        verify_ms,
        prove_phases,
        verify_phases,
        piop_bytes: proof.piop_bytes(),
        bitz_bytes: proof.bitz().to_bytes().len(),
        forests,
    }
}

fn bench_shape<P: IopSecurityProfile>(
    exponent: usize,
    reps: usize,
    root_seed: u64,
    threads: usize,
) {
    let shape_seed =
        root_seed ^ (exponent as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x6368_6169_6e5f_7368; // "chain_sh"
    let slug = format!("chain-2p{exponent}");

    let setup_started_recording =
        bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
    let setup_started = tracing::info_span!("sha256_chain:setup_started").entered();
    let prepared = match prepare_sha256_chain_batch_with_profile::<P>(exponent)
        .and_then(|p| p.with_ligerito(common::ligerito_selection(P::LIGERITO_TARGET_BITS)))
    {
        Ok(prepared) => prepared,
        Err(
            error @ (Sha256ConstraintError::Profile(_) | Sha256ConstraintError::PrimeProfile(_)),
        ) => {
            println!();
            println!("sha256_chain {slug} profile={}: SKIPPED - {error}", P::NAME);
            return;
        }
        Err(error) => panic!("prepare failed: {error}"),
    };
    println!(
        "LIGERITO_CONFIG {}",
        common::ligerito_report(
            prepared
                .ligerito_configuration()
                .expect("validated Ligerito"),
            prepared.security().ood
        )
    );
    let (pc, vc) = sha256_chain_configs(&prepared).expect("valid Ligerito config");
    let setup_ms = {
        drop(setup_started);
        bitz::observability::duration(
            &setup_started_recording
                .intervals()
                .expect("complete operation capture"),
            "sha256_chain:setup_started",
        )
        .expect("query completed operation")
    }
    .as_secs_f64()
        * 1e3;
    let compressions = prepared.instances();
    let message_bytes = 64 * compressions;
    let live_source_cells = 1 + SHA256_CHAIN_F_INSTANCE_BITS * compressions;
    let source_cells = prepared.source_params().cells();
    let opening = *prepared.assignment_params();
    let product_cells = opening.cells();

    println!();
    println!(
        "=== 2^{exponent} = {compressions} chained SHA-256 compressions ({message_bytes}-byte message, intermediate states private) ==="
    );
    println!(
        "  source: {live_source_cells} live / {source_cells} padded cells ({} committed bits/compression: block + hints; the 256 chaining-state bits are read from the previous compression)",
        SHA256_CHAIN_F_INSTANCE_BITS,
    );
    println!(
        "  product tensor: {} local cells × {compressions} instances = {product_cells} cells | chained map nnz={} | linear relation: {} rows/compression | setup {}",
        SHA256_CHAIN_H_BAR_LIVE_BITS,
        prepared.map().nnz(),
        SHA256_CONSTRAINTS,
        fmt_ms(setup_ms),
    );
    println!(
        "  public statement: {compressions} blocks + digest ({} words); initial state = SHA-256 IV (relation constant)",
        16 * compressions + 8
    );
    println!(
        "  security profile: {} (λ={})",
        prepared.security().profile_name,
        prepared.security().lambda,
    );

    let warm = run_once(&make_blocks(compressions, shape_seed), &prepared, &pc, &vc);
    println!(
        "  opening layout: direct product opening on the chained map | BitZ rows 2^{} × columns 2^{} | forests {} | read-off ≤ 2^{} integers per forest",
        opening.row_vars, opening.col_vars, warm.forests, opening.col_vars
    );
    warm.emit_trial("warmup");
    black_box(warm);

    let mut prover = common::StepSamples::default();
    let mut verifier = common::StepSamples::default();
    let mut witness_samples = Vec::with_capacity(reps);
    let mut last = None;
    for sample in 0..reps {
        let input_seed = shape_seed ^ ((sample + 1) as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
        let blocks = make_blocks(compressions, input_seed);
        let timing = run_once(&blocks, &prepared, &pc, &vc);
        println!(
            "  SAMPLE exponent={exponent} sample={} compressions={compressions} message_bytes={message_bytes} witness_ms={:.6} commit_ms={:.6} prove_ms={:.6} verify_ms={:.6} verified=true",
            sample + 1,
            timing.witness_ms,
            timing.commit_ms,
            timing.prove_ms,
            timing.verify_ms,
        );
        common::print_regression_phases(&timing.prove_phases);
        timing.emit_trial("sample");
        prover.record_prove(timing.prove_ms, timing.commit_ms, &timing.prove_phases);
        verifier.record_verify(timing.verify_ms, &timing.verify_phases);
        witness_samples.push(timing.witness_ms);
        last = Some(timing);
    }
    let last = last.expect("positive repetition count");

    let prover_medians = prover.medians();
    let throughput = compressions as f64 / (prover_medians.total / 1e3);
    println!(
        "  end-to-end prove: {} median | {throughput:10.0} compressions/s | {:8.2} MiB/s of message",
        fmt_ms(prover_medians.total),
        message_bytes as f64 / (1u64 << 20) as f64 / (prover_medians.total / 1e3),
    );
    let report = common::BenchReport {
        bench: "sha256_chain",
        shape: slug,
        extra: vec![
            common::ligerito_identity(
                prepared.ligerito_configuration().unwrap(),
                prepared.security().ood,
            ),
            ("profile".into(), prepared.security().profile_name.into()),
            ("compressions".into(), compressions.to_string()),
            ("message_bytes".into(), message_bytes.to_string()),
            ("mnum_rows".into(), product_cells.to_string()),
            ("step2_semantics".into(), "runtime_field_setup".into()),
            ("throughput_per_s".into(), format!("{throughput:.3}")),
            ("shape_seed".into(), format!("{shape_seed:#018x}")),
        ],
        lambda: Some(prepared.security().lambda),
        lambda_achieved: Some(prepared.security().accounting.achieved_bits()),
        lambda_bind: Some(prepared.security().accounting.binding_term().name.into()),
        threads,
        reps,
        seed: Some(root_seed),
        witness_ms: common::median(&witness_samples),
        setup_ms,
        prover: prover_medians,
        verifier: verifier.medians(),
        proof: common::ProofBytes {
            piop: last.piop_bytes,
            open: last.bitz_bytes,
        },
    };
    report.print_human();
}

fn main() {
    common::start_gkr_recording();
    #[cfg(feature = "bench-peak-memory")]
    let _heap_report = common::heap_run::Report::start();

    common::cli::EnvironmentCli::parse();
    let reps = common::reps(None, 3);
    let root_seed = common::seed(None, 0x4632_5a5f_4348_4149);
    let selected = common::security_profile(PrimePolicy::SingleDerived);
    let profile = selected.unwrap_or(common::SecurityProfile::Lambda100);

    let shapes = shapes();
    bitz::observability::install().expect("install Perfetto subscriber");
    let threads = common::init();

    println!(
        "SHA-256 chain: H_{{i+1}} = Compress(H_i, M_i) from the IV; source [1|block₀,hints₀|block₁,hints₁|…], chained map + direct product opening + virtual BitZ"
    );
    #[cfg(feature = "parallel")]
    println!("rayon threads: {threads}");
    println!("repetitions: {reps}; warmups: 1; root seed: {root_seed:#018x}");
    println!(
        "security profile: {}",
        common::profile_banner(selected, common::SecurityProfile::Lambda100)
    );

    for exponent in shapes {
        flock_core::scratch::clear();
        common::with_profile!(profile, bench_shape(exponent, reps, root_seed, threads));
    }
    flock_core::scratch::clear();
    common::print_gkr_schedules();
}
