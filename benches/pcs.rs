//! BitZ PCS benchmark: commit / prove / verify wall-clock, serialized proof
//! size, codec round-trip time, and peak heap per phase, per shape.
//!
//! Plain `harness = false` binary (no criterion — zero extra deps), in the
//! style of flock's `sha2_proof` bench; the peak-memory tracker reports the
//! live-heap high-water mark ("net outstanding bytes"), the same notion as
//! flock's benches and zinc-plus's `f2_int_ligerito_mem`, so numbers compare
//! directly across the three repos.
//!
//! Run:
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench pcs
//! ```
//!
//! Knobs:
//! - `BITZ_BENCH_SHAPES`: space/comma-separated `t:s:W` triples overriding the
//!   default shape list, e.g. `BITZ_BENCH_SHAPES="10:6:1 14:8:1"`.
//! - `BITZ_BENCH_REPS`: timing repetitions per shape (median reported;
//!   default 5).
//! - `BITZ_BENCH_FILL`: witness fill fraction in (0, 1] (default 1.0) — the
//!   trailing `(1 − fill)` of the columns are left ALL ZERO, i.e. the
//!   zero padding a witness of `N = fill·2^n` cells carries. With
//!   `BITZ_COL_ELIDE=1` (the default) the forest skips those trees; set
//!   `BITZ_COL_ELIDE=0` to measure the same instance un-elided. The
//!   printed `proof-fnv` is identical either way (byte-identity pin).
//! - `BITZ_LIG_PROFILE`: Ligerito profile at `m = m_p + 7 ≥ 22` —
//!   `custom:1:4` (DEFAULT; validator-gated Johnson geometry at base RS
//!   rate 1/2, initial_k = 4), `slim` (embedded; fewer queries + 16-bit
//!   grinding at the same 100-bit target, the proof-size profile), `fast`
//!   (base RS rate 1/2), `secure` (120-bit UDR), or any
//!   `custom:<log_inv_rate>:<initial_k>[:<bits>]` — the optional `bits`
//!   sets the round-by-round security target (default 100; e.g.
//!   `custom:3:4:128` for a ~128-bit opener, queries/grinding/OOD
//!   re-solved and flock-validator-gated), or
//!   `udr:<log_inv_rate>:<initial_k>[:<bits>]` — queries-ONLY security
//!   (UDR regime, zero grinding of either kind, zero OOD; ceiling ≈115
//!   bits at n=22 / ≈109 at n=28 from the L0 UDR fold error).
//!   `udrg:3:4` uses matched UDR with fold grinding. Unsupported shapes
//!   are rejected; production requests never fall back to ad-hoc settings.
//! - `BITZ_BENCH_EXT`: also run the extension-field arm against the same
//!   commitment — `1`/`gl2` = Goldilocks² (e=2), `bb4` = BabyBear⁴
//!   (X⁴ − 11, the Plonky3 challenge field; e=4), `kb5` = a KoalaBear
//!   quintic (e=5, leanVM's field shape; stand-in X⁵−3 reduction).
//!   Companion Plonky3 baseline:
//!   `~/Plonky3 uni-stark/benches/prove_mul_babybear.rs`.
//!
//! Protocol notes (from the zinc-plus measurement lore): idle the box first;
//! for quotable *time* numbers at big shapes run one shape per process (the
//! peak-memory numbers reset per shape and are fine in one process); quote
//! medians, expect ±5–15 % run-to-run.

mod common;
use clap::builder::TypedValueParser;

use std::hint::black_box;

use bitz::ext_proj::{ExtProjParams, sample_proj_point, sample_proj_prime};
use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{
    OodRoundParams, absorb_standalone_mod_q_claim, absorb_standalone_mod_q_statement,
    commit_rs_ligerito_rows, prove_mle_eval_mod_q_ligerito_with_ood,
    verify_mle_eval_mod_q_ligerito_runtime,
};
use bitz::pcs::{IntegerMatrixLayout, mod_q_chunk_width, mod_q_num_chunks, smallest_generator};
use flock_core::pcs::ligerito::{ProverConfig as LigPc, VerifierConfig as LigVc};

#[derive(clap::Parser)]
struct Env {
    // n=30/32 stay outside the default sweep for runtime, not memory.
    // At n>=26, measure one shape per process for quotable timings.
    #[arg(long, env = "BITZ_BENCH_SHAPES", default_value = "13:7:1 14:8:1 16:10:1 17:11:1 7:8:32", value_parser = parse_shapes)]
    shapes: common::cli::List<(usize, usize, usize)>,
    #[arg(long, env = "BITZ_BENCH_FILL", default_value_t = 1.0,
        value_parser = str::parse::<f64>.try_map(|fill| {
            if fill > 0.0 && fill <= 1.0 { Ok(fill) } else { Err("expected a fraction in (0, 1]") }
        }))]
    fill: f64,
    #[arg(long, env = "BITZ_BENCH_EXT", value_parser = ["1", "gl2", "bb4", "kb5"])]
    extension: Option<String>,
    #[arg(long, env = "BITZ_LIG_PROFILE", default_value = "custom:1:4")]
    ligerito: String,
}

fn parse_shapes(value: &str) -> Result<Vec<(usize, usize, usize)>, String> {
    common::cli::list::<String>(value)?
        .into_iter()
        .map(|shape| {
            let parts = shape
                .split(':')
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?;
            let [t, s, w]: [usize; 3] = parts
                .try_into()
                .map_err(|_| "expected t:s:W triple".to_owned())?;
            Ok((t, s, w))
        })
        .collect()
}

/// Resolve the explicitly requested Ligerito policy, or Johnson+OOD by default.
fn bench_lig_configs(
    m_p: usize,
    request: &str,
) -> (
    (LigPc, LigVc),
    String,
    Option<OodRoundParams>,
    bitz::ligerito_flock::ResolvedLigerito,
) {
    let target = if request == "secure" {
        128
    } else {
        request
            .split(':')
            .nth(3)
            .map(|n| n.parse::<usize>().expect("invalid Ligerito target"))
            .unwrap_or(100)
    };
    let resolved = bitz::ligerito_flock::LigeritoSelection::parse(request, target)
        .and_then(|selection| selection.resolve(m_p, target))
        .expect("unsupported Ligerito configuration");
    let ood = resolved
        .round0(target as u32)
        .expect("Round-0 security preflight");
    (
        (resolved.prover().clone(), resolved.verifier().clone()),
        resolved.selection().name(),
        ood,
        resolved,
    )
}

// Peak-heap tracker (wraps System): high-water mark of currently outstanding
// bytes. Negligible overhead (one relaxed atomic op per alloc/dealloc).
#[global_allocator]
static ALLOCATOR: common::peak_memory::PeakAlloc = common::peak_memory::PeakAlloc;

fn reset_peak() {
    common::peak_memory::reset_peak();
}
fn peak_mb() -> f64 {
    common::peak_memory::peak_bytes() as f64 / (1024.0 * 1024.0)
}
fn live_mb() -> f64 {
    common::peak_memory::live_bytes() as f64 / (1024.0 * 1024.0)
}

/// The evaluation prime is SAMPLED from the transcript after the
/// commitment, uniformly among the primes of `[2^(b−1), 2^b)` with
/// `b = min(113, 127 − t − W)` — the paper's Strategy-1 field policy capped
/// by the one-chunk exponent-fold width (the same rule as the Spartan
/// security profile's derived interval); the evaluation point follows.
fn standalone_q_bits(p: &IntegerMatrixLayout) -> usize {
    mod_q_chunk_width(p).min(113)
}

/// The transcript-sampled instance: prime, point-induced `eq` tables.
struct StandaloneInstance {
    q: u128,
    row_weights_q: Vec<u128>,
    col_weights_q: Vec<u128>,
}

fn sample_standalone_instance(
    transcript: &mut bitz::transcript::Blake3Transcript,
    p: &IntegerMatrixLayout,
    q_bits: usize,
) -> StandaloneInstance {
    let _g = tracing::info_span!("mq:sample_instance").entered();
    let proj = ExtProjParams {
        prime_bits: q_bits,
        ..ExtProjParams::default()
    };
    let q = sample_proj_prime(transcript, &proj).expect("bounded benchmark prime search");
    let arith = field::FpCtx::from_prime_u128(q);
    let r1: Vec<u128> = (0..p.row_vars)
        .map(|_| sample_proj_point(transcript, q))
        .collect();
    let r2: Vec<u128> = (0..p.col_vars)
        .map(|_| sample_proj_point(transcript, q))
        .collect();
    StandaloneInstance {
        q,
        row_weights_q: eq_table_mod_q(&arith, &r1),
        col_weights_q: eq_table_mod_q(&arith, &r2),
    }
}

/// `eq(b, r) mod q` over `b ∈ {0,1}^{r.len()}` (index bit `k` ↔ `r[k]`).
fn eq_table_mod_q(arith: &field::FpCtx<2>, r: &[u128]) -> Vec<u128> {
    let q = arith.modulus_u128();
    let mut table = vec![1u128 % q];
    for &coord in r {
        let mut next = Vec::with_capacity(table.len() * 2);
        for &v in &table {
            let v1 = arith.mul_u128(v, coord);
            let v0 = if v >= v1 { v - v1 } else { v + q - v1 };
            next.push(v0);
            next.push(v1);
        }
        table = next;
    }
    table
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_shape(t: usize, s: usize, w: usize, reps: usize, env: &Env) {
    let alpha = smallest_generator();
    let p = IntegerMatrixLayout {
        row_vars: t,
        col_vars: s,
        word_bits: w,
    };
    let q_bits = standalone_q_bits(&p);
    let m_p = packed_vars(&p);
    let lch = mod_q_num_chunks(&p, q_bits);
    let setup_started_recording =
        bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
    let setup_started = tracing::info_span!("pcs:setup_started").entered();
    let ((pc, vc), lig_tag, ood, resolved) = bench_lig_configs(m_p, &env.ligerito);
    println!(
        "LIGERITO_CONFIG {}",
        common::ligerito_report(&resolved, ood)
    );
    let setup_ms = {
        drop(setup_started);
        bitz::observability::duration(
            &setup_started_recording
                .intervals()
                .expect("complete operation capture"),
            "pcs:setup_started",
        )
        .expect("query completed operation")
    }
    .as_secs_f64()
        * 1e3;

    // Deterministic non-degenerate instance, generated STRAIGHT INTO the
    // per-column bit rows (`repack_leaf_bits` layout: bit `(b<<log₂W)|j`
    // of row `c` = bit `j` of cell `(b,c)`) — the `u128` cell tensor
    // (16 B per cell; 17 GB at n=30) never exists, mirroring the
    // upstream packed-transpose commit restructure.
    let witness_started_recording =
        bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
    let witness_started = tracing::info_span!("pcs:witness_started").entered();
    let mask = if w >= 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    };
    let cell = |b: usize, c: usize| -> u128 {
        (p.cell_index(b, c) as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask
    };
    let log_w = w.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    let words = row_len.div_ceil(64);
    // Fill fraction: columns `live..2^s` stay ALL ZERO — exactly the
    // padding of a witness with N = fill·2^n cells (the column axis is
    // the high-order index, so a zero-padded witness ends in whole zero
    // columns). `y` below is derived from SET BITS, so it stays correct.
    let fill = env.fill;
    let live = (((p.cols() as f64) * fill).ceil() as usize).clamp(1, p.cols());
    let rows: Vec<Vec<u64>> = (0..p.cols())
        .map(|c| {
            let mut wv = vec![0u64; words];
            if c >= live {
                return wv;
            }
            for b in 0..p.rows() {
                let v = cell(b, c);
                for j in 0..w {
                    if (v >> j) & 1 == 1 {
                        let i = (b << log_w) | j;
                        wv[i >> 6] |= 1u64 << (i & 63);
                    }
                }
            }
            wv
        })
        .collect();
    let witness_ms = {
        drop(witness_started);
        bitz::observability::duration(
            &witness_started_recording
                .intervals()
                .expect("complete operation capture"),
            "pcs:witness_started",
        )
        .expect("query completed operation")
    }
    .as_secs_f64()
        * 1e3;

    let n = t + s;
    println!(
        "\n=== n={n} (t={t}, s={s}, W={w}, m_p={m_p}, chunks={lch}, lig={lig_tag}@r1/{}k{}, data={} KiB, live={live}/{} elide={}) ===",
        1usize << pc.log_inv_rates[0],
        pc.initial_k,
        (p.cells() * w).div_ceil(8) >> 10,
        p.cols(),
        std::env::var("BITZ_COL_ELIDE").unwrap_or_else(|_| "1".into())
    );

    // Commit: timed + its own peak window (the commitment/hint stays live).
    reset_peak();
    let (hint, t0) = bitz::observability::measure(tracing::info_span!("pcs:hint"), || {
        commit_rs_ligerito_rows(&p, rows, &pc)
    })
    .expect("measure completed operation");
    let commit_ms = t0.as_secs_f64() * 1e3;
    println!(
        "  commit:  {commit_ms:8.2} ms   peak {:8.2} MB   live-after {:6.2} MB",
        peak_mb(),
        live_mb()
    );

    // The instance: the transcript-sampled prime and point (replayed inside
    // every timed prove/verify), then the claimed μ from the SET BITS of the
    // committed rows (O(popcount) mod-q adds).
    let instance = {
        let mut st = bitz::transcript::Blake3Transcript::new();
        absorb_standalone_mod_q_statement(&mut st, &hint.commitment, &p, alpha, q_bits, ood, &vc);
        resolved.bind(&mut st);
        let _ = bitz::ligerito_flock::bind_prover_ood(&mut st, &hint, ood);
        sample_standalone_instance(&mut st, &p, q_bits)
    };
    let q = instance.q;
    let arith = field::FpCtx::from_prime_u128(q);
    let pow2_q: Vec<u128> = (0..w).map(|j| arith.reduce_u128(1u128 << j)).collect();
    let mut y = 0u128;
    for (c, row) in hint.rows().iter().enumerate() {
        let mut acc = 0u128;
        for (wi, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let i = (wi << 6) | bit;
                let (b, j) = (i >> log_w, i & (w - 1));
                let term = if j == 0 {
                    instance.row_weights_q[b]
                } else {
                    arith.mul_u128(instance.row_weights_q[b], pow2_q[j])
                };
                acc = arith.add_u128(acc, term);
            }
        }
        y = arith.add_u128(y, arith.mul_u128(instance.col_weights_q[c], acc));
    }
    println!(
        "  instance: q ∈ [2^{}, 2^{q_bits}) transcript-sampled after the commitment; Round 0 (OOD): {}",
        q_bits - 1,
        match ood {
            Some(round) => format!("executed, {} grinding bits", round.grinding_bits),
            None => "skipped (unique-decoding opener)".to_string(),
        }
    );
    let prove_once = |hint: &bitz::ligerito_flock::FlockCommitHint| {
        let mut pt = bitz::transcript::Blake3Transcript::new();
        absorb_standalone_mod_q_statement(&mut pt, &hint.commitment, &p, alpha, q_bits, ood, &vc);
        resolved.bind(&mut pt);
        let bound_ood = bitz::ligerito_flock::bind_prover_ood(&mut pt, hint, ood);
        let sampled = sample_standalone_instance(&mut pt, &p, q_bits);
        assert_eq!(
            sampled.q, q,
            "the transcript-sampled prime must be reproducible"
        );
        absorb_standalone_mod_q_claim(&mut pt, q, y);
        prove_mle_eval_mod_q_ligerito_with_ood(
            &mut pt,
            hint,
            &p,
            &sampled.row_weights_q,
            q_bits,
            alpha,
            bound_ood,
            &pc,
        )
    };
    let verify_once = |proof: &bitz::ligerito_flock::IntEvalRsLigModQProof| {
        let mut vt = bitz::transcript::Blake3Transcript::new();
        absorb_standalone_mod_q_statement(&mut vt, &hint.commitment, &p, alpha, q_bits, ood, &vc);
        resolved.bind(&mut vt);
        let bound_ood =
            bitz::ligerito_flock::bind_verifier_ood(&mut vt, m_p, ood, proof.ood.as_ref())
                .expect("Round 0");
        let sampled = sample_standalone_instance(&mut vt, &p, q_bits);
        absorb_standalone_mod_q_claim(&mut vt, sampled.q, y);
        verify_mle_eval_mod_q_ligerito_runtime(
            &mut vt,
            &hint.commitment,
            proof,
            &p,
            &sampled.row_weights_q,
            &sampled.col_weights_q,
            alpha,
            y,
            sampled.q,
            q_bits,
            bound_ood,
            &vc,
        )
    };

    // Warm-up prove (excluded from stats).
    {
        let proof = prove_once(&hint);
        black_box(&proof);
    }

    // Timed reps: prove / serialize / deserialize / verify.
    let mut prove_ms = Vec::new();
    let mut verify_ms = Vec::new();
    let mut ser_us = Vec::new();
    let mut de_us = Vec::new();
    let mut bytes = 0usize;
    let mut proof_fnv = 0u64;
    for _ in 0..reps {
        let (proof, t0) =
            bitz::observability::measure(tracing::info_span!("pcs:proof"), || prove_once(&hint))
                .expect("measure completed operation");
        prove_ms.push(t0.as_secs_f64() * 1e3);

        let (ser, t1) =
            bitz::observability::measure(tracing::info_span!("pcs:ser"), || proof.to_bytes())
                .expect("measure completed operation");
        ser_us.push(t1.as_secs_f64() * 1e6);
        bytes = ser.len();
        // FNV-1a over the serialized proof: the byte-identity pin for
        // `BITZ_COL_ELIDE=0` vs `=1` at the same shape and fill.
        proof_fnv = ser.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });

        let (de, t2) = bitz::observability::measure(tracing::info_span!("pcs:de"), || {
            bitz::ligerito_flock::IntEvalRsLigModQProof::from_bytes(&ser).expect("codec")
        })
        .expect("measure completed operation");
        de_us.push(t2.as_secs_f64() * 1e6);
        black_box(&de);

        let t3_recording =
            bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t3 = tracing::info_span!("pcs:t3").entered();
        verify_once(&proof).expect("verify");
        verify_ms.push(
            {
                drop(t3);
                bitz::observability::duration(
                    &t3_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "pcs:t3",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3,
        );
    }

    // Peak + phase split over one prove (heap high-water; the commit hint
    // is live below it). Phase times come from completed Perfetto intervals;
    // trace extraction and querying happen after the heap snapshot.
    // Release flock's cross-prove scratch pool first: the reported prove
    // peak is the production single-prove shape (pool cold), not the
    // reps-warmed pool stacked under the forest. The timed medians above
    // deliberately keep the warm pool — that IS the steady-state timing.
    bitz::ligerito_flock::flock_scratch_clear();
    let recording =
        bitz::observability::Recording::start(Vec::new()).expect("start PCS phase probe");
    reset_peak();

    let split_proof = {
        let proof = prove_once(&hint);
        black_box(&proof);
        proof
    };
    let prove_peak = peak_mb();
    let phases = bitz::observability::totals(&recording.intervals().expect("query PCS phase probe"));

    let prove_median = median(prove_ms);
    let verify_median = median(verify_ms);
    println!("  prove:   {prove_median:8.2} ms   peak {prove_peak:8.2} MB      (median of {reps})");
    let mut forest_split = None;
    if !phases.is_empty() {
        let phase_ms = |labels: &[&str]| -> f64 {
            phases
                .iter()
                .filter(|(l, _)| labels.contains(&l.as_str()))
                .map(|(_, s)| s)
                .sum::<f64>()
                * 1e3
        };
        let forest_ms = phase_ms(&[
            "mc:pack",
            "mc:pow2",
            "mc:forest",
            "mc:fold_v",
            "mc:presum_tbls",
            "mc:presum_run",
        ]);
        let open_ms = phase_ms(&["mq:rings", "mq:bcomb", "mq:lig"]);
        println!(
            "  phases:  forest+presum {forest_ms:8.2} ms | ligerito open {open_ms:7.2} ms   (one profiled prove)"
        );
        forest_split = Some((forest_ms, open_ms));
    }
    println!("  verify:  {verify_median:8.2} ms");
    println!(
        "  proof:   {bytes:8} B ({:.1} KiB)   serialize {:.0} µs / deserialize {:.0} µs   fnv {proof_fnv:016x}",
        bytes as f64 / 1024.0,
        median(ser_us),
        median(de_us)
    );
    // Transmitted-payload accounting split (`mle_eval_mod_q_lig_size_breakdown`):
    // forest side = forest sumchecks/evals + chunk folds + pre-sumchecks;
    // open side = ring-switch `s_v` + the Ligerito proof.
    let (zb, lig_b) = bitz::ligerito_flock::mle_eval_mod_q_lig_size_breakdown(&split_proof);
    let forest_b = zb.total() - zb.s_v;
    let open_b = zb.s_v + lig_b;
    println!(
        "  split:   forest-side {:7.1} KiB | open-side {:7.1} KiB (s_v {:5.1} + lig {:7.1})",
        forest_b as f64 / 1024.0,
        open_b as f64 / 1024.0,
        zb.s_v as f64 / 1024.0,
        lig_b as f64 / 1024.0,
    );

    // Unified RESULT line (docs/bench-schema.md). PCS-only: the whole
    // measured prove is Steps 5.1–5.3, so steps 2/3/4/5.0 are `na` and the
    // end-to-end prover is commit + open. The forest/opener detail is
    // populated from one profiled prove.
    let na = common::StepMedians {
        total: 0.0,
        commit: None,
        project: None,
        piop: None,
        bitify: None,
        reduce: None,
        open: None,
        residual: 0.0,
        outer: None,
        bind: None,
        inner: None,
        forest: None,
        opener: None,
    };
    let report = common::BenchReport {
        bench: "pcs",
        shape: format!("{t}:{s}:{w}"),
        extra: vec![
            common::ligerito_identity(&resolved, ood),
            ("n".into(), n.to_string()),
            ("chunks".into(), lch.to_string()),
            ("lig".into(), lig_tag.clone()),
            ("fill".into(), format!("{fill}")),
        ],
        lambda: None,
        lambda_achieved: None,
        lambda_bind: None,
        threads: common::threads(),
        reps,
        seed: None,
        witness_ms,
        setup_ms,
        prover: common::StepMedians {
            total: commit_ms + prove_median,
            commit: Some(commit_ms),
            open: Some(prove_median),
            forest: forest_split.map(|(forest, _)| forest),
            opener: forest_split.map(|(_, open)| open),
            ..na
        },
        verifier: common::StepMedians {
            total: verify_median,
            open: Some(verify_median),
            ..na
        },
        proof: common::ProofBytes {
            piop: 0,
            open: bytes,
        },
    };
    println!("  {}", report.result_line());

    // ── Extension-field arm (`BITZ_BENCH_EXT`): the SAME committed
    // instance opened at an extension-field statement (default 100-bit
    // projection primes) — the delta vs the base run above is the
    // extension surcharge in the same process/thermal window.
    // `1`/`gl2` = Goldilocks² (e=2, q_bits=64); `bb4` = BabyBear⁴
    // (X⁴ = 11, the Plonky3 challenge field; e=4, q_bits=31). ──
    match env.extension.as_deref() {
        Some("1") | Some("gl2") => bench_ext_arm::<Fp2>(&p, &hint, alpha, reps, ood, &pc, &vc),
        Some("bb4") => bench_ext_arm::<BbFp4>(&p, &hint, alpha, reps, ood, &pc, &vc),
        Some("kb5") => bench_ext_arm::<KbFp5>(&p, &hint, alpha, reps, ood, &pc, &vc),
        Some(_) => unreachable!("clap validates the extension"),
        None => {}
    }
}

/// An evaluation extension field `K = F_q[X]/(h)` for the ext bench arm:
/// the verifier-side ring `R` plus the statement-shaping constants.
trait BenchExtField:
    Copy
    + PartialEq
    + std::fmt::Debug
    + From<u128>
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
{
    /// Extension degree `e = deg(h)`.
    const EXT_DEG: usize;
    /// `⌈log₂ q⌉` of the base characteristic (the ext API's `q_bits`).
    const Q_BITS: usize;
    /// The base characteristic `q`.
    const CHAR: u128;
    const NAME: &'static str;
    /// Element from its canonical coordinate vector (length `EXT_DEG`).
    fn from_coords(c: &[u128]) -> Self;
    /// The module-basis images `[1, X, …, X^{e−1}]`.
    fn basis() -> Vec<Self>;
}

/// Goldilocks p = 2^64 − 2^32 + 1; K = F_p[X]/(X² − 7).
const GL_P: u128 = 0xFFFF_FFFF_0000_0001;

#[derive(Clone, Copy, PartialEq, Debug)]
struct Fp2 {
    c0: u128,
    c1: u128,
}
impl From<u128> for Fp2 {
    fn from(v: u128) -> Self {
        Fp2 {
            c0: v % GL_P,
            c1: 0,
        }
    }
}
impl std::ops::Add for Fp2 {
    type Output = Fp2;
    fn add(self, o: Fp2) -> Fp2 {
        Fp2 {
            c0: (self.c0 + o.c0) % GL_P,
            c1: (self.c1 + o.c1) % GL_P,
        }
    }
}
impl std::ops::Mul for Fp2 {
    type Output = Fp2;
    fn mul(self, o: Fp2) -> Fp2 {
        let m = |a: u128, b: u128| (a * b) % GL_P; // operands < 2^64: exact in u128
        Fp2 {
            c0: (m(self.c0, o.c0) + m(7, m(self.c1, o.c1))) % GL_P,
            c1: (m(self.c0, o.c1) + m(self.c1, o.c0)) % GL_P,
        }
    }
}
impl BenchExtField for Fp2 {
    const EXT_DEG: usize = 2;
    const Q_BITS: usize = 64;
    const CHAR: u128 = GL_P;
    const NAME: &'static str = "Goldilocks²";
    fn from_coords(c: &[u128]) -> Self {
        Fp2 {
            c0: c[0] % GL_P,
            c1: c[1] % GL_P,
        }
    }
    fn basis() -> Vec<Self> {
        vec![Fp2 { c0: 1, c1: 0 }, Fp2 { c0: 0, c1: 1 }]
    }
}

/// BabyBear p = 2^31 − 2^27 + 1; K = F_p[X]/(X⁴ − 11) — the quartic
/// extension Plonky3 samples its BabyBear challenges from.
const BB_P: u128 = 0x7800_0001;

#[derive(Clone, Copy, PartialEq, Debug)]
struct BbFp4([u128; 4]);
impl From<u128> for BbFp4 {
    fn from(v: u128) -> Self {
        BbFp4([v % BB_P, 0, 0, 0])
    }
}
impl std::ops::Add for BbFp4 {
    type Output = BbFp4;
    fn add(self, o: BbFp4) -> BbFp4 {
        BbFp4(std::array::from_fn(|i| (self.0[i] + o.0[i]) % BB_P))
    }
}
impl std::ops::Mul for BbFp4 {
    type Output = BbFp4;
    fn mul(self, o: BbFp4) -> BbFp4 {
        // Schoolbook: coordinates < 2^31, so every partial product is
        // < 2^62 and the 7 convolution sums stay far below 2^128;
        // X⁴ ≡ 11 folds the top back with an ×11 (< 2^68).
        let mut prod = [0u128; 7];
        for i in 0..4 {
            for j in 0..4 {
                prod[i + j] += self.0[i] * o.0[j];
            }
        }
        BbFp4(std::array::from_fn(|k| {
            (prod[k] + 11 * prod.get(k + 4).copied().unwrap_or(0)) % BB_P
        }))
    }
}
impl BenchExtField for BbFp4 {
    const EXT_DEG: usize = 4;
    const Q_BITS: usize = 31;
    const CHAR: u128 = BB_P;
    const NAME: &'static str = "BabyBear⁴";
    fn from_coords(c: &[u128]) -> Self {
        BbFp4(std::array::from_fn(|i| c[i] % BB_P))
    }
    fn basis() -> Vec<Self> {
        (0..4)
            .map(|d| BbFp4(std::array::from_fn(|i| u128::from(i == d))))
            .collect()
    }
}

/// KoalaBear p = 2^31 − 2^24 + 1; a quintic extension (e = 5) — the shape
/// of leanVM's evaluation field. Stand-in reduction X⁵ = 3 (leanVM's
/// actual quintic is non-binomial; the BitZ-side costs depend only on
/// (e, q_bits), which match).
const KB_P: u128 = 0x7F00_0001;

#[derive(Clone, Copy, PartialEq, Debug)]
struct KbFp5([u128; 5]);
impl From<u128> for KbFp5 {
    fn from(v: u128) -> Self {
        let mut c = [0u128; 5];
        c[0] = v % KB_P;
        KbFp5(c)
    }
}
impl std::ops::Add for KbFp5 {
    type Output = KbFp5;
    fn add(self, o: KbFp5) -> KbFp5 {
        KbFp5(std::array::from_fn(|i| (self.0[i] + o.0[i]) % KB_P))
    }
}
impl std::ops::Mul for KbFp5 {
    type Output = KbFp5;
    fn mul(self, o: KbFp5) -> KbFp5 {
        let mut prod = [0u128; 9];
        for i in 0..5 {
            for j in 0..5 {
                prod[i + j] += self.0[i] * o.0[j];
            }
        }
        KbFp5(std::array::from_fn(|k| {
            (prod[k] + 3 * prod.get(k + 5).copied().unwrap_or(0)) % KB_P
        }))
    }
}
impl BenchExtField for KbFp5 {
    const EXT_DEG: usize = 5;
    const Q_BITS: usize = 31;
    const CHAR: u128 = KB_P;
    const NAME: &'static str = "KoalaBear⁵ (stand-in X⁵−3)";
    fn from_coords(c: &[u128]) -> Self {
        KbFp5(std::array::from_fn(|i| c[i] % KB_P))
    }
    fn basis() -> Vec<Self> {
        (0..5)
            .map(|d| KbFp5(std::array::from_fn(|i| u128::from(i == d))))
            .collect()
    }
}

/// The extension-field opening benchmarked against the SAME commitment as
/// the base arm: prove/verify medians, ext phase scopes on one profiled
/// prove AND one profiled verify, proof size + codec times.
fn bench_ext_arm<K: BenchExtField>(
    p: &IntegerMatrixLayout,
    hint: &bitz::ligerito_flock::FlockCommitHint,
    alpha: bitz::Gf128,
    reps: usize,
    ood: Option<OodRoundParams>,
    pc: &LigPc,
    vc: &LigVc,
) {
    use bitz::ligerito_flock::{
        prove_mle_eval_ext_ligerito_with_ood, verify_mle_eval_ext_ligerito_with_ood,
    };
    let q_bits = K::Q_BITS;
    let ext_deg = K::EXT_DEG;
    let proj = bitz::ext_proj::ExtProjParams::default();
    let basis = K::basis();
    let log_w = p.word_bits.trailing_zeros() as usize;
    let w_mask = p.word_bits - 1;

    // Coordinate-major lift of v⁽¹⁾ ∈ K^{2^t} (arbitrary < q), col weights
    // over K, and the claimed μ from the SET BITS of the committed rows.
    let coords: Vec<Vec<u128>> = (0..ext_deg)
        .map(|d| {
            (0..p.rows())
                .map(|b| {
                    (b as u128)
                        .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                        .wrapping_add(d as u128 + 7)
                        % K::CHAR
                })
                .collect()
        })
        .collect();
    let col_w: Vec<K> = (0..p.cols())
        .map(|c| {
            let cs: Vec<u128> = (0..ext_deg)
                .map(|d| {
                    (c as u128)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((d as u128) << 40)
                        % K::CHAR
                })
                .collect();
            K::from_coords(&cs)
        })
        .collect();
    let v1: Vec<K> = (0..p.rows())
        .map(|b| {
            let cs: Vec<u128> = (0..ext_deg).map(|d| coords[d][b]).collect();
            K::from_coords(&cs)
        })
        .collect();
    let mut y = K::from(0u128);
    for (c, row) in hint.rows().iter().enumerate() {
        let mut acc = K::from(0u128);
        for (wi, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let i = (wi << 6) | bit;
                let (b, j) = (i >> log_w, i & w_mask);
                let term = if j == 0 {
                    v1[b]
                } else {
                    v1[b] * K::from(1u128 << j)
                };
                acc = acc + term;
            }
        }
        y = y + col_w[c] * acc;
    }

    // Warm-up (excluded).
    {
        let mut pt = bitz::transcript::Blake3Transcript::new();
        let proof = prove_mle_eval_ext_ligerito_with_ood(
            &mut pt, hint, p, &coords, q_bits, &proj, alpha, ood, pc,
        )
        .expect("bounded projection prime search");
        black_box(&proof);
    }

    let mut prove_ms = Vec::new();
    let mut verify_ms = Vec::new();
    let mut ser_us = Vec::new();
    let mut de_us = Vec::new();
    let mut bytes = 0usize;
    for _ in 0..reps {
        let mut pt = bitz::transcript::Blake3Transcript::new();
        let (proof, t0) = bitz::observability::measure(tracing::info_span!("pcs:proof"), || {
            prove_mle_eval_ext_ligerito_with_ood(
                &mut pt, hint, p, &coords, q_bits, &proj, alpha, ood, pc,
            )
            .expect("bounded projection prime search")
        })
        .expect("measure completed operation");
        prove_ms.push(t0.as_secs_f64() * 1e3);

        let (ser, t1) =
            bitz::observability::measure(tracing::info_span!("pcs:ser"), || proof.to_bytes())
                .expect("measure completed operation");
        ser_us.push(t1.as_secs_f64() * 1e6);
        bytes = ser.len();
        let (de, t2) = bitz::observability::measure(tracing::info_span!("pcs:de"), || {
            bitz::ligerito_flock::IntEvalRsLigExtProof::from_bytes(&ser).expect("codec")
        })
        .expect("measure completed operation");
        de_us.push(t2.as_secs_f64() * 1e6);
        black_box(&de);

        let mut vt = bitz::transcript::Blake3Transcript::new();
        let t3_recording =
            bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t3 = tracing::info_span!("pcs:t3").entered();
        verify_mle_eval_ext_ligerito_with_ood(
            &mut vt,
            &hint.commitment,
            &proof,
            p,
            &coords,
            &col_w,
            &basis,
            alpha,
            y,
            q_bits,
            &proj,
            ood,
            vc,
        )
        .expect("ext verify");
        verify_ms.push(
            {
                drop(t3);
                bitz::observability::duration(
                    &t3_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "pcs:t3",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3,
        );
    }

    // Ext phase scopes over one profiled prove + one profiled verify.
    let recording =
        bitz::observability::Recording::start(Vec::new()).expect("start ext phase probe");

    let proof = {
        let mut pt = bitz::transcript::Blake3Transcript::new();
        let proof = prove_mle_eval_ext_ligerito_with_ood(
            &mut pt, hint, p, &coords, q_bits, &proj, alpha, ood, pc,
        )
        .expect("bounded projection prime search");
        black_box(&proof);
        proof
    };
    let prove_phases =
        bitz::observability::totals(&recording.intervals().expect("query ext prove probe"));
    let recording =
        bitz::observability::Recording::start(Vec::new()).expect("start ext verify probe");
    {
        let mut vt = bitz::transcript::Blake3Transcript::new();
        verify_mle_eval_ext_ligerito_with_ood(
            &mut vt,
            &hint.commitment,
            &proof,
            p,
            &coords,
            &col_w,
            &basis,
            alpha,
            y,
            q_bits,
            &proj,
            ood,
            vc,
        )
        .expect("ext verify (profiled)");
    }
    let verify_phases =
        bitz::observability::totals(&recording.intervals().expect("query ext verify probe"));
    let pick = |phases: &[(String, f64)], label: &str| -> f64 {
        phases
            .iter()
            .filter(|(l, _)| *l == label)
            .map(|(_, s)| s)
            .sum::<f64>()
            * 1e3
    };

    println!(
        "  ext(K={}, e={ext_deg}, q_bits={q_bits}, q'={}b):",
        K::NAME,
        proj.prime_bits
    );
    println!(
        "    prove:  {:8.2} ms   (median of {reps})",
        median(prove_ms)
    );
    if !prove_phases.is_empty() {
        println!(
            "    p-phases: step1_folds {:7.2} ms | sample_prime {:6.2} ms | project {:6.2} ms",
            pick(&prove_phases, "ext:step1_folds"),
            pick(&prove_phases, "ext:sample_prime"),
            pick(&prove_phases, "ext:project"),
        );
    }
    println!("    verify: {:8.2} ms", median(verify_ms));
    if !verify_phases.is_empty() {
        println!(
            "    v-phases: sample_prime {:6.2} ms | project {:6.2} ms | checks {:6.2} ms",
            pick(&verify_phases, "ext:sample_prime"),
            pick(&verify_phases, "ext:project"),
            pick(&verify_phases, "ext:checks"),
        );
    }
    println!(
        "    proof:  {bytes:8} B ({:.1} KiB)   serialize {:.0} µs / deserialize {:.0} µs",
        bytes as f64 / 1024.0,
        median(ser_us),
        median(de_us)
    );
}

fn main() {
    common::cli::EnvironmentCli::parse();
    let env: Env = common::cli::environment();
    let reps = common::reps(None, 5);
    bitz::observability::install().expect("install Perfetto subscriber");
    common::enforce_known_env();
    if std::env::var_os("BITZ_BENCH_LAMBDA").is_some() {
        common::warn(
            "BITZ_BENCH_LAMBDA is ignored by the PCS-only bench (no IOP security \
             profile here; the RESULT line reports lambda=na)",
        );
    }
    println!("BitZ PCS bench — commit/prove/verify + serialized size + peak heap per shape.");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    println!("(target: aarch64 + neon — the NEON GF(2^128) pipeline is active)");

    for &(t, s, w) in &env.shapes {
        bench_shape(t, s, w, reps, &env);
    }
}
