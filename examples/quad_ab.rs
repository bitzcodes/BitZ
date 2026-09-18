//! Historical kernel experiment; no production security claim.
//! Paired in-process A/B: base (arity-2) forest vs `BITZ_QUAD=2` (bottom-merge
//! quad plan) prover times at one shape. `BITZ_QUAD` is read per call (NOT
//! process-cached), so flipping it between proves inside one process is valid
//! — unlike the `BITZ_EQF_*` family. Alternates arm order every pair to cancel
//! the order artifact; prints per-pair times and medians.
//!
//! ```text
//! AB_SHAPE=17:11 AB_PAIRS=9 RUSTFLAGS="-C target-cpu=native" \
//!   cargo run --release --example quad_ab --features unchecked,span-metrics
//! ```


use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{commit_rs_ligerito_rows, prove_mle_eval_mod_q_ligerito, historical_sha_lig_configs};
use bitz::pcs::{IntegerMatrixLayout, smallest_generator};

const Q: u128 = (1u128 << 100) - 15;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    let alpha = smallest_generator();
    let q_bits = 100usize;
    let shape = std::env::var("AB_SHAPE").unwrap_or_else(|_| "17:11".into());
    let mut it = shape.split(':');
    let t: usize = it.next().unwrap().parse().unwrap();
    let s: usize = it.next().unwrap().parse().unwrap();
    let pairs: usize =
        std::env::var("AB_PAIRS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);

    let p = IntegerMatrixLayout {
        row_vars: t,
        col_vars: s,
        word_bits: 1,
    };
    let m_p = packed_vars(&p);
    let (pc, _vc) = historical_sha_lig_configs(m_p).expect("lig cfg");
    let cell = |b: usize, c: usize| -> u128 {
        (p.cell_index(b, c) as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15) & 1
    };
    let words = p.rows().div_ceil(64);
    let rows: Vec<Vec<u64>> = (0..p.cols())
        .map(|c| {
            let mut wv = vec![0u64; words];
            for b in 0..p.rows() {
                if cell(b, c) & 1 == 1 {
                    wv[b >> 6] |= 1u64 << (b & 63);
                }
            }
            wv
        })
        .collect();
    let hint = commit_rs_ligerito_rows(&p, rows, &pc);
    let rw_q: Vec<u128> = (0..p.rows())
        .map(|b| {
            (b as u128)
                .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                .wrapping_add(7)
                % Q
        })
        .collect();

    let prove_arm = |quad: bool| -> (f64, usize) {
        if quad {
            unsafe { std::env::set_var("BITZ_QUAD", "2") };
        } else {
            unsafe { std::env::remove_var("BITZ_QUAD") };
        }
        let mut pt = bitz::transcript::Blake3Transcript::new();
        let (proof, t0) = bitz::observability::measure(
            tracing::info_span!("quad_ab:proof"),
            || prove_mle_eval_mod_q_ligerito(&mut pt, &hint, &p, &rw_q, q_bits, alpha, &pc),
        ).expect("measure completed operation");
        let ms = t0.as_secs_f64() * 1e3;
        (ms, proof.to_bytes().len())
    };

    // Warm-up: one per arm (discarded).
    let _ = prove_arm(false);
    let _ = prove_arm(true);

    let (mut base, mut quad) = (Vec::new(), Vec::new());
    let mut bytes = (0usize, 0usize);
    for i in 0..pairs {
        // Alternate order to cancel the order artifact.
        let (first_quad, second_quad) = (i % 2 == 1, i % 2 == 0);
        let (m1, b1) = prove_arm(first_quad);
        let (m2, b2) = prove_arm(second_quad);
        let (bm, qm) = if first_quad { (m2, m1) } else { (m1, m2) };
        let (bb, qb) = if first_quad { (b2, b1) } else { (b1, b2) };
        base.push(bm);
        quad.push(qm);
        bytes = (bb, qb);
        println!(
            "  pair {i}: base {bm:8.2} ms | quad2 {qm:8.2} ms | delta {:+6.2}%",
            (qm / bm - 1.0) * 100.0
        );
    }
    let (bmed, qmed) = (median(base), median(quad));
    println!(
        "n={} base {bmed:.2} ms | quad2 {qmed:.2} ms | delta {:+.2}% | bytes {} -> {} ({:+} B)",
        t + s,
        (qmed / bmed - 1.0) * 100.0,
        bytes.0,
        bytes.1,
        bytes.1 as i64 - bytes.0 as i64
    );
}
