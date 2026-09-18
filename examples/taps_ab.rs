//! Historical kernel experiment; no production security claim.
//! Structured-taps A/B (EXPERIMENTAL, docs/rlc-structured-taps-phase0.md;
//! corrected semantics: 32-bit words along the ENTRY axis of W=1
//! bit-vectors, g = 5): the j = 2, k = 6 ROT/SHIFT/word-offset instance
//! proved four ways against ONE commitment and statement —
//!
//! * `single` — one identity claim alone via the batched tap-claims path
//!   (the cost unit);
//! * `tapf`   — the clustered stream family (`{b1,b3,b5}` over 6 streams
//!   and `{b2,b4,b6}` over 7; two eager case forests, per-cluster
//!   cascades, translated-eq openings, ONE proof);
//! * `vx6`    — all six claims through the batched tap-claims path (ONE
//!   8·2^s-tree padded forest + per-claim translated-eq openings);
//! * `ind6`   — six independent single-claim tap proofs.
//!
//! Alternated in-window reps, medians reported (repo measurement
//! protocol; small-n caveat: FS grinding luck is deterministic per
//! statement — vary `BITZ_TAPS_SEED` and average over statements before
//! reading small deltas). Verifies every variant once. With
//! `OBLONG_PROFILE=1` one extra profiled prove of `tapf` and `vx6` dumps
//! the phase tree per shape. `BITZ_AB_NO_FAMILY=1` skips the stream
//! family (its eager case forests are the memory hog — this unlocks the
//! n = 28 row for the single/vx6/ind6 comparison).
//!
//! `BITZ_AB_COLLAPSE=1` runs the SINGLE-TAP shared-point demo instead:
//! the instance's 13 deduped streams as 13 individual claims at ONE
//! point — `clp` (the weight-transform collapse, ≤ 4 inner claims) vs
//! `vx13` (13 batched tap claims, pads to 16 tree-sets) vs `ind13`.
//!
//! `BITZ_AB_MIX6=1` runs the uniform-op k = 6 variant of the original
//! instance at ONE point: 2 identity claims on the source columns plus
//! 4 claims `off^t(ROT^7(a₀ ⊕ a₁))`, t = 0..3 — the op stays OUTSIDE
//! the XOR, so the whole set is 0x44 territory (4 plain inner bodies,
//! no rings) — `clp` (the collapse) vs `vx6m` (the blocked batched tap
//! path on the composed lists) vs `ind6m`.
//!
//! `BITZ_AB_COLS4=1` runs a FOUR-column config (log_cols = 2): identity
//! claims on a₁..a₄ plus the XOR-mixed pairs `ROT¹(a₁) ⊕ off¹(a₂)` and
//! `ROT²(a₃) ⊕ off¹(a₄)`, one shared point, through the blocked
//! batched tap path — `single` vs `vx6` vs `ind6` (δ applies: the
//! sources are unconstrained on the 0x42 route).
//!
//! `BITZ_AB_BLAKE3=1` runs the Blake3 design model
//! (docs/blake3-taps-design.md): 8 committed columns, ONE commitment,
//! the 8-claim P-LIN layer (0x46 multiweight collapse, per-claim
//! column weights, δ = 0) + the 5-shape P-XOR layer (blocked 0x42 at
//! the δ knee) vs `vx13` (all 13 claims batched, collapse-unaware).
//!
//! `BITZ_AB_SCHED=1` runs the COMPOSED-collapse schedule demo instead:
//! 48 claims `off^t(x)` of ONE σ-style mixed combination
//! `x = ROT^7 a_0 ⊕ ROT^18 a_0 ⊕ SHIFT^3 a_0 ⊕ off^1 a_1` at ONE point
//! — `cmp` (the composed collapse, 2 inner tap bodies TOTAL) vs `vx48`
//! (the batched tap-claims path on the offset-folded lists; 24 blocked
//! 2-set forests since the block cap landed) vs `ind48`. Claim values
//! are computed through the FOLDED-list extraction route, so every
//! verified rep doubles as a distributed-extraction cross-check of the
//! weight-transform algebra. `BITZ_AB_ROUNDS` overrides the round count.
//!
//! ```text
//! BITZ_AB_N="22 24 26" BITZ_AB_REPS=5 RUSTFLAGS="-C target-cpu=native" \
//!   cargo run --release --example taps_ab --features unchecked,span-metrics
//! ```

use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{
    RlcFamilyClaim, TapClaim, TapComposedClaim, TapFamilyCluster, TapPointClaim, TapVerifyClaim,
    commit_rs_ligerito_rows, historical_sha_lig_configs,
    mle_eval_mod_q_lig_tap_family_size_breakdown, mle_eval_mod_q_lig_tap_size_breakdown,
    mle_eval_mod_q_lig_xor_proof_size_bytes, prove_mle_eval_mod_q_ligerito_tap_claims,
    prove_mle_eval_mod_q_ligerito_tap_collapse, prove_mle_eval_mod_q_ligerito_tap_composed,
    prove_mle_eval_mod_q_ligerito_tap_family, verify_mle_eval_mod_q_ligerito_tap_claims,
    verify_mle_eval_mod_q_ligerito_tap_collapse, verify_mle_eval_mod_q_ligerito_tap_composed,
    verify_mle_eval_mod_q_ligerito_tap_family,
};
use bitz::pcs::{
    FQ_BITS, FQ_MOD, IntegerMatrixLayout, Q100Element, ShaF2Layout, smallest_generator,
    virtual_xor_params,
};
use bitz::taps::{TapOp, extract_virtual_tap_rows};
use bitz::transcript::Blake3Transcript;

/// The instance word-group width (log2 bits per entry-axis word):
/// 32-bit words by default; `BITZ_TAPS_GRP=6` runs the SAME instances on
/// 64-bit words. Per-body cost is width-independent (the translated-eq
/// chains are O(g), the class count is fixed), so only the layout
/// envelope moves: `s >= g + 2` and the delta ceiling = g.
#[allow(non_snake_case)]
fn GRP() -> usize {
    static G: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        let g = std::env::var("BITZ_TAPS_GRP").map_or(5, |v| v.parse().unwrap());
        assert!((1..=8).contains(&g), "BITZ_TAPS_GRP must be in 1..=8");
        g
    })
}

/// n → the taps layout: 2 UAIR bit-columns (log_cols = 1, W = 1,
/// bit_vars = 0) over 2^{n−1} entries; x split t' vs s as even as
/// `tw ≥ 6` allows (the 32-bit group field lives in the clear axis:
/// s ≥ g + 2 so word offsets ≤ 2 stay in range). `BITZ_TAPS_DELTA`
/// sets `x_fold_extra` (the sched/vx paths only; δ ≤ g so collapse
/// outers with `2^δ | amt` stay in envelope — the schedule's pure-off
/// outers always are; the stream family requires δ = 0).
fn taps_layout(n: usize, log_cols: usize) -> ShaF2Layout {
    let tw = ((n - log_cols) / 2).max(6);
    let s = n - log_cols - tw;
    assert!(
        s >= GRP() + 2,
        "clear axis must hold the group field plus offsets"
    );
    let delta: usize = std::env::var("BITZ_TAPS_DELTA").map_or(0, |v| v.parse().unwrap());
    assert!(delta <= GRP(), "BITZ_TAPS_DELTA must be ≤ g = {}", GRP());
    ShaF2Layout {
        p: IntegerMatrixLayout {
            row_vars: log_cols + tw,
            col_vars: s,
            word_bits: 1,
        },
        num_cols: 1 << log_cols,
        log_cols,
        bit_vars: 0,
        num_vars: tw + s,
        tw,
        x_fold_extra: delta,
    }
}

/// The k = 6 instance's tap lists (32-bit entry-axis words): identities,
/// two three-tap single-column rotation convolutions, a cross-column
/// mix, and the lossy-SHIFT claim — the pinned spec of the session
/// prompt under the corrected semantics.
fn instance_claims() -> Vec<Vec<TapOp>> {
    let rot = |col, amt, off| TapOp {
        col,
        grp_log2: GRP(),
        bit_amt: amt,
        bit_dropout: false,
        off,
    };
    let shl = |col, amt, off| TapOp {
        col,
        grp_log2: GRP(),
        bit_amt: amt,
        bit_dropout: true,
        off,
    };
    vec![
        vec![TapOp::ident(0)],
        vec![TapOp::ident(1)],
        vec![rot(0, 1, 0), rot(0, 2, 1), rot(0, 3, 2)],
        vec![rot(1, 2, 0), rot(1, 5, 1), rot(1, 7, 2)],
        vec![rot(0, 1, 0), rot(1, 4, 0), rot(0, 6, 1)],
        vec![shl(0, 3, 0), shl(1, 5, 1), rot(1, 2, 2)],
    ]
}

/// The pinned two-cluster split: `{b1, b3, b5}` (6 streams, S1 shared) and
/// `{b2, b4, b6}` (7 streams).
#[allow(clippy::type_complexity)]
fn instance_clusters() -> (Vec<Vec<TapOp>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let rot = |col, amt, off| TapOp {
        col,
        grp_log2: GRP(),
        bit_amt: amt,
        bit_dropout: false,
        off,
    };
    let shl = |col, amt, off| TapOp {
        col,
        grp_log2: GRP(),
        bit_amt: amt,
        bit_dropout: true,
        off,
    };
    let streams1 = vec![
        TapOp::ident(0),
        rot(0, 1, 0),
        rot(0, 2, 1),
        rot(0, 3, 2),
        rot(1, 4, 0),
        rot(0, 6, 1),
    ];
    let streams2 = vec![
        TapOp::ident(1),
        rot(1, 2, 0),
        rot(1, 5, 1),
        rot(1, 7, 2),
        shl(0, 3, 0),
        shl(1, 5, 1),
        rot(1, 2, 2),
    ];
    let forms1 = vec![0b000001usize, 0b001110, 0b110010];
    let forms2 = vec![0b0000001usize, 0b0001110, 0b1110000];
    (
        vec![streams1, streams2],
        vec![forms1, forms2],
        vec![vec![0, 2, 4], vec![1, 3, 5]],
    )
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    let alpha = smallest_generator();
    let ns: Vec<usize> = std::env::var("BITZ_AB_N")
        .map(|v| v.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![22, 24]);
    let reps: usize = std::env::var("BITZ_AB_REPS").map_or(5, |v| v.parse().unwrap());
    let seed: u64 = std::env::var("BITZ_TAPS_SEED").map_or(1, |v| v.parse().unwrap());
    let profile = std::env::var("OBLONG_PROFILE").is_ok_and(|v| v == "1");

    for &n in &ns {
        let layout = taps_layout(n, 1);
        let p = &layout.p;
        let p_x = virtual_xor_params(&layout);
        let m_p = packed_vars(p);
        let (pc, vc) = historical_sha_lig_configs(m_p).expect("lig cfg");

        let words = p.rows().div_ceil(64);
        let rows: Vec<Vec<u64>> = (0..p.cols())
            .map(|c| {
                (0..words)
                    .map(|w| {
                        ((c as u64) << 32 | (w as u64) ^ seed)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .rotate_left(((c + w) & 63) as u32)
                    })
                    .collect()
            })
            .collect();
        let hint = commit_rs_ligerito_rows(p, rows, &pc);

        if std::env::var("BITZ_AB_OPEN8").is_ok_and(|v| v == "1") {
            // Pure MLE-opening measurement (no relation checks): 8
            // committed columns, 8 identity openings at ONE shared
            // point through a single 0x44 collapse sub-proof (8 plain
            // bodies, one Ligerito tail), at the δ knee (identity ops
            // are envelope-free). `single` = one opening, the marginal
            // anchor; compare externally against the base prover's
            // one-claim floor (`bitz <n> --profile fast`).
            let mut layout = taps_layout(n, 3);
            layout.x_fold_extra = std::env::var("BITZ_AB_OPEN_DELTA")
                .map_or(4, |v| v.parse().unwrap())
                .min(layout.p.col_vars - 1);
            let p = &layout.p;
            let p_x = virtual_xor_params(&layout);
            let (pc, vc) = historical_sha_lig_configs(packed_vars(p)).expect("lig cfg");
            let words = p.rows().div_ceil(64);
            let rows: Vec<Vec<u64>> = (0..p.cols())
                .map(|c| {
                    (0..words)
                        .map(|w| {
                            ((c as u64) << 32 | (w as u64) ^ seed)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                .rotate_left(((c + w) & 63) as u32)
                        })
                        .collect()
                })
                .collect();
            let hint = commit_rs_ligerito_rows(p, rows, &pc);
            let rw: Vec<u128> = (0..p_x.rows())
                .map(|b| {
                    (b as u128)
                        .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                        .wrapping_add(113 + (u128::from(seed) << 1))
                        % FQ_MOD
                })
                .collect();
            let colw: Vec<Q100Element> = (0..p_x.cols())
                .map(|c| {
                    Q100Element::from((c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3))
                })
                .collect();
            let sets: Vec<Vec<usize>> = (0..8).map(|c| vec![c]).collect();
            let pclaims: Vec<TapPointClaim<'_>> = sets
                .iter()
                .map(|set| {
                    let taps = [TapOp::ident(set[0])];
                    let a_rows = extract_virtual_tap_rows(&layout, hint.rows(), &taps);
                    let mut y = Q100Element::from(0u128);
                    for (c, row) in a_rows.iter().enumerate() {
                        let mut acc = Q100Element::from(0u128);
                        for (wi, &word) in row.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                let t = bits.trailing_zeros() as usize;
                                acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                                bits &= bits.wrapping_sub(1);
                            }
                        }
                        y = y + colw[c] * acc;
                    }
                    TapPointClaim {
                        cols: set,
                        op: bitz::taps::TapUniOp::ident(),
                        claimed: y.canonical_u128(),
                    }
                })
                .collect();
            let prove_k = |k: usize| {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut t,
                    &hint,
                    &layout,
                    &rw,
                    &colw,
                    &pclaims[..k],
                    alpha,
                    &pc,
                )
            };
            let (mut t_all, mut t_one) = (Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_k(8));
                t_all.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_k(1));
                t_one.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let proof8 = prove_k(8);
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut vt,
                    &hint.commitment,
                    &proof8,
                    &layout,
                    &rw,
                    &colw,
                    &pclaims,
                    alpha,
                    &vc,
                )
                .expect("open8 verifies");
            }
            let v8 = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let (m8, m1) = (median(t_all), median(t_one));
            println!(
                "n={n} OPEN8 δ{} (8 identity openings, one 0x44 sub-proof, {} bodies): \
                 {m8:.1} ms ({:.0} KB, verify {v8:.1} ms, {:.1} ms/opening marginal) | \
                 single {m1:.1} ms",
                layout.x_fold_extra,
                proof8.xors.len(),
                mle_eval_mod_q_lig_xor_proof_size_bytes(&proof8) as f64 / 1e3,
                (m8 - m1) / 7.0,
            );
            continue;
        }

        if std::env::var("BITZ_AB_B3OPEN").is_ok_and(|v| v == "1") {
            // Openings-ONLY b3 comparison (no relation checks — the
            // outer protocol owns constraints): commit SIX columns
            // (a,b,c,d,a',c'; d',b' virtual), open all eight — v(d'),
            // v(b') as the pair-XOR forms at the ROT-relabeled point.
            // Arms: `vx8` = ONE 0x44 sub-proof, 8 plain bodies (6
            // idents + the ident-op XOR sets {d,a'},{b,c'} — no rings,
            // δ knee); `fam6` = two j2 families + one 0x44 for {a},{c}
            // (3 sub-proofs); `fam4` = two j3 families absorbing a, c
            // (2 sub-proofs, δ0 — δ4 regresses j3).
            use bitz::ligerito_flock::{
                RlcFamilySpec, RlcSharedClaim, mle_eval_mod_q_lig_rlc_families_proof_size_bytes,
                prove_mle_eval_mod_q_ligerito_rlc_families_shared_point,
                prove_mle_eval_mod_q_ligerito_rlc_family_shared_point,
                verify_mle_eval_mod_q_ligerito_rlc_families_shared_point,
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point,
            };
            let od: usize = std::env::var("BITZ_AB_OPEN_DELTA").map_or(4, |v| v.parse().unwrap());
            let fd: usize = std::env::var("BITZ_AB_FAM_DELTA").map_or(4, |v| v.parse().unwrap());
            let mut layout = taps_layout(n, 3);
            layout.x_fold_extra = od.min(layout.p.col_vars - 1);
            let mut layout_fam = taps_layout(n, 3);
            layout_fam.x_fold_extra = fd.min(layout_fam.p.col_vars - 1);
            let mut layout_f4 = taps_layout(n, 3);
            layout_f4.x_fold_extra = 0;
            let p = &layout.p;
            let (pc, vc) = historical_sha_lig_configs(packed_vars(p)).expect("lig cfg");
            let words = p.rows().div_ceil(64);
            let rows: Vec<Vec<u64>> = (0..p.cols())
                .map(|c| {
                    (0..words)
                        .map(|w| {
                            ((c as u64) << 32 | (w as u64) ^ seed)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                .rotate_left(((c + w) & 63) as u32)
                        })
                        .collect()
                })
                .collect();
            let hint = commit_rs_ligerito_rows(p, rows, &pc);
            let eval_at = |layout_e: &ShaF2Layout,
                           cols: &[usize],
                           rw: &[u128],
                           colw: &[Q100Element]|
             -> u128 {
                let taps: Vec<TapOp> = cols.iter().map(|&c| TapOp::ident(c)).collect();
                let a_rows = extract_virtual_tap_rows(layout_e, hint.rows(), &taps);
                let mut y = Q100Element::from(0u128);
                for (c, row) in a_rows.iter().enumerate() {
                    let mut acc = Q100Element::from(0u128);
                    for (wi, &word) in row.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros() as usize;
                            acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                            bits &= bits.wrapping_sub(1);
                        }
                    }
                    y = y + colw[c] * acc;
                }
                y.canonical_u128()
            };
            let mk_rw = |len: usize, salt: u128| -> Vec<u128> {
                (0..len)
                    .map(|x| {
                        (x as u128)
                            .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                            .wrapping_add(salt + (u128::from(seed) << 1))
                            % FQ_MOD
                    })
                    .collect()
            };
            let mk_colw = |len: usize| -> Vec<Q100Element> {
                (0..len)
                    .map(|c| {
                        Q100Element::from(
                            (c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3),
                        )
                    })
                    .collect()
            };
            // vx8: one 0x44 with 8 plain claims (identity ops only).
            let p_x = virtual_xor_params(&layout);
            let rw = mk_rw(p_x.rows(), 127);
            let colw = mk_colw(p_x.cols());
            let sets: Vec<Vec<usize>> = vec![
                vec![0],
                vec![1],
                vec![2],
                vec![3],
                vec![4],
                vec![5],
                vec![3, 4],
                vec![1, 5],
            ];
            let pclaims: Vec<TapPointClaim<'_>> = sets
                .iter()
                .map(|set| TapPointClaim {
                    cols: set,
                    op: bitz::taps::TapUniOp::ident(),
                    claimed: eval_at(&layout, set, &rw, &colw),
                })
                .collect();
            // Families (fam6 at δ = fd; fam4 at δ0), same forms as the
            // full-family arms minus nothing — openings are the forms.
            let p_xf = virtual_xor_params(&layout_fam);
            let rwf = mk_rw(p_xf.rows(), 131);
            let colwf = mk_colw(p_xf.cols());
            let p_x4 = virtual_xor_params(&layout_f4);
            let rw4 = mk_rw(p_x4.rows(), 137);
            let colw4 = mk_colw(p_x4.cols());
            let fam1_cols = [3usize, 4, 0];
            let fam2_cols = [1usize, 5, 2];
            let mk_fam = |layout_e: &ShaF2Layout,
                          rw_e: &[u128],
                          colw_e: &[Q100Element],
                          cols: &[usize],
                          j3: bool|
             -> Vec<RlcSharedClaim> {
                let mut v = vec![
                    RlcSharedClaim {
                        form: 0b001,
                        claimed: eval_at(layout_e, &cols[..1], rw_e, colw_e),
                    },
                    RlcSharedClaim {
                        form: 0b010,
                        claimed: eval_at(layout_e, &cols[1..2], rw_e, colw_e),
                    },
                    RlcSharedClaim {
                        form: 0b011,
                        claimed: eval_at(layout_e, &cols[..2], rw_e, colw_e),
                    },
                ];
                if j3 {
                    v.push(RlcSharedClaim {
                        form: 0b100,
                        claimed: eval_at(layout_e, &cols[2..3], rw_e, colw_e),
                    });
                }
                v
            };
            let f1_j2 = mk_fam(&layout_fam, &rwf, &colwf, &fam1_cols, false);
            let f2_j2 = mk_fam(&layout_fam, &rwf, &colwf, &fam2_cols, false);
            let f1_j3 = mk_fam(&layout_f4, &rw4, &colw4, &fam1_cols, true);
            let f2_j3 = mk_fam(&layout_f4, &rw4, &colw4, &fam2_cols, true);
            let ac_sets: Vec<Vec<usize>> = vec![vec![0], vec![2]];
            let ac_pclaims: Vec<TapPointClaim<'_>> = ac_sets
                .iter()
                .map(|set| TapPointClaim {
                    cols: set,
                    op: bitz::taps::TapUniOp::ident(),
                    claimed: eval_at(&layout, set, &rw, &colw),
                })
                .collect();
            // fam6m: the MERGED proof — {d,a'} j2 + {b,c'} j2 + {a} j1
            // + {c} j1 in ONE transcript with ONE closing Ligerito call.
            let f3_cols = [0usize];
            let f4_cols = [2usize];
            let f3 = vec![RlcSharedClaim {
                form: 0b1,
                claimed: eval_at(&layout_fam, &f3_cols, &rwf, &colwf),
            }];
            let f4 = vec![RlcSharedClaim {
                form: 0b1,
                claimed: eval_at(&layout_fam, &f4_cols, &rwf, &colwf),
            }];
            let m_specs = [
                RlcFamilySpec {
                    family_cols: &fam1_cols[..2],
                    claims: &f1_j2,
                },
                RlcFamilySpec {
                    family_cols: &fam2_cols[..2],
                    claims: &f2_j2,
                },
                RlcFamilySpec {
                    family_cols: &f3_cols,
                    claims: &f3,
                },
                RlcFamilySpec {
                    family_cols: &f4_cols,
                    claims: &f4,
                },
            ];
            let prove_f6m = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_rlc_families_shared_point(
                    &mut t,
                    &hint,
                    &layout_fam,
                    &m_specs,
                    &rwf,
                    alpha,
                    &pc,
                )
            };
            let prove_vx8 = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut t, &hint, &layout, &rw, &colw, &pclaims, alpha, &pc,
                )
            };
            let prove_f6 = || {
                let mut t1 = Blake3Transcript::new();
                let a = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut t1,
                    &hint,
                    &layout_fam,
                    &fam1_cols[..2],
                    &rwf,
                    &f1_j2,
                    alpha,
                    &pc,
                );
                let mut t2 = Blake3Transcript::new();
                let b = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut t2,
                    &hint,
                    &layout_fam,
                    &fam2_cols[..2],
                    &rwf,
                    &f2_j2,
                    alpha,
                    &pc,
                );
                let mut t3 = Blake3Transcript::new();
                let c = prove_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut t3,
                    &hint,
                    &layout,
                    &rw,
                    &colw,
                    &ac_pclaims,
                    alpha,
                    &pc,
                );
                (a, b, c)
            };
            let prove_f4 = || {
                let mut t1 = Blake3Transcript::new();
                let a = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut t1,
                    &hint,
                    &layout_f4,
                    &fam1_cols[..],
                    &rw4,
                    &f1_j3,
                    alpha,
                    &pc,
                );
                let mut t2 = Blake3Transcript::new();
                let b = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut t2,
                    &hint,
                    &layout_f4,
                    &fam2_cols[..],
                    &rw4,
                    &f2_j3,
                    alpha,
                    &pc,
                );
                (a, b)
            };
            let (mut t_vx, mut t_f6, mut t_f4, mut t_f6m) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_vx8());
                t_vx.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_f6());
                t_f6.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_f4());
                t_f4.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_f6m());
                t_f6m.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let pv = prove_vx8();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut vt,
                    &hint.commitment,
                    &pv,
                    &layout,
                    &rw,
                    &colw,
                    &pclaims,
                    alpha,
                    &vc,
                )
                .expect("b3open vx8 verifies");
            }
            let v_vx = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let pf6 = prove_f6();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut vt,
                    &hint.commitment,
                    &pf6.0,
                    &layout_fam,
                    &fam1_cols[..2],
                    &rwf,
                    &f1_j2,
                    &colwf,
                    alpha,
                    &vc,
                )
                .expect("b3open fam6/1 verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut vt,
                    &hint.commitment,
                    &pf6.1,
                    &layout_fam,
                    &fam2_cols[..2],
                    &rwf,
                    &f2_j2,
                    &colwf,
                    alpha,
                    &vc,
                )
                .expect("b3open fam6/2 verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut vt,
                    &hint.commitment,
                    &pf6.2,
                    &layout,
                    &rw,
                    &colw,
                    &ac_pclaims,
                    alpha,
                    &vc,
                )
                .expect("b3open fam6/ac verifies");
            }
            let v_f6 = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let pf4 = prove_f4();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut vt,
                    &hint.commitment,
                    &pf4.0,
                    &layout_f4,
                    &fam1_cols[..],
                    &rw4,
                    &f1_j3,
                    &colw4,
                    alpha,
                    &vc,
                )
                .expect("b3open fam4/1 verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut vt,
                    &hint.commitment,
                    &pf4.1,
                    &layout_f4,
                    &fam2_cols[..],
                    &rw4,
                    &f2_j3,
                    &colw4,
                    alpha,
                    &vc,
                )
                .expect("b3open fam4/2 verifies");
            }
            let v_f4 = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let pf6m = prove_f6m();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_families_shared_point(
                    &mut vt,
                    &hint.commitment,
                    &pf6m,
                    &layout_fam,
                    &m_specs,
                    &rwf,
                    &colwf,
                    alpha,
                    &vc,
                )
                .expect("b3open fam6m verifies");
            }
            let v_f6m = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let fam_sz = bitz::ligerito_flock::mle_eval_mod_q_lig_rlc_family_proof_size_bytes;
            let sz_vx = mle_eval_mod_q_lig_xor_proof_size_bytes(&pv);
            let sz_f6 =
                fam_sz(&pf6.0) + fam_sz(&pf6.1) + mle_eval_mod_q_lig_xor_proof_size_bytes(&pf6.2);
            let sz_f4 = fam_sz(&pf4.0) + fam_sz(&pf4.1);
            let (m_vx, m_f6, m_f4, m_f6m) =
                (median(t_vx), median(t_f6), median(t_f4), median(t_f6m));
            println!(
                "n={n} B3OPEN (8 openings, NO checks; 6 cols committed, v(d')/v(b') = \
                 relabeled pair forms): vx8 δ{} {m_vx:.1} ms (8 bodies 1 tail, {:.0} KB, \
                 verify {v_vx:.1} ms) | fam6 δ{} {m_f6:.1} ms ({:.2}x, {:.0} KB, verify \
                 {v_f6:.1} ms) | fam4 δ0 {m_f4:.1} ms ({:.2}x, {:.0} KB, verify {v_f4:.1} ms)",
                layout.x_fold_extra,
                sz_vx as f64 / 1e3,
                layout_fam.x_fold_extra,
                m_f6 / m_vx,
                sz_f6 as f64 / 1e3,
                m_f4 / m_vx,
                sz_f4 as f64 / 1e3,
            );
            println!(
                "  fam6m δ{} (MERGED: 2×j2 + 2×j1, ONE tail): {m_f6m:.1} ms ({:.2}x of \
                 vx8, {:.0} KB, verify {v_f6m:.1} ms)",
                layout_fam.x_fold_extra,
                m_f6m / m_vx,
                mle_eval_mod_q_lig_rlc_families_proof_size_bytes(&pf6m) as f64 / 1e3,
            );
            continue;
        }

        if std::env::var("BITZ_AB_B3FAM").is_ok_and(|v| v == "1") {
            // The user's b3 family: 8 role vectors a,b,c,d,a',b',c',d'
            // with d' = ROT16(d⊕a'), b' = ROT12(b⊕c'),
            // off1(d) = ROT8(d'⊕off1(a)), off1(b) = ROT7(b'⊕off1(c));
            // MLE openings wanted for all eight. OPT arm: commit SIX
            // (d', b' virtual — relations 1,2 become definitions), all
            // 8 openings through ONE 0x44 collapse (identity sets + the
            // ROT-of-XOR sets {d,a'}, {b,c'} — 8 plain bodies, no
            // rings, δ = 2 by the 2^δ | 12 envelope) + the two COMPOSED
            // zero-checks R1 = off1(d)⊕ROT24(d)⊕ROT24(a')⊕ROT8 off1(a),
            // R2 = off1(b)⊕ROT19(b)⊕ROT19(c')⊕ROT7 off1(c) through
            // blocked 0x42 (2 mixed bodies, word-0 boundary row masked
            // by the zero-check weights). NAIVE arm: commit all 8, open
            // all 8 (0x44, δ = 4 — identities are envelope-free), check
            // the four relations as 3-tap mixed vectors (4 bodies).
            let g = GRP();
            assert!(g >= 5, "b3 family rotation amounts need g ≥ 5");
            let layout = taps_layout(n, 3);
            let mut layout_plain = taps_layout(n, 3);
            // ROT16 passes the δ-envelope at δ = 4 (2^4 | 16); ROT12
            // does not (8 ∤ 12), so the b'-opening moves to the 0x42
            // layer (envelope-free) and the plain layer keeps δ = 4 —
            // the δ2 alternative measured slower AND fatter (the plain
            // fold vectors quadruple).
            layout_plain.x_fold_extra = 4.min(taps_layout(n, 3).p.col_vars - 1);
            let mut layout_pl_naive = taps_layout(n, 3);
            layout_pl_naive.x_fold_extra = 4.min(layout.p.col_vars - 1);
            let p = &layout.p;
            let (pc, vc) = historical_sha_lig_configs(packed_vars(p)).expect("lig cfg");
            // Semantic generation: random a, a', c, c' and top words;
            // d, b run the BACKWARD recurrence d[j] = ROT24(d[j+1]) ⊕
            // ROT24(a'[j+1]) ⊕ ROT8(a[j]) (resp. ROT19/ROT7), so
            // relations (3),(4) hold at every word k ≥ 1; d', b' by
            // their definitions. Words are 2^g bits (g = 5: u32 lanes
            // of a u64; keep u64 and mask).
            let wbits = 1usize << g;
            let wmask: u64 = if wbits == 64 { !0 } else { (1u64 << wbits) - 1 };
            let rotl = |x: u64, r: usize| -> u64 {
                if r == 0 {
                    x
                } else {
                    ((x << r) | (x >> (wbits - r))) & wmask
                }
            };
            let nwords = 1usize << (layout.num_vars - g);
            let mut rng_state = 0x9E37_79B9_7F4A_7C15u64 ^ (seed << 7) ^ (n as u64);
            let mut rng = move || -> u64 {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                rng_state & wmask
            };
            let aw: Vec<u64> = (0..nwords).map(|_| rng()).collect();
            let apw: Vec<u64> = (0..nwords).map(|_| rng()).collect();
            let cw: Vec<u64> = (0..nwords).map(|_| rng()).collect();
            let cpw: Vec<u64> = (0..nwords).map(|_| rng()).collect();
            let mut dw = vec![0u64; nwords];
            let mut bw = vec![0u64; nwords];
            dw[nwords - 1] = rng();
            bw[nwords - 1] = rng();
            for j in (0..nwords - 1).rev() {
                dw[j] = rotl(dw[j + 1] ^ apw[j + 1], 24 % wbits) ^ rotl(aw[j], 8);
                bw[j] = rotl(bw[j + 1] ^ cpw[j + 1], 19 % wbits) ^ rotl(cw[j], 7);
            }
            let dpw: Vec<u64> = (0..nwords)
                .map(|k| rotl(dw[k] ^ apw[k], 16 % wbits))
                .collect();
            let bpw: Vec<u64> = (0..nwords).map(|k| rotl(bw[k] ^ cpw[k], 12)).collect();
            // Self-check of the family algebra: (3),(4) hold for k ≥ 1.
            for k in 1..nwords {
                assert_eq!(
                    dw[k - 1],
                    rotl(dpw[k] ^ aw[k - 1], 8),
                    "relation (3) at {k}"
                );
                assert_eq!(
                    bw[k - 1],
                    rotl(bpw[k] ^ cw[k - 1], 7),
                    "relation (4) at {k}"
                );
            }
            // Pack per-column words into committed clear rows: trace
            // p = (k ≪ g)|j of column col lands in row p & (2^s − 1),
            // bit (col ≪ tw) | (p ≫ s).
            let s_bits = p.col_vars;
            let tw = layout.tw;
            let row_words = (1usize << p.row_vars).div_ceil(64);
            let pack = |cols: &[&[u64]]| -> Vec<Vec<u64>> {
                let mut rows = vec![vec![0u64; row_words]; 1usize << s_bits];
                for (col, wds) in cols.iter().enumerate() {
                    for (k, &w) in wds.iter().enumerate() {
                        for j in 0..wbits {
                            if (w >> j) & 1 == 1 {
                                let tr = (k << g) | j;
                                let bit = (col << tw) | (tr >> s_bits);
                                rows[tr & ((1 << s_bits) - 1)][bit >> 6] |= 1u64 << (bit & 63);
                            }
                        }
                    }
                }
                rows
            };
            // OPT commitment: a,b,c,d,a',c' (cols 0..5; d',b' virtual).
            let hint = commit_rs_ligerito_rows(p, pack(&[&aw, &bw, &cw, &dw, &apw, &cpw]), &pc);
            // NAIVE commitment: all eight (d' = 6, b' = 7).
            let hint_nv = commit_rs_ligerito_rows(
                p,
                pack(&[&aw, &bw, &cw, &dw, &apw, &cpw, &dpw, &bpw]),
                &pc,
            );
            let rot = |col, amt: usize, off| TapOp {
                col,
                grp_log2: g,
                bit_amt: amt % wbits,
                bit_dropout: false,
                off,
            };
            let uni = |amt: usize, off: usize| bitz::taps::TapUniOp {
                grp_log2: g,
                bit_amt: amt % wbits,
                bit_dropout: false,
                off,
            };
            let mk_rw = |len: usize, salt: u128| -> Vec<u128> {
                (0..len)
                    .map(|x| {
                        (x as u128)
                            .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                            .wrapping_add(salt + (u128::from(seed) << 1))
                            % FQ_MOD
                    })
                    .collect()
            };
            let mk_colw = |len: usize| -> Vec<Q100Element> {
                (0..len)
                    .map(|c| {
                        Q100Element::from(
                            (c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3),
                        )
                    })
                    .collect()
            };
            let eval_with = |layout_e: &ShaF2Layout,
                             hint_e: &bitz::ligerito_flock::FlockCommitHint,
                             taps: &[TapOp],
                             rw: &[u128],
                             colw: &[Q100Element]|
             -> u128 {
                let a_rows = extract_virtual_tap_rows(layout_e, hint_e.rows(), taps);
                let mut y = Q100Element::from(0u128);
                for (c, row) in a_rows.iter().enumerate() {
                    let mut acc = Q100Element::from(0u128);
                    for (wi, &word) in row.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros() as usize;
                            acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                            bits &= bits.wrapping_sub(1);
                        }
                    }
                    y = y + colw[c] * acc;
                }
                y.canonical_u128()
            };
            // ── OPT plain layer (0x44, δ = 2): 6 identities + the two
            // ROT-of-XOR openings v(d') on {d,a'}, v(b') on {b,c'}.
            let p_x2 = virtual_xor_params(&layout_plain);
            let rw2 = mk_rw(p_x2.rows(), 83);
            let colw2 = mk_colw(p_x2.cols());
            let opt_sets: Vec<(Vec<usize>, bitz::taps::TapUniOp)> = vec![
                (vec![0], uni(0, 0)),
                (vec![1], uni(0, 0)),
                (vec![2], uni(0, 0)),
                (vec![3], uni(0, 0)),
                (vec![4], uni(0, 0)),
                (vec![5], uni(0, 0)),
                (vec![3, 4], uni(16, 0)),
            ];
            let opt_pclaims: Vec<TapPointClaim<'_>> = opt_sets
                .iter()
                .map(|(set, op)| {
                    let taps: Vec<TapOp> = set.iter().map(|&c| op.with_col(c)).collect();
                    TapPointClaim {
                        cols: set,
                        op: *op,
                        claimed: eval_with(&layout_plain, &hint, &taps, &rw2, &colw2),
                    }
                })
                .collect();
            // ── OPT mixed layer (0x42, env δ): the composed zero-checks,
            // word-0 masked out by the column weights.
            let p_xm = virtual_xor_params(&layout);
            let rwm = mk_rw(p_xm.rows(), 97);
            let colwm = mk_colw(p_xm.cols());
            let delta = layout.x_fold_extra;
            let mut colw_mask = colwm.clone();
            for c in colw_mask.iter_mut().take(wbits >> delta) {
                *c = Q100Element::from(0u128);
            }
            let r1 = vec![rot(3, 0, 1), rot(3, 24, 0), rot(4, 24, 0), rot(0, 8, 1)];
            let r2 = vec![rot(1, 0, 1), rot(1, 19, 0), rot(5, 19, 0), rot(2, 7, 1)];
            // v(b') = the ROT12-of-XOR opening, routed through 0x42.
            let bp_open = vec![rot(1, 12, 0), rot(5, 12, 0)];
            let bp_val = eval_with(&layout, &hint, &bp_open, &rwm, &colwm);
            for taps in [&r1, &r2] {
                assert_eq!(
                    eval_with(&layout, &hint, taps, &rwm, &colw_mask),
                    0,
                    "composed relation vector must vanish off word 0"
                );
            }
            let opt_mixed: Vec<TapClaim<'_>> = vec![
                TapClaim {
                    taps: &r1,
                    row_weights_q: &rwm,
                },
                TapClaim {
                    taps: &r2,
                    row_weights_q: &rwm,
                },
                TapClaim {
                    taps: &bp_open,
                    row_weights_q: &rwm,
                },
            ];
            let opt_mixed_v: Vec<TapVerifyClaim<'_, Q100Element>> = [&r1, &r2]
                .iter()
                .map(|taps| TapVerifyClaim {
                    taps,
                    row_weights_q: &rwm,
                    col_weights: &colw_mask,
                    claimed: Q100Element::from(0u128),
                })
                .chain(std::iter::once(TapVerifyClaim {
                    taps: &bp_open,
                    row_weights_q: &rwm,
                    col_weights: &colwm,
                    claimed: Q100Element::from(bp_val),
                }))
                .collect();
            // ── NAIVE arms: 8 identity openings (δ = 4) + 4 relation
            // vectors (relations 1,2 unmasked; 3,4 word-0 masked).
            let p_x4 = virtual_xor_params(&layout_pl_naive);
            let rw4 = mk_rw(p_x4.rows(), 89);
            let colw4 = mk_colw(p_x4.cols());
            let nv_sets: Vec<Vec<usize>> = (0..8).map(|c| vec![c]).collect();
            let nv_pclaims: Vec<TapPointClaim<'_>> = nv_sets
                .iter()
                .map(|set| {
                    let taps = [TapOp::ident(set[0])];
                    TapPointClaim {
                        cols: set,
                        op: bitz::taps::TapUniOp::ident(),
                        claimed: eval_with(&layout_pl_naive, &hint_nv, &taps, &rw4, &colw4),
                    }
                })
                .collect();
            let r1n = vec![TapOp::ident(6), rot(3, 16, 0), rot(4, 16, 0)];
            let r2n = vec![TapOp::ident(7), rot(1, 12, 0), rot(5, 12, 0)];
            let r3n = vec![rot(3, 0, 1), rot(6, 8, 0), rot(0, 8, 1)];
            let r4n = vec![rot(1, 0, 1), rot(7, 7, 0), rot(2, 7, 1)];
            for (taps, mask) in [(&r1n, false), (&r2n, false), (&r3n, true), (&r4n, true)] {
                let cwm = if mask { &colw_mask } else { &colwm };
                assert_eq!(
                    eval_with(&layout, &hint_nv, taps, &rwm, cwm),
                    0,
                    "naive relation vector must vanish"
                );
            }
            let nv_lists = [&r1n, &r2n, &r3n, &r4n];
            let nv_mixed: Vec<TapClaim<'_>> = nv_lists
                .iter()
                .map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rwm,
                })
                .collect();
            let nv_mixed_v: Vec<TapVerifyClaim<'_, Q100Element>> = nv_lists
                .iter()
                .zip([false, false, true, true])
                .map(|(taps, mask)| TapVerifyClaim {
                    taps,
                    row_weights_q: &rwm,
                    col_weights: if mask { &colw_mask } else { &colwm },
                    claimed: Q100Element::from(0u128),
                })
                .collect();
            // ── FAMILY arms: the opening layer through shared-point
            // RLC families (δ = 0 — the family paths' regime), with
            // v(d') and v(b') delivered as the pair-XOR form claims at
            // the PUBLICLY ROT-TRANSFORMED point (an opening of d⊕a'
            // IS an opening of d' = ROT16(d⊕a') at the relabeled point
            // — zero cost). fam6: two j=2 families {d,a'}, {b,c'}
            // (3 forms each) + the {a},{c} idents via 0x44 (δ4) + the
            // 2 zero-checks = 6 bodies. fam4: two j=3 families
            // {d,a',a}, {b,c',c} (4 forms each; a,c absorbed) + the 2
            // zero-checks = 4 bodies, at 4 AND channels per family.
            use bitz::ligerito_flock::{
                RlcSharedClaim, prove_mle_eval_mod_q_ligerito_rlc_family_shared_point,
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point,
            };
            let layout_f0 = taps_layout(n, 3); // δ = env applies to R-checks only
            // Family δ: BITZ_AB_FAM_DELTA (default 0). The RLC-family
            // core is p_x-parameterized end to end (extraction, folds,
            // forests, presum, discharge, rings all derive their shapes
            // from virtual_xor_params), so the re-split flows through.
            let mut layout_fam = taps_layout(n, 3);
            layout_fam.x_fold_extra =
                std::env::var("BITZ_AB_FAM_DELTA").map_or(0, |v| v.parse().unwrap());
            let p_xf = virtual_xor_params(&layout_fam);
            let rwf = mk_rw(p_xf.rows(), 101);
            let colwf = mk_colw(p_xf.cols());
            let _ = &layout_f0;
            let fam_val = |cols: &[usize]| -> u128 {
                let taps: Vec<TapOp> = cols.iter().map(|&c| TapOp::ident(c)).collect();
                eval_with(&layout_fam, &hint, &taps, &rwf, &colwf)
            };
            // Families: cols [d, a', a] / [b, c', c]; forms are bitmasks
            // over the FAMILY column list.
            let fam1_cols = [3usize, 4, 0];
            let fam2_cols = [1usize, 5, 2];
            let fam_claims = |cols: &[usize], j3: bool| -> Vec<RlcSharedClaim> {
                let mut v = vec![
                    RlcSharedClaim {
                        form: 0b001,
                        claimed: fam_val(&cols[..1]),
                    },
                    RlcSharedClaim {
                        form: 0b010,
                        claimed: fam_val(&cols[1..2]),
                    },
                    RlcSharedClaim {
                        form: 0b011,
                        claimed: fam_val(&cols[..2]),
                    },
                ];
                if j3 {
                    v.push(RlcSharedClaim {
                        form: 0b100,
                        claimed: fam_val(&cols[2..3]),
                    });
                }
                v
            };
            let f1_j2 = fam_claims(&fam1_cols, false);
            let f2_j2 = fam_claims(&fam2_cols, false);
            let f1_j3 = fam_claims(&fam1_cols, true);
            let f2_j3 = fam_claims(&fam2_cols, true);
            // The {a},{c} idents for fam6 (0x44 at δ4).
            let ac_sets: Vec<(Vec<usize>, bitz::taps::TapUniOp)> =
                vec![(vec![0], uni(0, 0)), (vec![2], uni(0, 0))];
            let ac_pclaims: Vec<TapPointClaim<'_>> = ac_sets
                .iter()
                .map(|(set, op)| {
                    let taps = [TapOp::ident(set[0])];
                    TapPointClaim {
                        cols: set,
                        op: *op,
                        claimed: eval_with(&layout_plain, &hint, &taps, &rw2, &colw2),
                    }
                })
                .collect();
            let rcheck_mixed: Vec<TapClaim<'_>> = vec![
                TapClaim {
                    taps: &r1,
                    row_weights_q: &rwm,
                },
                TapClaim {
                    taps: &r2,
                    row_weights_q: &rwm,
                },
            ];
            let rcheck_mixed_v: Vec<TapVerifyClaim<'_, Q100Element>> = [&r1, &r2]
                .iter()
                .map(|taps| TapVerifyClaim {
                    taps,
                    row_weights_q: &rwm,
                    col_weights: &colw_mask,
                    claimed: Q100Element::from(0u128),
                })
                .collect();
            let prove_fam = |j3: bool| {
                let (c1, c2) = if j3 {
                    (&f1_j3, &f2_j3)
                } else {
                    (&f1_j2, &f2_j2)
                };
                let mut t1 = Blake3Transcript::new();
                let fam1 = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut t1,
                    &hint,
                    &layout_fam,
                    if j3 { &fam1_cols[..] } else { &fam1_cols[..2] },
                    &rwf,
                    c1,
                    alpha,
                    &pc,
                );
                let mut t2 = Blake3Transcript::new();
                let fam2 = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut t2,
                    &hint,
                    &layout_fam,
                    if j3 { &fam2_cols[..] } else { &fam2_cols[..2] },
                    &rwf,
                    c2,
                    alpha,
                    &pc,
                );
                let plain = (!j3).then(|| {
                    let mut t = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_tap_collapse(
                        &mut t,
                        &hint,
                        &layout_plain,
                        &rw2,
                        &colw2,
                        &ac_pclaims,
                        alpha,
                        &pc,
                    )
                });
                let mut t3 = Blake3Transcript::new();
                let mixed = prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t3,
                    &hint,
                    &layout,
                    FQ_BITS,
                    &rcheck_mixed,
                    alpha,
                    &pc,
                );
                (fam1, fam2, plain, mixed)
            };
            let prove_opt = || {
                let mut t1 = Blake3Transcript::new();
                let plain = prove_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut t1,
                    &hint,
                    &layout_plain,
                    &rw2,
                    &colw2,
                    &opt_pclaims,
                    alpha,
                    &pc,
                );
                let mut t2 = Blake3Transcript::new();
                let mixed = prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t2, &hint, &layout, FQ_BITS, &opt_mixed, alpha, &pc,
                );
                (plain, mixed)
            };
            let prove_nv = || {
                let mut t1 = Blake3Transcript::new();
                let plain = prove_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut t1,
                    &hint_nv,
                    &layout_pl_naive,
                    &rw4,
                    &colw4,
                    &nv_pclaims,
                    alpha,
                    &pc,
                );
                let mut t2 = Blake3Transcript::new();
                let mixed = prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t2, &hint_nv, &layout, FQ_BITS, &nv_mixed, alpha, &pc,
                );
                (plain, mixed)
            };
            let (mut t_opt, mut t_nv, mut t_f6, mut t_f4) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_opt());
                t_opt.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_nv());
                t_nv.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_fam(false));
                t_f6.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_fam(true));
                t_f4.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let (opt_plain, opt_mx) = prove_opt();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut vt,
                    &hint.commitment,
                    &opt_plain,
                    &layout_plain,
                    &rw2,
                    &colw2,
                    &opt_pclaims,
                    alpha,
                    &vc,
                )
                .expect("b3fam opt plain verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &opt_mx,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &opt_mixed_v,
                    &vc,
                )
                .expect("b3fam opt mixed verifies");
            }
            let v_opt = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let (nv_plain, nv_mx) = prove_nv();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut vt,
                    &hint_nv.commitment,
                    &nv_plain,
                    &layout_pl_naive,
                    &rw4,
                    &colw4,
                    &nv_pclaims,
                    alpha,
                    &vc,
                )
                .expect("b3fam naive plain verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint_nv.commitment,
                    &nv_mx,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &nv_mixed_v,
                    &vc,
                )
                .expect("b3fam naive mixed verifies");
            }
            let v_nv = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            // Verify + size the family arms once.
            let verify_fam = |j3: bool,
                              pr: &(
                bitz::ligerito_flock::IntEvalRsLigRlcFamilyProof,
                bitz::ligerito_flock::IntEvalRsLigRlcFamilyProof,
                Option<bitz::ligerito_flock::IntEvalRsLigModQXorProof>,
                bitz::ligerito_flock::IntEvalRsLigModQTapProof,
            )| {
                let (c1, c2) = if j3 {
                    (&f1_j3, &f2_j3)
                } else {
                    (&f1_j2, &f2_j2)
                };
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut vt,
                    &hint.commitment,
                    &pr.0,
                    &layout_fam,
                    if j3 { &fam1_cols[..] } else { &fam1_cols[..2] },
                    &rwf,
                    c1,
                    &colwf,
                    alpha,
                    &vc,
                )
                .expect("b3fam family1 verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                    &mut vt,
                    &hint.commitment,
                    &pr.1,
                    &layout_fam,
                    if j3 { &fam2_cols[..] } else { &fam2_cols[..2] },
                    &rwf,
                    c2,
                    &colwf,
                    alpha,
                    &vc,
                )
                .expect("b3fam family2 verifies");
                if let Some(plain) = &pr.2 {
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_tap_collapse(
                        &mut vt,
                        &hint.commitment,
                        plain,
                        &layout_plain,
                        &rw2,
                        &colw2,
                        &ac_pclaims,
                        alpha,
                        &vc,
                    )
                    .expect("b3fam ac idents verify");
                }
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &pr.3,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &rcheck_mixed_v,
                    &vc,
                )
                .expect("b3fam fam mixed verifies");
            };
            let pf6 = prove_fam(false);
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            verify_fam(false, &pf6);
            let v_f6 = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let pf4 = prove_fam(true);
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            verify_fam(true, &pf4);
            let v_f4 = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let size_tap = |pr: &bitz::ligerito_flock::IntEvalRsLigModQTapProof| {
                let (b, lig) = mle_eval_mod_q_lig_tap_size_breakdown(pr);
                b.total() + lig
            };
            let size_fam = |pr: &(
                bitz::ligerito_flock::IntEvalRsLigRlcFamilyProof,
                bitz::ligerito_flock::IntEvalRsLigRlcFamilyProof,
                Option<bitz::ligerito_flock::IntEvalRsLigModQXorProof>,
                bitz::ligerito_flock::IntEvalRsLigModQTapProof,
            )| {
                bitz::ligerito_flock::mle_eval_mod_q_lig_rlc_family_proof_size_bytes(&pr.0)
                    + bitz::ligerito_flock::mle_eval_mod_q_lig_rlc_family_proof_size_bytes(&pr.1)
                    + pr.2
                        .as_ref()
                        .map_or(0, mle_eval_mod_q_lig_xor_proof_size_bytes)
                    + size_tap(&pr.3)
            };
            let sz_opt = mle_eval_mod_q_lig_xor_proof_size_bytes(&opt_plain) + size_tap(&opt_mx);
            let sz_nv = mle_eval_mod_q_lig_xor_proof_size_bytes(&nv_plain) + size_tap(&nv_mx);
            let (m_opt, m_nv, m_f6, m_f4) =
                (median(t_opt), median(t_nv), median(t_f6), median(t_f4));
            println!(
                "n={n} B3FAM g={g} (8 openings + 4 relations): opt {m_opt:.1} ms ({}+{} \
                 bodies, 6 cols committed, {:.0} KB, verify {v_opt:.1} ms) | naive \
                 {m_nv:.1} ms ({:.2}x of opt, {}+{} bodies, 8 cols, {:.0} KB, verify \
                 {v_nv:.1} ms)",
                opt_plain.xors.len(),
                opt_mx.tap_us.len(),
                sz_opt as f64 / 1e3,
                m_nv / m_opt,
                nv_plain.xors.len(),
                nv_mx.tap_us.len(),
                sz_nv as f64 / 1e3,
            );
            println!(
                "  fam6 (2×j2 fams + 2 idents + 2 checks = 6 bodies): {m_f6:.1} ms \
                 ({:.2}x of opt, {:.0} KB, verify {v_f6:.1} ms) | fam4 (2×j3 fams + 2 \
                 checks = 4 bodies): {m_f4:.1} ms ({:.2}x, {:.0} KB, verify {v_f4:.1} ms) \
                 | v(d'),v(b') = pair-XOR forms at the ROT-relabeled point",
                m_f6 / m_opt,
                size_fam(&pf6) as f64 / 1e3,
                m_f4 / m_opt,
                size_fam(&pf4) as f64 / 1e3,
            );
            continue;
        }

        if std::env::var("BITZ_AB_BLAKE3").is_ok_and(|v| v == "1") {
            // The Blake3 design model (docs/blake3-taps-design.md): 8
            // committed columns (A,B,C,D roles + K carries + X aux + 2
            // spare), ONE commitment, two sub-proofs at one shared row
            // point — the P-LIN layer (8 weighted single-tap claims →
            // 8 plain bodies via the 0x46 multiweight collapse, δ = 0:
            // intra-block weight patterns stay column-side) and the
            // P-XOR layer (the 4 xor-rot step shapes + the
            // finalization/chaining shape → 5 mixed bodies via blocked
            // 0x42 at the δ knee). Baseline `vx13`: all 13 claims
            // through the batched tap path (collapse-unaware routing).
            use bitz::ligerito_flock::{
                TapWeightedClaim, prove_mle_eval_mod_q_ligerito_tap_multiweight,
                verify_mle_eval_mod_q_ligerito_tap_multiweight,
            };
            let layout = taps_layout(n, 3);
            // The plain layer folds the WHOLE bit-position field
            // (δ = g = 5): every P-LIN read shares one j-profile (the
            // place values 2^j the add-checks want), so per-claim
            // variation is word-granular = column-side at δ = 5, and
            // the row weights stay shared. The fold vectors shrink 32×.
            let mut layout_plain = taps_layout(n, 3);
            layout_plain.x_fold_extra = GRP();
            let p = &layout.p;
            let p_x = virtual_xor_params(&layout);
            let p_x0 = virtual_xor_params(&layout_plain);
            let (pc, vc) = historical_sha_lig_configs(packed_vars(p)).expect("lig cfg");
            let words = p.rows().div_ceil(64);
            let rows: Vec<Vec<u64>> = (0..p.cols())
                .map(|c| {
                    (0..words)
                        .map(|w| {
                            ((c as u64) << 32 | (w as u64) ^ seed)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                .rotate_left(((c + w) & 63) as u32)
                        })
                        .collect()
                })
                .collect();
            // ONE commitment serves both layers: the commitment depends
            // on p only; x_fold_extra is a claim-path parameter.
            let hint = commit_rs_ligerito_rows(p, rows, &pc);
            let rot = |col, amt, off| TapOp {
                col,
                grp_log2: GRP(),
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let uni = |amt: usize, off: usize| bitz::taps::TapUniOp {
                grp_log2: GRP(),
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let mk_rw = |rows_len: usize, salt: u128| -> Vec<u128> {
                (0..rows_len)
                    .map(|b| {
                        (b as u128)
                            .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                            .wrapping_add(salt + (u128::from(seed) << 1))
                            % FQ_MOD
                    })
                    .collect()
            };
            let mk_colw = |cols_len: usize| -> [Vec<Q100Element>; 4] {
                let w_eq: Vec<Q100Element> = (0..cols_len)
                    .map(|c| {
                        Q100Element::from(
                            (c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3),
                        )
                    })
                    .collect();
                let w_pv: Vec<Q100Element> = (0..cols_len)
                    .map(|c| Q100Element::from(1u128 << (c % 24)))
                    .collect();
                let w_mask: Vec<Q100Element> = (0..cols_len)
                    .map(|c| {
                        if c % 2 == 0 {
                            w_eq[c]
                        } else {
                            Q100Element::from(0u128)
                        }
                    })
                    .collect();
                let w_perm: Vec<Q100Element> =
                    (0..cols_len).map(|c| w_eq[(c + 5) % cols_len]).collect();
                [w_eq, w_pv, w_mask, w_perm]
            };
            // ── The P-LIN layer (δ = 0): roles A=0 B=1 C=2 D=3, K=4,
            // X=5; identity/off¹ reads at per-claim weight patterns
            // (eq, place-value, parity-mask, permuted — the add-check,
            // carry, and schedule read shapes).
            let rw0 = mk_rw(p_x0.rows(), 67);
            let [w_eq0, w_pv0, w_mask0, w_perm0] = mk_colw(p_x0.cols());
            // Two reads per (column, branch) — the a-add, c-add,
            // boundary, and schedule relations each read the same
            // columns at different weight patterns; the collapse merges
            // all 16 reads into 8 bodies.
            let plain_spec: Vec<(Vec<usize>, bitz::taps::TapUniOp, &[Q100Element])> = vec![
                (vec![0], uni(0, 0), &w_eq0),
                (vec![0], uni(0, 0), &w_pv0),
                (vec![0], uni(0, 1), &w_eq0),
                (vec![0], uni(0, 1), &w_mask0),
                (vec![1], uni(0, 0), &w_pv0),
                (vec![1], uni(0, 0), &w_eq0),
                (vec![2], uni(0, 0), &w_eq0),
                (vec![2], uni(0, 0), &w_perm0),
                (vec![2], uni(0, 1), &w_mask0),
                (vec![2], uni(0, 1), &w_pv0),
                (vec![3], uni(0, 0), &w_pv0),
                (vec![3], uni(0, 0), &w_mask0),
                (vec![4], uni(0, 0), &w_mask0),
                (vec![4], uni(0, 0), &w_eq0),
                (vec![5], uni(0, 0), &w_perm0),
                (vec![5], uni(0, 0), &w_eq0),
            ];
            let eval_with = |layout_e: &ShaF2Layout,
                             taps: &[TapOp],
                             rw: &[u128],
                             colw: &[Q100Element]|
             -> u128 {
                let a_rows = extract_virtual_tap_rows(layout_e, hint.rows(), taps);
                let mut y = Q100Element::from(0u128);
                for (c, row) in a_rows.iter().enumerate() {
                    let mut acc = Q100Element::from(0u128);
                    for (wi, &word) in row.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros() as usize;
                            acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                            bits &= bits.wrapping_sub(1);
                        }
                    }
                    y = y + colw[c] * acc;
                }
                y.canonical_u128()
            };
            let plain_claims: Vec<TapWeightedClaim<'_>> = plain_spec
                .iter()
                .map(|(set, op, cw)| {
                    let taps: Vec<TapOp> = set.iter().map(|&c| op.with_col(c)).collect();
                    TapWeightedClaim {
                        cols: set,
                        op: *op,
                        col_weights: cw,
                        claimed: eval_with(&layout_plain, &taps, &rw0, cw),
                    }
                })
                .collect();
            // ── The P-XOR layer (δ from BITZ_TAPS_DELTA): the four
            // xor-rot step shapes + the finalization/chaining shape.
            let rw_x = mk_rw(p_x.rows(), 71);
            let [w_eqx, _, _, _] = mk_colw(p_x.cols());
            let mixed_lists: Vec<Vec<TapOp>> = vec![
                vec![TapOp::ident(3), rot(3, 16, 1), rot(0, 16, 0)],
                vec![TapOp::ident(1), rot(1, 12, 1), rot(2, 12, 0)],
                vec![TapOp::ident(3), rot(3, 8, 1), rot(0, 8, 0)],
                vec![TapOp::ident(1), rot(1, 7, 1), rot(2, 7, 0)],
                vec![TapOp::ident(5), rot(5, 0, 2)],
            ];
            let mixed_vals: Vec<u128> = mixed_lists
                .iter()
                .map(|taps| eval_with(&layout, taps, &rw_x, &w_eqx))
                .collect();
            let mixed_claims: Vec<TapClaim<'_>> = mixed_lists
                .iter()
                .map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rw_x,
                })
                .collect();
            let mixed_vclaims: Vec<TapVerifyClaim<'_, Q100Element>> = mixed_lists
                .iter()
                .zip(mixed_vals.iter())
                .map(|(taps, &v)| TapVerifyClaim {
                    taps,
                    row_weights_q: &rw_x,
                    col_weights: &w_eqx,
                    claimed: Q100Element::from(v),
                })
                .collect();
            // ── Baseline: all 13 claims through the batched tap path
            // (per-claim weights are legal there; one body per claim).
            let plain_lists: Vec<Vec<TapOp>> = plain_spec
                .iter()
                .map(|(set, op, _)| set.iter().map(|&c| op.with_col(c)).collect())
                .collect();
            let base_claims: Vec<TapClaim<'_>> = plain_lists
                .iter()
                .map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rw_x,
                })
                .chain(mixed_lists.iter().map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rw_x,
                }))
                .collect();
            let base_vclaims: Vec<TapVerifyClaim<'_, Q100Element>> = plain_lists
                .iter()
                .zip(plain_spec.iter())
                .map(|(taps, (_, _, _))| (taps, &w_eqx))
                .chain(mixed_lists.iter().map(|taps| (taps, &w_eqx)))
                .map(|(taps, cw)| TapVerifyClaim {
                    taps,
                    row_weights_q: &rw_x,
                    col_weights: cw,
                    claimed: Q100Element::from(eval_with(&layout, taps, &rw_x, cw)),
                })
                .collect();
            let prove_b3 = || {
                let mut t1 = Blake3Transcript::new();
                let plain = prove_mle_eval_mod_q_ligerito_tap_multiweight(
                    &mut t1,
                    &hint,
                    &layout_plain,
                    &rw0,
                    &plain_claims,
                    alpha,
                    &pc,
                );
                let mut t2 = Blake3Transcript::new();
                let mixed = prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t2,
                    &hint,
                    &layout,
                    FQ_BITS,
                    &mixed_claims,
                    alpha,
                    &pc,
                );
                (plain, mixed)
            };
            let prove_vx = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t,
                    &hint,
                    &layout,
                    FQ_BITS,
                    &base_claims,
                    alpha,
                    &pc,
                )
            };
            let (mut t_b3, mut t_vx) = (Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_b3());
                t_b3.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_vx());
                t_vx.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let (proof_plain, proof_mixed) = prove_b3();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_multiweight(
                    &mut vt,
                    &hint.commitment,
                    &proof_plain,
                    &layout_plain,
                    &rw0,
                    &plain_claims,
                    alpha,
                    &vc,
                )
                .expect("blake3 plain layer verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &proof_mixed,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &mixed_vclaims,
                    &vc,
                )
                .expect("blake3 mixed layer verifies");
            }
            let v_b3 = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let proof_vx = prove_vx();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &proof_vx,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &base_vclaims,
                    &vc,
                )
                .expect("blake3 vx13 verifies");
            }
            let v_vx = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let size_tap = |p: &bitz::ligerito_flock::IntEvalRsLigModQTapProof| {
                let (b, lig) = mle_eval_mod_q_lig_tap_size_breakdown(p);
                b.total() + lig
            };
            let sz_b3 =
                mle_eval_mod_q_lig_xor_proof_size_bytes(&proof_plain) + size_tap(&proof_mixed);
            let (m_b3, m_vx) = (median(t_b3), median(t_vx));
            println!(
                "n={n} BLAKE3 (8 cols; {}-read P-LIN → {} plain bodies δ{} + 5-shape P-XOR → \
                 {} mixed bodies δ{}): b3 {m_b3:.1} ms ({:.0} KB, verify {v_b3:.1} ms) | \
                 vx{} {m_vx:.1} ms ({:.2}x of b3, {:.0} KB, verify {v_vx:.1} ms)",
                plain_spec.len(),
                proof_plain.xors.len(),
                layout_plain.x_fold_extra,
                proof_mixed.tap_us.len(),
                layout.x_fold_extra,
                sz_b3 as f64 / 1e3,
                base_claims.len(),
                m_vx / m_b3,
                size_tap(&proof_vx) as f64 / 1e3,
            );
            continue;
        }

        if std::env::var("BITZ_AB_SCHED").is_ok_and(|v| v == "1") {
            // The composed collapse on a schedule-shaped instance: every
            // claim `off^t(x)` of ONE σ-style mixed combination — 2
            // inner tap bodies TOTAL — against the batched path on the
            // offset-folded lists (pads to the next power of two) and
            // independent proofs.
            let rounds: usize = std::env::var("BITZ_AB_ROUNDS").map_or(48, |v| v.parse().unwrap());
            let rot = |col, amt, off| TapOp {
                col,
                grp_log2: GRP(),
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let shl = |col, amt, off| TapOp {
                col,
                grp_log2: GRP(),
                bit_amt: amt,
                bit_dropout: true,
                off,
            };
            let src = vec![rot(0, 7, 0), rot(0, 18, 0), shl(0, 3, 0), rot(1, 0, 1)];
            // The composed path needs `rounds − 1 < 2^{s−g}` alone; the
            // FOLDED baseline additionally eats the source's own word
            // offset (its envelope is strictly narrower).
            let max_src_off = src.iter().map(|t| t.off).max().unwrap_or(0);
            assert!(
                rounds + max_src_off <= 1usize << (layout.p.col_vars - GRP()),
                "folded-baseline offsets out of range for this shape"
            );
            let rw: Vec<u128> = (0..p_x.rows())
                .map(|b| {
                    (b as u128)
                        .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                        .wrapping_add(43 + (u128::from(seed) << 1))
                        % FQ_MOD
                })
                .collect();
            let colw: Vec<Q100Element> = (0..p_x.cols())
                .map(|c| {
                    Q100Element::from((c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3))
                })
                .collect();
            let eval_list = |taps: &[TapOp]| -> u128 {
                let a_rows = extract_virtual_tap_rows(&layout, hint.rows(), taps);
                let mut y = Q100Element::from(0u128);
                for (c, row) in a_rows.iter().enumerate() {
                    let mut acc = Q100Element::from(0u128);
                    for (wi, &word) in row.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros() as usize;
                            acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                            bits &= bits.wrapping_sub(1);
                        }
                    }
                    y = y + colw[c] * acc;
                }
                y.canonical_u128()
            };
            // The folded lists (`off += t` per tap) drive both the
            // baseline paths AND the claim values — the independent
            // extraction route the composed algebra must reproduce.
            let folded: Vec<Vec<TapOp>> = (0..rounds)
                .map(|t| {
                    src.iter()
                        .map(|tap| {
                            let mut tap = *tap;
                            tap.off += t;
                            tap
                        })
                        .collect()
                })
                .collect();
            let vals: Vec<u128> = folded.iter().map(|taps| eval_list(taps)).collect();
            let cclaims: Vec<TapComposedClaim<'_>> = (0..rounds)
                .map(|t| TapComposedClaim {
                    source: &src,
                    outer: bitz::taps::TapUniOp {
                        grp_log2: GRP(),
                        bit_amt: 0,
                        bit_dropout: false,
                        off: t,
                    },
                    claimed: vals[t],
                })
                .collect();
            let tclaims: Vec<TapClaim<'_>> = folded
                .iter()
                .map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rw,
                })
                .collect();
            let tvclaims: Vec<TapVerifyClaim<'_, Q100Element>> = folded
                .iter()
                .zip(vals.iter())
                .map(|(taps, &v)| TapVerifyClaim {
                    taps,
                    row_weights_q: &rw,
                    col_weights: &colw,
                    claimed: Q100Element::from(v),
                })
                .collect();
            // The block cap flattened the batched path's working set
            // (peak ≈ 2 tree-sets for any k), so vx48 runs at every n.
            let run_vx = true;
            let prove_cmp = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_composed(
                    &mut t, &hint, &layout, &rw, &colw, &cclaims, alpha, &pc,
                )
            };
            let prove_vx = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t, &hint, &layout, FQ_BITS, &tclaims, alpha, &pc,
                )
            };
            let prove_ind = || {
                (0..tclaims.len())
                    .map(|i| {
                        let mut t = Blake3Transcript::new();
                        prove_mle_eval_mod_q_ligerito_tap_claims(
                            &mut t,
                            &hint,
                            &layout,
                            FQ_BITS,
                            &tclaims[i..i + 1],
                            alpha,
                            &pc,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let (mut t_cmp, mut t_vx, mut t_ind) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_cmp());
                t_cmp.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                if run_vx {
                    let t0_recording = bitz::observability::Recording::start(Vec::new())
                        .expect("start operation capture");
                    let t0 = tracing::info_span!("taps_ab:t0").entered();
                    drop(prove_vx());
                    t_vx.push(
                        {
                            drop(t0);
                            bitz::observability::duration(
                                &t0_recording
                                    .intervals()
                                    .expect("complete operation capture"),
                                "taps_ab:t0",
                            )
                            .expect("query completed operation")
                        }
                        .as_secs_f64()
                            * 1e3,
                    );
                }
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_ind());
                t_ind.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let proof_cmp = prove_cmp();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_composed(
                    &mut vt,
                    &hint.commitment,
                    &proof_cmp,
                    &layout,
                    &rw,
                    &colw,
                    &cclaims,
                    alpha,
                    &vc,
                )
                .expect("composed schedule verifies");
            }
            let v_cmp = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let size_tap = |p: &bitz::ligerito_flock::IntEvalRsLigModQTapProof| {
                let (b, lig) = mle_eval_mod_q_lig_tap_size_breakdown(p);
                b.total() + lig
            };
            let sz_cmp = size_tap(&proof_cmp);
            let (m_cmp, m_ind) = (median(t_cmp), median(t_ind));
            let vx_txt = if run_vx {
                let proof_vx = prove_vx();
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                {
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_tap_claims(
                        &mut vt,
                        &hint.commitment,
                        &proof_vx,
                        &layout,
                        alpha,
                        FQ_BITS,
                        &tvclaims,
                        &vc,
                    )
                    .expect("vx schedule verifies");
                }
                let v_vx = {
                    drop(t0);
                    bitz::observability::duration(
                        &t0_recording
                            .intervals()
                            .expect("complete operation capture"),
                        "taps_ab:t0",
                    )
                    .expect("query completed operation")
                }
                .as_secs_f64()
                    * 1e3;
                let m_vx = median(t_vx);
                format!(
                    "vx{rounds} {m_vx:.1} ms ({:.1}x of cmp, {:.0} KB, verify {v_vx:.1} ms)",
                    m_vx / m_cmp,
                    size_tap(&proof_vx) as f64 / 1e3,
                )
            } else {
                format!("vx{rounds} skipped (64-set pad vs memory)")
            };
            println!(
                "n={n} SCHED ({rounds} off^t claims of one mixed source): cmp {m_cmp:.1} ms \
                 ({} inner bodies, {:.0} KB, verify {v_cmp:.1} ms) | {vx_txt} | ind{rounds} \
                 {m_ind:.1} ms ({:.1}x of cmp)",
                proof_cmp.tap_us.len(),
                sz_cmp as f64 / 1e3,
                m_ind / m_cmp,
            );
            continue;
        }

        if std::env::var("BITZ_AB_COLS4").is_ok_and(|v| v == "1") {
            // FOUR committed columns (log_cols = 2): identity claims on
            // each plus the XOR-mixed pairs b₁ = ROT¹(a₁) ⊕ off¹(a₂)
            // and b₂ = ROT²(a₃) ⊕ off¹(a₄) (statement columns
            // 1-indexed; code 0-indexed), one shared point, through the
            // blocked batched tap path — `single` (one identity claim)
            // as the unit vs `vx6` (all six, blocks 2+2+2) vs `ind6`.
            let layout = taps_layout(n, 2);
            let p = &layout.p;
            let p_x = virtual_xor_params(&layout);
            let (pc, vc) = historical_sha_lig_configs(packed_vars(p)).expect("lig cfg");
            let words = p.rows().div_ceil(64);
            let rows: Vec<Vec<u64>> = (0..p.cols())
                .map(|c| {
                    (0..words)
                        .map(|w| {
                            ((c as u64) << 32 | (w as u64) ^ seed)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                .rotate_left(((c + w) & 63) as u32)
                        })
                        .collect()
                })
                .collect();
            let hint = commit_rs_ligerito_rows(p, rows, &pc);
            let rot = |col, amt, off| TapOp {
                col,
                grp_log2: GRP(),
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let claim_taps: Vec<Vec<TapOp>> = vec![
                vec![TapOp::ident(0)],
                vec![TapOp::ident(1)],
                vec![TapOp::ident(2)],
                vec![TapOp::ident(3)],
                vec![rot(0, 1, 0), rot(1, 0, 1)],
                vec![rot(2, 2, 0), rot(3, 0, 1)],
            ];
            let rw: Vec<u128> = (0..p_x.rows())
                .map(|b| {
                    (b as u128)
                        .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                        .wrapping_add(61 + (u128::from(seed) << 1))
                        % FQ_MOD
                })
                .collect();
            let colw: Vec<Q100Element> = (0..p_x.cols())
                .map(|c| {
                    Q100Element::from((c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3))
                })
                .collect();
            let vals: Vec<u128> = claim_taps
                .iter()
                .map(|taps| {
                    let a_rows = extract_virtual_tap_rows(&layout, hint.rows(), taps);
                    let mut y = Q100Element::from(0u128);
                    for (c, row) in a_rows.iter().enumerate() {
                        let mut acc = Q100Element::from(0u128);
                        for (wi, &word) in row.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                let t = bits.trailing_zeros() as usize;
                                acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                                bits &= bits.wrapping_sub(1);
                            }
                        }
                        y = y + colw[c] * acc;
                    }
                    y.canonical_u128()
                })
                .collect();
            let tclaims: Vec<TapClaim<'_>> = claim_taps
                .iter()
                .map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rw,
                })
                .collect();
            let tvclaims: Vec<TapVerifyClaim<'_, Q100Element>> = claim_taps
                .iter()
                .zip(vals.iter())
                .map(|(taps, &v)| TapVerifyClaim {
                    taps,
                    row_weights_q: &rw,
                    col_weights: &colw,
                    claimed: Q100Element::from(v),
                })
                .collect();
            let prove_k = |r: core::ops::Range<usize>| {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t,
                    &hint,
                    &layout,
                    FQ_BITS,
                    &tclaims[r],
                    alpha,
                    &pc,
                )
            };
            let prove_ind = || {
                (0..tclaims.len())
                    .map(|i| prove_k(i..i + 1))
                    .collect::<Vec<_>>()
            };
            if std::env::var("BITZ_AB_COLS4_FAM").is_ok_and(|v| v == "1") {
                // Research scan: the 4 IDENTITY claims alone, three ways —
                // blocked 0x42 (tap4, 2+2 bodies) vs two j=2 shared-point
                // families (fam2x2: each ≈ 1.3 forests + ONE AND channel)
                // vs one j=4 family (fam1x4: 11 AND channels, predicted
                // loser). Run at δ = 0 (the family paths predate δ).
                use bitz::ligerito_flock::{
                    RlcSharedClaim, prove_mle_eval_mod_q_ligerito_rlc_family_shared_point,
                    verify_mle_eval_mod_q_ligerito_rlc_family_shared_point,
                };
                assert_eq!(layout.x_fold_extra, 0, "family scan runs at δ = 0");
                let pair_claims = |a: usize, b: usize| {
                    vec![
                        RlcSharedClaim {
                            form: 0b01,
                            claimed: vals[a],
                        },
                        RlcSharedClaim {
                            form: 0b10,
                            claimed: vals[b],
                        },
                    ]
                };
                let cl01 = pair_claims(0, 1);
                let cl23 = pair_claims(2, 3);
                let cl4: Vec<RlcSharedClaim> = (0..4)
                    .map(|i| RlcSharedClaim {
                        form: 1 << i,
                        claimed: vals[i],
                    })
                    .collect();
                let prove_fam2 = || {
                    let mut t = Blake3Transcript::new();
                    let p1 = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut t,
                        &hint,
                        &layout,
                        &[0, 1],
                        &rw,
                        &cl01,
                        alpha,
                        &pc,
                    );
                    let mut t = Blake3Transcript::new();
                    let p2 = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut t,
                        &hint,
                        &layout,
                        &[2, 3],
                        &rw,
                        &cl23,
                        alpha,
                        &pc,
                    );
                    (p1, p2)
                };
                let prove_fam4 = || {
                    let mut t = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut t,
                        &hint,
                        &layout,
                        &[0, 1, 2, 3],
                        &rw,
                        &cl4,
                        alpha,
                        &pc,
                    )
                };
                let (mut t_tap4, mut t_fam2, mut t_fam4) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..reps {
                    let t0_recording = bitz::observability::Recording::start(Vec::new())
                        .expect("start operation capture");
                    let t0 = tracing::info_span!("taps_ab:t0").entered();
                    drop(prove_k(0..4));
                    t_tap4.push(
                        {
                            drop(t0);
                            bitz::observability::duration(
                                &t0_recording
                                    .intervals()
                                    .expect("complete operation capture"),
                                "taps_ab:t0",
                            )
                            .expect("query completed operation")
                        }
                        .as_secs_f64()
                            * 1e3,
                    );
                    let t0_recording = bitz::observability::Recording::start(Vec::new())
                        .expect("start operation capture");
                    let t0 = tracing::info_span!("taps_ab:t0").entered();
                    drop(prove_fam2());
                    t_fam2.push(
                        {
                            drop(t0);
                            bitz::observability::duration(
                                &t0_recording
                                    .intervals()
                                    .expect("complete operation capture"),
                                "taps_ab:t0",
                            )
                            .expect("query completed operation")
                        }
                        .as_secs_f64()
                            * 1e3,
                    );
                    let t0_recording = bitz::observability::Recording::start(Vec::new())
                        .expect("start operation capture");
                    let t0 = tracing::info_span!("taps_ab:t0").entered();
                    drop(prove_fam4());
                    t_fam4.push(
                        {
                            drop(t0);
                            bitz::observability::duration(
                                &t0_recording
                                    .intervals()
                                    .expect("complete operation capture"),
                                "taps_ab:t0",
                            )
                            .expect("query completed operation")
                        }
                        .as_secs_f64()
                            * 1e3,
                    );
                }
                let (p1, p2) = prove_fam2();
                {
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut vt,
                        &hint.commitment,
                        &p1,
                        &layout,
                        &[0, 1],
                        &rw,
                        &cl01,
                        &colw,
                        alpha,
                        &vc,
                    )
                    .expect("fam {0,1} verifies");
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut vt,
                        &hint.commitment,
                        &p2,
                        &layout,
                        &[2, 3],
                        &rw,
                        &cl23,
                        &colw,
                        alpha,
                        &vc,
                    )
                    .expect("fam {2,3} verifies");
                }
                {
                    let p4 = prove_fam4();
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut vt,
                        &hint.commitment,
                        &p4,
                        &layout,
                        &[0, 1, 2, 3],
                        &rw,
                        &cl4,
                        &colw,
                        alpha,
                        &vc,
                    )
                    .expect("fam j4 verifies");
                }
                println!(
                    "n={n} COLS4-FAM (the 4 identity claims alone): tap4 {:.1} ms | fam2x2 \
                     {:.1} ms ({:.2}x of tap4) | fam1x4 {:.1} ms ({:.2}x)",
                    median(t_tap4.clone()),
                    median(t_fam2.clone()),
                    median(t_fam2) / median(t_tap4.clone()),
                    median(t_fam4.clone()),
                    median(t_fam4) / median(t_tap4),
                );
                continue;
            }
            let (mut t_single, mut t_vx, mut t_ind) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_k(0..1));
                t_single.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_k(0..tclaims.len()));
                t_vx.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_ind());
                t_ind.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let proof_vx = prove_k(0..tclaims.len());
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &proof_vx,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &tvclaims,
                    &vc,
                )
                .expect("cols4 vx verifies");
            }
            let v_vx = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            {
                let mut vt = Blake3Transcript::new();
                let proof_s = prove_k(0..1);
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &proof_s,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &tvclaims[0..1],
                    &vc,
                )
                .expect("cols4 single verifies");
            }
            let size_tap = |p: &bitz::ligerito_flock::IntEvalRsLigModQTapProof| {
                let (b, lig) = mle_eval_mod_q_lig_tap_size_breakdown(p);
                b.total() + lig
            };
            let (m_s, m_vx, m_ind) = (median(t_single), median(t_vx), median(t_ind));
            println!(
                "n={n} COLS4 (t'={}, s={}, 4 cols: 4 identities + 2 mixed pairs): single \
                 {m_s:.1} ms | vx6 {m_vx:.1} ms ({:.2}x of single, {:.0} KB, verify \
                 {v_vx:.1} ms) | ind6 {m_ind:.1} ms ({:.2}x of vx6)",
                p_x.row_vars,
                p_x.col_vars,
                m_vx / m_s,
                size_tap(&proof_vx) as f64 / 1e3,
                m_ind / m_vx,
            );
            if profile {
                let profile =
                    bitz::observability::Recording::start(Vec::new()).expect("capture profile");
                let pr = prove_k(0..tclaims.len());
                drop(pr);
                bitz::observability::write_profile(
                    std::io::stderr().lock(),
                    &format!("cols4 vx6 n={n}"),
                    &profile.intervals().expect("profile intervals"),
                    None,
                )
                .expect("write profile");
            }
            continue;
        }

        if std::env::var("BITZ_AB_MIX6").is_ok_and(|v| v == "1") {
            // The uniform-op k=6 variant: 2 identities + 4 offsets of
            // ROT^7(a₀⊕a₁) at ONE point. ROT^7 is off the δ-envelope
            // for the collapse route (7 ∤ 2^δ), so this mode runs δ=0.
            assert_eq!(
                layout.x_fold_extra, 0,
                "mix6's ROT^7 outer is off the δ-envelope; unset BITZ_TAPS_DELTA"
            );
            let rot = |col, amt, off| TapOp {
                col,
                grp_log2: GRP(),
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let uni = |amt: usize, off: usize| bitz::taps::TapUniOp {
                grp_log2: GRP(),
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let rw: Vec<u128> = (0..p_x.rows())
                .map(|b| {
                    (b as u128)
                        .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                        .wrapping_add(53 + (u128::from(seed) << 1))
                        % FQ_MOD
                })
                .collect();
            let colw: Vec<Q100Element> = (0..p_x.cols())
                .map(|c| {
                    Q100Element::from((c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3))
                })
                .collect();
            let eval_list = |taps: &[TapOp]| -> u128 {
                let a_rows = extract_virtual_tap_rows(&layout, hint.rows(), taps);
                let mut y = Q100Element::from(0u128);
                for (c, row) in a_rows.iter().enumerate() {
                    let mut acc = Q100Element::from(0u128);
                    for (wi, &word) in row.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros() as usize;
                            acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                            bits &= bits.wrapping_sub(1);
                        }
                    }
                    y = y + colw[c] * acc;
                }
                y.canonical_u128()
            };
            // Claim shapes: (XOR set, uniform op); the batched/ind
            // baselines run the composed per-column tap lists, which
            // also supply the claim values (the distributed
            // cross-check route).
            let sets: Vec<Vec<usize>> = vec![
                vec![0],
                vec![1],
                vec![0, 1],
                vec![0, 1],
                vec![0, 1],
                vec![0, 1],
            ];
            let ops: Vec<bitz::taps::TapUniOp> = vec![
                uni(0, 0),
                uni(0, 0),
                uni(7, 0),
                uni(7, 1),
                uni(7, 2),
                uni(7, 3),
            ];
            let lists: Vec<Vec<TapOp>> = sets
                .iter()
                .zip(ops.iter())
                .map(|(set, op)| set.iter().map(|&c| rot(c, op.bit_amt, op.off)).collect())
                .collect();
            let vals: Vec<u128> = lists.iter().map(|taps| eval_list(taps)).collect();
            let pclaims: Vec<TapPointClaim<'_>> = sets
                .iter()
                .zip(ops.iter())
                .zip(vals.iter())
                .map(|((set, &op), &claimed)| TapPointClaim {
                    cols: set,
                    op,
                    claimed,
                })
                .collect();
            let tclaims: Vec<TapClaim<'_>> = lists
                .iter()
                .map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rw,
                })
                .collect();
            let tvclaims: Vec<TapVerifyClaim<'_, Q100Element>> = lists
                .iter()
                .zip(vals.iter())
                .map(|(taps, &v)| TapVerifyClaim {
                    taps,
                    row_weights_q: &rw,
                    col_weights: &colw,
                    claimed: Q100Element::from(v),
                })
                .collect();
            let prove_clp = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut t, &hint, &layout, &rw, &colw, &pclaims, alpha, &pc,
                )
            };
            let prove_vx = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t, &hint, &layout, FQ_BITS, &tclaims, alpha, &pc,
                )
            };
            let prove_ind = || {
                (0..tclaims.len())
                    .map(|i| {
                        let mut t = Blake3Transcript::new();
                        prove_mle_eval_mod_q_ligerito_tap_claims(
                            &mut t,
                            &hint,
                            &layout,
                            FQ_BITS,
                            &tclaims[i..i + 1],
                            alpha,
                            &pc,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let (mut t_clp, mut t_vx, mut t_ind) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_clp());
                t_clp.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_vx());
                t_vx.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_ind());
                t_ind.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let proof_clp = prove_clp();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut vt,
                    &hint.commitment,
                    &proof_clp,
                    &layout,
                    &rw,
                    &colw,
                    &pclaims,
                    alpha,
                    &vc,
                )
                .expect("mix6 collapse verifies");
            }
            let v_clp = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let proof_vx = prove_vx();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &proof_vx,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &tvclaims,
                    &vc,
                )
                .expect("mix6 vx verifies");
            }
            let v_vx = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let size_tap = |p: &bitz::ligerito_flock::IntEvalRsLigModQTapProof| {
                let (b, lig) = mle_eval_mod_q_lig_tap_size_breakdown(p);
                b.total() + lig
            };
            let (m_clp, m_vx, m_ind) = (median(t_clp), median(t_vx), median(t_ind));
            println!(
                "n={n} MIX6 (2 identities + 4 off^t(ROT^7(a0^a1)), one point): clp {m_clp:.1} \
                 ms ({} inner, {:.0} KB, verify {v_clp:.1} ms) | vx6m {m_vx:.1} ms ({:.2}x of \
                 clp, {:.0} KB, verify {v_vx:.1} ms) | ind6m {m_ind:.1} ms ({:.2}x of clp)",
                proof_clp.xors.len(),
                mle_eval_mod_q_lig_xor_proof_size_bytes(&proof_clp) as f64 / 1e3,
                m_vx / m_clp,
                size_tap(&proof_vx) as f64 / 1e3,
                m_ind / m_clp,
            );
            continue;
        }

        if std::env::var("BITZ_AB_COLLAPSE").is_ok_and(|v| v == "1") {
            assert_eq!(
                layout.x_fold_extra, 0,
                "the collapse demo's stream ops are off the δ-envelope; unset BITZ_TAPS_DELTA"
            );
            // 13 single-tap claims (the instance's deduped streams) at ONE
            // shared point: collapse vs batched tap claims vs independent.
            let (streams2, _, _) = instance_clusters();
            let all_taps: Vec<TapOp> = streams2.into_iter().flatten().collect::<Vec<_>>();
            let rw: Vec<u128> = (0..p_x.rows())
                .map(|b| {
                    (b as u128)
                        .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                        .wrapping_add(29 + (u128::from(seed) << 1))
                        % FQ_MOD
                })
                .collect();
            let colw: Vec<Q100Element> = (0..p_x.cols())
                .map(|c| {
                    Q100Element::from((c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3))
                })
                .collect();
            let vals: Vec<u128> = all_taps
                .iter()
                .map(|&tap| {
                    let a_rows = extract_virtual_tap_rows(&layout, hint.rows(), &[tap]);
                    let mut y = Q100Element::from(0u128);
                    for (c, row) in a_rows.iter().enumerate() {
                        let mut acc = Q100Element::from(0u128);
                        for (wi, &word) in row.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                let t = bits.trailing_zeros() as usize;
                                acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                                bits &= bits.wrapping_sub(1);
                            }
                        }
                        y = y + colw[c] * acc;
                    }
                    y.canonical_u128()
                })
                .collect();
            let single_sets: Vec<[usize; 1]> = all_taps.iter().map(|t| [t.col]).collect();
            let pclaims: Vec<TapPointClaim<'_>> = all_taps
                .iter()
                .zip(single_sets.iter())
                .zip(vals.iter())
                .map(|((tap, set), &claimed)| TapPointClaim {
                    cols: set,
                    op: tap.uni(),
                    claimed,
                })
                .collect();
            let single_lists: Vec<[TapOp; 1]> = all_taps.iter().map(|&t| [t]).collect();
            let tclaims: Vec<TapClaim<'_>> = single_lists
                .iter()
                .map(|taps| TapClaim {
                    taps,
                    row_weights_q: &rw,
                })
                .collect();
            let tvclaims: Vec<TapVerifyClaim<'_, Q100Element>> = single_lists
                .iter()
                .zip(vals.iter())
                .map(|(taps, &v)| TapVerifyClaim {
                    taps,
                    row_weights_q: &rw,
                    col_weights: &colw,
                    claimed: Q100Element::from(v),
                })
                .collect();
            let prove_clp = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut t, &hint, &layout, &rw, &colw, &pclaims, alpha, &pc,
                )
            };
            let prove_vx13 = || {
                let mut t = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_tap_claims(
                    &mut t, &hint, &layout, FQ_BITS, &tclaims, alpha, &pc,
                )
            };
            let prove_ind13 = || {
                (0..tclaims.len())
                    .map(|i| {
                        let mut t = Blake3Transcript::new();
                        prove_mle_eval_mod_q_ligerito_tap_claims(
                            &mut t,
                            &hint,
                            &layout,
                            FQ_BITS,
                            &tclaims[i..i + 1],
                            alpha,
                            &pc,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let (mut t_clp, mut t_vx, mut t_ind) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_clp());
                t_clp.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_vx13());
                t_vx.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("taps_ab:t0").entered();
                drop(prove_ind13());
                t_ind.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "taps_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
            }
            let proof_clp = prove_clp();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_collapse(
                    &mut vt,
                    &hint.commitment,
                    &proof_clp,
                    &layout,
                    &rw,
                    &colw,
                    &pclaims,
                    alpha,
                    &vc,
                )
                .expect("collapse verifies");
            }
            let v_clp = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let proof_vx = prove_vx13();
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            {
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_tap_claims(
                    &mut vt,
                    &hint.commitment,
                    &proof_vx,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &tvclaims,
                    &vc,
                )
                .expect("vx13 verifies");
            }
            let v_vx = {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3;
            let sz_clp = mle_eval_mod_q_lig_xor_proof_size_bytes(&proof_clp);
            let sz_vx = {
                let (b, lig) = mle_eval_mod_q_lig_tap_size_breakdown(&proof_vx);
                b.total() + lig
            };
            let (m_clp, m_vx, m_ind) = (median(t_clp), median(t_vx), median(t_ind));
            println!(
                "n={n} COLLAPSE (13 single-tap claims, one point): clp {m_clp:.1} ms ({} inner) \
                 | vx13 {m_vx:.1} ms ({:.2}x of clp) | ind13 {m_ind:.1} ms ({:.2}x) | proofs \
                 clp {:.0} KB, vx13 {:.0} KB | verify clp {v_clp:.1} ms, vx13 {v_vx:.1} ms",
                proof_clp.xors.len(),
                m_vx / m_clp,
                m_ind / m_clp,
                sz_clp as f64 / 1e3,
                sz_vx as f64 / 1e3,
            );
            continue;
        }

        let claim_taps = instance_claims();
        let rws: Vec<Vec<u128>> = (0..claim_taps.len())
            .map(|i| {
                (0..p_x.rows())
                    .map(|b| {
                        (b as u128)
                            .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                            .wrapping_add(11 + i as u128 + (u128::from(seed) << 1))
                            % FQ_MOD
                    })
                    .collect()
            })
            .collect();
        let colw: Vec<Q100Element> = (0..p_x.cols())
            .map(|c| Q100Element::from((c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3)))
            .collect();
        let cs: Vec<u128> = claim_taps
            .iter()
            .zip(rws.iter())
            .map(|(taps, rw)| {
                let a_rows = extract_virtual_tap_rows(&layout, hint.rows(), taps);
                let mut y = Q100Element::from(0u128);
                for (c, row) in a_rows.iter().enumerate() {
                    let mut acc = Q100Element::from(0u128);
                    for (wi, &word) in row.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros() as usize;
                            acc = acc + Q100Element::from(rw[(wi << 6) | t]);
                            bits &= bits.wrapping_sub(1);
                        }
                    }
                    y = y + colw[c] * acc;
                }
                y.canonical_u128()
            })
            .collect();

        // Family clusters.
        let (streams, forms, members) = instance_clusters();
        let cluster_claims: Vec<Vec<RlcFamilyClaim<'_>>> = (0..2)
            .map(|ci| {
                forms[ci]
                    .iter()
                    .zip(members[ci].iter())
                    .map(|(&form, &bi)| RlcFamilyClaim {
                        form,
                        row_weights_q: &rws[bi],
                        claimed: cs[bi],
                    })
                    .collect()
            })
            .collect();
        let clusters: Vec<TapFamilyCluster<'_>> = (0..2)
            .map(|ci| TapFamilyCluster {
                streams: &streams[ci],
                claims: &cluster_claims[ci],
            })
            .collect();

        let tap_claims_of = |idx: &[usize]| -> Vec<TapClaim<'_>> {
            idx.iter()
                .map(|&i| TapClaim {
                    taps: &claim_taps[i],
                    row_weights_q: &rws[i],
                })
                .collect()
        };
        let tap_vclaims_of = |idx: &[usize]| -> Vec<TapVerifyClaim<'_, Q100Element>> {
            idx.iter()
                .map(|&i| TapVerifyClaim {
                    taps: &claim_taps[i],
                    row_weights_q: &rws[i],
                    col_weights: &colw,
                    claimed: Q100Element::from(cs[i]),
                })
                .collect()
        };
        let all6: Vec<usize> = (0..6).collect();

        let prove_single = || {
            let mut t = Blake3Transcript::new();
            prove_mle_eval_mod_q_ligerito_tap_claims(
                &mut t,
                &hint,
                &layout,
                FQ_BITS,
                &tap_claims_of(&[0]),
                alpha,
                &pc,
            )
        };
        let prove_tapf = || {
            let mut t = Blake3Transcript::new();
            prove_mle_eval_mod_q_ligerito_tap_family(&mut t, &hint, &layout, &clusters, alpha, &pc)
        };
        let prove_vx6 = || {
            let mut t = Blake3Transcript::new();
            prove_mle_eval_mod_q_ligerito_tap_claims(
                &mut t,
                &hint,
                &layout,
                FQ_BITS,
                &tap_claims_of(&all6),
                alpha,
                &pc,
            )
        };
        let prove_ind6 = || {
            (0..6)
                .map(|i| {
                    let mut t = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_tap_claims(
                        &mut t,
                        &hint,
                        &layout,
                        FQ_BITS,
                        &tap_claims_of(&[i]),
                        alpha,
                        &pc,
                    )
                })
                .collect::<Vec<_>>()
        };

        // Alternated in-window reps.
        let no_family =
            std::env::var("BITZ_AB_NO_FAMILY").is_ok_and(|v| v == "1") || layout.x_fold_extra > 0;
        let (mut t_single, mut t_tapf, mut t_vx6, mut t_ind6) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..reps {
            let (pr, t0) =
                bitz::observability::measure(tracing::info_span!("taps_ab:pr"), || prove_single())
                    .expect("measure completed operation");
            t_single.push(t0.as_secs_f64() * 1e3);
            drop(pr);
            if !no_family {
                let (pr, t0) =
                    bitz::observability::measure(tracing::info_span!("taps_ab:pr"), || prove_tapf())
                        .expect("measure completed operation");
                t_tapf.push(t0.as_secs_f64() * 1e3);
                drop(pr);
            }
            let (pr, t0) =
                bitz::observability::measure(tracing::info_span!("taps_ab:pr"), || prove_vx6())
                    .expect("measure completed operation");
            t_vx6.push(t0.as_secs_f64() * 1e3);
            drop(pr);
            let (pr, t0) =
                bitz::observability::measure(tracing::info_span!("taps_ab:pr"), || prove_ind6())
                    .expect("measure completed operation");
            t_ind6.push(t0.as_secs_f64() * 1e3);
            drop(pr);
        }

        // Verify once each + sizes.
        let proof_single = prove_single();
        let proof_tapf = (!no_family).then(&prove_tapf);
        let proof_vx6 = prove_vx6();
        let proofs_ind = prove_ind6();
        {
            let mut vt = Blake3Transcript::new();
            verify_mle_eval_mod_q_ligerito_tap_claims(
                &mut vt,
                &hint.commitment,
                &proof_single,
                &layout,
                alpha,
                FQ_BITS,
                &tap_vclaims_of(&[0]),
                &vc,
            )
            .expect("single verifies");
        }
        let v_tapf = proof_tapf.as_ref().map(|pr| {
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("taps_ab:t0").entered();
            let mut vt = Blake3Transcript::new();
            verify_mle_eval_mod_q_ligerito_tap_family(
                &mut vt,
                &hint.commitment,
                pr,
                &layout,
                &clusters,
                &colw,
                alpha,
                &vc,
            )
            .expect("tapf verifies");
            {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "taps_ab:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3
        });
        let t0_recording =
            bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t0 = tracing::info_span!("taps_ab:t0").entered();
        {
            let mut vt = Blake3Transcript::new();
            verify_mle_eval_mod_q_ligerito_tap_claims(
                &mut vt,
                &hint.commitment,
                &proof_vx6,
                &layout,
                alpha,
                FQ_BITS,
                &tap_vclaims_of(&all6),
                &vc,
            )
            .expect("vx6 verifies");
        }
        let v_vx6 = {
            drop(t0);
            bitz::observability::duration(
                &t0_recording
                    .intervals()
                    .expect("complete operation capture"),
                "taps_ab:t0",
            )
            .expect("query completed operation")
        }
        .as_secs_f64()
            * 1e3;
        for (i, pr) in proofs_ind.iter().enumerate() {
            let mut vt = Blake3Transcript::new();
            verify_mle_eval_mod_q_ligerito_tap_claims(
                &mut vt,
                &hint.commitment,
                pr,
                &layout,
                alpha,
                FQ_BITS,
                &tap_vclaims_of(&[i]),
                &vc,
            )
            .expect("ind verifies");
        }

        let size_tap = |p: &bitz::ligerito_flock::IntEvalRsLigModQTapProof| {
            let (b, lig) = mle_eval_mod_q_lig_tap_size_breakdown(p);
            b.total() + lig
        };
        let sz_single = size_tap(&proof_single);
        let sz_vx6 = size_tap(&proof_vx6);
        let sz_ind6: usize = proofs_ind.iter().map(&size_tap).sum();

        let (m_single, m_vx6, m_ind6) = (median(t_single), median(t_vx6), median(t_ind6));
        let tapf_txt = match &proof_tapf {
            Some(_) => {
                let m_tapf = median(t_tapf.clone());
                format!("tapf {m_tapf:.1} ({:.2}x)", m_tapf / m_single)
            }
            None => "tapf skipped".to_string(),
        };
        println!(
            "n={n} (t'={}, s={}, tw={}): single {m_single:.1} ms | {tapf_txt} | \
             vx6 {m_vx6:.1} ({:.2}x) | ind6 {m_ind6:.1} ({:.2}x)",
            p_x.row_vars,
            p_x.col_vars,
            layout.tw,
            m_vx6 / m_single,
            m_ind6 / m_single,
        );
        let tapf_sz_txt = match &proof_tapf {
            Some(pr) => {
                let (bf, ligf) = mle_eval_mod_q_lig_tap_family_size_breakdown(pr);
                format!(
                    "tapf {:.0} KB (verify {:.1} ms, {} rings)",
                    (bf.total() + ligf) as f64 / 1e3,
                    v_tapf.unwrap_or(f64::NAN),
                    pr.rings.len(),
                )
            }
            None => "tapf -".to_string(),
        };
        println!(
            "  proofs: single {:.0} KB | {tapf_sz_txt} | vx6 {:.0} KB | ind6 {:.0} KB \
             | verify vx6 {v_vx6:.1} ms | rings vx6 {}",
            sz_single as f64 / 1e3,
            sz_vx6 as f64 / 1e3,
            sz_ind6 as f64 / 1e3,
            proof_vx6.rings.len(),
        );

        if profile {
            if !no_family {
                let profile =
                    bitz::observability::Recording::start(Vec::new()).expect("capture profile");
                let pr = prove_tapf();
                drop(pr);
                bitz::observability::write_profile(
                    std::io::stderr().lock(),
                    &format!("tapf n={n}"),
                    &profile.intervals().expect("profile intervals"),
                    None,
                )
                .expect("write profile");
            }
            let profile =
                bitz::observability::Recording::start(Vec::new()).expect("capture profile");
            let pr = prove_vx6();
            drop(pr);
            bitz::observability::write_profile(
                std::io::stderr().lock(),
                &format!("vx6 n={n}"),
                &profile.intervals().expect("profile intervals"),
                None,
            )
            .expect("write profile");
        }
    }
}
