//! Historical kernel experiment; no production security claim.
//! RLC-family A/B (EXPERIMENTAL, docs/rlc-family-proto-prompt.md): the XOR
//! triple (k = 3, j = 2, a₃ = a₁ ⊕ a₂) proved three ways against ONE
//! commitment and statement —
//!
//! * `single` — one claim alone via the batched-vx path (the cost unit);
//! * `rlc3`   — the RLC claim family (ONE 4-case forest + monomial
//!   discharge);
//! * `vx3`    — the batched virtual-XOR path (ONE 4·2^s-tree forest);
//! * `ind3`   — three independent single-claim vx proofs (no batching).
//!
//! Alternated in-window reps, medians reported (repo measurement
//! protocol). Verifies every variant once. With `OBLONG_PROFILE=1` one
//! extra profiled prove of `rlc3` and `vx3` dumps the phase tree per shape.
//!
//! ```text
//! BITZ_AB_N="22 24 26" BITZ_AB_REPS=5 RUSTFLAGS="-C target-cpu=native" \
//!   cargo run --release --example rlc_ab --features unchecked,span-metrics
//! ```

use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{
    RlcFamilyClaim, RlcSharedClaim, VirtualXorClaim, VirtualXorVerifyClaim,
    commit_rs_ligerito_rows, historical_sha_lig_configs,
    mle_eval_mod_q_lig_rlc_family_proof_size_bytes, mle_eval_mod_q_lig_xor_proof_size_bytes,
    prove_mle_eval_mod_q_ligerito_claims_only, prove_mle_eval_mod_q_ligerito_rlc_family,
    prove_mle_eval_mod_q_ligerito_rlc_family_shared_point,
    verify_mle_eval_mod_q_ligerito_claims_only, verify_mle_eval_mod_q_ligerito_rlc_family,
    verify_mle_eval_mod_q_ligerito_rlc_family_shared_point,
};
use bitz::pcs::{
    FQ_BITS, FQ_MOD, IntegerMatrixLayout, Q100Element, ShaF2Layout, extract_virtual_xor_rows,
    smallest_generator, virtual_xor_params,
};
use bitz::transcript::Blake3Transcript;

/// n → the A/B layout: 4 UAIR columns (log_cols = 2) of 32-bit words
/// (bit_vars = 5), the remaining n − 2 variables split t' vs s as evenly
/// as possible (x tensor t' = ⌊(n−2)/2⌋).
fn ab_layout(n: usize) -> ShaF2Layout {
    let log_cols = 2usize;
    let bit_vars = 5usize;
    let t_x = (n - log_cols) / 2;
    let s = n - log_cols - t_x;
    let tw = t_x - bit_vars;
    ShaF2Layout {
        p: IntegerMatrixLayout {
            row_vars: bit_vars + log_cols + tw,
            col_vars: s,
            word_bits: 1,
        },
        num_cols: 1 << log_cols,
        log_cols,
        bit_vars,
        num_vars: tw + s,
        tw,
        x_fold_extra: 0,
    }
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
    let profile = std::env::var("OBLONG_PROFILE").is_ok_and(|v| v == "1");

    for &n in &ns {
        let layout = ab_layout(n);
        let p = &layout.p;
        let p_x = virtual_xor_params(&layout);
        let m_p = packed_vars(p);
        let (pc, vc) = historical_sha_lig_configs(m_p).expect("lig cfg");

        // Pseudorandom committed bit rows (memory-honest packed-rows path).
        let words = p.rows().div_ceil(64);
        let rows: Vec<Vec<u64>> = (0..p.cols())
            .map(|c| {
                (0..words)
                    .map(|w| {
                        ((c as u64) << 32 | w as u64)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .rotate_left(((c + w) & 63) as u32)
                    })
                    .collect()
            })
            .collect();
        let hint = commit_rs_ligerito_rows(p, rows, &pc);

        // Statement: 3 claims on cols {0}, {1}, {0,1}, distinct row points,
        // shared column point.
        let family_cols = [0usize, 1];
        let forms = [0b01usize, 0b10, 0b11];
        let col_lists: Vec<Vec<usize>> = vec![
            vec![family_cols[0]],
            vec![family_cols[1]],
            family_cols.to_vec(),
        ];
        let rws: Vec<Vec<u128>> = (0..3)
            .map(|i| {
                (0..p_x.rows())
                    .map(|b| {
                        (b as u128)
                            .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                            .wrapping_add(11 + i as u128)
                            % FQ_MOD
                    })
                    .collect()
            })
            .collect();
        let colw: Vec<Q100Element> = (0..p_x.cols())
            .map(|c| Q100Element::from((c as u128).wrapping_mul(0xABCD_EF01_2345).wrapping_add(3)))
            .collect();
        let cs: Vec<u128> = (0..3)
            .map(|i| {
                let a_rows = extract_virtual_xor_rows(&layout, hint.rows(), &col_lists[i], 0, None);
                let mut y = Q100Element::from(0u128);
                for (c, row) in a_rows.iter().enumerate() {
                    let mut acc = Q100Element::from(0u128);
                    for (wi, &word) in row.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros() as usize;
                            acc = acc + Q100Element::from(rws[i][(wi << 6) | t]);
                            bits &= bits.wrapping_sub(1);
                        }
                    }
                    y = y + colw[c] * acc;
                }
                y.canonical_u128()
            })
            .collect();
        let claims: Vec<RlcFamilyClaim<'_>> = (0..3)
            .map(|i| RlcFamilyClaim {
                form: forms[i],
                row_weights_q: &rws[i],
                claimed: cs[i],
            })
            .collect();
        let vx_of = |idx: &[usize]| -> Vec<VirtualXorClaim<'_>> {
            idx.iter()
                .map(|&i| VirtualXorClaim {
                    cols: &col_lists[i],
                    constant: 0,
                    external_rows: None,
                    row_weights_q: &rws[i],
                })
                .collect()
        };

        // Optional single-claim comparisons (BITZ_AB_SINGLES=1): the same
        // lone claim through the family API (j=1, k=1), and a lone XOR
        // claim through both APIs (family j=2 k=1 form=11 — the elided
        // pure-XOR family — vs the vx extraction path).
        let singles = std::env::var("BITZ_AB_SINGLES").is_ok_and(|v| v == "1");
        let c_xor = cs[2];
        let single_family_col = [family_cols[0]];
        let rlc1_claims = vec![RlcFamilyClaim {
            form: 0b1,
            row_weights_q: &rws[0],
            claimed: cs[0],
        }];
        let rlcx1_claims = vec![RlcFamilyClaim {
            form: 0b11,
            row_weights_q: &rws[2],
            claimed: c_xor,
        }];
        if singles {
            let mut t_rlc1 = Vec::with_capacity(reps);
            let mut t_rlcx1 = Vec::with_capacity(reps);
            let mut t_vxx1 = Vec::with_capacity(reps);
            let mut t_single1 = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("rlc_ab:t0").entered();
                let pr = {
                    let mut pt = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_claims_only(
                        &mut pt,
                        &hint,
                        &layout,
                        FQ_BITS,
                        &vx_of(&[0]),
                        alpha,
                        &pc,
                    )
                };
                t_single1.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "rlc_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                std::hint::black_box(&pr);

                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("rlc_ab:t0").entered();
                let pr = {
                    let mut pt = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_rlc_family(
                        &mut pt,
                        &hint,
                        &layout,
                        &single_family_col,
                        &rlc1_claims,
                        alpha,
                        &pc,
                    )
                };
                t_rlc1.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "rlc_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                std::hint::black_box(&pr);

                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("rlc_ab:t0").entered();
                let pr = {
                    let mut pt = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_claims_only(
                        &mut pt,
                        &hint,
                        &layout,
                        FQ_BITS,
                        &vx_of(&[2]),
                        alpha,
                        &pc,
                    )
                };
                t_vxx1.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "rlc_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                std::hint::black_box(&pr);

                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("rlc_ab:t0").entered();
                let pr = {
                    let mut pt = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_rlc_family(
                        &mut pt,
                        &hint,
                        &layout,
                        &family_cols,
                        &rlcx1_claims,
                        alpha,
                        &pc,
                    )
                };
                t_rlcx1.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "rlc_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                std::hint::black_box(&pr);
            }
            // Sanity: the two family singles verify.
            {
                let mut pt = Blake3Transcript::new();
                let pr = prove_mle_eval_mod_q_ligerito_rlc_family(
                    &mut pt,
                    &hint,
                    &layout,
                    &single_family_col,
                    &rlc1_claims,
                    alpha,
                    &pc,
                );
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family(
                    &mut vt,
                    &hint.commitment,
                    &pr,
                    &layout,
                    &single_family_col,
                    &rlc1_claims,
                    &colw,
                    alpha,
                    &vc,
                )
                .expect("rlc1 verifies");
                let mut pt = Blake3Transcript::new();
                let pr = prove_mle_eval_mod_q_ligerito_rlc_family(
                    &mut pt,
                    &hint,
                    &layout,
                    &family_cols,
                    &rlcx1_claims,
                    alpha,
                    &pc,
                );
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family(
                    &mut vt,
                    &hint.commitment,
                    &pr,
                    &layout,
                    &family_cols,
                    &rlcx1_claims,
                    &colw,
                    alpha,
                    &vc,
                )
                .expect("rlcx1 (pure-XOR single) verifies");
            }
            println!(
                "n={n} singles: vx-single {:.1} ms | rlc1 {:.1} ms | vx-xor1 {:.1} ms | rlc-xor1 {:.1} ms",
                median(t_single1),
                median(t_rlc1),
                median(t_vxx1),
                median(t_rlcx1),
            );
        }

        // Optional j = 3 family A/B (BITZ_AB_J3=1): k = 4 claims on cols
        // {0},{1},{2},{0,1,2} — the cascade discharge (level-2 AND) path —
        // vs the batched-vx and independent baselines on the same statement.
        if std::env::var("BITZ_AB_J3").is_ok_and(|v| v == "1") {
            let family3 = [0usize, 1, 2];
            let forms3 = [0b001usize, 0b010, 0b100, 0b111];
            let col_lists3: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![2], vec![0, 1, 2]];
            let rws3: Vec<Vec<u128>> = (0..4)
                .map(|i| {
                    (0..p_x.rows())
                        .map(|b| {
                            (b as u128)
                                .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                                .wrapping_add(41 + i as u128)
                                % FQ_MOD
                        })
                        .collect()
                })
                .collect();
            let cs3: Vec<u128> = (0..4)
                .map(|i| {
                    let a_rows =
                        extract_virtual_xor_rows(&layout, hint.rows(), &col_lists3[i], 0, None);
                    let mut y = Q100Element::from(0u128);
                    for (c, row) in a_rows.iter().enumerate() {
                        let mut acc = Q100Element::from(0u128);
                        for (wi, &word) in row.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                let t = bits.trailing_zeros() as usize;
                                acc = acc + Q100Element::from(rws3[i][(wi << 6) | t]);
                                bits &= bits.wrapping_sub(1);
                            }
                        }
                        y = y + colw[c] * acc;
                    }
                    y.canonical_u128()
                })
                .collect();
            let claims3: Vec<RlcFamilyClaim<'_>> = (0..4)
                .map(|i| RlcFamilyClaim {
                    form: forms3[i],
                    row_weights_q: &rws3[i],
                    claimed: cs3[i],
                })
                .collect();
            let vx3_of = |idx: &[usize]| -> Vec<VirtualXorClaim<'_>> {
                idx.iter()
                    .map(|&i| VirtualXorClaim {
                        cols: &col_lists3[i],
                        constant: 0,
                        external_rows: None,
                        row_weights_q: &rws3[i],
                    })
                    .collect()
            };
            let mut t_rlc4 = Vec::with_capacity(reps);
            let mut t_vx4 = Vec::with_capacity(reps);
            let mut t_ind4 = Vec::with_capacity(reps);
            let mut sz = (0usize, 0usize, 0usize);
            for rep in 0..reps {
                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("rlc_ab:t0").entered();
                let pr_rlc = {
                    let mut pt = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_rlc_family(
                        &mut pt, &hint, &layout, &family3, &claims3, alpha, &pc,
                    )
                };
                t_rlc4.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "rlc_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                std::hint::black_box(&pr_rlc);

                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("rlc_ab:t0").entered();
                let pr_vx = {
                    let mut pt = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_claims_only(
                        &mut pt,
                        &hint,
                        &layout,
                        FQ_BITS,
                        &vx3_of(&[0, 1, 2, 3]),
                        alpha,
                        &pc,
                    )
                };
                t_vx4.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "rlc_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                std::hint::black_box(&pr_vx);

                let t0_recording = bitz::observability::Recording::start(Vec::new())
                    .expect("start operation capture");
                let t0 = tracing::info_span!("rlc_ab:t0").entered();
                let pr_inds: Vec<_> = (0..4)
                    .map(|i| {
                        let mut pt = Blake3Transcript::new();
                        prove_mle_eval_mod_q_ligerito_claims_only(
                            &mut pt,
                            &hint,
                            &layout,
                            FQ_BITS,
                            &vx3_of(&[i]),
                            alpha,
                            &pc,
                        )
                    })
                    .collect();
                t_ind4.push(
                    {
                        drop(t0);
                        bitz::observability::duration(
                            &t0_recording
                                .intervals()
                                .expect("complete operation capture"),
                            "rlc_ab:t0",
                        )
                        .expect("query completed operation")
                    }
                    .as_secs_f64()
                        * 1e3,
                );
                std::hint::black_box(&pr_inds);

                if rep == 0 {
                    sz = (
                        mle_eval_mod_q_lig_rlc_family_proof_size_bytes(&pr_rlc),
                        mle_eval_mod_q_lig_xor_proof_size_bytes(&pr_vx),
                        pr_inds
                            .iter()
                            .map(mle_eval_mod_q_lig_xor_proof_size_bytes)
                            .sum(),
                    );
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_rlc_family(
                        &mut vt,
                        &hint.commitment,
                        &pr_rlc,
                        &layout,
                        &family3,
                        &claims3,
                        &colw,
                        alpha,
                        &vc,
                    )
                    .expect("rlc j3 verifies");
                }
            }
            println!(
                "n={n} j3(k=4): rlc4 {:.1} ms ({} B) | vx4 {:.1} ms ({} B) | ind4 {:.1} ms ({} B)",
                median(t_rlc4),
                sz.0,
                median(t_vx4),
                sz.1,
                median(t_ind4),
                sz.2,
            );
        }

        // Optional SHARED-POINT maximal families (BITZ_AB_SHARED=1): the
        // full XOR-closure of j columns at ONE point — j = 2: k = 3,
        // j = 3: k = 7, j = 4: k = 15 — through (rlcS) the shared-point
        // API, (rlcG) the general family API on the same statement (k
        // copies of the one weight vector: isolates the collapsed absorb +
        // rank-1 case build), (vx) the batched virtual-XOR path, and (ind)
        // k independent proofs. Verify medians per arm. The vx forest pads
        // k to 2^⌈log₂k⌉ tree-sets of 16 B leaves and is SKIPPED above an
        // ~8 GB estimate (k = 15 at n = 28). `BITZ_AB_STMTS=S` averages
        // over S statements (FS grinding luck is deterministic per
        // statement — material at n ≤ 24).
        if std::env::var("BITZ_AB_SHARED").is_ok_and(|v| v == "1") {
            let stmts: usize = std::env::var("BITZ_AB_STMTS").map_or(1, |v| v.parse().unwrap());
            for j in [2usize, 3, 4] {
                let k = (1usize << j) - 1;
                let family: Vec<usize> = (0..j).collect();
                let forms: Vec<usize> = (1..1usize << j).collect();
                let col_lists_s: Vec<Vec<usize>> = forms
                    .iter()
                    .map(|&f| (0..j).filter(|&fi| (f >> fi) & 1 == 1).collect())
                    .collect();
                let vx_leaf_gb = ((k.next_power_of_two() << (n - 2) << 4) as f64) / 1e9;
                let run_vx = vx_leaf_gb < 8.0;
                if !run_vx {
                    println!(
                        "n={n} shared j{j}(k={k}): vx baseline SKIPPED (padded forest ~{vx_leaf_gb:.1} GB)"
                    );
                }
                let mut agg: Vec<Vec<f64>> = vec![Vec::new(); 8]; // per-variant medians across stmts
                for stmt in 0..stmts {
                    let rw_s: Vec<u128> = (0..p_x.rows())
                        .map(|b| {
                            (b as u128)
                                .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                                .wrapping_add(71 + (stmt * 17) as u128)
                                % FQ_MOD
                        })
                        .collect();
                    let cs_s: Vec<u128> = col_lists_s
                        .iter()
                        .map(|cl| {
                            let a_rows =
                                extract_virtual_xor_rows(&layout, hint.rows(), cl, 0, None);
                            let mut y = Q100Element::from(0u128);
                            for (c, row) in a_rows.iter().enumerate() {
                                let mut acc = Q100Element::from(0u128);
                                for (wi, &word) in row.iter().enumerate() {
                                    let mut bits = word;
                                    while bits != 0 {
                                        let t = bits.trailing_zeros() as usize;
                                        acc = acc + Q100Element::from(rw_s[(wi << 6) | t]);
                                        bits &= bits.wrapping_sub(1);
                                    }
                                }
                                y = y + colw[c] * acc;
                            }
                            y.canonical_u128()
                        })
                        .collect();
                    let sh_claims: Vec<RlcSharedClaim> = forms
                        .iter()
                        .zip(cs_s.iter())
                        .map(|(&form, &claimed)| RlcSharedClaim { form, claimed })
                        .collect();
                    let gen_claims: Vec<RlcFamilyClaim<'_>> = forms
                        .iter()
                        .zip(cs_s.iter())
                        .map(|(&form, &claimed)| RlcFamilyClaim {
                            form,
                            row_weights_q: &rw_s,
                            claimed,
                        })
                        .collect();
                    let vx_claims: Vec<VirtualXorClaim<'_>> = col_lists_s
                        .iter()
                        .map(|cl| VirtualXorClaim {
                            cols: cl,
                            constant: 0,
                            external_rows: None,
                            row_weights_q: &rw_s,
                        })
                        .collect();
                    let vx_verify: Vec<VirtualXorVerifyClaim<'_, Q100Element>> = col_lists_s
                        .iter()
                        .zip(cs_s.iter())
                        .map(|(cl, &c)| VirtualXorVerifyClaim {
                            cols: cl,
                            constant: 0,
                            has_external: false,
                            row_weights_q: &rw_s,
                            col_weights: &colw,
                            claimed: Q100Element::from(c),
                        })
                        .collect();

                    // medians: [pS, pG, pVx, pInd, vS, vG, vVx, vInd]
                    let mut t: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); 8];
                    let mut sz = (0usize, 0usize, 0usize);
                    for rep in 0..reps {
                        let t0_recording = bitz::observability::Recording::start(Vec::new())
                            .expect("start operation capture");
                        let t0 = tracing::info_span!("rlc_ab:t0").entered();
                        let pr_s = {
                            let mut pt = Blake3Transcript::new();
                            prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                                &mut pt, &hint, &layout, &family, &rw_s, &sh_claims, alpha, &pc,
                            )
                        };
                        t[0].push(
                            {
                                drop(t0);
                                bitz::observability::duration(
                                    &t0_recording
                                        .intervals()
                                        .expect("complete operation capture"),
                                    "rlc_ab:t0",
                                )
                                .expect("query completed operation")
                            }
                            .as_secs_f64()
                                * 1e3,
                        );
                        let t0_recording = bitz::observability::Recording::start(Vec::new())
                            .expect("start operation capture");
                        let t0 = tracing::info_span!("rlc_ab:t0").entered();
                        {
                            let mut vt = Blake3Transcript::new();
                            verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                                &mut vt,
                                &hint.commitment,
                                &pr_s,
                                &layout,
                                &family,
                                &rw_s,
                                &sh_claims,
                                &colw,
                                alpha,
                                &vc,
                            )
                            .expect("rlcS verifies");
                        }
                        t[4].push(
                            {
                                drop(t0);
                                bitz::observability::duration(
                                    &t0_recording
                                        .intervals()
                                        .expect("complete operation capture"),
                                    "rlc_ab:t0",
                                )
                                .expect("query completed operation")
                            }
                            .as_secs_f64()
                                * 1e3,
                        );

                        let t0_recording = bitz::observability::Recording::start(Vec::new())
                            .expect("start operation capture");
                        let t0 = tracing::info_span!("rlc_ab:t0").entered();
                        let pr_g = {
                            let mut pt = Blake3Transcript::new();
                            prove_mle_eval_mod_q_ligerito_rlc_family(
                                &mut pt,
                                &hint,
                                &layout,
                                &family,
                                &gen_claims,
                                alpha,
                                &pc,
                            )
                        };
                        t[1].push(
                            {
                                drop(t0);
                                bitz::observability::duration(
                                    &t0_recording
                                        .intervals()
                                        .expect("complete operation capture"),
                                    "rlc_ab:t0",
                                )
                                .expect("query completed operation")
                            }
                            .as_secs_f64()
                                * 1e3,
                        );
                        let t0_recording = bitz::observability::Recording::start(Vec::new())
                            .expect("start operation capture");
                        let t0 = tracing::info_span!("rlc_ab:t0").entered();
                        {
                            let mut vt = Blake3Transcript::new();
                            verify_mle_eval_mod_q_ligerito_rlc_family(
                                &mut vt,
                                &hint.commitment,
                                &pr_g,
                                &layout,
                                &family,
                                &gen_claims,
                                &colw,
                                alpha,
                                &vc,
                            )
                            .expect("rlcG verifies");
                        }
                        t[5].push(
                            {
                                drop(t0);
                                bitz::observability::duration(
                                    &t0_recording
                                        .intervals()
                                        .expect("complete operation capture"),
                                    "rlc_ab:t0",
                                )
                                .expect("query completed operation")
                            }
                            .as_secs_f64()
                                * 1e3,
                        );

                        let pr_vx = if run_vx {
                            let t0_recording = bitz::observability::Recording::start(Vec::new())
                                .expect("start operation capture");
                            let t0 = tracing::info_span!("rlc_ab:t0").entered();
                            let pr = {
                                let mut pt = Blake3Transcript::new();
                                prove_mle_eval_mod_q_ligerito_claims_only(
                                    &mut pt, &hint, &layout, FQ_BITS, &vx_claims, alpha, &pc,
                                )
                            };
                            t[2].push(
                                {
                                    drop(t0);
                                    bitz::observability::duration(
                                        &t0_recording
                                            .intervals()
                                            .expect("complete operation capture"),
                                        "rlc_ab:t0",
                                    )
                                    .expect("query completed operation")
                                }
                                .as_secs_f64()
                                    * 1e3,
                            );
                            let t0_recording = bitz::observability::Recording::start(Vec::new())
                                .expect("start operation capture");
                            let t0 = tracing::info_span!("rlc_ab:t0").entered();
                            {
                                let mut vt = Blake3Transcript::new();
                                verify_mle_eval_mod_q_ligerito_claims_only(
                                    &mut vt,
                                    &hint.commitment,
                                    &pr,
                                    &layout,
                                    alpha,
                                    FQ_BITS,
                                    &vx_verify,
                                    &vc,
                                )
                                .expect("vx verifies");
                            }
                            t[6].push(
                                {
                                    drop(t0);
                                    bitz::observability::duration(
                                        &t0_recording
                                            .intervals()
                                            .expect("complete operation capture"),
                                        "rlc_ab:t0",
                                    )
                                    .expect("query completed operation")
                                }
                                .as_secs_f64()
                                    * 1e3,
                            );
                            Some(pr)
                        } else {
                            None
                        };

                        let t0_recording = bitz::observability::Recording::start(Vec::new())
                            .expect("start operation capture");
                        let t0 = tracing::info_span!("rlc_ab:t0").entered();
                        let pr_inds: Vec<_> = (0..k)
                            .map(|i| {
                                let mut pt = Blake3Transcript::new();
                                prove_mle_eval_mod_q_ligerito_claims_only(
                                    &mut pt,
                                    &hint,
                                    &layout,
                                    FQ_BITS,
                                    core::slice::from_ref(&vx_claims[i]),
                                    alpha,
                                    &pc,
                                )
                            })
                            .collect();
                        t[3].push(
                            {
                                drop(t0);
                                bitz::observability::duration(
                                    &t0_recording
                                        .intervals()
                                        .expect("complete operation capture"),
                                    "rlc_ab:t0",
                                )
                                .expect("query completed operation")
                            }
                            .as_secs_f64()
                                * 1e3,
                        );
                        let t0_recording = bitz::observability::Recording::start(Vec::new())
                            .expect("start operation capture");
                        let t0 = tracing::info_span!("rlc_ab:t0").entered();
                        for (i, pr) in pr_inds.iter().enumerate() {
                            let mut vt = Blake3Transcript::new();
                            verify_mle_eval_mod_q_ligerito_claims_only(
                                &mut vt,
                                &hint.commitment,
                                pr,
                                &layout,
                                alpha,
                                FQ_BITS,
                                core::slice::from_ref(&vx_verify[i]),
                                &vc,
                            )
                            .unwrap_or_else(|e| panic!("ind[{i}] verifies: {e:?}"));
                        }
                        t[7].push(
                            {
                                drop(t0);
                                bitz::observability::duration(
                                    &t0_recording
                                        .intervals()
                                        .expect("complete operation capture"),
                                    "rlc_ab:t0",
                                )
                                .expect("query completed operation")
                            }
                            .as_secs_f64()
                                * 1e3,
                        );

                        if rep == 0 {
                            sz = (
                                mle_eval_mod_q_lig_rlc_family_proof_size_bytes(&pr_s),
                                pr_vx
                                    .as_ref()
                                    .map_or(0, mle_eval_mod_q_lig_xor_proof_size_bytes),
                                pr_inds
                                    .iter()
                                    .map(mle_eval_mod_q_lig_xor_proof_size_bytes)
                                    .sum(),
                            );
                        }
                    }
                    let med: Vec<f64> = t
                        .iter()
                        .map(|v| {
                            if v.is_empty() {
                                f64::NAN
                            } else {
                                median(v.clone())
                            }
                        })
                        .collect();
                    for (a, &m) in agg.iter_mut().zip(med.iter()) {
                        a.push(m);
                    }
                    println!(
                        "n={n} shared j{j}(k={k}) stmt{stmt}: rlcS {:.1} ms ({} B, v {:.1}) | \
                         rlcG {:.1} ms (v {:.1}) | vx {:.1} ms ({} B, v {:.1}) | \
                         ind {:.1} ms ({} B, v {:.1}) | rlcS/claim {:.1} ms",
                        med[0],
                        sz.0,
                        med[4],
                        med[1],
                        med[5],
                        med[2],
                        sz.1,
                        med[6],
                        med[3],
                        sz.2,
                        med[7],
                        med[0] / k as f64,
                    );
                }
                if stmts > 1 {
                    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
                    println!(
                        "n={n} shared j{j}(k={k}) MEAN of {stmts} stmts: rlcS {:.1} ms (v {:.1}) | \
                         rlcG {:.1} ms (v {:.1}) | vx {:.1} ms (v {:.1}) | ind {:.1} ms (v {:.1})",
                        mean(&agg[0]),
                        mean(&agg[4]),
                        mean(&agg[1]),
                        mean(&agg[5]),
                        mean(&agg[2]),
                        mean(&agg[6]),
                        mean(&agg[3]),
                        mean(&agg[7]),
                    );
                }
                // Phase trees for one profiled prove + verify (the
                // rlc:casew / rlc:pows prover shares and the rlcv:* verify
                // attribution — the rank-1 / case-table questions).
                if profile {
                    let rw_p: Vec<u128> = (0..p_x.rows())
                        .map(|b| {
                            (b as u128)
                                .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                                .wrapping_add(71)
                                % FQ_MOD
                        })
                        .collect();
                    let cs_p: Vec<u128> = col_lists_s
                        .iter()
                        .map(|cl| {
                            let a_rows =
                                extract_virtual_xor_rows(&layout, hint.rows(), cl, 0, None);
                            let mut y = Q100Element::from(0u128);
                            for (c, row) in a_rows.iter().enumerate() {
                                let mut acc = Q100Element::from(0u128);
                                for (wi, &word) in row.iter().enumerate() {
                                    let mut bits = word;
                                    while bits != 0 {
                                        let t = bits.trailing_zeros() as usize;
                                        acc = acc + Q100Element::from(rw_p[(wi << 6) | t]);
                                        bits &= bits.wrapping_sub(1);
                                    }
                                }
                                y = y + colw[c] * acc;
                            }
                            y.canonical_u128()
                        })
                        .collect();
                    let cl_p: Vec<RlcSharedClaim> = forms
                        .iter()
                        .zip(cs_p.iter())
                        .map(|(&form, &claimed)| RlcSharedClaim { form, claimed })
                        .collect();
                    let mut pt = Blake3Transcript::new();
                    let profile =
                        bitz::observability::Recording::start(Vec::new()).expect("capture profile");
                    let pr = prove_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut pt, &hint, &layout, &family, &rw_p, &cl_p, alpha, &pc,
                    );
                    bitz::observability::write_profile(
                        std::io::stderr().lock(),
                        &format!("rlcS prove j{j} n={n}"),
                        &profile.intervals().expect("profile intervals"),
                        None,
                    )
                    .expect("write profile");
                    let profile =
                        bitz::observability::Recording::start(Vec::new()).expect("capture profile");
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_rlc_family_shared_point(
                        &mut vt,
                        &hint.commitment,
                        &pr,
                        &layout,
                        &family,
                        &rw_p,
                        &cl_p,
                        &colw,
                        alpha,
                        &vc,
                    )
                    .expect("profiled rlcS verifies");
                    bitz::observability::write_profile(
                        std::io::stderr().lock(),
                        &format!("rlcS verify j{j} n={n}"),
                        &profile.intervals().expect("profile intervals"),
                        None,
                    )
                    .expect("write profile");
                }
            }
        }

        // Timed variants, alternated in-window.
        let mut t_single = Vec::with_capacity(reps);
        let mut t_rlc3 = Vec::with_capacity(reps);
        let mut t_vx3 = Vec::with_capacity(reps);
        let mut t_ind3 = Vec::with_capacity(reps);
        let mut sizes = (0usize, 0usize, 0usize); // rlc3, vx3, ind3
        for rep in 0..reps {
            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("rlc_ab:t0").entered();
            let pr_single = {
                let mut pt = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_claims_only(
                    &mut pt,
                    &hint,
                    &layout,
                    FQ_BITS,
                    &vx_of(&[0]),
                    alpha,
                    &pc,
                )
            };
            t_single.push(
                {
                    drop(t0);
                    bitz::observability::duration(
                        &t0_recording
                            .intervals()
                            .expect("complete operation capture"),
                        "rlc_ab:t0",
                    )
                    .expect("query completed operation")
                }
                .as_secs_f64()
                    * 1e3,
            );
            std::hint::black_box(&pr_single);

            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("rlc_ab:t0").entered();
            let pr_rlc = {
                let mut pt = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_rlc_family(
                    &mut pt,
                    &hint,
                    &layout,
                    &family_cols,
                    &claims,
                    alpha,
                    &pc,
                )
            };
            t_rlc3.push(
                {
                    drop(t0);
                    bitz::observability::duration(
                        &t0_recording
                            .intervals()
                            .expect("complete operation capture"),
                        "rlc_ab:t0",
                    )
                    .expect("query completed operation")
                }
                .as_secs_f64()
                    * 1e3,
            );
            std::hint::black_box(&pr_rlc);

            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("rlc_ab:t0").entered();
            let pr_vx3 = {
                let mut pt = Blake3Transcript::new();
                prove_mle_eval_mod_q_ligerito_claims_only(
                    &mut pt,
                    &hint,
                    &layout,
                    FQ_BITS,
                    &vx_of(&[0, 1, 2]),
                    alpha,
                    &pc,
                )
            };
            t_vx3.push(
                {
                    drop(t0);
                    bitz::observability::duration(
                        &t0_recording
                            .intervals()
                            .expect("complete operation capture"),
                        "rlc_ab:t0",
                    )
                    .expect("query completed operation")
                }
                .as_secs_f64()
                    * 1e3,
            );
            std::hint::black_box(&pr_vx3);

            let t0_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t0 = tracing::info_span!("rlc_ab:t0").entered();
            let pr_inds: Vec<_> = (0..3)
                .map(|i| {
                    let mut pt = Blake3Transcript::new();
                    prove_mle_eval_mod_q_ligerito_claims_only(
                        &mut pt,
                        &hint,
                        &layout,
                        FQ_BITS,
                        &vx_of(&[i]),
                        alpha,
                        &pc,
                    )
                })
                .collect();
            t_ind3.push(
                {
                    drop(t0);
                    bitz::observability::duration(
                        &t0_recording
                            .intervals()
                            .expect("complete operation capture"),
                        "rlc_ab:t0",
                    )
                    .expect("query completed operation")
                }
                .as_secs_f64()
                    * 1e3,
            );
            std::hint::black_box(&pr_inds);

            if rep == 0 {
                sizes = (
                    mle_eval_mod_q_lig_rlc_family_proof_size_bytes(&pr_rlc),
                    mle_eval_mod_q_lig_xor_proof_size_bytes(&pr_vx3),
                    pr_inds
                        .iter()
                        .map(mle_eval_mod_q_lig_xor_proof_size_bytes)
                        .sum(),
                );
                // Sanity: every variant verifies.
                let vx_vc = |idx: &[usize]| -> Vec<VirtualXorVerifyClaim<'_, Q100Element>> {
                    idx.iter()
                        .map(|&i| VirtualXorVerifyClaim {
                            cols: &col_lists[i],
                            constant: 0,
                            has_external: false,
                            row_weights_q: &rws[i],
                            col_weights: &colw,
                            claimed: Q100Element::from(cs[i]),
                        })
                        .collect()
                };
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_rlc_family(
                    &mut vt,
                    &hint.commitment,
                    &pr_rlc,
                    &layout,
                    &family_cols,
                    &claims,
                    &colw,
                    alpha,
                    &vc,
                )
                .expect("rlc3 verifies");
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_claims_only(
                    &mut vt,
                    &hint.commitment,
                    &pr_vx3,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &vx_vc(&[0, 1, 2]),
                    &vc,
                )
                .expect("vx3 verifies");
                for (i, pr) in pr_inds.iter().enumerate() {
                    let mut vt = Blake3Transcript::new();
                    verify_mle_eval_mod_q_ligerito_claims_only(
                        &mut vt,
                        &hint.commitment,
                        pr,
                        &layout,
                        alpha,
                        FQ_BITS,
                        &vx_vc(&[i]),
                        &vc,
                    )
                    .unwrap_or_else(|e| panic!("ind3[{i}] verifies: {e:?}"));
                }
                let mut vt = Blake3Transcript::new();
                verify_mle_eval_mod_q_ligerito_claims_only(
                    &mut vt,
                    &hint.commitment,
                    &pr_single,
                    &layout,
                    alpha,
                    FQ_BITS,
                    &vx_vc(&[0]),
                    &vc,
                )
                .expect("single verifies");
            }
        }
        let (mu, mr, mb, mi) = (
            median(t_single),
            median(t_rlc3),
            median(t_vx3),
            median(t_ind3),
        );
        println!(
            "n={n} (t'={}, s={}, m={}) reps={reps}\n  single {mu:8.1} ms\n  rlc3   {mr:8.1} ms  ({:.2}x single)  proof {} B\n  vx3    {mb:8.1} ms  ({:.2}x single)  proof {} B\n  ind3   {mi:8.1} ms  ({:.2}x single)  proof {} B",
            p_x.row_vars,
            p_x.col_vars,
            m_p + 7,
            mr / mu,
            sizes.0,
            mb / mu,
            sizes.1,
            mi / mu,
            sizes.2,
        );

        if profile {
            let mut pt = Blake3Transcript::new();
            let profile =
                bitz::observability::Recording::start(Vec::new()).expect("capture profile");
            let pr = prove_mle_eval_mod_q_ligerito_rlc_family(
                &mut pt,
                &hint,
                &layout,
                &family_cols,
                &claims,
                alpha,
                &pc,
            );
            std::hint::black_box(&pr);
            bitz::observability::write_profile(
                std::io::stderr().lock(),
                &format!("rlc3 n={n}"),
                &profile.intervals().expect("profile intervals"),
                None,
            )
            .expect("write profile");
            let mut pt = Blake3Transcript::new();
            let profile =
                bitz::observability::Recording::start(Vec::new()).expect("capture profile");
            let pr = prove_mle_eval_mod_q_ligerito_claims_only(
                &mut pt,
                &hint,
                &layout,
                FQ_BITS,
                &vx_of(&[0, 1, 2]),
                alpha,
                &pc,
            );
            std::hint::black_box(&pr);
            bitz::observability::write_profile(
                std::io::stderr().lock(),
                &format!("vx3 n={n}"),
                &profile.intervals().expect("profile intervals"),
                None,
            )
            .expect("write profile");
        }
    }
}
