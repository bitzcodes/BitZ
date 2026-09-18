//! Historical kernel experiment; no production security claim.
//! Byte-identity pin for the eq-factored pass-fusion experiment: dump one
//! deterministic proof's bytes to a file — run under different `BITZ_EQF_*`
//! flag combinations and `cmp` the outputs (the flags are process-cached,
//! so the comparison is cross-process).
//!
//! ```text
//! cargo run --release --example fuse_check --features unchecked -- /tmp/base.bin
//! BITZ_EQF_FUSE=1 cargo run --release --example fuse_check --features unchecked -- /tmp/fuse.bin
//! cmp /tmp/base.bin /tmp/fuse.bin
//! ```

use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{commit_rs_ligerito_rows, prove_mle_eval_mod_q_ligerito, historical_sha_lig_configs};
use bitz::pcs::{IntegerMatrixLayout, smallest_generator};

const Q: u128 = (1u128 << 100) - 15;

fn main() {
    let out_path = std::env::args().nth(1).expect("usage: fuse_check <out-file>");
    let alpha = smallest_generator();
    let q_bits = 100usize;
    let mut all = Vec::new();
    // Two shapes: one exercising every layer type at real depth, one small.
    for (t, s) in [(14usize, 8usize), (10, 5)] {
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
        let mut pt = bitz::transcript::Blake3Transcript::new();
        let proof = prove_mle_eval_mod_q_ligerito(&mut pt, &hint, &p, &rw_q, q_bits, alpha, &pc);
        all.extend_from_slice(&proof.to_bytes());
    }
    std::fs::write(&out_path, &all).expect("write proof bytes");
    println!("wrote {} bytes to {out_path}", all.len());
}
