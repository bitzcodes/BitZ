//! Historical kernel experiment; no production security claim.
//! Byte-identity pin for the end-to-end proof stream: proves a fixed
//! deterministic instance per shape and prints the BLAKE3 digest of the
//! serialized proof plus the commitment root. Run before and after any
//! dependency or kernel change that claims to be transcript-preserving —
//! matching digests mean byte-identical proofs (and, since the commitment
//! root and every challenge derive from the same stream, byte-identical
//! commits and transcripts too).
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example proof_digest
//! ```

use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{
    commit_rs_ligerito_rows, historical_sha_lig_configs, prove_mle_eval_mod_q_ligerito,
    verify_mle_eval_mod_q_ligerito,
};
use bitz::pcs::{IntegerMatrixLayout, mod_q_num_chunks, smallest_generator};

/// `𝔽_q`, `q = 2^100 − 15`.
const Q: u128 = (1u128 << 100) - 15;

#[derive(Clone, Copy, PartialEq, Debug)]
struct Fq(u128);
impl From<u128> for Fq {
    fn from(v: u128) -> Self {
        Fq(v % Q)
    }
}
impl core::ops::Add for Fq {
    type Output = Fq;
    fn add(self, o: Fq) -> Fq {
        let s = self.0 + o.0;
        Fq(if s >= Q { s - Q } else { s })
    }
}
impl core::ops::Mul for Fq {
    type Output = Fq;
    fn mul(self, o: Fq) -> Fq {
        let (mut a, mut b, mut acc) = (self.0, o.0, 0u128);
        while b != 0 {
            if b & 1 == 1 {
                let s = acc + a;
                acc = if s >= Q { s - Q } else { s };
            }
            let d = a << 1;
            a = if d >= Q { d - Q } else { d };
            b >>= 1;
        }
        Fq(acc)
    }
}

fn digest(t: usize, s: usize, w: usize) {
    let alpha = smallest_generator();
    let q_bits = 100usize;
    let p = IntegerMatrixLayout {
        row_vars: t,
        col_vars: s,
        word_bits: w,
    };
    let m_p = packed_vars(&p);
    let lch = mod_q_num_chunks(&p, q_bits);
    let (pc, vc) = historical_sha_lig_configs(m_p).expect("lig cfg");

    let mask = if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
    let cell = |b: usize, c: usize| -> u128 {
        (p.cell_index(b, c) as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask
    };
    let log_w = w.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    let words = row_len.div_ceil(64);
    let rows: Vec<Vec<u64>> = (0..p.cols())
        .map(|c| {
            let mut wv = vec![0u64; words];
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
    let rw_q: Vec<u128> = (0..p.rows())
        .map(|b| {
            (b as u128)
                .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                .wrapping_add(7)
                % Q
        })
        .collect();
    let cw: Vec<Fq> = (0..p.cols())
        .map(|c| Fq::from(((c as u128).wrapping_mul(5) & 7).wrapping_add(1)))
        .collect();
    let rw_fq: Vec<Fq> = rw_q.iter().map(|&x| Fq::from(x)).collect();
    let mut y = Fq::from(0u128);
    for (c, row) in rows.iter().enumerate() {
        let mut vc_acc = Fq::from(0u128);
        for (wi, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let i = (wi << 6) | bit;
                let (b, j) = (i >> log_w, i & (w - 1));
                let term = if j == 0 { rw_fq[b] } else { rw_fq[b] * Fq::from(1u128 << j) };
                vc_acc = vc_acc + term;
            }
        }
        y = y + cw[c] * vc_acc;
    }

    let hint = commit_rs_ligerito_rows(&p, rows, &pc);
    let root_hex: String = hint.commitment.root.iter().map(|b| format!("{b:02x}")).collect();

    let mut pt = bitz::transcript::Blake3Transcript::new();
    let proof = prove_mle_eval_mod_q_ligerito(&mut pt, &hint, &p, &rw_q, q_bits, alpha, &pc);
    let ser = proof.to_bytes();
    let dg = blake3::hash(&ser);

    let mut vt = bitz::transcript::Blake3Transcript::new();
    verify_mle_eval_mod_q_ligerito(
        &mut vt, &hint.commitment, &proof, &p, &rw_q, &cw, alpha, y, q_bits, &vc,
    )
    .expect("verify");

    let n = t + s;
    println!(
        "n={n:2} (t={t}, s={s}, W={w}, m_p={m_p}, chunks={lch}): {} B  root {}  proof-digest {}",
        ser.len(),
        &root_hex[..16],
        dg.to_hex(),
    );
}

fn main() {
    println!("BitZ proof byte-identity digests (deterministic instances)\n");
    digest(12, 6, 1); // n=18: ad-hoc config regime
    digest(13, 9, 1); // n=22: embedded FAST config regime
    digest(4, 8, 32); // 2-chunk (W=32) regime
}
