//! Historical kernel experiment; no production security claim.
//! Reference measurement for the BitZ ring-switch + Ligerito opener: prove /
//! verify wall-clock and serialized proof size at a few `n = t + s` shapes.
//!
//! Run with:
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --features span-metrics --example reference_measure
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure(t: usize, s: usize, w: usize, reps: usize) {
    let alpha = smallest_generator();
    let q_bits = 100usize;
    let p = IntegerMatrixLayout {
        row_vars: t,
        col_vars: s,
        word_bits: w,
    };
    let m_p = packed_vars(&p);
    let lch = mod_q_num_chunks(&p, q_bits);
    // The library's audited config boundary (embedded FAST at m ≥ 22).
    let (pc, vc) = historical_sha_lig_configs(m_p).expect("lig cfg");

    // Instance generated straight into the per-column bit rows — the
    // u128 cell tensor never exists (the memory-honest commit path).
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

    let mut prove_ms = Vec::new();
    let mut verify_ms = Vec::new();
    let mut bytes = 0usize;
    for _ in 0..reps {
        let mut pt = bitz::transcript::Blake3Transcript::new();
        let (proof, t0) =
            bitz::observability::measure(tracing::info_span!("reference_measure:proof"), || {
                prove_mle_eval_mod_q_ligerito(&mut pt, &hint, &p, &rw_q, q_bits, alpha, &pc)
            })
            .expect("measure completed operation");
        prove_ms.push(t0.as_secs_f64() * 1e3);

        let ser = proof.to_bytes();
        bytes = ser.len();

        let mut vt = bitz::transcript::Blake3Transcript::new();
        let t1_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t1 = tracing::info_span!("reference_measure:t1").entered();
        verify_mle_eval_mod_q_ligerito(
            &mut vt,
            &hint.commitment,
            &proof,
            &p,
            &rw_q,
            &cw,
            alpha,
            y,
            q_bits,
            &vc,
        )
        .expect("verify");
        verify_ms.push({ drop(t1); bitz::observability::duration(&t1_recording.intervals().expect("complete operation capture"), "reference_measure:t1").expect("query completed operation") }.as_secs_f64() * 1e3);
    }

    let n = t + s;
    println!(
        "n={n:2} (t={t}, s={s}, W={w}, m_p={m_p}, chunks={lch}): \
         prove {:7.2} ms | verify {:6.2} ms | proof {:6} B ({:.1} KiB)",
        median(prove_ms),
        median(verify_ms),
        bytes,
        bytes as f64 / 1024.0,
    );
}

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    println!("BitZ ring-switch + Ligerito opener — reference measurement (median of 5)\n");
    measure(10, 6, 1, 5); // n=16 headline
    measure(12, 6, 1, 5); // n=18
    measure(4, 8, 32, 5); // 2-chunk (W=32) regime
}
