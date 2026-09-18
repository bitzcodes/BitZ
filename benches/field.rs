//! Field micro-benchmark: `GF(2^128)` (GHASH) vs `GF(2^127)` (b127),
//! head-to-head on the PCS prover's actual hot patterns.
//!
//! Plain `harness = false` binary (no criterion), in the style of
//! `benches/pcs.rs`. Six patterns, each a proxy for a prover phase:
//!
//! | pattern      | proxy for                                              |
//! |--------------|--------------------------------------------------------|
//! | `mul/batch`  | forest layer products (independent, throughput-bound)  |
//! | `mul/chain`  | dependent product chains (latency-bound)               |
//! | `square`     | α-power place-value / comb window advances (latency)   |
//! | `powers`     | `FixedBasePow` comb, win 8, ~100-bit exponents — the   |
//! |              | `chunk_pow2_table` / root-recompute workload (and the  |
//! |              | reilabs `ghash-powers-bench` shape)                    |
//! | `wide-dot`   | delayed-reduction inner products (`WideMulAcc`)        |
//! | `eqf-round`  | the fused single-pair sumcheck round kernel            |
//! | `eqf-fold`   | the in-place multilinear bind cascade                  |
//!
//! Run (the NEON pipeline is target-feature-gated — the flag is
//! load-bearing):
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench field
//! ```
//! Knobs: `BITZ_BENCH_REPS` (timing repetitions per pattern, median
//! reported; default 5). Idle the box; expect ±5 % run-to-run.

mod common;

use std::hint::black_box;

use bitz::poly::univariate::binary_b127::B127;
use bitz::poly::univariate::binary_gf128::Gf128;
use bitz::utils::wide_mul::WideMulAcc;

// ---------------------------------------------------------------------
// Deterministic data (no rand dep in benches — the pcs.rs convention).
// ---------------------------------------------------------------------

/// splitmix64 — deterministic, well-mixed 64-bit stream.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_u128(state: &mut u64) -> u128 {
    (splitmix(state) as u128) | ((splitmix(state) as u128) << 64)
}

// ---------------------------------------------------------------------
// The field abstraction the harness is generic over. Monomorphized —
// no dispatch in the timed loops (mirrors the reilabs bench design).
// ---------------------------------------------------------------------

trait BF:
    Copy
    + PartialEq
    + core::fmt::Display
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + WideMulAcc
{
    // Self-labeling for ad-hoc printouts; the table harness uses fixed
    // column headers instead, so this is reference-only.
    #[allow(dead_code)]
    const NAME: &'static str;
    fn zero() -> Self;
    fn one() -> Self;
    /// Deterministic element from a 128-bit pattern (b127 folds bit 127).
    fn from_u128(v: u128) -> Self;
    fn square(&self) -> Self;
    fn inverse(&self) -> Self;
}

impl BF for Gf128 {
    const NAME: &'static str = "GF(2^128) GHASH";
    fn zero() -> Self {
        Gf128::zero()
    }
    fn one() -> Self {
        Gf128::one()
    }
    fn from_u128(v: u128) -> Self {
        Gf128::from_polynomial_bits(v)
    }
    fn square(&self) -> Self {
        Gf128::square(*self)
    }
    fn inverse(&self) -> Self {
        Gf128::inverse(self)
    }
}

impl BF for B127 {
    const NAME: &'static str = "GF(2^127) b127";
    fn zero() -> Self {
        B127::zero()
    }
    fn one() -> Self {
        B127::one()
    }
    fn from_u128(v: u128) -> Self {
        B127::from_polynomial_bits(v)
    }
    fn square(&self) -> Self {
        B127::square(*self)
    }
    fn inverse(&self) -> Self {
        B127::inverse(self)
    }
}

fn gen_vec<F: BF>(n: usize, seed: u64) -> Vec<F> {
    let mut st = seed;
    (0..n).map(|_| F::from_u128(rand_u128(&mut st))).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time the two fields' bodies (each processing `ops` field operations)
/// with ALTERNATING reps — g, b, g, b, … — so both fields sample the
/// same thermal/clock window and the RATIO is insulated from drift
/// (single-sided sweeps measured ±10–19 % swings on this box between
/// otherwise-identical runs). Returns (median g, median b) ns/op.
fn time_pair_ns_per_op<RG, RB>(
    reps: usize,
    ops: usize,
    mut g_body: impl FnMut() -> RG,
    mut b_body: impl FnMut() -> RB,
) -> (f64, f64) {
    black_box(g_body()); // warm-ups
    black_box(b_body());
    let mut gs = Vec::with_capacity(reps);
    let mut bs = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0_recording =
            bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t0 = tracing::info_span!("field:t0").entered();
        black_box(g_body());
        gs.push(
            {
                drop(t0);
                bitz::observability::duration(
                    &t0_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "field:t0",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e9
                / ops as f64,
        );
        let t1_recording =
            bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t1 = tracing::info_span!("field:t1").entered();
        black_box(b_body());
        bs.push(
            {
                drop(t1);
                bitz::observability::duration(
                    &t1_recording
                        .intervals()
                        .expect("complete operation capture"),
                    "field:t1",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e9
                / ops as f64,
        );
    }
    (median(gs), median(bs))
}

// ---------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------

const N_BATCH: usize = 1 << 21; // independent muls
const N_CHAIN: usize = 1 << 21; // dependent muls
const N_SQ: usize = 1 << 22; // dependent squarings
const N_INV: usize = 1 << 12; // independent inversions (Itoh–Tsujii)
const N_POW: usize = 1 << 16; // comb exponentiations
const POW_BITS: usize = 100; // mod-q row-weight width (q = 2^100 − 15)
const POW_WIN: usize = 8; // chunk_pow2_table's window
const N_WIDE: usize = 1 << 21; // wide-dot terms
const EQF_HALF: usize = 1 << 20; // eqf round slots
const N_FOLD: usize = 1 << 21; // fold-cascade start size

/// out[i] = a[i]·b[i] — the forest-layer shape.
fn batch_mul<F: BF>(a: &[F], b: &[F], out: &mut [F]) {
    for ((x, y), o) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *o = *x * *y;
    }
}

/// acc = ∏ v[i] — the dependent-latency shape.
fn chain_mul<F: BF>(v: &[F]) -> F {
    let mut acc = F::one();
    for x in v {
        acc = acc * *x;
    }
    acc
}

/// x ← x² N times — the squaring-chain shape.
fn square_chain<F: BF>(x0: F, n: usize) -> F {
    let mut x = x0;
    for _ in 0..n {
        x = x.square();
    }
    x
}

/// Fixed-base comb (the `pcs::FixedBasePow` construction, made generic
/// for the head-to-head): `table[i][d] = α^{d·2^{win·i}}`.
struct Comb<F> {
    table: Vec<Vec<F>>,
    win: usize,
}

impl<F: BF> Comb<F> {
    fn new(alpha: F, max_bits: usize, win: usize) -> Self {
        let num_windows = max_bits.div_ceil(win);
        let radix = 1usize << win;
        let mut table = Vec::with_capacity(num_windows);
        let mut base_i = alpha;
        for _ in 0..num_windows {
            let mut row = Vec::with_capacity(radix);
            let mut cur = F::one();
            for _ in 0..radix {
                row.push(cur);
                cur = cur * base_i;
            }
            table.push(row);
            for _ in 0..win {
                base_i = base_i.square();
            }
        }
        Self { table, win }
    }

    fn pow(&self, mut exp: u128) -> F {
        let mask = (1u128 << self.win).wrapping_sub(1);
        let mut acc = F::one();
        let mut i = 0usize;
        while exp != 0 {
            let d = (exp & mask) as usize;
            if d != 0 {
                acc = acc * self.table[i][d];
            }
            exp >>= self.win;
            i += 1;
        }
        acc
    }
}

/// Σ a[i]·b[i] through the delayed-reduction accumulator — the sumcheck
/// inner-product shape.
fn wide_dot<F: BF>(a: &[F], b: &[F]) -> F {
    let mut acc = F::wide_zero(&F::zero());
    for (x, y) in a.iter().zip(b.iter()) {
        F::wide_add_assign(&mut acc, &F::mul_wide(x, y));
    }
    F::from_wide(acc)
}

/// The bind cascade: fold 2^k → 2^{k-1} → … → 1 in place (each level is
/// `eqf_fold_in_place` at successive halves) — ~n muls total.
fn fold_cascade<F: BF>(v: &mut [F], rhos: &[F]) -> F {
    let mut len = v.len();
    let mut level = 0usize;
    while len > 1 {
        let half = len >> 1;
        assert!(F::eqf_fold_in_place(v, &rhos[level % rhos.len()], half));
        len = half;
        level += 1;
    }
    v[0]
}

// ---------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------

fn print_row(pattern: &str, g: f64, b: f64) {
    println!("{:<22} {:>14.3} {:>14.3} {:>8.2}x", pattern, g, b, g / b);
}

/// Every pattern, both fields, alternating reps within the pattern.
fn run_paired<G: BF, B: BF>(reps: usize) {
    // mul/batch
    let (ga, gb) = (gen_vec::<G>(N_BATCH, 0xA11CE), gen_vec::<G>(N_BATCH, 0xB0B));
    let (ba, bb) = (gen_vec::<B>(N_BATCH, 0xA11CE), gen_vec::<B>(N_BATCH, 0xB0B));
    let mut go = vec![G::zero(); N_BATCH];
    let mut bo = vec![B::zero(); N_BATCH];
    let (g, b) = time_pair_ns_per_op(
        reps,
        N_BATCH,
        || {
            batch_mul(black_box(&ga), black_box(&gb), black_box(&mut go));
            go[N_BATCH - 1]
        },
        || {
            batch_mul(black_box(&ba), black_box(&bb), black_box(&mut bo));
            bo[N_BATCH - 1]
        },
    );
    print_row("mul/batch", g, b);
    drop((go, bo));

    // mul/chain
    let gv = gen_vec::<G>(N_CHAIN, 0xC0FFEE);
    let bv = gen_vec::<B>(N_CHAIN, 0xC0FFEE);
    let (g, b) = time_pair_ns_per_op(
        reps,
        N_CHAIN,
        || chain_mul(black_box(&gv)),
        || chain_mul(black_box(&bv)),
    );
    print_row("mul/chain", g, b);
    drop((gv, bv));

    // square chain
    let seed = 0x1234_5678_9ABC_DEF0_0FED_CBA9_8765_4321u128;
    let (gx, bx) = (G::from_u128(seed), B::from_u128(seed));
    let (g, b) = time_pair_ns_per_op(
        reps,
        N_SQ,
        || square_chain(black_box(gx), N_SQ),
        || square_chain(black_box(bx), N_SQ),
    );
    print_row("square/chain", g, b);

    // inverse (Itoh–Tsujii ladder, independent elements — throughput)
    let ginv = gen_vec::<G>(N_INV, 0x1517);
    let binv = gen_vec::<B>(N_INV, 0x1517);
    let (g, b) = time_pair_ns_per_op(
        reps,
        N_INV,
        || {
            let mut acc = G::zero();
            for x in &ginv {
                acc = acc + x.inverse();
            }
            acc
        },
        || {
            let mut acc = B::zero();
            for x in &binv {
                acc = acc + x.inverse();
            }
            acc
        },
    );
    print_row("inverse", g, b);
    drop((ginv, binv));

    // powers (comb win 8, 100-bit exponents)
    let gcomb = Comb::new(G::from_u128(2), 128, POW_WIN);
    let bcomb = Comb::new(B::from_u128(2), 128, POW_WIN);
    let mut st = 0xE44_u64;
    let exps: Vec<u128> = (0..N_POW)
        .map(|_| rand_u128(&mut st) & ((1u128 << POW_BITS) - 1))
        .collect();
    let (g, b) = time_pair_ns_per_op(
        reps,
        N_POW,
        || {
            let mut acc = G::zero();
            for &e in &exps {
                acc = acc + gcomb.pow(black_box(e));
            }
            acc
        },
        || {
            let mut acc = B::zero();
            for &e in &exps {
                acc = acc + bcomb.pow(black_box(e));
            }
            acc
        },
    );
    print_row("powers/comb-w8-100b", g, b);

    // wide-dot
    let (gwa, gwb) = (gen_vec::<G>(N_WIDE, 0xD07), gen_vec::<G>(N_WIDE, 0xD08));
    let (bwa, bwb) = (gen_vec::<B>(N_WIDE, 0xD07), gen_vec::<B>(N_WIDE, 0xD08));
    let (g, b) = time_pair_ns_per_op(
        reps,
        N_WIDE,
        || wide_dot(black_box(&gwa), black_box(&gwb)),
        || wide_dot(black_box(&bwa), black_box(&bwb)),
    );
    print_row("wide-dot", g, b);
    drop((gwa, gwb, bwa, bwb));

    // eqf round (the fused sumcheck kernel; 3 products + 2 weight-folds
    // per slot → count 5·half products)
    let (gl, gr, gw) = (
        gen_vec::<G>(2 * EQF_HALF, 0xE9F1),
        gen_vec::<G>(2 * EQF_HALF, 0xE9F2),
        gen_vec::<G>(EQF_HALF, 0xE9F3),
    );
    let (bl, br, bw) = (
        gen_vec::<B>(2 * EQF_HALF, 0xE9F1),
        gen_vec::<B>(2 * EQF_HALF, 0xE9F2),
        gen_vec::<B>(EQF_HALF, 0xE9F3),
    );
    let (g, b) = time_pair_ns_per_op(
        reps,
        5 * EQF_HALF,
        || {
            G::eqf_single_pair_round(black_box(&gl), black_box(&gr), black_box(&gw), EQF_HALF)
                .expect("fused kernel")
        },
        || {
            B::eqf_single_pair_round(black_box(&bl), black_box(&br), black_box(&bw), EQF_HALF)
                .expect("fused kernel")
        },
    );
    print_row("eqf-round", g, b);
    drop((gl, gr, gw, bl, br, bw));

    // eqf fold cascade (~N_FOLD muls total across all levels)
    let g_rhos = gen_vec::<G>(24, 0xF01D);
    let b_rhos = gen_vec::<B>(24, 0xF01D);
    let g_base = gen_vec::<G>(N_FOLD, 0xF01E);
    let b_base = gen_vec::<B>(N_FOLD, 0xF01E);
    let mut g_buf = g_base.clone();
    let mut b_buf = b_base.clone();
    let (g, b) = time_pair_ns_per_op(
        reps,
        N_FOLD,
        || {
            g_buf.copy_from_slice(&g_base);
            fold_cascade(black_box(&mut g_buf), black_box(&g_rhos))
        },
        || {
            b_buf.copy_from_slice(&b_base);
            fold_cascade(black_box(&mut b_buf), black_box(&b_rhos))
        },
    );
    print_row("eqf-fold", g, b);
}

/// B127-only extra: the 3-PMULL Karatsuba product alternative vs the
/// schoolbook default, alternating on the batch-mul shape.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn run_b127_kara(reps: usize) {
    let a = gen_vec::<B127>(N_BATCH, 0xA11CE);
    let b = gen_vec::<B127>(N_BATCH, 0xB0B);
    let mut o1 = vec![B127::zero(); N_BATCH];
    let mut o2 = vec![B127::zero(); N_BATCH];
    let (school, kara) = time_pair_ns_per_op(
        reps,
        N_BATCH,
        || {
            batch_mul(black_box(&a), black_box(&b), black_box(&mut o1));
            o1[N_BATCH - 1]
        },
        || {
            for ((x, y), out) in a.iter().zip(b.iter()).zip(o2.iter_mut()) {
                *out = x.mul_karatsuba(*y);
            }
            o2[N_BATCH - 1]
        },
    );
    println!(
        "{:<22} {:>14} {:>14.3}   (vs b127 schoolbook {:.3}: {:.2}x)",
        "mul/batch b127-kara",
        "—",
        kara,
        school,
        school / kara
    );
}

/// B127-only extra: the GHASH-shaped PMULL-fold multiply (3-PMULL `0x6`
/// reduction + bit-127 canonicalization) paired against the GF128
/// baseline itself on the batch shape — the "buy the reduction with
/// PMULLs, exactly like GHASH does" endpoint. The sequence is
/// structurally GHASH's plus the canonicalization tax, so parity is its
/// ceiling; the row measures the tax.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn run_b127_pfold(reps: usize) {
    let ga = gen_vec::<Gf128>(N_BATCH, 0xA11CE);
    let gb = gen_vec::<Gf128>(N_BATCH, 0xB0B);
    let ba = gen_vec::<B127>(N_BATCH, 0xA11CE);
    let bb = gen_vec::<B127>(N_BATCH, 0xB0B);
    let mut o1 = vec![Gf128::zero(); N_BATCH];
    let mut o2 = vec![B127::zero(); N_BATCH];
    let (g, p) = time_pair_ns_per_op(
        reps,
        N_BATCH,
        || {
            batch_mul(black_box(&ga), black_box(&gb), black_box(&mut o1));
            o1[N_BATCH - 1]
        },
        || {
            for ((x, y), out) in ba.iter().zip(bb.iter()).zip(o2.iter_mut()) {
                *out = x.mul_pfold(*y);
            }
            o2[N_BATCH - 1]
        },
    );
    print_row("mul/batch b127-pfold", g, p);
}

fn main() {
    common::cli::EnvironmentCli::parse();
    let reps = common::reps(None, 5);
    common::enforce_known_env();
    bitz::observability::install().expect("install Perfetto subscriber");

    println!("BitZ field bench — GF(2^128) GHASH vs GF(2^127) b127, median of {reps} reps.");
    println!("(alternating reps per pattern: both fields share each thermal window)");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    println!("(target: aarch64 + neon — the NEON pipelines are active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    common::warn("scalar pipelines — build with RUSTFLAGS=\"-C target-cpu=native\"");

    println!(
        "\n{:<22} {:>14} {:>14} {:>9}",
        "pattern", "GF128 ns/op", "b127 ns/op", "speedup"
    );
    run_paired::<Gf128, B127>(reps);

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    run_b127_kara(reps);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    run_b127_pfold(reps);
}
