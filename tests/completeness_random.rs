//! Algebra-only randomized COMPLETENESS audit using historical configurations
//! (including small unaudited shapes) of the BitZ mod-q PCS pipeline
//! (`commit_rs_ligerito_rows` -> `prove_mle_eval_mod_q_ligerito` ->
//! `verify_mle_eval_mod_q_ligerito`).
//!
//! An honest prover must be able to prove every valid claim
//! `MLE[INT(D)](r) = y (mod q)` — the verifier must accept, and the accepted
//! claimed value must equal an INDEPENDENT naive reference
//! `y = sum_{b,c} INT(cell(b,c)) * eq(r_row, b) * eq(r_col, c) mod q`
//! implemented here from the eq-product definition with local 128-bit modular
//! arithmetic (the crate is used ONLY for commit/prove/verify/codec).
//!
//! Also runs cheap NEGATIVE spot-checks on fresh honest instances: wrong
//! claim (y+1), perturbed evaluation point, proof-byte flips, wrong prime.
//! Any acceptance there is a CRITICAL finding.
//!
//! Deterministic: the seed is printed; override with env AUDIT_SEED (u64) and
//! the bulk trial count with AUDIT_TRIALS.
//!
//! Run:
//!   RUSTFLAGS="-C target-cpu=native" \
//!     cargo test --release --test completeness_random -- --nocapture

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{
    IntEvalRsLigModQProof, commit_rs_flock_with, commit_rs_ligerito_rows,
    historical_sha_lig_configs, prove_mle_eval_mod_q_ligerito, verify_mle_eval_mod_q_ligerito,
};
use bitz::pcs::{IntegerMatrixLayout, mod_q_chunk_width, mod_q_num_chunks, smallest_generator};
use bitz::transcript::Blake3Transcript;

// ---------------------------------------------------------------------
// Independent modular arithmetic (q < 2^127; double-and-add modmul)
// ---------------------------------------------------------------------

fn mod_add(a: u128, b: u128, q: u128) -> u128 {
    // a, b < q <= 2^127 - 1, so a + b < 2^128: no overflow.
    let s = a + b;
    if s >= q { s - q } else { s }
}

/// `a*b mod q` by Russian-peasant double-and-add over the bits of the
/// smaller operand. Requires q < 2^127 (doubles stay below 2^128).
fn mod_mul(a: u128, b: u128, q: u128) -> u128 {
    let mut a = a % q;
    let mut b = b % q;
    if b > a {
        core::mem::swap(&mut a, &mut b);
    }
    let mut r = 0u128;
    while b != 0 {
        if b & 1 == 1 {
            r = mod_add(r, a, q);
        }
        b >>= 1;
        if b != 0 {
            a = mod_add(a, a, q);
        }
    }
    r
}

fn bits_of(q: u128) -> usize {
    (128 - q.leading_zeros()) as usize
}

// ---------------------------------------------------------------------
// The evaluation ring R = F_q as a const-generic wrapper (one type per
// audited prime; only Copy + PartialEq + From<u128> + Add + Mul are needed).
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Fq<const Q: u128>(u128);

impl<const Q: u128> From<u128> for Fq<Q> {
    fn from(v: u128) -> Self {
        Fq(v % Q)
    }
}
impl<const Q: u128> core::ops::Add for Fq<Q> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Fq(mod_add(self.0, o.0, Q))
    }
}
impl<const Q: u128> core::ops::Mul for Fq<Q> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Fq(mod_mul(self.0, o.0, Q))
    }
}

/// The audited primes, spanning the supported q range (2 .. 126 bits).
/// The PCS accepts q_bits in 1..=126; this also keeps the independent
/// double-and-add modular arithmetic below the u128 overflow boundary.
const PRIMES: [(u128, &str); 10] = [
    (3, "3"),
    (5, "5"),
    (17, "17"),
    (257, "257"),
    (65537, "65537"),
    (4294967291, "2^32-5"),
    ((1u128 << 61) - 1, "2^61-1"),
    ((1u128 << 80) - 65, "2^80-65"),
    ((1u128 << 100) - 15, "2^100-15"),
    ((1u128 << 126) - 137, "2^126-137"),
];

// ---------------------------------------------------------------------
// PRNG (SplitMix64) — self-contained, reproducible
// ---------------------------------------------------------------------

struct Rng {
    s: u64,
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { s: seed }
    }
    fn u64(&mut self) -> u64 {
        splitmix64(&mut self.s)
    }
    fn u128(&mut self) -> u128 {
        (u128::from(self.u64()) << 64) | u128::from(self.u64())
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u64() % n as u64) as usize
    }
    /// Uniform in [0, q) by rejection over bit_len(q)-bit draws.
    fn mod_q(&mut self, q: u128) -> u128 {
        let bits = bits_of(q);
        let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
        loop {
            let v = self.u128() & mask;
            if v < q {
                return v;
            }
        }
    }
}

// ---------------------------------------------------------------------
// eq tables + independent reference evaluator
// ---------------------------------------------------------------------

/// Little-endian eq table over F_q: `out[idx] = prod_l (bit_l(idx) ? r[l] :
/// 1 - r[l])`, coordinate `l` <-> bit `l` of the index.
fn eq_table(point: &[u128], q: u128) -> Vec<u128> {
    let mut acc = vec![1u128 % q];
    for &x in point {
        let xm = x % q;
        let one_minus = (1u128 % q + q - xm) % q;
        let mut next = Vec::with_capacity(acc.len() * 2);
        for &e in &acc {
            next.push(mod_mul(e, one_minus, q));
        }
        for &e in &acc {
            next.push(mod_mul(e, xm, q));
        }
        acc = next;
    }
    acc
}

/// eq(idx, point) straight from the product definition (spot-check oracle
/// for `eq_table`).
fn eq_direct(idx: usize, point: &[u128], q: u128) -> u128 {
    let mut acc = 1u128 % q;
    for (l, &x) in point.iter().enumerate() {
        let xm = x % q;
        let f = if (idx >> l) & 1 == 1 { xm } else { (1u128 % q + q - xm) % q };
        acc = mod_mul(acc, f, q);
    }
    acc
}

/// Data classes for the committed W-bit cell matrix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DataClass {
    Uniform,
    AllZero,
    AllOnes,
    Sparse,
    SingleBit,
}

impl DataClass {
    fn name(self) -> &'static str {
        match self {
            DataClass::Uniform => "unif",
            DataClass::AllZero => "zero",
            DataClass::AllOnes => "ones",
            DataClass::Sparse => "sparse",
            DataClass::SingleBit => "1bit",
        }
    }
}

/// Deterministic counter-based cell generator (no 2^n tensor is stored).
struct DataGen {
    class: DataClass,
    seed: u64,
    mask: u128,
    s: usize,
    single: (usize, usize, u32), // (b, c, j) of the single set bit
    replica: bool,               // the in-crate-test LCG instance
}

impl DataGen {
    fn h(&self, b: usize, c: usize) -> u64 {
        let mut x = self.seed ^ ((b as u64) << 32) ^ (c as u64).wrapping_mul(0x9E37_79B9);
        splitmix64(&mut x)
    }
    fn cell(&self, b: usize, c: usize) -> u128 {
        if self.replica {
            // The in-crate test / reference_measure instance:
            // cell = (flat_index * 0x9E37_79B9_7F4A_7C15) & mask.
            let idx = ((b << self.s) | c) as u128;
            return idx.wrapping_mul(0x9E37_79B9_7F4A_7C15) & self.mask;
        }
        match self.class {
            DataClass::Uniform => u128::from(self.h(b, c)) & self.mask,
            DataClass::AllZero => 0,
            DataClass::AllOnes => self.mask,
            DataClass::Sparse => {
                let h = self.h(b, c);
                if h & 127 == 0 { (u128::from(h) >> 7) & self.mask | 1 } else { 0 }
            }
            DataClass::SingleBit => {
                if b == self.single.0 && c == self.single.1 {
                    1u128 << self.single.2
                } else {
                    0
                }
            }
        }
    }
    fn is_all_zero(&self, rows: usize, cols: usize) -> bool {
        if self.replica {
            return false;
        }
        match self.class {
            DataClass::AllZero => true,
            DataClass::Sparse => (0..rows).all(|b| (0..cols).all(|c| self.cell(b, c) == 0)),
            _ => false,
        }
    }
}

/// Naive reference: `y = sum_c cw[c] * (sum_b rw[b] * INT(cell(b,c))) mod q`,
/// streaming over cells (never touching the crate's fold helpers).
fn reference_eval(
    rows: usize,
    cols: usize,
    dgen: &DataGen,
    rw: &[u128],
    cw: &[u128],
    q: u128,
) -> u128 {
    let mut y = 0u128;
    for c in 0..cols {
        let mut v = 0u128;
        for (b, &w) in rw.iter().enumerate().take(rows) {
            let cell = dgen.cell(b, c);
            if cell != 0 && w != 0 {
                v = mod_add(v, mod_mul(w, cell % q, q), q);
            }
        }
        y = mod_add(y, mod_mul(cw[c], v, q), q);
    }
    y
}

/// Per-column bit rows in the crate's committed layout: bit
/// `i = (b << log2 W) | j` of row `c` = bit `j` of cell `(b, c)`.
fn build_rows(t: usize, s: usize, w: usize, dgen: &DataGen) -> Vec<Vec<u64>> {
    let log_w = w.trailing_zeros() as usize;
    let row_len = (1usize << t) << log_w;
    let words = row_len.div_ceil(64);
    (0..1usize << s)
        .map(|c| {
            let mut wv = vec![0u64; words];
            for b in 0..1usize << t {
                let v = dgen.cell(b, c);
                for j in 0..w {
                    if (v >> j) & 1 == 1 {
                        let i = (b << log_w) | j;
                        wv[i >> 6] |= 1u64 << (i & 63);
                    }
                }
            }
            wv
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PointClass {
    Uniform,
    ZeroOneMax,
    Repeated,
}

impl PointClass {
    fn name(self) -> &'static str {
        match self {
            PointClass::Uniform => "unifpt",
            PointClass::ZeroOneMax => "01max",
            PointClass::Repeated => "reppt",
        }
    }
}

fn sample_point(rng: &mut Rng, len: usize, q: u128, class: PointClass) -> Vec<u128> {
    match class {
        PointClass::Uniform => (0..len).map(|_| rng.mod_q(q)).collect(),
        PointClass::ZeroOneMax => (0..len)
            .map(|_| match rng.below(3) {
                0 => 0,
                1 => 1 % q,
                _ => q - 1,
            })
            .collect(),
        PointClass::Repeated => {
            let x = rng.mod_q(q);
            vec![x; len]
        }
    }
}

// ---------------------------------------------------------------------
// Trial plumbing
// ---------------------------------------------------------------------

#[derive(Clone)]
struct TrialCfg {
    label: &'static str,
    t: usize,
    s: usize,
    w: usize,
    q_idx: usize,
    data: DataClass,
    point: PointClass,
    seed: u64,
    neg_point: bool,
    neg_bytes: bool,
    neg_prime: bool,
    replica: bool,
}

#[derive(Default)]
struct Report {
    lines: Vec<String>,
    /// Hard failures: an honest proof rejected, or a FALSE statement accepted
    /// (a completeness or soundness break). These abort the suite.
    failures: Vec<String>,
    /// Robustness findings that are NOT completeness/soundness breaks:
    /// encoding non-canonicality (a mutated byte accepted because it decodes
    /// to the identical valid proof) and decode/verify panics-instead-of-Err.
    /// Reported prominently but do not abort the suite.
    findings: Vec<String>,
    /// The CHARACTERIZED completeness gap: honest proofs rejected with
    /// RHatZero at multi-chunk (L>=2) openings whose top weight chunk is
    /// all-zero (boolean / small evaluation points). A real completeness
    /// limitation on structured points; reported, and counted separately.
    completeness_degeneracy: Vec<String>,
    degenerate_trials: u32,
    notes: Vec<String>,
    // (n, W, q_bits, data) -> (pass, fail)
    agg: BTreeMap<(usize, usize, usize, &'static str), (u32, u32)>,
    neg_claim: (u32, u32),          // attempted, rejected
    neg_point: (u32, u32, u32),     // attempted, rejected, skipped
    neg_flip: (u32, u32, u32, u32), // attempts, rejected@decode(clean), rejected@verify, panics(caught)
    neg_flip_noncanon: u32,         // byte flips accepted as the IDENTICAL proof (non-canonical)
    neg_flip_abort: u32,            // byte flips that ABORTED from_bytes (uncatchable alloc)
    neg_prime: (u32, u32, u32),     // attempted, rejected, skipped
    l_seen: BTreeMap<usize, u32>,   // chunk count L -> trials
}

impl Report {
    fn fail(&mut self, cfg: &TrialCfg, what: &str) {
        self.failures.push(format!(
            "FAIL [{} t={} s={} W={} q={} data={} point={} seed={:#018x}]: {}",
            cfg.label,
            cfg.t,
            cfg.s,
            cfg.w,
            PRIMES[cfg.q_idx].1,
            cfg.data.name(),
            cfg.point.name(),
            cfg.seed,
            what
        ));
    }
    fn finding(&mut self, cfg: &TrialCfg, what: &str) {
        self.findings.push(format!(
            "FINDING [{} t={} s={} W={} q={} seed={:#018x}]: {}",
            cfg.label,
            cfg.t,
            cfg.s,
            cfg.w,
            PRIMES[cfg.q_idx].1,
            cfg.seed,
            what
        ));
    }
}

/// The deterministic instance of one trial (rng draw order is part of the
/// reproducibility contract; the diagnostic tests rebuild instances through
/// this same helper).
struct Instance {
    rng: Rng,
    dgen: DataGen,
    r1: Vec<u128>,
    r2: Vec<u128>,
    rw: Vec<u128>,
    cw: Vec<u128>,
    y_rep: u128,
}

fn build_instance<const Q: u128>(cfg: &TrialCfg) -> Instance {
    let mut rng = Rng::new(cfg.seed);
    let mask = if cfg.w >= 128 { u128::MAX } else { (1u128 << cfg.w) - 1 };
    let rows_n = 1usize << cfg.t;
    let cols_n = 1usize << cfg.s;
    let dgen = DataGen {
        class: cfg.data,
        seed: rng.u64(),
        mask,
        s: cfg.s,
        single: (rng.below(rows_n), rng.below(cols_n), rng.below(cfg.w) as u32),
        replica: cfg.replica,
    };

    // Point + weights. The replica trial instead reuses the in-crate test's
    // ARBITRARY (non-eq) weight vectors — the API binds weight vectors, and
    // eq tables are the MLE special case.
    let (r1, r2, rw, cw): (Vec<u128>, Vec<u128>, Vec<u128>, Vec<u128>) = if cfg.replica {
        let rw: Vec<u128> = (0..rows_n)
            .map(|b| {
                (b as u128)
                    .wrapping_mul(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF1)
                    .wrapping_add(7)
                    % Q
            })
            .collect();
        let cw: Vec<u128> =
            (0..cols_n).map(|c| ((c as u128).wrapping_mul(5) & 7).wrapping_add(1) % Q).collect();
        (vec![], vec![], rw, cw)
    } else {
        let r1 = sample_point(&mut rng, cfg.t, Q, cfg.point);
        let r2 = sample_point(&mut rng, cfg.s, Q, cfg.point);
        let rw = eq_table(&r1, Q);
        let cw = eq_table(&r2, Q);
        // Spot-check the tables against the raw eq-product definition.
        for _ in 0..6 {
            let b = rng.below(rows_n);
            let c = rng.below(cols_n);
            assert_eq!(rw[b], eq_direct(b, &r1, Q), "eq_table(row) self-check");
            assert_eq!(cw[c], eq_direct(c, &r2, Q), "eq_table(col) self-check");
        }
        (r1, r2, rw, cw)
    };

    // Independent reference value.
    let y_rep = reference_eval(rows_n, cols_n, &dgen, &rw, &cw, Q);
    Instance { rng, dgen, r1, r2, rw, cw, y_rep }
}

/// One complete honest-instance trial (plus its scheduled negative checks).
fn run_trial<const Q: u128>(cfg: &TrialCfg, rep: &mut Report) {
    let t0 = Instant::now();
    let q_bits = bits_of(Q);
    let p = IntegerMatrixLayout { row_vars: cfg.t, col_vars: cfg.s, word_bits: cfg.w };
    let m_p = packed_vars(&p);
    let m = m_p + 7;
    let regime = if m >= 22 { "emb" } else { "adhoc" };
    let lch = mod_q_num_chunks(&p, q_bits);
    let c_w = mod_q_chunk_width(&p);
    *rep.l_seen.entry(lch).or_insert(0) += 1;

    let (pc, vc) = match historical_sha_lig_configs(m_p) {
        Ok(x) => x,
        Err(e) => {
            rep.fail(cfg, &format!("historical_sha_lig_configs(m_p={m_p}) errored on a legal shape: {e}"));
            return;
        }
    };

    let rows_n = 1usize << cfg.t;
    let cols_n = 1usize << cfg.s;
    let Instance { mut rng, dgen, r1, r2, rw, cw, y_rep } = build_instance::<Q>(cfg);
    let y = Fq::<Q>(y_rep);
    let cw_fq: Vec<Fq<Q>> = cw.iter().map(|&x| Fq::<Q>(x)).collect();

    // Commit from packed rows (the memory-honest entry point).
    let rows = build_rows(cfg.t, cfg.s, cfg.w, &dgen);
    let hint = commit_rs_ligerito_rows(&p, rows, &pc);
    let alpha = smallest_generator();

    // Prove.
    let mut pt = Blake3Transcript::new();
    let proof = prove_mle_eval_mod_q_ligerito(&mut pt, &hint, &p, &rw, q_bits, alpha, &pc);
    if proof.us.len() != lch {
        rep.fail(cfg, &format!("prover chunk count {} != expected L={lch}", proof.us.len()));
    }

    // Fresh-transcript verify wrapper (flock config/commitment types stay
    // inferred — they are not nameable from an integration test).
    let verify_y = |pr: &IntEvalRsLigModQProof,
                    rw_in: &[u128],
                    cw_in: &[Fq<Q>],
                    claimed: Fq<Q>|
     -> Result<(), String> {
        let mut vt = Blake3Transcript::new();
        verify_mle_eval_mod_q_ligerito(
            &mut vt,
            &hint.commitment,
            pr,
            &p,
            rw_in,
            cw_in,
            alpha,
            claimed,
            q_bits,
            &vc,
        )
        .map_err(|e| format!("{e:?}"))
    };

    // Honest verify at the reference value.
    let mut ok = true;
    if let Err(e) = verify_y(&proof, &rw, &cw_fq, y) {
        // Characterized COMPLETENESS DEGENERACY: a multi-chunk (L>=2) opening
        // whose row eq-weights leave the TOP weight chunk all-zero (boolean /
        // small evaluation points) makes the verifier's R̂ denominator zero,
        // so it rejects the honest proof with RHatZero. Reported as a
        // completeness finding, not counted as pass; other honest rejections
        // remain hard failures.
        if e.contains("RHatZero") && lch >= 2 {
            rep.degenerate_trials += 1;
            rep.completeness_degeneracy.push(format!(
                "DEGENERACY [{} t={} s={} W={} q={} (q_bits={q_bits}, L={lch}, c_w={c_w}) data={} point={} seed={:#018x}]: \
                 honest proof REJECTED with RHatZero — the top weight chunk is all-zero at this \
                 (boolean/small-coordinate) point, so R̂ = 0. Valid MLE opening; completeness gap, not soundness.",
                cfg.label, cfg.t, cfg.s, cfg.w, PRIMES[cfg.q_idx].1,
                cfg.data.name(), cfg.point.name(), cfg.seed
            ));
            rep.lines.push(format!(
                "DGEN n={:2} t={:2} s={:2} W={:2} q={:>9} L={lch} data={:6} point={:6} {} (RHatZero)",
                cfg.t + cfg.s, cfg.t, cfg.s, cfg.w, PRIMES[cfg.q_idx].1,
                cfg.data.name(), cfg.point.name(), cfg.label
            ));
            return;
        }
        rep.fail(cfg, &format!("verifier REJECTED an honest proof: {e}"));
        ok = false;
    }

    // Negative (i): wrong claim y+1 — always a false claim.
    rep.neg_claim.0 += 1;
    match verify_y(&proof, &rw, &cw_fq, y + Fq::<Q>(1 % Q)) {
        Err(_) => rep.neg_claim.1 += 1,
        Ok(()) => {
            rep.fail(cfg, "CRITICAL: verifier ACCEPTED the wrong claim y+1");
            ok = false;
        }
    }

    // Codec roundtrip: to_bytes -> from_bytes -> verify again.
    let bytes = proof.to_bytes();
    match IntEvalRsLigModQProof::from_bytes(&bytes) {
        Err(e) => {
            rep.fail(cfg, &format!("from_bytes failed on an honest proof: {e:?}"));
            ok = false;
        }
        Ok(decoded) => {
            if decoded.to_bytes() != bytes {
                rep.notes.push(format!(
                    "NOTE [{} seed={:#018x}]: codec re-encode differs from original bytes",
                    cfg.label, cfg.seed
                ));
            }
            if let Err(e) = verify_y(&decoded, &rw, &cw_fq, y) {
                rep.fail(cfg, &format!("decoded proof REJECTED: {e}"));
                ok = false;
            }
        }
    }

    let data_all_zero = dgen.is_all_zero(rows_n, cols_n);

    // Negative (ii): one r coordinate changed (claim must become false).
    if cfg.neg_point && !cfg.replica && !data_all_zero && cfg.t + cfg.s > 0 {
        rep.neg_point.0 += 1;
        let mut done = false;
        for _ in 0..8 {
            let k = rng.below(cfg.t + cfg.s);
            let delta = 1 + rng.u128() % (Q - 1);
            let (mut r1p, mut r2p) = (r1.clone(), r2.clone());
            if k < cfg.t {
                r1p[k] = mod_add(r1p[k], delta, Q);
            } else {
                r2p[k - cfg.t] = mod_add(r2p[k - cfg.t], delta, Q);
            }
            let rwp = eq_table(&r1p, Q);
            let cwp = eq_table(&r2p, Q);
            let y_p = reference_eval(rows_n, cols_n, &dgen, &rwp, &cwp, Q);
            if y_p == y_rep {
                continue; // claim would stay true at the perturbed point
            }
            let cwp_fq: Vec<Fq<Q>> = cwp.iter().map(|&x| Fq::<Q>(x)).collect();
            match verify_y(&proof, &rwp, &cwp_fq, y) {
                Err(_) => rep.neg_point.1 += 1,
                Ok(()) => {
                    rep.fail(cfg, "CRITICAL: verifier ACCEPTED a perturbed-point false claim");
                    ok = false;
                }
            }
            done = true;
            break;
        }
        if !done {
            rep.neg_point.0 -= 1;
            rep.neg_point.2 += 1;
        }
    }

    // Negative (iii): byte flips of the serialized proof. Each decode runs
    // in a forked child (fork_decode) so an UNCATCHABLE allocation abort from
    // a tampered length prefix cannot take the whole suite down; the parent
    // classifies from the child's exit status and re-runs the (safe)
    // decode+verify in-process only for inputs the child decoded cleanly.
    if cfg.neg_bytes {
        for _ in 0..10 {
            rep.neg_flip.0 += 1;
            let off = rng.below(bytes.len());
            let mut x = (rng.u64() & 0xFF) as u8;
            if x == 0 {
                x = 0x5A;
            }
            let mut mutated = bytes.clone();
            mutated[off] ^= x;
            match fork_decode(&mutated) {
                DecodeOutcome::Aborted => {
                    rep.neg_flip_abort += 1;
                    rep.finding(
                        cfg,
                        &format!(
                            "byte flip at offset {off} (xor {x:#04x}) made from_bytes ABORT the process \
                             (uncatchable alloc failure — unbounded Vec::with_capacity on a tampered length \
                             prefix). Denial-of-service on malformed input, not a soundness break."
                        ),
                    );
                }
                DecodeOutcome::DecodeErr => rep.neg_flip.1 += 1,
                DecodeOutcome::DecodePanic => {
                    rep.neg_flip.3 += 1;
                    rep.notes.push(format!(
                        "NOTE [{} seed={:#018x}]: from_bytes PANICKED (caught) on byte flip at offset {off} (xor {x:#04x})",
                        cfg.label, cfg.seed
                    ));
                }
                DecodeOutcome::DecodeOk => {
                    let dproof =
                        IntEvalRsLigModQProof::from_bytes(&mutated).expect("re-decode ok");
                    let vres =
                        catch_unwind(AssertUnwindSafe(|| verify_y(&dproof, &rw, &cw_fq, y)));
                    match vres {
                        Err(_) => {
                            rep.neg_flip.3 += 1;
                            rep.notes.push(format!(
                                "NOTE [{} seed={:#018x}]: verify PANICKED (caught) on byte-flipped proof at offset {off} (xor {x:#04x})",
                                cfg.label, cfg.seed
                            ));
                        }
                        Ok(Err(_)) => rep.neg_flip.2 += 1,
                        Ok(Ok(())) => {
                            if dproof.to_bytes() == bytes {
                                // The mutated byte is an unread hole: decode
                                // yields the IDENTICAL proof, which verifies.
                                // Encoding non-canonicality / malleability —
                                // NOT a forged statement. Reported, not fatal.
                                rep.neg_flip_noncanon += 1;
                                rep.finding(
                                    cfg,
                                    &format!(
                                        "byte flip at offset {off} (xor {x:#04x}) accepted — decodes to the \
                                         IDENTICAL proof (re-encode == original). Codec non-canonicality: \
                                         from_bytes tolerates a byte it does not read, contradicting the \
                                         codec's canonical/tamper-rejecting claim. Encoding malleability, \
                                         NOT statement forgery."
                                    ),
                                );
                            } else {
                                // A DIFFERENT proof verified: a genuine
                                // soundness break.
                                rep.fail(
                                    cfg,
                                    &format!(
                                        "CRITICAL (soundness): byte flip at offset {off} (xor {x:#04x}) \
                                         decodes to a DIFFERENT proof that VERIFIES"
                                    ),
                                );
                                ok = false;
                            }
                        }
                    }
                }
            }
        }
    }

    // Negative (iv): verify under a different prime q'. The claim must be
    // genuinely false there: checked against a local recombination of the
    // transmitted chunk folds (skip on coincidence / all-zero data). If the
    // chunk counts differ the verifier's shape check must already reject.
    if cfg.neg_prime && !data_all_zero {
        macro_rules! wrong_prime_arm {
            ($q2:expr) => {{
                const Q2C: u128 = $q2;
                let q2_bits = bits_of(Q2C);
                let lch2 = mod_q_num_chunks(&p, q2_bits);
                let claimed2 = y_rep % Q2C;
                let mut skip = false;
                if lch2 == lch {
                    let mut recomb = 0u128;
                    for c in 0..cols_n {
                        let mut v_c = 0u128;
                        let mut mult = 1u128 % Q2C;
                        for u_l in proof.us.iter() {
                            let u = u_l.get(c).copied().unwrap_or(0);
                            v_c = mod_add(v_c, mod_mul(mult, u % Q2C, Q2C), Q2C);
                            mult = mod_mul(mult, (1u128 << c_w) % Q2C, Q2C);
                        }
                        recomb = mod_add(recomb, mod_mul(cw[c] % Q2C, v_c, Q2C), Q2C);
                    }
                    if recomb == claimed2 {
                        skip = true;
                    }
                }
                if skip {
                    rep.neg_prime.2 += 1;
                } else {
                    rep.neg_prime.0 += 1;
                    let cw2: Vec<Fq<Q2C>> = cw.iter().map(|&x| Fq::<Q2C>::from(x)).collect();
                    let mut vt = Blake3Transcript::new();
                    match verify_mle_eval_mod_q_ligerito(
                        &mut vt,
                        &hint.commitment,
                        &proof,
                        &p,
                        &rw,
                        &cw2,
                        alpha,
                        Fq::<Q2C>(claimed2),
                        q2_bits,
                        &vc,
                    ) {
                        Err(_) => rep.neg_prime.1 += 1,
                        Ok(()) => {
                            rep.fail(
                                cfg,
                                &format!(
                                    "CRITICAL: verifier ACCEPTED under wrong prime q'={} (L'={lch2})",
                                    Q2C
                                ),
                            );
                            ok = false;
                        }
                    }
                }
            }};
        }
        let alt = if Q == 17 {
            1
        } else if Q == (1u128 << 61) - 1 {
            2
        } else if Q == (1u128 << 100) - 15 {
            0
        } else {
            (cfg.seed % 3) as usize
        };
        match alt {
            0 => wrong_prime_arm!(17u128),
            1 => wrong_prime_arm!((1u128 << 61) - 1),
            _ => wrong_prime_arm!((1u128 << 100) - 15),
        }
    }

    let e = rep.agg.entry((cfg.t + cfg.s, cfg.w, q_bits, cfg.data.name())).or_insert((0, 0));
    if ok {
        e.0 += 1;
    } else {
        e.1 += 1;
    }
    rep.lines.push(format!(
        "{} n={:2} t={:2} s={:2} W={:2} q={:>9} L={} cfg={:5} data={:6} point={:6} {} ({} ms)",
        if ok { "ok  " } else { "FAIL" },
        cfg.t + cfg.s,
        cfg.t,
        cfg.s,
        cfg.w,
        PRIMES[cfg.q_idx].1,
        lch,
        regime,
        cfg.data.name(),
        if cfg.replica { "n/a" } else { cfg.point.name() },
        cfg.label,
        t0.elapsed().as_millis()
    ));
}

// ---------------------------------------------------------------------
// Trial-plan construction
// ---------------------------------------------------------------------

const DATA_CLASSES: [DataClass; 5] = [
    DataClass::Uniform,
    DataClass::AllZero,
    DataClass::AllOnes,
    DataClass::Sparse,
    DataClass::SingleBit,
];
const POINT_CLASSES: [PointClass; 3] = [
    PointClass::Uniform,
    PointClass::ZeroOneMax,
    PointClass::Repeated,
];

/// Legal-shape sampler: W a power of two, t >= max(1, 7 - log2 W)
/// (ligerito_flock.rs:284), s >= 1, t + W <= 126 (pcs.rs:1005), and
/// m_p = t + log2 W + s - 7 >= 8 (flock default_config L0 feasibility for
/// the ad-hoc profile; embedded configs exist for m in 22..=35).
fn sample_shape(rng: &mut Rng, w: usize, n_max: usize) -> (usize, usize) {
    let log_w = w.trailing_zeros() as usize;
    let n_min = 15usize.saturating_sub(log_w).max(2);
    let n = n_min + rng.below(n_max.saturating_sub(n_min) + 1);
    let t_min = 7usize.saturating_sub(log_w).max(1);
    let t_max = n - 1; // s >= 1
    let t = t_min + rng.below(t_max - t_min + 1);
    (t, n - t)
}

fn dispatch_trial(cfg: &TrialCfg, rep: &mut Report) {
    match cfg.q_idx {
        0 => run_trial::<{ PRIMES[0].0 }>(cfg, rep),
        1 => run_trial::<{ PRIMES[1].0 }>(cfg, rep),
        2 => run_trial::<{ PRIMES[2].0 }>(cfg, rep),
        3 => run_trial::<{ PRIMES[3].0 }>(cfg, rep),
        4 => run_trial::<{ PRIMES[4].0 }>(cfg, rep),
        5 => run_trial::<{ PRIMES[5].0 }>(cfg, rep),
        6 => run_trial::<{ PRIMES[6].0 }>(cfg, rep),
        7 => run_trial::<{ PRIMES[7].0 }>(cfg, rep),
        8 => run_trial::<{ PRIMES[8].0 }>(cfg, rep),
        _ => run_trial::<{ PRIMES[9].0 }>(cfg, rep),
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn panic_msg(e: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".into()
    }
}

fn merge(into: &mut Report, from: Report) {
    into.lines.extend(from.lines);
    into.failures.extend(from.failures);
    into.findings.extend(from.findings);
    into.completeness_degeneracy.extend(from.completeness_degeneracy);
    into.degenerate_trials += from.degenerate_trials;
    into.notes.extend(from.notes);
    into.neg_flip_noncanon += from.neg_flip_noncanon;
    into.neg_flip_abort += from.neg_flip_abort;
    for (k, (p, f)) in from.agg {
        let e = into.agg.entry(k).or_insert((0, 0));
        e.0 += p;
        e.1 += f;
    }
    into.neg_claim.0 += from.neg_claim.0;
    into.neg_claim.1 += from.neg_claim.1;
    into.neg_point.0 += from.neg_point.0;
    into.neg_point.1 += from.neg_point.1;
    into.neg_point.2 += from.neg_point.2;
    into.neg_flip.0 += from.neg_flip.0;
    into.neg_flip.1 += from.neg_flip.1;
    into.neg_flip.2 += from.neg_flip.2;
    into.neg_flip.3 += from.neg_flip.3;
    into.neg_prime.0 += from.neg_prime.0;
    into.neg_prime.1 += from.neg_prime.1;
    into.neg_prime.2 += from.neg_prime.2;
    for (k, v) in from.l_seen {
        *into.l_seen.entry(k).or_insert(0) += v;
    }
}

// ---------------------------------------------------------------------
// The audit tests
// ---------------------------------------------------------------------

/// Replicate the in-crate `mle_eval_mod_q_ligerito_roundtrips` instance
/// (t=10, s=5, W=1, q = 2^100 - 15, its LCG data and arbitrary weights) with
/// THIS file's packer + reference evaluator, and additionally pin the
/// packed-rows commit against the crate's own u128-tensor commit path —
/// validating the harness's bit layout and evaluator against the crate.
#[test]
fn replica_of_in_crate_instance_validates_harness() {
    const Q100: u128 = (1u128 << 100) - 15;
    let cfg = TrialCfg {
        label: "replica",
        t: 10,
        s: 5,
        w: 1,
        q_idx: 8, // 2^100 - 15
        data: DataClass::Uniform,
        point: PointClass::Uniform,
        seed: 0xF2A0_0000_0000_0001,
        neg_point: false,
        neg_bytes: true,
        neg_prime: true,
        replica: true,
    };
    let mut rep = Report::default();
    run_trial::<Q100>(&cfg, &mut rep);
    for l in &rep.lines {
        println!("{l}");
    }
    for n in &rep.notes {
        println!("{n}");
    }
    assert!(rep.failures.is_empty(), "replica trial failed:\n{}", rep.failures.join("\n"));

    // Packed-rows commit == u128-tensor commit (same root) on this instance.
    let p = IntegerMatrixLayout { row_vars: 10, col_vars: 5, word_bits: 1 };
    let (pc, _vc) = historical_sha_lig_configs(packed_vars(&p)).expect("cfg");
    let dgen = DataGen {
        class: DataClass::Uniform,
        seed: 0,
        mask: 1,
        s: 5,
        single: (0, 0, 0),
        replica: true,
    };
    let rows = build_rows(10, 5, 1, &dgen);
    let hint_rows = commit_rs_ligerito_rows(&p, rows, &pc);
    let data: Vec<u128> =
        (0..p.cells()).map(|i| (i as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15) & 1).collect();
    let hint_tensor = commit_rs_flock_with(&p, &data, pc.log_inv_rates[0], pc.initial_k);
    assert_eq!(
        format!("{:?}", hint_rows.root()),
        format!("{:?}", hint_tensor.root()),
        "packed-rows commit and tensor commit disagree"
    );
    println!("replica: packer root check ok");
}

/// The main randomized completeness sweep.
#[test]
fn completeness_random_audit() {
    let seed = env_u64("AUDIT_SEED", 0xF2AC_0DE5_0000_0001);
    let bulk = env_u64("AUDIT_TRIALS", 150) as usize;
    println!("completeness audit seed = {seed:#018x}, bulk trials = {bulk}");
    let bitz_env: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("BITZ_"))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    println!(
        "BITZ_* env flags present: {}",
        if bitz_env.is_empty() { "none".into() } else { bitz_env.join(", ") }
    );

    let mut rng = Rng::new(seed);
    let mut plan: Vec<TrialCfg> = Vec::new();

    // Handcrafted edge shapes (domain corners + multi-chunk L coverage).
    let edges: [(usize, usize, usize, usize); 10] = [
        (12, 3, 64, 9), // c_w=51, q=2^126-137 -> L=3
        (10, 5, 32, 8), // c_w=85, q=2^100-15 -> L=2
        (14, 1, 1, 9),  // s=1 edge; c_w=112, L=2
        (7, 8, 1, 0),   // min t at W=1, q=3
        (2, 13, 32, 2), // min t at W=32
        (1, 14, 64, 3), // min t at W=64
        (16, 2, 1, 4),  // tall/thin
        (8, 10, 1, 1),  // wide, q=5
        (6, 9, 2, 6),   // W=2
        (3, 12, 16, 7), // W=16
    ];
    for (i, &(t, s, w, q_idx)) in edges.iter().enumerate() {
        plan.push(TrialCfg {
            label: "edge",
            t,
            s,
            w,
            q_idx,
            data: DATA_CLASSES[i % 5],
            point: POINT_CLASSES[i % 3],
            seed: rng.u64(),
            neg_point: true,
            neg_bytes: i % 4 == 0,
            neg_prime: true,
            replica: false,
        });
    }

    // Bulk random small-n trials (n <= 18).
    let widths = [1usize, 1, 1, 8, 8, 32, 32, 64, 4, 16];
    for i in 0..bulk {
        let w = widths[rng.below(widths.len())];
        let (t, s) = sample_shape(&mut rng, w, 18);
        plan.push(TrialCfg {
            label: "bulk",
            t,
            s,
            w,
            q_idx: i % PRIMES.len(),
            data: DATA_CLASSES[(i / 2) % 5],
            point: POINT_CLASSES[(i / 3) % 3],
            seed: rng.u64(),
            neg_point: i % 3 == 0,
            neg_bytes: i % 6 == 0,
            neg_prime: i % 5 == 0,
            replica: false,
        });
    }

    // n = 20 (W=1 ad-hoc m=20; W=4 embedded m=22).
    let n20: [(usize, usize, usize, usize); 8] = [
        (7, 13, 1, 0),
        (10, 10, 1, 6),
        (13, 7, 1, 8),
        (19, 1, 1, 9),
        (5, 15, 4, 2),
        (8, 12, 4, 5),
        (12, 8, 4, 7),
        (16, 4, 4, 9),
    ];
    for (i, &(t, s, w, q_idx)) in n20.iter().enumerate() {
        plan.push(TrialCfg {
            label: "n20",
            t,
            s,
            w,
            q_idx,
            data: DATA_CLASSES[i % 5],
            point: POINT_CLASSES[i % 3],
            seed: rng.u64(),
            neg_point: i % 2 == 0,
            neg_bytes: i == 3,
            neg_prime: i % 3 == 0,
            replica: false,
        });
    }

    // n = 22, W=1: m = 22 — the embedded config regime boundary.
    let n22: [(usize, usize, usize, usize, DataClass); 4] = [
        (7, 15, 1, 8, DataClass::Uniform),
        (11, 11, 1, 9, DataClass::Uniform),
        (15, 7, 1, 3, DataClass::Sparse),
        (21, 1, 1, 6, DataClass::AllZero),
    ];
    for (i, &(t, s, w, q_idx, data)) in n22.iter().enumerate() {
        plan.push(TrialCfg {
            label: "n22",
            t,
            s,
            w,
            q_idx,
            data,
            point: POINT_CLASSES[i % 3],
            seed: rng.u64(),
            neg_point: i == 1,
            neg_bytes: i == 0,
            neg_prime: i == 2,
            replica: false,
        });
    }

    let total = plan.len();
    let mut rep = Report::default();
    let t_start = Instant::now();
    for (i, cfg) in plan.iter().enumerate() {
        let res = catch_unwind(AssertUnwindSafe(|| {
            let mut local = Report::default();
            dispatch_trial(cfg, &mut local);
            local
        }));
        match res {
            Ok(local) => merge(&mut rep, local),
            Err(e) => {
                let msg = panic_msg(&e);
                rep.fail(cfg, &format!("PANIC during trial: {msg}"));
                rep.lines.push(format!(
                    "PANIC n={:2} t={:2} s={:2} W={:2} q={:>9} data={:6} {} seed={:#018x}",
                    cfg.t + cfg.s,
                    cfg.t,
                    cfg.s,
                    cfg.w,
                    PRIMES[cfg.q_idx].1,
                    cfg.data.name(),
                    cfg.label,
                    cfg.seed
                ));
                let e = rep
                    .agg
                    .entry((
                        cfg.t + cfg.s,
                        cfg.w,
                        bits_of(PRIMES[cfg.q_idx].0),
                        cfg.data.name(),
                    ))
                    .or_insert((0, 0));
                e.1 += 1;
            }
        }
        if (i + 1) % 25 == 0 {
            println!(
                "... {}/{} trials done ({:.1} s elapsed)",
                i + 1,
                total,
                t_start.elapsed().as_secs_f64()
            );
        }
    }

    println!("\n================ per-trial log ================");
    for l in &rep.lines {
        println!("{l}");
    }
    println!(
        "\n================ aggregate matrix (n, W, q_bits, data) -> pass/fail ================"
    );
    for ((n, w, qb, data), (pass, fail)) in &rep.agg {
        println!("n={n:2} W={w:2} q_bits={qb:3} data={data:6} : {pass:3} pass {fail} fail");
    }
    println!("\nchunk-count coverage (L -> trials): {:?}", rep.l_seen);
    let flip_accept_total = rep.neg_flip_noncanon; // non-canonical (identical proof); soundness accepts are in failures
    println!(
        "negative checks: wrong-claim {}/{} rejected; wrong-point {} attempted / {} rejected / {} skipped; \
         byte-flips {} attempted / {} rejected@decode / {} rejected@verify / {} panicked(caught) / {} accepted-noncanonical; \
         wrong-prime {} attempted / {} rejected / {} skipped",
        rep.neg_claim.1,
        rep.neg_claim.0,
        rep.neg_point.0,
        rep.neg_point.1,
        rep.neg_point.2,
        rep.neg_flip.0,
        rep.neg_flip.1,
        rep.neg_flip.2,
        rep.neg_flip.3,
        flip_accept_total,
        rep.neg_prime.0,
        rep.neg_prime.1,
        rep.neg_prime.2,
    );

    // Characterized COMPLETENESS gap (honest rejections on structured points).
    println!(
        "\n================ completeness gap (characterized) ================"
    );
    if rep.completeness_degeneracy.is_empty() {
        println!("  (none hit in this run)");
    } else {
        println!(
            "  {} trial(s) hit the RHatZero multi-chunk degeneracy: an HONEST proof is REJECTED when \
             L>=2 AND the row eq-weights leave the top weight chunk all-zero (boolean / small-coordinate \
             evaluation points). See test `rhatzero_boolean_point_completeness_gap` for the minimal \
             deterministic repro. Valid MLE openings, so this is a real completeness limitation on \
             structured points (NOT a soundness problem).",
            rep.degenerate_trials
        );
        for d in &rep.completeness_degeneracy {
            println!("  {d}");
        }
    }

    // Robustness findings (NOT completeness/soundness breaks): reported, but
    // the suite does not fail on them.
    println!(
        "\n================ robustness findings (non-soundness) ================\n\
         (These do NOT break completeness — an honest prover still proves every valid claim — \
         nor soundness — no FALSE statement was ever accepted. They are input-hardening gaps in \
         the proof CODEC / verifier on adversarially MALFORMED bytes.)"
    );
    if rep.findings.is_empty() && rep.neg_flip.3 == 0 && rep.neg_flip_abort == 0 {
        println!("  (none)");
    } else {
        if rep.neg_flip_abort > 0 {
            println!(
                "  - {} of {} byte-flip mutations made from_bytes ABORT the process (uncatchable alloc \
                 failure): unbounded `Vec::with_capacity(lch/n_layers/n_sv)` in \
                 IntEvalRsLigModQProof::from_bytes on a tampered length prefix (only n_u is bounded by \
                 remaining()). Denial-of-service on malformed input, not a soundness break.",
                rep.neg_flip_abort, rep.neg_flip.0
            );
        }
        if rep.neg_flip.3 > 0 {
            println!(
                "  - {} of {} byte-flip mutations made from_bytes OR verify PANIC (caught here) \
                 instead of returning an error — panic-instead-of-Err hardening gap (safe rejection).",
                rep.neg_flip.3, rep.neg_flip.0
            );
        }
        if rep.neg_flip_noncanon > 0 {
            println!(
                "  - {} byte-flip mutations were ACCEPTED because they decode to the IDENTICAL valid \
                 proof (unread bytes / non-canonical encoding), despite proof_codec.rs documenting a \
                 canonical, tamper-rejecting codec. Encoding malleability, not statement forgery.",
                rep.neg_flip_noncanon
            );
        }
        for f in &rep.findings {
            println!("  {f}");
        }
    }
    if !rep.notes.is_empty() {
        println!("\n---- notes ----");
        for n in &rep.notes {
            println!("{n}");
        }
    }
    println!("total wall time: {:.1} s over {total} trials", t_start.elapsed().as_secs_f64());

    // Suite verdict: COMPLETENESS + SOUNDNESS only. An honest proof rejected,
    // or a FALSE statement accepted, is a hard failure. The robustness
    // findings above are reported but do not fail the audit.
    assert_eq!(rep.neg_claim.0, rep.neg_claim.1, "some wrong-claim checks did not reject");
    assert!(
        rep.failures.is_empty(),
        "completeness/soundness failures ({}):\n{}",
        rep.failures.len(),
        rep.failures.join("\n")
    );
}

/// Empirical probes of the legal-domain boundary (the asserts the audit's
/// sampler steers around).
#[test]
fn domain_boundary_probes() {
    assert!(field::prime::is_probable_prime_public(
        &field::Uint::<2>::from(PRIMES[9].0),
    ));
    let p = IntegerMatrixLayout { row_vars: 7, col_vars: 8, word_bits: 1 };
    let weights = vec![1u128; p.rows()];
    let (pc, _) = historical_sha_lig_configs(packed_vars(&p)).expect("boundary configuration");
    let hint = commit_rs_ligerito_rows(&p, vec![vec![0; p.rows() / 64]; p.cols()], &pc);
    for q_bits in [126, 127] {
        let result = catch_unwind(AssertUnwindSafe(|| {
            prove_mle_eval_mod_q_ligerito(
                &mut Blake3Transcript::new(), &hint, &p, &weights, q_bits,
                smallest_generator(), &pc,
            )
        }));
        assert_eq!(result.is_ok(), q_bits == 126, "q_bits boundary: {q_bits}");
    }

    // (1) flock ad-hoc config floor: m_p >= 8 (L0 block 2^(m_p-2+2) at rate
    //     1/4 must cover the 148 UDR queries).
    assert!(historical_sha_lig_configs(7).is_err(), "m_p=7 should be rejected by the ad-hoc config");
    assert!(historical_sha_lig_configs(8).is_ok(), "m_p=8 should be accepted");
    println!("probe: historical_sha_lig_configs floor at m_p=8 confirmed (m = t+log2W+s >= 15)");

    // (2) t + log2 W < 7 must panic at the packing assert
    //     (ligerito_flock.rs:284 'packing needs t + log2(W) >= 7').
    let r = catch_unwind(|| {
        let p = IntegerMatrixLayout { row_vars: 6, col_vars: 9, word_bits: 1 };
        let (pc, _vc) = historical_sha_lig_configs(packed_vars(&p)).expect("cfg");
        let dgen = DataGen {
            class: DataClass::AllZero,
            seed: 0,
            mask: 1,
            s: 9,
            single: (0, 0, 0),
            replica: false,
        };
        let rows = build_rows(6, 9, 1, &dgen);
        commit_rs_ligerito_rows(&p, rows, &pc)
    });
    match r {
        Err(e) => println!("probe: t+log2W=6 commit panicked as expected: {}", panic_msg(&e)),
        Ok(_) => panic!("t + log2 W = 6 commit unexpectedly succeeded"),
    }

    // (3) t + W > 126 must panic in mod_q_chunk_width (pcs.rs:1005).
    let r = catch_unwind(|| {
        mod_q_chunk_width(&IntegerMatrixLayout {
            row_vars: 100,
            col_vars: 1,
            word_bits: 32,
        })
    });
    match r {
        Err(e) => println!("probe: t+W=132 chunk width panicked as expected: {}", panic_msg(&e)),
        Ok(v) => panic!("t+W=132 mod_q_chunk_width unexpectedly returned {v}"),
    }

    // (4) s = 0 (single column): probe the full pipeline and REPORT the
    //     outcome; a panic here documents the boundary, it is not a
    //     completeness failure (the shape is outside the audited domain).
    let r = catch_unwind(|| {
        const Q: u128 = (1u128 << 61) - 1;
        let cfg = TrialCfg {
            label: "s0probe",
            t: 15,
            s: 0,
            w: 1,
            q_idx: 6,
            data: DataClass::Uniform,
            point: PointClass::Uniform,
            seed: 0xD_00D5,
            neg_point: false,
            neg_bytes: false,
            neg_prime: false,
            replica: false,
        };
        let mut rep = Report::default();
        run_trial::<Q>(&cfg, &mut rep);
        rep
    });
    match r {
        Err(e) => println!("probe: s=0 pipeline PANICS: {}", panic_msg(&e)),
        Ok(rep) => {
            if rep.failures.is_empty() {
                println!("probe: s=0 pipeline works and verifies the reference value");
            } else {
                println!("probe: s=0 pipeline rejects/fails: {}", rep.failures.join(" | "));
            }
        }
    }

    // (5) Non-power-of-two W (W=3) probe further below.
    domain_probe_w3();
}

fn domain_probe_w3() {
    // Non-power-of-two W: the layout math (log2/mask) silently
    // assumes a power of two; document the observed behavior.
    let r = catch_unwind(|| {
        const Q: u128 = (1u128 << 61) - 1;
        let cfg = TrialCfg {
            label: "w3probe",
            t: 8,
            s: 8,
            w: 3,
            q_idx: 6,
            data: DataClass::Uniform,
            point: PointClass::Uniform,
            seed: 0xBAD_3,
            neg_point: false,
            neg_bytes: false,
            neg_prime: false,
            replica: false,
        };
        let mut rep = Report::default();
        run_trial::<Q>(&cfg, &mut rep);
        rep
    });
    match r {
        Err(e) => println!("probe: W=3 pipeline PANICS: {}", panic_msg(&e)),
        Ok(rep) => {
            if rep.failures.is_empty() {
                println!(
                    "probe: W=3 UNEXPECTEDLY verifies (would mean the layout tolerates non-power-of-two W)"
                );
            } else {
                println!(
                    "probe: W=3 honest instance FAILS as expected (non-power-of-two W is illegal-by-layout): {}",
                    rep.failures[0]
                );
            }
        }
    }
}

/// Deep-dive on the byte-flip findings of the main audit: classify EVERY
/// single-byte corruption of one proof's serialized stream — cleanly
/// rejected (at decode or at verify), silently IGNORED (decodes to the
/// identical proof: an encoding-malleability / canonicality gap), accepted
/// as a DIFFERENT proof (would be soundness-relevant), or PANICKING (with
/// the panic location captured via a scoped hook).
///
/// The instance is the exact trial the main audit flagged
/// (seed 0xe75d8278bfcda727, t=13 s=2 W=1 q=3 data=unif point=reppt).
///
/// Ignored by default; run with:
///   cargo test --release --test completeness_random -- --ignored --nocapture
#[test]
#[ignore = "diagnostic deep-dive; run explicitly with -- --ignored"]
fn diagnose_byte_flip_findings() {
    const Q: u128 = 3;
    let cfg = TrialCfg {
        label: "diag",
        t: 13,
        s: 2,
        w: 1,
        q_idx: 0,
        data: DataClass::Uniform,
        point: PointClass::Repeated,
        seed: 0xe75d_8278_bfcd_a727,
        neg_point: false,
        neg_bytes: false,
        neg_prime: false,
        replica: false,
    };
    let q_bits = bits_of(Q);
    let p = IntegerMatrixLayout { row_vars: cfg.t, col_vars: cfg.s, word_bits: cfg.w };
    let (pc, vc) = historical_sha_lig_configs(packed_vars(&p)).expect("cfg");
    let inst = build_instance::<Q>(&cfg);
    let rows = build_rows(cfg.t, cfg.s, cfg.w, &inst.dgen);
    let hint = commit_rs_ligerito_rows(&p, rows, &pc);
    let alpha = smallest_generator();
    let mut pt = Blake3Transcript::new();
    let proof = prove_mle_eval_mod_q_ligerito(&mut pt, &hint, &p, &inst.rw, q_bits, alpha, &pc);
    let bytes = proof.to_bytes();
    let y = Fq::<Q>(inst.y_rep);
    let cw_fq: Vec<Fq<Q>> = inst.cw.iter().map(|&x| Fq::<Q>(x)).collect();
    let mut vt = Blake3Transcript::new();
    verify_mle_eval_mod_q_ligerito(
        &mut vt, &hint.commitment, &proof, &p, &inst.rw, &cw_fq, alpha, y, q_bits, &vc,
    )
    .expect("honest baseline verifies");
    println!("diagnostic instance: proof = {} bytes", bytes.len());

    // The stream is [ zinc-side codec fields | 8-byte lig-blob length | lig
    // bincode blob ]. Flipping certain length-prefix bytes — in BOTH the
    // zinc-side codec (unbounded `Vec::with_capacity(lch/n_layers/n_sv)`) and
    // the bincode blob interior — mangles a count into a ~2^46-element
    // reservation whose allocation failure is an UNCATCHABLE process abort
    // (SIGABRT): `catch_unwind` cannot trap `handle_alloc_error`. So EVERY
    // decode is run in a forked child (from_bytes is single-threaded — no
    // rayon — so fork is safe), and the parent classifies from the child's
    // exit status. Decoded-OK cases are then re-decoded and verified in the
    // parent, where the rayon pool is healthy (a proof that decoded without
    // aborting cannot abort on a second decode).
    let total = bytes.len();
    let blob_start = (0..=total - 8)
        .find(|&o| {
            let l = u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
            (o as u64) + 8 + l == total as u64
        })
        .expect("locate lig-blob length prefix by self-consistency");
    println!(
        "zinc-side codec region = [0, {blob_start}); lig bincode blob = [{blob_start}, {total}) ({} B)",
        total - blob_start
    );

    // Parent silences its own panic printer too: verifying a decoded-but-
    // tampered proof legitimately panics in some verifier kernels (a
    // hardening gap), and those are trapped and counted, not printed.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    // A representative, bounded offset sample (forking 2*total times exhausts
    // process resources): the first 96 bytes densely — every zinc-side
    // length-prefix landmine (lch@0, per-chunk n_layers / n_sv) lives among
    // the codec's small integer fields — plus a spread across the whole
    // stream (zinc body and bincode blob) and the outer blob length prefix.
    let mut offsets: Vec<usize> = (0..96.min(total)).collect();
    offsets.extend((96..total).step_by((total / 400).max(1)));
    for o in blob_start..(blob_start + 8).min(total) {
        offsets.push(o); // outer blob length prefix (guarded — should reject cleanly)
    }
    offsets.sort_unstable();
    offsets.dedup();

    let xors: [u8; 2] = [0x01, 0x80];
    let mut aborted = 0usize; // uncatchable alloc/abort in child (from_bytes)
    let mut rejected_decode = 0usize; // from_bytes returned Err
    let mut decode_panic = 0usize; // from_bytes raised a (catchable) panic
    let mut rejected_verify = 0usize; // decoded-different, verify returned Err
    let mut verify_panic = 0usize; // decoded-different, verify PANICKED (caught)
    let mut ignored: Vec<(usize, u8)> = Vec::new(); // decodes to the identical proof
    let mut accept_diff: Vec<(usize, u8)> = Vec::new(); // CRITICAL: different proof accepted
    let mut abort_first: Option<(usize, u8, usize)> = None;
    let mut verify_panic_first: Option<(usize, u8)> = None;
    for &off in &offsets {
        for &x in &xors {
            let mut m = bytes.clone();
            m[off] ^= x;
            match fork_decode(&m) {
                DecodeOutcome::Aborted => {
                    aborted += 1;
                    if abort_first.is_none() {
                        let region = if off < blob_start { 0 } else { 1 };
                        abort_first = Some((off, x, region));
                    }
                }
                DecodeOutcome::DecodeErr => rejected_decode += 1,
                DecodeOutcome::DecodePanic => decode_panic += 1,
                DecodeOutcome::DecodeOk => {
                    // Safe to re-run in-process now (no abort on this input).
                    let d = IntEvalRsLigModQProof::from_bytes(&m).expect("re-decode ok");
                    if d.to_bytes() == bytes {
                        ignored.push((off, x));
                    } else {
                        let v = catch_unwind(AssertUnwindSafe(|| {
                            let mut vt = Blake3Transcript::new();
                            verify_mle_eval_mod_q_ligerito(
                                &mut vt, &hint.commitment, &d, &p, &inst.rw, &cw_fq, alpha, y,
                                q_bits, &vc,
                            )
                            .is_ok()
                        }));
                        match v {
                            Ok(false) => rejected_verify += 1,
                            Ok(true) => accept_diff.push((off, x)),
                            Err(_) => {
                                verify_panic += 1;
                                if verify_panic_first.is_none() {
                                    verify_panic_first = Some((off, x));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::panic::set_hook(prev_hook);

    let n_mut = offsets.len() * xors.len();
    println!(
        "sampled scan: {} offsets over [0,{total}) x {xors:?} = {n_mut} mutations:\n  \
         {aborted} ABORTED (uncatchable alloc failure in from_bytes),\n  \
         {rejected_decode} rejected@decode (clean Err), {decode_panic} rejected@decode (via PANIC),\n  \
         {rejected_verify} decoded-different+rejected@verify (clean), {verify_panic} decoded-different+rejected@verify (via PANIC),\n  \
         {} IGNORED (decode to identical proof), {} ACCEPT-DIFF (different proof verified)",
        offsets.len(),
        ignored.len(),
        accept_diff.len()
    );
    if let Some((off, x)) = verify_panic_first {
        println!(
            "  first verify-panic at offset {off} xor {x:#04x} — verifier panics (e.g. index-out-of-bounds \
             in a sumcheck kernel) on a decodable-but-tampered proof instead of returning Err (hardening gap; \
             safe rejection, not a soundness break)."
        );
    }
    if let Some((off, x, region)) = abort_first {
        let rname = if region == 0 { "ZINC-SIDE codec" } else { "bincode blob" };
        println!(
            "  first abort at offset {off} xor {x:#04x} (region: {rname}) — a tampered length \
             prefix reserves a huge Vec; unbounded `Vec::with_capacity` in from_bytes \
             (lch/n_layers/n_sv are NOT bounded by remaining(), unlike n_u)."
        );
    }
    // Where do the IGNORED bytes sit? Print each with its 8-byte-aligned window.
    let mut shown = 0usize;
    println!("IGNORED offsets (mutated byte is an unread hole; decode to identical proof):");
    for &(off, x) in &ignored {
        if shown < 40 {
            let base = off & !7;
            let win: Vec<String> = bytes[base..(base + 8).min(bytes.len())]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let region = if off < blob_start { "zinc" } else { "blob" };
            println!(
                "  off {off:6} ({region}) xor {x:#04x}  bytes[{base}..{}] = {}",
                base + 8,
                win.join(" ")
            );
            shown += 1;
        }
    }
    if ignored.len() > shown {
        println!("  ... and {} more", ignored.len() - shown);
    }

    // Targeted classification of the EXACT mutation the main audit flagged
    // (offset 4069, xor 0xec) — known-decodable (it round-tripped in the
    // audit), so this path is abort-safe.
    {
        let (off, x) = (4069usize, 0xecu8);
        let region = if off < blob_start { "zinc-codec" } else { "bincode-blob" };
        let mut m = bytes.clone();
        m[off] ^= x;
        match IntEvalRsLigModQProof::from_bytes(&m) {
            Ok(d) => {
                let same = d.to_bytes() == bytes;
                let mut vt = Blake3Transcript::new();
                let v = verify_mle_eval_mod_q_ligerito(
                    &mut vt, &hint.commitment, &d, &p, &inst.rw, &cw_fq, alpha, y, q_bits, &vc,
                );
                println!(
                    "AUDIT-FLAGGED flip @offset {off} (region: {region}) xor {x:#04x}: decodes OK, \
                     re-encode==original: {same}, verify: {:?} \
                     => {}",
                    v.is_ok(),
                    if same {
                        "NON-CANONICAL ENCODING (mutated byte is an unread hole; decodes to the \
                         identical valid proof). Encoding malleability, NOT statement forgery."
                    } else {
                        "DIFFERENT proof"
                    }
                );
            }
            Err(e) => println!("AUDIT-FLAGGED flip @offset {off}: rejected at decode: {e:?}"),
        }
    }

    println!(
        "\nSUMMARY: 0 mutations forged an accepted statement (accept-diff = {}). The two \
         non-soundness findings are (a) IGNORED bytes = non-canonical encoding tolerated by \
         from_bytes despite the codec's canonicality claim, and (b) ABORTED = uncatchable \
         process abort from unbounded Vec::with_capacity on a tampered length prefix (DoS).",
        accept_diff.len()
    );

    assert!(
        accept_diff.is_empty(),
        "soundness-relevant byte-flip acceptances (different proof verified): {accept_diff:?}"
    );
}

/// Outcome of decoding a mutated proof in a forked child.
#[derive(PartialEq)]
enum DecodeOutcome {
    /// Child aborted (uncatchable alloc failure / SIGABRT / any signal).
    Aborted,
    /// `from_bytes` returned `Ok`.
    DecodeOk,
    /// `from_bytes` returned `Err` (clean rejection).
    DecodeErr,
    /// `from_bytes` raised a catchable panic (safe rejection, but a codec
    /// hardening gap — a panic where an `Err` is expected).
    DecodePanic,
}

/// Run `IntEvalRsLigModQProof::from_bytes(m)` in a forked child so an
/// uncatchable allocation abort cannot take the parent down. `from_bytes`
/// touches no rayon pool, so forking a multithreaded process is safe here
/// (only the calling thread survives in the child, which is all the decode
/// uses). Unix-only; elsewhere it falls back to an in-process decode (which
/// may abort — those platforms are out of scope for this diagnostic).
fn fork_decode(m: &[u8]) -> DecodeOutcome {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        match unsafe { libc_fork() } {
            -1 => DecodeOutcome::DecodeErr, // fork failed: treat as non-fatal
            0 => {
                // Child: silence the default panic printer (tampered proofs
                // legitimately panic during decode), decode under
                // catch_unwind, then _exit hard WITHOUT running destructors.
                // An uncatchable alloc abort never reaches the exit call — it
                // raises SIGABRT, which the parent observes.
                std::panic::set_hook(Box::new(|_| {}));
                let r = catch_unwind(AssertUnwindSafe(|| {
                    IntEvalRsLigModQProof::from_bytes(m).is_ok()
                }));
                let code = match r {
                    Ok(true) => 10,  // decoded Ok
                    Ok(false) => 11, // clean Err
                    Err(_) => 12,    // catchable panic
                };
                unsafe { libc_exit(code) };
            }
            pid => {
                let mut status: i32 = 0;
                unsafe { libc_waitpid(pid, &mut status, 0) };
                let st = std::process::ExitStatus::from_raw(status);
                if st.signal().is_some() {
                    DecodeOutcome::Aborted
                } else {
                    match st.code() {
                        Some(10) => DecodeOutcome::DecodeOk,
                        Some(12) => DecodeOutcome::DecodePanic,
                        _ => DecodeOutcome::DecodeErr,
                    }
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        match catch_unwind(AssertUnwindSafe(|| {
            IntEvalRsLigModQProof::from_bytes(m).is_ok()
        })) {
            Ok(true) => DecodeOutcome::DecodeOk,
            Ok(false) => DecodeOutcome::DecodeErr,
            Err(_) => DecodeOutcome::DecodePanic,
        }
    }
}

// Minimal libc bindings for the forked decode (avoids a dev-dependency).
#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "fork"]
    fn libc_fork() -> i32;
    #[link_name = "waitpid"]
    fn libc_waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    #[link_name = "_exit"]
    fn libc_exit(code: i32) -> !;
}

/// COMPLETENESS-GAP repro: at a multi-chunk opening (L >= 2), evaluating at a
/// point whose ROW coordinates are all boolean (r1 in {0,1}^t) makes the row
/// eq-weight vector a unit vector (one entry = 1, the rest 0). Every such
/// weight is < 2^{c_w}, so the TOP weight chunk W_b^{(L-1)} is all zeros; the
/// verifier's R̂ = sum_(b,j) eq(.,z)*(alpha^{W_b^{(L-1)}*2^j} - 1)*eq(.,r*) is
/// then identically zero (alpha^0 - 1 = 0), and it rejects the HONEST proof
/// with Common(RHatZero). A boolean opening point is a standard, valid MLE
/// evaluation, so this is a genuine (structured-point) completeness gap — not
/// a soundness issue. The same shape at a GENERIC point verifies, isolating
/// the cause to the point structure.
///
/// Minimal deterministic instance: t=7, s=8, W=1 (t+log2W = 7 = packing floor,
/// m_p = 8 = config floor), q = 2^126-137 (so c_w = 127-7-1 = 119, q_bits = 126
/// > 119 => L = 2), with an all-boolean r1.
#[test]
fn rhatzero_boolean_point_completeness_gap() {
    const Q: u128 = (1u128 << 126) - 137;
    let (t, s, w) = (7usize, 8usize, 1usize);
    let q_bits = bits_of(Q);
    let p = IntegerMatrixLayout { row_vars: t, col_vars: s, word_bits: w };
    assert_eq!(mod_q_num_chunks(&p, q_bits), 2, "shape must be multi-chunk (L=2)");
    let (pc, vc) = historical_sha_lig_configs(packed_vars(&p)).expect("cfg");
    let alpha = smallest_generator();

    let dgen = DataGen {
        class: DataClass::Uniform,
        seed: 0xC0FFEE,
        mask: 1,
        s,
        single: (0, 0, 0),
        replica: false,
    };
    let rows = build_rows(t, s, w, &dgen);
    let hint = commit_rs_ligerito_rows(&p, rows, &pc);
    let rows_n = 1usize << t;
    let cols_n = 1usize << s;

    let verify_at = |r1: &[u128], r2: &[u128]| -> Result<(), String> {
        let rw = eq_table(r1, Q);
        let cw = eq_table(r2, Q);
        let y = reference_eval(rows_n, cols_n, &dgen, &rw, &cw, Q);
        let cw_fq: Vec<Fq<Q>> = cw.iter().map(|&x| Fq::<Q>(x)).collect();
        let mut pt = Blake3Transcript::new();
        let proof = prove_mle_eval_mod_q_ligerito(&mut pt, &hint, &p, &rw, q_bits, alpha, &pc);
        let mut vt = Blake3Transcript::new();
        verify_mle_eval_mod_q_ligerito(
            &mut vt,
            &hint.commitment,
            &proof,
            &p,
            &rw,
            &cw_fq,
            alpha,
            Fq::<Q>(y),
            q_bits,
            &vc,
        )
        .map_err(|e| format!("{e:?}"))
    };

    // Boolean row point => honest proof REJECTED with RHatZero.
    let r1_bool: Vec<u128> = (0..t).map(|k| (k & 1) as u128).collect(); // in {0,1}
    let r2_bool: Vec<u128> = (0..s).map(|k| ((k + 1) & 1) as u128).collect();
    let boolean = verify_at(&r1_bool, &r2_bool);
    println!("boolean-point (r1 in {{0,1}}^t, L=2) honest verify => {boolean:?}");
    assert!(
        boolean.as_ref().is_err_and(|e| e.contains("RHatZero")),
        "expected the honest proof at a boolean point to be rejected with RHatZero, got {boolean:?}"
    );

    // Same shape, a GENERIC point => honest proof VERIFIES (isolates cause).
    let r1_gen: Vec<u128> =
        (0..t).map(|k| (0x9E37_79B9_u128.wrapping_mul(k as u128 + 3)) % Q).collect();
    let r2_gen: Vec<u128> =
        (0..s).map(|k| (0xD1B5_4A32_u128.wrapping_mul(k as u128 + 5)) % Q).collect();
    let generic = verify_at(&r1_gen, &r2_gen);
    println!("generic-point (same shape, L=2) honest verify => {generic:?}");
    assert!(
        generic.is_ok(),
        "the SAME multi-chunk shape at a generic point must verify (cause is the point structure): {generic:?}"
    );
    println!("=> CONFIRMED: multi-chunk (L>=2) + boolean/small row point => RHatZero (honest rejection).");
}
