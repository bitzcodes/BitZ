//! Ring-switch primitives and the backend-independent prover/verifier prefix
//! for the integer-MLE-eval opener.
//!
//! This module holds the pieces the flock-backed opener
//! ([`crate::ligerito_flock`]) composes on top of the shared core in
//! [`crate::pcs`]:
//!
//! 1. **Common prefix** ([`prove_int_eval_common`] /
//!    [`prove_int_eval_merged_common`] and their verifier duals): the forest
//!    GKR that binds `α^{v_c}`, the `v` message, and a de-black-boxing
//!    degree-2 **pre-sumcheck** over the row-bit variables on `Σ_i R(i)·m_ξ(i)`
//!    (`R = eq(·,ρ)⊙(α-powers−1)` is [`row_bit_weights`], `m_ξ` the
//!    `eq(c,ξ)`-combined row). This strips the α-power weight factor, leaving a
//!    pure bit-MLE evaluation claim `M̂(r*, ξ) = μ`; the verifier's only
//!    non-succinct step is evaluating `R̂(r*)` (the `O(2^t·W)` `q_rowbit`
//!    exponentiation table).
//! 2. **Ring-switch** ([`ring_switch_prove`] / [`ring_switch_verify`], Flock
//!    paper App. B, modular version): the prover sends the 128 partial
//!    evaluations `s_v = M̂(r_hi, v)`; the verifier checks
//!    `Σ_v eq(r_lo,v)·s_v = μ`, batches with a fresh `r″` through the injective
//!    `F_2`-recombination (a 128×128 bit transpose, [`transpose_bits_128`]),
//!    and the residual claim `Σ_y B(y)·P̂(y) = β₀` (with
//!    `B(y) = Φ_{r″}(eq(r_hi,y))`) becomes an inner-product claim on the PACKED
//!    polynomial `P` for the recursive Ligerito opener.
//! 3. **Succinct residual basis** ([`tensor_eq_phi_eval`] /
//!    [`residual_b_evals`]): the Ligerito verifier evaluates `B̂` at the fold
//!    challenges via the tensor-algebra trick (`O(m·128²)` — no `2^m`-sized
//!    table).
//!
//! Algorithm provenance: the ring-switch follows Flock (`pcs/ring_switch.rs`,
//! Apache-2.0 OR MIT, Succinct Labs / Bünz / Wang; ring-switching due to
//! Diamond–Posen, Ligerito to Novakovic–Angeris, with Flock's `ring_switch`
//! itself derived from bcc-research/bolt-rs).
//!
//! The commitment root is published and absorbed by the caller before any
//! challenge is drawn (the test harnesses share one transcript prefix between
//! prover and verifier).

use crate::piop::lookup::gkr_product::{ProductForestProof, verify_product_forest};
use crate::piop::sumcheck::multi_degree::MultiDegreeSumcheckProof;
#[cfg(test)]
use crate::poly::coefficient::FieldRepresentation;
#[cfg(test)]
use crate::poly::mle::DenseMultilinearExtension;
use crate::poly::univariate::binary_gf128::{Gf128 as Gf, REDUCTION_LOW_GF128};
use crate::poly::utils::build_eq_x_r_vec;
use crate::transcript::traits::Transcript;

use crate::pcs::{
    GF128_MULT_ORDER, IntegerMatrixLayout, gf_pow, is_generator, max_fold_magnitude,
    row_bit_weights,
};
use crate::utils::{cfg_chunks_mut, cfg_into_iter};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Per-column bit rows, 64 bits per `u64` word (row-major over the row-bit
/// index `i = (b<<log₂W)|j`). Local copy — the sibling branches' commit
/// paths pack differently; the Ligerito stack owns its row layout.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn repack_leaf_bits(p: &IntegerMatrixLayout, data: &[u128]) -> Vec<Vec<u64>> {
    let log_w = p.word_bits.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    let words = (row_len + 63) >> 6;
    cfg_into_iter!(0..p.cols())
        .map(|c| {
            let mut w = vec![0u64; words];
            for b in 0..p.rows() {
                let cell = data[p.cell_index(b, c)];
                for j in 0..p.word_bits {
                    if (cell >> j) & 1 == 1 {
                        let i = (b << log_w) | j;
                        w[i >> 6] |= 1u64 << (i & 63);
                    }
                }
            }
            w
        })
        .collect()
}
/// Number of packed (in-pack) coordinates: 128 bits per `GF(2^128)` element.
pub const LOG_PACKING: usize = 7;

// ---------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------

/// Shape/soundness knobs of the RS opening.
#[derive(Clone, Copy, Debug)]
pub struct RsOpenConfig {
    /// RS rate `2^{-log_inv_rate}`.
    pub log_inv_rate: usize,
    /// `log₂` of the interleaved lane count (Ligero row dimension); the first
    /// `log_batch` sumcheck rounds are row-batch rounds.
    pub log_batch: usize,
    /// Number of FRI queries.
    pub num_queries: usize,
    /// `log₂` of the FRI epoch arity: folds per Merkle re-commit.
    pub log_fri_arity: usize,
}

impl RsOpenConfig {
    /// Rate-1/4, 4 lanes, arity-8 epochs. Query count for ~100-bit proximity
    /// at rate 1/4 in the unique-decoding regime (γ = 3/8 ⇒
    /// `⌈100/−log₂(5/8)⌉ = 148`).
    pub fn default_100bit() -> Self {
        Self {
            log_inv_rate: 2,
            log_batch: 2,
            num_queries: 148,
            log_fri_arity: 3,
        }
    }
}

// ---------------------------------------------------------------------
// Small field / table helpers
// ---------------------------------------------------------------------

/// `a·X` in `GF(2^128)`: shift by one with the `0x87` reduction.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn mul_by_x(a: Gf) -> Gf {
    let w = a.as_words();
    let carry = w[1] >> 63;
    let hi = (w[1] << 1) | (w[0] >> 63);
    let mut lo = w[0] << 1;
    if carry == 1 {
        lo ^= REDUCTION_LOW_GF128;
    }
    Gf::from_polynomial_words([lo, hi])
}

/// Absorb a `Gf` slice into the transcript with a domain tag (framed by
/// `absorb_slice`; little-endian words, matching the repo's byte layout).
#[allow(clippy::arithmetic_side_effects)]
fn absorb_gf_slice(transcript: &mut impl Transcript, tag: u8, vals: &[Gf]) {
    let mut bytes = Vec::with_capacity(vals.len() * 16 + 1);
    bytes.push(tag);
    for v in vals {
        let w = v.as_words();
        bytes.extend_from_slice(&w[0].to_le_bytes());
        bytes.extend_from_slice(&w[1].to_le_bytes());
    }
    transcript.absorb_slice(&bytes);
}

/// Absorb a virtual-XOR claim's external residuals (domain tag 0x37).
pub(crate) fn absorb_externals(transcript: &mut impl Transcript, vals: &[Gf]) {
    absorb_gf_slice(transcript, 0x37, vals);
}

/// Absorb the RLC-family discharge's committed closing openings
/// `ω_i = M̂_i(ρ)` (domain tag 0x38).
pub(crate) fn absorb_rlc_omegas(transcript: &mut impl Transcript, vals: &[Gf]) {
    absorb_gf_slice(transcript, 0x38, vals);
}

/// Absorb the two-phase RLC discharge's per-chunk phase-B entry sums β_l
/// (domain tag 0x39) — bound between phase A and phase B.
pub(crate) fn absorb_rlc_betas(transcript: &mut impl Transcript, vals: &[Gf]) {
    absorb_gf_slice(transcript, 0x39, vals);
}

/// Absorb the Round-0 (out-of-domain) value `y = MLE[P](ζ⃗)` (domain tag
/// 0x50) — bound right after the `ζ` draw, before any forest message.
pub(crate) fn absorb_ood_value(transcript: &mut impl Transcript, y: Gf) {
    absorb_gf_slice(transcript, 0x50, &[y]);
}

/// In-place multilinear bind of the LOWEST index bit:
/// `tbl'[i] = tbl[2i] + r·(tbl[2i] + tbl[2i+1])`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn bind_low(tbl: &mut Vec<Gf>, r: Gf) {
    let half = tbl.len() >> 1;
    for i in 0..half {
        let u = tbl[2 * i];
        let v = tbl[2 * i + 1];
        tbl[i] = u + r * (u + v);
    }
    tbl.truncate(half);
}

/// Evaluate the multilinear with value table `tbl` (index bit `k` ↔
/// `point[k]`) at `point`, by repeated low-bit binds.
#[allow(clippy::arithmetic_side_effects)]
pub fn mle_eval(tbl: &[Gf], point: &[Gf]) -> Gf {
    assert_eq!(
        tbl.len(),
        1usize << point.len(),
        "table/point size mismatch"
    );
    let mut buf = tbl.to_vec();
    for &r in point {
        bind_low(&mut buf, r);
    }
    buf[0]
}

/// 128×128 bit transpose of `GF(2^128)` elements viewed as `F_2^{128}`
/// vectors: `bit_v(out[u]) = bit_u(inp[v])`. The injective recombination
/// `s_u = Σ_v s_{u,v}·β_v` of the ring-switch (paper Eq. 8).
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn transpose_bits_128(inp: &[Gf]) -> Vec<Gf> {
    assert_eq!(inp.len(), 128);
    let mut out = vec![[0u64; 2]; 128];
    for (v, s) in inp.iter().enumerate() {
        let w = s.as_words();
        for u in 0..128 {
            if (w[u >> 6] >> (u & 63)) & 1 == 1 {
                out[u][v >> 6] |= 1u64 << (v & 63);
            }
        }
    }
    out.into_iter().map(Gf::from_polynomial_words).collect()
}

/// `Σ_u bit_w(X^u·y)·cols[u]` for every `w`: one tensor-algebra
/// right-leg multiplication step. (Shared with the structured-tap MPS
/// closure, [`crate::taps`].)
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn apply_right_mul(cols: &[Gf], y: Gf) -> Vec<Gf> {
    let mut out = vec![Gf::zero(); 128];
    let mut row = y; // X^u · y, starting at u = 0
    for cu in cols.iter().take(128) {
        if !cu.is_zero() {
            let w = row.as_words();
            for wi in 0..2usize {
                let mut bits = w[wi];
                while bits != 0 {
                    let t = bits.trailing_zeros() as usize;
                    out[(wi << 6) | t] += *cu;
                    bits &= bits.wrapping_sub(1);
                }
            }
        }
        row = mul_by_x(row);
    }
    out
}

/// Succinct evaluation of `B̂(chals)` where `B(y) = Φ_{r″}(eq(r_hi, y))` and
/// `Φ_{r″}: β_u ↦ eq_r2[u]` is the `F_2`-linear batching map — the
/// tensor-algebra trick (Diamond–Posen; Flock's `eval_rs_eq`):
/// maintain `E = ∏_k [(1+a_k)⊗(1+h_k) + a_k⊗h_k] ∈ K ⊗_{F_2} K` in the
/// column representation `E = Σ_u e_u ⊗ β_u`, then contract with `Φ`:
/// `B̂(chals) = Σ_u e_u · eq_r2[u]`. Cost `O(len·128²)` bit-conditional adds.
pub fn tensor_eq_phi_eval(chals: &[Gf], r_hi: &[Gf], eq_r2: &[Gf]) -> Gf {
    let evals = residual_b_evals(chals, 0, r_hi, eq_r2);
    evals[0]
}

/// The Ligerito residual-block variant: `B̂(prefix ++ bits(y))` for every
/// boolean tail `y ∈ [0, 2^{yr_log_n})` (tail bit `j` ↔ coordinate
/// `prefix.len() + j`). Shares the tensor prefix across all tails — the
/// boolean tail legs are single-term algebra multiplications, so the whole
/// block costs `O((prefix.len() + 2^{yr_log_n})·128²)` instead of
/// `2^{yr_log_n}` full evaluations. This is the succinct-basis hook the
/// Ligerito verifier calls once at its residual check.
#[allow(clippy::arithmetic_side_effects)]
pub fn residual_b_evals(prefix: &[Gf], yr_log_n: usize, r_hi: &[Gf], eq_r2: &[Gf]) -> Vec<Gf> {
    assert_eq!(
        prefix.len().wrapping_add(yr_log_n),
        r_hi.len(),
        "prefix + tail must cover the point"
    );
    assert_eq!(eq_r2.len(), 128);
    let one = Gf::one();
    let mut cols = vec![Gf::zero(); 128];
    cols[0] = one;
    for (a, h) in prefix.iter().zip(r_hi.iter()) {
        let f0 = apply_right_mul(&cols, one + *h);
        let f1 = apply_right_mul(&cols, *h);
        let la = one + *a;
        for w in 0..128 {
            cols[w] = la * f0[w] + *a * f1[w];
        }
    }
    // Bit tail expansion: at each step j, the left leg is 0 or 1, so
    // E·[(1+a)⊗(1+h) + a⊗h] collapses to the single term E·(1⊗(1+h)) (bit 0)
    // or E·(1⊗h) (bit 1). Tail index bit j is appended at weight 2^j.
    let mut cur: Vec<Vec<Gf>> = vec![cols];
    for j in 0..yr_log_n {
        let h = r_hi[prefix.len() + j];
        let stride = cur.len();
        let mut next: Vec<Vec<Gf>> = Vec::with_capacity(stride * 2);
        next.resize(stride * 2, Vec::new());
        for (i, e) in cur.iter().enumerate() {
            next[i] = apply_right_mul(e, one + h);
            next[i + stride] = apply_right_mul(e, h);
        }
        cur = next;
    }
    cur.iter()
        .map(|cols| {
            cols.iter()
                .zip(eq_r2.iter())
                .fold(Gf::zero(), |acc, (c, e)| acc + *c * *e)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Packed commitment
// ---------------------------------------------------------------------

/// `t + log₂W` — the row-bit index width.
pub(crate) fn row_bit_vars(p: &IntegerMatrixLayout) -> usize {
    let log_w = p.word_bits.trailing_zeros() as usize;
    p.row_vars.wrapping_add(log_w)
}

/// Number of packed variables `m_p = (t + log₂W − 7) + s`.
pub fn packed_vars(p: &IntegerMatrixLayout) -> usize {
    row_bit_vars(p)
        .wrapping_sub(LOG_PACKING)
        .wrapping_add(p.col_vars)
}

// ---------------------------------------------------------------------
// BaseFold: sumcheck ⊗ FRI fold for ⟨weights, P⟩ = target
// ---------------------------------------------------------------------

/// Errors of the RS opening (ring-switch + BaseFold).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RsOpenError {
    /// Malformed proof shape (lengths inconsistent with the config).
    Shape,
    /// `Σ_v eq(r_lo, v)·s_v ≠ μ`.
    RingSwitchClaim,
    /// A sent query position disagrees with the transcript-derived index.
    QueryIndex { sent: usize, expected: usize },
    /// A Merkle path failed.
    Merkle { query: usize, epoch: usize },
    /// An opened coset disagrees with the fold of the previous stage.
    CosetMismatch { query: usize, epoch: usize },
    /// The final codeword is not constant.
    FinalNotConstant,
    /// A query's fold chain landed off the final codeword.
    FinalMismatch { query: usize },
    /// The closing sumcheck identity `T = B̂(chals)·final_b` failed.
    FinalClaim,
    /// `R̂(r*) = 0` (cannot divide; negligible-probability event).
    WeightZero,
}

// ---------------------------------------------------------------------
// Ring-switch
// ---------------------------------------------------------------------

/// Ring-switch message: the 128 partial evaluations `s_v = M̂(r_hi, v)`.
#[derive(Clone, Debug)]
pub struct RingSwitchProof {
    pub s_v: Vec<Gf>,
}

/// Absorb an `s_v` message with the ring-switch domain tag (shared by the
/// single-poly and batched paths — the byte streams must match).
pub(crate) fn absorb_sv(transcript: &mut impl Transcript, s: &[Gf]) {
    absorb_gf_slice(transcript, 0x20, s);
}

/// Absorb the virtual opening's dual-basis batching message `h_i`
/// (fresh domain tag — NOT the `s_v` tag 0x20).
pub(crate) fn absorb_hs(transcript: &mut impl Transcript, s: &[Gf]) {
    absorb_gf_slice(transcript, 0x48, s);
}

// ---------------------------------------------------------------------
// Ring-switch fold kernels (flock-derived; `BITZ_RS_FAST`)
// ---------------------------------------------------------------------

/// Fast ring-switch/basis kernels — the default: the `s_v` in-pack marginals
/// run the method-of-four-Russians fold (flock's
/// `fold_1b_rows_1way_mfr_8wide_k4` shape) and the `Φ_{r″}` basis maps run 16
/// byte-table subset-sum lookups (flock's `fold_b128_elems` shape) instead of
/// data-dependent bit scans; the mod-q opener additionally fuses the Ligerito
/// round-0 message into the basis pass and calls flock's
/// `recursive_prover_with_basis_precomputed_round0`. `BITZ_RS_FAST=0` opts out
/// (restores the scalar bit-scan paths and the plain prover entry point —
/// diagnostic / A-B measurement). Byte-identical proofs either way (exact
/// field-op reassociation only; pinned cross-process by
/// `examples/fuse_check.rs`). Read once per process.
pub(crate) fn rs_fast() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BITZ_RS_FAST").map_or(true, |v| v != "0"))
}

/// 128-bit-word view of a packed-message element, so the fold kernels run
/// over both this module's `Gf` and the flock backend's `F128` without a
/// conversion pass over the 2^{m_p}-element message.
pub(crate) trait PackedBits: Sync {
    fn bit_words(&self) -> [u64; 2];
}

impl PackedBits for Gf {
    #[inline(always)]
    fn bit_words(&self) -> [u64; 2] {
        *self.as_words()
    }
}

/// Hacker's Delight §7-3 8×8 bit-matrix transpose stored in a `u64`
/// (bit `r·8 + c` of the input ↦ bit `c·8 + r` of the output).
#[inline(always)]
pub(crate) fn transpose_8x8_bits(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AAu64;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCCu64;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0u64;
    x ^ t ^ (t << 28)
}

/// 16-entry subset-sum table over 4 elements:
/// `sums[mask] = Σ_{k : bit_k(mask)} e[k]` (15 additions by doubling).
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
pub(crate) fn subset_sums_4(e: [Gf; 4]) -> [Gf; 16] {
    let mut sums = [Gf::zero(); 16];
    for (i, &v) in e.iter().enumerate() {
        let half = 1usize << i;
        for k in 0..half {
            sums[half + k] = sums[k] + v;
        }
    }
    sums
}

/// Scalar bit-scan tail: `s[j] += e` for every set bit `j` of `w`.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
pub(crate) fn sv_scalar_accum(s: &mut [Gf], w: [u64; 2], e: Gf) {
    for wi in 0..2usize {
        let mut bits = w[wi];
        while bits != 0 {
            let t = bits.trailing_zeros() as usize;
            s[(wi << 6) | t] += e;
            bits &= bits.wrapping_sub(1);
        }
    }
}

/// `s_v[j] = Σ_y eq[y]·bit_j(wit[y])` — the in-pack marginal, computed with
/// the method-of-four-Russians fold: per 8 witness elements, two 16-entry
/// subset-sum tables over their `eq` values, then per byte position one 8×8
/// bit transpose and per output bit **two table lookups + one accumulator
/// RMW**, independent of bit density (the scalar path pays one
/// data-dependent branchy add per set bit — ~64/element at random data).
/// Chunk-parallel with per-chunk partial accumulators; exact field sums, so
/// the result is bit-identical to the scalar scan for any summation order.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::needless_range_loop)] // r_byte loop mirrors flock's kernel 1:1
pub(crate) fn sv_fold_mfr<T: PackedBits>(wit: &[T], eq: &[Gf]) -> Vec<Gf> {
    assert_eq!(wit.len(), eq.len());
    const CHUNK: usize = 1 << 12; // multiple of 8 ⇒ only the global tail is scalar
    let n_chunks = wit.len().div_ceil(CHUNK).max(1);
    let partials: Vec<Vec<Gf>> = cfg_into_iter!(0..n_chunks)
        .map(|c| {
            let lo = c * CHUNK;
            let hi = (lo + CHUNK).min(wit.len());
            let mut s = vec![Gf::zero(); 128];
            let mut y = lo;
            while y + 8 <= hi {
                // Zero words contribute nothing: a block of eight zero
                // words (a virtual layout's unused lanes, zero padding)
                // skips its subset-sum tables and transposes.
                if wit[y..y + 8].iter().all(|w| w.bit_words() == [0, 0]) {
                    y += 8;
                    continue;
                }
                let lo_tbl = subset_sums_4([eq[y], eq[y + 1], eq[y + 2], eq[y + 3]]);
                let hi_tbl = subset_sums_4([eq[y + 4], eq[y + 5], eq[y + 6], eq[y + 7]]);
                let mut m_bytes = [[0u8; 16]; 8];
                for (e, slot) in m_bytes.iter_mut().enumerate() {
                    let w = wit[y + e].bit_words();
                    slot[..8].copy_from_slice(&w[0].to_le_bytes());
                    slot[8..].copy_from_slice(&w[1].to_le_bytes());
                }
                for r_byte in 0..16 {
                    let combined: u64 = (m_bytes[0][r_byte] as u64)
                        | ((m_bytes[1][r_byte] as u64) << 8)
                        | ((m_bytes[2][r_byte] as u64) << 16)
                        | ((m_bytes[3][r_byte] as u64) << 24)
                        | ((m_bytes[4][r_byte] as u64) << 32)
                        | ((m_bytes[5][r_byte] as u64) << 40)
                        | ((m_bytes[6][r_byte] as u64) << 48)
                        | ((m_bytes[7][r_byte] as u64) << 56);
                    let tb = transpose_8x8_bits(combined).to_le_bytes();
                    let base = r_byte * 8;
                    for (p, &mask) in tb.iter().enumerate() {
                        s[base + p] +=
                            lo_tbl[(mask & 0x0F) as usize] + hi_tbl[(mask >> 4) as usize];
                    }
                }
                y += 8;
            }
            while y < hi {
                sv_scalar_accum(&mut s, wit[y].bit_words(), eq[y]);
                y += 1;
            }
            s
        })
        .collect();
    let mut s = vec![Gf::zero(); 128];
    for part in &partials {
        for (a, b) in s.iter_mut().zip(part.iter()) {
            *a += *b;
        }
    }
    s
}

/// `Φ_{r″}` — the F₂-linear batching map on the bit representation:
/// `β_u ↦ eq_r2[u]`, applied to a K element by summing over its set bits.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
pub(crate) fn phi_bit_sum(ev: Gf, eq_r2: &[Gf]) -> Gf {
    let w = ev.as_words();
    let mut acc = Gf::zero();
    for wi in 0..2usize {
        let mut bits = w[wi];
        while bits != 0 {
            let t = bits.trailing_zeros() as usize;
            acc += eq_r2[(wi << 6) | t];
            bits &= bits.wrapping_sub(1);
        }
    }
    acc
}

/// 16 byte-position subset-sum tables of `scale·eq_r2`:
/// `T[pos·256 + v] = Σ_{bit j of v} scale·eq_r2[pos·8 + j]` (64 KB). A
/// `Φ_{r″}` image then costs 16 gathers + a XOR tree ([`phi_from_words`])
/// instead of a data-dependent bit scan, and premultiplying `scale` (the
/// batching `η`) into the tables removes the per-element `η·Φ(…)` field
/// multiply entirely.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn phi_byte_tables(eq_r2: &[Gf], scale: Gf) -> Vec<Gf> {
    debug_assert_eq!(eq_r2.len(), 128);
    let mut t = vec![Gf::zero(); 16 * 256];
    phi_byte_tables_into(&mut t, |i| scale * eq_r2[i]);
    t
}

/// Fill the byte-position subset sums of 128 coefficients, overwriting `out`.
/// The callback lets callers supply precomputed or scaled coefficients without
/// allocating an intermediate table or multiplying the unit-scale case.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn phi_byte_tables_into(out: &mut [Gf], coefficient: impl Fn(usize) -> Gf) {
    debug_assert_eq!(out.len(), 16 * 256);
    for pos in 0..16usize {
        let table = &mut out[pos << 8..(pos + 1) << 8];
        table[0] = Gf::zero();
        for j in 0..8usize {
            let base = coefficient((pos << 3) | j);
            let half = 1usize << j;
            for k in 0..half {
                table[half + k] = table[k] + base;
            }
        }
    }
}

/// `Σ_l T_l[…]` gather for one element: 16 byte-indexed lookups into a
/// [`phi_byte_tables`] table, tree-reduced. Equals
/// `scale·Φ_{r″}(element)` bit-for-bit.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
pub(crate) fn phi_from_words(w: [u64; 2], tables: &[Gf]) -> Gf {
    let lb = w[0].to_le_bytes();
    let hb = w[1].to_le_bytes();
    let p0 = tables[lb[0] as usize] + tables[(1 << 8) | lb[1] as usize];
    let p1 = tables[(2 << 8) | lb[2] as usize] + tables[(3 << 8) | lb[3] as usize];
    let p2 = tables[(4 << 8) | lb[4] as usize] + tables[(5 << 8) | lb[5] as usize];
    let p3 = tables[(6 << 8) | lb[6] as usize] + tables[(7 << 8) | lb[7] as usize];
    let p4 = tables[(8 << 8) | hb[0] as usize] + tables[(9 << 8) | hb[1] as usize];
    let p5 = tables[(10 << 8) | hb[2] as usize] + tables[(11 << 8) | hb[3] as usize];
    let p6 = tables[(12 << 8) | hb[4] as usize] + tables[(13 << 8) | hb[5] as usize];
    let p7 = tables[(14 << 8) | hb[6] as usize] + tables[(15 << 8) | hb[7] as usize];
    ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
}

/// Prover: compute and absorb `s_v`, draw `r″`, and produce the BaseFold
/// weight table `B(y) = Φ_{r″}(eq(r_hi, y))` (plus `eq_r2` and `β₀` for
/// debugging/tests).
#[allow(clippy::arithmetic_side_effects)]
pub fn ring_switch_prove<T: PackedBits>(
    transcript: &mut impl Transcript,
    p_msg: &[T],
    r_hi: &[Gf],
) -> (RingSwitchProof, Vec<Gf>, Gf) {
    ring_switch_prove_with(transcript, p_msg, r_hi, |b| b)
}

/// [`ring_switch_prove`] with the basis `B(y)` written through `convert`
/// as it is produced — a caller that needs it in another bit-compatible
/// element type (flock's `F128`) gets it without a second pass over the
/// 2^m-element table.
pub fn ring_switch_prove_with<T: PackedBits, O: Send>(
    transcript: &mut impl Transcript,
    p_msg: &[T],
    r_hi: &[Gf],
    convert: impl Fn(Gf) -> O + Sync + Send,
) -> (RingSwitchProof, Vec<O>, Gf) {
    let eq_hi = build_eq_x_r_vec(r_hi, &()).expect("non-empty r_hi");
    assert_eq!(eq_hi.len(), p_msg.len());

    // s_v = Σ_y eq_hi[y] · bit_v(P[y]): parallel partial accumulators over
    // y-chunks, merged by field addition (exact, order-independent).
    let s = if rs_fast() {
        sv_fold_mfr(p_msg, &eq_hi)
    } else {
        const SV_CHUNK: usize = 1 << 12;
        let num_chunks = p_msg.len().div_ceil(SV_CHUNK);
        let partials: Vec<Vec<Gf>> = cfg_into_iter!(0..num_chunks)
            .map(|ci| {
                let lo = ci * SV_CHUNK;
                let hi = (lo + SV_CHUNK).min(p_msg.len());
                let mut local = vec![Gf::zero(); 128];
                for y in lo..hi {
                    sv_scalar_accum(&mut local, p_msg[y].bit_words(), eq_hi[y]);
                }
                local
            })
            .collect();
        let mut s = vec![Gf::zero(); 128];
        for local in &partials {
            for (acc, l) in s.iter_mut().zip(local.iter()) {
                *acc += *l;
            }
        }
        s
    };
    absorb_sv(transcript, &s);
    let r2: Vec<Gf> = transcript.get_field_challenges(LOG_PACKING, &());
    let eq_r2 = build_eq_x_r_vec(&r2, &()).expect("r2 non-empty");

    // β₀ = Σ_u eq_r2[u]·s_u via the bit transpose.
    let s_u = transpose_bits_128(&s);
    let beta0 = s_u
        .iter()
        .zip(eq_r2.iter())
        .fold(Gf::zero(), |acc, (su, e)| acc + *su * *e);

    // B(y) = Φ_{r″}(eq_hi[y]) = Σ_{u: bit_u(eq_hi[y])} eq_r2[u]. Parallel per y.
    let phi_slow = |y: usize| {
        let w = eq_hi[y].as_words();
        let mut acc = Gf::zero();
        for wi in 0..2usize {
            let mut bits = w[wi];
            while bits != 0 {
                let t = bits.trailing_zeros() as usize;
                acc += eq_r2[(wi << 6) | t];
                bits &= bits.wrapping_sub(1);
            }
        }
        acc
    };
    let b_tbl: Vec<O> = if rs_fast() {
        let tables = phi_byte_tables(&eq_r2, Gf::one());
        cfg_into_iter!(0..eq_hi.len())
            .map(|y| convert(phi_from_words(*eq_hi[y].as_words(), &tables)))
            .collect()
    } else {
        cfg_into_iter!(0..eq_hi.len())
            .map(|y| convert(phi_slow(y)))
            .collect()
    };
    debug_assert_eq!(
        (0..eq_hi.len())
            .zip(p_msg.iter())
            .fold(Gf::zero(), |a, (y, p)| a + phi_slow(y)
                * Gf::from_polynomial_words(p.bit_words())),
        beta0,
        "ring-switch recombination identity"
    );

    (RingSwitchProof { s_v: s }, b_tbl, beta0)
}

/// Verifier: check `Σ_v eq(r_lo,v)·s_v = μ`, absorb, draw `r″`, and return
/// `(eq_r2, β₀)` for the BaseFold stage.
#[allow(clippy::arithmetic_side_effects)]
pub fn ring_switch_verify(
    transcript: &mut impl Transcript,
    proof: &RingSwitchProof,
    mu: Gf,
    r_lo: &[Gf],
) -> Result<(Vec<Gf>, Gf), RsOpenError> {
    if proof.s_v.len() != 128 || r_lo.len() != LOG_PACKING {
        return Err(RsOpenError::Shape);
    }
    let eq_lo = build_eq_x_r_vec(r_lo, &()).expect("r_lo non-empty");
    let claim = proof
        .s_v
        .iter()
        .zip(eq_lo.iter())
        .fold(Gf::zero(), |acc, (s, e)| acc + *s * *e);
    if claim != mu {
        return Err(RsOpenError::RingSwitchClaim);
    }
    absorb_sv(transcript, &proof.s_v);
    let r2: Vec<Gf> = transcript.get_field_challenges(LOG_PACKING, &());
    let eq_r2 = build_eq_x_r_vec(&r2, &()).expect("r2 non-empty");
    let s_u = transpose_bits_128(&proof.s_v);
    let beta0 = s_u
        .iter()
        .zip(eq_r2.iter())
        .fold(Gf::zero(), |acc, (su, e)| acc + *su * *e);
    Ok((eq_r2, beta0))
}

// ---------------------------------------------------------------------
// End-to-end integer-MLE evaluation with the RS opening
// ---------------------------------------------------------------------

/// Errors of the end-to-end RS-opened integer-MLE evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntEvalRsError {
    Forest,
    ChallengeNotGenerator,
    RootBinding {
        c: usize,
    },
    Magnitude {
        max: u128,
    },
    /// The pre-sumcheck rejected, or its claimed sum disagrees with the
    /// forest-derived batched claim `Y_ξ`.
    PreSumcheck,
    /// The RLC-family monomial discharge rejected: the sumcheck failed, a
    /// group's claimed sum disagrees with the η-batched |S| ≥ 2 residuals,
    /// or the closing `Â_S(ρ)·Π ω_i` check failed.
    Discharge,
    /// `R̂(r*) = 0` (negligible; resample).
    RHatZero,
    Open(RsOpenError),
    ReadOff,
}

/// [`xi_combined_rows`] with the monomial (AND) rows fused into the scan:
/// `m_ξ[i] = Σ_c eq_ξ[c]·∧_k M_k[c][i]` — the AND-of-streams row set is
/// never materialised.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn xi_combined_rows_and(
    p: &IntegerMatrixLayout,
    row_sets: &[&[Vec<u64>]],
    eq_xi: &[Gf],
) -> Vec<Gf> {
    let t_w = row_bit_vars(p);
    let len = 1usize << t_w;
    debug_assert!(!row_sets.is_empty());
    debug_assert!(
        len >= 64,
        "row_len below one word is out of scope (t_w >= 7 holds here)"
    );
    let mut m = vec![Gf::zero(); len];
    cfg_chunks_mut!(m, 64).enumerate().for_each(|(wi, block)| {
        for c in 0..p.cols() {
            let mut bits = row_sets[0][c][wi];
            for rs in &row_sets[1..] {
                bits &= rs[c][wi];
            }
            while bits != 0 {
                let t = bits.trailing_zeros() as usize;
                block[t] += eq_xi[c];
                bits &= bits.wrapping_sub(1);
            }
        }
    });
    m
}

/// `eq(c,ξ)`-combined rows: `m_ξ[i] = Σ_c eq_ξ[c]·M[c][i]`, from the packed
/// bit rows. Parallel over 64-entry output blocks (each block scans all
/// rows' matching word — exact field sums, order-independent in char 2).
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn xi_combined_rows(
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    eq_xi: &[Gf],
) -> Vec<Gf> {
    let t_w = row_bit_vars(p);
    let len = 1usize << t_w;
    debug_assert!(
        len >= 64,
        "row_len below one word is out of scope (t_w >= 7 holds here)"
    );
    let mut m = vec![Gf::zero(); len];
    cfg_chunks_mut!(m, 64).enumerate().for_each(|(wi, block)| {
        for (c, row) in rows.iter().enumerate() {
            let mut bits = row[wi];
            while bits != 0 {
                let t = bits.trailing_zeros() as usize;
                block[t] += eq_xi[c];
                bits &= bits.wrapping_sub(1);
            }
        }
    });
    m
}

/// [`xi_combined_rows`] off the hint's 64-column-per-word store: per lane
/// group `g`, precombine `eq_xi` into 8 byte-position subset-sum tables
/// (`tb[pos][byte] = Σ_{b∈byte} eq_xi[64g + 8·pos + b]`, built by
/// doubling), then every output position is `8·⌈cols/64⌉` table gathers —
/// no per-column scatter, no tz-walk dependency chain (the
/// [`phi_byte_tables`] trick on the ξ axis; ~2× the scatter form at
/// n = 28 and it reads `packed_cols` instead of re-scanning `rows`).
/// Exact char-2 re-association — byte-identical.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn xi_combined_rows_packed(
    p: &IntegerMatrixLayout,
    packed_cols: &[Vec<u64>],
    eq_xi: &[Gf],
) -> Vec<Gf> {
    let t_w = row_bit_vars(p);
    let len = 1usize << t_w;
    let cols = p.cols();
    let groups = cols.div_ceil(64);
    debug_assert!(packed_cols.len() >= groups && packed_cols[0].len() == len);
    let tables: Vec<Vec<Gf>> = cfg_into_iter!(0..groups)
        .map(|g| {
            let mut t = vec![Gf::zero(); 8 << 8];
            for pos in 0..8usize {
                let base_col = (g << 6) | (pos << 3);
                let tb = &mut t[pos << 8..(pos + 1) << 8];
                for b in 0..8usize {
                    let c = base_col + b;
                    if c >= cols {
                        break;
                    }
                    let w = eq_xi[c];
                    let lim = 1usize << b;
                    for m in 0..lim {
                        tb[m | lim] = tb[m] + w;
                    }
                }
            }
            t
        })
        .collect();
    // Group-OUTER accumulation per output chunk: each group's positions
    // stream sequentially, its 32 KB table stays L1-hot, and the chunk's
    // output slots are L1-resident RMW — no cross-group pointer chases in
    // the inner loop and no 256-deep serial add chain per output.
    let mut m = vec![Gf::zero(); len];
    cfg_chunks_mut!(m, 1 << 10)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci << 10;
            for (g, tg) in tables.iter().enumerate() {
                let src = &packed_cols[g][base..base + chunk.len()];
                for (slot, &x) in chunk.iter_mut().zip(src.iter()) {
                    let mut x = x;
                    let mut pos = 0usize;
                    while x != 0 {
                        let byte = (x & 0xFF) as usize;
                        if byte != 0 {
                            *slot += tg[(pos << 8) | byte];
                        }
                        x >>= 8;
                        pos += 1;
                    }
                }
            }
        });
    m
}

/// Column-lane packing `packed_cols[g][i]`: lane k of word g at row-bit i
/// holds `M[64g+k][i]` — the layout the branch-native bit-affine lazy
/// forest consumes. Built directly from the data tensor.
#[allow(clippy::arithmetic_side_effects)]
#[allow(dead_code)]
pub(crate) fn pack_columns_lanes(p: &IntegerMatrixLayout, data: &[u128]) -> Vec<Vec<u64>> {
    let log_w = p.word_bits.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    let num_groups = p.cols().div_ceil(64);
    cfg_into_iter!(0..num_groups)
        .map(|g| {
            let mut w = vec![0u64; row_len];
            for lane in 0..64usize {
                let c = (g << 6) | lane;
                if c >= p.cols() {
                    break;
                }
                for b in 0..p.rows() {
                    let cell = data[p.cell_index(b, c)];
                    for j in 0..p.word_bits {
                        if (cell >> j) & 1 == 1 {
                            w[(b << log_w) | j] |= 1u64 << lane;
                        }
                    }
                }
            }
            w
        })
        .collect()
}

/// Branch-native fast forest: the bit-affine lazy product forest over the
/// lane-packed columns (single weight set). Mirrors the mod-q prover's
/// internals with one chunk. Returns (forest proof, rho, leaf claims).
#[allow(clippy::arithmetic_side_effects)]
#[allow(dead_code)]
fn prove_fold_forest_fast(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    data: &[u128],
    row_weights: &[u128],
    alpha: Gf,
    packed_cols: Option<&[Vec<u64>]>,
) -> (ProductForestProof<Gf>, Vec<Gf>) {
    use crate::pcs::{
        build_column_layer1_halves, chunk_pow2_table, extract_column_bit_halves, leaf_tau_halves,
    };
    use crate::piop::lookup::gkr_product::{ForestLeafBits, prove_product_forest_lazy};
    let log_w = p.word_bits.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    let owned;
    let packed_cols: &[Vec<u64>] = match packed_cols {
        Some(pc) => pc,
        None => {
            owned = pack_columns_lanes(p, data);
            &owned
        }
    };
    let pow2 = chunk_pow2_table(p, row_weights, alpha);
    let one = Gf::one();
    let mask = p.cols().wrapping_sub(1);
    let pair_tbl = crate::pcs::layer1_pair_table(p, &pow2, log_w, row_len);
    let gen_layer1 = |k: usize| -> (Vec<Gf>, Vec<Gf>) {
        build_column_layer1_halves(
            p,
            packed_cols,
            k & mask,
            &pow2,
            &pair_tbl,
            one,
            log_w,
            row_len,
        )
    };
    let tau_sets = vec![leaf_tau_halves(p, &pow2, one, log_w, row_len)];
    let col_bits = extract_column_bit_halves(packed_cols, p.cols(), row_len);
    let bits_of = |k: usize| col_bits[k & mask].clone();
    let tau_set_of = |_k: usize| 0usize;
    let (forest, claims) = prove_product_forest_lazy(
        transcript,
        p.cols(),
        gen_layer1,
        ForestLeafBits {
            bits_of: &bits_of,
            tau_sets: &tau_sets,
            tau_set_of: &tau_set_of,
        },
        &(),
    );
    let rho: Vec<Gf> = claims.first().map(|(pt, _)| pt.clone()).unwrap_or_default();
    (forest, rho)
}

/// `v_c = Σ_b w_b·D[(b,c)]` computed from the packed bit rows (W-bit cells
/// reassembled per set bit) — avoids re-streaming the `u128` data tensor.
///
/// Two exact forms (u128 addition is associative and the total stays
/// `< 2^127` by the chunking bound, so reassociation is value-exact —
/// identical `us`, identical transcript):
///
/// - **nibble-LUT** (the default — S3 of `docs/forest-speedup-ideas.md`):
///   precombine each 4-bit group's weight sums ONCE
///   (`tbl[g≪4 | nib] = Σ_{i∈nib} w_{4g+i}·2^{j}`, `16·row_len/4` u128
///   entries shared by all `2^s` columns), then each column is an
///   unconditional table-add per nonzero nibble — no per-bit
///   trailing-zeros walk, no data-dependent shift;
/// - **tz-walk** (`BITZ_FOLDV_LUT=0`): the original per-set-bit scan.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn fold_values_bits(
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    row_weights: &[u128],
) -> Vec<u128> {
    fold_values_bits_width::<false>(p, rows, row_weights, p.word_bits)
}

pub(crate) fn fold_values_bits_bounded(
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    row_weights: &[u128],
    value_bits: usize,
) -> Vec<u128> {
    if value_bits == p.word_bits {
        return fold_values_bits(p, rows, row_weights);
    }
    fold_values_bits_width::<true>(p, rows, row_weights, value_bits)
}

fn fold_values_bits_width<const PADDED: bool>(
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    row_weights: &[u128],
    value_bits: usize,
) -> Vec<u128> {
    let log_w = p.word_bits.trailing_zeros() as usize;
    let mask = p.word_bits.wrapping_sub(1);
    if foldv_lut() {
        // Per-BIT weight of bit i: w_{i≫log_w}·2^{i&mask}; zero beyond
        // row_len (the packed words' padding bits are zero anyway, but
        // the table covers every word's 16 nibble groups).
        let row_len = p.rows() << log_w;
        let words = row_len.div_ceil(64);
        let groups = words << 4;
        let wbit = |i: usize| -> u128 {
            if i < row_len && (!PADDED || (i & mask) < value_bits) {
                row_weights[i >> log_w] << (i & mask)
            } else {
                0
            }
        };
        let tbl: Vec<u128> = {
            let rows_t: Vec<[u128; 16]> = cfg_into_iter!(0..groups, 1 << 12)
                .map(|g| {
                    let mut t = [0u128; 16];
                    for nib in 1..16usize {
                        // t[nib] = t[nib without its lowest bit] + that bit's weight.
                        t[nib] =
                            t[nib & (nib - 1)] + wbit((g << 2) | nib.trailing_zeros() as usize);
                    }
                    t
                })
                .collect();
            rows_t.into_flattened()
        };
        // Column blocks stream the shared table (8 MB at 2^17 rows —
        // L2-resident, not L1) once per block instead of once per column;
        // the block's accumulators are independent chains. Per-column
        // term order is unchanged (exact u128 sums either way).
        const MAX_BLOCK: usize = 32;
        let cols = p.cols();
        #[cfg(feature = "parallel")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let threads = 1;
        let block = (cols / (4 * threads)).clamp(1, MAX_BLOCK);
        let sums: Vec<[u128; MAX_BLOCK]> = cfg_into_iter!(0..cols.div_ceil(block))
            .map(|blk| {
                let c0 = blk * block;
                let n = block.min(cols - c0);
                let mut acc = [0u128; MAX_BLOCK];
                for wi in 0..words {
                    for (k, acc_k) in acc.iter_mut().enumerate().take(n) {
                        let mut w = rows[c0 + k][wi];
                        let mut g = wi << 4;
                        while w != 0 {
                            *acc_k += tbl[(g << 4) | (w & 15) as usize];
                            w >>= 4;
                            g += 1;
                        }
                    }
                }
                acc
            })
            .collect();
        return (0..cols).map(|c| sums[c / block][c % block]).collect();
    }
    cfg_into_iter!(0..p.cols())
        .map(|c| {
            let mut acc = 0u128;
            for (wi, &word) in rows[c].iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    let i = (wi << 6) | tz;
                    if !PADDED || (i & mask) < value_bits {
                        acc += row_weights[i >> log_w] << (i & mask);
                    }
                    bits &= bits.wrapping_sub(1);
                }
            }
            acc
        })
        .collect()
}

/// [`fold_values_bits`] form choice: `BITZ_FOLDV_LUT=0` opts back into the
/// per-set-bit trailing-zeros walk (diagnostic / A-B). Read once per
/// process.
fn foldv_lut() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("BITZ_FOLDV_LUT").map_or(true, |v| v != "0"))
}

/// One fused pass folding `K` weight sets at once: per column,
/// `acc[k] = Σ_b w_k[b]·D[(b,c)]`. The fixed-size array accumulator lets
/// the `K`-term inner body unroll fully, so the per-set-bit scan
/// (trailing-zeros walk, index math) is paid ONCE for all `K` sets — a
/// `Vec` accumulator measured as slow as `K` separate passes.
#[allow(clippy::arithmetic_side_effects)]
fn fold_cols_multi_k<const K: usize>(
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    sets: &[&[u128]],
) -> Vec<[u128; K]> {
    debug_assert_eq!(sets.len(), K);
    let w: [&[u128]; K] = core::array::from_fn(|k| sets[k]);
    let log_w = p.word_bits.trailing_zeros() as usize;
    let mask = p.word_bits.wrapping_sub(1);
    cfg_into_iter!(0..p.cols())
        .map(|c| {
            let mut acc = [0u128; K];
            for (wi, &word) in rows[c].iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    let i = (wi << 6) | tz;
                    let (b, sh) = (i >> log_w, i & mask);
                    for k in 0..K {
                        acc[k] += w[k][b] << sh;
                    }
                    bits &= bits.wrapping_sub(1);
                }
            }
            acc
        })
        .collect()
}

/// Multi-weight-set variant of [`fold_values_bits`]: `out[k][c] =
/// Σ_b w_k[b]·D[(b,c)]` for EVERY weight set `k`, the sets processed in
/// unrolled groups of up to 4 ([`fold_cols_multi_k`]) so the bit stream
/// and its scan are shared within a group (the extension path's Step-1
/// folds run `e·L₁` sets over the same bits). Value-exact per set: each
/// accumulator receives exactly [`fold_values_bits`]'s terms in the same
/// per-column order.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn fold_values_bits_multi(
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    weight_sets: &[&[u128]],
) -> Vec<Vec<u128>> {
    let mut out = Vec::with_capacity(weight_sets.len());
    let mut i = 0usize;
    while i < weight_sets.len() {
        let take = (weight_sets.len() - i).min(4);
        let group = &weight_sets[i..i + take];
        match take {
            4 => transpose_fold_group::<4>(fold_cols_multi_k::<4>(p, rows, group), &mut out),
            3 => transpose_fold_group::<3>(fold_cols_multi_k::<3>(p, rows, group), &mut out),
            2 => transpose_fold_group::<2>(fold_cols_multi_k::<2>(p, rows, group), &mut out),
            _ => out.push(fold_values_bits(p, rows, group[0])),
        }
        i += take;
    }
    out
}

/// Split a fused group's per-column `[u128; K]` accumulators into `K`
/// per-set fold vectors, appended to `out` in set order.
fn transpose_fold_group<const K: usize>(cols: Vec<[u128; K]>, out: &mut Vec<Vec<u128>>) {
    for k in 0..K {
        out.push(cols.iter().map(|a| a[k]).collect());
    }
}

/// Classic 64×64 bit-matrix transpose (6 mask/shift rounds): output word
/// `t`'s bit `k` = input word `k`'s bit `t`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn transpose_64x64(a: &mut [u64; 64]) {
    let mut j = 32usize;
    let mut m = 0x0000_0000_FFFF_FFFFu64;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            let idx = k + j;
            let x = (a[k] ^ (a[idx] << j)) & !m;
            a[k] ^= x;
            a[idx] ^= x >> j;
            let knext = (k + j + 1) & !j;
            k = if (k + 1) & j != 0 { knext } else { k + 1 };
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// Derive the column-lane packing from the row-major packed rows: block
/// (g, wi) of `packed_cols` is the bit-transpose of rows `64g..64g+64` at
/// word `wi`. One pass over the 8-byte-per-64-bits rows instead of a second
/// sweep of the u128 data tensor.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn pack_columns_from_rows(p: &IntegerMatrixLayout, rows: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let log_w = p.word_bits.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    let words = row_len.div_ceil(64);
    let num_groups = p.cols().div_ceil(64);
    cfg_into_iter!(0..num_groups)
        .map(|g| {
            let mut out = vec![0u64; row_len];
            for wi in 0..words {
                let mut blk = [0u64; 64];
                for k in 0..64usize {
                    let c = (g << 6) | k;
                    if c < p.cols() {
                        blk[k] = rows[c][wi];
                    }
                }
                transpose_64x64(&mut blk);
                // blk[t] now holds, per lane k, bit (64·wi + t) of row 64g+k
                // — wait: transpose maps bit t of blk[k] to bit k of blk[t].
                let base = wi << 6;
                for (t, &v) in blk.iter().enumerate() {
                    if base + t < row_len {
                        out[base + t] = v;
                    }
                }
            }
            out
        })
        .collect()
}

/// Inverse of [`pack_columns_from_rows`]: rebuild the per-column bit rows
/// from the 64-column-lane packed store (`packed_cols[g][i]` bit `k` =
/// column `64g+k`'s bit `i` — the layout `sha_f2_packed_cols` produces).
/// Parallel over groups.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn rows_from_packed_cols(
    p: &IntegerMatrixLayout,
    packed_cols: &[Vec<u64>],
) -> Vec<Vec<u64>> {
    let log_w = p.word_bits.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    let words = row_len.div_ceil(64);
    let num_groups = p.cols().div_ceil(64);
    debug_assert_eq!(packed_cols.len(), num_groups);
    let groups: Vec<Vec<Vec<u64>>> = cfg_into_iter!(0..num_groups)
        .map(|g| {
            let lanes = 64.min(p.cols() - (g << 6));
            let mut rows_g: Vec<Vec<u64>> = vec![vec![0u64; words]; lanes];
            for wi in 0..words {
                let mut blk = [0u64; 64];
                for (t, b) in blk.iter_mut().enumerate() {
                    let i = (wi << 6) | t;
                    *b = if i < row_len { packed_cols[g][i] } else { 0 };
                }
                transpose_64x64(&mut blk);
                for (k, row) in rows_g.iter_mut().enumerate() {
                    row[wi] = blk[k];
                }
            }
            rows_g
        })
        .collect();
    groups.into_iter().flatten().collect()
}

/// The backend-independent prover prefix: forest GKR, the `v` message, the
/// eq(ξ) batching, and the de-black-boxing pre-sumcheck. Returns the pieces
/// plus the residual claim point `(r*, ξ)` whose bit-MLE evaluation the
/// opening backend must prove.
#[allow(clippy::arithmetic_side_effects)]
#[allow(dead_code)]
pub(crate) fn prove_int_eval_common(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    data: &[u128],
    row_weights: &[u128],
    alpha: Gf,
    rows: &[Vec<u64>],
    packed_cols: Option<&[Vec<u64>]>,
) -> (
    ProductForestProof<Gf>,
    Vec<u128>,
    MultiDegreeSumcheckProof<Gf>,
    Vec<Gf>,
) {
    let (forest, rho) =
        prove_fold_forest_fast(transcript, p, data, row_weights, alpha, packed_cols);
    let v = fold_values_bits(p, rows, row_weights);

    // eq(c, ξ) batching of the per-column leaf claims.
    let xi: Vec<Gf> = transcript.get_field_challenges(p.col_vars, &());
    let eq_xi = if xi.is_empty() {
        vec![Gf::one()]
    } else {
        build_eq_x_r_vec(&xi, &()).expect("nonempty column point")
    };

    // Pre-sumcheck tables: R = q_rowbit(ρ), m_ξ = eq(ξ)-combined rows.
    let t_w = row_bit_vars(p);
    let r_tbl = row_bit_weights(p, row_weights, alpha, &rho);
    let m_tbl = xi_combined_rows(p, rows, &eq_xi);

    let (presum, r_star) = {
        let (values, weights) = crate::sumcheck::inner::binary::inputs(vec![[r_tbl, m_tbl]], t_w);
        crate::sumcheck::inner::binary::encode(
            crate::sumcheck::inner::prove_batched_inner_sumcheck(
                &field::Gf128Ops,
                transcript,
                crate::sumcheck::inner::InitialClaims::Compute,
                values,
                weights,
                &mut crate::sumcheck::UngrindedRoundBoundary,
            )
            .expect("valid post-GKR dot products"),
        )
    };

    // Residual claim point: M̂(r*, ξ) = μ.
    let point: Vec<Gf> = r_star.iter().chain(xi.iter()).copied().collect();
    (forest, v, presum, point)
}

/// Verify an integer-MLE evaluation with the RS/BaseFold opening. Mirrors
/// [`crate::pcs::verify`] stages (1), (2), (4); stage (3) is the
/// pre-sumcheck + ring-switch + BaseFold chain.
/// The backend-independent verifier prefix — stages (1) forest, (2) integer
/// binding, (3a) eq(ξ) batching + pre-sumcheck, (3b) `R̂(r*)` and `μ`.
/// Returns the residual claim `(point, μ)` for the opening backend; the
/// caller finishes with its opener and the read-off.
#[allow(clippy::arithmetic_side_effects)]
#[allow(dead_code)]
pub(crate) fn verify_int_eval_common(
    transcript: &mut impl Transcript,
    forest: &ProductForestProof<Gf>,
    v: &[u128],
    presum: &MultiDegreeSumcheckProof<Gf>,
    p: &IntegerMatrixLayout,
    row_weights: &[u128],
    alpha: Gf,
) -> Result<(Vec<Gf>, Gf), IntEvalRsError> {
    // (1) Forest → shared ρ + leaf claims.
    let t_w = row_bit_vars(p);
    let depths = vec![t_w; p.cols()];
    let vclaims = verify_product_forest(transcript, forest, &depths, &())
        .map_err(|_| IntEvalRsError::Forest)?;
    if v.len() != p.cols() || forest.roots.len() != p.cols() {
        return Err(IntEvalRsError::Forest);
    }
    let rho: Vec<Gf> = vclaims
        .first()
        .map(|(pt, _)| pt.clone())
        .unwrap_or_default();
    let leaf_evals: Vec<Gf> = vclaims.iter().map(|(_, e)| *e).collect();

    // (2) Bind the sent integers.
    if !is_generator(alpha) {
        return Err(IntEvalRsError::ChallengeNotGenerator);
    }
    let max = max_fold_magnitude(v);
    // ord(α) = 2^128 − 1 = u128::MAX, so `≥` collapses to `==`; keep the
    // protocol-shaped bound `max < ord(α)` as in `f2_int_eval::verify`.
    #[allow(clippy::absurd_extreme_comparisons)]
    if max >= GF128_MULT_ORDER {
        return Err(IntEvalRsError::Magnitude { max });
    }
    for (c, &vc) in v.iter().enumerate() {
        if gf_pow(alpha, vc) != forest.roots[c] {
            return Err(IntEvalRsError::RootBinding { c });
        }
    }

    // (3a) eq(ξ) batching + pre-sumcheck.
    let xi: Vec<Gf> = transcript.get_field_challenges(p.col_vars, &());
    let eq_xi = if xi.is_empty() {
        vec![Gf::one()]
    } else {
        build_eq_x_r_vec(&xi, &()).expect("nonempty column point")
    };
    let one = Gf::one();
    let y_xi = leaf_evals
        .iter()
        .zip(eq_xi.iter())
        .fold(Gf::zero(), |acc, (l, e)| acc + *e * (*l - one));
    let subclaims = presum
        .verify_as_subprotocol(transcript, t_w, &[2], &())
        .map_err(|_| IntEvalRsError::PreSumcheck)?;
    if presum.claimed_sums() != [y_xi] {
        return Err(IntEvalRsError::PreSumcheck);
    }
    let r_star = subclaims.point().to_vec();
    let expected = subclaims.expected_evaluations()[0];

    // (3b) The verifier's O(2^t·W) step: R̂(r*), then μ = expected / R̂(r*).
    let r_tbl = row_bit_weights(p, row_weights, alpha, &rho);
    let eq_rstar = build_eq_x_r_vec(&r_star, &()).expect("t_w >= 1");
    let r_hat = r_tbl
        .iter()
        .zip(eq_rstar.iter())
        .fold(Gf::zero(), |acc, (q, e)| acc + *q * *e);
    if r_hat.is_zero() {
        return Err(IntEvalRsError::RHatZero);
    }
    let mu = expected * r_hat.invert_nonzero();

    let point: Vec<Gf> = r_star.iter().chain(xi.iter()).copied().collect();
    Ok((point, mu))
}

/// The merged-forest analogue of [`prove_int_eval_common`]: one lazy
/// bit-affine merged forest over all `2^s` trees (payload O(Σ(s+k))
/// instead of `2·2^s·d` K-elements), whose exit point `(z_bj, z_c)`
/// REPLACES `(ρ, ξ)` — the eq(ξ) batching step dissolves into the forest.
/// Returns `(merged proof, v, pre-sumcheck, residual claim point)`. The
/// roots are NOT returned: they are `α^{v_c}` by construction, so the
/// verifier recomputes them from the sent folds (they never ride the
/// proof).
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::type_complexity)]
/// Degree-2 two-MLE product round evaluator for the pre-sumcheck's `R·m`
/// group, accumulating the three round-polynomial evaluations with
/// deferred-reduction ([`crate::utils::wide_mul::WideMulAcc`]) accumulators —
/// one reduction per accumulator per chunk instead of per product. Value-exact
/// vs the generic per-point gather (identical per-slot products at the nodes
/// `{0, 1, F::from(2)}`; reduction is `F₂`-linear and the outer sums are
/// exact), so the emitted proof is byte-identical. Attached under
/// [`rs_fast`] purely so `BITZ_RS_FAST=0` restores the generic path for A/B.
#[cfg(test)]
pub(crate) struct ProdPairWideEvaluator;

#[cfg(test)]
impl crate::piop::sumcheck::prover::RoundPolyEvaluator<Gf> for ProdPairWideEvaluator {
    #[allow(clippy::arithmetic_side_effects)]
    fn round_evals(
        &self,
        mles: &[DenseMultilinearExtension<
            <Gf as crate::poly::coefficient::FieldRepresentation>::Inner,
        >],
        round: usize,
        num_vars: usize,
        degree: usize,
        _config: &(),
    ) -> Vec<Gf> {
        use crate::poly::coefficient::PolynomialField;
        use crate::utils::inner_transparent_field::InnerTransparentField;
        use crate::utils::wide_mul::WideMulAcc;

        assert_eq!(degree, 2, "product-pair evaluator is degree-2 only");
        assert_eq!(
            mles.len(),
            2,
            "product-pair evaluator expects exactly 2 MLEs"
        );
        let half = 1usize << (num_vars - round);
        let x2 = Gf::interpolation_node(2, &());
        const CHUNK: usize = 1 << 12;
        let n_chunks = half.div_ceil(CHUNK).max(1);
        let partials: Vec<(Gf, Gf, Gf)> = cfg_into_iter!(0..n_chunks)
            .map(|c| {
                let a = &mles[0];
                let bm = &mles[1];
                let lo = c * CHUNK;
                let hi = (lo + CHUNK).min(half);
                let zero = Gf::zero();
                let mut e0 = <Gf as WideMulAcc>::wide_zero(&zero);
                let mut e1 = <Gf as WideMulAcc>::wide_zero(&zero);
                let mut e2 = <Gf as WideMulAcc>::wide_zero(&zero);
                for j in lo..hi {
                    let a0 = Gf::new_unchecked_with_cfg(a[2 * j], &());
                    let a1 = Gf::new_unchecked_with_cfg(a[2 * j + 1], &());
                    let b0 = Gf::new_unchecked_with_cfg(bm[2 * j], &());
                    let b1 = Gf::new_unchecked_with_cfg(bm[2 * j + 1], &());
                    <Gf as WideMulAcc>::wide_add_assign(
                        &mut e0,
                        &<Gf as WideMulAcc>::mul_wide(&a0, &b0),
                    );
                    <Gf as WideMulAcc>::wide_add_assign(
                        &mut e1,
                        &<Gf as WideMulAcc>::mul_wide(&a1, &b1),
                    );
                    // Node F::from(2): M_i(2) = M_i(0) + from(2)·(M_i(1) − M_i(0)),
                    // exactly the generic extrapolation (mul_by_node2 yields the
                    // identical field element).
                    let va = a0 + (a1 - a0).mul_by_node2(&x2);
                    let vb = b0 + (b1 - b0).mul_by_node2(&x2);
                    <Gf as WideMulAcc>::wide_add_assign(
                        &mut e2,
                        &<Gf as WideMulAcc>::mul_wide(&va, &vb),
                    );
                }
                (
                    <Gf as WideMulAcc>::from_wide(e0),
                    <Gf as WideMulAcc>::from_wide(e1),
                    <Gf as WideMulAcc>::from_wide(e2),
                )
            })
            .collect();
        let mut evals = vec![Gf::zero(); 3];
        for (p0, p1, p2) in &partials {
            evals[0] += *p0;
            evals[1] += *p1;
            evals[2] += *p2;
        }
        evals
    }
}

pub(crate) fn prove_int_eval_merged_common(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    packed_cols: Option<&[Vec<u64>]>,
    row_weights: &[u128],
    alpha: Gf,
) -> (
    crate::merged_forest::MergedForestProof,
    Vec<u128>,
    MultiDegreeSumcheckProof<Gf>,
    Vec<Gf>,
) {
    prove_int_eval_merged_bounded(
        transcript,
        p,
        rows,
        packed_cols,
        row_weights,
        alpha,
        p.word_bits,
    )
}

pub(crate) fn prove_int_eval_merged_bounded(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    packed_cols: Option<&[Vec<u64>]>,
    row_weights: &[u128],
    alpha: Gf,
    value_bits: usize,
) -> (
    crate::merged_forest::MergedForestProof,
    Vec<u128>,
    MultiDegreeSumcheckProof<Gf>,
    Vec<Gf>,
) {
    use crate::merged_forest::prove_merged_forest_lazy_from_rows;
    use crate::pcs::chunk_pow2_table;
    let t_w = row_bit_vars(p);
    let owned;
    let packed_cols: &[Vec<u64>] = match packed_cols {
        Some(pc) => pc,
        None => {
            let _g = tracing::info_span!("mc:pack").entered();
            owned = pack_columns_from_rows(p, rows);
            &owned
        }
    };
    let pow2 = {
        let _g = tracing::info_span!("mc:pow2").entered();
        chunk_pow2_table(p, row_weights, alpha)
    };
    let (_roots, mf, z, _e_d) = {
        let _g = tracing::info_span!("mc:forest").entered();
        if crate::merged_forest::quad_active(p) {
            crate::merged_forest::prove_merged_forest_lazy_quad_from_rows(
                transcript,
                p,
                Some(rows),
                packed_cols,
                &pow2,
            )
        } else {
            // A zero-padded witness ends in all-zero columns; those trees
            // are constant 1 and never get built (byte-identical proof).
            let live = {
                let _g = tracing::info_span!("mc:live_cols").entered();
                crate::merged_forest::live_cols(p, rows)
            };
            prove_merged_forest_lazy_from_rows(transcript, p, rows, packed_cols, &pow2, live)
        }
    };
    drop(pow2);
    let v = {
        let _g = tracing::info_span!("mc:fold_v").entered();
        fold_values_bits_bounded(p, rows, row_weights, value_bits)
    };

    let _g_tbls = tracing::info_span!("mc:presum_tbls").entered();
    let (z_bj, z_c) = z.split_at(t_w);
    let eq_zc = if z_c.is_empty() {
        vec![Gf::one()]
    } else {
        build_eq_x_r_vec(z_c, &()).expect("nonempty column point")
    };
    let r_tbl = row_bit_weights(p, row_weights, alpha, z_bj);
    let m_tbl = if rs_fast() {
        xi_combined_rows_packed(p, packed_cols, &eq_zc)
    } else {
        xi_combined_rows(p, rows, &eq_zc)
    };
    drop(_g_tbls);

    let (presum, r_star) = {
        let _g = tracing::info_span!("mc:presum_run").entered();
        {
            let (values, weights) =
                crate::sumcheck::inner::binary::inputs(vec![[r_tbl, m_tbl]], t_w);
            crate::sumcheck::inner::binary::encode(
                crate::sumcheck::inner::prove_batched_inner_sumcheck(
                    &field::Gf128Ops,
                    transcript,
                    crate::sumcheck::inner::InitialClaims::Compute,
                    values,
                    weights,
                    &mut crate::sumcheck::UngrindedRoundBoundary,
                )
                .expect("valid post-GKR dot products"),
            )
        }
    };

    // Residual claim point: M̂(r*, z_c) = μ.
    let point: Vec<Gf> = r_star.iter().chain(z_c.iter()).copied().collect();
    (mf, v, presum, point)
}

/// Batched multi-claim analogue of [`prove_int_eval_merged_common`] for
/// same-shape x-claims: ALL claims run as ONE merged forest
/// ([`prove_merged_forest_lazy_multi`][crate::merged_forest::prove_merged_forest_lazy_multi],
/// tree `(n ≪ s) | c`, padded to a power of two with zero-weight dummies
/// whose leaves are identically 1) and ONE N-group pre-sumcheck under
/// shared challenges — sharing every layer's round messages and the
/// per-layer driver floor. All claims exit at ONE shared residual point.
/// Returns `(merged proof, per-claim folds, presum, shared point)`.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::type_complexity)]
pub(crate) fn prove_x_claims_batched_common(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    claim_rows: &[&[Vec<u64>]],
    claim_weights: &[&[u128]],
    alpha: Gf,
) -> Result<
    (
        crate::merged_forest::MergedForestProof,
        Vec<Vec<u128>>,
        MultiDegreeSumcheckProof<Gf>,
        Vec<Gf>,
    ),
    crate::merged_forest::schedule::UnsupportedSchedule,
> {
    use crate::merged_forest::prove_merged_forest_lazy_multi;
    use crate::pcs::chunk_pow2_table;
    let n_real = claim_rows.len();
    assert!(n_real >= 1 && n_real == claim_weights.len(), "claim shape");
    let pad_n = n_real.next_power_of_two();
    let t_w = row_bit_vars(p);

    // Weight-set dedup: same-point claims share their α-power tables (and,
    // downstream, the merged forest's τ tables and the presum's q_rowbit).
    let mut w_reps: Vec<usize> = Vec::new();
    let w_of: Vec<usize> = claim_weights
        .iter()
        .enumerate()
        .map(
            |(n, w)| match w_reps.iter().position(|&r| claim_weights[r] == *w) {
                Some(u) => u,
                None => {
                    w_reps.push(n);
                    w_reps.len() - 1
                }
            },
        )
        .collect();

    let packed: Vec<Vec<Vec<u64>>> = {
        let _g = tracing::info_span!("mc:pack").entered();
        claim_rows
            .iter()
            .map(|rows| pack_columns_from_rows(p, rows))
            .collect()
    };
    let pow2s: Vec<Vec<Vec<Gf>>> = {
        let _g = tracing::info_span!("mc:pow2").entered();
        w_reps
            .iter()
            .map(|&r| chunk_pow2_table(p, claim_weights[r], alpha))
            .collect()
    };
    // Dummy padding: zero-weight tau chains (all 1 => leaves identically
    // 1); the bits are irrelevant — reuse claim 0's store.
    let ones_pow2: Vec<Vec<Gf>> = vec![vec![Gf::one(); p.word_bits]; p.rows()];
    let mut pairs: Vec<(&[Vec<u64>], &[Vec<Gf>])> = Vec::with_capacity(pad_n);
    for n in 0..pad_n {
        if n < n_real {
            pairs.push((&packed[n], &pow2s[w_of[n]]));
        } else {
            pairs.push((&packed[0], &ones_pow2));
        }
    }
    let (_roots, mf, z, _e_d) = {
        let _g = tracing::info_span!("mc:forest").entered();
        prove_merged_forest_lazy_multi(transcript, p, &pairs)?
    };
    let us: Vec<Vec<u128>> = {
        let _g = tracing::info_span!("mc:fold_v").entered();
        claim_rows
            .iter()
            .zip(claim_weights.iter())
            .map(|(rows, w)| fold_values_bits(p, rows, w))
            .collect()
    };

    let _g_tbls = tracing::info_span!("mc:presum_tbls").entered();
    let (z_bj, z_cn) = z.split_at(t_w);
    let (z_clear, z_claim) = z_cn.split_at(p.col_vars);
    let eq_clear = if z_clear.is_empty() {
        vec![Gf::one()]
    } else {
        build_eq_x_r_vec(z_clear, &()).expect("nonempty column point")
    };
    let eq_claim = if z_claim.is_empty() {
        vec![Gf::one()]
    } else {
        build_eq_x_r_vec(z_claim, &()).expect("log N >= 1")
    };
    // One q_rowbit table per UNIQUE weight set; per-claim groups take a
    // scaled copy.
    let r_tbls: Vec<Vec<Gf>> = w_reps
        .iter()
        .map(|&r| row_bit_weights(p, claim_weights[r], alpha, z_bj))
        .collect();
    let groups: Vec<[Vec<Gf>; 2]> = (0..n_real)
        .map(|n| {
            let mut r_tbl = r_tbls[w_of[n]].clone();
            let scale = eq_claim[n];
            for e in r_tbl.iter_mut() {
                *e = *e * scale;
            }
            let m_tbl = xi_combined_rows(p, claim_rows[n], &eq_clear);
            [r_tbl, m_tbl]
        })
        .collect();
    drop(_g_tbls);
    let (presum, r_star) = {
        let _g = tracing::info_span!("mc:presum_run").entered();
        {
            let (values, weights) = crate::sumcheck::inner::binary::inputs(groups, t_w);
            crate::sumcheck::inner::binary::encode(
                crate::sumcheck::inner::prove_batched_inner_sumcheck(
                    &field::Gf128Ops,
                    transcript,
                    crate::sumcheck::inner::InitialClaims::Compute,
                    values,
                    weights,
                    &mut crate::sumcheck::UngrindedRoundBoundary,
                )
                .expect("valid post-GKR dot products"),
            )
        }
    };

    // Shared residual point: every claim's M-hat_n(r*, z_clear) = mu_n.
    let point: Vec<Gf> = r_star.iter().chain(z_clear.iter()).copied().collect();
    Ok((mf, us, presum, point))
}

/// Verifier of [`prove_x_claims_batched_common`]: recompute the `N·2^s`
/// roots from the per-claim folds (dummy trees are 1), verify the ONE
/// merged forest and the ONE N-group pre-sumcheck (`Σ_n σ_n = e_d + 1`),
/// and extract the per-claim residuals
/// `μ_n = expected_n / (eq(n, z_claim)·R-hat_n(r*))` at the SHARED point.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn verify_x_claims_batched_common(
    transcript: &mut impl Transcript,
    mf: &crate::merged_forest::MergedForestProof,
    claim_us: &[&[u128]],
    presum: &MultiDegreeSumcheckProof<Gf>,
    p: &IntegerMatrixLayout,
    claim_weights: &[&[u128]],
    alpha: Gf,
) -> Result<(Vec<Gf>, Vec<Gf>), IntEvalRsError> {
    use crate::merged_forest::verify_merged_forest;
    let n_real = claim_us.len();
    if n_real == 0 || n_real != claim_weights.len() {
        return Err(IntEvalRsError::Forest);
    }
    let pad_n = n_real.next_power_of_two();
    let log_n = pad_n.trailing_zeros() as usize;
    let t_w = row_bit_vars(p);
    let one = Gf::one();

    if !is_generator(alpha) {
        return Err(IntEvalRsError::ChallengeNotGenerator);
    }
    for us in claim_us {
        if us.len() != p.cols() {
            return Err(IntEvalRsError::Forest);
        }
        let max = max_fold_magnitude(us);
        #[allow(clippy::absurd_extreme_comparisons)]
        if max >= GF128_MULT_ORDER {
            return Err(IntEvalRsError::Magnitude { max });
        }
    }
    let comb = field::FixedBasePow::<_, 2>::new_public(field::Gf128Ops, alpha.into(), 8);
    let mut roots = Vec::with_capacity(pad_n << p.col_vars);
    for n in 0..pad_n {
        if n < n_real {
            roots.extend(claim_us[n].iter().map(|&u| {
                Gf::from(comb.pow_public(&field::Uint::from_words([u as u64, (u >> 64) as u64])))
            }));
        } else {
            roots.extend(core::iter::repeat(one).take(p.cols()));
        }
    }
    let (z, e_d) =
        verify_merged_forest(transcript, &roots, mf, t_w, p.col_vars.wrapping_add(log_n))
            .map_err(|_| IntEvalRsError::Forest)?;

    let subclaims = presum
        .verify_as_subprotocol(transcript, t_w, &vec![2; n_real], &())
        .map_err(|_| IntEvalRsError::PreSumcheck)?;
    let sums = presum.claimed_sums();
    if sums.len() != n_real {
        return Err(IntEvalRsError::PreSumcheck);
    }
    let total = sums.iter().fold(Gf::zero(), |a, &b| a + b);
    if total != e_d + one {
        return Err(IntEvalRsError::PreSumcheck);
    }

    let (z_bj, z_cn) = z.split_at(t_w);
    let (z_clear, z_claim) = z_cn.split_at(p.col_vars);
    let eq_claim = if z_claim.is_empty() {
        vec![one]
    } else {
        build_eq_x_r_vec(z_claim, &()).expect("log N >= 1")
    };
    let r_star = subclaims.point().to_vec();
    let eq_rstar = build_eq_x_r_vec(&r_star, &()).expect("t_w >= 1");
    // The O(2^{t'}) q_rowbit tables: one per UNIQUE weight set (same-point
    // claims share theirs), in parallel across the unique sets.
    let mut w_reps: Vec<usize> = Vec::new();
    let w_of: Vec<usize> = claim_weights
        .iter()
        .enumerate()
        .map(
            |(n, w)| match w_reps.iter().position(|&r| claim_weights[r] == *w) {
                Some(u) => u,
                None => {
                    w_reps.push(n);
                    w_reps.len() - 1
                }
            },
        )
        .collect();
    let r_hats: Vec<Gf> = cfg_into_iter!(w_reps)
        .map(|r| {
            let r_tbl = row_bit_weights(p, claim_weights[r], alpha, z_bj);
            r_tbl
                .iter()
                .zip(eq_rstar.iter())
                .fold(Gf::zero(), |acc, (q, e)| acc + *q * *e)
        })
        .collect();
    let mut mus = Vec::with_capacity(n_real);
    for (n, expected) in subclaims.expected_evaluations().iter().enumerate() {
        let denom = eq_claim[n] * r_hats[w_of[n]];
        if denom.is_zero() {
            return Err(IntEvalRsError::RHatZero);
        }
        mus.push(*expected * denom.invert_nonzero());
    }

    let point: Vec<Gf> = r_star.iter().chain(z_clear.iter()).copied().collect();
    Ok((point, mus))
}

/// The merged-forest analogue of [`verify_int_eval_common`]: RECOMPUTE the
/// roots `α^{v_c}` from the sent integers (they never ride the proof — the
/// Fiat–Shamir absorb of the recomputed roots IS the binding: a prover
/// whose forest ran against different roots diverges the transcript and
/// fails the layer checks), verify the merged forest, check the
/// pre-sumcheck against the forest exit claim `e_d − 1`, and return the
/// residual claim `(point, μ)` for the opening backend.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn verify_int_eval_merged_common(
    transcript: &mut impl Transcript,
    mf: &crate::merged_forest::MergedForestProof,
    v: &[u128],
    presum: &MultiDegreeSumcheckProof<Gf>,
    p: &IntegerMatrixLayout,
    row_weights: &[u128],
    alpha: Gf,
) -> Result<(Vec<Gf>, Gf), IntEvalRsError> {
    use crate::merged_forest::verify_merged_forest;
    let t_w = row_bit_vars(p);
    if v.len() != p.cols() {
        return Err(IntEvalRsError::Forest);
    }

    // (1) Bind the sent integers: range first (injectivity of the exponent
    // map), then roots := α^{v_c} by construction.
    let _g_roots = tracing::info_span!("mv:roots").entered();
    if !is_generator(alpha) {
        return Err(IntEvalRsError::ChallengeNotGenerator);
    }
    let max = max_fold_magnitude(v);
    // ord(α) = 2^128 − 1 = u128::MAX, so `≥` collapses to `==`; keep the
    // protocol-shaped bound `max < ord(α)` as in `f2_int_eval::verify`.
    #[allow(clippy::absurd_extreme_comparisons)]
    if max >= GF128_MULT_ORDER {
        return Err(IntEvalRsError::Magnitude { max });
    }
    let comb = field::FixedBasePow::<_, 2>::new_public(field::Gf128Ops, alpha.into(), 8);
    let roots: Vec<Gf> = v
        .iter()
        .map(|&vc| {
            Gf::from(comb.pow_public(&field::Uint::from_words([vc as u64, (vc >> 64) as u64])))
        })
        .collect();
    drop(_g_roots);

    // (2) Merged forest against the recomputed roots → exit point
    // (z_bj, z_c) + exit eval e_d.
    let _g_forest = tracing::info_span!("mv:forest").entered();
    let (z, e_d) = if crate::merged_forest::quad_active(p) {
        crate::merged_forest::verify_merged_forest_quad(transcript, &roots, mf, t_w, p.col_vars)
            .map_err(|_| IntEvalRsError::Forest)?
    } else {
        verify_merged_forest(transcript, &roots, mf, t_w, p.col_vars)
            .map_err(|_| IntEvalRsError::Forest)?
    };
    drop(_g_forest);

    // (3a) Pre-sumcheck against the forest exit claim: for the bit-affine
    // leaves `1 + M·(A−1)`, `Σ eq·M·A = e_d − 1` (`= e_d + 1` in char 2).
    let _g_presum = tracing::info_span!("mv:presum").entered();
    let subclaims = presum
        .verify_as_subprotocol(transcript, t_w, &[2], &())
        .map_err(|_| IntEvalRsError::PreSumcheck)?;
    let one = Gf::one();
    if presum.claimed_sums() != [e_d + one] {
        return Err(IntEvalRsError::PreSumcheck);
    }
    drop(_g_presum);
    let (z_bj, z_c) = z.split_at(t_w);
    let r_star = subclaims.point().to_vec();
    let expected = subclaims.expected_evaluations()[0];

    // (3b) The verifier's O(2^t·W) step: R̂(r*), then μ = expected / R̂(r*).
    let _g_rhat = tracing::info_span!("mv:rhat").entered();
    let r_tbl = row_bit_weights(p, row_weights, alpha, z_bj);
    let eq_rstar = build_eq_x_r_vec(&r_star, &()).expect("t_w >= 1");
    let r_hat = r_tbl
        .iter()
        .zip(eq_rstar.iter())
        .fold(Gf::zero(), |acc, (q, e)| acc + *q * *e);
    if r_hat.is_zero() {
        return Err(IntEvalRsError::RHatZero);
    }
    let mu = expected * r_hat.invert_nonzero();
    drop(_g_rhat);

    let point: Vec<Gf> = r_star.iter().chain(z_c.iter()).copied().collect();
    Ok((point, mu))
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn sample(seed: u64) -> Gf {
        let hi = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0x1234_5678_9ABC_DEF0;
        Gf::from_polynomial_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    /// `rows_from_packed_cols` inverts `pack_columns_from_rows` exactly
    /// (incl. non-multiple-of-64 column counts).
    #[test]
    fn packed_cols_row_roundtrip() {
        for (t, s, w) in [(7usize, 3usize, 1usize), (8, 6, 1), (7, 4, 2)] {
            let p = IntegerMatrixLayout {
                row_vars: t,
                col_vars: s,
                word_bits: w,
            };
            let log_w = w.trailing_zeros() as usize;
            let row_len = p.rows() << log_w;
            let words = row_len.div_ceil(64);
            let rows: Vec<Vec<u64>> = (0..p.cols())
                .map(|c| {
                    (0..words)
                        .map(|wd| {
                            (c as u64 + 3)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                .wrapping_add(wd as u64)
                                .rotate_left((c + wd) as u32 & 63)
                        })
                        .collect()
                })
                .collect();
            let packed = pack_columns_from_rows(&p, &rows);
            let back = rows_from_packed_cols(&p, &packed);
            assert_eq!(rows, back, "(t={t},s={s},W={w})");
        }
    }

    #[test]
    fn tensor_eval_matches_bruteforce_and_table() {
        let m_p = 5usize;
        let chals: Vec<Gf> = (0..m_p).map(|i| sample(0x11 + i as u64)).collect();
        let r_hi: Vec<Gf> = (0..m_p).map(|i| sample(0x22 + i as u64)).collect();
        let r2: Vec<Gf> = (0..LOG_PACKING).map(|i| sample(0x33 + i as u64)).collect();
        let eq_r2 = build_eq_x_r_vec(&r2, &()).expect("r2");
        let eq_hi = build_eq_x_r_vec(&r_hi, &()).expect("hi");

        // Brute force: Σ_y eq(y, chals)·Φ(eq_hi[y]).
        let eq_ch = build_eq_x_r_vec(&chals, &()).expect("ch");
        let mut brute = Gf::zero();
        for y in 0..(1usize << m_p) {
            let w = eq_hi[y].as_words();
            let mut phi = Gf::zero();
            for u in 0..128usize {
                if (w[u >> 6] >> (u & 63)) & 1 == 1 {
                    phi += eq_r2[u];
                }
            }
            brute += eq_ch[y] * phi;
        }
        let fast = tensor_eq_phi_eval(&chals, &r_hi, &eq_r2);
        assert_eq!(fast, brute, "tensor eval ≠ brute force");

        // And equals the MLE of the prover-side B table at chals.
        let b_tbl: Vec<Gf> = eq_hi
            .iter()
            .map(|ev| {
                let w = ev.as_words();
                let mut acc = Gf::zero();
                for u in 0..128usize {
                    if (w[u >> 6] >> (u & 63)) & 1 == 1 {
                        acc += eq_r2[u];
                    }
                }
                acc
            })
            .collect();
        assert_eq!(mle_eval(&b_tbl, &chals), fast, "B-table MLE ≠ tensor eval");
    }

    /// The Ligerito residual-block hook agrees with per-point tensor evals
    /// at every boolean tail (tail bit j ↔ coordinate prefix.len()+j).
    #[test]
    fn residual_b_evals_matches_pointwise() {
        let (pre_len, yr) = (4usize, 3usize);
        let m_p = pre_len + yr;
        let prefix: Vec<Gf> = (0..pre_len).map(|i| sample(0x51 + i as u64)).collect();
        let r_hi: Vec<Gf> = (0..m_p).map(|i| sample(0x61 + i as u64)).collect();
        let r2: Vec<Gf> = (0..LOG_PACKING).map(|i| sample(0x71 + i as u64)).collect();
        let eq_r2 = build_eq_x_r_vec(&r2, &()).expect("r2");

        let block = residual_b_evals(&prefix, yr, &r_hi, &eq_r2);
        assert_eq!(block.len(), 1usize << yr);
        for y in 0..(1usize << yr) {
            let mut point = prefix.clone();
            for j in 0..yr {
                point.push(if (y >> j) & 1 == 1 {
                    Gf::one()
                } else {
                    Gf::zero()
                });
            }
            assert_eq!(
                block[y],
                tensor_eq_phi_eval(&point, &r_hi, &eq_r2),
                "residual block mismatch at y={y}"
            );
        }
    }

    /// The method-of-four-Russians `s_v` fold equals the scalar bit scan
    /// exactly, at 8-aligned and ragged lengths (incl. the sub-block tail).
    #[test]
    fn sv_fold_mfr_matches_scalar() {
        for &len in &[1usize, 7, 8, 9, 37, 64, 300, 4096, 4104] {
            let wit: Vec<Gf> = (0..len).map(|i| sample(0x3000 + i as u64)).collect();
            let eq: Vec<Gf> = (0..len).map(|i| sample(0x5000 + i as u64)).collect();
            let mut expect = vec![Gf::zero(); 128];
            for y in 0..len {
                sv_scalar_accum(&mut expect, *wit[y].as_words(), eq[y]);
            }
            assert_eq!(sv_fold_mfr(&wit, &eq), expect, "len {len}");
        }
    }

    #[test]
    fn phi_byte_tables_into_reuses_buffer() {
        let mut tables = vec![sample(1); 16 * 256];
        for seed in [0xA000, 0xB000] {
            let coefficients: Vec<_> = (0..128).map(|i| sample(seed + i)).collect();
            phi_byte_tables_into(&mut tables, |i| coefficients[i]);
            for (position, table) in tables.chunks_exact(256).enumerate() {
                for (bits, &actual) in table.iter().enumerate() {
                    let expected = (0..8)
                        .filter(|&bit| bits & (1 << bit) != 0)
                        .fold(Gf::zero(), |sum, bit| {
                            sum + coefficients[8 * position + bit]
                        });
                    assert_eq!(actual, expected, "position {position}, bits {bits}");
                }
            }
        }
    }

    /// The η-premultiplied byte tables reproduce `scale·Φ_{r″}` bit-for-bit.
    #[test]
    fn phi_byte_tables_match_bitscan() {
        let eq_r2: Vec<Gf> = (0..128).map(|i| sample(0x9100 + i as u64)).collect();
        for &scale_seed in &[0u64, 0x42, 0xFFFF] {
            let scale = if scale_seed == 0 {
                Gf::one()
            } else {
                sample(scale_seed)
            };
            let tables = phi_byte_tables(&eq_r2, scale);
            for i in 0..64u64 {
                let v = sample(0xA000 + i);
                let w = *v.as_words();
                let mut acc = Gf::zero();
                for wi in 0..2usize {
                    let mut bits = w[wi];
                    while bits != 0 {
                        let t = bits.trailing_zeros() as usize;
                        acc += eq_r2[(wi << 6) | t];
                        bits &= bits.wrapping_sub(1);
                    }
                }
                assert_eq!(phi_from_words(w, &tables), scale * acc, "elem {i}");
            }
        }
    }

    /// The wide-accumulating product-pair round evaluator emits the exact
    /// messages (and asserted sum) of the generic per-point gather, across
    /// every round of a degree-2 two-MLE product sumcheck.
    #[test]
    fn prod_pair_evaluator_matches_generic() {
        use crate::piop::sumcheck::prover::ProverState;
        let nv = 5usize;
        let n = 1usize << nv;
        let zero_inner = Gf::zero().into_inner();
        let mk = |seed: u64| {
            DenseMultilinearExtension::from_evaluations_vec(
                nv,
                (0..n)
                    .map(|i| sample(seed + i as u64).into_inner())
                    .collect(),
                zero_inner,
            )
        };
        let mles = vec![mk(0xB000), mk(0xC000)];
        let comb = |vals: &[Gf]| vals[0] * vals[1];
        let mut generic = ProverState::<Gf>::new(mles.clone(), nv, 2);
        let mut fused = ProverState::<Gf>::new(mles, nv, 2);
        fused.round_evaluator = Some(Box::new(super::ProdPairWideEvaluator));
        let mut v_msg: Option<Gf> = None;
        for round in 0..nv {
            let mg = generic.prove_round(&v_msg, comb, &());
            let mf = fused.prove_round(&v_msg, comb, &());
            assert_eq!(
                mg.0.tail_evaluations, mf.0.tail_evaluations,
                "round {round} message mismatch"
            );
            v_msg = Some(sample(0xD000 + round as u64));
        }
        assert_eq!(generic.asserted_sum, fused.asserted_sum);
    }
}
