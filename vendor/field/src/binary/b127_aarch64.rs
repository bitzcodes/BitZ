//! Accepted B127 PMULL and SHA3 schedules. AES is required; SHA3 has a separate gate.
#[cfg(not(target_feature = "sha3"))]
use core::arch::aarch64::vandq_u64;
use core::arch::aarch64::{
    uint64x2_t, vdupq_n_u64, veorq_u64, vextq_u64, vld1q_u64, vshlq_n_u64, vshrq_n_u64,
    vsriq_n_u64, vst1q_u64,
};
#[cfg(target_feature = "sha3")]
use core::arch::aarch64::{vbcaxq_u64, veor3q_u64};

const MASK_HI_B127: u64 = u64::MAX >> 1;
use crate::gf128::aarch64::{clmul128 as clmul_256, pmull_hi, pmull_lo};

/// `lo` mask complement: bit 127 — the one bit of `lo` that belongs
/// to `H` (as its bit 0), not to `L`.
const TOP_BIT: [u64; 2] = [0, !MASK_HI_B127];
/// `g` spurious bit: `(P >> 126)`'s bit 0 is `P`'s bit 126, which
/// belongs to `L`, not to `H << 1`.
const G0_BIT: [u64; 2] = [1, 0];
/// `lo` mask: keep bits 0..126 (`L = P & (2^127 − 1)`) — the
/// non-SHA3 fallback's AND constant.
#[cfg(not(target_feature = "sha3"))]
const MASK127: [u64; 2] = [u64::MAX, MASK_HI_B127];
/// `g` mask (non-SHA3 fallback): clear lane-0 bit 0.
#[cfg(not(target_feature = "sha3"))]
const CLEAR_G0: [u64; 2] = [u64::MAX << 1, u64::MAX];

/// Reduce a 256-bit product `(lo, hi)` modulo `X^127 + X + 1` — the
/// PMULL-free trinomial fold.
///
/// `H = P >> 127` (`deg ≤ 125` under the module contract: `P` is an
/// XOR of products of canonical operands), result
/// `= (P & (2^127−1)) ⊕ H ⊕ (H << 1)`, one fold, exact. Both shifted
/// views come straight off the product limbs with SRI (shift-right-
/// insert) — `H` limb-wise is `(hi << 1) ⊕ (pre >> 63)` and
/// `H << 1 = P >> 126` (bit 0 cleared) is `(hi << 2) ⊕ (pre >> 62)`,
/// where `pre = [p1, p2]` are the limbs the `X^127`/`X^191` cuts
/// straddle. With FEAT_SHA3 (Apple M-series; enabled by
/// `-C target-cpu=native`) both corrective masks fuse into BCAX
/// (`a ⊕ (b & ~c)`): 7 µops, ~8-cycle chain, no PMULL. The fallback
/// keeps the AND/EOR form (9 µops, balanced XOR tree).
#[inline(always)]
pub(crate) unsafe fn reduce_256_b127(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    // SAFETY: plain NEON shifts/XORs (see `binary_gf128::neon::pmull_lo`
    // for the module's target-feature contract).
    unsafe {
        let pre = vextq_u64::<1>(lo, hi); // [p1, p2]
        // H = P >> 127, limb-wise: h0 = (p2<<1)|(p1>>63), h1 = (p3<<1)|(p2>>63).
        let h = vsriq_n_u64::<63>(vshlq_n_u64::<1>(hi), pre);
        // (P >> 126): g0 = (p2<<2)|(p1>>62), g1 = (p3<<2)|(p2>>62);
        // its bit 0 (P's bit 126) is spurious — H<<1 has bit 0 zero.
        let g = vsriq_n_u64::<62>(vshlq_n_u64::<2>(hi), pre);
        #[cfg(target_feature = "sha3")]
        {
            // t = h ⊕ (g & ~bit0); out = t ⊕ (lo & ~bit127).
            let t = vbcaxq_u64(h, g, vld1q_u64(G0_BIT.as_ptr()));
            vbcaxq_u64(t, lo, vld1q_u64(TOP_BIT.as_ptr()))
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            let g = vandq_u64(g, vld1q_u64(CLEAR_G0.as_ptr()));
            let l = vandq_u64(lo, vld1q_u64(MASK127.as_ptr()));
            veorq_u64(veorq_u64(l, h), g)
        }
    }
}

/// Fully NEON-resident reduced multiply on word pairs: the shared
/// schoolbook 4-PMULL product + the PMULL-free trinomial fold.
#[inline(always)]
pub(crate) fn mul_words(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
    // SAFETY: as `reduce_256_b127`; loads/stores are on valid 16-byte
    // word pairs.
    unsafe {
        let va = vld1q_u64(a.as_ptr());
        let vb = vld1q_u64(b.as_ptr());
        let (lo, hi) = clmul_256(va, vb);
        let r = reduce_256_b127(lo, hi);
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), r);
        out
    }
}

/// 128×128 → 256-bit carryless product, KARATSUBA (3 PMULL + the
/// mid-recombination XOR/EXTs). Provided for measurement against the
/// schoolbook default: with the PMULL-free b127 reduction the PMULL
/// ports are less contended than in the GF128 pipeline, so the
/// trade-off may land differently — the field bench decides.
#[inline(always)]
pub(crate) unsafe fn clmul_256_kara(a: uint64x2_t, b: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    // SAFETY: as `reduce_256_b127`.
    unsafe {
        let t00 = pmull_lo(a, b); // a0·b0
        let t11 = pmull_hi(a, b); // a1·b1
        let asum = veorq_u64(a, vextq_u64::<1>(a, a)); // [a0^a1, ·]
        let bsum = veorq_u64(b, vextq_u64::<1>(b, b)); // [b0^b1, ·]
        let mid = veorq_u64(pmull_lo(asum, bsum), veorq_u64(t00, t11));
        let z = vdupq_n_u64(0);
        let mid_lo = vextq_u64::<1>(z, mid); // mid << 64
        let mid_hi = vextq_u64::<1>(mid, z); // mid >> 64
        (veorq_u64(t00, mid_lo), veorq_u64(t11, mid_hi))
    }
}

/// Reduced multiply via the Karatsuba product (bench alternative;
/// value-identical to [`mul_words`]).
#[inline(always)]
pub(crate) fn mul_words_kara(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
    // SAFETY: as `mul_words`.
    unsafe {
        let va = vld1q_u64(a.as_ptr());
        let vb = vld1q_u64(b.as_ptr());
        let (lo, hi) = clmul_256_kara(va, vb);
        let r = reduce_256_b127(lo, hi);
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), r);
        out
    }
}

/// Reduce a 256-bit product `(lo, hi)` modulo `X^127 + X + 1` via the
/// GHASH-SHAPED PMULL fold — the alternative that spends the
/// reduction on the PMULL ports instead of the shift/logical pipes.
///
/// `X^128 ≡ X² + X = 0x6 (mod f)`, so the high 128 bits fold exactly
/// as GHASH's do against `0x87`: `hi ⊗ 0x6` via two PMULLs (word
/// layout `w0 = y0, w1 = y1 ^ z0, w2 = z1 ≤ 2 bits`), then the word-2
/// spill refolds through a third PMULL. Lane-aligned throughout — the
/// cross-lane bit-127 extraction of [`reduce_256_b127`] never
/// happens. What GHASH does NOT owe afterwards is the
/// canonicalization: the folded 128-bit value may set bit 127, which
/// `X^127 ≡ X + 1` folds back (SHR + EXT + SHL/EOR + BCAX; the
/// non-SHA3 fallback spends one more EOR). Structurally this is
/// GHASH's `reduce_256` + that tax, which is the point of measuring
/// it: it upper-bounds b127-with-PMULL-reduction at GHASH parity.
#[inline(always)]
pub(crate) unsafe fn reduce_256_b127_pfold(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    // SAFETY: as `reduce_256_b127`.
    unsafe {
        let g = vdupq_n_u64(0x6);
        let p0 = pmull_lo(hi, g); // (X²+X)·p2 = [y0, y1], y1 ≤ 2 bits
        let p1 = pmull_hi(hi, g); // (X²+X)·p3 = [z0, z1], z1 ≤ 2 bits
        let z = vdupq_n_u64(0);
        // r ← lo ⊕ (hi ⊗ 0x6) words 0–1.
        let r = veorq_u64(lo, veorq_u64(p0, vextq_u64::<1>(z, p1)));
        // word-2 spill (z1): X^128·z1 ≡ 0x6·z1, lands in word 0.
        let r = veorq_u64(r, pmull_lo(vextq_u64::<1>(p1, z), g));
        // Canonicalize bit 127: c = r >> 127; r' = (r \ bit127) ⊕ 3c.
        let c = vextq_u64::<1>(vshrq_n_u64::<63>(r), z); // [c, 0]
        let fold = veorq_u64(vshlq_n_u64::<1>(c), c); // [3c, 0]
        #[cfg(target_feature = "sha3")]
        {
            // fold ⊕ (r & ~bit127) in one BCAX.
            vbcaxq_u64(fold, r, vld1q_u64(TOP_BIT.as_ptr()))
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            veorq_u64(vandq_u64(r, vld1q_u64(MASK127.as_ptr())), fold)
        }
    }
}

/// Reduced multiply via the GHASH-shaped PMULL fold (bench
/// alternative; value-identical to [`mul_words`]).
#[inline(always)]
pub(crate) fn mul_words_pfold(a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
    // SAFETY: as `mul_words`.
    unsafe {
        let va = vld1q_u64(a.as_ptr());
        let vb = vld1q_u64(b.as_ptr());
        let (lo, hi) = clmul_256(va, vb);
        let r = reduce_256_b127_pfold(lo, hi);
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), r);
        out
    }
}

/// NEON-resident reduced SQUARE: char-2 cross terms cancel, so two
/// PMULLs + a square-specialized trinomial fold — 2 PMULL total vs
/// GHASH's 5.
///
/// The fold exploits the spread structure directly: with
/// `a = a_0 + X^{64} a_1`, `a² = S(a_0) + X^{128}·S(a_1)`
/// (`S = ` the even-position bit spread, one PMULL each), and
/// `X^{128} ≡ X² + X`, so
/// `a² ≡ S(a_0) ⊕ (S(a_1) << 1) ⊕ (S(a_1) << 2)` — and because the
/// canonical invariant zeroes `a_1`'s bit 63, `deg S(a_1) ≤ 124` and
/// `(X²+X)·S(a_1)` never reaches `X^127`: one fold, no masks at all
/// (6 µops with EOR3, 7 without).
/// One squaring step on a vector-resident value — the spread + the
/// square-specialized trinomial fold (see [`square_words`] for the
/// derivation). Canonical in, canonical out.
#[inline(always)]
unsafe fn square_step(va: uint64x2_t) -> uint64x2_t {
    // SAFETY: as `reduce_256_b127`.
    unsafe {
        let s_lo = pmull_lo(va, va); // S(a0), even bits, deg ≤ 126
        let s_hi = pmull_hi(va, va); // S(a1), even bits, deg ≤ 124
        let z = vdupq_n_u64(0);
        let e = vextq_u64::<1>(z, s_hi); // [0, s_hi.0] — the lane carries
        let t1 = vsriq_n_u64::<63>(vshlq_n_u64::<1>(s_hi), e); // S(a1) << 1
        let t2 = vsriq_n_u64::<62>(vshlq_n_u64::<2>(s_hi), e); // S(a1) << 2
        #[cfg(target_feature = "sha3")]
        {
            veor3q_u64(s_lo, t1, t2)
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            veorq_u64(veorq_u64(s_lo, t1), t2)
        }
    }
}

#[inline(always)]
pub(crate) fn square_words(a: &[u64; 2]) -> [u64; 2] {
    // SAFETY: as `mul_words`.
    unsafe {
        let r = square_step(vld1q_u64(a.as_ptr()));
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), r);
        out
    }
}

/// `n` successive squarings with the value held in a vector register
/// throughout — one load, one store, no per-step `Uint` bounce. The
/// Itoh–Tsujii ladder's squaring runs.
#[inline]
pub(crate) fn square_n_words(a: &[u64; 2], n: usize) -> [u64; 2] {
    // SAFETY: as `mul_words`.
    unsafe {
        let mut v = vld1q_u64(a.as_ptr());
        for _ in 0..n {
            v = square_step(v);
        }
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), v);
        out
    }
}

use crate::B127;

/// Load one field element as a vector.
#[inline(always)]
pub(crate) unsafe fn ld(x: &B127) -> uint64x2_t {
    // SAFETY: `uint.as_words()` is a valid 16-byte word pair.
    unsafe { vld1q_u64(x.as_words().as_ptr()) }
}

/// Reduce a 256-bit vector accumulator and build the field element.
#[inline(always)]
unsafe fn to_elt(acc: (uint64x2_t, uint64x2_t)) -> B127 {
    // SAFETY: as `reduce_256_b127`.
    unsafe {
        let r = reduce_256_b127(acc.0, acc.1);
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), r);
        B127::from_canonical_words(out)
    }
}

/// One eq-factored round slot, fully NEON-resident — the GF128 slot
/// with the trinomial weight-fold reductions:
/// `a0 += (w·l0)·r0`, `a2 += (w·Δl)·Δr`, `a1 += w11 ⊕ wc0 ⊕ wc2`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn eqf_slot(
    w: uint64x2_t,
    l0: uint64x2_t,
    l1: uint64x2_t,
    r0: uint64x2_t,
    r1: uint64x2_t,
    a0: &mut (uint64x2_t, uint64x2_t),
    a1: &mut (uint64x2_t, uint64x2_t),
    a2: &mut (uint64x2_t, uint64x2_t),
) {
    // SAFETY: as `reduce_256_b127`.
    unsafe {
        let (tl, th) = clmul_256(w, l0);
        let l0w = reduce_256_b127(tl, th);
        let (tl, th) = clmul_256(w, l1);
        let l1w = reduce_256_b127(tl, th);
        let (c0l, c0h) = clmul_256(l0w, r0);
        let (c1l, c1h) = clmul_256(l1w, r1);
        let dl = veorq_u64(l1w, l0w);
        let dr = veorq_u64(r1, r0);
        let (c2l, c2h) = clmul_256(dl, dr);
        a0.0 = veorq_u64(a0.0, c0l);
        a0.1 = veorq_u64(a0.1, c0h);
        a2.0 = veorq_u64(a2.0, c2l);
        a2.1 = veorq_u64(a2.1, c2h);
        a1.0 = veorq_u64(a1.0, veorq_u64(c1l, veorq_u64(c0l, c2l)));
        a1.1 = veorq_u64(a1.1, veorq_u64(c1h, veorq_u64(c0h, c2h)));
    }
}

/// NEON-resident single-pair round body (two interleaved slot
/// chains, accumulators in vector registers, one cheap reduction per
/// accumulator at the end).
pub(crate) fn eqf_single_pair_round(
    l: &[B127],
    r: &[B127],
    w: &[B127],
    half: usize,
) -> (B127, B127, B127) {
    // SAFETY: as `reduce_256_b127`; all indices are in bounds by the
    // driver's contract (`l`, `r` have 2·half entries, `w` half).
    unsafe {
        let z = vdupq_n_u64(0);
        let mut a0a = (z, z);
        let mut a1a = (z, z);
        let mut a2a = (z, z);
        let mut a0b = (z, z);
        let mut a1b = (z, z);
        let mut a2b = (z, z);
        let mut b = 0usize;
        while b + 2 <= half {
            let e = b << 1;
            eqf_slot(
                ld(&w[b]),
                ld(&l[e]),
                ld(&l[e | 1]),
                ld(&r[e]),
                ld(&r[e | 1]),
                &mut a0a,
                &mut a1a,
                &mut a2a,
            );
            let e = (b + 1) << 1;
            eqf_slot(
                ld(&w[b + 1]),
                ld(&l[e]),
                ld(&l[e | 1]),
                ld(&r[e]),
                ld(&r[e | 1]),
                &mut a0b,
                &mut a1b,
                &mut a2b,
            );
            b += 2;
        }
        if b < half {
            let e = b << 1;
            eqf_slot(
                ld(&w[b]),
                ld(&l[e]),
                ld(&l[e | 1]),
                ld(&r[e]),
                ld(&r[e | 1]),
                &mut a0a,
                &mut a1a,
                &mut a2a,
            );
        }
        a0a.0 = veorq_u64(a0a.0, a0b.0);
        a0a.1 = veorq_u64(a0a.1, a0b.1);
        a1a.0 = veorq_u64(a1a.0, a1b.0);
        a1a.1 = veorq_u64(a1a.1, a1b.1);
        a2a.0 = veorq_u64(a2a.0, a2b.0);
        a2a.1 = veorq_u64(a2a.1, a2b.1);
        (to_elt(a0a), to_elt(a1a), to_elt(a2a))
    }
}

/// NEON-resident two-pair round body (the fraction-GKR layer): the
/// two pairs are the two independent chains.
pub(crate) fn eqf_two_pair_round(
    l0: &[B127],
    r0: &[B127],
    l1: &[B127],
    r1: &[B127],
    w: &[B127],
    half: usize,
) -> (B127, B127, B127) {
    // SAFETY: as `eqf_single_pair_round`.
    unsafe {
        let z = vdupq_n_u64(0);
        let mut a0a = (z, z);
        let mut a1a = (z, z);
        let mut a2a = (z, z);
        let mut a0b = (z, z);
        let mut a1b = (z, z);
        let mut a2b = (z, z);
        let mut b = 0usize;
        while b < half {
            let e = b << 1;
            let wv = ld(&w[b]);
            eqf_slot(
                wv,
                ld(&l0[e]),
                ld(&l0[e | 1]),
                ld(&r0[e]),
                ld(&r0[e | 1]),
                &mut a0a,
                &mut a1a,
                &mut a2a,
            );
            eqf_slot(
                wv,
                ld(&l1[e]),
                ld(&l1[e | 1]),
                ld(&r1[e]),
                ld(&r1[e | 1]),
                &mut a0b,
                &mut a1b,
                &mut a2b,
            );
            b += 1;
        }
        a0a.0 = veorq_u64(a0a.0, a0b.0);
        a0a.1 = veorq_u64(a0a.1, a0b.1);
        a1a.0 = veorq_u64(a1a.0, a1b.0);
        a1a.1 = veorq_u64(a1a.1, a1b.1);
        a2a.0 = veorq_u64(a2a.0, a2b.0);
        a2a.1 = veorq_u64(a2a.1, a2b.1);
        (to_elt(a0a), to_elt(a1a), to_elt(a2a))
    }
}

/// NEON-resident in-place fold `v[b] ← v[2b] ⊕ ρ·(v[2b+1] ⊕ v[2b])`,
/// two independent entries per iteration.
pub(crate) fn eqf_fold_in_place(v: &mut [B127], rho: &B127, half: usize) {
    // SAFETY: as `reduce_256_b127`; write index `b` is only ever read
    // at the earlier iteration `b/2`, so in-place is safe.
    unsafe {
        let rv = ld(rho);
        let mut b = 0usize;
        while b + 2 <= half {
            let v0a = ld(&v[b << 1]);
            let v1a = ld(&v[(b << 1) | 1]);
            let v0b = ld(&v[(b + 1) << 1]);
            let v1b = ld(&v[((b + 1) << 1) | 1]);
            let (pl, ph) = clmul_256(rv, veorq_u64(v1a, v0a));
            let pa = reduce_256_b127(pl, ph);
            let (pl, ph) = clmul_256(rv, veorq_u64(v1b, v0b));
            let pb = reduce_256_b127(pl, ph);
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), veorq_u64(v0a, pa));
            v[b] = B127::from_canonical_words(out);
            vst1q_u64(out.as_mut_ptr(), veorq_u64(v0b, pb));
            v[b + 1] = B127::from_canonical_words(out);
            b += 2;
        }
        if b < half {
            let v0 = ld(&v[b << 1]);
            let v1 = ld(&v[(b << 1) | 1]);
            let (pl, ph) = clmul_256(rv, veorq_u64(v1, v0));
            let p = reduce_256_b127(pl, ph);
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), veorq_u64(v0, p));
            v[b] = B127::from_canonical_words(out);
        }
    }
}
#[inline]
pub(crate) fn mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    mul_words(&a, &b)
}
#[inline]
pub(crate) fn square_n(a: [u64; 2], n: usize) -> [u64; 2] {
    square_n_words(&a, n)
}
#[inline]
pub(crate) fn reduce_product(lo: uint64x2_t, hi: uint64x2_t) -> [u64; 2] {
    crate::gf128::aarch64::store(unsafe { reduce_256_b127(lo, hi) })
}
