//! Accepted BitZ PMULL sumcheck schedules, moved from the root binary field.
//! This module is compiled only with NEON and AES/PMULL enabled.
use crate::gf128::aarch64::{clmul128 as clmul_256, pmull_hi, pmull_lo, reduce_256};
use core::arch::aarch64::{uint64x2_t, vdupq_n_u64, veorq_u64, vextq_u64, vld1q_u64, vst1q_u64};

#[inline(always)]
unsafe fn fold_x64(t0: uint64x2_t, t1: uint64x2_t, g: uint64x2_t, z: uint64x2_t) -> uint64x2_t {
    unsafe { veorq_u64(t0, veorq_u64(vextq_u64::<1>(z, t1), pmull_hi(t1, g))) }
}
use crate::Gf128;

/// Load one field element as a vector.
#[inline(always)]
pub(crate) unsafe fn ld(x: &Gf128) -> uint64x2_t {
    // SAFETY: `uint.as_words()` is a valid 16-byte word pair.
    unsafe { vld1q_u64(core::ptr::addr_of!(x.lo)) }
}

/// Reduce a 256-bit vector accumulator and build the field element.
#[inline(always)]
unsafe fn to_elt(acc: (uint64x2_t, uint64x2_t)) -> Gf128 {
    // SAFETY: as `pmull_lo`.
    unsafe {
        let r = reduce_256(acc.0, acc.1);
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), r);
        Gf128::from(out)
    }
}

/// One eq-factored round slot, fully NEON-resident: weight-fold the
/// two `L` entries (reduced), then XOR the three products into the
/// 256-bit vector accumulators unreduced —
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
    // SAFETY: as `pmull_lo`.
    unsafe {
        let g = vdupq_n_u64(0x87);
        let z = vdupq_n_u64(0);
        let l0w = mul_red(w, l0, g, z);
        let l1w = mul_red(w, l1, g, z);
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
/// chains, accumulators in vector registers, one reduction per
/// accumulator at the end).
pub(crate) fn eqf_single_pair_round(
    l: &[Gf128],
    r: &[Gf128],
    w: &[Gf128],
    half: usize,
) -> (Gf128, Gf128, Gf128) {
    // SAFETY: as `pmull_lo`; all indices are in bounds by the
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
    l0: &[Gf128],
    r0: &[Gf128],
    l1: &[Gf128],
    r1: &[Gf128],
    w: &[Gf128],
    half: usize,
) -> (Gf128, Gf128, Gf128) {
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
pub(crate) fn eqf_fold_in_place(v: &mut [Gf128], rho: &Gf128, half: usize) {
    // SAFETY: as `pmull_lo`; write index `b` is only ever read at the
    // earlier iteration `b/2`, so in-place is safe (the scalar body's
    // argument).
    unsafe {
        let rv = ld(rho);
        let mut b = 0usize;
        while b + 2 <= half {
            let v0a = ld(&v[b << 1]);
            let v1a = ld(&v[(b << 1) | 1]);
            let v0b = ld(&v[(b + 1) << 1]);
            let v1b = ld(&v[((b + 1) << 1) | 1]);
            let (pl, ph) = clmul_256(rv, veorq_u64(v1a, v0a));
            let pa = reduce_256(pl, ph);
            let (pl, ph) = clmul_256(rv, veorq_u64(v1b, v0b));
            let pb = reduce_256(pl, ph);
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), veorq_u64(v0a, pa));
            v[b] = Gf128::from(out);
            vst1q_u64(out.as_mut_ptr(), veorq_u64(v0b, pb));
            v[b + 1] = Gf128::from(out);
            b += 2;
        }
        if b < half {
            let v0 = ld(&v[b << 1]);
            let v1 = ld(&v[(b << 1) | 1]);
            let (pl, ph) = clmul_256(rv, veorq_u64(v1, v0));
            let p = reduce_256(pl, ph);
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), veorq_u64(v0, p));
            v[b] = Gf128::from(out);
        }
    }
}

/// Store a vector back into a field element.
#[inline(always)]
unsafe fn st(x: &mut Gf128, v: uint64x2_t) {
    // SAFETY: `out` is a valid 16-byte word pair.
    unsafe {
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), v);
        *x = Gf128::from(out);
    }
}

/// The deferred fold of one entry pair: `v0 ⊕ ρ·(v1 ⊕ v0)`, reduced —
/// the exact value [`eqf_fold_in_place`] writes.
#[inline(always)]
unsafe fn fold1(rv: uint64x2_t, v0: uint64x2_t, v1: uint64x2_t) -> uint64x2_t {
    // SAFETY: as `pmull_lo`.
    unsafe {
        let (pl, ph) = clmul_256(rv, veorq_u64(v1, v0));
        veorq_u64(v0, reduce_256(pl, ph))
    }
}

// -- fixed-scalar (preprocessed-multiplier) kernels ---------------
//
// For a multiplier FIXED across a pass, precompute `R0 = r` and
// `R1 = X^64·r mod f`, interleaved as `rl = [R0.lo, R1.lo]`,
// `rh = [R0.hi, R1.hi]`. Then `a·r = a0·R0 ⊕ a1·R1` and the four
// products are `pmull_lo/hi(a, rl)` and `pmull_lo/hi(a, rh)` — no
// operand shuffles — landing in a 191-bit `(t_lo, t_mid)` domain
// (value `t_lo ⊕ X^64·t_mid`, `t_mid` ≤ 127 bits) finished by ONE
// [`fold_x64`]: 5 PMULLs per reduced multiply vs 7 for the composed
// `clmul_256` + `reduce_256` element. XOR-summed aggregates
// accumulate in `(t_lo, t_mid)` and share the final fold: 4 PMULLs
// per term (the arity-4 double fold below: 12+1 per 4 inputs vs 21
// over two composed levels). Value-exact: the same carryless
// products and the same unique remainder mod `f`.

/// Preprocess a pass-fixed multiplier into the interleaved
/// `(rl, rh)` pair. `X^64·r = (0, r0) ⊕ g·r1` with `g·r1 ≤ 70`
/// bits, so `R1` is already reduced. One 64×64 clmul — amortised
/// over the pass.
#[inline(always)]
pub(crate) unsafe fn prep_fixed(rho: &Gf128) -> (uint64x2_t, uint64x2_t) {
    let w = &[rho.lo, rho.hi];
    let rg = super::super::kernels::clmul_64x64(w[1], 0x87);
    let rl = [w[0], rg[0]];
    let rh = [w[1], w[0] ^ rg[1]];
    // SAFETY: valid 16-byte word pairs.
    unsafe { (vld1q_u64(rl.as_ptr()), vld1q_u64(rh.as_ptr())) }
}

/// The 191-bit unreduced fixed-scalar product `(t_lo, t_mid)`:
/// 4 shuffle-free PMULLs, 2 EORs.
#[inline(always)]
pub(crate) unsafe fn mul_fixed_wide(
    av: uint64x2_t,
    rl: uint64x2_t,
    rh: uint64x2_t,
) -> (uint64x2_t, uint64x2_t) {
    // SAFETY: as `pmull_lo`.
    unsafe {
        (
            veorq_u64(pmull_lo(av, rl), pmull_hi(av, rl)),
            veorq_u64(pmull_lo(av, rh), pmull_hi(av, rh)),
        )
    }
}

/// One element-at-a-time reduced fixed-scalar multiply from the
/// stored `(rl, rh)` word pairs of a [`crate::PreparedGf128Mul`]: 4
/// shuffle-free PMULLs + one [`fold_x64`]. Value-exact vs the
/// composed multiply.
#[inline(always)]
pub(crate) fn fixed_mul_words(rl: &[u64; 2], rh: &[u64; 2], a: &Gf128) -> Gf128 {
    // SAFETY: valid 16-byte word pairs; intrinsics as `mul_fixed`.
    unsafe {
        let g = vdupq_n_u64(0x87);
        let z = vdupq_n_u64(0);
        let rlv = vld1q_u64(rl.as_ptr());
        let rhv = vld1q_u64(rh.as_ptr());
        let r = mul_fixed(ld(a), rlv, rhv, g, z);
        let mut out = [0u64; 2];
        vst1q_u64(out.as_mut_ptr(), r);
        Gf128::from(out)
    }
}

/// Reduced fixed-scalar multiply: 4 PMULLs + one [`fold_x64`].
#[inline(always)]
unsafe fn mul_fixed(
    av: uint64x2_t,
    rl: uint64x2_t,
    rh: uint64x2_t,
    g: uint64x2_t,
    z: uint64x2_t,
) -> uint64x2_t {
    // SAFETY: as `pmull_lo`.
    unsafe {
        let (tl, tm) = mul_fixed_wide(av, rl, rh);
        fold_x64(tl, tm, g, z)
    }
}

/// [`fold1`] with the pass-fixed ρ preprocessed: 5 PMULLs vs 7.
#[inline(always)]
unsafe fn fold1_fixed(
    rl: uint64x2_t,
    rh: uint64x2_t,
    g: uint64x2_t,
    z: uint64x2_t,
    v0: uint64x2_t,
    v1: uint64x2_t,
) -> uint64x2_t {
    // SAFETY: as `pmull_lo`.
    unsafe { veorq_u64(v0, mul_fixed(veorq_u64(v1, v0), rl, rh, g, z)) }
}

/// [`eqf_fold_in_place`] with the pass-fixed ρ preprocessed.
pub(crate) fn eqf_fold_in_place_fixed(v: &mut [Gf128], rho: &Gf128, half: usize) {
    // SAFETY: as `eqf_fold_in_place` (same access pattern: the write
    // index `b` is only ever read at the earlier iteration `b/2`).
    unsafe {
        let (rl, rh) = prep_fixed(rho);
        let g = vdupq_n_u64(0x87);
        let z = vdupq_n_u64(0);
        let mut b = 0usize;
        while b + 2 <= half {
            let v0a = ld(&v[b << 1]);
            let v1a = ld(&v[(b << 1) | 1]);
            let v0b = ld(&v[(b + 1) << 1]);
            let v1b = ld(&v[((b + 1) << 1) | 1]);
            let pa = fold1_fixed(rl, rh, g, z, v0a, v1a);
            let pb = fold1_fixed(rl, rh, g, z, v0b, v1b);
            st(&mut v[b], pa);
            st(&mut v[b + 1], pb);
            b += 2;
        }
        if b < half {
            let v0 = ld(&v[b << 1]);
            let v1 = ld(&v[(b << 1) | 1]);
            let p = fold1_fixed(rl, rh, g, z, v0, v1);
            st(&mut v[b], p);
        }
    }
}

/// [`eqf_fused_fold_round`] with the four fold products per slot
/// going through the preprocessed-ρ multiply (5 PMULLs each vs 7);
/// the message chain ([`eqf_slot`]) is unchanged.
pub(crate) fn eqf_fused_fold_round_fixed(
    l: &mut [Gf128],
    r: &mut [Gf128],
    rho: &Gf128,
    w: &[Gf128],
    half: usize,
) -> (Gf128, Gf128, Gf128) {
    // SAFETY: as `eqf_fused_fold_round` (same access pattern).
    unsafe {
        let (rl, rh) = prep_fixed(rho);
        let g = vdupq_n_u64(0x87);
        let z = vdupq_n_u64(0);
        let mut a0a = (z, z);
        let mut a1a = (z, z);
        let mut a2a = (z, z);
        let mut a0b = (z, z);
        let mut a1b = (z, z);
        let mut a2b = (z, z);
        let mut b = 0usize;
        while b + 2 <= half {
            let base = b << 2;
            let fl0 = fold1_fixed(rl, rh, g, z, ld(&l[base]), ld(&l[base + 1]));
            let fl1 = fold1_fixed(rl, rh, g, z, ld(&l[base + 2]), ld(&l[base + 3]));
            let fr0 = fold1_fixed(rl, rh, g, z, ld(&r[base]), ld(&r[base + 1]));
            let fr1 = fold1_fixed(rl, rh, g, z, ld(&r[base + 2]), ld(&r[base + 3]));
            let e = b << 1;
            st(&mut l[e], fl0);
            st(&mut l[e | 1], fl1);
            st(&mut r[e], fr0);
            st(&mut r[e | 1], fr1);
            eqf_slot(ld(&w[b]), fl0, fl1, fr0, fr1, &mut a0a, &mut a1a, &mut a2a);
            let c = b + 1;
            let base = c << 2;
            let gl0 = fold1_fixed(rl, rh, g, z, ld(&l[base]), ld(&l[base + 1]));
            let gl1 = fold1_fixed(rl, rh, g, z, ld(&l[base + 2]), ld(&l[base + 3]));
            let gr0 = fold1_fixed(rl, rh, g, z, ld(&r[base]), ld(&r[base + 1]));
            let gr1 = fold1_fixed(rl, rh, g, z, ld(&r[base + 2]), ld(&r[base + 3]));
            let e = c << 1;
            st(&mut l[e], gl0);
            st(&mut l[e | 1], gl1);
            st(&mut r[e], gr0);
            st(&mut r[e | 1], gr1);
            eqf_slot(ld(&w[c]), gl0, gl1, gr0, gr1, &mut a0b, &mut a1b, &mut a2b);
            b += 2;
        }
        if b < half {
            let base = b << 2;
            let fl0 = fold1_fixed(rl, rh, g, z, ld(&l[base]), ld(&l[base + 1]));
            let fl1 = fold1_fixed(rl, rh, g, z, ld(&l[base + 2]), ld(&l[base + 3]));
            let fr0 = fold1_fixed(rl, rh, g, z, ld(&r[base]), ld(&r[base + 1]));
            let fr1 = fold1_fixed(rl, rh, g, z, ld(&r[base + 2]), ld(&r[base + 3]));
            let e = b << 1;
            st(&mut l[e], fl0);
            st(&mut l[e | 1], fl1);
            st(&mut r[e], fr0);
            st(&mut r[e | 1], fr1);
            eqf_slot(ld(&w[b]), fl0, fl1, fr0, fr1, &mut a0a, &mut a1a, &mut a2a);
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

/// Reduced multiply on register-resident vectors — the
/// [`mul_words`] 3-limb fold-chain shape (6 PMULLs).
#[inline(always)]
unsafe fn mul_red(a: uint64x2_t, b: uint64x2_t, g: uint64x2_t, z: uint64x2_t) -> uint64x2_t {
    // SAFETY: as `pmull_lo`.
    unsafe {
        let t00 = pmull_lo(a, b);
        let t11 = pmull_hi(a, b);
        let bsw = vextq_u64(b, b, 1);
        let mid = veorq_u64(pmull_lo(a, bsw), pmull_hi(a, bsw));
        let t1 = fold_x64(mid, t11, g, z);
        fold_x64(t00, t1, g, z)
    }
}

/// One logical quad's four folded values under `d ≤ 2` deferred
/// challenges: arity-4 shared-reduction fixed fold at `d = 2`
/// (weights `ρ₁, ρ₂, ρ₁ρ₂` preprocessed; 12+1 PMULLs per value),
/// fixed pair fold at `d = 1`, plain loads at `d = 0`. Value-exact
/// vs `fold_logical`'s composed multiplies: the arity-4 expansion is
/// `v₀ ⊕ ρ₁·(v₁⊕v₀) ⊕ ρ₂·(v₂⊕v₀) ⊕ ρ₁ρ₂·(v₃⊕v₂⊕v₁⊕v₀)` and
/// reduction is `F₂`-linear.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_quad_logical(
    v: &[Gf128],
    b: usize,
    d: usize,
    w1l: uint64x2_t,
    w1h: uint64x2_t,
    w2l: uint64x2_t,
    w2h: uint64x2_t,
    w3l: uint64x2_t,
    w3h: uint64x2_t,
    g: uint64x2_t,
    z: uint64x2_t,
) -> [uint64x2_t; 4] {
    // SAFETY: as `pmull_lo`; indices in bounds by the caller's
    // contract (`v` has `(quads ≪ 2) ≪ d` entries).
    unsafe {
        match d {
            0 => core::array::from_fn(|i| ld(&v[(b << 2) | i])),
            1 => core::array::from_fn(|i| {
                let p = ((b << 2) | i) << 1;
                fold1_fixed(w1l, w1h, g, z, ld(&v[p]), ld(&v[p + 1]))
            }),
            _ => core::array::from_fn(|i| {
                let p = ((b << 2) | i) << 2;
                let v0 = ld(&v[p]);
                let v1 = ld(&v[p + 1]);
                let v2 = ld(&v[p + 2]);
                let v3 = ld(&v[p + 3]);
                let d1 = veorq_u64(v1, v0);
                let d2 = veorq_u64(v2, v0);
                let d3 = veorq_u64(d1, veorq_u64(v3, v2));
                let (mut tl, mut tm) = mul_fixed_wide(d1, w1l, w1h);
                let (l2, m2) = mul_fixed_wide(d2, w2l, w2h);
                let (l3, m3) = mul_fixed_wide(d3, w3l, w3h);
                tl = veorq_u64(tl, veorq_u64(l2, l3));
                tm = veorq_u64(tm, veorq_u64(m2, m3));
                veorq_u64(v0, fold_x64(tl, tm, g, z))
            }),
        }
    }
}

/// The 3×3 node grid `{0, 1, ∞}²` of a logical quad — rows `v = x₂`
/// node, columns `u = x₁` node, differences only (char 2: XORs).
/// Mirrors the generic pass's `grid` closure element-for-element.
#[inline(always)]
unsafe fn node_grid(a: &[uint64x2_t; 4]) -> [uint64x2_t; 9] {
    // SAFETY: plain NEON XORs.
    unsafe {
        let d00 = veorq_u64(a[1], a[0]);
        let d01 = veorq_u64(a[3], a[2]);
        [
            a[0],
            a[1],
            d00,
            a[2],
            a[3],
            d01,
            veorq_u64(a[2], a[0]),
            veorq_u64(a[3], a[1]),
            veorq_u64(d01, d00),
        ]
    }
}

/// The dense grid pass as one NEON kernel (see
/// `WideMulAcc::eqf_grid_pass` for the contract): fold the deferred
/// challenges per quad ([`fold_quad_logical`] — the arity-4
/// shared-reduction fixed fold at `d = 2`), write the folded quad to
/// the buffer prefix, weight the `L` side by the suffix tensor
/// ([`mul_red`]), and accumulate the nine node-grid products into
/// 256-bit vector accumulators — one reduction per node at the end,
/// then the `X₁`-monomial conversion (`a₁ = H(1) ⊕ H(0) ⊕ H(∞)` in
/// char 2). Value-exact vs the generic pass.
pub(crate) fn eqf_grid_pass(
    l: &mut [Gf128],
    r: &mut [Gf128],
    pending: &[Gf128],
    suffix: &[Gf128],
    quads: usize,
) -> [Gf128; 9] {
    let d = pending.len();
    debug_assert!(d <= 2, "grid kernel handles at most 2 deferred challenges");
    // SAFETY: as `pmull_lo`; per quad the `4·2^d` reads at
    // `((b≪2)|i)≪d` complete before the 4 prefix writes at `(b≪2)|i`
    // (all four logical values are in registers first), and later
    // quads read strictly above every earlier write.
    unsafe {
        let g = vdupq_n_u64(0x87);
        let z = vdupq_n_u64(0);
        let (w1l, w1h) = if d >= 1 {
            prep_fixed(&pending[0])
        } else {
            (z, z)
        };
        let (w2l, w2h, w3l, w3h) = if d == 2 {
            let p12 = pending[0] * &pending[1];
            let (al, ah) = prep_fixed(&pending[1]);
            let (bl, bh) = prep_fixed(&p12);
            (al, ah, bl, bh)
        } else {
            (z, z, z, z)
        };
        let mut acc = [(z, z); 9];
        for b in 0..quads {
            let lv = fold_quad_logical(l, b, d, w1l, w1h, w2l, w2h, w3l, w3h, g, z);
            let rv = fold_quad_logical(r, b, d, w1l, w1h, w2l, w2h, w3l, w3h, g, z);
            if d > 0 {
                let base = b << 2;
                for (i, (lf, rf)) in lv.iter().zip(rv.iter()).enumerate() {
                    st(&mut l[base | i], *lf);
                    st(&mut r[base | i], *rf);
                }
            }
            let wv = ld(&suffix[b]);
            let lw = [
                mul_red(wv, lv[0], g, z),
                mul_red(wv, lv[1], g, z),
                mul_red(wv, lv[2], g, z),
                mul_red(wv, lv[3], g, z),
            ];
            let lg = node_grid(&lw);
            let rg = node_grid(&rv);
            for (a, (x, y)) in acc.iter_mut().zip(lg.iter().zip(rg.iter())) {
                let (pl, ph) = clmul_256(*x, *y);
                a.0 = veorq_u64(a.0, pl);
                a.1 = veorq_u64(a.1, ph);
            }
        }
        let e: [uint64x2_t; 9] = core::array::from_fn(|k| reduce_256(acc[k].0, acc[k].1));
        core::array::from_fn(|i| {
            let (u, v) = (i / 3, i % 3);
            let base = v * 3;
            let val = match u {
                0 => e[base],
                2 => e[base + 2],
                _ => veorq_u64(e[base + 1], veorq_u64(e[base], e[base + 2])),
            };
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), val);
            Gf128::from(out)
        })
    }
}

/// NEON-resident fused deferred-fold + single-pair round body (the
/// pass-fusion path): per slot the four fold products are independent
/// PMULL chains, the folded entries store to the buffer prefix
/// (`2b, 2b+1` — writes trail the `4b..4b+4` reads), and the message
/// chain ([`eqf_slot`]) runs on the folded values still in registers.
/// Two interleaved slot chains, one reduction per accumulator at the
/// end.
pub(crate) fn eqf_fused_fold_round(
    l: &mut [Gf128],
    r: &mut [Gf128],
    rho: &Gf128,
    w: &[Gf128],
    half: usize,
) -> (Gf128, Gf128, Gf128) {
    // SAFETY: as `pmull_lo`; indices in bounds by the driver's
    // contract (`l`, `r` have 4·half entries, `w` half). Each slot
    // loads `4b..4b+4` into registers before storing `2b, 2b+1`, and
    // later slots read strictly above every earlier write.
    unsafe {
        let rv = ld(rho);
        let z = vdupq_n_u64(0);
        let mut a0a = (z, z);
        let mut a1a = (z, z);
        let mut a2a = (z, z);
        let mut a0b = (z, z);
        let mut a1b = (z, z);
        let mut a2b = (z, z);
        let mut b = 0usize;
        while b + 2 <= half {
            let base = b << 2;
            let fl0 = fold1(rv, ld(&l[base]), ld(&l[base + 1]));
            let fl1 = fold1(rv, ld(&l[base + 2]), ld(&l[base + 3]));
            let fr0 = fold1(rv, ld(&r[base]), ld(&r[base + 1]));
            let fr1 = fold1(rv, ld(&r[base + 2]), ld(&r[base + 3]));
            let e = b << 1;
            st(&mut l[e], fl0);
            st(&mut l[e | 1], fl1);
            st(&mut r[e], fr0);
            st(&mut r[e | 1], fr1);
            eqf_slot(ld(&w[b]), fl0, fl1, fr0, fr1, &mut a0a, &mut a1a, &mut a2a);
            let c = b + 1;
            let base = c << 2;
            let gl0 = fold1(rv, ld(&l[base]), ld(&l[base + 1]));
            let gl1 = fold1(rv, ld(&l[base + 2]), ld(&l[base + 3]));
            let gr0 = fold1(rv, ld(&r[base]), ld(&r[base + 1]));
            let gr1 = fold1(rv, ld(&r[base + 2]), ld(&r[base + 3]));
            let e = c << 1;
            st(&mut l[e], gl0);
            st(&mut l[e | 1], gl1);
            st(&mut r[e], gr0);
            st(&mut r[e | 1], gr1);
            eqf_slot(ld(&w[c]), gl0, gl1, gr0, gr1, &mut a0b, &mut a1b, &mut a2b);
            b += 2;
        }
        if b < half {
            let base = b << 2;
            let fl0 = fold1(rv, ld(&l[base]), ld(&l[base + 1]));
            let fl1 = fold1(rv, ld(&l[base + 2]), ld(&l[base + 3]));
            let fr0 = fold1(rv, ld(&r[base]), ld(&r[base + 1]));
            let fr1 = fold1(rv, ld(&r[base + 2]), ld(&r[base + 3]));
            let e = b << 1;
            st(&mut l[e], fl0);
            st(&mut l[e | 1], fl1);
            st(&mut r[e], fr0);
            st(&mut r[e | 1], fr1);
            eqf_slot(ld(&w[b]), fl0, fl1, fr0, fr1, &mut a0a, &mut a1a, &mut a2a);
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
