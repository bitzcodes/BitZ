//! **Quad (arity-4) eq-factored sumcheck** for the merged forest's QUAD
//! layers (EXPERIMENTAL, `BITZ_QUAD=1`): proves
//! `Σ_x Σ_t eq(x; q)·scale_t·A_t(x)·B_t(x)·C_t(x)·D_t(x)` — one GKR layer
//! certifying TWO product-tree levels at once (the four multiplicands are
//! the quarters of level ℓ+2). Round polynomials have degree 5 (six
//! nodes `from(0..=5)`); the transcript ops mirror the generic prover
//! exactly (`nvars`/`degree` header, tail `P(1..=5)` absorb, post-draw
//! challenge re-absorb), so [`MLSumcheck::verify_as_subprotocol`] with
//! `degree = 5` verifies the proofs unchanged.
//!
//! Challenges and values live in `GF(2^128)` throughout — the layer merge
//! is SOUND for any generator α (degree-5 soundness 5/|K| per round).
//! Ported from the worktree-gf8 experiment with the order-255 byte-dlog
//! input surfaces stripped: inputs arrive as materialised K vectors, and
//! the win is structural (half the layer passes and line steps over the
//! stored region, even-levels-only chain), not representational. The
//! round bodies are scalar Karatsuba chains (~28 PMULL-class ops/slot) —
//! the fused NEON degree-5 kernel is the known follow-up
//! ([`MLSumcheck`]: crate::piop::sumcheck::MLSumcheck).

use crate::piop::sumcheck::eq_factored::{
    Pair2FoldTables, Pair2TauSet, build_leaf_fold_tables, build_pair2_fold_tables, leaf3_idx,
};
use crate::poly::univariate::binary_gf128::Gf128 as Gf;
use crate::transcript::traits::Transcript;
use crate::utils::wide_mul::WideMulAcc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::SumcheckProof;
use super::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};

/// One eq-weighted quad group (a tree in phase A; the tree axis itself in
/// phase B): the four multiplicand evaluation vectors over `{0,1}^k`.
pub struct QuadGroup {
    /// The shared eq point (`z_x` / `z_c`). All groups must agree.
    pub q: Vec<Gf>,
    /// Per-group scalar (eq(c, z_c) in phase A; 1 in phase B).
    pub scale: Gf,
    /// The four multiplicands `[Q00, Q10, Q01, Q11]` (`m = a | b≪1`).
    pub bufs: [Vec<Gf>; 4],
}

/// Per-slot degree-4 coefficient accumulation of
/// `Π_{m} (a_m + c·δ_m)` (char 2: `δ = v(0) + v(1)`), Karatsuba over the
/// two pairs, weighted by `w` into the wide accumulators.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn quad_slot(w: &Gf, a: [Gf; 4], d: [Gf; 4], acc: &mut [<Gf as WideMulAcc>::Wide; 5]) {
    // (a1 + cδ1)(a2 + cδ2) = p0 + p1 c + p2 c².
    let p0 = a[0] * a[1];
    let p2 = d[0] * d[1];
    let p1 = (a[0] + d[0]) * (a[1] + d[1]) + p0 + p2;
    let q0 = a[2] * a[3];
    let q2 = d[2] * d[3];
    let q1 = (a[2] + d[2]) * (a[3] + d[3]) + q0 + q2;
    // Quartic product coefficients.
    let h0 = p0 * q0;
    let h1 = p0 * q1 + p1 * q0;
    let h2 = p0 * q2 + p1 * q1 + p2 * q0;
    let h3 = p1 * q2 + p2 * q1;
    let h4 = p2 * q2;
    Gf::wide_add_assign(&mut acc[0], &Gf::mul_wide(w, &h0));
    Gf::wide_add_assign(&mut acc[1], &Gf::mul_wide(w, &h1));
    Gf::wide_add_assign(&mut acc[2], &Gf::mul_wide(w, &h2));
    Gf::wide_add_assign(&mut acc[3], &Gf::mul_wide(w, &h3));
    Gf::wide_add_assign(&mut acc[4], &Gf::mul_wide(w, &h4));
}

/// Restructured slot body — the DEFAULT (`BITZ_QUAD_KERNEL=0` restores
/// [`quad_slot`], diagnostic / A-B): the suffix weight is pre-folded into
/// the FIRST pair's operands (`w·a₀`, `w·d₀` — associativity moves it
/// inside the product), the two pair-Karatsubas emit reduced quadratic
/// coefficients, and the quadratic×quadratic cross stage runs 3-segment
/// Karatsuba — SIX wide products instead of nine reduced ones —
/// accumulated UNREDUCED straight into the five coefficient
/// accumulators: the cross stage performs zero reductions. Value-exact
/// vs [`quad_slot`] (associativity + distributivity + the F₂-linear
/// reduction), hence transcript-identical; ~80 PMULL-class ops/slot vs
/// ~125.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn quad_slot_k(w: &Gf, a: [Gf; 4], d: [Gf; 4], acc: &mut [<Gf as WideMulAcc>::Wide; 5]) {
    // First pair, w-prefolded: p = (w·A₀)·A₁ coefficients in the round var.
    let wa0 = *w * a[0];
    let wd0 = *w * d[0];
    let p0 = wa0 * a[1];
    let p2 = wd0 * d[1];
    let p1 = (wa0 + wd0) * (a[1] + d[1]) + p0 + p2;
    // Second pair, plain.
    let q0 = a[2] * a[3];
    let q2 = d[2] * d[3];
    let q1 = (a[2] + d[2]) * (a[3] + d[3]) + q0 + q2;
    quad_cross_k([p0, p1, p2], [q0, q1, q2], acc);
}

/// The [`quad_slot_k`] cross stage over precomputed pair coefficients
/// (the weight already folded into `p`): `h = p·q` by 3-segment
/// Karatsuba, all products wide, accumulated UNREDUCED — zero
/// cross-stage reductions.
///   h0 = m0; h1 = m01+m0+m1; h2 = m02+m0+m1+m2; h3 = m12+m1+m2;
///   h4 = m2.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn quad_cross_k(p: [Gf; 3], q: [Gf; 3], acc: &mut [<Gf as WideMulAcc>::Wide; 5]) {
    let [p0, p1, p2] = p;
    let [q0, q1, q2] = q;
    let m0 = Gf::mul_wide(&p0, &q0);
    let m1 = Gf::mul_wide(&p1, &q1);
    let m2 = Gf::mul_wide(&p2, &q2);
    let m01 = Gf::mul_wide(&(p0 + p1), &(q0 + q1));
    let m02 = Gf::mul_wide(&(p0 + p2), &(q0 + q2));
    let m12 = Gf::mul_wide(&(p1 + p2), &(q1 + q2));
    Gf::wide_add_assign(&mut acc[0], &m0);
    Gf::wide_add_assign(&mut acc[1], &m01);
    Gf::wide_add_assign(&mut acc[1], &m0);
    Gf::wide_add_assign(&mut acc[1], &m1);
    Gf::wide_add_assign(&mut acc[2], &m02);
    Gf::wide_add_assign(&mut acc[2], &m0);
    Gf::wide_add_assign(&mut acc[2], &m1);
    Gf::wide_add_assign(&mut acc[2], &m2);
    Gf::wide_add_assign(&mut acc[3], &m12);
    Gf::wide_add_assign(&mut acc[3], &m1);
    Gf::wide_add_assign(&mut acc[3], &m2);
    Gf::wide_add_assign(&mut acc[4], &m2);
}

/// The restructured-body knob: default ON; `BITZ_QUAD_KERNEL=0` restores
/// the naive slot/node bodies. Read per prove call (NOT once per
/// process) so the byte-identity pin can toggle it in one test process.
fn quad_kernel_on() -> bool {
    std::env::var("BITZ_QUAD_KERNEL").map_or(true, |v| v != "0")
}

/// One round's message/transcript close, shared by the quad drivers:
/// convert the per-group coefficient quintuples `H_t` to the six-node
/// message `M(c) = Σ_t a_t·eq1(c; qj)·H_t(c)`, absorb the tail, draw and
/// re-absorb ρ, and advance the per-group prefix scalars. Sets
/// `claimed_sum` on the first round. Returns ρ. Two value-exact node
/// conversions selected by `kernel`, transcript-identical either way.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
fn quad_round_close(
    transcript: &mut impl Transcript,
    hs: &[[Gf; 5]],
    a_scalars: &mut [Gf],
    qj: &Gf,
    nodes: &[Gf],
    node_pows: &[[Gf; 5]],
    kernel: bool,
    first_round: bool,
    claimed_sum: &mut Gf,
    messages: &mut Vec<ProverMsg<Gf>>,
    buf: &mut Vec<u8>,
) -> Gf {
    let zero = Gf::zero();
    let one = Gf::one();
    // M(c) = Σ_t A_t·eq1(c; q[j−1])·H_t(c) at the six nodes.
    let e0 = one + *qj;
    let mut m_nodes = [zero; 6];
    if kernel {
        // Σ_t a_t·eq1(c)·H_t(c) = eq1(c)·Σ_t (a_t·H_t)(c): fold a_t
        // into the coefficients once per group (5 muls), evaluate
        // nodes 0/1 mul-free (c = 0 → h₀; c = 1 → Σ h_i, char-2
        // bit-pattern nodes), dot the power rows for the rest, and
        // apply the node factor eq1(c) once per node AFTER the group
        // sum. Value-exact (distributivity); ~21 muls per group
        // instead of ~42.
        for (t, h) in hs.iter().enumerate() {
            let a_t = a_scalars[t];
            let ah: [Gf; 5] = [a_t * h[0], a_t * h[1], a_t * h[2], a_t * h[3], a_t * h[4]];
            m_nodes[0] += ah[0];
            m_nodes[1] += ah[0] + ah[1] + ah[2] + ah[3] + ah[4];
            for (c, slot) in m_nodes.iter_mut().enumerate().skip(2) {
                let pw = &node_pows[c];
                let mut hv = ah[0];
                for (i, ahi) in ah.iter().enumerate().skip(1) {
                    hv += *ahi * pw[i];
                }
                *slot += hv;
            }
        }
        for (c, slot) in m_nodes.iter_mut().enumerate() {
            let cn = nodes[c];
            let eq1 = e0 * (one + cn) + *qj * cn;
            *slot = eq1 * *slot;
        }
    } else {
        for (t, h) in hs.iter().enumerate() {
            let a_t = a_scalars[t];
            for (c, slot) in m_nodes.iter_mut().enumerate() {
                let cn = nodes[c];
                let eq1 = e0 * (one + cn) + *qj * cn;
                let pw = &node_pows[c];
                let mut hv = zero;
                for (i, hi) in h.iter().enumerate() {
                    hv += *hi * pw[i];
                }
                *slot += a_t * eq1 * hv;
            }
        }
    }

    if first_round {
        *claimed_sum = m_nodes[0] + m_nodes[1];
    }
    let tail: Vec<Gf> = m_nodes[1..].to_vec();
    transcript.absorb_random_field_slice(&tail, buf);
    messages.push(ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail)));

    let rho: Gf = transcript.get_field_challenge(&());
    transcript.absorb_random_field(&rho, buf);

    for a in a_scalars.iter_mut() {
        let e = e0 * (one + rho) + *qj * rho;
        *a = *a * e;
    }
    rho
}

/// Prove the quad relation (see the module doc). Returns
/// `(proof, point, finals)` with `finals[t] = [A,B,C,D](point)`.
///
/// `k = q.len()` must be ≥ 1 (the `k = 0` root layer is the caller's
/// phase-B-only special case).
#[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
pub fn prove_quad_eq_sumcheck(
    transcript: &mut impl Transcript,
    groups: Vec<QuadGroup>,
) -> (SumcheckProof<Gf>, Vec<Gf>, Vec<[Gf; 4]>) {
    let k = groups.first().map_or(0, |g| g.q.len());
    assert!(k >= 1, "quad sumcheck needs >= 1 variable");
    debug_assert!(
        groups
            .iter()
            .all(|g| g.q == groups[0].q && g.bufs.iter().all(|v| v.len() == 1 << k))
    );
    let zero = Gf::zero();
    let one = Gf::one();
    let kernel = quad_kernel_on();
    // Six Lagrange nodes 0..=5 (bit-pattern convention) + their power rows
    // for the coefficient → node conversion.
    let nodes: Vec<Gf> = (0u128..6).map(Gf::from_polynomial_bits).collect();
    let node_pows: Vec<[Gf; 5]> = nodes
        .iter()
        .map(|&c| {
            let c2 = c * c;
            [one, c, c2, c2 * c, c2 * c2]
        })
        .collect();

    let suffix = crate::piop::sumcheck::eq_factored::suffix_tensors(&groups[0].q, &());
    let q_pt = groups[0].q.clone();
    let mut a_scalars: Vec<Gf> = groups.iter().map(|g| g.scale).collect();
    let mut bufs: Vec<[Vec<Gf>; 4]> = groups.into_iter().map(|g| g.bufs).collect();

    let mut buf = vec![0u8; 16];
    transcript.absorb_random_field(&Gf::from_polynomial_bits(k as u128), &mut buf);
    transcript.absorb_random_field(&Gf::from_polynomial_bits(5), &mut buf);

    let mut randomness: Vec<Gf> = Vec::with_capacity(k);
    let mut messages: Vec<ProverMsg<Gf>> = Vec::with_capacity(k);
    let mut claimed_sum = zero;
    // Pass fusion (the driver's `pending_rho` pattern): a round's fold is
    // deferred into the NEXT round's message pass — one sweep reads the
    // unfolded buffers, folds ρ_{j−1} in registers, accumulates the
    // message over the folded quads, and lands the folded values in
    // place.
    let mut pending_rho: Option<Gf> = None;

    for j in 1..=k {
        let half = 1usize << (k - j);
        let suffix_j = suffix.tensor(j - 1);

        let hs: Vec<[Gf; 5]> = if let Some(rho_prev) = pending_rho.take() {
            // Fused fold + message: buffers hold 4·half unfolded entries.
            let fused = |b: &mut [Vec<Gf>; 4]| -> [Gf; 5] {
                let mut acc = [
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                ];
                for s in 0..half {
                    let base = s << 2;
                    let mut a = [zero; 4];
                    let mut d = [zero; 4];
                    for m in 0..4 {
                        let v = &mut b[m];
                        let f0 = v[base] + rho_prev * (v[base] + v[base + 1]);
                        let f1 = v[base + 2] + rho_prev * (v[base + 2] + v[base + 3]);
                        a[m] = f0;
                        d[m] = f1 + f0;
                        let e = s << 1;
                        v[e] = f0;
                        v[e | 1] = f1;
                    }
                    if kernel {
                        quad_slot_k(&suffix_j[s], a, d, &mut acc);
                    } else {
                        quad_slot(&suffix_j[s], a, d, &mut acc);
                    }
                }
                for v in b.iter_mut() {
                    v.truncate(half << 1);
                }
                acc.map(Gf::from_wide)
            };
            #[cfg(feature = "parallel")]
            {
                let min_len = crate::piop::sumcheck::eq_factored::par_min_len(bufs.len(), half);
                bufs.par_iter_mut()
                    .with_min_len(min_len)
                    .map(fused)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                bufs.iter_mut().map(fused).collect()
            }
        } else {
            // Round 1: plain message over the (2·half)-entry buffers — no
            // fold yet.
            let compute = |b: &[Vec<Gf>; 4]| -> [Gf; 5] {
                let mut acc = [
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                    Gf::wide_zero(&zero),
                ];
                for s in 0..half {
                    let e = s << 1;
                    let mut a = [zero; 4];
                    let mut d = [zero; 4];
                    for m in 0..4 {
                        let v0 = b[m][e];
                        a[m] = v0;
                        d[m] = b[m][e | 1] + v0;
                    }
                    if kernel {
                        quad_slot_k(&suffix_j[s], a, d, &mut acc);
                    } else {
                        quad_slot(&suffix_j[s], a, d, &mut acc);
                    }
                }
                acc.map(Gf::from_wide)
            };
            #[cfg(feature = "parallel")]
            {
                let min_len = crate::piop::sumcheck::eq_factored::par_min_len(bufs.len(), half);
                bufs.par_iter().with_min_len(min_len).map(compute).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                bufs.iter().map(compute).collect()
            }
        };

        let rho = quad_round_close(
            transcript,
            &hs,
            &mut a_scalars,
            &q_pt[j - 1],
            &nodes,
            &node_pows,
            kernel,
            j == 1,
            &mut claimed_sum,
            &mut messages,
            &mut buf,
        );

        if j < k {
            // Defer the fold into the next round's fused message pass.
            pending_rho = Some(rho);
            randomness.push(rho);
        } else {
            // Final interpolation at ρ_k: after the round-k message pass
            // (which applied any pending fold) the buffers hold TWO
            // folded entries per multiplicand.
            let finals: Vec<[Gf; 4]> = bufs
                .iter()
                .map(|b| {
                    let interp = |v: &[Gf]| -> Gf { v[0] + rho * (v[0] + v[1]) };
                    [interp(&b[0]), interp(&b[1]), interp(&b[2]), interp(&b[3])]
                })
                .collect();
            randomness.push(rho);
            return (
                SumcheckProof {
                    messages,
                    claimed_sum,
                },
                randomness,
                finals,
            );
        }
    }
    unreachable!("the final round returns")
}

// ========================================================================
// The BOTTOM quad layer (`BITZ_QUAD=2`): the arity-2 plan's pair and leaf
// layers merged into ONE arity-4 bit-driven layer — output d−2, consuming
// the LEAVES, whose four quarter multiplicands are never materialised.
// ========================================================================

/// One tree's inputs to the bottom quad layer: the midpoint-split
/// committed bit halves — the same per-tree arrays the arity-2 cascades
/// read (`lbits[i]` selects leaf `i`, `rbits[i]` leaf `i + 2^{d−1}`).
pub struct QuadBitGroup<'a> {
    pub scale: Gf,
    pub lbits: &'a [u64],
    pub rbits: &'a [u64],
}

/// The tree-shared tables of the bottom layer: `te`/`to` (round 1's
/// `p(0)`/`p(1)` gathers ARE pair-layer values) and the UNWEIGHTED
/// 16-case ΔΔ table
/// `t_dd[p≪4 | lw | (rw≪2)] = Σ_{ε,δ} lw_ε·rw_δ·τ_l[2p+ε]·τ_r[2p+δ]`
/// (E-side position pairs at `p < 2^{k−1}`, O-side at `2^{k−1} + p`),
/// plus the leaf τ halves — the stash builders' inputs.
pub struct QuadBottomTables<'a> {
    pub te: &'a [Gf],
    pub to: &'a [Gf],
    pub t_dd: &'a [Gf],
    pub tau_l: &'a [Gf],
    pub tau_r: &'a [Gf],
}

/// Phase A of the bottom quad layer: `Σ_x eq(x; q)·scale_t·Π_m Q_m(x)`
/// with `Q_m(x) = leaf(x + m·2^k)` (`k = d−2`, `m = a | b≪1`) — the
/// arity-2 plan's pair and leaf layers as ONE degree-5 sumcheck run off
/// the committed bits (S1 of `docs/forest-speedup-ideas.md`, the design
/// of `docs/quad-bottom-merge-prompt.md`):
///
/// - **round 1**: each Karatsuba pair's quadratic coefficients are
///   GATHERS — pairing the multiplicands ACROSS the top bit (E = Q00×Q01,
///   O = Q10×Q11; bracketing is free, a product's coefficients are
///   pairing-invariant) makes `p(0)`/`p(1)` `te`/`to` entries and `p₂`
///   one ΔΔ entry; the weight folds into `p` by three multiplies
///   ([`quad_cross_k`]).
/// - **round 1's fold** keeps the bits and stashes the arity-2 cascade's
///   F₁ ([`build_leaf_fold_tables`], τ-sliced per quarter); round 2
///   gathers `(a, δ)` per multiplicand from it (aligned nibble windows)
///   and runs the plain slot kernel.
/// - **round 2's fold** stashes F₂ ([`build_pair2_fold_tables`],
///   16-case); round 3 gathers byte-keyed values inline ([`leaf3_idx`]
///   windows per quarter); **round 3's fold** materialises the four
///   quarter buffers (`2^{k−3}` each — half the arity-2 plan's dense
///   residue).
/// - **rounds 4..k**: the Dense fused flow of [`prove_quad_eq_sumcheck`].
///
/// Returns finals in the CLOSE order `[Q00, Q10, Q01, Q11]` (the internal
/// E/O pairing permuted back), so the caller's phase B and 0x33 close are
/// the standard quad blocks. Transcript-identical to
/// [`prove_quad_eq_sumcheck`] over materialised quarters — every gathered
/// value is the exact field element the dense flow computes (the table
/// builders' char-2 identities), and exact ops on equal values are
/// byte-equal — pinned by `bottom_matches_dense_quad`.
#[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
pub fn prove_quad_bottom_sumcheck(
    transcript: &mut impl Transcript,
    q_pt: Vec<Gf>,
    groups: Vec<QuadBitGroup>,
    tables: &QuadBottomTables<'_>,
) -> (SumcheckProof<Gf>, Vec<Gf>, Vec<[Gf; 4]>) {
    let k = q_pt.len();
    assert!(
        k >= 4,
        "the bottom quad needs k >= 4 (three bit-driven rounds, then a dense fold)"
    );
    let q1b = 1usize << k; // quarter size = the O-side bit offset
    let hdd = 1usize << (k - 1); // t_dd's O-side pair base
    let zero = Gf::zero();
    let one = Gf::one();
    let kernel = quad_kernel_on();
    let words = (2 * q1b).div_ceil(64);
    debug_assert!(
        groups
            .iter()
            .all(|g| g.lbits.len() == words && g.rbits.len() == words)
    );
    debug_assert_eq!(tables.te.len(), q1b << 2);
    debug_assert_eq!(tables.to.len(), q1b << 2);
    debug_assert_eq!(tables.t_dd.len(), q1b << 4);
    debug_assert_eq!(tables.tau_l.len(), q1b << 1);
    debug_assert_eq!(tables.tau_r.len(), q1b << 1);
    let nodes: Vec<Gf> = (0u128..6).map(Gf::from_polynomial_bits).collect();
    let node_pows: Vec<[Gf; 5]> = nodes
        .iter()
        .map(|&c| {
            let c2 = c * c;
            [one, c, c2, c2 * c, c2 * c2]
        })
        .collect();
    let suffix = crate::piop::sumcheck::eq_factored::suffix_tensors(&q_pt, &());
    let mut a_scalars: Vec<Gf> = groups.iter().map(|g| g.scale).collect();

    let mut buf = vec![0u8; 16];
    transcript.absorb_random_field(&Gf::from_polynomial_bits(k as u128), &mut buf);
    transcript.absorb_random_field(&Gf::from_polynomial_bits(5), &mut buf);

    let mut randomness: Vec<Gf> = Vec::with_capacity(k);
    let mut messages: Vec<ProverMsg<Gf>> = Vec::with_capacity(k);
    let mut claimed_sum = zero;
    let wide5 = |z: &Gf| -> [<Gf as WideMulAcc>::Wide; 5] {
        [
            Gf::wide_zero(z),
            Gf::wide_zero(z),
            Gf::wide_zero(z),
            Gf::wide_zero(z),
            Gf::wide_zero(z),
        ]
    };
    let bit2 = |bits: &[u64], p: usize| -> usize { ((bits[p >> 6] >> (p & 63)) & 3) as usize };
    let bit4 = |bits: &[u64], p: usize| -> usize { ((bits[p >> 6] >> (p & 63)) & 15) as usize };
    let bit8 = |bits: &[u64], p: usize| -> usize { ((bits[p >> 6] >> (p & 63)) & 255) as usize };

    macro_rules! par_hs {
        ($body:expr, $half:expr) => {{
            #[cfg(feature = "parallel")]
            {
                let min_len = crate::piop::sumcheck::eq_factored::par_min_len(groups.len(), $half);
                groups
                    .par_iter()
                    .with_min_len(min_len)
                    .map($body)
                    .collect::<Vec<[Gf; 5]>>()
            }
            #[cfg(not(feature = "parallel"))]
            {
                groups.iter().map($body).collect::<Vec<[Gf; 5]>>()
            }
        }};
    }

    // ---- round 1: coefficients straight off te/to + the ΔΔ table ----
    let half1 = 1usize << (k - 1);
    {
        let suffix_1 = suffix.tensor(0);
        let r1 = |g: &QuadBitGroup| -> [Gf; 5] {
            let mut acc = wide5(&zero);
            for s in 0..half1 {
                let y = s << 1;
                // E pair (Q00 × Q01): two te gathers + one ΔΔ gather.
                let lw = bit2(&g.lbits, y);
                let rw = bit2(&g.rbits, y);
                let p0 = tables.te[(y << 2) | (lw & 1) | ((rw & 1) << 1)];
                let pv1 = tables.te[((y | 1) << 2) | (lw >> 1) | (rw & 2)];
                let p2 = tables.t_dd[(s << 4) | lw | (rw << 2)];
                let p1 = p0 + pv1 + p2;
                // O pair (Q10 × Q11): the same at bit offset 2^k.
                let lo = bit2(&g.lbits, q1b + y);
                let ro = bit2(&g.rbits, q1b + y);
                let q0 = tables.to[(y << 2) | (lo & 1) | ((ro & 1) << 1)];
                let qv1 = tables.to[((y | 1) << 2) | (lo >> 1) | (ro & 2)];
                let q2c = tables.t_dd[((hdd + s) << 4) | lo | (ro << 2)];
                let q1c = q0 + qv1 + q2c;
                let w = suffix_1[s];
                quad_cross_k([w * p0, w * p1, w * p2], [q0, q1c, q2c], &mut acc);
            }
            acc.map(Gf::from_wide)
        };
        let hs = par_hs!(r1, half1);
        let rho = quad_round_close(
            transcript,
            &hs,
            &mut a_scalars,
            &q_pt[0],
            &nodes,
            &node_pows,
            kernel,
            true,
            &mut claimed_sum,
            &mut messages,
            &mut buf,
        );
        randomness.push(rho);
    }
    // Round 1's fold keeps the bits: the arity-2 cascade's F₁ tables,
    // τ-sliced per quarter (Q00/Q10 through t_l, Q01/Q11 through t_r).
    let f1 = {
        let ft = build_leaf_fold_tables(&randomness[0], &one, tables.tau_l, tables.tau_r);
        Pair2TauSet {
            te: ft.t_l,
            to: ft.t_r,
        }
    };

    // ---- round 2: (a, δ) per multiplicand off F₁ (nibble windows) ----
    let half2 = 1usize << (k - 2);
    {
        let suffix_2 = suffix.tensor(1);
        let qoff = 1usize << (k - 1); // Q10/Q11's F₁ pair-position base
        let r2 = |g: &QuadBitGroup| -> [Gf; 5] {
            let mut acc = wide5(&zero);
            for s in 0..half2 {
                let e = s << 1;
                let n00 = bit4(&g.lbits, s << 2);
                let n01 = bit4(&g.rbits, s << 2);
                let n10 = bit4(&g.lbits, q1b + (s << 2));
                let n11 = bit4(&g.rbits, q1b + (s << 2));
                let val = |t: &[Gf], base: usize, ei: usize, c: usize| t[((base + ei) << 2) | c];
                // Slot order [Q00, Q01, Q10, Q11]: pairs (0,1) = E, (2,3) = O.
                let a0 = val(&f1.te, 0, e, n00 & 3);
                let v0 = val(&f1.te, 0, e | 1, n00 >> 2);
                let a1 = val(&f1.to, 0, e, n01 & 3);
                let v1 = val(&f1.to, 0, e | 1, n01 >> 2);
                let a2 = val(&f1.te, qoff, e, n10 & 3);
                let v2 = val(&f1.te, qoff, e | 1, n10 >> 2);
                let a3 = val(&f1.to, qoff, e, n11 & 3);
                let v3 = val(&f1.to, qoff, e | 1, n11 >> 2);
                let a = [a0, a1, a2, a3];
                let d = [v0 + a0, v1 + a1, v2 + a2, v3 + a3];
                if kernel {
                    quad_slot_k(&suffix_2[s], a, d, &mut acc);
                } else {
                    quad_slot(&suffix_2[s], a, d, &mut acc);
                }
            }
            acc.map(Gf::from_wide)
        };
        let hs = par_hs!(r2, half2);
        let rho = quad_round_close(
            transcript,
            &hs,
            &mut a_scalars,
            &q_pt[1],
            &nodes,
            &node_pows,
            kernel,
            false,
            &mut claimed_sum,
            &mut messages,
            &mut buf,
        );
        randomness.push(rho);
    }
    // Round 2's fold stashes F₂ (16-case per position pair).
    let f2: Pair2FoldTables<Gf> = build_pair2_fold_tables(&randomness[1], &one, &f1);
    drop(f1);

    // ---- round 3: byte-keyed values inline off F₂ ----
    let half3 = 1usize << (k - 3);
    let qoff3 = 1usize << (k - 2); // Q10/Q11's F₂ position base
    {
        let suffix_3 = suffix.tensor(2);
        let r3 = |g: &QuadBitGroup| -> [Gf; 5] {
            let mut acc = wide5(&zero);
            for s in 0..half3 {
                let e = s << 1;
                let b00 = bit8(&g.lbits, s << 3);
                let b01 = bit8(&g.rbits, s << 3);
                let b10 = bit8(&g.lbits, q1b + (s << 3));
                let b11 = bit8(&g.rbits, q1b + (s << 3));
                let val = |t: &[Gf], base: usize, ei: usize, nib: usize| {
                    t[((base + ei) << 4) | leaf3_idx(nib)]
                };
                let a0 = val(&f2.f_e, 0, e, b00 & 15);
                let v0 = val(&f2.f_e, 0, e | 1, b00 >> 4);
                let a1 = val(&f2.f_o, 0, e, b01 & 15);
                let v1 = val(&f2.f_o, 0, e | 1, b01 >> 4);
                let a2 = val(&f2.f_e, qoff3, e, b10 & 15);
                let v2 = val(&f2.f_e, qoff3, e | 1, b10 >> 4);
                let a3 = val(&f2.f_o, qoff3, e, b11 & 15);
                let v3 = val(&f2.f_o, qoff3, e | 1, b11 >> 4);
                let a = [a0, a1, a2, a3];
                let d = [v0 + a0, v1 + a1, v2 + a2, v3 + a3];
                if kernel {
                    quad_slot_k(&suffix_3[s], a, d, &mut acc);
                } else {
                    quad_slot(&suffix_3[s], a, d, &mut acc);
                }
            }
            acc.map(Gf::from_wide)
        };
        let hs = par_hs!(r3, half3);
        let rho = quad_round_close(
            transcript,
            &hs,
            &mut a_scalars,
            &q_pt[2],
            &nodes,
            &node_pows,
            kernel,
            false,
            &mut claimed_sum,
            &mut messages,
            &mut buf,
        );
        randomness.push(rho);
    }
    // Round 3's fold materialises the four quarter buffers (2^{k−3} each).
    let rho3 = randomness[2];
    let mat_group = |g: &QuadBitGroup| -> [Vec<Gf>; 4] {
        let mat = |t: &[Gf], base: usize, bits: &[u64], bitbase: usize| -> Vec<Gf> {
            (0..half3)
                .map(|p| {
                    let by = bit8(bits, bitbase + (p << 3));
                    let e = p << 1;
                    let v0 = t[((base + e) << 4) | leaf3_idx(by & 15)];
                    let v1 = t[((base + (e | 1)) << 4) | leaf3_idx(by >> 4)];
                    v0 + rho3 * (v0 + v1)
                })
                .collect()
        };
        [
            mat(&f2.f_e, 0, &g.lbits, 0),
            mat(&f2.f_o, 0, &g.rbits, 0),
            mat(&f2.f_e, qoff3, &g.lbits, q1b),
            mat(&f2.f_o, qoff3, &g.rbits, q1b),
        ]
    };
    #[cfg(feature = "parallel")]
    let mut bufs: Vec<[Vec<Gf>; 4]> = {
        let min_len = crate::piop::sumcheck::eq_factored::par_min_len(groups.len(), half3);
        groups
            .par_iter()
            .with_min_len(min_len)
            .map(mat_group)
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let mut bufs: Vec<[Vec<Gf>; 4]> = groups.iter().map(mat_group).collect();
    drop(f2);
    drop(groups);

    // ---- rounds 4..k: the Dense fused flow ([`prove_quad_eq_sumcheck`]'s
    // loop over the materialised quarters; slot order [Q00,Q01,Q10,Q11]
    // throughout — pairing-invariant coefficients). ----
    let mut pending_rho: Option<Gf> = None;
    for j in 4..=k {
        let half = 1usize << (k - j);
        let suffix_j = suffix.tensor(j - 1);
        let hs: Vec<[Gf; 5]> = if let Some(rho_prev) = pending_rho.take() {
            let fused = |b: &mut [Vec<Gf>; 4]| -> [Gf; 5] {
                let mut acc = wide5(&zero);
                for s in 0..half {
                    let base = s << 2;
                    let mut a = [zero; 4];
                    let mut d = [zero; 4];
                    for m in 0..4 {
                        let v = &mut b[m];
                        let f0 = v[base] + rho_prev * (v[base] + v[base + 1]);
                        let f1 = v[base + 2] + rho_prev * (v[base + 2] + v[base + 3]);
                        a[m] = f0;
                        d[m] = f1 + f0;
                        let e = s << 1;
                        v[e] = f0;
                        v[e | 1] = f1;
                    }
                    if kernel {
                        quad_slot_k(&suffix_j[s], a, d, &mut acc);
                    } else {
                        quad_slot(&suffix_j[s], a, d, &mut acc);
                    }
                }
                for v in b.iter_mut() {
                    v.truncate(half << 1);
                }
                acc.map(Gf::from_wide)
            };
            #[cfg(feature = "parallel")]
            {
                let min_len = crate::piop::sumcheck::eq_factored::par_min_len(bufs.len(), half);
                bufs.par_iter_mut()
                    .with_min_len(min_len)
                    .map(fused)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                bufs.iter_mut().map(fused).collect()
            }
        } else {
            let compute = |b: &[Vec<Gf>; 4]| -> [Gf; 5] {
                let mut acc = wide5(&zero);
                for s in 0..half {
                    let e = s << 1;
                    let mut a = [zero; 4];
                    let mut d = [zero; 4];
                    for m in 0..4 {
                        let v0 = b[m][e];
                        a[m] = v0;
                        d[m] = b[m][e | 1] + v0;
                    }
                    if kernel {
                        quad_slot_k(&suffix_j[s], a, d, &mut acc);
                    } else {
                        quad_slot(&suffix_j[s], a, d, &mut acc);
                    }
                }
                acc.map(Gf::from_wide)
            };
            #[cfg(feature = "parallel")]
            {
                let min_len = crate::piop::sumcheck::eq_factored::par_min_len(bufs.len(), half);
                bufs.par_iter().with_min_len(min_len).map(compute).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                bufs.iter().map(compute).collect()
            }
        };
        let rho = quad_round_close(
            transcript,
            &hs,
            &mut a_scalars,
            &q_pt[j - 1],
            &nodes,
            &node_pows,
            kernel,
            false,
            &mut claimed_sum,
            &mut messages,
            &mut buf,
        );
        if j < k {
            pending_rho = Some(rho);
            randomness.push(rho);
        } else {
            // Finals, permuted from the slot order back to the CLOSE
            // order [Q00, Q10, Q01, Q11].
            let finals: Vec<[Gf; 4]> = bufs
                .iter()
                .map(|b| {
                    let interp = |v: &[Gf]| -> Gf { v[0] + rho * (v[0] + v[1]) };
                    [interp(&b[0]), interp(&b[2]), interp(&b[1]), interp(&b[3])]
                })
                .collect();
            randomness.push(rho);
            return (
                SumcheckProof {
                    messages,
                    claimed_sum,
                },
                randomness,
                finals,
            );
        }
    }
    unreachable!("the final round returns")
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crate::transcript::Blake3Transcript;

    fn sample(seed: u64) -> Gf {
        let hi = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0x1234_5678_9ABC_DEF0;
        Gf::from_polynomial_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    /// The bottom driver is transcript-, challenge- and finals-identical
    /// to the dense quad driver over the materialised leaf quarters —
    /// every gathered value is the exact field element the dense flow
    /// computes, and the product coefficients are pairing-invariant.
    #[test]
    fn bottom_matches_dense_quad() {
        for &(k, ngroups) in &[(4usize, 3usize), (5, 2), (6, 1), (7, 4)] {
            let q1b = 1usize << k;
            let q2b = q1b << 1;
            let one = Gf::one();
            let tau_l: Vec<Gf> = (0..q2b).map(|i| sample(0xA000 + i as u64)).collect();
            let tau_r: Vec<Gf> = (0..q2b).map(|i| sample(0xB000 + i as u64)).collect();
            // te/to: the 4-case pair-value tables ({1, v_lo, v_hi, v_lo·v_hi}).
            let cases = |base: usize| -> Vec<Gf> {
                let mut t = Vec::with_capacity(q1b << 2);
                for y in 0..q1b {
                    let vl = one + tau_l[base + y];
                    let vh = one + tau_r[base + y];
                    t.extend([one, vl, vh, vl * vh]);
                }
                t
            };
            let te = cases(0);
            let to = cases(q1b);
            // The unweighted 16-case ΔΔ table over position pairs of [0..2q1b).
            let mut t_dd = Vec::with_capacity(q1b << 4);
            for p in 0..q1b {
                let base = p << 1;
                for c in 0..16usize {
                    let mut v = Gf::zero();
                    for eps in 0..2usize {
                        for del in 0..2usize {
                            if (c >> eps) & 1 == 1 && (c >> (2 + del)) & 1 == 1 {
                                v += tau_l[base + eps] * tau_r[base + del];
                            }
                        }
                    }
                    t_dd.push(v);
                }
            }
            let words = q2b.div_ceil(64);
            let mkbits = |seed: u64| -> Vec<u64> {
                (0..words)
                    .map(|w| {
                        let x = sample(seed + w as u64).as_words()[0];
                        if q2b < 64 { x & ((1u64 << q2b) - 1) } else { x }
                    })
                    .collect()
            };
            let q_pt: Vec<Gf> = (0..k).map(|i| sample(0xC000 + i as u64)).collect();

            let mut groups_ref = Vec::new();
            let mut groups_bot = Vec::new();
            let bits: Vec<_> = (0..ngroups)
                .map(|t| {
                    (
                        mkbits(0xD000 + 97 * t as u64),
                        mkbits(0xE000 + 131 * t as u64),
                    )
                })
                .collect();
            for (t, (lbits, rbits)) in bits.iter().enumerate() {
                let scale = sample(0xF000 + t as u64);
                let quarter = |bits: &[u64], tau: &[Gf], base: usize| -> Vec<Gf> {
                    (0..q1b)
                        .map(|y| {
                            let i = base + y;
                            if (bits[i >> 6] >> (i & 63)) & 1 == 1 {
                                one + tau[i]
                            } else {
                                one
                            }
                        })
                        .collect()
                };
                let q00 = quarter(&lbits, &tau_l, 0);
                let q10 = quarter(&lbits, &tau_l, q1b);
                let q01 = quarter(&rbits, &tau_r, 0);
                let q11 = quarter(&rbits, &tau_r, q1b);
                groups_ref.push(QuadGroup {
                    q: q_pt.clone(),
                    scale,
                    bufs: [q00, q10, q01, q11],
                });
                groups_bot.push(QuadBitGroup {
                    scale,
                    lbits,
                    rbits,
                });
            }

            let mut t_ref = Blake3Transcript::new();
            let (p_ref, r_ref, f_ref) = prove_quad_eq_sumcheck(&mut t_ref, groups_ref);
            let mut t_bot = Blake3Transcript::new();
            let tabs = QuadBottomTables {
                te: &te,
                to: &to,
                t_dd: &t_dd,
                tau_l: &tau_l,
                tau_r: &tau_r,
            };
            let (p_bot, r_bot, f_bot) =
                prove_quad_bottom_sumcheck(&mut t_bot, q_pt.clone(), groups_bot, &tabs);

            assert_eq!(r_ref, r_bot, "challenges diverge at k={k}");
            assert_eq!(f_ref, f_bot, "finals diverge at k={k}");
            assert_eq!(
                p_ref.claimed_sum, p_bot.claimed_sum,
                "claimed sums diverge at k={k}"
            );
            assert_eq!(p_ref.messages.len(), p_bot.messages.len());
            // Equal transcript states ⇒ equal subsequent draws.
            let c_ref: Gf = t_ref.get_field_challenge(&());
            let c_bot: Gf = t_bot.get_field_challenge(&());
            assert_eq!(c_ref, c_bot, "transcript states diverge at k={k}");
        }
    }
}
