//! Merged product forest — the multi-instance GKR ("tree index as MLE
//! variables") replacing the per-tree layer evals of
//! `crate::piop::lookup::gkr_product` for the f2-int pipeline.
//!
//! The per-tree forest carries 2 K-elements per tree per layer
//! (2·2^s·d ≈ 512 KiB at n=26). Here the 2^s trees' layers form ONE
//! multilinear family L_ℓ over (x ∈ {0,1}^ℓ, c ∈ {0,1}^s) (x = in-tree
//! index, LOW bits; c = tree index, HIGH bits) with the TOP-bit product
//! recursion `L_ℓ(x, c) = L_{ℓ+1}(x, 0, c)·L_{ℓ+1}(x, 1, c)`; the verifier
//! enters by evaluating the published roots' MLE at a random ζ ∈ K^s
//! ITSELF, and each layer is reduced by sumcheck — payload O(s+ℓ) per
//! layer instead of O(2^s).
//!
//! **Layer structure (v3, grouped)**: the layer-ℓ claim `L̂_ℓ(z)` with
//! `z = (z_x ∈ K^ℓ, z_c ∈ K^s)` factors coordinate-wise as
//!
//! ```text
//!   Σ_{x,c} eq(x,z_x)·eq(c,z_c)·E_c(x)·O_c(x)
//!     = Σ_c eq(c,z_c)·[ Σ_x eq(x,z_x)·E_c(x)·O_c(x) ]
//! ```
//!
//! — a shared-`q` multi-group instance of the eq-factored driver
//! ([`prove_eq_inner_sumcheck_mixed`]): **phase A** runs the trees as
//! groups (q = z_x shared, per-group scale eq(c,z_c)) binding the ℓ
//! in-tree variables with the SAME kernels as the per-tree forest
//! (WideMulAcc delayed reduction, the fused single-pair round, and the
//! bit-affine case-LUT leaf round — the leaf layer is never
//! materialised); **phase B** binds the s tree-index variables with one
//! more (tiny) driver call over the per-group finals, which stay
//! prover-internal — nothing per-tree is ever absorbed. A closing
//! child pair + line challenge μ chain to the next layer at
//! `z' = (r_x ++ [μ], r_c)`.
//!
//! Exit claim: `L̂_d(z)` with `z = (z_bj ∈ K^{t+log₂W}, z_c ∈ K^s)`. For
//! the f2-int leaves `1 + M·(A−1)` this gives `e_d − 1 =
//! Σ_{c,bj} eq(c,z_c)·eq(bj,z_bj)·M[c][bj]·A(bj)` — exactly the batched
//! claim the pre-sumcheck consumes, with (z_c, z_bj) replacing the old
//! (ξ, ρ).

pub mod schedule;
use core::mem::MaybeUninit;
use schedule::{ForestPath, configured};

use crate::pcs::IntegerMatrixLayout;
use crate::piop::sumcheck::eq_factored::{
    EqInnerGroupMixed, FlatDense, GroupBufs, PRFM_DIST, Pair2TauSet, PreRound, SharedPointInput,
    SuffixTensorArena, prove_eq_inner_sumcheck_mixed_gruen, prove_eq_inner_sumcheck_mixed_pre,
    prove_eq_inner_sumcheck_mixed_prepared, suffix_tensors, verify_eq_inner_sumcheck_gruen,
};
use crate::piop::sumcheck::{MLSumcheck, SumcheckProof};
use crate::poly::univariate::binary_gf128::Gf128 as Gf;
use crate::poly::utils::{build_eq_x_r_vec, eq_eval};
use crate::transcript::traits::Transcript;
use crate::utils::wide_mul::WideMulAcc;
use crate::utils::{cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// One merged layer: the phase-A (in-tree variables; `None` at layer 0,
/// which has none) and phase-B (tree-index variables) sumchecks, plus the
/// closing child pair.
#[derive(Clone, Debug)]
pub struct MergedLayer {
    pub sc_x: Option<SumcheckProof<Gf>>,
    pub sc_c: SumcheckProof<Gf>,
    pub pair: (Gf, Gf),
    /// QUAD layers (arity 4, `BITZ_QUAD=1`): the second half of the
    /// closing quad — `pair = (Q00, Q10)`, `pair2 = (Q01, Q11)`, the four
    /// quarter evaluations of level ℓ+2 at the exit point. `None` on
    /// arity-2 layers.
    pub pair2: Option<(Gf, Gf)>,
}

/// Merged-forest proof (roots live in the caller's proof object).
#[derive(Clone, Debug)]
pub struct MergedForestProof {
    pub layers: Vec<MergedLayer>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergedForestError {
    Shape,
    /// A layer's sumcheck rejected or a chaining check failed.
    LayerClaim {
        layer: usize,
    },
}

#[allow(clippy::arithmetic_side_effects)]
fn absorb_gfs(transcript: &mut impl Transcript, tag: u8, vals: &[Gf]) {
    let mut bytes = Vec::with_capacity(vals.len() * 16 + 1);
    bytes.push(tag);
    for v in vals {
        let w = v.as_words();
        bytes.extend_from_slice(&w[0].to_le_bytes());
        bytes.extend_from_slice(&w[1].to_le_bytes());
    }
    transcript.absorb_slice(&bytes);
}

/// Evaluate the multilinear with table `tbl` (bit k ↔ point[k]) at `point`.
#[allow(clippy::arithmetic_side_effects)]
fn mle_at(tbl: &[Gf], point: &[Gf]) -> Gf {
    debug_assert_eq!(tbl.len(), 1usize << point.len());
    let mut buf = tbl.to_vec();
    for &r in point {
        let half = buf.len() >> 1;
        for i in 0..half {
            let (u, v) = (buf[2 * i], buf[2 * i + 1]);
            buf[i] = u + r * (u + v);
        }
        buf.truncate(half);
    }
    buf[0]
}

/// Per-tree levels of the upper product tree, LEVEL-major:
/// `levels[ℓ−1][t]` = tree `t`'s level-`ℓ` values as TOP-bit-split
/// `(first half, second half)` — level ℓ has `2^ℓ` values, halves of
/// `2^{ℓ−1}`. Levels run `ℓ = 1..=depth−1`; the leaf level (`2^depth`) is
/// supplied lazily by the layer-(d−1) groups.
type TreeLevels = Vec<Vec<(Vec<Gf>, Vec<Gf>)>>;

/// Parent level's halves from a child's: values `v[i] = L[i]·R[i]`,
/// re-split at the midpoint (top bit of the parent index).
#[allow(clippy::arithmetic_side_effects)]
fn parent_halves_top(l: &[Gf], r: &[Gf]) -> (Vec<Gf>, Vec<Gf>) {
    let n = l.len();
    let h = n >> 1;
    (
        (0..h).map(|i| l[i] * r[i]).collect(),
        (h..n).map(|i| l[i] * r[i]).collect(),
    )
}

/// Build every tree's levels `1..=top` from its level-`top` halves
/// (`gen_top`), parallel across trees, then regroup LEVEL-major. Also
/// returns the roots. `top = depth` materialises everything (eager);
/// `top < depth` leaves the upper levels to bit-driven layers.
#[allow(clippy::arithmetic_side_effects)]
fn build_levels(
    num_trees: usize,
    top: usize,
    gen_top: impl Fn(usize) -> (Vec<Gf>, Vec<Gf>) + Sync,
) -> (TreeLevels, Vec<Gf>) {
    // Per-tree chains [level top, top−1, …, 1], parallel across trees.
    let _g = tracing::info_span!("mf:build_levels").entered();
    let chains: Vec<Vec<(Vec<Gf>, Vec<Gf>)>> = cfg_into_iter!(0..num_trees)
        .map(|c| {
            let mut chain = Vec::with_capacity(top);
            chain.push(gen_top(c));
            for _ in 1..top {
                let last = chain.last().expect("non-empty chain");
                let parent = parent_halves_top(&last.0, &last.1);
                chain.push(parent);
            }
            chain
        })
        .collect();
    let mut levels: TreeLevels = (0..top).map(|_| Vec::with_capacity(num_trees)).collect();
    let mut roots = Vec::with_capacity(num_trees);
    for chain in chains {
        {
            let l1 = chain.last().expect("chain has level 1");
            roots.push(l1.0[0] * l1.1[0]);
        }
        for (i, lvl) in chain.into_iter().enumerate() {
            // chain[i] = level (top−i)  →  slot (top−1−i).
            levels[top - 1 - i].push(lvl);
        }
    }
    (levels, roots)
}

/// The stored upper levels, either per-tree ([`TreeLevels`]) or as ONE
/// flat store per level ([`FlatDense`], slot ℓ−1 = level ℓ, stride
/// `2^{ℓ−1}` per side, segments = real trees only). The flat form replaces
/// `2·top` allocations per tree with 2 per level — the wide-shallow
/// forest's `build_levels` allocation floor — and its level-major
/// construction streams contiguously. Same products, same values.
pub(crate) enum ForestLevels {
    PerTree(TreeLevels),
    Flat(Vec<Option<FlatDense<Gf>>>),
}

/// Flat-forest gate: `BITZ_FLAT_FOREST=0/1` forces the per-tree/flat
/// stored-level + driver path; unset (the default) engages flat exactly
/// on the wide-shallow half (`s ≥ depth`) it was built for. Byte-identical
/// either way — the layout changes storage, not values.
///
/// Why not everywhere: each flat level/JIT store is one FRESH multi-GB
/// region, and deep-narrow shapes make them huge — measured at n=30
/// 18:12 (2026-08-27, /usr/bin/time -l): 3.04M page reclaims / 12.2 GB
/// footprint flat vs 0.43M / 6.0 GB per-tree (≈48 GB of zero-fill
/// first-touches; equal instructions retired), prove 4.59 s vs 1.71 s —
/// a pure VM-churn tax that also inflates untouched scopes (mats_tile
/// 4.7×). The per-tree path's thousands of uniform smaller blocks
/// recycle through malloc instead. Wide-shallow inverts it: the
/// per-group allocation floor dominates and flat wins (n=30 15:15
/// −14 % same-window) with the stores 8× smaller. A per-prove arena
/// that recycles one region across levels/JIT would lift the gate.
/// Read once per process.
fn flat_forest(s: usize, depth: usize) -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| match std::env::var("BITZ_FLAT_FOREST") {
        Ok(v) if v == "0" => Some(false),
        Ok(v) if v == "1" => Some(true),
        _ => None,
    });
    env.unwrap_or(s >= depth)
}

/// Allocates an uninitialized `Gf` store without ever creating references to
/// uninitialized `Gf` values. Builders write every slot through
/// [`MaybeUninit::write`] and convert only after full coverage.
fn gf_uninit(n: usize) -> Vec<MaybeUninit<Gf>> {
    let mut values = Vec::with_capacity(n);
    values.resize_with(n, MaybeUninit::uninit);
    values
}

/// Converts a fully initialized temporary store into its final `Gf` vector.
///
/// # Safety
///
/// Every element of `values` must have been initialized exactly once.
unsafe fn gf_assume_init(mut values: Vec<MaybeUninit<Gf>>) -> Vec<Gf> {
    let pointer = values.as_mut_ptr().cast::<Gf>();
    let length = values.len();
    let capacity = values.capacity();
    core::mem::forget(values);
    // SAFETY: the caller guarantees all `length` elements are initialized;
    // `MaybeUninit<Gf>` has the same layout as `Gf`.
    unsafe { Vec::from_raw_parts(pointer, length, capacity) }
}

/// [`build_levels`] in the flat layout: level `top` is generated per tree
/// straight into its segment (`gen_top(c, l_seg, r_seg)`), each lower
/// level is one level-major elementwise pass (`parent_l[i] = cl[i]·cr[i]`,
/// `parent_r[i] = cl[i+h]·cr[i+h]` — [`parent_halves_top`]'s exact
/// products). Only live trees have storage.
/// Returns the slots plus the live trees' roots.
#[allow(clippy::arithmetic_side_effects)]
fn build_levels_flat(
    live: usize,
    top: usize,
    gen_top: impl Fn(usize, &mut [MaybeUninit<Gf>], &mut [MaybeUninit<Gf>]) + Sync,
) -> (Vec<Option<FlatDense<Gf>>>, Vec<Gf>) {
    let _g = tracing::info_span!("mf:build_levels").entered();
    let nseg = live;
    debug_assert!(top >= 1);
    let mut slots: Vec<Option<FlatDense<Gf>>> = (0..top).map(|_| None).collect();
    // Level `top`: generate live segments.
    let seg_top = 1usize << (top - 1);
    let mut lt = gf_uninit(nseg * seg_top);
    let mut rt = gf_uninit(nseg * seg_top);
    cfg_chunks_mut!(lt, seg_top)
        .zip(cfg_chunks_mut!(rt, seg_top))
        .enumerate()
        .for_each(|(c, (lseg, rseg))| {
            gen_top(c, lseg, rseg);
        });
    // SAFETY: the chunk traversal covers every segment, and each branch
    // initializes every slot in both stores.
    let lt = unsafe { gf_assume_init(lt) };
    // SAFETY: same coverage argument as for `lt`.
    let rt = unsafe { gf_assume_init(rt) };
    slots[top - 1] = Some(FlatDense {
        l: lt,
        r: rt,
        seg: seg_top,
    });
    // Levels top−1 .. 1: one elementwise pass each.
    for lvl in (1..top).rev() {
        let cseg = 1usize << lvl; // child (level lvl+1) stride
        let pseg = cseg >> 1; // parent (level lvl) stride
        let child = slots[lvl].as_ref().expect("child level built");
        let mut pl = gf_uninit(nseg * pseg);
        let mut pr = gf_uninit(nseg * pseg);
        cfg_chunks_mut!(pl, pseg)
            .zip(cfg_chunks_mut!(pr, pseg))
            .zip(cfg_chunks!(child.l, cseg))
            .zip(cfg_chunks!(child.r, cseg))
            .for_each(|(((plseg, prseg), cl), cr)| {
                for i in 0..pseg {
                    plseg[i].write(cl[i] * cr[i]);
                    prseg[i].write(cl[i + pseg] * cr[i + pseg]);
                }
            });
        // SAFETY: every parent chunk writes all `pseg` slots exactly once.
        let pl = unsafe { gf_assume_init(pl) };
        // SAFETY: same coverage argument as for `pl`.
        let pr = unsafe { gf_assume_init(pr) };
        slots[lvl - 1] = Some(FlatDense {
            l: pl,
            r: pr,
            seg: pseg,
        });
    }
    // Roots from level 1 (stride 1): live trees only.
    let l1 = slots[0].as_ref().expect("level 1 built");
    let roots: Vec<Gf> = (0..live).map(|t| l1.l[t] * l1.r[t]).collect();
    (slots, roots)
}

/// Where a level-(d−2) value comes from: the precombined 16-case `T4`
/// gather, or — probe I4 of `docs/lut-width-ideas.md`
/// (`BITZ_T4_FACTORED=1`) — the same product recomputed from the 4-case
/// `te`/`to` tables (`T4[y≪4|(cE≪2)|cO] = te[(y≪2)|cE]·to[(y≪2)|cO]`, the
/// build's own association): one multiply per value against two
/// line-local streams with half the footprint, in place of a
/// data-dependent 256-B-group line pick. Identical field elements either
/// way, so the transcript is byte-identical.
#[derive(Clone, Copy)]
enum T4Src<'a> {
    Pre(&'a [Gf]),
    Fact { te: &'a [Gf], to: &'a [Gf] },
}

impl T4Src<'_> {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn get(&self, j: usize, ce: usize, co: usize) -> Gf {
        match self {
            T4Src::Pre(t4) => t4[(j << 4) | (ce << 2) | co],
            T4Src::Fact { te, to } => te[(j << 2) | ce] * to[(j << 2) | co],
        }
    }
}

/// Factored-T4 knob: `BITZ_T4_FACTORED=0/1` forces precombined/factored
/// for the single-instance prover's `gen_top`/JIT consumers; unset (the
/// default) picks by schedule — factored on L/2 and L/4 (where `T4` is
/// then not built at all: −16·2^{d−2}·16 B footprint and the build
/// multiplies), precombined on L/8 (T4Bits needs the table anyway, and
/// its 4-gather `gen_top` amplifies the factored form's extra
/// multiplies). Measured (alternated in-window pairs): L/4 `gen_top`
/// −23% at n=28 (53.5→41.0 ms medians, far less volatile) and
/// build/bitgen −9..−12% at n=29, 3/3 pairs each; l8 at n=30 the JIT
/// wins −30% (2/3) but `gen_top` is wash-to-worse on a churned box.
/// Byte-identical either way (the build's own association is `te·to`).
/// Read once per process.
fn t4_factored() -> Option<bool> {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| match std::env::var("BITZ_T4_FACTORED") {
        Ok(v) if v == "0" => Some(false),
        Ok(v) if v == "1" => Some(true),
        _ => None,
    })
}

/// The [`T4Src`] for a consumer site under the resolved knob setting.
#[inline]
fn t4_src<'a>(fact: bool, t4: &'a [Gf], te: &'a [Gf], to: &'a [Gf]) -> T4Src<'a> {
    if fact {
        T4Src::Fact { te, to }
    } else {
        T4Src::Pre(t4)
    }
}

/// Level-(d−2) values straight from one tree's TRANSPOSED leaf-bit halves
/// and the 16-case 4-leaf table: position `j` selects
/// `t4[(j≪4) | (cE≪2) | cO]` with `cE = lbits.bit(j) | rbits.bit(j)≪1` and
/// `cO = lbits.bit(j+q1) | rbits.bit(j+q1)≪1` — leaves `{j, j+q2}` live in
/// the E/O halves at offset `j`, `{j+q1, j+q1+q2}` at offset `j+q1`.
/// Reading the per-tree packed halves word-wise costs 4 u64 loads per 64
/// values — ~64× less read traffic than per-bit `packed_col_bit` gathers
/// over the column-lane store, which touch one full word per bit (the
/// build/JIT gen was the dominant residual read stream at big n).
#[allow(clippy::arithmetic_side_effects)]
fn t4_level_values(lbits: &[u64], rbits: &[u64], t4: T4Src, q1: usize) -> Vec<Gf> {
    let mut out = Vec::with_capacity(q1);
    if q1 >= 64 {
        let wq = q1 >> 6;
        for w in 0..wq {
            let le = lbits[w];
            let ro = rbits[w];
            let lo = lbits[w + wq];
            let roo = rbits[w + wq];
            for b in 0..64 {
                let ce = (((le >> b) & 1) | (((ro >> b) & 1) << 1)) as usize;
                let co = (((lo >> b) & 1) | (((roo >> b) & 1) << 1)) as usize;
                out.push(t4.get((w << 6) | b, ce, co));
            }
        }
    } else {
        let bit = |bits: &[u64], p: usize| -> usize { ((bits[p >> 6] >> (p & 63)) & 1) as usize };
        for j in 0..q1 {
            let ce = bit(lbits, j) | (bit(rbits, j) << 1);
            let co = bit(lbits, j + q1) | (bit(rbits, j + q1) << 1);
            out.push(t4.get(j, ce, co));
        }
    }
    out
}

/// [`t4_level_values`] writing the TOP-bit halves directly — the L/2
/// build's `gen_top`. Same word-wise sweep (4 `u64` loads per 64 values),
/// but exact-capacity halves instead of one `2^{d−2}` buffer that would
/// then be sliced (a transient 2× on the biggest level the forest ever
/// stores).
#[allow(clippy::arithmetic_side_effects)]
fn t4_level_halves(lbits: &[u64], rbits: &[u64], t4: T4Src, q1: usize) -> (Vec<Gf>, Vec<Gf>) {
    let hh = q1 >> 1;
    let mut e = Vec::with_capacity(hh);
    let mut o = Vec::with_capacity(hh);
    if q1 >= 128 {
        // hh is a multiple of 64 here, so the word loop switches target
        // cleanly at the midpoint.
        let wq = q1 >> 6;
        for w in 0..wq {
            let le = lbits[w];
            let ro = rbits[w];
            let lo = lbits[w + wq];
            let roo = rbits[w + wq];
            let out = if (w << 6) < hh { &mut e } else { &mut o };
            for b in 0..64 {
                let ce = (((le >> b) & 1) | (((ro >> b) & 1) << 1)) as usize;
                let co = (((lo >> b) & 1) | (((roo >> b) & 1) << 1)) as usize;
                out.push(t4.get((w << 6) | b, ce, co));
            }
        }
    } else {
        let at = T4At {
            lbits,
            rbits,
            t4,
            q1,
        };
        e.extend((0..hh).map(|j| at.at(j)));
        o.extend((hh..q1).map(|j| at.at(j)));
    }
    (e, o)
}

/// A layer whose phase-A group buffers come straight from the committed
/// bits (with the shared tau tables the driver's case-LUT rounds consume)
/// instead of a materialised level.
struct BitLayer<'a> {
    bufs: Vec<GroupBufs<'a, Gf>>,
    tau_sets: Vec<(Vec<Gf>, Vec<Gf>)>,
    pair_tau_sets: Vec<Pair2TauSet<Gf>>,
    t4_sets: Vec<Vec<Gf>>,
    /// Round-1 coefficients `(A0, A1, A2)` per group, when the generation
    /// pass fused the round-1 message ([`dense_jit_fused_round1`], the JIT
    /// layers): forwarded to the driver so its round 1 absorbs the same
    /// bytes without a message pass — the buffers' first read is then
    /// round 2's fused fold+message pass.
    round1: Option<PreRound<Gf>>,
    /// Flat single-pair storage for the layer's groups (`bufs` is then
    /// empty and the driver runs all-[`GroupBufs::Flat`] markers over it —
    /// the const tail's segment included). The flat JIT layer's form.
    flat: Option<FlatDense<Gf>>,
}

// Keep schedule-specific leaf construction specialized at its original call sites.
#[inline(always)]
fn leaf_bit_layer<'a>(
    bits: Vec<(&'a [u64], &'a [u64])>,
    depth: usize,
    leaf_tau: &(Vec<Gf>, Vec<Gf>),
) -> BitLayer<'a> {
    let deep = depth >= 5 && forest_lut3();
    let deep4 = depth >= 6 && forest_lut4();
    BitLayer {
        bufs: bits
            .into_iter()
            .map(|(lbits, rbits)| {
                if deep4 {
                    GroupBufs::Leaf4Bits {
                        lbits,
                        rbits,
                        tau_set: 0,
                    }
                } else if deep {
                    GroupBufs::Leaf3Bits {
                        lbits,
                        rbits,
                        tau_set: 0,
                    }
                } else {
                    GroupBufs::Leaf2Bits {
                        lbits,
                        rbits,
                        tau_set: 0,
                    }
                }
            })
            .collect(),
        tau_sets: vec![leaf_tau.clone()],
        pair_tau_sets: Vec::new(),
        t4_sets: Vec::new(),
        round1: None,
        flat: None,
    }
}

/// Fuse the JIT layers' round-1 message into their generation pass
/// (default ON; `BITZ_JIT_R1=0` opts out — diagnostic / A-B measurement).
/// Byte-identical proofs either way. Read once per prove call.
fn jit_round1_fuse() -> bool {
    std::env::var("BITZ_JIT_R1").map_or(true, |v| v != "0")
}

/// Whether the JIT generation pass fuses the double-fold GRID (rounds 1
/// AND 2) rather than just round 1's coefficient triple — `BITZ_JIT_GRID=0`
/// keeps the triple. Byte-identical either way; this is the trade of nine
/// accumulators inside a gather-bound generation pass against one
/// streaming round-2 pass in the driver.
fn jit_grid() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BITZ_JIT_GRID").map_or(true, |v| v != "0"))
}

/// Per-position level-(d−2) reader off the bits + shared `T4` table — the
/// [`t4_level_values`] formula as a struct accessor (for the fused JIT
/// builders, whose access pattern is slot-paired rather than streaming):
/// position `j` selects `t4[(j≪4) | (cE≪2) | cO]` with
/// `cE = lbits.bit(j) | rbits.bit(j)≪1`, `cO` the same at `j + q1`.
///
/// A concrete struct with an `#[inline(always)]` accessor, NOT a returned
/// `impl Fn`: the opaque-closure form compiled to a real call per gather
/// at the build/JIT sites — millions per prove, measured ~5% of n=28
/// prove-side samples in `<&F as FnMut>::call_mut`.
#[derive(Clone, Copy)]
struct T4At<'a> {
    lbits: &'a [u64],
    rbits: &'a [u64],
    t4: T4Src<'a>,
    q1: usize,
}

impl T4At<'_> {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn at(&self, j: usize) -> Gf {
        #[inline(always)]
        fn bit(bits: &[u64], p: usize) -> usize {
            ((bits[p >> 6] >> (p & 63)) & 1) as usize
        }
        let ce = bit(self.lbits, j) | (bit(self.rbits, j) << 1);
        let co = bit(self.lbits, j + self.q1) | (bit(self.rbits, j + self.q1) << 1);
        self.t4.get(j, ce, co)
    }

    /// `prfm pldl1keep` for position `j`'s table line — the same index
    /// computation as [`Self::at`], issued ahead of use ([`t4_prfm`]).
    /// Precombined source only: the factored streams are line-local in
    /// the position, which the hardware prefetcher already covers.
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn prefetch_at(&self, j: usize) {
        let T4Src::Pre(t4) = self.t4 else { return };
        #[inline(always)]
        fn bit(bits: &[u64], p: usize) -> usize {
            ((bits[p >> 6] >> (p & 63)) & 1) as usize
        }
        let ce = bit(self.lbits, j) | (bit(self.rbits, j) << 1);
        let co = bit(self.lbits, j + self.q1) | (bit(self.rbits, j + self.q1) << 1);
        crate::piop::sumcheck::eq_factored::prefetch_l1(t4, (j << 4) | (ce << 2) | co);
    }
}

#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
fn t4_parent_value(at: &T4At<'_>, y: usize, half: usize, prefetch: bool) -> Gf {
    if prefetch {
        let ahead = y + PRFM_DIST;
        if ahead < half {
            at.prefetch_at(ahead);
            at.prefetch_at(ahead + half);
        }
    }
    at.at(y) * at.at(y + half)
}

/// Software prefetch on the T4-gather build/JIT sites (`gen_top` and the
/// JIT regeneration): the per-position 16-case line pick is
/// data-dependent (committed bits), which defeats the hardware
/// prefetcher, but the indices are cheaply recomputable ahead.
/// `BITZ_T4_PRFM=0/1` forces off/on; unset (the default) turns on iff the
/// shared `t4` table is ≥ 16 MiB (past the P-cluster L2 — the n ≥ 30
/// regime; at n ≤ 28 the table is L2-resident and the recompute overhead
/// loses, as measured for the stash-gather sites). Semantically inert.
/// Env read once per process.
fn t4_prfm(t4_bytes: usize) -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| match std::env::var("BITZ_T4_PRFM") {
        Ok(v) if v == "0" => Some(false),
        Ok(v) if v == "1" => Some(true),
        _ => None,
    });
    env.unwrap_or(t4_bytes >= 16 << 20)
}

/// Build one tree's Dense `(E, O)` halves from a per-position value
/// generator AND accumulate that tree's round-1 coefficients
/// `(A0, A1, A2)` in the same pass — the fused-JIT replacement for
/// "materialise, then run the round-1 message pass over what was just
/// written". `value(j)` supplies position `j ∈ 0..2·hh` (`E` half below
/// `hh`, `O` half above); `v1` is the layer's round-1 suffix tensor
/// `V_1` (length `hh/2` — [`suffix_tensors`]'s `tensor(0)` over the layer's `q`,
/// the SAME tensor the driver builds for its remaining rounds).
///
/// Value-exact vs the driver's round-1 kernel over the same buffers: the
/// identical weight-folds (`w·l0`, `w·l1` reduced) and the identical wide
/// products, XOR-accumulated (order-free) and reduced once per
/// accumulator — reduction is `F_2`-linear, so the chain structure does
/// not affect the reduced coefficients. With pass fusion the driver then
/// first reads these buffers in round 2's fused fold+message pass: the
/// generation write is the layer's ONLY full-buffer pass before folding.
#[allow(clippy::arithmetic_side_effects)]
fn dense_jit_fused_round1(
    hh: usize,
    v1: &[Gf],
    value: impl Fn(usize) -> Gf,
    look: impl Fn(usize),
) -> ((Vec<Gf>, Vec<Gf>), (Gf, Gf, Gf)) {
    let mut l = gf_uninit(hh);
    let mut r = gf_uninit(hh);
    let res = dense_jit_fused_round1_into(hh, v1, value, look, &mut l, &mut r);
    // SAFETY: the round-1 builder writes both entries of every pair, covering
    // exactly `0..hh` in each store.
    let l = unsafe { gf_assume_init(l) };
    // SAFETY: same coverage argument as for `l`.
    let r = unsafe { gf_assume_init(r) };
    ((l, r), res)
}

/// [`dense_jit_fused_round1`] writing into caller-provided segments (the
/// flat JIT layer) — identical operations in identical order.
#[allow(clippy::arithmetic_side_effects)]
fn dense_jit_fused_round1_into(
    hh: usize,
    v1: &[Gf],
    value: impl Fn(usize) -> Gf,
    look: impl Fn(usize),
    l: &mut [MaybeUninit<Gf>],
    r: &mut [MaybeUninit<Gf>],
) -> (Gf, Gf, Gf) {
    let half = hh >> 1;
    debug_assert_eq!(hh & 1, 0, "round-1 store consists of complete pairs");
    debug_assert_eq!(v1.len(), half, "V_1 tensor length = round-1 slot count");
    let zero = Gf::zero();
    let mut a0 = Gf::wide_zero(&zero);
    let mut a1 = Gf::wide_zero(&zero);
    let mut a2 = Gf::wide_zero(&zero);
    for b in 0..half {
        // Prefetch hint for slot b + PRFM_DIST's four positions (`look`
        // is a no-op when the caller's gate is off).
        if b + PRFM_DIST < half {
            let ep = (b + PRFM_DIST) << 1;
            look(ep);
            look(ep | 1);
            look(hh + ep);
            look(hh + ep + 1);
        }
        let e = b << 1;
        let l0 = value(e);
        let l1 = value(e | 1);
        let r0 = value(hh + e);
        let r1 = value(hh + e + 1);
        l[e].write(l0);
        l[e | 1].write(l1);
        r[e].write(r0);
        r[e | 1].write(r1);
        // The dense single-pair round-1 slot (weight folded into L).
        let w = v1[b];
        let l0w = w * l0;
        let l1w = w * l1;
        let wc0 = Gf::mul_wide(&l0w, &r0);
        let w11 = Gf::mul_wide(&l1w, &r1);
        let dr = r1 - r0;
        let dl = l1w - l0w;
        let wc2 = Gf::mul_wide(&dl, &dr);
        Gf::wide_add_assign(&mut a0, &wc0);
        Gf::wide_add_assign(&mut a2, &wc2);
        Gf::wide_add_assign(&mut a1, &w11);
        Gf::wide_sub_assign(&mut a1, &wc0);
        Gf::wide_sub_assign(&mut a1, &wc2);
    }
    (Gf::from_wide(a0), Gf::from_wide(a1), Gf::from_wide(a2))
}

/// [`dense_jit_fused_round1`] one variable deeper: the generation pass
/// accumulates the whole **bivariate grid** for rounds 1 AND 2, so the
/// driver spends both without reading the buffers at all and its first
/// pass over them (round 3) already folds two challenges. On the JIT
/// layer — the largest dense buffer the forest ever holds (`2^{n−2}`
/// values) — that removes one full streaming pass in each direction.
/// `v2` is the layer's round-2 suffix tensor `V_2` ([`suffix_tensors`]'s
/// `tensor(1)`,
/// length `hh/4`). Value-exact against the driver's own grid pass over
/// the same buffers (same weighted node grids, same wide products,
/// `F₂`-linear reduction).
#[allow(clippy::arithmetic_side_effects)]
fn dense_jit_fused_grid(
    hh: usize,
    v2: &[Gf],
    value: impl Fn(usize) -> Gf,
    look: impl Fn(usize),
) -> ((Vec<Gf>, Vec<Gf>), [Gf; 9]) {
    let mut l = gf_uninit(hh);
    let mut r = gf_uninit(hh);
    let g = dense_jit_fused_grid_into(hh, v2, value, look, &mut l, &mut r);
    // SAFETY: the grid builder writes all four entries of every quad,
    // covering exactly `0..hh` in each store.
    let l = unsafe { gf_assume_init(l) };
    // SAFETY: same coverage argument as for `l`.
    let r = unsafe { gf_assume_init(r) };
    ((l, r), g)
}

/// [`dense_jit_fused_grid`] writing into caller-provided segments (the
/// flat JIT layer) — identical operations in identical order.
#[allow(clippy::arithmetic_side_effects)]
fn dense_jit_fused_grid_into(
    hh: usize,
    v2: &[Gf],
    value: impl Fn(usize) -> Gf,
    look: impl Fn(usize),
    l: &mut [MaybeUninit<Gf>],
    r: &mut [MaybeUninit<Gf>],
) -> [Gf; 9] {
    let quads = hh >> 2;
    debug_assert_eq!(hh & 3, 0, "grid store consists of complete quads");
    debug_assert_eq!(v2.len(), quads, "V_2 tensor length = round-2 quad count");
    let zero = Gf::zero();
    let mut acc = core::array::from_fn::<_, 9, _>(|_| Gf::wide_zero(&zero));
    for b in 0..quads {
        if b + PRFM_DIST < quads {
            let qp = (b + PRFM_DIST) << 2;
            for i in 0..4 {
                look(qp + i);
                look(hh + qp + i);
            }
        }
        let base = b << 2;
        let lv: [Gf; 4] = core::array::from_fn(|i| value(base + i));
        let rv: [Gf; 4] = core::array::from_fn(|i| value(hh + base + i));
        for i in 0..4 {
            l[base + i].write(lv[i]);
            r[base + i].write(rv[i]);
        }
        // Weight rides the L side; the node grids are pure differences.
        let w = v2[b];
        let lw: [Gf; 4] = core::array::from_fn(|i| w * lv[i]);
        let grid = |v: &[Gf; 4]| -> [Gf; 9] {
            let d00 = v[1] - v[0];
            let d01 = v[3] - v[2];
            [
                v[0],
                v[1],
                d00,
                v[2],
                v[3],
                d01,
                v[2] - v[0],
                v[3] - v[1],
                d01 - d00,
            ]
        };
        let lg = grid(&lw);
        let rg = grid(&rv);
        for (a, (x, y)) in acc.iter_mut().zip(lg.iter().zip(rg.iter())) {
            Gf::wide_add_assign(a, &Gf::mul_wide(x, y));
        }
    }
    let e: [Gf; 9] = acc.map(Gf::from_wide);
    // Node → X₁-monomial per x₂ node (a₁ = H(1) − H(0) − H(∞)).
    core::array::from_fn(|i| {
        let (u, v) = (i / 3, i % 3);
        let base = v * 3;
        match u {
            0 => e[base],
            2 => e[base + 2],
            _ => e[base + 1] - e[base] - e[base + 2],
        }
    })
}

/// A JIT layer's generation: build every tree's Dense halves and fuse as
/// much of the layer's opening sumcheck into that ONE pass as its width
/// allows — the rounds-1-and-2 bivariate grid under the double-fold
/// (`k ≥ 3`), else round 1's coefficient triple. `mk(c)` hands out tree
/// `c`'s per-position value reader and its prefetch hook.
#[allow(clippy::arithmetic_side_effects)]
fn jit_layer_generate<'a, V, L, MK>(
    hh: usize,
    zx: &[Gf],
    num_trees: usize,
    tensors: &SuffixTensorArena<Gf>,
    mk: MK,
) -> (Vec<GroupBufs<'a, Gf>>, Option<PreRound<Gf>>)
where
    MK: Fn(usize) -> (V, L) + Sync,
    V: Fn(usize) -> Gf,
    L: Fn(usize),
{
    if crate::piop::sumcheck::eq_factored::eqf_double() && zx.len() >= 3 && jit_grid() {
        let v2 = tensors.tensor(1);
        let generated: Vec<(GroupBufs<'_, Gf>, [Gf; 9])> = cfg_into_iter!(0..num_trees)
            .map(|c| {
                let (value, look) = mk(c);
                let (pair, g) = dense_jit_fused_grid(hh, v2, value, look);
                (GroupBufs::Dense(vec![pair]), g)
            })
            .collect();
        let mut bufs = Vec::with_capacity(num_trees);
        let mut grid = Vec::with_capacity(num_trees);
        for (b, g) in generated {
            bufs.push(b);
            grid.push(g);
        }

        (bufs, Some(PreRound::Grid(grid)))
    } else {
        let v1 = tensors.tensor(0);
        let generated: Vec<(GroupBufs<'_, Gf>, (Gf, Gf, Gf))> = cfg_into_iter!(0..num_trees)
            .map(|c| {
                let (value, look) = mk(c);
                let (pair, coeffs) = dense_jit_fused_round1(hh, v1, value, look);
                (GroupBufs::Dense(vec![pair]), coeffs)
            })
            .collect();
        let mut bufs = Vec::with_capacity(num_trees);
        let mut round1 = Vec::with_capacity(num_trees);
        for (b, c) in generated {
            bufs.push(b);
            round1.push(c);
        }

        (bufs, Some(PreRound::Coeffs(round1)))
    }
}

/// [`jit_layer_generate`] into ONE flat store: the same fused generators
/// write each real tree's segment directly. Padding has no storage.
/// Two allocations replace `2·live`.
#[allow(clippy::arithmetic_side_effects)]
fn jit_layer_generate_flat<V, L, MK>(
    hh: usize,
    zx: &[Gf],
    live: usize,
    tensors: &SuffixTensorArena<Gf>,
    mk: MK,
) -> (FlatDense<Gf>, PreRound<Gf>)
where
    MK: Fn(usize) -> (V, L) + Sync,
    V: Fn(usize) -> Gf,
    L: Fn(usize),
{
    let nseg = live;
    let mut l = gf_uninit(nseg * hh);
    let mut r = gf_uninit(nseg * hh);
    let pre = if crate::piop::sumcheck::eq_factored::eqf_double() && zx.len() >= 3 && jit_grid() {
        let v2 = tensors.tensor(1);
        let grids: Vec<[Gf; 9]> = cfg_chunks_mut!(l, hh)
            .zip(cfg_chunks_mut!(r, hh))
            .enumerate()
            .map(|(c, (lseg, rseg))| {
                let (value, look) = mk(c);
                dense_jit_fused_grid_into(hh, v2, value, look, lseg, rseg)
            })
            .collect();
        PreRound::Grid(grids)
    } else {
        let v1 = tensors.tensor(0);
        let coeffs: Vec<(Gf, Gf, Gf)> = cfg_chunks_mut!(l, hh)
            .zip(cfg_chunks_mut!(r, hh))
            .enumerate()
            .map(|(c, (lseg, rseg))| {
                let (value, look) = mk(c);
                dense_jit_fused_round1_into(hh, v1, value, look, lseg, rseg)
            })
            .collect();
        PreRound::Coeffs(coeffs)
    };
    // SAFETY: either fused branch partitions both stores into complete pairs
    // or quads and initializes every slot in every segment.
    let l = unsafe { gf_assume_init(l) };
    // SAFETY: same coverage argument as for `l`.
    let r = unsafe { gf_assume_init(r) };
    (FlatDense { l, r, seg: hh }, pre)
}

/// The shared layer loop: layer ℓ = phase A (trees as groups over the ℓ
/// in-tree variables; skipped at ℓ = 0) + phase B (one group over the s
/// tree-index variables) + the closing pair / line challenge.
/// `bit_layer(ℓ)` may supply layer ℓ's phase-A buffers from the bits
/// (LeafBits / Pair2Bits); every other layer consumes `levels[ℓ]`.
#[allow(clippy::arithmetic_side_effects)]
fn drive_grouped<'a>(
    transcript: &mut impl Transcript,
    roots: Vec<Gf>,
    mut levels: ForestLevels,
    mut bit_layer: impl FnMut(usize, &[Gf], &SuffixTensorArena<Gf>) -> Option<BitLayer<'a>>,
    depth: usize,
    s: usize,
    live: usize,
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    assert!(depth >= 1, "merged forest needs depth >= 1");
    let num_trees = roots.len();
    assert!(
        live >= 1 && live <= num_trees,
        "live columns must be in 1..=2^s"
    );
    // Elided trees are constant 1 (see `col_elide`); only their aggregate
    // equality weight enters phase A.
    let one = Gf::one();
    absorb_gfs(transcript, 0x30, &roots);
    let zeta: Vec<Gf> = transcript.get_field_challenges(s, &());
    let mut claim = mle_at(&roots, &zeta);

    let mut z_x: Vec<Gf> = Vec::new();
    let mut z_c: Vec<Gf> = zeta;
    let mut out_layers = Vec::with_capacity(depth);
    // Layer ℓ uses level ℓ+1 = levels[ℓ] unless bit-driven; consumed via
    // mem::take per slot (front-first order).
    for ell in 0..depth {
        // Phase A: bind the ℓ in-tree variables (ℓ ≥ 1).
        let (sc_x, r_x, e_vec, o_vec) = if ell == 0 {
            let (mut e, mut o) = match &mut levels {
                ForestLevels::PerTree(levels) => {
                    let lvl1 = core::mem::take(&mut levels[0]);
                    let mut e = Vec::with_capacity(num_trees);
                    let mut o = Vec::with_capacity(num_trees);
                    for (l, r) in lvl1 {
                        e.push(l[0]);
                        o.push(r[0]);
                    }
                    (e, o)
                }
                ForestLevels::Flat(slots) => {
                    // Stride-1 store: `l`/`r` are the live root halves.
                    let fs = slots[0].take().expect("level 1 present");
                    debug_assert_eq!(fs.seg, 1, "level-1 stride");
                    let (mut e, mut o) = (fs.l, fs.r);
                    e.truncate(live);
                    o.truncate(live);
                    (e, o)
                }
            };
            e.resize(num_trees, one);
            o.resize(num_trees, one);
            (None, Vec::new(), e, o)
        } else {
            let _g = tracing::info_span!("mf:phaseA").entered();
            let eq_zc = if z_c.is_empty() {
                vec![one]
            } else {
                build_eq_x_r_vec(&z_c, &()).expect("nonempty tree point")
            };
            // The elided tail's aggregate equality weight.
            let const_scale = eq_zc[live..].iter().fold(Gf::zero(), |a, &b| a + b);
            // Only real groups enter buffer processing. Padding is carried
            // algebraically by `constant_weight` in the Gruen sumcheck.
            let mk_groups = |bufs: Vec<GroupBufs<'a, Gf>>| -> Vec<EqInnerGroupMixed<'_, Gf>> {
                assert_eq!(bufs.len(), live);
                bufs.into_iter()
                    .enumerate()
                    .map(|(c, bufs)| EqInnerGroupMixed {
                        q: z_x.as_slice().into(),
                        scale: eq_zc[c],
                        bufs,
                    })
                    .collect()
            };
            let mk_groups_flat = |nseg: usize| -> Vec<EqInnerGroupMixed<'_, Gf>> {
                assert_eq!(nseg, live);
                (0..nseg)
                    .map(|c| EqInnerGroupMixed {
                        q: (if c == 0 { z_x.as_slice() } else { &[] }).into(),
                        scale: eq_zc[c],
                        bufs: GroupBufs::Flat,
                    })
                    .collect()
            };
            // The JIT first message and the sumcheck use the same suffixes.
            let prepared_suffix = suffix_tensors(&z_x, &());
            let (groups, tau_sets, pair_tau_sets, t4_sets, pre_round1, flat_store) =
                if let Some(bl) = {
                    let _g = tracing::info_span!("mf:bitgen").entered();
                    bit_layer(ell, &z_x, &prepared_suffix)
                } {
                    if let Some(fs) = bl.flat {
                        let nseg = fs.l.len() / fs.seg;
                        let groups = mk_groups_flat(nseg);
                        (
                            groups,
                            bl.tau_sets,
                            bl.pair_tau_sets,
                            bl.t4_sets,
                            bl.round1,
                            Some(fs),
                        )
                    } else {
                        let groups = mk_groups(bl.bufs);
                        (
                            groups,
                            bl.tau_sets,
                            bl.pair_tau_sets,
                            bl.t4_sets,
                            bl.round1,
                            None,
                        )
                    }
                } else {
                    match &mut levels {
                        ForestLevels::PerTree(levels) => {
                            let lvl = core::mem::take(&mut levels[ell]);
                            let bufs = lvl
                                .into_iter()
                                .map(|pair| GroupBufs::Dense(vec![pair]))
                                .collect();
                            let groups = mk_groups(bufs);
                            (groups, Vec::new(), Vec::new(), Vec::new(), None, None)
                        }
                        ForestLevels::Flat(slots) => {
                            let fs = slots[ell].take().expect("stored flat level");
                            let nseg = fs.l.len() / fs.seg;
                            let groups = mk_groups_flat(nseg);
                            (groups, Vec::new(), Vec::new(), Vec::new(), None, Some(fs))
                        }
                    }
                };
            let (sc, r_x, finals) = prove_eq_inner_sumcheck_mixed_prepared(
                transcript,
                SharedPointInput {
                    groups,
                    constant_weight: const_scale,
                },
                &tau_sets,
                &pair_tau_sets,
                &t4_sets,
                pre_round1,
                flat_store,
                true,
                &(),
                Some(prepared_suffix),
            );
            // Elided trees evaluate to (1,1) at every point; restore them
            // below in their original tree-index order for phase B.
            debug_assert_eq!(finals.len(), live);
            let mut e = Vec::with_capacity(num_trees);
            let mut o = Vec::with_capacity(num_trees);
            for f in finals.iter().take(live) {
                let (fe, fo) = f[0];
                e.push(fe);
                o.push(fo);
            }
            e.resize(num_trees, one);
            o.resize(num_trees, one);
            (Some(sc), r_x, e, o)
        };

        // Phase B: bind the s tree-index variables over the per-tree finals.
        let _g = tracing::info_span!("mf:phaseB").entered();
        let group_b = EqInnerGroupMixed {
            q: z_c.as_slice().into(),
            scale: one,
            bufs: GroupBufs::Dense(vec![(e_vec, o_vec)]),
        };
        let (sc_c, r_c, finals_b) =
            prove_eq_inner_sumcheck_mixed_gruen(transcript, vec![group_b], &[], &[], &[], &());
        let pair = finals_b[0][0];
        drop(_g);

        absorb_gfs(transcript, 0x32, &[pair.0, pair.1]);
        let mu: Gf = transcript.get_field_challenge(&());
        claim = pair.0 + mu * (pair.0 + pair.1);
        let mut nx = r_x;
        nx.push(mu);
        z_x = nx;
        z_c = r_c;
        out_layers.push(MergedLayer {
            sc_x,
            sc_c,
            pair,
            pair2: None,
        });
    }
    let mut z = z_x;
    z.extend_from_slice(&z_c);
    (roots, MergedForestProof { layers: out_layers }, z, claim)
}

/// Prove all `2^s` per-tree grand products over the flat leaf table
/// (`leaves[i + 2^d·c]`, in-tree index low, tree index high). Returns
/// `(roots, proof, exit_point (len d+s), exit_eval)`.
///
/// Eager reference: materialised leaves in, the leaf layer runs as Dense
/// groups. Byte-identical to [`prove_merged_forest_lazy`] over the same
/// leaf values (the driver's LeafBits round is an exact char-2 identity).
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_merged_forest(
    transcript: &mut impl Transcript,
    leaves: &[Gf],
    depth: usize,
    s: usize,
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    let num_trees = 1usize << s;
    assert_eq!(leaves.len(), 1usize << (depth + s), "leaf table shape");
    assert!(depth >= 1, "depth must be positive");
    let per = 1usize << depth;
    let half = per >> 1;
    let leaf_halves = |c: usize| -> (Vec<Gf>, Vec<Gf>) {
        let base = c << depth;
        (
            leaves[base..base + half].to_vec(),
            leaves[base + half..base + per].to_vec(),
        )
    };
    // Everything materialised: the leaf level is `levels[depth−1]`.
    let (levels, roots) = build_levels(num_trees, depth, leaf_halves);
    drive_grouped(
        transcript,
        roots,
        ForestLevels::PerTree(levels),
        |_, _, _| None,
        depth,
        s,
        num_trees,
    )
}

/// Lazy bit-affine merged-forest prover over the committed bits: the leaf
/// layer AND the first product level (d−1) are never materialised, and at
/// depth ≥ 4 the stored chain tops out at level d−3 (≈ L/4 instead of L/2)
/// — level d−3 is generated per tree straight from the lane-packed bits by
/// TOP-paired products of the 16-case 4-leaf table `T4`, level d−2 is
/// regenerated JIT (Dense) at its own layer, layer d−2's sumcheck runs the
/// driver's [`GroupBufs::Pair2Bits`] case-LUT round, and layer d−1's runs
/// the two-round bit-affine [`GroupBufs::Leaf2Bits`] round (dense buffers
/// only after round 2, ≈ L/4) — all over the SAME per-tree leaf bit halves
/// + shared tau tables. Under the L8 schedule every bottom
/// layer goes ONE round deeper from bits (T4Bits / Pair3Bits / Leaf3Bits,
/// build top d−4) — every stage ≈ L/8. `pow2` is
/// [`crate::pcs::chunk_pow2_table`]'s per-row α-power chains.
#[allow(clippy::arithmetic_side_effects)]
/// `live` is the number of leading columns that carry data — the trailing
/// `2^s − live` trees are known-constant and are elided (see
/// [`col_elide`]); pass `p.cols()` for the un-elided forest. Byte-identical
/// either way.
pub fn prove_merged_forest_lazy(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    packed_cols: &[Vec<u64>],
    pow2: &[Vec<Gf>],
    live: usize,
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    prove_merged_forest_lazy_sched(
        transcript,
        p,
        packed_cols,
        pow2,
        configured(p, ForestPath::Single).expect("single forest schedule"),
        live,
    )
}

pub(crate) fn prove_merged_forest_lazy_from_rows(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    rows: &[Vec<u64>],
    packed_cols: &[Vec<u64>],
    pow2: &[Vec<Gf>],
    live: usize,
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    prove_merged_forest_lazy_impl(
        transcript,
        p,
        Some(rows),
        packed_cols,
        pow2,
        configured(p, ForestPath::Single).expect("single forest schedule"),
        live,
    )
}

/// Where the stored level chain tops out — the forest's time/memory knob,
/// one notch per level. Every schedule is byte-identical: they differ only
/// in which levels are stored, regenerated, or read straight off the bits.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ForestSchedule {
    /// Build top d−2: level d−2 is STORED, so the layer-(d−3) JIT
    /// regeneration disappears. Peak ≈ `2^n·8 B` — double L/4, in
    /// exchange for one fewer full `T4` sweep. The RAM-for-time end of
    /// the knob (`F2_FOREST_SCHEDULE=l2`).
    L2,
    /// Build top d−3, level d−2 regenerated JIT at its own layer.
    /// Peak ≈ `2^n·4 B`.
    L4,
    /// Build top d−4 + one more bit-driven round per bottom layer. Peak
    /// ≈ `2^n·2 B` (`F2_FOREST_SCHEDULE=l8`).
    L8,
}

impl ForestSchedule {
    fn is_l8(self) -> bool {
        matches!(self, ForestSchedule::L8)
    }
}

/// Deeper L/4 bit-driven prefixes — the DEFAULT: one more LUT round per
/// bottom layer (Pair2Bits → Pair3Bits, Leaf2Bits → Leaf3Bits; depth ≥ 5)
/// WITHOUT changing the L/4 build top — both LUT materialization residues
/// halve and the following dense cascades start one round smaller (small
/// consistent win at DRAM-scale shapes on top of pass fusion; a wash at
/// cache-adjacent shapes). Byte-identical either way (every variant is an
/// exact char-2 identity, pinned against the eager forest). `BITZ_LUT3=0`
/// opts out. Read once per prove call.
pub(crate) fn forest_lut3() -> bool {
    std::env::var("BITZ_LUT3").map_or(true, |v| v != "0")
}

/// FOUR bit-driven leaf rounds (`Leaf4Bits` — probe I2 of
/// `docs/lut-width-ideas.md`): the leaf residue halves again
/// (`2^{d−4}`/side) with the shared-table footprint frozen at the 16-case
/// level (round 3's fold ρ₃-reweights the stashed sets instead of
/// building the width law's 256-case `F₃`). `BITZ_LUT4=1` opts in (needs
/// depth ≥ 6, i.e. leaf k ≥ 5); default off pending measurement.
pub(crate) fn forest_lut4() -> bool {
    std::env::var("BITZ_LUT4").is_ok_and(|v| v == "1")
}

/// **Live-column elision** — the padding lever. A committed column whose
/// bits are all zero folds to `u_c = 0`, so every leaf of its tree is
/// `α^0 = 1`, every level is 1, and its root is 1. Such a tree costs the
/// forest a full `2^d`-value pipeline (bit extraction, the `T4` build, the
/// JIT regen, and one dense group per layer) to prove a constant.
///
/// The driver's per-layer message is `Σ_c eq(z_c, c)·H_c(·)` with `H_c`
/// linear in the group's `scale`, and every constant-1 tree has the SAME
/// `H` — so the whole zero tail contributes the scalar
/// `C = Σ_{c ≥ live} eq(z_c, c)` without buffers. Char-2 addition
/// is XOR, so re-associating the sum is exact: the round polynomials, the
/// absorbed roots and the whole transcript are **byte-identical** to the
/// un-elided forest. Nothing moves on the verifier side.
///
/// A witness of `N` cells padded to `2^n` therefore pays the forest only
/// for `⌈N / 2^{t+log₂W}⌉` columns — the residual waste is under one
/// column (0.05 % of `2^28` at `t = 17`). `BITZ_COL_ELIDE=0` opts out.
pub(crate) fn col_elide() -> bool {
    std::env::var("BITZ_COL_ELIDE").map_or(true, |v| v != "0")
}

/// The number of leading columns the forest must actually build: the
/// trailing run of all-zero bit rows is elided (see [`col_elide`]). Scans
/// only the tail it elides — `2^n/64` word reads worst case, sub-ms at
/// `n = 28`. Always returns at least 1 (the driver needs one real group).
pub(crate) fn live_cols(p: &IntegerMatrixLayout, rows: &[Vec<u64>]) -> usize {
    let cols = p.cols();
    if !col_elide() || rows.len() < cols {
        return cols;
    }
    let mut live = cols;
    while live > 1 && rows[live - 1].iter().all(|&w| w == 0) {
        live -= 1;
    }
    live
}

/// Extend a `live`-length root list to the full `2^s` with the elided
/// trees' known root `α^0 = 1`, so the transcript absorb and the phase-B
/// sumcheck see the un-elided forest.
fn pad_roots(mut roots: Vec<Gf>, num_trees: usize) -> Vec<Gf> {
    roots.resize(num_trees, Gf::one());
    roots
}

/// [`prove_merged_forest_lazy`] with the schedule explicit — the testable
/// entry point; ALL schedules are pinned byte-identical to the eager
/// prover.
#[allow(clippy::arithmetic_side_effects)]
fn prove_merged_forest_lazy_sched(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    packed_cols: &[Vec<u64>],
    pow2: &[Vec<Gf>],
    sched: ForestSchedule,
    live: usize,
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    prove_merged_forest_lazy_impl(transcript, p, None, packed_cols, pow2, sched, live)
}

fn prove_merged_forest_lazy_impl(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    rows: Option<&[Vec<u64>]>,
    packed_cols: &[Vec<u64>],
    pow2: &[Vec<Gf>],
    sched: ForestSchedule,
    live: usize,
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    let l8 = sched.is_l8();
    use crate::pcs::{
        build_column_layer1_halves, extract_column_bit_halves, layer1_pair_table, leaf_tau_halves,
    };
    let log_w = p.word_bits.trailing_zeros() as usize;
    let mask_w = p.word_bits.wrapping_sub(1);
    let row_len = p.rows() << log_w;
    let depth = row_len.trailing_zeros() as usize;
    let s = p.col_vars;
    let num_trees = p.cols();
    // Only the `live` leading trees are generated; the driver accounts
    // for the constant tail analytically (see `col_elide`).
    let live = live.clamp(1, num_trees);
    let one = Gf::one();
    if depth < 3 {
        // Tiny trees: the LeafBits round needs k ≥ 2 — materialise.
        let dense: Vec<Gf> = (0..(row_len << s))
            .map(|idx| {
                let (c, i) = (idx >> depth, idx & (row_len - 1));
                if (packed_cols[c >> 6][i] >> (c & 63)) & 1 == 1 {
                    pow2[i >> log_w][i & mask_w]
                } else {
                    one
                }
            })
            .collect();
        return prove_merged_forest(transcript, &dense, depth, s);
    }

    let _g_pre = tracing::info_span!("mf:l1tabs").entered();
    let pair_tbl = layer1_pair_table(p, pow2, log_w, row_len);
    let leaf_tau = leaf_tau_halves(p, pow2, one, log_w, row_len);
    drop(_g_pre);
    let _g_ext = tracing::info_span!("mf:extract_bits").entered();
    let extracted;
    let halves: Vec<(&[u64], &[u64])> = if let Some(rows) = rows.filter(|_| row_len >= 128) {
        assert_eq!(rows.len(), p.cols(), "column count");
        rows[..live]
            .iter()
            .map(|row| {
                assert!(row.len() >= row_len / 64, "column bit length");
                row[..row_len / 64].split_at(row_len / 128)
            })
            .collect()
    } else {
        extracted = extract_column_bit_halves(packed_cols, live, row_len);
        extracted
            .iter()
            .map(|(l, r)| (l.as_slice(), r.as_slice()))
            .collect()
    };
    let mut col_bits = Some(halves);
    drop(_g_ext);

    if depth < 4 {
        // Only the leaf layer is bit-driven (Pair2Bits needs k = d−2 ≥ 2).
        let (levels, roots) = build_levels(live, depth - 1, |c| {
            build_column_layer1_halves(p, packed_cols, c, pow2, &pair_tbl, one, log_w, row_len)
        });
        let bit_layer =
            |ell: usize, _zx: &[Gf], _suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
                (ell == depth - 1).then(|| BitLayer {
                    bufs: col_bits
                        .take()
                        .expect("leaf bits consumed once")
                        .into_iter()
                        .map(|(lbits, rbits)| GroupBufs::LeafBits {
                            lbits,
                            rbits,
                            tau_set: 0,
                        })
                        .collect(),
                    tau_sets: vec![leaf_tau.clone()],
                    pair_tau_sets: Vec::new(),
                    t4_sets: Vec::new(),
                    round1: None,
                    flat: None,
                })
            };
        return drive_grouped(
            transcript,
            pad_roots(roots, num_trees),
            ForestLevels::PerTree(levels),
            bit_layer,
            depth,
            s,
            live,
        );
    }

    // The per-position 4-case VALUE tables of the level-(d−1) products:
    // E-side position y pairs leaves (y, y+2^{d−1}); O-side pairs
    // (y+2^{d−2}, y+3·2^{d−2}). Case = bit_lo | bit_hi≪1; case 3 comes
    // from the shared pair table.
    let q1 = row_len >> 2; // 2^{d−2}
    let q2 = row_len >> 1; // 2^{d−1}
    let v = |i: usize| -> Gf { pow2[i >> log_w][i & mask_w] };
    let _g_teto = tracing::info_span!("mf:teto").entered();
    let build_cases = |base: usize| -> Vec<Gf> {
        let mut t = Vec::with_capacity(q1 << 2);
        for y in 0..q1 {
            let lo = base + y;
            t.push(one);
            t.push(v(lo));
            t.push(v(lo + q2));
            t.push(pair_tbl[lo]);
        }
        t
    };
    let te = build_cases(0);
    let to = build_cases(q1);
    drop(_g_teto);

    // 4-leaf product table for building level d−2 straight from the bits:
    // T4[y≪4 | (cE≪2|cO)] = te[4y+cE]·to[4y+cO]. Collect the INDEXED
    // Vec<[Gf; 16]> and flatten in place: a parallel `.flatten()` here
    // treats every 16-entry row as its own nested parallel iterator and
    // the collect goes unindexed — measured ~836× slower than the same
    // arithmetic serially (upstream zinc-plus arithmetic fix).
    // Under [`t4_factored`] the consumers recompute te·to themselves, so
    // the table is only built where the T4Bits round needs it precombined
    // (the L/8 schedule).
    let t4f = t4_factored().unwrap_or(!(l8 && depth >= 5));
    let t4: Vec<Gf> = if t4f && !(l8 && depth >= 5) {
        Vec::new()
    } else {
        let rows: Vec<[Gf; 16]> = cfg_into_iter!(0..q1, 1 << 10)
            .map(|y| {
                let mut row = [Gf::one(); 16];
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = te[(y << 2) | (c >> 2)] * to[(y << 2) | (c & 3)];
                }
                row
            })
            .collect();
        rows.into_flattened()
    };

    if sched == ForestSchedule::L2 {
        // The L/2 schedule — RAM for time: the stored chain tops out at
        // level d−2 itself, so layer d−3 consumes a STORED Dense level
        // and the JIT regeneration (a second full `T4` sweep over every
        // tree, ~2^{n−2} gathers) disappears. `T4` dies with the build;
        // the pair and leaf layers are the L/4 ones verbatim (they read
        // `te`/`to`/`leaf_tau` and the bits, never a level). Peak ≈
        // `2^n·8 B`: level d−2 alone is `2^n·4 B` and the chain under it
        // sums to as much again.
        let (levels, roots) = build_levels(live, depth - 2, |c| {
            let cb = col_bits.as_ref().expect("leaf bits alive for the build");
            let (lb, rb) = &cb[c];
            t4_level_halves(lb, rb, t4_src(t4f, &t4, &te, &to), q1)
        });
        drop(t4);
        let bit_layer =
            |ell: usize, _zx: &[Gf], _suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
                if ell == depth - 2 {
                    let deep = depth >= 5 && forest_lut3();
                    Some(BitLayer {
                        bufs: col_bits
                            .as_ref()
                            .expect("leaf bits alive for the pair layer")
                            .iter()
                            .map(|(lbits, rbits)| {
                                let (lbits, rbits) = (*lbits, *rbits);
                                if deep {
                                    GroupBufs::Pair3Bits {
                                        lbits,
                                        rbits,
                                        tau_set: 0,
                                    }
                                } else {
                                    GroupBufs::Pair2Bits {
                                        lbits,
                                        rbits,
                                        tau_set: 0,
                                    }
                                }
                            })
                            .collect(),
                        tau_sets: Vec::new(),
                        pair_tau_sets: vec![Pair2TauSet {
                            te: te.clone(),
                            to: to.clone(),
                        }],
                        t4_sets: Vec::new(),
                        round1: None,
                        flat: None,
                    })
                } else if ell == depth - 1 {
                    Some(leaf_bit_layer(
                        col_bits.take().expect("leaf bits consumed once"),
                        depth,
                        &leaf_tau,
                    ))
                } else {
                    None
                }
            };
        return drive_grouped(
            transcript,
            pad_roots(roots, num_trees),
            ForestLevels::PerTree(levels),
            bit_layer,
            depth,
            s,
            live,
        );
    }

    if depth == 4 || !l8 {
        // The L/4 schedule — the DEFAULT (and forced at depth 4, where
        // the deeper chain degenerates: T4Bits needs k = d−3 ≥ 2,
        // Leaf3Bits k = d−1 ≥ 4): stored chain tops at level d−3 via
        // TOP-paired T4 products, level d−2 JIT, then Pair2Bits /
        // Leaf2Bits.
        let h3 = q1 >> 1; // level d−3 positions = 2^{d−3}
        let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
        let use_flat = flat_forest(s, depth);
        // Level d−3 straight from paired T4 gathers — the full
        // level-(d−2) buffer (2^{d−2} elements per tree) never
        // exists: each of its positions is gathered exactly once
        // here, so this is the same gather count with the
        // materialise-then-read round trip removed.
        let gen3_at = |c: usize| {
            let cb = col_bits.as_ref().expect("leaf bits alive for the build");
            let (lb, rb) = &cb[c];
            T4At {
                lbits: lb,
                rbits: rb,
                t4: t4_src(t4f, &t4, &te, &to),
                q1,
            }
        };
        let gen3_flat = |c: usize, lseg: &mut [MaybeUninit<Gf>], rseg: &mut [MaybeUninit<Gf>]| {
            let at = gen3_at(c);
            let hh = h3 >> 1;
            for y in 0..hh {
                lseg[y].write(t4_parent_value(&at, y, h3, t4_pf));
            }
            for y in hh..h3 {
                rseg[y - hh].write(t4_parent_value(&at, y, h3, t4_pf));
            }
        };
        let (levels_l4, roots) = if use_flat {
            let (slots, roots) = build_levels_flat(live, depth - 3, gen3_flat);
            (ForestLevels::Flat(slots), roots)
        } else {
            let (lv, roots) = build_levels(live, depth - 3, |c| {
                let at = gen3_at(c);
                let hh = h3 >> 1;
                let l = (0..hh)
                    .map(|y| t4_parent_value(&at, y, h3, t4_pf))
                    .collect();
                let r = (hh..h3)
                    .map(|y| t4_parent_value(&at, y, h3, t4_pf))
                    .collect();
                (l, r)
            });
            (ForestLevels::PerTree(lv), roots)
        };
        let mut t4 = t4;
        let jit_r1 = jit_round1_fuse();
        let bit_layer =
            |ell: usize, zx: &[Gf], suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
                if ell == depth - 3 {
                    // JIT: regenerate level d−2 (Dense) per tree from the
                    // bits + T4 — alive only while this layer runs. Default:
                    // the generation pass ALSO accumulates the layer's
                    // round-1 coefficients ([`dense_jit_fused_round1`]) so
                    // the driver's round-1 message pass never reads what was
                    // just written (byte-identical; `BITZ_JIT_R1=0` opts out).
                    let hh = q1 >> 1;
                    let cb = col_bits
                        .as_ref()
                        .expect("leaf bits alive for the JIT regen");
                    let (bufs, round1, flat) = if use_flat {
                        if jit_r1 {
                            let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
                            let (fs, pre) = jit_layer_generate_flat(hh, zx, live, suffix, |c| {
                                let (lb, rb) = &cb[c];
                                let at = T4At {
                                    lbits: lb,
                                    rbits: rb,
                                    t4: t4_src(t4f, &t4, &te, &to),
                                    q1,
                                };
                                (
                                    move |j| at.at(j),
                                    move |j| {
                                        if t4_pf {
                                            at.prefetch_at(j);
                                        }
                                    },
                                )
                            });
                            (Vec::new(), Some(pre), Some(fs))
                        } else {
                            // Diagnostic (`BITZ_JIT_R1=0`): plain flat generation,
                            // no fused round 1 — same values, flat segments.
                            let nseg = live;
                            let mut l = gf_uninit(nseg * hh);
                            let mut r = gf_uninit(nseg * hh);
                            cfg_chunks_mut!(l, hh)
                                .zip(cfg_chunks_mut!(r, hh))
                                .enumerate()
                                .for_each(|(c, (lseg, rseg))| {
                                    let (lb, rb) = &cb[c];
                                    let full =
                                        t4_level_values(lb, rb, t4_src(t4f, &t4, &te, &to), q1);
                                    for (slot, &value) in lseg.iter_mut().zip(&full[..hh]) {
                                        slot.write(value);
                                    }
                                    for (slot, &value) in rseg.iter_mut().zip(&full[hh..]) {
                                        slot.write(value);
                                    }
                                });
                            // SAFETY: every flat segment is fully initialized by
                            // the live-column copy.
                            let l = unsafe { gf_assume_init(l) };
                            // SAFETY: same coverage argument as for `l`.
                            let r = unsafe { gf_assume_init(r) };
                            (Vec::new(), None, Some(FlatDense { l, r, seg: hh }))
                        }
                    } else if jit_r1 {
                        let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
                        let (bufs, round1) = jit_layer_generate(hh, zx, live, suffix, |c| {
                            let (lb, rb) = &cb[c];
                            let at = T4At {
                                lbits: lb,
                                rbits: rb,
                                t4: t4_src(t4f, &t4, &te, &to),
                                q1,
                            };
                            (
                                move |j| at.at(j),
                                move |j| {
                                    if t4_pf {
                                        at.prefetch_at(j);
                                    }
                                },
                            )
                        });
                        (bufs, round1, None)
                    } else {
                        let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..live)
                            .map(|c| {
                                let (lb, rb) = &cb[c];
                                // Exact-capacity halves: these buffers live (and
                                // get truncate()-folded, which never releases
                                // capacity) through the whole layer — a split_off
                                // would carry a 2× allocation.
                                let full = t4_level_values(lb, rb, t4_src(t4f, &t4, &te, &to), q1);
                                GroupBufs::Dense(vec![(full[..hh].to_vec(), full[hh..].to_vec())])
                            })
                            .collect();
                        (bufs, None, None)
                    };
                    t4 = Vec::new();
                    Some(BitLayer {
                        bufs,
                        tau_sets: Vec::new(),
                        pair_tau_sets: Vec::new(),
                        t4_sets: Vec::new(),
                        round1,
                        flat,
                    })
                } else if ell == depth - 2 {
                    // Under `BITZ_LUT3` (depth ≥ 5, so k = d−2 ≥ 3) the pair
                    // layer runs one more bit-driven round (Pair3Bits): its
                    // materialized residue halves. Both layers borrow the same
                    // immutable packed bits.
                    let deep = depth >= 5 && forest_lut3();
                    Some(BitLayer {
                        bufs: cfg_iter!(
                            col_bits
                                .as_ref()
                                .expect("leaf bits alive for the pair layer")
                        )
                        .map(|(lbits, rbits)| {
                            let (lbits, rbits) = (*lbits, *rbits);
                            if deep {
                                GroupBufs::Pair3Bits {
                                    lbits,
                                    rbits,
                                    tau_set: 0,
                                }
                            } else {
                                GroupBufs::Pair2Bits {
                                    lbits,
                                    rbits,
                                    tau_set: 0,
                                }
                            }
                        })
                        .collect(),
                        tau_sets: Vec::new(),
                        pair_tau_sets: vec![Pair2TauSet {
                            te: te.clone(),
                            to: to.clone(),
                        }],
                        t4_sets: Vec::new(),
                        round1: None,
                        flat: None,
                    })
                } else if ell == depth - 1 {
                    // Two bit-driven rounds (k = d−1 = 3): dense buffers only
                    // after round 2. Under `BITZ_LUT3` (depth ≥ 5, so k ≥ 4)
                    // three rounds (Leaf3Bits): the leaf residue halves.
                    Some(leaf_bit_layer(
                        col_bits.take().expect("leaf bits consumed once"),
                        depth,
                        &leaf_tau,
                    ))
                } else {
                    None
                }
            };
        return drive_grouped(
            transcript,
            pad_roots(roots, num_trees),
            levels_l4,
            bit_layer,
            depth,
            s,
            live,
        );
    }

    // depth ≥ 5 — the L/8 schedule: the stored chain tops out at level
    // d−4 (TOP-paired products of T4 pairs), layer d−4 JIT-regenerates
    // level d−3, and layers d−3 / d−2 / d−1 run 1 / 2 / 3 bit-driven
    // rounds (T4Bits / Pair3Bits / Leaf3Bits) — every stage's resident
    // set is ≈ L/8. T4 stays alive through the T4Bits layer (moved into
    // its BitLayer, zero-copy).
    let h3 = q1 >> 1; // level d−3 positions = 2^{d−3}
    let h4 = q1 >> 2; // level d−4 positions = 2^{d−4}
    let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
    let (levels, roots) = build_levels(live, depth - 4, |c| {
        let cb = col_bits.as_ref().expect("leaf bits alive for the build");
        let (lb, rb) = &cb[c];
        // Level d−4 straight from T4 gathers (each position once) — the
        // full level-(d−2) buffer never exists (see the L/4 build).
        let at = T4At {
            lbits: lb,
            rbits: rb,
            t4: t4_src(t4f, &t4, &te, &to),
            q1,
        };
        let v4 = |y: usize| -> Gf {
            if t4_pf {
                let yp = y + PRFM_DIST;
                if yp < h4 {
                    at.prefetch_at(yp);
                    at.prefetch_at(yp + h3);
                    at.prefetch_at(yp + h4);
                    at.prefetch_at(yp + h4 + h3);
                }
            }
            (at.at(y) * at.at(y + h3)) * (at.at(y + h4) * at.at(y + h4 + h3))
        };
        let hh = h4 >> 1;
        ((0..hh).map(v4).collect(), (hh..h4).map(v4).collect())
    });
    let mut t4 = t4;
    let jit_r1 = jit_round1_fuse();

    let bit_layer = |ell: usize, zx: &[Gf], suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
        if ell == depth - 4 {
            // JIT: regenerate level d−3 (Dense, exact-capacity halves)
            // per tree — TOP-paired T4 products; transient ≈ L/8. The
            // default fuses the round-1 message into this generation
            // pass, exactly as the L/4 JIT layer does.
            let hh = h3 >> 1;
            let cb = col_bits
                .as_ref()
                .expect("leaf bits alive for the JIT regen");
            let (bufs, round1) = if jit_r1 {
                let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
                jit_layer_generate(hh, zx, live, suffix, |c| {
                    let (lb, rb) = &cb[c];
                    let at = T4At {
                        lbits: lb,
                        rbits: rb,
                        t4: t4_src(t4f, &t4, &te, &to),
                        q1,
                    };
                    (
                        move |y| at.at(y) * at.at(y + h3),
                        move |y| {
                            if t4_pf {
                                at.prefetch_at(y);
                                at.prefetch_at(y + h3);
                            }
                        },
                    )
                })
            } else {
                let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..live)
                    .map(|c| {
                        let (lb, rb) = &cb[c];
                        let full = t4_level_values(lb, rb, t4_src(t4f, &t4, &te, &to), q1);
                        let e: Vec<Gf> = (0..hh).map(|y| full[y] * full[y + h3]).collect();
                        let o: Vec<Gf> = (hh..h3).map(|y| full[y] * full[y + h3]).collect();
                        GroupBufs::Dense(vec![(e, o)])
                    })
                    .collect();
                (bufs, None)
            };
            Some(BitLayer {
                bufs,
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1,
                flat: None,
            })
        } else if ell == depth - 3 {
            // One bit-driven round straight off T4 (k = d−3 ≥ 2): the
            // layer's input level is never stored nor regenerated; the
            // fold materialises ≈ L/8.
            let cb = col_bits.as_ref().expect("leaf bits alive for the T4 layer");
            Some(BitLayer {
                bufs: cb
                    .iter()
                    .map(|(lbits, rbits)| GroupBufs::T4Bits {
                        lbits,
                        rbits,
                        tau_set: 0,
                    })
                    .collect(),
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: vec![core::mem::take(&mut t4)],
                round1: None,
                flat: None,
            })
        } else if ell == depth - 2 {
            // Two bit-driven rounds (k = d−2 ≥ 3): fold materialises ≈ L/8.
            Some(BitLayer {
                bufs: col_bits
                    .as_ref()
                    .expect("leaf bits alive for the pair layer")
                    .iter()
                    .map(|(lbits, rbits)| GroupBufs::Pair3Bits {
                        lbits,
                        rbits,
                        tau_set: 0,
                    })
                    .collect(),
                tau_sets: Vec::new(),
                pair_tau_sets: vec![Pair2TauSet {
                    te: te.clone(),
                    to: to.clone(),
                }],
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else if ell == depth - 1 {
            // Three bit-driven rounds (k = d−1 ≥ 4): the leaf-round set
            // is ≈ L/8 — four under `BITZ_LUT4` (depth ≥ 6): ≈ L/16.
            let deep4 = depth >= 6 && forest_lut4();
            Some(BitLayer {
                bufs: col_bits
                    .take()
                    .expect("leaf bits consumed once")
                    .into_iter()
                    .map(|(lbits, rbits)| {
                        if deep4 {
                            GroupBufs::Leaf4Bits {
                                lbits,
                                rbits,
                                tau_set: 0,
                            }
                        } else {
                            GroupBufs::Leaf3Bits {
                                lbits,
                                rbits,
                                tau_set: 0,
                            }
                        }
                    })
                    .collect(),
                tau_sets: vec![leaf_tau.clone()],
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else {
            None
        }
    };
    drive_grouped(
        transcript,
        pad_roots(roots, num_trees),
        ForestLevels::PerTree(levels),
        bit_layer,
        depth,
        s,
        live,
    )
}

// =====================================================================
// QUAD forest (`BITZ_QUAD=1`, EXPERIMENTAL): arity-4 GKR layers over the
// stored/JIT region — each layer proves `L_ℓ = Σ eq·Q00·Q10·Q01·Q11` over
// the QUARTERS of level ℓ+2, certifying TWO product-tree levels per
// degree-5 sumcheck. K challenges and K values throughout — sound for
// any generator α (ported from the worktree-gf8 experiment with the
// order-255 byte-dlog surfaces stripped). The win at full-order α is
// structural, not representational: the stored chain keeps only EVEN
// levels (−33% region traffic), and the phase-B sumchecks + line steps
// halve over the region. The round bodies are scalar degree-5 Karatsuba
// (~28 PMULL-class ops/slot vs the arity-2 NEON stack's ~9
// vector-resident) — mult-bound at cache-resident sizes until the fused
// NEON degree-5 kernel lands; opt in for DRAM-bound shapes. The
// leaf/pair LUT-cascade layers stay arity-2 verbatim (their bit-driven
// prefix is already the stronger structure). The layer plan is
// depth-parity-deterministic; quad layers close on FOUR values (`pair` +
// `pair2`, tag 0x33) and draw TWO line challenges — a DIFFERENT
// transcript shape from the arity-2 forest, so prover and verifier
// dispatch on [`quad_active`] together.

use crate::piop::sumcheck::quad::{
    QuadBitGroup, QuadBottomTables, QuadGroup, prove_quad_bottom_sumcheck, prove_quad_eq_sumcheck,
};

/// Does the QUAD forest apply? `BITZ_QUAD=1`, the L/4 schedule (the quad
/// plan builds its chain at level d−3), depth ≥ 8. Transcript-shape
/// changing: prover and verifier BOTH dispatch through this — the env
/// var is the experiment's out-of-band configuration. Read per call.
pub fn quad_active(p: &IntegerMatrixLayout) -> bool {
    // A single tree uses the existing binary zero-variable reduction.
    if p.col_vars == 0 {
        return false;
    }
    let knob = std::env::var("BITZ_QUAD").unwrap_or_default();
    let forced = knob == "force" || knob == "force2";
    if !(forced || knob == "1" || knob == "2")
        || configured(p, ForestPath::Single).expect("single forest schedule") != ForestSchedule::L4
    {
        return false;
    }
    let log_w = p.word_bits.trailing_zeros() as usize;
    let row_len = p.rows() << log_w;
    if row_len < 256 {
        return false;
    }
    // The measured knee (below): quad and the double-fold are
    // SUBSTITUTES, so `=1` engages only where arity 4 still wins. The
    // v2 bottom merge extends the winning region (see [`QUAD2_N_MAX`]).
    let n_max = if knob == "2" { QUAD2_N_MAX } else { QUAD_N_MAX };
    forced || row_len.trailing_zeros() as usize + p.col_vars <= n_max
}

/// The QUAD knee in `n = depth + s`, measured 2026-07-31 in paired
/// in-window runs against the double-fold on one box (prove, medians of
/// 3): n=22 −12 %, n=24 −20 %, **n=26 +6 %, n=28 +5 %**.
///
/// Arity-4 layers run on their OWN degree-5 driver
/// ([`prove_quad_eq_sumcheck`]) covering the stored/JIT region — exactly
/// the region where binding two variables per pass wins most — so
/// enabling quad DISPLACES the double-fold from it. Below the knee the
/// halved layer count still dominates (and the double-fold keeps the
/// arity-2 remnants: the leaf/pair post-LUT cascades and the parity
/// layer, worth a further 1–3 %); above it the arity-2 cascade with two
/// variables per pass is simply the better body and quad gives back more
/// than it buys. The two are alternatives, not a stack.
///
/// `BITZ_QUAD=force` overrides the gate — for re-measuring the crossover
/// on a memory-fresh box, or after porting the double-fold into
/// `quad.rs` (a 5×5 node grid, 25 wide accumulators against the arity-2
/// case's 9), which is what would push this knee back up.
const QUAD_N_MAX: usize = 25;

/// The v2 (bottom-merge) knee: with the pair and leaf layers merged into
/// one bit-driven arity-4 layer, the quad plan wins THROUGH the measured
/// range (2026-08-20, paired in-window runs vs base, prove): n=24
/// −25.3 % (v1 −18.5 %), n=26 −3..−9 % (v1 +3.4 %), n=28 −2..−6.3 %
/// (3/3 pairs; v1 was a wash there). Beyond n=28 unmeasured (n ≥ 30
/// needs a memory-fresh box) — `BITZ_QUAD=force2` to probe.
const QUAD2_N_MAX: usize = 28;

/// The BOTTOM-MERGE variant (`BITZ_QUAD=2` / `force2` — S1 of
/// `docs/forest-speedup-ideas.md`, design in
/// `docs/quad-bottom-merge-prompt.md`): the arity-2 pair and leaf layers
/// are replaced by ONE arity-4 bit-driven layer (output d−2, consuming
/// the leaves — [`prove_quad_bottom_sumcheck`]). Transcript-shape
/// changing exactly like [`quad_active`] itself; prover and verifier
/// both read it. Read per call.
pub(crate) fn quad_v2() -> bool {
    matches!(std::env::var("BITZ_QUAD").as_deref(), Ok("2") | Ok("force2"))
}

/// The quad layer plan for tree depth `d`: quads deliver claims at even
/// levels `2, 4, …, P` (`P = d−2` if even, else `d−3`); a single arity-2
/// "parity" layer bridges `P → d−2` when `d−2` is odd; the pair (`d−2`)
/// and leaf (`d−1`) layers are always arity-2 — except under
/// [`quad_v2`], where ONE bottom quad layer (output d−2) replaces them.
pub(crate) struct QuadPlan {
    /// Output levels of the quad layers (ascending: 0, 2, …, P−2).
    pub quad_outputs: Vec<usize>,
    /// Whether the arity-2 parity layer (output `d−3`) exists.
    pub parity: bool,
}

pub(crate) fn quad_plan(depth: usize) -> QuadPlan {
    assert!(depth >= 8, "quad forest needs depth >= 8");
    let p = if (depth - 2) % 2 == 0 {
        depth - 2
    } else {
        depth - 3
    };
    QuadPlan {
        quad_outputs: (0..p).step_by(2).collect(),
        parity: p == depth - 3,
    }
}

/// 2-variable multilinear interpolation of a closing quad
/// `[Q00, Q10, Q01, Q11]` (m = a | b≪1) at `(μ_a, μ_b)`.
#[allow(clippy::arithmetic_side_effects)]
fn quad_interp(q: [Gf; 4], mu_a: Gf, mu_b: Gf) -> Gf {
    let one = Gf::one();
    let (na, nb) = (one + mu_a, one + mu_b);
    na * nb * q[0] + mu_a * nb * q[1] + na * mu_b * q[2] + mu_a * mu_b * q[3]
}

/// Split one tree's level halves `(e, o)` (TOP-split) into the four
/// quarter multiplicands `m = a | b≪1` (`b` = top bit ⇒ e/o; `a` = next
/// bit ⇒ halves of halves).
fn quad_quarters(mut e: Vec<Gf>, mut o: Vec<Gf>) -> [Vec<Gf>; 4] {
    let h = e.len() >> 1;
    let e_hi = e.split_off(h);
    let o_hi = o.split_off(h);
    [e, e_hi, o, o_hi]
}

/// Build every tree's EVEN levels `2, 4, … ≤ top` from its level-`top`
/// halves (`gen_top`), keeping the ODD levels transient: each is computed
/// as the stepping stone to the even level below and freed on the next
/// step — it never reaches the stored chain (per-tree it is L2-scale,
/// so its writes stay cache-local instead of streaming to DRAM). The
/// kept levels land in the same `TreeLevels` slots as [`build_levels`]
/// (odd slots empty); also returns the roots. Values are identical to
/// [`build_levels`]'s — this is a scheduling variant, not a semantic
/// one (the chain shrinks ≈ 3×: Σ_{even ℓ ≤ top} 2^ℓ vs Σ_{ℓ ≤ top} 2^ℓ).
#[allow(clippy::arithmetic_side_effects)]
fn build_levels_quad(
    num_trees: usize,
    top: usize,
    gen_top: impl Fn(usize) -> (Vec<Gf>, Vec<Gf>) + Sync,
) -> (TreeLevels, Vec<Gf>) {
    let _g = tracing::info_span!("mf:build_levels").entered();
    let chains: Vec<(Vec<(usize, (Vec<Gf>, Vec<Gf>))>, Gf)> = cfg_into_iter!(0..num_trees)
        .map(|c| {
            let mut kept: Vec<(usize, (Vec<Gf>, Vec<Gf>))> = Vec::new();
            let mut cur = gen_top(c);
            let mut level = top;
            while level > 1 {
                let parent = parent_halves_top(&cur.0, &cur.1);
                if level % 2 == 0 {
                    kept.push((level, cur)); // moved, never copied
                } // odd levels: `cur` freed on reassign — transient
                cur = parent;
                level -= 1;
            }
            let root = cur.0[0] * cur.1[0];
            (kept, root)
        })
        .collect();
    let mut levels: TreeLevels = (0..top).map(|_| Vec::with_capacity(num_trees)).collect();
    let mut roots = Vec::with_capacity(num_trees);
    for (kept, root) in chains {
        roots.push(root);
        for (level, halves) in kept {
            levels[level - 1].push(halves);
        }
    }
    (levels, roots)
}

/// One arity-2 layer of the quad drive (parity / pair / leaf) — the body
/// of [`drive_grouped`]'s loop as a standalone step over a [`BitLayer`].
/// Returns the proof layer and advances `(z_x, z_c, claim)`.
#[allow(clippy::arithmetic_side_effects)]
fn run_arity2_layer(
    transcript: &mut impl Transcript,
    bl: BitLayer,
    z_x: &mut Vec<Gf>,
    z_c: &mut Vec<Gf>,
    claim: &mut Gf,
    num_trees: usize,
) -> MergedLayer {
    let one = Gf::one();
    let _g = tracing::info_span!("mf:phaseA").entered();
    let eq_zc = if z_c.is_empty() {
        vec![one]
    } else {
        build_eq_x_r_vec(z_c, &()).expect("nonempty tree point")
    };
    let groups: Vec<EqInnerGroupMixed<'_, Gf>> = bl
        .bufs
        .into_iter()
        .zip(eq_zc.iter())
        .map(|(bufs, &scale)| EqInnerGroupMixed {
            q: z_x.as_slice().into(),
            scale,
            bufs,
        })
        .collect();
    let (sc, r_x, finals) = prove_eq_inner_sumcheck_mixed_pre(
        transcript,
        groups,
        &bl.tau_sets,
        &bl.pair_tau_sets,
        &bl.t4_sets,
        bl.round1,
        None,
        true,
        &(),
    );
    let mut e_vec = Vec::with_capacity(num_trees);
    let mut o_vec = Vec::with_capacity(num_trees);
    for f in finals {
        let (fe, fo) = f[0];
        e_vec.push(fe);
        o_vec.push(fo);
    }
    drop(_g);

    let _g = tracing::info_span!("mf:phaseB").entered();
    let group_b = EqInnerGroupMixed {
        q: z_c.as_slice().into(),
        scale: one,
        bufs: GroupBufs::Dense(vec![(e_vec, o_vec)]),
    };
    let (sc_c, r_c, finals_b) =
        prove_eq_inner_sumcheck_mixed_gruen(transcript, vec![group_b], &[], &[], &[], &());
    let pair = finals_b[0][0];
    drop(_g);

    absorb_gfs(transcript, 0x32, &[pair.0, pair.1]);
    let mu: Gf = transcript.get_field_challenge(&());
    *claim = pair.0 + mu * (pair.0 + pair.1);
    let mut nx = r_x;
    nx.push(mu);
    *z_x = nx;
    *z_c = r_c;
    MergedLayer {
        sc_x: Some(sc),
        sc_c,
        pair,
        pair2: None,
    }
}

/// The QUAD forest prover — the L/4 lazy prover's arity-4 sibling: the
/// same table builds and stored-chain construction (tops at level d−3
/// via paired T4 gathers), but only EVEN levels retained; quad layers
/// over the stored/JIT region; then the arity-2 parity (odd depths) /
/// pair / leaf layers via the existing driver machinery. Same signature
/// as [`prove_merged_forest_lazy`]; callers gate on [`quad_active`].
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_merged_forest_lazy_quad(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    packed_cols: &[Vec<u64>],
    pow2: &[Vec<Gf>],
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    prove_merged_forest_lazy_quad_from_rows(transcript, p, None, packed_cols, pow2)
}

pub(crate) fn prove_merged_forest_lazy_quad_from_rows(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    rows: Option<&[Vec<u64>]>,
    packed_cols: &[Vec<u64>],
    pow2: &[Vec<Gf>],
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    use crate::pcs::{extract_column_bit_halves, layer1_pair_table, leaf_tau_halves};
    let log_w = p.word_bits.trailing_zeros() as usize;
    let mask_w = p.word_bits.wrapping_sub(1);
    let row_len = p.rows() << log_w;
    let depth = row_len.trailing_zeros() as usize;
    let s = p.col_vars;
    let num_trees = p.cols();
    let one = Gf::one();
    assert!(
        depth >= 8,
        "quad forest needs depth >= 8 (callers gate on quad_active)"
    );
    let plan = quad_plan(depth);

    let pair_tbl = layer1_pair_table(p, pow2, log_w, row_len);
    let leaf_tau = leaf_tau_halves(p, pow2, one, log_w, row_len);
    let extracted;
    let halves: Vec<(&[u64], &[u64])> = if let Some(rows) = rows {
        assert_eq!(rows.len(), num_trees, "column count");
        rows.iter()
            .map(|row| row[..row_len / 64].split_at(row_len / 128))
            .collect()
    } else {
        extracted = extract_column_bit_halves(packed_cols, num_trees, row_len);
        extracted
            .iter()
            .map(|(l, r)| (l.as_slice(), r.as_slice()))
            .collect()
    };
    let mut col_bits = Some(halves);

    // The shared 4-case / 16-case tables — exactly the L/4 build.
    let q1 = row_len >> 2;
    let q2 = row_len >> 1;
    let v = |i: usize| -> Gf { pow2[i >> log_w][i & mask_w] };
    let build_cases = |base: usize| -> Vec<Gf> {
        let mut t = Vec::with_capacity(q1 << 2);
        for y in 0..q1 {
            let lo = base + y;
            t.push(one);
            t.push(v(lo));
            t.push(v(lo + q2));
            t.push(pair_tbl[lo]);
        }
        t
    };
    let te = build_cases(0);
    let to = build_cases(q1);
    // QUAD path: factored T4 consumption is opt-in only (unmeasured
    // here; `T4` is always built — the arity-4 layers keep reading it).
    let t4f = t4_factored().unwrap_or(false);
    let t4: Vec<Gf> = {
        let rows: Vec<[Gf; 16]> = cfg_into_iter!(0..q1, 1 << 10)
            .map(|y| {
                let mut row = [Gf::one(); 16];
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = te[(y << 2) | (c >> 2)] * to[(y << 2) | (c & 3)];
                }
                row
            })
            .collect();
        rows.into_flattened()
    };

    // Stored chain: the L/4 top (level d−3 via paired T4 gathers), but
    // EVEN levels only — the quad layers consume nothing else, and the
    // odd levels stay per-tree transients inside the walk
    // ([`build_levels_quad`]): the chain is ≈ 3× smaller than L/4's.
    let h3 = q1 >> 1;
    let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
    let (mut levels, roots) = build_levels_quad(num_trees, depth - 3, |c| {
        let cb = col_bits.as_ref().expect("leaf bits alive for the build");
        let (lb, rb) = &cb[c];
        let at = T4At {
            lbits: lb,
            rbits: rb,
            t4: t4_src(t4f, &t4, &te, &to),
            q1,
        };
        let v3 = |y: usize| -> Gf {
            if t4_pf {
                let yp = y + PRFM_DIST;
                if yp < h3 {
                    at.prefetch_at(yp);
                    at.prefetch_at(yp + h3);
                }
            }
            at.at(y) * at.at(y + h3)
        };
        let hh = h3 >> 1;
        ((0..hh).map(v3).collect(), (hh..h3).map(v3).collect())
    });

    // ---- drive ----
    absorb_gfs(transcript, 0x30, &roots);
    let zeta: Vec<Gf> = transcript.get_field_challenges(s, &());
    let mut claim = mle_at(&roots, &zeta);
    let mut z_x: Vec<Gf> = Vec::new();
    let mut z_c: Vec<Gf> = zeta;
    let mut out_layers: Vec<MergedLayer> =
        Vec::with_capacity(plan.quad_outputs.len() + usize::from(plan.parity) + 2);

    for &ell in &plan.quad_outputs {
        let input_level = ell + 2;
        // Per-tree quarters of level ℓ+2 (`m = a | b≪1`; `b` = the
        // level's top bit, `a` the next — matching the two line
        // challenges' order).
        let quarters: Vec<[Vec<Gf>; 4]> = if input_level == depth - 2 {
            // The JIT level: gathered per tree straight off the bits +
            // T4 (never stored), quarter-contiguous.
            let _g = tracing::info_span!("mf:bitgen").entered();
            let cb = col_bits.as_ref().expect("leaf bits alive for the JIT quad");
            let hq = q1 >> 2;
            cfg_into_iter!(0..num_trees)
                .map(|c| {
                    let (lb, rb) = &cb[c];
                    let at = T4At {
                        lbits: lb,
                        rbits: rb,
                        t4: t4_src(t4f, &t4, &te, &to),
                        q1,
                    };
                    let quarter = |lo: usize| -> Vec<Gf> {
                        (lo..lo + hq)
                            .map(|j| {
                                if t4_pf {
                                    let jp = j + PRFM_DIST;
                                    if jp < lo + hq {
                                        at.prefetch_at(jp);
                                    }
                                }
                                at.at(j)
                            })
                            .collect()
                    };
                    [quarter(0), quarter(hq), quarter(2 * hq), quarter(3 * hq)]
                })
                .collect()
        } else {
            let lvl = core::mem::take(&mut levels[input_level - 1]);
            assert!(
                !lvl.is_empty(),
                "stored quad input level {input_level} retained"
            );
            lvl.into_iter().map(|(e, o)| quad_quarters(e, o)).collect()
        };

        let (sc_x, r_x, finals): (Option<SumcheckProof<Gf>>, Vec<Gf>, Vec<[Gf; 4]>) = if ell == 0 {
            // Root layer: no phase A — the four level-2 values per
            // tree are scalars.
            let finals: Vec<[Gf; 4]> = quarters
                .iter()
                .map(|q| [q[0][0], q[1][0], q[2][0], q[3][0]])
                .collect();
            (None, Vec::new(), finals)
        } else {
            let _g = tracing::info_span!("mf:phaseA").entered();
            let eq_zc = if z_c.is_empty() {
                vec![one]
            } else {
                build_eq_x_r_vec(&z_c, &()).expect("nonempty tree point")
            };
            let groups: Vec<QuadGroup> = quarters
                .into_iter()
                .zip(eq_zc.iter())
                .map(|(bufs, &scale)| QuadGroup {
                    q: z_x.as_slice().into(),
                    scale,
                    bufs,
                })
                .collect();
            let (sc, r_x, finals) = prove_quad_eq_sumcheck(transcript, groups);
            (Some(sc), r_x, finals)
        };

        // Phase B: Σ_c eq(c, z_c)·Π_m Q_m(r_x, c), degree 5 over s vars.
        let _g = tracing::info_span!("mf:phaseB").entered();
        let mut bufs_b: [Vec<Gf>; 4] = [
            Vec::with_capacity(num_trees),
            Vec::with_capacity(num_trees),
            Vec::with_capacity(num_trees),
            Vec::with_capacity(num_trees),
        ];
        for f in &finals {
            for m in 0..4 {
                bufs_b[m].push(f[m]);
            }
        }
        let group_b = QuadGroup {
            q: z_c.as_slice().into(),
            scale: one,
            bufs: bufs_b,
        };
        let (sc_c, r_c, finals_b) = prove_quad_eq_sumcheck(transcript, vec![group_b]);
        let quad = finals_b[0];
        drop(_g);

        absorb_gfs(transcript, 0x33, &quad);
        let mu_a: Gf = transcript.get_field_challenge(&());
        let mu_b: Gf = transcript.get_field_challenge(&());
        claim = quad_interp(quad, mu_a, mu_b);
        let mut nx = r_x;
        nx.push(mu_a);
        nx.push(mu_b);
        z_x = nx;
        z_c = r_c;
        out_layers.push(MergedLayer {
            sc_x,
            sc_c,
            pair: (quad[0], quad[1]),
            pair2: Some((quad[2], quad[3])),
        });
    }

    // Parity layer (odd d−2): arity-2 over the JIT level d−2, generated
    // with the fused round-1 exactly like the L/4 JIT layer.
    if plan.parity {
        let hh = q1 >> 1;
        let (bufs, round1) = {
            let _g = tracing::info_span!("mf:bitgen").entered();
            let cb = col_bits
                .as_ref()
                .expect("leaf bits alive for the parity layer");
            if jit_round1_fuse() {
                let tensors = suffix_tensors(&z_x, &());
                let v1 = tensors.tensor(0);
                let generated: Vec<(GroupBufs<'_, Gf>, (Gf, Gf, Gf))> =
                    cfg_into_iter!(0..num_trees)
                        .map(|c| {
                            let (lb, rb) = &cb[c];
                            let at = T4At {
                                lbits: lb,
                                rbits: rb,
                                t4: t4_src(t4f, &t4, &te, &to),
                                q1,
                            };
                            let (pair, coeffs) = dense_jit_fused_round1(
                                hh,
                                v1,
                                |j| at.at(j),
                                |j| {
                                    if t4_pf {
                                        at.prefetch_at(j);
                                    }
                                },
                            );
                            (GroupBufs::Dense(vec![pair]), coeffs)
                        })
                        .collect();
                let mut bufs = Vec::with_capacity(num_trees);
                let mut round1 = Vec::with_capacity(num_trees);
                for (b, h) in generated {
                    bufs.push(b);
                    round1.push(h);
                }
                (bufs, Some(PreRound::Coeffs(round1)))
            } else {
                let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..num_trees)
                    .map(|c| {
                        let (lb, rb) = &cb[c];
                        let full = t4_level_values(lb, rb, T4Src::Pre(&t4), q1);
                        GroupBufs::Dense(vec![(full[..hh].to_vec(), full[hh..].to_vec())])
                    })
                    .collect();
                (bufs, None)
            }
        };
        let bl = BitLayer {
            bufs,
            tau_sets: Vec::new(),
            pair_tau_sets: Vec::new(),
            t4_sets: Vec::new(),
            round1,
            flat: None,
        };
        out_layers.push(run_arity2_layer(
            transcript, bl, &mut z_x, &mut z_c, &mut claim, num_trees,
        ));
    }
    drop(t4);

    if quad_v2() {
        // BOTTOM MERGE (`BITZ_QUAD=2`): the pair and leaf layers as ONE
        // arity-4 bit-driven layer — output d−2, consuming the leaves,
        // whose quarters are never materialised
        // ([`prove_quad_bottom_sumcheck`]). One phase A over d−2 vars
        // replaces the two arity-2 phase As (and one phase B + line step
        // disappear); the exit claim shape is unchanged.
        let (sc_x, r_x, finals) = {
            let _g = tracing::info_span!("mf:phaseA").entered();
            // The unweighted 16-case ΔΔ table: round 1's p₂ gathers —
            // subset sums of the leaf-affine Δ cross products per
            // position pair, both table halves.
            let t_dd: Vec<Gf> = {
                let rows: Vec<[Gf; 16]> = cfg_into_iter!(0..(q2 >> 1), 1 << 10)
                    .map(|p| {
                        let base = p << 1;
                        let c00 = leaf_tau.0[base] * leaf_tau.1[base];
                        let c10 = leaf_tau.0[base + 1] * leaf_tau.1[base];
                        let c01 = leaf_tau.0[base] * leaf_tau.1[base + 1];
                        let c11 = leaf_tau.0[base + 1] * leaf_tau.1[base + 1];
                        let mut row = [Gf::zero(); 16];
                        for (c, slot) in row.iter_mut().enumerate() {
                            let mut v = Gf::zero();
                            if c & 0b0101 == 0b0101 {
                                v += c00;
                            }
                            if c & 0b0110 == 0b0110 {
                                v += c10;
                            }
                            if c & 0b1001 == 0b1001 {
                                v += c01;
                            }
                            if c & 0b1010 == 0b1010 {
                                v += c11;
                            }
                            *slot = v;
                        }
                        row
                    })
                    .collect();
                rows.into_flattened()
            };
            let eq_zc = if z_c.is_empty() {
                vec![one]
            } else {
                build_eq_x_r_vec(&z_c, &()).expect("nonempty tree point")
            };
            let groups: Vec<QuadBitGroup> = col_bits
                .take()
                .expect("leaf bits consumed once")
                .into_iter()
                .zip(eq_zc.iter())
                .map(|((lbits, rbits), &scale)| QuadBitGroup {
                    scale,
                    lbits,
                    rbits,
                })
                .collect();
            let tabs = QuadBottomTables {
                te: &te,
                to: &to,
                t_dd: &t_dd,
                tau_l: &leaf_tau.0,
                tau_r: &leaf_tau.1,
            };
            let (sc, r_x, finals) =
                prove_quad_bottom_sumcheck(transcript, z_x.clone(), groups, &tabs);
            (Some(sc), r_x, finals)
        };
        let _g = tracing::info_span!("mf:phaseB").entered();
        let mut bufs_b: [Vec<Gf>; 4] = [
            Vec::with_capacity(num_trees),
            Vec::with_capacity(num_trees),
            Vec::with_capacity(num_trees),
            Vec::with_capacity(num_trees),
        ];
        for f in &finals {
            for m in 0..4 {
                bufs_b[m].push(f[m]);
            }
        }
        let group_b = QuadGroup {
            q: z_c.as_slice().into(),
            scale: one,
            bufs: bufs_b,
        };
        let (sc_c, r_c, finals_b) = prove_quad_eq_sumcheck(transcript, vec![group_b]);
        let quad = finals_b[0];
        drop(_g);

        absorb_gfs(transcript, 0x33, &quad);
        let mu_a: Gf = transcript.get_field_challenge(&());
        let mu_b: Gf = transcript.get_field_challenge(&());
        claim = quad_interp(quad, mu_a, mu_b);
        let mut nx = r_x;
        nx.push(mu_a);
        nx.push(mu_b);
        z_x = nx;
        z_c = r_c;
        out_layers.push(MergedLayer {
            sc_x,
            sc_c,
            pair: (quad[0], quad[1]),
            pair2: Some((quad[2], quad[3])),
        });

        let mut z = z_x;
        z.extend_from_slice(&z_c);
        return (roots, MergedForestProof { layers: out_layers }, z, claim);
    }

    // Pair layer (output d−2): the L/4 Pair3Bits/Pair2Bits round over
    // the shared 4-case tables.
    {
        let deep = depth >= 5 && forest_lut3();
        let bl = BitLayer {
            bufs: col_bits
                .as_ref()
                .expect("leaf bits alive for the pair layer")
                .iter()
                .map(|(lbits, rbits)| {
                    let (lbits, rbits) = (*lbits, *rbits);
                    if deep {
                        GroupBufs::Pair3Bits {
                            lbits,
                            rbits,
                            tau_set: 0,
                        }
                    } else {
                        GroupBufs::Pair2Bits {
                            lbits,
                            rbits,
                            tau_set: 0,
                        }
                    }
                })
                .collect(),
            tau_sets: Vec::new(),
            pair_tau_sets: vec![Pair2TauSet { te, to }],
            t4_sets: Vec::new(),
            round1: None,
            flat: None,
        };
        out_layers.push(run_arity2_layer(
            transcript, bl, &mut z_x, &mut z_c, &mut claim, num_trees,
        ));
    }

    // Leaf layer (output d−1): the L/4 Leaf3Bits/Leaf2Bits round over
    // the τ tables.
    {
        let deep = depth >= 5 && forest_lut3();
        let bl = BitLayer {
            bufs: col_bits
                .take()
                .expect("leaf bits consumed once")
                .into_iter()
                .map(|(lbits, rbits)| {
                    if deep {
                        GroupBufs::Leaf3Bits {
                            lbits,
                            rbits,
                            tau_set: 0,
                        }
                    } else {
                        GroupBufs::Leaf2Bits {
                            lbits,
                            rbits,
                            tau_set: 0,
                        }
                    }
                })
                .collect(),
            tau_sets: vec![leaf_tau],
            pair_tau_sets: Vec::new(),
            t4_sets: Vec::new(),
            round1: None,
            flat: None,
        };
        out_layers.push(run_arity2_layer(
            transcript, bl, &mut z_x, &mut z_c, &mut claim, num_trees,
        ));
    }

    let mut z = z_x;
    z.extend_from_slice(&z_c);
    (roots, MergedForestProof { layers: out_layers }, z, claim)
}

/// Verify a QUAD forest proof — [`verify_merged_forest`]'s mirror under
/// the depth-deterministic [`quad_plan`]: quad layers verify at degree 5,
/// close on four values (tag 0x33) and draw TWO line challenges; arity-2
/// layers are verbatim. Returns `(exit_point, exit_eval)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_merged_forest_quad(
    transcript: &mut impl Transcript,
    roots: &[Gf],
    proof: &MergedForestProof,
    depth: usize,
    s: usize,
) -> Result<(Vec<Gf>, Gf), MergedForestError> {
    if depth < 8 {
        return Err(MergedForestError::Shape);
    }
    let plan = quad_plan(depth);
    // The layer-kind sequence: quad outputs, then the arity-2 tail —
    // parity? + pair + leaf under v1; parity? + the BOTTOM quad
    // (output d−2, [`quad_v2`]) under v2.
    enum LKind {
        Quad(usize),
        Arity2,
    }
    let mut kinds: Vec<LKind> = plan.quad_outputs.iter().map(|&e| LKind::Quad(e)).collect();
    if plan.parity {
        kinds.push(LKind::Arity2);
    }
    if quad_v2() {
        kinds.push(LKind::Quad(depth - 2));
    } else {
        kinds.push(LKind::Arity2);
        kinds.push(LKind::Arity2);
    }
    if roots.len() != 1usize << s || proof.layers.len() != kinds.len() {
        return Err(MergedForestError::Shape);
    }
    let one = Gf::one();
    absorb_gfs(transcript, 0x30, roots);
    let zeta: Vec<Gf> = transcript.get_field_challenges(s, &());
    let mut claim = mle_at(roots, &zeta);
    let mut z_x: Vec<Gf> = Vec::new();
    let mut z_c: Vec<Gf> = zeta;

    for (li, (layer, kind)) in proof.layers.iter().zip(&kinds).enumerate() {
        match kind {
            LKind::Quad(ell) => {
                let ell = *ell;
                let quad = match (layer.pair, layer.pair2) {
                    ((q0, q1), Some((q2, q3))) => [q0, q1, q2, q3],
                    _ => return Err(MergedForestError::Shape),
                };
                let r_x = if ell == 0 {
                    if layer.sc_x.is_some() {
                        return Err(MergedForestError::Shape);
                    }
                    if layer.sc_c.claimed_sum != claim {
                        return Err(MergedForestError::LayerClaim { layer: li });
                    }
                    Vec::new()
                } else {
                    let sc_x = layer.sc_x.as_ref().ok_or(MergedForestError::Shape)?;
                    if sc_x.claimed_sum != claim {
                        return Err(MergedForestError::LayerClaim { layer: li });
                    }
                    let sub =
                        MLSumcheck::<Gf>::verify_as_subprotocol(transcript, ell, 5, sc_x, &())
                            .map_err(|_| MergedForestError::LayerClaim { layer: li })?;
                    let eqx =
                        eq_eval(&sub.point, &z_x, one).map_err(|_| MergedForestError::Shape)?;
                    if sub.expected_evaluation != eqx * layer.sc_c.claimed_sum {
                        return Err(MergedForestError::LayerClaim { layer: li });
                    }
                    sub.point
                };

                let sub_c =
                    MLSumcheck::<Gf>::verify_as_subprotocol(transcript, s, 5, &layer.sc_c, &())
                        .map_err(|_| MergedForestError::LayerClaim { layer: li })?;
                let eqc = eq_eval(&sub_c.point, &z_c, one).map_err(|_| MergedForestError::Shape)?;
                if sub_c.expected_evaluation != eqc * quad[0] * quad[1] * quad[2] * quad[3] {
                    return Err(MergedForestError::LayerClaim { layer: li });
                }

                absorb_gfs(transcript, 0x33, &quad);
                let mu_a: Gf = transcript.get_field_challenge(&());
                let mu_b: Gf = transcript.get_field_challenge(&());
                claim = quad_interp(quad, mu_a, mu_b);
                let mut nx = r_x;
                nx.push(mu_a);
                nx.push(mu_b);
                z_x = nx;
                z_c = sub_c.point;
            }
            LKind::Arity2 => {
                if layer.pair2.is_some() {
                    return Err(MergedForestError::Shape);
                }
                let sc_x = layer.sc_x.as_ref().ok_or(MergedForestError::Shape)?;
                if sc_x.claimed_sum != claim {
                    return Err(MergedForestError::LayerClaim { layer: li });
                }
                let sub = verify_eq_inner_sumcheck_gruen(transcript, &z_x, sc_x, &())
                    .map_err(|_| MergedForestError::LayerClaim { layer: li })?;
                let eqx = eq_eval(&sub.point, &z_x, one).map_err(|_| MergedForestError::Shape)?;
                if sub.expected_evaluation != eqx * layer.sc_c.claimed_sum {
                    return Err(MergedForestError::LayerClaim { layer: li });
                }
                let sub_c = verify_eq_inner_sumcheck_gruen(transcript, &z_c, &layer.sc_c, &())
                    .map_err(|_| MergedForestError::LayerClaim { layer: li })?;
                let (p_, q_) = layer.pair;
                let eqc = eq_eval(&sub_c.point, &z_c, one).map_err(|_| MergedForestError::Shape)?;
                if sub_c.expected_evaluation != eqc * p_ * q_ {
                    return Err(MergedForestError::LayerClaim { layer: li });
                }
                absorb_gfs(transcript, 0x32, &[p_, q_]);
                let mu: Gf = transcript.get_field_challenge(&());
                claim = p_ + mu * (p_ + q_);
                let mut nx = sub.point;
                nx.push(mu);
                z_x = nx;
                z_c = sub_c.point;
            }
        }
    }
    let mut z = z_x;
    z.extend_from_slice(&z_c);
    Ok((z, claim))
}

/// Per-position level-(d−1) reader of the RLC j=2 forest: position `j`'s
/// value is the 16-case select `t4[(j≪4) | (cE≪2) | cO]` with
/// `cE = m1.bit(j) | m2.bit(j)≪1` (the leaf case at `j`) and `cO` the case
/// at `j + 2^{d−1}` — the TOP-paired leaf. Same concrete-struct shape as
/// [`T4At`] (the opaque-closure lesson), over the two family-column bit
/// streams instead of the transposed leaf-bit halves.
struct RlcT4At<'a> {
    m1: &'a [u64],
    m2: &'a [u64],
    t4: &'a [Gf],
    q2: usize,
}

impl RlcT4At<'_> {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn idx(&self, j: usize) -> usize {
        #[inline(always)]
        fn bit(bits: &[u64], p: usize) -> usize {
            ((bits[p >> 6] >> (p & 63)) & 1) as usize
        }
        let ce = bit(self.m1, j) | (bit(self.m2, j) << 1);
        let co = bit(self.m1, j + self.q2) | (bit(self.m2, j + self.q2) << 1);
        (j << 4) | (ce << 2) | co
    }

    #[inline(always)]
    fn at(&self, j: usize) -> Gf {
        self.t4[self.idx(j)]
    }

    #[inline(always)]
    fn prefetch_at(&self, j: usize) {
        crate::piop::sumcheck::eq_factored::prefetch_l1(self.t4, self.idx(j));
    }
}

/// Lazy merged-forest prover for the RLC-family **j = 2** leaves
/// (EXPERIMENTAL, `docs/rlc-family-note-prompt.md`): the leaf at position
/// `i` of tree `c` is the 4-case select `case_pow[i][m(i,c)]` on the two
/// family columns' bits (`m = m1 | m2≪1`, case 0 = `α^0 = 1`). The 2^j-case
/// select IS structurally the driver's existing table family, one level
/// shifted vs the bit-affine forest:
///
/// * leaf layer (k = d−1) → [`GroupBufs::Pair3Bits`] (two bit-driven
///   rounds; [`GroupBufs::Pair2Bits`] under `BITZ_LUT3=0`) over the
///   [`Pair2TauSet`] `te[4y+c] = case_pow[y][c]`, `to` at `y + 2^{d−1}`;
/// * level d−1 (16 cases over TOP-paired leaf positions) →
///   [`GroupBufs::T4Bits`] with `t4[(j≪4)|(cE≪2)|cO] = te[4j+cE]·to[4j+cO]`;
/// * level d−2 → Dense JIT from paired T4 gathers; stored chain tops at
///   level d−3 (the L/4 memory shape).
///
/// The round kernels are untouched — only the tables and the bit streams
/// differ — so the transcript is byte-identical to [`prove_merged_forest`]
/// over the same leaf values (pinned by a test). `m1_rows[c]` /
/// `m2_rows[c]` are the family columns' per-tree x-tensor bit rows
/// (`2^{t'}` bits, 64 per word). Requires depth ≥ 4 (the deployed x shapes
/// have `t' ≥ 6`).
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_merged_forest_lazy_rlc2(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    m1_rows: &[Vec<u64>],
    m2_rows: &[Vec<u64>],
    case_pow: &[Vec<Gf>],
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    assert_eq!(p.word_bits, 1, "RLC j=2 leaves live on the W=1 x tensor");
    let row_len = p.rows();
    let depth = row_len.trailing_zeros() as usize;
    let s = p.col_vars;
    let num_trees = p.cols();
    assert!(
        depth >= 4,
        "RLC j=2 lazy forest needs depth >= 4; got {depth}"
    );
    assert!(
        m1_rows.len() == num_trees && m2_rows.len() == num_trees,
        "one bit row per tree"
    );
    debug_assert!(
        case_pow.len() == row_len && case_pow.iter().all(|r| r.len() == 4),
        "case_pow must be [2^d][4]"
    );
    debug_assert!(
        case_pow.iter().all(|r| r[0] == Gf::one()),
        "case 0 must be α^0 = 1 (linear forms vanish at 0)"
    );

    let q2 = row_len >> 1; // 2^{d−1}: leaf-pair offset = leaf-layer slot count
    let h2 = row_len >> 2; // level d−2 positions
    let h3 = row_len >> 3; // level d−3 positions

    // Leaf-layer 4-case tables: te = positions 0..2^{d−1}, to = the rest.
    let build_cases = |base: usize| -> Vec<Gf> {
        let mut t = Vec::with_capacity(q2 << 2);
        for y in 0..q2 {
            t.extend_from_slice(&case_pow[base + y]);
        }
        t
    };
    let te = build_cases(0);
    let to = build_cases(q2);

    // Level-(d−1) 16-case table (indexed collect + in-place flatten — see
    // the bit-affine build's parallel-flatten note).
    let t4: Vec<Gf> = {
        let rows: Vec<[Gf; 16]> = cfg_into_iter!(0..q2, 1 << 10)
            .map(|y| {
                let mut row = [Gf::one(); 16];
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = te[(y << 2) | (c >> 2)] * to[(y << 2) | (c & 3)];
                }
                row
            })
            .collect();
        rows.into_flattened()
    };

    // Stored chain tops at level d−3 via TOP-paired products of T4-pair
    // products — the level d−1/d−2 buffers never exist at build.
    let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
    let (levels, roots) = build_levels(num_trees, depth - 3, |c| {
        let at = RlcT4At {
            m1: &m1_rows[c],
            m2: &m2_rows[c],
            t4: &t4,
            q2,
        };
        let v3 = |y: usize| -> Gf {
            if t4_pf {
                let yp = y + PRFM_DIST;
                if yp < h3 {
                    at.prefetch_at(yp);
                    at.prefetch_at(yp + h2);
                    at.prefetch_at(yp + h3);
                    at.prefetch_at(yp + h3 + h2);
                }
            }
            (at.at(y) * at.at(y + h2)) * (at.at(y + h3) * at.at(y + h3 + h2))
        };
        let hh = h3 >> 1;
        ((0..hh).map(v3).collect(), (hh..h3).map(v3).collect())
    });

    let mut t4 = t4;
    let mut te = te;
    let mut to = to;
    let deep_leaf = forest_lut3();
    let bit_layer = |ell: usize, _zx: &[Gf], _suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
        if ell == depth - 3 {
            // JIT: regenerate level d−2 (Dense, exact-capacity halves) per
            // tree from the bits + T4 — alive only while this layer runs.
            let hh = h2 >> 1;
            let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..num_trees)
                .map(|c| {
                    let at = RlcT4At {
                        m1: &m1_rows[c],
                        m2: &m2_rows[c],
                        t4: &t4,
                        q2,
                    };
                    let v2 = |j: usize| -> Gf {
                        if t4_pf {
                            let jp = j + PRFM_DIST;
                            if jp < h2 {
                                at.prefetch_at(jp);
                                at.prefetch_at(jp + h2);
                            }
                        }
                        at.at(j) * at.at(j + h2)
                    };
                    GroupBufs::Dense(vec![(
                        (0..hh).map(v2).collect(),
                        (hh..h2).map(v2).collect(),
                    )])
                })
                .collect();
            Some(BitLayer {
                bufs,
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else if ell == depth - 2 {
            // One bit-driven round straight off T4 (k = d−2 ≥ 2): the
            // layer's input level is never stored nor regenerated.
            Some(BitLayer {
                bufs: (0..num_trees)
                    .map(|c| GroupBufs::T4Bits {
                        lbits: &m1_rows[c],
                        rbits: &m2_rows[c],
                        tau_set: 0,
                    })
                    .collect(),
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: vec![core::mem::take(&mut t4)],
                round1: None,
                flat: None,
            })
        } else if ell == depth - 1 {
            // The 4-case LEAF round (k = d−1 ≥ 3): Pair3Bits — two
            // bit-driven rounds — by default; Pair2Bits under `BITZ_LUT3=0`.
            Some(BitLayer {
                bufs: (0..num_trees)
                    .map(|c| {
                        let (lbits, rbits) = (m1_rows[c].as_slice(), m2_rows[c].as_slice());
                        if deep_leaf {
                            GroupBufs::Pair3Bits {
                                lbits,
                                rbits,
                                tau_set: 0,
                            }
                        } else {
                            GroupBufs::Pair2Bits {
                                lbits,
                                rbits,
                                tau_set: 0,
                            }
                        }
                    })
                    .collect(),
                tau_sets: Vec::new(),
                pair_tau_sets: vec![Pair2TauSet {
                    te: core::mem::take(&mut te),
                    to: core::mem::take(&mut to),
                }],
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else {
            None
        }
    };
    drive_grouped(
        transcript,
        roots,
        ForestLevels::PerTree(levels),
        bit_layer,
        depth,
        s,
        num_trees,
    )
}

/// The j-bit case of position `p` gathered from up to 4 family-column bit
/// streams: `m = Σ_i bit_i(p) ≪ i`.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
fn rlc_case(streams: &[&[u64]], p: usize) -> usize {
    let mut m = 0usize;
    for (fi, s) in streams.iter().enumerate() {
        m |= (((s[p >> 6] >> (p & 63)) & 1) as usize) << fi;
    }
    m
}

/// Lazy merged-forest prover for the RLC-family **j = 3, 4** leaves
/// (EXPERIMENTAL): the leaf at position `i` of tree `c` is the
/// 2^j-case select `case_pow[i][m(i,c)]` on the j family columns' bits.
/// Without 8/16-case leaf-round kernels (the open lever), the bottom
/// layers run **Dense with JIT-generated buffers** — the leaf layer's
/// values are gathered per tree straight from the shared case table, the
/// level above from the shared `2^{2j}`-case TOP-pair product table
/// `T2[(i ≪ 2j) | (c_E ≪ j) | c_O] = case_pow[i][c_E]·case_pow[i+2^{d−1}][c_O]`
/// (1 gather per value), and the stored chain tops at level d−3 — so the
/// eager path's separate leaf materialisation and full `build_levels`
/// chain never exist. Values are identical to the eager reference, hence
/// the transcript is byte-identical (pinned by a test). Requires
/// depth ≥ 4.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_merged_forest_lazy_rlc_general(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    m_rows: &[&[Vec<u64>]],
    case_pow: &[Vec<Gf>],
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    assert_eq!(p.word_bits, 1, "RLC leaves live on the W=1 x tensor");
    let j = m_rows.len();
    assert!(
        (2..=4).contains(&j),
        "general RLC lazy forest supports j in [2, 4]"
    );
    let row_len = p.rows();
    let depth = row_len.trailing_zeros() as usize;
    let s = p.col_vars;
    let num_trees = p.cols();
    assert!(depth >= 4, "RLC lazy forest needs depth >= 4; got {depth}");
    let cases = 1usize << j;
    debug_assert!(
        case_pow.len() == row_len && case_pow.iter().all(|r| r.len() == cases),
        "case_pow must be [2^d][2^j]"
    );

    let q2 = row_len >> 1; // 2^{d−1}: leaf TOP-pair offset
    let h2 = row_len >> 2; // level d−2 positions
    let h3 = row_len >> 3; // level d−3 positions

    // Shared level-(d−1) product table (indexed collect + in-place
    // flatten — the parallel-flatten lesson): 2^{2j}·2^{d−1} entries
    // (j = 3: 64 cases, j = 4: 256).
    let two_j = j << 1;
    let t2: Vec<Gf> = {
        let rows: Vec<Vec<Gf>> = cfg_into_iter!(0..q2, 1 << 9)
            .map(|i| {
                let mut row = Vec::with_capacity(1usize << two_j);
                for ce in 0..cases {
                    for co in 0..cases {
                        row.push(case_pow[i][ce] * case_pow[i + q2][co]);
                    }
                }
                row
            })
            .collect();
        rows.into_iter().flatten().collect()
    };
    // ld1(i) = level-(d−1) value at slot i, one T2 gather.
    let ld1 = |streams: &[&[u64]], i: usize| -> Gf {
        let ce = rlc_case(streams, i);
        let co = rlc_case(streams, i + q2);
        t2[(i << two_j) | (ce << j) | co]
    };

    // Stored chain tops at level d−3 via TOP-paired products of T2 pairs.
    let (levels, roots) = build_levels(num_trees, depth - 3, |c| {
        let streams: Vec<&[u64]> = m_rows.iter().map(|r| &r[c][..]).collect();
        let v3 = |y: usize| -> Gf {
            (ld1(&streams, y) * ld1(&streams, y + h2))
                * (ld1(&streams, y + h3) * ld1(&streams, y + h3 + h2))
        };
        let hh = h3 >> 1;
        ((0..hh).map(v3).collect(), (hh..h3).map(v3).collect())
    });

    let bit_layer = |ell: usize, _zx: &[Gf], _suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
        if ell == depth - 3 {
            // JIT: level d−2 (Dense) per tree — paired T2 gathers.
            let hh = h2 >> 1;
            let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..num_trees)
                .map(|c| {
                    let streams: Vec<&[u64]> = m_rows.iter().map(|r| &r[c][..]).collect();
                    let v2 = |i: usize| ld1(&streams, i) * ld1(&streams, i + h2);
                    GroupBufs::Dense(vec![(
                        (0..hh).map(v2).collect(),
                        (hh..h2).map(v2).collect(),
                    )])
                })
                .collect();
            Some(BitLayer {
                bufs,
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else if ell == depth - 2 {
            // JIT: level d−1 (Dense) per tree — one T2 gather per value.
            let hh = q2 >> 1;
            let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..num_trees)
                .map(|c| {
                    let streams: Vec<&[u64]> = m_rows.iter().map(|r| &r[c][..]).collect();
                    GroupBufs::Dense(vec![(
                        (0..hh).map(|i| ld1(&streams, i)).collect(),
                        (hh..q2).map(|i| ld1(&streams, i)).collect(),
                    )])
                })
                .collect();
            Some(BitLayer {
                bufs,
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else if ell == depth - 1 {
            // The LEAF layer (Dense, JIT): direct case-table gathers.
            // (t2 stays alive to the end — a few MB at deployed depths.)
            let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..num_trees)
                .map(|c| {
                    let streams: Vec<&[u64]> = m_rows.iter().map(|r| &r[c][..]).collect();
                    let leaf = |i: usize| case_pow[i][rlc_case(&streams, i)];
                    GroupBufs::Dense(vec![(
                        (0..q2).map(leaf).collect(),
                        (q2..row_len).map(leaf).collect(),
                    )])
                })
                .collect();
            Some(BitLayer {
                bufs,
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else {
            None
        }
    };
    drive_grouped(
        transcript,
        roots,
        ForestLevels::PerTree(levels),
        bit_layer,
        depth,
        s,
        num_trees,
    )
}

/// Multi-claim batched lazy prover: `claims.len()` same-shape claims —
/// each `2^s` trees of the same depth over its OWN lane-packed bits and
/// τ chains `(packed_cols, pow2)` — run as ONE merged forest of
/// `N·2^s` trees (tree index `(n ≪ s) | c`, claim index HIGH), sharing
/// every layer's sumcheck rounds and messages. Per-claim τ selection
/// rides the driver's existing `tau_set` index — the kernels are
/// untouched. `N` must be a power of two (callers pad with zero-weight
/// claims: their τ chains are all 1, so their leaves are identically 1
/// whatever bits they carry) and the depth ≥ 4 (the deployed x shapes
/// have `t' ≥ 6`). Uses the shared automatic policy or an explicit L4/L8 schedule.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_merged_forest_lazy_multi(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    claims: &[(&[Vec<u64>], &[Vec<Gf>])],
) -> Result<(Vec<Gf>, MergedForestProof, Vec<Gf>, Gf), schedule::UnsupportedSchedule> {
    let schedule = configured(p, ForestPath::Multi)?;
    Ok(prove_merged_forest_lazy_multi_sched(
        transcript,
        p,
        claims,
        schedule.is_l8(),
    ))
}

/// [`prove_merged_forest_lazy_multi`] with the schedule explicit — the
/// testable entry point; both schedules pinned byte-identical to eager.
#[allow(clippy::arithmetic_side_effects)]
fn prove_merged_forest_lazy_multi_sched(
    transcript: &mut impl Transcript,
    p: &IntegerMatrixLayout,
    claims: &[(&[Vec<u64>], &[Vec<Gf>])],
    l8: bool,
) -> (Vec<Gf>, MergedForestProof, Vec<Gf>, Gf) {
    use crate::pcs::{extract_column_bit_halves, layer1_pair_table, leaf_tau_halves};
    let n_claims = claims.len();
    assert!(
        n_claims.is_power_of_two(),
        "pad the claim list to a power of two"
    );
    let log_n = n_claims.trailing_zeros() as usize;
    let log_w = p.word_bits.trailing_zeros() as usize;
    let mask_w = p.word_bits.wrapping_sub(1);
    let row_len = p.rows() << log_w;
    let depth = row_len.trailing_zeros() as usize;
    assert!(
        depth >= 4,
        "the batched x prover assumes t' >= 4 (deployed: >= 6)"
    );
    let s = p.col_vars;
    let per = p.cols();
    let num_trees = n_claims * per;
    let s_batch = s + log_n;
    let one = Gf::one();

    // Per-claim tables — the single-claim ones, DEDUPED by τ-chain
    // identity: claims passing the same `pow2` slice (same-point claims,
    // and every zero-weight dummy) share one table set; the driver's
    // `tau_set` field selects per tree via `tab_of`.
    let q1 = row_len >> 2;
    let q2 = row_len >> 1;
    struct TauTables {
        leaf_tau: (Vec<Gf>, Vec<Gf>),
        te: Vec<Gf>,
        to: Vec<Gf>,
        t4: Vec<Gf>,
    }
    let mut uniq_pow2: Vec<(*const Vec<Gf>, usize)> = Vec::new();
    let tab_of: Vec<usize> = claims
        .iter()
        .map(|&(_, pow2)| {
            let key = (pow2.as_ptr(), pow2.len());
            match uniq_pow2.iter().position(|&k| k == key) {
                Some(u) => u,
                None => {
                    uniq_pow2.push(key);
                    uniq_pow2.len() - 1
                }
            }
        })
        .collect();
    let uniq_reps: Vec<usize> = {
        let mut reps = vec![usize::MAX; uniq_pow2.len()];
        for (n, &u) in tab_of.iter().enumerate() {
            if reps[u] == usize::MAX {
                reps[u] = n;
            }
        }
        reps
    };
    drop(uniq_pow2);
    let mut tabs: Vec<TauTables> = uniq_reps
        .iter()
        .map(|&n| {
            let pow2 = claims[n].1;
            let pair_tbl = layer1_pair_table(p, pow2, log_w, row_len);
            let leaf_tau = leaf_tau_halves(p, pow2, one, log_w, row_len);
            let v = |i: usize| -> Gf { pow2[i >> log_w][i & mask_w] };
            let build_cases = |base: usize| -> Vec<Gf> {
                let mut t = Vec::with_capacity(q1 << 2);
                for y in 0..q1 {
                    let lo = base + y;
                    t.push(one);
                    t.push(v(lo));
                    t.push(v(lo + q2));
                    t.push(pair_tbl[lo]);
                }
                t
            };
            let te = build_cases(0);
            let to = build_cases(q1);
            // Indexed collect + in-place flatten — the parallel-`flatten`
            // nested-iterator pathology (see the single prover's t4 build).
            let t4: Vec<Gf> = {
                let rows: Vec<[Gf; 16]> = cfg_into_iter!(0..q1, 1 << 10)
                    .map(|y| {
                        let mut row = [Gf::one(); 16];
                        for (c, slot) in row.iter_mut().enumerate() {
                            *slot = te[(y << 2) | (c >> 2)] * to[(y << 2) | (c & 3)];
                        }
                        row
                    })
                    .collect();
                rows.into_flattened()
            };
            TauTables {
                leaf_tau,
                te,
                to,
                t4,
            }
        })
        .collect();
    let extracted: Vec<_> = claims
        .iter()
        .map(|&(packed_cols, _)| extract_column_bit_halves(packed_cols, per, row_len))
        .collect();
    if depth == 4 || !l8 {
        // The L/4 schedule — the DEFAULT (and forced at depth 4, where
        // the deeper chain degenerates): stored chain tops at level d−3,
        // level d−2 JIT, Pair2Bits / Leaf2Bits.
        let h3 = q1 >> 1; // level d−3 positions = 2^{d−3}
        let (levels, roots) = build_levels(num_trees, depth - 3, |k| {
            let (n, c) = (k >> s, k & (per - 1));
            let t4 = &tabs[tab_of[n]].t4;
            let (lb, rb) = &extracted[n][c];
            // Paired T4 gathers — level d−2 never materialises (see the
            // single prover's L/4 build).
            let at = T4At {
                lbits: lb,
                rbits: rb,
                t4: T4Src::Pre(t4),
                q1,
            };
            let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
            let v3 = |y: usize| -> Gf {
                if t4_pf {
                    let yp = y + PRFM_DIST;
                    if yp < h3 {
                        at.prefetch_at(yp);
                        at.prefetch_at(yp + h3);
                    }
                }
                at.at(y) * at.at(y + h3)
            };
            let hh = h3 >> 1;
            ((0..hh).map(v3).collect(), (hh..h3).map(v3).collect())
        });
        let jit_r1 = jit_round1_fuse();
        let bit_layer =
            |ell: usize, zx: &[Gf], suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
                if ell == depth - 3 {
                    // JIT: regenerate level d−2 (Dense) per tree — alive
                    // only while this layer runs; the default fuses the
                    // round-1 message into the generation pass.
                    let hh = q1 >> 1;
                    let (bufs, round1) = if jit_r1 {
                        jit_layer_generate(hh, zx, num_trees, suffix, |k| {
                            let (n, c) = (k >> s, k & (per - 1));
                            let t4 = &tabs[tab_of[n]].t4;
                            let (lb, rb) = &extracted[n][c];
                            let at = T4At {
                                lbits: lb,
                                rbits: rb,
                                t4: T4Src::Pre(t4),
                                q1,
                            };
                            let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
                            (
                                move |j| at.at(j),
                                move |j| {
                                    if t4_pf {
                                        at.prefetch_at(j);
                                    }
                                },
                            )
                        })
                    } else {
                        let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..num_trees)
                            .map(|k| {
                                let (n, c) = (k >> s, k & (per - 1));
                                let t4 = &tabs[tab_of[n]].t4;
                                let (lb, rb) = &extracted[n][c];
                                // Exact-capacity halves: a split_off would carry
                                // 2× allocation through the layer.
                                let full = t4_level_values(lb, rb, T4Src::Pre(t4), q1);
                                GroupBufs::Dense(vec![(full[..hh].to_vec(), full[hh..].to_vec())])
                            })
                            .collect();
                        (bufs, None)
                    };
                    for tab in tabs.iter_mut() {
                        tab.t4 = Vec::new();
                    }
                    Some(BitLayer {
                        bufs,
                        tau_sets: Vec::new(),
                        pair_tau_sets: Vec::new(),
                        t4_sets: Vec::new(),
                        round1,
                        flat: None,
                    })
                } else if ell == depth - 2 {
                    Some(BitLayer {
                        bufs: (0..num_trees)
                            .map(|k| {
                                let (n, c) = (k >> s, k & (per - 1));
                                let (lbits, rbits) = &extracted[n][c];
                                GroupBufs::Pair2Bits {
                                    lbits,
                                    rbits,
                                    tau_set: tab_of[n],
                                }
                            })
                            .collect(),
                        tau_sets: Vec::new(),
                        pair_tau_sets: tabs
                            .iter()
                            .map(|t| Pair2TauSet {
                                te: t.te.clone(),
                                to: t.to.clone(),
                            })
                            .collect(),
                        t4_sets: Vec::new(),
                        round1: None,
                        flat: None,
                    })
                } else if ell == depth - 1 {
                    // Two bit-driven rounds (k = d−1 = 3): dense buffers only
                    // after round 2.
                    Some(BitLayer {
                        bufs: extracted
                            .iter()
                            .enumerate()
                            .flat_map(|(n, cb)| {
                                let ts = tab_of[n];
                                cb.iter().map(move |(lbits, rbits)| GroupBufs::Leaf2Bits {
                                    lbits,
                                    rbits,
                                    tau_set: ts,
                                })
                            })
                            .collect(),
                        tau_sets: tabs.iter().map(|t| t.leaf_tau.clone()).collect(),
                        pair_tau_sets: Vec::new(),
                        t4_sets: Vec::new(),
                        round1: None,
                        flat: None,
                    })
                } else {
                    None
                }
            };
        return drive_grouped(
            transcript,
            roots,
            ForestLevels::PerTree(levels),
            bit_layer,
            depth,
            s_batch,
            num_trees,
        );
    }

    // depth ≥ 5 — the L/8 schedule (see the single prover): stored chain
    // tops at level d−4, layer d−4 JIT-regenerates level d−3, layers
    // d−3 / d−2 / d−1 run T4Bits / Pair3Bits / Leaf3Bits. T4 moves into
    // the T4Bits layer's tau sets (zero-copy) and dies with it.
    let h3 = q1 >> 1; // level d−3 positions = 2^{d−3}
    let h4 = q1 >> 2; // level d−4 positions = 2^{d−4}
    let (levels, roots) = build_levels(num_trees, depth - 4, |k| {
        let (n, c) = (k >> s, k & (per - 1));
        let t4 = &tabs[tab_of[n]].t4;
        let (lb, rb) = &extracted[n][c];
        // Paired T4 gathers — level d−2 never materialises (see the
        // single prover's L/8 build).
        let at = T4At {
            lbits: lb,
            rbits: rb,
            t4: T4Src::Pre(t4),
            q1,
        };
        let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
        let v4 = |y: usize| -> Gf {
            if t4_pf {
                let yp = y + PRFM_DIST;
                if yp < h4 {
                    at.prefetch_at(yp);
                    at.prefetch_at(yp + h3);
                    at.prefetch_at(yp + h4);
                    at.prefetch_at(yp + h4 + h3);
                }
            }
            (at.at(y) * at.at(y + h3)) * (at.at(y + h4) * at.at(y + h4 + h3))
        };
        let hh = h4 >> 1;
        ((0..hh).map(v4).collect(), (hh..h4).map(v4).collect())
    });

    let jit_r1 = jit_round1_fuse();
    let bit_layer = |ell: usize, zx: &[Gf], suffix: &SuffixTensorArena<Gf>| -> Option<BitLayer> {
        if ell == depth - 4 {
            // JIT: regenerate level d−3 (Dense, exact-capacity halves)
            // per tree — transient ≈ L/8; the default fuses the round-1
            // message into the generation pass.
            let hh = h3 >> 1;
            let (bufs, round1) = if jit_r1 {
                jit_layer_generate(hh, zx, num_trees, suffix, |k| {
                    let (n, c) = (k >> s, k & (per - 1));
                    let t4 = &tabs[tab_of[n]].t4;
                    let (lb, rb) = &extracted[n][c];
                    let at = T4At {
                        lbits: lb,
                        rbits: rb,
                        t4: T4Src::Pre(t4),
                        q1,
                    };
                    let t4_pf = t4_prfm(t4.len() * core::mem::size_of::<Gf>());
                    (
                        move |y| at.at(y) * at.at(y + h3),
                        move |y| {
                            if t4_pf {
                                at.prefetch_at(y);
                                at.prefetch_at(y + h3);
                            }
                        },
                    )
                })
            } else {
                let bufs: Vec<GroupBufs<'_, Gf>> = cfg_into_iter!(0..num_trees)
                    .map(|k| {
                        let (n, c) = (k >> s, k & (per - 1));
                        let t4 = &tabs[tab_of[n]].t4;
                        let (lb, rb) = &extracted[n][c];
                        let full = t4_level_values(lb, rb, T4Src::Pre(t4), q1);
                        let e: Vec<Gf> = (0..hh).map(|y| full[y] * full[y + h3]).collect();
                        let o: Vec<Gf> = (hh..h3).map(|y| full[y] * full[y + h3]).collect();
                        GroupBufs::Dense(vec![(e, o)])
                    })
                    .collect();
                (bufs, None)
            };
            Some(BitLayer {
                bufs,
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1,
                flat: None,
            })
        } else if ell == depth - 3 {
            // One bit-driven round straight off T4 (k = d−3 ≥ 2).
            let bufs: Vec<GroupBufs<'_, Gf>> = (0..num_trees)
                .map(|k| {
                    let (n, c) = (k >> s, k & (per - 1));
                    let (lbits, rbits) = &extracted[n][c];
                    GroupBufs::T4Bits {
                        lbits,
                        rbits,
                        tau_set: tab_of[n],
                    }
                })
                .collect();
            Some(BitLayer {
                bufs,
                tau_sets: Vec::new(),
                pair_tau_sets: Vec::new(),
                t4_sets: tabs
                    .iter_mut()
                    .map(|t| core::mem::take(&mut t.t4))
                    .collect(),
                round1: None,
                flat: None,
            })
        } else if ell == depth - 2 {
            // Two bit-driven rounds (k = d−2 ≥ 3).
            Some(BitLayer {
                bufs: (0..num_trees)
                    .map(|k| {
                        let (n, c) = (k >> s, k & (per - 1));
                        let (lbits, rbits) = &extracted[n][c];
                        GroupBufs::Pair3Bits {
                            lbits,
                            rbits,
                            tau_set: tab_of[n],
                        }
                    })
                    .collect(),
                tau_sets: Vec::new(),
                pair_tau_sets: tabs
                    .iter()
                    .map(|t| Pair2TauSet {
                        te: t.te.clone(),
                        to: t.to.clone(),
                    })
                    .collect(),
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else if ell == depth - 1 {
            // Three bit-driven rounds (k = d−1 ≥ 4): leaf set ≈ L/8.
            Some(BitLayer {
                bufs: extracted
                    .iter()
                    .enumerate()
                    .flat_map(|(n, cb)| {
                        let ts = tab_of[n];
                        cb.iter().map(move |(lbits, rbits)| GroupBufs::Leaf3Bits {
                            lbits,
                            rbits,
                            tau_set: ts,
                        })
                    })
                    .collect(),
                tau_sets: tabs.iter().map(|t| t.leaf_tau.clone()).collect(),
                pair_tau_sets: Vec::new(),
                t4_sets: Vec::new(),
                round1: None,
                flat: None,
            })
        } else {
            None
        }
    };
    drive_grouped(
        transcript,
        roots,
        ForestLevels::PerTree(levels),
        bit_layer,
        depth,
        s_batch,
        num_trees,
    )
}

/// Verify; returns `(exit_point, exit_eval)`. The caller supplies the roots
/// (they are part of the enclosing proof and get absorbed here).
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_merged_forest(
    transcript: &mut impl Transcript,
    roots: &[Gf],
    proof: &MergedForestProof,
    depth: usize,
    s: usize,
) -> Result<(Vec<Gf>, Gf), MergedForestError> {
    if roots.len() != 1usize << s || proof.layers.len() != depth {
        return Err(MergedForestError::Shape);
    }
    let one = Gf::one();
    absorb_gfs(transcript, 0x30, roots);
    let zeta: Vec<Gf> = transcript.get_field_challenges(s, &());
    let mut claim = mle_at(roots, &zeta);

    let mut z_x: Vec<Gf> = Vec::new();
    let mut z_c: Vec<Gf> = zeta;
    for (ell, layer) in proof.layers.iter().enumerate() {
        // Arity-2 layers never carry a closing quad — reject it here so a
        // stream with a smuggled `pair2` (codec flag bit 2) cannot decode
        // to an accepted proof it would otherwise silently ignore.
        if layer.pair2.is_some() {
            return Err(MergedForestError::Shape);
        }
        // Phase A: the ℓ in-tree variables. Its claimed sum must be the
        // running layer claim; its expected evaluation factors as
        // eq(r_x, z_x) · (phase B's claimed sum).
        let r_x = if ell == 0 {
            if layer.sc_x.is_some() {
                return Err(MergedForestError::Shape);
            }
            if layer.sc_c.claimed_sum != claim {
                return Err(MergedForestError::LayerClaim { layer: ell });
            }
            Vec::new()
        } else {
            let sc_x = layer.sc_x.as_ref().ok_or(MergedForestError::Shape)?;
            if sc_x.claimed_sum != claim {
                return Err(MergedForestError::LayerClaim { layer: ell });
            }
            let sub = verify_eq_inner_sumcheck_gruen(transcript, &z_x, sc_x, &())
                .map_err(|_| MergedForestError::LayerClaim { layer: ell })?;
            let eqx = eq_eval(&sub.point, &z_x, one).map_err(|_| MergedForestError::Shape)?;
            if sub.expected_evaluation != eqx * layer.sc_c.claimed_sum {
                return Err(MergedForestError::LayerClaim { layer: ell });
            }
            sub.point
        };

        // Phase B: the s tree-index variables, closing on the child pair.
        let sub_c = verify_eq_inner_sumcheck_gruen(transcript, &z_c, &layer.sc_c, &())
            .map_err(|_| MergedForestError::LayerClaim { layer: ell })?;
        let (p_, q_) = layer.pair;
        let eqc = eq_eval(&sub_c.point, &z_c, one).map_err(|_| MergedForestError::Shape)?;
        if sub_c.expected_evaluation != eqc * p_ * q_ {
            return Err(MergedForestError::LayerClaim { layer: ell });
        }

        absorb_gfs(transcript, 0x32, &[p_, q_]);
        let mu: Gf = transcript.get_field_challenge(&());
        claim = p_ + mu * (p_ + q_);
        let mut nx = r_x;
        nx.push(mu);
        z_x = nx;
        z_c = sub_c.point;
    }
    let mut z = z_x;
    z.extend_from_slice(&z_c);
    Ok((z, claim))
}

/// Proof bytes: the per-layer sumcheck messages + closing pair.
#[allow(clippy::arithmetic_side_effects)]
pub fn merged_forest_proof_size_bytes(proof: &MergedForestProof) -> usize {
    use crate::transcript::traits::Transcribable;
    let mut n = 0usize;
    for l in &proof.layers {
        if let Some(sc) = &l.sc_x {
            n += sc.get_num_bytes();
        }
        n += l.sc_c.get_num_bytes();
        n += 2 * 16;
        if l.pair2.is_some() {
            n += 2 * 16;
        }
    }
    n
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

    #[test]
    fn merged_forest_roundtrips() {
        for (depth, s) in [(1usize, 0usize), (1, 2), (3, 2), (5, 3), (6, 4)] {
            let leaves: Vec<Gf> = (0..(1usize << (depth + s)))
                .map(|i| sample(0x9000 + i as u64))
                .collect();
            let mut pt = Blake3Transcript::new();
            let (roots, proof, z_p, e_p) = prove_merged_forest(&mut pt, &leaves, depth, s);
            // Roots are the per-tree products.
            for c in 0..(1usize << s) {
                let prod = leaves[c << depth..(c + 1) << depth]
                    .iter()
                    .fold(Gf::one(), |a, &b| a * b);
                assert_eq!(roots[c], prod, "root {c}");
            }
            let mut vt = Blake3Transcript::new();
            let (z_v, e_v) =
                verify_merged_forest(&mut vt, &roots, &proof, depth, s).expect("verify");
            assert_eq!((&z_p, e_p), (&z_v, e_v), "exit claims agree");
            // Exit claim is the leaf MLE at the exit point.
            assert_eq!(mle_at(&leaves, &z_v), e_v, "exit eval");

            // Tampers.
            if depth >= 2 {
                let mut bad = proof.clone();
                if let Some(sc) = &mut bad.layers[1].sc_x {
                    sc.claimed_sum += Gf::one();
                }
                let mut vt = Blake3Transcript::new();
                assert!(verify_merged_forest(&mut vt, &roots, &bad, depth, s).is_err());
            }
            let mut bad = proof.clone();
            bad.layers[0].pair.0 += Gf::one();
            let mut vt = Blake3Transcript::new();
            assert!(verify_merged_forest(&mut vt, &roots, &bad, depth, s).is_err());
            let mut bad = proof.clone();
            bad.layers[depth - 1].sc_c.claimed_sum += Gf::one();
            let mut vt = Blake3Transcript::new();
            assert!(verify_merged_forest(&mut vt, &roots, &bad, depth, s).is_err());
        }
    }

    /// The multi-claim batched prover must be byte-identical to the eager
    /// prover over the concatenated per-claim leaf tables (tree index
    /// `(n ≪ s) | c`, claim high). The last claim's τ chains are all 1 —
    /// the zero-weight DUMMY-padding shape (leaves identically 1).
    #[test]
    fn lazy_multi_matches_eager() {
        use crate::ligerito::pack_columns_from_rows;
        // Depths 4..=8: 4 = the multi prover's minimum (degenerate JIT
        // edge — stored chain tops at level 1), 5 = shallow chain, 6/7 =
        // the deployed x shapes (scalar t4-gen), 8 = the word-wise
        // t4-gen path (q1 = 64).
        for (t, s, n_claims) in [
            (4usize, 2usize, 2usize),
            (5, 1, 2),
            (6, 2, 2),
            (7, 3, 4),
            (6, 1, 4),
            (8, 1, 2),
        ] {
            let p = IntegerMatrixLayout {
                row_vars: t,
                col_vars: s,
                word_bits: 1,
            };
            let row_len = p.rows();
            let depth = t;
            let log_n = n_claims.trailing_zeros() as usize;
            let words = row_len.div_ceil(64);
            let mut rows_all = Vec::with_capacity(n_claims);
            let mut pow2_all = Vec::with_capacity(n_claims);
            for n in 0..n_claims {
                let rows: Vec<Vec<u64>> = (0..p.cols())
                    .map(|c| {
                        (0..words)
                            .map(|wd| {
                                (c as u64 + 7 * n as u64 + 1)
                                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                    .wrapping_add(wd as u64)
                                    .rotate_left((c + 5 * wd + n) as u32 & 63)
                            })
                            .collect()
                    })
                    .collect();
                let pow2: Vec<Vec<Gf>> = (0..p.rows())
                    .map(|b| {
                        if n == n_claims - 1 {
                            vec![Gf::one()] // the dummy-padding shape
                        } else {
                            vec![sample(0xB0B + (n * p.rows() + b) as u64)]
                        }
                    })
                    .collect();
                rows_all.push(rows);
                pow2_all.push(pow2);
            }
            // SAME-WEIGHT sharing: claim 1 uses claim 0's τ chains, passed
            // as the SAME slice reference — the multi prover's table dedup
            // must stay byte-identical to the eager run over those values.
            if n_claims >= 3 {
                pow2_all[1] = pow2_all[0].clone();
            }
            let packed_all: Vec<Vec<Vec<u64>>> = rows_all
                .iter()
                .map(|r| pack_columns_from_rows(&p, r))
                .collect();

            // Dense concatenated leaves, tree index (n ≪ s) | c.
            let dense: Vec<Gf> = (0..(row_len << (s + log_n)))
                .map(|idx| {
                    let k = idx >> depth;
                    let i = idx & (row_len - 1);
                    let (n, c) = (k >> s, k & ((1 << s) - 1));
                    if (rows_all[n][c][i >> 6] >> (i & 63)) & 1 == 1 {
                        pow2_all[n][i][0]
                    } else {
                        Gf::one()
                    }
                })
                .collect();

            let mut t_eager = Blake3Transcript::new();
            let eager = prove_merged_forest(&mut t_eager, &dense, depth, s + log_n);
            let claim_refs: Vec<(&[Vec<u64>], &[Vec<Gf>])> = packed_all
                .iter()
                .enumerate()
                .map(|(n, pc)| {
                    let pw = if n == 1 && n_claims >= 3 {
                        &pow2_all[0]
                    } else {
                        &pow2_all[n]
                    };
                    (&pc[..], &pw[..])
                })
                .collect();
            let ce: Gf = t_eager.get_field_challenge(&());
            // BOTH schedules must be byte-identical to the eager run
            // (l8 = false → the L/4 default; true → the opt-in L/8,
            // which silently falls back to L/4 at depth 4).
            for l8 in [false, true] {
                let mut t_multi = Blake3Transcript::new();
                let multi = prove_merged_forest_lazy_multi_sched(&mut t_multi, &p, &claim_refs, l8);

                assert_eq!(eager.0, multi.0, "roots (t={t},s={s},N={n_claims},l8={l8})");
                assert_eq!(eager.2, multi.2, "exit point (l8={l8})");
                assert_eq!(eager.3, multi.3, "exit eval (l8={l8})");
                for (k, (le, lm)) in eager.1.layers.iter().zip(multi.1.layers.iter()).enumerate() {
                    assert_eq!(le.sc_x, lm.sc_x, "layer {k} sc_x (l8={l8})");
                    assert_eq!(le.sc_c, lm.sc_c, "layer {k} sc_c (l8={l8})");
                    assert_eq!(le.pair, lm.pair, "layer {k} pair (l8={l8})");
                }
                let cm: Gf = t_multi.get_field_challenge(&());
                assert_eq!(ce, cm, "transcript states diverged (l8={l8})");
                // Dummy-shaped claim: roots of the last claim's trees are
                // the products of its ALL-ONES leaves.
                for c in 0..1usize << s {
                    assert_eq!(
                        multi.0[((n_claims - 1) << s) | c],
                        Gf::one(),
                        "dummy root {c} (l8={l8})"
                    );
                }
                let mut vt = Blake3Transcript::new();
                let (z_v, e_v) =
                    verify_merged_forest(&mut vt, &multi.0, &multi.1, depth, s + log_n)
                        .expect("verify multi");
                assert_eq!(z_v, multi.2);
                assert_eq!(e_v, multi.3);
            }
        }
    }

    /// The lazy bit-affine prover must be byte-identical to the eager one
    /// over the same `bit ? α-power : 1` leaves: same roots, same proof
    /// values, same exit claim, same transcript state.
    #[test]
    fn lazy_matches_eager() {
        use crate::ligerito::pack_columns_from_rows;
        // Depths 3..=9 (row_len = 2^t·W): 3 = LeafBits-only fallback; 4 =
        // the degenerate JIT edge (stored chain tops at level 1, Pair2Bits
        // k=2, Leaf2Bits k=3); 5/6 = shallow JIT chains (scalar t4-gen
        // path, q1 < 64); 7 = deep scalar; 8/9 = the WORD-wise t4-gen
        // path (q1 = 64 / 128, one / two block strides).
        for (t, s, w) in [
            (3usize, 1usize, 1usize),
            (4, 2, 1),
            (5, 1, 1),
            (6, 2, 1),
            (7, 3, 1),
            (5, 2, 4),
            (8, 1, 1),
            (9, 2, 1),
        ] {
            let p = IntegerMatrixLayout {
                row_vars: t,
                col_vars: s,
                word_bits: w,
            };
            let log_w = w.trailing_zeros() as usize;
            let row_len = p.rows() << log_w;
            let depth = row_len.trailing_zeros() as usize;
            // Arbitrary per-row α-power chains (shape only; values free).
            let pow2: Vec<Vec<Gf>> = (0..p.rows())
                .map(|b| {
                    (0..w)
                        .map(|j| sample(0xA11CE + (b * w + j) as u64))
                        .collect()
                })
                .collect();
            let words = row_len.div_ceil(64);
            let rows: Vec<Vec<u64>> = (0..p.cols())
                .map(|c| {
                    (0..words)
                        .map(|wd| {
                            (c as u64 + 1)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                .wrapping_add(wd as u64)
                                .rotate_left((c + 3 * wd) as u32 & 63)
                        })
                        .collect()
                })
                .collect();
            let packed_cols = pack_columns_from_rows(&p, &rows);
            let mask_w = w.wrapping_sub(1);
            let dense: Vec<Gf> = (0..(row_len << s))
                .map(|idx| {
                    let (c, i) = (idx >> depth, idx & (row_len - 1));
                    if (rows[c][i >> 6] >> (i & 63)) & 1 == 1 {
                        pow2[i >> log_w][i & mask_w]
                    } else {
                        Gf::one()
                    }
                })
                .collect();

            let mut t_eager = Blake3Transcript::new();
            let eager = prove_merged_forest(&mut t_eager, &dense, depth, s);
            let ce: Gf = t_eager.get_field_challenge(&());
            // EVERY schedule must be byte-identical (L/2 = level d−2
            // stored, no JIT; L/4 = the default; L/8 = opt-in, falling
            // back to L/4 at d ≤ 4).
            let mut last = None;
            for sched in [ForestSchedule::L2, ForestSchedule::L4, ForestSchedule::L8] {
                let mut t_lazy = Blake3Transcript::new();
                let lazy = prove_merged_forest_lazy_sched(
                    &mut t_lazy,
                    &p,
                    &packed_cols,
                    &pow2,
                    sched,
                    p.cols(),
                );

                assert_eq!(eager.0, lazy.0, "roots (t={t},s={s},W={w},{sched:?})");
                assert_eq!(eager.2, lazy.2, "exit point ({sched:?})");
                assert_eq!(eager.3, lazy.3, "exit eval ({sched:?})");
                for (k, (le, ll)) in eager.1.layers.iter().zip(lazy.1.layers.iter()).enumerate() {
                    assert_eq!(le.sc_x, ll.sc_x, "layer {k} sc_x ({sched:?})");
                    assert_eq!(le.sc_c, ll.sc_c, "layer {k} sc_c ({sched:?})");
                    assert_eq!(le.pair, ll.pair, "layer {k} pair ({sched:?})");
                }
                let cl: Gf = t_lazy.get_field_challenge(&());
                assert_eq!(ce, cl, "transcript states diverged ({sched:?})");
                let mut t_borrowed = Blake3Transcript::new();
                let borrowed = prove_merged_forest_lazy_impl(
                    &mut t_borrowed,
                    &p,
                    Some(&rows),
                    &packed_cols,
                    &pow2,
                    sched,
                    p.cols(),
                );
                assert_eq!(borrowed.0, lazy.0, "borrowed roots");
                assert_eq!(borrowed.2, lazy.2, "borrowed point");
                assert_eq!(borrowed.3, lazy.3, "borrowed evaluation");
                for (a, b) in borrowed.1.layers.iter().zip(&lazy.1.layers) {
                    assert_eq!(a.sc_x, b.sc_x);
                    assert_eq!(a.sc_c, b.sc_c);
                    assert_eq!(a.pair, b.pair);
                }
                assert_eq!(
                    t_borrowed.get_field_challenge::<Gf>(&()),
                    ce,
                    "borrowed transcript"
                );
                last = Some(lazy);
            }
            let lazy = last.expect("every schedule ran");

            // And the verifier accepts the lazy proof.
            let mut vt = Blake3Transcript::new();
            let (z_v, e_v) =
                verify_merged_forest(&mut vt, &lazy.0, &lazy.1, depth, s).expect("verify lazy");
            assert_eq!(z_v, lazy.2);
            assert_eq!(e_v, lazy.3);
        }
    }

    /// **Live-column elision is byte-identical.** A zero-padded witness
    /// ends in all-zero columns, whose trees are constant `α^0 = 1`;
    /// eliding them (building only `live` trees and carrying the constant
    /// contribution analytically) must reproduce the
    /// un-elided forest exactly — same roots, same round polynomials,
    /// same exit claim, same transcript state — under every schedule, and
    /// the un-elided verifier must accept.
    #[test]
    fn col_elision_matches_full() {
        use crate::ligerito::pack_columns_from_rows;
        // Depths 3..=9 again, with ≥ 2 columns so a zero tail exists;
        // `live` sweeps the interesting fills (a single live column, an
        // odd split, and the no-tail case).
        for (t, s, w) in [
            (3usize, 1usize, 1usize),
            (4, 2, 1),
            (6, 2, 1),
            (7, 3, 1),
            (5, 2, 4),
            (9, 2, 1),
        ] {
            let p = IntegerMatrixLayout {
                row_vars: t,
                col_vars: s,
                word_bits: w,
            };
            let log_w = w.trailing_zeros() as usize;
            let row_len = p.rows() << log_w;
            let pow2: Vec<Vec<Gf>> = (0..p.rows())
                .map(|b| (0..w).map(|j| sample(0xB0B + (b * w + j) as u64)).collect())
                .collect();
            let words = row_len.div_ceil(64);
            for live in 0..=p.cols() {
                // Columns `live..2^s` are ALL ZERO — the padding tail.
                let rows: Vec<Vec<u64>> = (0..p.cols())
                    .map(|c| {
                        (0..words)
                            .map(|wd| {
                                if c >= live {
                                    0
                                } else {
                                    (c as u64 + 1)
                                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                        .wrapping_add(wd as u64)
                                        .rotate_left((c + 3 * wd) as u32 & 63)
                                }
                            })
                            .collect()
                    })
                    .collect();
                let packed_cols = pack_columns_from_rows(&p, &rows);
                // The detector must find exactly the zero tail (the
                // last live column is non-zero by construction).
                assert_eq!(
                    live_cols(&p, &rows),
                    live.max(1),
                    "detector (t={t},s={s},W={w})"
                );

                for sched in [ForestSchedule::L2, ForestSchedule::L4, ForestSchedule::L8] {
                    let mut t_full = Blake3Transcript::new();
                    let full = prove_merged_forest_lazy_sched(
                        &mut t_full,
                        &p,
                        &packed_cols,
                        &pow2,
                        sched,
                        p.cols(),
                    );
                    let cf: Gf = t_full.get_field_challenge(&());

                    let mut t_el = Blake3Transcript::new();
                    let el = prove_merged_forest_lazy_sched(
                        &mut t_el,
                        &p,
                        &packed_cols,
                        &pow2,
                        sched,
                        live,
                    );
                    let ce: Gf = t_el.get_field_challenge(&());

                    let tag = format!("t={t},s={s},W={w},live={live},{sched:?}");
                    assert_eq!(full.0, el.0, "roots ({tag})");
                    assert_eq!(full.2, el.2, "exit point ({tag})");
                    assert_eq!(full.3, el.3, "exit eval ({tag})");
                    assert_eq!(
                        full.1.layers.len(),
                        el.1.layers.len(),
                        "layer count ({tag})"
                    );
                    for (k, (lf, le)) in full.1.layers.iter().zip(el.1.layers.iter()).enumerate() {
                        assert_eq!(lf.sc_x, le.sc_x, "layer {k} sc_x ({tag})");
                        assert_eq!(lf.sc_c, le.sc_c, "layer {k} sc_c ({tag})");
                        assert_eq!(lf.pair, le.pair, "layer {k} pair ({tag})");
                    }
                    assert_eq!(cf, ce, "transcript states diverged ({tag})");

                    // The elided proof verifies against the un-elided
                    // verifier, which knows nothing about elision.
                    let mut vt = Blake3Transcript::new();
                    let depth = row_len.trailing_zeros() as usize;
                    let (z_v, e_v) = verify_merged_forest(&mut vt, &el.0, &el.1, depth, s)
                        .expect("verify elided");
                    assert_eq!(z_v, el.2, "verifier exit point ({tag})");
                    assert_eq!(e_v, el.3, "verifier exit eval ({tag})");
                }
            }
        }
    }
}
