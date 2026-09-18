//! Structured-tap virtual claims (EXPERIMENTAL — `docs/rlc-structured-taps-phase0.md`).
//!
//! Setting (2026-07-27 corrected semantics): W = 1 bit-vectors whose flat
//! entry index `p` is grouped into `2^g`-bit words along the ENTRY axis —
//! `p = (k ≪ g) | j`, word `k`, position `j` within the group (the
//! motivating instance: 32-bit words, `g = 5`). A *tap* applies an
//! `F₂`-linear, index-structured op to a committed UAIR column:
//!
//! * `ROT^r`: output `(k, j)` reads input `(k, (j − r) mod 2^g)` — a
//!   cyclic translation of the low-`g` field;
//! * `SHIFT^r`: the same with dropout (output `j < r` is zero, input
//!   `j ≥ 2^g − r` drops out);
//! * `off^o` (word offset): output word `k` reads input word `k − o`,
//!   zero for `k < o` — a translation by `o·2^g` on the word field.
//!
//! Equivalently `b = M·a` for a banded block-rotation matrix `M`; the tap
//! descriptor is the succinctness-preserving normal form of that `M` —
//! an arbitrary `M` would cost the verifier `O(N)` at the residual
//! closure, while every tap term is an index *translation on a contiguous
//! bit-field*, so its opening basis is a **translated-eq**: not an eq
//! tensor (index translation is not a coordinate permutation — the
//! Phase-0 note records the counterexample) but a **carry matrix product
//! of bond dimension 2** (binary addition, LSB→MSB). This module holds
//! the machinery all tap surfaces share:
//!
//! * the tapped-row extraction (whole-run gathers: the group field lives
//!   in the clear axis, so ROT/SHIFT permute clear rows and the word
//!   offset shifts the `row_hi` runs by the borrow),
//! * the per-class *in-pack tables* `A_β(v)` (the ring-switch claim check),
//! * the per-class *support factor tables* (translated slices of plain eq
//!   tables) that drive the prover's sparse `s_v` walks and basis fills,
//! * the per-class *closure descriptor* + the matrix-product-state
//!   generalization of [`crate::ligerito::residual_b_evals`] (the
//!   verifier's succinct `O(m·128²)` basis evaluation).
//!
//! Classes: the group chain is confined to the low-`g` clear coordinates
//! (always outside the pack); only the word chain crosses the clear/fold
//! (`s`-bit) boundary and possibly the pack cut, so the weight splits
//! `K̃(v, y) = Σ_β A_β(v)·B_β(y)` over the reachable
//! `β = (γ, c₇) ∈ {(0,0)} ∪ {(1,0), (1,1) if off > 0}` — at most **3**
//! classes, exactly one nonzero per position. Scope asserts (v1):
//! `x_fold_extra = 0`, `tw + log_cols ≥ 7` (in-pack coordinates are
//! `row_hi` and column bits only), `g ≤ s` (the group field inside the
//! clear axis), `off < 2^{s−g}` (word offsets below the clear field's
//! width).

use crate::pcs::ShaF2Layout;
use crate::poly::univariate::binary_gf128::Gf128 as Gf;
use crate::poly::utils::build_eq_x_r_vec;
use crate::utils::cfg_into_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// One structured tap: an `F₂`-linear, index-structured operand applied
/// to a committed UAIR column, with the entry axis grouped into
/// `2^{grp_log2}`-bit words. The identity tap is
/// `{ col, grp_log2: _, bit_amt: 0, bit_dropout: false, off: 0 }`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TapOp {
    /// Committed UAIR column index (`< layout.num_cols`).
    pub col: usize,
    /// The word-group width `g` (log₂ bits per word along the entry
    /// axis; 5 for 32-bit words). Load-bearing whenever `bit_amt > 0`
    /// (rotation field) or `off > 0` (word stride `2^g`).
    pub grp_log2: usize,
    /// Within-group amount `r ∈ [0, 2^g)`: output position `j` reads
    /// input `(j − r) mod 2^g` (rotation) or `j − r` with dropout (shift).
    pub bit_amt: usize,
    /// `false` = `ROT` (cyclic); `true` = `SHIFT` (input positions
    /// `≥ 2^g − r` drop out, output positions `< r` are zero).
    pub bit_dropout: bool,
    /// Word offset along the entry axis: output word `k` reads input
    /// word `k − off` (zero for `k < off`). Must be `< 2^{s − g}` in v1.
    pub off: usize,
}

impl TapOp {
    /// The identity tap on `col` (a plain virtual-column term).
    pub fn ident(col: usize) -> Self {
        Self {
            col,
            grp_log2: 0,
            bit_amt: 0,
            bit_dropout: false,
            off: 0,
        }
    }

    /// Whether this tap is the identity op (plain column slice).
    pub fn is_ident(&self) -> bool {
        self.bit_amt == 0 && !self.bit_dropout && self.off == 0
    }

    /// The op part alone (drop the column).
    pub fn uni(&self) -> TapUniOp {
        TapUniOp {
            grp_log2: self.grp_log2,
            bit_amt: self.bit_amt,
            bit_dropout: self.bit_dropout,
            off: self.off,
        }
    }
}

/// A column-free tap op — the uniform operand a shared-point collapse
/// claim applies OUTSIDE its XOR set (`op(⊕_i a_i)`); field semantics as
/// [`TapOp`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TapUniOp {
    /// The word-group width `g` (log₂ bits per entry-axis word).
    pub grp_log2: usize,
    /// Within-group amount `r` (rotation, or shift when `bit_dropout`).
    pub bit_amt: usize,
    /// `false` = `ROT` (cyclic); `true` = `SHIFT` (dropout).
    pub bit_dropout: bool,
    /// Word offset along the entry axis (`< 2^{s − g}`).
    pub off: usize,
}

impl TapUniOp {
    /// The identity op.
    pub fn ident() -> Self {
        Self {
            grp_log2: 0,
            bit_amt: 0,
            bit_dropout: false,
            off: 0,
        }
    }

    /// Attach a column, giving a full [`TapOp`].
    pub fn with_col(self, col: usize) -> TapOp {
        TapOp {
            col,
            grp_log2: self.grp_log2,
            bit_amt: self.bit_amt,
            bit_dropout: self.bit_dropout,
            off: self.off,
        }
    }
}

/// The canonical sort key of a tap descriptor (lexicographic field
/// order).
pub(crate) fn tap_sort_key(t: &TapOp) -> (usize, usize, usize, bool, usize) {
    (t.col, t.grp_log2, t.bit_amt, t.bit_dropout, t.off)
}

/// Canonical form of a tap SOURCE combination (an XOR list of taps):
/// per-tap normal form (`bit_amt = 0` clears the dropout flag —
/// `SHIFT^0 = ROT^0`; full identities clear the group width), sorted by
/// [`tap_sort_key`], identical PAIRS cancelled (char 2). Sources equal
/// as vectors but distinct as canonical descriptor lists stay distinct
/// (sound, merely less merged).
pub fn tap_canonical_ops(taps: &[TapOp]) -> Vec<TapOp> {
    let mut v: Vec<TapOp> = taps
        .iter()
        .map(|t| {
            let mut t = *t;
            if t.bit_amt == 0 {
                t.bit_dropout = false;
            }
            if t.is_ident() {
                t.grp_log2 = 0;
            }
            t
        })
        .collect();
    v.sort_unstable_by_key(tap_sort_key);
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0usize;
    while i < v.len() {
        if i.wrapping_add(1) < v.len() && v[i] == v[i.wrapping_add(1)] {
            i = i.wrapping_add(2);
        } else {
            out.push(v[i]);
            i = i.wrapping_add(1);
        }
    }
    out
}

/// Assert the v1 support envelope for tap claims on this layout. With
/// `x_fold_extra = δ > 0` the LOW δ clear variables join the folded
/// side (the flat x-index order is unchanged, so the translated-eq
/// machinery — which is built on the flat geometry — is δ-independent;
/// only the extraction's row split and the weight-vector shapes move).
pub(crate) fn assert_tap_layout(layout: &ShaF2Layout) {
    assert_eq!(
        layout.p.word_bits, 1,
        "tap claims assume the W=1 SHA layout"
    );
    assert!(
        layout.x_fold_extra < layout.p.col_vars,
        "x_fold_extra must leave a clear variable"
    );
    assert!(
        layout.x_fold_extra == 0 || layout.bit_vars.wrapping_add(layout.tw) >= 6,
        "x_fold_extra needs word-aligned base rows (t' ≥ 6)"
    );
    assert!(
        layout.tw + layout.log_cols >= 7,
        "tap claims need tw + log_cols ≥ 7 (in-pack = row_hi/col bits only); got {} + {}",
        layout.tw,
        layout.log_cols
    );
}

/// Validate a column-free op against the layout.
pub(crate) fn assert_tap_op(layout: &ShaF2Layout, op: &TapUniOp) {
    let s = layout.p.col_vars;
    assert!(
        op.grp_log2 <= s,
        "tap group width g = {} must fit inside the clear axis (s = {s})",
        op.grp_log2
    );
    if op.bit_amt > 0 || op.bit_dropout {
        assert!(
            op.bit_amt < (1usize << op.grp_log2),
            "tap amount {} out of range (< 2^g = {})",
            op.bit_amt,
            1usize << op.grp_log2
        );
    }
    assert!(
        op.off < (1usize << (s - op.grp_log2)),
        "tap word offset {} out of range (< 2^(s−g) = {}) — larger offsets not built",
        op.off,
        1usize << (s - op.grp_log2)
    );
}

/// Validate one tap against the layout.
pub(crate) fn assert_tap(layout: &ShaF2Layout, tap: &TapOp) {
    assert!(
        tap.col < layout.num_cols,
        "tap column {} out of range",
        tap.col
    );
    assert_tap_op(layout, &tap.uni());
}

// ---------------------------------------------------------------------
// Extraction: tapped virtual rows from the committed rows
// ---------------------------------------------------------------------

/// XOR the source run `src[src_off .. src_off + len)` (bit offsets into
/// `src`) into `dst[dst_off .. dst_off + len)`, shifted `sh` bits toward
/// higher positions: `dst bit (dst_off + t) ^= src bit (src_off + t − sh)`
/// for `t ∈ [sh, len)`. Offsets and `len` are multiples of 64 when
/// `len ≥ 64` (word path); below that, offsets are multiples of `len` and
/// runs sit inside one word on both sides (the extraction's run geometry).
#[allow(clippy::arithmetic_side_effects)]
fn xor_run_shifted(
    dst: &mut [u64],
    dst_off: usize,
    src: &[u64],
    src_off: usize,
    len: usize,
    sh: usize,
) {
    if sh >= len {
        return;
    }
    if len >= 64 {
        debug_assert!(
            dst_off.is_multiple_of(64) && src_off.is_multiple_of(64) && len.is_multiple_of(64)
        );
        let nw = len >> 6;
        let dw = dst_off >> 6;
        let sw = src_off >> 6;
        let wsh = sh >> 6;
        let bsh = sh & 63;
        if bsh == 0 {
            for d in wsh..nw {
                dst[dw + d] ^= src[sw + d - wsh];
            }
        } else {
            for d in wsh..nw {
                let hi = src[sw + d - wsh];
                let lo = if d > wsh { src[sw + d - wsh - 1] } else { 0 };
                dst[dw + d] ^= (hi << bsh) | (lo >> (64 - bsh));
            }
        }
    } else {
        let mask = (1u64 << len) - 1;
        let bits = (src[src_off >> 6] >> (src_off & 63)) & mask;
        dst[dst_off >> 6] ^= ((bits << sh) & mask) << (dst_off & 63);
    }
}

/// Extract the per-clear-row bit rows of the tapped virtual vector
/// `x = ⊕_taps op(col)` in the x layout (`2^{s−δ}` rows of `2^{t'+δ}`
/// bits, `t' = bit_vars + tw`, `δ = x_fold_extra`). The group field
/// lives in the clear axis, so a tap is a whole-run gather: output
/// clear row `rl` reads source clear row `σ⁻¹(rl)` (group translation
/// on the low `g` bits, word translation on the rest) with the
/// `row_hi` runs shifted by the word borrow; under `δ > 0` the natural
/// rows are then regrouped exactly as in
/// [`crate::pcs::extract_virtual_xor_rows`] (the low-δ clear variables
/// land at the top of the new row index). The committed code is
/// `F₂`-linear, so no new commitment.
#[allow(clippy::arithmetic_side_effects)]
pub fn extract_virtual_tap_rows(
    layout: &ShaF2Layout,
    rows: &[Vec<u64>],
    taps: &[TapOp],
) -> Vec<Vec<u64>> {
    assert_tap_layout(layout);
    for tap in taps {
        assert_tap(layout, tap);
    }
    assert!(!taps.is_empty(), "tap claim needs at least one term");
    let tw = layout.tw;
    let lc = layout.log_cols;
    let bv = layout.bit_vars;
    let s = layout.p.col_vars;
    let delta = layout.x_fold_extra;
    let n_lo = 1usize << s;
    let run = 1usize << tw;
    let base_words = (1usize << (bv + tw)).div_ceil(64);
    let base: Vec<Vec<u64>> = cfg_into_iter!(0..n_lo)
        .map(|rl| {
            let mut out = vec![0u64; base_words];
            for tap in taps {
                let g = tap.grp_log2;
                let gmask = (1usize << g) - 1;
                let r5 = rl & gmask;
                let wlo = rl >> g;
                // Group-field source position (whole output row zero when
                // a SHIFT drops it).
                let src_r5 = if tap.bit_dropout {
                    if r5 < tap.bit_amt {
                        continue;
                    }
                    r5 - tap.bit_amt
                } else if g == 0 {
                    0
                } else {
                    (r5 + (1usize << g) - tap.bit_amt) & gmask
                };
                // Word-field source + the row_hi borrow.
                let (src_wlo, borrow) = if wlo >= tap.off {
                    (wlo - tap.off, 0usize)
                } else {
                    (wlo + (1usize << (s - g)) - tap.off, 1usize)
                };
                if borrow >= run {
                    continue;
                }
                let src_rl = (src_wlo << g) | src_r5;
                let row = &rows[src_rl];
                for jj in 0..1usize << bv {
                    let src_off = (jj << (lc + tw)) | (tap.col << tw);
                    let dst_off = jj << tw;
                    xor_run_shifted(&mut out, dst_off, row, src_off, run, borrow);
                }
            }
            out
        })
        .collect();
    if delta == 0 {
        return base;
    }
    // Re-split for x_fold_extra: new row = the 2^δ consecutive natural
    // rows concatenated (the moved low-δ clear variables land at the
    // top of the new row index).
    let mut regrouped = Vec::with_capacity(n_lo >> delta);
    let mut it = base.into_iter();
    for _ in 0..n_lo >> delta {
        let mut row = it.next().expect("base rows cover the regroup");
        row.reserve_exact(((1usize << delta) - 1) * base_words);
        for _ in 1..1usize << delta {
            row.extend_from_slice(&it.next().expect("base rows cover the regroup"));
        }
        regrouped.push(row);
    }
    regrouped
}

// ---------------------------------------------------------------------
// Classes and the per-class weight tables
// ---------------------------------------------------------------------

/// One bond class of a tap opening's committed-side weight split
/// `K̃(v, y) = Σ_β A_β(v)·B_β(y)`: `gamma` = the word-chain carry at the
/// clear/fold (`s`-bit) boundary of the trace index, `cut` = the carry at
/// the pack cut (committed coordinate 7; only live when `tw > 7`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct TapClass {
    pub gamma: u8,
    pub cut: u8,
}

/// The reachable classes of a tap on this layout, in canonical order.
/// The group chain never crosses a class boundary (it is confined to the
/// low-`g` clear coordinates), so only the word offset contributes:
/// `(γ, c₇) ∈ {(0,0)} ∪ {(1,0), (1,1) if off > 0}`; `(1,1)` needs the
/// cut inside the row_hi chain (`tw > 7`). Both sides derive the list.
pub(crate) fn tap_classes(layout: &ShaF2Layout, tap: &TapOp) -> Vec<TapClass> {
    let mut v = vec![TapClass { gamma: 0, cut: 0 }];
    if tap.off > 0 {
        v.push(TapClass { gamma: 1, cut: 0 });
        if layout.tw > 7 {
            v.push(TapClass { gamma: 1, cut: 1 });
        }
    }
    v
}

/// The class's in-pack table `A_β(v)`, `v ∈ [0, 128)` over the committed
/// in-pack coordinates (`row_hi` bits `0..min(tw,7)`, then — when
/// `tw < 7` — the low `7 − tw` column-index bits, pinned to `tap.col`):
/// the word-chain product with carry-in `γ` at `row_hi` bit 0, ending in
/// carry `cut` at the pack cut (`tw > 7`) or in the trace validity
/// (carry 0 out of `row_hi`'s top, `tw ≤ 7`). `pt_x` is the x-layout
/// exit point (`t' + s` coordinates; `row_hi` coords are `pt_x[..tw]`).
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn tap_inpack_table(
    layout: &ShaF2Layout,
    tap: &TapOp,
    pt_x: &[Gf],
    class: TapClass,
) -> Vec<Gf> {
    let one = Gf::one();
    let tw = layout.tw;
    let hi_bits = tw.min(7);
    let mut out = vec![Gf::zero(); 128];
    for (v, slot) in out.iter_mut().enumerate() {
        if tw < 7 {
            let npin = 7 - tw; // = min(7 − tw, log_cols): the layout assert gives lc ≥ 7 − tw
            let vcol = (v >> tw) & ((1usize << npin) - 1);
            if vcol != tap.col & ((1usize << npin) - 1) {
                continue;
            }
        }
        let mut c = class.gamma;
        let mut acc = one;
        for (i, &h) in pt_x.iter().enumerate().take(hi_bits) {
            let y = ((v >> i) & 1) as u8;
            let z = y ^ c;
            acc *= if z == 1 { h } else { one + h };
            c &= y; // maj(y, 0, c)
        }
        let ok = if tw <= 7 { c == 0 } else { c == class.cut };
        if ok {
            *slot = acc;
        }
    }
    out
}

/// The class's out-pack support factors: translated slices of the plain
/// eq tables over the source coordinate segments above the pack cut. The
/// support index is `yx = (rl ≪ (bv + hs)) | (jm ≪ hs) | hi` with
/// `hs = max(tw − 7, 0)`, `jm` the (untapped) word-bit coordinates and
/// `rl` the clear row — exactly the `x-index ≫ p0` order of the deployed
/// sparse walks — and the flat weight is
/// `B_β(yx) = t_lo[rl]·t_mid[jm]·t_hi[hi]`, zero outside `lo_band`.
pub(crate) struct TapSupportTables {
    /// Row_hi continuation factor (`2^{hs}` entries; `[1]` when `tw ≤ 7`).
    pub t_hi: Vec<Gf>,
    /// Untapped word-bit-axis factor (`2^{bit_vars}` entries, plain eq).
    pub t_mid: Vec<Gf>,
    /// Clear-axis factor (`2^s` entries): group translation on the low
    /// `g` bits ⊗ the γ-branch word translation on the rest.
    pub t_lo: Vec<Gf>,
    /// Nonzero clear-row band `[start, end)` (the γ-branch support,
    /// scaled by the group width).
    pub lo_band: (usize, usize),
    /// `hs = max(tw − 7, 0)` — the row_hi coordinate count above the cut.
    pub hs: usize,
}

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn tap_support_tables(
    layout: &ShaF2Layout,
    tap: &TapOp,
    pt_x: &[Gf],
    class: TapClass,
) -> TapSupportTables {
    let one = Gf::one();
    let tw = layout.tw;
    let bv = layout.bit_vars;
    let s = layout.p.col_vars;
    let t_x = tw + bv;
    let g = tap.grp_log2;
    let n_g = 1usize << g;
    let n_w = 1usize << (s - g);
    let hs = tw.saturating_sub(7);
    let t_hi: Vec<Gf> = if hs == 0 {
        vec![one]
    } else {
        let eq_hi = build_eq_x_r_vec(&pt_x[7..tw], &()).expect("tw > 7");
        let c = class.cut as usize;
        (0..1usize << hs)
            .map(|h| {
                if h + c < (1usize << hs) {
                    eq_hi[h + c]
                } else {
                    Gf::zero()
                }
            })
            .collect()
    };
    let t_mid: Vec<Gf> = if bv == 0 {
        vec![one]
    } else {
        build_eq_x_r_vec(&pt_x[tw..t_x], &()).expect("bv >= 1")
    };
    // Group factor over the low g clear coordinates.
    let t_grp: Vec<Gf> = if g == 0 {
        vec![one]
    } else {
        let eq_g = build_eq_x_r_vec(&pt_x[t_x..t_x + g], &()).expect("g >= 1");
        (0..n_g)
            .map(|j| {
                if tap.bit_dropout {
                    if j + tap.bit_amt < n_g {
                        eq_g[j + tap.bit_amt]
                    } else {
                        Gf::zero()
                    }
                } else {
                    eq_g[(j + tap.bit_amt) & (n_g - 1)]
                }
            })
            .collect()
    };
    // Word factor over the remaining clear coordinates (γ branch).
    let t_wrd: Vec<Gf> = if s == g {
        vec![one]
    } else {
        let eq_w = build_eq_x_r_vec(&pt_x[t_x + g..], &()).expect("s > g");
        (0..n_w)
            .map(|x| {
                if class.gamma == 0 {
                    if x < n_w - tap.off {
                        eq_w[x + tap.off]
                    } else {
                        Gf::zero()
                    }
                } else if x >= n_w - tap.off {
                    eq_w[x + tap.off - n_w]
                } else {
                    Gf::zero()
                }
            })
            .collect()
    };
    let (w_start, w_end) = if class.gamma == 0 {
        (0, n_w - tap.off)
    } else {
        (n_w - tap.off, n_w)
    };
    let t_lo: Vec<Gf> = (0..1usize << s)
        .map(|rl| t_grp[rl & (n_g - 1)] * t_wrd[rl >> g])
        .collect();
    TapSupportTables {
        t_hi,
        t_mid,
        t_lo,
        lo_band: (w_start << g, w_end << g),
        hs,
    }
}

// ---------------------------------------------------------------------
// The succinct closure: matrix-product-state residual evaluation
// ---------------------------------------------------------------------

/// Per-committed-coordinate closure descriptor (coordinates `7..n`, the
/// packed `m_p` coordinates in positional order).
#[derive(Clone, Debug)]
pub(crate) enum TapCoord {
    /// Plain eq factor against a fixed right-leg point (boolean column
    /// pins; also any untwisted coordinate).
    Plain(Gf),
    /// One bit of a carry chain: right-leg point `h`, addend bit `d`;
    /// `entry` pins the carry-in when this bit starts a segment, `exit`
    /// contracts the carry-out when it ends one.
    Chain {
        h: Gf,
        d: u8,
        entry: Option<u8>,
        exit: Option<TapExit>,
    },
}

/// Segment-end contraction: pin the carry (dropout validity / class
/// match) or sum both branches (cyclic wrap).
#[derive(Clone, Copy, Debug)]
pub(crate) enum TapExit {
    Pin(u8),
    Sum,
}

/// The class's closure descriptor over the `m_p` packed coordinates:
/// the word chain on `row_hi` (carry from the class), the column pins,
/// the untapped word-bit coordinates, the group chain on the low-`g`
/// clear coordinates, and the word chain's clear part ending in the
/// class's `γ`. Untwisted segments emit [`TapCoord::Plain`].
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn tap_closure_desc(
    layout: &ShaF2Layout,
    tap: &TapOp,
    pt_x: &[Gf],
    class: TapClass,
) -> Vec<TapCoord> {
    let one = Gf::one();
    let zero = Gf::zero();
    let tw = layout.tw;
    let lc = layout.log_cols;
    let bv = layout.bit_vars;
    let s = layout.p.col_vars;
    let t_x = tw + bv;
    let g = tap.grp_log2;
    let mut out = Vec::with_capacity(layout.p.row_vars + s - 7);
    // Row_hi continuation: the word chain when off > 0, plain otherwise.
    for (k, &h) in pt_x.iter().enumerate().take(tw).skip(7) {
        if tap.off > 0 {
            out.push(TapCoord::Chain {
                h,
                d: 0,
                entry: if k == 7 { Some(class.cut) } else { None },
                exit: if k == tw - 1 {
                    Some(TapExit::Pin(0))
                } else {
                    None
                },
            });
        } else {
            out.push(TapCoord::Plain(h));
        }
    }
    for k in tw.max(7)..tw + lc {
        out.push(TapCoord::Plain(if (tap.col >> (k - tw)) & 1 == 1 {
            one
        } else {
            zero
        }));
    }
    for m in 0..bv {
        out.push(TapCoord::Plain(pt_x[tw + m]));
    }
    // Group chain on the low-g clear coordinates.
    for m in 0..g {
        let h = pt_x[t_x + m];
        if tap.bit_amt > 0 || tap.bit_dropout {
            out.push(TapCoord::Chain {
                h,
                d: ((tap.bit_amt >> m) & 1) as u8,
                entry: if m == 0 { Some(0) } else { None },
                exit: if m == g - 1 {
                    Some(if tap.bit_dropout {
                        TapExit::Pin(0)
                    } else {
                        TapExit::Sum
                    })
                } else {
                    None
                },
            });
        } else {
            out.push(TapCoord::Plain(h));
        }
    }
    // Word chain on the remaining clear coordinates.
    for m in g..s {
        let h = pt_x[t_x + m];
        if tap.off > 0 {
            out.push(TapCoord::Chain {
                h,
                d: ((tap.off >> (m - g)) & 1) as u8,
                entry: if m == g { Some(0) } else { None },
                exit: if m == s - 1 {
                    Some(TapExit::Pin(class.gamma))
                } else {
                    None
                },
            });
        } else {
            out.push(TapCoord::Plain(h));
        }
    }
    debug_assert_eq!(out.len(), layout.p.row_vars + s - 7);
    out
}

/// One evaluator state: the K⊗K column representation, per live carry
/// value (length 1 = outside any chain segment, 2 = inside one).
struct TapState {
    states: Vec<Vec<Gf>>,
}

#[allow(clippy::arithmetic_side_effects)]
fn tap_state_apply(st: &mut TapState, coord: &TapCoord, left: (Gf, Gf), boolean: Option<u8>) {
    use crate::ligerito::apply_right_mul;
    let one = Gf::one();
    match coord {
        TapCoord::Plain(h) => {
            for cols in st.states.iter_mut() {
                let new = match boolean {
                    Some(b) => {
                        let f = if b == 1 { *h } else { one + *h };
                        apply_right_mul(cols, f)
                    }
                    None => {
                        let f0 = apply_right_mul(cols, one + *h);
                        let f1 = apply_right_mul(cols, *h);
                        let (la, a) = left;
                        (0..128).map(|u| la * f0[u] + a * f1[u]).collect()
                    }
                };
                *cols = new;
            }
        }
        TapCoord::Chain { h, d, entry, exit } => {
            if let Some(c0) = entry {
                debug_assert_eq!(st.states.len(), 1, "chain entry from plain mode");
                let cur = st.states.pop().expect("state");
                let zeros = vec![Gf::zero(); 128];
                st.states = if *c0 == 0 {
                    vec![cur, zeros]
                } else {
                    vec![zeros, cur]
                };
            }
            debug_assert_eq!(st.states.len(), 2, "chain bit inside a segment");
            let mut new = vec![vec![Gf::zero(); 128], vec![Gf::zero(); 128]];
            for c in 0..2u8 {
                let cols = &st.states[c as usize];
                if cols.iter().all(|g| g.is_zero()) {
                    continue;
                }
                for y in 0..2u8 {
                    let lscale = match boolean {
                        Some(b) => {
                            if b != y {
                                continue;
                            }
                            one
                        }
                        None => {
                            if y == 0 {
                                left.0
                            } else {
                                left.1
                            }
                        }
                    };
                    let z = y ^ d ^ c;
                    let c2 = (y & *d) | (y & c) | (*d & c);
                    let f = if z == 1 { *h } else { one + *h };
                    let add = apply_right_mul(cols, f);
                    let dstc = &mut new[c2 as usize];
                    for (o, x) in dstc.iter_mut().zip(add.iter()) {
                        *o += lscale * *x;
                    }
                }
            }
            match exit {
                Some(TapExit::Pin(c)) => {
                    let kept = new.swap_remove(*c as usize);
                    st.states = vec![kept];
                }
                Some(TapExit::Sum) => {
                    let (a, b) = (new.remove(0), new.remove(0));
                    st.states = vec![a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()];
                }
                None => st.states = new,
            }
        }
    }
}

/// Matrix-product-state generalization of
/// [`crate::ligerito::residual_b_evals`]: `Φ_{r″}∘B_β` evaluated at
/// `(prefix ++ bits(tail))` for every boolean tail, where `B_β` is the
/// class's translated-eq basis described by `desc` (one entry per packed
/// coordinate). For an all-[`TapCoord::Plain`] descriptor this is exactly
/// `residual_b_evals` (2 right-multiplications per coordinate); chain
/// bits cost ≤ 4.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn residual_b_evals_tap(
    prefix: &[Gf],
    yr_log_n: usize,
    desc: &[TapCoord],
    eq_r2: &[Gf],
) -> Vec<Gf> {
    assert_eq!(
        prefix.len() + yr_log_n,
        desc.len(),
        "prefix + tail must cover the coordinates"
    );
    assert_eq!(eq_r2.len(), 128);
    let one = Gf::one();
    let mut init = vec![Gf::zero(); 128];
    init[0] = one;
    let mut st = TapState { states: vec![init] };
    for (a, coord) in prefix.iter().zip(desc.iter()) {
        tap_state_apply(&mut st, coord, (one + *a, *a), None);
    }
    let mut cur: Vec<TapState> = vec![st];
    for j in 0..yr_log_n {
        let coord = &desc[prefix.len() + j];
        let mut next: Vec<TapState> = Vec::with_capacity(cur.len() * 2);
        for b in 0..2u8 {
            for stt in cur.iter() {
                let mut branch = TapState {
                    states: stt.states.to_vec(),
                };
                tap_state_apply(&mut branch, coord, (one, one), Some(b));
                next.push(branch);
            }
        }
        // Tail index bit j is appended at weight 2^j: order [b=0 block, b=1 block].
        cur = next;
    }
    cur.iter()
        .map(|stt| {
            debug_assert_eq!(stt.states.len(), 1, "all chains closed at the end");
            stt.states[0]
                .iter()
                .zip(eq_r2.iter())
                .fold(Gf::zero(), |acc, (c, e)| acc + *c * *e)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ligerito::mle_eval;
    use crate::pcs::IntegerMatrixLayout;

    /// W=1 tap test layout, `tw = 6` (pack cut at the column bit): 2
    /// UAIR bit-columns over 2^14 trace rows; committed t = 7, s = 8
    /// (n = 15); x tensor t' = 6, s = 8. Group width g = 3 (8-bit words).
    fn tap_layout_tw6() -> ShaF2Layout {
        ShaF2Layout {
            p: IntegerMatrixLayout {
                row_vars: 7,
                col_vars: 8,
                word_bits: 1,
            },
            num_cols: 2,
            log_cols: 1,
            bit_vars: 0,
            num_vars: 14,
            tw: 6,
            x_fold_extra: 0,
        }
    }

    /// A `tw = 9 > 7` layout (three-class offsets): 2 bit-columns over
    /// 2^16 trace rows, s = 7; g = 5 (32-bit words), off < 2^{s−g} = 4.
    fn tap_layout_tw9() -> ShaF2Layout {
        ShaF2Layout {
            p: IntegerMatrixLayout {
                row_vars: 10,
                col_vars: 7,
                word_bits: 1,
            },
            num_cols: 2,
            log_cols: 1,
            bit_vars: 0,
            num_vars: 16,
            tw: 9,
            x_fold_extra: 0,
        }
    }

    /// A layout with an (untapped) word-bit axis coexisting with the
    /// entry-axis grouping: bit_vars = 3, tw = 6, s = 7, g = 3.
    fn tap_layout_bv3() -> ShaF2Layout {
        ShaF2Layout {
            p: IntegerMatrixLayout {
                row_vars: 10,
                col_vars: 7,
                word_bits: 1,
            },
            num_cols: 2,
            log_cols: 1,
            bit_vars: 3,
            num_vars: 13,
            tw: 6,
            x_fold_extra: 0,
        }
    }

    /// The layout's tap group width for the tests (8-bit words on the
    /// small layouts, 32-bit on tw9).
    fn grp_of(layout: &ShaF2Layout) -> usize {
        if layout.tw == 9 { 5 } else { 3 }
    }

    fn test_rows(layout: &ShaF2Layout, seed: u64) -> Vec<Vec<u64>> {
        let words = (1usize << layout.p.row_vars).div_ceil(64).max(1);
        (0..1usize << layout.p.col_vars)
            .map(|c| {
                (0..words)
                    .map(|w| {
                        (c as u64 ^ seed)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add((w as u64).wrapping_mul(0xD134_2543_DE82_EF95))
                            .rotate_left((w % 61) as u32)
                    })
                    .collect()
            })
            .collect()
    }

    fn committed_bit(rows: &[Vec<u64>], b: usize, c: usize) -> u64 {
        (rows[c][b >> 6] >> (b & 63)) & 1
    }

    /// Naive per-bit tap extraction, straight from the corrected
    /// semantics: output entry `p = (k ≪ g)|j` of stream `op(col)` reads
    /// committed entry `((k − off) ≪ g)|((j − r) mod 2^g)` (dropouts →
    /// zero), at every word-bit position of the untapped bv axis.
    fn extract_naive(layout: &ShaF2Layout, rows: &[Vec<u64>], taps: &[TapOp]) -> Vec<Vec<u64>> {
        let tw = layout.tw;
        let lc = layout.log_cols;
        let bv = layout.bit_vars;
        let s = layout.p.col_vars;
        let nv = layout.num_vars;
        let words = (1usize << (bv + tw)).div_ceil(64).max(1);
        let mut out = vec![vec![0u64; words]; 1usize << s];
        for trace in 0..1usize << nv {
            let (row_hi, row_lo) = (trace >> s, trace & ((1 << s) - 1));
            for jj in 0..1usize << bv {
                let mut bit = 0u64;
                for tap in taps {
                    let g = tap.grp_log2;
                    let n_g = 1usize << g;
                    let (k, j) = (trace >> g, trace & (n_g - 1));
                    if k < tap.off {
                        continue;
                    }
                    let src_j = if tap.bit_dropout {
                        if j < tap.bit_amt {
                            continue;
                        }
                        j - tap.bit_amt
                    } else {
                        (j + n_g - tap.bit_amt) & (n_g - 1)
                    };
                    let src_trace = ((k - tap.off) << g) | src_j;
                    let b = (jj << (lc + tw)) | (tap.col << tw) | (src_trace >> s);
                    bit ^= committed_bit(rows, b, src_trace & ((1 << s) - 1));
                }
                if bit & 1 == 1 {
                    let pos = (jj << tw) | row_hi;
                    out[row_lo][pos >> 6] |= 1u64 << (pos & 63);
                }
            }
        }
        out
    }

    #[test]
    fn tap_extraction_matches_naive() {
        for layout in [tap_layout_tw6(), tap_layout_tw9(), tap_layout_bv3()] {
            let g = grp_of(&layout);
            let rows = test_rows(&layout, 7);
            let rot = |col, amt, off| TapOp {
                col,
                grp_log2: g,
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let shl = |col, amt, off| TapOp {
                col,
                grp_log2: g,
                bit_amt: amt,
                bit_dropout: true,
                off,
            };
            let cases: Vec<Vec<TapOp>> = vec![
                vec![TapOp::ident(0)],
                vec![rot(0, 1, 0)],
                vec![shl(1, 5, 0)],
                vec![rot(0, 0, 1)],
                vec![rot(1, 3, 2)],
                // The b_3-style claim: three taps on one column.
                vec![rot(0, 1, 0), rot(0, 2, 1), rot(0, 3, 2)],
                // Cross-column with shift and rot.
                vec![shl(0, 3, 0), shl(1, 5, 1), rot(1, 2, 2)],
            ];
            for taps in &cases {
                let fast = extract_virtual_tap_rows(&layout, &rows, taps);
                let naive = extract_naive(&layout, &rows, taps);
                assert_eq!(fast, naive, "taps {taps:?} on layout tw={}", layout.tw);
            }
        }
    }

    /// The δ re-split of the extraction equals the manual regroup of
    /// the natural-split rows (2^δ consecutive rows concatenated).
    #[test]
    fn tap_extraction_delta_resplits() {
        let base = tap_layout_tw6();
        let rows = test_rows(&base, 11);
        let g = grp_of(&base);
        let taps = vec![
            TapOp {
                col: 0,
                grp_log2: g,
                bit_amt: 2,
                bit_dropout: false,
                off: 1,
            },
            TapOp {
                col: 1,
                grp_log2: g,
                bit_amt: 3,
                bit_dropout: true,
                off: 0,
            },
        ];
        let flat = extract_virtual_tap_rows(&base, &rows, &taps);
        for delta in [1usize, 2] {
            let mut layout = tap_layout_tw6();
            layout.x_fold_extra = delta;
            let got = extract_virtual_tap_rows(&layout, &rows, &taps);
            let m = 1usize << delta;
            assert_eq!(got.len(), flat.len() / m);
            for (c, row) in got.iter().enumerate() {
                let want: Vec<u64> = (0..m)
                    .flat_map(|lc| flat[(c << delta) | lc].iter().copied())
                    .collect();
                assert_eq!(*row, want, "delta {delta} row {c}");
            }
        }
    }

    /// 64-bit words (g = 6) with wide amounts (≥ 32): the fast
    /// extraction matches the per-bit naive on the tw6 layout
    /// (s = 8 ≥ g + 2 — 64-bit words, offsets < 4).
    #[test]
    fn tap_extraction_matches_naive_g6() {
        let layout = tap_layout_tw6();
        let rows = test_rows(&layout, 21);
        let rot = |col, amt, off| TapOp {
            col,
            grp_log2: 6,
            bit_amt: amt,
            bit_dropout: false,
            off,
        };
        let shl = |col, amt, off| TapOp {
            col,
            grp_log2: 6,
            bit_amt: amt,
            bit_dropout: true,
            off,
        };
        let cases: Vec<Vec<TapOp>> = vec![
            vec![rot(0, 33, 0)],
            vec![shl(1, 40, 1)],
            vec![rot(1, 1, 3)],
            vec![rot(0, 47, 1), rot(1, 9, 2), shl(0, 63, 0)],
        ];
        for taps in &cases {
            assert_eq!(
                extract_virtual_tap_rows(&layout, &rows, taps),
                extract_naive(&layout, &rows, taps),
                "taps {taps:?}"
            );
        }
    }

    #[test]
    fn tap_canonical_ops_normalizes() {
        let rot = |col, amt, off| TapOp {
            col,
            grp_log2: 3,
            bit_amt: amt,
            bit_dropout: false,
            off,
        };
        // SHIFT^0 normalizes to ROT^0; a full identity clears the group
        // width; an off-only tap keeps it (the word stride).
        let shl0 = TapOp {
            col: 1,
            grp_log2: 3,
            bit_amt: 0,
            bit_dropout: true,
            off: 0,
        };
        assert_eq!(tap_canonical_ops(&[shl0]), vec![TapOp::ident(1)]);
        let id5 = TapOp {
            col: 0,
            grp_log2: 5,
            bit_amt: 0,
            bit_dropout: false,
            off: 0,
        };
        assert_eq!(tap_canonical_ops(&[id5]), vec![TapOp::ident(0)]);
        let off_tap = rot(0, 0, 2);
        assert_eq!(tap_canonical_ops(&[off_tap]), vec![off_tap]);
        // Sorting + identical-pair cancellation (char 2): a ⊕ b ⊕ a = b.
        let a = rot(1, 2, 0);
        let b = rot(0, 1, 1);
        assert_eq!(tap_canonical_ops(&[b, a]), vec![b, a]);
        assert_eq!(tap_canonical_ops(&[a, b, a]), vec![b]);
        assert_eq!(tap_canonical_ops(&[a, a]), Vec::<TapOp>::new());
    }

    fn test_point(len: usize, seed: u64) -> Vec<Gf> {
        (0..len)
            .map(|i| {
                let x = (i as u64 ^ seed)
                    .wrapping_mul(0xA24B_AED4_963E_E407)
                    .wrapping_add(0x9FB2_1C65_1E98_DF25);
                Gf::from_polynomial_words([x, x.rotate_left(17) ^ seed])
            })
            .collect()
    }

    /// The load-bearing split identity: for every source x-index `x'`,
    /// `Σ_β A_β(v)·B_β(yx) = eq(ζ, σ(x'))·[valid]` with
    /// `(v, y) = embed(x', col)` split at the pack, `yx = x' ≫ p0`.
    #[test]
    fn tap_weight_split_recombines() {
        for layout in [tap_layout_tw6(), tap_layout_tw9(), tap_layout_bv3()] {
            let g = grp_of(&layout);
            let tw = layout.tw;
            let bv = layout.bit_vars;
            let s = layout.p.col_vars;
            let t_x = tw + bv;
            let n_x = t_x + s;
            let n_g = 1usize << g;
            let pt = test_point(n_x, 0xBEEF ^ (tw as u64));
            let eqx = build_eq_x_r_vec(&pt, &()).unwrap();
            let p0 = 7usize.saturating_sub((7usize.saturating_sub(tw)).min(layout.log_cols));
            let rot = |col, amt, off| TapOp {
                col,
                grp_log2: g,
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let shl = |col, amt, off| TapOp {
                col,
                grp_log2: g,
                bit_amt: amt,
                bit_dropout: true,
                off,
            };
            for tap in [
                TapOp::ident(1),
                rot(0, 3, 0),
                shl(1, 5, 1),
                rot(0, 6, 2),
                rot(0, 0, 3),
            ] {
                let classes = tap_classes(&layout, &tap);
                let plans: Vec<(Vec<Gf>, TapSupportTables)> = classes
                    .iter()
                    .map(|&cl| {
                        (
                            tap_inpack_table(&layout, &tap, &pt, cl),
                            tap_support_tables(&layout, &tap, &pt, cl),
                        )
                    })
                    .collect();
                for xp in 0..1usize << n_x {
                    let row_hi = xp & ((1 << tw) - 1);
                    let jm = (xp >> tw) & ((1 << bv) - 1);
                    let rl = xp >> t_x;
                    // Forward image σ(x') and validity on the trace index.
                    let src_trace = (row_hi << s) | rl;
                    let (k, j) = (src_trace >> g, src_trace & (n_g - 1));
                    let k_out = k + tap.off;
                    let trace_ok = k_out < (1usize << (layout.num_vars - g));
                    let (dst_j, bit_ok) = if tap.bit_dropout {
                        (j + tap.bit_amt, j + tap.bit_amt < n_g)
                    } else {
                        ((j + tap.bit_amt) & (n_g - 1), true)
                    };
                    let expected = if trace_ok && bit_ok {
                        let dst_trace = (k_out << g) | dst_j;
                        let dst_x =
                            (dst_trace >> s) | (jm << tw) | ((dst_trace & ((1 << s) - 1)) << t_x);
                        eqx[dst_x]
                    } else {
                        Gf::zero()
                    };
                    let z = crate::ligerito_flock::embed_xor_index(&layout, xp, tap.col);
                    let v = z & 127;
                    let yx = xp >> p0;
                    let mut got = Gf::zero();
                    for (a_tbl, sup) in &plans {
                        let hs = sup.hs;
                        let hi = yx & ((1usize << hs) - 1);
                        let jmid = (yx >> hs) & ((1 << bv) - 1);
                        let lo = yx >> (hs + bv);
                        got += a_tbl[v] * sup.t_lo[lo] * sup.t_mid[jmid] * sup.t_hi[hi];
                    }
                    assert_eq!(got, expected, "tap {tap:?} x'={xp} tw={tw}");
                }
            }
        }
    }

    /// The MPS closure equals the naive Φ∘B multilinear evaluation.
    #[test]
    fn tap_closure_matches_naive() {
        for layout in [tap_layout_tw6(), tap_layout_tw9(), tap_layout_bv3()] {
            let g = grp_of(&layout);
            let tw = layout.tw;
            let bv = layout.bit_vars;
            let s = layout.p.col_vars;
            let t_x = tw + bv;
            let n = layout.p.row_vars + s;
            let m_p = n - 7;
            let n_x = t_x + s;
            let pt = test_point(n_x, 0xC0FFEE ^ (tw as u64));
            let eq_r2 = build_eq_x_r_vec(&test_point(7, 99)[..7], &()).unwrap();
            let phi = |gg: Gf| -> Gf {
                let wds = gg.as_words();
                let mut acc = Gf::zero();
                for wi in 0..2usize {
                    let mut bits = wds[wi];
                    while bits != 0 {
                        let t = bits.trailing_zeros() as usize;
                        acc += eq_r2[(wi << 6) | t];
                        bits &= bits.wrapping_sub(1);
                    }
                }
                acc
            };
            let p0 = 7usize.saturating_sub((7usize.saturating_sub(tw)).min(layout.log_cols));
            let rot = |col, amt, off| TapOp {
                col,
                grp_log2: g,
                bit_amt: amt,
                bit_dropout: false,
                off,
            };
            let shl = |col, amt, off| TapOp {
                col,
                grp_log2: g,
                bit_amt: amt,
                bit_dropout: true,
                off,
            };
            for tap in [TapOp::ident(0), rot(1, 2, 0), shl(0, 4, 1), rot(1, 7, 2)] {
                for &cl in &tap_classes(&layout, &tap) {
                    let sup = tap_support_tables(&layout, &tap, &pt, cl);
                    // Naive: scatter Φ(B_β) into the packed y-space and
                    // evaluate its MLE at (prefix ++ tails).
                    let mut bphi = vec![Gf::zero(); 1usize << m_p];
                    for yx in 0..1usize << (n_x - p0) {
                        let hs = sup.hs;
                        let hi = yx & ((1usize << hs) - 1);
                        let jmid = (yx >> hs) & ((1 << bv) - 1);
                        let lo = yx >> (hs + bv);
                        let val = sup.t_lo[lo] * sup.t_mid[jmid] * sup.t_hi[hi];
                        if val.is_zero() {
                            continue;
                        }
                        let z = crate::ligerito_flock::embed_xor_index(&layout, yx << p0, tap.col);
                        bphi[z >> 7] = phi(val);
                    }
                    let desc = tap_closure_desc(&layout, &tap, &pt, cl);
                    for yr in [0usize, 2] {
                        let prefix = test_point(m_p - yr, 0xF00D + yr as u64);
                        let got = residual_b_evals_tap(&prefix, yr, &desc, &eq_r2);
                        for (tail, &got_t) in got.iter().enumerate() {
                            let mut point = prefix.clone();
                            for tb in 0..yr {
                                point.push(if (tail >> tb) & 1 == 1 {
                                    Gf::one()
                                } else {
                                    Gf::zero()
                                });
                            }
                            let want = mle_eval(&bphi, &point);
                            assert_eq!(got_t, want, "tap {tap:?} class {cl:?} tail {tail}");
                        }
                    }
                }
            }
        }
    }
}
