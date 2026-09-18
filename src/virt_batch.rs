//! Prover-side dual-basis batching for packed-source tensor repetitions
//! (the SHA-256 product layout: `global column = 1 + instance·w + local`,
//! one shared constant column, power-of-two instance count).
//!
//! The virtual opening's batching protocol sends the 128 dual-basis plane
//! openings `h_b = Σ_j bit_b(W_j)·P[j»7]·A(e_{j&127})` and then folds the
//! ρ-batched Ligerito basis `a′(y) = Σ_v Φ_ρ(W_{(y,v)})·A(e_v)`, where for a
//! packed-source repetition the source weights factor per chunk `l` as
//! `W_{(i,c)} = Σ_l e_{l,i}·s_{l,c}` (instance `i`, local column `c`;
//! [`crate::ligerito_flock`]'s `VirtColumnWeights::PackedSourceRepeated`).
//! The per-cell kernels pay one weight multiply, one basis multiply and a
//! 128-way bit scatter (resp. 16 table gathers) per source cell.
//!
//! This module restructures both passes around the dual-basis identity
//!
//! ```text
//! bit_b(e·s) = c₀(e·s·A(e_b)) = Σ_a bit_a(e·A(e_b)) · bit_a(ŝ),   ŝ := A⁻¹(s),
//! ```
//!
//! (`c₀(X^u·A(e_v)) = δ_{uv}`, [`crate::dual_basis`]) which separates the
//! instance factor from the local-column factor. With the LOCAL plane
//! packings
//!
//! ```text
//! R_a(y, i) := Σ_{v : cell (y,v) ∈ instance i} bit_a(ŝ_{c(y,v)}) · A(e_v)
//! ```
//!
//! — a bit-plane transpose of the instance's local columns in pack `y`,
//! shared by every instance whose first cell sits at the same pack offset
//! (its *phase*) — both messages become
//!
//! ```text
//! h_b   = Σ_i Σ_a bit_a(e_i·A(e_b)) · Q_i[a],    Q_i[a]  = Σ_y P[y]·R_a(y, i),
//! a′(y) = Σ_i Σ_a ρ′_{i,a} · R_a(y, i),         ρ′_{i,a} = Σ_b ρ_b·bit_a(e_i·A(e_b)),
//! ```
//!
//! so each source cell costs ONE unreduced fixed-scalar GF(2^128) multiply
//! per pass (4 shuffle-free PMULLs into a 2-limb accumulator, one fold per
//! accumulator; a 3-PMULL Karatsuba form with the varying operand pre-split
//! measured as a wash on the M4 — µop-bound), plus `O(128²)` field work per
//! instance: the `Q_i` read-off
//! through 16 byte tables, and `ρ′_i = Σ_{u ∈ e_i} C_u` from the 16
//! byte-indexed tables of `C_{u,a} = Σ_b ρ_b·bit_a(X^u·A(e_b))`, which
//! depend on ρ alone. The a′ pass also emits flock's Ligerito round-0
//! message (`fill_phi_basis_round0`'s contract). Only exact field
//! identities are used, so `h`, `a′` and the round-0 pair are bit-identical
//! to the per-cell kernels (pinned by `virtual_planes_match_cellwise` in
//! `ligerito_flock`).
//!
//! The verifier computes the same `a′` (its Ligerito basis) from the public
//! weights alone through [`PackedSourcePlanes::add_a_prime`], and the
//! identity compact tail of the SHA-256 + ECDSA map through
//! [`AffineTailPlanes`]: the same identity with the high row index as the
//! instance and the low row index as the local column.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    ligerito::{
        LOG_PACKING, PackedBits, phi_bit_sum, phi_byte_tables, phi_byte_tables_into,
        phi_from_words, transpose_8x8_bits,
    },
    poly::univariate::binary_gf128::Gf128 as Gf,
    utils::{cfg_chunks_mut, cfg_into_iter, cfg_iter, wide_mul::WideMulAcc},
};

/// Cells per source pack.
const PACK: usize = 1 << LOG_PACKING;

/// Packs per parallel task (even, so round-0 pairs never straddle tasks):
/// bounds the duplicated instance read-off at task boundaries to one
/// instance per ~2^11 packs while keeping hundreds of tasks at the
/// production shapes.
const TASK_PACKS: usize = 1 << 11;

/// Upper bound on the precomputed plane tables (all chunks, all phases).
const MAX_TABLE_BYTES: usize = 256 << 20;

/// Narrowest instance the engine accepts: below this the per-instance
/// read-offs outweigh the per-cell savings over the streamed kernels.
const MIN_LOCAL_WIDTH: usize = 512;

// ---------------------------------------------------------------------
// Fixed-scalar unreduced multiply-accumulate kernel
// ---------------------------------------------------------------------

mod kernel {
    use super::Gf;
    pub(super) use field::{Gf128PreparedAcc as Acc, PreparedGf128Mul as Fixed};
    #[inline(always)]
    pub(super) fn fixed(x: &Gf) -> Fixed {
        Fixed::new((*x).into())
    }
    #[inline(always)]
    pub(super) fn zero() -> Acc {
        Acc::zero()
    }
    #[inline(always)]
    pub(super) fn mul_acc(acc: &mut Acc, x: &Gf, fixed: &Fixed) {
        acc.add_mul(&(*x).into(), fixed);
    }
    #[inline(always)]
    pub(super) fn reduce(acc: &Acc) -> Gf {
        acc.reduce().into()
    }
}

// ---------------------------------------------------------------------
// Dual-basis bit kernels
// ---------------------------------------------------------------------

/// Corrections of the dual basis beyond the coordinate reversal, indexed by
/// bits `1..=6` of the slot vector: `A(e_1) = X^127 + X^6 + X`,
/// `A(e_v) = X^{128−v} + X^{7−v}` for `2 ≤ v ≤ 6` (see [`crate::dual_basis`]).
const DUAL_CORR: [u64; 64] = {
    let mut table = [0u64; 64];
    let mut mask = 1usize;
    while mask < 64 {
        let mut corr = 0u64;
        let mut v = 1usize;
        while v <= 6 {
            if (mask >> (v - 1)) & 1 == 1 {
                corr ^= if v == 1 {
                    (1 << 6) | (1 << 1)
                } else {
                    1 << (7 - v)
                };
            }
            v += 1;
        }
        table[mask] = corr;
        mask += 1;
    }
    table
};

/// `Σ_v bit_v(u)·A(e_v)`: the dual-basis packing of a slot-indexed bit
/// vector — the reversal `v ↦ 128 − v` (`v ≥ 1`), `e₀ ↦ 1`, plus the seven
/// XOR corrections.
#[inline(always)]
pub(crate) fn dual_pack(u: [u64; 2]) -> Gf {
    let r0 = u[1].reverse_bits();
    let r1 = u[0].reverse_bits();
    let lo = ((r0 << 1) | (u[0] & 1)) ^ DUAL_CORR[((u[0] >> 1) & 63) as usize];
    let hi = (r1 << 1) | (r0 >> 63);
    Gf::from_polynomial_words([lo, hi])
}

/// `A⁻¹(s)`: the slot vector with `bit_a = c₀(X^a·s)`, so that
/// `s = Σ_a bit_a(ŝ)·A(e_a)` and `c₀(g·s) = Σ_a bit_a(g)·bit_a(ŝ)`.
///
/// [`dual_pack`] is a bit reversal of the slots `1..=127` onto the
/// monomials `X^127..=X^1`, the slot `0` onto `X^0`, plus the seven
/// corrections `DUAL_CORR` on `X^1..=X^6` selected by slots `1..=6`; this
/// inverts it step by step: slot `0` is bit `0` of the low word, slots
/// `1..=63` are the reversal of the high word's bits `63..=1` (which the
/// corrections never touch), the corrections are then known, and slots
/// `64..=127` are the reversal of the corrected low word's bits `63..=1`
/// together with bit `0` of the high word. `dual_unpack_by_multiplication`
/// (the defining `c₀(X^a·s)` scan) is the test oracle.
#[inline]
pub(crate) fn dual_unpack(s: Gf) -> [u64; 2] {
    let [lo, hi] = *s.as_words();
    let u0 = (hi >> 1).reverse_bits() | (lo & 1);
    let corrections = DUAL_CORR[((u0 >> 1) & 63) as usize];
    let r0 = ((lo ^ corrections) >> 1) | ((hi & 1) << 63);
    [u0, r0.reverse_bits()]
}

/// The defining scan of [`dual_unpack`]: `bit_a(ŝ) = c₀(X^a·s)` by a
/// multiply-by-`X` chain. Kept as the test oracle.
#[cfg(test)]
pub(crate) fn dual_unpack_by_multiplication(s: Gf) -> [u64; 2] {
    let mut z = s;
    let mut out = [0u64; 2];
    for a in 0..PACK {
        out[a >> 6] |= (z.as_words()[0] & 1) << (a & 63);
        z = z.mul_x();
    }
    out
}

/// 128×128 bit transpose: `out[c]` bit `r` = `rows[r]` bit `c`.
pub(crate) fn transpose_128x128(rows: &[[u64; 2]; PACK]) -> [[u64; 2]; PACK] {
    let mut out = [[0u64; 2]; PACK];
    for row_group in 0..16usize {
        for col_byte in 0..16usize {
            let mut block = 0u64;
            for i in 0..8usize {
                let row = rows[(row_group << 3) | i];
                let byte = (row[col_byte >> 3] >> ((col_byte & 7) << 3)) & 0xFF;
                block |= byte << (i << 3);
            }
            let t = transpose_8x8_bits(block);
            for j in 0..8usize {
                let byte = (t >> (j << 3)) & 0xFF;
                out[(col_byte << 3) | j][row_group >> 3] |= byte << ((row_group & 7) << 3);
            }
        }
    }
    out
}

/// Aligned bit planes, zero-padding only the final partial source pack.
fn aligned_planes(unpacked: &[[u64; 2]]) -> Vec<[[u64; 2]; PACK]> {
    let (full, tail) = unpacked.as_chunks::<PACK>();
    cfg_into_iter!(0..unpacked.len().div_ceil(PACK))
        .map(|block| {
            if let Some(rows) = full.get(block) {
                return transpose_128x128(rows);
            }
            let mut rows = [[0u64; 2]; PACK];
            rows[..tail.len()].copy_from_slice(tail);
            transpose_128x128(&rows)
        })
        .collect()
}

/// `x << n` on a 128-bit slot vector, `0 ≤ n < 128`.
#[inline(always)]
fn shl128(x: [u64; 2], n: usize) -> [u64; 2] {
    match n {
        0 => x,
        1..=63 => [x[0] << n, (x[1] << n) | (x[0] >> (64 - n))],
        64 => [0, x[0]],
        _ => [0, x[0] << (n - 64)],
    }
}

/// `x >> n` on a 128-bit slot vector, `0 ≤ n < 128`.
#[inline(always)]
fn shr128(x: [u64; 2], n: usize) -> [u64; 2] {
    match n {
        0 => x,
        1..=63 => [(x[0] >> n) | (x[1] << (64 - n)), x[1] >> n],
        64 => [x[1], 0],
        _ => [x[1] >> (n - 64), 0],
    }
}

/// `e·A(e_b)` for `b = 0..128` (the instance factor's images of the dual
/// basis): a multiply-by-`X` chain plus the seven corrections.
fn instance_dual_images(e: Gf) -> [Gf; PACK] {
    let mut pow = [Gf::zero(); PACK]; // pow[n] = e·X^n
    pow[0] = e;
    for n in 1..PACK {
        pow[n] = pow[n - 1].mul_x();
    }
    let mut g = [Gf::zero(); PACK];
    g[0] = e;
    for b in 1..PACK {
        let mut v = pow[PACK - b];
        if (2..=6).contains(&b) {
            v += pow[7 - b];
        }
        if b == 1 {
            v += pow[6] + pow[1];
        }
        g[b] = v;
    }
    g
}

/// The monomial `X^u`.
#[cfg(test)]
#[inline]
fn monomial(u: usize) -> Gf {
    let mut w = [0u64; 2];
    w[u >> 6] = 1u64 << (u & 63);
    Gf::from_polynomial_words(w)
}

/// The ρ-only tables of the instance coefficients
/// `ρ′_a(e) = Σ_b ρ_b·bit_a(e·A(e_b)) = Σ_{u : bit_u(e)} C_{u,a}`,
/// `C_{u,a} = Σ_b ρ_b·bit_a(X^u·A(e_b))`, byte-indexed:
/// `T[pos][val][a] = Σ_{j ∈ val} C_{8·pos + j, a}` (8 MiB), so one instance
/// costs 16 row gathers of 128 elements.
pub(crate) struct RhoTables {
    t: Vec<Gf>,
    /// `C_{u,a}` at `u·128 + a`.
    c: Vec<Gf>,
}

impl RhoTables {
    /// Build coefficient tables for the 128 batching weights.
    pub(crate) fn new(rho: &[Gf]) -> Self {
        debug_assert_eq!(rho.len(), PACK);
        // `C_{u,a} = Σ_b ρ_b·bit_a(X^u·A(e_b)) = Φ_ρ(X^u·A(e_a))`: the pairing
        // `bit_a(g·A(e_b)) = c₀(g·A(e_b)·A(e_a))` is symmetric in `a` and `b`.
        // `X^u·A(e_a)` is a monomial plus at most the seven dual-basis
        // corrections, shifted and reduced, so `Φ_ρ` of it is a sum over its
        // few set bits (`rho_tables_match_transpose` pins this against the
        // definition). Column `a` walks `u` by a multiply-by-`X` chain.
        let columns = crate::dual_basis::dual_basis_cols();
        let by_a: Vec<[Gf; PACK]> = cfg_into_iter!(0..PACK)
            .map(|a| {
                let mut image = columns[a];
                let mut out = [Gf::zero(); PACK];
                for slot in out.iter_mut() {
                    *slot = phi_bit_sum(image, rho);
                    image = image.mul_x();
                }
                out
            })
            .collect();
        let c: Vec<[Gf; PACK]> = (0..PACK)
            .map(|u| core::array::from_fn(|a| by_a[a][u]))
            .collect();
        let flat: Vec<Gf> = c.iter().flat_map(|row| row.iter().copied()).collect();
        let mut t = vec![Gf::zero(); 16 * 256 * PACK];
        cfg_chunks_mut!(t, 256 * PACK)
            .enumerate()
            .for_each(|(pos, block)| {
                for j in 0..8usize {
                    let c_row = &c[(pos << 3) | j];
                    let half = 1usize << j;
                    for k in 0..half {
                        let (lower, upper) = block.split_at_mut((half + k) << LOG_PACKING);
                        let src = &lower[k << LOG_PACKING..(k + 1) << LOG_PACKING];
                        let dst = &mut upper[..PACK];
                        for ((d, s), c_ua) in dst.iter_mut().zip(src).zip(c_row.iter()) {
                            *d = *s + *c_ua;
                        }
                    }
                }
            });
        Self { t, c: flat }
    }

    /// `C_{u,·}`: the coefficient row of the monomial `X^u`.
    #[inline]
    fn row(&self, u: usize) -> &[Gf] {
        &self.c[u << LOG_PACKING..(u + 1) << LOG_PACKING]
    }

    /// `ρ′_a(e)` for `a = 0..128` into `out`.
    #[inline]
    fn coefficients(&self, e: Gf, out: &mut [Gf]) {
        debug_assert_eq!(out.len(), PACK);
        let w = e.as_words();
        let lb = w[0].to_le_bytes();
        let hb = w[1].to_le_bytes();
        out.fill(Gf::zero());
        for pos in 0..16usize {
            let byte = if pos < 8 { lb[pos] } else { hb[pos - 8] } as usize;
            let row = &self.t[((pos << 8) | byte) << LOG_PACKING..][..PACK];
            for (target, value) in out.iter_mut().zip(row) {
                *target += *value;
            }
        }
    }
}

// ---------------------------------------------------------------------
// The plane engine
// ---------------------------------------------------------------------

/// The precomputed plane tables of one packed-source repetition.
pub(crate) struct PackedSourcePlanes<'a> {
    /// `w`: nonconstant source cells per instance.
    local_width: usize,
    /// `1 + w·instances`; every later source column has weight zero.
    live_cols: usize,
    /// Per chunk `l`, per instance `i`: `e_{l,i}`.
    eq_inst: &'a [Vec<Gf>],
    /// Per chunk `l`, per phase `φ` (empty when no instance has that
    /// phase): `R_a(m)` for local pack `m`, stored at `m·128 + a` (the
    /// a′ pass walks one pack's 128 planes).
    r_tables: Vec<Vec<Vec<Gf>>>,
    /// The same tables plane-major, `R_a(m)` at `a·n_m + m` (the `h` pass
    /// sums one plane over an instance's packs).
    r_tables_by_plane: Vec<Vec<Vec<Gf>>>,
    /// Weight of the shared constant column (global column 0).
    constant_weight: Gf,
}

impl<'a> PackedSourcePlanes<'a> {
    /// Table bytes the engine would allocate for this shape.
    fn table_bytes(local_width: usize, instances: usize, chunks: usize) -> Option<usize> {
        let mut phases = [false; PACK];
        let period = PACK / gcd(local_width % PACK, PACK).max(1);
        for i in 0..instances.min(period) {
            phases[(1 + i * local_width) & (PACK - 1)] = true;
        }
        let mut entries = 0usize;
        for (phase, &used) in phases.iter().enumerate() {
            if used {
                entries = entries.checked_add(Self::local_packs(phase, local_width))?;
            }
        }
        entries
            .checked_mul(PACK)?
            .checked_mul(chunks)?
            .checked_mul(core::mem::size_of::<Gf>())
    }

    /// Whether this shape should use the plane engine: instances wide
    /// enough to amortise the per-instance `O(128²)` read-offs, tables
    /// within budget.
    pub(crate) fn eligible(local_width: usize, instances: usize, chunks: usize) -> bool {
        local_width >= MIN_LOCAL_WIDTH
            && instances >= 2
            && Self::table_bytes(local_width, instances, chunks)
                .is_some_and(|bytes| bytes <= MAX_TABLE_BYTES)
    }

    /// Global packs touched by an instance of phase `phase`.
    #[inline]
    fn local_packs(phase: usize, local_width: usize) -> usize {
        ((phase + local_width - 1) >> LOG_PACKING) + 1
    }

    /// Build the tables. `s[l]` holds the chunk-`l` local-column sums
    /// (index 0 = the constant column, `1..=w` the instance cells) and
    /// `eq_inst[l]` the instance equality table.
    pub(crate) fn new(
        local_width: usize,
        instances: usize,
        eq_inst: &'a [Vec<Gf>],
        s: &[Vec<Gf>],
        constant_weight: Gf,
    ) -> Self {
        Self::new_with(local_width, instances, eq_inst, s, constant_weight, true)
    }

    /// [`Self::new`] without the plane-major copy of the tables, which only
    /// [`Self::hs_fold`] reads: the verifier's form.
    pub(crate) fn new_basis_only(
        local_width: usize,
        instances: usize,
        eq_inst: &'a [Vec<Gf>],
        s: &[Vec<Gf>],
        constant_weight: Gf,
    ) -> Self {
        Self::new_with(local_width, instances, eq_inst, s, constant_weight, false)
    }

    fn new_with(
        local_width: usize,
        instances: usize,
        eq_inst: &'a [Vec<Gf>],
        s: &[Vec<Gf>],
        constant_weight: Gf,
        plane_major: bool,
    ) -> Self {
        assert!(local_width >= 1 && instances >= 1);
        assert_eq!(eq_inst.len(), s.len());
        assert!(eq_inst.iter().all(|table| table.len() == instances));
        assert!(s.iter().all(|table| table.len() == local_width + 1));
        let live_cols = 1 + local_width * instances;
        let chunks = s.len();

        let mut phases = [false; PACK];
        let period = PACK / gcd(local_width % PACK, PACK).max(1);
        for i in 0..instances.min(period) {
            phases[(1 + i * local_width) & (PACK - 1)] = true;
        }

        // Aligned local bit planes: planes[l][l'][a] bit j = bit_a(ŝ_{l, 1 + 128l' + j}).
        let planes: Vec<Vec<[[u64; 2]; PACK]>> = s
            .iter()
            .map(|s_l| {
                let unpacked: Vec<[u64; 2]> = cfg_iter!(s_l[1..])
                    .map(|&value| dual_unpack(value))
                    .collect();
                aligned_planes(&unpacked)
            })
            .collect();

        let mut jobs = Vec::new();
        for l in 0..chunks {
            for (phase, &used) in phases.iter().enumerate() {
                if used {
                    jobs.push((l, phase));
                }
            }
        }
        let built: Vec<((usize, usize), Vec<Gf>)> = cfg_into_iter!(jobs)
            .map(|(l, phase)| {
                let n_m = Self::local_packs(phase, local_width);
                let planes_l = &planes[l];
                let mut table = vec![Gf::zero(); n_m * PACK];
                for m in 0..n_m {
                    let cur = planes_l.get(m);
                    let prev = if m >= 1 { planes_l.get(m - 1) } else { None };
                    for a in 0..PACK {
                        let cur_a = cur.map_or([0u64; 2], |p| p[a]);
                        let u = if phase == 0 {
                            cur_a
                        } else {
                            let prev_a = prev.map_or([0u64; 2], |p| p[a]);
                            let lo = shl128(cur_a, phase);
                            let hi = shr128(prev_a, PACK - phase);
                            [lo[0] | hi[0], lo[1] | hi[1]]
                        };
                        table[(m << LOG_PACKING) | a] = dual_pack(u);
                    }
                }
                ((l, phase), table)
            })
            .collect();
        let mut r_tables: Vec<Vec<Vec<Gf>>> = (0..chunks).map(|_| vec![Vec::new(); PACK]).collect();
        let mut r_tables_by_plane: Vec<Vec<Vec<Gf>>> =
            (0..chunks).map(|_| vec![Vec::new(); PACK]).collect();
        for ((l, phase), table) in built {
            if plane_major {
                let n_m = table.len() >> LOG_PACKING;
                let mut by_plane = vec![Gf::zero(); table.len()];
                for m in 0..n_m {
                    for a in 0..PACK {
                        by_plane[a * n_m + m] = table[(m << LOG_PACKING) | a];
                    }
                }
                r_tables_by_plane[l][phase] = by_plane;
            }
            r_tables[l][phase] = table;
        }

        Self {
            local_width,
            live_cols,
            eq_inst,
            r_tables,
            r_tables_by_plane,
            constant_weight,
        }
    }

    /// The instance runs of source pack `y`: `(instance, phase, local pack)`
    /// for every instance with cells in the pack.
    #[inline]
    fn runs(&self, y: usize, mut visit: impl FnMut(usize, usize, usize)) {
        let w = self.local_width;
        let mut column = (y << LOG_PACKING).max(1);
        let end = ((y + 1) << LOG_PACKING).min(self.live_cols);
        while column < end {
            let instance = (column - 1) / w;
            let start = 1 + instance * w;
            visit(instance, start & (PACK - 1), y - (start >> LOG_PACKING));
            column = start + w;
        }
    }

    /// The batching message `h` (128 plane openings). Instance-major within
    /// each task: every `Q_{l,i}[a]` is summed in registers over the
    /// instance's packs (plane-major tables, one fold per plane), then read
    /// off through the byte tables of `Q_{l,i}`. An instance split across
    /// tasks is read off per part — exact by linearity.
    pub(crate) fn hs_fold<T: PackedBits>(&self, p_msg: &[T]) -> Box<[Gf; PACK]> {
        assert!(
            self.r_tables_by_plane
                .iter()
                .flatten()
                .any(|table| !table.is_empty())
                || self.r_tables.iter().flatten().all(|table| table.is_empty()),
            "the batching message needs the plane-major tables (`PackedSourcePlanes::new`)"
        );
        let w = self.local_width;
        let live_packs = self.live_cols.div_ceil(PACK).min(p_msg.len());
        let n_tasks = live_packs.div_ceil(TASK_PACKS);
        let partials: Vec<[Gf; PACK]> = cfg_into_iter!(0..n_tasks)
            .map(|task| {
                let y_lo = task * TASK_PACKS;
                let y_hi = (y_lo + TASK_PACKS).min(live_packs);
                let mut hs = [Gf::zero(); PACK];
                if y_lo == 0 {
                    // The shared constant column: weight W₀, basis A(e₀) = 1.
                    let p0 = Gf::from_polynomial_words(p_msg[0].bit_words());
                    let w0 = self.constant_weight.as_words();
                    for (b, h) in hs.iter_mut().enumerate() {
                        if (w0[b >> 6] >> (b & 63)) & 1 == 1 {
                            *h += p0;
                        }
                    }
                }
                let col_lo = (y_lo << LOG_PACKING).max(1);
                let col_hi = (y_hi << LOG_PACKING).min(self.live_cols);
                if col_lo >= col_hi {
                    return hs;
                }
                let mut tables = vec![Gf::zero(); 16 * 256];
                let mut fixed: Vec<kernel::Fixed> =
                    Vec::with_capacity(TASK_PACKS.min(w / PACK + 2));
                let mut reduced = [Gf::zero(); PACK];
                for instance in (col_lo - 1) / w..=(col_hi - 2) / w {
                    let start = 1 + instance * w;
                    let phase = start & (PACK - 1);
                    let y_i = start >> LOG_PACKING;
                    let n_m = Self::local_packs(phase, w);
                    let m_lo = y_lo.max(y_i) - y_i;
                    let m_hi = y_hi.min(y_i + n_m) - y_i;
                    fixed.clear();
                    fixed.extend((m_lo..m_hi).map(|m| {
                        kernel::fixed(&Gf::from_polynomial_words(p_msg[y_i + m].bit_words()))
                    }));
                    for (l, eq_inst_l) in self.eq_inst.iter().enumerate() {
                        let table = &self.r_tables_by_plane[l][phase];
                        for (a, slot) in reduced.iter_mut().enumerate() {
                            let row = &table[a * n_m + m_lo..a * n_m + m_hi];
                            let mut acc = kernel::zero();
                            for (r, f) in row.iter().zip(fixed.iter()) {
                                kernel::mul_acc(&mut acc, r, f);
                            }
                            *slot = kernel::reduce(&acc);
                        }
                        phi_byte_tables_into(&mut tables, |i| reduced[i]);
                        let images = instance_dual_images(eq_inst_l[instance]);
                        for (h, g) in hs.iter_mut().zip(images.iter()) {
                            *h += phi_from_words(*g.as_words(), &tables);
                        }
                    }
                }
                hs
            })
            .collect();
        let mut hs = Box::new([Gf::zero(); PACK]);
        for partial in partials {
            for (target, value) in hs.iter_mut().zip(partial) {
                *target += value;
            }
        }
        hs
    }

    /// One task of the plain-part basis: adds `Σ_v Φ_ρ(W_{(y,v)})·A(e_v)`
    /// (the constant column included at pack 0) to `slots[y − y_lo]` for the
    /// task's packs, one unreduced fixed-scalar multiply-accumulate per
    /// cell, the instance coefficients read off once per instance.
    fn a_prime_task(
        &self,
        coefficient_tables: &RhoTables,
        rho_tables: &[Gf],
        y_lo: usize,
        slots: &mut [Gf],
    ) {
        let chunks = self.r_tables.len();
        let mut cache = CoefficientCache::<1>::new(chunks);
        for (offset, slot) in slots.iter_mut().enumerate() {
            let y = y_lo + offset;
            let mut acc = kernel::zero();
            if y == 0 {
                // Φ_ρ(W₀)·A(e₀) = Φ_ρ(W₀).
                *slot += phi_from_words(*self.constant_weight.as_words(), rho_tables);
            }
            self.runs(y, |instance, phase, m| {
                for l in 0..chunks {
                    let fixed =
                        cache.fixed(coefficient_tables, l, instance, self.eq_inst[l][instance]);
                    let r = &self.r_tables[l][phase][m << LOG_PACKING..(m + 1) << LOG_PACKING];
                    for (r_a, f_a) in r.iter().zip(fixed.iter()) {
                        kernel::mul_acc(&mut acc, r_a, f_a);
                    }
                }
            });
            *slot += kernel::reduce(&acc);
        }
    }

    /// Adds the plain part of the ρ-batched basis `a′` to `basis` (one entry
    /// per source pack), tasks of `task_packs` packs in parallel; each pack
    /// is owned by one task, so the sum is deterministic whatever the task
    /// size or thread count. The verifier's entry point ([`Self::a_prime`]
    /// is the prover's, which also emits the round-0 pair from the same task
    /// kernel).
    pub(crate) fn add_a_prime(
        &self,
        coefficient_tables: &RhoTables,
        rho_tables: &[Gf],
        basis: &mut [Gf],
        task_packs: usize,
    ) {
        cfg_chunks_mut!(basis, task_packs)
            .enumerate()
            .for_each(|(task, slots)| {
                self.a_prime_task(coefficient_tables, rho_tables, task * task_packs, slots);
            });
    }

    /// The ρ-batched dual-basis Ligerito basis `a′` over `p_msg.len()`
    /// packs, and flock's round-0 pair `(Σ_j P[2j]·a′[2j],
    /// Σ_j (P[2j]+P[2j+1])·(a′[2j]+a′[2j+1]))` (the
    /// `recursive_prover_with_basis_precomputed_round0` contract, exactly
    /// as `fill_phi_basis_round0` computes it).
    pub(crate) fn a_prime<T: PackedBits>(&self, rho: &[Gf], p_msg: &[T]) -> (Vec<Gf>, (Gf, Gf)) {
        let n_packs = p_msg.len();
        debug_assert!(
            n_packs.is_multiple_of(2) || n_packs == 1,
            "flock messages are even-sized"
        );
        let rho_tables = phi_byte_tables(rho, Gf::one());
        let coefficient_tables = RhoTables::new(rho);
        let mut out = vec![Gf::zero(); n_packs];
        let partials: Vec<(Gf, Gf)> = cfg_chunks_mut!(out, TASK_PACKS)
            .enumerate()
            .map(|(task, slots)| {
                let y_lo = task * TASK_PACKS;
                self.a_prime_task(&coefficient_tables, &rho_tables, y_lo, slots);
                // Flock's round-0 pair over this task's (even-aligned) pairs.
                let zero = Gf::zero();
                let mut u0 = Gf::wide_zero(&zero);
                let mut u2 = Gf::wide_zero(&zero);
                let mut j = 0usize;
                while j + 1 < slots.len() {
                    let b0 = slots[j];
                    let b1 = slots[j + 1];
                    let f0 = Gf::from_polynomial_words(p_msg[y_lo + j].bit_words());
                    let f1 = Gf::from_polynomial_words(p_msg[y_lo + j + 1].bit_words());
                    Gf::wide_add_assign(&mut u0, &Gf::mul_wide(&f0, &b0));
                    Gf::wide_add_assign(&mut u2, &Gf::mul_wide(&(f0 + f1), &(b0 + b1)));
                    j += 2;
                }
                (Gf::from_wide(u0), Gf::from_wide(u2))
            })
            .collect();
        let mut u0 = Gf::zero();
        let mut u2 = Gf::zero();
        for (p0, p2) in &partials {
            u0 += *p0;
            u2 += *p2;
        }
        (out, (u0, u2))
    }
}

// ---------------------------------------------------------------------
// The affine tail: an identity compact tail's basis from its eq tensors
// ---------------------------------------------------------------------

/// The verifier's ρ-batched basis of an IDENTITY compact tail
/// ([`circuit::linear_map::binary::ChainedSourceTail`] whose local map is the identity):
/// source column `j ∈ [source_start, source_start + len)` carries the
/// derived-row weight of row `r(j) = row_start + (j − source_start)`,
///
/// ```text
/// E_r = Σ_l zc_l[r >> t] · eq_l[r & (2^t − 1)]
/// ```
///
/// (`VirtRowCoeffs`: per weight chunk `l` the η-scaled high-index table
/// `zc_l` and the low-index eq table `eq_l`). Those weights are never
/// materialized. As for the packed-source planes, the dual-basis identity
/// separates the two factors — with the high index `c` as the instance and
/// the low index `b` as the local column —
///
/// ```text
/// a′_tail(y) = Σ_v Φ_ρ(E_{r(128y+v)})·A(e_v)
///            = Σ_l Σ_c Σ_a ρ′_{l,a}(zc_l[c]) · R_{l,a}(y, c),
/// R_{l,a}(y, c) = Σ_{v : r(128y+v) >> t = c} bit_a(êq_l[b(128y+v)]) · A(e_v),
/// ```
///
/// so a pack costs 128 unreduced fixed-scalar multiply-accumulates per
/// chunk. The local planes `R` depend on the pack only through
/// `b(128y) = (128y + row_start − source_start) mod 2^t`, i.e. through the
/// aligned block `m = b(128y) >> 7` and the phase `φ = b(128y) & 127`
/// (constant over the tail): one 128-plane table per chunk and block
/// serves every pack. Packs whose 128 cells straddle a high-index step
/// (`m` the last block, `φ ≠ 0`) take the block's upper part with `c` and
/// block 0's lower part with `c + 1`; the (at most two) packs partially
/// covered by the tail take masked pieces transposed on the spot.
pub(crate) struct AffineTailPlanes {
    source_start: usize,
    len: usize,
    row_start: usize,
    /// `t`: low row bits.
    t: usize,
    /// Per chunk: `zc_l` (indexed by the high index `c`).
    zc: Vec<Vec<Gf>>,
    /// Per chunk: `êq_l[b]` (the dual unpacking of `eq_l[b]`) for `b < 2^t`.
    unpacked: Vec<Vec<[u64; 2]>>,
    /// Per chunk, per aligned block `m`, per plane `a` (at `m·128 + a`):
    /// `Σ_v bit_a(êq_l[128m + φ + v])·A(e_v)`, the planes of a pack whose
    /// first cell sits at low index `128m + φ` (valid for `m + 1 < 2^{t−7}`
    /// when `φ ≠ 0`, every `m` when `φ = 0`).
    r_tables: Vec<Vec<Gf>>,
    /// Per chunk: the straddling pack's two halves for `φ ≠ 0` — cells
    /// `v < 128 − φ` from the last block (`r_low`) and `v ≥ 128 − φ` from
    /// block 0 of the next high index (`r_high`).
    r_low: Vec<[Gf; PACK]>,
    r_high: Vec<[Gf; PACK]>,
}

impl AffineTailPlanes {
    /// Builds the tables from the chunk tensors. `eq[l].len() = 2^t ≥ 128`;
    /// `zc[l].len()` covers every high index the tail's rows reach.
    pub(crate) fn new(
        source_start: usize,
        len: usize,
        row_start: usize,
        eq: &[Vec<Gf>],
        zc: &[Vec<Gf>],
    ) -> Self {
        assert_eq!(eq.len(), zc.len());
        assert!(!eq.is_empty());
        let rows = eq[0].len();
        assert!(rows.is_power_of_two() && rows >= PACK);
        assert!(eq.iter().all(|table| table.len() == rows));
        let t = rows.trailing_zeros() as usize;
        let phase = (row_start.wrapping_sub(source_start)) & (PACK - 1);
        let blocks = rows >> LOG_PACKING;
        let unpacked: Vec<Vec<[u64; 2]>> = eq
            .iter()
            .map(|table| cfg_iter!(table).map(|&value| dual_unpack(value)).collect())
            .collect();
        // The planes of every aligned block: planes[l][m][a].
        let planes: Vec<Vec<[[u64; 2]; PACK]>> =
            unpacked.iter().map(|table| aligned_planes(table)).collect();
        let r_tables = planes
            .iter()
            .map(|planes_l| {
                let mut table = vec![Gf::zero(); blocks * PACK];
                for m in 0..blocks {
                    for a in 0..PACK {
                        let u = if phase == 0 {
                            planes_l[m][a]
                        } else if m + 1 < blocks {
                            let lo = shr128(planes_l[m][a], phase);
                            let hi = shl128(planes_l[m + 1][a], PACK - phase);
                            [lo[0] | hi[0], lo[1] | hi[1]]
                        } else {
                            // The straddling block is served by `r_low`/`r_high`.
                            [0, 0]
                        };
                        table[(m << LOG_PACKING) | a] = dual_pack(u);
                    }
                }
                table
            })
            .collect();
        let (r_low, r_high) = if phase == 0 {
            (Vec::new(), Vec::new())
        } else {
            planes
                .iter()
                .map(|planes_l| {
                    let mut low = [Gf::zero(); PACK];
                    let mut high = [Gf::zero(); PACK];
                    for a in 0..PACK {
                        low[a] = dual_pack(shr128(planes_l[blocks - 1][a], phase));
                        high[a] = dual_pack(shl128(planes_l[0][a], PACK - phase));
                    }
                    (low, high)
                })
                .unzip()
        };
        Self {
            source_start,
            len,
            row_start,
            t,
            zc: zc.to_vec(),
            unpacked,
            r_tables,
            r_low,
            r_high,
        }
    }

    /// The derived row of source column `j`.
    #[inline]
    fn row(&self, j: usize) -> usize {
        self.row_start + (j - self.source_start)
    }

    /// Adds the tail's basis to `basis` (one entry per source pack); packs
    /// in tasks of `task_packs`, each pack owned by one task (the sum is
    /// deterministic whatever the task size or thread count). Full packs go
    /// through the per-block lookup tables when the tail spans at least
    /// [`LOOKUP_MIN_HIGH_PER_BLOCK`] high indices per block, amortizing the
    /// table build over the tail's packs; otherwise they use plane products.
    /// The two agree exactly
    /// (`affine_tail_lookup_matches_products`).
    pub(crate) fn add_a_prime(
        &self,
        coefficient_tables: &RhoTables,
        basis: &mut [Gf],
        task_packs: usize,
    ) {
        let highs = (self.len >> self.t).saturating_sub(1);
        let lookup = highs >= LOOKUP_MIN_HIGH_PER_BLOCK;
        self.add_a_prime_with(coefficient_tables, basis, task_packs, lookup);
    }

    /// [`Self::add_a_prime`] with the full-pack mode chosen by the caller.
    pub(crate) fn add_a_prime_with(
        &self,
        coefficient_tables: &RhoTables,
        basis: &mut [Gf],
        task_packs: usize,
        lookup: bool,
    ) {
        if self.len == 0 {
            return;
        }
        let tables = lookup.then(|| self.lookup_tables(coefficient_tables));
        let first_pack = self.source_start >> LOG_PACKING;
        let last_pack = (self.source_start + self.len - 1) >> LOG_PACKING;
        cfg_chunks_mut!(basis, task_packs)
            .enumerate()
            .for_each(|(task, slots)| {
                let y_lo = task * task_packs;
                let y_hi = y_lo + slots.len();
                if y_hi <= first_pack || y_lo > last_pack {
                    return;
                }
                let mut cache = CoefficientCache::<2>::new(self.zc.len());
                for y in y_lo.max(first_pack)..y_hi.min(last_pack + 1) {
                    let slot = &mut slots[y - y_lo];
                    let column_lo = (y << LOG_PACKING).max(self.source_start);
                    let column_hi = ((y + 1) << LOG_PACKING).min(self.source_start + self.len);
                    if column_hi - column_lo == PACK {
                        *slot += match &tables {
                            Some(tables) => self.full_pack_lookup(tables, y),
                            None => self.full_pack(coefficient_tables, &mut cache, y),
                        };
                    } else {
                        *slot += self.partial_pack(
                            coefficient_tables,
                            &mut cache,
                            y,
                            column_lo,
                            column_hi,
                        );
                    }
                }
            });
    }

    /// The per-block lookup tables: for chunk `l` and block `m` (the
    /// straddling halves as two extra blocks when the phase is nonzero),
    /// `D_{l,m}[u] = Σ_a C_{u,a}·R_{l,a}(m)` and its 16 byte-position
    /// subset-sum tables, so that a full pack at high index `c` costs
    /// `Φ`-style 16 gathers: `Σ_a ρ′_{l,a}(zc_l[c])·R_{l,a}(m)
    /// = Σ_a Σ_u bit_u(zc_l[c])·C_{u,a}·R_{l,a}(m) = Σ_u bit_u(zc_l[c])·D_{l,m}[u]`
    /// (`ρ′_{l,a}(e) = Σ_{u : bit_u(e)} C_{u,a}` is F₂-linear in `e`).
    fn lookup_tables(&self, coefficient_tables: &RhoTables) -> TailLookupTables {
        let chunks = self.zc.len();
        let blocks = 1usize << (self.t - LOG_PACKING);
        let straddle = !self.r_low.is_empty();
        let per_chunk = blocks + if straddle { 2 } else { 0 };
        let plane = |l: usize, block: usize| -> &[Gf] {
            if block < blocks {
                &self.r_tables[l][block << LOG_PACKING..(block + 1) << LOG_PACKING]
            } else if block == blocks {
                &self.r_low[l]
            } else {
                &self.r_high[l]
            }
        };
        let d: Vec<Gf> = cfg_into_iter!(0..chunks * per_chunk * PACK)
            .map(|index| {
                let u = index & (PACK - 1);
                let block = (index >> LOG_PACKING) % per_chunk;
                let l = (index >> LOG_PACKING) / per_chunk;
                let r = plane(l, block);
                let mut acc = kernel::zero();
                for (c_ua, r_a) in coefficient_tables.row(u).iter().zip(r.iter()) {
                    kernel::mul_acc(&mut acc, r_a, &kernel::fixed(c_ua));
                }
                kernel::reduce(&acc)
            })
            .collect();
        let mut tables = vec![Gf::zero(); chunks * per_chunk * 16 * 256];
        cfg_chunks_mut!(tables, 16 * 256)
            .enumerate()
            .for_each(|(block, table)| {
                phi_byte_tables_into(table, |i| d[(block << LOG_PACKING) + i]);
            });
        TailLookupTables {
            tables,
            per_chunk,
            blocks,
        }
    }

    /// A full pack by the lookup tables.
    fn full_pack_lookup(&self, tables: &TailLookupTables, y: usize) -> Gf {
        let rows = 1usize << self.t;
        let r0 = self.row(y << LOG_PACKING);
        let c = r0 >> self.t;
        let b0 = r0 & (rows - 1);
        let m = b0 >> LOG_PACKING;
        let phase = b0 & (PACK - 1);
        let mut acc = Gf::zero();
        if phase == 0 || m + 1 < tables.blocks {
            for (l, zc_l) in self.zc.iter().enumerate() {
                acc += phi_from_words(*zc_l[c].as_words(), tables.table(l, m));
            }
        } else {
            for (l, zc_l) in self.zc.iter().enumerate() {
                acc += phi_from_words(*zc_l[c].as_words(), tables.table(l, tables.blocks));
                acc += phi_from_words(*zc_l[c + 1].as_words(), tables.table(l, tables.blocks + 1));
            }
        }
        acc
    }

    /// A pack entirely inside the tail: the tabulated planes.
    fn full_pack(
        &self,
        coefficient_tables: &RhoTables,
        cache: &mut CoefficientCache<2>,
        y: usize,
    ) -> Gf {
        let rows = 1usize << self.t;
        let blocks = rows >> LOG_PACKING;
        let r0 = self.row(y << LOG_PACKING);
        let c = r0 >> self.t;
        let b0 = r0 & (rows - 1);
        let m = b0 >> LOG_PACKING;
        let phase = b0 & (PACK - 1);
        let mut acc = kernel::zero();
        if phase == 0 || m + 1 < blocks {
            for l in 0..self.zc.len() {
                let fixed = cache.fixed(coefficient_tables, l, c, self.zc[l][c]);
                let r = &self.r_tables[l][m << LOG_PACKING..(m + 1) << LOG_PACKING];
                for (r_a, f_a) in r.iter().zip(fixed.iter()) {
                    kernel::mul_acc(&mut acc, r_a, f_a);
                }
            }
        } else {
            for l in 0..self.zc.len() {
                let fixed = cache.fixed(coefficient_tables, l, c, self.zc[l][c]);
                for (r_a, f_a) in self.r_low[l].iter().zip(fixed.iter()) {
                    kernel::mul_acc(&mut acc, r_a, f_a);
                }
                let fixed = cache.fixed(coefficient_tables, l, c + 1, self.zc[l][c + 1]);
                for (r_a, f_a) in self.r_high[l].iter().zip(fixed.iter()) {
                    kernel::mul_acc(&mut acc, r_a, f_a);
                }
            }
        }
        kernel::reduce(&acc)
    }

    /// A pack the tail covers only on `[column_lo, column_hi)`: the covered
    /// cells' planes, transposed on the spot, one piece per high index.
    fn partial_pack(
        &self,
        coefficient_tables: &RhoTables,
        cache: &mut CoefficientCache<2>,
        y: usize,
        column_lo: usize,
        column_hi: usize,
    ) -> Gf {
        let rows = 1usize << self.t;
        let base = y << LOG_PACKING;
        let mut acc = kernel::zero();
        let mut column = column_lo;
        while column < column_hi {
            let r = self.row(column);
            let c = r >> self.t;
            let b = r & (rows - 1);
            let piece = (column_hi - column).min(rows - b);
            for l in 0..self.zc.len() {
                let mut block = [[0u64; 2]; PACK];
                block[column - base..column - base + piece]
                    .copy_from_slice(&self.unpacked[l][b..b + piece]);
                let planes = transpose_128x128(&block);
                let fixed = cache.fixed(coefficient_tables, l, c, self.zc[l][c]);
                for (plane, f_a) in planes.iter().zip(fixed.iter()) {
                    kernel::mul_acc(&mut acc, &dual_pack(*plane), f_a);
                }
            }
            column += piece;
        }
        kernel::reduce(&acc)
    }

    /// The per-cell weights of pack `y` (the plain sum `Σ_l zc_l[c]·eq_l[b]`
    /// per covered cell): the tests' reference.
    #[cfg(test)]
    pub(crate) fn add_pack_weights(&self, eq: &[Vec<Gf>], y: usize, out: &mut [Gf; PACK]) {
        let base = y << LOG_PACKING;
        let lo = base.max(self.source_start);
        let hi = (base + PACK).min(self.source_start + self.len);
        for column in lo..hi {
            let r = self.row(column);
            let (c, b) = (r >> self.t, r & ((1usize << self.t) - 1));
            for (zc_l, eq_l) in self.zc.iter().zip(eq) {
                out[column - base] += zc_l[c] * eq_l[b];
            }
        }
    }
}

/// High indices per block from which [`AffineTailPlanes::add_a_prime`]
/// builds the lookup tables: the build costs `128·128` products per block,
/// the plane products `128` per pack, and a block serves one pack per high
/// index.
const LOOKUP_MIN_HIGH_PER_BLOCK: usize = 256;

/// The affine tail's per-block byte tables (see
/// [`AffineTailPlanes::lookup_tables`]).
struct TailLookupTables {
    /// Per chunk, per block, 16 byte positions × 256 values.
    tables: Vec<Gf>,
    per_chunk: usize,
    blocks: usize,
}

impl TailLookupTables {
    #[inline]
    fn table(&self, l: usize, block: usize) -> &[Gf] {
        let start = (l * self.per_chunk + block) * 16 * 256;
        &self.tables[start..start + 16 * 256]
    }
}

/// Prepared `ρ′_{l,·}(e)` multipliers per chunk. Repetition tasks need one
/// cached index; tail tasks need two so packs straddling a row keep both halves.
struct CoefficientCache<const ENTRIES: usize> {
    entries: Vec<CachedCoefficients>,
    scratch: Vec<Gf>,
}

#[derive(Clone)]
struct CachedCoefficients {
    key: Option<usize>,
    fixed: [kernel::Fixed; PACK],
}

impl<const ENTRIES: usize> CoefficientCache<ENTRIES> {
    fn new(chunks: usize) -> Self {
        Self {
            entries: vec![
                CachedCoefficients {
                    key: None,
                    fixed: [kernel::fixed(&Gf::zero()); PACK],
                };
                ENTRIES * chunks
            ],
            scratch: vec![Gf::zero(); PACK],
        }
    }

    /// The prepared multipliers of chunk `l` at index `c` with factor `e`.
    fn fixed(
        &mut self,
        coefficient_tables: &RhoTables,
        l: usize,
        c: usize,
        e: Gf,
    ) -> &[kernel::Fixed] {
        let entry = &mut self.entries[ENTRIES * l + c % ENTRIES];
        if entry.key != Some(c) {
            coefficient_tables.coefficients(e, &mut self.scratch);
            for (f, value) in entry.fixed.iter_mut().zip(self.scratch.iter()) {
                *f = kernel::fixed(value);
            }
            entry.key = Some(c);
        }
        &entry.fixed
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crate::dual_basis::{c0_bit, dual_basis_cols};

    fn splitmix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn sample(seed: u64) -> Gf {
        Gf::from_polynomial_words([splitmix(seed), splitmix(seed ^ 0xD1CE)])
    }

    fn bit(w: &[u64; 2], i: usize) -> u64 {
        (w[i >> 6] >> (i & 63)) & 1
    }

    /// `dual_pack` IS `Σ_v bit_v(u)·A(e_v)` for the tabulated columns.
    #[test]
    fn dual_pack_matches_columns() {
        let cols = dual_basis_cols();
        for t in 0..256u64 {
            let u = [splitmix(0x1000 + t), splitmix(0x2000 + t)];
            let mut expect = Gf::zero();
            for (v, col) in cols.iter().enumerate() {
                if bit(&u, v) == 1 {
                    expect += *col;
                }
            }
            assert_eq!(dual_pack(u), expect, "sample {t}");
        }
        for v in 0..PACK {
            let mut u = [0u64; 2];
            u[v >> 6] = 1u64 << (v & 63);
            assert_eq!(dual_pack(u), cols[v], "unit vector {v}");
        }
    }

    /// The closed-form `dual_unpack` IS the defining `c₀(X^a·s)` scan: on
    /// random elements, every monomial, every dual-basis column and the
    /// all-ones element.
    #[test]
    fn dual_unpack_closed_form_matches_multiplication_chain() {
        let mut inputs: Vec<Gf> = (0..2048u64).map(|t| sample(0xF000 + t)).collect();
        inputs.extend((0..PACK).map(monomial));
        inputs.extend(dual_basis_cols());
        inputs.push(Gf::from_polynomial_words([u64::MAX, u64::MAX]));
        inputs.push(Gf::zero());
        for (index, &s) in inputs.iter().enumerate() {
            assert_eq!(
                dual_unpack(s),
                dual_unpack_by_multiplication(s),
                "input {index}"
            );
        }
    }

    /// `dual_unpack` inverts `dual_pack`, and the bit-extraction identity
    /// `bit_b(e·s) = Σ_a bit_a(e·A(e_b))·bit_a(ŝ)` holds.
    #[test]
    fn dual_unpack_inverts_and_extracts_bits() {
        for t in 0..64u64 {
            let s = sample(0x3000 + t);
            let unpacked = dual_unpack(s);
            assert_eq!(dual_pack(unpacked), s, "round trip {t}");
            let e = sample(0x4000 + t);
            let product = e * s;
            let images = instance_dual_images(e);
            for b in 0..PACK {
                let g = images[b].as_words();
                let parity =
                    ((g[0] & unpacked[0]).count_ones() + (g[1] & unpacked[1]).count_ones()) & 1;
                assert_eq!(
                    u64::from(parity),
                    bit(product.as_words(), b),
                    "sample {t} bit {b}"
                );
                assert_eq!(
                    c0_bit(product * dual_basis_cols()[b]),
                    bit(product.as_words(), b)
                );
            }
        }
    }

    #[test]
    fn instance_dual_images_match_columns() {
        let cols = dual_basis_cols();
        for t in 0..16u64 {
            let e = sample(0x5000 + t);
            let images = instance_dual_images(e);
            for b in 0..PACK {
                assert_eq!(images[b], e * cols[b], "sample {t} column {b}");
            }
        }
    }

    #[test]
    fn transpose_128x128_is_a_transpose() {
        let mut rows = [[0u64; 2]; PACK];
        for (r, row) in rows.iter_mut().enumerate() {
            *row = [splitmix(0x6000 + r as u64), splitmix(0x7000 + r as u64)];
        }
        let cols = transpose_128x128(&rows);
        for r in 0..PACK {
            for c in 0..PACK {
                assert_eq!(bit(&cols[c], r), bit(&rows[r], c), "({r}, {c})");
            }
        }
        assert_eq!(transpose_128x128(&cols), rows);
    }

    #[test]
    fn shifts_agree_with_u128() {
        for t in 0..32u64 {
            let x = [splitmix(0x8000 + t), splitmix(0x9000 + t)];
            let value = u128::from(x[0]) | (u128::from(x[1]) << 64);
            for n in 0..PACK {
                let l = shl128(x, n);
                let r = shr128(x, n);
                let lv = value << n;
                let rv = value >> n;
                assert_eq!(u128::from(l[0]) | (u128::from(l[1]) << 64), lv, "shl {n}");
                assert_eq!(u128::from(r[0]) | (u128::from(r[1]) << 64), rv, "shr {n}");
            }
        }
    }

    /// The fixed-scalar accumulator reproduces `Σ x_k·f` exactly.
    #[test]
    fn kernel_accumulates_products() {
        for t in 0..16u64 {
            let f = sample(0xB000 + t);
            let fixed = kernel::fixed(&f);
            let mut acc = kernel::zero();
            let mut expect = Gf::zero();
            for k in 0..200u64 {
                let x = sample(0xC000 + t * 1000 + k);
                kernel::mul_acc(&mut acc, &x, &fixed);
                expect += x * f;
                assert_eq!(kernel::reduce(&acc), expect, "sample {t} term {k}");
            }
        }
    }

    /// The affine-tail engine IS the per-cell sum `Σ_v Φ_ρ(E_{r(128y+v)})·A(e_v)`
    /// over the tail's packs: random eq/zc tensors, one and two chunks,
    /// every phase class (aligned, unaligned, straddling), tails that start
    /// and end inside a pack, tails shorter than a pack, and offsets that
    /// place the rows at the top of the low index.
    #[test]
    fn affine_tail_planes_match_cellwise_sum() {
        let cols = dual_basis_cols();
        let rho: Vec<Gf> = (0..PACK).map(|i| sample(0x1_0000 + i as u64)).collect();
        let coefficient_tables = RhoTables::new(&rho);
        let phi = |x: Gf| -> Gf {
            let w = x.as_words();
            let mut acc = Gf::zero();
            for i in 0..PACK {
                if (w[i >> 6] >> (i & 63)) & 1 == 1 {
                    acc += rho[i];
                }
            }
            acc
        };
        let mut trial = 0u64;
        for t in [7usize, 8, 10] {
            let rows = 1usize << t;
            for chunks in [1usize, 2] {
                for (source_start, len, row_start) in [
                    (0usize, 5 * rows + 3, 0usize),
                    (1, 4 * rows, 7 * rows + 1),
                    (3, 3 * rows + 100, 2 * rows + 8),
                    (129, 2 * rows + 1, 65),
                    (200, 50, 3 * rows + 200),
                    (128, 3 * rows, 3 * rows - 128),
                    (700, rows - 1, rows + 12),
                ] {
                    trial += 1;
                    let eq: Vec<Vec<Gf>> = (0..chunks)
                        .map(|l| {
                            (0..rows)
                                .map(|b| sample(0x2_0000 + trial * 4096 + (l * rows + b) as u64))
                                .collect()
                        })
                        .collect();
                    let highs = (row_start + len).div_ceil(rows) + 1;
                    let zc: Vec<Vec<Gf>> = (0..chunks)
                        .map(|l| {
                            (0..highs)
                                .map(|c| sample(0x3_0000 + trial * 64 + (l * highs + c) as u64))
                                .collect()
                        })
                        .collect();
                    let tail = AffineTailPlanes::new(source_start, len, row_start, &eq, &zc);
                    let n_packs = (source_start + len).div_ceil(PACK) + 1;
                    let mut basis = vec![Gf::zero(); n_packs];
                    for (y, slot) in basis.iter_mut().enumerate() {
                        *slot = sample(0x4_0000 + y as u64); // the engine ADDS
                    }
                    let before = basis.clone();
                    // Three packs per task: task boundaries fall inside the tail;
                    // alternate the full-pack mode so both are checked cellwise.
                    tail.add_a_prime_with(
                        &coefficient_tables,
                        &mut basis,
                        3,
                        trial.is_multiple_of(2),
                    );
                    for y in 0..n_packs {
                        let mut expect = before[y];
                        let mut weights = [Gf::zero(); PACK];
                        tail.add_pack_weights(&eq, y, &mut weights);
                        for v in 0..PACK {
                            let j = (y << LOG_PACKING) | v;
                            if j < source_start || j >= source_start + len {
                                assert_eq!(weights[v], Gf::zero());
                                continue;
                            }
                            let r = row_start + (j - source_start);
                            let mut w = Gf::zero();
                            for l in 0..chunks {
                                w += zc[l][r >> t] * eq[l][r & (rows - 1)];
                            }
                            assert_eq!(weights[v], w, "trial {trial} pack {y} cell {v}");
                            expect += phi(w) * cols[v];
                        }
                        assert_eq!(
                            basis[y], expect,
                            "trial {trial} t {t} chunks {chunks} pack {y}"
                        );
                    }
                }
            }
        }
    }

    /// The lookup-table mode and the plane-product mode of the affine tail
    /// agree pack for pack, on random tensors at every phase class, with
    /// one and two chunks (the lookup mode is otherwise selected only for
    /// tails spanning hundreds of high indices).
    #[test]
    fn affine_tail_lookup_matches_products() {
        let rho: Vec<Gf> = (0..PACK).map(|i| sample(0x5_0000 + i as u64)).collect();
        let coefficient_tables = RhoTables::new(&rho);
        let mut trial = 0u64;
        for t in [7usize, 9] {
            let rows = 1usize << t;
            for chunks in [1usize, 2] {
                for (source_start, len, row_start) in [
                    (0usize, 40 * rows + 3, 0usize),
                    (5, 33 * rows, 2 * rows + 9),
                    (129, 20 * rows + 1, 65),
                ] {
                    trial += 1;
                    let eq: Vec<Vec<Gf>> = (0..chunks)
                        .map(|l| {
                            (0..rows)
                                .map(|b| sample(0x6_0000 + trial * 4096 + (l * rows + b) as u64))
                                .collect()
                        })
                        .collect();
                    let highs = (row_start + len).div_ceil(rows) + 1;
                    let zc: Vec<Vec<Gf>> = (0..chunks)
                        .map(|l| {
                            (0..highs)
                                .map(|c| sample(0x7_0000 + trial * 64 + (l * highs + c) as u64))
                                .collect()
                        })
                        .collect();
                    let tail = AffineTailPlanes::new(source_start, len, row_start, &eq, &zc);
                    let n_packs = (source_start + len).div_ceil(PACK) + 1;
                    let mut products = vec![Gf::zero(); n_packs];
                    let mut lookup = vec![Gf::zero(); n_packs];
                    tail.add_a_prime_with(&coefficient_tables, &mut products, 5, false);
                    tail.add_a_prime_with(&coefficient_tables, &mut lookup, 7, true);
                    assert_eq!(lookup, products, "trial {trial} t {t} chunks {chunks}");
                }
            }
        }
    }

    /// `RhoTables::coefficients` IS `Σ_b ρ_b·bit_a(e·A(e_b))`.
    #[test]
    fn rho_tables_match_transpose() {
        let rho: Vec<Gf> = (0..PACK).map(|i| sample(0xD000 + i as u64)).collect();
        let tables = RhoTables::new(&rho);
        let mut got = vec![Gf::zero(); PACK];
        for t in 0..8u64 {
            let e = sample(0xE000 + t);
            tables.coefficients(e, &mut got);
            let images = instance_dual_images(e);
            for a in 0..PACK {
                let mut expect = Gf::zero();
                for (b, g) in images.iter().enumerate() {
                    if bit(g.as_words(), a) == 1 {
                        expect += rho[b];
                    }
                }
                assert_eq!(got[a], expect, "sample {t} coefficient {a}");
            }
        }
    }
}
