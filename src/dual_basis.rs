//! The GHASH dual-basis embedding of the paper's "Bilinear Embeddings"
//! appendix (§Coefficient projection): the triple
//! `(W, A, H)` with `⟨w, a⟩_{F₂} = H(W(w)·A(a))` over
//! `K = F₂[X]/(f)`, `f = X¹²⁸ + X⁷ + X² + X + 1`, instantiated as
//!
//!   * `W = Id` — the commitment's monomial-basis packing (bit `v` of a
//!     cell ↔ `X^v`, exactly `commit_rs_flock_from_rows`'s layout);
//!   * `H = c₀` — projection to the coefficient of `X⁰` (`f₀ = 1`, so no
//!     inversion anywhere);
//!   * `A` — the `μ_H`-dual basis of the monomial basis, the theorem's
//!     bordered upper anti-triangular Hankel matrix `A₀₀ = 1`,
//!     `A_{ij} = −f_{i+j}/f₀` for `i, j ≥ 1`, `i + j ≤ 128`.
//!
//! For GHASH the columns come out as a REVERSAL of the `v ≥ 1`
//! coordinates plus seven XOR corrections:
//!
//!   * `A(e₀) = 1`;
//!   * `A(e_v) = X^{128−v}` for `7 ≤ v ≤ 127`;
//!   * `A(e_v) = X^{128−v} + X^{7−v}` for `2 ≤ v ≤ 6` (five corrections);
//!   * `A(e₁) = X¹²⁷ + X⁶ + X` (two corrections).
//!
//! The unit tests below re-derive `A` by brute force (Gaussian
//! elimination of the pairing matrix `c₀(X^i·X^j)`) and pin the
//! plane-decomposition and per-pack identities the virtual opening's
//! batching protocol relies on.

use crate::poly::univariate::binary_gf128::Gf128 as Gf;

/// The monomial `X^k` as a field element (bit `k` set).
#[inline]
pub(crate) fn monomial(k: usize) -> Gf {
    debug_assert!(k < 128);
    let mut w = [0u64; 2];
    w[k >> 6] = 1u64 << (k & 63);
    Gf::from_polynomial_words(w)
}

/// `H = c₀`: the coefficient of `X⁰` (bit 0).
#[inline(always)]
pub(crate) fn c0_bit(g: Gf) -> u64 {
    g.as_words()[0] & 1
}

/// The 128 columns `A(e_v)` of the dual-basis embedding (see the module
/// header). `dual_basis_cols()[v]` pairs to `δ_{uv}` against `X^u` under
/// `μ_H(x, y) = c₀(x·y)`.
pub(crate) fn dual_basis_cols() -> [Gf; 128] {
    let mut cols = [Gf::zero(); 128];
    cols[0] = monomial(0);
    for (v, col) in cols.iter_mut().enumerate().skip(1) {
        let mut g = monomial(128 - v);
        if v <= 6 {
            g = g + monomial(7 - v);
        }
        if v == 1 {
            g = g + monomial(1);
        }
        *col = g;
    }
    cols
}

/// Evaluates `sum_v q_v * A(e_v)` with a Horner chain in `mul_x`.
/// Coefficients `q_1..q_127` appear in reverse monomial order; the final
/// seven additions implement the GHASH dual-basis corrections exactly.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn dual_basis_linear_combination(q: &[Gf; 128]) -> Gf {
    let mut acc = q[1];
    for &coefficient in &q[2..=121] {
        acc = acc.mul_x() + coefficient;
    }
    acc = acc.mul_x() + q[122] + q[1];
    acc = acc.mul_x() + q[123] + q[2];
    acc = acc.mul_x() + q[124] + q[3];
    acc = acc.mul_x() + q[125] + q[4];
    acc = acc.mul_x() + q[126] + q[5];
    acc = acc.mul_x() + q[127] + q[6] + q[1];
    acc.mul_x() + q[0]
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn splitmix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn sample(seed: u64) -> Gf {
        Gf::from_polynomial_words([splitmix(seed), splitmix(seed ^ 0xD1CE)])
    }

    /// `Φ_ρ(x) = Σ_i ρ_i·bit_i(x)` — the scalar reference (the production
    /// path uses `phi_byte_tables`/`phi_from_words`, pinned equal by
    /// `phi_byte_tables_match_bitscan` in `ligerito.rs`).
    fn phi(x: Gf, rho: &[Gf]) -> Gf {
        let w = x.as_words();
        let mut acc = Gf::zero();
        for i in 0..128usize {
            if (w[i >> 6] >> (i & 63)) & 1 == 1 {
                acc += rho[i];
            }
        }
        acc
    }

    /// Brute force: invert the pairing matrix `P_{ij} = c₀(X^i·X^j)` by
    /// F₂ Gaussian elimination — the unique `μ_H`-dual of the monomial
    /// basis — and assert it IS `dual_basis_cols()`.
    #[test]
    fn dual_basis_matches_brute_force_solve() {
        // Row i of the system: Σ_j P_{ij}·d_j = δ_{iv} per column v.
        // Store each row as a 128-bit set; augment with the identity.
        let mut rows = [[0u64; 2]; 128]; // P
        let mut aug = [[0u64; 2]; 128]; // running inverse
        for i in 0..128 {
            for j in 0..128usize {
                if c0_bit(monomial(i) * monomial(j)) == 1 {
                    rows[i][j >> 6] |= 1u64 << (j & 63);
                }
            }
            aug[i][i >> 6] |= 1u64 << (i & 63);
        }
        // Gauss–Jordan over F₂.
        for col in 0..128usize {
            let pivot = (col..128)
                .find(|&r| (rows[r][col >> 6] >> (col & 63)) & 1 == 1)
                .expect("pairing matrix is invertible");
            rows.swap(col, pivot);
            aug.swap(col, pivot);
            for r in 0..128 {
                if r != col && (rows[r][col >> 6] >> (col & 63)) & 1 == 1 {
                    for w in 0..2 {
                        rows[r][w] ^= rows[col][w];
                        aug[r][w] ^= aug[col][w];
                    }
                }
            }
        }
        // aug now holds P⁻¹ (rows); column v of P⁻¹ is the dual of X^v.
        let cols = dual_basis_cols();
        for v in 0..128usize {
            let mut d = [0u64; 2];
            for j in 0..128usize {
                if (aug[j][v >> 6] >> (v & 63)) & 1 == 1 {
                    d[j >> 6] |= 1u64 << (j & 63);
                }
            }
            assert_eq!(Gf::from_polynomial_words(d), cols[v], "dual column {v}");
        }
    }

    /// The defining property, checked directly: `c₀(X^u·A(e_v)) = δ_{uv}`.
    #[test]
    fn dual_basis_pairs_to_identity() {
        let cols = dual_basis_cols();
        for u in 0..128 {
            for (v, &col) in cols.iter().enumerate() {
                let expect = u64::from(u == v);
                assert_eq!(c0_bit(monomial(u) * col), expect, "pairing ({u}, {v})");
            }
        }
    }

    /// The theorem's Hankel form (`A₀₀ = 1`, `A_{ij} = −f_{i+j}/f₀` for
    /// `i, j ≥ 1`, `i+j ≤ d`) reproduces the columns, and the deviation
    /// from a pure `v ↦ 128−v` reversal is EXACTLY seven set bits.
    #[test]
    fn dual_basis_is_hankel_with_seven_corrections() {
        // f = X^128 + X^7 + X^2 + X + 1.
        let f_coeff = |k: usize| -> u64 { u64::from(matches!(k, 0 | 1 | 2 | 7 | 128)) };
        let cols = dual_basis_cols();
        let mut corrections = 0usize;
        for (v, &col) in cols.iter().enumerate() {
            let mut w = [0u64; 2];
            if v == 0 {
                w[0] |= 1; // A₀₀ = 1
            } else {
                for i in 1..=(128 - v) {
                    if f_coeff(i + v) == 1 {
                        w[i >> 6] |= 1u64 << (i & 63);
                    }
                }
            }
            assert_eq!(Gf::from_polynomial_words(w), col, "Hankel column {v}");
            if v >= 1 {
                let rev = monomial(128 - v);
                let diff = col + rev;
                corrections +=
                    (diff.as_words()[0].count_ones() + diff.as_words()[1].count_ones()) as usize;
            }
        }
        assert_eq!(
            corrections, 7,
            "reversal plus exactly seven XOR corrections"
        );
    }

    #[test]
    fn dual_basis_linear_combination_matches_dense_reference() {
        let cols = dual_basis_cols();
        for trial in 0..64u64 {
            let q: [Gf; 128] =
                core::array::from_fn(|v| sample(0xD000_0000 + (trial << 8) + v as u64));
            let expected = q
                .iter()
                .zip(cols.iter())
                .fold(Gf::zero(), |acc, (&value, &column)| acc + value * column);
            assert_eq!(dual_basis_linear_combination(&q), expected, "trial {trial}");
        }
    }

    /// The n = 1 embedding identity on random blocks:
    /// `⟨w, a⟩_{F₂} = c₀(pack(w)·Σ_v a_v·A(e_v))`.
    #[test]
    fn inner_product_identity_on_random_blocks() {
        let cols = dual_basis_cols();
        for t in 0..64u64 {
            let w = [splitmix(0xAA00 + t), splitmix(0xBB00 + t)];
            let a = [splitmix(0xCC00 + t), splitmix(0xDD00 + t)];
            let mut dot = 0u64;
            let mut packed_a = Gf::zero();
            for v in 0..128usize {
                let av = (a[v >> 6] >> (v & 63)) & 1;
                dot ^= av & ((w[v >> 6] >> (v & 63)) & 1);
                if av == 1 {
                    packed_a += cols[v];
                }
            }
            assert_eq!(
                c0_bit(Gf::from_polynomial_words(w) * packed_a),
                dot,
                "block {t}"
            );
        }
    }

    /// Plane decomposition over a tiny 2-pack shape:
    /// `⟨W, f⟩_K = Σ_i X^i·⟨W-plane-i, f⟩_{F₂}` (valid because `f`'s
    /// entries are bits and K-addition is coordinatewise F₂-addition).
    #[test]
    fn plane_decomposition_identity() {
        let n_cells = 256usize; // 2 packs
        let weights: Vec<Gf> = (0..n_cells).map(|j| sample(0x11_0000 + j as u64)).collect();
        let f_bit = |j: usize| splitmix(0x22_0000 + j as u64) & 1;

        let mut lhs = Gf::zero();
        for (j, &wj) in weights.iter().enumerate() {
            if f_bit(j) == 1 {
                lhs += wj;
            }
        }
        let mut rhs = Gf::zero();
        for i in 0..128usize {
            let mut plane_dot = 0u64;
            for (j, wj) in weights.iter().enumerate() {
                let w = wj.as_words();
                plane_dot ^= ((w[i >> 6] >> (i & 63)) & 1) & f_bit(j);
            }
            if plane_dot == 1 {
                rhs += monomial(i);
            }
        }
        assert_eq!(lhs, rhs);
    }

    /// The batching protocol on a tiny 2-pack shape: with
    /// `h_i = ⟨pack(f), A(a_i)⟩_K` (planes `a_i` of the weights),
    /// (a) step 3: `⟨W, f⟩ = Σ_i c₀(h_i)·X^i`, and (b) step 4 with the
    /// per-pack batched basis `a′(y) = Σ_v Φ_ρ(W_{(v,y)})·A(e_v)`:
    /// `Σ_i ρ_i·h_i = ⟨pack(f), a′⟩_K` for random `ρ`.
    #[test]
    fn batching_protocol_identities_tiny() {
        let cols = dual_basis_cols();
        let packs = 2usize;
        let weights: Vec<Gf> = (0..packs * 128)
            .map(|j| sample(0x33_0000 + j as u64))
            .collect();
        let f_bit = |j: usize| splitmix(0x44_0000 + j as u64) & 1;
        let pack = |y: usize| -> Gf {
            let mut w = [0u64; 2];
            for v in 0..128usize {
                w[v >> 6] |= (f_bit((y << 7) | v)) << (v & 63);
            }
            Gf::from_polynomial_words(w)
        };
        let rho: Vec<Gf> = (0..128).map(|i| sample(0x55_0000 + i as u64)).collect();

        // h_i = Σ_y P[y]·(Σ_v bit_i(W_{(v,y)})·A(e_v)).
        let mut hs = [Gf::zero(); 128];
        for y in 0..packs {
            let p = pack(y);
            for v in 0..128usize {
                let w = weights[(y << 7) | v].as_words();
                let g = p * cols[v];
                for (i, h) in hs.iter_mut().enumerate() {
                    if (w[i >> 6] >> (i & 63)) & 1 == 1 {
                        *h += g;
                    }
                }
            }
        }

        // (a) Step 3 against the direct ⟨W, f⟩.
        let mut h_direct = Gf::zero();
        for (j, &wj) in weights.iter().enumerate() {
            if f_bit(j) == 1 {
                h_direct += wj;
            }
        }
        let mut assembled = [0u64; 2];
        for (i, h) in hs.iter().enumerate() {
            assembled[i >> 6] |= c0_bit(*h) << (i & 63);
        }
        assert_eq!(Gf::from_polynomial_words(assembled), h_direct, "step 3");

        // (b) Step 4: ρ-batched target vs the per-pack batched basis.
        let mut lhs = Gf::zero();
        for (i, &h) in hs.iter().enumerate() {
            lhs += rho[i] * h;
        }
        let mut rhs = Gf::zero();
        for y in 0..packs {
            let mut a_prime = Gf::zero();
            for v in 0..128usize {
                a_prime += phi(weights[(y << 7) | v], &rho) * cols[v];
            }
            rhs += pack(y) * a_prime;
        }
        assert_eq!(lhs, rhs, "step 4 per-pack basis");
    }
}
