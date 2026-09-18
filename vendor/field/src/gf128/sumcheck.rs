//! Fused binary-field arithmetic. Protocol drivers own traversal and parallelism.
use super::kernels::{clmul_128x128, reduce_256_to_128};
use crate::{Gf128, Gf128Ops, Gf128Product, SumcheckKernels};
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[path = "kernels/sumcheck_aarch64.rs"]
mod neon;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "pclmulqdq",
    target_feature = "sse4.1"
))]
#[path = "kernels/sumcheck_x86_64.rs"]
mod x86;
fn reduce_wide(words: [u64; 4]) -> Gf128 {
    Gf128Product::from_polynomial_words(words).reduce()
}
impl Gf128 {
    /// Hand-fused round body over raw words: two INDEPENDENT slot chains
    /// per iteration with their own register-resident 256-bit accumulator
    /// sets, so the PMULL pipes stay fed instead of serialising one slot's
    /// `w·l → (w·l)·r → accumulate` latency chain. Value-exact: the same
    /// carryless products, XOR-combined (commutative), reduced once per
    /// accumulator at the end — identical field elements to the generic
    /// loop.
    #[allow(clippy::arithmetic_side_effects)]
    fn bitz_eqf_single_pair_round(
        l: &[Self],
        r: &[Self],
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        // NEON-resident pipeline (schoolbook PMULL products + PMULL-fold
        // reduction, accumulators in vector registers — no NEON↔GPR
        // bounces). Value-exact vs the word pipeline below.
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            return Some(neon::eqf_single_pair_round(l, r, w, half));
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            #[inline(always)]
            fn slot(
                w: &Gf128,
                l0: &Gf128,
                l1: &Gf128,
                r0: &Gf128,
                r1: &Gf128,
                a0: &mut [u64; 4],
                a1: &mut [u64; 4],
                a2: &mut [u64; 4],
            ) {
                let ww = w.as_words();
                let l0w = reduce_256_to_128(clmul_128x128(ww, l0.as_words()));
                let l1w = reduce_256_to_128(clmul_128x128(ww, l1.as_words()));
                let r0w = r0.as_words();
                let r1w = r1.as_words();
                let wc0 = clmul_128x128(&l0w, r0w);
                let w11 = clmul_128x128(&l1w, r1w);
                let dl = [l1w[0] ^ l0w[0], l1w[1] ^ l0w[1]];
                let dr = [r1w[0] ^ r0w[0], r1w[1] ^ r0w[1]];
                let wc2 = clmul_128x128(&dl, &dr);
                let mut i = 0;
                while i < 4 {
                    a0[i] ^= wc0[i];
                    a2[i] ^= wc2[i];
                    a1[i] ^= w11[i] ^ wc0[i] ^ wc2[i];
                    i += 1;
                }
            }
            let (mut a0a, mut a1a, mut a2a) = ([0u64; 4], [0u64; 4], [0u64; 4]);
            let (mut a0b, mut a1b, mut a2b) = ([0u64; 4], [0u64; 4], [0u64; 4]);
            let mut b = 0usize;
            while b + 2 <= half {
                slot(
                    &w[b],
                    &l[b << 1],
                    &l[(b << 1) | 1],
                    &r[b << 1],
                    &r[(b << 1) | 1],
                    &mut a0a,
                    &mut a1a,
                    &mut a2a,
                );
                let c = b + 1;
                slot(
                    &w[c],
                    &l[c << 1],
                    &l[(c << 1) | 1],
                    &r[c << 1],
                    &r[(c << 1) | 1],
                    &mut a0b,
                    &mut a1b,
                    &mut a2b,
                );
                b += 2;
            }
            if b < half {
                slot(
                    &w[b],
                    &l[b << 1],
                    &l[(b << 1) | 1],
                    &r[b << 1],
                    &r[(b << 1) | 1],
                    &mut a0a,
                    &mut a1a,
                    &mut a2a,
                );
            }
            let mut i = 0;
            while i < 4 {
                a0a[i] ^= a0b[i];
                a1a[i] ^= a1b[i];
                a2a[i] ^= a2b[i];
                i += 1;
            }
            Some((reduce_wide(a0a), reduce_wide(a1a), reduce_wide(a2a)))
        }
    }

    /// Hand-fused TWO-pair round body (the fraction-GKR layer combine):
    /// per slot the two pairs are the two INDEPENDENT PMULL chains — each
    /// with its own register-resident 256-bit accumulator set — and the
    /// shared weight folds into each pair's `L` side up front, so no
    /// per-slot reduction is ever taken (the generic multi-pair loop
    /// reduces its pair-sums once per slot to apply the weight).
    /// Value-exact: `w·(A + B) = w·A + w·B`, the same carryless products
    /// XOR-combined, reduced once per accumulator at the end.
    #[allow(clippy::arithmetic_side_effects)]
    fn bitz_eqf_two_pair_round(
        l0: &[Self],
        r0: &[Self],
        l1: &[Self],
        r1: &[Self],
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        // NEON-resident pipeline; value-exact vs the word pipeline below.
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            return Some(neon::eqf_two_pair_round(l0, r0, l1, r1, w, half));
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            #[inline(always)]
            fn slot(
                w: &Gf128,
                l0: &Gf128,
                l1: &Gf128,
                r0: &Gf128,
                r1: &Gf128,
                a0: &mut [u64; 4],
                a1: &mut [u64; 4],
                a2: &mut [u64; 4],
            ) {
                let ww = w.as_words();
                let l0w = reduce_256_to_128(clmul_128x128(ww, l0.as_words()));
                let l1w = reduce_256_to_128(clmul_128x128(ww, l1.as_words()));
                let r0w = r0.as_words();
                let r1w = r1.as_words();
                let wc0 = clmul_128x128(&l0w, r0w);
                let w11 = clmul_128x128(&l1w, r1w);
                let dl = [l1w[0] ^ l0w[0], l1w[1] ^ l0w[1]];
                let dr = [r1w[0] ^ r0w[0], r1w[1] ^ r0w[1]];
                let wc2 = clmul_128x128(&dl, &dr);
                let mut i = 0;
                while i < 4 {
                    a0[i] ^= wc0[i];
                    a2[i] ^= wc2[i];
                    a1[i] ^= w11[i] ^ wc0[i] ^ wc2[i];
                    i += 1;
                }
            }
            let (mut a0a, mut a1a, mut a2a) = ([0u64; 4], [0u64; 4], [0u64; 4]);
            let (mut a0b, mut a1b, mut a2b) = ([0u64; 4], [0u64; 4], [0u64; 4]);
            let mut b = 0usize;
            while b < half {
                let e = b << 1;
                slot(
                    &w[b],
                    &l0[e],
                    &l0[e | 1],
                    &r0[e],
                    &r0[e | 1],
                    &mut a0a,
                    &mut a1a,
                    &mut a2a,
                );
                slot(
                    &w[b],
                    &l1[e],
                    &l1[e | 1],
                    &r1[e],
                    &r1[e | 1],
                    &mut a0b,
                    &mut a1b,
                    &mut a2b,
                );
                b += 1;
            }
            let mut i = 0;
            while i < 4 {
                a0a[i] ^= a0b[i];
                a1a[i] ^= a1b[i];
                a2a[i] ^= a2b[i];
                i += 1;
            }
            Some((reduce_wide(a0a), reduce_wide(a1a), reduce_wide(a2a)))
        }
    }

    /// Fused in-place fold `v[b] ← v[2b] ⊕ ρ·(v[2b+1] ⊕ v[2b])`, two
    /// independent entries per iteration (loads complete before the pair's
    /// stores, and writes at `b, b+1` never overtake the reads at
    /// `2b..2b+4`). Value-exact per entry.
    #[allow(clippy::arithmetic_side_effects)]
    fn bitz_eqf_fold_in_place(v: &mut [Self], rho: &Self, half: usize) -> bool {
        // NEON-resident pipeline; value-exact vs the word pipeline below.
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            if true {
                neon::eqf_fold_in_place_fixed(v, rho, half);
            } else {
                neon::eqf_fold_in_place(v, rho, half);
            }
            return true;
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            let rw = *rho.as_words();
            let mut b = 0usize;
            while b + 2 <= half {
                let v0a = *v[b << 1].as_words();
                let v1a = *v[(b << 1) | 1].as_words();
                let v0b = *v[(b + 1) << 1].as_words();
                let v1b = *v[((b + 1) << 1) | 1].as_words();
                let da = [v1a[0] ^ v0a[0], v1a[1] ^ v0a[1]];
                let db = [v1b[0] ^ v0b[0], v1b[1] ^ v0b[1]];
                let pa = reduce_256_to_128(clmul_128x128(&rw, &da));
                let pb = reduce_256_to_128(clmul_128x128(&rw, &db));
                v[b] = Gf128::from([v0a[0] ^ pa[0], v0a[1] ^ pa[1]]);
                v[b + 1] = Gf128::from([v0b[0] ^ pb[0], v0b[1] ^ pb[1]]);
                b += 2;
            }
            if b < half {
                let v0 = *v[b << 1].as_words();
                let v1 = *v[(b << 1) | 1].as_words();
                let d = [v1[0] ^ v0[0], v1[1] ^ v0[1]];
                let p = reduce_256_to_128(clmul_128x128(&rw, &d));
                v[b] = Gf128::from([v0[0] ^ p[0], v0[1] ^ p[1]]);
            }
            true
        }
    }

    /// Hand-fused deferred-fold + round body (the pass-fusion path): per
    /// slot the four fold products are independent PMULL chains, the
    /// folded entries store to the buffer prefix on the way, and the
    /// message chain XORs its three products into 256-bit accumulators —
    /// one pass over the unfolded buffers instead of a fold pass plus a
    /// message pass. Value-exact: the folded entries are the exact
    /// `v0 ⊕ ρ·(v1 ⊕ v0)` reduced values [`Self::eqf_fold_in_place`]
    /// writes, and the message products are the single-pair body's,
    /// XOR-combined and reduced once per accumulator at the end.
    #[allow(clippy::arithmetic_side_effects)]
    fn bitz_eqf_fused_fold_round(
        l: &mut [Self],
        r: &mut [Self],
        rho: &Self,
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        // NEON-resident pipeline; value-exact vs the word pipeline below.
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            if true {
                return Some(neon::eqf_fused_fold_round_fixed(l, r, rho, w, half));
            }
            return Some(neon::eqf_fused_fold_round(l, r, rho, w, half));
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            let rw = *rho.as_words();
            let fold1 = |v0: [u64; 2], v1: [u64; 2]| -> [u64; 2] {
                let d = [v1[0] ^ v0[0], v1[1] ^ v0[1]];
                let p = reduce_256_to_128(clmul_128x128(&rw, &d));
                [v0[0] ^ p[0], v0[1] ^ p[1]]
            };
            let (mut a0a, mut a1a, mut a2a) = ([0u64; 4], [0u64; 4], [0u64; 4]);
            for b in 0..half {
                let base = b << 2;
                let fl0 = fold1(*l[base].as_words(), *l[base + 1].as_words());
                let fl1 = fold1(*l[base + 2].as_words(), *l[base + 3].as_words());
                let fr0 = fold1(*r[base].as_words(), *r[base + 1].as_words());
                let fr1 = fold1(*r[base + 2].as_words(), *r[base + 3].as_words());
                let e = b << 1;
                l[e] = Gf128::from(fl0);
                l[e | 1] = Gf128::from(fl1);
                r[e] = Gf128::from(fr0);
                r[e | 1] = Gf128::from(fr1);
                let ww = w[b].as_words();
                let l0w = reduce_256_to_128(clmul_128x128(ww, &fl0));
                let l1w = reduce_256_to_128(clmul_128x128(ww, &fl1));
                let wc0 = clmul_128x128(&l0w, &fr0);
                let w11 = clmul_128x128(&l1w, &fr1);
                let dl = [l1w[0] ^ l0w[0], l1w[1] ^ l0w[1]];
                let dr = [fr1[0] ^ fr0[0], fr1[1] ^ fr0[1]];
                let wc2 = clmul_128x128(&dl, &dr);
                let mut i = 0;
                while i < 4 {
                    a0a[i] ^= wc0[i];
                    a2a[i] ^= wc2[i];
                    a1a[i] ^= w11[i] ^ wc0[i] ^ wc2[i];
                    i += 1;
                }
            }
            Some((reduce_wide(a0a), reduce_wide(a1a), reduce_wide(a2a)))
        }
    }

    /// Hand kernel for the double-fold dense grid pass: the deferred
    /// challenges fold as ONE arity-4 fixed-weight XOR-sum with a shared
    /// reduction (weights `ρ₁, ρ₂, ρ₁ρ₂` preprocessed — 12+1 PMULLs per
    /// logical value vs 3 composed multiplies' 21), and the weighting,
    /// node grids, and nine wide products stay vector-resident per quad.
    /// Value-exact vs the generic pass (pinned by
    /// `grid_kernel_matches_generic_pass`).
    fn bitz_eqf_grid_pass(
        l: &mut [Self],
        r: &mut [Self],
        pending: &[Self],
        suffix: &[Self],
        quads: usize,
    ) -> Option<[Self; 9]> {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            if pending.len() <= 2 && true {
                return Some(neon::eqf_grid_pass(l, r, pending, suffix, quads));
            }
            None
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "pclmulqdq",
                target_feature = "sse4.1"
            ))]
            if pending.len() <= 2 {
                return Some(x86::grid_pass(l, r, pending, suffix, quads));
            }
            let _ = (l, r, pending, suffix, quads);
            None
        }
    }
}

fn pair(l: usize, r: usize, w: usize, n: usize, arity: usize) {
    let needed = n.checked_mul(arity).expect("sumcheck shape overflow");
    assert!(
        l >= needed && r >= needed && w >= n,
        "insufficient sumcheck input"
    );
}
fn array(v: (Gf128, Gf128, Gf128)) -> [Gf128; 3] {
    [v.0, v.1, v.2]
}
impl SumcheckKernels for Gf128Ops {
    fn eqf_single_pair_round(&self, l: &[Gf128], r: &[Gf128], w: &[Gf128], n: usize) -> [Gf128; 3] {
        pair(l.len(), r.len(), w.len(), n, 2);
        array(Gf128::bitz_eqf_single_pair_round(l, r, w, n).unwrap())
    }
    fn eqf_two_pair_round(
        &self,
        l0: &[Gf128],
        r0: &[Gf128],
        l1: &[Gf128],
        r1: &[Gf128],
        w: &[Gf128],
        n: usize,
    ) -> [Gf128; 3] {
        pair(l0.len(), r0.len(), w.len(), n, 2);
        pair(l1.len(), r1.len(), w.len(), n, 2);
        array(Gf128::bitz_eqf_two_pair_round(l0, r0, l1, r1, w, n).unwrap())
    }
    fn eqf_fold_in_place(&self, v: &mut [Gf128], rho: &Gf128, n: usize) {
        pair(v.len(), v.len(), n, n, 2);
        Gf128::bitz_eqf_fold_in_place(v, rho, n);
    }
    fn eqf_fused_fold_round(
        &self,
        l: &mut [Gf128],
        r: &mut [Gf128],
        rho: &Gf128,
        w: &[Gf128],
        n: usize,
    ) -> [Gf128; 3] {
        pair(l.len(), r.len(), w.len(), n, 4);
        array(Gf128::bitz_eqf_fused_fold_round(l, r, rho, w, n).unwrap())
    }
    fn eqf_grid_pass(
        &self,
        l: &mut [Gf128],
        r: &mut [Gf128],
        p: &[Gf128],
        s: &[Gf128],
        n: usize,
    ) -> [Gf128; 9] {
        assert!(p.len() <= 2, "at most two deferred folds");
        pair(l.len(), r.len(), s.len(), n, 4 << p.len());
        Gf128::bitz_eqf_grid_pass(l, r, p, s, n)
            .unwrap_or_else(|| crate::batch::grid_pass(self, l, r, p, s, n))
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "aes"))]
mod tests {
    use super::*;
    fn rand_elt(state: &mut u64) -> Gf128 {
        fn word(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        Gf128::new(word(state), word(state))
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fixed_scalar_kernels_match_composed() {
        let mut rng = 0xF15Eu64;
        let ones = Gf128::from([u64::MAX, u64::MAX]);
        for half in [1usize, 2, 37, 64] {
            let mut rhos = vec![Gf128::ZERO, Gf128::ONE, ones, rand_elt(&mut rng)];
            for rho in rhos.drain(..) {
                // In-place fold.
                let n = half << 1;
                let v0: Vec<_> = (0..n).map(|_| rand_elt(&mut rng)).collect();
                let mut a = v0.clone();
                let mut b = v0.clone();
                neon::eqf_fold_in_place(&mut a, &rho, half);
                neon::eqf_fold_in_place_fixed(&mut b, &rho, half);
                assert_eq!(&a[..half], &b[..half], "fold (half = {half})");

                // Fused fold+round pass.
                let n = half << 2;
                let l0: Vec<_> = (0..n).map(|_| rand_elt(&mut rng)).collect();
                let r0: Vec<_> = (0..n).map(|_| rand_elt(&mut rng)).collect();
                let w: Vec<_> = (0..half).map(|_| rand_elt(&mut rng)).collect();
                let (mut la, mut ra) = (l0.clone(), r0.clone());
                let (mut lb, mut rb) = (l0.clone(), r0.clone());
                let ca = neon::eqf_fused_fold_round(&mut la, &mut ra, &rho, &w, half);
                let cb = neon::eqf_fused_fold_round_fixed(&mut lb, &mut rb, &rho, &w, half);
                assert_eq!(ca, cb, "fused coefficients (half = {half})");
                assert_eq!(
                    &la[..n >> 1],
                    &lb[..n >> 1],
                    "fused folded L (half = {half})"
                );
                assert_eq!(
                    &ra[..n >> 1],
                    &rb[..n >> 1],
                    "fused folded R (half = {half})"
                );

                // fold(0, a) = ρ·a pins the fixed multiply vs `Mul`.
                let a_s = rand_elt(&mut rng);
                let mut buf = vec![Gf128::ZERO, a_s];
                neon::eqf_fold_in_place_fixed(&mut buf, &rho, 1);
                assert_eq!(buf[0], a_s * &rho, "fixed multiply");
            }
        }
    }
}
