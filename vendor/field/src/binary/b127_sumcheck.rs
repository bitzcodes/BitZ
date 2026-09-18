//! B127 fused arithmetic retained from the root implementation.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use super::b127_aarch64 as neon;
use crate::gf128::kernels::clmul_128x128;
use crate::{B127, B127Ops, SumcheckKernels};
fn reduce_256_to_127(p: [u64; 4]) -> [u64; 2] {
    *super::reduce_b127(p).as_words()
}
fn reduce_wide(p: [u64; 4]) -> B127 {
    super::reduce_b127(p)
}
impl B127 {
    /// Hand-fused round body — the GF128 kernel with the trinomial drain:
    /// two independent slot chains, register-resident 256-bit accumulators,
    /// one (cheap, PMULL-free) reduction per accumulator at the end.
    /// Value-exact vs the generic loop (same carryless products,
    /// XOR-combined; reduction is `F_2`-linear).
    #[allow(clippy::arithmetic_side_effects)]
    fn bitz_eqf_single_pair_round(
        l: &[Self],
        r: &[Self],
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            return Some(neon::eqf_single_pair_round(l, r, w, half));
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            #[inline(always)]
            fn slot(
                w: &B127,
                l0: &B127,
                l1: &B127,
                r0: &B127,
                r1: &B127,
                a0: &mut [u64; 4],
                a1: &mut [u64; 4],
                a2: &mut [u64; 4],
            ) {
                let ww = w.as_words();
                let l0w = reduce_256_to_127(clmul_128x128(ww, l0.as_words()));
                let l1w = reduce_256_to_127(clmul_128x128(ww, l1.as_words()));
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

    /// Hand-fused TWO-pair round body (the fraction-GKR layer combine) —
    /// the GF128 kernel with the trinomial drain. Value-exact.
    #[allow(clippy::arithmetic_side_effects)]
    fn bitz_eqf_two_pair_round(
        l0: &[Self],
        r0: &[Self],
        l1: &[Self],
        r1: &[Self],
        w: &[Self],
        half: usize,
    ) -> Option<(Self, Self, Self)> {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            return Some(neon::eqf_two_pair_round(l0, r0, l1, r1, w, half));
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            #[inline(always)]
            fn slot(
                w: &B127,
                l0: &B127,
                l1: &B127,
                r0: &B127,
                r1: &B127,
                a0: &mut [u64; 4],
                a1: &mut [u64; 4],
                a2: &mut [u64; 4],
            ) {
                let ww = w.as_words();
                let l0w = reduce_256_to_127(clmul_128x128(ww, l0.as_words()));
                let l1w = reduce_256_to_127(clmul_128x128(ww, l1.as_words()));
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
    /// independent entries per iteration. Value-exact per entry.
    #[allow(clippy::arithmetic_side_effects)]
    fn bitz_eqf_fold_in_place(v: &mut [Self], rho: &Self, half: usize) -> bool {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            neon::eqf_fold_in_place(v, rho, half);
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
                let pa = reduce_256_to_127(clmul_128x128(&rw, &da));
                let pb = reduce_256_to_127(clmul_128x128(&rw, &db));
                v[b] = B127::from_canonical_words([v0a[0] ^ pa[0], v0a[1] ^ pa[1]]);
                v[b + 1] = B127::from_canonical_words([v0b[0] ^ pb[0], v0b[1] ^ pb[1]]);
                b += 2;
            }
            if b < half {
                let v0 = *v[b << 1].as_words();
                let v1 = *v[(b << 1) | 1].as_words();
                let d = [v1[0] ^ v0[0], v1[1] ^ v0[1]];
                let p = reduce_256_to_127(clmul_128x128(&rw, &d));
                v[b] = B127::from_canonical_words([v0[0] ^ p[0], v0[1] ^ p[1]]);
            }
            true
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
fn array(v: (B127, B127, B127)) -> [B127; 3] {
    [v.0, v.1, v.2]
}
impl SumcheckKernels for B127Ops {
    fn eqf_single_pair_round(&self, l: &[B127], r: &[B127], w: &[B127], n: usize) -> [B127; 3] {
        pair(l.len(), r.len(), w.len(), n, 2);
        array(B127::bitz_eqf_single_pair_round(l, r, w, n).unwrap())
    }
    fn eqf_two_pair_round(
        &self,
        l0: &[B127],
        r0: &[B127],
        l1: &[B127],
        r1: &[B127],
        w: &[B127],
        n: usize,
    ) -> [B127; 3] {
        pair(l0.len(), r0.len(), w.len(), n, 2);
        pair(l1.len(), r1.len(), w.len(), n, 2);
        array(B127::bitz_eqf_two_pair_round(l0, r0, l1, r1, w, n).unwrap())
    }
    fn eqf_fold_in_place(&self, v: &mut [B127], rho: &B127, n: usize) {
        pair(v.len(), v.len(), n, n, 2);
        B127::bitz_eqf_fold_in_place(v, rho, n);
    }
    fn eqf_fused_fold_round(
        &self,
        l: &mut [B127],
        r: &mut [B127],
        rho: &B127,
        w: &[B127],
        n: usize,
    ) -> [B127; 3] {
        crate::batch::fused(self, l, r, rho, w, n)
    }
    fn eqf_grid_pass(
        &self,
        l: &mut [B127],
        r: &mut [B127],
        p: &[B127],
        s: &[B127],
        n: usize,
    ) -> [B127; 9] {
        crate::batch::grid_pass(self, l, r, p, s, n)
    }
}
