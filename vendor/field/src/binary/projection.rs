//! Polynomial projection with fixed public bounds and four independent lanes.
use crate::{Gf128, Gf128Ops, RingOps};

/// Powers of a public point, reused across private polynomial inputs.
#[derive(Clone, Debug)]
pub struct PreparedGf128Projection {
    powers: Vec<Gf128>,
}
impl PreparedGf128Projection {
    pub fn new(point: Gf128, bits: usize) -> Self {
        assert!(bits <= 64, "packed projection is limited to one word");
        let mut powers = Vec::with_capacity(bits);
        let mut power = Gf128::ONE;
        for _ in 0..bits {
            powers.push(power);
            power = Gf128Ops.mul(&power, &point);
        }
        Self { powers }
    }
    pub fn project(&self, bits: u64) -> Gf128 {
        let mut out = [0u64; 2];
        for (i, power) in self.powers.iter().enumerate() {
            let mask = 0u64.wrapping_sub((bits >> i) & 1);
            out[0] ^= power.lo & mask;
            out[1] ^= power.hi & mask;
        }
        Gf128::from_polynomial_words(out)
    }
    /// All buffers have equal length; every input uses exactly `D` low bits.
    pub fn project_into<const D: usize>(&self, input: &[u64], out: &mut [Gf128]) {
        assert_eq!(self.powers.len(), D, "projection width mismatch");
        assert_eq!(input.len(), out.len(), "projection shape mismatch");
        let mut chunks = input.chunks_exact(4);
        let mut output = out.chunks_exact_mut(4);
        for (cells, dst) in (&mut chunks).zip(&mut output) {
            dst.copy_from_slice(&project_four::<D>(cells.try_into().unwrap(), &self.powers));
        }
        for (&bits, dst) in chunks.remainder().iter().zip(output.into_remainder()) {
            *dst = self.project(bits);
        }
    }
}

/// Retained ARM NEON four-lane kernel and scalar fallback.
#[inline]
fn project_four<const D: usize>(cells: [u64; 4], alpha_powers: &[Gf128]) -> [Gf128; 4] {
    debug_assert!(
        alpha_powers.len() >= D,
        "project_four: powers slice ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    debug_assert!(D <= 64, "project_four: D ({D}) must be ≤ 64");

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::{vandq_u64, vdupq_n_u64, veorq_u64, vld1q_u64, vst1q_u64};
        // SAFETY: NEON enabled at compile time; we hold the four
        // accumulators in independent registers and only ever read 16
        // bytes from a `Gf128` (which is `#[repr(transparent)]`
        // over `Uint<2>`, i.e. `[u64; 2]`).
        unsafe {
            let mut a0 = vdupq_n_u64(0);
            let mut a1 = vdupq_n_u64(0);
            let mut a2 = vdupq_n_u64(0);
            let mut a3 = vdupq_n_u64(0);

            for i in 0..D {
                let pw = vld1q_u64(alpha_powers[i].as_words().as_ptr());

                let m0 = vdupq_n_u64(0u64.wrapping_sub((cells[0] >> i) & 1));
                let m1 = vdupq_n_u64(0u64.wrapping_sub((cells[1] >> i) & 1));
                let m2 = vdupq_n_u64(0u64.wrapping_sub((cells[2] >> i) & 1));
                let m3 = vdupq_n_u64(0u64.wrapping_sub((cells[3] >> i) & 1));

                a0 = veorq_u64(a0, vandq_u64(pw, m0));
                a1 = veorq_u64(a1, vandq_u64(pw, m1));
                a2 = veorq_u64(a2, vandq_u64(pw, m2));
                a3 = veorq_u64(a3, vandq_u64(pw, m3));
            }

            let mut w0 = [0u64; 2];
            let mut w1 = [0u64; 2];
            let mut w2 = [0u64; 2];
            let mut w3 = [0u64; 2];
            vst1q_u64(w0.as_mut_ptr(), a0);
            vst1q_u64(w1.as_mut_ptr(), a1);
            vst1q_u64(w2.as_mut_ptr(), a2);
            vst1q_u64(w3.as_mut_ptr(), a3);

            [
                Gf128::from_polynomial_words(w0),
                Gf128::from_polynomial_words(w1),
                Gf128::from_polynomial_words(w2),
                Gf128::from_polynomial_words(w3),
            ]
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        // Portable scalar fallback — same algorithm, 4 cells in
        // parallel sharing the α^i load, but with regular u64 XOR/AND
        // instead of NEON intrinsics. Loses the NEON dispatch advantage
        // but still amortises the α load across 4 cells.
        let mut a0 = [0u64; 2];
        let mut a1 = [0u64; 2];
        let mut a2 = [0u64; 2];
        let mut a3 = [0u64; 2];
        for i in 0..D {
            let pw = alpha_powers[i].as_words();
            let m0 = 0u64.wrapping_sub((cells[0] >> i) & 1);
            let m1 = 0u64.wrapping_sub((cells[1] >> i) & 1);
            let m2 = 0u64.wrapping_sub((cells[2] >> i) & 1);
            let m3 = 0u64.wrapping_sub((cells[3] >> i) & 1);
            a0[0] ^= pw[0] & m0;
            a0[1] ^= pw[1] & m0;
            a1[0] ^= pw[0] & m1;
            a1[1] ^= pw[1] & m1;
            a2[0] ^= pw[0] & m2;
            a2[1] ^= pw[1] & m2;
            a3[0] ^= pw[0] & m3;
            a3[1] ^= pw[1] & m3;
        }
        [
            Gf128::from_polynomial_words(a0),
            Gf128::from_polynomial_words(a1),
            Gf128::from_polynomial_words(a2),
            Gf128::from_polynomial_words(a3),
        ]
    }
}
