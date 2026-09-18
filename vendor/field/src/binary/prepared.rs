#[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
use crate::Gf128Ops;
use crate::{Gf128, RingOps};

/// A reusable multiplier. ARM retains the production interleaved preparation
/// so each multiplication uses five PMULL instructions, including reduction.
#[derive(Clone, Copy, Debug)]
pub struct PreparedGf128Mul {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    low: [u64; 2],
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    high: [u64; 2],
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    scalar: Gf128,
}
impl PreparedGf128Mul {
    pub fn new(scalar: Gf128) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            use crate::gf128::aarch64::{load, pmull_lo, store};
            let rg = store(pmull_lo(load([scalar.hi, 0]), load([0x87, 0])));
            Self {
                low: [scalar.lo, rg[0]],
                high: [scalar.hi, scalar.lo ^ rg[1]],
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            Self { scalar }
        }
    }
    #[inline]
    pub fn mul(&self, value: &Gf128) -> Gf128 {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            use crate::gf128::aarch64::{load, pmull_hi, pmull_lo, store};
            use core::arch::aarch64::*;
            // SAFETY: AES/NEON are enabled for this implementation. The scalar
            // preparation and all accesses have fixed public positions.
            unsafe {
                let a = load([value.lo, value.hi]);
                let low = load(self.low);
                let high = load(self.high);
                let tl = veorq_u64(pmull_lo(a, low), pmull_hi(a, low));
                let tm = veorq_u64(pmull_lo(a, high), pmull_hi(a, high));
                let z = vdupq_n_u64(0);
                let correction = pmull_lo(vextq_u64::<1>(tm, z), load([0x87, 0]));
                let result = store(veorq_u64(veorq_u64(tl, vextq_u64::<1>(z, tm)), correction));
                Gf128::new(result[0], result[1])
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            Gf128Ops.mul(value, &self.scalar)
        }
    }
    pub fn mul_into(&self, input: &[Gf128], out: &mut [Gf128]) {
        assert_eq!(input.len(), out.len(), "prepared multiply shape mismatch");
        for (value, dst) in input.iter().zip(out) {
            *dst = self.mul(value);
        }
    }
}

impl crate::batch::PreparedRoundMul for crate::Gf128Ops {
    type Multiplier = PreparedGf128Mul;
    fn prepare_round_mul(&self, value: &Gf128) -> Self::Multiplier {
        PreparedGf128Mul::new(*value)
    }
    fn mul_round_prepared(&self, multiplier: &Self::Multiplier, value: &Gf128) -> Gf128 {
        multiplier.mul(value)
    }
}
macro_rules! scalar_round_mul {
    ($ctx:ty) => {
        impl crate::batch::PreparedRoundMul for $ctx {
            type Multiplier = Self::Elem;
            fn prepare_round_mul(&self, value: &Self::Elem) -> Self::Multiplier {
                *value
            }
            fn mul_round_prepared(
                &self,
                multiplier: &Self::Multiplier,
                value: &Self::Elem,
            ) -> Self::Elem {
                self.mul(multiplier, value)
            }
        }
    };
}
scalar_round_mul!(crate::F2Ops);
scalar_round_mul!(crate::Gf8Ops);
scalar_round_mul!(crate::B127Ops);

/// XOR-sum of products with prepared GF128 multipliers. ARM retains the
/// interleaved 191-bit domain, allowing one final fold for the entire sum.
#[derive(Clone, Copy)]
pub struct Gf128PreparedAcc {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    words: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    words: crate::Gf128Product,
}
impl Gf128PreparedAcc {
    #[inline(always)]
    pub fn zero() -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            let z = crate::gf128::aarch64::load([0, 0]);
            Self { words: (z, z) }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            Self {
                words: crate::Gf128Product::zero(),
            }
        }
    }
    #[inline(always)]
    pub fn add_mul(&mut self, value: &Gf128, fixed: &PreparedGf128Mul) {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            use crate::gf128::aarch64::{load, pmull_hi, pmull_lo};
            use core::arch::aarch64::veorq_u64;
            // SAFETY: all required AES/NEON instructions are enabled.
            unsafe {
                let a = load([value.lo, value.hi]);
                let rl = load(fixed.low);
                let rh = load(fixed.high);
                self.words.0 = veorq_u64(self.words.0, veorq_u64(pmull_lo(a, rl), pmull_hi(a, rl)));
                self.words.1 = veorq_u64(self.words.1, veorq_u64(pmull_lo(a, rh), pmull_hi(a, rh)));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.words ^= value.mul_unreduced(fixed.scalar);
        }
    }
    #[inline(always)]
    pub fn reduce(self) -> Gf128 {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            use crate::gf128::aarch64::{load, pmull_hi, store};
            use core::arch::aarch64::{veorq_u64, vextq_u64};
            // SAFETY: public fixed-shape vector operations under the AES gate.
            unsafe {
                let r = veorq_u64(
                    self.words.0,
                    veorq_u64(
                        vextq_u64::<1>(load([0, 0]), self.words.1),
                        pmull_hi(self.words.1, load([0x87, 0x87])),
                    ),
                );
                Gf128::from(store(r))
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.words.reduce()
        }
    }
}
impl crate::MergeAccumulator for Gf128PreparedAcc {
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn merge_assign(&mut self, rhs: &Self) {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        // SAFETY: AES implies the NEON support used by this accumulator.
        unsafe {
            self.words.0 = core::arch::aarch64::veorq_u64(self.words.0, rhs.words.0);
            self.words.1 = core::arch::aarch64::veorq_u64(self.words.1, rhs.words.1);
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.words ^= rhs.words;
        }
    }
}
impl crate::BatchMulAcc<Gf128, PreparedGf128Mul> for crate::Gf128Ops {
    type Accumulator = Gf128PreparedAcc;
    #[inline(always)]
    fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &Gf128, rhs: &PreparedGf128Mul) {
        acc.add_mul(lhs, rhs);
    }
    fn batch_mul_acc(&self, lhs: &[Gf128], rhs: &[PreparedGf128Mul]) -> Self::Accumulator {
        assert_eq!(lhs.len(), rhs.len(), "prepared MAC shape mismatch");
        self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
    }
    fn batch_mul_acc_map(
        &self,
        len: usize,
        mut term: impl FnMut(usize) -> (Gf128, PreparedGf128Mul),
    ) -> Self::Accumulator {
        let mut acc = Gf128PreparedAcc::zero();
        for i in 0..len {
            let (a, b) = term(i);
            acc.add_mul(&a, &b);
        }
        acc
    }
}
impl crate::Reduce<Gf128PreparedAcc> for crate::Gf128Ops {
    type Output = Gf128;
    fn reduce(&self, acc: Gf128PreparedAcc) -> Gf128 {
        acc.reduce()
    }
}
