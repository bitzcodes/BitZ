use super::*;
use crate::integer::product::Words;

/// An exact bounded polynomial over F2, not a field element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F2Poly<const BITS: usize, const WORDS: usize> {
    words: Uint<WORDS>,
}
impl<const BITS: usize, const WORDS: usize> F2Poly<BITS, WORDS> {
    const MASK: u64 = {
        assert!(BITS > 0 && WORDS == BITS.div_ceil(64));
        if BITS % 64 == 0 {
            u64::MAX
        } else {
            (1u64 << (BITS % 64)) - 1
        }
    };
    pub const ZERO: Self = {
        let _ = Self::MASK;
        Self { words: Uint::ZERO }
    };
    pub fn from_words_ct(words: [u64; WORDS]) -> CtValue<Self> {
        let valid = (words[WORDS - 1] & !Self::MASK).ct_is_zero();
        CtValue::new(
            Self {
                words: Uint::ct_select(&Uint::ZERO, &Uint::from_words(words), valid),
            },
            valid,
        )
    }
    /// Trusted polynomial words; padding above BITS must be zero.
    pub const fn from_polynomial_words(words: [u64; WORDS]) -> Self {
        debug_assert!(
            words[WORDS - 1] & !Self::MASK == 0,
            "polynomial padding must be zero"
        );
        Self {
            words: Uint::from_words(words),
        }
    }
    pub fn bit(&self, public_index: usize) -> Bit {
        assert!(public_index < BITS, "polynomial bit index out of range");
        self.words.bit(public_index)
    }
    pub fn set_bit(&mut self, public_index: usize, value: Bit) {
        assert!(public_index < BITS, "polynomial bit index out of range");
        let mask = 1u64 << (public_index % 64);
        let word = &mut self.words.0[public_index / 64];
        *word = (*word & !mask) | (u64::ct_select(&0, &mask, value.mask()));
    }
    pub fn as_words(&self) -> &[u64; WORDS] {
        self.words.as_words()
    }
    pub fn xor(&self, rhs: &Self) -> Self {
        Self {
            words: self.words ^ rhs.words,
        }
    }
}
impl<const BITS: usize, const WORDS: usize> CtEq for F2Poly<BITS, WORDS> {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        self.words.ct_eq(&rhs.words)
    }
    fn ct_is_zero(&self) -> CtMask {
        self.words.ct_is_zero()
    }
}
impl<const BITS: usize, const WORDS: usize> CtSelect for F2Poly<BITS, WORDS> {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self {
            words: Uint::ct_select(&a.words, &b.words, mask),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct F2PolyProduct<const A: usize, const B: usize> {
    words: UintProduct<A, B>,
}
impl<const A: usize, const B: usize> F2PolyProduct<A, B> {
    pub fn as_parts(&self) -> (&[u64; A], &[u64; B]) {
        self.words.as_parts()
    }
}
impl<const A: usize, const B: usize> XorPayload for F2PolyProduct<A, B> {
    fn zero() -> Self {
        Self {
            words: UintProduct::ZERO,
        }
    }
    fn xor_assign(&mut self, rhs: Self) {
        for i in 0..A + B {
            self.words
                .set_word(i, self.words.word(i) ^ rhs.words.word(i));
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct F2PolyOps;
impl<const AB: usize, const A: usize, const BB: usize, const B: usize>
    WideMul<F2Poly<AB, A>, F2Poly<BB, B>> for F2PolyOps
{
    type Product = F2PolyProduct<A, B>;
    fn mul_wide(&self, lhs: &F2Poly<AB, A>, rhs: &F2Poly<BB, B>) -> Self::Product {
        let mut out = F2PolyProduct::zero();
        for i in 0..A {
            for j in 0..B {
                let product = crate::gf128::kernels::clmul_64x64(lhs.words.0[i], rhs.words.0[j]);
                out.words
                    .set_word(i + j, out.words.word(i + j) ^ product[0]);
                out.words
                    .set_word(i + j + 1, out.words.word(i + j + 1) ^ product[1]);
            }
        }
        out
    }
}
impl<const AB: usize, const A: usize, const BB: usize, const B: usize>
    BatchMulAcc<F2Poly<AB, A>, F2Poly<BB, B>> for F2PolyOps
{
    type Accumulator = XorAccumulator<F2PolyProduct<A, B>>;
    fn batch_mul_acc(&self, lhs: &[F2Poly<AB, A>], rhs: &[F2Poly<BB, B>]) -> Self::Accumulator {
        assert_eq!(lhs.len(), rhs.len());
        self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
    }
    fn batch_mul_acc_map(
        &self,
        len: usize,
        mut term: impl FnMut(usize) -> (F2Poly<AB, A>, F2Poly<BB, B>),
    ) -> Self::Accumulator {
        let mut out = Self::Accumulator::zero();
        for i in 0..len {
            let (a, b) = term(i);
            out.payload.xor_assign(self.mul_wide(&a, &b));
        }
        out
    }
}
impl<const A: usize, const B: usize> Reduce<F2PolyProduct<A, B>> for Gf128Ops {
    type Output = Gf128;
    fn reduce(&self, value: F2PolyProduct<A, B>) -> Gf128 {
        let mut out = Gf128::ZERO;
        // Word-at-a-time Horner reduction; no bit-at-a-time wide fallback.
        for i in (0..A + B).rev() {
            out = Gf128::from(crate::gf128::kernels::reduce_256_to_128([
                value.words.word(i),
                out.lo,
                out.hi,
                0,
            ]));
        }
        out
    }
}
impl<const A: usize, const B: usize> Reduce<XorAccumulator<F2PolyProduct<A, B>>> for Gf128Ops {
    type Output = Gf128;
    fn reduce(&self, value: XorAccumulator<F2PolyProduct<A, B>>) -> Gf128 {
        self.reduce(value.payload)
    }
}

impl<const B: usize, const W: usize> core::ops::Add for F2Poly<B, W> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.xor(&rhs)
    }
}
impl<const B: usize, const W: usize> core::ops::AddAssign for F2Poly<B, W> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.xor(&rhs);
    }
}
impl<const B: usize, const W: usize> Default for F2Poly<B, W> {
    fn default() -> Self {
        Self::ZERO
    }
}
impl Gf128Ops {
    /// Evaluate a polynomial with a fixed public bit bound. The coefficient
    /// bits choose masks, never loop lengths or memory addresses.
    pub fn evaluate_polynomial<const B: usize, const W: usize>(
        &self,
        polynomial: &F2Poly<B, W>,
        point: &Gf128,
    ) -> Gf128 {
        let mut acc = Gf128::ZERO;
        for i in (0..B).rev() {
            acc = self.mul(&acc, point);
            acc.lo ^= polynomial.bit(i).as_u64();
        }
        acc
    }
}
