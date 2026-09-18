//! Exact products and accumulators with one headroom limb.

use super::{Uint, Z};
use crate::ct::{CtEq, CtMask, CtValue};
use crate::traits::{BatchMulAcc, MergeAccumulator, WideMul};
#[path = "kernels.rs"]
mod kernels;

pub(crate) trait Words {
    fn len(&self) -> usize;
    fn word(&self, public_index: usize) -> u64;
}
impl<const L: usize> Words for Uint<L> {
    fn len(&self) -> usize {
        L
    }
    fn word(&self, i: usize) -> u64 {
        self.0[i]
    }
}
impl Words for [u64] {
    fn len(&self) -> usize {
        <[u64]>::len(self)
    }
    fn word(&self, i: usize) -> u64 {
        self[i]
    }
}

/// Borrowed two's-complement magnitude, without a temporary word vector.
/// The carry of a two's-complement negation reaches word i exactly when all
/// lower words are zero. Cache the first nonzero position with a fixed scan.
pub(crate) struct SignedMagnitude<'a> {
    words: &'a [u64],
    negative: CtMask,
    first_nonzero: usize,
}
impl<'a> SignedMagnitude<'a> {
    pub(crate) fn new(words: &'a [u64]) -> Self {
        use crate::CtSelect;
        let negative = CtMask::from_lsb(words[words.len() - 1] >> 63);
        let mut first_nonzero = words.len() as u64;
        let mut prefix_zero = CtMask::TRUE;
        for (i, word) in words.iter().enumerate() {
            let nonzero = !word.ct_is_zero();
            first_nonzero = u64::ct_select(&first_nonzero, &(i as u64), prefix_zero & nonzero);
            prefix_zero = prefix_zero & !nonzero;
        }
        Self {
            words,
            negative,
            first_nonzero: first_nonzero as usize,
        }
    }
    pub(crate) fn negative(&self) -> CtMask {
        self.negative
    }
}
impl Words for SignedMagnitude<'_> {
    fn len(&self) -> usize {
        self.words.len()
    }
    fn word(&self, i: usize) -> u64 {
        use crate::{CtOrd, CtSelect};
        let carry = !(self.first_nonzero as u64).ct_lt(&(i as u64));
        let negated = (!self.words[i]).wrapping_add(carry.word() & 1);
        u64::ct_select(&self.words[i], &negated, self.negative)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UintProduct<const A: usize, const B: usize> {
    pub(crate) low: [u64; A],
    pub(crate) high: [u64; B],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZProduct<const A: usize, const B: usize>(pub(crate) UintProduct<A, B>);

/// Exact sum of fewer than 2^64 full-width products, including merged batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UintAccumulator<const A: usize, const B: usize> {
    pub(crate) low: [u64; A],
    pub(crate) high: [u64; B],
    pub(crate) head: u64,
}
/// Signed counterpart of `UintAccumulator`; merging has the same term bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZAccumulator<const A: usize, const B: usize>(pub(crate) UintAccumulator<A, B>);

#[derive(Clone, Copy, Debug, Default)]
pub struct IntegerOps;

impl IntegerOps {
    /// Signed product modulo 2^(64*OUT), with a schedule fixed by the widths.
    /// This is an exact signed product when the caller's public bounds place
    /// the result in `Z<OUT>`; otherwise it has wrapping semantics.
    #[inline(always)]
    pub fn wrapping_signed_product<const A: usize, const B: usize, const OUT: usize>(
        &self,
        lhs: &Z<A>,
        rhs: &Z<B>,
    ) -> Z<OUT> {
        const {
            assert!(A > 0 && B > 0 && OUT > 0 && OUT <= A + B);
        }
        let mut words = [0; OUT];
        for i in 0..A.min(OUT) {
            let mut carry = 0;
            for j in 0..B.min(OUT - i) {
                let sum = lhs.0.0[i] as u128 * rhs.0.0[j] as u128 + words[i + j] as u128 + carry;
                words[i + j] = sum as u64;
                carry = sum >> 64;
            }
            if i + B < OUT {
                words[i + B] = carry as u64;
            }
        }
        // a = unsigned(a) - sign(a)*2^(64*A), and likewise for b.
        // The product of the two sign corrections vanishes modulo 2^(64*OUT).
        let sign_a = lhs.is_negative_ct().word();
        let sign_b = rhs.is_negative_ct().word();
        let mut borrow = 0;
        for i in A..OUT {
            let (word, b0) = words[i].overflowing_sub(rhs.0.0[i - A] & sign_a);
            let (word, b1) = word.overflowing_sub(borrow);
            words[i] = word;
            borrow = u64::from(b0 | b1);
        }
        let mut borrow = 0;
        for i in B..OUT {
            let (word, b0) = words[i].overflowing_sub(lhs.0.0[i - B] & sign_b);
            let (word, b1) = word.overflowing_sub(borrow);
            words[i] = word;
            borrow = u64::from(b0 | b1);
        }
        Z::from_twos_complement_words(words)
    }
}

impl<const A: usize, const B: usize> Words for UintProduct<A, B> {
    fn len(&self) -> usize {
        A + B
    }
    fn word(&self, i: usize) -> u64 {
        if i < A { self.low[i] } else { self.high[i - A] }
    }
}
impl<const A: usize, const B: usize> Words for UintAccumulator<A, B> {
    fn len(&self) -> usize {
        A + B + 1
    }
    fn word(&self, i: usize) -> u64 {
        if i < A {
            self.low[i]
        } else if i < A + B {
            self.high[i - A]
        } else {
            assert_eq!(i, A + B);
            self.head
        }
    }
}

impl<const A: usize, const B: usize> UintProduct<A, B> {
    pub const ZERO: Self = {
        assert!(A > 0 && B > 0);
        Self {
            low: [0; A],
            high: [0; B],
        }
    };
    pub const fn as_parts(&self) -> (&[u64; A], &[u64; B]) {
        (&self.low, &self.high)
    }
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Uint<M>> {
        extract_unsigned(self)
    }
    pub(crate) fn set_word(&mut self, i: usize, value: u64) {
        if i < A {
            self.low[i] = value;
        } else {
            self.high[i - A] = value;
        }
    }
    pub(crate) fn conditional_negate(&mut self, mask: CtMask) {
        let mut carry = mask.word() & 1;
        for i in 0..self.len() {
            let sum = (self.word(i) ^ mask.word()) as u128 + carry as u128;
            self.set_word(i, sum as u64);
            carry = (sum >> 64) as u64;
        }
    }
}
impl<const A: usize, const B: usize> ZProduct<A, B> {
    pub fn as_twos_complement_parts(&self) -> (&[u64; A], &[u64; B]) {
        self.0.as_parts()
    }
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Z<M>> {
        extract_signed(&self.0)
    }
    pub fn unsigned_abs(&self) -> UintProduct<A, B> {
        let mut out = self.0;
        out.conditional_negate(CtMask::from_lsb(self.0.word(A + B - 1) >> 63));
        out
    }
}

pub(crate) fn extract_unsigned<const M: usize>(words: &impl Words) -> CtValue<Uint<M>> {
    let mut out = Uint::ZERO;
    let mut excess = 0;
    for i in 0..words.len() {
        if i < M {
            out.0[i] = words.word(i);
        } else {
            excess |= words.word(i);
        }
    }
    CtValue::new(out, excess.ct_is_zero())
}
pub(crate) fn extract_signed<const M: usize>(words: &(impl Words + ?Sized)) -> CtValue<Z<M>> {
    const {
        assert!(M > 0);
    }
    let sign = 0u64.wrapping_sub(words.word(words.len() - 1) >> 63);
    let out = Z::from_twos_complement_words(core::array::from_fn(|i| {
        if i < words.len() { words.word(i) } else { sign }
    }));
    let expected = out.is_negative_ct().word();
    let mut excess = sign ^ expected;
    for i in M..words.len() {
        excess |= words.word(i) ^ expected;
    }
    CtValue::new(out, excess.ct_is_zero())
}

impl<const A: usize, const B: usize> UintAccumulator<A, B> {
    pub const ZERO: Self = {
        assert!(A > 0 && B > 0);
        Self {
            low: [0; A],
            high: [0; B],
            head: 0,
        }
    };
    pub const fn as_parts(&self) -> (&[u64; A], &[u64; B], u64) {
        (&self.low, &self.high, self.head)
    }
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Uint<M>> {
        extract_unsigned(self)
    }
    pub(crate) fn set_word(&mut self, i: usize, value: u64) {
        if i < A {
            self.low[i] = value;
        } else if i < A + B {
            self.high[i - A] = value;
        } else {
            assert_eq!(i, A + B);
            self.head = value;
        }
    }
    pub(crate) fn wrapping_merge(&mut self, rhs: &Self) -> u64 {
        let mut carry = 0;
        for i in 0..self.len() {
            let sum = self.word(i) as u128 + rhs.word(i) as u128 + carry as u128;
            self.set_word(i, sum as u64);
            carry = (sum >> 64) as u64;
        }
        carry
    }
    pub(crate) fn conditional_negate(&mut self, mask: CtMask) {
        let mut carry = mask.word() & 1;
        for i in 0..self.len() {
            let sum = (self.word(i) ^ mask.word()) as u128 + carry as u128;
            self.set_word(i, sum as u64);
            carry = (sum >> 64) as u64;
        }
    }
    /// Private infallible MAC. Shapes and the total-term bound are established by callers.
    #[inline]
    pub(crate) fn mac(&mut self, lhs: &Uint<A>, rhs: &Uint<B>) {
        // Const dimensions select the retained production schedule. These
        // branches disappear in each monomorphized batch loop.
        if A == 2 && B == 2 {
            let mut words = [
                self.low[0],
                self.low[1],
                self.high[0],
                self.high[1],
                self.head,
            ];
            kernels::multiply_accumulate_2x2(
                &mut words,
                [lhs.0[0], lhs.0[1]],
                [rhs.0[0], rhs.0[1]],
            );
            self.low[0] = words[0];
            self.low[1] = words[1];
            self.high[0] = words[2];
            self.high[1] = words[3];
            self.head = words[4];
            return;
        }
        if A == 2 && B == 1 {
            let mut words = [self.low[0], self.low[1], self.high[0], self.head, 0];
            kernels::multiply_accumulate_2x1(&mut words, [lhs.0[0], lhs.0[1]], rhs.0[0]);
            self.low[0] = words[0];
            self.low[1] = words[1];
            self.high[0] = words[2];
            self.head = words[3];
            return;
        }
        for i in 0..A {
            let mut carry = 0;
            for j in 0..B {
                let sum = lhs.0[i] as u128 * rhs.0[j] as u128 + self.word(i + j) as u128 + carry;
                self.set_word(i + j, sum as u64);
                carry = sum >> 64;
            }
            for k in i + B..A + B + 1 {
                let sum = self.word(k) as u128 + carry;
                self.set_word(k, sum as u64);
                carry = sum >> 64;
            }
            debug_assert_eq!(carry, 0, "exact accumulator capacity exceeded");
        }
    }
}
impl<const A: usize, const B: usize> From<UintProduct<A, B>> for UintAccumulator<A, B> {
    fn from(product: UintProduct<A, B>) -> Self {
        Self {
            low: product.low,
            high: product.high,
            head: 0,
        }
    }
}
impl<const A: usize, const B: usize> From<ZProduct<A, B>> for ZAccumulator<A, B> {
    fn from(product: ZProduct<A, B>) -> Self {
        let head = 0u64.wrapping_sub(product.0.word(A + B - 1) >> 63);
        Self(UintAccumulator {
            low: product.0.low,
            high: product.0.high,
            head,
        })
    }
}
impl<const A: usize, const B: usize> MergeAccumulator for UintAccumulator<A, B> {
    fn zero() -> Self {
        Self::ZERO
    }
    #[inline]
    fn merge_assign(&mut self, rhs: &Self) {
        let carry = self.wrapping_merge(rhs);
        debug_assert_eq!(carry, 0, "exact accumulator capacity exceeded");
    }
}
impl<const A: usize, const B: usize> ZAccumulator<A, B> {
    pub const ZERO: Self = Self(UintAccumulator::ZERO);
    pub fn as_twos_complement_parts(&self) -> (&[u64; A], &[u64; B], u64) {
        self.0.as_parts()
    }
    pub fn checked_resize_ct<const M: usize>(&self) -> CtValue<Z<M>> {
        extract_signed(&self.0)
    }
    pub fn is_negative_ct(&self) -> CtMask {
        CtMask::from_lsb(self.0.head >> 63)
    }
    pub fn unsigned_abs(&self) -> UintAccumulator<A, B> {
        let mut out = self.0;
        out.conditional_negate(self.is_negative_ct());
        out
    }
    pub(crate) fn add_product(&mut self, product: ZProduct<A, B>) {
        self.merge_assign(&product.into());
    }
}
impl<const A: usize, const B: usize> MergeAccumulator for ZAccumulator<A, B> {
    fn zero() -> Self {
        Self::ZERO
    }
    fn merge_assign(&mut self, rhs: &Self) {
        let before = self.0.head;
        self.0.wrapping_merge(&rhs.0);
        debug_assert_eq!(
            ((before ^ self.0.head) & (rhs.0.head ^ self.0.head)) >> 63,
            0,
            "signed accumulator capacity exceeded"
        );
    }
}

impl<const A: usize, const B: usize> WideMul<Uint<A>, Uint<B>> for IntegerOps {
    type Product = UintProduct<A, B>;
    #[inline]
    fn mul_wide(&self, lhs: &Uint<A>, rhs: &Uint<B>) -> Self::Product {
        let mut out = UintProduct::ZERO;
        for i in 0..A {
            let mut carry = 0;
            for j in 0..B {
                let sum = lhs.0[i] as u128 * rhs.0[j] as u128 + out.word(i + j) as u128 + carry;
                out.set_word(i + j, sum as u64);
                carry = sum >> 64;
            }
            out.set_word(i + B, carry as u64);
        }
        out
    }
}
impl<const A: usize, const B: usize> WideMul<Uint<A>, Z<B>> for IntegerOps {
    type Product = ZProduct<A, B>;
    fn mul_wide(&self, lhs: &Uint<A>, rhs: &Z<B>) -> Self::Product {
        let mut product = self.mul_wide(lhs, &rhs.unsigned_abs());
        product.conditional_negate(rhs.is_negative_ct());
        ZProduct(product)
    }
}
impl<const A: usize, const B: usize> WideMul<Z<A>, Uint<B>> for IntegerOps {
    type Product = ZProduct<A, B>;
    fn mul_wide(&self, lhs: &Z<A>, rhs: &Uint<B>) -> Self::Product {
        let mut product = self.mul_wide(&lhs.unsigned_abs(), rhs);
        product.conditional_negate(lhs.is_negative_ct());
        ZProduct(product)
    }
}
impl<const A: usize, const B: usize> WideMul<Z<A>, Z<B>> for IntegerOps {
    type Product = ZProduct<A, B>;
    fn mul_wide(&self, lhs: &Z<A>, rhs: &Z<B>) -> Self::Product {
        let mut product = self.mul_wide(&lhs.unsigned_abs(), &rhs.unsigned_abs());
        product.conditional_negate(CtMask::from_lsb((lhs.0.0[A - 1] ^ rhs.0.0[B - 1]) >> 63));
        ZProduct(product)
    }
}

impl<const A: usize, const B: usize> BatchMulAcc<Uint<A>, Uint<B>> for IntegerOps {
    type Accumulator = UintAccumulator<A, B>;
    #[inline(always)]
    fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &Uint<A>, rhs: &Uint<B>) {
        acc.mac(lhs, rhs);
    }
    fn batch_mul_acc(&self, lhs: &[Uint<A>], rhs: &[Uint<B>]) -> Self::Accumulator {
        assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
        self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
    }
    fn batch_mul_acc_map(
        &self,
        len: usize,
        mut term: impl FnMut(usize) -> (Uint<A>, Uint<B>),
    ) -> Self::Accumulator {
        let mut out = UintAccumulator::ZERO;
        for i in 0..len {
            let (lhs, rhs) = term(i);
            out.mac(&lhs, &rhs);
        }
        out
    }
}
macro_rules! signed_batch {
    ($lhs:ident, $rhs:ident) => {
        impl<const A: usize, const B: usize> BatchMulAcc<$lhs<A>, $rhs<B>> for IntegerOps {
            type Accumulator = ZAccumulator<A, B>;
            #[inline(always)]
            fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &$lhs<A>, rhs: &$rhs<B>) {
                acc.add_product(self.mul_wide(lhs, rhs));
            }
            fn batch_mul_acc(&self, lhs: &[$lhs<A>], rhs: &[$rhs<B>]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
            }
            fn batch_mul_acc_map(
                &self,
                len: usize,
                mut term: impl FnMut(usize) -> ($lhs<A>, $rhs<B>),
            ) -> Self::Accumulator {
                let mut out = ZAccumulator::ZERO;
                for i in 0..len {
                    let (lhs, rhs) = term(i);
                    out.add_product(self.mul_wide(&lhs, &rhs));
                }
                out
            }
        }
    };
}
signed_batch!(Uint, Z);
signed_batch!(Z, Uint);
signed_batch!(Z, Z);

macro_rules! native_product {
    ($lhs:ty,$rhs:ty,$a:expr,$b:expr,$left:expr,$right:expr) => {
        impl WideMul<$lhs, $rhs> for IntegerOps {
            type Product = UintProduct<$a, $b>;
            fn mul_wide(&self, lhs: &$lhs, rhs: &$rhs) -> Self::Product {
                self.mul_wide(&($left)(*lhs), &($right)(*rhs))
            }
        }
        impl BatchMulAcc<$lhs, $rhs> for IntegerOps {
            type Accumulator = UintAccumulator<$a, $b>;
            fn batch_mul_acc(&self, lhs: &[$lhs], rhs: &[$rhs]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len());
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
            }
            fn batch_mul_acc_map(
                &self,
                len: usize,
                mut term: impl FnMut(usize) -> ($lhs, $rhs),
            ) -> Self::Accumulator {
                let mut out = UintAccumulator::ZERO;
                for i in 0..len {
                    let (a, b) = term(i);
                    out.mac(&($left)(a), &($right)(b));
                }
                out
            }
        }
    };
}
macro_rules! native_rhs {
    ($lhs:ty,$a:expr,$left:expr) => {
        native_product!($lhs, u32, $a, 1, $left, |v: u32| Uint::from_words([
            v as u64
        ]));
        native_product!($lhs, u64, $a, 1, $left, |v: u64| Uint::from_words([v]));
        native_product!($lhs, u128, $a, 2, $left, |v: u128| Uint::from_words([
            v as u64,
            (v >> 64) as u64
        ]));
    };
}
native_rhs!(u32, 1, |v: u32| Uint::from_words([v as u64]));
native_rhs!(u64, 1, |v: u64| Uint::from_words([v]));
native_rhs!(u128, 2, |v: u128| Uint::from_words([
    v as u64,
    (v >> 64) as u64
]));
