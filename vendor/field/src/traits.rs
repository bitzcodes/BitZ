//! Operation providers: the element type belongs to the provider type.

use crate::{
    ct::{CtEq, CtSelect, CtValue},
    integer::Uint,
};

pub trait RingOps {
    type Elem: Copy + Send + Sync + CtEq + CtSelect;
    fn zero(&self) -> Self::Elem;
    /// Allocate initialized additive identities. Providers with a zero bit
    /// representation can request zeroed allocation without a serial fill.
    fn zero_vec(&self, len: usize) -> Vec<Self::Elem> {
        vec![self.zero(); len]
    }

    fn one(&self) -> Self::Elem;
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    fn square(&self, a: &Self::Elem) -> Self::Elem {
        self.mul(a, a)
    }

    /// Scan the complete public exponent width, including leading zeroes.
    fn pow_ct<const E: usize>(&self, base: &Self::Elem, exponent: &Uint<E>) -> Self::Elem {
        let mut result = self.one();
        for i in (0..E * 64).rev() {
            result = self.square(&result);
            let product = self.mul(&result, base);
            result = Self::Elem::ct_select(&result, &product, exponent.bit(i).mask());
        }
        result
    }
}

pub trait FieldOps: RingOps {
    fn inverse_ct(&self, a: &Self::Elem) -> CtValue<Self::Elem>;
    fn div_ct(&self, numerator: &Self::Elem, denominator: &Self::Elem) -> CtValue<Self::Elem> {
        self.inverse_ct(denominator)
            .map(|inverse| self.mul(numerator, &inverse))
    }
}

// Borrowing a context preserves its associated element type and specialized
// scalar implementation. Prepared operations can retain `&field` directly.
impl<C: RingOps + ?Sized> RingOps for &C {
    type Elem = C::Elem;
    fn zero(&self) -> Self::Elem {
        (**self).zero()
    }
    fn zero_vec(&self, len: usize) -> Vec<Self::Elem> {
        (**self).zero_vec(len)
    }
    fn one(&self) -> Self::Elem {
        (**self).one()
    }
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        (**self).add(a, b)
    }
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        (**self).sub(a, b)
    }
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        (**self).neg(a)
    }
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        (**self).mul(a, b)
    }
    fn square(&self, a: &Self::Elem) -> Self::Elem {
        (**self).square(a)
    }
    fn pow_ct<const E: usize>(&self, a: &Self::Elem, e: &Uint<E>) -> Self::Elem {
        (**self).pow_ct(a, e)
    }
}
impl<C: FieldOps + ?Sized> FieldOps for &C {
    fn inverse_ct(&self, a: &Self::Elem) -> CtValue<Self::Elem> {
        (**self).inverse_ct(a)
    }
}
impl<C: IntegerEmbedding<T> + ?Sized, T> IntegerEmbedding<T> for &C {
    fn from_integer(&self, value: &T) -> Self::Elem {
        (**self).from_integer(value)
    }
}

pub trait IntegerEmbedding<T>: RingOps {
    fn from_integer(&self, value: &T) -> Self::Elem;
}
pub trait FieldEmbedding<T>: FieldOps {
    fn embed(&self, value: &T) -> Self::Elem;
}
pub trait WideMul<Lhs, Rhs = Lhs> {
    type Product;
    fn mul_wide(&self, lhs: &Lhs, rhs: &Rhs) -> Self::Product;
}
pub trait BatchMulAcc<Lhs, Rhs = Lhs> {
    type Accumulator: MergeAccumulator + Send;
    /// Add one exact product. The caller establishes the total term bound,
    /// including merged workers, before entering the loop.
    fn mul_acc(&self, accumulator: &mut Self::Accumulator, lhs: &Lhs, rhs: &Rhs) {
        accumulator.merge_assign(
            &self.batch_mul_acc(core::slice::from_ref(lhs), core::slice::from_ref(rhs)),
        );
    }
    fn batch_mul_acc(&self, lhs: &[Lhs], rhs: &[Rhs]) -> Self::Accumulator;
    /// Invoke `term` exactly once for each index, in ascending order.
    fn batch_mul_acc_map(
        &self,
        len: usize,
        term: impl FnMut(usize) -> (Lhs, Rhs),
    ) -> Self::Accumulator;
}
/// Exact merge under the documented total-term bound. No count metadata is stored.
pub trait MergeAccumulator: Sized {
    fn zero() -> Self;
    fn merge_assign(&mut self, rhs: &Self);
}
pub trait Reduce<Input> {
    type Output;
    fn reduce(&self, input: Input) -> Self::Output;

    /// Prepare reduction once from a public total product count. Each input
    /// must satisfy that bound, including all merged worker contributions.
    /// The returned operation retains this context and performs no allocation.
    fn prepare_reduce(&self, _max_terms: usize) -> impl Fn(Input) -> Self::Output + Send + Sync + '_
    where
        Self: Sync,
    {
        move |input| self.reduce(input)
    }
}

/// A reusable linear combination of at most sixteen declared-width operands.
/// Preparation depends only on the public coefficients and field context.
/// Implementations must process every declared operand limb, including zeroes.
pub trait PreparedLinearCombination<Src>: FieldOps {
    type Prepared<'a, const TERMS: usize>: Send + Sync
    where
        Self: 'a;
    fn prepare_linear_combination<const TERMS: usize>(
        &self,
        coefficients: [Self::Elem; TERMS],
    ) -> Self::Prepared<'_, TERMS>;
    /// The preparation borrows its field context, so evaluation cannot use a
    /// different modulus. `read` runs once per coefficient, in ascending order.
    fn linear_combination<const TERMS: usize>(
        prepared: &Self::Prepared<'_, TERMS>,
        read: impl FnMut(usize) -> Src,
    ) -> Self::Elem;
}

pub trait BatchFieldOps: FieldOps {
    fn batch_mul_into(&self, lhs: &[Self::Elem], rhs: &[Self::Elem], out: &mut [Self::Elem]) {
        assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
        assert_eq!(lhs.len(), out.len(), "batch output length differs");
        for ((a, b), dst) in lhs.iter().zip(rhs).zip(out) {
            *dst = self.mul(a, b);
        }
    }
    fn batch_mul(&self, lhs: &[Self::Elem], rhs: &[Self::Elem]) -> Vec<Self::Elem> {
        let mut out = vec![self.zero(); lhs.len()];
        self.batch_mul_into(lhs, rhs, &mut out);
        out
    }
    /// Scratch must have at least `input.len()` entries. Zero entries stay zero.
    fn batch_invert_or_zero_ct_into(
        &self,
        input: &[Self::Elem],
        out: &mut [Self::Elem],
        scratch: &mut [Self::Elem],
    ) {
        assert_eq!(input.len(), out.len(), "batch output length differs");
        assert!(
            scratch.len() >= input.len(),
            "insufficient inversion scratch"
        );
        if input.is_empty() {
            return;
        }
        let one = self.one();
        let zero = self.zero();
        let mut product = one;
        for (a, prefix) in input.iter().zip(scratch.iter_mut()) {
            *prefix = product;
            let nonzero = Self::Elem::ct_select(a, &one, a.ct_is_zero());
            product = self.mul(&product, &nonzero);
        }
        // Replacing all zero inputs by one makes the product invertible.
        let mut inverse = *self.inverse_ct(&product).value();
        for i in (0..input.len()).rev() {
            let is_zero = input[i].ct_is_zero();
            let value = self.mul(&inverse, &scratch[i]);
            out[i] = Self::Elem::ct_select(&value, &zero, is_zero);
            let factor = Self::Elem::ct_select(&input[i], &one, is_zero);
            inverse = self.mul(&inverse, &factor);
        }
    }
    fn batch_invert_or_zero_ct(&self, input: &[Self::Elem]) -> Vec<Self::Elem> {
        let mut out = vec![self.zero(); input.len()];
        let mut scratch = vec![self.zero(); input.len()];
        self.batch_invert_or_zero_ct_into(input, &mut out, &mut scratch);
        out
    }
}

pub trait CheckedArithmetic: Sized {
    fn checked_add_ct(&self, rhs: &Self) -> CtValue<Self>;
    fn checked_sub_ct(&self, rhs: &Self) -> CtValue<Self>;
    fn checked_mul_ct(&self, rhs: &Self) -> CtValue<Self>;
    fn checked_neg_ct(&self) -> CtValue<Self>;
    fn checked_div_rem_ct(&self, rhs: &Self) -> CtValue<(Self, Self)>;
}
pub trait WrappingArithmetic: Sized {
    fn wrapping_add(&self, rhs: &Self) -> Self;
    fn wrapping_sub(&self, rhs: &Self) -> Self;
    fn wrapping_mul(&self, rhs: &Self) -> Self;
    fn wrapping_neg(&self) -> Self;
}

/// Adjacent, little-endian Boolean folding. Challenges must already be known.
pub trait FoldPairs<Src, Dst>: RingOps {
    /// dst[i] = (1-r) src[2i] + r src[2i+1]. Requires src.len()=2*dst.len().
    fn fold_pairs_into(&self, src: &[Src], dst: &mut [Dst], challenge: &Self::Elem);
    /// The mapped source has exactly 2*dst.len() entries. Used for split-limb
    /// storage or public zero padding without allocating a source vector.
    fn fold_pairs_map_into(
        &self,
        read: impl FnMut(usize) -> Src,
        dst: &mut [Dst],
        challenge: &Self::Elem,
    );

    /// dst[i] = Σ_b eq(challenges,b) src[2^k*i+b], r[0] binds the low bit.
    /// Allocates 2^k small equality weights; native conversion is fused into
    /// each batch MAC. This is Boolean folding, not univariate interpolation.
    fn fold_prefix_into(&self, src: &[Src], dst: &mut [Dst], challenges: &[Self::Elem])
    where
        Src: Copy,
        Self: BatchMulAcc<Self::Elem, Src>
            + Reduce<<Self as BatchMulAcc<Self::Elem, Src>>::Accumulator, Output = Dst>,
    {
        let width = 1usize
            .checked_shl(u32::try_from(challenges.len()).expect("fold depth overflow"))
            .expect("fold depth overflow");
        assert_eq!(
            Some(src.len()),
            dst.len().checked_mul(width),
            "fold lengths differ"
        );
        self.fold_prefix_map_into(|i| src[i], dst, challenges);
    }
    /// Mapped counterpart of fold_prefix_into; reads each source entry once.
    fn fold_prefix_map_into(
        &self,
        mut read: impl FnMut(usize) -> Src,
        dst: &mut [Dst],
        challenges: &[Self::Elem],
    ) where
        Self: BatchMulAcc<Self::Elem, Src>
            + Reduce<<Self as BatchMulAcc<Self::Elem, Src>>::Accumulator, Output = Dst>,
    {
        let width = 1usize
            .checked_shl(u32::try_from(challenges.len()).expect("fold depth overflow"))
            .expect("fold depth overflow");
        dst.len().checked_mul(width).expect("fold extent overflow");
        let mut weights = vec![self.one(); width];
        for (bit, r) in challenges.iter().enumerate() {
            let complement = self.sub(&self.one(), r);
            for (i, w) in weights.iter_mut().enumerate() {
                *w = self.mul(w, if (i >> bit) & 1 == 0 { &complement } else { r });
            }
        }
        for (i, out) in dst.iter_mut().enumerate() {
            *out = Reduce::reduce(
                self,
                self.batch_mul_acc_map(width, |b| (weights[b], read(i * width + b))),
            );
        }
    }
}

// Prepared linear maps may retain a borrowed provider. Borrowing must preserve
// its exact accumulator type and its prepared reduction, just as RingOps does.
impl<C: BatchMulAcc<L, R> + ?Sized, L, R> BatchMulAcc<L, R> for &C {
    type Accumulator = C::Accumulator;
    fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &L, rhs: &R) {
        (**self).mul_acc(acc, lhs, rhs)
    }
    fn batch_mul_acc(&self, lhs: &[L], rhs: &[R]) -> Self::Accumulator {
        (**self).batch_mul_acc(lhs, rhs)
    }
    fn batch_mul_acc_map(&self, len: usize, term: impl FnMut(usize) -> (L, R)) -> Self::Accumulator {
        (**self).batch_mul_acc_map(len, term)
    }
}
impl<C: Reduce<I> + Sync + ?Sized, I> Reduce<I> for &C {
    type Output = C::Output;
    fn reduce(&self, input: I) -> Self::Output {
        (**self).reduce(input)
    }
    fn prepare_reduce(&self, max_terms: usize) -> impl Fn(I) -> Self::Output + Send + Sync + '_ {
        (**self).prepare_reduce(max_terms)
    }
}
