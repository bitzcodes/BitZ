//! Prime fields in Montgomery form. Runtime modulus matching is a caller contract.

use crate::integer::{IntegerOps, UintAccumulator, UintProduct, ZAccumulator, ZProduct};
use crate::traits::*;
use crate::{Bit, CtEq, CtMask, CtOrd, CtSelect, CtValue, Uint, Z};
use core::{fmt, marker::PhantomData};

mod barrett128;
mod canonical_u128;
mod montgomery128;
mod params;
use params::PrimeParameters;
mod sampling;
pub use sampling::*;
mod dot;
mod fold;
mod prepared_linear;
pub use prepared_linear::{PreparedProductReduction, PreparedWordWeights};
mod operators;
mod projection;
pub use projection::PreparedSignedProjection;

#[derive(Clone, Copy, Debug)]
pub struct RuntimePrime;
#[derive(Clone, Copy, Debug)]
pub struct StaticPrime<P>(PhantomData<fn() -> P>);

/// Compact Montgomery encoding. Construction belongs to its field provider.
#[repr(transparent)]
pub struct PrimeValue<Id, const L: usize> {
    words: Uint<L>,
    marker: PhantomData<fn() -> Id>,
}
impl<Id, const L: usize> Copy for PrimeValue<Id, L> {}
impl<Id, const L: usize> Clone for PrimeValue<Id, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<Id, const L: usize> fmt::Debug for PrimeValue<Id, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Montgomery").field(&self.words).finish()
    }
}
impl<Id, const L: usize> PartialEq for PrimeValue<Id, L> {
    fn eq(&self, rhs: &Self) -> bool {
        self.words == rhs.words
    }
}
impl<Id, const L: usize> Eq for PrimeValue<Id, L> {}
impl<Id, const L: usize> PrimeValue<Id, L> {
    /// The stored Montgomery encoding. This is not the canonical field value.
    pub const fn as_montgomery_integer(&self) -> &Uint<L> {
        &self.words
    }

    fn new(words: Uint<L>) -> Self {
        Self {
            words,
            marker: PhantomData,
        }
    }
}
impl<Id, const L: usize> CtEq for PrimeValue<Id, L> {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        self.words.ct_eq(&rhs.words)
    }
    fn ct_is_zero(&self) -> CtMask {
        self.words.ct_is_zero()
    }
}
impl<Id, const L: usize> CtSelect for PrimeValue<Id, L> {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self::new(Uint::ct_select(&a.words, &b.words, mask))
    }
}
pub type Fp<const L: usize> = PrimeValue<RuntimePrime, L>;
pub type StaticFp<P, const L: usize> = PrimeValue<StaticPrime<P>, L>;

/// One immutable context for arbitrarily many compact field elements.
#[derive(Clone, Debug)]
pub struct FpCtx<const L: usize> {
    parameters: PrimeParameters<L>,
}

/// The caller supplies an odd prime. No primality test runs in release builds.
pub fn create_prime_field<const L: usize>(prime: Uint<L>) -> FpCtx<L> {
    debug_assert!(
        is_probable_prime_public(&prime),
        "caller-supplied modulus is not a probable prime"
    );
    FpCtx {
        parameters: PrimeParameters::new(prime),
    }
}
impl<const L: usize> FpCtx<L> {
    fn params(&self) -> &PrimeParameters<L> {
        &self.parameters
    }
    pub fn modulus_bits(&self) -> usize {
        self.modulus()
            .as_words()
            .iter()
            .rposition(|&word| word != 0)
            .map_or(0, |i| {
                i * 64 + (64 - self.modulus().as_words()[i].leading_zeros()) as usize
            })
    }

    pub fn modulus(&self) -> &Uint<L> {
        &self.parameters.modulus
    }
}

/// A trusted declaration. Use `define_prime_field!` to generate its CI test.
pub trait PrimeSpec<const L: usize>: Send + Sync + 'static {
    const MODULUS: Uint<L>;
}
pub struct StaticFpOps<P, const L: usize>(PhantomData<fn() -> P>);
impl<P, const L: usize> Copy for StaticFpOps<P, L> {}
impl<P, const L: usize> Clone for StaticFpOps<P, L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: PrimeSpec<L>, const L: usize> Default for StaticFpOps<P, L> {
    fn default() -> Self {
        Self::new()
    }
}
impl<P: PrimeSpec<L>, const L: usize> StaticFpOps<P, L> {
    const PARAMETERS: PrimeParameters<L> = PrimeParameters::new(P::MODULUS);
    pub const fn new() -> Self {
        Self(PhantomData)
    }
    fn params(&self) -> &PrimeParameters<L> {
        &Self::PARAMETERS
    }
    pub fn modulus(&self) -> &Uint<L> {
        &Self::PARAMETERS.modulus
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MontyR;
#[derive(Clone, Copy, Debug)]
pub struct MontyR2;
pub struct ScaledProduct<Id, Product, Scale> {
    payload: Product,
    marker: PhantomData<fn() -> (Id, Scale)>,
}
pub struct ScaledAccumulator<Id, Acc, Scale> {
    payload: Acc,
    marker: PhantomData<fn() -> (Id, Scale)>,
}
macro_rules! scaled_value {
    ($name:ident) => {
        impl<Id, A, S> $name<Id, A, S> {
            fn new(payload: A) -> Self {
                Self {
                    payload,
                    marker: PhantomData,
                }
            }
        }
        impl<Id, A: Copy, S> Copy for $name<Id, A, S> {}
        impl<Id, A: Clone, S> Clone for $name<Id, A, S> {
            fn clone(&self) -> Self {
                Self::new(self.payload.clone())
            }
        }
        impl<Id, A: fmt::Debug, S> fmt::Debug for $name<Id, A, S> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.payload.fmt(f)
            }
        }
    };
}
scaled_value!(ScaledProduct);
scaled_value!(ScaledAccumulator);
impl<Id, A: MergeAccumulator, S> MergeAccumulator for ScaledAccumulator<Id, A, S> {
    fn zero() -> Self {
        Self::new(A::zero())
    }
    fn merge_assign(&mut self, rhs: &Self) {
        self.payload.merge_assign(&rhs.payload);
    }
}
impl<Id, A, S> ScaledAccumulator<Id, A, S> {
    /// Inspect the exact sum without changing its scale or field identity.
    pub fn unreduced_integer(&self) -> &A {
        &self.payload
    }
}
impl<Id, A: MergeAccumulator, S> Default for ScaledAccumulator<Id, A, S> {
    fn default() -> Self {
        Self::zero()
    }
}
impl<Id, A: MergeAccumulator, S> core::ops::AddAssign for ScaledAccumulator<Id, A, S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.merge_assign(&rhs);
    }
}
impl<Id, A: MergeAccumulator, S> core::ops::Add for ScaledAccumulator<Id, A, S> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self.merge_assign(&rhs);
        self
    }
}
impl<Id, P, A: From<P>, S> From<ScaledProduct<Id, P, S>> for ScaledAccumulator<Id, A, S> {
    fn from(product: ScaledProduct<Id, P, S>) -> Self {
        Self::new(product.payload.into())
    }
}

pub type PrimeProduct<Id, const L: usize> = ScaledProduct<Id, UintProduct<L, L>, MontyR2>;
pub type PrimeProductAcc<Id, const L: usize> =
    ScaledAccumulator<Id, UintAccumulator<L, L>, MontyR2>;
pub type PrimeLinearProduct<Id, const L: usize, const N: usize> =
    ScaledProduct<Id, UintProduct<L, N>, MontyR>;
pub type PrimeLinearAcc<Id, const L: usize, const N: usize> =
    ScaledAccumulator<Id, UintAccumulator<L, N>, MontyR>;
pub type PrimeSignedLinearProduct<Id, const L: usize, const N: usize> =
    ScaledProduct<Id, ZProduct<L, N>, MontyR>;
pub type PrimeSignedLinearAcc<Id, const L: usize, const N: usize> =
    ScaledAccumulator<Id, ZAccumulator<L, N>, MontyR>;
pub type FpProduct<const L: usize> = PrimeProduct<RuntimePrime, L>;
pub type FpProductAcc<const L: usize> = PrimeProductAcc<RuntimePrime, L>;
pub type FpLinearProduct<const L: usize, const N: usize> = PrimeLinearProduct<RuntimePrime, L, N>;
pub type FpLinearAcc<const L: usize, const N: usize> = PrimeLinearAcc<RuntimePrime, L, N>;
pub type FpSignedLinearProduct<const L: usize, const N: usize> =
    PrimeSignedLinearProduct<RuntimePrime, L, N>;
pub type FpSignedLinearAcc<const L: usize, const N: usize> =
    PrimeSignedLinearAcc<RuntimePrime, L, N>;
pub type StaticFpProduct<P, const L: usize> = PrimeProduct<StaticPrime<P>, L>;
pub type StaticFpProductAcc<P, const L: usize> = PrimeProductAcc<StaticPrime<P>, L>;
pub type StaticFpLinearProduct<P, const L: usize, const N: usize> =
    PrimeLinearProduct<StaticPrime<P>, L, N>;
pub type StaticFpLinearAcc<P, const L: usize, const N: usize> =
    PrimeLinearAcc<StaticPrime<P>, L, N>;
pub type StaticFpSignedLinearProduct<P, const L: usize, const N: usize> =
    PrimeSignedLinearProduct<StaticPrime<P>, L, N>;
pub type StaticFpSignedLinearAcc<P, const L: usize, const N: usize> =
    PrimeSignedLinearAcc<StaticPrime<P>, L, N>;

impl<Id, const L: usize> PrimeProductAcc<Id, L> {
    /// Adds one field product to a batch whose total length was bounded by
    /// its owner. The accumulator, including merged batches, must contain
    /// fewer than 2^64 full-width terms.
    #[inline(always)]
    pub fn accumulate(&mut self, lhs: &PrimeValue<Id, L>, rhs: &PrimeValue<Id, L>) {
        self.payload.mac(&lhs.words, &rhs.words);
    }
}

impl<Id, const L: usize, const N: usize> PrimeLinearAcc<Id, L, N> {
    /// Adds a field × integer term under the same batch-capacity contract as
    /// `PrimeProductAcc::accumulate`, without projecting the integer first.
    #[inline(always)]
    pub fn accumulate(&mut self, lhs: &PrimeValue<Id, L>, rhs: &Uint<N>) {
        self.payload.mac(&lhs.words, rhs);
    }
}

impl<Id, const L: usize, const N: usize> PrimeSignedLinearAcc<Id, L, N> {
    /// Adds a field × signed integer term at its declared width.
    #[inline(always)]
    pub fn accumulate(&mut self, lhs: &PrimeValue<Id, L>, rhs: &Z<N>) {
        self.payload
            .add_product(IntegerOps.mul_wide(&lhs.words, rhs));
    }
}

macro_rules! impl_prime {
    ([$($generic:tt)*] $provider:ty, $id:ty) => {
        impl<$($generic)*> $provider {
            /// Two prepared Montgomery coefficients times exact integers,
            /// with canonical integer output. No input projection is needed.
            #[inline]
            pub fn weighted_pair_to_integer<const N: usize>(&self, coefficients: &[PrimeValue<$id,L>; 2], values: &[Uint<N>; 2]) -> Uint<L> {
                fold::fold_unsigned(self.params(), &coefficients[0].words, &coefficients[1].words, &values[0], &values[1], true)
            }
            /// The same mixed pair with Montgomery field output.
            #[inline]
            pub fn weighted_pair<const N: usize>(&self, coefficients: &[PrimeValue<$id,L>; 2], values: &[Uint<N>; 2]) -> PrimeValue<$id,L> {
                PrimeValue::new(fold::fold_unsigned(self.params(), &coefficients[0].words, &coefficients[1].words, &values[0], &values[1], false))
            }
            pub fn to_integer(&self, value: &PrimeValue<$id, L>) -> Uint<L> { self.params().to_canonical(&value.words) }
            /// Import a trusted, reduced Montgomery encoding owned by this
            /// context. The caller guarantees the matching modulus and scale;
            /// debug builds also check the reduced range. Use the canonical
            /// codec for untrusted bytes.
            #[inline]
            pub fn from_montgomery_integer(&self, value: Uint<L>) -> PrimeValue<$id,L> {
                debug_assert!(value.ct_lt(self.modulus()).declassify());
                PrimeValue::new(value)
            }
            /// Encodes a trusted canonical integer retained with this context.
            /// The owner establishes the range once at its input boundary.
            #[inline]
            pub fn from_canonical_integer(&self, value: &Uint<L>) -> PrimeValue<$id,L> {
                debug_assert!(value.ct_lt(self.modulus()).declassify());
                PrimeValue::new(self.params().from_canonical(value))
            }
            /// Reduce a fixed-width integer without encoding it in Montgomery
            /// form. This is the explicit canonical-output preparation boundary.
            pub fn reduce_integer<const N:usize>(&self, value:&Uint<N>)->Uint<L> {
                self.params().remainder(value)
            }
            /// Canonical integer residue × Montgomery coefficient → canonical
            /// integer residue. The owner of `value` retains this context.
            #[inline]
            pub fn mul_canonical(&self, value:&Uint<L>, coefficient:&PrimeValue<$id,L>)->Uint<L> {
                debug_assert!(value.ct_lt(self.modulus()).declassify());
                if L == 2 {
                    let words = montgomery128::mul_fios(
                        [value.as_words()[0], value.as_words()[1]],
                        [coefficient.words.as_words()[0], coefficient.words.as_words()[1]],
                        [self.modulus().as_words()[0], self.modulus().as_words()[1]],
                        self.params().neg_inv,
                    );
                    return Uint::from_words(core::array::from_fn(|i| words[i]));
                }
                self.params().mul(value,&coefficient.words)
            }
            pub fn mul_canonical_into(&self, values:&[Uint<L>], coefficients:&[PrimeValue<$id,L>], out:&mut[Uint<L>]) {
                assert_eq!(values.len(),coefficients.len(),"batch input lengths differ");
                assert_eq!(values.len(),out.len(),"batch output length differs");
                for ((value,coefficient),dst) in values.iter().zip(coefficients).zip(out) {
                    *dst=self.mul_canonical(value,coefficient);
                }
            }
            /// Explicit canonical output avoids changing `Reduce<Input>`'s
            /// unique Montgomery output for this accumulator type.
            pub fn reduce_linear_to_integer<const N:usize>(&self, input:PrimeLinearAcc<$id,L,N>)->Uint<L> {
                self.params().to_canonical(&self.params().remainder(&input.payload))
            }
            pub fn from_canonical_ct(&self, value: &Uint<L>) -> CtValue<PrimeValue<$id, L>> {
                let valid = value.ct_lt(&self.params().modulus);
                let words = self.params().from_canonical(value);
                CtValue::new(PrimeValue::new(Uint::ct_select(&Uint::ZERO, &words, valid)), valid)
            }
        }
        impl<$($generic)*> RingOps for $provider {
            type Elem = PrimeValue<$id, L>;
            fn zero(&self) -> Self::Elem { PrimeValue::new(Uint::ZERO) }
            fn zero_vec(&self, len: usize) -> Vec<Self::Elem> {
                // SAFETY: PrimeValue contains only Uint<L> ([u64; L]) and a
                // PhantomData marker. All-zero bytes are valid for both, and
                // encode Montgomery zero at every supported modulus. Box
                // preserves the element alignment and allocation layout.
                unsafe { Box::<[Self::Elem]>::new_zeroed_slice(len).assume_init().into_vec() }
            }
            fn one(&self) -> Self::Elem { PrimeValue::new(self.params().one) }
            #[inline] fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem { PrimeValue::new(self.params().add(&a.words, &b.words)) }
            #[inline] fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem { PrimeValue::new(self.params().sub(&a.words, &b.words)) }
            #[inline] fn neg(&self, a: &Self::Elem) -> Self::Elem { PrimeValue::new(self.params().neg(&a.words)) }
            #[inline] fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem { PrimeValue::new(self.params().mul(&a.words, &b.words)) }
        }
        impl<$($generic)*> crate::batch::RoundArithmetic for $provider {
            type Acc=UintAccumulator<L,L>;
            fn zero_acc(&self)->Self::Acc {UintAccumulator::ZERO}
            fn mac(&self,acc:&mut Self::Acc,a:&Self::Elem,b:&Self::Elem) {acc.mac(&a.words,&b.words);}
            fn finish(&self,acc:Self::Acc)->Self::Elem {PrimeValue::new(self.params().reduce_product_acc(&acc))}
        }
        impl<$($generic)*> crate::batch::PreparedRoundMul for $provider {
            type Multiplier=PrimeValue<$id,L>;
            fn prepare_round_mul(&self,value:&Self::Elem)->Self::Multiplier {*value}
            fn mul_round_prepared(&self,multiplier:&Self::Multiplier,value:&Self::Elem)->Self::Elem {self.mul(multiplier,value)}
        }
        crate::batch::implement_sumcheck!([$($generic)*] $provider);
        impl<$($generic)*> FieldOps for $provider {
            fn inverse_ct(&self, a: &Self::Elem) -> CtValue<Self::Elem> {
                let exponent = self.params().modulus.sbb(&Uint::from_u64(2)).0;
                CtValue::new(self.pow_ct(a, &exponent), !a.ct_is_zero())
            }
        }
        impl<$($generic)*> BatchFieldOps for $provider {
            fn batch_mul_into(&self, lhs:&[Self::Elem], rhs:&[Self::Elem], out:&mut[Self::Elem]) {
                assert_eq!(lhs.len(),rhs.len(),"batch input lengths differ");
                assert_eq!(lhs.len(),out.len(),"batch output length differs");
                #[cfg(target_endian="little")]
                if L==2 {
                    montgomery128::batch_mul(self.params(),lhs,rhs,out);
                    return;
                }
                for ((a,b),out) in lhs.iter().zip(rhs).zip(out) { *out=self.mul(a,b); }
            }
        }
        impl<$($generic)*> crate::encoding::CanonicalCodec<PrimeValue<$id,L>> for $provider {
            fn encoded_len(&self)->usize {L*8}
            fn encode_into(&self,value:&PrimeValue<$id,L>,out:&mut[u8]) {crate::encoding::encode_words(&self.to_integer(value),out);}
            fn decode_ct(&self,input:&[u8])->Result<CtValue<PrimeValue<$id,L>>,crate::encoding::DecodeError> {
                Ok(self.from_canonical_ct(&crate::encoding::decode_words(input)?))
            }
        }
        impl<$($generic)*> WideMul<PrimeValue<$id, L>> for $provider {
            type Product = PrimeProduct<$id, L>;
            fn mul_wide(&self, lhs: &PrimeValue<$id, L>, rhs: &PrimeValue<$id, L>) -> Self::Product { ScaledProduct::new(IntegerOps.mul_wide(&lhs.words, &rhs.words)) }
        }
        impl<$($generic)*> BatchMulAcc<PrimeValue<$id, L>> for $provider {
            type Accumulator = PrimeProductAcc<$id, L>;
            #[inline(always)]
            fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &PrimeValue<$id, L>, rhs: &PrimeValue<$id, L>) { acc.accumulate(lhs, rhs); }
            fn batch_mul_acc(&self, lhs: &[PrimeValue<$id, L>], rhs: &[PrimeValue<$id, L>]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
            }
            fn batch_mul_acc_map(&self, len: usize, mut term: impl FnMut(usize) -> (PrimeValue<$id, L>, PrimeValue<$id, L>)) -> Self::Accumulator {
                let mut acc = UintAccumulator::ZERO;
                for i in 0..len { let (lhs, rhs) = term(i); acc.mac(&lhs.words, &rhs.words); }
                ScaledAccumulator::new(acc)
            }
        }
        impl<$($generic)*> Reduce<PrimeProductAcc<$id, L>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: PrimeProductAcc<$id, L>) -> Self::Output {
                PrimeValue::new(self.params().reduce_product_acc(&input.payload))
            }
            fn prepare_reduce(&self, max_terms: usize) -> impl Fn(PrimeProductAcc<$id, L>) -> Self::Output + Send + Sync + '_ {
                let prepared = PreparedProductReduction::<L, $id>::new(self.params(), max_terms);
                move |acc| prepared.reduce(acc)
            }
        }
        impl<$($generic)*> Reduce<PrimeProduct<$id, L>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: PrimeProduct<$id, L>) -> Self::Output { PrimeValue::new(self.params().redc(input.payload)) }
        }
        impl<$($generic)*, const N: usize> IntegerEmbedding<Uint<N>> for $provider {
            fn from_integer(&self, value: &Uint<N>) -> Self::Elem {
                PrimeValue::new(self.params().from_canonical(&self.params().remainder(value)))
            }
        }
        impl<$($generic)*, const N: usize> IntegerEmbedding<Z<N>> for $provider {
            fn from_integer(&self, value: &Z<N>) -> Self::Elem {
                let magnitude = self.from_integer(&value.unsigned_abs());
                Self::Elem::ct_select(&magnitude, &self.neg(&magnitude), value.is_negative_ct())
            }
        }
        impl<$($generic)*> IntegerEmbedding<crate::UintRef<'_>> for $provider {
            fn from_integer(&self, value: &crate::UintRef<'_>) -> Self::Elem {
                PrimeValue::new(self.params().from_canonical(&self.params().remainder(value.as_words())))
            }
        }
        impl<$($generic)*> IntegerEmbedding<crate::ZRef<'_>> for $provider {
            fn from_integer(&self, value: &crate::ZRef<'_>) -> Self::Elem {
                let input = crate::integer::product::SignedMagnitude::new(value.as_words());
                let magnitude = PrimeValue::new(self.params().from_canonical(&self.params().remainder(&input)));
                Self::Elem::ct_select(&magnitude, &self.neg(&magnitude), input.negative())
            }
        }
        impl<$($generic)*, const N: usize> WideMul<PrimeValue<$id, L>, Uint<N>> for $provider {
            type Product = PrimeLinearProduct<$id, L, N>;
            fn mul_wide(&self, lhs: &PrimeValue<$id, L>, rhs: &Uint<N>) -> Self::Product { ScaledProduct::new(IntegerOps.mul_wide(&lhs.words, rhs)) }
        }
        impl<$($generic)*, const N: usize> BatchMulAcc<PrimeValue<$id, L>, Uint<N>> for $provider {
            type Accumulator = PrimeLinearAcc<$id, L, N>;
            #[inline(always)]
            fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &PrimeValue<$id, L>, rhs: &Uint<N>) { acc.accumulate(lhs, rhs); }
            fn batch_mul_acc(&self, lhs: &[PrimeValue<$id, L>], rhs: &[Uint<N>]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
            }
            fn batch_mul_acc_map(&self, len: usize, mut term: impl FnMut(usize) -> (PrimeValue<$id, L>, Uint<N>)) -> Self::Accumulator {
                let mut acc = UintAccumulator::ZERO;
                for i in 0..len { let (lhs, rhs) = term(i); acc.mac(&lhs.words, &rhs); }
                ScaledAccumulator::new(acc)
            }
        }
        impl<$($generic)*, const N: usize> WideMul<PrimeValue<$id, L>, Z<N>> for $provider {
            type Product = PrimeSignedLinearProduct<$id, L, N>;
            fn mul_wide(&self, lhs: &PrimeValue<$id, L>, rhs: &Z<N>) -> Self::Product { ScaledProduct::new(IntegerOps.mul_wide(&lhs.words, rhs)) }
        }
        impl<$($generic)*, const N: usize> BatchMulAcc<PrimeValue<$id, L>, Z<N>> for $provider {
            type Accumulator = PrimeSignedLinearAcc<$id, L, N>;
            #[inline(always)]
            fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &PrimeValue<$id, L>, rhs: &Z<N>) { acc.accumulate(lhs, rhs); }
            fn batch_mul_acc(&self, lhs: &[PrimeValue<$id, L>], rhs: &[Z<N>]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
            }
            fn batch_mul_acc_map(&self, len: usize, mut term: impl FnMut(usize) -> (PrimeValue<$id, L>, Z<N>)) -> Self::Accumulator {
                let mut acc = ZAccumulator::ZERO;
                for i in 0..len { let (lhs, rhs) = term(i); acc.add_product(IntegerOps.mul_wide(&lhs.words, &rhs)); }
                ScaledAccumulator::new(acc)
            }
        }
        impl<$($generic)*, const N: usize> Reduce<PrimeLinearAcc<$id, L, N>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: PrimeLinearAcc<$id, L, N>) -> Self::Output { PrimeValue::new(self.params().remainder(&input.payload)) }
        }
        impl<$($generic)*, const N: usize> Reduce<PrimeSignedLinearAcc<$id, L, N>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: PrimeSignedLinearAcc<$id, L, N>) -> Self::Output {
                let magnitude = PrimeValue::new(self.params().remainder(&input.payload.unsigned_abs()));
                Self::Output::ct_select(&magnitude, &self.neg(&magnitude), input.payload.is_negative_ct())
            }
        }
        impl<$($generic)*, const N: usize> Reduce<PrimeLinearProduct<$id, L, N>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: PrimeLinearProduct<$id, L, N>) -> Self::Output { PrimeValue::new(self.params().remainder(&input.payload)) }
        }
        impl<$($generic)*, const N: usize> Reduce<PrimeSignedLinearProduct<$id, L, N>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: PrimeSignedLinearProduct<$id, L, N>) -> Self::Output {
                let acc: PrimeSignedLinearAcc<$id, L, N> = input.into(); self.reduce(acc)
            }
        }
        impl<$($generic)*, const A: usize, const B: usize> Reduce<UintAccumulator<A, B>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: UintAccumulator<A, B>) -> Self::Output { PrimeValue::new(self.params().from_canonical(&self.params().remainder(&input))) }
        }
        impl<$($generic)*, const A: usize, const B: usize> Reduce<ZAccumulator<A, B>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: ZAccumulator<A, B>) -> Self::Output {
                let magnitude = PrimeValue::new(self.params().from_canonical(&self.params().remainder(&input.unsigned_abs())));
                Self::Output::ct_select(&magnitude, &self.neg(&magnitude), input.is_negative_ct())
            }
        }
        impl<$($generic)*, const A: usize, const B: usize> Reduce<UintProduct<A, B>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: UintProduct<A, B>) -> Self::Output { self.reduce(UintAccumulator::from(input)) }
        }
        impl<$($generic)*, const A: usize, const B: usize> Reduce<ZProduct<A, B>> for $provider {
            type Output = PrimeValue<$id, L>;
            fn reduce(&self, input: ZProduct<A, B>) -> Self::Output { self.reduce(ZAccumulator::from(input)) }
        }
    };
}
impl_prime!([const L: usize] FpCtx<L>, RuntimePrime);
impl_prime!([P: PrimeSpec<L>, const L: usize] StaticFpOps<P, L>, StaticPrime<P>);

macro_rules! native_operand {
    ([$($generic:tt)*] $provider:ty, $id:ty, $native:ty, $n:expr, $convert:expr) => {
        impl<$($generic)*> IntegerEmbedding<$native> for $provider {
            fn from_integer(&self, value: &$native) -> Self::Elem { self.from_integer(&($convert)(*value)) }
        }
        impl<$($generic)*> WideMul<PrimeValue<$id, L>, $native> for $provider {
            type Product = PrimeLinearProduct<$id, L, $n>;
            fn mul_wide(&self, lhs: &PrimeValue<$id, L>, rhs: &$native) -> Self::Product { self.mul_wide(lhs, &($convert)(*rhs)) }
        }
        impl<$($generic)*> BatchMulAcc<PrimeValue<$id, L>, $native> for $provider {
            type Accumulator = PrimeLinearAcc<$id, L, $n>;
            #[inline(always)]
            fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &PrimeValue<$id, L>, rhs: &$native) { acc.accumulate(lhs, &($convert)(*rhs)); }
            fn batch_mul_acc(&self, lhs: &[PrimeValue<$id, L>], rhs: &[$native]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
            }
            fn batch_mul_acc_map(&self, len: usize, mut term: impl FnMut(usize) -> (PrimeValue<$id, L>, $native)) -> Self::Accumulator {
                let mut acc = UintAccumulator::ZERO;
                for i in 0..len { let (lhs, rhs) = term(i); acc.mac(&lhs.words, &($convert)(rhs)); }
                ScaledAccumulator::new(acc)
            }
        }
    };
}
macro_rules! native_signed_operand {
    ([$($generic:tt)*] $provider:ty, $id:ty, $native:ty, $n:expr, $convert:expr) => {
        impl<$($generic)*> IntegerEmbedding<$native> for $provider {
            fn from_integer(&self, value: &$native) -> Self::Elem {
                self.from_integer(&($convert)(*value))
            }
        }
        impl<$($generic)*> WideMul<PrimeValue<$id, L>, $native> for $provider {
            type Product = PrimeSignedLinearProduct<$id, L, $n>;
            fn mul_wide(&self, lhs: &PrimeValue<$id, L>, rhs: &$native) -> Self::Product {
                self.mul_wide(lhs, &($convert)(*rhs))
            }
        }
        impl<$($generic)*> BatchMulAcc<PrimeValue<$id, L>, $native> for $provider {
            type Accumulator = PrimeSignedLinearAcc<$id, L, $n>;
            #[inline(always)]
            fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &PrimeValue<$id, L>, rhs: &$native) { acc.accumulate(lhs, &($convert)(*rhs)); }
            fn batch_mul_acc(&self, lhs: &[PrimeValue<$id, L>], rhs: &[$native]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len(), "batch input lengths differ");
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], ($convert)(rhs[i])))
            }
            fn batch_mul_acc_map(&self, len: usize, mut term: impl FnMut(usize) -> (PrimeValue<$id, L>, $native)) -> Self::Accumulator {
                self.batch_mul_acc_map(len, |i| {
                    let (lhs, rhs) = term(i);
                    (lhs, ($convert)(rhs))
                })
            }
        }
    };
}
macro_rules! native_operands {
    ([$($generic:tt)*] $provider:ty, $id:ty) => {
        native_signed_operand!([$($generic)*] $provider, $id, i32, 1, |v: i32| Z::from_twos_complement_words([v as i64 as u64]));
        native_signed_operand!([$($generic)*] $provider, $id, i64, 1, |v: i64| Z::from_twos_complement_words([v as u64]));
        native_signed_operand!([$($generic)*] $provider, $id, i128, 2, |v: i128| Z::from_twos_complement_words([v as u64, (v >> 64) as u64]));
        native_operand!([$($generic)*] $provider, $id, u32, 1, |v: u32| Uint::from_words([v as u64]));
        native_operand!([$($generic)*] $provider, $id, u64, 1, |v: u64| Uint::from_words([v]));
        native_operand!([$($generic)*] $provider, $id, u128, 2, |v: u128| Uint::from_words([v as u64, (v >> 64) as u64]));
        native_operand!([$($generic)*] $provider, $id, Bit, 1, |v: Bit| Uint::from_words([v.as_u64()]));
    };
}
native_operands!([const L: usize] FpCtx<L>, RuntimePrime);
native_operands!([P: PrimeSpec<L>, const L: usize] StaticFpOps<P, L>, StaticPrime<P>);

/// Existing PCS prime, 2^100 - 15.
pub const Q100: u128 = (1u128 << 100) - 15;
crate::define_prime_field! {
    pub Q100Prime {
        limbs: 2,
        modulus: [Q100 as u64, (Q100 >> 64) as u64],
        element: Q100Element,
        context: Q100Field,
    }
}

impl<const L: usize> PartialEq for FpCtx<L> {
    fn eq(&self, other: &Self) -> bool {
        self.modulus() == other.modulus()
    }
}
impl<const L: usize> Eq for FpCtx<L> {}
