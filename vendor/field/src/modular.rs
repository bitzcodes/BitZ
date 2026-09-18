//! Canonical residues modulo an arbitrary public modulus greater than one.

use crate::integer::product::Words;
use crate::integer::{IntegerOps, UintAccumulator, UintProduct, ZAccumulator, ZProduct};
use crate::traits::*;
use crate::{CtEq, CtMask, CtOrd, CtSelect, CtValue, Uint, Z};

pub(crate) mod reduction;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextError {
    InvalidModulus,
    ZeroDivisor,
    EvenModulus,
}
impl core::fmt::Display for ContextError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "arithmetic context: {self:?}")
    }
}
impl std::error::Error for ContextError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Residue<const L: usize>(pub(crate) Uint<L>);
impl<const L: usize> CtEq for Residue<L> {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        self.0.ct_eq(&rhs.0)
    }
    fn ct_is_zero(&self) -> CtMask {
        self.0.ct_is_zero()
    }
}
impl<const L: usize> CtSelect for Residue<L> {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self(Uint::ct_select(&a.0, &b.0, mask))
    }
}

#[derive(Clone, Debug)]
pub struct ModRingCtx<const L: usize> {
    modulus: Uint<L>,
    reduction: reduction::Barrett<L>,
}
impl<const L: usize> ModRingCtx<L> {
    pub fn new(modulus: Uint<L>) -> Result<Self, ContextError> {
        if modulus.ct_le(&Uint::ONE).declassify() {
            return Err(ContextError::InvalidModulus);
        }
        Ok(Self {
            reduction: reduction::Barrett::new(&modulus),
            modulus,
        })
    }
    pub fn modulus(&self) -> &Uint<L> {
        &self.modulus
    }
    pub fn to_integer(&self, value: &Residue<L>) -> Uint<L> {
        value.0
    }
    pub fn from_canonical_ct(&self, value: &Uint<L>) -> CtValue<Residue<L>> {
        let valid = value.ct_lt(&self.modulus);
        CtValue::new(Residue(Uint::ct_select(&Uint::ZERO, value, valid)), valid)
    }
    pub(crate) fn remainder(&self, input: &(impl Words + ?Sized)) -> Uint<L> {
        self.reduction.remainder(input, &self.modulus)
    }

    /// Fixed-iteration extended Euclid, retaining coefficients modulo the modulus.
    /// Nonunits return a masked invalid value. This portable fallback also supports
    /// even moduli; prepared odd-modulus kernels may specialize it separately.
    pub fn inverse_ct(&self, value: &Residue<L>) -> CtValue<Residue<L>> {
        let (mut r0, mut r1) = (self.modulus, value.0);
        let (mut t0, mut t1) = (self.zero(), self.one());
        // Euclid takes fewer than twice the input bit width in division steps.
        for _ in 0..2 * 64 * L {
            let active = !r1.ct_is_zero();
            let (q, remainder) = r0.div_rem_words(&r1);
            let q = Residue(self.remainder(&q));
            let next = self.sub(&t0, &self.mul(&q, &t1));
            r0 = Uint::ct_select(&r0, &r1, active);
            r1 = Uint::ct_select(&r1, &remainder, active);
            t0 = Residue::ct_select(&t0, &t1, active);
            t1 = Residue::ct_select(&t1, &next, active);
        }
        let valid = r0.ct_eq(&Uint::ONE);
        CtValue::new(Residue::ct_select(&self.zero(), &t0, valid), valid)
    }
}
impl<const L: usize> RingOps for ModRingCtx<L> {
    type Elem = Residue<L>;
    fn zero(&self) -> Self::Elem {
        Residue(Uint::ZERO)
    }
    fn one(&self) -> Self::Elem {
        Residue(Uint::ONE)
    }
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        let (sum, carry) = a.0.adc(&b.0);
        let (reduced, borrow) = sum.sbb(&self.modulus);
        Residue(Uint::ct_select(
            &sum,
            &reduced,
            CtMask::from_lsb(carry | (borrow ^ 1)),
        ))
    }
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        let (difference, borrow) = a.0.sbb(&b.0);
        let corrected = difference.adc(&self.modulus).0;
        Residue(Uint::ct_select(
            &difference,
            &corrected,
            CtMask::from_lsb(borrow),
        ))
    }
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        self.sub(&self.zero(), a)
    }
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        Residue(self.remainder(&IntegerOps.mul_wide(&a.0, &b.0)))
    }
}
impl<const L: usize, const N: usize> IntegerEmbedding<Uint<N>> for ModRingCtx<L> {
    fn from_integer(&self, value: &Uint<N>) -> Self::Elem {
        Residue(self.remainder(value))
    }
}
impl<const L: usize, const N: usize> IntegerEmbedding<Z<N>> for ModRingCtx<L> {
    fn from_integer(&self, value: &Z<N>) -> Self::Elem {
        let magnitude = self.from_integer(&value.unsigned_abs());
        Residue::ct_select(&magnitude, &self.neg(&magnitude), value.is_negative_ct())
    }
}
impl<const L: usize> IntegerEmbedding<crate::UintRef<'_>> for ModRingCtx<L> {
    fn from_integer(&self, value: &crate::UintRef<'_>) -> Self::Elem {
        Residue(self.remainder(value.as_words()))
    }
}
impl<const L: usize> IntegerEmbedding<crate::ZRef<'_>> for ModRingCtx<L> {
    fn from_integer(&self, value: &crate::ZRef<'_>) -> Self::Elem {
        let input = crate::integer::product::SignedMagnitude::new(value.as_words());
        let magnitude = Residue(self.remainder(&input));
        Residue::ct_select(&magnitude, &self.neg(&magnitude), input.negative())
    }
}
macro_rules! ring_native {
    ($ty:ty,$convert:expr) => {
        impl<const L: usize> IntegerEmbedding<$ty> for ModRingCtx<L> {
            fn from_integer(&self, value: &$ty) -> Self::Elem {
                self.from_integer(&($convert)(*value))
            }
        }
    };
}
ring_native!(u32, |v: u32| Uint::from_words([v as u64]));
ring_native!(u64, |v: u64| Uint::from_words([v]));
ring_native!(u128, |v: u128| Uint::from_words([
    v as u64,
    (v >> 64) as u64
]));
impl<const L: usize, const A: usize, const B: usize> Reduce<UintProduct<A, B>> for ModRingCtx<L> {
    type Output = Residue<L>;
    fn reduce(&self, input: UintProduct<A, B>) -> Self::Output {
        Residue(self.remainder(&input))
    }
}
impl<const L: usize, const A: usize, const B: usize> Reduce<UintAccumulator<A, B>>
    for ModRingCtx<L>
{
    type Output = Residue<L>;
    fn reduce(&self, input: UintAccumulator<A, B>) -> Self::Output {
        Residue(self.remainder(&input))
    }
}
impl<const L: usize, const A: usize, const B: usize> Reduce<ZProduct<A, B>> for ModRingCtx<L> {
    type Output = Residue<L>;
    fn reduce(&self, input: ZProduct<A, B>) -> Self::Output {
        self.reduce(ZAccumulator::from(input))
    }
}
impl<const L: usize, const A: usize, const B: usize> Reduce<ZAccumulator<A, B>> for ModRingCtx<L> {
    type Output = Residue<L>;
    fn reduce(&self, input: ZAccumulator<A, B>) -> Self::Output {
        let magnitude = Residue(self.remainder(&input.unsigned_abs()));
        Residue::ct_select(&magnitude, &self.neg(&magnitude), input.is_negative_ct())
    }
}
impl<const L: usize> crate::CanonicalCodec<Residue<L>> for ModRingCtx<L> {
    fn encoded_len(&self) -> usize {
        L * 8
    }
    fn encode_into(&self, value: &Residue<L>, out: &mut [u8]) {
        crate::encoding::encode_words(&value.0, out);
    }
    fn decode_ct(&self, input: &[u8]) -> Result<CtValue<Residue<L>>, crate::DecodeError> {
        Ok(self.from_canonical_ct(&crate::encoding::decode_words(input)?))
    }
}
