//! Native inner-product rounds and folds. The integer witness is borrowed
//! until its consuming operation; folded witnesses use canonical word storage.
use super::*;
use crate::DotProductKernels;

fn fold_shape(input: usize, weights_out: usize, values_out: usize) {
    assert_eq!(weights_out, values_out, "fold outputs differ");
    assert_eq!(weights_out % 2, 0, "fold output must contain pairs");
    assert_eq!(
        input,
        weights_out.checked_mul(2).expect("fold shape overflow")
    );
}

fn native_pair<Id, const L: usize, const N: usize>(
    parameters: &PrimeParameters<L>,
    weights: &[PrimeValue<Id, L>],
    mut read: impl FnMut(usize) -> Uint<N>,
) -> [PrimeValue<Id, L>; 2] {
    assert_eq!(weights.len() % 2, 0, "dot input must contain pairs");
    let mut acc = [UintAccumulator::<L, N>::ZERO; 2];
    for (i, m) in weights.chunks_exact(2).enumerate() {
        let a = read(2 * i);
        let b = read(2 * i + 1);
        acc[0].mac(&m[0].words, &a);
        let (difference, borrow) = b.sbb(&a);
        let negative = CtMask::from_lsb(borrow);
        let magnitude = Uint::ct_select(&difference, &difference.wrapping_neg(), negative);
        let delta = parameters.sub(&m[1].words, &m[0].words);
        let signed_delta = Uint::ct_select(&delta, &parameters.neg(&delta), negative);
        acc[1].mac(&signed_delta, &magnitude);
    }
    acc.map(|a| PrimeValue::new(parameters.remainder(&a)))
}

fn native_fold<Id, const L: usize, const N: usize>(
    parameters: &PrimeParameters<L>,
    weights: &[PrimeValue<Id, L>],
    mut read: impl FnMut(usize) -> Uint<N>,
    weights_out: &mut [PrimeValue<Id, L>],
    values_out: &mut [Uint<L>],
    challenge: &PrimeValue<Id, L>,
) -> [PrimeValue<Id, L>; 2] {
    fold_shape(weights.len(), weights_out.len(), values_out.len());
    let complement = parameters.sub(&parameters.one, &challenge.words);
    let mut acc = [UintAccumulator::<L, L>::ZERO; 2];
    for (i, ((m, mo), wo)) in weights
        .chunks_exact(4)
        .zip(weights_out.chunks_exact_mut(2))
        .zip(values_out.chunks_exact_mut(2))
        .enumerate()
    {
        for j in 0..2 {
            let a = read(4 * i + 2 * j);
            let b = read(4 * i + 2 * j + 1);
            mo[j] = PrimeValue::new(parameters.add(
                &m[2 * j].words,
                &parameters.mul(
                    &challenge.words,
                    &parameters.sub(&m[2 * j + 1].words, &m[2 * j].words),
                ),
            ));
            wo[j] =
                super::fold::fold_unsigned(parameters, &complement, &challenge.words, &a, &b, true);
        }
        acc[0].mac(&mo[0].words, &wo[0]);
        acc[1].mac(
            &parameters.sub(&mo[1].words, &mo[0].words),
            &parameters.sub(&wo[1], &wo[0]),
        );
    }
    acc.map(|a| PrimeValue::new(parameters.remainder(&a)))
}

/// Subsequent rounds know their integer-stored witness is canonical. Preserve
/// that knowledge, so each witness fold needs one REDC and no re-embedding.
fn plain_fold<Id, const L: usize>(
    parameters: &PrimeParameters<L>,
    weights: &[PrimeValue<Id, L>],
    values: &[Uint<L>],
    weights_out: &mut [PrimeValue<Id, L>],
    values_out: &mut [Uint<L>],
    challenge: &PrimeValue<Id, L>,
) -> [PrimeValue<Id, L>; 2] {
    assert_eq!(weights.len(), values.len(), "dot inputs differ");
    fold_shape(weights.len(), weights_out.len(), values_out.len());
    let mut acc = [UintAccumulator::<L, L>::ZERO; 2];
    for (((m, w), mo), wo) in weights
        .chunks_exact(4)
        .zip(values.chunks_exact(4))
        .zip(weights_out.chunks_exact_mut(2))
        .zip(values_out.chunks_exact_mut(2))
    {
        for j in 0..2 {
            debug_assert!(w[2 * j].ct_lt(&parameters.modulus).declassify());
            debug_assert!(w[2 * j + 1].ct_lt(&parameters.modulus).declassify());
            mo[j] = PrimeValue::new(parameters.add(
                &m[2 * j].words,
                &parameters.mul(
                    &challenge.words,
                    &parameters.sub(&m[2 * j + 1].words, &m[2 * j].words),
                ),
            ));
            wo[j] = parameters.add(
                &w[2 * j],
                &parameters.mul(&challenge.words, &parameters.sub(&w[2 * j + 1], &w[2 * j])),
            );
        }
        acc[0].mac(&mo[0].words, &wo[0]);
        acc[1].mac(
            &parameters.sub(&mo[1].words, &mo[0].words),
            &parameters.sub(&wo[1], &wo[0]),
        );
    }
    acc.map(|a| PrimeValue::new(parameters.remainder(&a)))
}

fn signed_pair<Id, const L: usize, const N: usize>(
    parameters: &PrimeParameters<L>,
    weights: &[PrimeValue<Id, L>],
    mut read: impl FnMut(usize) -> Z<N>,
) -> [PrimeValue<Id, L>; 2] {
    assert_eq!(weights.len() % 2, 0, "dot input must contain pairs");
    let mut acc = [ZAccumulator::<L, N>::ZERO; 2];
    for (i, m) in weights.chunks_exact(2).enumerate() {
        let a = read(2 * i);
        let b = read(2 * i + 1);
        acc[0].add_product(IntegerOps.mul_wide(&m[0].words, &a));
        let delta = parameters.sub(&m[1].words, &m[0].words);
        // Avoid forming b-a in the operand width: MIN/MAX needs another bit.
        acc[1].add_product(IntegerOps.mul_wide(&delta, &b));
        acc[1].add_product(IntegerOps.mul_wide(&parameters.neg(&delta), &a));
    }
    acc.map(|a| PrimeValue::new(super::fold::reduce_signed(parameters, &a, false)))
}

fn field_pair<Id, const L: usize>(
    parameters: &PrimeParameters<L>,
    weights: &[PrimeValue<Id, L>],
    mut read: impl FnMut(usize) -> PrimeValue<Id, L>,
) -> [PrimeValue<Id, L>; 2] {
    assert_eq!(weights.len() % 2, 0, "dot input must contain pairs");
    let mut acc = [UintAccumulator::<L, L>::ZERO; 2];
    for (i, m) in weights.chunks_exact(2).enumerate() {
        let a = read(2 * i);
        let b = read(2 * i + 1);
        acc[0].mac(&m[0].words, &a.words);
        acc[1].mac(
            &parameters.sub(&m[1].words, &m[0].words),
            &parameters.sub(&b.words, &a.words),
        );
    }
    acc.map(|a| PrimeValue::new(parameters.reduce_product_acc(&a)))
}

/// The projection callback returns one folded canonical pair. It consumes only
/// that pair's borrowed inputs; the fold and next message share one traversal.
fn projected_fold<Id, const L: usize>(
    parameters: &PrimeParameters<L>,
    weights: &[PrimeValue<Id, L>],
    mut fold: impl FnMut(usize, &Uint<L>) -> Uint<L>,
    weights_out: &mut [PrimeValue<Id, L>],
    values_out: &mut [Uint<L>],
    challenge: &PrimeValue<Id, L>,
) -> [PrimeValue<Id, L>; 2] {
    fold_shape(weights.len(), weights_out.len(), values_out.len());
    let complement = parameters.sub(&parameters.one, &challenge.words);
    let mut acc = [UintAccumulator::<L, L>::ZERO; 2];
    for (i, ((m, mo), wo)) in weights
        .chunks_exact(4)
        .zip(weights_out.chunks_exact_mut(2))
        .zip(values_out.chunks_exact_mut(2))
        .enumerate()
    {
        for j in 0..2 {
            mo[j] = PrimeValue::new(parameters.add(
                &m[2 * j].words,
                &parameters.mul(
                    &challenge.words,
                    &parameters.sub(&m[2 * j + 1].words, &m[2 * j].words),
                ),
            ));
            wo[j] = fold(4 * i + 2 * j, &complement);
        }
        acc[0].mac(&mo[0].words, &wo[0]);
        acc[1].mac(
            &parameters.sub(&mo[1].words, &mo[0].words),
            &parameters.sub(&wo[1], &wo[0]),
        );
    }
    acc.map(|a| PrimeValue::new(parameters.remainder(&a)))
}

macro_rules! projected_source {
    ([$($generic:tt)*] $provider:ty, $id:ty, $src:ty, $pair:ident, $fold:expr) => {
        impl<$($generic)*> DotProductKernels<$src, Uint<L>> for $provider {
            fn dot_pair_round(&self,weights:&[Self::Elem],values:&[$src])->[Self::Elem;2] {
                assert_eq!(weights.len(),values.len(),"dot inputs differ");
                $pair(self.params(),weights,|i|values[i])
            }
            fn dot_pair_round_map(&self,weights:&[Self::Elem],read:impl FnMut(usize)->$src)->[Self::Elem;2] {
                $pair(self.params(),weights,read)
            }
            fn dot_fold_round_into(&self,weights:&[Self::Elem],values:&[$src],weights_out:&mut[Self::Elem],values_out:&mut[Uint<L>],challenge:&Self::Elem)->[Self::Elem;2] {
                assert_eq!(weights.len(),values.len(),"dot inputs differ");
                self.dot_fold_round_map_into(weights,|i|values[i],weights_out,values_out,challenge)
            }
            fn dot_fold_round_map_into(&self,weights:&[Self::Elem],mut read:impl FnMut(usize)->$src,weights_out:&mut[Self::Elem],values_out:&mut[Uint<L>],challenge:&Self::Elem)->[Self::Elem;2] {
                projected_fold(self.params(),weights,|i,complement|{
                    let a=read(i); let b=read(i+1);
                    ($fold)(self.params(),complement,&challenge.words,a,b)
                },weights_out,values_out,challenge)
            }
        }
    };
}

macro_rules! native_source {
    ([$($generic:tt)*] $provider:ty, $id:ty, $src:ty, $convert:expr) => {
        impl<$($generic)*> DotProductKernels<$src, Uint<L>> for $provider {
            fn dot_pair_round(&self, weights:&[Self::Elem], values:&[$src]) -> [Self::Elem;2] {
                assert_eq!(weights.len(),values.len(),"dot inputs differ");
                self.dot_pair_round_map(weights,|i|values[i])
            }
            fn dot_pair_round_map(&self, weights:&[Self::Elem], mut read:impl FnMut(usize)->$src) -> [Self::Elem;2] {
                native_pair(self.params(),weights,|i|($convert)(read(i)))
            }
            fn dot_fold_round_into(&self, weights:&[Self::Elem], values:&[$src], weights_out:&mut[Self::Elem], values_out:&mut[Uint<L>], challenge:&Self::Elem)->[Self::Elem;2] {
                assert_eq!(weights.len(),values.len(),"dot inputs differ");
                self.dot_fold_round_map_into(weights,|i|values[i],weights_out,values_out,challenge)
            }
            fn dot_fold_round_map_into(&self,weights:&[Self::Elem],mut read:impl FnMut(usize)->$src,weights_out:&mut[Self::Elem],values_out:&mut[Uint<L>],challenge:&Self::Elem)->[Self::Elem;2] {
                native_fold(self.params(),weights,|i|($convert)(read(i)),weights_out,values_out,challenge)
            }
        }
    };
}
macro_rules! sources {
    ([$($generic:tt)*] $provider:ty,$id:ty) => {
        native_source!([$($generic)*] $provider,$id,u32,|v:u32|Uint::from_words([v as u64]));
        native_source!([$($generic)*] $provider,$id,u64,|v:u64|Uint::from_words([v]));
        native_source!([$($generic)*] $provider,$id,u128,|v:u128|Uint::from_words([v as u64,(v>>64)as u64]));
        native_source!([$($generic)*] $provider,$id,Bit,|v:Bit|Uint::from_words([v.as_u64()]));
        native_source!([$($generic)*, const N:usize] $provider,$id,Uint<N>,|v:Uint<N>|v);
        projected_source!([$($generic)*,const N:usize] $provider,$id,Z<N>,signed_pair,
            |p:&PrimeParameters<L>,c:&Uint<L>,r:&Uint<L>,a:Z<N>,b:Z<N>|super::fold::fold_signed(p,c,r,&a,&b,true));
        projected_source!([$($generic)*] $provider,$id,PrimeValue<$id,L>,field_pair,
            |p:&PrimeParameters<L>,_:&Uint<L>,r:&Uint<L>,a:PrimeValue<$id,L>,b:PrimeValue<$id,L>|{
                p.to_canonical(&p.add(&a.words,&p.mul(r,&p.sub(&b.words,&a.words))))
            });
        impl<$($generic)*> $provider {
            /// Values must already be canonical residues in [0,p). Public
            /// shapes are checked in release; canonicality is a debug contract.
            pub fn dot_fold_plain_round_into(&self,weights:&[PrimeValue<$id,L>],values:&[Uint<L>],weights_out:&mut[PrimeValue<$id,L>],values_out:&mut[Uint<L>],challenge:&PrimeValue<$id,L>)->[PrimeValue<$id,L>;2] {
                plain_fold(self.params(),weights,values,weights_out,values_out,challenge)
            }
        }
    };
}
sources!([const L:usize] FpCtx<L>,RuntimePrime);
sources!([P:PrimeSpec<L>,const L:usize] StaticFpOps<P,L>,StaticPrime<P>);
