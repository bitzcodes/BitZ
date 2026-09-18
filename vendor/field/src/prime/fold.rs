use super::*;
use crate::integer::product::Words;

/// The native inputs remain integers until they reach their consuming MAC.
/// `plain` selects the representation needed by the next fold, outside the loop.
#[inline(always)]
pub(super) fn fold_unsigned<const L: usize, const N: usize>(
    parameters: &PrimeParameters<L>,
    complement: &Uint<L>,
    challenge: &Uint<L>,
    left: &Uint<N>,
    right: &Uint<N>,
    plain: bool,
) -> Uint<L> {
    let mut acc = UintAccumulator::<L, N>::ZERO;
    acc.mac(complement, left);
    acc.mac(challenge, right);
    if plain && N < L {
        // Each weight is below p; two N-limb integers give a sum below pR
        // when N < L. A single REDC therefore yields the canonical fold.
        let product = UintProduct {
            low: core::array::from_fn(|i| if i < acc.len() { acc.word(i) } else { 0 }),
            high: core::array::from_fn(|i| {
                if L + i < acc.len() {
                    acc.word(L + i)
                } else {
                    0
                }
            }),
        };
        parameters.redc(product)
    } else {
        let value = parameters.remainder(&acc);
        if plain {
            parameters.to_canonical(&value)
        } else {
            value
        }
    }
}

pub(super) fn reduce_signed<const L: usize, const N: usize>(
    parameters: &PrimeParameters<L>,
    input: &ZAccumulator<L, N>,
    plain: bool,
) -> Uint<L> {
    let magnitude = parameters.remainder(&input.unsigned_abs());
    let reduced = Uint::ct_select(
        &magnitude,
        &parameters.neg(&magnitude),
        input.is_negative_ct(),
    );
    if plain {
        parameters.to_canonical(&reduced)
    } else {
        reduced
    }
}

pub(super) fn fold_signed<const L: usize, const N: usize>(
    parameters: &PrimeParameters<L>,
    complement: &Uint<L>,
    challenge: &Uint<L>,
    left: &Z<N>,
    right: &Z<N>,
    plain: bool,
) -> Uint<L> {
    let mut acc = ZAccumulator::ZERO;
    acc.add_product(IntegerOps.mul_wide(complement, left));
    acc.add_product(IntegerOps.mul_wide(challenge, right));
    reduce_signed(parameters, &acc, plain)
}

macro_rules! fold_unsigned_source {
    ([$($generic:tt)*] $provider:ty, $id:ty, $src:ty, $convert:expr) => {
        impl<$($generic)*> FoldPairs<$src, PrimeValue<$id,L>> for $provider {
            fn fold_pairs_into(&self, src:&[$src], out:&mut[PrimeValue<$id,L>], challenge:&Self::Elem) {
                assert_eq!(src.len(),out.len().checked_mul(2).expect("fold extent overflow"));
                self.fold_pairs_map_into(|i|src[i],out,challenge);
            }
            fn fold_pairs_map_into(&self,  mut read:impl FnMut(usize)->$src, out:&mut[PrimeValue<$id,L>], challenge:&Self::Elem) {
                out.len().checked_mul(2).expect("fold extent overflow");
                let start=0;
                let complement=self.params().sub(&self.params().one,&challenge.words);
                for(i,dst) in out.iter_mut().enumerate() {
                    let left=($convert)(read(start+2*i));
                    let right=($convert)(read(start+2*i+1));
                    *dst=PrimeValue::new(fold_unsigned(self.params(),&complement,&challenge.words,&left,&right,false));
                }
            }
        }
        impl<$($generic)*> FoldPairs<$src, Uint<L>> for $provider {
            fn fold_pairs_into(&self, src:&[$src], out:&mut[Uint<L>], challenge:&Self::Elem) {
                assert_eq!(src.len(),out.len().checked_mul(2).expect("fold extent overflow"));
                self.fold_pairs_map_into(|i|src[i],out,challenge);
            }
            fn fold_pairs_map_into(&self,  mut read:impl FnMut(usize)->$src, out:&mut[Uint<L>], challenge:&Self::Elem) {
                out.len().checked_mul(2).expect("fold extent overflow");
                let start=0;
                let complement=self.params().sub(&self.params().one,&challenge.words);
                for(i,dst) in out.iter_mut().enumerate() {
                    let left=($convert)(read(start+2*i));
                    let right=($convert)(read(start+2*i+1));
                    *dst=fold_unsigned(self.params(),&complement,&challenge.words,&left,&right,true);
                }
            }
        }
    };
}

fn source_start(source_len: usize, pair_offset: usize, output_len: usize) -> usize {
    assert_eq!(source_len % 2, 0, "fold input length must be even");
    let start = pair_offset.checked_mul(2).expect("fold offset overflow");
    let end = output_len
        .checked_mul(2)
        .and_then(|n| start.checked_add(n))
        .expect("fold range overflow");
    assert!(end <= source_len, "fold range exceeds input");
    start
}

macro_rules! fold_source {
    ([$($generic:tt)*] $provider:ty, $id:ty, $src:ty, $project:expr) => {
        impl<$($generic)*> FoldPairs<$src, PrimeValue<$id,L>> for $provider {
            fn fold_pairs_into(&self, src: &[$src], dst: &mut [PrimeValue<$id,L>], challenge: &Self::Elem) {
                assert_eq!(src.len(),dst.len().checked_mul(2).expect("fold extent overflow"));
                self.fold_pairs_map_into( |i| src[i], dst, challenge);
            }
            fn fold_pairs_map_into(&self,  mut read: impl FnMut(usize)->$src, dst: &mut [PrimeValue<$id,L>], challenge: &Self::Elem) {
                dst.len().checked_mul(2).expect("fold extent overflow");
                let start=0;
                for (i,out) in dst.iter_mut().enumerate() {
                    let left=($project)(self, read(start+2*i));
                    let right=($project)(self, read(start+2*i+1));
                    *out=self.add(&left,&self.mul(challenge,&self.sub(&right,&left)));
                }
            }
        }
        impl<$($generic)*> FoldPairs<$src, Uint<L>> for $provider {
            fn fold_pairs_into(&self, src: &[$src], dst: &mut [Uint<L>], challenge: &Self::Elem) {
                assert_eq!(src.len(),dst.len().checked_mul(2).expect("fold extent overflow"));
                self.fold_pairs_map_into( |i| src[i], dst, challenge);
            }
            fn fold_pairs_map_into(&self,  mut read: impl FnMut(usize)->$src, dst: &mut [Uint<L>], challenge: &Self::Elem) {
                dst.len().checked_mul(2).expect("fold extent overflow");
                let start=0;
                for (i,out) in dst.iter_mut().enumerate() {
                    let left=($project)(self, read(start+2*i));
                    let right=($project)(self, read(start+2*i+1));
                    let folded=self.add(&left,&self.mul(challenge,&self.sub(&right,&left)));
                    *out=self.to_integer(&folded);
                }
            }
        }
    };
}
macro_rules! fold_sources {
    ([$($generic:tt)*] $provider:ty, $id:ty) => {
        fold_source!([$($generic)*] $provider,$id,PrimeValue<$id,L>,|_: &$provider,v: PrimeValue<$id,L>|v);
        fold_unsigned_source!([$($generic)*] $provider,$id,u32,|v:u32|Uint::from_words([v as u64]));
        fold_unsigned_source!([$($generic)*] $provider,$id,u64,|v:u64|Uint::from_words([v]));
        fold_unsigned_source!([$($generic)*] $provider,$id,u128,|v:u128|Uint::from_words([v as u64,(v>>64)as u64]));
        fold_unsigned_source!([$($generic)*] $provider,$id,Bit,|v:Bit|Uint::from_words([v.as_u64()]));
        fold_unsigned_source!([$($generic)*, const N:usize] $provider,$id,Uint<N>,|v:Uint<N>|v);
        impl<$($generic)*, const N:usize> FoldPairs<Z<N>, PrimeValue<$id,L>> for $provider {
            fn fold_pairs_into(&self,src:&[Z<N>],out:&mut[PrimeValue<$id,L>],challenge:&Self::Elem) {
                assert_eq!(src.len(),out.len().checked_mul(2).expect("fold extent overflow"));
                self.fold_pairs_map_into(|i|src[i],out,challenge);
            }
            fn fold_pairs_map_into(&self,mut read:impl FnMut(usize)->Z<N>,out:&mut[PrimeValue<$id,L>],challenge:&Self::Elem) {
                out.len().checked_mul(2).expect("fold extent overflow");
                let start=0;
                let complement=self.params().sub(&self.params().one,&challenge.words);
                for(i,dst) in out.iter_mut().enumerate() {
                    let left=read(start+2*i); let right=read(start+2*i+1);
                    *dst=PrimeValue::new(fold_signed(self.params(),&complement,&challenge.words,&left,&right,false));
                }
            }
        }
        impl<$($generic)*, const N:usize> FoldPairs<Z<N>, Uint<L>> for $provider {
            fn fold_pairs_into(&self,src:&[Z<N>],out:&mut[Uint<L>],challenge:&Self::Elem) {
                assert_eq!(src.len(),out.len().checked_mul(2).expect("fold extent overflow"));
                self.fold_pairs_map_into(|i|src[i],out,challenge);
            }
            fn fold_pairs_map_into(&self,mut read:impl FnMut(usize)->Z<N>,out:&mut[Uint<L>],challenge:&Self::Elem) {
                out.len().checked_mul(2).expect("fold extent overflow");
                let start=0;
                let complement=self.params().sub(&self.params().one,&challenge.words);
                for(i,dst) in out.iter_mut().enumerate() {
                    let left=read(start+2*i); let right=read(start+2*i+1);
                    *dst=fold_signed(self.params(),&complement,&challenge.words,&left,&right,true);
                }
            }
        }
        impl<$($generic)*> $provider {
            /// Writes the first `half` entries; the caller retains/truncates the vector.
            pub fn fold_in_place(&self, values: &mut [PrimeValue<$id,L>], challenge: &PrimeValue<$id,L>, half:usize) {
                source_start(values.len(),0,half);
                for i in 0..half {
                    let left=values[2*i]; let right=values[2*i+1];
                    values[i]=self.add(&left,&self.mul(challenge,&self.sub(&right,&left)));
                }
            }
            /// Canonical integer storage stays canonical without an intermediate vector.
            pub fn fold_plain_in_place(&self, values:&mut[Uint<L>],challenge:&PrimeValue<$id,L>,half:usize) {
                source_start(values.len(),0,half);
                for i in 0..half {
                    let left=values[2*i]; let right=values[2*i+1];
                    debug_assert!(left.ct_lt(self.modulus()).declassify());
                    debug_assert!(right.ct_lt(self.modulus()).declassify());
                    let difference=self.params().sub(&right,&left);
                    // Montgomery × plain, followed by REDC, is plain again.
                    values[i]=self.params().add(&left,&self.params().mul(&challenge.words,&difference));
                }
            }
        }
    };
}
fold_sources!([const L:usize] FpCtx<L>, RuntimePrime);
fold_sources!([P:PrimeSpec<L>,const L:usize] StaticFpOps<P,L>, StaticPrime<P>);
