//! Public-coefficient word weights with one reduction per mixed dot product.
use super::*;
use crate::integer::product::Words;

fn limb_weights<const L: usize, const N: usize>(
    p: &PrimeParameters<L>,
    radix: &Uint<L>,
    c: &Fp<L>,
) -> [Uint<L>; N] {
    let mut weight = p.from_canonical(&c.words);
    core::array::from_fn(|_| {
        let result = weight;
        weight = p.mul(&weight, radix);
        result
    })
}

/// Weights contain one extra Montgomery factor: REDC of their word products
/// is already a Montgomery field element, rather than a canonical integer.
pub struct PreparedWordWeights<const L: usize, const N: usize> {
    positive: Vec<[Uint<L>; N]>,
    negative: Vec<[Uint<L>; N]>,
    negative_radix: Vec<Uint<L>>,
}

impl<const L: usize, const N: usize> PreparedWordWeights<L, N> {
    fn new(field: &FpCtx<L>, coefficients: &[Fp<L>]) -> Self {
        assert!(N > 0 && N <= 256, "unsupported public operand width");
        let p = field.params();
        let radix = field.from_integer(&Uint::<2>::from_words([0, 1]));
        let positive: Vec<_> = coefficients
            .iter()
            .map(|c| limb_weights(p, &radix.words, c))
            .collect();
        let negative = positive.iter().map(|w| w.map(|v| p.neg(&v))).collect();
        let negative_radix = if N > 2 {
            positive
                .iter()
                .map(|w| p.neg(&p.mul(&w[N - 1], &radix.words)))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            positive,
            negative,
            negative_radix,
        }
    }

    /// Prepare equality weights for bounded exact signed accumulation.
    /// Each accumulator may receive at most 2^32 operands, with N <= 256.
    /// Widths and the term bound must come from public input dimensions.
    pub fn signed(field: &FpCtx<L>, coefficients: &[Fp<L>]) -> Self {
        Self::new(field, coefficients)
    }

    /// Add one signed operand using a public magnitude width. The caller must
    /// prove that its magnitude fits USED words; this never inspects leading
    /// zero words to select an arithmetic schedule.
    #[inline(always)]
    pub fn accumulate_signed<const USED: usize>(
        &self,
        accumulator: &mut UintAccumulator<L, 1>,
        index: usize,
        value: Z<N>,
    ) {
        const {
            assert!(USED <= N);
        }
        let sign = value.is_negative_ct();
        if USED > 2 {
            // Low two's-complement words represent value + sign*2^(64*USED).
            // Correct the radix once, instead of negating/selecting every limb.
            for j in 0..USED {
                accumulator.mac(
                    &self.positive[index][j],
                    &Uint::from_words([value.as_words()[j]]),
                );
            }
            let correction = if USED < N {
                &self.negative[index][USED]
            } else {
                &self.negative_radix[index]
            };
            let correction = Uint::ct_select(&Uint::ZERO, correction, sign);
            accumulator.merge_assign(&UintAccumulator {
                low: *correction.as_words(),
                high: [0],
                head: 0,
            });
        } else {
            let magnitude = value.unsigned_abs();
            for j in 0..USED {
                let weight =
                    Uint::ct_select(&self.positive[index][j], &self.negative[index][j], sign);
                accumulator.mac(&weight, &Uint::from_words([magnitude.as_words()[j]]));
            }
        }
    }

    /// Reduce an accumulator whose weights were prepared by this object, at
    /// the same modulus, respecting the public term bound above. The weights
    /// have scale R^2 and the returned field element has scale R.
    #[inline(always)]
    pub fn finish(&self, field: &FpCtx<L>, accumulator: UintAccumulator<L, 1>) -> Fp<L> {
        finish_words(field.params(), accumulator)
    }
}

#[inline(always)]
fn finish_words<const L: usize>(
    p: &PrimeParameters<L>,
    accumulator: UintAccumulator<L, 1>,
) -> Fp<L> {
    // At most 2^32 operands of at most 256 words, each <2^64,
    // plus one radix correction <p: 256*(2^64-1)+1 < 2^72.
    // For L>=2, S < p*2^104 < p*R. Weights have scale R^2;
    // one REDC returns the required Montgomery scale R.
    let value = if L >= 2 {
        p.redc(UintProduct {
            low: core::array::from_fn(|i| accumulator.word(i)),
            high: core::array::from_fn(|i| {
                if L + i < accumulator.len() {
                    accumulator.word(L + i)
                } else {
                    0
                }
            }),
        })
    } else {
        p.to_canonical(&p.remainder(&accumulator))
    };
    Fp::new(value)
}

/// A fixed public fold width, bound to the context which prepared its weights.
/// The same weights support signed and unsigned values of the declared width.
pub struct PreparedFixedWordWeights<'a, const L: usize, const N: usize, const TERMS: usize> {
    params: &'a PrimeParameters<L>,
    positive: [[Uint<L>; N]; TERMS],
    // For N<=2: negated weights. For wider operands: -weight*2^(64*N)
    // in slot zero, allowing one signed correction per operand.
    corrections: [[Uint<L>; N]; TERMS],
}
impl<'a, const L: usize, const N: usize, const TERMS: usize>
    PreparedFixedWordWeights<'a, L, N, TERMS>
{
    fn new(field: &'a FpCtx<L>, coefficients: [Fp<L>; TERMS]) -> Self {
        assert!(
            TERMS <= 16 && N > 0 && N <= 256,
            "unsupported public fold dimensions"
        );
        let p = field.params();
        let radix = field.from_integer(&Uint::<2>::from_words([0, 1]));
        let positive = coefficients.map(|c| limb_weights(p, &radix.words, &c));
        let corrections = positive.map(|w| {
            if N > 2 {
                let mut correction = [Uint::ZERO; N];
                correction[0] = p.neg(&p.mul(&w[N - 1], &radix.words));
                correction
            } else {
                w.map(|v| p.neg(&v))
            }
        });
        Self {
            params: p,
            positive,
            corrections,
        }
    }
    #[inline(always)]
    fn unsigned_dot(&self, mut read: impl FnMut(usize) -> Uint<N>) -> Fp<L> {
        let mut acc = UintAccumulator::<L, 1>::ZERO;
        for i in 0..TERMS {
            let value = read(i);
            for j in 0..N {
                acc.mac(
                    &self.positive[i][j],
                    &Uint::from_words([value.as_words()[j]]),
                );
            }
        }
        finish_words(self.params, acc)
    }
    #[inline(always)]
    fn signed_dot(&self, mut read: impl FnMut(usize) -> Z<N>) -> Fp<L> {
        let mut acc = UintAccumulator::<L, 1>::ZERO;
        for i in 0..TERMS {
            let value = read(i);
            let sign = value.is_negative_ct();
            if N > 2 {
                // The unsigned word representation is value + sign*2^(64*N).
                // Keep its limb products intact and correct the radix once.
                for j in 0..N {
                    acc.mac(
                        &self.positive[i][j],
                        &Uint::from_words([value.as_words()[j]]),
                    );
                }
                let correction = Uint::ct_select(&Uint::ZERO, &self.corrections[i][0], sign);
                acc.merge_assign(&UintAccumulator {
                    low: *correction.as_words(),
                    high: [0],
                    head: 0,
                });
            } else {
                let magnitude = value.unsigned_abs();
                for j in 0..N {
                    let w = Uint::ct_select(&self.positive[i][j], &self.corrections[i][j], sign);
                    acc.mac(&w, &Uint::from_words([magnitude.as_words()[j]]));
                }
            }
        }
        finish_words(self.params, acc)
    }
}
impl<const L: usize, const N: usize> PreparedLinearCombination<Uint<N>> for FpCtx<L> {
    type Prepared<'a, const TERMS: usize> = PreparedFixedWordWeights<'a, L, N, TERMS>;
    fn prepare_linear_combination<const TERMS: usize>(
        &self,
        c: [Fp<L>; TERMS],
    ) -> Self::Prepared<'_, TERMS> {
        PreparedFixedWordWeights::new(self, c)
    }
    #[inline(always)]
    fn linear_combination<const TERMS: usize>(
        p: &Self::Prepared<'_, TERMS>,
        read: impl FnMut(usize) -> Uint<N>,
    ) -> Fp<L> {
        p.unsigned_dot(read)
    }
}
impl<const L: usize, const N: usize> PreparedLinearCombination<Z<N>> for FpCtx<L> {
    type Prepared<'a, const TERMS: usize> = PreparedFixedWordWeights<'a, L, N, TERMS>;
    fn prepare_linear_combination<const TERMS: usize>(
        &self,
        c: [Fp<L>; TERMS],
    ) -> Self::Prepared<'_, TERMS> {
        PreparedFixedWordWeights::new(self, c)
    }
    #[inline(always)]
    fn linear_combination<const TERMS: usize>(
        p: &Self::Prepared<'_, TERMS>,
        read: impl FnMut(usize) -> Z<N>,
    ) -> Fp<L> {
        p.signed_dot(read)
    }
}
macro_rules! native {
    ($src:ty, $n:literal, $convert:expr) => {
        impl<const L: usize> PreparedLinearCombination<$src> for FpCtx<L> {
            type Prepared<'a, const TERMS: usize> = PreparedFixedWordWeights<'a, L, $n, TERMS>;
            fn prepare_linear_combination<const TERMS: usize>(
                &self,
                c: [Fp<L>; TERMS],
            ) -> Self::Prepared<'_, TERMS> {
                PreparedFixedWordWeights::new(self, c)
            }
            #[inline(always)]
            fn linear_combination<const TERMS: usize>(
                p: &Self::Prepared<'_, TERMS>,
                mut read: impl FnMut(usize) -> $src,
            ) -> Fp<L> {
                p.unsigned_dot(|i| ($convert)(read(i)))
            }
        }
    };
}
native!(u32, 1, |v: u32| Uint::from_words([u64::from(v)]));
native!(u64, 1, |v: u64| Uint::from_words([v]));
native!(u128, 2, |v: u128| Uint::from_words([
    v as u64,
    (v >> 64) as u64
]));

pub struct PreparedFieldWeights<'a, const L: usize, const TERMS: usize> {
    field: &'a FpCtx<L>,
    coefficients: [Fp<L>; TERMS],
    affine_pair: bool,
}
impl<const L: usize> PreparedLinearCombination<Fp<L>> for FpCtx<L> {
    type Prepared<'a, const TERMS: usize> = PreparedFieldWeights<'a, L, TERMS>;
    fn prepare_linear_combination<const TERMS: usize>(
        &self,
        coefficients: [Fp<L>; TERMS],
    ) -> Self::Prepared<'_, TERMS> {
        assert!(TERMS <= 16);
        let affine_pair = TERMS == 2 && self.add(&coefficients[0], &coefficients[1]) == self.one();
        PreparedFieldWeights {
            field: self,
            coefficients,
            affine_pair,
        }
    }
    #[inline(always)]
    fn linear_combination<const TERMS: usize>(
        p: &Self::Prepared<'_, TERMS>,
        mut read: impl FnMut(usize) -> Fp<L>,
    ) -> Fp<L> {
        // Public challenge coefficients of an ordinary fold sum to one.
        // Preserve its one-multiplication interpolation on field-valued rows.
        if TERMS == 2 && p.affine_pair {
            let left = read(0);
            let right = read(1);
            return p.field.add(
                &left,
                &p.field.mul(&p.coefficients[1], &p.field.sub(&right, &left)),
            );
        }
        let mut accumulator = FpProductAcc::zero();
        for i in 0..TERMS {
            accumulator.accumulate(&p.coefficients[i], &read(i));
        }
        p.field.reduce_product_with_public_bound(accumulator, TERMS)
    }
}

/// Reduction schedule selected once from a public MAC count and modulus.
/// Each use must receive at most that many products, including merged terms.
pub struct PreparedProductReduction<'a, const L: usize, Id = RuntimePrime> {
    parameters: &'a PrimeParameters<L>,
    mode: ProductReductionMode,
    id: core::marker::PhantomData<fn() -> Id>,
}
#[derive(Clone, Copy)]
enum ProductReductionMode {
    Zero,
    Montgomery,
    General,
}
impl<const L: usize, Id> PreparedProductReduction<'_, L, Id> {
    pub(super) fn new(
        parameters: &PrimeParameters<L>,
        max_terms: usize,
    ) -> PreparedProductReduction<'_, L, Id> {
        let bits = parameters
            .modulus
            .as_words()
            .iter()
            .rposition(|&w| w != 0)
            .map_or(0, |i| {
                64 * i + 64 - parameters.modulus.as_words()[i].leading_zeros() as usize
            });
        let mode = if max_terms == 0 {
            ProductReductionMode::Zero
        } else {
            let log_terms = (usize::BITS - (max_terms - 1).leading_zeros()) as usize;
            // S < m*p² < p*R: only public dimensions select the schedule.
            if bits + log_terms <= 64 * L {
                ProductReductionMode::Montgomery
            } else {
                ProductReductionMode::General
            }
        };
        PreparedProductReduction {
            parameters,
            mode,
            id: core::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub fn reduce(&self, accumulator: PrimeProductAcc<Id, L>) -> PrimeValue<Id, L> {
        match self.mode {
            ProductReductionMode::Zero => PrimeValue::new(Uint::ZERO),
            ProductReductionMode::Montgomery => {
                let (low, high, head) = accumulator.unreduced_integer().as_parts();
                debug_assert_eq!(head, 0);
                PrimeValue::new(self.parameters.redc(UintProduct {
                    low: *low,
                    high: *high,
                }))
            }
            ProductReductionMode::General => {
                PrimeValue::new(self.parameters.reduce_product_acc(&accumulator.payload))
            }
        }
    }
}
impl<const L: usize> FpCtx<L> {
    pub fn prepare_product_reduction(&self, max_terms: usize) -> PreparedProductReduction<'_, L> {
        PreparedProductReduction::new(self.params(), max_terms)
    }
    /// Reduce a sum bounded by a PUBLIC total MAC count. Products have scale
    /// R^2, and the result has scale R. No private magnitude selects a schedule.
    #[inline]
    pub fn reduce_product_with_public_bound(
        &self,
        accumulator: FpProductAcc<L>,
        max_terms: usize,
    ) -> Fp<L> {
        self.prepare_product_reduction(max_terms)
            .reduce(accumulator)
    }
}
