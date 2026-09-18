//! Arithmetic capabilities for the shared integer/field outer traversal.
use field::{
    FieldOps, Fp, FpCtx, IntegerOps, PreparedLinearCombination, PreparedWordWeights, Reduce,
    RingOps, Uint, UintAccumulator, Z,
};

/// Exact working arithmetic for ordinary and univariate first rounds.
/// Implementations must cover every finite-difference intermediate for K<=4.
/// The traversal chooses widths from the input type and public K, never values.
/// Lifts, exact arithmetic, and prepared folds must use compatible embeddings
/// into this field. A fresh accumulator represents zero; finishing it returns
/// the field image of `sum(weights[index] * residual)` for all accumulated terms.
/// Weights and accumulators must be prepared and finished in the same context.
pub trait OuterArithmetic<AB, C>:
    FieldOps + PreparedLinearCombination<AB> + PreparedLinearCombination<C>
{
    type WidenedAB: Copy + Send + Sync;
    type WidenedC: Copy + Send + Sync;
    type Residual: Copy + Send + Sync;
    type Weights: Send + Sync;
    type Accumulator: Send;

    fn lift_ab(&self, value: AB) -> Self::WidenedAB;
    fn lift_c(&self, value: C) -> Self::WidenedC;
    fn add_ab(&self, a: Self::WidenedAB, b: Self::WidenedAB) -> Self::WidenedAB;
    fn sub_ab(&self, a: Self::WidenedAB, b: Self::WidenedAB) -> Self::WidenedAB;
    fn add_c(&self, a: Self::WidenedC, b: Self::WidenedC) -> Self::WidenedC;
    fn sub_c(&self, a: Self::WidenedC, b: Self::WidenedC) -> Self::WidenedC;
    fn product(&self, a: Self::WidenedAB, b: Self::WidenedAB) -> Self::Residual;
    fn residual(&self, product: Self::Residual, c: Self::WidenedC) -> Self::Residual;
    fn prepare_weights(&self, weights: &[Self::Elem]) -> Self::Weights;
    fn accumulator(&self) -> Self::Accumulator;
    /// Add one residual; at most 2^32 calls may feed an accumulator.
    /// With PREFIX=false the residual must be an original-row residual or a
    /// product of original-row differences, before interpolation. Implementations
    /// may use the corresponding narrower public magnitude bound in that case.
    fn accumulate<const PREFIX: bool>(
        &self,
        acc: &mut Self::Accumulator,
        weights: &Self::Weights,
        index: usize,
        value: Self::Residual,
    );
    fn finish_accumulator(&self, weights: &Self::Weights, acc: Self::Accumulator) -> Self::Elem;
}

// K=4's largest positive/negative interpolation coefficient sum is
// 1,374,322,688, including all finite-difference intermediates. Consequently
// the signed AB/product bounds are 64/126, 96/190, and 160/318 bits.
// The ordinary magnitude bounds are respectively 64, 128, and 256 bits.
macro_rules! integer_arithmetic {
    ($ab:ty, $c:ty, $wide:ty, $res:ty, $words:literal, $ordinary:literal,
     $lift_ab:expr, $lift_c:expr, $product:expr, $signed:expr) => {
        impl<const L: usize> OuterArithmetic<$ab, $c> for FpCtx<L> {
            type WidenedAB = $wide;
            type WidenedC = $res;
            type Residual = $res;
            type Weights = PreparedWordWeights<L, $words>;
            type Accumulator = UintAccumulator<L, 1>;
            #[inline(always)]
            fn lift_ab(&self, v: $ab) -> $wide {
                ($lift_ab)(v)
            }
            #[inline(always)]
            fn lift_c(&self, v: $c) -> $res {
                ($lift_c)(v)
            }
            #[inline(always)]
            fn add_ab(&self, a: $wide, b: $wide) -> $wide {
                a + b
            }
            #[inline(always)]
            fn sub_ab(&self, a: $wide, b: $wide) -> $wide {
                a - b
            }
            #[inline(always)]
            fn add_c(&self, a: $res, b: $res) -> $res {
                a + b
            }
            #[inline(always)]
            fn sub_c(&self, a: $res, b: $res) -> $res {
                a - b
            }
            #[inline(always)]
            fn product(&self, a: $wide, b: $wide) -> $res {
                ($product)(a, b)
            }
            #[inline(always)]
            fn residual(&self, p: $res, c: $res) -> $res {
                p - c
            }
            fn prepare_weights(&self, w: &[Fp<L>]) -> Self::Weights {
                PreparedWordWeights::signed(self, w)
            }
            #[inline(always)]
            fn accumulator(&self) -> Self::Accumulator {
                UintAccumulator::ZERO
            }
            #[inline(always)]
            fn accumulate<const PREFIX: bool>(
                &self,
                a: &mut Self::Accumulator,
                w: &Self::Weights,
                i: usize,
                v: $res,
            ) {
                if PREFIX {
                    w.accumulate_signed::<$words>(a, i, ($signed)(v));
                } else {
                    w.accumulate_signed::<$ordinary>(a, i, ($signed)(v));
                }
            }
            #[inline(always)]
            fn finish_accumulator(&self, w: &Self::Weights, a: Self::Accumulator) -> Fp<L> {
                w.finish(self, a)
            }
        }
    };
}
integer_arithmetic!(
    u32,
    u64,
    i64,
    i128,
    2,
    1,
    i64::from,
    i128::from,
    |a: i64, b: i64| i128::from(a) * i128::from(b),
    |v: i128| Z::from_twos_complement_words([v as u64, (v >> 64) as u64])
);
integer_arithmetic!(
    u64,
    u128,
    i128,
    Z<3>,
    3,
    2,
    i128::from,
    |v: u128| Z::from_twos_complement_words([v as u64, (v >> 64) as u64, 0]),
    |a: i128, b: i128| IntegerOps
        .wrapping_signed_product::<2, 2, 3>(&Z::<2>::from(a), &Z::<2>::from(b)),
    |v| v
);
integer_arithmetic!(
    u64,
    u64,
    i128,
    Z<3>,
    3,
    2,
    i128::from,
    |v: u64| Z::from_twos_complement_words([v, 0, 0]),
    |a: i128, b: i128| IntegerOps
        .wrapping_signed_product::<2, 2, 3>(&Z::<2>::from(a), &Z::<2>::from(b)),
    |v| v
);
integer_arithmetic!(
    u128,
    Uint<4>,
    Z<3>,
    Z<5>,
    5,
    4,
    |v: u128| Z::from_twos_complement_words([v as u64, (v >> 64) as u64, 0]),
    |v: Uint<4>| Z::from_twos_complement_words([
        v.as_words()[0],
        v.as_words()[1],
        v.as_words()[2],
        v.as_words()[3],
        0
    ]),
    |a: Z<3>, b: Z<3>| IntegerOps.wrapping_signed_product::<3, 3, 5>(&a, &b),
    |v| v
);
// MultiSwap's public matrix coefficient sums bound every exact row below
// 2^4096, including malformed assignments within their declared 2048 bits.
// K<=4 adds at most 31 magnitude bits per operand; 130 signed words cover
// every product/residual. Ordinary differences need only 128 magnitude words.
integer_arithmetic!(
    Uint<64>,
    Uint<64>,
    Z<65>,
    Z<130>,
    130,
    128,
    |v: Uint<64>| Z::from_twos_complement_words(*v.zero_extend::<65>().as_words()),
    |v: Uint<64>| Z::from_twos_complement_words(*v.zero_extend::<130>().as_words()),
    |a: Z<65>, b: Z<65>| IntegerOps.wrapping_signed_product::<65, 65, 130>(&a, &b),
    |v| v
);
integer_arithmetic!(
    Z<2>,
    Z<4>,
    Z<3>,
    Z<5>,
    5,
    4,
    |v: Z<2>| v.sign_extend::<3>(),
    |v: Z<4>| v.sign_extend::<5>(),
    |a: Z<3>, b: Z<3>| IntegerOps.wrapping_signed_product::<3, 3, 5>(&a, &b),
    |v| v
);
// Signed five-word operands have magnitude <=2^319. K<=4 adds at most
// 31 bits: products/residuals fit 701 magnitude bits (eleven signed words),
// including the widened nine-word C. Ordinary differences need ten words.
integer_arithmetic!(
    Z<5>,
    Z<9>,
    Z<6>,
    Z<11>,
    11,
    10,
    |v: Z<5>| v.sign_extend::<6>(),
    |v: Z<9>| v.sign_extend::<11>(),
    |a: Z<6>, b: Z<6>| IntegerOps.wrapping_signed_product::<6, 6, 11>(&a, &b),
    |v| v
);
integer_arithmetic!(
    Z<9>,
    Z<9>,
    Z<10>,
    Z<20>,
    20,
    18,
    |v: Z<9>| v.sign_extend::<10>(),
    |v: Z<9>| v.sign_extend::<20>(),
    |a: Z<10>, b: Z<10>| IntegerOps.wrapping_signed_product::<10, 10, 20>(&a, &b),
    |v| v
);

impl<const L: usize> OuterArithmetic<Fp<L>, Fp<L>> for FpCtx<L> {
    type WidenedAB = Fp<L>;
    type WidenedC = Fp<L>;
    type Residual = Fp<L>;
    type Weights = Vec<Fp<L>>;
    type Accumulator = field::FpProductAcc<L>;
    #[inline(always)]
    fn lift_ab(&self, v: Fp<L>) -> Fp<L> {
        v
    }
    #[inline(always)]
    fn lift_c(&self, v: Fp<L>) -> Fp<L> {
        v
    }
    #[inline(always)]
    fn add_ab(&self, a: Fp<L>, b: Fp<L>) -> Fp<L> {
        self.add(&a, &b)
    }
    #[inline(always)]
    fn sub_ab(&self, a: Fp<L>, b: Fp<L>) -> Fp<L> {
        self.sub(&a, &b)
    }
    #[inline(always)]
    fn add_c(&self, a: Fp<L>, b: Fp<L>) -> Fp<L> {
        self.add(&a, &b)
    }
    #[inline(always)]
    fn sub_c(&self, a: Fp<L>, b: Fp<L>) -> Fp<L> {
        self.sub(&a, &b)
    }
    #[inline(always)]
    fn product(&self, a: Fp<L>, b: Fp<L>) -> Fp<L> {
        self.mul(&a, &b)
    }
    #[inline(always)]
    fn residual(&self, p: Fp<L>, c: Fp<L>) -> Fp<L> {
        self.sub(&p, &c)
    }
    fn prepare_weights(&self, w: &[Fp<L>]) -> Self::Weights {
        w.to_vec()
    }
    #[inline(always)]
    fn accumulator(&self) -> Self::Accumulator {
        Default::default()
    }
    #[inline(always)]
    fn accumulate<const PREFIX: bool>(
        &self,
        a: &mut Self::Accumulator,
        w: &Self::Weights,
        i: usize,
        v: Fp<L>,
    ) {
        a.accumulate(&w[i], &v);
    }
    #[inline(always)]
    fn finish_accumulator(&self, _: &Self::Weights, a: Self::Accumulator) -> Fp<L> {
        Reduce::reduce(self, a)
    }
}
