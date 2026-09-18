//! Borrowed native operands and split products. Width dispatch occurs before
//! any hot loop; integer projection is fused with the first outer MAC/fold.
use super::*;
#[cfg(test)]
use crate::piop::spartan::raw_monty::RawFieldStorage;
#[cfg(test)]
use crate::utils::delayed_reduction::EncodedMac;
use field::Uint;
#[cfg(test)]
use field::{CtMask, CtOrd, CtSelect, Fp, FpLinearAcc, MergeAccumulator, RingOps, WideMul};

#[derive(Clone, Copy)]
pub struct NativeWideProducts<'a, T> {
    az: &'a [T],
    bz: &'a [T],
    cz_lo: &'a [T],
    cz_hi: &'a [T],
    rows: usize,
}
impl<'a, T> NativeWideProducts<'a, T> {
    pub(crate) fn new(
        az: &'a [T],
        bz: &'a [T],
        cz_lo: &'a [T],
        cz_hi: &'a [T],
        rows: usize,
    ) -> Self {
        assert!(rows.is_power_of_two());
        assert!(az.len() <= rows);
        assert_eq!(az.len(), bz.len());
        assert_eq!(az.len(), cz_lo.len());
        assert_eq!(az.len(), cz_hi.len());
        Self {
            az,
            bz,
            cz_lo,
            cz_hi,
            rows,
        }
    }
}

/// Input variants used only by the independent legacy arithmetic tests.
#[cfg(test)]
#[derive(Clone)]
pub(crate) enum NativeInput<'a> {
    U32(NativeProducts<'a>),
    Residues(&'a RawProducts),
    U64(NativeWideProducts<'a, u64>),
    U128(NativeWideProducts<'a, u128>),
    Integers4096(super::super::OuterInputs<Uint<64>>),
    Signed128(super::super::OuterInputs<field::Z<2>, field::Z<4>>),
}
impl super::super::inputs::OuterRows for NativeProducts<'_> {
    type AB = u32;
    type C = u64;
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.az.len(), self.bz.len(), self.cz.len())
    }
    // The enclosing native relation establishes the public 32-bit width bound.
    #[inline(always)]
    fn a(&self, i: usize) -> u32 {
        self.az[i] as u32
    }
    #[inline(always)]
    fn b(&self, i: usize) -> u32 {
        self.bz[i] as u32
    }
    #[inline(always)]
    fn c(&self, i: usize) -> u64 {
        self.cz[i]
    }
}
impl super::super::inputs::OuterRows for NativeWideProducts<'_, u64> {
    type AB = u64;
    type C = u128;
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.rows, self.rows, self.rows)
    }
    #[inline(always)]
    fn a(&self, i: usize) -> u64 {
        self.az.get(i).copied().unwrap_or(0)
    }
    #[inline(always)]
    fn b(&self, i: usize) -> u64 {
        self.bz.get(i).copied().unwrap_or(0)
    }
    #[inline(always)]
    fn c(&self, i: usize) -> u128 {
        self.cz_lo.get(i).copied().unwrap_or(0) as u128
            | ((self.cz_hi.get(i).copied().unwrap_or(0) as u128) << 64)
    }
}
impl super::super::inputs::OuterRows for NativeWideProducts<'_, u128> {
    type AB = u128;
    type C = Uint<4>;
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.rows, self.rows, self.rows)
    }
    #[inline(always)]
    fn a(&self, i: usize) -> u128 {
        self.az.get(i).copied().unwrap_or(0)
    }
    #[inline(always)]
    fn b(&self, i: usize) -> u128 {
        self.bz.get(i).copied().unwrap_or(0)
    }
    #[inline(always)]
    fn c(&self, i: usize) -> Uint<4> {
        let lo = self.cz_lo.get(i).copied().unwrap_or(0);
        let hi = self.cz_hi.get(i).copied().unwrap_or(0);
        Uint::from_words([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64])
    }
}

#[cfg(test)]
pub(super) struct ResidueRows<'a> {
    pub field: &'a field::FpCtx<2>,
    pub products: &'a RawProducts,
}
#[cfg(test)]
impl super::super::inputs::OuterRows for ResidueRows<'_> {
    type AB = Field;
    type C = Field;
    fn dimensions(&self) -> (usize, usize, usize) {
        (
            self.products.az.len(),
            self.products.bz.len(),
            self.products.cz.len(),
        )
    }
    #[inline(always)]
    fn a(&self, i: usize) -> Field {
        shared_raw(self.field, self.products.az[i])
    }
    #[inline(always)]
    fn b(&self, i: usize) -> Field {
        shared_raw(self.field, self.products.bz[i])
    }
    #[inline(always)]
    fn c(&self, i: usize) -> Field {
        shared_raw(self.field, self.products.cz[i])
    }
}

/// Independent retained arithmetic reference for differential tests.
#[cfg(test)]
pub(crate) fn legacy_dispatch<T: Transcript>(
    transcript: &mut T,
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    claim: Field,
    tau: &[Field],
    low: Vec<Raw>,
    high: Vec<Raw>,
    input: NativeInput<'_>,
    known_zero: bool,
) -> Result<OuterSumcheckOutput<Field>, SumcheckError> {
    match input {
        NativeInput::Signed128(p) => {
            use field::IntegerEmbedding;
            super::prepare_encoded_reference(
                transcript,
                ctx,
                reducer,
                claim,
                tau,
                low,
                high,
                RawProducts {
                    az: p.ax.iter().map(|v| ctx.raw(&ctx.from_integer(v))).collect(),
                    bz: p.bx.iter().map(|v| ctx.raw(&ctx.from_integer(v))).collect(),
                    cz: p.cx.iter().map(|v| ctx.raw(&ctx.from_integer(v))).collect(),
                },
                &mut UngrindedRoundBoundary,
                known_zero,
            )
        }

        NativeInput::Integers4096(p) => {
            use field::IntegerEmbedding;
            let project = |values: &[Uint<64>]| {
                values
                    .iter()
                    .map(|v| ctx.raw(&ctx.from_integer(v)))
                    .collect()
            };
            super::prepare_encoded_reference(
                transcript,
                ctx,
                reducer,
                claim,
                tau,
                low,
                high,
                RawProducts {
                    az: project(&p.ax),
                    bz: project(&p.bx),
                    cz: project(&p.cx),
                },
                &mut UngrindedRoundBoundary,
                known_zero,
            )
        }

        NativeInput::U32(p) => {
            if p.az.len() != p.bz.len() || p.az.len() != p.cz.len() {
                return Err(SumcheckError::InvalidProductDimensions);
            }
            prove_native_prefix(
                transcript,
                ctx,
                reducer,
                claim,
                tau,
                low,
                high,
                p.len(),
                known_zero,
                || {
                    [
                        ctx.native_residue(p.az[0]),
                        ctx.native_residue(p.bz[0]),
                        ctx.native_residue(p.cz[0]),
                    ]
                },
                |weights, endpoint| {
                    native_cofactor_evaluations_raw(ctx, reducer, p, weights, endpoint)
                },
                |out, r, weights| {
                    fold_native_products_and_cofactor_evaluations_raw(
                        ctx, reducer, p, out, r, weights,
                    )
                },
            )
        }
        NativeInput::Residues(p) => {
            if p.az.len() != p.bz.len() || p.az.len() != p.cz.len() {
                return Err(SumcheckError::InvalidProductDimensions);
            }
            prove_native_prefix(
                transcript,
                ctx,
                reducer,
                claim,
                tau,
                low,
                high,
                p.len(),
                known_zero,
                || [p.az[0], p.bz[0], p.cz[0]],
                |weights, endpoint| cofactor_evaluations_raw(ctx, reducer, p, weights, endpoint),
                |out, r, weights| match weights {
                    Some((w, e)) => {
                        fold_products_and_cofactor_evaluations_raw(ctx, reducer, p, out, r, w, e)
                    }
                    None => {
                        fold_products_raw(ctx, p, out, r);
                        [0; 2]
                    }
                },
            )
        }
        NativeInput::U64(p) => prove_native_prefix(
            transcript,
            ctx,
            reducer,
            claim,
            tau,
            low,
            high,
            p.rows,
            known_zero,
            || p.folded(ctx, &[ctx.one(), ctx.zero()], 0),
            |weights, endpoint| {
                wide_round0(
                    ctx,
                    reducer,
                    p.rows,
                    |i, e| p.cofactor(i, e),
                    weights,
                    endpoint,
                )
            },
            |out, r, weights| {
                wide_fold(
                    ctx,
                    reducer,
                    p.rows,
                    |c, i| p.folded(ctx, c, i),
                    out,
                    r,
                    weights,
                )
            },
        ),
        NativeInput::U128(p) => prove_native_prefix(
            transcript,
            ctx,
            reducer,
            claim,
            tau,
            low,
            high,
            p.rows,
            known_zero,
            || p.folded(ctx, &[ctx.one(), ctx.zero()], 0),
            |weights, endpoint| {
                wide_round0(
                    ctx,
                    reducer,
                    p.rows,
                    |i, e| p.cofactor(i, e),
                    weights,
                    endpoint,
                )
            },
            |out, r, weights| {
                wide_fold(
                    ctx,
                    reducer,
                    p.rows,
                    |c, i| p.folded(ctx, c, i),
                    out,
                    r,
                    weights,
                )
            },
        ),
    }
}

#[cfg(test)]
fn magnitude<const N: usize>(a: Uint<N>, b: Uint<N>) -> (Uint<N>, CtMask) {
    let negative = a.ct_lt(&b);
    let difference = a.wrapping_sub(&b);
    (
        Uint::ct_select(&difference, &difference.wrapping_neg(), negative),
        negative,
    )
}

#[cfg(test)]
macro_rules! wide_source {
    ($native:ty, $a:literal, $p:literal, $operand:expr, $product:expr) => {
        impl NativeWideProducts<'_, $native> {
            #[inline]
            fn cofactor(self, pair: usize, endpoint: FactoredEndpoint) -> [(Uint<$p>, CtMask); 2] {
                let read =
                    |values: &[$native], i: usize| ($operand)(values.get(i).copied().unwrap_or(0));
                let [a0, a1] = [read(self.az, 2 * pair), read(self.az, 2 * pair + 1)];
                let [b0, b1] = [read(self.bz, 2 * pair), read(self.bz, 2 * pair + 1)];
                let e = match endpoint {
                    FactoredEndpoint::Zero | FactoredEndpoint::KnownZero => 0,
                    FactoredEndpoint::One => 1,
                };
                let residual = if endpoint == FactoredEndpoint::KnownZero {
                    (Uint::ZERO, CtMask::FALSE)
                } else {
                    let (a, b) = if e == 0 { (a0, b0) } else { (a1, b1) };
                    let c = ($product)(
                        self.cz_lo.get(2 * pair + e).copied().unwrap_or(0),
                        self.cz_hi.get(2 * pair + e).copied().unwrap_or(0),
                    );
                    let ab = *field::IntegerOps
                        .mul_wide(&a, &b)
                        .checked_resize_ct::<$p>()
                        .value();
                    magnitude(ab, c)
                };
                let (da, na) = magnitude(a1, a0);
                let (db, nb) = magnitude(b1, b0);
                let infinity = *field::IntegerOps
                    .mul_wide(&da, &db)
                    .checked_resize_ct::<$p>()
                    .value();
                [residual, (infinity, na ^ nb)]
            }
            #[inline]
            fn folded(
                self,
                field: &field::FpCtx<2>,
                coefficients: &[Fp<2>; 2],
                index: usize,
            ) -> [Raw; 3] {
                let fold_operand = |values: &[$native]| {
                    let v = [
                        ($operand)(values.get(index).copied().unwrap_or(0)),
                        ($operand)(values.get(index + 1).copied().unwrap_or(0)),
                    ];
                    raw_shared(field.weighted_pair(coefficients, &v))
                };
                let product = |i| {
                    ($product)(
                        self.cz_lo.get(i).copied().unwrap_or(0),
                        self.cz_hi.get(i).copied().unwrap_or(0),
                    )
                };
                [
                    fold_operand(self.az),
                    fold_operand(self.bz),
                    raw_shared(
                        field.weighted_pair(coefficients, &[product(index), product(index + 1)]),
                    ),
                ]
            }
        }
    };
}
#[cfg(test)]
wide_source!(
    u64,
    1,
    2,
    |v: u64| Uint::<1>::from_words([v]),
    |lo: u64, hi: u64| Uint::<2>::from_words([lo, hi])
);
#[cfg(test)]
wide_source!(
    u128,
    2,
    4,
    |v: u128| Uint::<2>::from_words(raw_to_words(v)),
    |lo: u128, hi: u128| Uint::<4>::from_words([
        lo as u64,
        (lo >> 64) as u64,
        hi as u64,
        (hi >> 64) as u64
    ])
);

#[cfg(test)]
fn wide_round0<const N: usize>(
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    len: usize,
    cofactor: impl Fn(usize, FactoredEndpoint) -> [(Uint<N>, CtMask); 2] + Sync,
    weights: RawEqWeights<'_>,
    endpoint: FactoredEndpoint,
) -> [Raw; 2] {
    let f = ctx;
    let count = len / 2;
    assert_eq!(count, weights.pair_count());
    let zero = || [FpLinearAcc::<2, N>::zero(); 2];
    let merge = |mut a: [FpLinearAcc<2, N>; 2], b: [FpLinearAcc<2, N>; 2]| {
        for i in 0..2 {
            a[i].merge_assign(&b[i]);
        }
        a
    };
    let block = |start: usize, end: usize, low: Option<&[Raw]>| {
        let mut acc = zero();
        for i in start..end {
            let w = shared_raw(
                f,
                low.map_or_else(|| weights.pair_weight(ctx, i), |v| v[i - start]),
            );
            let neg = f.neg(&w);
            for (a, (value, negative)) in acc.iter_mut().zip(cofactor(i, endpoint)) {
                a.accumulate(&Fp::ct_select(&w, &neg, negative), &value);
            }
        }
        acc
    };
    let reduce = |a: [FpLinearAcc<2, N>; 2]| a.map(|a| raw_shared(field::Reduce::reduce(f, a)));
    if let Some(low) = weights.two_level_low() {
        let bucket = |hi: usize| {
            let inner = reduce(block(hi * low.len(), (hi + 1) * low.len(), Some(low)));
            let mut out = product_pair();
            for i in 0..2 {
                out[i].accumulate_encoded(ctx, weights.high[hi], inner[i]);
            }
            out
        };
        #[cfg(feature = "parallel")]
        if parallel(count) {
            return reduce_product_pair(
                (0..weights.high.len())
                    .into_par_iter()
                    .map(bucket)
                    .reduce(product_pair, merge_product_pair),
                reducer,
            );
        }
        return reduce_product_pair(
            (0..weights.high.len())
                .map(bucket)
                .fold(product_pair(), merge_product_pair),
            reducer,
        );
    }
    #[cfg(feature = "parallel")]
    if parallel(count) {
        return reduce(
            (0..count.div_ceil(FOLD_BLOCK))
                .into_par_iter()
                .map(|b| block(b * FOLD_BLOCK, ((b + 1) * FOLD_BLOCK).min(count), None))
                .reduce(zero, merge),
        );
    }
    reduce(block(0, count, None))
}

#[cfg(test)]
fn wide_fold(
    ctx: &field::FpCtx<2>,
    reducer: &field::FpCtx<2>,
    len: usize,
    folded: impl Fn(&[Fp<2>; 2], usize) -> [Raw; 3] + Sync,
    out: &mut RawProducts,
    challenge: Raw,
    weights: Option<(RawEqWeights<'_>, FactoredEndpoint)>,
) -> [Raw; 2] {
    assert_eq!(len, 2 * out.len());
    let f = ctx;
    let challenge = shared_raw(f, challenge);
    let coefficients = [f.sub(&f.one(), &challenge), challenge];
    let Some((weights, endpoint)) = weights else {
        assert_eq!(out.len(), 1);
        let [a, b, c] = folded(&coefficients, 0);
        out.az[0] = a;
        out.bz[0] = b;
        out.cz[0] = c;
        return [0, 0];
    };
    let pairs = out.len() / 2;
    assert_eq!(pairs, weights.pair_count());
    let process = |start: usize, a: &mut [Raw], b: &mut [Raw], c: &mut [Raw]| {
        let mut total = product_pair();
        let low_count = weights.two_level_low().map_or(a.len() / 2, |v| v.len());
        for base in (0..a.len() / 2).step_by(low_count) {
            let mut inner = product_pair();
            for j in 0..low_count {
                let pair = base + j;
                let i = 4 * (start + pair);
                let o = 2 * pair;
                let [a0, b0, c0] = folded(&coefficients, i);
                let [a1, b1, c1] = folded(&coefficients, i + 2);
                a[o] = a0;
                a[o + 1] = a1;
                b[o] = b0;
                b[o + 1] = b1;
                c[o] = c0;
                c[o + 1] = c1;
                let weight = weights
                    .two_level_low()
                    .map_or_else(|| weights.pair_weight(ctx, start + pair), |v| v[j]);
                accumulate_cofactor_raw(ctx, &mut inner, weight, endpoint, a0, a1, b0, b1, c0, c1);
            }
            if weights.two_level_low().is_some() {
                let inner = reduce_product_pair(inner, reducer);
                let high = weights.high[(start + base) / low_count];
                for i in 0..2 {
                    total[i].accumulate_encoded(ctx, high, inner[i]);
                }
            } else {
                total = merge_product_pair(total, inner);
            }
        }
        total
    };
    #[cfg(feature = "parallel")]
    if parallel(pairs) {
        let block = weights.two_level_low().map_or(FOLD_BLOCK / 2, |v| {
            (FOLD_BLOCK / 2).div_ceil(v.len()).max(1) * v.len()
        });
        let total = (
            out.az.par_chunks_mut(2 * block),
            out.bz.par_chunks_mut(2 * block),
            out.cz.par_chunks_mut(2 * block),
        )
            .into_par_iter()
            .enumerate()
            .map(|(i, (a, b, c))| process(i * block, a, b, c))
            .reduce(product_pair, merge_product_pair);
        return reduce_product_pair(total, reducer);
    }
    reduce_product_pair(process(0, &mut out.az, &mut out.bz, &mut out.cz), reducer)
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{RngExt, SeedableRng, rngs::StdRng};

    #[test]
    fn native_wide_rounds_match_projected_full_width_and_padded_inputs() {
        let mut rng = StdRng::seed_from_u64(0x4f75_7465_725f_7769);
        for q in [(1u128 << 100) - 15, u128::MAX - 158] {
            let cfg = Field::make_cfg(&Uint::from_words(raw_to_words(q))).unwrap();
            let ctx = crate::piop::spartan::raw_monty::field_context(&cfg);
            let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            for log in [1, 2, 5, 14] {
                let rows = 1usize << log;
                for live in [rows, rows - 1] {
                    let mut values = || {
                        (0..live)
                            .map(|i| {
                                if i < 4 {
                                    [0, u128::MAX, 1, 1u128 << 127][i]
                                } else {
                                    rng.random()
                                }
                            })
                            .collect::<Vec<u128>>()
                    };
                    let a = values();
                    let b = values();
                    let lo = values();
                    let hi = values();
                    let native = NativeWideProducts::new(&a, &b, &lo, &hi, rows);
                    let two128 = cfg.mul(
                        &(Field::from_with_cfg(1u128 << 127, &cfg)),
                        &(&Field::from_with_cfg(2u64, &cfg)),
                    );
                    let project = |v: &[u128]| {
                        (0..rows)
                            .map(|i| {
                                ctx.raw(&Field::from_with_cfg(v.get(i).copied().unwrap_or(0), &cfg))
                            })
                            .collect()
                    };
                    let product = (0..rows)
                        .map(|i| {
                            ctx.raw(
                                &(cfg.add(
                                    &(Field::from_with_cfg(lo.get(i).copied().unwrap_or(0), &cfg)),
                                    &(&(cfg.mul(
                                        &(two128.clone()),
                                        &(&Field::from_with_cfg(
                                            hi.get(i).copied().unwrap_or(0),
                                            &cfg,
                                        )),
                                    ))),
                                )),
                            )
                        })
                        .collect();
                    let projected = RawProducts {
                        az: project(&a),
                        bz: project(&b),
                        cz: product,
                    };
                    let tau: Vec<_> = (0..log - 1)
                        .map(|_| ctx.native_residue_u128(rng.random()))
                        .collect();
                    let split = tau.len() / 2;
                    let low = eq_table_raw(&ctx, &tau[..split]);
                    let high = eq_table_raw(&ctx, &tau[split..]);
                    for endpoint in [FactoredEndpoint::Zero, FactoredEndpoint::One] {
                        let weights = weights_of(&low, &high);
                        assert_eq!(
                            wide_round0(
                                &ctx,
                                &reducer,
                                native.rows,
                                |i, e| native.cofactor(i, e),
                                weights,
                                endpoint
                            ),
                            cofactor_evaluations_raw(&ctx, &reducer, &projected, weights, endpoint)
                        );
                    }
                    let challenge = ctx.native_residue_u128(rng.random());
                    let mut expected = RawProducts::zeros(rows / 2);
                    let mut actual = RawProducts::zeros(rows / 2);
                    if rows == 2 {
                        wide_fold(
                            &ctx,
                            &reducer,
                            native.rows,
                            |c, i| native.folded(&ctx, c, i),
                            &mut actual,
                            challenge,
                            None,
                        );
                        fold_products_raw(&ctx, &projected, &mut expected, challenge);
                    } else {
                        let tau = &tau[..tau.len() - 1];
                        let split = tau.len() / 2;
                        let low = eq_table_raw(&ctx, &tau[..split]);
                        let high = eq_table_raw(&ctx, &tau[split..]);
                        let weights = weights_of(&low, &high);
                        let want = fold_products_and_cofactor_evaluations_raw(
                            &ctx,
                            &reducer,
                            &projected,
                            &mut expected,
                            challenge,
                            weights,
                            FactoredEndpoint::One,
                        );
                        let got = wide_fold(
                            &ctx,
                            &reducer,
                            native.rows,
                            |c, i| native.folded(&ctx, c, i),
                            &mut actual,
                            challenge,
                            Some((weights, FactoredEndpoint::One)),
                        );
                        assert_eq!(got, want);
                    }
                    assert_eq!(actual.az, expected.az);
                    assert_eq!(actual.bz, expected.bz);
                    assert_eq!(actual.cz, expected.cz);
                }
            }
        }
    }
}
