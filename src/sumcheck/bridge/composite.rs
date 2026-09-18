//! A repeated local factor followed by a corrected circuit tail. This owner
//! is shared by binding and the existing structured inner-sumcheck MLE adapter.
use crate::piop::spartan::matrix::make_equality_factors;
use crate::poly::mle::EqualityWeights;
#[cfg(test)]
use crate::poly::mle::FactoredMultilinearExtension;
use crate::sumcheck::SumcheckError;
use crate::sumcheck::inner::native::{RawFieldStorage, make_equality_factors_raw, raw_to_words};
use crate::sumcheck::inner::packed::CompactCompositeMle;
use circuit::linear_map::ColumnValues;
use circuit::montgomery_tail::MontgomeryTail;
use field::RingOps;

/// One owned coefficient representation: repeated SHA factors, a compact
/// Montgomery tail, and the constant-wire adjustment.
pub(crate) struct CompositeCoefficients {
    pub(crate) num_vars: usize,
    pub(crate) instance_weights: Vec<field::Fp<2>>,
    pub(crate) sha_local_evaluations: Vec<field::Fp<2>>,
    pub(crate) p256_tail: MontgomeryTail,
    pub(crate) constant_weight: field::Fp<2>,
}

impl CompositeCoefficients {
    pub(crate) fn as_mle(
        &self,
        cfg: &field::FpCtx<2>,
    ) -> Result<CompactCompositeMle<'_>, SumcheckError> {
        CompactCompositeMle::new(
            self.num_vars,
            &self.sha_local_evaluations,
            &self.instance_weights,
            &self.p256_tail,
            self.constant_weight,
            cfg,
        )
        .map_err(|_| SumcheckError::InvalidProductDimensions)
    }

    pub(crate) fn evaluate(
        &self,
        point: &[field::Fp<2>],
        cfg: &field::FpCtx<2>,
    ) -> Result<field::Fp<2>, SumcheckError> {
        let _scope = tracing::info_span!("ecdsa:coefficient_evaluate").entered();
        self.as_mle(cfg)?
            .evaluate(point, cfg)
            .map_err(|_| SumcheckError::InvalidProductDimensions)
    }
}

/// The tail cells' equality weights as forward-pass column values. A scalar
/// column `j` is `eq(offset + j) = low[(offset + j) mod L] · high[(offset + j) / L]`;
/// a power group's `Σ_{k<len} 2^k · eq(offset + first + k)` uses the backward
/// recurrence of [`evaluate_tail_by_runs`] (`Q[t] = Σ_{i ≥ t} 2^(i − t) · low[i]`,
/// no inverses): the group's intersection with one high block `[lo, hi)`
/// contributes `high[b] · (Q[lo] − 2^(hi − lo) · Q[hi])` times the running
/// power of two, so a group costs a few multiplications per block it spans.
pub(crate) struct TailEqualityColumns {
    equality: RawEqualityWeights,
    offset: usize,
    len: usize,
    /// `Q[t] = low[t] + 2 · Q[t + 1]`, `Q[L] = 0`.
    suffix: Vec<u128>,
    /// `2^k` in Montgomery form for `k ≤ L`.
    pow2: Vec<u128>,
}

impl TailEqualityColumns {
    /// The equality tables of `point`, split at half the point (as
    /// `make_equality_factors` splits them).
    pub(crate) fn new(
        ctx: &field::FpCtx<2>,
        offset: usize,
        len: usize,
        point: &[field::Fp<2>],
    ) -> Self {
        let equality = RawEqualityWeights::new(ctx, point);
        let block = equality.low.len();
        let mut suffix = vec![0 as u128; block + 1];
        for t in (0..block).rev() {
            let doubled = ctx.add_raw(suffix[t + 1], suffix[t + 1]);
            suffix[t] = ctx.add_raw(equality.low[t], doubled);
        }
        let mut pow2 = Vec::with_capacity(block + 1);
        pow2.push(ctx.native_residue(1));
        for k in 0..block {
            pow2.push(ctx.add_raw(pow2[k], pow2[k]));
        }
        Self {
            equality,
            offset,
            len,
            suffix,
            pow2,
        }
    }

    /// `eq(point, index)` over the whole assignment domain.
    #[inline]
    pub(crate) fn eq_at(&self, index: usize) -> u128 {
        self.equality.at(index)
    }
}

impl ColumnValues<field::Fp<2>> for TailEqualityColumns {
    fn len(&self) -> usize {
        self.len
    }
    fn scalar(&self, column: usize) -> field::Fp<2> {
        crate::sumcheck::inner::native::shared_raw(
            &self.equality.ctx,
            self.eq_at(self.offset + column),
        )
    }

    fn power_sum(&self, first: usize, len: usize) -> field::Fp<2> {
        let equality = &self.equality;
        let ctx = &equality.ctx;
        let block = equality.low.len();
        let mut sum = 0 as u128;
        let mut base = self.pow2[0];
        let mut index = self.offset + first;
        let end = index + len;
        while index < end {
            let b = index >> equality.shift;
            let lo = index & equality.mask;
            let hi = (lo + (end - index)).min(block);
            let part = ctx.sub_raw(
                self.suffix[lo],
                ctx.mul_raw(self.pow2[hi - lo], self.suffix[hi]),
            );
            sum = ctx.add_raw(sum, ctx.mul_raw(ctx.mul_raw(base, part), equality.high[b]));
            base = ctx.mul_raw(base, self.pow2[hi - lo]);
            index += hi - lo;
        }
        crate::sumcheck::inner::native::shared_raw(ctx, sum)
    }
}

/// [`EqualityWeights`] in raw residues: `eq(index) = low[index mod L] · high[index / L]`
/// with the tables split at half the point, as `make_equality_factors` splits
/// them, so every weight is the same residue one raw product later.
pub(crate) struct RawEqualityWeights {
    ctx: field::FpCtx<2>,
    low: Vec<u128>,
    high: Vec<u128>,
    shift: u32,
    mask: usize,
}

impl RawEqualityWeights {
    pub(crate) fn new(ctx: &field::FpCtx<2>, point: &[field::Fp<2>]) -> Self {
        let (low, high) = make_equality_factors_raw(ctx, point);
        let block = low.len();
        Self {
            ctx: ctx.clone(),
            low,
            high,
            shift: block.ilog2(),
            mask: block - 1,
        }
    }

    #[inline]
    pub(crate) fn at(&self, index: usize) -> u128 {
        self.ctx
            .mul_raw(self.low[index & self.mask], self.high[index >> self.shift])
    }

    /// `Σ_j values[j] · eq(j)`, with the high factor applied once per block:
    /// `Σ_h high[h] · Σ_l low[l] · values[h·L + l]`.
    pub(crate) fn dot(&self, values: &[u128]) -> u128 {
        let ctx = &self.ctx;
        let mut sum = 0 as u128;
        for (h, block) in values.chunks(self.low.len()).enumerate() {
            let inner = block
                .iter()
                .zip(&self.low)
                .fold(0 as u128, |acc, (&value, &low)| {
                    ctx.add_raw(acc, ctx.mul_raw(value, low))
                });
            sum = ctx.add_raw(sum, ctx.mul_raw(inner, self.high[h]));
        }
        sum
    }
}

pub(crate) fn equality_weights(
    point: &[field::Fp<2>],
    cfg: &field::FpCtx<2>,
) -> Result<EqualityWeights<field::Fp<2>>, crate::sumcheck::SumcheckError> {
    let (low, high) =
        make_equality_factors(point, cfg).map_err(|_| SumcheckError::InvalidProductDimensions)?;
    Ok(EqualityWeights::from_tables(
        low.evaluations,
        high.evaluations,
        cfg,
    ))
}

#[cfg(test)]
pub(crate) fn evaluate_sha_factors(
    instances: &[field::Fp<2>],
    sha: &[field::Fp<2>],
    point: &[field::Fp<2>],
    cfg: &field::FpCtx<2>,
) -> Result<field::Fp<2>, crate::sumcheck::SumcheckError> {
    FactoredMultilinearExtension::from_factors(point.len(), sha, instances, cfg)
        .and_then(|mle| mle.evaluate(point, cfg))
        .map_err(|_| SumcheckError::InvalidProductDimensions)
}

/// `Σ_j tail[j] · eq(point, offset + j)` where `tail` is geometric on `runs`
/// (`tail[start + k] = base · 2^k`) and arbitrary elsewhere. With
/// `eq(offset + j) = low[(offset + j) mod L] · high[(offset + j) / L]`, a run's
/// intersection with one high block `[lo, hi)` contributes
/// `base_t · high[b] · (Q[lo] − 2^(hi − lo) · Q[hi])` where
/// `Q[t] = Σ_{i ≥ t} 2^(i − t) · low[i]` (a backward recurrence, no inverses),
/// so the runs cost a few multiplications each; columns outside every run pay
/// two multiplications.
#[cfg(test)]
pub(crate) fn evaluate_tail_by_runs(
    cfg: &field::FpCtx<2>,
    ctx: &field::FpCtx<2>,
    offset: usize,
    tail: &[field::Fp<2>],
    runs: &[(usize, usize, field::Fp<2>)],
    equality: &EqualityWeights<field::Fp<2>>,
) -> field::Fp<2> {
    let (low, high) = (ctx.raw_vec(equality.low()), ctx.raw_vec(equality.high()));
    let block = low.len();
    let shift = block.ilog2();
    let mask = block - 1;
    // Q[t] = low[t] + 2 · Q[t + 1], Q[block] = 0.
    let mut suffix = vec![0 as u128; block + 1];
    for t in (0..block).rev() {
        let doubled = ctx.add_raw(suffix[t + 1], suffix[t + 1]);
        suffix[t] = ctx.add_raw(low[t], doubled);
    }
    // 2^k in Montgomery form for k ≤ block.
    let mut pow2 = Vec::with_capacity(block + 1);
    pow2.push(ctx.native_residue(1));
    for k in 0..block {
        pow2.push(ctx.add_raw(pow2[k], pow2[k]));
    }
    let mut sum = 0 as u128;
    let mut cursor = 0usize;
    let scalar = |sum: &mut u128, from: usize, to: usize| {
        for column in from..to {
            let index = offset + column;
            let weight = ctx.mul_raw(low[index & mask], high[index >> shift]);
            *sum = ctx.add_raw(*sum, ctx.mul_raw(weight, ctx.raw(&tail[column])));
        }
    };
    for &(start, len, ref base) in runs {
        scalar(&mut sum, cursor, start);
        let mut base_t = ctx.raw(base);
        let mut index = offset + start;
        let end = offset + start + len;
        while index < end {
            let b = index >> shift;
            let lo = index & mask;
            let hi = (lo + (end - index)).min(block);
            let part = ctx.sub_raw(suffix[lo], ctx.mul_raw(pow2[hi - lo], suffix[hi]));
            sum = ctx.add_raw(sum, ctx.mul_raw(ctx.mul_raw(base_t, part), high[b]));
            base_t = ctx.mul_raw(base_t, pow2[hi - lo]);
            index += hi - lo;
        }
        cursor = start + len;
    }
    scalar(&mut sum, cursor, tail.len());
    crate::utils::delayed_reduction::element(&cfg, sum)
}

/// Challenge-dependent seeds, already factored by the relation adapter. Public
/// corrections may alias and are added, never overwritten.
pub(crate) struct CompositeRows<'a> {
    pub instances: &'a [u128],
    pub local: &'a [u128],
    pub tail_rows: &'a [u128],
    pub correction_columns: &'a [usize],
    pub corrections: &'a [u128],
    pub constant: u128,
}
pub(crate) struct CompositeBinding<'a, 't> {
    pub field: &'a field::FpCtx<2>,
    pub tape: &'a mut circuit::linear_map::circuit::PreparedWengertEvaluator<'t>,
    pub num_vars: usize,
    pub tail_offset: usize,
    pub tail_columns: usize,
}
impl CompositeBinding<'_, '_> {
    fn validate(&self, rows: &CompositeRows<'_>) -> Result<(), crate::sumcheck::SumcheckError> {
        let domain = 1usize
            .checked_shl(self.num_vars as u32)
            .ok_or(SumcheckError::InvalidProductDimensions)?;
        if !rows.instances.len().is_power_of_two()
            || rows.instances.len().checked_mul(rows.local.len()) != Some(self.tail_offset)
            || self
                .tail_offset
                .checked_add(self.tail_columns)
                .is_none_or(|n| n > domain)
            || rows.corrections.len() != rows.correction_columns.len()
            || rows
                .correction_columns
                .iter()
                .any(|&i| i >= self.tail_columns)
            || rows.tail_rows.len() != 3 * self.tape.row_count()
            || self.tail_columns != self.tape.column_count()
        {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        Ok(())
    }
}
impl CompositeBinding<'_, '_> {
    pub(crate) fn bind_rows(
        &mut self,
        rows: &CompositeRows<'_>,
    ) -> Result<CompositeCoefficients, crate::sumcheck::SumcheckError> {
        let mut out = CompositeCoefficients {
            num_vars: self.num_vars,
            instance_weights: Vec::new(),
            sha_local_evaluations: Vec::new(),
            p256_tail: MontgomeryTail::new(self.field.clone(), 0, vec![], vec![])
                .expect("empty tail is valid"),
            constant_weight: self.field.zero(),
        };
        self.bind_rows_into(rows, &mut out)?;
        Ok(out)
    }
    pub(crate) fn bind_rows_into(
        &mut self,
        rows: &CompositeRows<'_>,
        out: &mut CompositeCoefficients,
    ) -> Result<(), crate::sumcheck::SumcheckError> {
        self.validate(rows)?;
        let field = self.field;
        let mut tail = self
            .tape
            .adjoint_map_structured(rows.tail_rows.len() / 3, |r, k| {
                raw_to_words(rows.tail_rows[3 * r + k])
            })
            .map_err(|_| SumcheckError::InvalidProductDimensions)?;
        let corrections: Vec<_> = rows
            .correction_columns
            .iter()
            .zip(rows.corrections)
            .map(|(&column, &value)| (column, raw_to_words(value)))
            .collect();
        tail.add_sparse(&corrections)
            .map_err(|_| SumcheckError::InvalidProductDimensions)?;
        out.p256_tail = tail;
        let fill = |src: &[u128], dst: &mut Vec<field::Fp<2>>| {
            dst.clear();
            dst.extend(
                src.iter()
                    .map(|&r| crate::sumcheck::inner::native::shared_raw(field, r)),
            );
        };
        fill(rows.instances, &mut out.instance_weights);
        fill(rows.local, &mut out.sha_local_evaluations);
        out.num_vars = self.num_vars;
        out.constant_weight = crate::sumcheck::inner::native::shared_raw(field, rows.constant);
        Ok(())
    }
    pub(crate) fn evaluate_at(
        &mut self,
        rows: &CompositeRows<'_>,
        point: &[field::Fp<2>],
    ) -> Result<field::Fp<2>, crate::sumcheck::SumcheckError> {
        self.validate(rows)?;
        if point.len() != self.num_vars {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        let field = self.field;
        let columns = {
            let _s = tracing::info_span!("ecdsa:ce_columns").entered();
            TailEqualityColumns::new(field, self.tail_offset, self.tail_columns, point)
        };
        let mut value = {
            let _s = tracing::info_span!("ecdsa:ce_sha_eval").entered();
            let split = rows.instances.len().ilog2() as usize;
            let instances = RawEqualityWeights::new(field, &point[..split]);
            let local = RawEqualityWeights::new(field, &point[split..]);
            field.mul_raw(instances.dot(rows.instances), local.dot(rows.local))
        };
        value = field.add_raw(value, field.mul_raw(rows.constant, columns.eq_at(0)));
        let tail = {
            let _s = tracing::info_span!("ecdsa:ce_forward").entered();
            self.tape
                .evaluate_bilinear_map(
                    rows.tail_rows.len() / 3,
                    |r, k| raw_to_words(rows.tail_rows[3 * r + k]),
                    &columns,
                )
                .map_err(|_| SumcheckError::InvalidProductDimensions)?
        };
        value = field.add_raw(value, field.raw(&tail));
        for (&column, &weight) in rows.correction_columns.iter().zip(rows.corrections) {
            value = field.add_raw(
                value,
                field.mul_raw(weight, columns.eq_at(self.tail_offset + column)),
            );
        }
        Ok(crate::sumcheck::inner::native::shared_raw(field, value))
    }
}
