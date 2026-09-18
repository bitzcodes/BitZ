//! Sparse and block-selector binding for native Spartan inputs.
use crate::piop::spartan::{
    baby_bear_mul::{BABY_BEAR_MODULUS, BabyBearMulCoefficient},
    matrix::{
        BlockSelectorLayout, PrefixUnivariateRowFactors, PreparedConstraintMatrices,
        SpartanMatrixCoefficient,
    },
    u64_mul::{U64_MUL_LIMB_BASE, U64MulCoefficient},
};
use crate::sumcheck::bridge::PreparedBinding;
#[cfg(feature = "parallel")]
use crate::sumcheck::inner::native::parallel;
use crate::sumcheck::inner::native::{
    BlockScales, NativeWeights, RawFieldStorage, eq_table_raw_into,
};
use circuit::linear_map::ColumnValues;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
/// Aggregates a [`BlockSelectorLayout`] into per-block scales for `ρ`.
pub(crate) fn block_scales_raw<C: RawMontyCoefficient>(
    ctx: &field::FpCtx<2>,
    layout: &BlockSelectorLayout<C>,
    rho: u128,
    num_column_vars: usize,
) -> BlockScales {
    let blocks = (1usize << num_column_vars) / layout.block_len;
    let prepared = C::prepare_raw(ctx);
    let mut scales = vec![None; blocks];
    let factors = [ctx.one_raw(), rho, ctx.mul_raw(rho, rho)];
    for (runs, factor) in [
        (&layout.a, factors[0]),
        (&layout.b, factors[1]),
        (&layout.c, factors[2]),
    ] {
        for run in runs {
            let scale = ctx.mul_raw(
                factor,
                run.coefficient.raw_scale(&prepared, ctx.one_raw(), ctx),
            );
            let slot = &mut scales[run.start / layout.block_len];
            *slot = Some(match *slot {
                Some(existing) => ctx.add_raw(existing, scale),
                None => scale,
            });
        }
    }
    BlockScales {
        block_len: layout.block_len,
        rows: layout.rows,
        scales,
    }
}

// ---------------------------------------------------------------------------
// Matrix binding
// ---------------------------------------------------------------------------

/// Coefficient scaling on raw residues: the prover-side twin of
/// [`SpartanMatrixCoefficient::scale`], with per-proof constants prepared once
/// instead of per matrix entry.
pub trait RawMontyCoefficient: Sync {
    /// Constants derived from the field context once per binding.
    type Prepared: Sync;

    fn prepare_raw(ctx: &field::FpCtx<2>) -> Self::Prepared;

    /// `self · value`.
    fn raw_scale(&self, prepared: &Self::Prepared, value: u128, ctx: &field::FpCtx<2>) -> u128;
}

impl RawMontyCoefficient for bool {
    type Prepared = ();

    fn prepare_raw(_ctx: &field::FpCtx<2>) -> Self::Prepared {}

    #[inline(always)]
    fn raw_scale(&self, _prepared: &(), value: u128, _ctx: &field::FpCtx<2>) -> u128 {
        if *self { value } else { 0 }
    }
}

impl RawMontyCoefficient for field::Fp<2> {
    type Prepared = ();

    fn prepare_raw(_ctx: &field::FpCtx<2>) -> Self::Prepared {}

    #[inline(always)]
    fn raw_scale(&self, _prepared: &(), value: u128, ctx: &field::FpCtx<2>) -> u128 {
        ctx.mul_raw(ctx.raw(self), value)
    }
}

impl RawMontyCoefficient for U64MulCoefficient {
    /// The public limb base `2^64` as a residue of the runtime field.
    type Prepared = u128;

    fn prepare_raw(ctx: &field::FpCtx<2>) -> Self::Prepared {
        ctx.native_residue_u128(U64_MUL_LIMB_BASE)
    }

    #[inline(always)]
    fn raw_scale(&self, limb_base: &u128, value: u128, ctx: &field::FpCtx<2>) -> u128 {
        match self {
            Self::One => value,
            Self::LimbBase => ctx.mul_raw(*limb_base, value),
        }
    }
}

impl RawMontyCoefficient for BabyBearMulCoefficient {
    /// The embedded BabyBear modulus as a residue of the runtime field.
    type Prepared = u128;

    fn prepare_raw(ctx: &field::FpCtx<2>) -> Self::Prepared {
        ctx.native_residue(BABY_BEAR_MODULUS)
    }

    #[inline(always)]
    fn raw_scale(&self, modulus: &u128, value: u128, ctx: &field::FpCtx<2>) -> u128 {
        match self {
            Self::One => value,
            Self::Modulus => ctx.mul_raw(*modulus, value),
        }
    }
}

/// `D(j) = Σ_i row_weights[i] (A[i,j] + ρ B[i,j] + ρ² C[i,j])` over the padded
/// column domain, used by differential tests.
#[cfg(test)]
pub(crate) fn reference_native_binding<C>(
    ctx: &field::FpCtx<2>,
    matrices: &PreparedConstraintMatrices<field::Fp<2>, C>,
    row_weights: &[u128],
    rho: u128,
) -> Vec<u128>
where
    C: SpartanMatrixCoefficient<field::Fp<2>> + RawMontyCoefficient,
{
    let mut evaluations = Vec::new();
    contract_native_columns(ctx, matrices, row_weights, rho, &mut evaluations);
    evaluations
}

fn contract_native_columns<C>(
    ctx: &field::FpCtx<2>,
    matrices: &PreparedConstraintMatrices<field::Fp<2>, C>,
    row_weights: &[u128],
    rho: u128,
    evaluations: &mut Vec<u128>,
) where
    C: SpartanMatrixCoefficient<field::Fp<2>> + RawMontyCoefficient,
{
    debug_assert_eq!(row_weights.len(), 1usize << matrices.num_row_vars());
    let rho_squared = ctx.mul_raw(rho, rho);
    let prepared = C::prepare_raw(ctx);
    let m = matrices.matrices();
    let live_columns = m
        .a()
        .column_count()
        .min(m.b().column_count())
        .min(m.c().column_count());
    let dot = |source: &circuit::linear_map::CscMatrix<Box<[C]>>, column| {
        circuit::linear_map::contraction::segment_dot(
            source.column(column).unwrap(),
            0,
            |row, c| c.raw_scale(&prepared, row_weights[row], ctx),
            |a, b| ctx.add_raw(a, b),
        )
    };
    let evaluate = |column| {
        let mut sum = dot(m.a(), column);
        for (source, scale) in [(m.b(), rho), (m.c(), rho_squared)] {
            if !source.column(column).unwrap().is_empty() {
                sum = ctx.add_raw(sum, ctx.mul_raw(scale, dot(source, column)));
            }
        }
        sum
    };
    evaluations.resize(1usize << matrices.num_column_vars(), 0);
    evaluations[live_columns..].fill(0);
    let live = &mut evaluations[..live_columns];
    circuit::linear_map::contraction::columns_into(live, cfg!(feature = "parallel"), evaluate);
}

/// The row functional binding the matrices for the inner sumcheck.
#[derive(Clone, Copy)]
pub(crate) enum RowFunctional<'a> {
    /// `eq(·, point)` over the padded row domain (the standard outer).
    Point(&'a [field::Fp<2>]),
    /// The prefix-univariate factors (the univariate-skip outer).
    Prefix(&'a PrefixUnivariateRowFactors<field::Fp<2>>),
    /// Explicit weights over the complete logical row domain.
    Explicit(&'a [field::Fp<2>]),
    /// Independently weighted A, B, C rows; no common rho factor is applied.
    Independent([&'a [field::Fp<2>]; 3]),
}

/// Materializes the prefix-univariate row weights `prefix[s] · tail(x)` in
/// canonical row order (`s + 2^K x`), the raw twin of
/// `PrefixUnivariateRowFactors::materialize`.
pub(crate) fn prefix_row_weights_raw(
    ctx: &field::FpCtx<2>,
    factors: &PrefixUnivariateRowFactors<field::Fp<2>>,
) -> Vec<u128> {
    let mut weights = Vec::new();
    prefix_row_weights_raw_into(ctx, factors, &mut weights);
    weights
}
fn prefix_row_weights_raw_into(
    ctx: &field::FpCtx<2>,
    factors: &PrefixUnivariateRowFactors<field::Fp<2>>,
    weights: &mut Vec<u128>,
) {
    let parts = factors.parts();
    let prefix = ctx.raw_vec(parts.prefix);
    let tail_low = ctx.raw_vec(parts.tail_low);
    let tail_high = ctx.raw_vec(parts.tail_high);
    let block_len = 1usize << parts.skip_vars;
    let low_mask = tail_low.len() - 1;
    let total_rows = 1usize << parts.num_row_vars;
    weights.resize(total_rows, 0);
    let fill = |suffix: usize, block: &mut [u128]| {
        let tail = ctx.mul_raw(
            tail_low[suffix & low_mask],
            tail_high[suffix >> parts.tail_low_vars],
        );
        for (weight, &prefix_weight) in block.iter_mut().zip(&prefix) {
            *weight = ctx.mul_raw(prefix_weight, tail);
        }
    };
    #[cfg(feature = "parallel")]
    if parallel(total_rows) {
        weights
            .par_chunks_mut(block_len)
            .enumerate()
            .for_each(|(suffix, block)| fill(suffix, block));
        return;
    }
    for (suffix, block) in weights.chunks_mut(block_len).enumerate() {
        fill(suffix, block);
    }
}

/// Disjoint
/// unit-selector matrices are filled one prefix block at a time, in parallel,
/// without materializing the row-weight tensor; other layouts materialize
/// the weights and bind generically.
#[cfg(test)]
pub(crate) fn reference_prefix_binding<C>(
    ctx: &field::FpCtx<2>,
    matrices: &PreparedConstraintMatrices<field::Fp<2>, C>,
    factors: &PrefixUnivariateRowFactors<field::Fp<2>>,
    rho: u128,
) -> Vec<u128>
where
    C: SpartanMatrixCoefficient<field::Fp<2>> + RawMontyCoefficient,
{
    let mut evaluations = Vec::new();
    contract_prefix_selectors(ctx, matrices, factors, rho, &mut evaluations);
    evaluations
}
fn contract_prefix_selectors<C>(
    ctx: &field::FpCtx<2>,
    matrices: &PreparedConstraintMatrices<field::Fp<2>, C>,
    factors: &PrefixUnivariateRowFactors<field::Fp<2>>,
    rho: u128,
    evaluations: &mut Vec<u128>,
) where
    C: SpartanMatrixCoefficient<field::Fp<2>> + RawMontyCoefficient,
{
    let parts = factors.parts();
    let prefix = ctx.raw_vec(parts.prefix);
    let tail_low = ctx.raw_vec(parts.tail_low);
    let tail_high = ctx.raw_vec(parts.tail_high);
    let block_len = 1usize << parts.skip_vars;
    let low_mask = tail_low.len() - 1;
    let tail_weight = |suffix: usize| -> u128 {
        ctx.mul_raw(
            tail_low[suffix & low_mask],
            tail_high[suffix >> parts.tail_low_vars],
        )
    };
    let domain = 1usize << matrices.num_column_vars();

    let layout = matrices.selector_layout().filter(|[rows, a, b, c]| {
        let mut offsets = [*a, *b, *c];
        offsets.sort_unstable();
        offsets[0] + rows <= offsets[1]
            && offsets[1] + rows <= offsets[2]
            && offsets[2] + rows <= domain
    });
    let Some([rows, a_offset, b_offset, c_offset]) = layout else {
        let weights = prefix_row_weights_raw(ctx, factors);
        contract_native_columns(ctx, matrices, &weights, rho, evaluations);
        return;
    };

    let rho_squared = ctx.mul_raw(rho, rho);
    evaluations.resize(domain, 0);
    evaluations.fill(0);
    // Carve the three disjoint selector regions out of the table.
    let mut order = [(a_offset, 0usize), (b_offset, 1), (c_offset, 2)];
    order.sort_unstable();
    let (first, rest) = evaluations[order[0].0..].split_at_mut(rows);
    let (second, rest) = rest[order[1].0 - order[0].0 - rows..].split_at_mut(rows);
    let third = &mut rest[order[2].0 - order[1].0 - rows..][..rows];
    let mut regions: [Option<&mut [u128]>; 3] = [None, None, None];
    regions[order[0].1] = Some(first);
    regions[order[1].1] = Some(second);
    regions[order[2].1] = Some(third);
    let [Some(a_region), Some(b_region), Some(c_region)] = regions else {
        unreachable!("every selector region is assigned exactly once");
    };

    let fill = |suffix: usize, a: &mut [u128], b: &mut [u128], c: &mut [u128]| {
        let tail = tail_weight(suffix);
        for (((a, b), c), &prefix_weight) in a.iter_mut().zip(b).zip(c).zip(&prefix) {
            let weight = ctx.mul_raw(prefix_weight, tail);
            *a = weight;
            *b = ctx.mul_raw(rho, weight);
            *c = ctx.mul_raw(rho_squared, weight);
        }
    };
    #[cfg(feature = "parallel")]
    if parallel(rows) {
        (
            a_region.par_chunks_mut(block_len),
            b_region.par_chunks_mut(block_len),
            c_region.par_chunks_mut(block_len),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(suffix, (a, b, c))| fill(suffix, a, b, c));
        return;
    }
    for (suffix, ((a, b), c)) in a_region
        .chunks_mut(block_len)
        .zip(b_region.chunks_mut(block_len))
        .zip(c_region.chunks_mut(block_len))
        .enumerate()
    {
        fill(suffix, a, b, c);
    }
}

/// Retains the caller's field and the validated sparse/selector relation.
pub(crate) struct NativeBinding<'a, C> {
    field: &'a field::FpCtx<2>,
    matrices: &'a PreparedConstraintMatrices<field::Fp<2>, C>,
    rho: field::Fp<2>,
    row_workspace: Vec<u128>,
}
impl<'a, C> NativeBinding<'a, C>
where
    C: SpartanMatrixCoefficient<field::Fp<2>> + RawMontyCoefficient,
{
    pub(crate) fn new(
        field: &'a field::FpCtx<2>,
        matrices: &'a PreparedConstraintMatrices<field::Fp<2>, C>,
        rho: field::Fp<2>,
    ) -> Self {
        Self {
            field,
            matrices,
            rho,
            row_workspace: Vec::new(),
        }
    }
    fn validate(&self, rows: &RowFunctional<'_>) -> Result<(), crate::sumcheck::SumcheckError> {
        let vars = self.matrices.num_row_vars();
        let valid = match rows {
            RowFunctional::Point(p) => p.len() == vars,
            RowFunctional::Prefix(p) => p.parts().num_row_vars == vars,
            RowFunctional::Explicit(w) => w.len() == 1 << vars,
            RowFunctional::Independent(ws) => ws.iter().all(|w| w.len() == 1 << vars),
        };
        if valid {
            Ok(())
        } else {
            Err(crate::sumcheck::SumcheckError::InvalidProductDimensions)
        }
    }
}
impl<C> NativeBinding<'_, C>
where
    C: SpartanMatrixCoefficient<field::Fp<2>> + RawMontyCoefficient,
{
    pub(crate) fn bind_structured_rows(
        &mut self,
        rows: &RowFunctional<'_>,
    ) -> Result<NativeWeights, crate::sumcheck::SumcheckError> {
        let mut out = NativeWeights::Dense {
            matrix: Vec::new(),
            live: 0,
        };
        self.bind_structured_rows_into(rows, &mut out)?;
        Ok(out)
    }
    pub(crate) fn bind_structured_rows_into(
        &mut self,
        rows: &RowFunctional<'_>,
        out: &mut NativeWeights,
    ) -> Result<(), crate::sumcheck::SumcheckError> {
        self.validate(rows)?;
        let ctx = self.field;
        let live = self.matrices.matrices().column_count();
        let num_vars = self.matrices.num_column_vars();
        let rho = ctx.raw(&self.rho);
        if let RowFunctional::Independent(weights) = rows {
            // General independent seeds cannot share a single selector factor.
            let matrix = match out {
                NativeWeights::Dense {
                    matrix,
                    live: out_live,
                } => {
                    *out_live = live;
                    matrix
                }
                _ => {
                    *out = NativeWeights::Dense {
                        matrix: Vec::new(),
                        live,
                    };
                    let NativeWeights::Dense { matrix, .. } = out else {
                        unreachable!()
                    };
                    matrix
                }
            };
            matrix.resize(1 << num_vars, 0);
            matrix.fill(0);
            let prepared = C::prepare_raw(ctx);
            let m = self.matrices.matrices();
            for (source, w) in [m.a(), m.b(), m.c()].into_iter().zip(weights) {
                for (j, dst) in matrix[..live].iter_mut().enumerate() {
                    for (row, c) in source.column(j).unwrap() {
                        *dst = ctx.add_raw(*dst, c.raw_scale(&prepared, ctx.raw(&w[row]), ctx));
                    }
                }
            }
            return Ok(());
        }
        if let Some(layout) = self.matrices.block_selector().filter(|l| l.block_len >= 2) {
            let scales = block_scales_raw(ctx, layout, rho, num_vars);
            if let NativeWeights::Blocks {
                weights: dst,
                scales: dst_scales,
                live: dst_live,
                num_vars: dst_vars,
            } = out
            {
                fill_weights(ctx, rows, dst);
                *dst_scales = scales;
                *dst_live = live;
                *dst_vars = num_vars;
            } else {
                let mut weights = Vec::new();
                fill_weights(ctx, rows, &mut weights);
                *out = NativeWeights::Blocks {
                    weights,
                    scales,
                    live,
                    num_vars,
                };
            }
        } else {
            if !matches!(out, NativeWeights::Dense { .. }) {
                *out = NativeWeights::Dense {
                    matrix: Vec::new(),
                    live,
                };
            }
            let NativeWeights::Dense {
                matrix,
                live: dst_live,
            } = out
            else {
                unreachable!()
            };
            *dst_live = live;
            if let RowFunctional::Prefix(factors) = rows {
                // Preserve the disjoint-selector streaming kernel.
                contract_prefix_selectors(ctx, self.matrices, factors, rho, matrix);
            } else {
                fill_weights(ctx, rows, &mut self.row_workspace);
                contract_native_columns(ctx, self.matrices, &self.row_workspace, rho, matrix);
            }
        }
        Ok(())
    }
    pub(crate) fn evaluate_structured_at(
        &mut self,
        rows: &RowFunctional<'_>,
        point: &[field::Fp<2>],
    ) -> Result<field::Fp<2>, crate::sumcheck::SumcheckError> {
        self.validate(rows)?;
        let result = match rows {
            RowFunctional::Point(p) => self
                .matrices
                .structured()
                .evaluate_equality(p, &self.rho, point)
                .map_err(crate::sumcheck::SumcheckError::from),
            RowFunctional::Prefix(p) => self
                .matrices
                .structured()
                .evaluate_prefix(p, &self.rho, point)
                .map_err(crate::sumcheck::SumcheckError::from),
            RowFunctional::Explicit(w) => self.matrices.binding(&self.rho).evaluate_at(w, point),
            RowFunctional::Independent(weights) => {
                if point.len() != self.matrices.num_column_vars() {
                    return Err(crate::sumcheck::SumcheckError::InvalidProductDimensions);
                }
                let columns = super::dense::EqualityColumns::new(
                    self.field,
                    point,
                    self.matrices.matrices().column_count(),
                );
                let prepared = C::prepare_raw(self.field);
                let m = self.matrices.matrices();
                let mut total = 0;
                for (source, w) in [m.a(), m.b(), m.c()].into_iter().zip(weights) {
                    for (j, column) in source.columns().enumerate() {
                        let mut value = 0;
                        for (row, c) in column {
                            value = self.field.add_raw(
                                value,
                                c.raw_scale(&prepared, self.field.raw(&w[row]), self.field),
                            );
                        }
                        total = self.field.add_raw(
                            total,
                            self.field
                                .mul_raw(value, self.field.raw(&columns.scalar(j))),
                        );
                    }
                }
                return Ok(crate::sumcheck::inner::native::shared_raw(
                    self.field, total,
                ));
            }
        };
        result.map_err(|_| crate::sumcheck::SumcheckError::InvalidProductDimensions)
    }
}

fn fill_weights(ctx: &field::FpCtx<2>, rows: &RowFunctional<'_>, out: &mut Vec<u128>) {
    match rows {
        RowFunctional::Point(p) => eq_table_raw_into(ctx, &ctx.raw_vec(p), out),
        RowFunctional::Prefix(p) => prefix_row_weights_raw_into(ctx, p, out),
        RowFunctional::Explicit(w) => {
            out.clear();
            out.extend(w.iter().map(|v| ctx.raw(v)));
        }
        RowFunctional::Independent(_) => unreachable!("independent rows are contracted separately"),
    }
}

impl<C: SpartanMatrixCoefficient<field::Fp<2>> + RawMontyCoefficient>
    super::PreparedBinding<field::FpCtx<2>> for NativeBinding<'_, C>
{
    type Bound = NativeWeights;
    fn bind_rows(
        &mut self,
        rows: &[field::Fp<2>],
    ) -> Result<NativeWeights, crate::sumcheck::SumcheckError> {
        self.bind_structured_rows(&RowFunctional::Explicit(rows))
    }
    fn bind_rows_into(
        &mut self,
        rows: &[field::Fp<2>],
        out: &mut NativeWeights,
    ) -> Result<(), crate::sumcheck::SumcheckError> {
        self.bind_structured_rows_into(&RowFunctional::Explicit(rows), out)
    }
    fn evaluate_at(
        &mut self,
        rows: &[field::Fp<2>],
        point: &[field::Fp<2>],
    ) -> Result<field::Fp<2>, crate::sumcheck::SumcheckError> {
        self.evaluate_structured_at(&RowFunctional::Explicit(rows), point)
    }
}
