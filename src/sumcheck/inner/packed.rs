//! Packed-bit and small-value prefix arithmetic for the shared inner sumcheck.
//!
//! The verifier sees an ordinary degree-two sumcheck. The prefix length is a
//! prover-only implementation choice: every round still absorbs the same three
//! field coefficients, performs the same optional grinding step, and samples
//! the same challenge as the field-only prover.

use crate::piop::spartan::SpartanField as _;
#[cfg(test)]
use field::Uint;
use field::{BatchMulAcc, MergeAccumulator, Reduce};
use field::{RingOps, Uint as FieldUint};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use field::{CtMask, CtSelect};

use crate::piop::spartan::{SpartanField, bitz::SpartanBitzField, grinding::GrindingDomain};

#[cfg(test)]
use crate::sumcheck::SumcheckProof;
use crate::sumcheck::{SumcheckError, arithmetic::merge_accumulators};

type Field = SpartanBitzField;
type FieldConfig = field::FpCtx<2>;
type LinearAccumulator = field::FpLinearAcc<2, 1>;
type ProductAccumulator = <field::FpCtx<2> as BatchMulAcc<Field>>::Accumulator;
type RawMontgomery = [u64; 2];

/// Lazy Bit source for the flat assignment table.
///
/// A packed word slice retains strict canonical-padding validation. A callback
/// source is useful when the flat bit table is itself spread across packed
/// column rows; it is queried only at indices below `live_len`.
pub(crate) trait Sha256InnerBitSource: Sync {
    fn bit_at(&self, index: usize) -> Result<u64, SumcheckError>;

    /// The `count <= 64` bits at `index .. index + count`, bit `i` of the result
    /// being the bit at `index + i`. Callers read aligned prefix blocks, so a
    /// packed source serves this from one or two words; the default reads bit
    /// by bit.
    fn bits_at(&self, index: usize, count: usize) -> Result<u64, SumcheckError> {
        debug_assert!(count <= u64::BITS as usize);
        let mut word = 0u64;
        for i in 0..count {
            let bit = self.bit_at(index + i)?;
            if bit > 1 {
                return Err(SumcheckError::InvalidProductDimensions);
            }
            word |= bit << i;
        }
        Ok(word)
    }

    fn validate_shape(&self, _live_len: usize, _table_len: usize) -> Result<(), SumcheckError> {
        Ok(())
    }
}

#[inline]
fn low_bits_mask(count: usize) -> u64 {
    if count >= u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << count) - 1
    }
}

impl Sha256InnerBitSource for [u64] {
    #[inline]
    fn bit_at(&self, index: usize) -> Result<u64, SumcheckError> {
        Ok((self[index / u64::BITS as usize] >> (index % u64::BITS as usize)) & 1)
    }

    #[inline]
    fn bits_at(&self, index: usize, count: usize) -> Result<u64, SumcheckError> {
        let word = index / u64::BITS as usize;
        let shift = index % u64::BITS as usize;
        let mut value = self[word] >> shift;
        if shift + count > u64::BITS as usize && word + 1 < self.len() {
            value |= self[word + 1] << (u64::BITS as usize - shift);
        }
        Ok(value & low_bits_mask(count))
    }

    fn validate_shape(&self, live_len: usize, table_len: usize) -> Result<(), SumcheckError> {
        let live_words = live_len.div_ceil(u64::BITS as usize);
        let domain_words = table_len.div_ceil(u64::BITS as usize);
        if self.len() != live_words && self.len() != domain_words {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        let used_live_bits = live_len % u64::BITS as usize;
        if (used_live_bits != 0 && self[live_words - 1] >> used_live_bits != 0)
            || self[live_words..].iter().any(|word| *word != 0)
        {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        Ok(())
    }
}

impl Sha256InnerBitSource for Vec<u64> {
    #[inline]
    fn bit_at(&self, index: usize) -> Result<u64, SumcheckError> {
        self.as_slice().bit_at(index)
    }

    #[inline]
    fn bits_at(&self, index: usize, count: usize) -> Result<u64, SumcheckError> {
        self.as_slice().bits_at(index, count)
    }

    fn validate_shape(&self, live_len: usize, table_len: usize) -> Result<(), SumcheckError> {
        self.as_slice().validate_shape(live_len, table_len)
    }
}

impl<F> Sha256InnerBitSource for F
where
    F: Fn(usize) -> Result<u64, SumcheckError> + Sync,
{
    #[inline]
    fn bit_at(&self, index: usize) -> Result<u64, SumcheckError> {
        self(index)
    }
}

/// A flat bit table stored column-major in packed words: index
/// `column · 2^row_vars + row` is bit `row % 64` of `columns[column][row / 64]`
/// (the ECDSA assignment layout). Every column holds a power of two rows, at
/// least 64, so an aligned block of at most 64 bits never crosses a word.
pub(crate) struct ColumnMajorPackedBits<'a> {
    columns: &'a [Vec<u64>],
    row_vars: usize,
}

impl<'a> ColumnMajorPackedBits<'a> {
    pub(crate) fn new(columns: &'a [Vec<u64>], row_vars: usize) -> Self {
        debug_assert!(row_vars >= 6);
        Self { columns, row_vars }
    }
}

impl Sha256InnerBitSource for ColumnMajorPackedBits<'_> {
    #[inline]
    fn bit_at(&self, index: usize) -> Result<u64, SumcheckError> {
        let row = index & ((1usize << self.row_vars) - 1);
        Ok((self.columns[index >> self.row_vars][row / 64] >> (row % 64)) & 1)
    }

    #[inline]
    fn bits_at(&self, index: usize, count: usize) -> Result<u64, SumcheckError> {
        let row = index & ((1usize << self.row_vars) - 1);
        let shift = row % 64;
        debug_assert!(shift + count <= 64, "unaligned packed read");
        Ok((self.columns[index >> self.row_vars][row / 64] >> shift) & low_bits_mask(count))
    }
}

use crate::poly::mle::FactoredMultilinearExtension;

/// MLE access and specialized prefix folding for the inner prover.
pub(crate) trait InnerSumcheckMleSource: Sync {
    fn declared_num_vars(&self) -> Option<usize> {
        None
    }
    fn evaluation_at(&self, index: usize) -> Result<Field, SumcheckError>;

    fn validate_shape(
        &self,
        _live_len: usize,
        _field_cfg: &FieldConfig,
    ) -> Result<(), SumcheckError> {
        Ok(())
    }

    /// Whether every coefficient returned by this source is known to use the
    /// shared canonical field configuration. Callback sources are checked at
    /// every access; the factored source validates its two small factor tables
    /// once by the source adapter.
    fn evaluations_prevalidated(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn build_prefix_accumulators<const K: usize, H>(
        &self,
        num_vars: usize,
        live_len: usize,
        h_source: &H,
        field_cfg: &FieldConfig,
        zero: &Field,
    ) -> Result<PrefixAccumulators, SumcheckError>
    where
        H: Sha256InnerBitSource + ?Sized,
    {
        build_prefix_accumulators_generic::<K, _, _>(
            num_vars, live_len, self, h_source, field_cfg, zero,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_prefix_table<const K: usize>(
        &self,
        num_vars: usize,
        live_len: usize,
        challenges: &[Field],
        field_cfg: &FieldConfig,
        zero: &Field,
        one: &Field,
    ) -> Result<CompactPrefixVTable, SumcheckError> {
        fold_prefix_v_table_generic::<K, _>(
            num_vars, live_len, self, challenges, field_cfg, zero, one,
        )
    }
}

impl<F> InnerSumcheckMleSource for F
where
    F: Fn(usize) -> Result<Field, SumcheckError> + Sync,
{
    #[inline]
    fn evaluation_at(&self, index: usize) -> Result<Field, SumcheckError> {
        self(index)
    }
}

impl InnerSumcheckMleSource for FactoredMultilinearExtension<'_, Field> {
    fn declared_num_vars(&self) -> Option<usize> {
        Some(self.num_vars())
    }
    #[inline]
    fn evaluation_at(&self, index: usize) -> Result<Field, SumcheckError> {
        FactoredMultilinearExtension::evaluation_at(self, index)
            .map_err(|_| SumcheckError::InvalidProductDimensions)
    }

    fn validate_shape(
        &self,
        live_len: usize,
        field_cfg: &FieldConfig,
    ) -> Result<(), SumcheckError> {
        if live_len != self.live_len() {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        validate_factored_mle(self, field_cfg)
    }

    fn evaluations_prevalidated(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn build_prefix_accumulators<const K: usize, H>(
        &self,
        num_vars: usize,
        live_len: usize,
        h_source: &H,
        field_cfg: &FieldConfig,
        zero: &Field,
    ) -> Result<PrefixAccumulators, SumcheckError>
    where
        H: Sha256InnerBitSource + ?Sized,
    {
        build_factored_prefix_accumulators::<K, _>(
            num_vars, live_len, self, h_source, field_cfg, zero,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_prefix_table<const K: usize>(
        &self,
        num_vars: usize,
        live_len: usize,
        challenges: &[Field],
        field_cfg: &FieldConfig,
        zero: &Field,
        one: &Field,
    ) -> Result<CompactPrefixVTable, SumcheckError> {
        fold_factored_prefix_v_table::<K>(
            num_vars, live_len, self, challenges, field_cfg, zero, one,
        )
    }
}

/// Largest supported number of native-small prefix rounds.
pub const SHA256_INNER_PREFIX_MAX_VARS: usize = 4;

mod composite;
#[cfg(feature = "ecdsa")]
pub(crate) use composite::CompactCompositeMle;
mod state;
pub use state::PackedInput;

/// Prover output for the transcript-identical SHA-256 inner sumcheck.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Sha256InnerSumcheckOutput {
    /// Ordinary degree-two sumcheck proof; `prefix_vars` is intentionally absent.
    pub sumcheck_proof: SumcheckProof<Field, 3>,
    /// One nonce per round when grinding is enabled, otherwise empty.
    pub round_nonces: Vec<u64>,
    /// Fiat--Shamir point in low-coordinate-first order.
    pub eval_points: Vec<Field>,
    /// Terminal product `V(eval_points) * H(eval_points)`.
    pub final_claim: Field,
    /// Terminal evaluation of the field-valued `V` table.
    pub v_evaluation: Field,
    /// Terminal evaluation of the packed Bit `H` table.
    pub h_evaluation: Field,
}

#[cfg(test)]
impl Sha256InnerSumcheckOutput {
    pub(crate) fn from_inner(
        out: crate::sumcheck::inner::InnerSumcheckOutput<Field>,
        round_nonces: Vec<u64>,
    ) -> Self {
        let [v_evaluation, h_evaluation] = out.terminal_evaluations;
        Self {
            sumcheck_proof: out.proof,
            round_nonces,
            eval_points: out.point,
            final_claim: out.final_claim,
            v_evaluation,
            h_evaluation,
        }
    }
}

/// Typed proof-of-work domain for SHA-256 inner-sumcheck rounds.
pub(crate) enum Sha256InnerGrinding {}

impl GrindingDomain for Sha256InnerGrinding {
    const DOMAIN: &'static [u8] = b"bitz/spartan-sha256/grinding/inner/v1";
}

fn validate_inputs<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    num_vars: usize,
    live_len: usize,
    h_source: &H,
) -> Result<(), SumcheckError> {
    if K > SHA256_INNER_PREFIX_MAX_VARS
        || K > num_vars
        || num_vars >= usize::BITS as usize
        || live_len == 0
        || live_len > (1usize << num_vars)
    {
        return Err(SumcheckError::InvalidProductDimensions);
    }

    let table_len = 1usize << num_vars;
    h_source.validate_shape(live_len, table_len)?;

    Ok(())
}

#[inline]
fn validate_field_value(value: &Field, field_cfg: &FieldConfig) -> Result<(), SumcheckError> {
    validate_field_values(std::iter::once(value), field_cfg)
}

fn validate_field_values<'a>(
    values: impl IntoIterator<Item = &'a Field>,
    field_cfg: &FieldConfig,
) -> Result<(), SumcheckError> {
    let modulus = Field::canonical_modulus_encoding(field_cfg);
    for value in values {
        value
            .validate_element(&modulus)
            .map_err(|_| SumcheckError::NonCanonicalFieldElement)?;
    }
    Ok(())
}

fn validate_factored_mle(
    mle: &FactoredMultilinearExtension<'_, Field>,
    field_cfg: &FieldConfig,
) -> Result<(), SumcheckError> {
    validate_field_values(
        std::iter::once(mle.leading_value())
            .chain(mle.outer_factor())
            .chain(mle.inner_factor()),
        field_cfg,
    )
}

struct PrefixBuildState {
    partial_sums: Vec<LinearAccumulator>,
    v_values: Vec<Field>,
    v_scratch: Vec<Field>,
    h_values: Vec<i64>,
    h_scratch: Vec<i64>,
}

impl PrefixBuildState {
    fn new<const K: usize>(zero: &Field) -> Self {
        let prefix_size = 1usize << K;
        let extension_size = pow3(K);
        Self {
            partial_sums: (0..extension_size)
                .map(|_| linear_accumulator_zero())
                .collect(),
            v_values: vec![zero.clone(); prefix_size],
            v_scratch: vec![zero.clone(); extension_size],
            h_values: vec![0; prefix_size],
            h_scratch: vec![0; extension_size],
        }
    }
}

/// Per-worker state for the factored prefix pass. Interior prefix blocks are
/// accumulated without multiplying every local coefficient by `u_i`:
/// `d(beta) * H(beta)` is summed for a whole instance first, then scaled by
/// `u_i` once per ternary point.
struct FactoredPrefixBuildState {
    partial_sums: Vec<ProductAccumulator>,
    local_sums: Vec<LinearAccumulator>,
    d_values: Vec<Field>,
    d_scratch: Vec<Field>,
    h_values: Vec<i64>,
    h_scratch: Vec<i64>,
}

impl FactoredPrefixBuildState {
    fn new<const K: usize>(zero: &Field) -> Self {
        let prefix_size = 1usize << K;
        let extension_size = pow3(K);
        Self {
            partial_sums: (0..extension_size)
                .map(|_| product_accumulator_zero())
                .collect(),
            local_sums: (0..extension_size)
                .map(|_| linear_accumulator_zero())
                .collect(),
            d_values: vec![zero.clone(); prefix_size],
            d_scratch: vec![zero.clone(); extension_size],
            h_values: vec![0; prefix_size],
            h_scratch: vec![0; extension_size],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_factored_prefix_accumulators<const K: usize, H>(
    num_vars: usize,
    live_len: usize,
    coefficients: &FactoredMultilinearExtension<'_, Field>,
    h_source: &H,
    _field_cfg: &FieldConfig,
    zero: &Field,
) -> Result<PrefixAccumulators, SumcheckError>
where
    H: Sha256InnerBitSource + ?Sized,
{
    debug_assert!(K > 0);
    debug_assert_eq!(live_len, coefficients.live_len());
    debug_assert!(live_len <= 1usize << num_vars);
    let suffix_count = live_len.div_ceil(1usize << K);
    let extended = extend_block_coefficients::<K>(coefficients, zero, &_field_cfg);
    let extended = extended.as_deref();

    #[cfg(feature = "parallel")]
    let interior =
        if coefficients.outer_factor().len() >= 1 << 6 && rayon::current_num_threads() > 1 {
            (0..coefficients.outer_factor().len())
                .into_par_iter()
                .try_fold(
                    || FactoredPrefixBuildState::new::<K>(zero),
                    |mut state, instance| -> Result<_, SumcheckError> {
                        accumulate_factored_instance::<K, _>(
                            &mut state,
                            coefficients,
                            extended,
                            h_source,
                            instance,
                            zero,
                            _field_cfg,
                        )?;
                        Ok(state)
                    },
                )
                .try_reduce(
                    || FactoredPrefixBuildState::new::<K>(zero),
                    |left, right| Ok(merge_factored_prefix_states(left, right)),
                )?
        } else {
            accumulate_factored_instances_sequential::<K, _>(
                coefficients,
                extended,
                h_source,
                zero,
                _field_cfg,
            )?
        };

    #[cfg(not(feature = "parallel"))]
    let interior = accumulate_factored_instances_sequential::<K, _>(
        coefficients,
        extended,
        h_source,
        zero,
        _field_cfg,
    )?;

    // Only the block containing the shared cell, the at-most-one block at
    // each instance boundary, and an incomplete final block take this path.
    // Their mixed `u_i` factors cannot be pulled outside interpolation.
    #[cfg(feature = "parallel")]
    let boundary = if suffix_count >= 1 << 10 && rayon::current_num_threads() > 1 {
        (0..suffix_count)
            .into_par_iter()
            .filter(|&suffix| !factored_suffix_is_interior::<K>(coefficients, suffix))
            .try_fold(
                || PrefixBuildState::new::<K>(zero),
                |mut state, suffix| -> Result<_, SumcheckError> {
                    accumulate_suffix::<K, _, _>(
                        &mut state,
                        &_field_cfg,
                        live_len,
                        suffix,
                        coefficients,
                        h_source,
                        zero,
                    )?;
                    Ok(state)
                },
            )
            .try_reduce(
                || PrefixBuildState::new::<K>(zero),
                |left, right| Ok(merge_prefix_states(left, right)),
            )?
    } else {
        accumulate_factored_boundaries_sequential::<K, _>(
            suffix_count,
            live_len,
            coefficients,
            h_source,
            zero,
            _field_cfg,
        )?
    };

    #[cfg(not(feature = "parallel"))]
    let boundary = accumulate_factored_boundaries_sequential::<K, _>(
        suffix_count,
        live_len,
        coefficients,
        h_source,
        zero,
        _field_cfg,
    )?;

    let interior_values = interior
        .partial_sums
        .into_iter()
        .map(|accumulator| product_reduce(accumulator, &_field_cfg))
        .collect::<Result<Vec<_>, _>>()?;
    let boundary_values = boundary
        .partial_sums
        .into_iter()
        .map(|accumulator| linear_reduce(accumulator, &_field_cfg))
        .collect::<Result<Vec<_>, _>>()?;
    let beta_values = interior_values
        .into_iter()
        .zip(boundary_values)
        .map(|(interior, boundary)| _field_cfg.add(&(interior), &(&boundary)))
        .collect::<Vec<_>>();
    Ok(scatter_beta_values::<K>(&beta_values, zero, &_field_cfg))
}

fn accumulate_factored_instances_sequential<const K: usize, H>(
    coefficients: &FactoredMultilinearExtension<'_, Field>,
    extended: Option<&[ExtendedBlock]>,
    h_source: &H,
    zero: &Field,
    reducer: &field::FpCtx<2>,
) -> Result<FactoredPrefixBuildState, SumcheckError>
where
    H: Sha256InnerBitSource + ?Sized,
{
    let mut state = FactoredPrefixBuildState::new::<K>(zero);
    for instance in 0..coefficients.outer_factor().len() {
        accumulate_factored_instance::<K, _>(
            &mut state,
            coefficients,
            extended,
            h_source,
            instance,
            zero,
            reducer,
        )?;
    }
    Ok(state)
}

fn accumulate_factored_instance<const K: usize, H>(
    state: &mut FactoredPrefixBuildState,
    coefficients: &FactoredMultilinearExtension<'_, Field>,
    extended: Option<&[ExtendedBlock]>,
    h_source: &H,
    instance: usize,
    zero: &Field,
    reducer: &field::FpCtx<2>,
) -> Result<(), SumcheckError>
where
    H: Sha256InnerBitSource + ?Sized,
{
    let prefix_size = 1usize << K;
    let block_width = coefficients.inner_factor().len();
    let block_start = coefficients.tensor_start() + instance * block_width;
    let block_end = block_start + block_width;
    let first_suffix = block_start.div_ceil(prefix_size);
    let suffix_end = block_end / prefix_size;
    if first_suffix >= suffix_end {
        return Ok(());
    }

    for suffix in first_suffix..suffix_end {
        let base = suffix << K;
        let local_base = base - block_start;
        state.h_values.resize(prefix_size, 0);
        let word = h_source.bits_at(base, prefix_size)?;
        for prefix in 0..prefix_size {
            state.h_values[prefix] = ((word >> prefix) & 1) as i64;
        }
        extend_lsb::<i64, K, _>(
            &mut state.h_values,
            &mut state.h_scratch,
            &0,
            |high, low| *high - *low,
        );
        if let Some(extended) = extended {
            let block = &extended[local_base / prefix_size];
            for beta in 0..pow3(K) {
                linear_multiply_accumulate_signed_fast(
                    reducer,
                    &mut state.local_sums[beta],
                    &block.values[beta],
                    &block.negated[beta],
                    state.h_values[beta],
                );
            }
            continue;
        }
        state.d_values.resize(prefix_size, zero.clone());
        for prefix in 0..prefix_size {
            state.d_values[prefix] = coefficients.inner_factor()[local_base + prefix].clone();
        }
        extend_lsb::<Field, K, _>(
            &mut state.d_values,
            &mut state.d_scratch,
            zero,
            |high, low| reducer.sub(&(high.clone()), &(low)),
        );
        for beta in 0..pow3(K) {
            if state.h_values[beta] != 0 {
                linear_multiply_accumulate_signed(
                    reducer,
                    &mut state.local_sums[beta],
                    &state.d_values[beta],
                    state.h_values[beta],
                    zero,
                );
            }
        }
    }

    for (partial, local) in state.partial_sums.iter_mut().zip(&mut state.local_sums) {
        let local = core::mem::replace(local, linear_accumulator_zero());
        let local = linear_reduce(local, &reducer)?;
        product_multiply_accumulate(
            reducer,
            partial,
            &coefficients.outer_factor()[instance],
            &local,
        );
    }
    Ok(())
}

/// The ternary extension of one aligned prefix block of `block_coefficients`
/// and its negation. When every block run starts on a prefix boundary, all
/// factored instances reuse the same `width / 2^K` blocks, so each block is
/// extended once instead of once per instance.
struct ExtendedBlock {
    values: Vec<Field>,
    negated: Vec<Field>,
}

fn extend_block_coefficients<const K: usize>(
    coefficients: &FactoredMultilinearExtension<'_, Field>,
    zero: &Field,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) -> Option<Vec<ExtendedBlock>> {
    let prefix_size = 1usize << K;
    let width = coefficients.inner_factor().len();
    if coefficients.tensor_start() % prefix_size != 0 || width % prefix_size != 0 {
        return None;
    }
    let mut scratch = Vec::new();
    Some(
        coefficients
            .inner_factor()
            .chunks_exact(prefix_size)
            .map(|block| {
                let mut values = block.to_vec();
                extend_lsb::<Field, K, _>(&mut values, &mut scratch, zero, |high, low| {
                    field_config.sub(&(high.clone()), &(low))
                });
                let negated = values
                    .iter()
                    .map(|value| field_config.sub(&(zero.clone()), &(value)))
                    .collect();
                ExtendedBlock { values, negated }
            })
            .collect(),
    )
}

fn factored_suffix_is_interior<const K: usize>(
    coefficients: &FactoredMultilinearExtension<'_, Field>,
    suffix: usize,
) -> bool {
    let prefix_size = 1usize << K;
    let base = suffix << K;
    if base < coefficients.tensor_start() || base + prefix_size > coefficients.live_len() {
        return false;
    }
    let local = (base - coefficients.tensor_start()) % coefficients.inner_factor().len();
    local + prefix_size <= coefficients.inner_factor().len()
}

fn accumulate_factored_boundaries_sequential<const K: usize, H>(
    suffix_count: usize,
    live_len: usize,
    coefficients: &FactoredMultilinearExtension<'_, Field>,
    h_source: &H,
    zero: &Field,
    reducer: &field::FpCtx<2>,
) -> Result<PrefixBuildState, SumcheckError>
where
    H: Sha256InnerBitSource + ?Sized,
{
    let mut state = PrefixBuildState::new::<K>(zero);
    for suffix in 0..suffix_count {
        if factored_suffix_is_interior::<K>(coefficients, suffix) {
            continue;
        }
        accumulate_suffix::<K, _, _>(
            &mut state,
            &reducer,
            live_len,
            suffix,
            coefficients,
            h_source,
            zero,
        )?;
    }
    Ok(state)
}

#[cfg(feature = "parallel")]
fn merge_factored_prefix_states(
    mut left: FactoredPrefixBuildState,
    right: FactoredPrefixBuildState,
) -> FactoredPrefixBuildState {
    for (left, right) in left.partial_sums.iter_mut().zip(right.partial_sums) {
        product_merge(left, right);
    }
    left
}

pub(crate) struct PrefixAccumulators {
    rounds: Vec<Vec<[Field; 2]>>,
}

impl PrefixAccumulators {
    fn new<const K: usize>(zero: &Field) -> Self {
        let rounds = (0..K)
            .map(|round| vec![[zero.clone(), zero.clone()]; pow3(round)])
            .collect();
        Self { rounds }
    }

    fn evaluate_round(
        &self,
        round: usize,
        coefficients: &[Field],
        reducer: &field::FpCtx<2>,
    ) -> Result<[Field; 2], SumcheckError> {
        let buckets = &self.rounds[round];
        debug_assert_eq!(buckets.len(), coefficients.len());
        let mut at_infinity = product_accumulator_zero();
        let mut at_zero = product_accumulator_zero();
        for (coefficient, bucket) in coefficients.iter().zip(buckets) {
            product_multiply_accumulate(reducer, &mut at_infinity, coefficient, &bucket[0]);
            product_multiply_accumulate(reducer, &mut at_zero, coefficient, &bucket[1]);
        }
        Ok([
            product_reduce(at_infinity, &reducer)?,
            product_reduce(at_zero, &reducer)?,
        ])
    }
}

fn build_prefix_accumulators_generic<const K: usize, S, H>(
    num_vars: usize,
    live_len: usize,
    coefficients: &S,
    h_source: &H,
    field_cfg: &FieldConfig,
    zero: &Field,
) -> Result<PrefixAccumulators, SumcheckError>
where
    S: InnerSumcheckMleSource + ?Sized,
    H: Sha256InnerBitSource + ?Sized,
{
    debug_assert!(K > 0);
    debug_assert!(live_len <= 1usize << num_vars);
    let suffix_count = live_len.div_ceil(1usize << K);

    #[cfg(feature = "parallel")]
    let state = if suffix_count >= 1 << 10 && rayon::current_num_threads() > 1 {
        (0..suffix_count)
            .into_par_iter()
            .try_fold(
                || PrefixBuildState::new::<K>(zero),
                |mut state, suffix| -> Result<_, SumcheckError> {
                    accumulate_suffix::<K, _, _>(
                        &mut state,
                        field_cfg,
                        live_len,
                        suffix,
                        coefficients,
                        h_source,
                        zero,
                    )?;
                    Ok(state)
                },
            )
            .try_reduce(
                || PrefixBuildState::new::<K>(zero),
                |left, right| Ok(merge_prefix_states(left, right)),
            )?
    } else {
        accumulate_suffixes_sequential::<K, _, _>(
            suffix_count,
            live_len,
            coefficients,
            h_source,
            field_cfg,
            zero,
        )?
    };

    #[cfg(not(feature = "parallel"))]
    let state = accumulate_suffixes_sequential::<K, _, _>(
        suffix_count,
        live_len,
        coefficients,
        h_source,
        field_cfg,
        zero,
    )?;

    let beta_values = state
        .partial_sums
        .into_iter()
        .map(|accumulator| linear_reduce(accumulator, field_cfg))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(scatter_beta_values::<K>(&beta_values, zero, &field_cfg))
}

fn accumulate_suffixes_sequential<const K: usize, S, H>(
    suffix_count: usize,
    live_len: usize,
    coefficients: &S,
    h_source: &H,
    field_cfg: &FieldConfig,
    zero: &Field,
) -> Result<PrefixBuildState, SumcheckError>
where
    S: InnerSumcheckMleSource + ?Sized,
    H: Sha256InnerBitSource + ?Sized,
{
    let mut state = PrefixBuildState::new::<K>(zero);
    for suffix in 0..suffix_count {
        accumulate_suffix::<K, _, _>(
            &mut state,
            field_cfg,
            live_len,
            suffix,
            coefficients,
            h_source,
            zero,
        )?;
    }
    Ok(state)
}

fn accumulate_suffix<const K: usize, S, H>(
    state: &mut PrefixBuildState,
    field_cfg: &FieldConfig,
    live_len: usize,
    suffix: usize,
    coefficients: &S,
    h_source: &H,
    zero: &Field,
) -> Result<(), SumcheckError>
where
    S: InnerSumcheckMleSource + ?Sized,
    H: Sha256InnerBitSource + ?Sized,
{
    let prefix_size = 1usize << K;
    let base = suffix << K;
    state.v_values.resize(prefix_size, zero.clone());
    state.h_values.resize(prefix_size, 0);
    state.v_values.fill(zero.clone());
    state.h_values.fill(0);
    let active_prefixes = prefix_size.min(live_len - base);
    let word = h_source.bits_at(base, active_prefixes)?;
    for prefix in 0..active_prefixes {
        let index = base | prefix;
        let value = coefficients.evaluation_at(index)?;
        if !coefficients.evaluations_prevalidated() {
            validate_field_value(&value, field_cfg)?;
        }
        state.v_values[prefix] = value;
        state.h_values[prefix] = ((word >> prefix) & 1) as i64;
    }

    extend_lsb::<Field, K, _>(
        &mut state.v_values,
        &mut state.v_scratch,
        zero,
        |high, low| field_cfg.sub(&(high.clone()), &(low)),
    );
    extend_lsb::<i64, K, _>(
        &mut state.h_values,
        &mut state.h_scratch,
        &0,
        |high, low| *high - *low,
    );

    for beta in 0..pow3(K) {
        if state.h_values[beta] != 0 {
            linear_multiply_accumulate_signed(
                field_cfg,
                &mut state.partial_sums[beta],
                &state.v_values[beta],
                state.h_values[beta],
                zero,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn merge_prefix_states(mut left: PrefixBuildState, right: PrefixBuildState) -> PrefixBuildState {
    for (left, right) in left.partial_sums.iter_mut().zip(right.partial_sums) {
        linear_merge(left, right);
    }
    left
}

fn scatter_beta_values<const K: usize>(
    beta_values: &[Field],
    zero: &Field,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) -> PrefixAccumulators {
    let mut accumulators = PrefixAccumulators::new::<K>(zero);
    for (beta, value) in beta_values.iter().enumerate() {
        for round in 0..K {
            let coordinate = (beta / pow3(round)) % 3;
            if coordinate == 2 || !ternary_suffix_is_binary(beta, round + 1, K) {
                continue;
            }
            let prefix = beta % pow3(round);
            let endpoint = usize::from(coordinate == 1);
            accumulators.rounds[round][prefix][endpoint] =
                field_config.add(&(accumulators.rounds[round][prefix][endpoint]), &(value));
        }
    }
    accumulators
}

fn ternary_suffix_is_binary(beta: usize, start: usize, variables: usize) -> bool {
    let mut remaining = beta / pow3(start);
    for _ in start..variables {
        if remaining % 3 == 0 {
            return false;
        }
        remaining /= 3;
    }
    true
}

fn extend_lsb<T, const K: usize, S>(
    values: &mut Vec<T>,
    scratch: &mut Vec<T>,
    zero: &T,
    subtract: S,
) where
    T: Clone,
    S: Fn(&T, &T) -> T,
{
    debug_assert_eq!(values.len(), 1usize << K);
    let mut current_len = 1usize << K;

    for coordinate in 0..K {
        let low_stride = pow3(coordinate);
        let high_groups = 1usize << (K - coordinate - 1);
        let next_len = current_len / 2 * 3;
        if coordinate % 2 == 0 {
            scratch.resize(next_len, zero.clone());
            extend_lsb_axis(
                &values[..current_len],
                &mut scratch[..next_len],
                low_stride,
                high_groups,
                &subtract,
            );
        } else {
            values.resize(next_len, zero.clone());
            extend_lsb_axis(
                &scratch[..current_len],
                &mut values[..next_len],
                low_stride,
                high_groups,
                &subtract,
            );
        }
        current_len = next_len;
    }

    if K % 2 == 1 {
        values.clear();
        values.extend_from_slice(&scratch[..current_len]);
    } else {
        values.truncate(current_len);
    }
}

fn extend_lsb_axis<T, S>(
    input: &[T],
    output: &mut [T],
    low_stride: usize,
    high_groups: usize,
    subtract: &S,
) where
    T: Clone,
    S: Fn(&T, &T) -> T,
{
    for high_group in 0..high_groups {
        let input_base = high_group * 2 * low_stride;
        let output_base = high_group * 3 * low_stride;
        for low_index in 0..low_stride {
            let low = &input[input_base + low_index];
            let high = &input[input_base + low_stride + low_index];
            output[output_base + low_index] = subtract(high, low);
            output[output_base + low_stride + low_index] = low.clone();
            output[output_base + 2 * low_stride + low_index] = high.clone();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_prefix_v_table<const K: usize, S>(
    num_vars: usize,
    live_len: usize,
    coefficients: &S,
    challenges: &[Field],
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<CompactPrefixVTable, SumcheckError>
where
    S: InnerSumcheckMleSource + ?Sized,
{
    coefficients.fold_prefix_table::<K>(num_vars, live_len, challenges, field_cfg, zero, one)
}

#[allow(clippy::too_many_arguments)]
fn fold_prefix_v_table_generic<const K: usize, S>(
    num_vars: usize,
    live_len: usize,
    coefficients: &S,
    challenges: &[Field],
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<CompactPrefixVTable, SumcheckError>
where
    S: InnerSumcheckMleSource + ?Sized,
{
    debug_assert_eq!(challenges.len(), K);
    debug_assert!(live_len <= 1usize << num_vars);
    let prefix_size = 1usize << K;
    let suffix_count = live_len.div_ceil(prefix_size);
    let weights = equality_weights_lsb(challenges, zero, one, &field_cfg);
    let fold_suffix = |suffix: usize| -> Result<RawMontgomery, SumcheckError> {
        let base = suffix << K;

        if K == 0 {
            let value = coefficients.evaluation_at(base)?;
            if !coefficients.evaluations_prevalidated() {
                validate_field_value(&value, field_cfg)?;
            }
            return Ok(raw_montgomery(&value));
        }

        let mut v_accumulator = product_accumulator_zero();
        let active_prefixes = prefix_size.min(live_len - base);
        for (prefix, weight) in weights.iter().take(active_prefixes).enumerate() {
            let index = base | prefix;
            let value = coefficients.evaluation_at(index)?;
            if !coefficients.evaluations_prevalidated() {
                validate_field_value(&value, field_cfg)?;
            }
            product_multiply_accumulate(field_cfg, &mut v_accumulator, weight, &value);
        }
        let v = product_reduce(v_accumulator, field_cfg)?;
        Ok(raw_montgomery(&v))
    };

    let zero_raw = raw_montgomery(zero);
    // Round up by one raw residue so the first tail fold can overwrite every
    // adjacent V pair with one interleaved [V, H] cell. This is at most 16
    // bytes beyond the exact one-residue-per-suffix representation.
    let storage_len = suffix_count
        .checked_add(1)
        .ok_or(SumcheckError::InvalidProductDimensions)?
        & !1;
    let mut table = CompactPrefixVTable {
        values: vec![zero_raw; storage_len],
        suffix_count,
    };

    #[cfg(feature = "parallel")]
    if suffix_count >= 1 << 10 && rayon::current_num_threads() > 1 {
        table.values[..suffix_count]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(suffix, v_out)| -> Result<(), SumcheckError> {
                *v_out = fold_suffix(suffix)?;
                Ok(())
            })?;
    } else {
        for (suffix, v_out) in table.values[..suffix_count].iter_mut().enumerate() {
            *v_out = fold_suffix(suffix)?;
        }
    }

    #[cfg(not(feature = "parallel"))]
    for (suffix, v_out) in table.values[..suffix_count].iter_mut().enumerate() {
        *v_out = fold_suffix(suffix)?;
    }

    Ok(table)
}

/// Block-aware prefix fold for the physical SHA layout. A `2^K` prefix block
/// may contain the shared cell and may cross an instance boundary. Split it
/// into maximal same-instance runs and compute
///
/// `sum_j weight_j (u_i d_j) = u_i sum_j weight_j d_j`
///
/// once per run. This is exact for every block width; in particular it does
/// not rely on SHA's 20,456-cell block being divisible by `2^K`.
#[allow(clippy::too_many_arguments)]
fn fold_factored_prefix_v_table<const K: usize>(
    num_vars: usize,
    live_len: usize,
    coefficients: &FactoredMultilinearExtension<'_, Field>,
    challenges: &[Field],
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<CompactPrefixVTable, SumcheckError> {
    debug_assert_eq!(challenges.len(), K);
    debug_assert_eq!(live_len, coefficients.live_len());
    debug_assert!(live_len <= 1usize << num_vars);
    let prefix_size = 1usize << K;
    let suffix_count = live_len.div_ceil(prefix_size);
    let weights = equality_weights_lsb(challenges, zero, one, &field_cfg);
    let block_width = coefficients.inner_factor().len();

    let fold_suffix = |suffix: usize| -> Result<RawMontgomery, SumcheckError> {
        let base = suffix << K;
        if K == 0 {
            return Ok(raw_montgomery(
                &coefficients
                    .evaluation_at(base)
                    .map_err(|_| SumcheckError::InvalidProductDimensions)?,
            ));
        }
        let end = live_len.min(base + prefix_size);
        let mut cursor = base;
        let mut total = product_accumulator_zero();

        if cursor < coefficients.tensor_start() {
            product_multiply_accumulate(
                field_cfg,
                &mut total,
                &weights[0],
                coefficients.leading_value(),
            );
            cursor = 1;
        }

        while cursor < end {
            let offset = cursor - coefficients.tensor_start();
            let instance = offset / block_width;
            let local = offset % block_width;
            let run_len = (block_width - local).min(end - cursor);
            let mut local_fold = product_accumulator_zero();
            for run_offset in 0..run_len {
                let prefix = cursor + run_offset - base;
                product_multiply_accumulate(
                    field_cfg,
                    &mut local_fold,
                    &weights[prefix],
                    &coefficients.inner_factor()[local + run_offset],
                );
            }
            let local_fold = product_reduce(local_fold, field_cfg)?;
            product_multiply_accumulate(
                field_cfg,
                &mut total,
                &coefficients.outer_factor()[instance],
                &local_fold,
            );
            cursor += run_len;
        }

        Ok(raw_montgomery(&product_reduce(total, field_cfg)?))
    };

    let zero_raw = raw_montgomery(zero);
    let storage_len = suffix_count
        .checked_add(1)
        .ok_or(SumcheckError::InvalidProductDimensions)?
        & !1;
    let mut table = CompactPrefixVTable {
        values: vec![zero_raw; storage_len],
        suffix_count,
    };

    #[cfg(feature = "parallel")]
    if suffix_count >= 1 << 10 && rayon::current_num_threads() > 1 {
        table.values[..suffix_count]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(suffix, output)| -> Result<(), SumcheckError> {
                *output = fold_suffix(suffix)?;
                Ok(())
            })?;
    } else {
        for (suffix, output) in table.values[..suffix_count].iter_mut().enumerate() {
            *output = fold_suffix(suffix)?;
        }
    }

    #[cfg(not(feature = "parallel"))]
    for (suffix, output) in table.values[..suffix_count].iter_mut().enumerate() {
        *output = fold_suffix(suffix)?;
    }

    // The source adapter validates every factor against `field_cfg`; retain an
    // explicit shape/config assertion at this boundary for future callers.

    Ok(table)
}

pub(crate) struct CompactPrefixVTable {
    /// Before the first tail round, one V residue per suffix plus at most one
    /// zero placeholder. Afterwards, interleaved [V, H] cells in the same
    /// allocation.
    values: Vec<RawMontgomery>,
    suffix_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn folded_packed_h<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    suffix: usize,
    live_len: usize,
    h_source: &H,
    prefix_weights: &[Field],
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<Field, SumcheckError> {
    let base = suffix << K;
    debug_assert!(base < live_len);
    if K == 0 {
        return Ok(select_field_by_bit(zero, one, source_bit(h_source, base)?));
    }

    let active_prefixes = (1usize << K).min(live_len - base);
    let word = h_source.bits_at(base, active_prefixes)?;
    let mut accumulator = linear_accumulator_zero();
    // Branch-free: the bits are random, so multiplying by 0/1 beats skipping.
    for (prefix, weight) in prefix_weights.iter().take(active_prefixes).enumerate() {
        linear_multiply_accumulate(field_cfg, &mut accumulator, weight, &((word >> prefix) & 1));
    }
    // The reduced value is canonical by construction; comparing its
    // configuration here cost a full parameter comparison per suffix.
    let value = linear_reduce(accumulator, field_cfg)?;

    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_first_tail_pair<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    accumulators: &mut [ProductAccumulator; 2],
    table: &CompactPrefixVTable,
    pair: usize,
    live_len: usize,
    h_source: &H,
    prefix_weights: &[Field],
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<(), SumcheckError> {
    let low_suffix = 2 * pair;
    let high_suffix = low_suffix + 1;
    let v_zero = field_from_raw(&table.values[low_suffix], field_cfg);
    let h_zero = folded_packed_h::<K, _>(
        low_suffix,
        live_len,
        h_source,
        prefix_weights,
        field_cfg,
        zero,
        one,
    )?;
    let (v_one, h_one) = if high_suffix < table.suffix_count {
        (
            field_from_raw(&table.values[high_suffix], field_cfg),
            folded_packed_h::<K, _>(
                high_suffix,
                live_len,
                h_source,
                prefix_weights,
                field_cfg,
                zero,
                one,
            )?,
        )
    } else {
        (zero.clone(), zero.clone())
    };

    product_multiply_accumulate(field_cfg, &mut accumulators[0], &v_zero, &h_zero);
    product_multiply_accumulate(
        field_cfg,
        &mut accumulators[1],
        &(field_cfg.sub(&(v_one), &(&v_zero))),
        &(field_cfg.sub(&(h_one), &(&h_zero))),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sum_first_tail_round<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    table: &CompactPrefixVTable,
    live_len: usize,
    h_source: &H,
    prefix_weights: &[Field],
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<[Field; 2], SumcheckError> {
    let pair_count = table.suffix_count.div_ceil(2);

    #[cfg(feature = "parallel")]
    if pair_count >= 1 << 10 && rayon::current_num_threads() > 1 {
        let accumulators = (0..pair_count)
            .into_par_iter()
            .try_fold(
                || std::array::from_fn(|_| product_accumulator_zero()),
                |mut accumulators, pair| -> Result<_, SumcheckError> {
                    accumulate_first_tail_pair::<K, _>(
                        &mut accumulators,
                        table,
                        pair,
                        live_len,
                        h_source,
                        prefix_weights,
                        field_cfg,
                        zero,
                        one,
                    )?;
                    Ok(accumulators)
                },
            )
            .try_reduce(
                || std::array::from_fn(|_| product_accumulator_zero()),
                |left, right| Ok(merge_accumulators(left, right)),
            )?;
        return reduce_product_accumulators(accumulators, field_cfg);
    }

    let mut accumulators = std::array::from_fn(|_| product_accumulator_zero());
    for pair in 0..pair_count {
        accumulate_first_tail_pair::<K, _>(
            &mut accumulators,
            table,
            pair,
            live_len,
            h_source,
            prefix_weights,
            field_cfg,
            zero,
            one,
        )?;
    }
    reduce_product_accumulators(accumulators, field_cfg)
}

#[allow(clippy::too_many_arguments)]
fn fold_first_tail_pair_in_place<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    pair: usize,
    values: &mut [RawMontgomery],
    suffix_count: usize,
    live_len: usize,
    h_source: &H,
    prefix_weights: &[Field],
    challenge: &Field,
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<[Field; 2], SumcheckError> {
    debug_assert_eq!(values.len(), 2);
    let low_suffix = 2 * pair;
    let high_suffix = low_suffix + 1;
    let h_zero = folded_packed_h::<K, _>(
        low_suffix,
        live_len,
        h_source,
        prefix_weights,
        field_cfg,
        zero,
        one,
    )?;
    let h_one = if high_suffix < suffix_count {
        folded_packed_h::<K, _>(
            high_suffix,
            live_len,
            h_source,
            prefix_weights,
            field_cfg,
            zero,
            one,
        )?
    } else {
        zero.clone()
    };
    let folded_v = interpolate_raw_pair(
        &values[0],
        (high_suffix < suffix_count).then_some(&values[1]),
        challenge,
        field_cfg,
        zero,
    );
    let folded_h = field_cfg.add(
        &(h_zero.clone()),
        &(field_cfg.mul(&(challenge), &(&(field_cfg.sub(&(h_one), &(&h_zero)))))),
    );
    values[0] = raw_montgomery(&folded_v);
    values[1] = raw_montgomery(&folded_h);
    Ok([folded_v, folded_h])
}

#[allow(clippy::too_many_arguments)]
fn fold_first_tail_round_in_place<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    table: &mut CompactPrefixVTable,
    live_len: usize,
    h_source: &H,
    prefix_weights: &[Field],
    challenge: &Field,
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
) -> Result<(), SumcheckError> {
    let suffix_count = table.suffix_count;
    let fold_pair = |pair: usize, values: &mut [RawMontgomery]| {
        fold_first_tail_pair_in_place::<K, _>(
            pair,
            values,
            suffix_count,
            live_len,
            h_source,
            prefix_weights,
            challenge,
            field_cfg,
            zero,
            one,
        )
        .map(|_| ())
    };

    #[cfg(feature = "parallel")]
    if table.values.len() / 2 >= 1 << 10 && rayon::current_num_threads() > 1 {
        table
            .values
            .par_chunks_mut(2)
            .enumerate()
            .try_for_each(|(pair, values)| fold_pair(pair, values))?;
        table.suffix_count = table.values.len() / 2;
        return Ok(());
    }

    for (pair, values) in table.values.chunks_mut(2).enumerate() {
        fold_pair(pair, values)?;
    }
    table.suffix_count = table.values.len() / 2;
    Ok(())
}

/// Folds the first tail coordinate and prepares the following round's
/// `[c0, c2]` buckets while the new interleaved `[V, H]` cells are hot.
#[allow(clippy::too_many_arguments)]
fn fold_first_tail_round_and_prepare_next_in_place<
    const K: usize,
    H: Sha256InnerBitSource + ?Sized,
>(
    table: &mut CompactPrefixVTable,
    live_len: usize,
    h_source: &H,
    prefix_weights: &[Field],
    challenge: &Field,
    field_cfg: &FieldConfig,
    zero: &Field,
    one: &Field,
    reducer: &field::FpCtx<2>,
) -> Result<[Field; 2], SumcheckError> {
    let suffix_count = table.suffix_count;
    let fold_and_accumulate = |mut accumulators: [ProductAccumulator; 2],
                               superchunk: usize,
                               values: &mut [RawMontgomery]|
     -> Result<[ProductAccumulator; 2], SumcheckError> {
        debug_assert!(values.len() == 2 || values.len() == 4);
        let first_pair = 2 * superchunk;
        let folded_zero = fold_first_tail_pair_in_place::<K, _>(
            first_pair,
            &mut values[..2],
            suffix_count,
            live_len,
            h_source,
            prefix_weights,
            challenge,
            field_cfg,
            zero,
            one,
        )?;
        let folded_one = if values.len() == 4 {
            fold_first_tail_pair_in_place::<K, _>(
                first_pair + 1,
                &mut values[2..],
                suffix_count,
                live_len,
                h_source,
                prefix_weights,
                challenge,
                field_cfg,
                zero,
                one,
            )?
        } else {
            [zero.clone(), zero.clone()]
        };

        product_multiply_accumulate(
            reducer,
            &mut accumulators[0],
            &folded_zero[0],
            &folded_zero[1],
        );
        product_multiply_accumulate(
            reducer,
            &mut accumulators[1],
            &(field_cfg.sub(&(folded_one[0].clone()), &(&folded_zero[0]))),
            &(field_cfg.sub(&(folded_one[1].clone()), &(&folded_zero[1]))),
        );
        Ok(accumulators)
    };

    #[cfg(feature = "parallel")]
    if table.values.len() / 2 >= 1 << 10 && rayon::current_num_threads() > 1 {
        let accumulators = table
            .values
            .par_chunks_mut(4)
            .enumerate()
            .try_fold(
                || std::array::from_fn(|_| product_accumulator_zero()),
                |accumulators, (superchunk, values)| {
                    fold_and_accumulate(accumulators, superchunk, values)
                },
            )
            .try_reduce(
                || std::array::from_fn(|_| product_accumulator_zero()),
                |left, right| Ok(merge_accumulators(left, right)),
            )?;
        table.suffix_count = table.values.len() / 2;
        return reduce_product_accumulators(accumulators, field_cfg);
    }

    let mut accumulators = std::array::from_fn(|_| product_accumulator_zero());
    for (superchunk, values) in table.values.chunks_mut(4).enumerate() {
        accumulators = fold_and_accumulate(accumulators, superchunk, values)?;
    }
    table.suffix_count = table.values.len() / 2;
    reduce_product_accumulators(accumulators, field_cfg)
}

#[cfg(test)]
fn sum_interleaved_round_coefficients(
    values: &[RawMontgomery],
    stride: usize,
    field_cfg: &FieldConfig,
) -> Result<[Field; 2], SumcheckError> {
    debug_assert!(!values.is_empty());
    debug_assert_eq!(values.len() % 2, 0);
    let chunk_len = 4 * stride;
    let zero = Field::zero_with_cfg(field_cfg);

    #[cfg(feature = "parallel")]
    if values.len().div_ceil(chunk_len) >= 1 << 10 && rayon::current_num_threads() > 1 {
        let accumulators = values
            .par_chunks(chunk_len)
            .fold(
                || std::array::from_fn(|_| product_accumulator_zero()),
                |mut accumulators, values| {
                    accumulate_interleaved_chunk(
                        &mut accumulators,
                        values,
                        stride,
                        field_cfg,
                        &zero,
                    );
                    accumulators
                },
            )
            .reduce(
                || std::array::from_fn(|_| product_accumulator_zero()),
                |left, right| merge_accumulators(left, right),
            );
        return reduce_product_accumulators(accumulators, field_cfg);
    }

    let accumulators = values.chunks(chunk_len).fold(
        std::array::from_fn(|_| product_accumulator_zero()),
        |mut accumulators, values| {
            accumulate_interleaved_chunk(&mut accumulators, values, stride, field_cfg, &zero);
            accumulators
        },
    );
    reduce_product_accumulators(accumulators, field_cfg)
}

#[cfg(test)]
fn accumulate_interleaved_chunk(
    accumulators: &mut [ProductAccumulator; 2],
    values: &[RawMontgomery],
    stride: usize,
    field_cfg: &FieldConfig,
    zero: &Field,
) {
    debug_assert!(values.len() >= 2);
    debug_assert!(values.len() <= 4 * stride);
    let high_offset = 2 * stride;
    let v_zero = field_from_raw(&values[0], field_cfg);
    let h_zero = field_from_raw(&values[1], field_cfg);
    let v_one = values
        .get(high_offset)
        .map_or_else(|| zero.clone(), |raw| field_from_raw(raw, field_cfg));
    let h_one = values
        .get(high_offset + 1)
        .map_or_else(|| zero.clone(), |raw| field_from_raw(raw, field_cfg));

    product_multiply_accumulate(field_cfg, &mut accumulators[0], &v_zero, &h_zero);
    product_multiply_accumulate(
        field_cfg,
        &mut accumulators[1],
        &(field_cfg.sub(&(v_one), &(&v_zero))),
        &(field_cfg.sub(&(h_one), &(&h_zero))),
    );
}

fn fold_interleaved_chunk_in_place(
    values: &mut [RawMontgomery],
    stride: usize,
    challenge: &Field,
    field_cfg: &FieldConfig,
    zero: &Field,
) -> [Field; 2] {
    debug_assert!(values.len() >= 2);
    debug_assert!(values.len() <= 4 * stride);
    let high_offset = 2 * stride;
    let folded_v = interpolate_raw_pair(
        &values[0],
        values.get(high_offset),
        challenge,
        field_cfg,
        zero,
    );
    let folded_h = interpolate_raw_pair(
        &values[1],
        values.get(high_offset + 1),
        challenge,
        field_cfg,
        zero,
    );
    values[0] = raw_montgomery(&folded_v);
    values[1] = raw_montgomery(&folded_h);
    [folded_v, folded_h]
}

fn fold_interleaved_in_place(
    values: &mut [RawMontgomery],
    stride: usize,
    challenge: &Field,
    field_cfg: &FieldConfig,
    zero: &Field,
) {
    debug_assert!(!values.is_empty());
    debug_assert_eq!(values.len() % 2, 0);
    let chunk_len = 4 * stride;

    #[cfg(feature = "parallel")]
    if values.len().div_ceil(chunk_len) >= 1 << 10 && rayon::current_num_threads() > 1 {
        values.par_chunks_mut(chunk_len).for_each(|values| {
            fold_interleaved_chunk_in_place(values, stride, challenge, field_cfg, zero);
        });
        return;
    }

    for chunk in values.chunks_mut(chunk_len) {
        fold_interleaved_chunk_in_place(chunk, stride, challenge, field_cfg, zero);
    }
}

/// Folds one interleaved tail coordinate and prepares the following round's
/// `[c0, c2]` buckets from adjacent pairs of newly folded cells.
fn fold_interleaved_and_prepare_next_round_in_place(
    values: &mut [RawMontgomery],
    stride: usize,
    challenge: &Field,
    field_cfg: &FieldConfig,
    zero: &Field,
) -> Result<[Field; 2], SumcheckError> {
    debug_assert!(!values.is_empty());
    debug_assert_eq!(values.len() % 2, 0);
    let fold_chunk_len = 4 * stride;
    let superchunk_len = 2 * fold_chunk_len;
    let fold_and_accumulate = |mut accumulators: [ProductAccumulator; 2],
                               values: &mut [RawMontgomery]| {
        debug_assert!(values.len() >= 2);
        debug_assert!(values.len() <= superchunk_len);
        let first_len = values.len().min(fold_chunk_len);
        let (first, second) = values.split_at_mut(first_len);
        let folded_zero =
            fold_interleaved_chunk_in_place(first, stride, challenge, field_cfg, zero);
        let folded_one = if second.is_empty() {
            [zero.clone(), zero.clone()]
        } else {
            fold_interleaved_chunk_in_place(second, stride, challenge, field_cfg, zero)
        };

        product_multiply_accumulate(
            field_cfg,
            &mut accumulators[0],
            &folded_zero[0],
            &folded_zero[1],
        );
        product_multiply_accumulate(
            field_cfg,
            &mut accumulators[1],
            &(field_cfg.sub(&(folded_one[0].clone()), &(&folded_zero[0]))),
            &(field_cfg.sub(&(folded_one[1].clone()), &(&folded_zero[1]))),
        );
        accumulators
    };

    #[cfg(feature = "parallel")]
    if values.len().div_ceil(fold_chunk_len) >= 1 << 10 && rayon::current_num_threads() > 1 {
        let accumulators = values
            .par_chunks_mut(superchunk_len)
            .fold(
                || std::array::from_fn(|_| product_accumulator_zero()),
                fold_and_accumulate,
            )
            .reduce(
                || std::array::from_fn(|_| product_accumulator_zero()),
                |left, right| merge_accumulators(left, right),
            );
        return reduce_product_accumulators(accumulators, field_cfg);
    }

    let accumulators = values.chunks_mut(superchunk_len).fold(
        std::array::from_fn(|_| product_accumulator_zero()),
        fold_and_accumulate,
    );
    reduce_product_accumulators(accumulators, field_cfg)
}

#[inline]
fn interpolate_raw_pair(
    low: &RawMontgomery,
    high: Option<&RawMontgomery>,
    challenge: &Field,
    field_cfg: &FieldConfig,
    zero: &Field,
) -> Field {
    let low = field_from_raw(low, field_cfg);
    let high = high.map_or_else(|| zero.clone(), |raw| field_from_raw(raw, field_cfg));
    field_cfg.add(
        &(low.clone()),
        &(field_cfg.mul(&(challenge), &(&(field_cfg.sub(&(high), &(&low)))))),
    )
}

#[inline]
fn raw_montgomery(value: &Field) -> RawMontgomery {
    let words = value.as_montgomery_integer().as_words();
    [words[0], words[1]]
}

#[inline]
fn field_from_raw(raw: &RawMontgomery, field_cfg: &FieldConfig) -> Field {
    field_cfg.from_montgomery_integer(FieldUint::from_words(*raw))
}

#[inline]
fn reduce_product_accumulators(
    accumulators: [ProductAccumulator; 2],
    config: &FieldConfig,
) -> Result<[Field; 2], SumcheckError> {
    let [at_zero, leading] = accumulators;
    Ok([
        product_reduce(at_zero, config)?,
        product_reduce(leading, config)?,
    ])
}

fn equality_weights_lsb(
    challenges: &[Field],
    zero: &Field,
    one: &Field,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) -> Vec<Field> {
    let mut weights = vec![one.clone()];
    for challenge in challenges {
        let old_len = weights.len();
        weights.resize(2 * old_len, zero.clone());
        let one_minus_challenge = field_config.sub(&(one.clone()), &(challenge));
        for index in 0..old_len {
            let parent = weights[index].clone();
            weights[index + old_len] = field_config.mul(&(parent.clone()), &(challenge));
            weights[index] = field_config.mul(&(parent), &(&one_minus_challenge));
        }
    }
    weights
}

fn extend_lagrange_coefficients(
    coefficients: &mut Vec<Field>,
    challenge: &Field,
    one: &Field,
    zero: &Field,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) {
    // For U_2 = {infinity, 0, 1}, infinity denotes the quadratic leading
    // coefficient. Thus p(r) = L_inf(r)p_inf + L_0(r)p(0) + L_1(r)p(1).
    let at_one = challenge.clone();
    let at_zero = field_config.sub(&(one.clone()), &(challenge));
    let at_infinity = field_config.mul(
        &(challenge.clone()),
        &(field_config.sub(&(challenge.clone()), &(one))),
    );
    let old_len = coefficients.len();
    let mut next = vec![zero.clone(); 3 * old_len];
    for (prefix, coefficient) in coefficients.iter().enumerate() {
        next[prefix] = field_config.mul(&(coefficient.clone()), &(&at_infinity));
        next[old_len + prefix] = field_config.mul(&(coefficient.clone()), &(&at_zero));
        next[2 * old_len + prefix] = field_config.mul(&(coefficient.clone()), &(&at_one));
    }
    *coefficients = next;
}

#[inline]
fn source_bit<H: Sha256InnerBitSource + ?Sized>(
    source: &H,
    index: usize,
) -> Result<u64, SumcheckError> {
    let bit = source.bit_at(index)?;
    if bit > 1 {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    Ok(bit)
}

#[cfg(test)]
#[inline]
fn packed_bit(words: &[u64], index: usize) -> u64 {
    (words[index / u64::BITS as usize] >> (index % u64::BITS as usize)) & 1
}

#[inline]
fn select_field_by_bit(zero: &Field, one: &Field, bit: u64) -> Field {
    CtSelect::ct_select(zero, &one, CtMask::from_lsb(bit))
}

/// [`linear_multiply_accumulate_signed`] with the negation precomputed and a
/// zero coefficient skipped: the accumulated value is identical, and Bit
/// extensions are zero at roughly half of their ternary points.
#[inline]
fn linear_multiply_accumulate_signed_fast(
    reducer: &field::FpCtx<2>,
    accumulator: &mut LinearAccumulator,
    value: &Field,
    negated_value: &Field,
    signed_coefficient: i64,
) {
    if signed_coefficient == 0 {
        return;
    }
    let magnitude = signed_coefficient.unsigned_abs();
    let selected = if signed_coefficient < 0 {
        negated_value
    } else {
        value
    };
    linear_multiply_accumulate(reducer, accumulator, selected, &magnitude);
}

fn linear_multiply_accumulate_signed(
    reducer: &field::FpCtx<2>,
    accumulator: &mut LinearAccumulator,
    value: &Field,
    signed_coefficient: i64,
    zero: &Field,
) {
    // A Bit K<=4 extension has magnitude at most 2^(K-1), but perform the
    // sign extraction in i128 so the signed-magnitude conversion is total.
    let signed_coefficient = i128::from(signed_coefficient);
    let sign_mask = (signed_coefficient >> 127) as u128;
    let magnitude = ((signed_coefficient as u128) ^ sign_mask).wrapping_sub(sign_mask) as u64;
    let negative_value = reducer.sub(&(zero.clone()), &(value));
    let selected_value =
        CtSelect::ct_select(value, &negative_value, CtMask::from_lsb(sign_mask as u64));
    linear_multiply_accumulate(reducer, accumulator, &selected_value, &magnitude);
}

#[inline]
fn linear_accumulator_zero() -> LinearAccumulator {
    LinearAccumulator::zero()
}

#[inline]
fn linear_multiply_accumulate(
    reducer: &field::FpCtx<2>,
    accumulator: &mut LinearAccumulator,
    lhs: &Field,
    rhs: &u64,
) {
    reducer.mul_acc(accumulator, lhs, rhs);
}

#[inline]
fn linear_merge(accumulator: &mut LinearAccumulator, other: LinearAccumulator) {
    accumulator.merge_assign(&other);
}

#[inline]
fn linear_reduce(
    accumulator: LinearAccumulator,
    config: &FieldConfig,
) -> Result<Field, SumcheckError> {
    Ok(Reduce::reduce(config, accumulator))
}

#[inline]
fn product_accumulator_zero() -> ProductAccumulator {
    <field::FpCtx<2> as BatchMulAcc<Field>>::Accumulator::zero()
}

#[inline]
fn product_multiply_accumulate(
    reducer: &field::FpCtx<2>,
    accumulator: &mut ProductAccumulator,
    lhs: &Field,
    rhs: &Field,
) {
    <field::FpCtx<2> as BatchMulAcc<Field>>::mul_acc(reducer, accumulator, lhs, rhs);
}

#[inline]
fn product_merge(accumulator: &mut ProductAccumulator, other: ProductAccumulator) {
    accumulator.merge_assign(&other);
}

#[inline]
fn product_reduce(
    accumulator: ProductAccumulator,
    config: &FieldConfig,
) -> Result<Field, SumcheckError> {
    Ok(Reduce::reduce(config, accumulator))
}

const fn pow3(exponent: usize) -> usize {
    let mut result = 1usize;
    let mut index = 0usize;
    while index < exponent {
        result *= 3;
        index += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::{
        piop::spartan::bitz::spartan_bitz_field_config,
        piop::spartan::sumcheck::{
            prove_inner_sumcheck_with_reducer, prove_inner_sumcheck_with_reducer_grinded,
        },
        poly::mle::DenseMultilinearExtension,
        transcript::{Blake3Transcript, traits::Transcript},
    };

    use super::*;

    fn field(value: u64, field_cfg: &FieldConfig) -> Field {
        Field::from_with_cfg(value, field_cfg)
    }

    fn fixture(
        num_vars: usize,
        field_cfg: &FieldConfig,
    ) -> (DenseMultilinearExtension<Field>, Vec<u64>, Field) {
        let table_len = 1usize << num_vars;
        let zero = Field::zero_with_cfg(field_cfg);
        let v = (0..table_len)
            .map(|index| field((17 * index as u64 + 3) % 251, field_cfg))
            .collect::<Vec<_>>();
        let mut h_words = vec![0u64; table_len.div_ceil(u64::BITS as usize)];
        let mut claim = zero.clone();
        for (index, value) in v.iter().enumerate() {
            let bit = (((index as u64).wrapping_mul(0x9e37_79b9) ^ (index as u64 >> 1))
                .count_ones()
                & 1) as u64;
            h_words[index / u64::BITS as usize] |= bit << (index % u64::BITS as usize);
            if bit != 0 {
                claim = field_cfg.add(&(claim), &(value));
            }
        }
        (
            DenseMultilinearExtension::from_evaluations_vec(num_vars, v, zero),
            h_words,
            claim,
        )
    }

    #[test]
    fn factored_block_source_is_transcript_identical_across_shared_and_block_boundaries() {
        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();

        // These widths exercise aligned blocks, blocks smaller than a K=4
        // prefix, and blocks that cross both K=3 and K=4 prefix boundaries.
        for block_width in [7usize, 8, 15, 16, 17] {
            let instance_weights = (0..3)
                .map(|instance| field(31 * instance as u64 + 5, &field_cfg))
                .collect::<Vec<_>>();
            let block_coefficients = (0..block_width)
                .map(|local| field(19 * local as u64 + 11, &field_cfg))
                .collect::<Vec<_>>();
            let source = FactoredMultilinearExtension::with_leading_value(
                (1 + instance_weights.len() * block_coefficients.len())
                    .next_power_of_two()
                    .ilog2() as usize,
                field(137, &field_cfg),
                &instance_weights,
                &block_coefficients,
                &field_cfg,
            )
            .unwrap();
            let live_len = source.live_len();
            let num_vars = live_len.next_power_of_two().ilog2() as usize;
            let mut h_words = vec![0u64; live_len.div_ceil(u64::BITS as usize)];
            let mut initial_claim = zero.clone();
            for index in 0..live_len {
                let bit = ((index * 29 + block_width).count_ones() & 1) as u64;
                h_words[index / 64] |= bit << (index % 64);
                if bit != 0 {
                    initial_claim = field_cfg.add(
                        &(initial_claim),
                        &(&source
                            .evaluation_at(index)
                            .map_err(|_| SumcheckError::InvalidProductDimensions)
                            .unwrap()),
                    );
                }
            }

            for grinding_bits in [0, 2] {
                for prefix_vars in 0..=SHA256_INNER_PREFIX_MAX_VARS.min(num_vars) {
                    let mut callback_transcript = Blake3Transcript::new();
                    let callback = {
                        let coefficients = &|index: usize| {
                            source
                                .evaluation_at(index)
                                .map_err(|_| SumcheckError::InvalidProductDimensions)
                        };
                        let mut boundary =
                            crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                                crate::sumcheck::inner::packed::Sha256InnerGrinding,
                            >::with_round_offset(grinding_bits, 0);
                        crate::sumcheck::inner::prove_inner_sumcheck(
                            &field_cfg,
                            &mut callback_transcript,
                            initial_claim.clone(),
                            crate::sumcheck::inner::packed::PackedInput::new(
                                coefficients,
                                &h_words,
                                num_vars,
                                live_len,
                                prefix_vars,
                            ),
                            (),
                            &mut boundary,
                        )
                        .map(|out| {
                            crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                                out,
                                boundary.into_nonces(),
                            )
                        })
                    }
                    .unwrap();
                    let callback_continuation = callback_transcript.get_challenge::<u128>();

                    let mut factored_transcript = Blake3Transcript::new();
                    let factored = {
                        let coefficients = &source;
                        let mut boundary =
                            crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                                crate::sumcheck::inner::packed::Sha256InnerGrinding,
                            >::with_round_offset(grinding_bits, 0);
                        crate::sumcheck::inner::prove_inner_sumcheck(
                            &field_cfg,
                            &mut factored_transcript,
                            initial_claim.clone(),
                            crate::sumcheck::inner::packed::PackedInput::new(
                                coefficients,
                                &h_words,
                                num_vars,
                                coefficients.live_len(),
                                prefix_vars,
                            ),
                            (),
                            &mut boundary,
                        )
                        .map(|out| {
                            crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                                out,
                                boundary.into_nonces(),
                            )
                        })
                    }
                    .unwrap();

                    assert_eq!(factored, callback, "D={block_width}, K={prefix_vars}");
                    assert_eq!(
                        factored_transcript.get_challenge::<u128>(),
                        callback_continuation,
                        "D={block_width}, K={prefix_vars}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_factored_prefix_and_boundary_reductions_are_transcript_identical() {
        const INSTANCES: usize = 64;
        const BLOCK_WIDTH: usize = 257;
        const PREFIX_VARS: usize = 4;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let field_cfg = spartan_bitz_field_config();
            let zero = Field::zero_with_cfg(&field_cfg);
            let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
            let instance_weights = (0..INSTANCES)
                .map(|instance| field(37 * instance as u64 + 3, &field_cfg))
                .collect::<Vec<_>>();
            let block_coefficients = (0..BLOCK_WIDTH)
                .map(|local| field(23 * local as u64 + 7, &field_cfg))
                .collect::<Vec<_>>();
            let source = FactoredMultilinearExtension::with_leading_value(
                (1 + instance_weights.len() * block_coefficients.len())
                    .next_power_of_two()
                    .ilog2() as usize,
                field(149, &field_cfg),
                &instance_weights,
                &block_coefficients,
                &field_cfg,
            )
            .unwrap();
            let live_len = source.live_len();
            let num_vars = live_len.next_power_of_two().ilog2() as usize;
            let mut h_words = vec![0u64; live_len.div_ceil(64)];
            let mut initial_claim = zero;
            for index in 0..live_len {
                let bit = ((index * 41 + 13).count_ones() & 1) as u64;
                h_words[index / 64] |= bit << (index % 64);
                if bit != 0 {
                    initial_claim = field_cfg.add(
                        &(initial_claim),
                        &(&source
                            .evaluation_at(index)
                            .map_err(|_| SumcheckError::InvalidProductDimensions)
                            .unwrap()),
                    );
                }
            }

            let mut callback_transcript = Blake3Transcript::new();
            let callback = {
                let coefficients = &|index: usize| {
                    source
                        .evaluation_at(index)
                        .map_err(|_| SumcheckError::InvalidProductDimensions)
                };
                let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                    crate::sumcheck::inner::packed::Sha256InnerGrinding,
                >::with_round_offset(0, 0);
                crate::sumcheck::inner::prove_inner_sumcheck(
                    &field_cfg,
                    &mut callback_transcript,
                    initial_claim.clone(),
                    crate::sumcheck::inner::packed::PackedInput::new(
                        coefficients,
                        &h_words,
                        num_vars,
                        live_len,
                        PREFIX_VARS,
                    ),
                    (),
                    &mut boundary,
                )
                .map(|out| {
                    crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                        out,
                        boundary.into_nonces(),
                    )
                })
            }
            .unwrap();
            let callback_continuation = callback_transcript.get_challenge::<u128>();

            let mut factored_transcript = Blake3Transcript::new();
            let factored = {
                let coefficients = &source;
                let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                    crate::sumcheck::inner::packed::Sha256InnerGrinding,
                >::with_round_offset(0, 0);
                crate::sumcheck::inner::prove_inner_sumcheck(
                    &field_cfg,
                    &mut factored_transcript,
                    initial_claim,
                    crate::sumcheck::inner::packed::PackedInput::new(
                        coefficients,
                        &h_words,
                        num_vars,
                        coefficients.live_len(),
                        PREFIX_VARS,
                    ),
                    (),
                    &mut boundary,
                )
                .map(|out| {
                    crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                        out,
                        boundary.into_nonces(),
                    )
                })
            }
            .unwrap();

            assert_eq!(factored, callback);
            assert_eq!(
                factored_transcript.get_challenge::<u128>(),
                callback_continuation
            );
        });
    }

    fn assert_sha_width_factored_fold_matches_generic<const K: usize>() {
        const SHA_BLOCK_WIDTH: usize = 20_456;

        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let one = Field::one_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let instance_weights = [field(7, &field_cfg), field(23, &field_cfg)];
        let block_coefficients = (0..SHA_BLOCK_WIDTH)
            .map(|local| field((13 * local as u64 + 17) % 65_521, &field_cfg))
            .collect::<Vec<_>>();
        let source = FactoredMultilinearExtension::with_leading_value(
            (1 + instance_weights.len() * block_coefficients.len())
                .next_power_of_two()
                .ilog2() as usize,
            field(101, &field_cfg),
            &instance_weights,
            &block_coefficients,
            &field_cfg,
        )
        .unwrap();
        let live_len = source.live_len();
        let num_vars = live_len.next_power_of_two().ilog2() as usize;
        let challenges = (0..K)
            .map(|round| field(43 * round as u64 + 29, &field_cfg))
            .collect::<Vec<_>>();

        let generic = fold_prefix_v_table_generic::<K, _>(
            num_vars,
            live_len,
            &|index| {
                source
                    .evaluation_at(index)
                    .map_err(|_| SumcheckError::InvalidProductDimensions)
            },
            &challenges,
            &field_cfg,
            &zero,
            &one,
        )
        .unwrap();
        let factored = fold_prefix_v_table::<K, _>(
            num_vars,
            live_len,
            &source,
            &challenges,
            &field_cfg,
            &zero,
            &one,
        )
        .unwrap();

        assert_eq!(factored.suffix_count, generic.suffix_count);
        assert_eq!(factored.values, generic.values);
    }

    #[test]
    fn sha_width_factored_fold_handles_divisible_by_eight_not_sixteen_boundaries() {
        assert_sha_width_factored_fold_matches_generic::<3>();
        assert_sha_width_factored_fold_matches_generic::<4>();
    }

    fn assert_first_tail_fused_matches_reference<const K: usize>() {
        const NUM_VARS: usize = 8;
        const LIVE_LEN: usize = 13;

        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let one = Field::one_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let values = (0..LIVE_LEN)
            .map(|index| field(19 * index as u64 + 5, &field_cfg))
            .collect::<Vec<_>>();
        let mut h_words = vec![0u64; LIVE_LEN.div_ceil(64)];
        for index in 0..LIVE_LEN {
            let bit = ((index * 7 + 3).count_ones() & 1) as u64;
            h_words[index / 64] |= bit << (index % 64);
        }
        let prefix_challenges = (0..K)
            .map(|index| field(11 * index as u64 + 7, &field_cfg))
            .collect::<Vec<_>>();
        let prefix_weights = equality_weights_lsb(&prefix_challenges, &zero, &one, &field_cfg);
        let table = fold_prefix_v_table::<K, _>(
            NUM_VARS,
            LIVE_LEN,
            &|index: usize| Ok(values[index].clone()),
            &prefix_challenges,
            &field_cfg,
            &zero,
            &one,
        )
        .unwrap();
        let mut reference = CompactPrefixVTable {
            values: table.values.clone(),
            suffix_count: table.suffix_count,
        };
        let mut fused = table;
        let challenge = field(113, &field_cfg);

        fold_first_tail_round_in_place::<K, _>(
            &mut reference,
            LIVE_LEN,
            &h_words,
            &prefix_weights,
            &challenge,
            &field_cfg,
            &zero,
            &one,
        )
        .unwrap();
        let expected =
            sum_interleaved_round_coefficients(&reference.values, 1, &field_cfg).unwrap();
        let actual = fold_first_tail_round_and_prepare_next_in_place::<K, _>(
            &mut fused,
            LIVE_LEN,
            &h_words,
            &prefix_weights,
            &challenge,
            &field_cfg,
            &zero,
            &one,
            &reducer,
        )
        .unwrap();

        assert_eq!(fused.values, reference.values, "K={K}");
        assert_eq!(fused.suffix_count, reference.suffix_count, "K={K}");
        assert_eq!(actual, expected, "K={K}");
    }

    #[test]
    fn fused_first_tail_fold_and_prepare_matches_separate_passes() {
        assert_first_tail_fused_matches_reference::<0>();
        assert_first_tail_fused_matches_reference::<1>();
        assert_first_tail_fused_matches_reference::<2>();
        assert_first_tail_fused_matches_reference::<3>();
        assert_first_tail_fused_matches_reference::<4>();
    }

    #[test]
    fn fused_interleaved_fold_and_prepare_matches_separate_passes() {
        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();

        for stride in [1usize, 2, 4] {
            let fold_chunk_len = 4 * stride;
            let lengths = [
                2,
                2 * stride + 2,
                fold_chunk_len,
                fold_chunk_len + 2,
                2 * fold_chunk_len,
                3 * fold_chunk_len + 2,
                fold_chunk_len * (1 << 10),
            ];
            for length in lengths {
                for challenge in [
                    zero.clone(),
                    Field::one_with_cfg(&field_cfg),
                    field(211, &field_cfg),
                ] {
                    let values = (0..length)
                        .map(|index| raw_montgomery(&field(13 * index as u64 + 17, &field_cfg)))
                        .collect::<Vec<_>>();
                    let mut reference = values.clone();
                    let mut fused = values;

                    fold_interleaved_in_place(
                        &mut reference,
                        stride,
                        &challenge,
                        &field_cfg,
                        &zero,
                    );
                    let expected =
                        sum_interleaved_round_coefficients(&reference, 2 * stride, &field_cfg)
                            .unwrap();
                    let actual = fold_interleaved_and_prepare_next_round_in_place(
                        &mut fused, stride, &challenge, &field_cfg, &zero,
                    )
                    .unwrap();

                    assert_eq!(fused, reference, "stride={stride}, length={length}");
                    assert_eq!(actual, expected, "stride={stride}, length={length}");
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prove_dense_test<T: Transcript>(
        transcript: &mut T,
        initial_claim: Field,
        v_mle: DenseMultilinearExtension<Field>,
        h_words: &[u64],
        prefix_vars: usize,
        field_cfg: &FieldConfig,
        grinding_bits: u32,
    ) -> Result<Sha256InnerSumcheckOutput, SumcheckError> {
        let num_vars = v_mle.num_vars;
        let live_len = v_mle.evaluations.len();
        {
            let coefficients = &|index: usize| Ok(v_mle.evaluations[index].clone());
            let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                crate::sumcheck::inner::packed::Sha256InnerGrinding,
            >::with_round_offset(grinding_bits, 0);
            crate::sumcheck::inner::prove_inner_sumcheck(
                field_cfg,
                transcript,
                initial_claim,
                crate::sumcheck::inner::packed::PackedInput::new(
                    coefficients,
                    h_words,
                    num_vars,
                    live_len,
                    prefix_vars,
                ),
                (),
                &mut boundary,
            )
            .map(|out| {
                crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                    out,
                    boundary.into_nonces(),
                )
            })
        }
    }

    #[test]
    fn every_prefix_width_matches_ordinary_field_prover_with_and_without_grinding() {
        const NUM_VARS: usize = 7;

        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let one = Field::one_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let (v_mle, h_words, initial_claim) = fixture(NUM_VARS, &field_cfg);
        let h_mle = DenseMultilinearExtension::from_evaluations_vec(
            NUM_VARS,
            (0..v_mle.evaluations.len())
                .map(|index| select_field_by_bit(&zero, &one, packed_bit(&h_words, index)))
                .collect(),
            zero,
        );

        for grinding_bits in [0, 3] {
            let mut reference_transcript = Blake3Transcript::new();
            let (ordinary, reference_nonces) =
                prove_inner_sumcheck_with_reducer_grinded::<Sha256InnerGrinding, _, _>(
                    &mut reference_transcript,
                    initial_claim.clone(),
                    v_mle.clone(),
                    h_mle.clone(),
                    &field_cfg,
                    &reducer,
                    grinding_bits,
                    0,
                )
                .unwrap();
            let reference = Sha256InnerSumcheckOutput {
                sumcheck_proof: ordinary.sumcheck.proof,
                round_nonces: reference_nonces,
                eval_points: ordinary.sumcheck.eval_points,
                final_claim: ordinary.sumcheck.final_claim,
                v_evaluation: ordinary.batched_matrix_evaluation,
                h_evaluation: ordinary.witness_evaluation,
            };
            let reference_continuation = reference_transcript.get_challenge::<u128>();

            for prefix_vars in 0..=SHA256_INNER_PREFIX_MAX_VARS {
                let mut prover_transcript = Blake3Transcript::new();
                let output = prove_dense_test(
                    &mut prover_transcript,
                    initial_claim.clone(),
                    v_mle.clone(),
                    &h_words,
                    prefix_vars,
                    &field_cfg,
                    grinding_bits,
                )
                .unwrap();
                let prover_continuation = prover_transcript.get_challenge::<u128>();

                assert_eq!(output, reference, "K={prefix_vars}, bits={grinding_bits}");
                assert_eq!(prover_continuation, reference_continuation);
                assert_eq!(
                    output.round_nonces.len(),
                    if grinding_bits == 0 { 0 } else { NUM_VARS }
                );

                let mut verifier_transcript = Blake3Transcript::new();
                let (eval_points, final_claim) = output
                    .sumcheck_proof
                    .verify_grinded::<crate::sumcheck::inner::packed::Sha256InnerGrinding>(
                        &mut verifier_transcript,
                        initial_claim.clone(),
                        NUM_VARS,
                        &field_cfg,
                        &output.round_nonces,
                        grinding_bits,
                    )
                    .unwrap();
                assert_eq!(eval_points, output.eval_points);
                assert_eq!(final_claim, output.final_claim);
                assert_eq!(
                    final_claim,
                    field_cfg.mul(&(output.v_evaluation.clone()), &(&output.h_evaluation))
                );
                assert_eq!(
                    verifier_transcript.get_challenge::<u128>(),
                    reference_continuation
                );
            }
        }
    }

    #[test]
    fn live_prefix_storage_omits_the_global_zero_suffix_and_matches_dense_reference() {
        const NUM_VARS: usize = 7;
        const LIVE_LEN: usize = 61;

        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let one = Field::one_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let table_len = 1usize << NUM_VARS;
        let mut v = vec![zero.clone(); table_len];
        let mut h_padded = vec![0u64; table_len.div_ceil(u64::BITS as usize)];
        let mut initial_claim = zero.clone();
        for index in 0..LIVE_LEN {
            v[index] = field((29 * index as u64 + 11) % 251, &field_cfg);
            let bit = ((index * 13 + 5).count_ones() & 1) as u64;
            h_padded[index / 64] |= bit << (index % 64);
            if bit != 0 {
                initial_claim = field_cfg.add(&(initial_claim), &(&v[index]));
            }
        }
        let h_live = h_padded[..LIVE_LEN.div_ceil(64)].to_vec();
        let v_mle =
            DenseMultilinearExtension::from_evaluations_vec(NUM_VARS, v.clone(), zero.clone());
        let h_mle = DenseMultilinearExtension::from_evaluations_vec(
            NUM_VARS,
            (0..table_len)
                .map(|index| select_field_by_bit(&zero, &one, packed_bit(&h_padded, index)))
                .collect(),
            zero.clone(),
        );
        let mut reference_transcript = Blake3Transcript::new();
        let reference = prove_inner_sumcheck_with_reducer(
            &mut reference_transcript,
            initial_claim.clone(),
            v_mle,
            h_mle,
            &field_cfg,
            &reducer,
        )
        .unwrap();
        let reference_continuation = reference_transcript.get_challenge::<u128>();

        for prefix_vars in 0..=SHA256_INNER_PREFIX_MAX_VARS {
            for h_words in [&h_live[..], &h_padded[..]] {
                let largest_query = AtomicUsize::new(0);
                let oracle = |index: usize| {
                    if index >= LIVE_LEN {
                        return Err(SumcheckError::InvalidProductDimensions);
                    }
                    largest_query.fetch_max(index, Ordering::Relaxed);
                    Ok(v[index].clone())
                };
                let mut transcript = Blake3Transcript::new();
                let output = {
                    let coefficients = &oracle;
                    let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                        crate::sumcheck::inner::packed::Sha256InnerGrinding,
                    >::with_round_offset(0, 0);
                    crate::sumcheck::inner::prove_inner_sumcheck(
                        &field_cfg,
                        &mut transcript,
                        initial_claim.clone(),
                        crate::sumcheck::inner::packed::PackedInput::new(
                            coefficients,
                            h_words,
                            NUM_VARS,
                            LIVE_LEN,
                            prefix_vars,
                        ),
                        (),
                        &mut boundary,
                    )
                    .map(|out| {
                        crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                            out,
                            boundary.into_nonces(),
                        )
                    })
                }
                .unwrap();

                assert_eq!(largest_query.load(Ordering::Relaxed), LIVE_LEN - 1);
                assert_eq!(output.sumcheck_proof, reference.sumcheck.proof);
                assert_eq!(output.eval_points, reference.sumcheck.eval_points);
                assert_eq!(output.final_claim, reference.sumcheck.final_claim);
                assert_eq!(output.v_evaluation, reference.batched_matrix_evaluation);
                assert_eq!(output.h_evaluation, reference.witness_evaluation);
                assert_eq!(transcript.get_challenge::<u128>(), reference_continuation);
            }
        }

        assert_eq!(core::mem::size_of::<RawMontgomery>(), 16);
        assert_eq!(
            core::mem::size_of::<Field>(),
            core::mem::size_of::<RawMontgomery>()
        );
        let challenges = [field(7, &field_cfg), field(19, &field_cfg)];
        let table = fold_prefix_v_table::<2, _>(
            NUM_VARS,
            LIVE_LEN,
            &|index: usize| Ok(v[index].clone()),
            &challenges,
            &field_cfg,
            &zero,
            &one,
        )
        .unwrap();
        let suffix_count = LIVE_LEN.div_ceil(1 << 2);
        assert_eq!(table.suffix_count, suffix_count);
        assert_eq!(table.values.len(), (suffix_count + 1) & !1);
        assert!(table.values.len() < 2 * suffix_count);
        assert!(table.values.iter().all(|raw| {
            field_from_raw(raw, &field_cfg)
                .as_montgomery_integer()
                .as_words()
                == raw
        }));
    }

    #[test]
    fn lazy_bit_source_spanning_packed_rows_matches_contiguous_words() {
        const NUM_VARS: usize = 7;
        const LIVE_LEN: usize = 93;
        const ROW_BITS: usize = 19;

        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let table_len = 1usize << NUM_VARS;
        let mut v = vec![zero.clone(); table_len];
        let mut h_words = vec![0u64; LIVE_LEN.div_ceil(64)];
        let mut h_rows = vec![0u64; LIVE_LEN.div_ceil(ROW_BITS)];
        let mut initial_claim = zero;
        for index in 0..LIVE_LEN {
            v[index] = field((41 * index as u64 + 7) % 251, &field_cfg);
            let bit = ((index * 23 + 9).count_ones() & 1) as u64;
            h_words[index / 64] |= bit << (index % 64);
            h_rows[index / ROW_BITS] |= bit << (index % ROW_BITS);
            if bit != 0 {
                initial_claim = field_cfg.add(&(initial_claim), &(&v[index]));
            }
        }

        for prefix_vars in 0..=SHA256_INNER_PREFIX_MAX_VARS {
            let mut packed_transcript = Blake3Transcript::new();
            let packed = {
                let coefficients = &|index: usize| Ok(v[index].clone());
                let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                    crate::sumcheck::inner::packed::Sha256InnerGrinding,
                >::with_round_offset(0, 0);
                crate::sumcheck::inner::prove_inner_sumcheck(
                    &field_cfg,
                    &mut packed_transcript,
                    initial_claim.clone(),
                    crate::sumcheck::inner::packed::PackedInput::new(
                        coefficients,
                        &h_words,
                        NUM_VARS,
                        LIVE_LEN,
                        prefix_vars,
                    ),
                    (),
                    &mut boundary,
                )
                .map(|out| {
                    crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                        out,
                        boundary.into_nonces(),
                    )
                })
            }
            .unwrap();

            let largest_query = AtomicUsize::new(0);
            let row_source = |index: usize| {
                if index >= LIVE_LEN {
                    return Err(SumcheckError::InvalidProductDimensions);
                }
                largest_query.fetch_max(index, Ordering::Relaxed);
                Ok((h_rows[index / ROW_BITS] >> (index % ROW_BITS)) & 1)
            };
            let mut lazy_transcript = Blake3Transcript::new();
            let lazy = {
                let coefficients = &|index: usize| Ok(v[index].clone());
                let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                    crate::sumcheck::inner::packed::Sha256InnerGrinding,
                >::with_round_offset(0, 0);
                crate::sumcheck::inner::prove_inner_sumcheck(
                    &field_cfg,
                    &mut lazy_transcript,
                    initial_claim.clone(),
                    crate::sumcheck::inner::packed::PackedInput::new(
                        coefficients,
                        &row_source,
                        NUM_VARS,
                        LIVE_LEN,
                        prefix_vars,
                    ),
                    (),
                    &mut boundary,
                )
                .map(|out| {
                    crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                        out,
                        boundary.into_nonces(),
                    )
                })
            }
            .unwrap();

            assert_eq!(largest_query.load(Ordering::Relaxed), LIVE_LEN - 1);
            assert_eq!(lazy, packed);
            assert_eq!(
                lazy_transcript.get_challenge::<u128>(),
                packed_transcript.get_challenge::<u128>()
            );
        }
    }

    #[test]
    fn noncanonical_structured_inputs_are_rejected_before_the_transcript() {
        let cfg = spartan_bitz_field_config();
        let invalid = crate::piop::spartan::noncanonical_test_value(&cfg);
        for invalid_part in 0..3 {
            let mut high = [field(2, &cfg), field(3, &cfg)];
            let low = [field(5, &cfg), field(7, &cfg)];
            let mut tail = [field(11, &cfg), field(13, &cfg)];
            let mut origin = cfg.zero();
            match invalid_part {
                0 => high[1] = invalid,
                1 => tail[1] = invalid,
                _ => origin = invalid,
            }
            let coefficients = crate::poly::mle::CompositeMultilinearExtension::from_parts(
                3, &high, &low, &tail, origin, &cfg,
            )
            .unwrap();
            let bits = vec![0u64];
            let mut transcript = Blake3Transcript::new();
            let mut untouched = transcript.clone();
            let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                Sha256InnerGrinding,
            >::with_round_offset(0, 0);
            let result = crate::sumcheck::inner::prove_inner_sumcheck(
                &cfg,
                &mut transcript,
                cfg.zero(),
                PackedInput::new(&coefficients, &bits, 3, 6, 2),
                (),
                &mut boundary,
            );
            assert!(matches!(
                result,
                Err(SumcheckError::NonCanonicalFieldElement)
            ));
            assert_eq!(
                transcript.get_challenge::<u128>(),
                untouched.get_challenge::<u128>()
            );
        }
    }

    #[test]
    fn noncanonical_callback_is_rejected_before_the_transcript() {
        let field_cfg = spartan_bitz_field_config();
        let (_, h_words, initial_claim) = fixture(3, &field_cfg);
        let mut transcript = Blake3Transcript::new();
        let mut untouched = transcript.clone();
        let result = {
            let coefficients = &|_| Ok(crate::piop::spartan::noncanonical_test_value(&field_cfg));
            let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                crate::sumcheck::inner::packed::Sha256InnerGrinding,
            >::with_round_offset(0, 0);
            crate::sumcheck::inner::prove_inner_sumcheck(
                &field_cfg,
                &mut transcript,
                initial_claim,
                crate::sumcheck::inner::packed::PackedInput::new(coefficients, &h_words, 3, 8, 2),
                (),
                &mut boundary,
            )
            .map(|out| {
                crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                    out,
                    boundary.into_nonces(),
                )
            })
        };
        assert_eq!(result, Err(SumcheckError::NonCanonicalFieldElement));
        assert_eq!(
            transcript.get_challenge::<u128>(),
            untouched.get_challenge::<u128>()
        );
    }

    #[test]
    fn k0_is_identical_to_the_existing_field_quadratic_prover() {
        const NUM_VARS: usize = 7;

        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let one = Field::one_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let (v_mle, h_words, initial_claim) = fixture(NUM_VARS, &field_cfg);
        let h_mle = DenseMultilinearExtension::from_evaluations_vec(
            NUM_VARS,
            (0..v_mle.evaluations.len())
                .map(|index| select_field_by_bit(&zero, &one, packed_bit(&h_words, index)))
                .collect(),
            zero,
        );

        let mut ordinary_transcript = Blake3Transcript::new();
        let ordinary = prove_inner_sumcheck_with_reducer(
            &mut ordinary_transcript,
            initial_claim.clone(),
            v_mle.clone(),
            h_mle,
            &field_cfg,
            &reducer,
        )
        .unwrap();
        let ordinary_continuation = ordinary_transcript.get_challenge::<u128>();

        let mut k0_transcript = Blake3Transcript::new();
        let k0 = prove_dense_test(
            &mut k0_transcript,
            initial_claim,
            v_mle,
            &h_words,
            0,
            &field_cfg,
            0,
        )
        .unwrap();

        assert_eq!(k0.sumcheck_proof, ordinary.sumcheck.proof);
        assert_eq!(k0.eval_points, ordinary.sumcheck.eval_points);
        assert_eq!(k0.final_claim, ordinary.sumcheck.final_claim);
        assert_eq!(k0.v_evaluation, ordinary.batched_matrix_evaluation);
        assert_eq!(k0.h_evaluation, ordinary.witness_evaluation);
        assert!(k0.round_nonces.is_empty());
        assert_eq!(k0_transcript.get_challenge::<u128>(), ordinary_continuation);
    }

    #[test]
    fn extension_uses_low_coordinate_first_u2_layout() {
        // f(x0, x1) = x0 XOR x1 in binary table order with x0 as the low bit.
        let mut values = vec![0i64, 1, 1, 0];
        let mut scratch = vec![0i64; pow3(2)];
        extend_lsb::<i64, 2, _>(&mut values, &mut scratch, &0, |high, low| *high - *low);

        // Ternary digit order per coordinate is [infinity, 0, 1], with x0
        // the least-significant digit.
        assert_eq!(values, vec![-2, 1, -1, 1, 0, 1, -1, 1, 0]);
    }

    #[test]
    fn first_round_is_pinned_to_the_word_lsb() {
        let field_cfg = spartan_bitz_field_config();
        let zero = Field::zero_with_cfg(&field_cfg);
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let v_mle = DenseMultilinearExtension::from_evaluations_vec(
            2,
            [2, 5, 11, 17]
                .into_iter()
                .map(|value| field(value, &field_cfg))
                .collect(),
            zero.clone(),
        );
        // h = [1, 0, 1, 1], so the exact first polynomial is
        // 13 + 7 X - 3 X^2 when coordinate zero is the word's low bit.
        let initial_claim = field(30, &field_cfg);

        for prefix_vars in [0, 1, 2] {
            let output = prove_dense_test(
                &mut Blake3Transcript::new(),
                initial_claim.clone(),
                v_mle.clone(),
                &[0b1101],
                prefix_vars,
                &field_cfg,
                0,
            )
            .unwrap();
            assert_eq!(
                output.sumcheck_proof.round_polynomials[0],
                [
                    field(13, &field_cfg),
                    field(7, &field_cfg),
                    field_cfg.sub(&(zero.clone()), &(field(3, &field_cfg))),
                ]
            );
        }
    }

    #[test]
    fn zero_variable_and_fully_native_prefixes_match_k0() {
        let field_cfg = spartan_bitz_field_config();
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();

        let zero_var = prove_dense_test(
            &mut Blake3Transcript::new(),
            field(9, &field_cfg),
            DenseMultilinearExtension::zero_vars(field(9, &field_cfg)),
            &[1],
            0,
            &field_cfg,
            0,
        )
        .unwrap();
        assert!(zero_var.sumcheck_proof.round_polynomials.is_empty());
        assert!(zero_var.eval_points.is_empty());
        assert_eq!(zero_var.v_evaluation, field(9, &field_cfg));
        assert_eq!(zero_var.h_evaluation, field(1, &field_cfg));

        let (v_mle, h_words, initial_claim) = fixture(4, &field_cfg);
        let reference = prove_dense_test(
            &mut Blake3Transcript::new(),
            initial_claim.clone(),
            v_mle.clone(),
            &h_words,
            0,
            &field_cfg,
            0,
        )
        .unwrap();
        let fully_native = prove_dense_test(
            &mut Blake3Transcript::new(),
            initial_claim,
            v_mle,
            &h_words,
            4,
            &field_cfg,
            0,
        )
        .unwrap();
        assert_eq!(fully_native, reference);
        assert_eq!(
            fully_native.final_claim,
            field_cfg.mul(
                &(fully_native.v_evaluation.clone()),
                &(&fully_native.h_evaluation)
            )
        );
    }

    #[test]
    fn invalid_prefix_or_noncanonical_padding_does_not_touch_transcript() {
        let field_cfg = spartan_bitz_field_config();
        let reducer = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let (v_mle, h_words, initial_claim) = fixture(3, &field_cfg);

        let mut invalid_k_transcript = Blake3Transcript::new();
        let untouched_k = invalid_k_transcript.clone();
        assert_eq!(
            prove_dense_test(
                &mut invalid_k_transcript,
                initial_claim.clone(),
                v_mle.clone(),
                &h_words,
                SHA256_INNER_PREFIX_MAX_VARS + 1,
                &field_cfg,
                0,
            ),
            Err(SumcheckError::InvalidProductDimensions)
        );
        assert_eq!(
            invalid_k_transcript.get_challenge::<u128>(),
            untouched_k.clone().get_challenge::<u128>()
        );

        let mut too_wide_transcript = Blake3Transcript::new();
        let mut untouched_width = too_wide_transcript.clone();
        assert_eq!(
            prove_dense_test(
                &mut too_wide_transcript,
                initial_claim.clone(),
                v_mle.clone(),
                &h_words,
                4,
                &field_cfg,
                0,
            ),
            Err(SumcheckError::InvalidProductDimensions)
        );
        assert_eq!(
            too_wide_transcript.get_challenge::<u128>(),
            untouched_width.get_challenge::<u128>()
        );

        for prefix_vars in [0, 2] {
            let mut invalid_bit_transcript = Blake3Transcript::new();
            let mut untouched_bit = invalid_bit_transcript.clone();
            assert_eq!(
                {
                    let coefficients = &|index: usize| Ok(v_mle.evaluations[index].clone());
                    let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                        crate::sumcheck::inner::packed::Sha256InnerGrinding,
                    >::with_round_offset(0, 0);
                    crate::sumcheck::inner::prove_inner_sumcheck(
                        &field_cfg,
                        &mut invalid_bit_transcript,
                        initial_claim.clone(),
                        crate::sumcheck::inner::packed::PackedInput::new(
                            coefficients,
                            &|_| Ok(2),
                            v_mle.num_vars,
                            v_mle.evaluations.len(),
                            prefix_vars,
                        ),
                        (),
                        &mut boundary,
                    )
                    .map(|out| {
                        crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                            out,
                            boundary.into_nonces(),
                        )
                    })
                },
                Err(SumcheckError::InvalidProductDimensions)
            );
            assert_eq!(
                invalid_bit_transcript.get_challenge::<u128>(),
                untouched_bit.get_challenge::<u128>()
            );
        }

        let mut padded_words = h_words;
        padded_words[0] |= 1 << 63;
        let mut padded_transcript = Blake3Transcript::new();
        let mut untouched_padding = padded_transcript.clone();
        assert_eq!(
            prove_dense_test(
                &mut padded_transcript,
                initial_claim,
                v_mle,
                &padded_words,
                2,
                &field_cfg,
                0,
            ),
            Err(SumcheckError::InvalidProductDimensions)
        );
        assert_eq!(
            padded_transcript.get_challenge::<u128>(),
            untouched_padding.get_challenge::<u128>()
        );
    }
}
