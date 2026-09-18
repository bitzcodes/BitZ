//! Validated R1CS matrices and multilinear-table helpers for Spartan.
//!
//! This module starts at the field-valued R1CS boundary. Constraint generation
//! and the F2-to-integer witness map are intentionally out of scope: callers
//! supply the three matrices `A`, `B`, and `C`, their row products, and the
//! complete assignment consumed by those matrices.

#[cfg(test)]
use crate::sumcheck::bridge::PreparedBinding;
use circuit::linear_map::SparseMatrixError;
use field::RingOps;
#[cfg(test)]
use field::{Fp, IntegerEmbedding, Uint};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;

use blake3::Hasher;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;

use super::spliced_digest::{SplicedDigestBuilder, SplicedStreamDigest};
use crate::poly::mle::DenseMultilinearExtension;

/// Domain tag of the prepared-statement digest (`v2`), shared verbatim by
/// the entry-wise walk and the skeleton's cached stream.
const CONSTRAINT_MATRIX_DIGEST_DOMAIN: &[u8] = b"bitz/spartan/constraint-matrices/v2";

use circuit::linear_map::CscMatrix;

use super::{SpartanField, SpartanFieldError, sumcheck::R1csProductMles};

/// Form field row operands from the complete padded Montgomery assignment.
/// Traversal depends only on public sparse structure, never on witness values.
pub(crate) fn products_from_montgomery_assignment<C>(
    matrices: &PreparedConstraintMatrices<field::Fp<2>, C>,
    assignment: &[u128],
) -> Result<crate::sumcheck::outer::OuterInputs<field::Fp<2>>, super::SpartanError>
where
    C: SpartanMatrixCoefficient<field::Fp<2>>,
{
    use super::raw_monty::{raw_shared, shared_raw};
    use field::{CtEq, CtOrd};
    let _scope = tracing::info_span!("sp:field_assignment_products").entered();
    let field = matrices.config();
    let logical = matrices.matrices().column_count();
    if assignment.len() != 1usize << matrices.num_column_vars() {
        return Err(super::SpartanError::InvalidAssignmentDimensions);
    }
    let modulus = field.modulus();
    let valid = assignment.iter().fold(field::CtMask::TRUE, |valid, value| {
        valid & field::Uint::<2>::from(*value).ct_lt(modulus)
    });
    if !valid.declassify() {
        return Err(super::SpartanError::FieldConfigurationMismatch);
    }
    if !field::Uint::<2>::from(assignment[0])
        .ct_eq(&field::Uint::<2>::from(raw_shared(field.one())))
        .declassify()
    {
        return Err(SpartanMatrixError::InvalidAssignmentConstant.into());
    }
    if assignment[logical..]
        .iter()
        .fold(0, |any, &value| any | value)
        != 0
    {
        return Err(super::SpartanError::InvalidAssignmentPadding);
    }
    let product = |matrix: &CscMatrix<Box<[C]>>| {
        let mut out = vec![field.zero(); 1usize << matrices.num_row_vars()];
        let offsets = matrix.column_offsets();
        for column in 0..matrix.column_count() {
            let value = shared_raw(field, assignment[column]);
            for index in offsets[column]..offsets[column + 1] {
                let row = matrix.row_indices()[index];
                let term = matrix.coefficients()[index].scale(&value, field);
                out[row] = field.add(&out[row], &term);
            }
        }
        out
    };
    let m = matrices.matrices();
    #[cfg(feature = "parallel")]
    if m.a().nnz() + m.b().nnz() + m.c().nnz() >= 4096 && rayon::current_num_threads() > 1 {
        let (ax, (bx, cx)) = rayon::join(
            || product(m.a()),
            || rayon::join(|| product(m.b()), || product(m.c())),
        );
        return Ok(crate::sumcheck::outer::OuterInputs { ax, bx, cx });
    }
    Ok(crate::sumcheck::outer::OuterInputs {
        ax: product(m.a()),
        bx: product(m.b()),
        cx: product(m.c()),
    })
}

/// A sparse R1CS coefficient that can act on values in `F`.
///
/// The coefficient's canonical encoding is always the encoding of the
/// corresponding element of `F`. Consequently, a Bit matrix containing
/// `true` has the same prepared-statement digest as a field-valued matrix
/// containing `F::one_with_cfg(field_config)` at the same coordinates.
pub trait SpartanMatrixCoefficient<F>: Clone + Send + Sync
where
    F: SpartanField,
{
    /// Validates coefficient-specific invariants against the prepared field.
    fn validate(&self, field_modulus_encoding: &[u8]) -> Result<(), SpartanMatrixError>;

    /// Whether this is an explicit zero, which sparse matrices forbid.
    fn is_zero(&self) -> bool;

    /// Canonical encoding of this coefficient as an element of `F`.
    fn canonical_field_encoding<'a>(
        &'a self,
        field_config: &F::Config,
        field_one_encoding: &'a [u8],
    ) -> Cow<'a, [u8]>;

    /// Multiplies a field value by this coefficient.
    fn scale(&self, value: &F, field_config: &F::Config) -> F;
}

impl<F> SpartanMatrixCoefficient<F> for F
where
    F: SpartanField,
{
    fn validate(&self, field_modulus_encoding: &[u8]) -> Result<(), SpartanMatrixError> {
        validate_element_field(self, field_modulus_encoding)
    }

    fn is_zero(&self) -> bool {
        F::is_zero(self)
    }

    fn canonical_field_encoding<'a>(
        &'a self,
        field_config: &F::Config,
        _field_one_encoding: &'a [u8],
    ) -> Cow<'a, [u8]> {
        Cow::Owned(self.canonical_element_encoding(field_config))
    }

    fn scale(&self, value: &F, field_config: &F::Config) -> F {
        (field_config).mul(value, self)
    }
}

impl<F> SpartanMatrixCoefficient<F> for bool
where
    F: SpartanField,
{
    fn validate(&self, _field_modulus_encoding: &[u8]) -> Result<(), SpartanMatrixError> {
        Ok(())
    }

    fn is_zero(&self) -> bool {
        !*self
    }

    fn canonical_field_encoding<'a>(
        &'a self,
        field_config: &F::Config,
        field_one_encoding: &'a [u8],
    ) -> Cow<'a, [u8]> {
        if *self {
            Cow::Borrowed(field_one_encoding)
        } else {
            Cow::Owned(F::zero_with_cfg(field_config).canonical_element_encoding(field_config))
        }
    }

    fn scale(&self, value: &F, field_config: &F::Config) -> F {
        if *self {
            value.clone()
        } else {
            F::zero_with_cfg(field_config)
        }
    }
}

/// A sparse coefficient whose field action and canonical encoding are the
/// same under every runtime field configuration.
///
/// This is the contract that lets a [`ConstraintMatricesSkeleton`] hoist the
/// per-statement clone, validation, canonical digesting, and selector
/// detection out of the per-proof [`PreparedConstraintMatrices`] construction
/// when the modulus is drawn from the transcript. Implementations promise
/// that, for every configuration accepted by `F::validate_config`:
///
/// - [`SpartanMatrixCoefficient::validate`] succeeds unconditionally,
/// - [`SpartanMatrixCoefficient::canonical_field_encoding`] returns the exact
///   bytes written by [`Self::write_modulus_independent_encoding`], and
/// - [`Self::is_unit`] is `true` exactly when that encoding equals the
///   field's canonical one-encoding.
///
/// Under this contract [`PreparedConstraintMatrices::from_skeleton`] produces
/// the same statement digest as [`PreparedConstraintMatrices::new`] for every
/// accepted configuration.
pub trait ModulusIndependentCoefficient<F>: SpartanMatrixCoefficient<F>
where
    F: SpartanField,
{
    /// Appends the exact bytes
    /// [`SpartanMatrixCoefficient::canonical_field_encoding`] returns under
    /// every accepted configuration.
    fn write_modulus_independent_encoding(&self, out: &mut Vec<u8>);

    /// Whether this coefficient acts as the multiplicative unit under every
    /// accepted configuration.
    fn is_unit(&self) -> bool;
}

/// Failures while constructing or evaluating a Spartan matrix statement.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum SpartanMatrixError {
    /// The generic CSC matrix is malformed.
    #[error(transparent)]
    CscMatrix(#[from] SparseMatrixError),

    /// `A`, `B`, and `C` do not describe one common R1CS shape.
    #[error("A, B, and C must have identical, nonempty dimensions")]
    InvalidR1csShape,

    /// A table length cannot be rounded up to a Bit-hypercube domain.
    #[error("the requested Bit-hypercube domain is too large")]
    DomainTooLarge,

    /// A supplied value belongs to a different runtime field configuration.
    #[error("a field element does not use the prepared field configuration")]
    FieldConfigurationMismatch,

    /// The selected runtime field is unsafe for Spartan.
    #[error(transparent)]
    InvalidFieldConfiguration(#[from] SpartanFieldError),

    /// Explicit zero entries give a sparse matrix more than one encoding.
    #[error("matrix {matrix} contains an explicit zero at row {row}, column {column}")]
    ExplicitZeroCoefficient {
        matrix: &'static str,
        row: usize,
        column: usize,
    },

    /// One of `Az`, `Bz`, or `Cz` has the wrong logical row count.
    #[error("product table has length {actual}, expected {expected}")]
    InvalidProductLength { expected: usize, actual: usize },

    /// The assignment does not have the declared logical column count.
    #[error("assignment has length {actual}, expected {expected}")]
    InvalidAssignmentLength { expected: usize, actual: usize },

    /// R1CS column zero is the constant-one assignment entry.
    #[error("the assignment must begin with the constant one")]
    InvalidAssignmentConstant,

    /// A row-domain evaluation point has the wrong width.
    #[error("row point has width {actual}, expected {expected}")]
    InvalidRowPointLength { expected: usize, actual: usize },

    /// A row-domain weight table has the wrong padded length.
    #[error("row-weight table has length {actual}, expected {expected}")]
    InvalidRowWeightsLength { expected: usize, actual: usize },

    /// A column-domain evaluation point has the wrong width.
    #[error("column point has width {actual}, expected {expected}")]
    InvalidColumnPointLength { expected: usize, actual: usize },

    /// A dense MLE's metadata and complete table disagree.
    #[error("invalid dense multilinear-extension table")]
    InvalidMleOperation,
}

/// Why a full MLE table does not discharge a scaled evaluation claim.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum MleClaimError {
    /// The supplied table, point, or field configuration is malformed.
    #[error("invalid multilinear polynomial: {0}")]
    InvalidPolynomial(#[from] SpartanMatrixError),

    /// The table evaluation does not equal the claimed scaled value.
    #[error("the multilinear evaluation does not match the claim")]
    InvalidEvaluation,
}

/// Compact row functional produced by a prefix-univariate outer reduction.
///
/// For `M = 2^K` and `row = s + M * x`, the represented weight is
///
/// `prefix[s] * tail_low[x_low] * tail_high[x_high]`.
///
/// Keeping the three factors separate avoids materializing the complete
/// `2^num_row_vars` row-weight table when the prepared matrix layout supports
/// streamed row binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PrefixUnivariateRowFactors<F> {
    skip_vars: usize,
    prefix: Box<[F]>,
    tail_low: Box<[F]>,
    tail_high: Box<[F]>,
    tail_low_vars: usize,
    num_row_vars: usize,
}

impl<F> PrefixUnivariateRowFactors<F>
where
    F: SpartanField,
{
    pub(crate) fn new(
        skip_vars: usize,
        prefix: Vec<F>,
        tail_low: DenseMultilinearExtension<F>,
        tail_high: DenseMultilinearExtension<F>,
        num_row_vars: usize,
    ) -> Result<Self, SpartanMatrixError> {
        let expected_prefix = domain_size(skip_vars)?;
        let expected_low = domain_size(tail_low.num_vars)?;
        let expected_high = domain_size(tail_high.num_vars)?;
        let actual_rows = prefix
            .len()
            .checked_mul(tail_low.evaluations.len())
            .and_then(|length| length.checked_mul(tail_high.evaluations.len()))
            .ok_or(SpartanMatrixError::DomainTooLarge)?;
        let expected_rows = domain_size(num_row_vars)?;
        if prefix.len() != expected_prefix
            || tail_low.evaluations.len() != expected_low
            || tail_high.evaluations.len() != expected_high
            || skip_vars
                .checked_add(tail_low.num_vars)
                .and_then(|width| width.checked_add(tail_high.num_vars))
                != Some(num_row_vars)
            || actual_rows != expected_rows
        {
            return Err(SpartanMatrixError::InvalidRowWeightsLength {
                expected: expected_rows,
                actual: actual_rows,
            });
        }

        Ok(Self {
            skip_vars,
            prefix: prefix.into_boxed_slice(),
            tail_low: tail_low.evaluations.into_boxed_slice(),
            tail_high: tail_high.evaluations.into_boxed_slice(),
            tail_low_vars: tail_low.num_vars,
            num_row_vars,
        })
    }

    /// Visits every logical row in little-endian `s + 2^K*x` order.
    pub(crate) fn for_each_row(
        &self,
        logical_rows: usize,
        mut consume: impl FnMut(usize, F),
        field_config: &F::Config,
    ) {
        debug_assert!(logical_rows <= 1usize << self.num_row_vars);
        let block_len = 1usize << self.skip_vars;
        let low_mask = self.tail_low.len() - 1;
        let suffixes = logical_rows.div_ceil(block_len);

        for suffix in 0..suffixes {
            let low_index = suffix & low_mask;
            let high_index = suffix >> self.tail_low_vars;
            let tail_weight =
                (field_config).mul(&self.tail_low[low_index], &self.tail_high[high_index]);
            let row_start = suffix * block_len;
            let active = block_len.min(logical_rows - row_start);
            for prefix_index in 0..active {
                consume(
                    row_start + prefix_index,
                    (field_config).mul(&self.prefix[prefix_index], &tail_weight),
                );
            }
        }
    }

    /// Borrowed view of the three factors, for the raw prover-side binding
    /// kernels.
    pub(crate) fn parts(&self) -> PrefixRowFactorParts<'_, F> {
        PrefixRowFactorParts {
            skip_vars: self.skip_vars,
            prefix: &self.prefix,
            tail_low: &self.tail_low,
            tail_high: &self.tail_high,
            tail_low_vars: self.tail_low_vars,
            num_row_vars: self.num_row_vars,
        }
    }

    /// Reference/fallback materialization in canonical row order.
    pub(crate) fn materialize(&self, field_config: &F::Config) -> Vec<F> {
        let mut weights = Vec::with_capacity(1usize << self.num_row_vars);
        self.for_each_row(
            1usize << self.num_row_vars,
            |row, weight| {
                debug_assert_eq!(row, weights.len());
                weights.push(weight);
            },
            &field_config,
        );
        weights
    }
}

/// The borrowed factors of a [`PrefixUnivariateRowFactors`]: the weight at
/// row `s + 2^K x` is `prefix[s] · tail_low[x mod 2^tail_low_vars] ·
/// tail_high[x >> tail_low_vars]`.
pub(crate) struct PrefixRowFactorParts<'a, F> {
    pub skip_vars: usize,
    pub prefix: &'a [F],
    pub tail_low: &'a [F],
    pub tail_high: &'a [F],
    pub tail_low_vars: usize,
    pub num_row_vars: usize,
}

/// A row functional in product form, as the verifier derives it from the
/// outer reduction: the weight of logical row `s + 2^K·x` (`s < 2^K`) is
///
/// `prefix[s] · eq(tail_point, x)`.
///
/// `skip_vars = 0` with `prefix = [1]` is the plain equality functional
/// `eq(tail_point, ·)` of the standard outer sumcheck; `skip_vars = K ≥ 1`
/// with `prefix = (L_s(z))_{s < 2^K}` is the prefix-univariate functional of
/// the univariate-skip reduction, whose materialized form is
/// [`PrefixUnivariateRowFactors`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProductRowFunctional<'a, F> {
    pub skip_vars: usize,
    pub prefix: &'a [F],
    pub tail_point: &'a [F],
}

/// The three field-valued matrices defining an R1CS relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintMatrices<F> {
    a: CscMatrix<Box<[F]>>,
    b: CscMatrix<Box<[F]>>,
    c: CscMatrix<Box<[F]>>,
}

impl<F> ConstraintMatrices<F> {
    /// Validates that `A`, `B`, and `C` share one nonempty shape.
    pub fn new(
        a: CscMatrix<Box<[F]>>,
        b: CscMatrix<Box<[F]>>,
        c: CscMatrix<Box<[F]>>,
    ) -> Result<Self, SpartanMatrixError> {
        let rows = a.row_count();
        let columns = a.column_count();
        if rows == 0
            || columns == 0
            || b.row_count() != rows
            || c.row_count() != rows
            || b.column_count() != columns
            || c.column_count() != columns
        {
            return Err(SpartanMatrixError::InvalidR1csShape);
        }

        Ok(Self { a, b, c })
    }

    /// Left R1CS matrix.
    pub const fn a(&self) -> &CscMatrix<Box<[F]>> {
        &self.a
    }

    /// Right R1CS matrix.
    pub const fn b(&self) -> &CscMatrix<Box<[F]>> {
        &self.b
    }

    /// Output R1CS matrix.
    pub const fn c(&self) -> &CscMatrix<Box<[F]>> {
        &self.c
    }

    /// Shared logical row count.
    pub fn row_count(&self) -> usize {
        self.a.row_count()
    }

    /// Shared logical column count.
    pub const fn column_count(&self) -> usize {
        self.a.column_count()
    }
}

/// A validated, digest-bound R1CS matrix statement prepared for reuse.
#[derive(Clone, Copy, Debug)]
struct DisjointUnitSelectorTriplet {
    rows: usize,
    a_offset: usize,
    b_offset: usize,
    c_offset: usize,
}

/// One block-aligned selector run of a sparse matrix: columns
/// `start + r` for `r < rows` each hold exactly the entry `(r, coefficient)`.
#[derive(Clone, Debug)]
pub(crate) struct SelectorRun<C> {
    pub start: usize,
    pub coefficient: C,
}

/// A description of R1CS matrices whose every nonzero column is part of a
/// block-aligned unit-row selector run with a run-constant coefficient (the
/// shape of the generated u32, BabyBear and CM-AND relations):
/// `M = Σ_runs coefficient · Sel(start)` with `Sel(start)[i, start + i] = 1`
/// for `i < rows`, detected from the validated matrices by
/// [`detect_block_selector_layout_with`].
///
/// With row weights `W`, the batched column MLE is then
/// `D(k · block_len + r) = W[r] · Σ_{runs starting at k · block_len} f · coefficient`
/// for `r < rows` (and zero elsewhere), where `f ∈ {1, ρ, ρ²}` is the run's
/// matrix factor: scaled copies of ONE weight vector, which the raw prover
/// binds and folds without materializing `D`. The verifier evaluates the
/// batched matrices at its terminal point through the same structure in
/// time logarithmic in the relation size
/// ([`crate::sumcheck::bridge::structured::StructuredBinding::evaluate_product`]).
#[derive(Clone, Debug)]
pub(crate) struct BlockSelectorLayout<C> {
    /// Logical row count, the length of every run.
    pub rows: usize,
    /// Power-of-two block length: at least `rows`, dividing every run start
    /// and the padded column domain.
    pub block_len: usize,
    pub a: Vec<SelectorRun<C>>,
    pub b: Vec<SelectorRun<C>>,
    pub c: Vec<SelectorRun<C>>,
}

/// The modulus-independent core of a prepared R1CS statement: validated,
/// digest-ready matrices plus the padded domain widths and the detected
/// selector layout — everything [`PreparedConstraintMatrices::new`] derives
/// that does not depend on the runtime field configuration.
///
/// Protocols that draw their field modulus from the transcript (the paper
/// Step-2 prime) build this once per relation and instantiate each proof's
/// [`PreparedConstraintMatrices`] with
/// [`PreparedConstraintMatrices::from_skeleton`]. That replaces the
/// per-proof clone, re-validation, selector re-detection, and `O(nnz)`
/// re-digest of the sparse matrices with a shared-ownership handle and an
/// `O(log nnz)` digest replay — while producing bit-identical prepared
/// statements (the digest included) for every accepted configuration.
#[derive(Clone, Debug)]
pub struct ConstraintMatricesSkeleton<F, C = F>
where
    F: SpartanField,
{
    matrices: Arc<ConstraintMatrices<C>>,
    digest_stream: SplicedStreamDigest,
    num_row_vars: usize,
    num_column_vars: usize,
    selector_triplet: Option<DisjointUnitSelectorTriplet>,
    block_selector: Option<Arc<BlockSelectorLayout<C>>>,
    _field: PhantomData<fn() -> F>,
}

impl<F, C> ConstraintMatricesSkeleton<F, C>
where
    F: SpartanField,
    C: ModulusIndependentCoefficient<F>,
{
    /// Validates the matrices and caches the modulus-independent part of the
    /// statement digest.
    pub fn new(matrices: ConstraintMatrices<C>) -> Result<Self, SpartanMatrixError> {
        let num_row_vars = padded_num_vars(matrices.row_count())?;
        let num_column_vars = padded_num_vars(matrices.column_count())?;

        // The digest stream is q-dependent only inside the modulus-encoding
        // hole right after the domain tag and length prefix; every accepted
        // configuration writes the same fixed-width encoding there.
        let modulus_width = F::canonical_encoding_width();
        let hole_start = CONSTRAINT_MATRIX_DIGEST_DOMAIN
            .len()
            .checked_add(8)
            .ok_or(SpartanMatrixError::DomainTooLarge)?;
        let hole_end = hole_start
            .checked_add(modulus_width)
            .ok_or(SpartanMatrixError::DomainTooLarge)?;
        let mut builder = SplicedDigestBuilder::new(hole_start..hole_end);
        stream_constraint_matrix_digest_bytes(&matrices, modulus_width, &mut builder)?;
        let digest_stream = builder
            .finish()
            .map_err(|_| SpartanMatrixError::DomainTooLarge)?;

        let selector_triplet =
            detect_disjoint_unit_selector_triplet_with(&matrices, |coefficient: &C| {
                coefficient.is_unit()
            });
        // Modulus-independent coefficient equality: identical encodings under
        // every accepted configuration.
        let mut left_encoding = Vec::new();
        let mut right_encoding = Vec::new();
        let block_selector =
            detect_block_selector_layout_with(&matrices, num_column_vars, |left: &C, right: &C| {
                left_encoding.clear();
                right_encoding.clear();
                left.write_modulus_independent_encoding(&mut left_encoding);
                right.write_modulus_independent_encoding(&mut right_encoding);
                left_encoding == right_encoding
            })
            .map(Arc::new);
        Ok(Self {
            matrices: Arc::new(matrices),
            digest_stream,
            num_row_vars,
            num_column_vars,
            selector_triplet,
            block_selector,
            _field: PhantomData,
        })
    }

    /// Validated `A`, `B`, and `C` matrices.
    pub fn matrices(&self) -> &ConstraintMatrices<C> {
        &self.matrices
    }
}

#[derive(Clone, Debug)]
pub struct PreparedConstraintMatrices<F, C = F>
where
    F: SpartanField,
{
    matrices: Arc<ConstraintMatrices<C>>,
    field_config: F::Config,
    field_modulus_encoding: Vec<u8>,
    digest: [u8; 32],
    num_row_vars: usize,
    num_column_vars: usize,
    selector_triplet: Option<DisjointUnitSelectorTriplet>,
    block_selector: Option<Arc<BlockSelectorLayout<C>>>,
}

impl<F, C> PreparedConstraintMatrices<F, C>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    /// Validates the coefficient field, computes padded domain widths, and
    /// commits to the complete public statement with BLAKE3.
    pub fn new(
        matrices: ConstraintMatrices<C>,
        field_config: &F::Config,
    ) -> Result<Self, SpartanMatrixError> {
        F::validate_config(field_config)?;
        let num_row_vars = padded_num_vars(matrices.row_count())?;
        let num_column_vars = padded_num_vars(matrices.column_count())?;
        let field_modulus_encoding = F::canonical_modulus_encoding(field_config);
        let digest = constraint_matrix_digest(&matrices, field_config, &field_modulus_encoding)?;
        let selector_triplet =
            detect_disjoint_unit_selector_triplet::<F, C>(&matrices, field_config);
        let field_one_encoding =
            F::one_with_cfg(field_config).canonical_element_encoding(field_config);
        let block_selector =
            detect_block_selector_layout_with(&matrices, num_column_vars, |left: &C, right: &C| {
                left.canonical_field_encoding(field_config, &field_one_encoding)
                    == right.canonical_field_encoding(field_config, &field_one_encoding)
            })
            .map(Arc::new);

        Ok(Self {
            matrices: Arc::new(matrices),
            field_config: field_config.clone(),
            field_modulus_encoding,
            digest,
            num_row_vars,
            num_column_vars,
            selector_triplet,
            block_selector,
        })
    }

    /// Instantiates the prepared statement for one runtime field
    /// configuration from a modulus-independent skeleton, sharing the
    /// skeleton's validated matrices instead of cloning them.
    ///
    /// This is bit-identical to [`Self::new`] on the same matrices and
    /// configuration — the statement digest included, by splicing the
    /// configuration's modulus encoding into the cached digest stream — but
    /// runs in `O(log nnz)` instead of `O(nnz)`.
    pub fn from_skeleton(
        skeleton: &ConstraintMatricesSkeleton<F, C>,
        field_config: &F::Config,
    ) -> Result<Self, SpartanMatrixError>
    where
        C: ModulusIndependentCoefficient<F>,
    {
        F::validate_config(field_config)?;
        let field_modulus_encoding = F::canonical_modulus_encoding(field_config);
        let digest = skeleton
            .digest_stream
            .digest_with(&field_modulus_encoding)
            .map_err(|_| SpartanMatrixError::FieldConfigurationMismatch)?;

        Ok(Self {
            matrices: Arc::clone(&skeleton.matrices),
            field_config: field_config.clone(),
            field_modulus_encoding,
            digest,
            num_row_vars: skeleton.num_row_vars,
            num_column_vars: skeleton.num_column_vars,
            selector_triplet: skeleton.selector_triplet,
            block_selector: skeleton.block_selector.clone(),
        })
    }

    /// Validated `A`, `B`, and `C` matrices.
    pub fn matrices(&self) -> &ConstraintMatrices<C> {
        &self.matrices
    }

    /// Runtime field configuration used by every prepared operation.
    pub const fn config(&self) -> &F::Config {
        &self.field_config
    }

    /// Canonical encoding of the runtime modulus bound into [`Self::digest`].
    pub fn field_modulus_encoding(&self) -> &[u8] {
        &self.field_modulus_encoding
    }

    /// BLAKE3 digest of the field identity, shape, sparse topology, and values.
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Number of variables in the padded row domain.
    pub const fn num_row_vars(&self) -> usize {
        self.num_row_vars
    }

    /// Number of variables in the padded column domain.
    pub const fn num_column_vars(&self) -> usize {
        self.num_column_vars
    }

    pub(crate) fn binding<'a>(
        &'a self,
        rho: &'a F,
    ) -> crate::sumcheck::bridge::dense::DenseBinding<'a, F, C> {
        crate::sumcheck::bridge::dense::DenseBinding::new(self, rho)
    }
    pub(crate) fn structured(
        &self,
    ) -> crate::sumcheck::bridge::structured::StructuredBinding<'_, F, C> {
        crate::sumcheck::bridge::structured::StructuredBinding::new(self)
    }
    /// The detected disjoint unit-selector layout as
    /// `[rows, a_offset, b_offset, c_offset]`, for the raw prover-side
    /// binding kernel; `None` for every other matrix layout.
    pub(crate) fn selector_layout(&self) -> Option<[usize; 4]> {
        self.selector_triplet.map(|layout| {
            [
                layout.rows,
                layout.a_offset,
                layout.b_offset,
                layout.c_offset,
            ]
        })
    }

    /// The detected block-selector layout, if every matrix is a union of
    /// block-aligned selector runs (see [`BlockSelectorLayout`]); for the raw
    /// prover-side structured inner sumcheck.
    pub(crate) fn block_selector(&self) -> Option<&BlockSelectorLayout<C>> {
        self.block_selector.as_deref()
    }
}

/// Pads a complete field-valued assignment to the prepared column domain.
///
/// `assignment[0]` must be the R1CS constant one. Values retain their logical
/// order and all padding is appended, so the table uses little-endian index
/// order without a permutation.
pub fn build_assignment_mle<F>(
    assignment: &[F],
    expected_columns: usize,
    field_config: &F::Config,
) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError>
where
    F: SpartanField,
{
    if assignment.len() != expected_columns {
        return Err(SpartanMatrixError::InvalidAssignmentLength {
            expected: expected_columns,
            actual: assignment.len(),
        });
    }
    let modulus_encoding = F::canonical_modulus_encoding(field_config);
    validate_elements_field(assignment, &modulus_encoding)?;
    let one = F::one_with_cfg(field_config);
    if assignment.first() != Some(&one) {
        return Err(SpartanMatrixError::InvalidAssignmentConstant);
    }

    padded_mle(assignment, field_config)
}

/// Convenience conversion for a Bit assignment beginning with `true`.
pub fn build_boolean_assignment_mle<F>(
    assignment: &[bool],
    expected_columns: usize,
    field_config: &F::Config,
) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError>
where
    F: SpartanField,
{
    if assignment.len() != expected_columns {
        return Err(SpartanMatrixError::InvalidAssignmentLength {
            expected: expected_columns,
            actual: assignment.len(),
        });
    }

    let zero = F::zero_with_cfg(field_config);
    let one = F::one_with_cfg(field_config);
    let field_assignment: Vec<_> = assignment
        .iter()
        .map(|bit| if *bit { one.clone() } else { zero.clone() })
        .collect();
    build_assignment_mle(&field_assignment, expected_columns, field_config)
}

/// Checks and pads the three R1CS product tables to one row domain.
pub fn build_product_mles<F>(
    az: &[F],
    bz: &[F],
    cz: &[F],
    expected_rows: usize,
    field_config: &F::Config,
) -> Result<R1csProductMles<F>, SpartanMatrixError>
where
    F: SpartanField,
{
    for actual in [az.len(), bz.len(), cz.len()] {
        if actual != expected_rows {
            return Err(SpartanMatrixError::InvalidProductLength {
                expected: expected_rows,
                actual,
            });
        }
    }

    let modulus_encoding = F::canonical_modulus_encoding(field_config);
    validate_elements_field(az, &modulus_encoding)?;
    validate_elements_field(bz, &modulus_encoding)?;
    validate_elements_field(cz, &modulus_encoding)?;

    Ok(R1csProductMles {
        az: padded_mle(az, field_config)?,
        bz: padded_mle(bz, field_config)?,
        cz: padded_mle(cz, field_config)?,
    })
}

/// Evaluates the multilinear equality polynomial
///
/// `eq(left, right) = product_i(left_i right_i + (1-left_i)(1-right_i))`.
pub fn eq_eval<F>(
    left: &[F],
    right: &[F],
    field_config: &F::Config,
) -> Result<F, SpartanMatrixError>
where
    F: SpartanField,
{
    if left.len() != right.len() {
        return Err(SpartanMatrixError::InvalidMleOperation);
    }
    let modulus_encoding = F::canonical_modulus_encoding(field_config);
    validate_elements_field(left, &modulus_encoding)?;
    validate_elements_field(right, &modulus_encoding)?;

    let one = F::one_with_cfg(field_config);
    let mut value = one.clone();
    for (left_i, right_i) in left.iter().zip(right) {
        // (1 - left_i) + right_i (2 left_i - 1)
        let mut twice_left_minus_one = left_i.clone();
        twice_left_minus_one = field_config.add(&(twice_left_minus_one), &(left_i));
        twice_left_minus_one = field_config.sub(&(twice_left_minus_one), &(&one));
        twice_left_minus_one = field_config.mul(&(twice_left_minus_one), &(right_i));

        let mut coordinate = one.clone();
        coordinate = field_config.sub(&(coordinate), &(left_i));
        coordinate = field_config.add(&(coordinate), &(&twice_left_minus_one));
        value = field_config.mul(&(value), &(&coordinate));
    }
    Ok(value)
}

/// Builds `eq(boolean_index, point)` in exact little-endian index order.
///
/// Coordinate `i` corresponds to bit `i` of the table index. The empty point
/// therefore has the one-entry table `[1]`.
pub fn eq_table<F>(point: &[F], field_config: &F::Config) -> Result<Vec<F>, SpartanMatrixError>
where
    F: SpartanField,
{
    let modulus_encoding = F::canonical_modulus_encoding(field_config);
    validate_elements_field(point, &modulus_encoding)?;
    let table_len = domain_size(point.len())?;
    let zero = F::zero_with_cfg(field_config);
    let mut table = vec![zero; table_len];
    table[0] = F::one_with_cfg(field_config);

    for (coordinate, challenge) in point.iter().enumerate() {
        let half = domain_size(coordinate)?;
        let (zero_children, one_children) = table[..2 * half].split_at_mut(half);
        for (zero_child, one_child) in zero_children.iter_mut().zip(one_children) {
            let parent = zero_child.clone();
            *one_child = (field_config).mul(&parent, challenge);
            *zero_child = (field_config).sub(&parent, one_child);
        }
    }

    Ok(table)
}

/// PROVER-side [`eq_table`] with the per-level doubling parallelized
/// (each level's pair expansions are independent; the products — and
/// therefore the table — are identical to the sequential build). The
/// verifier's evaluation path keeps the plain [`eq_table`].
pub(crate) fn eq_table_prover<F>(
    point: &[F],
    field_config: &F::Config,
) -> Result<Vec<F>, SpartanMatrixError>
where
    F: SpartanField,
{
    let modulus_encoding = F::canonical_modulus_encoding(field_config);
    validate_elements_field(point, &modulus_encoding)?;
    let table_len = domain_size(point.len())?;
    let zero = F::zero_with_cfg(field_config);
    let mut table = vec![zero; table_len];
    table[0] = F::one_with_cfg(field_config);

    for (coordinate, challenge) in point.iter().enumerate() {
        let half = domain_size(coordinate)?;
        let (zero_children, one_children) = table[..2 * half].split_at_mut(half);
        let expand = |zero_child: &mut F, one_child: &mut F| {
            let parent = zero_child.clone();
            *one_child = (field_config).mul(&parent, challenge);
            *zero_child = (field_config).sub(&parent, one_child);
        };
        #[cfg(feature = "parallel")]
        if half >= (1 << 13) && rayon::current_num_threads() > 1 {
            zero_children
                .par_iter_mut()
                .zip(one_children.par_iter_mut())
                .for_each(|(zero_child, one_child)| expand(zero_child, one_child));
            continue;
        }
        for (zero_child, one_child) in zero_children.iter_mut().zip(one_children) {
            expand(zero_child, one_child);
        }
    }

    Ok(table)
}

/// Materializes the low- and high-coordinate equality factors used by the
/// factored outer sumcheck.
///
/// The low, earlier coordinates are always returned first. Thus the tensor
/// product of the two tables agrees with [`eq_table`]'s little-endian order.
pub fn make_equality_factors<F>(
    point: &[F],
    field_config: &F::Config,
) -> Result<(DenseMultilinearExtension<F>, DenseMultilinearExtension<F>), SpartanMatrixError>
where
    F: SpartanField,
{
    let split = point.len() / 2;
    let (low, high) = point.split_at(split);
    let low = DenseMultilinearExtension {
        evaluations: eq_table(low, field_config)?,
        num_vars: low.len(),
    };
    let high = DenseMultilinearExtension {
        evaluations: eq_table(high, field_config)?,
        num_vars: high.len(),
    };
    Ok((low, high))
}

/// A scaled MLE evaluation claim `scale * polynomial(point) = value`.
///
/// The scale is retained instead of divided away because it may be zero.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaledMleEvaluationClaim<F> {
    point: Box<[F]>,
    scale: F,
    value: F,
}

impl<F> ScaledMleEvaluationClaim<F>
where
    F: SpartanField,
{
    /// Constructs a scaled claim. Structural checks occur during verification.
    pub fn new(point: Box<[F]>, scale: F, value: F) -> Self {
        Self {
            point,
            scale,
            value,
        }
    }

    /// Claimed evaluation point, in low-coordinate-first order.
    pub fn point(&self) -> &[F] {
        &self.point
    }

    /// Multiplicative scale on the polynomial evaluation.
    pub const fn scale(&self) -> &F {
        &self.scale
    }

    /// Claimed scaled value.
    pub const fn value(&self) -> &F {
        &self.value
    }

    /// Directly discharges the claim against a complete evaluation table.
    ///
    /// This is the witness-aware path used until the assignment opening is
    /// connected to the BitZ PCS.
    pub fn nonsuccinct_verify(
        &self,
        polynomial: &DenseMultilinearExtension<F>,
        field_config: &F::Config,
    ) -> Result<(), MleClaimError> {
        let modulus_encoding = F::canonical_modulus_encoding(field_config);
        validate_element_field(&self.scale, &modulus_encoding)?;
        validate_element_field(&self.value, &modulus_encoding)?;
        let evaluation = evaluate_mle(polynomial, &self.point, field_config)?;
        if (field_config).mul(&self.scale, &evaluation) != self.value {
            return Err(MleClaimError::InvalidEvaluation);
        }
        Ok(())
    }
}

/// `Σ_{i < rows} W(i) · eq(point, i)` for the product functional `W` over
/// `n = point.len()` row variables (`rows ≤ 2^n`), in
/// `O(2^skip_vars + n)` field operations.
///
/// With `K = skip_vars`, `M = 2^K`, `rows = Q·M + R` (`R < M`) and
/// `i = s + M·x`, both `W(i)` and `eq(point, i)` split along the prefix
/// `s` and the tail `x`, so
///
/// `Σ_{i < rows} W(i) eq(point, i)
///   = Σ_{x < Q} eq(t, x) eq(y_hi, x) · Σ_{s < M} prefix[s] eq(y_lo, s)
///   + eq(t, Q) eq(y_hi, Q) · Σ_{s < R} prefix[s] eq(y_lo, s)`
///
/// where `y_lo = point[..K]`, `y_hi = point[K..]`, `t = tail_point`, and the
/// second term is present only for `R > 0` (then `Q < 2^{n − K}` is a
/// Bit index of the tail domain).
pub(crate) fn product_row_prefix_sum<F>(
    functional: &ProductRowFunctional<'_, F>,
    rows: usize,
    point: &[F],
    field_config: &F::Config,
) -> Result<F, SpartanMatrixError>
where
    F: SpartanField,
{
    let skip_vars = functional.skip_vars;
    let block_len = domain_size(skip_vars)?;
    let tail_vars = point
        .len()
        .checked_sub(skip_vars)
        .ok_or(SpartanMatrixError::InvalidMleOperation)?;
    if functional.prefix.len() != block_len
        || functional.tail_point.len() != tail_vars
        || rows > domain_size(point.len())?
    {
        return Err(SpartanMatrixError::InvalidMleOperation);
    }
    let (low_point, high_point) = point.split_at(skip_vars);
    let low_table = eq_table(low_point, field_config)?;
    let zero = F::zero_with_cfg(field_config);
    let prefix_sum = |count: usize| -> F {
        let mut sum = zero.clone();
        for (weight, equality) in functional.prefix[..count].iter().zip(&low_table[..count]) {
            sum = field_config.add(&(sum), &(&(field_config).mul(weight, equality)));
        }
        sum
    };

    let full_blocks = rows >> skip_vars;
    let partial_rows = rows & (block_len - 1);
    let mut sum = (field_config).mul(
        &hypercube_prefix_sum(functional.tail_point, high_point, full_blocks, field_config)?,
        &prefix_sum(block_len),
    );
    if partial_rows != 0 {
        let block_weight = (field_config).mul(
            &eq_at_boolean_index(functional.tail_point, full_blocks, field_config)?,
            &eq_at_boolean_index(high_point, full_blocks, field_config)?,
        );
        sum = field_config.add(
            &(sum),
            &(&(field_config).mul(&block_weight, &prefix_sum(partial_rows))),
        );
    }
    Ok(sum)
}

/// `Σ_{x < count} eq(left, x) · eq(right, x)` over the `m`-variable Bit
/// hypercube (`m = left.len()`, `count ≤ 2^m`), in `O(m)` field operations.
///
/// For `count = 2^m` the sum is `eq(left, right)`. Otherwise every `x < count`
/// agrees with `count` above some bit `k` at which `count` is 1 and `x` is 0,
/// and is free below `k`:
///
/// `Σ_{k : count_k = 1} Π_{j > k} m_j(count_j) · (1 − l_k)(1 − r_k) · Π_{j < k} (l_j r_j + (1 − l_j)(1 − r_j))`
///
/// with `m_j(1) = l_j r_j` and `m_j(0) = (1 − l_j)(1 − r_j)`.
fn hypercube_prefix_sum<F>(
    left: &[F],
    right: &[F],
    count: usize,
    field_config: &F::Config,
) -> Result<F, SpartanMatrixError>
where
    F: SpartanField,
{
    let vars = left.len();
    let domain = domain_size(vars)?;
    if right.len() != vars || count > domain {
        return Err(SpartanMatrixError::InvalidMleOperation);
    }
    let one = F::one_with_cfg(field_config);
    let match_one = left
        .iter()
        .zip(right)
        .map(|(l, r)| (field_config).mul(l, r))
        .collect::<Vec<_>>();
    let match_zero = left
        .iter()
        .zip(right)
        .map(|(l, r)| {
            (field_config).mul(&(field_config).sub(&one, l), &(field_config).sub(&one, r))
        })
        .collect::<Vec<_>>();
    if count == domain {
        let mut value = one;
        for (m1, m0) in match_one.iter().zip(&match_zero) {
            value = (field_config).mul(&value, &(field_config).add(m1, m0));
        }
        return Ok(value);
    }

    // free_prefix[k] = Π_{j < k} (m_j(1) + m_j(0)): the bits below `k` are free.
    let mut free_prefix = Vec::with_capacity(vars + 1);
    free_prefix.push(one.clone());
    for (m1, m0) in match_one.iter().zip(&match_zero) {
        let last = free_prefix.last().expect("seeded with one");
        free_prefix.push((field_config).mul(last, &(field_config).add(m1, m0)));
    }
    let mut sum = F::zero_with_cfg(field_config);
    let mut high_match = one;
    for k in (0..vars).rev() {
        if (count >> k) & 1 == 1 {
            sum = field_config.add(
                &(sum),
                &(&(field_config).mul(
                    &(field_config).mul(&high_match, &match_zero[k]),
                    &free_prefix[k],
                )),
            );
            high_match = (field_config).mul(&high_match, &match_one[k]);
        } else {
            high_match = (field_config).mul(&high_match, &match_zero[k]);
        }
    }
    Ok(sum)
}

/// `eq(point, index) = Π_k (index_k ? point_k : 1 − point_k)`: entry `index`
/// of [`eq_table`] without the table (`index < 2^{point.len()}`).
pub(crate) fn eq_at_boolean_index<F>(
    point: &[F],
    index: usize,
    field_config: &F::Config,
) -> Result<F, SpartanMatrixError>
where
    F: SpartanField,
{
    if index >= domain_size(point.len())? {
        return Err(SpartanMatrixError::InvalidMleOperation);
    }
    let one = F::one_with_cfg(field_config);
    let mut value = one.clone();
    for (k, coordinate) in point.iter().enumerate() {
        let factor = if (index >> k) & 1 == 1 {
            coordinate.clone()
        } else {
            (field_config).sub(&one, coordinate)
        };
        value = (field_config).mul(&value, &factor);
    }
    Ok(value)
}

fn padded_mle<F>(
    values: &[F],
    field_config: &F::Config,
) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError>
where
    F: SpartanField,
{
    let num_vars = padded_num_vars(values.len())?;
    let mut evaluations = values.to_vec();
    evaluations.resize(domain_size(num_vars)?, F::zero_with_cfg(field_config));
    Ok(DenseMultilinearExtension {
        evaluations,
        num_vars,
    })
}

fn evaluate_mle<F>(
    polynomial: &DenseMultilinearExtension<F>,
    point: &[F],
    field_config: &F::Config,
) -> Result<F, SpartanMatrixError>
where
    F: SpartanField,
{
    if point.len() != polynomial.num_vars {
        return Err(SpartanMatrixError::InvalidColumnPointLength {
            expected: polynomial.num_vars,
            actual: point.len(),
        });
    }
    let expected_evaluations = domain_size(polynomial.num_vars)?;
    if polynomial.evaluations.len() != expected_evaluations {
        return Err(SpartanMatrixError::InvalidMleOperation);
    }

    let modulus_encoding = F::canonical_modulus_encoding(field_config);
    validate_elements_field(point, &modulus_encoding)?;
    validate_elements_field(&polynomial.evaluations, &modulus_encoding)?;

    let mut evaluations = polynomial.evaluations.clone();
    let mut active_len = evaluations.len();
    for challenge in point {
        let next_len = active_len / 2;
        for index in 0..next_len {
            let left = evaluations[2 * index].clone();
            let difference = (field_config).sub(&evaluations[2 * index + 1], &left);
            evaluations[index] =
                (field_config).add(&left, &(field_config).mul(challenge, &difference));
        }
        active_len = next_len;
    }

    evaluations
        .into_iter()
        .next()
        .ok_or(SpartanMatrixError::InvalidMleOperation)
}

fn validate_matrix_coefficients<F, C>(
    matrix_name: &'static str,
    rows: &circuit::linear_map::CsrMatrix<Box<[&C]>>,
    modulus_encoding: &[u8],
) -> Result<(), SpartanMatrixError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    for (row_index, row) in rows.rows().enumerate() {
        for (column, coefficient) in row {
            coefficient.validate(modulus_encoding)?;
            if coefficient.is_zero() {
                return Err(SpartanMatrixError::ExplicitZeroCoefficient {
                    matrix: matrix_name,
                    row: row_index,
                    column,
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_elements_field<F>(
    values: &[F],
    modulus_encoding: &[u8],
) -> Result<(), SpartanMatrixError>
where
    F: SpartanField,
{
    for value in values {
        validate_element_field(value, modulus_encoding)?;
    }
    Ok(())
}

pub(crate) fn validate_element_field<F>(
    value: &F,
    modulus_encoding: &[u8],
) -> Result<(), SpartanMatrixError>
where
    F: SpartanField,
{
    value.validate_element(modulus_encoding)?;
    Ok(())
}

/// Detects the [`BlockSelectorLayout`] of `matrices`, with `same` deciding
/// coefficient equality. Runs in `O(columns + nnz)`; the topological checks
/// fail fast on general matrices before any coefficient comparison.
fn detect_block_selector_layout_with<C: Clone>(
    matrices: &ConstraintMatrices<C>,
    num_column_vars: usize,
    mut same: impl FnMut(&C, &C) -> bool,
) -> Option<BlockSelectorLayout<C>> {
    let rows = matrices.row_count();
    let domain = domain_size(num_column_vars).ok()?;
    if rows == 0 || rows > domain {
        return None;
    }

    let mut runs_of = |matrix: &CscMatrix<Box<[C]>>| -> Option<Vec<SelectorRun<C>>> {
        if !matrix.nnz().is_multiple_of(rows) {
            return None;
        }
        let offsets = matrix.column_offsets();
        let row_indices = matrix.row_indices();
        let coefficients = matrix.coefficients();
        let columns = matrix.column_count();
        let mut runs = Vec::with_capacity(matrix.nnz() / rows);
        let mut column = 0;
        while column < columns {
            if offsets[column + 1] == offsets[column] {
                column += 1;
                continue;
            }
            let start = column;
            if start.checked_add(rows)? > columns {
                return None;
            }
            let base = offsets[start];
            let coefficient = &coefficients[base];
            for row in 0..rows {
                if offsets[start + row] != base + row
                    || offsets[start + row + 1] != base + row + 1
                    || row_indices[base + row] != row
                    || !same(coefficient, &coefficients[base + row])
                {
                    return None;
                }
            }
            runs.push(SelectorRun {
                start,
                coefficient: coefficient.clone(),
            });
            column = start + rows;
        }
        Some(runs)
    };
    let a = runs_of(matrices.a())?;
    let b = runs_of(matrices.b())?;
    let c = runs_of(matrices.c())?;
    if a.is_empty() && b.is_empty() && c.is_empty() {
        return None;
    }

    // The block length is the largest power of two dividing every run start
    // (and the domain); every run must fit inside its block.
    let mut block_len = domain;
    for run in a.iter().chain(&b).chain(&c) {
        if run.start != 0 {
            block_len = block_len.min(run.start & run.start.wrapping_neg());
        }
    }
    if block_len < rows {
        return None;
    }
    debug_assert!(block_len.is_power_of_two());
    Some(BlockSelectorLayout {
        rows,
        block_len,
        a,
        b,
        c,
    })
}

fn detect_disjoint_unit_selector_triplet<F, C>(
    matrices: &ConstraintMatrices<C>,
    field_config: &F::Config,
) -> Option<DisjointUnitSelectorTriplet>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    let field_one_encoding = F::one_with_cfg(field_config).canonical_element_encoding(field_config);
    detect_disjoint_unit_selector_triplet_with(matrices, |coefficient: &C| {
        coefficient
            .canonical_field_encoding(field_config, &field_one_encoding)
            .as_ref()
            == field_one_encoding
    })
}

fn detect_disjoint_unit_selector_triplet_with<C>(
    matrices: &ConstraintMatrices<C>,
    is_unit: impl Fn(&C) -> bool + Copy,
) -> Option<DisjointUnitSelectorTriplet> {
    let rows = matrices.row_count();
    let columns = matrices.column_count();
    let a_offset = contiguous_unit_selector_offset_with(matrices.a(), is_unit)?;
    let b_offset = contiguous_unit_selector_offset_with(matrices.b(), is_unit)?;
    let c_offset = contiguous_unit_selector_offset_with(matrices.c(), is_unit)?;

    let a_end = a_offset.checked_add(rows)?;
    let b_end = b_offset.checked_add(rows)?;
    let c_end = c_offset.checked_add(rows)?;
    if a_end > b_offset || b_end > c_offset || c_end > columns {
        return None;
    }

    Some(DisjointUnitSelectorTriplet {
        rows,
        a_offset,
        b_offset,
        c_offset,
    })
}

fn contiguous_unit_selector_offset_with<C>(
    matrix: &CscMatrix<Box<[C]>>,
    is_unit: impl Fn(&C) -> bool,
) -> Option<usize> {
    let rows = matrix.row_count();
    if matrix.nnz() != rows {
        return None;
    }

    // The first nonzero CSC boundary immediately follows the first occupied
    // column. `partition_point` avoids scanning the potentially enormous
    // empty prefix of selector matrices.
    let first_nonzero_boundary = matrix
        .column_offsets()
        .partition_point(|entry_offset| *entry_offset == 0);
    let offset = first_nonzero_boundary.checked_sub(1)?;
    if offset.checked_add(rows)? > matrix.column_count() {
        return None;
    }

    for row in 0..rows {
        if matrix.column_offsets()[offset + row] != row
            || matrix.column_offsets()[offset + row + 1] != row + 1
        {
            return None;
        }
        let entry_row = matrix.row_indices()[row];
        let coefficient = &matrix.coefficients()[row];
        if entry_row != row || !is_unit(coefficient) {
            return None;
        }
    }

    Some(offset)
}

fn constraint_matrix_digest<F, C>(
    matrices: &ConstraintMatrices<C>,
    field_config: &F::Config,
    field_modulus_encoding: &[u8],
) -> Result<[u8; 32], SpartanMatrixError>
where
    F: SpartanField,
    C: SpartanMatrixCoefficient<F>,
{
    let mut hash = Hasher::new();
    hash.update(CONSTRAINT_MATRIX_DIGEST_DOMAIN);
    hash_bytes(&mut hash, field_modulus_encoding)?;
    hash_usize(&mut hash, matrices.row_count())?;
    hash_usize(&mut hash, matrices.column_count())?;
    let field_one_encoding = F::one_with_cfg(field_config).canonical_element_encoding(field_config);

    for (matrix_name, label, matrix) in [
        ("A", b'A', matrices.a()),
        ("B", b'B', matrices.b()),
        ("C", b'C', matrices.c()),
    ] {
        // Construct and drop one flat row-major view at a time. Preparation
        // therefore uses O(rows + nnz(matrix)) auxiliary memory, not three
        // nested row-vector transposes held simultaneously.
        let rows = matrix.borrowed_csr()?;
        validate_matrix_coefficients::<F, C>(matrix_name, &rows, field_modulus_encoding)?;
        hash.update(&[label]);
        for row in rows.rows() {
            hash_usize(&mut hash, row.len())?;
            for (column, coefficient) in row {
                hash_usize(&mut hash, column)?;
                let encoding =
                    coefficient.canonical_field_encoding(field_config, &field_one_encoding);
                hash_bytes(&mut hash, encoding.as_ref())?;
            }
        }
    }

    Ok(*hash.finalize().as_bytes())
}

/// Streams the exact byte sequence [`constraint_matrix_digest`] hashes into
/// `builder`, with `modulus_width` placeholder zeros in the modulus-encoding
/// hole, using each coefficient's modulus-independent encoding.
///
/// Mirrors [`constraint_matrix_digest`]'s bytes and its validation: explicit
/// zeros are rejected with the identical row-major error coordinates, and by
/// the [`ModulusIndependentCoefficient`] contract per-entry validation cannot
/// fail for any accepted configuration.
fn stream_constraint_matrix_digest_bytes<F, C>(
    matrices: &ConstraintMatrices<C>,
    modulus_width: usize,
    builder: &mut SplicedDigestBuilder,
) -> Result<(), SpartanMatrixError>
where
    F: SpartanField,
    C: ModulusIndependentCoefficient<F>,
{
    let push_usize =
        |builder: &mut SplicedDigestBuilder, value: usize| -> Result<(), SpartanMatrixError> {
            let encoded = u64::try_from(value).map_err(|_| SpartanMatrixError::DomainTooLarge)?;
            builder.push(&encoded.to_le_bytes());
            Ok(())
        };

    builder.push(CONSTRAINT_MATRIX_DIGEST_DOMAIN);
    push_usize(builder, modulus_width)?;
    builder.push(&vec![0; modulus_width]);
    push_usize(builder, matrices.row_count())?;
    push_usize(builder, matrices.column_count())?;

    let mut encoding = Vec::new();
    for (matrix_name, label, matrix) in [
        ("A", b'A', matrices.a()),
        ("B", b'B', matrices.b()),
        ("C", b'C', matrices.c()),
    ] {
        let rows = matrix.borrowed_csr()?;
        builder.push(&[label]);
        for (row_index, row) in rows.rows().enumerate() {
            push_usize(builder, row.len())?;
            for (column, coefficient) in row {
                if coefficient.is_zero() {
                    return Err(SpartanMatrixError::ExplicitZeroCoefficient {
                        matrix: matrix_name,
                        row: row_index,
                        column,
                    });
                }
                push_usize(builder, column)?;
                encoding.clear();
                coefficient.write_modulus_independent_encoding(&mut encoding);
                push_usize(builder, encoding.len())?;
                builder.push(&encoding);
            }
        }
    }
    Ok(())
}

fn hash_bytes(hash: &mut Hasher, bytes: &[u8]) -> Result<(), SpartanMatrixError> {
    hash_usize(hash, bytes.len())?;
    hash.update(bytes);
    Ok(())
}

fn hash_usize(hash: &mut Hasher, value: usize) -> Result<(), SpartanMatrixError> {
    let encoded = u64::try_from(value).map_err(|_| SpartanMatrixError::DomainTooLarge)?;
    hash.update(&encoded.to_le_bytes());
    Ok(())
}

fn padded_num_vars(logical_len: usize) -> Result<usize, SpartanMatrixError> {
    logical_len
        .max(1)
        .checked_next_power_of_two()
        .map(|length| length.ilog2() as usize)
        .ok_or(SpartanMatrixError::DomainTooLarge)
}

pub(crate) fn domain_size(num_vars: usize) -> Result<usize, SpartanMatrixError> {
    1usize
        .checked_shl(u32::try_from(num_vars).map_err(|_| SpartanMatrixError::DomainTooLarge)?)
        .ok_or(SpartanMatrixError::DomainTooLarge)
}

#[cfg(test)]
mod tests {

    use super::*;

    const TEST_MODULUS: u128 = (1_u128 << 100) - 15;
    const OTHER_TEST_MODULUS: u128 = (1_u128 << 127) - 1;

    fn config() -> <Fp<2> as crate::piop::spartan::SpartanField>::Config {
        Fp::<2>::make_cfg(&Uint::from(TEST_MODULUS)).expect("odd test modulus")
    }

    fn field(value: u64, config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config) -> Fp<2> {
        Fp::<2>::from_with_cfg(value, config)
    }

    #[test]
    fn montgomery_products_support_generic_coefficients_and_validate_assignment() {
        use super::super::raw_monty::raw_shared;
        let field = Fp::<2>::make_cfg(&Uint::from(crate::pcs::FQ_MOD)).unwrap();
        let columns = vec![
            vec![(0, true)],
            vec![(1, true)],
            vec![(2, true)],
            vec![],
            vec![(0, true)],
        ];
        let bool_matrix = CscMatrix::try_from_columns(3, columns.clone()).unwrap();
        let field_matrix = CscMatrix::try_from_columns(
            3,
            columns
                .into_iter()
                .map(|column| {
                    column
                        .into_iter()
                        .map(|(row, _)| (row, field.one()))
                        .collect()
                })
                .collect(),
        )
        .unwrap();
        let boolean = PreparedConstraintMatrices::<Fp<2>, bool>::new(
            ConstraintMatrices::new(bool_matrix.clone(), bool_matrix.clone(), bool_matrix).unwrap(),
            &field,
        )
        .unwrap();
        let ordinary = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(field_matrix.clone(), field_matrix.clone(), field_matrix)
                .unwrap(),
            &field,
        )
        .unwrap();
        let mut values = vec![0; 8];
        for (i, value) in values[..5].iter_mut().enumerate() {
            *value = raw_shared(field.from_integer(&Uint::<1>::from_u64((i + 1) as u64)));
        }
        let a = products_from_montgomery_assignment(&boolean, &values).unwrap();
        let b = products_from_montgomery_assignment(&ordinary, &values).unwrap();
        assert_eq!(a.ax, b.ax);
        assert_eq!(a.bx, b.bx);
        assert_eq!(a.cx, b.cx);
        assert!(matches!(
            products_from_montgomery_assignment(&boolean, &values[..7]),
            Err(super::super::SpartanError::InvalidAssignmentDimensions)
        ));
        values[7] = values[0];
        assert!(matches!(
            products_from_montgomery_assignment(&boolean, &values),
            Err(super::super::SpartanError::InvalidAssignmentPadding)
        ));
        values[7] = 0;
        values[0] = 0;
        assert!(matches!(
            products_from_montgomery_assignment(&boolean, &values),
            Err(super::super::SpartanError::Matrix(
                SpartanMatrixError::InvalidAssignmentConstant
            ))
        ));
        values[0] = raw_shared(field.one());
        values[1] = u128::from(*field.modulus());
        assert!(matches!(
            products_from_montgomery_assignment(&boolean, &values),
            Err(super::super::SpartanError::FieldConfigurationMismatch)
        ));
    }

    fn columns_from_rows<F: Clone>(
        column_count: usize,
        rows: &[Vec<(usize, F)>],
    ) -> Vec<Vec<(usize, F)>> {
        let mut columns: Vec<Vec<(usize, F)>> = (0..column_count).map(|_| Vec::new()).collect();
        for (row, entries) in rows.iter().enumerate() {
            for (column, coefficient) in entries {
                columns[*column].push((row, coefficient.clone()));
            }
        }
        columns
    }

    fn legacy_row_major_digest(
        rows: [&[Vec<(usize, Fp<2>)>]; 3],
        row_count: usize,
        column_count: usize,
        field_config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> [u8; 32] {
        let mut hash = Hasher::new();
        hash.update(b"bitz/spartan/constraint-matrices/v2");
        hash_bytes(
            &mut hash,
            &Fp::<2>::canonical_modulus_encoding(field_config),
        )
        .unwrap();
        hash_usize(&mut hash, row_count).unwrap();
        hash_usize(&mut hash, column_count).unwrap();

        for (label, matrix_rows) in [b'A', b'B', b'C'].into_iter().zip(rows) {
            hash.update(&[label]);
            for row in matrix_rows {
                hash_usize(&mut hash, row.len()).unwrap();
                for (column, coefficient) in row {
                    hash_usize(&mut hash, *column).unwrap();
                    hash_bytes(
                        &mut hash,
                        &coefficient.canonical_element_encoding(field_config),
                    )
                    .unwrap();
                }
            }
        }

        *hash.finalize().as_bytes()
    }

    #[test]
    fn row_constructor_materializes_flat_csc_and_retains_empty_shape() {
        let config = config();
        let two = field(2, &config);
        let three = field(3, &config);
        let five = field(5, &config);
        let seven = field(7, &config);
        let matrix = CscMatrix::<Box<[_]>>::try_from_rows(
            5,
            vec![
                vec![(0, two.clone()), (3, three.clone())],
                vec![],
                vec![(0, five.clone()), (2, seven.clone())],
                vec![],
            ],
        )
        .unwrap();

        assert_eq!(matrix.row_count(), 4);
        assert_eq!(matrix.column_count(), 5);
        assert_eq!(matrix.nnz(), 4);
        assert_eq!(matrix.columns().len(), 5);
        let column = matrix.column(0).unwrap();
        assert_eq!(column.indices(), &[0, 2]);
        assert_eq!(column.coefficients(), &[two, five]);
        assert!(matrix.column(1).unwrap().is_empty());
        let column = matrix.column(2).unwrap();
        assert_eq!(column.indices(), &[2]);
        assert_eq!(column.coefficients(), &[seven]);
        let column = matrix.column(3).unwrap();
        assert_eq!(column.indices(), &[0]);
        assert_eq!(column.coefficients(), &[three]);
        assert!(matrix.column(4).unwrap().is_empty());
        assert!(matrix.column(5).is_none());
        assert!(matrix.column(usize::MAX).is_none());
    }

    #[test]
    fn row_and_column_constructors_produce_identical_canonical_csc() {
        let config = config();
        let rows = vec![
            vec![(0, field(2, &config)), (3, field(3, &config))],
            vec![],
            vec![(0, field(5, &config)), (2, field(7, &config))],
            vec![],
        ];
        let from_rows = CscMatrix::<Box<[_]>>::try_from_rows(5, rows.clone()).unwrap();
        let from_columns =
            CscMatrix::<Box<[_]>>::try_from_columns(rows.len(), columns_from_rows(5, &rows))
                .unwrap();
        let from_csc = CscMatrix::<Box<[_]>>::try_from_csc(
            rows.len(),
            vec![0, 2, 2, 3, 4, 4],
            vec![
                (0, field(2, &config)),
                (2, field(5, &config)),
                (2, field(7, &config)),
                (0, field(3, &config)),
            ],
        )
        .unwrap();

        assert_eq!(from_rows, from_columns);
        assert_eq!(from_rows, from_csc);
    }

    #[test]
    fn constructors_reject_noncanonical_or_out_of_bounds_coordinates() {
        let config = config();
        let value = || field(1, &config);

        assert_eq!(
            CscMatrix::<Box<[_]>>::try_from_rows(2, vec![vec![(1, value()), (1, value())]]),
            Err(SparseMatrixError::ColumnsNotStrictlyIncreasing {
                row: 0,
                previous: 1,
                column: 1,
            })
        );
        assert_eq!(
            CscMatrix::<Box<[_]>>::try_from_rows(2, vec![vec![(2, value())]]),
            Err(SparseMatrixError::ColumnOutOfBounds {
                row: 0,
                column: 2,
                columns: 2,
            })
        );
        assert_eq!(
            CscMatrix::<Box<[_]>>::try_from_columns(
                2,
                vec![vec![(1, value()), (0, value())], vec![]]
            ),
            Err(SparseMatrixError::RowsNotStrictlyIncreasing {
                column: 0,
                previous: 1,
                row: 0,
            })
        );
        assert_eq!(
            CscMatrix::<Box<[_]>>::try_from_columns(2, vec![vec![], vec![(2, value())]]),
            Err(SparseMatrixError::RowOutOfBounds {
                column: 1,
                row: 2,
                rows: 2,
            })
        );
        assert_eq!(
            CscMatrix::<Box<[Fp<2>]>>::try_from_csc(2, vec![], vec![]),
            Err(SparseMatrixError::InvalidCscOffsets)
        );
        assert_eq!(
            CscMatrix::<Box<[Fp<2>]>>::try_from_csc(2, vec![1], vec![]),
            Err(SparseMatrixError::InvalidCscOffsets)
        );
        assert_eq!(
            CscMatrix::<Box<[_]>>::try_from_csc(2, vec![0, 2], vec![(0, value())]),
            Err(SparseMatrixError::InvalidCscOffsets)
        );
        assert_eq!(
            CscMatrix::<Box<[Fp<2>]>>::try_from_csc(2, vec![0, 1, 0], vec![]),
            Err(SparseMatrixError::InvalidCscOffsets)
        );
        assert_eq!(
            CscMatrix::<Box<[_]>>::try_from_csc(2, vec![0, 2], vec![(1, value()), (0, value())]),
            Err(SparseMatrixError::RowsNotStrictlyIncreasing {
                column: 0,
                previous: 1,
                row: 0,
            })
        );
        assert_eq!(
            CscMatrix::<Box<[_]>>::try_from_csc(2, vec![0, 1], vec![(2, value())]),
            Err(SparseMatrixError::RowOutOfBounds {
                column: 0,
                row: 2,
                rows: 2,
            })
        );
    }

    #[test]
    fn equality_tables_are_little_endian_and_split_low_coordinates_first() {
        let config = config();
        let r0 = field(2, &config);
        let r1 = field(7, &config);
        let one = Fp::<2>::one_with_cfg(&config);
        let table = eq_table(&[r0.clone(), r1.clone()], &config).unwrap();

        assert_eq!(
            table,
            vec![
                (config).mul(&(config).sub(&one, &r0), &(config).sub(&one, &r1)),
                (config).mul(&r0, &(config).sub(&one, &r1)),
                (config).mul(&(config).sub(&one, &r0), &r1),
                (config).mul(&r0, &r1),
            ]
        );

        let r2 = field(11, &config);
        let point = [r0, r1, r2];
        let (low, high) = make_equality_factors(&point, &config).unwrap();
        assert_eq!(low.num_vars, 1);
        assert_eq!(low.evaluations, eq_table(&point[..1], &config).unwrap());
        assert_eq!(high.num_vars, 2);
        assert_eq!(high.evaluations, eq_table(&point[1..], &config).unwrap());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn prover_equality_table_matches_sequential_at_parallel_threshold() {
        let config = config();
        let point: Vec<_> = (0..14)
            .map(|coordinate| field((coordinate + 2) as u64, &config))
            .collect();
        let expected = eq_table(&point, &config).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let actual = pool.install(|| eq_table_prover(&point, &config)).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn bound_table_and_independent_sparse_evaluation_agree() {
        let config = config();
        let a = CscMatrix::<Box<[_]>>::try_from_rows(
            3,
            vec![
                vec![(0, field(2, &config)), (2, field(3, &config))],
                vec![(1, field(5, &config))],
            ],
        )
        .unwrap();
        let b = CscMatrix::<Box<[_]>>::try_from_rows(
            3,
            vec![vec![(1, field(7, &config))], vec![(2, field(11, &config))]],
        )
        .unwrap();
        let c = CscMatrix::<Box<[_]>>::try_from_rows(
            3,
            vec![vec![(0, field(13, &config))], vec![(0, field(17, &config))]],
        )
        .unwrap();
        let prepared =
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), &config)
                .unwrap();
        let row_point = [field(19, &config)];
        let column_point = [field(23, &config), field(29, &config)];
        let rho = field(31, &config);

        let bound = crate::piop::spartan::matrix::eq_table_prover(&row_point, prepared.config())
            .map_err(crate::sumcheck::SumcheckError::from)
            .and_then(|weights| prepared.binding(&rho).bind_rows(&weights))
            .unwrap();
        let dense_evaluation = evaluate_mle(&bound, &column_point, &config).unwrap();
        let sparse_evaluation = prepared
            .structured()
            .evaluate_equality(&row_point, &rho, &column_point)
            .unwrap();
        assert_eq!(dense_evaluation, sparse_evaluation);
    }

    #[test]
    fn explicit_equality_row_weights_match_point_based_matrix_operations() {
        let config = config();
        let a = CscMatrix::<Box<[_]>>::try_from_rows(
            3,
            vec![
                vec![(0, field(2, &config)), (2, field(3, &config))],
                vec![(1, field(5, &config))],
                vec![(0, field(7, &config))],
            ],
        )
        .unwrap();
        let b = CscMatrix::<Box<[_]>>::try_from_rows(
            3,
            vec![
                vec![(1, field(11, &config))],
                vec![(2, field(13, &config))],
                vec![],
            ],
        )
        .unwrap();
        let c = CscMatrix::<Box<[_]>>::try_from_rows(
            3,
            vec![
                vec![(0, field(17, &config))],
                vec![],
                vec![(2, field(19, &config))],
            ],
        )
        .unwrap();
        let prepared =
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), &config)
                .unwrap();
        let row_point = [field(23, &config), field(29, &config)];
        let row_weights = eq_table(&row_point, &config).unwrap();
        let column_point = [field(31, &config), field(37, &config)];
        let rho = field(41, &config);

        assert_eq!(
            crate::piop::spartan::matrix::eq_table_prover(&row_point, prepared.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| prepared.binding(&rho).bind_rows(&weights))
                .unwrap(),
            prepared.binding(&rho).bind_rows(&row_weights).unwrap()
        );
        assert_eq!(
            prepared
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap(),
            prepared
                .binding(&rho)
                .evaluate_at(&row_weights, &column_point)
                .unwrap()
        );
    }

    fn from_skeleton_and_new_agree_semantically(
        matrices: &ConstraintMatrices<bool>,
        config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) {
        let skeleton = ConstraintMatricesSkeleton::<Fp<2>, bool>::new(matrices.clone()).unwrap();
        let from_skeleton =
            PreparedConstraintMatrices::<Fp<2>, bool>::from_skeleton(&skeleton, config).unwrap();
        let from_new =
            PreparedConstraintMatrices::<Fp<2>, bool>::new(matrices.clone(), config).unwrap();

        assert_eq!(
            from_skeleton.digest(),
            from_new.digest(),
            "the skeleton replay must reproduce the entry-wise digest exactly"
        );
        assert_eq!(from_skeleton.matrices(), from_new.matrices());
        assert_eq!(from_skeleton.num_row_vars(), from_new.num_row_vars());
        assert_eq!(from_skeleton.num_column_vars(), from_new.num_column_vars());
        assert_eq!(
            from_skeleton.field_modulus_encoding(),
            from_new.field_modulus_encoding()
        );
        // The two constructors detect the identical selector layout: for
        // Bit coefficients `is_unit` and the encoding comparison agree
        // under every valid configuration.
        match (from_skeleton.selector_triplet, from_new.selector_triplet) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.rows, b.rows);
                assert_eq!(a.a_offset, b.a_offset);
                assert_eq!(a.b_offset, b.b_offset);
                assert_eq!(a.c_offset, b.c_offset);
            }
            (a, b) => panic!("selector detection diverged: {a:?} vs {b:?}"),
        }

        let rho = field(47, config);
        let row_point: Vec<Fp<2>> = (0..from_new.num_row_vars())
            .map(|index| field(3 + index as u64, config))
            .collect();
        let column_point: Vec<Fp<2>> = (0..from_new.num_column_vars())
            .map(|index| field(29 + index as u64, config))
            .collect();
        assert_eq!(
            crate::piop::spartan::matrix::eq_table_prover(&row_point, from_skeleton.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| from_skeleton.binding(&rho).bind_rows(&weights))
                .unwrap(),
            crate::piop::spartan::matrix::eq_table_prover(&row_point, from_new.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| from_new.binding(&rho).bind_rows(&weights))
                .unwrap()
        );
        assert_eq!(
            from_skeleton
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap(),
            from_new
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap()
        );
    }

    #[test]
    fn skeleton_preparation_matches_direct_preparation() {
        let config = config();
        let rows = 6;
        let columns = 32;
        let selector = |offset: usize| {
            CscMatrix::<Box<[_]>>::try_from_rows(
                columns,
                (0..rows).map(|row| vec![(offset + row, true)]).collect(),
            )
            .unwrap()
        };
        let selectors = ConstraintMatrices::new(selector(8), selector(16), selector(24)).unwrap();
        from_skeleton_and_new_agree_semantically(&selectors, &config);

        let generic = |entries: Vec<Vec<(usize, bool)>>| {
            CscMatrix::<Box<[_]>>::try_from_rows(8, entries).unwrap()
        };
        let irregular = ConstraintMatrices::new(
            generic(vec![vec![(0, true), (7, true)], vec![(3, true)]]),
            generic(vec![vec![(1, true)], vec![(2, true), (5, true)]]),
            generic(vec![vec![(4, true)], vec![]]),
        )
        .unwrap();
        from_skeleton_and_new_agree_semantically(&irregular, &config);
    }

    #[test]
    fn skeleton_replay_reproduces_direct_digest_per_modulus() {
        let config = config();
        let other_config =
            Fp::<2>::make_cfg(&Uint::from(OTHER_TEST_MODULUS)).expect("odd test modulus");
        let matrix = |column: usize| {
            CscMatrix::<Box<[_]>>::try_from_rows(
                8,
                vec![vec![(column, true)], vec![(column + 1, true)]],
            )
            .unwrap()
        };
        let matrices = ConstraintMatrices::new(matrix(0), matrix(1), matrix(2)).unwrap();
        let skeleton = ConstraintMatricesSkeleton::<Fp<2>, bool>::new(matrices.clone()).unwrap();

        for field_config in [&config, &other_config] {
            let replayed =
                PreparedConstraintMatrices::<Fp<2>, bool>::from_skeleton(&skeleton, field_config)
                    .unwrap();
            let direct =
                PreparedConstraintMatrices::<Fp<2>, bool>::new(matrices.clone(), field_config)
                    .unwrap();
            assert_eq!(replayed.digest(), direct.digest());
        }

        // The digest still separates moduli and topologies.
        let at_config =
            PreparedConstraintMatrices::<Fp<2>, bool>::from_skeleton(&skeleton, &config).unwrap();
        let at_other =
            PreparedConstraintMatrices::<Fp<2>, bool>::from_skeleton(&skeleton, &other_config)
                .unwrap();
        assert_ne!(at_config.digest(), at_other.digest());

        let moved = ConstraintMatrices::new(matrix(0), matrix(1), matrix(3)).unwrap();
        let moved_skeleton = ConstraintMatricesSkeleton::<Fp<2>, bool>::new(moved).unwrap();
        let moved_prepared =
            PreparedConstraintMatrices::<Fp<2>, bool>::from_skeleton(&moved_skeleton, &config)
                .unwrap();
        assert_ne!(at_config.digest(), moved_prepared.digest());
    }

    #[test]
    fn skeleton_rejects_explicit_zero_with_direct_constructor_coordinates() {
        let config = config();
        let with_zero = || {
            let unit =
                CscMatrix::<Box<[_]>>::try_from_rows(4, vec![vec![(0, true)], vec![(1, true)]])
                    .unwrap();
            let zeroed = CscMatrix::<Box<[_]>>::try_from_rows(
                4,
                vec![vec![(0, true)], vec![(2, false), (3, true)]],
            )
            .unwrap();
            ConstraintMatrices::new(unit.clone(), zeroed, unit).unwrap()
        };
        let skeleton_error =
            ConstraintMatricesSkeleton::<Fp<2>, bool>::new(with_zero()).unwrap_err();
        let direct_error =
            PreparedConstraintMatrices::<Fp<2>, bool>::new(with_zero(), &config).unwrap_err();
        assert_eq!(
            skeleton_error,
            SpartanMatrixError::ExplicitZeroCoefficient {
                matrix: "B",
                row: 1,
                column: 2,
            }
        );
        assert_eq!(skeleton_error, direct_error);
    }

    #[test]
    fn prefix_univariate_factors_stream_disjoint_selectors_exactly() {
        let config = config();
        let rows = 6;
        let columns = 32;
        let selector = |offset: usize| {
            CscMatrix::<Box<[_]>>::try_from_rows(
                columns,
                (0..rows).map(|row| vec![(offset + row, true)]).collect(),
            )
            .unwrap()
        };
        let prepared = PreparedConstraintMatrices::<Fp<2>, bool>::new(
            ConstraintMatrices::new(selector(8), selector(16), selector(24)).unwrap(),
            &config,
        )
        .unwrap();
        assert!(prepared.selector_triplet.is_some());

        let tail_point = [field(43, &config)];
        let (tail_low, tail_high) = make_equality_factors(&tail_point, &config).unwrap();
        let factors = PrefixUnivariateRowFactors::new(
            2,
            [2, 3, 5, 7]
                .into_iter()
                .map(|value| field(value, &config))
                .collect(),
            tail_low,
            tail_high,
            prepared.num_row_vars(),
        )
        .unwrap();
        let row_weights = factors.materialize(&config);
        let rho = field(47, &config);
        let column_point = [
            field(53, &config),
            field(59, &config),
            field(61, &config),
            field(67, &config),
            field(71, &config),
        ];

        assert_eq!(
            prepared.structured().bind_prefix(&factors, &rho).unwrap(),
            prepared.binding(&rho).bind_rows(&row_weights).unwrap()
        );
        assert_eq!(
            prepared
                .structured()
                .evaluate_prefix(&factors, &rho, &column_point)
                .unwrap(),
            prepared
                .binding(&rho)
                .evaluate_at(&row_weights, &column_point)
                .unwrap()
        );
    }

    #[test]
    fn prefix_univariate_factors_preserve_generic_matrix_fallback() {
        let config = config();
        let matrix = || {
            CscMatrix::<Box<[_]>>::try_from_rows(
                8,
                vec![
                    vec![(0, field(2, &config)), (7, field(3, &config))],
                    vec![(2, field(5, &config))],
                    vec![(4, field(7, &config))],
                    vec![],
                ],
            )
            .unwrap()
        };
        let prepared = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(matrix(), matrix(), matrix()).unwrap(),
            &config,
        )
        .unwrap();
        assert!(prepared.selector_triplet.is_none());

        let tail_point = [field(11, &config)];
        let (tail_low, tail_high) = make_equality_factors(&tail_point, &config).unwrap();
        let factors = PrefixUnivariateRowFactors::new(
            1,
            vec![field(13, &config), field(17, &config)],
            tail_low,
            tail_high,
            prepared.num_row_vars(),
        )
        .unwrap();
        let row_weights = factors.materialize(&config);
        let rho = field(19, &config);
        let column_point = [field(23, &config), field(29, &config), field(31, &config)];

        assert_eq!(
            prepared.structured().bind_prefix(&factors, &rho).unwrap(),
            prepared.binding(&rho).bind_rows(&row_weights).unwrap()
        );
        assert_eq!(
            prepared
                .structured()
                .evaluate_prefix(&factors, &rho, &column_point)
                .unwrap(),
            prepared
                .binding(&rho)
                .evaluate_at(&row_weights, &column_point)
                .unwrap()
        );
    }

    /// The block-selector relation shapes of the crate — u32 (three unit
    /// selector blocks), BabyBear (a two-run output matrix with a non-unit
    /// coefficient), CM-AND (empty `A`/`B`, a four-run `C` with negative
    /// coefficients, a non-power-of-two column count), a run at column 0,
    /// everything in one block, and a capacity wider than the live rows (the
    /// u32 relation's 256-gate minimum) — as `(name, columns, [A, B, C])` in
    /// row-major form for `rows` live rows.
    fn block_selector_families(
        rows: usize,
        config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> Vec<(&'static str, usize, [Vec<Vec<(usize, Fp<2>)>>; 3])> {
        let cap = rows.next_power_of_two();
        let wide = 4 * cap;
        let one = field(1, config);
        let mut minus_one = Fp::<2>::zero_with_cfg(config);
        minus_one = config.sub(&(minus_one), &(&one));
        let mut minus_two = minus_one.clone();
        minus_two = config.sub(&(minus_two), &(&one));
        let baby_bear = field(2_013_265_921, config);
        let run = |start: usize, coefficient: &Fp<2>| -> Vec<Vec<(usize, Fp<2>)>> {
            (0..rows)
                .map(|row| vec![(start + row, coefficient.clone())])
                .collect()
        };
        let merge = |runs: Vec<Vec<Vec<(usize, Fp<2>)>>>| -> Vec<Vec<(usize, Fp<2>)>> {
            (0..rows)
                .map(|row| {
                    runs.iter()
                        .flat_map(|matrix_rows| matrix_rows[row].iter().cloned())
                        .collect()
                })
                .collect()
        };
        let empty = vec![Vec::new(); rows];
        vec![
            (
                "u32",
                4 * cap,
                [run(cap, &one), run(2 * cap, &one), run(3 * cap, &one)],
            ),
            (
                "baby-bear",
                5 * cap,
                [
                    run(cap, &one),
                    run(2 * cap, &one),
                    merge(vec![run(3 * cap, &one), run(4 * cap, &baby_bear)]),
                ],
            ),
            (
                "cm-and",
                5 * cap + 3,
                [
                    empty.clone(),
                    empty,
                    merge(vec![
                        run(cap, &one),
                        run(2 * cap, &one),
                        run(3 * cap, &minus_two),
                        run(4 * cap, &minus_one),
                    ]),
                ],
            ),
            (
                "zero-run",
                3 * cap,
                [
                    run(0, &field(3, config)),
                    run(cap, &field(5, config)),
                    merge(vec![
                        run(0, &field(7, config)),
                        run(2 * cap, &field(11, config)),
                    ]),
                ],
            ),
            (
                "one-block",
                cap,
                [run(0, &one), run(0, &one), run(0, &field(2, config))],
            ),
            (
                "wide-capacity",
                4 * wide,
                [run(wide, &one), run(2 * wide, &one), run(3 * wide, &one)],
            ),
        ]
    }

    fn prepared_from_rows(
        columns: usize,
        [a, b, c]: [Vec<Vec<(usize, Fp<2>)>>; 3],
        config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> PreparedConstraintMatrices<Fp<2>, Fp<2>> {
        let matrix = |rows| CscMatrix::<Box<[_]>>::try_from_rows(columns, rows).unwrap();
        PreparedConstraintMatrices::new(
            ConstraintMatrices::new(matrix(a), matrix(b), matrix(c)).unwrap(),
            config,
        )
        .unwrap()
    }

    /// The materialized row weights of a product functional, in the
    /// matrices' little-endian row order: the reference the verifier's
    /// closed form is pinned to.
    fn materialized_product_functional(
        functional: &ProductRowFunctional<'_, Fp<2>>,
        num_row_vars: usize,
        config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> Vec<Fp<2>> {
        if functional.skip_vars == 0 {
            let table = eq_table(functional.tail_point, config).unwrap();
            return table
                .iter()
                .map(|weight| (config).mul(&functional.prefix[0], weight))
                .collect();
        }
        let (tail_low, tail_high) = make_equality_factors(functional.tail_point, config).unwrap();
        PrefixUnivariateRowFactors::new(
            functional.skip_vars,
            functional.prefix.to_vec(),
            tail_low,
            tail_high,
            num_row_vars,
        )
        .unwrap()
        .materialize(&config)
    }

    #[test]
    fn block_selector_closed_form_matches_the_materialized_sparse_reference() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        let config = config();
        let mut rng = StdRng::seed_from_u64(0xf2f2_5eed);
        let random = |rng: &mut StdRng| field(rng.random::<u64>(), &config);
        let mut checked = 0;
        for rows in [
            1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 23, 31, 32, 33, 50, 64, 65, 100, 127,
            128,
        ] {
            for (name, columns, matrices) in block_selector_families(rows, &config) {
                let prepared = prepared_from_rows(columns, matrices, &config);
                assert!(
                    prepared.block_selector.is_some(),
                    "{name} rows={rows}: block-selector layout not detected"
                );
                let num_row_vars = prepared.num_row_vars();
                let num_column_vars = prepared.num_column_vars();
                for skip_vars in 0..=num_row_vars.min(4) {
                    let prefix = if skip_vars == 0 {
                        vec![field(1, &config)]
                    } else {
                        (0..1usize << skip_vars).map(|_| random(&mut rng)).collect()
                    };
                    let tail_point = (0..num_row_vars - skip_vars)
                        .map(|_| random(&mut rng))
                        .collect::<Vec<_>>();
                    let rho = random(&mut rng);
                    let column_point = (0..num_column_vars)
                        .map(|_| random(&mut rng))
                        .collect::<Vec<_>>();
                    let functional = ProductRowFunctional {
                        skip_vars,
                        prefix: &prefix,
                        tail_point: &tail_point,
                    };

                    let row_weights =
                        materialized_product_functional(&functional, num_row_vars, &config);
                    let expected = prepared
                        .binding(&rho)
                        .evaluate_at(&row_weights, &column_point)
                        .unwrap();
                    assert_eq!(
                        prepared
                            .structured()
                            .evaluate_product(&functional, &rho, &column_point)
                            .unwrap(),
                        expected,
                        "{name} rows={rows} skip_vars={skip_vars}"
                    );
                    if skip_vars == 0 {
                        assert_eq!(
                            prepared
                                .structured()
                                .evaluate_equality(&tail_point, &rho, &column_point)
                                .unwrap(),
                            expected,
                            "{name} rows={rows}: evaluate_batched"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked >= 500, "checked only {checked} shapes");
    }

    #[test]
    fn product_row_functional_matches_the_reference_on_generic_matrices() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        let config = config();
        let mut rng = StdRng::seed_from_u64(0x9e37_79b9);
        let random = |rng: &mut StdRng| field(rng.random::<u64>(), &config);
        // Not a block-selector layout: a column with two entries and a row
        // with none.
        let matrix = |scale: u64| {
            CscMatrix::<Box<[_]>>::try_from_rows(
                8,
                vec![
                    vec![(0, field(2 * scale, &config)), (7, field(3, &config))],
                    vec![(2, field(5, &config)), (7, field(scale, &config))],
                    vec![(4, field(7, &config))],
                    vec![],
                    vec![(1, field(11, &config))],
                ],
            )
            .unwrap()
        };
        let prepared = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(matrix(1), matrix(2), matrix(3)).unwrap(),
            &config,
        )
        .unwrap();
        assert!(prepared.block_selector.is_none());
        assert!(prepared.selector_triplet.is_none());
        let num_row_vars = prepared.num_row_vars();
        for skip_vars in 0..=num_row_vars {
            let prefix = if skip_vars == 0 {
                vec![field(1, &config)]
            } else {
                (0..1usize << skip_vars).map(|_| random(&mut rng)).collect()
            };
            let tail_point = (0..num_row_vars - skip_vars)
                .map(|_| random(&mut rng))
                .collect::<Vec<_>>();
            let rho = random(&mut rng);
            let column_point = (0..prepared.num_column_vars())
                .map(|_| random(&mut rng))
                .collect::<Vec<_>>();
            let functional = ProductRowFunctional {
                skip_vars,
                prefix: &prefix,
                tail_point: &tail_point,
            };
            let row_weights = materialized_product_functional(&functional, num_row_vars, &config);
            assert_eq!(
                prepared
                    .structured()
                    .evaluate_product(&functional, &rho, &column_point)
                    .unwrap(),
                prepared
                    .binding(&rho)
                    .evaluate_at(&row_weights, &column_point)
                    .unwrap(),
                "skip_vars={skip_vars}"
            );
        }
    }

    #[test]
    fn product_row_functional_rejects_malformed_inputs() {
        let config = config();
        let other_config =
            Fp::<2>::make_cfg(&Uint::from(OTHER_TEST_MODULUS)).expect("odd test modulus");
        let rows = 6;
        let (name, columns, matrices) = block_selector_families(rows, &config).swap_remove(0);
        assert_eq!(name, "u32");
        let prepared = prepared_from_rows(columns, matrices, &config);
        assert!(prepared.block_selector.is_some());
        assert_eq!(prepared.num_row_vars(), 3);
        assert_eq!(prepared.num_column_vars(), 5);

        let prefix = [2, 3, 5, 7]
            .into_iter()
            .map(|value| field(value, &config))
            .collect::<Vec<_>>();
        let tail_point = [field(43, &config)];
        let rho = field(47, &config);
        let column_point = [53u64, 59, 61, 67, 71]
            .into_iter()
            .map(|value| field(value, &config))
            .collect::<Vec<_>>();
        let functional = ProductRowFunctional {
            skip_vars: 2,
            prefix: &prefix,
            tail_point: &tail_point,
        };
        prepared
            .structured()
            .evaluate_product(&functional, &rho, &column_point)
            .unwrap();

        // Prefix length must be exactly 2^skip_vars.
        let bad = ProductRowFunctional {
            prefix: &prefix[..3],
            ..functional
        };
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&bad, &rho, &column_point),
            Err(SpartanMatrixError::InvalidRowWeightsLength {
                expected: 4,
                actual: 3
            })
        ));
        // Tail length must be exactly num_row_vars - skip_vars.
        let bad = ProductRowFunctional {
            tail_point: &[],
            ..functional
        };
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&bad, &rho, &column_point),
            Err(SpartanMatrixError::InvalidRowPointLength {
                expected: 1,
                actual: 0
            })
        ));
        // skip_vars beyond the row domain.
        let wide_prefix = (0..16)
            .map(|value| field(value, &config))
            .collect::<Vec<_>>();
        let bad = ProductRowFunctional {
            skip_vars: 4,
            prefix: &wide_prefix,
            tail_point: &[],
        };
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&bad, &rho, &column_point),
            Err(SpartanMatrixError::InvalidRowPointLength {
                expected: 3,
                actual: 4
            })
        ));
        // Column point of the wrong width.
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&functional, &rho, &column_point[..4]),
            Err(SpartanMatrixError::InvalidColumnPointLength {
                expected: 5,
                actual: 4
            })
        ));
        // Elements from another field configuration, anywhere.
        let foreign = crate::piop::spartan::noncanonical_test_value(&config);
        let mut foreign_column = column_point.clone();
        foreign_column[2] = foreign.clone();
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&functional, &rho, &foreign_column),
            Err(SpartanMatrixError::InvalidFieldConfiguration(
                SpartanFieldError::NonCanonicalElement
            ))
        ));
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&functional, &foreign, &column_point),
            Err(SpartanMatrixError::InvalidFieldConfiguration(
                SpartanFieldError::NonCanonicalElement
            ))
        ));
        let foreign_tail = [foreign.clone()];
        let bad = ProductRowFunctional {
            tail_point: &foreign_tail,
            ..functional
        };
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&bad, &rho, &column_point),
            Err(SpartanMatrixError::InvalidFieldConfiguration(
                SpartanFieldError::NonCanonicalElement
            ))
        ));
        let mut foreign_prefix = prefix.clone();
        foreign_prefix[1] = foreign;
        let bad = ProductRowFunctional {
            prefix: &foreign_prefix,
            ..functional
        };
        assert!(matches!(
            prepared
                .structured()
                .evaluate_product(&bad, &rho, &column_point),
            Err(SpartanMatrixError::InvalidFieldConfiguration(
                SpartanFieldError::NonCanonicalElement
            ))
        ));
    }

    #[test]
    fn hypercube_prefix_sum_and_boolean_index_match_brute_force() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        let config = config();
        let mut rng = StdRng::seed_from_u64(0x0dd_b1a5);
        for vars in 0..=6 {
            let left = (0..vars)
                .map(|_| field(rng.random::<u64>(), &config))
                .collect::<Vec<_>>();
            let right = (0..vars)
                .map(|_| field(rng.random::<u64>(), &config))
                .collect::<Vec<_>>();
            let left_table = eq_table(&left, &config).unwrap();
            let right_table = eq_table(&right, &config).unwrap();
            for index in 0..left_table.len() {
                assert_eq!(
                    eq_at_boolean_index(&left, index, &config).unwrap(),
                    left_table[index],
                    "vars={vars} index={index}"
                );
            }
            assert!(eq_at_boolean_index(&left, left_table.len(), &config).is_err());
            let mut running = Fp::<2>::zero_with_cfg(&config);
            for count in 0..=left_table.len() {
                assert_eq!(
                    hypercube_prefix_sum(&left, &right, count, &config).unwrap(),
                    running,
                    "vars={vars} count={count}"
                );
                if count < left_table.len() {
                    running = config.add(
                        &(running),
                        &(&(config).mul(&left_table[count], &right_table[count])),
                    );
                }
            }
            assert_eq!(
                hypercube_prefix_sum(&left, &right, left_table.len(), &config).unwrap(),
                eq_eval(&left, &right, &config).unwrap()
            );
            assert!(hypercube_prefix_sum(&left, &right, left_table.len() + 1, &config).is_err());
        }
    }

    #[test]
    fn explicit_row_weight_operations_validate_length_and_field() {
        let config = config();
        let other_config = Fp::<2>::make_cfg(&Uint::from(OTHER_TEST_MODULUS)).unwrap();
        let matrix = || {
            CscMatrix::<Box<[_]>>::try_from_rows(
                1,
                vec![vec![(0, field(1, &config))], vec![], vec![]],
            )
            .unwrap()
        };
        let prepared = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(matrix(), matrix(), matrix()).unwrap(),
            &config,
        )
        .unwrap();
        let rho = field(2, &config);
        let column_point: [Fp<2>; 0] = [];
        let short_weights = vec![field(3, &config); 3];

        assert_eq!(
            prepared.binding(&rho).bind_rows(&short_weights),
            Err(crate::sumcheck::SumcheckError::Matrix(
                SpartanMatrixError::InvalidRowWeightsLength {
                    expected: 4,
                    actual: 3,
                }
            ))
        );
        assert_eq!(
            prepared
                .binding(&rho)
                .evaluate_at(&short_weights, &column_point),
            Err(crate::sumcheck::SumcheckError::Matrix(
                SpartanMatrixError::InvalidRowWeightsLength {
                    expected: 4,
                    actual: 3,
                }
            ))
        );

        let foreign_weights = vec![crate::piop::spartan::noncanonical_test_value(&config); 4];
        assert_eq!(
            prepared.binding(&rho).bind_rows(&foreign_weights),
            Err(crate::sumcheck::SumcheckError::Matrix(
                SpartanMatrixError::InvalidFieldConfiguration(
                    SpartanFieldError::NonCanonicalElement
                )
            ))
        );
        assert_eq!(
            prepared
                .binding(&rho)
                .evaluate_at(&foreign_weights, &column_point),
            Err(crate::sumcheck::SumcheckError::Matrix(
                SpartanMatrixError::InvalidFieldConfiguration(
                    SpartanFieldError::NonCanonicalElement
                )
            ))
        );
    }

    #[test]
    fn boolean_one_coefficients_match_field_one_statement_and_evaluation() {
        let config = config();
        let one = Fp::<2>::one_with_cfg(&config);
        let field_matrix = CscMatrix::<Box<[_]>>::try_from_rows(
            4,
            vec![
                vec![(0, one.clone()), (3, one.clone())],
                vec![(1, one.clone())],
                vec![(2, one)],
            ],
        )
        .unwrap();
        let boolean_matrix = CscMatrix::<Box<[_]>>::try_from_rows(
            4,
            vec![vec![(0, true), (3, true)], vec![(1, true)], vec![(2, true)]],
        )
        .unwrap();
        let field_prepared = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(field_matrix.clone(), field_matrix.clone(), field_matrix)
                .unwrap(),
            &config,
        )
        .unwrap();
        let boolean_prepared = PreparedConstraintMatrices::<Fp<2>, bool>::new(
            ConstraintMatrices::new(
                boolean_matrix.clone(),
                boolean_matrix.clone(),
                boolean_matrix,
            )
            .unwrap(),
            &config,
        )
        .unwrap();

        assert_eq!(field_prepared.digest(), boolean_prepared.digest());

        let row_point = [field(7, &config), field(11, &config)];
        let column_point = [field(13, &config), field(17, &config)];
        let rho = field(19, &config);
        assert_eq!(
            crate::piop::spartan::matrix::eq_table_prover(&row_point, field_prepared.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| field_prepared.binding(&rho).bind_rows(&weights))
                .unwrap(),
            crate::piop::spartan::matrix::eq_table_prover(&row_point, boolean_prepared.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| boolean_prepared.binding(&rho).bind_rows(&weights))
                .unwrap()
        );
        assert_eq!(
            field_prepared
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap(),
            boolean_prepared
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap()
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn indexed_parallel_binding_is_exact_for_generic_coefficients_at_threshold() {
        let config = config();
        let column_count = (1 << 12) + 1;
        let boolean_rows = vec![
            vec![(0, true), (1 << 12, true)],
            vec![(1, true), (1 << 11, true)],
            vec![(2, true)],
            vec![(3, true), ((1 << 12) - 1, true)],
        ];
        let one = Fp::<2>::one_with_cfg(&config);
        let field_rows: Vec<Vec<(usize, Fp<2>)>> = boolean_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(column, _)| (*column, one.clone()))
                    .collect()
            })
            .collect();
        let boolean_matrix =
            || CscMatrix::<Box<[_]>>::try_from_rows(column_count, boolean_rows.clone()).unwrap();
        let field_matrix =
            || CscMatrix::<Box<[_]>>::try_from_rows(column_count, field_rows.clone()).unwrap();
        let boolean_prepared = PreparedConstraintMatrices::<Fp<2>, bool>::new(
            ConstraintMatrices::new(boolean_matrix(), boolean_matrix(), boolean_matrix()).unwrap(),
            &config,
        )
        .unwrap();
        let field_prepared = PreparedConstraintMatrices::<Fp<2>>::new(
            ConstraintMatrices::new(field_matrix(), field_matrix(), field_matrix()).unwrap(),
            &config,
        )
        .unwrap();
        let row_point = [field(7, &config), field(11, &config)];
        let rho = field(13, &config);
        let sequential_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let sequential_boolean = sequential_pool
            .install(|| {
                crate::piop::spartan::matrix::eq_table_prover(&row_point, boolean_prepared.config())
                    .map_err(crate::sumcheck::SumcheckError::from)
                    .and_then(|weights| boolean_prepared.binding(&rho).bind_rows(&weights))
            })
            .unwrap();
        let parallel_boolean = parallel_pool
            .install(|| {
                crate::piop::spartan::matrix::eq_table_prover(&row_point, boolean_prepared.config())
                    .map_err(crate::sumcheck::SumcheckError::from)
                    .and_then(|weights| boolean_prepared.binding(&rho).bind_rows(&weights))
            })
            .unwrap();
        let sequential_field = sequential_pool
            .install(|| {
                crate::piop::spartan::matrix::eq_table_prover(&row_point, field_prepared.config())
                    .map_err(crate::sumcheck::SumcheckError::from)
                    .and_then(|weights| field_prepared.binding(&rho).bind_rows(&weights))
            })
            .unwrap();
        let parallel_field = parallel_pool
            .install(|| {
                crate::piop::spartan::matrix::eq_table_prover(&row_point, field_prepared.config())
                    .map_err(crate::sumcheck::SumcheckError::from)
                    .and_then(|weights| field_prepared.binding(&rho).bind_rows(&weights))
            })
            .unwrap();

        assert_eq!(parallel_boolean, sequential_boolean);
        assert_eq!(parallel_field, sequential_field);
        assert_eq!(parallel_boolean, parallel_field);
        assert!(
            parallel_boolean.evaluations[column_count..]
                .iter()
                .all(|value| <Fp<2> as crate::piop::spartan::SpartanField>::is_zero(value))
        );
    }

    #[test]
    fn prepared_boolean_statement_rejects_explicit_false_coefficients() {
        let config = config();
        let false_matrix = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, false)]]).unwrap();
        let true_matrix = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, true)]]).unwrap();
        let matrices =
            ConstraintMatrices::new(false_matrix, true_matrix.clone(), true_matrix).unwrap();

        assert!(matches!(
            PreparedConstraintMatrices::<Fp<2>, bool>::new(matrices, &config),
            Err(SpartanMatrixError::ExplicitZeroCoefficient {
                matrix: "A",
                row: 0,
                column: 0,
            })
        ));
    }

    #[test]
    fn csc_binding_matches_dense_oracle_with_empty_and_padded_domains() {
        let config = config();
        let row_count = 3;
        let column_count = 6;
        let zero = Fp::<2>::zero_with_cfg(&config);

        let a_rows = vec![
            vec![(0, field(2, &config)), (4, field(3, &config))],
            vec![],
            vec![(2, field(5, &config))],
        ];
        let b_rows = vec![
            vec![(1, field(7, &config))],
            vec![(4, field(11, &config))],
            vec![],
        ];
        let c_rows = vec![
            vec![],
            vec![(0, field(13, &config))],
            vec![(3, field(17, &config))],
        ];
        let matrices = ConstraintMatrices::new(
            CscMatrix::<Box<[_]>>::try_from_rows(column_count, a_rows.clone()).unwrap(),
            CscMatrix::<Box<[_]>>::try_from_rows(column_count, b_rows.clone()).unwrap(),
            CscMatrix::<Box<[_]>>::try_from_rows(column_count, c_rows.clone()).unwrap(),
        )
        .unwrap();
        let prepared = PreparedConstraintMatrices::new(matrices, &config).unwrap();
        let row_point = [field(19, &config), field(23, &config)];
        let column_point = [field(29, &config), field(31, &config), field(37, &config)];
        let rho = field(41, &config);
        let rho_squared = (config).mul(&rho, &rho);
        let row_weights = eq_table(&row_point, &config).unwrap();
        let column_weights = eq_table(&column_point, &config).unwrap();

        let coefficient_at = |rows: &[Vec<(usize, Fp<2>)>], row: usize, column: usize| {
            rows[row]
                .iter()
                .find_map(|(entry_column, coefficient)| {
                    (*entry_column == column).then(|| coefficient.clone())
                })
                .unwrap_or_else(|| zero.clone())
        };
        let mut expected_bound = Vec::with_capacity(8);
        for column in 0..column_count {
            let mut column_evaluation = zero.clone();
            for row in 0..row_count {
                let mut batched_coefficient = coefficient_at(&a_rows, row, column);
                batched_coefficient = config.add(
                    &(batched_coefficient),
                    &(&(config).mul(&rho, &coefficient_at(&b_rows, row, column))),
                );
                batched_coefficient = config.add(
                    &(batched_coefficient),
                    &(&(config).mul(&rho_squared, &coefficient_at(&c_rows, row, column))),
                );
                column_evaluation = config.add(
                    &(column_evaluation),
                    &(&(config).mul(&row_weights[row], &batched_coefficient)),
                );
            }
            expected_bound.push(column_evaluation);
        }
        expected_bound.resize(8, zero.clone());

        let bound = crate::piop::spartan::matrix::eq_table_prover(&row_point, prepared.config())
            .map_err(crate::sumcheck::SumcheckError::from)
            .and_then(|weights| prepared.binding(&rho).bind_rows(&weights))
            .unwrap();
        assert_eq!(bound.num_vars, 3);
        assert_eq!(bound.evaluations, expected_bound);
        assert_eq!(bound.evaluations[5], zero);

        let expected_evaluation = expected_bound.iter().zip(&column_weights).fold(
            Fp::<2>::zero_with_cfg(&config),
            |mut evaluation, (bound_value, column_weight)| {
                evaluation =
                    config.add(&(evaluation), &(&(config).mul(bound_value, column_weight)));
                evaluation
            },
        );
        assert_eq!(
            prepared
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap(),
            expected_evaluation
        );
    }

    #[test]
    fn csc_preparation_preserves_legacy_row_major_digest() {
        let config = config();
        let column_count = 5;
        let a_rows = vec![
            vec![(0, field(2, &config)), (3, field(3, &config))],
            vec![],
            vec![(1, field(5, &config))],
            vec![],
        ];
        let b_rows = vec![
            vec![],
            vec![(2, field(7, &config)), (4, field(11, &config))],
            vec![],
            vec![(0, field(13, &config))],
        ];
        let c_rows = vec![
            vec![(4, field(17, &config))],
            vec![],
            vec![(0, field(19, &config)), (3, field(23, &config))],
            vec![],
        ];
        let row_count = a_rows.len();

        let from_rows = ConstraintMatrices::new(
            CscMatrix::<Box<[_]>>::try_from_rows(column_count, a_rows.clone()).unwrap(),
            CscMatrix::<Box<[_]>>::try_from_rows(column_count, b_rows.clone()).unwrap(),
            CscMatrix::<Box<[_]>>::try_from_rows(column_count, c_rows.clone()).unwrap(),
        )
        .unwrap();
        let from_columns = ConstraintMatrices::new(
            CscMatrix::<Box<[_]>>::try_from_columns(
                row_count,
                columns_from_rows(column_count, &a_rows),
            )
            .unwrap(),
            CscMatrix::<Box<[_]>>::try_from_columns(
                row_count,
                columns_from_rows(column_count, &b_rows),
            )
            .unwrap(),
            CscMatrix::<Box<[_]>>::try_from_columns(
                row_count,
                columns_from_rows(column_count, &c_rows),
            )
            .unwrap(),
        )
        .unwrap();

        let expected = legacy_row_major_digest(
            [&a_rows, &b_rows, &c_rows],
            row_count,
            column_count,
            &config,
        );
        let row_prepared = PreparedConstraintMatrices::new(from_rows, &config).unwrap();
        let column_prepared = PreparedConstraintMatrices::new(from_columns, &config).unwrap();
        assert_eq!(*row_prepared.digest(), expected);
        assert_eq!(column_prepared.digest(), row_prepared.digest());
    }

    #[test]
    fn csc_validation_preserves_canonical_row_major_error_order() {
        let config = config();
        let zero = Fp::<2>::zero_with_cfg(&config);
        let one = Fp::<2>::one_with_cfg(&config);
        // CSC visits (row 1, column 0) before (row 0, column 1), but the
        // statement's canonical validation order remains row-major.
        let a = CscMatrix::<Box<[_]>>::try_from_columns(
            2,
            vec![vec![(1, zero.clone())], vec![(0, zero)]],
        )
        .unwrap();
        let valid = || {
            CscMatrix::<Box<[_]>>::try_from_columns(2, vec![vec![(0, one.clone())], Vec::new()])
                .unwrap()
        };

        assert!(matches!(
            PreparedConstraintMatrices::new(
                ConstraintMatrices::new(a, valid(), valid()).unwrap(),
                &config,
            ),
            Err(SpartanMatrixError::ExplicitZeroCoefficient {
                matrix: "A",
                row: 0,
                column: 1,
            })
        ));
    }

    #[test]
    fn checked_builders_pad_only_at_the_end() {
        let config = config();
        let one = Fp::<2>::one_with_cfg(&config);
        let zero = Fp::<2>::zero_with_cfg(&config);
        let assignment = [one.clone(), field(2, &config), field(3, &config)];
        let assignment_mle = build_assignment_mle(&assignment, 3, &config).unwrap();
        assert_eq!(assignment_mle.num_vars, 2);
        assert_eq!(
            assignment_mle.evaluations,
            vec![
                assignment[0].clone(),
                assignment[1].clone(),
                assignment[2].clone(),
                zero
            ]
        );

        let products = build_product_mles(
            &[field(4, &config), field(5, &config), field(6, &config)],
            &[field(7, &config), field(8, &config), field(9, &config)],
            &[field(10, &config), field(11, &config), field(12, &config)],
            3,
            &config,
        )
        .unwrap();
        assert_eq!(products.az.num_vars, 2);
        assert_eq!(products.bz.num_vars, 2);
        assert_eq!(products.cz.num_vars, 2);

        let invalid = [field(2, &config), one];
        assert_eq!(
            build_assignment_mle(&invalid, 2, &config),
            Err(SpartanMatrixError::InvalidAssignmentConstant)
        );
    }

    #[test]
    fn scaled_claim_checks_without_dividing_by_the_scale() {
        let config = config();
        let point = [field(2, &config), field(3, &config)];
        let polynomial = DenseMultilinearExtension {
            evaluations: vec![
                field(5, &config),
                field(7, &config),
                field(11, &config),
                field(13, &config),
            ],
            num_vars: 2,
        };
        let evaluation = evaluate_mle(&polynomial, &point, &config).unwrap();
        let scale = field(17, &config);
        let claim = ScaledMleEvaluationClaim::new(
            point.to_vec().into_boxed_slice(),
            scale.clone(),
            (config).mul(&scale, &evaluation),
        );
        claim.nonsuccinct_verify(&polynomial, &config).unwrap();

        let zero = Fp::<2>::zero_with_cfg(&config);
        let zero_scaled =
            ScaledMleEvaluationClaim::new(point.to_vec().into_boxed_slice(), zero.clone(), zero);
        zero_scaled
            .nonsuccinct_verify(&polynomial, &config)
            .unwrap();
    }

    #[test]
    fn prepared_statement_rejects_coefficients_from_another_modulus() {
        let config = config();
        let other_config = Fp::<2>::make_cfg(&Uint::from(OTHER_TEST_MODULUS)).unwrap();
        let foreign = crate::piop::spartan::noncanonical_test_value(&config);
        let local = field(1, &config);
        let a = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, foreign)]]).unwrap();
        let b = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, local.clone())]]).unwrap();
        let c = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, local)]]).unwrap();
        let matrices = ConstraintMatrices::new(a, b, c).unwrap();

        assert!(matches!(
            PreparedConstraintMatrices::new(matrices, &config),
            Err(SpartanMatrixError::InvalidFieldConfiguration(
                SpartanFieldError::NonCanonicalElement
            ))
        ));
    }

    #[test]
    fn prepared_digest_binds_the_runtime_modulus() {
        let config = config();
        let other_config = Fp::<2>::make_cfg(&Uint::from(OTHER_TEST_MODULUS)).unwrap();

        let prepare = |config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config| {
            let matrix = || {
                CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, field(1, config))]]).unwrap()
            };
            PreparedConstraintMatrices::new(
                ConstraintMatrices::new(matrix(), matrix(), matrix()).unwrap(),
                config,
            )
            .unwrap()
        };
        let prepared = prepare(&config);
        let other_prepared = prepare(&other_config);

        assert_ne!(
            prepared.field_modulus_encoding(),
            other_prepared.field_modulus_encoding()
        );
        assert_ne!(prepared.digest(), other_prepared.digest());
    }

    #[test]
    fn prepared_statement_rejects_unsafe_runtime_fields() {
        let matrices = |config: &<Fp<2> as crate::piop::spartan::SpartanField>::Config| {
            let matrix = || {
                CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, field(1, config))]]).unwrap()
            };
            ConstraintMatrices::new(matrix(), matrix(), matrix()).unwrap()
        };

        assert_eq!(
            Fp::<2>::make_cfg(&Uint::from((1_u128 << 100) - 17)),
            Err(SpartanFieldError::CompositeModulus)
        );

        let undersized = Fp::<2>::make_cfg(&Uint::from(97_u128)).unwrap();
        assert!(matches!(
            PreparedConstraintMatrices::new(matrices(&undersized), &undersized),
            Err(SpartanMatrixError::InvalidFieldConfiguration(
                SpartanFieldError::ModulusTooSmall { actual_bits: 7 }
            ))
        ));
    }

    #[test]
    fn prepared_statement_rejects_explicit_sparse_zeroes() {
        let config = config();
        let zero = Fp::<2>::zero_with_cfg(&config);
        let one = Fp::<2>::one_with_cfg(&config);
        let a = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, zero)]]).unwrap();
        let b = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, one.clone())]]).unwrap();
        let c = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, one)]]).unwrap();

        assert!(matches!(
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), &config),
            Err(SpartanMatrixError::ExplicitZeroCoefficient {
                matrix: "A",
                row: 0,
                column: 0,
            })
        ));
    }

    #[test]
    fn prepared_statement_rejects_unchecked_noncanonical_residues() {
        let config = config();
        let malformed = field::FpCtx::from_prime_u128(u128::MAX - 158)
            .from_montgomery_integer(*config.modulus());
        let one = Fp::<2>::one_with_cfg(&config);
        let a = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, malformed)]]).unwrap();
        let b = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, one.clone())]]).unwrap();
        let c = CscMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, one)]]).unwrap();

        assert!(matches!(
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), &config),
            Err(SpartanMatrixError::InvalidFieldConfiguration(
                SpartanFieldError::NonCanonicalElement
            ))
        ));
    }
}
