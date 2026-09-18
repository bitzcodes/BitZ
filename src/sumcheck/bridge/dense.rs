//! Generic field-valued CSC binding; native binding shares the column scheduler.
use crate::{
    piop::spartan::{
        SpartanField,
        matrix::{PreparedConstraintMatrices, SpartanMatrixCoefficient, SpartanMatrixError},
    },
    poly::mle::DenseMultilinearExtension,
};
use circuit::linear_map::{BilinearEval, LeftMul};
use field::RingOps;
/// One prepared ordinary R1CS linear map: A + rho B + rho² C.
/// The matrices remain separate; no combined sparse matrix is constructed.
pub(crate) struct DenseBinding<'a, F: SpartanField, C> {
    matrices: &'a PreparedConstraintMatrices<F, C>,
    rho: &'a F,
}
impl<'a, F: SpartanField, C: SpartanMatrixCoefficient<F>> DenseBinding<'a, F, C> {
    pub(crate) fn new(matrices: &'a PreparedConstraintMatrices<F, C>, rho: &'a F) -> Self {
        Self { matrices, rho }
    }
    fn validate_rows(&self, rows: &[F]) -> Result<(), crate::sumcheck::SumcheckError> {
        let expected = 1usize << self.matrices.num_row_vars();
        if rows.len() != expected {
            return Err(SpartanMatrixError::InvalidRowWeightsLength {
                expected,
                actual: rows.len(),
            }
            .into());
        }
        // Runtime contexts remain the caller's contract. Debug builds retain diagnostics.
        #[cfg(any(test, debug_assertions))]
        {
            crate::piop::spartan::matrix::validate_elements_field(
                rows,
                self.matrices.field_modulus_encoding(),
            )?;
        }
        crate::piop::spartan::matrix::validate_element_field(
            self.rho,
            self.matrices.field_modulus_encoding(),
        )?;
        Ok(())
    }
    fn column_value(&self, column: usize, rows: &[F], rho_squared: &F) -> F {
        let field = self.matrices.config();
        let m = self.matrices.matrices();
        let dot = |matrix: &circuit::linear_map::CscMatrix<Box<[C]>>| {
            circuit::linear_map::contraction::segment_dot(
                matrix.column(column).unwrap(),
                field.zero(),
                |row, c| c.scale(&rows[row], field),
                |a, b| field.add(&a, &b),
            )
        };
        let mut sum = dot(m.a());
        for (matrix, scale) in [(m.b(), self.rho), (m.c(), rho_squared)] {
            if !matrix.column(column).unwrap().is_empty() {
                sum = field.add(&sum, &field.mul(scale, &dot(matrix)));
            }
        }
        sum
    }
}
impl<F: SpartanField, C: SpartanMatrixCoefficient<F>> LeftMul<F> for DenseBinding<'_, F, C> {
    type Output = F;
    fn mul_left_into(
        &mut self,
        weights: &[F],
        out: &mut [F],
    ) -> Result<(), circuit::linear_map::LinearMapError> {
        let m = self.matrices.matrices();
        for (kind, expected, actual) in [
            ("row weights", m.row_count(), weights.len()),
            ("column output", m.column_count(), out.len()),
        ] {
            if expected != actual {
                return Err(circuit::linear_map::LinearMapError::Length {
                    kind,
                    expected,
                    actual,
                });
            }
        }
        let rho_squared = self.matrices.config().mul(self.rho, self.rho);
        circuit::linear_map::contraction::columns_into(out, cfg!(feature = "parallel"), |j| {
            self.column_value(j, weights, &rho_squared)
        });
        Ok(())
    }
}
impl<F: SpartanField, C: SpartanMatrixCoefficient<F>> BilinearEval<F::Config>
    for DenseBinding<'_, F, C>
{
    fn evaluate_bilinear(
        &mut self,
        weights: &[F],
        columns: &impl circuit::linear_map::ColumnValues<F>,
    ) -> Result<F, circuit::linear_map::LinearMapError> {
        let m = self.matrices.matrices();
        for (kind, expected, actual) in [
            ("row weights", m.row_count(), weights.len()),
            ("column values", m.column_count(), columns.len()),
        ] {
            if expected != actual {
                return Err(circuit::linear_map::LinearMapError::Length {
                    kind,
                    expected,
                    actual,
                });
            }
        }
        let field = self.matrices.config();
        let rho_squared = field.mul(self.rho, self.rho);
        let mut sum = field.zero();
        for j in 0..m.column_count() {
            sum = field.add(
                &sum,
                &field.mul(
                    &self.column_value(j, weights, &rho_squared),
                    &columns.scalar(j),
                ),
            );
        }
        Ok(sum)
    }
}
impl<F: SpartanField, C: SpartanMatrixCoefficient<F>> super::PreparedBinding<F::Config>
    for DenseBinding<'_, F, C>
{
    type Bound = DenseMultilinearExtension<F>;
    fn bind_rows(&mut self, rows: &[F]) -> Result<Self::Bound, crate::sumcheck::SumcheckError> {
        let mut out = DenseMultilinearExtension {
            evaluations: Vec::new(),
            num_vars: self.matrices.num_column_vars(),
        };
        self.bind_rows_into(rows, &mut out)?;
        Ok(out)
    }
    fn bind_rows_into(
        &mut self,
        rows: &[F],
        out: &mut Self::Bound,
    ) -> Result<(), crate::sumcheck::SumcheckError> {
        self.validate_rows(rows)?;
        let live = self.matrices.matrices().column_count();
        let vars = self.matrices.num_column_vars();
        out.evaluations
            .resize(1usize << vars, self.matrices.config().zero());
        out.evaluations[live..].fill(self.matrices.config().zero());
        out.num_vars = vars;
        self.mul_left_into(
            &rows[..self.matrices.matrices().row_count()],
            &mut out.evaluations[..live],
        )
        .map_err(|_| crate::sumcheck::SumcheckError::InvalidProductDimensions)
    }
    fn evaluate_at(
        &mut self,
        rows: &[F],
        point: &[F],
    ) -> Result<F, crate::sumcheck::SumcheckError> {
        self.validate_rows(rows)?;
        let expected = self.matrices.num_column_vars();
        if point.len() != expected {
            return Err(SpartanMatrixError::InvalidColumnPointLength {
                expected,
                actual: point.len(),
            }
            .into());
        }
        crate::piop::spartan::matrix::validate_elements_field(
            point,
            self.matrices.field_modulus_encoding(),
        )?;
        let columns = EqualityColumns::new(
            self.matrices.config(),
            point,
            self.matrices.matrices().column_count(),
        );
        self.evaluate_bilinear(&rows[..self.matrices.matrices().row_count()], &columns)
            .map_err(|_| crate::sumcheck::SumcheckError::InvalidProductDimensions)
    }
}

/// Explicit-row bridge for any shared-library mixed MAC implementation.
impl<F, C> super::PreparedBinding<F>
    for circuit::linear_map::PreparedColumns<
        '_,
        F,
        Box<[C]>,
        circuit::linear_map::DirectCoefficients,
    >
where
    F: field::RingOps
        + field::BatchMulAcc<F::Elem, C>
        + field::Reduce<<F as field::BatchMulAcc<F::Elem, C>>::Accumulator, Output = F::Elem>
        + Sync,
    C: Copy + Sync,
{
    type Bound = Vec<F::Elem>;
    fn bind_rows(
        &mut self,
        rows: &[F::Elem],
    ) -> Result<Self::Bound, crate::sumcheck::SumcheckError> {
        let mut out = Vec::new();
        self.bind_rows_into(rows, &mut out)?;
        Ok(out)
    }
    fn bind_rows_into(
        &mut self,
        rows: &[F::Elem],
        out: &mut Self::Bound,
    ) -> Result<(), crate::sumcheck::SumcheckError> {
        if rows.len() != self.row_count() {
            return Err(crate::sumcheck::SumcheckError::InvalidProductDimensions);
        }
        out.resize(self.column_count(), self.field().zero());
        self.mul_left_into(rows, out)
            .map_err(|_| crate::sumcheck::SumcheckError::InvalidProductDimensions)
    }
    fn evaluate_at(
        &mut self,
        rows: &[F::Elem],
        point: &[F::Elem],
    ) -> Result<F::Elem, crate::sumcheck::SumcheckError> {
        let domain = 1usize
            .checked_shl(point.len() as u32)
            .ok_or(crate::sumcheck::SumcheckError::InvalidProductDimensions)?;
        if domain < self.column_count() {
            return Err(crate::sumcheck::SumcheckError::InvalidProductDimensions);
        }
        let columns = EqualityColumns::new(self.field(), point, self.column_count());
        self.evaluate_bilinear(rows, &columns)
            .map_err(|_| crate::sumcheck::SumcheckError::InvalidProductDimensions)
    }
}
pub(super) struct EqualityColumns<'a, F: RingOps> {
    field: &'a F,
    low: Vec<F::Elem>,
    high: Vec<F::Elem>,
    len: usize,
}
impl<'a, F: RingOps> EqualityColumns<'a, F> {
    pub(super) fn new(field: &'a F, point: &[F::Elem], len: usize) -> Self {
        let table = |p: &[F::Elem]| {
            let mut w = field.zero_vec(1 << p.len());
            w[0] = field.one();
            for (bit, r) in p.iter().enumerate() {
                for i in 0..1 << bit {
                    let t = field.mul(&w[i], r);
                    w[i] = field.sub(&w[i], &t);
                    w[i + (1 << bit)] = t;
                }
            }
            w
        };
        let split = point.len() / 2;
        Self {
            field,
            low: table(&point[..split]),
            high: table(&point[split..]),
            len,
        }
    }
}
impl<F: RingOps + Sync> circuit::linear_map::ColumnValues<F::Elem> for EqualityColumns<'_, F> {
    fn len(&self) -> usize {
        self.len
    }
    fn scalar(&self, i: usize) -> F::Elem {
        self.field.mul(
            &self.low[i % self.low.len()],
            &self.high[i / self.low.len()],
        )
    }
    fn power_sum(&self, first: usize, len: usize) -> F::Elem {
        (first..first + len).rev().fold(self.field.zero(), |s, i| {
            self.field.add(&self.field.add(&s, &s), &self.scalar(i))
        })
    }
}
