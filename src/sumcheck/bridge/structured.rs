//! Structured row functionals preserve selector factoring and streamed prefix execution.
use super::PreparedBinding;
use crate::piop::spartan::{
    SpartanField,
    matrix::{
        BlockSelectorLayout, PrefixUnivariateRowFactors, PreparedConstraintMatrices,
        ProductRowFunctional, SelectorRun, SpartanMatrixCoefficient, SpartanMatrixError,
        domain_size, eq_at_boolean_index, eq_table, make_equality_factors, product_row_prefix_sum,
        validate_element_field, validate_elements_field,
    },
};
use crate::poly::mle::DenseMultilinearExtension;
use field::RingOps;

pub(crate) struct StructuredBinding<'a, F: SpartanField, C> {
    matrices: &'a PreparedConstraintMatrices<F, C>,
}
impl<'a, F: SpartanField, C: SpartanMatrixCoefficient<F>> StructuredBinding<'a, F, C> {
    pub(crate) fn new(matrices: &'a PreparedConstraintMatrices<F, C>) -> Self {
        Self { matrices }
    }
    /// Constructs the dense batched column MLE from a three-factor
    /// prefix-univariate row functional.
    ///
    /// Disjoint unit-selector matrices are streamed a row block at a time, so
    /// the complete row-weight tensor is never allocated. Other matrix layouts
    /// retain the existing materialized implementation as a correctness- and
    /// latency-preserving fallback.
    pub fn bind_prefix(
        &self,
        factors: &PrefixUnivariateRowFactors<F>,
        rho: &F,
    ) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError> {
        if factors.parts().num_row_vars != self.matrices.num_row_vars() {
            return Err(SpartanMatrixError::InvalidRowWeightsLength {
                expected: domain_size(self.matrices.num_row_vars())?,
                actual: domain_size(factors.parts().num_row_vars)?,
            });
        }

        let Some([rows, a_offset, b_offset, c_offset]) = self.matrices.selector_layout() else {
            let row_weights = factors.materialize(self.matrices.config());
            return self
                .matrices
                .binding(rho)
                .bind_rows(&row_weights)
                .map_err(binding_matrix_error);
        };

        let zero = F::zero_with_cfg(self.matrices.config());
        let rho_squared = (self.matrices.config()).mul(rho, rho);
        let mut evaluations = vec![zero; domain_size(self.matrices.num_column_vars())?];
        factors.for_each_row(
            rows,
            |row, weight| {
                evaluations[a_offset + row] = weight.clone();
                evaluations[b_offset + row] = (self.matrices.config()).mul(rho, &weight);
                evaluations[c_offset + row] = (self.matrices.config()).mul(&rho_squared, &weight);
            },
            self.matrices.config(),
        );

        Ok(DenseMultilinearExtension {
            evaluations,
            num_vars: self.matrices.num_column_vars(),
        })
    }
    /// Directly evaluates
    ///
    /// `A(row_point, column_point) + rho B(row_point, column_point)
    /// + rho^2 C(row_point, column_point)`.
    ///
    /// This deliberately does not construct or evaluate the dense table from
    /// row binding, keeping the verifier path independent of the
    /// prover's materialization kernel.
    pub fn evaluate_equality(
        &self,
        row_point: &[F],
        rho: &F,
        column_point: &[F],
    ) -> Result<F, SpartanMatrixError> {
        if row_point.len() != self.matrices.num_row_vars() {
            return Err(SpartanMatrixError::InvalidRowPointLength {
                expected: self.matrices.num_row_vars(),
                actual: row_point.len(),
            });
        }
        if column_point.len() != self.matrices.num_column_vars() {
            return Err(SpartanMatrixError::InvalidColumnPointLength {
                expected: self.matrices.num_column_vars(),
                actual: column_point.len(),
            });
        }
        validate_elements_field(row_point, self.matrices.field_modulus_encoding())?;
        validate_elements_field(column_point, self.matrices.field_modulus_encoding())?;
        validate_element_field(rho, self.matrices.field_modulus_encoding())?;

        if self.matrices.block_selector().is_some() {
            // The equality functional is the `K = 0` product functional:
            // succinct on a block-selector layout, and never a `2^num_row_vars`
            // table plus an `O(nnz)` walk.
            let one = F::one_with_cfg(self.matrices.config());
            let functional = ProductRowFunctional {
                skip_vars: 0,
                prefix: std::slice::from_ref(&one),
                tail_point: row_point,
            };
            return self.evaluate_product(&functional, rho, column_point);
        }

        let row_weights = eq_table(row_point, self.matrices.config())?;
        self.matrices
            .binding(rho)
            .evaluate_at(&row_weights, column_point)
            .map_err(binding_matrix_error)
    }
    /// Evaluates the batched matrices against a product-form row functional
    /// `W` ([`ProductRowFunctional`]):
    ///
    /// `Σ_i W(i) (A(i, column_point) + ρ B(i, column_point) + ρ² C(i, column_point))`.
    ///
    /// On a validated [`BlockSelectorLayout`] this runs in
    /// `O(2^K + num_row_vars + num_column_vars + runs)` field operations
    /// ([`Self::evaluate_block_selector_layout`]); on any other matrix shape
    /// it is the streamed or materialized sparse evaluation of the same
    /// value. Every input is validated against the prepared field before
    /// use.
    pub fn evaluate_product(
        &self,
        functional: &ProductRowFunctional<'_, F>,
        rho: &F,
        column_point: &[F],
    ) -> Result<F, SpartanMatrixError> {
        let skip_vars = functional.skip_vars;
        if skip_vars > self.matrices.num_row_vars() {
            return Err(SpartanMatrixError::InvalidRowPointLength {
                expected: self.matrices.num_row_vars(),
                actual: skip_vars,
            });
        }
        let expected_prefix = domain_size(skip_vars)?;
        if functional.prefix.len() != expected_prefix {
            return Err(SpartanMatrixError::InvalidRowWeightsLength {
                expected: expected_prefix,
                actual: functional.prefix.len(),
            });
        }
        let tail_vars = self.matrices.num_row_vars() - skip_vars;
        if functional.tail_point.len() != tail_vars {
            return Err(SpartanMatrixError::InvalidRowPointLength {
                expected: tail_vars,
                actual: functional.tail_point.len(),
            });
        }
        if column_point.len() != self.matrices.num_column_vars() {
            return Err(SpartanMatrixError::InvalidColumnPointLength {
                expected: self.matrices.num_column_vars(),
                actual: column_point.len(),
            });
        }
        validate_elements_field(functional.prefix, self.matrices.field_modulus_encoding())?;
        validate_elements_field(
            functional.tail_point,
            self.matrices.field_modulus_encoding(),
        )?;
        validate_elements_field(column_point, self.matrices.field_modulus_encoding())?;
        validate_element_field(rho, self.matrices.field_modulus_encoding())?;

        if let Some(layout) = self.matrices.block_selector() {
            return self.evaluate_block_selector_layout(layout, functional, rho, column_point);
        }

        // Generic matrices: the sparse reference evaluation under the
        // materialized (or, for disjoint unit selectors, streamed) weights.
        if skip_vars == 0 {
            let row_weights = eq_table(functional.tail_point, self.matrices.config())?;
            return self
                .matrices
                .binding(rho)
                .evaluate_at(&row_weights, column_point)
                .map_err(binding_matrix_error);
        }
        let (tail_low, tail_high) =
            make_equality_factors(functional.tail_point, self.matrices.config())?;
        let factors = PrefixUnivariateRowFactors::new(
            skip_vars,
            functional.prefix.to_vec(),
            tail_low,
            tail_high,
            self.matrices.num_row_vars(),
        )?;
        self.evaluate_prefix(&factors, rho, column_point)
    }
    /// The succinct evaluation behind
    /// [`Self::evaluate_product`] on a validated
    /// [`BlockSelectorLayout`].
    ///
    /// Every matrix is `Σ_runs coefficient · Sel(start)` with
    /// `Sel(start)[i, start + i] = 1` for `i < rows`, where `start` is a
    /// multiple of the power-of-two `block_len ≥ rows` and
    /// `start + rows ≤ columns ≤ 2^num_column_vars` (the detector's
    /// invariants, re-checked here). With `n = num_row_vars` we have
    /// `rows ≤ 2^n ≤ block_len = 2^ℓ`, so the column index `start + i` of a
    /// run entry has bits `[0, n)` equal to `i`, bits `[n, ℓ)` zero and bits
    /// `[ℓ, num_column_vars)` equal to `start >> ℓ`. For the little-endian
    /// equality table at `y = column_point` this gives
    ///
    /// `eq(y, start + i) = eq(y[..n], i) · Π_{k ∈ [n, ℓ)} (1 − y_k) · eq(y[ℓ..], start >> ℓ)`
    ///
    /// and the batched evaluation factors as
    ///
    /// `S · Z · Σ_{M ∈ {A, B, C}} f_M · Σ_{runs of M} coefficient · eq(y[ℓ..], start >> ℓ)`
    ///
    /// with `S = Σ_{i < rows} W(i) eq(y[..n], i)` ([`product_row_prefix_sum`]),
    /// `Z = Π_{k ∈ [n, ℓ)} (1 − y_k)` and `(f_A, f_B, f_C) = (1, ρ, ρ²)`. The
    /// result is the same field element as the sparse evaluation.
    fn evaluate_block_selector_layout(
        &self,
        layout: &BlockSelectorLayout<C>,
        functional: &ProductRowFunctional<'_, F>,
        rho: &F,
        column_point: &[F],
    ) -> Result<F, SpartanMatrixError> {
        let field_config = self.matrices.config();
        let num_row_vars = self.matrices.num_row_vars();
        let rows = layout.rows;
        let block_len = layout.block_len;
        let columns = self.matrices.matrices().column_count();
        let log_block_len = block_len.trailing_zeros() as usize;
        if rows == 0
            || rows != self.matrices.matrices().row_count()
            || !block_len.is_power_of_two()
            || rows > block_len
            || domain_size(num_row_vars)? > block_len
            || log_block_len > self.matrices.num_column_vars()
            || column_point.len() != self.matrices.num_column_vars()
        {
            return Err(SpartanMatrixError::InvalidMleOperation);
        }

        let one = F::one_with_cfg(field_config);
        let zero = F::zero_with_cfg(field_config);
        let (low_point, rest) = column_point.split_at(num_row_vars);
        let (padding_point, high_point) = rest.split_at(log_block_len - num_row_vars);

        let row_sum = product_row_prefix_sum(functional, rows, low_point, field_config)?;
        let mut padding = one.clone();
        for coordinate in padding_point {
            padding = (field_config).mul(&padding, &(field_config).sub(&one, coordinate));
        }

        let run_sum = |runs: &[SelectorRun<C>]| -> Result<F, SpartanMatrixError> {
            let mut sum = zero.clone();
            for run in runs {
                if run.start % block_len != 0
                    || run.start.checked_add(rows).is_none_or(|end| end > columns)
                {
                    return Err(SpartanMatrixError::InvalidMleOperation);
                }
                let block_weight =
                    eq_at_boolean_index(high_point, run.start >> log_block_len, field_config)?;
                sum = field_config.add(
                    &(sum),
                    &(&run.coefficient.scale(&block_weight, field_config)),
                );
            }
            Ok(sum)
        };
        let mut batched = run_sum(&layout.a)?;
        batched = field_config.add(
            &(batched),
            &(&(field_config).mul(rho, &run_sum(&layout.b)?)),
        );
        batched = field_config.add(
            &(batched),
            &(&(field_config).mul(&(field_config).mul(rho, rho), &run_sum(&layout.c)?)),
        );
        Ok((field_config).mul(&(field_config).mul(&row_sum, &padding), &batched))
    }
    /// Evaluates the batched matrices against a three-factor
    /// prefix-univariate row functional without constructing its tensor
    /// product when the prepared selector layout supports row streaming.
    pub fn evaluate_prefix(
        &self,
        factors: &PrefixUnivariateRowFactors<F>,
        rho: &F,
        column_point: &[F],
    ) -> Result<F, SpartanMatrixError> {
        if factors.parts().num_row_vars != self.matrices.num_row_vars() {
            return Err(SpartanMatrixError::InvalidRowWeightsLength {
                expected: domain_size(self.matrices.num_row_vars())?,
                actual: domain_size(factors.parts().num_row_vars)?,
            });
        }
        if column_point.len() != self.matrices.num_column_vars() {
            return Err(SpartanMatrixError::InvalidColumnPointLength {
                expected: self.matrices.num_column_vars(),
                actual: column_point.len(),
            });
        }

        let Some([rows, a_offset, b_offset, c_offset]) = self.matrices.selector_layout() else {
            let row_weights = factors.materialize(self.matrices.config());
            return self
                .matrices
                .binding(rho)
                .evaluate_at(&row_weights, column_point)
                .map_err(binding_matrix_error);
        };

        let column_weights = eq_table(column_point, self.matrices.config())?;
        let zero = F::zero_with_cfg(self.matrices.config());
        let mut a_evaluation = zero.clone();
        let mut b_evaluation = zero.clone();
        let mut c_evaluation = zero;
        factors.for_each_row(
            rows,
            |row, weight| {
                a_evaluation = self.matrices.config().add(
                    &(a_evaluation),
                    &(&(self.matrices.config()).mul(&weight, &column_weights[a_offset + row])),
                );
                b_evaluation = self.matrices.config().add(
                    &(b_evaluation),
                    &(&(self.matrices.config()).mul(&weight, &column_weights[b_offset + row])),
                );
                c_evaluation = self.matrices.config().add(
                    &(c_evaluation),
                    &(&(self.matrices.config()).mul(&weight, &column_weights[c_offset + row])),
                );
            },
            self.matrices.config(),
        );

        let rho_squared = (self.matrices.config()).mul(rho, rho);
        a_evaluation = self.matrices.config().add(
            &(a_evaluation),
            &(&(self.matrices.config()).mul(rho, &b_evaluation)),
        );
        a_evaluation = self.matrices.config().add(
            &(a_evaluation),
            &(&(self.matrices.config()).mul(&rho_squared, &c_evaluation)),
        );
        Ok(a_evaluation)
    }
}
fn binding_matrix_error(error: crate::sumcheck::SumcheckError) -> SpartanMatrixError {
    match error {
        crate::sumcheck::SumcheckError::Matrix(error) => error,
        _ => SpartanMatrixError::InvalidMleOperation,
    }
}
