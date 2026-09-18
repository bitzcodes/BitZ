//! Equality polynomial `eq(r, x) = ∏_i (r_i x_i + (1-r_i)(1-x_i))`.

use crate::traits::mod_engine::SumcheckField;

/// Equality polynomial parameterized by a vector `r`. Carries the field's
/// modulus context so that constructors like `F::one(&params)` can be
/// called without an external context argument.
pub struct EqPolynomial<F: SumcheckField> {
  /// The point `r` the polynomial is parameterized by.
  pub r: Vec<F>,
  pub(crate) params: F::Params,
}

impl<F: SumcheckField> EqPolynomial<F> {
  /// Construct an `EqPolynomial` with the given parameter vector and the
  /// field's modulus context.
  pub fn new(r: Vec<F>, params: F::Params) -> Self {
    Self { r, params }
  }

  /// Access the stored field modulus context.
  pub fn params(&self) -> &F::Params {
    &self.params
  }

  /// Evaluate `eq(self.r, rx)` for a single point `rx`.
  pub fn evaluate(&self, rx: &[F]) -> F {
    assert_eq!(self.r.len(), rx.len());
    let one = F::one(&self.params);
    // Fold (rather than `.product()`) because dynamic-modulus fields
    // can't supply a context-free `Product::product` identity.
    rx.iter().zip(self.r.iter()).fold(one, |acc, (rxi, ri)| {
      acc * (*rxi * *ri + (one - *rxi) * (one - *ri))
    })
  }

  /// Build the full eq-table: `evals[k] = eq(r, k)` for every
  /// `k ∈ {0,1}^{|r|}`, returned as a length-`2^|r|` vector.
  ///
  /// **Bit-ordering convention** (matches `src/polys/eq.rs`): `r` is
  /// consumed in **reverse** so that the resulting table aligns with
  /// the top-down binding order used by `MultilinearPolynomial::
  /// bind_poly_var_top`. After binding the top variable to challenge
  /// `r_0` (round 0), then `r_1` (round 1), …, the final value
  /// `evals[0]` of the bound table equals `eq(r, [r_0, r_1, …, r_{n-1}])`
  /// — i.e. the natural `EqPolynomial::new(r, ...).evaluate(&challenges)`
  /// without reversal.
  pub fn evals_from_points(r: &[F], params: &F::Params) -> Vec<F> {
    let one = F::one(params);
    let zero = F::zero(params);
    let mut evals = vec![zero; 1usize << r.len()];
    evals[0] = one;
    let mut size = 1usize;
    for &r_i in r.iter().rev() {
      let one_minus_r = one - r_i;
      for i in 0..size {
        let v = evals[i];
        evals[i] = v * one_minus_r;
        evals[size + i] = v * r_i;
      }
      size *= 2;
    }
    evals
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Engine;
  use ff::Field;

  type F = <PallasHyraxEngine as Engine>::Scalar;

  #[test]
  fn eq_table_consistent_with_evaluate() {
    let r: Vec<F> = (0..4).map(|i| F::from((7 + i) as u64)).collect();
    let table = EqPolynomial::evals_from_points(&r, &());
    let eq = EqPolynomial::new(r.clone(), ());
    let n = r.len();
    for (k, &table_val) in table.iter().enumerate() {
      let x: Vec<F> = (0..n)
        .map(|i| {
          if (k >> (n - 1 - i)) & 1 == 1 {
            F::ONE
          } else {
            F::ZERO
          }
        })
        .collect();
      assert_eq!(eq.evaluate(&x), table_val);
    }
  }

  #[test]
  fn eq_table_sums_to_one() {
    let r: Vec<F> = (0..5).map(|i| F::from((3 * i + 1) as u64)).collect();
    let table = EqPolynomial::evals_from_points(&r, &());
    let sum: F = table.iter().copied().fold(F::ZERO, |a, b| a + b);
    assert_eq!(sum, F::ONE);
  }
}
