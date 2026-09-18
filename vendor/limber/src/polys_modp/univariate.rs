//! Dense univariate polynomial and its compressed (omit-linear-term)
//! form, used for sumcheck round polynomials.

use crate::{
  errors::SpartanError,
  traits::{mod_engine::SumcheckField, transcript::TranscriptReprTrait},
};

/// Univariate dense polynomial stored in **big-endian** order:
/// `coeffs[0]` is the constant term, `coeffs[k]` is the `x^k` term.
///
/// Does **not** carry `Params` directly: all operations work in terms of
/// existing field values (the coefficients), so they don't need to
/// materialize fresh field elements from a modulus context. The single
/// constructor that does (`from_evals`, for the `1/2` and `1/6` constants
/// in Lagrange interpolation) takes `params` as an argument.
#[derive(Clone, Debug)]
pub struct UniPoly<F: SumcheckField> {
  /// Coefficients, lowest degree first.
  pub coeffs: Vec<F>,
}

/// `UniPoly` with the linear coefficient (`coeffs[1]`) omitted.
/// Recoverable from the round's running claim via
/// `p(0) + p(1) = claim`.
#[derive(Clone, Debug)]
pub struct CompressedUniPoly<F: SumcheckField> {
  /// `coeffs` with index 1 removed.
  pub coeffs_except_linear_term: Vec<F>,
}

impl<F: SumcheckField> UniPoly<F> {
  /// New from coefficient vector (big-endian).
  pub fn new(coeffs: Vec<F>) -> Self {
    Self { coeffs }
  }

  /// Construct from evaluations at consecutive points `x = 0, 1, …, n-1`.
  /// Supports degree-2 (3 evals) and degree-3 (4 evals) — those are the
  /// shapes sumcheck round polynomials take.
  pub fn from_evals(evals: &[F], params: &F::Params) -> Result<Self, SpartanError> {
    let two_inv = F::from_u64(params, 2)
      .invert()
      .ok_or(SpartanError::InternalError {
        reason: "2 is not invertible in this field".to_string(),
      })?;
    match evals.len() {
      3 => {
        // p(x) = a x^2 + b x + c, evaluated at 0, 1, 2.
        // a = (p(0) - 2 p(1) + p(2)) / 2
        // b = p(1) - p(0) - a
        // c = p(0)
        let c = evals[0];
        let two_p1 = evals[1] + evals[1];
        let a = (evals[0] - two_p1 + evals[2]) * two_inv;
        let b = evals[1] - evals[0] - a;
        Ok(Self {
          coeffs: vec![c, b, a],
        })
      }
      4 => {
        // Degree-3 Lagrange via finite differences (avoids 4×4 linear
        // solve): Δ_i = p(i+1) - p(i); ΔΔ_i = Δ_{i+1} - Δ_i; ΔΔΔ = 6a.
        let six_inv = F::from_u64(params, 6)
          .invert()
          .ok_or(SpartanError::InternalError {
            reason: "6 is not invertible in this field".to_string(),
          })?;
        let d = evals[0];
        let delta1 = evals[1] - evals[0];
        let delta2 = evals[2] - evals[1];
        let delta3 = evals[3] - evals[2];
        let dd1 = delta2 - delta1;
        let dd2 = delta3 - delta2;
        let ddd = dd2 - dd1;
        let a = ddd * six_inv;
        let three_a = a + a + a;
        let b = dd1 * two_inv - three_a;
        let c = delta1 - a - b;
        Ok(Self {
          coeffs: vec![d, c, b, a],
        })
      }
      n => Err(SpartanError::InternalError {
        reason: format!("UniPoly::from_evals supports degree 2 or 3 only, got {n} evals"),
      }),
    }
  }

  /// Polynomial degree.
  pub fn degree(&self) -> usize {
    self.coeffs.len().saturating_sub(1)
  }

  /// Evaluate at `x` via Horner's rule.
  ///
  /// Starts the accumulator from the top-degree coefficient so we don't
  /// need a context-free `F::zero()`.
  pub fn evaluate(&self, x: &F) -> F {
    assert!(!self.coeffs.is_empty(), "empty UniPoly");
    let mut iter = self.coeffs.iter().rev();
    let mut acc = *iter.next().unwrap();
    for c in iter {
      acc = acc * *x + *c;
    }
    acc
  }

  /// Drop the linear coefficient. Recoverable from the running claim.
  pub fn compress(&self) -> CompressedUniPoly<F> {
    assert!(
      self.coeffs.len() >= 2,
      "compress requires at least a linear term to omit"
    );
    let mut omitted = Vec::with_capacity(self.coeffs.len() - 1);
    omitted.push(self.coeffs[0]);
    omitted.extend_from_slice(&self.coeffs[2..]);
    CompressedUniPoly {
      coeffs_except_linear_term: omitted,
    }
  }
}

impl<F: SumcheckField> CompressedUniPoly<F> {
  /// Recover the full polynomial given the previous round's claim
  /// `hint = p(0) + p(1)`.
  ///
  /// `coeffs[1] = hint - 2·coeffs[0] - (coeffs[2] + … + coeffs[k])`.
  pub fn decompress(&self, hint: &F) -> UniPoly<F> {
    assert!(
      !self.coeffs_except_linear_term.is_empty(),
      "compressed polynomial must contain at least the constant term"
    );
    let c0 = self.coeffs_except_linear_term[0];
    // Sum of high-degree coefficients (everything except c0 and the
    // omitted c1). Fold rather than `.sum()` because dynamic-modulus
    // fields can't supply a context-free `Sum::sum` identity. Start
    // the accumulator from a value-derived zero (`c0 - c0`) so we
    // don't need `F::Params` plumbed into `CompressedUniPoly`.
    #[allow(clippy::eq_op)]
    let mut high_sum = c0 - c0;
    for c in &self.coeffs_except_linear_term[1..] {
      high_sum += *c;
    }
    let c1 = *hint - c0 - c0 - high_sum;

    let mut coeffs = Vec::with_capacity(self.coeffs_except_linear_term.len() + 1);
    coeffs.push(c0);
    coeffs.push(c1);
    coeffs.extend_from_slice(&self.coeffs_except_linear_term[1..]);
    UniPoly { coeffs }
  }

  /// Number of stored coefficients (= `degree + 1` of the original poly,
  /// minus one for the omitted linear term).
  pub fn degree(&self) -> usize {
    self.coeffs_except_linear_term.len()
  }

  /// Whether every stored coefficient belongs to the modulus context
  /// `expected`. `pub(crate)` (not private): the callers live in the
  /// sibling modules `sumcheck_modp` and `imod_spartan_modp`, where a
  /// private method would be unreachable.
  pub(crate) fn is_in_context(&self, expected: &F::Params) -> bool {
    self
      .coeffs_except_linear_term
      .iter()
      .all(|c| c.is_in_context(expected))
  }
}

impl<F: SumcheckField> TranscriptReprTrait for UniPoly<F> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // Absorb the compressed coefficients so the bytes match what the
    // verifier reconstructs.
    self
      .compress()
      .coeffs_except_linear_term
      .iter()
      .flat_map(|c| c.to_le_bytes())
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Engine;

  type F = <PallasHyraxEngine as Engine>::Scalar;

  #[test]
  fn from_evals_then_evaluate_roundtrips_degree_2() {
    let coeffs = vec![F::from(7), F::from(11), F::from(13)];
    let original = UniPoly::new(coeffs.clone());
    let evals: Vec<F> = (0..3).map(|i| original.evaluate(&F::from(i))).collect();
    let recovered = UniPoly::from_evals(&evals, &()).unwrap();
    assert_eq!(recovered.coeffs, coeffs);
  }

  #[test]
  fn from_evals_then_evaluate_roundtrips_degree_3() {
    let coeffs = vec![F::from(2), F::from(3), F::from(5), F::from(7)];
    let original = UniPoly::new(coeffs.clone());
    let evals: Vec<F> = (0..4).map(|i| original.evaluate(&F::from(i))).collect();
    let recovered = UniPoly::from_evals(&evals, &()).unwrap();
    assert_eq!(recovered.coeffs, coeffs);
  }

  #[test]
  fn compress_decompress_roundtrip() {
    let coeffs = vec![F::from(7), F::from(11), F::from(13), F::from(17)];
    let p = UniPoly::new(coeffs.clone());
    let compressed = p.compress();
    let claim = p.evaluate(&F::from_u64(&(), 0)) + p.evaluate(&F::from_u64(&(), 1));
    let recovered = compressed.decompress(&claim);
    assert_eq!(recovered.coeffs, coeffs);
  }
}
