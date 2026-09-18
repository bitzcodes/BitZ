//! Parallel sumcheck protocol bound on `SumcheckEngine`, used by the
//! dual-field IntMod-Spartan driver (`imod_spartan_modp`). Lives alongside `src/sumcheck.rs` rather than
//! replacing it, so the field-generic stack stays untouched and the two
//! implementations can be differentially tested; a later unification
//! remains possible.
//!
//! Deliberately minimal: no BDDT round-0 optimization, no Gruen
//! eq-factoring, no delayed-reduction Montgomery accumulators. Just the
//! protocol math, easy to read and verify. Optimizations can be added
//! later if benchmarks demand.
//!
//! All constructors of `Self::Scalar`
//! (`zero`, `one`, `from_u64`) take the field's `Params` context, which
//! comes from the polynomials passed in (each `MultilinearPolynomial`
//! carries its own `Params`).

use crate::{
  errors::SpartanError,
  polys_modp::{
    eq::EqPolynomial,
    multilinear::MultilinearPolynomial,
    univariate::{CompressedUniPoly, UniPoly},
  },
  traits::{
    mod_engine::{SumcheckEngine, SumcheckField},
    transcript::{ByteTranscript, TranscriptEngineTrait},
  },
};
use rayon::prelude::*;

/// A sumcheck proof — the per-round compressed univariate polynomials.
#[derive(Clone, Debug)]
pub struct SumcheckProof<E: SumcheckEngine> {
  pub(crate) compressed_polys: Vec<CompressedUniPoly<E::Scalar>>,
}

impl<E: SumcheckEngine> SumcheckProof<E> {
  /// Construct from per-round compressed polys (used by `prove_*`).
  fn new(compressed_polys: Vec<CompressedUniPoly<E::Scalar>>) -> Self {
    Self { compressed_polys }
  }

  /// Whether every round polynomial's coefficients belong to the modulus
  /// context `expected`. `pub(crate)` because the caller
  /// (`imod_spartan_modp`) lives in a sibling module.
  pub(crate) fn is_in_context(&self, expected: &<E::Scalar as SumcheckField>::Params) -> bool {
    self
      .compressed_polys
      .iter()
      .all(|p| p.is_in_context(expected))
  }

  /// Verify the proof. Performs intra-round consistency
  /// (`p_i(0) + p_i(1) == running_claim`) and degree checks; returns
  /// `(final_claim, challenges)`. The caller is responsible for
  /// re-checking `final_claim` against the integrand's structure
  /// (e.g. for our outer SC, `eq(tau, r_x) * (v_a v_b - v_c - v_m v_q)`).
  ///
  /// `params` is the field's modulus context (`&()` for static fields).
  pub fn verify(
    &self,
    initial_claim: E::Scalar,
    num_rounds: usize,
    degree_bound: usize,
    params: &<E::Scalar as SumcheckField>::Params,
    transcript: &mut E::TE,
  ) -> Result<(E::Scalar, Vec<E::Scalar>), SpartanError> {
    if self.compressed_polys.len() != num_rounds {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    let mut r = Vec::with_capacity(num_rounds);
    let mut claim = initial_claim;
    let zero = E::Scalar::zero(params);
    let one = E::Scalar::one(params);

    for compressed in &self.compressed_polys {
      if compressed.degree() != degree_bound {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      let poly = compressed.decompress(&claim);

      // intra-round consistency: p(0) + p(1) == running claim
      let p0 = poly.evaluate(&zero);
      let p1 = poly.evaluate(&one);
      if p0 + p1 != claim {
        return Err(SpartanError::InvalidSumcheckProof);
      }

      transcript.absorb(b"p", &poly);
      let r_i = transcript.squeeze(b"c")?;
      claim = poly.evaluate(&r_i);
      r.push(r_i);
    }

    Ok((claim, r))
  }

  /// Degree-2 sumcheck on the integrand `A(x) * B(x)`. Returns
  /// `(proof, challenges, [A(r), B(r)])`.
  ///
  /// `poly_A` and `poly_B` must carry the same `Params` (the field's
  /// modulus context); the prover reads it from `poly_A`.
  #[allow(non_snake_case, clippy::too_many_arguments)]
  pub fn prove_quad(
    claim: &E::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    assert_eq!(poly_A.len(), poly_B.len());
    assert_eq!(poly_A.len(), 1 << num_rounds);

    let params = poly_A.params().clone();
    let zero = E::Scalar::zero(&params);
    let two = E::Scalar::from_u64(&params, 2);

    let mut r = Vec::with_capacity(num_rounds);
    let mut compressed_polys = Vec::with_capacity(num_rounds);
    let mut running_claim = *claim;

    for _ in 0..num_rounds {
      let n = poly_A.len() / 2;
      let (pa, pb) = (&*poly_A, &*poly_B);

      // Evaluate the round polynomial at X ∈ {0, 1, 2}, summing the per-
      // hypercube-point contributions in parallel (the DynPrime field ops
      // dominate, so this is the prover's hottest loop).
      let (eval_0, eval_1, eval_2) = (0..n)
        .into_par_iter()
        .map(|i| {
          let a0 = pa[i];
          let a1 = pa[n + i];
          let b0 = pb[i];
          let b1 = pb[n + i];

          let a2 = two * a1 - a0;
          let b2 = two * b1 - b0;
          (a0 * b0, a1 * b1, a2 * b2)
        })
        .reduce(
          || (zero, zero, zero),
          |(x0, x1, x2), (y0, y1, y2)| (x0 + y0, x1 + y1, x2 + y2),
        );

      debug_assert_eq!(eval_0 + eval_1, running_claim);

      let round_poly = UniPoly::from_evals(&[eval_0, eval_1, eval_2], &params)?;
      transcript.absorb(b"p", &round_poly);
      let r_i = transcript.squeeze(b"c")?;
      compressed_polys.push(round_poly.compress());

      poly_A.bind_poly_var_top(&r_i);
      poly_B.bind_poly_var_top(&r_i);

      running_claim = round_poly.evaluate(&r_i);
      r.push(r_i);
    }

    let claims = vec![poly_A[0], poly_B[0]];
    Ok((Self::new(compressed_polys), r, claims))
  }

  /// Degree-3 sumcheck on the imod outer-SC integrand
  ///   `eq(tau, x) * (A(x) * B(x) - C(x) - M(x) * Q(x))`.
  ///
  /// Returns `(proof, challenges, [v_a, v_b, v_c, v_m, v_q])`.
  /// All polynomials must carry the same `Params`.
  #[allow(non_snake_case, clippy::too_many_arguments)]
  pub fn prove_cubic_with_five_inputs(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    poly_C: &mut MultilinearPolynomial<E::Scalar>,
    poly_M: &mut MultilinearPolynomial<E::Scalar>,
    poly_Q: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    let num_rounds = taus.len();
    let expected_len = 1usize << num_rounds;
    assert_eq!(poly_A.len(), expected_len);
    assert_eq!(poly_B.len(), expected_len);
    assert_eq!(poly_C.len(), expected_len);
    assert_eq!(poly_M.len(), expected_len);
    assert_eq!(poly_Q.len(), expected_len);

    let params = poly_A.params().clone();
    let zero = E::Scalar::zero(&params);
    let two = E::Scalar::from_u64(&params, 2);
    let three = E::Scalar::from_u64(&params, 3);

    let eq_evals = EqPolynomial::evals_from_points(&taus, &params);
    let mut poly_eq = MultilinearPolynomial::new(eq_evals, params.clone());

    let mut r = Vec::with_capacity(num_rounds);
    let mut compressed_polys = Vec::with_capacity(num_rounds);
    let mut running_claim = *claim;

    for _ in 0..num_rounds {
      let n = poly_A.len() / 2;
      let (peq, pa, pb, pc, pm, pq) = (&poly_eq, &*poly_A, &*poly_B, &*poly_C, &*poly_M, &*poly_Q);

      // Evaluate the round polynomial at X ∈ {0, 1, 2, 3}, summing the
      // per-hypercube-point contributions in parallel.
      let (eval_0, eval_1, eval_2, eval_3) = (0..n)
        .into_par_iter()
        .map(|i| {
          let eq0 = peq[i];
          let eq1 = peq[n + i];
          let a0 = pa[i];
          let a1 = pa[n + i];
          let b0 = pb[i];
          let b1 = pb[n + i];
          let c0 = pc[i];
          let c1 = pc[n + i];
          let m0 = pm[i];
          let m1 = pm[n + i];
          let q0 = pq[i];
          let q1 = pq[n + i];

          let eq2 = two * eq1 - eq0;
          let eq3 = three * eq1 - two * eq0;
          let a2 = two * a1 - a0;
          let a3 = three * a1 - two * a0;
          let b2 = two * b1 - b0;
          let b3 = three * b1 - two * b0;
          let c2 = two * c1 - c0;
          let c3 = three * c1 - two * c0;
          let m2 = two * m1 - m0;
          let m3 = three * m1 - two * m0;
          let q2 = two * q1 - q0;
          let q3 = three * q1 - two * q0;

          (
            eq0 * (a0 * b0 - c0 - m0 * q0),
            eq1 * (a1 * b1 - c1 - m1 * q1),
            eq2 * (a2 * b2 - c2 - m2 * q2),
            eq3 * (a3 * b3 - c3 - m3 * q3),
          )
        })
        .reduce(
          || (zero, zero, zero, zero),
          |(x0, x1, x2, x3), (y0, y1, y2, y3)| (x0 + y0, x1 + y1, x2 + y2, x3 + y3),
        );

      debug_assert_eq!(eval_0 + eval_1, running_claim);

      let round_poly = UniPoly::from_evals(&[eval_0, eval_1, eval_2, eval_3], &params)?;
      transcript.absorb(b"p", &round_poly);
      let r_i = transcript.squeeze(b"c")?;
      compressed_polys.push(round_poly.compress());

      poly_eq.bind_poly_var_top(&r_i);
      poly_A.bind_poly_var_top(&r_i);
      poly_B.bind_poly_var_top(&r_i);
      poly_C.bind_poly_var_top(&r_i);
      poly_M.bind_poly_var_top(&r_i);
      poly_Q.bind_poly_var_top(&r_i);

      running_claim = round_poly.evaluate(&r_i);
      r.push(r_i);
    }

    let claims = vec![poly_A[0], poly_B[0], poly_C[0], poly_M[0], poly_Q[0]];
    Ok((Self::new(compressed_polys), r, claims))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::PallasHyraxEngine;
  use crate::traits::Engine;
  use ff::Field;
  use rand::SeedableRng;
  use rand::rngs::StdRng;

  type E = PallasHyraxEngine;
  type F = <E as Engine>::Scalar;

  fn rand_vec(n: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| F::random(&mut rng)).collect()
  }

  #[test]
  #[allow(non_snake_case)]
  fn prove_quad_roundtrips() {
    let num_rounds = 4usize;
    let n = 1usize << num_rounds;
    let A = rand_vec(n, 1);
    let B = rand_vec(n, 2);
    let claim: F = A
      .iter()
      .zip(B.iter())
      .fold(F::ZERO, |acc, (a, b)| acc + *a * *b);

    let mut poly_A = MultilinearPolynomial::new(A, ());
    let mut poly_B = MultilinearPolynomial::new(B, ());

    let mut pt = <E as Engine>::TE::new(b"test");
    let (proof, r_prover, final_evals) =
      SumcheckProof::<E>::prove_quad(&claim, num_rounds, &mut poly_A, &mut poly_B, &mut pt)
        .unwrap();

    let mut vt = <E as Engine>::TE::new(b"test");
    let (final_claim, r_verifier) = proof.verify(claim, num_rounds, 2, &(), &mut vt).unwrap();

    assert_eq!(r_prover, r_verifier);
    assert_eq!(final_claim, final_evals[0] * final_evals[1]);
  }

  #[test]
  #[allow(non_snake_case)]
  fn prove_cubic_with_five_inputs_roundtrips_on_satisfying_witness() {
    let num_rounds = 3usize;
    let n = 1usize << num_rounds;
    let A = rand_vec(n, 10);
    let B = rand_vec(n, 11);
    let M = rand_vec(n, 12);
    let Q = rand_vec(n, 13);
    let C: Vec<F> = (0..n).map(|i| A[i] * B[i] - M[i] * Q[i]).collect();

    let taus = rand_vec(num_rounds, 20);

    let mut poly_A = MultilinearPolynomial::new(A, ());
    let mut poly_B = MultilinearPolynomial::new(B, ());
    let mut poly_C = MultilinearPolynomial::new(C, ());
    let mut poly_M = MultilinearPolynomial::new(M, ());
    let mut poly_Q = MultilinearPolynomial::new(Q, ());

    let mut pt = <E as Engine>::TE::new(b"test");
    let (proof, r_prover, finals) = SumcheckProof::<E>::prove_cubic_with_five_inputs(
      &F::ZERO,
      taus.clone(),
      &mut poly_A,
      &mut poly_B,
      &mut poly_C,
      &mut poly_M,
      &mut poly_Q,
      &mut pt,
    )
    .unwrap();

    let mut vt = <E as Engine>::TE::new(b"test");
    let (final_claim, r_verifier) = proof.verify(F::ZERO, num_rounds, 3, &(), &mut vt).unwrap();

    assert_eq!(r_prover, r_verifier);

    let (va, vb, vc, vm, vq) = (finals[0], finals[1], finals[2], finals[3], finals[4]);
    let eq_tau_r = EqPolynomial::new(taus, ()).evaluate(&r_verifier);
    let expected = eq_tau_r * (va * vb - vc - vm * vq);
    assert_eq!(final_claim, expected);
  }

  #[test]
  #[allow(non_snake_case)]
  fn verify_rejects_tampered_proof() {
    let num_rounds = 3usize;
    let n = 1usize << num_rounds;
    let A = rand_vec(n, 30);
    let B = rand_vec(n, 31);
    let claim: F = A
      .iter()
      .zip(B.iter())
      .fold(F::ZERO, |acc, (a, b)| acc + *a * *b);

    let mut poly_A = MultilinearPolynomial::new(A, ());
    let mut poly_B = MultilinearPolynomial::new(B, ());

    let mut pt = <E as Engine>::TE::new(b"test");
    let (proof, _r, _finals) =
      SumcheckProof::<E>::prove_quad(&claim, num_rounds, &mut poly_A, &mut poly_B, &mut pt)
        .unwrap();

    let mut short = proof.clone();
    short.compressed_polys.pop();
    let mut vt = <E as Engine>::TE::new(b"test");
    assert!(short.verify(claim, num_rounds, 2, &(), &mut vt).is_err());

    let mut wrong_deg = proof;
    wrong_deg.compressed_polys[0]
      .coeffs_except_linear_term
      .pop();
    let mut vt2 = <E as Engine>::TE::new(b"test");
    assert!(
      wrong_deg
        .verify(claim, num_rounds, 2, &(), &mut vt2)
        .is_err()
    );
  }

  /// The load-bearing dynamic-field test: run the 5-input cubic sumcheck over the
  /// dynamic-prime field `DynPrime<2>` (not a static curve scalar). Same
  /// shape as `prove_cubic_with_five_inputs_roundtrips_on_satisfying_witness`
  /// but with `Scalar = DynPrime<2>` parameterized by a runtime 256-bit
  /// prime. Exercises DynPrime arithmetic + transcript-over-DynPrime +
  /// the polys_modp types end-to-end.
  #[test]
  #[allow(non_snake_case)]
  fn prove_cubic_over_dynprime_roundtrips() {
    use crate::dyn_prime::DynPrime;
    use crate::provider::T256DynPrimeEngine;
    use crypto_bigint::{Odd, U128, modular::FixedMontyParams};

    type ME = T256DynPrimeEngine;
    type DP = DynPrime<2>;

    // Mersenne prime 2^127 - 1: prime (so 2/3/6 are invertible) and odd
    // (required for Montgomery). Stands in for a verifier-sampled prime.
    let modulus = U128::from_be_hex("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    let params = FixedMontyParams::new(Odd::new(modulus).unwrap());

    let num_rounds = 3usize;
    let n = 1usize << num_rounds;
    let mk = |seed: u64| -> Vec<DP> {
      (0..n)
        .map(|i| DP::from_u64(&params, seed.wrapping_mul(i as u64 + 1).wrapping_add(1)))
        .collect()
    };
    let A = mk(7);
    let B = mk(11);
    let M = mk(13);
    let Q = mk(17);
    let C: Vec<DP> = (0..n).map(|i| A[i] * B[i] - M[i] * Q[i]).collect();
    let taus: Vec<DP> = (0..num_rounds)
      .map(|i| DP::from_u64(&params, (i as u64) + 5))
      .collect();

    let mut poly_A = MultilinearPolynomial::new(A, params);
    let mut poly_B = MultilinearPolynomial::new(B, params);
    let mut poly_C = MultilinearPolynomial::new(C, params);
    let mut poly_M = MultilinearPolynomial::new(M, params);
    let mut poly_Q = MultilinearPolynomial::new(Q, params);

    let mut pt = <ME as SumcheckEngine>::TE::new_with_params(b"test", params);
    let (proof, r_prover, finals) = SumcheckProof::<ME>::prove_cubic_with_five_inputs(
      &DP::zero(&params),
      taus.clone(),
      &mut poly_A,
      &mut poly_B,
      &mut poly_C,
      &mut poly_M,
      &mut poly_Q,
      &mut pt,
    )
    .unwrap();

    let mut vt = <ME as SumcheckEngine>::TE::new_with_params(b"test", params);
    let (final_claim, r_verifier) = proof
      .verify(DP::zero(&params), num_rounds, 3, &params, &mut vt)
      .unwrap();

    assert_eq!(r_prover, r_verifier);

    let (va, vb, vc, vm, vq) = (finals[0], finals[1], finals[2], finals[3], finals[4]);
    let eq_tau_r = EqPolynomial::new(taus, params).evaluate(&r_verifier);
    let expected = eq_tau_r * (va * vb - vc - vm * vq);
    assert_eq!(final_claim, expected);
  }
}
