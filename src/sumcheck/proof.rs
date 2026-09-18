use super::{SumcheckError, boundary::*};
use crate::piop::spartan::{
    SpartanField, absorb_field_elements, grinding::GrindingDomain, squeeze_field,
};
use crate::transcript::traits::Transcript;
use field::RingOps;
/// Sumcheck round polynomials in coefficient form.
///
/// `COEFFS` is the maximum degree plus one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof<F, const COEFFS: usize> {
    pub round_polynomials: Vec<[F; COEFFS]>,
}

impl<F, const COEFFS: usize> SumcheckProof<F, COEFFS>
where
    F: SpartanField,
{
    /// Verifies all round reductions and returns `(evaluation_point, final_claim)`.
    ///
    /// The protocol using the generic reduction remains responsible for its
    /// terminal identity.
    pub fn verify(
        &self,
        transcript: &mut impl Transcript,
        initial_claim: F,
        expected_rounds: usize,
        field_cfg: &F::Config,
    ) -> Result<(Vec<F>, F), SumcheckError> {
        let mut round_boundary = UngrindedRoundBoundary;
        self.verify_with_round_boundary(
            transcript,
            initial_claim,
            expected_rounds,
            field_cfg,
            &mut round_boundary,
        )
    }

    /// Verifies a sumcheck with one typed proof-of-work boundary between each
    /// absorbed round polynomial and its Fiat--Shamir challenge.
    pub(crate) fn verify_grinded<D: GrindingDomain>(
        &self,
        transcript: &mut impl Transcript,
        initial_claim: F,
        expected_rounds: usize,
        field_cfg: &F::Config,
        grinding_nonces: &[u64],
        grinding_bits: u32,
    ) -> Result<(Vec<F>, F), SumcheckError> {
        let mut round_boundary =
            VerifierGrindingRoundBoundary::<D>::new(grinding_bits, grinding_nonces);
        self.verify_with_round_boundary(
            transcript,
            initial_claim,
            expected_rounds,
            field_cfg,
            &mut round_boundary,
        )
    }

    pub fn verify_with_round_boundary<P>(
        &self,
        transcript: &mut impl Transcript,
        initial_claim: F,
        expected_rounds: usize,
        field_cfg: &F::Config,
        round_boundary: &mut P,
    ) -> Result<(Vec<F>, F), SumcheckError>
    where
        P: RoundBoundaryPolicy,
    {
        let (point, [claim]) = Self::verify_batch_with_round_boundary(
            [self],
            transcript,
            &[initial_claim],
            expected_rounds,
            field_cfg,
            round_boundary,
        )?;
        Ok((point, claim))
    }

    /// Verify separate claims sharing each round challenge. The caller still
    /// discharges every terminal evaluation against its matrix/PCS statement.
    pub fn verify_batch_with_round_boundary<const K: usize>(
        proofs: [&Self; K],
        transcript: &mut impl Transcript,
        initial_claims: &[F; K],
        expected_rounds: usize,
        field: &F::Config,
        boundary: &mut impl RoundBoundaryPolicy,
    ) -> Result<(Vec<F>, [F; K]), SumcheckError> {
        if K == 0 {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        if COEFFS == 0 {
            return Err(SumcheckError::EmptyRoundPolynomial);
        }
        for proof in proofs {
            if proof.round_polynomials.len() != expected_rounds {
                return Err(SumcheckError::InvalidRoundCount {
                    expected: expected_rounds,
                    actual: proof.round_polynomials.len(),
                });
            }
        }
        boundary.validate(expected_rounds)?;
        validate_field_elements(initial_claims, field)?;
        for proof in proofs {
            for coefficients in &proof.round_polynomials {
                validate_field_elements(coefficients, field)?;
            }
        }
        let zero = field.zero();
        let mut claims = *initial_claims;
        let mut point = Vec::with_capacity(expected_rounds);
        for round in 0..expected_rounds {
            for (proof, claim) in proofs.iter().zip(&claims) {
                let coefficients = &proof.round_polynomials[round];
                absorb_field_elements(transcript, coefficients, field);
                let at_one = coefficients.iter().fold(zero, |sum, c| field.add(&sum, c));
                if field.add(&coefficients[0], &at_one) != *claim {
                    return Err(SumcheckError::InvalidRoundClaim { round });
                }
            }
            boundary.after_round(transcript, round)?;
            let challenge = squeeze_field(transcript, field)?;
            for (proof, claim) in proofs.iter().zip(&mut claims) {
                *claim =
                    evaluate_polynomial(&proof.round_polynomials[round], &challenge, &zero, field);
            }
            point.push(challenge);
        }
        Ok((point, claims))
    }
}

/// Local output produced while writing a sumcheck proof to the transcript.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SumcheckProverOutput<F, const COEFFS: usize> {
    pub proof: SumcheckProof<F, COEFFS>,
    pub eval_points: Vec<F>,
    pub final_claim: F,
}
/// Proof of the equality-weighted R1CS residual sum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckProof<F> {
    /// Cubic rounds in coefficient form `[c0, c1, c2, c3]`.
    pub sumcheck: SumcheckProof<F, 4>,
    pub az_mle_claim: F,
    pub bz_mle_claim: F,
    pub cz_mle_claim: F,
}

/// Prover-local result of the outer sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OuterSumcheckOutput<F> {
    pub proof: OuterSumcheckProof<F>,
    pub eval_points: Vec<F>,
    pub final_claim: F,
}

/// Transcript-derived point and terminal evaluations returned by verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OuterSumcheckVerifierOutput<F> {
    pub eval_points: Vec<F>,
    pub az_mle_claim: F,
    pub bz_mle_claim: F,
    pub cz_mle_claim: F,
}

impl<F> OuterSumcheckProof<F>
where
    F: SpartanField,
{
    /// Verifies the outer reduction and its terminal R1CS identity.
    pub(crate) fn verify(
        &self,
        transcript: &mut impl Transcript,
        initial_claim: F,
        tau: &[F],
        field_cfg: &F::Config,
    ) -> Result<OuterSumcheckVerifierOutput<F>, SumcheckError> {
        let mut round_boundary = UngrindedRoundBoundary;
        self.verify_with_round_boundary(
            transcript,
            initial_claim,
            tau,
            field_cfg,
            &mut round_boundary,
        )
    }

    /// Verifies a cubic outer sumcheck with one PoW nonce adjacent to every
    /// round polynomial.
    ///
    /// The nonce at index `i` is checked after round polynomial `i` is
    /// absorbed and before challenge `i` is sampled. `D` supplies the typed,
    /// protocol-specific grinding domain; `grinding_bits` is fixed across all
    /// rounds in this proof.
    #[allow(dead_code)]
    pub(crate) fn verify_grinded<D: GrindingDomain>(
        &self,
        transcript: &mut impl Transcript,
        initial_claim: F,
        tau: &[F],
        field_cfg: &F::Config,
        grinding_nonces: &[u64],
        grinding_bits: u32,
    ) -> Result<OuterSumcheckVerifierOutput<F>, SumcheckError> {
        let mut round_boundary =
            VerifierGrindingRoundBoundary::<D>::new(grinding_bits, grinding_nonces);
        self.verify_with_round_boundary(
            transcript,
            initial_claim,
            tau,
            field_cfg,
            &mut round_boundary,
        )
    }

    pub(crate) fn verify_with_round_boundary<P>(
        &self,
        transcript: &mut impl Transcript,
        initial_claim: F,
        tau: &[F],
        field_cfg: &F::Config,
        round_boundary: &mut P,
    ) -> Result<OuterSumcheckVerifierOutput<F>, SumcheckError>
    where
        P: RoundBoundaryPolicy,
    {
        let out = super::outer::verify_outer_sumcheck(
            field_cfg,
            transcript,
            initial_claim,
            tau,
            &self.sumcheck,
            super::outer::OuterEvaluations {
                ax: self.az_mle_claim,
                bx: self.bz_mle_claim,
                cx: self.cz_mle_claim,
            },
            round_boundary,
        )?;
        Ok(OuterSumcheckVerifierOutput {
            eval_points: out.point,
            az_mle_claim: out.evaluations.ax,
            bz_mle_claim: out.evaluations.bx,
            cz_mle_claim: out.evaluations.cx,
        })
    }
}
pub(crate) fn reconstruct_round_coefficients<F, const INPUT_COEFFS: usize, const COEFFS: usize>(
    current_claim: &F,
    coefficients_without_linear: &[F; INPUT_COEFFS],
    zero: &F,
    field_config: &F::Config,
) -> [F; COEFFS]
where
    F: SpartanField,
{
    assert!(INPUT_COEFFS >= 1);
    assert_eq!(COEFFS, INPUT_COEFFS + 1);

    let mut coefficients = std::array::from_fn(|_| zero.clone());
    coefficients[0] = coefficients_without_linear[0].clone();
    coefficients[2..].clone_from_slice(&coefficients_without_linear[1..]);

    let at_one_without_c1 = coefficients
        .iter()
        .fold(zero.clone(), |mut sum, coefficient| {
            sum = field_config.add(&(sum), &(coefficient));
            sum
        });
    coefficients[1] = (field_config).sub(
        &(field_config).sub(current_claim, &coefficients[0]),
        &at_one_without_c1,
    );
    coefficients
}

#[inline]
pub(crate) fn evaluate_polynomial<F, const COEFFS: usize>(
    coefficients: &[F; COEFFS],
    point: &F,
    zero: &F,
    field_config: &F::Config,
) -> F
where
    F: SpartanField,
{
    coefficients
        .iter()
        .rev()
        .fold(zero.clone(), |value, coefficient| {
            (field_config).add(&(field_config).mul(&value, point), coefficient)
        })
}

/// Completes and records one sumcheck round, then samples its challenge.
#[cfg(test)]
pub(crate) fn recover_full_round_polynomial_and_sample_next_challenge<
    F,
    const INPUT_COEFFS: usize,
    const COEFFS: usize,
>(
    transcript: &mut impl Transcript,
    current_claim: &mut F,
    coefficients_without_linear: &[F; INPUT_COEFFS],
    round_polynomials: &mut Vec<[F; COEFFS]>,
    eval_points: &mut Vec<F>,
    zero: &F,
    field_cfg: &F::Config,
) -> Result<F, SumcheckError>
where
    F: SpartanField,
{
    let mut round_boundary = UngrindedRoundBoundary;
    recover_full_round_polynomial_and_sample_next_challenge_with_boundary(
        transcript,
        current_claim,
        coefficients_without_linear,
        round_polynomials,
        eval_points,
        zero,
        field_cfg,
        &mut round_boundary,
    )
}

/// Completes and records one round under an explicit message/challenge
/// boundary policy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recover_full_round_polynomial_and_sample_next_challenge_with_boundary<
    F,
    P,
    const INPUT_COEFFS: usize,
    const COEFFS: usize,
>(
    transcript: &mut impl Transcript,
    current_claim: &mut F,
    coefficients_without_linear: &[F; INPUT_COEFFS],
    round_polynomials: &mut Vec<[F; COEFFS]>,
    eval_points: &mut Vec<F>,
    zero: &F,
    field_cfg: &F::Config,
    round_boundary: &mut P,
) -> Result<F, SumcheckError>
where
    F: SpartanField,
    P: RoundBoundaryPolicy,
{
    let coefficients = reconstruct_round_coefficients(
        current_claim,
        coefficients_without_linear,
        zero,
        &field_cfg,
    );
    let at_one = coefficients
        .iter()
        .fold(zero.clone(), |mut sum, coefficient| {
            sum = field_cfg.add(&(sum), &(coefficient));
            sum
        });
    debug_assert_eq!(*current_claim, (field_cfg).add(&coefficients[0], &at_one));

    absorb_field_elements(transcript, &coefficients, &field_cfg);
    round_boundary.after_round(transcript, round_polynomials.len())?;
    let challenge = squeeze_field(transcript, field_cfg)?;
    *current_claim = evaluate_polynomial(&coefficients, &challenge, zero, &field_cfg);
    round_polynomials.push(coefficients);
    eval_points.push(challenge.clone());
    Ok(challenge)
}

/// Allocation-free view of equality weights with the active coordinate
pub(crate) fn eq_eval<F>(left: &[F], right: &[F], field_cfg: &F::Config) -> Result<F, SumcheckError>
where
    F: SpartanField,
{
    if left.len() != right.len() {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }

    let one = F::one_with_cfg(field_cfg);
    let mut result = one.clone();
    for (left_i, right_i) in left.iter().zip(right) {
        // (1 - x) + y * (2x - 1), equivalent to
        // x*y + (1-x)*(1-y), including in characteristic two.
        let two_x_minus_one = (field_cfg).sub(&(field_cfg).add(left_i, left_i), &one);
        let factor = (field_cfg).add(
            &(field_cfg).sub(&one, left_i),
            &(field_cfg).mul(right_i, &two_x_minus_one),
        );
        result = field_cfg.mul(&(result), &(&factor));
    }
    Ok(result)
}
pub(crate) fn validate_field_elements<F>(
    values: &[F],
    field_cfg: &F::Config,
) -> Result<(), SumcheckError>
where
    F: SpartanField,
{
    let expected_modulus = F::canonical_modulus_encoding(field_cfg);
    for value in values {
        value
            .validate_element(&expected_modulus)
            .map_err(|_| SumcheckError::NonCanonicalFieldElement)?;
    }
    Ok(())
}
