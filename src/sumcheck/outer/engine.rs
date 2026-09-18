//! The single transcript/round state machine, shared by every outer prefix.
use super::ordinary::equality_coordinate_evaluation;
use crate::piop::spartan::{SpartanField, absorb_field_elements};
use crate::sumcheck::{RoundBoundaryPolicy, SumcheckError, proof::*};
use crate::transcript::traits::Transcript;
use field::RingOps;

pub(super) struct RoundState<E> {
    pub claim: E,
    pub equality_scale: E,
    pub proof: SumcheckProof<E, 4>,
    pub point: Vec<E>,
}

impl<E: SpartanField> RoundState<E> {
    pub fn new(claim: E, rounds: usize, field: &E::Config) -> Self {
        Self {
            claim,
            equality_scale: field.one(),
            proof: SumcheckProof {
                round_polynomials: Vec::with_capacity(rounds),
            },
            point: Vec::with_capacity(rounds),
        }
    }

    /// Absorb all four cubic coefficients before the boundary and challenge.
    /// Keeping this operation here also preserves absolute round numbering
    /// when a native/zerocheck prefix hands control to the ordinary tail.
    pub fn sample(
        &mut self,
        field: &E::Config,
        transcript: &mut impl Transcript,
        tau: &[E],
        coefficients: &[E; 3],
        boundary: &mut impl RoundBoundaryPolicy,
    ) -> Result<E, SumcheckError> {
        let round = self.point.len();
        let challenge = recover_full_round_polynomial_and_sample_next_challenge_with_boundary(
            transcript,
            &mut self.claim,
            coefficients,
            &mut self.proof.round_polynomials,
            &mut self.point,
            &field.zero(),
            field,
            boundary,
        )?;
        self.equality_scale = field.mul(
            &self.equality_scale,
            &equality_coordinate_evaluation(&tau[round], &challenge, &field.one(), field),
        );
        Ok(challenge)
    }

    /// Arithmetic callbacks fold the tables and prepare the next message in
    /// the same traversal. They never absorb bytes or sample challenges.
    pub fn continue_with(
        &mut self,
        field: &E::Config,
        transcript: &mut impl Transcript,
        tau: &[E],
        mut coefficients: [E; 3],
        boundary: &mut impl RoundBoundaryPolicy,
        mut fold: impl FnMut(usize, E, &Self) -> Result<[E; 3], SumcheckError>,
    ) -> Result<(), SumcheckError> {
        while self.point.len() < tau.len() {
            let round = self.point.len();
            let challenge = self.sample(field, transcript, tau, &coefficients, boundary)?;
            coefficients = fold(round, challenge, self)?;
        }
        Ok(())
    }

    pub fn finish(
        self,
        field: &E::Config,
        transcript: &mut impl Transcript,
        [a, b, c]: [E; 3],
    ) -> Result<OuterSumcheckOutput<E>, SumcheckError> {
        let expected = field.mul(&self.equality_scale, &field.sub(&field.mul(&a, &b), &c));
        if self.claim != expected {
            return Err(SumcheckError::InvalidTerminalClaim);
        }
        absorb_field_elements(transcript, &[a, b, c], field);
        Ok(OuterSumcheckOutput {
            proof: OuterSumcheckProof {
                sumcheck: self.proof,
                az_mle_claim: a,
                bz_mle_claim: b,
                cz_mle_claim: c,
            },
            eval_points: self.point,
            final_claim: self.claim,
        })
    }
}
