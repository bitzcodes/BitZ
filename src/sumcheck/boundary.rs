use super::SumcheckError;
use crate::piop::spartan::grinding::{
    GrindingDomain, GrindingError, GrindingRound, MAX_GRINDING_BITS, grind_and_absorb,
    verify_and_absorb,
};
use crate::transcript::traits::Transcript;
/// Controls the transcript boundary between an absorbed round polynomial and
/// the verifier challenge that follows it.
pub trait RoundBoundaryPolicy {
    /// Validates policy-level proof shape before the transcript is mutated.
    fn validate(&self, _expected_rounds: usize) -> Result<(), SumcheckError> {
        Ok(())
    }

    /// Processes the prover message after absorption and before its challenge.
    fn after_round<T: Transcript>(
        &mut self,
        transcript: &mut T,
        round: usize,
    ) -> Result<(), SumcheckError>;
}

/// Existing sumcheck transcript behavior: no bytes between message and
/// challenge.
pub struct UngrindedRoundBoundary;

impl RoundBoundaryPolicy for UngrindedRoundBoundary {
    fn after_round<T: Transcript>(
        &mut self,
        _transcript: &mut T,
        _round: usize,
    ) -> Result<(), SumcheckError> {
        Ok(())
    }
}

#[allow(dead_code)]
pub struct ProverGrindingRoundBoundary<D> {
    bits: u32,
    round_offset: usize,
    pub(crate) nonces: Vec<u64>,
    _domain: core::marker::PhantomData<fn() -> D>,
}

impl<D> ProverGrindingRoundBoundary<D> {
    #[allow(dead_code)]
    pub fn with_round_offset(bits: u32, round_offset: usize) -> Self {
        Self {
            bits,
            round_offset,
            nonces: Vec::new(),
            _domain: core::marker::PhantomData,
        }
    }

    /// The nonces found so far, in round order (empty at difficulty 0).
    #[allow(dead_code)]
    pub fn into_nonces(self) -> Vec<u64> {
        self.nonces
    }
}

impl<D: GrindingDomain> RoundBoundaryPolicy for ProverGrindingRoundBoundary<D> {
    fn validate(&self, _expected_rounds: usize) -> Result<(), SumcheckError> {
        // Difficulty 0 = the boundary does not exist: no transcript bytes,
        // no nonces (the λ = 100 profiles).
        if self.bits == 0 {
            return Ok(());
        }
        validate_grinding_configuration::<D>(self.bits)
    }

    fn after_round<T: Transcript>(
        &mut self,
        transcript: &mut T,
        round: usize,
    ) -> Result<(), SumcheckError> {
        if self.bits == 0 {
            return Ok(());
        }
        let _scope = tracing::info_span!("spartan:round_grinding_prove").entered();
        let round = self
            .round_offset
            .checked_add(round)
            .and_then(|round| u64::try_from(round).ok())
            .expect("an in-memory sumcheck round index fits in u64");
        let nonce = grind_and_absorb::<D, _>(transcript, GrindingRound::new(round), self.bits)?;
        self.nonces.push(nonce);
        Ok(())
    }
}

pub struct VerifierGrindingRoundBoundary<'a, D> {
    bits: u32,
    nonces: &'a [u64],
    _domain: core::marker::PhantomData<fn() -> D>,
}

impl<'a, D> VerifierGrindingRoundBoundary<'a, D> {
    pub fn new(bits: u32, nonces: &'a [u64]) -> Self {
        Self {
            bits,
            nonces,
            _domain: core::marker::PhantomData,
        }
    }
}

impl<D: GrindingDomain> RoundBoundaryPolicy for VerifierGrindingRoundBoundary<'_, D> {
    fn validate(&self, expected_rounds: usize) -> Result<(), SumcheckError> {
        // Difficulty 0: the boundary does not exist, so a canonical proof
        // carries NO nonces.
        let expected = if self.bits == 0 {
            0
        } else {
            validate_grinding_configuration::<D>(self.bits)?;
            expected_rounds
        };
        if self.nonces.len() != expected {
            return Err(SumcheckError::InvalidGrindingNonceCount {
                expected,
                actual: self.nonces.len(),
            });
        }
        Ok(())
    }

    fn after_round<T: Transcript>(
        &mut self,
        transcript: &mut T,
        round: usize,
    ) -> Result<(), SumcheckError> {
        if self.bits == 0 {
            return Ok(());
        }
        let _scope = tracing::info_span!("spartan:round_grinding_verify").entered();
        let round_index =
            u64::try_from(round).expect("an in-memory sumcheck round index fits in u64");
        verify_and_absorb::<D, _>(
            transcript,
            GrindingRound::new(round_index),
            self.bits,
            self.nonces[round],
        )?;
        Ok(())
    }
}

fn validate_grinding_configuration<D: GrindingDomain>(bits: u32) -> Result<(), SumcheckError> {
    if !(1..=MAX_GRINDING_BITS).contains(&bits) {
        return Err(GrindingError::InvalidDifficulty { bits }.into());
    }
    if D::DOMAIN.is_empty() {
        return Err(GrindingError::EmptyDomain.into());
    }
    Ok(())
}
