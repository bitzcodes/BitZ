//! Affine Round-0 state: a composed protocol binds once before its PIOP and
//! moves the resulting claim into the final authenticated opening.

use super::*;

/// Opaque prover state. Existing standalone callers may still supply optional
/// parameters; composed callers use [`bind_prover_ood`] before their PIOP.
pub struct ProverOod(ProverState);
enum ProverState {
    AtOpening(Option<OodRoundParams>),
    Bound(Option<OodProverClaim>),
}

impl From<Option<OodRoundParams>> for ProverOod {
    fn from(params: Option<OodRoundParams>) -> Self {
        Self(ProverState::AtOpening(params))
    }
}

impl ProverOod {
    pub(super) fn bind(
        self,
        transcript: &mut (impl Transcript + Send),
        hint: &FlockCommitHint,
    ) -> Self {
        Self(ProverState::Bound(self.claim(transcript, hint)))
    }

    pub(super) fn claim(
        self,
        transcript: &mut (impl Transcript + Send),
        hint: &FlockCommitHint,
    ) -> Option<OodProverClaim> {
        match self.0 {
            ProverState::AtOpening(params) => {
                params.map(|params| prove_ood_round(transcript, hint, params))
            }
            ProverState::Bound(claim) => claim,
        }
    }
}

/// Opaque verifier state, tied to the exact Round-0 payload already checked.
pub struct VerifierOod(VerifierState);
enum VerifierState {
    AtOpening(Option<OodRoundParams>),
    Bound(Option<(OodVerifierClaim, OodRound)>),
}

impl From<Option<OodRoundParams>> for VerifierOod {
    fn from(params: Option<OodRoundParams>) -> Self {
        Self(VerifierState::AtOpening(params))
    }
}

impl VerifierOod {
    pub(super) fn bind(
        self,
        transcript: &mut (impl Transcript + Send),
        vars: usize,
        round: Option<&OodRound>,
    ) -> Result<Self, FlockRsError> {
        Ok(Self(VerifierState::Bound(
            self.claim(transcript, vars, round)?
                .map(|claim| (claim, *round.expect("verified OOD"))),
        )))
    }

    pub(super) fn claim(
        self,
        transcript: &mut (impl Transcript + Send),
        packed_vars: usize,
        round: Option<&OodRound>,
    ) -> Result<Option<OodVerifierClaim>, FlockRsError> {
        match self.0 {
            VerifierState::AtOpening(params) => {
                verify_optional(transcript, packed_vars, params, round)
            }
            VerifierState::Bound(Some((claim, bound))) => {
                if claim.point.len() != packed_vars
                    || !round.is_some_and(|round| round.y == bound.y && round.nonce == bound.nonce)
                {
                    return Err(FlockRsError::OodRound);
                }
                Ok(Some(claim))
            }
            VerifierState::Bound(None) if round.is_none() => Ok(None),
            VerifierState::Bound(None) => Err(FlockRsError::OodRound),
        }
    }
}

/// Run exactly after binding the source commitment/configuration and before
/// sampling any projection prime or witness-dependent PIOP challenge.
pub fn bind_prover_ood(
    transcript: &mut (impl Transcript + Send),
    hint: &FlockCommitHint,
    params: Option<OodRoundParams>,
) -> ProverOod {
    let _scope = tracing::info_span!("step0:ood_prove").entered();
    ProverOod(ProverState::Bound(
        params.map(|params| prove_ood_round(transcript, hint, params)),
    ))
}

/// Mirror [`bind_prover_ood`] before the verifier samples the projection prime.
pub fn bind_verifier_ood(
    transcript: &mut (impl Transcript + Send),
    packed_vars: usize,
    params: Option<OodRoundParams>,
    round: Option<&OodRound>,
) -> Result<VerifierOod, FlockRsError> {
    let _scope = tracing::info_span!("step0:ood_verify").entered();
    let claim = verify_optional(transcript, packed_vars, params, round)?;
    Ok(VerifierOod(VerifierState::Bound(claim.map(|claim| {
        (claim, round.expect("verified Round 0").clone())
    }))))
}

fn verify_optional(
    transcript: &mut (impl Transcript + Send),
    packed_vars: usize,
    params: Option<OodRoundParams>,
    round: Option<&OodRound>,
) -> Result<Option<OodVerifierClaim>, FlockRsError> {
    match (params, round) {
        (Some(params), Some(round)) => {
            verify_ood_round(transcript, packed_vars, params, round).map(Some)
        }
        (None, None) => Ok(None),
        _ => Err(FlockRsError::OodRound),
    }
}
