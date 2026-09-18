//! Ordinary degree-two inner products, with native inputs and shared-challenge batches.
//! Optimized borrowed, packed, and factored integrations use the same round engine.
pub(crate) mod binary;
mod dense;
pub(crate) mod input;
pub use input::InitialClaims;
pub mod evaluation_form;
pub(crate) mod native;
pub(crate) mod packed;
#[cfg(test)]
pub(crate) mod reference;

use crate::piop::spartan::{SpartanField, absorb_field_elements, squeeze_field};
use crate::sumcheck::{RoundBoundaryPolicy, SumcheckError, SumcheckProof};
use crate::transcript::traits::Transcript;
use field::{BatchMulAcc, FieldOps, PreparedLinearCombination, Reduce, RingOps};

type Elem<F> = <F as RingOps>::Elem;
type Acc<F, T> = <F as BatchMulAcc<Elem<F>, T>>::Accumulator;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerSumcheckOutput<E> {
    pub proof: SumcheckProof<E, 3>,
    pub point: Vec<E>,
    pub final_claim: E,
    /// [weight(point), value(point)].
    pub terminal_evaluations: [E; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedInnerSumcheckOutput<E, const K: usize> {
    pub proofs: [SumcheckProof<E, 3>; K],
    pub point: Vec<E>,
    pub final_claims: [E; K],
    pub terminal_evaluations: [[E; 2]; K],
}

/// Prove one dot product. Input representations select preparation/folding only;
/// all messages, challenges and terminal checks belong to the batched prover.
pub fn prove_inner_sumcheck<F, V>(
    field: &F,
    transcript: &mut impl Transcript,
    initial_claim: F::Elem,
    values: V,
    weights: V::Weights,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<InnerSumcheckOutput<F::Elem>, SumcheckError>
where
    F: FieldOps + Sync,
    V: input::Input<F>,
{
    let output = prove_batched_inner_sumcheck(
        field,
        transcript,
        &[initial_claim],
        [values],
        [weights],
        boundary,
    )?;
    let [proof] = output.proofs;
    Ok(InnerSumcheckOutput {
        proof,
        point: output.point,
        final_claim: output.final_claims[0],
        terminal_evaluations: output.terminal_evaluations[0],
    })
}

/// Prove independent dot products using one challenge per round. Arrays retain
/// their compile-time batch size; vectors support protocol-defined runtime batches.
/// No random linear combination of the claims is introduced.
pub fn prove_batched_inner_sumcheck<'a, F, B>(
    field: &F,
    transcript: &mut impl Transcript,
    initial_claims: impl Into<InitialClaims<'a, F::Elem>>,
    values: B,
    weights: B::Weights,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<B::Output, SumcheckError>
where
    F: FieldOps + Sync,
    F::Elem: 'a,
    B: input::Batch<F>,
{
    use input::{Codec, State};
    let initial_claims = initial_claims.into();
    let mut states = values.prepare(field, weights)?;
    let slots = states.as_mut();
    if slots.is_empty() {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    let rounds = slots[0].state.num_vars();
    if slots.iter().any(|s| s.state.num_vars() != rounds) {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    boundary.validate(rounds)?;
    match initial_claims {
        InitialClaims::Known(claims) => {
            if claims.len() != slots.len() {
                return Err(SumcheckError::InvalidProductDimensions);
            }
            for (slot, claim) in slots.iter_mut().zip(claims) {
                slot.claim = *claim;
            }
        }
        InitialClaims::Compute => {
            for slot in &mut *slots {
                slot.claim = slot
                    .state
                    .initial_claim()
                    .ok_or(SumcheckError::InvalidProductDimensions)?;
            }
        }
    }
    for slot in &*slots {
        slot.state.validate_claim(field, &slot.claim)?;
    }
    B::Codec::start(field, transcript, rounds, slots.len())?;
    let mut point = Vec::with_capacity(rounds);
    for round in 0..rounds {
        for slot in &mut *slots {
            let [c0, c2] = slot.state.coefficients(field)?;
            let c1 = field.sub(&field.sub(&slot.claim, &field.add(&c0, &c0)), &c2);
            let message = [c0, c1, c2];
            B::Codec::absorb(field, transcript, &message);
            slot.proof.round_polynomials.push(message);
        }
        boundary.after_round(transcript, round)?;
        let challenge = B::Codec::challenge(field, transcript)?;
        point.push(challenge);
        let advance = |slot: &mut input::Slot<F, B::State>| {
            let &[c0, c1, c2] = slot.proof.round_polynomials.last().unwrap();
            slot.claim = field.add(
                &c0,
                &field.mul(&challenge, &field.add(&c1, &field.mul(&challenge, &c2))),
            );
            slot.state.fold(field, &challenge)
        };
        #[cfg(feature = "parallel")]
        if slots.len() > 1 {
            use rayon::prelude::*;
            slots.par_iter_mut().try_for_each(advance)?;
        } else {
            for slot in &mut *slots {
                advance(slot)?;
            }
        }
        #[cfg(not(feature = "parallel"))]
        for slot in &mut *slots {
            advance(slot)?;
        }
    }
    for slot in &mut *slots {
        slot.terminal = slot.state.terminal(field)?;
        let [weight, value] = slot.terminal;
        if !field::CtEq::ct_eq(&slot.claim, &field.mul(&weight, &value)).declassify() {
            return Err(SumcheckError::InvalidTerminalClaim);
        }
    }
    Ok(B::finish(states, point))
}

/// Runtime-sized counterpart of `BatchedInnerSumcheckOutput<E, K>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicBatchedInnerSumcheckOutput<E> {
    pub proofs: Vec<SumcheckProof<E, 3>>,
    pub point: Vec<E>,
    pub final_claims: Vec<E>,
    pub terminal_evaluations: Vec<[E; 2]>,
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod skipped_experiment;
