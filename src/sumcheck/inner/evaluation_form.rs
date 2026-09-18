//! Multi-degree sumcheck: runs multiple degree groups in lockstep with
//! shared verifier randomness, producing a common evaluation point.
//!
//! # Protocol
//!
//! Given G degree groups each with (degree_g, mles_g, comb_fn_g):
//!
//! 1. Absorb metadata: num_vars, num_groups, per-group degrees
//! 2. For each round `i = 1..num_vars`:
//!    - Each group computes its round polynomial `P_g` (parallelizable)
//!    - Absorb all round messages in deterministic order
//!    - Sample ONE shared challenge `r_i`
//!    - All groups fix variable `i` at `r_i`
//! 3. Each group produces a subclaim at the shared point r = (r_1, ..., r_n)

use crate::poly::coefficient::PolynomialField;
#[cfg(test)]
use crate::poly::mle::DenseMultilinearExtension;
use crate::transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use crate::utils::{add, inner_transparent_field::InnerTransparentField, mul};
#[cfg(test)]
use crate::utils::{cfg_iter, cfg_iter_mut};

#[cfg(test)]
use num_traits::Zero;
#[cfg(all(test, feature = "parallel"))]
use rayon::prelude::*;
#[cfg(test)]
use std::marker::PhantomData;

#[cfg(test)]
use crate::piop::CombFn;

#[cfg(test)]
use crate::piop::sumcheck::prover::{ProverState as SumcheckProverState, RoundPolyEvaluator};
use crate::piop::sumcheck::{
    SumCheckError,
    prover::{NatEvaluatedPolyWithoutConstant, ProverMsg as SumcheckProverMsg},
    verifier::VerifierState,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Output of a [`Round1FastPath`]'s round-1 message computation.
#[cfg(test)]
pub struct Round1Output<F> {
    /// `p_1(0) + p_1(1)` — the asserted sum the verifier reconstructs
    /// against on the first round. The fast path computes this directly
    /// since it knows the round-1 polynomial in closed form.
    pub asserted_sum: F,
    /// `[p_1(1), p_1(2), ..., p_1(degree)]` — the standard tail-form
    /// round-1 message. Length must equal the group's `degree`.
    pub tail_evaluations: Vec<F>,
}

/// Optional per-group hook that lets a degree group bypass the standard
/// round-1 sumcheck loop. Used when the round-1 polynomial has a closed
/// form (e.g. booleanity zerocheck on bit-slice MLEs that are 0/1
/// pre-fold).
///
/// Contract:
/// - `round_1_message` is invoked *before* the verifier samples `r_1`.
///   It must produce the same message a faithful run of standard
///   `prove_round` would have emitted from the group's `poly`.
/// - `fold_with_r1` is invoked *after* `r_1` is sampled. It consumes
///   the fast-path data and returns the post-round-1 MLEs (size
///   `2^(num_vars - 1)`), in the same order/shape the group's
///   `comb_fn` expects. The framework places these into the prover
///   state and sets `skip_next_fold = true` so the standard path does
///   not double-fold them in round 2.
#[cfg(test)]
pub trait Round1FastPath<F: PolynomialField>: Send + Sync {
    fn round_1_message(&self, config: &F::Config) -> Round1Output<F>;
    fn fold_with_r1(
        self: Box<Self>,
        r_1: &F,
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>>;
}

/// A single degree group for the multi-degree sumcheck: (degree, mles,
/// comb_fn).
#[cfg(test)]
pub struct MultiDegreeSumcheckGroup<F: PolynomialField> {
    degree: usize,
    poly: Vec<DenseMultilinearExtension<F::Inner>>,
    comb_fn: CombFn<F>,
    round_1_fast: Option<Box<dyn Round1FastPath<F>>>,
    round_evaluator: Option<Box<dyn RoundPolyEvaluator<F>>>,
}

#[cfg(test)]
impl<F: PolynomialField> MultiDegreeSumcheckGroup<F> {
    pub fn new(
        degree: usize,
        poly: Vec<DenseMultilinearExtension<F::Inner>>,
        comb_fn: CombFn<F>,
    ) -> Self {
        Self {
            degree,
            poly,
            comb_fn,
            round_1_fast: None,
            round_evaluator: None,
        }
    }

    /// Construct a group whose round-1 message is produced by a custom
    /// [`Round1FastPath`]. `poly` may be empty here — the fast path
    /// supplies the post-round-1 MLEs via `fold_with_r1`.
    pub fn with_round_1_fast(
        degree: usize,
        poly: Vec<DenseMultilinearExtension<F::Inner>>,
        comb_fn: CombFn<F>,
        round_1_fast: Box<dyn Round1FastPath<F>>,
    ) -> Self {
        Self {
            degree,
            poly,
            comb_fn,
            round_1_fast: Some(round_1_fast),
            round_evaluator: None,
        }
    }

    /// Attach a fused [`RoundPolyEvaluator`] used for rounds ≥ 2 in place of
    /// the generic per-point gather. Byte-identical; see the trait. Builder
    /// form so it composes with [`Self::with_round_1_fast`].
    pub fn with_round_evaluator(mut self, evaluator: Box<dyn RoundPolyEvaluator<F>>) -> Self {
        self.round_evaluator = Some(evaluator);
        self
    }
}

/// Proof for a multi-degree sumcheck.
///
/// `group_messages[g][round]` = prover message for group g in that round.
/// All groups share verifier challenges, common evaluation point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiDegreeSumcheckProof<F> {
    /// List of prover messages, one for each round per group.
    group_messages: Vec<Vec<SumcheckProverMsg<F>>>,
    // The claimed sum for the first round polynomial per group.
    claimed_sums: Vec<F>,
    // Max degrees per group.
    degrees: Vec<usize>,
}

impl<F> MultiDegreeSumcheckProof<F> {
    pub(crate) fn quadratic(
        group_messages: Vec<Vec<SumcheckProverMsg<F>>>,
        claimed_sums: Vec<F>,
    ) -> Self {
        let degrees = vec![2; claimed_sums.len()];
        Self {
            group_messages,
            claimed_sums,
            degrees,
        }
    }
    /// Wrap a single generic sumcheck (e.g. the eq-factored driver's
    /// output) as a one-group multi-degree proof, so it can ride the
    /// protocol `Proof`'s existing `combined_sumcheck` field. Used by the
    /// linear-direct lane's selector-pin zerocheck.
    pub fn from_single_group(
        messages: Vec<SumcheckProverMsg<F>>,
        claimed_sum: F,
        degree: usize,
    ) -> Self {
        Self {
            group_messages: vec![messages],
            claimed_sums: vec![claimed_sum],
            degrees: vec![degree],
        }
    }

    /// Inverse of [`Self::from_single_group`]: returns
    /// `(messages, claimed_sum, degree)` when the proof holds exactly one
    /// group, else `None`.
    pub fn into_single_group(mut self) -> Option<(Vec<SumcheckProverMsg<F>>, F, usize)> {
        if self.group_messages.len() != 1 || self.claimed_sums.len() != 1 || self.degrees.len() != 1
        {
            return None;
        }
        Some((
            self.group_messages.pop()?,
            self.claimed_sums.pop()?,
            self.degrees.pop()?,
        ))
    }
}

impl<F> MultiDegreeSumcheckProof<F> {
    /// Needed by the verifier to check against expected
    /// sums before running the sumcheck.
    pub fn claimed_sums(&self) -> &[F] {
        &self.claimed_sums
    }

    /// Whether the proof has exactly the expected groups, degrees, and round
    /// message shapes.
    ///
    /// Protocol wrappers use this before transcript mutation so malformed
    /// decoded proofs cannot reach the verifier's assertion-based internal
    /// invariants or select a prover-controlled degree bound.
    pub(crate) fn has_shape(&self, num_vars: usize, degrees: &[usize]) -> bool {
        self.degrees == degrees
            && self.group_messages.len() == degrees.len()
            && self.claimed_sums.len() == degrees.len()
            && self
                .group_messages
                .iter()
                .zip(degrees)
                .all(|(messages, degree)| {
                    messages.len() == num_vars
                        && messages
                            .iter()
                            .all(|message| message.0.tail_evaluations.len() == *degree)
                })
    }
}

impl<F: PolynomialField> GenTranscribable for MultiDegreeSumcheckProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let mod_size = F::Modulus::NUM_BYTES;
        let cfg = crate::transcript::read_field_cfg::<F>(&bytes[..mod_size]);
        let bytes = &bytes[mod_size..];

        let (num_groups, bytes) = u32::read_transcription_bytes_subset(bytes);
        let num_groups = usize::try_from(num_groups).expect("group count must fit into usize");

        let (num_vars, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let num_vars = usize::try_from(num_vars).expect("num_vars must fit into usize");

        let mut degrees = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let (deg, rest) = u32::read_transcription_bytes_subset(bytes);
            degrees.push(usize::try_from(deg).expect("degree must fit into usize"));
            bytes = rest;
        }

        let mut group_messages = Vec::with_capacity(num_groups);
        for &deg in &degrees {
            let msg_bytes = mul!(deg, F::Inner::NUM_BYTES);
            let mut msgs = Vec::with_capacity(num_vars);
            for _ in 0..num_vars {
                let tail_evaluations =
                    crate::transcript::read_field_vec_with_cfg(&bytes[..msg_bytes], &cfg);
                msgs.push(SumcheckProverMsg(NatEvaluatedPolyWithoutConstant {
                    tail_evaluations,
                }));
                bytes = &bytes[msg_bytes..];
            }
            group_messages.push(msgs);
        }

        let mut claimed_sums = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let cs = F::Inner::read_transcription_bytes_exact(&bytes[..F::Inner::NUM_BYTES]);
            let cs = F::new_unchecked_with_cfg(cs, &cfg);
            claimed_sums.push(cs);
            bytes = &bytes[F::Inner::NUM_BYTES..];
        }

        Self {
            group_messages,
            claimed_sums,
            degrees,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        buf = crate::transcript::append_field_cfg::<F>(buf, &self.claimed_sums[0].modulus());

        let num_groups =
            u32::try_from(self.group_messages.len()).expect("num groups must fit into u32");
        num_groups.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];

        // All groups share the same number of rounds (num_vars).
        let num_vars =
            u32::try_from(self.group_messages[0].len()).expect("num_vars must fit into u32");
        num_vars.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];

        for &deg in &self.degrees {
            let deg = u32::try_from(deg).expect("degree must fit into u32");
            deg.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
            buf = &mut buf[u32::NUM_BYTES..];
        }

        for group in &self.group_messages {
            for msg in group {
                buf = crate::transcript::append_field_vec_inner(buf, &msg.0.tail_evaluations);
            }
        }

        for cs in &self.claimed_sums {
            cs.inner()
                .write_transcription_bytes_exact(&mut buf[..F::Inner::NUM_BYTES]);
            buf = &mut buf[F::Inner::NUM_BYTES..];
        }
    }
}

impl<F: PolynomialField> Transcribable for MultiDegreeSumcheckProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        let num_groups = self.group_messages.len();
        let num_vars = self.group_messages[0].len();
        // total_evals = Σ_g (degree_g × num_vars)
        let total_evals: usize = self.degrees.iter().map(|&d| mul!(d, num_vars)).sum();

        // [field_cfg][num_groups][num_vars][deg₀..degₙ][evals...][claimed_sums]
        let header = add!(F::Modulus::NUM_BYTES, add!(u32::NUM_BYTES, u32::NUM_BYTES));
        let degrees = mul!(num_groups, u32::NUM_BYTES);
        let eval_data = mul!(total_evals, F::Inner::NUM_BYTES);
        let claimed = mul!(num_groups, F::Inner::NUM_BYTES);

        add!(header, add!(degrees, add!(eval_data, claimed)))
    }
}

/// Sub-claims: shared evaluation point + per-group expected evaluation.
#[derive(Debug)]
pub struct MultiDegreeSubClaims<F> {
    point: Vec<F>,
    expected_evaluations: Vec<F>,
}

impl<F> MultiDegreeSubClaims<F> {
    pub fn point(&self) -> &[F] {
        &self.point
    }

    pub fn expected_evaluations(&self) -> &[F] {
        &self.expected_evaluations
    }
}

// ---------------------------------------------------------------------------
// MultiDegreeSumcheck
// ---------------------------------------------------------------------------

#[cfg(test)]
pub struct MultiDegreeSumcheck<F>(PhantomData<F>);

#[cfg(test)]
impl<F: PolynomialField> MultiDegreeSumcheck<F> {
    /// Multi-degree sumcheck prover.
    ///
    /// Runs the prover side of the sumcheck protocol for G degree groups
    /// sharing one verifier challenge per round. Proves the claim:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_g(x) =
    /// \text{claimed\\_sum}_g \quad \forall g
    /// $$
    ///
    /// where $G_g(x) = \text{comb\\_fn}_g(\text{mles}_g(x))$ is the combination
    /// function for group $g$ applied to its MLEs.
    ///
    /// It is designed to be used as a subprotocol within a larger system.
    /// since it takes the FS transcript (`transcript` argument) as input
    /// and returns the **internal ProverState** alongside the sumcheck proof.
    ///
    /// Claimed sums are derived by the prover during the first round.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript.
    /// * `groups`: One [`MultiDegreeSumcheckGroup`] per degree bucket, each
    ///   carrying its MLEs and combination function.
    /// * `num_vars`: Number of variables (must be consistent across all
    ///   groups).
    /// * `config`: Field configuration.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///
    /// 1. [`MultiDegreeSumcheckProof<F>`]: The proof (group messages, claimed
    ///    sums, degrees).
    /// 2. `Vec<ProverState<F>>`: Per-group prover states — needed by the caller
    ///    to evaluate MLEs at the shared point after the sumcheck.
    ///
    /// # Panics
    ///
    /// * Panics if `num_vars == 0` or `groups` is empty.
    #[allow(clippy::type_complexity)]
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        groups: Vec<MultiDegreeSumcheckGroup<F>>,
        num_vars: usize,
        config: &F::Config,
    ) -> (MultiDegreeSumcheckProof<F>, Vec<SumcheckProverState<F>>)
    where
        F: InnerTransparentField + Send + Sync,
        F::Inner: ConstTranscribable + Zero,
        F::Modulus: ConstTranscribable,
    {
        assert!(
            num_vars > 0,
            "Attempts to prove a constant: num_vars must be > 0"
        );
        assert!(!groups.is_empty(), "need at least one degree group");

        let num_groups = groups.len();
        let mut buf = vec![0; F::Inner::NUM_BYTES];
        let nvars_field = F::interpolation_node(num_vars as u64, config);
        let ngroups_field = F::interpolation_node(num_groups as u64, config);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        transcript.absorb_random_field(&ngroups_field, &mut buf);

        let mut group_messages: Vec<Vec<SumcheckProverMsg<F>>> = (0..num_groups)
            .map(|_| Vec::with_capacity(num_vars))
            .collect();
        let mut claimed_sums = Vec::with_capacity(num_groups);

        let mut prover_states: Vec<SumcheckProverState<F>> = Vec::with_capacity(num_groups);
        let mut comb_fns: Vec<CombFn<F>> = Vec::with_capacity(num_groups);
        let mut fast_paths: Vec<Option<Box<dyn Round1FastPath<F>>>> =
            Vec::with_capacity(num_groups);
        for group in groups {
            let degree_field = F::interpolation_node(group.degree as u64, config);
            transcript.absorb_random_field(&degree_field, &mut buf);
            let mut state = SumcheckProverState::new(group.poly, num_vars, group.degree);
            state.round_evaluator = group.round_evaluator;
            prover_states.push(state);
            comb_fns.push(group.comb_fn);
            fast_paths.push(group.round_1_fast);
        }

        // ---- Round 1 ---------------------------------------------------
        let _g_r1 = tracing::info_span!("mds:round1").entered();
        let mut round_1_msgs: Vec<SumcheckProverMsg<F>> = Vec::with_capacity(num_groups);
        for ((state, comb_fn), fp_slot) in prover_states
            .iter_mut()
            .zip(comb_fns.iter())
            .zip(fast_paths.iter())
        {
            let msg = if let Some(fp) = fp_slot {
                let out = fp.round_1_message(config);
                debug_assert_eq!(
                    out.tail_evaluations.len(),
                    state.max_degree,
                    "fast-path round-1 tail must have length equal to group's degree"
                );
                state.asserted_sum = Some(out.asserted_sum);
                state.round = 1;
                SumcheckProverMsg(NatEvaluatedPolyWithoutConstant::new(out.tail_evaluations))
            } else {
                state.prove_round(&None, comb_fn, config)
            };
            round_1_msgs.push(msg);
        }
        for msg in &round_1_msgs {
            transcript.absorb_random_field_slice(&msg.0.tail_evaluations, &mut buf);
        }
        for (j, msg) in round_1_msgs.into_iter().enumerate() {
            group_messages[j].push(msg);
        }
        let r_1: F = transcript.get_field_challenge(config);
        transcript.absorb_random_field(&r_1, &mut buf);
        let mut verifier_msg = Some(r_1.clone());

        // For fast-path groups, materialize the round-1-folded MLEs and
        // mark the next standard fold to be skipped.
        for (state, fp_slot) in prover_states.iter_mut().zip(fast_paths.iter_mut()) {
            if let Some(fp) = fp_slot.take() {
                let folded = fp.fold_with_r1(&r_1, config);
                state.mles = folded;
                state.skip_next_fold = true;
            }
        }
        drop(_g_r1);

        // ---- Rounds 2..num_vars ---------------------------------------
        let _g_rest = tracing::info_span!("mds:rounds").entered();
        for _ in 1..num_vars {
            // Parallel: each group computes its round polynomial independently
            let round_msgs: Vec<SumcheckProverMsg<F>> = cfg_iter_mut!(prover_states)
                .zip(cfg_iter!(comb_fns))
                .map(|(state, comb_fn)| state.prove_round(&verifier_msg, comb_fn, config))
                .collect();

            // Sequential: absorb in deterministic order, sample one shared challenge
            for msg in &round_msgs {
                transcript.absorb_random_field_slice(&msg.0.tail_evaluations, &mut buf);
            }

            for (j, msg) in round_msgs.into_iter().enumerate() {
                group_messages[j].push(msg);
            }

            let next_verifier_msg = transcript.get_field_challenge(config);
            transcript.absorb_random_field(&next_verifier_msg, &mut buf);

            verifier_msg = Some(next_verifier_msg);
        }

        prover_states.iter_mut().for_each(|state| {
            let sum = state
                .asserted_sum
                .clone()
                .expect("asserted sum should be recorded after the first prover round");
            claimed_sums.push(sum);

            if let Some(ref vmsg) = verifier_msg {
                state.randomness.push(vmsg.clone());
            }
        });

        let degrees = prover_states.iter().map(|s| s.max_degree).collect();

        (
            MultiDegreeSumcheckProof {
                group_messages,
                claimed_sums,
                degrees,
            },
            prover_states,
        )
    }
}

impl<F: PolynomialField> MultiDegreeSumcheckProof<F> {
    /// Multi-degree sumcheck verifier.
    ///
    /// Runs the verifier side of the sumcheck protocol for G degree groups
    /// sharing one verifier challenge per round. Verifies the claim:
    ///
    /// $$
    /// \sum_{x \in \{0, 1\}^{\text{num\\_vars}}} G_g(x) =
    /// \text{claimed\\_sum}_g \quad \forall g
    /// $$
    ///
    /// where $G_g(x) = \text{comb\\_fn}_g(\text{mles}_g(x))$.
    ///
    /// It is designed to be used as a subprotocol within a larger system.
    /// If successful, it returns **Subclaim** for each group, a final equation
    /// that the outer protocol must satisfy for the overall sumcheck proof
    /// to be valid.
    ///
    /// Mirrors the prover transcript exactly: absorbs metadata, then per-round
    /// absorbs all group messages, samples one shared challenge, and calls
    /// [`VerifierState::check_and_generate_subclaim`] per group. Per-group
    /// degrees are read from the proof — no external degree parameter needed.
    ///
    /// # Arguments
    ///
    /// * `transcript`: Fiat-Shamir transcript (must match prover state at the
    ///   start of the sumcheck).
    /// * `num_vars`: Number of variables (sumcheck rounds).
    /// * `proof`: The [`MultiDegreeSumcheckProof`] produced by the prover.
    /// * `config`: Field configuration.
    ///
    /// # Returns
    ///
    /// * `Ok(MultiDegreeSubClaims<F>)`: Shared evaluation point `r*` and
    ///   per-group expected evaluations. The caller must verify each group's
    ///   MLE combination at `r*` equals its expected evaluation.
    /// * `Err(SumCheckError<F>)`: If any round check fails.
    ///
    /// # Panics
    ///
    /// * Panics if `num_vars == 0` or the proof has no groups.
    pub fn verify_as_subprotocol(
        &self,
        transcript: &mut impl Transcript,
        num_vars: usize,
        expected_degrees: &[usize],
        config: &F::Config,
    ) -> Result<MultiDegreeSubClaims<F>, SumCheckError<F>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        if num_vars == 0
            || expected_degrees.is_empty()
            || !self.has_shape(num_vars, expected_degrees)
        {
            return Err(SumCheckError::InvalidProofShape);
        }
        let num_groups = expected_degrees.len();

        let mut buf = vec![0; F::Inner::NUM_BYTES];
        let nvars_field = F::interpolation_node(num_vars as u64, config);
        let ngroups_field = F::interpolation_node(num_groups as u64, config);
        transcript.absorb_random_field(&nvars_field, &mut buf);
        transcript.absorb_random_field(&ngroups_field, &mut buf);

        let mut verifier_states: Vec<VerifierState<F>> = (0..num_groups)
            .map(|j| {
                let degree = expected_degrees[j];
                let degree_field = F::interpolation_node(degree as u64, config);
                transcript.absorb_random_field(&degree_field, &mut buf);

                VerifierState::new(num_vars, degree, config)
            })
            .collect();

        for i in 0..num_vars {
            self.group_messages.iter().for_each(|msg| {
                transcript.absorb_random_field_slice(&msg[i].0.tail_evaluations, &mut buf)
            });

            let shared_challenge: F = transcript.get_field_challenge(config);
            transcript.absorb_random_field(&shared_challenge, &mut buf);

            verifier_states
                .iter_mut()
                .zip(self.group_messages.iter())
                .for_each(|(state, msg)| {
                    state.verify_round_with_challenge(&msg[i], shared_challenge.clone())
                });
        }

        let mut shared_point: Option<Vec<F>> = None;
        let mut expected_evaluations = Vec::with_capacity(num_groups);
        // TODO: parallelize when multiple lookup groups exist
        for (j, state) in verifier_states.into_iter().enumerate() {
            let subclaim = state.check_and_generate_subclaim(self.claimed_sums[j].clone())?;
            if let Some(ref p) = shared_point {
                debug_assert_eq!(p, &subclaim.point);
            } else {
                shared_point = Some(subclaim.point)
            }

            expected_evaluations.push(subclaim.expected_evaluation);
        }

        Ok(MultiDegreeSubClaims {
            point: shared_point.expect("at least one group"),
            expected_evaluations,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn message(tail_len: usize) -> SumcheckProverMsg<u64> {
        SumcheckProverMsg(NatEvaluatedPolyWithoutConstant {
            tail_evaluations: vec![0; tail_len],
        })
    }

    #[test]
    fn proof_shape_check_pins_groups_rounds_and_degrees() {
        let valid = MultiDegreeSumcheckProof {
            group_messages: vec![vec![message(2); 3]],
            claimed_sums: vec![0],
            degrees: vec![2],
        };
        assert!(valid.has_shape(3, &[2]));
        assert!(!valid.has_shape(2, &[2]));
        assert!(!valid.has_shape(3, &[3]));

        let no_groups = MultiDegreeSumcheckProof::<u64> {
            group_messages: vec![],
            claimed_sums: vec![],
            degrees: vec![],
        };
        assert!(!no_groups.has_shape(3, &[2]));

        let wrong_tail = MultiDegreeSumcheckProof {
            group_messages: vec![vec![message(1); 3]],
            claimed_sums: vec![0],
            degrees: vec![2],
        };
        assert!(!wrong_tail.has_shape(3, &[2]));
    }
}
