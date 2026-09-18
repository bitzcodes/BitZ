//! Plonky3 WHIR adapter for the BabyBear multiplication assignment.
//!
//! The committed private data are four native BabyBear columns
//! `A || B || C || K`.  The public assignment represented by the terminal
//! claim is
//!
//! ```text
//! f = [e_0 | A | B | C | K | 0 | 0 | 0].
//! ```
//!
//! Spartan/BitZ numbers gate coordinates least-significant-coordinate first,
//! while Plonky3 stores a hypercube table in lexicographic (big-endian)
//! coordinate order.  [`p3_opening_point`] performs the one explicit reversal
//! at the adapter boundary.

#![allow(dead_code)] // Boundary accessors are also exercised by the adapter tests.

use crate::common::plonky3::{self, baby_bear as stack};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout, SuffixProver};
use p3_sumcheck::{OpeningProtocol, PrescribedPointPcs};
use p3_whir::{
    DomainSeparator, FoldingFactor, ProtocolParameters, SecurityAssumption, VerifierError,
    WhirConfigError,
};
use thiserror::Error;

/// Base field used for the committed integer columns.
pub type Val = stack::Val;
/// Degree of the native BabyBear extension used for WHIR challenges.
#[cfg(feature = "plonky3-whir-degree4-bench")]
pub const CHALLENGE_EXTENSION_DEGREE: usize = 4;
#[cfg(not(feature = "plonky3-whir-degree4-bench"))]
pub const CHALLENGE_EXTENSION_DEGREE: usize = 5;
/// Native WHIR challenge field (about 155 bits).
pub type Challenge = BinomialExtensionField<Val, CHALLENGE_EXTENSION_DEGREE>;

type Challenger = stack::Challenger;
type WhirLayout = SuffixProver<Val, Challenge>;
type Pcs = stack::Pcs<Challenge>;

/// Public Merkle commitment returned by WHIR.
pub type Commitment = <Pcs as MultilinearPcs<Challenge, Challenger>>::Commitment;
/// Non-hiding WHIR prescribed-opening proof.
pub type Proof = <Pcs as MultilinearPcs<Challenge, Challenger>>::Proof;
type ProverData = <Pcs as MultilinearPcs<Challenge, Challenger>>::ProverData;
type Witness = <Pcs as MultilinearPcs<Challenge, Challenger>>::Witness;

/// Six internal margin bits cover the union of WHIR soundness events while
/// preserving the campaign's external 100-bit target.
pub const SECURITY_BITS: usize = 106;
/// Cap per-round grinding so larger shapes trade proof size for practical runtime.
pub const MAX_POW_BITS: usize = 12;
pub const FOLDING: usize = 4;
pub const STARTING_LOG_INV_RATE: usize = 1;
#[cfg(feature = "plonky3-whir-degree4-bench")]
pub const SECURITY_ASSUMPTION: SecurityAssumption = SecurityAssumption::UniqueDecoding;
#[cfg(not(feature = "plonky3-whir-degree4-bench"))]
pub const SECURITY_ASSUMPTION: SecurityAssumption = SecurityAssumption::JohnsonBound;
#[cfg(feature = "plonky3-whir-degree4-bench")]
pub const SECURITY_ASSUMPTION_LABEL: &str = "UniqueDecoding";
#[cfg(not(feature = "plonky3-whir-degree4-bench"))]
pub const SECURITY_ASSUMPTION_LABEL: &str = "JohnsonBound";

const NUM_PRIVATE_COLUMNS: usize = 4;
const NUM_BLOCK_BITS: usize = 3;

// `WHIR`, `BENC`, `PCS1`, version 1. Every word is canonical in BabyBear.
const BENCHMARK_DOMAIN_TAG: [u32; 4] = [0x5748_4952, 0x4245_4e43, 0x5043_5331, 1];
// `WHIR`, `TERM`, `MLE1`, version 1. Every word is canonical in BabyBear.
const TERMINAL_DOMAIN_TAG: [u32; 4] = [0x5748_4952, 0x5445_524d, 0x4d4c_4531, 1];

#[derive(Debug, Error)]
pub enum WhirAdapterError {
    #[error("WHIR capacity must be a non-zero power of two; got {0}")]
    InvalidCapacity(usize),

    #[error("WHIR capacity must be at least {minimum}; got {capacity}")]
    CapacityTooSmall { capacity: usize, minimum: usize },

    #[error(transparent)]
    Columns(#[from] plonky3::ColumnError),

    #[error(transparent)]
    Config(#[from] WhirConfigError),

    #[error(transparent)]
    Verification(#[from] VerifierError),

    #[error("the verifier did not replay the prover's {0} challenge")]
    TranscriptClaimMismatch(&'static str),

    #[error("WHIR returned an opening batch with an unexpected shape")]
    UnexpectedOpeningShape,

    #[error("the four WHIR openings do not satisfy the terminal assignment-MLE claim")]
    TerminalClaimMismatch,

    #[error("postcard serialization failed: {0}")]
    Serialization(String),
}

/// One native scaled claim `D * f(x, beta) = V`.
///
/// Both coordinate vectors use Spartan/BitZ's little-endian logical ordering:
/// `gate_point_lsb_first[0]` selects the adjacent pair of gate-table entries,
/// and `beta_lsb_first` selects blocks `000=e0, 001=A, ..., 100=K`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalClaim {
    gate_point_lsb_first: Vec<Challenge>,
    beta_lsb_first: [Challenge; NUM_BLOCK_BITS],
    scale: Challenge,
    value: Challenge,
}

impl TerminalClaim {
    pub fn gate_point_lsb_first(&self) -> &[Challenge] {
        &self.gate_point_lsb_first
    }

    pub const fn beta_lsb_first(&self) -> &[Challenge; NUM_BLOCK_BITS] {
        &self.beta_lsb_first
    }

    pub const fn scale(&self) -> Challenge {
        self.scale
    }

    pub const fn value(&self) -> Challenge {
        self.value
    }
}

/// Static WHIR configuration. Construct this once; setup is outside all phase timers.
pub struct WhirBackend {
    pcs: Pcs,
    protocol: OpeningProtocol,
    domain_separator: DomainSeparator<Challenge, Val>,
    base_challenger: Challenger,
    capacity: usize,
    gate_vars: usize,
    folding: usize,
}

/// Compact preflight metadata derived by `WhirConfig` for this shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecuritySummary {
    pub target_bits: usize,
    pub configured_max_pow_bits: usize,
    pub derived_max_pow_bits: usize,
    pub commitment_ood_samples: usize,
    pub starting_folding_pow_bits: usize,
    pub folding_schedule: Vec<usize>,
    pub round_queries: Vec<usize>,
    pub round_ood_samples: Vec<usize>,
    pub round_folding_factors: Vec<usize>,
    pub round_log_inverse_rates: Vec<usize>,
    pub round_pow_bits: Vec<usize>,
    pub round_folding_pow_bits: Vec<usize>,
    pub final_queries: usize,
    pub final_pow_bits: usize,
    pub final_sumcheck_rounds: usize,
    pub final_folding_pow_bits: usize,
    pub folding_factor: usize,
    pub starting_log_inverse_rate: usize,
}

/// Result of the timed integer-to-field materialization phase.
pub struct MaterializedWitness {
    witness: Witness,
}

/// Result of the timed commitment phase.
pub struct CommittedWitness {
    commitment: Commitment,
    prover_data: ProverData,
    prover_challenger: Challenger,
    trial_seed: u64,
}

impl CommittedWitness {
    pub const fn commitment(&self) -> &Commitment {
        &self.commitment
    }

    pub const fn trial_seed(&self) -> u64 {
        self.trial_seed
    }
}

/// Commitment plus a terminal claim whose prover and verifier transcripts are
/// both ready immediately before the prescribed WHIR opening.
pub struct ReadyTerminalClaim {
    commitment: Commitment,
    prover_data: ProverData,
    prover_challenger: Challenger,
    verifier_challenger: Challenger,
    opening_point: Point<Challenge>,
    claim: TerminalClaim,
    trial_seed: u64,
}

impl ReadyTerminalClaim {
    pub const fn commitment(&self) -> &Commitment {
        &self.commitment
    }

    pub const fn claim(&self) -> &TerminalClaim {
        &self.claim
    }

    pub const fn trial_seed(&self) -> u64 {
        self.trial_seed
    }
}

/// Output of the opening phase. It retains a verifier transcript already at
/// the post-claim boundary, so [`WhirBackend::verify`] measures only the PCS
/// verification and the final scalar linkage check.
pub struct OpenedProof {
    commitment: Commitment,
    proof: Proof,
    verifier_challenger: Challenger,
    opening_point: Point<Challenge>,
    claim: TerminalClaim,
    trial_seed: u64,
}

impl OpenedProof {
    pub const fn commitment(&self) -> &Commitment {
        &self.commitment
    }

    pub const fn proof(&self) -> &Proof {
        &self.proof
    }

    pub const fn claim(&self) -> &TerminalClaim {
        &self.claim
    }

    pub const fn trial_seed(&self) -> u64 {
        self.trial_seed
    }
}

/// The four direct column evaluations authenticated by WHIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedOpening {
    pub a: Challenge,
    pub b: Challenge,
    pub c: Challenge,
    pub k: Challenge,
}

impl VerifiedOpening {
    pub const fn as_array(self) -> [Challenge; NUM_PRIVATE_COLUMNS] {
        [self.a, self.b, self.c, self.k]
    }
}

impl WhirBackend {
    /// Performs one-time public setup for four columns of `capacity` entries.
    pub fn setup(capacity: usize) -> Result<Self, WhirAdapterError> {
        Self::setup_with_params(capacity, FOLDING, STARTING_LOG_INV_RATE, MAX_POW_BITS)
    }

    pub fn setup_with_params(
        capacity: usize,
        folding: usize,
        starting_log_inv_rate: usize,
        max_pow_bits: usize,
    ) -> Result<Self, WhirAdapterError> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(WhirAdapterError::InvalidCapacity(capacity));
        }
        let minimum = 1usize << folding;
        if capacity < minimum {
            return Err(WhirAdapterError::CapacityTooSmall { capacity, minimum });
        }

        let gate_vars = capacity.trailing_zeros() as usize;
        // Four equal-width columns occupy four contiguous selector slots.
        let committed_num_variables = gate_vars + 2;
        let folding_factor = FoldingFactor::Constant(folding);
        let params = ProtocolParameters {
            security_level: SECURITY_BITS,
            pow_bits: max_pow_bits,
            // Empty means: derive Plonky3's standard per-round rate schedule.
            round_log_inv_rates: Vec::new(),
            folding_factor,
            soundness_type: SECURITY_ASSUMPTION,
            starting_log_inv_rate,
        };
        let pcs = stack::pcs::<Challenge>(committed_num_variables, params)?;
        let protocol = plonky3::column_opening(gate_vars, NUM_PRIVATE_COLUMNS);

        let mut domain_separator = DomainSeparator::new(Vec::new());
        pcs.add_domain_separator::<8>(&mut domain_separator);

        Ok(Self {
            pcs,
            protocol,
            domain_separator,
            base_challenger: stack::challenger(),
            capacity,
            gate_vars,
            folding,
        })
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn gate_vars(&self) -> usize {
        self.gate_vars
    }

    /// Largest grinding step actually derived for this shape.
    pub fn max_pow_bits(&self) -> usize {
        self.pcs.config.max_pow_bits()
    }

    /// Security and query metadata for benchmark preflight and result rows.
    pub fn security_summary(&self) -> SecuritySummary {
        let config = &self.pcs.config;
        SecuritySummary {
            target_bits: config.params.security_level,
            configured_max_pow_bits: config.params.pow_bits,
            derived_max_pow_bits: config.max_pow_bits(),
            commitment_ood_samples: config.commitment_ood_samples,
            starting_folding_pow_bits: config.starting_folding_pow_bits,
            folding_schedule: config.folding_schedule.clone(),
            round_queries: config
                .round_parameters
                .iter()
                .map(|round| round.num_queries)
                .collect(),
            round_ood_samples: config
                .round_parameters
                .iter()
                .map(|round| round.ood_samples)
                .collect(),
            round_folding_factors: config
                .round_parameters
                .iter()
                .map(|round| round.folding_factor)
                .collect(),
            round_log_inverse_rates: config
                .round_parameters
                .iter()
                .map(|round| round.log_inv_rate)
                .collect(),
            round_pow_bits: config
                .round_parameters
                .iter()
                .map(|round| round.pow_bits)
                .collect(),
            round_folding_pow_bits: config
                .round_parameters
                .iter()
                .map(|round| round.folding_pow_bits)
                .collect(),
            final_queries: config.final_queries,
            final_pow_bits: config.final_pow_bits,
            final_sumcheck_rounds: config.final_sumcheck_rounds,
            final_folding_pow_bits: config.final_folding_pow_bits,
            folding_factor: self.folding,
            starting_log_inverse_rate: config.params.starting_log_inv_rate,
        }
    }

    /// Materializes four complete columns. Values are rejected rather than
    /// silently reduced, so the commitment binds the original integers.
    pub fn materialize(
        &self,
        a: &[u64],
        b: &[u64],
        c: &[u64],
        k: &[u64],
    ) -> Result<MaterializedWitness, WhirAdapterError> {
        let table =
            plonky3::column_table::<Val>(self.capacity, &[("A", a), ("B", b), ("C", c), ("K", k)])?;
        let witness = WhirLayout::new_witness(vec![table], self.folding);
        debug_assert_eq!(witness.table_shapes(), self.protocol.table_shapes());
        Ok(MaterializedWitness { witness })
    }

    /// Commits the materialized columns and retains the transcript immediately
    /// after the root. `commit` itself absorbs that root exactly once.
    pub fn commit(&self, materialized: MaterializedWitness, trial_seed: u64) -> CommittedWitness {
        let mut prover_challenger = self.challenger(trial_seed);
        let (commitment, prover_data) = <Pcs as MultilinearPcs<Challenge, Challenger>>::commit(
            &self.pcs,
            materialized.witness,
            &mut prover_challenger,
        );
        CommittedWitness {
            commitment,
            prover_data,
            prover_challenger,
            trial_seed,
        }
    }

    /// Samples independent native-field equality challenges after commitment,
    /// derives the honest scaled claim, binds it, and prepares the matching
    /// verifier transcript. This entire phase belongs before an `open_at` timer.
    pub fn derive_and_bind_terminal_claim(
        &self,
        mut committed: CommittedWitness,
    ) -> Result<ReadyTerminalClaim, WhirAdapterError> {
        observe_terminal_domain(&mut committed.prover_challenger, self.gate_vars);

        let gate_point_lsb_first = (0..self.gate_vars)
            .map(|_| committed.prover_challenger.sample_algebra_element())
            .collect::<Vec<Challenge>>();
        let beta_lsb_first = std::array::from_fn(|_| {
            committed
                .prover_challenger
                .sample_algebra_element::<Challenge>()
        });
        let scale = committed
            .prover_challenger
            .sample_algebra_element::<Challenge>();
        let opening_point = p3_opening_point(&gate_point_lsb_first);

        // This untimed fixture derivation models the terminal value already
        // supplied by Spartan. `open_at` will independently evaluate the same
        // four columns inside the measured opening phase.
        let table = committed.prover_data.table(0);
        let opened = std::array::from_fn(|column| table.poly(column).eval_base(&opening_point));
        let mut claim = TerminalClaim {
            gate_point_lsb_first,
            beta_lsb_first,
            scale,
            value: Challenge::ZERO,
        };
        claim.value = terminal_lhs(&claim, opened);
        observe_terminal_claim(&mut committed.prover_challenger, &claim);

        // Build the verifier state now, so verify timing begins from the same
        // ready terminal claim rather than re-counting transcript preparation.
        let mut verifier_challenger = self.challenger(committed.trial_seed);
        verifier_challenger.observe(committed.commitment.clone());
        replay_and_bind_terminal_claim(&mut verifier_challenger, self.gate_vars, &claim)?;

        Ok(ReadyTerminalClaim {
            commitment: committed.commitment,
            prover_data: committed.prover_data,
            prover_challenger: committed.prover_challenger,
            verifier_challenger,
            opening_point,
            claim,
            trial_seed: committed.trial_seed,
        })
    }

    /// Proves the four prescribed evaluations. The input transcript is already
    /// positioned after the complete terminal claim.
    pub fn open(&self, ready: ReadyTerminalClaim) -> OpenedProof {
        let ReadyTerminalClaim {
            commitment,
            prover_data,
            mut prover_challenger,
            verifier_challenger,
            opening_point,
            claim,
            trial_seed,
        } = ready;
        let proof = self.pcs.open_at(
            prover_data,
            &self.protocol,
            std::slice::from_ref(&opening_point),
            &mut prover_challenger,
        );
        OpenedProof {
            commitment,
            proof,
            verifier_challenger,
            opening_point,
            claim,
            trial_seed,
        }
    }

    /// Verifies the WHIR proof, then enforces the public linkage back to
    /// `D * f(x, beta) = V`. No division by `D` occurs, so `D = 0` is valid.
    pub fn verify(&self, opened: &OpenedProof) -> Result<VerifiedOpening, WhirAdapterError> {
        let mut challenger = opened.verifier_challenger.clone();
        let evals = self.pcs.verify_at(
            &opened.commitment,
            &opened.proof,
            &self.protocol,
            std::slice::from_ref(&opened.opening_point),
            &mut challenger,
        )?;

        let [batch] = evals.as_slice() else {
            return Err(WhirAdapterError::UnexpectedOpeningShape);
        };
        if !batch.next().is_empty() || batch.current().len() != NUM_PRIVATE_COLUMNS {
            return Err(WhirAdapterError::UnexpectedOpeningShape);
        }
        let values: [Challenge; NUM_PRIVATE_COLUMNS] = batch
            .current()
            .try_into()
            .map_err(|_| WhirAdapterError::UnexpectedOpeningShape)?;

        if terminal_lhs(&opened.claim, values) != opened.claim.value {
            return Err(WhirAdapterError::TerminalClaimMismatch);
        }

        Ok(VerifiedOpening {
            a: values[0],
            b: values[1],
            c: values[2],
            k: values[3],
        })
    }

    fn challenger(&self, trial_seed: u64) -> Challenger {
        let mut challenger = self.base_challenger.clone();
        self.domain_separator
            .observe_domain_separator(&mut challenger);
        observe_benchmark_domain_and_seed(&mut challenger, trial_seed);
        challenger
    }
}

/// Postcard size of the initial public commitment. This is reported separately
/// from the opening proof.
pub fn commitment_bytes(commitment: &Commitment) -> Result<usize, WhirAdapterError> {
    postcard::to_allocvec(commitment)
        .map(|bytes| bytes.len())
        .map_err(|error| WhirAdapterError::Serialization(error.to_string()))
}

/// Postcard size of the WHIR opening proof, including its four claimed values.
pub fn proof_bytes(proof: &Proof) -> Result<usize, WhirAdapterError> {
    postcard::to_allocvec(proof)
        .map(|bytes| bytes.len())
        .map_err(|error| WhirAdapterError::Serialization(error.to_string()))
}

/// Converts a low-coordinate-first gate point into Plonky3's lexicographic,
/// big-endian point convention.
fn p3_opening_point(gate_point_lsb_first: &[Challenge]) -> Point<Challenge> {
    plonky3::opening_point(gate_point_lsb_first)
}

/// Computes the left side of the terminal claim from four authenticated column
/// values. Block-selector order is little-endian:
/// `000=e0, 001=A, 010=B, 011=C, 100=K`.
fn terminal_lhs(claim: &TerminalClaim, opened: [Challenge; NUM_PRIVATE_COLUMNS]) -> Challenge {
    claim.scale
        * plonky3::assignment_eval(&claim.gate_point_lsb_first, &claim.beta_lsb_first, &opened)
}

fn observe_terminal_domain(challenger: &mut Challenger, gate_vars: usize) {
    for word in TERMINAL_DOMAIN_TAG {
        challenger.observe(Val::from_u32(word));
    }
    challenger.observe(Val::from_usize(gate_vars));
}

fn observe_benchmark_domain_and_seed(challenger: &mut Challenger, trial_seed: u64) {
    for word in BENCHMARK_DOMAIN_TAG {
        challenger.observe(Val::from_u32(word));
    }

    // Three radix-2^30 limbs encode all 64 seed bits injectively, without
    // reducing a u32 modulo the 31-bit BabyBear prime.
    const LIMB_BITS: usize = 30;
    const LIMB_MASK: u64 = (1_u64 << LIMB_BITS) - 1;
    for shift in [0, LIMB_BITS, 2 * LIMB_BITS] {
        challenger.observe(Val::from_u64((trial_seed >> shift) & LIMB_MASK));
    }
}

fn observe_terminal_claim(challenger: &mut Challenger, claim: &TerminalClaim) {
    challenger.observe_algebra_slice(&claim.gate_point_lsb_first);
    challenger.observe_algebra_slice(&claim.beta_lsb_first);
    challenger.observe_algebra_element(claim.scale);
    challenger.observe_algebra_element(claim.value);
}

fn replay_and_bind_terminal_claim(
    challenger: &mut Challenger,
    gate_vars: usize,
    claim: &TerminalClaim,
) -> Result<(), WhirAdapterError> {
    observe_terminal_domain(challenger, gate_vars);
    let gate_point = (0..gate_vars)
        .map(|_| challenger.sample_algebra_element())
        .collect::<Vec<Challenge>>();
    if gate_point != claim.gate_point_lsb_first {
        return Err(WhirAdapterError::TranscriptClaimMismatch("gate-point"));
    }
    let beta = std::array::from_fn(|_| challenger.sample_algebra_element::<Challenge>());
    if beta != claim.beta_lsb_first {
        return Err(WhirAdapterError::TranscriptClaimMismatch("block-point"));
    }
    let scale = challenger.sample_algebra_element::<Challenge>();
    if scale != claim.scale {
        return Err(WhirAdapterError::TranscriptClaimMismatch("scale"));
    }
    observe_terminal_claim(challenger, claim);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_changed_claim_and_openings() {
        use p3_sumcheck::OpeningBatch;
        let backend = WhirBackend::setup(16).unwrap();
        let witness = backend
            .materialize(&[2; 16], &[3; 16], &[6; 16], &[0; 16])
            .unwrap();
        let committed = backend.commit(witness, 42);
        let ready = backend.derive_and_bind_terminal_claim(committed).unwrap();
        let mut opened = backend.open(ready);
        backend.verify(&opened).unwrap();
        opened.claim.value += Challenge::ONE;
        assert!(matches!(
            backend.verify(&opened),
            Err(WhirAdapterError::TerminalClaimMismatch)
        ));
        opened.claim.value -= Challenge::ONE;
        opened.proof.evals[0] =
            OpeningBatch::new(vec![Challenge::ZERO; NUM_PRIVATE_COLUMNS], Vec::new());
        assert!(backend.verify(&opened).is_err());
    }

    fn challenge(value: u64) -> Challenge {
        Challenge::from_u64(value)
    }

    #[test]
    fn p3_point_reverses_spartan_gate_coordinates_once() {
        let low_first = vec![challenge(2), challenge(3), challenge(5)];
        let point = p3_opening_point(&low_first);
        assert_eq!(
            point.as_slice(),
            &[challenge(5), challenge(3), challenge(2)]
        );
    }

    #[test]
    fn zero_scale_is_checked_without_division() {
        let claim = TerminalClaim {
            gate_point_lsb_first: vec![challenge(2), challenge(7)],
            beta_lsb_first: [challenge(3), challenge(5), challenge(11)],
            scale: Challenge::ZERO,
            value: Challenge::ZERO,
        };
        let opened = [challenge(13), challenge(17), challenge(19), challenge(23)];
        assert_eq!(terminal_lhs(&claim, opened), claim.value);
    }

    #[test]
    fn terminal_selector_uses_all_four_private_blocks() {
        let opened = [challenge(13), challenge(17), challenge(19), challenge(23)];
        for (block, expected) in opened.into_iter().enumerate() {
            let index = block + 1;
            let claim = TerminalClaim {
                gate_point_lsb_first: vec![Challenge::ONE],
                beta_lsb_first: [
                    Challenge::from_bool(index & 1 != 0),
                    Challenge::from_bool(index & 2 != 0),
                    Challenge::from_bool(index & 4 != 0),
                ],
                scale: Challenge::ONE,
                value: expected,
            };
            assert_eq!(terminal_lhs(&claim, opened), expected);
        }
    }

    #[test]
    fn verified_opening_array_order_is_a_b_c_k() {
        let opening = VerifiedOpening {
            a: challenge(1),
            b: challenge(2),
            c: challenge(3),
            k: challenge(4),
        };
        assert_eq!(
            opening.as_array(),
            [challenge(1), challenge(2), challenge(3), challenge(4)]
        );
    }
}
