//! Plonky3 WHIR adapter for exact u32 multiplication assignments.
//!
//! The three committed columns are `X || Y || Product`. Every value is
//! injected canonically into Goldilocks; the largest possible u32 product is
//! `(2^32-1)^2`, which is strictly below the Goldilocks modulus.

#![allow(dead_code)]

use crate::common::plonky3::{self, goldilocks as stack};
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

pub type Val = stack::Val;
#[cfg(feature = "plonky3-whir-goldilocks-degree2-bench")]
pub const CHALLENGE_EXTENSION_DEGREE: usize = 2;
#[cfg(not(feature = "plonky3-whir-goldilocks-degree2-bench"))]
pub const CHALLENGE_EXTENSION_DEGREE: usize = 5;
pub type Challenge = BinomialExtensionField<Val, CHALLENGE_EXTENSION_DEGREE>;

type Challenger = stack::Challenger;
type WhirLayout = SuffixProver<Val, Challenge>;
type Pcs = stack::Pcs<Challenge>;

pub type Commitment = <Pcs as MultilinearPcs<Challenge, Challenger>>::Commitment;
pub type Proof = <Pcs as MultilinearPcs<Challenge, Challenger>>::Proof;
type ProverData = <Pcs as MultilinearPcs<Challenge, Challenger>>::ProverData;
type Witness = <Pcs as MultilinearPcs<Challenge, Challenger>>::Witness;

/// Six extra internal bits leave room for the union of protocol events while
/// retaining a user-visible target of at least 100 bits.
pub const SECURITY_BITS: usize = 106;
pub const MAX_POW_BITS: usize = 12;
pub const FOLDING: usize = 4;
pub const STARTING_LOG_INV_RATE: usize = 1;
#[cfg(feature = "plonky3-whir-goldilocks-degree2-bench")]
pub const SECURITY_ASSUMPTION: SecurityAssumption = SecurityAssumption::UniqueDecoding;
#[cfg(not(feature = "plonky3-whir-goldilocks-degree2-bench"))]
pub const SECURITY_ASSUMPTION: SecurityAssumption = SecurityAssumption::JohnsonBound;
#[cfg(feature = "plonky3-whir-goldilocks-degree2-bench")]
pub const SECURITY_ASSUMPTION_LABEL: &str = "UniqueDecoding";
#[cfg(not(feature = "plonky3-whir-goldilocks-degree2-bench"))]
pub const SECURITY_ASSUMPTION_LABEL: &str = "JohnsonBound";

const NUM_COLUMNS: usize = 3;
const NUM_BLOCK_BITS: usize = 2;
const BENCHMARK_DOMAIN_TAG: [u32; 4] = [0x5748_4952, 0x5533_324d, 0x5043_5331, 1];
const TERMINAL_DOMAIN_TAG: [u32; 4] = [0x5748_4952, 0x5533_3254, 0x4d4c_4531, 1];

#[derive(Debug, Error)]
pub enum Error {
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
    #[error("the three WHIR openings do not satisfy the u32 terminal claim")]
    TerminalClaimMismatch,
    #[error("postcard serialization failed: {0}")]
    Serialization(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalClaim {
    gate_point_lsb_first: Vec<Challenge>,
    beta_lsb_first: [Challenge; NUM_BLOCK_BITS],
    scale: Challenge,
    value: Challenge,
}

pub struct Backend {
    pcs: Pcs,
    protocol: OpeningProtocol,
    domain_separator: DomainSeparator<Challenge, Val>,
    base_challenger: Challenger,
    capacity: usize,
    gate_vars: usize,
    folding: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecuritySummary {
    pub target_bits: usize,
    pub configured_max_pow_bits: usize,
    pub derived_max_pow_bits: usize,
    pub commitment_ood_samples: usize,
    pub round_queries: Vec<usize>,
    pub round_pow_bits: Vec<usize>,
    pub final_queries: usize,
    pub final_pow_bits: usize,
    pub folding_factor: usize,
    pub starting_log_inverse_rate: usize,
}

pub struct MaterializedWitness {
    witness: Witness,
}

pub struct CommittedWitness {
    commitment: Commitment,
    prover_data: ProverData,
    prover_challenger: Challenger,
    trial_seed: u64,
}

pub struct ReadyClaim {
    commitment: Commitment,
    prover_data: ProverData,
    prover_challenger: Challenger,
    verifier_challenger: Challenger,
    opening_point: Point<Challenge>,
    claim: TerminalClaim,
    trial_seed: u64,
}

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
}

impl Backend {
    pub fn setup(capacity: usize) -> Result<Self, Error> {
        Self::setup_with_params(capacity, FOLDING, STARTING_LOG_INV_RATE, MAX_POW_BITS)
    }

    pub fn setup_with_params(
        capacity: usize,
        folding: usize,
        starting_log_inv_rate: usize,
        max_pow_bits: usize,
    ) -> Result<Self, Error> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(Error::InvalidCapacity(capacity));
        }
        let minimum = 1usize << folding;
        if capacity < minimum {
            return Err(Error::CapacityTooSmall { capacity, minimum });
        }
        let gate_vars = capacity.trailing_zeros() as usize;
        let params = ProtocolParameters {
            security_level: SECURITY_BITS,
            pow_bits: max_pow_bits,
            round_log_inv_rates: Vec::new(),
            folding_factor: FoldingFactor::Constant(folding),
            soundness_type: SECURITY_ASSUMPTION,
            starting_log_inv_rate,
        };
        let pcs = stack::pcs::<Challenge>(gate_vars + 2, params)?;
        let protocol = plonky3::column_opening(gate_vars, NUM_COLUMNS);

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

    pub fn max_pow_bits(&self) -> usize {
        self.pcs.config.max_pow_bits()
    }

    pub fn security_summary(&self) -> SecuritySummary {
        let config = &self.pcs.config;
        SecuritySummary {
            target_bits: config.params.security_level,
            configured_max_pow_bits: config.params.pow_bits,
            derived_max_pow_bits: config.max_pow_bits(),
            commitment_ood_samples: config.commitment_ood_samples,
            round_queries: config
                .round_parameters
                .iter()
                .map(|round| round.num_queries)
                .collect(),
            round_pow_bits: config
                .round_parameters
                .iter()
                .map(|round| round.pow_bits)
                .collect(),
            final_queries: config.final_queries,
            final_pow_bits: config.final_pow_bits,
            folding_factor: self.folding,
            starting_log_inverse_rate: config.params.starting_log_inv_rate,
        }
    }

    pub fn materialize(
        &self,
        witness: &bitz::piop::spartan::MulWitness<u32>,
    ) -> Result<MaterializedWitness, Error> {
        let capacity = witness.layout().capacity();
        if capacity != self.capacity {
            return Err(plonky3::ColumnError::Length {
                column: "X",
                expected: self.capacity,
                actual: capacity,
            }
            .into());
        }
        let table = plonky3::column_table_from_fn::<Val>(capacity, 3, |col, i| match col {
            0 => ("X", u64::from(witness.x_values()[i])),
            1 => ("Y", u64::from(witness.y_values()[i])),
            _ => ("Product", witness.product(i)),
        })?;
        let witness = WhirLayout::new_witness(vec![table], self.folding);
        debug_assert_eq!(witness.table_shapes(), self.protocol.table_shapes());
        Ok(MaterializedWitness { witness })
    }

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

    pub fn derive_and_bind_claim(
        &self,
        mut committed: CommittedWitness,
    ) -> Result<ReadyClaim, Error> {
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
        let opening_point = plonky3::opening_point(&gate_point_lsb_first);
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

        let mut verifier_challenger = self.challenger(committed.trial_seed);
        verifier_challenger.observe(committed.commitment.clone());
        replay_and_bind_claim(&mut verifier_challenger, self.gate_vars, &claim)?;
        Ok(ReadyClaim {
            commitment: committed.commitment,
            prover_data: committed.prover_data,
            prover_challenger: committed.prover_challenger,
            verifier_challenger,
            opening_point,
            claim,
            trial_seed: committed.trial_seed,
        })
    }

    pub fn open(&self, ready: ReadyClaim) -> OpenedProof {
        let ReadyClaim {
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

    pub fn verify(&self, opened: &OpenedProof) -> Result<[Challenge; NUM_COLUMNS], Error> {
        let mut challenger = opened.verifier_challenger.clone();
        let evals = self.pcs.verify_at(
            &opened.commitment,
            &opened.proof,
            &self.protocol,
            std::slice::from_ref(&opened.opening_point),
            &mut challenger,
        )?;
        let [batch] = evals.as_slice() else {
            return Err(Error::UnexpectedOpeningShape);
        };
        if !batch.next().is_empty() || batch.current().len() != NUM_COLUMNS {
            return Err(Error::UnexpectedOpeningShape);
        }
        let values: [Challenge; NUM_COLUMNS] = batch
            .current()
            .try_into()
            .map_err(|_| Error::UnexpectedOpeningShape)?;
        if terminal_lhs(&opened.claim, values) != opened.claim.value {
            return Err(Error::TerminalClaimMismatch);
        }
        Ok(values)
    }

    fn challenger(&self, trial_seed: u64) -> Challenger {
        let mut challenger = self.base_challenger.clone();
        self.domain_separator
            .observe_domain_separator(&mut challenger);
        for word in BENCHMARK_DOMAIN_TAG {
            challenger.observe(Val::from_u32(word));
        }
        challenger.observe(Val::from_u32(trial_seed as u32));
        challenger.observe(Val::from_u32((trial_seed >> 32) as u32));
        challenger
    }
}

pub fn commitment_bytes(commitment: &Commitment) -> Result<usize, Error> {
    postcard::to_allocvec(commitment)
        .map(|bytes| bytes.len())
        .map_err(|error| Error::Serialization(error.to_string()))
}

pub fn proof_bytes(proof: &Proof) -> Result<usize, Error> {
    postcard::to_allocvec(proof)
        .map(|bytes| bytes.len())
        .map_err(|error| Error::Serialization(error.to_string()))
}

fn terminal_lhs(claim: &TerminalClaim, opened: [Challenge; NUM_COLUMNS]) -> Challenge {
    claim.scale
        * plonky3::assignment_eval(&claim.gate_point_lsb_first, &claim.beta_lsb_first, &opened)
}

fn observe_terminal_domain(challenger: &mut Challenger, gate_vars: usize) {
    for word in TERMINAL_DOMAIN_TAG {
        challenger.observe(Val::from_u32(word));
    }
    challenger.observe(Val::from_usize(gate_vars));
}

fn observe_terminal_claim(challenger: &mut Challenger, claim: &TerminalClaim) {
    challenger.observe_algebra_slice(&claim.gate_point_lsb_first);
    challenger.observe_algebra_slice(&claim.beta_lsb_first);
    challenger.observe_algebra_element(claim.scale);
    challenger.observe_algebra_element(claim.value);
}

fn replay_and_bind_claim(
    challenger: &mut Challenger,
    gate_vars: usize,
    claim: &TerminalClaim,
) -> Result<(), Error> {
    observe_terminal_domain(challenger, gate_vars);
    let gate_point = (0..gate_vars)
        .map(|_| challenger.sample_algebra_element())
        .collect::<Vec<Challenge>>();
    if gate_point != claim.gate_point_lsb_first {
        return Err(Error::TranscriptClaimMismatch("gate-point"));
    }
    let beta = std::array::from_fn(|_| challenger.sample_algebra_element::<Challenge>());
    if beta != claim.beta_lsb_first {
        return Err(Error::TranscriptClaimMismatch("block-point"));
    }
    let scale = challenger.sample_algebra_element::<Challenge>();
    if scale != claim.scale {
        return Err(Error::TranscriptClaimMismatch("scale"));
    }
    observe_terminal_claim(challenger, claim);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_changed_claim_and_openings() {
        use super::*;
        let backend = Backend::setup(256).unwrap();
        let witness = backend
            .materialize(&bitz::piop::spartan::MulWitness::<u32>::from_fn(256, |_| (2, 3)).unwrap())
            .unwrap();
        let committed = backend.commit(witness, 42);
        let ready = backend.derive_and_bind_claim(committed).unwrap();
        let mut opened = backend.open(ready);
        let _ = backend.verify(&opened).unwrap();
        opened.claim.value += Challenge::ONE;
        assert!(matches!(
            backend.verify(&opened),
            Err(Error::TerminalClaimMismatch)
        ));
        opened.claim.value -= Challenge::ONE;
        opened.proof.evals[0] =
            p3_sumcheck::OpeningBatch::new(vec![Challenge::ZERO; NUM_COLUMNS], Vec::new());
        assert!(backend.verify(&opened).is_err());
    }
}
