//! Binius64 BaseFold adapter for an exact 128-bit packed integer witness.
//!
//! One `B128` row stores one logical integer tuple.  Ring switching interprets
//! those rows as a single bit-MLE with seven low coordinates selecting the bit
//! inside a row and the remaining coordinates selecting the row.  The timed
//! opening is the complete ring-switch reduction plus BaseFold proof.

#![allow(dead_code)]

use std::error::Error;

use binius_field::{Ghash128b, arch::OptimalPackedB128};
use binius_hash::StdHashSuite;
use binius_iop::{
    basefold::compiler::BaseFoldVerifierCompiler,
    channel::{IOPVerifierChannel, OracleSpec},
    fri::MinProofSizeStrategy,
    merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::{basefold::compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
    FieldBuffer,
    multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate_inplace_scalars},
    ntt::{NeighborsLastMultiThread, domain_context::GaoMateerPreExpanded},
};
use binius_prover::ring_switch;
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{config::StdChallenger, ring_switch as verifier_ring_switch};

pub type B128 = Ghash128b;
type P = OptimalPackedB128;
type Ntt = NeighborsLastMultiThread<GaoMateerPreExpanded<B128>>;

/// Target for the Diamond--Posen/BaseFold query calculation.
pub const SECURITY_BITS: usize = 100;
/// Rate 1/2 is the default; the comparison driver can sweep other rates.
pub const DEFAULT_LOG_INV_RATE: usize = 1;
pub const COMMITMENT_BYTES: usize = 32;
pub const PACKING_BITS: usize = 128;

const DOMAIN_TAG: &[u8] = b"bitz/integer-pcs-compare/binius64/v1";

const ROOT_SCOPE: &str = "pcs-compare:verified_trial";
const MATERIALIZE_SCOPE: &str = "pcs-compare:materialize";
const COMMIT_SCOPE: &str = "pcs-compare:commit";
const CLAIM_SCOPE: &str = "pcs-compare:claim_setup";
const OPENING_SCOPE: &str = "pcs-compare:opening";
const VERIFY_SCOPE: &str = "pcs-compare:verification";

pub const COMMIT_ORACLE_SCOPE: &str = "binius:basefold_commit_oracle";
pub const SAMPLE_POINT_SCOPE: &str = "binius:sample_opening_point";
pub const EVALUATE_CLAIM_SCOPE: &str = "binius:evaluate_bit_mle";
pub const RING_SWITCH_SCOPE: &str = "binius:ring_switch_reduction";
pub const BASEFOLD_OPEN_SCOPE: &str = "binius:basefold_opening";
pub const RING_SWITCH_VERIFY_SCOPE: &str = "binius:verify_ring_switch";
pub const BASEFOLD_VERIFY_SCOPE: &str = "binius:verify_basefold";

pub struct BiniusBackend {
    verifier: BaseFoldVerifierCompiler<B128>,
    prover: BaseFoldProverCompiler<P, Ntt>,
    /// Working buffers come from a pool held for the backend's lifetime, as
    /// Binius64's own `Prover` does (`prove.rs`: `let alloc = &self.pool`).
    /// Measuring it against `GlobalAllocator` elsewhere in this repo showed
    /// the malloc path costs the Binius prover ~14 % at 2^20-scale buffers,
    /// so using the pool keeps this comparison fair to Binius64.
    pool: binius_compute::BufferPool,
    log_rows: usize,
    log_inv_rate: usize,
    n_test_queries: usize,
}

pub struct TrialOutput {
    pub proof_bytes: usize,
    pub commitment_bytes: usize,
    pub public_claim_bytes: usize,
}

impl TrialOutput {
    pub fn opening_proof_bytes(&self) -> usize {
        self.proof_bytes.saturating_sub(self.commitment_bytes)
    }

    pub fn total_wire_bytes(&self) -> usize {
        self.proof_bytes + self.public_claim_bytes
    }
}

impl BiniusBackend {
    pub fn setup(log_rows: usize, log_inv_rate: usize) -> Self {
        assert!(
            log_inv_rate > 0,
            "BaseFold inverse rate must be at least two"
        );
        let n_test_queries = min_test_queries(log_rows, log_inv_rate);
        let scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
        let verifier = BaseFoldVerifierCompiler::new(
            &scheme,
            vec![OracleSpec::new(log_rows)],
            log_inv_rate,
            n_test_queries,
            &MinProofSizeStrategy,
        );
        let domain = GaoMateerPreExpanded::generate(verifier.max_log_domain_size());
        let threads = binius_utils::rayon::current_num_threads().max(1);
        let log_num_shares = threads.ilog2() as usize;
        let ntt = NeighborsLastMultiThread::new(domain, log_num_shares);
        let prover = BaseFoldProverCompiler::from_verifier_compiler(&verifier, ntt);
        Self {
            verifier,
            prover,
            pool: binius_compute::BufferPool::new(),
            log_rows,
            log_inv_rate,
            n_test_queries,
        }
    }

    pub const fn log_rows(&self) -> usize {
        self.log_rows
    }

    pub const fn log_inv_rate(&self) -> usize {
        self.log_inv_rate
    }

    pub const fn n_test_queries(&self) -> usize {
        self.n_test_queries
    }

    pub fn estimated_soundness_bits(&self) -> f64 {
        dp24_eq42_soundness_bits(self.log_rows, self.log_inv_rate, self.n_test_queries)
    }

    /// Runs one independently verified opening while emitting the canonical
    /// comparison scopes. `packed_rows` must contain exactly `2^log_rows`
    /// exact integer encodings.
    pub fn run_trial(
        &self,
        materialize_rows: impl FnOnce() -> Vec<u128>,
        trial_seed: u64,
    ) -> Result<TrialOutput, Box<dyn Error>> {
        let root = tracing::info_span!(ROOT_SCOPE).entered();
        let witness = {
            let _phase = tracing::info_span!(MATERIALIZE_SCOPE).entered();
            let packed_rows = materialize_rows();
            if packed_rows.len() != 1usize << self.log_rows {
                return Err(format!(
                    "Binius row count {} does not match setup shape 2^{}",
                    packed_rows.len(),
                    self.log_rows
                )
                .into());
            }
            let scalars = packed_rows
                .iter()
                .copied()
                .map(B128::new)
                .collect::<Vec<_>>();
            // Drawn from the same pool the channel allocates from, so the
            // oracle buffer is recyclable and type-matches `finalize_oracle`.
            FieldBuffer::from_values_in(&&self.pool, &scalars)
        };

        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        observe_statement(&mut prover_transcript, trial_seed);
        let (mut prover_channel, oracle) = {
            let _phase = tracing::info_span!(COMMIT_SCOPE).entered();
            let mut channel = self
                .prover
                .create_channel_without_zk_from_transcript::<
                    StdHashSuite,
                    StdChallenger,
                    _,
                    &binius_compute::BufferPool,
                >(&mut prover_transcript, &self.pool);
            let oracle = {
                let _procedure = tracing::info_span!(COMMIT_ORACLE_SCOPE).entered();
                channel.send_oracle(witness.as_view())
            };
            (channel, oracle)
        };

        let (point, evaluation_claim) = {
            let _phase = tracing::info_span!(CLAIM_SCOPE).entered();
            let point = {
                let _procedure = tracing::info_span!(SAMPLE_POINT_SCOPE).entered();
                IPProverChannel::sample_many(&mut prover_channel, self.log_rows + 7)
            };
            let evaluation_claim = {
                let _procedure = tracing::info_span!(EVALUATE_CLAIM_SCOPE).entered();
                evaluate_bit_mle(&witness, &point)
            };
            (point, evaluation_claim)
        };

        let proof = {
            let _phase = tracing::info_span!(OPENING_SCOPE).entered();
            let ring_switch::RingSwitchOutput {
                rs_eq_ind,
                sumcheck_claim,
            } = {
                let _procedure = tracing::info_span!(RING_SWITCH_SCOPE).entered();
                ring_switch::prove(
                    &&self.pool,
                    witness.as_view(),
                    &point,
                    &mut prover_channel,
                )
            };
            prover_channel.prove_oracle_relation(oracle.clone(), rs_eq_ind, sumcheck_claim);
            prover_channel.finalize_oracle(oracle, witness);
            {
                let _procedure = tracing::info_span!(BASEFOLD_OPEN_SCOPE).entered();
                prover_channel.finish();
            }
            prover_transcript.finalize()
        };

        {
            let _phase = tracing::info_span!(VERIFY_SCOPE).entered();
            let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof.clone());
            observe_verifier_statement(&mut transcript, trial_seed);
            let mut channel = self
                .verifier
                .create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(&mut transcript);
            let oracle = channel.recv_oracle(self.log_rows, true)?;
            let verifier_point = IPVerifierChannel::sample_many(&mut channel, self.log_rows + 7);
            if verifier_point != point {
                return Err("Binius verifier did not replay the opening point".into());
            }
            let verifier_ring_switch::RingSwitchVerifyOutput {
                eq_r_double_prime,
                sumcheck_claim,
            } = {
                let _procedure = tracing::info_span!(RING_SWITCH_VERIFY_SCOPE).entered();
                verifier_ring_switch::verify::<B128, _>(
                    evaluation_claim,
                    &verifier_point,
                    &mut channel,
                )?
            };
            let high_point = verifier_point[PACKING_BITS.ilog2() as usize..].to_vec();
            {
                let _procedure = tracing::info_span!(BASEFOLD_VERIFY_SCOPE).entered();
                channel.verify_oracle_relation(
                    oracle,
                    Box::new(move |query: &[B128]| {
                        verifier_ring_switch::eval_rs_eq(&high_point, query, &eq_r_double_prime)
                    }),
                    sumcheck_claim,
                )?;
                channel.finish()?;
            }
            transcript.finalize()?;
        }
        std::hint::black_box(evaluation_claim);
        drop(root);

        let public_claim_bytes = (self.log_rows + 8) * 16;
        Ok(TrialOutput {
            proof_bytes: proof.len(),
            commitment_bytes: COMMITMENT_BYTES,
            public_claim_bytes,
        })
    }
}

/// Smallest integer query count whose Diamond--Posen Eq. 42 bound reaches the
/// requested target at this exact message shape and rate.
pub fn min_test_queries(log_rows: usize, log_inv_rate: usize) -> usize {
    (1..=usize::MAX)
        .find(|&queries| {
            dp24_eq42_soundness_bits(log_rows, log_inv_rate, queries) >= SECURITY_BITS as f64
        })
        .expect("a positive query count eventually reaches the target")
}

/// Concrete `-log2` soundness estimate used by the comparison report.
pub fn dp24_eq42_soundness_bits(
    log_rows: usize,
    log_inv_rate: usize,
    n_test_queries: usize,
) -> f64 {
    let field_size = 2_f64.powi(128);
    let ell = log_rows as f64;
    let rate_bits = log_inv_rate as f64;
    let first = (128.0 + 2.0 * ell) / field_size;
    let second = (128.0 / (field_size - 1.0)) * 2_f64.powf(ell + rate_bits) / field_size;
    let per_query = 0.5 + 1.0 / (2.0 * 2_f64.powf(rate_bits));
    -(first + second + per_query.powi(n_test_queries as i32)).log2()
}

fn observe_statement(transcript: &mut ProverTranscript<StdChallenger>, seed: u64) {
    let mut observer = transcript.observe();
    observer.write_bytes(DOMAIN_TAG);
    observer.write(&seed);
}

fn observe_verifier_statement(transcript: &mut VerifierTranscript<StdChallenger>, seed: u64) {
    let mut observer = transcript.observe();
    observer.write_bytes(DOMAIN_TAG);
    observer.write(&seed);
}

fn evaluate_bit_mle<Data>(witness: &FieldBuffer<P, Data>, point: &[B128]) -> B128
where
    Data: std::ops::Deref<Target = [P]>,
{
    let (bit_point, row_point) = point.split_at(7);
    let split = row_point.len().min(ring_switch::LOG_SPLIT_BLOCK);
    let (row_lo, row_hi) = row_point.split_at(split);
    let eq_lo = eq_ind_partial_eval::<B128>(row_lo);
    let eq_hi = eq_ind_partial_eval::<B128>(row_hi);
    let partials = ring_switch::fold_1b_rows_for_b128_split(witness, &eq_lo, &eq_hi);
    evaluate_inplace_scalars(partials.as_ref().to_vec(), bit_point)
}
