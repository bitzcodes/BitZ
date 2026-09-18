//! Independent u32 wrapping multiplication through Plonky3's univariate STARK and FRI.
#[cfg(test)]
use super::mod32_air::{LIMB_BASE, VALUE_COLUMNS, set_value};
use super::mod32_air::{MulAir, TRACE_WIDTH, generate};
use super::trace_capture::TrialScopes;
use super::{Corpus, Timing, Workload, captured};
use bitz::observability::Recording;
use p3_air::symbolic::AirLayout;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{Field, extension::BinomialExtensionField};
#[cfg(test)]
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks, default_goldilocks_poseidon2_8};
#[cfg(test)]
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{
    Proof, ProvenSecurity, StarkConfig, StarkSecurityParams, get_log_num_quotient_chunks, prove,
    verify,
};
use serde_json::{Value, json};
use std::sync::Arc;

const EXTENSION_DEGREE: usize = 5;
const TARGET_BITS: usize = 100;
/// Rate 1/2, the campaign's uniform comparison rate (2026-09-13 suite).
const LOG_BLOWUP: usize = 1;
/// Ceiling for the query solve; rate 1/2 needs a few hundred proven queries.
const MAX_QUERIES: usize = 1024;

type Val = Goldilocks;
type Challenge = BinomialExtensionField<Val, EXTENSION_DEGREE>;
type Perm = Poseidon2Goldilocks<8>;
type Packed = <Val as Field>::Packing;
type Hash = PaddingFreeSponge<Perm, 8, 4, 4>;
type Compress = TruncatedPermutation<Perm, 2, 4, 8>;
type ValMmcs = MerkleTreeMmcs<Packed, Packed, Hash, Compress, 2, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 8, 4>;
type Pcs = TwoAdicFriPcs<Val, Radix2DitParallel<Val>, ValMmcs, ChallengeMmcs>;
type Config = StarkConfig<Pcs, Challenge, Challenger>;

fn configuration_at_rate(
    trace_len: usize,
    log_blowup: usize,
) -> (Config, StarkSecurityParams, usize) {
    assert!(
        (1..=3).contains(&log_blowup),
        "log inverse rate must be 1, 2, or 3"
    );
    assert!(trace_len.is_power_of_two());
    let perm = default_goldilocks_poseidon2_8();
    let mmcs = ValMmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let layout = AirLayout::from_air::<Val>(&MulAir);
    let quotient_chunks = 1 << get_log_num_quotient_chunks::<Val, _>(&MulAir, layout, trace_len, 0);
    let assemble = |num_queries: usize| {
        let fri = FriParameters {
            log_blowup,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: ChallengeMmcs::new(mmcs.clone()),
        };
        // Floor log2(p^5) and half of log2(p^4) conservatively. The report
        // includes AIR composition, DEEP-ALI, FRI and batched-opening terms.
        let mut security = StarkSecurityParams::from_air::<Val, Challenge, _>(
            fri.security_regime(),
            &MulAir,
            layout,
            319,
            127,
            1,
        );
        security.num_batched_functions = TRACE_WIDTH + EXTENSION_DEGREE * quotient_chunks;
        (fri, security)
    };
    // Smallest query count whose proven round-by-round report clears the
    // target at this trace length — the same smallest-clearing rule the BitZ
    // opener applies to its per-round targets. Monotone in the query count.
    let clears = |num_queries: usize| {
        ProvenSecurity::compute(&assemble(num_queries).1, trace_len).security_bits() >= TARGET_BITS
    };
    assert!(
        clears(MAX_QUERIES),
        "Plonky3-FRI cannot reach {TARGET_BITS} bits at rate 1/2^{log_blowup} within {MAX_QUERIES} queries"
    );
    let (mut lo, mut hi) = (1usize, MAX_QUERIES);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if clears(mid) { hi = mid } else { lo = mid + 1 }
    }
    let num_queries = lo;
    let (fri, security) = assemble(num_queries);
    require_security(ProvenSecurity::compute(&security, trace_len));
    let pcs = Pcs::new(Radix2DitParallel::default(), mmcs, fri);
    (
        Config::new(pcs, Challenger::new(perm)),
        security,
        num_queries,
    )
}
fn require_security(security: ProvenSecurity) {
    assert!(
        security.security_bits() >= TARGET_BITS,
        "Plonky3-FRI proven round-by-round security is below {TARGET_BITS} bits: {security:?}"
    );
}

pub(super) struct Context {
    corpus: Arc<Corpus>,
    config: Config,
    security: StarkSecurityParams,
    num_queries: usize,
}
impl Context {
    #[cfg(test)]
    pub(super) fn setup(corpus: Arc<Corpus>) -> Self {
        Self::setup_at_rate(corpus, 1)
    }
    pub(super) fn setup_at_rate(corpus: Arc<Corpus>, rate: usize) -> Self {
        assert_eq!(
            corpus.workload,
            Workload::U32,
            "Plonky3-FRI supports u32 only"
        );
        let (config, security, num_queries) = configuration_at_rate(corpus.len(), rate);
        Self {
            corpus,
            config,
            security,
            num_queries,
        }
    }
    pub(super) fn config(&self) -> Value {
        let security = ProvenSecurity::compute(&self.security, self.corpus.len());
        json!({
            "piop":"Plonky3 univariate STARK AIR quotient", "pcs":"FRI",
            "base_field":"Goldilocks", "extension_degree":EXTENSION_DEGREE,
            "target_bits":TARGET_BITS, "security_scope":"proven round-by-round",
            "proven_bits":security.security_bits(), "unique_decoding_bits":security.unique_decoding_bits,
            "list_decoding_bits":security.list_decoding_bits, "hash":"Poseidon2Goldilocks-width8",
            "log_inv_rate":self.security.fri_log_blowup, "num_queries":self.security.fri_num_queries, "max_log_arity":1,
            "log_final_poly_len":0, "commit_pow_bits":0, "query_pow_bits":0,
            "trace_width":TRACE_WIDTH, "num_constraints":self.security.num_constraints,
            "max_constraint_degree":self.security.air_max_constraint_degree,
            "num_batched_functions":self.security.num_batched_functions,
            "revision":super::common::local_vendor_revision("p3-fri"),
        })
    }
    pub(super) fn run(&self) -> Timing {
        let recording = Recording::start(Vec::new()).expect("start Perfetto trial");
        let proof_bytes = self.prove_and_verify();
        let raw = recording.intervals().expect("query Perfetto trial");
        let trial = TrialScopes::from_spans(&raw, "benchmark");
        let wend = trial.witness.end_ns;
        let ready = trial.witness_to_proof.end_ns;
        let commit = captured(&raw, "commit to trace data", wend, ready);
        let piop = captured(&raw, "AIR quotient PIOP", commit.end_ns, ready);
        let opening = captured(&raw, "open", piop.end_ns, ready);
        let mut timing = Timing::from_trial(&trial, proof_bytes);
        timing.add("commit", "commit", commit.start_ns, commit.end_ns);
        timing.add("piop", "constraint-proof", piop.start_ns, piop.end_ns);
        timing.add("opening", "opening-proof", opening.start_ns, opening.end_ns);
        timing
    }

    pub(super) fn prove_and_verify(&self) -> usize {
        let trial = tracing::info_span!(
            "Verified trial",
            component = "benchmark.verified-trial",
            scope_kind = "scope",
            tag_end_to_end = true
        )
        .entered();
        let proving = tracing::info_span!(
            "Witness to proof",
            component = "benchmark.witness-to-proof",
            scope_kind = "scope"
        )
        .entered();
        let witness_scope = tracing::info_span!(
            "Witness generation",
            component = "benchmark.witness-evaluation",
            scope_kind = "phase",
            tag_witness_generation = true
        )
        .entered();
        let trace = generate(&self.corpus);
        drop(witness_scope);
        let proof: Proof<Config> = prove(&self.config, &MulAir, trace, &[]);
        drop(proving);
        let bytes = postcard::to_allocvec(&proof).expect("encode Plonky3-FRI proof");
        assert_eq!(proof.degree_bits, self.corpus.len().ilog2() as usize);
        require_security(proof.proven_security(&self.security));
        let verification = tracing::info_span!(
            "Verification",
            component = "benchmark.verification",
            scope_kind = "phase",
            tag_verification = true
        )
        .entered();
        verify(&self.config, &MulAir, &proof, &[]).expect("Plonky3-FRI full proof verifies");
        drop(verification);
        drop(trial);
        let proof_bytes = bytes.len();
        std::hint::black_box(proof);
        proof_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary_corpus() -> Corpus {
        let values = [0, 1, LIMB_BASE - 1, LIMB_BASE, u32::MAX as u64];
        let mut inputs: Vec<_> = values
            .into_iter()
            .flat_map(|a| values.map(|b| (a, b)))
            .collect();
        inputs.resize(inputs.len().next_power_of_two(), (0, 0));
        Corpus::from_inputs(Workload::U32, inputs)
    }
    fn assert_invalid(trace: RowMajorMatrix<Val>) {
        assert!(
            std::panic::catch_unwind(|| p3_air::check_constraints(&MulAir, &trace, &[])).is_err()
        );
    }
    #[test]
    fn wrapping_air_checks_boundaries_and_both_carries() {
        let corpus = boundary_corpus();
        let trace = generate(&corpus);
        p3_air::check_constraints(&MulAir, &trace, &[]);
        for (row, &(a, b)) in trace.values.chunks_exact(TRACE_WIDTH).zip(corpus.inputs()) {
            let z = row[4].as_canonical_u64() + LIMB_BASE * row[5].as_canonical_u64();
            assert_eq!(z, (a * b) & u32::MAX as u64);
        }
        for column in [4, 5, 6, 7] {
            let mut wrong = trace.clone();
            set_value(&mut wrong.values[..TRACE_WIDTH], column, 1);
            assert_invalid(wrong);
        }
        let mut wrong = trace.clone();
        set_value(&mut wrong.values[..TRACE_WIDTH], 0, LIMB_BASE);
        assert_invalid(wrong);
        let mut wrong = trace;
        wrong.values[VALUE_COLUMNS] = Val::TWO;
        assert_invalid(wrong);
    }
    #[test]
    fn rejects_goldilocks_alias_of_zero_product() {
        // The naive four-u32 equation accepts (0,0,1,2^32-1): its RHS is p.
        let modulus = (1u128 << 64) - (1u128 << 32) + 1;
        assert_eq!(1 + (1u128 << 32) * u128::from(u32::MAX), modulus);
        let corpus = Corpus::from_inputs(Workload::U32, vec![(0, 0); 16]);
        let mut trace = generate(&corpus);
        set_value(&mut trace.values[..TRACE_WIDTH], 4, 1);
        assert_invalid(trace);
    }
    #[test]
    fn actual_air_and_every_supported_shape_reach_security_target() {
        for rate in 1..=3 {
            for exponent in 4..=29 {
                let (_, params, num_queries) = configuration_at_rate(1 << exponent, rate);
                assert_eq!(params.num_constraints, 139);
                assert_eq!(params.air_max_constraint_degree, 2);
                assert_eq!(params.max_combo, 1);
                assert_eq!(params.num_batched_functions, 142);
                assert!((1..=MAX_QUERIES).contains(&num_queries));
                require_security(ProvenSecurity::compute(&params, 1 << exponent));
            }
        }
    }
    #[test]
    fn fri_proof_roundtrip_and_opening_tamper_rejection() {
        for rate in 1..=3 {
            let corpus = boundary_corpus();
            let (config, security, _) = configuration_at_rate(corpus.len(), rate);
            let mut proof = prove(&config, &MulAir, generate(&corpus), &[]);
            verify(&config, &MulAir, &proof, &[]).unwrap();
            require_security(proof.proven_security(&security));
            let bytes = postcard::to_allocvec(&proof).unwrap();
            assert!(!bytes.is_empty());
            let decoded: Proof<Config> = postcard::from_bytes(&bytes).unwrap();
            verify(&config, &MulAir, &decoded, &[]).unwrap();
            proof.opened_values.trace_local[4] += Challenge::ONE;
            assert!(verify(&config, &MulAir, &proof, &[]).is_err());
        }
    }
}
