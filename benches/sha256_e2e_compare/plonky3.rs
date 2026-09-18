//! Full Plonky3 SHA-256 AIR with WHIR RS commitments and prescribed openings.

use std::{
    borrow::{Borrow, Cow},
    hint::black_box,
};

use super::common;
use super::common::plonky3::baby_bear as stack;
use super::common::whir_tuning;
use super::trace_capture::TrialScopes;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess, utils::pack_bits_le};
use p3_challenger::{CanObserve, CanSample};
use p3_field::{PrimeCharacteristicRing, PrimeField32, extension::BinomialExtensionField};
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove,
    setup, verify,
};
use p3_sha256_air::{
    BITS_PER_LIMB, INPUT_WORDS, NUM_SHA256_COLS, Sha256Air, Sha256Cols, generate_trace_rows,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use p3_whir::DomainSeparator;
pub use whir_tuning::Params;

use super::{
    CapturedSpan, CompressionCase, Corpus, SHA256_IV, SemanticSpan, TrialMetrics, humanize,
    operation,
};

type F = stack::Val;
type Challenger = stack::Challenger;
type Mmcs = stack::Mmcs;

const PUBLIC_WORDS: usize = 16 + 8;
const PUBLIC_COLUMNS: usize = 2 * PUBLIC_WORDS;

/// The complete upstream SHA AIR plus 48 public periodic columns. The main
/// trace remains exactly `NUM_SHA256_COLS` columns; this is deliberately not
/// the unrelated 16-column SHA surrogate.
#[derive(Clone)]
struct PublicSha256Air {
    periodic: Vec<Vec<F>>,
}

impl PublicSha256Air {
    fn new(corpus: &Corpus) -> Self {
        let mut periodic = vec![Vec::with_capacity(corpus.cases.len()); PUBLIC_COLUMNS];
        for case in &corpus.cases {
            for (word_index, word) in case.block.iter().chain(&case.output).copied().enumerate() {
                periodic[2 * word_index].push(F::from_u16(word as u16));
                periodic[2 * word_index + 1].push(F::from_u16((word >> 16) as u16));
            }
        }
        Self { periodic }
    }

    /// Canonical statement encoding: version, fixed IV, column count, then
    /// each column's length and canonical u32 field values, all little-endian.
    /// Hash the actual AIR values, including their order and dimensions, so
    /// verification never relies on a cached digest supplied by the prover.
    fn statement_digest(&self) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"bitz/bench/plonky3-sha256/public-statement/v1");
        for word in SHA256_IV {
            hasher.update(&word.to_le_bytes());
        }
        hasher.update(&(self.periodic.len() as u64).to_le_bytes());
        for column in &self.periodic {
            hasher.update(&(column.len() as u64).to_le_bytes());
            for value in column {
                hasher.update(&value.as_canonical_u32().to_le_bytes());
            }
        }
        hasher.finalize()
    }
}

impl BaseAir<F> for PublicSha256Air {
    fn width(&self) -> usize {
        NUM_SHA256_COLS
    }

    fn num_periodic_columns(&self) -> usize {
        PUBLIC_COLUMNS
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        Cow::Borrowed(&self.periodic)
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        BaseAir::<F>::max_constraint_degree(&Sha256Air)
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for PublicSha256Air {
    fn eval(&self, builder: &mut AB) {
        Sha256Air.eval(builder);

        let main = builder.main();
        let row: &Sha256Cols<AB::Var> = main.current_slice().borrow();
        let periodic = builder.periodic_values().to_vec();
        debug_assert_eq!(periodic.len(), PUBLIC_COLUMNS);

        // The input chaining state is the fixed FIPS SHA-256 IV.
        for (word, value) in SHA256_IV.into_iter().enumerate() {
            builder.assert_eq(row.h_in[word][0], F::from_u16(value as u16));
            builder.assert_eq(row.h_in[word][1], F::from_u16((value >> 16) as u16));
        }

        // Bind the 16 public block words and 8 public output words. Both are
        // represented as little-endian bits in the native full SHA row.
        for word in 0..16 {
            let lo: AB::Expr = pack_bits_le(row.w[word][..BITS_PER_LIMB].iter().copied());
            let hi: AB::Expr = pack_bits_le(row.w[word][BITS_PER_LIMB..].iter().copied());
            builder.assert_eq(lo, periodic[2 * word]);
            builder.assert_eq(hi, periodic[2 * word + 1]);
        }
        for word in 0..8 {
            let lo: AB::Expr = pack_bits_le(row.h_out[word][..BITS_PER_LIMB].iter().copied());
            let hi: AB::Expr = pack_bits_le(row.h_out[word][BITS_PER_LIMB..].iter().copied());
            let column = 2 * (16 + word);
            builder.assert_eq(lo, periodic[column]);
            builder.assert_eq(hi, periodic[column + 1]);
        }
    }
}

fn trace_inputs(corpus: &Corpus) -> Vec<[u32; INPUT_WORDS]> {
    corpus
        .cases
        .iter()
        .map(|case| {
            let mut input = [0u32; INPUT_WORDS];
            input[..16].copy_from_slice(&case.block);
            input[16..].copy_from_slice(&SHA256_IV);
            input
        })
        .collect()
}

fn assert_trace_outputs(trace: &RowMajorMatrix<F>, cases: &[CompressionCase]) {
    assert_eq!(trace.values.len(), cases.len() * NUM_SHA256_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to::<Sha256Cols<F>>() };
    assert!(prefix.is_empty() && suffix.is_empty());
    for (row, case) in rows.iter().zip(cases) {
        let output = std::array::from_fn(|word| {
            row.h_out[word]
                .iter()
                .rev()
                .fold(0u32, |acc, bit| (acc << 1) | bit.as_canonical_u32())
        });
        assert_eq!(
            output, case.output,
            "Plonky3 trace differs from the corpus output"
        );
    }
}

macro_rules! degree_backend {
    ($module:ident, $degree:literal) => {
        mod $module {
            use super::*;
            use p3_multi_stark::config::MultiStarkConfig;

            type EF = BinomialExtensionField<F, $degree>;
            type WhirLayout = SuffixProver<F, EF>;
            type Pcs = stack::Pcs<EF>;

            pub struct Config {
                pcs: Pcs,
                folding: usize,
            }

            impl MultiStarkConfig for Config {
                type Val = F;
                type Challenge = EF;
                type Challenger = Challenger;
                type Pcs = Pcs;

                fn pcs(&self) -> &Self::Pcs {
                    &self.pcs
                }

                fn min_num_variables(&self) -> usize {
                    self.folding
                }

                fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
                    WhirLayout::new_witness(tables, self.folding)
                }

                fn committed_table<'a>(
                    &self,
                    prover_data: &'a p3_whir::WhirProverData<F, EF, Mmcs, WhirLayout>,
                    table_index: usize,
                ) -> &'a Table<F> {
                    prover_data.table(table_index)
                }
            }

            pub struct Context {
                config: Config,
                air: PublicSha256Air,
                pk: p3_multi_stark::ProvingKey<Config>,
                vk: p3_multi_stark::VerifyingKey<Config>,
                corpus: Corpus,
                pub setup_ms: f64,
                pub security: serde_json::Value,
            }

            fn challenger(config: &Config, air: &PublicSha256Air) -> Challenger {
                let mut challenger = stack::challenger();
                let mut separator = DomainSeparator::new(Vec::new());
                config.pcs.add_domain_separator::<8>(&mut separator);
                separator.observe_domain_separator(&mut challenger);
                // Periodic columns are not public-value inputs to MultiStark.
                // Bind their digest before setup/prove/verify samples anything.
                // Byte-wise absorption is injective into BabyBear (no reduction).
                for &byte in air.statement_digest().as_bytes() {
                    challenger.observe(F::from_u8(byte));
                }
                challenger
            }

            fn decode_proof_strict(bytes: &[u8]) -> Result<MultiStarkProof<Config>, String> {
                let (proof, remaining) =
                    postcard::take_from_bytes(bytes).map_err(|error| error.to_string())?;
                if remaining.is_empty() {
                    Ok(proof)
                } else {
                    Err(format!(
                        "{} trailing bytes after Plonky3 proof",
                        remaining.len()
                    ))
                }
            }

            impl Context {
                pub fn setup(corpus: &Corpus, params: Params) -> Result<Self, String> {
                    let started_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
                    let started = tracing::info_span!("sha256_e2e_compare/plonky3:started").entered();
                    let stacked_num_variables =
                        log2_ceil_usize(corpus.cases.len() * NUM_SHA256_COLS);
                    let air = PublicSha256Air::new(corpus);
                    let (protocol, security) =
                        whir_tuning::select_protocol::<F, EF, stack::Challenger>(
                            stacked_num_variables,
                            whir_tuning::air_shape::<F, EF, _>(
                                &air,
                                log2_strict_usize(corpus.cases.len()),
                            ),
                            params,
                        )?;
                    let pcs = stack::pcs::<EF>(stacked_num_variables, protocol)
                        .map_err(|error| error.to_string())?;
                    let config = Config {
                        pcs,
                        folding: params.folding,
                    };
                    assert_eq!(air.width(), NUM_SHA256_COLS);
                    let (pk, vk) = setup(&config, &[&air], &mut challenger(&config, &air));
                    Ok(Self {
                        config,
                        air,
                        pk,
                        vk,
                        corpus: corpus.clone(),
                        setup_ms: { drop(started); bitz::observability::duration(&started_recording.intervals().expect("complete operation capture"), "sha256_e2e_compare/plonky3:started").expect("query completed operation") }.as_secs_f64() * 1e3,
                        security,
                    })
                }

                pub fn run(&self) -> (TrialMetrics, Vec<SemanticSpan>) {
                    let recording = common::perfetto::Recording::start(Vec::new())
                        .expect("start Perfetto trial");
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
                    let trace = generate_trace_rows::<F>(trace_inputs(&self.corpus), 0);
                    assert_trace_outputs(&trace, &self.corpus.cases);
                    let witness_table = Table::new(trace.transpose());
                    drop(witness_scope);
                    let proof: MultiStarkProof<Config> = prove(
                        &self.config,
                        ProverInstances::new(vec![ProverInstance::new(
                            &self.air,
                            witness_table,
                            &self.pk,
                            &[],
                        )]),
                        0,
                        &mut challenger(&self.config, &self.air),
                    );
                    drop(proving);
                    let proof_bytes = postcard::to_allocvec(&proof)
                        .expect("serialize complete Plonky3 proof")
                        .len();
                    let verification = tracing::info_span!(
                        "Verification",
                        component = "benchmark.verification",
                        scope_kind = "phase",
                        tag_verification = true
                    )
                    .entered();
                    verify(
                        &self.config,
                        VerifierInstances::new(vec![VerifierInstance::new(
                            &self.air,
                            &self.vk,
                            log2_strict_usize(self.corpus.cases.len()),
                            &[],
                        )]),
                        &proof,
                        0,
                        &mut challenger(&self.config, &self.air),
                    )
                    .expect("Plonky3 full SHA AIR proof verifies");
                    drop(verification);
                    drop(trial);
                    let raw = recording.intervals().expect("query Perfetto trial");
                    black_box(&proof);
                    let spans = semantic_spans(&raw);
                    (TrialMetrics::from_spans(&spans, proof_bytes), spans)
                }

                #[allow(dead_code)]
                fn verifies(&self, air: &PublicSha256Air, proof: &MultiStarkProof<Config>) -> bool {
                    verify(
                        &self.config,
                        VerifierInstances::new(vec![VerifierInstance::new(
                            air,
                            &self.vk,
                            log2_strict_usize(self.corpus.cases.len()),
                            &[],
                        )]),
                        proof,
                        0,
                        &mut challenger(&self.config, air),
                    )
                    .is_ok()
                }

                #[allow(dead_code)]
                pub fn tamper_self_test(&self) {
                    // Directly check transcript binding: ordinary false-input
                    // rejection alone does not detect an unbound statement.
                    let challenges = |air: &PublicSha256Air| -> [F; 8] {
                        challenger(&self.config, air).sample_array()
                    };
                    let expected_challenges = challenges(&self.air);
                    assert_eq!(expected_challenges, challenges(&self.air.clone()));
                    for column in 0..PUBLIC_COLUMNS {
                        for row in 0..self.corpus.cases.len() {
                            let mut changed = self.air.clone();
                            changed.periodic[column][row] += F::ONE;
                            assert_ne!(
                                expected_challenges,
                                challenges(&changed),
                                "every public block/output limb must affect Fiat–Shamir"
                            );
                        }
                    }
                    let mut reordered = self.air.clone();
                    for column in &mut reordered.periodic {
                        column.swap(0, 1);
                    }
                    assert_ne!(expected_challenges, challenges(&reordered));
                    let mut shorter = self.air.clone();
                    for column in &mut shorter.periodic {
                        column.pop();
                    }
                    assert_ne!(expected_challenges, challenges(&shorter));

                    let trace = generate_trace_rows::<F>(trace_inputs(&self.corpus), 0);
                    assert_trace_outputs(&trace, &self.corpus.cases);
                    let proof: MultiStarkProof<Config> = prove(
                        &self.config,
                        ProverInstances::new(vec![ProverInstance::new(
                            &self.air,
                            Table::new(trace.transpose()),
                            &self.pk,
                            &[],
                        )]),
                        0,
                        &mut challenger(&self.config, &self.air),
                    );
                    assert!(self.verifies(&self.air, &proof));

                    let mut changed_block = self.air.clone();
                    changed_block.periodic[0][0] += F::ONE;
                    assert!(
                        !self.verifies(&changed_block, &proof),
                        "changing a public Plonky3 block limb must invalidate the proof"
                    );

                    let mut changed_output = self.air.clone();
                    changed_output.periodic[2 * 16][0] += F::ONE;
                    assert!(
                        !self.verifies(&changed_output, &proof),
                        "changing a public Plonky3 output limb must invalidate the proof"
                    );
                    assert!(
                        !self.verifies(&reordered, &proof),
                        "reordering the public corpus must invalidate the proof"
                    );

                    let encoded = postcard::to_allocvec(&proof)
                        .expect("serialize Plonky3 proof for tamper test");
                    let mut corrupted = encoded.clone();
                    let middle = corrupted.len() / 2;
                    corrupted[middle] ^= 1;
                    if let Ok(decoded) = decode_proof_strict(&corrupted) {
                        assert!(
                            !self.verifies(&self.air, &decoded),
                            "a corrupted Plonky3 proof must not verify"
                        );
                    }

                    let mut trailing = encoded;
                    trailing.push(0);
                    assert!(
                        decode_proof_strict(&trailing).is_err(),
                        "trailing Plonky3 transcript data must be rejected"
                    );
                }
            }
        }
    };
}

degree_backend!(degree4, 4);
degree_backend!(degree5, 5);

pub enum Context {
    Degree4(degree4::Context),
    Degree5(degree5::Context),
}

impl Context {
    pub fn security(&self) -> serde_json::Value {
        match self {
            Self::Degree4(c) => c.security.clone(),
            Self::Degree5(c) => c.security.clone(),
        }
    }
    pub fn setup(corpus: &Corpus, params: Params) -> Result<Self, String> {
        match params.extension_degree {
            4 => degree4::Context::setup(corpus, params).map(Self::Degree4),
            5 => degree5::Context::setup(corpus, params).map(Self::Degree5),
            degree => Err(format!("unsupported WHIR extension degree {degree}")),
        }
    }

    pub fn setup_ms(&self) -> f64 {
        match self {
            Self::Degree4(context) => context.setup_ms,
            Self::Degree5(context) => context.setup_ms,
        }
    }

    pub fn run(&self) -> (TrialMetrics, Vec<SemanticSpan>) {
        match self {
            Self::Degree4(context) => context.run(),
            Self::Degree5(context) => context.run(),
        }
    }
}

pub fn tamper_self_test() {
    let corpus = Corpus::new(128, 0x5033_5348_415f_5445);
    let mut tested = 0;
    for extension_degree in [4, 5] {
        let params = Params {
            extension_degree,
            folding: 4,
            starting_log_inv_rate: 1,
            max_pow_bits: 12,
            max_round_log_inv_rate: Some(4),
        };
        let Ok(context) = Context::setup(&corpus, params) else {
            continue;
        };
        tested += 1;
        match context {
            Context::Degree4(context) => context.tamper_self_test(),
            Context::Degree5(context) => context.tamper_self_test(),
        }
    }
    assert!(
        tested > 0,
        "at least one Johnson-bound configuration must run proof rejection tests"
    );
}

#[cfg(test)]
pub(super) fn security_schedule_self_test() {
    type EF = BinomialExtensionField<F, 5>;
    for exponent in 7..=16 {
        let corpus = Corpus::new(1 << exponent, 0x5033_5348_415f_5445);
        let air = PublicSha256Air::new(&corpus);
        let shape = whir_tuning::air_shape::<F, EF, _>(&air, exponent);
        let num_variables = log2_ceil_usize(corpus.cases.len() * NUM_SHA256_COLS);
        let eligible = Params::candidates(&[5]).into_iter().find_map(|params| {
            whir_tuning::select_protocol::<F, EF, Challenger>(num_variables, shape, params).ok()
        });
        let (_, report) =
            eligible.unwrap_or_else(|| panic!("no eligible SHA schedule at 2^{exponent}"));
        assert!(report["achieved_bits"].as_f64().unwrap() >= 100.0);
    }
}

fn semantic_spans(raw: &[CapturedSpan]) -> Vec<SemanticSpan> {
    let trial = TrialScopes::from_spans(raw, "benchmark");
    let root_start = trial.verified.start_ns;
    let root_end = trial.verified.end_ns;
    let witness_to_proof_start = trial.witness_to_proof.start_ns;
    let witness_start = trial.witness.start_ns;
    let witness_end = trial.witness.end_ns;
    let verify_start = trial.verification.start_ns;
    let verify_end = trial.verification.end_ns;
    let online_start = trial.witness.end_ns;
    let online_end = trial.witness_to_proof.end_ns;
    let encode = raw
        .iter()
        .filter(|span| {
            span.name == "encode" && span.start_ns >= online_start && span.end_ns <= online_end
        })
        .min_by_key(|span| span.start_ns)
        .expect("Plonky3 initial WHIR encoding span");
    let commit_matrix = raw
        .iter()
        .filter(|span| {
            span.name == "commit_matrix"
                && span.start_ns >= encode.end_ns
                && span.end_ns <= online_end
        })
        .min_by_key(|span| span.start_ns)
        .expect("Plonky3 initial WHIR Merkle commitment span");
    let opening = raw
        .iter()
        .filter(|span| {
            matches!(span.name.as_str(), "add_virtual_eval" | "eval_at")
                && span.start_ns >= commit_matrix.end_ns
                && span.end_ns <= online_end
        })
        .min_by_key(|span| span.start_ns)
        .expect("Plonky3 WHIR prescribed-opening span");

    let mut spans = vec![
        span(
            "p3-root",
            None,
            "plonky3.verified-trial",
            "Complete verified Plonky3 trial",
            "end-to-end",
            vec!["end-to-end"],
            root_start,
            root_end,
            Some("end-to-end"),
            false,
        ),
        span(
            "p3-serialization",
            Some("p3-root"),
            "plonky3.serialization",
            "Serialize complete WHIR proof",
            "serialization",
            vec!["serialization"],
            online_end,
            verify_start,
            Some("serialization"),
            false,
        ),
        span(
            "p3-witness-to-proof",
            Some("p3-root"),
            "plonky3.witness-to-proof",
            "Plonky3 trace generation through proof readiness",
            "proving",
            vec!["proving"],
            witness_to_proof_start,
            online_end,
            None,
            false,
        ),
        span(
            "p3-witness",
            Some("p3-witness-to-proof"),
            "plonky3.witness",
            "Generate full SHA-256 AIR trace",
            "witness-generation",
            vec!["witness-generation"],
            witness_start,
            witness_end,
            None,
            true,
        ),
        span(
            "p3-online",
            Some("p3-root"),
            "plonky3.online-prover",
            "Plonky3 online prover",
            "proving",
            vec!["proving"],
            online_start,
            online_end,
            Some("proving"),
            false,
        ),
        span(
            "p3-commit",
            Some("p3-online"),
            "plonky3.commit",
            "WHIR initial trace encoding and Merkle commitment",
            "commit",
            vec!["commit", "pcs", "proving"],
            encode.start_ns,
            commit_matrix.end_ns,
            Some("commit"),
            true,
        ),
        span(
            "p3-piop",
            Some("p3-online"),
            "plonky3.piop",
            "Full SHA AIR zerocheck and sumcheck",
            "constraint-proof",
            vec!["constraint-proof", "sumcheck", "proving"],
            commit_matrix.end_ns,
            opening.start_ns,
            Some("constraint-proof"),
            true,
        ),
        span(
            "p3-iop",
            Some("p3-online"),
            "plonky3.iop",
            "WHIR multilinear opening",
            "opening-proof",
            vec!["opening-proof", "pcs", "fri", "proving"],
            opening.start_ns,
            online_end,
            Some("opening-proof"),
            true,
        ),
        span(
            "p3-verify",
            Some("p3-root"),
            "plonky3.verify",
            "Verify full SHA AIR proof",
            "verification",
            vec!["verification"],
            verify_start,
            verify_end,
            Some("verification"),
            true,
        ),
    ];
    spans.extend(
        raw.iter()
            .enumerate()
            .filter(|(_, raw)| raw.start_ns >= encode.start_ns && raw.end_ns <= online_end)
            .map(|(index, raw)| SemanticSpan {
                id: format!("p3-detail-{index}"),
                parent: Some("p3-online".to_owned()),
                operation: format!("plonky3.{}", operation(&raw.name)),
                name: humanize(&raw.name),
                short_name: "Procedure".to_owned(),
                primary_phase: "proving",
                phase_tags: vec!["proving"],
                start_ns: raw.start_ns,
                end_ns: raw.end_ns,
                scope_kind: "procedure",
                scope_tag: None,
                primary_sequence: false,
                math_latex: vec![],
            }),
    );
    spans
}

#[allow(clippy::too_many_arguments)]
fn span(
    id: &str,
    parent: Option<&str>,
    operation: &str,
    name: &str,
    primary_phase: &'static str,
    phase_tags: Vec<&'static str>,
    start_ns: u64,
    end_ns: u64,
    scope_tag: Option<&'static str>,
    primary_sequence: bool,
) -> SemanticSpan {
    SemanticSpan {
        id: id.to_owned(),
        parent: parent.map(str::to_owned),
        operation: operation.to_owned(),
        name: name.to_owned(),
        short_name: name.to_owned(),
        primary_phase,
        phase_tags,
        start_ns,
        end_ns,
        scope_kind: "phase",
        scope_tag,
        primary_sequence,
        math_latex: match primary_phase {
            "witness-generation" => {
                vec![r"\widehat H_i=\operatorname{Compress}_{\mathrm{SHA256}}(\mathrm{IV},M_i)"]
            }
            "commit" => vec![r"C_T=\operatorname{Merkle}(\operatorname{WHIR.Encode}(T))"],
            "constraint-proof" => vec![r"\sum_x\operatorname{eq}(r,x)\,C_{\mathrm{AIR}}(T(x))=0"],
            "opening-proof" => vec![r"\operatorname{WHIR.Open}(C_T,r,T(r))"],
            _ => vec![],
        },
    }
}
