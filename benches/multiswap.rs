//! End-to-end benchmark of Limber's MultiSwap Mod-R1CS through BitZ.
//!
//! The circuit is the wired RSA-accumulator instance ported from
//! `lucasxia01/limber-impl@benches/multiswap_modp.rs` (`k = 0`, the only
//! configuration Limber's authors mark quotable): four real square-and-
//! multiply chains with 352-bit exponents mod RSA-2048, wired Pocklington
//! hash-to-prime chains, chained Poseidon-cost rows, and full bit
//! decomposition/reconstruction.  BitZ commits the witness and quotient
//! values as raw bits (`2^25` committed bits), Spartan proves the relation
//! over a transcript-sampled 128-bit fingerprint prime, and the terminal
//! assignment claim is reduced by Step 5.0 (exact integer lift plus a
//! grinded fresh 113-bit prime) and opened against the bit commitment.
//!
//! Output follows the unified schema (`docs/bench-schema.md`): the online
//! prover includes bit-packing, commitment, both prime draws, Spartan,
//! bitification, Step 5.0, and the BitZ opening. With
//! `BITZ_MULTISWAP_TRACE_PATH`, every warmup/sample additionally records a
//! fresh witness synthesis, the online prover, and verification as exact
//! `zkperf.trace/v1` intervals. Relation preparation remains outside those
//! measured boundaries.
//!
//! Run single-threaded for numbers comparable to Limber's published table:
//!
//! ```text
//! RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" \
//!   cargo bench --bench multiswap --features unchecked
//! ```
//!
//! `BITZ_BENCH_REPS` selects the measured repetitions (default 5, plus one
//! untimed warmup; `BITZ_MULTISWAP_REPS` is a deprecated alias).
//! `BITZ_BENCH_SHAPES` selects the Limber `k` parameter (default `0`).
//! `BITZ_BENCH_LAMBDA` selects the security profile: MultiSwap's relation
//! needs a two-prime (Strategy 2) profile, so `114` (`Limber114`, the
//! pinned comparison target and the default) is the only admissible value
//! today; the single-prime profiles abort with that list.
//! Every measured proof is verified.

#![recursion_limit = "512"]

use ::bitz::ligerito_flock::IntEvalRsLigVirtProof;
use ::bitz::piop::spartan::protocol::Proof;

pub(crate) mod common;
#[cfg(feature = "bench-peak-memory")]
#[global_allocator]
static HEAP_ALLOCATOR: common::peak_memory::PeakAlloc = common::peak_memory::PeakAlloc;

use common::output::{BenchmarkOutput, FileMode, JsonlWriter};

use std::{collections::HashMap, fs::File, hint::black_box, io::BufWriter, process::Command};

use bitz::observability::Interval;
use bitz::piop::spartan::multiswap::{
    MULTISWAP_VALUE_BITS, MultiswapAssignment, MultiswapCircuit, MultiswapDims,
    PreparedMultiswapRelation, commit_multiswap_witness, prove_multiswap_mod_r1cs,
    verify_multiswap_mod_r1cs,
};
use bitz::piop::spartan::{IopSecurityProfile, PrimePolicy};
use bitz::transcript::Blake3Transcript;
use serde_json::{Value, json};

const CONSTRAINT_DIGEST_DOMAIN: &str = "bitz/multiswap/circuit-digest/v1";

#[derive(Clone, Copy)]
enum Trial {
    Warmup(usize),
    Sample(usize),
}

impl Trial {
    fn id_fragment(self) -> String {
        match self {
            Self::Warmup(index) => format!("warmup-{index}"),
            Self::Sample(index) => format!("sample-{index}"),
        }
    }

    fn json(self) -> Value {
        match self {
            Self::Warmup(index) => json!({"kind": "warmup", "warmup_index": index}),
            Self::Sample(index) => json!({"kind": "sample", "sample_index": index}),
        }
    }
}

#[derive(Clone)]
struct WitnessStats {
    witness_entries: usize,
    witness_nonzero_entries: usize,
    witness_set_bits: u64,
    witness_nonzero_16bit_chunks: usize,
    quotient_entries: usize,
    quotient_nonzero_entries: usize,
    quotient_set_bits: u64,
    quotient_nonzero_16bit_chunks: usize,
    block_assignment_entries: usize,
    assignment_nonzero_entries: usize,
    witness_max_bits: u64,
    quotient_max_bits: u64,
    assignment_set_bits: u64,
    assignment_nonzero_16bit_chunks: usize,
    assignment_digest_blake3: String,
}

impl WitnessStats {
    fn collect(circuit: &MultiswapCircuit, assignment: &MultiswapAssignment) -> Self {
        let nonzero = |values: &[field::Uint<32>]| {
            values
                .iter()
                .filter(|value| value.as_words().iter().any(|&word| word != 0))
                .count()
        };
        let max_bits = |values: &[field::Uint<32>]| {
            values
                .iter()
                .map(|v| {
                    v.as_words()
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, word)| **word != 0)
                        .map_or(0, |(i, word)| {
                            (i * 64) as u64 + 64 - word.leading_zeros() as u64
                        })
                })
                .max()
                .unwrap_or(0)
        };
        let set_bits = |values: &[field::Uint<32>]| {
            values
                .iter()
                .flat_map(|v| v.as_words())
                .map(|limb| u64::from(limb.count_ones()))
                .sum()
        };
        let nonzero_16bit_chunks = |values: &[field::Uint<32>]| {
            values
                .iter()
                .flat_map(|value| value.as_words())
                .map(|word| (0..4).filter(|i| (word >> (16 * i)) & 0xffff != 0).count())
                .sum()
        };
        let domain = assignment.layout().assignment_len();
        let values = || (0..domain).map(|index| assignment.value(index));
        let assignment_set_bits = values()
            .map(|v| {
                v.as_words()
                    .iter()
                    .map(|w| u64::from(w.count_ones()))
                    .sum::<u64>()
            })
            .sum();
        let assignment_nonzero_16bit_chunks = values()
            .map(|v| {
                v.as_words()
                    .iter()
                    .map(|w| (0..4).filter(|i| (w >> (16 * i)) & 0xffff != 0).count())
                    .sum::<usize>()
            })
            .sum();

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"bitz/multiswap/integer-assignment/v1");
        hasher.update(&(domain as u64).to_le_bytes());
        // Historical diagnostic digest uses minimal unsigned byte encodings.
        // This value-dependent formatting is outside all prover kernels.
        for value in values() {
            let mut bytes: Vec<_> = value
                .as_words()
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let len = bytes.iter().rposition(|b| *b != 0).map_or(1, |i| i + 1);
            bytes.truncate(len);
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }

        Self {
            witness_entries: circuit.witness().len(),
            witness_nonzero_entries: nonzero(circuit.witness()),
            witness_set_bits: set_bits(circuit.witness()),
            witness_nonzero_16bit_chunks: nonzero_16bit_chunks(circuit.witness()),
            quotient_entries: circuit.quotients().len(),
            quotient_nonzero_entries: nonzero(circuit.quotients()),
            quotient_set_bits: set_bits(circuit.quotients()),
            quotient_nonzero_16bit_chunks: nonzero_16bit_chunks(circuit.quotients()),
            block_assignment_entries: domain,
            assignment_nonzero_entries: values()
                .filter(|v| v.as_words().iter().any(|w| *w != 0))
                .count(),
            witness_max_bits: max_bits(circuit.witness()),
            quotient_max_bits: max_bits(circuit.quotients()),
            assignment_set_bits,
            assignment_nonzero_16bit_chunks,
            assignment_digest_blake3: hasher.finalize().to_hex().to_string(),
        }
    }

    fn json(&self) -> Value {
        json!({
            "witness_entries": self.witness_entries,
            "witness_nonzero_entries": self.witness_nonzero_entries,
            "witness_set_bits": self.witness_set_bits,
            "witness_nonzero_16bit_chunks": self.witness_nonzero_16bit_chunks,
            "quotient_entries": self.quotient_entries,
            "quotient_nonzero_entries": self.quotient_nonzero_entries,
            "quotient_set_bits": self.quotient_set_bits,
            "quotient_nonzero_16bit_chunks": self.quotient_nonzero_16bit_chunks,
            "witness_max_bits": self.witness_max_bits,
            "quotient_max_bits": self.quotient_max_bits,
            "native_committed_integer_blocks": ["W", "Q"],
            "block_assignment_entries": self.block_assignment_entries,
            "assignment_nonzero_entries": self.assignment_nonzero_entries,
            "assignment_set_bits": self.assignment_set_bits,
            "assignment_nonzero_16bit_chunks": self.assignment_nonzero_16bit_chunks,
            "active_logup_blocks": null,
            "assignment_digest_domain": "bitz/multiswap/integer-assignment/v1",
            "assignment_digest_blake3": self.assignment_digest_blake3,
        })
    }
}

struct RepTiming {
    witness_ms: f64,
    commit_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    prove_phases: Vec<(String, f64)>,
    verify_phases: Vec<(String, f64)>,
    intervals: Vec<Interval>,
    measurements_ns: MeasurementsNs,
    witness_stats: WitnessStats,
    commitment_bytes: usize,
    peak_rss_bytes: u64,
}

#[derive(clap::Parser)]
struct Env {
    #[arg(long, env = "BITZ_MULTISWAP_BATCH_COUNT", default_value_t = 1)]
    batch_count: usize,
    #[arg(long, env = "BITZ_MULTISWAP_CHECK_ONLY", default_value = "0")]
    check_only: String,
    #[arg(long, env = "BITZ_MULTISWAP_TRACE_PATH")]
    trace_path: Option<std::path::PathBuf>,
    #[arg(long, env = "BITZ_MULTISWAP_EXPECTED_CONSTRAINT_DIGEST", value_parser = normalized_digest)]
    expected_constraint_digest: Option<String>,
}

fn normalized_digest(value: &str) -> Result<String, std::convert::Infallible> {
    Ok(value
        .strip_prefix("0x")
        .unwrap_or(value)
        .to_ascii_lowercase())
}

struct TraceWriter {
    output: JsonlWriter<BufWriter<File>>,
    campaign_id: String,
    git_rev: String,
    git_dirty: Option<bool>,
    build_profile: String,
    cpu: String,
    threads: usize,
    setup_ns: u64,
    expected_constraint_digest: Option<String>,
}

impl TraceWriter {
    fn new(env: &Env, threads: usize, setup_ns: u64) -> Option<Self> {
        let path = env.trace_path.as_deref()?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            BenchmarkOutput::new(parent)
                .create_dir_all()
                .expect("create MultiSwap trace directory");
        }
        let output = BenchmarkOutput::new("")
            .jsonl(path, FileMode::CreateNew)
            .expect("create new MultiSwap trace JSONL without overwriting");
        let campaign_id = std::env::var("BITZ_MULTISWAP_CAMPAIGN_ID")
            .unwrap_or_else(|_| "multiswap-matched-v1".to_owned());
        let git_rev = std::env::var("BITZ_MULTISWAP_GIT_REV").unwrap_or_else(|_| {
            common::environment::revision()
        });
        let git_dirty = common::environment::dirty();
        let cpu = std::env::var("BITZ_MULTISWAP_CPU").unwrap_or_else(|_| {
            command_output(
                "sysctl",
                &["-n", "machdep.cpu.brand_string"],
                "Apple Silicon",
            )
        });
        let build_profile =
            std::env::var("BITZ_MULTISWAP_BUILD_PROFILE").unwrap_or_else(|_| "bench".to_owned());
        let expected_constraint_digest = env.expected_constraint_digest.clone();
        Some(Self {
            output,
            campaign_id,
            git_rev,
            git_dirty,
            build_profile,
            cpu,
            threads,
            setup_ns,
            expected_constraint_digest,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn write_run(
        &mut self,
        k: usize,
        circuit: &MultiswapCircuit,
        prepared: &PreparedMultiswapRelation,
        trial: Trial,
        timing: &RepTiming,
        piop_bytes: usize,
        opening_bytes: usize,
    ) {
        let roots = timing
            .intervals
            .iter()
            .filter(|interval| interval.parent.is_none())
            .collect::<Vec<_>>();
        assert_eq!(
            roots.len(),
            1,
            "a traced MultiSwap run has exactly one root"
        );
        assert_eq!(roots[0].label(), "multiswap-trace:verified_trial");
        for label in [
            "multiswap-trace:witness_generation",
            "multiswap-trace:end_to_end_prove",
            "multiswap-trace:commit",
            "step2:project_prove",
            "step3:piop_prove",
            "step4:bitify_prove",
            "step5_0:reduce_prove",
            "step5:open_prove",
            "multiswap-trace:verification",
        ] {
            assert_eq!(
                timing
                    .intervals
                    .iter()
                    .filter(|interval| interval.label() == label)
                    .count(),
                1,
                "complete MultiSwap trace must contain exactly one {label} interval"
            );
        }

        let constraint_digest = hex_bytes(circuit.canonical_statement_digest());
        assert_eq!(
            prepared.statement_digest(),
            &circuit.statement_digest(),
            "prepared relation must bind the measured integer statement"
        );
        if let Some(expected) = &self.expected_constraint_digest {
            assert_eq!(
                expected, &constraint_digest,
                "BITZ_MULTISWAP_EXPECTED_CONSTRAINT_DIGEST does not match the measured relation"
            );
        }

        let exact_rows = circuit
            .mods()
            .iter()
            .filter(|modulus| modulus.iter().all(|&word| word == 0))
            .count();
        let modular_rows = circuit.mods().len() - exact_rows;
        let live_exact_rows = circuit
            .mods()
            .iter()
            .take(circuit.live_rows())
            .filter(|modulus| modulus.iter().all(|&word| word == 0))
            .count();
        let live_modular_rows = circuit.live_rows() - live_exact_rows;
        let trial_fragment = trial.id_fragment();
        let batch_count = circuit.batch_count();
        let run_id = format!(
            "{}-bitz-k{k}-b{batch_count}-{}t-{trial_fragment}",
            self.campaign_id, self.threads
        );
        let series_id = format!(
            "{}-bitz-k{k}-b{batch_count}-{}-{}t-{}",
            self.campaign_id, self.git_rev, self.threads, self.build_profile
        );
        let root_span_id = span_id(roots[0].id);
        let thread_policy = if self.threads == 1 {
            "single Rayon worker"
        } else {
            "configured multi-worker Rayon pool; no affinity pinning"
        };
        let run = json!({
            "schema": "zkperf.trace/v1",
            "record": "run",
            "run_id": run_id,
            "series_id": series_id,
            "root_span_id": root_span_id,
            "benchmark": {
                "suite": "multiswap-matched",
                "name": "wired-multiswap-rsa-cost-model",
                "label": "Limber paper wired MultiSwap/RSA cost-model",
                "algorithm": "integer Mod-R1CS / Spartan / virtual BitZ",
                "implementation": "bitz-ligerito",
                "git_rev": self.git_rev,
                "git_dirty": self.git_dirty,
                "build_profile": self.build_profile,
            },
            "trial": trial.json(),
            "clock": {
                "id": format!("mono-process-{}-{run_id}", std::process::id()),
                "kind": "monotonic",
                "unit": "ns",
                "source": "Perfetto SDK",
            },
            "status": "ok",
            "trace_complete": true,
            "environment": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "cpu": self.cpu,
                "threads": self.threads,
                "thread_policy": thread_policy,
            },
            "parameters": {
                "input": {
                    "workload_id": "multiswap-rsa-wired-cost-model-v1",
                    "workload_class": "wired MultiSwap/RSA cost-model",
                    "limber_k": k,
                    "batch_count": circuit.batch_count(),
                    "public_input_count": 0,
                    "public_inputs": [],
                    "statement_contract": statement_contract(circuit),
                    "constraint_digest_domain": CONSTRAINT_DIGEST_DOMAIN,
                    "constraint_digest_blake3": constraint_digest,
                    "live_rows": circuit.live_rows(),
                    "padded_rows": circuit.num_cons(),
                    "live_columns": circuit.live_columns(),
                    "padded_columns": circuit.num_vars(),
                    "constant_column": circuit.const_col(),
                    "exact_rows": exact_rows,
                    "modular_rows": modular_rows,
                    "live_exact_rows": live_exact_rows,
                    "live_modular_rows": live_modular_rows,
                    "padding_rows": circuit.num_cons() - circuit.live_rows(),
                    "nnz_a": circuit.a_entries().len(),
                    "nnz_b": circuit.b_entries().len(),
                    "nnz_c": circuit.c_entries().len(),
                    "nnz_total": circuit.a_entries().len()
                        + circuit.b_entries().len()
                        + circuit.c_entries().len(),
                    "nnz_folded_relation": prepared.relation().nnz(),
                    "value_bits": MULTISWAP_VALUE_BITS,
                    "assignment_block_count": 4,
                    "assignment_block_capacity": prepared.layout().capacity(),
                    "committed_integer_block_count": 2,
                    "committed_bits": 1usize << (prepared.params().row_vars + prepared.params().col_vars),
                    "witness_stats": timing.witness_stats.json(),
                },
                "security": security_metadata(prepared),
                "recursion": {"max_depth": 0, "instance_count": 1},
                "repetition": {"count": 1},
            },
            "measurements_ns": timing.measurements_ns,
            "artifacts": {
                "proof_bytes": timing.commitment_bytes + piop_bytes + opening_bytes,
                "commitment_bytes": timing.commitment_bytes,
                "proof_size_kind": "serialized commitment/opening plus analytical PIOP and bridge estimate",
                "peak_rss_bytes": timing.peak_rss_bytes,
                "memory_boundary": "process high-water RSS including setup and warmups; compiler excluded",
                "piop_and_bridge_bytes": piop_bytes,
                "pcs_opening_bytes": opening_bytes,
            },
            "validation": {
                "integer_relation_satisfied": true,
                "constraint_digest_matches_prepared_relation": true,
                "constraint_digest_matches_expected": true,
                "assignment_digest_matches_source_fixture": true,
                "proof_verified": true,
            },
            "tags": {
                "campaign_id": self.campaign_id,
                "root_boundary": "witness generation plus online prover plus verification; setup excluded",
                "online_prover_boundary": "source bit packing, commitment, transcript-derived primes, Spartan, bridge, and PCS opening",
                "application_total_boundary": "witness generation plus online prover; setup and verification excluded",
                "total_pcs_definition": "interval union of commitment and PCS opening",
                "timeline": "observed half-open intervals",
                "setup_ns": self.setup_ns.to_string(),
                "expected_constraint_digest_provided": self.expected_constraint_digest.is_some().to_string(),
                "bitz_virt_id_fast": env_setting("BITZ_VIRT_ID_FAST", "default:on"),
                "bitz_rs_fast": env_setting("BITZ_RS_FAST", "default:on"),
                "bitz_flat_forest": env_setting("BITZ_FLAT_FOREST", "default:shape-dependent"),
                "f2_forest_schedule": env_setting("F2_FOREST_SCHEDULE", "default:l4"),
                "arithmetic": "delayed-barrett",
            },
        });
        self.output.write(&run).expect("write MultiSwap trace run");

        let by_order = timing
            .intervals
            .iter()
            .map(|interval| (interval.id, interval))
            .collect::<HashMap<_, _>>();
        let mut totals = HashMap::<(Option<u64>, &'static str), usize>::new();
        for interval in &timing.intervals {
            *totals
                .entry((interval.parent, interval.label()))
                .or_default() += 1;
        }
        let mut seen = HashMap::<(Option<u64>, &'static str), usize>::new();
        for interval in &timing.intervals {
            let key = (interval.parent, interval.label());
            let occurrence_index = seen.entry(key).or_default();
            let occurrence_count = totals[&key];
            let descriptor = describe_span(interval, &by_order);
            let coordinate = if occurrence_count > 1 {
                json!({
                    "occurrence_index": *occurrence_index,
                    "occurrence_count": occurrence_count,
                })
            } else {
                json!({})
            };
            *occurrence_index += 1;

            let mut attributes = json!({
                "scope_kind": descriptor.scope_kind,
                "short_name": descriptor.short_name,
                "primary_sequence": descriptor.primary_sequence,
                "constraint_digest_domain": CONSTRAINT_DIGEST_DOMAIN,
            });
            if let Some(scope_tag) = descriptor.scope_tag {
                attributes["scope_tag"] = json!(scope_tag);
            }
            if !descriptor.math_latex.is_empty() {
                attributes["math_latex"] = json!(descriptor.math_latex);
            }
            let span = json!({
                "schema": "zkperf.trace/v1",
                "record": "span",
                "run_id": run_id,
                "span_id": span_id(interval.id),
                "parent_span_id": interval.parent.map(span_id),
                "operation": descriptor.operation,
                "name": descriptor.name,
                "primary_phase": descriptor.primary_phase,
                "phase_tags": descriptor.phase_tags,
                "start_ns": interval.start_ns.to_string(),
                "end_ns": interval.end_ns.to_string(),
                "duration_ns": interval.end_ns.saturating_sub(interval.start_ns).to_string(),
                "lane": {"process": "bitz-benchmark", "thread": "control"},
                "coordinate": coordinate,
                "attributes": attributes,
            });
            self.output
                .write(&span)
                .expect("write MultiSwap trace span");
        }
        self.output.flush().expect("flush MultiSwap trace JSONL");
    }
}

struct SpanDescriptor {
    operation: String,
    name: String,
    short_name: String,
    primary_phase: &'static str,
    phase_tags: Vec<&'static str>,
    scope_kind: &'static str,
    scope_tag: Option<&'static str>,
    primary_sequence: bool,
    math_latex: Vec<&'static str>,
}

fn describe_span(interval: &Interval, by_order: &HashMap<u64, &Interval>) -> SpanDescriptor {
    let mut labels = Vec::new();
    let mut cursor = Some(interval);
    while let Some(current) = cursor {
        labels.push(current.label());
        cursor = current
            .parent
            .and_then(|order| by_order.get(&order).copied());
    }
    let under = |label: &str| labels.iter().any(|candidate| *candidate == label);
    let root = interval.parent.is_none();
    let verifying = under("multiswap-trace:verification");
    let witness = under("multiswap-trace:witness_generation");
    let committing = under("multiswap-trace:commit");
    let projection = under("step2:project_prove") || under("step2:project_verify");
    let piop = under("step3:piop_prove") || under("step3:piop_verify");
    let bridge = under("step4:bitify_prove")
        || under("step4:bitify_verify")
        || under("step5_0:reduce_prove")
        || under("step5_0:reduce_verify");
    let opening = under("step5:open_prove") || under("step5:open_verify");
    let sumcheck = interval.label().contains("sumcheck")
        || interval.label().contains("presum_run")
        || under("spartan:outer_sumcheck")
        || under("spartan:inner_sumcheck");
    let fri = opening
        && (interval.label().contains("forest")
            || interval.label().contains("fold_v")
            || interval.label().contains(":lig"));

    let primary_phase = if root {
        "end-to-end"
    } else if verifying {
        "verification"
    } else if witness {
        "witness-generation"
    } else if committing {
        "commit"
    } else if sumcheck {
        "sumcheck"
    } else if opening {
        "opening-proof"
    } else if piop {
        "constraint-proof"
    } else if projection || bridge {
        "preparation"
    } else {
        "proving"
    };
    let mut phase_tags = Vec::new();
    push_tag(&mut phase_tags, primary_phase);
    if !root && !witness {
        push_tag(
            &mut phase_tags,
            if verifying { "verification" } else { "proving" },
        );
    }
    if committing {
        push_tag(&mut phase_tags, "commit");
        push_tag(&mut phase_tags, "pcs");
    }
    if projection || bridge {
        push_tag(&mut phase_tags, "preparation");
    }
    if piop {
        push_tag(&mut phase_tags, "constraint-proof");
    }
    if opening {
        push_tag(&mut phase_tags, "opening-proof");
        push_tag(&mut phase_tags, "pcs");
    }
    if sumcheck {
        push_tag(&mut phase_tags, "sumcheck");
    }
    if fri {
        push_tag(&mut phase_tags, "fri");
    }

    let (name, short_name) = span_names(interval.label());
    let scope_kind = match interval.label() {
        "multiswap-trace:verified_trial" => "scope",
        "multiswap-trace:witness_generation"
        | "multiswap-trace:end_to_end_prove"
        | "multiswap-trace:commit"
        | "multiswap-trace:source_packing"
        | "multiswap-trace:pcs_commitment"
        | "multiswap-trace:proof"
        | "multiswap-trace:verification"
        | "step2:project_prove"
        | "step3:piop_prove"
        | "step4:bitify_prove"
        | "step5_0:reduce_prove"
        | "step5:open_prove" => "phase",
        _ if interval.label().contains("round") => "round",
        _ => "procedure",
    };
    let scope_tag = match interval.label() {
        "multiswap-trace:verified_trial" => Some("end-to-end"),
        "multiswap-trace:witness_generation" => Some("witness-generation"),
        "multiswap-trace:end_to_end_prove" => Some("proving"),
        "multiswap-trace:commit" => Some("commit"),
        "step3:piop_prove" => Some("constraint-proof"),
        "step5:open_prove" => Some("opening-proof"),
        "multiswap-trace:verification" => Some("verification"),
        _ => None,
    };
    let primary_sequence = matches!(
        interval.label(),
        "multiswap-trace:witness_generation"
            | "multiswap-trace:commit"
            | "step2:project_prove"
            | "step3:piop_prove"
            | "step4:bitify_prove"
            | "step5_0:reduce_prove"
            | "step5:open_prove"
            | "multiswap-trace:verification"
    );
    SpanDescriptor {
        operation: operation_name(interval.label()),
        name,
        short_name,
        primary_phase,
        phase_tags,
        scope_kind,
        scope_tag,
        primary_sequence,
        math_latex: span_math(interval.label()),
    }
}

fn span_names(label: &str) -> (String, String) {
    let known = match label {
        "multiswap-trace:verified_trial" => Some(("Complete verified trial", "Verified trial")),
        "multiswap-trace:witness_generation" => {
            Some(("Synthesize integer MultiSwap witness", "Witness"))
        }
        "multiswap-trace:circuit_synthesis" => {
            Some(("Synthesize wired MultiSwap/RSA circuit", "Circuit"))
        }
        "multiswap-trace:integer_satisfaction" => {
            Some(("Check exact integer relation", "Sat check"))
        }
        "multiswap-trace:assignment_materialization" => {
            Some(("Materialize witness and quotient assignment", "Assignment"))
        }
        "multiswap-trace:end_to_end_prove" => Some(("Online BitZ prover", "Prover")),
        "multiswap-trace:commit" => Some(("Total BitZ commitment", "Commit")),
        "multiswap-trace:source_packing" => {
            Some(("Pack integer assignment into source bits", "Source packing"))
        }
        "multiswap-trace:pcs_commitment" => {
            Some(("Commit packed source with Ligerito", "PCS commit"))
        }
        "multiswap-trace:proof" => Some(("Spartan, bridge, and BitZ opening proof", "Proof")),
        "multiswap-trace:verification" => Some(("Verify complete BitZ proof", "Verify")),
        "step2:project_prove" => Some((
            "Transcript prime draw and relation projection",
            "Projection",
        )),
        "step3:piop_prove" => Some(("Spartan outer and inner sumchecks", "PIOP")),
        "step4:bitify_prove" => Some(("Bitify terminal opening claim", "Bitify")),
        "step5_0:reduce_prove" => Some(("Exact lift and runtime-prime reduction", "Exact bridge")),
        "step5:open_prove" => Some(("Virtual BitZ PCS opening", "BitZ opening")),
        "mqv:pack" => Some(("Pack derived rows", "Derived packing")),
        "mc:forest" => Some(("Merged-forest GKR", "Merged GKR")),
        "mc:fold_v" => Some(("Fold integer v-message", "Integer fold")),
        "mc:presum_tbls" => Some(("Construct pre-sumcheck tables", "Pre-SC tables")),
        "mc:presum_run" => Some(("Run pre-sumcheck rounds", "Pre-sumcheck")),
        "mqv:wprep" => Some(("Prepare ring-switch weights", "Weight prep")),
        "mqv:hs" => Some(("Fold h_i plane messages", "h_i fold")),
        "mqv:aprime" => Some(("Construct a-prime basis", "a-prime")),
        "mq:rings" => Some(("Construct direct ring-switch messages", "Ring switch")),
        "mq:bcomb" => Some(("Combine ring-switch bases", "Basis combine")),
        "mq:lig" => Some(("Recursive Ligerito opening", "Ligerito")),
        _ => None,
    };
    known.map_or_else(
        || {
            let human = label
                .chars()
                .map(|character| {
                    if matches!(character, ':' | '_') {
                        ' '
                    } else {
                        character
                    }
                })
                .collect::<String>();
            (human.clone(), human)
        },
        |(name, short)| (name.to_owned(), short.to_owned()),
    )
}

fn span_math(label: &str) -> Vec<&'static str> {
    match label {
        "multiswap-trace:witness_generation" => {
            vec!["Az\\circ Bz=Cz+\\mathbf m\\circ\\mathbf q\\text{ over }\\mathbb Z"]
        }
        "multiswap-trace:source_packing" => vec!["z_j=\\sum_{b=0}^{2047}2^b z_{j,b}"],
        "multiswap-trace:pcs_commitment" => {
            vec!["C_z=\\operatorname{Com}_{\\mathbb F_{2^{128}}}(\\operatorname{bits}(z))"]
        }
        "step2:project_prove" => vec![
            "Q\\leftarrow\\operatorname{PrimeSample}(\\mathsf{tr},[2^{127},2^{128}))",
            "Az\\circ Bz=Cz+\\mathbf m\\circ\\mathbf q\\pmod Q",
        ],
        "step3:piop_prove" => vec!["\\sum_x\\operatorname{eq}(r,x)(Az(x)Bz(x)-Cz(x)-m(x)q(x))=0"],
        "step4:bitify_prove" => {
            vec!["\\widetilde z(r)=\\sum_{j,b}\\operatorname{eq}(r,(j,b))2^b z_{j,b}"]
        }
        "step5_0:reduce_prove" => vec![
            "\\mu'\\equiv\\mu\\pmod Q,\\quad q'\\leftarrow\\operatorname{PrimeSample}(\\mathsf{tr})",
        ],
        "step5:open_prove" => vec!["\\widetilde{\\operatorname{bits}(z)}(r)=v"],
        "mc:presum_tbls" | "mc:presum_run" => vec!["g_j(X)=\\sum_{b\\in\\{0,1\\}}g_{j+1}(X,b)"],
        "mqv:wprep" => vec![
            "S_{\\ell,c}=\\eta_\\ell\\sum_{r\\in\\operatorname{col}(c)}\\operatorname{eq}(p_{\\ell,\\mathrm{local}},r)",
        ],
        "mqv:hs" => {
            vec!["W_{i,c}=\\sum_\\ell\\operatorname{eq}(p_{\\ell,\\mathrm{inst}},i)S_{\\ell,c}"]
        }
        "mqv:aprime" => vec!["a'=T^*a"],
        "mq:rings" => vec!["s_v=\\sum_y\\operatorname{eq}(r_{\\mathrm{hi}},y)\\,p_{v,y}"],
        "mq:bcomb" => {
            vec!["a'=\\sum_{\\ell}\\eta_\\ell\\Phi_\\rho(\\operatorname{eq}(r_\\ell,\\cdot))"]
        }
        "mq:lig" => vec!["\\operatorname{Open}_{\\mathrm{Lig}}(C_z,r,v)"],
        _ => Vec::new(),
    }
}

fn push_tag(tags: &mut Vec<&'static str>, tag: &'static str) {
    if !tags.contains(&tag) {
        tags.push(tag);
    }
}

fn operation_name(label: &str) -> String {
    let mut output = String::with_capacity(label.len());
    let mut separator = false;
    for character in label.chars() {
        let mapped = match character {
            'A'..='Z' => character.to_ascii_lowercase(),
            'a'..='z' | '0'..='9' | '_' | '-' => character,
            _ => '.',
        };
        if mapped == '.' {
            if separator || output.is_empty() {
                continue;
            }
            separator = true;
        } else {
            separator = false;
        }
        output.push(mapped);
    }
    while output.ends_with('.') {
        output.pop();
    }
    if output.is_empty() {
        "scope".to_owned()
    } else {
        output
    }
}

fn span_id(order: u64) -> String {
    format!("span-{order}")
}

fn command_output(program: &str, args: &[&str], fallback: &str) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| fallback.to_owned())
}

fn env_setting(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn hex_bytes(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn duration_for(intervals: &[Interval], label: &str) -> u64 {
    maybe_duration_for(intervals, label).unwrap_or(0)
}

fn maybe_duration_for(intervals: &[Interval], label: &str) -> Option<u64> {
    let matching = intervals
        .iter()
        .filter(|interval| interval.label() == label)
        .collect::<Vec<_>>();
    (!matching.is_empty()).then(|| {
        matching
            .iter()
            .map(|interval| interval.end_ns.saturating_sub(interval.start_ns))
            .sum()
    })
}

type MeasurementsNs = std::collections::BTreeMap<&'static str, Nanoseconds>;

struct Nanoseconds(u64);
impl serde::Serialize for Nanoseconds {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&self.0)
    }
}

fn insert_ns(values: &mut MeasurementsNs, name: &'static str, value: u64) {
    values.insert(name, Nanoseconds(value));
}

fn insert_optional_ns(
    values: &mut MeasurementsNs,
    intervals: &[Interval],
    name: &'static str,
    label: &str,
) {
    if let Some(value) = maybe_duration_for(intervals, label) {
        insert_ns(values, name, value);
    }
}

fn measurements(intervals: &[Interval], setup_ns: u64) -> MeasurementsNs {
    if intervals.is_empty() {
        return [("setup", Nanoseconds(setup_ns))].into();
    }
    let witness = duration_for(intervals, "multiswap-trace:witness_generation");
    let commit = duration_for(intervals, "multiswap-trace:commit");
    let projection = duration_for(intervals, "step2:project_prove");
    let piop = duration_for(intervals, "step3:piop_prove");
    let bitify = duration_for(intervals, "step4:bitify_prove");
    let reduce = duration_for(intervals, "step5_0:reduce_prove");
    let opening = duration_for(intervals, "step5:open_prove");
    let prover = duration_for(intervals, "multiswap-trace:end_to_end_prove");
    let verification = duration_for(intervals, "multiswap-trace:verification");
    let verified_trial = duration_for(intervals, "multiswap-trace:verified_trial");
    let mut values = MeasurementsNs::new();
    insert_ns(&mut values, "setup", setup_ns);
    insert_ns(&mut values, "witness_generation", witness);
    insert_ns(
        &mut values,
        "source_packing",
        duration_for(intervals, "multiswap-trace:source_packing"),
    );
    insert_ns(
        &mut values,
        "pcs_commitment",
        duration_for(intervals, "multiswap-trace:pcs_commitment"),
    );
    insert_ns(&mut values, "commitment", commit);
    insert_ns(&mut values, "projection", projection);
    insert_ns(&mut values, "piop", piop);
    insert_ns(&mut values, "bitify", bitify);
    insert_ns(&mut values, "exact_lift_and_reduction", reduce);
    insert_ns(&mut values, "bridge", bitify.saturating_add(reduce));
    insert_ns(&mut values, "pcs_opening", opening);
    insert_ns(&mut values, "pcs_total", commit.saturating_add(opening));
    insert_ns(&mut values, "online_prover", prover);
    insert_ns(
        &mut values,
        "application_total",
        witness.saturating_add(prover),
    );
    insert_ns(&mut values, "verification", verification);
    insert_ns(&mut values, "verified_trial", verified_trial);

    for (name, label) in [
        ("derived_row_packing", "mqv:pack"),
        ("merged_forest_gkr", "mc:forest"),
        ("integer_folds", "mc:fold_v"),
        ("pre_sumcheck_table_construction", "mc:presum_tbls"),
        ("pre_sumcheck_protocol_rounds", "mc:presum_run"),
        ("ring_switch_weight_preparation", "mqv:wprep"),
        ("ring_switch_h_fold", "mqv:hs"),
        ("ring_switch_a_prime_construction", "mqv:aprime"),
        ("ring_switch_direct_messages", "mq:rings"),
        ("ring_switch_basis_combination", "mq:bcomb"),
        ("recursive_ligerito", "mq:lig"),
    ] {
        insert_optional_ns(&mut values, intervals, name, label);
    }
    let general_ring_switch = ["mqv:wprep", "mqv:hs", "mqv:aprime"]
        .iter()
        .filter_map(|label| maybe_duration_for(intervals, label))
        .sum::<u64>();
    let direct_ring_switch = ["mq:rings", "mq:bcomb"]
        .iter()
        .filter_map(|label| maybe_duration_for(intervals, label))
        .sum::<u64>();
    if general_ring_switch != 0 || direct_ring_switch != 0 {
        insert_ns(
            &mut values,
            "ring_switch_total",
            general_ring_switch.saturating_add(direct_ring_switch),
        );
    }
    values
}

fn proof_sizes(proof: &Proof<IntEvalRsLigVirtProof>) -> (usize, usize) {
    let opening_bytes = proof.bitz().to_bytes().len();
    let piop_bytes = proof.spartan_payload_elements() * 16 + proof.mu_prime_bytes() + 8;
    (piop_bytes, opening_bytes)
}

fn run_once(
    dims: MultiswapDims,
    batch_count: usize,
    prepared: &PreparedMultiswapRelation,
    pc: &flock_core::pcs::ligerito::ProverConfig,
    vc: &flock_core::pcs::ligerito::VerifierConfig,
    setup_ns: u64,
) -> (RepTiming, Proof<IntEvalRsLigVirtProof>) {
    let recording =
        bitz::observability::Recording::start(Vec::new()).expect("start Multiswap trial");
    let root_scope = tracing::info_span!("multiswap-trace:verified_trial").entered();

    let witness_scope = tracing::info_span!("multiswap-trace:witness_generation").entered();
    let circuit = {
        let _scope = tracing::info_span!("multiswap-trace:circuit_synthesis").entered();
        MultiswapCircuit::build_batch(dims, batch_count).expect("build circuit")
    };
    let assignment = {
        let _scope = tracing::info_span!("multiswap-trace:assignment_materialization").entered();
        MultiswapAssignment::new(&circuit).expect("build assignment")
    };
    drop(witness_scope);

    let prover_scope = tracing::info_span!("multiswap-trace:end_to_end_prove").entered();
    let commit_scope = tracing::info_span!("multiswap-trace:commit").entered();
    let rows = {
        let _scope = tracing::info_span!("multiswap-trace:source_packing").entered();
        assignment.bitz_bit_rows()
    };
    let hint = {
        let _scope = tracing::info_span!("multiswap-trace:pcs_commitment").entered();
        commit_multiswap_witness(prepared.params(), rows, pc).expect("commit")
    };
    drop(commit_scope);

    let mut prover_transcript = Blake3Transcript::new();
    let proof = {
        let _scope = tracing::info_span!("multiswap-trace:proof").entered();
        prove_multiswap_mod_r1cs(&mut prover_transcript, prepared, &assignment, &hint, pc)
            .expect("prove")
    };
    drop(prover_scope);

    {
        let _scope = tracing::info_span!("multiswap-trace:verification").entered();
        let mut verifier_transcript = Blake3Transcript::new();
        verify_multiswap_mod_r1cs(
            &mut verifier_transcript,
            prepared,
            &hint.commitment,
            &proof,
            vc,
        )
        .expect("verify");
    }
    drop(root_scope);
    common::proof_fingerprint::nonlinear(&proof, &hint.commitment.root, &prover_transcript);
    // Provenance scans are deliberately outside all reported timing
    // boundaries; they validate the trial but are not protocol work.
    assert_eq!(prepared.statement_digest(), &circuit.statement_digest());
    let witness_stats = WitnessStats::collect(&circuit, &assignment);
    let intervals = recording.intervals().expect("query Multiswap trial");
    let totals = bitz::observability::totals(&intervals);
    let witness_ms = common::span_ms(&intervals, "multiswap-trace:witness_generation");
    let commit_ms = common::span_ms(&intervals, "multiswap-trace:commit");
    let prove_ms = common::span_ms(&intervals, "multiswap-trace:end_to_end_prove");
    let verify_ms = common::span_ms(&intervals, "multiswap-trace:verification");
    let measurements_ns = measurements(&intervals, setup_ns);
    black_box(&proof);

    (
        RepTiming {
            witness_ms,
            commit_ms,
            prove_ms,
            verify_ms,
            prove_phases: totals.clone(),
            verify_phases: totals,
            intervals,
            measurements_ns,
            witness_stats,
            commitment_bytes: bincode::serialized_size(&hint.commitment).expect("commitment size")
                as usize,
            peak_rss_bytes: peak_rss_bytes(),
        },
        proof,
    )
}

/// One-time public preprocessing under the selected profile.
fn prepare<P: IopSecurityProfile>(circuit: &MultiswapCircuit) -> PreparedMultiswapRelation {
    PreparedMultiswapRelation::new_with_profile::<P>(circuit).expect("prepare relation")
}

fn main() {
    common::start_gkr_recording();
    #[cfg(feature = "bench-peak-memory")]
    let _heap_report = common::heap_run::Report::start();

    common::cli::EnvironmentCli::parse();
    let env: Env = common::cli::environment();
    let reps = common::reps(Some("BITZ_MULTISWAP_REPS"), 5);
    let k = common::shape_values(None, str::parse::<usize>).map_or(0, |shapes| {
        assert_eq!(shapes.len(), 1, "the MultiSwap bench takes one k shape");
        shapes[0]
    });
    let batch_count = env.batch_count;
    let selected = common::security_profile(PrimePolicy::TwoFullWidthFingerprint);
    let profile = selected.unwrap_or(common::SecurityProfile::Limber114);

    bitz::observability::install().expect("install Perfetto subscriber");
    let threads = common::init();

    // Bootstrap the canonical relation outside measured trials. Each trial
    // repeats synthesis and assignment materialization within its witness timer.
    let circuit = MultiswapCircuit::build_batch(MultiswapDims::multiswap(k), batch_count)
        .expect("build circuit");
    circuit
        .is_sat_integer()
        .expect("integer relation satisfied");
    let bootstrap_assignment =
        MultiswapAssignment::new(&circuit).expect("build bootstrap assignment");
    let bootstrap_stats = WitnessStats::collect(&circuit, &bootstrap_assignment);

    // One-time public preprocessing is excluded from every traced boundary.
    let setup_started_recording =
        bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
    let setup_started = tracing::info_span!("multiswap:setup_started").entered();
    let prepared = common::with_profile!(profile, prepare(&circuit));
    let (pc, vc) = prepared.ligerito_configs();
    let setup_elapsed = {
        drop(setup_started);
        bitz::observability::duration(
            &setup_started_recording
                .intervals()
                .expect("complete operation capture"),
            "multiswap:setup_started",
        )
        .expect("query completed operation")
    };
    let setup_ns = u64::try_from(setup_elapsed.as_nanos()).unwrap_or(u64::MAX);
    let setup_ms = setup_elapsed.as_secs_f64() * 1e3;

    let constraint_digest = hex_bytes(circuit.canonical_statement_digest());
    if let Some(expected) = &env.expected_constraint_digest {
        assert_eq!(
            expected, &constraint_digest,
            "canonical constraint digest mismatch"
        );
    }
    if env.check_only == "1" {
        println!(
            "MATCHED_PREFLIGHT {}",
            json!({
                "statement_contract": statement_contract(&circuit),
                "assignment_digest_blake3": bootstrap_stats.assignment_digest_blake3,
                "security": security_metadata(&prepared),
            })
        );
        return;
    }
    let mut trace_writer = TraceWriter::new(&env, threads, setup_ns);

    let p = *prepared.params();
    let layout = *prepared.layout();
    println!(
        "MultiSwap through BitZ (Limber k={k}, batch={batch_count} circuit): {} live rows, {} live columns, \
         nnz {} (mods folded into C), capacity 2^{}",
        circuit.live_rows(),
        circuit.live_columns(),
        prepared.relation().nnz(),
        layout.gate_vars(),
    );
    println!("  canonical constraint digest: {constraint_digest}");
    println!(
        "  committed bits: 2^{} ({} B) = 2 blocks x 2^{} gates x {} bits | \
         bitz t={} s={} W={} | fingerprint Q in [2^127, 2^128), step5.0 q' in [2^112, 2^113) | \
         threads={threads} reps={reps}",
        p.row_vars + p.col_vars,
        (1usize << (p.row_vars + p.col_vars)) / 8,
        layout.gate_vars(),
        MULTISWAP_VALUE_BITS,
        p.row_vars,
        p.col_vars,
        p.word_bits,
    );
    println!(
        "  security profile: {}",
        common::profile_banner(selected, common::SecurityProfile::Limber114)
    );

    let mut prover = common::StepSamples::default();
    let mut verifier = common::StepSamples::default();
    let mut witness_samples = Vec::with_capacity(reps);
    let mut last_proof = None;
    let mut commitment_bytes = 0;

    for rep in 0..reps + 1 {
        let trial = if rep == 0 {
            Trial::Warmup(0)
        } else {
            Trial::Sample(rep - 1)
        };
        let (timing, proof) = run_once(
            MultiswapDims::multiswap(k),
            batch_count,
            &prepared,
            &pc,
            &vc,
            setup_ns,
        );
        assert_eq!(
            timing.witness_stats.assignment_digest_blake3, bootstrap_stats.assignment_digest_blake3,
            "per-trial assignment must match the canonical source fixture"
        );
        let (piop_bytes, opening_bytes) = proof_sizes(&proof);
        if let Some(writer) = trace_writer.as_mut() {
            writer.write_run(
                k,
                &circuit,
                &prepared,
                trial,
                &timing,
                piop_bytes,
                opening_bytes,
            );
        }
        if rep != 0 {
            witness_samples.push(timing.witness_ms);
            common::print_regression_phases(&timing.prove_phases);
            prover.record_prove(timing.prove_ms, timing.commit_ms, &timing.prove_phases);
            verifier.record_verify(timing.verify_ms, &timing.verify_phases);
            last_proof = Some(proof);
            commitment_bytes = timing.commitment_bytes;
        }
    }

    let proof = last_proof.expect("at least one measured repetition");
    let (piop_bytes, bitz_bytes) = proof_sizes(&proof);
    println!(
        "  campaign witness generation (median of {reps}): {:.2} ms",
        common::median(&witness_samples)
    );

    let report = common::BenchReport {
        bench: "multiswap",
        shape: format!("k{k}"),
        extra: vec![
            ("profile".into(), prepared.security().profile_name.into()),
            ("batch_count".into(), batch_count.to_string()),
            ("rows".into(), circuit.live_rows().to_string()),
            (
                "committed_bits".into(),
                (1usize << (p.row_vars + p.col_vars)).to_string(),
            ),
            ("constraint_digest".into(), constraint_digest),
        ],
        lambda: Some(prepared.security().lambda),
        lambda_achieved: Some(prepared.security().accounting.achieved_bits()),
        lambda_bind: Some(prepared.security().accounting.binding_term().name.into()),
        threads,
        reps,
        seed: None,
        witness_ms: common::median(&witness_samples),
        setup_ms,
        prover: prover.medians(),
        verifier: verifier.medians(),
        proof: common::ProofBytes {
            piop: piop_bytes,
            open: bitz_bytes,
        },
    };
    report.print_human_with_commitment(commitment_bytes);
    common::print_gkr_schedules();
}

fn statement_contract(circuit: &MultiswapCircuit) -> Value {
    json!({
        "domain": "bitz-limber/multiswap-statement/v2",
        "digest_blake3": hex_bytes(circuit.canonical_comparison_statement_digest()),
        "batch_count": circuit.batch_count(), "public_input_count": 0, "public_inputs": [],
        "value_bits": MULTISWAP_VALUE_BITS, "integer_domain": "unsigned",
        "public_roles": ["matrices", "moduli"], "private_roles": ["witness", "quotients"],
        "constant": 1, "padding": "zero-witness,zero-quotients,modulus-two",
        "live_rows": circuit.live_rows(), "live_columns": circuit.live_columns(),
        "padded_rows": circuit.num_cons(), "padded_columns": circuit.num_vars(),
    })
}

fn security_metadata(prepared: &PreparedMultiswapRelation) -> Value {
    let sec = prepared.security();
    let (pc, _) = prepared.ligerito_configs();
    json!({
        "model": "per-check-round-minimum/v1", "profile": sec.profile_name,
        "target_bits": sec.lambda, "achieved_bits": sec.accounting.achieved_bits(),
        "binding_term": sec.accounting.binding_term().name,
        "terms": sec.accounting.terms.iter().map(|t| json!({
            "name": t.name, "bits": t.bits, "grinding_bits": t.grinding_bits, "floor": t.floor,
        })).collect::<Vec<_>>(),
        "transcript_hash": "BLAKE3", "fingerprint_prime_bits": 128,
        "fingerprint_min": sec.projection_min.to_string(), "fingerprint_max": sec.projection_max.to_string(),
        "reduction_prime_bits": 113,
        "reduction_grinding_bits": prepared.profile().reduction_grinding_bits(),
        "reduction_min": prepared.profile().reduction_interval().0.to_string(),
        "reduction_max": prepared.profile().reduction_interval().1.to_string(),
        "ligerito_target_bits": sec.ligerito_target_bits,
        "ligerito_config_digest": hex_bytes(*prepared.opening_config_digest()),
        "ligerito_queries": pc.queries, "ligerito_query_grinding": pc.grinding_bits,
        "ligerito_fold_grinding": pc.fold_grinding_bits,
        "ligerito_log_inv_rates": pc.log_inv_rates,
        "verifier_randomness": "native Fiat-Shamir transcript", "commitment_field": "GF(2^128)",
    })
}

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // getrusage writes the entire output on success; no witness data is exposed.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(result, 0, "getrusage failed");
    let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        rss
    } else {
        rss * 1024
    }
}

#[cfg(test)]
mod reporting_tests {
    use super::{MeasurementsNs, insert_ns, insert_optional_ns, measurements};

    #[test]
    fn sparse_nanoseconds_are_exact_decimal_strings() {
        assert_eq!(
            serde_json::to_string(&measurements(&[], u64::MAX)).unwrap(),
            r#"{"setup":"18446744073709551615"}"#
        );
        let mut values = MeasurementsNs::new();
        insert_ns(&mut values, "setup", 0);
        insert_optional_ns(&mut values, &[], "missing", "no interval");
        assert_eq!(serde_json::to_string(&values).unwrap(), r#"{"setup":"0"}"#);
    }
}
