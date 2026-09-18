//! Binius64 multiplication and PIOP with the BitZ opener, at the CLI-selected
//! rate and security accounting, with a target of 100 bits.
use super::trace_capture::{BiniusLigeritoPhases, TrialScopes};
use super::{CapturedSpan, Corpus, Timing, Workload, binius};
use binius_frontend::Circuit;
use bitz::binius_ligerito::{Accounting, Prepared};
use bitz::observability::Recording;
use serde_json::{Value, json};
use std::sync::Arc;

pub(super) struct Context {
    corpus: Arc<Corpus>,
    circuit: Circuit,
    wires: Vec<binius::Wires>,
    prepared: Prepared,
}

impl Context {
    pub(super) fn setup_at_rate(
        corpus: Arc<Corpus>,
        rate: usize,
        accounting: Accounting,
    ) -> Result<Self, bitz::binius_ligerito::Error> {
        let (circuit, wires) = binius::compile(&corpus);
        let prepared = Prepared::with_options(circuit.constraint_system(), rate, accounting)?;
        Ok(Self {
            corpus,
            circuit,
            wires,
            prepared,
        })
    }

    pub(super) fn config(&self) -> Value {
        let piop = match self.corpus.workload {
            Workload::U32 => {
                "Binius64 native multiplication with 32-bit inputs and low-32-bit result"
            }
            Workload::U64 => "Binius64 native 64 x 64 -> 128 integer multiplication",
            Workload::U128 => {
                "Binius64 bignum 128 x 128 -> 256 multiplication (four native imul limb products with carry chains)"
            }
        };
        let cs = self.circuit.constraint_system();
        let security = self.prepared.security();
        let witness = self.prepared.opener(0);
        json!({
            "piop": piop,
            "pcs": "BitZ opener: ring switching + Johnson-regime Ligerito with fold/query grinding and Round 0",
            "log_inv_rate": self.prepared.log_inv_rate(),
            "regime": "johnson-ood",
            "accounting": security.accounting.name(),
            "target_bits": security.target_bits,
            "whole_protocol_bits": security.algebraic_bits,
            "hash": "BLAKE3", "transcript": "BLAKE3",
            "security_terms": security.terms.iter().map(|term|
                json!({"name":term.name, "error_bound":term.error_bound})
            ).collect::<Vec<_>>(),
            "union_bound_bits": security.union_bound_bits,
            "round_by_round_bits": security.round_by_round_bits,
            "binding_term": security.binding_term().map(|t| format!("{}:{:.2}", t.name, -t.error_bound.log2())),
            "ligerito_component_bits": self.prepared.component_bits(),
            "level0_queries": witness.level0_queries(),
            "level0_query_grinding_bits": witness.level0_query_grinding_bits(),
            "level0_fold_grinding_bits": witness.level0_fold_grinding_bits(),
            "ood_grinding_bits": witness.ood_grinding_bits(),
            "oracle_logs": self.prepared.oracle_specs().iter().map(|s| s.log_msg_len).collect::<Vec<_>>(),
            "oracles": (0..self.prepared.oracle_specs().len()).map(|i| {
                let pcs = self.prepared.opener(i);
                json!({"configuration":pcs.config(), "ood_grinding_bits":pcs.ood_grinding_bits()})
            }).collect::<Vec<_>>(),
            "word_constraints": {"and": cs.n_and_constraints(), "imul": cs.n_imul_constraints(),
                "zero": cs.n_zero_constraints(), "bmul": cs.n_bmul_constraints()},
        })
    }

    pub(super) fn run(&self) -> Timing {
        let recording = Recording::start(Vec::new()).expect("start Perfetto trial");
        let proof_bytes = self.prove_and_verify();
        timing_from_spans(
            &recording.intervals().expect("query Perfetto trial"),
            proof_bytes,
        )
    }

    /// The memory-only child executes the same work without a capture buffer or
    /// query process affecting its process high-water mark.
    pub(super) fn prove_and_verify(&self) -> usize {
        let trial = tracing::info_span!(
            "Verified trial",
            component = "binius-ligerito.verified-trial",
            scope_kind = "scope",
            tag_end_to_end = true,
        )
        .entered();
        let proving = tracing::info_span!(
            "Witness to proof",
            component = "binius-ligerito.witness-to-proof",
            scope_kind = "scope",
        )
        .entered();
        let witness = tracing::info_span!(
            "Witness generation",
            component = "binius-ligerito.witness-evaluation",
            scope_kind = "phase",
            tag_witness_generation = true,
        )
        .in_scope(|| {
            binius::populate(&self.corpus, &self.circuit, &self.wires, false)
                .expect("Binius witness evaluation")
                .into_value_vec()
        });
        let proof = self
            .prepared
            .prove(&witness)
            .expect("binius64-ligerito full proof");
        let bytes = proof.to_bytes();
        drop(proving);
        let verification = tracing::info_span!(
            "Verification",
            component = "binius-ligerito.verification",
            scope_kind = "phase",
            tag_verification = true,
        )
        .entered();
        let decoded = self
            .prepared
            .proof_from_bytes(&bytes)
            .expect("binius64-ligerito proof decodes");
        self.prepared
            .verify(witness.inout(), &decoded)
            .expect("binius64-ligerito full verification");
        drop(verification);
        drop(trial);
        bytes.len()
    }
}

fn timing_from_spans(raw: &[CapturedSpan], proof_bytes: usize) -> Timing {
    let trial = TrialScopes::from_spans(raw, "binius-ligerito");
    let phases = BiniusLigeritoPhases::from_spans(raw);
    let mut t = Timing::from_trial(&trial, proof_bytes);
    t.add("commit", "commit", phases.commit.0, phases.commit.1);
    for (start, end) in phases.piop {
        t.add("piop", "constraint-proof", start, end);
    }
    for (start, end) in phases.opening {
        t.add("opening", "opening-proof", start, end);
    }
    t
}

#[cfg(test)]
mod tests {
    #[test]
    fn trial_metrics_use_operation_endpoints_not_wrapper_endpoints() {
        let raw = super::super::trace_capture::phase_tests::trial_fixture();
        let timing = super::timing_from_spans(&raw, 123);
        timing.validate();
        assert_eq!(
            timing
                .phases
                .iter()
                .take(6)
                .map(|p| p.name)
                .collect::<Vec<_>>(),
            [
                "verified_trial",
                "witness_to_proof",
                "online_prover",
                "witness",
                "verify",
                "post_proof"
            ]
        );
        let metrics = timing.metrics();
        assert_eq!(metrics.witness_ms, 8.0 / 1e6);
        assert_eq!(metrics.online_prover_ms, 130.0 / 1e6);
        assert_eq!(metrics.witness_to_proof_ms, 139.0 / 1e6);
        assert_eq!(metrics.verify_ms, 10.0 / 1e6);
        assert_eq!(metrics.verified_trial_ms, 160.0 / 1e6);
        assert_eq!(metrics.post_proof_ms, 5.0 / 1e6);
        assert_eq!(metrics.commit_ms, 10.0 / 1e6);
        assert_eq!(metrics.piop_ms, 55.0 / 1e6);
        assert_eq!(metrics.opening_ms, 50.0 / 1e6);
    }

    #[test]
    #[ignore = "requires PERFETTO_TRACE_PROCESSOR; exercises the native measurement backend"]
    fn span_metrics_cover_repeated_verified_multiplication_trials() {
        use super::*;
        use tracing_subscriber::prelude::*;

        // The BitZ opener requires packed log >= 13.
        let context = Context::setup_at_rate(
            Arc::new(Corpus::new(Workload::U32, 11, 7)),
            1,
            Accounting::UnionBound,
        )
        .unwrap();
        let witness = binius::populate(&context.corpus, &context.circuit, &context.wires, false)
            .unwrap()
            .into_value_vec();
        let expected =
            tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
                context.prepared.prove(&witness).unwrap().to_bytes()
            });
        let memory_bytes =
            tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
                context.prove_and_verify()
            });
        assert_eq!(memory_bytes, expected.len());
        tracing::subscriber::with_default(
            tracing_subscriber::registry().with(bitz::observability::layer()),
            || {
                let recording = Recording::start(Vec::new()).unwrap();
                let proof = context.prepared.prove(&witness).unwrap().to_bytes();
                assert_eq!(proof, expected, "instrumentation changed the proof bytes");
                let phases = BiniusLigeritoPhases::from_spans(&recording.intervals().unwrap());
                assert_eq!(
                    phases.opening.len(),
                    context.prepared.oracle_specs().len() + 1
                );
                // Warmup followed by five independent samples: capture state must reset.
                for _ in 0..6 {
                    let timing = context.run();
                    timing.validate();
                    assert_eq!(timing.proof_bytes, proof.len());
                    let first_opening = timing
                        .phases
                        .iter()
                        .find(|p| p.tag == "opening-proof")
                        .unwrap();
                    assert!(
                        timing.phases.iter().any(|p| {
                            p.tag == "constraint-proof" && p.start >= first_opening.end
                        }),
                        "Round 0 must remain before subsequent PIOP work"
                    );
                }
            },
        );
    }
}
