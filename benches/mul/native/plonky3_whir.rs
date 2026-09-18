//! Optional WHIR backend for the shared mod32 AIR, with per-run tuning.
use super::common::plonky3 as stacks;
use super::common::whir_tuning::{self, Params};
#[cfg(test)]
use super::mod32_air::{LIMB_BASE, TRACE_WIDTH, set_value};
use super::mod32_air::{MulAir, generate};
use super::trace_capture::TrialScopes;
use super::{Corpus, Timing, Workload, captured};
use bitz::observability::Recording;
use p3_air::BaseAir;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances,
    config::MultiStarkConfig, prove, setup, verify,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_whir::DomainSeparator;
use serde_json::{Value, json};
use std::sync::Arc;

macro_rules! backend {
    ($module:ident,$stack:ident,$degree:literal) => {
        mod $module {
            use super::stacks::$stack as stack;
            use super::*;
            type F = stack::Val;
            type EF = BinomialExtensionField<F, $degree>;
            type WhirLayout = SuffixProver<F, EF>;
            struct Config {
                pcs: stack::Pcs<EF>,
                folding: usize,
            }
            impl MultiStarkConfig for Config {
                type Val = F;
                type Challenge = EF;
                type Challenger = stack::Challenger;
                type Pcs = stack::Pcs<EF>;
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
                    data: &'a p3_whir::WhirProverData<F, EF, stack::Mmcs, WhirLayout>,
                    index: usize,
                ) -> &'a Table<F> {
                    data.table(index)
                }
            }
            pub struct Context {
                corpus: Arc<Corpus>,
                air: MulAir,
                config: Config,
                pk: p3_multi_stark::ProvingKey<Config>,
                vk: p3_multi_stark::VerifyingKey<Config>,
                pub(super) security: Value,
                pub(super) params: Params,
            }
            fn challenger(config: &Config) -> stack::Challenger {
                let mut c = stack::challenger();
                let mut domain = DomainSeparator::new(Vec::new());
                config.pcs.add_domain_separator::<8>(&mut domain);
                domain.observe_domain_separator(&mut c);
                c
            }
            impl Context {
                pub(super) fn setup(corpus: Arc<Corpus>, params: Params) -> Result<Self, String> {
                    let air = MulAir;
                    let width = <MulAir as BaseAir<F>>::width(&air);
                    let num_vars =
                        corpus.len().ilog2() as usize + width.next_power_of_two().ilog2() as usize;
                    let (protocol, security) =
                        whir_tuning::select_protocol::<F, EF, stack::Challenger>(
                            num_vars,
                            whir_tuning::air_shape::<F, EF, _>(&air, corpus.len().ilog2() as usize),
                            params,
                        )?;
                    let pcs = stack::pcs::<EF>(num_vars, protocol).map_err(|e| e.to_string())?;
                    let config = Config {
                        pcs,
                        folding: params.folding,
                    };
                    let (pk, vk) = setup(&config, &[&air], &mut challenger(&config));
                    Ok(Self {
                        corpus,
                        air,
                        config,
                        pk,
                        vk,
                        security,
                        params,
                    })
                }
                #[cfg(test)]
                pub(super) fn rejection_self_test(&self) {
                    let make_proof = |trace: RowMajorMatrix<F>| -> MultiStarkProof<Config> {
                        prove(
                            &self.config,
                            ProverInstances::new(vec![ProverInstance::new(
                                &self.air,
                                Table::new(trace.transpose()),
                                &self.pk,
                                &[],
                            )]),
                            0,
                            &mut challenger(&self.config),
                        )
                    };
                    let accepts = |proof: &MultiStarkProof<Config>| {
                        verify(
                            &self.config,
                            VerifierInstances::new(vec![VerifierInstance::new(
                                &self.air,
                                &self.vk,
                                self.corpus.len().ilog2() as usize,
                                &[],
                            )]),
                            proof,
                            0,
                            &mut challenger(&self.config),
                        )
                        .is_ok()
                    };
                    let proof = make_proof(generate(&self.corpus));
                    assert!(accepts(&proof));
                    let encoded = postcard::to_allocvec(&proof).unwrap();
                    let mut changed: MultiStarkProof<Config> =
                        postcard::from_bytes(&encoded).unwrap();
                    changed.opening.whir.initial_ood_answers[0] += EF::ONE;
                    assert!(!accepts(&changed), "PCS must reject an altered OOD answer");
                    let mut changed: MultiStarkProof<Config> =
                        postcard::from_bytes(&encoded).unwrap();
                    changed.sumcheck.claimed_sum = EF::ONE;
                    assert!(
                        !accepts(&changed),
                        "PIOP must reject a nonzero zerocheck claim"
                    );
                    // Release mode bypasses debug constraint checking: the actual
                    // verifier must reject proofs constructed for invalid witnesses.
                    #[cfg(not(debug_assertions))]
                    {
                        // Corrupt result/carry values while updating their bit columns,
                        // so arithmetic constraints must catch the change.
                        for column in [4, 5, 6, 7] {
                            let mut wrong = generate(&self.corpus);
                            set_value(&mut wrong.values[..TRACE_WIDTH], column, 1);
                            assert!(!accepts(&make_proof(wrong)), "wrong result or carry");
                        }
                        let mut wrong = generate(&self.corpus);
                        set_value(&mut wrong.values[..TRACE_WIDTH], 0, LIMB_BASE);
                        assert!(!accepts(&make_proof(wrong)), "operand limb exceeds 16 bits");
                    }
                }

                pub(super) fn run(&self) -> Timing {
                    let recording = Recording::start(Vec::new()).expect("start Perfetto trial");
                    let proof_bytes = self.prove_and_verify();
                    let raw = recording.intervals().expect("query Perfetto trial");
                    let trial = TrialScopes::from_spans(&raw, "benchmark");
                    let wend = trial.witness.end_ns;
                    let ready = trial.witness_to_proof.end_ns;
                    let encode = captured(&raw, "encode", wend, ready);
                    let commit = captured(&raw, "commit_matrix", encode.end_ns, ready);
                    let opening = raw
                        .iter()
                        .filter(|s| {
                            matches!(s.name.as_str(), "add_virtual_eval" | "eval_at")
                                && s.start_ns >= commit.end_ns
                                && s.end_ns <= ready
                        })
                        .min_by_key(|s| s.start_ns)
                        .expect("WHIR opening boundary");
                    let mut t = Timing::from_trial(&trial, proof_bytes);
                    t.add("commit", "commit", encode.start_ns, commit.end_ns);
                    t.add("piop", "constraint-proof", commit.end_ns, opening.start_ns);
                    t.add("opening", "opening-proof", opening.start_ns, ready);
                    t
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
                    let table = Table::new(trace.transpose());
                    drop(witness_scope);
                    let proof: MultiStarkProof<Config> = prove(
                        &self.config,
                        ProverInstances::new(vec![ProverInstance::new(
                            &self.air,
                            table,
                            &self.pk,
                            &[],
                        )]),
                        0,
                        &mut challenger(&self.config),
                    );
                    drop(proving);
                    let bytes = postcard::to_allocvec(&proof).expect("encode Plonky3 proof");
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
                            self.corpus.len().ilog2() as usize,
                            &[],
                        )]),
                        &proof,
                        0,
                        &mut challenger(&self.config),
                    )
                    .expect("Plonky3 full proof verifies");
                    drop(verification);
                    drop(trial);
                    let proof_bytes = bytes.len();
                    proof_bytes
                }
            }
        }
    };
}
backend!(goldilocks2, goldilocks, 2);
backend!(goldilocks5, goldilocks, 5);

pub(super) enum Context {
    U32Degree2(goldilocks2::Context),
    U32Degree5(goldilocks5::Context),
}
impl Context {
    #[cfg(test)]
    fn rejection_self_test(&self) {
        match self {
            Self::U32Degree2(x) => x.rejection_self_test(),
            Self::U32Degree5(x) => x.rejection_self_test(),
        }
    }
    pub(super) fn setup_with_params(corpus: Arc<Corpus>, params: Params) -> Result<Self, String> {
        match (corpus.workload, params.extension_degree) {
            (Workload::U32, 2) => goldilocks2::Context::setup(corpus, params).map(Self::U32Degree2),
            (Workload::U32, 5) => goldilocks5::Context::setup(corpus, params).map(Self::U32Degree5),
            _ => Err(format!(
                "unsupported WHIR {} degree {}",
                corpus.workload.slug(),
                params.extension_degree
            )),
        }
    }
    pub(super) fn run(&self) -> Timing {
        match self {
            Self::U32Degree2(x) => x.run(),
            Self::U32Degree5(x) => x.run(),
        }
    }
    pub(super) fn prove_and_verify(&self) -> usize {
        match self {
            Self::U32Degree2(x) => x.prove_and_verify(),
            Self::U32Degree5(x) => x.prove_and_verify(),
        }
    }
    pub(super) fn security(&self) -> Value {
        match self {
            Self::U32Degree2(x) => x.security.clone(),
            Self::U32Degree5(x) => x.security.clone(),
        }
    }
    pub(super) fn config(&self) -> Value {
        let (base_field, params) = match self {
            Self::U32Degree2(x) => ("Goldilocks", x.params),
            Self::U32Degree5(x) => ("Goldilocks", x.params),
        };
        json!({"piop":"Plonky3 multi-stark AIR zerocheck/sumcheck","pcs":"WHIR",
            "encoding":"Reed-Solomon", "opening_claim":"prescribed multilinear evaluation",
            "base_field":base_field,"params":params,"security":self.security(),
            "revision":super::common::local_vendor_revision("p3-whir")})
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    #[test]
    fn full_proofs_verify_mod32_boundaries_and_reject_invalid_claims() {
        let corpus = Arc::new(super::super::edge_corpus(Workload::U32));
        let mut tested = 0;
        for extension_degree in [2, 5] {
            let params = Params {
                extension_degree,
                ..Params::default()
            };
            if let Ok(context) = Context::setup_with_params(Arc::clone(&corpus), params) {
                context.rejection_self_test();
                tested += 1;
            }
        }
        assert!(
            tested > 0,
            "an eligible configuration must exercise the mod32 AIR"
        );
    }
}
