//! Native multilinear AIR + WHIR comparison policy. The PCS commits RS
//! codewords and opens the AIR's prescribed multilinear claims directly.
//! This module is deliberately separate from the historical PCS-only profiles.

use p3_air::{Air, AirLayout, SymbolicAirBuilder, get_all_symbolic_constraints};
use p3_field::{ExtensionField, Field, PrimeField64};
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

pub const TARGET_BITS: usize = 100;
// Search enough additional round strength to cover composition, while always
// deciding eligibility from the evaluated protocol rather than this ceiling.
const MAX_ROUND_TARGET_BITS: usize = TARGET_BITS + 16;

fn cardinality_bits_floor<F: PrimeField64>(degree: usize) -> usize {
    // Goldilocks is just below 2^64: an f64 logarithm rounds up to 64.
    // BigUint keeps this bound below log2(p^degree) even in that case.
    (num_bigint::BigUint::from(F::ORDER_U64)
        .pow(degree as u32)
        .bits()
        - 1) as usize
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub extension_degree: usize,
    pub folding: usize,
    pub starting_log_inv_rate: usize,
    pub max_pow_bits: usize,
    #[serde(default)]
    pub max_round_log_inv_rate: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            extension_degree: 5,
            folding: 4,
            starting_log_inv_rate: 1,
            max_pow_bits: 12,
            max_round_log_inv_rate: None,
        }
    }
}

impl Params {
    pub fn label(self) -> String {
        format!(
            "jb100-d{}-fold{}-rate{}-pow{}-cap{}",
            self.extension_degree,
            self.folding,
            self.starting_log_inv_rate,
            self.max_pow_bits,
            self.max_round_log_inv_rate
                .map_or_else(|| "auto".into(), |cap| cap.to_string())
        )
    }

    pub fn protocol(self) -> ProtocolParameters {
        ProtocolParameters {
            security_level: TARGET_BITS,
            pow_bits: self.max_pow_bits,
            round_log_inv_rates: Vec::new(),
            folding_factor: FoldingFactor::Constant(self.folding),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: self.starting_log_inv_rate,
        }
    }

    pub fn candidates(degrees: &[usize]) -> Vec<Self> {
        let mut result = Vec::new();
        for &extension_degree in degrees {
            for folding in [2, 4] {
                for starting_log_inv_rate in 1..=3 {
                    for max_pow_bits in [8, 12] {
                        for max_round_log_inv_rate in [None, Some(4)] {
                            result.push(Self {
                                extension_degree,
                                folding,
                                starting_log_inv_rate,
                                max_pow_bits,
                                max_round_log_inv_rate,
                            });
                        }
                    }
                }
            }
        }
        result
    }
}

/// Explicit replay only. Ordinary invocations never read a previous winner.
pub fn replay() -> Result<Option<Params>, String> {
    let Some(path) = std::env::var_os("BITZ_WHIR_CONFIG") else {
        return Ok(None);
    };
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let value: Value = serde_json::from_reader(file).map_err(|e| e.to_string())?;
    serde_json::from_value(value.get("selected").unwrap_or(&value).clone())
        .map(Some)
        .map_err(|e| e.to_string())
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct AirShape {
    pub log_height: usize,
    pub width: usize,
    pub constraints: usize,
    pub constraint_degree: usize,
}

pub fn air_shape<F, EF, A>(air: &A, log_height: usize) -> AirShape
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base, ext) = get_all_symbolic_constraints::<F, EF, A>(air, AirLayout::from_air::<F>(air));
    assert!(
        air.preprocessed_width() == 0 && air.main_next_row_columns().is_empty(),
        "native WHIR security accounting requires no preprocessing or next-row openings"
    );
    let degree = base
        .iter()
        .map(|c| c.poly_degree(2, &[]))
        .chain(ext.iter().map(|c| c.poly_degree(2, &[])))
        .max()
        .unwrap_or(0);
    AirShape {
        log_height,
        width: air.width(),
        constraints: base.len() + ext.len(),
        constraint_degree: degree,
    }
}

#[derive(Debug, Serialize)]
struct Term {
    label: String,
    bits: f64,
}

fn union_bits(terms: &[Term]) -> f64 {
    let weakest = terms.iter().map(|t| t.bits).fold(f64::INFINITY, f64::min);
    weakest
        - terms
            .iter()
            .map(|t| 2_f64.powf(weakest - t.bits))
            .sum::<f64>()
            .log2()
}

/// A conservative union of the pinned WHIR error primitives and the native
/// AIR zerocheck's Schwartz-Zippel bounds. Grinding is credited only at the
/// corresponding WHIR site, using the dependency's work-factor model.
/// There is one AIR, no lookups, no preprocessing commitment, and no next-row
/// openings in the supported native workloads. Do not reuse this for uni-STARK.
pub fn security_report<F, EF, Ch>(
    config: &WhirConfig<EF, F, Ch>,
    shape: AirShape,
) -> Result<Value, String>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
{
    if config.params.soundness_type != SecurityAssumption::JohnsonBound {
        return Err("native WHIR requires JohnsonBound".into());
    }
    // Rounding down cannot overstate the size of the challenge field.
    let field_bits = cardinality_bits_floor::<F>(EF::DIMENSION);
    let jb = SecurityAssumption::JohnsonBound;
    let mut terms = Vec::new();
    let mut add = |label: String, bits: f64, count: usize| {
        if count != 0 {
            terms.push(Term {
                label,
                bits: bits - (count as f64).log2(),
            });
        }
    };
    let field = field_bits as f64;
    let list_bits = jb.list_size_bits(config.num_variables, config.params.starting_log_inv_rate);
    // The unground outer zerocheck first batches constraints, tests its MLE
    // at tau, then runs degree-(AIR degree + 1) sumcheck for log_height rounds.
    // Account conservatively for a pair of candidate codewords in the initial
    // Johnson list; there is a single stacked trace commitment in these AIRs.
    let air_bits = field - 2.0 * list_bits;
    add(
        "air.constraint_batching".into(),
        air_bits,
        shape.constraints.saturating_sub(1),
    );
    add("air.zerocheck_tau".into(), air_bits, shape.log_height);
    add(
        "air.sumcheck".into(),
        air_bits,
        shape.log_height * (shape.constraint_degree + 1),
    );
    let claims = shape.width + config.commitment_ood_samples;
    add(
        "whir.opening_batching".into(),
        field - list_bits,
        claims.saturating_sub(1),
    );
    add(
        "whir.initial_ood".into(),
        jb.ood_error(
            config.num_variables,
            config.params.starting_log_inv_rate,
            field_bits,
            config.commitment_ood_samples,
        ),
        1,
    );
    // The pinned proximity primitive retains the dominant theorem term.
    // Doubling it safely covers the two omitted positive terms for m=10,
    // redundant rates, and these power-of-two domains.
    add(
        "whir.initial_folding.proximity".into(),
        jb.prox_gaps_error(
            config.num_variables,
            config.params.starting_log_inv_rate,
            field_bits,
            2,
        ) - 1.0
            + config.starting_folding_pow_bits as f64,
        config.folding_schedule[0],
    );
    add(
        "whir.initial_folding.sumcheck".into(),
        jb.fold_sumcheck_error(
            field_bits,
            config.num_variables,
            config.params.starting_log_inv_rate,
        ) + config.starting_folding_pow_bits as f64,
        config.folding_schedule[0],
    );
    let mut old_rate = config.params.starting_log_inv_rate;
    for (i, round) in config.round_parameters.iter().enumerate() {
        add(
            format!("whir.round_{i}.queries"),
            jb.queries_error(old_rate, round.num_queries) + round.pow_bits as f64,
            1,
        );
        add(
            format!("whir.round_{i}.combination"),
            jb.queries_combination_error(
                field_bits,
                round.num_variables,
                round.log_inv_rate,
                round.ood_samples,
                round.num_queries,
            ) + round.pow_bits as f64,
            1,
        );
        add(
            format!("whir.round_{i}.ood"),
            jb.ood_error(
                round.num_variables,
                round.log_inv_rate,
                field_bits,
                round.ood_samples,
            ),
            1,
        );
        add(
            format!("whir.round_{i}.folding.proximity"),
            jb.prox_gaps_error(round.num_variables, round.log_inv_rate, field_bits, 2) - 1.0
                + round.folding_pow_bits as f64,
            config.folding_schedule[i + 1],
        );
        add(
            format!("whir.round_{i}.folding.sumcheck"),
            jb.fold_sumcheck_error(field_bits, round.num_variables, round.log_inv_rate)
                + round.folding_pow_bits as f64,
            config.folding_schedule[i + 1],
        );
        old_rate = round.log_inv_rate;
    }
    add(
        "whir.final_queries".into(),
        jb.queries_error(old_rate, config.final_queries) + config.final_pow_bits as f64,
        1,
    );
    add(
        "whir.final_sumcheck".into(),
        field - 1.0 + config.final_folding_pow_bits as f64,
        config.final_sumcheck_rounds,
    );
    // Poseidon2 stacks have 8 BabyBear or 4 Goldilocks digest/capacity elements.
    // Cap at the permutation's 128-bit target as well as the birthday bound.
    let digest_elements = if F::ORDER_U64 <= u32::MAX as u64 {
        8
    } else {
        4
    };
    let hash_bits = (0.5 * cardinality_bits_floor::<F>(digest_elements) as f64).min(128.0);
    add("commitment_and_transcript_hash".into(), hash_bits, 1);
    if terms.iter().any(|t| !t.bits.is_finite()) {
        return Err("non-finite security term".into());
    }
    let achieved_bits = union_bits(&terms);
    if achieved_bits < TARGET_BITS as f64 {
        return Err(format!(
            "Johnson-bound accounting gives {achieved_bits:.3} bits; need {TARGET_BITS}"
        ));
    }
    Ok(json!({
        "model": "native-air-whir-johnson-union/v1", "target_bits": TARGET_BITS,
        "achieved_bits": achieved_bits, "round_target_bits": config.params.security_level,
        "assumption": "JohnsonBound", "grinding_model": "pinned WHIR per-site work-factor boost",
        "challenge_field_bits_floor": field_bits, "air": shape, "terms": terms,
        "piop": "multilinear AIR zerocheck/sumcheck", "encoding": "Reed-Solomon",
        "opening_claim": "prescribed multilinear evaluation",
        "schedule": {
            "num_variables": config.num_variables, "folding": config.folding_schedule,
            "starting_log_inv_rate": config.params.starting_log_inv_rate,
            "initial_ood_samples": config.commitment_ood_samples,
            "initial_folding_pow_bits": config.starting_folding_pow_bits,
            "rounds": config.round_parameters.iter().map(|r| json!({
                "num_variables": r.num_variables, "log_inv_rate": r.log_inv_rate,
                "queries": r.num_queries, "ood_samples": r.ood_samples,
                "query_pow_bits": r.pow_bits, "folding_pow_bits": r.folding_pow_bits,
            })).collect::<Vec<_>>(),
            "final_queries": config.final_queries, "final_pow_bits": config.final_pow_bits,
            "final_sumcheck_rounds": config.final_sumcheck_rounds,
            "final_folding_pow_bits": config.final_folding_pow_bits,
        }
    }))
}

/// Derive the least integer round target passing the complete accounting for
/// this AIR and commitment shape. This needs no witness, FFT, or proof work.
/// Replays deterministically derive the same schedule from the saved params.
pub fn select_protocol<F, EF, Ch>(
    num_variables: usize,
    shape: AirShape,
    params: Params,
) -> Result<(ProtocolParameters, Value), String>
where
    F: PrimeField64 + p3_field::TwoAdicField,
    EF: ExtensionField<F> + p3_field::TwoAdicField,
    Ch: p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
{
    let mut last_reason = String::new();
    for target in TARGET_BITS..=MAX_ROUND_TARGET_BITS {
        let mut protocol = params.protocol();
        protocol.security_level = target;
        if let Some(cap) = params.max_round_log_inv_rate {
            let folding = protocol
                .folding_factor
                .compute_folding_schedule(num_variables)
                .map_err(|error| error.to_string())?;
            let mut rate = params.starting_log_inv_rate;
            protocol.round_log_inv_rates = folding
                .iter()
                .take(folding.len() - 1)
                .map(|fold| {
                    rate = (rate + fold - 1).min(cap);
                    rate
                })
                .collect();
        }
        match WhirConfig::<EF, F, Ch>::new(num_variables, protocol.clone()) {
            Ok(config) => match security_report(&config, shape) {
                Ok(report) => return Ok((protocol, report)),
                Err(reason) => last_reason = reason,
            },
            Err(error) => last_reason = error.to_string(),
        }
    }
    Err(format!(
        "no round target in {TARGET_BITS}..={MAX_ROUND_TARGET_BITS} passes: {last_reason}"
    ))
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TuningReport {
    pub selected: Params,
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective: Option<&'static str>,
    pub security: Value,
    pub tuning_ms: f64,
    pub candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workload: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exponent: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus_digest: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Candidate {
    Eligible {
        params: Params,
        preflight_ms: f64,
        security: Value,
        #[serde(flatten)]
        finalist: Option<Finalist>,
    },
    Ineligible {
        params: Params,
        reason: String,
    },
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Finalist {
    tuning_samples_ms: Vec<f64>,
    tuning_median_ms: f64,
}

/// Run in the calling benchmark's fixed thread pool. `run` must generate a
/// fresh witness, verify the proof, and return only witness-to-proof latency.
/// Expected configuration failures are returned by setup; proof failures panic
/// and must never be reclassified as ineligible or out-of-memory candidates.
pub fn tune<C>(
    degrees: &[usize],
    explicit: Option<Params>,
    setup: impl FnMut(Params) -> Result<C, String>,
    run: impl FnMut(&C) -> f64,
    report: impl Fn(&C) -> Value,
) -> Result<(Params, TuningReport), String> {
    let reps = if explicit.is_some() {
        0
    } else {
        super::cli::env::<std::num::NonZeroUsize>("BITZ_WHIR_TUNING_REPS").map_or(5, usize::from)
    };
    tune_with_reps(degrees, explicit, reps, setup, run, report)
}

/// Explicit benchmark options; no ambient experiment environment is consulted.
pub fn tune_with_reps<C>(
    degrees: &[usize],
    explicit: Option<Params>,
    reps: usize,
    mut setup: impl FnMut(Params) -> Result<C, String>,
    mut run: impl FnMut(&C) -> f64,
    report: impl Fn(&C) -> Value,
) -> Result<(Params, TuningReport), String> {
    let recording = bitz::observability::Recording::start(Vec::new()).map_err(|e| e.to_string())?;
    let campaign = tracing::info_span!("whir:tuning").entered();
    if let Some(params) = explicit {
        let context = setup(params)?;
        return Ok((
            params,
            TuningReport {
                selected: params,
                mode: "explicit",
                objective: None,
                security: report(&context),
                tuning_ms: {
                    drop(campaign);
                    bitz::observability::duration(
                        &recording.intervals().map_err(|e| e.to_string())?,
                        "whir:tuning",
                    )
                    .map_err(|e| e.to_string())?
                    .as_secs_f64()
                        * 1e3
                },
                candidates: vec![],
                workload: None,
                exponent: None,
                corpus_digest: None,
            },
        ));
    }
    let mut candidates = Vec::new();
    let mut eligible = Vec::new();
    for params in Params::candidates(degrees) {
        match setup(params) {
            Ok(context) => {
                let ms = run(&context);
                assert!(ms.is_finite() && ms > 0.0, "invalid tuning measurement");
                eprintln!("  WHIR {}: {ms:.3} ms witness-to-proof", params.label());
                let index = candidates.len();
                candidates.push(Candidate::Eligible {
                    params,
                    preflight_ms: ms,
                    security: report(&context),
                    finalist: None,
                });
                eligible.push((ms, params, index));
            }
            Err(reason) => candidates.push(Candidate::Ineligible { params, reason }),
        }
    }
    eligible.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    eligible.truncate(4);
    let mut finalists = Vec::new();
    for (_, params, index) in eligible {
        let context = setup(params)?;
        std::hint::black_box(run(&context));
        let samples: Vec<_> = (0..reps).map(|_| run(&context)).collect();
        assert!(samples.iter().all(|ms| ms.is_finite() && *ms > 0.0));
        let median = super::median(&samples);
        let Candidate::Eligible { finalist, .. } = &mut candidates[index] else {
            unreachable!("only eligible candidates advance to finalists");
        };
        *finalist = Some(Finalist {
            tuning_samples_ms: samples,
            tuning_median_ms: median,
        });
        finalists.push((median, params, report(&context)));
    }
    finalists.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let (_, params, security) = finalists
        .first()
        .ok_or_else(|| format!("no eligible WHIR configuration: {}", json!(candidates)))?;
    Ok((
        *params,
        TuningReport {
            selected: *params,
            mode: "tuned-per-invocation-and-size",
            objective: Some("witness_to_proof_ms"),
            security: security.clone(),
            candidates,
            tuning_ms: {
                drop(campaign);
                bitz::observability::duration(
                    &recording.intervals().map_err(|e| e.to_string())?,
                    "whir:tuning",
                )
                .map_err(|e| e.to_string())?
                .as_secs_f64()
                    * 1e3
            },
            workload: None,
            exponent: None,
            corpus_digest: None,
        },
    ))
}

pub fn save(path: &Path, record: &impl serde::Serialize) -> Result<(), Box<dyn std::error::Error>> {
    super::output::BenchmarkOutput::new("").write_json(
        path,
        record,
        super::output::FileMode::CreateNew,
        super::output::JsonStyle::Pretty,
    )
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn subprocess_security_report_preserves_float_identity() {
        // These evaluated bounds changed by one ULP when a memory result was
        // parsed and reserialized without serde_json's float_roundtrip feature.
        let report = json!({"achieved_bits": 100.37906411365205,
            "terms": [{"bits": 101.22551584756663}, {"bits": 101.55092293332149}]});
        let encoded = serde_json::to_vec(&report).unwrap();
        let decoded: Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(report, decoded);
        assert_eq!(encoded, serde_json::to_vec(&decoded).unwrap());
    }

    #[test]
    fn union_accounts_for_all_errors() {
        let terms = vec![
            Term {
                label: "a".into(),
                bits: 100.0,
            },
            Term {
                label: "b".into(),
                bits: 100.0,
            },
        ];
        assert_eq!(union_bits(&terms), 99.0);
    }

    #[test]
    fn field_bound_never_rounds_goldilocks_up_to_a_power_of_two() {
        assert_eq!(cardinality_bits_floor::<p3_goldilocks::Goldilocks>(2), 127);
        assert_eq!(cardinality_bits_floor::<p3_baby_bear::BabyBear>(5), 154);
    }

    #[test]
    fn tuning_excludes_ineligible_candidates_and_repeats_on_every_call() {
        let _trace = super::super::test_tracing();
        use std::cell::Cell;
        let executions = Cell::new(0);
        for _ in 0..2 {
            let before = executions.get();
            let (params, record) = tune(
                &[5],
                None,
                |p| {
                    if p.starting_log_inv_rate == 1 && p.max_pow_bits == 8 {
                        Ok(p)
                    } else {
                        Err("ineligible fixture".into())
                    }
                },
                |p| {
                    executions.set(executions.get() + 1);
                    10.0 / p.folding as f64
                },
                |_| json!({"fixture":true}),
            )
            .unwrap();
            assert_eq!(params.folding, 4);
            assert!(executions.get() > before);
            assert_eq!(record.mode, "tuned-per-invocation-and-size");
            assert_eq!(
                record
                    .candidates
                    .iter()
                    .filter(|v| matches!(v, Candidate::Eligible { .. }))
                    .count(),
                4
            );
        }
    }

    #[test]
    fn target_label_does_not_pass_an_underconfigured_protocol() {
        type F = p3_baby_bear::BabyBear;
        type EF = p3_field::extension::BinomialExtensionField<F, 5>;
        type Ch = super::super::plonky3::baby_bear::Challenger;
        let shape = AirShape {
            log_height: 10,
            width: 3,
            constraints: 1,
            constraint_degree: 2,
        };
        let (protocol, report) =
            select_protocol::<F, EF, Ch>(16, shape, Params::default()).unwrap();
        assert!(report["achieved_bits"].as_f64().unwrap() >= 100.0);
        assert!(protocol.security_level < MAX_ROUND_TARGET_BITS);
        let mut config = WhirConfig::<EF, F, Ch>::new(16, protocol).unwrap();
        assert!(security_report(&config, shape).is_ok());
        config.final_queries = 0;
        config.final_pow_bits = 0;
        assert!(security_report(&config, shape).is_err());
    }
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    #[test]
    fn candidate_fields_and_explicit_report_omissions() {
        let _trace = super::super::test_tracing();
        let params = Params::default();
        let expected_params = serde_json::to_value(params).unwrap();
        let eligible = |finalist| Candidate::Eligible {
            params,
            preflight_ms: 2.0,
            security: json!({"bits":100}),
            finalist,
        };
        assert_eq!(
            serde_json::to_value(eligible(None)).unwrap(),
            json!({
                "status":"eligible","params":expected_params,"preflight_ms":2.0,"security":{"bits":100}
            })
        );
        assert_eq!(
            serde_json::to_value(eligible(Some(Finalist {
                tuning_samples_ms: vec![4.0, 2.0],
                tuning_median_ms: 4.0
            })))
            .unwrap(),
            json!({
                "status":"eligible","params":expected_params,"preflight_ms":2.0,"security":{"bits":100},
                "tuning_samples_ms":[4.0,2.0],"tuning_median_ms":4.0
            })
        );
        assert_eq!(
            serde_json::to_value(Candidate::Ineligible {
                params,
                reason: "budget".into()
            })
            .unwrap(),
            json!({"status":"ineligible","params":expected_params,"reason":"budget"})
        );
        let (_, mut report) = tune(
            &[],
            Some(params),
            |_| Ok(()),
            |_| panic!("explicit mode must not run trials"),
            |_| json!({"bits":100}),
        )
        .unwrap();
        report.tuning_ms = 0.0;
        assert_eq!(
            serde_json::to_value(&report).unwrap(),
            json!({
                "selected":expected_params,"mode":"explicit","security":{"bits":100},
                "tuning_ms":0.0,"candidates":[]
            })
        );
        report.workload = Some("u32".into());
        report.exponent = Some(4);
        report.corpus_digest = Some("digest".into());
        assert_eq!(serde_json::to_value(report).unwrap()["exponent"], 4);
    }
}
