use super::{Metrics, Run, config::Workload};
use ::bitz::{
    pcs::FQ_MOD,
    piop::spartan::{
        baby_bear_mul::sample_baby_bear_operand_with,
        mul::{MulLayout, MulWitness},
        protocol::{self, PreparedRelation, RelationSpec, terminal::PreparedTerminalOpening},
        *,
    },
    transcript::{Blake3Transcript, traits::Transcript},
};
use anyhow::{Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::time::Instant;
#[cfg(feature = "native-mul-compare")]
#[path = "../integer_pcs_compare/binius.rs"]
mod binius;
#[cfg(feature = "native-mul-compare")]
#[path = "../integer_pcs_compare/ligerito.rs"]
mod ligerito;
#[cfg(feature = "native-mul-compare")]
#[path = "../baby_bear_pcs_compare/whir.rs"]
mod whir_bb;
#[cfg(feature = "native-mul-compare")]
#[path = "../integer_pcs_compare/whir_goldilocks.rs"]
mod whir_u32;

enum Witness {
    U32(MulWitness<u32>),
    BabyBear(BabyBearMulWitness),
}
impl Witness {
    fn row(&self, i: usize) -> [u128; 4] {
        match self {
            Self::U32(w) => [
                w.x_values()[i] as u128,
                w.y_values()[i] as u128,
                w.product(i) as u128,
                0,
            ],
            Self::BabyBear(w) => [
                w.a_values()[i] as u128,
                w.b_values()[i] as u128,
                w.c_values()[i] as u128,
                w.k_values()[i] as u128,
            ],
        }
    }
    fn gate_vars(&self) -> usize {
        match self {
            Self::U32(w) => w.layout().gate_vars(),
            Self::BabyBear(w) => w.layout().gate_vars(),
        }
    }
    fn selectors(&self) -> usize {
        match self {
            Self::U32(_) => 2,
            Self::BabyBear(_) => 3,
        }
    }
    fn rows(&self) -> Vec<Vec<u64>> {
        match self {
            Self::U32(w) => w.bitz_bit_rows(),
            Self::BabyBear(w) => w.bitz_bit_rows(),
        }
    }
    #[cfg(feature = "native-mul-compare")]
    fn packed(&self) -> Vec<u128> {
        (0..1 << self.gate_vars())
            .map(|i| {
                let [a, b, c, k] = self.row(i);
                match self {
                    Self::U32(_) => a | (b << 32) | (c << 64),
                    Self::BabyBear(_) => a | (b << 31) | (c << 62) | (k << 93),
                }
            })
            .collect()
    }
    fn digest(&self) -> String {
        match self {
            Self::U32(w) => crate::common::mul_witness::u32_digest(w),
            Self::BabyBear(w) => crate::common::mul_witness::baby_bear_digest(w),
        }
    }
    fn transcript(&self, commitment: &[u8], seed: u64) -> Blake3Transcript {
        let mut t = Blake3Transcript::new();
        t.absorb_slice(match self {
            Self::U32(_) => b"bitz/u32-pcs-compare/terminal-claim/v1",
            Self::BabyBear(_) => b"bitz/baby-bear-pcs-compare/terminal-claim/v1",
        });
        t.absorb_slice(&seed.to_le_bytes());
        t.absorb_slice(commitment);
        t
    }
    fn claim(
        &self,
        mut p: Blake3Transcript,
        mut v: Blake3Transcript,
    ) -> (
        ScaledMleEvaluationClaim<SpartanBitzField>,
        Blake3Transcript,
        Blake3Transcript,
    ) {
        let cfg = spartan_bitz_field_config();
        let draw = |t: &mut Blake3Transcript| {
            t.begin_sampling();
            SpartanBitzField::sample_uniform(t, &cfg).expect("public claim sampling")
        };
        let point: Vec<_> = (0..self.gate_vars() + self.selectors())
            .map(|_| draw(&mut p))
            .collect();
        let scale = draw(&mut p);
        let vp: Vec<_> = (0..point.len()).map(|_| draw(&mut v)).collect();
        assert_eq!(point, vp);
        assert_eq!(scale, draw(&mut v));
        let arith = field::FpCtx::from_prime_u128(FQ_MOD);
        let canonical: Vec<u128> = point.iter().map(|x| cfg.to_integer(x).into()).collect();
        let equality = |point: &[u128]| {
            let mut table = vec![1];
            for &r in point {
                let n = table.len();
                table.resize(n * 2, 0);
                for i in 0..n {
                    let one = arith.mul_u128(table[i], r);
                    table[i] = if one == 0 {
                        table[i]
                    } else {
                        arith.add_u128(table[i], FQ_MOD - one)
                    };
                    table[i + n] = one;
                }
            }
            table
        };
        let eq = equality(&canonical[..self.gate_vars()]);
        let selector = equality(&canonical[self.gate_vars()..]);
        let mut value = arith.mul_u128(selector[0], eq[0]);
        for (i, &weight) in eq.iter().enumerate() {
            let mut selected = 0;
            for (j, x) in self
                .row(i)
                .into_iter()
                .take(if self.selectors() == 2 { 3 } else { 4 })
                .enumerate()
            {
                selected = arith.add_u128(selected, arith.mul_u128(selector[j + 1], x));
            }
            value = arith.add_u128(value, arith.mul_u128(weight, selected));
        }
        value = arith.mul_u128(u128::from(cfg.to_integer(&scale)), value);
        (
            ScaledMleEvaluationClaim::new(
                point.into_boxed_slice(),
                scale,
                SpartanBitzField::from_with_cfg(value, &cfg),
            ),
            p,
            v,
        )
    }
}
struct Sizes {
    commitment: usize,
    claim: usize,
    opening: usize,
}
fn loop_trials(
    run: &mut Run,
    w: &Witness,
    security: Value,
    setup_ms: f64,
    mut trial: impl FnMut(u64) -> Result<Sizes>,
) -> Result<()> {
    run.effective = json!({"corpus_digest":w.digest(),"security":security,"boundary":"pcs-opening","packing_bits":if w.selectors()==2 {vec![32,32,64]}else{vec![31,31,31,31]}});
    let seed = super::proof::shape_seed(&run.job.case);
    let tag = match (w.selectors(), run.job.case.backend.as_str()) {
        (2, "bitz") => 0x4632_5a00_5533_0001,
        (2, "plonky3-whir") => 0x5748_4952_5533_0001,
        (2, "binius64-basefold") => 0x4249_4e49_5533_0001,
        (2, _) => 0x4c49_4745_5533_0001,
        (_, "bitz") => 0x4632_5a00_0000_0001,
        (_, "plonky3-whir") => 0x5748_4952_0000_0001,
        (_, "binius64-basefold") => 0x4249_4e49_5553_0001,
        (_, _) => 0x4c49_4745_0000_0001,
    };
    for i in 0..run.trials() {
        run.begin_memory();
        let recording = run
            .latency()
            .then(|| ::bitz::observability::Recording::start(Vec::new()))
            .transpose()?;
        let sizes = trial(mix_seed(seed ^ tag ^ (i as u64).rotate_left(17)))?;
        run.end_memory();
        ensure!(
            sizes.commitment > 0 && sizes.opening > 0,
            "empty PCS artifact"
        );
        let mut m = Metrics::from([
            ("setup_ms".into(), setup_ms),
            ("commitment_bytes".into(), sizes.commitment as f64),
            ("claim_bytes".into(), sizes.claim as f64),
            ("opening_bytes".into(), sizes.opening as f64),
            (
                "proof_bytes".into(),
                (sizes.commitment + sizes.claim + sizes.opening) as f64,
            ),
        ]);
        if let Some(recording) = recording {
            let spans = recording.intervals()?;
            for (metric, scope) in [
                ("materialize_ms", "materialize"),
                ("commit_ms", "commit"),
                ("claim_ms", "claim_setup"),
                ("opening_ms", "opening"),
                ("verify_ms", "verification"),
                ("verified_trial_ms", "verified_trial"),
            ] {
                m.insert(
                    metric.into(),
                    crate::common::span_ms(&spans, &format!("pcs-compare:{scope}")),
                );
            }
            m.insert(
                "pcs_ms".into(),
                m["materialize_ms"] + m["commit_ms"] + m["opening_ms"],
            );
        }
        run.sample(i, m);
    }
    Ok(())
}
fn mix_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
fn terminal<S: RelationSpec>(
    run: &mut Run,
    w: &Witness,
    prepared: PreparedTerminalOpening<S>,
    setup_ms: f64,
    security: Value,
) -> Result<()> {
    loop_trials(run, w, security, setup_ms, |seed| {
        let root = tracing::info_span!("pcs-compare:verified_trial").entered();
        let rows = tracing::info_span!("pcs-compare:materialize").in_scope(|| w.rows());
        let (hint, encoding, p, v) =
            tracing::info_span!("pcs-compare:commit").in_scope(|| -> Result<_> {
                let hint = protocol::terminal::commit(&prepared, rows)?;
                let encoding = bincode::serialize(&hint.commitment)?;
                let p = w.transcript(&encoding, seed);
                let v = w.transcript(&encoding, seed);
                Ok((hint, encoding, p, v))
            })?;
        let (claim, mut p, mut v) =
            tracing::info_span!("pcs-compare:claim_setup").in_scope(|| w.claim(p, v));
        let proof = tracing::info_span!("pcs-compare:opening")
            .in_scope(|| protocol::terminal::prove(&mut p, &prepared, &hint, &claim))?;
        tracing::info_span!("pcs-compare:verification").in_scope(|| {
            protocol::terminal::verify(&mut v, &prepared, &hint.commitment, &claim, &proof)
        })?;
        drop(root);
        let bytes = ::bitz::ligerito_flock::IntEvalRsLigModQProof::to_bytes(&proof);
        let decoded = ::bitz::ligerito_flock::IntEvalRsLigModQProof::from_bytes(&bytes)?;
        ensure!(decoded.to_bytes() == bytes, "PCS codec roundtrip");
        Ok(Sizes {
            commitment: encoding.len(),
            claim: (w.gate_vars() + w.selectors() + 2) * 16,
            opening: bytes.len(),
        })
    })
}
pub fn run(run: &mut Run) -> Result<()> {
    let case = run.job.case.clone();
    let mut rng = StdRng::seed_from_u64(super::proof::shape_seed(&case));
    let n = 1 << case.log_n;
    let w = if case.workload == Workload::U32Full {
        Witness::U32(MulWitness::from_fn(n, |_| (rng.random(), rng.random()))?)
    } else {
        Witness::BabyBear(BabyBearMulWitness::from_fn(n, |_| {
            (
                sample_baby_bear_operand_with(|| rng.random()),
                sample_baby_bear_operand_with(|| rng.random()),
            )
        })?)
    };
    let start = Instant::now();
    if case.backend == "bitz" {
        let selection = super::proof::selection(&case);
        match &w {
            Witness::U32(witness) => {
                let relation = PreparedRelation::<MulLayout<u32>>::new_with_profile_and_ligerito::<
                    Lambda100,
                >(*witness.layout(), selection)?;
                let hint = protocol::commit(&relation, witness.bitz_bit_rows())?;
                let commitment = hint.commitment.clone();
                drop(hint);
                flock_core::scratch::clear();
                let p = ::bitz::piop::spartan::bitz::prepare_u32_terminal_bitz_opening(
                    &relation,
                    &commitment,
                )?;
                let lig = relation.ligerito_configuration();
                let security = lig.report(&lig.selection().name(), relation.security().ood);
                drop((relation, commitment));
                terminal(run, &w, p, start.elapsed().as_secs_f64() * 1000., security)
            }
            Witness::BabyBear(witness) => {
                let layout = *witness.layout();
                let hint = baby_bear_bitz::commit_baby_bear_mul_witness_with_ligerito(
                    &layout,
                    witness.bitz_bit_rows(),
                    selection,
                )?;
                let commitment = hint.commitment.clone();
                drop(hint);
                flock_core::scratch::clear();
                let matrices = prepare_baby_bear_mul_relation(layout, &spartan_bitz_field_config())?;
                let p = baby_bear_bitz::prepare_baby_bear_terminal_bitz_opening_with_ligerito(
                    &matrices,
                    &layout,
                    &commitment,
                    selection,
                )?;
                drop((matrices, commitment));
                let lig = selection
                    .resolve(case.log_n, 100)
                    .map_err(anyhow::Error::msg)?;
                let security = lig.report(
                    &lig.selection().name(),
                    lig.round0(100).map_err(anyhow::Error::msg)?,
                );
                terminal(run, &w, p, start.elapsed().as_secs_f64() * 1000., security)
            }
        }
    } else {
        #[cfg(feature = "native-mul-compare")]
        {
            competitor(run, &w, start)
        }
        #[cfg(not(feature = "native-mul-compare"))]
        anyhow::bail!("comparison requires native-mul-compare")
    }
}
#[cfg(feature = "native-mul-compare")]
fn competitor(run: &mut Run, w: &Witness, start: Instant) -> Result<()> {
    let case = run.job.case.clone();
    macro_rules! binary {
        ($backend:expr,$security:expr) => {{
            let backend = $backend;
            let security = $security(&backend);
            loop_trials(
                run,
                w,
                security,
                start.elapsed().as_secs_f64() * 1000.,
                |seed| {
                    let output = backend
                        .run_trial(|| w.packed(), seed)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    Ok(Sizes {
                        commitment: output.commitment_bytes,
                        claim: output.public_claim_bytes,
                        opening: output.opening_proof_bytes(),
                    })
                },
            )
        }};
    }
    match case.backend.as_str() {
        "binius64-basefold" => binary!(
            binius::BiniusBackend::setup(case.log_n, case.log_inv_rate.unwrap_or(1) as usize),
            |b: &binius::BiniusBackend| json!({"target_bits":binius::SECURITY_BITS,"estimated_soundness_bits":b.estimated_soundness_bits(),"log_inv_rate":b.log_inv_rate(),"queries":b.n_test_queries()})
        ),
        "bitz-ligerito-binary" => binary!(
            ligerito::LigeritoBackend::setup(case.log_n).map_err(|e| anyhow::anyhow!("{e}"))?,
            |b: &ligerito::LigeritoBackend| json!({"target_bits":ligerito::SECURITY_BITS,"soundness_bits":b.soundness_bits(),"component_bits":b.component_bits(),"log_inv_rate":b.log_inv_rate(),"queries":b.n_test_queries()})
        ),
        "plonky3-whir" => whir(run, w, start),
        _ => unreachable!("validated PCS backend"),
    }
}
#[cfg(feature = "native-mul-compare")]
fn whir(run: &mut Run, w: &Witness, start: Instant) -> Result<()> {
    let case = run.job.case.clone();
    let params = case.whir.expect("resolved WHIR configuration");
    let folding = params.folding;
    macro_rules! trial {
        ($module:ident,$ty:ident,$wit:ident,$materialize:expr,$claim:ident)=>{{
            let backend=$module::$ty::setup_with_params(1<<w.gate_vars(),folding,params.log_inv_rate,params.max_pow_bits).map_err(|e|super::Unsupported(e.to_string()))?;
            let summary=backend.security_summary();
            let security=json!({"target_bits":summary.target_bits,"degree":$module::CHALLENGE_EXTENSION_DEGREE,"assumption":$module::SECURITY_ASSUMPTION_LABEL,"folding":summary.folding_factor,"log_inv_rate":summary.starting_log_inverse_rate,"max_pow_bits":summary.configured_max_pow_bits,"derived_pow_bits":summary.derived_max_pow_bits,"round_queries":summary.round_queries,"round_pow_bits":summary.round_pow_bits,"final_queries":summary.final_queries,"final_pow_bits":summary.final_pow_bits});
            loop_trials(run,w,security,start.elapsed().as_secs_f64()*1000.,|seed| {
                let root=tracing::info_span!("pcs-compare:verified_trial").entered();
                let materialized=tracing::info_span!("pcs-compare:materialize").in_scope(||$materialize(&backend,$wit))?;
                let committed=tracing::info_span!("pcs-compare:commit").in_scope(||backend.commit(materialized,seed));
                let ready=tracing::info_span!("pcs-compare:claim_setup").in_scope(||backend.$claim(committed))?;
                let opened=tracing::info_span!("pcs-compare:opening").in_scope(||backend.open(ready));
                let _verified = tracing::info_span!("pcs-compare:verification").in_scope(||backend.verify(&opened))?;
                drop(root);
                Ok(Sizes {commitment:$module::commitment_bytes(opened.commitment())?,claim:(w.gate_vars()+w.selectors()+2)*16,opening:$module::proof_bytes(opened.proof())?})
            })
        }}
    }
    match w {
        Witness::U32(wit) => trial!(
            whir_u32,
            Backend,
            wit,
            |b: &whir_u32::Backend, w: &MulWitness<u32>| b.materialize(w),
            derive_and_bind_claim
        ),
        Witness::BabyBear(wit) => trial!(
            whir_bb,
            WhirBackend,
            wit,
            |b: &whir_bb::WhirBackend, w: &BabyBearMulWitness| b.materialize(
                w.a_values(),
                w.b_values(),
                w.c_values(),
                w.k_values()
            ),
            derive_and_bind_terminal_claim
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn claims_match_dense_assignment_folding() {
        let mut rng = StdRng::seed_from_u64(7);
        let witnesses = [
            Witness::U32(MulWitness::from_fn(16, |_| (rng.random(), rng.random())).unwrap()),
            Witness::BabyBear(
                BabyBearMulWitness::from_fn(16, |_| {
                    (
                        sample_baby_bear_operand_with(|| rng.random()),
                        sample_baby_bear_operand_with(|| rng.random()),
                    )
                })
                .unwrap(),
            ),
        ];
        let arith = field::FpCtx::from_prime_u128(FQ_MOD);
        for w in witnesses {
            let (claim, _, _) = w.claim(w.transcript(b"test", 7), w.transcript(b"test", 7));
            let mut values: Vec<u128> = match &w {
                Witness::U32(w) => w.assignment().into_iter().map(u128::from).collect(),
                Witness::BabyBear(w) => w.assignment().iter().copied().map(u128::from).collect(),
            };
            values.resize(1 << claim.point().len(), 0);
            let cfg = spartan_bitz_field_config();
            for r in claim.point() {
                let r = u128::from(cfg.to_integer(r));
                values = values
                    .chunks_exact(2)
                    .map(|pair| {
                        let difference = arith
                            .add_u128(pair[1], if pair[0] == 0 { 0 } else { FQ_MOD - pair[0] });
                        arith.add_u128(pair[0], arith.mul_u128(r, difference))
                    })
                    .collect();
            }
            assert_eq!(values.len(), 1);
            assert_eq!(
                arith.mul_u128(values[0], u128::from(cfg.to_integer(claim.scale()))),
                u128::from(cfg.to_integer(claim.value()))
            );
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "production-size PCS proof roundtrips"]
fn terminal_openings_verify_for_both_witness_layouts() {
    use clap::Parser;
    let args = super::config::Args::try_parse_from([
        "mul",
        "pcs",
        "--workload",
        "u32-full,baby-bear",
        "--log-n",
        "15",
        "--threads",
        "1",
        "--reps",
        "1",
        "--warmups",
        "0",
        "--memory",
        "rss",
    ])
    .unwrap();
    for job in args.expand(false).unwrap() {
        let mut result = Run::new(job);
        run(&mut result).unwrap();
        assert_eq!(result.samples.len(), 1);
        assert!(result.samples[0].metrics["proof_bytes"] > 0.);
    }
}
