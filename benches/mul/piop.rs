use crate::common;
use bitz::piop::spartan::mul::MulWitness;

use std::hint::black_box;

use bitz::{
    piop::spartan::{
        SpartanBitzField, SpartanPiopProof, UnivariateSkipSpartanPiopProof,
        prepare_u32_mul_relation, project_u32_mul_native_witness, prove_spartan_piop_u32_native,
        prove_spartan_piop_u32_native_with_univariate_skip, spartan_bitz_field_config,
        verify_spartan_proof, verify_spartan_univariate_skip_proof,
    },
    transcript::Blake3Transcript,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const ASSIGNMENT_BINDING: [u8; 32] = [0x73; 32];

use super::{Metrics, Run};
use serde_json::json;
struct ProofShape {
    outer_fields: usize,
    proof_fields: usize,
}
fn standard_proof_shape(
    proof: &SpartanPiopProof<SpartanBitzField>,
    row_vars: usize,
    column_vars: usize,
) -> ProofShape {
    assert_eq!(proof.outer.sumcheck.round_polynomials.len(), row_vars);
    assert_eq!(proof.inner.round_polynomials.len(), column_vars);
    let outer_fields = 4 * proof.outer.sumcheck.round_polynomials.len() + 3;
    let proof_fields = outer_fields + 3 * proof.inner.round_polynomials.len();
    assert_eq!(outer_fields, 4 * row_vars + 3);
    assert_eq!(proof_fields, 4 * row_vars + 3 + 3 * column_vars);
    ProofShape {
        outer_fields,
        proof_fields,
    }
}

fn skip_proof_shape(
    proof: &UnivariateSkipSpartanPiopProof<SpartanBitzField>,
    row_vars: usize,
    column_vars: usize,
    skip_vars: usize,
) -> ProofShape {
    assert_eq!(proof.outer.skip.skip_vars, skip_vars as u8);
    assert_eq!(
        proof.outer.skip.finite_q_evaluations.len(),
        (1usize << skip_vars) - 2
    );
    assert_eq!(
        proof.outer.tail.sumcheck.round_polynomials.len(),
        row_vars - skip_vars
    );
    assert_eq!(proof.inner.round_polynomials.len(), column_vars);

    let skip_fields = proof.outer.skip.finite_q_evaluations.len() + 1;
    let tail_fields = 4 * proof.outer.tail.sumcheck.round_polynomials.len() + 3;
    let outer_fields = skip_fields + tail_fields;
    let proof_fields = outer_fields + 3 * proof.inner.round_polynomials.len();
    let expected_outer_fields = 4 * row_vars - 4 * skip_vars + (1usize << skip_vars) + 2;
    assert_eq!(outer_fields, expected_outer_fields);
    assert_eq!(proof_fields, expected_outer_fields + 3 * column_vars);
    ProofShape {
        outer_fields,
        proof_fields,
    }
}

pub fn run(run: &mut Run) -> anyhow::Result<()> {
    let case = &run.job.case;
    let mut rng = StdRng::seed_from_u64(super::proof::shape_seed(case));
    let witness = MulWitness::<u32>::from_fn(1 << case.log_n, |_| (rng.random(), rng.random()))?;
    let relation = prepare_u32_mul_relation(*witness.layout(), &spartan_bitz_field_config())?;
    let (assignment, products) = project_u32_mul_native_witness(&witness).into_parts();
    let k = case
        .variant
        .as_deref()
        .unwrap()
        .strip_prefix("skip")
        .map(|s| s.parse::<usize>().unwrap());
    run.effective = json!({"boundary":"whole-piop","corpus_digest":common::mul_witness::u32_digest(&witness),"assignment_binding":ASSIGNMENT_BINDING});
    for i in 0..run.trials() {
        let sample_products = products.clone();
        let sample_assignment = assignment.clone();
        run.begin_memory();
        let recording = run
            .latency()
            .then(|| bitz::observability::Recording::start(Vec::new()))
            .transpose()?;
        let mut p = Blake3Transcript::new();
        let mut v = Blake3Transcript::new();
        macro_rules! trial {
            ($prove:expr,$verify:ident,$shape:expr) => {{
                let start = std::time::Instant::now();
                let proving = tracing::info_span!("mul:piop-prove").entered();
                let (proof, claim) = $prove?;
                drop(proving);
                let prove_ms = start.elapsed().as_secs_f64() * 1000.;
                let start = std::time::Instant::now();
                let verification = tracing::info_span!("mul:piop-verify").entered();
                let verified = $verify(&mut v, &relation, &ASSIGNMENT_BINDING, &proof)?;
                drop(verification);
                let verify_ms = start.elapsed().as_secs_f64() * 1000.;
                anyhow::ensure!(claim == verified, "PIOP claim mismatch");
                let shape = $shape(&proof);
                black_box(proof);
                (prove_ms, verify_ms, shape)
            }};
        }
        let (prove_ms, verify_ms, shape) = if let Some(k) = k {
            trial!(
                prove_spartan_piop_u32_native_with_univariate_skip(
                    &mut p,
                    &relation,
                    &ASSIGNMENT_BINDING,
                    sample_products,
                    sample_assignment,
                    k
                ),
                verify_spartan_univariate_skip_proof,
                |proof| skip_proof_shape(
                    proof,
                    relation.num_row_vars(),
                    relation.num_column_vars(),
                    k
                )
            )
        } else {
            trial!(
                prove_spartan_piop_u32_native(
                    &mut p,
                    &relation,
                    &ASSIGNMENT_BINDING,
                    sample_products,
                    sample_assignment
                ),
                verify_spartan_proof,
                |proof| standard_proof_shape(
                    proof,
                    relation.num_row_vars(),
                    relation.num_column_vars()
                )
            )
        };
        run.end_memory();
        let mut m = Metrics::from([
            ("piop_ms".into(), prove_ms),
            ("verify_ms".into(), verify_ms),
            (
                "analytical_piop_bytes".into(),
                (shape.proof_fields * 16) as f64,
            ),
            ("outer_fields".into(), shape.outer_fields as f64),
        ]);
        if let Some(r) = recording {
            let spans = r.intervals()?;
            for (prefix, scope) in [("prove", "mul:piop-prove"), ("verify", "mul:piop-verify")] {
                for (name, ms) in super::phase_milliseconds(&spans, scope)? {
                    m.insert(format!("{prefix}/{name}_ms"), ms);
                }
            }
        }
        run.sample(i, m);
    }
    Ok(())
}
