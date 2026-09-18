use super::{Metrics, Run, config::Workload};
use bitz::{
    piop::spartan::SpartanField,
    sumcheck::{UngrindedRoundBoundary, outer::*},
    transcript::Blake3Transcript,
};
use field::{Fp, FpCtx, IntegerEmbedding, RingOps, Uint, WideMul};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::time::Instant;
fn bench_inputs<A, C>(
    run: &mut Run,
    field: &FpCtx<2>,
    a: &[A],
    b: &[A],
    c: &[C],
) -> anyhow::Result<()>
where
    A: Copy + Send + Sync,
    C: Copy + Send + Sync,
    FpCtx<2>: OuterArithmetic<A, C>,
{
    let n = a.len().ilog2() as usize;
    let tau: Vec<_> = (0..n)
        .map(|i| field.from_integer(&(i as u64 + 7)))
        .collect();
    {
        let protocol = run.job.case.variant.as_deref().unwrap();
        let k: usize = protocol
            .strip_prefix("skip")
            .map_or(0, |s| s.parse().unwrap());
        if n < k {
            unreachable!("validated prefix length");
        }
        let prepared = (k > 0).then(|| prepare_univariate_skip(field, k as u8).unwrap());
        let invoke = || {
            let mut transcript = Blake3Transcript::new();

            let start = Instant::now();
            let (out, prefix) = if let Some(prepared) = &prepared {
                let out = bitz::sumcheck::outer::prove_outer_zerocheck_with_skip(
                    field,
                    &mut transcript,
                    prepared,
                    &tau[k..],
                    bitz::sumcheck::outer::OuterSlices {
                        ax: a,
                        bx: b,
                        cx: c,
                    },
                    None,
                    &mut UngrindedRoundBoundary,
                )
                .unwrap();
                (out.tail, Some(out.prefix))
            } else {
                let out = if protocol == "standard" {
                    bitz::sumcheck::outer::prove_outer_sumcheck(
                        field,
                        &mut transcript,
                        bitz::sumcheck::outer::OuterClaim::Sum(field.zero()),
                        &tau,
                        bitz::sumcheck::outer::OuterSlices {
                            ax: a,
                            bx: b,
                            cx: c,
                        },
                        None,
                        &mut UngrindedRoundBoundary,
                    )
                    .unwrap()
                } else {
                    bitz::sumcheck::outer::prove_outer_sumcheck(
                        field,
                        &mut transcript,
                        bitz::sumcheck::outer::OuterClaim::RowwiseZero,
                        &tau,
                        bitz::sumcheck::outer::OuterSlices {
                            ax: a,
                            bx: b,
                            cx: c,
                        },
                        None,
                        &mut UngrindedRoundBoundary,
                    )
                    .unwrap()
                };
                (out, None)
            };
            let nanos = start.elapsed().as_nanos();
            let digest = transcript.state_digest();
            let mut verifier = Blake3Transcript::new();
            if let Some(prefix) = &prefix {
                verify_outer_zerocheck_with_skip(
                    field,
                    &mut verifier,
                    prepared.as_ref().unwrap(),
                    &tau[k..],
                    prefix,
                    &out.proof,
                    out.evaluations,
                    &mut UngrindedRoundBoundary,
                )
                .unwrap();
            } else {
                verify_outer_sumcheck(
                    field,
                    &mut verifier,
                    field.zero(),
                    &tau,
                    &out.proof,
                    out.evaluations,
                    &mut UngrindedRoundBoundary,
                )
                .unwrap();
            }
            assert_eq!(digest, verifier.state_digest());
            (nanos, out, prefix, digest)
        };

        run.begin_memory();
        let mut reference = None;
        let trials = run.trials();
        let mut samples = Vec::new();
        for _ in 0..trials {
            let (ns, out, p, d) = invoke();
            if let Some((expected, prefix, digest)) = &reference {
                assert_eq!(&out, expected);
                assert_eq!(&p, prefix);
                assert_eq!(&d, digest);
            } else {
                reference = Some((out, p, d));
            }
            samples.push(ns);
        }
        run.end_memory();
        for (i, ns) in samples.into_iter().enumerate() {
            run.sample(i, Metrics::from([("outer_ms".into(), ns as f64 / 1e6)]));
        }
    }
    Ok(())
}
pub fn run(run: &mut Run) -> anyhow::Result<()> {
    if run.job.case.preset.as_deref() == Some("regression") {
        return regression(run);
    }
    let case = &run.job.case;
    let field = Fp::<2>::make_cfg(&Uint::from((1u128 << 100) - 15)).unwrap();
    let mut rng = StdRng::seed_from_u64(case.seed ^ case.log_n as u64);
    let a: Vec<u128> = (0..1 << case.log_n).map(|_| rng.random()).collect();
    let b: Vec<u128> = (0..a.len()).map(|_| rng.random()).collect();
    let corpus = blake3::hash(&bincode::serialize(&(&a, &b))?)
        .to_hex()
        .to_string();
    run.effective = json!({"boundary":"outer-kernel","fixture_digest":corpus,"field_modulus":((1u128<<100)-15).to_string()});
    match run.job.case.workload {
        Workload::U32Full | Workload::U32Mod32 => {
            let a: Vec<u32> = a.iter().map(|&x| x as u32).collect();
            let b: Vec<u32> = b.iter().map(|&x| x as u32).collect();
            let c: Vec<u64> = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| a as u64 * b as u64)
                .collect();
            bench_inputs(run, &field, &a, &b, &c)
        }
        Workload::U64 => {
            let a: Vec<u64> = a.iter().map(|&x| x as u64).collect();
            let b: Vec<u64> = b.iter().map(|&x| x as u64).collect();
            let c: Vec<u128> = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| a as u128 * b as u128)
                .collect();
            bench_inputs(run, &field, &a, &b, &c)
        }
        Workload::U128 => {
            let c: Vec<Uint<4>> = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| {
                    *field::IntegerOps
                        .mul_wide(&Uint::<2>::from(a), &Uint::<2>::from(b))
                        .checked_resize_ct::<4>()
                        .value()
                })
                .collect();
            bench_inputs(run, &field, &a, &b, &c)
        }
        Workload::Field => {
            let a: Vec<_> = a.iter().map(|x| field.from_integer(x)).collect();
            let b: Vec<_> = b.iter().map(|x| field.from_integer(x)).collect();
            let c: Vec<_> = a.iter().zip(&b).map(|(a, b)| field.mul(a, b)).collect();
            bench_inputs(run, &field, &a, &b, &c)
        }
        _ => unreachable!("validated outer workload"),
    }
}
fn regression(run: &mut Run) -> anyhow::Result<()> {
    let case = &run.job.case;
    let bits = match case.workload {
        Workload::U32Full | Workload::U32Mod32 => 32,
        Workload::U64 => 64,
        Workload::U128 => 128,
        _ => unreachable!(),
    };
    let k = case
        .variant
        .as_deref()
        .unwrap()
        .strip_prefix("skip")
        .map_or(0, |s| s.parse().unwrap());
    run.begin_memory();
    let (fixture, samples) =
        bitz::piop::spartan::outer_regression::samples(bits, case.log_n, case.seed, k, run.trials());
    run.end_memory();
    run.effective = json!({"boundary":"outer-regression","fixture_digest":fixture,"comparison":"production and generic proofs must match"});
    for (i, (production, generic)) in samples.into_iter().enumerate() {
        run.sample(
            i,
            Metrics::from([
                ("production_ms".into(), production),
                ("generic_ms".into(), generic),
            ]),
        );
    }
    Ok(())
}
