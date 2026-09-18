//! Per-scope profile of one `protocol::prove` at a chosen size:
//! `cargo run --release --features unchecked,span-metrics --example u64_mul_probe -- 21`.

use ::bitz::piop::spartan::protocol;
use ::bitz::piop::spartan::protocol::PreparedRelation;
use bitz::piop::spartan::mul::{MulLayout, MulWitness};

use bitz::transcript::Blake3Transcript;

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    let e: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(18);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let witness = MulWitness::<u64>::from_fn(1 << e, |i| {
        let x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let y = (i as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f) | 1;
        (x, y)
    })
    .unwrap();
    // Experiment: BITZ_U64_SPLIT_SHIFT=k moves k gate variables from rows to columns.
    let shift: i8 = std::env::var("BITZ_U64_SPLIT_SHIFT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let witness = witness.with_split_shift(shift).unwrap();
    let params = witness.layout().bitz_params();
    println!(
        "split: shift {shift} -> t={} s={}",
        params.row_vars, params.col_vars
    );
    let prepared = PreparedRelation::<MulLayout<u64>>::new(*witness.layout()).unwrap();
    let hint = protocol::commit(&prepared, witness.bitz_bit_rows()).unwrap();
    // warm-up

    let p = protocol::prove(&mut Blake3Transcript::new(), &prepared, &witness, &hint).unwrap();
    protocol::verify(
        &mut Blake3Transcript::new(),
        &prepared,
        &hint.commitment,
        &p,
    )
    .unwrap();

    let mut totals: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut vtotals: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut wall = vec![];
    let mut vwall = vec![];
    for _ in 0..reps {
        let profile =
            bitz::observability::Recording::start(Vec::new()).expect("capture prover profile");
        let (p, t) = bitz::observability::measure(tracing::info_span!("u64_mul_probe:p"), || {
            protocol::prove(&mut Blake3Transcript::new(), &prepared, &witness, &hint).unwrap()
        })
        .expect("measure completed operation");
        wall.push(t.as_secs_f64() * 1e3);
        for (label, secs) in bitz::observability::totals(&profile.intervals().expect("prover intervals")) {
            totals.entry(label.to_string()).or_default().push(secs * 1e3);
        }
        let profile =
            bitz::observability::Recording::start(Vec::new()).expect("capture verifier profile");
        let t_recording =
            bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t = tracing::info_span!("u64_mul_probe:t").entered();
        protocol::verify(
            &mut Blake3Transcript::new(),
            &prepared,
            &hint.commitment,
            &p,
        )
        .unwrap();
        vwall.push(
            {
                drop(t);
                bitz::observability::duration(
                    &t_recording.intervals().expect("complete operation capture"),
                    "u64_mul_probe:t",
                )
                .expect("query completed operation")
            }
            .as_secs_f64()
                * 1e3,
        );
        for (label, secs) in
            bitz::observability::totals(&profile.intervals().expect("verifier intervals"))
        {
            vtotals
                .entry(label.to_string())
                .or_default()
                .push(secs * 1e3);
        }
        std::hint::black_box(p);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let report =
        |title: &str, wall: &mut Vec<f64>, totals: std::collections::BTreeMap<String, Vec<f64>>| {
            println!(
                "2^{e}: {title} wall median {:.1} ms over {reps} reps",
                med(wall)
            );
            let mut rows: Vec<_> = totals.into_iter().collect();
            rows.sort_by(|a, b| {
                b.1.iter()
                    .cloned()
                    .fold(0.0, f64::max)
                    .partial_cmp(&a.1.iter().cloned().fold(0.0, f64::max))
                    .unwrap()
            });
            for (label, mut v) in rows {
                println!("  {:<48} {:8.2} ms", label, med(&mut v));
            }
        };
    report("prove", &mut wall, totals);
    report("verify", &mut vwall, vtotals);
    let p = protocol::prove(&mut Blake3Transcript::new(), &prepared, &witness, &hint).unwrap();
    println!(
        "proof bytes: {}",
        hint.commitment.root.len() + p.size_bytes(prepared.security())
    );
}
