//! benches/logup_gkr.rs
//! Criterion benchmarks for the LogUp-GKR range proof
//! (`limber::logup_gkr::LogUpRangeProof`): prove + verify of a 16-bit range
//! check on `N` witnesses, over the T256 Hyrax base field. Sweeps `N`; the
//! table side is fixed at `2^16`.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench logup_gkr
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use limber::{
  logup_gkr::LogUpRangeProof,
  provider::T256HyraxEngine,
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Duration;

type E = T256HyraxEngine;

const BITS: usize = 16;

/// `N` random witnesses, each in `[0, 2^BITS)`.
fn make_witness(n: usize) -> Vec<u64> {
  let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ n as u64);
  (0..n).map(|_| rng.gen_range(0..(1u64 << BITS))).collect()
}

/// Analytical proof size in field elements: per fraction tree of depth `d`,
/// layer `k` carries `4k` round-poly evals + `4` leaf evals, plus the 4 roots.
fn proof_field_elems(d_lhs: usize, d_rhs: usize) -> usize {
  let per_tree = |d: usize| 2 * d * d + 2 * d; // sum_{k<d}(4k+4)
  per_tree(d_lhs) + per_tree(d_rhs) + 4
}

fn logup_gkr_benches(c: &mut Criterion) {
  let log_ns: &[usize] = &[10, 12, 14, 16, 18, 20];

  // Report shape + proof size once per config.
  for &log_n in log_ns {
    let d_lhs = log_n; // N is a power of two here
    let fe = proof_field_elems(d_lhs, BITS);
    println!(
      "logup_gkr bits={BITS} N=2^{log_n}: proof ≈ {fe} field elements (~{} KB at 32 B/elem)",
      fe * 32 / 1024
    );
  }

  let mut g = c.benchmark_group("logup_gkr");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(200));
  g.measurement_time(Duration::from_secs(10));

  for &log_n in log_ns {
    let n = 1usize << log_n;
    let tag = format!("bits{BITS}_N2^{log_n}");

    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || make_witness(n),
        |witness| {
          let mut t = <E as Engine>::TE::new(b"logup_bench");
          let _ = LogUpRangeProof::<E>::prove(BITS, &witness, &mut t).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let witness = make_witness(n);
          let mut t = <E as Engine>::TE::new(b"logup_bench");
          let (proof, _claims) = LogUpRangeProof::<E>::prove(BITS, &witness, &mut t).unwrap();
          proof
        },
        |proof| {
          let mut t = <E as Engine>::TE::new(b"logup_bench");
          let _ = proof.verify(BITS, &mut t).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }
  g.finish();
}

criterion_group!(benches, logup_gkr_benches);
criterion_main!(benches);
