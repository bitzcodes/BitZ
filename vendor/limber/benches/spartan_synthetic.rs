//! benches/spartan_synthetic.rs
//! Plain-Spartan baseline for direct comparison with `imod_spartan.rs`.
//! The circuit is N independent multiplications `a_i · b_i = c_i`, picked
//! so that the resulting R1CS shape sizes line up with the imod bench
//! (`num_cons` ≈ N, `num_vars` ≈ 3N rounded up to a power of two).
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench spartan_synthetic`
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ff::{Field, PrimeField};
use limber::{
  provider::T256HyraxEngine,
  spartan::SpartanSNARK,
  traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use std::{marker::PhantomData, time::Duration};

type E = T256HyraxEngine;

#[derive(Clone, Debug)]
struct SyntheticMulCircuit<F: PrimeField> {
  n: usize,
  /// Full-width witness values (offset by ~random 255-bit constants)
  /// instead of small 1–100 values, so the witness commit can't take
  /// Hyrax's small-scalar MSM fast path. Used by the MultiSwap-shaped
  /// config to stay value-comparable with `imod_spartan_modp`'s
  /// `msshape` (uniform 256-bit operands).
  wide: bool,
  _p: PhantomData<F>,
}

impl<F: PrimeField> SyntheticMulCircuit<F> {
  fn new(n: usize) -> Self {
    Self {
      n,
      wide: false,
      _p: PhantomData,
    }
  }

  fn new_wide(n: usize) -> Self {
    Self {
      n,
      wide: true,
      _p: PhantomData,
    }
  }
}

impl<E: Engine> SpartanCircuit<E> for SyntheticMulCircuit<E::Scalar> {
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(vec![])
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    // Fixed full-width offsets for `wide` mode: any nothing-up-my-sleeve
    // large constants work — the point is full-width limbs per witness
    // value, not cryptographic randomness.
    let (off_a, off_b) = if self.wide {
      let two = E::Scalar::from(2u64);
      let off = two.pow_vartime([251u64]); // ≈ 2^251, full-width limbs
      (off, off * E::Scalar::from(3u64))
    } else {
      (E::Scalar::from(0u64), E::Scalar::from(0u64))
    };
    for i in 0..self.n {
      let a = AllocatedNum::alloc(cs.namespace(|| format!("a_{i}")), || {
        Ok(off_a + E::Scalar::from((i as u64 % 100) + 1))
      })?;
      let b = AllocatedNum::alloc(cs.namespace(|| format!("b_{i}")), || {
        Ok(off_b + E::Scalar::from(((i as u64 * 7) % 100) + 1))
      })?;
      let _c = a.mul(cs.namespace(|| format!("c_{i}")), &b)?;
    }
    Ok(())
  }
}

fn spartan_synthetic_benches(c: &mut Criterion) {
  // PSIZE=1: serialized proof size per msshape config (plain-Spartan
  // baseline for the paper's proof-size comparison).
  if std::env::var_os("PSIZE").is_some() {
    for &n in &[682usize, 2730, 10922] {
      let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new_wide(n);
      let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
      let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
      let (proof, _) = SpartanSNARK::<E>::prove(&pk, circuit, prep, false).unwrap();
      proof.verify(&vk).unwrap();
      println!(
        "spartan msshape c2^{} proof size: {} bytes",
        (n.next_power_of_two() as u64).ilog2(),
        bincode::serialized_size(&proof).map_or(0, |v| v as usize)
      );
    }
    return;
  }

  // Match imod_spartan(_modp) bench: num_cons targets 2^k for k ∈ {6, 8,
  // 10, 12, 14}. Plain Spartan pads internally, so the realised shape
  // may be slightly larger than N.
  let configs: &[usize] = &[1 << 6, 1 << 8, 1 << 10, 1 << 12, 1 << 14];

  // Per-part timing breakdown for the msshape config, gated on
  // `RUST_LOG` (mirrors imod_spartan_modp): one setup/prove/verify with
  // the library spans (witness commit, matvec, sumchecks, PCS opens)
  // visible without criterion's iteration noise.
  if std::env::var_os("RUST_LOG").is_some() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
      .try_init();
    // SPAN_GATES overrides the msshape config (default 2730 = c2^12).
    let span_gates: usize = std::env::var("SPAN_GATES")
      .ok()
      .and_then(|v| v.parse().ok())
      .unwrap_or(2730);
    let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new_wide(span_gates);
    let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
    let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
    let (proof, _) = SpartanSNARK::<E>::prove(&pk, circuit, prep, false).unwrap();
    proof.verify(&vk).unwrap();
  }

  for &n in configs {
    let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
    let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
    let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).unwrap();
    let (proof, _) = SpartanSNARK::<E>::prove(&pk, circuit, prep, true).unwrap();
    let proof_bytes = bincode::serialize(&proof).unwrap();
    println!(
      "PlainSpartan synthetic n=2^{}: proof_size={} bytes",
      (n as u64).ilog2(),
      proof_bytes.len()
    );
  }

  let mut g = c.benchmark_group("spartan_synthetic");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(10));

  for &n in configs {
    let tag = format!("n2^{}", (n as u64).ilog2());

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter(|| {
        let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
        let _ = SpartanSNARK::<E>::setup(circuit).unwrap();
      });
    });

    g.bench_function(format!("prep_prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          SpartanSNARK::<E>::setup(circuit).unwrap().0
        },
        |pk| {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          let _ = SpartanSNARK::<E>::prep_prove(&pk, circuit, true).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
          let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).unwrap();
          let (_proof, prep_back) =
            SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, true).unwrap();
          (pk, circuit, prep_back)
        },
        |(pk, circuit, prep)| {
          let _ = SpartanSNARK::<E>::prove(&pk, circuit, prep, true).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new(n);
          let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
          let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), true).unwrap();
          let (proof, _) = SpartanSNARK::<E>::prove(&pk, circuit, prep, true).unwrap();
          (vk, proof)
        },
        |(vk, proof)| {
          proof.verify(&vk).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }

  // MultiSwap-shaped *native baseline* sweep for
  // `imod_spartan_modp/.../msshape_cN`: the same R1CS dimensions
  // (gates → next-pow2 cons, 3·gates → vars) and full-width witness
  // values, but native field gates — it does NOT express the imod
  // side's mod-p (Tom-256 base field) statement, which natively would
  // need limb-decomposition gadgets. The ratio therefore reads as
  // "integer-machinery overhead vs the same shape natively". `wide`
  // keeps witness values full-width so the witness commit can't take
  // the small-scalar MSM fast path. Tags mirror the imod side.
  for &n in &[682usize, 2730, 10922, 43690, 174762] {
    let tag = format!("msshape_c{}", n.next_power_of_two().ilog2());
    // `is_small = false`: full-width witness values do NOT fit machine
    // words; claiming otherwise produces an invalid commitment via the
    // small-scalar MSM path and the proof fails verification.
    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new_wide(n);
          let (pk, _vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
          let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
          let (_proof, prep_back) =
            SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, false).unwrap();
          (pk, circuit, prep_back)
        },
        |(pk, circuit, prep)| {
          let _ = SpartanSNARK::<E>::prove(&pk, circuit, prep, false).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let circuit = SyntheticMulCircuit::<<E as Engine>::Scalar>::new_wide(n);
          let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).unwrap();
          let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
          let (proof, _) = SpartanSNARK::<E>::prove(&pk, circuit, prep, false).unwrap();
          (vk, proof)
        },
        |(vk, proof)| {
          proof.verify(&vk).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }
  g.finish();
}

criterion_group!(benches, spartan_synthetic_benches);
criterion_main!(benches);
