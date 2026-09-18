//! benches/imod_spartan.rs
//! Criterion benchmarks for IntModSpartan {setup, prove, verify} on synthetic
//! Integer-Mod-R1CS instances of varying size. Each constraint is one
//! independent modular multiplication `a · b = c + N · q`.
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan`
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ff::Field;
use limber::{
  imod_r1cs::{IntModR1CSShape, IntModR1CSWitness},
  imod_spartan::IntModSpartanSNARK,
  provider::T256HyraxEngine,
  traits::Engine,
};
use std::time::Duration;

type E = T256HyraxEngine;
type Sc = <E as Engine>::Scalar;

/// Synthetic shape: `num_cons` independent modular multiplications, each
/// using three fresh witness columns. Witness columns are laid out as
/// `w = [a_0, b_0, c_0, a_1, b_1, c_1, …]` padded to `num_vars`.
const SYNTH_MODULUS: u64 = 7;

/// The relation/circuit — built in the (untimed) bench setup closure.
fn make_shape(num_cons: usize, num_vars: usize) -> IntModR1CSShape<E> {
  assert!(
    3 * num_cons <= num_vars,
    "num_vars must hold 3·num_cons columns"
  );

  let one = Sc::ONE;
  let num_io = 0;

  let mut a_entries = Vec::with_capacity(num_cons);
  let mut b_entries = Vec::with_capacity(num_cons);
  let mut c_entries = Vec::with_capacity(num_cons);
  for i in 0..num_cons {
    a_entries.push((i, 3 * i, one));
    b_entries.push((i, 3 * i + 1, one));
    c_entries.push((i, 3 * i + 2, one));
  }

  let mods = vec![Sc::from(SYNTH_MODULUS); num_cons];

  IntModR1CSShape::<E>::from_entries(
    num_cons, num_vars, num_io, &a_entries, &b_entries, &c_entries, mods,
  )
  .unwrap()
}

/// Witness generation: the prover's per-proof work, timed alongside prove.
/// Picks small a, b so a·b fits in u64; chooses c, q so a·b = c + N·q.
fn synth_witness(num_cons: usize, num_vars: usize) -> (Vec<Sc>, Vec<Sc>) {
  let mut w = vec![Sc::ZERO; num_vars];
  let mut q = vec![Sc::ZERO; num_cons];
  for i in 0..num_cons {
    let a = (i as u64 % 100) + 1;
    let b = ((i as u64 * 7) % 100) + 1;
    let ab = a * b;
    w[3 * i] = Sc::from(a);
    w[3 * i + 1] = Sc::from(b);
    w[3 * i + 2] = Sc::from(ab % SYNTH_MODULUS);
    q[i] = Sc::from(ab / SYNTH_MODULUS);
  }
  (w, q)
}

fn make_shape_and_witness(
  num_cons: usize,
  num_vars: usize,
) -> (IntModR1CSShape<E>, Vec<Sc>, Vec<Sc>) {
  let shape = make_shape(num_cons, num_vars);
  let (w, q) = synth_witness(num_cons, num_vars);
  (shape, w, q)
}

fn imod_spartan_benches(c: &mut Criterion) {
  // (num_cons, num_vars) — num_vars = next power-of-two ≥ 4·num_cons.
  // Smallest config matches `benches/imod_spartan_modp.rs` for direct
  // p=q vs p≠q comparison; larger ones are the single-field prototype's
  // historical sizes.
  let configs: &[(usize, usize)] = &[
    (1usize << 6, 1usize << 8),
    (1usize << 10, 1usize << 12),
    (1usize << 12, 1usize << 14),
    (1usize << 14, 1usize << 16),
  ];

  // Report proof sizes once (outside measurements).
  for &(num_cons, num_vars) in configs {
    let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
    let (pk, _vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
    let (witness, instance) = IntModR1CSWitness::<E>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
    let proof = IntModSpartanSNARK::<E>::prove(&pk, &instance, &witness).unwrap();
    let proof_bytes = bincode::serialize(&proof).unwrap();
    println!(
      "IntModSpartan num_cons=2^{} num_vars=2^{}: proof_size={} bytes",
      num_cons.ilog2(),
      num_vars.ilog2(),
      proof_bytes.len()
    );
  }

  let mut g = c.benchmark_group("imod_spartan");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(10));

  for &(num_cons, num_vars) in configs {
    let tag = format!("c2^{}_v2^{}", num_cons.ilog2(), num_vars.ilog2());

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter_batched(
        || make_shape_and_witness(num_cons, num_vars).0,
        |shape| {
          let _ = IntModSpartanSNARK::<E>::setup(shape).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    // Timed region = the full prover pipeline: witness generation
    // (`synth_witness`) + witness commitment (`IntModR1CSWitness::new`) +
    // prove. The untimed setup closure holds only the circuit/shape and the
    // SNARK setup (PCS key derivation), which has its own `setup/` group.
    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let shape = make_shape(num_cons, num_vars);
          let (pk, _vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
          (pk, shape)
        },
        |(pk, shape)| {
          let (w, q) = synth_witness(num_cons, num_vars);
          let (witness, instance) =
            IntModR1CSWitness::<E>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let _ = IntModSpartanSNARK::<E>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
          let (pk, vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
          let (witness, instance) =
            IntModR1CSWitness::<E>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let proof = IntModSpartanSNARK::<E>::prove(&pk, &instance, &witness).unwrap();
          (vk, instance, proof)
        },
        |(vk, instance, proof)| {
          proof.verify(&vk, &instance).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }
  g.finish();
}

criterion_group!(benches, imod_spartan_benches);
criterion_main!(benches);
