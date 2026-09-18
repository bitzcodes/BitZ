//! examples/spartan_part_timing.rs — DISPOSABLE local instrumentation
//! (not upstream). Per-phase timing of the Limber prover on the msshape
//! configs, isolating the Spartan-style Z_p sumcheck part
//! (sample_p/reduce/spmv/outer_sumcheck/inner_setup/inner_sumcheck/
//! eval_recover) from the Mod-PCS part (wq_commit + wq_open), via the
//! library's existing `start_span!`/`info!(elapsed_ms=..)` events.
//!
//! Run:
//!   RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" RUST_LOG=info \
//!     cargo run --release --example spartan_part_timing
//!
//! Markers `###REP gates=<g> rep=<i>` / `###VERIFY` delimit each timed
//! commit+prove region in the log stream for post-hoc parsing.

use ff::Field;
use limber::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::{
    IntModSpartanModpProverKey, IntModSpartanModpSNARK, IntModSpartanModpVerifierKey,
  },
  provider::T256DynPrimeEngine,
  provider::pcs::integer_modpcs::{DEFAULT_K, IntEvalParams},
  provider::pt256::t256,
};
use num_bigint::{BigUint, RandBigInt};
use rand::SeedableRng;
use tracing_subscriber::EnvFilter;

type M = T256DynPrimeEngine;

const MSSHAPE_LOG_T: usize = 64;
const MSSHAPE_LOG_T_F: usize = 256;

fn t256_base_modulus() -> BigUint {
  use ff::PrimeField;
  let p_minus_1 = (-<t256::Base as Field>::ONE).to_repr();
  BigUint::from_bytes_le(p_minus_1.as_ref()) + 1u32
}

/// Mirror of the bench's msshape circuit: `num_real` gates
/// `a·b ≡ c (mod p_base)` with 3 fresh columns per gate.
fn make_msshape_shape_and_operands(
  num_real: usize,
) -> (IntModR1CSShapeModp<M>, Vec<(BigUint, BigUint)>) {
  let num_cons = num_real.next_power_of_two();
  let num_vars = (3 * num_real).next_power_of_two();
  let n = t256_base_modulus();
  let mut rng = rand::rngs::StdRng::seed_from_u64(0x6d73_7368);

  let one = BigUint::from(1u32);
  let mut a_entries = Vec::with_capacity(num_real);
  let mut b_entries = Vec::with_capacity(num_real);
  let mut c_entries = Vec::with_capacity(num_real);
  for i in 0..num_real {
    a_entries.push((i, 3 * i, one.clone()));
    b_entries.push((i, 3 * i + 1, one.clone()));
    c_entries.push((i, 3 * i + 2, one.clone()));
  }
  let mods = vec![n.clone(); num_cons];
  let shape =
    IntModR1CSShapeModp::<M>::new(num_cons, num_vars, 0, a_entries, b_entries, c_entries, mods)
      .unwrap();
  let operands: Vec<(BigUint, BigUint)> = (0..num_real)
    .map(|_| (rng.gen_biguint_below(&n), rng.gen_biguint_below(&n)))
    .collect();
  (shape, operands)
}

fn msshape_witness(
  shape: &IntModR1CSShapeModp<M>,
  operands: &[(BigUint, BigUint)],
) -> (Vec<BigUint>, Vec<BigUint>) {
  let n = t256_base_modulus();
  let zero = BigUint::from(0u32);
  let mut w = vec![zero.clone(); shape.num_vars()];
  let mut q = vec![zero; shape.num_cons()];
  for (i, (a, b)) in operands.iter().enumerate() {
    let ab = a * b;
    q[i] = &ab / &n;
    w[3 * i] = a.clone();
    w[3 * i + 1] = b.clone();
    w[3 * i + 2] = &ab % &n;
  }
  (w, q)
}

fn setup_msshape(
  shape: IntModR1CSShapeModp<M>,
) -> (
  IntModSpartanModpProverKey<M>,
  IntModSpartanModpVerifierKey<M>,
) {
  let n = shape.num_vars().max(shape.num_cons());
  let log_n = (n as u64).ilog2() as usize;
  let params = IntEvalParams::derive(MSSHAPE_LOG_T_F, MSSHAPE_LOG_T, DEFAULT_K, log_n)
    .expect("msshape params satisfy bounds");
  IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap()
}

fn main() {
  let _ = tracing_subscriber::fmt()
    .with_target(false)
    .without_time()
    .with_env_filter(EnvFilter::from_default_env())
    .try_init();

  // (gates, reps): c10..c16 msshape shapes; middle entry (2730) is the
  // MultiSwap-k0-shaped one.
  let configs: &[(usize, usize)] = &[(682, 5), (2730, 5), (10922, 5), (43690, 3)];

  for &(gates, reps) in configs {
    let (shape, operands) = make_msshape_shape_and_operands(gates);
    println!(
      "###CONFIG gates={gates} cons=2^{} vars=2^{}",
      (shape.num_cons() as u64).ilog2(),
      (shape.num_vars() as u64).ilog2()
    );
    let (pk, vk) = setup_msshape(shape.clone());
    for rep in 0..reps {
      println!("###REP gates={gates} rep={rep}");
      let (w, q) = msshape_witness(&shape, &operands);
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      if rep == reps - 1 {
        println!("###VERIFY gates={gates}");
        proof.verify(&vk, &instance).unwrap();
        println!("###VERIFY_OK gates={gates}");
      }
    }
  }
  println!("###DONE");
}
