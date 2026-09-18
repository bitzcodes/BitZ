//! benches/imod_spartan_modp.rs
//! Criterion benchmarks for IntModSpartanModpSNARK {setup, prove,
//! verify} on synthetic Integer-Mod-R1CS instances of varying size.
//!
//! Mirror of `benches/imod_spartan.rs` but parameterized over the dual-field
//! driver (T256DynPrimeEngine + IntegerModPCS). Each constraint is one
//! independent modular multiplication `a · b = c + N · q`.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan_modp
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ff::{Field, PrimeField};
use limber::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::{
    IntModSpartanModpProverKey, IntModSpartanModpSNARK, IntModSpartanModpVerifierKey,
  },
  provider::T256DynPrimeEngine,
  provider::pcs::integer_modpcs::{DEFAULT_K, DEFAULT_LOG_T_F, IntEvalParams},
  provider::pt256::t256,
};
use num_bigint::{BigUint, RandBigInt};
use rand::SeedableRng;
use std::time::Duration;
use tracing_subscriber::EnvFilter;

type M = T256DynPrimeEngine;

/// Limb bound (bits) for the MultiSwap-shaped config; matches the
/// MultiSwap bench's `LOG_T`.
const MSSHAPE_LOG_T: usize = 64;
/// Norm bound (bits) on committed values for the MultiSwap-shaped
/// config: operands are full T256-scalar-field elements (< 2^256).
const MSSHAPE_LOG_T_F: usize = 256;

/// The T256 **base-field** modulus `p` as a `BigUint`, computed from
/// `-Base::ONE`. This is a foreign 256-bit modulus from the proof
/// system's perspective (the native field is the scalar field `q ≠ p`),
/// so a gate `a·b ≡ c (mod p)` is NOT expressible as one native R1CS
/// constraint — it's the realistic workload class (foreign-field /
/// curve-coordinate arithmetic) that the integer Mod-PCS exists for.
fn t256_base_modulus() -> BigUint {
  let p_minus_1 = (-<t256::Base as Field>::ONE).to_repr();
  BigUint::from_bytes_le(p_minus_1.as_ref()) + 1u32
}

/// Setup honoring an optional `IMOD_K` env override for IntEval's
/// per-iteration variable count `k` (default `DEFAULT_K = 9`), so a
/// sweep can compare derived `(log_p, s, t)` trade-offs without code
/// edits.
fn setup_for(
  shape: IntModR1CSShapeModp<M>,
) -> (
  IntModSpartanModpProverKey<M>,
  IntModSpartanModpVerifierKey<M>,
) {
  match std::env::var("IMOD_K").ok().and_then(|s| s.parse().ok()) {
    Some(k) => {
      let n = shape.num_vars().max(shape.num_cons());
      let log_n = (n as u64).ilog2() as usize;
      let params = IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, k, log_n)
        .expect("IMOD_K params satisfy bounds");
      IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap()
    }
    None => IntModSpartanModpSNARK::<M>::setup(shape).unwrap(),
  }
}

/// MultiSwap-shaped synthetic config: `2730 = ⌊2^13 / 3⌋` random
/// multiplication gates `a·b ≡ c (mod p)` where `p` is the T256
/// **base-field** modulus — a foreign 256-bit modulus the native proof
/// system (scalar field `q ≠ p`) cannot handle in one constraint — and
/// `a, b` are uniform 256-bit values below `p`. Pads to the same
/// `(2^12 cons, 2^13 vars)` shape as MultiSwap k=0 (real RSA-2048 rows
/// pad identically), and exercises the same machinery —
/// `log_t_f = 256`, `log_t = 64` → `numlimb = 4` — at
/// 256-bit instead of 2048-bit width. `spartan_synthetic`'s `msshape`
/// config provides the shape-matched *native baseline* (same R1CS
/// dimensions, full-width values, native gates): it does NOT express
/// this statement — natively it would need limb-decomposition gadgets
/// (~10²-10³ constraints per gate) — so the ratio reads as "machinery
/// overhead vs what the same shape costs natively", not same-statement.
fn make_msshape_shape_and_operands(
  num_real: usize,
) -> (IntModR1CSShapeModp<M>, Vec<(BigUint, BigUint)>) {
  let num_cons = num_real.next_power_of_two();
  let num_vars = (3 * num_real).next_power_of_two();
  let n = t256_base_modulus();
  let mut rng = rand::rngs::StdRng::seed_from_u64(0x6d73_7368);

  let one = BigUint::from(1u32);
  let num_io = 0;
  let mut a_entries = Vec::with_capacity(num_real);
  let mut b_entries = Vec::with_capacity(num_real);
  let mut c_entries = Vec::with_capacity(num_real);
  for i in 0..num_real {
    a_entries.push((i, 3 * i, one.clone()));
    b_entries.push((i, 3 * i + 1, one.clone()));
    c_entries.push((i, 3 * i + 2, one.clone()));
  }
  // Padding rows (i ≥ num_real) are all-zero: 0·0 = 0 + q·0.
  let mods = vec![n.clone(); num_cons];

  let shape = IntModR1CSShapeModp::<M>::new(
    num_cons, num_vars, num_io, a_entries, b_entries, c_entries, mods,
  )
  .unwrap();

  let operands: Vec<(BigUint, BigUint)> = (0..num_real)
    .map(|_| (rng.gen_biguint_below(&n), rng.gen_biguint_below(&n)))
    .collect();
  (shape, operands)
}

/// Witness generation for the msshape config: the per-gate divmods
/// producing `c = a·b mod p` and the quotient advice `q = a·b div p`.
/// Counted inside the timed prove region (the plain-Spartan baseline's
/// witness synthesis is likewise inside its `prove`).
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

/// Convenience wrapper (one-shot + verify-bench setup paths).
fn make_msshape_shape_and_witness(
  num_real: usize,
) -> (IntModR1CSShapeModp<M>, Vec<BigUint>, Vec<BigUint>) {
  let (shape, operands) = make_msshape_shape_and_operands(num_real);
  let (w, q) = msshape_witness(&shape, &operands);
  (shape, w, q)
}

/// Gate counts for the msshape size sweep: `⌊2^v / 3⌋` gates fill
/// `(2^(v−1) cons, 2^v vars)` with the 3-fresh-columns layout, mirroring
/// MultiSwap k=0's padding (2715 real rows → 2^12/2^13). The middle
/// entry is the MultiSwap k=0 shape itself.
const MSSHAPE_GATES: &[usize] = &[682, 2730, 10922, 43690, 174762];

/// Setup for the MultiSwap-shaped config: `log_t_f = 256` (limb-split
/// into 4 limbs of 64 bits), honoring the `IMOD_K` override.
fn setup_msshape(
  shape: IntModR1CSShapeModp<M>,
) -> (
  IntModSpartanModpProverKey<M>,
  IntModSpartanModpVerifierKey<M>,
) {
  let k = std::env::var("IMOD_K")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(DEFAULT_K);
  let n = shape.num_vars().max(shape.num_cons());
  let log_n = (n as u64).ilog2() as usize;
  let params = IntEvalParams::derive(MSSHAPE_LOG_T_F, MSSHAPE_LOG_T, k, log_n)
    .expect("msshape params satisfy bounds");
  IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap()
}

/// Synthetic modulus shared by `make_shape` and `synth_witness`.
const SYNTH_MODULUS: u64 = 7;

/// Synthetic shape: `num_cons` independent modular multiplications, each
/// using three fresh witness columns. Witness columns are laid out as
/// `w = [a_0, b_0, c_0, a_1, b_1, c_1, …]` padded to `num_vars`. This is
/// the relation/circuit — built in the (untimed) bench setup closure, not
/// part of the prover's per-proof work.
fn make_shape(num_cons: usize, num_vars: usize) -> IntModR1CSShapeModp<M> {
  assert!(
    3 * num_cons <= num_vars,
    "num_vars must hold 3·num_cons columns"
  );

  let one = BigUint::from(1u32);
  let num_io = 0;

  let mut a_entries = Vec::with_capacity(num_cons);
  let mut b_entries = Vec::with_capacity(num_cons);
  let mut c_entries = Vec::with_capacity(num_cons);
  for i in 0..num_cons {
    a_entries.push((i, 3 * i, one.clone()));
    b_entries.push((i, 3 * i + 1, one.clone()));
    c_entries.push((i, 3 * i + 2, one.clone()));
  }

  let mods = vec![BigUint::from(SYNTH_MODULUS); num_cons];

  IntModR1CSShapeModp::<M>::new(
    num_cons, num_vars, num_io, a_entries, b_entries, c_entries, mods,
  )
  .unwrap()
}

/// Witness generation: the prover's per-proof work, so it belongs in the
/// timed region. Picks small a, b so a·b fits in u64; chooses c, q so
/// a·b = c + N·q under `SYNTH_MODULUS`.
fn synth_witness(num_cons: usize, num_vars: usize) -> (Vec<BigUint>, Vec<BigUint>) {
  let zero = BigUint::from(0u32);
  let mut w = vec![zero.clone(); num_vars];
  let mut q = vec![zero; num_cons];
  for i in 0..num_cons {
    let a = (i as u64 % 100) + 1;
    let b = ((i as u64 * 7) % 100) + 1;
    let ab = a * b;
    w[3 * i] = BigUint::from(a);
    w[3 * i + 1] = BigUint::from(b);
    w[3 * i + 2] = BigUint::from(ab % SYNTH_MODULUS);
    q[i] = BigUint::from(ab / SYNTH_MODULUS);
  }
  (w, q)
}

fn make_shape_and_witness(
  num_cons: usize,
  num_vars: usize,
) -> (IntModR1CSShapeModp<M>, Vec<BigUint>, Vec<BigUint>) {
  let shape = make_shape(num_cons, num_vars);
  let (w, q) = synth_witness(num_cons, num_vars);
  (shape, w, q)
}

fn imod_spartan_modp_benches(c: &mut Criterion) {
  // KSWEEP=1: msshape size × k × log_t sweep. Times (witness-commit + prove)
  // per config with Instant, verifies each (soundness gate), prints a table,
  // and returns (skips criterion). Maps the optimal (k, T) per circuit size.
  if std::env::var_os("KSWEEP").is_some() {
    use std::time::Instant;
    for &gates in &[682usize, 1365, 2730, 5461] {
      let (shape, w, q) = make_msshape_shape_and_witness(gates);
      let n = shape.num_vars().max(shape.num_cons());
      let log_n = (n as u64).ilog2() as usize;
      println!(
        "\nmsshape gates={gates} (cons=2^{}, vars=2^{}):",
        (shape.num_cons() as u64).ilog2(),
        (shape.num_vars() as u64).ilog2()
      );
      for &log_t in &[16usize, 32, 64] {
        for k in [7usize, 8, 9, 10, 11, 12] {
          let Ok(params) = IntEvalParams::derive(MSSHAPE_LOG_T_F, log_t, k, log_n) else {
            continue;
          };
          let (sval, lpval, nl) = (params.s, params.log_p, params.numlimb);
          let (pk, vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
          let t0 = Instant::now();
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w.clone(), q.clone(), vec![]).unwrap();
          let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
          let ms = t0.elapsed().as_secs_f64() * 1e3;
          proof.verify(&vk, &instance).unwrap();
          println!(
            "  T=2^{log_t:<2} k={k:<2} (s={sval:<2} log_p={lpval:<2} numlimb={nl:<2}): \
             commit+prove {ms:8.1} ms"
          );
        }
      }
    }
    return;
  }

  // PSIZE=1: full-impl proof size on the msshape configs (the §7.2
  // comparison vs plain Spartan). The proof isn't fully `Serialize`
  // (dynamic-prime sumcheck side), so report `eval_arg_bytes` — the
  // dominant Mod-PCS batch argument (per-poly commits, LogUp-GKR range
  // check, combined open) — plus the analytical sumcheck remainder.
  // Deterministic: one prove each, then return (skip criterion).
  // Densified to c10..c14 (adds c11, c13) to expand the size curve.
  if std::env::var_os("PSIZE").is_some() {
    let k = std::env::var("IMOD_K")
      .ok()
      .and_then(|s| s.parse::<usize>().ok())
      .unwrap_or(DEFAULT_K);
    println!(
      "\n§7.2 full-impl proof size (log_t_f={MSSHAPE_LOG_T_F}, log_t={MSSHAPE_LOG_T}, k={k}):"
    );
    for &gates in &[682usize, 1365, 2730, 5461, 10922] {
      let (shape, w, q) = make_msshape_shape_and_witness(gates);
      let lc = (shape.num_cons() as u64).ilog2();
      let lv = (shape.num_vars() as u64).ilog2();
      let (pk, vk) = setup_msshape(shape.clone());
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      proof.verify(&vk, &instance).unwrap();
      let arg = proof.eval_arg_bytes().expect("eval_arg serializes").len();
      let (pp, rc, co) = proof.eval_arg_component_sizes();
      // Dynamic-prime sumcheck side: `lc` cubic outer rounds (3 coeffs)
      // + `lv` quadratic inner rounds (2 coeffs) + 6 claimed evals, at
      // 16 B per 2-limb (128-bit) scalar.
      let dyn_bytes = (lc as usize * 3 + lv as usize * 2 + 6) * 16;
      let total = arg + dyn_bytes;
      println!(
        "  c2^{lc} v2^{lv}: proof≈{:>6.1} KB  [per_poly {pp}, range_check {rc} ({:.0}%), \
         combined_open {co}, sumcheck {dyn_bytes}]",
        total as f64 / 1e3,
        100.0 * rc as f64 / total as f64,
      );
    }
    return;
  }

  // (num_cons, num_vars) — num_vars is the next power-of-two ≥ 4·num_cons.
  // The Mod-PCS open point has length log_2(num_vars); for num_vars > 2^k
  // (default k = 7) this exercises IntEval's partial-eval iteration path.
  let configs: &[(usize, usize)] = &[
    (1usize << 6, 1usize << 8),   // point.len=8 → t=1 IntEval iteration
    (1usize << 8, 1usize << 10),  // point.len=10 → t=1
    (1usize << 10, 1usize << 12), // point.len=12 → t=1
    (1usize << 12, 1usize << 14), // point.len=14 → t=1
    (1usize << 14, 1usize << 16), // point.len=16 → t=1
  ];

  // Per-part timing breakdown, gated entirely on `RUST_LOG` so a plain
  // `cargo bench` installs no global subscriber and runs no extra work.
  // With `RUST_LOG=info` we install a fmt subscriber and run one
  // setup/prove/verify per config, printing the section spans
  // (imod_pcs_chain_openings, imod_pcs_rc_ab, …) so you can see where
  // prove/verify time goes without criterion's iteration noise.
  // PSIZE=1: serialized eval-argument size per msshape config (the
  // dominant proof component; the dynamic-prime sumcheck side adds
  // ~1 KB). Mirrors the multiswap bench's PSIZE block.
  if std::env::var_os("PSIZE").is_some() {
    for &gates in MSSHAPE_GATES {
      let (shape, w, q) = make_msshape_shape_and_witness(gates);
      let (pk, vk) = setup_msshape(shape.clone());
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      proof.verify(&vk, &instance).unwrap();
      println!(
        "msshape c2^{} proof size: eval_arg {} bytes",
        (shape.num_cons() as u64).ilog2(),
        proof.eval_arg_bytes().expect("eval_arg serializes").len()
      );
    }
    return;
  }

  if std::env::var_os("RUST_LOG").is_some() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();
    // Span-timing sweep over the large msshape gates (c14/c16/c18) to
    // localize the superlinear prover component.
    for &gates in &[10922usize, 43690, 174762] {
      let (shape, w, q) = make_msshape_shape_and_witness(gates);
      println!(
        "\n===== msshape c2^{} v2^{} (gates={gates}) =====",
        (shape.num_cons() as u64).ilog2(),
        (shape.num_vars() as u64).ilog2()
      );
      let (pk, vk) = setup_msshape(shape.clone());
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      proof.verify(&vk, &instance).unwrap();
    }
  }

  let mut g = c.benchmark_group("imod_spartan_modp");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(20));

  for &(num_cons, num_vars) in configs {
    let tag = format!("c2^{}_v2^{}", num_cons.ilog2(), num_vars.ilog2());

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter_batched(
        || make_shape_and_witness(num_cons, num_vars).0,
        |shape| {
          let _ = IntModSpartanModpSNARK::<M>::setup(shape).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    // Timed region = the full prover pipeline: witness generation
    // (`synth_witness`) + witness commitment (`IntModR1CSWitnessModp::new`
    // commits w and q) + prove. The untimed setup closure holds only the
    // circuit/shape and the SNARK setup (PCS key derivation) — which is
    // reusable, verifier-shared work and also has its own `setup/` group.
    // The plain-Spartan baseline likewise synthesizes + commits its
    // witness inside the timed prove.
    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let shape = make_shape(num_cons, num_vars);
          let (pk, _vk) = setup_for(shape.clone());
          (pk, shape)
        },
        |(pk, shape)| {
          let (w, q) = synth_witness(num_cons, num_vars);
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let _ = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_shape_and_witness(num_cons, num_vars);
          let (pk, vk) = setup_for(shape.clone());
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
          (vk, instance, proof)
        },
        |(vk, instance, proof)| {
          proof.verify(&vk, &instance).unwrap();
        },
        BatchSize::LargeInput,
      );
    });
  }

  // MultiSwap-shaped size sweep: full-width gates mod the T256 (Tom-256)
  // base-field modulus, foreign to the native scalar field. Shape-matched
  // against plain Spartan's `spartan_synthetic/.../msshape_cN` native
  // baselines. Tag is `msshape_c{log2 cons}`.
  for &gates in MSSHAPE_GATES {
    let lc = gates.next_power_of_two().ilog2();
    let tag = format!("msshape_c{lc}");
    // Timed region = witness generation (per-gate divmods producing c
    // and the quotient advice) + witness commitment + prove, matching
    // the plain-Spartan baseline whose prove synthesizes and commits
    // its witness internally.
    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, operands) = make_msshape_shape_and_operands(gates);
          let (pk, _vk) = setup_msshape(shape.clone());
          (pk, shape, operands)
        },
        |(pk, shape, operands)| {
          let (w, q) = msshape_witness(&shape, &operands);
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let _ = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q) = make_msshape_shape_and_witness(gates);
          let (pk, vk) = setup_msshape(shape.clone());
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
          let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
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

criterion_group!(benches, imod_spartan_modp_benches);
criterion_main!(benches);
