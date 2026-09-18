//! benches/multiswap_modp.rs
//! Prover/verifier cost of proving **MultiSwap** (Ozdemir, Wahby,
//! Whitehat, Boneh, *Scaling Verifiable Computation Using Efficient Set
//! Accumulators*, USENIX Security 2020, §3–4) with
//! `IntModSpartanModpSNARK`.
//!
//! MultiSwap verifies a batch of `k` swaps against an RSA accumulator by
//! checking two Wesolowski proofs (one batch insertion, one batch
//! removal) that share a Fiat-Shamir prime challenge `ℓ`:
//!   Q_ins^ℓ · ⟦S⟧^(∏ H∆(yᵢ) mod ℓ) = ⟦S'⟧   in  G = (Z/N)*/{±1}
//! and symmetrically for removal. Its cost (paper Fig. 3) is dominated by
//! multiprecision modular arithmetic: 4 group exponentiations with
//! `|ℓ|≈352`-bit exponents mod a `b_N≈2048`-bit modulus `N`, 2 group
//! mults, the hash-to-prime `Hp`, and per-swap `∏ H∆ mod ℓ`.
//!
//! The IntMod-R1CS relation `A·z ∘ B·z = C·z + m∘q` over Z has one
//! **per-row modulus** `mᵢ` and a prover quotient `qᵢ` — so one row is one
//! modular multiply `LC_A·LC_B ≡ LC_C (mod mᵢ)`. A `mod N` multiply that
//! costs ~7044·(2048/352) R1CS constraints in the paper's xJsnark/F_p
//! representation is a single imod row here. This bench measures exactly
//! that collapse.
//!
//! Fidelity: the 4 group
//! exponentiations are **real wired square-and-multiply chains** with
//! witness exponents, bit decomposition, and reconstruction constraints.
//! The bases are fixed constants baked into matrix coefficients (avoiding
//! a degree-3 conditional-multiply decomposition). The hashes (`H`, `Hp`,
//! `H∆`) and RSA group structure are *modeled by operation count*, not
//! faithful crypto circuits, and are flagged as such. As of the
//! faithful-cost extension, `Hp` is charged at faithful cost and
//! structure: 600 Pocklington-exponentiation rows + 3 chained
//! Poseidon-cost permutations (243 mul rows each, synthetic operands,
//! real chain shape and modulus) + 640 decomposition bit rows + 10
//! reconstruction rows — ~2.0k rows total vs the paper's 217,703 F_p
//! constraints for the same component.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use limber::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::IntModSpartanModpSNARK,
  provider::{
    T256DynPrimeEngine,
    pcs::integer_modpcs::{DEFAULT_K, IntEvalParams, LAMBDA_BOUND2},
  },
};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::Zero;
use serde_json::{Value, json};
use std::{
  collections::HashMap,
  fs::{self, File, OpenOptions},
  io::{BufWriter, Write},
  path::Path,
  process::Command,
  time::{Duration, Instant},
};
use tracing_subscriber::EnvFilter;

type M = T256DynPrimeEngine;

/// Limb bound (bits) for the IntEval range checks.
const LOG_T: usize = 64;

/// Base-hash model: imod rows charged per `H` invocation.
const H_ROWS: usize = 8;

/// Hash-to-prime (`Hp`) Pocklington certificate: number of wired
/// square-and-multiply chains and exponent bits per chain. 4 chains of
/// 50 bits — 4·(3·50+1) = 604 rows — matching the ~600-row operation
/// count of the earlier model, but as REAL wired chains (bit
/// decomposition, reconstruction, chained accumulators) over the
/// Mersenne-prime moduli 2^61−1, 2^89−1, 2^107−1, 2^127−1.
const HP_EXPS: usize = 4;
const HP_EXP_BITS: usize = 50;

/// Faithful-cost Poseidon permutation: 81 x^5 S-boxes × 3 mul rows
/// (x², x⁴, x⁵); the MDS and round-constant layers are linear and fold
/// into the LCs for free. Operands are synthetic but the operation
/// count, chaining structure, and modulus are faithful.
const POSEIDON_ROWS_PER_PERM: usize = 243;
/// Poseidon permutations charged inside one `Hp` invocation
/// (candidate generation for the Pocklington chain).
const HP_POSEIDON_PERMS: usize = 3;
/// Bit rows for the Pocklington side-condition decompositions: the
/// Poseidon output (255 bits) and the four chain outputs (61 + 89 +
/// 107 + 127 bits) are fully bit-decomposed — 639 exact mod-0 binary
/// rows WIRED to their values by the reconstruction rows below.
const HP_DECOMP_BITS: usize = 639;
/// One exact (mod-0) reconstruction row per decomposed value.
const HP_DECOMP_RECON: usize = 5;

/// Number of group exponentiations per MultiSwap proof.
const N_GROUP_EXPS: usize = 4;
/// Group mults per MultiSwap proof.
const N_GROUP_MULS: usize = 2;

/// Exponent bit-length for the Fiat-Shamir prime challenge `ℓ`.
const ELL_BITS: usize = 352;

#[derive(Clone, Copy)]
struct Dims {
  /// Faithful-cost hash extension: chained Poseidon-cost rows,
  /// decomposition bit rows, and reconstruction rows for `Hp`.
  poseidon_rows: usize,
  decomp_bits: usize,
  decomp_recon: usize,
  k: usize,
  ell_bits: usize,
  n_group_exps: usize,
  n_group_muls: usize,
  hp_exps: usize,
  hp_exp_bits: usize,
  h_rows: usize,
}

impl Dims {
  fn multiswap(k: usize) -> Self {
    Self {
      k,
      poseidon_rows: HP_POSEIDON_PERMS * POSEIDON_ROWS_PER_PERM,
      decomp_bits: HP_DECOMP_BITS,
      decomp_recon: HP_DECOMP_RECON,
      ell_bits: ELL_BITS,
      n_group_exps: N_GROUP_EXPS,
      n_group_muls: N_GROUP_MULS,
      hp_exps: HP_EXPS,
      hp_exp_bits: HP_EXP_BITS,
      h_rows: H_ROWS,
    }
  }

  /// Rows of one wired Hp certificate chain.
  fn rows_per_hp_exp(&self) -> usize {
    3 * self.hp_exp_bits + 1
  }

  fn rows_per_exp(&self) -> usize {
    3 * self.ell_bits + 1
  }

  fn cols_per_exp(&self) -> usize {
    3 * self.ell_bits + 1
  }

  /// Unwired generic rows remaining: only the per-swap `H∆` models
  /// (k > 0). Everything at k = 0 is wired.
  fn generic_rows(&self) -> usize {
    2 * self.k + 2 * self.k * self.h_rows
  }

  /// Wired hash/Hp rows: group mults (operands = exp outputs, 1 fresh
  /// result column each), 4 Hp certificate chains, the Poseidon seed
  /// reduction row, the chained Poseidon rows, the decomposition bit
  /// rows, their reconstruction rows (0 fresh columns), and the final
  /// mod-ℓ reduction row wired to the Poseidon output.
  fn wired_ext_rows(&self) -> usize {
    self.n_group_muls
      + self.hp_exps * self.rows_per_hp_exp()
      + 1
      + self.poseidon_rows
      + self.decomp_bits
      + self.decomp_recon
      + 1
  }

  fn wired_ext_cols(&self) -> usize {
    self.n_group_muls
      + self.hp_exps * self.rows_per_hp_exp()
      + 1
      + self.poseidon_rows
      + self.decomp_bits
      + 1
  }

  fn non_exp_rows(&self) -> usize {
    self.generic_rows() + self.wired_ext_rows()
  }

  fn num_real_rows(&self) -> usize {
    self.n_group_exps * self.rows_per_exp() + self.non_exp_rows()
  }

  fn num_real_cols(&self) -> usize {
    self.n_group_exps * self.cols_per_exp() + 3 * self.generic_rows() + self.wired_ext_cols()
  }
}

fn modulus_n() -> BigUint {
  let hex = "c7970ceedcc3b0754490201a7aa613cd73911081c790f5f1a8726f463550bb5b\
             7ff0db8e1ea1189ec72f93d1650011bd721aeeacc2acde32a04107f0648c2813\
             a31f5b0b7765ff8b44b4b6ffc93384b646eb09c7cf5e8592d40ea33c80039f35\
             b4f14a04b51f7bfd781be4d1673164ba8eb991c2c4d730bbbe35f592bdef524a\
             f7e8daefd26c66fc02c479af89d64d373f442709439de66ceb955f3ea37d5159\
             f6135809f85334b5cb1813addc80cd05609f10ac6a95ad65872c909525bdad32\
             bc729592642920f24c61dc5b3c3b7923e56b16a4d9d373d8721f24a3fc0f1b31\
             31f55615172866bccc30f95054c824e733a5eb6817f7bc16399d48c6361cc7e5";
  BigUint::parse_bytes(hex.as_bytes(), 16).expect("valid RSA-2048 hex")
}

fn modulus_ell() -> BigUint {
  BigUint::from_bytes_be(&[0xc3u8; 44])
}

fn modulus_p_hash() -> BigUint {
  let hex = "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
  BigUint::parse_bytes(hex.as_bytes(), 16).expect("valid BLS12-381 scalar hex")
}

/// Moduli for the wired Hp certificate chains: Mersenne primes
/// 2^61−1, 2^89−1, 2^107−1, 2^127−1 (clean fixed constants standing in
/// for a Pocklington prime chain of growing widths).
fn hp_moduli() -> [BigUint; 4] {
  [
    (BigUint::from(1u32) << 61) - 1u32,
    (BigUint::from(1u32) << 89) - 1u32,
    (BigUint::from(1u32) << 107) - 1u32,
    (BigUint::from(1u32) << 127) - 1u32,
  ]
}

/// Synthetic (base, exponent) pairs for the Hp chains — deterministic,
/// bounded by each chain modulus / the chain bit width.
fn hp_chain_inputs(bits: usize) -> [(BigUint, BigUint); 4] {
  let ms = hp_moduli();
  core::array::from_fn(|i| {
    let base = &ms[i] - BigUint::from(1000u32 + 37 * i as u32);
    let exponent = (BigUint::from(0x9e37_79b9_7f4a_7c15u64) >> (64 - bits)) ^ BigUint::from(i);
    (base, exponent)
  })
}

fn exp_bases() -> [BigUint; 4] {
  let n = modulus_n();
  core::array::from_fn(|i| &n - BigUint::from(37u64 * i as u64 + 3))
}

fn exp_exponents(ell_bits: usize) -> [BigUint; 4] {
  core::array::from_fn(|i| {
    let seed = (i as u64 + 1) * 0x0123_4567_89AB_CDEFu64;
    let mut bytes = vec![0u8; ell_bits.div_ceil(8)];
    for (k, b) in bytes.iter_mut().enumerate() {
      *b = ((seed.wrapping_mul(k as u64 + 1).wrapping_add(0xDEAD)) & 0xFF) as u8;
    }
    if !ell_bits.is_multiple_of(8) {
      bytes[0] &= (1u8 << (ell_bits % 8)) - 1;
    }
    let msb_byte = (ell_bits - 1) / 8;
    let msb_bit = (ell_bits - 1) % 8;
    let msb_idx = bytes.len() - 1 - msb_byte;
    bytes[msb_idx] |= 1u8 << msb_bit;
    BigUint::from_bytes_be(&bytes)
  })
}

#[allow(clippy::too_many_arguments)]
fn build_exp_circuit(
  base: &BigUint,
  exponent: &BigUint,
  n: &BigUint,
  ell_bits: usize,
  row_base: usize,
  col_base: usize,
  const_col: usize,
  a_entries: &mut Vec<(usize, usize, BigUint)>,
  b_entries: &mut Vec<(usize, usize, BigUint)>,
  c_entries: &mut Vec<(usize, usize, BigUint)>,
  mods: &mut Vec<BigUint>,
  w: &mut [BigUint],
  q: &mut [BigUint],
) -> usize {
  let one = BigUint::from(1u32);
  let g_minus_1 = base - &one;

  let bit_col = |j: usize| col_base + j;
  let exp_col = col_base + ell_bits;
  let acc_col = |j: usize| col_base + ell_bits + 1 + j;
  let sq_col = |j: usize| col_base + 2 * ell_bits + 1 + j;

  let bits: Vec<u8> = (0..ell_bits)
    .map(|j| {
      let bit_pos = ell_bits - 1 - j;
      u8::from(exponent.bit(bit_pos as u64))
    })
    .collect();

  for j in 0..ell_bits {
    w[bit_col(j)] = BigUint::from(bits[j]);
  }
  w[exp_col] = exponent.clone();

  let mut row = row_base;
  for j in 0..ell_bits {
    let acc_val = if j == 0 {
      one.clone()
    } else {
      w[acc_col(j - 1)].clone()
    };

    // Square row
    let sq_prod = &acc_val * &acc_val;
    let (sq_q, sq_val) = sq_prod.div_rem(n);
    w[sq_col(j)] = sq_val.clone();
    q[row] = sq_q;

    let acc_j_col = if j == 0 { const_col } else { acc_col(j - 1) };
    a_entries.push((row, acc_j_col, one.clone()));
    b_entries.push((row, acc_j_col, one.clone()));
    c_entries.push((row, sq_col(j), one.clone()));
    mods.push(n.clone());
    row += 1;

    // Conditional-multiply row
    let b_val = BigUint::from(bits[j]) * &g_minus_1 + &one;
    let cm_prod = &sq_val * &b_val;
    let (cm_q, acc_next) = cm_prod.div_rem(n);
    w[acc_col(j)] = acc_next;
    q[row] = cm_q;

    a_entries.push((row, sq_col(j), one.clone()));
    b_entries.push((row, bit_col(j), g_minus_1.clone()));
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, acc_col(j), one.clone()));
    mods.push(n.clone());
    row += 1;
  }

  // Binary constraints, as EXACT integer rows (modulus 0 ⇒ the m·q term
  // vanishes, so the row enforces b² = b over ℤ, i.e. b ∈ {0,1},
  // unconditionally). Modulus N is also computationally sound here
  // (non-binary solutions within the range bound are benign lifts of
  // 0/1 or nontrivial idempotents of Z_N, and exhibiting the latter
  // factors N) — but mod-0 is assumption-free, costs the same, and stays
  // sound if the pattern is reused for moduli with known factorization
  // (e.g. mod-ℓ exponent bits in a future Hp gadget).
  for j in 0..ell_bits {
    a_entries.push((row, bit_col(j), one.clone()));
    b_entries.push((row, bit_col(j), one.clone()));
    c_entries.push((row, bit_col(j), one.clone()));
    q[row] = BigUint::from(0u32);
    mods.push(BigUint::from(0u32));
    row += 1;
  }

  // Reconstruction
  for j in 0..ell_bits {
    let power = BigUint::from(1u32) << (ell_bits - 1 - j);
    a_entries.push((row, bit_col(j), power));
  }
  b_entries.push((row, const_col, one.clone()));
  c_entries.push((row, exp_col, one.clone()));
  q[row] = BigUint::from(0u32);
  mods.push(n.clone());
  row += 1;

  let expected = base.modpow(exponent, n);
  assert_eq!(
    w[acc_col(ell_bits - 1)],
    expected,
    "exponentiation circuit witness mismatch"
  );

  row - row_base
}

#[allow(clippy::too_many_arguments)]
fn compute_witness_advice(
  bases: &[BigUint; 4],
  exponents: &[BigUint; 4],
  n: &BigUint,
  ell: &BigUint,
  p_hash: &BigUint,
  d: Dims,
) -> Vec<BigUint> {
  let one = BigUint::from(1u32);
  let mut out = Vec::new();

  let mut exp_outs = Vec::with_capacity(d.n_group_exps);
  for i in 0..d.n_group_exps {
    let g_minus_1 = &bases[i] - &one;
    let mut acc = one.clone();
    for j in 0..d.ell_bits {
      let bit_pos = d.ell_bits - 1 - j;
      let bit = u8::from(exponents[i].bit(bit_pos as u64));
      let sq = (&acc * &acc).div_rem(n).1;
      let b_val = BigUint::from(bit) * &g_minus_1 + &one;
      acc = (&sq * &b_val).div_rem(n).1;
    }
    exp_outs.push(acc.clone());
    out.push(acc);
  }

  // Wired group mults from the exponentiation outputs.
  for i in 0..d.n_group_muls {
    out.push((&exp_outs[2 * i] * &exp_outs[2 * i + 1]).div_rem(n).1);
  }

  // Wired Hp certificate chains (square-and-multiply mod the Mersenne
  // moduli).
  let hp_ms = hp_moduli();
  let hp_inputs = hp_chain_inputs(d.hp_exp_bits);
  for i in 0..d.hp_exps {
    let g_minus_1 = &hp_inputs[i].0 - &one;
    let mut acc = one.clone();
    for j in 0..d.hp_exp_bits {
      let bit_pos = d.hp_exp_bits - 1 - j;
      let bit = u8::from(hp_inputs[i].1.bit(bit_pos as u64));
      let sq = (&acc * &acc).div_rem(&hp_ms[i]).1;
      let b_val = BigUint::from(bit) * &g_minus_1 + &one;
      acc = (&sq * &b_val).div_rem(&hp_ms[i]).1;
    }
    out.push(acc);
  }

  // Poseidon seeded from the first exponentiation output, then the
  // chained S-box values; final mod-ℓ reduction of the hash output.
  // (Bit rows need no divmods.)
  let mut x = exp_outs[0].div_rem(p_hash).1;
  for _ in 0..(d.poseidon_rows / 3) {
    let x2 = (&x * &x).div_rem(p_hash).1;
    let x4 = (&x2 * &x2).div_rem(p_hash).1;
    let x5 = (&x4 * &x).div_rem(p_hash).1;
    out.push(x2);
    out.push(x4);
    out.push(x5.clone());
    x = x5;
  }
  out.push(x.div_rem(ell).1);

  // Per-swap `H∆` models (k > 0 only).
  let groups: &[(&BigUint, usize)] = &[(ell, 2 * d.k), (p_hash, 2 * d.k * d.h_rows)];
  let mut r = 0usize;
  for &(m, count) in groups {
    for _ in 0..count {
      let a = m - BigUint::from((r as u64 % 17) + 1);
      let b = m - BigUint::from(((r as u64 * 7) % 19) + 2);
      out.push((&a * &b).div_rem(m).1);
      r += 1;
    }
  }

  out
}

/// Which statement the bench proves. `MSCFG=full` (default): the OWWB20
/// `SetBench` statement from `limber::multiswap` with real dataflow
/// (`MSSWAPS` swaps, default 1). `MSCFG=paper`: the cost-model circuit
/// above (the paper's submission-time row). `MSCFG=bare`: the four variable-base 352-bit
/// exponentiations of the Garuda / Zinc+ comparison rows. Returns
/// `(shape, w, q, public_io)`.
type Workload<MM> = (
  IntModR1CSShapeModp<MM>,
  Vec<BigUint>,
  Vec<BigUint>,
  Vec<BigUint>,
);

#[derive(Clone, Debug)]
struct WitnessStats {
  w_nonzero: usize,
  q_nonzero: usize,
  w_max_bits: u64,
  q_max_bits: u64,
  w_set_bits: u64,
  q_set_bits: u64,
  w_nonzero_16bit_chunks: u64,
  q_nonzero_16bit_chunks: u64,
}

#[derive(Clone, Debug)]
struct PaperMetadata {
  batch_count: usize,
  comparison_digest: [u8; 32],
  statement_digest: [u8; 32],
  assignment_digest: [u8; 32],
  live_rows: usize,
  live_cols: usize,
  a_nnz: usize,
  b_nnz: usize,
  c_nnz: usize,
  witness: WitnessStats,
}

fn ws<MM: limber::traits::mod_engine::ModEngine>(d: Dims) -> Workload<MM> {
  use limber::multiswap::{
    poseidon::PoseidonParams,
    statement::{self, Config},
  };
  let cfg = cfg_name();
  match cfg.as_str() {
    "full" | "rsa" | "bare" => {
      let config = if cfg == "full" {
        let swaps = std::env::var("MSSWAPS")
          .ok()
          .and_then(|v| v.parse().ok())
          .unwrap_or(1);
        Config::Full { swaps }
      } else {
        Config::Rsa
      };
      let st = statement::build::<MM>(
        &config,
        &PoseidonParams::bls12_381_owwb20(),
        std::env::var_os("MSSEG").is_some(),
      )
      .expect("statement builds");
      (st.built.shape, st.built.w, st.built.q, st.built.io)
    }
    _ => {
      let (shape, w, q) = multiswap_shape_and_witness::<MM>(d);
      (shape, w, q, vec![])
    }
  }
}

fn cfg_name() -> String {
  std::env::var("MSCFG").unwrap_or_else(|_| "full".to_string())
}

fn multiswap_shape_and_witness_with_metadata<MM: limber::traits::mod_engine::ModEngine>(
  d: Dims,
) -> (
  IntModR1CSShapeModp<MM>,
  Vec<BigUint>,
  Vec<BigUint>,
  PaperMetadata,
) {
  let (shape, w, q, metadata) = multiswap_shape_and_witness_impl::<MM>(d, true);
  (shape, w, q, metadata.expect("metadata requested"))
}

fn multiswap_shape_and_witness<MM: limber::traits::mod_engine::ModEngine>(
  d: Dims,
) -> (IntModR1CSShapeModp<MM>, Vec<BigUint>, Vec<BigUint>) {
  let (shape, w, q, metadata) = multiswap_shape_and_witness_impl::<MM>(d, false);
  debug_assert!(metadata.is_none());
  (shape, w, q)
}

fn multiswap_shape_and_witness_impl<MM: limber::traits::mod_engine::ModEngine>(
  d: Dims,
  collect_metadata: bool,
) -> (
  IntModR1CSShapeModp<MM>,
  Vec<BigUint>,
  Vec<BigUint>,
  Option<PaperMetadata>,
) {
  // This boundary intentionally excludes provenance hashing and statistics
  // below. It measures only workload/circuit and assignment materialization,
  // matching the F2Z witness-generation boundary.
  let witness_generation = limber::bench_trace::scope("matched:witness_generation");
  let n = modulus_n();
  let ell = modulus_ell();
  let p_hash = modulus_p_hash();
  let bases = exp_bases();
  let exponents = exp_exponents(d.ell_bits);

  let num_cons = d.num_real_rows().next_power_of_two();
  let num_vars = d.num_real_cols().next_power_of_two();
  let num_io = 0;
  let const_col = num_vars;

  let mut a_entries = Vec::new();
  let mut b_entries = Vec::new();
  let mut c_entries = Vec::new();
  let mut mods = Vec::new();
  let mut w = vec![BigUint::from(0u32); num_vars];
  let mut q = vec![BigUint::from(0u32); num_cons];
  let one = BigUint::from(1u32);

  for i in 0..d.n_group_exps {
    build_exp_circuit(
      &bases[i],
      &exponents[i],
      &n,
      d.ell_bits,
      i * d.rows_per_exp(),
      i * d.cols_per_exp(),
      const_col,
      &mut a_entries,
      &mut b_entries,
      &mut c_entries,
      &mut mods,
      &mut w,
      &mut q,
    );
  }

  let mut row = d.n_group_exps * d.rows_per_exp();
  let mut col = d.n_group_exps * d.cols_per_exp();

  // Wired group mults: operands are the exponentiation outputs
  // (Q^ℓ-style products), one fresh result column each.
  let exp_out = |i: usize| i * d.cols_per_exp() + 2 * d.ell_bits;
  for i in 0..d.n_group_muls {
    let a_col = exp_out(2 * i);
    let b_col = exp_out(2 * i + 1);
    let (qi, ci) = (&w[a_col] * &w[b_col]).div_rem(&n);
    w[col] = ci;
    q[row] = qi;
    a_entries.push((row, a_col, one.clone()));
    b_entries.push((row, b_col, one.clone()));
    c_entries.push((row, col, one.clone()));
    mods.push(n.clone());
    row += 1;
    col += 1;
  }

  // Wired Hp certificate chains: real square-and-multiply over the
  // Mersenne moduli, with bit decomposition and reconstruction —
  // structurally identical to the main Wesolowski chains.
  let hp_ms = hp_moduli();
  let hp_inputs = hp_chain_inputs(d.hp_exp_bits);
  let mut hp_out_cols = [0usize; 4];
  for i in 0..d.hp_exps {
    build_exp_circuit(
      &hp_inputs[i].0,
      &hp_inputs[i].1,
      &hp_ms[i],
      d.hp_exp_bits,
      row,
      col,
      const_col,
      &mut a_entries,
      &mut b_entries,
      &mut c_entries,
      &mut mods,
      &mut w,
      &mut q,
    );
    hp_out_cols[i] = col + 2 * d.hp_exp_bits;
    row += d.rows_per_hp_exp();
    col += d.rows_per_hp_exp();
  }

  // Poseidon seed: reduce the first exponentiation output mod p_hash —
  // the hash input is wired to real circuit data.
  let seed_col = col;
  {
    let (qi, ci) = w[exp_out(0)].div_rem(&p_hash);
    w[seed_col] = ci;
    q[row] = qi;
    a_entries.push((row, exp_out(0), one.clone()));
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, seed_col, one.clone()));
    mods.push(p_hash.clone());
    row += 1;
    col += 1;
  }

  // Chained Poseidon-cost rows mod p_hash: per S-box x² = x·x,
  // x⁴ = x²·x², x⁵ = x⁴·x — one fresh column per row, the x⁵ output
  // feeding the next S-box (the linear MDS/round-constant layers fold
  // into the LCs of the following rows for free, exactly as a
  // constants-faithful build would).
  let zero = BigUint::from(0u32);
  let mut x_col = seed_col;
  for _ in 0..(d.poseidon_rows / 3) {
    let x = w[x_col].clone();
    let (q2, x2) = (&x * &x).div_rem(&p_hash);
    let (q4, x4) = (&x2 * &x2).div_rem(&p_hash);
    let (q5, x5) = (&x4 * &x).div_rem(&p_hash);
    // x² = x·x
    w[col] = x2;
    a_entries.push((row, x_col, one.clone()));
    b_entries.push((row, x_col, one.clone()));
    c_entries.push((row, col, one.clone()));
    mods.push(p_hash.clone());
    q[row] = q2;
    row += 1;
    // x⁴ = x²·x²
    w[col + 1] = x4;
    a_entries.push((row, col, one.clone()));
    b_entries.push((row, col, one.clone()));
    c_entries.push((row, col + 1, one.clone()));
    mods.push(p_hash.clone());
    q[row] = q4;
    row += 1;
    // x⁵ = x⁴·x
    w[col + 2] = x5;
    a_entries.push((row, col + 1, one.clone()));
    b_entries.push((row, x_col, one.clone()));
    c_entries.push((row, col + 2, one.clone()));
    mods.push(p_hash.clone());
    q[row] = q5;
    row += 1;
    x_col = col + 2;
    col += 3;
  }
  let pos_out_col = x_col;

  // Decomposition bit rows WIRED to real values: fully decompose the
  // Poseidon output and the four Hp chain outputs; each value gets an
  // exact (mod-0) reconstruction row referencing its bit columns —
  // zero fresh columns for reconstruction.
  let decomp_targets: Vec<(usize, usize)> = std::iter::once((pos_out_col, 255))
    .chain((0..4).map(|i| (hp_out_cols[i], [61usize, 89, 107, 127][i])))
    .collect();
  debug_assert_eq!(
    decomp_targets.iter().map(|&(_, b)| b).sum::<usize>(),
    d.decomp_bits
  );
  for &(val_col, nbits) in &decomp_targets {
    let val = w[val_col].clone();
    let bit_base = col;
    for j in 0..nbits {
      let bit = u8::from(val.bit((nbits - 1 - j) as u64));
      w[col] = BigUint::from(bit);
      a_entries.push((row, col, one.clone()));
      b_entries.push((row, col, one.clone()));
      c_entries.push((row, col, one.clone()));
      mods.push(zero.clone());
      q[row] = zero.clone();
      row += 1;
      col += 1;
    }
    for j in 0..nbits {
      let power = BigUint::from(1u32) << (nbits - 1 - j);
      a_entries.push((row, bit_base + j, power));
    }
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, val_col, one.clone()));
    mods.push(zero.clone());
    q[row] = zero.clone();
    row += 1;
  }

  // Final mod-ℓ reduction row, wired to the Poseidon output.
  {
    let (qi, ci) = w[pos_out_col].div_rem(&ell);
    w[col] = ci;
    q[row] = qi;
    a_entries.push((row, pos_out_col, one.clone()));
    b_entries.push((row, const_col, one.clone()));
    c_entries.push((row, col, one.clone()));
    mods.push(ell.clone());
    row += 1;
    col += 1;
  }

  // Per-swap `H∆` models (k > 0 only): still generic 3-column rows,
  // flagged as unfaithful — do not quote k > 0 configurations.
  let groups: Vec<(&BigUint, usize)> = vec![(&ell, 2 * d.k), (&p_hash, 2 * d.k * d.h_rows)];
  let mut r = 0usize;
  for (m, count) in &groups {
    for _ in 0..*count {
      let a_val = *m - BigUint::from((r as u64 % 17) + 1);
      let b_val = *m - BigUint::from(((r as u64 * 7) % 19) + 2);
      let prod = &a_val * &b_val;
      let (qi, ci) = prod.div_rem(m);
      w[col] = a_val;
      w[col + 1] = b_val;
      w[col + 2] = ci;
      q[row] = qi;
      a_entries.push((row, col, one.clone()));
      b_entries.push((row, col + 1, one.clone()));
      c_entries.push((row, col + 2, one.clone()));
      mods.push((*m).clone());
      row += 1;
      col += 3;
      r += 1;
    }
  }
  debug_assert_eq!(row, d.num_real_rows());
  debug_assert_eq!(col, d.num_real_cols());

  mods.resize(num_cons, BigUint::from(2u32));
  let batch_count = env_usize("MATCHED_BATCH_COUNT", 1);
  assert!(
    batch_count.is_power_of_two() && batch_count <= 16 && (batch_count == 1 || d.k == 0),
    "invalid paper batch"
  );
  let (num_cons, num_vars) = if batch_count == 1 {
    (num_cons, num_vars)
  } else {
    let live_rows = d.num_real_rows();
    let live_cols = d.num_real_cols();
    let target_rows = (batch_count * live_rows).next_power_of_two();
    let target_cols = (batch_count * live_cols).next_power_of_two();
    let expand = |entries: &[(usize, usize, BigUint)]| {
      (0..batch_count)
        .flat_map(|copy| {
          entries.iter().map(move |(r, c, v)| {
            (
              r + copy * live_rows,
              if *c == num_vars {
                target_cols
              } else {
                c + copy * live_cols
              },
              v.clone(),
            )
          })
        })
        .collect::<Vec<_>>()
    };
    a_entries = expand(&a_entries);
    b_entries = expand(&b_entries);
    c_entries = expand(&c_entries);
    let mut target_mods = vec![BigUint::from(2u8); target_rows];
    let mut target_w = vec![BigUint::zero(); target_cols];
    let mut target_q = vec![BigUint::zero(); target_rows];
    for copy in 0..batch_count {
      target_mods[copy * live_rows..(copy + 1) * live_rows].clone_from_slice(&mods[..live_rows]);
      target_q[copy * live_rows..(copy + 1) * live_rows].clone_from_slice(&q[..live_rows]);
      target_w[copy * live_cols..(copy + 1) * live_cols].clone_from_slice(&w[..live_cols]);
    }
    mods = target_mods;
    w = target_w;
    q = target_q;
    (target_rows, target_cols)
  };
  let shape = IntModR1CSShapeModp::<MM>::new(
    num_cons, num_vars, num_io, a_entries, b_entries, c_entries, mods,
  )
  .expect("valid IntMod-R1CS shape");
  drop(witness_generation);

  let metadata = collect_metadata.then(|| {
    let (a_entries, b_entries, c_entries, mods) = shape.benchmark_constraint_parts();
    let statement_digest = paper_statement_digest(d, num_cons, num_vars, a_entries, b_entries, c_entries, mods);
    let mut h = blake3::Hasher::new();
    h.update(b"bitz-limber/multiswap-statement/v2");
    h.update(&statement_digest);
    for v in [batch_count, 0, 2048, d.num_real_rows()*batch_count, d.num_real_cols()*batch_count] {
      h.update(&(v as u64).to_le_bytes());
    }
    h.update(b"public:matrices,moduli;private:witness,quotients;unsigned;constant:one;padding:zero-witness,zero-quotients,modulus-two");
    PaperMetadata {
      batch_count, comparison_digest: *h.finalize().as_bytes(), statement_digest,
      assignment_digest: f2z_assignment_digest(&w, &q, num_vars.max(num_cons)),
      live_rows: d.num_real_rows() * batch_count,
      live_cols: d.num_real_cols() * batch_count,
      a_nnz: a_entries.len(),
      b_nnz: b_entries.len(),
      c_nnz: c_entries.len(),
      witness: witness_stats(&w, &q),
    }
  });

  (shape, w, q, metadata)
}

fn paper_statement_digest(
  d: Dims,
  num_cons: usize,
  num_vars: usize,
  a: &[(usize, usize, BigUint)],
  b: &[(usize, usize, BigUint)],
  c: &[(usize, usize, BigUint)],
  mods: &[BigUint],
) -> [u8; 32] {
  let mut hasher = blake3::Hasher::new();
  hasher.update(b"bitz/multiswap/circuit-digest/v1");
  for value in [
    d.k,
    d.ell_bits,
    d.n_group_exps,
    d.n_group_muls,
    d.hp_exps,
    d.hp_exp_bits,
    d.poseidon_rows,
    d.decomp_bits,
    d.decomp_recon,
    d.h_rows,
    num_cons,
    num_vars,
    2048usize,
  ] {
    hasher.update(&(value as u64).to_le_bytes());
  }
  for entries in [a, b, c] {
    hasher.update(&(entries.len() as u64).to_le_bytes());
    for (row, column, value) in entries {
      hasher.update(&(*row as u64).to_le_bytes());
      hasher.update(&(*column as u64).to_le_bytes());
      let bytes = value.to_bytes_le();
      hasher.update(&(bytes.len() as u64).to_le_bytes());
      hasher.update(&bytes);
    }
  }
  hasher.update(&(mods.len() as u64).to_le_bytes());
  for modulus in mods {
    let bytes = modulus.to_bytes_le();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
  }
  *hasher.finalize().as_bytes()
}

fn f2z_assignment_digest(w: &[BigUint], q: &[BigUint], capacity: usize) -> [u8; 32] {
  assert!(w.len() <= capacity);
  assert!(q.len() <= capacity);
  let zero = BigUint::zero();
  let one = BigUint::from(1u8);
  let mut hasher = blake3::Hasher::new();
  hasher.update(b"bitz/multiswap/integer-assignment/v1");
  hasher.update(&(4u64 * capacity as u64).to_le_bytes());
  let mut absorb = |value: &BigUint| {
    let bytes = value.to_bytes_le();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
  };
  absorb(&one);
  for _ in 1..capacity {
    absorb(&zero);
  }
  for value in w {
    absorb(value);
  }
  for _ in w.len()..capacity {
    absorb(&zero);
  }
  for value in q {
    absorb(value);
  }
  for _ in q.len()..capacity {
    absorb(&zero);
  }
  for _ in 0..capacity {
    absorb(&zero);
  }
  *hasher.finalize().as_bytes()
}

fn set_bits(values: &[BigUint]) -> u64 {
  values
    .iter()
    .flat_map(BigUint::to_bytes_le)
    .map(|byte| u64::from(byte.count_ones()))
    .sum()
}

fn nonzero_16bit_chunks(values: &[BigUint]) -> u64 {
  values
    .iter()
    .map(|value| {
      value
        .to_bytes_le()
        .chunks(2)
        .filter(|chunk| chunk.iter().any(|&byte| byte != 0))
        .count() as u64
    })
    .sum()
}

fn witness_stats(w: &[BigUint], q: &[BigUint]) -> WitnessStats {
  WitnessStats {
    w_nonzero: w.iter().filter(|value| !value.is_zero()).count(),
    q_nonzero: q.iter().filter(|value| !value.is_zero()).count(),
    w_max_bits: w.iter().map(BigUint::bits).max().unwrap_or(0),
    q_max_bits: q.iter().map(BigUint::bits).max().unwrap_or(0),
    w_set_bits: set_bits(w),
    q_set_bits: set_bits(q),
    w_nonzero_16bit_chunks: nonzero_16bit_chunks(w),
    q_nonzero_16bit_chunks: nonzero_16bit_chunks(q),
  }
}

fn params_for(shape: &IntModR1CSShapeModp<M>, int_k: usize) -> IntEvalParams {
  let n = shape.num_vars().max(shape.num_cons());
  let log_n = (n as u64).ilog2() as usize;
  IntEvalParams::derive(2048, LOG_T, int_k, log_n).expect("IntEval params satisfy bounds")
}

/// IntEval `k` for the Hyrax instantiation: `IMOD_K=<k>` overrides the
/// tuned `DEFAULT_K` (mirrors `BDK` for the Brakedown path).
fn hyrax_k() -> usize {
  std::env::var("IMOD_K")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(DEFAULT_K)
}

#[derive(Clone, Copy)]
enum MatchedTrial {
  Warmup(usize),
  Sample(usize),
}

impl MatchedTrial {
  fn fragment(self) -> String {
    match self {
      Self::Warmup(index) => format!("warmup-{index}"),
      Self::Sample(index) => format!("sample-{index}"),
    }
  }

  fn json(self) -> Value {
    match self {
      Self::Warmup(index) => json!({"kind": "warmup", "warmup_index": index}),
      Self::Sample(index) => json!({"kind": "sample", "sample_index": index}),
    }
  }
}

struct MatchedArtifacts {
  commitment_bytes: usize,
  opening_argument_bytes: usize,
  dynamic_sumcheck_bytes_estimate: usize,
  active_logup_blocks: usize,
  total_logup_blocks: usize,
}

struct MatchedTraceWriter {
  output: BufWriter<File>,
  campaign_id: String,
  git_rev: String,
  git_dirty: bool,
  cpu: String,
}

impl MatchedTraceWriter {
  fn from_env() -> Self {
    let path = std::env::var_os("MATCHED_TRACE_PATH")
      .expect("MATCHED_TRACE_PATH is required with MATCHED_BENCH=1");
    let path = Path::new(&path);
    if let Some(parent) = path
      .parent()
      .filter(|parent| !parent.as_os_str().is_empty())
    {
      fs::create_dir_all(parent).expect("create matched trace directory");
    }
    let output = BufWriter::new(
      OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("create new matched trace JSONL"),
    );
    Self {
      output,
      campaign_id: std::env::var("MATCHED_CAMPAIGN_ID")
        .unwrap_or_else(|_| "multiswap-rsa-v1".to_owned()),
      git_rev: env!("LIMBER_REVISION").to_owned(),
      // The shared build verifier checks this snapshot against root provenance.
      git_dirty: false,
      cpu: command_output(
        "sysctl",
        &["-n", "machdep.cpu.brand_string"],
        std::env::consts::ARCH,
      ),
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn write_trial(
    &mut self,
    backend: &str,
    multiswap_k: usize,
    int_k: usize,
    params: &IntEvalParams,
    setup_ns: u64,
    trial: MatchedTrial,
    metadata: &PaperMetadata,
    artifacts: &MatchedArtifacts,
    intervals: &[limber::bench_trace::ProfileInterval],
  ) {
    let roots = intervals
      .iter()
      .filter(|interval| interval.parent_order.is_none())
      .collect::<Vec<_>>();
    assert_eq!(roots.len(), 1, "one traced root per matched trial");
    assert_eq!(roots[0].label, "matched:verified_trial");
    let threads = rayon::current_num_threads();
    let trial_fragment = trial.fragment();
    let batch_count = metadata.batch_count;
    let run_id = format!(
      "multiswap-{}-k{multiswap_k}-b{batch_count}-{backend}-{threads}t-{}-{trial_fragment}",
      self.campaign_id,
      std::process::id(),
    );
    let implementation = format!("limber-{backend}");
    let backend_label = if backend == "hyrax" {
      "Hyrax"
    } else {
      "Brakedown"
    };
    let statement_digest = hex::encode(metadata.statement_digest);
    let assignment_digest = hex::encode(metadata.assignment_digest);
    let assignment_capacity = metadata
      .live_rows
      .next_power_of_two()
      .max(metadata.live_cols.next_power_of_two());
    let run = json!({
      "schema": "zkperf.trace/v1",
      "record": "run",
      "run_id": run_id,
      "series_id": format!(
        "{}-k{multiswap_k}-b{batch_count}-{implementation}-{threads}t-{}",
        self.campaign_id, self.git_rev
      ),
      "root_span_id": span_id(roots[0].order),
      "benchmark": {
        "suite": "limber",
        "name": "multiswap-rsa-matched",
        "label": format!(
          "Limber {backend_label}: wired MultiSwap/RSA cost-model (k={multiswap_k})"
        ),
        "algorithm": format!(
          "wired MultiSwap/RSA cost-model / Spartan + IntEval/LogUp-GKR + {backend_label}"
        ),
        "implementation": implementation,
        "git_rev": self.git_rev,
        "git_dirty": self.git_dirty,
        "build_profile": "bench",
      },
      "trial": trial.json(),
      "clock": {
        "id": format!("mono-process-{}-{run_id}", std::process::id()),
        "kind": "monotonic",
        "unit": "ns",
        "source": "std::time::Instant",
      },
      "status": "ok",
      "trace_complete": true,
      "environment": {
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": self.cpu,
        "threads": threads,
        "thread_policy": "Rayon pool; no affinity pinning",
      },
      "parameters": {
        "input": {
          "workload_id": "multiswap-rsa-wired-cost-model-v1",
          "workload": format!(
            "wired MultiSwap/RSA cost-model (MSCFG=paper, k={multiswap_k})"
          ),
          "workload_k": multiswap_k,
          "limber_k": multiswap_k,
          "batch_count": metadata.batch_count, "public_input_count": 0, "public_inputs": [],
          "statement_contract": statement_contract(metadata),
          "semantic_scope": "real RSA exponentiation chains; hash/Poseidon portions are cost-modeled",
          "live_rows": metadata.live_rows,
          "live_cols": metadata.live_cols,
          "num_cons": metadata.live_rows.next_power_of_two(),
          "num_vars": metadata.live_cols.next_power_of_two(),
          "value_bits": 2048,
          "a_nnz": metadata.a_nnz,
          "b_nnz": metadata.b_nnz,
          "c_nnz": metadata.c_nnz,
        },
        "statement": {
          "constraint_digest_domain": "bitz/multiswap/circuit-digest/v1",
          "constraint_digest_blake3": statement_digest,
          "assignment_digest_domain": "bitz/multiswap/integer-assignment/v1",
          "assignment_digest_blake3": assignment_digest,
          "assignment_layout": format!(
            "[constant block | W | Q | zero block], each block padded to {assignment_capacity}"
          ),
          "same_integer_relation": true,
          "native_transcript_challenges": true,
        },
        "witness": {
          "w_nonzero": metadata.witness.w_nonzero,
          "q_nonzero": metadata.witness.q_nonzero,
          "w_max_bits": metadata.witness.w_max_bits,
          "q_max_bits": metadata.witness.q_max_bits,
          "w_set_bits": metadata.witness.w_set_bits,
          "q_set_bits": metadata.witness.q_set_bits,
          "w_nonzero_16bit_chunks": metadata.witness.w_nonzero_16bit_chunks,
          "q_nonzero_16bit_chunks": metadata.witness.q_nonzero_16bit_chunks,
        },
        "int_eval": {
          "backend": backend,
          "log_t": LOG_T,
          "k": int_k,
          "log_q": params.log_q,
          "log_p": params.log_p,
          "small_primes": params.s,
          "log_t_f": params.log_t_f,
          "numlimb": params.numlimb,
          "numlimb_var": params.numlimb_var,
        },
        "security": security_metadata(backend, params, assignment_capacity.ilog2() as usize),
        "repetition": {"count": 1},
        "setup_ns_excluded": setup_ns.to_string(),
      },
      "artifacts": {
        "proof_bytes": artifacts.commitment_bytes + artifacts.opening_argument_bytes + artifacts.dynamic_sumcheck_bytes_estimate,
        "proof_size_kind": "serialized commitment/opening plus analytical sumcheck estimate",
        "peak_rss_bytes": peak_rss_bytes(),
        "memory_boundary": "process high-water RSS including setup and warmups; compiler excluded",
        "commitment_bytes": artifacts.commitment_bytes,
        "opening_argument_bytes": artifacts.opening_argument_bytes,
        "dynamic_sumcheck_bytes_estimate": artifacts.dynamic_sumcheck_bytes_estimate,
        "total_wire_bytes_estimate": artifacts.commitment_bytes
          + artifacts.opening_argument_bytes
          + artifacts.dynamic_sumcheck_bytes_estimate,
      },
      "validation": {
        "proof_verified": true,
        "integer_relation_preflight": true,
        "statement_digest_match_required": true,
        "statement_digest_blake3": hex::encode(metadata.statement_digest),
        "assignment_digest_blake3": hex::encode(metadata.assignment_digest),
        "active_logup_blocks": artifacts.active_logup_blocks,
        "total_logup_blocks": artifacts.total_logup_blocks,
      },
      "tags": {
        "campaign_id": self.campaign_id,
        "root_boundary": "witness generation through verified proof; setup excluded",
        "timeline": "observed half-open intervals",
      },
    });
    serde_json::to_writer(&mut self.output, &run).expect("write matched run");
    writeln!(self.output).expect("terminate matched run");

    let by_order = intervals
      .iter()
      .map(|interval| (interval.order, interval))
      .collect::<HashMap<_, _>>();
    let mut occurrences = HashMap::<(Option<u64>, &'static str), usize>::new();
    let mut totals = HashMap::<(Option<u64>, &'static str), usize>::new();
    for interval in intervals {
      *totals
        .entry((interval.parent_order, interval.label))
        .or_default() += 1;
    }
    for interval in intervals {
      let descriptor = describe_matched_span(interval, &by_order);
      let key = (interval.parent_order, interval.label);
      let occurrence = occurrences.entry(key).or_default();
      let occurrence_count = totals[&key];
      let coordinate = if occurrence_count > 1 {
        json!({"occurrence_index": *occurrence, "occurrence_count": occurrence_count})
      } else {
        json!({})
      };
      *occurrence += 1;
      let mut attributes = json!({
        "scope_kind": descriptor.scope_kind,
        "short_name": descriptor.short_name,
        "primary_sequence": descriptor.primary_sequence,
        "source_label": interval.label,
      });
      if !descriptor.math_latex.is_empty() {
        attributes["math_latex"] = json!(descriptor.math_latex);
      }
      let span = json!({
        "schema": "zkperf.trace/v1",
        "record": "span",
        "run_id": run_id,
        "span_id": span_id(interval.order),
        "parent_span_id": interval.parent_order.map(span_id),
        "operation": descriptor.operation,
        "name": descriptor.name,
        "primary_phase": descriptor.primary_phase,
        "phase_tags": descriptor.phase_tags,
        "start_ns": interval.start_ns.to_string(),
        "end_ns": interval.end_ns.to_string(),
        "duration_ns": interval.end_ns.saturating_sub(interval.start_ns).to_string(),
        "lane": {"process": "benchmark", "thread": "control"},
        "coordinate": coordinate,
        "attributes": attributes,
      });
      serde_json::to_writer(&mut self.output, &span).expect("write matched span");
      writeln!(self.output).expect("terminate matched span");
    }
    self.output.flush().expect("flush matched trace JSONL");
  }
}

struct MatchedSpanDescriptor {
  operation: String,
  name: String,
  short_name: String,
  primary_phase: &'static str,
  phase_tags: Vec<&'static str>,
  scope_kind: &'static str,
  primary_sequence: bool,
  math_latex: Vec<&'static str>,
}

fn describe_matched_span(
  interval: &limber::bench_trace::ProfileInterval,
  by_order: &HashMap<u64, &limber::bench_trace::ProfileInterval>,
) -> MatchedSpanDescriptor {
  let mut ancestors = Vec::new();
  let mut cursor = interval.parent_order;
  while let Some(order) = cursor {
    let parent = by_order[&order];
    ancestors.push(parent.label);
    cursor = parent.parent_order;
  }
  let under_open = interval.label == "imod_modp_wq_open"
    || ancestors.iter().any(|label| *label == "imod_modp_wq_open");
  let under_commit =
    interval.label == "matched:commit" || ancestors.iter().any(|label| *label == "matched:commit");
  let under_verify =
    interval.label == "matched:verify" || ancestors.iter().any(|label| *label == "matched:verify");
  let (operation, name, short_name, primary_phase, phase_tags, primary_sequence) =
    match interval.label {
      "matched:verified_trial" => (
        "multiswap.trial".to_owned(),
        "Verified trial".to_owned(),
        "End-to-end".to_owned(),
        "end-to-end",
        vec!["end-to-end"],
        false,
      ),
      "matched:witness_generation" => (
        "multiswap.witness_generation".to_owned(),
        "Generate wired MultiSwap/RSA witness".to_owned(),
        "Witness".to_owned(),
        "witness-generation",
        vec!["witness-generation"],
        true,
      ),
      "matched:commit" => (
        "multiswap.commit".to_owned(),
        "Commit W and Q".to_owned(),
        "Commit".to_owned(),
        "commit",
        vec!["commit", "pcs"],
        true,
      ),
      "matched:prover" => (
        "multiswap.prover".to_owned(),
        "Total prover".to_owned(),
        "Prover".to_owned(),
        "proving",
        vec!["proving"],
        false,
      ),
      "matched:verify" => (
        "multiswap.verify".to_owned(),
        "Verify proof".to_owned(),
        "Verify".to_owned(),
        "verification",
        vec!["verification"],
        true,
      ),
      label if under_verify => (
        format!("limber.verify.{}", label.replace([':', '_'], ".")),
        label.replace('_', " "),
        label.replace('_', " "),
        "verification",
        vec!["verification"],
        false,
      ),
      "imod_spartan_modp_prove" => (
        "limber.native.prover-pipeline".to_owned(),
        "Limber prover pipeline".to_owned(),
        "Prover pipeline".to_owned(),
        "proving",
        vec!["proving"],
        false,
      ),
      "imod_modp_projection" => (
        "limber.projection".to_owned(),
        "Project integer relation into the transcript field".to_owned(),
        "Projection".to_owned(),
        "preparation",
        vec!["proving", "preparation"],
        true,
      ),
      "imod_modp_piop" => (
        "limber.piop".to_owned(),
        "Spartan relation proof".to_owned(),
        "PIOP".to_owned(),
        "constraint-proof",
        vec!["proving", "constraint-proof"],
        true,
      ),
      "imod_modp_sample_p" => projection_descriptor("sample_prime", "Sample fingerprint prime"),
      "imod_modp_reduce" => projection_descriptor("reduce", "Reduce relation modulo p"),
      "imod_modp_spmv" => projection_descriptor("spmv", "Sparse matrix products"),
      "imod_modp_outer_sumcheck" => piop_descriptor("outer_sumcheck", "Outer sumcheck"),
      "imod_modp_inner_setup" => piop_descriptor("inner_setup", "Inner sumcheck setup"),
      "imod_modp_inner_sumcheck" => piop_descriptor("inner_sumcheck", "Inner sumcheck"),
      "imod_modp_eval_recover" => piop_descriptor("eval_recover", "Recover witness evaluation"),
      "imod_modp_wq_open" => (
        "limber.pcs.opening".to_owned(),
        "Limber PCS opening".to_owned(),
        "PCS opening".to_owned(),
        "opening-proof",
        vec!["pcs", "opening-proof"],
        true,
      ),
      label => {
        let phase = if under_verify {
          "verification"
        } else if under_commit {
          "commit"
        } else if under_open {
          "opening-proof"
        } else {
          "proving"
        };
        let tags = match phase {
          "verification" => vec!["verification"],
          "commit" => vec!["commit", "pcs"],
          "opening-proof" => vec!["pcs", "opening-proof"],
          _ => vec!["proving"],
        };
        (
          format!("limber.native.{}", label.replace([':', '_'], ".")),
          label.replace('_', " "),
          label.replace('_', " "),
          phase,
          tags,
          false,
        )
      }
    };
  MatchedSpanDescriptor {
    operation,
    name,
    short_name,
    primary_phase,
    phase_tags,
    scope_kind: match interval.label {
      "matched:verified_trial" => "scope",
      "matched:witness_generation" => "operation",
      _ if primary_sequence => "phase",
      _ if interval.label.contains("round") => "round",
      _ => "procedure",
    },
    primary_sequence,
    math_latex: matched_math(interval.label),
  }
}

fn matched_math(label: &str) -> Vec<&'static str> {
  match label {
    "matched:witness_generation" => {
      vec!["A\\mathbf z\\circ B\\mathbf z=C\\mathbf z+\\mathbf m\\circ\\mathbf Q"]
    }
    "matched:commit" | "imod_modp_wq_commit" => {
      vec!["(C_W,C_Q)\\leftarrow\\operatorname{Commit}(\\mathbf W,\\mathbf Q)"]
    }
    "imod_modp_sample_p" => {
      vec!["p\\leftarrow\\operatorname{FS}(C_W,C_Q,\\mathbf x)"]
    }
    "imod_modp_reduce" | "imod_modp_spmv" => {
      vec!["A_p\\mathbf z_p\\circ B_p\\mathbf z_p=C_p\\mathbf z_p+\\mathbf m_p\\circ\\mathbf Q_p"]
    }
    "imod_modp_outer_sumcheck" => vec![
      "\\sum_{u\\in\\{0,1\\}^n}\\widetilde{eq}(\\tau,u)(\\widetilde{Az}\\widetilde{Bz}-\\widetilde{Cz}-\\widetilde m\\widetilde Q)=0",
    ],
    "imod_modp_inner_setup" | "imod_modp_inner_sumcheck" => {
      vec!["\\widetilde{Mz}(r)=\\sum_j\\widetilde M(r,j)\\widetilde z(j)"]
    }
    "imod_modp_wq_open" => {
      vec!["\\sum_i\\lambda^i C(z_i)=\\sum_i\\lambda^i v_i"]
    }
    "rc_logup_gkr" => vec!["\\sum_b\\sum_i(r+w_{b,i})^{-1}=\\sum_{j=0}^{2^{16}-1}m_j(r+j)^{-1}"],
    _ => Vec::new(),
  }
}

fn piop_descriptor(
  suffix: &'static str,
  name: &'static str,
) -> (
  String,
  String,
  String,
  &'static str,
  Vec<&'static str>,
  bool,
) {
  (
    format!("limber.piop.{suffix}"),
    name.to_owned(),
    name.to_owned(),
    "constraint-proof",
    vec!["proving", "constraint-proof"],
    false,
  )
}

fn projection_descriptor(
  suffix: &'static str,
  name: &'static str,
) -> (
  String,
  String,
  String,
  &'static str,
  Vec<&'static str>,
  bool,
) {
  (
    format!("limber.projection.{suffix}"),
    name.to_owned(),
    name.to_owned(),
    "preparation",
    vec!["proving", "preparation"],
    false,
  )
}

fn span_id(order: u64) -> String {
  format!("span-{order:06}")
}

fn command_output(program: &str, args: &[&str], fallback: &str) -> String {
  Command::new(program)
    .args(args)
    .output()
    .ok()
    .filter(|output| output.status.success())
    .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    .filter(|output| !output.is_empty())
    .unwrap_or_else(|| fallback.to_owned())
}

fn env_usize(name: &str, default: usize) -> usize {
  std::env::var(name)
    .ok()
    .and_then(|value| value.parse().ok())
    .unwrap_or(default)
}

fn run_matched_campaign_for<E, Setup, LogupCounts>(
  backend: &'static str,
  multiswap_k: usize,
  int_k: usize,
  setup: Setup,
  logup_counts: LogupCounts,
) where
  E: limber::traits::mod_engine::ModEngine<TE = limber::provider::keccak::Keccak256Transcript<E>>,
  Setup: Fn(
    IntModR1CSShapeModp<E>,
    IntEvalParams,
  ) -> Result<
    (
      limber::imod_spartan_modp::IntModSpartanModpProverKey<E>,
      limber::imod_spartan_modp::IntModSpartanModpVerifierKey<E>,
    ),
    limber::errors::SpartanError,
  >,
  LogupCounts: Fn(&IntModSpartanModpSNARK<E>) -> (usize, usize),
{
  let dims = Dims::multiswap(multiswap_k);
  let (template_shape, template_w, template_q, expected_metadata) =
    multiswap_shape_and_witness_with_metadata::<E>(dims);
  if multiswap_k == 0 {
    assert_eq!(
      expected_metadata.live_rows,
      6209 * expected_metadata.batch_count
    );
    assert_eq!(
      expected_metadata.live_cols,
      6204 * expected_metadata.batch_count
    );
  }
  let log_n = (template_shape.num_vars().max(template_shape.num_cons()) as u64).ilog2() as usize;
  let (_, integer_target) = matched_security_targets();
  let params = IntEvalParams::derive_with_target(2048, LOG_T, int_k, log_n, integer_target)
    .expect("matched IntEval params satisfy bounds");
  // Reject an unreachable comparison target before setup or any proof work.
  let security = security_metadata(backend, &params, log_n);
  if std::env::var("MATCHED_CHECK_ONLY").as_deref() == Ok("1") {
    println!(
      "MATCHED_PREFLIGHT {}",
      json!({
        "statement_contract": statement_contract(&expected_metadata),
        "assignment_digest_blake3": hex::encode(expected_metadata.assignment_digest),
        "security": security,
      })
    );
    return;
  }
  let setup_start = Instant::now();
  let (pk, vk) = setup(template_shape.clone(), params.clone()).expect("matched setup");
  let setup_ns = u64::try_from(setup_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

  // One untimed integer-relation and commitment preflight.  This protects the
  // campaign from timing a malformed fixture and warms public setup caches;
  // the configured warmup proof remains separate and is still emitted.
  let (preflight_witness, preflight_instance) =
    IntModR1CSWitnessModp::<E>::new(&template_shape, pk.ck(), template_w, template_q, vec![])
      .expect("matched preflight commitment");
  template_shape
    .is_sat(pk.ck(), &preflight_instance, &preflight_witness)
    .expect("matched integer relation is satisfiable");

  let warmups = env_usize("MATCHED_WARMUPS", 1);
  let samples = env_usize("MATCHED_SAMPLES", 5);
  assert!(
    warmups >= 1,
    "matched campaign requires at least one warmup"
  );
  assert!(
    samples >= 1,
    "matched campaign requires at least one measured sample"
  );
  let trials = (0..warmups)
    .map(MatchedTrial::Warmup)
    .chain((0..samples).map(MatchedTrial::Sample));
  let mut writer = MatchedTraceWriter::from_env();
  limber::bench_trace::set_enabled(true);
  for trial in trials {
    let (metadata, instance, proof, trial_witness, trial_shape);
    {
      let _root = limber::bench_trace::scope("matched:verified_trial");
      let (shape, w, q) = multiswap_shape_and_witness::<E>(dims);

      // Total prover is commit + Spartan/PCS prove, matching the F2Z headline
      // boundary. The nested commitment interval remains independently usable.
      let prover = limber::bench_trace::scope("matched:prover");
      let (witness, trial_instance) = {
        let _commit = limber::bench_trace::scope("matched:commit");
        IntModR1CSWitnessModp::<E>::new(&shape, pk.ck(), w, q, vec![])
          .expect("matched W/Q commitment")
      };
      let trial_proof = IntModSpartanModpSNARK::<E>::prove(&pk, &trial_instance, &witness)
        .expect("matched proof generation");
      drop(prover);
      {
        let _verify = limber::bench_trace::scope("matched:verify");
        trial_proof
          .verify(&vk, &trial_instance)
          .expect("matched proof verifies");
      }
      instance = trial_instance;
      proof = trial_proof;
      trial_witness = witness;
      trial_shape = shape;
    }
    // Canonical statement/assignment provenance was computed and checked by
    // the untimed template preflight. Reusing it here keeps hashing/statistics
    // out of every measured trial.
    let (trial_w, trial_q) = trial_witness.benchmark_assignment();
    let (a, b, c, m) = trial_shape.benchmark_constraint_parts();
    assert_eq!(
      paper_statement_digest(
        dims,
        trial_shape.num_cons(),
        trial_shape.num_vars(),
        a,
        b,
        c,
        m
      ),
      expected_metadata.statement_digest
    );
    assert_eq!(
      f2z_assignment_digest(
        trial_w,
        trial_q,
        trial_shape.num_vars().max(trial_shape.num_cons())
      ),
      expected_metadata.assignment_digest
    );
    metadata = expected_metadata.clone();
    let intervals = limber::bench_trace::take_intervals();
    // Serialization and reporting are deliberately outside the root timing
    // boundary; only witness generation through successful verification is
    // measured.
    let commitment_bytes = instance
      .commitment_bytes()
      .expect("matched commitment serialization")
      .len();
    let opening_argument_bytes = proof
      .eval_arg_bytes()
      .expect("matched opening argument serialization")
      .len();
    let (active_logup_blocks, total_logup_blocks) = logup_counts(&proof);
    // A block has at most 2^16 slots; shifted tops add at most as many.
    assert!(
      2 * (total_logup_blocks as u128) * (1u128 << 16) <= range_slot_bound(&params, log_n),
      "range-check size exceeds security accounting cap"
    );
    let outer_rounds = template_shape.num_cons().ilog2() as usize;
    let inner_rounds = (2 * template_shape.num_vars()).ilog2() as usize;
    // The dynamic-prime proof is not Serialize yet.  This is the same
    // explicit field-count estimate used by the pre-existing PSIZE path.
    let dynamic_sumcheck_bytes_estimate = (outer_rounds * 3 + inner_rounds * 2 + 6) * 16;
    let artifacts = MatchedArtifacts {
      commitment_bytes,
      opening_argument_bytes,
      dynamic_sumcheck_bytes_estimate,
      active_logup_blocks,
      total_logup_blocks,
    };
    writer.write_trial(
      backend,
      multiswap_k,
      int_k,
      &params,
      setup_ns,
      trial,
      &metadata,
      &artifacts,
      &intervals,
    );
  }
  limber::bench_trace::set_enabled(false);
}

fn run_matched_campaign() {
  let cfg = cfg_name();
  assert_eq!(cfg, "paper", "matched campaign requires MSCFG=paper");
  let multiswap_k = env_usize("MATCHED_K", 0);
  match std::env::var("MATCHED_BACKEND")
    .unwrap_or_else(|_| "hyrax".to_owned())
    .as_str()
  {
    "hyrax" => run_matched_campaign_for::<M, _, _>(
      "hyrax",
      multiswap_k,
      9,
      IntModSpartanModpSNARK::<M>::setup_with_params,
      IntModSpartanModpSNARK::<M>::logup_block_counts,
    ),
    "brakedown" => {
      type BE = limber::provider::T256DynPrimeBdEngine;
      run_matched_campaign_for::<BE, _, _>(
        "brakedown",
        multiswap_k,
        11,
        IntModSpartanModpSNARK::<BE>::setup_with_params,
        IntModSpartanModpSNARK::<BE>::logup_block_counts,
      );
    }
    backend => panic!("MATCHED_BACKEND must be hyrax or brakedown, got {backend:?}"),
  }
}

fn paper_fp_constraints(k: usize) -> u64 {
  let f = 255u64;
  let b_h_delta = 2048u64;
  let ell_bits = 352u64;
  let c_he = 316;
  let c_hin = 316;
  let c_split = 388;
  let c_add_ell = 16 + f;
  let c_mul_ell = 479;
  let c_e_g = 7044 * ell_bits;
  let c_x_g = 7563;
  let c_hp = 217703;
  let c_mod_ell = 16 + b_h_delta;

  let per_op = 2 * (c_he + c_hin + c_split + c_add_ell + c_mul_ell);
  let per_proof = 4 * c_e_g + 2 * c_x_g + c_hp + c_mod_ell;
  (k as u64) * per_op + per_proof
}

fn multiswap_modp_benches(c: &mut Criterion) {
  if std::env::var_os("MATCHED_BENCH").is_some() {
    run_matched_campaign();
    return;
  }
  let ks: &[usize] = &[0];

  // BDPCS=1: measure the Brakedown-backed instantiation (hash
  // commitments, non-hiding) on the same workload: commit+prove,
  // verify, and serialized proof size. The comparison point against
  // code-commitment systems.
  if std::env::var_os("BDPCS").is_some() {
    use limber::provider::T256DynPrimeBdEngine as BE;
    use std::time::Instant;
    if std::env::var_os("RUST_LOG").is_some() {
      let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    }
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<BE>(dims);
    let log_n = (shape.num_vars().max(shape.num_cons()) as u64).ilog2() as usize;
    let bdk: usize = std::env::var("BDK")
      .ok()
      .and_then(|v| v.parse().ok())
      .unwrap_or(11); // k=11 dominates k=9 across the sweep for the hash backend
    let params =
      IntEvalParams::derive(2048, LOG_T, bdk, log_n).expect("IntEval params satisfy bounds");
    let (pk, vk) = IntModSpartanModpSNARK::<BE>::setup_with_params(shape.clone(), params).unwrap();
    // Pre-warm the per-length code layouts (deterministic public
    // matrices; conceptually part of setup, not of commit).
    let tw = Instant::now();
    let nvars = shape.num_vars().max(shape.num_cons());
    let f_chunk_len = (nvars * 32 * 4).next_power_of_two();
    let _ = limber::provider::pcs::prewarm_brakedown_params(f_chunk_len);
    println!(
      "  (params prewarm for f-chunk length: {:.1} ms)",
      tw.elapsed().as_secs_f64() * 1e3
    );
    let t0 = Instant::now();
    let (witness, instance) = IntModR1CSWitnessModp::<BE>::new(&shape, pk.ck(), w, q, io).unwrap();
    let t_commit = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let proof = IntModSpartanModpSNARK::<BE>::prove(&pk, &instance, &witness).unwrap();
    let t_prove = t1.elapsed().as_secs_f64() * 1e3;
    let proof_bytes = proof.eval_arg_bytes().expect("eval_arg serializes").len();
    let t2 = Instant::now();
    proof.verify(&vk, &instance).unwrap();
    let t_verify = t2.elapsed().as_secs_f64() * 1e3;
    if let Some(path) = std::env::var_os("BDDUMP") {
      let bytes = bincode::serialize(proof.eval_arg_ref()).unwrap();
      std::fs::write(&path, &bytes).unwrap();
      println!("  dumped eval_arg ({} bytes) to {:?}", bytes.len(), path);
    }
    if std::env::var_os("BDANATOMY").is_some() {
      let open_args = proof.bd_open_args();
      for (g, a) in open_args.groups.iter().enumerate() {
        let (rows, cols, auth) = a.component_sizes();
        println!("  group {g}: combined rows {rows} B, columns {cols} B, auth {auth} B");
      }
      for (d, a) in open_args.direct.iter().enumerate() {
        println!("  direct {d}: {} B", a.size());
      }
    }
    println!(
      "MultiSwap {} 2^{} / Brakedown Mod-PCS: commit {t_commit:.1} ms, prove {t_prove:.1} ms, \
       total {:.1} ms, verify {t_verify:.1} ms, proof {} bytes ({:.2} MB)",
      cfg_name(),
      log_n,
      t_commit + t_prove,
      proof_bytes,
      proof_bytes as f64 / 1e6,
    );
    return;
  }

  // PSIZE=1: serialized proof size of the Hyrax-backed instantiation on
  // the standard workload. `eval_arg_bytes` covers the Mod-PCS batch
  // argument (commitments, GKR, combined opening) — the dominant part;
  // the dynamic-prime side (outer/inner sumcheck round polynomials and
  // claimed evals, ~1.2 KB at 2^13) is not yet `Serialize` and is
  // reported analytically alongside.
  // M127=1: the small-field instantiation (F127 + Brakedown) on the
  // standard workload — first honest numbers for the fast-prover
  // operating point. Unoptimized: eager delayed-reduction, per-target
  // Brakedown openings (no two-tree batching yet).
  if std::env::var_os("M127").is_some() {
    use std::time::Instant;
    if std::env::var_os("RUST_LOG").is_some() {
      let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    }
    type Sf = limber::provider::M127DynPrimeBdEngine;
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<Sf>(dims);
    let log_n = (shape.num_vars().max(shape.num_cons()) as u64).ilog2() as usize;
    // q = 127; 16-bit limbs (= chunks); k = 5 per the parameter grid.
    let params =
      IntEvalParams::derive_for_q(127, 2048, 16, 5, log_n).expect("M127 params satisfy bounds");
    println!(
      "M127 params: log_p={} s={} k={} numlimb={}",
      params.log_p, params.s, params.k, params.numlimb
    );
    let (pk, vk) = IntModSpartanModpSNARK::<Sf>::setup_with_params(shape.clone(), params).unwrap();
    let t0 = Instant::now();
    let (witness, instance) = IntModR1CSWitnessModp::<Sf>::new(&shape, pk.ck(), w, q, io).unwrap();
    let proof = IntModSpartanModpSNARK::<Sf>::prove(&pk, &instance, &witness).unwrap();
    let t_prove = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    proof.verify(&vk, &instance).unwrap();
    let t_verify = t1.elapsed().as_secs_f64() * 1e3;
    let (pp, rc, co) = proof.eval_arg_component_sizes();
    println!(
      "MultiSwap 2^13 / M127-Brakedown: commit+prove {t_prove:.2} s, verify {t_verify:.1} ms, \
       eval_arg {:.2} MB [per_poly {pp} B, range_check {rc} B, combined_open {co} B]",
      (pp + rc + co) as f64 / 1e6,
    );
    return;
  }

  if std::env::var_os("PSIZE").is_some() {
    use std::time::Instant;
    if std::env::var_os("RUST_LOG").is_some() {
      let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    }
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<M>(dims);
    let log_n = (shape.num_vars().max(shape.num_cons()) as u64).ilog2() as usize;
    let params =
      IntEvalParams::derive(2048, LOG_T, hyrax_k(), log_n).expect("IntEval params satisfy bounds");
    let (pk, vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
    let t0 = Instant::now();
    let (witness, instance) = IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
    let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
    let total = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    proof.verify(&vk, &instance).unwrap();
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    let arg_bytes = proof.eval_arg_bytes().expect("serialize eval_arg").len();
    let (pp, rc, co) = proof.eval_arg_component_sizes();
    println!("  breakdown: per_poly {pp} B, range_check {rc} B, combined_open {co} B");
    // PSDUMP=<path>: write the serialized eval_arg so its compressed
    // size can be measured externally (e.g. `zstd -19`).
    if let Some(path) = std::env::var_os("PSDUMP") {
      let bytes = bincode::serialize(proof.eval_arg_ref()).unwrap();
      std::fs::write(&path, &bytes).unwrap();
      println!("  dumped eval_arg ({} bytes) to {:?}", bytes.len(), path);
    }
    // Dynamic-prime remainder: 13 cubic outer rounds (3 coeffs each) +
    // 14 quadratic inner rounds (2 coeffs) + 6 claimed evals, 16 B per
    // 2-limb scalar.
    let dyn_bytes = (13 * 3 + 14 * 2 + 6) * 16;
    println!(
      "MultiSwap {} 2^{} / Hyrax Mod-PCS proof size: eval_arg {arg_bytes} bytes \
       + ~{dyn_bytes} B sumcheck side ≈ {:.1} KB  (commit+prove {total:.2} s, verify {verify_ms:.1} ms)",
      cfg_name(),
      log_n,
      (arg_bytes + dyn_bytes) as f64 / 1e3,
    );
    return;
  }

  // KSWEEP=1: time (commit + prove) per k at the current LOG_T, verify each
  // (soundness gate), print, and return (skip criterion). Used to re-find the
  // optimal k after a LOG_T change.
  if std::env::var_os("KSWEEP").is_some() {
    use std::time::Instant;
    let dims = Dims::multiswap(0);
    let (shape, w, q, io) = ws::<M>(dims);
    println!(
      "\nMultiSwap k-sweep (LOG_T={LOG_T}, 2^{} rows):",
      (shape.num_cons() as u64).ilog2(),
    );
    for k in [9usize, 10, 11, 12, 13] {
      let params = params_for(&shape, k);
      let (sval, lpval, nl) = (params.s, params.log_p, params.numlimb);
      let (pk, vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
      let t0 = Instant::now();
      let (witness, instance) =
        IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w.clone(), q.clone(), io.clone()).unwrap();
      let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
      let ms = t0.elapsed().as_secs_f64() * 1e3;
      proof.verify(&vk, &instance).unwrap();
      println!("  k={k:<2} (s={sval}, log_p={lpval}, numlimb={nl}): commit+prove {ms:8.1} ms");
    }
    return;
  }

  if std::env::var_os("RUST_LOG").is_some() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();
    for &k in ks {
      let dims = Dims::multiswap(k);
      let (shape, w, q, io) = ws::<M>(dims);
      for int_k in 7..=10usize {
        let params = params_for(&shape, int_k);
        println!(
          "=== IntEval k={int_k}: log_p={} s={} numlimb={} numlimb_var={} (batch k={k}, cons=2^{}, vars=2^{}) ===",
          params.log_p,
          params.s,
          params.numlimb,
          params.numlimb_var,
          (shape.num_cons() as u64).ilog2(),
          (shape.num_vars() as u64).ilog2(),
        );
        let (pk, vk) =
          IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
        let (witness, instance) =
          IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w.clone(), q.clone(), io.clone())
            .unwrap();
        shape.is_sat(pk.ck(), &instance, &witness).unwrap();
        println!("is_sat passed");
        let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        println!("prove passed");
        proof.verify(&vk, &instance).unwrap();
        println!("verify passed");
      }
    }
  }

  for &k in ks {
    let dims = Dims::multiswap(k);
    let (shape, _w, _q, _io) = ws::<M>(dims);
    println!(
      "MultiSwap k={k}: num_cons=2^{} num_vars=2^{} (imod rows) vs paper F_p≈{} constraints",
      (shape.num_cons() as u64).ilog2(),
      (shape.num_vars() as u64).ilog2(),
      paper_fp_constraints(k),
    );
  }

  let mut g = c.benchmark_group("multiswap_modp");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(20));

  for &k in ks {
    let dims = Dims::multiswap(k);
    let (shape0, _, _, _) = ws::<M>(dims);
    let tag = format!(
      "{}_k{k}_c2^{}",
      cfg_name(),
      (shape0.num_cons() as u64).ilog2()
    );

    g.bench_function(format!("setup/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, _, _, _) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
          (shape, params)
        },
        |(shape, params)| {
          let _ = IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("advice/{tag}"), |b| {
      b.iter_batched(
        || {
          (
            exp_bases(),
            exp_exponents(dims.ell_bits),
            modulus_n(),
            modulus_ell(),
            modulus_p_hash(),
          )
        },
        |(bases, exponents, n, ell, p_hash)| {
          let _ = compute_witness_advice(&bases, &exponents, &n, &ell, &p_hash, dims);
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("commit_witness/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q, io) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
          let (pk, _vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
          (pk, shape, w, q, io)
        },
        |(pk, shape, w, q, io)| {
          let _ = IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    // Timed region = the full prover pipeline: witness generation
    // (`multiswap_shape_and_witness`, dominated by the real RSA-2048
    // exponentiation advice) + witness commitment + prove. The untimed
    // setup closure holds only the SNARK setup (PCS key derivation); the
    // shape it builds there is discarded except for the keys, and is
    // regenerated alongside the witness in the routine. `witness_advice`
    // and `commit_witness` above isolate the two pre-prove phases.
    g.bench_function(format!("prove/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, _, _, _) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
          let (pk, _vk) = IntModSpartanModpSNARK::<M>::setup_with_params(shape, params).unwrap();
          pk
        },
        |pk| {
          let (shape, w, q, io) = ws::<M>(dims);
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
          let _ = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).unwrap();
        },
        BatchSize::LargeInput,
      );
    });

    g.bench_function(format!("verify/{tag}"), |b| {
      b.iter_batched(
        || {
          let (shape, w, q, io) = ws::<M>(dims);
          let params = params_for(&shape, hyrax_k());
          let (pk, vk) =
            IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).unwrap();
          let (witness, instance) =
            IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, io).unwrap();
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

criterion_group!(benches, multiswap_modp_benches);
criterion_main!(benches);

fn statement_contract(metadata: &PaperMetadata) -> serde_json::Value {
  json!({
    "domain": "bitz-limber/multiswap-statement/v2", "digest_blake3": hex::encode(metadata.comparison_digest),
    "batch_count": metadata.batch_count, "public_input_count": 0, "public_inputs": [],
    "value_bits": 2048, "integer_domain": "unsigned",
    "public_roles": ["matrices", "moduli"], "private_roles": ["witness", "quotients"],
    "constant": 1, "padding": "zero-witness,zero-quotients,modulus-two",
    "live_rows": metadata.live_rows, "live_columns": metadata.live_cols,
    "padded_rows": metadata.live_rows.next_power_of_two(), "padded_columns": metadata.live_cols.next_power_of_two(),
  })
}

fn matched_security_targets() -> (usize, usize) {
  let target = env_usize("MATCHED_SECURITY_BITS", 114);
  assert!(matches!(target, 112 | 114), "matched comparison target must be 112 or 114");
  // Preserve Limber's native CRT target for the 114-bit comparison. The
  // 112-bit setting remains available solely for historical reproduction.
  let expected_integer_target = if target == 114 { 128 } else { 112 };
  let integer_target = env_usize("MATCHED_INTEGER_SECURITY_BITS", expected_integer_target);
  assert_eq!(integer_target, expected_integer_target, "matched integer target differs from comparison policy");
  (target, integer_target)
}

fn security_metadata(backend: &str, params: &IntEvalParams, log_n: usize) -> serde_json::Value {
  let (target, integer_target) = matched_security_targets();
  assert_eq!(params.security_bits, integer_target, "matched IntEval target differs from actual parameters");
  let fingerprint_bits = log2_prime_count(128) - ((8210u32 / 127) as f64).log2();
  let mut terms = vec![
    json!({"name": "fingerprint", "bits": fingerprint_bits}),
    json!({"name": "spartan-round", "bits": 127.0 - 3f64.log2()}),
    json!({"name": "spartan-batching", "bits": 127.0 - ((log_n+1) as f64).log2()}),
    json!({"name": "integer-crt", "bits": params.prime_security_bits(log_n)}),
    json!({"name": "integer-challenges", "bits": (params.log_q-1) as f64 - ((params.s*(log_n+params.numlimb_var)).max(1) as f64).log2()}),
  ];
  let bd_bits = env_usize("BDLAMBDA", 114);
  terms.push(json!({"name": "commitment-opening", "bits": if backend == "brakedown" { bd_bits as f64 } else { 128.0 }}));
  // Conservative public cap on all chunk and shifted-top lookup entries.
  // Checked against the actual range-check bitmap for every measured proof.
  // The rational identity's numerator degree is <= slots + table size;
  // GKR's two-variable prefix has total degree <=6.
  let slots = range_slot_bound(params, log_n);
  terms.push(json!({"name": "range-lookup", "bits": (params.log_q-1) as f64 - ((slots + (1u128<<16)) as f64).log2()}));
  terms.push(json!({"name": "range-gkr-round", "bits": (params.log_q-1) as f64 - 6f64.log2()}));
  terms.push(
    json!({"name": "range-batching", "bits": (params.log_q-1) as f64 - (slots as f64).log2()}),
  );
  let achieved = terms
    .iter()
    .map(|t| t["bits"].as_f64().unwrap())
    .fold(f64::INFINITY, f64::min);
  assert!(achieved >= target as f64, "matched security target unreachable");
  json!({
    "model": "per-check-round-minimum/v1", "profile": format!("limber{}", target),
    "target_bits": target, "achieved_bits": achieved, "terms": terms,
    "fingerprint_prime_bits": 128, "fingerprint_min": (BigUint::from(1u8)<<127usize).to_string(),
    "fingerprint_max": ((BigUint::from(1u8)<<128usize)-BigUint::from(1u8)).to_string(),
    "runtime_prime_policy": "verifier-sampled from Fiat-Shamir after W/Q commitments",
    "transcript_hash": "Keccak-256", "integer_target_bits": params.security_bits,
    "integer_challenge_target_bits": LAMBDA_BOUND2,
    "challenge_bits": 128, "brakedown_target_bits": if backend == "brakedown" { Some(bd_bits) } else { None },
    "key_format_version": params.format_version,
    "brakedown_configuration": if backend == "brakedown" { Some(limber::provider::pcs::brakedown_configuration()) } else { None },
    "log_q": params.log_q, "log_p": params.log_p, "small_primes": params.s,
    "log_t": params.log_t, "log_t_f": params.log_t_f, "int_k": params.k,
    "numlimb_var": params.numlimb_var, "range_slot_bound": slots.to_string(),
  })
}

fn log2_prime_count(bits: u32) -> f64 {
  let hi = (bits as f64) * std::f64::consts::LN_2;
  let lo = ((bits - 1) as f64) * std::f64::consts::LN_2;
  (bits as f64) + (1.0 / hi - 1.25506 / (2.0 * lo)).log2()
}

fn peak_rss_bytes() -> u64 {
  let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
  assert_eq!(
    unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
    0
  );
  let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
  if cfg!(target_os = "macos") {
    rss
  } else {
    rss * 1024
  }
}

fn range_slot_bound(params: &IntEvalParams, log_n: usize) -> u128 {
  // Two source polynomials, <=16 chunks per q-side value, and at most
  // four partial-evaluation value families per prime/coordinate. Deliberately
  // ignore shrinking tables, then double for power-of-two and shifted-top padding.
  4 * (1u128 << (log_n + params.numlimb_var))
    * 16
    * (1 + 4 * params.s as u128 * (log_n + params.numlimb_var + 1) as u128)
}
