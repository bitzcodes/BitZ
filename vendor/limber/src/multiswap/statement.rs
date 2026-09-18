//! Assembly of the MultiSwap statement as an Integer Mod-R1CS circuit.
//!
//! [`Config::Full`] is the OWWB20 `SetBench` statement for `t` swaps:
//! public inputs are the initial and final accumulator digests; the
//! circuit hashes the items, derives the Wesolowski challenge `ℓ` by the
//! Pocklington hash-to-prime over `(A, A', item hashes)`, reduces each
//! `H∆` modulo `ℓ`, and checks the two proofs of exponentiation
//! `Q^ℓ · A^r = D` and `Q'^ℓ · A'^{r'} = D` in the quotient group.
//! Every value is real dataflow from a generated instance; the builder
//! asserts every row as it is emitted.
//!
//! [`Config::Rsa`] is the RSA accumulator update kernel of the Garuda
//! and Zinc+ comparison rows: the two Wesolowski checks
//! `Q^ℓ · A^r = D` and `Q'^ℓ · A'^{r'} = D` with `A, A', D, ℓ, r, r'`
//! as public inputs — the verifier recomputes and checks the challenge
//! `ℓ` natively, outside the circuit, which is a complete protocol
//! whenever the element hashes are public.

use super::accumulator::{
  DIGEST_LIMB_BITS, DIGEST_LIMBS, MultiSwapInstance, PoE, RsaQuotientGroup, di_offset,
};
use super::circuit::{Builder, Built, Lc, Var};
use super::hash_gadgets::{mimc_permute, poseidon_hash};
use super::mimc;
use super::pocklington::PocklingtonCertificate;
use super::poseidon::{FR_CAPACITY, PoseidonParams, mod_inv};
use crate::errors::SpartanError;
use crate::traits::mod_engine::ModEngine;
use num_bigint::BigUint;
use num_traits::One;

/// Which statement to build.
#[derive(Clone, Debug)]
pub enum Config {
  /// The RSA accumulator update kernel: both Wesolowski checks with the
  /// digests, the common result, the challenge, and the reduced
  /// exponents as public inputs (challenge verified natively by the
  /// SNARK verifier).
  Rsa,
  /// The full OWWB20 statement with `swaps` removals and insertions.
  Full {
    /// Number of swaps.
    swaps: usize,
  },
}

/// Bit width allocated for the challenge `ℓ` and the reduced exponents.
pub const ELL_BITS: usize = 322;

/// Per-section row counts of a build, for reporting.
#[derive(Clone, Debug, Default)]
pub struct Sections(pub Vec<(String, usize)>);

impl Sections {
  fn mark(&mut self, b: &Builder, name: &str, start: usize) {
    self.0.push((name.to_string(), b.num_rows() - start));
  }
}

/// A built statement with its section breakdown and nonzero count.
pub struct Statement<M: ModEngine> {
  /// The finalized circuit.
  pub built: Built<M>,
  /// Rows per section.
  pub sections: Sections,
  /// Non-zero matrix entries.
  pub nnz: usize,
}

/// Bits of `bits` (LSB first) as MSB-first combinations.
fn msb_lcs(bits: &[Var]) -> Vec<Lc> {
  bits.iter().rev().map(|v| Lc::var(*v)).collect()
}

/// The entropy pool as circuit bits, consumed from the end like the
/// reference `EntropySource`.
struct PoolCursor {
  pool: Vec<Var>,
  next: usize,
}

impl PoolCursor {
  fn pop(&mut self) -> Var {
    self.next -= 1;
    self.pool[self.next]
  }

  /// `leading_ones` ones, `random_bits` popped bits (first = most
  /// significant), `trailing` zeros — as a combination, plus the popped
  /// bits MSB first.
  fn nat(&mut self, leading_ones: usize, random_bits: usize, trailing: usize) -> (Lc, Vec<Var>) {
    let mut lc = Lc::default();
    for i in 0..leading_ones {
      lc = lc.add_const(&(BigUint::one() << (random_bits + trailing + leading_ones - 1 - i)));
    }
    let mut bits = Vec::with_capacity(random_bits);
    for j in 0..random_bits {
      let b = self.pop();
      bits.push(b);
      lc = lc.add_term(b, BigUint::one() << (trailing + random_bits - 1 - j));
    }
    (lc, bits)
  }
}

/// Reduce `x` modulo the witness `ℓ`: `x = ℓ·k + r`, `r < ℓ`. Three rows.
fn reduce_by_witness(b: &mut Builder, x: &Lc, ell: &Lc) -> Var {
  let xv = b.eval(x);
  let lv = b.eval(ell);
  let (k, r) = (&xv / &lv, &xv % &lv);
  let k = b.alloc(k);
  let t = b.mul_exact(ell, &Lc::var(k));
  let r = b.alloc(r);
  b.assert_exact(
    x,
    &Lc::constant(BigUint::one()),
    &Lc::var(t).add_term(r, BigUint::one()),
  );
  b.assert_lt(&Lc::var(r), ell);
  r
}

/// The Wesolowski check `Q^ℓ · base^r = common` in the quotient group.
#[allow(clippy::too_many_arguments)]
fn poe_side(
  b: &mut Builder,
  group: &RsaQuotientGroup,
  base: Var,
  poe: &PoE,
  ell_bits_msb: &[Lc],
  r: &Lc,
  common: Var,
) {
  let n = &group.n;
  debug_assert_eq!(b.eval(r), poe.r);
  let r_bits = b.bits_of(r, ELL_BITS);
  let q = b.alloc(poe.q.clone());
  let one = Lc::constant(BigUint::one());
  let qm1 = b.sub(&Lc::var(q), &one);
  let ql = b.exp_var_base(&Lc::var(q), &Lc::var(qm1), ell_bits_msb, n, None);
  let qlc = b.canon(&Lc::var(ql), n);
  let bm1 = b.sub(&Lc::var(base), &one);
  let br = b.exp_var_base(&Lc::var(base), &Lc::var(bm1), &msb_lcs(&r_bits), n, None);
  let brc = b.canon(&Lc::var(br), n);
  let prod = b.mul_mod(&Lc::var(qlc), &Lc::var(brc), n);
  let pc = b.canon(&Lc::var(prod), n);
  b.assert_exact(&Lc::var(pc), &one, &Lc::var(common));
}

/// Prime widths along the certificate: the base prime, then after each
/// extension (from the plan's maxima, so they are prover-independent).
fn prime_widths(plan: &super::pocklington::PocklingtonPlan) -> Vec<usize> {
  let mut widths = vec![plan.base_bits()];
  let mut max = (BigUint::one() << plan.base_bits()) - BigUint::one();
  for e in &plan.extensions {
    max = max * e.max_value() + BigUint::one();
    widths.push(max.bits() as usize);
  }
  widths
}

/// The hash-to-prime certificate as rows; returns `ℓ` as a combination.
fn pocklington_rows(
  b: &mut Builder,
  cursor: &mut PoolCursor,
  plan: &super::pocklington::PocklingtonPlan,
  cert: &PocklingtonCertificate,
  sections: &mut Sections,
) -> Lc {
  let one = Lc::constant(BigUint::one());
  let widths = prime_widths(plan);

  // Base prime: `random = 1 ‖ r`, `prime = random ‖ nonce`, nonce ≡ 3 (mod 4).
  let start = b.num_rows();
  let (rand_lc, rand_bits) = cursor.nat(1, plan.base_random_bits, 0);
  debug_assert_eq!(b.eval(&rand_lc), cert.base_random);
  let nonce = b.alloc(BigUint::from(cert.base_nonce));
  let nb = b.bits_of(&Lc::var(nonce), plan.base_nonce_bits);
  b.assert_exact(&Lc::var(nb[0]), &one, &one);
  b.assert_exact(&Lc::var(nb[1]), &one, &one);
  let prime_lc = rand_lc
    .scale(&(BigUint::one() << plan.base_nonce_bits))
    .add_term(nonce, BigUint::one());
  debug_assert_eq!(b.eval(&prime_lc), cert.base_prime);
  // a = (prime − 1) / 2, MSB first: 1, random bits, nonce bits [msb..1].
  let mut a_bits: Vec<Lc> = vec![one.clone()];
  a_bits.extend(rand_bits.iter().map(|v| Lc::var(*v)));
  a_bits.extend(nb[1..].iter().rev().map(|v| Lc::var(*v)));
  for base in [2u32, 7, 61] {
    let pow = b.exp_const_base_witness_mod(&BigUint::from(base), &a_bits, &prime_lc);
    // (pow − 1) · (prime − 1 − pow) = 0.
    let d1 = b.sub(&Lc::var(pow), &one);
    let pv = b.value(pow);
    let d2 = b.alloc(b.eval(&prime_lc) - BigUint::one() - &pv);
    b.assert_exact(
      &Lc::var(d2)
        .add_term(pow, BigUint::one())
        .add_const(&BigUint::one()),
      &one,
      &prime_lc,
    );
    b.assert_exact(&Lc::var(d1), &Lc::var(d2), &Lc::default());
  }
  sections.mark(b, "hp_base_prime_mr32", start);

  let mut p_lc = prime_lc;
  for (i, (pe, ext)) in plan.extensions.iter().zip(&cert.extensions).enumerate() {
    let start = b.num_rows();
    let (r_lc, r_bits) = cursor.nat(0, pe.random_bits, 0);
    debug_assert_eq!(b.eval(&r_lc), ext.random);
    let nonce = b.alloc(BigUint::from(ext.nonce));
    let nb = b.bits_of(&Lc::var(nonce), pe.nonce_bits);
    let ext_lc = Lc::constant(BigUint::one() << (pe.nonce_bits + pe.random_bits))
      .plus(&r_lc.scale(&(BigUint::one() << pe.nonce_bits)))
      .add_term(nonce, BigUint::one());
    debug_assert_eq!(b.eval(&ext_lc), pe.evaluate(&ext.random, ext.nonce));
    // ext ≤ p + 1 (Pocklington's size condition).
    b.assert_lt(&ext_lc, &p_lc.clone().add_const(&BigUint::from(2u32)));
    let t = b.mul_exact(&p_lc, &ext_lc);
    let n_lc = Lc::var(t).add_const(&BigUint::one());
    debug_assert_eq!(b.eval(&n_lc), ext.result);
    // part = a^ext mod n.
    let mut ext_bits: Vec<Lc> = vec![one.clone()];
    ext_bits.extend(r_bits.iter().map(|v| Lc::var(*v)));
    ext_bits.extend(nb.iter().rev().map(|v| Lc::var(*v)));
    let a = b.alloc(ext.checking_base.clone());
    let am1 = b.sub(&Lc::var(a), &one);
    let part = b.exp_var_base(
      &Lc::var(a),
      &Lc::var(am1),
      &ext_bits,
      &BigUint::one(),
      Some(&n_lc),
    );
    // gcd(part − 1, n) = 1 via a pseudo-inverse: (part − 1)·inv = 1 + n·k.
    let pm1 = b.sub(&Lc::var(part), &one);
    let nv = b.eval(&n_lc);
    let pm1v = b.value(pm1);
    let inv_v = mod_inv(&(&pm1v % &nv), &nv);
    let k_v = (&pm1v * &inv_v - BigUint::one()) / &nv;
    let inv = b.alloc(inv_v);
    let k = b.alloc(k_v);
    let t2 = b.mul_exact(&n_lc, &Lc::var(k));
    b.assert_exact(
      &Lc::var(pm1),
      &Lc::var(inv),
      &Lc::var(t2).add_const(&BigUint::one()),
    );
    // part^p ≡ 1 (mod n).
    let p_bits = b.bits_of(&p_lc, widths[i]);
    let power = b.exp_var_base(
      &Lc::var(part),
      &Lc::var(pm1),
      &msb_lcs(&p_bits),
      &BigUint::one(),
      Some(&n_lc),
    );
    b.assert_exact(&Lc::var(power), &one, &one);
    p_lc = n_lc;
    sections.mark(b, &format!("hp_extension_{i}"), start);
  }
  debug_assert_eq!(b.eval(&p_lc), *cert.number());
  p_lc
}

/// Build the statement.
pub fn build<M: ModEngine>(
  cfg: &Config,
  poseidon: &PoseidonParams,
  segment: bool,
) -> Result<Statement<M>, SpartanError> {
  let mut b = Builder::new();
  if segment {
    b = b.with_width_segments();
  }
  let mut sections = Sections::default();
  let one = Lc::constant(BigUint::one());
  match cfg {
    Config::Rsa => {
      let inst = MultiSwapInstance::generate(poseidon, 0, 1);
      let group = &inst.group;
      // Public inputs; the challenge and reduced exponents are checked
      // natively by the verifier (hash + Pocklington certificate).
      let a_init = b.alloc_io(inst.initial.clone());
      let a_final = b.alloc_io(inst.final_digest.clone());
      let d = b.alloc_io(inst.common.clone());
      let ell = b.alloc_io(inst.ell().clone());
      let r_ins = b.alloc_io(inst.poe_insert.r.clone());
      let r_rem = b.alloc_io(inst.poe_remove.r.clone());
      let start = b.num_rows();
      let ell_bits = b.bits_of(&Lc::var(ell), ELL_BITS);
      let ell_bits_msb = msb_lcs(&ell_bits);
      sections.mark(&b, "ell_bits", start);
      let start = b.num_rows();
      poe_side(
        &mut b,
        group,
        a_init,
        &inst.poe_insert,
        &ell_bits_msb,
        &Lc::var(r_ins),
        d,
      );
      sections.mark(&b, "poe_insert", start);
      let start = b.num_rows();
      poe_side(
        &mut b,
        group,
        a_final,
        &inst.poe_remove,
        &ell_bits_msb,
        &Lc::var(r_rem),
        d,
      );
      sections.mark(&b, "poe_remove", start);
    }
    Config::Full { swaps } => {
      let inst = MultiSwapInstance::generate(poseidon, 0, *swaps);
      let group = &inst.group;
      let n = &group.n;
      let p = &poseidon.p;

      // Public inputs and their 16-bit chunks (block-checked), packed
      // into the transcript limbs.
      let start = b.num_rows();
      let a_init = b.alloc_io(inst.initial.clone());
      let a_final = b.alloc_io(inst.final_digest.clone());
      let pack = |b: &mut Builder, v: Var| -> Vec<Lc> {
        let chunks = b.chunks_of(&Lc::var(v), 128);
        (0..DIGEST_LIMBS)
          .map(|k| {
            let mut lc = Lc::default();
            for (j, c) in chunks.iter().enumerate().skip(15 * k).take(15) {
              lc = lc.add_term(*c, BigUint::one() << (16 * (j - 15 * k)));
            }
            lc
          })
          .collect()
      };
      let limbs_init = pack(&mut b, a_init);
      let limbs_final = pack(&mut b, a_final);
      debug_assert_eq!(b.eval(&limbs_init[0]), inst.challenge_inputs[0]);
      debug_assert_eq!(DIGEST_LIMB_BITS, 240);
      sections.mark(&b, "digest_chunks", start);

      // Items: Poseidon hash and the H∆ decomposition
      // `h = x + 2^254·b`, `x = Σ 2^{16j} c_j + 2^240·c15`, `c15 < 2^14`.
      let start = b.num_rows();
      let mut item_hashes: Vec<Var> = Vec::new();
      let mut item_x: Vec<Lc> = Vec::new();
      for (item, (h_val, x_val, _)) in inst
        .inserted
        .iter()
        .zip(&inst.inserted_hashes)
        .chain(inst.removed.iter().zip(&inst.removed_hashes))
      {
        let elems: Vec<Lc> = item.iter().map(|v| Lc::var(b.alloc(v.clone()))).collect();
        let h = poseidon_hash(&mut b, poseidon, &elems);
        debug_assert_eq!(&b.value(h), h_val);
        let mask = BigUint::from(0xffffu32);
        let chunks: Vec<Var> = (0..15)
          .map(|j| b.alloc_small((x_val >> (16 * j)) & &mask))
          .collect();
        let c15 = b.alloc((x_val >> 240) & BigUint::from((1u32 << 14) - 1));
        let _c15_bits = b.bits_of(&Lc::var(c15), 14);
        let top = b.alloc_bit(h_val.bit(FR_CAPACITY as u64));
        let mut x_lc = Lc::default();
        for (j, c) in chunks.iter().enumerate() {
          x_lc = x_lc.add_term(*c, BigUint::one() << (16 * j));
        }
        x_lc = x_lc.add_term(c15, BigUint::one() << 240);
        debug_assert_eq!(&b.eval(&x_lc), x_val);
        b.assert_exact(
          &x_lc.clone().add_term(top, BigUint::one() << FR_CAPACITY),
          &one,
          &Lc::var(h),
        );
        item_hashes.push(h);
        item_x.push(x_lc);
      }
      sections.mark(&b, "item_hashes", start);

      // Challenge hash over (A limbs, A' limbs, insertion hashes, removal hashes).
      let start = b.num_rows();
      let mut inputs: Vec<Lc> = Vec::new();
      inputs.extend(limbs_init);
      inputs.extend(limbs_final);
      inputs.extend(item_hashes.iter().map(|h| Lc::var(*h)));
      let h0 = poseidon_hash(&mut b, poseidon, &inputs);
      debug_assert_eq!(b.value(h0), inst.entropy.elems[0]);
      sections.mark(&b, "challenge_hash", start);

      // Entropy pool: [h0, MiMC(h0)], low 254 bits each.
      let start = b.num_rows();
      let keys = mimc::round_keys(p);
      let h1 = mimc_permute(&mut b, &keys, &Lc::var(h0), p);
      debug_assert_eq!(b.value(h1), inst.entropy.elems[1]);
      let mut pool = Vec::with_capacity(2 * FR_CAPACITY);
      for h in [h0, h1] {
        let bits = b.bits_of(&Lc::var(h), FR_CAPACITY + 1);
        pool.extend_from_slice(&bits[..FR_CAPACITY]);
      }
      let mut cursor = PoolCursor {
        next: pool.len(),
        pool,
      };
      sections.mark(&b, "entropy_pool", start);

      // Hash-to-prime certificate → ℓ.
      let ell_lc = pocklington_rows(
        &mut b,
        &mut cursor,
        &inst.plan,
        &inst.certificate,
        &mut sections,
      );
      let start = b.num_rows();
      let ell_bits = b.bits_of(&ell_lc, ELL_BITS);
      let ell_bits_msb = msb_lcs(&ell_bits);

      // H∆ mod ℓ per item: (x + OFFSET mod ℓ) mod ℓ.
      let off_red = reduce_by_witness(&mut b, &Lc::constant(di_offset()), &ell_lc);
      let reds: Vec<Var> = item_x
        .iter()
        .map(|x| {
          let sum = x.clone().add_term(off_red, BigUint::one());
          reduce_by_witness(&mut b, &sum, &ell_lc)
        })
        .collect();
      let (ins_reds, rem_reds) = reds.split_at(*swaps);
      let fold = |b: &mut Builder, reds: &[Var]| -> Lc {
        let mut acc = Lc::var(reds[0]);
        for r in &reds[1..] {
          acc = Lc::var(b.mul_mod_witness(&acc, &Lc::var(*r), &ell_lc));
        }
        acc
      };
      let r_ins = fold(&mut b, ins_reds);
      let r_rem = fold(&mut b, rem_reds);
      sections.mark(&b, "mod_ell_reductions", start);

      // The two proofs of exponentiation meet at the common digest.
      let common = b.alloc(inst.common.clone());
      let start = b.num_rows();
      poe_side(
        &mut b,
        group,
        a_init,
        &inst.poe_insert,
        &ell_bits_msb,
        &r_ins,
        common,
      );
      sections.mark(&b, "poe_insert", start);
      let start = b.num_rows();
      poe_side(
        &mut b,
        group,
        a_final,
        &inst.poe_remove,
        &ell_bits_msb,
        &r_rem,
        common,
      );
      sections.mark(&b, "poe_remove", start);
      let _ = n;
    }
  }
  let nnz = b.nnz;
  let built = b.finalize::<M>()?;
  Ok(Statement {
    built,
    sections,
    nnz,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256DynPrimeEngine;
  use crate::{
    imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
    imod_spartan_modp::IntModSpartanModpSNARK,
    provider::{
      T256DynPrimeBdEngine,
      pcs::integer_modpcs::{DEFAULT_K, IntEvalParams},
    },
  };

  /// The bench's IntEval parameters: 2048-bit norm bound, 64-bit limbs.
  fn params_for<M: ModEngine>(shape: &IntModR1CSShapeModp<M>) -> IntEvalParams {
    let n = shape.num_vars().max(shape.num_cons());
    IntEvalParams::derive(2048, 64, DEFAULT_K, n.trailing_zeros() as usize).unwrap()
  }

  /// The full statement is satisfied (`is_sat`), and a tampered witness
  /// — the prover's `Q` bumped by one — is not.
  /// Width histogram of the full-statement witness (--ignored). Decides
  /// how many width segments the commitment should group into and the
  /// achievable savings: chunks scale with sum(numlimb over values).
  #[test]
  #[ignore = "measurement"]
  fn full_witness_width_histogram() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<T256DynPrimeEngine>(&Config::Full { swaps: 1 }, &poseidon, false).unwrap();
    let w = &st.built.w;
    let block = st.built.block_len;
    // bucket by bit-width class
    let mut buckets = std::collections::BTreeMap::<usize, usize>::new();
    let mut nonzero = 0usize;
    for (i, v) in w.iter().enumerate() {
      let _ = v;
      let b = st.built.bounds[i].bits().saturating_sub(1) as usize; // bound 2^k -> k bits
      let cls = if b == 0 {
        0
      } else if b <= 16 {
        16
      } else if b <= 254 {
        254
      } else if b <= 322 {
        322
      } else if b <= 512 {
        512
      } else if b <= 1024 {
        1024
      } else {
        2048
      };
      *buckets.entry(cls).or_default() += 1;
      if b > 0 {
        nonzero += 1;
      }
    }
    println!(
      "full witness: {} w-values ({} nonzero), small block_len={}",
      w.len(),
      nonzero,
      block
    );
    for (cls, cnt) in &buckets {
      println!(
        "  <= {cls:>4} bits: {cnt:>6}  ({:.1}%)",
        100.0 * *cnt as f64 / w.len() as f64
      );
    }
    // chunks at uniform 2048 vs grouped-by-class (numlimb = ceil(width/64))
    let nl = |bits: usize| -> usize { bits.div_ceil(64).max(1) };
    let uniform: usize = w.len() * nl(2048);
    let grouped: usize = buckets
      .iter()
      .map(|(cls, cnt)| cnt * nl((*cls).max(1)))
      .sum();
    println!(
      "chunks (numlimb sum): uniform@2048 = {uniform}, grouped = {grouped}  => {:.2}x fewer",
      uniform as f64 / grouped as f64
    );
  }

  #[test]
  fn segmented_full_prove_verify() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<T256DynPrimeEngine>(&Config::Full { swaps: 1 }, &poseidon, true).unwrap();
    let (pk, vk) = IntModSpartanModpSNARK::<T256DynPrimeEngine>::setup_with_params(
      st.built.shape.clone(),
      params_for(&st.built.shape),
    )
    .unwrap();
    let (witness, instance) = IntModR1CSWitnessModp::<T256DynPrimeEngine>::new(
      &st.built.shape,
      pk.ck(),
      st.built.w.clone(),
      st.built.q.clone(),
      st.built.io.clone(),
    )
    .unwrap();
    assert!(
      instance.comm_w.len() > 1,
      "expected multiple segment commitments"
    );
    let proof =
      IntModSpartanModpSNARK::<T256DynPrimeEngine>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();
  }

  #[test]
  fn segmented_full_is_sat_and_tiles() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<T256DynPrimeEngine>(&Config::Full { swaps: 1 }, &poseidon, true).unwrap();
    let shape = &st.built.shape;
    let segs = shape.width_segments();
    assert!(!segs.is_empty(), "segmentation should produce segments");

    // Tile check: sorted, contiguous, aligned, cover [0, num_vars).
    let mut cursor = 0usize;
    let nl = |ltf: usize| ltf.div_ceil(64).max(1);
    let mut seg_chunks = 0usize;
    for s in segs {
      assert_eq!(s.start, cursor, "segment not contiguous");
      assert_eq!(s.start % s.size(), 0, "segment not aligned");
      seg_chunks += s.size() * nl(s.log_t_f);
      cursor += s.size();
    }
    assert_eq!(cursor, shape.num_vars(), "segments must cover all columns");
    let uniform_chunks = shape.num_vars() * nl(2048);
    println!(
      "segmented full: num_vars={}, {} segments, chunks {} vs uniform {} => {:.2}x fewer",
      shape.num_vars(),
      segs.len(),
      seg_chunks,
      uniform_chunks,
      uniform_chunks as f64 / seg_chunks as f64
    );
    for s in segs {
      println!(
        "  seg start={:>6} size={:>6} log_t_f={:>4}",
        s.start,
        s.size(),
        s.log_t_f
      );
    }

    // is_sat still holds under the reordered layout.
    let (pk, _vk) = IntModSpartanModpSNARK::<T256DynPrimeEngine>::setup_with_params(
      shape.clone(),
      params_for(shape),
    )
    .unwrap();
    let (witness, instance) = IntModR1CSWitnessModp::<T256DynPrimeEngine>::new(
      shape,
      pk.ck(),
      st.built.w.clone(),
      st.built.q.clone(),
      st.built.io.clone(),
    )
    .unwrap();
    shape.is_sat(pk.ck(), &instance, &witness).unwrap();
  }

  #[test]
  fn full_statement_is_sat_and_rejects_tamper() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<T256DynPrimeEngine>(&Config::Full { swaps: 1 }, &poseidon, false).unwrap();
    let (pk, _vk) = IntModSpartanModpSNARK::<T256DynPrimeEngine>::setup_with_params(
      st.built.shape.clone(),
      params_for(&st.built.shape),
    )
    .unwrap();
    let (witness, instance) = IntModR1CSWitnessModp::<T256DynPrimeEngine>::new(
      &st.built.shape,
      pk.ck(),
      st.built.w.clone(),
      st.built.q.clone(),
      st.built.io.clone(),
    )
    .unwrap();
    st.built.shape.is_sat(pk.ck(), &instance, &witness).unwrap();
    // Tamper: bump every witness value in turn is too slow; bump one
    // 2048-bit value (the largest witness is a PoE `Q` or a quotient).
    let mut w = st.built.w.clone();
    let idx = (0..w.len()).max_by_key(|&i| w[i].bits()).unwrap();
    w[idx] += BigUint::one();
    let (witness, instance) = IntModR1CSWitnessModp::<T256DynPrimeEngine>::new(
      &st.built.shape,
      pk.ck(),
      w,
      st.built.q.clone(),
      st.built.io.clone(),
    )
    .unwrap();
    assert!(st.built.shape.is_sat(pk.ck(), &instance, &witness).is_err());
  }

  /// End-to-end prove/verify of the full statement in hash mode
  /// (Brakedown). Slow (seconds); run with `--ignored`.
  #[test]
  #[ignore]
  fn full_statement_proves_hash_mode() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<T256DynPrimeBdEngine>(&Config::Full { swaps: 1 }, &poseidon, false).unwrap();
    let (pk, vk) = IntModSpartanModpSNARK::<T256DynPrimeBdEngine>::setup_with_params(
      st.built.shape.clone(),
      params_for(&st.built.shape),
    )
    .unwrap();
    let (witness, instance) = IntModR1CSWitnessModp::<T256DynPrimeBdEngine>::new(
      &st.built.shape,
      pk.ck(),
      st.built.w.clone(),
      st.built.q.clone(),
      st.built.io.clone(),
    )
    .unwrap();
    let proof =
      IntModSpartanModpSNARK::<T256DynPrimeBdEngine>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();
    // The proof must be bound to the public digests: change A' by one
    // and the same proof must be rejected.
    let mut bad = instance.clone();
    bad.x[1] += BigUint::one();
    assert!(proof.verify(&vk, &bad).is_err());
    // And to the initial digest.
    let mut bad = instance.clone();
    bad.x[0] -= BigUint::one();
    assert!(proof.verify(&vk, &bad).is_err());
  }

  #[test]
  fn rsa_statement_builds_in_2_13_and_is_sat() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<T256DynPrimeEngine>(&Config::Rsa, &poseidon, false).unwrap();
    eprintln!(
      "rsa: rows {} cols {} nnz {}",
      st.built.real_rows, st.built.real_cols, st.nnz
    );
    assert!(st.built.shape.num_cons() <= 8192);
    assert!(st.built.shape.num_vars() <= 8192);
    assert_eq!(st.built.io.len(), 6);
    let params = params_for(&st.built.shape);
    let (pk, _vk) = IntModSpartanModpSNARK::<T256DynPrimeEngine>::setup_with_params(
      st.built.shape.clone(),
      params,
    )
    .unwrap();
    let (witness, instance) = IntModR1CSWitnessModp::<T256DynPrimeEngine>::new(
      &st.built.shape,
      pk.ck(),
      st.built.w.clone(),
      st.built.q.clone(),
      st.built.io.clone(),
    )
    .unwrap();
    st.built.shape.is_sat(pk.ck(), &instance, &witness).unwrap();
  }

  #[test]
  fn full_statement_builds_in_2_14() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let st = build::<T256DynPrimeEngine>(&Config::Full { swaps: 1 }, &poseidon, false).unwrap();
    for (name, rows) in &st.sections.0 {
      eprintln!("{name:>24}: {rows}");
    }
    eprintln!(
      "rows {} cols {} block {} nnz {}",
      st.built.real_rows, st.built.real_cols, st.built.block_len, st.nnz
    );
    assert_eq!(st.built.io.len(), 2);
    assert!(
      st.built.shape.num_cons() <= 16384,
      "rows {}",
      st.built.real_rows
    );
    assert!(
      st.built.shape.num_vars() <= 16384,
      "cols {}",
      st.built.real_cols
    );
  }
}
