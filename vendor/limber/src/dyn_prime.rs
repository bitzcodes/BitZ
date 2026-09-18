//! Dynamic-modulus prime field `DynPrime<LIMBS>`, backed by
//! `crypto_bigint::modular::FixedMontyForm<LIMBS>`. The runtime modulus
//! is sampled by the verifier and carried inside each value via its
//! `FixedMontyParams<LIMBS>` context.
//!
//! Used by the dual-field IntMod-Spartan driver (`imod_spartan_modp`)
//! when the SNARK arithmetic happens modulo a verifier-sampled prime
//! that isn't known at compile time.

use crate::traits::{mod_engine::SumcheckField, transcript::TranscriptReprTrait};
use crypto_bigint::{
  Uint,
  modular::{FixedMontyForm, FixedMontyParams},
};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A field element in a runtime-modulus prime field, stored in Montgomery
/// form. The modulus (and the Montgomery constants derived from it) lives
/// inside the wrapped `FixedMontyForm`'s `FixedMontyParams`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynPrime<const LIMBS: usize> {
  inner: FixedMontyForm<LIMBS>,
}

impl<const LIMBS: usize> DynPrime<LIMBS> {
  /// Build from a canonical (non-Montgomery) `Uint` value and the modulus
  /// context. The constructor performs the Montgomery conversion.
  pub fn new(value: Uint<LIMBS>, params: &FixedMontyParams<LIMBS>) -> Self {
    Self {
      inner: FixedMontyForm::new(&value, params),
    }
  }

  /// Wrap an already-Montgomery-form `FixedMontyForm` directly.
  pub(crate) fn from_inner(inner: FixedMontyForm<LIMBS>) -> Self {
    Self { inner }
  }

  /// Extract the canonical (non-Montgomery) representation.
  pub fn retrieve(&self) -> Uint<LIMBS> {
    self.inner.retrieve()
  }

  /// Reference to the modulus context carried by this value.
  pub fn params(&self) -> &FixedMontyParams<LIMBS> {
    self.inner.params()
  }
}

// ---- Arithmetic ops -------------------------------------------------------

impl<const LIMBS: usize> Add for DynPrime<LIMBS> {
  type Output = Self;
  fn add(self, rhs: Self) -> Self {
    Self::from_inner(self.inner + rhs.inner)
  }
}

impl<const LIMBS: usize> Sub for DynPrime<LIMBS> {
  type Output = Self;
  fn sub(self, rhs: Self) -> Self {
    Self::from_inner(self.inner - rhs.inner)
  }
}

impl<const LIMBS: usize> Mul for DynPrime<LIMBS> {
  type Output = Self;
  fn mul(self, rhs: Self) -> Self {
    Self::from_inner(self.inner * rhs.inner)
  }
}

impl<const LIMBS: usize> Neg for DynPrime<LIMBS> {
  type Output = Self;
  fn neg(self) -> Self {
    Self::from_inner(-self.inner)
  }
}

impl<const LIMBS: usize> AddAssign for DynPrime<LIMBS> {
  fn add_assign(&mut self, rhs: Self) {
    self.inner += rhs.inner;
  }
}

impl<const LIMBS: usize> SubAssign for DynPrime<LIMBS> {
  fn sub_assign(&mut self, rhs: Self) {
    self.inner -= rhs.inner;
  }
}

impl<const LIMBS: usize> MulAssign for DynPrime<LIMBS> {
  fn mul_assign(&mut self, rhs: Self) {
    self.inner *= rhs.inner;
  }
}

// ---- SumcheckField --------------------------------------------------------

impl<const LIMBS: usize> SumcheckField for DynPrime<LIMBS> {
  type Params = FixedMontyParams<LIMBS>;

  fn zero(params: &Self::Params) -> Self {
    Self::from_inner(FixedMontyForm::zero(params))
  }

  fn one(params: &Self::Params) -> Self {
    Self::from_inner(FixedMontyForm::one(params))
  }

  fn from_u64(params: &Self::Params, v: u64) -> Self {
    Self::new(Uint::from(v), params)
  }

  fn invert(&self) -> Option<Self> {
    Option::<FixedMontyForm<LIMBS>>::from(self.inner.invert()).map(Self::from_inner)
  }

  fn to_le_bytes(&self) -> Vec<u8> {
    self.retrieve().to_le_bytes().as_slice().to_vec()
  }

  fn is_in_context(&self, expected: &Self::Params) -> bool {
    self.params().modulus() == expected.modulus()
  }

  fn from_bytes_reduce(params: &Self::Params, bytes: &[u8]) -> Self {
    let n = Uint::<LIMBS>::BYTES;
    if bytes.len() <= n {
      let mut buf = vec![0u8; n];
      buf[..bytes.len()].copy_from_slice(bytes);
      let value = Uint::<LIMBS>::from_le_slice(&buf);
      return Self::new(value, params);
    }
    // Value exceeds LIMBS limbs — Horner reduction over n-byte chunks.
    // V = c_0 + c_1·B + c_2·B² + …  where B = 2^(n·8).
    let two_32 = Self::from_u64(params, 1u64 << 32);
    let two_64 = two_32 * two_32;
    let mut shift = Self::one(params);
    for _ in 0..LIMBS {
      shift *= two_64;
    }
    let num_chunks = bytes.len().div_ceil(n);
    let mut acc = Self::zero(params);
    for ci in (0..num_chunks).rev() {
      let start = ci * n;
      let end = (start + n).min(bytes.len());
      let mut buf = vec![0u8; n];
      buf[..end - start].copy_from_slice(&bytes[start..end]);
      let chunk_value = Uint::<LIMBS>::from_le_slice(&buf);
      acc = acc * shift + Self::new(chunk_value, params);
    }
    acc
  }
}

// Absorbing a DynPrime into a ByteTranscript: hand over its canonical LE
// bytes. The dual-field SNARK driver needs this to absorb mods, IO, and the
// outer-SC claims into the shared Fiat-Shamir chain.
impl<const LIMBS: usize> TranscriptReprTrait for DynPrime<LIMBS> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_le_bytes()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crypto_bigint::{Odd, U256};

  #[test]
  #[ignore]
  fn width_microbench() {
    use crypto_bigint::U128;
    use std::time::Instant;
    // Same 128-bit prime (top bit set) in both carriers.
    let m128: U128 = U128::from_be_hex("ffffffffffffffffffffffffffffff61");
    let m256: U256 =
      U256::from_be_hex("00000000000000000000000000000000ffffffffffffffffffffffffffffff61");
    let p2 = FixedMontyParams::new(Odd::new(m128).unwrap());
    let p4 = FixedMontyParams::new(Odd::new(m256).unwrap());
    const ITERS: usize = 10_000_000;
    let mut a2 = DynPrime::<2>::new(U128::from(0x1234_5678_9abc_def0_u64), &p2);
    let b2 = DynPrime::<2>::new(U128::from(0x0fed_cba9_8765_4321_u64), &p2);
    let t = Instant::now();
    for _ in 0..ITERS {
      a2 *= b2;
    }
    let ns2 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a2);
    let mut a4 = DynPrime::<4>::new(U256::from(0x1234_5678_9abc_def0_u64), &p4);
    let b4 = DynPrime::<4>::new(U256::from(0x0fed_cba9_8765_4321_u64), &p4);
    let t = Instant::now();
    for _ in 0..ITERS {
      a4 *= b4;
    }
    let ns4 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a4);
    let t = Instant::now();
    for _ in 0..ITERS {
      a2 += b2;
    }
    let add2 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a2);
    let t = Instant::now();
    for _ in 0..ITERS {
      a4 += b4;
    }
    let add4 = t.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(a4);
    println!(
      "mul: DynPrime<2> {ns2:.2} ns  DynPrime<4> {ns4:.2} ns  ratio {:.2}x",
      ns4 / ns2
    );
    println!(
      "add: DynPrime<2> {add2:.2} ns  DynPrime<4> {add4:.2} ns  ratio {:.2}x",
      add4 / add2
    );
  }

  /// ff_derive fixed-modulus field over M127 = 2^127 − 1, to compare
  /// generic compile-time 2-limb Montgomery against DynPrime's
  /// runtime-modulus form and a hand-rolled Mersenne reduction.
  /// Generator 3 is a quadratic nonresidue mod M127 (p ≡ 7 mod 12).
  #[derive(ff::PrimeField)]
  #[PrimeFieldModulus = "170141183460469231731687303715884105727"]
  #[PrimeFieldGenerator = "3"]
  #[PrimeFieldReprEndianness = "little"]
  struct F127Derived([u64; 2]);

  /// Hand-rolled M127 = 2^127 − 1 arithmetic on a bare u128:
  /// schoolbook 128×128→256 via u64 halves, then the Mersenne fold
  /// X = hi·2^128 + lo ≡ 2·hi + (lo >> 127) + (lo & M) (mod 2^127−1).
  const M127: u128 = (1u128 << 127) - 1;
  #[inline(always)]
  fn m127_mul(a: u128, b: u128) -> u128 {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let ll = a0 * b0;
    let mid = a0 * b1 + a1 * b0; // < 2^128: each term < 2^127
    let hh = a1 * b1;
    let (lo, carry) = ll.overflowing_add(mid << 64);
    let hi = hh + (mid >> 64) + carry as u128; // < 2^126
    let mut r = (lo & M127) + (lo >> 127) + (hi << 1);
    r = (r & M127) + (r >> 127);
    if r >= M127 {
      r -= M127;
    }
    r
  }

  /// Four-way 128-bit field candidate microbench (dependent-chain mul
  /// latency + independent-slot mul throughput), all single-threaded.
  /// Run: cargo test --release field_128_candidates -- --ignored --nocapture
  #[test]
  #[ignore]
  fn field_128_candidates_microbench() {
    use crate::provider::pt256::t256;
    use crypto_bigint::U128;
    use ff::{Field, PrimeField as _};
    use rand_core::OsRng;
    use std::time::Instant;

    const CHAIN: usize = 10_000_000;
    const SLOTS: usize = 4096;
    const PASSES: usize = 4096; // SLOTS × PASSES ≈ 16.8M muls

    // --- t256::Scalar (4-limb fixed Montgomery, today's q-side) ---
    let mut x = t256::Scalar::random(OsRng);
    let y = t256::Scalar::random(OsRng);
    let t = Instant::now();
    for _ in 0..CHAIN {
      x *= y;
    }
    let t256_lat = t.elapsed().as_nanos() as f64 / CHAIN as f64;
    std::hint::black_box(x);
    let mut av: Vec<t256::Scalar> = (0..SLOTS).map(|_| t256::Scalar::random(OsRng)).collect();
    let bv: Vec<t256::Scalar> = (0..SLOTS).map(|_| t256::Scalar::random(OsRng)).collect();
    let t = Instant::now();
    for _ in 0..PASSES {
      for i in 0..SLOTS {
        av[i] *= bv[i];
      }
    }
    let t256_tp = t.elapsed().as_nanos() as f64 / (SLOTS * PASSES) as f64;
    std::hint::black_box(&av);

    // --- DynPrime<2> over M127 (runtime-modulus 2-limb Montgomery) ---
    let m: U128 = U128::from_be_hex("7fffffffffffffffffffffffffffffff");
    let params = FixedMontyParams::new(Odd::new(m).unwrap());
    let mut x = DynPrime::<2>::new(U128::from(0x1234_5678_9abc_def0_u64), &params);
    let y = DynPrime::<2>::new(U128::from(0x0fed_cba9_8765_4321_u64), &params);
    let t = Instant::now();
    for _ in 0..CHAIN {
      x *= y;
    }
    let dyn_lat = t.elapsed().as_nanos() as f64 / CHAIN as f64;
    std::hint::black_box(x);
    let mut av: Vec<DynPrime<2>> = (0..SLOTS)
      .map(|i| DynPrime::<2>::new(U128::from(0x9e37_79b9_7f4a_7c15_u64 ^ i as u64), &params))
      .collect();
    let bv: Vec<DynPrime<2>> = (0..SLOTS)
      .map(|i| DynPrime::<2>::new(U128::from(0xc2b2_ae3d_27d4_eb4f_u64 ^ i as u64), &params))
      .collect();
    let t = Instant::now();
    for _ in 0..PASSES {
      for i in 0..SLOTS {
        av[i] *= bv[i];
      }
    }
    let dyn_tp = t.elapsed().as_nanos() as f64 / (SLOTS * PASSES) as f64;
    std::hint::black_box(&av);

    // --- ff_derive F127 (compile-time 2-limb Montgomery, same prime) ---
    let mut x = F127Derived::random(OsRng);
    let y = F127Derived::random(OsRng);
    let t = Instant::now();
    for _ in 0..CHAIN {
      x *= y;
    }
    let drv_lat = t.elapsed().as_nanos() as f64 / CHAIN as f64;
    std::hint::black_box(x);
    let mut av: Vec<F127Derived> = (0..SLOTS).map(|_| F127Derived::random(OsRng)).collect();
    let bv: Vec<F127Derived> = (0..SLOTS).map(|_| F127Derived::random(OsRng)).collect();
    let t = Instant::now();
    for _ in 0..PASSES {
      for i in 0..SLOTS {
        av[i] *= bv[i];
      }
    }
    let drv_tp = t.elapsed().as_nanos() as f64 / (SLOTS * PASSES) as f64;
    std::hint::black_box(&av);

    // --- hand-rolled M127 (u128 + Mersenne fold, no Montgomery) ---
    // Correctness spot-check against the derived field first.
    for (a, b) in [(3u128, 5u128), (M127 - 1, M127 - 1), (1u128 << 126, 12345)] {
      let expect = {
        let fa = F127Derived::from_u128(a);
        let fb = F127Derived::from_u128(b);
        let mut le = [0u8; 16];
        le.copy_from_slice((fa * fb).to_repr().as_ref());
        u128::from_le_bytes(le)
      };
      assert_eq!(m127_mul(a, b), expect, "m127_mul({a}, {b})");
    }
    let mut x: u128 = 0x1234_5678_9abc_def0;
    let y: u128 = (1 << 126) | 0x0fed_cba9_8765_4321;
    let t = Instant::now();
    for _ in 0..CHAIN {
      x = m127_mul(x, y);
    }
    let m127_lat = t.elapsed().as_nanos() as f64 / CHAIN as f64;
    std::hint::black_box(x);
    let mut av: Vec<u128> = (0..SLOTS)
      .map(|i| ((i as u128) << 64) | 0xdead_beef)
      .collect();
    let bv: Vec<u128> = (0..SLOTS)
      .map(|i| (1u128 << 126) | ((i as u128) * 0x9e37_79b9))
      .collect();
    let t = Instant::now();
    for _ in 0..PASSES {
      for i in 0..SLOTS {
        av[i] = m127_mul(av[i], bv[i]);
      }
    }
    let m127_tp = t.elapsed().as_nanos() as f64 / (SLOTS * PASSES) as f64;
    std::hint::black_box(&av);

    println!("mul latency   (dependent chain, ns/op):");
    println!("  t256 (4-limb)      {t256_lat:.2}");
    println!(
      "  DynPrime<2>        {dyn_lat:.2}   ({:.2}x vs t256)",
      t256_lat / dyn_lat
    );
    println!(
      "  ff_derive F127     {drv_lat:.2}   ({:.2}x vs t256)",
      t256_lat / drv_lat
    );
    println!(
      "  hand M127          {m127_lat:.2}   ({:.2}x vs t256)",
      t256_lat / m127_lat
    );
    println!("mul throughput (independent slots, ns/op):");
    println!("  t256 (4-limb)      {t256_tp:.2}");
    println!(
      "  DynPrime<2>        {dyn_tp:.2}   ({:.2}x vs t256)",
      t256_tp / dyn_tp
    );
    println!(
      "  ff_derive F127     {drv_tp:.2}   ({:.2}x vs t256)",
      t256_tp / drv_tp
    );
    println!(
      "  hand M127          {m127_tp:.2}   ({:.2}x vs t256)",
      t256_tp / m127_tp
    );
  }

  /// NTT-encode cost gate for the RS-code PCS direction (WHIR/Basefold
  /// over an FFT-friendly q-side field). Radix-2 in-place NTT over
  /// bn254::Fr (two-adicity 28, already a dependency) at Brakedown-
  /// comparable sizes; compare against the measured expander encode
  /// (~130 ms at 2^20 with rate 1.64).
  /// Run: cargo test --release ntt_encode_gate -- --ignored --nocapture
  #[test]
  #[ignore]
  fn ntt_encode_gate() {
    use ff::{Field, PrimeField};
    use halo2curves::bn256::Fr;
    use rand_core::OsRng;
    use std::time::Instant;

    fn ntt(a: &mut [Fr], omega: Fr) {
      let n = a.len();
      let log_n = n.trailing_zeros();
      // bit-reversal permutation
      for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - log_n);
        if (j as usize) > i {
          a.swap(i, j as usize);
        }
      }
      let mut len = 2;
      while len <= n {
        let w_len = omega.pow_vartime([(n / len) as u64]);
        for start in (0..n).step_by(len) {
          let mut w = Fr::ONE;
          for k in 0..len / 2 {
            let u = a[start + k];
            let v = a[start + k + len / 2] * w;
            a[start + k] = u + v;
            a[start + k + len / 2] = u - v;
            w *= w_len;
          }
        }
        len <<= 1;
      }
    }

    for log_n in [20usize, 21] {
      let n = 1usize << log_n;
      // 2^log_n-th root of unity: g^{(q-1)/2^log_n} for generator-derived g
      let omega = Fr::ROOT_OF_UNITY.pow_vartime([1u64 << (Fr::S - log_n as u32)]);
      assert_eq!(omega.pow_vartime([n as u64]), Fr::ONE);
      let mut a: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
      let t = Instant::now();
      ntt(&mut a, omega);
      let ms = t.elapsed().as_secs_f64() * 1e3;
      std::hint::black_box(&a);
      println!("NTT bn254::Fr 2^{log_n}: {ms:.1} ms single-thread");
    }
  }

  // Small Mersenne prime 2^61 - 1, easy to verify against u128 arithmetic.
  fn test_params() -> FixedMontyParams<4> {
    let modulus: U256 = U256::from(0x1fff_ffff_ffff_ffff_u64);
    FixedMontyParams::new(Odd::new(modulus).unwrap())
  }

  #[test]
  fn add_mul_reduce_mod_modulus() {
    let p = test_params();
    let p_int: u128 = 0x1fff_ffff_ffff_ffff;

    let a_int: u64 = 0x1234_5678_9abc_def0;
    let b_int: u64 = 0x0fed_cba9_8765_4321;
    let a = DynPrime::<4>::from_u64(&p, a_int);
    let b = DynPrime::<4>::from_u64(&p, b_int);

    let sum_int = ((a_int as u128) + (b_int as u128)) % p_int;
    let sum_low = u64::from_le_bytes((a + b).to_le_bytes()[..8].try_into().unwrap());
    assert_eq!(sum_low as u128, sum_int);

    let prod_int = ((a_int as u128) * (b_int as u128)) % p_int;
    let prod_low = u64::from_le_bytes((a * b).to_le_bytes()[..8].try_into().unwrap());
    assert_eq!(prod_low as u128, prod_int);
  }

  #[test]
  fn neg_then_add_gives_zero() {
    let p = test_params();
    let a = DynPrime::<4>::from_u64(&p, 12345);
    let zero = DynPrime::<4>::zero(&p);
    assert_eq!(a + (-a), zero);
  }

  #[test]
  fn invert_then_mul_gives_one() {
    let p = test_params();
    let a = DynPrime::<4>::from_u64(&p, 7);
    let a_inv = a.invert().unwrap();
    assert_eq!(a * a_inv, DynPrime::<4>::one(&p));
  }

  #[test]
  fn zero_has_no_inverse() {
    let p = test_params();
    let zero = DynPrime::<4>::zero(&p);
    assert!(zero.invert().is_none());
  }

  #[test]
  fn from_bytes_reduce_handles_wide_values() {
    let p = test_params();
    // 64-byte value (512 bits) — wider than Uint<4> (256 bits).
    let mut bytes = vec![0u8; 64];
    bytes[32] = 1; // represents 2^256
    let result = DynPrime::<4>::from_bytes_reduce(&p, &bytes);
    // Verify by computing 2^256 mod p via repeated squaring in DynPrime
    let two = DynPrime::<4>::from_u64(&p, 2);
    let mut pow = DynPrime::<4>::one(&p);
    for _ in 0..256 {
      pow *= two;
    }
    assert_eq!(
      result, pow,
      "from_bytes_reduce should correctly reduce values wider than LIMBS"
    );
  }
}
