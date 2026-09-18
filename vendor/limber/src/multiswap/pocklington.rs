//! Hash-to-prime via recursive Pocklington certificates, ported from the
//! OWWB20 reference (`bellman-bignat`, `hash/pocklington`). A plan for
//! `entropy` random bits builds a ~32-bit base prime from the entropy
//! pool and then extends it: `p' = p · (1 ‖ random ‖ nonce) + 1`, each
//! extension certified by a base `a` with `a^{p'−1} ≡ 1 (mod p')` and
//! `gcd(a^{ext} − 1, p') = 1`. For 256 bits of entropy the plan is a
//! base prime plus four extensions and the result has at most 322 bits.
//!
//! The entropy pool is a Poseidon hash of the inputs stretched with
//! MiMC-7 permutations; bits are consumed from the END of the pool
//! (most-significant end of the last element first), exactly as the
//! reference's `EntropySource::get_bit` (`Vec::pop`).

use super::mimc;
use super::poseidon::{FR_CAPACITY, PoseidonParams};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

/// Probability that a random `bits`-bit number is prime.
fn prime_density(bits: usize) -> f64 {
  let log2e = std::f64::consts::E.log2();
  let b = bits as f64;
  log2e / b - log2e * log2e / b / b
}

/// Number of random `bits`-bit numbers that must be checked to find a
/// prime with all but `p_fail` probability.
pub fn prime_trials(bits: usize, p_fail: f64) -> usize {
  let p = prime_density(bits);
  (p_fail.log(1.0 - p).ceil() + 0.1) as usize
}

/// Nonce bits needed to generate a `bits`-bit prime with all but `2^-64`
/// probability.
pub fn nonce_bits_needed(bits: usize) -> usize {
  let trials = prime_trials(bits, 2.0f64.powi(-64));
  ((trials as f64).log2().ceil() + 0.1) as usize
}

/// One planned extension: `ext = 1 ‖ random_bits ‖ nonce_bits`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedExtension {
  /// Width of the nonce field.
  pub nonce_bits: usize,
  /// Width of the random field.
  pub random_bits: usize,
}

impl PlannedExtension {
  /// `2^(nonce_bits + random_bits)` — the leading one.
  pub fn min_value(&self) -> BigUint {
    BigUint::one() << (self.nonce_bits + self.random_bits)
  }
  /// The largest extension value.
  pub fn max_value(&self) -> BigUint {
    (BigUint::one() << (self.nonce_bits + self.random_bits + 1)) - BigUint::one()
  }
  /// `1 ‖ random ‖ nonce`.
  pub fn evaluate(&self, random: &BigUint, nonce: u64) -> BigUint {
    self.min_value() + (random << self.nonce_bits) + BigUint::from(nonce)
  }
  /// Bit width of the extension value.
  pub fn bits(&self) -> usize {
    self.nonce_bits + self.random_bits + 1
  }
}

/// The certification plan for a target entropy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PocklingtonPlan {
  /// Nonce bits of the base prime (its two low bits are forced to 1).
  pub base_nonce_bits: usize,
  /// Random bits of the base prime (below a forced leading one).
  pub base_random_bits: usize,
  /// The extensions, in order.
  pub extensions: Vec<PlannedExtension>,
}

impl PocklingtonPlan {
  /// The reference's plan for `entropy` random bits.
  pub fn new(entropy: usize) -> Self {
    let nonce_bits_in_base = nonce_bits_needed(32) + 1;
    let mut plan = Self {
      base_nonce_bits: nonce_bits_in_base,
      base_random_bits: entropy.min(31 - nonce_bits_in_base),
      extensions: Vec::new(),
    };
    while plan.entropy() < entropy {
      let max_extension_bits = plan.min_value().bits() as usize - 1;
      let max_nonce_bits_needed = nonce_bits_needed(max_extension_bits + plan.max_bits());
      assert!(max_nonce_bits_needed < max_extension_bits);
      let max_random_bits = max_extension_bits - max_nonce_bits_needed - 1;
      let random_bits = (entropy - plan.entropy()).min(max_random_bits);
      plan.extensions.push(PlannedExtension {
        nonce_bits: max_nonce_bits_needed,
        random_bits,
      });
    }
    plan
  }

  /// Total random bits consumed.
  pub fn entropy(&self) -> usize {
    self.extensions.iter().map(|e| e.random_bits).sum::<usize>() + self.base_random_bits
  }

  /// Largest possible prime.
  pub fn max_value(&self) -> BigUint {
    self.extensions.iter().fold(
      (BigUint::one() << (self.base_random_bits + self.base_nonce_bits + 1)) - BigUint::one(),
      |acc, e| acc * e.max_value() + BigUint::one(),
    )
  }

  /// Smallest possible prime.
  pub fn min_value(&self) -> BigUint {
    self.extensions.iter().fold(
      BigUint::one() << (self.base_random_bits + self.base_nonce_bits),
      |acc, e| acc * e.min_value() + BigUint::one(),
    )
  }

  /// Bit width of the largest possible prime.
  pub fn max_bits(&self) -> usize {
    self.max_value().bits() as usize
  }

  /// Bit width of the base prime.
  pub fn base_bits(&self) -> usize {
    self.base_random_bits + self.base_nonce_bits + 1
  }
}

/// One certified extension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PocklingtonExtension {
  /// The plan entry.
  pub plan: PlannedExtension,
  /// The random field (from the entropy pool).
  pub random: BigUint,
  /// The nonce found by search.
  pub nonce: u64,
  /// The certifying base `a`.
  pub checking_base: BigUint,
  /// The resulting prime `p · ext + 1`.
  pub result: BigUint,
}

/// A recursive Pocklington certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PocklingtonCertificate {
  /// The base prime's random field (with its leading one).
  pub base_random: BigUint,
  /// The base prime `random ‖ nonce` (`nonce ≡ 3 mod 4`).
  pub base_prime: BigUint,
  /// The base nonce.
  pub base_nonce: u64,
  /// The extensions, in order.
  pub extensions: Vec<PocklingtonExtension>,
}

impl PocklingtonCertificate {
  /// The certified prime.
  pub fn number(&self) -> &BigUint {
    self
      .extensions
      .last()
      .map(|e| &e.result)
      .unwrap_or(&self.base_prime)
  }
}

/// The reference entropy pool: field elements `[h, MiMC(h), MiMC²(h), …]`
/// each contributing its low `FR_CAPACITY` bits (LSB first); bits are
/// consumed from the end.
#[derive(Clone, Debug)]
pub struct EntropySource {
  /// The pool elements, in order.
  pub elems: Vec<BigUint>,
  bits: Vec<bool>,
}

impl EntropySource {
  /// Build the pool from a hash for `bits_needed` bits.
  pub fn new(hash: BigUint, bits_needed: usize, p: &BigUint) -> Self {
    let elems_needed = (bits_needed - 1) / FR_CAPACITY + 1;
    let keys = mimc::round_keys(p);
    let mut elems = vec![hash];
    while elems.len() < elems_needed {
      let next = mimc::permute(elems.last().expect("non-empty"), &keys, p);
      elems.push(next);
    }
    let mut bits = Vec::with_capacity(elems_needed * FR_CAPACITY);
    for e in &elems {
      bits.extend((0..FR_CAPACITY as u64).map(|i| e.bit(i)));
    }
    Self { elems, bits }
  }

  /// Number of pool elements.
  pub fn num_elems(&self) -> usize {
    self.elems.len()
  }

  /// Pop one bit (from the end of the pool).
  pub fn get_bit(&mut self) -> bool {
    self.bits.pop().expect("entropy pool exhausted")
  }

  /// Index (into the flat pool) of the next bit to be consumed.
  pub fn next_index(&self) -> usize {
    self.bits.len() - 1
  }

  /// `leading_ones` ones, then `random_bits` popped bits (first popped =
  /// most significant), then `trailing` zeros.
  pub fn get_bits_as_nat(
    &mut self,
    leading_ones: usize,
    random_bits: usize,
    trailing: usize,
  ) -> BigUint {
    let mut acc = BigUint::zero();
    for _ in 0..leading_ones {
      acc = (acc << 1) + BigUint::one();
    }
    for _ in 0..random_bits {
      let b = self.get_bit();
      acc = (acc << 1) + BigUint::from(b as u8);
    }
    acc << trailing
  }
}

/// Miller–Rabin round with base `b` (the reference's `miller_rabin_round`).
pub fn miller_rabin_round(n: &BigUint, b: &BigUint) -> bool {
  if n <= &BigUint::from(2u32) || n.is_even() {
    return false;
  }
  let n_less_one = n - BigUint::one();
  let s = n_less_one.trailing_zeros().unwrap_or(0);
  let d = &n_less_one >> s;
  let mut pow = b.modpow(&d, n);
  if pow.is_one() || pow == n_less_one {
    return true;
  }
  for _ in 0..s.saturating_sub(1) {
    pow = (&pow * &pow) % n;
    if pow == n_less_one {
      return true;
    }
  }
  false
}

/// The reference's 32-bit primality test: bases 2, 7, 61.
pub fn miller_rabin_32b(n: &BigUint) -> bool {
  [2u32, 7, 61]
    .iter()
    .all(|b| miller_rabin_round(n, &BigUint::from(*b)))
}

/// Miller–Rabin with the first `rounds` primes as bases.
pub fn miller_rabin(n: &BigUint, rounds: usize) -> bool {
  let mut ps = vec![2u32];
  let mut next = 3u32;
  while ps.len() < rounds {
    if !ps.iter().any(|p| next.is_multiple_of(*p)) {
      ps.push(next);
    }
    next += 1;
  }
  ps.iter().all(|b| miller_rabin_round(n, &BigUint::from(*b)))
}

/// Search the base prime: `random = 1 ‖ r` then `(random ‖ nonce)` with
/// `nonce ≡ 3 (mod 4)` passing the 32-bit test.
pub fn attempt_base(
  plan: &PocklingtonPlan,
  src: &mut EntropySource,
) -> Option<PocklingtonCertificate> {
  let random = src.get_bits_as_nat(1, plan.base_random_bits, 0);
  for nonce in 0..(1u64 << plan.base_nonce_bits) {
    if nonce & 0b11 == 0b11 {
      let base = (&random << plan.base_nonce_bits) + BigUint::from(nonce);
      if miller_rabin_32b(&base) {
        return Some(PocklingtonCertificate {
          base_random: random,
          base_prime: base,
          base_nonce: nonce,
          extensions: Vec::new(),
        });
      }
    }
  }
  None
}

/// Search one extension (the reference's `attempt_pocklington_extension`).
pub fn attempt_extension(
  mut cert: PocklingtonCertificate,
  plan: &PlannedExtension,
  src: &mut EntropySource,
) -> Option<PocklingtonCertificate> {
  let random = src.get_bits_as_nat(0, plan.random_bits, 0);
  let one = BigUint::one();
  for nonce in 0..(1u64 << plan.nonce_bits) {
    let extension = plan.evaluate(&random, nonce);
    let number = cert.number() * &extension + &one;
    let mut base = BigUint::from(2u32);
    while base < number {
      let part = base.modpow(&extension, &number);
      if !part.modpow(cert.number(), &number).is_one() {
        break;
      }
      if (&part - &one).gcd(&number).is_one() {
        cert.extensions.push(PocklingtonExtension {
          plan: plan.clone(),
          random,
          nonce,
          checking_base: base,
          result: number,
        });
        return Some(cert);
      }
      base += &one;
    }
  }
  None
}

/// Hash `inputs` to a certified prime with `entropy` random bits. Returns
/// the certificate and the entropy pool it was drawn from.
pub fn hash_to_pocklington_prime(
  poseidon: &PoseidonParams,
  inputs: &[BigUint],
  entropy: usize,
) -> Option<(PocklingtonCertificate, EntropySource)> {
  let plan = PocklingtonPlan::new(entropy);
  let hash = poseidon.hash(inputs);
  let mut src = EntropySource::new(hash, plan.entropy(), &poseidon.p);
  let mut cert = attempt_base(&plan, &mut src)?;
  for ext in &plan.extensions {
    cert = attempt_extension(cert, ext, &mut src)?;
  }
  Some((cert, src))
}

/// Native re-verification of a certificate against its plan: the checks
/// the circuit enforces (base prime shape + 32-bit test; per extension
/// `ext = 1 ‖ random ‖ nonce` with `nonce < 2^nonce_bits`, `n = p·ext +
/// 1`, `a^{ext·p} ≡ 1`, `gcd(a^{ext} − 1, n) = 1`, and `ext ≤ p + 1`).
pub fn verify_certificate(plan: &PocklingtonPlan, cert: &PocklingtonCertificate) -> bool {
  let one = BigUint::one();
  if cert.base_nonce & 0b11 != 0b11 || cert.base_nonce >= (1u64 << plan.base_nonce_bits) {
    return false;
  }
  if cert.base_random.bits() as usize != plan.base_random_bits + 1 {
    return false;
  }
  let base = (&cert.base_random << plan.base_nonce_bits) + BigUint::from(cert.base_nonce);
  if base != cert.base_prime || !miller_rabin_32b(&base) {
    return false;
  }
  if cert.extensions.len() != plan.extensions.len() {
    return false;
  }
  let mut p = cert.base_prime.clone();
  for (e, pe) in cert.extensions.iter().zip(&plan.extensions) {
    if e.plan != *pe
      || e.nonce >= (1u64 << pe.nonce_bits)
      || e.random.bits() as usize > pe.random_bits
    {
      return false;
    }
    let ext = pe.evaluate(&e.random, e.nonce);
    if ext > &p + &one {
      return false;
    }
    let n = &p * &ext + &one;
    if n != e.result {
      return false;
    }
    let part = e.checking_base.modpow(&ext, &n);
    if !part.modpow(&p, &n).is_one() || !(&part - &one).gcd(&n).is_one() {
      return false;
    }
    p = n;
  }
  true
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn plan_256_matches_reference_shape() {
    let plan = PocklingtonPlan::new(256);
    assert_eq!(plan.base_nonce_bits, 11);
    assert_eq!(plan.base_random_bits, 20);
    let ext: Vec<(usize, usize)> = plan
      .extensions
      .iter()
      .map(|e| (e.nonce_bits, e.random_bits))
      .collect();
    assert_eq!(ext, vec![(11, 19), (12, 48), (13, 107), (14, 62)]);
    assert_eq!(plan.entropy(), 256);
    assert_eq!(plan.max_bits(), 322);
  }

  #[test]
  fn hash_to_prime_certifies_and_is_prime() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let inputs: Vec<BigUint> = (1..=10u32).map(BigUint::from).collect();
    let (cert, src) = hash_to_pocklington_prime(&poseidon, &inputs, 256).expect("certificate");
    let plan = PocklingtonPlan::new(256);
    assert!(verify_certificate(&plan, &cert));
    assert!(miller_rabin(cert.number(), 20));
    assert!(cert.number().bits() as usize <= 322);
    assert_eq!(src.num_elems(), 2);
  }

  #[test]
  fn entropy_pops_from_the_end() {
    let p = super::super::poseidon::bls12_381_r();
    let mut src = EntropySource::new(BigUint::from(5u32), 8, &p);
    // Single element (8 bits needed); its low 254 bits are the pool. The
    // last pool bit is bit 253 of the element, which is 0 for `5`.
    assert_eq!(src.num_elems(), 1);
    assert_eq!(src.next_index(), 253);
    assert!(!src.get_bit());
  }
}
