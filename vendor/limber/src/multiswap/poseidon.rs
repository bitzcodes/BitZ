//! Poseidon over the BLS12-381 scalar field, with the parameters of the
//! OWWB20 reference implementation (`sapling-crypto`, branch
//! `bls12-poseidon`): width `t = 6`, `R_F = 8` full rounds, `R_P = 57`
//! partial rounds, quintic S-box, rate 5 / capacity 1, one output
//! element. Round constants are derived exactly as the reference does
//! (Keccak-256 over the personalization tag, the group-hash seed block,
//! and a big-endian nonce; little-endian repr; rejected unless in
//! `[1, r)`). The MDS matrix is a Cauchy matrix `1/(x_i − y_j)` with
//! `x, y` drawn from a Keccak-seeded stream instead of the reference's
//! `rand 0.4` ChaCha stream (same construction, different sample) — the
//! hash is therefore not bit-compatible with the reference, but has the
//! identical circuit cost and security parameters.
//!
//! Arithmetic is plain `BigUint` modulo `r`: this module only produces
//! witnesses and the circuit builder's LC coefficients.

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Zero};
use sha3::{Digest, Keccak256};

/// The BLS12-381 scalar field modulus `r`.
pub fn bls12_381_r() -> BigUint {
  BigUint::parse_bytes(
    b"73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    16,
  )
  .expect("valid hex")
}

/// Capacity (in bits) of the field: values below `2^254` embed uniquely.
pub const FR_CAPACITY: usize = 254;

/// The 64-byte group-hash seed block of `sapling-crypto`.
const GH_FIRST_BLOCK: &[u8; 64] =
  b"096b36a5804bfacef1691e173c366a47ff5ba84a44f26ddd7e8d9f79d5b42df0";

/// Poseidon parameters: `t`, round counts, round keys, and the MDS matrix
/// (row-major, `t × t`).
#[derive(Clone, Debug)]
pub struct PoseidonParams {
  /// State width.
  pub t: usize,
  /// Number of full rounds (split `⌈R_F/2⌉` before and `⌊R_F/2⌋` after
  /// the partial rounds).
  pub r_f: usize,
  /// Number of partial rounds.
  pub r_p: usize,
  /// Full-round keys, `t` per round, `r_f` rounds.
  pub full_keys: Vec<BigUint>,
  /// Partial-round keys, `t` per round, `r_p` rounds.
  pub partial_keys: Vec<BigUint>,
  /// MDS matrix, row-major.
  pub mds: Vec<BigUint>,
  /// The field modulus.
  pub p: BigUint,
}

fn keccak_constant(tag: &[u8], nonce: u32) -> BigUint {
  let mut h = Keccak256::new();
  h.update(tag);
  h.update(GH_FIRST_BLOCK);
  h.update(nonce.to_be_bytes());
  BigUint::from_bytes_le(&h.finalize())
}

/// Round-constant stream of the reference: increment a nonce until the
/// required number of nonzero in-field constants has been collected.
fn round_constants(tag: &[u8], count: usize, p: &BigUint) -> Vec<BigUint> {
  let mut out = Vec::with_capacity(count);
  let mut nonce = 0u32;
  while out.len() < count {
    let c = keccak_constant(tag, nonce);
    if !c.is_zero() && &c < p {
      out.push(c);
    }
    nonce += 1;
  }
  out
}

/// Modular inverse via the extended Euclidean algorithm.
pub fn mod_inv(a: &BigUint, p: &BigUint) -> BigUint {
  let e = BigInt::from(a.clone()).extended_gcd(&BigInt::from(p.clone()));
  assert!(e.gcd.is_one(), "mod_inv: not invertible");
  let x = e.x.mod_floor(&BigInt::from(p.clone()));
  x.to_biguint().expect("non-negative")
}

impl PoseidonParams {
  /// The OWWB20 reference parameters (`t = 6`, `R_F = 8`, `R_P = 57`).
  pub fn bls12_381_owwb20() -> Self {
    Self::new(6, 8, 57)
  }

  /// Parameters for arbitrary `(t, R_F, R_P)` with the reference's
  /// constant-derivation scheme.
  pub fn new(t: usize, r_f: usize, r_p: usize) -> Self {
    let p = bls12_381_r();
    let full_keys = round_constants(b"Hadesr_f", r_f * t, &p);
    let partial_keys = round_constants(b"Hadesr_p", r_p * t, &p);
    // Cauchy MDS from a Keccak-seeded stream of `2t` distinct elements.
    let mds = {
      let mut xs: Vec<BigUint> = Vec::new();
      let mut nonce = 0u32;
      while xs.len() < 2 * t {
        let c = keccak_constant(b"Hadesmds", nonce);
        nonce += 1;
        if c < p && !xs.contains(&c) {
          xs.push(c);
        }
      }
      let (x, y) = xs.split_at(t);
      let mut m = Vec::with_capacity(t * t);
      for xi in x {
        for yj in y {
          let diff = (xi + &p - yj) % &p;
          m.push(mod_inv(&diff, &p));
        }
      }
      m
    };
    Self {
      t,
      r_f,
      r_p,
      full_keys,
      partial_keys,
      mds,
      p,
    }
  }

  /// Round key of full round `round` (`t` elements).
  pub fn full_key(&self, round: usize) -> &[BigUint] {
    &self.full_keys[round * self.t..(round + 1) * self.t]
  }

  /// Round key of partial round `round` (`t` elements).
  pub fn partial_key(&self, round: usize) -> &[BigUint] {
    &self.partial_keys[round * self.t..(round + 1) * self.t]
  }

  /// Row `i` of the MDS matrix.
  pub fn mds_row(&self, i: usize) -> &[BigUint] {
    &self.mds[i * self.t..(i + 1) * self.t]
  }

  /// Number of full rounds applied before the partial rounds.
  pub fn pre_full_rounds(&self) -> usize {
    self.r_f - self.r_f / 2
  }

  /// Absorption rate (elements per permutation): `t − 1`.
  pub fn rate(&self) -> usize {
    self.t - 1
  }

  /// `x^5 mod p`.
  pub fn sbox(&self, x: &BigUint) -> BigUint {
    let x2 = (x * x) % &self.p;
    let x4 = (&x2 * &x2) % &self.p;
    (&x4 * x) % &self.p
  }

  /// `M · state`.
  pub fn mds_apply(&self, state: &[BigUint]) -> Vec<BigUint> {
    (0..self.t)
      .map(|i| {
        let mut acc = BigUint::zero();
        for (m, s) in self.mds_row(i).iter().zip(state) {
          acc += m * s;
        }
        acc % &self.p
      })
      .collect()
  }

  /// One round: add the round key, S-box (all elements for a full round,
  /// element 0 for a partial round), then the MDS layer unless `last`.
  pub fn round(&self, state: &mut Vec<BigUint>, key: &[BigUint], full: bool, last: bool) {
    for (s, k) in state.iter_mut().zip(key) {
      *s = (&*s + k) % &self.p;
    }
    if full {
      for s in state.iter_mut() {
        *s = self.sbox(s);
      }
    } else {
      state[0] = self.sbox(&state[0]);
    }
    if !last {
      *state = self.mds_apply(state);
    }
  }

  /// The Poseidon permutation, as sequenced by the reference: `⌈R_F/2⌉`
  /// full rounds, `R_P` partial rounds, `⌊R_F/2⌋` full rounds, the last
  /// of which skips the MDS layer.
  pub fn permute(&self, state: &mut Vec<BigUint>) {
    assert_eq!(state.len(), self.t);
    let pre = self.pre_full_rounds();
    for r in 0..pre {
      self.round(state, self.full_key(r), true, false);
    }
    for r in 0..self.r_p {
      self.round(state, self.partial_key(r), false, false);
    }
    for r in pre..self.r_f {
      self.round(state, self.full_key(r), true, r == self.r_f - 1);
    }
  }

  /// The reference sponge: permute an all-zero state, then per absorption
  /// cycle add `rate` inputs (zero-padded) into the bottom words and
  /// permute; output is word 0.
  pub fn hash(&self, inputs: &[BigUint]) -> BigUint {
    let rate = self.rate();
    let cycles = inputs.len().div_ceil(rate).max(1);
    let mut input = inputs.to_vec();
    input.resize(cycles * rate, BigUint::zero());
    let mut state = vec![BigUint::zero(); self.t];
    self.permute(&mut state);
    for i in 0..cycles {
      for (w, a) in state.iter_mut().zip(&input[i * rate..(i + 1) * rate]) {
        *w = (&*w + a) % &self.p;
      }
      self.permute(&mut state);
    }
    state[0].clone()
  }

  /// Number of permutations `hash` performs on `n` inputs (the initial
  /// zero-state permutation is input-independent and folds into
  /// constants in a circuit).
  pub fn absorb_cycles(&self, n: usize) -> usize {
    n.div_ceil(self.rate()).max(1)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn params_have_reference_shape() {
    let p = PoseidonParams::bls12_381_owwb20();
    assert_eq!(p.t, 6);
    assert_eq!(p.full_keys.len(), 8 * 6);
    assert_eq!(p.partial_keys.len(), 57 * 6);
    assert_eq!(p.mds.len(), 36);
    assert_eq!(p.rate(), 5);
    assert!(p.full_keys.iter().all(|k| !k.is_zero() && k < &p.p));
    // MDS entries are inverses of distinct differences: nonzero.
    assert!(p.mds.iter().all(|m| !m.is_zero()));
  }

  #[test]
  fn permutation_is_deterministic_and_sbox_count_matches() {
    let p = PoseidonParams::bls12_381_owwb20();
    let mut s1: Vec<BigUint> = (0..6).map(|i| BigUint::from(i as u32 + 1)).collect();
    let mut s2 = s1.clone();
    p.permute(&mut s1);
    p.permute(&mut s2);
    assert_eq!(s1, s2);
    assert!(s1.iter().all(|x| x < &p.p));
    // 8 full rounds × 6 + 57 partial = 105 S-boxes.
    assert_eq!(p.r_f * p.t + p.r_p, 105);
  }

  #[test]
  fn hash_absorbs_at_rate_five() {
    let p = PoseidonParams::bls12_381_owwb20();
    assert_eq!(p.absorb_cycles(5), 1);
    assert_eq!(p.absorb_cycles(6), 2);
    assert_eq!(p.absorb_cycles(20), 4);
    let a = p.hash(&[BigUint::from(1u32)]);
    let b = p.hash(&[BigUint::from(2u32)]);
    assert_ne!(a, b);
    assert!(a < p.p);
  }
}
