//! The OWWB20 RSA-accumulator set with Wesolowski proofs of exponentiation,
//! ported from `bellman-bignat` (`group.rs`, `hash/division_intractable.rs`,
//! `set/rsa.rs`, `wesolowski.rs`). Elements are hashed with the
//! division-intractable hash `H∆(y) = OFFSET + low254(Poseidon(y))`
//! (`OFFSET` the reference's 2048-bit constant); the accumulator lives in
//! the quotient group `(Z/N)^*/{±1}` with canonical representatives
//! `min(x, N − x)`; a batch update `A → A^{∏ H∆(y_i)}` is proven with
//! `Q^ℓ · A^r = A'`, `Q = A^{⌊∏/ℓ⌋}`, `r = ∏ (H∆ mod ℓ) mod ℓ`, for the
//! Pocklington challenge `ℓ = Hp(A, A', hashes)`.

use super::pocklington::{self, PocklingtonCertificate, PocklingtonPlan};
use super::poseidon::{FR_CAPACITY, PoseidonParams};
use num_bigint::BigUint;
use num_traits::One;

/// RSA-2048 (the RSA Factoring Challenge modulus), as in the reference.
pub const RSA_2048: &str = "25195908475657893494027183240048398571429282126204032027777137836043662020707595556264018525880784406918290641249515082189298559149176184502808489120072844992687392807287776735971418347270261896375014971824691165077613379859095700097330459748808428401797429100642458691817195118746121515172654632282216869987549182422433637259085141865462043576798423387184774447920739934236584823824281198163815010674810451660377306056201619676256133844143603833904414952634432190114657544454178424020924616515723350778707749817125772467962926386356373289912154831438167899885040445364023527381951378636564391212010397122822120720357";

/// The reference's division-intractable-hash offset for 2048-bit elements.
pub const OFFSET_2048: &str = "30731438344250145947882657666206403727243332864808664054575262055190442942812700108124167942976653745028212341196692947492080562974589240558404052155436479139607283861572110186639866316589725954212169900473106847592072353357762907262662369230376196184226071545259316873351199416881666739376881925207433619609913435128355340248285568061176332195286623104126482371089555666194830543043595601648501184952472930075767818065617175977748228906417030406830990961578747315754348300610520710090878042950122953510395835606916522592211024941845938097013497415239566963754154588561352876059012472806373183052035005766579987123343";

/// Entropy of the Wesolowski challenge prime (the reference's
/// `n_bits_challenge` in `set_bench`).
pub const CHALLENGE_ENTROPY: usize = 256;

/// Elements per item (the reference's `ELEMENT_SIZE`).
pub const ITEM_LEN: usize = 5;

/// The RSA quotient group `(Z/N)^*/{±1}` with generator `g`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaQuotientGroup {
  /// Generator.
  pub g: BigUint,
  /// Modulus.
  pub n: BigUint,
}

impl RsaQuotientGroup {
  /// The reference group: `g = 2`, `N = RSA-2048`.
  pub fn owwb20() -> Self {
    Self {
      g: BigUint::from(2u32),
      n: BigUint::parse_bytes(RSA_2048.as_bytes(), 10).expect("decimal"),
    }
  }

  /// Canonical representative `min(x, N − x)`.
  pub fn canon(&self, x: &BigUint) -> BigUint {
    let x = x % &self.n;
    let y = &self.n - &x;
    if y < x { y } else { x }
  }

  /// Group operation.
  pub fn op(&self, a: &BigUint, b: &BigUint) -> BigUint {
    self.canon(&((a * b) % &self.n))
  }

  /// Exponentiation.
  pub fn power(&self, b: &BigUint, e: &BigUint) -> BigUint {
    self.canon(&b.modpow(e, &self.n))
  }
}

/// Bit width of the hash prefix taken by `H∆` (`FR_CAPACITY`).
pub const DI_BITS: usize = FR_CAPACITY;

/// The reference offset as an integer.
pub fn di_offset() -> BigUint {
  BigUint::parse_bytes(OFFSET_2048.as_bytes(), 10).expect("decimal")
}

/// `H∆(item) = OFFSET + low254(Poseidon(item))`. Returns `(hash, x, H∆)`.
pub fn di_hash(poseidon: &PoseidonParams, item: &[BigUint]) -> (BigUint, BigUint, BigUint) {
  let h = poseidon.hash(item);
  let x = &h & ((BigUint::one() << DI_BITS) - BigUint::one());
  let d = di_offset() + &x;
  (h, x, d)
}

/// The reference's synthetic items: `item_len` field elements, decimal
/// strings `"{tag}{i:06}{j:03}"`.
pub fn synthetic_item(tag: u32, i: usize) -> Vec<BigUint> {
  (0..ITEM_LEN)
    .map(|j| BigUint::parse_bytes(format!("{tag}{i:06}{j:03}").as_bytes(), 10).expect("decimal"))
    .collect()
}

/// Transcript encoding of a 2048-bit group element for the challenge
/// hash: `limbs` little-endian limbs of `limb_bits` bits each (the
/// reference absorbs 64 limbs of 32 bits; this implementation absorbs 9
/// limbs of 240 bits — see the circuit builder).
pub fn digest_limbs(x: &BigUint, limb_bits: usize, limbs: usize) -> Vec<BigUint> {
  let mask = (BigUint::one() << limb_bits) - BigUint::one();
  (0..limbs).map(|i| (x >> (i * limb_bits)) & &mask).collect()
}

/// Limb width used by this implementation's challenge transcript.
pub const DIGEST_LIMB_BITS: usize = 240;
/// Limbs per 2048-bit digest at that width.
pub const DIGEST_LIMBS: usize = 9;

/// One Wesolowski proof of exponentiation for `base^{∏ e_i} = result`.
#[derive(Clone, Debug)]
pub struct PoE {
  /// `Q = base^{⌊∏ e_i / ℓ⌋}` (canonical).
  pub q: BigUint,
  /// `r = ∏ (e_i mod ℓ) mod ℓ`.
  pub r: BigUint,
}

/// Compute the Wesolowski proof natively.
pub fn prove_poe(group: &RsaQuotientGroup, base: &BigUint, exps: &[BigUint], ell: &BigUint) -> PoE {
  let mut prod = BigUint::one();
  for e in exps {
    prod *= e;
  }
  let q = group.power(base, &(&prod / ell));
  let mut r = BigUint::one();
  for e in exps {
    r = (r * (e % ell)) % ell;
  }
  PoE { q, r }
}

/// Check `Q^ℓ · base^r = result` in the quotient group.
pub fn verify_poe(
  group: &RsaQuotientGroup,
  base: &BigUint,
  poe: &PoE,
  ell: &BigUint,
  result: &BigUint,
) -> bool {
  let ql = group.power(&poe.q, ell);
  let br = group.power(base, &poe.r);
  group.op(&ql, &br) == group.canon(result)
}

/// A complete MultiSwap instance for `t` swaps: the initial and final
/// digests, the items, their hashes, the challenge certificate, and the
/// two Wesolowski proofs of `SetBench::verify_swap_all`
/// (`initial^{∏H∆(ins)} = final^{∏H∆(rem)}`).
#[derive(Clone, Debug)]
pub struct MultiSwapInstance {
  /// The group.
  pub group: RsaQuotientGroup,
  /// Digest of the initial set (untouched ∪ removed).
  pub initial: BigUint,
  /// Digest of the final set (untouched ∪ inserted).
  pub final_digest: BigUint,
  /// Removed items.
  pub removed: Vec<Vec<BigUint>>,
  /// Inserted items.
  pub inserted: Vec<Vec<BigUint>>,
  /// `(hash, x, H∆)` per removed item.
  pub removed_hashes: Vec<(BigUint, BigUint, BigUint)>,
  /// `(hash, x, H∆)` per inserted item.
  pub inserted_hashes: Vec<(BigUint, BigUint, BigUint)>,
  /// The challenge-hash inputs (digest limbs, insertion hashes, removal hashes).
  pub challenge_inputs: Vec<BigUint>,
  /// The Pocklington certificate of the challenge `ℓ`.
  pub certificate: PocklingtonCertificate,
  /// The entropy pool the certificate was drawn from.
  pub entropy: pocklington::EntropySource,
  /// The plan.
  pub plan: PocklingtonPlan,
  /// PoE for `initial^{∏H∆(ins)}`.
  pub poe_insert: PoE,
  /// PoE for `final^{∏H∆(rem)}`.
  pub poe_remove: PoE,
  /// The common digest both sides reach.
  pub common: BigUint,
}

impl MultiSwapInstance {
  /// The challenge prime `ℓ`.
  pub fn ell(&self) -> &BigUint {
    self.certificate.number()
  }

  /// Build the reference `SetBench` instance with `n_untouched` untouched
  /// items and `t` removals / insertions (synthetic items), computing
  /// every witness natively.
  pub fn generate(poseidon: &PoseidonParams, n_untouched: usize, t: usize) -> Self {
    let group = RsaQuotientGroup::owwb20();
    let untouched: Vec<Vec<BigUint>> = (0..n_untouched).map(|i| synthetic_item(1, i)).collect();
    let removed: Vec<Vec<BigUint>> = (0..t).map(|i| synthetic_item(2, i)).collect();
    let inserted: Vec<Vec<BigUint>> = (0..t).map(|i| synthetic_item(3, i)).collect();
    let hash_all = |items: &[Vec<BigUint>]| -> Vec<(BigUint, BigUint, BigUint)> {
      items.iter().map(|it| di_hash(poseidon, it)).collect()
    };
    let untouched_hashes = hash_all(&untouched);
    let removed_hashes = hash_all(&removed);
    let inserted_hashes = hash_all(&inserted);
    let digest_of = |sets: &[&[(BigUint, BigUint, BigUint)]]| -> BigUint {
      let mut d = group.g.clone();
      for s in sets {
        for (_, _, e) in s.iter() {
          d = group.power(&d, e);
        }
      }
      d
    };
    let initial = digest_of(&[&untouched_hashes, &removed_hashes]);
    let final_digest = digest_of(&[&untouched_hashes, &inserted_hashes]);

    let mut challenge_inputs = Vec::new();
    challenge_inputs.extend(digest_limbs(&initial, DIGEST_LIMB_BITS, DIGEST_LIMBS));
    challenge_inputs.extend(digest_limbs(&final_digest, DIGEST_LIMB_BITS, DIGEST_LIMBS));
    challenge_inputs.extend(inserted_hashes.iter().map(|(h, _, _)| h.clone()));
    challenge_inputs.extend(removed_hashes.iter().map(|(h, _, _)| h.clone()));
    let (certificate, entropy) =
      pocklington::hash_to_pocklington_prime(poseidon, &challenge_inputs, CHALLENGE_ENTROPY)
        .expect("Pocklington search succeeds with overwhelming probability");
    let plan = PocklingtonPlan::new(CHALLENGE_ENTROPY);
    let ell = certificate.number().clone();

    let ins_exps: Vec<BigUint> = inserted_hashes.iter().map(|(_, _, e)| e.clone()).collect();
    let rem_exps: Vec<BigUint> = removed_hashes.iter().map(|(_, _, e)| e.clone()).collect();
    let poe_insert = prove_poe(&group, &initial, &ins_exps, &ell);
    let poe_remove = prove_poe(&group, &final_digest, &rem_exps, &ell);
    let common = digest_of(&[&untouched_hashes, &removed_hashes, &inserted_hashes]);
    debug_assert!(verify_poe(&group, &initial, &poe_insert, &ell, &common));
    debug_assert!(verify_poe(
      &group,
      &final_digest,
      &poe_remove,
      &ell,
      &common
    ));

    Self {
      group,
      initial,
      final_digest,
      removed,
      inserted,
      removed_hashes,
      inserted_hashes,
      challenge_inputs,
      certificate,
      entropy,
      plan,
      poe_insert,
      poe_remove,
      common,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn quotient_group_canonicalizes() {
    let g = RsaQuotientGroup::owwb20();
    let x = &g.n - BigUint::from(5u32);
    assert_eq!(g.canon(&x), BigUint::from(5u32));
    assert_eq!(
      g.op(&BigUint::from(3u32), &BigUint::from(5u32)),
      BigUint::from(15u32)
    );
  }

  #[test]
  fn di_hash_is_offset_plus_low_bits() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let (h, x, d) = di_hash(&poseidon, &synthetic_item(2, 0));
    assert!(x.bits() as usize <= DI_BITS);
    assert_eq!(d, di_offset() + &x);
    assert!(h < poseidon.p);
    assert_eq!(d.bits(), 2048);
  }

  #[test]
  fn instance_poes_verify_and_meet() {
    let poseidon = PoseidonParams::bls12_381_owwb20();
    let inst = MultiSwapInstance::generate(&poseidon, 0, 1);
    let ell = inst.ell().clone();
    assert!(ell.bits() as usize <= 322);
    assert!(pocklington::verify_certificate(
      &inst.plan,
      &inst.certificate
    ));
    assert!(verify_poe(
      &inst.group,
      &inst.initial,
      &inst.poe_insert,
      &ell,
      &inst.common
    ));
    assert!(verify_poe(
      &inst.group,
      &inst.final_digest,
      &inst.poe_remove,
      &ell,
      &inst.common
    ));
    assert_eq!(inst.challenge_inputs.len(), 2 * DIGEST_LIMBS + 2);
    assert_eq!(poseidon.absorb_cycles(inst.challenge_inputs.len()), 4);
  }
}
