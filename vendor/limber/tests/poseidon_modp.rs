//! Integration tests for the combined Poseidon2 benchmark workload
//! (`limber::poseidon2`): KAT agreement with the independent Python
//! generator (per field and composed across the three blocks in
//! `FIELD_ORDER`), circuit ↔ reference agreement in every block,
//! input/bound policies, satisfiability, negative cases, and combined
//! proof round-trips.
//!
//! Proof round-trips run at `H = 1` per field (3 hashes, 1,299 rows →
//! 2^11) so the release CI suite stays fast; the full-size combined
//! `H = 10` proof is `#[ignore]`d. A cheap, non-proving `H = 10`
//! per-field test asserts the 30-hash headline structure.

use limber::{
  errors::SpartanError,
  imod_r1cs_modp::IntModR1CSWitnessModp,
  imod_spartan_modp::IntModSpartanModpSNARK,
  poseidon_bench::{DEFAULT_BD_K, DEFAULT_HYRAX_K},
  poseidon2::{
    FIELD_ORDER, Field, PoseidonVerifierKey, build_all_params, build_inputs, build_params,
    build_shape, check_canonical_io, compute_advice, expected_chain, permute, validate_advice,
    verify_poseidon_chain,
  },
  provider::{T256DynPrimeBdEngine, T256DynPrimeEngine, pcs::integer_modpcs::IntEvalParams},
};
use num_bigint::BigUint;
use num_traits::Zero;

type Hy = T256DynPrimeEngine;
type Bd = T256DynPrimeBdEngine;

/// The persisted per-backend IntEval `k` defaults, so proofs are tested
/// under the same parameters the benchmark resolves.
const HYRAX_K: usize = DEFAULT_HYRAX_K;
const BD_K: usize = DEFAULT_BD_K;

fn kat_fixture() -> serde_json::Value {
  let path =
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/poseidon2_kat_v1.json");
  serde_json::from_str(&std::fs::read_to_string(path).expect("KAT fixture readable"))
    .expect("KAT fixture parses")
}

fn from_hex(v: &serde_json::Value) -> BigUint {
  let s = v.as_str().expect("hex string");
  assert_eq!(s.len(), 64, "32-byte lowercase hex");
  BigUint::parse_bytes(s.as_bytes(), 16).expect("valid hex")
}

fn derive_params(log_n: usize, k: usize) -> IntEvalParams {
  IntEvalParams::derive(256, 64, k, log_n).expect("IntEval params satisfy bounds")
}

/// KAT — standalone: `permute([1, 2, 3])` per field against the
/// independent generator's full 3-lane output.
#[test]
fn kat_standalone_permutation() {
  let fixture = kat_fixture();
  for field in FIELD_ORDER {
    let params = build_params(field).unwrap();
    let entry = &fixture["fields"][field.name()];
    let input: [BigUint; 3] = core::array::from_fn(|i| from_hex(&entry["standalone_input"][i]));
    assert_eq!(input, core::array::from_fn(|i| BigUint::from(i as u32 + 1)));
    let out = permute(&params, input).unwrap();
    let expected: [BigUint; 3] = core::array::from_fn(|i| from_hex(&entry["standalone_output"][i]));
    assert_eq!(out, expected, "standalone KAT for {}", field.name());
  }
}

/// KAT — chain and composition: all of `h_1..h_10` per field, and the
/// three final fixture digests — assembled by the fixture's explicit
/// `field_order` array, never by JSON-map iteration — match the combined
/// circuit's ordered public digests.
#[test]
fn kat_chain_states_and_composition() {
  let fixture = kat_fixture();
  let messages = build_inputs(10).unwrap();
  // The fixture's explicit order array is the composition order.
  let order: Vec<String> = fixture["field_order"]
    .as_array()
    .expect("field_order array")
    .iter()
    .map(|v| v.as_str().expect("field name").to_string())
    .collect();
  assert_eq!(
    order,
    FIELD_ORDER.map(|f| f.name().to_string()).to_vec(),
    "fixture field_order must equal FIELD_ORDER"
  );
  let set = build_all_params().unwrap();
  for field in FIELD_ORDER {
    let entry = &fixture["fields"][field.name()];
    // The fixture's messages must equal the benchmark inputs.
    let fixture_msgs: Vec<BigUint> = (0..10).map(|j| from_hex(&entry["messages"][j])).collect();
    assert_eq!(fixture_msgs, messages);
    let chain = expected_chain(set.get(field), &messages).unwrap();
    let expected: Vec<BigUint> = (0..10).map(|i| from_hex(&entry["chain"][i])).collect();
    assert_eq!(chain, expected, "chain KAT for {}", field.name());
  }
  // Composition: assemble the three final fixture digests by the
  // explicit order and compare against the combined circuit's advice.
  let (_shape, layout) = build_shape::<Hy>(&set, 10).unwrap();
  let (_w, _q, digests) = compute_advice(&set, &layout, &messages).unwrap();
  for (f, name) in order.iter().enumerate() {
    let fixture_h10 = from_hex(&fixture["fields"][name.as_str()]["chain"][9]);
    assert_eq!(digests[f], fixture_h10, "public-IO slot {f} ({name})");
  }
}

/// Circuit ↔ reference: every materialized chain state `h_{f,i}` in all
/// three blocks equals that field's `expected_chain`.
#[test]
fn circuit_matches_reference_chain() {
  let h = 3usize;
  let set = build_all_params().unwrap();
  let (_shape, layout) = build_shape::<Hy>(&set, h).unwrap();
  let messages = build_inputs(h).unwrap();
  let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
  validate_advice(&set, &layout, &w, &q, &digests).unwrap();
  let block_cols = 434 * h - 1;
  for (f, field) in FIELD_ORDER.iter().enumerate() {
    let chain = expected_chain(set.get(*field), &messages).unwrap();
    for (i, expected) in chain.iter().enumerate().take(h - 1) {
      // Terminal reduce row of the block's permutation i: local row
      // 433·(i+1) − 1; its output column is block_base + H + local_row.
      let col = f * block_cols + h + 433 * (i + 1) - 1;
      assert_eq!(
        &w[col],
        expected,
        "materialized h_{} in block {}",
        i + 1,
        field.name()
      );
    }
    assert_eq!(&digests[f], chain.last().unwrap());
  }
}

/// Hash-count and canonical-input policies on every public entry point.
#[test]
fn hash_count_and_input_bounds() {
  let set = build_all_params().unwrap();
  let bn_p = set.get(Field::Bn254Fr).modulus().clone();

  // H = 0 rejected before any dimension arithmetic.
  assert!(build_inputs(0).is_err());
  assert!(build_shape::<Hy>(&set, 0).is_err());
  assert!(expected_chain(set.get(Field::Bn254Fr), &[]).is_err());

  // H > u32::MAX rejected (64-bit targets), and combined 3H arithmetic
  // is checked.
  #[cfg(target_pointer_width = "64")]
  {
    let too_many = u32::MAX as usize + 1;
    assert!(build_inputs(too_many).is_err());
    assert!(build_shape::<Hy>(&set, too_many).is_err());
  }

  // Canonical-input policy: a lane or message >= the modulus of the
  // block using it is rejected, not silently reduced.
  let params = build_params(Field::Bn254Fr).unwrap();
  assert!(permute(&params, [bn_p.clone(), BigUint::zero(), BigUint::zero()]).is_err());
  assert!(expected_chain(&params, std::slice::from_ref(&bn_p)).is_err());
  let (_shape, layout) = build_shape::<Hy>(&set, 1).unwrap();
  assert!(compute_advice(&set, &layout, &[bn_p]).is_err());
  // Wrong message count.
  assert!(compute_advice(&set, &layout, &build_inputs(2).unwrap()).is_err());
}

/// Padding — values: `w[real_cols..] == 0` and `q[real_rows..] == 0`.
#[test]
fn padding_values_are_zero() {
  let set = build_all_params().unwrap();
  let (_shape, layout) = build_shape::<Hy>(&set, 2).unwrap();
  let (w, q, digests) = compute_advice(&set, &layout, &build_inputs(2).unwrap()).unwrap();
  assert!(w[layout.real_cols()..].iter().all(BigUint::is_zero));
  assert!(q[layout.real_rows()..].iter().all(BigUint::is_zero));
  validate_advice(&set, &layout, &w, &q, &digests).unwrap();
}

/// Canonicality policy directly — no proof argument needed: reject
/// lengths other than 3, accept `p_f − 1`, and reject `p_f` independently
/// at every ordered slot.
#[test]
fn canonicality_policy_is_directly_testable() {
  let set = build_all_params().unwrap();
  let ok: Vec<BigUint> = FIELD_ORDER
    .iter()
    .map(|f| set.get(*f).modulus() - BigUint::from(1u32))
    .collect();
  check_canonical_io(&ok, &set).unwrap();
  assert!(check_canonical_io(&[], &set).is_err());
  assert!(check_canonical_io(&ok[..2], &set).is_err());
  for (f, field) in FIELD_ORDER.iter().enumerate() {
    let mut bad = ok.clone();
    bad[f] = set.get(*field).modulus().clone();
    assert!(check_canonical_io(&bad, &set).is_err(), "slot {f}");
  }
}

/// One combined `shape.is_sat` covers all three field blocks, plus the
/// negative relation case: a tampered witness must fail `is_sat`.
/// `prove` is deliberately not called on the unsatisfied witness (it
/// would trip a debug assertion in the sumcheck prover).
#[test]
fn satisfiability_and_tampered_witness() {
  let set = build_all_params().unwrap();
  let (shape, layout) = build_shape::<Hy>(&set, 1).unwrap();
  let messages = build_inputs(1).unwrap();
  let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
  let ie_params = derive_params(layout.log_n(), HYRAX_K);
  let (pk, _vk) =
    IntModSpartanModpSNARK::<Hy>::setup_with_params(shape.clone(), ie_params).unwrap();
  let (witness, instance) =
    IntModR1CSWitnessModp::<Hy>::new(&shape, pk.ck(), w.clone(), q.clone(), digests.to_vec())
      .unwrap();
  shape.is_sat(pk.ck(), &instance, &witness).unwrap();

  // Tamper one witness value (in the middle block): the combined
  // relation must fail.
  let mut w_bad = w;
  let middle_block_col = 434 - 1; // first message column of block 1 (H = 1)
  w_bad[middle_block_col] += BigUint::from(1u32);
  let (witness_bad, instance_bad) =
    IntModR1CSWitnessModp::<Hy>::new(&shape, pk.ck(), w_bad, q, digests.to_vec()).unwrap();
  assert!(shape.is_sat(pk.ck(), &instance_bad, &witness_bad).is_err());
}

/// Combined proof round-trips at `H = 1` per field for both backends,
/// verified through `verify_poseidon_chain` (the three-digest
/// canonicality wrapper), plus the noncanonical-IO negative case: for
/// each index `f`, only `x[f] >= p_f` is set; rejected at the wrapper
/// without touching the proof.
macro_rules! roundtrip_h1 {
  ($engine:ty, $k:expr) => {{
    let set = build_all_params().unwrap();
    let (shape, layout) = build_shape::<$engine>(&set, 1).unwrap();
    let messages = build_inputs(1).unwrap();
    let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
    validate_advice(&set, &layout, &w, &q, &digests).unwrap();
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let chain = expected_chain(set.get(*field), &messages).unwrap();
      assert_eq!(&digests[f], chain.last().unwrap());
    }
    let ie_params = derive_params(layout.log_n(), $k);
    let (pk, vk) =
      IntModSpartanModpSNARK::<$engine>::setup_with_params(shape.clone(), ie_params).unwrap();
    let pvk = PoseidonVerifierKey::new(vk, &set, &layout).unwrap();
    let (witness, instance) = IntModR1CSWitnessModp::<$engine>::new(
      &shape,
      pk.ck(),
      w.clone(),
      q.clone(),
      digests.to_vec(),
    )
    .unwrap();
    let proof = IntModSpartanModpSNARK::<$engine>::prove(&pk, &instance, &witness).unwrap();
    verify_poseidon_chain(&pvk, &instance, &proof).unwrap();
    assert!(!proof.eval_arg_bytes().unwrap().is_empty());

    // Negative — noncanonical IO at each ordered slot: set only x[f] to
    // p_f, holding the others fixed; rejected at the wrapper without
    // proving. Built through the public constructor (same commitments,
    // different public IO).
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let mut x_bad = digests.to_vec();
      x_bad[f] = set.get(*field).modulus().clone();
      let (_witness_bad, bad) =
        IntModR1CSWitnessModp::<$engine>::new(&shape, pk.ck(), w.clone(), q.clone(), x_bad)
          .unwrap();
      assert!(
        verify_poseidon_chain(&pvk, &bad, &proof).is_err(),
        "noncanonical x[{f}] must be rejected"
      );
    }
    instance
  }};
}

#[test]
fn roundtrip_h1_combined_hyrax() {
  let instance = roundtrip_h1!(Hy, HYRAX_K);
  // H = 1: n = 2^11, f_chunk = 2^15, 16 rows per commitment. `comm_w`
  // is now a per-segment Vec (one width segment here), so its 8-byte
  // outer length prefix is added on top of the two commitments:
  // 8 (outer Vec) + 2 × (8-byte Vec length + 16 × 33 bytes) = 1,080.
  assert_eq!(instance.commitment_bytes().unwrap().len(), 1080);
}

#[test]
fn roundtrip_h1_combined_brakedown() {
  let instance = roundtrip_h1!(Bd, BD_K);
  // Brakedown: two 32-byte Merkle roots + the 8-byte outer length
  // prefix of the per-segment `comm_w` Vec = 72.
  assert_eq!(instance.commitment_bytes().unwrap().len(), 72);
}

/// Cheap, non-proving `H = 10`-per-field headline structure: the 30-hash
/// schedule, rows, columns, padded dimensions, and padding boundary via
/// the public getters. (NNZ, modulus segments, and public-IO order need
/// `pub(crate)` matrix access and live in the crate-internal unit tests.)
#[test]
fn headline_h10_structure() {
  let set = build_all_params().unwrap();
  let (_shape, layout) = build_shape::<Hy>(&set, 10).unwrap();
  assert_eq!(layout.hashes_per_field(), 10);
  assert_eq!(layout.total_hashes(), 30);
  assert_eq!(layout.real_rows(), 12990);
  assert_eq!(layout.real_cols(), 13017);
  assert_eq!(layout.num_cons(), 1 << 14);
  assert_eq!(layout.num_vars(), 1 << 14);
  assert_eq!(layout.log_n(), 14);
  assert!(layout.real_rows() < layout.num_cons());
  assert!(layout.real_cols() < layout.num_vars());
  let messages = build_inputs(10).unwrap();
  let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
  validate_advice(&set, &layout, &w, &q, &digests).unwrap();
}

/// Full-size combined `H = 10` proof round-trips (both backends), with
/// the §11 input-commitment size pins: the Hyrax pair is 8,464 canonical
/// bytes (2 × 128 group elements at the combined 2^18 chunk length)
/// versus Brakedown's 64 (2 × 32-byte roots) — 132.25× on the pair.
/// Ignored in CI; run with
/// `cargo test --release --test poseidon_modp -- --ignored`.
#[test]
#[ignore]
fn full_size_h10_proof_roundtrip() {
  let hy_instance = {
    let set = build_all_params().unwrap();
    let (shape, layout) = build_shape::<Hy>(&set, 10).unwrap();
    let messages = build_inputs(10).unwrap();
    let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
    let ie = derive_params(layout.log_n(), HYRAX_K);
    let (pk, vk) = IntModSpartanModpSNARK::<Hy>::setup_with_params(shape.clone(), ie).unwrap();
    let pvk = PoseidonVerifierKey::new(vk, &set, &layout).unwrap();
    let (witness, instance) =
      IntModR1CSWitnessModp::<Hy>::new(&shape, pk.ck(), w, q, digests.to_vec()).unwrap();
    let proof = IntModSpartanModpSNARK::<Hy>::prove(&pk, &instance, &witness).unwrap();
    verify_poseidon_chain(&pvk, &instance, &proof).unwrap();
    instance
  };
  assert_eq!(hy_instance.commitment_bytes().unwrap().len(), 8464);

  let bd_instance = {
    let set = build_all_params().unwrap();
    let (shape, layout) = build_shape::<Bd>(&set, 10).unwrap();
    let messages = build_inputs(10).unwrap();
    let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
    let ie = derive_params(layout.log_n(), BD_K);
    let (pk, vk) = IntModSpartanModpSNARK::<Bd>::setup_with_params(shape.clone(), ie).unwrap();
    let pvk = PoseidonVerifierKey::new(vk, &set, &layout).unwrap();
    let (witness, instance) =
      IntModR1CSWitnessModp::<Bd>::new(&shape, pk.ck(), w, q, digests.to_vec()).unwrap();
    let proof = IntModSpartanModpSNARK::<Bd>::prove(&pk, &instance, &witness).unwrap();
    verify_poseidon_chain(&pvk, &instance, &proof).unwrap();
    instance
  };
  assert_eq!(bd_instance.commitment_bytes().unwrap().len(), 64);
}

/// A proof for one digest triple must not verify against a different
/// (canonical) public digest under `verify_poseidon_chain` — the generic
/// SNARK layer rejects it cleanly (`InvalidFieldContext` when the
/// transcript change re-samples the runtime prime, a proof error
/// otherwise).
#[test]
fn wrong_canonical_digest_is_rejected() {
  let set = build_all_params().unwrap();
  let (shape, layout) = build_shape::<Hy>(&set, 1).unwrap();
  let messages = build_inputs(1).unwrap();
  let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
  let ie = derive_params(layout.log_n(), HYRAX_K);
  let (pk, vk) = IntModSpartanModpSNARK::<Hy>::setup_with_params(shape.clone(), ie).unwrap();
  let pvk = PoseidonVerifierKey::new(vk, &set, &layout).unwrap();
  let (witness, instance) =
    IntModR1CSWitnessModp::<Hy>::new(&shape, pk.ck(), w.clone(), q.clone(), digests.to_vec())
      .unwrap();
  let proof = IntModSpartanModpSNARK::<Hy>::prove(&pk, &instance, &witness).unwrap();
  verify_poseidon_chain(&pvk, &instance, &proof).unwrap();

  // Change only the middle digest to another canonical value, holding
  // the other two fixed.
  let mut x_bad = digests.to_vec();
  x_bad[1] = (&digests[1] + BigUint::from(1u32)) % set.get(Field::Bls12381Fr).modulus();
  let (_witness_bad, bad) = IntModR1CSWitnessModp::<Hy>::new(&shape, pk.ck(), w, q, x_bad).unwrap();
  let err = verify_poseidon_chain(&pvk, &bad, &proof).unwrap_err();
  assert!(
    matches!(
      err,
      SpartanError::InvalidFieldContext
        | SpartanError::ProofVerifyError { .. }
        | SpartanError::InvalidSumcheckProof
    ),
    "unexpected error: {err:?}"
  );
}
