//! End-to-end tests of the CM-AND relation with the F₂-VIRTUAL block
//! (paper `s:to_f2_virtual` / `\Relation_{CM}`): Spartan proves
//! `x + y − w − 2z = 0` per gate over q = 2^100−15, the committed vector
//! carries only the `x`/`y`/`z` bits, and the virtual BitZ opening derives
//! every `w` bit as `x ⊕ y` structurally — so acceptance FORCES
//! `z = x ∧ y`.
//!
//! Battery: honest roundtrips at two shapes; the proof codec (canonical,
//! tamper-rejecting); a FALSE relation with consistent bits is rejected;
//! an honest relation with INCONSISTENT committed bits is rejected; a
//! mismatched statement is rejected; the production entry points gate
//! unaudited configurations.

use ::bitz::ligerito_flock::IntEvalRsLigVirtProof;
use ::bitz::piop::spartan::protocol::Proof;
use ::bitz::piop::spartan::protocol::ProtocolError;

use std::panic::{AssertUnwindSafe, catch_unwind};

use bitz::ligerito::packed_vars;
use bitz::ligerito_flock::{LigConfig, lig_configs};
use bitz::piop::spartan::{
    CmAndLayout, CmAndWitness, SpartanBitzField, commit_cm_and_witness_with_config,
    prepare_cm_and_relation, prove_cm_and_bitz, prove_cm_and_bitz_with_config,
    verify_cm_and_bitz_with_config,
};
use bitz::transcript::Blake3Transcript;

const PRODUCTION_GATES: usize = 1 << 15;

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn adhoc_configs(
    layout: &CmAndLayout,
) -> (
    flock_core::pcs::ligerito::ProverConfig,
    flock_core::pcs::ligerito::VerifierConfig,
) {
    let p = layout.bitz_params();
    lig_configs(packed_vars(&p), LigConfig::Adhoc { log_batch: 2, log_inv_rate: 2 })
        .expect("ad-hoc test configuration")
}

fn audited_configs(
    relation: &bitz::piop::spartan::PreparedCmAndRelation,
) -> (
    flock_core::pcs::ligerito::ProverConfig,
    flock_core::pcs::ligerito::VerifierConfig,
) {
    let resolved = relation.ligerito_configuration().expect("audited test configuration");
    (resolved.prover().clone(), resolved.verifier().clone())
}

/// Full honest pipeline at `gates`, returning everything a tamper case
/// needs.
struct Fixture {
    relation: bitz::piop::spartan::PreparedCmAndRelation,
    witness: CmAndWitness,
    hint: bitz::ligerito_flock::FlockCommitHint,
    vc: flock_core::pcs::ligerito::VerifierConfig,
    proof: Proof<IntEvalRsLigVirtProof>,
}

fn honest_fixture(gates: usize, seed: u64) -> Fixture {
    let config = bitz::piop::spartan::spartan_bitz_field_config();
    let witness = CmAndWitness::from_fn(gates, |i| {
        let r = splitmix(seed ^ i as u64);
        (r as u32, (r >> 32) as u32)
    })
    .unwrap();
    let layout = *witness.layout();
    let relation = prepare_cm_and_relation(layout, &config).unwrap();
    let (pc, vc) = audited_configs(&relation);
    let hint = commit_cm_and_witness_with_config(&layout, witness.f_bit_rows(), &pc).unwrap();

    let mut pt = Blake3Transcript::new();
    let proof = prove_cm_and_bitz_with_config(&mut pt, &relation, &witness, &hint, &pc).unwrap();
    Fixture {
        relation,
        witness,
        hint,
        vc,
        proof,
    }
}

fn verify_fixture(fx: &Fixture, proof: &Proof<IntEvalRsLigVirtProof>) -> Result<(), ProtocolError> {
    let mut vt = Blake3Transcript::new();
    verify_cm_and_bitz_with_config(&mut vt, &fx.relation, &fx.hint.commitment, proof, &fx.vc)
}

#[test]
fn cm_and_virtual_roundtrips() {
    for (gates, seed) in [
        (PRODUCTION_GATES, 0xC0_0001u64),
        (PRODUCTION_GATES + 77, 0xC0_0002),
    ] {
        let fx = honest_fixture(gates, seed);
        verify_fixture(&fx, &fx.proof)
            .unwrap_or_else(|e| panic!("honest CM-AND roundtrip ({gates} gates) failed: {e:?}"));
    }
}

#[test]
fn cm_and_small_explicit_domains_roundtrip_and_reject_false_witnesses() {
    let config = bitz::piop::spartan::spartan_bitz_field_config();
    for gates in [1, 2, 127, 128, 129, 256] {
        for invalid in [false, true] {
            let witness = CmAndWitness::from_gate_values(gates, |i| {
                let r = splitmix(0xC0_5A11 ^ i as u64);
                let (x, y) = (r as u32, (r >> 32) as u32);
                let z = (x & y) ^ u32::from(invalid && i == 0);
                (x, y, z, x ^ y)
            })
            .unwrap();
            let layout = *witness.layout();
            let relation = prepare_cm_and_relation(layout, &config).unwrap();
            let (pc, vc) = adhoc_configs(&layout);
            let hint =
                commit_cm_and_witness_with_config(&layout, witness.f_bit_rows(), &pc).unwrap();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                prove_cm_and_bitz_with_config(
                    &mut Blake3Transcript::new(),
                    &relation,
                    &witness,
                    &hint,
                    &pc,
                )
            }))
            .ok()
            .and_then(Result::ok)
            .map(|proof| {
                verify_cm_and_bitz_with_config(
                    &mut Blake3Transcript::new(),
                    &relation,
                    &hint.commitment,
                    &proof,
                    &vc,
                )
            });
            assert_eq!(
                matches!(outcome, Some(Ok(()))),
                !invalid,
                "CM-AND domain {gates}, invalid={invalid}: {outcome:?}",
            );
        }
    }
}

#[test]
fn cm_and_proof_codec_roundtrips_and_rejects_tampering() {
    let fx = honest_fixture(PRODUCTION_GATES, 0xC0DE_C0DE);
    let bytes = fx.proof.bitz().to_bytes();
    let decoded = IntEvalRsLigVirtProof::from_bytes(&bytes).expect("canonical decode");
    assert_eq!(
        decoded.to_bytes(),
        bytes,
        "codec is a bijection on its image"
    );
    let reproof =
        Proof::<IntEvalRsLigVirtProof>::from_parts(fx.proof.prefix().clone(), None, decoded);
    verify_fixture(&fx, &reproof).expect("decoded proof verifies");

    // Every truncation must fail to decode.
    for cut in [1usize, bytes.len() / 2, bytes.len() - 1] {
        assert!(IntEvalRsLigVirtProof::from_bytes(&bytes[..cut]).is_err());
    }
    // A flipped byte must not yield a DIFFERENT proof that still verifies.
    // Following the house convention (see tests/completeness_random.rs),
    // a decode PANIC on tampered bytes counts as a rejection.
    for &position in &[0usize, 40, bytes.len() / 2, bytes.len() - 20] {
        let mut tampered = bytes.clone();
        tampered[position] ^= 1;
        let decoded = catch_unwind(AssertUnwindSafe(|| {
            IntEvalRsLigVirtProof::from_bytes(&tampered)
        }))
        .ok()
        .and_then(|r| r.ok());
        if let Some(decoded) = decoded {
            let reproof = Proof::<IntEvalRsLigVirtProof>::from_parts(
                fx.proof.prefix().clone(),
                None,
                decoded,
            );
            assert!(
                verify_fixture(&fx, &reproof).is_err(),
                "tampered byte {position} verified"
            );
        }
    }
}

/// A false relation (one gate's `z` is NOT `x ∧ y`) with fully CONSISTENT
/// committed bits must not produce an accepting proof: the outer sumcheck
/// claim is zero but the true residual sum is not.
#[test]
fn cm_and_rejects_a_false_relation_with_consistent_bits() {
    let config = bitz::piop::spartan::spartan_bitz_field_config();
    let witness = CmAndWitness::from_gate_values(PRODUCTION_GATES, |i| {
        let r = splitmix(0xBAD ^ i as u64);
        let (x, y) = (r as u32, (r >> 32) as u32);
        let z = if i == 1 { (x & y) ^ 4 } else { x & y };
        (x, y, z, x ^ y)
    })
    .unwrap();
    let layout = *witness.layout();
    let relation = prepare_cm_and_relation(layout, &config).unwrap();
    let (pc, vc) = audited_configs(&relation);
    let hint = commit_cm_and_witness_with_config(&layout, witness.f_bit_rows(), &pc).unwrap();

    // The prover may panic on internal debug assertions (debug builds) or
    // complete; either way no accepting transcript may exist.
    let accepted = catch_unwind(AssertUnwindSafe(|| {
        let mut pt = Blake3Transcript::new();
        prove_cm_and_bitz_with_config(&mut pt, &relation, &witness, &hint, &pc)
    }))
    .ok()
    .and_then(|r| r.ok())
    .and_then(|proof| {
        let mut vt = Blake3Transcript::new();
        verify_cm_and_bitz_with_config(&mut vt, &relation, &hint.commitment, &proof, &vc).ok()
    });
    assert!(accepted.is_none(), "a false AND relation was accepted");
}

/// Spartan alone accepts these integers because `x + y - w - 2z = 0`, but
/// the virtual map requires `w = x XOR y`. The combined proof must therefore
/// reject even though the R1CS residual is exactly zero.
#[test]
fn cm_and_rejects_spartan_valid_but_xor_invalid_witness() {
    let config = bitz::piop::spartan::spartan_bitz_field_config();
    let witness = CmAndWitness::from_gate_values(PRODUCTION_GATES, |i| {
        if i == 0 {
            // 3 + 5 - 8 - 2*0 = 0, but 8 != 3 XOR 5.
            (3, 5, 0, 8)
        } else {
            let r = splitmix(0xC05E ^ i as u64);
            let (x, y) = (r as u32, (r >> 32) as u32);
            (x, y, x & y, x ^ y)
        }
    })
    .unwrap();
    let layout = *witness.layout();
    let relation = prepare_cm_and_relation(layout, &config).unwrap();
    let (pc, vc) = audited_configs(&relation);
    let hint = commit_cm_and_witness_with_config(&layout, witness.f_bit_rows(), &pc).unwrap();

    let accepted = catch_unwind(AssertUnwindSafe(|| {
        let mut pt = Blake3Transcript::new();
        prove_cm_and_bitz_with_config(&mut pt, &relation, &witness, &hint, &pc)
    }))
    .ok()
    .and_then(|result| result.ok())
    .and_then(|proof| {
        let mut vt = Blake3Transcript::new();
        verify_cm_and_bitz_with_config(&mut vt, &relation, &hint.commitment, &proof, &vc).ok()
    });
    assert!(
        accepted.is_none(),
        "a Spartan-valid witness with an invalid virtual XOR was accepted"
    );
}

/// An honest relation whose COMMITTED bits disagree with the Spartan
/// assignment (one flipped `z` bit in `f`) must be rejected: the
/// bitified terminal claim no longer matches the virtual read-off.
#[test]
fn cm_and_rejects_inconsistent_committed_bits() {
    let config = bitz::piop::spartan::spartan_bitz_field_config();
    let witness = CmAndWitness::from_fn(PRODUCTION_GATES, |i| {
        let r = splitmix(0xB17 ^ i as u64);
        (r as u32, (r >> 32) as u32)
    })
    .unwrap();
    let layout = *witness.layout();
    let relation = prepare_cm_and_relation(layout, &config).unwrap();
    let (pc, vc) = audited_configs(&relation);

    let mut rows = witness.f_bit_rows();
    // Flip one committed z bit of gate 0 (slot 64 = z bit 0).
    let (b, c) = layout.cell(64, 0).unwrap();
    rows[c][b / 64] ^= 1u64 << (b % 64);
    let hint = commit_cm_and_witness_with_config(&layout, rows, &pc).unwrap();

    let accepted = catch_unwind(AssertUnwindSafe(|| {
        let mut pt = Blake3Transcript::new();
        prove_cm_and_bitz_with_config(&mut pt, &relation, &witness, &hint, &pc)
    }))
    .ok()
    .and_then(|r| r.ok())
    .and_then(|proof| {
        let mut vt = Blake3Transcript::new();
        verify_cm_and_bitz_with_config(&mut vt, &relation, &hint.commitment, &proof, &vc).ok()
    });
    assert!(accepted.is_none(), "inconsistent committed bits were accepted");
}

/// A proof does not verify against a different statement (same capacity,
/// different live-gate count => different matrices and binding).
#[test]
fn cm_and_rejects_a_mismatched_statement() {
    let fx = honest_fixture(PRODUCTION_GATES + 77, 0x57A7);
    let config = bitz::piop::spartan::spartan_bitz_field_config();
    let other_layout = CmAndLayout::new(PRODUCTION_GATES + 78).unwrap();
    assert_eq!(other_layout.capacity(), fx.witness.layout().capacity());
    let other = prepare_cm_and_relation(other_layout, &config).unwrap();
    let mut vt = Blake3Transcript::new();
    assert!(
        verify_cm_and_bitz_with_config(&mut vt, &other, &fx.hint.commitment, &fx.proof, &fx.vc)
            .is_err()
    );
}

/// The ungated production entry points refuse unaudited (small) shapes.
#[test]
fn cm_and_production_entry_points_gate_small_shapes() {
    let config = bitz::piop::spartan::spartan_bitz_field_config();
    let witness = CmAndWitness::from_inputs(&[(1, 2)]).unwrap();
    let layout = *witness.layout();
    let relation = prepare_cm_and_relation(layout, &config).unwrap();
    let (pc, _) = adhoc_configs(&layout);
    let hint = commit_cm_and_witness_with_config(&layout, witness.f_bit_rows(), &pc).unwrap();
    let mut pt = Blake3Transcript::new();
    assert!(matches!(
        prove_cm_and_bitz(&mut pt, &relation, &witness, &hint),
        Err(ProtocolError::UnauditedBitzParameters)
    ));
}
