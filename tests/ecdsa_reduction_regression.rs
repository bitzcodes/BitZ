#![cfg(feature = "ecdsa")]

#[path = "../benches/common/peak_memory.rs"]
mod peak_memory;

use bitz::{
    piop::spartan::ecdsa_sha256::{
        OuterMode, Sha256EcdsaStatement, commit_sha256_ecdsa, generate_sha256_ecdsa_witness,
        prepare_sha256_ecdsa, prove_sha256_ecdsa, verify_sha256_ecdsa,
    },
    transcript::{Blake3Transcript, traits::Transcript},
};
use p256::ecdsa::{Signature, SigningKey, signature::Signer};
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: peak_memory::PeakAlloc = peak_memory::PeakAlloc;

/// Run separately so allocator measurements exclude concurrent tests.
#[test]
#[ignore = "deterministic end-to-end proof compatibility and allocation measurement"]
fn proof_bytes_and_verifier_allocations() {
    for target in [100, 128] {
        for mode in [OuterMode::Split, OuterMode::AllRows] {
            let prepared = prepare_sha256_ecdsa(3, target, mode).unwrap();
            let message: Vec<_> = (0..prepared.message_bytes()).map(|i| i as u8).collect();
            let key = SigningKey::from_bytes((&[7u8; 32]).into()).unwrap();
            let signature: Signature = key.sign(&message);
            let point = key.verifying_key().to_encoded_point(false);
            let (r, s) = signature.split_bytes();
            let statement = Sha256EcdsaStatement {
                log_compressions: 3,
                qx: point.x().unwrap().as_slice().try_into().unwrap(),
                qy: point.y().unwrap().as_slice().try_into().unwrap(),
                r: r.into(),
                s: s.into(),
            };
            let witness = generate_sha256_ecdsa_witness(&prepared, &statement, &message).unwrap();
            let hint = commit_sha256_ecdsa(&prepared, &witness).unwrap();
            let mut prover = Blake3Transcript::new();
            let started = Instant::now();
            let proof =
                prove_sha256_ecdsa(&mut prover, &prepared, &statement, &witness, &hint, 4).unwrap();
            let prove_ms = started.elapsed().as_secs_f64() * 1000.;
            let digest = blake3::hash(&proof.to_bytes()).to_hex().to_string();
            let mut verifier = Blake3Transcript::new();
            let live = peak_memory::live_bytes();
            peak_memory::reset_peak();
            let started = Instant::now();
            verify_sha256_ecdsa(
                &mut verifier,
                &prepared,
                &statement,
                &hint.commitment,
                &proof,
            )
            .unwrap();
            let verify_ms = started.elapsed().as_secs_f64() * 1000.;
            let verify_peak = peak_memory::peak_bytes().saturating_sub(live);
            let challenge = prover.get_challenge::<u128>();
            assert_eq!(challenge, verifier.get_challenge::<u128>());
            // Pins are refreshed only after verification, for intentional
            // shared-codec or transcript changes.
            let (expected_digest, expected_challenge) = match (target, mode) {
                (100, OuterMode::Split) => (
                    "dd7914e068445d04c621181c115c5a2129430418ef6e150325e43bcd6d69067a",
                    161748810141005428170281968810474012494,
                ),
                (100, OuterMode::AllRows) => (
                    "ad33c0eb4ab212a52b581e7e5e46e00ed106101f59d2a69242fa318a4fda6682",
                    242733522091643991383879358049197793165,
                ),
                (128, OuterMode::Split) => (
                    "a600b6038c7226a643f27c2fc2d12543db3b60d94095e50f9153b408449d3911",
                    237220613375296302577607028912355843102,
                ),
                (128, OuterMode::AllRows) => (
                    "e3f508c1632914e532dc3c7076643dc426f7be0124832a5cdd542dbb1070ba7c",
                    18240983802203675248285318561107227572,
                ),
                _ => unreachable!(),
            };
            if std::env::var_os("BITZ_RECORD_PINS").is_none() {
                assert_eq!(digest, expected_digest, "target={target} mode={mode:?}");
                assert_eq!(challenge, expected_challenge);
            }
            println!(
                "target={target} mode={mode:?} digest={digest} challenge={challenge} prove_ms={prove_ms:.2} verify_ms={verify_ms:.2} verify_peak={verify_peak}"
            );
        }
    }
}
