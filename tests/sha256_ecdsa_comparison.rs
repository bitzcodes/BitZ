#![cfg(feature = "sha256-ecdsa-compare")]

#[path = "../benches/support/sha256_ecdsa_fixture.rs"]
mod fixture;
#[path = "../benches/support/sha256_ecdsa_test_vectors.rs"]
mod vectors;

use bitz::{piop::spartan::ecdsa_sha256::*, transcript::Blake3Transcript};

#[test]
fn both_s_forms_and_exceptional_nonce_verify_in_all_native_methods() {
    let fixtures = vectors::vectors();
    for mode in [OuterMode::Split, OuterMode::AllRows] {
        let prepared = prepare_sha256_ecdsa(3, 100, mode).unwrap();
        for f in &fixtures {
            let statement = Sha256EcdsaStatement {
                log_compressions: 3,
                qx: f.qx,
                qy: f.qy,
                r: f.r,
                s: f.s,
            };
            let witness = generate_sha256_ecdsa_witness(&prepared, &statement, &f.message).unwrap();
            let hint = commit_sha256_ecdsa(&prepared, &witness).unwrap();
            let proof = prove_sha256_ecdsa(
                &mut Blake3Transcript::new(),
                &prepared,
                &statement,
                &witness,
                &hint,
                4,
            )
            .unwrap();
            verify_sha256_ecdsa(
                &mut Blake3Transcript::new(),
                &prepared,
                &statement,
                &hint.commitment,
                &proof,
            )
            .unwrap();
            for field in ["exponent", "qx", "qy", "r", "s"] {
                let mut changed = statement.clone();
                match field {
                    "exponent" => changed.log_compressions += 1,
                    "qx" => changed.qx[31] ^= 1,
                    "qy" => changed.qy[31] ^= 1,
                    "r" => changed.r[31] ^= 1,
                    "s" => changed.s[31] ^= 1,
                    _ => unreachable!(),
                }
                assert!(
                    verify_sha256_ecdsa(
                        &mut Blake3Transcript::new(),
                        &prepared,
                        &changed,
                        &hint.commitment,
                        &proof,
                    )
                    .is_err(),
                    "accepted changed {field} in {mode:?}"
                );
            }
        }
    }

}

#[test]
fn changed_private_message_cannot_prove_the_original_signature() {
    let fixture = fixture::SignedFixture::generate(3, 0).unwrap();
    let statement = Sha256EcdsaStatement {
        log_compressions: 3,
        qx: fixture.qx,
        qy: fixture.qy,
        r: fixture.r,
        s: fixture.s,
    };
    let mut changed = fixture.message;
    changed[0] ^= 1;
    for mode in [OuterMode::Split, OuterMode::AllRows] {
        let prepared = prepare_sha256_ecdsa(3, 100, mode).unwrap();
        // Regenerate all hints for the changed message without validating the
        // fixture: rejection must come from the composed proof relation.
        let witness = generate_sha256_ecdsa_witness(&prepared, &statement, &changed).unwrap();
        let hint = commit_sha256_ecdsa(&prepared, &witness).unwrap();
        if let Ok(proof) = prove_sha256_ecdsa(
            &mut Blake3Transcript::new(),
            &prepared,
            &statement,
            &witness,
            &hint,
            4,
        ) {
            assert!(
                verify_sha256_ecdsa(
                    &mut Blake3Transcript::new(),
                    &prepared,
                    &statement,
                    &hint.commitment,
                    &proof,
                )
                .is_err(),
                "accepted changed message in {mode:?}"
            );
        }
    }
}

#[test]
fn fixtures_roundtrip_and_support_full_length_range() {
    for exponent in [6, 11, 16] {
        let f = fixture::SignedFixture::generate(exponent, 1).unwrap();
        assert_eq!(f.message.len(), 64 * ((1usize << exponent) - 1));
        let path = std::env::temp_dir().join(format!(
            "sha256-ecdsa-{}-i{exponent}.json",
            std::process::id()
        ));
        f.write(&path).unwrap();
        let decoded = fixture::SignedFixture::read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(decoded.id, f.id);
    }
}
