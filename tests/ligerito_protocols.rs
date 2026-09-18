//! Integration checks for production entrypoints; these are correctness tests,
//! not timing or memory benchmarks.
use ::bitz::ligerito_flock::{LigeritoSelection, ResolvedLigerito};

#[cfg(feature = "ecdsa")]
#[test]
fn sha_ecdsa_both_regimes_preflight_all_supported_shapes() {
    use ::bitz::piop::spartan::ecdsa_sha256::{prepare_sha256_ecdsa, OuterMode};
    for exponent in 3..=16 {
        for mode in [OuterMode::Split, OuterMode::AllRows] {
            for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
                let prepared = prepare_sha256_ecdsa(exponent, 100, mode)
                    .unwrap()
                    .with_ligerito(selection)
                    .unwrap();
                assert!(
                    prepared.security().unwrap().compute_economic_security_bits() >= 100.0
                );
                assert_eq!(prepared.ligerito_configuration().selection(), selection);
            }
        }
    }
}

#[test]
fn cm_and_both_regimes_roundtrip_and_bind_roots() {
    use ::bitz::{piop::spartan::*, transcript::Blake3Transcript};
    let witness = CmAndWitness::from_fn(1 << 15, |i| (i as u32, u32::MAX)).unwrap();
    let field = spartan_bitz_field_config();
    for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
        let p = prepare_cm_and_relation(*witness.layout(), &field)
            .unwrap()
            .with_ligerito(selection)
            .unwrap();
        let r = p.ligerito_configuration().unwrap();
        let hint = cm::commit_cm_and_witness_with_config(
            witness.layout(),
            witness.f_bit_rows(),
            r.prover(),
        )
        .unwrap();
        let proof = prove_cm_and_bitz(&mut Blake3Transcript::new(), &p, &witness, &hint).unwrap();
        verify_cm_and_bitz(&mut Blake3Transcript::new(), &p, &hint.commitment, &proof).unwrap();
        let bytes = proof.bitz().to_bytes();
        let decoded = ::bitz::ligerito_flock::IntEvalRsLigVirtProof::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes(), bytes);
        let mut root = hint.commitment.clone();
        root.root[0] ^= 1;
        assert!(verify_cm_and_bitz(&mut Blake3Transcript::new(), &p, &root, &proof).is_err());
    }
}

#[cfg(feature = "hybrid")]
#[test]
fn balanced_hybrid_both_regimes_roundtrip() {
    use ::bitz::hybrid::*;
    let inputs: Vec<_> = (0..1 << 15).map(|i| (i, i + 1)).collect();
    let blocks = [[7; 16]; 128];
    for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
        let p = PreparedHybrid::new_with_ligerito(
            Parameters {
                multiplications: 1 << 15,
                sha_compressions: 128,
            },
            selection,
        )
        .unwrap();
        let committed = p.commit(&inputs, &blocks).unwrap();
        let proof = p.prove(&committed).unwrap();
        let decoded = p
            .proof_from_bytes(committed.statement(), &proof.to_bytes())
            .unwrap();
        p.verify(committed.statement(), &decoded).unwrap();
    }
}

#[test]
fn result_identity_roundtrip_rejects_tampered_config_and_metadata() {
    for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
        let r = selection.resolve(15, 100).unwrap();
        let report = r.report(&selection.name(), r.round0(100).unwrap());
        let encoded = ResolvedLigerito::encode_report(&report);
        assert_eq!(ResolvedLigerito::decode_report(&encoded).unwrap(), report);
        for key in [
            "regime",
            "configuration_fingerprint",
            "protocol_version",
            "recursive_ood",
        ] {
            let mut bad = report.clone();
            bad[key] = serde_json::Value::Null;
            assert!(ResolvedLigerito::validate_report(&bad).is_err());
        }
        let mut bad = report.clone();
        bad["configuration"]["initial_k"] = serde_json::json!(3);
        assert!(ResolvedLigerito::validate_report(&bad).is_err());
        assert!(ResolvedLigerito::decode_report("åx").is_err());
    }
}

#[cfg(feature = "bench-internals")]
#[test]
fn fixed98_product_layouts_preflight_without_witnesses() {
    use ::bitz::piop::spartan::sha256::prepare_sha256_compression_batch_for_product_t_fixed98;
    for t in 7..=28 {
        for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
            let p = prepare_sha256_compression_batch_for_product_t_fixed98(14, t)
                .unwrap()
                .with_ligerito(selection)
                .unwrap();
            assert_eq!(p.ligerito_configuration().unwrap().selection(), selection);
        }
    }
    assert!(prepare_sha256_compression_batch_for_product_t_fixed98(14, 6).is_err());
    assert!(prepare_sha256_compression_batch_for_product_t_fixed98(14, 29).is_err());
}
