//! Shape-only preflights: no large witness, commitment, or proof allocation.
use crate::piop::spartan::baby_bear_mul::BabyBearMulLayout;
use crate::piop::spartan::mul::MulLayout;
use crate::piop::spartan::protocol::PreparedRelation;

use super::*;
use crate::piop::spartan::{self, IopSecurityProfile, Lambda100, Lambda128};

#[test]
fn commitment_validation_preserves_typed_errors() {
    let mutations: [(&str, fn(&mut LigVerifierConfig)); 8] = [
        ("empty inverse rates", |c| c.log_inv_rates.clear()),
        ("empty OOD samples", |c| c.ood_samples.clear()),
        ("zero recursive levels", |c| c.recursive_steps = 0),
        ("level count overflow", |c| c.recursive_steps = usize::MAX),
        ("message dimension overflow", |c| {
            c.initial_log_msg_cols = usize::MAX
        }),
        ("block length overflow", |c| {
            c.log_inv_rates[0] = usize::BITS as usize
        }),
        ("oversized query count", |c| c.queries[0] = usize::MAX),
        ("zero fold dimension", |c| c.recursive_ks[0] = 0),
    ];

    for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
        let resolved = selection.resolve(15, 100).unwrap();
        let pc = resolved.prover();
        let vc = resolved.verifier();
        let commitment = Commitment {
            root: [0; 32],
            params: PcsParams {
                m: 15 + LOG_PACKING,
                log_inv_rate: pc.log_inv_rates[0],
                log_batch_size: pc.initial_k,
                profile: ligerito::LigeritoProfile::Fast,
                merkle_hash: pc.merkle_hash,
            },
        };
        assert_eq!(validate_ligerito_commitment(&commitment, pc), Ok(()));
        assert_eq!(validate_ligerito_commitment(&commitment, vc), Ok(()));

        for (name, mutate) in mutations {
            let mut malformed = vc.clone();
            mutate(&mut malformed);
            assert_eq!(
                validate_ligerito_commitment(&commitment, &malformed),
                Err(FlockRsError::CommitmentConfig),
                "{selection:?}: {name}"
            );
        }

        let mut mismatched = commitment.clone();
        mismatched.params.m += 1;
        assert_eq!(
            validate_ligerito_commitment(&mismatched, vc),
            Err(FlockRsError::CommitmentConfig)
        );
    }
}

fn check(params: crate::pcs::IntegerMatrixLayout, facts: spartan::IopInstanceFacts) {
    for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
        let resolved = selection
            .resolve(crate::ligerito::packed_vars(&params), 100)
            .unwrap();
        let mut security = Lambda100::instantiate(&facts).unwrap();
        security.adopt_ood_round(resolved.ood_bits()).unwrap();
        assert_eq!(
            security.ood.is_some(),
            selection == LigeritoSelection::JOHNSON
        );
        assert!(security.accounting.achieved_bits() >= 100.);
        ResolvedLigerito::validate_report(&resolved.report(&selection.name(), security.ood))
            .unwrap();
    }
}

#[test]
fn multiplication_shapes_preflight_without_witnesses() {
    for exponent in 15..=28 {
        for width in [1, 8] {
            let layout = MulLayout::<u32>::new_with_word_bits(1 << exponent, width).unwrap();
            let p = layout.bitz_params();
            check(p, spartan::bitz::u32_mul_instance_facts(&p, exponent));
        }
        let p = spartan::BabyBearMulLayout::new(1 << exponent)
            .unwrap()
            .bitz_params();
        check(
            p,
            spartan::baby_bear_bitz::baby_bear_mul_instance_facts(&p, exponent),
        );
        let cm = spartan::cm::CmAndLayout::new(1 << exponent)
            .unwrap()
            .bitz_params();
        for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
            selection
                .resolve(crate::ligerito::packed_vars(&cm), 100)
                .unwrap()
                .round0(100)
                .unwrap();
        }
        if exponent <= 27 {
            let p = MulLayout::<u64>::new(1 << exponent).unwrap().bitz_params();
            check(p, spartan::u64_bitz::u64_mul_instance_facts(&p, exponent));
        }
        if exponent <= 26 {
            let p = MulLayout::<u128>::new(1 << exponent).unwrap().bitz_params();
            check(p, spartan::u128_bitz::u128_mul_instance_facts(&p, exponent));
        }
    }
    for m in [19, 36] {
        assert!(LigeritoSelection::JOHNSON.resolve(m - 7, 100).is_err());
    }
    assert!(
        PreparedRelation::<MulLayout<u32>>::new(MulLayout::<u32>::new(1 << 14).unwrap()).is_err()
    );
    assert!(
        PreparedRelation::<MulLayout<u64>>::new(MulLayout::<u64>::new(1 << 14).unwrap()).is_err()
    );
    assert!(
        PreparedRelation::<MulLayout<u128>>::new(MulLayout::<u128>::new(1 << 14).unwrap()).is_err()
    );
    assert!(
        PreparedRelation::<BabyBearMulLayout>::new(
            spartan::BabyBearMulLayout::new(1 << 14).unwrap()
        )
        .is_err()
    );
}

#[test]
fn sha_profiles_layouts_and_boundaries_preflight() {
    use spartan::sha256::*;
    for exponent in 7..=16 {
        for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
            let p = prepare_sha256_compression_batch(exponent)
                .unwrap()
                .with_ligerito(selection)
                .unwrap();
            let r = p.ligerito_configuration().unwrap();
            ResolvedLigerito::validate_report(&r.report(&selection.name(), p.security().ood))
                .unwrap();
            let p = prepare_sha256_chain_batch(exponent)
                .unwrap()
                .with_ligerito(selection)
                .unwrap();
            assert_eq!(
                p.security().ood.is_some(),
                selection == LigeritoSelection::JOHNSON
            );
            let rows = prepare_sha256_compression_batch_for_assignment_rows(exponent + 14)
                .unwrap()
                .with_ligerito(selection)
                .unwrap();
            assert!(rows.ligerito_configuration().is_ok());
            let inner = prepare_sha256_compression_batch_with_profile_and_layout::<Lambda100>(
                exponent,
                Sha256OpeningLayout::InnerSumcheck { row_vars: 13 },
            )
            .unwrap()
            .with_ligerito(selection)
            .unwrap();
            assert!(inner.ligerito_configuration().is_ok());
        }
        match prepare_sha256_compression_batch_with_profile::<Lambda128>(exponent) {
            Ok(high) => assert_eq!(
                high.ligerito_configuration().unwrap().selection(),
                LigeritoSelection::ValidatedUdr
            ),
            Err(Sha256ConstraintError::Profile(spartan::ProfileError::GrindingTooExpensive {
                bits,
                cap,
                ..
            })) => {
                assert_eq!(cap, 24);
                assert!(bits > cap);
                eprintln!(
                    "preserved Lambda128 grinding-cap boundary at SHA K={exponent}: {bits}>{cap}"
                );
            }
            Err(error) => panic!("unexpected Lambda128 preflight error at K={exponent}: {error}"),
        }
    }
    for exponent in [0, 4, 5, 6, 17] {
        assert!(prepare_sha256_compression_batch(exponent).is_err());
        assert!(prepare_sha256_chain_batch(exponent).is_err());
    }
}

#[test]
fn explicit_legacy_profiles_and_identity_validation() {
    for request in [
        "udr:1:4",
        "udrg:1:4",
        "custom:1:4",
        "udrg:3:4",
        "custom:3:4",
    ] {
        let r = LigeritoSelection::parse(request, 100)
            .unwrap()
            .resolve(15, 100)
            .unwrap();
        let report = r.report(request, r.round0(100).unwrap());
        ResolvedLigerito::validate_report(&report).unwrap();
        for key in [
            "configuration_fingerprint",
            "protocol_version",
            "requested_profile",
            "outer_ood",
        ] {
            let mut bad = report.clone();
            bad.as_object_mut().unwrap().remove(key);
            assert!(ResolvedLigerito::validate_report(&bad).is_err(), "{key}");
        }
    }
    assert_eq!(
        LigeritoSelection::for_target(100),
        LigeritoSelection::JOHNSON
    );
    assert_eq!(
        LigeritoSelection::for_target(128),
        LigeritoSelection::ValidatedUdr
    );
    assert!(LigeritoSelection::parse("custom:3:4:100", 106).is_err());
}
