//! Constructive vectors shared by native and Binius integration tests.
use crate::fixture::SignedFixture;
use p256::{
    NistP256, ProjectivePoint, PublicKey, Scalar,
    ecdsa::Signature,
    elliptic_curve::{
        Curve,
        bigint::{Encoding, U256},
        ops::Reduce,
        sec1::ToEncodedPoint,
    },
};
use sha2::{Digest, Sha256};

pub fn vectors() -> Vec<SignedFixture> {
    let original = SignedFixture::generate(3, 0).unwrap();
    let signature = Signature::from_scalars(original.r, original.s).unwrap();
    let mut alternate = original.clone();
    alternate.s = (-signature.s().as_ref()).to_bytes().into();
    alternate.id = alternate.compute_id();
    alternate.validate().unwrap();

    // R.x=n+3 is on P-256. Choose r=3,s=1 and Q=(R-zG)/3, so
    // R=zG+rQ for this actual SHA-chain message. No rare nonce search is needed.
    let mut exceptional = original.clone();
    let x = NistP256::ORDER
        .wrapping_add(&U256::from(3u64))
        .to_be_bytes();
    let mut compressed = [0; 33];
    compressed[0] = 2;
    compressed[1..].copy_from_slice(&x);
    let r_point = PublicKey::from_sec1_bytes(&compressed).unwrap();
    let digest: [u8; 32] = Sha256::digest(&exceptional.message).into();
    let z = <Scalar as Reduce<U256>>::reduce_bytes(&digest.into());
    let q = (ProjectivePoint::from(*r_point.as_affine()) - ProjectivePoint::GENERATOR * z)
        * Scalar::from(3u64).invert().unwrap();
    let q = q.to_affine().to_encoded_point(false);
    exceptional.qx = (*q.x().unwrap()).into();
    exceptional.qy = (*q.y().unwrap()).into();
    exceptional.r = [0; 32];
    exceptional.r[31] = 3;
    exceptional.s = [0; 32];
    exceptional.s[31] = 1;
    exceptional.id = exceptional.compute_id();
    exceptional.validate().unwrap();
    vec![original, alternate, exceptional]
}
