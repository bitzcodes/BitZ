// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module implements the Spartan traits for BN254 (also known as BN256 or alt_bn128).
use crate::{
  impl_traits,
  provider::{
    msm::{msm, msm_shared_weights, msm_small},
    traits::{DlogGroup, DlogGroupExt, PairingGroup},
  },
  traits::{Group, PrimeFieldExt, transcript::TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::FromUniformBytes;
use halo2curves::{
  CurveAffine, CurveExt,
  bn256::{
    Fq12 as Bn256Fq12, G1 as Bn256G1, G1Affine as Bn256G1Affine, G2 as Bn256G2,
    G2Affine as Bn256G2Affine, Gt as Bn256Gt, multi_miller_loop as bn256_multi_miller_loop,
  },
  group::{Curve, Group as AnotherGroup, GroupEncoding, cofactor::CofactorCurveAffine},
  pairing::MillerLoopResult,
};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Num, ToPrimitive};
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

/// Re-exports that give access to the standard aliases used in the code base, for bn254
pub mod types {
  pub use halo2curves::bn256::{Fq as Base, Fr as Scalar, G1 as Point, G1Affine as Affine};
}

impl_traits!(
  types,
  Bn256G1,
  Bn256G1Affine,
  // Fr (scalar field) modulus
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  // Fq (base field) modulus
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
);

// Implement big_num traits for BN254 scalar field (Fr)
crate::impl_field_reduction_constants!(types::Scalar);
crate::impl_montgomery_limbs!(types::Scalar);

// BN254 is not a cycle pair, so we need to manually implement TranscriptReprTrait for the Base field
impl TranscriptReprTrait for types::Base {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_bytes().into_iter().rev().collect()
  }
}

// Absorbing G2 points into a G1-based transcript uses the compressed encoding.
impl TranscriptReprTrait for Bn256G2Affine {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    GroupEncoding::to_bytes(self).as_ref().to_vec()
  }
}

impl PairingGroup for types::Point {
  type G2 = Bn256G2;
  type G2Affine = Bn256G2Affine;
  type Gt = Bn256Gt;
  type MillerLoopOutput = Bn256Fq12;

  fn g2_generator() -> Self::G2 {
    Bn256G2::generator()
  }

  fn g2_to_affine(p: &Self::G2) -> Self::G2Affine {
    p.to_affine()
  }

  fn batch_g2_to_affine(points: &[Self::G2]) -> Vec<Self::G2Affine> {
    let mut affine = vec![Bn256G2Affine::identity(); points.len()];
    <Bn256G2 as Curve>::batch_normalize(points, &mut affine);
    affine
  }

  fn g2_from_affine(p: &Self::G2Affine) -> Self::G2 {
    Bn256G2::from(*p)
  }

  fn multi_miller_loop(
    pairs: &[(&Self::AffineGroupElement, &Self::G2Affine)],
  ) -> Self::MillerLoopOutput {
    bn256_multi_miller_loop(pairs)
  }

  fn final_exponentiation(m: &Self::MillerLoopOutput) -> Self::Gt {
    m.final_exponentiation()
  }
}

#[cfg(test)]
mod big_num_tests {
  crate::test_field_reduction_constants!(scalar_frc, crate::provider::bn254::types::Scalar);
  crate::test_montgomery!(scalar_mont, crate::provider::bn254::types::Scalar);
  crate::test_delayed_reduction!(scalar_dr, crate::provider::bn254::types::Scalar);
}

#[cfg(test)]
mod pairing_tests {
  use super::*;
  use crate::provider::traits::PairingGroup;
  use ff::Field;
  use rand_core::OsRng;

  // Bilinearity: e(a·g1, b·g2) = e(b·g1, a·g2).
  // Multi-pairing: e(a·g1, b·g2) · e(b·g1, -a·g2) · e(g1, g2) = e(g1, g2).
  #[test]
  fn test_pairing_bilinearity_and_batching() {
    type G1 = types::Point;
    let mut rng = OsRng;
    let a = types::Scalar::random(&mut rng);
    let b = types::Scalar::random(&mut rng);

    let g1 = G1::generator();
    let g2 = G1::g2_generator();
    let g1_aff = G1::affine(&g1);
    let g2_aff = G1::g2_to_affine(&g2);

    let ag1_aff = G1::affine(&(g1 * a));
    let bg1_aff = G1::affine(&(g1 * b));
    let ag2_aff = G1::g2_to_affine(&(g2 * a));
    let bg2_aff = G1::g2_to_affine(&(g2 * b));
    let neg_ag2_aff = G1::g2_to_affine(&(-(g2 * a)));

    let p1 = G1::pairing(&ag1_aff, &bg2_aff);
    let p2 = G1::pairing(&bg1_aff, &ag2_aff);
    assert_eq!(p1, p2, "bilinearity: e(a·g1, b·g2) != e(b·g1, a·g2)");

    let batched = G1::multi_pairing(&[
      (&ag1_aff, &bg2_aff),
      (&bg1_aff, &neg_ag2_aff),
      (&g1_aff, &g2_aff),
    ]);
    let expected = G1::pairing(&g1_aff, &g2_aff);
    assert_eq!(batched, expected, "multi-pairing batching mismatch");
  }
}
