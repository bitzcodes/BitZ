// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module defines various traits required by the users of the library to implement.
use core::fmt::Debug;
use ff::{PrimeField, PrimeFieldBits};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

pub mod circuit;
pub mod mod_engine;
pub mod pcs;
pub mod snark;
pub mod transcript;

use pcs::PCSEngineTrait;
use transcript::{TranscriptEngineTrait, TranscriptReprTrait};

pub use crate::big_num::{DelayedReduction, FieldReductionConstants, MontgomeryLimbs};

/// Represents an element of a group
/// This is currently tailored for an elliptic curve group
pub trait Group: Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq {
  /// A type representing an element of the base field of the group
  type Base: PrimeFieldBits + Serialize + for<'de> Deserialize<'de>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeFieldBits + PrimeFieldExt + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Returns A, B, the order of the group, the size of the base field as big integers
  fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt);
}

/// A collection of engines that are required by the library
pub trait Engine: Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq + 'static {
  /// A type representing an element of the base field of the group
  type Base: PrimeFieldBits + TranscriptReprTrait + Serialize + for<'de> Deserialize<'de>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeFieldBits
    + PrimeFieldExt
    + Send
    + Sync
    + TranscriptReprTrait
    + Serialize
    + for<'de> Deserialize<'de>
    + FieldReductionConstants
    + DelayedReduction<Self::Scalar>
    + MontgomeryLimbs;

  /// A type that represents an element of the group
  type GE: Group<Base = Self::Base, Scalar = Self::Scalar> + Serialize + for<'de> Deserialize<'de>;

  /// A type that provides a generic Fiat-Shamir transcript to be used when externalizing proofs
  type TE: TranscriptEngineTrait<Self>;

  /// A type that defines a commitment engine over scalars in the group
  type PCS: PCSEngineTrait<Self>;
}

/// Defines additional methods on `PrimeField` objects. The
/// [`DelayedReduction`] supertrait gives every such field a wide
/// unreduced accumulator (Montgomery fields via the blanket impl;
/// others supply an eager one) — hot dot-product kernels like the
/// expander encoder depend on it.
pub trait PrimeFieldExt: PrimeField + DelayedReduction<Self> {
  /// Returns a scalar representing the bytes
  fn from_uniform(bytes: &[u8]) -> Self;

  /// Lift a `CHUNK_BITS`-sized value (< 2^16) into the field. Default
  /// is a plain `from`; engines may override with a cached table (the
  /// curve scalars do — one Montgomery multiplication per lift adds up
  /// over 2^20-element chunk polynomials).
  fn from_chunk(c: u64) -> Self {
    Self::from(c)
  }

  /// `self · v · 2^-64 (mod p)` — the uniform `2^-64` scale is fixed by
  /// pre-multiplying the OTHER factor once with [`Self::scale_shift64`].
  /// The default is correct but slow (a full multiplication plus a
  /// `2^-64` fixup); 4-limb Montgomery fields override it with a
  /// one-fold multiply (~4x cheaper). The GKR leaf-skip accumulates
  /// small-integer data against field weights with it.
  fn mul_u64_scaled(&self, v: u64) -> Self {
    let inv64 = {
      // TWO_INV^64 via 6 squarings.
      let mut t = Self::TWO_INV;
      for _ in 0..6 {
        t = t.square();
      }
      t
    };
    *self * Self::from(v) * inv64
  }

  /// Signed companion of [`Self::mul_u64_scaled`].
  fn mul_i64_scaled(&self, v: i64) -> Self {
    let m = self.mul_u64_scaled(v.unsigned_abs());
    if v < 0 { -m } else { m }
  }

  /// `self · 2^64` — pre-scales a factor so that a subsequent
  /// [`Self::mul_u64_scaled`] accumulation comes out exact.
  fn scale_shift64(&self) -> Self {
    *self * Self::from(1u64 << 63).double()
  }
}
