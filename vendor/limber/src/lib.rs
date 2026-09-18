// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Limber: SNARKs for integer Mod-R1CS — constraints with arbitrary,
//! per-row integer moduli — built on Spartan's sum-check pipeline.
//!
//! The integer proof system lives in [`imod_spartan_modp`] (the
//! dual-field driver: sumcheck over a verifier-sampled prime `p`, with
//! an integer Mod-PCS bridging to the commitment field `q`) and its
//! relation [`imod_r1cs_modp`]; the Mod-PCS is
//! `provider::pcs::integer_modpcs`, instantiable over Hyrax (curve mode)
//! or Brakedown (hash mode). [`imod_spartan`] / [`imod_r1cs`] are the
//! simpler single-field (`p = q`) prototype. The fork also retains the
//! original PCS-generic Spartan ([`spartan`]).
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![deny(unsafe_code)]

// private modules
mod digest;
mod math;
mod r1cs;

#[macro_use]
mod macros;

// public modules
pub mod bellpepper;
#[doc(hidden)]
pub mod bench_trace;
pub mod errors;
pub mod provider;
pub mod traits;

// internal modules
mod big_num;
mod polys;
mod sumcheck;

// ECDSA-verify MSM circuit gadgets (secp256k1): row-count / prove-time
// experiments, test-only.
#[cfg(test)]
mod ecdsa_msm;

// dynamic-prime (dual-field) sumcheck stack
pub mod dyn_prime;
pub mod polys_modp;
pub mod sumcheck_modp;

// public modules for proof systems
pub mod imod_r1cs; // Integer Mod-R1CS relation (paper Def 5.4)
pub mod imod_r1cs_modp; // Integer Mod-R1CS over a ModEngine (dual-field, verifier-sampled prime)
pub mod imod_spartan; // Spartan over Integer Mod-R1CS
pub mod imod_spartan_modp; // Dual-field SNARK driver over a ModEngine
pub mod logup_gkr; // LogUp-GKR range proof (16-bit range check via fractional-sum GKR)
pub mod multiswap; // OWWB20 MultiSwap statement: primitives and instance generation
#[doc(hidden)] // bench/test support; not part of the supported API
pub mod poseidon2;
#[doc(hidden)] // bench/test support; not part of the supported API
pub mod poseidon_bench;
pub mod spartan; // Spartan without zero-knowledge

/// Start a tracing span plus an exact benchmark interval and wall-clock timer.
///
/// The owned entered-span guard is deliberately returned to the caller.  The
/// previous implementation returned the `Span` after dropping its entered
/// guard inside this macro, so nested tracing subscribers never observed the
/// body of the operation.  Keeping both guards alive also lets the matched
/// benchmark campaign capture the existing phase annotations without changing
/// protocol code or its transcript.
macro_rules! start_span {
    ($name:expr $(, $($fmt:tt)+)?) => {{
        let span = tracing::info_span!($name $(, $($fmt)+)?);
        let guard = crate::bench_trace::SpanGuard::new(
            span.entered(),
            crate::bench_trace::scope($name),
        );
        (guard, std::time::Instant::now())
    }};
}
pub(crate) use start_span;

// The default width used for monolithic commitments.
pub(crate) const DEFAULT_COMMITMENT_WIDTH: usize = 2048;

use traits::{Engine, pcs::PCSEngineTrait};
type CommitmentKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::CommitmentKey;
type VerifierKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::VerifierKey;
type Commitment<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Commitment;
type PCS<E> = <E as Engine>::PCS;
type Blind<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Blind;
