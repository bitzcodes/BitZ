// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module provides implementations of polynomial commitment schemes (PCS).

// helper code for polynomial commitment schemes
pub mod ipa;

// implementations of polynomial commitment schemes
pub mod basefold;
pub mod brakedown;
pub(crate) mod commit_backend;
pub mod hyrax_pc;
pub mod integer_modpcs;

pub use integer_modpcs::f_chunk_len;

/// Active Brakedown settings bound into Mod-PCS keys: target, spec, row length,
/// and direct-opening threshold. Exposed for benchmark provenance.
pub fn brakedown_configuration() -> [usize; 4] {
  commit_backend::bd_configuration()
}

/// Pre-build the deterministic Brakedown layout for a given polynomial
/// length (public code matrices; conceptually setup work). Returns the
/// column-open count for informational use.
pub fn prewarm_brakedown_params(n: usize) -> usize {
  commit_backend::bd_params::<crate::provider::pt256::t256::Scalar>(n).n_col_opens
}

/// Snapshot of the Brakedown retained-data cache's audit counters.
/// Benchmark support: the Poseidon2 bench resets the cache before every
/// audited iteration and records these counts in its run manifest.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct BdRetainedCacheStats {
  /// Retained-data lookups served from the cache.
  pub hits: u64,
  /// Retained-data lookups that found no entry.
  pub misses: u64,
  /// Recommit-path lookups that forced a full re-encode (a subset of the
  /// misses; plain first-time commits are not re-encodes).
  pub reencodes: u64,
  /// Wholesale clears triggered by the 8-entry insertion bound.
  pub wholesale_clears: u64,
}

/// Clear the Brakedown retained-data cache and zero the t256 audit
/// counters, giving the deterministic empty-cache state every measured
/// Brakedown benchmark iteration starts from. Benchmark support
/// (Criterion benches compile as separate crates, and `commit_backend`
/// is `pub(crate)`, hence this public doc-hidden wrapper).
#[doc(hidden)]
pub fn bd_retained_cache_reset() {
  commit_backend::bd_retained_cache_reset_for::<crate::provider::pt256::t256::Scalar>();
}

/// Snapshot the t256 Brakedown retained-cache audit counters (benchmark
/// support; see [`bd_retained_cache_reset`]).
#[doc(hidden)]
pub fn bd_retained_cache_stats() -> BdRetainedCacheStats {
  commit_backend::bd_retained_cache_stats_for::<crate::provider::pt256::t256::Scalar>()
}
