//! Polynomial types for IntMod-Spartan, generic over `SumcheckField`.
//!
//! Parallel to `src/polys/` — same protocol math, but bounded on
//! `SumcheckField` (no static modulus assumption) so the types work
//! over both static curve scalars and dynamic-prime fields like
//! `DynPrime`.
//!
//! Deliberately minimal: only the methods that `sumcheck_modp.rs` and
//! `imod_spartan_modp` need. Optimizations from `src/polys/` (zero-skip
//! prefixes, delayed-reduction accumulators, fast bind paths) are
//! omitted — they're either static-modulus-specific or can be added
//! back later if benchmarks demand.

pub mod eq;
pub mod multilinear;
pub mod univariate;
