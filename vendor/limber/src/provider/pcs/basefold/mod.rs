//! BaseFold multilinear polynomial commitment (hash-based, field-agnostic).
//!
//! Work in progress. Built to give the hash-mode backend `O(log²n)` proofs
//! (vs Brakedown's `O(√n)`, ~9 MB), while staying on our sampled ~256-bit
//! field where WHIR/FRI cannot go (they need a smooth/FFT-friendly field;
//! BaseFold's random foldable code does not). Reuses the Brakedown Merkle
//! tree and deterministic sampling.
//!
//! Status: the cryptographic core AND the `CommitBackend` wiring are complete
//! and validated (round-trip + tamper tests + the code's defining invariants):
//!
//! - `code`: field-agnostic foldable code (encode/fold, evaluation basis)
//! - `query`: FRI-style proximity IOP (Merkle commit + fold/consistency)
//! - `open`: full multilinear opening (eq-sumcheck interleaved with the fold),
//!   over the crate `ByteTranscript`
//! - `backend`: `CommitBackend` impl (`BfBackend`) — commit / open / verify
//!
//! Next: multi-target batching (one shared fold), then Mod-PCS integration
//! (route the chunk-oracle / range-check opening through this backend).

pub mod backend;
pub mod code;
pub mod open;
pub mod query;
