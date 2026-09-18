//! The MultiSwap statement of OWWB20 (*Scaling Verifiable Computation
//! Using Efficient Set Accumulators*, USENIX Security 2020): a batch of
//! swaps against an RSA accumulator verified through two Wesolowski
//! proofs of exponentiation sharing one hash-to-prime challenge. This
//! module ports the reference implementation's primitives
//! (`bellman-bignat`) so the `multiswap_modp` benchmark can prove a
//! real, fully wired instance:
//!
//! - [`poseidon`]: Poseidon over the BLS12-381 scalar field with the
//!   reference parameters (`t = 6`, `R_F = 8`, `R_P = 57`);
//! - [`mimc`]: the MiMC-7 permutation used to stretch the entropy pool;
//! - [`pocklington`]: hash-to-prime via recursive Pocklington
//!   certificates (plan, entropy source, search, native verification);
//! - [`accumulator`]: the quotient group `(Z/N)^*/{±1}`, the
//!   division-intractable element hash, Wesolowski proofs, and instance
//!   generation;
//! - [`circuit`]: the Integer Mod-R1CS row builder and gadgets;
//! - [`hash_gadgets`]: Poseidon and MiMC-7 as rows;
//! - [`statement`]: the statement assembly (full and bare configurations).
//!
//! Deviations from the reference, all cost-neutral or documented in the
//! benchmark: the Poseidon MDS matrix is drawn from a Keccak-seeded
//! stream rather than the reference's ChaCha stream, and the challenge
//! hash absorbs the digests as nine 240-bit limbs instead of sixty-four
//! 32-bit limbs.

pub mod accumulator;
pub mod circuit;
pub mod hash_gadgets;
pub mod mimc;
pub mod pocklington;
pub mod poseidon;
pub mod statement;
