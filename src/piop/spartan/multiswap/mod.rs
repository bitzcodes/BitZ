//! MultiSwap through BitZ: Limber's integer Mod-R1CS proved with a Spartan
//! PIOP over a transcript-sampled prime and an BitZ bit-commitment opening.
//!
//! This module realizes, on Limber's own flagship benchmark circuit, the
//! comparison BitZ targets: Limber commits limb-split integers with a wrapped
//! PCS and range-checks every limb with LogUp-GKR; BitZ commits the same
//! values as raw bits — a committed F2 bit is in `{0,1}` by construction —
//! and binds their integer evaluations through the exponent-fold forest.
//!
//! Pipeline: [`circuit`] builds the wired 6209-row instance ported verbatim
//! from `lucasxia01/limber-impl`; [`relation`] folds the per-row moduli into
//! the output matrix and lays the witness and quotient values out as a
//! `2^25`-bit BitZ tensor; [`prime`] samples the post-commitment 128-bit
//! fingerprint prime and the 113-bit Step 5.0 reduction prime; [`reduce`]
//! implements Step 5.0 (the paper's large-field modulus reduction);
//! [`proof`] runs the stock Spartan PIOP over the fingerprint field and
//! discharges the reduced terminal claim with the runtime-prime BitZ
//! opening.  The two-prime schedule reaches the documented 114-bit floors
//! with a single `2^10`-hash grind.

pub mod circuit;
pub mod prime;
pub mod proof;
pub mod reduce;
pub mod relation;

pub use circuit::{
    MULTISWAP_VALUE_BITS, MultiswapCircuit, MultiswapCircuitError, MultiswapDims, modulus_ell,
    modulus_n, modulus_p_hash,
};
pub use prime::{
    MultiswapPrimeError, MultiswapPrimeProfile, sample_multiswap_fingerprint_context,
    sample_multiswap_reduction_prime,
};
pub use proof::{
    MultiswapSpec, PreparedMultiswapRelation, commit_multiswap_witness, multiswap_lig_configs,
    prove_multiswap_mod_r1cs, verify_multiswap_mod_r1cs,
};
pub use reduce::{step50_accepts_lift, step50_integer_lift, step50_mu_prime_bound, step50_reduce};
pub use relation::{
    MULTISWAP_QUOS_SLOT_START, MULTISWAP_SLOT_VARS, MULTISWAP_SLOTS, MULTISWAP_W_SLOT_START,
    MultiswapAssignment, MultiswapIntegerRelation, MultiswapLayout, MultiswapLayoutError,
};
