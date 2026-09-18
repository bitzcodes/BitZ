//! # BitZ — an integer-MLE-evaluation PCS over an `F_2` commitment
//!
//! BitZ proves `MLE[INT(D)](r) = y ∈ F_q` for data `D` committed over a
//! cheap characteristic-2 (`F_2`) code, by folding the row variables **in
//! the exponent** of `K = GF(2^128)` (`α^{v_c} = ∏_b α^{w_b·D[(b,c)]}`,
//! certified by a GKR grand-product forest that touches the commitment only
//! through `K`-linear queries) and reading the column combination off in the
//! clear over `F_q`. Construction due to Lev Soukhanov (char2-fieldswitch §9);
//! this crate is the standalone extraction of the most-optimized
//! implementation from the `zinc-plus` repository.
//!
//! ## Opener
//!
//! The **only** PCS opener is the flock-backed **ring-switch + recursive
//! Ligerito** pipeline ([`ligerito_flock`]): the committed bit-matrix is
//! packed 128 bits per `GF(2^128)` element and RS-encoded/Merkleized by
//! [`flock-core`](flock_core); each mod-`q` limb chunk contributes a
//! [merged product forest][merged_forest] + a de-black-boxing pre-sumcheck,
//! and the L chunk claims are `η`-batched into ONE recursive Ligerito call
//! whose closing residual is evaluated succinctly by the ring-switch
//! tensor-algebra ([`ligerito::tensor_eq_phi_eval`]).
//!
//! Entry points: [`ligerito_flock::commit_rs_flock`] /
//! [`ligerito_flock::commit_rs_flock_with`],
//! [`ligerito_flock::prove_mle_eval_mod_q_ligerito`],
//! [`ligerito_flock::verify_mle_eval_mod_q_ligerito`], with the proof object
//! [`ligerito_flock::IntEvalRsLigModQProof`] and its
//! [`to_bytes`][ligerito_flock::IntEvalRsLigModQProof::to_bytes] /
//! [`from_bytes`][ligerito_flock::IntEvalRsLigModQProof::from_bytes] host
//! codec. See `docs/DESIGN.md` for the protocol and the serialization format.
//!
//! ## F₂-virtualization
//!
//! Claims about a DERIVED vector `h = M·f` (a public canonical CSC
//! [`F₂`-linear map][circuit::linear_map::binary::PreparedVirtualMap] of the committed bits) are
//! opened against the commitment to `f` alone —
//! [`ligerito_flock::prove_mle_eval_mod_q_ligerito_virtual`] /
//! [`ligerito_flock::verify_mle_eval_mod_q_ligerito_virtual`]: the
//! synthesis supplies both `f` and `h`; per-chunk forests and pre-sumchecks
//! run on `h` without ever touching the oracle, the terminal claims are
//! transposed through `Mᵀ` at the
//! commitment field (XOR is addition in char 2), and the transposed
//! arbitrary-weight inner product is opened NATIVELY by the dual-basis
//! ring switch ([`dual_basis`], the paper's bilinear-embedding batching
//! protocol): a 128-element plane message `h_i`, one zero-evader `ρ`,
//! and ONE Ligerito call — no bridge sumcheck, no point opening. The
//! verifier's `M`-dependent cost is `O(L·nnz + #cols)` field ops. When `M`
//! is the identity on a shared row layout the opening
//! routes to the plain base path instead (the identity fast path,
//! `BITZ_VIRT_ID_FAST`), skipping the derived-vector machinery entirely.
//! [`piop::spartan::cm`] wires a full R1CS through this path — the
//! paper's CM relation: batched `x ∧ y = z` via one LINEAR constraint
//! per gate with `w = x ⊕ y` as a virtual (derived, uncommitted) block.

pub mod binary_pcs;
#[cfg(feature = "binius64-bench")]
pub mod binius_ligerito;
pub mod dual_basis;
pub mod ext_proj;
pub mod f2map;
#[cfg(feature = "hybrid")]
pub mod hybrid;
pub mod ligerito;
pub mod ligerito_flock;
pub mod merged_forest;
#[cfg(feature = "span-metrics")]
pub mod observability;
pub mod pcs;
pub mod piop;
pub mod poly;
pub mod proof_codec;

pub mod taps;
pub mod transcript;
pub mod utils;
pub(crate) mod virt_batch;

pub use ligerito_flock::{
    FlockRsError, IntEvalRsLigModQProof, LigConfig, commit_rs_flock, commit_rs_flock_with,
    lig_configs, prove_mle_eval_mod_q_ligerito, verify_mle_eval_mod_q_ligerito,
};
// Round 0 of the paper's `c:core_iop` (the out-of-domain sample) and the
// standalone statement binding the `bitz` CLI / bench use around it.
pub use ligerito_flock::{
    OodRound, OodRoundParams, absorb_standalone_mod_q_claim, absorb_standalone_mod_q_statement,
    ood_round_bits, ood_round_params, prove_mle_eval_mod_q_ligerito_with_ood, sha_lig_ood_params,
    verify_mle_eval_mod_q_ligerito_runtime, verify_mle_eval_mod_q_ligerito_with_ood,
};
// Extension-field evaluation claims (paper `c:core_iop` Steps 1–3): the
// random-prime projection of a degree-≥2 extension-field claim onto the
// ordinary mod-q' pipeline.
pub use ext_proj::ExtProjParams;
pub use ligerito_flock::{
    IntEvalRsLigExtProof, prove_mle_eval_ext_ligerito, prove_mle_eval_ext_ligerito_with_ood,
    verify_mle_eval_ext_ligerito, verify_mle_eval_ext_ligerito_with_ood,
};
// EXPERIMENTAL — mod-q RLC claim families (docs/rlc-family-note-prompt.md):
// k claims on F₂-linear forms of j committed columns via ONE γ-RLC forest
// per chunk + a degree-(j+1) monomial discharge. No proof_codec wiring.
pub use ligerito_flock::{
    IntEvalRsLigRlcFamilyProof, RlcFamilyClaim, mle_eval_mod_q_lig_rlc_family_proof_size_bytes,
    prove_mle_eval_mod_q_ligerito_rlc_family, verify_mle_eval_mod_q_ligerito_rlc_family,
};
// EXPERIMENTAL — structured-tap virtual claims (ROT/SHIFT/entry-offset
// taps; docs/rlc-structured-taps-phase0.md): tapped rows through the
// batched x-forest with translated-eq committed openings.
pub use ligerito_flock::{
    IntEvalRsLigModQTapProof, TapClaim, TapVerifyClaim, mle_eval_mod_q_lig_tap_size_breakdown,
    prove_mle_eval_mod_q_ligerito_tap_claims, verify_mle_eval_mod_q_ligerito_tap_claims,
};
pub use ligerito_flock::{
    IntEvalRsLigTapFamilyProof, TapFamilyCluster, TapFamilyClusterSide,
    mle_eval_mod_q_lig_tap_family_size_breakdown, prove_mle_eval_mod_q_ligerito_tap_family,
    verify_mle_eval_mod_q_ligerito_tap_family,
};
// EXPERIMENTAL — the single-tap shared-point collapse: k single-tap
// claims at one point become ≤ #columns × 2 plain single-column claims
// (weight transform; no streams, channels, or translated-eq rings).
pub use circuit::linear_map::binary::{PreparedVirtualMap, PreparedVirtualMapError};
pub use ligerito_flock::{
    TapPointClaim, prove_mle_eval_mod_q_ligerito_tap_collapse,
    verify_mle_eval_mod_q_ligerito_tap_collapse,
};
pub use pcs::IntegerMatrixLayout;
pub use poly::univariate::binary_b127::B127;
pub use poly::univariate::binary_gf128::Gf128;

pub use taps::{TapOp, extract_virtual_tap_rows};

pub mod sumcheck;
