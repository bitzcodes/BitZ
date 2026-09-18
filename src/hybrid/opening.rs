//! Round 0 (the out-of-domain sample), one ring switch and one Ligerito
//! continuation, with two-root authentication.
//!
//! The shared opener runs in the Johnson (list-decoding) regime, at rate
//! 1/2 by default (rate 1/8 selectable through the prepared Ligerito
//! selection). The paper's theorem covers that regime only with Round 0: right
//! after the statement, before any other challenge, the prover sends
//! `y = MLE[V](ζ⃗)` of the virtual packed witness `V` at a transcript-drawn
//! `ζ⃗ = (ζ, ζ², ζ⁴, …)`, which pins it to one element of the level-0
//! list before the first forest or PIOP challenge. The claim is `K`-linear
//! in `V`, so it rides the final opening: one extra batching draw `η_ood`
//! adds `η_ood·eq(·, ζ⃗)` to the Ligerito basis and `η_ood·y` to its
//! target, and the verifier folds that term succinctly. Every primitive is
//! the audited one of [`crate::ligerito_flock`] (Round 0 of the paper's
//! core IOP); nothing here re-derives a bound.
use super::{CompositionProfile, Error, Gf};
use crate::{
    ligerito::{RingSwitchProof, residual_b_evals, ring_switch_prove_with, ring_switch_verify},
    ligerito_flock::{
        OodProverClaim, OodRound, OodRoundParams, OodVerifierClaim, ZincChallenger, add_ood_basis,
        ood_residual_evals, prove_ood_round_packed, verify_ood_round,
    },
    piop::spartan::profile::{IopSecurityProfile, MAX_DERIVED_GRINDING_BITS},
    transcript::{Blake3Transcript, traits::Transcript},
};
use flock_core::{
    field::Gf128 as F,
    merkle::{self, Hash},
    pcs::{
        commit::{PcsParams, ProverData},
        ligerito::{self, LigeritoProof, RecursiveProof},
    },
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Default Reed–Solomon inverse-rate exponent of the shared opener
/// (rate 1/2). The effective rate is the prepared Ligerito selection's
/// level-0 rate — [`Geometry::params`] takes it explicitly — and the commit
/// rate MUST equal that level-0 configuration rate: the opener queries the
/// committed codewords.
pub(super) const LOG_INV_RATE: usize = 1;

#[derive(Clone, Debug)]
pub(super) struct Geometry {
    pub logs: [usize; 2],
    pub physical_logs: [usize; 2],
    pub position_log: usize,
    pub lane_logs: [usize; 2],
    pub virtual_lane_log: usize,
}

impl Geometry {
    pub fn new(logs: [usize; 2]) -> Result<Self, Error> {
        if logs.iter().any(|&n| !(9..=27).contains(&n)) {
            return Err(Error::Invalid("packed witness logarithm outside 9..=27"));
        }
        let position_log = logs[0].max(logs[1]) - 3;
        let physical_logs = logs.map(|l| l.max(position_log));
        let lane_logs = physical_logs.map(|l| l - position_log);
        let virtual_lane_log = 4;
        Ok(Self {
            logs,
            physical_logs,
            position_log,
            lane_logs,
            virtual_lane_log,
        })
    }
    pub fn packed_log(&self) -> usize {
        self.position_log + self.virtual_lane_log
    }
    pub fn bit_log(&self) -> usize {
        self.packed_log() + 7
    }
    pub fn lanes(&self) -> usize {
        1 << self.virtual_lane_log
    }
    pub fn offset(&self, branch: usize) -> usize {
        branch << (self.virtual_lane_log - 1)
    }
    #[cfg(test)]
    pub fn embed(&self, branch: usize, original: usize) -> usize {
        let k = self.lane_logs[branch];
        ((original >> k) << self.virtual_lane_log)
            + self.offset(branch)
            + (original & ((1 << k) - 1))
    }
    pub fn project_point(&self, branch: usize, point: &[F]) -> (Vec<F>, F) {
        let k = self.lane_logs[branch];
        let mut original = point[..7 + k].to_vec();
        let high = self.logs[branch] - k;
        original.extend_from_slice(&point[11..11 + high]);
        let mut padding = F::ONE;
        for j in k..self.virtual_lane_log {
            padding *= if branch == 1 && j == self.virtual_lane_log - 1 {
                point[7 + j]
            } else {
                F::ONE + point[7 + j]
            };
        }
        for &r in &point[11 + high..] {
            padding *= F::ONE + r;
        }
        (original, padding)
    }
    /// Commitment parameters of one branch, committing at `log_inv_rate` —
    /// the prepared Ligerito selection's level-0 rate. `profile` is inert
    /// here: the hybrid path never consults flock's embedded profiles, its
    /// opener configuration is the prepared Ligerito resolver; `commit` reads
    /// only `m`, `log_inv_rate`, `log_batch_size` and `merkle_hash`.
    pub fn params(&self, branch: usize, log_inv_rate: usize) -> PcsParams {
        PcsParams {
            m: self.physical_logs[branch] + 7,
            log_inv_rate,
            log_batch_size: self.lane_logs[branch],
            profile: ligerito::LigeritoProfile::Secure,
            merkle_hash: merkle::HashKind::Blake3,
        }
    }
    /// H_r = eq(r,.) minus its restrictions to the two logical supports.
    /// In characteristic two subtraction is addition. This represents the
    /// padding mask with exactly three equality bases, without a dense mask.
    fn padding_bases(&self, r: Vec<Gf>, eta: Gf) -> Vec<(Vec<Gf>, Gf)> {
        let mut bases = vec![(r.clone(), eta)];
        for branch in 0..2 {
            let k = self.lane_logs[branch];
            let high = self.logs[branch] - k;
            let mut clamped = r.clone();
            let mut scale = eta;
            for coordinate in (k..4).chain(4 + high..self.packed_log()) {
                let bit = coordinate == 3 && branch == 1;
                scale *= if bit {
                    r[coordinate]
                } else {
                    Gf::one() + r[coordinate]
                };
                clamped[coordinate] = if bit { Gf::one() } else { Gf::zero() };
            }
            bases.push((clamped, scale));
        }
        bases
    }
    pub fn virtual_packed(&self, sources: [&[F]; 2]) -> Vec<F> {
        let lanes = self.lanes();
        let mut out = vec![F::ZERO; 1 << self.packed_log()];
        // One lane group per position: `embed` places word `(g << k) | l`
        // of branch `b` at lane `offset(b) + l` of group `g`.
        crate::utils::cfg_chunks_mut!(out, lanes)
            .enumerate()
            .for_each(|(g, group)| {
                for branch in 0..2 {
                    let k = self.lane_logs[branch];
                    let start = self.offset(branch);
                    let words = &sources[branch][g << k..(g + 1) << k];
                    group[start..start + words.len()].copy_from_slice(words);
                }
            });
        out
    }
}

#[derive(Clone, Debug)]
pub(super) struct Proof {
    /// Round 0: `y = MLE[V](ζ⃗)` and the proof-of-work nonce before the
    /// `ζ` draw.
    pub ood: Option<OodRound>,
    pub ring: RingSwitchProof,
    pub ligerito: LigeritoProof,
    pub paths: [Vec<Hash>; 2],
}

/// Round 0 on the prover side, on the virtual packed witness `packed`.
/// Must run right after the statement, before any other challenge.
pub(super) fn prove_ood(
    t: &mut Blake3Transcript,
    params: Option<OodRoundParams>,
    packed: &[F],
) -> Option<OodProverClaim> {
    params.map(|params| prove_ood_round_packed(t, packed, params))
}

/// Round 0 on the verifier side: the same frame and draw, the nonce
/// checked against `params`, the prover's `y` absorbed.
pub(super) fn verify_ood(
    t: &mut Blake3Transcript,
    geometry: &Geometry,
    params: Option<OodRoundParams>,
    round: Option<&OodRound>,
) -> Result<Option<OodVerifierClaim>, Error> {
    match (params, round) {
        (None, None) => Ok(None),
        (Some(params), Some(round)) => verify_ood_round(t, geometry.packed_log(), params, round)
            .map(Some)
            .map_err(|_| Error::Invalid("Round 0 (out-of-domain sample)")),
        _ => Err(Error::Invalid(
            "Round-0 presence disagrees with Ligerito regime",
        )),
    }
}

pub(super) fn ood_parameters(
    resolved: &crate::ligerito_flock::ResolvedLigerito,
) -> Result<Option<(f64, OodRoundParams)>, Error> {
    resolved
        .ood_bits()
        .map(|bits| {
            let grinding_bits = (CompositionProfile::LAMBDA as f64 - bits).ceil().max(0.) as u32;
            if grinding_bits > MAX_DERIVED_GRINDING_BITS {
                return Err(Error::Config(
                    "Round 0 exceeds the derived grinding cap".into(),
                ));
            }
            Ok((bits, OodRoundParams { grinding_bits }))
        })
        .transpose()
}

fn sample_padding(t: &mut Blake3Transcript, geometry: &Geometry) -> Vec<(Vec<Gf>, Gf)> {
    t.absorb_slice(b"hybrid/zero-padding/three-equality-bases/v1");
    let point = (0..geometry.packed_log())
        .map(|_| t.get_field_challenge(&()))
        .collect();
    let eta = t.get_field_challenge(&());
    geometry.padding_bases(point, eta)
}

pub(super) fn prove(
    t: &mut Blake3Transcript,
    geometry: &Geometry,
    statement: &Hash,
    packed: Vec<F>,
    ood: Option<&OodProverClaim>,
    resolved: &crate::ligerito_flock::ResolvedLigerito,
    data: [&ProverData; 2],
    point: &[Gf],
) -> Result<Proof, Error> {
    let ring_scope = tracing::info_span!("op:ring_switch").entered();
    // flock's packed words are bit-compatible with `Gf`: the ring switch
    // reads them in place and writes the basis in flock's element type (no
    // 2^m-element conversion pass either way).
    let (ring, mut basis, mut target) =
        ring_switch_prove_with(t, &packed, &point[7..], |value| value);
    drop(ring_scope);
    let basis_scope = tracing::info_span!("op:extra_bases").entered();
    // Batch the Round-0 claim into the same opening: one draw adds
    // `η_ood·eq(·, ζ⃗)` to the basis and `η_ood·y` to the target.
    if let Some(ood) = ood {
        let eta_ood: Gf = t.get_field_challenge(&());
        add_ood_basis(&mut basis, &packed, &ood.point, eta_ood, None);
        target += eta_ood * ood.y;
    }
    // Sample after all ring-switch messages. The authenticated target of this
    // independent claim is zero; a nonzero padded message cannot be discarded.
    for (point, scale) in sample_padding(t, geometry) {
        add_ood_basis(&mut basis, &packed, &point, scale, None);
    }
    drop(basis_scope);
    let _lig_scope = tracing::info_span!("op:ligerito").entered();
    let pc = resolved.prover();
    let mut paths = [Vec::new(), Vec::new()];
    let proof = ligerito::recursive_prover_with_basis_initial(
        &pc,
        packed,
        basis,
        (target),
        *statement,
        |positions, lanes, queries| {
            let mut rows = vec![vec![F::ZERO; lanes]; queries.len()];
            for branch in 0..2 {
                let width = 1 << geometry.lane_logs[branch];
                let start = geometry.offset(branch);
                for (row, &q) in rows.iter_mut().zip(queries) {
                    row[start..start + width]
                        .copy_from_slice(&data[branch].codeword[q * width..(q + 1) * width]);
                }
                paths[branch] =
                    merkle::merkle_multi_proof(&data[branch].merkle_tree, positions, queries);
            }
            RecursiveProof {
                opened_rows: rows,
                merkle_proof: Vec::new(),
            }
        },
        &mut ZincChallenger(t),
    );
    Ok(Proof {
        ood: ood.map(|claim| claim.round),
        ring,
        ligerito: proof,
        paths,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify(
    t: &mut Blake3Transcript,
    geometry: &Geometry,
    statement: &Hash,
    roots: &[Hash; 2],
    point: &[Gf],
    value: F,
    ood: Option<&OodVerifierClaim>,
    resolved: &crate::ligerito_flock::ResolvedLigerito,
    proof: &Proof,
) -> Result<(), Error> {
    let (eq_r2, mut target) = ring_switch_verify(t, &proof.ring, (value), &point[..7])
        .map_err(|_| Error::Invalid("ring switch"))?;
    let eta_ood: Option<Gf> = ood.map(|claim| {
        let eta: Gf = t.get_field_challenge(&());
        target += eta * claim.y;
        eta
    });
    let padding = sample_padding(t, geometry);
    let vc = resolved.verifier();
    // Deeper levels take explicit out-of-domain samples and grind their
    // fold challenges (tapered one bit per round, as in flock's prover);
    // level 0's binding is Round 0. flock's verifier consumes both
    // vectors exactly and rejects leftovers; the counts are fixed here as
    // well so a malformed proof fails on shape, not deep inside.
    let expected_ood_values: usize = vc.ood_samples.iter().skip(1).sum();
    let level_ks = std::iter::once(vc.initial_k).chain(vc.recursive_ks.iter().copied());
    let expected_fold_nonces: usize = vc
        .fold_grinding_bits
        .iter()
        .zip(level_ks)
        .map(|(&bits, k)| (0..k).filter(|&j| bits.saturating_sub(j) > 0).count())
        .sum();
    if proof.ligerito.recursive_roots.len() != vc.recursive_steps
        || proof.ligerito.recursive_proofs.len() + 1 != vc.recursive_steps
        || proof.ligerito.grinding_nonces.len() != vc.recursive_steps + 1
        || proof.ligerito.ood_values.len() != expected_ood_values
        || proof.ligerito.fold_grinding_nonces.len() != expected_fold_nonces
    {
        return Err(Error::Invalid("Ligerito proof shape"));
    }
    let valid = ligerito::recursive_verifier_with_basis_initial(
        &vc,
        &proof.ligerito,
        geometry.packed_log(),
        (target),
        statement,
        |prefix, log_y| {
            let prefix_gf: Vec<_> = prefix.iter().copied().collect();
            let mut out = residual_b_evals(&prefix_gf, log_y, &point[7..], &eq_r2);
            if let (Some(ood), Some(eta)) = (ood, eta_ood) {
                for (slot, term) in out
                    .iter_mut()
                    .zip(ood_residual_evals(prefix, log_y, &ood.point, eta))
                {
                    *slot += term;
                }
            }
            for (point, scale) in &padding {
                for (slot, term) in out
                    .iter_mut()
                    .zip(ood_residual_evals(prefix, log_y, point, *scale))
                {
                    *slot += term;
                }
            }
            out.into_iter().collect()
        },
        |positions, lanes, queries, opening| {
            if !opening.merkle_proof.is_empty() || lanes != geometry.lanes() {
                return false;
            }
            let mut hashes = [
                Vec::with_capacity(queries.len()),
                Vec::with_capacity(queries.len()),
            ];
            for row in &opening.opened_rows {
                // Authentication of both real slices AND every public zero lane.
                for branch in 0..2 {
                    let start = geometry.offset(branch);
                    let end = start + (1 << geometry.lane_logs[branch]);
                    let half_end = start + lanes / 2;
                    if row[end..half_end].iter().any(|&x| x != F::ZERO) {
                        return false;
                    }
                    let mut bytes = Vec::with_capacity((end - start) * 16);
                    for word in &row[start..end] {
                        bytes.extend_from_slice(&word.lo.to_le_bytes());
                        bytes.extend_from_slice(&word.hi.to_le_bytes());
                    }
                    hashes[branch].push(merkle::hash_leaf(&bytes, vc.merkle_hash));
                }
            }
            (0..2).all(|branch| {
                merkle::verify_merkle_multi_proof(
                    &roots[branch],
                    positions,
                    queries,
                    &hashes[branch],
                    &proof.paths[branch],
                    vc.merkle_hash,
                )
            })
        },
        &mut ZincChallenger(t),
    );
    if valid {
        Ok(())
    } else {
        Err(Error::Invalid("two-root Ligerito opening"))
    }
}
