//! The BitZ opener as a stand-alone binary-field polynomial commitment.
//!
//! Commit `2^{m_p}` packed `GF(2^128)` words as an interleaved Reed–Solomon
//! codeword at rate 1/2 under a BLAKE3 Merkle tree, pin the committed word
//! with Round 0 (the out-of-domain sample) right after the commitment, and
//! discharge `K`-linear relations `⟨b, f⟩ = c` on the packed words — the
//! ring-switched bit-MLE claim, or any transparent basis — with flock's basis
//! Ligerito in the Johnson (list-decoding) regime, fold and query grinding
//! included, `η_ood`-batched with the Round-0 claim. This is the shared opener
//! of [`crate::hybrid`] restricted to one root, packaged so that a foreign
//! binary-field PIOP (Binius64's, see [`crate::binius_ligerito`]) can use the
//! same commitment layer, proximity regime and grinding as BitZ itself.
//!
//! Every primitive is the audited one of [`crate::ligerito_flock`]; nothing
//! here re-derives a bound. The security accounting ([`BinaryPcs::security_terms`])
//! is the per-opener slice of `hybrid::security::account`.
use crate::{
    ligerito::{
        LOG_PACKING, RingSwitchProof, residual_b_evals, ring_switch_prove, ring_switch_verify,
    },
    ligerito_flock::{
        OodProverClaim, OodRound, OodRoundParams, OodVerifierClaim, ZincChallenger, add_ood_basis,
        custom_johnson_config_bits, ood_residual_evals, ood_round_bits, ood_round_params,
        prove_ood_round_packed, verify_ood_round,
    },
    piop::spartan::profile::MAX_DERIVED_GRINDING_BITS,
    poly::univariate::binary_gf128::Gf128 as Gf,
    proof_codec::{CodecError, Reader, Writer},
    transcript::{Blake3Transcript, traits::Transcript},
};
use bincode::Options;
use flock_core::{
    field::Gf128,
    merkle::{self, Hash},
    pcs::{
        commit::{Commitment, PcsParams, ProverData, commit},
        ligerito::{
            self, LigeritoProfile, LigeritoProof, LigeritoSecurityConfig, ProverConfig,
            VerifierConfig,
        },
    },
};

/// Default Reed–Solomon inverse-rate exponent of the commitment and of the
/// opener's level 0 (rate 1/2, BitZ's own default); [`BinaryPcs::with_rate`]
/// takes another. The commit rate MUST equal the level-0 configuration rate:
/// the opener queries the committed codeword.
pub const LOG_INV_RATE: usize = 1;
/// Interleaving of the commitment = the opener's level-0 fold arity
/// (`initial_k`): flock's default 32 lanes, 512-byte leaves.
pub const LOG_BATCH_SIZE: usize = 5;
/// Round 0 is topped up to this λ by proof of work, exactly as the standalone
/// relations and the hybrid account it (`step0:ood-draw`).
pub const OOD_LAMBDA: u32 = 108;
/// Smallest supported packed witness (`2^13` words): the Johnson ladder needs
/// at least one recursive level below the 32-lane level 0, and at rate 1/2 a
/// `2^12` ladder's second level (64 positions) is narrower than its Johnson
/// query count (the solver now rejects such ladders at configuration time).
pub const MIN_PACKED_LOG: usize = 13;
/// Largest supported packed witness (`2^27` words = 2 GiB packed).
pub const MAX_PACKED_LOG: usize = 27;
/// Bound on one embedded Ligerito blob when decoding.
pub const MAX_PROOF_BYTES: usize = 64 << 20;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("binary PCS configuration: {0}")]
    Config(String),
    #[error("invalid binary PCS proof: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Codec(#[from] CodecError),
}

/// One term of the whole-protocol union bound.
#[derive(Clone, Debug)]
pub struct SecurityTerm {
    pub name: &'static str,
    pub error_bound: f64,
}

/// The prover's Round-0 state: the drawn point `ζ⃗`, the value `y = MLE[f](ζ⃗)`
/// and the wire messages.
pub struct Round0Prover(OodProverClaim);

impl Round0Prover {
    /// The messages that go on the wire (`y` and the proof-of-work nonce).
    pub fn round(&self) -> OodRound {
        self.0.round
    }
    pub fn y(&self) -> Gf {
        self.0.y
    }
}

/// The verifier's Round-0 state: the replayed point and the claimed value.
pub struct Round0Verifier(OodVerifierClaim);

impl Round0Verifier {
    pub fn y(&self) -> Gf {
        self.0.y
    }
    /// The replayed point `ζ⃗`.
    pub fn point(&self) -> &[Gf] {
        &self.0.point
    }
}

/// A bit-MLE opening: the ring switch's 128 partial evaluations and the
/// Ligerito continuation (which carries the `η_ood`-batched Round-0 claim).
#[derive(Clone, Debug)]
pub struct BitMleOpening {
    pub ring: RingSwitchProof,
    pub lig: LigeritoProof,
}

/// A commitment shape: rate 1/2, 32 lanes, one Johnson-regime Ligerito ladder
/// solved by flock's machinery to `component_bits` per round, Round 0 at
/// [`OOD_LAMBDA`].
#[derive(Clone)]
pub struct BinaryPcs {
    packed_log: usize,
    log_inv_rate: usize,
    component_bits: usize,
    config: LigeritoSecurityConfig,
    pc: ProverConfig,
    vc: VerifierConfig,
    params: PcsParams,
    ood: OodRoundParams,
    ood_bits: f64,
}

impl BinaryPcs {
    /// Solve and validate the opener for `2^packed_log` packed words at the
    /// round-by-round target `component_bits`, at the default rate
    /// [`LOG_INV_RATE`].
    pub fn new(packed_log: usize, component_bits: usize) -> Result<Self, Error> {
        Self::with_log_inv_rate(packed_log, component_bits, LOG_INV_RATE)
    }

    /// Solve at an explicit rate, with the rate preceding the component target.
    pub fn with_rate(
        packed_log: usize,
        log_inv_rate: usize,
        component_bits: usize,
    ) -> Result<Self, Error> {
        Self::with_log_inv_rate(packed_log, component_bits, log_inv_rate)
    }

    /// Solve the same Johnson opener at an explicit initial commitment rate.
    pub fn with_log_inv_rate(
        packed_log: usize,
        component_bits: usize,
        log_inv_rate: usize,
    ) -> Result<Self, Error> {
        if !(1..=3).contains(&log_inv_rate) {
            return Err(Error::Config("log inverse rate must be 1, 2, or 3".into()));
        }
        if !(MIN_PACKED_LOG..=MAX_PACKED_LOG).contains(&packed_log) {
            return Err(Error::Config(format!(
                "packed log {packed_log} outside {MIN_PACKED_LOG}..={MAX_PACKED_LOG}"
            )));
        }
        let m = packed_log + LOG_PACKING;
        let mut config =
            custom_johnson_config_bits(m, log_inv_rate, LOG_BATCH_SIZE, Some(component_bits));
        config.hash = "blake3".into();
        config.validate().map_err(Error::Config)?;
        if config.levels.first().map(|level| level.log_inv_rate) != Some(log_inv_rate) {
            return Err(Error::Config(
                "level-0 rate must equal the commit rate".into(),
            ));
        }
        let (pc, vc) = config.to_prover_verifier_configs().map_err(Error::Config)?;
        if pc.initial_k != LOG_BATCH_SIZE || pc.log_inv_rates[0] != log_inv_rate {
            return Err(Error::Config(
                "opener level 0 does not match the commitment".into(),
            ));
        }
        let params = PcsParams {
            m,
            log_inv_rate,
            log_batch_size: pc.initial_k,
            profile: LigeritoProfile::Secure,
            merkle_hash: merkle::HashKind::Blake3,
        };
        let ood_bits = ood_round_bits(&config, packed_log).ok_or_else(|| {
            Error::Config("the opener must run in the Johnson regime with Round 0".into())
        })?;
        let ood = ood_round_params(&config, packed_log, OOD_LAMBDA)
            .ok_or_else(|| Error::Config("Round 0 parameters".into()))?;
        if ood.grinding_bits > MAX_DERIVED_GRINDING_BITS {
            return Err(Error::Config(format!(
                "step0:ood-draw needs {} grinding bits, above the {}-bit cap",
                ood.grinding_bits, MAX_DERIVED_GRINDING_BITS
            )));
        }
        Ok(Self {
            packed_log,
            log_inv_rate,
            component_bits,
            config,
            pc,
            vc,
            params,
            ood,
            ood_bits,
        })
    }

    pub fn packed_log(&self) -> usize {
        self.packed_log
    }
    /// Level-0 (= commitment) inverse-rate exponent.
    pub fn log_inv_rate(&self) -> usize {
        self.log_inv_rate
    }
    pub fn component_bits(&self) -> usize {
        self.component_bits
    }
    pub fn config(&self) -> &LigeritoSecurityConfig {
        &self.config
    }
    pub fn params(&self) -> &PcsParams {
        &self.params
    }
    /// flock's verifier configuration of the ladder.
    pub fn verifier_config(&self) -> &VerifierConfig {
        &self.vc
    }
    /// Level-0 codeword queries.
    pub fn level0_queries(&self) -> usize {
        self.config.levels[0].queries
    }
    /// Level-0 fold-challenge grinding bits (tapered one bit per round).
    pub fn level0_fold_grinding_bits(&self) -> usize {
        self.config.levels[0].fold_grinding_bits
    }
    /// Level-0 query grinding bits.
    pub fn level0_query_grinding_bits(&self) -> usize {
        self.config.levels[0].grinding_bits
    }
    pub fn ood_grinding_bits(&self) -> u32 {
        self.ood.grinding_bits
    }
    /// `-log₂` of the Round-0 collision bound before grinding.
    pub fn ood_round_bits(&self) -> f64 {
        self.ood_bits
    }

    /// The opener's terms of a whole-protocol union bound: Round 0, every
    /// level's proximity folds, queries and (deeper levels') out-of-domain
    /// samples, the field rounds, and the `η_ood` batching draw. Callers add
    /// [`Self::ring_switch_term`] when the opening is a bit-MLE one.
    pub fn security_terms(&self) -> Vec<SecurityTerm> {
        let k_inv = 2f64.powi(-128);
        let mut terms = vec![SecurityTerm {
            name: "step0:ood-draw",
            error_bound: 2f64.powf(-(self.ood_bits + f64::from(self.ood.grinding_bits))),
        }];
        for (index, level) in self.config.levels.iter().enumerate() {
            let (pg, query) = level.paper_predicted_bits();
            // One term per fold round of the level: round `j` carries the
            // row-union factor `2^{ℓ-1-j}` and a grind of `fold_bits - j`
            // (flock's taper), so every round's error is exactly
            // `2^-(eps_pg + fold_grinding_bits)`; their sum is the level's
            // `k_recursive`-fold union.
            for _ in 0..level.k_recursive {
                terms.push(SecurityTerm {
                    name: "Ligerito proximity folds",
                    error_bound: 2f64.powf(-pg - level.fold_grinding_bits as f64),
                });
            }
            terms.push(SecurityTerm {
                name: "Ligerito queries",
                error_bound: 2f64.powf(-query - level.grinding_bits as f64),
            });
            // Deeper levels bind their list element with explicit out-of-domain
            // samples inside the opener; level 0's binding is Round 0 above.
            if index > 0 {
                if let Some(bits) = level.paper_predicted_ood_bits() {
                    terms.push(SecurityTerm {
                        name: "Ligerito OOD samples",
                        error_bound: 2f64.powf(-bits),
                    });
                }
            }
        }
        terms.push(SecurityTerm {
            name: "Ligerito field rounds",
            error_bound: (16 * self.packed_log + 128) as f64 * k_inv,
        });
        terms.push(SecurityTerm {
            name: "opening batching draw",
            error_bound: k_inv,
        });
        terms
    }

    /// The ring switch's 128 partial evaluations.
    pub fn ring_switch_term() -> SecurityTerm {
        SecurityTerm {
            name: "ring switching",
            error_bound: 128.0 * 2f64.powi(-128),
        }
    }

    /// Sum of [`Self::security_terms`].
    pub fn error_sum(&self) -> f64 {
        self.security_terms().iter().map(|t| t.error_bound).sum()
    }

    /// Commit `2^packed_log` packed words.
    pub fn commit(&self, packed: &[Gf128]) -> Result<(Commitment, ProverData), Error> {
        if packed.len() != 1usize << self.packed_log {
            return Err(Error::Invalid("packed witness length"));
        }
        Ok(commit(packed, &self.params))
    }

    /// Round 0: must run right after the commitment's root is bound, before
    /// any other challenge.
    pub fn prove_round0(&self, t: &mut Blake3Transcript, packed: &[Gf128]) -> Round0Prover {
        debug_assert_eq!(packed.len(), 1usize << self.packed_log);
        Round0Prover(prove_ood_round_packed(t, packed, self.ood))
    }

    /// Round 0 on the verifier side: the same frame and draw, the nonce
    /// checked, the prover's `y` absorbed.
    pub fn verify_round0(
        &self,
        t: &mut Blake3Transcript,
        round: &OodRound,
    ) -> Result<Round0Verifier, Error> {
        verify_ood_round(t, self.packed_log, self.ood, round)
            .map(Round0Verifier)
            .map_err(|_| Error::Invalid("Round 0 (out-of-domain sample)"))
    }

    /// Open the `K`-linear relation `⟨basis, packed⟩ = target`, batched by one
    /// draw `η_ood` with the Round-0 claim `⟨eq(·, ζ⃗), packed⟩ = y`.
    pub fn open_basis(
        &self,
        t: &mut Blake3Transcript,
        packed: &[Gf128],
        data: &ProverData,
        round0: &Round0Prover,
        mut basis: Vec<Gf128>,
        target: Gf,
    ) -> LigeritoProof {
        assert_eq!(
            basis.len(),
            packed.len(),
            "basis must cover the packed witness"
        );
        let eta: Gf = t.get_field_challenge(&());
        add_ood_basis(&mut basis, packed, &round0.0.point, eta, None);
        let target = target + eta * round0.0.y;
        ligerito::recursive_prover_with_basis(
            &self.pc,
            packed,
            basis,
            target,
            &data.codeword,
            &data.merkle_tree,
            &mut ZincChallenger(t),
        )
    }

    /// Verify a basis opening. `eval_b(prefix, log_y)` returns the basis MLE
    /// at `prefix ++ bits(y)` for every `y < 2^log_y` (tail bit `j` at
    /// coordinate `prefix.len() + j`); the Round-0 term is added here.
    pub fn verify_basis(
        &self,
        t: &mut Blake3Transcript,
        root: &Hash,
        round0: &Round0Verifier,
        target: Gf,
        eval_b: impl Fn(&[Gf], usize) -> Vec<Gf>,
        proof: &LigeritoProof,
    ) -> Result<(), Error> {
        let eta: Gf = t.get_field_challenge(&());
        let target = target + eta * round0.0.y;
        self.check_shape(proof)?;
        let point = &round0.0.point;
        let eval = |prefix: &[Gf128], log_y: usize| -> Vec<Gf128> {
            let prefix_gf: Vec<Gf> = prefix.iter().copied().collect();
            let mut out = eval_b(&prefix_gf, log_y);
            let add = ood_residual_evals(prefix, log_y, point, eta);
            for (slot, term) in out.iter_mut().zip(add) {
                *slot = *slot + term;
            }
            out.into_iter().collect()
        };
        let ok = ligerito::recursive_verifier_with_basis_succinct(
            &self.vc,
            proof,
            self.packed_log,
            target,
            root,
            eval,
            &mut ZincChallenger(t),
        );
        if ok {
            Ok(())
        } else {
            Err(Error::Invalid("Ligerito opening"))
        }
    }

    /// Open the bit-MLE claim `MLE[bits(packed)](point) = value` (`point` has
    /// `packed_log + 7` coordinates, the low seven selecting the bit inside a
    /// word): ring switch, then [`Self::open_basis`].
    pub fn open_bit_mle(
        &self,
        t: &mut Blake3Transcript,
        packed: &[Gf128],
        data: &ProverData,
        round0: &Round0Prover,
        point: &[Gf],
    ) -> BitMleOpening {
        assert_eq!(
            point.len(),
            self.packed_log + LOG_PACKING,
            "bit-MLE point length"
        );
        // The ring switch reads the packed words in place (no `Gf` copy).
        let (ring, basis, target) = ring_switch_prove(t, packed, &point[LOG_PACKING..]);
        let basis: Vec<Gf128> = basis.into_iter().collect();
        let lig = self.open_basis(t, packed, data, round0, basis, target);
        BitMleOpening { ring, lig }
    }

    /// Verify a bit-MLE opening against `root`.
    pub fn verify_bit_mle(
        &self,
        t: &mut Blake3Transcript,
        root: &Hash,
        round0: &Round0Verifier,
        point: &[Gf],
        value: Gf,
        proof: &BitMleOpening,
    ) -> Result<(), Error> {
        if point.len() != self.packed_log + LOG_PACKING {
            return Err(Error::Invalid("bit-MLE point length"));
        }
        let (r_lo, r_hi) = point.split_at(LOG_PACKING);
        let (eq_r2, target) = ring_switch_verify(t, &proof.ring, value, r_lo)
            .map_err(|_| Error::Invalid("ring switch"))?;
        self.verify_basis(
            t,
            root,
            round0,
            target,
            |prefix, log_y| residual_b_evals(prefix, log_y, r_hi, &eq_r2),
            &proof.lig,
        )
    }

    /// Fixed counts of a well-formed proof for this configuration, so a
    /// malformed proof fails on shape rather than deep inside flock.
    fn check_shape(&self, proof: &LigeritoProof) -> Result<(), Error> {
        let vc = &self.vc;
        let expected_ood_values: usize = vc.ood_samples.iter().skip(1).sum();
        let level_ks = std::iter::once(vc.initial_k).chain(vc.recursive_ks.iter().copied());
        let expected_fold_nonces: usize = vc
            .fold_grinding_bits
            .iter()
            .zip(level_ks)
            .map(|(&bits, k)| (0..k).filter(|&j| bits.saturating_sub(j) > 0).count())
            .sum();
        if proof.recursive_roots.len() != vc.recursive_steps
            || proof.recursive_proofs.len() + 1 != vc.recursive_steps
            || proof.grinding_nonces.len() != vc.recursive_steps + 1
            || proof.ood_values.len() != expected_ood_values
            || proof.fold_grinding_nonces.len() != expected_fold_nonces
        {
            return Err(Error::Invalid("Ligerito proof shape"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Wire format helpers (canonical; the Ligerito blob is flock's bincode 1.3)
// ---------------------------------------------------------------------

pub fn write_round0(w: &mut Writer, round: &OodRound) {
    w.gf(&round.y);
    match round.nonce {
        None => w.len(0),
        Some(nonce) => {
            w.len(1);
            w.len(nonce as usize);
        }
    }
}

pub fn read_round0(r: &mut Reader<'_>) -> Result<OodRound, CodecError> {
    let y = r.gf()?;
    let nonce = match r.len()? {
        0 => None,
        1 => Some(r.len()? as u64),
        _ => return Err(CodecError::NonCanonical),
    };
    Ok(OodRound { y, nonce })
}

pub fn write_ring_switch(w: &mut Writer, ring: &RingSwitchProof) {
    assert_eq!(
        ring.s_v.len(),
        128,
        "ring switch sends 128 partial evaluations"
    );
    for g in &ring.s_v {
        w.gf(g);
    }
}

pub fn read_ring_switch(r: &mut Reader<'_>) -> Result<RingSwitchProof, CodecError> {
    let mut s_v = Vec::with_capacity(128);
    for _ in 0..128 {
        s_v.push(r.gf()?);
    }
    Ok(RingSwitchProof { s_v })
}

pub fn write_ligerito(w: &mut Writer, proof: &LigeritoProof) {
    let bytes = bincode::serialize(proof).expect("Ligerito proof serialization");
    w.len(bytes.len());
    w.bytes(&bytes);
}

pub fn read_ligerito(r: &mut Reader<'_>) -> Result<LigeritoProof, CodecError> {
    let n = r.len()?;
    if n > MAX_PROOF_BYTES || n > r.remaining() {
        return Err(CodecError::Truncated);
    }
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(n as u64)
        .reject_trailing_bytes()
        .deserialize(r.take(n)?)
        .map_err(|e| CodecError::Bincode(e.to_string()))
}

impl BitMleOpening {
    pub fn write(&self, w: &mut Writer) {
        write_ring_switch(w, &self.ring);
        write_ligerito(w, &self.lig);
    }
    pub fn read(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(Self {
            ring: read_ring_switch(r)?,
            lig: read_ligerito(r)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq_table(point: &[Gf]) -> Vec<Gf> {
        let mut out = vec![Gf::one()];
        for &r in point {
            let n = out.len();
            out.resize(2 * n, Gf::zero());
            for i in 0..n {
                let high = out[i] * r;
                out[n + i] = high;
                out[i] = out[i] + high;
            }
        }
        out
    }

    /// Reference bit-MLE: bit `j` of word `i` sits at index `128·i + j`.
    fn bit_mle(packed: &[Gf128], point: &[Gf]) -> Gf {
        let low = eq_table(&point[..LOG_PACKING]);
        let high = eq_table(&point[LOG_PACKING..]);
        let mut acc = Gf::zero();
        for (i, word) in packed.iter().enumerate() {
            let bits = word.lo as u128 | ((word.hi as u128) << 64);
            for (j, &l) in low.iter().enumerate() {
                if bits >> j & 1 == 1 {
                    acc = acc + l * high[i];
                }
            }
        }
        acc
    }

    fn random_packed(seed: u64, packed_log: usize) -> Vec<Gf128> {
        let mut s = seed;
        (0..1usize << packed_log)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                Gf128 {
                    lo: s,
                    hi: s.rotate_left(29),
                }
            })
            .collect()
    }

    #[test]
    fn bit_mle_opening_round_trips_and_rejects_a_wrong_value() {
        let pcs = BinaryPcs::new(13, 100).unwrap();
        // The opener's terms sum to roughly 22·2^-target over the ladder's
        // levels; a composition gated at 100 bits therefore targets 105–106.
        assert!(pcs.error_sum().log2() < -94.0);
        assert!(BinaryPcs::new(13, 106).unwrap().error_sum().log2() < -100.0);
        let packed = random_packed(0x1234_5678_9abc_def1, 13);
        let (c, data) = pcs.commit(&packed).unwrap();

        let mut t = Blake3Transcript::new();
        t.absorb_slice(&c.root);
        let r0 = pcs.prove_round0(&mut t, &packed);
        let point: Vec<Gf> = (0..13 + LOG_PACKING)
            .map(|_| t.get_field_challenge(&()))
            .collect();
        let value = bit_mle(&packed, &point);
        crate::ligerito::absorb_ood_value(&mut t, value);
        let opening = pcs.open_bit_mle(&mut t, &packed, &data, &r0, &point);

        let mut w = Writer::new();
        write_round0(&mut w, &r0.round());
        opening.write(&mut w);
        let bytes = w.into_vec();
        let mut r = Reader::new(&bytes);
        let round = read_round0(&mut r).unwrap();
        let decoded = BitMleOpening::read(&mut r).unwrap();
        assert_eq!(r.remaining(), 0);

        let verify = |value: Gf, round: &OodRound, proof: &BitMleOpening| {
            let mut t = Blake3Transcript::new();
            t.absorb_slice(&c.root);
            let r0 = pcs.verify_round0(&mut t, round)?;
            let point: Vec<Gf> = (0..13 + LOG_PACKING)
                .map(|_| t.get_field_challenge(&()))
                .collect();
            crate::ligerito::absorb_ood_value(&mut t, value);
            pcs.verify_bit_mle(&mut t, &c.root, &r0, &point, value, proof)
        };
        verify(value, &round, &decoded).unwrap();
        assert!(verify(value + Gf::one(), &round, &decoded).is_err());
        let wrong_y = OodRound {
            y: round.y + Gf::one(),
            ..round
        };
        assert!(verify(value, &wrong_y, &decoded).is_err());
    }

    #[test]
    fn basis_opening_round_trips() {
        for packed_log in [13, 16] {
            basis_opening_round_trips_at(packed_log);
        }
    }

    /// An arbitrary (non-tensor) basis, evaluated densely on the verifier
    /// side: the basis prover/verifier pair must not assume eq structure.
    #[test]
    fn arbitrary_basis_opening_round_trips() {
        for packed_log in [13, 16] {
            arbitrary_basis_opening_round_trips_at(packed_log);
        }
    }

    fn arbitrary_basis_opening_round_trips_at(packed_log: usize) {
        let pcs = BinaryPcs::new(packed_log, 100).unwrap();
        let packed = random_packed(0x7777_0001_2345_6789, packed_log);
        let basis = random_packed(0x1111_2222_3333_4444, packed_log);
        let (c, data) = pcs.commit(&packed).unwrap();
        let mut t = Blake3Transcript::new();
        t.absorb_slice(&c.root);
        let r0 = pcs.prove_round0(&mut t, &packed);
        let target = packed
            .iter()
            .zip(&basis)
            .fold(Gf::zero(), |acc, (&f, &b)| acc + (f) * (b));
        crate::ligerito::absorb_ood_value(&mut t, target);
        let proof = pcs.open_basis(&mut t, &packed, &data, &r0, basis.clone(), target);

        let mut t = Blake3Transcript::new();
        t.absorb_slice(&c.root);
        let r0v = pcs.verify_round0(&mut t, &r0.round()).unwrap();
        crate::ligerito::absorb_ood_value(&mut t, target);
        let basis_gf: Vec<Gf> = basis.iter().copied().collect();
        let eval_b = |prefix: &[Gf], log_y: usize| -> Vec<Gf> {
            (0..1usize << log_y)
                .map(|y| {
                    let mut point = prefix.to_vec();
                    for j in 0..log_y {
                        point.push(if y >> j & 1 == 1 {
                            Gf::one()
                        } else {
                            Gf::zero()
                        });
                    }
                    basis_gf
                        .iter()
                        .zip(eq_table(&point))
                        .fold(Gf::zero(), |acc, (&b, e)| acc + b * e)
                })
                .collect()
        };
        pcs.verify_basis(&mut t, &c.root, &r0v, target, eval_b, &proof)
            .unwrap();
    }

    fn basis_opening_round_trips_at(packed_log: usize) {
        let pcs = BinaryPcs::new(packed_log, 100).unwrap();
        let packed = random_packed(0x0f0f_1234_5678_0001 ^ packed_log as u64, packed_log);
        let (c, data) = pcs.commit(&packed).unwrap();
        // A transparent basis: eq(·, z) for a transcript point z.
        let mut t = Blake3Transcript::new();
        t.absorb_slice(&c.root);
        let r0 = pcs.prove_round0(&mut t, &packed);
        let z: Vec<Gf> = (0..packed_log)
            .map(|_| t.get_field_challenge(&()))
            .collect();
        let basis_gf = eq_table(&z);
        let target = packed
            .iter()
            .zip(&basis_gf)
            .fold(Gf::zero(), |acc, (&f, &b)| acc + (f) * b);
        crate::ligerito::absorb_ood_value(&mut t, target);
        let basis: Vec<Gf128> = basis_gf.iter().copied().collect();
        let proof = pcs.open_basis(&mut t, &packed, &data, &r0, basis, target);

        let mut t = Blake3Transcript::new();
        t.absorb_slice(&c.root);
        let r0v = pcs.verify_round0(&mut t, &r0.round()).unwrap();
        let zv: Vec<Gf> = (0..packed_log)
            .map(|_| t.get_field_challenge(&()))
            .collect();
        assert_eq!(zv, z);
        crate::ligerito::absorb_ood_value(&mut t, target);
        let eval_b = |prefix: &[Gf], log_y: usize| -> Vec<Gf> {
            (0..1usize << log_y)
                .map(|y| {
                    let mut point = prefix.to_vec();
                    for j in 0..log_y {
                        point.push(if y >> j & 1 == 1 {
                            Gf::one()
                        } else {
                            Gf::zero()
                        });
                    }
                    point
                        .iter()
                        .zip(&zv)
                        .fold(Gf::one(), |p, (&a, &b)| p * (Gf::one() + a + b))
                })
                .collect()
        };
        pcs.verify_basis(&mut t, &c.root, &r0v, target, eval_b, &proof)
            .unwrap();
    }
}
