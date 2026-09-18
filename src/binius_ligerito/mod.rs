//! Binius64's PIOP discharged through the BitZ opener: the `binius64-ligerito`
//! scheme of the comparison benchmarks.
//!
//! Binius64 proves its constraint system exactly as upstream does — the
//! IntMul, BinMul (GHASH-field multiplication), BitAnd, zero and shift
//! reductions on a BLAKE3 Fiat–Shamir transcript — up to the witness
//! evaluation claim its own ring switch would consume
//! (`IOPProver::prove_to_evaluation`, exposed by the vendored fork).
//! Everything below that line is BitZ's: every oracle the PIOP commits is an
//! interleaved Reed–Solomon codeword under BLAKE3 (rate 1/2 by default,
//! [`Prepared::with_rate`] takes another), pinned by Round 0 right after its
//! root is bound; the witness evaluation claim is discharged
//! by BitZ's ring switch and a Johnson-regime Ligerito opening with fold and
//! query grinding; every other oracle relation the PIOP queued (the IntMul
//! reduction's logup* pushforward, when the circuit multiplies) is discharged
//! by its own Ligerito opening. The whole protocol is gated at 100 bits under
//! one of two [`Accounting`] models — a union bound over every error term, or
//! the round-by-round minimum (every term on its own, the figure BitZ's own
//! rows report) — rather than by Binius64's query-phase target.
pub(crate) mod channel;

use crate::{
    binary_pcs::{
        self, BinaryPcs, BitMleOpening, SecurityTerm, read_ligerito, read_round0, write_ligerito,
        write_round0,
    },
    ligerito_flock::OodRound,
    poly::univariate::binary_gf128::Gf128 as Gf,
    proof_codec::{CodecError, Reader, Writer},
    transcript::{Blake3Transcript, traits::Transcript},
};
use binius_core::{
    constraint_system::{ConstraintSystem, InoutSegment, ValueVec},
    word::Word,
};
use binius_field::Field;
use binius_iop::channel::OracleSpec;
use binius_prover::{IOPProver, OptimalPackedB128, protocols::shift::KeyCollection};
use binius_verifier::{IOPVerifier, config::B128};
use channel::{ProverChannel, ProverRelation, VerifierChannel, VerifierRelation};
use flock_core::{field::Gf128, merkle::Hash, pcs::ligerito::LigeritoProof};

/// Oracle handles are commitment indices.
pub type Oracle = usize;
/// The whole-protocol gate.
pub const TARGET_BITS: u32 = 100;
/// The opener's round-by-round target is the smallest in this range whose
/// union bound clears [`TARGET_BITS`]; every extra bit doubles the level-0
/// fold grind.
pub const MIN_COMPONENT_BITS: usize = 100;
pub const MAX_COMPONENT_BITS: usize = 112;
const PROTOCOL: &[u8] = b"bitz/binius64-ligerito/non-zk/v1";
const EVALUATION_DOMAIN: &[u8] = b"bitz/binius64-ligerito/evaluation/v1";
const RELATIONS_DOMAIN: &[u8] = b"bitz/binius64-ligerito/relations/v1";
const FORK_DOMAIN: &[u8] = b"bitz/binius64-ligerito/opening-fork/v1";
/// Fork tag of the witness bit-MLE opening; relation group `i` uses `1 + i`.
const WITNESS_FORK: u64 = 0;
const MAGIC: &[u8; 8] = b"BLIG\x01\0\0\0";
/// Bound on the PIOP message count when decoding.
const MAX_MESSAGES: usize = 1 << 24;

pub(crate) fn b128_to_f128(x: B128) -> Gf128 {
    let v = u128::from(x);
    Gf128 {
        lo: v as u64,
        hi: (v >> 64) as u64,
    }
}

pub(crate) fn f128_to_b128(f: Gf128) -> B128 {
    B128::new(u128::from(f.lo) | (u128::from(f.hi) << 64))
}

fn b128_to_gf(x: B128) -> Gf {
    (b128_to_f128(x))
}

fn gf_to_b128(g: Gf) -> B128 {
    f128_to_b128((g))
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid binius64-ligerito proof or parameters: {0}")]
    Invalid(&'static str),
    #[error("binius64-ligerito configuration: {0}")]
    Config(String),
    #[error("Binius: {0}")]
    Binius(String),
    #[error(transparent)]
    Pcs(#[from] binary_pcs::Error),
    #[error(transparent)]
    Codec(#[from] CodecError),
}

/// How the whole-protocol figure is formed from the error terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accounting {
    /// `-log2` of the SUM of every term: a union bound, which needs every
    /// term about `log2(#terms)` bits above the gate.
    UnionBound,
    /// `-log2` of the LARGEST term: the round-by-round minimum, the figure
    /// BitZ's own rows report (`SoundnessAccounting::achieved_bits`); every
    /// term must clear the gate on its own.
    RoundByRound,
}

impl Accounting {
    pub fn name(self) -> &'static str {
        match self {
            Self::UnionBound => "union-bound",
            Self::RoundByRound => "round-by-round",
        }
    }
    fn tag(self) -> usize {
        match self {
            Self::UnionBound => 0,
            Self::RoundByRound => 1,
        }
    }
}

/// Algebraic/IOP error accounting of the complete composition, using the
/// pinned implementations' soundness analyses and grinding model. BLAKE3
/// Fiat–Shamir and 256-bit Merkle hashing remain cryptographic assumptions;
/// this is not an unconditional Fiat–Shamir theorem.
#[derive(Clone, Debug)]
pub struct SecurityReport {
    pub target_bits: u32,
    /// The model the gate was applied under.
    pub accounting: Accounting,
    /// The gated figure under `accounting`.
    pub algebraic_bits: f64,
    /// `-log2 Σ terms`.
    pub union_bound_bits: f64,
    /// `-log2 max term`.
    pub round_by_round_bits: f64,
    pub terms: Vec<SecurityTerm>,
}

impl SecurityReport {
    /// The term with the largest error bound.
    pub fn binding_term(&self) -> Option<&SecurityTerm> {
        self.terms
            .iter()
            .max_by(|a, b| a.error_bound.total_cmp(&b.error_bound))
    }
}

/// The proof: the PIOP prefix's messages, every oracle's root and Round-0
/// messages in commitment order, the witness bit-MLE opening, and one basis
/// opening per oracle that had relations queued.
#[derive(Clone, Debug)]
pub struct Proof {
    pub messages: Vec<u128>,
    pub roots: Vec<Hash>,
    pub rounds0: Vec<OodRound>,
    pub witness: BitMleOpening,
    pub relations: Vec<LigeritoProof>,
}

/// Prepared circuit, PIOP keys and openers; reusable across witnesses.
pub struct Prepared {
    verifier: IOPVerifier,
    prover: IOPProver,
    specs: Vec<OracleSpec>,
    pcs: Vec<BinaryPcs>,
    log_inv_rate: usize,
    accounting: Accounting,
    component_bits: usize,
    /// Working buffers for the PIOP prefix come from a pool this `Prepared`
    /// keeps for its lifetime, recycling blocks across proofs — exactly what
    /// Binius64's own `Prover` does (`prove.rs`: `let alloc = &self.pool`).
    /// With `GlobalAllocator` instead, the shared PIOP measured ~14 % slower
    /// at 2^20 (u64: 552 vs 483 ms) purely from malloc/mmap churn on the
    /// large per-round buffers, which biased the comparison against this
    /// scheme. Proofs are unaffected; the pool costs retained memory.
    pool: binius_compute::BufferPool,
    security: SecurityReport,
    statement_digest: [u8; 32],
}

impl Prepared {
    /// Prepare at the opener's default rate ([`binary_pcs::LOG_INV_RATE`]).
    pub fn new(cs: &ConstraintSystem) -> Result<Self, Error> {
        Self::with_rate(cs, binary_pcs::LOG_INV_RATE)
    }

    /// Prepare at an explicit rate using union-bound accounting.
    pub fn with_log_inv_rate(cs: &ConstraintSystem, log_inv_rate: usize) -> Result<Self, Error> {
        Self::with_rate(cs, log_inv_rate)
    }

    /// Prepare with every oracle committed and opened at the level-0 inverse
    /// rate exponent `log_inv_rate` (1 = rate 1/2, 3 = rate 1/8), gated by
    /// the union bound.
    pub fn with_rate(cs: &ConstraintSystem, log_inv_rate: usize) -> Result<Self, Error> {
        Self::with_options(cs, log_inv_rate, Accounting::UnionBound)
    }

    /// Prepare at `log_inv_rate` with the 100-bit gate applied under
    /// `accounting`; both are part of the statement digest.
    pub fn with_options(
        cs: &ConstraintSystem,
        log_inv_rate: usize,
        accounting: Accounting,
    ) -> Result<Self, Error> {
        if !(1..=3).contains(&log_inv_rate) {
            return Err(Error::Config("log inverse rate must be 1, 2, or 3".into()));
        }
        cs.validate().map_err(|e| Error::Binius(e.to_string()))?;
        let verifier = IOPVerifier::new(cs.clone(), cs.log_public_words(InoutSegment::Public));
        let specs = verifier.oracle_specs(false);
        if specs.is_empty() || specs.iter().any(|spec| spec.is_zk) {
            return Err(Error::Invalid("oracle specification"));
        }
        let k_inv = 2f64.powi(-128);
        // AND's univariate degree is <=126 (64-bit word domain), with three
        // skipped Bit coordinates; the subsequent zerocheck/shift rounds
        // have degree <=3. 4096 per coordinate bounds the initial identity
        // tests, skipped rounds, all operand batches and their sumchecks —
        // the same overcount `hybrid::security` applies to the SHA circuit.
        let piop_dims = verifier.log_witness_words()
            + cs.log_and_constraints().unwrap_or(0)
            + cs.log_zero_constraints().unwrap_or(0)
            + 64;
        let mut piop_terms = vec![SecurityTerm {
            name: "Binius64 PIOP (AND, zero, shift)",
            error_bound: (4096 * piop_dims) as f64 * k_inv,
        }];
        if let Some(log_imul) = cs.log_imul_constraints() {
            // The IntMul reduction: a 64-layer GKR step on the exponentiated
            // product, the batched Frobenius/product sumchecks, the depth-2
            // limb product check and the logup* lookup over the 2^16-row
            // generator table with its final batched sumcheck. 4096 per
            // coordinate over the row, word-bit, limb-table and batching
            // dimensions overcounts every round and draw.
            piop_terms.push(SecurityTerm {
                name: "Binius64 IntMul reduction",
                error_bound: (4096 * (log_imul + 64 + 16 + 8)) as f64 * k_inv,
            });
        }
        if let Some(log_bmul) = cs.log_bmul_constraints() {
            // The BinMul reduction: a degree-2 GHASH-field zerocheck/MLE-check
            // over the row dimension (the zerocheck draw and one round per
            // row variable), the 64-bit word-domain collapse of the six
            // per-bit evaluation vectors into the shift claim, and the
            // batching draws. It commits no extra oracle. 4096 per coordinate
            // overcounts every round and draw, as above.
            piop_terms.push(SecurityTerm {
                name: "Binius64 BinMul reduction",
                error_bound: (4096 * (log_bmul + 64 + 8)) as f64 * k_inv,
            });
        }
        let mut chosen = None;
        let mut last_error = None;
        for target in MIN_COMPONENT_BITS..=MAX_COMPONENT_BITS {
            let pcs: Result<Vec<BinaryPcs>, _> = specs
                .iter()
                .map(|spec| BinaryPcs::with_log_inv_rate(spec.log_msg_len, target, log_inv_rate))
                .collect();
            let pcs = match pcs {
                Ok(pcs) => pcs,
                Err(e) => {
                    last_error = Some(e.to_string());
                    continue;
                }
            };
            let mut terms = piop_terms.clone();
            terms.push(BinaryPcs::ring_switch_term());
            for opener in &pcs {
                terms.extend(opener.security_terms());
            }
            let total: f64 = terms.iter().map(|t| t.error_bound).sum();
            let union_bound_bits = -total.log2();
            let worst = terms.iter().map(|t| t.error_bound).fold(0.0, f64::max);
            let round_by_round_bits = -worst.log2();
            let algebraic_bits = match accounting {
                Accounting::UnionBound => union_bound_bits,
                Accounting::RoundByRound => round_by_round_bits,
            };
            if algebraic_bits >= f64::from(TARGET_BITS) {
                chosen = Some((
                    target,
                    pcs,
                    SecurityReport {
                        target_bits: TARGET_BITS,
                        accounting,
                        algebraic_bits,
                        union_bound_bits,
                        round_by_round_bits,
                        terms,
                    },
                ));
                break;
            }
        }
        let (component_bits, pcs, security) = chosen.ok_or_else(|| {
            Error::Config(format!(
                "no opener target in {MIN_COMPONENT_BITS}..={MAX_COMPONENT_BITS} reaches {TARGET_BITS} bits{}",
                last_error.map(|e| format!(" ({e})")).unwrap_or_default()
            ))
        })?;
        let prover = IOPProver::new(
            verifier.clone(),
            KeyCollection::build(cs, InoutSegment::Public),
        );
        let mut h = blake3::Hasher::new();
        h.update(PROTOCOL);
        for n in [
            verifier.log_witness_elems(),
            verifier.log_public_words(),
            cs.n_and_constraints(),
            cs.n_zero_constraints(),
            cs.n_imul_constraints(),
            cs.n_bmul_constraints(),
            log_inv_rate,
            accounting.tag(),
            component_bits,
            u32::try_from(TARGET_BITS).map_or(0, |b| b as usize),
        ] {
            h.update(&(n as u64).to_le_bytes());
        }
        // The constants are shared data the reduction reads as the head of
        // the public segment; the statement carries only the in/out words, so
        // their values are bound here.
        h.update(&(cs.constants.len() as u64).to_le_bytes());
        for word in &cs.constants {
            h.update(&word.0.to_le_bytes());
        }
        for (spec, opener) in specs.iter().zip(&pcs) {
            h.update(&(spec.log_msg_len as u64).to_le_bytes());
            h.update(
                &bincode::serialize(opener.config()).map_err(|e| Error::Config(e.to_string()))?,
            );
        }
        Ok(Self {
            verifier,
            prover,
            specs,
            pcs,
            log_inv_rate,
            accounting,
            component_bits,
            pool: binius_compute::BufferPool::new(),
            security,
            statement_digest: *h.finalize().as_bytes(),
        })
    }

    pub fn security(&self) -> &SecurityReport {
        &self.security
    }
    pub fn component_bits(&self) -> usize {
        self.component_bits
    }
    /// Level-0 (= commitment) inverse-rate exponent of every opener.
    pub fn log_inv_rate(&self) -> usize {
        self.log_inv_rate
    }
    /// The model the 100-bit gate was applied under.
    pub fn accounting(&self) -> Accounting {
        self.accounting
    }
    pub fn oracle_specs(&self) -> &[OracleSpec] {
        &self.specs
    }
    /// The opener of oracle `index` (0 = the witness).
    pub fn opener(&self, index: usize) -> &BinaryPcs {
        &self.pcs[index]
    }
    pub fn verifier(&self) -> &IOPVerifier {
        &self.verifier
    }

    fn transcript(&self, public: &[Word]) -> Blake3Transcript {
        let mut t = Blake3Transcript::new();
        t.absorb_slice(&self.statement_digest);
        t.absorb_slice(&(public.len() as u64).to_le_bytes());
        for word in public {
            t.absorb_slice(&word.0.to_le_bytes());
        }
        t
    }

    fn absorb_evaluation(t: &mut Blake3Transcript, value: B128) {
        t.absorb_slice(EVALUATION_DOMAIN);
        t.absorb_slice(&u128::from(value).to_le_bytes());
    }

    /// Every opening runs on its own fork of the transcript, taken after the
    /// PIOP prefix, the evaluation claim and every relation claim are bound
    /// and domain-separated by `tag`: no opening's challenges depend on the
    /// transcript state another opening leaves behind. This is not only
    /// hygiene: flock's Ligerito prover and verifier end their final level in
    /// different transcript states (nothing in this repository continues a
    /// transcript after a Ligerito opening, so it never mattered), which the
    /// `pushforward_relation_is_consistent_across_prover_and_verifier` test
    /// observed as the second opening rejecting when run sequentially.
    fn fork(t: &Blake3Transcript, tag: u64) -> Blake3Transcript {
        let mut fork = t.clone();
        fork.absorb_slice(FORK_DOMAIN);
        fork.absorb_slice(&tag.to_le_bytes());
        fork
    }

    /// Bind a relation group's claims and draw its batching weights: one
    /// relation is opened as is, several are combined by powers of one draw.
    fn relation_weights(t: &mut Blake3Transcript, oracle: Oracle, claims: &[B128]) -> Vec<Gf> {
        t.absorb_slice(RELATIONS_DOMAIN);
        t.absorb_slice(&(oracle as u64).to_le_bytes());
        t.absorb_slice(&(claims.len() as u64).to_le_bytes());
        for claim in claims {
            t.absorb_slice(&u128::from(*claim).to_le_bytes());
        }
        if claims.len() == 1 {
            return vec![Gf::one()];
        }
        let gamma: Gf = t.get_field_challenge(&());
        let mut weights = Vec::with_capacity(claims.len());
        let mut power = Gf::one();
        for _ in claims {
            weights.push(power);
            power = power * gamma;
        }
        weights
    }

    /// Prove `witness` satisfies the constraint system. The public words are
    /// bound first. Phase boundaries are exposed as `tracing` spans.
    pub fn prove(&self, witness: &ValueVec) -> Result<Proof, Error> {
        let public = witness.inout();
        let mut t = self.transcript(public);
        let mut channel = ProverChannel {
            transcript: &mut t,
            messages: Vec::new(),
            specs: self.specs.clone(),
            pcs: &self.pcs,
            oracles: Vec::new(),
            relations: Vec::new(),
        };
        let alloc = &self.pool;
        let (oracle, _packed, point, value) = tracing::info_span!(
            "PIOP prefix",
            component = "binius-ligerito.piop",
            scope_kind = "scope",
            tag_proving = true,
        )
        .in_scope(|| {
            self.prover
                .prove_to_evaluation::<_, OptimalPackedB128, _>(witness, &mut channel, &alloc)
                .map_err(|e| Error::Binius(e.to_string()))
        })?;
        if !channel.specs.is_empty() {
            return Err(Error::Invalid("unconsumed oracle specification"));
        }
        let ProverChannel {
            messages,
            oracles,
            relations,
            ..
        } = channel;

        let opening = tracing::info_span!(
            "Opening proof",
            component = "binius-ligerito.opening",
            scope_kind = "phase",
            tag_proving = true,
            tag_pcs = true,
            tag_opening_proof = true,
        )
        .entered();
        Self::absorb_evaluation(&mut t, value);
        // Bind every relation group's claims (and draw their batching
        // weights) before any opening; then each opening runs on its own fork.
        let groups: Vec<(Oracle, Vec<&ProverRelation>)> = (0..oracles.len())
            .map(|index| {
                (
                    index,
                    relations.iter().filter(|r| r.oracle == index).collect(),
                )
            })
            .filter(|(_, group): &(Oracle, Vec<&ProverRelation>)| !group.is_empty())
            .collect();
        let group_weights: Vec<Vec<Gf>> = groups
            .iter()
            .map(|(index, group)| {
                let claims: Vec<B128> = group.iter().map(|r| r.claim).collect();
                Self::relation_weights(&mut t, *index, &claims)
            })
            .collect();
        let point_gf: Vec<Gf> = point.iter().map(|&x| b128_to_gf(x)).collect();
        let witness_oracle = &oracles[oracle];
        let witness_opening = self.pcs[oracle].open_bit_mle(
            &mut Self::fork(&t, WITNESS_FORK),
            &witness_oracle.packed,
            &witness_oracle.data,
            &witness_oracle.round0,
            &point_gf,
        );
        let mut relation_proofs = Vec::new();
        for ((index, group), weights) in groups.iter().zip(&group_weights) {
            let index = *index;
            let (basis, target) = if group.len() == 1 {
                (group[0].basis.clone(), b128_to_gf(group[0].claim))
            } else {
                let mut basis = vec![Gf::zero(); group[0].basis.len()];
                let mut target = Gf::zero();
                for (relation, &w) in group.iter().zip(weights) {
                    for (slot, &b) in basis.iter_mut().zip(&relation.basis) {
                        *slot = *slot + w * (b);
                    }
                    target = target + w * b128_to_gf(relation.claim);
                }
                (basis.into_iter().collect(), target)
            };
            let o = &oracles[index];
            relation_proofs.push(self.pcs[index].open_basis(
                &mut Self::fork(&t, 1 + index as u64),
                &o.packed,
                &o.data,
                &o.round0,
                basis,
                target,
            ));
        }
        drop(opening);
        let roots = oracles.iter().map(|o| o.root).collect();
        let rounds0 = oracles.iter().map(|o| o.round0.round()).collect();
        Ok(Proof {
            messages,
            roots,
            rounds0,
            witness: witness_opening,
            relations: relation_proofs,
        })
    }

    /// Verify `proof` for the statement `public`: the constraint system's
    /// in/out values (`ValueVec::inout`, `n_inout` of them). The constants
    /// are the verifier's own, bound through the statement digest.
    pub fn verify(&self, public: &[Word], proof: &Proof) -> Result<(), Error> {
        if proof.roots.len() != self.specs.len() || proof.rounds0.len() != self.specs.len() {
            return Err(Error::Invalid("oracle count"));
        }
        let mut t = self.transcript(public);
        let mut channel = VerifierChannel {
            transcript: &mut t,
            messages: &proof.messages,
            specs: self.specs.clone(),
            pcs: &self.pcs,
            roots: &proof.roots,
            rounds0: &proof.rounds0,
            oracles: Vec::new(),
            relations: Vec::new(),
        };
        let (oracle, point, value) = self
            .verifier
            .verify_to_evaluation(public, &mut channel)
            .map_err(|e| Error::Binius(e.to_string()))?;
        if !channel.messages.is_empty() || !channel.specs.is_empty() {
            return Err(Error::Invalid("trailing PIOP prefix data"));
        }
        let VerifierChannel {
            oracles, relations, ..
        } = channel;

        Self::absorb_evaluation(&mut t, value);
        let groups: Vec<(Oracle, Vec<&VerifierRelation>)> = (0..oracles.len())
            .map(|index| {
                (
                    index,
                    relations.iter().filter(|r| r.oracle == index).collect(),
                )
            })
            .filter(|(_, group): &(Oracle, Vec<&VerifierRelation>)| !group.is_empty())
            .collect();
        let group_weights: Vec<Vec<Gf>> = groups
            .iter()
            .map(|(index, group)| {
                let claims: Vec<B128> = group.iter().map(|r| r.claim).collect();
                Self::relation_weights(&mut t, *index, &claims)
            })
            .collect();
        let point_gf: Vec<Gf> = point.iter().map(|&x| b128_to_gf(x)).collect();
        let witness_oracle = &oracles[oracle];
        self.pcs[oracle].verify_bit_mle(
            &mut Self::fork(&t, WITNESS_FORK),
            &witness_oracle.root,
            &witness_oracle.round0,
            &point_gf,
            b128_to_gf(value),
            &proof.witness,
        )?;
        let mut used = 0;
        for ((index, group), weights) in groups.iter().zip(&group_weights) {
            let index = *index;
            let lig = proof
                .relations
                .get(used)
                .ok_or(Error::Invalid("missing relation opening"))?;
            used += 1;
            let target = group
                .iter()
                .zip(weights)
                .fold(Gf::zero(), |acc, (r, &w)| acc + w * b128_to_gf(r.claim));
            let eval_b = |prefix: &[Gf], log_y: usize| -> Vec<Gf> {
                (0..1usize << log_y)
                    .map(|y| {
                        let mut pt: Vec<B128> = prefix.iter().map(|&g| gf_to_b128(g)).collect();
                        for j in 0..log_y {
                            pt.push(if y >> j & 1 == 1 {
                                B128::ONE
                            } else {
                                B128::ZERO
                            });
                        }
                        group.iter().zip(weights).fold(Gf::zero(), |acc, (r, &w)| {
                            acc + w * b128_to_gf((r.transparent)(&pt))
                        })
                    })
                    .collect()
            };
            let o = &oracles[index];
            self.pcs[index].verify_basis(
                &mut Self::fork(&t, 1 + index as u64),
                &o.root,
                &o.round0,
                target,
                eval_b,
                lig,
            )?;
        }
        if used != proof.relations.len() {
            return Err(Error::Invalid("extra relation openings"));
        }
        Ok(())
    }

    /// Decode for this prepared relation. Verification remains a separate
    /// operation; successful decoding does not accept a proof.
    pub fn proof_from_bytes(&self, bytes: &[u8]) -> Result<Proof, Error> {
        let mut r = Reader::new(bytes);
        if r.take(8)? != MAGIC {
            return Err(Error::Invalid("proof magic"));
        }
        let n = r.len()?;
        if n > MAX_MESSAGES || n > r.remaining() / 16 {
            return Err(Error::Codec(CodecError::Truncated));
        }
        let mut messages = Vec::with_capacity(n);
        for _ in 0..n {
            messages.push(r.u128()?);
        }
        let oracles = r.len()?;
        if oracles != self.specs.len() {
            return Err(Error::Invalid("oracle count"));
        }
        let mut roots = Vec::with_capacity(oracles);
        for _ in 0..oracles {
            roots.push(r.take(32)?.try_into().expect("32 bytes"));
        }
        let mut rounds0 = Vec::with_capacity(oracles);
        for _ in 0..oracles {
            rounds0.push(read_round0(&mut r)?);
        }
        let witness = BitMleOpening::read(&mut r)?;
        let n = r.len()?;
        if n > oracles {
            return Err(Error::Invalid("relation opening count"));
        }
        let mut relations = Vec::with_capacity(n);
        for _ in 0..n {
            relations.push(read_ligerito(&mut r)?);
        }
        if r.remaining() != 0 {
            return Err(Error::Codec(CodecError::NonCanonical));
        }
        Ok(Proof {
            messages,
            roots,
            rounds0,
            witness,
            relations,
        })
    }
}

impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(MAGIC);
        w.len(self.messages.len());
        for &m in &self.messages {
            w.u128(m);
        }
        w.len(self.roots.len());
        for root in &self.roots {
            w.bytes(root);
        }
        for round in &self.rounds0 {
            write_round0(&mut w, round);
        }
        self.witness.write(&mut w);
        w.len(self.relations.len());
        for lig in &self.relations {
            write_ligerito(&mut w, lig);
        }
        w.into_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_frontend::{Circuit, CircuitBuilder, Wire};

    /// `2^log` native `u32 × u32 → u64` gates: the same wires as the native
    /// multiplication benchmark's Binius64 circuit.
    fn mul_circuit(log: usize) -> (Circuit, Vec<[Wire; 3]>) {
        let builder = CircuitBuilder::new();
        let wires = (0..1usize << log)
            .map(|_| {
                let a = builder.add_witness();
                let b = builder.add_witness();
                let c = builder.add_witness();
                builder.assert_zero("a is u32", builder.shr(a, 32));
                builder.assert_zero("b is u32", builder.shr(b, 32));
                let (hi, lo) = builder.imul(a, b);
                builder.assert_zero("product high word", hi);
                builder.assert_eq("full u64 product", lo, c);
                [a, b, c]
            })
            .collect();
        (builder.build(), wires)
    }

    /// Fill the gates; a corrupted product fails Binius64's own evaluation.
    fn mul_witness(circuit: &Circuit, wires: &[[Wire; 3]], corrupt: bool) -> Option<ValueVec> {
        let mut filler = circuit.new_witness_filler();
        for (i, [a, b, c]) in wires.iter().enumerate() {
            let x = (i as u64).wrapping_mul(0x9e37_79b9) & 0xffff_ffff;
            let y = u64::from(u32::MAX) - i as u64;
            filler[*a] = Word(x);
            filler[*b] = Word(y);
            filler[*c] = Word(x * y ^ u64::from(corrupt && i == 0));
        }
        circuit.populate_wire_witness(&mut filler).ok()?;
        Some(filler.into_value_vec())
    }

    /// One BMUL gate per row, product bound to committed witness words.
    fn bmul_circuit(log: usize) -> (Circuit, Vec<[Wire; 6]>) {
        let builder = CircuitBuilder::new();
        let wires = (0..1usize << log)
            .map(|_| {
                let [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi]: [Wire; 6] =
                    std::array::from_fn(|_| builder.add_witness());
                let (lo, hi) = builder.bmul(a_lo, a_hi, b_lo, b_hi);
                builder.assert_eq("ghash product low", lo, c_lo);
                builder.assert_eq("ghash product high", hi, c_hi);
                [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi]
            })
            .collect();
        (builder.build(), wires)
    }

    /// Expected products computed with BitZ's own GF(2^128) — the same GHASH
    /// field and the same `(lo, hi)` coefficient words as the BMUL gate.
    fn bmul_witness(circuit: &Circuit, wires: &[[Wire; 6]], corrupt: bool) -> Option<ValueVec> {
        let mut filler = circuit.new_witness_filler();
        for (i, [a_lo, a_hi, b_lo, b_hi, c_lo, c_hi]) in wires.iter().enumerate() {
            let a = Gf::from_polynomial_words([(i as u64).wrapping_mul(0x9e37_79b9), !(i as u64)]);
            let b = Gf::from_polynomial_words([
                0x0123_4567_89ab_cdef ^ i as u64,
                ((i as u64) << 32) | 1,
            ]);
            let c = a * b;
            filler[*a_lo] = Word(a.as_words()[0]);
            filler[*a_hi] = Word(a.as_words()[1]);
            filler[*b_lo] = Word(b.as_words()[0]);
            filler[*b_hi] = Word(b.as_words()[1]);
            filler[*c_lo] = Word(c.as_words()[0] ^ u64::from(corrupt && i == 0));
            filler[*c_hi] = Word(c.as_words()[1]);
        }
        circuit.populate_wire_witness(&mut filler).ok()?;
        Some(filler.into_value_vec())
    }

    #[test]
    fn bmul_circuit_round_trips_through_the_bitz_opener() {
        let (circuit, wires) = bmul_circuit(11);
        let prepared = Prepared::new(circuit.constraint_system()).unwrap();
        // BinMul commits no extra oracle: the witness is the only one.
        assert_eq!(prepared.oracle_specs().len(), 1);
        assert!(
            prepared
                .security()
                .terms
                .iter()
                .any(|t| t.name == "Binius64 BinMul reduction")
        );
        assert!(prepared.security().algebraic_bits >= 100.0);
        assert!(bmul_witness(&circuit, &wires, true).is_none());
        let witness = bmul_witness(&circuit, &wires, false).unwrap();
        let proof = prepared.prove(&witness).unwrap();
        let bytes = proof.to_bytes();
        prepared
            .verify(witness.inout(), &prepared.proof_from_bytes(&bytes).unwrap())
            .unwrap();
        // Tampering in the PIOP prefix, mid-proof and in the opening rejects.
        for at in [8 + 16 * 3, bytes.len() / 2, bytes.len() - 40] {
            let mut tampered = bytes.clone();
            tampered[at] ^= 1;
            let rejected = match prepared.proof_from_bytes(&tampered) {
                Ok(decoded) => prepared.verify(witness.inout(), &decoded).is_err(),
                Err(_) => true,
            };
            assert!(rejected, "byte {at}");
        }
    }

    #[test]
    fn selectable_rates_bind_both_oracles_and_reject_cross_rate_proofs() {
        let (circuit, wires) = mul_circuit(11);
        let witness = mul_witness(&circuit, &wires, false).unwrap();
        let prepared: Vec<_> = (1..=3)
            .map(|rate| Prepared::with_log_inv_rate(circuit.constraint_system(), rate).unwrap())
            .collect();
        for (i, relation) in prepared.iter().enumerate() {
            assert!(relation.security().algebraic_bits >= 100.0);
            for pcs in &relation.pcs {
                assert_eq!(pcs.params().log_inv_rate, i + 1);
                assert_eq!(pcs.config().levels[0].log_inv_rate, i + 1);
            }
            let proof = relation.prove(&witness).unwrap();
            let bytes = proof.to_bytes();
            relation
                .verify(witness.inout(), &relation.proof_from_bytes(&bytes).unwrap())
                .unwrap();
            for (j, other) in prepared.iter().enumerate() {
                if i != j {
                    assert!(other.verify(witness.inout(), &proof).is_err());
                }
            }
        }
        for invalid_rate in [0, 4] {
            assert!(
                Prepared::with_log_inv_rate(circuit.constraint_system(), invalid_rate).is_err()
            );
        }
    }

    #[test]
    fn multiplication_circuit_round_trips_through_the_bitz_opener() {
        let (circuit, wires) = mul_circuit(11);
        let prepared = Prepared::new(circuit.constraint_system()).unwrap();
        assert_eq!(
            prepared.oracle_specs().len(),
            2,
            "witness + logup* pushforward"
        );
        assert!(prepared.security().algebraic_bits >= 100.0);
        assert!(mul_witness(&circuit, &wires, true).is_none());
        let witness = mul_witness(&circuit, &wires, false).unwrap();
        let proof = prepared.prove(&witness).unwrap();
        assert_eq!(proof.relations.len(), 1);
        let bytes = proof.to_bytes();
        let decoded = prepared.proof_from_bytes(&bytes).unwrap();
        prepared.verify(witness.inout(), &decoded).unwrap();

        // Tampering anywhere is rejected: the PIOP prefix, a Round-0 value,
        // the witness opening and the pushforward opening.
        for at in [
            8 + 16 * 3,
            bytes.len() / 3,
            bytes.len() / 2,
            bytes.len() - 40,
        ] {
            let mut tampered = bytes.clone();
            tampered[at] ^= 1;
            let rejected = match prepared.proof_from_bytes(&tampered) {
                Ok(decoded) => prepared.verify(witness.inout(), &decoded).is_err(),
                Err(_) => true,
            };
            assert!(rejected, "byte {at}");
        }
    }

    #[test]
    fn multiplication_circuit_round_trips_at_rate_one_eighth() {
        let (circuit, wires) = mul_circuit(11);
        let half = Prepared::new(circuit.constraint_system()).unwrap();
        let eighth = Prepared::with_rate(circuit.constraint_system(), 3).unwrap();
        assert_eq!(eighth.log_inv_rate(), 3);
        assert!(eighth.security().algebraic_bits >= 100.0);
        assert!(
            eighth.opener(0).level0_queries() < half.opener(0).level0_queries(),
            "a lower rate needs fewer level-0 queries"
        );
        let witness = mul_witness(&circuit, &wires, false).unwrap();
        let proof = eighth.prove(&witness).unwrap();
        let bytes = proof.to_bytes();
        eighth
            .verify(witness.inout(), &eighth.proof_from_bytes(&bytes).unwrap())
            .unwrap();
        // A verifier prepared at the other rate rejects it (the rate is in
        // the statement digest and the opener geometry).
        let rejected = match half.proof_from_bytes(&bytes) {
            Ok(decoded) => half.verify(witness.inout(), &decoded).is_err(),
            Err(_) => true,
        };
        assert!(rejected);
    }

    #[test]
    fn round_by_round_accounting_gates_every_term_at_100() {
        let (circuit, wires) = mul_circuit(11);
        let union = Prepared::new(circuit.constraint_system()).unwrap();
        let rbr = Prepared::with_options(circuit.constraint_system(), 1, Accounting::RoundByRound)
            .unwrap();
        assert_eq!(rbr.accounting(), Accounting::RoundByRound);
        // Every term on its own clears 100 at the smallest component target,
        // while their sum does not — that is the union bound's extra margin.
        assert_eq!(rbr.component_bits(), 100);
        assert!(rbr.security().round_by_round_bits >= 100.0);
        assert!(rbr.security().union_bound_bits < 100.0);
        assert!(union.component_bits() > rbr.component_bits());
        assert!(union.security().union_bound_bits >= 100.0);
        assert!(rbr.opener(0).level0_queries() < union.opener(0).level0_queries());
        let witness = mul_witness(&circuit, &wires, false).unwrap();
        let proof = rbr.prove(&witness).unwrap();
        let bytes = proof.to_bytes();
        rbr.verify(witness.inout(), &rbr.proof_from_bytes(&bytes).unwrap())
            .unwrap();
        let rejected = match union.proof_from_bytes(&bytes) {
            Ok(decoded) => union.verify(witness.inout(), &decoded).is_err(),
            Err(_) => true,
        };
        assert!(rejected, "the accounting model is part of the statement");
    }

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

    /// The IntMul reduction's queued pushforward relation must mean the same
    /// thing on both sides: `⟨basis, oracle⟩ = claim` on the prover, and the
    /// verifier's transparent closure must be the basis MLE in BitZ's
    /// coordinate convention.
    #[test]
    fn pushforward_relation_is_consistent_across_prover_and_verifier() {
        let (circuit, wires) = mul_circuit(11);
        let prepared = Prepared::new(circuit.constraint_system()).unwrap();
        let witness = mul_witness(&circuit, &wires, false).unwrap();
        let public = witness.inout();

        let mut t = prepared.transcript(public);
        let mut channel = ProverChannel {
            transcript: &mut t,
            messages: Vec::new(),
            specs: prepared.specs.clone(),
            pcs: &prepared.pcs,
            oracles: Vec::new(),
            relations: Vec::new(),
        };
        let (oracle, _packed, point, value) = prepared
            .prover
            .prove_to_evaluation::<_, OptimalPackedB128, _>(
                &witness,
                &mut channel,
                &binius_compute::GlobalAllocator,
            )
            .unwrap();
        assert_eq!(oracle, 0);
        let ProverChannel {
            messages,
            oracles,
            relations,
            ..
        } = channel;
        assert_eq!(oracles.len(), 2);
        // The transparent logup* reduction opens the pushforward twice: an
        // eq-basis evaluation claim and a product claim against the table.
        assert_eq!(
            relations.len(),
            2,
            "logup* pushforward evaluation + product relations"
        );
        let pushforward = &oracles[1].packed;
        assert_eq!(pushforward.len(), 1 << 16);
        for (which, relation) in relations.iter().enumerate() {
            assert_eq!(relation.oracle, 1);
            let inner = relation
                .basis
                .iter()
                .zip(pushforward)
                .fold(Gf::zero(), |acc, (&b, &f)| acc + (b) * (f));
            assert_eq!(
                inner,
                b128_to_gf(relation.claim),
                "prover-side relation {which}"
            );
        }
        let claims: Vec<B128> = relations.iter().map(|r| r.claim).collect();

        let roots: Vec<Hash> = oracles.iter().map(|o| o.root).collect();
        let rounds0: Vec<OodRound> = oracles.iter().map(|o| o.round0.round()).collect();
        let mut tv = prepared.transcript(public);
        let mut vchannel = VerifierChannel {
            transcript: &mut tv,
            messages: &messages,
            specs: prepared.specs.clone(),
            pcs: &prepared.pcs,
            roots: &roots,
            rounds0: &rounds0,
            oracles: Vec::new(),
            relations: Vec::new(),
        };
        let (voracle, vpoint, vvalue) = prepared
            .verifier
            .verify_to_evaluation(public, &mut vchannel)
            .unwrap();
        assert_eq!((voracle, &vpoint, vvalue), (oracle, &point, value));
        assert_eq!(vchannel.relations.len(), 2);
        let mut scratch = Blake3Transcript::new();
        scratch.absorb_slice(b"diagnostic point");
        let pt: Vec<Gf> = (0..16).map(|_| scratch.get_field_challenge(&())).collect();
        let pt_b128: Vec<B128> = pt.iter().map(|&g| gf_to_b128(g)).collect();
        let reversed: Vec<B128> = pt_b128.iter().rev().copied().collect();
        for (which, (relation, vrelation)) in relations.iter().zip(&vchannel.relations).enumerate()
        {
            assert_eq!(vrelation.oracle, 1);
            assert_eq!(vrelation.claim, relation.claim);
            let dense = relation
                .basis
                .iter()
                .zip(eq_table(&pt))
                .fold(Gf::zero(), |acc, (&b, e)| acc + (b) * e);
            let via_closure = b128_to_gf((vrelation.transparent)(&pt_b128));
            let via_reversed = b128_to_gf((vrelation.transparent)(&reversed));
            assert!(
                dense == via_closure || dense == via_reversed,
                "relation {which}: transparent closure is not the basis MLE in either order"
            );
            assert_eq!(
                dense, via_closure,
                "relation {which}: transparent closure uses reversed coordinates"
            );
        }

        // Now the two openings, each on its own fork, verified one at a time.
        // The pushforward's two relations combine under one draw, as in
        // `prove`.
        Prepared::absorb_evaluation(&mut t, value);
        let weights = Prepared::relation_weights(&mut t, 1, &claims);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0], Gf::one());
        let mut combined = vec![Gf::zero(); pushforward.len()];
        let mut target = Gf::zero();
        for (relation, &w) in relations.iter().zip(&weights) {
            for (slot, &b) in combined.iter_mut().zip(&relation.basis) {
                *slot = *slot + w * (b);
            }
            target = target + w * b128_to_gf(relation.claim);
        }
        let combined_f128: Vec<Gf128> = combined.iter().copied().collect();
        let point_gf: Vec<Gf> = point.iter().map(|&x| b128_to_gf(x)).collect();
        let w = &oracles[0];
        let mut fork_w = Prepared::fork(&t, WITNESS_FORK);
        let witness_opening =
            prepared.pcs[0].open_bit_mle(&mut fork_w, &w.packed, &w.data, &w.round0, &point_gf);
        let o = &oracles[1];
        let pushforward_opening = prepared.pcs[1].open_basis(
            &mut Prepared::fork(&t, 2),
            &o.packed,
            &o.data,
            &o.round0,
            combined_f128.clone(),
            target,
        );

        drop(vchannel);
        // Replay the verifier up to the pushforward opening, so the succinct
        // and the dense flock verifiers can each be run on a fresh state.
        let replay = || {
            let mut tv = prepared.transcript(public);
            let mut vchannel = VerifierChannel {
                transcript: &mut tv,
                messages: &messages,
                specs: prepared.specs.clone(),
                pcs: &prepared.pcs,
                roots: &roots,
                rounds0: &rounds0,
                oracles: Vec::new(),
                relations: Vec::new(),
            };
            let (_, _, vvalue) = prepared
                .verifier
                .verify_to_evaluation(public, &mut vchannel)
                .unwrap();
            let VerifierChannel {
                oracles: voracles,
                relations: vrelations,
                ..
            } = vchannel;
            Prepared::absorb_evaluation(&mut tv, vvalue);
            let vclaims: Vec<B128> = vrelations.iter().map(|r| r.claim).collect();
            let vweights = Prepared::relation_weights(&mut tv, 1, &vclaims);
            assert_eq!(vweights, weights);
            let mut fork_w = Prepared::fork(&tv, WITNESS_FORK);
            prepared.pcs[0]
                .verify_bit_mle(
                    &mut fork_w,
                    &voracles[0].root,
                    &voracles[0].round0,
                    &point_gf,
                    b128_to_gf(vvalue),
                    &witness_opening,
                )
                .expect("witness bit-MLE opening verifies");
            (tv, voracles, vrelations)
        };
        // Dense flock verifier on the prover's combined basis.
        {
            let (tv, voracles, _) = replay();
            let mut tv = Prepared::fork(&tv, 2);
            let eta: Gf = tv.get_field_challenge(&());
            let mut dense = combined_f128.clone();
            crate::ligerito_flock::add_ood_basis(
                &mut dense,
                &oracles[1].packed,
                voracles[1].round0.point(),
                eta,
                None,
            );
            let target = target + eta * voracles[1].round0.y();
            let dense_ok = flock_core::pcs::ligerito::recursive_verifier_with_basis(
                prepared.pcs[1].verifier_config(),
                &pushforward_opening,
                &dense,
                (target),
                &voracles[1].root,
                &mut crate::ligerito_flock::ZincChallenger(&mut tv),
            );
            assert!(
                dense_ok,
                "flock's dense basis verifier rejects the pushforward opening"
            );
        }
        let (tv, voracles, vrelations) = replay();
        let mut tv = Prepared::fork(&tv, 2);
        let eval_b = |prefix: &[Gf], log_y: usize| -> Vec<Gf> {
            (0..1usize << log_y)
                .map(|y| {
                    let mut pt: Vec<B128> = prefix.iter().map(|&g| gf_to_b128(g)).collect();
                    for j in 0..log_y {
                        pt.push(if y >> j & 1 == 1 {
                            B128::ONE
                        } else {
                            B128::ZERO
                        });
                    }
                    vrelations
                        .iter()
                        .zip(&weights)
                        .fold(Gf::zero(), |acc, (r, &w)| {
                            acc + w * b128_to_gf((r.transparent)(&pt))
                        })
                })
                .collect()
        };
        // The combined closure at boolean tails must agree with the combined
        // dense basis.
        let prefix: Vec<Gf> = (0..11).map(|_| scratch.get_field_challenge(&())).collect();
        let via_closure = eval_b(&prefix, 5);
        for (y, &value) in via_closure.iter().enumerate() {
            let mut point = prefix.clone();
            for j in 0..5 {
                point.push(if y >> j & 1 == 1 {
                    Gf::one()
                } else {
                    Gf::zero()
                });
            }
            let dense = combined
                .iter()
                .zip(eq_table(&point))
                .fold(Gf::zero(), |acc, (&b, e)| acc + b * e);
            assert_eq!(dense, value, "boolean tail {y}");
        }
        prepared.pcs[1]
            .verify_basis(
                &mut tv,
                &voracles[1].root,
                &voracles[1].round0,
                target,
                eval_b,
                &pushforward_opening,
            )
            .expect("pushforward basis opening verifies");
    }

    #[test]
    fn and_only_circuit_uses_one_oracle() {
        let builder = CircuitBuilder::new();
        let gates = 1usize << 12;
        let wires: Vec<_> = (0..gates)
            .map(|i| {
                let a = builder.add_witness();
                let b = builder.add_witness();
                // The last output is public, so the statement check below has
                // an in/out word to flip.
                let c = if i + 1 == gates {
                    builder.add_inout()
                } else {
                    builder.add_witness()
                };
                builder.assert_eq("c = a & b", builder.band(a, b), c);
                [a, b, c]
            })
            .collect();
        let circuit = builder.build();
        let prepared = Prepared::new(circuit.constraint_system()).unwrap();
        assert_eq!(prepared.oracle_specs().len(), 1);
        let mut filler = circuit.new_witness_filler();
        for (i, [a, b, c]) in wires.iter().enumerate() {
            let x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let y = x.rotate_left(17) ^ 0x5555;
            filler[*a] = Word(x);
            filler[*b] = Word(y);
            filler[*c] = Word(x & y);
        }
        circuit.populate_wire_witness(&mut filler).unwrap();
        let witness = filler.into_value_vec();
        let proof = prepared.prove(&witness).unwrap();
        assert!(proof.relations.is_empty());
        let decoded = prepared.proof_from_bytes(&proof.to_bytes()).unwrap();
        prepared.verify(witness.inout(), &decoded).unwrap();
        assert_eq!(witness.inout().len(), 1);
        let mut wrong = witness.inout().to_vec();
        wrong[0].0 ^= 1;
        assert!(prepared.verify(&wrong, &decoded).is_err());
    }
}
