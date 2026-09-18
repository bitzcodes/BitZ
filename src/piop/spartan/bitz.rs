//! The u32 multiplication relation `x · y = product` over BitZ.
//!
//! Spartan constrains the integer assignment
//!
//! ```text
//! [constant block | x block | y block | product block]
//! ```
//!
//! over a transcript-selected prime. The commitment contains only the compact
//! `32 + 32 + 64` little-endian bits for each multiplication. This module
//! describes that relation to the shared protocol of [`super::protocol`]:
//! its committed tensor, its Bit selector matrices, the block-to-slot
//! map whose adjoint bitifies Spartan's terminal claim, its statement
//! binding and its transcript domains. The outer Spartan reduction uses a
//! fixed known-zero univariate-prefix skip.

use crate::piop::spartan::mul::{MulLayout, MulWitness};
#[cfg(any(test, feature = "bench-internals"))]
use crate::piop::spartan::protocol::PreparedRelation;
use crate::piop::spartan::protocol::ProtocolError;
#[cfg(feature = "bench-internals")]
use crate::piop::spartan::protocol::terminal::PreparedTerminalOpening;

use crate::piop::spartan::SpartanField as _;
use field::{RingOps, Uint};

use flock_core::pcs::{commit::Commitment, ligerito::ProverConfig as LigProverConfig};

#[cfg(test)]
use crate::pcs::Q100Element;
use crate::{
    ligerito_flock::ModQOpeningKind,
    pcs::{FQ_MOD, IntegerMatrixLayout},
};

use super::{
    profile::{IopInstanceFacts, IopSecurityParams},
    protocol::{
        self, BindingHasher, BlockTable, Domains, FieldConfig, Kernel, MatrixSource, PiopWitness,
        RelationSpec, SlotRange,
    },
    u32_mul::{
        U32_MUL_PRODUCT_BITS, U32_MUL_PRODUCT_SLOT_START, U32_MUL_X_BITS, U32_MUL_X_SLOT_START,
        U32_MUL_Y_BITS, U32_MUL_Y_SLOT_START, u32_mul_constraint_matrices,
    },
};

pub use super::protocol::{MIN_PRODUCTION_GATE_VARS, SpartanBitzField};

/// Constructs the fixed `q = 2^100 - 15` runtime field configuration.
pub fn spartan_bitz_field_config() -> <SpartanBitzField as crate::piop::spartan::SpartanField>::Config
{
    SpartanBitzField::make_cfg(&Uint::from(FQ_MOD)).expect("FQ_MOD is a valid odd prime modulus")
}

/// Number of little-endian row variables represented by the canonical
/// univariate-prefix reduction.
pub const U32_MUL_UNIVARIATE_SKIP_VARS: usize = 3;

/// Maximum degree of the first univariate-skip message polynomial:
/// `2 * (2^K - 1)` for [`U32_MUL_UNIVARIATE_SKIP_VARS`] `K`.
pub const U32_MUL_UNIVARIATE_SKIP_DEGREE: u32 = 2 * ((1_u32 << U32_MUL_UNIVARIATE_SKIP_VARS) - 1);

const U32_MUL_BINDING_DOMAIN: &[u8] =
    b"bitz/spartan-u32-mul/assignment/v3-runtime-skip-k3-profiled-ligerito";

/// The transcript domains of the u32 protocol. The per-draw PIOP grinding
/// domain guards EVERY challenge the Spartan PIOP draws (τ coordinates,
/// outer/bind/inner round challenges) at the profile's initial bound — the
/// maximum any single draw needs — so it is conservative but uniformly
/// round-by-round sound.
static U32_MUL_DOMAINS: Domains = Domains {
    statement_tag: b"u32-statement",
    prime_sampling: b"bitz/spartan-u32-mul/runtime-prime/v2",
    initial_grinding: b"bitz/spartan-u32-mul/grinding/initial/v1",
    piop_grinding: b"bitz/spartan-u32-mul/grinding/piop/v1",
    terminal_grinding: b"bitz/spartan-u32-mul/grinding/terminal/v1",
    bitified_claim: b"bitz/spartan-bitz/bitified-claim/v3",
    opening: ModQOpeningKind::U32Mul,
    claim_tag: b"",
    reduction_grinding: b"",
    reduction_prime: b"",
    scopes: crate::protocol_scopes!("spartan-bitz"),
};

/// The public statement facts the security-profile derivation consumes for
/// a u32 multiplication batch: per-row integer defects `|x*y - z|` are
/// below `2^65` (bounded to `2^80` conservatively, far below any sampled
/// prime), the Step-5.1 lift sums `2^t` terms, and the opening is the
/// DIRECT exponent-fold path, so the interval width is capped at
/// `c_w = 127 - t - W` and the row functional always fits ONE chunk.
pub(crate) fn u32_mul_instance_facts(p: &IntegerMatrixLayout, row_vars: usize) -> IopInstanceFacts {
    IopInstanceFacts {
        defect_log2_bound: 80,
        lift_arity_log2: p.row_vars as u32,
        opening_t: p.row_vars as u32,
        opening_word_bits: p.word_bits as u32,
        direct_opening: true,
        tau_arity: row_vars.max(1) as u32,
        piop_degree: U32_MUL_UNIVARIATE_SKIP_DEGREE,
        step50_magnitude_log2: 0,
    }
}

impl RelationSpec for MulLayout<u32> {
    type Coefficient = bool;
    type Witness = MulWitness<u32>;
    type Map = Self;

    fn domains(&self) -> &'static Domains {
        &U32_MUL_DOMAINS
    }

    fn committed_layout(&self) -> IntegerMatrixLayout {
        MulLayout::committed_layout(self)
    }
    fn opening_layout(&self) -> IntegerMatrixLayout {
        self.bitz_params()
    }
    fn opening_word_bits(&self) -> usize {
        self.word_bits()
    }
    fn map(&self) -> Option<&Self> {
        (!self.uses_direct_opening()).then_some(self)
    }
    fn derived_rows(&self, witness: &Self::Witness) -> Option<Vec<Vec<u64>>> {
        (!self.uses_direct_opening()).then(|| witness.derived_bit_rows())
    }
    fn claim_digest(&self, frame: protocol::ClaimFrame<'_>) -> Result<[u8; 32], ProtocolError> {
        self.packed_claim_digest(frame)
    }

    fn gate_vars(&self) -> usize {
        MulLayout::<u32>::gate_vars(self)
    }

    fn instance_facts(&self) -> IopInstanceFacts {
        let mut facts =
            u32_mul_instance_facts(&self.bitz_params(), MulLayout::<u32>::gate_vars(self));
        facts.opening_word_bits = self.word_bits() as u32;
        facts.direct_opening = self.uses_direct_opening();
        facts
    }

    fn matrices(&self) -> Result<MatrixSource<bool>, ProtocolError> {
        MatrixSource::skeleton(u32_mul_constraint_matrices(self, true)?)
    }

    fn validate_geometry(&self) -> Result<(), ProtocolError> {
        self.validate_protocol_geometry()
    }

    /// Block order in the integer assignment is 00=constant, 01=x, 10=y,
    /// 11=product, with the first block-selector coordinate as the low bit.
    fn block_table(&self) -> BlockTable {
        let table = BlockTable::new(
            2,
            vec![
                None,
                Some(SlotRange {
                    bit_slot_start: U32_MUL_X_SLOT_START,
                    bit_count: U32_MUL_X_BITS,
                }),
                Some(SlotRange {
                    bit_slot_start: U32_MUL_Y_SLOT_START,
                    bit_count: U32_MUL_Y_BITS,
                }),
                Some(SlotRange {
                    bit_slot_start: U32_MUL_PRODUCT_SLOT_START,
                    bit_count: U32_MUL_PRODUCT_BITS,
                }),
            ],
        )
        .expect("the u32 block table is complete");
        if self.uses_direct_opening() {
            table
        } else {
            table.with_word_packing(32, self.word_bits())
        }
    }

    fn kernel(&self) -> Kernel {
        Kernel::UnivariateSkip {
            skip_vars: U32_MUL_UNIVARIATE_SKIP_VARS,
        }
    }

    /// The shared bridge transcript covers both the forest/GKR draws and the
    /// ring-switch draw, so it must satisfy the stronger of their two
    /// profile requirements.
    fn opener_grinding_bits(&self, security: &IopSecurityParams) -> u32 {
        security
            .forest_round_grinding_bits
            .max(security.ring_switch_grinding_bits)
    }

    fn check_witness(&self, witness: &MulWitness<u32>) -> Result<(), ProtocolError> {
        if witness.layout() != self {
            return Err(ProtocolError::RelationWitnessLayoutMismatch);
        }
        Ok(())
    }

    fn assignment_binding(
        &self,
        commitment: &Commitment,
        security: &IopSecurityParams,
        ligerito: &LigProverConfig,
    ) -> Result<[u8; 32], ProtocolError> {
        let p = self.bitz_params();
        let mut hasher = BindingHasher::new();
        hasher.bytes(U32_MUL_BINDING_DOMAIN).bytes(&commitment.root);
        hasher.commitment_params(&commitment.params)?;
        hasher.prefixed(security.profile_name.as_bytes())?;
        hasher.u32(security.lambda)?;
        hasher
            .u128_le(security.projection_min)
            .u128_le(security.projection_max)
            .byte(u8::from(security.projection_full_width));
        hasher.u32(security.initial_grinding_bits)?;
        hasher.u32(security.piop_round_grinding_bits)?;
        hasher.u32(security.terminal_grinding_bits)?;
        match security.reduction {
            Some(reduction) => {
                hasher.byte(1).u128_le(reduction.min).u128_le(reduction.max);
                hasher.u32(reduction.grinding_bits)?;
            }
            None => {
                hasher.byte(0);
            }
        }
        hasher.u32(security.forest_round_grinding_bits)?;
        hasher.u32(security.ring_switch_grinding_bits)?;
        if let Some(ood) = security.ood {
            // Present only when Round 0 runs, so Round-0-less statements keep
            // their digest.
            hasher.byte(1);
            hasher.u32(ood.grinding_bits)?;
        }
        hasher.usize(security.ligerito_target_bits)?;
        hasher.ligerito_config(ligerito)?;
        hasher.usizes(&[
            self.multiplications(),
            self.capacity(),
            self.assignment_len(),
            MulLayout::<u32>::gate_vars(self),
            p.row_vars,
            p.col_vars,
            p.word_bits,
            U32_MUL_UNIVARIATE_SKIP_VARS,
            U32_MUL_UNIVARIATE_SKIP_DEGREE as usize,
        ])?;
        self.bind_packing(&mut hasher)?;
        Ok(hasher.finalize())
    }

    fn hash_bridge_constants(&self, hasher: &mut BindingHasher) -> Result<(), ProtocolError> {
        let p = self.bitz_params();
        hasher.usizes(&[
            self.multiplications(),
            self.capacity(),
            self.assignment_len(),
            MulLayout::<u32>::gate_vars(self),
            p.row_vars,
            p.col_vars,
            p.word_bits,
            U32_MUL_X_SLOT_START,
            U32_MUL_X_BITS,
            U32_MUL_Y_SLOT_START,
            U32_MUL_Y_BITS,
            U32_MUL_PRODUCT_SLOT_START,
            U32_MUL_PRODUCT_BITS,
        ])?;
        // Mapping version 2: little-endian bits, 00/01/10/11 block order, and
        // canonical nonzero-scale normalization onto the folded row functional.
        hasher.bytes(&[2, 0, 0, 1, 2, 3]);
        self.bind_packing(hasher)?;
        Ok(())
    }

    /// The exact products are the zero-padded operand blocks of the witness
    /// and the assignment is its block table: lend both, no copy.
    fn piop_witness<'w>(
        &self,
        witness: &'w MulWitness<u32>,
        _config: &FieldConfig,
    ) -> Result<PiopWitness<'w>, ProtocolError> {
        Ok(PiopWitness::Mul32(witness))
    }
}

/// Prepares a fixed-q, PCS-only terminal-opening context.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub fn prepare_u32_terminal_bitz_opening(
    prepared: &PreparedRelation<MulLayout<u32>>,
    commitment: &Commitment,
) -> Result<PreparedTerminalOpening<MulLayout<u32>>, ProtocolError> {
    protocol::terminal::prepare(
        prepared,
        commitment,
        b"u32-terminal/early-ood/v2",
        protocol::terminal::StatementPayload::BridgeDigest,
        protocol::terminal::Binding::Paper,
    )
}

#[cfg(test)]
mod tests {
    use crate::piop::spartan::protocol::Proof;
    use crate::piop::spartan::protocol::packed_variables;
    use crate::transcript::traits::Transcript;

    use super::*;
    use crate::{
        pcs::{FQ_BITS, ModQWeightChunks, eq_le_table_fq, fq_sub},
        piop::spartan::{
            matrix::ScaledMleEvaluationClaim,
            profile::{Lambda100, Lambda128, Limber114, ProfileError},
            protocol::{SpartanProof, bitify},
        },
        transcript::Blake3Transcript,
    };

    fn skip_proof(
        proof: &Proof,
    ) -> &super::super::univariate_skip::UnivariateSkipSpartanPiopProof<SpartanBitzField> {
        proof
            .spartan()
            .univariate_skip()
            .expect("the u32 kernel is the univariate skip")
    }

    fn skip_proof_mut(
        proof: &mut Proof,
    ) -> &mut super::super::univariate_skip::UnivariateSkipSpartanPiopProof<SpartanBitzField> {
        match &mut proof.prefix_mut().spartan {
            SpartanProof::UnivariateSkip(spartan) => spartan,
            SpartanProof::Plain(_) => panic!("the u32 kernel is the univariate skip"),
        }
    }

    #[test]
    fn u32_mul_roundtrips_with_a_partial_gate_block() {
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // 2^15 + 77 live gates over a 2^16 capacity: the live rows fill the
        // row domain partially, and not along a univariate-skip block
        // boundary (4105 full blocks of 8 rows plus a 5-row block) — the
        // case the verifier's closed-form matrix binding handles with its
        // two prefix-sum terms.
        let multiplications = (1usize << 15) + 77;
        let witness = MulWitness::<u32>::from_fn_with_word_bits(multiplications, 1, |i| {
            let x = (i as u32).wrapping_mul(0x9e37_79b9) ^ 0x5bd1_e995;
            let y = (i as u32).wrapping_mul(0x85eb_ca6b) | 1;
            (x, y)
        })
        .unwrap();
        let layout = *witness.layout();
        assert_eq!(layout.capacity(), 1 << 16);
        let prepared = PreparedRelation::<MulLayout<u32>>::new(layout).unwrap();
        assert_eq!(
            prepared.skeleton().unwrap().matrices().row_count(),
            multiplications
        );

        let hint = protocol::commit(&prepared, witness.bitz_bit_rows()).unwrap();
        let mut prover_transcript = Blake3Transcript::new();
        let proof = protocol::prove(&mut prover_transcript, &prepared, &witness, &hint).unwrap();
        assert_eq!(
            skip_proof(&proof)
                .outer
                .tail
                .sumcheck
                .round_polynomials
                .len(),
            16 - U32_MUL_UNIVARIATE_SKIP_VARS
        );
        let mut verifier_transcript = Blake3Transcript::new();
        protocol::verify(
            &mut verifier_transcript,
            &prepared,
            &hint.commitment,
            &proof,
        )
        .unwrap();

        // A wrong terminal matrix evaluation is caught downstream: perturb
        // the inner sumcheck's last round (which moves the final claim the
        // verifier's own matrix evaluation must scale into the opening).
        let one = SpartanBitzField::from_with_cfg(1u64, &spartan_bitz_field_config());
        let mut tampered = proof.clone();
        {
            let spartan = skip_proof_mut(&mut tampered);
            let last = spartan.inner.round_polynomials.len() - 1;
            spartan.inner.round_polynomials[last][1] = spartan_bitz_field_config()
                .add(&(spartan.inner.round_polynomials[last][1]), &(&one));
            spartan.inner.round_polynomials[last][2] = spartan_bitz_field_config()
                .sub(&(spartan.inner.round_polynomials[last][2]), &(&one));
        }
        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            protocol::verify(
                &mut verifier_transcript,
                &prepared,
                &hint.commitment,
                &tampered
            )
            .is_err()
        );
    }

    #[test]
    fn early_ood_binds_regime_root_and_payload() {
        use crate::ligerito_flock::{IntEvalRsLigModQProof, LigeritoSelection};
        let witness = MulWitness::<u32>::from_fn(1 << 15, |i| (i as u32, u32::MAX)).unwrap();
        for selection in [LigeritoSelection::JOHNSON, LigeritoSelection::MATCHED_UDR] {
            let p = PreparedRelation::<MulLayout<u32>>::new_with_profile_and_ligerito::<Lambda100>(
                *witness.layout(),
                selection,
            )
            .unwrap();
            let hint = protocol::commit(&p, witness.bitz_bit_rows()).unwrap();
            let mut pt = Blake3Transcript::new();
            let mut proof = protocol::prove(&mut pt, &p, &witness, &hint).unwrap();
            *proof.bitz_mut().direct_mut().unwrap() =
                IntEvalRsLigModQProof::from_bytes(&proof.bitz().to_bytes()).unwrap();
            let check = |proof: &Proof| {
                protocol::verify(&mut Blake3Transcript::new(), &p, &hint.commitment, proof)
            };
            check(&proof).unwrap();
            let mut bad = proof.clone();
            if let Some(round) = bad.bitz_mut().direct_mut().unwrap().ood.as_mut() {
                round.y = round.y + crate::poly::univariate::binary_gf128::Gf128::one();
            } else {
                bad.bitz_mut().direct_mut().unwrap().ood = Some(crate::ligerito_flock::OodRound {
                    y: crate::poly::univariate::binary_gf128::Gf128::zero(),
                    nonce: None,
                });
            }
            assert!(check(&bad).is_err());
            if proof.bitz().ood().is_some() {
                let mut bad = proof.clone();
                bad.bitz_mut().direct_mut().unwrap().ood = None;
                assert!(check(&bad).is_err());
                let mut bad = proof.clone();
                bad.bitz_mut()
                    .direct_mut()
                    .unwrap()
                    .ood
                    .as_mut()
                    .unwrap()
                    .nonce = Some(u64::MAX);
                assert!(check(&bad).is_err());
            }
            let mut root = hint.commitment.clone();
            root.root[0] ^= 1;
            assert!(protocol::verify(&mut Blake3Transcript::new(), &p, &root, &proof).is_err());
            let other = if selection == LigeritoSelection::JOHNSON {
                LigeritoSelection::MATCHED_UDR
            } else {
                LigeritoSelection::JOHNSON
            };
            let foreign = PreparedRelation::<MulLayout<u32>>::new_with_profile_and_ligerito::<
                Lambda100,
            >(*witness.layout(), other)
            .unwrap();
            assert!(
                protocol::verify(
                    &mut Blake3Transcript::new(),
                    &foreign,
                    &hint.commitment,
                    &proof
                )
                .is_err()
            );
        }
    }

    #[test]
    fn u32_mul_roundtrips_and_is_deterministic() {
        // Hold the shared env lock so tests that toggle transcript-shaping
        // `BITZ_*` variables cannot flip them between our prove and verify.
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let multiplications = 1usize << 15;
        let witness = MulWitness::<u32>::from_fn_with_word_bits(multiplications, 1, |i| {
            let x = (i as u32).wrapping_mul(0x9e37_79b9) | 1;
            let y = (i as u32).wrapping_mul(0x85eb_ca6b) | 1;
            (x, y)
        })
        .unwrap();
        let layout = *witness.layout();
        let prepared = PreparedRelation::<MulLayout<u32>>::new(layout).unwrap();

        // The derived interval keeps one exponent-fold chunk and no grinding.
        let security = prepared.security();
        assert_eq!(security.lambda, 100);
        assert_eq!(security.initial_grinding_bits, 0);
        assert_eq!(security.piop_round_grinding_bits, 0);
        assert_eq!(security.forest_round_grinding_bits, 0);
        assert_eq!(
            u32_mul_instance_facts(&layout.bitz_params(), layout.gate_vars()).piop_degree,
            U32_MUL_UNIVARIATE_SKIP_DEGREE
        );
        let p = layout.bitz_params();
        let width = (128 - security.projection_max.leading_zeros()) as usize;
        assert!(width <= 127 - p.row_vars - p.word_bits, "q_bits <= c_w");
        assert_eq!(
            crate::pcs::mod_q_num_chunks(&p, width),
            1,
            "the runtime interval is one-chunk by construction"
        );

        let hint = protocol::commit(&prepared, witness.bitz_bit_rows()).unwrap();
        let mut prover_transcript = Blake3Transcript::new();
        let proof = protocol::prove(&mut prover_transcript, &prepared, &witness, &hint).unwrap();
        assert_eq!(
            skip_proof(&proof).outer.skip.skip_vars,
            U32_MUL_UNIVARIATE_SKIP_VARS as u8
        );
        assert_eq!(skip_proof(&proof).outer.skip.finite_q_evaluations.len(), 6);
        assert_eq!(
            skip_proof(&proof)
                .outer
                .tail
                .sumcheck
                .round_polynomials
                .len(),
            12
        );
        assert_eq!(skip_proof(&proof).inner.round_polynomials.len(), 17);
        assert_eq!(proof.spartan_payload_elements(), 109);
        let mut verifier_transcript = Blake3Transcript::new();
        protocol::verify(
            &mut verifier_transcript,
            &prepared,
            &hint.commitment,
            &proof,
        )
        .unwrap();

        // Determinism: a second prove is byte-identical.
        let mut second_transcript = Blake3Transcript::new();
        let second = protocol::prove(&mut second_transcript, &prepared, &witness, &hint).unwrap();
        assert_eq!(second.bitz().to_bytes(), proof.bitz().to_bytes());
        assert_eq!(second.spartan(), proof.spartan());

        // Tampering with the Spartan claim is rejected.
        let one = SpartanBitzField::from_with_cfg(1u64, &spartan_bitz_field_config());
        let reject = |mutate: &dyn Fn(&mut Proof)| {
            let mut tampered = proof.clone();
            mutate(&mut tampered);
            let mut verifier_transcript = Blake3Transcript::new();
            assert!(
                protocol::verify(
                    &mut verifier_transcript,
                    &prepared,
                    &hint.commitment,
                    &tampered
                )
                .is_err()
            );
        };
        reject(&|proof| {
            let value = &mut skip_proof_mut(proof).outer.tail.az_mle_claim;
            *value = spartan_bitz_field_config().add(value, &one);
        });
        reject(&|proof| {
            let value = &mut skip_proof_mut(proof).outer.skip.finite_q_evaluations[0];
            *value = spartan_bitz_field_config().add(value, &one);
        });
        reject(&|proof| {
            let value = &mut skip_proof_mut(proof).outer.skip.q_at_infinity;
            *value = spartan_bitz_field_config().add(value, &one);
        });
        reject(&|proof| {
            let value = &mut skip_proof_mut(proof).outer.tail.sumcheck.round_polynomials[0][0];
            *value = spartan_bitz_field_config().add(value, &one);
        });

        // The verifier must not accept a proof-selected skip width under the
        // fixed K=3 security profile, even though the generic PIOP supports
        // K=1 through K=4.
        for actual in [2, 4] {
            let mut wrong_k = proof.clone();
            skip_proof_mut(&mut wrong_k).outer.skip.skip_vars = actual;
            let mut verifier_transcript = Blake3Transcript::new();
            assert!(matches!(
                protocol::verify(&mut verifier_transcript, &prepared, &hint.commitment, &wrong_k),
                Err(ProtocolError::UnexpectedUnivariateSkipVariables {
                    expected: 3,
                    actual: rejected
                }) if rejected == actual
            ));
            assert_eq!(
                verifier_transcript.get_challenge::<u128>(),
                Blake3Transcript::new().get_challenge::<u128>(),
                "wrong K must be rejected before transcript mutation"
            );
        }

        // Strategy-2 profiles are rejected up front.
        assert!(matches!(
            PreparedRelation::<MulLayout<u32>>::new_with_profile::<Limber114>(layout),
            Err(ProtocolError::UnsupportedProfile)
                | Err(ProtocolError::Profile(
                    ProfileError::GrindingTooExpensive { .. }
                ))
        ));

        // λ = 128 is genuine on this path too: the initial boundary, every
        // PIOP draw, and the forest rounds all carry proof-of-work.
        let prepared128 =
            PreparedRelation::<MulLayout<u32>>::new_with_profile::<Lambda128>(layout).unwrap();
        assert_eq!(prepared128.security().initial_grinding_bits, 22);
        assert_eq!(prepared128.security().terminal_grinding_bits, 22);
        assert_eq!(prepared128.security().forest_round_grinding_bits, 2);
        assert_eq!(prepared128.security().ring_switch_grinding_bits, 1);
        assert_eq!(layout.opener_grinding_bits(prepared128.security()), 2);
        assert_ne!(
            (
                &prepared.ligerito_configuration().prover().queries,
                &prepared
                    .ligerito_configuration()
                    .prover()
                    .fold_grinding_bits,
            ),
            (
                &prepared128.ligerito_configuration().prover().queries,
                &prepared128
                    .ligerito_configuration()
                    .prover()
                    .fold_grinding_bits,
            ),
            "the Ligerito configuration must follow the profile target"
        );
        let hint128 = protocol::commit(&prepared128, witness.bitz_bit_rows()).unwrap();
        let mut prover_transcript = Blake3Transcript::new();
        let proof128 =
            protocol::prove(&mut prover_transcript, &prepared128, &witness, &hint128).unwrap();
        assert_eq!(prepared128.security().piop_round_grinding_bits, 22);
        assert_eq!(proof128.piop_nonces().len(), 43);
        assert!(!proof128.opening_grinding_nonces().is_empty());
        let mut verifier_transcript = Blake3Transcript::new();
        protocol::verify(
            &mut verifier_transcript,
            &prepared128,
            &hint128.commitment,
            &proof128,
        )
        .unwrap();
        let mut tampered = proof128.clone();
        tampered.prefix_mut().piop_nonces[3] ^= 1;
        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            protocol::verify(
                &mut verifier_transcript,
                &prepared128,
                &hint128.commitment,
                &tampered
            )
            .is_err()
        );
    }

    fn terminal_claim(
        point: &[Q100Element],
        scale: Q100Element,
        value: Q100Element,
    ) -> ScaledMleEvaluationClaim<SpartanBitzField> {
        let config = spartan_bitz_field_config();
        let point = point
            .iter()
            .map(|coordinate| SpartanBitzField::from_with_cfg(coordinate.canonical_u128(), &config))
            .collect::<Vec<_>>();
        ScaledMleEvaluationClaim::new(
            point.into_boxed_slice(),
            SpartanBitzField::from_with_cfg(scale.canonical_u128(), &config),
            SpartanBitzField::from_with_cfg(value.canonical_u128(), &config),
        )
    }

    struct PreparedClaim {
        chunks: ModQWeightChunks,
        col_weights: Vec<Q100Element>,
    }

    fn prepared_row_weight(prepared: &PreparedClaim, row: usize) -> u128 {
        let mut value = 0_u128;
        let mut shift = 0_usize;
        for chunk in prepared.chunks.chunks() {
            value |= chunk[row] << shift;
            shift += prepared.chunks.chunk_width();
        }
        value
    }

    fn bitify_test_claim(
        claim: &ScaledMleEvaluationClaim<SpartanBitzField>,
        layout: &MulLayout<u32>,
    ) -> Result<bitify::BitifiedClaim, ProtocolError> {
        bitify::bitify(
            claim,
            layout.bitz_params(),
            layout.gate_vars(),
            &layout.block_table(),
            protocol::ScaleSide::Rows,
            &field::FpCtx::from_prime_u128(FQ_MOD),
        )
    }

    fn prepare_test_claim(
        opening: &bitify::BitifiedClaim,
        layout: &MulLayout<u32>,
    ) -> Result<PreparedClaim, ProtocolError> {
        let arith = field::FpCtx::from_prime_u128(FQ_MOD);
        Ok(PreparedClaim {
            chunks: bitify::prepare_chunks(opening, &layout.block_table(), FQ_BITS, &arith)?,
            col_weights: bitify::column_weights(opening, &arith)?
                .into_iter()
                .map(Q100Element::from)
                .collect(),
        })
    }

    #[test]
    fn fast_bitify_field_helpers_match_reference_arithmetic() {
        let point = [
            Q100Element::from_u128(0),
            Q100Element::from_u128(1),
            Q100Element::from_u128(FQ_MOD - 1),
            Q100Element::from_u128(123_456_789),
        ];
        assert_eq!(
            bitify::eq_le_table_fq_fast_with(
                &point.map(|x| x.canonical_u128()),
                &field::FpCtx::from_prime_u128(FQ_MOD)
            )
            .unwrap(),
            eq_le_table_fq(&point)
                .into_iter()
                .map(|x| x.canonical_u128())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn bitification_is_the_adjoint_of_integer_reconstruction() {
        for width in [1, 8] {
            let witness = MulWitness::<u32>::from_inputs_with_word_bits(
                &[(0, u32::MAX), (1, 7), (u32::MAX, u32::MAX)],
                width,
            )
            .unwrap();
            let layout = witness.layout();
            let p = layout.bitz_params();

            // Non-Bit selector coordinates exercise all four assignment
            // blocks, rather than reducing this to a single block lookup.
            let gate_point = (0..layout.gate_vars())
                .map(|coordinate| Q100Element::from_u128((coordinate + 2) as u128))
                .collect::<Vec<_>>();
            assert_eq!(gate_point.len(), layout.gate_vars());
            let block_low = Q100Element::from_u128(7);
            let block_high = Q100Element::from_u128(11);
            let scale = Q100Element::from_u128(13);
            let one = Q100Element::from_u128(1);
            let one_minus_low =
                Q100Element::from_u128(fq_sub(one.canonical_u128(), block_low.canonical_u128()));
            let one_minus_high =
                Q100Element::from_u128(fq_sub(one.canonical_u128(), block_high.canonical_u128()));
            let factors = [
                one_minus_low * one_minus_high,
                block_low * one_minus_high,
                one_minus_low * block_high,
                block_low * block_high,
            ];
            let eq_gate = eq_le_table_fq(&gate_point);

            let assignment = witness.assignment();
            let mut assignment_evaluation = Q100Element::from_u128(0);
            for block in 0..4 {
                for gate in 0..layout.capacity() {
                    assignment_evaluation = assignment_evaluation
                        + factors[block]
                            * eq_gate[gate]
                            * Q100Element::from(u128::from(
                                assignment[block * layout.capacity() + gate],
                            ));
                }
            }
            let value = scale * assignment_evaluation;

            let mut point = gate_point.to_vec();
            point.extend([block_low, block_high]);
            let terminal = terminal_claim(&point, scale, value);
            let opening = bitify_test_claim(&terminal, layout).unwrap();
            let prepared = prepare_test_claim(&opening, layout).unwrap();

            let rows = witness.bitz_bit_rows();
            let mut read_off = Q100Element::from_u128(0);
            for b in 0..p.rows() {
                for c in 0..p.cols() {
                    let mut cell = 0_u128;
                    for j in 0..p.word_bits {
                        let packed_bit = b * p.word_bits + j;
                        let bit = (rows[c][packed_bit / u64::BITS as usize]
                            >> (packed_bit % u64::BITS as usize))
                            & 1;
                        cell |= u128::from(bit) << j;
                    }
                    read_off = read_off
                        + Q100Element::from(cell)
                            * Q100Element::from_u128(prepared_row_weight(&prepared, b))
                            * prepared.col_weights[c];
                }
            }
            assert_eq!(read_off.canonical_u128(), opening.claimed);
        }
    }

    #[test]
    fn zero_scale_keeps_a_nonzero_row_functional() {
        let layout = MulLayout::<u32>::new(3).unwrap();
        let mut point = (0..layout.gate_vars())
            .map(|coordinate| Q100Element::from_u128((coordinate + 2) as u128))
            .collect::<Vec<_>>();
        point.extend([Q100Element::from_u128(7), Q100Element::from_u128(11)]);
        let opening = bitify_test_claim(
            &terminal_claim(&point, Q100Element::from_u128(0), Q100Element::from_u128(0)),
            &layout,
        )
        .unwrap();
        let prepared = prepare_test_claim(&opening, &layout).unwrap();

        assert!((0..opening.params.rows()).any(|row| prepared_row_weight(&prepared, row) != 0));
        assert!(
            prepared
                .col_weights
                .iter()
                .all(|&weight| weight == Q100Element::from_u128(0))
        );
        assert_eq!(opening.claimed, 0);
    }

    #[test]
    fn constant_only_claim_uses_the_deterministic_dummy_functional() {
        let layout = MulLayout::<u32>::new(3).unwrap();
        let gate_point = (0..layout.gate_vars())
            .map(|coordinate| Q100Element::from_u128((coordinate + 2) as u128))
            .collect::<Vec<_>>();
        let scale = Q100Element::from_u128(13);
        let constant_evaluation = eq_le_table_fq(&gate_point)[0];
        let mut point = gate_point.to_vec();
        point.extend([Q100Element::from_u128(0), Q100Element::from_u128(0)]);

        let opening = bitify_test_claim(
            &terminal_claim(&point, scale, scale * constant_evaluation),
            &layout,
        )
        .unwrap();
        let prepared = prepare_test_claim(&opening, &layout).unwrap();
        assert_eq!(prepared_row_weight(&prepared, 0), 1);
        assert!((1..opening.params.rows()).all(|row| prepared_row_weight(&prepared, row) == 0));
        assert!(
            prepared
                .col_weights
                .iter()
                .all(|&weight| weight == Q100Element::from_u128(0))
        );
        assert_eq!(opening.claimed, 0);
    }

    #[test]
    fn malformed_claim_residue_is_rejected_before_canonical_projection() {
        let layout = MulLayout::<u32>::new(3).unwrap();
        let config = spartan_bitz_field_config();
        let malformed = field::FpCtx::from_prime_u128(u128::MAX - 158)
            .from_montgomery_integer(*config.modulus());
        let zero = SpartanBitzField::from_with_cfg(0_u128, &config);
        let mut point = vec![zero.clone(); layout.gate_vars() + 2];
        point[0] = malformed;
        let claim = ScaledMleEvaluationClaim::new(point.into_boxed_slice(), zero.clone(), zero);

        assert!(matches!(
            bitify_test_claim(&claim, &layout),
            Err(ProtocolError::ClaimFieldMismatch)
        ));
    }

    #[test]
    fn combined_protocol_uses_only_validator_gated_production_profiles() {
        let small = MulLayout::<u32>::new(3).unwrap();
        assert!(matches!(
            PreparedRelation::<MulLayout<u32>>::new(small),
            Err(ProtocolError::UnauditedBitzParameters)
        ));

        for width in [1, 8] {
            let production =
                MulLayout::<u32>::new_with_word_bits(1 << MIN_PRODUCTION_GATE_VARS, width).unwrap();
            PreparedRelation::<MulLayout<u32>>::new(production)
                .expect("the smallest validated profile is available");

            let largest = MulLayout::<u32>::new_with_word_bits(1 << 25, width).unwrap();
            let packed = packed_variables(&largest.bitz_params()).unwrap();
            for target in [100, 128] {
                crate::ligerito_flock::LigeritoSelection::ValidatedUdr
                    .resolve(packed, target)
                    .expect("the full advertised benchmark range has a validated config");
            }
            // The raw-performance table's Johnson opener at the 100-bit target.
            crate::ligerito_flock::LigeritoSelection::CustomJohnson {
                log_inv_rate: 1,
                initial_k: 4,
            }
            .resolve(packed, 100)
            .expect("the custom Johnson geometry validates over the benchmark range");
        }
    }

    #[test]
    #[ignore = "runs one production-sized W=8 Spartan/BitZ proof"]
    fn u32_mul_w8_proof_verifies() {
        let witness =
            MulWitness::<u32>::from_fn_with_word_bits(1 << MIN_PRODUCTION_GATE_VARS, 8, |index| {
                let value = (index as u32).wrapping_mul(0x9E37_79B9);
                (value, value.rotate_left(13) ^ 0xA5A5_5A5A)
            })
            .unwrap();
        let layout = *witness.layout();
        let prepared = PreparedRelation::<MulLayout<u32>>::new(layout).unwrap();
        let hint = protocol::commit(&prepared, witness.bitz_bit_rows()).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        let proof = protocol::prove(&mut prover_transcript, &prepared, &witness, &hint).unwrap();
        let mut verifier_transcript = Blake3Transcript::new();
        protocol::verify(
            &mut verifier_transcript,
            &prepared,
            &hint.commitment,
            &proof,
        )
        .unwrap();
    }
}

#[cfg(feature = "hybrid")]
pub(crate) mod hybrid;
