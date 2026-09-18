//! The BabyBear multiplication relation `a · b = c + p · k` over BitZ.
//!
//! The integer R1CS assignment has five logical blocks,
//!
//! ```text
//! [constant block | a block | b block | c block | k block],
//! ```
//!
//! zero-padded to eight blocks for Spartan's assignment MLE, with the
//! BabyBear modulus `p` a public coefficient of matrix C. The commitment
//! stores the four 31-bit little-endian values of each multiplication in
//! 124 of 128 physical slots. This module describes that relation to the
//! shared protocol of [`super::protocol`] and keeps the PCS-only
//! terminal-opening benchmark path.

use crate::piop::spartan::baby_bear_mul::BabyBearMulLayout;
use crate::piop::spartan::protocol::PreparedRelation;
use crate::piop::spartan::protocol::ProtocolError;
#[cfg(feature = "bench-internals")]
use crate::piop::spartan::protocol::terminal::PreparedTerminalOpening;

use crate::piop::spartan::SpartanField as _;
use field::RingOps;
use std::borrow::Cow;

use flock_core::pcs::{commit::Commitment, ligerito::ProverConfig as LigProverConfig};

use crate::{
    ligerito::LOG_PACKING,
    ligerito_flock::{FlockCommitHint, LigeritoSelection, ModQOpeningKind},
    pcs::{FQ_MOD, IntegerMatrixLayout},
};

use super::{
    baby_bear_mul::{
        BABY_BEAR_MODULUS, BABY_BEAR_MUL_A_SLOT_START, BABY_BEAR_MUL_B_SLOT_START,
        BABY_BEAR_MUL_BIT_SLOTS, BABY_BEAR_MUL_C_SLOT_START, BABY_BEAR_MUL_K_SLOT_START,
        BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS, BABY_BEAR_MUL_VALUE_BITS, BabyBearMulCoefficient,
        BabyBearMulError, BabyBearMulWitness, baby_bear_mul_constraint_matrices,
    },
    profile::{IopInstanceFacts, IopSecurityParams},
    protocol::{
        self, BindingHasher, BlockTable, Domains, FieldConfig, Kernel, MatrixSource, PiopWitness,
        RelationSpec, SlotRange, checked_pow2, packed_variables,
    },
};

impl From<BabyBearMulError> for ProtocolError {
    fn from(error: BabyBearMulError) -> Self {
        Self::relation(error)
    }
}

const ASSIGNMENT_BINDING_DOMAIN: &[u8] = b"bitz/spartan-baby-bear-bitz/assignment/v1";
const BABY_BEAR_PAPER_BINDING_DOMAIN: &[u8] = b"bitz/spartan-baby-bear-bitz/assignment/v2-runtime";
const ASSIGNMENT_BLOCK_ORDER: &[u8] = b"e0|a|b|c|k|zero|zero|zero";

const LOGICAL_ASSIGNMENT_BLOCKS: usize = 5;
const PADDED_ASSIGNMENT_BLOCKS: usize = 8;

static BABY_BEAR_DOMAINS: Domains = Domains {
    statement_tag: b"baby-bear-paper-statement",
    prime_sampling: b"bitz/spartan-baby-bear-mul/runtime-prime/v1",
    initial_grinding: b"bitz/spartan-baby-bear-mul/grinding/initial/v1",
    piop_grinding: b"bitz/spartan-baby-bear-mul/grinding/piop/v1",
    terminal_grinding: b"bitz/spartan-baby-bear-mul/grinding/terminal/v1",
    bitified_claim: b"bitz/spartan-baby-bear-bitz/bitified-claim/v2",
    opening: ModQOpeningKind::BabyBearMul,
    claim_tag: b"",
    reduction_grinding: b"",
    reduction_prime: b"",
    scopes: crate::protocol_scopes!("baby-bear-spartan-bitz"),
};

/// The public statement facts the security-profile derivation consumes for
/// a BabyBear multiplication batch.
pub fn baby_bear_mul_instance_facts(
    params: &IntegerMatrixLayout,
    row_vars: usize,
) -> IopInstanceFacts {
    IopInstanceFacts {
        defect_log2_bound: 80,
        lift_arity_log2: params.row_vars as u32,
        opening_t: params.row_vars as u32,
        opening_word_bits: params.word_bits as u32,
        direct_opening: true,
        tau_arity: row_vars.max(1) as u32,
        piop_degree: 3,
        step50_magnitude_log2: 0,
    }
}

fn validate_layout_geometry(layout: &BabyBearMulLayout) -> Result<(), ProtocolError> {
    let params = layout.bitz_params();
    if params.word_bits != 1
        || params.row_vars < LOG_PACKING
        || params.col_vars > layout.gate_vars()
        || params.row_vars.saturating_add(params.word_bits) > 126
    {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    let total_vars = params
        .row_vars
        .checked_add(params.col_vars)
        .ok_or(ProtocolError::InvalidBitzParameters)?;
    if total_vars
        != layout
            .gate_vars()
            .checked_add(7)
            .ok_or(ProtocolError::InvalidBitzParameters)?
        || BABY_BEAR_MUL_BIT_SLOTS != 1_usize << 7
        || BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS != 4 * BABY_BEAR_MUL_VALUE_BITS
        || layout.assignment_len() != LOGICAL_ASSIGNMENT_BLOCKS * layout.capacity()
        || layout.padded_assignment_len() != PADDED_ASSIGNMENT_BLOCKS * layout.capacity()
    {
        return Err(ProtocolError::InvalidBitzParameters);
    }

    let row_count = checked_pow2(params.row_vars)?;
    let col_count = checked_pow2(params.col_vars)?;
    let cells = row_count
        .checked_mul(col_count)
        .ok_or(ProtocolError::InvalidBitzParameters)?;
    let expected_cells = BABY_BEAR_MUL_BIT_SLOTS
        .checked_mul(layout.capacity())
        .ok_or(ProtocolError::InvalidBitzParameters)?;
    if cells != expected_cells || packed_variables(&params)? != layout.gate_vars() {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    Ok(())
}

/// The fixed-q statement binding of the PCS-only terminal path: layout and
/// commitment alone (no profile parameters).
fn fixed_q_assignment_binding(
    layout: &BabyBearMulLayout,
    commitment: &Commitment,
) -> Result<[u8; 32], ProtocolError> {
    let p = layout.bitz_params();
    let mut hasher = BindingHasher::new();
    hasher
        .bytes(ASSIGNMENT_BINDING_DOMAIN)
        .bytes(ASSIGNMENT_BLOCK_ORDER)
        .bytes(&commitment.root);
    hasher.commitment_params(&commitment.params)?;
    hasher.u128_le(FQ_MOD).u64_le(BABY_BEAR_MODULUS);
    hasher.usizes(&[
        layout.multiplications(),
        layout.capacity(),
        layout.assignment_len(),
        layout.padded_assignment_len(),
        layout.gate_vars(),
        LOGICAL_ASSIGNMENT_BLOCKS,
        PADDED_ASSIGNMENT_BLOCKS,
        BABY_BEAR_MUL_VALUE_BITS,
        BABY_BEAR_MUL_A_SLOT_START,
        BABY_BEAR_MUL_B_SLOT_START,
        BABY_BEAR_MUL_C_SLOT_START,
        BABY_BEAR_MUL_K_SLOT_START,
        BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS,
        BABY_BEAR_MUL_BIT_SLOTS,
        p.row_vars,
        p.col_vars,
        p.word_bits,
    ])?;
    Ok(hasher.finalize())
}

impl RelationSpec for BabyBearMulLayout {
    type Coefficient = BabyBearMulCoefficient;
    type Witness = BabyBearMulWitness;
    type Map = circuit::linear_map::binary::RepeatedVirtualMap;

    fn domains(&self) -> &'static Domains {
        &BABY_BEAR_DOMAINS
    }

    fn committed_layout(&self) -> IntegerMatrixLayout {
        self.bitz_params()
    }

    fn gate_vars(&self) -> usize {
        BabyBearMulLayout::gate_vars(self)
    }

    fn instance_facts(&self) -> IopInstanceFacts {
        let row_vars = self.multiplications().next_power_of_two().trailing_zeros() as usize;
        baby_bear_mul_instance_facts(&self.bitz_params(), row_vars)
    }

    fn matrices(&self) -> Result<MatrixSource<BabyBearMulCoefficient>, ProtocolError> {
        MatrixSource::skeleton(baby_bear_mul_constraint_matrices(self)?)
    }

    fn validate_geometry(&self) -> Result<(), ProtocolError> {
        validate_layout_geometry(self)
    }

    /// Little-endian block-selector order is 000=e0, 001=a, 010=b, 011=c,
    /// 100=k, and 101..111 are public zero padding. Slots 124..128 are
    /// committed but carry public zero weight.
    fn block_table(&self) -> BlockTable {
        let block = |bit_slot_start: usize| {
            Some(SlotRange {
                bit_slot_start,
                bit_count: BABY_BEAR_MUL_VALUE_BITS,
            })
        };
        BlockTable::new(
            3,
            vec![
                None,
                block(BABY_BEAR_MUL_A_SLOT_START),
                block(BABY_BEAR_MUL_B_SLOT_START),
                block(BABY_BEAR_MUL_C_SLOT_START),
                block(BABY_BEAR_MUL_K_SLOT_START),
                None,
                None,
                None,
            ],
        )
        .expect("the BabyBear block table is complete")
    }

    fn kernel(&self) -> Kernel {
        Kernel::Plain
    }

    fn check_witness(&self, witness: &BabyBearMulWitness) -> Result<(), ProtocolError> {
        if witness.layout() != self {
            return Err(ProtocolError::RelationWitnessLayoutMismatch);
        }
        Ok(())
    }

    fn assignment_binding(
        &self,
        commitment: &Commitment,
        security: &IopSecurityParams,
        _ligerito: &LigProverConfig,
    ) -> Result<[u8; 32], ProtocolError> {
        let p = self.bitz_params();
        let mut hasher = BindingHasher::new();
        hasher
            .bytes(BABY_BEAR_PAPER_BINDING_DOMAIN)
            .bytes(ASSIGNMENT_BLOCK_ORDER)
            .bytes(&commitment.root);
        hasher.commitment_params(&commitment.params)?;
        hasher
            .u128_le(security.projection_min)
            .u128_le(security.projection_max);
        hasher.u32(security.lambda)?;
        hasher.usize(security.ligerito_target_bits)?;
        hasher.u32(security.initial_grinding_bits)?;
        hasher.u32(security.piop_round_grinding_bits)?;
        hasher.u32(security.terminal_grinding_bits)?;
        hasher.u32(security.forest_round_grinding_bits)?;
        hasher.u64_le(BABY_BEAR_MODULUS);
        hasher.usizes(&[
            self.multiplications(),
            self.capacity(),
            self.assignment_len(),
            self.padded_assignment_len(),
            BabyBearMulLayout::gate_vars(self),
            LOGICAL_ASSIGNMENT_BLOCKS,
            PADDED_ASSIGNMENT_BLOCKS,
            BABY_BEAR_MUL_VALUE_BITS,
            BABY_BEAR_MUL_A_SLOT_START,
            BABY_BEAR_MUL_B_SLOT_START,
            BABY_BEAR_MUL_C_SLOT_START,
            BABY_BEAR_MUL_K_SLOT_START,
            BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS,
            BABY_BEAR_MUL_BIT_SLOTS,
            p.row_vars,
            p.col_vars,
            p.word_bits,
        ])?;
        Ok(hasher.finalize())
    }

    fn hash_bridge_constants(&self, hasher: &mut BindingHasher) -> Result<(), ProtocolError> {
        let p = self.bitz_params();
        hasher.u64_le(BABY_BEAR_MODULUS);
        hasher.usizes(&[
            self.multiplications(),
            self.capacity(),
            self.assignment_len(),
            self.padded_assignment_len(),
            BabyBearMulLayout::gate_vars(self),
            LOGICAL_ASSIGNMENT_BLOCKS,
            PADDED_ASSIGNMENT_BLOCKS,
            p.row_vars,
            p.col_vars,
            p.word_bits,
            BABY_BEAR_MUL_A_SLOT_START,
            BABY_BEAR_MUL_B_SLOT_START,
            BABY_BEAR_MUL_C_SLOT_START,
            BABY_BEAR_MUL_K_SLOT_START,
            BABY_BEAR_MUL_VALUE_BITS,
            BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS,
            BABY_BEAR_MUL_BIT_SLOTS,
        ])?;
        // Mapping version two: little-endian bits, e0/a/b/c/k/zero/zero/zero
        // assignment order, raw k reconstruction, and nonzero scale normalized
        // onto the folded row factors.
        hasher.bytes(&[2, 0, 1, 2, 3, 4, 5, 6, 7]);
        Ok(())
    }

    /// Borrows the operand blocks and logical assignment; only the exact
    /// `c + p·k` products are materialized.
    fn piop_witness<'w>(
        &self,
        witness: &'w BabyBearMulWitness,
        _config: &FieldConfig,
    ) -> Result<PiopWitness<'w>, ProtocolError> {
        let capacity = self.capacity();
        let multiplications = self.multiplications();
        let product_len = multiplications.next_power_of_two();
        let assignment = witness.w();
        let cz: Vec<u64> = (0..product_len)
            .map(|index| {
                if index < multiplications {
                    witness.c_values()[index] + BABY_BEAR_MODULUS * witness.k_values()[index]
                } else {
                    0
                }
            })
            .collect();
        Ok(PiopWitness::Native {
            az: Cow::Borrowed(&assignment[capacity..capacity + product_len]),
            bz: Cow::Borrowed(&assignment[2 * capacity..2 * capacity + product_len]),
            cz: Cow::Owned(cz),
            assignment,
            constant_prefix: None,
        })
    }
}

/// Commits compact bit rows under the Johnson opener at the 100-bit target
/// (the PCS-only comparison's preflight commitment).
pub fn commit_baby_bear_mul_witness(
    layout: &BabyBearMulLayout,
    rows: Vec<Vec<u64>>,
) -> Result<FlockCommitHint, ProtocolError> {
    commit_baby_bear_mul_witness_with_ligerito(layout, rows, LigeritoSelection::JOHNSON)
}

/// Commits compact bit rows under an explicit opener at the 100-bit target.
pub fn commit_baby_bear_mul_witness_with_ligerito(
    layout: &BabyBearMulLayout,
    rows: Vec<Vec<u64>>,
    selection: LigeritoSelection,
) -> Result<FlockCommitHint, ProtocolError> {
    let prepared = PreparedRelation::<BabyBearMulLayout>::new_with_profile_and_ligerito::<
        super::profile::Lambda100,
    >(*layout, selection)?;
    protocol::commit(&prepared, rows)
}

/// Prepares the fixed-q, PCS-only terminal-opening context under the Johnson
/// opener.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub fn prepare_baby_bear_terminal_bitz_opening(
    matrices: &super::PreparedConstraintMatrices<protocol::SpartanBitzField, BabyBearMulCoefficient>,
    layout: &BabyBearMulLayout,
    commitment: &Commitment,
) -> Result<PreparedTerminalOpening<BabyBearMulLayout>, ProtocolError> {
    prepare_baby_bear_terminal_bitz_opening_with_ligerito(
        matrices,
        layout,
        commitment,
        LigeritoSelection::JOHNSON,
    )
}

/// Prepares the fixed-q, PCS-only terminal-opening context: relation
/// projection, opener resolution and statement binding are excluded from
/// all trial timers. `matrices` must be the relation at `q = 2^100 - 15`.
#[cfg(feature = "bench-internals")]
pub fn prepare_baby_bear_terminal_bitz_opening_with_ligerito(
    matrices: &super::PreparedConstraintMatrices<protocol::SpartanBitzField, BabyBearMulCoefficient>,
    layout: &BabyBearMulLayout,
    commitment: &Commitment,
    selection: LigeritoSelection,
) -> Result<PreparedTerminalOpening<BabyBearMulLayout>, ProtocolError> {
    use super::SpartanField;
    let expected = protocol::SpartanBitzField::canonical_modulus_encoding(
        &super::bitz::spartan_bitz_field_config(),
    );
    if matrices.field_modulus_encoding() != expected {
        return Err(ProtocolError::UnsupportedFieldModulus);
    }
    if matrices.matrices().row_count() != layout.multiplications()
        || matrices.matrices().column_count() != layout.assignment_len()
    {
        return Err(ProtocolError::RelationWitnessLayoutMismatch);
    }
    let prepared = PreparedRelation::<BabyBearMulLayout>::new_with_profile_and_ligerito::<
        super::profile::Lambda100,
    >(*layout, selection)?;
    protocol::terminal::prepare(
        &prepared,
        commitment,
        b"baby-bear-terminal/early-ood/v2",
        protocol::terminal::StatementPayload::AssignmentBinding {
            relation_tag: b"terminal-relation",
        },
        protocol::terminal::Binding::Custom(fixed_q_assignment_binding),
    )
}

#[cfg(test)]
mod tests {
    use crate::transcript::traits::Transcript;

    use super::*;
    use crate::{
        pcs::{FQ_BITS, ModQWeightChunks, Q100Element, eq_le_table_fq, fq_sub},
        piop::spartan::{
            baby_bear_mul::sample_baby_bear_operand_with,
            bitz::{SpartanBitzField, spartan_bitz_field_config},
            matrix::ScaledMleEvaluationClaim,
            profile::Lambda128,
            protocol::bitify,
        },
        transcript::Blake3Transcript,
    };

    fn witness(multiplications: usize) -> BabyBearMulWitness {
        let mut state = 0x4242_4242_u32;
        BabyBearMulWitness::from_fn(multiplications, |_| {
            let mut next = || {
                state = state.wrapping_mul(0x9e37_79b9).wrapping_add(1);
                state
            };
            (
                sample_baby_bear_operand_with(&mut next),
                sample_baby_bear_operand_with(&mut next),
            )
        })
        .unwrap()
    }

    #[test]
    fn baby_bear_paper_path_roundtrips_at_both_targets() {
        // Hold the shared env lock so tests that toggle transcript-shaping
        // `BITZ_*` variables cannot flip them between our prove and verify.
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let witness = witness(1 << 15);
        let layout = *witness.layout();
        // λ = 100: no grinding anywhere, one chunk by construction.
        let prepared = PreparedRelation::<BabyBearMulLayout>::new(layout).unwrap();
        let hint = protocol::commit(&prepared, witness.bitz_bit_rows()).unwrap();
        assert_eq!(prepared.security().lambda, 100);
        assert_eq!(prepared.security().ligerito_target_bits, 100);
        assert_eq!(prepared.security().initial_grinding_bits, 0);
        let mut prover_transcript = Blake3Transcript::new();
        let proof = protocol::prove(&mut prover_transcript, &prepared, &witness, &hint).unwrap();
        assert!(proof.piop_nonces().is_empty());
        assert!(proof.opening_grinding_nonces().is_empty());
        let mut verifier_transcript = Blake3Transcript::new();
        protocol::verify(
            &mut verifier_transcript,
            &prepared,
            &hint.commitment,
            &proof,
        )
        .unwrap();

        // Replaying the same statement is deterministic.
        {
            let mut second_transcript = Blake3Transcript::new();
            let second =
                protocol::prove(&mut second_transcript, &prepared, &witness, &hint).unwrap();
            assert_eq!(second.bitz().to_bytes(), proof.bitz().to_bytes());
            assert_eq!(second.spartan(), proof.spartan());
            assert_eq!(
                second_transcript.state_digest(),
                prover_transcript.state_digest()
            );
        }

        // λ = 128: initial + per-draw PIOP + forest boundaries all armed.
        let prepared128 =
            PreparedRelation::<BabyBearMulLayout>::new_with_profile::<Lambda128>(layout).unwrap();
        assert!(prepared128.security().initial_grinding_bits > 0);
        assert!(prepared128.security().piop_round_grinding_bits > 0);
        assert_eq!(prepared128.security().forest_round_grinding_bits, 2);
        let hint128 = protocol::commit(&prepared128, witness.bitz_bit_rows()).unwrap();
        let mut prover_transcript = Blake3Transcript::new();
        let proof128 =
            protocol::prove(&mut prover_transcript, &prepared128, &witness, &hint128).unwrap();
        assert_eq!(proof128.piop_nonces().len(), 2 * 15 + 1 + 15 + 3);
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

    fn row_weight(chunks: &ModQWeightChunks, row: usize) -> u128 {
        let mut value = 0_u128;
        let mut shift = 0_usize;
        for chunk in chunks.chunks() {
            value |= chunk[row] << shift;
            shift += chunks.chunk_width();
        }
        value
    }

    #[test]
    fn bitification_is_the_adjoint_of_five_block_integer_reconstruction() {
        let witness = BabyBearMulWitness::from_inputs(&[
            (0, 2_013_265_920),
            (1, 7),
            (2_013_265_920, 2_013_265_920),
        ])
        .unwrap();
        let layout = witness.layout();
        let p = layout.bitz_params();
        let arith = field::FpCtx::from_prime_u128(FQ_MOD);

        // Non-Bit selector coordinates exercise all eight assignment
        // blocks (the padding blocks are identically zero).
        let gate_point = (0..layout.gate_vars())
            .map(|coordinate| Q100Element::from_u128((coordinate + 2) as u128))
            .collect::<Vec<_>>();
        let selector = [
            Q100Element::from_u128(7),
            Q100Element::from_u128(11),
            Q100Element::from_u128(5),
        ];
        let scale = Q100Element::from_u128(13);
        let one = Q100Element::from_u128(1);
        let factor = |code: usize| {
            selector.iter().enumerate().fold(one, |acc, (bit, &s)| {
                acc * if (code >> bit) & 1 == 1 {
                    s
                } else {
                    Q100Element::from_u128(fq_sub(one.canonical_u128(), s.canonical_u128()))
                }
            })
        };
        let eq_gate = eq_le_table_fq(&gate_point);

        let assignment = witness.w();
        let mut assignment_evaluation = Q100Element::from_u128(0);
        for block in 0..5 {
            for gate in 0..layout.capacity() {
                assignment_evaluation = assignment_evaluation
                    + factor(block)
                        * eq_gate[gate]
                        * Q100Element::from(u128::from(
                            assignment[block * layout.capacity() + gate],
                        ));
            }
        }
        let value = scale * assignment_evaluation;

        let mut point = gate_point.to_vec();
        point.extend(selector);
        let terminal = terminal_claim(&point, scale, value);
        let table = layout.block_table();
        let opening = bitify::bitify(
            &terminal,
            p,
            layout.gate_vars(),
            &table,
            protocol::ScaleSide::Rows,
            &arith,
        )
        .unwrap();
        let chunks = bitify::prepare_chunks(&opening, &table, FQ_BITS, &arith).unwrap();
        let col_weights = bitify::column_weights(&opening, &arith).unwrap();

        let rows = witness.bitz_bit_rows();
        let mut read_off = Q100Element::from_u128(0);
        for b in 0..p.rows() {
            for c in 0..p.cols() {
                let bit = (rows[c][b / u64::BITS as usize] >> (b % u64::BITS as usize)) & 1;
                read_off = read_off
                    + Q100Element::from(u128::from(bit))
                        * Q100Element::from_u128(row_weight(&chunks, b))
                        * Q100Element::from_u128(col_weights[c]);
            }
        }
        assert_eq!(read_off.canonical_u128(), opening.claimed);
        // The four unused slots carry zero weight.
        for slot in BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS..BABY_BEAR_MUL_BIT_SLOTS {
            for gate_high in 0..(p.rows() / BABY_BEAR_MUL_BIT_SLOTS) {
                assert_eq!(
                    row_weight(
                        &chunks,
                        slot * (p.rows() / BABY_BEAR_MUL_BIT_SLOTS) + gate_high
                    ),
                    0
                );
            }
        }
    }

    #[test]
    fn combined_protocol_uses_only_validator_gated_production_profiles() {
        let small = BabyBearMulLayout::new(3).unwrap();
        assert!(matches!(
            PreparedRelation::<BabyBearMulLayout>::new(small),
            Err(ProtocolError::UnauditedBitzParameters)
        ));
        let production = BabyBearMulLayout::new(1 << protocol::MIN_PRODUCTION_GATE_VARS).unwrap();
        PreparedRelation::<BabyBearMulLayout>::new(production)
            .expect("the smallest validated profile is available");
    }
}
