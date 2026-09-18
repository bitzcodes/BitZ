//! The u64 multiplication relation `x · y = z_lo + 2^64 · z_hi` over BitZ.
//!
//! The integer R1CS assignment has five logical blocks,
//!
//! ```text
//! [constant block | x block | y block | z_lo block | z_hi block],
//! ```
//!
//! and is zero-padded to eight blocks for Spartan's assignment MLE. The
//! commitment stores four 64-bit little-endian values per multiplication in
//! 256 physical slots. This module describes that relation to the shared
//! protocol of [`super::protocol`]: the assignment, native operands and
//! split-limb products remain borrowed. Mixed first-round kernels fuse
//! projection with accumulation and folding.

use crate::piop::spartan::mul::{MulLayout, MulWitness};
use crate::piop::spartan::protocol::ProtocolError;

use field::RingOps;
use flock_core::pcs::{commit::Commitment, ligerito::ProverConfig as LigProverConfig};

use crate::{ligerito_flock::ModQOpeningKind, pcs::IntegerMatrixLayout};

use super::{
    profile::{IopInstanceFacts, IopSecurityParams},
    protocol::{
        BindingHasher, BlockTable, Domains, FieldConfig, Kernel, MatrixSource, PiopWitness,
        RelationSpec, SlotRange,
    },
    u64_mul::{
        U64_MUL_BIT_SLOTS, U64_MUL_LIMB_BASE, U64_MUL_LOGICAL_ASSIGNMENT_BLOCKS,
        U64_MUL_PADDED_ASSIGNMENT_BLOCKS, U64_MUL_VALUE_BITS, U64_MUL_X_SLOT_START,
        U64_MUL_Y_SLOT_START, U64_MUL_Z_HI_SLOT_START, U64_MUL_Z_LO_SLOT_START, U64MulCoefficient,
        u64_mul_constraint_matrices,
    },
};

const BINDING_DOMAIN: &[u8] = b"bitz/spartan-u64-bitz/assignment/v1-runtime";
const ASSIGNMENT_BLOCK_ORDER: &[u8] = b"e0|x|y|zlo|zhi|zero|zero|zero";

static U64_MUL_DOMAINS: Domains = Domains {
    statement_tag: b"u64-mul-statement",
    prime_sampling: b"bitz/spartan-u64-mul/runtime-prime/v1",
    initial_grinding: b"bitz/spartan-u64-mul/grinding/initial/v1",
    piop_grinding: b"bitz/spartan-u64-mul/grinding/piop/v1",
    terminal_grinding: b"bitz/spartan-u64-mul/grinding/terminal/v1",
    bitified_claim: b"bitz/spartan-u64-bitz/bitified-claim/v1",
    opening: ModQOpeningKind::U64Mul,
    claim_tag: b"",
    reduction_grinding: b"",
    reduction_prime: b"",
    scopes: crate::protocol_scopes!("u64-spartan-bitz"),
};

/// The public statement facts the security-profile derivation consumes for
/// a u64 multiplication batch: per-row integer defects are below `2^130`.
pub fn u64_mul_instance_facts(params: &IntegerMatrixLayout, row_vars: usize) -> IopInstanceFacts {
    IopInstanceFacts {
        defect_log2_bound: 130,
        lift_arity_log2: params.row_vars as u32,
        opening_t: params.row_vars as u32,
        opening_word_bits: params.word_bits as u32,
        direct_opening: true,
        tau_arity: row_vars.max(1) as u32,
        piop_degree: 3,
        step50_magnitude_log2: 0,
    }
}

impl RelationSpec for MulLayout<u64> {
    type Coefficient = U64MulCoefficient;
    type Witness = MulWitness<u64>;
    type Map = Self;

    fn domains(&self) -> &'static Domains {
        &U64_MUL_DOMAINS
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
    fn claim_digest(
        &self,
        frame: super::protocol::ClaimFrame<'_>,
    ) -> Result<[u8; 32], ProtocolError> {
        self.packed_claim_digest(frame)
    }

    fn gate_vars(&self) -> usize {
        MulLayout::<u64>::gate_vars(self)
    }

    fn instance_facts(&self) -> IopInstanceFacts {
        let row_vars = self.multiplications().next_power_of_two().trailing_zeros() as usize;
        let mut facts = u64_mul_instance_facts(&self.bitz_params(), row_vars);
        facts.opening_word_bits = self.word_bits() as u32;
        facts.direct_opening = self.uses_direct_opening();
        facts
    }

    fn matrices(&self) -> Result<MatrixSource<U64MulCoefficient>, ProtocolError> {
        MatrixSource::skeleton(u64_mul_constraint_matrices(self)?)
    }

    fn validate_geometry(&self) -> Result<(), ProtocolError> {
        self.validate_protocol_geometry()
    }

    /// Little-endian block-selector order is 000=e0, 001=x, 010=y,
    /// 011=z_lo, 100=z_hi, and 101..111 are public zero padding.
    fn block_table(&self) -> BlockTable {
        let block = |bit_slot_start: usize| {
            Some(SlotRange {
                bit_slot_start,
                bit_count: U64_MUL_VALUE_BITS,
            })
        };
        let table = BlockTable::new(
            3,
            vec![
                None,
                block(U64_MUL_X_SLOT_START),
                block(U64_MUL_Y_SLOT_START),
                block(U64_MUL_Z_LO_SLOT_START),
                block(U64_MUL_Z_HI_SLOT_START),
                None,
                None,
                None,
            ],
        )
        .expect("the u64 block table is complete");
        if self.uses_direct_opening() {
            table
        } else {
            table.with_word_packing(64, self.word_bits())
        }
    }

    fn kernel(&self) -> Kernel {
        Kernel::Plain
    }

    fn check_witness(&self, witness: &MulWitness<u64>) -> Result<(), ProtocolError> {
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
            .bytes(BINDING_DOMAIN)
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
        hasher.bytes(&U64_MUL_LIMB_BASE.to_le_bytes());
        hasher.usizes(&[
            self.multiplications(),
            self.capacity(),
            self.assignment_len(),
            self.padded_assignment_len(),
            MulLayout::<u64>::gate_vars(self),
            U64_MUL_LOGICAL_ASSIGNMENT_BLOCKS,
            U64_MUL_PADDED_ASSIGNMENT_BLOCKS,
            U64_MUL_VALUE_BITS,
            U64_MUL_X_SLOT_START,
            U64_MUL_Y_SLOT_START,
            U64_MUL_Z_LO_SLOT_START,
            U64_MUL_Z_HI_SLOT_START,
            U64_MUL_BIT_SLOTS,
            p.row_vars,
            p.col_vars,
            p.word_bits,
        ])?;
        self.bind_packing(&mut hasher)?;
        Ok(hasher.finalize())
    }

    fn hash_bridge_constants(&self, hasher: &mut BindingHasher) -> Result<(), ProtocolError> {
        let p = self.bitz_params();
        hasher.bytes(&U64_MUL_LIMB_BASE.to_le_bytes());
        hasher.usizes(&[
            self.multiplications(),
            self.capacity(),
            self.assignment_len(),
            self.padded_assignment_len(),
            MulLayout::<u64>::gate_vars(self),
            U64_MUL_LOGICAL_ASSIGNMENT_BLOCKS,
            U64_MUL_PADDED_ASSIGNMENT_BLOCKS,
            p.row_vars,
            p.col_vars,
            p.word_bits,
            U64_MUL_X_SLOT_START,
            U64_MUL_Y_SLOT_START,
            U64_MUL_Z_LO_SLOT_START,
            U64_MUL_Z_HI_SLOT_START,
            U64_MUL_VALUE_BITS,
            U64_MUL_BIT_SLOTS,
        ])?;
        // Mapping version one: little-endian bits, e0/x/y/zlo/zhi/zero/zero/zero
        // assignment order, raw z_hi reconstruction, and nonzero scale
        // normalized onto the folded row factors.
        hasher.bytes(&[1, 0, 1, 2, 3, 4, 5, 6, 7]);
        self.bind_packing(hasher)?;
        Ok(())
    }

    /// Borrow native assignment and split products without projection tables.
    fn piop_witness<'w>(
        &self,
        witness: &'w MulWitness<u64>,
        _config: &FieldConfig,
    ) -> Result<PiopWitness<'w>, ProtocolError> {
        Ok(PiopWitness::Mul64(witness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piop::spartan::protocol::{self, PreparedRelation};
    use crate::transcript::Blake3Transcript;
    use crate::transcript::traits::Transcript;

    fn witness(multiplications: usize, salt: u64) -> MulWitness<u64> {
        let mut state = 0x243f_6a88_85a3_08d3_u64 ^ salt;
        MulWitness::<u64>::from_fn(multiplications, |index| {
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            match index % 7 {
                0 => (u64::MAX, u64::MAX),
                1 => (next(), 0),
                2 => (1, next()),
                _ => (next(), next()),
            }
        })
        .unwrap()
    }

    #[test]
    fn u64_paper_path_roundtrips_and_is_deterministic() {
        // Hold the shared env lock so tests that toggle transcript-shaping
        // `BITZ_*` variables cannot flip them between our prove and verify.
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let witness = witness(1 << 15, 0);
        let layout = *witness.layout();
        let prepared = PreparedRelation::<MulLayout<u64>>::new(layout).unwrap();
        assert_eq!(prepared.security().lambda, 100);
        assert_eq!(prepared.params().row_vars, 8 + 15 - 7);
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
        assert!(proof.size_bytes(prepared.security()) > 0);

        let mut second_transcript = Blake3Transcript::new();
        let second = protocol::prove(&mut second_transcript, &prepared, &witness, &hint).unwrap();
        assert_eq!(proof.bitz().to_bytes(), second.bitz().to_bytes());
        assert_eq!(proof.piop_nonces(), second.piop_nonces());
    }

    #[test]
    fn u64_paper_path_rejects_a_wrong_product() {
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let honest = witness(1 << 15, 1);
        let layout = *honest.layout();
        let prepared = PreparedRelation::<MulLayout<u64>>::new(layout).unwrap();

        // Flip the committed z_hi bit of gate 3: the committed bits and the
        // projected products no longer satisfy x·y = z_lo + 2^64·z_hi.
        let mut rows = honest.bitz_bit_rows();
        let (b, c) = layout.bitz_cell(U64_MUL_Z_HI_SLOT_START, 3).unwrap();
        rows[c][b / 64] ^= 1 << (b % 64);
        let hint = protocol::commit(&prepared, rows).unwrap();

        // An honest prover with a mismatching commitment must not produce a
        // verifying proof.
        let mut prover_transcript = Blake3Transcript::new();
        let outcome = protocol::prove(&mut prover_transcript, &prepared, &honest, &hint);
        if let Ok(proof) = outcome {
            let mut verifier_transcript = Blake3Transcript::new();
            assert!(
                protocol::verify(
                    &mut verifier_transcript,
                    &prepared,
                    &hint.commitment,
                    &proof
                )
                .is_err()
            );
        }
    }

    #[test]
    fn small_layouts_are_rejected_by_the_production_api() {
        let witness = witness(1 << 10, 2);
        assert!(matches!(
            PreparedRelation::<MulLayout<u64>>::new(*witness.layout()),
            Err(ProtocolError::UnauditedBitzParameters)
        ));
    }
}
