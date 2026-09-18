//! The u128 multiplication relation `x · y = z` (`z < 2^256`) over BitZ.
//!
//! The integer R1CS assignment has four blocks, `[e0 | x | y | z]`, with
//! `x, y < 2^128` and `z < 2^256`, so the assignment MLE needs no padding
//! and the block selector is two coordinates. The commitment stores the
//! 512 bits of one multiplication per gate (`x`, `y`, then `z`). This
//! module describes that relation to the shared protocol of
//! [`super::protocol`]: the Spartan PIOP borrows the native x/y/z segments.
//! Mixed arithmetic consumes the 128- and 256-bit values directly in the
//! first rounds and fuses projection into the required folded output.

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
    u128_mul::{
        U128_MUL_ASSIGNMENT_BLOCKS, U128_MUL_BIT_SLOTS, U128_MUL_OPERAND_BITS,
        U128_MUL_PRODUCT_BITS, U128_MUL_X_SLOT_START, U128_MUL_Y_SLOT_START, U128_MUL_Z_SLOT_START,
        u128_mul_constraint_matrices,
    },
};

const BINDING_DOMAIN: &[u8] = b"bitz/spartan-u128-bitz/assignment/v1-runtime";
const ASSIGNMENT_BLOCK_ORDER: &[u8] = b"e0|x|y|z";

static U128_MUL_DOMAINS: Domains = Domains {
    statement_tag: b"u128-mul-statement",
    prime_sampling: b"bitz/spartan-u128-mul/runtime-prime/v1",
    initial_grinding: b"bitz/spartan-u128-mul/grinding/initial/v1",
    piop_grinding: b"bitz/spartan-u128-mul/grinding/piop/v1",
    terminal_grinding: b"bitz/spartan-u128-mul/grinding/terminal/v1",
    bitified_claim: b"bitz/spartan-u128-bitz/bitified-claim/v1",
    opening: ModQOpeningKind::U128Mul,
    claim_tag: b"",
    reduction_grinding: b"",
    reduction_prime: b"",
    scopes: crate::protocol_scopes!("u128-spartan-bitz"),
};

/// The public statement facts the security-profile derivation consumes for
/// a u128 multiplication batch: per-row integer defects are below `2^258`.
pub fn u128_mul_instance_facts(params: &IntegerMatrixLayout, row_vars: usize) -> IopInstanceFacts {
    IopInstanceFacts {
        defect_log2_bound: 258,
        lift_arity_log2: params.row_vars as u32,
        opening_t: params.row_vars as u32,
        opening_word_bits: params.word_bits as u32,
        direct_opening: true,
        tau_arity: row_vars.max(1) as u32,
        piop_degree: 3,
        step50_magnitude_log2: 0,
    }
}

impl RelationSpec for MulLayout<u128> {
    type Coefficient = bool;
    type Witness = MulWitness<u128>;
    type Map = Self;

    fn domains(&self) -> &'static Domains {
        &U128_MUL_DOMAINS
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
        MulLayout::<u128>::gate_vars(self)
    }

    fn instance_facts(&self) -> IopInstanceFacts {
        let row_vars = self.multiplications().next_power_of_two().trailing_zeros() as usize;
        let mut facts = u128_mul_instance_facts(&self.bitz_params(), row_vars);
        facts.opening_word_bits = self.word_bits() as u32;
        facts.direct_opening = self.uses_direct_opening();
        facts
    }

    fn matrices(&self) -> Result<MatrixSource<bool>, ProtocolError> {
        MatrixSource::skeleton(u128_mul_constraint_matrices(self)?)
    }

    fn validate_geometry(&self) -> Result<(), ProtocolError> {
        self.validate_protocol_geometry()
    }

    /// Block order is 00=e0, 01=x, 10=y, 11=z with little-endian bit weights
    /// `2^0 … 2^255` on the committed slots.
    fn block_table(&self) -> BlockTable {
        let table = BlockTable::new(
            2,
            vec![
                None,
                Some(SlotRange {
                    bit_slot_start: U128_MUL_X_SLOT_START,
                    bit_count: U128_MUL_OPERAND_BITS,
                }),
                Some(SlotRange {
                    bit_slot_start: U128_MUL_Y_SLOT_START,
                    bit_count: U128_MUL_OPERAND_BITS,
                }),
                Some(SlotRange {
                    bit_slot_start: U128_MUL_Z_SLOT_START,
                    bit_count: U128_MUL_PRODUCT_BITS,
                }),
            ],
        )
        .expect("the u128 block table is complete");
        if self.uses_direct_opening() {
            table
        } else {
            table.with_word_packing(128, self.word_bits())
        }
    }

    fn kernel(&self) -> Kernel {
        Kernel::Plain
    }

    fn check_witness(&self, witness: &MulWitness<u128>) -> Result<(), ProtocolError> {
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
        hasher.usizes(&[
            self.multiplications(),
            self.capacity(),
            self.assignment_len(),
            MulLayout::<u128>::gate_vars(self),
            U128_MUL_ASSIGNMENT_BLOCKS,
            U128_MUL_OPERAND_BITS,
            U128_MUL_PRODUCT_BITS,
            U128_MUL_X_SLOT_START,
            U128_MUL_Y_SLOT_START,
            U128_MUL_Z_SLOT_START,
            U128_MUL_BIT_SLOTS,
            p.row_vars,
            p.col_vars,
            p.word_bits,
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
            MulLayout::<u128>::gate_vars(self),
            U128_MUL_ASSIGNMENT_BLOCKS,
            p.row_vars,
            p.col_vars,
            p.word_bits,
            U128_MUL_X_SLOT_START,
            U128_MUL_Y_SLOT_START,
            U128_MUL_Z_SLOT_START,
            U128_MUL_OPERAND_BITS,
            U128_MUL_PRODUCT_BITS,
            U128_MUL_BIT_SLOTS,
        ])?;
        // Mapping version one: little-endian bits, e0/x/y/z block order, and
        // nonzero scale normalized onto the folded row factors.
        hasher.bytes(&[1, 0, 1, 2, 3]);
        self.bind_packing(hasher)?;
        Ok(())
    }

    /// Borrow native operand/product segments for the mixed first rounds.
    fn piop_witness<'w>(
        &self,
        witness: &'w MulWitness<u128>,
        _config: &FieldConfig,
    ) -> Result<PiopWitness<'w>, ProtocolError> {
        Ok(PiopWitness::Mul128(witness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piop::spartan::protocol::{self, PreparedRelation};
    use crate::transcript::Blake3Transcript;
    use crate::transcript::traits::Transcript;

    fn witness(multiplications: usize, salt: u64) -> MulWitness<u128> {
        let mut state = 0x243f_6a88_85a3_08d3_u64 ^ salt;
        MulWitness::<u128>::from_fn(multiplications, |index| {
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let wide = |a: u64, b: u64| (u128::from(a) << 64) | u128::from(b);
            match index % 7 {
                0 => (u128::MAX, u128::MAX),
                1 => (wide(next(), next()), 0),
                2 => (1, wide(next(), next())),
                3 => (u128::from(next()), u128::from(next())),
                _ => (wide(next(), next()), wide(next(), next())),
            }
        })
        .unwrap()
    }

    #[test]
    fn u128_paper_path_roundtrips_and_is_deterministic() {
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let witness = witness(1 << 15, 0);
        let layout = *witness.layout();
        let prepared = PreparedRelation::<MulLayout<u128>>::new(layout).unwrap();
        assert_eq!(prepared.security().lambda, 100);
        assert_eq!(prepared.params().row_vars, 9 + 15 - 7);
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
    fn u128_paper_path_rejects_a_wrong_product() {
        let _env = crate::utils::QUAD_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let honest = witness(1 << 15, 1);
        let layout = *honest.layout();
        let prepared = PreparedRelation::<MulLayout<u128>>::new(layout).unwrap();

        // Flip one committed bit of the product's high half at gate 3; the
        // commitment no longer matches the honest assignment.
        let mut rows = honest.bitz_bit_rows();
        let (b, c) = layout.bitz_cell(U128_MUL_Z_SLOT_START + 200, 3).unwrap();
        rows[c][b / 64] ^= 1 << (b % 64);
        let hint = protocol::commit(&prepared, rows).unwrap();

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
            PreparedRelation::<MulLayout<u128>>::new(*witness.layout()),
            Err(ProtocolError::UnauditedBitzParameters)
        ));
    }
}
