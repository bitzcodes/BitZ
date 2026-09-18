//! Bitification: the adjoint of a relation's public bit-reconstruction map,
//! applied to Spartan's terminal scaled assignment-MLE claim.
//!
//! Every block-structured relation lays its integer assignment out as
//! `2^selector_vars` blocks of `capacity` gates, block 0 being the constant
//! block `[1, 0, …]` and the others integer values reconstructed from
//! contiguous bit-slot ranges of the committed tensor. Spartan's terminal
//! point is low-coordinate-first: its last `selector_vars` coordinates select
//! the block, the preceding ones select a gate. BitZ places the low gate
//! coordinates on the clear column axis; the high gate coordinates and the
//! word slots form the folded row axis.

use crate::piop::spartan::SpartanField as _;

use crate::{
    pcs::{IntegerMatrixLayout, ModQWeightChunks, mod_q_num_chunks},
    utils::{cfg_chunks_mut, cfg_iter, cfg_iter_mut},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    ProtocolError, SpartanBitzField, SpartanField, binding::BindingHasher, checked_pow2,
    matrix::ScaledMleEvaluationClaim,
};

/// A contiguous range of bit slots of one gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlotRange {
    pub bit_slot_start: usize,
    pub bit_count: usize,
}

/// How assignment blocks map to bit slots of the committed tensor.
///
/// `blocks[code]` describes the block selected by the little-endian block
/// code `code` (the first selector coordinate is the low bit): `None` for
/// the constant block (code 0) and for public zero padding blocks, `Some`
/// for a block whose values are reconstructed from that slot range with
/// little-endian bit weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockTable {
    selector_vars: usize,
    blocks: Vec<Option<SlotRange>>,
    packing: Option<(usize, usize)>,
}

impl BlockTable {
    pub fn new(
        selector_vars: usize,
        blocks: Vec<Option<SlotRange>>,
    ) -> Result<Self, ProtocolError> {
        if blocks.len() != checked_pow2(selector_vars)? || blocks[0].is_some() {
            return Err(ProtocolError::InvalidBlockTable);
        }
        Ok(Self {
            selector_vars,
            blocks,
            packing: None,
        })
    }

    /// Native limbs are split into logical W-bit cells with a padded stride.
    pub(crate) fn with_word_packing(mut self, limb_bits: usize, value_bits: usize) -> Self {
        self.packing = Some((limb_bits, value_bits));
        self
    }

    pub fn selector_vars(&self) -> usize {
        self.selector_vars
    }

    /// The variable blocks in block-code order.
    pub fn variable_blocks(&self) -> impl Iterator<Item = (usize, SlotRange)> + '_ {
        self.blocks
            .iter()
            .enumerate()
            .filter_map(|(code, range)| range.map(|range| (code, range)))
    }
}

/// Where a nonzero Spartan scale goes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleSide {
    /// Onto the folded row factors; the clear column table stays the raw
    /// equality table (the integer-multiplication relations).
    Rows,
    /// Onto the clear column table; the row factors stay unscaled (the
    /// CM-AND relation).
    Columns,
}

/// The factorized claim bound between Spartan and BitZ: the row functional is
/// one factor per variable block times the slot weights, the column
/// functional is the equality table of the low gate coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BitifiedClaim {
    pub params: IntegerMatrixLayout,
    pub gate_point: Box<[u128]>,
    pub rows: BitifiedRows,
    pub col_scale: u128,
    pub claimed: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BitifiedRows {
    /// One factor per variable block, in block-code order.
    Structured(Vec<u128>),
    /// The variable part vanishes: a deterministic dummy row with an all-zero
    /// clear read-off.
    ConstantDummy,
}

impl BitifiedClaim {
    pub fn gate_low(&self) -> &[u128] {
        &self.gate_point[..self.params.col_vars]
    }

    pub fn gate_high(&self) -> &[u128] {
        &self.gate_point[self.params.col_vars..]
    }
}

/// Applies the adjoint of the block reconstruction map to a terminal claim
/// over the runtime prime `q`.
pub fn bitify(
    claim: &ScaledMleEvaluationClaim<SpartanBitzField>,
    params: IntegerMatrixLayout,
    gate_vars: usize,
    table: &BlockTable,
    scale_side: ScaleSide,
    arith: &field::FpCtx<2>,
) -> Result<BitifiedClaim, ProtocolError> {
    let q = arith.modulus_u128();
    if claim.point().len() != gate_vars.saturating_add(table.selector_vars)
        || params.col_vars > gate_vars
    {
        return Err(ProtocolError::InvalidClaimPoint);
    }

    let modulus_encoding = SpartanBitzField::canonical_modulus_encoding(arith);
    let project = |value: &SpartanBitzField| -> Result<u128, ProtocolError> {
        value
            .validate_element(&modulus_encoding)
            .map_err(|_| ProtocolError::ClaimFieldMismatch)?;
        Ok(u128::from(arith.to_integer(value)))
    };
    let sub = |left: u128, right: u128| -> u128 { arith.sub_u128(left, right) };
    let mul = |left: u128, right: u128| arith.mul_u128(left, right);

    let gate_point = claim.point()[..gate_vars]
        .iter()
        .map(|value| project(value))
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    let selector = claim.point()[gate_vars..]
        .iter()
        .map(|value| project(value))
        .collect::<Result<Vec<_>, _>>()?;
    let one = 1;
    let one_minus: Vec<u128> = selector
        .iter()
        .map(|coordinate| sub(one, *coordinate))
        .collect();

    // The block factor is the equality-table entry of the selector point at
    // the block code: coordinate `i` contributes `s_i` when bit `i` of the
    // code is set and `1 - s_i` otherwise.
    let block_factor = |code: usize| -> u128 {
        selector
            .iter()
            .zip(&one_minus)
            .enumerate()
            .fold(one, |acc, (bit, (set, clear))| {
                mul(acc, if (code >> bit) & 1 == 1 { *set } else { *clear })
            })
    };

    let scale = project(claim.scale())?;
    let value = project(claim.value())?;
    let constant_evaluation = gate_point
        .iter()
        .copied()
        .fold(block_factor(0), |acc, coordinate| {
            mul(acc, sub(one, coordinate))
        });
    let adjusted_claim = sub(value, arith.mul_u128(scale, constant_evaluation));

    let factors: Vec<u128> = table
        .variable_blocks()
        .map(|(code, _)| block_factor(code))
        .collect();

    // At a block point where the variable part vanishes the BitZ protocol
    // still needs a nonempty row functional: a deterministic dummy row with
    // an all-zero clear read-off.
    if factors.iter().all(|factor| *factor == 0) {
        if adjusted_claim != 0 {
            return Err(ProtocolError::InvalidConstantOnlyClaim);
        }
        return Ok(BitifiedClaim {
            params,
            gate_point,
            rows: BitifiedRows::ConstantDummy,
            col_scale: 0,
            claimed: adjusted_claim,
        });
    }

    let (rows, col_scale) = match scale_side {
        // Put a nonzero Spartan scale on the folded row side, avoiding a
        // dense column-table scaling pass. A zero scale stays on the clear
        // side so it does not erase the row functional that exponent
        // folding certifies.
        ScaleSide::Rows if scale == 0 => (BitifiedRows::Structured(factors), 0),
        ScaleSide::Rows if scale == one => (BitifiedRows::Structured(factors), one),
        ScaleSide::Rows => (
            BitifiedRows::Structured(
                factors
                    .into_iter()
                    .map(|factor| mul(scale, factor))
                    .collect(),
            ),
            one,
        ),
        // The scale rides the clear column table; the rows stay unscaled.
        ScaleSide::Columns => (BitifiedRows::Structured(factors), scale),
    };

    Ok(BitifiedClaim {
        params,
        gate_point,
        rows,
        col_scale,
        claimed: adjusted_claim,
    })
}

/// Little-endian equality table with one fixed-factor multiplication per
/// parent and one allocation for the complete table.
pub fn eq_le_table_fq_fast_with(
    point: &[u128],
    arith: &field::FpCtx<2>,
) -> Result<Vec<u128>, ProtocolError> {
    let table_len = checked_pow2(point.len())?;
    let mut table = vec![0; table_len];
    table[0] = 1;

    let mut half = 1_usize;
    for &coordinate in point {
        let active_len = half
            .checked_mul(2)
            .ok_or(ProtocolError::InvalidBitzParameters)?;
        let factor = arith.prepare_multiplier_u128(coordinate);
        let (zero_children, one_children) = table[..active_len].split_at_mut(half);
        let expand = |zero: &mut u128, one: &mut u128| {
            let parent = *zero;
            let one_child = arith.mul_canonical_u128(parent, &factor);
            *zero = arith.sub_canonical_u128(parent, one_child);
            *one = one_child;
        };
        if half < 256 {
            zero_children
                .iter_mut()
                .zip(one_children.iter_mut())
                .for_each(|(zero, one)| expand(zero, one));
        } else {
            cfg_iter_mut!(zero_children, 256)
                .zip(cfg_iter_mut!(one_children, 256))
                .for_each(|(zero, one)| expand(zero, one));
        }
        half = active_len;
    }
    Ok(table)
}

/// The dense canonical row weights of a structured opening, in BitZ row
/// order `(word_slot << h) | gate_high`:
///
/// `w[(word_slot << h) | g] = block_factor(word_slot) · 2^{W · (word_slot − block_word_start)} · eq(gate_high_point, g)`
///
/// for every variable block, zero for word slots outside every block. The
/// per-word scalars are formed first, then every row is one fixed-factor
/// multiplication of the shared `eq` table.
fn structured_row_weights(
    params: &IntegerMatrixLayout,
    gate_high: &[u128],
    table: &BlockTable,
    factors: &[u128],
    arith: &field::FpCtx<2>,
) -> Result<Vec<u128>, ProtocolError> {
    if let Some((limb_bits, value_bits)) = table.packing {
        return packed_row_weights(
            params, gate_high, table, factors, arith, limb_bits, value_bits,
        );
    }
    let word_bits = params.word_bits;
    if !word_bits.is_power_of_two() {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    let high_gate_count = checked_pow2(gate_high.len())?;
    let row_count = checked_pow2(params.row_vars)?;
    if !row_count.is_multiple_of(high_gate_count) {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    let word_slots = row_count / high_gate_count;

    // Per-word scalars: the block factor times 2^{W·(word within the block)}.
    let pow2_word = arith.prepare_multiplier_u128(1_u128 << word_bits);
    let mut word_scalars = vec![0_u128; word_slots];
    for ((_, range), block_factor) in table.variable_blocks().zip(factors) {
        if !range.bit_slot_start.is_multiple_of(word_bits)
            || !range.bit_count.is_multiple_of(word_bits)
        {
            return Err(ProtocolError::InvalidBitzParameters);
        }
        let mut scalar = arith.reduce_u128(*block_factor);
        for word in
            range.bit_slot_start / word_bits..(range.bit_slot_start + range.bit_count) / word_bits
        {
            let Some(slot) = word_scalars.get_mut(word) else {
                return Err(ProtocolError::InvalidBitzParameters);
            };
            *slot = scalar;
            scalar = arith.mul_canonical_u128(scalar, &pow2_word);
        }
    }

    let eq_high = eq_le_table_fq_fast_with(gate_high, arith)?;
    let mut weights = vec![0_u128; row_count];
    cfg_chunks_mut!(weights, high_gate_count)
        .zip(cfg_iter!(word_scalars))
        .for_each(|(rows, &scalar)| {
            let factor = arith.prepare_multiplier_u128(scalar);
            for (row, equality) in rows.iter_mut().zip(&eq_high) {
                *row = arith.mul_canonical_u128(*equality, &factor);
            }
        });
    Ok(weights)
}

fn packed_row_weights(
    params: &IntegerMatrixLayout,
    gate_high: &[u128],
    table: &BlockTable,
    factors: &[u128],
    arith: &field::FpCtx<2>,
    limb_bits: usize,
    value_bits: usize,
) -> Result<Vec<u128>, ProtocolError> {
    let high = checked_pow2(gate_high.len())?;
    let cells = limb_bits.div_ceil(value_bits).next_power_of_two();
    let mut scalars = vec![0; params.rows() / high];
    for ((_, range), &factor) in table.variable_blocks().zip(factors) {
        if range.bit_slot_start % limb_bits != 0 || range.bit_count % limb_bits != 0 {
            return Err(ProtocolError::InvalidBitzParameters);
        }
        let mut power = 1;
        for bit in 0..range.bit_count {
            if bit % limb_bits % value_bits == 0 {
                let limb = (range.bit_slot_start + bit) / limb_bits;
                let cell = limb * cells + (bit % limb_bits) / value_bits;
                *scalars
                    .get_mut(cell)
                    .ok_or(ProtocolError::InvalidBitzParameters)? = arith.mul_u128(factor, power);
            }
            power = arith.add_u128(power, power);
        }
    }
    let eq = eq_le_table_fq_fast_with(gate_high, arith)?;
    let mut weights = vec![0; params.rows()];
    cfg_chunks_mut!(weights, high)
        .zip(cfg_iter!(scalars))
        .for_each(|(target, &scalar)| {
            let factor = arith.prepare_multiplier_u128(scalar);
            for (out, &weight) in target.iter_mut().zip(&eq) {
                *out = arith.mul_canonical_u128(weight, &factor);
            }
        });
    Ok(weights)
}

/// The dense canonical row weights of an opening (the dummy row functional
/// is `e_0`).
pub fn dense_row_weights(
    opening: &BitifiedClaim,
    table: &BlockTable,
    arith: &field::FpCtx<2>,
) -> Result<Vec<u128>, ProtocolError> {
    match &opening.rows {
        BitifiedRows::ConstantDummy => {
            let mut weights = vec![0_u128; checked_pow2(opening.params.row_vars)?];
            weights[0] = 1;
            Ok(weights)
        }
        BitifiedRows::Structured(factors) => {
            structured_row_weights(&opening.params, opening.gate_high(), table, factors, arith)
        }
    }
}

/// Compiles only the folded row functional into the mod-q chunk
/// representation. The prover never reads the clear column weights or the
/// claimed value, so keeping those verifier-only avoids an entire `2^s`
/// equality table on the proving path.
pub(crate) fn prepare_chunks(
    opening: &BitifiedClaim,
    table: &BlockTable,
    q_bits: usize,
    arith: &field::FpCtx<2>,
) -> Result<ModQWeightChunks, ProtocolError> {
    let params = opening.params;
    if opening.gate_point.len() < params.col_vars {
        return Err(ProtocolError::InvalidBitzParameters);
    }

    match &opening.rows {
        BitifiedRows::ConstantDummy => {
            let mut chunks = ModQWeightChunks::zeroed(&params, q_bits)
                .map_err(|_| ProtocolError::InvalidBitzParameters)?;
            chunks
                .set_weight_range(0, &[1])
                .map_err(|_| ProtocolError::InvalidBitzParameters)?;
            Ok(chunks)
        }
        BitifiedRows::Structured(factors) => {
            let weights =
                structured_row_weights(&params, opening.gate_high(), table, factors, arith)?;
            if mod_q_num_chunks(&params, q_bits) == 1 {
                ModQWeightChunks::from_single_chunk(&params, q_bits, weights)
                    .map_err(|_| ProtocolError::InvalidBitzParameters)
            } else {
                let mut chunks = ModQWeightChunks::zeroed(&params, q_bits)
                    .map_err(|_| ProtocolError::InvalidBitzParameters)?;
                chunks
                    .set_weight_range(0, &weights)
                    .map_err(|_| ProtocolError::InvalidBitzParameters)?;
                Ok(chunks)
            }
        }
    }
}

/// The clear column weights `col_scale · eq(gate_low, ·)` (all zero when the
/// clear read-off is switched off).
pub fn column_weights(
    opening: &BitifiedClaim,
    arith: &field::FpCtx<2>,
) -> Result<Vec<u128>, ProtocolError> {
    let params = opening.params;
    if opening.col_scale == 0 {
        return Ok(vec![0; checked_pow2(params.col_vars)?]);
    }
    let mut eq_low = eq_le_table_fq_fast_with(opening.gate_low(), arith)?;
    if opening.col_scale != 1 {
        let factor = arith.prepare_multiplier_u128(opening.col_scale);
        cfg_iter_mut!(&mut eq_low, 256).for_each(|weight| {
            *weight = arith.mul_canonical_u128(*weight, &factor);
        });
    }
    Ok(eq_low)
}

/// The digest binding the terminal Spartan claim and its bitification to the
/// relation, the statement binding and the runtime prime:
///
/// `domain ‖ binding ‖ len(modulus) ‖ modulus ‖ relation digest ‖ q ‖ <relation constants> ‖ claim ‖ gate point ‖ rows ‖ col_scale ‖ claimed`
///
/// where the relation writes its own constants section (layout numbers,
/// slot constants, mapping version) through `constants`.
pub fn bridge_digest(
    domain: &[u8],
    assignment_binding: &[u8; 32],
    relation_modulus_encoding: &[u8],
    relation_digest: &[u8; 32],
    modulus: u128,
    constants: impl FnOnce(&mut BindingHasher) -> Result<(), ProtocolError>,
    terminal_claim: &ScaledMleEvaluationClaim<SpartanBitzField>,
    opening: &BitifiedClaim,
    field_config: &super::FieldConfig,
) -> Result<[u8; 32], ProtocolError> {
    let mut hasher = BindingHasher::new();
    hasher.bytes(domain).bytes(assignment_binding);
    hasher.prefixed(relation_modulus_encoding)?;
    hasher.bytes(relation_digest).u128_le(modulus);
    constants(&mut hasher)?;

    hasher.usize(terminal_claim.point().len())?;
    for coordinate in terminal_claim.point() {
        hasher.element(coordinate, field_config);
    }
    hasher.element(terminal_claim.scale(), field_config);
    hasher.element(terminal_claim.value(), field_config);

    hasher.usize(opening.gate_low().len())?;
    for coordinate in opening.gate_low() {
        hasher.u128_le(*coordinate);
    }
    hasher.usize(opening.gate_high().len())?;
    for coordinate in opening.gate_high() {
        hasher.u128_le(*coordinate);
    }
    match &opening.rows {
        BitifiedRows::Structured(factors) => {
            hasher.byte(0);
            for factor in factors {
                hasher.u128_le(*factor);
            }
        }
        BitifiedRows::ConstantDummy => {
            hasher.byte(1);
        }
    }
    hasher.u128_le(opening.col_scale);
    hasher.u128_le(opening.claimed);
    Ok(hasher.finalize())
}
