//! Integer-level R1CS relation for batched `u64 * u64 = u128` multiplication.
//!
//! Each live row proves `x · y = z_lo + 2^64 · z_hi` over the integers for
//! `x, y, z_lo, z_hi < 2^64`. Spartan sees the exact integer assignment
//! `[e0 | x | y | z_lo | z_hi]` (five blocks, zero-padded to eight for the
//! assignment MLE); the product limbs are recombined by matrix `C` with the
//! public coefficient `2^64`, exactly like the BabyBear relation recombines
//! `c + p·k`. The 64/64/64/64-bit representation is materialized separately,
//! in the 256-slot-per-gate layout expected by the BitZ commitment, so bit
//! variables never become part of the R1CS statement.
//!
//! Unlike the u32 relation, a row's products (`x · y` and `z_lo + 2^64 z_hi`,
//! both below `2^128`) do not fit the native `u64` first-round path, so the
//! Spartan PIOP runs on values projected into the runtime field.

use crate::piop::spartan::SpartanField as _;
use crate::piop::spartan::mul::{MulError, MulLayout, MulWitness};
use circuit::linear_map::CscMatrix;
use field::RingOps;
use std::borrow::Cow;

use crate::{
    poly::mle::DenseMultilinearExtension,
    utils::{cfg_iter, cfg_iter_mut},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    ConstraintMatrices, ModulusIndependentCoefficient, PreparedConstraintMatrices, R1csProductMles,
    SpartanBitzField, SpartanField, SpartanMatrixCoefficient, SpartanMatrixError,
};

/// Number of committed little-endian bits used for each of `x`, `y`,
/// `z_lo`, and `z_hi`.
pub const U64_MUL_VALUE_BITS: usize = 64;
/// First bit slot occupied by the left operand.
pub const U64_MUL_X_SLOT_START: usize = 0;
/// First bit slot occupied by the right operand.
pub const U64_MUL_Y_SLOT_START: usize = U64_MUL_X_SLOT_START + U64_MUL_VALUE_BITS;
/// First bit slot occupied by the low product limb.
pub const U64_MUL_Z_LO_SLOT_START: usize = U64_MUL_Y_SLOT_START + U64_MUL_VALUE_BITS;
/// First bit slot occupied by the high product limb.
pub const U64_MUL_Z_HI_SLOT_START: usize = U64_MUL_Z_LO_SLOT_START + U64_MUL_VALUE_BITS;
/// Total number of bit slots committed for each multiplication (all of them
/// carry witness bits: there is no padding slot).
pub const U64_MUL_BIT_SLOTS: usize = 4 * U64_MUL_VALUE_BITS;
/// `log2` of [`U64_MUL_BIT_SLOTS`]: the physical slot coordinates on the
/// folded row axis.
pub const U64_MUL_SLOT_VARS: usize = 8;
/// The public limb base `2^64` recombining the product limbs in matrix `C`.
pub const U64_MUL_LIMB_BASE: u128 = 1 << U64_MUL_VALUE_BITS;

/// Little-endian canonical encoding of [`U64_MUL_LIMB_BASE`] as an element of
/// every accepted (at least 100-bit) Spartan field.
const LIMB_BASE_FIELD_ENCODING: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Five logical assignment blocks: `[e0 | x | y | z_lo | z_hi]`.
pub(super) const U64_MUL_LOGICAL_ASSIGNMENT_BLOCKS: usize = 5;
/// The assignment MLE pads the five logical blocks to eight.
pub(super) const U64_MUL_PADDED_ASSIGNMENT_BLOCKS: usize = 8;
// Keep even small relation fixtures in the geometry accepted by the BitZ row
// packer. The combined production proof applies its stricter 2^15 minimum.
#[cfg(test)]
const MIN_CAPACITY: usize = 1 << 8;

/// Compact coefficients used by the u64 multiplication matrices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum U64MulCoefficient {
    /// The field element one.
    One,
    /// The public limb base `2^64`.
    LimbBase,
}

impl SpartanMatrixCoefficient<SpartanBitzField> for U64MulCoefficient {
    fn validate(&self, _field_modulus_encoding: &[u8]) -> Result<(), SpartanMatrixError> {
        // Every Spartan field is at least 100 bits, so both public coefficients
        // are nonzero canonical elements in every accepted configuration.
        Ok(())
    }

    fn is_zero(&self) -> bool {
        false
    }

    fn canonical_field_encoding<'a>(
        &'a self,
        _field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
        field_one_encoding: &'a [u8],
    ) -> Cow<'a, [u8]> {
        match self {
            Self::One => Cow::Borrowed(field_one_encoding),
            Self::LimbBase => Cow::Borrowed(&LIMB_BASE_FIELD_ENCODING),
        }
    }

    fn scale(
        &self,
        value: &SpartanBitzField,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> SpartanBitzField {
        match self {
            Self::One => value.clone(),
            Self::LimbBase => {
                let coefficient = SpartanBitzField::from_with_cfg(U64_MUL_LIMB_BASE, field_config);
                let mut scaled = value.clone();
                scaled = field_config.mul(&(scaled), &(&coefficient));
                scaled
            }
        }
    }
}

/// Both public coefficients encode to modulus-independent bytes: `One` to
/// the field's canonical one (exactly like a Bit `true`) and `LimbBase`
/// to `2^64`, which is canonical and never the unit in any accepted (at
/// least 100-bit) Spartan field.
impl ModulusIndependentCoefficient<SpartanBitzField> for U64MulCoefficient {
    fn write_modulus_independent_encoding(&self, out: &mut Vec<u8>) {
        match self {
            Self::One => {
                ModulusIndependentCoefficient::<SpartanBitzField>::write_modulus_independent_encoding(
                    &true, out,
                )
            }
            Self::LimbBase => out.extend_from_slice(&LIMB_BASE_FIELD_ENCODING),
        }
    }

    fn is_unit(&self) -> bool {
        matches!(self, Self::One)
    }
}

impl MulWitness<u64> {
    /// Materializes the five logical blocks for dense reference consumers.
    pub fn assignment(&self) -> Vec<u64> {
        let n = self.layout.capacity();
        let mut out = Vec::with_capacity(self.layout.assignment_len());
        out.resize(n, 0);
        out[0] = 1;
        for block in [
            self.x_values(),
            self.y_values(),
            self.z_lo_values(),
            self.z_hi_values(),
        ] {
            out.extend_from_slice(block);
        }
        out
    }
}

/// Builds the compact CSC matrices for the u64 integer relation.
///
/// For live row `i` and capacity `M`, the only nonzero entries are
///
/// ```text
/// A[i, M+i]  = 1
/// B[i, 2M+i] = 1
/// C[i, 3M+i] = 1
/// C[i, 4M+i] = 2^64.
/// ```
pub fn u64_mul_constraint_matrices(
    layout: &MulLayout<u64>,
) -> Result<ConstraintMatrices<U64MulCoefficient>, MulError> {
    let a = super::mul::selector_matrix(layout, &[(1, U64MulCoefficient::One)])?;
    let b = super::mul::selector_matrix(layout, &[(2, U64MulCoefficient::One)])?;
    let c = super::mul::selector_matrix(
        layout,
        &[
            (3, U64MulCoefficient::One),
            (4, U64MulCoefficient::LimbBase),
        ],
    )?;
    Ok(ConstraintMatrices::new(a, b, c)?)
}

/// Generates and prepares the compact u64 matrices over the Spartan/BitZ
/// field.
pub fn prepare_u64_mul_relation(
    layout: MulLayout<u64>,
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<PreparedConstraintMatrices<SpartanBitzField, U64MulCoefficient>, MulError> {
    let matrices = u64_mul_constraint_matrices(&layout)?;
    Ok(PreparedConstraintMatrices::new(matrices, field_config)?)
}

/// Projects the exact assignment and products into a Spartan field.
///
/// Every assignment entry is below `2^64` and every product below `2^128`;
/// both are reduced modulo the field's prime, which is the Step-2 projection
/// of the integer relation. The tables are written straight into their padded
/// Spartan layouts (eight assignment blocks, power-of-two product rows) and
/// the conversions run in parallel: at `2^21` multiplications this is the
/// difference between a 0.6 s and a sub-0.1 s projection.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_u64_mul_witness<F>(
    witness: &MulWitness<u64>,
    field_config: &F::Config,
) -> Result<(DenseMultilinearExtension<F>, R1csProductMles<F>), MulError>
where
    F: SpartanField + Send + Sync,
    F::Config: Sync,
{
    F::validate_config(field_config).map_err(SpartanMatrixError::from)?;
    const MIN_LEN: usize = 4096;

    let layout = witness.layout;
    let capacity = layout.capacity;
    let live = layout.multiplications;
    let zero = F::zero_with_cfg(field_config);

    // Assignment MLE: `[e0 | x | y | z_lo | z_hi | 0 | 0 | 0]`, only the live
    // gates of the four value blocks are converted.
    let mut assignment = vec![zero.clone(); layout.padded_assignment_len()];
    assignment[0] = F::one_with_cfg(field_config);
    for (block, values) in [
        (1, witness.x_values()),
        (2, witness.y_values()),
        (3, witness.z_lo_values()),
        (4, witness.z_hi_values()),
    ] {
        let target = &mut assignment[block * capacity..block * capacity + live];
        cfg_iter_mut!(target, MIN_LEN)
            .zip(cfg_iter!(values[..live], MIN_LEN))
            .for_each(|(slot, &value)| *slot = F::from_with_cfg(value, field_config));
    }

    // Products: `Az = x`, `Bz = y` are the converted operand blocks; `Cz` is
    // the exact 128-bit product reduced into the field.
    let product_len = live.next_power_of_two();
    let product_vars = product_len.ilog2() as usize;
    let mut az = vec![zero.clone(); product_len];
    az[..live].clone_from_slice(&assignment[capacity..capacity + live]);
    let mut bz = vec![zero.clone(); product_len];
    bz[..live].clone_from_slice(&assignment[2 * capacity..2 * capacity + live]);
    let mut cz = vec![zero; product_len];
    let z_lo = witness.z_lo_values();
    let z_hi = witness.z_hi_values();
    cfg_iter_mut!(cz[..live], MIN_LEN)
        .zip(cfg_iter!(z_lo[..live], MIN_LEN))
        .zip(cfg_iter!(z_hi[..live], MIN_LEN))
        .for_each(|((slot, &lo), &hi)| {
            let product = u128::from(lo) | (u128::from(hi) << U64_MUL_VALUE_BITS);
            *slot = F::from_with_cfg(product, field_config);
        });

    let mle = |evaluations: Vec<F>, num_vars: usize| DenseMultilinearExtension {
        evaluations,
        num_vars,
    };
    Ok((
        mle(assignment, layout.assignment_vars()),
        R1csProductMles {
            az: mle(az, product_vars),
            bz: mle(bz, product_vars),
            cz: mle(cz, product_vars),
        },
    ))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::piop::spartan::spartan_bitz_field_config;

    fn inputs(n: usize) -> Vec<(u64, u64)> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        (0..n)
            .map(|index| {
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                match index % 5 {
                    0 => (u64::MAX, u64::MAX),
                    1 => (0, next()),
                    2 => (next(), 1),
                    _ => (next(), next()),
                }
            })
            .collect()
    }

    #[test]
    fn layout_shapes() {
        let layout = MulLayout::<u64>::new(1000).unwrap();
        assert_eq!(layout.capacity(), 1024);
        assert_eq!(layout.gate_vars(), 10);
        assert_eq!(layout.assignment_len(), 5 * 1024);
        assert_eq!(layout.padded_assignment_len(), 8 * 1024);
        assert_eq!(layout.assignment_vars(), 13);
        let p = layout.bitz_params();
        assert_eq!((p.row_vars, p.col_vars, p.word_bits), (8 + 5, 5, 1));
        assert_eq!(p.rows() * p.cols(), U64_MUL_BIT_SLOTS * layout.capacity());
        assert_eq!(MulLayout::<u64>::new(1).unwrap().capacity(), MIN_CAPACITY);
        assert_eq!(MulLayout::<u64>::new(0), Err(MulError::EmptyBatch));
        assert_eq!(U64_MUL_BIT_SLOTS, 1 << U64_MUL_SLOT_VARS);
        assert_eq!(
            u128::from_le_bytes(LIMB_BASE_FIELD_ENCODING),
            U64_MUL_LIMB_BASE
        );
    }

    #[test]
    fn witness_products_are_exact() {
        let inputs = inputs(37);
        let witness = MulWitness::<u64>::from_inputs(&inputs).unwrap();
        assert_eq!(witness.assignment()[0], 1);
        for (index, &(x, y)) in inputs.iter().enumerate() {
            assert_eq!(witness.x_values()[index], x);
            assert_eq!(witness.y_values()[index], y);
            assert_eq!(witness.product(index), u128::from(x) * u128::from(y));
        }
        assert!(witness.x_values()[37..].iter().all(|&v| v == 0));
        assert!(witness.z_hi_values()[37..].iter().all(|&v| v == 0));
        assert_eq!(
            witness.product(0),
            u128::from(u64::MAX) * u128::from(u64::MAX)
        );
    }

    #[test]
    fn bit_rows_match_cell_map_and_transposed_packer() {
        // 2^12 gates: s = 6, h = 6, high_gate_count = 64 → the transposed
        // packer runs; compare with the bitwise reference on the same witness.
        let inputs = inputs(3000);
        let witness = MulWitness::<u64>::from_inputs(&inputs).unwrap();
        let layout = *witness.layout();
        assert_eq!(layout.gate_vars(), 12);
        let rows = witness.bitz_bit_rows();
        let params = layout.bitz_params();
        let mut reference = vec![vec![0_u64; params.rows() / 64]; params.cols()];
        witness.write_bit_rows_bitwise(&mut reference);
        assert_eq!(rows, reference);

        let bit = |slot: usize, gate: usize| {
            let (b, c) = layout.bitz_cell(slot, gate).unwrap();
            (rows[c][b / 64] >> (b % 64)) & 1
        };
        for (gate, &(x, y)) in inputs.iter().enumerate().take(300) {
            let product = u128::from(x) * u128::from(y);
            for j in 0..64 {
                assert_eq!(bit(U64_MUL_X_SLOT_START + j, gate), (x >> j) & 1);
                assert_eq!(bit(U64_MUL_Y_SLOT_START + j, gate), (y >> j) & 1);
                assert_eq!(
                    bit(U64_MUL_Z_LO_SLOT_START + j, gate),
                    ((product >> j) & 1) as u64
                );
                assert_eq!(
                    bit(U64_MUL_Z_HI_SLOT_START + j, gate),
                    ((product >> (64 + j)) & 1) as u64
                );
            }
        }
        // Padding gates are all-zero.
        assert_eq!(bit(U64_MUL_X_SLOT_START, 3500), 0);
    }

    #[test]
    fn small_layout_uses_bitwise_path() {
        let inputs = inputs(300);
        let witness = MulWitness::<u64>::from_inputs(&inputs).unwrap();
        let layout = *witness.layout();
        assert_eq!(layout.gate_vars(), 9);
        let rows = witness.bitz_bit_rows();
        let (b, c) = layout.bitz_cell(U64_MUL_Y_SLOT_START + 3, 7).unwrap();
        assert_eq!((rows[c][b / 64] >> (b % 64)) & 1, (inputs[7].1 >> 3) & 1);
    }

    #[test]
    fn matrices_select_the_expected_columns() {
        let layout = MulLayout::<u64>::new(300).unwrap();
        let matrices = u64_mul_constraint_matrices(&layout).unwrap();
        let capacity = layout.capacity();
        assert_eq!(matrices.a().row_count(), 300);
        assert_eq!(matrices.a().column_count(), layout.assignment_len());
        for row in [0, 1, 299] {
            let single = |m: &CscMatrix<Box<[U64MulCoefficient]>>, column: usize| {
                let (row, coefficient) = m
                    .column(column)
                    .expect("column in range")
                    .single()
                    .expect("one entry");
                (row, *coefficient)
            };
            assert_eq!(
                single(matrices.a(), capacity + row),
                (row, U64MulCoefficient::One)
            );
            assert_eq!(
                single(matrices.b(), 2 * capacity + row),
                (row, U64MulCoefficient::One)
            );
            assert_eq!(
                single(matrices.c(), 3 * capacity + row),
                (row, U64MulCoefficient::One)
            );
            assert_eq!(
                single(matrices.c(), 4 * capacity + row),
                (row, U64MulCoefficient::LimbBase)
            );
        }
        assert!(
            matrices
                .a()
                .column(capacity + 300)
                .unwrap()
                .single()
                .is_none()
        );
        assert!(matrices.c().column(0).unwrap().single().is_none());
    }

    #[test]
    fn projection_satisfies_the_relation_in_the_field() {
        let inputs = inputs(300);
        let witness = MulWitness::<u64>::from_inputs(&inputs).unwrap();
        let config = spartan_bitz_field_config();
        let (assignment, products) =
            project_u64_mul_witness::<SpartanBitzField>(&witness, &config).unwrap();
        assert_eq!(
            assignment.evaluations.len(),
            witness.layout().padded_assignment_len()
        );
        let base = SpartanBitzField::from_with_cfg(U64_MUL_LIMB_BASE, &config);
        for index in 0..300 {
            let mut lhs = products.az.evaluations[index].clone();
            lhs = config.mul(&(lhs), &(&products.bz.evaluations[index]));
            assert_eq!(lhs, products.cz.evaluations[index]);
            let capacity = witness.layout().capacity();
            let mut recombined = assignment.evaluations[4 * capacity + index].clone();
            recombined = config.mul(&(recombined), &(&base));
            recombined = config.add(
                &(recombined),
                &(&assignment.evaluations[3 * capacity + index]),
            );
            assert_eq!(recombined, products.cz.evaluations[index]);
        }
        // The prepared matrices accept the coefficient encoding.
        let layout = *witness.layout();
        let prepared = prepare_u64_mul_relation(layout, &config).unwrap();
        assert_eq!(prepared.matrices().row_count(), 300);
    }
}
