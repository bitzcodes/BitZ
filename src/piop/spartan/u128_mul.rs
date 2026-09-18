//! Integer-level R1CS relation for batched `u128 * u128 = u256` multiplication.
//!
//! Each live row proves `x · y = z` over the integers for `x, y < 2^128` and
//! `z < 2^256`. Unlike the u64 relation, the assignment entries are the
//! integers themselves — `[e0 | x | y | z]`, four blocks, so the assignment
//! MLE needs no padding block — and the three matrices are plain Bit
//! selectors: the Step-2 projection reduces `x`, `y`, `z` modulo the sampled
//! prime and the bitification carries the weights `2^0 … 2^255` of the
//! committed bits (512 per gate: `x` in slots `0..128`, `y` in `128..256`,
//! `z` in `256..512`). No matrix coefficient ever exceeds one, so the
//! prime-independent skeleton applies unchanged.
//!
//! The witness stores every value as `u64` limbs (`x`, `y` as two limbs,
//! `z` as four) purely as host representation; the relation is over `ℤ`.

use crate::piop::spartan::SpartanField as _;
use crate::piop::spartan::mul::{MulError, MulLayout, MulWitness};
use circuit::linear_map::CscMatrix;
use field::RingOps;

use crate::{
    poly::mle::DenseMultilinearExtension,
    utils::{cfg_iter, cfg_iter_mut},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    ConstraintMatrices, PreparedConstraintMatrices, R1csProductMles, SpartanField,
    SpartanMatrixError,
};

/// Committed little-endian bits of each operand.
pub const U128_MUL_OPERAND_BITS: usize = 128;
/// Committed little-endian bits of each product.
pub const U128_MUL_PRODUCT_BITS: usize = 256;
/// First bit slot occupied by the left operand.
pub const U128_MUL_X_SLOT_START: usize = 0;
/// First bit slot occupied by the right operand.
pub const U128_MUL_Y_SLOT_START: usize = U128_MUL_X_SLOT_START + U128_MUL_OPERAND_BITS;
/// First bit slot occupied by the product.
pub const U128_MUL_Z_SLOT_START: usize = U128_MUL_Y_SLOT_START + U128_MUL_OPERAND_BITS;
/// Total number of bit slots committed for each multiplication.
pub const U128_MUL_BIT_SLOTS: usize = U128_MUL_Z_SLOT_START + U128_MUL_PRODUCT_BITS;
/// `log2` of [`U128_MUL_BIT_SLOTS`]: the physical slot coordinates on the
/// folded row axis.
pub const U128_MUL_SLOT_VARS: usize = 9;

/// Four assignment blocks: `[e0 | x | y | z]`.
pub(super) const U128_MUL_ASSIGNMENT_BLOCKS: usize = 4;

/// The exact 256-bit product of two 128-bit integers as `(low, high)`
/// 128-bit halves.
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
pub const fn mul_u128_full(x: u128, y: u128) -> (u128, u128) {
    let (x0, x1) = (x as u64 as u128, x >> 64);
    let (y0, y1) = (y as u64 as u128, y >> 64);
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    // Middle terms: p01 + p10 < 2^129, split around bit 64.
    let (middle, middle_carry) = p01.overflowing_add(p10);
    let middle_low = middle << 64;
    let middle_high = (middle >> 64) | ((middle_carry as u128) << 64);
    let (low, low_carry) = p00.overflowing_add(middle_low);
    // p11 + middle_high + carry < 2^128: the full product is below 2^256.
    let high = p11 + middle_high + low_carry as u128;
    (low, high)
}

/// Builds the three Bit selector matrices for this relation.
///
/// Each live row has one nonzero in each matrix:
///
/// `A[i,M+i] = 1`, `B[i,2M+i] = 1`, and `C[i,3M+i] = 1`.
pub fn u128_mul_constraint_matrices(
    layout: &MulLayout<u128>,
) -> Result<ConstraintMatrices<bool>, MulError> {
    let a = super::mul::selector_matrix(layout, &[(1, true)])?;
    let b = super::mul::selector_matrix(layout, &[(2, true)])?;
    let c = super::mul::selector_matrix(layout, &[(3, true)])?;
    Ok(ConstraintMatrices::new(a, b, c)?)
}

/// Generates and prepares the Bit selector matrices over a Spartan field.
pub fn prepare_u128_mul_relation<F>(
    layout: MulLayout<u128>,
    field_config: &F::Config,
) -> Result<PreparedConstraintMatrices<F, bool>, MulError>
where
    F: SpartanField,
{
    let matrices = u128_mul_constraint_matrices(&layout)?;
    Ok(PreparedConstraintMatrices::new(matrices, field_config)?)
}

/// Projects the exact assignment and products into a Spartan field: `x`,
/// `y`, and the 256-bit `z` reduced modulo the field's prime (the Step-2
/// projection of the integer relation), written straight into the Spartan
/// layouts in parallel.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_u128_mul_witness<F>(
    witness: &MulWitness<u128>,
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
    // 2^128 as a field element: 2^127 fits a u128, then double.
    let two_pow_128 = {
        let mut value = F::from_with_cfg(1_u128 << 127, field_config);
        let two = F::from_with_cfg(2_u64, field_config);
        value = field_config.mul(&(value), &(&two));
        value
    };
    let reduce_z = |lo: u128, hi: u128| -> F {
        let mut value = F::from_with_cfg(hi, field_config);
        value = field_config.mul(&(value), &(&two_pow_128));
        value = field_config.add(&(value), &(&F::from_with_cfg(lo, field_config)));
        value
    };

    let mut assignment = vec![zero.clone(); layout.assignment_len()];
    assignment[0] = F::one_with_cfg(field_config);
    for (block, values) in [(1, witness.x_values()), (2, witness.y_values())] {
        let target = &mut assignment[block * capacity..block * capacity + live];
        cfg_iter_mut!(target, MIN_LEN)
            .zip(cfg_iter!(values[..live], MIN_LEN))
            .for_each(|(slot, &value)| *slot = F::from_with_cfg(value, field_config));
    }
    {
        let target = &mut assignment[3 * capacity..3 * capacity + live];
        cfg_iter_mut!(target, MIN_LEN)
            .zip(cfg_iter!(witness.z_lo_values()[..live], MIN_LEN))
            .zip(cfg_iter!(witness.z_hi_values()[..live], MIN_LEN))
            .for_each(|((slot, &lo), &hi)| *slot = reduce_z(lo, hi));
    }

    let product_len = live.next_power_of_two();
    let product_vars = product_len.ilog2() as usize;
    let mut az = vec![zero.clone(); product_len];
    az[..live].clone_from_slice(&assignment[capacity..capacity + live]);
    let mut bz = vec![zero.clone(); product_len];
    bz[..live].clone_from_slice(&assignment[2 * capacity..2 * capacity + live]);
    let mut cz = vec![zero; product_len];
    cz[..live].clone_from_slice(&assignment[3 * capacity..3 * capacity + live]);

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
    use num_bigint::BigUint;

    use super::*;
    use crate::piop::spartan::{SpartanBitzField, spartan_bitz_field_config};

    fn inputs(n: usize) -> Vec<(u128, u128)> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..n)
            .map(|index| {
                let wide = |a: u64, b: u64| (u128::from(a) << 64) | u128::from(b);
                match index % 5 {
                    0 => (u128::MAX, u128::MAX),
                    1 => (0, wide(next(), next())),
                    2 => (wide(next(), next()), 1),
                    3 => (u128::from(next()), u128::from(next())),
                    _ => (wide(next(), next()), wide(next(), next())),
                }
            })
            .collect()
    }

    #[test]
    fn full_product_matches_bigint() {
        for (x, y) in inputs(500) {
            let (lo, hi) = mul_u128_full(x, y);
            let expected = BigUint::from(x) * BigUint::from(y);
            let actual = (BigUint::from(hi) << 128) | BigUint::from(lo);
            assert_eq!(actual, expected, "{x} * {y}");
        }
    }

    #[test]
    fn layout_shapes() {
        let layout = MulLayout::<u128>::new(1000).unwrap();
        assert_eq!(layout.capacity(), 1024);
        assert_eq!(layout.gate_vars(), 10);
        assert_eq!(layout.assignment_len(), 4 * 1024);
        assert_eq!(layout.assignment_vars(), 12);
        let p = layout.bitz_params();
        assert_eq!((p.row_vars, p.col_vars, p.word_bits), (9 + 5, 5, 1));
        assert_eq!(p.rows() * p.cols(), U128_MUL_BIT_SLOTS * layout.capacity());
        assert_eq!(U128_MUL_BIT_SLOTS, 1 << U128_MUL_SLOT_VARS);
        assert_eq!(MulLayout::<u128>::new(0), Err(MulError::EmptyBatch));
    }

    #[test]
    fn bit_rows_match_cell_map_and_transposed_packer() {
        // 2^12 gates: s = 6, h = 6, high_gate_count = 64 → the transposed
        // packer runs; compare with the bitwise reference on the same witness.
        let inputs = inputs(3000);
        let witness = MulWitness::<u128>::from_inputs(&inputs).unwrap();
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
        for (gate, &(x, y)) in inputs.iter().enumerate().take(200) {
            let (lo, hi) = mul_u128_full(x, y);
            for j in 0..128 {
                assert_eq!(bit(U128_MUL_X_SLOT_START + j, gate), ((x >> j) & 1) as u64);
                assert_eq!(bit(U128_MUL_Y_SLOT_START + j, gate), ((y >> j) & 1) as u64);
                assert_eq!(bit(U128_MUL_Z_SLOT_START + j, gate), ((lo >> j) & 1) as u64);
                assert_eq!(
                    bit(U128_MUL_Z_SLOT_START + 128 + j, gate),
                    ((hi >> j) & 1) as u64
                );
            }
        }
        assert_eq!(bit(U128_MUL_X_SLOT_START, 3500), 0);
    }

    #[test]
    fn matrices_select_the_expected_columns() {
        let layout = MulLayout::<u128>::new(300).unwrap();
        let matrices = u128_mul_constraint_matrices(&layout).unwrap();
        let capacity = layout.capacity();
        assert_eq!(matrices.a().row_count(), 300);
        assert_eq!(matrices.a().column_count(), layout.assignment_len());
        for row in [0, 1, 299] {
            let single = |m: &CscMatrix<Box<[bool]>>, column: usize| {
                let (row, coefficient) = m
                    .column(column)
                    .expect("column in range")
                    .single()
                    .expect("one entry");
                (row, *coefficient)
            };
            assert_eq!(single(matrices.a(), capacity + row), (row, true));
            assert_eq!(single(matrices.b(), 2 * capacity + row), (row, true));
            assert_eq!(single(matrices.c(), 3 * capacity + row), (row, true));
        }
        assert!(matrices.c().column(0).unwrap().single().is_none());
    }

    #[test]
    fn projection_satisfies_the_relation_in_the_field() {
        let inputs = inputs(300);
        let witness = MulWitness::<u128>::from_inputs(&inputs).unwrap();
        let config = spartan_bitz_field_config();
        let (assignment, products) =
            project_u128_mul_witness::<SpartanBitzField>(&witness, &config).unwrap();
        assert_eq!(
            assignment.evaluations.len(),
            witness.layout().assignment_len()
        );
        let capacity = witness.layout().capacity();
        for index in 0..300 {
            let mut lhs = products.az.evaluations[index].clone();
            lhs = config.mul(&(lhs), &(&products.bz.evaluations[index]));
            assert_eq!(lhs, products.cz.evaluations[index]);
            assert_eq!(
                assignment.evaluations[3 * capacity + index],
                products.cz.evaluations[index]
            );
        }
        let prepared =
            prepare_u128_mul_relation::<SpartanBitzField>(*witness.layout(), &config).unwrap();
        assert_eq!(prepared.matrices().row_count(), 300);
    }
}
