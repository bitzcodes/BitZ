//! Integer-level R1CS relation for batched `u32 * u32 = u64` multiplication.
//!
//! Spartan sees only integer-valued assignment entries. The 32/32/64-bit
//! representation is materialized separately, in the layout expected by the
//! BitZ commitment, so bit variables never become part of the R1CS statement.
//! Splitting the product into its low and high 32-bit limbs gives the equivalent
//! modular relation `x * y = z + 2^32 * w`, with all four limbs range constrained
//! by the committed bit representation.

#[cfg(test)]
use crate::piop::spartan::MulRow;
use crate::piop::spartan::mul::{MulError, MulLayout, MulWitness};
#[cfg(test)]
use field::RingOps;
#[cfg(test)]
use field::{Fp, Uint};

use crate::poly::mle::DenseMultilinearExtension;

use super::{ConstraintMatrices, PreparedConstraintMatrices, R1csProductMles, SpartanField};

/// Number of committed little-endian bits used for each left operand.
pub const U32_MUL_X_BITS: usize = 32;
/// Number of committed little-endian bits used for each right operand.
pub const U32_MUL_Y_BITS: usize = 32;
/// Number of committed little-endian bits used for each product.
pub const U32_MUL_PRODUCT_BITS: usize = 64;
/// First bit slot occupied by the left operand.
pub const U32_MUL_X_SLOT_START: usize = 0;
/// First bit slot occupied by the right operand.
pub const U32_MUL_Y_SLOT_START: usize = U32_MUL_X_SLOT_START + U32_MUL_X_BITS;
/// First bit slot occupied by the product.
pub const U32_MUL_PRODUCT_SLOT_START: usize = U32_MUL_Y_SLOT_START + U32_MUL_Y_BITS;
/// Total number of bit slots committed for each multiplication.
pub const U32_MUL_BIT_SLOTS: usize = U32_MUL_X_BITS + U32_MUL_Y_BITS + U32_MUL_PRODUCT_BITS;

impl MulWitness<u32> {
    #[cfg(test)]
    fn from_product_fn(
        n: usize,
        width: usize,
        mut input: impl FnMut(usize) -> (u32, u32, u64),
    ) -> Result<Self, MulError> {
        Self::from_row_fn(MulLayout::new_with_word_bits(n, width)?, |i| {
            let (x, y, p) = input(i);
            super::MulRow {
                x,
                y,
                lo: p as u32,
                hi: (p >> 32) as u32,
            }
        })
    }

    /// Materializes the logical assignment for callers requiring a dense table.
    pub fn assignment(&self) -> Vec<u64> {
        let n = self.layout.capacity();
        let mut out = vec![0; 4 * n];
        out[0] = 1;
        for i in 0..n {
            out[n + i] = u64::from(self.x_values()[i]);
            out[2 * n + i] = u64::from(self.y_values()[i]);
            out[3 * n + i] = self.product(i);
        }
        out
    }
    pub fn product_values(&self) -> Vec<u64> {
        (0..self.layout.capacity())
            .map(|i| self.product(i))
            .collect()
    }
    pub fn az(&self) -> Vec<u64> {
        self.x_values()[..self.layout.multiplications()]
            .iter()
            .map(|&x| u64::from(x))
            .collect()
    }
    pub fn bz(&self) -> Vec<u64> {
        self.y_values()[..self.layout.multiplications()]
            .iter()
            .map(|&x| u64::from(x))
            .collect()
    }
    pub fn cz(&self) -> Vec<u64> {
        (0..self.layout.multiplications())
            .map(|i| self.product(i))
            .collect()
    }
}

/// Builds the three generic CSC selector matrices for this relation.
///
/// Each live row has one nonzero in each matrix:
///
/// `A[i,M+i] = 1`, `B[i,2M+i] = 1`, and `C[i,3M+i] = 1`.
pub fn u32_mul_constraint_matrices<C: Clone>(
    layout: &MulLayout<u32>,
    one: C,
) -> Result<ConstraintMatrices<C>, MulError> {
    let a = super::mul::selector_matrix(layout, &[(1, one.clone())])?;
    let b = super::mul::selector_matrix(layout, &[(2, one.clone())])?;
    let c = super::mul::selector_matrix(layout, &[(3, one)])?;
    Ok(ConstraintMatrices::new(a, b, c)?)
}

/// Generates and prepares the field-valued selector matrices.
pub fn prepare_u32_mul_relation<F>(
    layout: MulLayout<u32>,
    field_config: &F::Config,
) -> Result<PreparedConstraintMatrices<F, bool>, MulError>
where
    F: SpartanField,
{
    let matrices = u32_mul_constraint_matrices(&layout, true)?;
    Ok(PreparedConstraintMatrices::new(matrices, field_config)?)
}

/// Pads the exact integer assignment and products without projecting them to
/// the Spartan field.
///
/// This is the input boundary for a native first sumcheck round. A prover must
/// reduce each resulting round claim to `F` before transcript absorption and
/// must fold these tables into field-valued MLEs before a later multiplication.
pub fn project_u32_mul_native_witness(
    witness: &MulWitness<u32>,
) -> super::EvaluatedSpartanAssignment<u64> {
    let assignment = DenseMultilinearExtension {
        evaluations: witness.assignment(),
        num_vars: witness.layout.assignment_vars(),
    };

    let product_len = witness.layout.multiplications.next_power_of_two();
    let product_vars = product_len.ilog2() as usize;
    let product = |read: &dyn Fn(usize) -> u64| DenseMultilinearExtension {
        evaluations: (0..product_len).map(read).collect(),
        num_vars: product_vars,
    };
    let products = R1csProductMles {
        az: product(&|i| u64::from(witness.x_values()[i])),
        bz: product(&|i| u64::from(witness.y_values()[i])),
        cz: product(&|i| witness.product(i)),
    };
    super::EvaluatedSpartanAssignment::new(assignment, products)
}

#[cfg(test)]
mod tests {

    use super::*;

    const TEST_MODULUS: u128 = (1_u128 << 100) - 15;

    fn config() -> <Fp<2> as crate::piop::spartan::SpartanField>::Config {
        Fp::<2>::make_cfg(&Uint::from(TEST_MODULUS)).expect("odd test modulus")
    }

    #[test]
    fn layout_pads_to_a_power_of_two_and_maps_bits_for_each_word_width() {
        assert_eq!(MulLayout::<u32>::new(0), Err(MulError::EmptyBatch));

        for (multiplications, capacity) in [(1, 256), (3, 256), (256, 256), (257, 512)] {
            assert_eq!(
                MulLayout::<u32>::new(multiplications).unwrap(),
                MulLayout::<u32>::new_with_word_bits(multiplications, 1).unwrap()
            );

            for width in [1, 8] {
                let layout = MulLayout::<u32>::new_with_word_bits(multiplications, width).unwrap();
                assert_eq!(layout.multiplications(), multiplications);
                assert_eq!(layout.capacity(), capacity);
                assert_eq!(layout.assignment_len(), 4 * capacity);
                assert_eq!(layout.word_bits(), width);

                let p = layout.bitz_params();
                let word_bits = width;
                let s = layout.gate_vars() / 2;
                let h = layout.gate_vars() - s;
                assert_eq!(p.word_bits, word_bits);
                assert_eq!(p.row_vars, h + 7 - word_bits.trailing_zeros() as usize);
                assert_eq!(p.cells() * word_bits, U32_MUL_BIT_SLOTS * capacity);
                assert_eq!(layout.bitz_bit_position(U32_MUL_BIT_SLOTS, 0), None);
                assert_eq!(layout.bitz_bit_position(0, capacity), None);
                for slot in 0..U32_MUL_BIT_SLOTS {
                    for gate in 0..capacity {
                        let (b, c, j) = layout.bitz_bit_position(slot, gate).unwrap();
                        assert_eq!(b, ((slot / word_bits) << h) | (gate >> s));
                        assert_eq!(c, gate & ((1usize << s) - 1));
                        assert_eq!(j, slot % word_bits);
                        assert_eq!(p.cell_index(b, c), (slot / word_bits) * capacity + gate);
                        assert_eq!(layout.bitz_cell(slot, gate), Some((b, c)));
                    }
                }
            }
        }
    }

    #[test]
    fn exact_witness_uses_block_layout_and_zero_padding() {
        let inputs = [(0, u32::MAX), (1, 7), (u32::MAX, u32::MAX)];
        let witness = MulWitness::<u32>::from_inputs(&inputs).unwrap();
        let capacity = witness.layout().capacity();

        assert_eq!(capacity, 256);
        assert_eq!(witness.layout().word_bits(), 1);
        assert_eq!(witness.assignment()[0], 1);
        assert!(
            witness.assignment()[1..capacity]
                .iter()
                .all(|&value| value == 0)
        );
        assert_eq!(&witness.x_values()[..3], &[0, 1, u32::MAX]);
        assert_eq!(&witness.y_values()[..3], &[u32::MAX, 7, u32::MAX]);
        assert_eq!(
            &witness.product_values()[..3],
            &[0, 7, u64::from(u32::MAX) * u64::from(u32::MAX)]
        );
        assert!(witness.x_values()[3..].iter().all(|&value| value == 0));
        assert!(witness.y_values()[3..].iter().all(|&value| value == 0));
        assert!(
            witness.product_values()[3..]
                .iter()
                .all(|&value| value == 0)
        );
        assert_eq!(witness.az(), vec![0, 1, u64::from(u32::MAX)]);
        assert_eq!(
            witness.bz(),
            vec![u64::from(u32::MAX), 7, u64::from(u32::MAX)]
        );
        assert_eq!(witness.cz(), &witness.product_values()[..3]);
    }

    #[test]
    fn witness_initialization_covers_padding_and_allocation_boundary() {
        for count in [1, 255, 256, 257, 511, 512, 513, 1 << 19, (1 << 19) + 1] {
            for width in [1, 8] {
                let mut calls = 0;
                let witness = MulWitness::<u32>::from_product_fn(count, width, |index| {
                    assert_eq!(index, calls);
                    calls += 1;
                    // Include supplied products that intentionally do not
                    // equal x*y: construction must preserve their claims.
                    let x = (index as u32).wrapping_mul(0x9e37_79b9);
                    let y = !(index as u32);
                    (x, y, (index as u64).wrapping_mul(u64::MAX - 16))
                })
                .unwrap();
                assert_eq!(calls, count);
                let capacity = witness.layout().capacity();
                for (block, values) in witness.assignment().chunks_exact(capacity).enumerate() {
                    for (index, &value) in values.iter().enumerate() {
                        let expected = match (block, index < count) {
                            (0, _) => u64::from(index == 0),
                            (1, true) => u64::from((index as u32).wrapping_mul(0x9e37_79b9)),
                            (2, true) => u64::from(!(index as u32)),
                            (3, true) => (index as u64).wrapping_mul(u64::MAX - 16),
                            _ => 0,
                        };
                        assert_eq!(value, expected, "count={count} block={block} index={index}");
                    }
                }
            }
        }
    }

    #[test]
    fn modular_rows_preserve_exact_products_and_wrap_the_low_limb() {
        let rows: [MulRow<u32>; 5] = [
            MulRow::<u32>::new(0, u32::MAX),
            MulRow::<u32>::new(1, u32::MAX),
            MulRow::<u32>::new(65_536, 65_536),
            MulRow::<u32>::new(u32::MAX, u32::MAX),
            MulRow::<u32>::new(0x8000_0001, 3),
        ];
        assert_eq!(rows[0].lo, 0);
        assert_eq!(rows[0].hi, 0);
        assert_eq!(rows[1].lo, u32::MAX);
        assert_eq!(rows[1].hi, 0);
        assert_eq!(rows[2].lo, 0);
        assert_eq!(rows[2].hi, 1);
        assert_eq!(rows[3].lo, 1);
        assert_eq!(rows[3].hi, u32::MAX - 1);

        for row in rows {
            assert_eq!(row.lo, row.x.wrapping_mul(row.y));
            assert_eq!(row.product(), u64::from(row.x) * u64::from(row.y));
        }
        let witness = MulWitness::<u32>::from_rows(&rows).unwrap();
        assert_eq!(witness.rows().len(), rows.len());
        assert_eq!(witness.rows().collect::<Vec<_>>(), rows);
        let inputs = rows.map(|row| (row.x, row.y));
        assert_eq!(witness, MulWitness::<u32>::from_inputs(&inputs).unwrap());
        assert_eq!(MulWitness::<u32>::from_rows(&[]), Err(MulError::EmptyBatch));
    }

    #[test]
    fn supplied_incorrect_modular_limbs_reach_the_assignment_unchanged() {
        let correct = MulRow::<u32>::new(u32::MAX, u32::MAX);
        let incorrect = [
            MulRow {
                lo: correct.lo ^ 1,
                ..correct
            },
            MulRow {
                hi: correct.hi ^ 1,
                ..correct
            },
        ];
        let witness = MulWitness::<u32>::from_rows(&incorrect).unwrap();
        assert_eq!(witness.rows().collect::<Vec<_>>(), incorrect);
        for (index, row) in incorrect.iter().enumerate() {
            assert_eq!(witness.az()[index], u64::from(row.x));
            assert_eq!(witness.bz()[index], u64::from(row.y));
            assert_eq!(witness.cz()[index], row.product());
            assert_ne!(
                witness.az()[index] * witness.bz()[index],
                witness.cz()[index]
            );
        }
    }

    #[test]
    fn modular_claims_commit_four_consecutive_32_bit_limbs() {
        let claims = [
            MulRow::<u32>::new(u32::MAX, u32::MAX),
            MulRow::<u32>::new(65_536, 65_536),
            // Deliberately supplied limbs exercise every field independently
            // of whether the claimed multiplication is satisfied.
            MulRow {
                x: 0x1234_5678,
                y: 0x8765_4321,
                lo: 0xaaaa_5555,
                hi: 0x5555_aaaa,
            },
        ];
        for width in [1, 8] {
            let witness = MulWitness::<u32>::from_rows_with_word_bits(&claims, width).unwrap();
            let layout = witness.layout();
            let packed_rows = witness.bitz_bit_rows();
            for (gate, row) in claims.iter().enumerate() {
                for (limb, value) in [row.x, row.y, row.lo, row.hi].into_iter().enumerate() {
                    for bit in 0..32 {
                        let (b, c, j) = layout.bitz_bit_position(limb * 32 + bit, gate).unwrap();
                        let packed_bit = b * width + j;
                        let actual = (packed_rows[c][packed_bit / 64] >> (packed_bit % 64)) & 1;
                        assert_eq!(actual, u64::from((value >> bit) & 1));
                    }
                }
            }
        }
    }

    #[test]
    fn from_fn_generates_each_input_once_without_an_input_buffer() {
        let mut calls = Vec::new();
        let witness = MulWitness::<u32>::from_fn_with_word_bits(5, 8, |index| {
            calls.push(index);
            (index as u32, (index + 1) as u32)
        })
        .unwrap();

        assert_eq!(calls, (0..5).collect::<Vec<_>>());
        assert_eq!(witness.cz(), &[0, 2, 6, 12, 20]);
        assert_eq!(witness.layout().word_bits(), 8);
    }

    #[test]
    fn native_mles_preserve_values_and_pad_only_the_row_domain() {
        let inputs = [(2, 3), (u32::MAX, u32::MAX), (11, 13)];
        let witness = MulWitness::<u32>::from_inputs(&inputs).unwrap();
        let native = project_u32_mul_native_witness(&witness);

        assert_eq!(
            native.assignment().num_vars,
            witness.layout().gate_vars() + 2
        );
        assert_eq!(native.assignment().evaluations, witness.assignment());
        assert_eq!(native.products().az.num_vars, 2);
        assert_eq!(native.products().bz.num_vars, 2);
        assert_eq!(native.products().cz.num_vars, 2);
        assert_eq!(&native.products().az.evaluations[..3], witness.az());
        assert_eq!(&native.products().bz.evaluations[..3], witness.bz());
        assert_eq!(&native.products().cz.evaluations[..3], witness.cz());
        assert_eq!(native.products().az.evaluations[3], 0);
        assert_eq!(native.products().bz.evaluations[3], 0);
        assert_eq!(native.products().cz.evaluations[3], 0);
    }

    #[test]
    fn generic_matrices_are_exact_csc_selectors() {
        let layout = MulLayout::<u32>::new(3).unwrap();
        let matrices = u32_mul_constraint_matrices(&layout, 1_u64).unwrap();
        let capacity = layout.capacity();

        assert_eq!(matrices.row_count(), 3);
        assert_eq!(matrices.column_count(), 4 * capacity);
        for matrix in [matrices.a(), matrices.b(), matrices.c()] {
            assert_eq!(matrix.nnz(), 3);
        }

        for row in 0..layout.multiplications() {
            for column in [
                matrices.a().column(capacity + row).unwrap(),
                matrices.b().column(2 * capacity + row).unwrap(),
                matrices.c().column(3 * capacity + row).unwrap(),
            ] {
                assert_eq!(column.indices(), &[row]);
                assert_eq!(column.coefficients(), &[1]);
            }
        }
        assert!(matrices.a().column(0).unwrap().is_empty());
        assert!(matrices.a().column(2 * capacity).unwrap().is_empty());
        assert!(matrices.b().column(capacity).unwrap().is_empty());
        assert!(matrices.c().column(2 * capacity).unwrap().is_empty());
    }

    #[test]
    fn packed_rows_reconstruct_the_32_32_64_bit_witness_for_each_word_width() {
        let inputs = [(0x8000_0001, 3), (u32::MAX, u32::MAX), (17, 19)];
        for width in [1, 8] {
            let witness = MulWitness::<u32>::from_inputs_with_word_bits(&inputs, width).unwrap();
            let layout = witness.layout();
            let p = layout.bitz_params();
            let rows = witness.bitz_bit_rows();

            assert_eq!(rows.len(), p.cols());
            assert!(
                rows.iter()
                    .all(|row| row.len() == p.rows() * p.word_bits / 64)
            );

            for gate in 0..layout.capacity() {
                let values = [
                    (
                        U32_MUL_X_SLOT_START,
                        U32_MUL_X_BITS,
                        u64::from(witness.x_values()[gate]),
                    ),
                    (
                        U32_MUL_Y_SLOT_START,
                        U32_MUL_Y_BITS,
                        u64::from(witness.y_values()[gate]),
                    ),
                    (
                        U32_MUL_PRODUCT_SLOT_START,
                        U32_MUL_PRODUCT_BITS,
                        witness.product(gate),
                    ),
                ];
                for (slot_offset, bit_width, value) in values {
                    for bit in 0..bit_width {
                        let (b, c, j) = layout.bitz_bit_position(slot_offset + bit, gate).unwrap();
                        let packed_bit = b * p.word_bits + j;
                        let committed_bit = (rows[c][packed_bit / 64] >> (packed_bit % 64)) & 1;
                        assert_eq!(committed_bit, (value >> bit) & 1);
                    }
                }
            }
        }
    }

    #[test]
    fn transposed_bit_rows_match_the_bitwise_packing_for_each_word_width() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        // gate_vars 10 (W1 bitwise, W8 transposed), 11 (one W1 word per
        // lane), 13, and 15/16 (production layouts); live counts off the
        // power of two exercise the zero padding.
        for width in [1, 8] {
            for (multiplications, seed) in [
                (700, 1),
                (1500, 2),
                (5000, 3),
                (1 << 15, 4),
                ((1 << 15) + 37, 5),
            ] {
                let mut rng = StdRng::seed_from_u64(seed);
                let witness =
                    MulWitness::<u32>::from_fn_with_word_bits(multiplications, width, |_| {
                        (rng.random::<u32>(), rng.random::<u32>())
                    })
                    .unwrap();
                let p = witness.layout().bitz_params();
                let mut expected = vec![vec![0_u64; p.rows() * p.word_bits / 64]; p.cols()];
                witness.write_bit_rows_bitwise(&mut expected);
                assert_eq!(
                    witness.bitz_bit_rows(),
                    expected,
                    "width={width:?} multiplications={multiplications}"
                );
            }
        }
    }

    #[test]
    fn prepared_u32_relation_uses_boolean_selectors() {
        let config = config();
        let layout = MulLayout::<u32>::new(3).unwrap();
        let relation = prepare_u32_mul_relation::<Fp<2>>(layout, &config).unwrap();
        let capacity = layout.capacity();

        for (column, row) in [
            (relation.matrices().a().column(capacity).unwrap(), 0),
            (relation.matrices().b().column(2 * capacity + 1).unwrap(), 1),
            (relation.matrices().c().column(3 * capacity + 2).unwrap(), 2),
        ] {
            assert_eq!(column.indices(), &[row]);
            assert_eq!(column.coefficients(), &[true]);
        }
    }
}
