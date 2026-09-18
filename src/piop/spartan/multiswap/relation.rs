//! BitZ embedding of the MultiSwap integer Mod-R1CS.
//!
//! The per-row modulus term is folded into the output matrix: with the
//! quotients placed in their own assignment block, `mods[r] * quos[r]` is the
//! linear term `mods[r] * z[quos_column(r)]`, so the Mod-R1CS becomes the
//! plain R1CS `A·z ∘ B·z = C'·z` over
//!
//! ```text
//! z = [constant block | witness block | quotient block | zero block],
//! ```
//!
//! four blocks of one shared power-of-two gate capacity.  Only `z[0] = 1` is
//! nonzero in the constant block, and the fourth block is identically zero
//! (it carries no committed bits and no matrix entries, so the opening
//! forces it to zero).
//!
//! The BitZ commitment stores, for every gate, the
//! [`MULTISWAP_VALUE_BITS`]-bit little-endian decompositions of its witness
//! and quotient entries: `2^12` bit slots per gate (witness bits first).
//! With `s` low gate coordinates on the clear column axis and the remaining
//! `h` high gate coordinates joining the twelve slot coordinates on the
//! folded row axis, the committed tensor has parameters
//! `t = 12 + h`, `s`, `W = 1`.  The layout keeps `t <= 13` so a 113-bit
//! runtime prime needs exactly one mod-q weight chunk.

use super::super::raw_monty::NativeLimbWitness;
#[cfg(test)]
use crate::piop::spartan::{R1csProductMles, build_assignment_mle, build_product_mles};
#[cfg(test)]
use crate::poly::mle::DenseMultilinearExtension;
use circuit::linear_map::CscMatrix;
use field::CtOrd;
#[cfg(test)]
use field::RingOps;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use circuit::integer_storage::UnsignedIntegerTable;
use field::{Fp, FpCtx, IntegerEmbedding, Uint, UintRef};
#[cfg(test)]
use num_bigint::BigUint;
#[cfg(test)]
use num_traits::Zero;
use thiserror::Error;

use crate::pcs::IntegerMatrixLayout;

use super::super::{
    ConstraintMatrices, PreparedConstraintMatrices, SpartanField, SpartanMatrixError,
};
use super::circuit::{IntegerCoo, MULTISWAP_VALUE_BITS, MultiswapCircuit};

/// Number of committed bit slots per gate (`witness || quotient`).
pub const MULTISWAP_SLOTS: usize = 2 * MULTISWAP_VALUE_BITS;
/// `log2` of [`MULTISWAP_SLOTS`].
pub const MULTISWAP_SLOT_VARS: usize = 12;
/// First bit slot of the witness value.
pub const MULTISWAP_W_SLOT_START: usize = 0;
/// First bit slot of the quotient value.
pub const MULTISWAP_QUOS_SLOT_START: usize = MULTISWAP_VALUE_BITS;

const ASSIGNMENT_BLOCKS: usize = 4;
/// High gate coordinates kept on the folded row axis whenever the gate
/// domain allows it: `t = 12 + h` stays at 13, the one-chunk boundary for a
/// 113-bit runtime prime.
const MAX_HIGH_GATE_VARS: usize = 1;

/// Failures in layout construction or witness materialization.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum MultiswapLayoutError {
    /// The circuit's padded gate domain does not fit the layout.
    #[error("multiswap gate domain is empty or too large")]
    InvalidGateDomain,

    #[error("multiswap public coefficient sum exceeds the 2048-bit row bound")]
    ProductWidthExceeded,

    /// A generated matrix or assignment is malformed.
    #[error(transparent)]
    Matrix(#[from] SpartanMatrixError),
}

/// Shared shape of the block assignment and its compact BitZ bit tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiswapLayout {
    capacity: usize,
    gate_vars: usize,
    s: usize,
    h: usize,
}

impl MultiswapLayout {
    /// Derives the layout from the circuit's padded row/column counts.
    pub fn new(circuit: &MultiswapCircuit) -> Result<Self, MultiswapLayoutError> {
        let capacity = circuit.num_cons().max(circuit.num_vars());
        if !capacity.is_power_of_two() || capacity < 2 {
            return Err(MultiswapLayoutError::InvalidGateDomain);
        }
        let gate_vars = capacity.trailing_zeros() as usize;
        capacity
            .checked_mul(ASSIGNMENT_BLOCKS)
            .and_then(|len| len.checked_mul(MULTISWAP_SLOTS))
            .ok_or(MultiswapLayoutError::InvalidGateDomain)?;
        let h = gate_vars.min(MAX_HIGH_GATE_VARS);
        Ok(Self {
            capacity,
            gate_vars,
            s: gate_vars - h,
            h,
        })
    }

    /// Power-of-two gate capacity shared by all four assignment blocks.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of gate-selecting variables.
    pub const fn gate_vars(&self) -> usize {
        self.gate_vars
    }

    /// Number of clear column variables.
    pub const fn column_vars(&self) -> usize {
        self.s
    }

    /// Number of high gate variables on the folded row axis.
    pub const fn high_gate_vars(&self) -> usize {
        self.h
    }

    /// Complete assignment length: four blocks of `capacity` entries.
    pub const fn assignment_len(&self) -> usize {
        ASSIGNMENT_BLOCKS * self.capacity
    }

    /// First assignment index of the witness block.
    pub const fn witness_block_start(&self) -> usize {
        self.capacity
    }

    /// First assignment index of the quotient block.
    pub const fn quotient_block_start(&self) -> usize {
        2 * self.capacity
    }

    /// BitZ shape of the committed bit tensor.
    pub const fn bitz_params(&self) -> IntegerMatrixLayout {
        IntegerMatrixLayout {
            row_vars: MULTISWAP_SLOT_VARS + self.h,
            col_vars: self.s,
            word_bits: 1,
        }
    }

    /// Maps `(bit_slot, gate)` to the folded row and clear column indices.
    pub const fn bitz_bit_position(&self, bit_slot: usize, gate: usize) -> Option<(usize, usize)> {
        if bit_slot >= MULTISWAP_SLOTS || gate >= self.capacity {
            return None;
        }
        let column_mask = (1usize << self.s) - 1;
        let row = (bit_slot << self.h) | (gate >> self.s);
        Some((row, gate & column_mask))
    }
}

/// The q-independent integer relation in the block column space.
///
/// Matrices are column-major with the per-row moduli already folded into
/// `C` on the quotient block.  Coefficients are the exact circuit integers;
/// projection modulo the runtime prime happens per proof.
#[derive(Clone, Debug)]
pub struct MultiswapIntegerRelation {
    layout: MultiswapLayout,
    live_rows: usize,
    coefficients: UnsignedIntegerTable,
    a: Vec<Vec<(usize, usize)>>,
    b: Vec<Vec<(usize, usize)>>,
    c: Vec<Vec<(usize, usize)>>,
}

impl MultiswapIntegerRelation {
    /// Remaps the circuit's COO matrices into the block column space and
    /// folds `mods` into `C`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(circuit: &MultiswapCircuit) -> Result<Self, MultiswapLayoutError> {
        let layout = MultiswapLayout::new(circuit)?;
        let live_rows = circuit.live_rows();
        let columns = layout.assignment_len();
        let const_col = circuit.const_col();
        let witness_start = layout.witness_block_start();
        let quotient_start = layout.quotient_block_start();

        let remap = |column: usize| {
            if column == const_col {
                0
            } else {
                witness_start + column
            }
        };

        let mut coefficients = UnsignedIntegerTable::default();
        let mut collect = |entries: &IntegerCoo| -> Vec<Vec<(usize, usize)>> {
            let mut columns_entries = vec![Vec::new(); columns];
            for (index, (row, column, _)) in entries.iter().enumerate() {
                let coefficient = coefficients.len();
                entries.copy_coefficient_to(index, &mut coefficients);
                columns_entries[remap(column)].push((row, coefficient));
            }
            columns_entries
        };
        let a = collect(circuit.a_entries());
        let b = collect(circuit.b_entries());
        let mut c = collect(circuit.c_entries());
        for (row, modulus) in circuit.mods().iter().enumerate().take(live_rows) {
            if modulus.iter().any(|&word| word != 0) {
                let coefficient = coefficients.len();
                circuit.mods().copy_row_to(row, &mut coefficients);
                c[quotient_start + row].push((row, coefficient));
            }
        }

        let mut relation = Self {
            coefficients,
            layout,
            live_rows,
            a,
            b,
            c,
        };
        relation.normalize();
        relation.validate_product_width()?;
        Ok(relation)
    }

    /// Bound row products using public coefficients and the declared assignment
    /// width. Each input is <2^2048 and each coefficient sum is <=2^2048,
    /// hence every A/B/C row is strictly below 2^4096, independently of values.
    fn validate_product_width(&self) -> Result<(), MultiswapLayoutError> {
        let limit = Uint::<33>::ONE.truncating_shl(2048);
        for matrix in [&self.a, &self.b, &self.c] {
            let mut sums = vec![Uint::<33>::ZERO; self.live_rows];
            for &(row, index) in matrix.iter().flatten() {
                let words = &self.coefficients[index];
                if words.len() > 32 {
                    return Err(MultiswapLayoutError::ProductWidthExceeded);
                }
                let mut value = [0; 33];
                value[..words.len()].copy_from_slice(words);
                // Fewer than 2^64 public entries, each <2^2048, fit 33 limbs.
                sums[row] = sums[row].wrapping_add(&Uint::from_words(value));
            }
            if sums.iter().any(|sum| !sum.ct_le(&limit).declassify()) {
                return Err(MultiswapLayoutError::ProductWidthExceeded);
            }
        }
        Ok(())
    }

    /// The private circuit builder emits each coordinate exactly once. Sort
    /// by public row and check that invariant before sparse projection.
    fn normalize(&mut self) {
        for matrix in [&mut self.a, &mut self.b, &mut self.c] {
            for column in matrix {
                column.sort_unstable_by_key(|(row, _)| *row);
                assert!(
                    column.windows(2).all(|pair| pair[0].0 != pair[1].0),
                    "MultiSwap circuit contains a duplicate coordinate"
                );
            }
        }
    }

    /// Layout shared with the committed bit tensor.
    pub const fn layout(&self) -> &MultiswapLayout {
        &self.layout
    }

    /// Number of live constraint rows.
    pub const fn live_rows(&self) -> usize {
        self.live_rows
    }

    /// Total stored coefficients across the three folded matrices.
    pub fn nnz(&self) -> usize {
        [&self.a, &self.b, &self.c]
            .iter()
            .map(|matrix| matrix.iter().map(Vec::len).sum::<usize>())
            .sum()
    }

    /// Projects the integer matrices modulo the runtime field.
    ///
    /// Coefficients that reduce to zero are dropped: the projected sparse
    /// matrix is exactly the matrix of the mod-q relation, and prover and
    /// verifier drop identically.
    pub fn project<F>(
        &self,
        field_config: &F::Config,
    ) -> Result<PreparedConstraintMatrices<F, F>, MultiswapLayoutError>
    where
        F: SpartanField,
    {
        let encoding = F::canonical_modulus_encoding(field_config);
        assert!(
            encoding[16..].iter().all(|&byte| byte == 0),
            "MultiSwap field exceeds two limbs"
        );
        let field = FpCtx::from_prime_u128(u128::from_le_bytes(encoding[..16].try_into().unwrap()));
        let project_matrix = |matrix: &Vec<Vec<(usize, usize)>>| {
            let columns = matrix
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .filter_map(|(row, value)| {
                            let residue = u128::from(field.to_integer(&public_coefficient(
                                &field,
                                &self.coefficients[*value],
                            )));
                            if residue == 0 {
                                None
                            } else {
                                Some((*row, F::from_with_cfg(residue, field_config)))
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            CscMatrix::try_from_columns(self.live_rows, columns)
        };
        let a = project_matrix(&self.a).map_err(SpartanMatrixError::from)?;
        let b = project_matrix(&self.b).map_err(SpartanMatrixError::from)?;
        let c = project_matrix(&self.c).map_err(SpartanMatrixError::from)?;
        let matrices = ConstraintMatrices::new(a, b, c)?;
        Ok(PreparedConstraintMatrices::new(matrices, field_config)?)
    }
}

/// The q-independent committed witness: block assignment values and the
/// packed BitZ bit rows.
#[derive(Clone, Debug)]
pub struct MultiswapAssignment {
    layout: MultiswapLayout,
    witness: Vec<Uint<32>>,
    quotients: Vec<Uint<32>>,
}

impl MultiswapAssignment {
    /// Materializes the block assignment from the circuit witness.
    pub fn new(circuit: &MultiswapCircuit) -> Result<Self, MultiswapLayoutError> {
        let layout = MultiswapLayout::new(circuit)?;
        Ok(Self {
            layout,
            witness: circuit.witness().to_vec(),
            quotients: circuit.quotients().to_vec(),
        })
    }

    /// Layout shared with the integer relation.
    pub const fn layout(&self) -> &MultiswapLayout {
        &self.layout
    }

    /// Borrowed declared-width blocks; the one and zero blocks are implicit.
    pub fn native(&self) -> NativeLimbWitness<'_> {
        NativeLimbWitness::new(self.layout.capacity, &self.witness, &self.quotients)
    }

    /// Reads one logical assignment entry, including structural padding.
    pub fn value(&self, index: usize) -> Uint<32> {
        self.native().read(index)
    }

    /// Builds the packed per-column BitZ bit rows.
    ///
    /// Column `c` packs, little-endian within each `u64` word, the
    /// `2^t` folded-row bits of every gate with low coordinates `c`: bit
    /// slot `j` of gate `g` lands at folded row `(j << h) | (g >> s)`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn bitz_bit_rows(&self) -> Vec<Vec<u64>> {
        let p = self.layout.bitz_params();
        let words_per_column = p.rows() / u64::BITS as usize;
        let mut rows = vec![vec![0u64; words_per_column]; p.cols()];
        // Valid MultiSwap layouts have exactly one high gate coordinate.
        debug_assert_eq!(self.layout.h, 1);
        let columns = p.cols();
        let write_column = |(column, output): (usize, &mut Vec<u64>)| {
            for (slot_start, block_start) in [
                (MULTISWAP_W_SLOT_START, self.layout.witness_block_start()),
                (
                    MULTISWAP_QUOS_SLOT_START,
                    self.layout.quotient_block_start(),
                ),
            ] {
                let even = self.value(block_start + column);
                let odd = self.value(block_start + column + columns);
                let offset = 2 * slot_start / 64;
                for (limb, (&a, &b)) in even.as_words().iter().zip(odd.as_words()).enumerate() {
                    let pair = crate::utils::bit_packing::interleave_words(a, b);
                    output[offset + 2 * limb..offset + 2 * limb + 2].copy_from_slice(&pair);
                }
            }
        };
        #[cfg(feature = "parallel")]
        if columns >= 512 && rayon::current_num_threads() > 1 {
            rows.par_iter_mut()
                .enumerate()
                .with_min_len(64)
                .for_each(write_column);
            return rows;
        }
        rows.iter_mut().enumerate().for_each(write_column);
        rows
    }

    /// Project the original declared-width blocks once after the prime draw.
    /// Storage is canonical Montgomery residues, as required by RawWitness::Field.
    pub(crate) fn projected_assignment(&self, field: &FpCtx<2>) -> Vec<u128> {
        use field::RingOps;
        let mut out = vec![0; self.layout.assignment_len()];
        out[0] = super::super::raw_monty::raw_shared(field.one());
        let (_, tail) = out.split_at_mut(self.layout.witness_block_start());
        let (witness, tail) = tail.split_at_mut(self.layout.capacity);
        let (quotients, _) = tail.split_at_mut(self.layout.capacity);
        project_unsigned_blocks(
            field,
            [(&self.witness, witness), (&self.quotients, quotients)],
        );
        out
    }

    /// Exact row operands for the generic outer sumcheck. The relation checks
    /// the coefficient-sum bound once using public data during preparation.
    pub fn integer_products(
        &self,
        relation: &MultiswapIntegerRelation,
    ) -> crate::sumcheck::outer::OuterInputs<Uint<64>> {
        assert_eq!(self.layout, relation.layout);
        let product = |matrix: &Vec<Vec<(usize, usize)>>| {
            let mut out = vec![Uint::<64>::ZERO; self.layout.capacity];
            for (column, entries) in matrix.iter().enumerate() {
                if entries.is_empty() {
                    continue;
                } // Public sparse structure.
                let value = self.value(column);
                for &(row, coefficient) in entries {
                    let term = super::circuit::public_coefficient_product(
                        &relation.coefficients[coefficient],
                        &value,
                    );
                    let words = core::array::from_fn(|i| term.as_words()[i]);
                    // All summands are nonnegative; the checked public bound
                    // covers every partial sum as well as the completed row.
                    out[row] = out[row].wrapping_add(&Uint::from_words(words));
                }
            }
            out
        };
        crate::sumcheck::outer::OuterInputs {
            ax: product(&relation.a),
            bx: product(&relation.b),
            cx: product(&relation.c),
        }
    }

    /// Independent projected reference, retained only for differential tests.
    #[cfg(test)]
    pub fn project<F>(
        &self,
        relation: &MultiswapIntegerRelation,
        field_config: &F::Config,
    ) -> Result<(DenseMultilinearExtension<F>, R1csProductMles<F>), MultiswapLayoutError>
    where
        F: SpartanField,
    {
        let modulus = field_modulus(F::canonical_modulus_encoding(field_config));
        let zero = F::zero_with_cfg(field_config);
        let field_assignment: Vec<F> = (0..self.layout.assignment_len())
            .map(|index| {
                let bytes: Vec<_> = self
                    .value(index)
                    .as_words()
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                F::from_with_cfg(
                    reduce_biguint(&BigUint::from_bytes_le(&bytes), &modulus),
                    field_config,
                )
            })
            .collect();

        let product = |matrix: &Vec<Vec<(usize, usize)>>| -> Vec<F> {
            let mut out = vec![zero.clone(); relation.live_rows()];
            for (column, entries) in matrix.iter().enumerate() {
                if entries.is_empty() || F::is_zero(&field_assignment[column]) {
                    continue;
                }
                let z = &field_assignment[column];
                for (row, value) in entries {
                    let residue =
                        reduce_biguint(&test_biguint(&relation.coefficients[*value]), &modulus);
                    if residue == 0 {
                        continue;
                    }
                    let mut term = F::from_with_cfg(residue, field_config);
                    term = field_config.mul(&(term), &(z));
                    out[*row] = field_config.add(&(out[*row]), &(&term));
                }
            }
            out
        };

        let az = product(&relation.a);
        let bz = product(&relation.b);
        let cz = product(&relation.c);
        let products = build_product_mles(&az, &bz, &cz, relation.live_rows(), field_config)?;
        let assignment = build_assignment_mle(
            &field_assignment,
            self.layout.assignment_len(),
            field_config,
        )?;
        Ok((assignment, products))
    }
}

// Public coefficients retain their source's declared width through preparation.
fn public_coefficient(field: &FpCtx<2>, value: &[u64]) -> Fp<2> {
    field.from_integer(&UintRef::new(value))
}
/// Canonical `u128` runtime modulus recovered from its field encoding.
#[cfg(test)]
fn field_modulus(encoding: Vec<u8>) -> BigUint {
    BigUint::from_bytes_le(&encoding)
}

/// Reduces a nonnegative integer modulo the runtime prime into `u128`.
/// One unsigned reduction preparation shared by every declared-width block.
fn project_unsigned_blocks<const N: usize, const BLOCKS: usize>(
    field: &FpCtx<2>,
    blocks: [(&[Uint<N>], &mut [u128]); BLOCKS],
) {
    use field::{PreparedLinearCombination, RingOps};
    let prepared = <FpCtx<2> as PreparedLinearCombination<Uint<N>>>::prepare_linear_combination(
        field,
        [field.one()],
    );
    for (values, output) in blocks {
        assert!(values.len() <= output.len());
        let project = |(value, output): (&Uint<N>, &mut u128)| {
            let projected = <FpCtx<2> as PreparedLinearCombination<Uint<N>>>::linear_combination(
                &prepared,
                |_| *value,
            );
            *output = super::super::raw_monty::raw_shared(projected);
        };
        #[cfg(feature = "parallel")]
        if values.len() >= 4096 && rayon::current_num_threads() > 1 {
            values
                .par_iter()
                .zip(output.par_iter_mut())
                .with_min_len(256)
                .for_each(project);
            continue;
        }
        values.iter().zip(output.iter_mut()).for_each(project);
    }
}

#[allow(clippy::arithmetic_side_effects)]
#[cfg(test)]
fn reduce_biguint(value: &BigUint, modulus: &BigUint) -> u128 {
    let reduced = value % modulus;
    let digits = reduced.iter_u64_digits().collect::<Vec<_>>();
    match digits.len() {
        0 => 0,
        1 => u128::from(digits[0]),
        2 => u128::from(digits[0]) | (u128::from(digits[1]) << 64),
        _ => unreachable!("a residue modulo a 128-bit prime has at most two u64 digits"),
    }
}

#[cfg(test)]
fn test_biguint(words: &[u64]) -> BigUint {
    BigUint::from_bytes_le(
        &words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {

    use super::super::circuit::MultiswapDims;
    use super::*;

    fn mini() -> (
        MultiswapCircuit,
        MultiswapIntegerRelation,
        MultiswapAssignment,
    ) {
        let circuit = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
        let relation = MultiswapIntegerRelation::new(&circuit).unwrap();
        let assignment = MultiswapAssignment::new(&circuit).unwrap();
        (circuit, relation, assignment)
    }

    fn test_config() -> <Fp<2> as crate::piop::spartan::SpartanField>::Config {
        // The fixed BitZ evaluation prime is a convenient valid runtime field.
        Fp::<2>::make_cfg(&Uint::from(crate::pcs::FQ_MOD)).unwrap()
    }

    #[test]
    fn every_batch_embedding_preserves_the_canonical_linear_forms() {
        use std::collections::BTreeMap;
        for batch in [1, 2, 4, 8, 16] {
            let circuit =
                MultiswapCircuit::build_batch(MultiswapDims::multiswap(0), batch).unwrap();
            let relation = MultiswapIntegerRelation::new(&circuit).unwrap();
            let layout = relation.layout();
            for (is_c, source, embedded) in [
                (false, circuit.a_entries(), &relation.a),
                (false, circuit.b_entries(), &relation.b),
                (true, circuit.c_entries(), &relation.c),
            ] {
                let mut expected = BTreeMap::<(usize, usize), BigUint>::new();
                for (row, col, value) in source.iter() {
                    *expected.entry((row, col)).or_default() += test_biguint(value);
                }
                expected.retain(|_, v| !v.is_zero());
                let mut recovered = BTreeMap::<(usize, usize), BigUint>::new();
                let mut quotient_rows = Vec::new();
                for (column, entries) in embedded.iter().enumerate() {
                    for (row, value) in entries {
                        if column >= layout.quotient_block_start() {
                            assert!(is_c);
                            assert_eq!(column - layout.quotient_block_start(), *row);
                            assert!(*row < circuit.live_rows());
                            assert_eq!(&relation.coefficients[*value], &circuit.mods()[*row]);
                            quotient_rows.push(*row);
                        } else {
                            let source_column = if column == 0 {
                                circuit.const_col()
                            } else {
                                assert!(column >= layout.witness_block_start());
                                column - layout.witness_block_start()
                            };
                            *recovered.entry((*row, source_column)).or_default() +=
                                test_biguint(&relation.coefficients[*value]);
                        }
                    }
                }
                recovered.retain(|_, v| !v.is_zero());
                assert_eq!(recovered, expected);
                if is_c {
                    quotient_rows.sort_unstable();
                    assert_eq!(
                        quotient_rows,
                        (0..circuit.live_rows())
                            .filter(|r| circuit.mods()[*r].iter().any(|&word| word != 0))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn layout_keeps_one_chunk_geometry() {
        let (_, relation, _) = mini();
        let layout = *relation.layout();
        let p = layout.bitz_params();
        assert_eq!(p.word_bits, 1);
        assert!(p.row_vars <= 13);
        assert_eq!(
            p.row_vars + p.col_vars,
            MULTISWAP_SLOT_VARS + layout.gate_vars()
        );
        assert_eq!(
            p.cells(),
            MULTISWAP_SLOTS * layout.capacity(),
            "one bit cell per committed slot"
        );
    }

    #[test]
    fn bit_rows_reconstruct_the_assignment_values() {
        let (_, _, assignment) = mini();
        let layout = *assignment.layout();
        let p = layout.bitz_params();
        let rows = assignment.bitz_bit_rows();
        assert_eq!(rows.len(), p.cols());
        assert!(rows.iter().all(|row| row.len() == p.rows() / 64));

        for gate in 0..layout.capacity() {
            for (slot_start, block_start) in [
                (MULTISWAP_W_SLOT_START, layout.witness_block_start()),
                (MULTISWAP_QUOS_SLOT_START, layout.quotient_block_start()),
            ] {
                let mut reconstructed = BigUint::zero();
                for slot in 0..MULTISWAP_VALUE_BITS {
                    let (row, column) = layout.bitz_bit_position(slot_start + slot, gate).unwrap();
                    let bit = (rows[column][row / 64] >> (row % 64)) & 1;
                    if bit == 1 {
                        reconstructed.set_bit(slot as u64, true);
                    }
                }
                assert_eq!(
                    reconstructed,
                    BigUint::from_bytes_le(
                        &assignment
                            .value(block_start + gate)
                            .as_words()
                            .iter()
                            .flat_map(|w| w.to_le_bytes())
                            .collect::<Vec<_>>()
                    )
                );
            }
        }
    }

    fn synthetic_assignment(capacity: usize) -> MultiswapAssignment {
        let gate_vars = capacity.trailing_zeros() as usize;
        let values = |length| {
            (0..length)
                .map(|gate| {
                    Uint::from_words(core::array::from_fn(|limb| match gate % 5 {
                        0 => 0,
                        1 => u64::MAX,
                        2 => 0xaaaa_5555_aaaa_5555,
                        3 => 1u64 << (limb % 64),
                        _ => (gate as u64)
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            .rotate_left(limb as u32),
                    }))
                })
                .collect()
        };
        MultiswapAssignment {
            layout: MultiswapLayout {
                capacity,
                gate_vars,
                s: gate_vars - 1,
                h: 1,
            },
            witness: values(capacity - 1),
            quotients: values(capacity / 2 + 1),
        }
    }

    #[test]
    fn word_packing_preserves_patterns_and_structural_padding() {
        let assignment = synthetic_assignment(16);
        let layout = assignment.layout();
        let rows = assignment.bitz_bit_rows();
        for gate in 0..layout.capacity() {
            for (slot_start, block_start) in [
                (MULTISWAP_W_SLOT_START, layout.witness_block_start()),
                (MULTISWAP_QUOS_SLOT_START, layout.quotient_block_start()),
            ] {
                let value = assignment.value(block_start + gate);
                for bit in 0..MULTISWAP_VALUE_BITS {
                    let (row, column) = layout.bitz_bit_position(slot_start + bit, gate).unwrap();
                    assert_eq!(
                        (rows[column][row / 64] >> (row % 64)) & 1,
                        (value.as_words()[bit / 64] >> (bit % 64)) & 1
                    );
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn word_packing_matches_across_parallel_threshold() {
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        for columns in [256, 512, 1024, 4096, 8192] {
            let assignment = synthetic_assignment(2 * columns);
            assert_eq!(
                serial.install(|| assignment.bitz_bit_rows()),
                parallel.install(|| assignment.bitz_bit_rows())
            );
        }
    }

    fn check_unsigned_projection<const N: usize>(field: &FpCtx<2>) {
        let values = [
            Uint::<N>::ZERO,
            Uint::<N>::ONE,
            Uint::from_words([u64::MAX; N]),
            Uint::from_words(core::array::from_fn(
                |i| if i + 1 == N { 1 << 63 } else { 0 },
            )),
        ];
        let mut output = [0; 6];
        project_unsigned_blocks(field, [(&values, &mut output)]);
        for (value, raw) in values.iter().zip(output) {
            let expected = field.from_integer(value);
            assert_eq!(
                super::super::super::raw_monty::shared_raw(field, raw),
                expected
            );
        }
        assert_eq!(&output[4..], &[0, 0]);
    }

    #[test]
    fn prepared_unsigned_projection_preserves_high_bits_and_padding() {
        for modulus in [crate::pcs::FQ_MOD, 260337761016399727832017529560925459939] {
            let field = Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
            check_unsigned_projection::<1>(&field);
            check_unsigned_projection::<2>(&field);
            check_unsigned_projection::<4>(&field);
            check_unsigned_projection::<32>(&field);
            check_unsigned_projection::<64>(&field);
            let assignment = synthetic_assignment(16);
            let projected = assignment.projected_assignment(&field);
            for (index, raw) in projected.into_iter().enumerate() {
                let expected = field.from_integer(&assignment.value(index));
                assert_eq!(
                    super::super::super::raw_monty::shared_raw(&field, raw),
                    expected
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn witness_projection_matches_worker_counts() {
        let assignment = synthetic_assignment(8192);
        let field = test_config();
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        assert_eq!(
            serial.install(|| assignment.projected_assignment(&field)),
            parallel.install(|| assignment.projected_assignment(&field))
        );
    }

    #[test]
    fn witness_first_preserves_products_piop_and_transcript() {
        use crate::piop::spartan::{
            matrix::products_from_montgomery_assignment,
            piop::prove_spartan_piop_raw_products_raw_witness, raw_monty::RawWitness,
        };
        use crate::transcript::Blake3Transcript;
        let (_, relation, assignment) = mini();
        for modulus in [crate::pcs::FQ_MOD, 260337761016399727832017529560925459939] {
            let field = Fp::<2>::make_cfg(&Uint::from(modulus)).unwrap();
            let matrices = relation.project::<Fp<2>>(&field).unwrap();
            let raw = assignment.projected_assignment(&field);
            let products = products_from_montgomery_assignment(&matrices, &raw).unwrap();
            let integers = assignment.integer_products(&relation);
            for (projected, exact) in [
                (&products.ax, &integers.ax),
                (&products.bx, &integers.bx),
                (&products.cx, &integers.cx),
            ] {
                for (projected, exact) in projected.iter().zip(exact) {
                    assert_eq!(*projected, field.from_integer(exact));
                }
            }
            let mut old_transcript = Blake3Transcript::new();
            let mut new_transcript = Blake3Transcript::new();
            let old = prove_spartan_piop_raw_products_raw_witness(
                &mut old_transcript,
                &matrices,
                &[7; 32],
                integers,
                RawWitness::Limbs(assignment.native()),
            )
            .unwrap();
            let new = prove_spartan_piop_raw_products_raw_witness(
                &mut new_transcript,
                &matrices,
                &[7; 32],
                products,
                RawWitness::Field(raw),
            )
            .unwrap();
            assert_eq!(old, new);
            assert_eq!(old_transcript.state_digest(), new_transcript.state_digest());
        }
    }

    #[test]
    fn projected_relation_is_satisfied_row_by_row() {
        let (_, relation, assignment) = mini();
        let config = test_config();
        let matrices = relation.project::<Fp<2>>(&config).unwrap();
        let (mle, products) = assignment.project::<Fp<2>>(&relation, &config).unwrap();

        assert_eq!(mle.evaluations.len(), relation.layout().assignment_len());
        assert_eq!(matrices.matrices().row_count(), relation.live_rows());
        assert_eq!(
            matrices.matrices().column_count(),
            relation.layout().assignment_len()
        );
        for row in 0..relation.live_rows() {
            let mut left = products.az.evaluations[row].clone();
            left = config.mul(&(left), &(&products.bz.evaluations[row]));
            assert_eq!(left, products.cz.evaluations[row], "row {row}");
        }
    }

    #[test]
    fn exact_product_width_checks_public_coefficient_sum_boundaries() {
        let (_, mut relation, _) = mini();
        for matrix in [&mut relation.a, &mut relation.b, &mut relation.c] {
            matrix.iter_mut().for_each(Vec::clear);
        }
        relation.coefficients = UnsignedIntegerTable::default();
        relation
            .coefficients
            .push(Uint::<32>::from_words([u64::MAX; 32]));
        relation.coefficients.push(Uint::<1>::ONE);
        // (2^2048 - 1) + 1 reaches the inclusive public bound exactly.
        relation.a[0].push((0, 0));
        relation.a[1].push((0, 1));
        relation.validate_product_width().unwrap();

        relation.a[2].push((0, 1));
        assert!(matches!(
            relation.validate_product_width(),
            Err(MultiswapLayoutError::ProductWidthExceeded)
        ));

        for column in &mut relation.a {
            column.clear();
        }
        relation.coefficients.push(Uint::<33>::ONE);
        relation.a[0].push((0, 2));
        assert!(matches!(
            relation.validate_product_width(),
            Err(MultiswapLayoutError::ProductWidthExceeded)
        ));
    }

    #[test]
    fn exact_products_match_bigint_for_valid_and_maximum_assignments() {
        let (_, relation, mut assignment) = mini();
        let big = |words: &[u64]| {
            BigUint::from_bytes_le(
                &words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        for maximum in [false, true] {
            if maximum {
                assignment.witness.fill(Uint::from_words([u64::MAX; 32]));
                assignment.quotients.fill(Uint::from_words([u64::MAX; 32]));
            }
            let exact = assignment.integer_products(&relation);
            for (matrix, actual) in [
                (&relation.a, &exact.ax),
                (&relation.b, &exact.bx),
                (&relation.c, &exact.cx),
            ] {
                let mut expected = vec![BigUint::zero(); relation.layout.capacity];
                for (column, entries) in matrix.iter().enumerate() {
                    for &(row, coefficient) in entries {
                        expected[row] += big(&relation.coefficients[coefficient])
                            * big(assignment.value(column).as_words());
                    }
                }
                for (value, expected) in actual.iter().zip(expected) {
                    assert_eq!(big(value.as_words()), expected);
                }
            }
        }
    }

    #[test]
    fn tampering_one_quotient_breaks_the_projected_relation() {
        let (mut circuit, _, _) = {
            let circuit = MultiswapCircuit::build(MultiswapDims::mini()).unwrap();
            (circuit.clone(), (), ())
        };
        // Perturb one live quotient: the folded C' row must move.
        let target = 0;
        let mut quos = circuit.quotients().to_vec();
        quos[target] = quos[target].wrapping_add(&field::Uint::ONE);
        circuit = tampered_with_quotients(circuit, quos);

        let relation = MultiswapIntegerRelation::new(&circuit).unwrap();
        let assignment = MultiswapAssignment::new(&circuit).unwrap();
        let config = test_config();
        let (_, products) = assignment.project::<Fp<2>>(&relation, &config).unwrap();
        let mut left = products.az.evaluations[target].clone();
        left = config.mul(&(left), &(&products.bz.evaluations[target]));
        assert_ne!(left, products.cz.evaluations[target]);
    }

    fn tampered_with_quotients(
        circuit: MultiswapCircuit,
        quos: Vec<field::Uint<32>>,
    ) -> MultiswapCircuit {
        circuit.with_quotients_for_tests(quos)
    }
}
