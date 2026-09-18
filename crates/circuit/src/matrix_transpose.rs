//! Materialization and multiplication of the transposed Boolean matrix.
//!
//! [`MTransposeGenerator`] builds a compact row-major representation of `M^T`
//! from circuit structure alone. [`CscMatrix`] computes `r * M`
//! as parallel, disjoint column gathers over the GHASH field.

use crate::linear_map::{CscMatrix, ImplicitOnes};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use field::Gf128;

use crate::witgen::Z;
use crate::{BoolWitness, Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

const MATRIX_INLINE_SUPPORT: usize = 4;
const MATRIX_ARENA_SUPPORT: u8 = u8::MAX;
static NEXT_MATRIX_ID: AtomicU64 = AtomicU64::new(1);

/// A canonical sparse F2 expression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixBit {
    constant_term: bool,
    support_len: u8,
    support: [u32; MATRIX_INLINE_SUPPORT],
    matrix_id: u32,
}

impl MatrixBit {
    fn constant(value: bool) -> Self {
        let mut support = [0; MATRIX_INLINE_SUPPORT];
        support[0] = 0;
        Self {
            constant_term: value,
            support_len: u8::from(value),
            support,
            matrix_id: 0,
        }
    }

    fn inline(support: &[u32], matrix_id: u32) -> Self {
        debug_assert!(support.len() <= MATRIX_INLINE_SUPPORT);
        let mut inline = [0; MATRIX_INLINE_SUPPORT];
        inline[..support.len()].copy_from_slice(support);
        Self {
            constant_term: support.first() == Some(&0),
            support_len: support.len() as u8,
            support: inline,
            matrix_id,
        }
    }

    const fn constant_term(&self) -> bool {
        self.constant_term
    }
}

impl From<bool> for MatrixBit {
    fn from(value: bool) -> Self {
        Self::constant(value)
    }
}

impl BoolWitness for MatrixBit {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
}

fn symmetric_difference(left: &[u32], right: &[u32], output: &mut [u32]) -> usize {
    let (mut left_index, mut right_index, mut output_index) = (0, 0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => {
                output[output_index] = left[left_index];
                left_index += 1;
                output_index += 1;
            }
            std::cmp::Ordering::Greater => {
                output[output_index] = right[right_index];
                right_index += 1;
                output_index += 1;
            }
            std::cmp::Ordering::Equal => {
                left_index += 1;
                right_index += 1;
            }
        }
    }
    for &entry in &left[left_index..] {
        output[output_index] = entry;
        output_index += 1;
    }
    for &entry in &right[right_index..] {
        output[output_index] = entry;
        output_index += 1;
    }
    output_index
}

/// Row recorder finalized into CSC for `M` (row-major `M^T`).
#[derive(Debug)]
pub(crate) struct MTransposeRecorder {
    id: u32,
    witness_count: usize,
    expression_offsets: Vec<u32>,
    expression_entries: Vec<u32>,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
}

impl MTransposeRecorder {
    pub(crate) fn with_witnesses(witness_count: usize) -> Self {
        assert!(
            witness_count < u32::MAX as usize,
            "too many Boolean witnesses for M"
        );
        let id = u32::try_from(NEXT_MATRIX_ID.fetch_add(1, Ordering::Relaxed))
            .expect("matrix identifier overflow");
        assert_ne!(id, 0, "matrix identifier overflow");
        Self {
            id,
            witness_count,
            expression_offsets: vec![0],
            expression_entries: Vec::new(),
            row_offsets: vec![0, 1],
            column_indices: vec![0],
        }
    }

    pub(crate) fn witness(&self, index: usize) -> MatrixBit {
        assert!(
            index < self.witness_count,
            "Boolean witness is not allocated"
        );
        MatrixBit {
            constant_term: false,
            support_len: 1,
            support: [
                u32::try_from(index + 1).expect("too many Boolean witnesses for M"),
                0,
                0,
                0,
            ],
            matrix_id: self.id,
        }
    }

    pub(crate) fn allocate_witness(&mut self) -> MatrixBit {
        let index = self.witness_count;
        self.witness_count = self
            .witness_count
            .checked_add(1)
            .filter(|count| *count < u32::MAX as usize)
            .expect("too many Boolean witnesses for M");
        self.witness(index)
    }

    fn check_owner(&self, value: MatrixBit) {
        assert!(
            value.matrix_id == 0 || value.matrix_id == self.id,
            "Boolean expression belongs to another matrix recorder"
        );
    }

    fn support<'a>(&'a self, value: &'a MatrixBit) -> &'a [u32] {
        if value.support_len == MATRIX_ARENA_SUPPORT {
            let index = value.support[0] as usize;
            let start = self.expression_offsets[index] as usize;
            let end = self.expression_offsets[index + 1] as usize;
            &self.expression_entries[start..end]
        } else {
            &value.support[..usize::from(value.support_len)]
        }
    }

    fn bit_from_support(&mut self, support: &[u32]) -> MatrixBit {
        if support.len() <= MATRIX_INLINE_SUPPORT {
            return MatrixBit::inline(support, self.id);
        }
        let expression = self.expression_offsets.len() - 1;
        self.expression_entries.extend_from_slice(support);
        self.expression_offsets.push(
            u32::try_from(self.expression_entries.len())
                .expect("too many Boolean-expression entries for M"),
        );
        MatrixBit {
            constant_term: support.first() == Some(&0),
            support_len: MATRIX_ARENA_SUPPORT,
            support: [
                u32::try_from(expression).expect("too many Boolean expressions for M"),
                0,
                0,
                0,
            ],
            matrix_id: self.id,
        }
    }

    pub(crate) fn xor(&mut self, lhs: MatrixBit, rhs: MatrixBit) -> MatrixBit {
        self.check_owner(lhs);
        self.check_owner(rhs);
        if lhs.support_len == 0 {
            return rhs;
        }
        if rhs.support_len == 0 {
            return lhs;
        }

        let left = self.support(&lhs);
        let right = self.support(&rhs);
        if left.len() + right.len() <= 2 * MATRIX_INLINE_SUPPORT {
            let mut merged = [0; 2 * MATRIX_INLINE_SUPPORT];
            let len = symmetric_difference(left, right, &mut merged);
            return self.bit_from_support(&merged[..len]);
        }

        let mut merged = vec![0; left.len() + right.len()];
        let len = symmetric_difference(left, right, &mut merged);
        merged.truncate(len);
        self.bit_from_support(&merged)
    }

    pub(crate) fn push_row(&mut self, value: &MatrixBit) {
        self.check_owner(*value);
        if value.support_len == MATRIX_ARENA_SUPPORT {
            let index = value.support[0] as usize;
            let start = self.expression_offsets[index] as usize;
            let end = self.expression_offsets[index + 1] as usize;
            self.column_indices
                .extend_from_slice(&self.expression_entries[start..end]);
        } else {
            self.column_indices
                .extend_from_slice(&value.support[..usize::from(value.support_len)]);
        }
        self.row_offsets
            .push(u32::try_from(self.column_indices.len()).expect("too many nonzeros in M"));
    }

    pub(crate) fn finish(self) -> CscMatrix<ImplicitOnes, u32> {
        let nnz = self.column_indices.len();
        crate::linear_map::CsrMatrix::try_from_parts(
            self.witness_count + 1,
            self.row_offsets,
            self.column_indices,
            ImplicitOnes::new(nnz),
        )
        .expect("canonical recorded binary rows")
        .into_csc()
        .expect("compact recorded indices")
    }
}

/// Materializes compact `M^T` from a circuit's static Boolean structure.
#[derive(Debug)]
pub struct MTransposeGenerator {
    recorder: MTransposeRecorder,
    inputs: Box<[MatrixBit]>,
}

impl MTransposeGenerator {
    /// Creates a matrix generator with preallocated Boolean input columns.
    pub fn new(input_count: usize) -> Self {
        let recorder = MTransposeRecorder::with_witnesses(input_count);
        let inputs = (0..input_count)
            .map(|index| recorder.witness(index))
            .collect();
        Self { recorder, inputs }
    }

    /// Moves every input into a fixed-size boxed array without cloning.
    pub fn take_boxed_inputs<const N: usize>(&mut self) -> Box<[MatrixBit; N]> {
        assert_eq!(N, self.inputs.len(), "input witness count mismatch");
        std::mem::take(&mut self.inputs)
            .try_into()
            .unwrap_or_else(|_| unreachable!("input length was checked"))
    }

    /// Moves dynamically sized input handles out without cloning.
    pub fn take_inputs(&mut self) -> Box<[MatrixBit]> {
        std::mem::take(&mut self.inputs)
    }

    /// Finishes the compact transposed Boolean matrix.
    pub fn finish(self) -> CscMatrix<ImplicitOnes, u32> {
        self.recorder.finish()
    }
}

impl Circuit for MTransposeGenerator {
    type Bool = MatrixBit;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = Z<LIMBS>;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Z<LIMBS> {
        crate::witgen::integer_from_words(words)
    }

    fn xor(&mut self, lhs: MatrixBit, rhs: MatrixBit) -> MatrixBit {
        self.recorder.xor(lhs, rhs)
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<MatrixBit, N>
    where
        H: Fn(&dyn WitnessContext<Z<LIMBS>, MatrixBit, Z<LIMBS>>) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        ScalarBits(std::array::from_fn(|_| self.recorder.allocate_witness()))
    }

    fn bitz<const LIMBS: usize>(&mut self, value: MatrixBit) -> Z<LIMBS> {
        self.recorder.push_row(&value);
        Z::from(u64::from(value.constant_term()))
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        bits_le: &<MatrixBit as BoolWitness>::Repr<N, M>,
    ) -> (Z<LIMBS>, Z<LIMBS>) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        let values: [bool; N] = std::array::from_fn(|index| {
            let bit = &bits_le.0[index];
            self.recorder.push_row(bit);
            bit.constant_term()
        });
        (
            crate::witgen::integer_from_bits(&values),
            crate::witgen::integer_from_bits(&values[..LOW]),
        )
    }

    fn assert_r1c<const LIMBS: usize>(&mut self, _: Z<LIMBS>, _: Z<LIMBS>, _: Z<LIMBS>) {}

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: Z<FROM_LIMBS>,
    ) -> Z<TO_LIMBS> {
        value.sign_extend()
    }
}

#[cfg(test)]
mod tests {
    use crate::constraints::ConstraintGenerator;
    use crate::sha256::{COMPRESSION_HINT_BITS, COMPRESSION_INPUT_BITS, compression_circuit};
    use crate::{BoolRepresentation, BoolWitness, Circuit};

    use super::*;

    fn example_circuit<CS: Circuit>(circuit: &mut CS, inputs: &[CS::Bool; 3]) {
        let xy = circuit.xor(inputs[0].clone(), inputs[1].clone());
        let not_xy = circuit.xor(xy.clone(), CS::Bool::from(true));
        let _ = circuit.bitz::<1>(xy);
        let _ = circuit.bitz::<1>(not_xy);

        let captured = inputs[2].clone();
        let hinted = circuit.hint::<1, 2, 1, _>(move |context| {
            let value = context.eval_bool(&captured);
            Ok(crate::PackedBits::from_array([value, !value]))
        });
        let hinted_zero =
            <<<CS as Circuit>::Bool as BoolWitness>::Repr<2, 1> as BoolRepresentation<
                CS::Bool,
                2,
                1,
            >>::bit(&hinted, 0);
        let mixed = circuit.xor(hinted_zero, inputs[0].clone());
        let _ = circuit.bitz::<1>(mixed);
        let _: (CS::Z<1>, CS::Z<1>) = circuit.bitz_unsigned::<1, 2, 1, 1>(&hinted);
    }

    fn challenges(count: usize) -> Vec<Gf128> {
        (0..count)
            .map(|index| {
                Gf128::new(
                    (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03) ^ 0xa5a5,
                )
            })
            .collect()
    }

    #[test]
    fn materialized_transpose_matches_constraint_generation() {
        let mut materializer = MTransposeGenerator::new(3);
        let inputs = materializer.take_boxed_inputs();
        example_circuit(&mut materializer, &inputs);
        let transpose = materializer.finish();

        let mut generator = ConstraintGenerator::new(3);
        let symbolic = generator.inputs();
        example_circuit(&mut generator, &symbolic);
        let matrices = generator.into_matrices();
        let r = challenges(matrices.m.row_count());

        let mut expected = vec![Gf128::new(0, 0); matrices.m.column_count()];
        for (row, challenge) in matrices.m.rows().zip(&r) {
            for &column in row.indices() {
                expected[column] += *challenge;
            }
        }

        assert_eq!(transpose.row_count(), matrices.m.row_count());
        assert_eq!(transpose.column_count(), matrices.m.column_count());
        assert_eq!(transpose.mul_left(&r).unwrap(), expected);
    }

    #[test]
    fn materializes_a_full_sha256_compression_from_structure() {
        let mut materializer = MTransposeGenerator::new(COMPRESSION_INPUT_BITS);
        let matrix_inputs = materializer.take_boxed_inputs();
        let _ = compression_circuit(&mut materializer, &matrix_inputs);

        let transpose = materializer.finish();
        assert_eq!(
            transpose.column_count(),
            COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS + 1
        );
        assert_eq!(transpose.row_count(), 20_457);
        assert_eq!(transpose.nnz(), 42_361);
        assert!(transpose.topology_bytes() < 194 * 1024);
    }

    #[test]
    fn parallel_and_sequential_matrix_gathers_agree() {
        const WIDTH: usize = 4096;
        let mut recorder = MTransposeRecorder::with_witnesses(WIDTH);
        for index in 0..WIDTH / 2 {
            let left = recorder.witness(index);
            let right = recorder.witness(index + WIDTH / 2);
            let root = recorder.xor(left, right);
            recorder.push_row(&root);
        }
        let transpose = recorder.finish();
        let r = challenges(transpose.row_count());
        let mut sequential = vec![Gf128::new(0, 0); transpose.column_count()];
        let mut parallel = vec![Gf128::new(0, 0); transpose.column_count()];
        transpose
            .mul_left_kernel(&r, &mut sequential, false)
            .unwrap();
        transpose.mul_left_kernel(&r, &mut parallel, true).unwrap();
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn materialized_supports_can_exceed_the_inline_capacity() {
        let mut recorder = MTransposeRecorder::with_witnesses(6);
        let mut expression = MatrixBit::from(false);
        for index in 0..6 {
            expression = recorder.xor(expression, recorder.witness(index));
        }
        recorder.push_row(&expression);
        let transpose = recorder.finish();
        let challenge = Gf128::new(7, 11);
        let product = transpose.mul_left(&[Gf128::new(3, 5), challenge]).unwrap();

        assert_eq!(transpose.nnz(), 7);
        assert_eq!(product[0], Gf128::new(3, 5));
        assert!(product[1..].iter().all(|value| *value == challenge));
    }

    #[test]
    fn apply_rejects_the_wrong_challenge_length() {
        let transpose = MTransposeRecorder::with_witnesses(0).finish();
        assert_eq!(
            transpose.mul_left(&[]),
            Err(crate::linear_map::LinearMapError::Length {
                kind: "row weights",
                expected: 1,
                actual: 0,
            })
        );
    }
}
