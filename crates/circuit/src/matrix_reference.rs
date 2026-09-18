//! Materialized sparse integer matrices for `r * (A + x B + x^2 C)`.
//!
//! [`MaterializedAbc`] retains every nonzero coefficient as an exact signed
//! integer until the runtime modulus is known. It stores the three matrices
//! together in column-major order so a prepared evaluator can gather each
//! output column independently and in parallel.

use field::ModRingCtx;

use std::error::Error;
use std::fmt::{self, Display};
use std::mem::size_of;

use field::{FpCtx, IntegerEmbedding, Uint, create_prime_field};
use rayon::prelude::*;

use crate::constraints::ConstraintMatrices;
use crate::integer_storage::IntegerTable;
use crate::linear_map::CsrMatrix;
use crate::linear_map::circuit::{add_representatives, mul_representatives, neg_representative};

const PARALLEL_NNZ_THRESHOLD: usize = 1 << 15;
const PARALLEL_VECTOR_THRESHOLD: usize = 1 << 14;
const ROW_MASK: u32 = (1 << 28) - 1;
const KIND_SHIFT: u32 = 28;
const CLASS_SHIFT: u32 = 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
enum MatrixKind {
    A = 0,
    B = 1,
    C = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
enum CoefficientClass {
    General = 0,
    One = 1,
    NegativeOne = 2,
}

#[derive(Debug)]
struct IntegerEntry {
    coordinate: u32,
    coefficient: usize,
    kind: MatrixKind,
}

impl IntegerEntry {
    fn new(row: usize, kind: MatrixKind, coefficient: usize, words: &[u64]) -> Self {
        assert!(row <= ROW_MASK as usize, "too many sparse matrix rows");
        let class = if words[0] == 1 && words[1..].iter().all(|&w| w == 0) {
            CoefficientClass::One
        } else if words.iter().all(|&w| w == u64::MAX) {
            CoefficientClass::NegativeOne
        } else {
            CoefficientClass::General
        };
        Self {
            coordinate: u32::try_from(row).expect("too many sparse matrix rows")
                | (kind as u32) << KIND_SHIFT
                | (class as u32) << CLASS_SHIFT,
            coefficient,
            kind,
        }
    }
}

/// Column-major materialization of sparse integer `A`, `B`, and `C`.
#[derive(Debug)]
pub struct MaterializedAbc {
    row_count: usize,
    column_offsets: Box<[u32]>,
    coordinates: Box<[u32]>,
    coefficients: IntegerTable,
    matrix_nonzeros: [usize; 3],
}

impl MaterializedAbc {
    /// Transposes and stores the integer matrices without choosing a modulus.
    pub fn from_matrices(matrices: &ConstraintMatrices) -> Self {
        let row_count = matrices.a.row_count();
        let column_count = matrices.a.column_count();
        assert_eq!(matrices.b.row_count(), row_count);
        assert_eq!(matrices.c.row_count(), row_count);
        assert_eq!(matrices.b.column_count(), column_count);
        assert_eq!(matrices.c.column_count(), column_count);
        assert!(
            row_count <= ROW_MASK as usize,
            "too many sparse matrix rows"
        );

        let sources = [
            (&matrices.a, MatrixKind::A),
            (&matrices.b, MatrixKind::B),
            (&matrices.c, MatrixKind::C),
        ];
        let matrix_nonzeros = sources.map(|(matrix, _)| nonzero_count(matrix));
        let total_nonzeros = matrix_nonzeros.iter().sum::<usize>();
        assert!(
            total_nonzeros < u32::MAX as usize,
            "too many sparse entries"
        );

        let mut column_offsets = vec![0_u32; column_count + 1];
        for (matrix, _) in sources {
            for row in matrix.rows() {
                for &column in row.indices() {
                    column_offsets[column + 1] = column_offsets[column + 1]
                        .checked_add(1)
                        .expect("too many entries in one sparse column");
                }
            }
        }
        for column in 0..column_count {
            column_offsets[column + 1] = column_offsets[column + 1]
                .checked_add(column_offsets[column])
                .expect("too many sparse entries");
        }

        let mut columns = (0..column_count)
            .map(|column| {
                Vec::with_capacity((column_offsets[column + 1] - column_offsets[column]) as usize)
            })
            .collect::<Vec<Vec<IntegerEntry>>>();
        for (matrix, kind) in sources {
            for (row, sparse_row) in matrix.rows().enumerate() {
                for (&column, coefficient) in
                    sparse_row.indices().iter().zip(sparse_row.entry_range())
                {
                    columns[column].push(IntegerEntry::new(
                        row,
                        kind,
                        coefficient,
                        &matrix.coefficients()[coefficient],
                    ));
                }
            }
        }
        let mut coordinates = Vec::with_capacity(total_nonzeros);
        let mut coefficients = IntegerTable::default();
        for entry in columns.into_iter().flatten() {
            coordinates.push(entry.coordinate);
            sources[entry.kind as usize]
                .0
                .coefficients()
                .copy_row_to(entry.coefficient, &mut coefficients);
        }
        debug_assert_eq!(coordinates.len(), total_nonzeros);

        Self {
            row_count,
            column_offsets: column_offsets.into_boxed_slice(),
            coordinates: coordinates.into_boxed_slice(),
            coefficients,
            matrix_nonzeros,
        }
    }

    /// Number of R1CS rows, and therefore required challenge elements.
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of integer-witness columns, including constant column zero.
    pub const fn column_count(&self) -> usize {
        self.column_offsets.len() - 1
    }

    /// Nonzero coefficients in `A`, `B`, and `C`, respectively.
    pub const fn matrix_nonzero_counts(&self) -> [usize; 3] {
        self.matrix_nonzeros
    }

    /// Total nonzero coefficients across the three matrices.
    pub const fn nonzero_count(&self) -> usize {
        self.coordinates.len()
    }

    /// Bytes occupied by the exact-integer CSC payload.
    pub fn payload_bytes(&self) -> usize {
        self.column_offsets.len() * size_of::<u32>()
            + self.coordinates.len() * size_of::<u32>()
            + self.coefficients.payload_bytes()
    }

    /// Reduces all integer coefficients and allocates reusable apply storage.
    pub fn prepare(
        &self,
        modulus: &ModRingCtx<2>,
    ) -> Result<PreparedMaterializedAbc<'_>, SparseAbcApplyError> {
        let modulus_words = *modulus.modulus().as_words();
        if modulus_words[0] & 1 == 0 {
            return Err(SparseAbcApplyError::EvenModulus);
        }
        let field = create_prime_field(Uint::from_words(modulus_words));
        let view = self.coefficients.view();
        let coefficients = (0..view.len())
            .into_par_iter()
            .map(|index| {
                *field
                    .from_integer(&field::ZRef::from_twos_complement_words(&view[index]))
                    .as_montgomery_integer()
                    .as_words()
            })
            .collect();
        Ok(PreparedMaterializedAbc {
            matrix: self,
            field,
            coefficients,
            weighted_challenges: vec![[[0; 2]; 3]; self.row_count()],
            output: vec![[0; 2]; self.column_count()],
        })
    }
}

fn nonzero_count(matrix: &CsrMatrix<IntegerTable>) -> usize {
    matrix.nnz()
}

/// Modulus-prepared sparse-matrix evaluator with Montgomery input and output.
#[derive(Debug)]
pub struct PreparedMaterializedAbc<'a> {
    matrix: &'a MaterializedAbc,
    field: FpCtx<2>,
    coefficients: Vec<[u64; 2]>,
    weighted_challenges: Vec<[[u64; 2]; 3]>,
    output: Vec<[u64; 2]>,
}

impl PreparedMaterializedAbc<'_> {
    /// Converts a canonical element into Montgomery form for this modulus.
    pub fn to_montgomery(&self, canonical: [u64; 2]) -> [u64; 2] {
        *self
            .field
            .from_integer(&Uint::from_words(canonical))
            .as_montgomery_integer()
            .as_words()
    }

    /// Bytes added by reduced coefficients and reusable apply vectors.
    pub fn workspace_bytes(&self) -> usize {
        self.coefficients.len() * size_of::<[u64; 2]>()
            + self.weighted_challenges.len() * size_of::<[[u64; 2]; 3]>()
            + self.output.len() * size_of::<[u64; 2]>()
    }

    /// Computes `r * (A + x B + x^2 C)` with Montgomery input and output.
    pub fn apply(
        &mut self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
    ) -> Result<&[[u64; 2]], SparseAbcApplyError> {
        if challenges.len() != self.matrix.row_count {
            return Err(SparseAbcApplyError::ChallengeLength {
                expected: self.matrix.row_count,
                actual: challenges.len(),
            });
        }

        let x_squared = mul_representatives(x, x, &self.field);
        let prepare_challenge = |challenge: &[u64; 2]| {
            let r = *challenge;
            let rx = mul_representatives(r, x, &self.field);
            [r, rx, mul_representatives(r, x_squared, &self.field)]
        };
        if rayon::current_num_threads() > 1
            && self.weighted_challenges.len() >= PARALLEL_VECTOR_THRESHOLD
        {
            self.weighted_challenges
                .par_iter_mut()
                .zip(challenges.par_iter())
                .for_each(|(weighted, challenge)| *weighted = prepare_challenge(challenge));
        } else {
            self.weighted_challenges
                .iter_mut()
                .zip(challenges.iter())
                .for_each(|(weighted, challenge)| *weighted = prepare_challenge(challenge));
        }

        let evaluate = |column: usize| {
            let start = self.matrix.column_offsets[column] as usize;
            let end = self.matrix.column_offsets[column + 1] as usize;
            let mut value = [0_u64; 2];
            let mut initialized = false;
            for index in start..end {
                let coordinate = self.matrix.coordinates[index];
                let row = (coordinate & ROW_MASK) as usize;
                let kind = ((coordinate >> KIND_SHIFT) & 3) as usize;
                let class = coordinate >> CLASS_SHIFT;
                let source = self.weighted_challenges[row][kind];
                let contribution = match class {
                    value if value == CoefficientClass::One as u32 => source,
                    value if value == CoefficientClass::NegativeOne as u32 => {
                        neg_representative(source, &self.field)
                    }
                    _ => mul_representatives(source, self.coefficients[index], &self.field),
                };
                if initialized {
                    value = add_representatives(value, contribution, &self.field);
                } else {
                    value = contribution;
                    initialized = true;
                }
            }
            value
        };
        if rayon::current_num_threads() > 1
            && self.matrix.coordinates.len() >= PARALLEL_NNZ_THRESHOLD
        {
            self.output
                .par_iter_mut()
                .enumerate()
                .for_each(|(column, value)| *value = evaluate(column));
        } else {
            self.output
                .iter_mut()
                .enumerate()
                .for_each(|(column, value)| *value = evaluate(column));
        }
        Ok(&self.output)
    }
}

/// Failure to prepare or apply a sparse `A/B/C` materialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseAbcApplyError {
    ChallengeLength { expected: usize, actual: usize },
    EvenModulus,
}

impl Display for SparseAbcApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChallengeLength { expected, actual } => write!(
                formatter,
                "challenge vector has length {actual}, expected {expected}"
            ),
            Self::EvenModulus => write!(formatter, "Montgomery evaluation needs an odd modulus"),
        }
    }
}

impl Error for SparseAbcApplyError {}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use num_traits::One;

    use super::*;
    use crate::Circuit;
    use crate::constraints::ConstraintGenerator;
    use crate::linear_map::circuit::WengertGenerator;

    fn example_circuit<CS: Circuit>(circuit: &mut CS, inputs: &[CS::Bool; 3]) {
        let a = circuit.bitz::<2>(inputs[0].clone());
        let b = circuit.bitz::<2>(inputs[1].clone());
        let c = circuit.bitz::<2>(inputs[2].clone());
        circuit.assert_r1c(
            a.clone() * CS::Coefficient::<2>::from(7) - b.clone(),
            b.clone() * CS::Coefficient::<2>::from(11)
                + CS::Z::<2>::from(CS::Coefficient::<2>::from(5)),
            c.clone() * CS::Coefficient::<2>::from(13),
        );
        circuit.assert_r1c(
            a + c.clone(),
            -c,
            b + CS::Z::<2>::from(CS::Coefficient::<2>::from(19)),
        );
    }

    #[test]
    fn sparse_integer_runner_matches_wengert_tape() {
        let mut matrix_generator = ConstraintGenerator::new(3);
        let matrix_inputs = matrix_generator.inputs();
        example_circuit(&mut matrix_generator, &matrix_inputs);
        let matrices = matrix_generator.into_matrices();
        let sparse = MaterializedAbc::from_matrices(&matrices);
        assert_eq!(
            sparse.matrix_nonzero_counts(),
            [
                nonzero_count(&matrices.a),
                nonzero_count(&matrices.b),
                nonzero_count(&matrices.c),
            ]
        );

        let mut tape_generator = WengertGenerator::new(3);
        let tape_inputs = tape_generator.take_boxed_inputs();
        example_circuit(&mut tape_generator, &tape_inputs);
        let tape = tape_generator.finish();

        let modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(
                &((BigUint::one() << 128_usize) - BigUint::from(159_u64)),
            ),
        ))
        .unwrap();
        let challenges = [[23, 0], [29, 0]];
        let x = [17, 0];
        let mut sparse_evaluator = sparse.prepare(&modulus).unwrap();
        let challenges = challenges.map(|value| sparse_evaluator.to_montgomery(value));
        let x = sparse_evaluator.to_montgomery(x);
        let sparse_output = sparse_evaluator.apply(&challenges, x).unwrap().to_vec();

        let mut tape_evaluator = tape.prepare(&modulus).unwrap();
        let tape_output = tape_evaluator.apply(&challenges, x).unwrap();
        assert_eq!(sparse_output, tape_output);
    }
}
