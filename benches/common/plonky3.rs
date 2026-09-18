//! Shared adapters to Plonky3's native fields, tables, and WHIR stack.
//! Relation constraints and transcript encodings belong to each caller.

use p3_challenger::{CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::{point::Point, poly::Poly};
use p3_sumcheck::layout::{SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_whir::{ProtocolParameters, WhirConfig, WhirConfigError, WhirProver};

/// Keep Plonky3's concrete PCS and proof types visible to callers.
pub type Pcs<F, EF, Ch, M> = WhirProver<EF, F, Radix2DFTSmallBatch<F>, M, Ch, SuffixProver<F, EF>>;

fn new_pcs<F, EF, Ch, M>(
    num_variables: usize,
    params: ProtocolParameters,
    mmcs: M,
) -> Result<Pcs<F, EF, Ch, M>, WhirConfigError>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanSampleUniformBits<F>,
    M: Mmcs<F>,
{
    let config = WhirConfig::new(num_variables, params)?;
    let dft = Radix2DFTSmallBatch::new(1 << config.max_fft_size());
    Ok(WhirProver::new(config, dft, mmcs))
}

pub mod baby_bear {
    use p3_baby_bear::{Poseidon2BabyBear, default_babybear_poseidon2_16};
    use p3_challenger::DuplexChallenger;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use super::*;
    pub use p3_baby_bear::BabyBear as Val;

    type Perm = Poseidon2BabyBear<16>;
    type Packed = <Val as Field>::Packing;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    pub type Mmcs = MerkleTreeMmcs<Packed, Packed, Hash, Compress, 2, 8>;
    pub type Pcs<EF> = super::Pcs<Val, EF, Challenger, Mmcs>;

    pub fn challenger() -> Challenger {
        Challenger::new(default_babybear_poseidon2_16())
    }

    pub fn pcs<EF: ExtensionField<Val> + TwoAdicField>(
        num_variables: usize,
        params: ProtocolParameters,
    ) -> Result<Pcs<EF>, WhirConfigError> {
        let perm = default_babybear_poseidon2_16();
        let mmcs = Mmcs::new(Hash::new(perm.clone()), Compress::new(perm), 0);
        super::new_pcs(num_variables, params, mmcs)
    }
}

pub mod goldilocks {
    use p3_challenger::DuplexChallenger;
    use p3_goldilocks::{Poseidon2Goldilocks, default_goldilocks_poseidon2_8};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use super::*;
    pub use p3_goldilocks::Goldilocks as Val;

    type Perm = Poseidon2Goldilocks<8>;
    type Packed = <Val as Field>::Packing;
    type Hash = PaddingFreeSponge<Perm, 8, 4, 4>;
    type Compress = TruncatedPermutation<Perm, 2, 4, 8>;
    pub type Challenger = DuplexChallenger<Val, Perm, 8, 4>;
    pub type Mmcs = MerkleTreeMmcs<Packed, Packed, Hash, Compress, 2, 4>;
    pub type Pcs<EF> = super::Pcs<Val, EF, Challenger, Mmcs>;

    pub fn challenger() -> Challenger {
        Challenger::new(default_goldilocks_poseidon2_8())
    }

    pub fn pcs<EF: ExtensionField<Val> + TwoAdicField>(
        num_variables: usize,
        params: ProtocolParameters,
    ) -> Result<Pcs<EF>, WhirConfigError> {
        let perm = default_goldilocks_poseidon2_8();
        let mmcs = Mmcs::new(Hash::new(perm.clone()), Compress::new(perm), 0);
        super::new_pcs(num_variables, params, mmcs)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ColumnError {
    #[error("column {column} has length {actual}; expected {expected}")]
    Length {
        column: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("column {column}[{index}]={value} is not canonical (< {modulus})")]
    NonCanonical {
        column: &'static str,
        index: usize,
        value: u64,
        modulus: u64,
    },
    #[error("column materialization size overflowed usize")]
    SizeOverflow,
}

/// Inject integers exactly; modular reduction would change the statement.
/// Each source column becomes one polynomial row in Plonky3's table.
pub fn column_table<F: PrimeField64>(
    capacity: usize,
    columns: &[(&'static str, &[u64])],
) -> Result<Table<F>, ColumnError> {
    for &(column, input) in columns {
        if input.len() != capacity {
            return Err(ColumnError::Length {
                column,
                expected: capacity,
                actual: input.len(),
            });
        }
    }
    column_table_from_fn(capacity, columns.len(), |col, index| {
        (columns[col].0, columns[col].1[index])
    })
}

pub fn column_table_from_fn<F: PrimeField64>(
    capacity: usize,
    columns: usize,
    read: impl Fn(usize, usize) -> (&'static str, u64),
) -> Result<Table<F>, ColumnError> {
    let len = capacity
        .checked_mul(columns)
        .ok_or(ColumnError::SizeOverflow)?;
    let mut values = Vec::with_capacity(len);
    for col in 0..columns {
        for index in 0..capacity {
            let (column, value) = read(col, index);
            if value >= F::ORDER_U64 {
                return Err(ColumnError::NonCanonical {
                    column,
                    index,
                    value,
                    modulus: F::ORDER_U64,
                });
            }
            values.push(F::from_u64(value));
        }
    }
    Ok(Table::new(RowMajorMatrix::new(values, capacity)))
}

pub fn column_opening(gate_vars: usize, columns: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(gate_vars, columns),
        vec![OpeningBatch::new((0..columns).collect(), Vec::new())],
    )])
}

/// BitZ's first coordinate selects adjacent entries; Plonky3's last does.
pub fn opening_point<F: Field>(lsb_first: &[F]) -> Point<F> {
    Point::new(lsb_first.iter().rev().copied().collect())
}

/// Evaluate the assignment `[e0 | columns... | zero padding]` using Plonky3.
pub fn assignment_eval<F: Field>(
    gate_point_lsb_first: &[F],
    block_point_lsb_first: &[F],
    columns: &[F],
) -> F {
    let e0 = Point::eval_eq(
        &F::zero_vec(gate_point_lsb_first.len()),
        gate_point_lsb_first,
    );
    let mut blocks = F::zero_vec(1 << block_point_lsb_first.len());
    blocks[0] = e0;
    blocks[1..=columns.len()].copy_from_slice(columns);
    Poly::new(blocks).eval_base(&opening_point(block_point_lsb_first))
}

#[cfg(test)]
mod tests {
    #[test]
    fn columns_preserve_layout_and_reject_lossy_conversion() {
        use super::*;
        use p3_field::PrimeCharacteristicRing;
        type F = baby_bear::Val;
        let table = column_table::<F>(2, &[("a", &[2, 3]), ("b", &[5, 7])]).unwrap();
        for (column, expected) in [[2, 3], [5, 7]].into_iter().enumerate() {
            for (row, value) in expected.into_iter().enumerate() {
                assert_eq!(
                    table.poly(column).eval_base(&Point::<F>::hypercube(row, 1)),
                    F::from_u64(value)
                );
            }
        }
        assert!(matches!(
            column_table::<F>(2, &[("short", &[1])]),
            Err(ColumnError::Length { .. })
        ));
        assert!(matches!(
            column_table::<F>(1, &[("modulus", &[F::ORDER_U64])]),
            Err(ColumnError::NonCanonical { .. })
        ));
        type G = goldilocks::Val;
        assert!(matches!(
            column_table::<G>(1, &[("modulus", &[G::ORDER_U64])]),
            Err(ColumnError::NonCanonical { .. })
        ));
    }

    #[test]
    fn assignment_matches_direct_folding_with_constant_and_padding() {
        use super::*;
        use p3_field::PrimeCharacteristicRing;
        type F = baby_bear::Val;
        // The complete assignment is [e0 | A | B | C | K | 0 | 0 | 0].
        let columns = [[2, 3], [5, 7], [11, 13], [17, 19]].map(|c| c.map(F::from_u64));
        for coords in [[0, 0, 0, 0], [1, 1, 1, 1], [2, 3, 5, 7]] {
            let [gate, b0, b1, b2] = coords.map(F::from_u64);
            let opened = columns.map(|[a, b]| a + gate * (b - a));
            let mut assignment = vec![F::ONE, F::ZERO];
            assignment.extend(columns.into_iter().flatten());
            assignment.resize(16, F::ZERO);
            // Independent adjacent-pair folding in BitZ's coordinate order.
            for r in [gate, b0, b1, b2] {
                assignment = assignment
                    .chunks_exact(2)
                    .map(|p| p[0] + r * (p[1] - p[0]))
                    .collect();
            }
            assert_eq!(
                assignment_eval(&[gate], &[b0, b1, b2], &opened),
                assignment[0]
            );
        }
    }
}
