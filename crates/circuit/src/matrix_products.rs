//! Exact matrix-product recording and parallel runtime-field reduction.
//!
//! [`crate::witgen::ProductWitgen`] evaluates each rank-1 input over the
//! gadget-local fixed integer type and records dense integer `A(Mw)`, `B(Mw)`,
//! and `C(Mw)` vectors during witness generation. This module subsequently
//! reduces those vectors modulo a runtime modulus. Large batches use Rayon;
//! small batches stay sequential to avoid scheduling overhead.

use crate::integer_storage::IntegerTable;
use field::{IntegerEmbedding, ModRingCtx, ZRef};
#[cfg(test)]
use std::array;

#[cfg(test)]
use num_bigint::BigUint;
#[cfg(test)]
use num_traits::One;
use rayon::prelude::*;

use crate::witgen::Z as Integer;

/// Converts a borrowed fixed-width signed integer directly to canonical output.
/// The caller supplies the declared width; this never trims sign limbs.
pub fn project_signed<const L: usize>(modulus: &ModRingCtx<L>, words: &[u64]) -> [u64; L] {
    *modulus
        .to_integer(&modulus.from_integer(&ZRef::from_twos_complement_words(words)))
        .as_words()
}

#[cfg(test)]
pub(crate) fn biguint_words<const L: usize>(value: &BigUint) -> [u64; L] {
    assert!(value.bits() <= (64 * L) as u64);
    let digits = value.to_u64_digits();
    array::from_fn(|index| digits.get(index).copied().unwrap_or(0))
}

/// A dense vector of canonical runtime-field elements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModularVector<const PRIME_LIMBS: usize> {
    values: Vec<[u64; PRIME_LIMBS]>,
}

impl<const PRIME_LIMBS: usize> ModularVector<PRIME_LIMBS> {
    /// Number of materialized elements.
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the vector contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Dense canonical field elements in row order.
    pub fn values(&self) -> &[[u64; PRIME_LIMBS]] {
        &self.values
    }

    /// Returns one canonical field element.
    pub fn get(&self, index: usize) -> [u64; PRIME_LIMBS] {
        self.values[index]
    }
}

/// Dense runtime-field `A(Mw)`, `B(Mw)`, and `C(Mw)` vectors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixProducts<const PRIME_LIMBS: usize> {
    pub a_mw: ModularVector<PRIME_LIMBS>,
    pub b_mw: ModularVector<PRIME_LIMBS>,
    pub c_mw: ModularVector<PRIME_LIMBS>,
}

/// Dense exact integer `A(Mw)`, `B(Mw)`, and `C(Mw)` vectors.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IntegerProducts {
    pub a_mw: IntegerTable,
    pub b_mw: IntegerTable,
    pub c_mw: IntegerTable,
}

impl IntegerProducts {
    /// Records one rank-1 row using its gadget-local integer width.
    pub fn push<const LIMBS: usize>(
        &mut self,
        a: Integer<LIMBS>,
        b: Integer<LIMBS>,
        c: Integer<LIMBS>,
    ) {
        self.a_mw.push(a);
        self.b_mw.push(b);
        self.c_mw.push(c);
    }

    /// Reduces every materialized element modulo `modulus`.
    ///
    /// Vectors with at least 32,768 entries use Rayon. Smaller vectors remain
    /// sequential because dispatching them to the pool costs more than the
    /// available parallel work.
    pub fn reduce_parallel<const PRIME_LIMBS: usize>(
        &self,
        modulus: &ModRingCtx<PRIME_LIMBS>,
    ) -> MatrixProducts<PRIME_LIMBS> {
        MatrixProducts {
            a_mw: reduce_vector(&self.a_mw, modulus),
            b_mw: reduce_vector(&self.b_mw, modulus),
            c_mw: reduce_vector(&self.c_mw, modulus),
        }
    }
}

fn reduce_vector<const PRIME_LIMBS: usize>(
    values: &IntegerTable,
    modulus: &ModRingCtx<PRIME_LIMBS>,
) -> ModularVector<PRIME_LIMBS> {
    const PARALLEL_THRESHOLD: usize = 1 << 15;
    let view = values.view();
    let project = |row: usize| project_signed(modulus, &view[row]);
    let reduced = if values.len() >= PARALLEL_THRESHOLD {
        (0..values.len()).into_par_iter().map(project).collect()
    } else {
        (0..values.len()).map(project).collect()
    };
    ModularVector { values: reduced }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use num_traits::Signed;

    use super::*;
    use crate::constraints::{ConstraintGenerator, ConstraintMatrices};
    use crate::integer_storage::IntegerTable;
    use crate::linear_map::CsrMatrix;
    use crate::sha256::{SHA256_2KB_MESSAGE_BITS, SHA256_2KB_WITNESS_BITS, sha256_2kb_circuit};
    use crate::witgen::{PackedWitness, ProductWitgen};

    fn stored_bigint(value: &[u64]) -> BigInt {
        let bytes = value
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        BigInt::from_signed_bytes_le(&bytes)
    }

    fn direct_row(
        matrix: &CsrMatrix<IntegerTable>,
        row: usize,
        integer_witness: &PackedWitness,
    ) -> BigInt {
        matrix
            .row(row)
            .unwrap()
            .iter()
            .map(|(column, coefficient)| (column, coefficient.as_words()))
            .filter(|(column, _)| integer_witness.bit(*column))
            .map(|(_, coefficient)| stored_bigint(coefficient))
            .sum()
    }

    fn reduce_bigint<const P: usize>(value: BigInt, modulus: &BigUint) -> [u64; P] {
        let modulus = BigInt::from(modulus.clone());
        let mut reduced = value % &modulus;
        if reduced.is_negative() {
            reduced += &modulus;
        }
        biguint_words(&reduced.to_biguint().unwrap())
    }

    fn assert_products_match_direct<const P: usize>(
        exact: &IntegerProducts,
        products: &MatrixProducts<P>,
        matrices: &ConstraintMatrices,
        integer_witness: &PackedWitness,
        modulus: &BigUint,
    ) {
        assert_eq!(exact.a_mw.len(), matrices.a.row_count());
        assert_eq!(exact.b_mw.len(), matrices.b.row_count());
        assert_eq!(exact.c_mw.len(), matrices.c.row_count());
        assert_eq!(products.a_mw.len(), matrices.a.row_count());
        assert_eq!(products.b_mw.len(), matrices.b.row_count());
        assert_eq!(products.c_mw.len(), matrices.c.row_count());
        for row in 0..matrices.a.row_count() {
            let expected_a = direct_row(&matrices.a, row, integer_witness);
            let expected_b = direct_row(&matrices.b, row, integer_witness);
            let expected_c = direct_row(&matrices.c, row, integer_witness);

            assert_eq!(
                stored_bigint(&exact.a_mw[row]),
                expected_a,
                "exact A(Mw) differs at row {row}"
            );
            assert_eq!(
                stored_bigint(&exact.b_mw[row]),
                expected_b,
                "exact B(Mw) differs at row {row}"
            );
            assert_eq!(
                stored_bigint(&exact.c_mw[row]),
                expected_c,
                "exact C(Mw) differs at row {row}"
            );
            assert_eq!(
                products.a_mw.get(row),
                reduce_bigint(expected_a, modulus),
                "A(Mw) differs at row {row}"
            );
            assert_eq!(
                products.b_mw.get(row),
                reduce_bigint(expected_b, modulus),
                "B(Mw) differs at row {row}"
            );
            assert_eq!(
                products.c_mw.get(row),
                reduce_bigint(expected_c, modulus),
                "C(Mw) differs at row {row}"
            );
        }
    }

    #[test]
    fn signed_fixed_integers_reduce_correctly() {
        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(modulus.clone())),
        ))
        .unwrap();

        for value in [0_i128, 1, 7, -7, i128::from(i64::MIN)] {
            let stored = Integer::<2>::from(value);
            let mut expected = BigInt::from(value) % BigInt::from(modulus.clone());
            if expected.is_negative() {
                expected += BigInt::from(modulus.clone());
            }
            assert_eq!(
                project_signed(&runtime_modulus, stored.as_words()),
                biguint_words::<2>(&expected.to_biguint().unwrap())
            );
        }
    }

    #[test]
    fn fixed_width_reduction_matches_bigint_for_wide_values() {
        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(modulus.clone())),
        ))
        .unwrap();
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        for index in 0..1_000 {
            let mut words = [0; 9];
            for word in &mut words {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *word = state;
            }
            words[8] &= i64::MAX as u64;
            let bits: Vec<_> = (0..9 * 64)
                .map(|bit| words[bit / 64] >> (bit % 64) & 1 != 0)
                .collect();
            let mut integer = crate::witgen::integer_from_bits::<9>(&bits);
            if index % 2 != 0 {
                integer = -integer;
            }
            let stored = integer;
            assert_eq!(
                project_signed(&runtime_modulus, stored.as_words()),
                reduce_bigint(stored_bigint(stored.as_words()), &modulus)
            );
        }
    }

    #[test]
    fn modulus_round_trips() {
        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(modulus.clone())),
        ))
        .unwrap();
        assert_eq!(
            runtime_modulus.modulus().as_words(),
            &biguint_words::<2>(&modulus)
        );
    }

    #[test]
    fn sha256_recorded_products_match_direct_matrices() {
        let message: Box<[bool]> = (0..SHA256_2KB_MESSAGE_BITS)
            .map(|bit| {
                let byte = (bit / 8) as u8;
                byte & (1 << (7 - bit % 8)) != 0
            })
            .collect();
        let message: Box<[bool; SHA256_2KB_MESSAGE_BITS]> = message.try_into().unwrap();

        let mut witgen =
            ProductWitgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
        let _ = sha256_2kb_circuit(&mut witgen, &message);

        let mut generator = ConstraintGenerator::new(SHA256_2KB_MESSAGE_BITS);
        let symbolic_inputs = generator.boxed_inputs();
        let _ = sha256_2kb_circuit(&mut generator, &symbolic_inputs);
        let matrices = generator.into_matrices();

        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(modulus.clone())),
        ))
        .unwrap();
        let products = witgen.products().reduce_parallel(&runtime_modulus);
        assert_products_match_direct(
            witgen.products(),
            &products,
            &matrices,
            witgen.integer_witness(),
            &modulus,
        );
    }
}
