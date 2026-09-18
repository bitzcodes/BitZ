//! Sparse constraint generation for the BitZ circuit language.
//!
//! The generated matrices follow Freigen's convention. `M` maps the Boolean
//! witness, prefixed by a constant one, to the integer witness. Its first row
//! is the implicit integer constant one. `A`, `B`, and `C` then encode the
//! rank-1 constraints `(A z) * (B z) = C z` over that integer witness. Every
//! integer coefficient retains the compile-time limb width declared by its gadget.

use std::array;
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::integer_storage::IntegerTable;
use crate::linear_map::{CsrBuilder, CsrMatrix, ImplicitOnes};
use field::{CheckedArithmetic, CtEq, CtMask, CtSelect, CtValue, IntegerOps, WideMul, Z};
use num_traits::{One, Zero};

use crate::witgen::PackedWitness;
use crate::{BoolWitness, Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

#[derive(Clone, Copy, Debug)]
struct RowCheck {
    limbs: usize,
    check: fn(&ConstraintMatrices, usize, &[bool]) -> bool,
}
impl PartialEq for RowCheck {
    fn eq(&self, other: &Self) -> bool {
        self.limbs == other.limbs
    }
}
impl Eq for RowCheck {}

/// The four sparse matrices generated for an BitZ circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintMatrices {
    /// Boolean-to-integer witness matrix.
    pub m: CsrMatrix<ImplicitOnes>,
    /// Left R1CS matrix.
    pub a: CsrMatrix<IntegerTable>,
    /// Right R1CS matrix.
    pub b: CsrMatrix<IntegerTable>,
    /// Output R1CS matrix.
    pub c: CsrMatrix<IntegerTable>,
    row_checks: Vec<RowCheck>,
}

/// Why a Boolean witness does not satisfy a generated constraint system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SatisfactionError {
    /// The packed Boolean witness has the wrong number of entries.
    WitnessLength { expected: usize, actual: usize },
    /// The indicated R1CS row does not satisfy `a * b = c`.
    Constraint { row: usize },
}

impl Display for SatisfactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WitnessLength { expected, actual } => write!(
                formatter,
                "Boolean witness has length {actual}, expected {expected}"
            ),
            Self::Constraint { row } => write!(formatter, "R1CS row {row} is unsatisfied"),
        }
    }
}

impl Error for SatisfactionError {}

impl ConstraintMatrices {
    /// Applies `M` to a packed Boolean witness.
    ///
    /// The returned vector starts with the implicit constant one and is the
    /// witness consumed by `A`, `B`, and `C`.
    pub fn integer_witness(&self, witness: &PackedWitness) -> Result<Vec<bool>, SatisfactionError> {
        let expected = self.m.column_count().saturating_sub(1);
        if witness.bit_len() != expected {
            return Err(SatisfactionError::WitnessLength {
                expected,
                actual: witness.bit_len(),
            });
        }

        Ok(self
            .m
            .rows()
            .map(|row| {
                row.indices().iter().fold(false, |value, column| {
                    value
                        ^ if *column == 0 {
                            true
                        } else {
                            witness.bit(column - 1)
                        }
                })
            })
            .collect())
    }

    /// Checks every materialized R1CS row against a packed Boolean witness.
    pub fn check_witness(&self, witness: &PackedWitness) -> Result<(), SatisfactionError> {
        let integer_witness = self.integer_witness(witness)?;

        // Visit every row; private values do not choose how much work is done.
        let mut first_failure = usize::MAX;
        for (row, checker) in self.row_checks.iter().enumerate() {
            let valid = (checker.check)(self, row, &integer_witness);
            let take = CtMask::from_lsb((!valid & (first_failure == usize::MAX)) as u64);
            first_failure = u64::ct_select(&(first_failure as u64), &(row as u64), take) as usize;
        }
        if first_failure != usize::MAX {
            return Err(SatisfactionError::Constraint { row: first_failure });
        }

        Ok(())
    }

    /// Whether every materialized row is satisfied by the witness.
    pub fn is_satisfied(&self, witness: &PackedWitness) -> bool {
        self.check_witness(witness).is_ok()
    }
}

fn evaluate_integer_row<const L: usize>(
    matrix: &CsrMatrix<IntegerTable>,
    row: usize,
    witness: &[bool],
) -> Z<L> {
    matrix
        .row(row)
        .unwrap()
        .iter()
        .map(|(column, coefficient)| (column, coefficient.as_words()))
        .fold(Z::ZERO, |sum, (column, words)| {
            // Width and positive/negative subset bounds were checked at preparation.
            let coefficient = Z::from_twos_complement_words(
                words.try_into().expect("declared coefficient width"),
            );
            sum.wrapping_add(&Z::ct_select(
                &Z::ZERO,
                &coefficient,
                CtMask::from_lsb(witness[column] as u64),
            ))
        })
}

fn check_integer_constraint<const L: usize>(
    matrices: &ConstraintMatrices,
    row: usize,
    witness: &[bool],
) -> bool {
    let a = evaluate_integer_row::<L>(&matrices.a, row, witness);
    let b = evaluate_integer_row::<L>(&matrices.b, row, witness);
    let c = evaluate_integer_row::<L>(&matrices.c, row, witness);
    let product = IntegerOps.mul_wide(&a, &b).checked_resize_ct::<L>();
    (product.validity() & product.value().ct_eq(&c)).declassify()
}

/// Public symbolic preparation must never silently wrap a coefficient.
fn exact_public<T>(value: CtValue<T>) -> T {
    let (value, valid) = value.into_parts();
    assert!(
        valid.declassify(),
        "declared circuit coefficient width exceeded"
    );
    value
}

/// A symbolic linear combination over F2.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BoolLinearCombination {
    constant: bool,
    witnesses: BTreeSet<usize>,
}

impl BoolLinearCombination {
    /// The Boolean constant term.
    pub const fn constant(&self) -> bool {
        self.constant
    }

    /// Zero-based Boolean witness indices with coefficient one.
    pub fn witnesses(&self) -> &BTreeSet<usize> {
        &self.witnesses
    }

    pub(crate) fn witness(index: usize) -> Self {
        Self {
            constant: false,
            witnesses: BTreeSet::from([index]),
        }
    }

    pub(crate) fn xor(mut self, rhs: Self) -> Self {
        self.constant ^= rhs.constant;
        for witness in rhs.witnesses {
            if !self.witnesses.insert(witness) {
                self.witnesses.remove(&witness);
            }
        }
        self
    }
}

impl From<bool> for BoolLinearCombination {
    fn from(constant: bool) -> Self {
        Self {
            constant,
            witnesses: BTreeSet::new(),
        }
    }
}

impl BoolWitness for BoolLinearCombination {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
}

/// A symbolic integer linear combination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearCombination<const L: usize> {
    constant: Z<L>,
    witnesses: BTreeMap<usize, Z<L>>,
}

impl<const L: usize> LinearCombination<L> {
    /// The integer constant term.
    pub fn constant(&self) -> &Z<L> {
        &self.constant
    }

    /// Nonzero coefficients keyed by zero-based integer witness index.
    pub fn witnesses(&self) -> &BTreeMap<usize, Z<L>> {
        &self.witnesses
    }

    fn witness(index: usize) -> Self {
        Self {
            constant: Z::<L>::zero(),
            witnesses: BTreeMap::from([(index, Z::<L>::one())]),
        }
    }

    fn add_term(&mut self, index: usize, coefficient: Z<L>) {
        if coefficient.is_zero() {
            return;
        }
        match self.witnesses.entry(index) {
            Entry::Vacant(entry) => {
                entry.insert(coefficient);
            }
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = exact_public(entry.get().checked_add_ct(&coefficient));
                if entry.get().is_zero() {
                    entry.remove();
                }
            }
        }
    }

    fn validate_boolean_bounds(&self) {
        // Every Boolean subset lies between the sum of all negative and all
        // positive terms. These public bounds justify wrapping addition in the
        // fixed-width private evaluator, including signed-minimum coefficients.
        let mut negative = Z::<L>::ZERO;
        let mut positive = Z::<L>::ZERO;
        for coefficient in std::iter::once(&self.constant).chain(self.witnesses.values()) {
            if coefficient.is_negative_ct().declassify() {
                negative = exact_public(negative.checked_add_ct(coefficient));
            } else {
                positive = exact_public(positive.checked_add_ct(coefficient));
            }
        }
    }

    fn sign_extend<const M: usize>(self) -> LinearCombination<M> {
        LinearCombination {
            constant: self.constant.sign_extend(),
            witnesses: self
                .witnesses
                .into_iter()
                .map(|(i, c)| (i, c.sign_extend()))
                .collect(),
        }
    }

    fn append_to(self, builder: &mut CsrBuilder<IntegerTable>) {
        self.validate_boolean_bounds();
        let constant = (!self.constant.is_zero()).then_some((0, self.constant));
        builder
            .push_row(
                constant.into_iter().chain(
                    self.witnesses
                        .into_iter()
                        .map(|(witness, coefficient)| (witness + 1, coefficient)),
                ),
            )
            .expect("canonical circuit columns fit usize");
    }
}

impl<const L: usize> From<Z<L>> for LinearCombination<L> {
    fn from(constant: Z<L>) -> Self {
        Self {
            constant,
            witnesses: BTreeMap::new(),
        }
    }
}

impl<const L: usize> Zero for LinearCombination<L> {
    fn zero() -> Self {
        Self::from(Z::<L>::zero())
    }

    fn is_zero(&self) -> bool {
        self.constant.is_zero() && self.witnesses.is_empty()
    }
}

impl<const L: usize> Add for LinearCombination<L> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.constant = exact_public(self.constant.checked_add_ct(&rhs.constant));
        for (witness, coefficient) in rhs.witnesses {
            self.add_term(witness, coefficient);
        }
        self
    }
}

impl<const L: usize> AddAssign for LinearCombination<L> {
    fn add_assign(&mut self, rhs: Self) {
        self.constant = exact_public(self.constant.checked_add_ct(&rhs.constant));
        for (witness, coefficient) in rhs.witnesses {
            self.add_term(witness, coefficient);
        }
    }
}

impl<const L: usize> Neg for LinearCombination<L> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.constant = exact_public(self.constant.checked_neg_ct());
        self.witnesses = self
            .witnesses
            .into_iter()
            .map(|(witness, coefficient)| (witness, exact_public(coefficient.checked_neg_ct())))
            .collect();
        self
    }
}

impl<const L: usize> Sub for LinearCombination<L> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<const L: usize> SubAssign for LinearCombination<L> {
    fn sub_assign(&mut self, rhs: Self) {
        *self += -rhs;
    }
}

impl<const L: usize> Mul<Z<L>> for LinearCombination<L> {
    type Output = Self;

    fn mul(mut self, rhs: Z<L>) -> Self::Output {
        self.constant = exact_public(self.constant.checked_mul_ct(&rhs));
        self.witnesses = self
            .witnesses
            .into_iter()
            .filter_map(|(witness, coefficient)| {
                let coefficient = exact_public(coefficient.checked_mul_ct(&rhs));
                (!coefficient.is_zero()).then_some((witness, coefficient))
            })
            .collect();
        self
    }
}

impl<const L: usize> Sum for LinearCombination<L> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

/// Circuit backend that records sparse M/A/B/C matrices without evaluating hints.
#[derive(Clone, Debug)]
pub struct ConstraintGenerator {
    input_witnesses: usize,
    next_boolean_witness: usize,
    m_rows: Vec<BoolLinearCombination>,
    a: CsrBuilder<IntegerTable>,
    b: CsrBuilder<IntegerTable>,
    c: CsrBuilder<IntegerTable>,
    row_checks: Vec<RowCheck>,
}

impl ConstraintGenerator {
    /// Starts a generator with `input_witnesses` preallocated Boolean inputs.
    pub fn new(input_witnesses: usize) -> Self {
        Self {
            input_witnesses,
            next_boolean_witness: input_witnesses,
            m_rows: Vec::new(),
            a: CsrBuilder::new(IntegerTable::default()),
            b: CsrBuilder::new(IntegerTable::default()),
            c: CsrBuilder::new(IntegerTable::default()),
            row_checks: Vec::new(),
        }
    }

    /// Returns one of the preallocated input witnesses.
    pub fn input(&self, index: usize) -> BoolLinearCombination {
        assert!(
            index < self.input_witnesses,
            "input witness is not allocated"
        );
        BoolLinearCombination::witness(index)
    }

    /// Returns all preallocated inputs as an array.
    pub fn inputs<const N: usize>(&self) -> [BoolLinearCombination; N] {
        assert_eq!(N, self.input_witnesses, "input witness count mismatch");
        array::from_fn(BoolLinearCombination::witness)
    }

    /// Returns every preallocated input in one heap allocation.
    ///
    /// Large gadgets should use this instead of constructing a large symbolic
    /// array on the stack merely to pass it by reference.
    pub fn boxed_inputs<const N: usize>(&self) -> Box<[BoolLinearCombination; N]> {
        assert_eq!(N, self.input_witnesses, "input witness count mismatch");
        let inputs: Box<[BoolLinearCombination]> = (0..self.input_witnesses)
            .map(BoolLinearCombination::witness)
            .collect();
        match inputs.try_into() {
            Ok(inputs) => inputs,
            Err(_) => unreachable!("boxed input length was checked"),
        }
    }

    /// Finishes generation and materializes the four sparse matrices.
    pub fn into_matrices(self) -> ConstraintMatrices {
        let Self {
            input_witnesses: _,
            next_boolean_witness,
            m_rows,
            a,
            b,
            c,
            row_checks,
        } = self;
        let integer_columns = m_rows.len() + 1;

        let mut materialized_m = CsrBuilder::new(ImplicitOnes::default());
        materialized_m.push_row([(0, true)]).expect("constant row");
        for row in m_rows {
            let constant = row.constant.then_some((0, true));
            materialized_m
                .push_row(
                    constant
                        .into_iter()
                        .chain(row.witnesses.into_iter().map(|witness| (witness + 1, true))),
                )
                .expect("canonical Boolean columns");
        }
        ConstraintMatrices {
            m: materialized_m
                .finish(next_boolean_witness + 1)
                .expect("allocated Boolean columns"),
            a: a.finish(integer_columns)
                .expect("allocated integer columns"),
            b: b.finish(integer_columns)
                .expect("allocated integer columns"),
            c: c.finish(integer_columns)
                .expect("allocated integer columns"),
            row_checks,
        }
    }
}

impl Circuit for ConstraintGenerator {
    type Bool = BoolLinearCombination;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = LinearCombination<LIMBS>;

    fn xor(
        &mut self,
        lhs: BoolLinearCombination,
        rhs: BoolLinearCombination,
    ) -> BoolLinearCombination {
        lhs.xor(rhs)
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<BoolLinearCombination, N>
    where
        H: Fn(
                &dyn WitnessContext<LinearCombination<LIMBS>, BoolLinearCombination, Z<LIMBS>>,
            ) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        let first = self.next_boolean_witness;
        self.next_boolean_witness = self
            .next_boolean_witness
            .checked_add(N)
            .expect("Boolean witness count overflow");
        ScalarBits(array::from_fn(|index| {
            BoolLinearCombination::witness(first + index)
        }))
    }

    fn bitz<const LIMBS: usize>(
        &mut self,
        value: BoolLinearCombination,
    ) -> LinearCombination<LIMBS> {
        let witness = self.m_rows.len();
        self.m_rows.push(value);
        LinearCombination::witness(witness)
    }

    fn assert_r1c<const LIMBS: usize>(
        &mut self,
        a: LinearCombination<LIMBS>,
        b: LinearCombination<LIMBS>,
        c: LinearCombination<LIMBS>,
    ) {
        a.append_to(&mut self.a);
        b.append_to(&mut self.b);
        c.append_to(&mut self.c);
        self.row_checks.push(RowCheck {
            limbs: LIMBS,
            check: check_integer_constraint::<LIMBS>,
        });
    }

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: LinearCombination<FROM_LIMBS>,
    ) -> LinearCombination<TO_LIMBS> {
        assert!(
            TO_LIMBS >= FROM_LIMBS,
            "cannot sign-extend into fewer limbs"
        );
        value.sign_extend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witgen::Witgen;

    #[test]
    fn materializes_freigen_matrix_conventions_and_checks_witnesses() {
        let mut generator = ConstraintGenerator::new(2);
        let [x, y] = generator.inputs();
        let sum = generator.xor(x.clone(), y.clone());
        let z_sum = generator.bitz::<1>(sum);
        let z_x = generator.bitz::<1>(x);
        let z_y = generator.bitz::<1>(y);
        generator.assert_r1c::<1>(z_x.clone() * Z::from(2u64), z_y.clone(), z_x + z_y - z_sum);
        let mut matrices = generator.into_matrices();
        assert_eq!(matrices.m.row_count(), 4);
        assert_eq!(matrices.m.column_count(), 3);
        assert_eq!(matrices.a.row_count(), 1);
        assert_eq!(matrices.a.column_count(), 4);
        assert_eq!(matrices.m.row(0).unwrap().indices(), &[0]);
        assert_eq!(matrices.m.row(1).unwrap().indices(), &[1, 2]);
        assert_eq!(
            matrices
                .a
                .row(0)
                .unwrap()
                .iter()
                .map(|(column, coefficient)| (column, coefficient.as_words()))
                .collect::<Vec<_>>(),
            [(2, &[2][..])]
        );
        assert_eq!(
            matrices
                .c
                .row(0)
                .unwrap()
                .iter()
                .map(|(column, coefficient)| (column, coefficient.as_words()))
                .collect::<Vec<_>>(),
            [(1, &[u64::MAX][..]), (2, &[1][..]), (3, &[1][..])]
        );
        let satisfying = Witgen::with_inputs(&[true, false]);
        assert!(matrices.is_satisfied(satisfying.witness()));
        let mut changed = CsrBuilder::new(IntegerTable::default());
        changed
            .push_row([
                (0, Z::<1>::ONE),
                (1, -Z::<1>::ONE),
                (2, Z::<1>::ONE),
                (3, Z::<1>::ONE),
            ])
            .unwrap();
        matrices.c = changed.finish(4).unwrap();
        assert_eq!(
            matrices.check_witness(satisfying.witness()),
            Err(SatisfactionError::Constraint { row: 0 })
        );
    }

    #[test]
    fn mixed_declared_widths_and_exact_satisfaction() {
        let mut generator = ConstraintGenerator::new(0);
        let huge = Z::<9>::from_twos_complement_words([0, 0, 0, 0, 0, 0, 0, 0, 1]);
        generator.assert_r1c::<9>(Z::ONE.into(), huge.into(), huge.into());
        generator.assert_r1c::<1>(Z::ONE.into(), Z::MIN.into(), Z::MIN.into());
        let matrices = generator.into_matrices();
        assert_eq!(
            matrices
                .b
                .coefficients()
                .iter()
                .map(<[u64]>::len)
                .collect::<Vec<_>>(),
            [9, 1]
        );
        assert!(matrices.is_satisfied(Witgen::with_inputs(&[]).witness()));
        let mut generator = ConstraintGenerator::new(0);
        // A wrapping product would incorrectly accept this row.
        generator.assert_r1c::<1>(Z::MIN.into(), Z::from(2u64).into(), Z::ZERO.into());
        assert!(
            !generator
                .into_matrices()
                .is_satisfied(Witgen::with_inputs(&[]).witness())
        );
    }

    #[test]
    #[should_panic(expected = "declared circuit coefficient width exceeded")]
    fn coefficient_overflow_is_rejected() {
        let _ = LinearCombination::<1>::from(Z::MAX) + LinearCombination::from(Z::ONE);
    }

    #[test]
    #[should_panic(expected = "declared circuit coefficient width exceeded")]
    fn public_subset_bound_is_checked() {
        let mut generator = ConstraintGenerator::new(1);
        let x = generator.bitz::<1>(generator.input(0));
        generator.assert_r1c::<1>(
            x + LinearCombination::from(Z::MAX),
            Z::ONE.into(),
            Z::ONE.into(),
        );
    }
}
