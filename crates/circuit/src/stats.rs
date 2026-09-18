//! A value-free circuit backend that only counts the generated layout.

use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use num_traits::{One, Zero};

use crate::{BoolWitness, Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

/// A zero-sized placeholder for coefficients and both kinds of witnesses.
///
/// Every operation discards its operands. This is sufficient for circuit
/// construction because [`Stats`] never invokes hint bodies or evaluates a
/// witness.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dummy;

impl From<bool> for Dummy {
    fn from(_: bool) -> Self {
        Self
    }
}

impl From<u64> for Dummy {
    fn from(_: u64) -> Self {
        Self
    }
}

impl BoolWitness for Dummy {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
}

impl Zero for Dummy {
    fn zero() -> Self {
        Self
    }

    fn is_zero(&self) -> bool {
        true
    }
}

impl One for Dummy {
    fn one() -> Self {
        Self
    }
}

impl Add for Dummy {
    type Output = Self;

    fn add(self, _: Self) -> Self::Output {
        Self
    }
}

impl AddAssign for Dummy {
    fn add_assign(&mut self, _: Self) {}
}

impl Sub for Dummy {
    type Output = Self;

    fn sub(self, _: Self) -> Self::Output {
        Self
    }
}

impl SubAssign for Dummy {
    fn sub_assign(&mut self, _: Self) {}
}

impl Mul for Dummy {
    type Output = Self;

    fn mul(self, _: Self) -> Self::Output {
        Self
    }
}

impl Neg for Dummy {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self
    }
}

impl Sum for Dummy {
    fn sum<I: Iterator<Item = Self>>(_: I) -> Self {
        Self
    }
}

/// The dimensions printed by Freigen's constraint-system statistics.
///
/// Freigen defines `mRows` as the number of BitZ rows plus one. Its `mCols` is
/// the largest referenced, zero-based witness index plus two; for a densely
/// allocated circuit whose final witness is referenced, that is the witness
/// count plus one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LeanStats {
    pub m_rows: usize,
    pub m_cols: usize,
    pub r1cs_rows: usize,
}

/// Counts circuit allocation without retaining symbolic expressions or values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    /// Boolean witnesses supplied as inputs or allocated by hints.
    pub witnesses: usize,
    /// Calls to [`Circuit::bitz`].
    pub bitz_calls: usize,
    /// Calls to [`Circuit::assert_r1c`].
    pub constraints: usize,
}

impl Stats {
    /// Starts a counter with `input_witnesses` already allocated.
    pub const fn new(input_witnesses: usize) -> Self {
        Self {
            witnesses: input_witnesses,
            bitz_calls: 0,
            constraints: 0,
        }
    }

    /// Accounts for additional externally allocated Boolean witnesses.
    pub fn add_input_witnesses(&mut self, count: usize) {
        self.witnesses = self
            .witnesses
            .checked_add(count)
            .expect("witness count overflow");
    }

    /// Converts the raw counts to Freigen's dimensions, assuming witnesses are
    /// densely allocated and the final allocated witness is referenced.
    pub const fn lean_stats(&self) -> LeanStats {
        LeanStats {
            m_rows: self.bitz_calls + 1,
            m_cols: self.witnesses + 1,
            r1cs_rows: self.constraints,
        }
    }
}

impl Circuit for Stats {
    type Bool = Dummy;
    type Coefficient<const LIMBS: usize> = Dummy;
    type Z<const LIMBS: usize> = Dummy;

    fn xor(&mut self, _: Dummy, _: Dummy) -> Dummy {
        Dummy
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _hint: H,
    ) -> ScalarBits<Dummy, N>
    where
        H: Fn(&dyn WitnessContext<Dummy, Dummy, Dummy>) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        self.witnesses = self
            .witnesses
            .checked_add(N)
            .expect("witness count overflow");
        ScalarBits([Dummy; N])
    }

    fn bitz<const LIMBS: usize>(&mut self, _: Dummy) -> Dummy {
        self.bitz_calls = self.bitz_calls.checked_add(1).expect("bitz count overflow");
        Dummy
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        _: &<Dummy as BoolWitness>::Repr<N, M>,
    ) -> (Dummy, Dummy) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        self.bitz_calls = self.bitz_calls.checked_add(N).expect("bitz count overflow");
        (Dummy, Dummy)
    }

    fn assert_r1c<const LIMBS: usize>(&mut self, _: Dummy, _: Dummy, _: Dummy) {
        self.constraints = self
            .constraints
            .checked_add(1)
            .expect("constraint count overflow");
    }
    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(&mut self, _: Dummy) -> Dummy {
        Dummy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_arithmetic_discards_values() {
        let mut value = Dummy::one();
        value += Dummy;
        value -= Dummy;
        assert_eq!(-value * Dummy, Dummy);
        assert!(<Dummy as Zero>::zero().is_zero());
    }

    #[test]
    fn counts_inputs_and_operations() {
        let mut stats = Stats::new(3);
        stats.add_input_witnesses(2);
        let _: ScalarBits<Dummy, 4> =
            stats.hint::<1, 4, 1, _>(|_| Ok(PackedBits::<4, 1>::from_u64(0)));
        stats.bitz::<1>(Dummy);
        stats.assert_r1c::<1>(Dummy, Dummy, Dummy);
        assert_eq!(stats.sign_extend_z::<1, 128>(Dummy), Dummy);

        assert_eq!(
            stats,
            Stats {
                witnesses: 9,
                bitz_calls: 1,
                constraints: 1,
            }
        );
        assert_eq!(
            stats.lean_stats(),
            LeanStats {
                m_rows: 2,
                m_cols: 10,
                r1cs_rows: 1,
            }
        );
    }
}
