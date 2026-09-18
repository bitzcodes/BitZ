//! Backend-independent operations used to build an BitZ circuit.
//!
//! The traits in this crate deliberately separate symbolic witnesses from the
//! values used while generating a witness.  In particular, a [`Circuit::hint`]
//! receives a [`WitnessContext`], through which it can evaluate captured
//! symbolic witnesses when witness generation is run.

use std::array;
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use num_traits::{One, Zero};

pub mod constraints;
#[cfg(feature = "full")]
pub mod ecdsa_sha256;
pub mod integer_storage;
pub mod linear_map;
pub mod matrix_products;
#[cfg(feature = "full")]
#[cfg(test)]
mod matrix_reference;
#[cfg(feature = "full")]
pub mod matrix_transpose;
#[cfg(feature = "full")]
pub mod montgomery_tail;
#[cfg(feature = "full")]
pub mod p256;
pub mod sha256;
pub mod stats;
pub mod witgen;

#[cfg(all(test, feature = "full"))]
mod projection_tests;

/// An error raised while evaluating a witness hint.
///
/// Hints are fallible because their inputs may not be in the domain expected by
/// a gadget (for example, an unsigned decomposition hint may receive a negative
/// value).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HintError {
    message: String,
}

impl HintError {
    /// Creates a hint error with a human-readable message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for HintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for HintError {}

impl From<String> for HintError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for HintError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// The result returned by a witness hint.
pub type HintResult<T> = Result<T, HintError>;

/// Values available to a hint while generating a witness.
///
/// A context is intentionally read-only and exposes only evaluation.  This
/// prevents a hint from adding constraints or allocating witnesses as a side
/// effect.  The symbolic witness types may be handles, linear combinations, or
/// another backend-specific representation.
pub trait WitnessContext<ZW, BW, C> {
    /// Evaluates a Z-side symbolic witness (or linear combination).
    fn eval_z(&self, witness: &ZW) -> C;

    /// Borrows the exact little-endian two's-complement storage of a Z-side
    /// value when the backend has a fixed-width integer representation.
    ///
    /// This lets arithmetic-heavy hints avoid allocating a `BigInt` merely to
    /// inspect a value. Symbolic backends may retain the default.
    fn eval_z_words<'a>(&self, _witness: &'a ZW) -> Option<&'a [u64]> {
        None
    }

    /// Evaluates an F2-side symbolic witness (or linear combination).
    fn eval_bool(&self, witness: &BW) -> bool;
}

/// Arithmetic required of coefficients on the Z side of an BitZ circuit.
///
/// This is a ring-like programming interface, rather than a claim that every
/// implementation obeys the ring laws.  It is kept separate from the symbolic
/// witness type so backends are not forced to use integers for coefficients.
pub trait Coefficient:
    Clone
    + From<u64>
    + Zero
    + One
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Sum<Self>
{
}

impl<T> Coefficient for T where
    T: Clone
        + From<u64>
        + Zero
        + One
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Mul<Output = T>
        + Neg<Output = T>
        + Sum<T>
{
}

/// Arithmetic required of a symbolic Z witness or linear combination.
///
/// Scalar multiplication is written `witness * coefficient`.  Putting the
/// backend-local witness type on the left lets generic backends implement the
/// operation even when the coefficient type comes from another crate.
pub trait ZWitness<C: Coefficient>:
    Clone
    + Send
    + Sync
    + 'static
    + From<C>
    + Zero
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
    + Mul<C, Output = Self>
    + Sum<Self>
{
}

impl<T, C> ZWitness<C> for T
where
    C: Coefficient,
    T: Clone
        + Send
        + Sync
        + 'static
        + From<C>
        + Zero
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Neg<Output = T>
        + Mul<C, Output = T>
        + Sum<T>,
{
}

/// Storage and word operations selected by a Boolean witness representation.
pub trait BoolRepresentation<BW, const N: usize, const M: usize>:
    Clone + Send + Sync + 'static
where
    BW: BoolWitness,
{
    /// Number of packed `u64` limbs used by this representation.
    const LIMBS: usize = M;

    fn from_array(bits: [BW; N]) -> Self;
    fn from_packed(bits: PackedBits<N, M>) -> Self;
    fn from_u64(value: u64) -> Self;
    fn bit(&self, index: usize) -> BW;
    fn slice<const K: usize, const L: usize>(
        &self,
        offset: usize,
    ) -> <BW as BoolWitness>::Repr<K, L> {
        assert!(offset.checked_add(K).is_some_and(|end| end <= N));
        <BW as BoolWitness>::Repr::from_array(array::from_fn(|index| self.bit(offset + index)))
    }
    fn xor<CS>(&self, circuit: &mut CS, rhs: &Self) -> Self
    where
        CS: Circuit<Bool = BW>;
    fn rotate_right(&self, amount: usize) -> Self;
    fn shift_right(&self, amount: usize) -> Self;
    fn evaluate<ZW, C>(&self, context: &dyn WitnessContext<ZW, BW, C>) -> PackedBits<N, M>;
}

/// Scalar storage for symbolic Boolean witnesses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarBits<BW, const N: usize>(pub [BW; N]);

impl<BW, const N: usize, const M: usize> BoolRepresentation<BW, N, M> for ScalarBits<BW, N>
where
    BW: BoolWitness,
{
    fn from_array(bits: [BW; N]) -> Self {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        Self(bits)
    }

    fn from_packed(bits: PackedBits<N, M>) -> Self {
        Self(array::from_fn(|index| BW::from(bits.bit(index))))
    }

    fn from_u64(value: u64) -> Self {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        Self(array::from_fn(|bit| {
            BW::from(bit < 64 && (value >> bit) & 1 == 1)
        }))
    }

    fn bit(&self, index: usize) -> BW {
        self.0[index].clone()
    }

    fn xor<CS>(&self, circuit: &mut CS, rhs: &Self) -> Self
    where
        CS: Circuit<Bool = BW>,
    {
        Self(array::from_fn(|i| {
            circuit.xor(self.0[i].clone(), rhs.0[i].clone())
        }))
    }

    fn rotate_right(&self, amount: usize) -> Self {
        if N == 0 {
            return self.clone();
        }
        let amount = amount % N;
        Self(array::from_fn(|i| self.0[(i + amount) % N].clone()))
    }

    fn shift_right(&self, amount: usize) -> Self {
        Self(array::from_fn(|i| {
            i.checked_add(amount)
                .filter(|&source| source < N)
                .map_or_else(BW::zero, |source| self.0[source].clone())
        }))
    }

    fn evaluate<ZW, C>(&self, context: &dyn WitnessContext<ZW, BW, C>) -> PackedBits<N, M> {
        PackedBits::from_fn(|index| context.eval_bool(&self.0[index]))
    }
}

/// An arbitrary-width bit string packed least-significant-bit first.
///
/// Stable Rust cannot yet use `N.div_ceil(64)` directly as an array length, so
/// `M` is explicit and every constructor asserts `M == N.div_ceil(64)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedBits<const N: usize, const M: usize = 1> {
    words: [u64; M],
}

impl<const N: usize, const M: usize> PackedBits<N, M> {
    pub const BITS: usize = N;
    pub const LIMBS: usize = M;

    fn assert_width() {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
    }

    /// Constructs an all-zero packed value.
    pub fn zero() -> Self {
        Self::assert_width();
        Self { words: [0; M] }
    }

    /// Packs an array of Boolean values.
    pub fn from_array(bits: [bool; N]) -> Self {
        Self::from_fn(|index| bits[index])
    }

    /// Builds a packed value by evaluating one function per bit.
    pub fn from_fn(mut bit: impl FnMut(usize) -> bool) -> Self {
        Self::assert_width();
        let mut words = [0_u64; M];
        for index in 0..N {
            words[index / 64] |= u64::from(bit(index)) << (index % 64);
        }
        Self { words }
    }

    /// Constructs a packed value whose low 64 bits are `value` and whose
    /// remaining bits are zero.
    pub fn from_u64(value: u64) -> Self {
        Self::assert_width();
        let mut words = [0; M];
        if M != 0 {
            words[0] = value;
        }
        if !N.is_multiple_of(64) && M != 0 {
            words[M - 1] &= (1_u64 << (N % 64)) - 1;
        }
        Self { words }
    }

    /// Constructs a packed value from little-endian `u64` limbs.
    pub fn from_words(mut words: [u64; M]) -> Self {
        Self::assert_width();
        if !N.is_multiple_of(64) && M != 0 {
            words[M - 1] &= (1_u64 << (N % 64)) - 1;
        }
        Self { words }
    }

    /// Packed little-endian storage limbs.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Returns bit `index`.
    pub fn bit(&self, index: usize) -> bool {
        assert!(index < N);
        self.words()[index / 64] >> (index % 64) & 1 == 1
    }

    /// Returns the low 64 bits, truncating wider values.
    pub fn low_u64(&self) -> u64 {
        self.words().first().copied().unwrap_or(0)
    }

    /// Returns the value as a `u64` when all upper bits are zero.
    pub fn to_u64(&self) -> Option<u64> {
        self.words()
            .get(1..)
            .unwrap_or_default()
            .iter()
            .all(|word| *word == 0)
            .then(|| self.low_u64())
    }

    /// Returns the low `K` bits using `L` limbs.
    pub fn truncate<const K: usize, const L: usize>(&self) -> PackedBits<K, L> {
        PackedBits::from_fn(|index| index < N && self.bit(index))
    }
}

impl<const N: usize, const M: usize> Default for PackedBits<N, M> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize, const M: usize> BoolRepresentation<bool, N, M> for PackedBits<N, M> {
    fn from_array(bits: [bool; N]) -> Self {
        Self::from_array(bits)
    }

    fn from_packed(bits: PackedBits<N, M>) -> Self {
        bits
    }

    fn from_u64(value: u64) -> Self {
        Self::from_u64(value)
    }

    fn bit(&self, index: usize) -> bool {
        self.bit(index)
    }

    fn slice<const K: usize, const L: usize>(&self, offset: usize) -> PackedBits<K, L> {
        assert!(offset.checked_add(K).is_some_and(|end| end <= N));
        let first_word = offset / 64;
        let shift = offset % 64;
        PackedBits::from_words(array::from_fn(|index| {
            let source = first_word + index;
            let low = self.words.get(source).copied().unwrap_or(0) >> shift;
            let high = if shift == 0 {
                0
            } else {
                self.words.get(source + 1).copied().unwrap_or(0) << (64 - shift)
            };
            low | high
        }))
    }

    fn xor<CS>(&self, _: &mut CS, rhs: &Self) -> Self
    where
        CS: Circuit<Bool = bool>,
    {
        Self {
            words: array::from_fn(|index| self.words[index] ^ rhs.words[index]),
        }
    }

    fn rotate_right(&self, amount: usize) -> Self {
        if N == 0 {
            return Self::zero();
        }
        let amount = amount % N;
        if M == 1 {
            let value = self.low_u64();
            return if amount == 0 {
                *self
            } else {
                Self::from_u64((value >> amount) | (value << (N - amount)))
            };
        }
        Self::from_fn(|index| self.bit((index + amount) % N))
    }

    fn shift_right(&self, amount: usize) -> Self {
        if amount >= N {
            return Self::zero();
        }
        if M == 1 {
            return Self::from_u64(self.low_u64() >> amount);
        }
        Self::from_fn(|index| index + amount < N && self.bit(index + amount))
    }

    fn evaluate<ZW, C>(&self, _: &dyn WitnessContext<ZW, bool, C>) -> PackedBits<N, M> {
        *self
    }
}

/// Operations required of a symbolic Boolean witness or F2 linear combination.
pub trait BoolWitness: Clone + From<bool> + Send + Sync + Sized + 'static {
    /// Backend-selected storage for an `N`-bit word.
    type Repr<const N: usize, const M: usize>: BoolRepresentation<Self, N, M>;

    /// The additive identity in F2.
    fn zero() -> Self {
        Self::from(false)
    }
}

impl BoolWitness for bool {
    type Repr<const N: usize, const M: usize> = PackedBits<N, M>;
}

/// Operations needed to construct an BitZ circuit.
///
/// Z witnesses and coefficients are backend-owned type families indexed by a
/// gadget-local limb bound. This lets one polymorphic circuit combine gadgets
/// with different evaluation widths without imposing the largest width on the
/// entire witness generator.
pub trait Circuit {
    type Bool: BoolWitness;
    type Coefficient<const LIMBS: usize>: Coefficient;
    type Z<const LIMBS: usize>: ZWitness<Self::Coefficient<LIMBS>>;

    /// Constructs a nonnegative coefficient from little-endian machine words.
    ///
    /// The default only relies on ring operations. Fixed-integer backends
    /// should override this to copy limbs directly.
    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Self::Coefficient<LIMBS> {
        let mut radix = Self::Coefficient::<LIMBS>::one();
        for _ in 0..64 {
            radix += radix.clone();
        }
        words
            .iter()
            .rev()
            .fold(Self::Coefficient::<LIMBS>::zero(), |value, word| {
                value * radix.clone() + Self::Coefficient::<LIMBS>::from(*word)
            })
    }

    /// Adds two Boolean expressions over F2.
    ///
    /// Boolean operations receive the backend explicitly so recording
    /// backends can append to an arena without shared mutability in witness
    /// handles.
    fn xor(&mut self, lhs: Self::Bool, rhs: Self::Bool) -> Self::Bool;

    /// Allocates `N` Boolean witnesses whose packed values are computed by
    /// `hint`.
    ///
    /// The closure is called during witness generation, not necessarily while
    /// the circuit is being built.  It can capture symbolic witnesses and use
    /// the supplied sub-context to evaluate them.  Consequently, captured data
    /// must be owned and safe to retain and invoke from a worker thread.
    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        hint: H,
    ) -> <Self::Bool as BoolWitness>::Repr<N, M>
    where
        H: Fn(
                &dyn WitnessContext<Self::Z<LIMBS>, Self::Bool, Self::Coefficient<LIMBS>>,
            ) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static;

    /// Converts an F2 linear combination into a constrained Z witness.
    fn bitz<const LIMBS: usize>(&mut self, value: Self::Bool) -> Self::Z<LIMBS>;

    /// Lifts a little-endian bit-vector and returns both its full unsigned
    /// value and the value of its low `LOW` bits.
    ///
    /// This is a fusion hook: its default implementation is exactly `N`
    /// calls to [`Circuit::bitz`] followed by the corresponding linear
    /// combinations. Value-oriented backends can override it to evaluate a
    /// packed machine word directly, while layout-building backends retain the
    /// scalar `bitz` calls and therefore produce the identical circuit.
    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        bits_le: &<Self::Bool as BoolWitness>::Repr<N, M>,
    ) -> (Self::Z<LIMBS>, Self::Z<LIMBS>) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        let mut full = Self::Z::<LIMBS>::zero();
        let mut low = Self::Z::<LIMBS>::zero();
        let mut power = Self::Coefficient::<LIMBS>::one();
        for index in 0..N {
            let lifted = self.bitz::<LIMBS>(bits_le.bit(index));
            let term = lifted * power.clone();
            full += term.clone();
            if index < LOW {
                low += term;
            }
            power += power.clone();
        }
        (full, low)
    }

    /// Asserts the rank-1 constraint `a * b = c` on the Z side.
    fn assert_r1c<const LIMBS: usize>(
        &mut self,
        a: Self::Z<LIMBS>,
        b: Self::Z<LIMBS>,
        c: Self::Z<LIMBS>,
    );

    /// Explicitly changes a Z witness into a wider local representation.
    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: Self::Z<FROM_LIMBS>,
    ) -> Self::Z<TO_LIMBS>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Mod7(i32);

    impl Mod7 {
        fn new(value: i32) -> Self {
            Self(value.rem_euclid(7))
        }
    }

    impl From<u64> for Mod7 {
        fn from(value: u64) -> Self {
            Self::new((value % 7) as i32)
        }
    }

    impl Zero for Mod7 {
        fn zero() -> Self {
            Self(0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }

    impl One for Mod7 {
        fn one() -> Self {
            Self(1)
        }
    }

    impl Add for Mod7 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self::new(self.0 + rhs.0)
        }
    }

    impl AddAssign for Mod7 {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sub for Mod7 {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(self.0 - rhs.0)
        }
    }

    impl SubAssign for Mod7 {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl Mul for Mod7 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self::new(self.0 * rhs.0)
        }
    }

    impl Neg for Mod7 {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self::new(-self.0)
        }
    }

    impl Sum for Mod7 {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), Add::add)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Z(Mod7);

    impl From<Mod7> for Z {
        fn from(value: Mod7) -> Self {
            Self(value)
        }
    }

    impl Zero for Z {
        fn zero() -> Self {
            Self(Mod7::zero())
        }

        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }

    impl Add for Z {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl AddAssign for Z {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sub for Z {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl SubAssign for Z {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl Neg for Z {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl Mul<Mod7> for Z {
        type Output = Self;

        fn mul(self, rhs: Mod7) -> Self::Output {
            Self(self.0 * rhs)
        }
    }

    impl Sum for Z {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self(Mod7::zero()), Add::add)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Bit(bool);

    impl From<bool> for Bit {
        fn from(value: bool) -> Self {
            Self(value)
        }
    }

    impl Zero for Bit {
        fn zero() -> Self {
            Self(false)
        }

        fn is_zero(&self) -> bool {
            !self.0
        }
    }

    impl Add for Bit {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 != rhs.0)
        }
    }

    impl AddAssign for Bit {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sum for Bit {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self(false), Add::add)
        }
    }

    impl BoolWitness for Bit {
        type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
    }

    struct Values;

    impl WitnessContext<Z, Bit, Mod7> for Values {
        fn eval_z(&self, witness: &Z) -> Mod7 {
            witness.0
        }

        fn eval_bool(&self, witness: &Bit) -> bool {
            witness.0
        }
    }

    #[derive(Default)]
    struct TestCircuit {
        constraints: Vec<(Z, Z, Z)>,
    }

    impl Circuit for TestCircuit {
        type Bool = Bit;
        type Coefficient<const LIMBS: usize> = Mod7;
        type Z<const LIMBS: usize> = Z;

        fn xor(&mut self, lhs: Bit, rhs: Bit) -> Bit {
            lhs + rhs
        }

        fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
            &mut self,
            hint: H,
        ) -> ScalarBits<Bit, N>
        where
            H: Fn(&dyn WitnessContext<Z, Bit, Mod7>) -> HintResult<PackedBits<N, M>>
                + Send
                + Sync
                + 'static,
        {
            ScalarBits::from_packed(hint(&Values).expect("test hint should succeed"))
        }

        fn bitz<const LIMBS: usize>(&mut self, value: Bit) -> Z {
            Z(Mod7::new(i32::from(value.0)))
        }

        fn assert_r1c<const LIMBS: usize>(&mut self, a: Z, b: Z, c: Z) {
            self.constraints.push((a, b, c));
        }

        fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(&mut self, value: Z) -> Z {
            value
        }
    }

    #[test]
    fn hints_evaluate_both_witness_kinds_through_the_context() {
        let mut circuit = TestCircuit::default();
        let z = Z(Mod7::new(5));
        let bit = Bit(true);
        let hint_z = z;
        let hint_bit = bit;

        let result = circuit.hint::<1, 2, 1, _>(move |context| {
            let z = context.eval_z(&hint_z).0;
            let bit = context.eval_bool(&hint_bit);
            Ok(PackedBits::<2, 1>::from_array([z >= 4 && bit, z % 2 == 1]))
        });
        let [high, low] = result.0;

        assert_eq!([high, low], [Bit(true), Bit(true)]);

        let lifted = circuit.bitz::<1>(high);
        let expression = lifted * Mod7::new(3) + Z::from(Mod7::new(2));
        circuit.assert_r1c::<1>(Z::from(Mod7::one()), expression, z);
        assert_eq!(circuit.constraints, vec![(Z(Mod7::one()), z, z)]);
    }

    #[test]
    fn hint_error_preserves_its_message() {
        let error = HintError::new("negative input");
        assert_eq!(error.message(), "negative input");
        assert_eq!(error.to_string(), "negative input");
    }
}
