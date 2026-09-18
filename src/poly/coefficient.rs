//! Coefficient contracts used by polynomial storage and interpolation.
//! Arithmetic is supplied by `field`; these traits describe consumer needs.
use core::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::*,
};
use num_traits::{CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, One, Pow, Zero};

/// Fixed-width coefficients support both checked and wrapping polynomial work.
pub trait Coefficient:
    Sized
    + Debug
    + Display
    + Clone
    + Eq
    + Send
    + Sync
    + Hash
    + CheckedAdd
    + CheckedSub
    + CheckedMul
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
{
}

pub trait SignedCoefficient: Coefficient + Neg<Output = Self> + CheckedNeg {}
pub trait FixedCoefficient:
    Coefficient
    + Default
    + Zero
    + One
    + Sum
    + Product
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
{
}
impl<T> FixedCoefficient for T where
    T: Coefficient
        + Default
        + Zero
        + One
        + Sum
        + Product
        + for<'a> Sum<&'a Self>
        + for<'a> Product<&'a Self>
{
}

macro_rules! primitive_coefficients {
    ($($t:ty),*) => {$(impl Coefficient for $t {})*};
}
primitive_coefficients!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);
macro_rules! signed_coefficients {
    ($($t:ty),*) => {$(impl SignedCoefficient for $t {})*};
}
signed_coefficients!(i8, i16, i32, i64, i128);

/// Representation required by polynomial evaluation and transcript adapters.
pub trait FieldRepresentation:
    SignedCoefficient
    + Pow<u32, Output = Self>
    + Div<Output = Self>
    + DivAssign
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> DivAssign<&'a Self>
{
    type Inner: Debug + Eq + Clone + Send + Sync;
    type Modulus: Debug + Eq + Clone + Send + Sync;
    fn inner(&self) -> &Self::Inner;
    fn set_inner(&mut self, value: Self::Inner);
    fn into_inner(self) -> Self::Inner;
}

/// Fields used by the generic polynomial algorithms. Runtime prime-field
/// consumers use an explicit `field::FpCtx` instead of this static interface.
pub trait PolynomialField: FieldRepresentation {
    type Config: Debug + Clone + Send + Sync + 'static;
    fn cfg(&self) -> &Self::Config;
    fn config_from_modulus(modulus: &Self::Modulus) -> Option<Self::Config>;
    fn is_zero(value: &Self) -> bool;
    fn modulus(&self) -> Self::Modulus;
    fn new_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self;
    fn new_unchecked_with_cfg(inner: Self::Inner, cfg: &Self::Config) -> Self;
    fn zero_with_cfg(cfg: &Self::Config) -> Self;
    fn one_with_cfg(cfg: &Self::Config) -> Self;
    /// Enumerate distinct interpolation nodes. In a binary extension field,
    /// `index` encodes polynomial coefficients; this is not integer embedding.
    fn interpolation_node(index: u64, cfg: &Self::Config) -> Self;
}

/// Embed a coefficient into the evaluation field. Implementations choose the
/// algebraic embedding, independently of interpolation-node enumeration.
pub trait FromCoefficient<T>: PolynomialField {
    fn from_coefficient(value: T, cfg: &Self::Config) -> Self;
}
impl<F: PolynomialField + From<T>, T> FromCoefficient<T> for F {
    fn from_coefficient(value: T, _: &Self::Config) -> Self {
        Self::from(value)
    }
}
pub trait IntoCoefficient<F: PolynomialField> {
    fn into_coefficient(self, cfg: &F::Config) -> F;
}
impl<F: PolynomialField + FromCoefficient<T>, T> IntoCoefficient<F> for T {
    fn into_coefficient(self, cfg: &F::Config) -> F {
        F::from_coefficient(self, cfg)
    }
}

impl Coefficient for field::Bit {}
impl<const L: usize> Coefficient for field::Uint<L> {}
impl<const L: usize> Coefficient for field::Z<L> {}
impl<const L: usize> SignedCoefficient for field::Z<L> {}
