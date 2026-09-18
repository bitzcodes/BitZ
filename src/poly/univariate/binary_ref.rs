use crate::poly::coefficient::{Coefficient, PolynomialField};
use crate::poly::{
    ConstCoeffBitWidth, EvaluatablePolynomial, EvaluationError, Polynomial,
    univariate::{F2AddAssign, dense::DensePolynomial, prepare_projection},
};
use crate::transcript::traits::{ConstTranscribable, GenTranscribable};
use crate::utils::{
    from_ref::FromRef,
    inner_product::{BooleanInnerProductAdd, InnerProduct, InnerProductError},
    mul_by_scalar::MulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};
use derive_more::{
    Add, AddAssign, AsRef, Display, From, Mul, MulAssign, Product, Sub, SubAssign, Sum,
};
use field::Bit;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstZero, One, Zero};
use rand::{distr::StandardUniform, prelude::*};
use std::{
    array,
    hash::Hash,
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign, Sub, SubAssign},
};

#[derive(
    Add,
    AddAssign,
    AsRef,
    Clone,
    Copy,
    Debug,
    From,
    Default,
    Display,
    Hash,
    PartialEq,
    Eq,
    Mul,
    MulAssign,
    Sub,
    SubAssign,
    Sum,
    Product,
)]
#[repr(transparent)]
pub struct BinaryRefPoly<const DEGREE_PLUS_ONE: usize>(DensePolynomial<Bit, DEGREE_PLUS_ONE>);

impl<const DEGREE_PLUS_ONE: usize> BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    pub const fn inner(&self) -> &DensePolynomial<Bit, DEGREE_PLUS_ONE> {
        &self.0
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<BinaryRefPoly<DEGREE_PLUS_ONE>>
    for DensePolynomial<Bit, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(binary_poly: BinaryRefPoly<DEGREE_PLUS_ONE>) -> Self {
        binary_poly.0
    }
}

impl From<u32> for BinaryRefPoly<32> {
    fn from(value: u32) -> Self {
        Self(DensePolynomial {
            coeffs: array::from_fn(|i| Bit::from_bool(value & (1 << i) != 0)),
        })
    }
}

impl From<u64> for BinaryRefPoly<64> {
    fn from(value: u64) -> Self {
        Self(DensePolynomial {
            coeffs: array::from_fn(|i| Bit::from_bool(value & (1 << i) != 0)),
        })
    }
}

impl<const DEGREE_PLUS_ONE: usize> BinaryRefPoly<DEGREE_PLUS_ONE> {
    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new(coeffs: impl AsRef<[Bit]>) -> Self {
        Self(DensePolynomial::new(coeffs))
    }

    /// Create a new polynomial with the given coefficients.
    /// If the input has fewer than N+1 coefficients, the remaining slots will
    /// be filled with zeros. If the input has more than N+1 coefficients,
    /// it will panic.
    #[inline(always)]
    pub fn new_padded(coeffs: impl AsRef<[Bit]>) -> Self {
        Self(DensePolynomial::new_with_zero(coeffs, Bit::ZERO))
    }
}

impl<const DEGREE_PLUS_ONE: usize> Zero for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn zero() -> Self {
        Self(DensePolynomial::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<const DEGREE_PLUS_ONE: usize> One for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn one() -> Self {
        Self(DensePolynomial::one())
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Add<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Sub<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<const DEGREE_PLUS_ONE: usize> Mul for BinaryRefPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Mul<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> AddAssign<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0.add_assign(&rhs.0);
    }
}

impl<const DEGREE_PLUS_ONE: usize> F2AddAssign for BinaryRefPoly<DEGREE_PLUS_ONE> {
    /// XOR each coefficient — bypasses `Bit::AddAssign`'s overflow
    /// check so that `1 + 1 = 0` rather than panicking.
    #[inline(always)]
    fn f2_add_assign(&mut self, rhs: &Self) {
        for (a, b) in self.0.coeffs.iter_mut().zip(rhs.0.coeffs.iter()) {
            *a = Bit::from_bool((bool::from(*a)) ^ (bool::from(*b)));
        }
    }
}

impl<const DEGREE_PLUS_ONE: usize> crate::poly::univariate::F2PackU64
    for BinaryRefPoly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn pack_u64(&self) -> u64 {
        assert!(
            DEGREE_PLUS_ONE <= 64,
            "F2PackU64 requires DEGREE_PLUS_ONE <= 64; got {DEGREE_PLUS_ONE}",
        );
        let mut v: u64 = 0;
        for (i, c) in self.0.coeffs.iter().enumerate() {
            if bool::from(*c) {
                #[allow(clippy::arithmetic_side_effects)]
                {
                    v |= 1u64 << i;
                }
            }
        }
        v
    }

    #[inline(always)]
    fn unpack_u64(value: u64) -> Self {
        assert!(
            DEGREE_PLUS_ONE <= 64,
            "F2PackU64 requires DEGREE_PLUS_ONE <= 64; got {DEGREE_PLUS_ONE}",
        );
        Self(DensePolynomial {
            coeffs: array::from_fn(|i| Bit::from_bool((value & (1u64 << i)) != 0)),
        })
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> SubAssign<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.0.sub_assign(&rhs.0);
    }
}

impl<const DEGREE_PLUS_ONE: usize> MulAssign for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(&rhs.0);
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> MulAssign<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Self) {
        self.0.mul_assign(&rhs.0);
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedAdd for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0.checked_add(&other.0)?))
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedSub for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0.checked_sub(&other.0)?))
    }
}

impl<const DEGREE_PLUS_ONE: usize> CheckedMul for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        Some(Self(self.0.checked_mul(&other.0)?))
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Sum<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| &x.0).sum())
    }
}

impl<'a, const DEGREE_PLUS_ONE: usize> Product<&'a Self> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| &x.0).product())
    }
}

impl<const DEGREE_PLUS_ONE: usize> Coefficient for BinaryRefPoly<DEGREE_PLUS_ONE> {}

impl<const DEGREE_PLUS_ONE: usize> Distribution<BinaryRefPoly<DEGREE_PLUS_ONE>>
    for StandardUniform
{
    #[inline(always)]
    fn sample<Gen: Rng + ?Sized>(&self, rng: &mut Gen) -> BinaryRefPoly<DEGREE_PLUS_ONE> {
        let coeffs: [Bit; DEGREE_PLUS_ONE] = array::from_fn(|_| Bit::from_bool(rng.random()));

        // I didn't manage to delegate this one to
        // `DensePolynomial::sample` because of unsatisfied
        // traits.

        BinaryRefPoly(DensePolynomial::new(coeffs))
    }
}

//
// Zip-specific traits
//
impl<const DEGREE_PLUS_ONE: usize> Polynomial<Bit> for BinaryRefPoly<DEGREE_PLUS_ONE> {
    const DEGREE_BOUND: usize = DensePolynomial::<Bit, DEGREE_PLUS_ONE>::DEGREE_BOUND;
}

impl<R: Clone + Zero + One + CheckedAdd + CheckedMul, const DEGREE_PLUS_ONE: usize>
    EvaluatablePolynomial<Bit, R> for BinaryRefPoly<DEGREE_PLUS_ONE>
{
    type EvaluationPoint = R;

    fn evaluate_at_point(&self, point: &R) -> Result<R, EvaluationError> {
        if DEGREE_PLUS_ONE.is_one() {
            return Ok(R::zero());
        }

        let result = self.0.coeffs[1..]
            .iter()
            .try_fold(
                (
                    if bool::from(self.0.coeffs[0]) {
                        R::one()
                    } else {
                        R::zero()
                    },
                    R::one(),
                ),
                |(mut acc, mut pow), coeff| {
                    pow = pow.checked_mul(point).ok_or(EvaluationError::Overflow)?;

                    if bool::from(*coeff) {
                        acc = acc.checked_add(&pow).ok_or(EvaluationError::Overflow)?;
                    }

                    Ok((acc, pow))
                },
            )?
            .0;

        Ok(result)
    }
}

impl<const DEGREE_PLUS_ONE: usize> ConstCoeffBitWidth for BinaryRefPoly<DEGREE_PLUS_ONE> {
    const COEFF_BIT_WIDTH: usize = DensePolynomial::<Bit, DEGREE_PLUS_ONE>::COEFF_BIT_WIDTH;
}

impl<const DEGREE_PLUS_ONE: usize> Named for BinaryRefPoly<DEGREE_PLUS_ONE> {
    fn type_name() -> String {
        format!("BPoly<{}>", Self::DEGREE_BOUND)
    }
}

impl<const DEGREE_PLUS_ONE: usize> GenTranscribable for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let value = u64::read_transcription_bytes_exact(bytes);
        Self(DensePolynomial {
            coeffs: array::from_fn(|i| Bit::from_bool(value & (1 << i) != 0)),
        })
    }

    #[inline(always)]
    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let mut value: u64 = 0;

        self.0.coeffs.iter().enumerate().for_each(|(i, coeff)| {
            if bool::from(*coeff) {
                value |= 1 << i;
            }
        });

        value.write_transcription_bytes_exact(buf);
    }
}

impl<const DEGREE_PLUS_ONE: usize> ConstTranscribable for BinaryRefPoly<DEGREE_PLUS_ONE> {
    const NUM_BYTES: usize = u64::NUM_BYTES;
}

impl<const DEGREE_PLUS_ONE: usize> FromRef<BinaryRefPoly<DEGREE_PLUS_ONE>>
    for BinaryRefPoly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from_ref(poly: &BinaryRefPoly<DEGREE_PLUS_ONE>) -> Self {
        poly.clone()
    }
}

impl<const DEGREE_PLUS_ONE: usize> From<&BinaryRefPoly<DEGREE_PLUS_ONE>>
    for BinaryRefPoly<DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from(value: &BinaryRefPoly<DEGREE_PLUS_ONE>) -> Self {
        Self::from_ref(value)
    }
}

impl<const DEGREE_PLUS_ONE: usize> BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Bit> {
        self.0.iter()
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Bit> {
        self.0.iter_mut()
    }
}

impl<const DEGREE_PLUS_ONE: usize> Deref for BinaryRefPoly<DEGREE_PLUS_ONE> {
    type Target = [Bit];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<const DEGREE_PLUS_ONE: usize> DerefMut for BinaryRefPoly<DEGREE_PLUS_ONE> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

#[derive(Clone, Debug)]
pub struct BinaryRefPolyInnerProduct<R, const DEGREE_PLUS_ONE: usize>(PhantomData<R>);

impl<Rhs, Out, const DEGREE_PLUS_ONE: usize> InnerProduct<BinaryRefPoly<DEGREE_PLUS_ONE>, Rhs, Out>
    for BinaryRefPolyInnerProduct<Rhs, DEGREE_PLUS_ONE>
where
    Rhs: Clone,
    Out: FromRef<Rhs> + CheckedAdd + field::CtSelect + Clone,
{
    #[inline(always)]
    fn inner_product<const CHECK: bool>(
        lhs: &BinaryRefPoly<DEGREE_PLUS_ONE>,
        rhs: &[Rhs],
        zero: Out,
    ) -> Result<Out, InnerProductError> {
        BooleanInnerProductAdd::inner_product::<CHECK>(&lhs.0.coeffs, rhs, zero)
    }
}

impl<F, const DEGREE_PLUS_ONE: usize> ProjectableToField<F> for BinaryRefPoly<DEGREE_PLUS_ONE>
where
    F: PolynomialField + FromRef<F> + 'static,
{
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static {
        prepare_projection::<F, Self, _, DEGREE_PLUS_ONE>(sampled_value, |poly, i| {
            bool::from(poly.0.coeffs[i])
        })
    }
}

// This could've been more generic, but keeping implementation consistent with
// `BinaryU64Poly`.
impl<const DEGREE_PLUS_ONE: usize> MulByScalar<&i64, DensePolynomial<i64, DEGREE_PLUS_ONE>>
    for BinaryRefPoly<DEGREE_PLUS_ONE>
{
    fn mul_by_scalar<const CHECK: bool>(
        &self,
        rhs: &i64,
    ) -> Option<DensePolynomial<i64, DEGREE_PLUS_ONE>> {
        let mut coeffs: [i64; DEGREE_PLUS_ONE] = [0_i64; DEGREE_PLUS_ONE];

        coeffs.iter_mut().enumerate().for_each(|(i, out)| {
            if bool::from(self.0.coeffs[i]) {
                *out = *rhs;
            }
        });

        Some(DensePolynomial { coeffs })
    }
}
