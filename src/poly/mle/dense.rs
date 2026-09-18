mod try_collect_dense_mle;

use core::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{
    ops::{Deref, DerefMut},
    slice::SliceIndex,
};

use crate::poly::coefficient::{Coefficient, PolynomialField, SignedCoefficient};
use crate::poly::{
    EvaluationError,
    mle::{MultilinearExtension, MultilinearExtensionRand},
};

use crate::utils::{
    CHECKED, add, cfg_into_iter, inner_transparent_field::InnerTransparentField,
    mul_by_scalar::MulByScalar, projectable_to_field::ProjectableToField, sub,
};
use rand::{distr::StandardUniform, prelude::*};

use super::MultilinearExtensionWithConfig;

pub use try_collect_dense_mle::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<T> {
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: Vec<T>,
    /// Number of variables
    pub num_vars: usize,
}

impl<R> DenseMultilinearExtension<R> {
    pub fn zero_vars(evaluation: R) -> Self {
        Self {
            evaluations: vec![evaluation],
            num_vars: 0,
        }
    }
}

impl<R: Clone> DenseMultilinearExtension<R> {
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[R], zero: R) -> Self {
        Self::from_evaluations_vec(num_vars, evaluations.to_vec(), zero)
    }

    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<R>, zero: R) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert!(
            evaluations.len() <= 1 << num_vars,
            "The size of evaluations should not exceed 2^num_vars. \n eval len: {:?}. num vars: {num_vars}",
            evaluations.len()
        );

        if evaluations.len() != 1 << num_vars {
            let mut evaluations = evaluations;
            evaluations.resize(1 << num_vars, zero);
            return Self {
                num_vars,
                evaluations,
            };
        }

        Self {
            num_vars,
            evaluations,
        }
    }
}

impl<R: Default> DenseMultilinearExtension<R> {
    pub fn from_evaluations_vec_pad(mut evaluations: Vec<R>) -> Self {
        let len = evaluations.len();

        evaluations.resize_with(len.next_power_of_two(), Default::default);

        let num_vars = crate::utils::log2(evaluations.len()) as usize;

        Self {
            evaluations,
            num_vars,
        }
    }
}

impl<R: Clone> DenseMultilinearExtension<R> {
    pub fn from_evaluations_vec_pad_with_zero(mut evaluations: Vec<R>, zero: &R) -> Self {
        let len = evaluations.len();

        evaluations.resize(len.next_power_of_two(), zero.clone());

        let num_vars = crate::utils::log2(evaluations.len()) as usize;

        Self {
            evaluations,
            num_vars,
        }
    }
}

// Keeping Send bound here to match the FromParallelIterator impl
impl<R: Send + Default> FromIterator<R> for DenseMultilinearExtension<R> {
    fn from_iter<T: IntoIterator<Item = R>>(iter: T) -> Self {
        Self::from_evaluations_vec_pad(iter.into_iter().collect())
    }
}

impl<R> Deref for DenseMultilinearExtension<R> {
    type Target = [R];

    fn deref(&self) -> &Self::Target {
        &self.evaluations
    }
}

impl<R> DerefMut for DenseMultilinearExtension<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evaluations
    }
}

impl<R> IntoIterator for DenseMultilinearExtension<R> {
    type Item = R;

    type IntoIter = std::vec::IntoIter<R>;

    fn into_iter(self) -> Self::IntoIter {
        self.evaluations.into_iter()
    }
}

#[cfg(feature = "parallel")]
impl<R: Send + Default> FromParallelIterator<R> for DenseMultilinearExtension<R> {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = R>,
    {
        Self::from_evaluations_vec_pad(par_iter.into_par_iter().collect())
    }
}

#[cfg(feature = "parallel")]
impl<R: Send + Sync> IntoParallelIterator for DenseMultilinearExtension<R> {
    type Iter = rayon::vec::IntoIter<R>;

    type Item = R;

    fn into_par_iter(self) -> Self::Iter {
        self.evaluations.into_par_iter()
    }
}

#[cfg(feature = "parallel")]
impl<'data, R: Send + Sync> IntoParallelRefIterator<'data> for &'data DenseMultilinearExtension<R> {
    type Iter = rayon::slice::Iter<'data, R>;

    type Item = &'data R;

    fn par_iter(&'data self) -> Self::Iter {
        self.evaluations.par_iter()
    }
}

#[cfg(feature = "parallel")]
impl<'data, R: Send + Sync> IntoParallelRefMutIterator<'data>
    for &'data mut DenseMultilinearExtension<R>
{
    type Iter = rayon::slice::IterMut<'data, R>;

    type Item = &'data mut R;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.evaluations.par_iter_mut()
    }
}

impl<R: Coefficient> DenseMultilinearExtension<R> {
    pub fn evaluate<S>(&self, point: &[S], zero: R) -> Result<R, EvaluationError>
    where
        R: for<'a> MulByScalar<&'a S>,
    {
        if point.len() == self.num_vars {
            Ok(self
                .fixed_variables(point, zero)
                .into_iter()
                .next()
                .expect("Evaluations should not be empty"))
        } else {
            Err(EvaluationError::WrongPointWidth {
                expected: self.num_vars,
                actual: point.len(),
            })
        }
    }

    fn unary<G>(&mut self, f: G)
    where
        G: FnMut(&mut R),
    {
        self.iter_mut().for_each(f);
    }

    fn binary<G>(&mut self, other: &Self, mut f: G)
    where
        G: FnMut(&mut R, &R),
    {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| f(a, b));
    }
}

impl<F> MultilinearExtensionWithConfig<F> for DenseMultilinearExtension<F::Inner>
where
    F: InnerTransparentField,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn fix_variables_with_config(
        &mut self,
        partial_point: &[F],
        config: &<F as PolynomialField>::Config,
    ) {
        assert!(
            partial_point.len() <= self.num_vars,
            "too many partial points"
        );

        if partial_point.len().is_zero() {
            return;
        }

        let nv = self.num_vars;
        let dim = partial_point.len();

        let mut r = partial_point[0].clone();
        for i in 1..dim + 1 {
            for b in 0..1 << (nv - i) {
                r.set_inner(partial_point[i - 1].inner().clone());
                if self[2 * b + 1] != self[2 * b] {
                    // a = f(1) - f(0)
                    let a = F::sub_inner(&self[2 * b + 1], &self[2 * b], config);

                    // self[b] = f(0) + r * a
                    r.mul_assign_by_inner(&a);
                    self[b] = F::add_inner(&self[2 * b], r.inner(), config);
                } else {
                    self[b] = self[2 * b].clone();
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = sub!(nv, dim);
    }

    fn fixed_variables_with_config(
        &self,
        partial_point: &[F],
        config: &<F as PolynomialField>::Config,
    ) -> Self {
        let mut res = self.clone();
        res.fix_variables_with_config(partial_point, config);
        res
    }

    fn evaluate_with_config(
        mut self,
        point: &[F],
        config: &<F as PolynomialField>::Config,
    ) -> Result<F, EvaluationError> {
        if point.len() == self.num_vars {
            self.fix_variables_with_config(point, config);
            Ok(F::new_unchecked_with_cfg(
                self.into_iter()
                    .next()
                    .expect("Evaluations should not be empty"),
                config,
            ))
        } else {
            Err(EvaluationError::WrongPointWidth {
                expected: point.len(),
                actual: self.num_vars,
            })
        }
    }
}

impl<R> MultilinearExtension<R> for DenseMultilinearExtension<R>
where
    R: Coefficient,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn fix_variables<S>(&mut self, partial_point: &[S], zero: R)
    where
        R: for<'a> MulByScalar<&'a S>,
    {
        assert!(
            partial_point.len() <= self.num_vars,
            "too many partial points"
        );

        let nv = self.num_vars;
        let dim = partial_point.len();

        for i in 1..dim + 1 {
            let r = &partial_point[i - 1];
            for b in 0..1 << (nv - i) {
                let left = &self[2 * b];
                let right = &self[2 * b + 1];
                // a = f(1) - f(0)
                let a = sub!(*right, left);
                if a != zero {
                    // self[b] = f(0) + r * a
                    let ar = a
                        .mul_by_scalar::<CHECKED>(r)
                        .expect("Multiplication overflow");
                    self[b] = add!(*left, ar);
                } else {
                    self[b] = left.clone();
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = sub!(nv, dim);
    }

    fn fixed_variables<S>(&self, partial_point: &[S], zero: R) -> Self
    where
        R: for<'a> MulByScalar<&'a S>,
    {
        let mut res = self.clone();
        res.fix_variables(partial_point, zero);
        res
    }
}

impl<R> MultilinearExtensionRand<R> for DenseMultilinearExtension<R>
where
    R: Send + Clone + Default,
    StandardUniform: Distribution<R>,
{
    fn rand<G: Rng + ?Sized>(num_vars: usize, rng: &mut G) -> Self {
        (0..1 << num_vars).map(|_| rng.random::<R>()).collect()
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for DenseMultilinearExtension<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for DenseMultilinearExtension<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.evaluations[index]
    }
}

impl<R: SignedCoefficient> Neg for DenseMultilinearExtension<R> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.unary(|v| *v = v.checked_neg().expect("Negation overflow"));
        self
    }
}

impl<R: Coefficient> Add for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<R: Coefficient> Add<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a += b);
        self
    }
}

impl<R: Coefficient> Sub<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a -= b);
        self
    }
}

impl<R: Coefficient> Mul<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a *= b);
        self
    }
}

impl<R: Coefficient> Mul<R> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(mut self, rhs: R) -> Self::Output {
        self.unary(|v| *v *= &rhs);
        self
    }
}

impl<R: Coefficient> AddAssign<&Self> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn add_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a += b);
    }
}

impl<R: Coefficient> SubAssign<&Self> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn sub_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a -= b);
    }
}

impl<R: Coefficient> MulAssign<&Self> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn mul_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a *= b);
    }
}

impl<R: Coefficient> AddAssign<(R, &Self)> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn add_assign(&mut self, rhs: (R, &Self)) {
        let coeff = rhs.0;
        self.binary(rhs.1, |a, b| *a += b.clone() * &coeff);
    }
}

pub fn project_coeffs<F: PolynomialField, R: ProjectableToField<F> + Send + Sync>(
    mle: DenseMultilinearExtension<R>,
    sampled_value: &F,
) -> DenseMultilinearExtension<F::Inner> {
    let projection = R::prepare_projection(sampled_value);

    DenseMultilinearExtension {
        evaluations: cfg_into_iter!(mle.evaluations)
            .map(|x| projection(&x).into_inner())
            .collect(),
        num_vars: mle.num_vars,
    }
}

pub trait CollectDenseMleWithZero: Iterator {
    fn collect_dense_mle_with_zero(
        self,
        zero: &Self::Item,
    ) -> DenseMultilinearExtension<Self::Item>;
}

impl<T> CollectDenseMleWithZero for T
where
    T: Iterator,
    T::Item: Clone,
{
    fn collect_dense_mle_with_zero(
        self,
        zero: &Self::Item,
    ) -> DenseMultilinearExtension<Self::Item> {
        let evaluations = self.collect();

        DenseMultilinearExtension::from_evaluations_vec_pad_with_zero(evaluations, zero)
    }
}
