//! Borrowed MLEs with a tensor-product table and implicit zero padding.

use crate::piop::spartan::SpartanField;
use field::RingOps;

use crate::poly::EvaluationError;

/// Bit evaluations `[leading?, outer[0] * inner[..], outer[1] * inner[..], …]`.
/// The remaining entries of the `2^num_vars` domain are zero.
pub(crate) struct FactoredMultilinearExtension<'a, F: SpartanField> {
    field: F::Config,
    /// Domain size is `2^num_vars`, including zero padding.
    num_vars: usize,
    /// Zero, or one when a leading value precedes the tensor product.
    tensor_start: usize,
    /// Optional entry preceding the tensor product; zero when absent.
    leading_value: F,
    /// One multiplier per contiguous inner-factor block.
    outer_factor: &'a [F],
    /// Entries within each block.
    inner_factor: &'a [F],
    /// `tensor_start + outer_factor.len() * inner_factor.len()`.
    live_len: usize,
}

/// Bit evaluations `[outer ⊗ inner, tail]`, with `origin_adjustment` added
/// at index zero. The tail starts at the unpadded tensor-product length.
pub(crate) struct CompositeMultilinearExtension<'a, F: SpartanField> {
    /// Initial tensor-product segment, with no leading value.
    repeated: FactoredMultilinearExtension<'a, F>,
    /// Entries immediately following the unpadded tensor product.
    tail_evaluations: &'a [F],
    /// Added to the tensor product's entry at index zero.
    origin_adjustment: F,
    /// `repeated.live_len() + tail_evaluations.len()`.
    live_len: usize,
    /// Optional prover-side structure of the tail: sorted, disjoint runs
    /// `(start, len, base)` over tail indices with `tail[start + k] = base · 2^k`;
    /// tail entries outside every run are unstructured. Never read by
    /// [`Self::evaluation_at`], so it cannot change what is proved.
    tail_runs: Option<&'a [(usize, usize, F)]>,
}

impl<'a, F: SpartanField> FactoredMultilinearExtension<'a, F> {
    pub(crate) fn from_factors(
        num_vars: usize,
        outer_factor: &'a [F],
        inner_factor: &'a [F],
        cfg: &F::Config,
    ) -> Result<Self, EvaluationError> {
        Self::from_parts(num_vars, None, outer_factor, inner_factor, cfg)
    }

    pub(crate) fn with_leading_value(
        num_vars: usize,
        leading_value: F,
        outer_factor: &'a [F],
        inner_factor: &'a [F],
        cfg: &F::Config,
    ) -> Result<Self, EvaluationError> {
        Self::from_parts(
            num_vars,
            Some(leading_value),
            outer_factor,
            inner_factor,
            cfg,
        )
    }

    pub(crate) fn num_vars(&self) -> usize {
        self.num_vars
    }
    pub(crate) fn live_len(&self) -> usize {
        self.live_len
    }
    pub(crate) fn tensor_start(&self) -> usize {
        self.tensor_start
    }
    pub(crate) fn leading_value(&self) -> &F {
        &self.leading_value
    }
    pub(crate) fn outer_factor(&self) -> &[F] {
        self.outer_factor
    }
    pub(crate) fn inner_factor(&self) -> &[F] {
        self.inner_factor
    }

    pub(crate) fn evaluation_at(&self, index: usize) -> Result<F, EvaluationError> {
        if index >= domain_size(self.num_vars)? {
            return Err(EvaluationError::InvalidShape);
        }
        if index >= self.live_len {
            return Ok(F::zero_with_cfg(&self.field));
        }
        if index < self.tensor_start {
            return Ok(self.leading_value.clone());
        }
        let offset = index - self.tensor_start;
        Ok(self.field.mul(
            &self.outer_factor[offset / self.inner_factor.len()],
            &self.inner_factor[offset % self.inner_factor.len()],
        ))
    }

    pub(crate) fn evaluate(&self, point: &[F], cfg: &F::Config) -> Result<F, EvaluationError> {
        check_point(self.num_vars, point)?;
        if self.tensor_start == 0 && self.inner_factor.len().is_power_of_two() {
            let split = self.inner_factor.len().ilog2() as usize;
            return Ok(cfg.mul(
                &dot_with_equality(self.inner_factor, &point[..split], cfg),
                &dot_with_equality(self.outer_factor, &point[split..], cfg),
            ));
        }
        let weights = EqualityWeights::new(point, cfg);
        let mut value = F::zero_with_cfg(cfg);
        if self.tensor_start != 0 {
            value = cfg.add(&value, &cfg.mul(&self.leading_value, &weights.at(0)));
        }
        for (outer_index, outer) in self.outer_factor.iter().enumerate() {
            for (inner_index, inner) in self.inner_factor.iter().enumerate() {
                let index = self.tensor_start + outer_index * self.inner_factor.len() + inner_index;
                value = cfg.add(&value, &cfg.mul(&cfg.mul(outer, inner), &weights.at(index)));
            }
        }
        Ok(value)
    }

    fn from_parts(
        num_vars: usize,
        leading: Option<F>,
        outer_factor: &'a [F],
        inner_factor: &'a [F],
        cfg: &F::Config,
    ) -> Result<Self, EvaluationError> {
        if outer_factor.is_empty() || inner_factor.is_empty() {
            return Err(EvaluationError::EmptyPolynomial);
        }
        let tensor_start = usize::from(leading.is_some());
        let live_len = outer_factor
            .len()
            .checked_mul(inner_factor.len())
            .and_then(|len| len.checked_add(tensor_start))
            .ok_or(EvaluationError::Overflow)?;
        if live_len > domain_size(num_vars)? {
            return Err(EvaluationError::InvalidShape);
        }
        Ok(Self {
            field: cfg.clone(),
            num_vars,
            tensor_start,
            leading_value: leading.unwrap_or_else(|| F::zero_with_cfg(cfg)),
            outer_factor,
            inner_factor,
            live_len,
        })
    }
}

impl<'a, F: SpartanField> CompositeMultilinearExtension<'a, F> {
    pub(crate) fn from_parts(
        num_vars: usize,
        high: &'a [F],
        low: &'a [F],
        tail: &'a [F],
        origin_adjustment: F,
        cfg: &F::Config,
    ) -> Result<Self, EvaluationError> {
        if !low.len().is_power_of_two() {
            return Err(EvaluationError::InvalidShape);
        }
        let repeated = FactoredMultilinearExtension::from_factors(num_vars, high, low, cfg)?;
        let live_len = repeated
            .live_len()
            .checked_add(tail.len())
            .ok_or(EvaluationError::Overflow)?;
        if live_len > domain_size(num_vars)? {
            return Err(EvaluationError::InvalidShape);
        }
        Ok(Self {
            repeated,
            tail_evaluations: tail,
            origin_adjustment,
            live_len,
            tail_runs: None,
        })
    }

    /// Attaches the tail's geometric runs (prover-side acceleration only). The
    /// runs must be sorted by start, disjoint, and lie inside the tail.
    pub(crate) fn with_tail_runs(
        mut self,
        runs: &'a [(usize, usize, F)],
    ) -> Result<Self, EvaluationError> {
        let mut end = 0usize;
        for &(start, len, _) in runs {
            if start < end || len == 0 {
                return Err(EvaluationError::InvalidShape);
            }
            end = start.checked_add(len).ok_or(EvaluationError::Overflow)?;
        }
        if end > self.tail_evaluations.len() {
            return Err(EvaluationError::InvalidShape);
        }
        self.tail_runs = Some(runs);
        Ok(self)
    }

    pub(crate) fn tail_runs(&self) -> Option<&'a [(usize, usize, F)]> {
        self.tail_runs
    }

    pub(crate) fn num_vars(&self) -> usize {
        self.repeated.num_vars()
    }
    pub(crate) fn live_len(&self) -> usize {
        self.live_len
    }
    pub(crate) fn repeated(&self) -> &FactoredMultilinearExtension<'a, F> {
        &self.repeated
    }
    pub(crate) fn tail_evaluations(&self) -> &[F] {
        self.tail_evaluations
    }
    pub(crate) fn origin_adjustment(&self) -> &F {
        &self.origin_adjustment
    }

    pub(crate) fn evaluation_at(&self, index: usize) -> Result<F, EvaluationError> {
        if index >= domain_size(self.num_vars())? {
            return Err(EvaluationError::InvalidShape);
        }
        let mut value = if index < self.repeated.live_len() {
            self.repeated.evaluation_at(index)?
        } else if index < self.live_len {
            self.tail_evaluations[index - self.repeated.live_len()].clone()
        } else {
            F::zero_with_cfg(&self.repeated.field)
        };
        if index == 0 {
            value = self.repeated.field.add(&value, &self.origin_adjustment);
        }
        Ok(value)
    }

    pub(crate) fn evaluate(&self, point: &[F], cfg: &F::Config) -> Result<F, EvaluationError> {
        let mut value = self.repeated.evaluate(point, cfg)?;
        let weights = EqualityWeights::new(point, cfg);
        for (index, entry) in self.tail_evaluations.iter().enumerate() {
            value = cfg.add(
                &value,
                &cfg.mul(entry, &weights.at(self.repeated.live_len() + index)),
            );
        }
        value = cfg.add(&value, &cfg.mul(&self.origin_adjustment, &weights.at(0)));
        Ok(value)
    }
}

fn domain_size(num_vars: usize) -> Result<usize, EvaluationError> {
    let shift = u32::try_from(num_vars).map_err(|_| EvaluationError::Overflow)?;
    1usize.checked_shl(shift).ok_or(EvaluationError::Overflow)
}

fn check_point<F>(num_vars: usize, point: &[F]) -> Result<(), EvaluationError> {
    if point.len() != num_vars {
        return Err(EvaluationError::WrongPointWidth {
            expected: num_vars,
            actual: point.len(),
        });
    }
    Ok(())
}

/// Equality weights use O(sqrt(domain)) storage, including for unaligned tails.
pub(crate) struct EqualityWeights<F: SpartanField> {
    field: F::Config,
    low: Vec<F>,
    high: Vec<F>,
}

impl<F: SpartanField> EqualityWeights<F> {
    fn new(point: &[F], cfg: &F::Config) -> Self {
        let split = point.len() / 2;
        Self::from_tables(
            equality_table(&point[..split], cfg),
            equality_table(&point[split..], cfg),
            cfg,
        )
    }
    pub(crate) fn from_tables(low: Vec<F>, high: Vec<F>, field: &F::Config) -> Self {
        assert!(low.len().is_power_of_two() && high.len().is_power_of_two());
        Self {
            low,
            high,
            field: field.clone(),
        }
    }
    pub(crate) fn low(&self) -> &[F] {
        &self.low
    }
    pub(crate) fn high(&self) -> &[F] {
        &self.high
    }
    pub(crate) fn at(&self, index: usize) -> F {
        self.field.mul(
            &self.low[index & (self.low.len() - 1)],
            &self.high[index >> self.low.len().ilog2()],
        )
    }
}

fn equality_table<F: SpartanField>(point: &[F], cfg: &F::Config) -> Vec<F> {
    let len = 1usize
        .checked_shl(point.len() as u32)
        .expect("checked equality table shape");
    let mut out = vec![cfg.zero(); len];
    out[0] = cfg.one();
    for (i, r) in point.iter().enumerate() {
        let half = 1usize << i;
        for j in 0..half {
            let hi = cfg.mul(&out[j], r);
            out[j + half] = hi;
            out[j] = cfg.sub(&out[j], &hi);
        }
    }
    out
}

fn dot_with_equality<F: SpartanField>(values: &[F], point: &[F], cfg: &F::Config) -> F {
    let weights = EqualityWeights::new(point, cfg);
    values
        .iter()
        .enumerate()
        .fold(F::zero_with_cfg(cfg), |mut sum, (index, value)| {
            sum = cfg.add(&sum, &cfg.mul(value, &weights.at(index)));
            sum
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::mle::DenseMultilinearExtension;
    use field::{Fp as Prime, Uint};
    type F128 = Prime<2>;

    fn evaluate_dense(
        mle: &DenseMultilinearExtension<F128>,
        point: &[F128],
        cfg: &field::FpCtx<2>,
    ) -> F128 {
        let mut values = mle.evaluations.clone();
        for coordinate in point {
            values = values
                .chunks_exact(2)
                .map(|pair| cfg.add(&pair[0], &cfg.mul(&cfg.sub(&pair[1], &pair[0]), coordinate)))
                .collect();
        }
        values.pop().unwrap()
    }

    #[test]
    fn structured_evaluations_match_dense_with_boundaries_and_padding() {
        let cfg = F128::make_cfg(&Uint::from(97u128)).unwrap();
        let f = |n| F128::from_with_cfg(n, &cfg);
        let high = [f(2u64), f(3), f(5)];
        for width in [1usize, 2, 3, 4] {
            let low: Vec<_> = (0..width).map(|n| f(n as u64 + 7)).collect();
            for leading in [None, Some(f(19))] {
                let mle =
                    FactoredMultilinearExtension::from_parts(5, leading.clone(), &high, &low, &cfg)
                        .unwrap();
                let mut values: Vec<_> = leading.into_iter().collect();
                values.extend(high.iter().flat_map(|a| low.iter().map(|b| cfg.mul(a, b))));
                let dense = DenseMultilinearExtension::from_evaluations_vec(5, values, f(0));
                for (index, expected) in dense.evaluations.iter().enumerate() {
                    assert_eq!(&mle.evaluation_at(index).unwrap(), expected);
                }
                for point in [
                    vec![f(0); 5],
                    vec![f(1); 5],
                    vec![f(11), f(13), f(17), f(23), f(29)],
                ] {
                    assert_eq!(
                        mle.evaluate(&point, &cfg).unwrap(),
                        evaluate_dense(&dense, &point, &cfg)
                    );
                }
                assert!(mle.evaluate(&[f(0)], &cfg).is_err());
                assert!(mle.evaluation_at(32).is_err());
            }
            if width.is_power_of_two() {
                let tail = [f(31), f(37), f(41)];
                let mle =
                    CompositeMultilinearExtension::from_parts(5, &high, &low, &tail, f(43), &cfg)
                        .unwrap();
                let mut values: Vec<_> = high
                    .iter()
                    .flat_map(|a| low.iter().map(|b| cfg.mul(a, b)))
                    .collect();
                values.extend(tail.iter().cloned());
                values[0] = cfg.add(&values[0], &f(43));
                let dense = DenseMultilinearExtension::from_evaluations_vec(5, values, f(0));
                for index in 0..32 {
                    assert_eq!(mle.evaluation_at(index).unwrap(), dense.evaluations[index]);
                }
                let point = [f(11), f(13), f(17), f(23), f(29)];
                assert_eq!(
                    mle.evaluate(&point, &cfg).unwrap(),
                    evaluate_dense(&dense, &point, &cfg)
                );
                assert!(mle.evaluate(&point[..4], &cfg).is_err());
            }
        }
        assert!(FactoredMultilinearExtension::from_factors(0, &high, &high, &cfg).is_err());
        assert!(
            FactoredMultilinearExtension::from_factors(usize::MAX, &high, &high, &cfg).is_err()
        );
        assert!(FactoredMultilinearExtension::from_factors(3, &[], &high, &cfg).is_err());
        assert!(
            CompositeMultilinearExtension::from_parts(5, &high, &high, &[], f(0), &cfg).is_err()
        );
    }
}
