//! Variable-time public sampling. Never use this interface for private randomness.

use super::*;
use core::ops::RangeInclusive;

pub trait PublicRandomSource {
    /// Fill from an appropriate public RNG or transcript adapter.
    fn fill_bytes(&mut self, output: &mut [u8]);
}

#[derive(Clone, Copy, Debug)]
pub struct PrimeSearchPolicy {
    pub target_security_bits: u32,
    pub max_candidates: u64,
    pub max_rejection_draws: usize,
}
impl Default for PrimeSearchPolicy {
    fn default() -> Self {
        Self {
            target_security_bits: 128,
            max_candidates: 4096,
            max_rejection_draws: 256,
        }
    }
}
impl PrimeSearchPolicy {
    /// Minimum rejection budget bounding exhaustion over the entire search.
    /// Each masked rejection draw succeeds with probability at least 1/2.
    /// Union-bound all candidate draws and Miller–Rabin base draws separately
    /// from the whole-search composite-acceptance budget.
    pub fn minimum_rejection_draws(&self) -> Result<usize, PrimeSearchError> {
        self.budgets().map(|(_, draws)| draws)
    }

    fn budgets(&self) -> Result<(u32, usize), PrimeSearchError> {
        if self.max_candidates == 0 {
            return Err(PrimeSearchError::InvalidPolicy);
        }
        let security = self
            .target_security_bits
            .checked_add(16)
            .ok_or(PrimeSearchError::InvalidPolicy)?
            .max(128);
        let candidate_bits = 64 - (self.max_candidates - 1).leading_zeros();
        let rounds = security
            .checked_add(candidate_bits)
            .ok_or(PrimeSearchError::InvalidPolicy)?
            .div_ceil(2);
        // ceil(log2(rounds + 1)), avoiding multiplication by max_candidates.
        let draws_per_candidate_bits = 32 - rounds.leading_zeros();
        let rejection_bits = security
            .checked_add(candidate_bits)
            .and_then(|bits| bits.checked_add(draws_per_candidate_bits))
            .ok_or(PrimeSearchError::InvalidPolicy)?;
        Ok((rounds, rejection_bits as usize))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingError {
    EmptyRange,
    Exhausted,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimeSearchError {
    InvalidPolicy,
    InvalidInterval,
    Exhausted,
    Sampling(SamplingError),
}
impl core::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "public sampling: {self:?}")
    }
}
impl std::error::Error for SamplingError {}
impl core::fmt::Display for PrimeSearchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "public prime search: {self:?}")
    }
}
impl std::error::Error for PrimeSearchError {}

/// A tested candidate, carrying its already-prepared arithmetic constants.
#[derive(Clone, Debug)]
pub struct ProbablePrime<const L: usize> {
    parameters: PrimeParameters<L>,
}
impl<const L: usize> ProbablePrime<L> {
    pub fn modulus(&self) -> &Uint<L> {
        &self.parameters.modulus
    }
}

pub trait FieldSampling: FieldOps {
    fn sample_public(
        &self,
        source: &mut impl PublicRandomSource,
        max_draws: usize,
    ) -> Result<Self::Elem, SamplingError>;
    fn sample_nonzero_public(
        &self,
        source: &mut impl PublicRandomSource,
        max_draws: usize,
    ) -> Result<Self::Elem, SamplingError>;
}

pub(crate) fn bit_length_public<const L: usize>(value: &Uint<L>) -> usize {
    for i in (0..L).rev() {
        if value.0[i] != 0 {
            return i * 64 + (64 - value.0[i].leading_zeros()) as usize;
        }
    }
    0
}

pub(crate) fn sample_below<const L: usize>(
    source: &mut impl PublicRandomSource,
    bound: &Uint<L>,
    max_draws: usize,
) -> Result<Uint<L>, SamplingError> {
    let bits = bit_length_public(bound);
    if bits == 0 {
        return Err(SamplingError::EmptyRange);
    }
    let bits = bit_length_public(&bound.sbb(&Uint::ONE).0);
    if bits == 0 {
        return Ok(Uint::ZERO);
    }
    let words = bits.div_ceil(64);
    for _ in 0..max_draws {
        let mut candidate = Uint::ZERO;
        for word in &mut candidate.0[..words] {
            let mut bytes = [0u8; 8];
            source.fill_bytes(&mut bytes);
            *word = u64::from_le_bytes(bytes);
        }
        if bits % 64 != 0 {
            candidate.0[words - 1] &= (1u64 << (bits % 64)) - 1;
        }
        if candidate.ct_lt(bound).declassify() {
            return Ok(candidate);
        }
    }
    Err(SamplingError::Exhausted)
}

fn pow_public<const L: usize>(field: &FpCtx<L>, base: &Fp<L>, exponent: &Uint<L>) -> Fp<L> {
    // Sliding windows save multiplications in every Miller–Rabin round. Both
    // the exponent and the table index are public in this sampling interface.
    let mut remaining = bit_length_public(exponent);
    if remaining == 0 {
        return field.one();
    }
    let squared = field.square(base);
    let mut odd_powers = [*base; 8];
    for i in 1..odd_powers.len() {
        odd_powers[i] = field.mul(&odd_powers[i - 1], &squared);
    }
    let mut result = field.one();
    while remaining > 0 {
        if exponent.bit(remaining - 1).as_u64() == 0 {
            result = field.square(&result);
            remaining -= 1;
            continue;
        }
        let mut low = remaining.saturating_sub(4);
        while exponent.bit(low).as_u64() == 0 {
            low += 1;
        }
        let mut window = 0;
        for i in (low..remaining).rev() {
            result = field.square(&result);
            window = (window << 1) | exponent.bit(i).as_u64() as usize;
        }
        result = field.mul(&result, &odd_powers[window >> 1]);
        remaining = low;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_windows_match_binary_exponentiation() {
        // Include windows crossing the limb boundary and long runs of zeros.
        let field = FpCtx {
            parameters: PrimeParameters::new(Uint::<2>::from_words([
                u64::MAX,
                0x7fff_ffff_ffff_ffff,
            ])),
        };
        for words in [
            [0, 0],
            [1, 0],
            [0, 1],
            [u64::MAX, u64::MAX],
            [0x8000_0000_0000_0001, 0x8000_0000_0000_0001],
            [0x1234_5678_9abc_def0, 0x0fed_cba9_8765_4321],
        ] {
            let exponent = Uint::from_words(words);
            for value in [0, 1, 2, 17, u64::MAX] {
                let base = field.from_integer(&Uint::<2>::from_u64(value));
                assert_eq!(
                    pow_public(&field, &base, &exponent),
                    field.pow_ct(&base, &exponent)
                );
            }
        }
    }
}

fn small_prime_result<const L: usize>(candidate: &Uint<L>) -> Option<bool> {
    if candidate.ct_lt(&Uint::from_u64(2)).declassify() {
        return Some(false);
    }
    for divisor in [
        2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    ] {
        if *candidate == Uint::from_u64(divisor) {
            return Some(true);
        }
        let mut rem = 0u128;
        for word in candidate.0.iter().rev() {
            rem = ((rem << 64) | *word as u128) % divisor as u128;
        }
        if rem == 0 {
            return Some(false);
        }
    }
    None
}
fn factor_minus_one<const L: usize>(candidate: &Uint<L>) -> (Uint<L>, usize) {
    let mut odd = candidate.sbb(&Uint::ONE).0;
    let mut twos = 0;
    while odd.0[0] & 1 == 0 {
        odd = odd.shr(1);
        twos += 1;
    }
    (odd, twos)
}
fn strong_round<const L: usize>(
    field: &FpCtx<L>,
    base: &Uint<L>,
    odd: &Uint<L>,
    twos: usize,
) -> bool {
    let minus_one = field.neg(&field.one());
    let mut value = pow_public(field, &field.from_integer(base), odd);
    if value == field.one() || value == minus_one {
        return true;
    }
    for _ in 1..twos {
        value = field.square(&value);
        if value == minus_one {
            return true;
        }
    }
    false
}

/// Deterministic probable-prime check for declaration tests and debug assertions.
/// Search security uses independent randomized bases instead of these fixed bases.
pub fn is_probable_prime_public<const L: usize>(candidate: &Uint<L>) -> bool {
    if let Some(result) = small_prime_result(candidate) {
        return result;
    }
    let field = FpCtx {
        parameters: PrimeParameters::new(*candidate),
    };
    let (odd, twos) = factor_minus_one(candidate);
    for base in [
        2u64, 325, 9375, 28178, 450775, 9780504, 1795265022, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    ] {
        let reduced = field.params().remainder(&Uint::<1>::from_u64(base));
        if reduced == Uint::ZERO || reduced == Uint::ONE {
            continue;
        }
        if !strong_round(&field, &reduced, &odd, twos) {
            return false;
        }
    }
    true
}

/// Sample uniform odd candidates >=3, using bounded probabilistic primality tests.
/// Composite acceptance and rejection exhaustion each have a whole-search budget.
/// Bounded search is not a claim of an exactly uniform distribution over primes.
pub fn sample_prime_public<const L: usize>(
    source: &mut impl PublicRandomSource,
    interval: RangeInclusive<Uint<L>>,
    policy: &PrimeSearchPolicy,
) -> Result<ProbablePrime<L>, PrimeSearchError> {
    let (rounds, minimum_draws) = policy.budgets()?;
    if policy.max_rejection_draws < minimum_draws {
        return Err(PrimeSearchError::InvalidPolicy);
    }
    let (mut low, high) = interval.into_inner();
    if high.ct_lt(&low).declassify() || high.ct_lt(&Uint::from_u64(3)).declassify() {
        return Err(PrimeSearchError::InvalidInterval);
    }
    if low.ct_lt(&Uint::from_u64(3)).declassify() {
        low = Uint::from_u64(3);
    }
    // odd=2*k+1. This halves the interval without overflowing at Uint::MAX.
    let first = low.shr(1);
    let last = if high.0[0] & 1 == 1 {
        high.shr(1)
    } else {
        high.shr(1).sbb(&Uint::ONE).0
    };
    if last.ct_lt(&first).declassify() {
        return Err(PrimeSearchError::InvalidInterval);
    }
    let count = last.sbb(&first).0.adc(&Uint::ONE).0;
    for _ in 0..policy.max_candidates {
        let index = sample_below(source, &count, policy.max_rejection_draws)
            .map_err(PrimeSearchError::Sampling)?;
        let mut candidate = first.adc(&index).0.truncating_shl(1);
        candidate.0[0] |= 1;
        if let Some(result) = small_prime_result(&candidate) {
            if result {
                return Ok(ProbablePrime {
                    parameters: PrimeParameters::new(candidate),
                });
            }
            continue;
        }
        let field = FpCtx {
            parameters: PrimeParameters::new(candidate),
        };
        let (odd, twos) = factor_minus_one(&candidate);
        let base_count = candidate.sbb(&Uint::from_u64(3)).0;
        let mut accepted = true;
        for _ in 0..rounds {
            let base = sample_below(source, &base_count, policy.max_rejection_draws)
                .map_err(PrimeSearchError::Sampling)?
                .adc(&Uint::from_u64(2))
                .0;
            if !strong_round(&field, &base, &odd, twos) {
                accepted = false;
                break;
            }
        }
        if accepted {
            return Ok(ProbablePrime {
                parameters: field.parameters,
            });
        }
    }
    Err(PrimeSearchError::Exhausted)
}

impl<const L: usize> FpCtx<L> {
    pub fn from_prime(prime: ProbablePrime<L>) -> Self {
        Self {
            parameters: prime.parameters,
        }
    }
    pub fn sample_prime_public(
        source: &mut impl PublicRandomSource,
        interval: RangeInclusive<Uint<L>>,
        policy: &PrimeSearchPolicy,
    ) -> Result<Self, PrimeSearchError> {
        sample_prime_public(source, interval, policy).map(Self::from_prime)
    }
}

macro_rules! sampling {
    ([$($generic:tt)*] $provider:ty) => {
        impl<$($generic)*> FieldSampling for $provider {
            fn sample_public(&self, source: &mut impl PublicRandomSource, max_draws: usize) -> Result<Self::Elem, SamplingError> {
                let value = sample_below(source, self.modulus(), max_draws)?;
                Ok(PrimeValue::new(self.params().from_canonical(&value)))
            }
            fn sample_nonzero_public(&self, source: &mut impl PublicRandomSource, max_draws: usize) -> Result<Self::Elem, SamplingError> {
                let value = sample_below(source, &self.modulus().sbb(&Uint::ONE).0, max_draws)?.adc(&Uint::ONE).0;
                Ok(PrimeValue::new(self.params().from_canonical(&value)))
            }
        }
    };
}
sampling!([const L: usize] FpCtx<L>);
sampling!([P: PrimeSpec<L>, const L: usize] StaticFpOps<P,L>);

/// Declare a static prime and aliases, with a probable-prime test in the caller's tests.
///
/// ```
/// field::define_prime_field! {
///     pub ExamplePrime {
///         limbs: 2,
///         modulus: [17, 0],
///         element: ExampleFp,
///         context: ExampleField,
///     }
/// }
/// use field::{IntegerEmbedding, RingOps};
/// let field = ExampleField::new();
/// let a: ExampleFp = field.from_integer(&3u64);
/// assert_eq!(field.to_integer(&field.mul(&a, &a)).as_words(), &[9, 0]);
/// ```
#[macro_export]
macro_rules! define_prime_field {
    ($vis:vis $spec:ident { limbs: $limbs:expr, modulus: $modulus:expr, element: $element:ident, context: $context:ident $(,)? }) => {
        #[derive(Clone, Copy, Debug)]
        $vis struct $spec;
        impl $crate::PrimeSpec<$limbs> for $spec {
            const MODULUS: $crate::Uint<$limbs> = $crate::Uint::from_words($modulus);
        }
        $vis type $element = $crate::StaticFp<$spec, $limbs>;
        $vis type $context = $crate::StaticFpOps<$spec, $limbs>;
        #[cfg(test)]
        #[test]
        #[allow(non_snake_case)]
        fn $context() {
            assert!($crate::prime::is_probable_prime_public(&<$spec as $crate::PrimeSpec<$limbs>>::MODULUS), "static field declaration is composite");
        }
    };
}
