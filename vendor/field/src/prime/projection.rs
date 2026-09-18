use super::{Fp, FpCtx};
use crate::{CtMask, CtOrd, CtSelect, IntegerEmbedding, Uint, Z};

/// Signed projection with a prepared radix and public-width sign corrections.
/// Two-limb fields use a prepared limb dot product and one Montgomery reduction.
/// Other widths use Horner steps in the canonical representation.
#[derive(Clone, Debug)]
pub struct PreparedSignedProjection<const L: usize> {
    field: FpCtx<L>,
    radix: Fp<L>,
    width_powers: Vec<Uint<L>>,
    native_words_fit: bool,
    dot128: Option<SignedDot128>,
}

#[derive(Clone, Debug)]
struct SignedDot128 {
    /// B^i R^2 mod p, where B=2^64 and R=2^128.
    weights: Vec<[u64; 2]>,
    /// B^i R mod p, for the signed two's-complement correction.
    corrections: Vec<[u64; 2]>,
}

impl SignedDot128 {
    fn project(&self, words: &[u64], modulus: [u64; 2], neg_inv: u64) -> [u64; 2] {
        // Every term is below B*p. There are fewer than B terms on supported
        // targets, so the sum is below p*R and fits the REDC input bound. No
        // operand-dependent trimming, lookup, or reduction is performed.
        const { assert!(usize::BITS <= 64) };
        let mut acc = [0u64; 4];
        for (&word, &[lo, hi]) in words.iter().zip(&self.weights) {
            let low = word as u128 * lo as u128 + acc[0] as u128;
            let mid = word as u128 * hi as u128 + acc[1] as u128 + (low >> 64);
            let high = acc[2] as u128 + (mid >> 64);
            acc = [
                low as u64,
                mid as u64,
                high as u64,
                acc[3] + (high >> 64) as u64,
            ];
        }
        super::montgomery128::redc(acc, modulus, neg_inv)
    }
}

impl<const L: usize> PreparedSignedProjection<L> {
    pub fn new(field: FpCtx<L>, max_words: usize) -> Self {
        let radix = field.from_integer(&(1u128 << 64));
        let mut width_powers =
            Vec::with_capacity(max_words.checked_add(1).expect("projection width overflow"));
        width_powers.push(Uint::ONE);
        for index in 0..max_words {
            width_powers.push(field.params().mul(&width_powers[index], &radix.words));
        }
        let native_words_fit = Uint::from_u64(u64::MAX).ct_lt(field.modulus()).declassify();
        let dot128 = (L == 2).then(|| {
            let corrections: Vec<_> = width_powers
                .iter()
                .map(|power| field.params().from_canonical(power))
                .collect();
            let weights = corrections[..max_words]
                .iter()
                .map(|power| {
                    let scaled = field.params().from_canonical(power);
                    [scaled.0[0], scaled.0[1]]
                })
                .collect();
            SignedDot128 {
                weights,
                corrections: corrections
                    .iter()
                    .map(|power| [power.0[0], power.0[1]])
                    .collect(),
            }
        });
        Self {
            field,
            radix,
            width_powers,
            native_words_fit,
            dot128,
        }
    }

    /// Declared maximum input width, fixed before reading private values.
    pub fn max_words(&self) -> usize {
        self.width_powers.len() - 1
    }

    /// Explicit canonical output for the next canonical-input operation.
    pub fn project_canonical(&self, words: &[u64]) -> Uint<L> {
        assert!(
            words.len() <= self.max_words(),
            "integer exceeds prepared width"
        );
        if let Some(dot) = &self.dot128 {
            self.field.params().to_canonical(&self.dot_kernel(words, dot))
        } else {
            self.canonical_kernel(words)
        }
    }

    pub fn project(&self, words: &[u64]) -> Fp<L> {
        assert!(
            words.len() <= self.max_words(),
            "integer exceeds prepared width"
        );
        if let Some(dot) = &self.dot128 {
            return Fp::new(self.dot_kernel(words, dot));
        }
        Fp::new(
            self.field
                .params()
                .from_canonical(&self.canonical_kernel(words)),
        )
    }

    /// Reuses output and validates the public shape once before the row loop.
    pub fn project_into<const N: usize>(&self, input: &[Z<N>], out: &mut [Fp<L>]) {
        assert_eq!(input.len(), out.len());
        assert!(N <= self.max_words(), "integer exceeds prepared width");
        if let Some(dot) = &self.dot128 {
            for (value, slot) in input.iter().zip(out) {
                *slot = Fp::new(self.dot_kernel(value.as_words(), dot));
            }
            return;
        }
        for (value, slot) in input.iter().zip(out) {
            *slot = Fp::new(
                self.field
                    .params()
                    .from_canonical(&self.canonical_kernel(value.as_words())),
            );
        }
    }

    fn dot_kernel(&self, words: &[u64], dot: &SignedDot128) -> Uint<L> {
        let params = self.field.params();
        // dot128 exists only for L=2; its tables cover every allowed width.
        let projected = dot.project(
            words,
            [params.modulus.0[0], params.modulus.0[1]],
            params.neg_inv,
        );
        let projected = Uint(core::array::from_fn(|i| projected[i]));
        let correction = Uint(core::array::from_fn(|i| dot.corrections[words.len()][i]));
        let corrected = params.sub(&projected, &correction);
        let negative = CtMask::from_lsb(words.last().copied().unwrap_or(0) >> 63);
        Uint::ct_select(&projected, &corrected, negative)
    }

    fn canonical_kernel(&self, words: &[u64]) -> Uint<L> {
        let params = self.field.params();
        let mut plain = Uint::ZERO;
        for &word in words.iter().rev() {
            let native = Uint::from_u64(word);
            let native = if self.native_words_fit {
                native
            } else {
                params.remainder(&native)
            };
            plain = params.add(&params.mul(&plain, &self.radix.words), &native);
        }
        let corrected = params.sub(&plain, &self.width_powers[words.len()]);
        let negative = CtMask::from_lsb(words.last().copied().unwrap_or(0) >> 63);
        Uint::ct_select(&plain, &corrected, negative)
    }
}
