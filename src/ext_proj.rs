//! Step-3 random-prime projection for **extension-field** evaluation claims
//! (paper `c:core_iop`, Step 3).
//!
//! When the evaluation field `K = F_q[X]/(h(X))` is a proper extension
//! (degree `e ≥ 2`), the linear claim `⟨π_q(bits), v⟩ = μ` cannot ride the
//! `GF(2^128)` exponent fold directly: the lifted row weights
//! `π_canon^{-1}(v^{(1)})` are integer *polynomials*, not integers. The
//! paper's Step 3 collapses the claim to a prime field first:
//!
//! 1. the prover sends the exact integer-polynomial folds
//!    `μ_c = ⟨bits_c, π_canon^{-1}(v^{(1)})⟩ ∈ ℤ[X]` (Step 1; per-coefficient
//!    chunk folds in this implementation),
//! 2. the verifier samples a **random prime** `q'` from the set
//!    `𝒫 = {primes in [2^{bits−1}, 2^{bits})}` and a point `α' ∈ F_{q'}`,
//! 3. both sides project the row weights,
//!    `γ = π_{q'}^{-1}(π_{q',α'}(π_canon^{-1}(v^{(1)}))) ∈ [0, q')^{2^t}`,
//!    and run the ordinary mod-`q'` opening on `γ`; the verifier finally
//!    checks the certified folds against the Step-1 polynomials at `α'`:
//!    `⟨bits_c, γ⟩ ≡ μ_c(α') (mod q')`.
//!
//! A lie in the sent `μ_c` is a nonzero difference polynomial of degree
//! `< e` with `~(c_w+t+W)`-bit coefficients; by the generalized
//! Schwartz–Zippel / prime-divisibility argument (paper `l:reduction_lemma`)
//! it survives the random `(q', α')` with probability
//! `≈ B/(log q'·|𝒫|) + (e−1)/q'` — negligible at the default 100-bit primes.
//!
//! This module holds the pieces that are *new* relative to the prime-field
//! path: the transcript prime/point sampling (deterministic and identical on
//! both sides), fast mod-`q'` scalar arithmetic for a runtime modulus
//! (Montgomery arithmetic from `field`), and the weight
//! projection `γ`. The opening itself reuses the existing mod-`q` pipeline
//! verbatim (see `ligerito_flock::prove_mle_eval_ext_ligerito`).

use crate::poly::univariate::binary_gf128::Gf128 as Gf;
use crate::transcript::traits::Transcript;
use crate::utils::cfg_into_iter;

use field::{FpCtx, PrimeSearchPolicy, PublicRandomSource, Uint};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Protocol parameters of the Step-3 projection. Prover and verifier must
/// agree on these (they are part of the protocol description, like the code
/// or the chunk width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtProjParams {
    /// Bit-length of the sampled projection primes: `𝒫` is the set of
    /// primes in `[2^{prime_bits−1}, 2^{prime_bits})`. Also the `q_bits`
    /// under which the projected claim's chunk count is derived.
    pub prime_bits: usize,
    /// Number of transcript-derived Miller–Rabin bases (on top of a fixed
    /// base-2 pre-filter). A composite candidate survives all of them with
    /// probability ≤ `4^{-mr_rounds}` per tested composite, so 64 rounds push
    /// that candidate's acceptance probability to `2^{-128}`.
    pub mr_rounds: usize,
}

impl Default for ExtProjParams {
    fn default() -> Self {
        Self {
            prime_bits: 100,
            mr_rounds: 64,
        }
    }
}

impl ExtProjParams {
    /// Panic on parameter combinations the arithmetic below does not
    /// support. `prime_bits ≤ 120` keeps the existing projection
    /// protocol interval and chunk-shift bounds;
    /// `≥ 32` keeps the candidate range clear of the tiny primes and the
    /// Miller–Rabin base range `[2, q'−2]` nonempty.
    pub fn validate(&self) {
        assert!(
            (32..=120).contains(&self.prime_bits),
            "ExtProjParams::prime_bits must be in [32, 120]; got {}",
            self.prime_bits
        );
        assert!(
            (1..=256).contains(&self.mr_rounds),
            "ExtProjParams::mr_rounds must be in [1, 256]; got {}",
            self.mr_rounds
        );
    }
}

/// One uniform 128-bit integer squeezed from the transcript (the two
/// little-endian words of a `GF(2^128)` challenge — the same encoding
/// [`crate::pcs::fq_challenge`] uses).
fn transcript_u128(transcript: &mut impl Transcript) -> u128 {
    let g: Gf = transcript.get_field_challenge(&());
    let w = g.as_words();
    u128::from(w[0]) | (u128::from(w[1]) << 64)
}

/// A 256-bit transcript draw reduced modulo `m` (statistical distance
/// `≤ m/2^256 ≈ 2^{-156}` from uniform at 100-bit `m` — the plain 128-bit
/// draw would be `~2^{-28}`-biased, which matters for the soundness-carrying
/// `α'` and the Miller–Rabin bases).
#[allow(clippy::arithmetic_side_effects)] // reductions keep everything < m < 2^127
fn transcript_uniform_mod(transcript: &mut impl Transcript, m: u128) -> u128 {
    debug_assert!(m > 1 && m < (1u128 << 127));
    let lo = transcript_u128(transcript);
    let hi = transcript_u128(transcript);
    use field::{IntegerEmbedding, ModRingCtx};
    let ring = ModRingCtx::new(Uint::from(m)).expect("public sampling modulus exceeds one");
    let draw = Uint::from_words([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64]);
    u128::from(ring.to_integer(&ring.from_integer(&draw)))
}

/// Errors returned by the bounded transcript prime sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PrimeSamplingError {
    #[error(transparent)]
    Shared(#[from] field::PrimeSearchError),

    #[error("prime interval is empty: min {min} exceeds max {max}")]
    InvalidInterval { min: u128, max: u128 },
    #[error("prime interval maximum must be below 2^126; got {max}")]
    MaximumTooLarge { max: u128 },
    #[error("prime interval [{min}, {max}] contains no odd candidate")]
    NoOddCandidate { min: u128, max: u128 },
    #[error("exact-uniform transcript sampling exhausted for upper bound {upper_bound}")]
    UniformSamplingExhausted { upper_bound: u128 },
    #[error("no prime found in [{min}, {max}] after {attempts} transcript candidates")]
    PrimeSearchExhausted {
        min: u128,
        max: u128,
        attempts: usize,
    },
}

pub(crate) struct TranscriptRandom<'a, T: Transcript + ?Sized>(pub &'a mut T);
impl<T: Transcript + ?Sized> PublicRandomSource for TranscriptRandom<'_, T> {
    fn fill_bytes(&mut self, output: &mut [u8]) {
        self.0.fill_sampling_bytes(output);
    }
}

/// Shared bounded search, including full-width 128-bit intervals. The
/// interval and whole-search policy are bound before internal sampling reads.
/// The caller owns any protocol grinding boundary around this search.
pub fn sample_prime_context(
    transcript: &mut impl Transcript,
    min: u128,
    max: u128,
    security_bits: u32,
) -> Result<FpCtx<2>, PrimeSamplingError> {
    if min > max {
        return Err(PrimeSamplingError::InvalidInterval { min, max });
    }
    let first = min.max(3) | 1;
    if first > max {
        return Err(PrimeSamplingError::NoOddCandidate { min, max });
    }
    let attempts = if first == (max - u128::from(max & 1 == 0)) {
        1
    } else {
        64 * (128 - max.leading_zeros()) as usize
    };
    let mut policy = PrimeSearchPolicy {
        target_security_bits: security_bits,
        max_candidates: attempts as u64,
        max_rejection_draws: 0,
    };
    policy.max_rejection_draws = policy.minimum_rejection_draws()?;
    transcript.absorb_slice(b"bitz/shared-prime-sampling/v1");
    transcript.absorb_slice(&min.to_le_bytes());
    transcript.absorb_slice(&max.to_le_bytes());
    transcript.absorb_slice(&policy.target_security_bits.to_le_bytes());
    transcript.absorb_slice(&policy.max_candidates.to_le_bytes());
    transcript.absorb_slice(&(policy.max_rejection_draws as u64).to_le_bytes());
    FpCtx::sample_prime_public(
        &mut TranscriptRandom(transcript),
        Uint::from(min)..=Uint::from(max),
        &policy,
    )
    .map_err(|error| match error {
        field::PrimeSearchError::Exhausted => {
            PrimeSamplingError::PrimeSearchExhausted { min, max, attempts }
        }
        error => error.into(),
    })
}

/// Bounded projection-prime search over the existing supported interval.
/// Candidate and base rejection are handled by the shared sampler; bounded
/// probable-prime search is not described as exactly uniform over primes.
pub fn sample_prime_in_interval(
    transcript: &mut impl Transcript,
    min: u128,
    max: u128,
) -> Result<u128, PrimeSamplingError> {
    let _g = tracing::info_span!("ext:sample_interval_prime").entered();
    if min > max {
        return Err(PrimeSamplingError::InvalidInterval { min, max });
    }
    if max >= (1u128 << 126) {
        return Err(PrimeSamplingError::MaximumTooLarge { max });
    }
    sample_prime_context(transcript, min, max, 128).map(|field| u128::from(*field.modulus()))
}

/// Sample the projection prime with an error on bounded-search exhaustion.
/// The configured per-candidate rounds set the minimum security target;
/// the shared policy additionally accounts for the complete search.
pub fn sample_proj_prime(
    transcript: &mut impl Transcript,
    proj: &ExtProjParams,
) -> Result<u128, PrimeSamplingError> {
    let _g = tracing::info_span!("ext:sample_prime").entered();
    proj.validate();
    let top = 1u128 << (proj.prime_bits - 1);
    sample_prime_context(transcript, top, top | (top - 1), 2 * proj.mr_rounds as u32)
        .map(|field| u128::from(*field.modulus()))
}

/// Sample the Step-3 evaluation point `α' ∈ F_{q'}` from the transcript
/// (256-bit reduction — see [`transcript_uniform_mod`]).
pub fn sample_proj_point(transcript: &mut impl Transcript, q_proj: u128) -> u128 {
    transcript_uniform_mod(transcript, q_proj)
}

/// The projected row weights of Step 3:
/// `γ_b = π_{q'}^{-1}(π_{q',α'}(π_canon^{-1}(v^{(1)})_b)) = (Σ_d coords[d][b]·α'^d) mod q'`,
/// from the coordinate-major integer lift `coords[d][b] ∈ [0, q)` of
/// `v^{(1)} ∈ K^{2^t}` (coordinate `d` in the module basis `1, X, …, X^{e−1}`).
/// This is the extension-field replacement for the plain canonical lift the
/// prime-field path feeds to the chunker.
pub fn projected_row_weights(coords: &[Vec<u128>], q_proj: u128, alpha_proj: u128) -> Vec<u128> {
    let _g = tracing::info_span!("ext:project").entered();
    let ext_deg = coords.len();
    assert!(ext_deg >= 1, "at least one coordinate vector");
    let rows = coords[0].len();
    for c in coords {
        assert_eq!(c.len(), rows, "coordinate vectors must share the row count");
    }
    let zq = field::FpCtx::from_prime_u128(q_proj);
    // α'^d as prepared Montgomery factors (d ≥ 1; the d = 0 term is the
    // plain coordinate itself) — each row term is then ONE Montgomery
    // multiplication via the plain×monty trick, no domain conversions.
    let pow_monty: Vec<field::Fp<2>> = zq
        .powers_u128(alpha_proj, ext_deg)
        .into_iter()
        .skip(1)
        .map(|p| zq.prepare_multiplier_u128(p))
        .collect();
    cfg_into_iter!(0..rows)
        .map(|b| {
            let mut acc = zq.reduce_u128(coords[0][b]);
            for (d, pw) in pow_monty.iter().enumerate() {
                acc = zq.add_canonical_u128(
                    acc,
                    zq.mul_prepared_u128(coords[d.wrapping_add(1)][b], pw),
                );
            }
            acc
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    // Independent test oracle for the 256-bit projection reduction.
    fn mulmod_generic(a: u128, b: u128, m: u128) -> u128 {
        use num_traits::ToPrimitive;
        ((num_bigint::BigUint::from(a) * num_bigint::BigUint::from(b))
            % num_bigint::BigUint::from(m))
        .to_u128()
        .unwrap()
    }
    use crate::transcript::Blake3Transcript;

    /// Reference naive primality by trial division (test-only, small inputs).
    fn is_prime_naive(n: u128) -> bool {
        if n < 2 {
            return false;
        }
        let mut d = 2u128;
        while d * d <= n {
            if n % d == 0 {
                return false;
            }
            d += 1;
        }
        true
    }

    #[test]
    fn proj_arith_matches_naive() {
        let q = (1u128 << 100) - 15; // prime
        let zq = field::FpCtx::from_prime_u128(q);
        let a = 0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABCu128;
        let b = 0x0123_4567_89AB_CDEF_0011_2233_4455u128;
        assert_eq!(zq.mul_u128(a, b), mulmod_generic(a, b, q));
        assert_eq!(zq.add_u128(a, b), (a % q + b % q) % q);
        let pows = zq.powers_u128(a, 4);
        assert_eq!(pows[0], 1);
        assert_eq!(pows[1], a % q);
        assert_eq!(pows[2], mulmod_generic(a, a, q));
        assert_eq!(pows[3], mulmod_generic(pows[2], a, q));
    }

    /// Prime sampling is deterministic in the transcript state, lands in the
    /// declared range, and (at a test-sized 40 bits) is actually prime.
    #[test]
    fn sampled_prime_is_prime_and_deterministic() {
        let proj = ExtProjParams {
            prime_bits: 40,
            mr_rounds: 8,
        };
        let mut t1 = Blake3Transcript::new();
        t1.absorb_slice(b"ext-proj-test");
        let q1 = sample_proj_prime(&mut t1, &proj).unwrap();
        let mut t2 = Blake3Transcript::new();
        t2.absorb_slice(b"ext-proj-test");
        let q2 = sample_proj_prime(&mut t2, &proj).unwrap();
        assert_eq!(q1, q2, "sampling must be deterministic in the transcript");
        assert!(q1 >= (1u128 << 39) && q1 < (1u128 << 40), "prime in range");
        assert!(is_prime_naive(q1), "sampled candidate must be prime");
        // The point lands in [0, q').
        let a1 = sample_proj_point(&mut t1, q1);
        let a2 = sample_proj_point(&mut t2, q2);
        assert_eq!(a1, a2);
        assert!(a1 < q1);
    }

    #[test]
    fn interval_prime_is_deterministic_and_in_range() {
        let min = (1u128 << 31) + 123;
        let max = (1u128 << 32) - 57;
        let mut t1 = Blake3Transcript::new();
        t1.absorb_slice(b"bounded-prime-test");
        let q1 = sample_prime_in_interval(&mut t1, min, max).expect("interval contains primes");
        let mut t2 = Blake3Transcript::new();
        t2.absorb_slice(b"bounded-prime-test");
        let q2 = sample_prime_in_interval(&mut t2, min, max).expect("interval contains primes");

        assert_eq!(q1, q2, "sampling must be deterministic in the transcript");
        assert!((min..=max).contains(&q1));
        assert_eq!(q1 & 1, 1);
        assert!(is_prime_naive(q1));
    }

    #[test]
    fn interval_sampler_handles_single_small_prime() {
        // 251 is in SMALL_PRIMES and must not be mistaken for a proper
        // multiple of itself by the trial-division filter.
        let mut transcript = Blake3Transcript::new();
        transcript.absorb_slice(b"singleton-small-prime");
        assert_eq!(sample_prime_in_interval(&mut transcript, 251, 251), Ok(251));

        let mut transcript = Blake3Transcript::new();
        transcript.absorb_slice(b"small-prime-three");
        assert_eq!(sample_prime_in_interval(&mut transcript, 2, 3), Ok(3));
    }

    #[test]
    fn interval_sampler_rejects_composites() {
        // A strong base-2 pseudoprime whose factors are larger than the small
        // sieve table exercises the shared randomized rounds.
        let pseudoprime = 341_550_071_728_321u128;
        let mut sample_transcript = Blake3Transcript::new();
        sample_transcript.absorb_slice(b"singleton-composite");
        assert_eq!(
            sample_prime_in_interval(&mut sample_transcript, pseudoprime, pseudoprime),
            Err(PrimeSamplingError::PrimeSearchExhausted {
                min: pseudoprime,
                max: pseudoprime,
                attempts: 1,
            })
        );
    }

    #[test]
    fn interval_sampler_reports_invalid_ranges() {
        let mut transcript = Blake3Transcript::new();
        assert_eq!(
            sample_prime_in_interval(&mut transcript, 12, 10),
            Err(PrimeSamplingError::InvalidInterval { min: 12, max: 10 })
        );
        assert_eq!(
            sample_prime_in_interval(&mut transcript, 2, 2),
            Err(PrimeSamplingError::NoOddCandidate { min: 2, max: 2 })
        );
        assert_eq!(
            sample_prime_in_interval(&mut transcript, 3, 1u128 << 126),
            Err(PrimeSamplingError::MaximumTooLarge { max: 1u128 << 126 })
        );
    }

    /// γ agrees with a naive per-row Horner evaluation.
    #[test]
    fn projected_weights_match_horner() {
        let q = 0x0000_00E8_D4A5_1027u128; // 10^12 + 39, prime
        assert!(is_prime_naive(q));
        let alpha = 0x1234_5678u128 % q;
        let coords: Vec<Vec<u128>> = (0..3)
            .map(|d: u128| {
                (0..16)
                    .map(|b: u128| (b + 1) * (d + 2) * 0x9E37_79B9 % (1u128 << 40))
                    .collect()
            })
            .collect();
        let gamma = projected_row_weights(&coords, q, alpha);
        for b in 0..16 {
            let mut acc = 0u128;
            for d in (0..3).rev() {
                acc = (mulmod_generic(acc, alpha, q) + coords[d][b] % q) % q;
            }
            assert_eq!(gamma[b], acc, "row {b}");
        }
    }
}
