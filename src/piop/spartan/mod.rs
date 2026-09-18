//! Spartan's field-generic R1CS PIOP.
//!
//! This module deliberately treats a field as the pair `(F, F::Config)`.
//! That distinction is load-bearing for runtime-configured field types such as
//! [`Fp`]: one Rust type can represent elements modulo many different
//! primes.  The prepared statement binds the configuration, while the helpers
//! in this module give every protocol phase one canonical element encoding and
//! one unbiased Fiat--Shamir challenge sampler.

use crate::piop::spartan::SpartanField as _;
use field::{Fp, Uint};
pub mod baby_bear_bitz;
pub mod baby_bear_mul;
pub mod cm;
#[cfg(feature = "ecdsa")]
pub mod ecdsa_sha256;
pub mod bitz;
pub mod grinding;
pub mod matrix;
pub mod mul;
pub mod multiswap;
pub mod opening_mode;
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[path = "../../../benches/outer_regression/driver.rs"]
pub mod outer_regression;
pub mod piop;
pub mod profile;
pub mod protocol;
pub(crate) mod raw_monty;
pub mod sha256;
pub(crate) mod slot_rows;
pub(crate) mod spliced_digest;
pub mod sumcheck;
pub mod u128_bitz;
pub mod u128_mul;
pub mod u32_mul;
pub use mul::{MulError, MulLayout, MulRow, MulWitness, MulWord};
pub mod u64_bitz;
pub mod u64_mul;
#[cfg(test)]
#[cfg(test)]
pub(crate) use crate::sumcheck::outer::native_skip as univariate_skip_native;
pub use crate::sumcheck::outer::univariate as univariate_skip;

pub use baby_bear_bitz::{
    baby_bear_mul_instance_facts, commit_baby_bear_mul_witness,
    commit_baby_bear_mul_witness_with_ligerito,
};
pub use baby_bear_mul::{
    BABY_BEAR_MODULUS, BabyBearMulCoefficient, BabyBearMulError, BabyBearMulLayout,
    BabyBearMulWitness, baby_bear_mul_constraint_matrices, prepare_baby_bear_mul_relation,
    project_baby_bear_mul_native_witness, project_baby_bear_mul_witness,
    sample_baby_bear_operand_with,
};
pub use cm::{
    CM_AND_F_LIVE_SLOTS, CM_AND_H_SLOTS, CM_AND_WORD_BITS, CmAndError, CmAndLayout, CmAndSpec,
    CmAndWitness, PreparedCmAndRelation, cm_and_map, commit_cm_and_witness,
    commit_cm_and_witness_with_config, prepare_cm_and_relation, project_cm_and_witness,
    prove_cm_and_bitz, prove_cm_and_bitz_with_config, verify_cm_and_bitz,
    verify_cm_and_bitz_with_config,
};
pub use bitz::{
    SpartanBitzField, U32_MUL_UNIVARIATE_SKIP_DEGREE, U32_MUL_UNIVARIATE_SKIP_VARS,
    spartan_bitz_field_config,
};

pub use circuit::linear_map::SparseMatrixError;
pub use matrix::{
    ConstraintMatrices, ConstraintMatricesSkeleton, MleClaimError, ModulusIndependentCoefficient,
    PreparedConstraintMatrices, ScaledMleEvaluationClaim, SpartanMatrixCoefficient,
    SpartanMatrixError, build_assignment_mle, build_boolean_assignment_mle, build_product_mles,
    eq_eval, eq_table, make_equality_factors,
};
pub use opening_mode::EvaluatedSpartanAssignment;
pub use piop::{
    SPARTAN_ASSIGNMENT_ORACLE_DOMAIN, SPARTAN_PIOP_DOMAIN, SPARTAN_UNIVARIATE_SKIP_PIOP_DOMAIN,
    SpartanError, SpartanPiopProof, prove_spartan_nonsuccinct, prove_spartan_piop,
    prove_spartan_piop_field, prove_spartan_piop_u32_native,
    prove_spartan_piop_u32_native_with_univariate_skip, prove_spartan_piop_with_univariate_skip,
    verify_spartan_proof, verify_spartan_univariate_skip_proof, verify_spartan_with_mle_claim,
};
pub use profile::{
    IopInstanceFacts, IopSecurityParams, IopSecurityProfile, Lambda100, Lambda128, Limber112,
    Limber114, PrimePolicy, ProfileError, ReductionPrimeParams, Sha128ReferenceSchedule,
    SoundnessAccounting, SoundnessTerm,
};
pub use sha256::{
    PreparedSha256ChainBatch, SHA256_CHAIN_F_BAR_LIVE_BITS, SHA256_CHAIN_F_INSTANCE_BITS,
    SHA256_CHAIN_H_BAR_LIVE_BITS, SHA256_CHAIN_H_INSTANCE_BITS, SHA256_CHAIN_TERMINAL_BITS,
    Sha256ChainStatement, Sha256ChainWitnessBatch, commit_sha256_chain_witness,
    commit_sha256_chain_witness_with_config, generate_sha256_chain_witnesses,
    prepare_sha256_chain_batch, prepare_sha256_chain_batch_with_profile,
    prepare_sha256_chain_batch_with_profile_and_initial_state, prove_sha256_chain,
    prove_sha256_chain_with_config, sha256_chain_configs, sha256_compress, verify_sha256_chain,
    verify_sha256_chain_with_config,
};
pub use sha256::{
    PreparedSha256CompressionBatch, SHA256_COMMITMENT_FIELD_BITS, SHA256_CONSTRAINTS,
    SHA256_DEFAULT_INNER_PREFIX_VARS, SHA256_F_BAR_LIVE_BITS, SHA256_F_INSTANCE_BITS,
    SHA256_F_LIVE_BITS, SHA256_H_BAR_LIVE_BITS, SHA256_H_INSTANCE_BITS,
    SHA256_INNER_PREFIX_MAX_VARS, SHA256_MAX_LOG_COMPRESSIONS, SHA256_MIN_LOG_COMPRESSIONS,
    Sha256CompressionInput, Sha256CompressionStatement, Sha256CompressionWitnessBatch,
    Sha256ConstraintError, Sha256OpeningLayout, Sha256PrimeError, Sha256WitnessError,
    commit_sha256_compression_witness, commit_sha256_compression_witness_with_config,
    generate_sha256_compression_witnesses, prepare_sha256_compression_batch,
    prepare_sha256_compression_batch_for_assignment_rows,
    prepare_sha256_compression_batch_for_assignment_rows_with_profile,
    prepare_sha256_compression_batch_with_profile,
    prepare_sha256_compression_batch_with_profile_and_layout, prove_sha256_compressions,
    prove_sha256_compressions_with_config, prove_sha256_compressions_with_prefix_vars,
    prove_sha256_compressions_with_prefix_vars_and_config, sha256_compression_configs,
    verify_sha256_compressions, verify_sha256_compressions_with_config,
};
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub use sha256::{
    SHA256_FIXED_98_INITIAL_GRINDING_BITS, SHA256_FIXED_98_PRIME, SHA256_FIXED_98_PRIME_BITS,
    SHA256_FIXED_98_TERMINAL_GRINDING_BITS, prepare_sha256_compression_batch_for_product_t_fixed98,
};
pub use sumcheck::{OuterSumcheckProof, R1csProductMles, SumcheckError, SumcheckProof};
pub use u32_mul::{
    U32_MUL_BIT_SLOTS, U32_MUL_PRODUCT_BITS, U32_MUL_X_BITS, U32_MUL_Y_BITS,
    prepare_u32_mul_relation, project_u32_mul_native_witness, u32_mul_constraint_matrices,
};
pub use u64_bitz::u64_mul_instance_facts;
pub use u64_mul::{
    U64_MUL_BIT_SLOTS, U64_MUL_LIMB_BASE, U64_MUL_VALUE_BITS, U64MulCoefficient,
    prepare_u64_mul_relation, project_u64_mul_witness, u64_mul_constraint_matrices,
};
pub use u128_bitz::u128_mul_instance_facts;
pub use u128_mul::{
    U128_MUL_BIT_SLOTS, U128_MUL_OPERAND_BITS, U128_MUL_PRODUCT_BITS, mul_u128_full,
    prepare_u128_mul_relation, project_u128_mul_witness, u128_mul_constraint_matrices,
};
pub use univariate_skip::{
    UnivariateSkipOuterSumcheckProof, UnivariateSkipProof, UnivariateSkipSpartanPiopProof,
};

use std::slice;

use num_traits::ConstOne;
use thiserror::Error;

use crate::transcript::traits::Transcript;

const SPARTAN_TRANSCRIPT_FRAME_DOMAIN: &[u8] = b"bitz/spartan/transcript-frame/v2";
const FIELD_ELEMENTS_TAG: &[u8] = b"field-elements";

/// Minimum modulus size accepted by the Spartan PIOP.
///
/// This is the protocol's minimum accepted field-size security boundary;
/// adapters may select any prime modulus meeting it.
pub const SPARTAN_MIN_MODULUS_BITS: u32 = 100;

/// Why a runtime field configuration is unsafe for Spartan.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum SpartanFieldError {
    /// The modulus is prime but below Spartan's minimum soundness boundary.
    #[error("Spartan requires at least a 100-bit modulus, got {actual_bits} bits")]
    ModulusTooSmall { actual_bits: u32 },

    /// Prime-field arithmetic is unsound when the runtime modulus is composite.
    #[error("the configured Spartan modulus is composite")]
    CompositeModulus,

    /// An unchecked constructor produced an invalid internal field residue.
    #[error("a Spartan field element has a noncanonical internal residue")]
    NonCanonicalElement,
}

/// Field operations required by the native Spartan protocol.
///
/// In addition to [`PrimeField`], an implementation fixes four protocol
/// details that `PrimeField` intentionally leaves unspecified:
///
/// - `validate_config` establishes that a runtime configuration is a genuine,
///   sufficiently large field before it reaches any protocol arithmetic.
/// - `canonical_element_encoding` is an injective, platform-independent
///   encoding of a reduced field element.  In particular, an implementation
///   must not expose an implementation-specific Montgomery residue.
/// - `canonical_modulus_encoding` identifies the configured field.  Protocol
///   statement digests must bind this encoding because `Self` alone need not
///   determine the modulus.
/// - `sample_uniform` maps transcript output to an *exactly uniform* field
///   element.  Reducing one fixed-width random integer modulo the field order
///   is generally biased and is not a valid implementation.
///
/// Rejection sampling is permitted.  Every rejected draw must still advance
/// the transcript, so prover and verifier consume the identical stream.
pub trait SpartanField:
    Copy + core::fmt::Debug + Eq + Send + Sync + field::CtEq + field::CtSelect
{
    type Inner: Clone + core::fmt::Debug + Eq + Send + Sync;
    type Config: field::BatchMulAcc<Self>
        + field::Reduce<<Self::Config as field::BatchMulAcc<Self>>::Accumulator, Output = Self>
        + crate::sumcheck::outer::OuterArithmetic<Self, Self>
        + field::BatchFieldOps<Elem = Self>
        + field::CanonicalCodec<Self>
        + field::IntegerEmbedding<u64>
        + field::IntegerEmbedding<u128>
        + field::IntegerEmbedding<i64>
        + Clone
        + core::fmt::Debug
        + Send
        + Sync;
    fn validate_config(field: &Self::Config) -> Result<(), SpartanFieldError>;
    fn validate_element(&self, modulus_encoding: &[u8]) -> Result<(), SpartanFieldError>;
    fn canonical_element_encoding(&self, field: &Self::Config) -> Vec<u8> {
        let mut out = vec![0; Self::canonical_encoding_width()];
        field::CanonicalCodec::encode_into(field, self, &mut out);
        out
    }
    fn canonical_modulus_encoding(field: &Self::Config) -> Vec<u8>;
    fn canonical_encoding_width() -> usize;
    fn sample_uniform<T: Transcript>(
        transcript: &mut T,
        field: &Self::Config,
    ) -> Result<Self, field::SamplingError>;
    fn zero_with_cfg(field: &Self::Config) -> Self {
        field::RingOps::zero(field)
    }
    fn one_with_cfg(field: &Self::Config) -> Self {
        field::RingOps::one(field)
    }
    fn is_zero(value: &Self) -> bool {
        field::CtEq::ct_is_zero(value).declassify()
    }
    fn new_with_cfg(value: Self::Inner, field: &Self::Config) -> Self;
    fn new_unchecked_with_cfg(value: Self::Inner, field: &Self::Config) -> Self;
    fn make_cfg(modulus: &Self::Inner) -> Result<Self::Config, SpartanFieldError>;
    fn from_with_cfg<T>(value: T, field: &Self::Config) -> Self
    where
        Self::Config: field::IntegerEmbedding<T>,
    {
        field::IntegerEmbedding::from_integer(field, &value)
    }
}

impl<const L: usize> SpartanField for field::Fp<L> {
    type Inner = field::Uint<L>;
    type Config = field::FpCtx<L>;
    fn validate_config(field: &Self::Config) -> Result<(), SpartanFieldError> {
        let words = field.modulus().as_words();
        let bits = words
            .iter()
            .rposition(|w| *w != 0)
            .map_or(0, |i| 64 * i + 64 - words[i].leading_zeros() as usize)
            as u32;
        if bits < SPARTAN_MIN_MODULUS_BITS {
            return Err(SpartanFieldError::ModulusTooSmall { actual_bits: bits });
        }
        if !field::is_probable_prime_public(field.modulus()) {
            return Err(SpartanFieldError::CompositeModulus);
        }
        Ok(())
    }
    fn validate_element(&self, modulus_encoding: &[u8]) -> Result<(), SpartanFieldError> {
        use field::{CanonicalCodec, CtOrd};
        let modulus: field::Uint<L> = field::IntegerOps
            .decode_public(modulus_encoding)
            .map_err(|_| SpartanFieldError::NonCanonicalElement)?;
        if !self.as_montgomery_integer().ct_lt(&modulus).declassify() {
            return Err(SpartanFieldError::NonCanonicalElement);
        }
        Ok(())
    }
    fn canonical_modulus_encoding(field: &Self::Config) -> Vec<u8> {
        let mut out = vec![0; L * 8];
        field::CanonicalCodec::encode_into(&field::IntegerOps, field.modulus(), &mut out);
        out
    }
    fn canonical_encoding_width() -> usize {
        L * 8
    }
    fn sample_uniform<T: Transcript>(
        transcript: &mut T,
        field: &Self::Config,
    ) -> Result<Self, field::SamplingError> {
        use field::FieldSampling;
        field.sample_public(&mut crate::ext_proj::TranscriptRandom(transcript), 256)
    }
    fn new_with_cfg(value: Self::Inner, field: &Self::Config) -> Self {
        field::IntegerEmbedding::from_integer(field, &value)
    }
    fn new_unchecked_with_cfg(value: Self::Inner, field: &Self::Config) -> Self {
        field.from_montgomery_integer(value)
    }
    fn make_cfg(modulus: &Self::Inner) -> Result<Self::Config, SpartanFieldError> {
        if !field::is_probable_prime_public(modulus) {
            return Err(SpartanFieldError::CompositeModulus);
        }
        Ok(field::create_prime_field(*modulus))
    }
}

/// Bit matrices act by the field's one, whose canonical encoding is the
/// same fixed-width transcription of `1` under every configuration, so their
/// prepared statements can be cached modulus-independently.
impl<const LIMBS: usize> ModulusIndependentCoefficient<Fp<LIMBS>> for bool {
    fn write_modulus_independent_encoding(&self, out: &mut Vec<u8>) {
        // Mirrors `canonical_element_encoding` of the field one: the
        // canonical residue written through the same fixed-width
        // transcription. Explicit zeros are rejected before encoding, but
        // stay total and correct here regardless.
        let value = if *self {
            Uint::<LIMBS>::ONE
        } else {
            Uint::<LIMBS>::ZERO
        };
        let start = out.len();
        out.resize(start + LIMBS * 8, 0);
        field::CanonicalCodec::encode_into(&field::IntegerOps, &value, &mut out[start..]);
    }

    fn is_unit(&self) -> bool {
        *self
    }
}

/// Absorbs field elements using [`SpartanField`]'s canonical encoding.
///
/// The collection count and every element length are encoded inside one typed,
/// length-prefixed Spartan frame. This is intentionally stronger than calling
/// [`Transcript::absorb_slice`] for each element: that legacy delimiter-only
/// framing is not injective for arbitrary byte strings.
pub(crate) fn absorb_field_elements<F, T>(
    transcript: &mut T,
    values: &[F],
    field_config: &F::Config,
) where
    F: SpartanField,
    T: Transcript,
{
    let mut payload = Vec::new();
    extend_frame_len(&mut payload, values.len());
    for value in values {
        let encoding = value.canonical_element_encoding(field_config);
        extend_frame_len(&mut payload, encoding.len());
        payload.extend_from_slice(&encoding);
    }
    absorb_spartan_message(transcript, FIELD_ELEMENTS_TAG, &payload);
}

/// Absorbs one typed, self-delimiting Spartan transcript message.
pub(crate) fn absorb_spartan_message(transcript: &mut impl Transcript, tag: &[u8], payload: &[u8]) {
    let mut frame =
        Vec::with_capacity(SPARTAN_TRANSCRIPT_FRAME_DOMAIN.len() + tag.len() + payload.len() + 16);
    frame.extend_from_slice(SPARTAN_TRANSCRIPT_FRAME_DOMAIN);
    extend_frame_len(&mut frame, tag.len());
    frame.extend_from_slice(tag);
    extend_frame_len(&mut frame, payload.len());
    frame.extend_from_slice(payload);
    transcript.absorb_slice(&frame);
}

fn extend_frame_len(frame: &mut Vec<u8>, len: usize) {
    let len = u64::try_from(len).expect("an in-memory transcript message length fits u64");
    frame.extend_from_slice(&len.to_le_bytes());
}

/// Draws an unbiased field challenge and re-absorbs its canonical encoding.
///
/// Re-absorbing the accepted value is intentional: existing protocols in this
/// crate draw a challenge and then absorb that challenge before continuing.
/// Keeping the operation here prevents prover/verifier transcript schedules
/// from drifting apart.
pub(crate) fn squeeze_field<F, T>(
    transcript: &mut T,
    field_cfg: &F::Config,
) -> Result<F, sumcheck::SumcheckError>
where
    F: SpartanField,
    T: Transcript,
{
    transcript.begin_sampling();
    let challenge = F::sample_uniform(transcript, field_cfg)
        .map_err(|_| sumcheck::SumcheckError::SamplingExhausted)?;
    absorb_field_elements(transcript, slice::from_ref(&challenge), &field_cfg);
    Ok(challenge)
}

#[cfg(test)]
mod tests {
    use crate::transcript::traits::GenTranscribable;
    use std::collections::VecDeque;

    use super::*;
    use crate::transcript::traits::ConstTranscribable;

    struct ScriptedTranscript {
        draws: VecDeque<Vec<u8>>,
        draws_consumed: usize,
        absorbed: Vec<u8>,
    }

    impl ScriptedTranscript {
        fn new(draws: impl IntoIterator<Item = Uint<2>>) -> Self {
            Self {
                draws: draws
                    .into_iter()
                    .map(|draw| {
                        let mut bytes = vec![0; Uint::<2>::NUM_BYTES];
                        draw.write_transcription_bytes_exact(&mut bytes);
                        bytes
                    })
                    .collect(),
                draws_consumed: 0,
                absorbed: Vec::new(),
            }
        }
    }

    impl Transcript for ScriptedTranscript {
        fn fill_sampling_bytes(&mut self, output: &mut [u8]) {
            let bytes = self.draws.front_mut().expect("scripted sampling draw");
            assert!(output.len() <= bytes.len());
            output.copy_from_slice(&bytes[..output.len()]);
            bytes.drain(..output.len());
            if bytes.is_empty() {
                self.draws.pop_front();
                self.draws_consumed += 1;
            }
        }

        fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
            self.draws_consumed += 1;
            let bytes = self.draws.pop_front().expect("scripted challenge draw");
            assert_eq!(bytes.len(), T::NUM_BYTES);
            T::read_transcription_bytes_exact(&bytes)
        }

        fn absorb_inner(&mut self, value: &[u8]) {
            self.absorbed.extend_from_slice(value);
        }
    }

    #[test]
    fn uniform_sampler_rejects_the_incomplete_interval_and_reabsorbs_the_result() {
        // The shared sampler masks to the public modulus width. The all-ones
        // candidate equals q and is rejected; the next candidate is 42.
        let modulus = (1_u128 << 127) - 1;
        let field_cfg = Fp::<2>::make_cfg(&Uint::from(modulus)).expect("Mersenne prime modulus");
        Fp::<2>::validate_config(&field_cfg).unwrap();
        let max = Uint::<2>::MAX;
        let accepted = Uint::from(42_u128);
        let mut transcript = ScriptedTranscript::new([max, accepted]);

        let challenge = squeeze_field::<Fp<2>, _>(&mut transcript, &field_cfg).unwrap();

        assert_eq!(challenge, Fp::<2>::from_with_cfg(42_u128, &field_cfg));
        assert_eq!(transcript.draws_consumed, 2);
        let encoding = challenge.canonical_element_encoding(&field_cfg);
        let mut payload = Vec::new();
        payload.extend_from_slice(&1_u64.to_le_bytes());
        payload.extend_from_slice(&(encoding.len() as u64).to_le_bytes());
        payload.extend_from_slice(&encoding);
        let mut expected_absorption = vec![0x6];
        let frame_len = SPARTAN_TRANSCRIPT_FRAME_DOMAIN.len()
            + 8
            + FIELD_ELEMENTS_TAG.len()
            + 8
            + payload.len();
        expected_absorption.extend_from_slice(&(frame_len as u64).to_le_bytes());
        expected_absorption.extend_from_slice(SPARTAN_TRANSCRIPT_FRAME_DOMAIN);
        expected_absorption.extend_from_slice(&(FIELD_ELEMENTS_TAG.len() as u64).to_le_bytes());
        expected_absorption.extend_from_slice(FIELD_ELEMENTS_TAG);
        expected_absorption.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        expected_absorption.extend_from_slice(&payload);
        expected_absorption.push(0x7);
        assert_eq!(transcript.absorbed, expected_absorption);
    }

    #[test]
    fn field_sampling_exhaustion_is_a_bounded_protocol_error() {
        let field = field::FpCtx::from_prime_u128((1u128 << 127) - 1);
        let mut transcript = ScriptedTranscript::new([Uint::<2>::MAX; 256]);
        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut transcript, &field),
            Err(sumcheck::SumcheckError::SamplingExhausted)
        );
        assert_eq!(transcript.draws_consumed, 256);
        assert!(transcript.absorbed.is_empty());
    }

    #[test]
    fn spartan_frames_distinguish_the_legacy_delimiter_collision() {
        let mut one_message = ScriptedTranscript::new([]);
        absorb_spartan_message(&mut one_message, b"test", &[0x7, 0x6]);

        let mut two_messages = ScriptedTranscript::new([]);
        absorb_spartan_message(&mut two_messages, b"test", &[]);
        absorb_spartan_message(&mut two_messages, b"test", &[]);

        assert_ne!(one_message.absorbed, two_messages.absorbed);
    }

    #[test]
    fn runtime_monty_config_must_be_prime_and_at_least_100_bits() {
        let q100 = Fp::<2>::make_cfg(&Uint::from((1_u128 << 100) - 15)).unwrap();
        assert_eq!(Fp::<2>::validate_config(&q100), Ok(()));

        assert_eq!(
            Fp::<2>::make_cfg(&Uint::from((1_u128 << 100) - 17)),
            Err(SpartanFieldError::CompositeModulus)
        );

        let undersized = Fp::<2>::make_cfg(&Uint::from(97_u128)).unwrap();
        assert_eq!(
            Fp::<2>::validate_config(&undersized),
            Err(SpartanFieldError::ModulusTooSmall { actual_bits: 7 })
        );
    }
}

#[cfg(test)]
pub(crate) fn noncanonical_test_value(field: &field::FpCtx<2>) -> field::Fp<2> {
    // Valid in the larger owner, out of range in the context under test.
    field::FpCtx::from_prime_u128(u128::MAX - 158).from_montgomery_integer(*field.modulus())
}
