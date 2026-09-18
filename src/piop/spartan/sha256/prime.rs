//! Transcript-derived prime-field parameters for the SHA-256 runtime-prime protocol.
//!
//! The commitment remains over `GF(2^128)`.  Only after that commitment is
//! bound into the Fiat--Shamir transcript do prover and verifier derive the
//! prime `q` used by Spartan and by the integer-to-field projection.

use crate::piop::spartan::SpartanField as _;
use field::{Fp, Uint};
use thiserror::Error;

use crate::{
    ext_proj::PrimeSamplingError,
    piop::spartan::{
        SpartanField, absorb_spartan_message,
        profile::{IopInstanceFacts, IopSecurityParams},
    },
    transcript::traits::Transcript,
};

use super::super::SpartanBitzField;
#[cfg(any(feature = "bench-internals", test))]
use super::super::profile::{SoundnessAccounting, SoundnessTerm};

const PRIME_SAMPLING_DOMAIN: &[u8] = b"bitz/spartan-sha256/runtime-prime/v1";
const FIXED_PRIME_DOMAIN: &[u8] = b"bitz/spartan-sha256/fixed-prime/v1";

/// Fixed 98-bit modulus used only by the controlled product-geometry sweep.
/// It is the largest prime below `2^98`.
pub const SHA256_FIXED_98_PRIME: u128 = (1_u128 << 98) - 51;
#[cfg(any(feature = "bench-internals", test))]
pub const SHA256_FIXED_98_PRIME_BITS: usize = 98;
#[cfg(any(feature = "bench-internals", test))]
pub const SHA256_FIXED_98_INITIAL_GRINDING_BITS: u32 = 8;
#[cfg(any(feature = "bench-internals", test))]
pub const SHA256_FIXED_98_TERMINAL_GRINDING_BITS: u32 = 4;

/// Smallest supported batch: `2^7` independent compressions.
///
/// This is the first size whose F2 source commitment reaches the validated
/// UDR Ligerito window `m >= 20` (see `validated_udr_lig_configs_for_target`):
/// `m = ceil-log2(SHA256_F_INSTANCE_BITS * 2^k + 1)` is 17/18/19 at k = 4/5/6,
/// so those batches have no validated configuration and are rejected here
/// rather than failing later at config selection.
pub const SHA256_MIN_LOG_COMPRESSIONS: usize = 7;
/// Largest supported batch: `2^16` independent compressions.
pub const SHA256_MAX_LOG_COMPRESSIONS: usize = 16;
/// Width of the fixed commitment/exponent field.
pub const SHA256_COMMITMENT_FIELD_BITS: usize = 128;

/// Arity of the flat SHA constraint-row challenge.  One compression has 184
/// live rows, so `2^(8 + log_instance_capacity)` is the smallest power-of-two
/// domain containing every flat row `184 * instance + local_row`.
pub const SHA256_TAU_LOCAL_VARS: u32 = 8;

/// The public statement facts the security-profile derivation consumes for
/// a SHA-256 batch with outer capacity `2^log_instance_capacity`: per-row integer defects are far
/// below any sampled prime (Bit assignment, coefficients `< 2^33`, a
/// few hundred entries per row — `< 2^96` conservatively), the Step-5.1
/// lift sums `2^t` terms, and the opening is the VIRTUAL path (fold width
/// capped from `q_bits`, so the one-chunk fold bound does not gate q).
pub(super) const fn sha256_instance_facts(log_instance_capacity: u32) -> IopInstanceFacts {
    IopInstanceFacts {
        defect_log2_bound: 96,
        lift_arity_log2: log_instance_capacity,
        opening_t: log_instance_capacity,
        opening_word_bits: 1,
        direct_opening: false,
        tau_arity: SHA256_TAU_LOCAL_VARS + log_instance_capacity,
        piop_degree: 2,
        step50_magnitude_log2: 0,
    }
}

/// Public interval and grinding parameters determined by the batch size and
/// the selected security profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Sha256PrimeProfile {
    log_instance_capacity: usize,
    pub(super) min_prime: u128,
    max_prime: u128,
    fixed_prime: bool,
    initial_grinding: usize,
    inner_grinding: usize,
    terminal_grinding: usize,
}

impl Sha256PrimeProfile {
    /// Adopts an instantiated security profile (single-prime policy).
    pub fn from_security(
        params: &IopSecurityParams,
        log_instance_capacity: usize,
    ) -> Result<Self, Sha256PrimeError> {
        #[cfg(not(test))]
        validate_instance_capacity_exponent(log_instance_capacity)?;
        #[cfg(test)]
        if log_instance_capacity > SHA256_MAX_LOG_COMPRESSIONS {
            return Err(Sha256PrimeError::UnsupportedInstanceCapacityExponent {
                actual: log_instance_capacity,
            });
        }
        if params.projection_full_width || params.reduction.is_some() {
            return Err(Sha256PrimeError::ProfileStrategyMismatch);
        }
        Ok(Self::adopt(params, log_instance_capacity))
    }

    fn adopt(params: &IopSecurityParams, log_instance_capacity: usize) -> Self {
        Self {
            log_instance_capacity,
            min_prime: params.projection_min,
            max_prime: params.projection_max,
            fixed_prime: false,
            initial_grinding: params.initial_grinding_bits as usize,
            inner_grinding: params.piop_round_grinding_bits as usize,
            terminal_grinding: params.terminal_grinding_bits as usize,
        }
    }

    /// Adopts the fixed 98-bit benchmark profile.
    pub(super) fn fixed_98(
        params: &IopSecurityParams,
        log_instance_capacity: usize,
    ) -> Result<Self, Sha256PrimeError> {
        validate_instance_capacity_exponent(log_instance_capacity)?;
        if params.profile_name != "sha-fixed98-lambda100"
            || params.projection_min != SHA256_FIXED_98_PRIME
            || params.projection_max != SHA256_FIXED_98_PRIME
            || params.projection_full_width
            || params.reduction.is_some()
        {
            return Err(Sha256PrimeError::ProfileStrategyMismatch);
        }
        Ok(Self {
            log_instance_capacity,
            min_prime: SHA256_FIXED_98_PRIME,
            max_prime: SHA256_FIXED_98_PRIME,
            fixed_prime: true,
            initial_grinding: params.initial_grinding_bits as usize,
            inner_grinding: params.piop_round_grinding_bits as usize,
            terminal_grinding: params.terminal_grinding_bits as usize,
        })
    }

    /// Initial proof-of-work bits before sampling `q` and the Spartan point.
    pub const fn initial_grinding_bits(self) -> usize {
        self.initial_grinding
    }

    /// Proof-of-work bits before each quadratic inner-sumcheck challenge.
    pub const fn inner_round_grinding_bits(self) -> usize {
        self.inner_grinding
    }

    /// Proof-of-work bits before the terminal opening challenges.
    pub const fn terminal_grinding_bits(self) -> usize {
        self.terminal_grinding
    }

    /// Checks the exact 128-bit no-wrap inequality without overflowing.
    pub const fn accepts_prime(self, q: u128) -> bool {
        q >= self.min_prime
            && q <= self.max_prime
            && q > 2
            && q & 1 == 1
            && q - 1 <= u128::MAX / ((1_u128 << self.log_instance_capacity) + 1)
    }

    pub const fn is_fixed(self) -> bool {
        self.fixed_prime
    }
}

/// Fixed-prime λ=100 accounting for the `2^14` SHA geometry sweep.
#[cfg(any(feature = "bench-internals", test))]
pub(super) fn fixed_98_security_params() -> IopSecurityParams {
    let log_q = (SHA256_FIXED_98_PRIME as f64).log2();
    let tau_bits = log_q - 22_f64.log2() + f64::from(SHA256_FIXED_98_INITIAL_GRINDING_BITS);
    let degree_two_bits = log_q - 2_f64.log2() + f64::from(SHA256_FIXED_98_TERMINAL_GRINDING_BITS);
    let terms = vec![
        SoundnessTerm {
            name: "step3:tau-draw",
            bits: tau_bits,
            grinding_bits: SHA256_FIXED_98_INITIAL_GRINDING_BITS,
            floor: false,
        },
        SoundnessTerm {
            name: "step3:piop-round",
            bits: degree_two_bits,
            grinding_bits: SHA256_FIXED_98_TERMINAL_GRINDING_BITS,
            floor: false,
        },
        SoundnessTerm {
            name: "step4:terminal-draw",
            bits: degree_two_bits,
            grinding_bits: SHA256_FIXED_98_TERMINAL_GRINDING_BITS,
            floor: false,
        },
        SoundnessTerm {
            name: "step5_2:gkr-round",
            bits: 128.0 - 3_f64.log2(),
            grinding_bits: 0,
            floor: false,
        },
        SoundnessTerm {
            name: "step5_3:ring-switch",
            bits: 128.0,
            grinding_bits: 0,
            floor: false,
        },
        SoundnessTerm {
            name: "step5_3:ligerito-tracked",
            bits: 100.0,
            grinding_bits: 0,
            floor: false,
        },
        SoundnessTerm {
            name: "step5_3:gf128-floor-untracked",
            bits: 128.0 - 3_f64.log2(),
            grinding_bits: 0,
            floor: true,
        },
    ];
    IopSecurityParams {
        profile_name: "sha-fixed98-lambda100",
        lambda: 100,
        projection_min: SHA256_FIXED_98_PRIME,
        projection_max: SHA256_FIXED_98_PRIME,
        projection_full_width: false,
        initial_grinding_bits: SHA256_FIXED_98_INITIAL_GRINDING_BITS,
        piop_round_grinding_bits: SHA256_FIXED_98_TERMINAL_GRINDING_BITS,
        terminal_grinding_bits: SHA256_FIXED_98_TERMINAL_GRINDING_BITS,
        reduction: None,
        forest_round_grinding_bits: 0,
        ring_switch_grinding_bits: 0,
        ligerito_target_bits: 100,
        ood: None,
        accounting: SoundnessAccounting { target: 100, terms },
    }
}

/// Samples `q` from the transcript and immediately binds its canonical
/// 16-byte encoding before any field-valued challenge is drawn.
pub(super) fn sample_sha256_mod_q_context(
    transcript: &mut impl Transcript,
    profile: Sha256PrimeProfile,
) -> Result<field::FpCtx<2>, Sha256PrimeError> {
    if profile.is_fixed() {
        absorb_spartan_message(transcript, b"prime-domain", FIXED_PRIME_DOMAIN);
        absorb_spartan_message(
            transcript,
            b"log-instance-capacity",
            &(profile.log_instance_capacity as u64).to_le_bytes(),
        );
        absorb_spartan_message(transcript, b"prime-q", &SHA256_FIXED_98_PRIME.to_le_bytes());
        validate_fixed_98_prime(SHA256_FIXED_98_PRIME)?;
        return Ok(field::create_prime_field(Uint::from(SHA256_FIXED_98_PRIME)));
    }
    absorb_spartan_message(transcript, b"prime-domain", PRIME_SAMPLING_DOMAIN);
    absorb_spartan_message(
        transcript,
        b"log-instance-capacity",
        &(profile.log_instance_capacity as u64).to_le_bytes(),
    );
    absorb_spartan_message(transcript, b"prime-min", &profile.min_prime.to_le_bytes());
    absorb_spartan_message(transcript, b"prime-max", &profile.max_prime.to_le_bytes());
    let field = crate::ext_proj::sample_prime_context(
        transcript,
        profile.min_prime,
        profile.max_prime,
        128,
    )?;
    let q = field.modulus_u128();
    absorb_spartan_message(transcript, b"prime-q", &q.to_le_bytes());
    Ok(field)
}

pub(super) fn validate_sha256_field_config(
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<(), Sha256PrimeError> {
    let encoding = SpartanBitzField::canonical_modulus_encoding(field_config);
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&encoding);
    let q = u128::from_le_bytes(bytes);
    if q == SHA256_FIXED_98_PRIME {
        validate_fixed_98_prime(q)
    } else {
        Fp::<2>::validate_config(field_config)
            .map_err(|_| Sha256PrimeError::InvalidFieldConfiguration)
    }
}

fn validate_fixed_98_prime(q: u128) -> Result<(), Sha256PrimeError> {
    if q != SHA256_FIXED_98_PRIME || !field::is_probable_prime_public(&field::Uint::from(q)) {
        return Err(Sha256PrimeError::InvalidFieldConfiguration);
    }
    Ok(())
}

fn validate_instance_capacity_exponent(
    log_instance_capacity: usize,
) -> Result<u32, Sha256PrimeError> {
    if !(SHA256_MIN_LOG_COMPRESSIONS..=SHA256_MAX_LOG_COMPRESSIONS).contains(&log_instance_capacity)
    {
        return Err(Sha256PrimeError::UnsupportedInstanceCapacityExponent {
            actual: log_instance_capacity,
        });
    }
    u32::try_from(log_instance_capacity).map_err(|_| {
        Sha256PrimeError::UnsupportedInstanceCapacityExponent {
            actual: log_instance_capacity,
        }
    })
}

/// Errors in deriving the SHA-256 runtime-prime profile.
#[derive(Debug, Error)]
pub enum Sha256PrimeError {
    /// Production supports exactly the configured SHA-256 batch window.
    #[error(
        "SHA-256 runtime-prime profile requires log-instance-capacity in [4, 16], got {actual}"
    )]
    UnsupportedInstanceCapacityExponent { actual: usize },
    /// The security profile is not a single-prime configuration.
    #[error("the SHA-256 path requires a single-prime security profile")]
    ProfileStrategyMismatch,
    /// A caller attempted to construct a context with an out-of-profile value.
    #[error("prime {q} is outside the SHA-256 runtime-prime profile")]
    PrimeOutsideProfile { q: u128 },
    /// The runtime field backend rejected the sampled modulus.
    #[error("failed to construct the runtime 128-bit prime field")]
    InvalidFieldConfiguration,
    /// Transcript prime sampling failed.
    #[error(transparent)]
    Sampling(#[from] PrimeSamplingError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piop::spartan::profile::{IopSecurityProfile, Sha128ReferenceSchedule};
    use crate::transcript::Blake3Transcript;

    fn reference_profile(log_compressions: usize) -> Result<Sha256PrimeProfile, Sha256PrimeError> {
        let exponent = validate_instance_capacity_exponent(log_compressions)?;
        let security = Sha128ReferenceSchedule::instantiate(&sha256_instance_facts(exponent))
            .expect("the reference schedule supports every production SHA-256 shape");
        Sha256PrimeProfile::from_security(&security, log_compressions)
    }

    #[test]
    fn runtime_prime_intervals_have_the_expected_width_and_no_wrap_bound() {
        for t in SHA256_MIN_LOG_COMPRESSIONS..=SHA256_MAX_LOG_COMPRESSIONS {
            let profile = reference_profile(t).unwrap();
            let min_bits = (u128::BITS - profile.min_prime.leading_zeros()) as usize;
            assert_eq!(min_bits, if t == 16 { 112 } else { 113 });
            assert!(profile.max_prime < (1_u128 << 113));
            assert!(profile.min_prime <= profile.max_prime);
            assert!(profile.max_prime - 1 <= u128::MAX / ((1_u128 << t) + 1));
        }

        let t14 = reference_profile(14).unwrap();
        assert_eq!(t14.max_prime, (1_u128 << 113) - 1);
        let t15 = reference_profile(15).unwrap();
        assert!(t15.max_prime < (1_u128 << 113) - 1);
        let t16 = reference_profile(16).unwrap();
        assert!(t16.max_prime < (1_u128 << 112) - 1);
    }

    #[test]
    fn transcript_sampling_is_deterministic_and_builds_one_field_context() {
        let profile = reference_profile(7).unwrap();
        let mut first = Blake3Transcript::new();
        let mut second = Blake3Transcript::new();
        let first = sample_sha256_mod_q_context(&mut first, profile).unwrap();
        let second = sample_sha256_mod_q_context(&mut second, profile).unwrap();
        assert_eq!(first.modulus_u128(), second.modulus_u128());
        assert_eq!(first.modulus_bits(), 113);
        assert!(profile.accepts_prime(first.modulus_u128()));
        assert_eq!(
            Fp::<2>::canonical_modulus_encoding(&first),
            first.modulus_u128().to_le_bytes()
        );
    }

    #[test]
    fn fixed_98_profile_binds_the_exact_prime_and_field() {
        let security = fixed_98_security_params();
        let profile = Sha256PrimeProfile::fixed_98(&security, 14).unwrap();
        let mut transcript = Blake3Transcript::new();
        let context = sample_sha256_mod_q_context(&mut transcript, profile).unwrap();

        assert_eq!(context.modulus_u128(), SHA256_FIXED_98_PRIME);
        assert_eq!(context.modulus_bits(), SHA256_FIXED_98_PRIME_BITS);
        assert_eq!(security.accounting.achieved_bits(), 100.0);
        assert_eq!(
            Fp::<2>::canonical_modulus_encoding(&context),
            SHA256_FIXED_98_PRIME.to_le_bytes()
        );
    }

    #[test]
    fn rejects_instance_capacities_outside_the_runtime_prime_profile() {
        assert!(matches!(
            reference_profile(3),
            Err(Sha256PrimeError::UnsupportedInstanceCapacityExponent { actual: 3 })
        ));
        assert!(matches!(
            reference_profile(17),
            Err(Sha256PrimeError::UnsupportedInstanceCapacityExponent { actual: 17 })
        ));
    }
}
