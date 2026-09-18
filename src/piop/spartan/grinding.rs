//! Typed Fiat--Shamir proof-of-work grinding for Spartan prover messages.
//!
//! A grinding boundary has a type-level protocol domain and a canonical
//! `u64` round index.  Prover and verifier first bind that boundary and the
//! configured difficulty into the current transcript, squeeze a 256-bit seed,
//! and then search/check
//!
//! ```text
//! BLAKE3(seed || nonce.to_le_bytes()).leading_zeros() >= difficulty.
//! ```
//!
//! The nonce is absorbed into the transcript as canonical little-endian bytes
//! before the next Fiat--Shamir challenge is drawn.  The prover scans nonces
//! with the eight-lane NEON BLAKE3 kernel of [`crate::utils::blake3x4`]; the
//! parallel search returns the smallest hit of the whole scanned prefix of
//! the nonce space, so enabling `parallel` does not change the proof or
//! transcript.

use core::marker::PhantomData;
use field::RingOps;

use thiserror::Error;

use crate::transcript::traits::{ConstTranscribable, GenTranscribable, Transcript};
#[cfg(feature = "parallel")]
use crate::utils::blake3x4::smallest_pow_nonce;
use crate::utils::blake3x4::{first_pow_nonce, pow_ok};

/// Transcript frame for every Spartan grinding boundary.
const GRINDING_TRANSCRIPT_DOMAIN: &[u8] = b"bitz/spartan/fiat-shamir-grinding/v1";
/// Frame separating the canonical nonce from the seed-derivation inputs.
const GRINDING_NONCE_DOMAIN: &[u8] = b"bitz/spartan/fiat-shamir-grinding/nonce/v1";

/// BLAKE3 outputs 256 bits, so no larger difficulty can be satisfied.
pub const MAX_GRINDING_BITS: u32 = 256;

/// The type-level domain of one class of prover-message grinding boundaries.
///
/// Use a distinct zero-sized marker type for protocol stages whose nonces are
/// not interchangeable.  For example, SHA commitment, Spartan outer-round,
/// and Spartan terminal messages should each implement this trait with a
/// different `DOMAIN` value.
pub trait GrindingDomain {
    /// A non-empty, versioned protocol-stage domain separator.
    const DOMAIN: &'static [u8];
}

/// One indexed grinding boundary in a [`GrindingDomain`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrindingRound<D> {
    index: u64,
    _domain: PhantomData<fn() -> D>,
}

impl<D> GrindingRound<D> {
    /// Constructs the boundary for `index`.
    pub const fn new(index: u64) -> Self {
        Self {
            index,
            _domain: PhantomData,
        }
    }

    /// Returns the canonical round index bound into the transcript.
    pub const fn index(&self) -> u64 {
        self.index
    }
}

/// A 256-bit proof-of-work seed derived from the current transcript.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GrindingSeed([u8; 32]);

impl GrindingSeed {
    /// Constructs a seed from its canonical bytes.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the canonical seed bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl GenTranscribable for GrindingSeed {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        Self(
            bytes
                .try_into()
                .expect("a grinding seed transcript challenge is exactly 32 bytes"),
        )
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), Self::NUM_BYTES);
        buf.copy_from_slice(&self.0);
    }
}

impl ConstTranscribable for GrindingSeed {
    const NUM_BYTES: usize = 32;
}

/// Failures while deriving, finding, or checking a grinding nonce.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum GrindingError {
    /// Zero silently disables grinding, while values above 256 cannot be
    /// satisfied by a BLAKE3 digest.
    #[error("grinding difficulty must be in 1..={MAX_GRINDING_BITS}, got {bits}")]
    InvalidDifficulty { bits: u32 },

    /// An empty protocol-stage domain defeats type-level domain separation.
    #[error("grinding domains must be non-empty")]
    EmptyDomain,

    /// No `u64` nonce satisfies the requested seed and difficulty.
    #[error("the u64 grinding nonce space was exhausted")]
    NonceSpaceExhausted,

    /// The supplied nonce does not meet the configured difficulty.
    #[error("grinding nonce {nonce} does not satisfy {bits} leading zero bits")]
    InvalidNonce { nonce: u64, bits: u32 },
}

/// Binds a typed boundary to `transcript` and squeezes its 256-bit PoW seed.
///
/// Validation happens before the transcript is mutated.  The exact absorbed
/// order is the global frame, type-level domain, round index as `u64` LE, and
/// difficulty as `u32` LE.
pub fn derive_grinding_seed<D, T>(
    transcript: &mut T,
    round: GrindingRound<D>,
    bits: u32,
) -> Result<GrindingSeed, GrindingError>
where
    D: GrindingDomain,
    T: Transcript,
{
    derive_grinding_seed_in_domain(transcript, D::DOMAIN, round.index, bits)
}

/// [`derive_grinding_seed`] for a domain chosen at runtime: the same bytes
/// as the typed boundary with `D::DOMAIN == domain`.
pub fn derive_grinding_seed_in_domain<T: Transcript>(
    transcript: &mut T,
    domain: &[u8],
    index: u64,
    bits: u32,
) -> Result<GrindingSeed, GrindingError> {
    validate_difficulty(bits)?;
    if domain.is_empty() {
        return Err(GrindingError::EmptyDomain);
    }
    transcript.absorb_slice(GRINDING_TRANSCRIPT_DOMAIN);
    transcript.absorb_slice(domain);
    transcript.absorb_slice(&index.to_le_bytes());
    transcript.absorb_slice(&bits.to_le_bytes());
    Ok(transcript.get_challenge())
}

/// [`grind_and_absorb`] for a domain chosen at runtime.
pub fn grind_and_absorb_in_domain<T: Transcript>(
    transcript: &mut T,
    domain: &[u8],
    index: u64,
    bits: u32,
) -> Result<u64, GrindingError> {
    let seed = derive_grinding_seed_in_domain(transcript, domain, index, bits)?;
    let nonce = find_grinding_nonce(&seed, bits)?;
    absorb_grinding_nonce(transcript, nonce);
    Ok(nonce)
}

/// [`verify_and_absorb`] for a domain chosen at runtime.
pub fn verify_and_absorb_in_domain<T: Transcript>(
    transcript: &mut T,
    domain: &[u8],
    index: u64,
    bits: u32,
    nonce: u64,
) -> Result<(), GrindingError> {
    let seed = derive_grinding_seed_in_domain(transcript, domain, index, bits)?;
    let valid = grinding_nonce_is_valid_unchecked(&seed, nonce, bits);
    absorb_grinding_nonce(transcript, nonce);
    if valid {
        Ok(())
    } else {
        Err(GrindingError::InvalidNonce { nonce, bits })
    }
}

/// Finds the smallest valid nonce and absorbs its canonical encoding.
///
/// With the `parallel` feature, sufficiently expensive searches use ordered
/// parallel waves.  The result remains byte-for-byte identical to a serial
/// scan from nonce zero.
pub fn grind_and_absorb<D, T>(
    transcript: &mut T,
    round: GrindingRound<D>,
    bits: u32,
) -> Result<u64, GrindingError>
where
    D: GrindingDomain,
    T: Transcript,
{
    let seed = derive_grinding_seed(transcript, round, bits)?;
    let nonce = find_grinding_nonce(&seed, bits)?;
    absorb_grinding_nonce(transcript, nonce);
    Ok(nonce)
}

/// Checks a proof nonce and absorbs its canonical encoding.
///
/// As in the existing Flock challenger bridge, the nonce is absorbed even
/// when it is invalid.  This gives prover and verifier one unambiguous proof
/// item boundary; callers must reject the returned error and must not derive
/// further protocol challenges from a failed verification.
pub fn verify_and_absorb<D, T>(
    transcript: &mut T,
    round: GrindingRound<D>,
    bits: u32,
    nonce: u64,
) -> Result<(), GrindingError>
where
    D: GrindingDomain,
    T: Transcript,
{
    let seed = derive_grinding_seed(transcript, round, bits)?;
    let valid = grinding_nonce_is_valid_unchecked(&seed, nonce, bits);
    absorb_grinding_nonce(transcript, nonce);
    if valid {
        Ok(())
    } else {
        Err(GrindingError::InvalidNonce { nonce, bits })
    }
}

/// Finds the smallest `u64` nonce satisfying `bits` for an explicit seed.
pub fn find_grinding_nonce(seed: &GrindingSeed, bits: u32) -> Result<u64, GrindingError> {
    validate_difficulty(bits)?;

    #[cfg(feature = "parallel")]
    {
        // Below this point the pool broadcast exceeds the expected search.
        const PARALLEL_SEARCH_MIN_BITS: u32 = 12;
        if bits >= PARALLEL_SEARCH_MIN_BITS {
            return find_grinding_nonce_parallel(seed, bits);
        }
    }

    find_grinding_nonce_sequential(seed, bits)
}

/// Checks `nonce` against an explicit seed after validating the difficulty.
pub fn grinding_nonce_is_valid(
    seed: &GrindingSeed,
    nonce: u64,
    bits: u32,
) -> Result<bool, GrindingError> {
    validate_difficulty(bits)?;
    Ok(grinding_nonce_is_valid_unchecked(seed, nonce, bits))
}

fn validate_difficulty(bits: u32) -> Result<(), GrindingError> {
    if !(1..=MAX_GRINDING_BITS).contains(&bits) {
        return Err(GrindingError::InvalidDifficulty { bits });
    }
    Ok(())
}

fn absorb_grinding_nonce(transcript: &mut impl Transcript, nonce: u64) {
    transcript.absorb_slice(GRINDING_NONCE_DOMAIN);
    transcript.absorb_slice(&nonce.to_le_bytes());
}

fn find_grinding_nonce_sequential(seed: &GrindingSeed, bits: u32) -> Result<u64, GrindingError> {
    first_pow_nonce(seed.as_bytes(), 0, u64::MAX, bits)
        .or_else(|| pow_ok(seed.as_bytes(), u64::MAX, bits).then_some(u64::MAX))
        .ok_or(GrindingError::NonceSpaceExhausted)
}

#[cfg(feature = "parallel")]
fn find_grinding_nonce_parallel(seed: &GrindingSeed, bits: u32) -> Result<u64, GrindingError> {
    smallest_pow_nonce(seed.as_bytes(), bits).ok_or(GrindingError::NonceSpaceExhausted)
}

fn grinding_nonce_is_valid_unchecked(seed: &GrindingSeed, nonce: u64, bits: u32) -> bool {
    pow_ok(seed.as_bytes(), nonce, bits)
}

// ---------------------------------------------------------------------
// Per-challenge grinding transcripts (the forest/GKR + ring-switch hooks)
// ---------------------------------------------------------------------

/// The forest/opening grinding domain: every challenge drawn inside the
/// exponent-fold opening region (forest sumcheck rounds, claim
/// unifications, the ring-switch/batching draws) is preceded by one typed
/// boundary in this domain when the security profile sets a nonzero
/// difficulty. Paper §Instantiation: each GKR round carries `3/|K|`
/// (~2^-126.4), so λ = 128 takes two bits per round; the ring-switch
/// round's `1/|K|` takes one — a uniform per-draw difficulty of
/// `max(2, 1)` covers both.
pub enum ForestRoundGrinding {}

impl GrindingDomain for ForestRoundGrinding {
    const DOMAIN: &'static [u8] = b"bitz/forest/round-grinding/v1";
}

/// Prover-side transcript adapter: before every challenge drawn through
/// it, grinds one boundary in the domain `D` (default
/// [`ForestRoundGrinding`]) at the configured difficulty and records the
/// nonce. At difficulty 0 it is a transparent pass-through — not one
/// transcript byte moves.
///
/// Wrap exactly the region whose rounds the difficulty covers and call
/// [`Self::finish`] to recover the nonces for the proof; leave any inner
/// Ligerito call OUTSIDE the wrapper (flock carries its own grinding
/// configuration).
pub struct ProverGrindingTranscript<'a, T, D = ForestRoundGrinding> {
    inner: &'a mut T,
    bits: u32,
    domain: &'static [u8],
    next_index: u64,
    nonces: Vec<u64>,
    _domain: PhantomData<fn() -> D>,
}

impl<'a, T: Transcript, D: GrindingDomain> ProverGrindingTranscript<'a, T, D> {
    /// Wraps `inner` at `bits` difficulty per drawn challenge.
    pub fn new(inner: &'a mut T, bits: u32) -> Self {
        Self::new_in_domain(inner, bits, D::DOMAIN)
    }
}

impl<'a, T: Transcript, D> ProverGrindingTranscript<'a, T, D> {
    /// Wraps `inner` at `bits` difficulty per drawn challenge, grinding in
    /// a domain chosen at runtime (the type parameter is then nominal).
    pub fn new_in_domain(inner: &'a mut T, bits: u32, domain: &'static [u8]) -> Self {
        Self {
            inner,
            bits,
            domain,
            next_index: 0,
            nonces: Vec::new(),
            _domain: PhantomData,
        }
    }

    /// The nonces ground so far, in draw order (empty at difficulty 0).
    pub fn finish(self) -> Vec<u64> {
        self.nonces
    }
}

impl<T: Transcript, D> Transcript for ProverGrindingTranscript<'_, T, D> {
    fn fill_sampling_bytes(&mut self, output: &mut [u8]) {
        self.inner.fill_sampling_bytes(output);
    }

    fn begin_sampling(&mut self) {
        if self.bits > 0 {
            let index = self.next_index;
            self.next_index = self.next_index.wrapping_add(1);
            let nonce = grind_and_absorb_in_domain(self.inner, self.domain, index, self.bits)
                .expect("per-round grinding difficulty is validated by the profile");
            self.nonces.push(nonce);
        }
        self.inner.begin_sampling();
    }

    fn get_challenge<C: ConstTranscribable>(&mut self) -> C {
        if self.bits > 0 {
            let index = self.next_index;
            self.next_index = self.next_index.wrapping_add(1);
            let nonce = grind_and_absorb_in_domain(self.inner, self.domain, index, self.bits)
                .expect("per-round grinding difficulty is validated by the profile");
            self.nonces.push(nonce);
        }
        self.inner.get_challenge()
    }

    fn absorb_inner(&mut self, v: &[u8]) {
        self.inner.absorb_inner(v);
    }
}

/// Verifier-side twin of [`ProverGrindingTranscript`]: before every drawn
/// challenge it checks (and absorbs) the next proof nonce at the same
/// difficulty. Nonce failures and count mismatches are deferred to
/// [`Self::finish`] so the transcript stays deterministic — the caller
/// MUST propagate that result before accepting the proof.
pub struct VerifierGrindingTranscript<'a, 'n, T, D = ForestRoundGrinding> {
    inner: &'a mut T,
    bits: u32,
    domain: &'static [u8],
    next_index: u64,
    nonces: &'n [u64],
    consumed: usize,
    failure: Option<GrindingError>,
    _domain: PhantomData<fn() -> D>,
}

impl<'a, 'n, T: Transcript, D: GrindingDomain> VerifierGrindingTranscript<'a, 'n, T, D> {
    /// Wraps `inner`, checking `nonces` at `bits` difficulty per draw.
    pub fn new(inner: &'a mut T, bits: u32, nonces: &'n [u64]) -> Self {
        Self::new_in_domain(inner, bits, nonces, D::DOMAIN)
    }
}

impl<'a, 'n, T: Transcript, D> VerifierGrindingTranscript<'a, 'n, T, D> {
    /// Wraps `inner`, checking `nonces` at `bits` difficulty per draw in a
    /// domain chosen at runtime.
    pub fn new_in_domain(
        inner: &'a mut T,
        bits: u32,
        nonces: &'n [u64],
        domain: &'static [u8],
    ) -> Self {
        Self {
            inner,
            bits,
            domain,
            next_index: 0,
            nonces,
            consumed: 0,
            failure: None,
            _domain: PhantomData,
        }
    }

    /// Succeeds iff every drawn challenge consumed one valid nonce and no
    /// nonce is left over.
    #[must_use = "an unverified grinding region proves nothing"]
    pub fn finish(self) -> Result<(), GrindingError> {
        if let Some(failure) = self.failure {
            return Err(failure);
        }
        if self.consumed != self.nonces.len() {
            return Err(GrindingError::InvalidNonce {
                nonce: self.nonces.get(self.consumed).copied().unwrap_or(0),
                bits: self.bits,
            });
        }
        Ok(())
    }
}

impl<T: Transcript, D> Transcript for VerifierGrindingTranscript<'_, '_, T, D> {
    fn fill_sampling_bytes(&mut self, output: &mut [u8]) {
        self.inner.fill_sampling_bytes(output);
    }

    fn begin_sampling(&mut self) {
        if self.bits > 0 {
            let index = self.next_index;
            self.next_index = self.next_index.wrapping_add(1);
            // A missing nonce absorbs a canonical zero so the transcript
            // stays deterministic; `finish` reports the failure.
            let nonce = self.nonces.get(self.consumed).copied().unwrap_or(0);
            self.consumed = self.consumed.saturating_add(1);
            if let Err(error) =
                verify_and_absorb_in_domain(self.inner, self.domain, index, self.bits, nonce)
            {
                self.failure.get_or_insert(error);
            }
        }
        self.inner.begin_sampling();
    }

    fn get_challenge<C: ConstTranscribable>(&mut self) -> C {
        if self.bits > 0 {
            let index = self.next_index;
            self.next_index = self.next_index.wrapping_add(1);
            // A missing nonce absorbs a canonical zero so the transcript
            // stays deterministic; `finish` reports the failure.
            let nonce = self.nonces.get(self.consumed).copied().unwrap_or(0);
            self.consumed = self.consumed.saturating_add(1);
            if let Err(error) =
                verify_and_absorb_in_domain(self.inner, self.domain, index, self.bits, nonce)
            {
                self.failure.get_or_insert(error);
            }
        }
        self.inner.get_challenge()
    }

    fn absorb_inner(&mut self, v: &[u8]) {
        self.inner.absorb_inner(v);
    }
}

#[cfg(test)]
mod tests {
    use crate::transcript::{Blake3Transcript, traits::Transcript};

    use super::*;

    enum InitialMessage {}
    enum OuterRound {}
    enum Empty {}

    impl GrindingDomain for InitialMessage {
        const DOMAIN: &'static [u8] = b"test/spartan-grinding/initial/v1";
    }

    impl GrindingDomain for OuterRound {
        const DOMAIN: &'static [u8] = b"test/spartan-grinding/outer-round/v1";
    }

    impl GrindingDomain for Empty {
        const DOMAIN: &'static [u8] = b"";
    }

    fn transcript() -> Blake3Transcript {
        let mut transcript = Blake3Transcript::new();
        transcript.absorb_slice(b"fixed public statement");
        transcript.absorb_slice(b"fixed prover message");
        transcript
    }

    #[test]
    fn internal_sampler_reads_do_not_create_grinding_rounds() {
        let mut prover = transcript();
        let mut verifier = transcript();
        let mut p = ProverGrindingTranscript::<_, OuterRound>::new(&mut prover, 1);
        let prime = crate::ext_proj::sample_prime_context(&mut p, 251, 251, 128).unwrap();
        let p_next: u64 = p.get_challenge();
        let nonces = p.finish();
        assert_eq!(
            nonces.len(),
            1,
            "only the explicit following challenge is ground"
        );
        let mut v = VerifierGrindingTranscript::<_, OuterRound>::new(&mut verifier, 1, &nonces);
        let replay = crate::ext_proj::sample_prime_context(&mut v, 251, 251, 128).unwrap();
        assert_eq!(prime.modulus(), replay.modulus());
        assert_eq!(p_next, v.get_challenge::<u64>());
        v.finish().unwrap();
        assert_eq!(
            prover.get_challenge::<u64>(),
            verifier.get_challenge::<u64>()
        );
    }

    #[test]
    fn prover_and_verifier_are_deterministic_and_continue_in_lockstep() {
        let mut prover = transcript();
        let mut verifier = transcript();

        let nonce =
            grind_and_absorb(&mut prover, GrindingRound::<InitialMessage>::new(0), 9).unwrap();
        verify_and_absorb(
            &mut verifier,
            GrindingRound::<InitialMessage>::new(0),
            9,
            nonce,
        )
        .unwrap();

        assert_eq!(
            prover.get_challenge::<u128>(),
            verifier.get_challenge::<u128>()
        );

        let mut repeated = transcript();
        assert_eq!(
            grind_and_absorb(&mut repeated, GrindingRound::<InitialMessage>::new(0), 9,).unwrap(),
            nonce
        );
    }

    #[test]
    fn invalid_nonce_is_rejected_and_still_absorbed_canonically() {
        const BITS: u32 = 8;
        let mut prover = transcript();
        let valid =
            grind_and_absorb(&mut prover, GrindingRound::<InitialMessage>::new(0), BITS).unwrap();

        let mut seed_transcript = transcript();
        let seed = derive_grinding_seed(
            &mut seed_transcript,
            GrindingRound::<InitialMessage>::new(0),
            BITS,
        )
        .unwrap();
        let invalid = (0..=u64::MAX)
            .find(|&nonce| nonce != valid && !grinding_nonce_is_valid(&seed, nonce, BITS).unwrap())
            .unwrap();

        let mut verifier = transcript();
        assert_eq!(
            verify_and_absorb(
                &mut verifier,
                GrindingRound::<InitialMessage>::new(0),
                BITS,
                invalid,
            ),
            Err(GrindingError::InvalidNonce {
                nonce: invalid,
                bits: BITS,
            })
        );

        // Reproduce the documented framing explicitly. This pins both the
        // nonce's byte order and the fact that invalid nonces are absorbed.
        seed_transcript.absorb_slice(GRINDING_NONCE_DOMAIN);
        seed_transcript.absorb_slice(&invalid.to_le_bytes());
        assert_eq!(
            verifier.get_challenge::<u128>(),
            seed_transcript.get_challenge::<u128>()
        );
    }

    #[test]
    fn domain_round_and_difficulty_separate_seeds() {
        let initial = derive_grinding_seed(
            &mut transcript(),
            GrindingRound::<InitialMessage>::new(0),
            8,
        )
        .unwrap();
        let other_domain =
            derive_grinding_seed(&mut transcript(), GrindingRound::<OuterRound>::new(0), 8)
                .unwrap();
        let other_round = derive_grinding_seed(
            &mut transcript(),
            GrindingRound::<InitialMessage>::new(1),
            8,
        )
        .unwrap();
        let other_difficulty = derive_grinding_seed(
            &mut transcript(),
            GrindingRound::<InitialMessage>::new(0),
            9,
        )
        .unwrap();

        assert_ne!(initial, other_domain);
        assert_ne!(initial, other_round);
        assert_ne!(initial, other_difficulty);
    }

    #[test]
    fn invalid_boundary_configuration_does_not_mutate_transcript() {
        for bits in [0, MAX_GRINDING_BITS + 1] {
            let mut actual = transcript();
            let mut untouched = actual.clone();
            assert_eq!(
                derive_grinding_seed(&mut actual, GrindingRound::<InitialMessage>::new(0), bits,),
                Err(GrindingError::InvalidDifficulty { bits })
            );
            assert_eq!(
                actual.get_challenge::<u128>(),
                untouched.get_challenge::<u128>()
            );
        }

        let mut actual = transcript();
        let mut untouched = actual.clone();
        assert_eq!(
            derive_grinding_seed(&mut actual, GrindingRound::<Empty>::new(0), 1),
            Err(GrindingError::EmptyDomain)
        );
        assert_eq!(
            actual.get_challenge::<u128>(),
            untouched.get_challenge::<u128>()
        );
    }

    #[test]
    fn grinding_transcripts_stay_in_lockstep_and_gate_the_nonces() {
        const BITS: u32 = 6;
        let mut prover_inner = transcript();
        let mut prover: ProverGrindingTranscript<_, ForestRoundGrinding> =
            ProverGrindingTranscript::new(&mut prover_inner, BITS);
        let a: u128 = prover.get_challenge();
        prover.absorb_slice(b"round message");
        let b: u128 = prover.get_challenge();
        let nonces = prover.finish();
        assert_eq!(nonces.len(), 2);

        let mut verifier_inner = transcript();
        let mut verifier: VerifierGrindingTranscript<_, ForestRoundGrinding> =
            VerifierGrindingTranscript::new(&mut verifier_inner, BITS, &nonces);
        let va: u128 = verifier.get_challenge();
        verifier.absorb_slice(b"round message");
        let vb: u128 = verifier.get_challenge();
        verifier.finish().unwrap();
        assert_eq!((a, b), (va, vb));

        // A tampered nonce is caught at finish.
        let mut bad = nonces.clone();
        bad[1] ^= 1;
        let mut verifier_inner = transcript();
        let mut verifier: VerifierGrindingTranscript<_, ForestRoundGrinding> =
            VerifierGrindingTranscript::new(&mut verifier_inner, BITS, &bad);
        let _: u128 = verifier.get_challenge();
        verifier.absorb_slice(b"round message");
        let _: u128 = verifier.get_challenge();
        assert!(verifier.finish().is_err());

        // Leftover nonces are caught at finish.
        let mut verifier_inner = transcript();
        let mut verifier: VerifierGrindingTranscript<_, ForestRoundGrinding> =
            VerifierGrindingTranscript::new(&mut verifier_inner, BITS, &nonces);
        let _: u128 = verifier.get_challenge();
        assert!(verifier.finish().is_err());
    }

    #[test]
    fn zero_difficulty_grinding_transcript_is_a_transparent_passthrough() {
        let mut wrapped_inner = transcript();
        let mut wrapped: ProverGrindingTranscript<_, ForestRoundGrinding> =
            ProverGrindingTranscript::new(&mut wrapped_inner, 0);
        wrapped.absorb_slice(b"message");
        let a: u128 = wrapped.get_challenge();
        assert!(wrapped.finish().is_empty());

        let mut plain = transcript();
        plain.absorb_slice(b"message");
        let b: u128 = plain.get_challenge();
        assert_eq!(a, b);

        let mut verifier_inner = transcript();
        let mut verifier: VerifierGrindingTranscript<_, ForestRoundGrinding> =
            VerifierGrindingTranscript::new(&mut verifier_inner, 0, &[]);
        verifier.absorb_slice(b"message");
        let c: u128 = verifier.get_challenge();
        verifier.finish().unwrap();
        assert_eq!(a, c);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_search_returns_the_serial_minimum() {
        let seed = GrindingSeed::from_bytes([0x5a; 32]);
        let serial = find_grinding_nonce_sequential(&seed, 11).unwrap();
        let parallel = find_grinding_nonce_parallel(&seed, 11).unwrap();
        assert_eq!(parallel, serial);
        assert_eq!(find_grinding_nonce(&seed, 11).unwrap(), serial);
    }
}

#[cfg(test)]
mod sampling_boundary_tests {
    use super::*;
    use crate::transcript::{Blake3Transcript, traits::ConstTranscribable};

    struct RetryingTranscript {
        inner: Blake3Transcript,
        word_reads: usize,
    }
    impl RetryingTranscript {
        fn new() -> Self {
            Self {
                inner: Blake3Transcript::new(),
                word_reads: 0,
            }
        }
    }
    impl Transcript for RetryingTranscript {
        fn get_challenge<C: ConstTranscribable>(&mut self) -> C {
            self.inner.get_challenge()
        }
        fn fill_sampling_bytes(&mut self, out: &mut [u8]) {
            self.inner.fill_sampling_bytes(out);
            self.word_reads += 1;
            out.fill(if self.word_reads <= 2 { 255 } else { 0 });
            if self.word_reads == 3 {
                out[0] = 42;
            }
        }
        fn absorb_inner(&mut self, value: &[u8]) {
            self.inner.absorb_inner(value);
        }
    }
    enum SamplingRound {}
    impl GrindingDomain for SamplingRound {
        const DOMAIN: &'static [u8] = b"test/field-sampling";
    }

    #[test]
    fn field_sampling_retries_share_one_grinding_boundary_and_replay() {
        let field = field::FpCtx::from_prime_u128((1u128 << 127) - 1);
        let mut prover = RetryingTranscript::new();
        let mut grinded = ProverGrindingTranscript::<_, SamplingRound>::new(&mut prover, 4);
        let value =
            crate::piop::spartan::squeeze_field::<field::Fp<2>, _>(&mut grinded, &field).unwrap();
        let nonces = grinded.finish();
        assert_eq!(nonces.len(), 1);
        assert_eq!(prover.word_reads, 4);
        assert_eq!(u128::from(field.to_integer(&value)), 42);
        let mut verifier = RetryingTranscript::new();
        let mut grinded =
            VerifierGrindingTranscript::<_, SamplingRound>::new(&mut verifier, 4, &nonces);
        let replay =
            crate::piop::spartan::squeeze_field::<field::Fp<2>, _>(&mut grinded, &field).unwrap();
        grinded.finish().unwrap();
        assert_eq!(value, replay);
        assert_eq!(verifier.word_reads, 4);
        assert_eq!(prover.inner.state_digest(), verifier.inner.state_digest());
    }
}
