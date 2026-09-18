pub mod traits;

use crate::poly::coefficient::PolynomialField;
use crate::transcript::traits::{ConstTranscribable, GenTranscribable, Transcript};

use crate::utils::add;

/// A cryptographic transcript implementation using the BLAKE3 hash
/// function. Used for Fiat-Shamir transformations in zero-knowledge proof
/// systems.
#[derive(Debug, Clone)]
pub struct Blake3Transcript {
    /// The underlying BLAKE3 hasher that maintains the transcript state.
    hasher: blake3::Hasher,
}

impl Default for Blake3Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Blake3Transcript {
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    /// The BLAKE3 digest of everything absorbed so far, without touching the
    /// transcript state. Two transcripts that absorbed the same bytes in the
    /// same order (and drew the same challenges) have the same digest, so
    /// this pins the Fiat–Shamir transcript independently of how a proof
    /// happens to be represented in memory.
    pub fn state_digest(&self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }

    /// Generates a specified number of pseudorandom bytes based on the current
    /// transcript state. Uses a counter-based approach to generate enough
    /// bytes from the hasher.
    ///
    /// Note that this does NOT update the internal state of the hasher
    #[allow(clippy::arithmetic_side_effects)]
    fn fill_with_random_bytes(&mut self, buf: &mut [u8]) {
        self.hasher.finalize_xof().fill(buf);
    }
}

impl Transcript for Blake3Transcript {
    fn fill_sampling_bytes(&mut self, output: &mut [u8]) {
        self.fill_with_random_bytes(output);
        self.hasher.update(b"bitz/sampling-read/v1");
        self.hasher.update(&(output.len() as u64).to_le_bytes());
        self.hasher.update(output);
    }

    fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.fill_with_random_bytes(&mut buf);
        self.hasher.update(&[0x12]);
        self.hasher.update(&buf);
        self.hasher.update(&[0x34]);
        // Canonical codecs reject high padding bits. Sampling those bits away
        // is unbiased for fixed-width binary/integer challenges.
        assert!(T::NUM_BITS <= T::NUM_BYTES * 8);
        for (i, byte) in buf.iter_mut().enumerate() {
            let remaining = T::NUM_BITS.saturating_sub(i * 8).min(8);
            *byte &= ((1u16 << remaining) - 1) as u8;
        }
        T::read_transcription_bytes_exact(&buf)
    }

    fn absorb_inner(&mut self, v: &[u8]) {
        // For large inputs (~MB+ — e.g. the public-column absorb at
        // the start of an F_2 commit, ~235 MB at SHA-256 F_2 nvars=22)
        // Blake3's rayon-parallel chunk-tree path beats the single-
        // threaded `update` by ~4-8× on M-series cores. The threshold
        // is set so small absorbs (challenges, framing bytes,
        // per-round transcript writes) still take the cheap path —
        // `update_rayon` has a one-shot rayon scope setup that
        // dominates for inputs under a few hundred KB.
        const RAYON_THRESHOLD: usize = 256 * 1024;
        if v.len() >= RAYON_THRESHOLD {
            self.hasher.update_rayon(v);
        } else {
            self.hasher.update(v);
        }
    }
}

pub fn read_field_cfg<F>(bytes: &[u8]) -> F::Config
where
    F: PolynomialField,
    F::Modulus: ConstTranscribable,
{
    let mod_size = F::Modulus::NUM_BYTES;
    let modulus = F::Modulus::read_transcription_bytes_exact(&bytes[..mod_size]);
    F::config_from_modulus(&modulus).expect("valid field modulus in proof transcription")
}

pub fn read_field_vec_with_cfg<F>(bytes: &[u8], field_cfg: &F::Config) -> Vec<F>
where
    F: PolynomialField,
    F::Inner: ConstTranscribable,
{
    let inner_size = F::Inner::NUM_BYTES;
    bytes
        .chunks_exact(inner_size)
        .map(F::Inner::read_transcription_bytes_exact)
        .map(|inner| F::new_unchecked_with_cfg(inner, field_cfg))
        .collect()
}

pub fn append_field_cfg<'a, F>(buf: &'a mut [u8], modulus: &F::Modulus) -> &'a mut [u8]
where
    F: PolynomialField,
    F::Modulus: ConstTranscribable,
{
    let mod_size = F::Modulus::NUM_BYTES;
    let (buf, rest) = buf.split_at_mut(mod_size);
    modulus.write_transcription_bytes_exact(buf);
    rest
}

pub fn append_field_vec_inner<'a, F>(buf: &'a mut [u8], slice: &[F]) -> &'a mut [u8]
where
    F: PolynomialField,
    F::Inner: ConstTranscribable,
{
    let inner_size = F::Inner::NUM_BYTES;
    let mut offset = 0;
    for elem in slice {
        let offset_end = add!(offset, inner_size);
        elem.inner()
            .write_transcription_bytes_exact(&mut buf[offset..offset_end]);
        offset = offset_end;
    }
    &mut buf[offset..]
}

// `#[macro_export]` macros land at the crate root; re-export them here so
// vendored `zinc_transcript::`-style paths keep working after the rename.
pub use crate::{delegate_const_transcribable, delegate_transcribable};

#[cfg(test)]
mod framing_tests {
    use super::*;
    #[test]
    fn bit_sampling_masks_padding_but_canonical_decoding_rejects_it() {
        use crate::transcript::traits::GenTranscribable;
        use field::Bit;
        let mut transcript = Blake3Transcript::new();
        let mut replay = Blake3Transcript::new();
        let bits: Vec<_> = (0..256)
            .map(|_| transcript.get_challenge::<Bit>())
            .collect();
        let expected: Vec<_> = (0..256).map(|_| replay.get_challenge::<Bit>()).collect();
        assert_eq!(bits, expected);
        assert!(bits.contains(&Bit::ZERO) && bits.contains(&Bit::ONE));
        assert!(std::panic::catch_unwind(|| Bit::read_transcription_bytes_exact(&[2])).is_err());
        for _ in 0..256 {
            let value: field::B127 = transcript.get_challenge();
            assert_eq!(value.as_words()[1] >> 63, 0);
        }
    }
    #[test]
    fn absorption_is_length_framed_and_sampling_advances() {
        let mut a = Blake3Transcript::new();
        let mut b = Blake3Transcript::new();
        a.absorb_slice(&[1, 7, 6, 2]);
        b.absorb_slice(&[1]);
        b.absorb_slice(&[2]);
        let mut first = [0; 16];
        let mut second = [0; 16];
        let mut other = [0; 16];
        a.fill_sampling_bytes(&mut first);
        a.fill_sampling_bytes(&mut second);
        b.fill_sampling_bytes(&mut other);
        assert_ne!(first, second);
        assert_ne!(first, other);
    }
}
