//! Byte-codec for the BitZ host proof stream.
//!
//! The complete BitZ proof object ([`crate::ligerito_flock::IntEvalRsLigModQProof`])
//! serializes field-by-field on the zinc side (merged forests, chunk folds,
//! pre-sumchecks, ring-switch messages) via this codec, with flock's serde
//! [`LigeritoProof`](flock_core::pcs::ligerito::LigeritoProof) embedded as a
//! single **length-prefixed `bincode` 1.3 blob** (bincode being flock's own
//! pinned encoder). Scalars are little-endian; the zinc sumcheck proofs reuse
//! the crate's [`Transcribable`] length-prefixed encoding.

use crate::poly::univariate::binary_gf128::Gf128 as Gf;
use crate::transcript::traits::Transcribable;

/// A decode failure of the BitZ proof stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// The buffer ended before a field could be fully read.
    Truncated,
    /// A field was encoded non-minimally, so two byte strings would decode
    /// to the same proof. The codec is canonical: every value has exactly
    /// one encoding, and the decoder rejects the others.
    NonCanonical,
    /// The embedded `bincode` `LigeritoProof` blob failed to decode.
    Bincode(String),
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CodecError::Truncated => write!(f, "proof stream truncated"),
            CodecError::NonCanonical => write!(f, "proof stream encoded non-minimally"),
            CodecError::Bincode(e) => write!(f, "LigeritoProof decode: {e}"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Append-only writer over a `Vec<u8>`.
#[derive(Default)]
pub struct Writer(Vec<u8>);

impl Writer {
    /// A fresh empty writer.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// The accumulated bytes.
    pub fn into_vec(self) -> Vec<u8> {
        self.0
    }

    /// A length / small integer as an 8-byte little-endian `u64`.
    pub fn len(&mut self, n: usize) {
        self.0.extend_from_slice(&(n as u64).to_le_bytes());
    }

    /// A `u128` as 16 little-endian bytes.
    pub fn u128(&mut self, u: u128) {
        self.0.extend_from_slice(&u.to_le_bytes());
    }

    /// A `GF(2^128)` element as its two little-endian 64-bit words (16 bytes).
    pub fn gf(&mut self, g: &Gf) {
        let w = g.as_words();
        self.0.extend_from_slice(&w[0].to_le_bytes());
        self.0.extend_from_slice(&w[1].to_le_bytes());
    }

    /// A [`Transcribable`] value (self-describing, length-prefixed).
    pub fn transcribable<T: Transcribable>(&mut self, x: &T) {
        let start = self.0.len();
        let n = T::LENGTH_NUM_BYTES + x.get_num_bytes();
        self.0.resize(start + n, 0);
        x.write_transcription_bytes_subset(&mut self.0[start..]);
    }

    /// Raw bytes (no length prefix — the caller frames these).
    pub fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
}

/// Cursored reader over a byte slice; the inverse of [`Writer`].
pub struct Reader<'a> {
    buf: &'a [u8],
    cur: usize,
}

impl<'a> Reader<'a> {
    /// A reader positioned at the start of `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, cur: 0 }
    }

    fn take_raw(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        let end = self.cur.checked_add(n).ok_or(CodecError::Truncated)?;
        if end > self.buf.len() {
            return Err(CodecError::Truncated);
        }
        let out = &self.buf[self.cur..end];
        self.cur = end;
        Ok(out)
    }

    /// Read an 8-byte little-endian length / small integer.
    pub fn len(&mut self) -> Result<usize, CodecError> {
        let b = self.take_raw(8)?;
        Ok(u64::from_le_bytes(b.try_into().expect("8 bytes")) as usize)
    }

    /// Read a 16-byte little-endian `u128`.
    pub fn u128(&mut self) -> Result<u128, CodecError> {
        let b = self.take_raw(16)?;
        Ok(u128::from_le_bytes(b.try_into().expect("16 bytes")))
    }

    /// Read a `GF(2^128)` element (two little-endian 64-bit words).
    pub fn gf(&mut self) -> Result<Gf, CodecError> {
        let b = self.take_raw(16)?;
        let lo = u64::from_le_bytes(b[0..8].try_into().expect("8 bytes"));
        let hi = u64::from_le_bytes(b[8..16].try_into().expect("8 bytes"));
        Ok(Gf::from_polynomial_words([lo, hi]))
    }

    /// Read a [`Transcribable`] value written by [`Writer::transcribable`].
    pub fn transcribable<T: Transcribable>(&mut self) -> Result<T, CodecError> {
        // The subset reader self-describes its length via the prefix. Guard BOTH
        // the length prefix AND the declared payload length before delegating, so
        // a truncated tail or a tampered length prefix is a clean `Truncated`
        // error rather than a panic inside `read_transcription_bytes_subset`
        // (this codec is required to be tamper-rejecting).
        if self.cur + T::LENGTH_NUM_BYTES > self.buf.len() {
            return Err(CodecError::Truncated);
        }
        let rem = &self.buf[self.cur..];
        let num_bytes = T::read_num_bytes(&rem[..T::LENGTH_NUM_BYTES]);
        let need = T::LENGTH_NUM_BYTES
            .checked_add(num_bytes)
            .ok_or(CodecError::Truncated)?;
        if need > rem.len() {
            return Err(CodecError::Truncated);
        }
        let (val, rest) = T::read_transcription_bytes_subset(rem);
        self.cur = self.buf.len() - rest.len();
        Ok(val)
    }

    /// Read `n` raw bytes (the framed `bincode` blob).
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        self.take_raw(n)
    }

    /// Bytes left unread. Callers bound attacker-controlled element counts
    /// by this before reserving, so a tampered length prefix costs a
    /// `Truncated` error rather than a huge speculative allocation.
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.cur)
    }
}
