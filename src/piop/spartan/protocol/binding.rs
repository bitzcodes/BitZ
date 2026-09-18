//! Digest helpers shared by every relation's statement binding and bridge
//! digest: one BLAKE3 hasher with the canonical field encodings.

use blake3::Hasher;
use flock_core::{
    merkle::HashKind,
    pcs::{
        commit::PcsParams,
        ligerito::{LigeritoProfile, ProverConfig as LigProverConfig},
    },
};

use super::{ProtocolError, SpartanBitzField};

/// Stable one-byte code of a flock Ligerito profile.
pub const fn profile_code(profile: LigeritoProfile) -> u8 {
    match profile {
        LigeritoProfile::Fast => 0,
        LigeritoProfile::Slim => 1,
        LigeritoProfile::Secure => 2,
        LigeritoProfile::Slim3 => 3,
    }
}

/// Stable one-byte code of a flock Merkle hash.
pub const fn hash_code(hash: HashKind) -> u8 {
    match hash {
        HashKind::Sha256 => 0,
        HashKind::Blake3 => 1,
    }
}

/// A BLAKE3 hasher with the encodings every binding uses: raw bytes, host
/// lengths as `u64` little-endian words, `u128` little-endian words and
/// canonical 16-byte field elements.
pub struct BindingHasher {
    hasher: Hasher,
}

impl Default for BindingHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl BindingHasher {
    pub fn new() -> Self {
        Self {
            hasher: Hasher::new(),
        }
    }

    /// Raw bytes, no framing.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.hasher.update(bytes);
        self
    }

    /// One raw byte.
    pub fn byte(&mut self, byte: u8) -> &mut Self {
        self.hasher.update(&[byte]);
        self
    }

    /// A host length or count as a `u64` little-endian word.
    pub fn usize(&mut self, value: usize) -> Result<&mut Self, ProtocolError> {
        let value = u64::try_from(value).map_err(|_| ProtocolError::BindingEncodingOverflow)?;
        self.hasher.update(&value.to_le_bytes());
        Ok(self)
    }

    /// Several host lengths in order.
    pub fn usizes(&mut self, values: &[usize]) -> Result<&mut Self, ProtocolError> {
        for &value in values {
            self.usize(value)?;
        }
        Ok(self)
    }

    /// A `u32` parameter, encoded like a host length.
    pub fn u32(&mut self, value: u32) -> Result<&mut Self, ProtocolError> {
        self.usize(value as usize)
    }

    /// A `u64` little-endian word.
    pub fn u64_le(&mut self, value: u64) -> &mut Self {
        self.hasher.update(&value.to_le_bytes());
        self
    }

    /// A `u128` little-endian word.
    pub fn u128_le(&mut self, value: u128) -> &mut Self {
        self.hasher.update(&value.to_le_bytes());
        self
    }

    /// A canonical 16-byte field element.
    pub fn element(&mut self, value: &SpartanBitzField, field: &super::FieldConfig) -> &mut Self {
        let encoding = u128::from(field.to_integer(value)).to_le_bytes();
        self.hasher.update(&encoding);
        self
    }

    /// A length-prefixed byte string.
    pub fn prefixed(&mut self, bytes: &[u8]) -> Result<&mut Self, ProtocolError> {
        self.usize(bytes.len())?;
        self.hasher.update(bytes);
        Ok(self)
    }

    /// The public commitment parameters, in the order every binding uses.
    pub fn commitment_params(&mut self, params: &PcsParams) -> Result<&mut Self, ProtocolError> {
        self.usize(params.m)?;
        self.usize(params.log_inv_rate)?;
        self.usize(params.log_batch_size)?;
        self.byte(profile_code(params.profile));
        self.byte(hash_code(params.merkle_hash));
        Ok(self)
    }

    /// The complete Ligerito prover configuration.
    pub fn ligerito_config(
        &mut self,
        config: &LigProverConfig,
    ) -> Result<&mut Self, ProtocolError> {
        for value in [
            config.recursive_steps,
            config.initial_log_msg_cols,
            config.initial_log_num_interleaved,
            config.initial_k,
        ] {
            self.usize(value)?;
        }
        for values in [
            config.log_inv_rates.as_slice(),
            config.recursive_log_msg_cols.as_slice(),
            config.recursive_ks.as_slice(),
            config.queries.as_slice(),
            config.grinding_bits.as_slice(),
            config.fold_grinding_bits.as_slice(),
            config.ood_samples.as_slice(),
        ] {
            self.usize(values.len())?;
            for &value in values {
                self.usize(value)?;
            }
        }
        self.byte(hash_code(config.merkle_hash));
        Ok(self)
    }

    pub fn finalize(self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }
}
