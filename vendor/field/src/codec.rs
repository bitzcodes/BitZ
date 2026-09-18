//! Spongefish codecs for the field types.
//!
//! Wire formats, fixed here and nowhere else:
//!
//! - `Gf128`: [`Gf128::to_bytes`], 16 bytes. Total in both directions — every
//!   16-byte string is an element — so `Decoding` squeezes 16 bytes and gets a
//!   uniform sample through the same bijection.
//! - Static prime fields: canonical little-endian limbs. Prime sampling uses
//!   the provider's bounded rejection sampler, not a biased `Decoding` cast.
//!
//! `NargSerialize` comes from spongefish's blanket impl over `Encoding`.

use spongefish::{
    ByteArray, Decoding, Encoding, NargDeserialize, VerificationError, VerificationResult,
};

use crate::{CanonicalCodec, Gf128, PrimeSpec, StaticFp, StaticFpOps};

impl Encoding<[u8]> for Gf128 {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.to_bytes()
    }
}

impl Decoding<[u8]> for Gf128 {
    type Repr = ByteArray<16>;

    fn decode(buf: Self::Repr) -> Self {
        Self::from_bytes(*buf.as_ref())
    }
}

impl NargDeserialize for Gf128 {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        <[u8; 16]>::deserialize_from_narg(buf).map(Self::from_bytes)
    }
}

impl<P: PrimeSpec<L>, const L: usize> Encoding<[u8]> for StaticFp<P, L> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut bytes = vec![0; L * 8];
        StaticFpOps::<P, L>::new().encode_into(self, &mut bytes);
        bytes
    }
}

impl<P: PrimeSpec<L>, const L: usize> NargDeserialize for StaticFp<P, L> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        // Stage the cursor: the contract requires `buf` untouched on failure,
        // and the range check can still fail after the read succeeds.
        let (bytes, rest) = buf.split_at_checked(L * 8).ok_or(VerificationError)?;
        let value = StaticFpOps::<P, L>::new()
            .decode_public(bytes)
            .map_err(|_| VerificationError)?;
        *buf = rest;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{ConstOne, ConstZero};
    use spongefish::NargSerialize;

    use super::*;
    use crate::{IntegerEmbedding, Q100, Q100Element, Q100Field};

    fn f128_cases() -> [Gf128; 4] {
        [
            Gf128::ZERO,
            Gf128::ONE,
            Gf128::GENERATOR,
            Gf128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
        ]
    }

    #[test]
    fn f128_encoding_is_to_bytes() {
        for a in f128_cases() {
            assert_eq!(a.encode().as_ref(), a.to_bytes());
            assert_eq!(a.serialize_into_new_narg().as_ref(), a.to_bytes());
        }
    }

    #[test]
    fn f128_decoding_is_from_bytes() {
        let mut repr = <Gf128 as Decoding>::Repr::default();
        repr.as_mut().copy_from_slice(&[0xa5; 16]);
        assert_eq!(Gf128::decode(repr), Gf128::from_bytes([0xa5; 16]));
    }

    #[test]
    fn f128_narg_round_trip() {
        for a in f128_cases() {
            let mut narg = Vec::new();
            a.serialize_into_narg(&mut narg);
            narg.extend_from_slice(b"tail");

            let mut buf = narg.as_slice();
            assert_eq!(Gf128::deserialize_from_narg(&mut buf).unwrap(), a);
            assert_eq!(buf, b"tail");
        }
    }

    #[test]
    fn f128_narg_rejects_short_input() {
        let narg = [0u8; 15];
        let mut buf = narg.as_slice();
        assert!(Gf128::deserialize_from_narg(&mut buf).is_err());
        assert_eq!(buf, narg);
    }

    #[test]
    fn fq_encoding_is_le_value() {
        for v in [0u128, 1, 12345, Q100 - 1] {
            let a = Q100Field::new().from_integer(&v);
            assert_eq!(a.encode().as_ref(), v.to_le_bytes());
        }
    }

    #[test]
    fn fq_narg_round_trip() {
        for v in [0u128, 1, 12345, Q100 - 1] {
            let a = Q100Field::new().from_integer(&v);
            let mut narg = Vec::new();
            a.serialize_into_narg(&mut narg);

            let mut buf = narg.as_slice();
            assert_eq!(Q100Element::deserialize_from_narg(&mut buf).unwrap(), a);
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn fq_narg_rejects_non_canonical() {
        for v in [Q100, Q100 + 1, u128::MAX] {
            let narg = v.to_le_bytes();
            let mut buf = narg.as_slice();
            assert!(Q100Element::deserialize_from_narg(&mut buf).is_err());
            assert_eq!(buf, narg, "cursor must not move on failure");
        }
    }
}
