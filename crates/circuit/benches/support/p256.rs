use circuit::p256::VERIFY_DIGEST_INPUT_BITS;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

pub const AUX_INPUT_BITS: usize = 6 * 256;

fn from_hex(value: &[u8]) -> BigUint {
    BigUint::parse_bytes(value, 16).unwrap()
}

fn scalar_modulus() -> BigUint {
    from_hex(b"ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551")
}

fn inverse(value: &BigUint, modulus: &BigUint) -> BigUint {
    let mut t = BigInt::zero();
    let mut new_t = BigInt::one();
    let mut r = BigInt::from(modulus.clone());
    let mut new_r = BigInt::from(value.clone());
    while !new_r.is_zero() {
        let quotient = &r / &new_r;
        (t, new_t) = (new_t.clone(), t - &quotient * new_t);
        (r, new_r) = (new_r.clone(), r - quotient * new_r);
    }
    assert_eq!(r, BigInt::one());
    let modulus = BigInt::from(modulus.clone());
    let mut t = t % &modulus;
    if t.sign() == Sign::Minus {
        t += modulus;
    }
    t.to_biguint().unwrap()
}

/// P-256 inputs for d = k = 1: Q = G, r = G.x, and s = z + r*d.
pub fn valid_aux_input(digest: &BigUint) -> Box<[bool; AUX_INPUT_BITS]> {
    let modulus = scalar_modulus();
    let gx = from_hex(b"6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");
    let gy = from_hex(b"4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5");
    let s = (digest + &gx) % &modulus;
    let values = [
        gx.clone(),
        gy,
        gx.clone(),
        s.clone(),
        inverse(&gx, &modulus),
        inverse(&s, &modulus),
    ];
    let bits: Box<[bool]> = (0..AUX_INPUT_BITS)
        .map(|index| values[index / 256].bit((index % 256) as u64))
        .collect();
    bits.try_into().unwrap()
}

/// A small valid ECDSA instance with z = 1.
pub fn valid_input() -> Box<[bool; VERIFY_DIGEST_INPUT_BITS]> {
    let digest = BigUint::one();
    let aux = valid_aux_input(&digest);
    let bits: Box<[bool]> = (0..VERIFY_DIGEST_INPUT_BITS)
        .map(|index| {
            if index < 256 {
                digest.bit(index as u64)
            } else {
                aux[index - 256]
            }
        })
        .collect();
    bits.try_into().unwrap()
}
