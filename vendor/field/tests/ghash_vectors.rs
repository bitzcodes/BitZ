//! GHASH known-answer tests against NIST's published GCM vectors.
//!
//! Every other test in this crate compares one of our implementations against
//! another of ours. That cannot catch a field that is internally consistent but
//! wrong — the wrong reduction polynomial, or the wrong bit-to-coefficient
//! order — because both implementations would be wrong together. GHASH is
//! arithmetic in exactly this field, so a published GCM tag is an answer nobody
//! here computed.
//!
//! AES is only used to recover the hash subkey `H = E_K(0)` and the value the
//! tag is masked with. It lives in `GF(2^8)` under a different polynomial, so
//! it shares no arithmetic with what is under test.

use aes::Aes128;
use aes::cipher::{BlockCipherEncrypt, KeyInit, array::Array};
use field::Gf128;
use num_traits::{ConstOne, ConstZero};

/// NIST numbers the bits of a block with the *most* significant bit of byte 0
/// as the coefficient of `X^0`. This crate uses bit `i` of the little-endian
/// value as the coefficient of `X^i`. Reversing the bits within each byte
/// converts between the two, and is its own inverse.
fn from_nist(block: [u8; 16]) -> Gf128 {
    Gf128::from_bytes(block.map(u8::reverse_bits))
}

fn to_nist(a: Gf128) -> [u8; 16] {
    a.to_bytes().map(u8::reverse_bits)
}

/// `GHASH_H(A, C)`: absorb each 16-byte block of the additional data and then
/// the ciphertext, zero-padding the last of each, then a block holding the two
/// lengths in bits. Every absorb is `X = (X + block) * H`.
fn ghash(h: Gf128, aad: &[u8], ct: &[u8]) -> [u8; 16] {
    let mut x = Gf128::ZERO;
    let mut absorb = |bytes: &[u8]| {
        for chunk in bytes.chunks(16) {
            let mut block = [0u8; 16];
            block[..chunk.len()].copy_from_slice(chunk);
            x = (x + from_nist(block)) * h;
        }
    };
    absorb(aad);
    absorb(ct);

    let mut lengths = [0u8; 16];
    lengths[..8].copy_from_slice(&(aad.len() as u64 * 8).to_be_bytes());
    lengths[8..].copy_from_slice(&(ct.len() as u64 * 8).to_be_bytes());
    x = (x + from_nist(lengths)) * h;

    to_nist(x)
}

fn encrypt_block(key: &[u8], block: [u8; 16]) -> [u8; 16] {
    let cipher = Aes128::new(&Array::try_from(key).expect("128-bit key"));
    let mut buf = Array::from(block);
    cipher.encrypt_block(&mut buf);
    buf.into()
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex: {s:?}");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

#[derive(Default, Clone)]
struct Case {
    tag_bytes: usize,
    key: Vec<u8>,
    iv: Vec<u8>,
    aad: Vec<u8>,
    ct: Vec<u8>,
    tag: Vec<u8>,
}

/// The `.rsp` format is groups of `[Name = value]` parameters followed by
/// records of `Name = value` lines separated by blank lines.
fn parse(text: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut tag_bytes = 0;
    let mut cur = Case::default();
    let mut in_record = false;

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            if in_record {
                cases.push(std::mem::take(&mut cur));
                in_record = false;
            }
            continue;
        }
        if let Some(param) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            if let Some(bits) = param.strip_prefix("Taglen = ") {
                tag_bytes = bits.parse::<usize>().expect("Taglen") / 8;
            }
            continue;
        }
        let (name, value) = line.split_once(" = ").unwrap_or((line, ""));
        match name {
            "Count" => {
                cur = Case {
                    tag_bytes,
                    ..Default::default()
                };
                in_record = true;
            }
            "Key" => cur.key = unhex(value),
            "IV" => cur.iv = unhex(value),
            "AAD" => cur.aad = unhex(value),
            "CT" => cur.ct = unhex(value),
            "Tag" => cur.tag = unhex(value),
            _ => {}
        }
    }
    if in_record {
        cases.push(cur);
    }
    cases
}

/// For a 96-bit IV the counter block is `IV || 0^31 || 1`, so
/// `Tag = GHASH_H(A, C) + E_K(IV || 0^31 || 1)` with no counter arithmetic in
/// the way. Every case is checked to its own tag length; the shorter ones still
/// pin the leading bytes.
#[test]
fn ghash_matches_nist_gcm_vectors() {
    let text = include_str!("vectors/gcm_encrypt_128.rsp");
    let cases = parse(text);
    assert_eq!(cases.len(), 525, "vector file changed shape");

    for (i, case) in cases.iter().enumerate() {
        assert_eq!(case.iv.len(), 12, "case {i} is not a 96-bit IV");

        let h = from_nist(encrypt_block(&case.key, [0u8; 16]));

        let mut counter = [0u8; 16];
        counter[..12].copy_from_slice(&case.iv);
        counter[15] = 1;
        let mask = encrypt_block(&case.key, counter);

        let digest = ghash(h, &case.aad, &case.ct);
        let tag: Vec<u8> = digest
            .iter()
            .zip(mask)
            .map(|(d, m)| d ^ m)
            .take(case.tag_bytes)
            .collect();

        assert_eq!(
            tag,
            case.tag,
            "case {i}: aad {} bytes, ct {} bytes",
            case.aad.len(),
            case.ct.len()
        );
    }
}

/// The shim is what makes the vectors applicable at all, so it is pinned
/// independently of them: reversing bits within each byte is an involution, and
/// it must send NIST's `X^0` to ours.
#[test]
fn nist_bit_order_shim_round_trips() {
    for i in 0..16 {
        let mut block = [0u8; 16];
        block[i] = 0x80; // NIST's leading bit of byte i
        let expected = Gf128::from_bytes({
            let mut w = [0u8; 16];
            w[i] = 0x01;
            w
        });
        assert_eq!(from_nist(block), expected, "byte {i}");
        assert_eq!(to_nist(from_nist(block)), block, "byte {i}");
    }
    // NIST's first bit is the constant term, so it is our multiplicative unit.
    let mut one = [0u8; 16];
    one[0] = 0x80;
    assert_eq!(from_nist(one), Gf128::ONE);
}
