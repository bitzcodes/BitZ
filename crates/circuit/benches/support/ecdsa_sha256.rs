use circuit::ecdsa_sha256::VERIFY_2KB_INPUT_BITS;
use circuit::sha256::SHA256_2KB_MESSAGE_BITS;
use num_bigint::BigUint;

use super::{p256, sha256};

pub fn valid_input() -> Box<[bool; VERIFY_2KB_INPUT_BITS]> {
    let message = sha256::message_2kb();
    let digest = BigUint::parse_bytes(
        b"10fc3c51a152e90e5b90319b601d92ccf37290ef53c35ff92507687d8a911a08",
        16,
    )
    .unwrap();
    let p256_inputs = p256::valid_aux_input(&digest);
    let bits: Box<[bool]> = (0..VERIFY_2KB_INPUT_BITS)
        .map(|index| {
            if index < SHA256_2KB_MESSAGE_BITS {
                message[index]
            } else {
                p256_inputs[index - SHA256_2KB_MESSAGE_BITS]
            }
        })
        .collect();
    bits.try_into().unwrap()
}
