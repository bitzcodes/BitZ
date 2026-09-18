use circuit::sha256::SHA256_2KB_MESSAGE_BITS;

pub fn message_2kb() -> Box<[bool; SHA256_2KB_MESSAGE_BITS]> {
    let bits: Box<[bool]> = (0..SHA256_2KB_MESSAGE_BITS)
        .map(|bit| {
            let byte = (bit / 8) as u8;
            byte & (1 << (7 - bit % 8)) != 0
        })
        .collect();
    bits.try_into()
        .unwrap_or_else(|_| unreachable!("message length is fixed"))
}
