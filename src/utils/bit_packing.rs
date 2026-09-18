/// Interleave little-endian bits from two words, with `even` in even slots.
/// The fixed shifts and masks do not depend on the input values.
#[inline]
pub(crate) fn interleave_words(even: u64, odd: u64) -> [u64; 2] {
    [
        spread(even as u32) | (spread(odd as u32) << 1),
        spread((even >> 32) as u32) | (spread((odd >> 32) as u32) << 1),
    ]
}

#[inline]
fn spread(value: u32) -> u64 {
    let mut value = u64::from(value);
    value = (value | (value << 16)) & 0x0000_ffff_0000_ffff;
    value = (value | (value << 8)) & 0x00ff_00ff_00ff_00ff;
    value = (value | (value << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    value = (value | (value << 2)) & 0x3333_3333_3333_3333;
    (value | (value << 1)) & 0x5555_5555_5555_5555
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaving_matches_independent_bit_positions() {
        let mut cases = vec![
            (0, 0),
            (u64::MAX, u64::MAX),
            (0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555),
        ];
        for bit in 0..64 {
            cases.extend([(1 << bit, 0), (0, 1 << bit)]);
        }
        let mut state = 0x6a09_e667_f3bc_c909u64;
        for _ in 0..256 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            cases.push((state, state.rotate_left(29)));
        }
        for (even, odd) in cases {
            let mut expected = [0; 2];
            for bit in 0..64 {
                expected[bit / 32] |= ((even >> bit) & 1) << (2 * (bit % 32));
                expected[bit / 32] |= ((odd >> bit) & 1) << (2 * (bit % 32) + 1);
            }
            assert_eq!(interleave_words(even, odd), expected);
        }
    }
}
