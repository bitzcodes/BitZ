//! Fixed-width exact MAC schedules retained from BitZ production.

#[inline(always)]
fn mac(accumulator: u64, lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
    let value = (lhs as u128) * (rhs as u128) + (accumulator as u128) + (carry as u128);
    (value as u64, (value >> 64) as u64)
}

#[inline(always)]
pub(super) fn multiply_accumulate_2x2(accumulator: &mut [u64; 5], lhs: [u64; 2], rhs: [u64; 2]) {
    // Finish both schoolbook rows before propagating into the two high limbs.
    // This avoids walking the high accumulator once per row.
    let (word_0, carry_00) = mac(accumulator[0], lhs[0], rhs[0], 0);
    let (row_0_word_1, row_0_carry) = mac(accumulator[1], lhs[0], rhs[1], carry_00);
    let (word_1, carry_10) = mac(row_0_word_1, lhs[1], rhs[0], 0);

    let column_2 = (accumulator[2] as u128) + (row_0_carry as u128);
    let (word_2, carry_11) = mac(column_2 as u64, lhs[1], rhs[1], carry_10);
    let column_3 = (accumulator[3] as u128) + (carry_11 as u128) + (column_2 >> 64);
    let column_4 = (accumulator[4] as u128) + (column_3 >> 64);

    accumulator[0] = word_0;
    accumulator[1] = word_1;
    accumulator[2] = word_2;
    accumulator[3] = column_3 as u64;
    accumulator[4] = column_4 as u64;
    debug_assert_eq!(column_4 >> 64, 0, "five-limb delayed accumulator overflow");
}

#[inline(always)]
pub(super) fn multiply_accumulate_2x1(accumulator: &mut [u64; 5], lhs: [u64; 2], rhs: u64) {
    let product_0 = (lhs[0] as u128) * (rhs as u128);
    let product_1 = (lhs[1] as u128) * (rhs as u128);

    let column_0 = (accumulator[0] as u128) + ((product_0 as u64) as u128);
    let column_1 = (accumulator[1] as u128)
        + (product_0 >> 64)
        + ((product_1 as u64) as u128)
        + (column_0 >> 64);
    let column_2 = (accumulator[2] as u128) + (product_1 >> 64) + (column_1 >> 64);
    let column_3 = (accumulator[3] as u128) + (column_2 >> 64);

    accumulator[0] = column_0 as u64;
    accumulator[1] = column_1 as u64;
    accumulator[2] = column_2 as u64;
    accumulator[3] = column_3 as u64;
    // Fewer than 2^64 products of a 128-bit value and a u64 fit in four
    // limbs, so the fifth limb remains unused for a valid linear accumulator.
    debug_assert_eq!(
        ((column_3 >> 64) as u64) | accumulator[4],
        0,
        "five-limb delayed accumulator overflow"
    );
}
