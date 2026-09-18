use field::*;

// Fp<2> deliberately retains u64 alignment. Exercise the fast batch kernel
// with addresses that cannot satisfy u128 alignment, plus odd lengths/tails.
#[repr(C, align(16))]
struct Offset {
    prefix: u64,
    values: [Fp<2>; 35],
}
#[test]
fn two_limb_batch_handles_word_alignment_and_empty_or_partial_slices() {
    assert_eq!(core::mem::size_of::<Fp<2>>(), 16);
    assert_eq!(core::mem::align_of::<Fp<2>>(), 8);
    let field = create_prime_field(Uint::from_words([u64::MAX - 158, u64::MAX]));
    let a = Offset {
        prefix: 0xaabbccdd,
        values: core::array::from_fn(|i| field.from_integer(&(u128::MAX - (i as u128)))),
    };
    let b = Offset {
        prefix: 0x11223344,
        values: core::array::from_fn(|i| field.from_integer(&((i as u128) * 7919))),
    };
    let sentinel = field.from_integer(&97u64);
    let mut out = Offset {
        prefix: 0x55aa55aa,
        values: [sentinel; 35],
    };
    assert_eq!(a.values.as_ptr() as usize % 16, 8);
    assert_eq!(b.values.as_ptr() as usize % 16, 8);
    assert_eq!(out.values.as_ptr() as usize % 16, 8);
    for len in [0, 1, 2, 3, 17, 34, 35] {
        out.values.fill(sentinel);
        field.batch_mul_into(&a.values[..len], &b.values[..len], &mut out.values[..len]);
        for i in 0..len {
            assert_eq!(out.values[i], field.mul(&a.values[i], &b.values[i]));
        }
        assert!(out.values[len..].iter().all(|v| *v == sentinel));
        assert_eq!(out.prefix, 0x55aa55aa);
    }
}
