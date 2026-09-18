use field::{Uint, Z};

#[test]
fn signed_native_construction_is_exact_or_rejected() {
    for value in [i64::MIN as i128, -1, 0, 1, i64::MAX as i128] {
        assert_eq!(Z::<1>::from(value).as_words(), &[value as u64]);
    }
    for value in [
        i128::MIN,
        i64::MIN as i128 - 1,
        i64::MAX as i128 + 1,
        i128::MAX,
    ] {
        assert!(std::panic::catch_unwind(|| Z::<1>::from(value)).is_err());
        assert!(
            !Z::<2>::from(value)
                .checked_resize_ct::<1>()
                .validity()
                .declassify()
        );
    }
    assert!(std::panic::catch_unwind(|| Z::<1>::from(u64::MAX)).is_err());
    assert!(
        !Uint::<1>::from(u64::MAX)
            .checked_to_signed_ct()
            .validity()
            .declassify()
    );
    assert_eq!(Z::<2>::from(u64::MAX).as_words(), &[u64::MAX, 0]);
    assert_eq!(
        Z::<1>::from_twos_complement_words([u64::MAX]),
        Z::<1>::from(-1i64)
    );
}
