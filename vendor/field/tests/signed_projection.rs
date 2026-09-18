use field::*;
use num_bigint::{BigInt, BigUint};

fn bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn check<const L: usize>(modulus: Uint<L>) {
    let field = create_prime_field(modulus);
    let prepared = PreparedSignedProjection::new(field.clone(), 18);
    let q = BigInt::from(BigUint::from_bytes_le(&bytes(modulus.as_words())));
    let mut seed = 0x123456789abcdef0u64;
    for width in 0..=18 {
        let mut cases = vec![vec![0; width], vec![u64::MAX; width]];
        if width > 0 {
            let mut min = vec![0; width];
            min[width - 1] = 1 << 63;
            cases.push(min);
        }
        for _ in 0..12 {
            cases.push(
                (0..width)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        seed
                    })
                    .collect(),
            );
        }
        for words in cases {
            let x = BigInt::from_signed_bytes_le(&bytes(&words));
            let expected = ((x % &q) + &q) % &q;
            let canonical = prepared.project_canonical(&words);
            assert_eq!(
                BigInt::from(BigUint::from_bytes_le(&bytes(canonical.as_words()))),
                expected
            );
            assert_eq!(field.to_integer(&prepared.project(&words)), canonical);
        }
    }
    let inputs = [Z::<9>::MIN, Z::MAX, Z::ZERO, -Z::ONE];
    let mut outputs = [field.zero(); 4];
    prepared.project_into(&inputs, &mut outputs);
    for (input, output) in inputs.iter().zip(outputs) {
        assert_eq!(output, field.from_integer(input));
    }
}

#[test]
fn signed_horner_full_padded_and_small_prime_widths() {
    check(Uint::from_words([101]));
    check(Uint::from_words([101, 0]));
    check(Uint::from_words([101, 0, 0, 0]));
    check(Uint::from_words([u64::MAX - 14, (1 << 36) - 1]));
    check(Uint::from_words([u64::MAX - 158, u64::MAX]));
    check(Uint::from_words([
        u64::MAX,
        0xffffffff,
        0,
        0xffffffff00000001,
    ]));
}

#[test]
fn signed_dot_carries_and_sign_corrections_over_long_inputs() {
    for modulus in [
        Uint::from_words([101, 0]),
        Uint::from_words([u64::MAX - 14, (1 << 36) - 1]),
        Uint::from_words([u64::MAX - 158, u64::MAX]),
    ] {
        let field = create_prime_field(modulus);
        let prepared = PreparedSignedProjection::new(field.clone(), 256);
        let q = BigInt::from(BigUint::from_bytes_le(&bytes(modulus.as_words())));
        for width in [31, 32, 64, 127, 128, 255, 256] {
            let mut min = vec![0; width];
            min[width - 1] = 1 << 63;
            let mut max = vec![u64::MAX; width];
            max[width - 1] >>= 1;
            for words in [vec![0; width], vec![u64::MAX; width], min, max] {
                let x = BigInt::from_signed_bytes_le(&bytes(&words));
                let expected = ((x % &q) + &q) % &q;
                let actual = field.to_integer(&prepared.project(&words));
                assert_eq!(
                    BigInt::from(BigUint::from_bytes_le(&bytes(actual.as_words()))),
                    expected
                );
                assert_eq!(actual, prepared.project_canonical(&words));
            }
        }
    }
}

#[test]
#[should_panic(expected = "integer exceeds prepared width")]
fn projection_rejects_a_larger_declared_width() {
    PreparedSignedProjection::new(create_prime_field(Uint::from_words([101])), 1).project(&[0, 0]);
}
