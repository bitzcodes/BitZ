use field::*;
use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64;

fn polynomial_product(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len() + b.len()];
    for i in 0..a.len() * 64 {
        if a[i / 64] >> (i % 64) & 1 == 0 {
            continue;
        }
        for j in 0..b.len() * 64 {
            out[(i + j) / 64] ^= ((b[j / 64] >> (j % 64)) & 1) << ((i + j) % 64);
        }
    }
    out
}
fn polynomial_remainder(mut words: Vec<u64>, degree: usize, terms: &[usize]) -> [u64; 2] {
    for i in (degree..words.len() * 64).rev() {
        if words[i / 64] >> (i % 64) & 1 == 0 {
            continue;
        }
        words[i / 64] ^= 1 << (i % 64);
        for &term in terms {
            let bit = i - degree + term;
            words[bit / 64] ^= 1 << (bit % 64);
        }
    }
    [words[0], words[1]]
}

#[test]
fn aes_field_embedding_preserves_products() {
    for a in 0..=255u8 {
        let embedding = Gf128Ops.embed(&Gf8(a));
        for b in 0..=255u8 {
            let product = Gf8Ops.mul(&Gf8(a), &Gf8(b));
            assert_eq!(Gf8Ops.reduce(Gf8Ops.mul_wide(&Gf8(a), &Gf8(b))), product);
            assert_eq!(
                Gf128Ops.embed(&product),
                Gf128Ops.mul(&embedding, &Gf128Ops.embed(&Gf8(b)))
            );
        }
    }
    // A numeric integer has characteristic-two embedding; a polynomial byte
    // and the AES subfield embedding deliberately have different meanings.
    assert_eq!(Gf128Ops.from_integer(&2u64), Gf128Ops.zero());
    assert_ne!(Gf128Ops.embed(&Gf8(2)), Gf128Ops.from_integer(&2u64));
}

#[test]
fn b127_arithmetic_matches_polynomial_long_division() {
    let mut rng = Pcg64::seed_from_u64(1201);
    let mut lhs = vec![
        B127::ZERO,
        B127::ONE,
        B127::from_polynomial_words([u64::MAX, u64::MAX]),
    ];
    lhs.extend((0..128).map(|_| B127::from_polynomial_words([rng.next_u64(), rng.next_u64()])));
    let rhs: Vec<_> = lhs.iter().copied().rev().collect();
    let mut expected = B127::ZERO;
    for (&a, &b) in lhs.iter().zip(&rhs) {
        let product = B127Ops.mul(&a, &b);
        assert_eq!(
            *product.as_words(),
            polynomial_remainder(polynomial_product(a.as_words(), b.as_words()), 127, &[0, 1])
        );
        assert_eq!(B127Ops.square(&a), B127Ops.mul(&a, &a));
        let inverse = B127Ops.inverse_ct(&a);
        assert_eq!(inverse.validity().declassify(), a != B127::ZERO);
        assert_eq!(
            B127Ops.mul(&a, inverse.value()),
            if a == B127::ZERO {
                B127::ZERO
            } else {
                B127::ONE
            }
        );
        expected = B127Ops.add(&expected, &product);
    }
    assert_eq!(B127Ops.reduce(B127Ops.batch_mul_acc(&lhs, &rhs)), expected);
    let mut acc = B127Ops.batch_mul_acc(&lhs[..17], &rhs[..17]);
    acc.merge_assign(&B127Ops.batch_mul_acc(&lhs[17..], &rhs[17..]));
    assert_eq!(B127Ops.reduce(acc), expected);
    let mut bad = [0xff; 16];
    assert!(!B127Ops.decode_ct(&bad).unwrap().validity().declassify());
    bad[15] &= 0x7f;
    assert!(B127Ops.decode_ct(&bad).unwrap().validity().declassify());
}

#[test]
fn gf128_mixed_mac_and_zero_preserving_batch_inverse() {
    let mut rng = Pcg64::seed_from_u64(1202);
    let lhs: Vec<_> = (0..256)
        .map(|_| Gf128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let rhs: Vec<_> = (0..=255).map(Gf8).collect();
    let expected = lhs
        .iter()
        .zip(&rhs)
        .fold(Gf128Ops.zero(), |s, (a, b)| s + *a * Gf128Ops.embed(b));
    assert_eq!(
        Gf128Ops.reduce(Gf128Ops.batch_mul_acc(&lhs, &rhs)),
        expected
    );
    assert_eq!(
        Gf128Ops.reduce(Gf128Ops.batch_mul_acc(&rhs, &lhs)),
        expected
    );
    let mut visited = 0;
    let acc = Gf128Ops.batch_mul_acc_map(lhs.len(), |i| {
        assert_eq!(i, visited);
        visited += 1;
        (rhs[i], lhs[i])
    });
    assert_eq!(visited, lhs.len());
    assert_eq!(Gf128Ops.reduce(acc), expected);
    for (a, b) in rhs.iter().zip(&lhs) {
        assert_eq!(
            Gf128Ops.reduce(Gf128Ops.mul_wide(a, b)),
            Gf128Ops.mul(&Gf128Ops.embed(a), b)
        );
    }
    let mut with_zeros = lhs.clone();
    for i in (0..256).step_by(7) {
        with_zeros[i] = Gf128Ops.zero();
    }
    let inverses = Gf128Ops.batch_invert_or_zero_ct(&with_zeros);
    for (a, inv) in with_zeros.iter().zip(inverses) {
        assert_eq!(
            *a * inv,
            if a.ct_is_zero().declassify() {
                Gf128Ops.zero()
            } else {
                Gf128Ops.one()
            }
        );
    }
    assert!(Gf128Ops.batch_invert_or_zero_ct(&[]).is_empty());
    assert_eq!(
        Gf128Ops.batch_invert_or_zero_ct(&[Gf128Ops.zero(); 7]),
        [Gf128Ops.zero(); 7]
    );
    let all_bytes: Vec<_> = (0..=255).map(Gf8).collect();
    for (a, inverse) in all_bytes
        .iter()
        .zip(Gf8Ops.batch_invert_or_zero_ct(&all_bytes))
    {
        assert_eq!(*a * inverse, if a.0 == 0 { Gf8::ZERO } else { Gf8::ONE });
    }
}

#[test]
fn exact_polynomials_keep_domain_and_coefficients() {
    let mut rng = Pcg64::seed_from_u64(1203);
    for _ in 0..64 {
        let a = *F2Poly::<127, 2>::from_words_ct([rng.next_u64(), rng.next_u64() >> 1]).value();
        let b = *F2Poly::<65, 2>::from_words_ct([rng.next_u64(), rng.next_u64() & 1]).value();
        let product = F2PolyOps.mul_wide(&a, &b);
        let (lo, hi) = product.as_parts();
        let expected = polynomial_product(a.as_words(), b.as_words());
        assert_eq!([lo.as_slice(), hi.as_slice()].concat(), expected);
        assert_eq!(Gf128Ops.reduce(product).to_bytes(), {
            let r = polynomial_remainder(expected, 128, &[0, 1, 2, 7]);
            Gf128::new(r[0], r[1]).to_bytes()
        });
        assert_eq!(
            Gf128Ops.reduce(F2PolyOps.batch_mul_acc(&[a, a], &[b, b])),
            Gf128Ops.zero()
        );
    }
    assert!(
        !F2Poly::<65, 2>::from_words_ct([0, 2])
            .validity()
            .declassify()
    );
}

#[test]
fn prepared_multiplication_is_consistent_and_b127_square_chains_match_polynomials() {
    let mut rng = Pcg64::seed_from_u64(1411);
    let mut scalars = vec![
        Gf128::new(0, 0),
        Gf128::new(1, 0),
        Gf128::new(u64::MAX, u64::MAX),
    ];
    scalars.extend((0..17).map(|_| Gf128::new(rng.next_u64(), rng.next_u64())));
    let values: Vec<_> = (0..35)
        .map(|_| Gf128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    for scalar in scalars {
        let prepared = PreparedGf128Mul::new(scalar);
        let mut out = vec![Gf128Ops.zero(); values.len()];
        prepared.mul_into(&values, &mut out);
        for (input, actual) in values.iter().zip(out) {
            assert_eq!(prepared.mul(input), actual);
        }
    }
    for input in values {
        let value = B127::from_polynomial_words([input.lo, input.hi]);
        for n in [0, 1, 2, 3, 6, 12, 24, 48, 96] {
            let mut expected = *value.as_words();
            for _ in 0..n {
                expected =
                    polynomial_remainder(polynomial_product(&expected, &expected), 127, &[0, 1]);
            }
            assert_eq!(*value.square_n(n).as_words(), expected);
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn gf8_serde_encodes_a_byte() {
    assert_eq!(
        serde_json::to_value(Gf8(197)).unwrap(),
        serde_json::json!(197)
    );
}
