use field::*;
use num_bigint::BigUint;

fn big(words: &[u64]) -> BigUint {
    BigUint::from_bytes_le(
        &words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

#[test]
fn composite_and_even_moduli_support_exact_units() {
    assert!(ModRingCtx::new(Uint::from_words([1])).is_err());
    for modulus in 2u64..40 {
        let ring = ModRingCtx::new(Uint::from_words([modulus])).unwrap();
        for a in 0..modulus {
            let value = ring.from_integer(&a);
            let inverse = ring.inverse_ct(&value);
            let expected = (0..modulus).find(|b| a * b % modulus == 1);
            assert_eq!(
                inverse.validity().declassify(),
                expected.is_some(),
                "{a} mod {modulus}"
            );
            assert_eq!(
                ring.to_integer(inverse.value()).as_words()[0],
                expected.unwrap_or(0)
            );
            for b in 0..modulus {
                let other = ring.from_integer(&b);
                assert_eq!(
                    ring.to_integer(&ring.add(&value, &other)).as_words()[0],
                    (a + b) % modulus
                );
                assert_eq!(
                    ring.to_integer(&ring.sub(&value, &other)).as_words()[0],
                    (modulus + a - b) % modulus
                );
                assert_eq!(
                    ring.to_integer(&ring.mul(&value, &other)).as_words()[0],
                    a * b % modulus
                );
            }
        }
    }
    // Full-width and padded moduli exercise the carry out of restoring division.
    for words in [[u64::MAX - 1, u64::MAX], [18, 0], [0, 1]] {
        let modulus = Uint::from_words(words);
        let ring = ModRingCtx::new(modulus).unwrap();
        let p = big(&words);
        for a in [Uint::ZERO, Uint::ONE, Uint::MAX, Uint::from_words([37, 1])] {
            let value = ring.from_integer(&a);
            assert_eq!(
                big(ring.to_integer(&value).as_words()),
                big(a.as_words()) % &p
            );
            assert_eq!(
                big(ring.to_integer(&ring.square(&value)).as_words()),
                big(a.as_words()).pow(2) % &p
            );
        }
        let minus_one = ring.neg(&ring.one());
        let inverse = ring.inverse_ct(&minus_one);
        assert!(inverse.validity().declassify());
        assert_eq!(*inverse.value(), minus_one);
    }
}

#[test]
fn prepared_division_and_products_match_bigints() {
    assert!(PreparedDivisor::new(Uint::<1>::ZERO).is_err());
    let lhs = [
        Uint::MAX,
        Uint::from_words([0, 1]),
        Uint::from_words([7, 0]),
    ];
    let rhs = [
        Uint::MAX,
        Uint::from_words([u64::MAX, 4]),
        Uint::from_words([11, 0]),
    ];
    let mut out = [UintProduct::ZERO; 3];
    let mut prepared = PreparedProducts::new(
        &lhs,
        &rhs,
        &mut out,
        PublicProductBounds {
            lhs_limbs: 2,
            rhs_limbs: 2,
        },
    )
    .unwrap();
    prepared.execute();
    for (i, value) in prepared.outputs().iter().enumerate() {
        let expected = big(lhs[i].as_words()) * big(rhs[i].as_words());
        let (lo, hi) = value.as_parts();
        assert_eq!(big(&[lo.as_slice(), hi.as_slice()].concat()), expected);
        for divisor in [
            Uint::ONE,
            Uint::MAX,
            Uint::from_words([0, 1]),
            Uint::from_words([17, 0]),
        ] {
            let prepared = PreparedDivisor::new(divisor).unwrap();
            let (q, r) = prepared.div_rem_product_ct(value);
            let (lo, hi) = q.as_parts();
            assert_eq!(
                big(&[lo.as_slice(), hi.as_slice()].concat()),
                &expected / big(divisor.as_words())
            );
            assert_eq!(big(r.as_words()), &expected % big(divisor.as_words()));
        }
    }
    let short = [Uint::from_words([u64::MAX, 0])];
    let mut short_out = [UintProduct::ZERO];
    let mut prepared = PreparedProducts::new(
        &short,
        &short,
        &mut short_out,
        PublicProductBounds {
            lhs_limbs: 1,
            rhs_limbs: 1,
        },
    )
    .unwrap();
    prepared.execute();
    assert_eq!(
        prepared.outputs()[0],
        IntegerOps.mul_wide(&short[0], &short[0])
    );
    assert!(
        PreparedProducts::new(
            &short,
            &short[..0],
            &mut short_out,
            PublicProductBounds {
                lhs_limbs: 1,
                rhs_limbs: 1
            }
        )
        .is_err()
    );
}

#[test]
fn prepared_inverse_and_fixed_base_ladder() {
    assert!(PreparedOddInverse::new(Uint::from_words([18])).is_err());
    let inverse = PreparedOddInverse::new(Uint::from_words([15])).unwrap();
    assert!(
        !inverse
            .inverse_ct(&Uint::from_words([5]))
            .validity()
            .declassify()
    );
    assert_eq!(
        inverse
            .inverse_ct(&Uint::from_words([19]))
            .value()
            .as_words(),
        &[4]
    );
    let field = create_prime_field(Uint::from_words([17]));
    let base = field.from_integer(&3u64);
    let prepared = field::preparation::FixedBasePow::<_, 2>::new(&field, base);
    for exponent in [Uint::ZERO, Uint::ONE, Uint::MAX, Uint::from_words([0, 1])] {
        assert_eq!(prepared.pow_ct(&exponent), field.pow_ct(&base, &exponent));
    }
    let projection = PreparedIntegerProjection::new(&field);
    let values = [0u128, u128::MAX, 1 << 127];
    let mut out = [field.zero(); 3];
    projection.project_into(&values, &mut out);
    for (a, b) in values.iter().zip(out) {
        assert_eq!(field.from_integer(a), b);
    }
}
