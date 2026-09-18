use field::*;
use num_bigint::BigUint;
use num_traits::ToPrimitive;

fn big<const N: usize>(value: &Uint<N>) -> BigUint {
    BigUint::from_bytes_le(
        &value
            .as_words()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

#[test]
fn canonical_u128_operations_and_bounded_redc_match_integer_oracle() {
    for q in [17u128, (1u128 << 100) - 15, u128::MAX - 158] {
        let field = FpCtx::from_prime_u128(q);
        let modulus = BigUint::from(q);
        let radix = BigUint::from(1u64) << 128usize;
        let inverse = radix.modpow(&(&modulus - 2u64), &modulus);
        for a in [0, 1, q - 1, q, u128::MAX] {
            for b in [0, 1, q - 1, u128::MAX] {
                let aa = BigUint::from(a);
                let bb = BigUint::from(b);
                assert_eq!(
                    field.mul_u128(a, b),
                    ((&aa * &bb) % &modulus).to_u128().unwrap()
                );
                assert_eq!(
                    field.add_u128(a, b),
                    ((&aa + &bb) % &modulus).to_u128().unwrap()
                );
                assert_eq!(
                    field.sub_u128(a, b),
                    ((aa % &modulus + &modulus - bb % &modulus) % &modulus)
                        .to_u128()
                        .unwrap()
                );
                let high = a % q;
                let input = Uint::from_words([
                    b as u64,
                    (b >> 64) as u64,
                    high as u64,
                    (high >> 64) as u64,
                ]);
                assert_eq!(
                    big(&field.reduce_montgomery_bounded(&input)),
                    (big(&input) * &inverse) % &modulus
                );
            }
        }
    }
}

fn mixed_pair<const N: usize>(field: &FpCtx<2>) {
    let values = [Uint::<N>::from_words([u64::MAX; N]), Uint::from_u64(1)];
    let weights = [field.from_integer(&u128::MAX), field.from_integer(&7u64)];
    let q = big(field.modulus());
    let expected = (big(&field.to_integer(&weights[0])) * big(&values[0])
        + big(&field.to_integer(&weights[1])) * big(&values[1]))
        % q;
    assert_eq!(
        big(&field.weighted_pair_to_integer(&weights, &values)),
        expected
    );
    assert_eq!(
        big(&field.to_integer(&field.weighted_pair(&weights, &values))),
        expected
    );
}

#[test]
fn weighted_pairs_cover_short_full_and_wide_integer_domains() {
    for q in [17, (1u128 << 100) - 15, u128::MAX - 158] {
        let field = FpCtx::from_prime_u128(q);
        mixed_pair::<1>(&field);
        mixed_pair::<2>(&field);
        mixed_pair::<4>(&field);
        mixed_pair::<32>(&field);
    }
}

#[test]
fn incremental_accumulators_and_merges_preserve_each_scale() {
    let field = FpCtx::from_prime_u128(u128::MAX - 158);
    let q = big(field.modulus());
    let a = field.from_integer(&(u128::MAX - 160));
    let b = field.from_integer(&7u64);
    let input = Uint::<4>::from_words([u64::MAX; 4]);
    let mut products = FpProductAcc::<2>::zero();
    let mut linear = FpLinearAcc::<2, 4>::zero();
    let mut signed = FpSignedLinearAcc::<2, 4>::zero();
    let negative = Z::from_twos_complement_words([0, 0, 0, 1u64 << 63]);
    for _ in 0..37 {
        products.accumulate(&a, &b);
        linear.accumulate(&a, &input);
        signed.accumulate(&a, &negative);
    }
    products += products;
    linear += linear;
    signed += signed;
    let coefficient = big(&field.to_integer(&a)) * 74u64;
    assert_eq!(
        big(&field.to_integer(&field.reduce(products))),
        (&coefficient * 7u64) % &q
    );
    assert_eq!(
        big(&field.to_integer(&field.reduce(linear))),
        (&coefficient * big(&input)) % &q
    );
    let magnitude = (&coefficient * big(&negative.unsigned_abs())) % &q;
    assert_eq!(
        big(&field.to_integer(&field.reduce(signed))),
        (&q - magnitude) % &q
    );
}
