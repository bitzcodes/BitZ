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
fn random(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *seed;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn qualify<const D: usize, const N: usize>(words: [u64; D]) {
    let modulus = Uint::from_words(words);
    let prepared = PreparedDivisor::new(modulus).unwrap();
    let m = big(&words);
    let mut seed = 183791;
    let mut values = vec![Uint::<N>::ZERO, Uint::ONE, Uint::MAX];
    for _ in 0..64 {
        values.push(Uint::from_words(core::array::from_fn(|_| {
            random(&mut seed)
        })));
    }
    for x in values {
        let (q, r) = prepared.div_rem_ct(&x);
        let n = big(x.as_words());
        assert_eq!(
            big(q.as_words()),
            &n / &m,
            "quotient, D={D}, N={N}, m={words:x?}"
        );
        assert_eq!(
            big(r.as_words()),
            &n % &m,
            "remainder, D={D}, N={N}, m={words:x?}"
        );
        if m > BigUint::from(1u64) {
            let ring = ModRingCtx::new(modulus).unwrap();
            assert_eq!(ring.to_integer(&ring.from_integer(&x)), r);
        }
    }
    let x = Uint::<N>::MAX;
    let product = IntegerOps.mul_wide(&x, &x);
    let (q, r) = prepared.div_rem_product_ct(&product);
    let n = big(x.as_words()).pow(2);
    let (lo, hi) = q.as_parts();
    let mut qwords = lo.to_vec();
    qwords.extend_from_slice(hi);
    assert_eq!(big(&qwords), &n / &m);
    assert_eq!(big(r.as_words()), &n % &m);
}

#[test]
fn prepared_division_all_widths_padding_and_power_boundaries() {
    for m in [
        1,
        2,
        3,
        17,
        1 << 32,
        (1 << 32) + 1,
        1 << 63,
        u64::MAX - 58,
        u64::MAX,
    ] {
        qualify::<1, 5>([m]);
        qualify::<3, 1>([m, 0, 0]);
        qualify::<3, 7>([m, 0, 0]);
    }
    for m in [[0, 1], [1, 1], [u64::MAX, 1], [0, 1 << 63], [u64::MAX; 2]] {
        qualify::<2, 1>(m);
        qualify::<2, 5>(m);
        qualify::<4, 9>([m[0], m[1], 0, 0]);
    }
    qualify::<4, 9>([u64::MAX, 0xffffffff, 0, 0xffffffff00000001]);
    qualify::<6, 64>([0x123456789, 1, 2, 3, 4, 1 << 31]);
    qualify::<32, 64>([u64::MAX; 32]);
    let mut seed = 473817;
    for _ in 0..32 {
        qualify::<3, 8>(core::array::from_fn(|_| random(&mut seed)));
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[test]
fn prepared_odd_inverse_units_nonunits_and_padded_widths() {
    for modulus in (3..100).step_by(2) {
        let one = PreparedOddInverse::new(Uint::from_words([modulus])).unwrap();
        let padded = PreparedOddInverse::new(Uint::from_words([modulus, 0])).unwrap();
        for value in 0..2 * modulus {
            let inverse = one.inverse_ct(&Uint::from_words([value]));
            let other = padded.inverse_ct(&Uint::from_words([value, 0]));
            let valid = gcd(value, modulus) == 1;
            assert_eq!(inverse.validity().declassify(), valid);
            assert_eq!(other.validity().declassify(), valid);
            assert_eq!(
                other.value().as_words(),
                &[inverse.value().as_words()[0], 0]
            );
            if valid {
                assert_eq!((value * inverse.value().as_words()[0]) % modulus, 1);
            } else {
                assert_eq!(inverse.value(), &Uint::ZERO);
            }
        }
    }
    let words = [u64::MAX, 0xffffffff, 0, 0xffffffff00000001];
    let modulus = big(&words);
    let inverse = PreparedOddInverse::new(Uint::from_words(words)).unwrap();
    let mut seed = 378123;
    for _ in 0..64 {
        let value = Uint::from_words(core::array::from_fn(|_| random(&mut seed)));
        let got = inverse.inverse_ct(&value);
        assert!(got.validity().declassify());
        assert_eq!(
            big(got.value().as_words()),
            big(value.as_words()).modpow(&(&modulus - BigUint::from(2u64)), &modulus)
        );
    }
}

#[test]
fn prepared_odd_inverse_full_width_composite_and_unreduced_inputs() {
    // A wide odd multiple of three, so this exercises nonunits as well as
    // units. BigUint Euclid and multiplication are independent of the backend.
    let words = [15, 0, 0, 3 << 60];
    let modulus = Uint::from_words(words);
    let prepared = PreparedOddInverse::new(modulus).unwrap();
    let m = big(&words);
    let zero = BigUint::from(0u64);
    let one = BigUint::from(1u64);
    let mut values = vec![Uint::ZERO, Uint::ONE, Uint::from_u64(3), modulus, Uint::MAX];
    let mut seed = 0x6721239;
    values.extend((0..128).map(|_| Uint::from_words(core::array::from_fn(|_| random(&mut seed)))));
    for input in values {
        let n = big(input.as_words());
        let (mut a, mut b) = (m.clone(), n.clone());
        while b != zero {
            (a, b) = (b.clone(), a % b);
        }
        let result = prepared.inverse_ct(&input);
        assert_eq!(result.validity().declassify(), a == one);
        let inverse = big(result.value().as_words());
        assert!(inverse < m);
        if a == one {
            assert_eq!((n * inverse) % &m, one);
        } else {
            assert_eq!(inverse, zero);
        }
    }
}
