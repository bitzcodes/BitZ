use field::*;
use num_bigint::{BigInt, BigUint};

struct Rng(u64);
impl Rng {
    fn word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    fn uint<const L: usize>(&mut self) -> Uint<L> {
        Uint::from_words(core::array::from_fn(|_| self.word()))
    }
}
fn big(words: &[u64]) -> BigUint {
    BigUint::from_bytes_le(
        &words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn signed(words: &[u64]) -> BigInt {
    BigInt::from_signed_bytes_le(
        &words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn product<const A: usize, const B: usize>(value: UintProduct<A, B>) -> BigUint {
    let (low, high) = value.as_parts();
    big(&[low.as_slice(), high.as_slice()].concat())
}
fn acc<const A: usize, const B: usize>(value: UintAccumulator<A, B>) -> BigUint {
    let (low, high, head) = value.as_parts();
    big(&[low.as_slice(), high.as_slice(), &[head]].concat())
}
fn signed_acc<const A: usize, const B: usize>(value: ZAccumulator<A, B>) -> BigInt {
    let (low, high, head) = value.as_twos_complement_parts();
    signed(&[low.as_slice(), high.as_slice(), &[head]].concat())
}

#[test]
fn native_checked_arithmetic_matches_rust() {
    let mut rng = Rng(0x5297);
    for _ in 0..2048 {
        let a = rng.word();
        let b = rng.word();
        let ua = Uint::from_words([a]);
        let ub = Uint::from_words([b]);
        for (actual, expected) in [
            (ua.checked_add_ct(&ub), a.checked_add(b)),
            (ua.checked_sub_ct(&ub), a.checked_sub(b)),
            (ua.checked_mul_ct(&ub), a.checked_mul(b)),
        ] {
            assert_eq!(actual.validity().declassify(), expected.is_some());
            if let Some(expected) = expected {
                assert_eq!(actual.value().as_words()[0], expected);
            }
        }
        let za = Z::from_twos_complement_words([a]);
        let zb = Z::from_twos_complement_words([b]);
        for (actual, expected) in [
            (za.checked_add_ct(&zb), (a as i64).checked_add(b as i64)),
            (za.checked_sub_ct(&zb), (a as i64).checked_sub(b as i64)),
            (za.checked_mul_ct(&zb), (a as i64).checked_mul(b as i64)),
        ] {
            assert_eq!(actual.validity().declassify(), expected.is_some());
            if let Some(expected) = expected {
                assert_eq!(actual.value().as_words()[0] as i64, expected);
            }
        }
        assert_eq!(za.ct_lt(&zb).declassify(), (a as i64) < (b as i64));
    }
}

fn check_carry_chains<const L: usize>() {
    let mut rng = Rng(0x51bb00);
    let modulus = BigUint::from(1u32) << (64 * L);
    let mut values = vec![Uint::<L>::ZERO, Uint::ONE, Uint::MAX];
    for i in 0..L {
        // Borrow propagation across every possible number of zero limbs.
        let mut words = [0; L];
        words[i] = 1;
        values.push(Uint::from_words(words));
        words[i] = 1 << 63;
        values.push(Uint::from_words(words));
    }
    for _ in 0..32 {
        values.push(rng.uint());
    }
    let signed_min = -(BigInt::from(1u32) << (64 * L - 1));
    let signed_max = -&signed_min - 1;
    for a in &values {
        for b in &values {
            let (aa, bb) = (big(a.as_words()), big(b.as_words()));
            let sum = &aa + &bb;
            let checked = a.checked_add_ct(b);
            assert_eq!(checked.validity().declassify(), sum < modulus);
            assert_eq!(big(checked.value().as_words()), &sum % &modulus);
            let expected = (&aa + &modulus - &bb) % &modulus;
            assert_eq!(big(a.wrapping_sub(b).as_words()), expected);
            let checked = a.checked_sub_ct(b);
            assert_eq!(checked.validity().declassify(), aa >= bb);
            assert_eq!(big(checked.value().as_words()), expected);

            let (sa, sb) = (
                Z::from_twos_complement_words(*a.as_words()),
                Z::from_twos_complement_words(*b.as_words()),
            );
            let difference = signed(a.as_words()) - signed(b.as_words());
            let sum = signed(a.as_words()) + signed(b.as_words());
            let checked_sum = sa.checked_add_ct(&sb);
            assert_eq!(
                checked_sum.validity().declassify(),
                sum >= signed_min && sum <= signed_max
            );
            assert_eq!(big(checked_sum.value().as_words()), (&aa + &bb) % &modulus);
            let checked = sa.checked_sub_ct(&sb);
            assert_eq!(
                checked.validity().declassify(),
                difference >= signed_min && difference <= signed_max
            );
            assert_eq!(big(checked.value().as_words()), expected);
        }
    }
}

#[test]
fn multiword_carry_chains_match_bigint() {
    check_carry_chains::<1>();
    check_carry_chains::<2>();
    check_carry_chains::<3>();
    check_carry_chains::<5>();
    check_carry_chains::<9>();
    check_carry_chains::<20>();
}

fn check_products<const A: usize, const B: usize>() {
    let mut rng = Rng(0x981671);
    let mut lhs = vec![Uint::<A>::MAX];
    let mut rhs = vec![Uint::<B>::MAX];
    for _ in 0..127 {
        lhs.push(rng.uint());
        rhs.push(rng.uint());
    }
    let mut expected = BigUint::from(0u32);
    for (a, b) in lhs.iter().zip(&rhs) {
        let term = big(a.as_words()) * big(b.as_words());
        assert_eq!(product(IntegerOps.mul_wide(a, b)), term);
        expected += term;
    }
    let whole = IntegerOps.batch_mul_acc(&lhs, &rhs);
    assert_eq!(acc(whole), expected);
    let mut halves = IntegerOps.batch_mul_acc(&lhs[..57], &rhs[..57]);
    halves.merge_assign(&IntegerOps.batch_mul_acc(&lhs[57..], &rhs[57..]));
    assert_eq!(whole, halves);
    let signed_rhs: Vec<_> = rhs
        .iter()
        .map(|v| Z::from_twos_complement_words(*v.as_words()))
        .collect();
    let expected: BigInt = lhs
        .iter()
        .zip(&signed_rhs)
        .map(|(a, b)| BigInt::from(big(a.as_words())) * signed(b.as_words()))
        .sum();
    assert_eq!(
        signed_acc(IntegerOps.batch_mul_acc(&lhs, &signed_rhs)),
        expected
    );
}
#[test]
fn exact_full_width_products_and_signed_batches() {
    check_products::<1, 1>();
    check_products::<2, 1>();
    check_products::<2, 2>();
    check_products::<1, 3>();
    check_products::<3, 2>();
    check_products::<4, 4>();
}

#[test]
fn division_and_width_conversions() {
    let mut rng = Rng(0x479821);
    for _ in 0..128 {
        let a = rng.uint::<3>();
        let b = rng.uint::<3>();
        let actual = a.checked_div_rem_ct(&b);
        assert!(actual.validity().declassify());
        assert_eq!(
            big(actual.value().0.as_words()),
            big(a.as_words()) / big(b.as_words())
        );
        assert_eq!(
            big(actual.value().1.as_words()),
            big(a.as_words()) % big(b.as_words())
        );
        let sa = Z::from_twos_complement_words(*a.as_words());
        let sb = Z::from_twos_complement_words(*b.as_words());
        let actual = sa.checked_div_rem_ct(&sb);
        assert!(actual.validity().declassify());
        assert_eq!(
            signed(actual.value().0.as_words()),
            signed(a.as_words()) / signed(b.as_words())
        );
        assert_eq!(
            signed(actual.value().1.as_words()),
            signed(a.as_words()) % signed(b.as_words())
        );
    }
    assert!(
        !Uint::<2>::ONE
            .checked_div_rem_ct(&Uint::ZERO)
            .validity()
            .declassify()
    );
    assert!(
        !Z::<2>::MIN
            .checked_div_rem_ct(&Z::from_twos_complement_words([u64::MAX; 2]))
            .validity()
            .declassify()
    );
    assert_eq!(Z::<2>::MIN.unsigned_abs().as_words(), &[0, 1 << 63]);
    assert!(!Z::<2>::MIN.checked_neg_ct().validity().declassify());
    assert!(
        !Uint::<2>::MAX
            .checked_resize_ct::<1>()
            .validity()
            .declassify()
    );
    assert!(
        !Uint::<2>::MAX
            .checked_to_signed_ct()
            .validity()
            .declassify()
    );
    let minus_one = Z::<1>::from_twos_complement_words([u64::MAX]);
    assert_eq!(minus_one.sign_extend::<3>().as_words(), &[u64::MAX; 3]);
    assert!(
        minus_one
            .sign_extend::<3>()
            .checked_resize_ct::<1>()
            .validity()
            .declassify()
    );
    assert!(
        !Z::<2>::from_twos_complement_words([1 << 63, 0])
            .checked_resize_ct::<1>()
            .validity()
            .declassify()
    );
    for shift in [0, 1, 63, 64, 65, 127, 128, 129] {
        let x = Z::<2>::from_twos_complement_words([0x12345678, u64::MAX]);
        let expected = signed(x.as_words()) >> shift;
        assert_eq!(signed(x.arithmetic_shr(shift).as_words()), expected);
    }
}

fn check_prime<const L: usize>(modulus: Uint<L>) {
    let field = create_prime_field(modulus);
    let p = big(modulus.as_words());
    let mut rng = Rng(0x658799);
    for _ in 0..48 {
        let a = rng.uint::<L>();
        let b = rng.uint::<L>();
        let fa = field.from_integer(&a);
        let fb = field.from_integer(&b);
        let (a, b) = (big(a.as_words()) % &p, big(b.as_words()) % &p);
        assert_eq!(big(field.to_integer(&fa).as_words()), a);
        assert_eq!(
            big(field.to_integer(&field.add(&fa, &fb)).as_words()),
            (&a + &b) % &p
        );
        assert_eq!(
            big(field.to_integer(&field.sub(&fa, &fb)).as_words()),
            (&a + &p - &b) % &p
        );
        assert_eq!(
            big(field.to_integer(&field.mul(&fa, &fb)).as_words()),
            (&a * &b) % &p
        );
        assert_eq!(field.reduce(field.mul_wide(&fa, &fb)), field.mul(&fa, &fb));
    }
    let weights: Vec<_> = (0..97)
        .map(|_| field.from_integer(&rng.uint::<L>()))
        .collect();
    let integers: Vec<_> = (0..97).map(|_| rng.uint::<4>()).collect();
    let mut products = vec![field.zero(); weights.len()];
    field.batch_mul_into(&weights, &weights, &mut products);
    for (weight, actual) in weights.iter().zip(products) {
        let value = big(field.to_integer(weight).as_words());
        assert_eq!(
            big(field.to_integer(&actual).as_words()),
            (&value * &value) % &p
        );
    }
    let expected: BigUint = weights
        .iter()
        .zip(&integers)
        .map(|(w, i)| big(field.to_integer(w).as_words()) * big(i.as_words()))
        .sum();
    let actual = field.reduce(field.batch_mul_acc(&weights, &integers));
    assert_eq!(big(field.to_integer(&actual).as_words()), expected % &p);
    let expected: BigUint = weights
        .iter()
        .map(|w| {
            let x = big(field.to_integer(w).as_words());
            &x * &x
        })
        .sum();
    let actual = field.reduce(field.batch_mul_acc(&weights, &weights));
    assert_eq!(big(field.to_integer(&actual).as_words()), expected % &p);
    let signed_values: Vec<_> = integers
        .iter()
        .map(|i| Z::from_twos_complement_words(*i.as_words()))
        .collect();
    let expected: BigInt = weights
        .iter()
        .zip(&signed_values)
        .map(|(w, i)| BigInt::from(big(field.to_integer(w).as_words())) * signed(i.as_words()))
        .sum();
    let signed_p = BigInt::from(p.clone());
    let expected = (expected % &signed_p + &signed_p) % &signed_p;
    let actual = field.reduce(field.batch_mul_acc(&weights, &signed_values));
    assert_eq!(
        BigInt::from(big(field.to_integer(&actual).as_words())),
        expected
    );
    let exact = IntegerOps.batch_mul_acc(&integers, &integers);
    let expected = acc(exact) % &p;
    assert_eq!(
        big(field.to_integer(&field.reduce(exact)).as_words()),
        expected
    );
    assert!(!field.from_canonical_ct(&modulus).validity().declassify());
    let a = field.from_integer(&3u64);
    assert_eq!(field.mul(&a, field.inverse_ct(&a).value()), field.one());
    assert!(!field.inverse_ct(&field.zero()).validity().declassify());
    let input = [field.zero(), field.one(), a, field.zero(), field.neg(&a)];
    let inverse = field.batch_invert_or_zero_ct(&input);
    for (a, b) in input.iter().zip(inverse) {
        let expected = if a.ct_is_zero().declassify() {
            field.zero()
        } else {
            field.one()
        };
        assert_eq!(field.mul(a, &b), expected);
    }
    assert_eq!(
        field.batch_invert_or_zero_ct(&[field.zero(); 3]),
        vec![field.zero(); 3]
    );
}
#[test]
fn dynamic_prime_scalar_mixed_and_delayed_arithmetic() {
    check_prime(Uint::from_words([17]));
    check_prime(Uint::from_words([u64::MAX - 58]));
    check_prime(Uint::from_words([17, 0]));
    check_prime(Uint::from_words([u64::MAX - 14, (1 << 36) - 1]));
    check_prime(Uint::from_words([u64::MAX - 158, u64::MAX]));
    check_prime(Uint::from_words([17, 0, 0]));
}

struct P17;
impl PrimeSpec<2> for P17 {
    const MODULUS: Uint<2> = Uint::from_words([17, 0]);
}
#[test]
fn static_context_layout_native_widths_and_mapped_order() {
    assert_eq!(size_of::<Fp<2>>(), 16);
    assert_eq!(size_of::<StaticFp<P17, 2>>(), 16);
    assert_eq!(size_of::<StaticFpOps<P17, 2>>(), 0);
    assert_eq!(size_of::<FpProductAcc<2>>(), 40);
    assert_eq!(size_of::<FpLinearAcc<2, 1>>(), 32);
    let field = StaticFpOps::<P17, 2>::new();
    let weights = [field.from_integer(&3u64), field.from_integer(&7u64)];
    let integers = [u64::MAX, 1];
    let expected = (3u128 * u64::MAX as u128 + 7) % 17;
    assert_eq!(
        field
            .to_integer(&field.reduce(field.batch_mul_acc(&weights, &integers)))
            .as_words()[0],
        expected as u64
    );
    let a32: StaticFpLinearAcc<P17, 2, 1> = field.batch_mul_acc(&weights, &[4u32, 6]);
    let a64: StaticFpLinearAcc<P17, 2, 1> = field.batch_mul_acc(&weights, &[4u64, 6]);
    assert_eq!(field.reduce(a32), field.reduce(a64));
    let a128 = field.batch_mul_acc(&weights, &[u128::MAX, 5]);
    let expected = (BigUint::from(u128::MAX) * BigUint::from(3u64) + BigUint::from(35u64))
        % BigUint::from(17u64);
    assert_eq!(
        big(field.to_integer(&field.reduce(a128)).as_words()),
        expected
    );
    let mut next = 0;
    let mapped = field.batch_mul_acc_map(2, |i| {
        assert_eq!(i, next);
        next += 1;
        (weights[i], integers[i])
    });
    assert_eq!(next, 2);
    assert_eq!(
        field.reduce(mapped),
        field.reduce(field.batch_mul_acc(&weights, &integers))
    );
    let empty: &[StaticFp<P17, 2>] = &[];
    let empty_acc: StaticFpProductAcc<P17, 2> = field.batch_mul_acc(empty, empty);
    assert_eq!(field.reduce(empty_acc), field.zero());
}

fn check_wrapping_signed_product<const A: usize, const B: usize, const OUT: usize>() {
    let mut rng = Rng(0x758192);
    let mut left = vec![Z::<A>::ZERO, Z::ONE, Z::MIN, Z::MAX, -Z::ONE];
    let mut right = vec![Z::<B>::ZERO, Z::ONE, Z::MIN, Z::MAX, -Z::ONE];
    for _ in 0..32 {
        left.push(Z::from_twos_complement_words(*rng.uint().as_words()));
        right.push(Z::from_twos_complement_words(*rng.uint().as_words()));
    }
    let modulus = BigInt::from(1u8) << (64 * OUT);
    for a in &left {
        for b in &right {
            let product = signed(a.as_words()) * signed(b.as_words());
            let expected = ((product % &modulus) + &modulus) % &modulus;
            let actual = IntegerOps.wrapping_signed_product::<A, B, OUT>(a, b);
            assert_eq!(BigInt::from(big(actual.as_words())), expected);
        }
    }
}
#[test]
fn wrapping_signed_products_match_bigint() {
    check_wrapping_signed_product::<1, 1, 1>();
    check_wrapping_signed_product::<1, 1, 2>();
    check_wrapping_signed_product::<2, 3, 1>();
    check_wrapping_signed_product::<2, 3, 3>();
    check_wrapping_signed_product::<3, 2, 4>();
    check_wrapping_signed_product::<2, 2, 3>();
    check_wrapping_signed_product::<3, 3, 5>();
    check_wrapping_signed_product::<10, 10, 20>();
}
