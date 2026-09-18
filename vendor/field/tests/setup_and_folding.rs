use field::*;

field::define_prime_field! {
    pub Seventeen {
        limbs:2, modulus:[17,0], element:SeventeenElement, context:SeventeenField,
    }
}

struct Rng(u64);
impl PublicRandomSource for Rng {
    fn fill_bytes(&mut self, output: &mut [u8]) {
        for chunk in output.chunks_mut(8) {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            let words = (x ^ (x >> 31)).to_le_bytes();
            chunk.copy_from_slice(&words[..chunk.len()]);
        }
    }
}
struct Ones(usize);
impl PublicRandomSource for Ones {
    fn fill_bytes(&mut self, output: &mut [u8]) {
        self.0 += 1;
        output.fill(255);
    }
}

#[test]
fn prime_search_rejects_weak_policy_before_reading_randomness() {
    let mut source = Ones(0);
    let mut policy = PrimeSearchPolicy::default();
    let minimum = policy.minimum_rejection_draws().unwrap();
    assert!(minimum <= policy.max_rejection_draws);
    for budget in [0, 1, minimum - 1] {
        policy.max_rejection_draws = budget;
        assert!(matches!(
            sample_prime_public(
                &mut source,
                Uint::<1>::from_u64(61)..=Uint::from_u64(67),
                &policy
            ),
            Err(PrimeSearchError::InvalidPolicy)
        ));
        assert_eq!(source.0, 0);
    }
    policy.max_candidates = 0;
    assert_eq!(
        policy.minimum_rejection_draws(),
        Err(PrimeSearchError::InvalidPolicy)
    );
    policy.max_candidates = u64::MAX;
    policy.target_security_bits = u32::MAX;
    assert_eq!(
        policy.minimum_rejection_draws(),
        Err(PrimeSearchError::InvalidPolicy)
    );
}

#[test]
fn prime_search_boundaries_and_failure_budgets() {
    let mut source = Rng(89178);
    let policy = PrimeSearchPolicy {
        max_candidates: 8,
        ..PrimeSearchPolicy::default()
    };
    for (low, high, prime) in [
        (17, 17, 17),
        (16, 17, 17),
        (61, 61, 61),
        (u64::MAX - 58, u64::MAX - 58, u64::MAX - 58),
    ] {
        let accepted = sample_prime_public(
            &mut source,
            Uint::from_u64(low)..=Uint::from_u64(high),
            &policy,
        )
        .unwrap();
        assert_eq!(accepted.modulus(), &Uint::<1>::from_u64(prime));
        let field = FpCtx::from_prime(accepted);
        assert_eq!(field.modulus(), &Uint::<1>::from_u64(prime));
    }
    for (low, high) in [(2, 2), (54, 54), (17, 16)] {
        assert!(matches!(
            sample_prime_public(
                &mut source,
                Uint::<1>::from_u64(low)..=Uint::from_u64(high),
                &policy
            ),
            Err(PrimeSearchError::InvalidInterval)
        ));
    }
    assert!(matches!(
        sample_prime_public(
            &mut source,
            Uint::<1>::from_u64(91)..=Uint::from_u64(91),
            &policy
        ),
        Err(PrimeSearchError::Exhausted)
    ));
    for candidate in [0, 1, 4, 561, 1105, 1729, 3215031751] {
        assert!(
            !is_probable_prime_public(&Uint::<1>::from_u64(candidate)),
            "{candidate}"
        );
    }
    for _ in 0..16 {
        let field = FpCtx::sample_prime_public(
            &mut source,
            Uint::<1>::from_u64(1000)..=Uint::from_u64(1100),
            &PrimeSearchPolicy::default(),
        )
        .unwrap();
        let value = field.modulus().as_words()[0];
        assert!((1000..=1100).contains(&value));
        assert!((2..=34).all(|d| value % d != 0));
    }
}

#[test]
fn public_field_sampling_is_bounded_and_nonzero_sampling_excludes_zero() {
    let field = SeventeenField::new();
    let mut ones = Ones(0);
    assert_eq!(
        field.sample_public(&mut ones, 5),
        Err(SamplingError::Exhausted)
    );
    assert_eq!(ones.0, 5);
    let mut source = Rng(12877);
    let mut seen = [false; 17];
    for _ in 0..1024 {
        let value = field.sample_public(&mut source, 256).unwrap();
        seen[field.to_integer(&value).as_words()[0] as usize] = true;
        let nonzero = field.sample_nonzero_public(&mut source, 256).unwrap();
        assert!(!nonzero.ct_is_zero().declassify());
    }
    assert!(seen.into_iter().all(|v| v));
}

#[test]
fn canonical_codec_rejects_modulus_and_preserves_full_width() {
    let field = SeventeenField::new();
    let mut bytes = [0; 16];
    let value = field.from_integer(&16u64);
    field.encode_into(&value, &mut bytes);
    assert_eq!(bytes[0], 16);
    assert!(bytes[1..].iter().all(|v| *v == 0));
    assert_eq!(field.decode_public(&bytes).unwrap(), value);
    bytes[0] = 17;
    assert_eq!(field.decode_public(&bytes), Err(DecodeError::NonCanonical));
    let invalid = field.decode_ct(&bytes).unwrap();
    assert!(!invalid.validity().declassify());
    assert_eq!(invalid.value(), &field.zero());
    assert!(matches!(
        field.decode_public(&bytes[..15]),
        Err(DecodeError::Length { .. })
    ));
    let words = Uint::from_words([u64::MAX, 0, 1 << 63]);
    let mut encoded = [0; 24];
    IntegerOps.encode_into(&words, &mut encoded);
    let decoded: Uint<3> = IntegerOps.decode_public(&encoded).unwrap();
    assert_eq!(decoded, words);
    let negative = Z::<3>::MIN;
    IntegerOps.encode_into(&negative, &mut encoded);
    let decoded: Z<3> = IntegerOps.decode_public(&encoded).unwrap();
    assert_eq!(decoded, negative);
}

#[test]
fn folds_preserve_offsets_output_representation_and_in_place_prefix() {
    let field = SeventeenField::new();
    let r = field.from_integer(&11u64);
    let native = [u128::MAX, 7, 19, 0, 0x10000000000000000, 52, 3, 6];
    let projected: Vec<_> = native.iter().map(|x| field.from_integer(x)).collect();
    let expected: Vec<_> = projected
        .chunks_exact(2)
        .map(|p| field.add(&p[0], &field.mul(&r, &field.sub(&p[1], &p[0]))))
        .collect();
    let mut output = vec![field.zero(); 4];
    field.fold_pairs_into(&native, &mut output, &r);
    assert_eq!(output, expected);
    let mut plain = vec![Uint::<2>::ZERO; 2];
    let mut visited = Vec::new();
    field.fold_pairs_map_into(
        |i| {
            visited.push(i + 2);
            native[i + 2]
        },
        &mut plain,
        &r,
    );
    assert_eq!(visited, vec![2, 3, 4, 5]);
    assert_eq!(
        plain,
        expected[1..3]
            .iter()
            .map(|x| field.to_integer(x))
            .collect::<Vec<_>>()
    );
    let mut inplace = projected.clone();
    let tail = inplace[4..].to_vec();
    field.fold_in_place(&mut inplace, &r, 4);
    assert_eq!(&inplace[..4], &expected);
    assert_eq!(&inplace[4..], &tail);
    let mut plain: Vec<_> = projected.iter().map(|x| field.to_integer(x)).collect();
    field.fold_plain_in_place(&mut plain, &r, 4);
    assert_eq!(
        &plain[..4],
        &expected
            .iter()
            .map(|x| field.to_integer(x))
            .collect::<Vec<_>>()
    );
}

#[test]
fn signed_fold_handles_difference_larger_than_signed_width() {
    let field = SeventeenField::new();
    let challenge = field.from_integer(&7u64);
    let source = [Z::<1>::MIN, Z::<1>::MAX, Z::<1>::MAX, Z::<1>::MIN];
    let mut out = [field.zero(); 2];
    field.fold_pairs_into(&source, &mut out, &challenge);
    for (pair, actual) in source.chunks_exact(2).zip(out) {
        let a = field.from_integer(&pair[0]);
        let b = field.from_integer(&pair[1]);
        assert_eq!(
            actual,
            field.add(&a, &field.mul(&challenge, &field.sub(&b, &a)))
        );
    }
}

#[test]
fn prefix_fold_matches_sequential_low_bit_folds_and_mapped_reads() {
    let field = create_prime_field(Uint::<2>::from_words([97, 0]));
    let src = [2u64, 5, 7, 11, 13, 17, 19, 23];
    let challenges = [
        field.from_integer(&3u64),
        field.from_integer(&9u64),
        field.from_integer(&12u64),
    ];
    for depth in 0..=3 {
        let mut expected: Vec<_> = src.iter().map(|x| field.from_integer(x)).collect();
        for r in &challenges[..depth] {
            let mut next = vec![field.zero(); expected.len() / 2];
            field.fold_pairs_into(&expected, &mut next, r);
            expected = next;
        }
        let mut actual = vec![field.zero(); 8 >> depth];
        field.fold_prefix_into(&src, &mut actual, &challenges[..depth]);
        assert_eq!(actual, expected);
        let mut visited = Vec::new();
        field.fold_prefix_map_into(
            |i| {
                visited.push(i);
                src[i]
            },
            &mut actual,
            &challenges[..depth],
        );
        assert_eq!(actual, expected);
        assert_eq!(visited, (0..8).collect::<Vec<_>>());
    }
}
