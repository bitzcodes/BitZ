use field::*;

fn evaluate_pairs<C: FieldOps>(
    field: &C,
    weights: &[C::Elem],
    values: &[C::Elem],
    x: &C::Elem,
) -> C::Elem {
    weights
        .chunks_exact(2)
        .zip(values.chunks_exact(2))
        .fold(field.zero(), |sum, (m, w)| {
            let m = field.add(&m[0], &field.mul(x, &field.sub(&m[1], &m[0])));
            let w = field.add(&w[0], &field.mul(x, &field.sub(&w[1], &w[0])));
            field.add(&sum, &field.mul(&m, &w))
        })
}
fn assert_message<C: FieldOps>(field: &C, m: &[C::Elem], w: &[C::Elem], message: [C::Elem; 2]) {
    let claim = m
        .iter()
        .zip(w)
        .fold(field.zero(), |s, (m, w)| field.add(&s, &field.mul(m, w)));
    let linear = field.sub(
        &field.sub(&claim, &field.add(&message[0], &message[0])),
        &message[1],
    );
    // Compare the polynomial at several independent challenges, not just its
    // coefficient formula. This also checks the claim's linear reconstruction.
    let mut x = field.zero();
    for _ in 0..7 {
        let actual = field.add(
            &message[0],
            &field.mul(&x, &field.add(&linear, &field.mul(&x, &message[1]))),
        );
        assert!(actual.ct_eq(&evaluate_pairs(field, m, w, &x)).declassify());
        x = field.add(&x, &field.one());
    }
}

fn check<T: Copy, const L: usize>(field: FpCtx<L>, values: Vec<T>)
where
    FpCtx<L>: RingOps<Elem = Fp<L>> + DotProductKernels<T, Uint<L>> + IntegerEmbedding<T>,
{
    let weights: Vec<_> = (0..values.len())
        .map(|i| {
            <FpCtx<L> as IntegerEmbedding<u64>>::from_integer(
                &field,
                &((i as u64).wrapping_mul(0xf817ed5ab6e92583)),
            )
        })
        .collect();
    let projected: Vec<_> = values.iter().map(|w| field.from_integer(w)).collect();
    let pair = field.dot_pair_round(&weights, &values);
    assert_message(&field, &weights, &projected, pair);
    let mut read_count = 0;
    let mapped = field.dot_pair_round_map(&weights, |i| {
        assert_eq!(i, read_count);
        read_count += 1;
        values[i]
    });
    assert_eq!(read_count, values.len());
    assert_eq!(pair, mapped);

    let challenge = <FpCtx<L> as IntegerEmbedding<u64>>::from_integer(&field, &19u64);
    let mut m = vec![field.zero(); values.len() / 2];
    let mut w = vec![Uint::ZERO; values.len() / 2];
    read_count = 0;
    let message = field.dot_fold_round_map_into(
        &weights,
        |i| {
            assert_eq!(i, read_count);
            read_count += 1;
            values[i]
        },
        &mut m,
        &mut w,
        &challenge,
    );
    assert_eq!(read_count, values.len());
    let mut m_slice = vec![field.zero(); m.len()];
    let mut w_slice = vec![Uint::ZERO; w.len()];
    assert_eq!(
        message,
        field.dot_fold_round_into(&weights, &values, &mut m_slice, &mut w_slice, &challenge)
    );
    assert_eq!((m.clone(), w.clone()), (m_slice, w_slice));
    for i in 0..m.len() {
        assert_eq!(
            m[i],
            field.add(
                &weights[2 * i],
                &field.mul(&challenge, &field.sub(&weights[2 * i + 1], &weights[2 * i]))
            )
        );
        let expected = field.add(
            &projected[2 * i],
            &field.mul(
                &challenge,
                &field.sub(&projected[2 * i + 1], &projected[2 * i]),
            ),
        );
        assert_eq!(w[i], field.to_integer(&expected));
    }
    let projected: Vec<_> = w
        .iter()
        .map(|w| <FpCtx<L> as IntegerEmbedding<Uint<L>>>::from_integer(&field, w))
        .collect();
    assert_message(&field, &m, &projected, message);

    if values.len() % 8 == 0 {
        let mut mo = vec![field.zero(); m.len() / 2];
        let mut wo = vec![Uint::ZERO; w.len() / 2];
        let message = field.dot_fold_plain_round_into(&m, &w, &mut mo, &mut wo, &challenge);
        let mut reference_m = vec![field.zero(); mo.len()];
        let mut reference_w = vec![Uint::ZERO; wo.len()];
        let reference = <FpCtx<L> as DotProductKernels<Uint<L>, Uint<L>>>::dot_fold_round_into(
            &field,
            &m,
            &w,
            &mut reference_m,
            &mut reference_w,
            &challenge,
        );
        assert_eq!((message, mo, wo), (reference, reference_m, reference_w));
    }
}

#[test]
fn native_rounds_and_folds_preserve_the_inner_product_polynomial() {
    for n in [0, 4, 8, 68, 72] {
        check(
            create_prime_field(Uint::from_words([17])),
            (0..n)
                .map(|i| if i % 2 == 0 { u32::MAX } else { 0 })
                .collect(),
        );
        let field = create_prime_field(Uint::from_words([277, 1 << 36]));
        check(
            field.clone(),
            (0..n)
                .map(|i| if i % 3 == 0 { u64::MAX } else { i as u64 })
                .collect(),
        );
        check(
            field.clone(),
            (0..n)
                .map(|i| {
                    if i % 3 == 1 {
                        u128::MAX
                    } else {
                        (i as u128) << 65
                    }
                })
                .collect(),
        );
        check(
            field.clone(),
            (0..n)
                .map(|i| Uint::from_words([i as u64, u64::MAX, 0, u64::MAX]))
                .collect(),
        );
        check(field, (0..n).map(|i| Bit::from_lsb(i as u64)).collect());
        check(
            create_prime_field(Uint::from_words([u64::MAX - 158, u64::MAX])),
            (0..n)
                .map(|i| if i % 2 == 0 { u128::MAX } else { 0 })
                .collect(),
        );
    }
}

#[test]
fn signed_and_field_rounds_preserve_extremes_and_read_order() {
    for n in [0, 4, 8, 68, 72] {
        for words in [[17, 0], [277, 1 << 36], [u64::MAX - 158, u64::MAX]] {
            let field = create_prime_field(Uint::from_words(words));
            let values: Vec<_> = (0..n)
                .map(|i| if i % 2 == 0 { Z::<4>::MIN } else { Z::<4>::MAX })
                .collect();
            check(field.clone(), values);
            let weights: Vec<_> = (0..n)
                .map(|i| field.from_integer(&(i as u64 + 7)))
                .collect();
            let values: Vec<_> = (0..n)
                .map(|i| field.from_integer(&(u128::MAX - i as u128)))
                .collect();
            assert_message(
                &field,
                &weights,
                &values,
                field.dot_pair_round(&weights, &values),
            );
            let mut mo = vec![field.zero(); n / 2];
            let mut wo = vec![Uint::ZERO; n / 2];
            let challenge = field.from_integer(&13u64);
            let mut calls = 0;
            let message = field.dot_fold_round_map_into(
                &weights,
                |i| {
                    assert_eq!(i, calls);
                    calls += 1;
                    values[i]
                },
                &mut mo,
                &mut wo,
                &challenge,
            );
            assert_eq!(calls, n);
            let mut expected = vec![field.zero(); n / 2];
            field.fold_pairs_into(&values, &mut expected, &challenge);
            assert_eq!(
                wo,
                expected
                    .iter()
                    .map(|v| field.to_integer(v))
                    .collect::<Vec<_>>()
            );
            assert_message(&field, &mo, &expected, message);
        }
    }
}

#[test]
fn canonical_products_keep_the_representation_without_a_projection_pass() {
    for words in [[17, 0], [277, 1 << 36], [u64::MAX - 158, u64::MAX]] {
        let field = create_prime_field(Uint::from_words(words));
        let coefficients = [field.from_integer(&13u64), field.from_integer(&u128::MAX)];
        let values = coefficients.map(|v| field.to_integer(&v));
        let mut output = [Uint::ZERO; 2];
        field.mul_canonical_into(&values, &coefficients, &mut output);
        assert_eq!(
            output,
            coefficients.map(|v| field.to_integer(&field.square(&v)))
        );
        let acc = field.batch_mul_acc(&coefficients, &values);
        assert_eq!(
            field.reduce_linear_to_integer(acc),
            field.to_integer(&field.reduce(acc))
        );
    }
}

#[test]
fn shape_errors_precede_reads_and_writes() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let field = create_prime_field(Uint::from_words([17]));
    let weights = [field.one(); 8];
    let mut mo = [field.one(); 2];
    let mut wo = [Uint::ONE; 2];
    let mut calls = 0;
    assert!(
        catch_unwind(AssertUnwindSafe(|| field.dot_fold_round_map_into(
            &weights,
            |_| {
                calls += 1;
                3u64
            },
            &mut mo,
            &mut wo,
            &field.one()
        )))
        .is_err()
    );
    assert_eq!(calls, 0);
    assert_eq!(mo, [field.one(); 2]);
    assert_eq!(wo, [Uint::ONE; 2]);
    assert!(
        catch_unwind(AssertUnwindSafe(|| field.dot_pair_round_map(
            &weights[..3],
            |_| {
                calls += 1;
                3u64
            }
        )))
        .is_err()
    );
    assert_eq!(calls, 0);
}
