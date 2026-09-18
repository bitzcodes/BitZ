use core::fmt::Debug;
use field::*;

fn lerp<C: RingOps>(field: &C, a: C::Elem, b: C::Elem, r: &C::Elem) -> C::Elem {
    field.add(
        &field.mul(&field.sub(&field.one(), r), &a),
        &field.mul(r, &b),
    )
}
fn naive_fold<C: RingOps>(field: &C, values: &[C::Elem], pending: &[C::Elem]) -> Vec<C::Elem> {
    let mut out = values.to_vec();
    for r in pending {
        out = out
            .chunks_exact(2)
            .map(|p| lerp(field, p[0], p[1], r))
            .collect();
    }
    out
}
fn eval<C: RingOps>(field: &C, coeff: [C::Elem; 3], r: &C::Elem) -> C::Elem {
    field.add(
        &coeff[0],
        &field.mul(r, &field.add(&coeff[1], &field.mul(r, &coeff[2]))),
    )
}
fn check<C: SumcheckKernels>(field: C, mut value: impl FnMut(usize) -> C::Elem)
where
    C::Elem: PartialEq + Debug,
{
    for n in [0, 1, 3, 17, 65] {
        let l: Vec<_> = (0..4 * n + 3).map(|i| value(i + 1)).collect();
        let r: Vec<_> = (0..4 * n + 3).map(|i| value(i + 191)).collect();
        let w: Vec<_> = (0..n).map(|i| value(i + 913)).collect();
        let one = field.eqf_single_pair_round(&l, &r, &w, n);
        let two = field.eqf_two_pair_round(&l, &r, &r, &l, &w, n);
        for t in [field.zero(), field.one(), value(231), value(1991)] {
            let expected = (0..n).fold(field.zero(), |sum, i| {
                field.add(
                    &sum,
                    &field.mul(
                        &w[i],
                        &field.mul(
                            &lerp(&field, l[2 * i], l[2 * i + 1], &t),
                            &lerp(&field, r[2 * i], r[2 * i + 1], &t),
                        ),
                    ),
                )
            });
            assert_eq!(eval(&field, one, &t), expected);
            assert_eq!(eval(&field, two, &t), field.add(&expected, &expected));
        }
        let rho = value(675);
        let expected_l = naive_fold(&field, &l[..4 * n], &[rho]);
        let expected_r = naive_fold(&field, &r[..4 * n], &[rho]);
        let expected = field.eqf_single_pair_round(&expected_l, &expected_r, &w, n);
        let mut fl = l.clone();
        let mut fr = r.clone();
        assert_eq!(
            field.eqf_fused_fold_round(&mut fl, &mut fr, &rho, &w, n),
            expected
        );
        assert_eq!(&fl[..2 * n], expected_l);
        assert_eq!(&fr[..2 * n], expected_r);
        assert_eq!(&fl[2 * n..], &l[2 * n..]);
        assert_eq!(&fr[2 * n..], &r[2 * n..]);
        let mut plain = l.clone();
        field.eqf_fold_in_place(&mut plain, &rho, 2 * n);
        assert_eq!(&plain[..2 * n], expected_l);
        assert_eq!(&plain[2 * n..], &l[2 * n..]);
    }
    for pending_count in 0..=2 {
        for quads in [0, 1, 3, 17] {
            let len = (4 * quads) << pending_count;
            let l: Vec<_> = (0..len + 5).map(|i| value(i + 35)).collect();
            let r: Vec<_> = (0..len + 5).map(|i| value(i + 781)).collect();
            let w: Vec<_> = (0..quads).map(|i| value(i + 391)).collect();
            let pending: Vec<_> = (0..pending_count).map(|i| value(i + 87)).collect();
            let expected_l = naive_fold(&field, &l[..len], &pending);
            let expected_r = naive_fold(&field, &r[..len], &pending);
            let mut fl = l.clone();
            let mut fr = r.clone();
            let coefficients = field.eqf_grid_pass(&mut fl, &mut fr, &pending, &w, quads);
            assert_eq!(&fl[..4 * quads], expected_l);
            assert_eq!(&fr[..4 * quads], expected_r);
            assert_eq!(&fl[4 * quads..], &l[4 * quads..]);
            assert_eq!(&fr[4 * quads..], &r[4 * quads..]);
            for t in [field.zero(), field.one(), value(291)] {
                for node in 0..3 {
                    let expected = (0..quads).fold(field.zero(), |sum, b| {
                        let a = &expected_l[4 * b..4 * b + 4];
                        let c = &expected_r[4 * b..4 * b + 4];
                        let l0 = lerp(&field, a[0], a[1], &t);
                        let l1 = lerp(&field, a[2], a[3], &t);
                        let r0 = lerp(&field, c[0], c[1], &t);
                        let r1 = lerp(&field, c[2], c[3], &t);
                        let (left, right) = match node {
                            0 => (l0, r0),
                            1 => (l1, r1),
                            _ => (field.sub(&l1, &l0), field.sub(&r1, &r0)),
                        };
                        field.add(&sum, &field.mul(&w[b], &field.mul(&left, &right)))
                    });
                    assert_eq!(
                        eval(
                            &field,
                            [
                                coefficients[node],
                                coefficients[3 + node],
                                coefficients[6 + node]
                            ],
                            &t
                        ),
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn weighted_rounds_fused_folds_and_grids_match_polynomial_evaluation() {
    let field = create_prime_field(Uint::from_words([u64::MAX - 14, (1 << 36) - 1]));
    check(field.clone(), |i| {
        field.from_integer(&((i as u128 * 193) << 65))
    });
    check(Gf128Ops, |i| Gf128::new(i as u64 * 391 + 2, i as u64 * 781));
    check(B127Ops, |i| {
        B127::from_polynomial_words([i as u64 * 17 + 2, i as u64 * 879])
    });
    check(Gf8Ops, |i| Gf8(i as u8));
}

#[test]
fn shape_errors_precede_writes() {
    let field = Gf128Ops;
    let mut l = [field.one(); 4];
    let mut r = l;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        field.eqf_fused_fold_round(&mut l, &mut r, &field.one(), &[], 1)
    }));
    assert!(result.is_err());
    assert_eq!(l, [field.one(); 4]);
    assert_eq!(r, l);
    assert!(
        std::panic::catch_unwind(|| field.eqf_single_pair_round(&[], &[], &[], usize::MAX))
            .is_err()
    );
}

#[test]
fn prepared_binary_accumulation_preserves_the_interleaved_product_domain() {
    let field = Gf128Ops;
    for n in [0, 1, 2, 3, 17, 129] {
        let values: Vec<_> = (0..n)
            .map(|i| {
                Gf128::new(
                    (i as u64).wrapping_mul(0xbeef123),
                    !(i as u64).rotate_left(19),
                )
            })
            .collect();
        let fixed: Vec<_> = (0..n)
            .map(|i| Gf128::new(!(i as u64), i as u64 * 191))
            .collect();
        let prepared: Vec<_> = fixed.iter().copied().map(PreparedGf128Mul::new).collect();
        let expected = field.reduce(field.batch_mul_acc(&values, &fixed));
        let mut acc = Gf128PreparedAcc::zero();
        for (a, b) in values.iter().zip(&prepared) {
            acc.add_mul(a, b);
        }
        assert_eq!(acc.reduce(), expected);
        assert_eq!(
            field.reduce(field.batch_mul_acc(&values, &prepared)),
            expected
        );
        assert_eq!(
            field.reduce(field.batch_mul_acc_map(n, |i| (values[i], prepared[i]))),
            expected
        );
    }
}
