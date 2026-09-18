use super::*;
use crate::integer_storage::IntegerTable;
use field::{Gf128Ops, IntegerEmbedding, RingOps, Uint, WideMul, Z, create_prime_field};

#[test]
fn integer_field_and_explicit_matrix_agree() {
    for p in [97u64, 101] {
        let f = create_prime_field(Uint::<1>::from_words([p]));
        let mut b = WengertBuilder::new(Vec::<i64>::new());
        let x = b.input();
        let y = b.input();
        let _unused = b.input();
        let a = b.linear_combination([(x, 3), (y, 2)]);
        let c = b.sub(x, y);
        b.output(a);
        b.output(c);
        b.output(a);
        let tape = b.finish();
        let w = [5u64, 7, 11].map(|v| f.from_integer(&v));
        let xs = [13u64, 17, 19].map(|v| f.from_integer(&v));
        let mut prepared = tape.prepare(&f);
        let mut out = vec![f.zero(); 3];
        prepared.mul_left_into(&w, &mut out).unwrap();
        assert_eq!(
            out,
            [f.from_integer(&55u64), f.from_integer(&25u64), f.zero()]
        );
        let forward = prepared
            .evaluate_bilinear(&w, &DenseColumns::new(&f, &xs))
            .unwrap();
        let dot = out
            .iter()
            .zip(xs)
            .fold(f.zero(), |s, (a, b)| f.add(&s, &f.mul(a, &b)));
        assert_eq!(forward, dot);
        let mut fb = WengertBuilder::new(FieldCoefficients::new(&f));
        let x = fb.input();
        let y = fb.input();
        fb.input();
        let a = fb.linear_combination([(x, f.from_integer(&3u64)), (y, f.from_integer(&2u64))]);
        let c = fb.sub(x, y);
        fb.output(a);
        fb.output(c);
        fb.output(a);
        let ft = fb.finish();
        ft.prepare().mul_left_into(&w, &mut out).unwrap();
        assert_eq!(
            forward,
            ft.prepare()
                .evaluate_bilinear(&w, &DenseColumns::new(&f, &xs))
                .unwrap()
        );
        prepared.mul_left_into(&[f.zero(); 3], &mut out).unwrap();
        assert_eq!(out, vec![f.zero(); 3]);
    }
}

#[test]
fn mixed_widths_and_nested_labels_do_not_wrap() {
    let mut b = WengertBuilder::new(IntegerTable::default());
    let x = b.input();
    let max = Z::<1>::from_twos_complement_words([i64::MAX as u64]);
    let wide = Z::<4>::from_twos_complement_words([5, 1, 0, 0]);
    let a = b.scale(x, max);
    let a = b.scale(a, Z::<1>::from(2u64));
    let c = b.scale(x, wide);
    let sum = b.add(a, c);
    b.output(sum);
    let tape = b.finish();
    assert_eq!(
        tape.coefficients()
            .iter()
            .map(|w| w.len())
            .collect::<Vec<_>>(),
        [1, 1, 4]
    );
    for p in [97u64, 101] {
        let f = create_prime_field(Uint::<1>::from_words([p]));
        let expected = f.add(
            &f.mul(&f.from_integer(&max), &f.from_integer(&2u64)),
            &f.from_integer(&wide),
        );
        let mut out = [f.zero()];
        tape.prepare(&f)
            .mul_left_into(&[f.one()], &mut out)
            .unwrap();
        assert_eq!(out, [expected]);
    }
}

#[test]
fn packed_full_low_zero_and_alias_boundaries() {
    let f = create_prime_field(Uint::<1>::from_words([97]));
    for bits in [1, 2, 7, 32, 33, 65, 4099, 8201] {
        for low in [0, 1, bits / 2, bits - 1, bits] {
            let mut b = WengertBuilder::new(Vec::<u64>::new());
            let unused = b.input();
            let packed = b.packed_inputs(bits, low);
            b.output(packed.full);
            b.output(packed.low);
            let zero = b.zero();
            b.output(zero);
            let _dead = b.scale(unused, 99);
            let tape = b.finish();
            let mut prepared = tape.prepare(&f);
            let w = [
                f.from_integer(&3u64),
                f.from_integer(&5u64),
                f.from_integer(&7u64),
            ];
            let mut out = vec![f.one(); bits + 1];
            prepared.mul_left_into(&w, &mut out).unwrap();
            assert_eq!(out[0], f.zero());
            let mut pow = f.one();
            for i in 0..bits {
                assert_eq!(
                    out[1 + i],
                    f.mul(&pow, &f.from_integer(&if i < low { 8u64 } else { 3u64 }))
                );
                pow = f.add(&pow, &pow);
            }
            let mut compact = vec![f.zero(); bits + 1];
            prepared
                .adjoint_map_segments(
                    w.len(),
                    |i| w[i],
                    |run| {
                        let mut value = run.base;
                        for slot in &mut compact[run.first_column..run.first_column + run.len] {
                            *slot = value;
                            value = f.add(&value, &value);
                        }
                    },
                )
                .unwrap();
            assert_eq!(compact, out);
            let xs = (0..=bits)
                .map(|i| f.from_integer(&(i as u64 + 1)))
                .collect::<Vec<_>>();
            let expected = out
                .iter()
                .zip(&xs)
                .fold(f.zero(), |s, (a, b)| f.add(&s, &f.mul(a, b)));
            assert_eq!(
                prepared
                    .evaluate_bilinear(&w, &DenseColumns::new(&f, &xs))
                    .unwrap(),
                expected
            );
            for run in prepared.power_runs() {
                let mut value = run.base;
                for i in 0..run.len {
                    assert_eq!(out[run.first_column + i], value);
                    value = f.add(&value, &value);
                }
            }
        }
    }
}

#[test]
fn binary_field_coefficients_and_zero_domain() {
    let f = Gf128Ops;
    let mut b = WengertBuilder::new(FieldCoefficients::new(&f));
    let x = b.input();
    let y = b.scale(x, f.one());
    let z = b.sub(y, x);
    b.output(z);
    let tape = b.finish();
    let mut out = [f.one()];
    tape.prepare().mul_left_into(&[f.one()], &mut out).unwrap();
    assert_eq!(out, [f.zero()]);
    let b = WengertBuilder::new(Vec::<u64>::new());
    let tape = b.finish();
    let prime = create_prime_field(Uint::<1>::from_words([97]));
    let mut p = tape.prepare(&prime);
    p.mul_left_into(&[], &mut []).unwrap();
    assert_eq!(
        p.evaluate_bilinear(&[], &DenseColumns::new(&prime, &[]))
            .unwrap(),
        prime.zero()
    );
}

#[test]
#[should_panic(expected = "another Wengert builder")]
fn rejects_foreign_nodes() {
    let mut a = WengertBuilder::new(Vec::<u64>::new());
    let mut b = WengertBuilder::new(Vec::<u64>::new());
    b.output(a.input());
}

#[test]
fn rejects_wrong_shapes_before_mutation() {
    let f = create_prime_field(Uint::<1>::from_words([97]));
    let mut b = WengertBuilder::new(Vec::<u64>::new());
    let x = b.input();
    b.output(x);
    let tape = b.finish();
    let mut p = tape.prepare(&f);
    let mut out = [f.one()];
    assert!(p.mul_left_into(&[], &mut out).is_err());
    assert_eq!(out, [f.one()]);
    assert!(
        p.evaluate_bilinear(&[f.one()], &DenseColumns::new(&f, &[]))
            .is_err()
    );
}

#[test]
fn sparse_mixed_mac_matches_field_embedding_and_bilinear_identity() {
    use super::PreparedColumns;
    use super::contraction::PreparedSignedSparse;
    let f = create_prime_field(Uint::<2>::from(97u64));
    let rows = vec![vec![(0, 1u64), (2, u64::MAX)], vec![(0, 7), (1, 9)], vec![]];
    let matrix = CscMatrix::<Box<[_]>>::try_from_rows(4, rows).unwrap();
    let embedded = matrix
        .clone()
        .with_coefficients(
            matrix
                .coefficients()
                .iter()
                .map(|x| f.from_integer(x))
                .collect::<Box<[_]>>(),
        )
        .unwrap();
    let w = [2u64, 3, 5].map(|x| f.from_integer(&x));
    let x = [7u64, 11, 13, 17].map(|x| f.from_integer(&x));
    let mut out = [f.zero(); 4];
    let mut expected = out;
    PreparedColumns::mixed(&f, &matrix)
        .mul_left_into(&w, &mut out)
        .unwrap();
    PreparedColumns::mixed(&f, &embedded)
        .mul_left_into(&w, &mut expected)
        .unwrap();
    assert_eq!(out, expected);
    let dot = out
        .iter()
        .zip(x)
        .fold(f.zero(), |s, (a, b)| f.add(&s, &f.mul(a, &b)));
    assert_eq!(
        PreparedColumns::mixed(&f, &matrix)
            .evaluate_bilinear(&w, &DenseColumns::new(&f, &x))
            .unwrap(),
        dot
    );
    let signed = CscMatrix::<Box<[_]>>::try_from_rows(
        4,
        vec![vec![(0, -1), (2, i64::MIN)], vec![(0, 1), (1, 9)], vec![]],
    )
    .unwrap();
    let mut prepared = PreparedSignedSparse::new(&f, &signed);
    for weights in [w, [0u64, 96, 17].map(|v| f.from_integer(&v)), w] {
        prepared.mul_left_into(&weights, &mut out).unwrap();
        for (j, column) in signed.columns().enumerate() {
            let value = column.into_iter().fold(f.zero(), |s, (row, c)| {
                f.add(&s, &f.mul(&weights[row], &f.from_integer(c)))
            });
            assert_eq!(out[j], value);
        }
        let dot = out
            .iter()
            .zip(x)
            .fold(f.zero(), |s, (a, b)| f.add(&s, &f.mul(a, &b)));
        assert_eq!(
            prepared
                .evaluate_bilinear(&weights, &DenseColumns::new(&f, &x))
                .unwrap(),
            dot
        );
    }
    let before = out;
    assert!(prepared.mul_left_into(&w[..2], &mut out).is_err());
    assert_eq!(out, before);
    assert!(
        PreparedColumns::mixed(&f, &matrix)
            .mul_left_into(&w[..2], &mut out)
            .is_err()
    );
}

#[test]
fn delayed_graph_reduction_matches_immediate_and_chunks_large_nodes() {
    // Each output seeds one shared input. Fan-out exceeds one chunk so the
    // adjoint must reduce before accumulating the remaining contributions.
    for prime in [97u128, (1u128 << 127) - 1] {
        let f = create_prime_field(Uint::<2>::from(prime));
        let mut b = WengertBuilder::new(Vec::<u64>::new());
        let x = b.input();
        for i in 0..65543u64 {
            let v = b.scale(x, i + 2);
            b.output(v);
        }
        let tape = b.finish();
        let weights = f
            .zero_vec(tape.output_count())
            .into_iter()
            .enumerate()
            .map(|(i, _)| f.from_integer(&(i as u64 + 7)))
            .collect::<Vec<_>>();
        let mut p = tape.prepare(&f);
        let mut ordinary = [f.zero()];
        let mut delayed = ordinary;
        p.mul_left_into(&weights, &mut ordinary).unwrap();
        p.adjoint_delayed_map_storage_into(weights.len(), |i| weights[i], &mut delayed, |x| x)
            .unwrap();
        assert_eq!(ordinary, delayed);
    }
}

#[test]
fn binary_adjoint_streams_unaligned_ranges_without_materializing_weights() {
    use super::binary::PreparedVirtualMap;
    use super::binary_adjoint::BinaryAdjoint;
    let matrix =
        CscMatrix::<Box<[_]>>::try_from_rows(256, (0..256).map(|i| vec![(i, true)]).collect())
            .unwrap();
    let map = PreparedVirtualMap::new(matrix.clone()).unwrap();
    assert_eq!(
        map,
        PreparedVirtualMap::from_implicit(
            matrix.with_coefficients(ImplicitOnes::new(256)).unwrap()
        )
        .unwrap()
    );
    let points = vec![
        (0..8)
            .map(|i| field::Gf128::from_polynomial_words([i + 2, 0]))
            .collect::<Vec<_>>(),
    ];
    let eta = [field::Gf128::one()];
    let equality = |p: &[field::Gf128], _: &()| {
        let mut v = vec![field::Gf128::zero(); 1 << p.len()];
        v[0] = field::Gf128::one();
        for (bit, r) in p.iter().enumerate() {
            for i in 0..1 << bit {
                let t = v[i] * r;
                v[i] += t;
                v[i + (1 << bit)] = t;
            }
        }
        Ok(v)
    };
    let weights = BinaryAdjoint::new_with_tail(&map, &points, &eta, 4, false, equality, false);
    let expected = equality(&points[0], &()).unwrap();
    assert_eq!(weights.column_count(), map.cols());
    for (first, len) in [(0, 256), (1, 254), (7, 130), (255, 1), (256, 0)] {
        let mut out = vec![field::Gf128::zero(); len];
        weights.fill_range(first, &mut out).unwrap();
        assert_eq!(out, expected[first..first + len]);
    }
    let mut out = [field::Gf128::one(); 2];
    assert!(weights.fill_range(255, &mut out).is_err());
    assert_eq!(out, [field::Gf128::one(); 2]);
}

#[test]
fn csr_csc_boolean_products_use_and_xor_and_overwrite_outputs() {
    let mut csr = CsrMatrix::<Box<[bool]>>::try_from_rows(
        3,
        vec![
            vec![(0, true), (1, true), (2, false)],
            vec![(1, true)],
            vec![],
        ],
    )
    .unwrap();
    let mut csc = csr.clone().into_csc().unwrap();
    let mut right = [true; 3];
    csr.mul_right_into(&[true, true, true], &mut right).unwrap();
    assert_eq!(right, [false, true, false]);
    let mut left = [true; 3];
    csc.mul_left_into(&[true, true, true], &mut left).unwrap();
    assert_eq!(left, [true, false, false]);
    let before = left;
    assert!(csc.mul_left_into(&[true], &mut left).is_err());
    assert_eq!(left, before);
    let mut implicit =
        CscMatrix::<ImplicitOnes, u32>::try_from_binary_rows(2, vec![vec![0, 1], vec![1]]).unwrap();
    let mut out = [true; 2];
    implicit.mul_left_into(&[true, true], &mut out).unwrap();
    assert_eq!(out, [true, false]);
}

#[test]
fn exact_integer_widths_are_proved_before_private_evaluation() {
    use crate::integer_storage::IntegerTable;
    let mut builder = CsrBuilder::<IntegerTable, u32>::new(IntegerTable::default());
    builder.push_row([(0, Z::<1>::MIN)]).unwrap();
    builder.push_row([(0, Z::<4>::ONE)]).unwrap();
    let matrix = builder.finish(1).unwrap();
    let mut prepared = PreparedIntegerRows::<bool, 1, u32>::new(&matrix).unwrap();
    let mut out = [Z::<1>::ZERO; 2];
    prepared.mul_right_into(&[true], &mut out).unwrap();
    assert_eq!(out, [Z::MIN, Z::ONE]);
    prepared.mul_right_into(&[false], &mut out).unwrap();
    assert_eq!(out, [Z::ZERO, Z::ZERO]);
    assert!(PreparedIntegerRows::<u64, 1, u32>::new(&matrix).is_err());
    let mut wide = PreparedIntegerRows::<u64, 2, u32>::new(&matrix).unwrap();
    let mut result = [Z::<2>::ZERO; 2];
    wide.mul_right_into(&[u64::MAX], &mut result).unwrap();
    assert_eq!(
        result[0],
        *field::IntegerOps
            .mul_wide(&Z::<1>::MIN, &Uint::<1>::from(u64::MAX))
            .checked_resize_ct::<2>()
            .value()
    );
    assert_eq!(result[1], Z::<2>::from(u64::MAX));
    let columns = matrix.into_csc().unwrap();
    let mut left = PreparedIntegerColumns::<bool, 1, u32>::new(&columns).unwrap();
    let mut sum = [Z::<1>::ZERO];
    left.mul_left_into(&[true, true], &mut sum).unwrap();
    assert_eq!(sum[0], Z::<1>::MIN.wrapping_add(&Z::ONE));

    let mut builder = CsrBuilder::new(IntegerTable::default());
    builder.push_row([(0, Z::<1>::ONE)]).unwrap();
    let matrix = builder.finish(1).unwrap();
    let mut signed = PreparedIntegerRows::<Z<1>, 1>::new(&matrix).unwrap();
    signed.mul_right_into(&[Z::MIN], &mut sum).unwrap();
    assert_eq!(sum, [Z::MIN]);
    assert!(signed.mul_right_into(&[], &mut sum).is_err());
    assert_eq!(sum, [Z::MIN], "invalid dimensions must not change output");

    let mut builder = CsrBuilder::<IntegerTable>::new(IntegerTable::default());
    builder.push_row([(0, Z::<1>::from(-1i64))]).unwrap();
    let matrix = builder.finish(1).unwrap();
    // Negating the minimum signed value needs one additional bit.
    assert!(PreparedIntegerRows::<Z<1>, 1>::new(&matrix).is_err());
    let mut signed = PreparedIntegerRows::<Z<1>, 2>::new(&matrix).unwrap();
    let mut sum = [Z::<2>::ZERO];
    signed.mul_right_into(&[Z::MIN], &mut sum).unwrap();
    assert_eq!(sum[0], Z::<2>::from(1u64 << 63));
}

#[test]
fn shared_mixed_width_preparation_and_native_inputs_match_expanded_fields() {
    use crate::integer_storage::IntegerTable;
    use std::sync::Arc;
    let f = create_prime_field(Uint::<2>::from((1u128 << 127) - 1));
    let mut table = IntegerTable::default();
    table.push(Z::<1>::from(-3i64));
    table.push(Z::<4>::from(7u64));
    let coefficients = IndexedCoefficients::new(Arc::new(table), vec![0u32, 1, 1]).unwrap();
    let csr =
        CsrMatrix::<_, u32>::try_from_parts(3, vec![0, 2, 3], vec![0, 2, 1], coefficients).unwrap();
    let csc = csr.clone().into_csc().unwrap();
    let mut right = PreparedRows::projected(&f, &csr, |c| f.from_integer(&c));
    let mut left = PreparedColumns::projected(&f, &csc, |c| f.from_integer(&c));
    assert_eq!(right.prepared_coefficient_count(), 2);
    assert_eq!(left.prepared_coefficient_count(), 2);
    let native = [u128::MAX, 1u128 << 100, 5];
    let values = native.map(|x| f.from_integer(&x));
    let weights = [f.from_integer(&11u64), f.from_integer(&13u64)];
    let mut rows = [f.zero(); 2];
    right.mul_right_into(&native, &mut rows).unwrap();
    let mut columns = [f.zero(); 3];
    left.mul_left_into(&weights, &mut columns).unwrap();
    let a = rows
        .iter()
        .zip(weights)
        .fold(f.zero(), |s, (x, w)| f.add(&s, &f.mul(x, &w)));
    let b = columns
        .iter()
        .zip(values)
        .fold(f.zero(), |s, (x, v)| f.add(&s, &f.mul(x, &v)));
    assert_eq!(a, b);
    assert_eq!(
        right
            .evaluate_bilinear(&weights, &DenseColumns::new(&f, &values))
            .unwrap(),
        a
    );
    assert_eq!(
        left.evaluate_bilinear(&weights, &DenseColumns::new(&f, &values))
            .unwrap(),
        a
    );

    let matrix = CsrMatrix::<Box<[_]>>::try_from_rows(1, vec![vec![(0, weights[0])]]).unwrap();
    let mut borrowed = PreparedRows::borrowed(&f, &matrix);
    let mut out = [f.zero()];
    borrowed.mul_right_into(&[u64::MAX], &mut out).unwrap();
    assert_eq!(out[0], f.mul(&weights[0], &f.from_integer(&u64::MAX)));
}

#[test]
fn wengert_right_product_matches_left_product_and_bilinear_evaluation() {
    let f = create_prime_field(Uint::<2>::from(97u64));
    let mut builder = WengertBuilder::new(Vec::<i64>::new());
    let x = builder.input();
    let y = builder.input();
    let z = builder.scale(x, -3);
    let w = builder.add(z, y);
    builder.output(w);
    builder.output(w);
    builder.output(x);
    let tape = builder.finish();
    let mut map = tape.prepare(&f);
    let values = [f.from_integer(&5u64), f.from_integer(&7u64)];
    let weights = [
        f.from_integer(&11u64),
        f.from_integer(&13u64),
        f.from_integer(&17u64),
    ];
    let mut right = [f.one(); 3];
    map.mul_right_into(&values, &mut right).unwrap();
    let v = f.add(&f.mul(&values[0], &f.from_integer(&-3i64)), &values[1]);
    assert_eq!(right, [v, v, values[0]]);
    let mut left = [f.one(); 2];
    map.mul_left_into(&weights, &mut left).unwrap();
    let a = right
        .iter()
        .zip(weights)
        .fold(f.zero(), |s, (v, w)| f.add(&s, &f.mul(v, &w)));
    let b = left
        .iter()
        .zip(values)
        .fold(f.zero(), |s, (w, v)| f.add(&s, &f.mul(w, &v)));
    assert_eq!(a, b);
    assert_eq!(
        map.evaluate_bilinear(&weights, &DenseColumns::new(&f, &values))
            .unwrap(),
        a
    );
    let old = right;
    assert!(map.mul_right_into(&values[..1], &mut right).is_err());
    assert_eq!(old, right);
}
