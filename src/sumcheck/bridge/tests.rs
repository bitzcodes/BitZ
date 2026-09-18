use super::{
    PreparedBinding,
    native::{NativeBinding, RowFunctional},
};
use crate::piop::spartan::matrix::{ConstraintMatrices, PreparedConstraintMatrices, eq_table};
use crate::sumcheck::inner::native::{NativeWeights, RawFieldStorage};
use circuit::linear_map::CscMatrix;
use field::{IntegerEmbedding, RingOps, Uint, create_prime_field};

fn expanded(f: &field::FpCtx<2>, out: &NativeWeights) -> Vec<field::Fp<2>> {
    let raw = match out {
        NativeWeights::Dense { matrix, .. } => matrix.clone(),
        NativeWeights::Blocks {
            weights,
            scales,
            num_vars,
            ..
        } => {
            let mut v = vec![0; 1 << num_vars];
            for (k, scale) in scales.scales.iter().enumerate() {
                if let Some(s) = scale {
                    for r in 0..scales.rows {
                        v[k * scales.block_len + r] = f.mul_raw(*s, weights[r]);
                    }
                }
            }
            v
        }
    };
    raw.into_iter()
        .map(|r| crate::sumcheck::inner::native::shared_raw(f, r))
        .collect()
}

#[test]
fn native_reuse_independent_rows_and_partial_domains_match_explicit_matrix() {
    let f = create_prime_field(Uint::<2>::from((1u128 << 127) - 1));
    let e = |x: u64| f.from_integer(&x);
    let mut reusable = NativeWeights::Dense {
        matrix: Vec::new(),
        live: 0,
    };
    for columns in [5, 2, 7, 5] {
        let a = CscMatrix::try_from_rows(
            columns,
            vec![vec![(0, e(3))], vec![(1, e(5))], vec![(0, f.neg(&e(1)))]],
        )
        .unwrap();
        let b = CscMatrix::try_from_rows(columns, vec![vec![(1, e(7))], vec![], vec![(0, e(11))]])
            .unwrap();
        let c = CscMatrix::try_from_rows(columns, vec![vec![], vec![(0, e(13))], vec![(1, e(17))]])
            .unwrap();
        let matrices =
            PreparedConstraintMatrices::new(ConstraintMatrices::new(a, b, c).unwrap(), &f).unwrap();
        let mut binding = NativeBinding::new(&f, &matrices, e(19));
        for seed in [0, 1, 23] {
            let point = [e(seed), e(seed + 1)];
            let rows = RowFunctional::Point(&point);
            binding
                .bind_structured_rows_into(&rows, &mut reusable)
                .unwrap();
            let expected = crate::piop::spartan::matrix::eq_table_prover(&point, matrices.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| matrices.binding(&e(19)).bind_rows(&weights))
                .unwrap();
            assert_eq!(expanded(&f, &reusable), expected.evaluations);
            let column_point = vec![e(7); matrices.num_column_vars()];
            assert_eq!(
                binding
                    .evaluate_structured_at(&rows, &column_point)
                    .unwrap(),
                matrices
                    .structured()
                    .evaluate_equality(&point, &e(19), &column_point)
                    .unwrap()
            );
        }
        let weights = [
            [e(2), e(3), e(5), e(0)],
            [e(7), e(11), e(13), e(0)],
            [e(17), e(19), e(23), e(0)],
        ];
        let rows = RowFunctional::Independent(weights.each_ref().map(|w| w.as_slice()));
        binding
            .bind_structured_rows_into(&rows, &mut reusable)
            .unwrap();
        let actual = expanded(&f, &reusable);
        let m = matrices.matrices();
        let mut expected = vec![f.zero(); 1 << matrices.num_column_vars()];
        for (matrix, w) in [m.a(), m.b(), m.c()].into_iter().zip(&weights) {
            for (j, column) in matrix.columns().enumerate() {
                for (r, c) in column {
                    expected[j] = f.add(&expected[j], &f.mul(&w[r], c));
                }
            }
        }
        assert_eq!(actual, expected);
        let point = vec![e(29); matrices.num_column_vars()];
        let eq = eq_table(&point, &f).unwrap();
        let dot = eq
            .iter()
            .zip(&actual)
            .fold(f.zero(), |s, (a, b)| f.add(&s, &f.mul(a, b)));
        assert_eq!(binding.evaluate_structured_at(&rows, &point).unwrap(), dot);
        let before = actual;
        assert!(
            binding
                .bind_structured_rows_into(
                    &RowFunctional::Explicit(&weights[0][..2]),
                    &mut reusable
                )
                .is_err()
        );
        assert_eq!(expanded(&f, &reusable), before);
    }
}

#[test]
fn typed_sparse_bridge_handles_zero_variable_and_reused_outputs() {
    use circuit::linear_map::PreparedColumns;
    let f = create_prime_field(Uint::<2>::from(97u64));
    let m = CscMatrix::try_from_rows(1, vec![vec![(0, 3u64)]]).unwrap();
    let mut p = PreparedColumns::mixed(&f, &m);
    let mut out = vec![f.one(); 8];
    for value in [7u64, 0, 11] {
        let w = [f.from_integer(&value)];
        p.bind_rows_into(&w[..], &mut out).unwrap();
        assert_eq!(out, vec![f.from_integer(&(3 * value))]);
        assert_eq!(p.evaluate_at(&w[..], &[]).unwrap(), out[0]);
    }
}
