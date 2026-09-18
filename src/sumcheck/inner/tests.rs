use super::*;
use crate::poly::mle::DenseMultilinearExtension;
use crate::sumcheck::UngrindedRoundBoundary;
use crate::transcript::Blake3Transcript;
use field::{Fp, FpCtx, IntegerEmbedding, MergeAccumulator, Uint, Z};

pub(super) fn field() -> FpCtx<2> {
    field::create_prime_field(Uint::from_words([u64::MAX, (1 << 63) - 1]))
}

fn compare_native<T: Copy + Send + Sync>(values: Vec<T>)
where
    FpCtx<2>: PreparedLinearCombination<T> + IntegerEmbedding<T> + BatchMulAcc<Fp<2>, T>,
    FpCtx<2>: Reduce<<FpCtx<2> as BatchMulAcc<Fp<2>, T>>::Accumulator, Output = Fp<2>>,
{
    let f = field();
    let weights: Vec<_> = (0..values.len())
        .map(|i| <FpCtx<2> as IntegerEmbedding<u64>>::from_integer(&f, &(i as u64 * 13 + 7)))
        .collect();
    let projected: Vec<_> = values.iter().map(|v| f.from_integer(v)).collect();
    let claim = <FpCtx<2> as Reduce<field::FpProductAcc<2>>>::reduce(
        &f,
        <FpCtx<2> as BatchMulAcc<Fp<2>>>::batch_mul_acc(&f, &weights, &projected),
    );
    let mut actual_t = Blake3Transcript::new();
    let actual = prove_inner_sumcheck(
        &f,
        &mut actual_t,
        claim,
        values,
        weights.clone(),
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    let mut reference_t = Blake3Transcript::new();
    let expected = reference::prove_inner_sumcheck(
        &mut reference_t,
        claim,
        DenseMultilinearExtension::from_evaluations_vec(
            weights.len().ilog2() as usize,
            weights,
            f.zero(),
        ),
        DenseMultilinearExtension::from_evaluations_vec(
            projected.len().ilog2() as usize,
            projected,
            f.zero(),
        ),
        &f,
    )
    .unwrap();
    assert_eq!(actual.proof, expected.sumcheck.proof);
    assert_eq!(actual.point, expected.sumcheck.eval_points);
    assert_eq!(
        actual.terminal_evaluations,
        [
            expected.batched_matrix_evaluation,
            expected.witness_evaluation
        ]
    );
    assert_eq!(
        squeeze_field::<Fp<2>, _>(&mut actual_t, &f).unwrap(),
        squeeze_field::<Fp<2>, _>(&mut reference_t, &f).unwrap()
    );
    let mut verifier = Blake3Transcript::new();
    assert_eq!(
        actual
            .proof
            .verify(&mut verifier, claim, actual.point.len(), &f)
            .unwrap(),
        (actual.point, actual.final_claim)
    );
}

#[test]
fn native_inputs_match_independent_dense_rounds() {
    compare_native(vec![0u32, u32::MAX, 7, 0, 1, 2, 3, 4]);
    compare_native(vec![u64::MAX, 0, 0, u64::MAX]);
    compare_native(vec![0u128, u128::MAX, u128::MAX, 0, 1, 2, 3, 4]);
    compare_native(vec![
        Z::<2>::from(i128::MIN),
        Z::from(i128::MAX),
        Z::ZERO,
        Z::from(-1i128),
    ]);
    compare_native(vec![
        Uint::from_words([u64::MAX; 4]),
        Uint::ZERO,
        Uint::ONE,
        Uint::from_words([0, 0, 0, 1]),
    ]);
    compare_native(vec![
        Z::from_twos_complement_words([0, 0, 0, 1 << 63]),
        Z::<4>::from(-1i64),
    ]);
    compare_native(vec![9u64]);
    // Exercise Rayon thresholds and a full continuation, not just tiny scalar tables.
    compare_native(
        (0..(1 << 14))
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect(),
    );
}

#[test]
fn shared_challenges_preserve_individual_claims() {
    let f = field();
    let values = [[2u64, 3, 5, 7], [11, 13, 17, 19], [23, 29, 31, 37]].map(Vec::from);
    let weights = core::array::from_fn::<_, 3, _>(|k| {
        (0..4)
            .map(|i| f.from_integer(&((k * 4 + i + 1) as u64)))
            .collect::<Vec<_>>()
    });
    let claims =
        core::array::from_fn(|i| Reduce::reduce(&f, f.batch_mul_acc(&weights[i], &values[i])));
    let out = prove_batched_inner_sumcheck(
        &f,
        &mut Blake3Transcript::new(),
        &claims,
        values,
        weights,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    let mut verifier = Blake3Transcript::new();
    let verified = SumcheckProof::verify_batch_with_round_boundary(
        core::array::from_fn(|i| &out.proofs[i]),
        &mut verifier,
        &claims,
        2,
        &f,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    assert_eq!(verified, (out.point.clone(), out.final_claims));
    for (claim, [w, v]) in out.final_claims.iter().zip(out.terminal_evaluations) {
        assert_eq!(*claim, f.mul(&w, &v));
    }
    let mut corrupt = out.proofs.clone();
    corrupt[1].round_polynomials[0][0] = f.add(&corrupt[1].round_polynomials[0][0], &f.one());
    assert!(
        SumcheckProof::verify_batch_with_round_boundary(
            core::array::from_fn(|i| &corrupt[i]),
            &mut Blake3Transcript::new(),
            &claims,
            2,
            &f,
            &mut UngrindedRoundBoundary
        )
        .is_err()
    );
}

#[test]
fn invalid_shapes_do_not_change_the_transcript_and_singletons_check_the_claim() {
    let f = field();
    let mut transcript = Blake3Transcript::new();
    assert!(
        prove_inner_sumcheck(
            &f,
            &mut transcript,
            f.zero(),
            vec![1u64, 2, 3],
            vec![f.one(); 3],
            &mut UngrindedRoundBoundary
        )
        .is_err()
    );
    assert_eq!(
        squeeze_field::<Fp<2>, _>(&mut transcript, &f).unwrap(),
        squeeze_field::<Fp<2>, _>(&mut Blake3Transcript::new(), &f).unwrap()
    );
    assert_eq!(
        prove_inner_sumcheck(
            &f,
            &mut Blake3Transcript::new(),
            f.zero(),
            vec![7u64],
            vec![f.one()],
            &mut UngrindedRoundBoundary
        ),
        Err(SumcheckError::InvalidTerminalClaim)
    );
    assert!(
        prove_batched_inner_sumcheck::<_, [Vec<u64>; 0]>(
            &f,
            &mut Blake3Transcript::new(),
            &[],
            [],
            [],
            &mut UngrindedRoundBoundary
        )
        .is_err()
    );
}

#[test]
fn streaming_prepared_reduction_matches_batch_and_worker_merges() {
    let f = field();
    let a: Vec<_> = (0..41u64).map(|i| f.from_integer(&i)).collect();
    let b: Vec<_> = (0..41u64).map(|i| f.from_integer(&(i * i + 3))).collect();
    let mut left = field::FpProductAcc::<2>::zero();
    let mut right = field::FpProductAcc::<2>::zero();
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        f.mul_acc(if i % 2 == 0 { &mut left } else { &mut right }, a, b);
    }
    left.merge_assign(&right);
    let reduce = <FpCtx<2> as Reduce<field::FpProductAcc<2>>>::prepare_reduce(&f, a.len());
    assert_eq!(reduce(left), Reduce::reduce(&f, f.batch_mul_acc(&a, &b)));
    let gf = field::Gf128Ops;
    let values: Vec<_> = (0..17u64).map(|v| field::Gf128::from([v, 0])).collect();
    let fixed: Vec<_> = values
        .iter()
        .map(|v| field::PreparedGf128Mul::new(*v))
        .collect();
    let mut a = gf.batch_mul_acc(&values[..8], &fixed[..8]);
    let b = gf.batch_mul_acc(&values[8..], &fixed[8..]);
    a.merge_assign(&b);
    assert_eq!(
        Reduce::reduce(&gf, a),
        Reduce::reduce(&gf, gf.batch_mul_acc(&values, &values))
    );
}

#[test]
fn runtime_batches_match_fixed_batches_and_reject_invalid_metadata() {
    let f = field();
    let values = [vec![2u64, 3, 5, 7], vec![11, 13, 17, 19]];
    let weights = [vec![f.one(); 4], vec![f.from_integer(&3u64); 4]];
    let claims: [_; 2] =
        core::array::from_fn(|i| Reduce::reduce(&f, f.batch_mul_acc(&weights[i], &values[i])));
    let mut fixed_t = Blake3Transcript::new();
    let fixed = prove_batched_inner_sumcheck(
        &f,
        &mut fixed_t,
        &claims,
        values.clone(),
        weights.clone(),
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    let mut dynamic_t = Blake3Transcript::new();
    let dynamic = prove_batched_inner_sumcheck(
        &f,
        &mut dynamic_t,
        &claims[..],
        values.to_vec(),
        weights.to_vec(),
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    assert_eq!(dynamic.proofs, fixed.proofs);
    assert_eq!(dynamic.point, fixed.point);
    assert_eq!(dynamic.final_claims, fixed.final_claims);
    assert_eq!(dynamic.terminal_evaluations, fixed.terminal_evaluations);
    assert_eq!(
        squeeze_field::<Fp<2>, _>(&mut fixed_t, &f).unwrap(),
        squeeze_field::<Fp<2>, _>(&mut dynamic_t, &f).unwrap()
    );

    for (v, w, claims) in [
        (
            values.to_vec(),
            weights.to_vec(),
            InitialClaims::Known(&claims[..1]),
        ),
        (
            values.to_vec(),
            weights[..1].to_vec(),
            InitialClaims::Known(&claims),
        ),
        (
            vec![vec![2u64; 2], vec![3u64; 4]],
            vec![vec![f.one(); 2], vec![f.one(); 4]],
            InitialClaims::Known(&claims),
        ),
        (values.to_vec(), weights.to_vec(), InitialClaims::Compute),
    ] {
        let mut t = Blake3Transcript::new();
        assert_eq!(
            prove_batched_inner_sumcheck(&f, &mut t, claims, v, w, &mut UngrindedRoundBoundary),
            Err(SumcheckError::InvalidProductDimensions)
        );
        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut t, &f).unwrap(),
            squeeze_field::<Fp<2>, _>(&mut Blake3Transcript::new(), &f).unwrap()
        );
    }
}

#[test]
fn native_dense_and_block_states_share_one_batch() {
    use native::{BlockScales, NativeWeights, RawFieldStorage, RawWitness};
    let f = field();
    let values: Vec<u64> = (1..=16).collect();
    let common: Vec<_> = (1..=4u64).map(|v| f.from_integer(&v)).collect();
    let scales = [
        f.zero(),
        f.one(),
        f.from_integer(&3u64),
        f.from_integer(&9u64),
    ];
    let expanded: Vec<_> = scales
        .iter()
        .flat_map(|s| common.iter().map(|w| f.mul(s, w)))
        .collect();
    let claim = Reduce::reduce(&f, f.batch_mul_acc(&expanded, &values));
    let mut native_t = Blake3Transcript::new();
    let native = prove_batched_inner_sumcheck(
        &f,
        &mut native_t,
        &[claim; 2],
        [
            RawWitness::native_borrowed(&values, 16),
            RawWitness::native_borrowed(&values, 16),
        ],
        [
            NativeWeights::Dense {
                matrix: expanded.iter().map(|v| f.raw(v)).collect(),
                live: 16,
            },
            NativeWeights::Blocks {
                weights: common.iter().map(|v| f.raw(v)).collect(),
                scales: BlockScales {
                    block_len: 4,
                    rows: 4,
                    scales: scales.iter().map(|v| Some(f.raw(v))).collect(),
                },
                live: 16,
                num_vars: 4,
            },
        ],
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    let mut dense_t = Blake3Transcript::new();
    let dense = prove_batched_inner_sumcheck(
        &f,
        &mut dense_t,
        &[claim; 2],
        [values.clone(), values],
        [expanded.clone(), expanded],
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    assert_eq!(native, dense);
    assert_eq!(
        squeeze_field::<Fp<2>, _>(&mut native_t, &f).unwrap(),
        squeeze_field::<Fp<2>, _>(&mut dense_t, &f).unwrap()
    );
}
