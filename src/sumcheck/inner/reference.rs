use crate::piop::spartan::sumcheck::{InnerSumcheckOutput, R1csProductMles};
pub(crate) use crate::sumcheck::arithmetic::*;
#[cfg(test)]
pub(crate) use crate::sumcheck::outer::ordinary::*;
pub use crate::sumcheck::proof::OuterSumcheckProof;
pub use crate::sumcheck::{SumcheckError, SumcheckProof};
pub(crate) use crate::sumcheck::{boundary::*, proof::*};
use field::RingOps;
use field::{BatchMulAcc, MergeAccumulator, Reduce};
#[cfg(test)]
use field::{Fp, Uint};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{poly::mle::DenseMultilinearExtension, transcript::traits::Transcript};

#[cfg(test)]
use crate::piop::spartan::grinding::{GrindingError, GrindingRound};
use crate::piop::spartan::{SpartanField, grinding::GrindingDomain};
#[cfg(test)]
use crate::piop::spartan::{absorb_field_elements, squeeze_field};

/// `initial_claim = sum_y batched_matrix(y) * witness(y)`.
#[cfg(test)]
pub(crate) fn prove_inner_sumcheck<F>(
    transcript: &mut impl Transcript,
    initial_claim: F,
    batched_matrix_mle: DenseMultilinearExtension<F>,
    witness_mle: DenseMultilinearExtension<F>,
    field_cfg: &F::Config,
) -> Result<InnerSumcheckOutput<F>, SumcheckError>
where
    F: SpartanField,
{
    let reducer = field_cfg.clone();
    prove_inner_sumcheck_with_reducer(
        transcript,
        initial_claim,
        batched_matrix_mle,
        witness_mle,
        field_cfg,
        &reducer,
    )
}

pub(crate) fn prove_inner_sumcheck_with_reducer<F, R>(
    transcript: &mut impl Transcript,
    initial_claim: F,
    batched_matrix_mle: DenseMultilinearExtension<F>,
    witness_mle: DenseMultilinearExtension<F>,
    field_cfg: &F::Config,
    reducer: &R,
) -> Result<InnerSumcheckOutput<F>, SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    let mut round_boundary = UngrindedRoundBoundary;
    prove_inner_sumcheck_with_reducer_and_round_boundary(
        transcript,
        initial_claim,
        batched_matrix_mle,
        witness_mle,
        field_cfg,
        reducer,
        &mut round_boundary,
    )
}

/// Proves an ordinary quadratic inner sumcheck with a typed grinding boundary
/// after every round message. `round_offset` lets a caller prepend
/// transcript-identical rounds computed by another prover kernel while keeping
/// the grinding round indices globally consecutive.
#[allow(dead_code)]
pub(crate) fn prove_inner_sumcheck_with_reducer_grinded<D, F, R>(
    transcript: &mut impl Transcript,
    initial_claim: F,
    batched_matrix_mle: DenseMultilinearExtension<F>,
    witness_mle: DenseMultilinearExtension<F>,
    field_cfg: &F::Config,
    reducer: &R,
    grinding_bits: u32,
    round_offset: usize,
) -> Result<(InnerSumcheckOutput<F>, Vec<u64>), SumcheckError>
where
    D: GrindingDomain,
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    let mut round_boundary =
        ProverGrindingRoundBoundary::<D>::with_round_offset(grinding_bits, round_offset);
    let output = prove_inner_sumcheck_with_reducer_and_round_boundary(
        transcript,
        initial_claim,
        batched_matrix_mle,
        witness_mle,
        field_cfg,
        reducer,
        &mut round_boundary,
    )?;
    Ok((output, round_boundary.nonces))
}

fn prove_inner_sumcheck_with_reducer_and_round_boundary<F, R, P>(
    transcript: &mut impl Transcript,
    initial_claim: F,
    batched_matrix_mle: DenseMultilinearExtension<F>,
    witness_mle: DenseMultilinearExtension<F>,
    field_cfg: &F::Config,
    reducer: &R,
    round_boundary: &mut P,
) -> Result<InnerSumcheckOutput<F>, SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
    P: RoundBoundaryPolicy,
{
    let num_vars = batched_matrix_mle.num_vars;
    if !has_dense_shape(&batched_matrix_mle)
        || !has_dense_shape(&witness_mle)
        || witness_mle.num_vars != num_vars
    {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    round_boundary.validate(num_vars)?;

    let zero = F::zero_with_cfg(field_cfg);
    let mut batched_matrix = batched_matrix_mle.evaluations;
    let mut witness = witness_mle.evaluations;
    let mut current_claim = initial_claim;
    let mut eval_points = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);

    if num_vars > 0 {
        let mut batched_matrix_scratch = vec![zero.clone(); batched_matrix.len() / 2];
        let mut witness_scratch = vec![zero.clone(); witness.len() / 2];
        let mut coefficients_without_linear = sum_inner_round_coefficients_without_linear(
            &batched_matrix,
            &witness,
            reducer,
            &field_cfg,
        )?;

        for _round in 0..num_vars {
            let challenge = recover_full_round_polynomial_and_sample_next_challenge_with_boundary(
                transcript,
                &mut current_claim,
                &coefficients_without_linear,
                &mut round_polynomials,
                &mut eval_points,
                &zero,
                field_cfg,
                round_boundary,
            )?;

            let next_len = batched_matrix.len() / 2;
            debug_assert_eq!(witness.len() / 2, next_len);
            debug_assert!(batched_matrix_scratch.len() >= next_len);
            debug_assert!(witness_scratch.len() >= next_len);
            batched_matrix_scratch.truncate(next_len);
            witness_scratch.truncate(next_len);

            if next_len == 1 {
                batched_matrix_scratch[0] = interpolate_pair(
                    &batched_matrix[0],
                    &batched_matrix[1],
                    &challenge,
                    &field_cfg,
                );
                witness_scratch[0] =
                    interpolate_pair(&witness[0], &witness[1], &challenge, &field_cfg);
            } else {
                coefficients_without_linear =
                    fold_and_compute_next_inner_round_coefficients_without_linear(
                        &batched_matrix,
                        &witness,
                        &mut batched_matrix_scratch,
                        &mut witness_scratch,
                        &challenge,
                        reducer,
                        &field_cfg,
                    )?;
            }

            std::mem::swap(&mut batched_matrix, &mut batched_matrix_scratch);
            std::mem::swap(&mut witness, &mut witness_scratch);
        }
    }

    let batched_matrix_evaluation = batched_matrix[0].clone();
    let witness_evaluation = witness[0].clone();
    debug_assert_eq!(
        current_claim,
        (field_cfg).mul(&batched_matrix_evaluation, &witness_evaluation)
    );

    Ok(InnerSumcheckOutput {
        sumcheck: SumcheckProverOutput {
            proof: SumcheckProof { round_polynomials },
            eval_points,
            final_claim: current_claim,
        },
        batched_matrix_evaluation,
        witness_evaluation,
    })
}

/// Proves the first inner round with the exact native u32 assignment, then
/// continues with field-valued witness and matrix tables using the fixed
/// production policy: delayed coefficients and delayed native folding.
#[cfg(test)]
pub(crate) fn prove_inner_sumcheck_u32_native_with_reducer<R>(
    transcript: &mut impl Transcript,
    initial_claim: Fp<2>,
    batched_matrix_mle: DenseMultilinearExtension<Fp<2>>,
    witness_mle: DenseMultilinearExtension<u64>,
    field_cfg: &field::FpCtx<2>,
    reducer: &R,
) -> Result<InnerSumcheckOutput<Fp<2>>, SumcheckError>
where
    R: BatchMulAcc<Fp<2>>
        + Reduce<<R as BatchMulAcc<Fp<2>>>::Accumulator, Output = Fp<2>>
        + Sync
        + SumcheckLinearReducer,
{
    let num_vars = batched_matrix_mle.num_vars;
    if !has_dense_shape(&batched_matrix_mle)
        || !has_dense_shape(&witness_mle)
        || witness_mle.num_vars != num_vars
    {
        return Err(SumcheckError::InvalidProductDimensions);
    }

    let zero = Fp::<2>::zero_with_cfg(field_cfg);
    let mut batched_matrix = batched_matrix_mle.evaluations;
    let native_witness = witness_mle.evaluations;
    let mut current_claim = initial_claim;
    let mut eval_points = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);

    if num_vars == 0 {
        let batched_matrix_evaluation = batched_matrix[0].clone();
        let witness_evaluation = native_to_field(native_witness[0], field_cfg);
        debug_assert_eq!(
            current_claim,
            (field_cfg).mul(&batched_matrix_evaluation, &witness_evaluation)
        );
        return Ok(InnerSumcheckOutput {
            sumcheck: SumcheckProverOutput {
                proof: SumcheckProof { round_polynomials },
                eval_points,
                final_claim: current_claim,
            },
            batched_matrix_evaluation,
            witness_evaluation,
        });
    }

    let coefficients_without_linear = {
        let _scope = tracing::info_span!("spartan:inner_native_coefficients").entered();
        sum_u32_native_inner_coefficients_without_linear(
            &batched_matrix,
            &native_witness,
            &zero,
            reducer,
            field_cfg,
        )?
    };
    let challenge = recover_full_round_polynomial_and_sample_next_challenge(
        transcript,
        &mut current_claim,
        &coefficients_without_linear,
        &mut round_polynomials,
        &mut eval_points,
        &zero,
        field_cfg,
    )?;

    let next_len = batched_matrix.len() / 2;
    let mut folded_matrix = vec![zero.clone(); next_len];
    fold_table(&batched_matrix, &mut folded_matrix, &challenge, &field_cfg);
    let mut witness = vec![zero.clone(); next_len];
    {
        let _scope = tracing::info_span!("spartan:inner_native_witness_fold").entered();
        fold_u64_table_to_field(
            &native_witness,
            &mut witness,
            &challenge,
            &zero,
            reducer,
            field_cfg,
        )?;
    }
    batched_matrix = folded_matrix;

    {
        let _scope = tracing::info_span!("spartan:inner_field_rounds").entered();
        let mut coefficients_without_linear = if next_len > 1 {
            sum_inner_round_coefficients_without_linear(
                &batched_matrix,
                &witness,
                reducer,
                field_cfg,
            )?
        } else {
            std::array::from_fn(|_| zero.clone())
        };
        let mut batched_matrix_scratch = vec![zero.clone(); batched_matrix.len() / 2];
        let mut witness_scratch = vec![zero.clone(); witness.len() / 2];

        while batched_matrix.len() > 1 {
            let challenge = recover_full_round_polynomial_and_sample_next_challenge(
                transcript,
                &mut current_claim,
                &coefficients_without_linear,
                &mut round_polynomials,
                &mut eval_points,
                &zero,
                field_cfg,
            )?;

            let next_len = batched_matrix.len() / 2;
            batched_matrix_scratch.truncate(next_len);
            witness_scratch.truncate(next_len);

            if next_len == 1 {
                batched_matrix_scratch[0] = interpolate_pair(
                    &batched_matrix[0],
                    &batched_matrix[1],
                    &challenge,
                    &field_cfg,
                );
                witness_scratch[0] =
                    interpolate_pair(&witness[0], &witness[1], &challenge, &field_cfg);
            } else {
                coefficients_without_linear =
                    fold_and_compute_next_inner_round_coefficients_without_linear(
                        &batched_matrix,
                        &witness,
                        &mut batched_matrix_scratch,
                        &mut witness_scratch,
                        &challenge,
                        reducer,
                        field_cfg,
                    )?;
            }

            std::mem::swap(&mut batched_matrix, &mut batched_matrix_scratch);
            std::mem::swap(&mut witness, &mut witness_scratch);
        }
    }

    let batched_matrix_evaluation = batched_matrix[0].clone();
    let witness_evaluation = witness[0].clone();
    debug_assert_eq!(
        current_claim,
        (field_cfg).mul(&batched_matrix_evaluation, &witness_evaluation)
    );

    Ok(InnerSumcheckOutput {
        sumcheck: SumcheckProverOutput {
            proof: SumcheckProof { round_polynomials },
            eval_points,
            final_claim: current_claim,
        },
        batched_matrix_evaluation,
        witness_evaluation,
    })
}

#[inline]
fn accumulate_inner_pair_coefficients_without_linear<F, R>(
    accumulators: &mut [<R as BatchMulAcc<F>>::Accumulator; 2],
    matrix_zero: &F,
    matrix_one: &F,
    witness_zero: &F,
    witness_one: &F,
    reducer: &R,
    field_config: &F::Config,
) where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    reducer.mul_acc(&mut accumulators[0], matrix_zero, witness_zero);
    reducer.mul_acc(
        &mut accumulators[1],
        &(field_config).sub(matrix_one, matrix_zero),
        &(field_config).sub(witness_one, witness_zero),
    );
}

fn sum_inner_round_coefficients_without_linear<F, R>(
    batched_matrix: &[F],
    witness: &[F],
    reducer: &R,
    field_config: &F::Config,
) -> Result<[F; 2], SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    debug_assert_eq!(batched_matrix.len(), witness.len());
    debug_assert!(batched_matrix.len() >= 2);

    #[cfg(feature = "parallel")]
    if should_parallelize(batched_matrix.len() / 2) {
        let accumulators = batched_matrix
            .par_chunks_exact(2)
            .zip(witness.par_chunks_exact(2))
            .fold(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |mut accumulators, (matrix, witness)| {
                    accumulate_inner_pair_coefficients_without_linear(
                        &mut accumulators,
                        &matrix[0],
                        &matrix[1],
                        &witness[0],
                        &witness[1],
                        reducer,
                        &field_config,
                    );
                    accumulators
                },
            )
            .reduce(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |left, right| merge_accumulators(left, right),
            );
        return reduce_two_accumulators(accumulators, reducer, &field_config);
    }

    let accumulators = batched_matrix
        .chunks_exact(2)
        .zip(witness.chunks_exact(2))
        .fold(
            std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
            |mut accumulators, (matrix, witness)| {
                accumulate_inner_pair_coefficients_without_linear(
                    &mut accumulators,
                    &matrix[0],
                    &matrix[1],
                    &witness[0],
                    &witness[1],
                    reducer,
                    &field_config,
                );
                accumulators
            },
        );
    reduce_two_accumulators(accumulators, reducer, &field_config)
}

#[inline]
fn fold_inner_chunk<F, R>(
    batched_matrix: &[F],
    witness: &[F],
    batched_matrix_output: &mut [F],
    witness_output: &mut [F],
    challenge: &F,
    reducer: &R,
    field_config: &F::Config,
) -> [<R as BatchMulAcc<F>>::Accumulator; 2]
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    debug_assert_eq!(batched_matrix.len(), 4);
    debug_assert_eq!(witness.len(), 4);
    debug_assert_eq!(batched_matrix_output.len(), 2);
    debug_assert_eq!(witness_output.len(), 2);

    let folded_matrix = [
        interpolate_pair(
            &batched_matrix[0],
            &batched_matrix[1],
            challenge,
            &field_config,
        ),
        interpolate_pair(
            &batched_matrix[2],
            &batched_matrix[3],
            challenge,
            &field_config,
        ),
    ];
    let folded_witness = [
        interpolate_pair(&witness[0], &witness[1], challenge, &field_config),
        interpolate_pair(&witness[2], &witness[3], challenge, &field_config),
    ];

    batched_matrix_output.clone_from_slice(&folded_matrix);
    witness_output.clone_from_slice(&folded_witness);
    let mut accumulators = std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero());
    accumulate_inner_pair_coefficients_without_linear(
        &mut accumulators,
        &folded_matrix[0],
        &folded_matrix[1],
        &folded_witness[0],
        &folded_witness[1],
        reducer,
        &field_config,
    );
    accumulators
}

/// Folds both active inner tables and prepares the next round's `[c0, c2]`.
fn fold_and_compute_next_inner_round_coefficients_without_linear<F, R>(
    batched_matrix: &[F],
    witness: &[F],
    batched_matrix_output: &mut [F],
    witness_output: &mut [F],
    challenge: &F,
    reducer: &R,
    field_config: &F::Config,
) -> Result<[F; 2], SumcheckError>
where
    F: SpartanField,
    R: BatchMulAcc<F> + Reduce<<R as BatchMulAcc<F>>::Accumulator, Output = F> + Sync,
{
    debug_assert_eq!(batched_matrix.len(), witness.len());
    debug_assert!(batched_matrix.len() >= 4);
    debug_assert_eq!(batched_matrix_output.len(), batched_matrix.len() / 2);
    debug_assert_eq!(witness_output.len(), witness.len() / 2);

    #[cfg(feature = "parallel")]
    if should_parallelize(batched_matrix.len() / 4) {
        let accumulators = batched_matrix
            .par_chunks_exact(4)
            .zip(witness.par_chunks_exact(4))
            .zip(batched_matrix_output.par_chunks_exact_mut(2))
            .zip(witness_output.par_chunks_exact_mut(2))
            .fold(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |accumulators, (((matrix, witness), matrix_output), witness_output)| {
                    let contribution = fold_inner_chunk(
                        matrix,
                        witness,
                        matrix_output,
                        witness_output,
                        challenge,
                        reducer,
                        &field_config,
                    );
                    merge_accumulators(accumulators, contribution)
                },
            )
            .reduce(
                || std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
                |left, right| merge_accumulators(left, right),
            );
        return reduce_two_accumulators(accumulators, reducer, &field_config);
    }

    let accumulators = batched_matrix
        .chunks_exact(4)
        .zip(witness.chunks_exact(4))
        .zip(batched_matrix_output.chunks_exact_mut(2))
        .zip(witness_output.chunks_exact_mut(2))
        .fold(
            std::array::from_fn(|_| <R as BatchMulAcc<F>>::Accumulator::zero()),
            |accumulators, (((matrix, witness), matrix_output), witness_output)| {
                let contribution = fold_inner_chunk(
                    matrix,
                    witness,
                    matrix_output,
                    witness_output,
                    challenge,
                    reducer,
                    &field_config,
                );
                merge_accumulators(accumulators, contribution)
            },
        );
    reduce_two_accumulators(accumulators, reducer, &field_config)
}

#[cfg(test)]
fn sum_u32_native_inner_coefficients_without_linear<R>(
    batched_matrix: &[Fp<2>],
    witness: &[u64],
    zero: &Fp<2>,
    reducer: &R,
    field_config: &crate::piop::spartan::protocol::FieldConfig,
) -> Result<[Fp<2>; 2], SumcheckError>
where
    R: SumcheckLinearReducer,
{
    debug_assert_eq!(batched_matrix.len(), witness.len());
    debug_assert!(batched_matrix.len() >= 2);

    let pair_count = batched_matrix.len() / 2;
    let accumulators = sum_linear_accumulators(
        pair_count,
        |accumulators, pair| {
            let index = 2 * pair;
            let matrix_delta =
                (field_config).sub(&batched_matrix[index + 1], &batched_matrix[index]);
            let neg_matrix_delta = (field_config).sub(zero, &matrix_delta);

            <R as SumcheckLinearReducer>::multiply_accumulate(
                reducer,
                &mut accumulators[0],
                &batched_matrix[index],
                &witness[index],
            );
            // (m1-m0)(w1-w0) = (m1-m0)w1 + (m0-m1)w0.
            <R as SumcheckLinearReducer>::multiply_accumulate(
                reducer,
                &mut accumulators[1],
                &matrix_delta,
                &witness[index + 1],
            );
            <R as SumcheckLinearReducer>::multiply_accumulate(
                reducer,
                &mut accumulators[1],
                &neg_matrix_delta,
                &witness[index],
            );
        },
        reducer,
    );

    let [c0, c2] = accumulators;
    Ok([
        <R as SumcheckLinearReducer>::reduce(reducer, c0, &field_config)?,
        <R as SumcheckLinearReducer>::reduce(reducer, c2, &field_config)?,
    ])
}

#[cfg(test)]
mod tests {

    use crate::{
        piop::spartan::{
            grinding::{derive_grinding_seed, grinding_nonce_is_valid},
            matrix::eq_table,
        },
        transcript::Blake3Transcript,
    };

    use super::*;

    const TEST_MODULUS: u128 = (1_u128 << 100) - 15;

    fn config() -> <Fp<2> as crate::piop::spartan::SpartanField>::Config {
        Fp::<2>::make_cfg(&Uint::from(TEST_MODULUS)).expect("prime test modulus")
    }

    fn field(
        value: u64,
        field_cfg: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> Fp<2> {
        Fp::<2>::from_with_cfg(value, field_cfg)
    }

    enum TestOuterGrinding {}

    impl GrindingDomain for TestOuterGrinding {
        const DOMAIN: &'static [u8] = b"test/spartan/outer-sumcheck-grinding/v1";
    }

    type OuterTestInstance = (
        Fp<2>,
        Vec<Fp<2>>,
        (
            DenseMultilinearExtension<Fp<2>>,
            DenseMultilinearExtension<Fp<2>>,
        ),
        R1csProductMles<Fp<2>>,
    );

    fn outer_test_instance(
        num_vars: usize,
        field_cfg: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
    ) -> OuterTestInstance {
        let zero = Fp::<2>::zero_with_cfg(field_cfg);
        let table_len = 1_usize << num_vars;
        let products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(index as u64 + 2, field_cfg))
                    .collect(),
                zero.clone(),
            ),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(3 * index as u64 + 5, field_cfg))
                    .collect(),
                zero.clone(),
            ),
            cz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field((index * index) as u64 + 7, field_cfg))
                    .collect(),
                zero,
            ),
        };
        let tau = (0..num_vars)
            .map(|index| field(2 * index as u64 + 2, field_cfg))
            .collect::<Vec<_>>();
        let full_equality = eq_table(&tau, field_cfg).unwrap();
        let mut initial_claim = Fp::<2>::zero_with_cfg(field_cfg);
        for (index, equality) in full_equality.iter().enumerate() {
            let residual = (field_cfg).sub(
                &(field_cfg).mul(
                    &products.az.evaluations[index],
                    &products.bz.evaluations[index],
                ),
                &products.cz.evaluations[index],
            );
            initial_claim =
                field_cfg.add(&(initial_claim), &(&(field_cfg).mul(equality, &residual)));
        }

        let split = num_vars / 2;
        let (low_point, high_point) = tau.split_at(split);
        let equality_factors = (
            DenseMultilinearExtension {
                evaluations: eq_table(low_point, field_cfg).unwrap(),
                num_vars: low_point.len(),
            },
            DenseMultilinearExtension {
                evaluations: eq_table(high_point, field_cfg).unwrap(),
                num_vars: high_point.len(),
            },
        );
        (initial_claim, tau, equality_factors, products)
    }

    #[test]
    fn one_variable_inner_sumcheck_has_the_expected_quadratic_and_replays() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let batched_matrix = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![field(2, &field_cfg), field(5, &field_cfg)],
            zero.clone(),
        );
        let witness = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![field(3, &field_cfg), field(7, &field_cfg)],
            zero,
        );
        let initial_claim = field(41, &field_cfg);
        let mut prover_transcript = Blake3Transcript::new();

        let output = prove_inner_sumcheck(
            &mut prover_transcript,
            initial_claim.clone(),
            batched_matrix,
            witness,
            &field_cfg,
        )
        .unwrap();

        assert_eq!(
            output.sumcheck.proof.round_polynomials,
            vec![[
                field(6, &field_cfg),
                field(17, &field_cfg),
                field(12, &field_cfg),
            ]]
        );

        let mut verifier_transcript = Blake3Transcript::new();
        let (point, final_claim) = output
            .sumcheck
            .proof
            .verify(&mut verifier_transcript, initial_claim, 1, &field_cfg)
            .unwrap();
        assert_eq!(point, output.sumcheck.eval_points);
        assert_eq!(final_claim, output.sumcheck.final_claim);
        assert_eq!(
            final_claim,
            (field_cfg).mul(
                &output.batched_matrix_evaluation,
                &output.witness_evaluation,
            )
        );
    }

    #[test]
    fn zero_variable_inner_sumcheck_proves_and_verifies_without_moving_transcript() {
        let field_cfg = config();
        let initial_claim = field(35, &field_cfg);
        let mut prover_transcript = Blake3Transcript::new();

        let output = prove_inner_sumcheck(
            &mut prover_transcript,
            initial_claim.clone(),
            DenseMultilinearExtension::zero_vars(field(5, &field_cfg)),
            DenseMultilinearExtension::zero_vars(field(7, &field_cfg)),
            &field_cfg,
        )
        .unwrap();

        assert!(output.sumcheck.proof.round_polynomials.is_empty());
        assert!(output.sumcheck.eval_points.is_empty());
        assert_eq!(output.sumcheck.final_claim, initial_claim);
        assert_eq!(output.batched_matrix_evaluation, field(5, &field_cfg));
        assert_eq!(output.witness_evaluation, field(7, &field_cfg));

        let mut verifier_transcript = Blake3Transcript::new();
        let (point, final_claim) = output
            .sumcheck
            .proof
            .verify(
                &mut verifier_transcript,
                initial_claim.clone(),
                0,
                &field_cfg,
            )
            .unwrap();
        assert!(point.is_empty());
        assert_eq!(final_claim, initial_claim);

        let prover_next = squeeze_field::<Fp<2>, _>(&mut prover_transcript, &field_cfg).unwrap();
        let verifier_next =
            squeeze_field::<Fp<2>, _>(&mut verifier_transcript, &field_cfg).unwrap();
        let mut fresh_transcript = Blake3Transcript::new();
        let fresh_next = squeeze_field::<Fp<2>, _>(&mut fresh_transcript, &field_cfg).unwrap();
        assert_eq!(prover_next, fresh_next);
        assert_eq!(verifier_next, fresh_next);
    }

    #[test]
    fn zero_variable_outer_sumcheck_proves_and_verifies() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let one = Fp::<2>::one_with_cfg(&field_cfg);
        let products = R1csProductMles {
            az: DenseMultilinearExtension::zero_vars(field(2, &field_cfg)),
            bz: DenseMultilinearExtension::zero_vars(field(3, &field_cfg)),
            cz: DenseMultilinearExtension::zero_vars(field(6, &field_cfg)),
        };
        let equality_factors = (
            DenseMultilinearExtension::zero_vars(one.clone()),
            DenseMultilinearExtension::zero_vars(one),
        );
        let mut prover_transcript = Blake3Transcript::new();

        let output =
            crate::sumcheck::outer::EqualityFactors::from_mles(equality_factors, &field_cfg)
                .and_then(|factors| {
                    crate::sumcheck::outer::prove_outer_sumcheck(
                        &field_cfg,
                        &mut prover_transcript,
                        crate::sumcheck::outer::OuterClaim::Sum(zero.clone()),
                        &[],
                        products,
                        Some(factors),
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                })
                .map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                .unwrap();

        assert!(output.proof.sumcheck.round_polynomials.is_empty());
        assert!(output.eval_points.is_empty());
        assert_eq!(output.final_claim, zero);
        assert_eq!(output.proof.az_mle_claim, field(2, &field_cfg));
        assert_eq!(output.proof.bz_mle_claim, field(3, &field_cfg));
        assert_eq!(output.proof.cz_mle_claim, field(6, &field_cfg));

        let mut verifier_transcript = Blake3Transcript::new();
        let verified = output
            .proof
            .verify(
                &mut verifier_transcript,
                Fp::<2>::zero_with_cfg(&field_cfg),
                &[],
                &field_cfg,
            )
            .unwrap();
        assert!(verified.eval_points.is_empty());
        assert_eq!(verified.az_mle_claim, field(2, &field_cfg));
        assert_eq!(verified.bz_mle_claim, field(3, &field_cfg));
        assert_eq!(verified.cz_mle_claim, field(6, &field_cfg));

        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut prover_transcript, &field_cfg).unwrap(),
            squeeze_field::<Fp<2>, _>(&mut verifier_transcript, &field_cfg).unwrap()
        );
    }

    #[test]
    fn grinded_outer_sumcheck_replays_and_continues_in_lockstep() {
        const NUM_VARS: usize = 4;
        const GRINDING_BITS: u32 = 8;

        let field_cfg = config();
        let (initial_claim, tau, equality_factors, products) =
            outer_test_instance(NUM_VARS, &field_cfg);
        let mut prover_transcript = Blake3Transcript::new();
        let mut boundary =
            ProverGrindingRoundBoundary::<TestOuterGrinding>::with_round_offset(GRINDING_BITS, 0);
        let output: crate::sumcheck::proof::OuterSumcheckOutput<_> =
            crate::sumcheck::outer::prove_outer_sumcheck(
                &field_cfg,
                &mut prover_transcript,
                crate::sumcheck::outer::OuterClaim::Sum(initial_claim),
                &tau,
                products,
                Some(
                    crate::sumcheck::outer::EqualityFactors::from_mles(
                        equality_factors,
                        &field_cfg,
                    )
                    .unwrap(),
                ),
                &mut boundary,
            )
            .unwrap()
            .into();
        let nonces = boundary.into_nonces();

        assert_eq!(nonces.len(), NUM_VARS);
        assert_eq!(nonces.len(), output.proof.sumcheck.round_polynomials.len());

        let mut verifier_transcript = Blake3Transcript::new();
        let verified = output
            .proof
            .verify_grinded::<TestOuterGrinding>(
                &mut verifier_transcript,
                initial_claim,
                &tau,
                &field_cfg,
                &nonces,
                GRINDING_BITS,
            )
            .unwrap();

        assert_eq!(verified.eval_points, output.eval_points);
        assert_eq!(verified.az_mle_claim, output.proof.az_mle_claim);
        assert_eq!(verified.bz_mle_claim, output.proof.bz_mle_claim);
        assert_eq!(verified.cz_mle_claim, output.proof.cz_mle_claim);
        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut prover_transcript, &field_cfg).unwrap(),
            squeeze_field::<Fp<2>, _>(&mut verifier_transcript, &field_cfg).unwrap(),
        );
    }

    #[test]
    fn outer_verifier_rejects_malformed_runtime_field_elements_before_absorption() {
        const NUM_VARS: usize = 2;

        let field_cfg = config();
        let other_cfg = Fp::<2>::make_cfg(&Uint::from((1_u128 << 127) - 1)).unwrap();
        let (initial_claim, tau, equality_factors, products) =
            outer_test_instance(NUM_VARS, &field_cfg);
        let output =
            crate::sumcheck::outer::EqualityFactors::from_mles(equality_factors, &field_cfg)
                .and_then(|factors| {
                    crate::sumcheck::outer::prove_outer_sumcheck(
                        &field_cfg,
                        &mut Blake3Transcript::new(),
                        crate::sumcheck::outer::OuterClaim::Sum(initial_claim.clone()),
                        &tau,
                        products,
                        Some(factors),
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                })
                .map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                .unwrap();

        let mut foreign_round = output.proof.clone();
        foreign_round.sumcheck.round_polynomials[0][0] =
            crate::piop::spartan::noncanonical_test_value(&field_cfg);
        let mut actual = Blake3Transcript::new();
        let mut untouched = actual.clone();
        assert_eq!(
            foreign_round.verify(&mut actual, initial_claim.clone(), &tau, &field_cfg),
            Err(SumcheckError::NonCanonicalFieldElement)
        );
        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut actual, &field_cfg).unwrap(),
            squeeze_field::<Fp<2>, _>(&mut untouched, &field_cfg).unwrap()
        );

        let mut malformed_terminal = output.proof;
        malformed_terminal.az_mle_claim = field::FpCtx::from_prime_u128(u128::MAX - 158)
            .from_montgomery_integer(*field_cfg.modulus());
        let mut actual = Blake3Transcript::new();
        let mut untouched = actual.clone();
        assert_eq!(
            malformed_terminal.verify(&mut actual, initial_claim, &tau, &field_cfg),
            Err(SumcheckError::NonCanonicalFieldElement)
        );
        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut actual, &field_cfg).unwrap(),
            squeeze_field::<Fp<2>, _>(&mut untouched, &field_cfg).unwrap()
        );
    }

    #[test]
    fn grinded_outer_sumcheck_rejects_invalid_and_miscounted_nonces() {
        const NUM_VARS: usize = 3;
        const GRINDING_BITS: u32 = 8;

        let field_cfg = config();
        let (initial_claim, tau, equality_factors, products) =
            outer_test_instance(NUM_VARS, &field_cfg);
        let mut prover_transcript = Blake3Transcript::new();
        let mut boundary =
            ProverGrindingRoundBoundary::<TestOuterGrinding>::with_round_offset(GRINDING_BITS, 0);
        let output: crate::sumcheck::proof::OuterSumcheckOutput<_> =
            crate::sumcheck::outer::prove_outer_sumcheck(
                &field_cfg,
                &mut prover_transcript,
                crate::sumcheck::outer::OuterClaim::Sum(initial_claim),
                &tau,
                products,
                Some(
                    crate::sumcheck::outer::EqualityFactors::from_mles(
                        equality_factors,
                        &field_cfg,
                    )
                    .unwrap(),
                ),
                &mut boundary,
            )
            .unwrap()
            .into();
        let nonces = boundary.into_nonces();

        let mut short_transcript = Blake3Transcript::new();
        let mut untouched_transcript = short_transcript.clone();
        assert_eq!(
            output.proof.verify_grinded::<TestOuterGrinding>(
                &mut short_transcript,
                initial_claim.clone(),
                &tau,
                &field_cfg,
                &nonces[..NUM_VARS - 1],
                GRINDING_BITS,
            ),
            Err(SumcheckError::InvalidGrindingNonceCount {
                expected: NUM_VARS,
                actual: NUM_VARS - 1,
            })
        );
        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut short_transcript, &field_cfg).unwrap(),
            squeeze_field::<Fp<2>, _>(&mut untouched_transcript, &field_cfg).unwrap(),
        );

        // Reconstruct the first boundary seed and choose a nonce that is
        // definitely invalid rather than relying on `valid_nonce + 1`.
        let mut seed_transcript = Blake3Transcript::new();
        absorb_field_elements(
            &mut seed_transcript,
            &output.proof.sumcheck.round_polynomials[0],
            &field_cfg,
        );
        let seed = derive_grinding_seed::<TestOuterGrinding, _>(
            &mut seed_transcript,
            GrindingRound::new(0),
            GRINDING_BITS,
        )
        .unwrap();
        let invalid_nonce = (0..=u64::MAX)
            .find(|&nonce| {
                nonce != nonces[0] && !grinding_nonce_is_valid(&seed, nonce, GRINDING_BITS).unwrap()
            })
            .unwrap();
        let mut invalid_nonces = nonces;
        invalid_nonces[0] = invalid_nonce;

        let mut verifier_transcript = Blake3Transcript::new();
        assert_eq!(
            output.proof.verify_grinded::<TestOuterGrinding>(
                &mut verifier_transcript,
                initial_claim,
                &tau,
                &field_cfg,
                &invalid_nonces,
                GRINDING_BITS,
            ),
            Err(SumcheckError::Grinding(GrindingError::InvalidNonce {
                nonce: invalid_nonce,
                bits: GRINDING_BITS,
            }))
        );
    }

    #[test]
    fn sumcheck_rejects_a_tampered_linear_coefficient() {
        let field_cfg = config();
        let proof = SumcheckProof::<Fp<2>, 4> {
            round_polynomials: vec![
                [
                    field(10, &field_cfg),
                    field(0, &field_cfg),
                    field(0, &field_cfg),
                    field(0, &field_cfg),
                ],
                [
                    field(1, &field_cfg),
                    field(3, &field_cfg),
                    field(3, &field_cfg),
                    field(3, &field_cfg),
                ],
            ],
        };
        let mut transcript = Blake3Transcript::new();

        assert_eq!(
            proof.verify(&mut transcript, field(20, &field_cfg), 2, &field_cfg),
            Err(SumcheckError::InvalidRoundClaim { round: 1 })
        );
    }

    #[test]
    fn outer_sumcheck_rejects_a_bad_terminal_evaluation() {
        let field_cfg = config();
        let proof = OuterSumcheckProof {
            sumcheck: SumcheckProof {
                round_polynomials: Vec::new(),
            },
            az_mle_claim: field(2, &field_cfg),
            bz_mle_claim: field(3, &field_cfg),
            cz_mle_claim: field(5, &field_cfg),
        };
        let mut transcript = Blake3Transcript::new();

        assert_eq!(
            proof.verify(
                &mut transcript,
                Fp::<2>::zero_with_cfg(&field_cfg),
                &[],
                &field_cfg,
            ),
            Err(SumcheckError::InvalidTerminalClaim)
        );
    }

    #[test]
    fn outer_sumcheck_supports_every_equality_factor_split() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let num_vars = 5;
        let table_len = 1 << num_vars;
        let products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(index as u64 + 2, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(3 * index as u64 + 5, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            cz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field((index * index) as u64 + 7, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
        };
        let tau = [2, 4, 6, 8, 10]
            .into_iter()
            .map(|value| field(value, &field_cfg))
            .collect::<Vec<_>>();
        let full_equality = eq_table(&tau, &field_cfg).unwrap();
        let mut initial_claim = zero.clone();
        for (index, equality) in full_equality.iter().enumerate() {
            let residual = (field_cfg).sub(
                &(field_cfg).mul(
                    &products.az.evaluations[index],
                    &products.bz.evaluations[index],
                ),
                &products.cz.evaluations[index],
            );
            initial_claim =
                field_cfg.add(&(initial_claim), &(&(field_cfg).mul(equality, &residual)));
        }

        let mut reference_proof = None;
        for split in 0..=num_vars {
            let (low_point, high_point) = tau.split_at(split);
            let equality_factors = (
                DenseMultilinearExtension {
                    evaluations: eq_table(low_point, &field_cfg).unwrap(),
                    num_vars: low_point.len(),
                },
                DenseMultilinearExtension {
                    evaluations: eq_table(high_point, &field_cfg).unwrap(),
                    num_vars: high_point.len(),
                },
            );
            let mut prover_transcript = Blake3Transcript::new();
            let output =
                crate::sumcheck::outer::EqualityFactors::from_mles(equality_factors, &field_cfg)
                    .and_then(|factors| {
                        crate::sumcheck::outer::prove_outer_sumcheck(
                            &field_cfg,
                            &mut prover_transcript,
                            crate::sumcheck::outer::OuterClaim::Sum(initial_claim.clone()),
                            &tau,
                            products.clone(),
                            Some(factors),
                            &mut crate::sumcheck::UngrindedRoundBoundary,
                        )
                    })
                    .map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                    .unwrap();

            assert_eq!(output.proof.sumcheck.round_polynomials.len(), num_vars);
            if let Some(reference_proof) = &reference_proof {
                assert_eq!(&output.proof, reference_proof);
            } else {
                reference_proof = Some(output.proof.clone());
            }

            let mut verifier_transcript = Blake3Transcript::new();
            let verified = output
                .proof
                .verify(
                    &mut verifier_transcript,
                    initial_claim.clone(),
                    &tau,
                    &field_cfg,
                )
                .unwrap();
            assert_eq!(verified.eval_points, output.eval_points);
            assert_eq!(verified.az_mle_claim, output.proof.az_mle_claim);
            assert_eq!(verified.bz_mle_claim, output.proof.bz_mle_claim);
            assert_eq!(verified.cz_mle_claim, output.proof.cz_mle_claim);
            assert_eq!(
                squeeze_field::<Fp<2>, _>(&mut prover_transcript, &field_cfg).unwrap(),
                squeeze_field::<Fp<2>, _>(&mut verifier_transcript, &field_cfg).unwrap(),
            );
        }
    }

    #[test]
    fn factorized_outer_matches_direct_cubic_reference() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let num_vars = 10;
        let table_len = 1 << num_vars;
        let products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(index as u64 + 2, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(3 * index as u64 + 5, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            cz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field((index * index) as u64 + 7, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
        };
        let tau = (0..num_vars)
            .map(|index| field(2 * index as u64 + 3, &field_cfg))
            .collect::<Vec<_>>();
        let (low_point, high_point) = tau.split_at(num_vars / 2);
        let equality_factors = (
            DenseMultilinearExtension {
                evaluations: eq_table(low_point, &field_cfg).unwrap(),
                num_vars: low_point.len(),
            },
            DenseMultilinearExtension {
                evaluations: eq_table(high_point, &field_cfg).unwrap(),
                num_vars: high_point.len(),
            },
        );

        let full_equality = eq_table(&tau, &field_cfg).unwrap();
        let mut initial_claim = zero;
        for (index, equality) in full_equality.iter().enumerate() {
            let residual = (field_cfg).sub(
                &(field_cfg).mul(
                    &products.az.evaluations[index],
                    &products.bz.evaluations[index],
                ),
                &products.cz.evaluations[index],
            );
            initial_claim =
                field_cfg.add(&(initial_claim), &(&(field_cfg).mul(equality, &residual)));
        }

        let mut direct_transcript = Blake3Transcript::new();
        let direct_output = prove_outer_sumcheck_direct_reference(
            &mut direct_transcript,
            initial_claim.clone(),
            &tau,
            products.clone(),
            &field_cfg,
        )
        .unwrap();
        let direct_continuation = squeeze_field(&mut direct_transcript, &field_cfg).unwrap();
        let mut transcript = Blake3Transcript::new();
        let immediate_output: OuterSumcheckOutput<_> =
            crate::sumcheck::outer::prove_outer_sumcheck(
                &field_cfg,
                &mut transcript,
                crate::sumcheck::outer::OuterClaim::Sum(initial_claim),
                &tau,
                products,
                Some(
                    crate::sumcheck::outer::EqualityFactors::from_mles(
                        equality_factors,
                        &field_cfg,
                    )
                    .unwrap(),
                ),
                &mut UngrindedRoundBoundary,
            )
            .unwrap()
            .into();
        let immediate_continuation = squeeze_field(&mut transcript, &field_cfg).unwrap();

        assert_eq!(immediate_output.proof, direct_output.proof);
        assert_eq!(immediate_output.eval_points, direct_output.eval_points);
        assert_eq!(immediate_output.final_claim, direct_output.final_claim);
        assert_eq!(immediate_continuation, direct_continuation);

        let mut verifier_transcript = Blake3Transcript::new();
        let verified = immediate_output
            .proof
            .verify(&mut verifier_transcript, initial_claim, &tau, &field_cfg)
            .unwrap();
        assert_eq!(verified.eval_points, immediate_output.eval_points);
        assert_eq!(
            squeeze_field::<Fp<2>, _>(&mut verifier_transcript, &field_cfg).unwrap(),
            immediate_continuation
        );
    }

    #[test]
    fn eq_factored_outer_is_direct_cubic_exact_at_tau_edges() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let one = Fp::<2>::one_with_cfg(&field_cfg);
        let two = (field_cfg).add(&one, &one);
        let half = *field::FieldOps::inverse_ct(&field_cfg, &two).value();
        let num_vars = 3;
        let table_len = 1 << num_vars;
        let products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(index as u64 * 5 + 2, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(index as u64 * 7 + 3, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            cz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                (0..table_len)
                    .map(|index| field(index as u64 * 11 + 1, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
        };

        let tau_cases = [
            vec![zero.clone(), zero.clone(), zero.clone()],
            vec![one.clone(), one.clone(), one.clone()],
            vec![zero.clone(), one.clone(), half],
            vec![
                field(2, &field_cfg),
                field(5, &field_cfg),
                field(9, &field_cfg),
            ],
        ];

        for tau in tau_cases {
            let equality = eq_table(&tau, &field_cfg).unwrap();
            let initial_claim =
                equality
                    .iter()
                    .enumerate()
                    .fold(zero.clone(), |mut claim, (index, weight)| {
                        claim = field_cfg.add(
                            &(claim),
                            &(&(field_cfg).mul(
                                weight,
                                &(field_cfg).sub(
                                    &(field_cfg).mul(
                                        &products.az.evaluations[index],
                                        &products.bz.evaluations[index],
                                    ),
                                    &products.cz.evaluations[index],
                                ),
                            )),
                        );
                        claim
                    });

            let mut direct_transcript = Blake3Transcript::new();
            let direct = prove_outer_sumcheck_direct_reference(
                &mut direct_transcript,
                initial_claim.clone(),
                &tau,
                products.clone(),
                &field_cfg,
            )
            .unwrap();
            let direct_continuation =
                squeeze_field::<Fp<2>, _>(&mut direct_transcript, &field_cfg).unwrap();

            for split in 0..=num_vars {
                let (low_tau, high_tau) = tau.split_at(split);
                let equality_factors = (
                    DenseMultilinearExtension {
                        evaluations: eq_table(low_tau, &field_cfg).unwrap(),
                        num_vars: low_tau.len(),
                    },
                    DenseMultilinearExtension {
                        evaluations: eq_table(high_tau, &field_cfg).unwrap(),
                        num_vars: high_tau.len(),
                    },
                );
                let mut transcript = Blake3Transcript::new();
                let output = crate::sumcheck::outer::EqualityFactors::from_mles(
                    equality_factors,
                    &field_cfg,
                )
                .and_then(|factors| {
                    crate::sumcheck::outer::prove_outer_sumcheck(
                        &field_cfg,
                        &mut transcript,
                        crate::sumcheck::outer::OuterClaim::Sum(initial_claim.clone()),
                        &tau,
                        products.clone(),
                        Some(factors),
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                })
                .map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                .unwrap();
                let continuation = squeeze_field::<Fp<2>, _>(&mut transcript, &field_cfg).unwrap();

                assert_eq!(output.proof, direct.proof);
                assert_eq!(output.eval_points, direct.eval_points);
                assert_eq!(output.final_claim, direct.final_claim);
                assert_eq!(continuation, direct_continuation);
            }
        }
    }

    #[test]
    fn native_u32_eq_factoring_matches_direct_cubic_at_signed_extremes() {
        fn prove_with_native_reducer<R>(
            initial_claim: &Fp<2>,
            tau: &[Fp<2>],
            equality_factors: &(
                DenseMultilinearExtension<Fp<2>>,
                DenseMultilinearExtension<Fp<2>>,
            ),
            products: &R1csProductMles<u64>,
            field_cfg: &<Fp<2> as crate::piop::spartan::SpartanField>::Config,
            reducer: &R,
        ) -> (OuterSumcheckOutput<Fp<2>>, Fp<2>)
        where
            R: BatchMulAcc<Fp<2>>
                + Reduce<<R as BatchMulAcc<Fp<2>>>::Accumulator, Output = Fp<2>>
                + Sync
                + SumcheckLinearReducer,
        {
            let mut transcript = Blake3Transcript::new();
            let output = prove_u32_first_round(
                &mut transcript,
                initial_claim.clone(),
                tau,
                equality_factors.clone(),
                products.clone(),
                field_cfg,
                reducer,
            )
            .unwrap();
            let continuation = squeeze_field::<Fp<2>, _>(&mut transcript, field_cfg).unwrap();
            (output, continuation)
        }

        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let one = Fp::<2>::one_with_cfg(&field_cfg);
        let num_vars = 2;
        let max_u32 = u64::from(u32::MAX);
        let native_products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vec![0, max_u32, max_u32, 0],
                0,
            ),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vec![0, max_u32, 0, max_u32],
                0,
            ),
            cz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vec![u64::MAX, 0, 0, u64::MAX],
                0,
            ),
        };
        let field_products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                native_products
                    .az
                    .evaluations
                    .iter()
                    .map(|&value| field(value, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                native_products
                    .bz
                    .evaluations
                    .iter()
                    .map(|&value| field(value, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
            cz: DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                native_products
                    .cz
                    .evaluations
                    .iter()
                    .map(|&value| field(value, &field_cfg))
                    .collect(),
                zero.clone(),
            ),
        };
        let tau_cases = [
            vec![zero.clone(), one.clone()],
            vec![one.clone(), zero.clone()],
            vec![field(7, &field_cfg), field(11, &field_cfg)],
        ];
        let optimized = crate::utils::delayed_reduction::prepare_field(&field_cfg).unwrap();
        let crypto_bigint = BigUintSumcheckOracle::new(&field_cfg).unwrap();

        for tau in tau_cases {
            let equality = eq_table(&tau, &field_cfg).unwrap();
            let initial_claim =
                equality
                    .iter()
                    .enumerate()
                    .fold(zero.clone(), |mut claim, (index, weight)| {
                        claim = field_cfg.add(
                            &(claim),
                            &(&(field_cfg).mul(
                                weight,
                                &(field_cfg).sub(
                                    &(field_cfg).mul(
                                        &field_products.az.evaluations[index],
                                        &field_products.bz.evaluations[index],
                                    ),
                                    &field_products.cz.evaluations[index],
                                ),
                            )),
                        );
                        claim
                    });
            let mut direct_transcript = Blake3Transcript::new();
            let direct = prove_outer_sumcheck_direct_reference(
                &mut direct_transcript,
                initial_claim.clone(),
                &tau,
                field_products.clone(),
                &field_cfg,
            )
            .unwrap();
            let direct_continuation =
                squeeze_field::<Fp<2>, _>(&mut direct_transcript, &field_cfg).unwrap();

            for split in 0..=num_vars {
                let (low_tau, high_tau) = tau.split_at(split);
                let equality_factors = (
                    DenseMultilinearExtension {
                        evaluations: eq_table(low_tau, &field_cfg).unwrap(),
                        num_vars: low_tau.len(),
                    },
                    DenseMultilinearExtension {
                        evaluations: eq_table(high_tau, &field_cfg).unwrap(),
                        num_vars: high_tau.len(),
                    },
                );

                let optimized_result = prove_with_native_reducer(
                    &initial_claim,
                    &tau,
                    &equality_factors,
                    &native_products,
                    &field_cfg,
                    &optimized,
                );
                let crypto_bigint_result = prove_with_native_reducer(
                    &initial_claim,
                    &tau,
                    &equality_factors,
                    &native_products,
                    &field_cfg,
                    &crypto_bigint,
                );
                for (output, continuation) in [optimized_result, crypto_bigint_result] {
                    assert_eq!(output.proof, direct.proof);
                    assert_eq!(output.eval_points, direct.eval_points);
                    assert_eq!(output.final_claim, direct.final_claim);
                    assert_eq!(continuation, direct_continuation);
                }
            }
        }
    }

    #[test]
    fn inner_sumcheck_folds_the_lowest_coordinate_first() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let matrix = DenseMultilinearExtension::from_evaluations_vec(
            2,
            [2, 5, 11, 17]
                .into_iter()
                .map(|value| field(value, &field_cfg))
                .collect(),
            zero.clone(),
        );
        let witness = DenseMultilinearExtension::from_evaluations_vec(
            2,
            [3, 7, 13, 19]
                .into_iter()
                .map(|value| field(value, &field_cfg))
                .collect(),
            zero,
        );
        let mut transcript = Blake3Transcript::new();

        let output = prove_inner_sumcheck(
            &mut transcript,
            field(507, &field_cfg),
            matrix,
            witness,
            &field_cfg,
        )
        .unwrap();

        assert_eq!(
            output.sumcheck.proof.round_polynomials[0],
            [
                field(149, &field_cfg),
                field(161, &field_cfg),
                field(48, &field_cfg),
            ]
        );
    }

    #[test]
    fn inner_sumcheck_rejects_mismatched_dimensions_before_absorption() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let matrix = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![field(1, &field_cfg), field(2, &field_cfg)],
            zero.clone(),
        );
        let witness = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                field(1, &field_cfg),
                field(2, &field_cfg),
                field(3, &field_cfg),
                field(4, &field_cfg),
            ],
            zero.clone(),
        );
        let mut transcript = Blake3Transcript::new();

        assert_eq!(
            prove_inner_sumcheck(&mut transcript, zero.clone(), matrix, witness, &field_cfg,),
            Err(SumcheckError::InvalidProductDimensions)
        );

        let rejected_next = squeeze_field::<Fp<2>, _>(&mut transcript, &field_cfg).unwrap();
        let mut fresh_transcript = Blake3Transcript::new();
        let fresh_next = squeeze_field::<Fp<2>, _>(&mut fresh_transcript, &field_cfg).unwrap();
        assert_eq!(rejected_next, fresh_next);
    }

    #[test]
    fn outer_sumcheck_rejects_malformed_dimensions_before_absorption() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let one = Fp::<2>::one_with_cfg(&field_cfg);
        let products = R1csProductMles {
            az: DenseMultilinearExtension::zero_vars(field(1, &field_cfg)),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                1,
                vec![field(2, &field_cfg), field(3, &field_cfg)],
                zero.clone(),
            ),
            cz: DenseMultilinearExtension::zero_vars(field(4, &field_cfg)),
        };
        let equality_factors = (
            DenseMultilinearExtension::zero_vars(one.clone()),
            DenseMultilinearExtension::zero_vars(one),
        );
        let mut transcript = Blake3Transcript::new();

        assert_eq!(
            crate::sumcheck::outer::EqualityFactors::from_mles(equality_factors, &field_cfg)
                .and_then(|factors| crate::sumcheck::outer::prove_outer_sumcheck(
                    &field_cfg,
                    &mut transcript,
                    crate::sumcheck::outer::OuterClaim::Sum(zero),
                    &[],
                    products,
                    Some(factors),
                    &mut crate::sumcheck::UngrindedRoundBoundary,
                ))
                .map(crate::sumcheck::proof::OuterSumcheckOutput::from),
            Err(SumcheckError::InvalidProductDimensions)
        );

        let rejected_next = squeeze_field::<Fp<2>, _>(&mut transcript, &field_cfg).unwrap();
        let mut fresh_transcript = Blake3Transcript::new();
        let fresh_next = squeeze_field::<Fp<2>, _>(&mut fresh_transcript, &field_cfg).unwrap();
        assert_eq!(rejected_next, fresh_next);
    }

    #[test]
    fn outer_sumcheck_rejects_tau_length_before_absorption() {
        let field_cfg = config();
        let zero = Fp::<2>::zero_with_cfg(&field_cfg);
        let one = Fp::<2>::one_with_cfg(&field_cfg);
        let products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations_vec(
                1,
                vec![field(1, &field_cfg), field(2, &field_cfg)],
                zero.clone(),
            ),
            bz: DenseMultilinearExtension::from_evaluations_vec(
                1,
                vec![field(3, &field_cfg), field(4, &field_cfg)],
                zero.clone(),
            ),
            cz: DenseMultilinearExtension::from_evaluations_vec(
                1,
                vec![field(5, &field_cfg), field(6, &field_cfg)],
                zero.clone(),
            ),
        };
        let equality_factors = (
            DenseMultilinearExtension::zero_vars(one),
            DenseMultilinearExtension {
                evaluations: vec![field(7, &field_cfg), field(8, &field_cfg)],
                num_vars: 1,
            },
        );
        let mut transcript = Blake3Transcript::new();

        assert_eq!(
            crate::sumcheck::outer::EqualityFactors::from_mles(equality_factors, &field_cfg)
                .and_then(|factors| crate::sumcheck::outer::prove_outer_sumcheck(
                    &field_cfg,
                    &mut transcript,
                    crate::sumcheck::outer::OuterClaim::Sum(zero),
                    &[],
                    products,
                    Some(factors),
                    &mut crate::sumcheck::UngrindedRoundBoundary,
                ))
                .map(crate::sumcheck::proof::OuterSumcheckOutput::from),
            Err(SumcheckError::InvalidEqualityDimensions)
        );

        let rejected_next = squeeze_field::<Fp<2>, _>(&mut transcript, &field_cfg).unwrap();
        let mut fresh_transcript = Blake3Transcript::new();
        let fresh_next = squeeze_field::<Fp<2>, _>(&mut fresh_transcript, &field_cfg).unwrap();
        assert_eq!(rejected_next, fresh_next);
    }
}
