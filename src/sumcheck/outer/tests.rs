use super::ordinary::prove_outer_sumcheck_direct_reference;
use super::*;
use crate::piop::spartan::{R1csProductMles, SpartanField, matrix::eq_table};
use crate::poly::mle::DenseMultilinearExtension;
use crate::sumcheck::{SumcheckError, UngrindedRoundBoundary};
use crate::transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use field::{Fp, FpCtx, IntegerEmbedding, RingOps, Uint, Z};

fn field() -> FpCtx<2> {
    Fp::<2>::make_cfg(&Uint::from((1u128 << 100) - 15)).unwrap()
}

#[test]
fn owned_rows_are_released_before_field_continuation() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    struct Rows(OuterInputs<u32, u64>, Arc<AtomicBool>);
    impl OuterRows for Rows {
        type AB = u32;
        type C = u64;
        fn dimensions(&self) -> (usize, usize, usize) {
            self.0.dimensions()
        }
        fn a(&self, i: usize) -> u32 {
            self.0.a(i)
        }
        fn b(&self, i: usize) -> u32 {
            self.0.b(i)
        }
        fn c(&self, i: usize) -> u64 {
            self.0.c(i)
        }
    }
    impl Drop for Rows {
        fn drop(&mut self) {
            self.1.store(true, Ordering::SeqCst);
        }
    }
    struct Boundary(Arc<AtomicBool>);
    impl crate::sumcheck::RoundBoundaryPolicy for Boundary {
        fn after_round<T: Transcript>(
            &mut self,
            _: &mut T,
            round: usize,
        ) -> Result<(), SumcheckError> {
            assert_eq!(self.0.load(Ordering::SeqCst), round > 0);
            Ok(())
        }
    }
    let f = field();
    let tau = vec![f.one(); 3];
    let dropped = Arc::new(AtomicBool::new(false));
    let rows = Rows(
        OuterInputs {
            ax: vec![3; 8],
            bx: vec![5; 8],
            cx: vec![15; 8],
        },
        dropped.clone(),
    );
    let (low, high) = crate::piop::spartan::matrix::make_equality_factors(&tau, &f).unwrap();
    let output = prove_outer_sumcheck(
        &f,
        &mut Blake3Transcript::new(),
        OuterClaim::RowwiseZero,
        &tau,
        rows,
        Some(EqualityFactors::new(low.evaluations, high.evaluations, &f)),
        &mut Boundary(dropped.clone()),
    )
    .unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    verify_outer_sumcheck(
        &f,
        &mut Blake3Transcript::new(),
        f.zero(),
        &tau,
        &output.proof,
        output.evaluations,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
}

#[test]
fn prepared_mixed_folds_and_signed_buckets_match_bigint() {
    use field::{PreparedLinearCombination, PreparedWordWeights, UintAccumulator};
    use num_bigint::{BigInt, BigUint, Sign};
    fn check<const L: usize, const N: usize>(p: Uint<L>) {
        let f = field::create_prime_field(p);
        let bytes = |words: &[u64]| {
            words
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let modulus = BigInt::from_bytes_le(Sign::Plus, &bytes(p.as_words()));
        let values: [Z<N>; 16] = core::array::from_fn(|i| {
            Z::from_twos_complement_words(core::array::from_fn(|j| match i % 4 {
                0 => u64::MAX,
                1 => {
                    if j == N - 1 {
                        1u64 << 63
                    } else {
                        0
                    }
                }
                2 => {
                    if j == N - 1 {
                        u64::MAX >> 1
                    } else {
                        u64::MAX
                    }
                }
                _ => (i as u64)
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .rotate_left(j as u32),
            }))
        });
        let coefficients: Vec<_> = (0..16).map(|i| f.from_integer(&(u128::MAX - i))).collect();
        let expected = |count: usize| {
            let total: BigInt = (0..count)
                .map(|i| {
                    let c = BigInt::from_bytes_le(
                        Sign::Plus,
                        &bytes(f.to_integer(&coefficients[i % 16]).as_words()),
                    );
                    let v = BigInt::from_signed_bytes_le(&bytes(values[i % 16].as_words()));
                    c * v
                })
                .sum();
            ((total % &modulus + &modulus) % &modulus)
                .to_biguint()
                .unwrap()
        };
        macro_rules! check_fold {
            ($count:literal) => {{
                let prepared =
                    <FpCtx<L> as PreparedLinearCombination<Z<N>>>::prepare_linear_combination(
                        &f,
                        core::array::from_fn::<_, $count, _>(|i| coefficients[i]),
                    );
                let actual = <FpCtx<L> as PreparedLinearCombination<Z<N>>>::linear_combination(
                    &prepared,
                    |i| values[i],
                );
                assert_eq!(
                    BigUint::from_bytes_le(&bytes(f.to_integer(&actual).as_words())),
                    expected($count)
                );
            }};
        }
        check_fold!(1);
        check_fold!(2);
        check_fold!(8);
        check_fold!(16);
        let prepared = PreparedWordWeights::<L, N>::signed(&f, &coefficients);
        let mut acc = UintAccumulator::ZERO;
        for i in 0..257 {
            prepared.accumulate_signed::<N>(&mut acc, i % 16, values[i % 16]);
        }
        let actual = prepared.finish(&f, acc);
        assert_eq!(
            BigUint::from_bytes_le(&bytes(f.to_integer(&actual).as_words())),
            expected(257)
        );
    }
    check::<1, 1>(Uint::from_words([(1u64 << 61) - 1]));
    check::<1, 4>(Uint::from_words([(1u64 << 61) - 1]));
    check::<1, 256>(Uint::from_words([(1u64 << 61) - 1]));
    for p in [(1u128 << 100) - 15, (1u128 << 127) - 1, u128::MAX - 158] {
        check::<2, 1>(Uint::from(p));
        check::<2, 2>(Uint::from(p));
        check::<2, 4>(Uint::from(p));
        check::<2, 9>(Uint::from(p));
        check::<2, 32>(Uint::from(p));
        check::<2, 64>(Uint::from(p));
        check::<2, 130>(Uint::from(p));
        check::<2, 256>(Uint::from(p));
    }
}

#[test]
fn signed_bucket_public_width_preserves_unsigned_top_bit() {
    use field::{PreparedWordWeights, UintAccumulator};
    use num_bigint::BigInt;
    let f = field();
    let coefficients = [f.zero(), f.one(), f.from_integer(&123456789u64)];
    let prepared = PreparedWordWeights::<2, 5>::signed(&f, &coefficients);
    let modulus = BigInt::from((1u128 << 100) - 15);
    let mut acc = UintAccumulator::ZERO;
    let mut expected = BigInt::from(0);
    for words in [[0; 4], [u64::MAX; 4], [0, 0, 0, 1u64 << 63], [1, 0, 0, 0]] {
        let positive = Z::from_twos_complement_words([words[0], words[1], words[2], words[3], 0]);
        for value in [positive, -positive] {
            for (index, coefficient) in coefficients.iter().enumerate() {
                prepared.accumulate_signed::<4>(&mut acc, index, value);
                let bytes: Vec<_> = value
                    .as_words()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                expected += BigInt::from_signed_bytes_le(&bytes)
                    * BigInt::from(u128::from(f.to_integer(coefficient)));
            }
        }
        // Leave an uncancelled term, including values with bit 255 set.
        prepared.accumulate_signed::<4>(&mut acc, 2, positive);
        let bytes: Vec<_> = positive
            .as_words()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        expected += BigInt::from_signed_bytes_le(&bytes) * BigInt::from(123456789u64);
    }
    let expected = (expected % &modulus + &modulus) % &modulus;
    assert_eq!(
        BigInt::from(u128::from(f.to_integer(&prepared.finish(&f, acc)))),
        expected
    );
}
fn fe(f: &FpCtx<2>, v: u64) -> Fp<2> {
    f.from_integer(&v)
}
fn tables(f: &FpCtx<2>, a: &[u64], b: &[u64], c: &[u128]) -> R1csProductMles<Fp<2>> {
    let n = a.len().ilog2() as usize;
    let mle = |v| DenseMultilinearExtension {
        num_vars: n,
        evaluations: v,
    };
    R1csProductMles {
        az: mle(a.iter().map(|x| f.from_integer(x)).collect()),
        bz: mle(b.iter().map(|x| f.from_integer(x)).collect()),
        cz: mle(c.iter().map(|x| f.from_integer(x)).collect()),
    }
}
#[test]
fn mixed_inputs_match_direct_cubic_and_transcript() {
    let f = field();
    for n in 0..=7 {
        let a: Vec<_> = (0..1 << n).map(|i| i as u64 * 3 + 7).collect();
        let b: Vec<_> = a.iter().map(|x| x + 11).collect();
        let c: Vec<_> = a.iter().map(|x| u128::from(*x) * 17).collect();
        let tau: Vec<_> = (0..n)
            .map(|i| {
                fe(
                    &f,
                    if i % 3 == 0 {
                        0
                    } else if i % 3 == 1 {
                        1
                    } else {
                        9
                    },
                )
            })
            .collect();
        let products = tables(&f, &a, &b, &c);
        let eq = eq_table(&tau, &f).unwrap();
        let claim = (0..a.len()).fold(f.zero(), |sum, i| {
            f.add(
                &sum,
                &f.mul(
                    &eq[i],
                    &f.sub(
                        &f.mul(&products.az.evaluations[i], &products.bz.evaluations[i]),
                        &products.cz.evaluations[i],
                    ),
                ),
            )
        });
        let mut reference = Blake3Transcript::new();
        let old = prove_outer_sumcheck_direct_reference(&mut reference, claim, &tau, products, &f)
            .unwrap();
        let mut prover = Blake3Transcript::new();
        let out = crate::sumcheck::outer::prove_outer_sumcheck(
            &f,
            &mut prover,
            crate::sumcheck::outer::OuterClaim::Sum(claim),
            &tau,
            crate::sumcheck::outer::OuterSlices {
                ax: &a,
                bx: &b,
                cx: &c,
            },
            None,
            &mut UngrindedRoundBoundary,
        )
        .unwrap();
        assert_eq!(out.proof, old.proof.sumcheck);
        assert_eq!(out.point, old.eval_points);
        assert_eq!(prover.state_digest(), reference.state_digest());
        let mut verifier = Blake3Transcript::new();
        let verified = verify_outer_sumcheck(
            &f,
            &mut verifier,
            claim,
            &tau,
            &out.proof,
            out.evaluations,
            &mut UngrindedRoundBoundary,
        )
        .unwrap();
        assert_eq!(verified.final_claim, out.final_claim);
        assert_eq!(prover.state_digest(), verifier.state_digest());
    }
}
#[test]
fn zero_prefix_and_owned_field_inputs_match_ordinary() {
    let f = field();
    for n in 0..=7 {
        let a: Vec<_> = (0..1 << n).map(|i| u64::MAX - i as u64).collect();
        let b: Vec<_> = a.iter().rev().copied().collect();
        let c: Vec<_> = a
            .iter()
            .zip(&b)
            .map(|(a, b)| u128::from(*a) * u128::from(*b))
            .collect();
        let tau = vec![fe(&f, 7); n];
        let mut ordinary = Blake3Transcript::new();
        let expected = crate::sumcheck::outer::prove_outer_sumcheck(
            &f,
            &mut ordinary,
            crate::sumcheck::outer::OuterClaim::Sum(f.zero()),
            &tau,
            crate::sumcheck::outer::OuterSlices {
                ax: &a,
                bx: &b,
                cx: &c,
            },
            None,
            &mut UngrindedRoundBoundary,
        )
        .unwrap();
        let mut zero = Blake3Transcript::new();
        let got = crate::sumcheck::outer::prove_outer_sumcheck(
            &f,
            &mut zero,
            crate::sumcheck::outer::OuterClaim::RowwiseZero,
            &tau,
            crate::sumcheck::outer::OuterSlices {
                ax: &a,
                bx: &b,
                cx: &c,
            },
            None,
            &mut UngrindedRoundBoundary,
        )
        .unwrap();
        assert_eq!(got, expected);
        assert_eq!(ordinary.state_digest(), zero.state_digest());
        let projected = tables(&f, &a, &b, &c);
        let inputs = OuterInputs {
            ax: projected.az.evaluations,
            bx: projected.bz.evaluations,
            cx: projected.cz.evaluations,
        };
        let mut owned = Blake3Transcript::new();
        assert_eq!(
            crate::sumcheck::outer::prove_outer_sumcheck(
                &f,
                &mut owned,
                crate::sumcheck::outer::OuterClaim::RowwiseZero,
                &tau,
                inputs,
                None,
                &mut UngrindedRoundBoundary,
            )
            .unwrap(),
            expected
        );
        assert_eq!(owned.state_digest(), ordinary.state_digest());
    }
}
#[test]
fn skip_all_widths_matches_preserved_native_kernel() {
    let f = field();
    for k in 1..=4u8 {
        for n in usize::from(k)..=usize::from(k) + 2 {
            let a: Vec<_> = (0..1 << n).map(|i| i as u64 + 3).collect();
            let b: Vec<_> = a.iter().map(|x| x + 11).collect();
            let c: Vec<_> = a.iter().zip(&b).map(|(a, b)| a * b).collect();
            let tau = vec![fe(&f, 7); n - usize::from(k)];
            let prepared = prepare_univariate_skip(&f, k).unwrap();
            let mut prover = Blake3Transcript::new();
            let got = crate::sumcheck::outer::prove_outer_zerocheck_with_skip(
                &f,
                &mut prover,
                &prepared,
                &tau,
                crate::sumcheck::outer::OuterSlices {
                    ax: &a,
                    bx: &b,
                    cx: &c,
                },
                None,
                &mut UngrindedRoundBoundary,
            )
            .unwrap();
            let (lo, hi) = crate::piop::spartan::raw_monty::make_equality_factors_raw(&f, &tau);
            let native = arithmetic::NativeProducts {
                az: &a,
                bz: &b,
                cz: &c,
            };
            let mut reference_transcript = Blake3Transcript::new();
            let reference = native_skip::prove_native_skip_reference(
                &mut reference_transcript,
                &f,
                k,
                &tau,
                lo.clone(),
                hi.clone(),
                native,
            )
            .unwrap();
            let actual: univariate::UnivariateSkipOuterSumcheckOutput<_> = got.clone().into();
            assert_eq!(actual, reference);
            assert_eq!(prover.state_digest(), reference_transcript.state_digest());
            let message =
                native_skip::encoded_native_message(&f, usize::from(k), &lo, &hi, native, &f, &f)
                    .unwrap();
            let prefix =
                UnivariateSkipProof::from_ordered_message(usize::from(k), message).unwrap();
            assert_eq!(prefix, got.prefix);
            let mut expected_transcript = Blake3Transcript::new();
            let reduction = prefix
                .verify_reduction(&mut expected_transcript, &f)
                .unwrap();
            let folded =
                native_skip::fold_encoded_lagrange(usize::from(k), native, &reduction.z, &f)
                    .unwrap();
            let tail = arithmetic::prepare_encoded_reference(
                &mut expected_transcript,
                &f,
                &f,
                reduction.q_at_z,
                &tau,
                lo,
                hi,
                folded,
                &mut UngrindedRoundBoundary,
                false,
            )
            .unwrap();
            assert_eq!(got.tail, tail.into());
            assert_eq!(prover.state_digest(), expected_transcript.state_digest());
            let mut verifier = Blake3Transcript::new();
            let verified = verify_outer_zerocheck_with_skip(
                &f,
                &mut verifier,
                &prepared,
                &tau,
                &got.prefix,
                &got.tail.proof,
                got.tail.evaluations,
                &mut UngrindedRoundBoundary,
            )
            .unwrap();
            assert_eq!(verified.prefix_challenge, got.prefix_challenge);
            assert_eq!(prover.state_digest(), verifier.state_digest());
        }
    }
}
#[test]
fn signed_and_four_limb_inputs() {
    let f = field();
    let tau = [fe(&f, 7)];
    let a = [
        Z::<2>::from_twos_complement_words([u64::MAX, u64::MAX]),
        Z::from_twos_complement_words([2, 0]),
    ];
    let c = [
        Z::<4>::from_twos_complement_words([1, 0, 0, 0]),
        Z::from_twos_complement_words([4, 0, 0, 0]),
    ];
    let mut transcript = Blake3Transcript::new();
    crate::sumcheck::outer::prove_outer_sumcheck(
        &f,
        &mut transcript,
        crate::sumcheck::outer::OuterClaim::RowwiseZero,
        &tau,
        crate::sumcheck::outer::OuterSlices {
            ax: &a,
            bx: &a,
            cx: &c,
        },
        None,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    let a = [u128::MAX, u128::MAX - 1];
    let c = a.map(|a| {
        let p =
            field::WideMul::mul_wide(&field::IntegerOps, &Uint::<2>::from(a), &Uint::<2>::from(a));
        *p.checked_resize_ct::<4>().value()
    });
    crate::sumcheck::outer::prove_outer_sumcheck(
        &f,
        &mut Blake3Transcript::new(),
        crate::sumcheck::outer::OuterClaim::RowwiseZero,
        &tau,
        crate::sumcheck::outer::OuterSlices {
            ax: &a,
            bx: &a,
            cx: &c,
        },
        None,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
}
#[test]
fn invalid_shape_and_singleton_claim_rejected() {
    let f = field();
    for len in [0, 3, 5] {
        let mut t = Blake3Transcript::new();
        let before = t.state_digest();
        let input = vec![1u64; len];
        assert_eq!(
            crate::sumcheck::outer::prove_outer_sumcheck(
                &f,
                &mut t,
                crate::sumcheck::outer::OuterClaim::Sum(f.zero()),
                &[],
                crate::sumcheck::outer::OuterSlices {
                    ax: &input,
                    bx: &input,
                    cx: &input
                },
                None,
                &mut UngrindedRoundBoundary,
            )
            .unwrap_err(),
            SumcheckError::InvalidProductDimensions
        );
        assert_eq!(before, t.state_digest());
    }
    assert_eq!(
        crate::sumcheck::outer::prove_outer_sumcheck(
            &f,
            &mut Blake3Transcript::new(),
            crate::sumcheck::outer::OuterClaim::Sum(f.zero()),
            &[],
            crate::sumcheck::outer::OuterSlices {
                ax: &[2u64],
                bx: &[3u64],
                cx: &[7u64]
            },
            None,
            &mut UngrindedRoundBoundary,
        )
        .unwrap_err(),
        SumcheckError::InvalidTerminalClaim
    );
    assert!(prepare_univariate_skip(&f, 0).is_err());
    assert!(prepare_univariate_skip(&f, 5).is_err());
    assert_eq!(
        prove_outer_sumcheck_direct_reference(
            &mut Blake3Transcript::new(),
            f.one(),
            &[],
            tables(&f, &[1], &[1], &[1]),
            &f,
        )
        .unwrap_err(),
        SumcheckError::InvalidTerminalClaim,
    );
}
struct ZeroChallenges;
impl Transcript for ZeroChallenges {
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T {
        T::read_transcription_bytes_exact(&vec![0; T::NUM_BYTES])
    }
    fn fill_sampling_bytes(&mut self, out: &mut [u8]) {
        out.fill(0);
    }
    fn absorb_inner(&mut self, _: &[u8]) {}
}
#[test]
fn vanishing_equality_scale_is_carried_without_division() {
    let f = field();
    let tau = [f.one(), f.zero()];
    let output = crate::sumcheck::outer::prove_outer_sumcheck(
        &f,
        &mut ZeroChallenges,
        crate::sumcheck::outer::OuterClaim::RowwiseZero,
        &tau,
        crate::sumcheck::outer::OuterSlices {
            ax: &[2u64, 3, 4, 5],
            bx: &[3u64, 4, 5, 6],
            cx: &[6u64, 12, 20, 30],
        },
        None,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    assert_eq!(output.point, vec![f.zero(); 2]);
    assert_eq!(output.final_claim, f.zero());
    verify_outer_sumcheck(
        &f,
        &mut ZeroChallenges,
        f.zero(),
        &tau,
        &output.proof,
        output.evaluations,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
}

#[test]
fn zero_weighted_claim_does_not_imply_rowwise_zerocheck() {
    let f = field();
    let tau = [f.zero()];
    let a = [2u64, 3];
    let b = [3u64, 4];
    let c = [6u64, 13];
    let out = crate::sumcheck::outer::prove_outer_sumcheck(
        &f,
        &mut Blake3Transcript::new(),
        crate::sumcheck::outer::OuterClaim::Sum(f.zero()),
        &tau,
        crate::sumcheck::outer::OuterSlices {
            ax: &a,
            bx: &b,
            cx: &c,
        },
        None,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    verify_outer_sumcheck(
        &f,
        &mut Blake3Transcript::new(),
        f.zero(),
        &tau,
        &out.proof,
        out.evaluations,
        &mut UngrindedRoundBoundary,
    )
    .unwrap();
    assert_eq!(
        crate::sumcheck::outer::prove_outer_sumcheck(
            &f,
            &mut Blake3Transcript::new(),
            crate::sumcheck::outer::OuterClaim::RowwiseZero,
            &tau,
            crate::sumcheck::outer::OuterSlices {
                ax: &a,
                bx: &b,
                cx: &c
            },
            None,
            &mut UngrindedRoundBoundary,
        )
        .unwrap_err(),
        SumcheckError::InvalidTerminalClaim
    );
}

#[test]
fn encoded_zero_prefix_preserves_grinding_rounds_and_transcript() {
    use crate::sumcheck::boundary::{ProverGrindingRoundBoundary, VerifierGrindingRoundBoundary};
    struct Domain;
    impl crate::piop::spartan::grinding::GrindingDomain for Domain {
        const DOMAIN: &'static [u8] = b"outer-refactor-test/v1";
    }
    let f = field();
    let a = [2u64, 3, 4, 5, 6, 7, 8, 9];
    let b = [3u64, 4, 5, 6, 7, 8, 9, 10];
    let c: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(a, b)| u128::from(*a) * u128::from(b))
        .collect();
    let products = tables(&f, &a, &b, &c);
    let tau = vec![fe(&f, 3); 3];
    let eq = crate::piop::spartan::matrix::make_equality_factors(&tau, &f).unwrap();
    let mut reference = Blake3Transcript::new();
    let mut reference_boundary = ProverGrindingRoundBoundary::<Domain>::with_round_offset(3, 0);
    let expected = ordinary::prove_field_with_boundary_reference(
        &mut reference,
        f.zero(),
        &tau,
        eq,
        products.clone(),
        &f,
        &f,
        &mut reference_boundary,
    )
    .unwrap();
    let (low, high) = crate::piop::spartan::raw_monty::make_equality_factors_raw(&f, &tau);
    let encoded = arithmetic::RawProducts::from_field(&f, &products);
    let mut prover = Blake3Transcript::new();
    let mut boundary = ProverGrindingRoundBoundary::<Domain>::with_round_offset(3, 0);
    let got = arithmetic::prove_encoded_zerocheck(
        &mut prover,
        &f,
        &tau,
        low,
        high,
        encoded,
        &mut boundary,
    )
    .unwrap();
    assert_eq!(expected, got);
    assert_eq!(prover.state_digest(), reference.state_digest());
    let nonces = boundary.into_nonces();
    assert_eq!(nonces, reference_boundary.into_nonces());
    assert_eq!(nonces.len(), 3);
    let mut verifier = Blake3Transcript::new();
    let mut boundary = VerifierGrindingRoundBoundary::<Domain>::new(3, &nonces);
    let got: OuterOutput<_> = got.into();
    verify_outer_sumcheck(
        &f,
        &mut verifier,
        f.zero(),
        &tau,
        &got.proof,
        got.evaluations,
        &mut boundary,
    )
    .unwrap();
    assert_eq!(prover.state_digest(), verifier.state_digest());
}
#[cfg(feature = "parallel")]
#[test]
fn parallel_first_fold_matches_serial_and_allows_concurrent_proofs() {
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let f = field();
    // Cross the first-fold cutoff; workers share immutable native inputs.
    for n in [12, 13, 14] {
        let a: Vec<u64> = (0..1usize << n)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let b: Vec<u64> = a.iter().map(|v| v.rotate_left(27)).collect();
        let c: Vec<u128> = a
            .iter()
            .zip(&b)
            .map(|(a, b)| *a as u128 * *b as u128)
            .collect();
        let tau: Vec<_> = (0..n)
            .map(|i| fe(&f, if i < 2 { i as u64 } else { i as u64 + 7 }))
            .collect();
        for zero in [false, true] {
            let prove = || {
                let mut prover = Blake3Transcript::new();
                let out = if zero {
                    crate::sumcheck::outer::prove_outer_sumcheck(
                        &f,
                        &mut prover,
                        crate::sumcheck::outer::OuterClaim::RowwiseZero,
                        &tau,
                        crate::sumcheck::outer::OuterSlices {
                            ax: &a,
                            bx: &b,
                            cx: &c,
                        },
                        None,
                        &mut UngrindedRoundBoundary,
                    )
                } else {
                    crate::sumcheck::outer::prove_outer_sumcheck(
                        &f,
                        &mut prover,
                        crate::sumcheck::outer::OuterClaim::Sum(f.zero()),
                        &tau,
                        crate::sumcheck::outer::OuterSlices {
                            ax: &a,
                            bx: &b,
                            cx: &c,
                        },
                        None,
                        &mut UngrindedRoundBoundary,
                    )
                }
                .unwrap();
                let mut verifier = Blake3Transcript::new();
                verify_outer_sumcheck(
                    &f,
                    &mut verifier,
                    f.zero(),
                    &tau,
                    &out.proof,
                    out.evaluations,
                    &mut UngrindedRoundBoundary,
                )
                .unwrap();
                assert_eq!(prover.state_digest(), verifier.state_digest());
                (out, prover.state_digest())
            };
            let expected = serial.install(prove);
            let (left, right) = parallel.install(|| rayon::join(prove, prove));
            assert_eq!(left, expected);
            assert_eq!(right, expected);
        }
    }
}

#[test]
fn unsigned_preparation_can_be_reused_for_signed_values() {
    use field::PreparedLinearCombination;
    let f = field();
    let p =
        <FpCtx<2> as PreparedLinearCombination<Uint<1>>>::prepare_linear_combination(&f, [f.one()]);
    let v = <FpCtx<2> as PreparedLinearCombination<Z<1>>>::linear_combination(&p, |_| Z::ONE);
    assert_eq!(v, f.one());
}

#[test]
fn zero_vectors_are_initialized_field_identities() {
    let f = field();
    for size in [0, 1, 17, 4096] {
        let mut v = f.zero_vec(size);
        assert_eq!(v, vec![f.zero(); size]);
        for x in &mut v {
            *x = f.add(x, &f.one());
        }
        assert!(v.iter().all(|x| *x == f.one()));
        assert_eq!((&f).zero_vec(size), vec![f.zero(); size]);
    }
}

#[test]
fn bounded_product_reduction_matches_biguint_across_fast_path_boundary() {
    use field::FpProductAcc;
    use num_bigint::BigUint;
    for p in [(1u128 << 100) - 15, (1u128 << 127) - 1, u128::MAX - 158] {
        let f = FpCtx::from_prime_u128(p);
        let a = f.from_integer(&(p - 1));
        let b = f.from_integer(&(p - 2));
        let expected_product = BigUint::from(p - 1) * BigUint::from(p - 2);
        assert_eq!(
            f.reduce_product_with_public_bound(FpProductAcc::default(), 0),
            f.zero()
        );
        let mut accumulator = FpProductAcc::default();
        accumulator.accumulate(&a, &b);
        // Exact merges cross m*p <= R without allocating enormous tables.
        for power in 0..=29 {
            let terms = 1usize << power;
            let expected = (&expected_product * BigUint::from(terms)) % BigUint::from(p);
            let result = f.reduce_product_with_public_bound(accumulator, terms);
            assert_eq!(BigUint::from(u128::from(f.to_integer(&result))), expected);
            let loose = f.reduce_product_with_public_bound(accumulator, terms + 1);
            assert_eq!(loose, result);
            accumulator += accumulator;
        }
        let mut accumulator = FpProductAcc::default();
        for terms in 1..=7 {
            accumulator.accumulate(&a, &b);
            let result = f.reduce_product_with_public_bound(accumulator, terms);
            let expected = (&expected_product * BigUint::from(terms)) % BigUint::from(p);
            assert_eq!(BigUint::from(u128::from(f.to_integer(&result))), expected);
        }
    }
}

#[test]
fn row_storage_matches_retained_ordinary_arithmetic() {
    use crate::piop::spartan::raw_monty::make_equality_factors_raw;
    use arithmetic::{NativeInput, NativeProducts, NativeWideProducts};
    use field::WideMul;
    let f = field();
    for n in 0..=5 {
        let rows = 1usize << n;
        let live = if n == 0 { 1 } else { rows - 1 };
        let tau: Vec<_> = (0..n).map(|i| fe(&f, [0, 1, 7][i % 3])).collect();
        for known_zero in [false, true] {
            let a32: Vec<u64> = (0..rows).map(|i| u64::from(u32::MAX) - i as u64).collect();
            let b32 = a32.clone();
            let mut c32: Vec<_> = a32.iter().zip(&b32).map(|(a, b)| a * b).collect();
            let a64: Vec<_> = (0..live).map(|i| u64::MAX - i as u64).collect();
            let b64 = a64.clone();
            let full64: Vec<_> = a64.iter().map(|a| *a as u128 * *a as u128).collect();
            let mut lo64: Vec<_> = full64.iter().map(|c| *c as u64).collect();
            let hi64: Vec<_> = full64.iter().map(|c| (c >> 64) as u64).collect();
            let a128: Vec<_> = (0..live).map(|i| u128::MAX - i as u128).collect();
            let b128 = a128.clone();
            let full128: Vec<_> = a128
                .iter()
                .map(|a| field::IntegerOps.mul_wide(&Uint::<2>::from(*a), &Uint::<2>::from(*a)))
                .collect();
            let mut lo128: Vec<_> = full128
                .iter()
                .map(|c| u128::from(Uint::from_words(*c.as_parts().0)))
                .collect();
            let hi128: Vec<_> = full128
                .iter()
                .map(|c| u128::from(Uint::from_words(*c.as_parts().1)))
                .collect();
            let claim = if known_zero {
                f.zero()
            } else {
                c32[0] -= 1;
                lo64[0] -= 1;
                lo128[0] -= 1;
                eq_table(&tau, &f).unwrap()[0]
            };
            let a4096: Vec<_> = (0..rows)
                .map(|i| {
                    Uint::<64>::from_words(core::array::from_fn(|j| u64::MAX - (i + j) as u64))
                })
                .collect();
            let mut c4096 = a4096.clone();
            if !known_zero {
                c4096[0] = c4096[0].wrapping_sub(&Uint::ONE);
            }
            let large = OuterInputs {
                ax: a4096,
                bx: vec![Uint::<64>::ONE; rows],
                cx: c4096,
            };
            let signed = OuterInputs {
                ax: vec![Z::<2>::from(-7i128); rows],
                bx: vec![Z::<2>::ONE; rows],
                cx: (0..rows)
                    .map(|i| Z::<4>::from(-7i128 - i128::from(!known_zero && i == 0)))
                    .collect(),
            };
            for input in [
                NativeInput::Integers4096(large.clone()),
                NativeInput::Signed128(signed.clone()),
                NativeInput::U32(NativeProducts {
                    az: &a32,
                    bz: &b32,
                    cz: &c32,
                }),
                NativeInput::U64(NativeWideProducts::new(&a64, &b64, &lo64, &hi64, rows)),
                NativeInput::U128(NativeWideProducts::new(&a128, &b128, &lo128, &hi128, rows)),
            ] {
                let (low, high) = make_equality_factors_raw(&f, &tau);
                let mut expected_t = Blake3Transcript::new();
                let expected = arithmetic::prove_native_reference(
                    &mut expected_t,
                    &f,
                    &f,
                    claim,
                    &tau,
                    low.clone(),
                    high.clone(),
                    input.clone(),
                    known_zero,
                )
                .unwrap();
                let mut actual_t = Blake3Transcript::new();
                let mode = if known_zero {
                    OuterClaim::RowwiseZero
                } else {
                    OuterClaim::Sum(claim)
                };
                let factors = Some(arithmetic::factors_from_raw(&f, low, high));
                macro_rules! prove {
                    ($rows:expr) => {
                        prove_outer_sumcheck(
                            &f,
                            &mut actual_t,
                            mode,
                            &tau,
                            $rows,
                            factors,
                            &mut UngrindedRoundBoundary,
                        )
                        .map(crate::sumcheck::proof::OuterSumcheckOutput::from)
                    };
                }
                let actual = match input {
                    NativeInput::Integers4096(rows) => prove!(rows),
                    NativeInput::Signed128(rows) => prove!(rows),
                    NativeInput::U32(rows) => prove!(rows),
                    NativeInput::U64(rows) => prove!(rows),
                    NativeInput::U128(rows) => prove!(rows),
                    NativeInput::Residues(_) => unreachable!("not an integer fixture"),
                }
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual_t.state_digest(), expected_t.state_digest());
            }
        }
    }
}

#[test]
fn skip_zero_one_coordinates_vanishing_scale_and_grinding() {
    use crate::sumcheck::boundary::{ProverGrindingRoundBoundary, VerifierGrindingRoundBoundary};
    struct Domain;
    impl crate::piop::spartan::grinding::GrindingDomain for Domain {
        const DOMAIN: &'static [u8] = b"generic-skip-boundary-test/v1";
    }
    let f = field();
    for k in 1..=4u8 {
        let a: Vec<u64> = (0..1usize << (k + 2))
            .map(|i| u64::MAX - i as u64)
            .collect();
        let b: Vec<u64> = a.iter().map(|v| v.rotate_left(23)).collect();
        let c: Vec<u128> = a
            .iter()
            .zip(&b)
            .map(|(a, b)| *a as u128 * *b as u128)
            .collect();
        let tau = [f.one(), f.zero()];
        let prepared = prepare_univariate_skip(&f, k).unwrap();
        let out = crate::sumcheck::outer::prove_outer_zerocheck_with_skip(
            &f,
            &mut ZeroChallenges,
            &prepared,
            &tau,
            crate::sumcheck::outer::OuterSlices {
                ax: &a,
                bx: &b,
                cx: &c,
            },
            None,
            &mut UngrindedRoundBoundary,
        )
        .unwrap();
        assert_eq!(out.prefix_challenge, f.zero());
        assert_eq!(out.tail.final_claim, f.zero());
        verify_outer_zerocheck_with_skip(
            &f,
            &mut ZeroChallenges,
            &prepared,
            &tau,
            &out.prefix,
            &out.tail.proof,
            out.tail.evaluations,
            &mut UngrindedRoundBoundary,
        )
        .unwrap();
        let mut prover = Blake3Transcript::new();
        let mut boundary = ProverGrindingRoundBoundary::<Domain>::with_round_offset(3, 0);
        let out = crate::sumcheck::outer::prove_outer_zerocheck_with_skip(
            &f,
            &mut prover,
            &prepared,
            &tau,
            crate::sumcheck::outer::OuterSlices {
                ax: &a,
                bx: &b,
                cx: &c,
            },
            None,
            &mut boundary,
        )
        .unwrap();
        let nonces = boundary.into_nonces();
        assert_eq!(nonces.len(), tau.len());
        let mut verifier = Blake3Transcript::new();
        let mut boundary = VerifierGrindingRoundBoundary::<Domain>::new(3, &nonces);
        verify_outer_zerocheck_with_skip(
            &f,
            &mut verifier,
            &prepared,
            &tau,
            &out.prefix,
            &out.tail.proof,
            out.tail.evaluations,
            &mut boundary,
        )
        .unwrap();
        assert_eq!(prover.state_digest(), verifier.state_digest());
    }
}

#[cfg(feature = "parallel")]
#[test]
fn skip_all_k_serial_parallel_proofs_and_transcripts_match() {
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(10)
        .build()
        .unwrap();
    let f = field();
    for k in 1..=4u8 {
        // Both message buckets and fused prefix folding cross their cutoffs.
        let n = usize::from(k) + 12;
        let a: Vec<u128> = (0..1usize << n).map(|i| u128::MAX - i as u128).collect();
        let b: Vec<_> = a.iter().map(|v| v.rotate_left(29)).collect();
        let c: Vec<Uint<4>> = a
            .iter()
            .zip(&b)
            .map(|(a, b)| {
                use field::WideMul;
                *field::IntegerOps
                    .mul_wide(&Uint::<2>::from(*a), &Uint::<2>::from(*b))
                    .checked_resize_ct()
                    .value()
            })
            .collect();
        let tau: Vec<_> = (0..12).map(|i| fe(&f, [0, 1, 7][i % 3])).collect();
        let prepared = prepare_univariate_skip(&f, k).unwrap();
        let prove = || {
            let mut prover = Blake3Transcript::new();
            let out = crate::sumcheck::outer::prove_outer_zerocheck_with_skip(
                &f,
                &mut prover,
                &prepared,
                &tau,
                crate::sumcheck::outer::OuterSlices {
                    ax: &a,
                    bx: &b,
                    cx: &c,
                },
                None,
                &mut UngrindedRoundBoundary,
            )
            .unwrap();
            let mut verifier = Blake3Transcript::new();
            verify_outer_zerocheck_with_skip(
                &f,
                &mut verifier,
                &prepared,
                &tau,
                &out.prefix,
                &out.tail.proof,
                out.tail.evaluations,
                &mut UngrindedRoundBoundary,
            )
            .unwrap();
            assert_eq!(prover.state_digest(), verifier.state_digest());
            (out, prover.state_digest())
        };
        assert_eq!(serial.install(prove), parallel.install(prove));
    }
}

#[test]
fn direct_row_api_rejects_inconsistent_mle_metadata_before_absorption() {
    let f = field();
    let mut products = R1csProductMles {
        az: DenseMultilinearExtension::from_evaluations_vec(2, vec![f.one(); 4], f.zero()),
        bz: DenseMultilinearExtension::from_evaluations_vec(2, vec![f.one(); 4], f.zero()),
        cz: DenseMultilinearExtension::from_evaluations_vec(2, vec![f.one(); 4], f.zero()),
    };
    // Lengths alone agree, but one MLE declares the wrong number of variables.
    products.bz.num_vars = 1;
    let mut t = Blake3Transcript::new();
    let before = t.state_digest();
    for claim in [OuterClaim::Sum(f.zero()), OuterClaim::RowwiseZero] {
        assert_eq!(
            prove_outer_sumcheck(
                &f,
                &mut t,
                claim,
                &[f.one(); 2],
                &products,
                None,
                &mut UngrindedRoundBoundary
            ),
            Err(SumcheckError::InvalidProductDimensions)
        );
        assert_eq!(t.state_digest(), before);
    }
    let prepared = prepare_univariate_skip(&f, 1).unwrap();
    assert_eq!(
        prove_outer_zerocheck_with_skip(
            &f,
            &mut t,
            &prepared,
            &[f.one()],
            &products,
            None,
            &mut UngrindedRoundBoundary
        ),
        Err(SumcheckError::InvalidProductDimensions)
    );
    assert_eq!(t.state_digest(), before);
    let invalid = DenseMultilinearExtension {
        num_vars: usize::MAX,
        evaluations: vec![f.one()],
    };
    assert!(matches!(
        EqualityFactors::from_mles((invalid, DenseMultilinearExtension::zero_vars(f.one())), &f),
        Err(SumcheckError::InvalidEqualityDimensions)
    ));
}
