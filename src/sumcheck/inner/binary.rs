//! The post-GKR quadratic batch, retaining its evaluation-form wire codec.
use super::evaluation_form::MultiDegreeSumcheckProof;
use crate::piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use crate::poly::coefficient::PolynomialField;
use crate::transcript::traits::Transcript;
use field::{Gf128 as F, Gf128Ops, Gf128Product, PreparedGf128Mul};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// The first pass also obtains the claims. Subsequent passes need only c0,c2.
fn first_round([a, b]: &[Vec<F>; 2]) -> [F; 3] {
    let zero = || [Gf128Product::zero(); 3];
    let accumulate = |mut acc: [Gf128Product; 3], (a, b): (&[F], &[F])| {
        acc[0] ^= a[0].mul_unreduced(b[0]);
        acc[1] ^= (a[0] + a[1]).mul_unreduced(b[0] + b[1]);
        acc[2] ^= a[1].mul_unreduced(b[1]);
        acc
    };
    #[cfg(feature = "parallel")]
    let acc = if crate::sumcheck::arithmetic::should_parallelize(a.len() / 2) {
        a.par_chunks_exact(2)
            .zip(b.par_chunks_exact(2))
            .fold(zero, accumulate)
            .reduce(zero, |mut a, b| {
                for i in 0..3 {
                    a[i] ^= b[i];
                }
                a
            })
    } else {
        a.chunks_exact(2)
            .zip(b.chunks_exact(2))
            .fold(zero(), accumulate)
    };
    #[cfg(not(feature = "parallel"))]
    let acc = a
        .chunks_exact(2)
        .zip(b.chunks_exact(2))
        .fold(zero(), accumulate);
    acc.map(Gf128Product::reduce)
}

/// In-place low-coordinate fold and next-round coefficients. Reading each four
/// source entries before writing the two outputs keeps the original allocation.
fn fold_round([a, b]: &mut [Vec<F>; 2], challenge: F) -> [F; 2] {
    let n = a.len() / 2;
    let prepared = PreparedGf128Mul::new(challenge);
    if n == 1 {
        a[0] = a[0] + prepared.mul(&(a[0] + a[1]));
        b[0] = b[0] + prepared.mul(&(b[0] + b[1]));
        a.truncate(1);
        b.truncate(1);
        return [F::ZERO; 2];
    }
    let mut acc = [Gf128Product::zero(); 2];
    for i in 0..n / 2 {
        let j = 4 * i;
        let a0 = a[j] + prepared.mul(&(a[j] + a[j + 1]));
        let a1 = a[j + 2] + prepared.mul(&(a[j + 2] + a[j + 3]));
        let b0 = b[j] + prepared.mul(&(b[j] + b[j + 1]));
        let b1 = b[j + 2] + prepared.mul(&(b[j + 2] + b[j + 3]));
        a[2 * i] = a0;
        a[2 * i + 1] = a1;
        b[2 * i] = b0;
        b[2 * i + 1] = b1;
        acc[0] ^= a0.mul_unreduced(b0);
        acc[1] ^= (a0 + a1).mul_unreduced(b0 + b1);
    }
    a.truncate(n);
    b.truncate(n);
    acc.map(Gf128Product::reduce)
}

/// Check the public GKR domain and retain the existing owned pair storage.
pub(crate) fn inputs(pairs: Vec<[Vec<F>; 2]>, num_vars: usize) -> (Vec<[Vec<F>; 2]>, Vec<()>) {
    assert!(num_vars > 0 && !pairs.is_empty());
    assert!(
        pairs
            .iter()
            .all(|[a, b]| a.len() == 1usize << num_vars && b.len() == a.len())
    );
    let weights = vec![(); pairs.len()];
    (pairs, weights)
}

pub struct State {
    pair: [Vec<F>; 2],
    coefficients: [F; 2],
    initial: F,
    num_vars: usize,
}
pub struct EvaluationCodec;
impl super::input::Codec<Gf128Ops> for EvaluationCodec {
    fn start(
        _: &Gf128Ops,
        t: &mut impl Transcript,
        rounds: usize,
        claims: usize,
    ) -> Result<(), super::SumcheckError> {
        let mut buf = [0; 16];
        for n in [rounds, claims]
            .into_iter()
            .chain(core::iter::repeat_n(2, claims))
        {
            t.absorb_random_field(&F::interpolation_node(n as u64, &()), &mut buf);
        }
        Ok(())
    }
    fn absorb(_: &Gf128Ops, t: &mut impl Transcript, &[c0, c1, c2]: &[F; 3]) {
        let node = F::interpolation_node(2, &());
        t.absorb_random_field_slice(&[c0 + c1 + c2, c0 + node * (c1 + node * c2)], &mut [0; 16]);
    }
    fn challenge(_: &Gf128Ops, t: &mut impl Transcript) -> Result<F, super::SumcheckError> {
        let r = t.get_field_challenge(&());
        t.absorb_random_field(&r, &mut [0; 16]);
        Ok(r)
    }
}
impl super::input::sealed::Input for [Vec<F>; 2] {}
impl super::input::Input<Gf128Ops> for [Vec<F>; 2] {
    type Weights = ();
    type State = State;
    type Codec = EvaluationCodec;
    fn prepare(self, _: &Gf128Ops, _: ()) -> Result<State, super::SumcheckError> {
        if self[0].len() < 2 || !self[0].len().is_power_of_two() || self[1].len() != self[0].len() {
            return Err(super::SumcheckError::InvalidProductDimensions);
        }
        let num_vars = self[0].len().ilog2() as usize;
        let [c0, c2, odd] = first_round(&self);
        Ok(State {
            pair: self,
            coefficients: [c0, c2],
            initial: c0 + odd,
            num_vars,
        })
    }
}
impl super::input::State<Gf128Ops> for State {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
    fn initial_claim(&self) -> Option<F> {
        Some(self.initial)
    }
    fn coefficients(&self, _: &Gf128Ops) -> Result<[F; 2], super::SumcheckError> {
        Ok(self.coefficients)
    }
    fn fold(&mut self, _: &Gf128Ops, r: &F) -> Result<(), super::SumcheckError> {
        self.coefficients = fold_round(&mut self.pair, *r);
        Ok(())
    }
    fn terminal(&self, _: &Gf128Ops) -> Result<[F; 2], super::SumcheckError> {
        Ok([self.pair[0][0], self.pair[1][0]])
    }
}

/// Preserve the enclosing protocol's evaluation-form proof envelope. This does
/// not run a sumcheck or mutate the transcript.
pub(crate) fn encode(
    output: super::DynamicBatchedInnerSumcheckOutput<F>,
) -> (MultiDegreeSumcheckProof<F>, Vec<F>) {
    let node = F::interpolation_node(2, &());
    let claimed_sums = output
        .proofs
        .iter()
        .map(|p| {
            let [_, c1, c2] = p.round_polynomials[0];
            c1 + c2
        })
        .collect();
    let messages = output
        .proofs
        .into_iter()
        .map(|p| {
            p.round_polynomials
                .into_iter()
                .map(|[c0, c1, c2]| {
                    ProverMsg(NatEvaluatedPolyWithoutConstant {
                        tail_evaluations: vec![c0 + c1 + c2, c0 + node * (c1 + node * c2)],
                    })
                })
                .collect()
        })
        .collect();
    (
        MultiDegreeSumcheckProof::quadratic(messages, claimed_sums),
        output.point,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piop::sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckGroup};
    use crate::poly::mle::DenseMultilinearExtension;
    use crate::transcript::Blake3Transcript;
    #[test]
    fn evaluation_codec_and_shared_challenges_match_legacy() {
        for (vars, k) in [(1, 1), (3, 3), (14, 2)] {
            let pairs: Vec<[Vec<F>; 2]> = (0..k)
                .map(|k| {
                    core::array::from_fn(|j| {
                        (0..1 << vars)
                            .map(|i| {
                                F::new(
                                    (i as u64).wrapping_mul(0x9e3779b97f4a7c15) + (k + j) as u64,
                                    i as u64,
                                )
                            })
                            .collect()
                    })
                })
                .collect();
            let groups = pairs
                .iter()
                .map(|pair| {
                    MultiDegreeSumcheckGroup::new(
                        2,
                        pair.iter()
                            .map(|v| {
                                DenseMultilinearExtension::from_evaluations_vec(
                                    vars,
                                    v.iter()
                                        .map(|f| field::Uint::from_words([f.lo, f.hi]))
                                        .collect(),
                                    field::Uint::ZERO,
                                )
                            })
                            .collect(),
                        Box::new(|v: &[F]| v[0] * v[1]),
                    )
                })
                .collect();
            let mut old = Blake3Transcript::new();
            let (expected, states) =
                MultiDegreeSumcheck::prove_as_subprotocol(&mut old, groups, vars, &());
            let mut new = Blake3Transcript::new();
            let (actual, point) = {
                let (values, weights) = crate::sumcheck::inner::binary::inputs(pairs, vars);
                crate::sumcheck::inner::binary::encode(
                    crate::sumcheck::inner::prove_batched_inner_sumcheck(
                        &field::Gf128Ops,
                        &mut new,
                        crate::sumcheck::inner::InitialClaims::Compute,
                        values,
                        weights,
                        &mut crate::sumcheck::UngrindedRoundBoundary,
                    )
                    .expect("valid post-GKR dot products"),
                )
            };
            assert_eq!(actual, expected);
            assert_eq!(point, states[0].randomness);
            assert_eq!(
                new.get_field_challenge::<F>(&()),
                old.get_field_challenge::<F>(&())
            );
            let verified = actual
                .verify_as_subprotocol(&mut Blake3Transcript::new(), vars, &vec![2; k], &())
                .unwrap();
            assert_eq!(point, verified.point());
        }
    }
}
