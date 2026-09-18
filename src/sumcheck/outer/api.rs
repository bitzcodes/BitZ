//! Owned and borrowed inputs for equality-weighted outer sumchecks.
use super::{OuterArithmetic, engine::RoundState, inputs::OuterRows, ordinary::*, traversal::*};
use crate::piop::spartan::{SpartanField, matrix::make_equality_factors};
use crate::sumcheck::{
    RoundBoundaryPolicy, SumcheckError, SumcheckProof, proof::OuterSumcheckOutput,
};
use crate::transcript::traits::Transcript;
use field::{BatchMulAcc, MergeAccumulator, Reduce};
use field::{FieldOps, RingOps};

/// A and B have the same input type; C can hold wider exact products.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterInputs<AB, C = AB> {
    pub ax: Vec<AB>,
    pub bx: Vec<AB>,
    pub cx: Vec<C>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterEvaluations<E> {
    pub ax: E,
    pub bx: E,
    pub cx: E,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterOutput<E> {
    pub proof: SumcheckProof<E, 4>,
    pub point: Vec<E>,
    pub final_claim: E,
    pub evaluations: OuterEvaluations<E>,
}
impl<E> From<OuterSumcheckOutput<E>> for OuterOutput<E> {
    fn from(out: OuterSumcheckOutput<E>) -> Self {
        Self {
            proof: out.proof.sumcheck,
            point: out.eval_points,
            final_claim: out.final_claim,
            evaluations: OuterEvaluations {
                ax: out.proof.az_mle_claim,
                bx: out.proof.bz_mle_claim,
                cx: out.proof.cz_mle_claim,
            },
        }
    }
}

impl<E> From<OuterOutput<E>> for OuterSumcheckOutput<E> {
    fn from(out: OuterOutput<E>) -> Self {
        Self {
            proof: crate::sumcheck::proof::OuterSumcheckProof {
                sumcheck: out.proof,
                az_mle_claim: out.evaluations.ax,
                bz_mle_claim: out.evaluations.bx,
                cz_mle_claim: out.evaluations.cx,
            },
            eval_points: out.point,
            final_claim: out.final_claim,
        }
    }
}

pub(super) fn validate_shape(a: usize, b: usize, c: usize) -> Result<usize, SumcheckError> {
    if !a.is_power_of_two() || a != b || a != c {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    Ok(a.ilog2() as usize)
}
/// The relation used to construct the first ordinary message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterClaim<E> {
    /// The equality-weighted sum of A·B−C; zero does not imply rowwise equality.
    Sum(E),
    /// A(x)B(x)=C(x) at every Boolean row. Enables the zerocheck first round.
    RowwiseZero,
}

/// Proves the equality-weighted outer relation directly from row storage.
///
/// Owned inputs are released after the first challenge fold. Borrow with `&rows`
/// to keep caller-owned storage. Supplied equality factors must be fresh tables
/// for `tau` in little-endian coordinate order; `None` prepares them here.
/// Arithmetic capabilities and thread-safe row access are resolved statically.
pub fn prove_outer_sumcheck<F, I: OuterRows>(
    field: &F,
    transcript: &mut impl Transcript,
    claim: OuterClaim<F::Elem>,
    tau: &[F::Elem],
    rows: I,
    factors: Option<EqualityFactors<F::Elem>>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<OuterOutput<F::Elem>, SumcheckError>
where
    F: OuterArithmetic<I::AB, I::C>,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    let (initial_claim, known_zero) = match claim {
        OuterClaim::Sum(value) => (value, false),
        OuterClaim::RowwiseZero => (field.zero(), true),
    };
    let prepared = prepare_first(
        field,
        transcript,
        initial_claim,
        tau,
        &rows,
        known_zero,
        boundary,
        factors,
    )?;
    drop(rows); // Release owned integers before continuing on field folds.
    finish_first(field, transcript, tau, prepared, boundary)
}

struct PreparedOrdinary<E> {
    state: RoundState<E>,
    factors: EqualityFactors<E>,
    products: R1csProductTableBuffers<E>,
    pending: Option<[E; 3]>,
    inverses: Vec<E>,
}

fn finish_first<F>(
    field: &F,
    transcript: &mut impl Transcript,
    tau: &[F::Elem],
    prepared: PreparedOrdinary<F::Elem>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<OuterOutput<F::Elem>, SumcheckError>
where
    F: FieldOps,
    F::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    #[cfg(feature = "bench-internals")]
    let _phase = super::measure::Phase::start(3);
    Ok(continue_field_with_inverses(
        transcript,
        field,
        field,
        tau,
        prepared.factors,
        prepared.products,
        prepared.state,
        prepared.pending,
        Some(prepared.inverses),
        boundary,
    )?
    .into())
}

#[allow(clippy::too_many_arguments)]
fn prepare_first<F, I: OuterRows>(
    field: &F,
    transcript: &mut impl Transcript,
    initial_claim: <F as RingOps>::Elem,
    tau: &[<F as RingOps>::Elem],
    rows: &I,
    known_zero: bool,
    boundary: &mut impl RoundBoundaryPolicy,
    prepared_factors: Option<EqualityFactors<F::Elem>>,
) -> Result<PreparedOrdinary<F::Elem>, SumcheckError>
where
    F: OuterArithmetic<I::AB, I::C>,
    <F as RingOps>::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    #[cfg(feature = "bench-internals")]
    let setup = super::measure::Phase::start(0);
    let n = rows.validate()?;
    let a = rows.dimensions().0;
    if tau.len() != n {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }
    boundary.validate(n)?;
    if known_zero && initial_claim != field.zero() {
        return Err(SumcheckError::InvalidTerminalClaim);
    }
    let mut factors = if let Some(factors) = prepared_factors {
        if !factors.matches_rows(a) {
            return Err(SumcheckError::InvalidEqualityDimensions);
        }
        factors
    } else {
        let (low, high) = make_equality_factors(tau, field)
            .map_err(|_| SumcheckError::InvalidEqualityDimensions)?;
        EqualityFactors::new(low.evaluations, high.evaluations, field)
    };
    let mut state = RoundState::new(initial_claim, n, field);
    if n == 0 {
        let [a, b, c] = PreparedFold::<_, _, 1>::new(field, [field.one()], rows).fold(0);
        return Ok(PreparedOrdinary {
            state,
            factors,
            products: R1csProductTableBuffers {
                az: vec![a],
                bz: vec![b],
                cz: vec![c],
            },
            pending: None,
            inverses: Vec::new(),
        });
    }
    #[cfg(feature = "bench-internals")]
    drop(setup);
    #[cfg(feature = "bench-internals")]
    let coefficients_phase = super::measure::Phase::start(1);
    let inverses = batch_invert_nonzero(tau, field);
    factors.strip(field);
    let endpoint = FactoredEndpoint::for_tau(&tau[0]);
    let evaluations = integer_buckets::<F, I::AB, I::C, 2>(
        field,
        factors.weights(),
        |acc, weights, pair, index| {
            let i = 2 * pair;
            let a0 = field.lift_ab(rows.a(i));
            let a1 = field.lift_ab(rows.a(i + 1));
            let b0 = field.lift_ab(rows.b(i));
            let b1 = field.lift_ab(rows.b(i + 1));
            let leading = field.product(field.sub_ab(a1, a0), field.sub_ab(b1, b0));
            field.accumulate::<false>(&mut acc[1], weights, index, leading);
            if !known_zero {
                let (a, b, i) = match endpoint {
                    FactoredEndpoint::Zero => (a0, b0, i),
                    #[cfg(test)]
                    FactoredEndpoint::KnownZero => (a0, b0, i),
                    FactoredEndpoint::One => (a1, b1, i + 1),
                };
                let residual = field.residual(field.product(a, b), field.lift_c(rows.c(i)));
                field.accumulate::<false>(&mut acc[0], weights, index, residual);
            }
        },
    )?;
    let coefficients = if known_zero {
        // (1−tau+(2tau−1)X) h₂ X(X−1), stored as [c₀,c₂,c₃].
        let e0 = field.sub(&field.one(), &tau[0]);
        let e1 = field.sub(&field.add(&tau[0], &tau[0]), &field.one());
        [
            field.zero(),
            field.mul(&field.sub(&e0, &e1), &evaluations[1]),
            field.mul(&e1, &evaluations[1]),
        ]
    } else {
        let inv = inverses[0];
        reconstruct_eq_factored_cubic_without_linear(
            &state.claim,
            &tau[0],
            &inv,
            endpoint,
            evaluations,
            &field.one(),
            &field.one(),
            field,
        )
    };
    let challenge = state.sample(field, transcript, tau, &coefficients, boundary)?;
    #[cfg(feature = "bench-internals")]
    drop(coefficients_phase);
    #[cfg(feature = "bench-internals")]
    let _fold = super::measure::Phase::start(2);
    let mut products = R1csProductTableBuffers::zeroed(a / 2, field);
    let coefficients = [field.sub(&field.one(), &challenge), challenge];
    let fold = PreparedFold::new(field, coefficients, rows);
    let endpoint = if n > 1 {
        FactoredEndpoint::for_tau(&tau[1])
    } else {
        FactoredEndpoint::Zero
    };
    let next = fold_and_message(field, &mut products, &mut factors, endpoint, fold)?;
    let pending = next.map(|values| {
        reconstruct_eq_factored_cubic_without_linear(
            &state.claim,
            &tau[1],
            &inverses[1],
            endpoint,
            values,
            &state.equality_scale,
            &field.one(),
            field,
        )
    });
    Ok(PreparedOrdinary {
        state,
        factors,
        products,
        pending,
        inverses,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterVerifierOutput<E> {
    pub point: Vec<E>,
    pub final_claim: E,
    pub evaluations: OuterEvaluations<E>,
}
/// Verifies the same four-coefficient proof for ordinary and zero first rounds.
pub fn verify_outer_sumcheck<F>(
    field: &F,
    transcript: &mut impl Transcript,
    initial_claim: <F as RingOps>::Elem,
    tau: &[<F as RingOps>::Elem],
    proof: &SumcheckProof<<F as RingOps>::Elem, 4>,
    evaluations: OuterEvaluations<<F as RingOps>::Elem>,
    boundary: &mut impl RoundBoundaryPolicy,
) -> Result<OuterVerifierOutput<<F as RingOps>::Elem>, SumcheckError>
where
    F: FieldOps,
    <F as RingOps>::Elem: SpartanField<Config = F>,
    F: BatchMulAcc<<F as field::RingOps>::Elem>
        + Reduce<
            <F as BatchMulAcc<<F as field::RingOps>::Elem>>::Accumulator,
            Output = <F as field::RingOps>::Elem,
        > + Sync,
{
    use crate::sumcheck::proof::{eq_eval, validate_field_elements};
    validate_field_elements(tau, field)?;
    validate_field_elements(&[evaluations.ax, evaluations.bx, evaluations.cx], field)?;
    let (point, final_claim) =
        proof.verify_with_round_boundary(transcript, initial_claim, tau.len(), field, boundary)?;
    let eq = eq_eval(tau, &point, field)?;
    let residual = field.sub(
        &field.mul(&evaluations.ax, &evaluations.bx),
        &evaluations.cx,
    );
    if final_claim != field.mul(&eq, &residual) {
        return Err(SumcheckError::InvalidTerminalClaim);
    }
    crate::piop::spartan::absorb_field_elements(
        transcript,
        &[evaluations.ax, evaluations.bx, evaluations.cx],
        field,
    );
    Ok(OuterVerifierOutput {
        point,
        final_claim,
        evaluations,
    })
}
