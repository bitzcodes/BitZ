//! Sealed conversions from supported storage into arithmetic-only round states.
//! Callers use ordinary vectors/arrays or the supplied representation constructors.
use super::*;

pub enum InitialClaims<'a, E> {
    Known(&'a [E]),
    /// Available for representations that obtain the sum in their first pass.
    Compute,
}
impl<'a, E, const K: usize> From<&'a [E; K]> for InitialClaims<'a, E> {
    fn from(value: &'a [E; K]) -> Self {
        Self::Known(value)
    }
}
impl<'a, E> From<&'a [E]> for InitialClaims<'a, E> {
    fn from(value: &'a [E]) -> Self {
        Self::Known(value)
    }
}

pub trait State<F: FieldOps>: Send {
    fn num_vars(&self) -> usize;
    fn validate_claim(&self, _field: &F, _claim: &F::Elem) -> Result<(), SumcheckError> {
        Ok(())
    }
    fn coefficients(&self, field: &F) -> Result<[F::Elem; 2], SumcheckError>;
    fn fold(&mut self, field: &F, challenge: &F::Elem) -> Result<(), SumcheckError>;
    fn terminal(&self, field: &F) -> Result<[F::Elem; 2], SumcheckError>;
    fn initial_claim(&self) -> Option<F::Elem> {
        None
    }
}
pub trait Codec<F: FieldOps> {
    fn start(
        _field: &F,
        _t: &mut impl Transcript,
        _rounds: usize,
        _claims: usize,
    ) -> Result<(), SumcheckError> {
        Ok(())
    }
    fn absorb(field: &F, t: &mut impl Transcript, message: &[F::Elem; 3]);
    fn challenge(field: &F, t: &mut impl Transcript) -> Result<F::Elem, SumcheckError>;
}
pub struct Canonical;
impl<F: FieldOps> Codec<F> for Canonical
where
    F::Elem: SpartanField<Config = F>,
{
    fn absorb(field: &F, t: &mut impl Transcript, message: &[F::Elem; 3]) {
        absorb_field_elements(t, message, field);
    }
    fn challenge(field: &F, t: &mut impl Transcript) -> Result<F::Elem, SumcheckError> {
        squeeze_field(t, field)
    }
}
pub(crate) mod sealed {
    pub trait Input {}
    pub trait Batch {}
}
pub trait Input<F: FieldOps>: sealed::Input {
    type Weights;
    type State: State<F>;
    type Codec: Codec<F>;
    fn prepare(self, field: &F, weights: Self::Weights) -> Result<Self::State, SumcheckError>;
}
impl<T> sealed::Input for Vec<T> {}
impl<F, T> Input<F> for Vec<T>
where
    F: FieldOps
        + PreparedLinearCombination<T>
        + BatchMulAcc<Elem<F>, T>
        + BatchMulAcc<Elem<F>>
        + Sync,
    F: Reduce<Acc<F, T>, Output = Elem<F>> + Reduce<Acc<F, Elem<F>>, Output = Elem<F>>,
    Elem<F>: SpartanField<Config = F>,
    T: Copy + Send + Sync,
{
    type Weights = Vec<F::Elem>;
    type State = dense::State<T, F::Elem>;
    type Codec = Canonical;
    fn prepare(self, field: &F, weights: Self::Weights) -> Result<Self::State, SumcheckError> {
        dense::State::new(field, self, weights)
    }
}

pub struct Slot<F: FieldOps, S> {
    pub state: S,
    pub claim: F::Elem,
    pub proof: SumcheckProof<F::Elem, 3>,
    pub terminal: [F::Elem; 2],
}
impl<F: FieldOps, S: State<F>> Slot<F, S> {
    fn new(field: &F, state: S) -> Self {
        let proof = SumcheckProof {
            round_polynomials: Vec::with_capacity(state.num_vars()),
        };
        Self {
            state,
            claim: field.zero(),
            proof,
            terminal: [field.zero(); 2],
        }
    }
}
pub trait Batch<F: FieldOps>: sealed::Batch {
    type Weights;
    type State: State<F>;
    type Codec: Codec<F>;
    type States: AsMut<[Slot<F, Self::State>]>;
    type Output;
    fn prepare(self, field: &F, weights: Self::Weights) -> Result<Self::States, SumcheckError>;
    fn finish(states: Self::States, point: Vec<F::Elem>) -> Self::Output;
}
impl<V, const K: usize> sealed::Batch for [V; K] {}
impl<V> sealed::Batch for Vec<V> {}
impl<F: FieldOps, V: Input<F>, const K: usize> Batch<F> for [V; K] {
    type Weights = [V::Weights; K];
    type State = V::State;
    type Codec = V::Codec;
    type States = [Slot<F, V::State>; K];
    type Output = BatchedInnerSumcheckOutput<F::Elem, K>;
    fn prepare(self, field: &F, weights: Self::Weights) -> Result<Self::States, SumcheckError> {
        let mut slots = core::array::from_fn::<_, K, _>(|_| None);
        for ((slot, values), weights) in slots.iter_mut().zip(self).zip(weights) {
            *slot = Some(Slot::new(field, values.prepare(field, weights)?));
        }
        Ok(slots.map(Option::unwrap))
    }
    fn finish(states: Self::States, point: Vec<F::Elem>) -> Self::Output {
        let final_claims = states.each_ref().map(|s| s.claim);
        let terminal_evaluations = states.each_ref().map(|s| s.terminal);
        let proofs = states.map(|s| s.proof);
        BatchedInnerSumcheckOutput {
            proofs,
            point,
            final_claims,
            terminal_evaluations,
        }
    }
}
impl<F: FieldOps, V: Input<F>> Batch<F> for Vec<V> {
    type Weights = Vec<V::Weights>;
    type State = V::State;
    type Codec = V::Codec;
    type States = Vec<Slot<F, V::State>>;
    type Output = DynamicBatchedInnerSumcheckOutput<F::Elem>;
    fn prepare(self, field: &F, weights: Self::Weights) -> Result<Self::States, SumcheckError> {
        if self.len() != weights.len() {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        self.into_iter()
            .zip(weights)
            .map(|(v, w)| Ok(Slot::new(field, v.prepare(field, w)?)))
            .collect()
    }
    fn finish(states: Self::States, point: Vec<F::Elem>) -> Self::Output {
        let final_claims = states.iter().map(|s| s.claim).collect();
        let terminal_evaluations = states.iter().map(|s| s.terminal).collect();
        let proofs = states.into_iter().map(|s| s.proof).collect();
        DynamicBatchedInnerSumcheckOutput {
            proofs,
            point,
            final_claims,
            terminal_evaluations,
        }
    }
}
