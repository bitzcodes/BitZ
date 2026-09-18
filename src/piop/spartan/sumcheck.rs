//! Spartan protocol composition types; inner provers live in `crate::sumcheck::inner`.
use crate::poly::mle::DenseMultilinearExtension;
pub(crate) use crate::sumcheck::arithmetic::*;
#[cfg(test)]
pub(crate) use crate::sumcheck::inner::reference::*;
pub use crate::sumcheck::proof::OuterSumcheckProof;
pub use crate::sumcheck::{SumcheckError, SumcheckProof};
pub(crate) use crate::sumcheck::{boundary::*, proof::*};

/// Dense Bit-row MLEs for `Az`, `Bz`, and `Cz`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csProductMles<F> {
    pub az: DenseMultilinearExtension<F>,
    pub bz: DenseMultilinearExtension<F>,
    pub cz: DenseMultilinearExtension<F>,
}

/// Prover-local result of the Spartan inner sumcheck.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InnerSumcheckOutput<F> {
    pub sumcheck: SumcheckProverOutput<F, 3>,
    pub batched_matrix_evaluation: F,
    pub witness_evaluation: F,
}

#[cfg(test)]
impl<F> From<crate::sumcheck::inner::InnerSumcheckOutput<F>> for InnerSumcheckOutput<F> {
    fn from(out: crate::sumcheck::inner::InnerSumcheckOutput<F>) -> Self {
        let [batched_matrix_evaluation, witness_evaluation] = out.terminal_evaluations;
        Self {
            sumcheck: SumcheckProverOutput {
                proof: out.proof,
                eval_points: out.point,
                final_claim: out.final_claim,
            },
            batched_matrix_evaluation,
            witness_evaluation,
        }
    }
}
