use crate::piop::spartan::grinding::GrindingError;
use crate::utils::delayed_reduction::DelayedReductionError;
use thiserror::Error;
/// Failures produced while reducing or checking a sumcheck claim.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum SumcheckError {
    #[error(transparent)]
    Matrix(#[from] crate::piop::spartan::matrix::SpartanMatrixError),
    #[error("field challenge sampling exhausted its rejection budget")]
    SamplingExhausted,
    #[error("univariate skip nodes collide in this field")]
    InvalidSkipField,
    #[error("a sumcheck round polynomial must contain at least one coefficient")]
    EmptyRoundPolynomial,
    #[error("sumcheck proof has {actual} rounds, expected {expected}")]
    InvalidRoundCount { expected: usize, actual: usize },
    #[error("sumcheck claim is inconsistent in round {round}")]
    InvalidRoundClaim { round: usize },
    #[error("sumcheck terminal claim is inconsistent")]
    InvalidTerminalClaim,
    #[error("sumcheck product tables have incompatible dimensions")]
    InvalidProductDimensions,
    #[error("sumcheck equality tables have incompatible dimensions")]
    InvalidEqualityDimensions,
    #[error("native u32 product tables contain a multiplicand wider than 32 bits")]
    NativeMultiplicandOutOfRange,
    #[error("invalid dense multilinear-extension table")]
    InvalidMleOperation,
    #[error("a sumcheck value uses a different field configuration")]
    FieldConfigurationMismatch,
    #[error("a sumcheck value has a noncanonical field representation")]
    NonCanonicalFieldElement,
    #[error(transparent)]
    Grinding(#[from] GrindingError),
    #[error("sumcheck proof has {actual} grinding nonces, expected {expected}")]
    InvalidGrindingNonceCount { expected: usize, actual: usize },
    #[error(transparent)]
    DelayedReduction(#[from] DelayedReductionError),
}
