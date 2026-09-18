//! Borrowed native integer adapter to the generic prefix and ordinary engine.
#[cfg(test)]
use crate::piop::spartan::raw_monty::NativeProducts;
use crate::piop::spartan::raw_monty::Raw;
use crate::sumcheck::SumcheckError;
type Field = field::Fp<2>;
type FieldConfig = field::FpCtx<2>;

#[cfg(test)]
mod reference;
#[cfg(test)]
pub(crate) use reference::*;

/// Retained integer interpolation reference for differential tests.
#[cfg(test)]
pub(crate) fn prove_native_skip_reference(
    transcript: &mut impl crate::transcript::traits::Transcript,
    field: &FieldConfig,
    skip_vars: u8,
    tau_tail: &[Field],
    low: Vec<Raw>,
    high: Vec<Raw>,
    products: NativeProducts<'_>,
) -> Result<super::univariate::UnivariateSkipOuterSumcheckOutput<Field>, SumcheckError> {
    use super::univariate::{
        PrefixUnivariateRowBinding, UnivariateSkipOuterSumcheckOutput,
        UnivariateSkipOuterSumcheckProof, UnivariateSkipProof,
    };
    let k = usize::from(skip_vars);
    if !(1..=4).contains(&k)
        || !products.len().is_power_of_two()
        || products.len().ilog2() as usize != k + tau_tail.len()
        || products.bz.len() != products.len()
        || products.cz.len() != products.len()
    {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    let message = {
        let _scope = tracing::info_span!("spartan:univariate_skip_message").entered();
        encoded_native_message(field, k, &low, &high, products, field, field)?
    };
    let skip = UnivariateSkipProof::from_ordered_message(k, message)?;
    let reduction = skip.verify_reduction(transcript, field)?;
    let folded = {
        let _scope = tracing::info_span!("spartan:univariate_skip_prefix_fold").entered();
        fold_encoded_lagrange(k, products, &reduction.z, field)?
    };
    let tail = {
        let _scope = tracing::info_span!("spartan:univariate_skip_tail").entered();
        super::arithmetic::prepare_encoded_reference(
            transcript,
            field,
            field,
            reduction.q_at_z,
            tau_tail,
            low,
            high,
            folded,
            &mut crate::sumcheck::UngrindedRoundBoundary,
            false,
        )?
    };
    Ok(UnivariateSkipOuterSumcheckOutput {
        proof: UnivariateSkipOuterSumcheckProof {
            skip,
            tail: tail.proof,
        },
        row_binding: PrefixUnivariateRowBinding {
            skip_vars,
            z: reduction.z,
            tail_point: tail.eval_points,
        },
        final_claim: tail.final_claim,
    })
}
