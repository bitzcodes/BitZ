use crate::poly::coefficient::PolynomialField;

/// Trait for preparing a projection function to a field element from a current
/// type.
pub trait ProjectableToField<F: PolynomialField> {
    /// Prepare a projection function that will project the current type
    /// to a prime field using the given sampled value.
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + Send + Sync + 'static;
}
