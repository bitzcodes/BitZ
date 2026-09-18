//! Assignment and evaluated matrix-product storage.

use super::sumcheck::R1csProductMles;
use crate::poly::mle::DenseMultilinearExtension;

/// An assignment and its already-evaluated `A`, `B`, and `C` products.
#[derive(Clone, Debug)]
pub struct EvaluatedSpartanAssignment<F> {
    assignment: DenseMultilinearExtension<F>,
    products: R1csProductMles<F>,
}

impl<F> EvaluatedSpartanAssignment<F> {
    /// Creates one internally consistent Spartan witness bundle.
    pub const fn new(
        assignment: DenseMultilinearExtension<F>,
        products: R1csProductMles<F>,
    ) -> Self {
        Self {
            assignment,
            products,
        }
    }

    /// Assignment MLE consumed by Spartan.
    pub const fn assignment(&self) -> &DenseMultilinearExtension<F> {
        &self.assignment
    }

    /// Evaluated `A`, `B`, and `C` product MLEs.
    pub const fn products(&self) -> &R1csProductMles<F> {
        &self.products
    }

    /// Moves out the assignment and products together.
    pub fn into_parts(self) -> (DenseMultilinearExtension<F>, R1csProductMles<F>) {
        (self.assignment, self.products)
    }
}
