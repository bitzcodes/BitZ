use crate::poly::coefficient::PolynomialField;

/// A trait for fields that allow to perform operations
/// on inner Montgomery representations of field elements.
pub trait InnerTransparentField: PolynomialField {
    /// Add inner Montgomery representations using a config.
    fn add_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner;

    /// Subtract inner Montgomery representations using a config.
    fn sub_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner;

    /// Multiply a field element by an inner Montgomery representation.
    fn mul_assign_by_inner(&mut self, rhs: &Self::Inner);

    /// Multiply `self` by `node2`, the degree-≤2 sumcheck's node-2
    /// coordinate `F::from(2)`. The default is a full field multiply, so
    /// any field is correct out of the box. Characteristic-2 fields — where
    /// `from(2) = X` (the bit-pattern convention) — override this with the
    /// cheap multiply-by-`X` (a left shift plus a conditional reduction),
    /// the hot-path lever for the `GF(2^128)` grand-product round message,
    /// which evaluates each degree-2 multiplicand at node `X`. The result
    /// is the *identical* field element, so round messages stay
    /// bit-for-bit unchanged (the eq-factored sumcheck's byte-identity to
    /// the generic prover is preserved).
    #[inline]
    #[allow(clippy::arithmetic_side_effects)] // field multiply: modular, cannot overflow
    fn mul_by_node2(&self, node2: &Self) -> Self {
        self.clone() * node2
    }
}
