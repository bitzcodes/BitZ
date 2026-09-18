//! Consumer adapters for shared, explicitly sized binary polynomials.
use crate::{poly::univariate::F2AddAssign, utils::from_ref::FromRef};
use field::F2Poly;
impl<const B: usize, const W: usize> F2AddAssign for F2Poly<B, W> {
    fn f2_add_assign(&mut self, rhs: &Self) {
        *self = self.xor(rhs);
    }
}
impl<const B: usize, const W: usize> FromRef<F2Poly<B, W>> for F2Poly<B, W> {
    fn from_ref(value: &Self) -> Self {
        *value
    }
}
