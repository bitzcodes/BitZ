use field::{Bit, Z};
use num_traits::{CheckedMul, ConstZero};

pub trait MulByScalar<Rhs, Out = Self>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: Rhs) -> Option<Out>;
}

macro_rules! impl_mul_by_scalar_for_primitives {
    ($($t:ty),*) => {
        $(
            impl MulByScalar<&$t> for $t {
                #[allow(clippy::arithmetic_side_effects)] // By design
                fn mul_by_scalar<const CHECK: bool>(&self, rhs: &$t) -> Option<Self> {
                    if CHECK {
                        self.checked_mul(rhs)
                    } else {
                        Some(self * rhs)
                    }
                }
            }
        )*
    };
}

impl_mul_by_scalar_for_primitives!(i8, i16, i32, i64, i128);

impl<const L: usize, const N: usize> MulByScalar<&Z<N>> for Z<L> {
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Z<N>) -> Option<Self> {
        use field::WideMul;
        let result = field::IntegerOps
            .mul_wide(self, rhs)
            .checked_resize_ct::<L>();
        (!CHECK || result.validity().declassify()).then_some(*result.value())
    }
}
macro_rules! signed_scalar {
    ($($ty:ty => $n:literal),*)=>{$(
        impl<const L:usize,const M:usize> MulByScalar<&$ty,Z<M>> for Z<L> {
            fn mul_by_scalar<const CHECK:bool>(&self,rhs:&$ty)->Option<Z<M>> {
                use field::WideMul;
                let rhs=Z::<$n>::from(*rhs);
                let result=field::IntegerOps.mul_wide(self,&rhs).checked_resize_ct::<M>();
                (!CHECK || result.validity().declassify()).then_some(*result.value())
            }
        }
    )*};
}
signed_scalar!(i8=>1,i16=>1,i32=>1,i64=>1,i128=>2);

impl<T> MulByScalar<&Bit> for T
where
    T: field::CtSelect + ConstZero,
{
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Bit) -> Option<Self> {
        Some(T::ct_select(&T::ZERO, self, rhs.mask()))
    }
}

impl MulByScalar<&i64, i128> for i32 {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // Not possible to overflow since we are widening the result to i128
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<i128> {
        Some(i128::from(*self) * i128::from(*rhs))
    }
}

impl MulByScalar<&i64> for i128 {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // By design
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<i128> {
        let rhs = i128::from(*rhs);
        if CHECK {
            self.checked_mul(&rhs)
        } else {
            Some(self * rhs)
        }
    }
}

impl MulByScalar<&i64, i128> for i64 {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)] // Not possible to overflow since we are widening the result to i128
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &i64) -> Option<i128> {
        Some(i128::from(*self) * i128::from(*rhs))
    }
}
