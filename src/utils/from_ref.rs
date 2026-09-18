use field::{Bit, Uint, Z};

//
// FromRef
//

/// This trait is essentially equivalent to `From<&T>`, other than it allows us
/// to implement it for external types that don't implement it out of the box,
/// most notably primitive types.
pub trait FromRef<T> {
    fn from_ref(value: &T) -> Self;
}

impl<T> FromRef<Bit> for T
where
    T: From<bool>,
{
    fn from_ref(value: &Bit) -> Self {
        T::from(bool::from(*value))
    }
}

macro_rules! impl_from_ref_for_primitive {
    ($dst:ty, [$($src:ty),+]) => {
        $(
            impl FromRef<$src> for $dst {
                fn from_ref(value: &$src) -> Self {
                    <$dst>::from(*value)
                }
            }
        )+
    };
}

impl_from_ref_for_primitive!(i128, [i128, i64, i32, i16, i8]);
impl_from_ref_for_primitive!(i64, [i64, i32, i16, i8]);
impl_from_ref_for_primitive!(i32, [i32, i16, i8]);
impl_from_ref_for_primitive!(i16, [i16, i8]);
impl_from_ref_for_primitive!(i8, [i8]);

macro_rules! impl_int_from_primitive_ref {
    ($($t:ty),+) => {
        $(
            impl<const LIMBS: usize> FromRef<$t> for Z<LIMBS> {
                #[inline(always)]
                fn from_ref(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )+
    };
}

impl_int_from_primitive_ref!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize, const LIMBS2: usize> FromRef<Z<LIMBS2>> for Z<LIMBS> {
    #[inline]
    fn from_ref(value: &Z<LIMBS2>) -> Self {
        {
            let resized = value.checked_resize_ct::<LIMBS>();
            assert!(
                resized.validity().declassify(),
                "Destination Z type is too small"
            );
            *resized.value()
        }
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> FromRef<Uint<LIMBS2>> for Uint<LIMBS> {
    #[inline]
    fn from_ref(value: &Uint<LIMBS2>) -> Self {
        {
            let resized = value.checked_resize_ct::<LIMBS>();
            assert!(
                resized.validity().declassify(),
                "Destination Uint type is too small"
            );
            *resized.value()
        }
    }
}
