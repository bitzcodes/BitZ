//! Binary fields. Integer embedding uses parity; polynomial-bit construction is explicit.

use crate::*;
mod embedding;
pub mod gf8;
mod operators;
mod polynomial;
mod prepared;
mod projection;
pub use prepared::{Gf128PreparedAcc, PreparedGf128Mul};
pub use projection::PreparedGf128Projection;
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(crate) mod b127_aarch64;
pub use embedding::AES_EMBEDDING_TABLE;
pub use gf8::Gf8;

/// AES-subfield embedding with fixed public basis reads.
#[inline]
pub fn embed_gf8(value: Gf8) -> Gf128 {
    Gf128Ops.embed(&value)
}
pub use polynomial::{F2Poly, F2PolyOps, F2PolyProduct};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct F2(Bit);
impl F2 {
    pub const ZERO: Self = Self(Bit::ZERO);
    pub const ONE: Self = Self(Bit::ONE);
    pub const fn from_bit(value: Bit) -> Self {
        Self(value)
    }
    pub const fn bit(self) -> Bit {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct B127(Uint<2>);
mod b127_sumcheck;
impl B127 {
    pub const fn as_integer(&self) -> &Uint<2> {
        &self.0
    }
    pub const fn from_polynomial_bits(bits: u128) -> Self {
        Self::from_polynomial_words([bits as u64, (bits >> 64) as u64])
    }
    pub const fn mul_x(self) -> Self {
        let w = self.as_words();
        let hi = (w[1] << 1) | (w[0] >> 63);
        Self::from_polynomial_words([(w[0] << 1) ^ ((hi >> 63) * 3), hi & 0x7fff_ffff_ffff_ffff])
    }
    pub fn reduce_wide(words: [u64; 4]) -> Self {
        reduce_b127(words)
    }
    pub fn try_from_words(words: [u64; 2]) -> Option<Self> {
        let v = Self::from_canonical_ct(words);
        v.validity().declassify().then_some(*v.value())
    }

    pub fn reduce_polynomial_words(words: [u64; 4]) -> Self {
        reduce_b127(words)
    }
    /// Accepted measurement alternative; the default remains schoolbook.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn mul_karatsuba(self, rhs: Self) -> Self {
        Self::from_canonical_words(b127_aarch64::mul_words_kara(
            self.as_words(),
            rhs.as_words(),
        ))
    }
    /// Accepted measurement alternative with a PMULL-based reduction.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn mul_pfold(self, rhs: Self) -> Self {
        Self::from_canonical_words(b127_aarch64::mul_words_pfold(
            self.as_words(),
            rhs.as_words(),
        ))
    }

    pub const ZERO: Self = Self(Uint::ZERO);
    pub const ONE: Self = Self(Uint::ONE);
    pub const fn from_polynomial_words(words: [u64; 2]) -> Self {
        let carry = words[1] >> 63;
        Self(Uint::from_words([
            words[0] ^ (carry * 3),
            words[1] & 0x7fff_ffff_ffff_ffff,
        ]))
    }
    /// Trusted canonical polynomial representation (degree below 127).
    #[inline]
    pub const fn from_canonical_words(words: [u64; 2]) -> Self {
        debug_assert!(words[1] >> 63 == 0, "noncanonical B127 element");
        Self(Uint::from_words(words))
    }
    pub fn from_canonical_ct(words: [u64; 2]) -> CtValue<Self> {
        let valid = !CtMask::from_lsb(words[1] >> 63);
        CtValue::new(
            Self(Uint::ct_select(
                &Uint::ZERO,
                &Uint::from_words(words),
                valid,
            )),
            valid,
        )
    }
    pub const fn as_words(&self) -> &[u64; 2] {
        self.0.as_words()
    }
    fn as_gf128(&self) -> Gf128 {
        Gf128::new(self.0.0[0], self.0.0[1])
    }
    pub fn square(self) -> Self {
        self.square_n(1)
    }
    pub fn square_n(self, n: usize) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            Self(Uint::from_words(b127_aarch64::square_n(
                *self.as_words(),
                n,
            )))
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            let mut value = self;
            for _ in 0..n {
                value = reduce_b127(Wide256::square(value.as_gf128()).words());
            }
            value
        }
    }
}

/// General polynomial reduction, adapted from BitZ's B127 implementation.
fn reduce_b127(prod: [u64; 4]) -> B127 {
    let h0 = (prod[1] >> 63) | (prod[2] << 1);
    let h1 = (prod[2] >> 63) | (prod[3] << 1);
    let h2 = prod[3] >> 63;
    let g0 = h0 ^ (h0 << 1);
    let g1 = h1 ^ (h1 << 1) ^ (h0 >> 63);
    let g2 = h2 ^ (h2 << 1) ^ (h1 >> 63);
    let t0 = prod[0] ^ g0;
    let t1 = (prod[1] & 0x7fff_ffff_ffff_ffff) ^ g1;
    let u = (t1 >> 63) | (g2 << 1);
    B127(Uint::from_words([
        t0 ^ u ^ (u << 1),
        t1 & 0x7fff_ffff_ffff_ffff,
    ]))
}

impl CtEq for Gf128 {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        ((self.lo ^ rhs.lo) | (self.hi ^ rhs.hi)).ct_is_zero()
    }
    fn ct_is_zero(&self) -> CtMask {
        (self.lo | self.hi).ct_is_zero()
    }
}
impl CtSelect for Gf128 {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self::new(
            u64::ct_select(&a.lo, &b.lo, mask),
            u64::ct_select(&a.hi, &b.hi, mask),
        )
    }
}
impl CtEq for Gf8 {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        (self.0 as u64).ct_eq(&(rhs.0 as u64))
    }
    fn ct_is_zero(&self) -> CtMask {
        (self.0 as u64).ct_is_zero()
    }
}
impl CtSelect for Gf8 {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self(u64::ct_select(&(a.0 as u64), &(b.0 as u64), mask) as u8)
    }
}
impl CtEq for F2 {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        self.0.ct_eq(&rhs.0)
    }
    fn ct_is_zero(&self) -> CtMask {
        self.0.ct_is_zero()
    }
}
impl CtSelect for F2 {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self(Bit::ct_select(&a.0, &b.0, mask))
    }
}
impl CtEq for B127 {
    fn ct_eq(&self, rhs: &Self) -> CtMask {
        self.0.ct_eq(&rhs.0)
    }
    fn ct_is_zero(&self) -> CtMask {
        self.0.ct_is_zero()
    }
}
impl CtSelect for B127 {
    fn ct_select(a: &Self, b: &Self, mask: CtMask) -> Self {
        Self(Uint::ct_select(&a.0, &b.0, mask))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct F2Ops;
#[derive(Clone, Copy, Debug, Default)]
pub struct Gf8Ops;
#[derive(Clone, Copy, Debug, Default)]
pub struct Gf128Ops;
#[derive(Clone, Copy, Debug, Default)]
pub struct B127Ops;

impl RingOps for F2Ops {
    type Elem = F2;
    fn zero(&self) -> F2 {
        F2::ZERO
    }
    fn one(&self) -> F2 {
        F2::ONE
    }
    fn add(&self, a: &F2, b: &F2) -> F2 {
        F2(Bit::from_lsb(a.0.as_u64() ^ b.0.as_u64()))
    }
    fn sub(&self, a: &F2, b: &F2) -> F2 {
        self.add(a, b)
    }
    fn neg(&self, a: &F2) -> F2 {
        *a
    }
    fn mul(&self, a: &F2, b: &F2) -> F2 {
        F2(Bit::from_lsb(a.0.as_u64() & b.0.as_u64()))
    }
}
impl FieldOps for F2Ops {
    fn inverse_ct(&self, a: &F2) -> CtValue<F2> {
        CtValue::new(*a, !a.ct_is_zero())
    }
}
impl RingOps for Gf8Ops {
    type Elem = Gf8;
    fn zero(&self) -> Gf8 {
        Gf8::ZERO
    }
    fn one(&self) -> Gf8 {
        Gf8::ONE
    }
    fn add(&self, a: &Gf8, b: &Gf8) -> Gf8 {
        *a + *b
    }
    fn sub(&self, a: &Gf8, b: &Gf8) -> Gf8 {
        *a + *b
    }
    fn neg(&self, a: &Gf8) -> Gf8 {
        *a
    }
    fn mul(&self, a: &Gf8, b: &Gf8) -> Gf8 {
        *a * *b
    }
}
impl FieldOps for Gf8Ops {
    fn inverse_ct(&self, a: &Gf8) -> CtValue<Gf8> {
        CtValue::new(a.inverse_or_zero(), !a.ct_is_zero())
    }
}
impl RingOps for Gf128Ops {
    type Elem = Gf128;
    fn zero(&self) -> Gf128 {
        Gf128::new(0, 0)
    }
    fn one(&self) -> Gf128 {
        Gf128::new(1, 0)
    }
    fn add(&self, a: &Gf128, b: &Gf128) -> Gf128 {
        *a + *b
    }
    fn sub(&self, a: &Gf128, b: &Gf128) -> Gf128 {
        *a + *b
    }
    fn neg(&self, a: &Gf128) -> Gf128 {
        *a
    }
    fn mul(&self, a: &Gf128, b: &Gf128) -> Gf128 {
        *a * *b
    }
    fn square(&self, a: &Gf128) -> Gf128 {
        a.square()
    }
}
impl FieldOps for Gf128Ops {
    fn inverse_ct(&self, a: &Gf128) -> CtValue<Gf128> {
        // The existing Itoh–Tsujii chain, without its early zero branch.
        let b1 = *a;
        let b2 = b1.square_n(1) * b1;
        let b3 = b2.square_n(1) * b1;
        let b6 = b3.square_n(3) * b3;
        let b12 = b6.square_n(6) * b6;
        let b24 = b12.square_n(12) * b12;
        let b48 = b24.square_n(24) * b24;
        let b96 = b48.square_n(48) * b48;
        let b120 = b96.square_n(24) * b24;
        let b126 = b120.square_n(6) * b6;
        let b127 = b126.square_n(1) * b1;
        CtValue::new(b127.square(), !a.ct_is_zero())
    }
}
impl RingOps for B127Ops {
    type Elem = B127;
    fn zero(&self) -> B127 {
        B127::ZERO
    }
    fn one(&self) -> B127 {
        B127::ONE
    }
    fn add(&self, a: &B127, b: &B127) -> B127 {
        B127(a.0 ^ b.0)
    }
    fn sub(&self, a: &B127, b: &B127) -> B127 {
        self.add(a, b)
    }
    fn neg(&self, a: &B127) -> B127 {
        *a
    }
    fn mul(&self, a: &B127, b: &B127) -> B127 {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            B127(Uint::from_words(b127_aarch64::mul(
                *a.as_words(),
                *b.as_words(),
            )))
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            reduce_b127(Wide256::mul(a.as_gf128(), b.as_gf128()).words())
        }
    }
    fn square(&self, a: &B127) -> B127 {
        a.square()
    }
}
impl FieldOps for B127Ops {
    fn inverse_ct(&self, a: &B127) -> CtValue<B127> {
        let t2 = self.mul(&a.square_n(1), a);
        let t3 = self.mul(&t2.square_n(1), a);
        let t6 = self.mul(&t3.square_n(3), &t3);
        let t12 = self.mul(&t6.square_n(6), &t6);
        let t24 = self.mul(&t12.square_n(12), &t12);
        let t48 = self.mul(&t24.square_n(24), &t24);
        let t96 = self.mul(&t48.square_n(48), &t48);
        let t120 = self.mul(&t96.square_n(24), &t24);
        let t126 = self.mul(&t120.square_n(6), &t6);
        CtValue::new(t126.square(), !a.ct_is_zero())
    }
}
impl BatchFieldOps for F2Ops {}
impl BatchFieldOps for Gf8Ops {}
impl BatchFieldOps for Gf128Ops {}
impl BatchFieldOps for B127Ops {}

/// AES embedding using eight public basis entries, never a secret-indexed table.
impl FieldEmbedding<Gf8> for Gf128Ops {
    fn embed(&self, value: &Gf8) -> Gf128 {
        let mut out = self.zero();
        for (i, basis) in embedding::PHI8_BASIS.iter().enumerate() {
            let mask = CtMask::from_lsb((value.0 as u64) >> i).word();
            out.lo ^= basis.lo & mask;
            out.hi ^= basis.hi & mask;
        }
        out
    }
}
macro_rules! binary_integers {
    ($ctx:ty) => {
        impl IntegerEmbedding<Bit> for $ctx {
            fn from_integer(&self, value: &Bit) -> Self::Elem {
                Self::Elem::ct_select(&self.zero(), &self.one(), value.mask())
            }
        }
        impl IntegerEmbedding<u32> for $ctx {
            fn from_integer(&self, value: &u32) -> Self::Elem {
                self.from_integer(&Bit::from_lsb(*value as u64))
            }
        }
        impl IntegerEmbedding<u64> for $ctx {
            fn from_integer(&self, value: &u64) -> Self::Elem {
                self.from_integer(&Bit::from_lsb(*value))
            }
        }
        impl IntegerEmbedding<u128> for $ctx {
            fn from_integer(&self, value: &u128) -> Self::Elem {
                self.from_integer(&Bit::from_lsb(*value as u64))
            }
        }
        impl<const N: usize> IntegerEmbedding<Uint<N>> for $ctx {
            fn from_integer(&self, value: &Uint<N>) -> Self::Elem {
                self.from_integer(&Bit::from_lsb(value.as_words()[0]))
            }
        }
        impl<const N: usize> IntegerEmbedding<Z<N>> for $ctx {
            fn from_integer(&self, value: &Z<N>) -> Self::Elem {
                self.from_integer(&Bit::from_lsb(value.as_words()[0]))
            }
        }
        impl FieldEmbedding<F2> for $ctx {
            fn embed(&self, value: &F2) -> Self::Elem {
                self.from_integer(&value.0)
            }
        }
    };
}
binary_integers!(F2Ops);
binary_integers!(Gf8Ops);
binary_integers!(Gf128Ops);
binary_integers!(B127Ops);

#[derive(Clone, Copy)]
pub struct Gf128Product(Wide256);

impl Gf128Product {
    #[inline]
    pub fn from_element(value: Gf128) -> Self {
        Self(Wide256::of(value))
    }

    #[inline]
    pub fn zero() -> Self {
        Self(Wide256::zero())
    }

    /// An unreduced polynomial in the GHASH product domain, low word first.
    #[inline]
    pub fn from_polynomial_words(words: [u64; 4]) -> Self {
        Self(Wide256::from_words(words))
    }

    #[inline]
    pub fn to_polynomial_words(self) -> [u64; 4] {
        self.0.words()
    }

    #[inline]
    pub fn reduce(self) -> Gf128 {
        self.0.reduce()
    }
}

impl core::ops::BitXorAssign for Gf128Product {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl core::ops::BitXor for Gf128Product {
    type Output = Self;
    #[inline]
    fn bitxor(mut self, rhs: Self) -> Self {
        self ^= rhs;
        self
    }
}

impl core::fmt::Debug for Gf128Product {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Gf128Product")
            .field(&self.to_polynomial_words())
            .finish()
    }
}

impl PartialEq for Gf128Product {
    fn eq(&self, rhs: &Self) -> bool {
        self.to_polynomial_words() == rhs.to_polynomial_words()
    }
}
impl Eq for Gf128Product {}
#[derive(Clone, Copy)]
pub struct B127Product(Wide256);
impl B127Product {
    #[inline]
    pub fn zero() -> Self {
        Self(Wide256::zero())
    }
    /// A canonical element is also in the product domain (degree <=252).
    #[inline]
    pub fn from_element(value: B127) -> Self {
        Self(Wide256::of(value.as_gf128()))
    }
    /// Encoded product-domain polynomial; the owner guarantees degree <=252.
    #[inline]
    pub fn from_product_words(words: [u64; 4]) -> Self {
        debug_assert_eq!(words[3] >> 61, 0, "B127 product exceeds degree 252");
        Self(Wide256::from_words(words))
    }
    #[inline]
    pub fn to_product_words(self) -> [u64; 4] {
        self.0.words()
    }
    #[inline]
    pub fn reduce(self) -> B127 {
        B127Ops.reduce(self)
    }
}
impl core::ops::BitXorAssign for B127Product {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl core::ops::BitXor for B127Product {
    type Output = Self;
    #[inline]
    fn bitxor(mut self, rhs: Self) -> Self {
        self ^= rhs;
        self
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Gf8Product(u16);
#[derive(Clone, Copy, Debug)]
pub struct XorAccumulator<P> {
    payload: P,
}
impl<P> XorAccumulator<P> {
    /// Extract the unreduced polynomial product for explicit output conversion.
    pub fn into_payload(self) -> P {
        self.payload
    }
}
trait XorPayload: Copy {
    fn zero() -> Self;
    fn xor_assign(&mut self, rhs: Self);
}
impl XorPayload for Gf128Product {
    fn zero() -> Self {
        Self(Wide256::zero())
    }
    fn xor_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl XorPayload for B127Product {
    fn zero() -> Self {
        Self(Wide256::zero())
    }
    fn xor_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl XorPayload for Gf8Product {
    fn zero() -> Self {
        Self(0)
    }
    fn xor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl XorPayload for F2 {
    fn zero() -> Self {
        Self::ZERO
    }
    fn xor_assign(&mut self, rhs: Self) {
        *self = F2Ops.add(self, &rhs);
    }
}
impl<P> From<P> for XorAccumulator<P> {
    fn from(payload: P) -> Self {
        Self { payload }
    }
}
impl<P: XorPayload> MergeAccumulator for XorAccumulator<P> {
    fn zero() -> Self {
        Self { payload: P::zero() }
    }
    fn merge_assign(&mut self, rhs: &Self) {
        self.payload.xor_assign(rhs.payload);
    }
}
impl WideMul<Gf128> for Gf128Ops {
    type Product = Gf128Product;
    fn mul_wide(&self, a: &Gf128, b: &Gf128) -> Gf128Product {
        Gf128Product(Wide256::mul(*a, *b))
    }
}
impl WideMul<Gf8> for Gf8Ops {
    type Product = Gf8Product;
    fn mul_wide(&self, a: &Gf8, b: &Gf8) -> Gf8Product {
        Gf8Product(gf8::clmul8(a.0, b.0))
    }
}
impl WideMul<B127> for B127Ops {
    type Product = B127Product;
    fn mul_wide(&self, a: &B127, b: &B127) -> B127Product {
        B127Product(Wide256::mul(a.as_gf128(), b.as_gf128()))
    }
}
impl WideMul<F2> for F2Ops {
    type Product = F2;
    fn mul_wide(&self, a: &F2, b: &F2) -> F2 {
        self.mul(a, b)
    }
}
impl Reduce<Gf128Product> for Gf128Ops {
    type Output = Gf128;
    fn reduce(&self, value: Gf128Product) -> Gf128 {
        value.0.reduce()
    }
}
impl Reduce<Gf8Product> for Gf8Ops {
    type Output = Gf8;
    fn reduce(&self, value: Gf8Product) -> Gf8 {
        Gf8(gf8::gf8_reduce(value.0))
    }
}
impl Reduce<B127Product> for B127Ops {
    type Output = B127;
    fn reduce(&self, value: B127Product) -> B127 {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            B127(Uint::from_words(value.0.reduce_b127_product()))
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            reduce_b127(value.0.words())
        }
    }
}
impl Reduce<F2> for F2Ops {
    type Output = F2;
    fn reduce(&self, value: F2) -> F2 {
        value
    }
}
macro_rules! binary_batch {
    ($ctx:ty,$value:ty,$product:ty) => {
        impl BatchMulAcc<$value> for $ctx {
            type Accumulator = XorAccumulator<$product>;
            #[inline(always)]
            fn mul_acc(&self, acc: &mut Self::Accumulator, lhs: &$value, rhs: &$value) {
                acc.payload.xor_assign(self.mul_wide(lhs, rhs));
            }
            fn batch_mul_acc(&self, lhs: &[$value], rhs: &[$value]) -> Self::Accumulator {
                assert_eq!(lhs.len(), rhs.len());
                self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
            }
            fn batch_mul_acc_map(
                &self,
                len: usize,
                mut term: impl FnMut(usize) -> ($value, $value),
            ) -> Self::Accumulator {
                let mut acc = XorAccumulator::<$product>::zero();
                for i in 0..len {
                    let (a, b) = term(i);
                    acc.payload.xor_assign(self.mul_wide(&a, &b));
                }
                acc
            }
        }
        impl Reduce<XorAccumulator<$product>> for $ctx {
            type Output = $value;
            fn reduce(&self, value: XorAccumulator<$product>) -> $value {
                self.reduce(value.payload)
            }
        }
        impl crate::batch::RoundArithmetic for $ctx {
            type Acc = XorAccumulator<$product>;
            fn zero_acc(&self) -> Self::Acc {
                Self::Acc::zero()
            }
            fn mac(&self, acc: &mut Self::Acc, a: &$value, b: &$value) {
                acc.payload.xor_assign(self.mul_wide(a, b));
            }
            fn finish(&self, acc: Self::Acc) -> $value {
                self.reduce(acc)
            }
        }
    };
}
binary_batch!(F2Ops, F2, F2);
binary_batch!(Gf8Ops, Gf8, Gf8Product);
binary_batch!(Gf128Ops, Gf128, Gf128Product);
binary_batch!(B127Ops, B127, B127Product);
crate::batch::implement_sumcheck!([] F2Ops);
crate::batch::implement_sumcheck!([] Gf8Ops);

impl WideMul<Gf128, Gf8> for Gf128Ops {
    type Product = Gf128Product;
    fn mul_wide(&self, a: &Gf128, b: &Gf8) -> Gf128Product {
        self.mul_wide(a, &self.embed(b))
    }
}
impl BatchMulAcc<Gf128, Gf8> for Gf128Ops {
    type Accumulator = XorAccumulator<Gf128Product>;
    fn batch_mul_acc(&self, lhs: &[Gf128], rhs: &[Gf8]) -> Self::Accumulator {
        assert_eq!(lhs.len(), rhs.len());
        self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
    }
    fn batch_mul_acc_map(
        &self,
        len: usize,
        mut term: impl FnMut(usize) -> (Gf128, Gf8),
    ) -> Self::Accumulator {
        let mut out = XorAccumulator::<Gf128Product>::zero();
        for i in 0..len {
            let (a, b) = term(i);
            out.payload.xor_assign(self.mul_wide(&a, &b));
        }
        out
    }
}
impl WideMul<Gf8, Gf128> for Gf128Ops {
    type Product = Gf128Product;
    fn mul_wide(&self, a: &Gf8, b: &Gf128) -> Self::Product {
        self.mul_wide(b, a)
    }
}
impl BatchMulAcc<Gf8, Gf128> for Gf128Ops {
    type Accumulator = XorAccumulator<Gf128Product>;
    fn batch_mul_acc(&self, lhs: &[Gf8], rhs: &[Gf128]) -> Self::Accumulator {
        assert_eq!(lhs.len(), rhs.len());
        self.batch_mul_acc_map(lhs.len(), |i| (lhs[i], rhs[i]))
    }
    fn batch_mul_acc_map(
        &self,
        len: usize,
        mut term: impl FnMut(usize) -> (Gf8, Gf128),
    ) -> Self::Accumulator {
        <Self as BatchMulAcc<Gf128, Gf8>>::batch_mul_acc_map(self, len, |i| {
            let (a, b) = term(i);
            (b, a)
        })
    }
}

impl CanonicalCodec<Gf128> for Gf128Ops {
    fn encoded_len(&self) -> usize {
        16
    }
    fn encode_into(&self, value: &Gf128, out: &mut [u8]) {
        assert_eq!(out.len(), 16);
        out.copy_from_slice(&value.to_bytes());
    }
    fn decode_ct(&self, input: &[u8]) -> Result<CtValue<Gf128>, DecodeError> {
        let words = crate::encoding::decode_words::<2>(input)?;
        Ok(CtValue::new(
            Gf128::new(words.0[0], words.0[1]),
            CtMask::TRUE,
        ))
    }
}
impl CanonicalCodec<Gf8> for Gf8Ops {
    fn encoded_len(&self) -> usize {
        1
    }
    fn encode_into(&self, value: &Gf8, out: &mut [u8]) {
        assert_eq!(out.len(), 1);
        out[0] = value.0;
    }
    fn decode_ct(&self, input: &[u8]) -> Result<CtValue<Gf8>, DecodeError> {
        if input.len() != 1 {
            return Err(DecodeError::Length {
                expected: 1,
                actual: input.len(),
            });
        }
        Ok(CtValue::new(Gf8(input[0]), CtMask::TRUE))
    }
}
impl CanonicalCodec<F2> for F2Ops {
    fn encoded_len(&self) -> usize {
        1
    }
    fn encode_into(&self, value: &F2, out: &mut [u8]) {
        assert_eq!(out.len(), 1);
        out[0] = value.0.as_u64() as u8;
    }
    fn decode_ct(&self, input: &[u8]) -> Result<CtValue<F2>, DecodeError> {
        if input.len() != 1 {
            return Err(DecodeError::Length {
                expected: 1,
                actual: input.len(),
            });
        }
        let valid = (input[0] as u64).ct_lt(&2);
        Ok(CtValue::new(
            F2::ct_select(&F2::ZERO, &F2(Bit::from_lsb(input[0] as u64)), valid),
            valid,
        ))
    }
}
impl CanonicalCodec<B127> for B127Ops {
    fn encoded_len(&self) -> usize {
        16
    }
    fn encode_into(&self, value: &B127, out: &mut [u8]) {
        crate::encoding::encode_words(&value.0, out);
    }
    fn decode_ct(&self, input: &[u8]) -> Result<CtValue<B127>, DecodeError> {
        Ok(B127::from_canonical_ct(
            *crate::encoding::decode_words::<2>(input)?.as_words(),
        ))
    }
}

macro_rules! fixed_mul {
    ($name:ident,$field:ident,$elem:ty) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            scalar: $elem,
        }
        impl $name {
            pub fn new(scalar: $elem) -> Self {
                Self { scalar }
            }
            pub fn mul(&self, value: &$elem) -> $elem {
                $field.mul(value, &self.scalar)
            }
            pub fn mul_into(&self, input: &[$elem], out: &mut [$elem]) {
                assert_eq!(input.len(), out.len());
                for (value, dst) in input.iter().zip(out) {
                    *dst = self.mul(value);
                }
            }
        }
    };
}
fixed_mul!(PreparedB127Mul, B127Ops, B127);

macro_rules! binary_sampling {
    ($ctx:ty,$bytes:expr,$make:expr) => {
        impl FieldSampling for $ctx {
            fn sample_public(
                &self,
                source: &mut impl PublicRandomSource,
                max_draws: usize,
            ) -> Result<Self::Elem, SamplingError> {
                if max_draws == 0 {
                    return Err(SamplingError::Exhausted);
                }
                let mut bytes = [0u8; $bytes];
                source.fill_bytes(&mut bytes);
                Ok(($make)(bytes))
            }
            fn sample_nonzero_public(
                &self,
                source: &mut impl PublicRandomSource,
                max_draws: usize,
            ) -> Result<Self::Elem, SamplingError> {
                for _ in 0..max_draws {
                    let value = self.sample_public(source, 1)?;
                    if !value.ct_is_zero().declassify() {
                        return Ok(value);
                    }
                }
                Err(SamplingError::Exhausted)
            }
        }
    };
}
binary_sampling!(F2Ops, 1, |b: [u8; 1]| F2::from_bit(Bit::from_lsb(
    b[0] as u64
)));
binary_sampling!(Gf8Ops, 1, |b: [u8; 1]| Gf8::new(b[0]));
binary_sampling!(Gf128Ops, 16, |b: [u8; 16]| Gf128::from_bytes(b));
binary_sampling!(B127Ops, 16, |b: [u8; 16]| {
    let mut w = *crate::encoding::decode_words::<2>(&b).unwrap().as_words();
    w[1] &= u64::MAX >> 1;
    B127(Uint::from_words(w))
});

impl WideMul<Gf128, Bit> for Gf128Ops {
    type Product = Gf128Product;
    fn mul_wide(&self, a: &Gf128, b: &Bit) -> Gf128Product {
        Gf128Product(Wide256::of(Gf128::ct_select(&self.zero(), a, b.mask())))
    }
}
macro_rules! gf128_integer_mac {
    ([$($generic:tt)*] $ty:ty,$bit:expr)=>{
        impl<$($generic)*> BatchMulAcc<Gf128,$ty> for Gf128Ops {
            type Accumulator=XorAccumulator<Gf128Product>;
            fn batch_mul_acc(&self,lhs:&[Gf128],rhs:&[$ty])->Self::Accumulator{assert_eq!(lhs.len(),rhs.len());self.batch_mul_acc_map(lhs.len(),|i|(lhs[i],rhs[i]))}
            fn batch_mul_acc_map(&self,len:usize,mut term:impl FnMut(usize)->(Gf128,$ty))->Self::Accumulator{
                let mut out=Self::Accumulator::zero();for i in 0..len{let(a,b)=term(i);out.payload.xor_assign(self.mul_wide(&a,&($bit)(b)));}out
            }
        }
    };
}
gf128_integer_mac!([] Bit,|b:Bit|b);
gf128_integer_mac!([] u32,|b:u32|Bit::from_lsb(b as u64));
gf128_integer_mac!([] u64,|b:u64|Bit::from_lsb(b));
gf128_integer_mac!([] u128,|b:u128|Bit::from_lsb(b as u64));
gf128_integer_mac!([const N:usize] Uint<N>,|b:Uint<N>|Bit::from_lsb(b.as_words()[0]));
gf128_integer_mac!([const N:usize] Z<N>,|b:Z<N>|Bit::from_lsb(b.as_words()[0]));
