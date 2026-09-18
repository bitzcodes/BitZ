//! Reusable preparation with caller-owned inputs and outputs.

use crate::*;

#[derive(Clone, Debug)]
pub struct PreparedDivisor<const D: usize> {
    divisor: Uint<D>,
    reduction: crate::modular::reduction::Barrett<D>,
}
impl<const D: usize> PreparedDivisor<D> {
    pub fn new(divisor: Uint<D>) -> Result<Self, ContextError> {
        if divisor.ct_is_zero().declassify() {
            return Err(ContextError::ZeroDivisor);
        }
        Ok(Self {
            reduction: crate::modular::reduction::Barrett::new(&divisor),
            divisor,
        })
    }
    pub fn div_rem_ct<const N: usize>(&self, dividend: &Uint<N>) -> (Uint<N>, Uint<D>) {
        let mut quotient = Uint::ZERO;
        let remainder = self
            .reduction
            .divide_into(dividend, &self.divisor, |i, w| quotient.0[i] = w);
        (quotient, remainder)
    }
    pub fn div_rem_product_ct<const A: usize, const B: usize>(
        &self,
        dividend: &UintProduct<A, B>,
    ) -> (UintProduct<A, B>, Uint<D>) {
        let mut quotient = UintProduct::ZERO;
        let remainder = self
            .reduction
            .divide_into(dividend, &self.divisor, |i, w| quotient.set_word(i, w));
        (quotient, remainder)
    }
}

#[derive(Clone, Debug)]
pub struct PreparedOddInverse<const L: usize> {
    #[cfg(target_pointer_width = "64")]
    modulus: crypto_bigint::Odd<crypto_bigint::Uint<L>>,
    #[cfg(not(target_pointer_width = "64"))]
    ring: ModRingCtx<L>,
}
impl<const L: usize> PreparedOddInverse<L> {
    pub fn new(modulus: Uint<L>) -> Result<Self, ContextError> {
        if modulus.as_words()[0] & 1 == 0 {
            return Err(ContextError::EvenModulus);
        }
        let ring = ModRingCtx::new(modulus)?;
        Ok(Self {
            #[cfg(target_pointer_width = "64")]
            modulus: crypto_bigint::Odd::new(crypto_bigint::Uint::from_words(
                *ring.modulus().as_words(),
            ))
            .expect("prepared modulus is odd"),
            #[cfg(not(target_pointer_width = "64"))]
            ring,
        })
    }
    /// Fixed-schedule inversion, with a zero output for nonunits. On 64-bit
    /// targets the backend batches divsteps instead of updating every full-width
    /// coefficient on each binary-GCD step. Both backends use the declared width.
    #[cfg(target_pointer_width = "64")]
    pub fn inverse_ct(&self, value: &Uint<L>) -> CtValue<Uint<L>> {
        let inverse =
            crypto_bigint::Uint::from_words(*value.as_words()).invert_odd_mod(&self.modulus);
        let valid = CtMask::from_lsb(inverse.is_some().to_u8() as u64);
        let value = inverse.unwrap_or(crypto_bigint::Uint::ZERO);
        CtValue::new(Uint::from_words(value.to_words()), valid)
    }

    #[cfg(not(target_pointer_width = "64"))]
    pub fn inverse_ct(&self, value: &Uint<L>) -> CtValue<Uint<L>> {
        let modulus = self.ring.modulus();
        let (mut u, mut v) = (
            *modulus,
            self.ring.to_integer(&self.ring.from_integer(value)),
        );
        let (mut r, mut s) = (Uint::ZERO, Uint::ONE);
        // Maintain u = value*r and v = value*s (mod m). Every active step
        // halves a positive operand, or subtracts two odds and halves the larger.
        // Thus bit_length(u)+bit_length(v) drops at least once per active step.
        for _ in 0..128 * L {
            let active = !u.ct_is_zero() & !v.ct_is_zero();
            let u_even = !u.bit(0).mask();
            let v_even = !v.bit(0).mask();
            let u_greater = v.ct_lt(&u);
            let halve_u = active & u_even;
            let halve_v = active & !u_even & v_even;
            let subtract_u = active & !u_even & !v_even & u_greater;
            let subtract_v = active & !u_even & !v_even & !u_greater;
            let difference_u = u.wrapping_sub(&v).shr(1);
            let difference_v = v.wrapping_sub(&u).shr(1);
            let difference_r = self.ring.sub(&Residue(r), &Residue(s)).0;
            let difference_s = self.ring.sub(&Residue(s), &Residue(r)).0;
            u = Uint::ct_select(&u, &u.shr(1), halve_u);
            u = Uint::ct_select(&u, &difference_u, subtract_u);
            v = Uint::ct_select(&v, &v.shr(1), halve_v);
            v = Uint::ct_select(&v, &difference_v, subtract_v);
            r = Uint::ct_select(&r, &self.halve_residue(&r), halve_u);
            r = Uint::ct_select(&r, &self.halve_residue(&difference_r), subtract_u);
            s = Uint::ct_select(&s, &self.halve_residue(&s), halve_v);
            s = Uint::ct_select(&s, &self.halve_residue(&difference_s), subtract_v);
        }
        let u_one = u.ct_eq(&Uint::ONE);
        let valid = u_one | v.ct_eq(&Uint::ONE);
        let inverse = Uint::ct_select(&s, &r, u_one);
        CtValue::new(Uint::ct_select(&Uint::ZERO, &inverse, valid), valid)
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn halve_residue(&self, value: &Uint<L>) -> Uint<L> {
        let odd = value.bit(0).mask();
        let (sum, carry) = value.adc(self.ring.modulus());
        let mut half = Uint::ct_select(value, &sum, odd).shr(1);
        half.0[L - 1] |= (carry & odd.word()) << 63;
        half
    }
}

enum PowerTable<T> {
    Private(Vec<T>),
    Public {
        base: T,
        rows: Vec<Vec<T>>,
        window: usize,
    },
}

/// Provider-owned fixed-base preparation with explicit exponent timing contracts.
pub struct FixedBasePow<C: RingOps, const EXP_LIMBS: usize> {
    field: C,
    table: PowerTable<C::Elem>,
}
impl<C: RingOps, const E: usize> FixedBasePow<C, E> {
    pub fn new(field: C, base: C::Elem) -> Self {
        const {
            assert!(E > 0);
        }
        let mut powers = Vec::with_capacity(E * 64);
        let mut value = base;
        for _ in 0..E * 64 {
            powers.push(value);
            value = field.square(&value);
        }
        Self {
            field,
            table: PowerTable::Private(powers),
        }
    }
    /// Window tables are indexed only by public exponents passed to `pow_public`.
    /// Private exponentiation remains available through `pow_ct`, without using
    /// these tables or accessing an exponent-dependent address.
    pub fn new_public(field: C, base: C::Elem, window: usize) -> Self {
        const {
            assert!(E > 0);
        }
        assert!((1..=16).contains(&window), "window must be in 1..=16");
        let mut rows = Vec::with_capacity((64 * E).div_ceil(window));
        let mut power = base;
        for _ in 0..(64 * E).div_ceil(window) {
            let mut row = Vec::with_capacity(1 << window);
            let mut value = field.one();
            for _ in 0..1 << window {
                row.push(value);
                value = field.mul(&value, &power);
            }
            rows.push(row);
            for _ in 0..window {
                power = field.square(&power);
            }
        }
        Self {
            field,
            table: PowerTable::Public { base, rows, window },
        }
    }
    pub fn pow_ct(&self, exponent: &Uint<E>) -> C::Elem {
        let powers = match &self.table {
            PowerTable::Private(powers) => powers,
            PowerTable::Public { base, .. } => return self.field.pow_ct(base, exponent),
        };
        let mut value = self.field.one();
        for i in 0..E * 64 {
            let product = self.field.mul(&value, &powers[i]);
            value = C::Elem::ct_select(&value, &product, exponent.bit(i).mask());
        }
        value
    }
    /// Variable-time and public-exponent-only: indices and zero skips depend on
    /// the exponent. `pow_ct` always retains its private-input contract.
    pub fn pow_public(&self, exponent: &Uint<E>) -> C::Elem {
        let PowerTable::Public { rows, window, .. } = &self.table else {
            return self.pow_ct(exponent);
        };
        if E == 2 && *window == 8 {
            // The verifier's 128-bit public weights use byte windows. A
            // balanced product exposes independent field multiplications instead
            // of putting every table lookup on one long dependency chain.
            // Keep the original factor order, including the identity at digit 0.
            let mut products = [self.field.one(); 8];
            for (i, product) in products.iter_mut().enumerate() {
                let word = exponent.0[i / 4];
                let shift = (i % 4) * 16;
                let low = ((word >> shift) & 255) as usize;
                let high = ((word >> (shift + 8)) & 255) as usize;
                *product = self.field.mul(&rows[2 * i][low], &rows[2 * i + 1][high]);
            }
            let mut count = products.len();
            while count > 1 {
                for i in 0..count / 2 {
                    products[i] = self.field.mul(&products[2 * i], &products[2 * i + 1]);
                }
                count /= 2;
            }
            return products[0];
        }
        let mut value = self.field.one();
        for (i, row) in rows.iter().enumerate() {
            let bit = i * window;
            let (word, shift) = (bit / 64, bit % 64);
            let mut digit = exponent.0[word] >> shift;
            if shift + window > 64 && word + 1 < E {
                digit |= exponent.0[word + 1] << (64 - shift);
            }
            digit &= (1 << window) - 1;
            if digit != 0 {
                value = self.field.mul(&value, &row[digit as usize]);
            }
        }
        value
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicProductBounds {
    pub lhs_limbs: usize,
    pub rhs_limbs: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeError {
    InputLengths,
    OutputLength,
    PublicWidth,
}
impl core::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "arithmetic shape: {self:?}")
    }
}
impl std::error::Error for ShapeError {}

/// Reusable storage binding. Values must fit the declared public active widths;
/// the witness validation layer establishes this before private execution.
pub struct PreparedProducts<'a, const A: usize, const B: usize> {
    lhs: &'a [Uint<A>],
    rhs: &'a [Uint<B>],
    out: &'a mut [UintProduct<A, B>],
    bounds: PublicProductBounds,
}
impl<'a, const A: usize, const B: usize> PreparedProducts<'a, A, B> {
    pub fn new(
        lhs: &'a [Uint<A>],
        rhs: &'a [Uint<B>],
        out: &'a mut [UintProduct<A, B>],
        bounds: PublicProductBounds,
    ) -> Result<Self, ShapeError> {
        if lhs.len() != rhs.len() {
            return Err(ShapeError::InputLengths);
        }
        if lhs.len() != out.len() {
            return Err(ShapeError::OutputLength);
        }
        if bounds.lhs_limbs == 0
            || bounds.lhs_limbs > A
            || bounds.rhs_limbs == 0
            || bounds.rhs_limbs > B
        {
            return Err(ShapeError::PublicWidth);
        }
        Ok(Self {
            lhs,
            rhs,
            out,
            bounds,
        })
    }
    pub fn execute(&mut self) {
        use crate::integer::product::Words;
        for ((a, b), out) in self.lhs.iter().zip(self.rhs).zip(self.out.iter_mut()) {
            debug_assert!(
                a.as_words()[self.bounds.lhs_limbs..]
                    .iter()
                    .all(|v| *v == 0)
            );
            debug_assert!(
                b.as_words()[self.bounds.rhs_limbs..]
                    .iter()
                    .all(|v| *v == 0)
            );
            *out = UintProduct::ZERO;
            for i in 0..self.bounds.lhs_limbs {
                let mut carry = 0;
                for j in 0..self.bounds.rhs_limbs {
                    let sum = a.as_words()[i] as u128 * b.as_words()[j] as u128
                        + out.word(i + j) as u128
                        + carry;
                    out.set_word(i + j, sum as u64);
                    carry = sum >> 64;
                }
                out.set_word(i + self.bounds.rhs_limbs, carry as u64);
            }
        }
    }
    pub fn outputs(&self) -> &[UintProduct<A, B>] {
        self.out
    }
}

/// Retains a provider once for repeated projection of full declared integer widths.
pub struct PreparedIntegerProjection<C> {
    field: C,
}
impl<C> PreparedIntegerProjection<C> {
    pub fn new(field: C) -> Self {
        Self { field }
    }
    pub fn project<T>(&self, value: &T) -> C::Elem
    where
        C: IntegerEmbedding<T>,
    {
        self.field.from_integer(value)
    }
    pub fn project_into<T>(&self, input: &[T], out: &mut [C::Elem])
    where
        C: IntegerEmbedding<T>,
    {
        assert_eq!(input.len(), out.len());
        for (value, dst) in input.iter().zip(out) {
            *dst = self.field.from_integer(value);
        }
    }
}
