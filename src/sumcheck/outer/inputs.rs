//! Borrowed table views. Native adapters assemble split words only when read.
/// Exact row operands read at public indices. Implementations must return
/// stable values within the declared arithmetic capability's public bounds.
/// `validate` must reject inconsistent storage metadata before any transcript use.
pub trait OuterRows: Sync {
    type AB: Copy + Send + Sync;
    type C: Copy + Send + Sync;
    fn dimensions(&self) -> (usize, usize, usize);
    fn validate(&self) -> Result<usize, crate::sumcheck::SumcheckError> {
        let (a, b, c) = self.dimensions();
        super::api::validate_shape(a, b, c)
    }
    fn a(&self, row: usize) -> Self::AB;
    fn b(&self, row: usize) -> Self::AB;
    fn c(&self, row: usize) -> Self::C;
}

/// Borrowed A/B/C slices; no table allocation or projection.
pub struct OuterSlices<'a, AB, C> {
    pub ax: &'a [AB],
    pub bx: &'a [AB],
    pub cx: &'a [C],
}
impl<AB: Copy + Send + Sync, C: Copy + Send + Sync> OuterRows for OuterSlices<'_, AB, C> {
    type AB = AB;
    type C = C;
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.ax.len(), self.bx.len(), self.cx.len())
    }
    #[inline(always)]
    fn a(&self, row: usize) -> AB {
        self.ax[row]
    }
    #[inline(always)]
    fn b(&self, row: usize) -> AB {
        self.bx[row]
    }
    #[inline(always)]
    fn c(&self, row: usize) -> C {
        self.cx[row]
    }
}

// Ownership changes storage lifetime, not row arithmetic or protocol scheduling.
impl<I: OuterRows + ?Sized> OuterRows for &I {
    type AB = I::AB;
    type C = I::C;
    fn validate(&self) -> Result<usize, crate::sumcheck::SumcheckError> {
        (**self).validate()
    }
    fn dimensions(&self) -> (usize, usize, usize) {
        (**self).dimensions()
    }
    #[inline(always)]
    fn a(&self, row: usize) -> Self::AB {
        (**self).a(row)
    }
    #[inline(always)]
    fn b(&self, row: usize) -> Self::AB {
        (**self).b(row)
    }
    #[inline(always)]
    fn c(&self, row: usize) -> Self::C {
        (**self).c(row)
    }
}

impl<AB: Copy + Send + Sync, C: Copy + Send + Sync> OuterRows for super::OuterInputs<AB, C> {
    type AB = AB;
    type C = C;
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.ax.len(), self.bx.len(), self.cx.len())
    }
    #[inline(always)]
    fn a(&self, row: usize) -> AB {
        self.ax[row]
    }
    #[inline(always)]
    fn b(&self, row: usize) -> AB {
        self.bx[row]
    }
    #[inline(always)]
    fn c(&self, row: usize) -> C {
        self.cx[row]
    }
}

impl<E: crate::piop::spartan::SpartanField> OuterRows for crate::piop::spartan::R1csProductMles<E> {
    type AB = E;
    type C = E;
    fn dimensions(&self) -> (usize, usize, usize) {
        (
            self.az.evaluations.len(),
            self.bz.evaluations.len(),
            self.cz.evaluations.len(),
        )
    }
    fn validate(&self) -> Result<usize, crate::sumcheck::SumcheckError> {
        let (a, b, c) = self.dimensions();
        let n = super::api::validate_shape(a, b, c)?;
        if [self.az.num_vars, self.bz.num_vars, self.cz.num_vars] != [n; 3] {
            return Err(crate::sumcheck::SumcheckError::InvalidProductDimensions);
        }
        Ok(n)
    }
    #[inline(always)]
    fn a(&self, row: usize) -> E {
        self.az.evaluations[row]
    }
    #[inline(always)]
    fn b(&self, row: usize) -> E {
        self.bz.evaluations[row]
    }
    #[inline(always)]
    fn c(&self, row: usize) -> E {
        self.cz.evaluations[row]
    }
}
