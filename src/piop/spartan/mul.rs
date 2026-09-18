//! Compact witnesses shared by the native integer multiplication relations.
use std::marker::PhantomData;

use super::SpartanMatrixError;
use crate::{pcs::IntegerMatrixLayout, sumcheck::outer::OuterRows};

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u128 {}
}

/// The supported native operand widths and their exact double-width products.
pub trait MulWord: sealed::Sealed + Copy + Default + Eq + std::fmt::Debug + Send + Sync {
    type Product: Copy + Send + Sync;
    const BITS: usize;
    fn multiply(x: Self, y: Self) -> (Self, Self);
    fn join(lo: Self, hi: Self) -> Self::Product;
    fn as_u128(self) -> u128;
    fn pack(
        witness: &MulWitness<Self>,
        layout: MulLayout<Self>,
        rows: &mut [Vec<u64>],
        high_gates: usize,
    );
}

/// Supplied product limbs are preserved, so the proof checks the caller's claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MulRow<T> {
    pub x: T,
    pub y: T,
    pub lo: T,
    pub hi: T,
}

impl<T: MulWord> MulRow<T> {
    pub fn new(x: T, y: T) -> Self {
        let (lo, hi) = T::multiply(x, y);
        Self { x, y, lo, hi }
    }
    pub fn product(&self) -> T::Product {
        T::join(self.lo, self.hi)
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error, Eq, PartialEq)]
pub enum MulError {
    #[error("a multiplication batch must not be empty")]
    EmptyBatch,
    #[error("the multiplication domain is too large")]
    DomainTooLarge,
    #[error("word width must be positive and satisfy t + W <= 126")]
    InvalidWordWidth,
    #[error("packed rows do not match the committed layout")]
    InvalidRowShape,
    #[error(transparent)]
    SpartanMatrix(#[from] SpartanMatrixError),
}

/// Assignment and commitment geometry; the word type also selects the relation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MulLayout<T: MulWord> {
    pub(super) multiplications: usize,
    pub(super) capacity: usize,
    pub(super) gate_vars: usize,
    col_vars: usize,
    pub(super) word_bits: usize,
    word: PhantomData<T>,
}

impl<T: MulWord> MulLayout<T> {
    pub fn new(multiplications: usize) -> Result<Self, MulError> {
        if multiplications == 0 {
            return Err(MulError::EmptyBatch);
        }
        let capacity = multiplications
            .max(256)
            .checked_next_power_of_two()
            .ok_or(MulError::DomainTooLarge)?;
        capacity
            .checked_mul(4 * T::BITS)
            .ok_or(MulError::DomainTooLarge)?;
        let gate_vars = capacity.trailing_zeros() as usize;
        // Keep the measured u64 row cap and the existing transcript geometry.
        let col_vars = if T::BITS == 64 {
            (gate_vars / 2).max(gate_vars.saturating_sub(10))
        } else {
            gate_vars / 2
        };
        Ok(Self {
            multiplications,
            capacity,
            gate_vars,
            col_vars,
            word_bits: 1,
            word: PhantomData,
        })
    }
    pub fn new_with_word_bits(n: usize, word_bits: usize) -> Result<Self, MulError> {
        let mut layout = Self::new(n)?;
        if word_bits == 0 || word_bits > 126 {
            return Err(MulError::InvalidWordWidth);
        }
        layout.word_bits = word_bits;
        layout.validate_packing()?;
        Ok(layout)
    }
    fn validate_packing(&self) -> Result<(), MulError> {
        let p = self.bitz_params();
        if p.row_vars + self.word_bits > 126 {
            return Err(MulError::InvalidWordWidth);
        }
        self.capacity
            .checked_mul(4 * self.cells_per_limb() * p.word_bits)
            .ok_or(MulError::DomainTooLarge)?;
        Ok(())
    }
    pub const fn cells_per_limb(&self) -> usize {
        T::BITS.div_ceil(self.word_bits).next_power_of_two()
    }
    pub const fn uses_direct_opening(&self) -> bool {
        self.word_bits == 1 || (T::BITS == 32 && self.word_bits == 8)
    }
    /// Compact source grid committed before challenges; new packing widths use W=1 here.
    pub const fn committed_layout(&self) -> IntegerMatrixLayout {
        if self.uses_direct_opening() {
            self.bitz_params()
        } else {
            let mut source = *self;
            source.word_bits = 1;
            source.bitz_params()
        }
    }
    pub const fn word_bits(&self) -> usize {
        self.word_bits
    }
    pub const fn multiplications(&self) -> usize {
        self.multiplications
    }
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
    pub const fn gate_vars(&self) -> usize {
        self.gate_vars
    }
    pub const fn col_vars(&self) -> usize {
        self.col_vars
    }
    pub const fn assignment_len(&self) -> usize {
        self.capacity * if T::BITS == 64 { 5 } else { 4 }
    }
    pub const fn assignment_vars(&self) -> usize {
        self.gate_vars + if T::BITS == 64 { 3 } else { 2 }
    }
    pub const fn padded_assignment_len(&self) -> usize {
        1 << self.assignment_vars()
    }
    pub fn with_split_shift(mut self, shift: i8) -> Result<Self, MulError> {
        let base = if T::BITS == 64 {
            (self.gate_vars / 2).max(self.gate_vars.saturating_sub(10))
        } else {
            self.gate_vars / 2
        };
        let columns = base as i64 + i64::from(shift);
        if columns < 0 || columns > self.gate_vars as i64 {
            return Err(MulError::DomainTooLarge);
        }
        self.col_vars = columns as usize;
        self.validate_packing()?;
        Ok(self)
    }
    /// Opening grid, with logical W-bit cells padded to a power-of-two stride.
    pub const fn bitz_params(&self) -> IntegerMatrixLayout {
        IntegerMatrixLayout {
            row_vars: self.gate_vars - self.col_vars
                + (4 * self.cells_per_limb()).trailing_zeros() as usize,
            col_vars: self.col_vars,
            word_bits: self.word_bits.next_power_of_two(),
        }
    }
    pub const fn bitz_bit_position(
        &self,
        bit_slot: usize,
        gate: usize,
    ) -> Option<(usize, usize, usize)> {
        if bit_slot >= 4 * T::BITS || gate >= self.capacity {
            return None;
        }
        let high = self.gate_vars - self.col_vars;
        let limb = bit_slot / T::BITS;
        let bit = bit_slot % T::BITS;
        let cell = limb * self.cells_per_limb() + bit / self.word_bits;
        Some((
            (cell << high) | (gate >> self.col_vars),
            gate & ((1 << self.col_vars) - 1),
            bit % self.word_bits,
        ))
    }
    pub const fn bitz_cell(&self, bit_slot: usize, gate: usize) -> Option<(usize, usize)> {
        match self.bitz_bit_position(bit_slot, gate) {
            Some((row, col, _)) => Some((row, col)),
            None => None,
        }
    }
}

/// Four padded native blocks `[x | y | lo | hi]`.
/// The constant assignment block and MLE padding are implicit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MulWitness<T: MulWord> {
    pub(super) layout: MulLayout<T>,
    values: [Box<[T]>; 4],
}

impl<T: MulWord> MulWitness<T> {
    pub fn from_inputs(inputs: &[(T, T)]) -> Result<Self, MulError> {
        Self::from_fn(inputs.len(), |i| inputs[i])
    }
    pub fn from_fn(
        multiplications: usize,
        input: impl FnMut(usize) -> (T, T),
    ) -> Result<Self, MulError> {
        Self::from_fn_with_layout(MulLayout::new(multiplications)?, input)
    }
    pub fn from_fn_with_word_bits(
        n: usize,
        bits: usize,
        input: impl FnMut(usize) -> (T, T),
    ) -> Result<Self, MulError> {
        Self::from_fn_with_layout(MulLayout::new_with_word_bits(n, bits)?, input)
    }
    pub fn from_inputs_with_word_bits(inputs: &[(T, T)], bits: usize) -> Result<Self, MulError> {
        Self::from_fn_with_word_bits(inputs.len(), bits, |i| inputs[i])
    }
    pub fn from_rows_with_word_bits(rows: &[MulRow<T>], bits: usize) -> Result<Self, MulError> {
        Self::from_rows_with_layout(MulLayout::new_with_word_bits(rows.len(), bits)?, rows)
    }
    pub fn from_fn_with_layout(
        layout: MulLayout<T>,
        mut input: impl FnMut(usize) -> (T, T),
    ) -> Result<Self, MulError> {
        Self::from_row_fn(layout, |i| {
            let (x, y) = input(i);
            MulRow::new(x, y)
        })
    }
    pub fn from_rows(rows: &[MulRow<T>]) -> Result<Self, MulError> {
        Self::from_rows_with_layout(MulLayout::new(rows.len())?, rows)
    }
    pub fn from_rows_with_layout(
        layout: MulLayout<T>,
        rows: &[MulRow<T>],
    ) -> Result<Self, MulError> {
        if rows.len() != layout.multiplications {
            return Err(MulError::DomainTooLarge);
        }
        Self::from_row_fn(layout, |i| rows[i])
    }
    pub(super) fn from_row_fn(
        layout: MulLayout<T>,
        mut input: impl FnMut(usize) -> MulRow<T>,
    ) -> Result<Self, MulError> {
        // Separate blocks let the allocator reuse medium-sized buffers; joining
        // them into one large allocation causes repeated page faults on glibc.
        // Large partially filled batches retain demand-zero padding pages.
        if layout.multiplications < layout.capacity
            && 4 * layout.capacity * std::mem::size_of::<T>() >= 16 * 1024 * 1024
        {
            let mut values =
                std::array::from_fn(|_| vec![T::default(); layout.capacity].into_boxed_slice());
            let [xs, ys, los, his] = values.each_mut().map(|v| &mut v[..layout.multiplications]);
            for i in 0..layout.multiplications {
                let row = input(i);
                xs[i] = row.x;
                ys[i] = row.y;
                los[i] = row.lo;
                his[i] = row.hi;
            }
            return Ok(Self { layout, values });
        }
        let mut values = std::array::from_fn(|_| Box::<[T]>::new_uninit_slice(layout.capacity));
        let [xs, ys, los, his] = &mut values;
        for block in [&mut *xs, &mut *ys, &mut *los, &mut *his] {
            block[layout.multiplications..].fill(std::mem::MaybeUninit::new(T::default()));
        }
        let [xs, ys, los, his] = values.each_mut().map(|v| &mut v[..layout.multiplications]);
        for i in 0..layout.multiplications {
            let row = input(i);
            xs[i].write(row.x);
            ys[i].write(row.y);
            los[i].write(row.lo);
            his[i].write(row.hi);
        }
        // SAFETY: each of the four blocks has its live prefix and padding
        // initialized above. A panicking callback only drops MaybeUninit<T>.
        let values = values.map(|block| unsafe { block.assume_init() });
        Ok(Self { layout, values })
    }
    pub const fn layout(&self) -> &MulLayout<T> {
        &self.layout
    }
    pub fn with_split_shift(mut self, shift: i8) -> Result<Self, MulError> {
        self.layout = self.layout.with_split_shift(shift)?;
        Ok(self)
    }
    #[inline]
    fn blocks(&self) -> [&[T]; 4] {
        self.values.each_ref().map(|v| v.as_ref())
    }
    pub(crate) fn native_products(&self) -> super::raw_monty::NativeWideProducts<'_, T> {
        let live = self.layout.multiplications;
        let [x, y, lo, hi] = self.blocks().map(|v| &v[..live]);
        super::raw_monty::NativeWideProducts::new(x, y, lo, hi, live.next_power_of_two())
    }
    pub fn x_values(&self) -> &[T] {
        self.blocks()[0]
    }
    pub fn y_values(&self) -> &[T] {
        self.blocks()[1]
    }
    pub fn z_lo_values(&self) -> &[T] {
        self.blocks()[2]
    }
    pub fn z_hi_values(&self) -> &[T] {
        self.blocks()[3]
    }
    pub fn product(&self, row: usize) -> T::Product {
        T::join(self.z_lo_values()[row], self.z_hi_values()[row])
    }
    pub fn rows(&self) -> impl ExactSizeIterator<Item = MulRow<T>> + '_ {
        (0..self.layout.multiplications).map(|i| MulRow {
            x: self.x_values()[i],
            y: self.y_values()[i],
            lo: self.z_lo_values()[i],
            hi: self.z_hi_values()[i],
        })
    }
    /// Packs the compact commitment grid returned by `committed_layout`.
    pub fn bitz_bit_rows(&self) -> Vec<Vec<u64>> {
        self.bit_rows(self.packing_layout())
    }
    /// Reuses caller-owned rows, avoiding allocation churn when packing batches
    /// of the same shape. Every output word is overwritten, including padding.
    pub fn write_bitz_bit_rows(&self, rows: &mut [Vec<u64>]) -> Result<(), MulError> {
        let layout = self.packing_layout();
        let p = layout.bitz_params();
        let words = (p.rows() * p.word_bits).div_ceil(64);
        if rows.len() != p.cols() || rows.iter().any(|row| row.len() != words) {
            return Err(MulError::InvalidRowShape);
        }
        self.pack_rows(layout, rows);
        Ok(())
    }
    fn packing_layout(&self) -> MulLayout<T> {
        let mut layout = self.layout;
        if !layout.uses_direct_opening() {
            layout.word_bits = 1;
        }
        layout
    }
    pub(crate) fn derived_bit_rows(&self) -> Vec<Vec<u64>> {
        self.bit_rows(self.layout)
    }
    fn bit_rows(&self, layout: MulLayout<T>) -> Vec<Vec<u64>> {
        let params = layout.bitz_params();
        let words = (params.rows() * params.word_bits).div_ceil(64);
        let mut rows = (0..params.cols())
            .map(|_| vec![0; words])
            .collect::<Vec<_>>();
        self.pack_rows(layout, &mut rows);
        rows
    }
    fn pack_rows(&self, layout: MulLayout<T>, rows: &mut [Vec<u64>]) {
        let params = layout.bitz_params();
        let high_gates = 1 << (layout.gate_vars - layout.col_vars);
        if (params.word_bits == 1 && high_gates % 64 == 0)
            || (T::BITS == 32 && layout.word_bits == 8 && high_gates % 8 == 0)
        {
            T::pack(self, layout, rows, high_gates);
        } else {
            for row in rows.iter_mut() {
                row.fill(0);
            }
            self.write_bit_rows_bitwise_at(layout, rows);
        }
    }
    #[cfg(test)]
    pub(super) fn write_bit_rows_bitwise(&self, rows: &mut [Vec<u64>]) {
        self.write_bit_rows_bitwise_at(self.layout, rows);
    }
    fn write_bit_rows_bitwise_at(&self, layout: MulLayout<T>, rows: &mut [Vec<u64>]) {
        let params = layout.bitz_params();
        for (i, row) in self.rows().enumerate() {
            for (limb, value) in [row.x, row.y, row.lo, row.hi].into_iter().enumerate() {
                let mut bits = value.as_u128();
                while bits != 0 {
                    let slot = limb * T::BITS + bits.trailing_zeros() as usize;
                    let (b, c, j) = layout
                        .bitz_bit_position(slot, i)
                        .expect("valid witness coordinate");
                    let bit = b * params.word_bits + j;
                    rows[c][bit / 64] |= 1 << (bit % 64);
                    bits &= bits - 1;
                }
            }
        }
    }
}

impl<T: MulWord> OuterRows for MulWitness<T> {
    type AB = T;
    type C = T::Product;
    fn dimensions(&self) -> (usize, usize, usize) {
        let n = self.layout.multiplications.next_power_of_two();
        (n, n, n)
    }
    #[inline(always)]
    fn a(&self, row: usize) -> T {
        self.x_values()[row]
    }
    #[inline(always)]
    fn b(&self, row: usize) -> T {
        self.y_values()[row]
    }
    #[inline(always)]
    fn c(&self, row: usize) -> T::Product {
        self.product(row)
    }
}

impl MulWord for u32 {
    type Product = u64;
    const BITS: usize = 32;
    #[inline(always)]
    fn multiply(x: Self, y: Self) -> (Self, Self) {
        let p = u64::from(x) * u64::from(y);
        (p as u32, (p >> 32) as u32)
    }
    #[inline(always)]
    fn join(lo: Self, hi: Self) -> u64 {
        u64::from(lo) | (u64::from(hi) << 32)
    }
    fn as_u128(self) -> u128 {
        self.into()
    }
    fn pack(w: &MulWitness<Self>, layout: MulLayout<Self>, rows: &mut [Vec<u64>], high: usize) {
        let limbs = w.blocks();
        if layout.word_bits == 8 {
            super::slot_rows::pack_slot_major_u32::<4, 8, 32>(
                rows,
                layout.col_vars,
                high,
                layout.multiplications,
                limbs,
            );
        } else {
            super::slot_rows::pack_slot_major_u32::<32, 1, 8>(
                rows,
                layout.col_vars,
                high,
                layout.multiplications,
                limbs,
            );
        }
    }
}
impl MulWord for u64 {
    type Product = u128;
    const BITS: usize = 64;
    #[inline(always)]
    fn multiply(x: Self, y: Self) -> (Self, Self) {
        let p = u128::from(x) * u128::from(y);
        (p as u64, (p >> 64) as u64)
    }
    #[inline(always)]
    fn join(lo: Self, hi: Self) -> u128 {
        u128::from(lo) | (u128::from(hi) << 64)
    }
    fn as_u128(self) -> u128 {
        self.into()
    }
    fn pack(w: &MulWitness<Self>, layout: MulLayout<Self>, rows: &mut [Vec<u64>], high: usize) {
        let [x, y, lo, hi] = w.blocks().map(|v| &v[..layout.multiplications]);
        super::slot_rows::pack_slot_major_rows_w1_words::<4, _>(
            rows,
            layout.col_vars,
            high,
            layout.multiplications,
            |i| [x[i], y[i], lo[i], hi[i]],
        );
    }
}
impl MulWord for u128 {
    type Product = field::Uint<4>;
    const BITS: usize = 128;
    #[inline(always)]
    fn multiply(x: Self, y: Self) -> (Self, Self) {
        super::u128_mul::mul_u128_full(x, y)
    }
    #[inline(always)]
    fn join(lo: Self, hi: Self) -> Self::Product {
        field::Uint::from_words([lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64])
    }
    fn as_u128(self) -> u128 {
        self
    }
    fn pack(w: &MulWitness<Self>, layout: MulLayout<Self>, rows: &mut [Vec<u64>], high: usize) {
        let [x, y, lo, hi] = w.blocks().map(|v| &v[..layout.multiplications]);
        super::slot_rows::pack_slot_major_rows_w1_words::<8, _>(
            rows,
            layout.col_vars,
            high,
            layout.multiplications,
            |i| {
                [
                    x[i] as u64,
                    (x[i] >> 64) as u64,
                    y[i] as u64,
                    (y[i] >> 64) as u64,
                    lo[i] as u64,
                    (lo[i] >> 64) as u64,
                    hi[i] as u64,
                    (hi[i] >> 64) as u64,
                ]
            },
        );
    }
}

impl From<MulError> for super::protocol::ProtocolError {
    fn from(error: MulError) -> Self {
        Self::Relation(Box::new(error))
    }
}

impl MulWitness<u32> {
    pub(crate) fn inner_witness(&self) -> super::raw_monty::RawWitness<'_> {
        super::raw_monty::RawWitness::Wide(super::raw_monty::NativeBlockWitness::u32(
            self.layout.capacity,
            self.x_values(),
            self.y_values(),
            self.z_lo_values(),
            self.z_hi_values(),
        ))
    }
}
impl MulWitness<u64> {
    pub(crate) fn inner_witness(&self) -> super::raw_monty::RawWitness<'_> {
        super::raw_monty::RawWitness::Wide(super::raw_monty::NativeBlockWitness::u64(
            self.layout.capacity,
            self.x_values(),
            self.y_values(),
            self.z_lo_values(),
            self.z_hi_values(),
        ))
    }
}
impl MulWitness<u128> {
    pub(crate) fn inner_witness(&self) -> super::raw_monty::RawWitness<'_> {
        super::raw_monty::RawWitness::Wide(super::raw_monty::NativeBlockWitness::new(
            self.layout.capacity,
            self.x_values(),
            self.y_values(),
            self.z_lo_values(),
            self.z_hi_values(),
        ))
    }
}

/// Flat CSC construction shared by all multiplication selectors.
pub(super) fn selector_matrix<T: MulWord, C: Clone>(
    layout: &MulLayout<T>,
    blocks: &[(usize, C)],
) -> Result<circuit::linear_map::CscMatrix<Box<[C]>>, SpartanMatrixError> {
    let rows = layout.multiplications;
    let mut offsets = vec![0; layout.assignment_len() + 1];
    let mut indices = Vec::with_capacity(rows * blocks.len());
    let mut coefficients = Vec::with_capacity(rows * blocks.len());
    let mut previous = 0;
    for (block, coefficient) in blocks {
        let offset = block * layout.capacity;
        offsets[previous..=offset].fill(indices.len());
        let start = indices.len();
        for row in 0..rows {
            offsets[offset + row + 1] = start + row + 1;
            indices.push(row);
            coefficients.push(coefficient.clone());
        }
        previous = offset + rows + 1;
    }
    offsets[previous..].fill(indices.len());
    Ok(circuit::linear_map::CscMatrix::try_from_csc_parts(
        rows,
        offsets,
        indices,
        coefficients,
    )?)
}

impl<T: MulWord> circuit::linear_map::binary::VirtualMap for MulLayout<T> {
    type ColumnRows<'a>
        = std::option::IntoIter<usize>
    where
        Self: 'a;
    fn rows(&self) -> usize {
        self.capacity * 4 * self.cells_per_limb() * self.word_bits.next_power_of_two()
    }
    fn cols(&self) -> usize {
        self.capacity * 4 * T::BITS
    }
    fn nnz(&self) -> usize {
        self.cols()
    }
    fn digest(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"bitz/multiplication-word-packing/v1");
        for value in [T::BITS, self.capacity, self.col_vars, self.word_bits] {
            h.update(&(value as u64).to_le_bytes());
        }
        *h.finalize().as_bytes()
    }
    fn is_identity(&self) -> bool {
        self.uses_direct_opening()
    }
    fn output_word_bits(&self, word_stride: usize) -> Option<usize> {
        (word_stride == self.word_bits.next_power_of_two()).then_some(self.word_bits)
    }
    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>> {
        if column >= self.cols() {
            return None;
        }
        let source = self.committed_layout();
        let source_log_w = source.word_bits.ilog2() as usize;
        let source_row_bits = source.row_vars + source_log_w;
        let c = column >> source_row_bits;
        let i = column & ((1 << source_row_bits) - 1);
        let high = self.gate_vars - self.col_vars;
        let gate_high = (i >> source_log_w) & ((1 << high) - 1);
        let bit_slot =
            (i >> (source_log_w + high)) * source.word_bits + (i & (source.word_bits - 1));
        let gate = (gate_high << self.col_vars) | c;
        let (b, c, j) = self.bitz_bit_position(bit_slot, gate)?;
        let p = self.bitz_params();
        Some(
            Some((c << (p.row_vars + p.word_bits.ilog2() as usize)) + b * p.word_bits + j)
                .into_iter(),
        )
    }
}

impl<T: MulWord> MulLayout<T> {
    pub(super) fn validate_protocol_geometry(&self) -> Result<(), super::protocol::ProtocolError> {
        self.validate_packing()?;
        for p in [self.committed_layout(), self.bitz_params()] {
            if p.col_vars > self.gate_vars
                || p.row_vars + (p.word_bits.ilog2() as usize) < crate::ligerito::LOG_PACKING
            {
                return Err(super::protocol::ProtocolError::InvalidBitzParameters);
            }
        }
        Ok(())
    }
    pub(super) fn bind_packing(
        &self,
        h: &mut super::protocol::BindingHasher,
    ) -> Result<(), super::protocol::ProtocolError> {
        if !self.uses_direct_opening() {
            h.bytes(b"packed-multiplication/v1")
                .bytes(&circuit::linear_map::binary::VirtualMap::digest(self));
            h.usize(self.word_bits)?;
        }
        Ok(())
    }
    pub(super) fn packed_claim_digest(
        &self,
        frame: super::protocol::ClaimFrame<'_>,
    ) -> Result<[u8; 32], super::protocol::ProtocolError> {
        let mut h = super::protocol::BindingHasher::new();
        h.bytes(b"bitz/packed-multiplication/claim/v1")
            .bytes(frame.binding)
            .bytes(frame.matrices_digest)
            .bytes(&circuit::linear_map::binary::VirtualMap::digest(self));
        h.u128_le(frame.field.modulus_u128());
        h.usize(frame.terminal_claim.point().len())?;
        for x in frame.terminal_claim.point() {
            h.element(x, frame.field);
        }
        h.element(frame.terminal_claim.scale(), frame.field)
            .element(frame.terminal_claim.value(), frame.field);
        for weights in [frame.row_weights, frame.col_weights] {
            h.usize(weights.len())?;
            for &w in weights {
                h.u128_le(w);
            }
        }
        h.u128_le(frame.opening.claimed);
        Ok(h.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        piop::spartan::protocol::{self, PreparedRelation, RelationSpec},
        transcript::Blake3Transcript,
    };
    use circuit::linear_map::binary::VirtualMap;

    fn check_packing<T: MulWord>(n: usize, width: usize, row: MulRow<T>) {
        let layout = MulLayout::<T>::new_with_word_bits(n, width).unwrap();
        // Use supplied limbs independent of multiplication: packing must preserve all bits.
        let witness = MulWitness::from_row_fn(layout, |_| row).unwrap();
        let source = witness.bitz_bit_rows();
        let derived = witness.derived_bit_rows();
        let p = layout.bitz_params();
        let mut expected = vec![vec![0_u64; p.rows() * p.word_bits / 64]; p.cols()];
        let high = 1 << (layout.gate_vars() - layout.col_vars());
        for gate in 0..n {
            for (limb, value) in [row.x, row.y, row.lo, row.hi].into_iter().enumerate() {
                for bit in 0..T::BITS {
                    if value.as_u128() >> bit & 1 == 0 {
                        continue;
                    }
                    let cell = limb * T::BITS.div_ceil(width).next_power_of_two() + bit / width;
                    let index = (cell * high + gate / p.cols()) * p.word_bits + bit % width;
                    expected[gate % p.cols()][index / 64] |= 1 << (index % 64);
                }
            }
        }
        assert_eq!(derived, expected, "{}-bit word, W={width}, n={n}", T::BITS);
        let mut mapped = vec![vec![0_u64; p.rows() * p.word_bits / 64]; p.cols()];
        let source_bits = source[0].len() * 64;
        let derived_bits = mapped[0].len() * 64;
        for column in 0..VirtualMap::cols(&layout) {
            let bit = source[column / source_bits][column % source_bits / 64] >> (column % 64) & 1;
            for row in layout.column_rows(column).unwrap() {
                mapped[row / derived_bits][row % derived_bits / 64] ^= bit << (row % 64);
            }
        }
        assert_eq!(mapped, expected);
        assert!(layout.column_rows(VirtualMap::cols(&layout)).is_none());
        crate::pcs::VirtualWordBound::new(p, width, &layout).unwrap();
    }

    #[test]
    fn arbitrary_packing_matches_public_map_and_native_limbs() {
        for width in [1, 2, 3, 5, 6, 7, 8, 17, 31, 32, 63, 64, 65, 100, 117] {
            check_packing(
                257,
                width,
                MulRow {
                    x: u32::MAX,
                    y: 0x80102041,
                    lo: 0xfefe1313,
                    hi: 0xfeedface,
                },
            );
            check_packing(
                257,
                width,
                MulRow {
                    x: u64::MAX,
                    y: 0x8010204180808081,
                    lo: 0xfefe131341341234,
                    hi: 0xfeedface87654321,
                },
            );
            check_packing(
                257,
                width,
                MulRow {
                    x: u128::MAX,
                    y: (1_u128 << 127) | 0x123456789abcdef,
                    lo: 0xfefe131341341234,
                    hi: (1_u128 << 120) | 0xfeedface87654321,
                },
            );
        }
        // These dimensions exercise the transposed u32 packer branch.
        for width in [5, 6, 7, 8] {
            check_packing(
                4097,
                width,
                MulRow {
                    x: u32::MAX,
                    y: 0,
                    lo: 0xffffffff,
                    hi: 0xffffffff,
                },
            );
        }
    }

    #[test]
    fn reused_packing_overwrites_live_values_and_padding() {
        fn check<T: MulWord>(n: usize, width: usize, x: T) {
            let witness = MulWitness::from_fn_with_word_bits(n, width, |_| (x, x)).unwrap();
            let expected = witness.bitz_bit_rows();
            let mut reused = expected.clone();
            for row in &mut reused {
                row.fill(u64::MAX);
            }
            witness.write_bitz_bit_rows(&mut reused).unwrap();
            assert_eq!(reused, expected);
            let zeros =
                MulWitness::from_fn_with_word_bits(n, width, |_| (T::default(), T::default()))
                    .unwrap();
            zeros.write_bitz_bit_rows(&mut reused).unwrap();
            assert!(reused.iter().flatten().all(|&word| word == 0));
            reused[0].pop();
            let malformed = reused.clone();
            assert_eq!(
                witness.write_bitz_bit_rows(&mut reused),
                Err(MulError::InvalidRowShape)
            );
            assert_eq!(reused, malformed);
        }
        for n in [37, 2053] {
            check(n, 1, u32::MAX);
            check(n, 8, u32::MAX);
            check(n, 7, u32::MAX);
            check(n, 1, u64::MAX);
            check(n, 3, u128::MAX);
        }
    }

    #[test]
    fn large_partial_batches_preserve_limbs_and_zero_padding() {
        fn check<T: MulWord>(n: usize, x: T, y: T) {
            let witness = MulWitness::from_fn(n, |_| (x, y)).unwrap();
            let (lo, hi) = T::multiply(x, y);
            for (values, expected) in [
                (witness.x_values(), x),
                (witness.y_values(), y),
                (witness.z_lo_values(), lo),
                (witness.z_hi_values(), hi),
            ] {
                assert_eq!(values.len(), witness.layout().capacity());
                assert!(values[..n].iter().all(|&v| v == expected));
                assert!(values[n..].iter().all(|&v| v == T::default()));
            }
        }
        check((1 << 20) + 1, u32::MAX, u32::MAX);
        check((1 << 19) + 1, u64::MAX, u64::MAX);
        check((1 << 18) + 1, u128::MAX, u128::MAX);
    }

    #[test]
    fn packing_bounds_reject_zero_overflow_and_invalid_splits() {
        for width in [0, 121, 127, 128, usize::MAX] {
            assert!(MulLayout::<u32>::new_with_word_bits(256, width).is_err());
        }
        let limit = MulLayout::<u32>::new_with_word_bits(256, 120).unwrap();
        assert_eq!(limit.bitz_params().row_vars + limit.word_bits(), 126);
        assert!(limit.with_split_shift(-1).is_err());
        assert!(MulLayout::<u64>::new(usize::MAX).is_err());
        assert!(
            MulLayout::<u64>::new(1 << 24)
                .unwrap()
                .with_split_shift(11)
                .is_err()
        );
    }

    fn roundtrip<T: MulWord>(width: usize, input: impl FnMut(usize) -> (T, T))
    where
        MulLayout<T>: RelationSpec<Witness = MulWitness<T>>,
    {
        let witness = MulWitness::from_fn_with_word_bits(1 << 15, width, input).unwrap();
        let prepared = PreparedRelation::new(*witness.layout()).unwrap();
        let hint = protocol::commit(&prepared, witness.bitz_bit_rows()).unwrap();
        let proof =
            protocol::prove(&mut Blake3Transcript::new(), &prepared, &witness, &hint).unwrap();
        protocol::verify(
            &mut Blake3Transcript::new(),
            &prepared,
            &hint.commitment,
            &proof,
        )
        .unwrap();
        let mut wrong_root = hint.commitment.clone();
        wrong_root.root[0] ^= 1;
        assert!(
            protocol::verify(&mut Blake3Transcript::new(), &prepared, &wrong_root, &proof).is_err()
        );
        let mut rows: Vec<_> = witness.rows().collect();
        rows[3].hi = T::default();
        let bad = MulWitness::from_rows_with_layout(*witness.layout(), &rows).unwrap();
        assert_ne!(bad, witness);
        let bad_hint = protocol::commit(&prepared, bad.bitz_bit_rows()).unwrap();
        if let Ok(bad_proof) =
            protocol::prove(&mut Blake3Transcript::new(), &prepared, &bad, &bad_hint)
        {
            assert!(
                protocol::verify(
                    &mut Blake3Transcript::new(),
                    &prepared,
                    &bad_hint.commitment,
                    &bad_proof
                )
                .is_err()
            );
        }
        let wrong_layout =
            MulLayout::<T>::new_with_word_bits(1 << 15, if width == 3 { 5 } else { 3 }).unwrap();
        let wrong_prepared = PreparedRelation::new(wrong_layout).unwrap();
        assert!(
            protocol::verify(
                &mut Blake3Transcript::new(),
                &wrong_prepared,
                &hint.commitment,
                &proof
            )
            .is_err()
        );
    }

    #[test]
    fn arbitrary_packing_end_to_end_and_tampering() {
        for width in [3, 5, 6, 7] {
            roundtrip(width, |i| (u32::MAX - i as u32, u32::MAX));
        }
        for width in [3, 8] {
            roundtrip(width, |i| (u64::MAX - i as u64, u64::MAX));
        }
        for width in [5, 65] {
            roundtrip(width, |i| (u128::MAX - i as u128, u128::MAX));
        }
    }
}
