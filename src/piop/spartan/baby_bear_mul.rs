//! Integer-level R1CS relation for batched BabyBear multiplication.
//!
//! For canonical BabyBear representatives `a`, `b`, and `c`, and the exact
//! integer quotient `k`, every live row enforces
//!
//! ```text
//! a * b = c + p * k,    p = 2^31 - 2^27 + 1.
//! ```
//!
//! The logical assignment is `[e0 | a | b | c | k]`. It contains five
//! capacity-sized blocks and is padded to eight blocks only when materialized
//! as Spartan's assignment MLE. The compact BitZ witness contains 31
//! little-endian bits for each of `a`, `b`, `c`, and `k`, followed by four
//! unused slots that the honest witness builder leaves zero. Verification does
//! not separately constrain those unused committed bits.
//!
//! Operand canonicality is enforced by the host constructors, not by an R1CS
//! comparison with `p`. The proof relation itself binds 31-bit integers and
//! the exact quotient identity.

use crate::piop::spartan::SpartanField as _;
use crate::sumcheck::bridge::PreparedBinding;
use circuit::linear_map::CscMatrix;
use field::RingOps;
use std::borrow::Cow;

use thiserror::Error;

use crate::{pcs::IntegerMatrixLayout, poly::mle::DenseMultilinearExtension};

use super::{
    ConstraintMatrices, ModulusIndependentCoefficient, PreparedConstraintMatrices, R1csProductMles,
    SpartanBitzField, SpartanField, SpartanMatrixCoefficient, SpartanMatrixError,
    build_assignment_mle, build_product_mles, slot_rows::pack_slot_major_rows_w1,
};

/// The BabyBear prime `2^31 - 2^27 + 1`.
pub const BABY_BEAR_MODULUS: u64 = 2_013_265_921;

/// Number of committed little-endian bits used for each BabyBear value.
pub(super) const BABY_BEAR_MUL_VALUE_BITS: usize = 31;
/// Number of committed bits used for the left operand.
pub(super) const BABY_BEAR_MUL_A_BITS: usize = BABY_BEAR_MUL_VALUE_BITS;
/// Number of committed bits used for the right operand.
pub(super) const BABY_BEAR_MUL_B_BITS: usize = BABY_BEAR_MUL_VALUE_BITS;
/// Number of committed bits used for the reduced product.
pub(super) const BABY_BEAR_MUL_C_BITS: usize = BABY_BEAR_MUL_VALUE_BITS;
/// Number of committed bits used for the integer quotient.
pub(super) const BABY_BEAR_MUL_K_BITS: usize = BABY_BEAR_MUL_VALUE_BITS;

/// First bit slot occupied by the left operand.
pub(super) const BABY_BEAR_MUL_A_SLOT_START: usize = 0;
/// First bit slot occupied by the right operand.
pub(super) const BABY_BEAR_MUL_B_SLOT_START: usize =
    BABY_BEAR_MUL_A_SLOT_START + BABY_BEAR_MUL_A_BITS;
/// First bit slot occupied by the reduced product.
pub(super) const BABY_BEAR_MUL_C_SLOT_START: usize =
    BABY_BEAR_MUL_B_SLOT_START + BABY_BEAR_MUL_B_BITS;
/// First bit slot occupied by the quotient.
pub(super) const BABY_BEAR_MUL_K_SLOT_START: usize =
    BABY_BEAR_MUL_C_SLOT_START + BABY_BEAR_MUL_C_BITS;
/// Number of semantic bit slots per multiplication.
pub(super) const BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS: usize =
    BABY_BEAR_MUL_A_BITS + BABY_BEAR_MUL_B_BITS + BABY_BEAR_MUL_C_BITS + BABY_BEAR_MUL_K_BITS;
/// Physical power-of-two bit slots committed per multiplication.
pub(super) const BABY_BEAR_MUL_BIT_SLOTS: usize = 128;

const LOGICAL_ASSIGNMENT_BLOCKS: usize = 5;
const PADDED_ASSIGNMENT_BLOCKS: usize = 8;
const BABY_BEAR_OPERAND_MASK: u32 = 0x7fff_ffff;
// Match the compact row-packing geometry accepted by the existing u32
// relation. The combined production proof applies its stricter 2^15 minimum.
const MIN_CAPACITY: usize = 1 << 8;

// `SpartanBitzField` is the 128-bit runtime-configured Montgomery field, whose
// canonical element encoding is a fixed-width, little-endian 16-byte string.
// BabyBear's modulus is 0x78000001. Keeping this encoding static lets matrix
// hashing borrow it for every modulus coefficient without a per-entry
// allocation or runtime-configuration cache.
const BABY_BEAR_MODULUS_FIELD_ENCODING: [u8; 16] = [
    0x01, 0x00, 0x00, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Samples a uniform canonical BabyBear operand from uniform `u32` draws.
///
/// Each draw is masked to 31 bits. Candidates in `[p, 2^31)` are rejected
/// and redrawn rather than reduced modulo `p`, avoiding modulo bias.
///
/// This rejection sampler has data-dependent running time and is not a
/// constant-time API.
pub fn sample_baby_bear_operand_with(mut next: impl FnMut() -> u32) -> u32 {
    loop {
        let candidate = next() & BABY_BEAR_OPERAND_MASK;
        if u64::from(candidate) < BABY_BEAR_MODULUS {
            return candidate;
        }
    }
}

/// Compact coefficients used by the BabyBear multiplication matrices.
///
/// Keeping the two public values as selectors avoids storing one 128-bit
/// runtime-configured field element for every matrix nonzero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BabyBearMulCoefficient {
    /// The field element one.
    One,
    /// The embedded BabyBear modulus.
    Modulus,
}

impl SpartanMatrixCoefficient<SpartanBitzField> for BabyBearMulCoefficient {
    fn validate(&self, _field_modulus_encoding: &[u8]) -> Result<(), SpartanMatrixError> {
        // Every Spartan field is at least 100 bits, so both public coefficients
        // are nonzero canonical elements in every accepted configuration.
        Ok(())
    }

    fn is_zero(&self) -> bool {
        false
    }

    fn canonical_field_encoding<'a>(
        &'a self,
        _field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
        field_one_encoding: &'a [u8],
    ) -> Cow<'a, [u8]> {
        match self {
            Self::One => Cow::Borrowed(field_one_encoding),
            Self::Modulus => Cow::Borrowed(&BABY_BEAR_MODULUS_FIELD_ENCODING),
        }
    }

    fn scale(
        &self,
        value: &SpartanBitzField,
        field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> SpartanBitzField {
        match self {
            Self::One => value.clone(),
            Self::Modulus => {
                let coefficient = SpartanBitzField::from_with_cfg(BABY_BEAR_MODULUS, field_config);
                let mut scaled = value.clone();
                scaled = field_config.mul(&(scaled), &(&coefficient));
                scaled
            }
        }
    }
}

/// Both public coefficients encode to modulus-independent bytes: `One` to
/// the field's canonical one (exactly like a Bit `true`) and `Modulus`
/// to the embedded BabyBear prime, which is canonical and never the unit in
/// any accepted (at least 100-bit) Spartan field.
impl ModulusIndependentCoefficient<SpartanBitzField> for BabyBearMulCoefficient {
    fn write_modulus_independent_encoding(&self, out: &mut Vec<u8>) {
        match self {
            Self::One => {
                ModulusIndependentCoefficient::<SpartanBitzField>::write_modulus_independent_encoding(
                    &true, out,
                )
            }
            Self::Modulus => out.extend_from_slice(&BABY_BEAR_MODULUS_FIELD_ENCODING),
        }
    }

    fn is_unit(&self) -> bool {
        matches!(self, Self::One)
    }
}

/// Failures while constructing the BabyBear multiplication relation.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum BabyBearMulError {
    /// A multiplication batch must contain at least one live row.
    #[error("a BabyBear multiplication batch must not be empty")]
    EmptyBatch,

    /// The padded assignment or committed-bit domain does not fit in `usize`.
    #[error("the BabyBear multiplication domain is too large")]
    DomainTooLarge,

    /// A host operand is not a canonical representative in `[0, p)`.
    #[error("operand {operand} at row {index} is not canonical: {value} >= 2013265921")]
    NonCanonicalOperand {
        /// Live multiplication row containing the invalid operand.
        index: usize,
        /// Operand name, either `a` or `b`.
        operand: &'static str,
        /// Rejected host value.
        value: u32,
    },

    /// The generated relation or projected witness is malformed.
    #[error(transparent)]
    SpartanMatrix(#[from] SpartanMatrixError),
}

/// Shared shape of the integer assignment and compact BitZ bit witness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BabyBearMulLayout {
    multiplications: usize,
    capacity: usize,
    gate_vars: usize,
}

impl BabyBearMulLayout {
    /// Creates a layout for `multiplications` live rows.
    ///
    /// The gate capacity is `max(256, multiplications).next_power_of_two()`.
    /// The production Spartan/BitZ API may impose a larger minimum.
    pub fn new(multiplications: usize) -> Result<Self, BabyBearMulError> {
        if multiplications == 0 {
            return Err(BabyBearMulError::EmptyBatch);
        }

        let capacity = multiplications
            .max(MIN_CAPACITY)
            .checked_next_power_of_two()
            .ok_or(BabyBearMulError::DomainTooLarge)?;

        capacity
            .checked_mul(LOGICAL_ASSIGNMENT_BLOCKS)
            .and_then(|_| capacity.checked_mul(PADDED_ASSIGNMENT_BLOCKS))
            .and_then(|_| capacity.checked_mul(BABY_BEAR_MUL_BIT_SLOTS))
            .ok_or(BabyBearMulError::DomainTooLarge)?;

        let gate_vars = capacity.trailing_zeros() as usize;
        Ok(Self {
            multiplications,
            capacity,
            gate_vars,
        })
    }

    /// Number of live multiplication rows.
    pub const fn multiplications(&self) -> usize {
        self.multiplications
    }

    /// Power-of-two gate capacity, including zero-padded gates.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of variables selecting one padded gate.
    pub const fn gate_vars(&self) -> usize {
        self.gate_vars
    }

    /// Logical assignment length: five blocks of `capacity` integers.
    pub const fn assignment_len(&self) -> usize {
        LOGICAL_ASSIGNMENT_BLOCKS * self.capacity
    }

    /// Assignment-MLE length after padding the five logical blocks to eight.
    pub const fn padded_assignment_len(&self) -> usize {
        PADDED_ASSIGNMENT_BLOCKS * self.capacity
    }

    /// Number of variables in the padded assignment MLE.
    pub const fn assignment_vars(&self) -> usize {
        self.gate_vars + 3
    }

    /// BitZ shape for the slot-major 31/31/31/31-bit witness.
    ///
    /// If `g = log2(capacity)`, the low `s = floor(g/2)` gate coordinates
    /// become BitZ columns. The remaining gate coordinates and seven physical
    /// slot coordinates become folded row variables.
    pub const fn bitz_params(&self) -> IntegerMatrixLayout {
        let s = self.gate_vars / 2;
        IntegerMatrixLayout {
            row_vars: 7 + self.gate_vars - s,
            col_vars: s,
            word_bits: 1,
        }
    }

    /// Maps `(bit_slot, gate)` to the BitZ row-major cell `(b, c)`.
    ///
    /// `params.cell_index(b, c) == bit_slot * capacity + gate`.
    pub const fn bitz_cell(&self, bit_slot: usize, gate: usize) -> Option<(usize, usize)> {
        if bit_slot >= BABY_BEAR_MUL_BIT_SLOTS || gate >= self.capacity {
            return None;
        }

        let s = self.gate_vars / 2;
        let column_mask = (1usize << s) - 1;
        let gate_high = gate >> s;
        let b = (bit_slot << (self.gate_vars - s)) | gate_high;
        let c = gate & column_mask;
        Some((b, c))
    }
}

/// Exact native assignment for a batch of BabyBear multiplications.
///
/// The assignment is logically `[e0 | a | b | c | k]`. Only the first
/// constant entry is one. Operands are canonical host values; `c` and `k`
/// are derived from their exact `u64` product.
///
/// Witness construction uses ordinary host integer multiplication, division,
/// and remainder. It is not a constant-time API and must not be treated as a
/// side-channel-hardened operation on secrets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BabyBearMulWitness {
    layout: BabyBearMulLayout,
    assignment: Box<[u64]>,
}

impl BabyBearMulWitness {
    /// Constructs the exact assignment from explicit operand pairs.
    ///
    /// Every pair is consumed exactly once and a noncanonical value is
    /// rejected; this constructor never redraws an invalid deterministic
    /// input. Quotient derivation uses variable-time host division and is not
    /// a constant-time API.
    pub fn from_inputs(inputs: &[(u32, u32)]) -> Result<Self, BabyBearMulError> {
        Self::from_fn(inputs.len(), |index| inputs[index])
    }

    /// Constructs the assignment without retaining a separate input vector.
    ///
    /// `input` is called exactly once for each row reached before success or
    /// the first invalid operand. It is not used as a rejection sampler.
    /// Quotient derivation uses variable-time host division and is not a
    /// constant-time API.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_fn(
        multiplications: usize,
        mut input: impl FnMut(usize) -> (u32, u32),
    ) -> Result<Self, BabyBearMulError> {
        let layout = BabyBearMulLayout::new(multiplications)?;
        let capacity = layout.capacity;
        let mut assignment = vec![0_u64; layout.assignment_len()];
        assignment[0] = 1;

        for index in 0..multiplications {
            let (a, b) = input(index);
            validate_operand(index, "a", a)?;
            validate_operand(index, "b", b)?;

            let a = u64::from(a);
            let b = u64::from(b);
            let product = a * b;
            assignment[capacity + index] = a;
            assignment[2 * capacity + index] = b;
            assignment[3 * capacity + index] = product % BABY_BEAR_MODULUS;
            assignment[4 * capacity + index] = product / BABY_BEAR_MODULUS;
        }

        Ok(Self {
            layout,
            assignment: assignment.into_boxed_slice(),
        })
    }

    /// Shape shared by this assignment and its bit representation.
    pub const fn layout(&self) -> &BabyBearMulLayout {
        &self.layout
    }

    /// Complete logical five-block integer assignment.
    pub fn assignment(&self) -> &[u64] {
        &self.assignment
    }

    /// Complete logical five-block integer assignment `w`.
    pub fn w(&self) -> &[u64] {
        self.assignment()
    }

    /// Padded left-operand block.
    pub fn a_values(&self) -> &[u64] {
        let capacity = self.layout.capacity;
        &self.assignment[capacity..2 * capacity]
    }

    /// Padded right-operand block.
    pub fn b_values(&self) -> &[u64] {
        let capacity = self.layout.capacity;
        &self.assignment[2 * capacity..3 * capacity]
    }

    /// Padded canonical-product block.
    pub fn c_values(&self) -> &[u64] {
        let capacity = self.layout.capacity;
        &self.assignment[3 * capacity..4 * capacity]
    }

    /// Padded exact-quotient block.
    pub fn k_values(&self) -> &[u64] {
        let capacity = self.layout.capacity;
        &self.assignment[4 * capacity..5 * capacity]
    }

    /// Alias for the padded exact-quotient block.
    pub fn quotient_values(&self) -> &[u64] {
        self.k_values()
    }

    /// Live `Aw` values.
    pub fn aw(&self) -> &[u64] {
        &self.a_values()[..self.layout.multiplications]
    }

    /// Live `Bw` values.
    pub fn bw(&self) -> &[u64] {
        &self.b_values()[..self.layout.multiplications]
    }

    /// Alias for the live `Aw` values using the sumcheck table name.
    pub fn az(&self) -> &[u64] {
        self.aw()
    }

    /// Alias for the live `Bw` values using the sumcheck table name.
    pub fn bz(&self) -> &[u64] {
        self.bw()
    }

    /// Builds compact `W=1` BitZ rows without materializing a cell tensor.
    ///
    /// Slots `0..31`, `31..62`, `62..93`, and `93..124` contain
    /// little-endian bits of `a`, `b`, `c`, and `k`. Slots `124..128` remain
    /// zero, as do all slots belonging to padded gates. Packing is
    /// intentionally variable-time, like the rest of witness generation.
    ///
    /// Row `c` is 128 lanes of `high_gate_count` bits: bit `gate_high` of
    /// lane `slot` is slot `slot` of gate `(gate_high << s) | c`. Whenever a
    /// lane spans whole words (`high_gate_count % 64 == 0`, every production
    /// layout) the rows are built by the block transposes of
    /// [`super::slot_rows`]; smaller layouts take the bitwise path. Both
    /// produce identical rows.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn bitz_bit_rows(&self) -> Vec<Vec<u64>> {
        let params = self.layout.bitz_params();
        let words_per_row = params.rows() / u64::BITS as usize;
        let mut rows = vec![vec![0_u64; words_per_row]; params.cols()];

        let s = self.layout.gate_vars / 2;
        let high_gate_count = 1_usize << (self.layout.gate_vars - s);
        if high_gate_count.is_multiple_of(u64::BITS as usize) {
            self.write_bit_rows_transposed(&mut rows, s, high_gate_count);
        } else {
            self.write_bit_rows_bitwise(&mut rows);
        }
        rows
    }

    /// Reference packing: one masked read-modify-write per committed bit.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_bit_rows_bitwise(&self, rows: &mut [Vec<u64>]) {
        for gate in 0..self.layout.multiplications {
            for (slot_offset, value) in [
                (BABY_BEAR_MUL_A_SLOT_START, self.a_values()[gate]),
                (BABY_BEAR_MUL_B_SLOT_START, self.b_values()[gate]),
                (BABY_BEAR_MUL_C_SLOT_START, self.c_values()[gate]),
                (BABY_BEAR_MUL_K_SLOT_START, self.k_values()[gate]),
            ] {
                write_value_bits(
                    rows,
                    &self.layout,
                    gate,
                    slot_offset,
                    BABY_BEAR_MUL_VALUE_BITS,
                    value,
                );
            }
        }
    }

    /// Block-transpose packing for layouts whose slot lanes span whole
    /// words (see [`super::slot_rows`]).
    fn write_bit_rows_transposed(&self, rows: &mut [Vec<u64>], s: usize, high_gate_count: usize) {
        let a_values = self.a_values();
        let b_values = self.b_values();
        let c_values = self.c_values();
        let k_values = self.k_values();
        pack_slot_major_rows_w1(
            rows,
            s,
            high_gate_count,
            self.layout.multiplications,
            |gate| {
                pack_gate_slots(
                    a_values[gate],
                    b_values[gate],
                    c_values[gate],
                    k_values[gate],
                )
            },
        );
    }

    /// Moves out the layout and logical assignment.
    pub fn into_parts(self) -> (BabyBearMulLayout, Box<[u64]>) {
        (self.layout, self.assignment)
    }
}

fn validate_operand(
    index: usize,
    operand: &'static str,
    value: u32,
) -> Result<(), BabyBearMulError> {
    if u64::from(value) >= BABY_BEAR_MODULUS {
        return Err(BabyBearMulError::NonCanonicalOperand {
            index,
            operand,
            value,
        });
    }
    Ok(())
}

#[allow(clippy::arithmetic_side_effects)]
fn write_value_bits(
    rows: &mut [Vec<u64>],
    layout: &BabyBearMulLayout,
    gate: usize,
    slot_offset: usize,
    bit_width: usize,
    value: u64,
) {
    for bit in 0..bit_width {
        if value & (1_u64 << bit) == 0 {
            continue;
        }
        let (b, c) = layout
            .bitz_cell(slot_offset + bit, gate)
            .expect("witness bit coordinates are in bounds");
        rows[c][b / u64::BITS as usize] |= 1_u64 << (b % u64::BITS as usize);
    }
}

/// Packs one gate's four values into its 128 committed slots
/// (`a | b << 31 | c << 62 | k << 93`, each value masked to its 31 committed
/// bits) and returns the words for slots `0..64` and `64..128`.
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)]
const fn pack_gate_slots(a: u64, b: u64, c: u64, k: u64) -> (u64, u64) {
    const VALUE_MASK: u128 = (1 << BABY_BEAR_MUL_VALUE_BITS) - 1;
    let packed = ((a as u128) & VALUE_MASK) << BABY_BEAR_MUL_A_SLOT_START
        | ((b as u128) & VALUE_MASK) << BABY_BEAR_MUL_B_SLOT_START
        | ((c as u128) & VALUE_MASK) << BABY_BEAR_MUL_C_SLOT_START
        | ((k as u128) & VALUE_MASK) << BABY_BEAR_MUL_K_SLOT_START;
    (packed as u64, (packed >> 64) as u64)
}

/// Builds the compact CSC matrices for the BabyBear integer relation.
///
/// For live row `i` and capacity `M`, the only nonzero entries are
///
/// ```text
/// A[i, M+i]  = 1
/// B[i, 2M+i] = 1
/// C[i, 3M+i] = 1
/// C[i, 4M+i] = p.
/// ```
pub fn baby_bear_mul_constraint_matrices(
    layout: &BabyBearMulLayout,
) -> Result<ConstraintMatrices<BabyBearMulCoefficient>, BabyBearMulError> {
    let a = selector_matrix(layout, 1, BabyBearMulCoefficient::One)?;
    let b = selector_matrix(layout, 2, BabyBearMulCoefficient::One)?;
    let c = output_matrix(layout)?;
    Ok(ConstraintMatrices::new(a, b, c)?)
}

#[allow(clippy::arithmetic_side_effects)]
fn selector_matrix(
    layout: &BabyBearMulLayout,
    block: usize,
    coefficient: BabyBearMulCoefficient,
) -> Result<CscMatrix<Box<[BabyBearMulCoefficient]>>, SpartanMatrixError> {
    let columns = layout.assignment_len();
    let rows = layout.multiplications;
    let offset = block * layout.capacity;

    let mut column_offsets = vec![0; columns + 1];
    for (row, boundary) in column_offsets[offset + 1..offset + rows + 1]
        .iter_mut()
        .enumerate()
    {
        *boundary = row + 1;
    }
    column_offsets[offset + rows + 1..].fill(rows);
    let entries = (0..rows).map(|row| (row, coefficient)).collect::<Vec<_>>();

    Ok(CscMatrix::try_from_csc(rows, column_offsets, entries)?)
}

#[allow(clippy::arithmetic_side_effects)]
fn output_matrix(
    layout: &BabyBearMulLayout,
) -> Result<CscMatrix<Box<[BabyBearMulCoefficient]>>, SpartanMatrixError> {
    let columns = layout.assignment_len();
    let rows = layout.multiplications;
    let c_offset = 3 * layout.capacity;
    let k_offset = 4 * layout.capacity;

    let mut column_offsets = vec![0; columns + 1];
    for (row, boundary) in column_offsets[c_offset + 1..c_offset + rows + 1]
        .iter_mut()
        .enumerate()
    {
        *boundary = row + 1;
    }
    column_offsets[c_offset + rows + 1..k_offset + 1].fill(rows);
    for (row, boundary) in column_offsets[k_offset + 1..k_offset + rows + 1]
        .iter_mut()
        .enumerate()
    {
        *boundary = rows + row + 1;
    }
    column_offsets[k_offset + rows + 1..].fill(2 * rows);

    let mut entries = Vec::with_capacity(2 * rows);
    entries.extend((0..rows).map(|row| (row, BabyBearMulCoefficient::One)));
    entries.extend((0..rows).map(|row| (row, BabyBearMulCoefficient::Modulus)));
    Ok(CscMatrix::try_from_csc(rows, column_offsets, entries)?)
}

/// Generates and prepares the compact BabyBear matrices over the Spartan/BitZ
/// field.
pub fn prepare_baby_bear_mul_relation(
    layout: BabyBearMulLayout,
    field_config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
) -> Result<PreparedConstraintMatrices<SpartanBitzField, BabyBearMulCoefficient>, BabyBearMulError> {
    let matrices = baby_bear_mul_constraint_matrices(&layout)?;
    Ok(PreparedConstraintMatrices::new(matrices, field_config)?)
}

/// Pads the exact logical assignment to eight blocks and materializes native
/// `Aw`, `Bw`, and `Cw` tables without projecting them into a field.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_baby_bear_mul_native_witness(
    witness: &BabyBearMulWitness,
) -> super::EvaluatedSpartanAssignment<u64> {
    // Allocate the final 8M table once. Extending a 5M clone to 8M can first
    // allocate and copy 5M values, then reallocate and copy them again at the
    // largest benchmark sizes.
    let mut assignment_evaluations = vec![0_u64; witness.layout.padded_assignment_len()];
    assignment_evaluations[..witness.layout.assignment_len()].copy_from_slice(witness.w());
    let assignment = DenseMultilinearExtension {
        evaluations: assignment_evaluations,
        num_vars: witness.layout.assignment_vars(),
    };

    let product_len = witness.layout.multiplications.next_power_of_two();
    let product_vars = product_len.ilog2() as usize;
    let live = witness.layout.multiplications;
    let mut aw = vec![0_u64; product_len];
    aw[..live].copy_from_slice(witness.aw());
    let mut bw = vec![0_u64; product_len];
    bw[..live].copy_from_slice(witness.bw());
    let mut cw = vec![0_u64; product_len];
    for (index, value) in cw[..live].iter_mut().enumerate() {
        *value = witness.c_values()[index] + BABY_BEAR_MODULUS * witness.k_values()[index];
    }
    let products = R1csProductMles {
        az: DenseMultilinearExtension {
            evaluations: aw,
            num_vars: product_vars,
        },
        bz: DenseMultilinearExtension {
            evaluations: bw,
            num_vars: product_vars,
        },
        cz: DenseMultilinearExtension {
            evaluations: cw,
            num_vars: product_vars,
        },
    };

    super::EvaluatedSpartanAssignment::new(assignment, products)
}

/// Converts the exact native assignment and products into any supported
/// Spartan field. All values are below `2^62`, so this conversion is exact for
/// every accepted field configuration.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_baby_bear_mul_witness<F>(
    witness: &BabyBearMulWitness,
    field_config: &F::Config,
) -> Result<(DenseMultilinearExtension<F>, R1csProductMles<F>), BabyBearMulError>
where
    F: SpartanField,
{
    F::validate_config(field_config).map_err(SpartanMatrixError::from)?;

    let field_assignment: Vec<F> = witness
        .assignment()
        .iter()
        .copied()
        .map(|value| F::from_with_cfg(value, field_config))
        .collect();

    let capacity = witness.layout.capacity;
    let live = witness.layout.multiplications;
    let field_cw = (0..live)
        .map(|index| {
            let value = witness.c_values()[index] + BABY_BEAR_MODULUS * witness.k_values()[index];
            F::from_with_cfg(value, field_config)
        })
        .collect::<Vec<_>>();
    let products = build_product_mles(
        &field_assignment[capacity..capacity + live],
        &field_assignment[2 * capacity..2 * capacity + live],
        &field_cw,
        live,
        field_config,
    )?;
    let assignment = build_assignment_mle(
        &field_assignment,
        witness.layout.assignment_len(),
        field_config,
    )?;

    Ok((assignment, products))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::piop::spartan::spartan_bitz_field_config;

    fn field(
        value: u64,
        config: &<SpartanBitzField as crate::piop::spartan::SpartanField>::Config,
    ) -> SpartanBitzField {
        SpartanBitzField::from_with_cfg(value, config)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn multiply_compact_matrix(
        matrix: &CscMatrix<Box<[BabyBearMulCoefficient]>>,
        assignment: &[u64],
    ) -> Vec<u64> {
        assert_eq!(matrix.column_count(), assignment.len());
        let mut products = vec![0_u64; matrix.row_count()];
        for (column, entries) in matrix.columns().enumerate() {
            for (row, coefficient) in entries {
                let coefficient = match coefficient {
                    BabyBearMulCoefficient::One => 1,
                    BabyBearMulCoefficient::Modulus => BABY_BEAR_MODULUS,
                };
                products[row] += coefficient * assignment[column];
            }
        }
        products
    }

    #[test]
    fn modulus_and_extreme_quotient_match_baby_bear() {
        assert_eq!(BABY_BEAR_MODULUS, (1_u64 << 31) - (1_u64 << 27) + 1);
        let maximum = BABY_BEAR_MODULUS - 1;
        let product = maximum * maximum;
        assert_eq!(product % BABY_BEAR_MODULUS, 1);
        assert_eq!(product / BABY_BEAR_MODULUS, BABY_BEAR_MODULUS - 2);
        assert!(product < (1_u64 << 62));
        assert!(BABY_BEAR_MODULUS - 2 < (1_u64 << BABY_BEAR_MUL_K_BITS));

        // BitZ proves that all four explicit assignment values are 31-bit, but
        // it deliberately does not prove that they are canonical BabyBear
        // representatives.  Bound both sides for the larger malicious-witness
        // domain as well: field equality still lifts to integer equality.
        let maximum_31_bit_value = (1_u64 << BABY_BEAR_MUL_VALUE_BITS) - 1;
        let maximum_lhs = maximum_31_bit_value * maximum_31_bit_value;
        let maximum_rhs = maximum_31_bit_value + BABY_BEAR_MODULUS * maximum_31_bit_value;
        assert!(maximum_lhs < (1_u64 << 62));
        assert!(maximum_rhs < (1_u64 << 62));
        assert!(u128::from(maximum_rhs) < crate::pcs::FQ_MOD);
    }

    #[test]
    fn layout_has_five_logical_blocks_and_eight_padded_blocks() {
        assert_eq!(BabyBearMulLayout::new(0), Err(BabyBearMulError::EmptyBatch));

        for (multiplications, capacity) in [(1, 256), (3, 256), (256, 256), (257, 512)] {
            let layout = BabyBearMulLayout::new(multiplications).unwrap();
            assert_eq!(layout.multiplications(), multiplications);
            assert_eq!(layout.capacity(), capacity);
            assert_eq!(layout.assignment_len(), 5 * capacity);
            assert_eq!(layout.padded_assignment_len(), 8 * capacity);
            assert_eq!(layout.assignment_vars(), layout.gate_vars() + 3);

            let params = layout.bitz_params();
            assert_eq!(params.word_bits, 1);
            assert_eq!(params.cells(), BABY_BEAR_MUL_BIT_SLOTS * capacity);
            for slot in 0..BABY_BEAR_MUL_BIT_SLOTS {
                for gate in 0..capacity {
                    let (b, c) = layout.bitz_cell(slot, gate).unwrap();
                    assert_eq!(params.cell_index(b, c), slot * capacity + gate);
                }
            }
            assert_eq!(layout.bitz_cell(BABY_BEAR_MUL_BIT_SLOTS, 0), None);
            assert_eq!(layout.bitz_cell(0, capacity), None);
        }
    }

    #[test]
    fn layout_rejects_overflow_boundaries_before_allocation() {
        assert_eq!(
            BabyBearMulLayout::new(usize::MAX),
            Err(BabyBearMulError::DomainTooLarge)
        );

        let largest_power_of_two = 1_usize << (usize::BITS - 1);
        assert_eq!(
            BabyBearMulLayout::new(largest_power_of_two),
            Err(BabyBearMulError::DomainTooLarge)
        );
    }

    #[test]
    fn rejection_sampler_consumes_forced_rejections_before_acceptance() {
        let modulus = BABY_BEAR_MODULUS as u32;
        let draws = [u32::MAX, modulus, modulus - 1];
        let mut consumed = 0;
        let sampled = sample_baby_bear_operand_with(|| {
            let draw = draws[consumed];
            consumed += 1;
            draw
        });

        assert_eq!(sampled, modulus - 1);
        assert_eq!(consumed, draws.len());
        assert_eq!(
            sample_baby_bear_operand_with(|| 0x8000_002a),
            42,
            "the source draw is masked to 31 bits"
        );
    }

    #[test]
    fn witness_constructors_reject_noncanonical_operands_without_redrawing() {
        let p = BABY_BEAR_MODULUS as u32;
        for invalid in [p, u32::MAX] {
            assert_eq!(
                BabyBearMulWitness::from_inputs(&[(invalid, 0)]),
                Err(BabyBearMulError::NonCanonicalOperand {
                    index: 0,
                    operand: "a",
                    value: invalid,
                })
            );
            assert_eq!(
                BabyBearMulWitness::from_inputs(&[(0, invalid)]),
                Err(BabyBearMulError::NonCanonicalOperand {
                    index: 0,
                    operand: "b",
                    value: invalid,
                })
            );
        }

        let mut calls = 0;
        let error = BabyBearMulWitness::from_fn(1, |_| {
            calls += 1;
            (p, 0)
        })
        .unwrap_err();
        assert_eq!(
            error,
            BabyBearMulError::NonCanonicalOperand {
                index: 0,
                operand: "a",
                value: p,
            }
        );
        assert_eq!(calls, 1, "from_fn must not redraw invalid inputs");
    }

    #[test]
    fn exact_witness_uses_explicit_blocks_and_zero_padding() {
        let p = BABY_BEAR_MODULUS as u32;
        let inputs = [(0, p - 1), (1, 7), (p - 1, p - 1)];
        let witness = BabyBearMulWitness::from_inputs(&inputs).unwrap();
        let capacity = witness.layout().capacity();

        assert_eq!(witness.assignment().len(), 5 * capacity);
        assert_eq!(witness.w(), witness.assignment());
        assert_eq!(witness.assignment()[0], 1);
        assert!(
            witness.assignment()[1..capacity]
                .iter()
                .all(|&value| value == 0)
        );
        assert_eq!(&witness.a_values()[..3], &[0, 1, BABY_BEAR_MODULUS - 1]);
        assert_eq!(
            &witness.b_values()[..3],
            &[BABY_BEAR_MODULUS - 1, 7, BABY_BEAR_MODULUS - 1]
        );
        assert_eq!(&witness.c_values()[..3], &[0, 7, 1]);
        assert_eq!(&witness.k_values()[..3], &[0, 0, BABY_BEAR_MODULUS - 2]);
        for values in [
            witness.a_values(),
            witness.b_values(),
            witness.c_values(),
            witness.k_values(),
        ] {
            assert!(values[3..].iter().all(|&value| value == 0));
        }
        assert_eq!(witness.aw(), &witness.a_values()[..3]);
        assert_eq!(witness.bw(), &witness.b_values()[..3]);
        assert_eq!(witness.az(), witness.aw());
        assert_eq!(witness.bz(), witness.bw());
    }

    #[test]
    fn from_fn_visits_inputs_once_and_derives_c_and_k() {
        let mut calls = Vec::new();
        let witness = BabyBearMulWitness::from_fn(5, |index| {
            calls.push(index);
            (index as u32, (index + 1) as u32)
        })
        .unwrap();

        assert_eq!(calls, (0..5).collect::<Vec<_>>());
        assert_eq!(&witness.c_values()[..5], &[0, 2, 6, 12, 20]);
        assert_eq!(&witness.k_values()[..5], &[0; 5]);
    }

    #[test]
    fn derived_values_satisfy_the_exact_integer_identity() {
        let p = BABY_BEAR_MODULUS as usize;
        let witness = BabyBearMulWitness::from_fn(1_000, |index| {
            let a = (index.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % p;
            let b = (index.wrapping_mul(2_654_435_761).wrapping_add(101_390_422)) % p;
            (a as u32, b as u32)
        })
        .unwrap();

        for row in 0..witness.layout().multiplications() {
            let product = witness.a_values()[row] * witness.b_values()[row];
            assert_eq!(
                product,
                witness.c_values()[row] + BABY_BEAR_MODULUS * witness.k_values()[row]
            );
            assert!(witness.c_values()[row] < BABY_BEAR_MODULUS);
            assert!(witness.k_values()[row] <= BABY_BEAR_MODULUS - 2);
        }
    }

    #[test]
    fn native_projection_exposes_padded_w_aw_bw_and_cw() {
        let p = BABY_BEAR_MODULUS as u32;
        let inputs = [(2, 3), (p - 1, p - 1), (11, 13)];
        let witness = BabyBearMulWitness::from_inputs(&inputs).unwrap();
        let native = project_baby_bear_mul_native_witness(&witness);

        assert_eq!(
            native.assignment().evaluations.len(),
            witness.layout().padded_assignment_len()
        );

        assert_eq!(
            native.assignment().num_vars,
            witness.layout().assignment_vars()
        );
        assert_eq!(
            &native.assignment().evaluations[..witness.assignment().len()],
            witness.assignment()
        );
        assert!(
            native.assignment().evaluations[witness.assignment().len()..]
                .iter()
                .all(|&value| value == 0)
        );
        assert_eq!(
            &native.products().az.evaluations[..3],
            &[2, BABY_BEAR_MODULUS - 1, 11]
        );
        assert_eq!(
            &native.products().bz.evaluations[..3],
            &[3, BABY_BEAR_MODULUS - 1, 13]
        );
        assert_eq!(
            &native.products().cz.evaluations[..3],
            &[6, (BABY_BEAR_MODULUS - 1).pow(2), 143]
        );
        for row in 0..inputs.len() {
            assert_eq!(
                native.products().az.evaluations[row] * native.products().bz.evaluations[row],
                native.products().cz.evaluations[row]
            );
        }
        assert_eq!(native.products().az.evaluations[3], 0);
        assert_eq!(native.products().bz.evaluations[3], 0);
        assert_eq!(native.products().cz.evaluations[3], 0);
    }

    #[test]
    fn compact_matrices_independently_produce_native_and_field_products() {
        let config = spartan_bitz_field_config();
        let p = BABY_BEAR_MODULUS as u32;
        let witness =
            BabyBearMulWitness::from_inputs(&[(0, p - 1), (2, 3), (p - 1, p - 1), (17, 19)])
                .unwrap();
        let matrices = baby_bear_mul_constraint_matrices(witness.layout()).unwrap();
        let independently_computed = [
            multiply_compact_matrix(matrices.a(), witness.w()),
            multiply_compact_matrix(matrices.b(), witness.w()),
            multiply_compact_matrix(matrices.c(), witness.w()),
        ];
        let native = project_baby_bear_mul_native_witness(&witness);
        let (_, projected) =
            project_baby_bear_mul_witness::<SpartanBitzField>(&witness, &config).unwrap();
        let live = witness.layout().multiplications();
        let native_products = [
            &native.products().az,
            &native.products().bz,
            &native.products().cz,
        ];
        let projected_products = [&projected.az, &projected.bz, &projected.cz];

        for ((expected, native_product), projected_product) in independently_computed
            .iter()
            .zip(native_products)
            .zip(projected_products)
        {
            assert_eq!(expected.as_slice(), &native_product.evaluations[..live]);
            for (projected_value, expected_value) in
                projected_product.evaluations[..live].iter().zip(expected)
            {
                assert_eq!(projected_value, &field(*expected_value, &config));
            }
        }

        for row in 0..live {
            assert_eq!(
                independently_computed[0][row] * independently_computed[1][row],
                independently_computed[2][row]
            );
        }
    }

    #[test]
    fn matrices_have_exact_csc_entries_at_padded_and_full_boundaries() {
        for multiplications in [3, MIN_CAPACITY] {
            let layout = BabyBearMulLayout::new(multiplications).unwrap();
            let matrices = baby_bear_mul_constraint_matrices(&layout).unwrap();
            let capacity = layout.capacity();

            assert_eq!(matrices.row_count(), multiplications);
            assert_eq!(matrices.column_count(), 5 * capacity);
            assert_eq!(matrices.a().nnz(), multiplications);
            assert_eq!(matrices.b().nnz(), multiplications);
            assert_eq!(matrices.c().nnz(), 2 * multiplications);

            for column in 0..layout.assignment_len() {
                let expected_a = if (capacity..capacity + multiplications).contains(&column) {
                    vec![(column - capacity, BabyBearMulCoefficient::One)]
                } else {
                    Vec::new()
                };
                let expected_b = if (2 * capacity..2 * capacity + multiplications).contains(&column)
                {
                    vec![(column - 2 * capacity, BabyBearMulCoefficient::One)]
                } else {
                    Vec::new()
                };
                let expected_c = if (3 * capacity..3 * capacity + multiplications).contains(&column)
                {
                    vec![(column - 3 * capacity, BabyBearMulCoefficient::One)]
                } else if (4 * capacity..4 * capacity + multiplications).contains(&column) {
                    vec![(column - 4 * capacity, BabyBearMulCoefficient::Modulus)]
                } else {
                    Vec::new()
                };

                let actual_a = matrices
                    .a()
                    .column(column)
                    .unwrap()
                    .into_iter()
                    .map(|(row, coefficient)| (row, *coefficient))
                    .collect::<Vec<_>>();
                let actual_b = matrices
                    .b()
                    .column(column)
                    .unwrap()
                    .into_iter()
                    .map(|(row, coefficient)| (row, *coefficient))
                    .collect::<Vec<_>>();
                let actual_c = matrices
                    .c()
                    .column(column)
                    .unwrap()
                    .into_iter()
                    .map(|(row, coefficient)| (row, *coefficient))
                    .collect::<Vec<_>>();
                assert_eq!(actual_a, expected_a);
                assert_eq!(actual_b, expected_b);
                assert_eq!(actual_c, expected_c);
            }

            assert!(matrices.a().column(layout.assignment_len()).is_none());
            assert!(matrices.b().column(layout.assignment_len()).is_none());
            assert!(matrices.c().column(layout.assignment_len()).is_none());
        }
    }

    #[test]
    fn packed_rows_reconstruct_values_and_leave_four_slots_zero() {
        let p = BABY_BEAR_MODULUS as u32;
        let inputs = [(0x4000_0001, 3), (p - 1, p - 1), (17, 19)];
        let witness = BabyBearMulWitness::from_inputs(&inputs).unwrap();
        let layout = witness.layout();
        let params = layout.bitz_params();
        let rows = witness.bitz_bit_rows();

        assert_eq!(rows.len(), params.cols());
        assert!(rows.iter().all(|row| row.len() == params.rows() / 64));

        for gate in 0..layout.capacity() {
            for (slot_offset, value) in [
                (BABY_BEAR_MUL_A_SLOT_START, witness.a_values()[gate]),
                (BABY_BEAR_MUL_B_SLOT_START, witness.b_values()[gate]),
                (BABY_BEAR_MUL_C_SLOT_START, witness.c_values()[gate]),
                (BABY_BEAR_MUL_K_SLOT_START, witness.k_values()[gate]),
            ] {
                for bit in 0..BABY_BEAR_MUL_VALUE_BITS {
                    let (b, c) = layout.bitz_cell(slot_offset + bit, gate).unwrap();
                    let committed_bit = (rows[c][b / 64] >> (b % 64)) & 1;
                    assert_eq!(committed_bit, (value >> bit) & 1);
                }
            }

            for slot in BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS..BABY_BEAR_MUL_BIT_SLOTS {
                let (b, c) = layout.bitz_cell(slot, gate).unwrap();
                assert_eq!((rows[c][b / 64] >> (b % 64)) & 1, 0);
            }
        }
    }

    fn random_witness(multiplications: usize, seed: u64) -> BabyBearMulWitness {
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        BabyBearMulWitness::from_fn(multiplications, |_| {
            (
                sample_baby_bear_operand_with(|| rng.random::<u32>()),
                sample_baby_bear_operand_with(|| rng.random::<u32>()),
            )
        })
        .unwrap()
    }

    #[test]
    fn transposed_bit_rows_match_the_bitwise_packing() {
        // gate_vars 10 (32-gate lanes: bitwise path), 11 (one word per
        // lane), 13 (two words), and 15 (the smallest production layout);
        // live counts off the power of two exercise the zero padding.
        for (multiplications, seed) in [
            (700, 1),
            (1500, 2),
            (5000, 3),
            (1 << 15, 4),
            ((1 << 15) + 37, 5),
        ] {
            let witness = random_witness(multiplications, seed);
            let params = witness.layout().bitz_params();
            let mut expected = vec![vec![0_u64; params.rows() / 64]; params.cols()];
            witness.write_bit_rows_bitwise(&mut expected);
            assert_eq!(
                witness.bitz_bit_rows(),
                expected,
                "multiplications={multiplications}"
            );
        }
    }

    #[test]
    fn pack_gate_slots_places_each_value_in_its_lane() {
        let p = BABY_BEAR_MODULUS;
        let (lo, hi) = pack_gate_slots(p - 1, 1, 0x4000_0001, p - 2);
        let packed = u128::from(lo) | (u128::from(hi) << 64);
        let mask = (1_u128 << BABY_BEAR_MUL_VALUE_BITS) - 1;
        assert_eq!(
            (packed >> BABY_BEAR_MUL_A_SLOT_START) & mask,
            u128::from(p - 1)
        );
        assert_eq!((packed >> BABY_BEAR_MUL_B_SLOT_START) & mask, 1);
        assert_eq!((packed >> BABY_BEAR_MUL_C_SLOT_START) & mask, 0x4000_0001);
        assert_eq!(
            (packed >> BABY_BEAR_MUL_K_SLOT_START) & mask,
            u128::from(p - 2)
        );
        assert_eq!(packed >> BABY_BEAR_MUL_SEMANTIC_BIT_SLOTS, 0);
        // Bits above the committed width are dropped, never smeared into
        // the neighbouring lane.
        assert_eq!(pack_gate_slots(1 << 31, 0, 0, 0), (0, 0));
    }

    #[test]
    fn compact_coefficients_borrow_fixed_canonical_encodings() {
        let config = spartan_bitz_field_config();
        let field_one_encoding = field(1, &config).canonical_element_encoding(&config);
        let expected_modulus_encoding =
            field(BABY_BEAR_MODULUS, &config).canonical_element_encoding(&config);
        let one_encoding =
            BabyBearMulCoefficient::One.canonical_field_encoding(&config, &field_one_encoding);
        let modulus_encoding =
            BabyBearMulCoefficient::Modulus.canonical_field_encoding(&config, &field_one_encoding);

        assert!(matches!(&one_encoding, Cow::Borrowed(_)));
        assert!(matches!(&modulus_encoding, Cow::Borrowed(_)));
        assert_eq!(one_encoding.as_ref(), field_one_encoding.as_slice());
        assert_eq!(
            modulus_encoding.as_ref(),
            expected_modulus_encoding.as_slice()
        );
        assert_eq!(
            modulus_encoding.as_ref(),
            BABY_BEAR_MODULUS_FIELD_ENCODING.as_slice()
        );
    }

    #[test]
    fn compact_coefficients_match_field_matrix_digest_and_evaluations() {
        let config = spartan_bitz_field_config();
        let layout = BabyBearMulLayout::new(3).unwrap();
        let capacity = layout.capacity();
        let one = field(1, &config);
        let modulus = field(BABY_BEAR_MODULUS, &config);

        let a_rows = (0..layout.multiplications())
            .map(|row| vec![(capacity + row, one.clone())])
            .collect();
        let b_rows = (0..layout.multiplications())
            .map(|row| vec![(2 * capacity + row, one.clone())])
            .collect();
        let c_rows = (0..layout.multiplications())
            .map(|row| {
                vec![
                    (3 * capacity + row, one.clone()),
                    (4 * capacity + row, modulus.clone()),
                ]
            })
            .collect();
        let field_matrices = ConstraintMatrices::new(
            CscMatrix::try_from_rows(layout.assignment_len(), a_rows).unwrap(),
            CscMatrix::try_from_rows(layout.assignment_len(), b_rows).unwrap(),
            CscMatrix::try_from_rows(layout.assignment_len(), c_rows).unwrap(),
        )
        .unwrap();
        let field_prepared =
            PreparedConstraintMatrices::<SpartanBitzField>::new(field_matrices, &config).unwrap();
        let compact_prepared = prepare_baby_bear_mul_relation(layout, &config).unwrap();

        assert_eq!(compact_prepared.digest(), field_prepared.digest());
        assert_eq!(compact_prepared.num_column_vars(), layout.assignment_vars());

        let row_point = (0..compact_prepared.num_row_vars())
            .map(|coordinate| field(coordinate as u64 + 2, &config))
            .collect::<Vec<_>>();
        let column_point = (0..compact_prepared.num_column_vars())
            .map(|coordinate| field(coordinate as u64 + 7, &config))
            .collect::<Vec<_>>();
        let rho = field(19, &config);
        let compact_bound =
            crate::piop::spartan::matrix::eq_table_prover(&row_point, compact_prepared.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| compact_prepared.binding(&rho).bind_rows(&weights))
                .unwrap();
        let field_bound =
            crate::piop::spartan::matrix::eq_table_prover(&row_point, field_prepared.config())
                .map_err(crate::sumcheck::SumcheckError::from)
                .and_then(|weights| field_prepared.binding(&rho).bind_rows(&weights))
                .unwrap();
        assert_eq!(compact_bound, field_bound);
        assert_eq!(
            compact_prepared
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap(),
            field_prepared
                .structured()
                .evaluate_equality(&row_point, &rho, &column_point)
                .unwrap()
        );
    }

    #[test]
    fn field_projection_preserves_assignment_and_integer_products() {
        let config = spartan_bitz_field_config();
        let p = BABY_BEAR_MODULUS as u32;
        let inputs = [(2, 3), (p - 1, p - 1), (11, 13)];
        let witness = BabyBearMulWitness::from_inputs(&inputs).unwrap();
        let relation = prepare_baby_bear_mul_relation(*witness.layout(), &config).unwrap();
        let (assignment, products) =
            project_baby_bear_mul_witness::<SpartanBitzField>(&witness, &config).unwrap();

        assert_eq!(
            assignment.evaluations.len(),
            witness.layout().padded_assignment_len()
        );
        assert_eq!(
            assignment.evaluations.len(),
            1 << relation.num_column_vars()
        );
        assert_eq!(assignment.evaluations[0], field(1, &config));
        for (projected, native) in assignment.evaluations[..witness.w().len()]
            .iter()
            .zip(witness.w())
        {
            assert_eq!(projected, &field(*native, &config));
        }
        assert!(
            assignment.evaluations[witness.w().len()..]
                .iter()
                .all(|value| value == &field(0, &config))
        );
        for row in 0..inputs.len() {
            let mut product = products.az.evaluations[row].clone();
            product = config.mul(&(product), &(&products.bz.evaluations[row]));
            assert_eq!(product, products.cz.evaluations[row]);

            let native_cw = witness.c_values()[row] + BABY_BEAR_MODULUS * witness.k_values()[row];
            assert_eq!(
                products.az.evaluations[row],
                field(witness.a_values()[row], &config)
            );
            assert_eq!(
                products.bz.evaluations[row],
                field(witness.b_values()[row], &config)
            );
            assert_eq!(products.cz.evaluations[row], field(native_cw, &config));
        }
        for row in inputs.len()..products.az.evaluations.len() {
            assert_eq!(products.az.evaluations[row], field(0, &config));
            assert_eq!(products.bz.evaluations[row], field(0, &config));
            assert_eq!(products.cz.evaluations[row], field(0, &config));
        }
    }
}
