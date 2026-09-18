//! Batched word-AND gates as a CM instance with an F₂-VIRTUAL block —
//! the paper's `\Relation_{CM}` / `r:CM_is_NP_complete` gadget
//! (`s:to_f2_virtual`) wired end to end through Spartan and the virtual
//! BitZ opening.
//!
//! Per gate: 32-bit words `x`, `y`, `z`, `w` with the single LINEAR
//! constraint
//!
//! ```text
//! x + y − w − 2·z = 0        over 𝔽_q (exact over ℤ: all values < 2^33)
//! ```
//!
//! Since `x + y = (x⊕y) + 2·(x∧y)` holds bitwise-exactly, the constraint
//! FORCES `z = x∧y` as soon as `w = x⊕y` bit-for-bit — and that identity
//! is not proven but imposed STRUCTURALLY: the committed vector `f`
//! carries only the `x`/`y`/`z` bits, while the R1CS's bit grid
//! `h = M·f` derives every `w` bit as the XOR of the matching `x` and
//! `y` bits ([`cm_and_map`]). The R1CS is `A = B = 0` with one `C` row
//! per gate — the pure CM shape (linear constraints over ℤ composed
//! with `F₂`-linear derivation), which is NP-complete.
//!
//! Layouts. The derived grid `h` has 128 bit slots per gate
//! (`x@0, y@32, z@64, w@96`); the committed grid `f` has the same shape
//! with slots `[96, 128)` structurally dead (never referenced by `M` or
//! the relation; zero for the honest prover, harmless if not). The
//! Spartan assignment is `[const | x | y | z | w]` — five logical blocks
//! padded to eight, so the terminal claim carries `gate_vars + 3`
//! coordinates. Bitification is the adjoint of the four 32-bit
//! reconstructions, exactly as in the direct `u32_mul` bridge but over
//! a 3-variable block selector.

use crate::ligerito_flock::IntEvalRsLigVirtProof;
use crate::piop::spartan::protocol::Proof;
use crate::piop::spartan::protocol::ProtocolError;

use crate::piop::spartan::SpartanField as _;
use circuit::linear_map::CscMatrix;
use field::RingOps;
#[cfg(test)]
use field::{Fp, Uint};

use flock_core::pcs::{
    commit::Commitment,
    ligerito::{ProverConfig as LigProverConfig, VerifierConfig as LigVerifierConfig},
};
use thiserror::Error;

use {
    crate::{
        f2map::cell_count,
        ligerito::{LOG_PACKING, packed_vars},
        ligerito_flock::{
            FlockCommitHint, LigeritoSelection, ModQOpeningKind, commit_rs_ligerito_rows,
            sha_lig_configs,
        },
        pcs::IntegerMatrixLayout,
        transcript::traits::Transcript,
    },
    circuit::linear_map::binary::{PreparedVirtualMap, PreparedVirtualMapError},
};

use super::{
    EvaluatedSpartanAssignment, SpartanField,
    bitz::{MIN_PRODUCTION_GATE_VARS, SpartanBitzField},
    matrix::{
        ConstraintMatrices, PreparedConstraintMatrices, SpartanMatrixError, build_assignment_mle,
        build_product_mles,
    },
    profile::{IopInstanceFacts, IopSecurityParams, Lambda100},
    protocol::{
        self, BindingHasher, BlockTable, ClaimFrame, Domains, FieldConfig, Kernel, MatrixSource,
        Opener, PiopWitness, PreparedRelation, PreparedRelationPrefix, PrimeStrategy, RelationSpec,
        ScaleSide, Schedule, SlotRange,
    },
};

/// Word width of every gate operand.
pub const CM_AND_WORD_BITS: usize = 32;
/// First derived bit slot of the left operand.
pub const CM_AND_X_SLOT: usize = 0;
/// First derived bit slot of the right operand.
pub const CM_AND_Y_SLOT: usize = 32;
/// First derived bit slot of the AND output.
pub const CM_AND_Z_SLOT: usize = 64;
/// First derived bit slot of the VIRTUAL XOR word `w = x ⊕ y`.
pub const CM_AND_W_SLOT: usize = 96;
/// Bit slots per gate in the derived grid `h` (a power of two).
pub const CM_AND_H_SLOTS: usize = 128;
/// Live committed bit slots per gate in `f` (`x`, `y`, `z`); the grid is
/// padded to [`CM_AND_H_SLOTS`] with structurally dead cells.
pub const CM_AND_F_LIVE_SLOTS: usize = 96;

/// Logical assignment blocks: `[const | x | y | z | w]`.
const ASSIGNMENT_BLOCKS: usize = 5;
/// Selector coordinates of the padded (8-block) assignment domain.
const SELECTOR_VARS: usize = 3;
/// Same floor as the direct bridge: keeps the compact row packing in its
/// supported geometry even for small test fixtures.
const MIN_CAPACITY: usize = 1 << 8;

const CM_ASSIGNMENT_BINDING_DOMAIN: &[u8] = b"bitz/cm-bitz/assignment/v1";
const CM_OPENING_CLAIM_DOMAIN: &[u8] = b"bitz/cm-bitz/opening/v1";

/// Failures while constructing the CM-AND relation or its witness.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CmAndError {
    /// A gate batch must contain at least one live gate.
    #[error("a CM-AND batch must not be empty")]
    EmptyBatch,

    /// The padded assignment or bit domain does not fit in `usize`.
    #[error("the CM-AND domain is too large")]
    DomainTooLarge,

    #[error("invalid Ligerito configuration: {0}")]
    LigeritoConfig(String),

    /// The generated relation or projected witness is malformed.
    #[error(transparent)]
    SpartanMatrix(#[from] SpartanMatrixError),

    /// The structural derivation map could not be prepared.
    #[error(transparent)]
    VirtualMap(#[from] PreparedVirtualMapError),
}

impl From<CmAndError> for ProtocolError {
    fn from(error: CmAndError) -> Self {
        Self::relation(error)
    }
}

/// Shared shape of the CM-AND assignment, the derived grid `h`, and the
/// committed grid `f`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CmAndLayout {
    gates: usize,
    capacity: usize,
    gate_vars: usize,
}

impl CmAndLayout {
    /// Creates a layout for `gates` live AND gates. The gate capacity is
    /// `max(256, gates).next_power_of_two()`.
    pub fn new(gates: usize) -> Result<Self, CmAndError> {
        if gates == 0 {
            return Err(CmAndError::EmptyBatch);
        }
        let capacity = gates
            .max(MIN_CAPACITY)
            .checked_next_power_of_two()
            .ok_or(CmAndError::DomainTooLarge)?;
        capacity
            .checked_mul(CM_AND_H_SLOTS)
            .and_then(|cells| cells.checked_mul(2))
            .ok_or(CmAndError::DomainTooLarge)?;
        let gate_vars = capacity.trailing_zeros() as usize;
        Ok(Self {
            gates,
            capacity,
            gate_vars,
        })
    }

    /// Number of live gates.
    pub const fn gates(&self) -> usize {
        self.gates
    }

    /// Power-of-two gate capacity, including zero-padded gates.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of variables selecting a gate.
    pub const fn gate_vars(&self) -> usize {
        self.gate_vars
    }

    /// Logical integer assignment length: five blocks of `capacity`.
    pub const fn assignment_len(&self) -> usize {
        ASSIGNMENT_BLOCKS * self.capacity
    }

    /// The shared BitZ shape of BOTH grids (`W = 1`,
    /// `s = floor(gate_vars/2)`, `t = 7 + gate_vars − s`): `h` and `f`
    /// use the same geometry; they differ only in which slots are live.
    pub const fn bitz_params(&self) -> IntegerMatrixLayout {
        let s = self.gate_vars / 2;
        IntegerMatrixLayout {
            row_vars: 7 + self.gate_vars - s,
            col_vars: s,
            word_bits: 1,
        }
    }

    /// Maps `(bit_slot, gate)` to the row-major cell `(b, c)` — identical
    /// convention for `h` and `f` (same shape).
    pub const fn cell(&self, bit_slot: usize, gate: usize) -> Option<(usize, usize)> {
        if bit_slot >= CM_AND_H_SLOTS || gate >= self.capacity {
            return None;
        }
        let s = self.gate_vars / 2;
        let column_mask = (1usize << s) - 1;
        let b = (bit_slot << (self.gate_vars - s)) | (gate >> s);
        let c = gate & column_mask;
        Some((b, c))
    }

    /// Flat cell index (`(c << t) | b`, the pack/point order shared with
    /// [`PreparedVirtualMap`]) of `(bit_slot, gate)`.
    const fn flat_cell(&self, bit_slot: usize, gate: usize) -> usize {
        let s = self.gate_vars / 2;
        let t = 7 + self.gate_vars - s;
        let b = (bit_slot << (self.gate_vars - s)) | (gate >> s);
        let c = gate & ((1usize << s) - 1);
        (c << t) | b
    }
}

/// Builds the structural derivation map directly in canonical CSC form.
/// Source columns for x/y bits feed both their identity row and matching w
/// row; z bits feed their identity row; padded source slots are empty.
#[allow(clippy::arithmetic_side_effects)]
pub fn cm_and_map(layout: &CmAndLayout) -> Result<PreparedVirtualMap, CmAndError> {
    let p = layout.bitz_params();
    let cells = cell_count(&p);
    let nnz = layout
        .capacity
        .checked_mul(CM_AND_F_LIVE_SLOTS + 2 * CM_AND_WORD_BITS)
        .ok_or(CmAndError::DomainTooLarge)?;

    let s = layout.gate_vars / 2;
    let t = 7 + layout.gate_vars - s;
    let high_bits = layout.gate_vars - s;
    let row_mask = (1usize << t) - 1;
    let gate_high_mask = (1usize << high_bits) - 1;

    let mut column_offsets = Vec::with_capacity(cells + 1);
    let mut row_indices = Vec::with_capacity(nnz);
    column_offsets.push(0);
    for source in 0..cells {
        let b = source & row_mask;
        let c = source >> t;
        let slot = b >> high_bits;
        let gate = ((b & gate_high_mask) << s) | c;
        match slot {
            CM_AND_X_SLOT..CM_AND_Y_SLOT => {
                let bit = slot - CM_AND_X_SLOT;
                row_indices.push(layout.flat_cell(slot, gate));
                row_indices.push(layout.flat_cell(CM_AND_W_SLOT + bit, gate));
            }
            CM_AND_Y_SLOT..CM_AND_Z_SLOT => {
                let bit = slot - CM_AND_Y_SLOT;
                row_indices.push(layout.flat_cell(slot, gate));
                row_indices.push(layout.flat_cell(CM_AND_W_SLOT + bit, gate));
            }
            CM_AND_Z_SLOT..CM_AND_W_SLOT => {
                row_indices.push(layout.flat_cell(slot, gate));
            }
            _ => {}
        }
        column_offsets.push(row_indices.len());
    }
    debug_assert_eq!(row_indices.len(), nnz);

    let matrix = CscMatrix::try_from_binary_csc(cells, column_offsets, row_indices)
        .map_err(SpartanMatrixError::from)?;
    Ok(PreparedVirtualMap::from_implicit(matrix)?)
}
/// Exact integer assignment for a batch of AND gates:
/// `z = [const | x | y | z | w]`, only `z[0]` nonzero in the constant
/// block, unused gates zero everywhere.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CmAndWitness {
    layout: CmAndLayout,
    assignment: Box<[u64]>,
}

impl CmAndWitness {
    /// Honest witness from operand pairs: `z = x ∧ y`, `w = x ⊕ y`.
    pub fn from_inputs(inputs: &[(u32, u32)]) -> Result<Self, CmAndError> {
        Self::from_fn(inputs.len(), |i| inputs[i])
    }

    /// Honest witness without retaining an input buffer.
    pub fn from_fn(
        gates: usize,
        mut input: impl FnMut(usize) -> (u32, u32),
    ) -> Result<Self, CmAndError> {
        Self::from_gate_values(gates, |i| {
            let (x, y) = input(i);
            (x, y, x & y, x ^ y)
        })
    }

    /// Arbitrary `(x, y, z, w)` gate values — no honesty is imposed, so
    /// adversarial fixtures (false relations, wrong XOR words) can be
    /// built through the same layout code.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_gate_values(
        gates: usize,
        mut values: impl FnMut(usize) -> (u32, u32, u32, u32),
    ) -> Result<Self, CmAndError> {
        let layout = CmAndLayout::new(gates)?;
        let capacity = layout.capacity;
        let mut assignment = vec![0u64; layout.assignment_len()];
        assignment[0] = 1;
        for gate in 0..gates {
            let (x, y, z, w) = values(gate);
            assignment[capacity + gate] = u64::from(x);
            assignment[2 * capacity + gate] = u64::from(y);
            assignment[3 * capacity + gate] = u64::from(z);
            assignment[4 * capacity + gate] = u64::from(w);
        }
        Ok(Self {
            layout,
            assignment: assignment.into_boxed_slice(),
        })
    }

    /// Shape shared by the assignment and both bit grids.
    pub const fn layout(&self) -> &CmAndLayout {
        &self.layout
    }

    /// Complete block-aligned integer assignment.
    pub fn assignment(&self) -> &[u64] {
        &self.assignment
    }

    fn block(&self, index: usize) -> &[u64] {
        let capacity = self.layout.capacity;
        &self.assignment[index * capacity..(index + 1) * capacity]
    }

    /// Builds the compact committed rows of `f` (`x`/`y`/`z` bits; slots
    /// `[96, 128)` zero) in the commit layout of the shared shape.
    pub fn f_bit_rows(&self) -> Vec<Vec<u64>> {
        self.bit_rows(&[(CM_AND_X_SLOT, 1), (CM_AND_Y_SLOT, 2), (CM_AND_Z_SLOT, 3)])
    }

    /// Builds synthesized derived rows of `h` (`x`/`y`/`z`/`w` bits).
    pub fn h_bit_rows(&self) -> Vec<Vec<u64>> {
        self.bit_rows(&[
            (CM_AND_X_SLOT, 1),
            (CM_AND_Y_SLOT, 2),
            (CM_AND_Z_SLOT, 3),
            (CM_AND_W_SLOT, 4),
        ])
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn bit_rows(&self, blocks: &[(usize, usize)]) -> Vec<Vec<u64>> {
        let p = self.layout.bitz_params();
        let words_per_row = p.rows() / u64::BITS as usize;
        let mut rows = vec![vec![0u64; words_per_row]; p.cols()];
        for gate in 0..self.layout.gates {
            for &(slot_start, block_index) in blocks {
                let value = self.block(block_index)[gate];
                for bit in 0..CM_AND_WORD_BITS {
                    if value & (1u64 << bit) != 0 {
                        let (b, c) = self
                            .layout
                            .cell(slot_start + bit, gate)
                            .expect("witness bit coordinates are in bounds");
                        rows[c][b / u64::BITS as usize] |= 1u64 << (b % u64::BITS as usize);
                    }
                }
            }
        }
        rows
    }
}

/// The CM-AND relation as the shared protocol sees it: the layout, the
/// runtime field the field-valued matrices are prepared at, and the
/// canonical derivation map with its digest.
pub struct CmAndSpec {
    layout: CmAndLayout,
    map: PreparedVirtualMap,
    field_config: FieldConfig,
}

impl RelationSpec for CmAndSpec {
    type Coefficient = SpartanBitzField;
    type Witness = CmAndWitness;
    type Map = PreparedVirtualMap;

    fn domains(&self) -> &'static Domains {
        &CM_AND_DOMAINS
    }

    /// No prime draw and no grinding; the Spartan scale rides the clear
    /// column side.
    fn schedule(&self) -> Schedule {
        Schedule {
            policy_bind: true,
            ood_round: true,
            piop_grinding: false,
            scale_side: ScaleSide::Columns,
            strategy: PrimeStrategy::Single,
        }
    }

    fn committed_layout(&self) -> IntegerMatrixLayout {
        self.layout.bitz_params()
    }

    fn gate_vars(&self) -> usize {
        self.layout.gate_vars()
    }

    fn instance_facts(&self) -> IopInstanceFacts {
        let p = self.layout.bitz_params();
        IopInstanceFacts {
            defect_log2_bound: 80,
            lift_arity_log2: p.row_vars as u32,
            opening_t: p.row_vars as u32,
            opening_word_bits: p.word_bits as u32,
            direct_opening: false,
            tau_arity: (self.layout.gate_vars() + SELECTOR_VARS) as u32,
            piop_degree: 3,
            step50_magnitude_log2: 0,
        }
    }

    /// `A = B = 0`, one `C` row `x + y − 2z − w = 0` per live gate, at the
    /// fixed runtime field.
    fn matrices(&self) -> Result<MatrixSource<SpartanBitzField>, ProtocolError> {
        let capacity = self.layout.capacity;
        let columns = self.layout.assignment_len();
        let live = self.layout.gates;
        let field_config = &self.field_config;

        let one = SpartanBitzField::one_with_cfg(field_config);
        let mut minus_one = SpartanBitzField::zero_with_cfg(field_config);
        minus_one = field_config.sub(&(minus_one), &(&one));
        let mut minus_two = minus_one.clone();
        minus_two = field_config.sub(&(minus_two), &(&one));

        let empty = CscMatrix::try_from_rows(columns, vec![Vec::new(); live])
            .map_err(SpartanMatrixError::from)
            .map_err(CmAndError::from)?;
        let c_rows: Vec<Vec<(usize, SpartanBitzField)>> = (0..live)
            .map(|i| {
                vec![
                    (capacity + i, one.clone()),
                    (2 * capacity + i, one.clone()),
                    (3 * capacity + i, minus_two.clone()),
                    (4 * capacity + i, minus_one.clone()),
                ]
            })
            .collect();
        let c = CscMatrix::try_from_rows(columns, c_rows)
            .map_err(SpartanMatrixError::from)
            .map_err(CmAndError::from)?;
        let matrices = PreparedConstraintMatrices::new(
            ConstraintMatrices::new(empty.clone(), empty, c).map_err(CmAndError::from)?,
            field_config,
        )
        .map_err(CmAndError::from)?;
        Ok(MatrixSource::Fixed(matrices))
    }

    fn validate_geometry(&self) -> Result<(), ProtocolError> {
        validate_cm_layout_geometry(&self.layout)?;
        let p = self.layout.bitz_params();
        if self.map.rows() != cell_count(&p) || self.map.cols() != cell_count(&p) {
            return Err(ProtocolError::InvalidBitzParameters);
        }
        Ok(())
    }

    /// Three block-selector coordinates: `const = 0, x = 1, y = 2, z = 3,
    /// w = 4`; blocks 5–7 are the zero padding of the assignment domain.
    fn block_table(&self) -> BlockTable {
        let block = |bit_slot_start: usize| {
            Some(SlotRange {
                bit_slot_start,
                bit_count: CM_AND_WORD_BITS,
            })
        };
        BlockTable::new(
            SELECTOR_VARS,
            vec![
                None,
                block(CM_AND_X_SLOT),
                block(CM_AND_Y_SLOT),
                block(CM_AND_Z_SLOT),
                block(CM_AND_W_SLOT),
                None,
                None,
                None,
            ],
        )
        .expect("the CM-AND block table is complete")
    }

    fn kernel(&self) -> Kernel {
        Kernel::Plain
    }

    fn check_witness(&self, witness: &CmAndWitness) -> Result<(), ProtocolError> {
        if witness.layout() != &self.layout {
            return Err(ProtocolError::RelationWitnessLayoutMismatch);
        }
        Ok(())
    }

    fn assignment_binding(
        &self,
        commitment: &Commitment,
        _security: &IopSecurityParams,
        _ligerito: &LigProverConfig,
    ) -> Result<[u8; 32], ProtocolError> {
        let p = self.layout.bitz_params();
        let mut hasher = BindingHasher::new();
        hasher
            .bytes(CM_ASSIGNMENT_BINDING_DOMAIN)
            .bytes(&commitment.root);
        hasher.commitment_params(&commitment.params)?;
        hasher.u128_le(self.modulus());
        hasher.usizes(&[
            self.layout.gates(),
            self.layout.capacity(),
            self.layout.gate_vars(),
            p.row_vars,
            p.col_vars,
            p.word_bits,
        ])?;
        hasher.bytes(&self.map.digest());
        Ok(hasher.finalize())
    }

    /// The fixed runtime field: nothing is drawn or absorbed.
    fn runtime_prime<T: Transcript>(
        &self,
        _transcript: &mut T,
        _security: &IopSecurityParams,
    ) -> Result<field::FpCtx<2>, ProtocolError> {
        protocol::runtime_field(self.modulus())
    }

    fn piop_witness<'w>(
        &self,
        witness: &'w CmAndWitness,
        _config: &FieldConfig,
    ) -> Result<PiopWitness<'w>, ProtocolError> {
        Ok(PiopWitness::CmAnd(witness))
    }

    fn map(&self) -> Option<&PreparedVirtualMap> {
        Some(&self.map)
    }

    /// The virtual opening runs against the derived grid `h = M·f`.
    fn derived_rows(&self, witness: &CmAndWitness) -> Option<Vec<Vec<u64>>> {
        Some(witness.h_bit_rows())
    }

    fn claim_digest(&self, frame: ClaimFrame<'_>) -> Result<[u8; 32], ProtocolError> {
        let mut hasher = BindingHasher::new();
        hasher
            .bytes(CM_OPENING_CLAIM_DOMAIN)
            .bytes(frame.binding)
            .bytes(frame.matrices_digest);
        hasher.usize(frame.terminal_claim.point().len())?;
        for coordinate in frame.terminal_claim.point() {
            hasher.element(coordinate, frame.field);
        }
        hasher.element(frame.terminal_claim.scale(), frame.field);
        hasher.element(frame.terminal_claim.value(), frame.field);
        hasher.usize(frame.row_weights.len())?;
        for weight in frame.row_weights {
            hasher.u128_le(*weight);
        }
        hasher.usize(frame.col_weights.len())?;
        for weight in frame.col_weights {
            hasher.u128_le(*weight);
        }
        hasher.u128_le(frame.opening.claimed);
        Ok(hasher.finalize())
    }
}

impl CmAndSpec {
    /// The fixed runtime modulus the matrices were prepared at.
    fn modulus(&self) -> u128 {
        let encoding = SpartanBitzField::canonical_modulus_encoding(&self.field_config);
        let mut bytes = [0_u8; 16];
        bytes.copy_from_slice(&encoding[..16]);
        u128::from_le_bytes(bytes)
    }
}

static CM_AND_DOMAINS: Domains = Domains {
    statement_tag: b"cm/early-ood/statement/v2",
    prime_sampling: b"",
    initial_grinding: b"",
    piop_grinding: b"",
    terminal_grinding: b"",
    bitified_claim: b"",
    opening: ModQOpeningKind::U32Mul,
    claim_tag: CM_OPENING_CLAIM_DOMAIN,
    reduction_grinding: b"",
    reduction_prime: b"",
    scopes: crate::protocol_scopes!("cm-bitz"),
};

enum CmPrepared {
    /// Production: a validator-gated opener resolved at the 100-bit target.
    Production(PreparedRelation<CmAndSpec>),
    /// Algebra-only fixtures below the audited size floor: no opener, no
    /// security claim; provable only with an explicit configuration.
    Unaudited(PreparedRelationPrefix<CmAndSpec>),
}

/// Prepared CM-AND statement: the layout, the field-valued Spartan
/// matrices (`A = B = 0`, one `C` row `x + y − 2z − w = 0` per live
/// gate), and the canonical derivation map with its digest.
pub struct PreparedCmAndRelation {
    prepared: CmPrepared,
}

impl PreparedCmAndRelation {
    fn prefix(&self) -> &PreparedRelationPrefix<CmAndSpec> {
        match &self.prepared {
            CmPrepared::Production(relation) => relation.prefix(),
            CmPrepared::Unaudited(prefix) => prefix,
        }
    }

    fn production(&self) -> Result<&PreparedRelation<CmAndSpec>, ProtocolError> {
        match &self.prepared {
            CmPrepared::Production(relation) => Ok(relation),
            CmPrepared::Unaudited(_) => Err(ProtocolError::UnauditedBitzParameters),
        }
    }

    /// Re-resolves the opener (production layouts only).
    pub fn with_ligerito(self, selection: LigeritoSelection) -> Result<Self, CmAndError> {
        let prefix = match self.prepared {
            CmPrepared::Production(relation) => relation.into_prefix(),
            CmPrepared::Unaudited(prefix) => prefix,
        };
        let relation = PreparedRelation::with_ligerito(prefix, selection)
            .map_err(|error| CmAndError::LigeritoConfig(error.to_string()))?;
        Ok(Self {
            prepared: CmPrepared::Production(relation),
        })
    }

    pub fn ligerito_configuration(
        &self,
    ) -> Result<&crate::ligerito_flock::ResolvedLigerito, ProtocolError> {
        Ok(self.production()?.ligerito_configuration())
    }

    /// The shared layout.
    pub fn layout(&self) -> &CmAndLayout {
        &self.prefix().layout().layout
    }

    /// Prepared Spartan matrices.
    pub fn matrices(&self) -> &PreparedConstraintMatrices<SpartanBitzField> {
        match self.prefix().matrices() {
            MatrixSource::Fixed(matrices) => matrices,
            _ => unreachable!("the CM-AND matrices are prepared at a fixed field"),
        }
    }

    /// The canonical structural derivation map `M`.
    pub fn map(&self) -> &PreparedVirtualMap {
        &self.prefix().layout().map
    }
}

/// Generates and prepares the CM-AND statement at the runtime field
/// `field_config` (the fixed comparison field in production).
pub fn prepare_cm_and_relation(
    layout: CmAndLayout,
    field_config: &FieldConfig,
) -> Result<PreparedCmAndRelation, CmAndError> {
    let map = cm_and_map(&layout)?;
    let spec = CmAndSpec {
        layout,
        map,
        field_config: field_config.clone(),
    };
    let prefix = PreparedRelationPrefix::new::<Lambda100>(spec)
        .map_err(|error| CmAndError::LigeritoConfig(error.to_string()))?;
    let prepared = if layout.gate_vars() >= MIN_PRODUCTION_GATE_VARS {
        CmPrepared::Production(
            PreparedRelation::with_ligerito(prefix, LigeritoSelection::JOHNSON)
                .map_err(|error| CmAndError::LigeritoConfig(error.to_string()))?,
        )
    } else {
        CmPrepared::Unaudited(prefix)
    };
    Ok(PreparedCmAndRelation { prepared })
}

impl crate::sumcheck::outer::OuterRows for CmAndWitness {
    type AB = field::Z<2>;
    type C = field::Z<4>;
    fn dimensions(&self) -> (usize, usize, usize) {
        let rows = self.layout.gates.next_power_of_two();
        (rows, rows, rows)
    }
    fn a(&self, _row: usize) -> Self::AB {
        field::Z::ZERO
    }
    fn b(&self, _row: usize) -> Self::AB {
        field::Z::ZERO
    }
    fn c(&self, row: usize) -> Self::C {
        let [x, y, z, w] = [1, 2, 3, 4].map(|block| i128::from(self.block(block)[row]));
        field::Z::from(x + y - 2 * z - w)
    }
}

/// Materializes the field assignment and matrix products for reference callers.
/// Production proofs borrow the native witness through `OuterRows`.
pub fn project_cm_and_witness<F: SpartanField>(
    witness: &CmAndWitness,
    field_config: &F::Config,
) -> Result<EvaluatedSpartanAssignment<F>, CmAndError> {
    F::validate_config(field_config).map_err(SpartanMatrixError::from)?;
    let field_assignment: Vec<F> = witness
        .assignment
        .as_ref()
        .iter()
        .copied()
        .map(|value| F::from_with_cfg(value, field_config))
        .collect();

    let live = witness.layout.gates;
    let zero = F::zero_with_cfg(field_config);
    let zeros = vec![zero; live];
    let cz: Vec<F> = (0..live)
        .map(|i| {
            // x + y − 2z − w, in the field.
            let mut acc = field_assignment[witness.layout.capacity + i].clone();
            acc = field_config.add(
                &(acc),
                &(&field_assignment[2 * witness.layout.capacity + i]),
            );
            let mut two_z = field_assignment[3 * witness.layout.capacity + i].clone();
            two_z = field_config.add(
                &(two_z),
                &(&field_assignment[3 * witness.layout.capacity + i]),
            );
            acc = field_config.sub(&(acc), &(&two_z));
            acc = field_config.sub(
                &(acc),
                &(&field_assignment[4 * witness.layout.capacity + i]),
            );
            acc
        })
        .collect();
    let products = build_product_mles(&zeros, &zeros, &cz, live, field_config)
        .expect("validated CM row dimensions");
    let assignment = build_assignment_mle(
        &field_assignment,
        witness.layout.assignment_len(),
        field_config,
    )
    .expect("validated CM assignment dimensions");
    Ok(EvaluatedSpartanAssignment::new(assignment, products))
}

fn checked_pow2(exponent: usize) -> Result<usize, ProtocolError> {
    protocol::checked_pow2(exponent)
}

fn validate_cm_layout_geometry(layout: &CmAndLayout) -> Result<(), ProtocolError> {
    let p = layout.bitz_params();
    if p.word_bits != 1
        || p.row_vars < LOG_PACKING
        || p.col_vars > layout.gate_vars()
        || p.row_vars.saturating_add(p.word_bits) > 126
        || CM_AND_H_SLOTS != 1usize << 7
    {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    let total_vars = p
        .row_vars
        .checked_add(p.col_vars)
        .ok_or(ProtocolError::InvalidBitzParameters)?;
    if total_vars
        != layout
            .gate_vars()
            .checked_add(7)
            .ok_or(ProtocolError::InvalidBitzParameters)?
        || packed_vars(&p)
            != p.row_vars
                .checked_sub(LOG_PACKING)
                .and_then(|f| f.checked_add(p.col_vars))
                .ok_or(ProtocolError::InvalidBitzParameters)?
    {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    let cells = checked_pow2(p.row_vars)?
        .checked_mul(checked_pow2(p.col_vars)?)
        .ok_or(ProtocolError::InvalidBitzParameters)?;
    if cells
        != CM_AND_H_SLOTS
            .checked_mul(layout.capacity())
            .ok_or(ProtocolError::InvalidBitzParameters)?
    {
        return Err(ProtocolError::InvalidBitzParameters);
    }
    Ok(())
}

fn cm_configs(layout: &CmAndLayout) -> Result<(LigProverConfig, LigVerifierConfig), ProtocolError> {
    if layout.gate_vars() < MIN_PRODUCTION_GATE_VARS {
        return Err(ProtocolError::UnauditedBitzParameters);
    }
    let p = layout.bitz_params();
    sha_lig_configs(packed_vars(&p)).map_err(ProtocolError::LigeritoConfig)
}

/// Commits prebuilt compact `f` rows (`x`/`y`/`z` bits) under an explicit
/// Ligerito configuration. Configurations below the audited `m ≥ 22`
/// regime are TEST-ONLY, exactly as for the direct bridge.
pub fn commit_cm_and_witness_with_config(
    layout: &CmAndLayout,
    rows: Vec<Vec<u64>>,
    pc: &LigProverConfig,
) -> Result<FlockCommitHint, ProtocolError> {
    validate_cm_layout_geometry(layout)?;
    let p = layout.bitz_params();
    protocol::validate_bit_rows(&p, &rows)?;
    let hint = commit_rs_ligerito_rows(&p, rows, pc);
    protocol::validate_commitment(&p, &hint.commitment, pc)?;
    Ok(hint)
}

/// Commits prebuilt compact `f` rows with the production (validator-gated,
/// `≥ 2^15` gate slots) configuration.
pub fn commit_cm_and_witness(
    layout: &CmAndLayout,
    rows: Vec<Vec<u64>>,
) -> Result<FlockCommitHint, ProtocolError> {
    let (pc, vc) = cm_configs(layout)?;
    let p = layout.bitz_params();
    protocol::validate_config_pair(&p, &pc, &vc)?;
    commit_cm_and_witness_with_config(layout, rows, &pc)
}

/// Proves the CM-AND relation and opens the derived-grid assignment claim
/// against the compact `f` commitment, under an explicit prover
/// configuration (no opener policy digest is bound).
pub fn prove_cm_and_bitz_with_config<T: Transcript + Send>(
    transcript: &mut T,
    relation: &PreparedCmAndRelation,
    witness: &CmAndWitness,
    hint_f: &FlockCommitHint,
    pc: &LigProverConfig,
) -> Result<Proof<IntEvalRsLigVirtProof>, ProtocolError> {
    let opener = Opener::Custom {
        prover: Some(pc.clone()),
        verifier: None,
    };
    protocol::prove_virtual_with_opener(transcript, relation.prefix(), &opener, witness, hint_f)
}

/// Proves with the production (validator-gated) configuration.
pub fn prove_cm_and_bitz<T: Transcript + Send>(
    transcript: &mut T,
    relation: &PreparedCmAndRelation,
    witness: &CmAndWitness,
    hint_f: &FlockCommitHint,
) -> Result<Proof<IntEvalRsLigVirtProof>, ProtocolError> {
    protocol::prove_virtual(transcript, relation.production()?, witness, hint_f)
}

/// Verifies both proof systems on one transcript, under an explicit
/// verifier configuration. The terminal Spartan claim is always derived
/// from the proof, never trusted from the prover.
pub fn verify_cm_and_bitz_with_config<T: Transcript + Send>(
    transcript: &mut T,
    relation: &PreparedCmAndRelation,
    commitment: &Commitment,
    proof: &Proof<IntEvalRsLigVirtProof>,
    vc: &LigVerifierConfig,
) -> Result<(), ProtocolError> {
    let opener = Opener::Custom {
        prover: None,
        verifier: Some(vc.clone()),
    };
    protocol::verify_virtual_with_opener(transcript, relation.prefix(), &opener, commitment, proof)
}

/// Verifies with the production (validator-gated) configuration.
pub fn verify_cm_and_bitz<T: Transcript + Send>(
    transcript: &mut T,
    relation: &PreparedCmAndRelation,
    commitment: &Commitment,
    proof: &Proof<IntEvalRsLigVirtProof>,
) -> Result<(), ProtocolError> {
    protocol::verify_virtual(transcript, relation.production()?, commitment, proof)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        pcs::{FQ_MOD, Q100Element, eq_le_table_fq},
        piop::spartan::{
            bitz::spartan_bitz_field_config, matrix::ScaledMleEvaluationClaim, protocol::bitify,
        },
    };

    #[test]
    fn layout_and_cells_are_slot_major() {
        assert_eq!(CmAndLayout::new(0), Err(CmAndError::EmptyBatch));
        for (gates, capacity) in [(1, 256), (200, 256), (257, 512)] {
            let layout = CmAndLayout::new(gates).unwrap();
            assert_eq!(layout.capacity(), capacity);
            assert_eq!(layout.assignment_len(), 5 * capacity);
            let p = layout.bitz_params();
            assert_eq!(p.row_vars + p.col_vars, layout.gate_vars() + 7);
            for slot in [0, 31, 32, 95, 96, 127] {
                for gate in [0, capacity - 1] {
                    let (b, c) = layout.cell(slot, gate).unwrap();
                    assert_eq!(layout.flat_cell(slot, gate), (c << p.row_vars) | b);
                }
            }
            assert!(layout.cell(128, 0).is_none());
        }
    }

    #[test]
    fn map_derives_w_as_xor_and_keeps_dead_slots_unreferenced() {
        let layout = CmAndLayout::new(3).unwrap();
        let map = cm_and_map(&layout).unwrap();
        let p = layout.bitz_params();
        assert_eq!(map.rows(), cell_count(&p));
        assert_eq!(map.cols(), cell_count(&p));
        assert_eq!(map.nnz(), layout.capacity() * (CM_AND_F_LIVE_SLOTS + 64));

        // Source x/y columns feed both their identity row and the matching w row.
        for (slot, gate) in [(0usize, 0usize), (40, 2), (63, 255)] {
            let source = layout.flat_cell(slot, gate);
            let column = map.matrix().column(source).unwrap();
            let bit = slot % CM_AND_WORD_BITS;
            assert_eq!(
                column.indices(),
                &[source, layout.flat_cell(CM_AND_W_SLOT + bit, gate)]
            );
        }
        // Source z columns feed only their identity row.
        for (slot, gate) in [(64usize, 0usize), (81, 2), (95, 255)] {
            let source = layout.flat_cell(slot, gate);
            assert_eq!(map.matrix().column(source).unwrap().indices(), &[source]);
        }
        // Padded source w slots are structurally dead.
        for (bit, gate) in [(0usize, 0usize), (17, 2), (31, 255)] {
            let source = layout.flat_cell(CM_AND_W_SLOT + bit, gate);
            assert!(map.matrix().column(source).unwrap().is_empty());
        }
    }

    #[test]
    fn witness_blocks_and_bit_rows_are_consistent() {
        let inputs = [(0xdead_beefu32, 0x0f0f_0f0f), (u32::MAX, 1), (7, 7)];
        let witness = CmAndWitness::from_inputs(&inputs).unwrap();
        let capacity = witness.layout().capacity();
        assert_eq!(witness.assignment()[0], 1);
        for (i, (x, y)) in inputs.iter().enumerate() {
            assert_eq!(witness.assignment()[capacity + i], u64::from(*x));
            assert_eq!(witness.assignment()[2 * capacity + i], u64::from(*y));
            assert_eq!(witness.assignment()[3 * capacity + i], u64::from(x & y));
            assert_eq!(witness.assignment()[4 * capacity + i], u64::from(x ^ y));
        }

        let f_rows = witness.f_bit_rows();
        let h_rows = witness.h_bit_rows();
        let layout = witness.layout();
        for (gate, (x, y)) in inputs.iter().enumerate() {
            for (slot_start, value) in [
                (CM_AND_X_SLOT, *x),
                (CM_AND_Y_SLOT, *y),
                (CM_AND_Z_SLOT, x & y),
            ] {
                for bit in 0..CM_AND_WORD_BITS {
                    let (b, c) = layout.cell(slot_start + bit, gate).unwrap();
                    let committed = (f_rows[c][b / 64] >> (b % 64)) & 1;
                    assert_eq!(committed, u64::from((value >> bit) & 1));
                    assert_eq!((h_rows[c][b / 64] >> (b % 64)) & 1, committed);
                }
            }
            // The w block is NOT committed.
            for bit in 0..CM_AND_WORD_BITS {
                let (b, c) = layout.cell(CM_AND_W_SLOT + bit, gate).unwrap();
                assert_eq!((f_rows[c][b / 64] >> (b % 64)) & 1, 0);
                assert_eq!(
                    (h_rows[c][b / 64] >> (b % 64)) & 1,
                    u64::from(((x ^ y) >> bit) & 1)
                );
            }
        }
    }

    #[test]
    fn honest_witness_satisfies_the_relation_and_false_one_does_not() {
        let config = spartan_bitz_field_config();
        let witness = CmAndWitness::from_inputs(&[(3, 5), (0xffff_0000, 0x00ff_00ff)]).unwrap();
        let projected = project_cm_and_witness::<SpartanBitzField>(&witness, &config).unwrap();
        let zero = SpartanBitzField::zero_with_cfg(&config);
        assert!(
            projected
                .products()
                .cz
                .evaluations
                .iter()
                .all(|v| v == &zero)
        );

        let bad = CmAndWitness::from_gate_values(2, |i| {
            let (x, y) = [(3u32, 5u32), (0xffff_0000, 0x00ff_00ff)][i];
            let z = if i == 0 { (x & y) ^ 1 } else { x & y };
            (x, y, z, x ^ y)
        })
        .unwrap();
        let projected = project_cm_and_witness::<SpartanBitzField>(&bad, &config).unwrap();
        assert_ne!(projected.products().cz.evaluations[0], zero);
        assert_eq!(projected.products().cz.evaluations[1], zero);
    }

    #[test]
    fn relation_coefficients_follow_the_runtime_field_modulus() {
        let config = Fp::<2>::make_cfg(&Uint::from((1_u128 << 127) - 1)).unwrap();
        let witness = CmAndWitness::from_inputs(&[(3, 5), (0xffff_0000, 0x00ff_00ff)]).unwrap();
        let relation = prepare_cm_and_relation(*witness.layout(), &config).unwrap();
        let projected = project_cm_and_witness::<Fp<2>>(&witness, &config).unwrap();

        let mut matrix_products = vec![Fp::<2>::zero_with_cfg(&config); witness.layout().gates()];
        for (column, entries) in relation.matrices().matrices().c().columns().enumerate() {
            for (row, coefficient) in entries {
                let mut term = coefficient.clone();
                term = config.mul(&(term), &(&projected.assignment().evaluations[column]));
                matrix_products[row] = config.add(&(matrix_products[row]), &(&term));
            }
        }

        assert_eq!(
            &projected.products().cz.evaluations[..witness.layout().gates()],
            matrix_products.as_slice(),
        );
        assert!(
            matrix_products
                .iter()
                .all(|value| <Fp<2> as crate::piop::spartan::SpartanField>::is_zero(value))
        );
    }

    #[test]
    fn bitification_is_the_adjoint_of_the_blocked_reconstruction() {
        let witness =
            CmAndWitness::from_inputs(&[(0, u32::MAX), (1, 7), (u32::MAX, u32::MAX)]).unwrap();
        let layout = *witness.layout();
        let p = layout.bitz_params();
        let config = spartan_bitz_field_config();
        let arith = field::FpCtx::from_prime_u128(FQ_MOD);
        let spec = CmAndSpec {
            layout,
            map: cm_and_map(&layout).unwrap(),
            field_config: config.clone(),
        };

        let gate_point: Vec<Q100Element> = (0..layout.gate_vars())
            .map(|i| Q100Element::from_u128((i + 2) as u128))
            .collect();
        let sel: [Q100Element; 3] = [
            Q100Element::from_u128(7),
            Q100Element::from_u128(11),
            Q100Element::from_u128(29),
        ];
        let scale = Q100Element::from_u128(13);
        let eq_sel = eq_le_table_fq(&sel);
        let eq_gate = eq_le_table_fq(&gate_point);

        // Z(point) directly from the five logical blocks.
        let mut z_eval = Q100Element::from_u128(0);
        for block in 0..ASSIGNMENT_BLOCKS {
            for gate in 0..layout.capacity() {
                z_eval = z_eval
                    + eq_sel[block]
                        * eq_gate[gate]
                        * Q100Element::from(u128::from(
                            witness.assignment()[block * layout.capacity() + gate],
                        ));
            }
        }
        let value = scale * z_eval;

        let mut point = gate_point.clone();
        point.extend(sel);
        let point_f: Vec<SpartanBitzField> = point
            .iter()
            .map(|c| SpartanBitzField::from_with_cfg(c.canonical_u128(), &config))
            .collect();
        let claim = ScaledMleEvaluationClaim::new(
            point_f.into_boxed_slice(),
            SpartanBitzField::from_with_cfg(scale.canonical_u128(), &config),
            SpartanBitzField::from_with_cfg(value.canonical_u128(), &config),
        );
        let table = spec.block_table();
        let opening = bitify::bitify(
            &claim,
            p,
            layout.gate_vars(),
            &table,
            ScaleSide::Columns,
            &arith,
        )
        .unwrap();
        let row_weights = bitify::dense_row_weights(&opening, &table, &arith).unwrap();
        let col_weights = bitify::column_weights(&opening, &arith).unwrap();

        // Read the DERIVED grid h = M·f through the opening weights.
        let h_rows = witness.h_bit_rows();
        let mut read_off = Q100Element::from_u128(0);
        for b in 0..p.rows() {
            for c in 0..p.cols() {
                let bit = (h_rows[c][b / 64] >> (b % 64)) & 1;
                read_off = read_off
                    + Q100Element::from(u128::from(bit))
                        * Q100Element::from_u128(row_weights[b])
                        * Q100Element::from_u128(col_weights[c]);
            }
        }
        assert_eq!(read_off.canonical_u128(), opening.claimed);
    }

    #[test]
    fn production_entry_points_gate_small_layouts() {
        let layout = CmAndLayout::new(3).unwrap();
        assert!(matches!(
            cm_configs(&layout),
            Err(ProtocolError::UnauditedBitzParameters)
        ));
        let production = CmAndLayout::new(1 << MIN_PRODUCTION_GATE_VARS).unwrap();
        cm_configs(&production).expect("the smallest embedded profile is available");
        let relation = prepare_cm_and_relation(layout, &spartan_bitz_field_config()).unwrap();
        assert!(matches!(
            relation.ligerito_configuration(),
            Err(ProtocolError::UnauditedBitzParameters)
        ));
    }
}
