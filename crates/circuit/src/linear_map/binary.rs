//! Prepared public binary linear maps and their canonical topology.
use super::sparse::{CscMatrix, ImplicitOnes};
use blake3::Hasher;
use core::slice;
use std::vec as alloc_vec;
use thiserror::Error;

/// Failures while validating public virtual-map metadata.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum PreparedVirtualMapError {
    /// A binary sparse map must store only implicit-one coefficients.
    #[error("virtual map contains an explicit false coefficient at row {row}, column {column}")]
    ExplicitFalse { row: usize, column: usize },

    /// The fixed-width canonical digest cannot encode this host shape.
    #[error("virtual map dimensions do not fit the canonical u64 encoding")]
    ShapeTooLarge,

    /// A repeated map needs at least one instance and all derived dimensions
    /// must fit the host index type.
    #[error("virtual map repetition has invalid or overflowing geometry")]
    InvalidRepetition,
}

/// Read-only interface needed by the virtual-opening protocol.
///
/// Arbitrary maps use [`PreparedVirtualMap`]'s canonical CSC storage. Large
/// batches can instead expose an implicit tensor repetition without allocating
/// the fully expanded CSC column-offset array.
pub trait VirtualMap: Sync {
    /// Structural bound on output bits within each `word_stride`-bit cell.
    /// Every touched row must satisfy `row % word_stride < bound`.
    /// Unknown maps are validated through their complete column support.
    fn output_word_bits(&self, _word_stride: usize) -> Option<usize> {
        None
    }

    /// Iterator over the derived rows touched by one source column.
    type ColumnRows<'a>: ExactSizeIterator<Item = usize>
    where
        Self: 'a;

    /// Number of derived cells.
    fn rows(&self) -> usize;

    /// Number of committed source cells.
    fn cols(&self) -> usize;

    /// Number of implicit-one entries.
    fn nnz(&self) -> usize;

    /// Canonical statement digest.
    fn digest(&self) -> [u8; 32];

    /// Whether the complete map is exactly the identity.
    fn is_identity(&self) -> bool;

    /// Derived row indices for one source column, in increasing order.
    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>>;

    /// The map's implicit tensor repetition, when it has one: the local
    /// map and the instance count, with global bit-cell indices laid out
    /// `global = local · instances + instance`. Lets the virtual-opening
    /// prover and verifier factor per-column work over the repetition.
    /// Default: no structure exposed.
    fn repetition(&self) -> Option<(&PreparedVirtualMap, usize)> {
        None
    }

    /// Mixed-layout tensor repetition used by
    /// [`PackedSourceRepeatedVirtualMap`]. Derived rows use
    /// `local_row * instances + instance` ([`PackedSourceOrder::LocalMajor`])
    /// or `instance * local_stride + local_row`
    /// ([`PackedSourceOrder::InstanceMajor`], see
    /// [`Self::packed_source_order`]), while nonconstant source columns
    /// are packed instance-major behind one shared constant column. This is
    /// deliberately separate from [`Self::repetition`], whose source layout
    /// is local-major.
    fn packed_source_repetition(&self) -> Option<(&PreparedVirtualMap, usize)> {
        None
    }

    /// Derived-index order of [`Self::packed_source_repetition`].
    fn packed_source_order(&self) -> PackedSourceOrder {
        PackedSourceOrder::LocalMajor
    }

    /// The chained packed-source repetition of a [`ChainedPackedSourceMap`]:
    /// the plain repetition plus the cross-instance (`prev`) and boundary
    /// (`first`/`last`) local maps, local-major derived order. Default: no
    /// structure exposed.
    fn chained_packed_source(&self) -> Option<ChainedPackedSourceParts<'_>> {
        None
    }

    /// A compact relation appended to the chained derived rows. Its source
    /// prefix aliases existing cells; all other columns occupy one dense tail.
    /// When present, both hooks together describe the complete map.
    fn chained_packed_source_tail(&self) -> Option<ChainedSourceTail<'_>> {
        None
    }
}

/// Compact correction to a chained map, with no globally expanded CSC matrix.
pub struct ChainedSourceTail<'a> {
    pub map: &'a PreparedVirtualMap,
    pub row_offset: usize,
    pub source_offset: usize,
    pub aliases: &'a [usize],
}

/// Derived-index order of a [`PackedSourceRepeatedVirtualMap`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedSourceOrder {
    /// `derived = local_row · instances + instance`: the low `log₂ instances`
    /// bits of a derived index select the instance, so an BitZ split with
    /// `t ≤ log₂ instances` puts instance bits on the rows and every local
    /// bit on the columns.
    LocalMajor,
    /// `derived = instance · local_stride + local_row` with
    /// `local_stride = local.rows().next_power_of_two()`: the low
    /// `log₂ local_stride` bits select the local row, so an BitZ split with
    /// `t ≥ log₂ local_stride` puts every local bit (plus the low instance
    /// bits) on the rows and only high instance bits on the columns. The
    /// per-instance padding rows are structurally zero.
    InstanceMajor,
}

/// A validated binary CSC matrix with transcript metadata cached once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedVirtualMap {
    matrix: CscMatrix<ImplicitOnes>,
    digest: [u8; 32],
    identity: bool,
}

impl PreparedVirtualMap {
    /// Validates a true-only binary CSC matrix and prepares its metadata.
    pub fn new(matrix: CscMatrix<Box<[bool]>>) -> Result<Self, PreparedVirtualMapError> {
        for (column, entries) in matrix.columns().enumerate() {
            for (row, coefficient) in entries {
                if !*coefficient {
                    return Err(PreparedVirtualMapError::ExplicitFalse { row, column });
                }
            }
        }

        let nnz = matrix.nnz();
        Self::from_implicit(
            matrix
                .with_coefficients(ImplicitOnes::new(nnz))
                .expect("unchanged entry count"),
        )
    }

    /// Prepare an implicit-one map without allocating any coefficient array.
    pub fn from_implicit(matrix: CscMatrix<ImplicitOnes>) -> Result<Self, PreparedVirtualMapError> {
        let digest = compute_digest(&matrix)?;
        let identity = compute_identity(&matrix);
        Ok(Self {
            matrix,
            digest,
            identity,
        })
    }

    /// The sole sparse topology backing this prepared map.
    pub const fn matrix(&self) -> &CscMatrix<ImplicitOnes> {
        &self.matrix
    }

    /// Number of derived cells on the `h` side.
    pub fn rows(&self) -> usize {
        self.matrix.row_count()
    }

    /// Number of committed source cells on the `f` side.
    pub fn cols(&self) -> usize {
        self.matrix.column_count()
    }

    /// Number of stored one-coefficients.
    pub fn nnz(&self) -> usize {
        self.matrix.nnz()
    }

    /// Canonical CSC statement digest.
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Whether this is exactly the square identity matrix.
    pub const fn is_identity(&self) -> bool {
        self.identity
    }
}

impl VirtualMap for PreparedVirtualMap {
    type ColumnRows<'a> = core::iter::Copied<slice::Iter<'a, usize>>;

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn nnz(&self) -> usize {
        self.nnz()
    }

    fn digest(&self) -> [u8; 32] {
        self.digest()
    }

    fn is_identity(&self) -> bool {
        self.is_identity()
    }

    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>> {
        Some(self.matrix.column(column)?.indices().iter().copied())
    }
}

/// Tensor repetition of one local CSC map over independent instances.
///
/// Global bit-cell indices follow the virtual protocol's row-low ordering:
/// `global = local * instances + instance`. Consequently a local edge
/// `local_source -> local_derived` becomes, for every `instance`,
/// `local_source * instances + instance -> local_derived * instances + instance`.
/// Only the local CSC is stored.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepeatedVirtualMap {
    local: PreparedVirtualMap,
    instances: usize,
    instance_bits: Option<u32>,
    rows: usize,
    cols: usize,
    nnz: usize,
    digest: [u8; 32],
}

/// Packed repetition of one local map with a single shared constant cell.
///
/// Unlike [`RepeatedVirtualMap`], this layout does not round every local
/// witness up to a power of two.  If the local map has shape `h x f`, the
/// live prefixes for `instances = N` are
///
/// ```text
/// source:     [1 | f_0[1..] | f_1[1..] | ... | f_{N-1}[1..]]
/// assignment: [1 | h_0[1..] | h_1[1..] | ... | h_{N-1}[1..]]
/// ```
///
/// and only the complete prefixes are padded to `cols` and `rows`.  The
/// leading constant is shared by every repetition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedRepeatedVirtualMap {
    local: PreparedVirtualMap,
    instances: usize,
    rows: usize,
    cols: usize,
    live_rows: usize,
    live_cols: usize,
    nnz: usize,
    digest: [u8; 32],
}

/// Tensor-derived repetition of one local map over a canonically packed source.
///
/// The committed source keeps the gap-free instance-major representation
///
/// ```text
/// [1 | f_0[1..] | f_1[1..] | ... | f_{N-1}[1..] | 0 ... 0],
/// ```
///
/// while the derived side is a proof-only local-major tensor
///
/// ```text
/// h[local, instance] = h_instance[local].
/// ```
///
/// In particular, local row zero is logically repeated for every instance,
/// but every repetition maps back to the one shared committed source cell.
/// This layout makes a product coefficient `u[instance] * d[local]` factor
/// directly across the BitZ row and column coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedSourceRepeatedVirtualMap {
    local: PreparedVirtualMap,
    instances: usize,
    rows: usize,
    cols: usize,
    live_rows: usize,
    live_cols: usize,
    nnz: usize,
    digest: [u8; 32],
    order: PackedSourceOrder,
    local_stride: usize,
}

impl PackedSourceRepeatedVirtualMap {
    /// Builds the mixed-layout repetition inside power-of-two global domains
    /// with the local-major derived order.
    pub fn new(
        local: PreparedVirtualMap,
        instances: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, PreparedVirtualMapError> {
        Self::new_with_order(local, instances, rows, cols, PackedSourceOrder::LocalMajor)
    }

    /// [`Self::new`] with an explicit derived-index order. The local-major
    /// digest is unchanged; the instance-major order is bound by a digest tag.
    pub fn new_with_order(
        local: PreparedVirtualMap,
        instances: usize,
        rows: usize,
        cols: usize,
        order: PackedSourceOrder,
    ) -> Result<Self, PreparedVirtualMapError> {
        if instances == 0
            || !instances.is_power_of_two()
            || local.rows() == 0
            || local.cols() == 0
            || !rows.is_power_of_two()
            || !cols.is_power_of_two()
        {
            return Err(PreparedVirtualMapError::InvalidRepetition);
        }
        let local_stride = match order {
            PackedSourceOrder::LocalMajor => instances,
            PackedSourceOrder::InstanceMajor => local
                .rows()
                .checked_next_power_of_two()
                .ok_or(PreparedVirtualMapError::InvalidRepetition)?,
        };
        let live_rows = match order {
            PackedSourceOrder::LocalMajor => local.rows().checked_mul(instances),
            PackedSourceOrder::InstanceMajor => local_stride.checked_mul(instances),
        }
        .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let live_cols = local
            .cols()
            .checked_sub(1)
            .and_then(|width| width.checked_mul(instances))
            .and_then(|cells| cells.checked_add(1))
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let nnz = local
            .nnz()
            .checked_mul(instances)
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        if live_rows > rows || live_cols > cols {
            return Err(PreparedVirtualMapError::InvalidRepetition);
        }

        let mut hash = Hasher::new();
        hash.update(b"bitz/packed-source-repeated-virtual-map/v1");
        hash.update(&local.digest());
        for value in [instances, rows, cols, live_rows, live_cols, nnz] {
            hash.update(
                &u64::try_from(value)
                    .map_err(|_| PreparedVirtualMapError::ShapeTooLarge)?
                    .to_le_bytes(),
            );
        }
        if order == PackedSourceOrder::InstanceMajor {
            hash.update(b"instance-major");
            hash.update(
                &u64::try_from(local_stride)
                    .map_err(|_| PreparedVirtualMapError::ShapeTooLarge)?
                    .to_le_bytes(),
            );
        }
        let digest = *hash.finalize().as_bytes();

        Ok(Self {
            local,
            instances,
            rows,
            cols,
            live_rows,
            live_cols,
            nnz,
            digest,
            order,
            local_stride,
        })
    }

    /// Derived-index order of this view.
    pub const fn order(&self) -> PackedSourceOrder {
        self.order
    }

    /// Row stride of one instance block under
    /// [`PackedSourceOrder::InstanceMajor`] (`local.rows()` rounded up to a
    /// power of two); equals `instances` under the local-major order.
    pub const fn local_stride(&self) -> usize {
        self.local_stride
    }

    /// Local canonical CSC map repeated by this view.
    pub const fn local(&self) -> &PreparedVirtualMap {
        &self.local
    }

    /// Number of independent repetitions.
    pub const fn instances(&self) -> usize {
        self.instances
    }

    /// Number of live derived cells before the one global suffix.
    pub const fn live_rows(&self) -> usize {
        self.live_rows
    }

    /// Number of live committed source cells before the one global suffix.
    pub const fn live_cols(&self) -> usize {
        self.live_cols
    }
}

/// Derived-row iterator for one packed source column, in increasing order
/// under either [`PackedSourceOrder`].
pub struct PackedSourceColumnRows<'a> {
    local_rows: &'a [usize],
    instance: Option<usize>,
    instances: usize,
    order: PackedSourceOrder,
    local_stride: usize,
    position: usize,
    repeat_index: usize,
    remaining: usize,
}

impl PackedSourceColumnRows<'_> {
    #[inline]
    fn derived(&self, local_row: usize, instance: usize) -> usize {
        match self.order {
            PackedSourceOrder::LocalMajor => local_row * self.instances + instance,
            PackedSourceOrder::InstanceMajor => instance * self.local_stride + local_row,
        }
    }
}

impl Iterator for PackedSourceColumnRows<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let output = match (self.instance, self.order) {
            (Some(instance), _) => {
                let local_row = *self.local_rows.get(self.position)?;
                self.position += 1;
                self.derived(local_row, instance)
            }
            // The shared constant column touches every instance. Local-major
            // ascends with the instance innermost, instance-major with the
            // local row innermost.
            (None, PackedSourceOrder::LocalMajor) => {
                let local_row = *self.local_rows.get(self.position)?;
                let output = self.derived(local_row, self.repeat_index);
                self.repeat_index += 1;
                if self.repeat_index == self.instances {
                    self.repeat_index = 0;
                    self.position += 1;
                }
                output
            }
            (None, PackedSourceOrder::InstanceMajor) => {
                let local_row = *self.local_rows.get(self.position)?;
                let output = self.derived(local_row, self.repeat_index);
                self.position += 1;
                if self.position == self.local_rows.len() {
                    self.position = 0;
                    self.repeat_index += 1;
                }
                output
            }
        };
        self.remaining -= 1;
        Some(output)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for PackedSourceColumnRows<'_> {}

impl VirtualMap for PackedSourceRepeatedVirtualMap {
    type ColumnRows<'a> = PackedSourceColumnRows<'a>;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn is_identity(&self) -> bool {
        self.instances == 1
            && self.live_rows == self.rows
            && self.live_cols == self.cols
            && self.local.is_identity()
    }

    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>> {
        if column >= self.cols {
            return None;
        }
        if column >= self.live_cols {
            return Some(PackedSourceColumnRows {
                local_rows: &[],
                instance: Some(0),
                instances: self.instances,
                order: self.order,
                local_stride: self.local_stride,
                position: 0,
                repeat_index: 0,
                remaining: 0,
            });
        }

        let (local_column, instance) = if column == 0 {
            (0, None)
        } else {
            let offset = column - 1;
            (
                1 + offset % (self.local.cols() - 1),
                Some(offset / (self.local.cols() - 1)),
            )
        };
        let local_rows = self.local.matrix().column(local_column)?.indices();
        let remaining = local_rows.len() * instance.map_or(self.instances, |_| 1);
        Some(PackedSourceColumnRows {
            local_rows,
            instance,
            instances: self.instances,
            order: self.order,
            local_stride: self.local_stride,
            position: 0,
            repeat_index: 0,
            remaining,
        })
    }

    fn packed_source_repetition(&self) -> Option<(&PreparedVirtualMap, usize)> {
        Some((&self.local, self.instances))
    }

    fn packed_source_order(&self) -> PackedSourceOrder {
        self.order
    }
}

/// The four local maps and instance count of a [`ChainedPackedSourceMap`],
/// exposed so the virtual-opening engines can factor their work over the
/// repetition and its cross-instance terms.
#[derive(Clone, Copy)]
pub struct ChainedPackedSourceParts<'a> {
    /// Instance `i`'s rows from instance `i`'s own cells (every instance).
    pub local: &'a PreparedVirtualMap,
    /// Instance `i`'s rows from instance `i - 1`'s nonconstant cells
    /// (`i ≥ 1`; the chain link).
    pub prev: &'a PreparedVirtualMap,
    /// Instance 0's rows from instance 0's cells and the shared constant.
    pub first: &'a PreparedVirtualMap,
    /// Instance `N - 1`'s rows from instance `N - 1`'s cells and the shared
    /// constant.
    pub last: &'a PreparedVirtualMap,
    /// Number of instances `N` (a power of two, at least two).
    pub instances: usize,
}

/// Packed-source repetition with the cross-instance references a CHAINED
/// relation needs — e.g. the SHA-256 Merkle–Damgård chain, where instance
/// `i`'s chaining-state rows read instance `i - 1`'s output cells instead of
/// committed cells of their own.
///
/// The committed source is the gap-free instance-major packing of
/// [`PackedSourceRepeatedVirtualMap`] (`[1 | f_0[1..] | … | f_{N-1}[1..] | 0…]`)
/// and the derived side is its local-major product tensor
/// `h[local, instance]` (derived index `local · N + instance`). Four local
/// maps of one shape `h_local × f_local` compose the global map:
///
/// * `local` — instance `i`'s rows from instance `i`'s own cells, for every
///   instance (the plain repetition);
/// * `prev` — instance `i`'s rows from instance `i - 1`'s NONCONSTANT cells,
///   for `i ≥ 1` (its constant column must be empty);
/// * `first` — instance 0's rows from instance 0's cells and the shared
///   constant (boundary constants such as the SHA-256 initial state);
/// * `last` — instance `N - 1`'s rows from instance `N - 1`'s cells and the
///   shared constant (boundary read-outs such as the terminal digest).
///
/// Every `(derived row, source cell)` pair is stored at most once: `first`
/// and `last` may not share an entry with `local` (validated per column),
/// and `prev`'s entries land on the NEXT instance, so they never coincide
/// with another map's. Only the local-major derived order is supported.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainedPackedSourceMap {
    local: PreparedVirtualMap,
    prev: PreparedVirtualMap,
    first: PreparedVirtualMap,
    last: PreparedVirtualMap,
    instances: usize,
    rows: usize,
    cols: usize,
    live_rows: usize,
    live_cols: usize,
    nnz: usize,
    digest: [u8; 32],
}

impl ChainedPackedSourceMap {
    /// Builds the chained repetition inside power-of-two global domains.
    pub fn new(
        local: PreparedVirtualMap,
        prev: PreparedVirtualMap,
        first: PreparedVirtualMap,
        last: PreparedVirtualMap,
        instances: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, PreparedVirtualMapError> {
        let shape = (local.rows(), local.cols());
        if instances < 2
            || !instances.is_power_of_two()
            || local.rows() == 0
            || local.cols() < 2
            || !rows.is_power_of_two()
            || !cols.is_power_of_two()
            || [&prev, &first, &last]
                .iter()
                .any(|map| (map.rows(), map.cols()) != shape)
            || prev
                .matrix()
                .column(0)
                .is_none_or(|column| !column.is_empty())
        {
            return Err(PreparedVirtualMapError::InvalidRepetition);
        }
        // `first`/`last` entries coexist with `local`'s on one instance and
        // must not cancel any of them.
        for boundary in [&first, &last] {
            for (own, extra) in local.matrix().columns().zip(boundary.matrix().columns()) {
                if sorted_intersect(own.indices(), extra.indices()) {
                    return Err(PreparedVirtualMapError::InvalidRepetition);
                }
            }
        }
        let live_rows = local
            .rows()
            .checked_mul(instances)
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let live_cols = (local.cols() - 1)
            .checked_mul(instances)
            .and_then(|cells| cells.checked_add(1))
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        if live_rows > rows || live_cols > cols {
            return Err(PreparedVirtualMapError::InvalidRepetition);
        }
        let nnz = local
            .nnz()
            .checked_mul(instances)
            .and_then(|count| count.checked_add(prev.nnz().checked_mul(instances - 1)?))
            .and_then(|count| count.checked_add(first.nnz()))
            .and_then(|count| count.checked_add(last.nnz()))
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;

        let mut hash = Hasher::new();
        hash.update(b"bitz/chained-packed-source-map/v1");
        for map in [&local, &prev, &first, &last] {
            hash.update(&map.digest());
        }
        for value in [instances, rows, cols, live_rows, live_cols, nnz] {
            hash.update(
                &u64::try_from(value)
                    .map_err(|_| PreparedVirtualMapError::ShapeTooLarge)?
                    .to_le_bytes(),
            );
        }
        let digest = *hash.finalize().as_bytes();

        Ok(Self {
            local,
            prev,
            first,
            last,
            instances,
            rows,
            cols,
            live_rows,
            live_cols,
            nnz,
            digest,
        })
    }

    /// The four local maps and the instance count.
    pub const fn parts(&self) -> ChainedPackedSourceParts<'_> {
        ChainedPackedSourceParts {
            local: &self.local,
            prev: &self.prev,
            first: &self.first,
            last: &self.last,
            instances: self.instances,
        }
    }

    /// The plain per-instance map.
    pub const fn local(&self) -> &PreparedVirtualMap {
        &self.local
    }

    /// The chain link: rows of instance `i` from cells of instance `i - 1`.
    pub const fn prev(&self) -> &PreparedVirtualMap {
        &self.prev
    }

    /// Instance 0's boundary map.
    pub const fn first(&self) -> &PreparedVirtualMap {
        &self.first
    }

    /// Instance `N - 1`'s boundary map.
    pub const fn last(&self) -> &PreparedVirtualMap {
        &self.last
    }

    /// Number of chained instances.
    pub const fn instances(&self) -> usize {
        self.instances
    }

    /// Number of live derived cells before the one global suffix.
    pub const fn live_rows(&self) -> usize {
        self.live_rows
    }

    /// Number of live committed source cells before the one global suffix.
    pub const fn live_cols(&self) -> usize {
        self.live_cols
    }

    /// Nonconstant local-column offsets `[lo, hi)` (`0..width`) that carry
    /// any entry of `map`; `(0, 0)` when there are none.
    pub fn nonconstant_column_span(map: &PreparedVirtualMap) -> (usize, usize) {
        let mut span: Option<(usize, usize)> = None;
        for (column, entries) in map.matrix().columns().enumerate().skip(1) {
            if entries.is_empty() {
                continue;
            }
            let offset = column - 1;
            span = Some(match span {
                None => (offset, offset + 1),
                Some((lo, _)) => (lo, offset + 1),
            });
        }
        span.unwrap_or((0, 0))
    }
}

/// Whether two strictly increasing index lists share an element.
fn sorted_intersect(left: &[usize], right: &[usize]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => return true,
        }
    }
    false
}

impl VirtualMap for ChainedPackedSourceMap {
    type ColumnRows<'a> = alloc_vec::IntoIter<usize>;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn is_identity(&self) -> bool {
        false
    }

    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>> {
        if column >= self.cols {
            return None;
        }
        let instances = self.instances;
        let mut derived = Vec::new();
        if column >= self.live_cols {
            return Some(derived.into_iter());
        }
        let push_rows =
            |derived: &mut Vec<usize>, map: &PreparedVirtualMap, local_column, instance| {
                if let Some(rows) = map.matrix().column(local_column) {
                    derived.extend(rows.indices().iter().map(|&row| row * instances + instance));
                }
            };
        if column == 0 {
            if let Some(rows) = self.local.matrix().column(0) {
                for &row in rows.indices() {
                    derived.extend((0..instances).map(|instance| row * instances + instance));
                }
            }
            push_rows(&mut derived, &self.first, 0, 0);
            push_rows(&mut derived, &self.last, 0, instances - 1);
        } else {
            let offset = column - 1;
            let width = self.local.cols() - 1;
            let instance = offset / width;
            let local_column = 1 + offset % width;
            push_rows(&mut derived, &self.local, local_column, instance);
            if instance + 1 < instances {
                push_rows(&mut derived, &self.prev, local_column, instance + 1);
            }
            if instance == 0 {
                push_rows(&mut derived, &self.first, local_column, 0);
            }
            if instance == instances - 1 {
                push_rows(&mut derived, &self.last, local_column, instance);
            }
        }
        derived.sort_unstable();
        debug_assert!(
            derived.windows(2).all(|pair| pair[0] < pair[1]),
            "chained map entries are unique per source column"
        );
        Some(derived.into_iter())
    }

    fn chained_packed_source(&self) -> Option<ChainedPackedSourceParts<'_>> {
        Some(self.parts())
    }
}

impl PackedRepeatedVirtualMap {
    /// Builds a packed repetition inside power-of-two global domains.
    pub fn new(
        local: PreparedVirtualMap,
        instances: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, PreparedVirtualMapError> {
        if instances == 0
            || local.rows() == 0
            || local.cols() == 0
            || !rows.is_power_of_two()
            || !cols.is_power_of_two()
        {
            return Err(PreparedVirtualMapError::InvalidRepetition);
        }
        let live_rows = local
            .rows()
            .checked_sub(1)
            .and_then(|width| width.checked_mul(instances))
            .and_then(|cells| cells.checked_add(1))
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let live_cols = local
            .cols()
            .checked_sub(1)
            .and_then(|width| width.checked_mul(instances))
            .and_then(|cells| cells.checked_add(1))
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        if live_rows > rows || live_cols > cols {
            return Err(PreparedVirtualMapError::InvalidRepetition);
        }

        let constant_column = local
            .matrix()
            .column(0)
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let constant_nnz = constant_column
            .indices()
            .iter()
            .try_fold(0usize, |count, &row| {
                count.checked_add(if row == 0 { 1 } else { instances })
            })
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let repeated_nnz = local
            .matrix()
            .columns()
            .skip(1)
            .try_fold(0usize, |count, column| {
                count.checked_add(column.len().checked_mul(instances)?)
            })
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let nnz = constant_nnz
            .checked_add(repeated_nnz)
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;

        let mut hash = Hasher::new();
        hash.update(b"bitz/packed-repeated-virtual-map/v1");
        hash.update(&local.digest());
        for value in [instances, rows, cols, live_rows, live_cols, nnz] {
            hash.update(
                &u64::try_from(value)
                    .map_err(|_| PreparedVirtualMapError::ShapeTooLarge)?
                    .to_le_bytes(),
            );
        }
        let digest = *hash.finalize().as_bytes();

        Ok(Self {
            local,
            instances,
            rows,
            cols,
            live_rows,
            live_cols,
            nnz,
            digest,
        })
    }

    pub const fn local(&self) -> &PreparedVirtualMap {
        &self.local
    }

    pub const fn instances(&self) -> usize {
        self.instances
    }

    pub const fn live_rows(&self) -> usize {
        self.live_rows
    }

    pub const fn live_cols(&self) -> usize {
        self.live_cols
    }
}

/// Derived-row iterator for one packed source column.
pub struct PackedColumnRows<'a> {
    local_rows: &'a [usize],
    local_width: usize,
    instance: Option<usize>,
    local_index: usize,
    repeat_index: usize,
    remaining: usize,
}

impl Iterator for PackedColumnRows<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let output = match self.instance {
            Some(instance) => {
                let row = self.local_rows[self.local_index];
                self.local_index += 1;
                if row == 0 {
                    0
                } else {
                    1 + instance * self.local_width + (row - 1)
                }
            }
            None if self.local_rows[self.local_index] == 0 => {
                self.local_index += 1;
                0
            }
            None => {
                let row = self.local_rows[self.local_index];
                let output = 1 + self.repeat_index * self.local_width + (row - 1);
                self.local_index += 1;
                if self.local_index == self.local_rows.len() {
                    self.repeat_index += 1;
                    self.local_index = usize::from(self.local_rows.first() == Some(&0));
                }
                output
            }
        };
        self.remaining -= 1;
        Some(output)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for PackedColumnRows<'_> {}

impl VirtualMap for PackedRepeatedVirtualMap {
    type ColumnRows<'a> = PackedColumnRows<'a>;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn is_identity(&self) -> bool {
        self.instances == 1
            && self.live_rows == self.rows
            && self.live_cols == self.cols
            && self.local.is_identity()
    }

    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>> {
        if column >= self.cols {
            return None;
        }
        if column >= self.live_cols {
            return Some(PackedColumnRows {
                local_rows: &[],
                local_width: self.local.rows() - 1,
                instance: Some(0),
                local_index: 0,
                repeat_index: 0,
                remaining: 0,
            });
        }

        let (local_column, instance) = if column == 0 {
            (0, None)
        } else {
            let offset = column - 1;
            (
                1 + offset % (self.local.cols() - 1),
                Some(offset / (self.local.cols() - 1)),
            )
        };
        let local_rows = self.local.matrix().column(local_column)?.indices();
        let remaining = match instance {
            Some(_) => local_rows.len(),
            None => local_rows
                .iter()
                .map(|&row| if row == 0 { 1 } else { self.instances })
                .sum(),
        };
        Some(PackedColumnRows {
            local_rows,
            local_width: self.local.rows() - 1,
            instance,
            local_index: 0,
            repeat_index: 0,
            remaining,
        })
    }
}

impl RepeatedVirtualMap {
    /// Prepares an implicit tensor repetition of `local`.
    pub fn new(
        local: PreparedVirtualMap,
        instances: usize,
    ) -> Result<Self, PreparedVirtualMapError> {
        if instances == 0 {
            return Err(PreparedVirtualMapError::InvalidRepetition);
        }
        let rows = local
            .rows()
            .checked_mul(instances)
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let cols = local
            .cols()
            .checked_mul(instances)
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;
        let nnz = local
            .nnz()
            .checked_mul(instances)
            .ok_or(PreparedVirtualMapError::InvalidRepetition)?;

        let mut hash = Hasher::new();
        hash.update(b"bitz/repeated-virtual-map/v1");
        hash.update(&local.digest());
        for value in [instances, rows, cols, nnz] {
            hash.update(
                &u64::try_from(value)
                    .map_err(|_| PreparedVirtualMapError::ShapeTooLarge)?
                    .to_le_bytes(),
            );
        }
        let digest = *hash.finalize().as_bytes();

        Ok(Self {
            local,
            instances,
            instance_bits: instances.is_power_of_two().then(|| instances.ilog2()),
            rows,
            cols,
            nnz,
            digest,
        })
    }

    /// Local canonical CSC map repeated by this view.
    pub const fn local(&self) -> &PreparedVirtualMap {
        &self.local
    }

    /// Number of independent repetitions.
    pub const fn instances(&self) -> usize {
        self.instances
    }
}

/// Row iterator for one column of a [`RepeatedVirtualMap`].
pub struct RepeatedColumnRows<'a> {
    rows: slice::Iter<'a, usize>,
    instance: usize,
    instances: usize,
}

impl Iterator for RepeatedColumnRows<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.rows
            .next()
            .map(|row| row * self.instances + self.instance)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.rows.size_hint()
    }
}

impl ExactSizeIterator for RepeatedColumnRows<'_> {}

impl VirtualMap for RepeatedVirtualMap {
    type ColumnRows<'a> = RepeatedColumnRows<'a>;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn digest(&self) -> [u8; 32] {
        self.digest
    }

    fn is_identity(&self) -> bool {
        self.local.is_identity() && self.local.rows() == self.local.cols()
    }

    fn column_rows(&self, column: usize) -> Option<Self::ColumnRows<'_>> {
        if column >= self.cols {
            return None;
        }
        let (local_column, instance) = match self.instance_bits {
            Some(bits) => (column >> bits, column & (self.instances - 1)),
            None => (column / self.instances, column % self.instances),
        };
        let rows = self.local.matrix().column(local_column)?.indices().iter();
        Some(RepeatedColumnRows {
            rows,
            instance,
            instances: self.instances,
        })
    }

    fn repetition(&self) -> Option<(&PreparedVirtualMap, usize)> {
        Some((&self.local, self.instances))
    }
}

fn compute_identity(matrix: &CscMatrix<ImplicitOnes>) -> bool {
    matrix.row_count() == matrix.column_count()
        && matrix.nnz() == matrix.row_count()
        && matrix
            .columns()
            .enumerate()
            .all(|(column, entries)| matches!(entries.single(), Some((row, true)) if row == column))
}

fn compute_digest(matrix: &CscMatrix<ImplicitOnes>) -> Result<[u8; 32], PreparedVirtualMapError> {
    let mut hash = Hasher::new();
    hash.update(b"bitz/f2-cell-map/v1");
    for value in [matrix.row_count(), matrix.column_count(), matrix.nnz()] {
        hash.update(
            &u64::try_from(value)
                .map_err(|_| PreparedVirtualMapError::ShapeTooLarge)?
                .to_le_bytes(),
        );
    }
    // Preserve the element-by-element byte stream while avoiding one hasher
    // call per entry on production-sized maps.
    const CHUNK: usize = 1 << 20;
    let mut bytes = Vec::with_capacity(CHUNK * core::mem::size_of::<u64>());
    for values in [matrix.column_offsets(), matrix.row_indices()] {
        for block in values.chunks(CHUNK) {
            bytes.clear();
            for &value in block {
                bytes.extend_from_slice(
                    &u64::try_from(value)
                        .map_err(|_| PreparedVirtualMapError::ShapeTooLarge)?
                        .to_le_bytes(),
                );
            }
            hash.update_rayon(&bytes);
        }
    }
    Ok(*hash.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use crate::linear_map::CscMatrix;

    use super::*;

    fn prepared(rows: usize, columns: Vec<Vec<(usize, bool)>>) -> PreparedVirtualMap {
        PreparedVirtualMap::new(CscMatrix::try_from_columns(rows, columns).unwrap()).unwrap()
    }

    #[test]
    fn rejects_explicit_false_coefficients() {
        let matrix = CscMatrix::try_from_columns(2, vec![vec![(1, false)]]).unwrap();
        assert_eq!(
            PreparedVirtualMap::new(matrix),
            Err(PreparedVirtualMapError::ExplicitFalse { row: 1, column: 0 })
        );
    }

    #[test]
    fn identity_detection_is_exact() {
        assert!(prepared(2, vec![vec![(0, true)], vec![(1, true)]]).is_identity());
        assert!(!prepared(2, vec![vec![(1, true)], vec![(0, true)]]).is_identity());
        assert!(!prepared(2, vec![vec![(0, true)], vec![]]).is_identity());
        assert!(!prepared(2, vec![vec![(0, true)], vec![(1, true)], vec![]]).is_identity());
    }

    #[test]
    fn digest_is_canonical_and_content_sensitive() {
        let a = prepared(
            2,
            vec![vec![(0, true)], vec![(1, true)], vec![], vec![(0, true)]],
        );
        let b = prepared(
            2,
            vec![vec![(0, true)], vec![(1, true)], vec![], vec![(0, true)]],
        );
        let c = prepared(
            2,
            vec![vec![(0, true)], vec![(1, true)], vec![], vec![(1, true)]],
        );
        assert_eq!(
            a.digest(),
            [
                0x7e, 0xd6, 0x0a, 0x91, 0x83, 0x41, 0x5a, 0x44, 0x2b, 0x72, 0x69, 0xcb, 0x21, 0x38,
                0xf8, 0x7a, 0x01, 0xfe, 0x63, 0xa7, 0xe9, 0xff, 0x49, 0x70, 0x7d, 0x1d, 0x1a, 0x2c,
                0x9e, 0xba, 0xda, 0x4e,
            ]
        );
        assert_eq!(a.digest(), b.digest());
        assert_ne!(a.digest(), c.digest());
    }

    #[test]
    fn repeated_map_is_an_implicit_tensor_product() {
        let local = prepared(3, vec![vec![(0, true), (2, true)], vec![(1, true)]]);
        let repeated = RepeatedVirtualMap::new(local.clone(), 4).unwrap();
        assert_eq!(repeated.rows(), 12);
        assert_eq!(repeated.cols(), 8);
        assert_eq!(repeated.nnz(), 12);
        assert_eq!(repeated.instances(), 4);
        assert_eq!(
            repeated.column_rows(2).unwrap().collect::<Vec<_>>(),
            vec![2, 10]
        );
        assert_eq!(
            repeated.column_rows(7).unwrap().collect::<Vec<_>>(),
            vec![7]
        );
        assert!(repeated.column_rows(8).is_none());
        assert_ne!(repeated.digest(), local.digest());
        assert_ne!(
            repeated.digest(),
            RepeatedVirtualMap::new(local, 2).unwrap().digest()
        );
    }

    #[test]
    fn repeated_map_rejects_zero_instances_and_preserves_identity() {
        let identity = prepared(2, vec![vec![(0, true)], vec![(1, true)]]);
        assert_eq!(
            RepeatedVirtualMap::new(identity.clone(), 0),
            Err(PreparedVirtualMapError::InvalidRepetition)
        );
        assert!(RepeatedVirtualMap::new(identity, 3).unwrap().is_identity());
    }

    #[test]
    fn packed_repeated_map_shares_the_constant_and_has_only_tail_padding() {
        let local = prepared(
            4,
            vec![
                vec![(0, true), (2, true), (3, true)],
                vec![(1, true), (3, true)],
                vec![(2, true)],
            ],
        );
        let packed = PackedRepeatedVirtualMap::new(local.clone(), 3, 16, 8).unwrap();

        assert_eq!(packed.instances(), 3);
        assert_eq!(packed.live_rows(), 10);
        assert_eq!(packed.live_cols(), 7);
        assert_eq!(packed.rows(), 16);
        assert_eq!(packed.cols(), 8);
        assert_eq!(packed.nnz(), 16);
        assert_eq!(
            packed.column_rows(0).unwrap().collect::<Vec<_>>(),
            vec![0, 2, 3, 5, 6, 8, 9]
        );
        assert_eq!(
            packed.column_rows(3).unwrap().collect::<Vec<_>>(),
            vec![4, 6]
        );
        assert!(packed.column_rows(7).unwrap().next().is_none());
        assert!(packed.column_rows(8).is_none());
        assert_ne!(packed.digest(), local.digest());
        assert_ne!(
            packed.digest(),
            PackedRepeatedVirtualMap::new(local, 2, 8, 8)
                .unwrap()
                .digest()
        );
    }

    #[test]
    fn packed_source_repeated_map_uses_a_local_major_derived_tensor() {
        let local = prepared(
            4,
            vec![
                vec![(0, true), (2, true), (3, true)],
                vec![(1, true), (3, true)],
                vec![(2, true)],
            ],
        );
        let repeated = PackedSourceRepeatedVirtualMap::new(local.clone(), 4, 16, 16).unwrap();

        assert_eq!(repeated.instances(), 4);
        assert_eq!(repeated.live_rows(), 16);
        assert_eq!(repeated.live_cols(), 9);
        assert_eq!(repeated.rows(), 16);
        assert_eq!(repeated.cols(), 16);
        assert_eq!(repeated.nnz(), 24);
        assert_eq!(
            repeated.column_rows(0).unwrap().collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
        );
        // Packed source column 3 is instance one, local source column one.
        assert_eq!(
            repeated.column_rows(3).unwrap().collect::<Vec<_>>(),
            vec![5, 13]
        );
        assert!(repeated.column_rows(9).unwrap().next().is_none());
        assert!(repeated.column_rows(16).is_none());
        assert_ne!(repeated.digest(), local.digest());
    }

    #[test]
    fn chained_packed_source_map_links_instances_and_boundaries() {
        // 4 local rows × (constant + 2 cells); 4 instances; rows 16, cols 16.
        let local = prepared(4, vec![vec![(0, true)], vec![(1, true)], vec![(2, true)]]);
        // Row 1 of instance i reads cell 2 of instance i - 1.
        let prev = prepared(4, vec![vec![], vec![], vec![(1, true)]]);
        // Instance 0's row 1 also reads the constant (an initial-state bit).
        let first = prepared(4, vec![vec![(1, true)], vec![], vec![]]);
        // Instance 3 exposes cell 2 in row 3.
        let last = prepared(4, vec![vec![], vec![], vec![(3, true)]]);
        let map = ChainedPackedSourceMap::new(
            local.clone(),
            prev.clone(),
            first.clone(),
            last.clone(),
            4,
            16,
            16,
        )
        .unwrap();
        assert_eq!(map.instances(), 4);
        assert_eq!(map.live_rows(), 16);
        assert_eq!(map.live_cols(), 9);
        assert_eq!(map.nnz(), 3 * 4 + 3 + 1 + 1);
        assert!(!map.is_identity());
        // Constant column: row 0 of every instance, plus instance 0's row 1.
        assert_eq!(
            map.column_rows(0).unwrap().collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        // Instance 1, cell 1 (global column 3): row 1 of instance 1 only.
        assert_eq!(map.column_rows(3).unwrap().collect::<Vec<_>>(), vec![5]);
        // Instance 1, cell 2 (global column 4): its own row 2, and row 1 of
        // instance 2 through the chain link.
        assert_eq!(map.column_rows(4).unwrap().collect::<Vec<_>>(), vec![6, 9]);
        // Instance 3, cell 2 (global column 8): own row 2, the terminal row
        // 3 through `last`, and no next instance.
        assert_eq!(
            map.column_rows(8).unwrap().collect::<Vec<_>>(),
            vec![11, 15]
        );
        assert!(map.column_rows(9).unwrap().next().is_none());
        assert!(map.column_rows(16).is_none());
        assert_eq!(
            ChainedPackedSourceMap::nonconstant_column_span(&prev),
            (1, 2)
        );
        assert_eq!(
            ChainedPackedSourceMap::nonconstant_column_span(&first),
            (0, 0)
        );

        // A boundary entry duplicating a `local` entry would cancel: rejected.
        let clashing_first = prepared(4, vec![vec![(0, true)], vec![], vec![]]);
        assert_eq!(
            ChainedPackedSourceMap::new(
                local.clone(),
                prev.clone(),
                clashing_first,
                last.clone(),
                4,
                16,
                16
            ),
            Err(PreparedVirtualMapError::InvalidRepetition)
        );
        // The chain link may not read the constant, and one instance is no chain.
        let constant_prev = prepared(4, vec![vec![(1, true)], vec![], vec![]]);
        assert_eq!(
            ChainedPackedSourceMap::new(
                local.clone(),
                constant_prev,
                first.clone(),
                last.clone(),
                4,
                16,
                16
            ),
            Err(PreparedVirtualMapError::InvalidRepetition)
        );
        assert_eq!(
            ChainedPackedSourceMap::new(local, prev, first, last, 1, 16, 16),
            Err(PreparedVirtualMapError::InvalidRepetition)
        );
    }

    #[test]
    fn packed_source_repeated_map_instance_major_order_strides_by_padded_local_rows() {
        // Three local rows pad to a stride of four; four instances.
        let local = prepared(
            3,
            vec![
                vec![(0, true), (2, true)],
                vec![(1, true), (2, true)],
                vec![(0, true)],
            ],
        );
        let local_major = PackedSourceRepeatedVirtualMap::new(local.clone(), 4, 16, 16).unwrap();
        let repeated = PackedSourceRepeatedVirtualMap::new_with_order(
            local.clone(),
            4,
            16,
            16,
            PackedSourceOrder::InstanceMajor,
        )
        .unwrap();

        assert_eq!(repeated.order(), PackedSourceOrder::InstanceMajor);
        assert_eq!(repeated.local_stride(), 4);
        assert_eq!(local_major.local_stride(), 4);
        assert_eq!(repeated.live_rows(), 16);
        assert_eq!(repeated.live_cols(), 9);
        assert_eq!(repeated.nnz(), 20);
        // The constant column touches local rows {0, 2} of every instance,
        // ascending with the local row innermost.
        assert_eq!(
            repeated.column_rows(0).unwrap().collect::<Vec<_>>(),
            vec![0, 2, 4, 6, 8, 10, 12, 14]
        );
        // Packed source column 3 is instance one, local source column one.
        assert_eq!(
            repeated.column_rows(3).unwrap().collect::<Vec<_>>(),
            vec![5, 6]
        );
        assert_eq!(
            repeated.column_rows(8).unwrap().collect::<Vec<_>>(),
            vec![12]
        );
        assert!(repeated.column_rows(9).unwrap().next().is_none());
        assert!(repeated.column_rows(16).is_none());
        // The order is bound by the digest; local-major bytes are untouched.
        assert_ne!(repeated.digest(), local_major.digest());
        assert_eq!(
            local_major.digest(),
            PackedSourceRepeatedVirtualMap::new_with_order(
                local,
                4,
                16,
                16,
                PackedSourceOrder::LocalMajor
            )
            .unwrap()
            .digest()
        );
    }
}
