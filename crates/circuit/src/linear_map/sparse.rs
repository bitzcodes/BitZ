//! Canonical CSR/CSC storage. Arithmetic is independent of coefficient storage.
use super::{CoefficientStore, StoreCoefficient};
use crate::integer_storage::IntegerTable;
use std::fmt::Debug;
use thiserror::Error;

/// Failures while constructing a canonical sparse matrix.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum SparseMatrixError {
    /// A sparse entry refers to a column outside the declared matrix width.
    #[error("column {column} in row {row} is outside a {columns}-column matrix")]
    ColumnOutOfBounds {
        row: usize,
        column: usize,
        columns: usize,
    },

    /// Sparse rows must have a unique canonical order.
    #[error("columns in row {row} are not strictly increasing: {previous}, then {column}")]
    ColumnsNotStrictlyIncreasing {
        row: usize,
        previous: usize,
        column: usize,
    },

    /// A sparse entry refers to a row outside the declared matrix height.
    #[error("row {row} in column {column} is outside a {rows}-row matrix")]
    RowOutOfBounds {
        column: usize,
        row: usize,
        rows: usize,
    },

    /// CSC columns must have a unique canonical order.
    #[error("rows in column {column} are not strictly increasing: {previous}, then {row}")]
    RowsNotStrictlyIncreasing {
        column: usize,
        previous: usize,
        row: usize,
    },

    /// Raw CSC offsets must start at zero, be nondecreasing, and end at the
    /// number of stored entries.
    #[error("invalid CSC column offsets")]
    InvalidCscOffsets,

    /// CSC row-index and coefficient arrays must have equal lengths.
    #[error("CSC row-index and coefficient arrays have different lengths")]
    InvalidCscEntryCount,
    #[error("sparse index does not fit the selected index width")]
    IndexOverflow,
    #[error("coefficient index outside its storage")]
    CoefficientOutOfBounds,
}

mod sealed {
    pub trait Index {}
    impl Index for usize {}
    impl Index for u32 {}
}
/// Index widths supported by compressed sparse storage.
pub trait SparseIndex: sealed::Index + Copy + Debug + Eq + Send + Sync {
    fn to_usize(self) -> usize;
    fn from_usize(value: usize) -> Result<Self, SparseMatrixError>;
}
impl SparseIndex for usize {
    fn to_usize(self) -> usize {
        self
    }
    fn from_usize(value: usize) -> Result<Self, SparseMatrixError> {
        Ok(value)
    }
}
impl SparseIndex for u32 {
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(value: usize) -> Result<Self, SparseMatrixError> {
        value
            .try_into()
            .map_err(|_| SparseMatrixError::IndexOverflow)
    }
}

/// Coefficient-free storage for entries equal to one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImplicitOnes {
    len: usize,
}
impl ImplicitOnes {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}
impl CoefficientStore for ImplicitOnes {
    type Ref<'a> = &'a bool;
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> &bool {
        assert!(index < self.len);
        &true
    }
    fn distinct_len(&self) -> usize {
        usize::from(self.len != 0)
    }
    fn coefficient_index(&self, _: usize) -> usize {
        0
    }
}
impl StoreCoefficient<bool> for ImplicitOnes {
    fn store(&mut self, value: bool) -> usize {
        assert!(value, "implicit-one storage cannot contain false");
        let index = self.len;
        self.len += 1;
        index
    }
}

/// Entry-aligned indices into a shared, possibly heterogeneous coefficient store.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedCoefficients<S, I = u32> {
    values: S,
    indices: Box<[I]>,
}
impl<S: CoefficientStore, I: SparseIndex> IndexedCoefficients<S, I> {
    pub fn new(values: S, indices: Vec<I>) -> Result<Self, SparseMatrixError> {
        if indices.iter().any(|i| i.to_usize() >= values.len()) {
            return Err(SparseMatrixError::CoefficientOutOfBounds);
        }
        Ok(Self {
            values,
            indices: indices.into_boxed_slice(),
        })
    }
    pub fn values(&self) -> &S {
        &self.values
    }
    pub fn indices(&self) -> &[I] {
        &self.indices
    }
}
impl<S: CoefficientStore, I: SparseIndex> CoefficientStore for IndexedCoefficients<S, I> {
    type Ref<'a>
        = S::Ref<'a>
    where
        Self: 'a;
    fn len(&self) -> usize {
        self.indices.len()
    }
    fn get(&self, index: usize) -> Self::Ref<'_> {
        self.values.get(self.indices[index].to_usize())
    }
    fn distinct_len(&self) -> usize {
        self.values.distinct_len()
    }
    fn distinct_coefficient(&self, index: usize) -> Self::Ref<'_> {
        self.values.distinct_coefficient(index)
    }
    fn coefficient_index(&self, entry: usize) -> usize {
        self.values
            .coefficient_index(self.indices[entry].to_usize())
    }
}

/// Reordering preserves declared coefficient widths; it never performs arithmetic.
pub trait PermuteCoefficients: CoefficientStore + Sized {
    /// Constant storage has no entry positions to permute.
    const REORDER_ENTRIES: bool = true;
    fn permute(self, order: &[usize]) -> Self;
}
impl<C> PermuteCoefficients for Box<[C]> {
    fn permute(self, order: &[usize]) -> Self {
        let mut values = self.into_vec();
        assert_eq!(values.len(), order.len(), "entry permutation length");
        let mut destination = vec![usize::MAX; order.len()];
        for (to, &from) in order.iter().enumerate() {
            assert!(
                from < destination.len() && destination[from] == usize::MAX,
                "entry permutation must be bijective"
            );
            destination[from] = to;
        }
        for i in 0..destination.len() {
            while destination[i] != i {
                let j = destination[i];
                values.swap(i, j);
                destination.swap(i, j);
            }
        }
        values.into_boxed_slice()
    }
}
impl PermuteCoefficients for IntegerTable {
    fn permute(self, order: &[usize]) -> Self {
        let mut out = Self::default();
        for &i in order {
            self.copy_row_to(i, &mut out);
        }
        out
    }
}
impl PermuteCoefficients for ImplicitOnes {
    const REORDER_ENTRIES: bool = false;
    fn permute(self, _: &[usize]) -> Self {
        self
    }
}
impl<S: CoefficientStore, I: SparseIndex> PermuteCoefficients for IndexedCoefficients<S, I> {
    fn permute(self, order: &[usize]) -> Self {
        let indices = order.iter().map(|&i| self.indices[i]).collect();
        Self {
            values: self.values,
            indices,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompressedTopology<I> {
    major_offsets: Box<[I]>,
    minor_indices: Box<[I]>,
    minor_dimension: usize,
}
impl<I: SparseIndex> CompressedTopology<I> {
    fn new(
        minor_dimension: usize,
        offsets: Vec<I>,
        indices: Vec<I>,
        csr: bool,
    ) -> Result<Self, SparseMatrixError> {
        if offsets.first().map(|i| i.to_usize()) != Some(0)
            || offsets.last().map(|i| i.to_usize()) != Some(indices.len())
            || offsets
                .windows(2)
                .any(|w| w[0].to_usize() > w[1].to_usize())
        {
            return Err(SparseMatrixError::InvalidCscOffsets);
        }
        for (major, w) in offsets.windows(2).enumerate() {
            let mut previous = None;
            for &index in &indices[w[0].to_usize()..w[1].to_usize()] {
                let minor = index.to_usize();
                if minor >= minor_dimension {
                    return Err(if csr {
                        SparseMatrixError::ColumnOutOfBounds {
                            row: major,
                            column: minor,
                            columns: minor_dimension,
                        }
                    } else {
                        SparseMatrixError::RowOutOfBounds {
                            column: major,
                            row: minor,
                            rows: minor_dimension,
                        }
                    });
                }
                if let Some(p) = previous {
                    if p >= minor {
                        return Err(if csr {
                            SparseMatrixError::ColumnsNotStrictlyIncreasing {
                                row: major,
                                previous: p,
                                column: minor,
                            }
                        } else {
                            SparseMatrixError::RowsNotStrictlyIncreasing {
                                column: major,
                                previous: p,
                                row: minor,
                            }
                        });
                    }
                }
                previous = Some(minor);
            }
        }
        Ok(Self {
            minor_dimension,
            major_offsets: offsets.into_boxed_slice(),
            minor_indices: indices.into_boxed_slice(),
        })
    }
    const fn major_count(&self) -> usize {
        (&*self.major_offsets).len() - 1
    }
    fn segment<'a, S: CoefficientStore>(
        &'a self,
        coefficients: &'a S,
        index: usize,
    ) -> Option<SparseSegment<'a, S, I>> {
        let start = self.major_offsets.as_ref().get(index)?.to_usize();
        let end = self
            .major_offsets
            .as_ref()
            .get(index.checked_add(1)?)?
            .to_usize();
        Some(SparseSegment {
            indices: &self.minor_indices[start..end],
            coefficients,
            start,
        })
    }
    /// Stable counting scatter, preserving each coefficient's entry position.
    fn regroup(&self, reorder_entries: bool) -> Result<(Self, Vec<usize>), SparseMatrixError> {
        let mut offsets = vec![0usize; self.minor_dimension + 1];
        for &i in &self.minor_indices {
            offsets[i.to_usize() + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut next = offsets.clone();
        let mut indices = vec![I::from_usize(0)?; self.minor_indices.len()];
        let mut order = if reorder_entries {
            vec![0; indices.len()]
        } else {
            Vec::new()
        };
        for (major, w) in self.major_offsets.windows(2).enumerate() {
            for entry in w[0].to_usize()..w[1].to_usize() {
                let minor = self.minor_indices[entry].to_usize();
                let at = next[minor];
                next[minor] += 1;
                indices[at] = I::from_usize(major)?;
                if reorder_entries {
                    order[at] = entry;
                }
            }
        }
        Ok((
            Self {
                minor_dimension: self.major_count(),
                major_offsets: offsets
                    .into_iter()
                    .map(I::from_usize)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
                minor_indices: indices.into_boxed_slice(),
            },
            order,
        ))
    }
}

/// Borrowed canonical row or column; coefficients are resolved by their store.
pub struct SparseSegment<'a, S: CoefficientStore + ?Sized, I = usize> {
    indices: &'a [I],
    coefficients: &'a S,
    start: usize,
}
impl<S: CoefficientStore + ?Sized, I> Copy for SparseSegment<'_, S, I> {}
impl<S: CoefficientStore + ?Sized, I> Clone for SparseSegment<'_, S, I> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, S: CoefficientStore + ?Sized, I: SparseIndex> SparseSegment<'a, S, I> {
    pub fn indices(self) -> &'a [I] {
        self.indices
    }
    pub fn len(self) -> usize {
        self.indices.len()
    }
    pub fn is_empty(self) -> bool {
        self.indices.is_empty()
    }
    pub fn entry_range(self) -> std::ops::Range<usize> {
        self.start..self.start + self.len()
    }
    pub fn single(self) -> Option<(usize, S::Ref<'a>)> {
        (self.len() == 1).then(|| {
            (
                self.indices[0].to_usize(),
                self.coefficients.get(self.start),
            )
        })
    }
    pub fn iter(self) -> SparseEntries<'a, S, I> {
        SparseEntries {
            segment: self,
            position: 0,
        }
    }
}
impl<'a, C, I: SparseIndex> SparseSegment<'a, Box<[C]>, I> {
    pub fn coefficients(self) -> &'a [C] {
        &self.coefficients[self.start..self.start + self.len()]
    }
}
impl<'a, S: CoefficientStore, I: SparseIndex, J: SparseIndex>
    SparseSegment<'a, IndexedCoefficients<S, J>, I>
{
    /// Entry locations paired with shared-table indices, without resolving values.
    pub fn indexed_entries(self) -> impl ExactSizeIterator<Item = (usize, usize)> + 'a {
        self.indices
            .iter()
            .zip(&self.coefficients.indices[self.entry_range()])
            .map(|(&index, &coefficient)| (index.to_usize(), coefficient.to_usize()))
    }
}
pub struct SparseEntries<'a, S: CoefficientStore + ?Sized, I> {
    segment: SparseSegment<'a, S, I>,
    position: usize,
}
impl<'a, S: CoefficientStore + ?Sized, I: SparseIndex> Iterator for SparseEntries<'a, S, I> {
    type Item = (usize, S::Ref<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        let &index = self.segment.indices.get(self.position)?;
        let coefficient = self
            .segment
            .coefficients
            .get(self.segment.start + self.position);
        self.position += 1;
        Some((index.to_usize(), coefficient))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.segment.len() - self.position;
        (n, Some(n))
    }
}
impl<S: CoefficientStore + ?Sized, I: SparseIndex> ExactSizeIterator for SparseEntries<'_, S, I> {}
impl<'a, S: CoefficientStore + ?Sized, I: SparseIndex> IntoIterator for SparseSegment<'a, S, I> {
    type Item = (usize, S::Ref<'a>);
    type IntoIter = SparseEntries<'a, S, I>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Canonical compressed sparse rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<S, I = usize> {
    topology: CompressedTopology<I>,
    coefficients: S,
}
/// Canonical compressed sparse columns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CscMatrix<S, I = usize> {
    topology: CompressedTopology<I>,
    coefficients: S,
}

macro_rules! matrix {
    ($name:ident,$csr:expr,$major_count:ident,$minor_count:ident,$offsets:ident,$indices:ident,$segment:ident,$segments:ident) => {
        impl<S: CoefficientStore, I: SparseIndex> $name<S, I> {
            pub fn try_from_parts(
                minor_dimension: usize,
                offsets: Vec<I>,
                indices: Vec<I>,
                coefficients: S,
            ) -> Result<Self, SparseMatrixError> {
                if indices.len() != coefficients.len() {
                    return Err(SparseMatrixError::InvalidCscEntryCount);
                }
                Ok(Self {
                    topology: CompressedTopology::new(minor_dimension, offsets, indices, $csr)?,
                    coefficients,
                })
            }
            pub const fn $major_count(&self) -> usize {
                self.topology.major_count()
            }
            pub const fn $minor_count(&self) -> usize {
                self.topology.minor_dimension
            }
            pub const fn nnz(&self) -> usize {
                (&*self.topology.minor_indices).len()
            }
            pub fn $offsets(&self) -> &[I] {
                &self.topology.major_offsets
            }
            pub fn $indices(&self) -> &[I] {
                &self.topology.minor_indices
            }
            pub fn coefficients(&self) -> &S {
                &self.coefficients
            }
            pub fn $segment(&self, index: usize) -> Option<SparseSegment<'_, S, I>> {
                self.topology.segment(&self.coefficients, index)
            }
            pub fn $segments(&self) -> impl ExactSizeIterator<Item = SparseSegment<'_, S, I>> {
                (0..self.$major_count()).map(|i| self.$segment(i).unwrap())
            }
            pub fn with_coefficients<T: CoefficientStore>(
                self,
                coefficients: T,
            ) -> Result<$name<T, I>, SparseMatrixError> {
                if coefficients.len() != self.nnz() {
                    return Err(SparseMatrixError::InvalidCscEntryCount);
                }
                Ok($name {
                    topology: self.topology,
                    coefficients,
                })
            }
            pub fn map_coefficients<T: CoefficientStore>(
                self,
                map: impl FnOnce(S) -> T,
            ) -> Result<$name<T, I>, SparseMatrixError> {
                let coefficients = map(self.coefficients);
                if coefficients.len() != self.topology.minor_indices.len() {
                    return Err(SparseMatrixError::InvalidCscEntryCount);
                }
                Ok($name {
                    topology: self.topology,
                    coefficients,
                })
            }
            pub fn into_coefficients(self) -> S {
                self.coefficients
            }
            pub fn topology_bytes(&self) -> usize {
                std::mem::size_of_val(&*self.topology.major_offsets)
                    + std::mem::size_of_val(&*self.topology.minor_indices)
            }
        }
    };
}
matrix!(
    CsrMatrix,
    true,
    row_count,
    column_count,
    row_offsets,
    column_indices,
    row,
    rows
);
matrix!(
    CscMatrix,
    false,
    column_count,
    row_count,
    column_offsets,
    row_indices,
    column,
    columns
);
impl<S, I> CsrMatrix<S, I> {
    /// O(1) mathematical transpose; does not reorder entries.
    pub fn transpose(self) -> CscMatrix<S, I> {
        CscMatrix {
            topology: self.topology,
            coefficients: self.coefficients,
        }
    }
}
impl<S, I> CscMatrix<S, I> {
    pub fn transpose(self) -> CsrMatrix<S, I> {
        CsrMatrix {
            topology: self.topology,
            coefficients: self.coefficients,
        }
    }
}
impl<S: PermuteCoefficients, I: SparseIndex> CsrMatrix<S, I> {
    pub fn into_csc(self) -> Result<CscMatrix<S, I>, SparseMatrixError> {
        let (topology, order) = self.topology.regroup(S::REORDER_ENTRIES)?;
        Ok(CscMatrix {
            topology,
            coefficients: self.coefficients.permute(&order),
        })
    }
}
impl<S: PermuteCoefficients, I: SparseIndex> CscMatrix<S, I> {
    pub fn into_csr(self) -> Result<CsrMatrix<S, I>, SparseMatrixError> {
        let (topology, order) = self.topology.regroup(S::REORDER_ENTRIES)?;
        Ok(CsrMatrix {
            topology,
            coefficients: self.coefficients.permute(&order),
        })
    }
}
impl<C, I: SparseIndex> CscMatrix<Box<[C]>, I> {
    pub fn borrowed_csr(&self) -> Result<CsrMatrix<Box<[&C]>, I>, SparseMatrixError> {
        let rows = self.row_count();
        let mut offsets = vec![I::from_usize(0)?; rows + 1];
        for &row in self.row_indices() {
            let slot = &mut offsets[row.to_usize() + 1];
            *slot = I::from_usize(slot.to_usize() + 1)?;
        }
        for i in 0..rows {
            offsets[i + 1] = I::from_usize(offsets[i].to_usize() + offsets[i + 1].to_usize())?;
        }
        let mut next = offsets.clone();
        let mut indices = vec![I::from_usize(0)?; self.nnz()];
        let mut coefficients = self
            .coefficients
            .first()
            .map_or_else(Vec::new, |c| vec![c; self.nnz()]);
        for (column, entries) in self.columns().enumerate() {
            for (row, c) in entries {
                let at = next[row].to_usize();
                next[row] = I::from_usize(at + 1)?;
                indices[at] = I::from_usize(column)?;
                coefficients[at] = c;
            }
        }
        CsrMatrix::try_from_parts(
            self.column_count(),
            offsets,
            indices,
            coefficients.into_boxed_slice(),
        )
    }
    /// Construct from a repeatable row source without an owned CSR staging table.
    /// Rows are visited twice in index order. Each visit must return the same canonical row.
    pub fn try_from_row_source<'a>(
        rows: usize,
        columns: usize,
        source: impl Fn(usize) -> SparseSegment<'a, Box<[C]>, I>,
    ) -> Result<Self, SparseMatrixError>
    where
        C: Clone + 'a,
        I: 'a,
    {
        let mut offsets = vec![I::from_usize(0)?; columns + 1];
        let mut first = None;
        for row in 0..rows {
            for (column, c) in source(row) {
                if column >= columns {
                    return Err(SparseMatrixError::ColumnOutOfBounds {
                        row,
                        column,
                        columns,
                    });
                }
                let slot = &mut offsets[column + 1];
                *slot = I::from_usize(slot.to_usize() + 1)?;
                if first.is_none() {
                    first = Some(c);
                }
            }
        }
        for j in 0..columns {
            offsets[j + 1] = I::from_usize(offsets[j].to_usize() + offsets[j + 1].to_usize())?;
        }
        let nnz = offsets[columns].to_usize();
        let mut next = offsets.clone();
        let mut indices = vec![I::from_usize(0)?; nnz];
        let mut coefficients = first.map_or_else(Vec::new, |c| vec![c.clone(); nnz]);
        for row in 0..rows {
            for (column, c) in source(row) {
                if column >= columns {
                    return Err(SparseMatrixError::ColumnOutOfBounds {
                        row,
                        column,
                        columns,
                    });
                }
                let at = next[column].to_usize();
                if at >= offsets[column + 1].to_usize() {
                    return Err(SparseMatrixError::InvalidCscEntryCount);
                }
                next[column] = I::from_usize(at + 1)?;
                indices[at] = I::from_usize(row)?;
                coefficients[at] = c.clone();
            }
        }
        if next[..columns] != offsets[1..] {
            return Err(SparseMatrixError::InvalidCscEntryCount);
        }
        Self::try_from_csc_parts(rows, offsets, indices, coefficients)
    }
    pub fn try_from_rows(
        columns: usize,
        rows: Vec<Vec<(usize, C)>>,
    ) -> Result<Self, SparseMatrixError> {
        CsrMatrix::<Box<[C]>, I>::try_from_rows(columns, rows)?.into_csc()
    }
    pub fn try_from_columns(
        rows: usize,
        columns: Vec<Vec<(usize, C)>>,
    ) -> Result<Self, SparseMatrixError> {
        let nnz = columns.iter().map(Vec::len).sum();
        let mut offsets = Vec::with_capacity(columns.len() + 1);
        offsets.push(I::from_usize(0)?);
        let mut indices = Vec::with_capacity(nnz);
        let mut coefficients = Vec::with_capacity(nnz);
        for column in columns {
            for (row, c) in column {
                indices.push(I::from_usize(row)?);
                coefficients.push(c);
            }
            offsets.push(I::from_usize(indices.len())?);
        }
        Self::try_from_csc_parts(rows, offsets, indices, coefficients)
    }
    pub fn try_from_csc_parts(
        rows: usize,
        offsets: Vec<I>,
        indices: Vec<I>,
        coefficients: Vec<C>,
    ) -> Result<Self, SparseMatrixError> {
        Self::try_from_parts(rows, offsets, indices, coefficients.into_boxed_slice())
    }
    pub fn try_from_csc(
        rows: usize,
        offsets: Vec<I>,
        entries: Vec<(usize, C)>,
    ) -> Result<Self, SparseMatrixError> {
        let (indices, coefficients): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        Self::try_from_csc_parts(
            rows,
            offsets,
            indices
                .into_iter()
                .map(I::from_usize)
                .collect::<Result<_, _>>()?,
            coefficients,
        )
    }
}
impl<C, I: SparseIndex> CsrMatrix<Box<[C]>, I> {
    pub fn try_from_rows(
        columns: usize,
        rows: Vec<Vec<(usize, C)>>,
    ) -> Result<Self, SparseMatrixError> {
        let nnz = rows.iter().map(Vec::len).sum();
        let mut builder = CsrBuilder::<Vec<C>, I>::new(Vec::with_capacity(nnz));
        builder.offsets.reserve(rows.len());
        builder.indices.reserve(nnz);
        for row in rows {
            builder.push_row(row)?;
        }
        let matrix = builder.finish(columns)?;
        Ok(Self {
            topology: matrix.topology,
            coefficients: matrix.coefficients.into_boxed_slice(),
        })
    }
}
impl<I: SparseIndex> CscMatrix<ImplicitOnes, I> {
    pub fn try_from_binary_csc(
        rows: usize,
        offsets: Vec<I>,
        indices: Vec<I>,
    ) -> Result<Self, SparseMatrixError> {
        let n = indices.len();
        Self::try_from_parts(rows, offsets, indices, ImplicitOnes::new(n))
    }
    pub fn try_from_binary_rows(
        columns: usize,
        rows: Vec<Vec<usize>>,
    ) -> Result<Self, SparseMatrixError> {
        let mut builder = CsrBuilder::<ImplicitOnes, I>::new(ImplicitOnes::default());
        builder.offsets.reserve(rows.len());
        builder.indices.reserve(rows.iter().map(Vec::len).sum());
        for row in rows {
            builder.push_row(row.into_iter().map(|i| (i, true)))?;
        }
        builder.finish(columns)?.into_csc()
    }
    pub fn column_rows(&self, column: usize) -> Option<&[I]> {
        self.column(column).map(|c| c.indices())
    }
}

/// Append canonical rows before the final column dimension is known.
#[derive(Clone, Debug)]
pub struct CsrBuilder<S, I = usize> {
    coefficients: S,
    offsets: Vec<I>,
    indices: Vec<I>,
    failed: bool,
}
impl<S: CoefficientStore, I: SparseIndex> CsrBuilder<S, I> {
    pub fn new(coefficients: S) -> Self {
        assert!(coefficients.is_empty());
        Self {
            coefficients,
            offsets: vec![I::from_usize(0).unwrap()],
            indices: Vec::new(),
            failed: false,
        }
    }
    pub fn push_row<C>(
        &mut self,
        entries: impl IntoIterator<Item = (usize, C)>,
    ) -> Result<(), SparseMatrixError>
    where
        S: StoreCoefficient<C>,
    {
        if self.failed {
            return Err(SparseMatrixError::InvalidCscOffsets);
        }
        self.failed = true;
        let mut previous = None;
        for (column, coefficient) in entries {
            if let Some(p) = previous {
                if p >= column {
                    return Err(SparseMatrixError::ColumnsNotStrictlyIncreasing {
                        row: self.offsets.len() - 1,
                        previous: p,
                        column,
                    });
                }
            }
            previous = Some(column);
            self.indices.push(I::from_usize(column)?);
            self.coefficients.store(coefficient);
        }
        self.offsets.push(I::from_usize(self.indices.len())?);
        self.failed = false;
        Ok(())
    }
    pub fn finish(self, columns: usize) -> Result<CsrMatrix<S, I>, SparseMatrixError> {
        if self.failed {
            return Err(SparseMatrixError::InvalidCscOffsets);
        }
        CsrMatrix::try_from_parts(columns, self.offsets, self.indices, self.coefficients)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_produce_the_same_compact_csc() {
        let rows = vec![vec![(0, 2u8), (3, 3)], vec![], vec![(0, 5), (2, 7)]];
        let from_rows = CscMatrix::<Box<[u8]>>::try_from_rows(5, rows).unwrap();
        let from_columns = CscMatrix::<Box<[u8]>>::try_from_columns(
            3,
            vec![
                vec![(0, 2), (2, 5)],
                vec![],
                vec![(2, 7)],
                vec![(0, 3)],
                vec![],
            ],
        )
        .unwrap();
        let from_parts = CscMatrix::<Box<[u8]>>::try_from_csc_parts(
            3,
            vec![0, 2, 2, 3, 4, 4],
            vec![0, 2, 2, 0],
            vec![2, 5, 7, 3],
        )
        .unwrap();

        assert_eq!(from_rows, from_columns);
        assert_eq!(from_rows, from_parts);
        let first = from_rows.column(0).unwrap();
        assert_eq!(first.indices(), &[0, 2]);
        assert_eq!(first.coefficients(), &[2, 5]);
        assert_eq!(from_rows.column(4).unwrap().len(), 0);
    }

    #[test]
    fn malformed_csc_is_rejected() {
        assert_eq!(
            CscMatrix::<Box<[u8]>>::try_from_csc_parts(2, vec![], vec![], vec![]),
            Err(SparseMatrixError::InvalidCscOffsets)
        );
        assert_eq!(
            CscMatrix::<Box<[u8]>>::try_from_csc_parts(2, vec![0, 1], vec![0], Vec::<u8>::new()),
            Err(SparseMatrixError::InvalidCscEntryCount)
        );
        assert!(matches!(
            CscMatrix::<ImplicitOnes>::try_from_binary_csc(2, vec![0, 2], vec![1, 0]),
            Err(SparseMatrixError::RowsNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn empty_dimensions_and_empty_columns_are_retained() {
        let empty = CscMatrix::<Box<[u8]>>::try_from_csc_parts(0, vec![0], vec![], vec![]).unwrap();
        assert_eq!(empty.row_count(), 0);
        assert_eq!(empty.column_count(), 0);
        assert_eq!(empty.nnz(), 0);

        let columns =
            CscMatrix::<ImplicitOnes>::try_from_binary_csc(0, vec![0, 0, 0], vec![]).unwrap();
        assert_eq!(columns.row_count(), 0);
        assert_eq!(columns.column_count(), 2);
        assert!(columns.columns().all(SparseSegment::is_empty));
    }

    #[test]
    fn conversion_is_not_transposition_and_preserves_declared_widths() {
        let mut rows = CsrBuilder::<IntegerTable, u32>::new(IntegerTable::default());
        rows.push_row([(0, field::Z::<1>::from(3i64)), (2, field::Z::<1>::MIN)])
            .unwrap();
        rows.push_row([(1, field::Z::<4>::from(7i64))]).unwrap();
        let csr = rows.finish(3).unwrap();
        let transposed = csr.clone().transpose();
        assert_eq!((transposed.row_count(), transposed.column_count()), (3, 2));
        let csc = csr.clone().into_csc().unwrap();
        assert_eq!((csc.row_count(), csc.column_count()), (2, 3));
        assert_eq!(
            csc.coefficients()
                .iter()
                .map(|w| w.len())
                .collect::<Vec<_>>(),
            [1, 4, 1]
        );
        assert_eq!(csc.into_csr().unwrap(), csr);
    }

    #[test]
    fn compact_overflow_invalid_offsets_and_failed_builder_are_rejected() {
        assert_eq!(
            u32::from_usize(u32::MAX as usize + 1),
            Err(SparseMatrixError::IndexOverflow)
        );
        assert!(CscMatrix::<ImplicitOnes>::try_from_binary_csc(1, vec![0, 9, 1], vec![0]).is_err());
        assert!(CscMatrix::<ImplicitOnes>::try_from_binary_csc(1, vec![0, 1], vec![1]).is_err());
        let mut rows = CsrBuilder::<Vec<u64>, u32>::new(Vec::new());
        assert!(rows.push_row([(1, 2), (0, 3)]).is_err());
        assert!(rows.push_row([(2, 4)]).is_err());
        assert!(rows.finish(3).is_err());
        assert!(IndexedCoefficients::new(vec![3u64], vec![1u32]).is_err());
    }

    #[test]
    fn indexed_conversion_keeps_the_shared_table() {
        let table = std::sync::Arc::new(vec![4i64, -3]);
        let coefficients = IndexedCoefficients::new(table.clone(), vec![0u32, 1, 0]).unwrap();
        let csr =
            CsrMatrix::<_, u32>::try_from_parts(3, vec![0, 2, 3], vec![0, 2, 1], coefficients)
                .unwrap();
        let csc = csr.clone().into_csc().unwrap();
        assert!(std::sync::Arc::ptr_eq(csc.coefficients().values(), &table));
        assert_eq!(
            csc.column(2).unwrap().indexed_entries().collect::<Vec<_>>(),
            [(0, 1)]
        );
        assert_eq!(csc.into_csr().unwrap(), csr);
    }
}

impl<I: SparseIndex> CscMatrix<ImplicitOnes, I> {
    pub fn mul_left(
        &self,
        weights: &[field::Gf128],
    ) -> Result<Vec<field::Gf128>, super::LinearMapError> {
        let mut out = vec![field::Gf128::new(0, 0); self.column_count()];
        self.mul_left_kernel(
            weights,
            &mut out,
            rayon::current_num_threads() > 1 && self.nnz() >= (1 << 15),
        )?;
        Ok(out)
    }
    pub(crate) fn mul_left_kernel(
        &self,
        weights: &[field::Gf128],
        out: &mut [field::Gf128],
        parallel: bool,
    ) -> Result<(), super::LinearMapError> {
        for (kind, expected, actual) in [
            ("row weights", self.row_count(), weights.len()),
            ("output columns", self.column_count(), out.len()),
        ] {
            if expected != actual {
                return Err(super::LinearMapError::Length {
                    kind,
                    expected,
                    actual,
                });
            }
        }
        let evaluate = |column: usize| {
            let mut lo = 0;
            let mut hi = 0;
            for &row in self.column(column).unwrap().indices() {
                let value = weights[row.to_usize()];
                lo ^= value.lo;
                hi ^= value.hi;
            }
            field::Gf128::new(lo, hi)
        };
        if parallel {
            use rayon::prelude::*;
            out.par_iter_mut()
                .enumerate()
                .for_each(|(j, v)| *v = evaluate(j));
        } else {
            for (j, v) in out.iter_mut().enumerate() {
                *v = evaluate(j);
            }
        }
        Ok(())
    }
}
impl<I: SparseIndex> super::LeftMul<field::Gf128> for CscMatrix<ImplicitOnes, I> {
    type Output = field::Gf128;
    fn mul_left_into(
        &mut self,
        weights: &[field::Gf128],
        out: &mut [Self::Output],
    ) -> Result<(), super::LinearMapError> {
        self.mul_left_kernel(
            weights,
            out,
            rayon::current_num_threads() > 1 && self.nnz() >= (1 << 15),
        )
    }
}

pub trait BooleanCoefficients: CoefficientStore {
    fn boolean(&self, entry: usize) -> bool;
}
impl BooleanCoefficients for ImplicitOnes {
    fn boolean(&self, _: usize) -> bool {
        true
    }
}
impl BooleanCoefficients for Box<[bool]> {
    fn boolean(&self, entry: usize) -> bool {
        self[entry]
    }
}
impl<S: BooleanCoefficients, I: SparseIndex> BooleanCoefficients for IndexedCoefficients<S, I> {
    fn boolean(&self, entry: usize) -> bool {
        self.values.boolean(self.indices[entry].to_usize())
    }
}
impl<S: BooleanCoefficients> BooleanCoefficients for std::sync::Arc<S> {
    fn boolean(&self, entry: usize) -> bool {
        (**self).boolean(entry)
    }
}
impl<S: BooleanCoefficients> BooleanCoefficients for &S {
    fn boolean(&self, entry: usize) -> bool {
        (**self).boolean(entry)
    }
}
macro_rules! boolean_operation {
    ($matrix:ident,$source_count:ident,$out_count:ident,$segments:ident,$op:ident,$method:ident) => {
        impl<S, I> super::$op<bool> for $matrix<S, I>
        where
            I: SparseIndex,
            S: BooleanCoefficients,
        {
            type Output = bool;
            fn $method(
                &mut self,
                values: &[bool],
                out: &mut [bool],
            ) -> Result<(), super::LinearMapError> {
                for (kind, expected, actual) in [
                    ("source", self.$source_count(), values.len()),
                    ("output", self.$out_count(), out.len()),
                ] {
                    if expected != actual {
                        return Err(super::LinearMapError::Length {
                            kind,
                            expected,
                            actual,
                        });
                    }
                }
                for (dst, segment) in out.iter_mut().zip(self.$segments()) {
                    *dst = segment.indices().iter().zip(segment.entry_range()).fold(
                        false,
                        |sum, (&index, entry)| {
                            sum ^ (self.coefficients.boolean(entry) & values[index.to_usize()])
                        },
                    );
                }
                Ok(())
            }
        }
    };
}
boolean_operation!(
    CsrMatrix,
    column_count,
    row_count,
    rows,
    RightMul,
    mul_right_into
);
boolean_operation!(
    CscMatrix,
    row_count,
    column_count,
    columns,
    LeftMul,
    mul_left_into
);
