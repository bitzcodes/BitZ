//! Linear operators shared by matrix binding and virtual-opening adjoints.
pub mod binary;
mod builder;
#[cfg(feature = "full")]
pub mod circuit;
mod coefficients;
mod compile;
mod evaluate;
mod integer;
mod operations;
mod prepared_field;
mod sparse;
pub use integer::{PreparedIntegerColumns, PreparedIntegerRows};
pub use operations::{BilinearEval, LeftMul, RightMul};
pub use prepared_field::{DirectCoefficients, PreparedCoefficients, PreparedColumns, PreparedRows};
#[cfg(test)]
mod tests;
pub use builder::{NodeId, OutputId, PackedInputs, WengertBuilder};
pub use coefficients::{CoefficientStore, FieldCoefficients, StoreCoefficient};
pub use evaluate::{ColumnValues, DenseColumns, LinearMapError, PowerRun, PreparedWengert};
pub use sparse::{
    CscMatrix, CsrBuilder, CsrMatrix, ImplicitOnes, IndexedCoefficients, PermuteCoefficients,
    SparseEntries, SparseIndex, SparseMatrixError, SparseSegment,
};

/// A compiled linear operator. Only integer-backed coefficients are reusable
/// across arbitrary runtime fields; field-backed storage retains its provider.
#[derive(Debug)]
pub struct WengertTape<S> {
    coefficients: S,
    graph: compile::CompiledGraph,
}
impl<S: CoefficientStore> WengertTape<S> {
    pub fn input_count(&self) -> usize {
        self.graph.inputs.len()
    }
    pub fn output_count(&self) -> usize {
        self.graph.output_count
    }
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
    pub fn edge_count(&self) -> usize {
        self.graph.edges.len()
    }
    pub fn level_count(&self) -> usize {
        self.graph.levels.len() - 1
    }
    pub fn coefficients(&self) -> &S {
        &self.coefficients
    }
    pub fn topology_bytes(&self) -> usize {
        self.graph.payload_bytes()
    }
}
pub mod binary_adjoint;
pub mod contraction;
