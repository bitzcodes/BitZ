//! Shared forward/adjoint kernels. Sources and arithmetic are statically dispatched.
use super::builder::{NEG_ONE, NONE, ONE};
use super::compile::CompiledGraph;
use field::{BatchMulAcc, MergeAccumulator, Reduce, RingOps};
use rayon::prelude::*;
use std::borrow::Cow;

const PARALLEL_LEVEL: usize = 1 << 16;
const OUTPUT_CHUNK: usize = 1 << 12;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LinearMapError {
    #[error("integer output width {output_limbs} limbs is insufficient for segment {segment}")]
    IntegerWidth { output_limbs: usize, segment: usize },
    #[error("linear map expects {expected} {kind} values, received {actual}")]
    Length {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
}
fn length(kind: &'static str, expected: usize, actual: usize) -> Result<(), LinearMapError> {
    if expected == actual {
        Ok(())
    } else {
        Err(LinearMapError::Length {
            kind,
            expected,
            actual,
        })
    }
}

pub trait ColumnValues<E>: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn scalar(&self, column: usize) -> E;
    /// Σ_{k<len} 2^k x[first+k]. Structured sources override this with a
    /// geometric-range calculation instead of expanding packed input nodes.
    fn power_sum(&self, first: usize, len: usize) -> E;
}

pub struct DenseColumns<'a, F: RingOps> {
    field: &'a F,
    values: &'a [F::Elem],
}
impl<'a, F: RingOps> DenseColumns<'a, F> {
    pub fn new(field: &'a F, values: &'a [F::Elem]) -> Self {
        Self { field, values }
    }
}
impl<F: RingOps + Sync> ColumnValues<F::Elem> for DenseColumns<'_, F> {
    fn len(&self) -> usize {
        self.values.len()
    }
    fn scalar(&self, column: usize) -> F::Elem {
        self.values[column]
    }
    fn power_sum(&self, first: usize, len: usize) -> F::Elem {
        self.values[first..first + len]
            .iter()
            .rev()
            .fold(self.field.zero(), |s, x| {
                self.field.add(&self.field.add(&s, &s), x)
            })
    }
}

/// Cached coefficients and independent, lazily allocated direction workspaces.
pub struct PreparedWengert<'a, F: RingOps> {
    pub(super) graph: &'a CompiledGraph,
    pub(super) field: F,
    coefficients: Cow<'a, [F::Elem]>,
    adjoints: Vec<F::Elem>,
    forward_values: Vec<F::Elem>,
    powers: Vec<F::Elem>,
    parallel: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
pub struct PowerRun<E> {
    pub first_column: usize,
    pub len: usize,
    pub base: E,
}

struct Arithmetic<'a, F: RingOps> {
    field: &'a F,
    coefficients: &'a [F::Elem],
}
impl<F: RingOps> Arithmetic<'_, F> {
    #[inline(always)]
    fn scale(&self, value: F::Elem, coefficient: u32) -> F::Elem {
        match coefficient {
            ONE => value,
            NEG_ONE => self.field.neg(&value),
            c => self.field.mul(&value, &self.coefficients[c as usize]),
        }
    }
    #[inline(always)]
    fn evaluate<K: NodeReduction<F>>(
        &self,
        graph: &CompiledGraph,
        node: usize,
        prior: &[F::Elem],
        seed: &impl Fn(usize) -> F::Elem,
        kernel: &K,
    ) -> F::Elem {
        let roots =
            &graph.roots[graph.root_offsets[node] as usize..graph.root_offsets[node + 1] as usize];
        let edges =
            &graph.edges[graph.edge_offsets[node] as usize..graph.edge_offsets[node + 1] as usize];
        if roots.is_empty() && edges.len() == 1 {
            return self.scale(prior[edges[0].node as usize], edges[0].coefficient);
        }
        kernel.sum(
            self,
            roots.len() + edges.len(),
            roots
                .iter()
                .map(|r| (seed(r.output as usize), r.coefficient))
                .chain(
                    edges
                        .iter()
                        .map(|e| (prior[e.node as usize], e.coefficient)),
                ),
        )
    }
}

trait NodeReduction<F: RingOps>: Sync {
    fn sum(
        &self,
        a: &Arithmetic<'_, F>,
        count: usize,
        terms: impl Iterator<Item = (F::Elem, u32)>,
    ) -> F::Elem;
}
struct Immediate;
impl<F: RingOps> NodeReduction<F> for Immediate {
    #[inline(always)]
    fn sum(
        &self,
        a: &Arithmetic<'_, F>,
        _count: usize,
        terms: impl Iterator<Item = (F::Elem, u32)>,
    ) -> F::Elem {
        let mut sum = a.field.zero();
        let mut initialized = false;
        for (value, c) in terms {
            let term = a.scale(value, c);
            sum = if initialized {
                a.field.add(&sum, &term)
            } else {
                term
            };
            initialized = true;
        }
        sum
    }
}
struct Delayed<R> {
    reduce: R,
    capacity: usize,
}
impl<F, R> NodeReduction<F> for Delayed<R>
where
    F: RingOps + BatchMulAcc<F::Elem>,
    R: Fn(<F as BatchMulAcc<F::Elem>>::Accumulator) -> F::Elem + Sync,
{
    #[inline]
    fn sum(
        &self,
        a: &Arithmetic<'_, F>,
        count: usize,
        terms: impl Iterator<Item = (F::Elem, u32)>,
    ) -> F::Elem {
        // Small fan-in, singleton and ±1-only work retain the cheap scalar path.
        if count < 8 {
            return Immediate.sum(a, count, terms);
        }
        let mut sum = a.field.zero();
        let mut acc = <F as BatchMulAcc<F::Elem>>::Accumulator::zero();
        let mut products = 0;
        for (value, c) in terms {
            if c == ONE {
                sum = a.field.add(&sum, &value);
            } else if c == NEG_ONE {
                sum = a.field.sub(&sum, &value);
            } else {
                a.field
                    .mul_acc(&mut acc, &value, &a.coefficients[c as usize]);
                products += 1;
                if products == self.capacity {
                    sum = a.field.add(&sum, &(self.reduce)(acc));
                    acc = <F as BatchMulAcc<F::Elem>>::Accumulator::zero();
                    products = 0;
                }
            }
        }
        if products != 0 {
            sum = a.field.add(&sum, &(self.reduce)(acc));
        }
        sum
    }
}

impl<'a, F: RingOps + Sync> PreparedWengert<'a, F> {
    pub(super) fn new(
        graph: &'a CompiledGraph,
        field: F,
        coefficients: Cow<'a, [F::Elem]>,
    ) -> Self {
        Self {
            graph,
            field,
            coefficients,
            adjoints: Vec::new(),
            forward_values: Vec::new(),
            powers: Vec::new(),
            parallel: None,
        }
    }
    #[cfg(test)]
    pub(super) fn set_parallel(&mut self, parallel: Option<bool>) {
        self.parallel = parallel;
    }
    pub fn workspace_bytes(&self) -> usize {
        let coefficient_bytes = if matches!(self.coefficients, Cow::Owned(_)) {
            self.coefficients.len() * std::mem::size_of::<F::Elem>()
        } else {
            0
        };
        coefficient_bytes
            + (self.adjoints.len() + self.forward_values.len() + self.powers.len())
                * std::mem::size_of::<F::Elem>()
    }
    /// Mᵀw using directly borrowed or mapped seeds, without a staging vector.
    pub fn adjoint_map_into(
        &mut self,
        seed_count: usize,
        seed: impl Fn(usize) -> F::Elem + Sync,
        out: &mut [F::Elem],
    ) -> Result<(), LinearMapError> {
        self.adjoint_map_storage_into(seed_count, seed, out, |value| value)
    }

    /// A statically mapped output representation; conversion is fused with
    /// each store, never a separate pass over the column vector.
    pub fn adjoint_map_storage_into<O: Send>(
        &mut self,
        seed_count: usize,
        seed: impl Fn(usize) -> F::Elem + Sync,
        out: &mut [O],
        write: impl Fn(F::Elem) -> O + Sync,
    ) -> Result<(), LinearMapError> {
        self.adjoint_kernel(seed_count, seed, out, write, &Immediate)
    }

    fn compute_adjoints<K: NodeReduction<F>>(
        &mut self,
        seed: &(impl Fn(usize) -> F::Elem + Sync),
        kernel: &K,
    ) {
        let Self {
            graph,
            field,
            coefficients,
            adjoints,
            powers,
            parallel,
            ..
        } = self;
        let field = &*field;
        let arithmetic = Arithmetic {
            field,
            coefficients,
        };
        let internal = *graph.levels.last().unwrap();
        if adjoints.is_empty() {
            *adjoints = field.zero_vec(internal);
        }
        let evaluate =
            |node: usize, prior: &[F::Elem]| arithmetic.evaluate(graph, node, prior, seed, kernel);
        for range in graph.levels.windows(2) {
            let (start, end) = (range[0], range[1]);
            let (prior, rest) = adjoints.split_at_mut(start);
            let current = &mut rest[..end - start];
            if parallel
                .unwrap_or(current.len() >= PARALLEL_LEVEL && rayon::current_num_threads() > 1)
            {
                current
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v = evaluate(start + i, prior));
            } else {
                for (i, v) in current.iter_mut().enumerate() {
                    *v = evaluate(start + i, prior);
                }
            }
        }
        let max = graph.max_power_len;
        if powers.len() < max {
            powers.clear();
            powers.reserve_exact(max);
            let mut p = field.one();
            for _ in 0..max {
                powers.push(p);
                p = field.add(&p, &p);
            }
        }
    }

    /// Visit scalar inputs and geometric ranges from the shared adjoints.
    /// Length-one ranges are scalar values; no dense output is allocated.
    pub(super) fn adjoint_map_segments(
        &mut self,
        seed_count: usize,
        seed: impl Fn(usize) -> F::Elem + Sync,
        mut emit: impl FnMut(PowerRun<F::Elem>),
    ) -> Result<(), LinearMapError> {
        length("output weights", self.graph.output_count, seed_count)?;
        self.compute_adjoints(&seed, &Immediate);
        let arithmetic = Arithmetic {
            field: &self.field,
            coefficients: &self.coefficients,
        };
        let scalar = |column| {
            let node = self.graph.inputs[column];
            PowerRun {
                first_column: column,
                len: 1,
                base: if node == NONE {
                    self.field.zero()
                } else {
                    arithmetic.evaluate(
                        self.graph,
                        node as usize,
                        &self.adjoints,
                        &seed,
                        &Immediate,
                    )
                },
            }
        };
        let mut cursor = 0;
        for run in self.power_runs() {
            if run.len == 0 {
                continue;
            }
            for column in cursor..run.first_column {
                emit(scalar(column));
            }
            cursor = run.first_column + run.len;
            emit(run);
        }
        for column in cursor..self.graph.inputs.len() {
            emit(scalar(column));
        }
        Ok(())
    }

    fn adjoint_kernel<O: Send, K: NodeReduction<F>>(
        &mut self,
        seed_count: usize,
        seed: impl Fn(usize) -> F::Elem + Sync,
        out: &mut [O],
        write: impl Fn(F::Elem) -> O + Sync,
        kernel: &K,
    ) -> Result<(), LinearMapError> {
        length("output weights", self.graph.output_count, seed_count)?;
        length("input coefficients", self.graph.inputs.len(), out.len())?;
        self.compute_adjoints(&seed, kernel);
        let Self {
            graph,
            field,
            coefficients,
            adjoints,
            powers,
            parallel,
            ..
        } = self;
        let field = &*field;
        let arithmetic = Arithmetic {
            field,
            coefficients,
        };
        let evaluate =
            |node: usize, prior: &[F::Elem]| arithmetic.evaluate(graph, node, prior, &seed, kernel);
        let fill = |chunk_index: usize, chunk: &mut [O]| {
            let begin = chunk_index * OUTPUT_CHUNK;
            let end = begin + chunk.len();
            let mut group_index = graph
                .groups
                .partition_point(|g| g.first as usize + g.len as usize <= begin);
            let mut column = begin;
            while column < end {
                let group = graph.groups.get(group_index);
                if group.is_none_or(|g| column < (g.first as usize)) {
                    let stop = group.map_or(end, |g| (g.first as usize).min(end));
                    for j in column..stop {
                        let node = graph.inputs[j];
                        chunk[j - begin] = write(if node == NONE {
                            field.zero()
                        } else {
                            evaluate(node as usize, adjoints)
                        });
                    }
                    column = stop;
                    continue;
                }
                let g = group.unwrap();
                let group_end = g.first as usize + g.len as usize;
                let low_end = g.first as usize + g.low_len as usize;
                let stop = if column < low_end {
                    low_end.min(group_end).min(end)
                } else {
                    group_end.min(end)
                };
                let read = |n: u32| {
                    if n == NONE {
                        field.zero()
                    } else {
                        adjoints[n as usize]
                    }
                };
                let mut value = read(g.full);
                if column < low_end && g.low != NONE {
                    value = field.add(&value, &read(g.low));
                }
                let shift = column - g.first as usize;
                if shift != 0 {
                    value = field.mul(&value, &powers[shift]);
                }
                let slots = &mut chunk[column - begin..stop - begin];
                if slots.len() >= 32 {
                    // Two independent doubling chains expose instruction-level
                    // parallelism while preserving contiguous coordinate order.
                    let half = slots.len() / 2;
                    let mut upper = field.mul(&value, &powers[half]);
                    let (low, high) = slots.split_at_mut(half);
                    for (lo, hi) in low.iter_mut().zip(high.iter_mut()) {
                        *lo = write(value);
                        *hi = write(upper);
                        value = field.add(&value, &value);
                        upper = field.add(&upper, &upper);
                    }
                    if high.len() != half {
                        high[half] = write(upper);
                    }
                } else {
                    for slot in slots {
                        *slot = write(value);
                        value = field.add(&value, &value);
                    }
                }
                column = stop;
                if column == group_end {
                    group_index += 1;
                }
            }
        };
        if parallel.unwrap_or(out.len() >= 1 << 14 && rayon::current_num_threads() > 1) {
            out.par_chunks_mut(OUTPUT_CHUNK)
                .enumerate()
                .for_each(|(i, c)| fill(i, c));
        } else {
            for (i, c) in out.chunks_mut(OUTPUT_CHUNK).enumerate() {
                fill(i, c);
            }
        }
        Ok(())
    }

    pub fn evaluate_bilinear_map(
        &mut self,
        seed_count: usize,
        seed: impl Fn(usize) -> F::Elem + Sync,
        columns: &impl ColumnValues<F::Elem>,
    ) -> Result<F::Elem, LinearMapError> {
        length("output weights", self.graph.output_count, seed_count)?;
        length("columns", self.graph.inputs.len(), columns.len())?;
        evaluate_forward(
            self.graph,
            &self.field,
            &self.coefficients,
            &mut self.forward_values,
            self.parallel,
            columns,
        );
        let graph = self.graph;
        let field = &self.field;
        let values = &self.forward_values;
        let arithmetic = Arithmetic {
            field,
            coefficients: &self.coefficients,
        };
        let mut result = field.zero();
        for (n, &value) in values.iter().enumerate() {
            for root in
                &graph.roots[graph.root_offsets[n] as usize..graph.root_offsets[n + 1] as usize]
            {
                result = field.add(
                    &result,
                    &field.mul(
                        &seed(root.output as usize),
                        &arithmetic.scale(value, root.coefficient),
                    ),
                );
            }
        }
        Ok(result)
    }

    /// Geometric metadata for the last successful adjoint. Public corrections
    /// must split these runs before handing them to a structured MLE consumer.
    pub fn power_runs(&self) -> Vec<PowerRun<F::Elem>> {
        let read = |n: u32| {
            if n == NONE {
                self.field.zero()
            } else {
                self.adjoints[n as usize]
            }
        };
        let mut runs = Vec::with_capacity(2 * self.graph.groups.len());
        for g in &self.graph.groups {
            let full = read(g.full);
            if g.low != NONE {
                runs.push(PowerRun {
                    first_column: g.first as usize,
                    len: g.low_len as usize,
                    base: self.field.add(&full, &read(g.low)),
                });
                if g.low_len < g.len {
                    runs.push(PowerRun {
                        first_column: (g.first + g.low_len) as usize,
                        len: (g.len - g.low_len) as usize,
                        base: self.field.mul(&full, &self.powers[g.low_len as usize]),
                    });
                }
            } else {
                runs.push(PowerRun {
                    first_column: g.first as usize,
                    len: g.len as usize,
                    base: full,
                });
            }
        }
        runs
    }
}

impl<F> PreparedWengert<'_, F>
where
    F: RingOps
        + Sync
        + Clone
        + BatchMulAcc<F::Elem>
        + Reduce<<F as BatchMulAcc<F::Elem>>::Accumulator, Output = F::Elem>,
{
    /// Bound each node independently from its public degree. No worker merges
    /// cross nodes; chunks are reduced before dependent nodes read the result.
    /// Caller supplies mapped seeds and output storage with no staging copies.
    pub fn adjoint_delayed_map_storage_into<O: Send>(
        &mut self,
        seed_count: usize,
        seed: impl Fn(usize) -> F::Elem + Sync,
        out: &mut [O],
        write: impl Fn(F::Elem) -> O + Sync,
    ) -> Result<(), LinearMapError> {
        let capacity = self.graph.max_terms.clamp(1, 1 << 16);
        let field = self.field.clone();
        let kernel = Delayed {
            reduce: field.prepare_reduce(capacity),
            capacity,
        };
        self.adjoint_kernel(seed_count, seed, out, write, &kernel)
    }
}

fn evaluate_forward<F: RingOps + Sync>(
    graph: &CompiledGraph,
    field: &F,
    coefficients: &[F::Elem],
    values: &mut Vec<F::Elem>,
    parallel: Option<bool>,
    columns: &impl ColumnValues<F::Elem>,
) {
    let arithmetic = Arithmetic {
        field,
        coefficients,
    };
    if values.is_empty() {
        *values = field.zero_vec(graph.node_count());
    }
    // Packed columns are private: their entire contribution is supplied by
    // power_sum. Skip their ranges instead of scanning each scalar slot.
    let mut next_column = 0;
    for group in &graph.groups {
        for column in next_column..group.first as usize {
            let node = graph.inputs[column];
            if node != NONE {
                values[node as usize] = columns.scalar(column);
            }
        }
        next_column = group.first as usize + group.len as usize;
        if group.full != NONE {
            values[group.full as usize] = if group.len == 1 {
                columns.scalar(group.first as usize)
            } else {
                columns.power_sum(group.first as usize, group.len as usize)
            };
        }
        if group.low != NONE {
            values[group.low as usize] = if group.low_len == 1 {
                columns.scalar(group.first as usize)
            } else {
                columns.power_sum(group.first as usize, group.low_len as usize)
            };
        }
    }
    for column in next_column..graph.inputs.len() {
        let node = graph.inputs[column];
        if node != NONE {
            values[node as usize] = columns.scalar(column);
        }
    }
    for range in graph.levels.windows(2).rev() {
        let (start, end) = (range[0], range[1]);
        let (before, later) = values.split_at_mut(end);
        let current = &mut before[start..end];
        let evaluate = |i: usize, slot: &mut F::Elem| {
            let n = start + i;
            let terms = &graph.forward
                [graph.forward_offsets[n] as usize..graph.forward_offsets[n + 1] as usize];
            if terms.is_empty() {
                return;
            }
            let mut result = None;
            for t in terms {
                let value = arithmetic.scale(later[t.node as usize - end], t.coefficient);
                result = Some(result.map_or(value, |s| field.add(&s, &value)));
            }
            *slot = result.unwrap_or_else(|| field.zero());
        };
        if parallel.unwrap_or(current.len() >= PARALLEL_LEVEL && rayon::current_num_threads() > 1) {
            current
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, v)| evaluate(i, v));
        } else {
            for (i, v) in current.iter_mut().enumerate() {
                evaluate(i, v);
            }
        }
    }
}
impl<F: RingOps + Sync> super::LeftMul<F::Elem> for PreparedWengert<'_, F> {
    type Output = F::Elem;
    fn mul_left_into(
        &mut self,
        weights: &[F::Elem],
        out: &mut [F::Elem],
    ) -> Result<(), LinearMapError> {
        self.adjoint_map_into(weights.len(), |i| weights[i], out)
    }
}
impl<F: RingOps + Sync> super::RightMul<F::Elem> for PreparedWengert<'_, F> {
    type Output = F::Elem;
    fn mul_right_into(
        &mut self,
        values: &[F::Elem],
        out: &mut [F::Elem],
    ) -> Result<(), LinearMapError> {
        length("column values", self.graph.inputs.len(), values.len())?;
        length("output rows", self.graph.output_count, out.len())?;
        let columns = DenseColumns::new(&self.field, values);
        evaluate_forward(
            self.graph,
            &self.field,
            &self.coefficients,
            &mut self.forward_values,
            self.parallel,
            &columns,
        );
        out.fill(self.field.zero());
        let arithmetic = Arithmetic {
            field: &self.field,
            coefficients: &self.coefficients,
        };
        for (n, &value) in self.forward_values.iter().enumerate() {
            for root in &self.graph.roots
                [self.graph.root_offsets[n] as usize..self.graph.root_offsets[n + 1] as usize]
            {
                let slot = &mut out[root.output as usize];
                *slot = self
                    .field
                    .add(slot, &arithmetic.scale(value, root.coefficient));
            }
        }
        Ok(())
    }
}
impl<F: RingOps + Sync> super::BilinearEval<F> for PreparedWengert<'_, F> {
    fn evaluate_bilinear(
        &mut self,
        weights: &[F::Elem],
        columns: &impl ColumnValues<F::Elem>,
    ) -> Result<F::Elem, LinearMapError> {
        self.evaluate_bilinear_map(weights.len(), |i| weights[i], columns)
    }
}
