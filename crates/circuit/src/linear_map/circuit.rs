//! Modulus-independent reverse-mode tape for `r * (A + x B + x^2 C)`.
//!
//! The circuit is replayed once without witness values or a field modulus.
//! Z-side linear arithmetic is recorded as a Wengert graph, then dead nodes are
//! removed and the remaining graph is transposed and ordered by reverse depth.
//! Applying the finished tape is reverse-mode automatic differentiation of
//! `r * (A + x B + x^2 C) * w`. Nodes at one depth write disjoint adjoints, so
//! sufficiently wide depths are evaluated in parallel without atomics.

use super::{BilinearEval, ColumnValues, LeftMul, RightMul};
use field::ModRingCtx;

use num_traits::{CheckedAdd, CheckedMul, CheckedNeg};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use std::sync::{Arc, Mutex};

use field::{FpCtx, IntegerEmbedding, RingOps, Uint, create_prime_field};
use num_traits::{One, Zero};

use crate::integer_storage::IntegerTable;
use crate::witgen::Z;
use crate::{BoolWitness, Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

const NO_NODE: u32 = u32::MAX;

/// A value-free Boolean handle used while recording the arithmetic tape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WengertBit;

impl From<bool> for WengertBit {
    fn from(_: bool) -> Self {
        Self
    }
}

impl BoolWitness for WengertBit {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
}

#[derive(Clone, Copy, Debug)]
struct RawTerm {
    node: u32,
    coefficient: u32,
}

impl RawTerm {
    fn is_zero(&self) -> bool {
        self.coefficient == 2
    }
}

#[derive(Debug)]
enum RawNode {
    Input,
    Packed,
    Sum(Box<[RawTerm]>),
}

#[derive(Clone, Copy, Debug)]
enum RootKind {
    A,
    B,
    C,
}

#[derive(Debug)]
struct RawRoot {
    row: u32,
    kind: RootKind,
    term: RawTerm,
}

/// Builds the symbolic integer-arithmetic graph as the circuit is replayed.
/// For example, `3*a + 2*b` records a sum pointing to `a` and `b` with weights
/// 3 and 2. Each constraint records which expressions form its A/B/C sides.
/// [`WengertGenerator::finish`] prunes unused nodes and arranges this graph into
/// a [`WengertTape`] for propagating row weights backward to witness columns.
#[derive(Debug)]
struct Recorder {
    /// Inputs and weighted sums in creation order; each node's ID is its index.
    nodes: Vec<RawNode>,
    /// Scalar input handles in witness-column order; packed private columns
    /// use NO_NODE. Column 0 is the explicit constant-one input.
    input_nodes: Vec<u32>,
    /// Three roots per constraint: its A/B/C expressions, row index, and coefficients.
    roots: Vec<RawRoot>,
    /// Bit-column ranges and their full/low packed sums, for later power-of-two expansion.
    power_groups: Vec<RawPowerGroup>,
    /// Number of constraints recorded so far; also the next constraint's row index.
    constraints: usize,
    coefficients: IntegerTable,
    coefficient_map: HashMap<Vec<u64>, u32>,
}

#[derive(Debug)]
struct RawPowerGroup {
    first_column: u32,
    len: u32,
    low_len: u32,
    full_node: u32,
    low_node: u32,
}

impl Recorder {
    fn new() -> Self {
        let mut recorder = Self {
            // Integer-witness column zero is the implicit constant one.
            nodes: vec![RawNode::Input],
            input_nodes: vec![0],
            roots: Vec::new(),
            power_groups: Vec::new(),
            constraints: 0,
            coefficients: IntegerTable::default(),
            coefficient_map: HashMap::new(),
        };
        recorder.intern(Z::<1>::ONE);
        recorder.intern(-Z::<1>::ONE);
        recorder.intern(Z::<1>::ZERO);
        recorder
    }

    // Coefficients are public circuit structure. Normalize only the hash key to
    // deduplicate equivalent values from different gadget widths; storage keeps L.
    fn intern<const L: usize>(&mut self, value: Z<L>) -> u32 {
        let words = value.as_words();
        let mut len = L;
        while len > 1 {
            let sign = 0u64.wrapping_sub(words[len - 2] >> 63);
            if words[len - 1] != sign {
                break;
            }
            len -= 1;
        }
        let key = &words[..len];
        if let Some(index) = self.coefficient_map.get(key) {
            return *index;
        }
        let index = u32::try_from(self.coefficients.len()).expect("too many tape coefficients");
        self.coefficients.push(value);
        self.coefficient_map.insert(key.to_vec(), index);
        index
    }

    fn push_input(&mut self) -> u32 {
        let node = u32::try_from(self.nodes.len()).expect("too many Wengert nodes");
        self.nodes.push(RawNode::Input);
        self.input_nodes.push(node);
        node
    }

    fn push_sum(&mut self, terms: Box<[RawTerm]>) -> u32 {
        debug_assert!(!terms.is_empty());
        let node = u32::try_from(self.nodes.len()).expect("too many Wengert nodes");
        debug_assert!(terms.iter().all(|term| term.node < node));
        self.nodes.push(RawNode::Sum(terms));
        node
    }
}

#[derive(Debug)]
struct SharedRecorder(Mutex<Option<Recorder>>);

impl SharedRecorder {
    fn with_mut<R>(&self, apply: impl FnOnce(&mut Recorder) -> R) -> R {
        let mut guard = self.0.lock().expect("Wengert recorder lock poisoned");
        apply(
            guard
                .as_mut()
                .expect("the Wengert generator has already been finished"),
        )
    }
}

/// A scaled Wengert handle with the gadget-local coefficient width.
#[derive(Clone, Debug)]
pub struct WengertValue<const LIMBS: usize> {
    location: ValueLocation,
    coefficient: Z<LIMBS>,
}

#[derive(Clone, Debug)]
enum ValueLocation {
    /// A coefficient times the implicit constant-one input.
    Constant,
    Node {
        recorder: Arc<SharedRecorder>,
        node: u32,
    },
}

impl<const LIMBS: usize> WengertValue<LIMBS> {
    fn attached(recorder: Arc<SharedRecorder>, node: u32, coefficient: Z<LIMBS>) -> Self {
        Self {
            location: ValueLocation::Node { recorder, node },
            coefficient,
        }
    }

    fn recorder(&self) -> Option<&Arc<SharedRecorder>> {
        match &self.location {
            ValueLocation::Constant => None,
            ValueLocation::Node { recorder, .. } => Some(recorder),
        }
    }

    fn raw_term(&self, recorder: &mut Recorder) -> RawTerm {
        let node = match self.location {
            ValueLocation::Constant => 0,
            ValueLocation::Node { node, .. } => node,
        };
        RawTerm {
            node,
            coefficient: recorder.intern(self.coefficient),
        }
    }

    fn same_recorder(left: &Arc<SharedRecorder>, right: &Arc<SharedRecorder>) {
        assert!(
            Arc::ptr_eq(left, right),
            "cannot combine values from different Wengert generators"
        );
    }
}

impl<const LIMBS: usize> From<Z<LIMBS>> for WengertValue<LIMBS> {
    fn from(coefficient: Z<LIMBS>) -> Self {
        Self {
            location: ValueLocation::Constant,
            coefficient,
        }
    }
}

impl<const LIMBS: usize> Zero for WengertValue<LIMBS> {
    fn zero() -> Self {
        Self::from(Z::zero())
    }

    fn is_zero(&self) -> bool {
        self.coefficient.is_zero()
    }
}

impl<const LIMBS: usize> Add for WengertValue<LIMBS> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if let (Some(a), Some(b)) = (self.recorder(), rhs.recorder()) {
            Self::same_recorder(a, b);
        }
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }

        match (&self.location, &rhs.location) {
            (ValueLocation::Constant, ValueLocation::Constant) => Self::from(
                self.coefficient
                    .checked_add(&rhs.coefficient)
                    .expect("constant expression exceeds its declared integer width"),
            ),
            (
                ValueLocation::Node {
                    recorder: left,
                    node: left_node,
                },
                ValueLocation::Node {
                    recorder: right,
                    node: right_node,
                },
            ) if left_node == right_node
                && self.coefficient.checked_add(&rhs.coefficient).is_some() =>
            {
                Self::same_recorder(left, right);
                Self::attached(
                    left.clone(),
                    *left_node,
                    self.coefficient.checked_add(&rhs.coefficient).unwrap(),
                )
            }
            _ => {
                let recorder = self
                    .recorder()
                    .or_else(|| rhs.recorder())
                    .expect("nonconstant Wengert sum needs a recorder")
                    .clone();
                if let Some(other) = self.recorder() {
                    Self::same_recorder(&recorder, other);
                }
                if let Some(other) = rhs.recorder() {
                    Self::same_recorder(&recorder, other);
                }
                let node = recorder.with_mut(|tape| {
                    let terms = [self.raw_term(tape), rhs.raw_term(tape)].into();
                    tape.push_sum(terms)
                });
                Self::attached(recorder, node, Z::one())
            }
        }
    }
}

impl<const LIMBS: usize> AddAssign for WengertValue<LIMBS> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<const LIMBS: usize> Neg for WengertValue<LIMBS> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        if let Some(value) = self.coefficient.checked_neg() {
            self.coefficient = value;
            return self;
        }
        let recorder = self
            .recorder()
            .expect("constant negation exceeds its declared integer width")
            .clone();
        let node = recorder.with_mut(|tape| {
            let term = self.raw_term(tape);
            tape.push_sum(Box::new([term]))
        });
        Self::attached(recorder, node, -Z::ONE)
    }
}

impl<const LIMBS: usize> Sub for WengertValue<LIMBS> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<const LIMBS: usize> SubAssign for WengertValue<LIMBS> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<const LIMBS: usize> Mul<Z<LIMBS>> for WengertValue<LIMBS> {
    type Output = Self;

    fn mul(mut self, rhs: Z<LIMBS>) -> Self::Output {
        if let Some(value) = self.coefficient.checked_mul(&rhs) {
            self.coefficient = value;
            return self;
        }
        let recorder = self
            .recorder()
            .expect("constant product exceeds its declared integer width")
            .clone();
        let node = recorder.with_mut(|tape| {
            let term = self.raw_term(tape);
            tape.push_sum(Box::new([term]))
        });
        Self::attached(recorder, node, rhs)
    }
}

impl<const LIMBS: usize> Sum for WengertValue<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

/// Circuit A/B/C output layout over the shared generic linear graph.
#[derive(Debug)]
pub struct WengertTape {
    core: super::WengertTape<IntegerTable>,
    constraints: usize,
}

impl WengertTape {
    fn from_recorder(recorder: Recorder) -> Self {
        use super::builder::{NEG_ONE, ONE};
        let Recorder {
            nodes,
            input_nodes: _,
            roots,
            power_groups,
            constraints,
            coefficients,
            ..
        } = recorder;
        let mut builder = super::WengertBuilder::new(coefficients);
        let mut mapped = vec![None; nodes.len()];
        let mut needed = vec![false; nodes.len()];
        let group_start = power_groups
            .iter()
            .enumerate()
            .map(|(i, g)| (g.full_node, i))
            .collect::<HashMap<_, _>>();
        for root in &roots {
            if !root.term.is_zero() {
                needed[root.term.node as usize] = true;
            }
        }
        for n in (0..nodes.len()).rev() {
            if !needed[n] {
                continue;
            }
            if let RawNode::Sum(terms) = &nodes[n] {
                for t in terms {
                    if !t.is_zero() {
                        needed[t.node as usize] = true;
                    }
                }
            }
        }
        let coefficient = |c| match c {
            0 => ONE,
            1 => NEG_ONE,
            c => c,
        };
        for (i, node) in nodes.iter().enumerate() {
            if let Some(&index) = group_start.get(&(i as u32)) {
                let g = &power_groups[index];
                let p = builder.packed_inputs(g.len as usize, g.low_len as usize);
                assert_eq!(p.columns.start, g.first_column as usize);
                mapped[g.full_node as usize] = Some(p.full);
                if g.low_node != NO_NODE {
                    mapped[g.low_node as usize] = Some(p.low);
                }
            }
            match node {
                RawNode::Input => mapped[i] = Some(builder.input()),
                RawNode::Sum(terms) if needed[i] && mapped[i].is_none() => {
                    let terms = terms
                        .iter()
                        .filter(|t| !t.is_zero())
                        .map(|t| {
                            (
                                mapped[t.node as usize]
                                    .expect("packed scalar input escaped its private aggregate"),
                                coefficient(t.coefficient),
                            )
                        })
                        .collect();
                    mapped[i] = Some(builder.sum_terms(terms));
                }
                _ => {}
            }
        }
        let zero = builder.zero();
        for (slot, root) in roots.into_iter().enumerate() {
            debug_assert_eq!(slot / 3, root.row as usize);
            debug_assert_eq!(slot % 3, root.kind as usize);
            if root.term.is_zero() {
                builder.output(zero);
            } else {
                builder.output_term(
                    mapped[root.term.node as usize].unwrap(),
                    coefficient(root.term.coefficient),
                );
            }
        }
        let core = builder.finish();
        assert_eq!(core.output_count(), 3 * constraints);
        Self { core, constraints }
    }
    pub fn row_count(&self) -> usize {
        self.constraints
    }
    pub fn column_count(&self) -> usize {
        self.core.input_count()
    }
    pub fn node_count(&self) -> usize {
        self.core.node_count()
    }
    pub fn edge_count(&self) -> usize {
        self.core.edge_count()
    }
    pub fn level_count(&self) -> usize {
        self.core.level_count()
    }
    pub fn coefficient_count(&self) -> usize {
        self.core.coefficients().len()
    }
    pub fn payload_bytes(&self) -> usize {
        self.core.topology_bytes() + self.core.coefficients().payload_bytes()
    }

    /// Retain the supplied arithmetic provider; do not rebuild its parameters.
    pub fn prepare_field(&self, field: &FpCtx<2>) -> PreparedWengertEvaluator<'_> {
        let projection = field::PreparedSignedProjection::new(
            field.clone(),
            self.core.coefficients().max_limbs(),
        );
        let view = self.core.coefficients().view();
        let values = (0..view.len())
            .map(|i| projection.project(&view[i]))
            .collect();
        let core = super::PreparedWengert::new(
            &self.core.graph,
            CircuitField(field.clone()),
            std::borrow::Cow::Owned(values),
        );
        PreparedWengertEvaluator {
            core,
            #[cfg(test)]
            output: Vec::new(),
            constraints: self.constraints,
        }
    }
    pub fn prepare(
        &self,
        modulus: &ModRingCtx<2>,
    ) -> Result<PreparedWengertEvaluator<'_>, WengertApplyError> {
        if modulus.modulus().as_words()[0] & 1 == 0 {
            return Err(WengertApplyError::EvenModulus);
        }
        Ok(self.prepare_field(&create_prime_field(*modulus.modulus())))
    }
    #[cfg(test)]
    pub fn apply(
        &self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &ModRingCtx<2>,
    ) -> Result<Vec<[u64; 2]>, WengertApplyError> {
        let mut out = Vec::new();
        self.apply_into(challenges, x, modulus, &mut out)?;
        Ok(out)
    }
    #[cfg(test)]
    pub fn apply_into(
        &self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &ModRingCtx<2>,
        out: &mut Vec<[u64; 2]>,
    ) -> Result<(), WengertApplyError> {
        self.apply_inner(challenges, x, modulus, out, None)
    }
    #[cfg(test)]
    fn apply_inner(
        &self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &ModRingCtx<2>,
        out: &mut Vec<[u64; 2]>,
        parallel: Option<bool>,
    ) -> Result<(), WengertApplyError> {
        let mut p = self.prepare(modulus)?;
        p.core.set_parallel(parallel);
        let weights = challenges
            .iter()
            .map(|&w| p.to_montgomery(w))
            .collect::<Vec<_>>();
        let x = p.to_montgomery(x);
        p.apply(&weights, x)?;
        out.resize(self.column_count(), [0; 2]);
        for (o, &v) in out.iter_mut().zip(&p.output) {
            *o = p.from_montgomery(v);
        }
        Ok(())
    }
}

/// Preserve the circuit backend's qualified FIOS multiplication kernel.
/// Both operands/results remain ordinary Montgomery elements; only execution
/// differs from the provider's default multiplication implementation.
#[derive(Clone)]
struct CircuitField(FpCtx<2>);
impl std::ops::Deref for CircuitField {
    type Target = FpCtx<2>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl RingOps for CircuitField {
    type Elem = field::Fp<2>;
    fn zero(&self) -> Self::Elem {
        self.0.zero()
    }
    fn zero_vec(&self, n: usize) -> Vec<Self::Elem> {
        self.0.zero_vec(n)
    }
    fn one(&self) -> Self::Elem {
        self.0.one()
    }
    #[inline(always)]
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.0.add(a, b)
    }
    #[inline(always)]
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.0.sub(a, b)
    }
    #[inline(always)]
    fn neg(&self, a: &Self::Elem) -> Self::Elem {
        self.0.neg(a)
    }
    #[inline(always)]
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        self.0
            .from_montgomery_integer(self.0.mul_canonical(a.as_montgomery_integer(), b))
    }
}

/// Word-storage adapter retained for existing circuit consumers.
/// New bindings write typed output directly through `adjoint_map_into`.
pub struct PreparedWengertEvaluator<'a> {
    core: super::PreparedWengert<'a, CircuitField>,
    #[cfg(test)]
    output: Vec<[u64; 2]>,
    constraints: usize,
}

#[cfg(test)]
trait ForwardColumns: Sync {
    fn scalar(&self, column: usize) -> [u64; 2];
    fn power_sum(&self, first: usize, len: usize) -> [u64; 2];
}
#[cfg(test)]
struct Columns<'a, V> {
    source: &'a V,
    field: &'a FpCtx<2>,
    len: usize,
}
#[cfg(test)]
impl<V: ForwardColumns> super::ColumnValues<field::Fp<2>> for Columns<'_, V> {
    fn len(&self) -> usize {
        self.len
    }
    fn scalar(&self, j: usize) -> field::Fp<2> {
        self.field
            .from_montgomery_integer(Uint::from_words(self.source.scalar(j)))
    }
    fn power_sum(&self, j: usize, n: usize) -> field::Fp<2> {
        self.field
            .from_montgomery_integer(Uint::from_words(self.source.power_sum(j, n)))
    }
}
impl PreparedWengertEvaluator<'_> {
    pub fn field(&self) -> &FpCtx<2> {
        &self.core.field.0
    }
    pub fn row_count(&self) -> usize {
        self.constraints
    }
    pub fn column_count(&self) -> usize {
        self.core.graph.inputs.len()
    }
    pub fn to_montgomery(&self, v: [u64; 2]) -> [u64; 2] {
        *self
            .core
            .field
            .from_integer(&Uint::from_words(v))
            .as_montgomery_integer()
            .as_words()
    }
    pub fn from_montgomery(&self, v: [u64; 2]) -> [u64; 2] {
        *self
            .core
            .field
            .to_integer(&self.core.field.from_montgomery_integer(Uint::from_words(v)))
            .as_words()
    }
    pub fn workspace_bytes(&self) -> usize {
        let bytes = self.core.workspace_bytes();
        #[cfg(test)]
        let bytes = bytes + self.output.len() * 16;
        bytes
    }
    #[cfg(test)]
    fn check(&self, n: usize) -> Result<(), WengertApplyError> {
        if n == self.constraints {
            Ok(())
        } else {
            Err(WengertApplyError::ChallengeLength {
                expected: self.constraints,
                actual: n,
            })
        }
    }
    #[cfg(test)]
    pub fn apply(
        &mut self,
        weights: &[[u64; 2]],
        x: [u64; 2],
    ) -> Result<&[[u64; 2]], WengertApplyError> {
        self.check(weights.len())?;
        let f = self.core.field.clone();
        let x = f.from_montgomery_integer(Uint::from_words(x));
        let factors = [f.one(), x, f.square(&x)];
        // Hoist batching once per row. Arbitrary supplied row triples use the
        // mapped entry point below and require no staging copy.
        let triples = weights
            .iter()
            .map(|&w| {
                let w = f.from_montgomery_integer(Uint::from_words(w));
                [
                    *w.as_montgomery_integer().as_words(),
                    *f.mul(&w, &factors[1]).as_montgomery_integer().as_words(),
                    *f.mul(&w, &factors[2]).as_montgomery_integer().as_words(),
                ]
            })
            .collect::<Vec<_>>();
        self.apply_weighted(&triples)
    }
    #[cfg(test)]
    pub fn apply_weighted(
        &mut self,
        weights: &[[[u64; 2]; 3]],
    ) -> Result<&[[u64; 2]], WengertApplyError> {
        self.check(weights.len())?;
        let f = self.core.field.clone();
        self.output.resize(self.core.graph.inputs.len(), [0; 2]);
        self.core
            .adjoint_map_storage_into(
                3 * weights.len(),
                |i| f.from_montgomery_integer(Uint::from_words(weights.as_flattened()[i])),
                &mut self.output,
                |e| *e.as_montgomery_integer().as_words(),
            )
            .expect("validated graph dimensions");
        Ok(&self.output)
    }
    /// Emit one compact Montgomery owner without a dense column table.
    pub fn adjoint_map_structured(
        &mut self,
        row_count: usize,
        weights: impl Fn(usize, usize) -> [u64; 2] + Sync,
    ) -> Result<crate::montgomery_tail::MontgomeryTail, &'static str> {
        use crate::montgomery_tail::{MontgomeryTail, TailSegment};
        if row_count != self.constraints {
            return Err("wrong number of tape row weights");
        }
        let f = self.core.field.clone();
        for row in 0..row_count {
            for kind in 0..3 {
                if u128::from(Uint::from_words(weights(row, kind))) >= u128::from(*f.modulus()) {
                    return Err("noncanonical Montgomery row weight");
                }
            }
        }
        let run_len: usize = self.core.graph.groups.iter().map(|g| g.len as usize).sum();
        let mut literals = Vec::with_capacity(self.column_count() - run_len);
        let mut segments = Vec::with_capacity(self.core.graph.groups.len() * 2 + 1);
        #[cfg(test)]
        {
            self.output = Vec::new();
        }
        self.core
            .adjoint_map_segments(
                3 * row_count,
                |i| f.from_montgomery_integer(Uint::from_words(weights(i / 3, i % 3))),
                |run| {
                    let base = *run.base.as_montgomery_integer().as_words();
                    if run.len == 1 {
                        let first = literals.len();
                        literals.push(base);
                        if let Some(TailSegment::Literals { start, values }) = segments.last_mut()
                            && *start + values.len() == run.first_column
                        {
                            values.end += 1;
                        } else {
                            segments.push(TailSegment::Literals {
                                start: run.first_column,
                                values: first..first + 1,
                            });
                        }
                    } else if base != [0; 2] {
                        segments.push(TailSegment::Geometric {
                            start: run.first_column,
                            len: run.len,
                            base,
                        });
                    }
                },
            )
            .map_err(|_| "wrong number of tape row weights")?;
        MontgomeryTail::new(f.0, self.column_count(), segments, literals)
    }

    pub fn apply_weighted_structured(
        &mut self,
        weights: &[[[u64; 2]; 3]],
    ) -> Result<crate::montgomery_tail::MontgomeryTail, &'static str> {
        self.adjoint_map_structured(weights.len(), |row, kind| weights[row][kind])
    }

    pub fn adjoint_map_into(
        &mut self,
        row_count: usize,
        weights: impl Fn(usize, usize) -> [u64; 2] + Sync,
        out: &mut [field::Fp<2>],
    ) -> Result<(), super::LinearMapError> {
        let f = self.core.field.clone();
        self.core.adjoint_map_into(
            3 * row_count,
            |i| f.from_montgomery_integer(Uint::from_words(weights(i / 3, i % 3))),
            out,
        )
    }
    #[cfg(test)]
    pub fn apply_forward_weighted<V: ForwardColumns>(
        &mut self,
        weights: &[[[u64; 2]; 3]],
        columns: &V,
    ) -> Result<[u64; 2], WengertApplyError> {
        self.check(weights.len())?;
        let field = self.core.field.0.clone();
        let columns = Columns {
            source: columns,
            field: &field,
            len: self.column_count(),
        };
        let result = self
            .evaluate_bilinear_map(weights.len(), |r, k| weights[r][k], &columns)
            .expect("validated graph dimensions");
        Ok(*result.as_montgomery_integer().as_words())
    }
    /// Mapped row seeds avoid allocating triples in protocol adapters.
    #[doc(hidden)]
    pub fn evaluate_bilinear_map(
        &mut self,
        row_count: usize,
        weights: impl Fn(usize, usize) -> [u64; 2] + Sync,
        columns: &impl ColumnValues<field::Fp<2>>,
    ) -> Result<field::Fp<2>, super::LinearMapError> {
        let f = self.core.field.clone();
        self.core.evaluate_bilinear_map(
            3 * row_count,
            |i| f.from_montgomery_integer(Uint::from_words(weights(i / 3, i % 3))),
            columns,
        )
    }
    pub fn power_runs(&self) -> Vec<PowerRun> {
        self.core
            .power_runs()
            .into_iter()
            .map(|r| PowerRun {
                first_column: r.first_column,
                len: r.len,
                base: *r.base.as_montgomery_integer().as_words(),
            })
            .collect()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowerRun {
    pub first_column: usize,
    pub len: usize,
    pub base: [u64; 2],
}

/// The input/output words may be canonical residues or Montgomery encodings;
/// addition and scaling by a Montgomery coefficient preserve that representation.
#[inline(always)]
pub(crate) fn mul_representatives(
    left: [u64; 2],
    coefficient: [u64; 2],
    field: &FpCtx<2>,
) -> [u64; 2] {
    *field
        .mul_canonical(
            &Uint::from_words(left),
            &field.from_montgomery_integer(Uint::from_words(coefficient)),
        )
        .as_words()
}

#[inline(always)]
pub(crate) fn add_representatives(left: [u64; 2], right: [u64; 2], field: &FpCtx<2>) -> [u64; 2] {
    *field
        .add(
            &field.from_montgomery_integer(Uint::from_words(left)),
            &field.from_montgomery_integer(Uint::from_words(right)),
        )
        .as_montgomery_integer()
        .as_words()
}

#[inline(always)]
#[cfg(test)]
pub(crate) fn neg_representative(value: [u64; 2], field: &FpCtx<2>) -> [u64; 2] {
    *field
        .neg(&field.from_montgomery_integer(Uint::from_words(value)))
        .as_montgomery_integer()
        .as_words()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WengertApplyError {
    ChallengeLength { expected: usize, actual: usize },
    EvenModulus,
}

impl Display for WengertApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChallengeLength { expected, actual } => write!(
                formatter,
                "challenge vector has length {actual}, expected {expected}"
            ),
            Self::EvenModulus => formatter.write_str("Wengert evaluation requires an odd modulus"),
        }
    }
}

impl Error for WengertApplyError {}

/// Records and preprocesses the circuit's Z-side linear arithmetic.
#[derive(Debug)]
pub struct WengertGenerator {
    recorder: Arc<SharedRecorder>,
    inputs: Box<[WengertBit]>,
}

impl WengertGenerator {
    /// Creates a generator with the circuit's Boolean input shape.
    pub fn new(input_count: usize) -> Self {
        Self {
            recorder: Arc::new(SharedRecorder(Mutex::new(Some(Recorder::new())))),
            inputs: vec![WengertBit; input_count].into_boxed_slice(),
        }
    }

    /// Moves every Boolean input handle into a fixed-size boxed array.
    pub fn take_boxed_inputs<const N: usize>(&mut self) -> Box<[WengertBit; N]> {
        assert_eq!(N, self.inputs.len(), "input witness count mismatch");
        std::mem::take(&mut self.inputs)
            .try_into()
            .unwrap_or_else(|_| unreachable!("input length was checked"))
    }

    /// Moves dynamically sized Boolean input handles out.
    pub fn take_inputs(&mut self) -> Box<[WengertBit]> {
        std::mem::take(&mut self.inputs)
    }

    /// Prunes, transposes, and reverse-depth-orders the recorded graph.
    pub fn finish(self) -> WengertTape {
        let recorder = self
            .recorder
            .0
            .lock()
            .expect("Wengert recorder lock poisoned")
            .take()
            .expect("the Wengert generator has already been finished");
        WengertTape::from_recorder(recorder)
    }
}

impl Circuit for WengertGenerator {
    type Bool = WengertBit;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = WengertValue<LIMBS>;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Z<LIMBS> {
        crate::witgen::integer_from_words(words)
    }

    fn xor(&mut self, _: WengertBit, _: WengertBit) -> WengertBit {
        WengertBit
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<WengertBit, N>
    where
        H: Fn(
                &dyn WitnessContext<WengertValue<LIMBS>, WengertBit, Z<LIMBS>>,
            ) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        ScalarBits([WengertBit; N])
    }

    fn bitz<const LIMBS: usize>(&mut self, _: WengertBit) -> WengertValue<LIMBS> {
        let node = self.recorder.with_mut(Recorder::push_input);
        WengertValue::attached(self.recorder.clone(), node, Z::one())
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        _: &<WengertBit as BoolWitness>::Repr<N, M>,
    ) -> (WengertValue<LIMBS>, WengertValue<LIMBS>) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        if N == 0 {
            return (WengertValue::zero(), WengertValue::zero());
        }
        // These bit columns are private to their packed aggregates. Record the
        // ranges directly instead of constructing and then discarding N input
        // nodes, N powers-of-two coefficients and the per-bit sum edges.
        let (full_node, low_node) = self.recorder.with_mut(|recorder| {
            let first_column = u32::try_from(recorder.input_nodes.len())
                .expect("too many integer-witness columns");
            let end = recorder
                .input_nodes
                .len()
                .checked_add(N)
                .expect("column count overflow");
            recorder.input_nodes.resize(end, NO_NODE);
            let full_node = u32::try_from(recorder.nodes.len()).expect("too many Wengert nodes");
            recorder.nodes.push(RawNode::Packed);
            let low_node = if LOW == 0 || LOW == N {
                NO_NODE
            } else {
                let low = u32::try_from(recorder.nodes.len()).expect("too many Wengert nodes");
                recorder.nodes.push(RawNode::Packed);
                low
            };
            recorder.power_groups.push(RawPowerGroup {
                first_column,
                len: u32::try_from(N).expect("power group is too wide"),
                low_len: u32::try_from(LOW).expect("low power group is too wide"),
                full_node,
                low_node,
            });
            (full_node, low_node)
        });
        let full = WengertValue::attached(self.recorder.clone(), full_node, Z::one());
        let low = if LOW == N {
            full.clone()
        } else if LOW == 0 {
            WengertValue::zero()
        } else {
            WengertValue::attached(self.recorder.clone(), low_node, Z::one())
        };
        (full, low)
    }

    fn assert_r1c<const LIMBS: usize>(
        &mut self,
        a: WengertValue<LIMBS>,
        b: WengertValue<LIMBS>,
        c: WengertValue<LIMBS>,
    ) {
        for value in [&a, &b, &c] {
            if let Some(recorder) = value.recorder() {
                WengertValue::<LIMBS>::same_recorder(&self.recorder, recorder);
            }
        }
        self.recorder.with_mut(|recorder| {
            let row = u32::try_from(recorder.constraints).expect("too many R1CS rows");
            recorder.constraints += 1;
            let terms = [
                a.raw_term(recorder),
                b.raw_term(recorder),
                c.raw_term(recorder),
            ];
            recorder.roots.extend([
                RawRoot {
                    row,
                    kind: RootKind::A,
                    term: terms[0],
                },
                RawRoot {
                    row,
                    kind: RootKind::B,
                    term: terms[1],
                },
                RawRoot {
                    row,
                    kind: RootKind::C,
                    term: terms[2],
                },
            ]);
        });
    }

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: WengertValue<FROM_LIMBS>,
    ) -> WengertValue<TO_LIMBS> {
        assert!(
            TO_LIMBS >= FROM_LIMBS,
            "cannot sign-extend into fewer limbs"
        );
        WengertValue {
            location: value.location,
            coefficient: value.coefficient.sign_extend(),
        }
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::{BigInt, BigUint};
    use num_traits::Signed;

    use super::*;
    use crate::constraints::{ConstraintGenerator, ConstraintMatrices};
    use crate::sha256::{COMPRESSION_INPUT_BITS, compression_circuit};
    use crypto_bigint::modular::{FixedMontyForm, FixedMontyParams};
    use crypto_bigint::{Odd, U128};

    #[test]
    fn overflowing_node_coefficients_remain_graph_operations() {
        let mut g = WengertGenerator::new(1);
        let a = g.bitz::<1>(WengertBit);
        let max = a.clone() * Z::<1>::MAX;
        let minimum = a * Z::<1>::MIN;
        g.assert_r1c(
            max.clone() + max.clone(),
            max * Z::<1>::from(2u64),
            -minimum,
        );
        let tape = g.finish();
        let field = create_prime_field(Uint::<2>::from(97u64));
        let mut p = tape.prepare_field(&field);
        let one = *field.one().as_montgomery_integer().as_words();
        let mut out = field.zero_vec(tape.column_count());
        p.adjoint_map_into(1, |_, _| one, &mut out).unwrap();
        assert_eq!(out[0], field.zero());
        assert_eq!(out[1], field.from_integer(&((5u128 << 63) - 4)));
    }

    fn example_circuit<CS: Circuit>(circuit: &mut CS, inputs: &[CS::Bool; 3]) {
        let a = circuit.bitz::<2>(inputs[0].clone());
        let b = circuit.bitz::<2>(inputs[1].clone());
        let c = circuit.bitz::<2>(inputs[2].clone());
        let seven = CS::Coefficient::<2>::from(7);
        let eleven = CS::Coefficient::<2>::from(11);
        let thirteen = CS::Coefficient::<2>::from(13);
        circuit.assert_r1c(
            a.clone() * seven.clone() - b.clone(),
            b.clone() * eleven + CS::Z::<2>::from(CS::Coefficient::<2>::from(5)),
            c.clone() * thirteen,
        );
        circuit.assert_r1c(
            a + c.clone(),
            -c,
            b + CS::Z::<2>::from(CS::Coefficient::<2>::from(19)),
        );
    }

    fn build_tape() -> WengertTape {
        let mut generator = WengertGenerator::new(3);
        let inputs = generator.take_boxed_inputs();
        example_circuit(&mut generator, &inputs);
        generator.finish()
    }

    fn direct_product(
        matrices: &ConstraintMatrices,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &BigUint,
    ) -> Vec<[u64; 2]> {
        let modulus_int = BigInt::from(modulus.clone());
        let as_bigint = |words: [u64; 2]| {
            BigInt::from(BigUint::from(words[0]) + (BigUint::from(words[1]) << 64_usize))
        };
        let x = as_bigint(x);
        let x_squared = &x * &x;
        let mut output = vec![BigInt::zero(); matrices.a.column_count()];
        for (row, challenge) in challenges.iter().enumerate() {
            let challenge = as_bigint(*challenge);
            for (column, coefficient) in matrices
                .a
                .row(row)
                .unwrap()
                .iter()
                .map(|(column, coefficient)| (column, coefficient.as_words()))
            {
                let coefficient = BigInt::from_signed_bytes_le(
                    &coefficient
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                output[column] += &challenge * coefficient;
            }
            for (column, coefficient) in matrices
                .b
                .row(row)
                .unwrap()
                .iter()
                .map(|(column, coefficient)| (column, coefficient.as_words()))
            {
                let coefficient = BigInt::from_signed_bytes_le(
                    &coefficient
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                output[column] += &challenge * &x * coefficient;
            }
            for (column, coefficient) in matrices
                .c
                .row(row)
                .unwrap()
                .iter()
                .map(|(column, coefficient)| (column, coefficient.as_words()))
            {
                let coefficient = BigInt::from_signed_bytes_le(
                    &coefficient
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                output[column] += &challenge * &x_squared * coefficient;
            }
        }
        output
            .into_iter()
            .map(|mut value| {
                value %= &modulus_int;
                if value.is_negative() {
                    value += &modulus_int;
                }
                let words = value.to_biguint().unwrap().to_u64_digits();
                [
                    words.first().copied().unwrap_or(0),
                    words.get(1).copied().unwrap_or(0),
                ]
            })
            .collect()
    }

    #[test]
    fn tape_matches_materialized_matrices_for_primes_known_afterward() {
        let tape = build_tape();
        let mut generator = ConstraintGenerator::new(3);
        let inputs = generator.inputs();
        example_circuit(&mut generator, &inputs);
        let matrices = generator.into_matrices();
        let challenges = [[23, 0], [29, 0]];
        let x = [17, 0];

        for modulus in [
            (BigUint::one() << 127_usize) - BigUint::one(),
            (BigUint::one() << 128_usize) - BigUint::from(159_u64),
        ] {
            let runtime = ModRingCtx::<2>::new(field::Uint::from_words(
                crate::matrix_products::biguint_words(&(modulus.clone())),
            ))
            .unwrap();
            assert_eq!(
                tape.apply(&challenges, x, &runtime).unwrap(),
                direct_product(&matrices, &challenges, x, &modulus)
            );
        }
    }

    #[test]
    fn canonical_apply_handles_unreduced_inputs_and_reuses_output() {
        let tape = build_tape();
        let mut generator = ConstraintGenerator::new(3);
        let inputs = generator.inputs();
        example_circuit(&mut generator, &inputs);
        let matrices = generator.into_matrices();
        let words = |value: u128| [value as u64, (value >> 64) as u64];
        let mut output = vec![[u64::MAX; 2]; tape.column_count() + 3];
        let output_ptr = output.as_ptr();
        let output_capacity = output.capacity();

        for prime in [
            3,
            101,
            (1_u128 << 64) - 59,
            (1_u128 << 127) - 1,
            u128::MAX - 158,
        ] {
            let modulus = BigUint::from(prime);
            let runtime = ModRingCtx::<2>::new(field::Uint::from_words(
                crate::matrix_products::biguint_words(&(modulus.clone())),
            ))
            .unwrap();
            let mut prepared = tape.prepare(&runtime).unwrap();
            let cases = [0, 1, prime - 1, prime, prime + 1, u128::MAX];
            for (i, &r) in cases.iter().enumerate() {
                let challenges = [words(r), words(cases[(i + 1) % cases.len()])];
                let montgomery_challenges = challenges.map(|r| prepared.to_montgomery(r));
                for &x in &cases {
                    let x = words(x);
                    let expected = direct_product(&matrices, &challenges, x, &modulus);
                    for parallel in [false, true] {
                        output.resize(tape.column_count() + 3, [u64::MAX; 2]);
                        output.fill([u64::MAX; 2]);
                        tape.apply_inner(&challenges, x, &runtime, &mut output, Some(parallel))
                            .unwrap();
                        assert_eq!(
                            output, expected,
                            "q={prime}, r={r}, x={x:?}, parallel={parallel}"
                        );
                        assert_eq!(output.as_ptr(), output_ptr);
                        assert_eq!(output.capacity(), output_capacity);
                    }

                    let montgomery_x = prepared.to_montgomery(x);
                    let expected_encoded: Vec<_> = expected
                        .iter()
                        .map(|value| prepared.to_montgomery(*value))
                        .collect();
                    let encoded = prepared
                        .apply(&montgomery_challenges, montgomery_x)
                        .unwrap();
                    assert_eq!(encoded, expected_encoded);
                }
            }
        }
    }

    #[test]
    fn parallel_and_sequential_reverse_batches_agree() {
        let tape = build_tape();
        let modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(
                &((BigUint::one() << 128_usize) - BigUint::from(159_u64)),
            ),
        ))
        .unwrap();
        let challenges = [[0x1234_5678_9abc_def0, 7], [0x0fed_cba9_8765_4321, 11]];
        let mut sequential = Vec::new();
        let mut parallel = Vec::new();
        tape.apply_inner(&challenges, [31, 3], &modulus, &mut sequential, Some(false))
            .unwrap();
        tape.apply_inner(&challenges, [31, 3], &modulus, &mut parallel, Some(true))
            .unwrap();
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn prepared_evaluator_reuses_storage_and_matches_one_shot_apply() {
        let tape = build_tape();
        let modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(
                &((BigUint::one() << 128_usize) - BigUint::from(159_u64)),
            ),
        ))
        .unwrap();
        let challenges = [[0x1234_5678_9abc_def0, 7], [0x0fed_cba9_8765_4321, 11]];
        let x = [31, 3];
        let expected = tape.apply(&challenges, x, &modulus).unwrap();
        let mut evaluator = tape.prepare(&modulus).unwrap();
        let montgomery_challenges: Vec<_> = challenges
            .into_iter()
            .map(|challenge| evaluator.to_montgomery(challenge))
            .collect();
        let montgomery_x = evaluator.to_montgomery(x);

        for _ in 0..2 {
            let output = evaluator
                .apply(&montgomery_challenges, montgomery_x)
                .unwrap()
                .to_vec();
            let output: Vec<_> = output
                .into_iter()
                .map(|value| evaluator.from_montgomery(value))
                .collect();
            assert_eq!(output, expected);
        }
    }

    /// Dense column values for the forward-pass test: every column value is
    /// stored, and a power group's sum is the plain doubling chain.
    struct DenseColumns {
        values: Vec<[u64; 2]>,
        field: FpCtx<2>,
    }

    impl ForwardColumns for DenseColumns {
        fn scalar(&self, column: usize) -> [u64; 2] {
            self.values[column]
        }

        fn power_sum(&self, first: usize, len: usize) -> [u64; 2] {
            let mut sum = [0; 2];
            for k in (0..len).rev() {
                sum = add_representatives(sum, sum, &self.field);
                sum = add_representatives(sum, self.values[first + k], &self.field);
            }
            sum
        }
    }

    /// The forward pass is the reverse pass's column vector dotted with the
    /// column values: on the example tape and the SHA-256 compression tape,
    /// random weight triples and random column values, at `2^127 − 1` and
    /// `2^128 − 159`.
    #[test]
    fn forward_pass_matches_reverse_dot_product() {
        let mut sha_generator = WengertGenerator::new(COMPRESSION_INPUT_BITS);
        let sha_inputs = sha_generator.take_boxed_inputs();
        let _ = compression_circuit(&mut sha_generator, &sha_inputs);
        let tapes = [build_tape(), sha_generator.finish()];
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for modulus in [
            (BigUint::one() << 127_usize) - BigUint::one(),
            (BigUint::one() << 128_usize) - BigUint::from(159_u64),
        ] {
            let runtime = ModRingCtx::<2>::new(Uint::from_words(
                crate::matrix_products::biguint_words(&modulus),
            ))
            .unwrap();
            for tape in &tapes {
                let mut evaluator = tape.prepare(&runtime).unwrap();
                let element = |random: &mut dyn FnMut() -> u64| {
                    *evaluator
                        .core
                        .field
                        .from_integer(&Uint::from_words([random(), random()]))
                        .as_montgomery_integer()
                        .as_words()
                };
                let weights: Vec<[[u64; 2]; 3]> = (0..tape.row_count())
                    .map(|_| {
                        [
                            element(&mut random),
                            element(&mut random),
                            element(&mut random),
                        ]
                    })
                    .collect();
                let columns = DenseColumns {
                    values: (0..tape.column_count())
                        .map(|_| element(&mut random))
                        .collect(),
                    field: evaluator.core.field.0.clone(),
                };
                assert!(evaluator.output.is_empty());
                let forward = evaluator
                    .apply_forward_weighted(&weights, &columns)
                    .unwrap();
                assert!(evaluator.output.is_empty());
                let reverse = evaluator.apply_weighted(&weights).unwrap().to_vec();
                let compact = evaluator.apply_weighted_structured(&weights).unwrap();
                assert!(evaluator.output.is_empty());
                for (column, value) in reverse.iter().enumerate() {
                    assert_eq!(compact.value(column), Some(*value));
                }
                assert!(evaluator.apply_weighted_structured(&weights[1..]).is_err());
                let mut invalid = weights.clone();
                invalid[0][0] = *evaluator.core.field.modulus().as_words();
                assert!(evaluator.apply_weighted_structured(&invalid).is_err());
                let mut dot = [0; 2];
                for (output, value) in reverse.iter().zip(&columns.values) {
                    let term = mul_representatives(*output, *value, &evaluator.core.field);
                    dot = add_representatives(dot, term, &evaluator.core.field);
                }
                assert_eq!(forward, dot);
                assert_eq!(
                    evaluator.apply_forward_weighted(&weights[1..], &columns),
                    Err(WengertApplyError::ChallengeLength {
                        expected: tape.row_count(),
                        actual: tape.row_count() - 1,
                    })
                );
            }
        }
    }

    #[test]
    fn two_limb_montgomery_kernel_matches_crypto_bigint() {
        let modulus_words = [u64::MAX - 158, u64::MAX];
        let modulus = U128::from_words(modulus_words);
        let params = FixedMontyParams::new_vartime(Odd::new(modulus).unwrap());
        let field = create_prime_field(Uint::from_words(modulus_words));
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..1_000 {
            let canonical_left = U128::from_words(
                *field
                    .reduce_integer(&Uint::from_words([random(), random()]))
                    .as_words(),
            );
            let canonical_right = U128::from_words(
                *field
                    .reduce_integer(&Uint::from_words([random(), random()]))
                    .as_words(),
            );
            let left = FixedMontyForm::new(&canonical_left, &params);
            let right = FixedMontyForm::new(&canonical_right, &params);
            let expected = (left * right).to_montgomery().to_words();
            let actual = mul_representatives(
                left.to_montgomery().to_words(),
                right.to_montgomery().to_words(),
                &field,
            );

            assert_eq!(actual, expected);
            assert_eq!(
                *field
                    .to_integer(&field.from_montgomery_integer(Uint::from_words(actual)))
                    .as_words(),
                (left * right).retrieve().to_words()
            );
        }
    }

    #[test]
    fn sha256_compression_tape_matches_materialized_sparse_matrices() {
        let mut tape_generator = WengertGenerator::new(COMPRESSION_INPUT_BITS);
        let tape_inputs = tape_generator.take_boxed_inputs();
        let _ = compression_circuit(&mut tape_generator, &tape_inputs);
        let tape = tape_generator.finish();

        let mut matrix_generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
        let matrix_inputs = matrix_generator.boxed_inputs();
        let _ = compression_circuit(&mut matrix_generator, &matrix_inputs);
        let matrices = matrix_generator.into_matrices();

        let challenges: Vec<_> = (0..tape.row_count())
            .map(|row| [(17 * row + 3) as u64, (5 * row + 1) as u64])
            .collect();
        let x = [0x1234_5678_9abc_def0, 0x0123_4567_89ab_cdef];
        let prime = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(prime.clone())),
        ))
        .unwrap();

        assert_eq!(tape.row_count(), matrices.a.row_count());
        assert_eq!(tape.column_count(), matrices.a.column_count());
        assert_eq!(
            tape.apply(&challenges, x, &modulus).unwrap(),
            direct_product(&matrices, &challenges, x, &prime)
        );
        assert!(tape.node_count() <= tape.edge_count());
        assert!(tape.payload_bytes() < 4 * 1024 * 1024);
    }

    #[test]
    fn rejects_wrong_challenge_length_and_even_modulus() {
        let tape = build_tape();
        let odd = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(BigUint::from(101_u64))),
        ))
        .unwrap();
        assert_eq!(
            tape.apply(&[[1, 0]], [2, 0], &odd),
            Err(WengertApplyError::ChallengeLength {
                expected: 2,
                actual: 1,
            })
        );
        let even = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(BigUint::from(100_u64))),
        ))
        .unwrap();
        assert_eq!(
            tape.apply(&[[1, 0], [2, 0]], [3, 0], &even),
            Err(WengertApplyError::EvenModulus)
        );
    }

    #[test]
    fn dead_arithmetic_is_pruned_and_unused_inputs_return_zero() {
        let mut generator = WengertGenerator::new(2);
        let inputs = generator.take_boxed_inputs::<2>();
        let used = generator.bitz::<1>(inputs[0]);
        let unused = generator.bitz::<1>(inputs[1]);
        let _dead = unused.clone() + unused;
        generator.assert_r1c(used, WengertValue::zero(), WengertValue::zero());
        let tape = generator.finish();
        let modulus = ModRingCtx::<2>::new(field::Uint::from_words(
            crate::matrix_products::biguint_words(&(BigUint::from(101_u64))),
        ))
        .unwrap();
        assert_eq!(
            tape.apply(&[[7, 0]], [3, 0], &modulus).unwrap(),
            [[0, 0], [7, 0], [0, 0]]
        );
        assert_eq!(tape.node_count(), 1);
    }
}

impl LeftMul<field::Fp<2>> for PreparedWengertEvaluator<'_> {
    type Output = field::Fp<2>;
    fn mul_left_into(
        &mut self,
        weights: &[field::Fp<2>],
        out: &mut [field::Fp<2>],
    ) -> Result<(), super::LinearMapError> {
        self.core.mul_left_into(weights, out)
    }
}
impl RightMul<field::Fp<2>> for PreparedWengertEvaluator<'_> {
    type Output = field::Fp<2>;
    fn mul_right_into(
        &mut self,
        values: &[field::Fp<2>],
        out: &mut [field::Fp<2>],
    ) -> Result<(), super::LinearMapError> {
        self.core.mul_right_into(values, out)
    }
}
impl BilinearEval<FpCtx<2>> for PreparedWengertEvaluator<'_> {
    fn evaluate_bilinear(
        &mut self,
        weights: &[field::Fp<2>],
        columns: &impl ColumnValues<field::Fp<2>>,
    ) -> Result<field::Fp<2>, super::LinearMapError> {
        self.core.evaluate_bilinear(weights, columns)
    }
}
