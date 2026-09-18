//! Explicit, value-free construction of a linear operator.
use super::{CoefficientStore, StoreCoefficient, WengertTape};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const NONE: u32 = u32::MAX;
pub(super) const ONE: u32 = u32::MAX;
pub(super) const NEG_ONE: u32 = u32::MAX - 1;
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId {
    owner: u64,
    pub(super) index: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputId(u32);
impl OutputId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

pub struct PackedInputs {
    pub columns: Range<usize>,
    pub full: NodeId,
    pub low: NodeId,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Term {
    pub node: u32,
    pub coefficient: u32,
}
#[derive(Debug)]
pub(super) enum Node {
    Input,
    Sum(Box<[Term]>),
    Packed,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct PowerGroup {
    pub first: u32,
    pub len: u32,
    pub low_len: u32,
    pub full: u32,
    pub low: u32,
}

pub struct WengertBuilder<S> {
    pub(super) coefficients: S,
    owner: u64,
    pub(super) nodes: Vec<Node>,
    pub(super) inputs: Vec<u32>,
    pub(super) outputs: Vec<Term>,
    pub(super) groups: Vec<PowerGroup>,
}

impl<S: CoefficientStore> WengertBuilder<S> {
    pub fn new(coefficients: S) -> Self {
        let owner = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
        assert_ne!(owner, 0, "graph ownership token overflow");
        Self {
            coefficients,
            owner,
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            groups: Vec::new(),
        }
    }
    fn check(&self, value: NodeId) -> u32 {
        assert_eq!(
            value.owner, self.owner,
            "node belongs to another Wengert builder"
        );
        assert!(
            (value.index as usize) < self.nodes.len(),
            "unknown Wengert node"
        );
        value.index
    }
    fn push(&mut self, node: Node) -> NodeId {
        let index = u32::try_from(self.nodes.len()).expect("too many Wengert nodes");
        assert_ne!(index, NONE);
        self.nodes.push(node);
        NodeId {
            owner: self.owner,
            index,
        }
    }
    pub fn input(&mut self) -> NodeId {
        let node = self.push(Node::Input);
        self.inputs.push(node.index);
        node
    }
    pub fn zero(&mut self) -> NodeId {
        self.push(Node::Sum(Box::new([])))
    }
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.sum(&[lhs, rhs])
    }
    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let terms = [
            Term {
                node: self.check(lhs),
                coefficient: ONE,
            },
            Term {
                node: self.check(rhs),
                coefficient: NEG_ONE,
            },
        ];
        self.push(Node::Sum(Box::new(terms)))
    }
    pub fn sum(&mut self, inputs: &[NodeId]) -> NodeId {
        let terms = inputs
            .iter()
            .map(|&n| Term {
                node: self.check(n),
                coefficient: ONE,
            })
            .collect();
        self.push(Node::Sum(terms))
    }
    pub fn scale<C>(&mut self, input: NodeId, coefficient: C) -> NodeId
    where
        S: StoreCoefficient<C>,
    {
        self.linear_combination([(input, coefficient)])
    }
    pub fn linear_combination<C>(&mut self, terms: impl IntoIterator<Item = (NodeId, C)>) -> NodeId
    where
        S: StoreCoefficient<C>,
    {
        let terms = terms
            .into_iter()
            .map(|(node, coefficient)| {
                let node = self.check(node);
                let coefficient = self.coefficients.store(coefficient);
                assert!(coefficient < NEG_ONE as usize, "too many coefficients");
                Term {
                    node,
                    coefficient: coefficient as u32,
                }
            })
            .collect();
        // Deliberately do not multiply nested labels or combine duplicates:
        // coefficient domains may have fixed-width wrapping operators.
        self.push(Node::Sum(terms))
    }
    pub fn packed_inputs(&mut self, bits: usize, low_bits: usize) -> PackedInputs {
        assert!(bits > 0 && low_bits <= bits, "invalid packed input widths");
        let start = self.inputs.len();
        let end = start.checked_add(bits).expect("input count overflow");
        let first = u32::try_from(start).expect("too many inputs");
        let len = u32::try_from(bits).expect("packed input too wide");
        self.inputs.resize(end, NONE);
        let full = self.push(Node::Packed);
        let low = if low_bits == bits {
            full
        } else if low_bits == 0 {
            self.zero()
        } else {
            self.push(Node::Packed)
        };
        self.groups.push(PowerGroup {
            first,
            len,
            low_len: low_bits as u32,
            full: full.index,
            low: if low_bits == bits || low_bits == 0 {
                NONE
            } else {
                low.index
            },
        });
        PackedInputs {
            columns: start..end,
            full,
            low,
        }
    }
    pub fn output(&mut self, value: NodeId) -> OutputId {
        self.output_term(value, ONE)
    }
    pub(super) fn output_term(&mut self, value: NodeId, coefficient: u32) -> OutputId {
        let node = self.check(value);
        let index = u32::try_from(self.outputs.len()).expect("too many outputs");
        let node = if matches!(&self.nodes[node as usize], Node::Sum(terms) if terms.is_empty()) {
            NONE
        } else {
            node
        };
        self.outputs.push(Term { node, coefficient });
        OutputId(index)
    }
    pub fn finish(self) -> WengertTape<S> {
        super::compile::compile(self)
    }

    /// Circuit adapters have already interned their exact coefficient labels.
    pub(super) fn sum_terms(&mut self, terms: Vec<(NodeId, u32)>) -> NodeId {
        let terms = terms
            .into_iter()
            .map(|(node, coefficient)| Term {
                node: self.check(node),
                coefficient,
            })
            .collect();
        self.push(Node::Sum(terms))
    }
}
