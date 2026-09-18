use super::WengertTape;
use super::builder::{NONE, Node, PowerGroup, Term, WengertBuilder};

#[derive(Clone, Copy, Debug)]
pub(super) struct Root {
    pub output: u32,
    pub coefficient: u32,
}

/// Immutable dual adjacency. Each level writes disjoint nodes.
#[derive(Debug)]
pub(super) struct CompiledGraph {
    pub inputs: Box<[u32]>,
    pub output_count: usize,
    pub max_terms: usize,
    pub max_power_len: usize,
    pub levels: Box<[usize]>,
    pub edge_offsets: Box<[u32]>,
    pub edges: Box<[Term]>,
    pub root_offsets: Box<[u32]>,
    pub roots: Box<[Root]>,
    pub forward_offsets: Box<[u32]>,
    pub forward: Box<[Term]>,
    pub groups: Box<[PowerGroup]>,
}
impl CompiledGraph {
    pub fn node_count(&self) -> usize {
        self.edge_offsets.len() - 1
    }
    pub fn payload_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.inputs)
            + std::mem::size_of_val(&*self.levels)
            + std::mem::size_of_val(&*self.edge_offsets)
            + std::mem::size_of_val(&*self.edges)
            + std::mem::size_of_val(&*self.root_offsets)
            + std::mem::size_of_val(&*self.roots)
            + std::mem::size_of_val(&*self.forward_offsets)
            + std::mem::size_of_val(&*self.forward)
            + std::mem::size_of_val(&*self.groups)
    }
}

fn offsets(counts: &[usize]) -> Vec<u32> {
    let mut result = Vec::with_capacity(counts.len() + 1);
    result.push(0u32);
    for &n in counts {
        result.push(
            result
                .last()
                .unwrap()
                .checked_add(u32::try_from(n).expect("too many edges"))
                .expect("too many edges"),
        );
    }
    result
}

pub(super) fn compile<S: super::CoefficientStore>(builder: WengertBuilder<S>) -> WengertTape<S> {
    let WengertBuilder {
        mut coefficients,
        nodes,
        inputs,
        outputs,
        groups,
        ..
    } = builder;
    let mut live = vec![false; nodes.len()];
    for root in &outputs {
        if root.node != NONE {
            live[root.node as usize] = true;
        }
    }
    let mut depth = vec![0usize; nodes.len()];
    for n in (0..nodes.len()).rev() {
        if !live[n] {
            continue;
        }
        if let Node::Sum(terms) = &nodes[n] {
            for term in terms {
                let source = term.node as usize;
                assert!(source < n, "graph must be constructed in topological order");
                live[source] = true;
                depth[source] = depth[source].max(depth[n] + 1);
            }
        }
    }
    let level_count = nodes
        .iter()
        .enumerate()
        .filter(|(i, n)| live[*i] && !matches!(n, Node::Input))
        .map(|(i, _)| depth[i] + 1)
        .max()
        .unwrap_or(0);
    let mut levels = vec![0usize; level_count + 1];
    for (i, n) in nodes.iter().enumerate() {
        if live[i] && !matches!(n, Node::Input) {
            levels[depth[i] + 1] += 1;
        }
    }
    for i in 0..level_count {
        levels[i + 1] += levels[i];
    }
    let internal = *levels.last().unwrap();
    let count = live.iter().filter(|&&x| x).count();
    let mut ordered = vec![0usize; count];
    let mut cursors = levels.clone();
    let mut next_input = internal;
    for (i, n) in nodes.iter().enumerate() {
        if !live[i] {
            continue;
        }
        let dest = if matches!(n, Node::Input) {
            let d = next_input;
            next_input += 1;
            d
        } else {
            let d = cursors[depth[i]];
            cursors[depth[i]] += 1;
            d
        };
        ordered[dest] = i;
    }
    let mut remap = vec![NONE; nodes.len()];
    for (new, &old) in ordered.iter().enumerate() {
        remap[old] = u32::try_from(new).expect("too many nodes");
    }
    let mut edge_counts = vec![0usize; count];
    let mut forward_counts = vec![0usize; count];
    let mut root_counts = vec![0usize; count];
    for (target, &old) in ordered.iter().enumerate() {
        if let Node::Sum(terms) = &nodes[old] {
            forward_counts[target] = terms.len();
            for t in terms {
                edge_counts[remap[t.node as usize] as usize] += 1;
            }
        }
    }
    for root in &outputs {
        if root.node != NONE {
            root_counts[remap[root.node as usize] as usize] += 1;
        }
    }
    let edge_offsets = offsets(&edge_counts);
    let forward_offsets = offsets(&forward_counts);
    let root_offsets = offsets(&root_counts);
    let empty = Term {
        node: 0,
        coefficient: 0,
    };
    let mut edges = vec![empty; *edge_offsets.last().unwrap() as usize];
    let mut forward = vec![empty; edges.len()];
    let mut edge_cursor = edge_offsets.clone();
    for (target, &old) in ordered.iter().enumerate() {
        if let Node::Sum(terms) = &nodes[old] {
            for (j, t) in terms.iter().enumerate() {
                let source = remap[t.node as usize];
                forward[forward_offsets[target] as usize + j] = Term {
                    node: source,
                    coefficient: t.coefficient,
                };
                let cursor = &mut edge_cursor[source as usize];
                edges[*cursor as usize] = Term {
                    node: target as u32,
                    coefficient: t.coefficient,
                };
                *cursor += 1;
            }
        }
    }
    let mut roots = vec![
        Root {
            output: 0,
            coefficient: 0
        };
        *root_offsets.last().unwrap() as usize
    ];
    let mut root_cursor = root_offsets.clone();
    for (i, t) in outputs.iter().enumerate() {
        if t.node == NONE {
            continue;
        }
        let cursor = &mut root_cursor[remap[t.node as usize] as usize];
        roots[*cursor as usize] = Root {
            output: i as u32,
            coefficient: t.coefficient,
        };
        *cursor += 1;
    }
    let mut used = vec![false; coefficients.len()];
    for c in edges
        .iter()
        .map(|e| e.coefficient)
        .chain(roots.iter().map(|r| r.coefficient))
    {
        if c < super::builder::NEG_ONE {
            used[c as usize] = true;
        }
    }
    if let Some(remap) = coefficients.compact(&used) {
        let remap = |c: &mut u32| {
            if *c < super::builder::NEG_ONE {
                *c = u32::try_from(remap[*c as usize]).expect("coefficient index overflow");
            }
        };
        for term in edges.iter_mut().chain(&mut forward) {
            remap(&mut term.coefficient);
        }
        for root in &mut roots {
            remap(&mut root.coefficient);
        }
    }
    let map = |id: u32| if id == NONE { NONE } else { remap[id as usize] };
    WengertTape {
        coefficients,
        graph: CompiledGraph {
            inputs: inputs.into_iter().map(map).collect(),
            output_count: outputs.len(),
            levels: levels.into(),
            max_terms: edge_counts
                .iter()
                .zip(&root_counts)
                .map(|(e, r)| e + r)
                .max()
                .unwrap_or(0),
            edge_offsets: edge_offsets.into(),
            edges: edges.into(),
            root_offsets: root_offsets.into(),
            roots: roots.into(),
            forward_offsets: forward_offsets.into(),
            forward: forward.into(),
            max_power_len: groups.iter().map(|g| g.len as usize).max().unwrap_or(0),
            groups: groups
                .into_iter()
                .map(|g| PowerGroup {
                    full: map(g.full),
                    low: map(g.low),
                    ..g
                })
                .collect(),
        },
    }
}
