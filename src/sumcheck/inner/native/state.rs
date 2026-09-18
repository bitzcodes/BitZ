//! Native table preparation and arithmetic-only state transitions.
use super::*;
use crate::sumcheck::inner::input;

pub enum NativeWeights {
    Dense {
        matrix: Vec<Raw>,
        live: usize,
    },
    Blocks {
        weights: Vec<Raw>,
        scales: BlockScales,
        live: usize,
        num_vars: usize,
    },
}

pub struct State<'a> {
    num_vars: usize,
    structured: bool,
    stage: Stage<'a>,
}
enum Stage<'a> {
    First {
        matrix: Vec<Raw>,
        witness: RawWitness<'a>,
        live: usize,
        coefficients: [Field; 2],
    },
    Integers(Dense<field::Uint<2>>),
    Field(Dense<Field>),
    BlockIntegers(Block<'a, field::Uint<2>>),
    BlockField(Block<'a, Field>),
    Terminal([Field; 2]),
}
struct Dense<W> {
    matrix: Vec<Raw>,
    witness: Vec<W>,
    live: usize,
    matrix_scratch: Vec<Raw>,
    witness_scratch: Vec<W>,
    coefficients: [Field; 2],
}
fn elements(ctx: &FieldConfig, p: [Raw; 2]) -> [Field; 2] {
    p.map(|x| shared_raw(ctx, x))
}
impl<W: FoldedValue> Dense<W> {
    fn new(
        ctx: &FieldConfig,
        matrix: Vec<Raw>,
        witness: Vec<W>,
        live: usize,
        coefficients: [Raw; 2],
    ) -> Self {
        Self {
            matrix,
            witness,
            live,
            // The source stage still owns its original tables while this state
            // is constructed. Allocate scratch only after that stage is dropped.
            matrix_scratch: Vec::new(),
            witness_scratch: Vec::new(),
            coefficients: elements(ctx, coefficients),
        }
    }
    fn fold(&mut self, ctx: &FieldConfig, r: Raw) {
        let n = self.matrix.len() / 2;
        let live = self.live.div_ceil(2);
        self.matrix_scratch.resize(n, 0);
        self.witness_scratch.resize(n, W::zero(ctx));
        if n == 1 {
            self.matrix_scratch[0] = ctx.interpolate(self.matrix[0], self.matrix[1], r);
            self.witness_scratch[0] = W::from_encoding(
                ctx,
                ctx.interpolate(self.witness[0].encoding(), self.witness[1].encoding(), r),
            );
        } else {
            let written = 2 * live.div_ceil(2);
            let c = folded::fold_round::<W, true, _>(
                ctx,
                &self.matrix[..2 * written],
                &self.witness[..2 * written],
                |v| v,
                &mut self.matrix_scratch[..written],
                &mut self.witness_scratch[..written],
                r,
            );
            self.matrix_scratch[written..].fill(0);
            self.witness_scratch[written..].fill(W::zero(ctx));
            self.coefficients = elements(ctx, c);
        }
        core::mem::swap(&mut self.matrix, &mut self.matrix_scratch);
        core::mem::swap(&mut self.witness, &mut self.witness_scratch);
        self.live = live;
    }
    fn terminal(&self, ctx: &FieldConfig) -> [Field; 2] {
        elements(ctx, [self.matrix[0], self.witness[0].final_raw(ctx)])
    }
}

fn terminal_source(ctx: &FieldConfig, witness: &RawWitness<'_>) -> Raw {
    match witness {
        RawWitness::Native { values, .. } => ctx.native_residue(values[0]),
        RawWitness::Field(v) => v[0],
        RawWitness::Wide(v) => raw_shared(field::IntegerEmbedding::from_integer(ctx, &v.read(0))),
        RawWitness::Limbs(v) => raw_shared(field::IntegerEmbedding::from_integer(ctx, &v.read(0))),
    }
}
fn first_coefficients(ctx: &FieldConfig, matrix: &[Raw], witness: &RawWitness<'_>) -> [Raw; 2] {
    match witness {
        RawWitness::Native { values, .. } => {
            inner_coefficients_native_raw(ctx, matrix, &values[..matrix.len()])
        }
        RawWitness::Field(v) => inner_coefficients_field_raw(ctx, matrix, &v[..matrix.len()]),
        RawWitness::Wide(v) if v.words() == 1 => {
            inner_coefficients_native_map(ctx, matrix, |i| v.read_u64(i))
        }
        RawWitness::Wide(v) => native_witness::wide_coefficients(ctx, matrix, |i| v.read(i)),
        RawWitness::Limbs(v) => native_witness::wide_coefficients(ctx, matrix, |i| v.read(i)),
    }
}
fn first_fold<'a>(
    ctx: &FieldConfig,
    matrix: &[Raw],
    witness: &RawWitness<'a>,
    live: usize,
    r: Raw,
) -> Stage<'a> {
    let n = matrix.len() / 2;
    let live = live.div_ceil(2);
    if n == 1 {
        let h = match witness {
            RawWitness::Native { values, .. } => {
                fold_native_pair(ctx, ctx.sub_raw(ctx.one_raw(), r), r, values[0], values[1])
            }
            RawWitness::Field(v) => ctx.interpolate(v[0], v[1], r),
            RawWitness::Wide(v) => raw_shared(ctx.weighted_pair(
                &[
                    shared_raw(ctx, ctx.sub_raw(ctx.one_raw(), r)),
                    shared_raw(ctx, r),
                ],
                &[v.read(0), v.read(1)],
            )),
            RawWitness::Limbs(v) => raw_shared(ctx.weighted_pair(
                &[
                    shared_raw(ctx, ctx.sub_raw(ctx.one_raw(), r)),
                    shared_raw(ctx, r),
                ],
                &[v.read(0), v.read(1)],
            )),
        };
        return Stage::Terminal(elements(ctx, [ctx.interpolate(matrix[0], matrix[1], r), h]));
    }
    let written = 2 * live.div_ceil(2);
    let mut mout = vec![0; n];
    match witness {
        RawWitness::Field(v) => {
            let mut out = vec![shared_raw(ctx, 0); n];
            let c = folded::fold_round::<Field, true, _>(
                ctx,
                &matrix[..2 * written],
                &v[..2 * written],
                |v| shared_raw(ctx, v),
                &mut mout[..written],
                &mut out[..written],
                r,
            );
            Stage::Field(Dense::new(ctx, mout, out, live, c))
        }
        _ => {
            let mut out = vec![field::Uint::<2>::ZERO; n];
            let c = match witness {
                RawWitness::Native { values, .. } => fold_inner_native_raw(
                    ctx,
                    &matrix[..2 * written],
                    &values[..2 * written],
                    &mut mout[..written],
                    &mut out[..written],
                    r,
                ),
                RawWitness::Wide(v) if v.words() == 1 => fold_native_map::<true>(
                    ctx,
                    &matrix[..2 * written],
                    |i| v.read_u64(i),
                    &mut mout[..written],
                    &mut out[..written],
                    r,
                ),
                RawWitness::Wide(v) => native_witness::wide_fold::<4, true>(
                    ctx,
                    &matrix[..2 * written],
                    |i| v.read(i),
                    &mut mout[..written],
                    &mut out[..written],
                    r,
                ),
                RawWitness::Limbs(v) => native_witness::wide_fold::<32, true>(
                    ctx,
                    &matrix[..2 * written],
                    |i| v.read(i),
                    &mut mout[..written],
                    &mut out[..written],
                    r,
                ),
                RawWitness::Field(_) => unreachable!(),
            };
            Stage::Integers(Dense::new(ctx, mout, out, live, c))
        }
    }
}

struct Block<'a, W> {
    witness: RawWitness<'a>,
    scales: BlockScales,
    live: usize,
    weights: Vec<Raw>,
    weights_scratch: Vec<Raw>,
    scaled: Vec<(usize, Raw)>,
    tables: Vec<Vec<W>>,
    scratches: Vec<Vec<W>>,
    lives: Vec<usize>,
    point: Vec<Raw>,
    len: usize,
    coefficients: [Field; 2],
}
fn block_values<'a>(
    witness: &'a RawWitness<'_>,
    block: usize,
    block_len: usize,
) -> BlockValues<'a> {
    let start = block * block_len;
    let end = start + block_len;
    match witness {
        RawWitness::Native {
            values,
            constant_prefix,
            ..
        } => {
            if block == 0 && constant_prefix.is_some_and(|p| p.0 == block_len) {
                BlockValues::ConstantOne
            } else {
                BlockValues::Native(values.get(start..end).unwrap_or(&[]))
            }
        }
        RawWitness::Field(v) => BlockValues::Field(&v[start..end]),
        RawWitness::Wide(v) => {
            assert_eq!(v.block_len(), block_len);
            v.block(block)
        }
        RawWitness::Limbs(v) => {
            assert_eq!(v.len() / 4, block_len);
            v.block(block)
        }
    }
}
fn combine(ctx: &FieldConfig, scaled: &[(usize, Raw)], partials: &[[Raw; 2]]) -> [Field; 2] {
    let mut c = [0; 2];
    for ((_, scale), p) in scaled.iter().zip(partials) {
        for i in 0..2 {
            c[i] = ctx.add_raw(c[i], ctx.mul_raw(*scale, p[i]));
        }
    }
    elements(ctx, c)
}
impl<'a, W: FoldedValue> Block<'a, W> {
    fn new(
        ctx: &FieldConfig,
        weights: Vec<Raw>,
        scales: BlockScales,
        witness: RawWitness<'a>,
        live: usize,
        num_vars: usize,
    ) -> Result<Self, SumcheckError> {
        let block_len = scales.block_len;
        let blocks = scales.scales.len();
        if num_vars >= usize::BITS as usize {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        let domain = 1usize << num_vars;
        if !block_len.is_power_of_two()
            || block_len < 2
            || block_len.checked_mul(blocks) != Some(domain)
            || !blocks.is_power_of_two()
            || witness.len() != domain
            || live > domain
            || scales.rows > block_len
            || weights.len() < scales.rows
        {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        let witness = match witness {
            RawWitness::Native {
                values,
                domain,
                constant_prefix,
            } if values.len() < live || !values.len().is_multiple_of(block_len) => {
                RawWitness::Native {
                    values,
                    domain,
                    constant_prefix,
                }
                .materialize()
            }
            other => other,
        };
        let mut wz = vec![0; block_len];
        wz[..scales.rows].copy_from_slice(&weights[..scales.rows]);
        let live_of = |block: usize| live.saturating_sub(block * block_len).min(block_len);
        let scaled: Vec<_> = (0..blocks)
            .filter_map(|b| scales.scales[b].map(|s| (b, s)))
            .filter(|(b, _)| live_of(*b) > 0)
            .collect();
        let _scope = tracing::info_span!("raw:inner_round0").entered();
        let partials: Vec<_> = scaled
            .iter()
            .map(|&(b, _)| {
                block_values(&witness, b, block_len)
                    .coefficients(ctx, &wz[..2 * live_of(b).div_ceil(2)])
            })
            .collect();
        let coefficients = combine(ctx, &scaled, &partials);
        let lives = scaled.iter().map(|&(b, _)| live_of(b)).collect();
        Ok(Self {
            witness,
            scales,
            live,
            weights: wz,
            weights_scratch: vec![0; block_len / 2],
            scaled,
            tables: Vec::new(),
            scratches: Vec::new(),
            lives,
            point: Vec::with_capacity(num_vars),
            len: block_len,
            coefficients,
        })
    }
    fn fold(&mut self, ctx: &FieldConfig, r: Raw) -> Option<Stage<'a>> {
        let first = self.tables.is_empty();
        let n = self.len / 2;
        self.point.push(r);
        self.weights_scratch.truncate(n);
        fold_table_raw(ctx, &self.weights, &mut self.weights_scratch, r);
        core::mem::swap(&mut self.weights, &mut self.weights_scratch);
        if n == 1 {
            let finals: Vec<_> = self
                .scaled
                .iter()
                .enumerate()
                .map(|(i, &(b, _))| {
                    if first {
                        block_values(&self.witness, b, self.scales.block_len).folded_pair(ctx, r)
                    } else {
                        let t = &self.tables[i];
                        W::from_encoding(ctx, ctx.interpolate(t[0].encoding(), t[1].encoding(), r))
                            .final_raw(ctx)
                    }
                })
                .collect();
            return Some(self.tail(ctx, finals));
        }
        let mut partials = Vec::with_capacity(self.scaled.len());
        for (i, &(b, _)) in self.scaled.iter().enumerate() {
            let live = self.lives[i].div_ceil(2);
            let written = 2 * live.div_ceil(2);
            if first {
                let mut out = vec![W::zero(ctx); n];
                let p = W::fold_initial(
                    ctx,
                    block_values(&self.witness, b, self.scales.block_len),
                    &self.weights[..written],
                    &mut out[..written],
                    r,
                );
                self.tables.push(out);
                self.scratches.push(vec![W::zero(ctx); n / 2]);
                partials.push(p);
            } else {
                let scratch = &mut self.scratches[i];
                scratch.truncate(n);
                let p = folded::fold_round::<W, false, _>(
                    ctx,
                    &self.weights[..written],
                    &self.tables[i][..2 * written],
                    |v| v,
                    &mut [],
                    &mut scratch[..written],
                    r,
                );
                scratch[written..].fill(W::zero(ctx));
                core::mem::swap(&mut self.tables[i], scratch);
                partials.push(p);
            }
            self.lives[i] = live;
        }
        self.coefficients = combine(ctx, &self.scaled, &partials);
        self.len = n;
        None
    }
    fn tail(&self, ctx: &FieldConfig, finals: Vec<Raw>) -> Stage<'a> {
        let blocks = self.scales.scales.len();
        let weight = self.weights[0];
        let eq_zero = self.point.iter().fold(ctx.one_raw(), |p, &r| {
            ctx.mul_raw(p, ctx.sub_raw(ctx.one_raw(), r))
        });
        let mut eq = None;
        let mut matrix = vec![0; blocks];
        let mut witness = vec![0; blocks];
        for (i, &(b, s)) in self.scaled.iter().enumerate() {
            matrix[b] = ctx.mul_raw(s, weight);
            witness[b] = finals[i];
        }
        for (b, v) in witness.iter_mut().enumerate() {
            if self
                .live
                .saturating_sub(b * self.scales.block_len)
                .min(self.scales.block_len)
                == 0
                || self.scales.scales[b].is_some()
            {
                continue;
            }
            let values = block_values(&self.witness, b, self.scales.block_len);
            *v = match sparse_block_value_raw(eq_zero, values) {
                Some(v) => v,
                None => weighted_block_sum_raw(
                    ctx,
                    eq.get_or_insert_with(|| eq_table_raw(ctx, &self.point)),
                    values,
                ),
            };
        }
        if blocks == 1 {
            return Stage::Terminal(elements(ctx, [matrix[0], witness[0]]));
        }
        let c = inner_coefficients_field_raw(ctx, &matrix, &witness);
        Stage::Field(Dense::new(
            ctx,
            matrix,
            witness.into_iter().map(|v| shared_raw(ctx, v)).collect(),
            blocks,
            c,
        ))
    }
}

impl input::sealed::Input for RawWitness<'_> {}
impl<'a> input::Input<FieldConfig> for RawWitness<'a> {
    type Weights = NativeWeights;
    type State = State<'a>;
    type Codec = input::Canonical;
    fn prepare(
        self,
        ctx: &FieldConfig,
        weights: NativeWeights,
    ) -> Result<State<'a>, SumcheckError> {
        let (num_vars, structured, stage) = match weights {
            NativeWeights::Dense { matrix, live } => {
                let n = matrix.len();
                if !n.is_power_of_two()
                    || self.len() != n
                    || live > n
                    || matches!(&self,RawWitness::Native{values,domain,..} if values.len()>*domain)
                {
                    return Err(SumcheckError::InvalidProductDimensions);
                }
                let witness = self.materialize();
                let num_vars = n.ilog2() as usize;
                let _scope = tracing::info_span!("raw:inner_round0").entered();
                let c = if n == 1 {
                    [ctx.zero(); 2]
                } else {
                    elements(
                        ctx,
                        first_coefficients(ctx, &matrix[..2 * live.div_ceil(2)], &witness),
                    )
                };
                (
                    num_vars,
                    false,
                    Stage::First {
                        matrix,
                        witness,
                        live,
                        coefficients: c,
                    },
                )
            }
            NativeWeights::Blocks {
                weights,
                scales,
                live,
                num_vars,
            } => {
                let stage = match self {
                    RawWitness::Field(_) => {
                        Stage::BlockField(Block::new(ctx, weights, scales, self, live, num_vars)?)
                    }
                    _ => Stage::BlockIntegers(Block::new(
                        ctx, weights, scales, self, live, num_vars,
                    )?),
                };
                (num_vars, true, stage)
            }
        };
        Ok(State {
            num_vars,
            structured,
            stage,
        })
    }
}
impl input::State<FieldConfig> for State<'_> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
    fn coefficients(&self, ctx: &FieldConfig) -> Result<[Field; 2], SumcheckError> {
        Ok(match &self.stage {
            Stage::First { coefficients, .. } => *coefficients,
            Stage::Integers(s) => s.coefficients,
            Stage::Field(s) => s.coefficients,
            Stage::BlockIntegers(s) => s.coefficients,
            Stage::BlockField(s) => s.coefficients,
            Stage::Terminal(_) => [ctx.zero(); 2],
        })
    }
    fn fold(&mut self, ctx: &FieldConfig, r: &Field) -> Result<(), SumcheckError> {
        let span = if self.structured {
            tracing::info_span!("raw:inner_block_rounds")
        } else if matches!(&self.stage, Stage::First { .. }) {
            tracing::info_span!("raw:inner_fold0")
        } else {
            tracing::info_span!("raw:inner_rounds")
        };
        let _scope = span.entered();
        let r = ctx.raw(r);
        let next = match &mut self.stage {
            Stage::First {
                matrix,
                witness,
                live,
                ..
            } => Some(first_fold(ctx, matrix, witness, *live, r)),
            Stage::Integers(s) => {
                s.fold(ctx, r);
                None
            }
            Stage::Field(s) => {
                s.fold(ctx, r);
                None
            }
            Stage::BlockIntegers(s) => s.fold(ctx, r),
            Stage::BlockField(s) => s.fold(ctx, r),
            Stage::Terminal(_) => return Err(SumcheckError::InvalidProductDimensions),
        };
        if let Some(next) = next {
            self.stage = next;
        }
        Ok(())
    }
    fn terminal(&self, ctx: &FieldConfig) -> Result<[Field; 2], SumcheckError> {
        Ok(match &self.stage {
            Stage::First {
                matrix, witness, ..
            } => elements(ctx, [matrix[0], terminal_source(ctx, witness)]),
            Stage::Integers(s) => s.terminal(ctx),
            Stage::Field(s) => s.terminal(ctx),
            Stage::Terminal(v) => *v,
            _ => return Err(SumcheckError::InvalidProductDimensions),
        })
    }
}
