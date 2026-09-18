//! Binary-map adjoints shared with virtual openings. No transcript, packing,
//! dual-basis, or Flock types enter this layer. Equality construction is supplied
//! by the polynomial layer so coordinate conventions have a single owner.
use field::{Gf128, PreparedGf128Mul};
const LOG_PACKING: usize = 7;

/// The η-combined per-derived-cell transpose coefficients
/// `E_r = Σ_l η_l·eq_{bits(r)}(pt_l)`, factored over the (row, column)
/// split of `r`'s flat index with the η's folded into the column tables
/// once — one product per (cell, chunk), exactly the old `mqv:wcoef`
/// formula, now evaluated on demand instead of materialized.
pub struct BinaryRowWeights {
    pub eq_rs: Vec<Vec<Gf128>>,
    pub scaled_zc: Vec<Vec<Gf128>>,
    pub t_wh: usize,
    pub h_mask: usize,
}

impl BinaryRowWeights {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(
        points: &[Vec<Gf128>],
        etas: &[Gf128],
        t_wh: usize,
        build_eq_x_r_vec: impl Fn(&[Gf128], &()) -> Result<Vec<Gf128>, ()>,
    ) -> Self {
        let eq_rs: Vec<Vec<Gf128>> = points
            .iter()
            .map(|pt| build_eq_x_r_vec(&pt[..t_wh], &()).expect("t_wh >= 1"))
            .collect();
        let scaled_zc: Vec<Vec<Gf128>> = points
            .iter()
            .zip(etas.iter())
            .map(|(pt, &eta)| {
                if pt.len() == t_wh {
                    vec![eta]
                } else {
                    build_eq_x_r_vec(&pt[t_wh..], &())
                        .expect("nonempty assignment-column point")
                        .into_iter()
                        .map(|x| eta * x)
                        .collect()
                }
            })
            .collect();
        Self {
            eq_rs,
            scaled_zc,
            t_wh,
            h_mask: (1usize << t_wh) - 1,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub fn coeff(&self, r: usize) -> Gf128 {
        let (c, b) = (r >> self.t_wh, r & self.h_mask);
        // Polynomial reduction is linear over F2. XOR the degree-at-most-254
        // products first; their sum fits the same width for any chunk count.
        let mut acc = field::Gf128Product::zero();
        for (l, zc) in self.scaled_zc.iter().enumerate() {
            acc ^= self.eq_rs[l][b].mul_unreduced(zc[c]);
        }
        acc.reduce()
    }
}

/// The batching message obtained by streaming CSC source columns and merging
/// fixed-size partial accumulators by field addition. Neither the transposed
/// weight vector nor a coefficient table is materialized.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
pub fn virtual_column_weight<M>(map: &M, column: usize, coeffs: &BinaryRowWeights) -> Gf128
where
    M: super::binary::VirtualMap,
{
    map.column_rows(column)
        .expect("source column in bounds")
        .fold(Gf128::zero(), |acc, row| acc + coeffs.coeff(row))
}

/// Per-prove column-weight engine for the batching passes: fills whole
/// 128-column source packs with `W_j = Σ_{r:M[r,j]=1} E_r`
/// (`E_r = Σ_l η_l·eq_{bits(r)}(pt_l)`).
///
/// For a power-of-two tensor repetition (`global = local·2^k + instance`,
/// [`super::binary::VirtualMap::repetition`]) the eq tensor factors over
/// the instance/local bit split, so
///
/// ```text
/// W_{(lc, inst)} = Σ_l eq_inst_l[inst] · S_{l,lc},
/// S_{l,lc}       = η_l · Σ_{lr ∈ localcol(lc)} eq_loc_l[lr],
/// ```
///
/// with the `S` tables precomputed once in `O(L·(local_rows + nnz_local))`
/// field ops. One weight then costs `L` multiplies instead of the
/// streamed fold's `L·deg`, and a local column with all `S_{·,lc} = 0`
/// zeroes its whole pack in one check. FieldRepresentation associativity and
/// distributivity make every value BIT-IDENTICAL to the streamed
/// per-nonzero fold — prover and verifier transcripts are unchanged
/// (pinned by `virtual_pack_weights_match_generic`).
/// One cross-instance or boundary term of a chained packed-source
/// repetition ([`super::binary::ChainedPackedSourceMap`]): source cell
/// `(inst, lc)` with `inst ∈ [inst_lo, inst_hi)` gains
/// `Σ_l eq_l[inst − inst_lo] · s_l[lc]`. The chain link (`prev`) carries the
/// instance tables ROTATED by one (instance `i − 1`'s cells feed instance
/// `i`'s rows, so source instance `j` takes `eq[j + 1]`); the boundary maps
/// (`first`/`last`) carry the single entry `eq[0]` / `eq[N − 1]`.
pub struct ExtraWeightTerm {
    /// Active source instances `[inst_lo, inst_hi)`.
    pub inst_lo: usize,
    pub inst_hi: usize,
    /// Per chunk: the instance factors of the active instances, in order.
    pub eq_inst: Vec<Vec<PreparedGf128Mul>>,
    /// Per chunk: the eta-scaled local-row equality sums of this term's
    /// local map (index 0 = the constant column, `1..=w` the cells).
    pub s: Vec<Vec<Gf128>>,
    /// Nonconstant local offsets `[col_lo, col_hi)` (`0..w`) that carry
    /// any entry; runs outside are skipped.
    pub col_lo: usize,
    pub col_hi: usize,
}

impl ExtraWeightTerm {
    /// Adds this term's weights of the run `[local_offset, local_offset +
    /// targets.len())` of instance `instance`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn accumulate(&self, instance: usize, local_offset: usize, targets: &mut [Gf128]) {
        if instance < self.inst_lo || instance >= self.inst_hi {
            return;
        }
        let lo = local_offset.max(self.col_lo);
        let hi = (local_offset + targets.len()).min(self.col_hi);
        if lo >= hi {
            return;
        }
        let slot = instance - self.inst_lo;
        for (eq_l, s_l) in self.eq_inst.iter().zip(self.s.iter()) {
            let fixed = &eq_l[slot];
            let source = &s_l[1 + lo..1 + hi];
            for (target, &value) in targets[lo - local_offset..hi - local_offset]
                .iter_mut()
                .zip(source)
            {
                *target += fixed.mul(&value);
            }
        }
    }

    /// Whether the term touches any nonconstant cell.
    const fn is_empty(&self) -> bool {
        self.col_lo >= self.col_hi || self.inst_lo >= self.inst_hi
    }
}

/// Verifier-side factored form of an identity compact tail
/// ([`super::binary::ChainedSourceTail`] whose local map is the identity):
/// source column `j ∈ [source_start, source_start + len)` carries the
/// derived-row weight `E_{row_start + (j − source_start)}` — one
/// [`BinaryRowWeights::coeff`] per cell, evaluated on demand by
/// [`BinaryAdjoint::pack_weights`] and consumed by the affine-tail plane
/// engine (the opening plane engine) without ever being materialized.
pub struct AffineTailWeights {
    pub source_start: usize,
    pub len: usize,
    pub row_start: usize,
    pub coeffs: BinaryRowWeights,
}

impl AffineTailWeights {
    pub fn end(&self) -> usize {
        self.source_start + self.len
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn add_pack(&self, pack: usize, out: &mut [Gf128; 128]) {
        let base = pack << LOG_PACKING;
        let lo = base.max(self.source_start);
        let hi = (base + 128).min(self.end());
        for column in lo..hi {
            out[column - base] += self
                .coeffs
                .coeff(self.row_start + (column - self.source_start));
        }
    }
}

pub struct DenseWeightCorrection {
    pub start: usize,
    pub weights: Vec<Gf128>,
}

impl DenseWeightCorrection {
    pub fn end(&self) -> usize {
        self.start + self.weights.len()
    }
    pub fn add_pack(&self, pack: usize, out: &mut [Gf128; 128]) {
        let base = pack << LOG_PACKING;
        let lo = base.max(self.start);
        let hi = (base + 128).min(self.end());
        if lo < hi {
            for (dst, src) in out[lo - base..hi - base]
                .iter_mut()
                .zip(&self.weights[lo - self.start..hi - self.start])
            {
                *dst += *src;
            }
        }
    }
}

pub enum BinaryAdjoint<'a, M: super::binary::VirtualMap> {
    /// Factored tensor-repetition tables.
    Repeated {
        columns: usize,
        /// `k`: the instance coordinates are the low `k` bits.
        instance_bits: usize,
        /// Per chunk `l`: eq table over `pt_l[..k]` (`2^k` entries).
        eq_inst: Vec<Vec<Gf128>>,
        /// Per chunk `l`: `η_l`-scaled local-column sums of the eq table
        /// over `pt_l[k..]`.
        s: Vec<Vec<Gf128>>,
        _map: core::marker::PhantomData<&'a M>,
    },
    /// Factored tensor tables for an instance-major packed source with one
    /// shared constant column. Unlike `Repeated`, source packs normally stay
    /// within one instance and walk consecutive local columns.
    PackedSourceRepeated {
        columns: usize,
        /// Number of nonconstant source cells in one local instance.
        local_width: usize,
        /// End of the live source prefix; the remaining source domain is zero.
        live_cols: usize,
        /// Per chunk and instance: a preprocessed multiplier for the instance
        /// equality weight, reused across each contiguous local-column run.
        eq_inst: Vec<Vec<PreparedGf128Mul>>,
        /// The same instance equality tables as plain field elements (the
        /// plane engine's instance factors).
        eq_inst_gf: Vec<Vec<Gf128>>,
        /// Number of instances (`2^k`).
        instances: usize,
        /// Per chunk and local column: the eta-scaled local-row equality sum.
        s: Vec<Vec<Gf128>>,
        /// Weight of the one source constant shared by every instance
        /// (including every boundary term's constant-column part).
        constant_weight: Gf128,
        /// Cross-instance and boundary terms of a chained repetition
        /// (empty for a plain repetition). They are folded into
        /// [`Self::pack_weights`]; the plane engine covers only the plain
        /// part, so the prover adds them through the opening plane adapter.
        extra: Vec<ExtraWeightTerm>,
        /// Appended compact relation and its source aliases, including column zero.
        corrections: Vec<DenseWeightCorrection>,
        /// Verifier-side: an identity compact tail kept factored instead of
        /// folded into `corrections` (see the factored-tail constructor).
        affine_tail: Option<AffineTailWeights>,
        _map: core::marker::PhantomData<&'a M>,
    },
    /// The streamed per-nonzero fold (any map).
    Generic {
        map: &'a M,
        coeffs: BinaryRowWeights,
    },
}

/// Splits a derived-domain point of a packed-source repetition into its
/// `(instance, local)` coordinate slices for the given derived order.
fn packed_source_point_split<'p>(
    order: super::binary::PackedSourceOrder,
    instance_bits: usize,
    local_bits: usize,
    point: &'p [Gf128],
) -> (&'p [Gf128], &'p [Gf128]) {
    match order {
        super::binary::PackedSourceOrder::LocalMajor => {
            (&point[..instance_bits], &point[instance_bits..])
        }
        super::binary::PackedSourceOrder::InstanceMajor => {
            (&point[local_bits..], &point[..local_bits])
        }
    }
}

impl<'a, M: super::binary::VirtualMap> BinaryAdjoint<'a, M> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new_with_tail(
        map: &'a M,
        points: &[Vec<Gf128>],
        etas: &[Gf128],
        t_wh: usize,
        factored_tail: bool,
        build_eq_x_r_vec: impl Fn(&[Gf128], &()) -> Result<Vec<Gf128>, ()>,
        parallel: bool,
    ) -> Self {
        if let Some(parts) = map.chained_packed_source()
            && parts.instances.is_power_of_two()
            && parts.instances > 1
            && parts.local.cols() > 1
        {
            let _g = tracing::info_span!("mqv:w_chained").entered();
            let instances = parts.instances;
            let k = instances.trailing_zeros() as usize;
            let local_width = parts.local.cols() - 1;
            let point_fits =
                |pt: &Vec<Gf128>| k < pt.len() && (1usize << (pt.len() - k)) >= parts.local.rows();
            if let Some(live_cols) = local_width
                .checked_mul(instances)
                .and_then(|width| width.checked_add(1))
                .filter(|&width| width <= map.cols())
                && points.iter().all(point_fits)
            {
                debug_assert!(parts.local.rows() * instances <= map.rows());
                let eq_inst_gf: Vec<Vec<Gf128>> = points
                    .iter()
                    .map(|pt| build_eq_x_r_vec(&pt[..k], &()).expect("k >= 1"))
                    .collect();
                let eq_loc: Vec<Vec<Gf128>> = points
                    .iter()
                    .map(|pt| build_eq_x_r_vec(&pt[k..], &()).expect("local coords non-empty"))
                    .collect();
                let scaled_sums = |local: &super::binary::PreparedVirtualMap| -> Vec<Vec<Gf128>> {
                    eq_loc
                        .iter()
                        .zip(etas.iter())
                        .map(|(eq_loc_l, &eta)| {
                            local
                                .matrix()
                                .columns()
                                .map(|column| {
                                    let sum = column
                                        .indices()
                                        .iter()
                                        .fold(Gf128::zero(), |acc, &lr| acc + eq_loc_l[lr]);
                                    eta * sum
                                })
                                .collect()
                        })
                        .collect()
                };
                let s = scaled_sums(parts.local);
                let eq_inst: Vec<Vec<PreparedGf128Mul>> = eq_inst_gf
                    .iter()
                    .map(|table| {
                        table
                            .iter()
                            .copied()
                            .map(|value| PreparedGf128Mul::new(value.into()))
                            .collect()
                    })
                    .collect();
                // Source instance `j` of the chain link feeds instance
                // `j + 1`'s rows: rotate the instance tables by one.
                let term = |inst_lo: usize,
                            inst_hi: usize,
                            shift: usize,
                            local: &super::binary::PreparedVirtualMap| {
                    let (col_lo, col_hi) =
                        super::binary::ChainedPackedSourceMap::nonconstant_column_span(local);
                    ExtraWeightTerm {
                        inst_lo,
                        inst_hi,
                        eq_inst: eq_inst_gf
                            .iter()
                            .map(|table| {
                                (inst_lo..inst_hi)
                                    .map(|inst| PreparedGf128Mul::new(table[inst + shift].into()))
                                    .collect()
                            })
                            .collect(),
                        s: scaled_sums(local),
                        col_lo,
                        col_hi,
                    }
                };
                let terms = [
                    term(0, instances - 1, 1, parts.prev),
                    term(0, 1, 0, parts.first),
                    term(instances - 1, instances, 0, parts.last),
                ];
                // Column zero: the plain part sums each chunk's instance
                // table to one; every boundary term adds its own
                // instance-weighted constant-column sum.
                let mut constant_weight = s.iter().fold(Gf128::zero(), |acc, s_l| acc + s_l[0]);
                for term in &terms {
                    for (eq_l, s_l) in term.eq_inst.iter().zip(term.s.iter()) {
                        if s_l[0] == Gf128::zero() {
                            continue;
                        }
                        for fixed in eq_l {
                            constant_weight += fixed.mul(&s_l[0]);
                        }
                    }
                }
                let extra = terms.into_iter().filter(|term| !term.is_empty()).collect();
                let mut corrections = Vec::new();
                let mut affine_tail = None;
                if let Some(tail) = map.chained_packed_source_tail() {
                    let _g = tracing::info_span!("mqv:w_tail").entered();
                    let coeffs = BinaryRowWeights::new(points, etas, t_wh, &build_eq_x_r_vec);
                    let matrix = tail.map.matrix();
                    let aliases_len = tail.aliases.len();
                    assert!(aliases_len <= matrix.columns().len());
                    let factored = factored_tail && tail.map.is_identity();
                    let column_weight = |column| {
                        if tail.map.is_identity() {
                            return coeffs.coeff(tail.row_offset + column);
                        }
                        matrix.column(column).map_or(Gf128::zero(), |col| {
                            col.indices().iter().fold(Gf128::zero(), |sum, &r| {
                                sum + coeffs.coeff(tail.row_offset + r)
                            })
                        })
                    };
                    let mut aliases = std::collections::BTreeMap::<usize, Gf128>::new();
                    for (tail_column, &source_column) in tail.aliases.iter().enumerate() {
                        *aliases.entry(source_column).or_insert(Gf128::zero()) +=
                            column_weight(tail_column);
                    }
                    for (column, weight) in aliases {
                        if let Some(last) = corrections
                            .last_mut()
                            .filter(|last: &&mut DenseWeightCorrection| column <= last.end() + 128)
                        {
                            last.weights.resize(column - last.start + 1, Gf128::zero());
                            last.weights[column - last.start] += weight;
                        } else {
                            corrections.push(DenseWeightCorrection {
                                start: column,
                                weights: vec![weight],
                            });
                        }
                    }
                    if factored {
                        affine_tail = Some(AffineTailWeights {
                            source_start: tail.source_offset,
                            len: matrix.columns().len() - aliases_len,
                            row_start: tail.row_offset + aliases_len,
                            coeffs,
                        });
                    } else {
                        // Collect the dense suffix directly into its final buffer.
                        // P-256 has 1.2M tail columns; copying this table would add
                        // a second large allocation and a serial memory pass.
                        corrections.push(DenseWeightCorrection {
                            start: tail.source_offset,
                            weights: {
                                let mut values = vec![
                                    Gf128::zero();
                                    matrix.columns().len() - tail.aliases.len()
                                ];
                                super::contraction::columns_into(&mut values, parallel, |i| {
                                    column_weight(i + tail.aliases.len())
                                });
                                values
                            },
                        });
                    }
                }
                return Self::PackedSourceRepeated {
                    columns: map.cols(),
                    local_width,
                    live_cols,
                    eq_inst,
                    eq_inst_gf,
                    instances,
                    s,
                    constant_weight,
                    extra,
                    corrections,
                    affine_tail,
                    _map: core::marker::PhantomData,
                };
            }
        }
        if let Some((local, instances)) = map.packed_source_repetition()
            && instances.is_power_of_two()
            && instances > 1
            && local.cols() > 1
        {
            let k = instances.trailing_zeros() as usize;
            let local_width = local.cols() - 1;
            // Where the instance and local coordinates sit in a derived
            // point: local-major keeps the instance in the low `k` bits,
            // instance-major keeps the local row in the low
            // `log₂ local_stride` bits (the rest of the point is exactly the
            // instance). The factorization below is otherwise identical.
            let order = map.packed_source_order();
            let local_bits = local.rows().next_power_of_two().trailing_zeros() as usize;
            let point_fits = |pt: &Vec<Gf128>| match order {
                super::binary::PackedSourceOrder::LocalMajor => {
                    k < pt.len() && (1usize << (pt.len() - k)) >= local.rows()
                }
                super::binary::PackedSourceOrder::InstanceMajor => pt.len() == local_bits + k,
            };
            if let Some(live_cols) = local_width
                .checked_mul(instances)
                .and_then(|width| width.checked_add(1))
                .filter(|&width| width <= map.cols())
                && points.iter().all(point_fits)
            {
                // Padded global columns are not `(instance, local-column)`
                // coordinates. Keep the structural boundary with the
                // factorized representation so those packs evaluate to zero.
                debug_assert!(local.rows() * instances <= map.rows());
                let eq_inst_gf: Vec<Vec<Gf128>> = points
                    .iter()
                    .map(|pt| {
                        build_eq_x_r_vec(packed_source_point_split(order, k, local_bits, pt).0, &())
                            .expect("k >= 1")
                    })
                    .collect();
                let eq_inst: Vec<Vec<PreparedGf128Mul>> = eq_inst_gf
                    .iter()
                    .map(|table| {
                        table
                            .iter()
                            .copied()
                            .map(|value| PreparedGf128Mul::new(value.into()))
                            .collect()
                    })
                    .collect();
                let s: Vec<Vec<Gf128>> = points
                    .iter()
                    .zip(etas.iter())
                    .map(|(pt, &eta)| {
                        let eq_loc = build_eq_x_r_vec(
                            packed_source_point_split(order, k, local_bits, pt).1,
                            &(),
                        )
                        .expect("local coords non-empty");
                        local
                            .matrix()
                            .columns()
                            .map(|column| {
                                let sum = column
                                    .indices()
                                    .iter()
                                    .fold(Gf128::zero(), |acc, &lr| acc + eq_loc[lr]);
                                eta * sum
                            })
                            .collect()
                    })
                    .collect();
                // Column zero is shared by every repetition. The instance
                // equality table sums to one, so its factored weight is just
                // the sum of the local constant-column terms.
                let constant_weight = s.iter().fold(Gf128::zero(), |acc, s_l| acc + s_l[0]);
                return Self::PackedSourceRepeated {
                    columns: map.cols(),
                    local_width,
                    live_cols,
                    eq_inst,
                    eq_inst_gf,
                    instances,
                    s,
                    constant_weight,
                    extra: Vec::new(),
                    corrections: Vec::new(),
                    affine_tail: None,
                    _map: core::marker::PhantomData,
                };
            }
        }
        if let Some((local, instances)) = map.repetition()
            && instances.is_power_of_two()
            && instances > 1
        {
            let k = instances.trailing_zeros() as usize;
            if points.iter().all(|pt| k < pt.len()) {
                debug_assert_eq!(local.rows() << k, map.rows());
                debug_assert_eq!(local.cols() << k, map.cols());
                let eq_inst: Vec<Vec<Gf128>> = points
                    .iter()
                    .map(|pt| build_eq_x_r_vec(&pt[..k], &()).expect("k >= 1"))
                    .collect();
                let s: Vec<Vec<Gf128>> = points
                    .iter()
                    .zip(etas.iter())
                    .map(|(pt, &eta)| {
                        let eq_loc =
                            build_eq_x_r_vec(&pt[k..], &()).expect("local coords non-empty");
                        local
                            .matrix()
                            .columns()
                            .map(|column| {
                                let sum = column
                                    .indices()
                                    .iter()
                                    .fold(Gf128::zero(), |acc, &lr| acc + eq_loc[lr]);
                                eta * sum
                            })
                            .collect()
                    })
                    .collect();
                return Self::Repeated {
                    columns: map.cols(),
                    instance_bits: k,
                    eq_inst,
                    s,
                    _map: core::marker::PhantomData,
                };
            }
        }
        Self::Generic {
            map,
            coeffs: BinaryRowWeights::new(points, etas, t_wh, &build_eq_x_r_vec),
        }
    }

    pub fn column_count(&self) -> usize {
        match self {
            Self::Repeated { columns, .. } | Self::PackedSourceRepeated { columns, .. } => *columns,
            Self::Generic { map, .. } => map.cols(),
        }
    }

    /// Stream any source-column range, preserving optimized aligned packs.
    /// An unaligned boundary uses at most 254 scalar evaluations. No weight
    /// table or per-range allocation is required, including for factored tails.
    pub fn fill_range(
        &self,
        first: usize,
        out: &mut [Gf128],
    ) -> Result<bool, super::LinearMapError> {
        let end = first
            .checked_add(out.len())
            .ok_or(super::LinearMapError::Length {
                kind: "column range",
                expected: self.column_count(),
                actual: usize::MAX,
            })?;
        if end > self.column_count() {
            return Err(super::LinearMapError::Length {
                kind: "column range",
                expected: self.column_count(),
                actual: end,
            });
        }
        let mut live = false;
        let mut offset = 0;
        while offset < out.len() {
            let column = first + offset;
            if column % 128 == 0 && out.len() - offset >= 128 {
                live |= self.pack_weights(
                    column / 128,
                    (&mut out[offset..offset + 128]).try_into().unwrap(),
                );
                offset += 128;
            } else {
                let value = self.scalar_weight(column);
                out[offset] = value;
                live |= value != Gf128::zero();
                offset += 1;
            }
        }
        Ok(live)
    }
    fn scalar_weight(&self, column: usize) -> Gf128 {
        match self {
            Self::Generic { map, coeffs } => virtual_column_weight(*map, column, coeffs),
            Self::Repeated {
                instance_bits,
                eq_inst,
                s,
                ..
            } => {
                let mask = (1usize << instance_bits) - 1;
                eq_inst.iter().zip(s).fold(Gf128::zero(), |sum, (e, s)| {
                    sum + e[column & mask] * s[column >> instance_bits]
                })
            }
            Self::PackedSourceRepeated {
                local_width,
                live_cols,
                eq_inst,
                s,
                constant_weight,
                extra,
                corrections,
                affine_tail,
                ..
            } => {
                let mut value = Gf128::zero();
                if column == 0 {
                    value = *constant_weight;
                } else if column < *live_cols {
                    let instance = (column - 1) / local_width;
                    let local = (column - 1) % local_width;
                    for (e, s) in eq_inst.iter().zip(s) {
                        value += e[instance].mul(&s[local + 1]);
                    }
                    for term in extra {
                        term.accumulate(instance, local, std::slice::from_mut(&mut value));
                    }
                }
                for correction in corrections {
                    if (correction.start..correction.end()).contains(&column) {
                        value += correction.weights[column - correction.start];
                    }
                }
                if let Some(tail) = affine_tail {
                    if (tail.source_start..tail.end()).contains(&column) {
                        value += tail
                            .coeffs
                            .coeff(tail.row_start + column - tail.source_start);
                    }
                }
                value
            }
        }
    }

    /// Fills the 128 weights of source pack `pack`. Returns `false` when
    /// the pack is structurally all-zero and may be skipped — exact: a
    /// zero weight contributes nothing to either batching pass.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub fn pack_weights(&self, pack: usize, out: &mut [Gf128; 128]) -> bool {
        let base = pack << LOG_PACKING;
        match self {
            Self::Repeated {
                instance_bits: k,
                eq_inst,
                s,
                ..
            } => {
                if *k >= LOG_PACKING {
                    // All 128 columns share one local column: hoist its
                    // scaled sums (pass-fixed multipliers).
                    let lc = base >> k;
                    let inst0 = base & ((1usize << k) - 1);
                    out.fill(Gf128::zero());
                    let mut live = false;
                    for (eq_inst_l, s_l) in eq_inst.iter().zip(s.iter()) {
                        let s_lc = s_l[lc];
                        if s_lc == Gf128::zero() {
                            continue;
                        }
                        live = true;
                        let fixed = PreparedGf128Mul::new(s_lc.into());
                        for (target, &weight) in
                            out.iter_mut().zip(eq_inst_l[inst0..inst0 + 128].iter())
                        {
                            *target += fixed.mul(&weight);
                        }
                    }
                    live
                } else {
                    let mask = (1usize << k) - 1;
                    let mut live = false;
                    for (slot, target) in out.iter_mut().enumerate() {
                        let column = base | slot;
                        let (lc, inst) = (column >> k, column & mask);
                        let mut acc = Gf128::zero();
                        for (eq_inst_l, s_l) in eq_inst.iter().zip(s.iter()) {
                            acc += eq_inst_l[inst] * s_l[lc];
                        }
                        live |= acc != Gf128::zero();
                        *target = acc;
                    }
                    live
                }
            }
            Self::PackedSourceRepeated {
                local_width,
                live_cols,
                eq_inst,
                s,
                constant_weight,
                extra,
                corrections,
                affine_tail,
                ..
            } => {
                out.fill(Gf128::zero());
                let end = (base + 128).min(*live_cols);
                let mut column = base;
                if column == 0 {
                    out[0] = *constant_weight;
                    column = 1;
                }

                // Nonconstant columns are instance-major. Split the pack only
                // at instance boundaries so division and fixed-multiplier
                // selection happen once per run, not once per source cell.
                while column < end {
                    let offset = column - 1;
                    let instance = offset / *local_width;
                    let local_offset = offset % *local_width;
                    let run_len = (end - column).min(*local_width - local_offset);
                    let targets = &mut out[column - base..column - base + run_len];
                    for (eq_inst_l, s_l) in eq_inst.iter().zip(s.iter()) {
                        let fixed = &eq_inst_l[instance];
                        let source = &s_l[1 + local_offset..1 + local_offset + run_len];
                        for (target, &value) in targets.iter_mut().zip(source) {
                            *target += fixed.mul(&value);
                        }
                    }
                    for term in extra {
                        term.accumulate(instance, local_offset, targets);
                    }
                    column += run_len;
                }
                for correction in corrections {
                    correction.add_pack(pack, out);
                }
                if let Some(tail) = affine_tail {
                    tail.add_pack(pack, out);
                }
                out.iter().any(|weight| *weight != Gf128::zero())
            }
            Self::Generic { map, coeffs } => {
                let mut live = false;
                for (slot, target) in out.iter_mut().enumerate() {
                    let weight = virtual_column_weight(*map, base | slot, coeffs);
                    live |= weight != Gf128::zero();
                    *target = weight;
                }
                live
            }
        }
    }

    /// Source packs touched by chained nonconstant terms or compact-tail
    /// corrections. Tail corrections can include the shared constant column;
    /// the SHA constant contribution is already in `constant_weight`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn extra_packs(&self) -> Vec<usize> {
        let Self::PackedSourceRepeated {
            local_width,
            extra,
            corrections,
            ..
        } = self
        else {
            return Vec::new();
        };
        let mut packs = Vec::new();
        for term in extra {
            if term.is_empty() {
                continue;
            }
            for instance in term.inst_lo..term.inst_hi {
                let first = 1 + instance * local_width + term.col_lo;
                let last = 1 + instance * local_width + term.col_hi - 1;
                packs.extend((first >> LOG_PACKING)..=(last >> LOG_PACKING));
            }
        }
        for correction in corrections {
            if !correction.weights.is_empty() {
                packs.extend(
                    (correction.start >> LOG_PACKING)..=((correction.end() - 1) >> LOG_PACKING),
                );
            }
        }
        packs.sort_unstable();
        packs.dedup();
        packs
    }

    /// The chained terms and compact tail's share of this pack's weights.
    /// Returns `false` when it is all zero.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn pack_weights_extra(&self, pack: usize, out: &mut [Gf128; 128]) -> bool {
        out.fill(Gf128::zero());
        let Self::PackedSourceRepeated {
            local_width,
            live_cols,
            extra,
            corrections,
            ..
        } = self
        else {
            return false;
        };
        let base = pack << LOG_PACKING;
        let end = (base + 128).min(*live_cols);
        let mut column = base.max(1);
        while column < end {
            let offset = column - 1;
            let instance = offset / *local_width;
            let local_offset = offset % *local_width;
            let run_len = (end - column).min(*local_width - local_offset);
            let targets = &mut out[column - base..column - base + run_len];
            for term in extra {
                term.accumulate(instance, local_offset, targets);
            }
            column += run_len;
        }
        for correction in corrections {
            correction.add_pack(pack, out);
        }
        out.iter().any(|weight| *weight != Gf128::zero())
    }
}
