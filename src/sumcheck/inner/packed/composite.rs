//! Aligned repeated coefficients followed by a compact unrelated circuit tail.
use super::*;
#[cfg(feature = "ecdsa")]
mod compact;
#[cfg(feature = "ecdsa")]
pub(crate) use compact::CompactCompositeMle;
use field::RingOps;

use crate::poly::mle::CompositeMultilinearExtension;

impl InnerSumcheckMleSource for CompositeMultilinearExtension<'_, Field> {
    fn declared_num_vars(&self) -> Option<usize> {
        Some(self.num_vars())
    }
    fn evaluation_at(&self, index: usize) -> Result<Field, SumcheckError> {
        CompositeMultilinearExtension::evaluation_at(self, index)
            .map_err(|_| SumcheckError::InvalidProductDimensions)
    }

    fn evaluations_prevalidated(&self) -> bool {
        true
    }
    fn validate_shape(&self, live_len: usize, cfg: &FieldConfig) -> Result<(), SumcheckError> {
        if live_len != self.live_len() {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        validate_factored_mle(self.repeated(), cfg)?;
        validate_field_values(
            self.tail_evaluations()
                .iter()
                .chain([self.origin_adjustment()]),
            cfg,
        )
    }
    fn build_prefix_accumulators<const K: usize, H: Sha256InnerBitSource + ?Sized>(
        &self,
        num_vars: usize,
        live_len: usize,
        h: &H,
        cfg: &FieldConfig,
        zero: &Field,
    ) -> Result<PrefixAccumulators, SumcheckError> {
        if self.repeated().inner_factor().len() < 1 << K {
            return build_prefix_accumulators_generic::<K, _, _>(
                num_vars, live_len, self, h, cfg, zero,
            );
        }
        let repeated = self.repeated();
        let mut result = if repeated.tensor_start() == 0
            && repeated.inner_factor().len() % (1 << K) == 0
            && K > 0
        {
            let beta_values = repeated_beta_values::<K, _>(
                repeated.outer_factor(),
                repeated.inner_factor(),
                h,
                zero,
                cfg,
            )?;
            scatter_beta_values::<K>(&beta_values, zero, &cfg)
        } else {
            build_factored_prefix_accumulators::<K, _>(
                num_vars,
                repeated.live_len(),
                repeated,
                h,
                cfg,
                zero,
            )?
        };
        let tail_start = self.repeated().live_len();
        let tail = match self.tail_runs() {
            // The run-structured pass needs the tail to start on a prefix-block
            // boundary (true whenever the SHA block width is a multiple of 2^K).
            Some(runs) if tail_start % (1 << K) == 0 => {
                let pieces = tail_pieces::<K>(self.tail_evaluations(), runs, cfg);
                let beta_values = tail_run_beta_values::<K, _>(
                    tail_start,
                    self.tail_evaluations().len(),
                    &pieces,
                    h,
                    cfg,
                    zero,
                )?;
                scatter_beta_values::<K>(&beta_values, zero, &cfg)
            }
            _ => build_prefix_accumulators_generic::<K, _, _>(
                num_vars,
                self.tail_evaluations().len(),
                &|i: usize| Ok(self.tail_evaluations()[i].clone()),
                &|i: usize| h.bit_at(tail_start + i),
                cfg,
                zero,
            )?,
        };
        let constant = build_prefix_accumulators_generic::<K, _, _>(
            num_vars,
            1 << K,
            &|i: usize| {
                Ok(if i == 0 {
                    self.origin_adjustment().clone()
                } else {
                    zero.clone()
                })
            },
            h,
            cfg,
            zero,
        )?;
        for other in [tail, constant] {
            for (dst, src) in result.rounds.iter_mut().zip(other.rounds) {
                for (dst, src) in dst.iter_mut().zip(src) {
                    dst[0] = cfg.add(&(dst[0]), &(&src[0]));
                    dst[1] = cfg.add(&(dst[1]), &(&src[1]));
                }
            }
        }
        Ok(result)
    }
    fn fold_prefix_table<const K: usize>(
        &self,
        num_vars: usize,
        live_len: usize,
        challenges: &[Field],
        cfg: &FieldConfig,
        zero: &Field,
        one: &Field,
    ) -> Result<CompactPrefixVTable, SumcheckError> {
        let width = self.repeated().inner_factor().len();
        let prefix = 1 << K;
        if width < prefix {
            return fold_prefix_v_table_generic::<K, _>(
                num_vars, live_len, self, challenges, cfg, zero, one,
            );
        }
        let weights = equality_weights_lsb(challenges, zero, one, &cfg);
        // Fold the low factor just once. Every SHA local wire reuses it.
        let low: Vec<Field> = self
            .repeated()
            .inner_factor()
            .chunks_exact(prefix)
            .map(|chunk| {
                let mut sum = product_accumulator_zero();
                for (a, b) in weights.iter().zip(chunk) {
                    product_multiply_accumulate(cfg, &mut sum, a, b);
                }
                product_reduce(sum, cfg)
            })
            .collect::<Result<_, _>>()?;
        let repeated_suffixes = self.repeated().live_len() / prefix;
        let suffix_count = live_len.div_ceil(prefix);
        let mut table = CompactPrefixVTable {
            values: vec![raw_montgomery(zero); (suffix_count + 1) & !1],
            suffix_count,
        };
        // With runs, a tail block folds to `Σ_pieces weight · F(lo, hi)` where
        // `F(lo, hi) = Σ_{lo ≤ k < hi} weights[k] · 2^(k − lo)` (the piece's weight
        // is its value at `lo`): one multiplication per piece.
        let structured_tail: Option<Vec<Field>> = match self.tail_runs() {
            Some(runs) if self.repeated().live_len() % prefix == 0 && K > 0 => {
                let pieces = tail_pieces::<K>(self.tail_evaluations(), runs, cfg);
                let two = Field::from_with_cfg(2u64, cfg);
                let mut two_pow = Vec::with_capacity(prefix + 1);
                let mut power = one.clone();
                for _ in 0..=prefix {
                    two_pow.push(power.clone());
                    power = cfg.mul(&(power), &(&two));
                }
                let mut factors: Vec<Option<Field>> = vec![None; (prefix + 1) * (prefix + 1)];
                let tail_blocks = self.tail_evaluations().len().div_ceil(prefix);
                let mut folded = vec![zero.clone(); tail_blocks];
                for piece in &pieces {
                    let shape = shape_index::<K>(piece.lo, piece.hi);
                    if factors[shape].is_none() {
                        let mut factor = zero.clone();
                        for k in piece.lo..piece.hi {
                            factor = cfg.add(
                                &(factor),
                                &(&(cfg.mul(&(weights[k].clone()), &(&two_pow[k - piece.lo])))),
                            );
                        }
                        factors[shape] = Some(factor);
                    }
                    let factor = factors[shape].as_ref().expect("just filled");
                    folded[piece.block] = cfg.add(
                        &(folded[piece.block]),
                        &(&(cfg.mul(&(piece.weight.clone()), &(factor)))),
                    );
                }
                Some(folded)
            }
            _ => None,
        };
        let fill = |(suffix, out): (usize, &mut RawMontgomery)| -> Result<(), SumcheckError> {
            let mut value = if suffix < repeated_suffixes {
                cfg.mul(
                    &(low[suffix % low.len()].clone()),
                    &(&self.repeated().outer_factor()[suffix / low.len()]),
                )
            } else if let Some(folded) = &structured_tail {
                folded[suffix - repeated_suffixes].clone()
            } else {
                let start = (suffix - repeated_suffixes) * prefix;
                let mut sum = product_accumulator_zero();
                for (a, b) in weights.iter().zip(
                    &self.tail_evaluations()
                        [start..self.tail_evaluations().len().min(start + prefix)],
                ) {
                    product_multiply_accumulate(cfg, &mut sum, a, b);
                }
                product_reduce(sum, cfg)?
            };
            if suffix == 0 {
                value = cfg.add(
                    &(value),
                    &(&(cfg.mul(&(weights[0].clone()), &(self.origin_adjustment())))),
                );
            }
            *out = raw_montgomery(&value);
            Ok(())
        };
        #[cfg(feature = "parallel")]
        table.values[..suffix_count]
            .par_iter_mut()
            .enumerate()
            .try_for_each(fill)?;
        #[cfg(not(feature = "parallel"))]
        table.values[..suffix_count]
            .iter_mut()
            .enumerate()
            .try_for_each(fill)?;
        Ok(table)
    }
}

/// One prefix block's intersection with a geometric run (or a lone tail
/// column): tail columns `block · 2^K + lo .. block · 2^K + hi` hold
/// `weight · 2^(i - lo)`, i.e. `weight` is the value at `lo` and the run
/// doubles along the block.
#[derive(Clone, Debug)]
struct TailPiece {
    block: usize,
    lo: usize,
    hi: usize,
    weight: Field,
}

/// Splits the tail into pieces in block order: every run contributes one piece
/// per prefix block it touches, and every column outside the runs is a piece
/// of length one carrying its own value. Together the pieces cover the tail
/// exactly once.
fn tail_pieces<const K: usize>(
    tail: &[Field],
    runs: &[(usize, usize, Field)],
    cfg: &FieldConfig,
) -> Vec<TailPiece> {
    let prefix = 1usize << K;
    let two = Field::from_with_cfg(2u64, cfg);
    let mut two_pow = Vec::with_capacity(prefix + 1);
    let mut power = Field::from_with_cfg(1u64, cfg);
    for _ in 0..=prefix {
        two_pow.push(power.clone());
        power = cfg.mul(&(power), &(&two));
    }
    let singles = |pieces: &mut Vec<TailPiece>, from: usize, to: usize| {
        for column in from..to {
            pieces.push(TailPiece {
                block: column / prefix,
                lo: column % prefix,
                hi: column % prefix + 1,
                weight: tail[column].clone(),
            });
        }
    };
    // Each run's pieces, preceded by the single columns between it and the
    // previous run; runs are independent, so they are built in parallel.
    let build = |index: usize| -> Vec<TailPiece> {
        let (start, len, ref base) = runs[index];
        let cursor = if index == 0 {
            0
        } else {
            runs[index - 1].0 + runs[index - 1].1
        };
        debug_assert!(start >= cursor);
        let mut pieces = Vec::with_capacity(start - cursor + len / prefix + 2);
        singles(&mut pieces, cursor, start);
        let end = start + len;
        let mut weight = base.clone();
        let mut column = start;
        while column < end {
            let block = column / prefix;
            let lo = column - block * prefix;
            let hi = (end - block * prefix).min(prefix);
            pieces.push(TailPiece {
                block,
                lo,
                hi,
                weight: weight.clone(),
            });
            weight = cfg.mul(&(weight), &(&two_pow[hi - lo]));
            column = block * prefix + hi;
        }
        pieces
    };
    #[cfg(feature = "parallel")]
    let per_run: Vec<Vec<TailPiece>> = if runs.len() >= 256 && rayon::current_num_threads() > 1 {
        (0..runs.len()).into_par_iter().map(build).collect()
    } else {
        (0..runs.len()).map(build).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let per_run: Vec<Vec<TailPiece>> = (0..runs.len()).map(build).collect();
    let mut pieces: Vec<TailPiece> = per_run.into_iter().flatten().collect();
    let cursor = runs.last().map_or(0, |&(start, len, _)| start + len);
    singles(&mut pieces, cursor, tail.len());
    pieces
}

/// `ext[beta][i]` for `beta < 3^K`, `i < 2^K`: the ternary extension of the
/// point-mass at `i`, in the coordinate convention of [`extend_lsb`]
/// (per coordinate, `0` is the slope `high − low`, `1` the value at `0`,
/// `2` the value at `1`; `beta = Σ beta_m 3^m`, `i_m` the m-th bit of `i`).
fn ternary_extension_table<const K: usize>() -> Vec<[i8; 16]> {
    let prefix = 1usize << K;
    let mut table = vec![[0i8; 16]; pow3(K)];
    for (beta, row) in table.iter_mut().enumerate() {
        for (i, cell) in row.iter_mut().enumerate().take(prefix) {
            let mut value = 1i8;
            let mut rest = beta;
            for m in 0..K {
                let bit = (i >> m) & 1 == 1;
                let factor = match rest % 3 {
                    0 => {
                        if bit {
                            1
                        } else {
                            -1
                        }
                    }
                    1 => {
                        if bit {
                            0
                        } else {
                            1
                        }
                    }
                    _ => {
                        if bit {
                            1
                        } else {
                            0
                        }
                    }
                };
                value *= factor;
                rest /= 3;
            }
            *cell = value;
        }
    }
    table
}

/// Per-thread accumulation state for the run-structured tail pass: one
/// accumulator per (piece shape, bit position), where a shape is the
/// `(lo, hi)` range of a piece inside its block.
struct TailShapeState {
    sums: Vec<LinearAccumulator>,
    used: Vec<bool>,
}

impl TailShapeState {
    fn new<const K: usize>() -> Self {
        let prefix = 1usize << K;
        let shapes = (prefix + 1) * (prefix + 1);
        Self {
            sums: (0..shapes * prefix)
                .map(|_| linear_accumulator_zero())
                .collect(),
            used: vec![false; shapes],
        }
    }
}

fn shape_index<const K: usize>(lo: usize, hi: usize) -> usize {
    lo * ((1usize << K) + 1) + hi
}

/// Adds every piece of one block to the shape accumulators: the piece's
/// weight goes to `(shape, i)` for each set bit `i` of the block.
fn accumulate_tail_block<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    state: &mut TailShapeState,
    pieces: &[TailPiece],
    block: usize,
    tail_start: usize,
    tail_len: usize,
    h: &H,
    reducer: &field::FpCtx<2>,
) -> Result<(), SumcheckError> {
    let prefix = 1usize << K;
    let base = block * prefix;
    let active = prefix.min(tail_len - base);
    let mut set_bits = [0usize; 16];
    let mut count = 0;
    // Iterate the set bits without a branch per bit (the bits are random).
    let mut word = h.bits_at(tail_start + base, active)?;
    while word != 0 {
        set_bits[count] = word.trailing_zeros() as usize;
        count += 1;
        word &= word - 1;
    }
    if count == 0 {
        return Ok(());
    }
    for piece in pieces {
        let shape = shape_index::<K>(piece.lo, piece.hi);
        state.used[shape] = true;
        for &i in &set_bits[..count] {
            linear_multiply_accumulate(
                reducer,
                &mut state.sums[shape * prefix + i],
                &piece.weight,
                &1u64,
            );
        }
    }
    Ok(())
}

/// The `3^K` prefix sums `Σ_blocks Ṽ(beta, block) · H̃(beta, block)` of the
/// tail, from its geometric runs.
///
/// With `V = weight · 2^(i − lo)` on each piece, `Ṽ(beta, block) = Σ_pieces weight ·
/// g_shape(beta)` where `g_{lo,hi}(beta) = Σ_{lo ≤ k < hi} ext[beta][k] · 2^(k − lo)`
/// depends only on the piece's shape, so the sum over blocks factors through
/// `A[shape][i] = Σ_{pieces of that shape} weight · h[block, i]`:
/// `S(beta) = Σ_shape g_shape(beta) · Σ_i ext[beta][i] · A[shape][i]`.
/// Every tail bit is read once and contributes field additions only; the
/// remaining work is constant in the tail length.
#[allow(clippy::too_many_arguments)]
fn tail_run_beta_values<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    tail_start: usize,
    tail_len: usize,
    pieces: &[TailPiece],
    h: &H,
    cfg: &FieldConfig,
    zero: &Field,
) -> Result<Vec<Field>, SumcheckError> {
    debug_assert!(K > 0 && K <= 4);
    let prefix = 1usize << K;
    let blocks = tail_len.div_ceil(prefix);
    // Pieces are in block order; index them by block.
    let mut block_ptr = vec![0usize; blocks + 1];
    for piece in pieces {
        block_ptr[piece.block + 1] += 1;
    }
    for b in 0..blocks {
        block_ptr[b + 1] += block_ptr[b];
    }
    debug_assert_eq!(block_ptr[blocks], pieces.len());

    #[cfg(feature = "parallel")]
    let state = if blocks >= 1 << 12 && rayon::current_num_threads() > 1 {
        (0..blocks)
            .into_par_iter()
            .try_fold(
                || TailShapeState::new::<K>(),
                |mut state, block| -> Result<_, SumcheckError> {
                    accumulate_tail_block::<K, _>(
                        &mut state,
                        &pieces[block_ptr[block]..block_ptr[block + 1]],
                        block,
                        tail_start,
                        tail_len,
                        h,
                        cfg,
                    )?;
                    Ok(state)
                },
            )
            .try_reduce(
                || TailShapeState::new::<K>(),
                |mut left, right| {
                    for (l, r) in left.sums.iter_mut().zip(right.sums) {
                        linear_merge(l, r);
                    }
                    for (l, r) in left.used.iter_mut().zip(right.used) {
                        *l |= r;
                    }
                    Ok(left)
                },
            )?
    } else {
        let mut state = TailShapeState::new::<K>();
        for block in 0..blocks {
            accumulate_tail_block::<K, _>(
                &mut state,
                &pieces[block_ptr[block]..block_ptr[block + 1]],
                block,
                tail_start,
                tail_len,
                h,
                cfg,
            )?;
        }
        state
    };
    #[cfg(not(feature = "parallel"))]
    let state = {
        let mut state = TailShapeState::new::<K>();
        for block in 0..blocks {
            accumulate_tail_block::<K, _>(
                &mut state,
                &pieces[block_ptr[block]..block_ptr[block + 1]],
                block,
                tail_start,
                tail_len,
                h,
                cfg,
            )?;
        }
        state
    };

    finish_tail_shapes::<K>(state, cfg, zero)
}

fn finish_tail_shapes<const K: usize>(state: TailShapeState, cfg: &FieldConfig, zero: &Field) -> Result<Vec<Field>, SumcheckError> {
    let prefix = 1usize << K;
    let shapes = (prefix + 1) * (prefix + 1);
    let TailShapeState { sums, used } = state;
    let mut reduced: Vec<Option<Vec<Field>>> = vec![None; shapes];
    let mut sums = sums.into_iter();
    for (shape, reduced) in reduced.iter_mut().enumerate() {
        let accumulators: Vec<LinearAccumulator> = sums.by_ref().take(prefix).collect();
        if used[shape] {
            *reduced = Some(
                accumulators
                    .into_iter()
                    .map(|acc| linear_reduce(acc, cfg))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }

    let ext = ternary_extension_table::<K>();
    let signed_field = |value: i64| -> Field {
        let magnitude = Field::from_with_cfg(value.unsigned_abs(), cfg);
        if value < 0 {
            cfg.sub(&(zero.clone()), &(&magnitude))
        } else {
            magnitude
        }
    };
    let mut beta_values = Vec::with_capacity(pow3(K));
    for row in &ext {
        // g_{lo,hi} = G[hi] − G[lo] with G[t] = Σ_{k<t} ext[k] · 2^k.
        let mut prefix_sums = [0i64; 17];
        for k in 0..prefix {
            prefix_sums[k + 1] = prefix_sums[k] + i64::from(row[k]) * (1i64 << k);
        }
        let mut total = zero.clone();
        for (shape, values) in reduced.iter().enumerate() {
            let Some(values) = values else { continue };
            let (lo, hi) = (shape / (prefix + 1), shape % (prefix + 1));
            // g_{lo,hi} = Σ_{lo ≤ k < hi} ext[k] · 2^(k − lo): the difference of prefix
            // sums is a multiple of 2^lo, so the division is exact.
            let g = (prefix_sums[hi] - prefix_sums[lo]) / (1i64 << lo);
            if g == 0 {
                continue;
            }
            let mut inner = linear_accumulator_zero();
            let mut touched = false;
            for (i, value) in values.iter().enumerate() {
                if row[i] != 0 {
                    linear_multiply_accumulate_signed(
                        cfg,
                        &mut inner,
                        value,
                        i64::from(row[i]),
                        zero,
                    );
                    touched = true;
                }
            }
            if touched {
                let inner = linear_reduce(inner, cfg)?;
                total = cfg.add(&(total), &(&(cfg.mul(&(signed_field(g)), &(&inner)))));
            }
        }
        beta_values.push(total);
    }
    Ok(beta_values)
}

/// Per-thread accumulation state for the weights-first SHA pass: one
/// accumulator per (instance block, bit position).
struct RepeatedState {
    sums: Vec<LinearAccumulator>,
}

/// The `3^K` prefix sums of the repeated part `V[j·N + i] = outer[j] · inner[i]`
/// (N instance weights per local wire, N a multiple of 2^K).
///
/// With blocks of 2^K consecutive instances, `Ṽ(beta, wire j, block p) = outer[j] ·
/// ũ_p(beta)` where `ũ_p` is the ternary extension of the inner factor's block
/// `p`, so the sum over wires factors through
/// `A[p][i] = Σ_j outer[j] · h[j·N + p·2^K + i]`: every bit costs one field
/// addition and the remaining `Σ_p ũ_p(beta) · Σ_i ext[beta][i] · A[p][i]` is
/// independent of the number of wires.
fn repeated_beta_values<const K: usize, H: Sha256InnerBitSource + ?Sized>(
    outer: &[Field],
    inner: &[Field],
    h: &H,
    zero: &Field,
    reducer: &field::FpCtx<2>,
) -> Result<Vec<Field>, SumcheckError> {
    debug_assert!(K > 0 && K <= 4);
    let prefix = 1usize << K;
    let width = inner.len();
    debug_assert_eq!(width % prefix, 0);
    let positions = width / prefix;
    let cells = positions * prefix;
    let accumulate_wire = |state: &mut RepeatedState, wire: usize| -> Result<(), SumcheckError> {
        let weight = &outer[wire];
        let base = wire * width;
        for position in 0..positions {
            let mut word = h.bits_at(base + position * prefix, prefix)?;
            while word != 0 {
                let i = word.trailing_zeros() as usize;
                linear_multiply_accumulate(
                    reducer,
                    &mut state.sums[position * prefix + i],
                    weight,
                    &1u64,
                );
                word &= word - 1;
            }
        }
        Ok(())
    };
    let new_state = || RepeatedState {
        sums: (0..cells).map(|_| linear_accumulator_zero()).collect(),
    };
    #[cfg(feature = "parallel")]
    let state = if outer.len() >= 1 << 8 && rayon::current_num_threads() > 1 {
        (0..outer.len())
            .into_par_iter()
            .try_fold(new_state, |mut state, wire| -> Result<_, SumcheckError> {
                accumulate_wire(&mut state, wire)?;
                Ok(state)
            })
            .try_reduce(new_state, |mut left, right| {
                for (l, r) in left.sums.iter_mut().zip(right.sums) {
                    linear_merge(l, r);
                }
                Ok(left)
            })?
    } else {
        let mut state = new_state();
        for wire in 0..outer.len() {
            accumulate_wire(&mut state, wire)?;
        }
        state
    };
    #[cfg(not(feature = "parallel"))]
    let state = {
        let mut state = new_state();
        for wire in 0..outer.len() {
            accumulate_wire(&mut state, wire)?;
        }
        state
    };
    let table: Vec<Field> = state
        .sums
        .into_iter()
        .map(|acc| linear_reduce(acc, &reducer))
        .collect::<Result<Vec<_>, _>>()?;

    let ext = ternary_extension_table::<K>();
    let mut beta_values = Vec::with_capacity(pow3(K));
    for row in &ext {
        let mut total = zero.clone();
        for position in 0..positions {
            let block = &inner[position * prefix..(position + 1) * prefix];
            let cells = &table[position * prefix..(position + 1) * prefix];
            let mut u = linear_accumulator_zero();
            let mut a = linear_accumulator_zero();
            let mut touched = false;
            for i in 0..prefix {
                if row[i] != 0 {
                    linear_multiply_accumulate_signed(
                        reducer,
                        &mut u,
                        &block[i],
                        i64::from(row[i]),
                        zero,
                    );
                    linear_multiply_accumulate_signed(
                        reducer,
                        &mut a,
                        &cells[i],
                        i64::from(row[i]),
                        zero,
                    );
                    touched = true;
                }
            }
            if touched {
                let u = linear_reduce(u, &reducer)?;
                let a = linear_reduce(a, &reducer)?;
                total = reducer.add(&(total), &(&(reducer.mul(&(u), &(&a)))));
            }
        }
        beta_values.push(total);
    }
    Ok(beta_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Blake3Transcript;

    /// The closed-form ternary extension table agrees with `extend_lsb`.
    #[test]
    fn ternary_extension_table_matches_extend_lsb() {
        fn check<const K: usize>() {
            let prefix = 1usize << K;
            let table = ternary_extension_table::<K>();
            let mut values: Vec<i64> = (0..prefix).map(|i| (i as i64 * 7 + 3) % 11 - 5).collect();
            let original = values.clone();
            let mut scratch = Vec::new();
            extend_lsb::<i64, K, _>(&mut values, &mut scratch, &0, |high, low| *high - *low);
            for (beta, row) in table.iter().enumerate() {
                let expected: i64 = (0..prefix).map(|i| i64::from(row[i]) * original[i]).sum();
                assert_eq!(values[beta], expected, "K={K} beta={beta}");
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    /// Structured tail accumulators and folds equal the generic ones on a
    /// synthetic composite whose tail mixes runs that straddle blocks, single
    /// columns, and an unstructured gap.
    #[test]
    fn structured_tail_matches_generic() {
        let cfg = crate::piop::spartan::bitz::spartan_bitz_field_config();
        let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
        let f = |x: u64| Field::from_with_cfg(x, &cfg);
        let zero = f(0);
        let one = f(1);
        let two = f(2);
        let width = 16usize;
        let high: Vec<_> = (0..3).map(|i| f(i * 41 + 5)).collect();
        let low: Vec<_> = (0..width).map(|i| f(i as u64 + 17)).collect();
        // Tail: [gap of 5 scalars][run A 23 long][run B 9 long][3 scalars][run C 40 long][gap 2].
        let runs_spec: [(usize, usize, u64); 3] = [(5, 23, 977), (28, 9, 1_234_567), (40, 40, 3)];
        let tail_len = 82usize;
        let mut tail = vec![zero.clone(); tail_len];
        for (j, value) in tail.iter_mut().enumerate() {
            *value = f((j as u64 * 1_000_003 + 12_345) % 1_000_000_007);
        }
        let mut runs = Vec::new();
        for &(start, len, base) in &runs_spec {
            let base = f(base);
            let mut value = base.clone();
            for k in 0..len {
                tail[start + k] = value.clone();
                value = cfg.mul(&(value), &(&two));
            }
            runs.push((start, len, base));
        }
        let num_vars = (high.len() * width + tail_len).next_power_of_two().ilog2() as usize;
        let plain =
            CompositeMultilinearExtension::from_parts(num_vars, &high, &low, &tail, f(29), &cfg)
                .unwrap();
        let structured =
            CompositeMultilinearExtension::from_parts(num_vars, &high, &low, &tail, f(29), &cfg)
                .unwrap()
                .with_tail_runs(&runs)
                .unwrap();
        let live = plain.live_len();
        let bit = |i: usize| Ok(u64::from((i * 17 + 5) % 11 < 5));
        fn check<const K: usize>(
            plain: &CompositeMultilinearExtension<'_, Field>,
            structured: &CompositeMultilinearExtension<'_, Field>,
            num_vars: usize,
            live: usize,
            bit: &(dyn Fn(usize) -> Result<u64, SumcheckError> + Sync),
            cfg: &FieldConfig,
            zero: &Field,
            one: &Field,
        ) {
            let bit_source = |i: usize| bit(i);
            let a = plain
                .build_prefix_accumulators::<K, _>(num_vars, live, &bit_source, cfg, zero)
                .unwrap();
            let b = structured
                .build_prefix_accumulators::<K, _>(num_vars, live, &bit_source, cfg, zero)
                .unwrap();
            for (round, (x, y)) in a.rounds.iter().zip(&b.rounds).enumerate() {
                assert!(x == y, "K={K}: accumulators differ in round {round}");
            }
            let challenges: Vec<_> = (0..K)
                .map(|i| Field::from_with_cfg(i as u64 * 977 + 31, cfg))
                .collect();
            let a = plain
                .fold_prefix_table::<K>(num_vars, live, &challenges, cfg, zero, one)
                .unwrap();
            let b = structured
                .fold_prefix_table::<K>(num_vars, live, &challenges, cfg, zero, one)
                .unwrap();
            assert_eq!(a.suffix_count, b.suffix_count);
            assert!(a.values == b.values, "K={K}: folded tables differ");
        }
        check::<1>(&plain, &structured, num_vars, live, &bit, &cfg, &zero, &one);
        check::<2>(&plain, &structured, num_vars, live, &bit, &cfg, &zero, &one);
        check::<3>(&plain, &structured, num_vars, live, &bit, &cfg, &zero, &one);
        check::<4>(&plain, &structured, num_vars, live, &bit, &cfg, &zero, &one);
    }

    #[test]
    fn composite_prefix_matches_generic_with_constant_tail_and_boundary() {
        let cfg = crate::piop::spartan::bitz::spartan_bitz_field_config();
        let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
        let f = |x| Field::from_with_cfg(x, &cfg);
        for width in [8usize, 16, 32] {
            let high = [f(5u64), f(13), f(0)];
            let low: Vec<_> = (0..width).map(|i| f(i as u64 + 17)).collect();
            let tail: Vec<_> = (0..19).map(|i| f(i + 101)).collect();
            let num_vars = (high.len() * low.len() + tail.len())
                .next_power_of_two()
                .ilog2() as usize;
            let coefficients = CompositeMultilinearExtension::from_parts(
                num_vars,
                &high,
                &low,
                &tail,
                f(29),
                &cfg,
            )
            .unwrap();
            let live = coefficients.live_len();
            let num_vars = live.next_power_of_two().ilog2() as usize;
            let bit = |i| Ok(u64::from((i * 17 + 5) % 11 < 5));
            let mut claim = f(0);
            for i in 0..live {
                if bit(i).unwrap() != 0 {
                    claim = cfg.add(&(claim), &(&coefficients.evaluation_at(i).unwrap()));
                }
            }
            for prefix in 0..=4 {
                let actual = {
                    let coefficients = &coefficients;
                    let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                        crate::sumcheck::inner::packed::Sha256InnerGrinding,
                    >::with_round_offset(0, 0);
                    crate::sumcheck::inner::prove_inner_sumcheck(
                        &cfg,
                        &mut Blake3Transcript::new(),
                        claim.clone(),
                        crate::sumcheck::inner::packed::PackedInput::new(
                            coefficients,
                            &bit,
                            num_vars,
                            coefficients.live_len(),
                            prefix,
                        ),
                        (),
                        &mut boundary,
                    )
                    .map(|out| {
                        crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                            out,
                            boundary.into_nonces(),
                        )
                    })
                }
                .unwrap();
                let expected = {
                    let coefficients = &|i| {
                        coefficients
                            .evaluation_at(i)
                            .map_err(|_| SumcheckError::InvalidProductDimensions)
                    };
                    let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
                        crate::sumcheck::inner::packed::Sha256InnerGrinding,
                    >::with_round_offset(0, 0);
                    crate::sumcheck::inner::prove_inner_sumcheck(
                        &cfg,
                        &mut Blake3Transcript::new(),
                        claim.clone(),
                        crate::sumcheck::inner::packed::PackedInput::new(
                            coefficients,
                            &bit,
                            num_vars,
                            live,
                            prefix,
                        ),
                        (),
                        &mut boundary,
                    )
                    .map(|out| {
                        crate::sumcheck::inner::packed::Sha256InnerSumcheckOutput::from_inner(
                            out,
                            boundary.into_nonces(),
                        )
                    })
                }
                .unwrap();
                assert!(actual == expected, "width {width}, prefix {prefix}");
            }
        }
    }
}
