//! Inner-sumcheck adapter over a borrowed compact Montgomery tail.
use super::*;
use crate::piop::spartan::raw_monty::{Raw, RawFieldStorage, words_to_raw};
use crate::poly::mle::EqualityWeights;
use circuit::montgomery_tail::MontgomeryTail;

pub(crate) struct CompactCompositeMle<'a> {
    repeated: FactoredMultilinearExtension<'a, Field>,
    tail: &'a MontgomeryTail,
    origin: Field,
}
impl<'a> CompactCompositeMle<'a> {
    pub(crate) fn new(
        num_vars: usize,
        high: &'a [Field],
        low: &'a [Field],
        tail: &'a MontgomeryTail,
        origin: Field,
        cfg: &FieldConfig,
    ) -> Result<Self, SumcheckError> {
        let repeated = FactoredMultilinearExtension::from_factors(num_vars, high, low, cfg)
            .map_err(|_| SumcheckError::InvalidProductDimensions)?;
        if repeated
            .live_len()
            .checked_add(tail.len())
            .is_none_or(|n| n > 1usize << num_vars)
            || tail.field().modulus() != cfg.modulus()
        {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        let this = Self {
            repeated,
            tail,
            origin,
        };
        this.validate_shape(this.live_len(), cfg)?;
        Ok(this)
    }
    pub(crate) fn num_vars(&self) -> usize {
        self.repeated.num_vars()
    }
    pub(crate) fn live_len(&self) -> usize {
        self.repeated.live_len() + self.tail.len()
    }
    pub(crate) fn evaluate(
        &self,
        point: &[Field],
        cfg: &FieldConfig,
    ) -> Result<Field, SumcheckError> {
        if point.len() != self.num_vars() || cfg.modulus() != self.tail.field().modulus() {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        for value in point {
            validate_field_value(value, cfg)?;
        }
        let (low, high) = crate::piop::spartan::matrix::make_equality_factors(point, cfg)
            .map_err(|_| SumcheckError::InvalidProductDimensions)?;
        let equality = EqualityWeights::from_tables(low.evaluations, high.evaluations, cfg);
        let repeated = self
            .repeated
            .evaluate(point, cfg)
            .map_err(|_| SumcheckError::InvalidProductDimensions)?;
        let origin = cfg.mul(&self.origin, &equality.at(0));
        let tail = evaluate_compact_tail(cfg, self.repeated.live_len(), self.tail, &equality);
        Ok(cfg.add(&cfg.add(&repeated, &origin), &tail))
    }
    pub(crate) fn evaluation_at(&self, index: usize) -> Result<Field, SumcheckError> {
        if index >= 1usize << self.num_vars() {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        let cfg = self.tail.field();
        let mut value = if index < self.repeated.live_len() {
            self.repeated
                .evaluation_at(index)
                .map_err(|_| SumcheckError::InvalidProductDimensions)?
        } else {
            crate::utils::delayed_reduction::element(
                cfg,
                crate::piop::spartan::raw_monty::words_to_raw(
                    &self
                        .tail
                        .value(index - self.repeated.live_len())
                        .unwrap_or([0; 2]),
                ),
            )
        };
        if index == 0 {
            value = cfg.add(&value, &self.origin);
        }
        Ok(value)
    }
}
impl InnerSumcheckMleSource for CompactCompositeMle<'_> {
    fn declared_num_vars(&self) -> Option<usize> {
        Some(self.num_vars())
    }
    fn evaluation_at(&self, index: usize) -> Result<Field, SumcheckError> {
        Self::evaluation_at(self, index)
    }
    fn evaluations_prevalidated(&self) -> bool {
        true
    }
    fn validate_shape(&self, len: usize, cfg: &FieldConfig) -> Result<(), SumcheckError> {
        if len != self.live_len() || self.tail.field().modulus() != cfg.modulus() {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        validate_factored_mle(&self.repeated, cfg)?;
        validate_field_value(&self.origin, cfg)
    }
    fn build_prefix_accumulators<const K: usize, H: Sha256InnerBitSource + ?Sized>(
        &self,
        num_vars: usize,
        live_len: usize,
        h: &H,
        cfg: &FieldConfig,
        zero: &Field,
    ) -> Result<PrefixAccumulators, SumcheckError> {
        let reducer = cfg;
        let prefix = 1usize << K;
        let offset = self.repeated.live_len();
        if K == 0 {
            return Ok(PrefixAccumulators::new::<K>(zero));
        }
        let mut result = if self.repeated.inner_factor().len() % prefix == 0 {
            let beta = repeated_beta_values::<K, _>(
                self.repeated.outer_factor(),
                self.repeated.inner_factor(),
                h,
                zero,
                reducer,
            )?;
            scatter_beta_values::<K>(&beta, zero, cfg)
        } else {
            // Include the complete boundary block's witness bits. Coefficients
            // beyond the repeated region are zero, but H there need not be.
            build_prefix_accumulators_generic::<K, _, _>(
                num_vars,
                (offset.div_ceil(prefix) * prefix).min(live_len),
                &|i| {
                    self.repeated
                        .evaluation_at(i)
                        .map_err(|_| SumcheckError::InvalidProductDimensions)
                },
                h,
                cfg,
                zero,
            )?
        };
        // Each worker owns a fixed-size block range and a bounded piece list.
        // No full-tail list or block-pointer array is constructed.
        const BLOCKS: usize = 1024;
        let first_block = offset / prefix;
        let blocks = live_len.div_ceil(prefix);
        let chunks = (blocks - first_block).div_ceil(BLOCKS);
        let process = |chunk: usize| -> Result<TailShapeState, SumcheckError> {
            let begin = first_block + chunk * BLOCKS;
            let end = blocks.min(begin + BLOCKS);
            let mut pieces = Vec::with_capacity((end - begin) * 2);
            let two = Field::from_with_cfg(2u64, cfg);
            let mut powers = vec![Field::one_with_cfg(cfg)];
            for i in 0..prefix {
                powers.push(cfg.mul(&powers[i], &two));
            }
            self.tail.visit(
                (begin * prefix).saturating_sub(offset)..self.tail.len().min(end * prefix - offset),
                |start, len, base| {
                    let mut pos = offset + start;
                    let end = pos + len;
                    let mut value = crate::utils::delayed_reduction::element(
                        cfg,
                        crate::piop::spartan::raw_monty::words_to_raw(&base),
                    );
                    while pos < end {
                        let block = pos / prefix;
                        let lo = pos % prefix;
                        let take = (prefix - lo).min(end - pos);
                        pieces.push(TailPiece {
                            block,
                            lo,
                            hi: lo + take,
                            weight: value.clone(),
                        });
                        value = cfg.mul(&value, &powers[take]);
                        pos += take;
                    }
                },
            );
            let mut state = TailShapeState::new::<K>();
            let mut cursor = 0;
            for block in begin..end {
                let first = cursor;
                while cursor < pieces.len() && pieces[cursor].block == block {
                    cursor += 1;
                }
                accumulate_tail_block::<K, _>(
                    &mut state,
                    &pieces[first..cursor],
                    block,
                    0,
                    live_len,
                    h,
                    reducer,
                )?;
            }
            Ok(state)
        };
        let merge = |mut a: TailShapeState, b: TailShapeState| -> Result<_, SumcheckError> {
            for (a, b) in a.sums.iter_mut().zip(b.sums) {
                linear_merge(a, b);
            }
            for (a, b) in a.used.iter_mut().zip(b.used) {
                *a |= b;
            }
            Ok(a)
        };
        #[cfg(feature = "parallel")]
        let state = (0..chunks)
            .into_par_iter()
            .map(process)
            .try_reduce(|| TailShapeState::new::<K>(), merge)?;
        #[cfg(not(feature = "parallel"))]
        let state = (0..chunks)
            .map(process)
            .try_fold(TailShapeState::new::<K>(), |a, b| merge(a, b?))?;
        let beta = finish_tail_shapes::<K>(state, cfg, zero)?;
        let tail = scatter_beta_values::<K>(&beta, zero, cfg);
        let constant = build_prefix_accumulators_generic::<K, _, _>(
            num_vars,
            prefix.min(live_len),
            &|i| {
                Ok(if i == 0 {
                    self.origin.clone()
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
                    dst[0] = cfg.add(&dst[0], &src[0]);
                    dst[1] = cfg.add(&dst[1], &src[1]);
                }
            }
        }
        Ok(result)
    }
    fn fold_prefix_table<const K: usize>(
        &self,
        _num_vars: usize,
        live_len: usize,
        challenges: &[Field],
        cfg: &FieldConfig,
        zero: &Field,
        one: &Field,
    ) -> Result<CompactPrefixVTable, SumcheckError> {
        use crate::piop::spartan::raw_monty::{RawFieldStorage, raw_to_words, words_to_raw};
        let reducer = cfg;
        let prefix = 1usize << K;
        let offset = self.repeated.live_len();
        let count = live_len.div_ceil(prefix);
        let weights = equality_weights_lsb(challenges, zero, one, cfg);
        let raw_weights = reducer.raw_vec(&weights);
        let mut powers = vec![reducer.raw(one)];
        for i in 0..prefix {
            powers.push(reducer.add_raw(powers[i], powers[i]));
        }
        let mut factors = vec![0; (prefix + 1) * (prefix + 1)];
        for lo in 0..prefix {
            let mut sum = 0;
            for hi in lo + 1..=prefix {
                sum = reducer.add_raw(
                    sum,
                    reducer.mul_raw(raw_weights[hi - 1], powers[hi - lo - 1]),
                );
                factors[shape_index::<K>(lo, hi)] = sum;
            }
        }
        let low = self.repeated.inner_factor();
        let folded_low: Vec<_> = low
            .chunks_exact(prefix)
            .filter(|_| low.len() % prefix == 0)
            .map(|chunk| {
                chunk.iter().zip(&raw_weights).fold(0, |sum, (v, w)| {
                    reducer.add_raw(sum, reducer.mul_raw(reducer.raw(v), *w))
                })
            })
            .collect();
        let mut table = CompactPrefixVTable {
            values: vec![[0; 2]; (count + 1) & !1],
            suffix_count: count,
        };
        let fill = |(chunk, out): (usize, &mut [RawMontgomery])| -> Result<(), SumcheckError> {
            const CHUNK: usize = 1024;
            let first = chunk * CHUNK;
            let end = first + out.len();
            for (i, value) in out.iter_mut().enumerate() {
                let block = first + i;
                if block * prefix >= offset {
                    break;
                }
                let folded = if !folded_low.is_empty() && (block + 1) * prefix <= offset {
                    reducer.mul_raw(
                        folded_low[block % folded_low.len()],
                        reducer.raw(&self.repeated.outer_factor()[block / folded_low.len()]),
                    )
                } else {
                    let mut sum = 0;
                    for (j, w) in raw_weights.iter().enumerate() {
                        let index = block * prefix + j;
                        if index >= offset {
                            break;
                        }
                        let v = self
                            .repeated
                            .evaluation_at(index)
                            .map_err(|_| SumcheckError::InvalidProductDimensions)?;
                        sum = reducer.add_raw(sum, reducer.mul_raw(reducer.raw(&v), *w));
                    }
                    sum
                };
                *value = raw_to_words(folded);
            }
            let lo = (first * prefix).saturating_sub(offset).min(self.tail.len());
            let hi = (end * prefix).saturating_sub(offset).min(self.tail.len());
            self.tail.visit(lo..hi, |start, len, base| {
                let mut pos = offset + start;
                let end = pos + len;
                let mut base = words_to_raw(&base);
                while pos < end {
                    let block = pos / prefix;
                    let lo = pos % prefix;
                    let take = (prefix - lo).min(end - pos);
                    let value = &mut out[block - first];
                    *value = raw_to_words(reducer.add_raw(
                        words_to_raw(value),
                        reducer.mul_raw(base, factors[shape_index::<K>(lo, lo + take)]),
                    ));
                    base = reducer.mul_raw(base, powers[take]);
                    pos += take;
                }
            });
            if first == 0 {
                out[0] = raw_to_words(reducer.add_raw(
                    words_to_raw(&out[0]),
                    reducer.mul_raw(raw_weights[0], reducer.raw(&self.origin)),
                ));
            }
            Ok(())
        };
        #[cfg(feature = "parallel")]
        table.values[..count]
            .par_chunks_mut(1024)
            .enumerate()
            .try_for_each(fill)?;
        #[cfg(not(feature = "parallel"))]
        table.values[..count]
            .chunks_mut(1024)
            .enumerate()
            .try_for_each(fill)?;
        Ok(table)
    }
}

/// Evaluate the authoritative tail directly. The backward low-weight sum
/// closes each geometric intersection without expansion or inverses.
fn evaluate_compact_tail(
    cfg: &FieldConfig,
    offset: usize,
    tail: &circuit::montgomery_tail::MontgomeryTail,
    equality: &EqualityWeights<Field>,
) -> Field {
    let ctx = tail.field();
    let (low, high) = (ctx.raw_vec(equality.low()), ctx.raw_vec(equality.high()));
    let block = low.len();
    let shift = block.ilog2();
    let mask = block - 1;
    let mut suffix = vec![0 as Raw; block + 1];
    for t in (0..block).rev() {
        suffix[t] = ctx.add_raw(low[t], ctx.add_raw(suffix[t + 1], suffix[t + 1]));
    }
    let mut pow2 = Vec::with_capacity(block + 1);
    pow2.push(ctx.native_residue(1));
    for k in 0..block {
        pow2.push(ctx.add_raw(pow2[k], pow2[k]));
    }
    let mut sum = 0 as Raw;
    tail.visit(0..tail.len(), |start, len, base| {
        let mut index = offset + start;
        let mut value = words_to_raw(&base);
        if len == 1 {
            let weight = ctx.mul_raw(low[index & mask], high[index >> shift]);
            sum = ctx.add_raw(sum, ctx.mul_raw(weight, value));
            return;
        }
        let end = index + len;
        while index < end {
            let b = index >> shift;
            let lo = index & mask;
            let hi = (lo + end - index).min(block);
            let part = ctx.sub_raw(suffix[lo], ctx.mul_raw(pow2[hi - lo], suffix[hi]));
            sum = ctx.add_raw(sum, ctx.mul_raw(ctx.mul_raw(value, part), high[b]));
            value = ctx.mul_raw(value, pow2[hi - lo]);
            index += hi - lo;
        }
    });
    crate::utils::delayed_reduction::element(cfg, sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piop::spartan::raw_monty::{RawFieldStorage, raw_to_words};
    use crate::transcript::{Blake3Transcript, traits::Transcript};
    use circuit::montgomery_tail::TailSegment;

    fn prove_source<
        T: Transcript,
        S: InnerSumcheckMleSource + ?Sized,
        H: Sha256InnerBitSource + ?Sized,
    >(
        transcript: &mut T,
        claim: Field,
        vars: usize,
        live: usize,
        coefficients: &S,
        bits: &H,
        prefix: usize,
        cfg: &FieldConfig,
    ) -> Result<Sha256InnerSumcheckOutput, SumcheckError> {
        let mut boundary = crate::sumcheck::boundary::ProverGrindingRoundBoundary::<
            Sha256InnerGrinding,
        >::with_round_offset(0, 0);
        crate::sumcheck::inner::prove_inner_sumcheck(
            cfg,
            transcript,
            claim,
            PackedInput::new(coefficients, bits, vars, live, prefix),
            (),
            &mut boundary,
        )
        .map(|out| Sha256InnerSumcheckOutput::from_inner(out, boundary.into_nonces()))
    }

    #[test]
    fn empty_tail_tiny_domains_and_strict_witness_bounds() {
        let cfg = Field::make_cfg(&field::Uint::from((1u128 << 89) - 1)).unwrap();
        let f = |n: u64| Field::from_with_cfg(n, &cfg);
        for vars in 0..=5 {
            for len in [1, 3, 5].into_iter().filter(|&n| n <= 1 << vars) {
                let low: Vec<_> = (0..len).map(|i| f(i as u64 + 2)).collect();
                let high = [f(3)];
                let tail = MontgomeryTail::new(cfg.clone(), 0, vec![], vec![]).unwrap();
                let compact =
                    CompactCompositeMle::new(vars, &high, &low, &tail, f(7), &cfg).unwrap();
                let mut dense: Vec<_> = low.iter().map(|v| cfg.mul(&high[0], v)).collect();
                dense[0] = cfg.add(&dense[0], &f(7));
                dense.resize(1 << vars, f(0));
                let oracle = |i: usize| Ok(dense[i].clone());
                let bits = |i: usize| {
                    if i >= len {
                        return Err(SumcheckError::InvalidProductDimensions);
                    }
                    Ok(u64::from(i % 2 == 0))
                };
                let claim = (0..len)
                    .step_by(2)
                    .fold(f(0), |sum, i| cfg.add(&sum, &dense[i]));
                for prefix in 0..=vars.min(4) {
                    let mut expected_t = Blake3Transcript::new();
                    let mut actual_t = Blake3Transcript::new();
                    let expected = prove_source(
                        &mut expected_t,
                        claim.clone(),
                        vars,
                        len,
                        &oracle,
                        &bits,
                        prefix,
                        &cfg,
                    )
                    .unwrap();
                    let actual = prove_source(
                        &mut actual_t,
                        claim.clone(),
                        vars,
                        len,
                        &compact,
                        &bits,
                        prefix,
                        &cfg,
                    )
                    .unwrap();
                    assert!(
                        actual == expected,
                        "vars={vars}, len={len}, prefix={prefix}"
                    );
                    assert_eq!(actual_t.state_digest(), expected_t.state_digest());
                }
            }
        }
    }

    #[test]
    fn compact_prefix_and_proof_match_independent_dense_coefficients() {
        for prime in [(1u128 << 89) - 1, (1u128 << 127) - 1] {
            let cfg = Field::make_cfg(&field::Uint::from(prime)).unwrap();
            let reducer = crate::utils::delayed_reduction::prepare_field(&cfg).unwrap();
            let f = |x: u64| Field::from_with_cfg(x, &cfg);
            for width in [1usize, 2, 3, 4, 7, 8, 16, 32] {
                let high = [f(3), f(7), f(0)];
                let low: Vec<_> = (0..width).map(|i| f(i as u64 + 5)).collect();
                let tail_len = if width == 16 { 16_441 } else { 99 };
                let mut integers = vec![0u128; tail_len];
                let mut value = 3;
                for v in &mut integers[3..tail_len - 7] {
                    *v = value;
                    value = 2 * value % prime;
                }
                integers[1] = 5;
                integers[tail_len - 2] = 7;
                let encode = |v: u128| raw_to_words(reducer.raw(&Field::from_with_cfg(v, &cfg)));
                let mut tail = MontgomeryTail::new(
                    reducer.clone(),
                    tail_len,
                    vec![
                        TailSegment::Literals {
                            start: 1,
                            values: 0..1,
                        },
                        TailSegment::Geometric {
                            start: 3,
                            len: tail_len - 10,
                            base: encode(3),
                        },
                        TailSegment::Literals {
                            start: tail_len - 2,
                            values: 1..2,
                        },
                    ],
                    vec![encode(5), encode(7)],
                )
                .unwrap();
                let updates = [(3, 9u128), (15, 11), (16, 2), (tail_len - 8, 4)];
                tail.add_sparse(&updates.map(|(i, v)| (i, encode(v))))
                    .unwrap();
                for (i, v) in updates {
                    integers[i] = (integers[i] + v) % prime;
                }
                let len = high.len() * width + tail_len;
                let vars = len.next_power_of_two().ilog2() as usize;
                let compact =
                    CompactCompositeMle::new(vars, &high, &low, &tail, f(13), &cfg).unwrap();
                // The oracle uses integer doubling and direct tensor products,
                // never the new segments, iterator, or coefficient accessor.
                let mut dense = Vec::with_capacity(1 << vars);
                for h in &high {
                    dense.extend(low.iter().map(|l| cfg.mul(h, l)));
                }
                dense.extend(integers.iter().map(|&v| Field::from_with_cfg(v, &cfg)));
                dense[0] = cfg.add(&dense[0], &f(13));
                dense.resize(1 << vars, f(0));
                let oracle = |i: usize| Ok(dense[i].clone());
                let bits = |i: usize| Ok(u64::from((i * 17 + 5) % 11 < 5));
                let claim = (0..len)
                    .filter(|&i| bits(i).unwrap() != 0)
                    .fold(f(0), |sum, i| cfg.add(&sum, &dense[i]));
                for (i, expected) in dense.iter().enumerate() {
                    assert!(
                        compact.evaluation_at(i).unwrap() == *expected,
                        "width={width}, index={i}"
                    );
                }
                for boolean in [false, true] {
                    let point: Vec<_> = (0..vars)
                        .map(|i| {
                            if boolean {
                                f((i % 2) as u64)
                            } else {
                                f(i as u64 + 5)
                            }
                        })
                        .collect();
                    let mut folded = dense.clone();
                    for r in &point {
                        folded = folded
                            .chunks_exact(2)
                            .map(|v| cfg.add(&v[0], &cfg.mul(r, &cfg.sub(&v[1], &v[0]))))
                            .collect();
                    }
                    assert_eq!(compact.evaluate(&point, &cfg).unwrap(), folded[0]);
                }
                for prefix in 0..=4 {
                    let mut expected_t = Blake3Transcript::new();
                    let mut actual_t = Blake3Transcript::new();
                    let expected = prove_source(
                        &mut expected_t,
                        claim.clone(),
                        vars,
                        len,
                        &oracle,
                        &bits,
                        prefix,
                        &cfg,
                    )
                    .unwrap();
                    let actual = prove_source(
                        &mut actual_t,
                        claim.clone(),
                        vars,
                        len,
                        &compact,
                        &bits,
                        prefix,
                        &cfg,
                    )
                    .unwrap();
                    assert!(
                        actual == expected,
                        "prime={prime}, width={width}, prefix={prefix}"
                    );
                    assert_eq!(actual_t.state_digest(), expected_t.state_digest());
                }
            }
        }
    }
}
