//! Degree-two sumcheck over a virtual, disjoint concatenation of bit witnesses.
//!
//! The first seven rounds use byte lookup tables: each round message is a
//! linear functional of the original packed bits. Only after those rounds do
//! we allocate field tables, at one element per packed word of the virtual
//! witness. In particular, no field table of the original bit domain exists.
use super::{BinaryClaim, Error, Gf, opening::Geometry};
use crate::ligerito::transpose_8x8_bits;
use crate::transcript::{Blake3Transcript, traits::Transcript};
use crate::utils::{cfg_chunks_mut, cfg_into_iter};
use flock_core::field::Gf128 as F;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Proof {
    pub rounds: Vec<[F; 2]>,
    pub value: F,
}

pub(crate) fn eq_table(point: &[F]) -> Vec<F> {
    let mut out = vec![F::ZERO; 1 << point.len()];
    out[0] = F::ONE;
    for (i, &r) in point.iter().enumerate() {
        let n = 1usize << i;
        let (lo, hi) = out.split_at_mut(n);
        let step = |(lo_j, hi_j): (&mut F, &mut F)| {
            let high = *lo_j * r;
            *hi_j = high;
            *lo_j += high;
        };
        #[cfg(feature = "parallel")]
        if n >= 1 << 12 {
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .with_min_len(1 << 10)
                .for_each(step);
            continue;
        }
        lo.iter_mut().zip(hi.iter_mut()).for_each(step);
    }
    out
}

pub(super) fn mle(table: &[F], point: &[F]) -> F {
    debug_assert_eq!(table.len(), 1 << point.len());
    table
        .iter()
        .zip(eq_table(point))
        .fold(F::ZERO, |s, (&v, e)| s + v * e)
}

fn eq_eval(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(F::ONE, |p, (&x, &y)| p * (F::ONE + x + y))
}

fn fold(table: &mut Vec<F>, r: F) {
    for i in 0..table.len() / 2 {
        table[i] = table[2 * i] + r * (table[2 * i] + table[2 * i + 1]);
    }
    table.truncate(table.len() / 2);
}

fn observe(t: &mut impl Transcript, values: &[F]) {
    for &v in values {
        t.absorb_slice(&v.lo.to_le_bytes());
        t.absorb_slice(&v.hi.to_le_bytes());
    }
}

fn sample(t: &mut impl Transcript) -> F {
    (t.get_field_challenge::<Gf>(&()))
}

fn evaluate_round([u0, u2]: [F; 2], sum: F, r: F) -> F {
    u0 + r * (sum + u2) + r * r * u2
}

/// LUT for the F2-linear map from a packed word to a field-valued bit sum.
fn byte_table(coefficients: &[F; 128]) -> Vec<F> {
    let mut table = vec![F::ZERO; 16 * 256];
    for byte in 0..16 {
        for val in 1usize..256 {
            table[byte * 256 + val] = table[byte * 256 + (val & (val - 1))]
                + coefficients[byte * 8 + val.trailing_zeros() as usize];
        }
    }
    table
}

fn apply(table: &[F], packed: F) -> F {
    let bytes = ((packed.lo as u128) | ((packed.hi as u128) << 64)).to_le_bytes();
    bytes
        .iter()
        .enumerate()
        .fold(F::ZERO, |s, (i, &b)| s + table[i * 256 + b as usize])
}

/// 16-entry subset-sum table over four elements:
/// `sums[mask] = Σ_{k : bit_k(mask)} e[k]` (15 additions by doubling).
#[inline(always)]
fn subset_sums_4(e: [F; 4]) -> [F; 16] {
    let mut sums = [F::ZERO; 16];
    for (i, &v) in e.iter().enumerate() {
        let half = 1usize << i;
        for k in 0..half {
            sums[half + k] = sums[k] + v;
        }
    }
    sums
}

#[cfg(feature = "parallel")]
fn sum_pairs(iter: impl ParallelIterator<Item = [F; 2]>) -> [F; 2] {
    iter.reduce(|| [F::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
}
#[cfg(not(feature = "parallel"))]
fn sum_pairs(iter: impl Iterator<Item = [F; 2]>) -> [F; 2] {
    iter.fold([F::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
}

/// Bit marginals of one packed source against its high weights:
/// `S[block·128 + bit] = Σ_i high[i] · bit_bit(packed[i·low_blocks + block])`
/// over the source's `high.len()·low_blocks` logical words.
///
/// Every message of the seven packed rounds is a linear functional of
/// these `128·low_blocks` sums (the per-word coefficients change per round,
/// the marginals do not), so they are accumulated ONCE with the
/// method-of-four-Russians fold of [`crate::ligerito::sv_fold_mfr`]: per
/// eight rows two 16-entry subset-sum tables of their `high` weights, then
/// per byte position one 8×8 bit transpose and per output bit two lookups
/// and one accumulate — instead of thirty-two byte-table lookups per word
/// per round. Exact field sums, so the messages are bit-identical to the
/// per-word scan for any accumulation order.
fn bit_marginals(packed: &[F], high: &[F], low_blocks: usize) -> Vec<F> {
    let rows = high.len();
    let packed = &packed[..rows * low_blocks];
    // Tasks are (block range, row range) pairs. Whole block ranges keep a
    // task's accumulators in L1 and need no merge; a source with few
    // blocks (SHA has one) is split over row ranges and merged at the end.
    const BLOCKS_PER_TASK: usize = 16;
    let block_tasks = low_blocks.div_ceil(BLOCKS_PER_TASK);
    #[cfg(feature = "parallel")]
    let wanted = 4 * rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let wanted = 1;
    let row_tasks = if block_tasks >= wanted {
        1
    } else {
        // Row ranges are multiples of eight so only the global tail is
        // scalar, and at least 2^10 rows each.
        (wanted / block_tasks).clamp(1, rows.div_ceil(1 << 10).max(1))
    };
    let rows_per_task = rows.div_ceil(row_tasks).div_ceil(8) * 8;
    let row_tasks = rows.div_ceil(rows_per_task).max(1);
    let partials: Vec<Vec<F>> = cfg_into_iter!(0..block_tasks * row_tasks)
        .map(|task| {
            let b0 = (task / row_tasks) * BLOCKS_PER_TASK;
            let b1 = (b0 + BLOCKS_PER_TASK).min(low_blocks);
            let i0 = (task % row_tasks) * rows_per_task;
            let i1 = (i0 + rows_per_task).min(rows);
            marginals_task(packed, high, low_blocks, b0..b1, i0..i1)
        })
        .collect();
    if row_tasks == 1 {
        return partials.into_iter().flatten().collect();
    }
    let mut out = vec![F::ZERO; low_blocks * 128];
    for (task, part) in partials.iter().enumerate() {
        let b0 = (task / row_tasks) * BLOCKS_PER_TASK;
        for (dst, &v) in out[b0 * 128..].iter_mut().zip(part) {
            *dst += v;
        }
    }
    out
}

fn marginals_task(
    packed: &[F],
    high: &[F],
    low_blocks: usize,
    blocks: std::ops::Range<usize>,
    rows: std::ops::Range<usize>,
) -> Vec<F> {
    let b0 = blocks.start;
    let mut s = vec![F::ZERO; blocks.len() * 128];
    let mut i = rows.start;
    while i + 8 <= rows.end {
        let lo_tbl = subset_sums_4([high[i], high[i + 1], high[i + 2], high[i + 3]]);
        let hi_tbl = subset_sums_4([high[i + 4], high[i + 5], high[i + 6], high[i + 7]]);
        for block in blocks.clone() {
            let acc = &mut s[(block - b0) * 128..(block - b0 + 1) * 128];
            let mut bytes = [[0u8; 16]; 8];
            for (e, slot) in bytes.iter_mut().enumerate() {
                let word = packed[(i + e) * low_blocks + block];
                slot[..8].copy_from_slice(&word.lo.to_le_bytes());
                slot[8..].copy_from_slice(&word.hi.to_le_bytes());
            }
            for r_byte in 0..16 {
                let combined = bytes
                    .iter()
                    .enumerate()
                    .fold(0u64, |c, (e, b)| c | ((b[r_byte] as u64) << (8 * e)));
                let masks = transpose_8x8_bits(combined).to_le_bytes();
                for (p, &mask) in masks.iter().enumerate() {
                    acc[r_byte * 8 + p] +=
                        lo_tbl[(mask & 0x0F) as usize] + hi_tbl[(mask >> 4) as usize];
                }
            }
        }
        i += 8;
    }
    while i < rows.end {
        for block in blocks.clone() {
            let acc = &mut s[(block - b0) * 128..(block - b0 + 1) * 128];
            let word = packed[i * low_blocks + block];
            for (w, half) in [word.lo, word.hi].into_iter().enumerate() {
                let mut bits = half;
                while bits != 0 {
                    acc[(w << 6) | bits.trailing_zeros() as usize] += high[i];
                    bits &= bits - 1;
                }
            }
        }
        i += 1;
    }
    s
}

/// Round `round` (0..7) message of one source from its bit marginals: the
/// `[u0, u2]` a scan of every word would produce, with the per-word linear
/// map replaced by `Σ_{block, bit} c_block[bit]·S[block·128 + bit]`, where
/// `c` carries the bound prefix and the (twice-folded) low weights of the
/// round.
fn packed_round(marginals: &[F], low: &[F], prefix_eq: &[F], round: usize) -> [F; 2] {
    let stride = 128 >> round;
    let low_blocks = low.len() / stride;
    debug_assert_eq!(marginals.len(), low_blocks * 128);
    let mask = (1usize << round) - 1;
    sum_pairs(cfg_into_iter!(0..low_blocks, 8).map(|block| {
        let s = &marginals[block * 128..(block + 1) * 128];
        let mut u0 = F::ZERO;
        let mut u2 = F::ZERO;
        for (bit, &m) in s.iter().enumerate() {
            let j = block * stride + (bit >> (round + 1)) * 2;
            let ps = prefix_eq[bit & mask] * m;
            if bit & (1 << round) == 0 {
                u0 += ps * low[j];
            }
            u2 += ps * (low[j] + low[j + 1]);
        }
        [u0, u2]
    }))
}

/// Working buffers of the dense rounds — the two virtual-witness tables and
/// the pair they fold into — kept by the prepared circuit across proofs.
/// They are 100 MB-class at the larger shapes, and a fresh allocation is
/// page-faulted on first touch on every proof; recycled, only the first
/// proof pays that (the same reason Binius64's prover pools its buffers).
#[derive(Default)]
pub(crate) struct Scratch {
    witness: Vec<F>,
    weights: Vec<F>,
    spare_x: Vec<F>,
    spare_w: Vec<F>,
}

/// Takes `buf` out of the scratch, emptied, with capacity for `n` elements.
fn take_cleared(buf: &mut Vec<F>, n: usize) -> Vec<F> {
    let mut v = std::mem::take(buf);
    v.clear();
    if v.capacity() < n {
        v.reserve_exact(n);
    }
    v
}

/// Fold both dense tables by `r` (`out[i] = in[2i] + r·(in[2i] + in[2i+1])`)
/// and return the next round's message over the folded pairs: one pass
/// over the tables instead of a fold pass and a message pass.
fn dense_fold(
    witness: &mut Vec<F>,
    weights: &mut Vec<F>,
    spare_x: &mut Vec<F>,
    spare_w: &mut Vec<F>,
    r: F,
) -> [F; 2] {
    const CHUNK: usize = 1 << 10;
    let n = witness.len() / 2;
    let (x, w): (&[F], &[F]) = (witness, weights);
    // The folded tables are written once, in parallel, into uninitialised
    // capacity (no serial zero-fill of 2·n elements per round); the spare
    // pair is recycled from the previous round's inputs.
    let mut next_x = take_cleared(spare_x, n);
    let mut next_w = take_cleared(spare_w, n);
    let message = {
        let sx = &mut next_x.spare_capacity_mut()[..n];
        let sw = &mut next_w.spare_capacity_mut()[..n];
        sum_pairs(
            cfg_chunks_mut!(sx, CHUNK)
                .zip(cfg_chunks_mut!(sw, CHUNK))
                .enumerate()
                .map(|(c, (xo, wo))| {
                    let base = c * CHUNK;
                    let mut u0 = F::ZERO;
                    let mut u2 = F::ZERO;
                    let mut k = 0;
                    while k + 1 < xo.len() {
                        let i = 2 * (base + k);
                        let x0 = x[i] + r * (x[i] + x[i + 1]);
                        let x1 = x[i + 2] + r * (x[i + 2] + x[i + 3]);
                        let w0 = w[i] + r * (w[i] + w[i + 1]);
                        let w1 = w[i + 2] + r * (w[i + 2] + w[i + 3]);
                        xo[k].write(x0);
                        xo[k + 1].write(x1);
                        wo[k].write(w0);
                        wo[k + 1].write(w1);
                        u0 += x0 * w0;
                        u2 += (x0 + x1) * (w0 + w1);
                        k += 2;
                    }
                    if k < xo.len() {
                        // Only the final fold (n = 1) has an unpaired element;
                        // its message is never sent.
                        let i = 2 * (base + k);
                        xo[k].write(x[i] + r * (x[i] + x[i + 1]));
                        wo[k].write(w[i] + r * (w[i] + w[i + 1]));
                    }
                    [u0, u2]
                }),
        )
    };
    // SAFETY: every one of the first `n` slots of both buffers was written
    // above (each chunk writes all of its elements).
    unsafe {
        next_x.set_len(n);
        next_w.set_len(n);
    }
    *spare_x = std::mem::replace(witness, next_x);
    *spare_w = std::mem::replace(weights, next_w);
    message
}

fn bind_claims(t: &mut Blake3Transcript, a: &BinaryClaim, b: &BinaryClaim) -> F {
    t.absorb_slice(b"hybrid/joint-bit-sumcheck/v1");
    observe(t, &[a.value, b.value]);
    sample(t)
}

/// Bind the protocol's random combination; the generic prover owns every round.
pub(super) fn inputs<'a>(
    t: &mut Blake3Transcript,
    geometry: &'a Geometry,
    sources: [&'a [F]; 2],
    claims: [&'a BinaryClaim; 2],
    scratch: &'a mut Scratch,
) -> (F, Input<'a>) {
    let rho = bind_claims(t, claims[0], claims[1]);
    (
        claims[0].value + rho * claims[1].value,
        Input {
            geometry,
            sources,
            claims,
            scratch,
            rho,
        },
    )
}

pub struct Input<'a> {
    geometry: &'a Geometry,
    sources: [&'a [F]; 2],
    claims: [&'a BinaryClaim; 2],
    scratch: &'a mut Scratch,
    rho: F,
}
pub struct State<'a> {
    input: Input<'a>,
    low: [Vec<F>; 2],
    high: [Vec<F>; 2],
    marginals: [Vec<F>; 2],
    point: [F; 7],
    round: usize,
    next: [F; 2],
}
pub struct CompressedCodec;
use crate::sumcheck::{SumcheckError, inner::input};
impl input::Codec<field::Gf128Ops> for CompressedCodec {
    fn absorb(_: &field::Gf128Ops, t: &mut impl Transcript, message: &[F; 3]) {
        observe(t, &[message[0], message[2]]);
    }
    fn challenge(_: &field::Gf128Ops, t: &mut impl Transcript) -> Result<F, SumcheckError> {
        Ok(sample(t))
    }
}
impl input::sealed::Input for Input<'_> {}
impl<'a> input::Input<field::Gf128Ops> for Input<'a> {
    type Weights = ();
    type State = State<'a>;
    type Codec = CompressedCodec;
    fn prepare(self, _: &field::Gf128Ops, _: ()) -> Result<Self::State, SumcheckError> {
        let _scope = tracing::info_span!("js:packed_rounds").entered();
        let low = [self.claims[0].low.clone(), self.claims[1].low.clone()];
        let high = [
            eq_table(&self.claims[0].high_point),
            eq_table(&self.claims[1].high_point),
        ];
        let marginals =
            std::array::from_fn(|b| bit_marginals(self.sources[b], &high[b], low[b].len() / 128));
        Ok(State {
            input: self,
            low,
            high,
            marginals,
            point: [F::ZERO; 7],
            round: 0,
            next: [F::ZERO; 2],
        })
    }
}
impl State<'_> {
    fn prepare_dense(&mut self) {
        self.marginals = [Vec::new(), Vec::new()];
        let geometry = self.input.geometry;
        let sources = self.input.sources;
        let scratch = &mut *self.input.scratch;
        let scales = [F::ONE, self.input.rho];
        let low = &self.low;
        let high = &self.high;
        let tables_scope = tracing::info_span!("js:tables").entered();
        let bit_eq: [F; 128] = eq_table(&self.point).try_into().expect("seven coordinates");
        let bit_table = byte_table(&bit_eq);
        const MAX_LANES: usize = 16;
        let lanes = geometry.lanes();
        assert!(lanes <= MAX_LANES && lanes % 2 == 0, "virtual lane group");
        let n = 1usize << geometry.packed_log();
        let logical = [1usize << geometry.logs[0], 1usize << geometry.logs[1]];
        let nlow = [low[0].len(), low[1].len()];
        // One lane group per position, written once in parallel into
        // uninitialised capacity: both branches' lane slices (zero lanes
        // included), then the first dense round's message over the group's
        // pairs (the group size is even, so every pair lies inside one group).
        let mut witness = take_cleared(&mut scratch.witness, n);
        let mut weights = take_cleared(&mut scratch.weights, n);
        let message = {
            let sx = &mut witness.spare_capacity_mut()[..n];
            let sw = &mut weights.spare_capacity_mut()[..n];
            sum_pairs(
                cfg_chunks_mut!(sx, lanes)
                    .zip(cfg_chunks_mut!(sw, lanes))
                    .enumerate()
                    .map(|(g, (wg, ww))| {
                        let mut x = [F::ZERO; MAX_LANES];
                        let mut w = [F::ZERO; MAX_LANES];
                        for lane in 0..lanes {
                            let branch = lane / (lanes / 2);
                            let l = lane % (lanes / 2);
                            let k = geometry.lane_logs[branch];
                            if l < 1 << k {
                                let index = (g << k) | l;
                                x[lane] = apply(&bit_table, sources[branch][index]);
                                if index < logical[branch] {
                                    w[lane] = scales[branch]
                                        * low[branch][index % nlow[branch]]
                                        * high[branch][index / nlow[branch]];
                                }
                            }
                            wg[lane].write(x[lane]);
                            ww[lane].write(w[lane]);
                        }
                        let mut u0 = F::ZERO;
                        let mut u2 = F::ZERO;
                        for j in 0..lanes / 2 {
                            u0 += x[2 * j] * w[2 * j];
                            u2 += (x[2 * j] + x[2 * j + 1]) * (w[2 * j] + w[2 * j + 1]);
                        }
                        [u0, u2]
                    }),
            )
        };
        // SAFETY: every group wrote all of its `lanes` slots in both buffers.
        unsafe {
            witness.set_len(n);
            weights.set_len(n);
        }
        self.low = [Vec::new(), Vec::new()];
        self.high = [Vec::new(), Vec::new()];
        drop(tables_scope);

        scratch.witness = witness;
        scratch.weights = weights;
        self.next = message;
    }
}
impl input::State<field::Gf128Ops> for State<'_> {
    fn num_vars(&self) -> usize {
        self.input.geometry.bit_log()
    }
    fn coefficients(&self, _: &field::Gf128Ops) -> Result<[F; 2], SumcheckError> {
        if self.round >= 7 {
            return Ok(self.next);
        }
        let _scope = tracing::info_span!("js:packed_rounds").entered();
        let eq = eq_table(&self.point[..self.round]);
        let a = packed_round(&self.marginals[0], &self.low[0], &eq, self.round);
        let b = packed_round(&self.marginals[1], &self.low[1], &eq, self.round);
        Ok([a[0] + self.input.rho * b[0], a[1] + self.input.rho * b[1]])
    }
    fn fold(&mut self, _: &field::Gf128Ops, r: &F) -> Result<(), SumcheckError> {
        if self.round < 7 {
            self.point[self.round] = *r;
            for w in &mut self.low {
                fold(w, *r);
            }
            self.round += 1;
            if self.round == 7 {
                self.prepare_dense();
            }
        } else {
            let _scope = tracing::info_span!("js:dense_rounds").entered();
            let s = &mut *self.input.scratch;
            self.next = dense_fold(
                &mut s.witness,
                &mut s.weights,
                &mut s.spare_x,
                &mut s.spare_w,
                *r,
            );
            self.round += 1;
        }
        Ok(())
    }
    fn terminal(&self, _: &field::Gf128Ops) -> Result<[F; 2], SumcheckError> {
        Ok([self.input.scratch.weights[0], self.input.scratch.witness[0]])
    }
}

/// Retain the hybrid proof envelope and bind its terminal witness opening.
pub(super) fn encode(
    t: &mut Blake3Transcript,
    output: crate::sumcheck::inner::InnerSumcheckOutput<F>,
) -> (Proof, Vec<Gf>) {
    let value = output.terminal_evaluations[1];
    observe(t, &[value]);
    (
        Proof {
            rounds: output
                .proof
                .round_polynomials
                .into_iter()
                .map(|[c0, _, c2]| [c0, c2])
                .collect(),
            value,
        },
        output.point,
    )
}

pub(super) fn verify(
    t: &mut Blake3Transcript,
    geometry: &Geometry,
    claims: [&BinaryClaim; 2],
    proof: &Proof,
) -> Result<Vec<Gf>, Error> {
    if proof.rounds.len() != geometry.bit_log() {
        return Err(Error::Invalid("joint sumcheck shape"));
    }
    let rho = bind_claims(t, claims[0], claims[1]);
    let mut value = claims[0].value + rho * claims[1].value;
    let mut point = Vec::with_capacity(proof.rounds.len());
    for &message in &proof.rounds {
        observe(t, &message);
        let r = sample(t);
        value = evaluate_round(message, value, r);
        point.push(r);
    }
    let eval = |branch: usize| {
        let (original, padding) = geometry.project_point(branch, &point);
        let claim = claims[branch];
        let nlow = claim.low.len().trailing_zeros() as usize;
        padding * mle(&claim.low, &original[..nlow]) * eq_eval(&claim.high_point, &original[nlow..])
    };
    if value != (eval(0) + rho * eval(1)) * proof.value {
        return Err(Error::Invalid("joint sumcheck terminal claim"));
    }
    observe(t, &[proof.value]);
    Ok(point.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prove_test(
        t: &mut Blake3Transcript,
        geometry: &Geometry,
        sources: [&[F]; 2],
        claims: [&BinaryClaim; 2],
        scratch: &mut Scratch,
    ) -> (Proof, Vec<Gf>) {
        let (claim, input) = inputs(t, geometry, sources, claims, scratch);
        let out = crate::sumcheck::inner::prove_inner_sumcheck(
            &field::Gf128Ops,
            t,
            claim,
            input,
            (),
            &mut crate::sumcheck::UngrindedRoundBoundary,
        )
        .unwrap();
        encode(t, out)
    }
    #[test]
    fn streamed_rounds_equal_dense_sumcheck_in_both_lane_orders() {
        let mut seed = 0x123456789abcdef0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            F {
                lo: seed,
                hi: seed.rotate_left(29),
            }
        };
        for logs in [[9, 10], [10, 9], [9, 9]] {
            let geometry = Geometry::new(logs).unwrap();
            let packed: [Vec<F>; 2] =
                std::array::from_fn(|b| (0..1 << logs[b]).map(|_| next()).collect());
            let claims: [BinaryClaim; 2] = std::array::from_fn(|b| {
                // Exercise both scheduling paths: many LUT blocks for the
                // first source and one block with a long scan for the second.
                let low_log = if b == 0 { 11 } else { 7 };
                let low: Vec<F> = (0..1 << low_log).map(|_| next()).collect();
                let high_point: Vec<F> = (0..logs[b] + 7 - low_log).map(|_| next()).collect();
                let high = eq_table(&high_point);
                let mut value = F::ZERO;
                for (i, word) in packed[b].iter().enumerate() {
                    let bits = word.lo as u128 | ((word.hi as u128) << 64);
                    for bit in 0..128 {
                        if bits & (1 << bit) != 0 {
                            let index = i * 128 + bit;
                            value += low[index & ((1 << low_log) - 1)] * high[index >> low_log];
                        }
                    }
                }
                BinaryClaim {
                    low,
                    high_point,
                    value,
                }
            });
            let mut actual_t = Blake3Transcript::new();
            let (actual, point) = prove_test(
                &mut actual_t,
                &geometry,
                [&packed[0], &packed[1]],
                [&claims[0], &claims[1]],
                &mut Scratch::default(),
            );
            let mut reference_t = Blake3Transcript::new();
            let rho = bind_claims(&mut reference_t, &claims[0], &claims[1]);
            let mut v = vec![F::ZERO; 1 << geometry.bit_log()];
            let mut w = v.clone();
            for b in 0..2 {
                let high = eq_table(&claims[b].high_point);
                for (i, word) in packed[b].iter().enumerate() {
                    let bits = word.lo as u128 | ((word.hi as u128) << 64);
                    for bit in 0..128 {
                        let index = i * 128 + bit;
                        let dst = geometry.embed(b, i) * 128 + bit;
                        v[dst] = if bits & (1 << bit) != 0 {
                            F::ONE
                        } else {
                            F::ZERO
                        };
                        w[dst] = claims[b].low[index % claims[b].low.len()]
                            * high[index / claims[b].low.len()]
                            * if b == 0 { F::ONE } else { rho };
                    }
                }
            }
            let mut rounds = Vec::new();
            let mut reference_point = Vec::new();
            while v.len() > 1 {
                let message = (0..v.len() / 2).fold([F::ZERO; 2], |mut sums, i| {
                    sums[0] += v[2 * i] * w[2 * i];
                    sums[1] += (v[2 * i] + v[2 * i + 1]) * (w[2 * i] + w[2 * i + 1]);
                    sums
                });
                observe(&mut reference_t, &message);
                let r = sample(&mut reference_t);
                rounds.push(message);
                reference_point.push((r));
                fold(&mut v, r);
                fold(&mut w, r);
            }
            observe(&mut reference_t, &v);
            assert_eq!(
                actual,
                Proof {
                    rounds,
                    value: v[0]
                }
            );
            assert_eq!(point, reference_point);
            assert_eq!(sample(&mut actual_t), sample(&mut reference_t));
            let verified = verify(
                &mut Blake3Transcript::new(),
                &geometry,
                [&claims[0], &claims[1]],
                &actual,
            )
            .unwrap();
            assert_eq!(point, verified);
        }
    }
}
