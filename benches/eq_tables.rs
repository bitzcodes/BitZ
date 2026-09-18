//! Controlled A/B benchmark for the equality-table builders optimized in PR 1.
//!
//! This is a plain `harness = false` benchmark. The `legacy_*` functions are
//! intentionally local copies of the pre-refactor algorithms: suffix tensors
//! allocate and clone every level and multiply both children, while the `Q100Element`
//! builder allocates a fresh table and multiplies both halves at every step.
//! Each current builder is checked entry-for-entry before any measurements.

mod common;

use std::hint::black_box;

use bitz::{
    Gf128,
    pcs::{FQ_MOD, Q100Element, eq_le_table_fq},
    piop::sumcheck::eq_factored::suffix_tensor_arena_for_bench,
};

const WIDTHS: [usize; 3] = [8, 12, 16];
const DEFAULT_SAMPLES: usize = 21;
const SUFFIX_GATE_SPEEDUP: f64 = 1.2;
const FQ_GATE_SPEEDUP: f64 = 1.5;

#[derive(Clone, Copy)]
struct PairMedians {
    legacy_ns: u128,
    current_ns: u128,
}

impl PairMedians {
    fn speedup(self) -> f64 {
        self.legacy_ns as f64 / self.current_ns.max(1) as f64
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn deterministic_gf128_point(width: usize) -> Vec<Gf128> {
    let mut state = 0x6571_5f73_7566_6669u64 ^ width as u64;
    (0..width)
        .map(|_| Gf128::from_polynomial_words([splitmix64(&mut state), splitmix64(&mut state)]))
        .collect()
}

fn deterministic_fq_point(width: usize) -> Vec<Q100Element> {
    let mut state = 0x6571_5f66_715f_6c65u64 ^ width as u64;
    (0..width)
        .map(|_| {
            let value =
                u128::from(splitmix64(&mut state)) | (u128::from(splitmix64(&mut state)) << 64);
            Q100Element::from_u128(value % FQ_MOD)
        })
        .collect()
}

/// The old suffix builder, preserved verbatim in shape for controlled A/B
/// timing. Returned levels are in round order `[V_1, ..., V_k]`.
fn legacy_suffix_tensors(q: &[Gf128]) -> Vec<Vec<Gf128>> {
    let one = Gf128::one();
    let zero = Gf128::zero();
    let mut suffix = Vec::with_capacity(q.len());
    let mut current = vec![one];
    suffix.push(current.clone());
    for round in (1..q.len()).rev() {
        let one_factor = q[round];
        let zero_factor = one - one_factor;
        let mut next = vec![zero; current.len() * 2];
        next.chunks_mut(2)
            .zip(current.iter())
            .for_each(|(children, parent)| {
                children[0] = *parent * zero_factor;
                children[1] = *parent * one_factor;
            });
        suffix.push(next.clone());
        current = next;
    }
    suffix.reverse();
    suffix
}

/// Converts round-ordered legacy levels to the arena's physical
/// `[V_k, ..., V_1]` representation.
fn flatten_legacy_suffix(levels: &[Vec<Gf128>]) -> (Vec<Gf128>, Vec<usize>) {
    let total_len = levels.iter().map(Vec::len).sum();
    let mut values = Vec::with_capacity(total_len);
    let mut offsets = vec![0usize; levels.len()];
    for round in (0..levels.len()).rev() {
        offsets[round] = values.len();
        values.extend_from_slice(&levels[round]);
    }
    (values, offsets)
}

fn arena_tensor<'a>(values: &'a [Gf128], offsets: &[usize], round: usize) -> &'a [Gf128] {
    let start = offsets[round];
    let end = if round == 0 {
        values.len()
    } else {
        offsets[round - 1]
    };
    &values[start..end]
}

fn verify_suffix_builder(point: &[Gf128]) {
    let legacy_levels = legacy_suffix_tensors(point);
    let (expected_values, expected_offsets) = flatten_legacy_suffix(&legacy_levels);
    let (values, offsets) = suffix_tensor_arena_for_bench(point, &());

    assert_eq!(offsets, expected_offsets, "suffix arena offsets differ");
    assert_eq!(values, expected_values, "flattened suffix arena differs");
    assert_eq!(legacy_levels.len(), offsets.len());
    for (round, expected) in legacy_levels.iter().enumerate() {
        assert_eq!(arena_tensor(&values, &offsets, round), expected);
    }
}

/// The old little-endian `Q100Element` table builder: two child multiplications and a
/// fresh allocation at every coordinate.
fn legacy_eq_le_table_fq(point: &[Q100Element]) -> Vec<Q100Element> {
    let one = Q100Element::from_u128(1);
    let mut table = vec![one];
    for challenge in point {
        let zero_factor = one - *challenge;
        let mut next = Vec::with_capacity(table.len() * 2);
        for &parent in &table {
            next.push(parent * zero_factor);
        }
        for &parent in &table {
            next.push(parent * *challenge);
        }
        table = next;
    }
    table
}

fn median(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Measures construction only. The result is black-boxed and dropped after
/// the timestamp, so differences in nested-container destruction are not
/// accidentally counted as builder time.
fn timed_ns<R>(body: &mut impl FnMut() -> R) -> u128 {
    let (result, started) =
        bitz::observability::measure(tracing::info_span!("eq_tables:result"), || body())
            .expect("measure completed operation");
    let elapsed = started.as_nanos();
    black_box(result);
    elapsed
}

/// One warm-up per implementation, then alternating execution order on every
/// sample so neither implementation consistently receives the colder slot.
fn alternating_medians<LegacyResult, CurrentResult>(
    samples: usize,
    mut legacy: impl FnMut() -> LegacyResult,
    mut current: impl FnMut() -> CurrentResult,
) -> PairMedians {
    black_box(legacy());
    black_box(current());

    let mut legacy_samples = Vec::with_capacity(samples);
    let mut current_samples = Vec::with_capacity(samples);
    for sample in 0..samples {
        if sample & 1 == 0 {
            legacy_samples.push(timed_ns(&mut legacy));
            current_samples.push(timed_ns(&mut current));
        } else {
            current_samples.push(timed_ns(&mut current));
            legacy_samples.push(timed_ns(&mut legacy));
        }
    }
    PairMedians {
        legacy_ns: median(legacy_samples),
        current_ns: median(current_samples),
    }
}

fn print_result(
    builder: &str,
    width: usize,
    entries: usize,
    medians: PairMedians,
    threshold: f64,
    samples: usize,
) -> bool {
    let speedup = medians.speedup();
    let meets_threshold = speedup >= threshold;
    let gate_required = width >= 12;
    let gate_pass = !gate_required || meets_threshold;
    println!(
        "RESULT builder={builder} width={width} entries={entries} samples={samples} \
         legacy_median_us={:.3} current_median_us={:.3} speedup_x={speedup:.3} \
         threshold_x={threshold:.1} meets_threshold={meets_threshold} \
         gate_required={gate_required} gate_pass={gate_pass}",
        medians.legacy_ns as f64 / 1_000.0,
        medians.current_ns as f64 / 1_000.0,
    );
    gate_pass
}

fn benchmark_suffix(samples: usize) -> bool {
    let mut gate_pass = true;
    for width in WIDTHS {
        let point = deterministic_gf128_point(width);
        verify_suffix_builder(&point);
        let medians = alternating_medians(
            samples,
            || legacy_suffix_tensors(black_box(&point)),
            || suffix_tensor_arena_for_bench(black_box(&point), &()),
        );
        let entries = (1usize << width) - 1;
        gate_pass &= print_result(
            "suffix_arena",
            width,
            entries,
            medians,
            SUFFIX_GATE_SPEEDUP,
            samples,
        );
    }
    gate_pass
}

fn benchmark_fq(samples: usize) -> bool {
    let mut gate_pass = true;
    for width in WIDTHS {
        let point = deterministic_fq_point(width);
        assert_eq!(
            eq_le_table_fq(&point),
            legacy_eq_le_table_fq(&point),
            "Q100Element equality tables differ at width {width}",
        );
        let medians = alternating_medians(
            samples,
            || legacy_eq_le_table_fq(black_box(&point)),
            || eq_le_table_fq(black_box(&point)),
        );
        gate_pass &= print_result(
            "fq_table",
            width,
            1usize << width,
            medians,
            FQ_GATE_SPEEDUP,
            samples,
        );
    }
    gate_pass
}

fn main() {
    common::cli::EnvironmentCli::parse();
    let samples = common::cli::env::<usize>("BITZ_EQ_TABLE_SAMPLES")
        .unwrap_or(DEFAULT_SAMPLES)
        .max(DEFAULT_SAMPLES);
    bitz::observability::install().expect("install Perfetto subscriber");
    common::enforce_known_env();
    let _ = flock_core::init_perf_thread_pool();
    println!(
        "PR1 equality-table benchmark; widths={WIDTHS:?}; warmups=1; \
         alternating_samples={samples}"
    );

    let suffix_gate_pass = benchmark_suffix(samples);
    let fq_gate_pass = benchmark_fq(samples);
    let overall_gate_pass = suffix_gate_pass && fq_gate_pass;
    println!(
        "GATE suffix_widths_12_16_pass={suffix_gate_pass} \
         fq_widths_12_16_pass={fq_gate_pass} overall_pass={overall_gate_pass}"
    );
    assert!(
        overall_gate_pass,
        "PR 1 equality-table performance gate failed"
    );
}
