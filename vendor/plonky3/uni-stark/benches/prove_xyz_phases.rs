//! Phase-by-phase cost breakdown of the uni-STARK prover.
//!
//! The statement proved is the simplest non-trivial one available: a trace of
//! three BabyBear columns `x, y, z` and the single constraint `x_i * y_i = z_i`
//! on every row. There are no boundary constraints, no transition constraints
//! and no next-row accesses, so the AIR itself is as close to free as it gets.
//! Whatever time the prover spends is therefore spent on the STARK machinery —
//! low-degree extension, Merkle hashing, quotient evaluation and FRI — which is
//! exactly what we want to attribute.
//!
//! Because the constraint has degree 2, the quotient polynomial fits in a single
//! chunk over a domain the size of the (blown-up) trace domain.
//!
//! Timings come from the `tracing` spans that Plonky3 already carries. A small
//! subscriber layer records the wall-clock lifetime of every span, keyed by its
//! full path, and the result is printed as a tree with each phase's share of
//! total proving time.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p p3-uni-stark --features parallel --bench prove_xyz_phases
//! ```
//!
//! Options (pass after `--`):
//! - `--log-heights 16,18,20` — trace heights to sweep, as log2 of the row count.
//! - `--reps 3` — proofs per height; the median run is the one broken down.
//! - `--max-depth 6` — how deep into the span tree to report.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use tracing::Subscriber;
use tracing::span::{Attributes, Id};
use tracing_subscriber::Registry;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

/// Column layout: `[x, y, z]`.
const TRACE_WIDTH: usize = 3;

/// The four spans that partition the bulk of `prove`, in the order the prover
/// reaches them. Reported separately as a headline summary.
const HEADLINE_PHASES: [&str; 4] = [
    "commit to trace data",
    "quotient_values",
    "commit to quotient poly chunks",
    "open",
];

/// Kinds of work that recur in several phases, so the tree splits them up.
/// Summed by span name wherever it occurs. These cut across the phase
/// breakdown rather than partitioning it.
const CROSS_CUTTING: [(&str, &str); 4] = [
    ("build merkle tree", "Merkle hashing (Poseidon2)"),
    (
        "coset_lde_batch_with_transform",
        "low-degree extension (DFT)",
    ),
    ("quotient_values", "constraint evaluation"),
    ("batch_multiplicative_inverse", "batch inversion"),
];

// -----------------------------------------------------------------------------
// The AIR
// -----------------------------------------------------------------------------

/// One multiplication per row: `x * y = z`, with `x`, `y`, `z` unconstrained
/// field elements.
///
/// Note that this proves a *field* multiplication. Nothing here range-checks the
/// inputs, so `z` is only pinned modulo the BabyBear prime; see the
/// `prove_mul_int` bench for what it costs to pin an integer product instead.
struct MulAir;

impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        TRACE_WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let (x, y, z) = (local[0], local[1], local[2]);
        builder.assert_eq(z, x.into() * y.into());
    }
}

/// Build a satisfying trace of `2^log_height` rows.
fn build_trace(rng: &mut SmallRng, log_height: usize) -> RowMajorMatrix<BabyBear> {
    let height = 1 << log_height;
    let mut values = BabyBear::zero_vec(height * TRACE_WIDTH);
    for row in values.chunks_exact_mut(TRACE_WIDTH) {
        let x: BabyBear = rng.random();
        let y: BabyBear = rng.random();
        row[0] = x;
        row[1] = y;
        row[2] = x * y;
    }
    RowMajorMatrix::new(values, TRACE_WIDTH)
}

// -----------------------------------------------------------------------------
// Span timing collection
// -----------------------------------------------------------------------------

/// Wall-clock state attached to each live span.
struct SpanTiming {
    start: Instant,
    /// Span names from the root down to and including this span. Captured at
    /// creation time, while every ancestor is guaranteed to still be in the
    /// registry.
    path: Vec<&'static str>,
}

/// Accumulated span durations, keyed by full path so that (say) the Merkle tree
/// built for the trace stays distinct from the one built for the quotient.
#[derive(Clone, Default)]
struct PhaseTotals {
    /// Paths in creation order, which for a sequential prover is a pre-order
    /// walk of the span tree.
    order: Vec<Vec<&'static str>>,
    totals: HashMap<Vec<&'static str>, (Duration, usize)>,
}

impl PhaseTotals {
    /// Register a path the first time its span is created, fixing tree order.
    fn register(&mut self, path: &[&'static str]) {
        if !self.totals.contains_key(path) {
            self.order.push(path.to_vec());
            self.totals.insert(path.to_vec(), (Duration::ZERO, 0));
        }
    }

    fn record(&mut self, path: &[&'static str], elapsed: Duration) {
        let entry = self
            .totals
            .get_mut(path)
            .expect("path registered at span creation");
        entry.0 += elapsed;
        entry.1 += 1;
    }

    fn get(&self, path: &[&'static str]) -> (Duration, usize) {
        self.totals.get(path).copied().unwrap_or_default()
    }

    /// Total time over every recorded span whose *own* name matches `name`,
    /// wherever it appears in the tree.
    fn total_by_name(&self, name: &str) -> Duration {
        self.order
            .iter()
            .filter(|path| path.last().is_some_and(|last| *last == name))
            .map(|path| self.get(path).0)
            .sum()
    }

    fn reset(&mut self) {
        self.order.clear();
        self.totals.clear();
    }
}

/// A `tracing` layer that times every span and files the result under its path.
#[derive(Clone, Default)]
struct PhaseRecorder(Arc<Mutex<PhaseTotals>>);

impl<S> Layer<S> for PhaseRecorder
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let path: Vec<&'static str> = span.scope().from_root().map(|s| s.name()).collect();
        self.0.lock().unwrap().register(&path);
        span.extensions_mut().insert(SpanTiming {
            start: Instant::now(),
            path,
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let Some(timing) = span.extensions_mut().remove::<SpanTiming>() else {
            return;
        };
        self.0
            .lock()
            .unwrap()
            .record(&timing.path, timing.start.elapsed());
    }
}

// -----------------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------------

fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.3} s")
    } else if secs >= 1e-3 {
        format!("{:.2} ms", secs * 1e3)
    } else {
        format!("{:.1} µs", secs * 1e6)
    }
}

/// Print the span tree, one line per path, with each node's share of `root`.
fn print_tree(totals: &PhaseTotals, max_depth: usize) {
    // Children of each path, in creation order.
    let mut children: HashMap<&[&'static str], Vec<usize>> = HashMap::new();
    let mut roots: Vec<usize> = Vec::new();
    for (idx, path) in totals.order.iter().enumerate() {
        let parent = &path[..path.len() - 1];
        if parent.is_empty() || !totals.totals.contains_key(parent) {
            roots.push(idx);
        } else {
            children.entry(parent).or_default().push(idx);
        }
    }

    let root_time = roots
        .first()
        .map_or(Duration::ZERO, |&idx| totals.get(&totals.order[idx]).0);

    println!(
        "  {:<52}{:>12}{:>9}{:>8}",
        "phase", "time", "% total", "calls"
    );
    println!("  (self) = time in a phase's own body, outside any nested span");

    let mut stack: Vec<(usize, String, bool)> = roots
        .iter()
        .rev()
        .enumerate()
        .map(|(rev_i, &idx)| (idx, String::new(), rev_i == 0))
        .collect();

    while let Some((idx, prefix, is_last)) = stack.pop() {
        let path = &totals.order[idx];
        let depth = path.len() - 1;
        let (time, calls) = totals.get(path);

        let (branch, child_prefix) = if depth == 0 {
            (String::new(), String::new())
        } else if is_last {
            (format!("{prefix}└─ "), format!("{prefix}   "))
        } else {
            (format!("{prefix}├─ "), format!("{prefix}│  "))
        };

        let share = if root_time.is_zero() {
            0.0
        } else {
            100.0 * time.as_secs_f64() / root_time.as_secs_f64()
        };
        let label = format!("{branch}{}", path.last().unwrap());
        println!(
            "  {label:<52}{:>12}{share:>8.1}%{calls:>8}",
            fmt_duration(time)
        );

        if depth + 1 > max_depth {
            continue;
        }
        let Some(kids) = children.get(path.as_slice()) else {
            continue;
        };

        // Time inside this span that none of its child spans account for: work
        // done directly in this phase's own body. Listed first so the `└─` glyph
        // still lands on the genuinely last child.
        let child_total: Duration = kids.iter().map(|&k| totals.get(&totals.order[k]).0).sum();
        let unaccounted = time.saturating_sub(child_total);
        if unaccounted.as_secs_f64() > 0.01 * root_time.as_secs_f64() {
            let share = 100.0 * unaccounted.as_secs_f64() / root_time.as_secs_f64();
            let label = format!("{child_prefix}├─ (self)");
            println!(
                "  {label:<52}{:>12}{share:>8.1}%{:>8}",
                fmt_duration(unaccounted),
                ""
            );
        }

        for (rev_i, &kid) in kids.iter().rev().enumerate() {
            stack.push((kid, child_prefix.clone(), rev_i == 0));
        }
    }
}

// -----------------------------------------------------------------------------
// Driver
// -----------------------------------------------------------------------------

/// Read `--flag value` from the command line, ignoring anything unrecognised
/// (`cargo bench` passes `--bench` through to us).
fn arg(flag: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
        if let Some(rest) = a.strip_prefix(&format!("{flag}=")) {
            return Some(rest.to_owned());
        }
    }
    None
}

#[allow(clippy::too_many_lines)]
fn main() {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug assertions are enabled. `prove` re-checks every constraint \
             row by row, so these numbers are meaningless. Use `cargo bench` or \
             `--release`."
        );
    }

    let log_heights: Vec<usize> = arg("--log-heights")
        .unwrap_or_else(|| "16,18,20".to_owned())
        .split(',')
        .map(|s| s.trim().parse().expect("--log-heights wants integers"))
        .collect();
    let reps: usize = arg("--reps").map_or(3, |s| s.parse().expect("--reps wants an integer"));
    let max_depth: usize =
        arg("--max-depth").map_or(6, |s| s.parse().expect("--max-depth wants an integer"));
    assert!(reps > 0, "--reps must be positive");

    let recorder = PhaseRecorder::default();
    Registry::default()
        .with(LevelFilter::DEBUG)
        .with(recorder.clone())
        .init();

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValPacking = <Val as Field>::Packing;
    type ValMmcs = MerkleTreeMmcs<ValPacking, ValPacking, MyHash, MyCompress, 2, 8>;
    let val_mmcs = ValMmcs::new(hash, compress, 0);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    let challenger = Challenger::new(perm);

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);
    let (log_blowup, num_queries) = (fri_params.log_blowup, fri_params.num_queries);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = MulAir;

    println!(
        "AIR: {TRACE_WIDTH} BabyBear columns (x, y, z), one constraint x*y = z, degree 2\n\
         PCS: TwoAdicFriPcs, blowup 2^{log_blowup}, {num_queries} queries, \
         Poseidon2 (width 16), challenges in a degree-4 extension\n\
         Threads: {}\n",
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
    );

    for log_height in log_heights {
        let trace = build_trace(&mut rng, log_height);

        // Correctness check and proof size, outside any timed region.
        let proof = prove(&config, &air, trace.clone(), &[]);
        verify(&config, &air, &proof, &[]).expect("proof must verify");
        let proof_bytes = postcard::to_allocvec(&proof).expect("unable to serialize proof");
        recorder.0.lock().unwrap().reset();

        // One proof per rep, each with a clean set of span totals.
        let mut runs: Vec<(Duration, PhaseTotals)> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let trace = trace.clone();
            let start = Instant::now();
            let proof = prove(&config, &air, trace, &[]);
            let elapsed = start.elapsed();
            drop(proof);
            let mut totals = recorder.0.lock().unwrap();
            runs.push((elapsed, totals.clone()));
            totals.reset();
        }
        runs.sort_by_key(|(elapsed, _)| *elapsed);
        let (median_time, median_totals) = &runs[reps / 2];

        let verify_start = Instant::now();
        for _ in 0..reps {
            verify(&config, &air, &proof, &[]).expect("proof must verify");
        }
        let verify_time = verify_start.elapsed() / reps as u32;
        recorder.0.lock().unwrap().reset();

        let rows = 1u64 << log_height;
        println!("{}", "=".repeat(85));
        println!(
            "2^{log_height} rows ({rows} multiplications), trace {} MiB",
            (rows as usize * TRACE_WIDTH * size_of::<Val>()) >> 20,
        );
        println!(
            "  prove   {} (median of {reps}; min {}, max {})",
            fmt_duration(*median_time),
            fmt_duration(runs[0].0),
            fmt_duration(runs[reps - 1].0),
        );
        println!(
            "  verify  {}     proof {} KiB     {:.2} µs / multiplication",
            fmt_duration(verify_time),
            proof_bytes.len() >> 10,
            median_time.as_secs_f64() * 1e6 / rows as f64,
        );

        // Committed shapes, read back off the proof rather than re-derived.
        // The quotient lives in the degree-4 challenge extension, so it flattens
        // to 4 base-field columns; `trace_local` has one entry per trace column.
        let trace_cols = proof.opened_values.trace_local.len();
        let num_chunks = proof.opened_values.quotient_chunks.len();
        let chunk_cols = proof.opened_values.quotient_chunks[0].len();
        let quotient_cols = num_chunks * chunk_cols;
        println!(
            "  committed: trace {trace_cols} cols x 2^{log_height}, \
             quotient {num_chunks} chunk x {chunk_cols} cols x 2^{} \
             => quotient is {:.2}x the trace",
            log_height - num_chunks.trailing_zeros() as usize,
            quotient_cols as f64 / trace_cols as f64,
        );

        // Headline: the four spans that partition the prover.
        let root = median_totals
            .order
            .first()
            .map_or(Duration::ZERO, |p| median_totals.get(p).0);
        println!("\n  Headline phases:");
        let mut covered = Duration::ZERO;
        for name in HEADLINE_PHASES {
            let time = median_totals.total_by_name(name);
            covered += time;
            let share = if root.is_zero() {
                0.0
            } else {
                100.0 * time.as_secs_f64() / root.as_secs_f64()
            };
            println!("    {name:<48}{:>12}{share:>8.1}%", fmt_duration(time));
        }
        let rest = root.saturating_sub(covered);
        let share = if root.is_zero() {
            0.0
        } else {
            100.0 * rest.as_secs_f64() / root.as_secs_f64()
        };
        println!(
            "    {:<48}{:>12}{share:>8.1}%",
            "everything else",
            fmt_duration(rest)
        );

        // Cross-cutting: the same kind of work shows up under several phases.
        println!("\n  By kind of work (overlaps the phases above):");
        for (span, label) in CROSS_CUTTING {
            let time = median_totals.total_by_name(span);
            let share = if root.is_zero() {
                0.0
            } else {
                100.0 * time.as_secs_f64() / root.as_secs_f64()
            };
            println!("    {label:<48}{:>12}{share:>8.1}%", fmt_duration(time));
        }

        println!("\n  Full span tree:");
        print_tree(median_totals, max_depth);
        println!();
    }
}
