//! Matched, standalone linear-map qualification. Run latency and allocation
//! measurements separately; `bench-memory` instruments allocation only.
use circuit::linear_map::circuit::{WengertGenerator, WengertTape};
use circuit::linear_map::{BilinearEval, ColumnValues, LeftMul};
use field::{FpCtx, ModRingCtx, RingOps, Uint};
use std::{hint::black_box, time::Instant};

#[cfg(feature = "bench-memory")]
#[path = "support/allocations.rs"]
mod allocations;
#[cfg(feature = "bench-memory")]
#[global_allocator]
static ALLOC: allocations::Counted = allocations::Counted;

fn build(p256: bool) -> WengertTape {
    if p256 {
        use circuit::p256::{VERIFY_DIGEST_INPUT_BITS, verify_digest_circuit};
        let mut b = WengertGenerator::new(VERIFY_DIGEST_INPUT_BITS);
        let inputs = b.take_boxed_inputs();
        verify_digest_circuit(&mut b, &inputs);
        b.finish()
    } else {
        use circuit::sha256::{SHA256_2KB_MESSAGE_BITS, sha256_2kb_circuit};
        let mut b = WengertGenerator::new(SHA256_2KB_MESSAGE_BITS);
        let inputs = b.take_boxed_inputs();
        let _ = sha256_2kb_circuit(&mut b, &inputs);
        b.finish()
    }
}
// Match the production split-equality geometric column functional, including
// an unaligned P-256 tail offset. Its preparation is outside traversal timing.
struct Columns {
    field: FpCtx<2>,
    low: Vec<field::Fp<2>>,
    high: Vec<field::Fp<2>>,
    suffix: Vec<field::Fp<2>>,
    powers: Vec<field::Fp<2>>,
    offset: usize,
    count: usize,
}
impl Columns {
    fn new(field: FpCtx<2>, count: usize, offset: usize) -> Self {
        let vars = (count + offset).next_power_of_two().ilog2() as usize;
        let point = (0..vars)
            .map(|i| field::IntegerEmbedding::from_integer(&field, &(i as u64 + 3)))
            .collect::<Vec<_>>();
        let table = |point: &[field::Fp<2>]| {
            let mut out = field.zero_vec(1 << point.len());
            out[0] = field.one();
            for (k, r) in point.iter().enumerate() {
                for i in 0..1 << k {
                    let right = field.mul(&out[i], r);
                    out[i] = field.sub(&out[i], &right);
                    out[i + (1 << k)] = right;
                }
            }
            out
        };
        let low = table(&point[..vars / 2]);
        let high = table(&point[vars / 2..]);
        let mut suffix = field.zero_vec(low.len() + 1);
        let mut powers = vec![field.one(); low.len() + 1];
        for i in (0..low.len()).rev() {
            suffix[i] = field.add(&low[i], &field.add(&suffix[i + 1], &suffix[i + 1]));
        }
        for i in 0..low.len() {
            powers[i + 1] = field.add(&powers[i], &powers[i]);
        }
        Self {
            field,
            low,
            high,
            suffix,
            powers,
            offset,
            count,
        }
    }
}
impl ColumnValues<field::Fp<2>> for Columns {
    fn len(&self) -> usize {
        self.count
    }
    fn scalar(&self, column: usize) -> field::Fp<2> {
        let i = self.offset + column;
        self.field.mul(
            &self.low[i % self.low.len()],
            &self.high[i / self.low.len()],
        )
    }
    fn power_sum(&self, first: usize, len: usize) -> field::Fp<2> {
        let mut i = self.offset + first;
        let end = i + len;
        let mut sum = self.field.zero();
        let mut base = self.field.one();
        while i < end {
            let high = i / self.low.len();
            let lo = i % self.low.len();
            let hi = (lo + end - i).min(self.low.len());
            let part = self.field.sub(
                &self.suffix[lo],
                &self.field.mul(&self.powers[hi - lo], &self.suffix[hi]),
            );
            sum = self.field.add(
                &sum,
                &self
                    .field
                    .mul(&self.field.mul(&base, &part), &self.high[high]),
            );
            base = self.field.mul(&base, &self.powers[hi - lo]);
            i += hi - lo;
        }
        sum
    }
}
fn measure(name: &str, samples: usize, mut f: impl FnMut()) {
    if std::env::var("PHASE").is_ok_and(|phase| phase != name) {
        return;
    }
    for _ in 0..5 {
        f();
    }
    #[cfg(not(feature = "bench-memory"))]
    {
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = Instant::now();
            f();
            times.push(start.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{name},{:.3},{:.3},{:.3}",
            times[samples / 2],
            times[samples / 10],
            times[samples * 9 / 10]
        );
    }
    #[cfg(feature = "bench-memory")]
    {
        let _ = samples;
        let _ = Instant::now();
        let (count, bytes, peak) = allocations::measure(f);
        println!("{name},{count},{bytes},{peak}");
    }
}
fn main() {
    let p256 = std::env::args().any(|s| s == "p256");
    let samples = std::env::var("SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40usize);
    if std::env::args().any(|s| s == "construct") {
        measure("construct", samples, || {
            black_box(build(p256));
        });
        return;
    }
    let tape = build(p256);
    let field = field::create_prime_field(Uint::from((1u128 << 127) - 1));
    let modulus = ModRingCtx::new(*field.modulus()).unwrap();
    let encoder = tape.prepare(&modulus).unwrap();
    let weights = (0..tape.row_count())
        .map(|i| {
            [
                encoder.to_montgomery([i as u64 + 3, 0]),
                encoder.to_montgomery([i as u64 + 7, 0]),
                encoder.to_montgomery([i as u64 + 11, 0]),
            ]
        })
        .collect::<Vec<_>>();
    let columns = Columns::new(
        field,
        tape.column_count(),
        if p256 { 20_457 * 8 } else { 0 },
    );
    drop(encoder);
    let weights: Vec<_> = weights
        .into_iter()
        .flatten()
        .map(|words| {
            columns
                .field
                .from_montgomery_integer(Uint::from_words(words))
        })
        .collect();
    let mut output = columns.field.zero_vec(tape.column_count());
    let mut adjoint = tape.prepare(&modulus).unwrap();
    let mut forward = tape.prepare(&modulus).unwrap();
    adjoint.mul_left_into(&weights, &mut output).unwrap();
    let dot = output
        .iter()
        .enumerate()
        .fold(columns.field.zero(), |sum, (j, value)| {
            columns
                .field
                .add(&sum, &columns.field.mul(value, &columns.scalar(j)))
        });
    assert_eq!(forward.evaluate_bilinear(&weights, &columns).unwrap(), dot);
    println!(
        "phase,{},{}",
        if cfg!(feature = "bench-memory") {
            "allocations,allocated_bytes"
        } else {
            "median_us,p10_us"
        },
        if cfg!(feature = "bench-memory") {
            "extra_peak_bytes"
        } else {
            "p90_us"
        }
    );
    measure("prepare", samples, || {
        black_box(tape.prepare(black_box(&modulus)).unwrap());
    });
    measure("bind_reused", samples, || {
        adjoint
            .mul_left_into(black_box(&weights), black_box(&mut output))
            .unwrap();
        black_box(&output);
    });
    measure("terminal_reused", samples, || {
        black_box(
            forward
                .evaluate_bilinear(black_box(&weights), black_box(&columns))
                .unwrap(),
        );
    });
    measure("prepare_bind", samples, || {
        let mut p = tape.prepare(black_box(&modulus)).unwrap();
        let mut output = columns.field.zero_vec(tape.column_count());
        p.mul_left_into(black_box(&weights), &mut output).unwrap();
        black_box(output);
    });
    eprintln!(
        "topology_bytes={},adjoint_workspace_bytes={},forward_workspace_bytes={}",
        tape.payload_bytes(),
        adjoint.workspace_bytes(),
        forward.workspace_bytes()
    );
}
