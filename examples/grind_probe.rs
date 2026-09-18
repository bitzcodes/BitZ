//! Grinding-cost probe (paper `tab:grinding-cost`): the time to find the
//! smallest `u64` nonce with `BLAKE3(seed ‖ nonce_le)` having at least `α`
//! leading zero bits, for a 256-bit seed, through the crate's own grinder
//! (`piop::spartan::grinding::find_grinding_nonce`: the eight-lane NEON
//! BLAKE3 kernel, chunked smallest-nonce parallel scan). One search per
//! random seed; the same seeds are reused for every thread count, so the
//! per-seed nonce (and attempt count) is identical across thread counts.
//! Prints mean/median wall time per `α` and thread count, plus the mean
//! attempt count (≈ 2^α) and the single-thread cost per attempt.
//!
//! ```text
//! GRIND_BITS="18 19 20 21 22 23 24 25" GRIND_THREADS="1 10" GRIND_SAMPLES=200 \
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --features span-metrics --example grind_probe
//! ```
use bitz::piop::spartan::grinding::{find_grinding_nonce, GrindingSeed};

fn env_list<T: std::str::FromStr>(name: &str, default: &[T]) -> Vec<T>
where
    T: Clone,
{
    std::env::var(name)
        .ok()
        .map(|v| {
            v.split_whitespace()
                .map(|p| p.parse().ok().unwrap_or_else(|| panic!("bad {name} entry {p}")))
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn seed_for(bits: u32, sample: usize) -> GrindingSeed {
    let hash = blake3::hash(format!("bitz/grind_probe/v1/{bits}/{sample}").as_bytes());
    GrindingSeed::from_bytes(*hash.as_bytes())
}

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    let bits_list = env_list::<u32>("GRIND_BITS", &[18, 19, 20, 21, 22, 23, 24, 25]);
    let threads_list = env_list::<usize>("GRIND_THREADS", &[1, 10]);
    let samples: usize = std::env::var("GRIND_SAMPLES").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let warmup: usize = std::env::var("GRIND_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    println!("csv,threads,bits,samples,mean_ms,median_ms,min_ms,max_ms,mean_attempts,ns_per_attempt");
    for &threads in &threads_list {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("pool");
        for &bits in &bits_list {
            for i in 0..warmup {
                let seed = seed_for(bits, usize::MAX - i);
                std::hint::black_box(pool.install(|| find_grinding_nonce(&seed, bits)).expect("grind"));
            }
            let mut times_ms = Vec::with_capacity(samples);
            let mut attempts_total = 0f64;
            let mut ns_per_attempt = Vec::with_capacity(samples);
            for i in 0..samples {
                let seed = seed_for(bits, i);
                let (nonce, t0) = bitz::observability::measure(
                    tracing::info_span!("grind_probe:nonce"),
                    || pool.install(|| find_grinding_nonce(&seed, bits)).expect("grind"),
                ).expect("measure completed operation");
                let dt = t0;
                std::hint::black_box(nonce);
                let attempts = nonce as f64 + 1.0;
                times_ms.push(dt.as_secs_f64() * 1e3);
                attempts_total += attempts;
                ns_per_attempt.push(dt.as_nanos() as f64 / attempts);
            }
            let mean = times_ms.iter().sum::<f64>() / samples as f64;
            let mut sorted = times_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if samples % 2 == 1 {
                sorted[samples / 2]
            } else {
                (sorted[samples / 2 - 1] + sorted[samples / 2]) / 2.0
            };
            let min = sorted[0];
            let max = sorted[samples - 1];
            let mean_attempts = attempts_total / samples as f64;
            let mut npa = ns_per_attempt.clone();
            npa.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let npa_median = npa[samples / 2];
            eprintln!(
                "threads={threads:2} bits={bits:2} samples={samples}: mean {mean:8.2} ms  median {median:8.2} ms  \
                 [min {min:.2}, max {max:.2}]  mean attempts 2^{:.2}  {npa_median:.2} ns/attempt",
                mean_attempts.log2()
            );
            println!(
                "csv,{threads},{bits},{samples},{mean:.3},{median:.3},{min:.3},{max:.3},{mean_attempts:.0},{npa_median:.3}"
            );
        }
    }
}
