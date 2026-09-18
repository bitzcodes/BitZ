//! Phase profile of one hybrid (mod-2^32 mul + chained SHA-256) prove via the
//! crate's Perfetto subscriber, with separate peak-RSS observations. Bench-identical inputs; one
//! warm-up prove is dumped and discarded, then `PROBE_REPS` profiled proves
//! (default 1) per shape. The nested tree lands on stderr.
//!
//! ```text
//! PROBE_SHAPES="19:11 20:12" RAYON_NUM_THREADS=8 \
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example hybrid_probe --features hybrid
//! ```
use bitz::hybrid::{Parameters, PreparedHybrid};
use bitz::piop::spartan::MulRow;
use tracing_subscriber::prelude::*;

fn rss_peak() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) }, 0, "read peak RSS");
    #[cfg(target_os = "macos")]
    { usage.ru_maxrss as u64 }
    #[cfg(not(target_os = "macos"))]
    { usage.ru_maxrss as u64 * 1024 }
}

fn main() {
    let memory = bitz::observability::memory::MemoryLayer::new(rss_peak);
    tracing_subscriber::registry().with(bitz::observability::layer()).with(memory.clone()).init();
    let shapes: Vec<(u32, u32)> = std::env::var("PROBE_SHAPES")
        .map(|v| {
            v.split_whitespace()
                .map(|p| {
                    let mut it = p.split(':');
                    (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|_| vec![(19, 11)]);
    let reps: usize = std::env::var("PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let verify = std::env::var("PROBE_VERIFY").map_or(true, |v| v != "0");
    for (mul_log, sha_log) in shapes {
        let parameters = Parameters {
            multiplications: 1 << mul_log,
            sha_compressions: 1 << sha_log,
        };
        let (prepared, t0) =
            bitz::observability::measure(tracing::info_span!("hybrid_probe:prepared"), || {
                PreparedHybrid::new(parameters).expect("prepare")
            })
            .expect("measure completed operation");
        eprintln!(
            "setup {}:{} {:.0} ms",
            mul_log,
            sha_log,
            t0.as_secs_f64() * 1e3
        );
        let inputs: Vec<_> = (0..parameters.multiplications as u32)
            .map(|i| (i.wrapping_mul(0x9e3779b9), u32::MAX - i))
            .collect();
        let blocks: Vec<[u32; 16]> = (0..parameters.sha_compressions as u32)
            .map(|i| std::array::from_fn(|j| i.wrapping_mul(0x85ebca6b).wrapping_add(j as u32)))
            .collect();
        for rep in 0..=reps {
            let _ = memory.take();
            let start_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let start = tracing::info_span!("hybrid_probe:start").entered();
            let rows: Vec<_> = inputs
                .iter()
                .map(|&(x, y)| MulRow::<u32>::new(x, y))
                .collect();
            let committed = prepared.commit_mod32(&rows, &blocks).expect("commit");
            drop(start);
            let proof = tracing::info_span!("hybrid_probe:proof").in_scope(|| prepared.prove(&committed).expect("prove"));
            let growth = memory.take();
            let intervals = start_recording.intervals().expect("prover intervals");
            let commit_ms = bitz::observability::duration(&intervals, "hybrid_probe:start").expect("commit duration").as_secs_f64() * 1e3;
            let prove_ms = bitz::observability::duration(&intervals, "hybrid_probe:proof").expect("prove duration").as_secs_f64() * 1e3;
            let bytes = proof.to_bytes();
            let header = if rep == 0 {
                format!("warmup {mul_log}:{sha_log} (discard) commit {commit_ms:.1} ms prove {prove_ms:.1} ms")
            } else {
                format!(
                    "{mul_log}:{sha_log} rep {rep}: commit {commit_ms:.1} ms + prove {prove_ms:.1} ms = {:.1} ms, proof {} B",
                    commit_ms + prove_ms,
                    bytes.len()
                )
            };
            bitz::observability::write_profile(std::io::stderr().lock(), &header, &intervals, Some(&growth)).expect("write profile");
            if verify {
                let t2_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
                let t2 = tracing::info_span!("hybrid_probe:t2").entered();
                let decoded = prepared.proof_from_bytes(committed.statement(), &bytes).expect("decode");
                prepared.verify(committed.statement(), &decoded).expect("verify");
                drop(t2);
                let growth = memory.take();
                let intervals = t2_recording.intervals().expect("verifier intervals");
                let verify_ms = bitz::observability::duration(&intervals, "hybrid_probe:t2").expect("verify duration").as_secs_f64() * 1e3;
                bitz::observability::write_profile(std::io::stderr().lock(), &format!("verify {mul_log}:{sha_log} rep {rep}: {verify_ms:.1} ms"), &intervals, Some(&growth)).expect("write profile");
            }
            eprintln!("digest {}", blake3::hash(&bytes).to_hex());
        }
    }
}
