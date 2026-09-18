//! Reproducible `(t,s)` sweep for `2^14` SHA-256 compressions.
//!
//! The total product-assignment width is fixed at 29 variables. By default
//! this runs every supported one-chunk split that stays below the memory cap,
//! `t=7..=27`, with one warmup and 21 measured samples per split. `t=28` is
//! reported as skipped because its projected peak exceeds 60 GiB. Select a subset with
//! `BITZ_SHA_PRODUCT_TS="13 17"` and override samples with
//! `BITZ_BENCH_REPS=3`.
//! Enable `bench-peak-memory` to report peak live heap per sample and its
//! maximum per split. Proof sizes are reported in both samples and summaries.

#[path = "sha256_compressions.rs"]
mod sha256_compressions;

// The shared profile-dispatch macro resolves this module at the crate root.
use sha256_compressions::common;

fn main() {
    common::cli::EnvironmentCli::parse();
    let env = sha256_compressions::Env::from_environment(true);
    let default_sweep = env.default_product_sweep;
    let result_path = env.result_path.clone();
    sha256_compressions::run(env);

    if default_sweep && let Some(path) = result_path {
        use std::io::Write;

        let mut output = common::output::BenchmarkOutput::new("")
            .file(path, common::output::FileMode::AppendExisting)
            .expect("reopen SHA result output");
        for t in 1..=6 {
            writeln!(
                output,
                "STATUS product_t={t} product_s={} compressions=16384 status=unsupported reason=t_below_log_packing_7",
                29 - t,
            )
            .expect("write unsupported SHA split");
        }
        writeln!(
            output,
            "STATUS product_t=28 product_s=1 compressions=16384 status=skipped reason=projected_peak_exceeds_60_gib projected_peak_bytes=75150743216 peak_cap_bytes=64424509440"
        )
        .expect("write memory-skipped SHA split");
        output.flush().expect("flush SHA split status");
    }
}
