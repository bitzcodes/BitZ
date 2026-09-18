//! Proof CLI using the same workload runner as the hybrid benchmark.
#[path = "../../benches/hybrid_u32_sha256/runner.rs"]
mod runner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}
