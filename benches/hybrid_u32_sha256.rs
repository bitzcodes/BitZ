//! Non-ZK, 100-bit-target SHA chain + multiplication-mod-2^32 comparison.
//! Hybrid, separate and all-Binius modes prove x*y = z + 2^32*w for four u32 limbs.
//!
//! RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=8 \
//!   cargo bench --bench hybrid_u32_sha256 --features hybrid -- \
//!   --sweep --mode hybrid --iterations 3
//!
//! Default sweep: six equal-witness pairs, from 15:7 through 20:12.
//! Custom pairs: --sweep --shapes 15:7,16:8. Compare backends: --mode all.
#[path = "hybrid_u32_sha256/runner.rs"]
mod runner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}
