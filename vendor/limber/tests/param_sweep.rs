//! Temporary diagnostic: sweep IntEvalParams::derive over k for the bench
//! shapes. Run with:
//!   cargo test --release --test param_sweep -- --nocapture

use limber::provider::pcs::integer_modpcs::{DEFAULT_LOG_T_F, IntEvalParams};

#[test]
fn sweep_k_for_bench_shapes() {
  // num_vars = log2(poly size) for the w-poly of each bench config.
  for &num_vars in &[8usize, 10, 12] {
    println!("\n== num_vars = {num_vars} (w-poly size 2^{num_vars}) ==");
    println!(
      "{:>3} {:>6} {:>3} {:>3} | {:>14} {:>16}",
      "k", "log_p", "s", "t", "opens s(3t+1)+", "per-chain commits"
    );
    for k in 2..=16 {
      match IntEvalParams::derive_no_limb_split(DEFAULT_LOG_T_F, k, num_vars) {
        Ok(p) => {
          let t = num_vars.saturating_sub(p.k).div_ceil(p.k);
          // F-query opens after current batching:
          // aprev_batch(1) + curr(t) + final(s) + j>=2 aprev (s*(t-1)) + rc value opens (1+2t)
          let opens = 1 + t + p.s + p.s * t.saturating_sub(1) + (1 + 2 * t);
          // committed entries per chain: sum_j 2 * 2^(num_vars - j*k)
          let commits: usize = (1..=t).map(|j| 2usize << (num_vars - j * p.k)).sum();
          println!(
            "{:>3} {:>6} {:>3} {:>3} | {:>14} {:>16}",
            p.k,
            p.log_p,
            p.s,
            t,
            opens,
            commits * p.s
          );
        }
        Err(e) => println!("{k:>3}  -- invalid: {e}"),
      }
    }
  }
}
