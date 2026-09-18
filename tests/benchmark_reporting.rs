//! Execute reporting tests in the bench sources with a real test harness.
#![allow(dead_code)]
#![recursion_limit = "512"]

// Legacy benchmark macros and PCS adapters refer to crate::common.
use multiswap::common;

#[path = "../benches/ligerito_bounds.rs"]
mod bounds;
#[path = "../benches/sha256_ecdsa_compare.rs"]
mod ecdsa;
#[path = "../benches/multiswap.rs"]
mod multiswap;
#[path = "../benches/mul/mod.rs"]
mod mul;

#[path = "../benches/hybrid_u32_sha256/runner.rs"]
mod hybrid;
