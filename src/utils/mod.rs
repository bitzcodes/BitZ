pub(crate) mod blake3x4;
pub(crate) mod bit_packing;
pub mod delayed_reduction;
pub mod from_ref;
pub mod inner_product;
pub mod inner_transparent_field;
pub mod mul_by_scalar;
pub mod named;
pub mod ops_macros;
pub mod parallel;
pub mod projectable_to_field;
pub mod wide_mul;

use crate::poly::coefficient::Coefficient;

// Can't use enums in const generics in stable Rust yet, so we use constants
// instead. `CHECKED` honours the `unchecked` feature (the zinc-plus
// convention: overflow-guarded integer ops by default, plain ops when the
// consumer opts in).
pub const CHECKED: bool = cfg!(not(feature = "unchecked"));
pub const UNCHECKED: bool = false;

/// Returns ceil(log2(x)).
/// Copied from ark-std.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
pub const fn log2(x: usize) -> u32 {
    if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub fn powers<R: Coefficient>(x: R, one: R, num_pows: usize) -> Vec<R> {
    if num_pows == 0 {
        return Vec::new();
    }

    let mut pows = Vec::with_capacity(num_pows);

    pows.push(one);

    if num_pows == 1 {
        return pows;
    }

    let mut curr_pow = x.clone();

    for _ in 1..num_pows {
        pows.push(curr_pow.clone());
        curr_pow *= &x;
    }

    pows
}

// `#[macro_export]` macros land at the crate root; re-export them here so
// vendored `zinc_utils::`-style paths keep working after the rename.
pub use crate::{
    add, cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter, cfg_iter_mut, mul, neg, sub,
};

/// Serializes tests that read or mutate process-global `BITZ_*` environment
/// switches. A test that toggles a transcript-shaping variable (for example
/// `BITZ_QUAD`) must hold this lock for its whole body; an end-to-end
/// prove/verify test that must observe one stable value across its run
/// takes the same lock. Poisoning is ignored: the lock only serializes.
#[cfg(test)]
pub(crate) static QUAD_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
