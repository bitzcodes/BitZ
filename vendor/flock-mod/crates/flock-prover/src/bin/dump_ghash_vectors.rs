//! Dump GF(2^128) GHASH test vectors from the *real* `flock` implementation so
//! the CUDA port (`cuda-ghash/`) can be checked bit-for-bit against it.
//!
//! On this x86_64 host `Gf128::mul` uses the PCLMULQDQ binius path — the same
//! algorithm the CUDA `ghash_mul_binius` mirrors with `clmad`.
//!
//! Output: little-endian binary to the path in argv[1] (default vectors.bin):
//!   magic  u32 = 0x47483132 ("GH12")
//!   count  u32
//!   count * { a.lo, a.hi, b.lo, b.hi, prod.lo, prod.hi } : u64 each
//!
//! Run:  cargo run --release --bin dump_ghash_vectors -- cuda-ghash/vectors.bin

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::Gf128;

/// SplitMix64 — same constants as the `gf128_kernels::tests::Rng`, so vectors are
/// reproducible and line up with the in-tree unit tests.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f128(&mut self) -> Gf128 {
        Gf128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "vectors.bin".to_string());
    let count: u32 = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let mut rng = Rng::new(0xC0FFEE);
    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x4748_3132u32.to_le_bytes())?; // "GH12"
    w.write_all(&count.to_le_bytes())?;

    // Include a few structured edge cases first, then random.
    let edges = [
        (Gf128::ZERO, Gf128::ZERO),
        (
            Gf128::ONE,
            Gf128 {
                lo: 0xDEAD_BEEF,
                hi: 0x1234,
            },
        ),
        (
            Gf128::GENERATOR,
            Gf128 {
                lo: 0,
                hi: 1u64 << 63,
            },
        ), // x · x^127 = 0x87
        (Gf128 { lo: 0, hi: 1 }, Gf128 { lo: 0, hi: 1 }), // x^64 · x^64
        (
            Gf128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
            Gf128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
        ),
    ];

    for i in 0..count as usize {
        let (a, b) = if i < edges.len() {
            edges[i]
        } else {
            (rng.next_f128(), rng.next_f128())
        };
        let p = a * b; // the canonical flock product
        for v in [a.lo, a.hi, b.lo, b.hi, p.lo, p.hi] {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    eprintln!("wrote {count} vectors to {path}");
    Ok(())
}
