//! Timings for the patterns the prover spends its time in.
//!
//! Run with `cargo bench -p field --bench gf128`. Divan drives the sampling;
//! sample counts and time budgets are tunable through `DIVAN_*` environment
//! variables.
//!
//! The operand arrays are 16 KiB, so a pair of them stays in L1 and what is
//! measured is arithmetic rather than the memory system.
//!
//! The header reports which kernel and which instructions the build selected.
//! Both are silent: `pmull` needs the `aes` target feature and the three-way
//! `eor3` needs `sha3`, and `aarch64-unknown-linux-gnu` enables neither by
//! default. Dropping `eor3` alone costs about a third of the multiply.

use divan::counter::ItemsCount;
use divan::{Bencher, black_box};
use field::gf128::KERNEL;
use field::{FixedBasePow, Gf128, Wide256};
use num_traits::{ConstZero, Inv, Pow};

/// Elements per invocation. 1024 of them is 16 KiB, so a pair of operand
/// arrays stays in L1.
const N: usize = 1024;

fn main() {
    let eor3 = cfg!(target_feature = "sha3");
    println!("kernel {KERNEL}, eor3 {}", if eor3 { "yes" } else { "no" });
    divan::main();
}

/// Deterministic operands. xorshift rather than a dependency: the bench only
/// needs values that are not all alike.
fn operands(count: usize, seed: u64) -> Vec<Gf128> {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    (0..count).map(|_| Gf128::new(next(), next())).collect()
}

fn xs() -> Vec<Gf128> {
    operands(N, 0x243f_6a88_85a3_08d3)
}

fn ys() -> Vec<Gf128> {
    operands(N, 0x1319_8a2e_0370_7344)
}

/// Independent products: throughput, the shape of the sumcheck round bodies.
#[divan::bench]
fn mul_batch(bencher: Bencher) {
    let (xs, ys) = (xs(), ys());
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut acc = Gf128::ZERO;
        for i in 0..N {
            acc += xs[i] * ys[i];
        }
        acc
    });
}

/// A dependent chain: latency, where throughput cannot be hidden.
#[divan::bench]
fn mul_chain(bencher: Bencher) {
    let (xs, ys) = (xs(), ys());
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut a = xs[0];
        for _ in 0..N {
            a *= ys[0];
        }
        a
    });
}

#[divan::bench]
fn square_chain(bencher: Bencher) {
    let xs = xs();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut a = xs[0];
        for _ in 0..N {
            a = a.square();
        }
        a
    });
}

/// The same products as `mul_batch`, accumulated unreduced and reduced once.
#[divan::bench]
fn wide_dot(bencher: Bencher) {
    let (xs, ys) = (xs(), ys());
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        let mut acc = Wide256::zero();
        for i in 0..N {
            acc += Wide256::mul(xs[i], ys[i]);
        }
        acc.reduce()
    });
}

#[divan::bench]
fn inverse(bencher: Bencher) {
    let xs = xs();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for &x in &xs {
            black_box(x.inv());
        }
    });
}

/// Full-width exponents: what the fold values are.
#[divan::bench]
fn pow_square_and_multiply(bencher: Bencher) {
    let exps: Vec<u128> = xs()
        .iter()
        .map(|x| (x.hi as u128) << 64 | x.lo as u128)
        .collect();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for &e in &exps {
            black_box(Gf128::GENERATOR.pow(e));
        }
    });
}

#[divan::bench]
fn pow_public_comb_w8(bencher: Bencher) {
    let comb = FixedBasePow::<_, 2>::new_public(field::Gf128Ops, Gf128::GENERATOR, 8);
    let exps: Vec<u128> = xs()
        .iter()
        .map(|x| (x.hi as u128) << 64 | x.lo as u128)
        .collect();
    bencher.counter(ItemsCount::new(N)).bench_local(|| {
        for &e in &exps {
            black_box(comb.pow_public(&field::Uint::from_words([e as u64, (e >> 64) as u64])));
        }
    });
}
