//! Benchmark proving 2^20 (~1M) multiplications of range-checked integers.
//!
//! Two widths are benchmarked: `16x16 -> 32` and `32x32 -> 64`.
//!
//! Each trace row holds one multiplication `a * b = c`:
//! - `a` and `b` are range-checked to `NUM_BITS` bits via boolean bit-decomposition
//!   columns, so a malicious prover cannot use out-of-range inputs.
//! - The field is Goldilocks (p = 2^64 - 2^32 + 1). The largest product of two
//!   32-bit inputs is (2^32 - 1)^2 = 2^64 - 2^33 + 1, which is smaller than p by
//!   exactly 2^32. So even at the 32-bit width the full product fits without
//!   field wraparound: `c` is pinned to the exact integer product, with no
//!   truncation and no overflow.
//!
//! Note that this exhausts the field's headroom at 32 bits: the *sum* of two
//! such products would wrap, so downstream constraints cannot accumulate
//! 32-bit products without decomposing them first.
//!
//! Run with `cargo bench -p p3-uni-stark --features parallel --bench prove_mul_int`.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Trace heights to sweep, as log2 of the number of multiplications proved.
const LOG_HEIGHTS: [usize; 6] = [15, 16, 17, 18, 19, 20];

/// Column layout: `[a, b, c, a_bits[0..NUM_BITS], b_bits[0..NUM_BITS]]`.
const fn trace_width(num_bits: usize) -> usize {
    3 + 2 * num_bits
}

/// One `NUM_BITS x NUM_BITS` multiplication per row, with bit-decomposition range checks.
struct MulAir<const NUM_BITS: usize>;

impl<F, const NUM_BITS: usize> BaseAir<F> for MulAir<NUM_BITS> {
    fn width(&self) -> usize {
        trace_width(NUM_BITS)
    }
}

impl<AB: AirBuilder, const NUM_BITS: usize> Air<AB> for MulAir<NUM_BITS> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let (a, b, c) = (local[0], local[1], local[2]);
        let a_bits = &local[3..3 + NUM_BITS];
        let b_bits = &local[3 + NUM_BITS..trace_width(NUM_BITS)];

        // Each input must recompose from boolean bits, so it lies in [0, 2^NUM_BITS).
        for (value, bits) in [(a, a_bits), (b, b_bits)] {
            let mut recomposed = AB::Expr::ZERO;
            for (i, &bit) in bits.iter().enumerate() {
                builder.assert_bool(bit);
                recomposed += bit.into() * AB::Expr::from_u64(1u64 << i);
            }
            builder.assert_eq(value, recomposed);
        }

        // Both inputs are below 2^NUM_BITS, so for NUM_BITS <= 32 the integer
        // product stays below p: this field equation forces c to be the exact
        // untruncated product.
        builder.assert_eq(c, a.into() * b.into());
    }
}

fn build_trace<const NUM_BITS: usize>(
    rng: &mut SmallRng,
    log_height: usize,
) -> RowMajorMatrix<Goldilocks> {
    let width = trace_width(NUM_BITS);
    let height = 1 << log_height;
    let mut values = Goldilocks::zero_vec(height * width);
    for row in values.chunks_exact_mut(width) {
        // Shifting down keeps the inputs inside NUM_BITS bits.
        let a: u32 = rng.random::<u32>() >> (32 - NUM_BITS);
        let b: u32 = rng.random::<u32>() >> (32 - NUM_BITS);
        row[0] = Goldilocks::from_u32(a);
        row[1] = Goldilocks::from_u32(b);
        row[2] = Goldilocks::from_u64(u64::from(a) * u64::from(b));
        for i in 0..NUM_BITS {
            row[3 + i] = Goldilocks::from_bool((a >> i) & 1 == 1);
            row[3 + NUM_BITS + i] = Goldilocks::from_bool((b >> i) & 1 == 1);
        }
    }
    RowMajorMatrix::new(values, width)
}

fn bench_width<const NUM_BITS: usize>(c: &mut Criterion) {
    type Val = Goldilocks;
    type Challenge = BinomialExtensionField<Val, 2>;

    type Perm = Poseidon2Goldilocks<8>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type MyHash = PaddingFreeSponge<Perm, 8, 4, 4>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 4, 8>;
    let compress = MyCompress::new(perm.clone());

    type ValPacking = <Val as Field>::Packing;
    type ValMmcs = MerkleTreeMmcs<ValPacking, ValPacking, MyHash, MyCompress, 2, 4>;
    let val_mmcs = ValMmcs::new(hash, compress, 0);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 8, 4>;
    let challenger = Challenger::new(perm);

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = MulAir::<NUM_BITS>;

    for log_height in LOG_HEIGHTS {
        let trace = build_trace::<NUM_BITS>(&mut rng, log_height);
        let label = format!("{NUM_BITS}x{NUM_BITS}_2^{log_height}_muls");

        // Sanity check outside the timed region: the proof must verify.
        let proof = prove(&config, &air, trace.clone(), &[]);
        verify(&config, &air, &proof, &[]).expect("proof must verify");

        let proof_bytes = postcard::to_allocvec(&proof).expect("unable to serialize proof");
        // Machine-readable line, consumed when tabulating results.
        println!(
            "RESULT bits={NUM_BITS} log_height={log_height} cols={} proof_bytes={}",
            trace_width(NUM_BITS),
            proof_bytes.len(),
        );

        let mut group = c.benchmark_group("prove_mul_int");
        group.sample_size(10);
        group.bench_function(&label, |b| {
            // The trace reaches hundreds of MB, so build exactly one per iteration.
            b.iter_batched(
                || trace.clone(),
                |trace| prove(&config, &air, trace, &[]),
                BatchSize::PerIteration,
            );
        });
        group.finish();

        let mut group = c.benchmark_group("verify_mul_int");
        group.bench_function(&label, |b| {
            b.iter(|| verify(&config, &air, &proof, &[]).expect("proof must verify"));
        });
        group.finish();
    }
}

fn bench_mul_int(c: &mut Criterion) {
    bench_width::<16>(c);
    bench_width::<32>(c);
}

criterion_group!(benches, bench_mul_int);
criterion_main!(benches);
