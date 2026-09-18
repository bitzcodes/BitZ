//! Benchmark proving 2^k multiplications of **native BabyBear elements**:
//! a 3-column AIR (`a`, `b`, `c`) with the single degree-2 constraint
//! `a·b = c` — no bit decompositions or range checks, this is field-native
//! multiplication. Challenges live in the quartic extension
//! `BinomialExtensionField<BabyBear, 4>`; the PCS is Plonky3's
//! `TwoAdicFriPcs` at `FriParameters::new_benchmark` (rate 1/2,
//! 100 queries, 16-bit query PoW), Poseidon2 Merkle hashing — the same
//! production-benchmark settings as `prove_mul_int`.
//!
//! Baseline for the F2Z-as-PCS comparison (f2z-pcs `F2Z_BENCH_EXT=bb4`):
//! a multilinear prover backed by F2Z commits the trace's bit
//! decomposition over F₂ and opens it at a point in the same quartic
//! extension.
//!
//! Plain main (no criterion): per height, one verified warm-up proof, then
//! `P3_REPS` timed prove/verify reps (medians reported) and the postcard
//! proof size. Machine-readable `RESULT` lines.
//!
//! Run:  cargo bench -p p3-uni-stark --features parallel --bench prove_mul_babybear
//! Env:  P3_LOG_HEIGHTS="15 16 17 18 19 20" (default), P3_REPS=3 (default)

use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// One native BabyBear multiplication per row: `[a, b, c]`, `a·b = c`.
struct MulBabyBearAir;

impl<F> BaseAir<F> for MulBabyBearAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for MulBabyBearAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let (a, b, c) = (local[0], local[1], local[2]);
        builder.assert_eq(c, a.into() * b.into());
    }
}

fn build_trace(rng: &mut SmallRng, log_height: usize) -> RowMajorMatrix<BabyBear> {
    let height = 1usize << log_height;
    let mut values = BabyBear::zero_vec(height * 3);
    for row in values.chunks_exact_mut(3) {
        // `from_u32` reduces mod p, so any u32 draw is a valid element.
        let a = BabyBear::from_u32(rng.random::<u32>());
        let b = BabyBear::from_u32(rng.random::<u32>());
        row[0] = a;
        row[1] = b;
        row[2] = a * b;
    }
    RowMajorMatrix::new(values, 3)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

fn main() {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    let val_mmcs = ValMmcs::new(hash, compress, 0);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    let challenger = Challenger::new(perm);

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = MulBabyBearAir;

    let heights: Vec<usize> = std::env::var("P3_LOG_HEIGHTS")
        .unwrap_or_else(|_| "15 16 17 18 19 20".into())
        .split_whitespace()
        .map(|x| {
            x.parse()
                .expect("P3_LOG_HEIGHTS: space-separated log2 heights")
        })
        .collect();
    let reps: usize = std::env::var("P3_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    println!("Plonky3 native BabyBear mul (3 cols, a*b=c, quartic challenges, FRI new_benchmark)");
    for log_height in heights {
        let trace = build_trace(&mut rng, log_height);

        // Verified warm-up proof (excluded from stats), also the size probe.
        let proof = prove(&config, &air, trace.clone(), &[]);
        verify(&config, &air, &proof, &[]).expect("proof must verify");
        let proof_bytes = postcard::to_allocvec(&proof).expect("serialize").len();

        let mut prove_ms = Vec::with_capacity(reps);
        let mut verify_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = trace.clone();
            let t0 = Instant::now();
            let p = prove(&config, &air, t, &[]);
            prove_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            let t1 = Instant::now();
            verify(&config, &air, &p, &[]).expect("proof must verify");
            verify_ms.push(t1.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "RESULT log_height={log_height} muls=2^{log_height} prove_ms={:.2} verify_ms={:.2} proof_bytes={proof_bytes} ({:.1} KiB)  (medians of {reps})",
            median(prove_ms),
            median(verify_ms),
            proof_bytes as f64 / 1024.0,
        );
    }
}
