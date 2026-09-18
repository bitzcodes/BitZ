//! Benchmark proving 2^k Blake3 compressions with Plonky3's `Blake3Air`
//! (one permutation per row, 9168 columns: 8704 boolean + 464 sixteen-bit
//! limbs ≈ 2^14 committed bits per compression) over BabyBear, challenges
//! in the quartic extension, `TwoAdicFriPcs` at
//! `FriParameters::new_benchmark` (rate 1/2, 100 queries, 16-bit query
//! PoW), Poseidon2 Merkle hashing — the same production-benchmark settings
//! as `prove_mul_int` / `prove_mul_babybear`.
//!
//! Baseline for the F2Z-as-PCS comparison (f2z-pcs `F2Z_BENCH_EXT=bb4` at
//! `t:s = 13:9 / 15:10 / 17:12` for 2^8 / 2^11 / 2^15 compressions — the
//! same trace committed as ~2^{14+k} bits, opened at a BabyBear⁴ point).
//!
//! Plain main (no criterion): per height, one verified warm-up proof, then
//! `P3_REPS` timed prove/verify reps (medians reported) and the postcard
//! proof size. Machine-readable `RESULT` lines.
//!
//! Run:  cargo bench -p p3-uni-stark --features parallel --bench prove_blake3
//! Env:  P3_LOG_HEIGHTS="8 11 12 15" (default), P3_REPS=3 (default)

use std::time::Instant;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_blake3_air::Blake3Air;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

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
    let log_blowup = fri_params.log_blowup;

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = Blake3Air {};

    let heights: Vec<usize> = std::env::var("P3_LOG_HEIGHTS")
        .unwrap_or_else(|_| "8 11 12 15".into())
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

    println!(
        "Plonky3 Blake3Air ({} cols/row, 1 compression/row, quartic challenges, FRI new_benchmark)",
        p3_blake3_air::NUM_BLAKE3_COLS
    );
    for log_height in heights {
        let num_hashes = 1usize << log_height;
        let inputs: Vec<[u32; 24]> = (0..num_hashes)
            .map(|_| std::array::from_fn(|_| rng.random()))
            .collect();
        let build_trace = || p3_blake3_air::generate_trace_rows::<Val>(inputs.clone(), log_blowup);

        // Verified warm-up proof (excluded from stats), also the size probe.
        let proof = prove(&config, &air, build_trace(), &[]);
        verify(&config, &air, &proof, &[]).expect("proof must verify");
        let proof_bytes = postcard::to_allocvec(&proof).expect("serialize").len();

        // Trace generation is timed separately (the witness build, not the
        // prover) and excluded from prove_ms, matching the F2Z bench where
        // instance generation is outside the timed region too.
        let mut trace_ms = Vec::with_capacity(reps);
        let mut prove_ms = Vec::with_capacity(reps);
        let mut verify_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            let t = build_trace();
            trace_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            let t1 = Instant::now();
            let p = prove(&config, &air, t, &[]);
            prove_ms.push(t1.elapsed().as_secs_f64() * 1e3);
            let t2 = Instant::now();
            verify(&config, &air, &p, &[]).expect("proof must verify");
            verify_ms.push(t2.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "RESULT log_height={log_height} hashes=2^{log_height} trace_ms={:.2} prove_ms={:.2} verify_ms={:.2} proof_bytes={proof_bytes} ({:.1} KiB)  (medians of {reps})",
            median(trace_ms),
            median(prove_ms),
            median(verify_ms),
            proof_bytes as f64 / 1024.0,
        );
    }
}
