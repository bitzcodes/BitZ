//! End-to-end Plonky3 BabyBear multiplication STARK benchmark.
//!
//! This benchmark adapts the multiplication AIR from `tests/mul_air.rs` and
//! isolates independent degree-two constraints of the form `a * b = c`.
//! By default it measures `2^15` through `2^24` multiplications using 16 AIR
//! lanes, one excluded warmup, and five verified samples per size.
//!
//! Run with:
//! ```text
//! RAYON_NUM_THREADS=10 RUSTFLAGS="-Ctarget-cpu=native" \
//!   cargo bench -p p3-uni-stark --features parallel --bench baby_bear_mul
//! ```
//!
//! The `P3_BB_EXPONENTS`, `P3_BB_REPS`, `P3_BB_LANES`, and
//! `P3_BB_UPSTREAM_EXTRAS` environment variables override those defaults.

use std::hint::black_box;
use std::time::Instant;

use p3_air::symbolic::AirLayout;
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
use p3_uni_stark::{StarkConfig, StarkSecurityParams, prove, verify};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const TRACE_COLS_PER_MUL: usize = 3;

#[derive(Clone, Copy)]
struct MulAir<const LANES: usize> {
    upstream_extras: bool,
}

impl<const LANES: usize> MulAir<LANES> {
    fn valid_trace<F: Field>(&self, rows: usize) -> RowMajorMatrix<F>
    where
        StandardUniform: Distribution<F>,
    {
        let width = TRACE_COLS_PER_MUL * LANES;
        let mut rng = SmallRng::seed_from_u64(1);
        let mut values = F::zero_vec(rows * width);
        for (index, abc) in values.chunks_exact_mut(3).enumerate() {
            let row = index / LANES;
            let a = if self.upstream_extras {
                F::from_usize(index)
            } else {
                rng.random()
            };
            let b = if self.upstream_extras && row == 0 {
                a.square() + F::ONE
            } else {
                rng.random()
            };
            abc[0] = a;
            abc[1] = b;
            abc[2] = a * b;
        }
        RowMajorMatrix::new(values, width)
    }
}

impl<F, const LANES: usize> BaseAir<F> for MulAir<LANES> {
    fn width(&self) -> usize {
        TRACE_COLS_PER_MUL * LANES
    }
}

impl<AB: AirBuilder, const LANES: usize> Air<AB> for MulAir<LANES> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        for lane in 0..LANES {
            let start = TRACE_COLS_PER_MUL * lane;
            let a = local[start];
            let b = local[start + 1];
            let c = local[start + 2];
            builder.assert_zero(a * b - c);
            if self.upstream_extras {
                builder.when_first_row().assert_eq(a * a + AB::Expr::ONE, b);
                builder
                    .when_transition()
                    .assert_eq(a + AB::Expr::from_u8(LANES as u8), next[start]);
            }
        }
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn parse_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).as_deref() {
        Ok("1" | "true" | "yes") => true,
        Ok("0" | "false" | "no") => false,
        Ok(value) => panic!("{name} must be 0/1 or false/true, got {value}"),
        Err(_) => default,
    }
}

fn run<const LANES: usize>(exponents: &[usize], reps: usize, upstream_extras: bool) {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type Config = StarkConfig<Pcs, Challenge, Challenger>;

    assert!(LANES > 0 && LANES < 256);
    let air = MulAir::<LANES> { upstream_extras };

    for &exponent in exponents {
        assert!(exponent >= 4, "operation exponent must be at least 4");
        let log_rows = exponent - 4;
        let rows = 1usize << log_rows;
        let multiplications = rows * LANES;

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri = FriParameters::new_benchmark(challenge_mmcs);
        let fri_regime = fri.security_regime();
        let pcs = Pcs::new(Dft::default(), val_mmcs, fri);
        let config = Config::new(pcs, Challenger::new(perm));

        let mut security = StarkSecurityParams::from_air::<Val, Challenge, _>(
            fri_regime,
            &air,
            AirLayout::from_air::<Val>(&air),
            Challenge::bits(),
            128,
            usize::from(upstream_extras) + 1,
        );

        let warm_trace = air.valid_trace::<Val>(rows);
        let warm_proof = prove(&config, &air, warm_trace, &[]);
        verify(&config, &air, &warm_proof, &[]).expect("warmup proof must verify");
        drop(warm_proof);

        let mut trace_samples = Vec::with_capacity(reps);
        let mut prove_samples = Vec::with_capacity(reps);
        let mut verify_samples = Vec::with_capacity(reps);
        let mut proof_sizes = Vec::with_capacity(reps);
        let mut conjectured_bits = 0;
        let mut proven_bits = 0;
        let mut proven_udr_bits = 0;
        let mut proven_ldr_bits = 0;

        for sample in 0..reps {
            let trace_started = Instant::now();
            let trace = air.valid_trace::<Val>(rows);
            let trace_ms = trace_started.elapsed().as_secs_f64() * 1_000.0;

            let prove_started = Instant::now();
            let proof = prove(&config, &air, trace, &[]);
            let prove_ms = prove_started.elapsed().as_secs_f64() * 1_000.0;

            let verify_started = Instant::now();
            verify(&config, &air, &proof, &[]).expect("measured proof must verify");
            let verify_ms = verify_started.elapsed().as_secs_f64() * 1_000.0;

            let proof_bytes = postcard::to_allocvec(&proof)
                .expect("proof serialization")
                .len();
            security.num_batched_functions =
                proof.opened_values.trace_local.len() + proof.opened_values.quotient_chunks.len();
            conjectured_bits = proof.conjectured_security(&security).security_bits;
            let proven = proof.proven_security(&security);
            proven_bits = proven.security_bits();
            proven_udr_bits = proven.unique_decoding_bits;
            proven_ldr_bits = proven.list_decoding_bits;

            black_box(&proof);
            trace_samples.push(trace_ms);
            prove_samples.push(prove_ms);
            verify_samples.push(verify_ms);
            proof_sizes.push(proof_bytes);
            println!(
                "P3_TRIAL schema=p3-babybear-mul/1 exponent={exponent} log_rows={log_rows} \
                 rows={rows} lanes={LANES} multiplications={multiplications} extras={} \
                 sample={sample} trace_ms={trace_ms:.6} prove_ms={prove_ms:.6} \
                 verify_ms={verify_ms:.6} proof_bytes={proof_bytes} \
                 conjectured_bits={conjectured_bits} proven_bits={proven_bits} \
                 proven_udr_bits={proven_udr_bits} proven_ldr_bits={proven_ldr_bits}",
                u8::from(upstream_extras),
            );
        }

        proof_sizes.sort_unstable();
        println!(
            "P3_RESULT schema=p3-babybear-mul/1 exponent={exponent} log_rows={log_rows} \
             rows={rows} lanes={LANES} multiplications={multiplications} extras={} reps={reps} \
             trace_ms={:.6} prove_ms={:.6} verify_ms={:.6} proof_bytes={} \
             conjectured_bits={conjectured_bits} proven_bits={proven_bits} \
             proven_udr_bits={proven_udr_bits} proven_ldr_bits={proven_ldr_bits} \
             fri_log_blowup=1 fri_queries=100 fri_query_pow_bits=16",
            u8::from(upstream_extras),
            median(&trace_samples),
            median(&prove_samples),
            median(&verify_samples),
            proof_sizes[proof_sizes.len() / 2],
        );
    }
}

fn main() {
    let exponents: Vec<usize> = std::env::var("P3_BB_EXPONENTS")
        .unwrap_or_else(|_| {
            (15..=24)
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .split([',', ' '])
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().expect("integer exponent"))
        .collect();
    let reps: usize = std::env::var("P3_BB_REPS")
        .unwrap_or_else(|_| "5".to_owned())
        .parse()
        .expect("positive repetition count");
    assert!(reps > 0);
    let lanes: usize = std::env::var("P3_BB_LANES")
        .unwrap_or_else(|_| "16".to_owned())
        .parse()
        .expect("lane count");
    let upstream_extras = parse_bool("P3_BB_UPSTREAM_EXTRAS", false);
    println!(
        "Plonky3 BabyBear MulAir: exponents={exponents:?} lanes={lanes} \
         extras={upstream_extras} reps={reps} (one excluded warmup; every sample verified)"
    );
    match lanes {
        16 => run::<16>(&exponents, reps, upstream_extras),
        20 => run::<20>(&exponents, reps, upstream_extras),
        _ => panic!("P3_BB_LANES must be 16 or 20"),
    }
}
