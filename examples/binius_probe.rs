//! Phase profile of one all-Binius64 prove: the `all-binius` circuit of
//! `benches/hybrid_u32_sha256` (N four-limb mod-2^32 multiplication gadgets and
//! M chained SHA-256 compressions) proved by Binius64's own prover, with its
//! tracing spans queried from Perfetto and printed as interval-union totals. One
//! warm-up prove is discarded, then `PROBE_REPS` (default 1) profiled proves
//! per shape; every profiled proof is verified.
//!
//! ```text
//! PROBE_SHAPES="14:14" RAYON_NUM_THREADS=8 BITZ_HYBRID_BINIUS_LOG_INV_RATE=3 \
//!   BITZ_HYBRID_BINIUS_SECURITY_BITS=100 RUSTFLAGS="-C target-cpu=native" \
//!   cargo run --release --example binius_probe --features hybrid,span-metrics
//! ```
use binius_circuits::sha256::compress::{State, sha256_compress_2x_seq};
use binius_core::word::Word;
use binius_frontend::CircuitBuilder;
use binius_hash::Blake3HashSuite;
use binius_prover::{OptimalPackedB128, Prover};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
use binius_verifier::Verifier;
use bitz::hybrid::chaining_value;
use bitz::piop::spartan::MulRow;

type Challenger = HasherChallenger<blake3::Hasher>;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    bitz::observability::install().expect("install Perfetto subscriber");
    let shapes: Vec<(u32, u32)> = std::env::var("PROBE_SHAPES")
        .map(|v| {
            v.split_whitespace()
                .map(|p| {
                    let mut it = p.split(':');
                    (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|_| vec![(14, 14)]);
    let reps = env_usize("PROBE_REPS", 1);
    let log_inv_rate = env_usize("BITZ_HYBRID_BINIUS_LOG_INV_RATE", 3);
    let security_bits = env_usize("BITZ_HYBRID_BINIUS_SECURITY_BITS", 100);
    for (mul_log, sha_log) in shapes {
        let multiplications = 1usize << mul_log;
        let compressions = 1usize << sha_log;
        let t0_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let t0 = tracing::info_span!("binius_probe:t0").entered();
        // The `all-binius` circuit of the bench runner.
        let builder = CircuitBuilder::new();
        let mul_wires: Vec<_> = (0..multiplications)
            .map(|_| bitz::hybrid::mod32_binius::add_u32_mul_mod32(&builder))
            .collect();
        let block_wires: Vec<[_; 16]> = (0..compressions)
            .map(|_| std::array::from_fn(|_| builder.add_witness()))
            .collect();
        let output: [_; 8] = std::array::from_fn(|_| builder.add_inout());
        let mut state = State::iv(&builder);
        for pair in block_wires.chunks_exact(2) {
            state = sha256_compress_2x_seq(&builder, state, [pair[0], pair[1]]);
        }
        let mask = builder.add_constant(Word(u32::MAX as u64));
        for (actual, expected) in state.0.into_iter().zip(output) {
            builder.assert_eq("final_sha_chaining_value", builder.band(actual, mask), expected);
        }
        let circuit = builder.build();
        let verifier = Verifier::<Blake3HashSuite>::setup_with_security_bits(
            circuit.constraint_system().clone(),
            log_inv_rate,
            security_bits,
        )
        .expect("verifier setup");
        let prover = Prover::<OptimalPackedB128, Blake3HashSuite>::setup(verifier.clone()).expect("prover setup");
        eprintln!(
            "setup {mul_log}:{sha_log} {:.0} ms (log_inv_rate={log_inv_rate}, security_bits={security_bits})",
            { drop(t0); bitz::observability::duration(&t0_recording.intervals().expect("complete operation capture"), "binius_probe:t0").expect("query completed operation") }.as_secs_f64() * 1e3
        );
        let inputs: Vec<_> = (0..multiplications as u32)
            .map(|i| (i.wrapping_mul(0x9e3779b9), u32::MAX - i))
            .collect();
        let blocks: Vec<[u32; 16]> = (0..compressions as u32)
            .map(|i| std::array::from_fn(|j| i.wrapping_mul(0x85ebca6b).wrapping_add(j as u32)))
            .collect();
        for rep in 0..=reps {
            let start_recording =
                bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let start = tracing::info_span!("binius_probe:start").entered();
            let rows: Vec<_> = inputs
                .iter()
                .map(|&(x, y)| MulRow::<u32>::new(x, y))
                .collect();
            let mut filler = circuit.new_witness_filler();
            for (wires, row) in mul_wires.iter().zip(&rows) {
                for (&wire, value) in wires.iter().zip([row.x, row.y, row.lo, row.hi]) {
                    filler[wire] = Word(value as u64);
                }
            }
            for (wires, block) in block_wires.iter().zip(&blocks) {
                for (&wire, &word) in wires.iter().zip(block) {
                    filler[wire] = Word(word as u64);
                }
            }
            for (&wire, word) in output.iter().zip(chaining_value(&blocks)) {
                filler[wire] = Word(word as u64);
            }
            circuit.populate_wire_witness(&mut filler).expect("witness");
            let witness = filler.into_value_vec();
            let witness_ms = { drop(start); bitz::observability::duration(&start_recording.intervals().expect("complete operation capture"), "binius_probe:start").expect("query completed operation") }.as_secs_f64() * 1e3;
            let t1_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t1 = tracing::info_span!("binius_probe:t1").entered();
            let label = if rep == 0 { "warmup (discard)".to_string() } else { format!("rep {rep}") };
            eprintln!("--- {mul_log}:{sha_log} {label}: prove spans follow");
            let mut t = ProverTranscript::new(Challenger::default());
            prover.prove(&witness, &mut t).expect("prove");
            let bytes = t.finalize();
            drop(t1);
            let prove_intervals = t1_recording.intervals().expect("complete prove capture");
            let prove_ms = bitz::observability::duration(&prove_intervals, "binius_probe:t1")
                .expect("query completed prove").as_secs_f64() * 1e3;
            bitz::observability::write_profile(std::io::stderr().lock(), &label, &prove_intervals, None)
                .expect("write prove profile");
            let t2_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
            let t2 = tracing::info_span!("binius_probe:t2").entered();
            let mut vt = VerifierTranscript::new(Challenger::default(), bytes.clone());
            verifier.verify(witness.inout(), &mut vt).expect("verify");
            vt.finalize().expect("finalize");
            let verify_ms = { drop(t2); bitz::observability::duration(&t2_recording.intervals().expect("complete operation capture"), "binius_probe:t2").expect("query completed operation") }.as_secs_f64() * 1e3;
            eprintln!(
                "=== {mul_log}:{sha_log} {label}: witness {witness_ms:.1} ms + prove {prove_ms:.1} ms = {:.1} ms, verify {verify_ms:.1} ms, proof {} B",
                witness_ms + prove_ms,
                bytes.len()
            );
        }
    }
}
