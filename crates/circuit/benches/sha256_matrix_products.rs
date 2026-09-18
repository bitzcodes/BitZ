//! Witgen, `M`, `ABC(Mw)`, `rM`, and Wengert `r(A+xB+x²C)` for SHA-256.
//!
//! Run with `cargo bench -p circuit --bench sha256_matrix_products`.

mod support;

use circuit::linear_map::circuit::{WengertGenerator, WengertTape};
use circuit::linear_map::{CscMatrix, ImplicitOnes, LeftMul};
use circuit::matrix_transpose::MTransposeGenerator;
use circuit::sha256::{
    SHA256_2KB_MESSAGE_BITS, SHA256_2KB_WITNESS_BITS, block_aligned_witness_bits,
    sha256_2kb_circuit, sha256_block_aligned_circuit,
};
use circuit::witgen::ProductWitgen;
use divan::{Bencher, black_box};
use field::Gf128;
use field::{ModRingCtx, Uint};
use num_bigint::BigUint;
use num_traits::One;
use std::sync::Once;

static REPORT_WENGERT_TAPE: Once = Once::new();
static REPORT_M_TRANSPOSE: Once = Once::new();

fn main() {
    divan::main();
}

fn build_transpose() -> CscMatrix<ImplicitOnes, u32> {
    let mut generator = MTransposeGenerator::new(SHA256_2KB_MESSAGE_BITS);
    let inputs = generator.take_boxed_inputs();
    let _ = sha256_2kb_circuit(&mut generator, &inputs);
    generator.finish()
}

fn report_m_transpose(transpose: &CscMatrix<ImplicitOnes, u32>) {
    let bytes = transpose.topology_bytes();
    REPORT_M_TRANSPOSE.call_once(|| {
        eprintln!(
            "SHA-256 2 KiB M (stored as M^T): {bytes} bytes ({:.2} MiB), {} nonzeros, \
             {} M rows x {} M columns",
            bytes as f64 / (1024.0 * 1024.0),
            transpose.nnz(),
            transpose.row_count(),
            transpose.column_count(),
        );
    });
}

fn build_wengert_tape() -> WengertTape {
    let mut generator = WengertGenerator::new(SHA256_2KB_MESSAGE_BITS);
    let inputs = generator.take_boxed_inputs();
    let _ = sha256_2kb_circuit(&mut generator, &inputs);
    generator.finish()
}

fn report_wengert_tape(tape: &WengertTape) {
    let bytes = tape.payload_bytes();
    REPORT_WENGERT_TAPE.call_once(|| {
        eprintln!(
            "SHA-256 2 KiB Wengert tape: {bytes} bytes ({:.2} MiB), {} nodes, {} edges, \
             {} reverse levels, {} coefficients, {} rows x {} columns",
            bytes as f64 / (1024.0 * 1024.0),
            tape.node_count(),
            tape.edge_count(),
            tape.level_count(),
            tape.coefficient_count(),
            tape.row_count(),
            tape.column_count(),
        );
    });
}

fn prime_128() -> BigUint {
    (BigUint::one() << 128_usize) - BigUint::from(159_u64)
}

fn prepare_parallel_reduction() {
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});
}

/// Generate `w`, `M * w`, and exact integer `A/B/C(M * w)`.
#[divan::bench]
fn sha256_2kb_witgen(bencher: Bencher) {
    let message = support::sha256::message_2kb();
    bencher.bench_local(|| {
        let mut witgen =
            ProductWitgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
        let digest = sha256_2kb_circuit(&mut witgen, black_box(message.as_ref()));
        black_box((digest, witgen.into_parts()))
    });
}

const SHA256_1_MIB_BITS: usize = 1024 * 1024 * 8;
const SHA256_1_MIB_WITNESS_BITS: usize = block_aligned_witness_bits(SHA256_1_MIB_BITS);

#[divan::bench(sample_count = 10, sample_size = 1)]
fn sha256_1_mib_witgen(bencher: Bencher) {
    let message = vec![0_u64; SHA256_1_MIB_BITS / 64];
    bencher.bench_local(|| {
        let message = black_box(&message);
        let mut witgen = ProductWitgen::with_packed_inputs_and_capacity(
            message,
            SHA256_1_MIB_BITS,
            SHA256_1_MIB_WITNESS_BITS,
        );
        let digest = sha256_block_aligned_circuit(&mut witgen, SHA256_1_MIB_BITS, |index| {
            message[index / 64] >> (index % 64) & 1 == 1
        });
        assert_eq!(witgen.witness().bit_len(), SHA256_1_MIB_WITNESS_BITS);
        black_box((digest, witgen.into_parts()))
    });
}

/// Generate the sparse representation of `M` used to compute `rM`.
#[divan::bench]
fn sha256_2kb_m(bencher: Bencher) {
    let reference_transpose = build_transpose();
    report_m_transpose(&reference_transpose);
    bencher.bench_local(|| {
        let mut generator = MTransposeGenerator::new(SHA256_2KB_MESSAGE_BITS);
        let inputs = generator.take_boxed_inputs();
        let digest = sha256_2kb_circuit(&mut generator, &inputs);
        black_box((digest, generator.finish()))
    });
}

/// Preprocess the modulus-independent tape for `r * (A + x B + x^2 C)`.
#[divan::bench]
fn sha256_2kb_abc_wengert_tape(bencher: Bencher) {
    let reference_tape = build_wengert_tape();
    report_wengert_tape(&reference_tape);
    bencher.bench_local(|| {
        let mut generator = WengertGenerator::new(SHA256_2KB_MESSAGE_BITS);
        let inputs = generator.take_boxed_inputs();
        let digest = sha256_2kb_circuit(&mut generator, &inputs);
        black_box((digest, generator.finish()))
    });
}

/// Reduce precomputed exact integer `A(M*w)`, `B(M*w)`, and `C(M*w)` values.
#[divan::bench]
fn sha256_2kb_abc_mw(bencher: Bencher) {
    prepare_parallel_reduction();
    let message = support::sha256::message_2kb();
    let mut witgen =
        ProductWitgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
    let _ = sha256_2kb_circuit(&mut witgen, &message);
    let (_, _, integer_products) = witgen.into_parts();
    let modulus = ModRingCtx::<2>::new(Uint::from_words([u64::MAX - 158, u64::MAX])).unwrap();
    bencher.bench_local(|| {
        black_box(black_box(&integer_products).reduce_parallel(black_box(&modulus)))
    });
}

#[divan::bench]
fn sha256_2kb_rm(bencher: Bencher) {
    prepare_parallel_reduction();
    let transpose = build_transpose();
    report_m_transpose(&transpose);
    let challenges: Vec<_> = (0..transpose.row_count())
        .map(|index| {
            Gf128::new(
                (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
            )
        })
        .collect();
    bencher.bench_local(|| black_box(transpose.mul_left(black_box(&challenges)).unwrap()));
}

/// Apply the preprocessed reverse-mode tape modulo a runtime prime.
#[divan::bench]
fn sha256_2kb_rabc(bencher: Bencher) {
    prepare_parallel_reduction();
    let tape = build_wengert_tape();
    report_wengert_tape(&tape);
    let challenges: Vec<_> = (0..tape.row_count())
        .map(|index| {
            [
                (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
            ]
        })
        .collect();
    let x = [0x243f_6a88_85a3_08d3, 0x1319_8a2e_0370_7344];
    let modulus = ModRingCtx::<2>::new(Uint::from_words([u64::MAX - 158, u64::MAX])).unwrap();
    let encoder = tape.prepare(&modulus).unwrap();
    let challenges: Vec<_> = challenges
        .into_iter()
        .map(|challenge| encoder.to_montgomery(challenge))
        .collect();
    let x = encoder.to_montgomery(x);
    drop(encoder);
    bencher.bench_local(|| {
        let mut evaluator = tape.prepare(black_box(&modulus)).unwrap();
        let field = evaluator.field();
        let x = field.from_montgomery_integer(field::Uint::from_words(black_box(x)));
        let x2 = field::RingOps::square(field, &x);
        let weights: Vec<_> = black_box(&challenges)
            .iter()
            .flat_map(|&words| {
                let w = field.from_montgomery_integer(field::Uint::from_words(words));
                [
                    w,
                    field::RingOps::mul(field, &w, &x),
                    field::RingOps::mul(field, &w, &x2),
                ]
            })
            .collect();
        let mut output = field::RingOps::zero_vec(field, tape.column_count());
        evaluator.mul_left_into(&weights, &mut output).unwrap();
        black_box(output);
    });
}
