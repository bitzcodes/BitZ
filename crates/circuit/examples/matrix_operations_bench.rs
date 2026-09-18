//! Sparse conversion, local SHA contraction, and binary opening-map adjoints.
use circuit::constraints::ConstraintGenerator;
use circuit::linear_map::contraction::PreparedSignedSparse;
use circuit::linear_map::{CscMatrix, LeftMul};
use circuit::matrix_transpose::MTransposeGenerator;
use circuit::sha256::{
    COMPRESSION_INPUT_BITS, SHA256_2KB_MESSAGE_BITS, compression_circuit, sha256_2kb_circuit,
};
use field::{Gf128, IntegerEmbedding, RingOps, Uint, create_prime_field};
use std::{hint::black_box, time::Instant};
#[cfg(feature = "bench-memory")]
#[path = "support/allocations.rs"]
mod allocations;
#[cfg(feature = "bench-memory")]
#[global_allocator]
static ALLOC: allocations::Counted = allocations::Counted;
fn report<I>(name: &str, samples: usize, mut prepare: impl FnMut() -> I, mut run: impl FnMut(I)) {
    if std::env::var("PHASE").is_ok_and(|phase| phase != name) {
        run(prepare());
        return;
    }
    for _ in 0..5 {
        run(prepare());
    }
    #[cfg(feature = "bench-memory")]
    {
        let input = prepare();
        let (count, bytes, peak) = allocations::measure(|| run(input));
        println!("{name},allocations={count},allocated_bytes={bytes},extra_peak_bytes={peak}");
    }
    #[cfg(not(feature = "bench-memory"))]
    {
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let input = prepare();
            let start = Instant::now();
            run(input);
            times.push(start.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{name},median_us={:.3},p10_us={:.3},p90_us={:.3}",
            times[samples / 2],
            times[samples / 10],
            times[samples * 9 / 10]
        );
    }
}
fn main() {
    let samples = std::env::var("SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let mut generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
    let inputs = generator.inputs();
    let _ = compression_circuit(&mut generator, &inputs);
    let matrices = generator.into_matrices();
    let rows = matrices
        .c
        .rows()
        .map(|row| {
            row.iter()
                .map(|(column, c)| (column, c.as_words()[0] as i64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let columns = matrices.c.column_count();
    report(
        "sha_signed_csc_conversion",
        samples,
        || rows.clone(),
        |rows| {
            black_box(CscMatrix::<Box<[_]>>::try_from_rows(columns, rows).unwrap());
        },
    );
    let matrix = CscMatrix::<Box<[_]>>::try_from_rows(columns, rows.clone()).unwrap();
    let field = create_prime_field(Uint::<2>::from((1u128 << 127) - 1));
    let mut weights = vec![field.one()];
    for i in 0..8 {
        let r = field.from_integer(&(i as u64 + 3));
        let n = weights.len();
        for j in 0..n {
            let right = field.mul(&weights[j], &r);
            weights[j] = field.sub(&weights[j], &right);
            weights.push(right);
        }
    }
    let mut output = field.zero_vec(columns);
    report(
        "sha_signed_local_binding",
        samples,
        || (),
        |_| {
            PreparedSignedSparse::new(&field, &matrix)
                .mul_left_into(&weights[..matrix.row_count()], &mut output)
                .unwrap();
            black_box(&output);
        },
    );
    let mut expected = field.zero_vec(columns);
    for (row, entries) in rows.iter().enumerate() {
        for &(column, c) in entries {
            expected[column] = field.add(
                &expected[column],
                &field.mul(&weights[row], &field.from_integer(&c)),
            );
        }
    }
    assert_eq!(output, expected);
    let mut generator = MTransposeGenerator::new(SHA256_2KB_MESSAGE_BITS);
    let inputs = generator.take_boxed_inputs();
    let _ = sha256_2kb_circuit(&mut generator, &inputs);
    let mut binary = generator.finish();
    let weights = (0..binary.row_count())
        .map(|i| Gf128::new(i as u64, i as u64 * 17))
        .collect::<Vec<_>>();
    let mut out = vec![Gf128::ZERO; binary.column_count()];
    report(
        "sha_binary_adjoint_reused",
        samples,
        || (),
        |_| {
            binary.mul_left_into(&weights, &mut out).unwrap();
            black_box(&out);
        },
    );
}
