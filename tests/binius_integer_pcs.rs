#[path = "../benches/integer_pcs_compare/binius.rs"]
mod binius;

#[test]
fn binius_opens_exact_packed_integer_rows() {
    for (log_rows, rate) in [(4, 1), (9, 2)] {
        let backend = binius::BiniusBackend::setup(log_rows, rate);
        let trial = backend
            .run_trial(
                || {
                    (0..1usize << log_rows)
                        .map(|i| {
                            // Exercise every packed bit, including bit 127, and zero/all-one rows.
                            match i % 4 {
                                0 => 0,
                                1 => u128::MAX,
                                _ => (i as u128).wrapping_mul(0x9e3779b97f4a7c15d6e8feb86659fd93),
                            }
                        })
                        .collect()
                },
                42,
            )
            .unwrap();
        assert!(trial.proof_bytes > trial.commitment_bytes);
        assert_eq!(trial.public_claim_bytes, (log_rows + 8) * 16);
    }
}
