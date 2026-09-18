//! End-to-end coverage of the B.6 forest/opening grinding hooks: at
//! `Lambda128` every challenge drawn in the BitZ opening region carries a
//! two-bit proof-of-work boundary and the nonces ride the proof stream.

use ::bitz::piop::spartan::protocol::linear::LinearProof;

use bitz::ligerito_flock::IntEvalRsLigVirtProof;
use bitz::piop::spartan::{
    Lambda128, PreparedSha256CompressionBatch, Sha128ReferenceSchedule, Sha256CompressionStatement,
    commit_sha256_compression_witness_with_config, generate_sha256_compression_witnesses,
    prepare_sha256_compression_batch_with_profile, prove_sha256_compressions_with_config,
    sha256_compression_configs, verify_sha256_compressions_with_config,
};
use bitz::transcript::Blake3Transcript;

const EXPONENT: usize = 7;

fn prove_under(
    prepared: &PreparedSha256CompressionBatch,
) -> (
    Vec<u8>,
    usize,
    LinearProof,
    flock_core::pcs::commit::Commitment,
    flock_core::pcs::ligerito::VerifierConfig,
    Vec<Sha256CompressionStatement>,
) {
    let (pc, vc) = sha256_compression_configs(prepared).expect("configs");
    let inputs: Vec<_> = (0..1usize << EXPONENT)
        .map(|i| {
            let word = |j: usize| (i as u32).wrapping_mul(0x9e37_79b9) ^ (j as u32);
            (
                std::array::from_fn(|j| word(j)),
                std::array::from_fn(|j| word(j + 16)),
            )
        })
        .collect();
    let witness = generate_sha256_compression_witnesses(prepared, &inputs).expect("witness");
    let statements: Vec<_> = inputs
        .iter()
        .copied()
        .zip(witness.outputs().iter().copied())
        .map(|(input, output)| Sha256CompressionStatement::new(input, output))
        .collect();
    let hint =
        commit_sha256_compression_witness_with_config(prepared, &witness, &pc).expect("commit");
    let mut prover_transcript = Blake3Transcript::new();
    let proof = prove_sha256_compressions_with_config(
        &mut prover_transcript,
        prepared,
        &statements,
        &witness,
        &hint,
        &pc,
    )
    .expect("prove");
    let mut verifier_transcript = Blake3Transcript::new();
    verify_sha256_compressions_with_config(
        &mut verifier_transcript,
        prepared,
        &statements,
        &hint.commitment,
        &proof,
        &vc,
    )
    .expect("verify");
    let bytes = proof.bitz().to_bytes();
    let nonces = proof.bitz().grinding_nonces.len();
    (bytes, nonces, proof, hint.commitment, vc, statements)
}

#[test]
fn lambda128_grinds_every_opening_round_and_gates_the_nonces() {
    // The 0-difficulty baseline at the SAME Ligerito target (128), so the
    // size comparison isolates the nonce section.
    let reference =
        prepare_sha256_compression_batch_with_profile::<Sha128ReferenceSchedule>(EXPONENT)
            .expect("reference-schedule prepare");
    let lambda128 = prepare_sha256_compression_batch_with_profile::<Lambda128>(EXPONENT)
        .expect("lambda128 prepare");
    assert_eq!(lambda128.security().forest_round_grinding_bits, 2);

    let (reference_bytes, reference_nonces, ..) = prove_under(&reference);
    assert_eq!(
        reference_nonces, 0,
        "the reference schedule does not grind the opening"
    );

    let (bytes, nonce_count, proof, commitment, vc, statements) = prove_under(&lambda128);
    assert!(nonce_count > 0, "λ=128 must grind the opening rounds");
    // The proof grows by the trailing nonce section (8-byte count prefix +
    // 8 bytes per boundary) — grinding costs bytes, not time. The residual
    // difference is Ligerito query-path wiggle: the two runs draw different
    // challenges, so Merkle path deduplication differs by a few hundred
    // bytes in either direction.
    let section = 8 + 8 * nonce_count;
    let delta = bytes.len() as i64 - reference_bytes.len() as i64;
    assert!(
        (delta - section as i64).unsigned_abs() < 2048,
        "delta {delta} B vs nonce section {section} B ({nonce_count} boundaries)"
    );
    println!(
        "forest grinding at 2^{EXPONENT}: {nonce_count} grinded boundaries, \
         nonce section {section} B, total proof delta {delta} B"
    );

    // Codec round-trip preserves the section byte-for-byte.
    let decoded = IntEvalRsLigVirtProof::from_bytes(&bytes).expect("codec");
    assert_eq!(decoded.to_bytes(), bytes);
    assert_eq!(decoded.grinding_nonces.len(), nonce_count);

    // A λ=128 proof does not verify under the reference (0-difficulty)
    // preparation: the grinding boundaries are transcript-visible.
    let mut verifier_transcript = Blake3Transcript::new();
    assert!(
        verify_sha256_compressions_with_config(
            &mut verifier_transcript,
            &reference,
            &statements,
            &commitment,
            &proof,
            &vc,
        )
        .is_err()
    );
}
