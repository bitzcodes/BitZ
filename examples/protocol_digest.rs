//! Byte-identity pin for the protocol proof streams (MultiSwap and SHA-256):
//! proves a fixed deterministic instance per protocol
//! and prints the BLAKE3 digest of the serialized proof body plus the
//! commitment root. Run before and after any change that claims to be
//! transcript-preserving — matching digests mean byte-identical proofs,
//! commitments, and transcripts. The companion `proof_digest` example pins
//! the base PCS opener alone.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example protocol_digest
//! ```

use ::bitz::piop::spartan::protocol;
use ::bitz::piop::spartan::protocol::PreparedRelation;
use bitz::piop::spartan::mul::{MulLayout, MulWitness};

use blake3::Hasher;
use bitz::piop::spartan::multiswap::{
    MultiswapAssignment, MultiswapCircuit, MultiswapDims, PreparedMultiswapRelation,
    commit_multiswap_witness, multiswap_lig_configs, prove_multiswap_mod_r1cs,
    verify_multiswap_mod_r1cs,
};
use bitz::piop::spartan::{
    Sha256CompressionStatement, commit_sha256_compression_witness,
    generate_sha256_compression_witnesses, prepare_sha256_compression_batch,
    prove_sha256_compressions, verify_sha256_compressions,
};
use bitz::transcript::Blake3Transcript;

fn digest_hex(parts: &[&[u8]]) -> String {
    let mut hasher = Hasher::new();
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().to_hex().to_string()
}

fn multiswap_digest() -> String {
    let circuit = MultiswapCircuit::build(MultiswapDims::mini()).expect("build circuit");
    circuit.is_sat_integer().expect("relation satisfied");
    let prepared = PreparedMultiswapRelation::new(&circuit).expect("prepare");
    let assignment = MultiswapAssignment::new(&circuit).expect("assignment");
    let (pc, vc) = multiswap_lig_configs(prepared.params()).expect("configs");
    let rows = assignment.bitz_bit_rows();
    let hint = commit_multiswap_witness(prepared.params(), rows, &pc).expect("commit");

    let mut prover_transcript = Blake3Transcript::new();
    let proof =
        prove_multiswap_mod_r1cs(&mut prover_transcript, &prepared, &assignment, &hint, &pc)
            .expect("prove");
    let mut verifier_transcript = Blake3Transcript::new();
    verify_multiswap_mod_r1cs(
        &mut verifier_transcript,
        &prepared,
        &hint.commitment,
        &proof,
        &vc,
    )
    .expect("verify");

    // Every transcript-visible proof component, framed.
    let bitz_bytes = proof.bitz().to_bytes();
    let mut mu_prime = [0u8; 40];
    field::CanonicalCodec::encode_into(
        &field::IntegerOps,
        proof.mu_prime().expect("lift"),
        &mut mu_prime,
    );
    let nonce = proof.reduction_nonce().expect("nonce").to_le_bytes();
    let spartan = format!("{:?}", proof.spartan());
    digest_hex(&[
        &hint.commitment.root,
        &bitz_bytes,
        &mu_prime,
        &nonce,
        spartan.as_bytes(),
    ])
}

fn sha256_digest() -> String {
    const EXPONENT: usize = 7;
    let prepared = prepare_sha256_compression_batch(EXPONENT).expect("prepare");
    let inputs: Vec<_> = (0..1usize << EXPONENT)
        .map(|i| {
            let word = |j: usize| (i as u32).wrapping_mul(0x9e37_79b9) ^ (j as u32);
            (
                std::array::from_fn(|j| word(j)),
                std::array::from_fn(|j| word(j + 16)),
            )
        })
        .collect();
    let witness = generate_sha256_compression_witnesses(&prepared, &inputs).expect("witness");
    let statements: Vec<_> = inputs
        .iter()
        .copied()
        .zip(witness.outputs().iter().copied())
        .map(|(input, output)| Sha256CompressionStatement::new(input, output))
        .collect();
    let hint = commit_sha256_compression_witness(&prepared, &witness).expect("commit");
    let mut prover_transcript = Blake3Transcript::new();
    let proof = prove_sha256_compressions(
        &mut prover_transcript,
        &prepared,
        &statements,
        &witness,
        &hint,
    )
    .expect("prove");
    let mut verifier_transcript = Blake3Transcript::new();
    verify_sha256_compressions(
        &mut verifier_transcript,
        &prepared,
        &statements,
        &hint.commitment,
        &proof,
    )
    .expect("verify");

    let bitz_bytes = proof.bitz().to_bytes();
    let inner = format!("{:?}", proof.inner());
    let nonces: Vec<u8> = proof
        .inner_nonces()
        .iter()
        .flat_map(|nonce| nonce.to_le_bytes())
        .chain(proof.initial_nonce().to_le_bytes())
        .chain(proof.terminal_nonce().to_le_bytes())
        .collect();
    digest_hex(&[&hint.commitment.root, &bitz_bytes, inner.as_bytes(), &nonces])
}

fn u32_mul_digest() -> String {
    let witness = MulWitness::<u32>::from_fn_with_word_bits(1usize << 15, 1, |i| {
        let x = (i as u32).wrapping_mul(0x9e37_79b9) | 1;
        let y = (i as u32).wrapping_mul(0x85eb_ca6b) | 1;
        (x, y)
    })
    .expect("witness");
    let layout = *witness.layout();
    let prepared = PreparedRelation::<MulLayout<u32>>::new(layout).expect("prepare");
    let hint = protocol::commit(&prepared, witness.bitz_bit_rows()).expect("commit");
    let mut prover_transcript = Blake3Transcript::new();
    let proof = protocol::prove(&mut prover_transcript, &prepared, &witness, &hint).expect("prove");
    let mut verifier_transcript = Blake3Transcript::new();
    protocol::verify(
        &mut verifier_transcript,
        &prepared,
        &hint.commitment,
        &proof,
    )
    .expect("verify");
    let bitz_bytes = proof.bitz().to_bytes();
    let spartan = format!("{:?}", proof.spartan());
    digest_hex(&[&hint.commitment.root, &bitz_bytes, spartan.as_bytes()])
}

fn main() {
    println!("multiswap-mini  {}", multiswap_digest());
    println!("sha256-2p7      {}", sha256_digest());
    println!("u32-mul-2p15    {}", u32_mul_digest());
}
