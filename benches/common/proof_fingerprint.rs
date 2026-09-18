//! Diagnostic encoding of every proof message, outside measured boundaries.
//! This is deliberately not a new public wire format.
use bitz::{
    piop::spartan::{
        protocol::{OpeningProof, Proof, SpartanProof, linear::LinearProof},
        sumcheck::{OuterSumcheckProof, SumcheckProof},
    },
    transcript::Blake3Transcript,
};
use field::Fp;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProofFingerprint {
    pub proof: String,
    pub transcript: String,
}
struct Encoder(blake3::Hasher);
impl Encoder {
    fn word(&mut self, value: u64) {
        self.0.update(&value.to_le_bytes());
    }
    fn bytes(&mut self, value: &[u8]) {
        self.word(value.len() as u64);
        self.0.update(value);
    }
    fn field(&mut self, value: &Fp<2>) {
        for &word in value.as_montgomery_integer().as_words() {
            self.word(word);
        }
    }
    fn nonces(&mut self, values: &[u64]) {
        self.word(values.len() as u64);
        for &value in values {
            self.word(value);
        }
    }
    fn rounds<const N: usize>(&mut self, proof: &SumcheckProof<Fp<2>, N>) {
        self.word(proof.round_polynomials.len() as u64);
        for round in &proof.round_polynomials {
            for value in round {
                self.field(value);
            }
        }
    }
    fn outer(&mut self, proof: &OuterSumcheckProof<Fp<2>>) {
        self.rounds(&proof.sumcheck);
        self.field(&proof.az_mle_claim);
        self.field(&proof.bz_mle_claim);
        self.field(&proof.cz_mle_claim);
    }
    fn finish(self, transcript: &Blake3Transcript) -> ProofFingerprint {
        ProofFingerprint {
            proof: self.0.finalize().to_hex().to_string(),
            transcript: blake3::Hash::from(transcript.state_digest())
                .to_hex()
                .to_string(),
        }
    }
}
fn enabled() -> bool {
    std::env::var("BITZ_BENCH_PROOF_FINGERPRINT").is_ok_and(|v| v == "1")
}
pub fn nonlinear<O: OpeningProof>(proof: &Proof<O>, root: &[u8], transcript: &Blake3Transcript) {
    if enabled() {
        println!(
            "PROOF_FINGERPRINT {}",
            serde_json::to_string(&nonlinear_fingerprint(proof, root, transcript)).unwrap()
        );
    }
}
pub fn nonlinear_fingerprint<O: OpeningProof>(
    proof: &Proof<O>,
    root: &[u8],
    transcript: &Blake3Transcript,
) -> ProofFingerprint {
    let mut out = Encoder(blake3::Hasher::new());
    out.bytes(b"benchmark/nonlinear-proof/v1");
    out.bytes(root);
    out.word(proof.initial_nonce());
    out.nonces(proof.piop_nonces());
    match proof.spartan() {
        SpartanProof::Plain(p) => {
            out.word(0);
            out.outer(&p.outer);
            out.rounds(&p.inner);
        }
        SpartanProof::UnivariateSkip(p) => {
            out.word(1);
            out.word(p.outer.skip.skip_vars.into());
            out.word(p.outer.skip.finite_q_evaluations.len() as u64);
            for value in &p.outer.skip.finite_q_evaluations {
                out.field(value);
            }
            out.field(&p.outer.skip.q_at_infinity);
            out.outer(&p.outer.tail);
            out.rounds(&p.inner);
        }
    }
    out.word(proof.terminal_nonce());
    if let Some(reduction) = proof.reduction() {
        out.word(1);
        for &word in reduction.mu_prime.as_words() {
            out.word(word);
        }
        out.word(reduction.nonce);
    } else {
        out.word(0);
    }
    out.bytes(&proof.bitz().to_bytes());
    out.finish(transcript)
}
pub fn linear(proof: &LinearProof, root: &[u8], transcript: &Blake3Transcript) {
    if !enabled() {
        return;
    }
    let mut out = Encoder(blake3::Hasher::new());
    out.bytes(b"benchmark/linear-proof/v1");
    out.bytes(root);
    out.word(proof.initial_nonce());
    out.rounds(proof.inner());
    out.nonces(proof.inner_nonces());
    out.word(proof.terminal_nonce());
    out.bytes(&proof.bitz().to_bytes());
    println!(
        "PROOF_FINGERPRINT {}",
        serde_json::to_string(&out.finish(transcript)).unwrap()
    );
}
