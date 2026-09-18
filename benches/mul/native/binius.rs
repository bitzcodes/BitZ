#[cfg(test)]
#[allow(unused_imports)]
use super::edge_corpus;
use super::trace_capture::TrialScopes;
use super::{Corpus, Timing, Workload, captured};
use binius_circuits::bignum::{self, BigUint};
use binius_core::{constraint_system::ValueVec, word::Word};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_hash::StdHashSuite;
use binius_prover::{OptimalPackedB128, Prover};
use binius_verifier::{
    Verifier,
    config::StdChallenger,
    transcript::{ProverTranscript, VerifierTranscript},
};
use bitz::observability::Recording;
use serde_json::{Value, json};
use std::sync::Arc;

pub(super) enum Wires {
    /// One 64-bit-or-narrower gate: operands, output, auxiliary value.
    Narrow {
        a: Wire,
        b: Wire,
        c: Wire,
        q: Option<Wire>,
    },
    /// One `u128` gate: two-limb operands and the four-limb product.
    Wide { x: BigUint, y: BigUint, z: BigUint },
}
pub(super) struct Context {
    corpus: Arc<Corpus>,
    circuit: Circuit,
    wires: Vec<Wires>,
    prover: Prover<OptimalPackedB128, StdHashSuite>,
    verifier: Verifier<StdHashSuite>,
}
impl Context {
    #[cfg(test)]
    pub(super) fn setup(corpus: Arc<Corpus>) -> Self {
        Self::setup_at_rate(corpus, 1)
    }
    pub(super) fn setup_at_rate(corpus: Arc<Corpus>, rate: usize) -> Self {
        let (circuit, wires) = compile(&corpus);
        let verifier = Verifier::<StdHashSuite>::setup_with_security_bits(
            circuit.constraint_system().clone(),
            rate,
            100,
        )
        .expect("Binius setup");
        let prover = Prover::setup(verifier.clone()).expect("Binius prover setup");
        Self {
            corpus,
            circuit,
            wires,
            prover,
            verifier,
        }
    }
    fn populate(&self, corrupt_output: bool) -> Result<ValueVec, String> {
        Ok(populate(&self.corpus, &self.circuit, &self.wires, corrupt_output)?.into_value_vec())
    }
    pub(super) fn config(&self) -> Value {
        let piop = match self.corpus.workload {
            Workload::U32 => {
                "Binius64 native multiplication with 32-bit inputs and low-32-bit result"
            }
            Workload::U64 => "Binius64 native 64 x 64 -> 128 integer multiplication",
            Workload::U128 => {
                "Binius64 bignum 128 x 128 -> 256 multiplication (four native imul limb products with carry chains)"
            }
        };
        let cs = self.circuit.constraint_system();
        json!({"piop":piop,"pcs":"ring switching/BaseFold","fri_query_target_bits":100,
            "security_scope":"fri-query-phase", "soundness_regime":"unique-decoding",
            "log_inv_rate":self.verifier.fri_params().rs_code().log_inv_rate(),
            "fri_queries":self.verifier.fri_params().n_test_queries(),
            "fri_fold_arities":self.verifier.fri_params().fold_arities(),
            "fri_log_message_len":self.verifier.fri_params().log_msg_len(),
            "fri_final_challenges":self.verifier.fri_params().n_final_challenges(),
            "hash":"SHA-256", "transcript":"StdChallenger", "fri_grinding_bits":0,
            "word_constraints":{"and":cs.n_and_constraints(),"imul":cs.n_imul_constraints(),
                "zero":cs.n_zero_constraints(),"bmul":cs.n_bmul_constraints()}})
    }
    pub(super) fn run(&self) -> Timing {
        let recording = Recording::start(Vec::new()).expect("start Perfetto trial");
        let proof_bytes = self.prove_and_verify();
        let raw = recording.intervals().expect("query Perfetto trial");
        let trial = TrialScopes::from_spans(&raw, "benchmark");
        let wend = trial.witness.end_ns;
        let ready = trial.witness_to_proof.end_ns;
        let mut t = Timing::from_trial(&trial, proof_bytes);
        let pack = captured(&raw, "prepare_witness", wend, ready);
        t.add(
            "witness_packing",
            "witness-generation",
            pack.start_ns,
            pack.end_ns,
        );
        let commit = captured(&raw, "commit_witness", wend, ready);
        let ring = captured(&raw, "ring_switching", wend, ready);
        t.add("commit", "commit", commit.start_ns, commit.end_ns);
        t.add("piop", "constraint-proof", commit.end_ns, ring.start_ns);
        t.add("opening", "opening-proof", ring.start_ns, ready);
        t
    }

    pub(super) fn prove_and_verify(&self) -> usize {
        let trial = tracing::info_span!(
            "Verified trial",
            component = "benchmark.verified-trial",
            scope_kind = "scope",
            tag_end_to_end = true
        )
        .entered();
        let proving = tracing::info_span!(
            "Witness to proof",
            component = "benchmark.witness-to-proof",
            scope_kind = "scope"
        )
        .entered();
        let witness_scope = tracing::info_span!(
            "Witness generation",
            component = "benchmark.witness-evaluation",
            scope_kind = "phase",
            tag_witness_generation = true
        )
        .entered();
        let witness = self.populate(false).expect("Binius witness evaluation");
        drop(witness_scope);
        let mut transcript = ProverTranscript::new(StdChallenger::default());
        self.prover
            .prove(&witness, &mut transcript)
            .expect("Binius full proof");
        let bytes = transcript.finalize();
        drop(proving);
        let verification = tracing::info_span!(
            "Verification",
            component = "benchmark.verification",
            scope_kind = "phase",
            tag_verification = true
        )
        .entered();
        let mut vt = VerifierTranscript::new(StdChallenger::default(), bytes.clone());
        self.verifier
            .verify(witness.inout(), &mut vt)
            .expect("Binius full verification");
        vt.finalize().expect("consume full Binius proof");
        drop(verification);
        drop(trial);
        let proof_bytes = bytes.len();
        proof_bytes
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    #[test]
    fn u32_mod32_constraints_and_bounds() {
        let corpus = Corpus::from_inputs(
            Workload::U32,
            vec![
                (0, 0),
                (0, u64::from(u32::MAX)),
                (1, u64::from(u32::MAX)),
                (u64::from(u32::MAX), u64::from(u32::MAX)),
                (1 << 31, 2),
            ],
        );
        let (circuit, wires) = compile(&corpus);
        let cs = circuit.constraint_system();
        assert_eq!(cs.n_imul_constraints(), corpus.len());
        // Current Binius emits one masked-product AND and linear range/equality
        // checks per row; validate bounds below independently of these cost counts.
        assert_eq!(cs.n_and_constraints(), corpus.len());
        assert_eq!(cs.n_zero_constraints(), 3 * corpus.len());
        assert_eq!(cs.n_bmul_constraints(), 0);
        let valid = populate(&corpus, &circuit, &wires, false).unwrap();
        cs.verify(valid.value_vec()).unwrap();
        let Wires::Narrow { a, b, c, .. } = wires[0] else {
            unreachable!()
        };
        // At 0*0=0, raising either operand to 2^32 still satisfies the
        // multiplication. Only the input bound can reject these witnesses.
        for (wire, value) in [(a, 1 << 32), (b, 1 << 32), (c, 1 << 32), (c, 1)] {
            let mut tampered = populate(&corpus, &circuit, &wires, false).unwrap();
            tampered[wire] = Word(value);
            assert!(cs.verify(tampered.value_vec()).is_err());
        }
        let Wires::Narrow { c, .. } = wires[3] else {
            unreachable!()
        };
        assert_eq!(valid[c].0, 1, "max*max wraps to one");
    }

    #[test]
    fn real_mod32_proof_rejects_tampered_transcript() {
        let corpus = Corpus::from_inputs(
            Workload::U32,
            (0..32).map(|i| (u64::from(u32::MAX) - i, i)).collect(),
        );
        let context = Context::setup(Arc::new(corpus));
        let witness = context.populate(false).unwrap();
        let mut transcript = ProverTranscript::new(StdChallenger::default());
        context.prover.prove(&witness, &mut transcript).unwrap();
        let mut bytes = transcript.finalize();
        let mut valid = VerifierTranscript::new(StdChallenger::default(), bytes.clone());
        context
            .verifier
            .verify(witness.inout(), &mut valid)
            .unwrap();
        valid.finalize().unwrap();
        bytes[0] ^= 1;
        let mut bad = VerifierTranscript::new(StdChallenger::default(), bytes);
        assert!(context.verifier.verify(witness.inout(), &mut bad).is_err());
    }

    #[test]
    fn wrong_product_is_rejected() {
        for workload in [Workload::U32, Workload::U64, Workload::U128] {
            let c = Context::setup(Arc::new(edge_corpus(workload)));
            // Population computes wires; acceptance is checked by the constraint verifier.
            let valid = c.populate(false).unwrap();
            c.circuit.constraint_system().verify(&valid).unwrap();
            assert!(c.populate(true).is_err());
        }
    }
}

pub(super) fn compile(corpus: &Corpus) -> (Circuit, Vec<Wires>) {
    let builder = CircuitBuilder::new();
    let mask = builder.add_constant_64(u64::from(u32::MAX));
    let wires = (0..corpus.len())
        .map(|i| {
            let b = builder.subcircuit(format!("multiply[{i}]"));
            if corpus.workload.is_wide() {
                // Native 128 x 128 -> 256 through Binius64's bignum circuit:
                // four `imul` limb products accumulated with carry chains,
                // asserted equal to the four witness limbs of the product.
                let x = BigUint::new_witness(&b, 2);
                let y = BigUint::new_witness(&b, 2);
                let z = BigUint::new_witness(&b, 4);
                let product = bignum::textbook_mul(&b, &x, &y);
                bignum::assert_eq(&b, "product limb", &product, &z);
                return Wires::Wide { x, y, z };
            }
            let a = b.add_witness();
            let rhs = b.add_witness();
            let c = b.add_witness();
            let (hi, lo) = b.imul(a, rhs);
            if corpus.workload == Workload::U64 {
                // Native 64 x 64 -> 128: both product words are witness
                // values and the operands are full words, so no range checks.
                let z_hi = b.add_witness();
                b.assert_eq("product low word", lo, c);
                b.assert_eq("product high word", hi, z_hi);
                return Wires::Narrow {
                    a,
                    b: rhs,
                    c,
                    q: Some(z_hi),
                };
            }
            b.assert_zero("a is u32", b.shr(a, 32));
            b.assert_zero("b is u32", b.shr(rhs, 32));
            let masked = b.band(lo, mask);
            b.assert_eq("product modulo 2^32", masked, c);
            Wires::Narrow {
                a,
                b: rhs,
                c,
                q: None,
            }
        })
        .collect();
    let circuit = builder.build();
    (circuit, wires)
}

pub(super) fn populate<'a>(
    corpus: &Corpus,
    circuit: &'a Circuit,
    wires: &[Wires],
    corrupt_output: bool,
) -> Result<binius_frontend::WitnessFiller<'a>, String> {
    let mut filler = circuit.new_witness_filler();
    let corrupt = |i: usize| u64::from(corrupt_output && i == 0);
    match &corpus.operands {
        super::Operands::Narrow(inputs) => {
            for (i, (w, &(a, b))) in wires.iter().zip(inputs).enumerate() {
                let Wires::Narrow {
                    a: wa,
                    b: wb,
                    c: wc,
                    q: wq,
                } = w
                else {
                    unreachable!("narrow corpus with wide wires")
                };
                let [_, _, c, q] = corpus.workload.native_row(a, b);
                filler[*wa] = Word(a);
                filler[*wb] = Word(b);
                filler[*wc] = Word(c ^ corrupt(i));
                if let Some(wire) = *wq {
                    filler[wire] = Word(q);
                }
            }
        }
        super::Operands::Wide(inputs) => {
            for (i, (w, &(x, y))) in wires.iter().zip(inputs).enumerate() {
                let Wires::Wide {
                    x: wx,
                    y: wy,
                    z: wz,
                } = w
                else {
                    unreachable!("wide corpus with narrow wires")
                };
                let [_, _, lo, hi] = corpus.workload.wide_row(x, y);
                wx.populate_limbs(&mut filler, &limbs(x));
                wy.populate_limbs(&mut filler, &limbs(y));
                let [z0, z1] = limbs(lo);
                let [z2, z3] = limbs(hi);
                wz.populate_limbs(&mut filler, &[z0 ^ corrupt(i), z1, z2, z3]);
            }
        }
    }
    circuit
        .populate_wire_witness(&mut filler)
        .map_err(|error| error.to_string())?;
    Ok(filler)
}

/// The two 64-bit limbs of a `u128`, least significant first.
fn limbs(value: u128) -> [u64; 2] {
    [value as u64, (value >> 64) as u64]
}

pub(super) fn audit(corpus: &Corpus) -> super::WitnessAudit {
    let (circuit, wires) = compile(corpus);
    let started = std::time::Instant::now();
    let filler = populate(corpus, &circuit, &wires, false).expect("Binius materialization");
    let generation_ms = started.elapsed().as_secs_f64() * 1000.;
    if corpus.workload.is_wide() {
        let read = |limbs: &[Wire]| {
            limbs.iter().rev().fold(0_u128, |acc, &limb| {
                (acc << 64) | u128::from(filler[limb].0)
            })
        };
        let rows = wires
            .iter()
            .map(|w| {
                let Wires::Wide { x, y, z } = w else {
                    unreachable!("wide corpus with narrow wires")
                };
                [
                    read(&x.limbs),
                    read(&y.limbs),
                    read(&z.limbs[..2]),
                    read(&z.limbs[2..]),
                ]
            })
            .collect();
        return super::WitnessAudit::check_wide(corpus, rows, generation_ms, "Binius wire values");
    }
    let rows = wires
        .iter()
        .map(|w| {
            let Wires::Narrow { a, b, c, q } = w else {
                unreachable!("narrow corpus with wide wires")
            };
            [
                filler[*a].0,
                filler[*b].0,
                filler[*c].0,
                q.map_or(0, |q| filler[q].0),
            ]
        })
        .collect();
    super::WitnessAudit::check(corpus, rows, generation_ms, "Binius wire values", false)
}
