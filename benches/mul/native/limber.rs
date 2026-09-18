//! Limber's own wrapping-multiplication arithmetization, proved in process.
//!
//! The program is the one Limber's `examples/int_mult.rs` uses: one modular
//! row `x * y = z_lo (mod 2^w)` per gate, whose quotient is the high half of
//! the exact `2w`-bit product. Every committed value — both operands, the low
//! half and the quotient — is then below `2^w`, so the IntEval limb range
//! check the Mod-PCS already runs *is* the operand range check and the program
//! needs no bit columns of its own.
//!
//! Unlike `int_mult` the gates are independent (it chains `a_{i+1} = c_i`),
//! because this corpus is independent operand pairs and every other backend
//! here proves independent gates.
//!
//! The commitment is Brakedown, not Hyrax: the crate documents
//! `T256DynPrimeBdEngine` as "the comparison instantiation against
//! code-commitment systems (fast prover, large proofs)", and it shares the
//! scalar field and prime sampling with the Hyrax engine, so only the
//! commitment scheme differs. Of its layout knobs (`BDLAMBDA`, `BDSPEC`,
//! `BDROWLEN`, `BDDIRECT`, `BDSPLIT`), the campaign runner clears all but
//! `BDLAMBDA`, which it pins to the suite's 100-bit Brakedown column-open
//! target; the values recorded in [`Context::config`] are what actually ran.
//! Limber's IntEval (128) and challenge (117) targets and its 2^-114
//! fingerprint term are crate constants and stay above 100.
use super::trace_capture::TrialScopes;
use super::{Corpus, Timing, Workload, captured};
use bitz::observability::Recording;
use limber::{
    imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
    imod_spartan_modp::{
        IntModSpartanModpProverKey, IntModSpartanModpSNARK, IntModSpartanModpVerifierKey,
    },
    provider::{
        T256DynPrimeBdEngine,
        pcs::integer_modpcs::{IntEvalParams, LAMBDA, LAMBDA_BOUND2},
    },
    traits::mod_engine::{ModEngine, SumcheckEngine, SumcheckField},
};
use num_bigint::BigUint;
use serde_json::{Value, json};
use std::sync::Arc;

type E = T256DynPrimeBdEngine;
/// IntEval per-iteration variable count, Limber's own default.
const K: usize = 9;
/// Brakedown layout defaults of the pinned Limber revision, reported so a
/// campaign records the configuration that produced its numbers. The runner
/// clears every override except `BDLAMBDA`, which it pins to the suite's
/// 100-bit column-open target, so these plus [`bd_target_bits`] are the
/// effective values.
const BD_LAMBDA_DEFAULT: usize = 114;
const BD_SPEC: usize = 4;

/// The Brakedown column-open security target the crate actually derives its
/// layout from: its `BDLAMBDA` env override, else its native 114-bit default
/// (`commit_backend::bd_lambda()` of the pinned revision). A present but
/// unparsable value aborts here instead of silently measuring the default
/// while the campaign records the override.
fn bd_target_bits() -> usize {
    match std::env::var("BDLAMBDA") {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("BDLAMBDA must be a usize, got {value:?}")),
        Err(std::env::VarError::NotPresent) => BD_LAMBDA_DEFAULT,
        Err(error) => panic!("invalid BDLAMBDA: {error}"),
    }
}
const BD_ROW_LEN_CAP: usize = 1 << 15;
const BD_DIRECT_OPEN_MAX: usize = 1 << 16;
/// Bit size of the T256 scalar field the Mod-PCS commits over.
const T256_SCALAR_BITS: usize = 256;

/// The wrapping-multiplication program.
///
/// Variables are laid out `[x_0..x_{g-1}, y_0..y_{g-1}, z_0..z_{g-1}]`, so a
/// gate costs three committed values plus its quotient.
struct Wrapping {
    /// Operand width `w`; the row modulus is `2^w`.
    bits: u32,
    gates: usize,
    num_vars: usize,
    num_cons: usize,
}
impl Wrapping {
    fn compile(corpus: &Corpus) -> Self {
        let bits = match corpus.workload {
            Workload::U32 => 32,
            Workload::U64 => 64,
            Workload::U128 => 128,
        };
        let gates = corpus.len();
        Self {
            bits,
            gates,
            num_vars: (3 * gates).next_power_of_two(),
            num_cons: gates.next_power_of_two(),
        }
    }
    fn modulus(&self) -> BigUint {
        BigUint::from(1u32) << self.bits
    }
    fn shape(&self) -> IntModR1CSShapeModp<E> {
        let one = BigUint::from(1u32);
        let column = |base: usize| -> Vec<(usize, usize, BigUint)> {
            (0..self.gates)
                .map(|i| (i, base + i, one.clone()))
                .collect()
        };
        IntModR1CSShapeModp::new(
            self.num_cons,
            self.num_vars,
            0,
            column(0),
            column(self.gates),
            column(2 * self.gates),
            // Padding rows keep the modulus: their empty combinations make
            // `0 * 0 = 0 (mod 2^w)` with quotient zero.
            vec![self.modulus(); self.num_cons],
        )
        .expect("Limber wrapping-multiplication shape")
    }
    fn mask(&self) -> u128 {
        if self.bits == 128 {
            u128::MAX
        } else {
            (1u128 << self.bits) - 1
        }
    }
    fn operands(&self, corpus: &Corpus, gate: usize) -> (u128, u128) {
        if corpus.workload == Workload::U128 {
            corpus.wide_inputs()[gate]
        } else {
            let (x, y) = corpus.inputs()[gate];
            (u128::from(x), u128::from(y))
        }
    }
    fn values(&self, corpus: &Corpus) -> Vec<u128> {
        let mut values = vec![0u128; self.num_vars];
        for gate in 0..self.gates {
            let (x, y) = self.operands(corpus, gate);
            values[gate] = x;
            values[self.gates + gate] = y;
            values[2 * self.gates + gate] = self.split(x, y).0;
        }
        values
    }
    /// The exact product of two in-range operands as `(low w bits, high w)`.
    fn split(&self, x: u128, y: u128) -> (u128, u128) {
        if self.bits == 128 {
            bitz::piop::spartan::mul_u128_full(x, y)
        } else {
            // Both operands are below `2^64` here, so the product fits `u128`.
            let product = x * y;
            (product & self.mask(), product >> self.bits)
        }
    }
    /// The native rows `[x, y, z_lo, z_hi]` recovered from the committed
    /// values alone, re-deriving each product so that a wrong output or an
    /// out-of-range operand is rejected here rather than inside the prover.
    fn native_rows(&self, values: &[u128]) -> Result<Vec<[u128; 4]>, String> {
        let mut rows = Vec::with_capacity(self.gates);
        for gate in 0..self.gates {
            let (x, y) = (values[gate], values[self.gates + gate]);
            if self.bits < 128 && (x >> self.bits != 0 || y >> self.bits != 0) {
                return Err(format!(
                    "operand of gate {gate} is not a {}-bit value",
                    self.bits
                ));
            }
            let (lo, hi) = self.split(x, y);
            if values[2 * self.gates + gate] != lo {
                return Err(format!("gate {gate} has the wrong low half"));
            }
            rows.push([x, y, lo, hi]);
        }
        Ok(rows)
    }
    fn quotients(&self, values: &[u128]) -> Result<Vec<BigUint>, String> {
        let mut q = vec![BigUint::from(0u32); self.num_cons];
        for (gate, row) in self.native_rows(values)?.iter().enumerate() {
            q[gate] = BigUint::from(row[3]);
        }
        Ok(q)
    }
    /// The IntEval bound each committed value is proven to respect.
    ///
    /// Widths up to 64 bits take a single limb, as in `int_mult`; 128-bit
    /// values are split into 32-bit limbs, since a single-limb 128-bit bound
    /// would need roughly ninety CRT primes.
    fn params(&self, arity: usize) -> IntEvalParams {
        if self.bits <= 64 {
            IntEvalParams::derive_no_limb_split(self.bits as usize, K, arity)
        } else {
            IntEvalParams::derive(self.bits as usize, 32, K, arity)
        }
        .expect("Limber IntEval parameters")
    }
    fn witness(&self, corpus: &Corpus) -> Result<(Vec<BigUint>, Vec<BigUint>), String> {
        let values = self.values(corpus);
        let q = self.quotients(&values)?;
        Ok((values.into_iter().map(BigUint::from).collect(), q))
    }
}

/// Bytes of the unsegmented PIOP transcript, a function of the public padded
/// shape alone.
///
/// The pinned Limber driver has no whole-proof serializer, so this payload is
/// counted analytically: three coefficients per outer round, two per inner
/// round, five outer evaluations and `eval_w`. Round counts are fixed by the
/// padded shape; no length prefixes or sampled modulus need be sent. The
/// convention matches the one the campaign runner checks, which came from the
/// authors' own `int_mult` records ($2^{15}$: 85 scalars, 1360 bytes).
fn piop_bytes(num_cons: usize, num_vars: usize, scalar_bytes: usize) -> usize {
    scalar_bytes * (3 * num_cons.ilog2() as usize + 2 * num_vars.ilog2() as usize + 6)
}

pub(super) struct Context {
    corpus: Arc<Corpus>,
    program: Wrapping,
    shape: IntModR1CSShapeModp<E>,
    params: IntEvalParams,
    pk: IntModSpartanModpProverKey<E>,
    vk: IntModSpartanModpVerifierKey<E>,
    scalar_bytes: usize,
}
impl Context {
    pub(super) fn setup(corpus: Arc<Corpus>) -> Self {
        let program = Wrapping::compile(&corpus);
        let shape = program.shape();
        let arity = program.num_cons.max(program.num_vars).ilog2() as usize;
        let params = program.params(arity);
        let (pk, vk) =
            IntModSpartanModpSNARK::<E>::setup_with_params(shape.clone(), params.clone())
                .expect("Limber setup");
        let scalar_bytes = <E as SumcheckEngine>::Scalar::zero(&E::bootstrap_params())
            .to_le_bytes()
            .len();
        Self {
            corpus,
            program,
            shape,
            params,
            pk,
            vk,
            scalar_bytes,
        }
    }
    pub(super) fn config(&self) -> Value {
        let w = &self.program;
        json!({"piop":"Integer-Mod Spartan","pcs":"Limber IntEval/Brakedown",
               "commitment_backend":"brakedown","engine":"T256DynPrimeBdEngine",
               "encoding":format!(
                   "one modular row x*y = z_lo (mod 2^{}) per gate, quotient = high half",
                   w.bits),
               "log_t_f":self.params.log_t_f,"log_t":self.params.log_t,"k":self.params.k,
               "log_p":self.params.log_p,"s":self.params.s,"numlimb":self.params.numlimb,
               "operand_bits":w.bits,"modulus":w.modulus().to_string(),
               "constraints":w.gates,"padded_constraints":w.num_cons,
               "variables":3*w.gates,"padded_variables":w.num_vars,"quotients":w.num_cons,
               "target_bits":bd_target_bits(),"log_q":T256_SCALAR_BITS,
               "challenge_target_bits":LAMBDA_BOUND2,"inteval_target_bits":LAMBDA,
               "brakedown":{"target_bits":bd_target_bits(),"spec":BD_SPEC,
                            "row_len_cap":BD_ROW_LEN_CAP,"direct_open_max":BD_DIRECT_OPEN_MAX,
                            "split":false},
               "piop_payload_bytes":piop_bytes(w.num_cons, w.num_vars, self.scalar_bytes),
               "security_policy":"BDLAMBDA-pinned Brakedown column-open target; native IntEval 128, challenge 117 and ~114-bit fingerprint floors"})
    }
    pub(super) fn run(&self) -> Timing {
        let recording = Recording::start(Vec::new()).expect("start Perfetto trial");
        let proof_bytes = self.prove_and_verify();
        let raw = recording.intervals().expect("query Perfetto trial");
        let trial = TrialScopes::from_spans(&raw, "benchmark");
        let wend = trial.witness.end_ns;
        let ready = trial.witness_to_proof.end_ns;
        let commit = captured(&raw, "imod_modp_wq_commit", wend, ready);
        let piop = captured(&raw, "imod_modp_piop", commit.end_ns, ready);
        let opening = captured(&raw, "imod_modp_wq_open", commit.end_ns, ready);
        let mut t = Timing::from_trial(&trial, proof_bytes);
        t.add("commit", "commit", commit.start_ns, commit.end_ns);
        t.add(
            "prime_projection",
            "preparation",
            commit.end_ns,
            piop.start_ns,
        );
        t.add("piop", "constraint-proof", piop.start_ns, piop.end_ns);
        t.add("opening", "opening-proof", opening.start_ns, opening.end_ns);
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
        let (w, q) = self
            .program
            .witness(&self.corpus)
            .expect("Limber witness satisfies exact and modular constraints");
        drop(witness_scope);
        let (witness, instance) =
            IntModR1CSWitnessModp::<E>::new(&self.shape, self.pk.ck(), w, q, vec![])
                .expect("Limber witness commitment");
        let proof = IntModSpartanModpSNARK::<E>::prove(&self.pk, &instance, &witness)
            .expect("Limber full proof");
        drop(proving);
        let verification = tracing::info_span!(
            "Verification",
            component = "benchmark.verification",
            scope_kind = "phase",
            tag_verification = true
        )
        .entered();
        proof
            .verify(&self.vk, &instance)
            .expect("Limber full verification");
        drop(verification);
        drop(trial);
        // Serialized commitments and batch opening, plus the analytically
        // counted PIOP payload the pinned driver does not serialize.
        let proof_bytes = instance
            .commitment_bytes()
            .expect("serialize Limber commitments")
            .len()
            + piop_bytes(
                self.program.num_cons,
                self.program.num_vars,
                self.scalar_bytes,
            )
            + proof
                .eval_arg_bytes()
                .expect("serialize Limber opening")
                .len();
        std::hint::black_box((proof, witness, instance));
        proof_bytes
    }
}

pub(super) fn audit(corpus: &Corpus) -> super::WitnessAudit {
    let program = Wrapping::compile(corpus);
    let started = std::time::Instant::now();
    let values = program.values(corpus);
    let rows = program.native_rows(&values).expect("integer constraints");
    let generation_ms = started.elapsed().as_secs_f64() * 1000.;
    let representation = "Limber wrapping rows and quotients";
    if corpus.workload == Workload::U128 {
        return super::WitnessAudit::check_wide(corpus, rows, generation_ms, representation);
    }
    // Both narrow workloads report the two halves the program commits: the
    // canonical row is `[x, y, z_lo, z_hi]`, with `z_lo` the modular output.
    let narrow = rows
        .iter()
        .map(|&[x, y, lo, hi]| {
            let cell = |v: u128| u64::try_from(v).expect("native value fits u64");
            [cell(x), cell(y), cell(lo), cell(hi)]
        })
        .collect();
    super::WitnessAudit::check(corpus, narrow, generation_ms, representation, false)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)] // `cargo bench` sets cfg(test) without running these.
    use super::*;
    #[test]
    fn wrapping_rows_reject_bad_outputs_and_ranges() {
        for workload in [Workload::U32, Workload::U64, Workload::U128] {
            let corpus = Corpus::new(workload, 4, 7);
            let shape = Wrapping::compile(&corpus);
            let values = shape.values(&corpus);
            let rows = shape.native_rows(&values).expect("honest witness");
            assert_eq!(rows.len(), corpus.len());
            // A wrong low half is caught by re-deriving the product.
            let mut wrong = values.clone();
            wrong[2 * shape.gates] ^= 1;
            assert!(shape.native_rows(&wrong).is_err());
            // So is an operand wider than the row modulus.
            if shape.bits < 128 {
                let mut wide = values.clone();
                wide[0] = 1u128 << shape.bits;
                assert!(shape.native_rows(&wide).is_err());
            }
        }
    }
    #[test]
    fn piop_accounting_matches_the_authors_recorded_payload() {
        // From the authors' own int_mult record at 2^15: padded shape
        // (2^15 constraints, 2^17 variables) and 1360 payload bytes.
        assert_eq!(piop_bytes(1 << 15, 1 << 17, 16), 1360);
    }
    #[test]
    fn shape_is_three_live_variables_and_one_quotient_per_gate() {
        let corpus = Corpus::new(Workload::U32, 5, 7);
        let w = Wrapping::compile(&corpus);
        let gates = 1 << 5;
        assert_eq!((w.gates, w.num_cons, w.num_vars), (gates, gates, 4 * gates));
        assert_eq!(w.quotients(&w.values(&corpus)).unwrap().len(), w.num_cons);
    }
}
