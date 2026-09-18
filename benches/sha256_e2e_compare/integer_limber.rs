//! Native Integer-Mod-R1CS SHA-256 for Limber's Brakedown engine.

use super::common;
use super::trace_capture::TrialScopes;
use std::{hint::black_box, sync::Arc};

use limber::{
    imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
    imod_spartan_modp::IntModSpartanModpSNARK,
    provider::{T256DynPrimeBdEngine, pcs::integer_modpcs::IntEvalParams},
    traits::mod_engine::ModEngine,
};
use num_bigint::BigUint;
use serde_json::{Value, json};

use super::{CapturedSpan, Corpus, SHA256_IV, SemanticSpan, TrialMetrics, humanize, operation};

const MOD_TWO: u64 = 2;
const MOD_U32: u64 = 1u64 << 32;
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Params {
    pub k: usize,
}

impl Params {
    pub fn label(self) -> String {
        format!("brakedown-k{}", self.k)
    }
}

pub fn security_metadata() -> Value {
    let bits = super::env_usize("BDLAMBDA", 114);
    assert!(bits >= 114, "SHA Brakedown requires BDLAMBDA >= 114");
    json!({
        "profile": "Limber Integer-Mod-R1CS/Brakedown",
        "engine": "T256DynPrimeBdEngine",
        "commitment_backend": "brakedown",
        "commitment_type": "hash-based",
        "inteval_target_bits": limber::provider::pcs::integer_modpcs::LAMBDA,
        "challenge_target_bits": limber::provider::pcs::integer_modpcs::LAMBDA_BOUND2,
        "brakedown": {
            "column_opening_target_bits": bits,
            "spec": super::env_usize("BDSPEC", 4),
            "row_len_cap": super::env_usize("BDROWLEN", 1 << 15),
            "direct_open_max": super::env_usize("BDDIRECT", 1 << 16),
            "split": std::env::var_os("BDSPLIT").is_some(),
        },
        "claim": "Native classical component targets; not a complete quantum security bound",
    })
}

#[derive(Clone, Copy)]
enum Column {
    Witness(usize),
    One,
    Public(usize),
}

#[derive(Clone)]
enum Assignment {
    Fixed(u64),
    PublicBit(usize),
    Xor(Vec<Bit>),
    And(Bit, Bit),
    Pack([Bit; 32]),
    AddMod { words: Vec<usize>, constant: u64 },
    BitOf { word: usize, bit: usize },
}

#[derive(Clone, Copy)]
struct Bit {
    column: Option<usize>,
    constant: bool,
}

impl Bit {
    const ZERO: Self = Self {
        column: None,
        constant: false,
    };
}

#[derive(Clone)]
struct Word {
    bits: [Bit; 32],
    packed: usize,
}

#[derive(Clone)]
struct Row {
    a: Vec<(Column, u64)>,
    b: Vec<(Column, u64)>,
    c: Vec<(Column, u64)>,
    modulus: u64,
}

#[derive(Clone)]
struct Program {
    assignments: Vec<Assignment>,
    rows: Vec<Row>,
    public: Vec<u64>,
    num_vars: usize,
    num_cons: usize,
}

impl Program {
    fn compile(corpus: &Corpus) -> Self {
        let mut compiler = Compiler::default();
        let iv: [Word; 8] = std::array::from_fn(|word| compiler.fixed_word(SHA256_IV[word]));
        for case in &corpus.cases {
            let block: [Word; 16] =
                std::array::from_fn(|word| compiler.public_input_word(case.block[word]));
            let output = compiler.compress(iv.clone(), block);
            for (word, expected) in output.iter().zip(case.output) {
                compiler.bind_public_output(word, expected);
            }
        }
        compiler.finish()
    }

    fn evaluate(&self) -> (Vec<BigUint>, Vec<BigUint>, Vec<BigUint>) {
        let mut values = Vec::<u64>::with_capacity(self.num_vars);
        for assignment in &self.assignments {
            let value = match assignment {
                Assignment::Fixed(value) => *value,
                Assignment::PublicBit(index) => self.public[*index],
                Assignment::Xor(bits) => {
                    bits.iter().fold(false, |acc, bit| acc ^ bit.value(&values)) as u64
                }
                Assignment::And(a, b) => (a.value(&values) & b.value(&values)) as u64,
                Assignment::Pack(bits) => {
                    bits.iter().enumerate().fold(0u64, |acc, (bit, value)| {
                        acc | ((value.value(&values) as u64) << bit)
                    })
                }
                Assignment::AddMod { words, constant } => {
                    words
                        .iter()
                        .fold(*constant, |acc, &word| acc + values[word])
                        & (MOD_U32 - 1)
                }
                Assignment::BitOf { word, bit } => (values[*word] >> bit) & 1,
            };
            values.push(value);
        }
        values.resize(self.num_vars, 0);

        let mut quotients = Vec::with_capacity(self.num_cons);
        for row in &self.rows {
            let a = eval_lc(&row.a, &values, &self.public);
            let b = eval_lc(&row.b, &values, &self.public);
            let c = eval_lc(&row.c, &values, &self.public);
            let residual = a
                .checked_mul(b)
                .expect("constraint product fits u128")
                .checked_sub(c)
                .expect("non-negative integer quotient");
            if row.modulus == 0 {
                assert_eq!(residual, 0, "exact Integer-Mod-R1CS row failed");
                quotients.push(BigUint::from(0u32));
            } else {
                assert_eq!(residual % row.modulus as u128, 0, "modular row failed");
                quotients.push(BigUint::from(residual / row.modulus as u128));
            }
        }
        quotients.resize(self.num_cons, BigUint::from(0u32));
        (
            values.into_iter().map(BigUint::from).collect(),
            quotients,
            self.public.iter().copied().map(BigUint::from).collect(),
        )
    }

    fn shape<M: ModEngine>(&self) -> IntModR1CSShapeModp<M> {
        let resolve = |column: Column| match column {
            Column::Witness(index) => index,
            Column::One => self.num_vars,
            Column::Public(index) => self.num_vars + 1 + index,
        };
        let entries = |select: fn(&Row) -> &Vec<(Column, u64)>| {
            self.rows
                .iter()
                .enumerate()
                .flat_map(|(row_index, row)| {
                    select(row).iter().map(move |&(column, coefficient)| {
                        (row_index, resolve(column), BigUint::from(coefficient))
                    })
                })
                .collect()
        };
        let mut mods = self
            .rows
            .iter()
            .map(|row| BigUint::from(row.modulus))
            .collect::<Vec<_>>();
        mods.resize(self.num_cons, BigUint::from(0u32));
        IntModR1CSShapeModp::new(
            self.num_cons,
            self.num_vars,
            self.public.len(),
            entries(|row| &row.a),
            entries(|row| &row.b),
            entries(|row| &row.c),
            mods,
        )
        .expect("valid power-of-two integer SHA shape")
    }
}

impl Bit {
    fn value(self, values: &[u64]) -> bool {
        self.column
            .map_or(self.constant, |column| values[column] != 0)
    }

    fn term(self) -> Option<(Column, u64)> {
        self.column
            .map(|column| (Column::Witness(column), 1))
            .or_else(|| self.constant.then_some((Column::One, 1)))
    }
}

#[derive(Default)]
struct Compiler {
    assignments: Vec<Assignment>,
    values: Vec<u64>,
    rows: Vec<Row>,
    public: Vec<u64>,
}

impl Compiler {
    fn finish(self) -> Program {
        let num_io = self.public.len();
        let num_vars = self.assignments.len().max(num_io + 1).next_power_of_two();
        let num_cons = self.rows.len().max(1).next_power_of_two();
        Program {
            assignments: self.assignments,
            rows: self.rows,
            public: self.public,
            num_vars,
            num_cons,
        }
    }

    fn alloc(&mut self, assignment: Assignment) -> usize {
        let value = match &assignment {
            Assignment::Fixed(value) => *value,
            Assignment::PublicBit(index) => self.public[*index],
            Assignment::Xor(bits) => {
                bits.iter()
                    .fold(false, |acc, bit| acc ^ bit.value(&self.values)) as u64
            }
            Assignment::And(a, b) => (a.value(&self.values) & b.value(&self.values)) as u64,
            Assignment::Pack(bits) => bits.iter().enumerate().fold(0u64, |acc, (bit, value)| {
                acc | ((value.value(&self.values) as u64) << bit)
            }),
            Assignment::AddMod { words, constant } => {
                words
                    .iter()
                    .fold(*constant, |acc, &word| acc + self.values[word])
                    & (MOD_U32 - 1)
            }
            Assignment::BitOf { word, bit } => (self.values[*word] >> bit) & 1,
        };
        let index = self.assignments.len();
        self.assignments.push(assignment);
        self.values.push(value);
        index
    }

    fn row(
        &mut self,
        a: Vec<(Column, u64)>,
        b: Vec<(Column, u64)>,
        c: Vec<(Column, u64)>,
        modulus: u64,
    ) {
        self.rows.push(Row { a, b, c, modulus });
    }

    fn boolean(&mut self, assignment: Assignment) -> Bit {
        let column = self.alloc(assignment);
        let term = vec![(Column::Witness(column), 1)];
        self.row(term.clone(), term.clone(), term, 0);
        Bit {
            column: Some(column),
            constant: false,
        }
    }

    fn fixed_word(&mut self, value: u32) -> Word {
        let bits = std::array::from_fn(|bit| {
            let expected = (value >> bit) & 1 == 1;
            let result = self.boolean(Assignment::Fixed(expected as u64));
            self.row(
                vec![(Column::Witness(result.column.unwrap()), 1)],
                vec![(Column::One, 1)],
                expected.then_some((Column::One, 1)).into_iter().collect(),
                0,
            );
            result
        });
        self.word_from_bits(bits)
    }

    fn public_input_word(&mut self, value: u32) -> Word {
        let bits = std::array::from_fn(|bit| {
            let expected = ((value >> bit) & 1) as u64;
            let public = self.public.len();
            self.public.push(expected);
            let result = self.boolean(Assignment::PublicBit(public));
            self.row(
                vec![(Column::Witness(result.column.unwrap()), 1)],
                vec![(Column::One, 1)],
                vec![(Column::Public(public), 1)],
                0,
            );
            result
        });
        self.word_from_bits(bits)
    }

    fn bind_public_output(&mut self, word: &Word, expected: u32) {
        assert_eq!(
            self.values[word.packed] as u32, expected,
            "integer SHA output mismatch"
        );
        for bit in 0..32 {
            let public = self.public.len();
            self.public.push(((expected >> bit) & 1) as u64);
            self.row(
                vec![(Column::Witness(word.bits[bit].column.unwrap()), 1)],
                vec![(Column::One, 1)],
                vec![(Column::Public(public), 1)],
                0,
            );
        }
    }

    fn word_from_bits(&mut self, bits: [Bit; 32]) -> Word {
        let packed = self.alloc(Assignment::Pack(bits));
        let a = bits
            .iter()
            .enumerate()
            .filter_map(|(bit, value)| {
                value
                    .term()
                    .map(|(column, coefficient)| (column, coefficient << bit))
            })
            .collect();
        self.row(
            a,
            vec![(Column::One, 1)],
            vec![(Column::Witness(packed), 1)],
            0,
        );
        Word { bits, packed }
    }

    fn word_from_mod_sum(&mut self, words: &[&Word], constant: u64) -> Word {
        let inputs = words.iter().map(|word| word.packed).collect::<Vec<_>>();
        let packed = self.alloc(Assignment::AddMod {
            words: inputs.clone(),
            constant,
        });
        let bits = std::array::from_fn(|bit| self.boolean(Assignment::BitOf { word: packed, bit }));
        let mut a = inputs
            .into_iter()
            .map(|word| (Column::Witness(word), 1))
            .collect::<Vec<_>>();
        if constant != 0 {
            a.push((Column::One, constant));
        }
        self.row(
            a,
            vec![(Column::One, 1)],
            vec![(Column::Witness(packed), 1)],
            MOD_U32,
        );
        let packed_a = bits
            .iter()
            .enumerate()
            .filter_map(|(bit, value)| {
                value
                    .term()
                    .map(|(column, coefficient)| (column, coefficient << bit))
            })
            .collect();
        self.row(
            packed_a,
            vec![(Column::One, 1)],
            vec![(Column::Witness(packed), 1)],
            0,
        );
        Word { bits, packed }
    }

    fn xor_bits(&mut self, inputs: &[Bit]) -> Bit {
        let out = self.boolean(Assignment::Xor(inputs.to_vec()));
        let mut a = inputs
            .iter()
            .filter_map(|bit| bit.term())
            .collect::<Vec<_>>();
        a.push((Column::Witness(out.column.unwrap()), 1));
        self.row(a, vec![(Column::One, 1)], Vec::new(), MOD_TWO);
        out
    }

    fn xor_words(&mut self, words: &[[Bit; 32]]) -> Word {
        let bits = std::array::from_fn(|bit| {
            let inputs = words.iter().map(|word| word[bit]).collect::<Vec<_>>();
            self.xor_bits(&inputs)
        });
        self.word_from_bits(bits)
    }

    fn and_words(&mut self, left: &Word, right: &Word) -> Word {
        let bits = std::array::from_fn(|bit| {
            let out = self.boolean(Assignment::And(left.bits[bit], right.bits[bit]));
            self.row(
                left.bits[bit].term().into_iter().collect(),
                right.bits[bit].term().into_iter().collect(),
                vec![(Column::Witness(out.column.unwrap()), 1)],
                0,
            );
            out
        });
        self.word_from_bits(bits)
    }

    fn rotr(word: &Word, amount: usize) -> [Bit; 32] {
        std::array::from_fn(|bit| word.bits[(bit + amount) % 32])
    }

    fn shr(word: &Word, amount: usize) -> [Bit; 32] {
        std::array::from_fn(|bit| {
            if bit + amount < 32 {
                word.bits[bit + amount]
            } else {
                Bit::ZERO
            }
        })
    }

    fn sigma(
        &mut self,
        word: &Word,
        first: usize,
        second: usize,
        third: usize,
        shift: bool,
    ) -> Word {
        let a = Self::rotr(word, first);
        let b = Self::rotr(word, second);
        let c = if shift {
            Self::shr(word, third)
        } else {
            Self::rotr(word, third)
        };
        self.xor_words(&[a, b, c])
    }

    fn choice(&mut self, e: &Word, f: &Word, g: &Word) -> Word {
        let fg = self.xor_words(&[f.bits, g.bits]);
        let selected = self.and_words(e, &fg);
        self.xor_words(&[g.bits, selected.bits])
    }

    fn majority(&mut self, a: &Word, b: &Word, c: &Word) -> Word {
        let ab_xor = self.xor_words(&[a.bits, b.bits]);
        let ab = self.and_words(a, b);
        let c_selected = self.and_words(c, &ab_xor);
        self.xor_words(&[ab.bits, c_selected.bits])
    }

    fn compress(&mut self, initial: [Word; 8], block: [Word; 16]) -> [Word; 8] {
        let mut schedule = block.to_vec();
        for round in 16..64 {
            let s0 = self.sigma(&schedule[round - 15], 7, 18, 3, true);
            let s1 = self.sigma(&schedule[round - 2], 17, 19, 10, true);
            let next =
                self.word_from_mod_sum(&[&s1, &schedule[round - 7], &s0, &schedule[round - 16]], 0);
            schedule.push(next);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = initial.clone();
        for round in 0..64 {
            let s1 = self.sigma(&e, 6, 11, 25, false);
            let ch = self.choice(&e, &f, &g);
            let t1 = self.word_from_mod_sum(&[&h, &s1, &ch, &schedule[round]], K[round] as u64);
            let s0 = self.sigma(&a, 2, 13, 22, false);
            let maj = self.majority(&a, &b, &c);
            let t2 = self.word_from_mod_sum(&[&s0, &maj], 0);
            let next_a = self.word_from_mod_sum(&[&t1, &t2], 0);
            let next_e = self.word_from_mod_sum(&[&d, &t1], 0);
            h = g;
            g = f;
            f = e;
            e = next_e;
            d = c;
            c = b;
            b = a;
            a = next_a;
        }
        let state = [a, b, c, d, e, f, g, h];
        std::array::from_fn(|word| self.word_from_mod_sum(&[&initial[word], &state[word]], 0))
    }
}

fn eval_lc(terms: &[(Column, u64)], witness: &[u64], public: &[u64]) -> u128 {
    terms
        .iter()
        .map(|&(column, coefficient)| {
            let value = match column {
                Column::Witness(index) => witness[index],
                Column::One => 1,
                Column::Public(index) => public[index],
            };
            value as u128 * coefficient as u128
        })
        .sum()
}

type E = T256DynPrimeBdEngine;

pub struct Context {
    program: Arc<Program>,
    pk: limber::imod_spartan_modp::IntModSpartanModpProverKey<E>,
    vk: limber::imod_spartan_modp::IntModSpartanModpVerifierKey<E>,
    pub setup_ms: f64,
}

impl Context {
    pub fn setup_ms(&self) -> f64 {
        self.setup_ms
    }

    pub fn setup(corpus: &Corpus, params: Params) -> Result<Self, String> {
        security_metadata();
        let program = Arc::new(Program::compile(corpus));
        let started_recording = bitz::observability::Recording::start(Vec::new()).expect("start operation capture");
        let started = tracing::info_span!("sha256_e2e_compare/integer_limber:started").entered();
        let shape = program.shape::<E>();
        let arity = shape.num_vars().max(shape.num_cons()).ilog2() as usize;
        let params =
            IntEvalParams::derive(32, 16, params.k, arity).map_err(|error| error.to_string())?;
        let (pk, vk) = IntModSpartanModpSNARK::<E>::setup_with_params(shape, params)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            program,
            pk,
            vk,
            setup_ms: { drop(started); bitz::observability::duration(&started_recording.intervals().expect("complete operation capture"), "sha256_e2e_compare/integer_limber:started").expect("query completed operation") }.as_secs_f64() * 1e3,
        })
    }

    pub fn run(&self) -> (TrialMetrics, Vec<SemanticSpan>) {
        let recording =
            common::perfetto::Recording::start(Vec::new()).expect("start Perfetto trial");
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
        let (w, q, x) = self.program.evaluate();
        drop(witness_scope);
        let shape = self.program.shape::<E>();
        let (witness, instance) = IntModR1CSWitnessModp::<E>::new(&shape, self.pk.ck(), w, q, x)
            .expect("integer SHA witness commitments succeed");
        let proof = IntModSpartanModpSNARK::<E>::prove(&self.pk, &instance, &witness)
            .expect("integer SHA proof succeeds");
        drop(proving);
        let commitment_bytes = instance
            .commitment_bytes()
            .expect("serialize Limber commitments")
            .len();
        let opening_bytes = proof
            .eval_arg_bytes()
            .expect("serialize Limber opening argument")
            .len();
        let dynamic_bytes = 16
            * (3 * self.program.num_cons.ilog2() as usize
                + 2 * (self.program.num_vars.ilog2() as usize + 1)
                + 6);
        let proof_bytes = commitment_bytes + opening_bytes + dynamic_bytes;
        let verification = tracing::info_span!(
            "Verification",
            component = "benchmark.verification",
            scope_kind = "phase",
            tag_verification = true
        )
        .entered();
        proof
            .verify(&self.vk, &instance)
            .expect("integer SHA proof verifies");
        drop(verification);
        drop(trial);
        let raw = recording.intervals().expect("query Perfetto trial");
        black_box((&proof, &instance, &witness));
        let spans = semantic_spans(&raw);
        (TrialMetrics::from_spans(&spans, proof_bytes), spans)
    }
}

pub fn constraint_self_test() {
    let corpus = Corpus::new(2, 0x1234_5678);
    let program = Program::compile(&corpus);
    let (w, q, x) = program.evaluate();
    let shape = program.shape::<E>();
    assert_eq!(w.len(), shape.num_vars());
    assert_eq!(q.len(), shape.num_cons());
    assert_eq!(x.len(), 2 * 24 * 32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brakedown_sha_proof_binds_public_inputs_and_outputs() {
        let _trace = common::test_tracing();
        let corpus = Corpus::new(1, 0x4252_414b_4544_4f57);
        let context = Context::setup(&corpus, Params { k: 9 }).unwrap();
        let shape = context.program.shape::<E>();
        let (w, q, x) = context.program.evaluate();
        let (witness, instance) = IntModR1CSWitnessModp::<E>::new(
            &shape,
            context.pk.ck(),
            w.clone(),
            q.clone(),
            x.clone(),
        )
        .unwrap();
        shape.is_sat(context.pk.ck(), &instance, &witness).unwrap();
        let proof = IntModSpartanModpSNARK::<E>::prove(&context.pk, &instance, &witness).unwrap();
        proof.verify(&context.vk, &instance).unwrap();
        assert!(!instance.commitment_bytes().unwrap().is_empty());
        assert!(!proof.eval_arg_bytes().unwrap().is_empty());

        for index in [0, 16 * 32] {
            let mut changed = x.clone();
            changed[index] ^= BigUint::from(1u32);
            let (bad_witness, bad_instance) = IntModR1CSWitnessModp::<E>::new(
                &shape,
                context.pk.ck(),
                w.clone(),
                q.clone(),
                changed,
            )
            .unwrap();
            assert!(
                shape
                    .is_sat(context.pk.ck(), &bad_instance, &bad_witness)
                    .is_err()
            );
            assert!(proof.verify(&context.vk, &bad_instance).is_err());
        }
    }
}

fn semantic_spans(raw: &[CapturedSpan]) -> Vec<SemanticSpan> {
    let trial = TrialScopes::from_spans(raw, "benchmark");
    let root_start = trial.verified.start_ns;
    let root_end = trial.verified.end_ns;
    let witness_to_proof_start = trial.witness_to_proof.start_ns;
    let witness_start = trial.witness.start_ns;
    let witness_end = trial.witness.end_ns;
    let verify_start = trial.verification.start_ns;
    let verify_end = trial.verification.end_ns;
    let proof_ready = trial.witness_to_proof.end_ns;
    let find = |name: &str| {
        raw.iter()
            .find(|span| span.name == name)
            .unwrap_or_else(|| panic!("missing integer Limber span {name}"))
    };
    let commit = find("imod_modp_wq_commit");
    let opening = find("imod_modp_wq_open");
    let mut spans = vec![
        make_span(
            "limber-root",
            None,
            "limber.verified-trial",
            "Complete verified integer Limber trial",
            "end-to-end",
            vec!["end-to-end"],
            root_start,
            root_end,
            Some("end-to-end"),
            false,
        ),
        make_span(
            "limber-witness-to-proof",
            Some("limber-root"),
            "limber.witness-to-proof",
            "Integer witness generation through proof readiness",
            "proving",
            vec!["proving"],
            witness_to_proof_start,
            proof_ready,
            None,
            false,
        ),
        make_span(
            "limber-witness",
            Some("limber-witness-to-proof"),
            "limber.witness",
            "Evaluate integer SHA assignment and quotients",
            "witness-generation",
            vec!["witness-generation"],
            witness_start,
            witness_end,
            None,
            true,
        ),
        make_span(
            "limber-online",
            Some("limber-root"),
            "limber.online",
            "Integer Limber online prover",
            "proving",
            vec!["proving"],
            commit.start_ns,
            proof_ready,
            Some("proving"),
            false,
        ),
        make_span(
            "limber-commit",
            Some("limber-online"),
            "limber.commit",
            "Commit integer witness and quotient",
            "commit",
            vec!["commit", "pcs", "proving"],
            commit.start_ns,
            commit.end_ns,
            Some("commit"),
            true,
        ),
        make_span(
            "limber-piop",
            Some("limber-online"),
            "limber.piop",
            "Integer Spartan outer and inner sumchecks",
            "constraint-proof",
            vec!["constraint-proof", "sumcheck", "proving"],
            commit.end_ns,
            opening.start_ns,
            Some("constraint-proof"),
            true,
        ),
        make_span(
            "limber-iop",
            Some("limber-online"),
            "limber.iop",
            "Integer Mod-PCS opening",
            "opening-proof",
            vec!["opening-proof", "pcs", "proving"],
            opening.start_ns,
            proof_ready,
            Some("opening-proof"),
            true,
        ),
        make_span(
            "limber-verify",
            Some("limber-root"),
            "limber.verify",
            "Verify integer Limber proof",
            "verification",
            vec!["verification"],
            verify_start,
            verify_end,
            Some("verification"),
            true,
        ),
    ];
    spans.extend(
        raw.iter()
            .enumerate()
            .filter(|(_, raw)| raw.start_ns >= witness_start && raw.end_ns <= proof_ready)
            .map(|(index, raw)| SemanticSpan {
                id: format!("limber-detail-{index}"),
                parent: Some("limber-witness-to-proof".to_owned()),
                operation: format!("limber.{}", operation(&raw.name)),
                name: humanize(&raw.name),
                short_name: "Procedure".to_owned(),
                primary_phase: "proving",
                phase_tags: vec!["proving"],
                start_ns: raw.start_ns,
                end_ns: raw.end_ns,
                scope_kind: "procedure",
                scope_tag: None,
                primary_sequence: false,
                math_latex: vec![],
            }),
    );
    spans
}

#[allow(clippy::too_many_arguments)]
fn make_span(
    id: &str,
    parent: Option<&str>,
    operation: &str,
    name: &str,
    primary_phase: &'static str,
    phase_tags: Vec<&'static str>,
    start_ns: u64,
    end_ns: u64,
    scope_tag: Option<&'static str>,
    primary_sequence: bool,
) -> SemanticSpan {
    SemanticSpan {
        id: id.to_owned(),
        parent: parent.map(str::to_owned),
        operation: operation.to_owned(),
        name: name.to_owned(),
        short_name: name.to_owned(),
        primary_phase,
        phase_tags,
        start_ns,
        end_ns,
        scope_kind: "phase",
        scope_tag,
        primary_sequence,
        math_latex: match primary_phase {
            "witness-generation" => vec![r"Az\circ Bz=Cz+m\circ q\quad\text{over }\mathbb Z"],
            "commit" => vec![r"(C_w,C_q)=\operatorname{IntCommit}(w,q)"],
            "constraint-proof" => vec![r"\sum_x\operatorname{eq}(\tau,x)(Az\,Bz-Cz-mq)=0"],
            "opening-proof" => vec![r"\operatorname{IntEval.Open}(C_w,C_q,r_x,r_y)"],
            _ => vec![],
        },
    }
}
