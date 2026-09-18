#![allow(dead_code)]
use super::proof::mod32;
use super::{Metrics as RawMetrics, Run, config};
use crate::common;
use bitz::piop::spartan::mul::{MulLayout, MulWitness};
mod binius;
mod binius_ligerito;
mod limber;
mod mod32_air;
mod plonky3;
mod plonky3_whir;
#[path = "../../common/trace_capture.rs"]
mod trace_capture;
use bitz::observability::Interval as CapturedSpan;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::sync::Arc;
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct Metrics {
    pub witness_ms: f64,
    pub commit_ms: f64,
    pub piop_ms: f64,
    pub opening_ms: f64,
    pub pcs_ms: f64,
    pub online_prover_ms: f64,
    pub witness_to_proof_ms: f64,
    pub post_proof_ms: f64,
    pub verify_ms: f64,
    pub verified_trial_ms: f64,
    pub proof_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Workload {
    #[value(name = "u32-mod32", alias = "u32")]
    U32,
    U64,
    U128,
}
impl Workload {
    fn slug(self) -> &'static str {
        match self {
            Self::U32 => "u32-mod32",
            Self::U64 => "u64",
            Self::U128 => "u128",
        }
    }
    fn algorithm(self) -> &'static str {
        match self {
            Self::U32 => "independent multiplication modulo 2^32",
            Self::U64 => "u64 multiplication",
            Self::U128 => "u128 multiplication",
        }
    }
    /// Backends with a native arithmetization of this workload.
    fn supports(self, backend: Backend) -> bool {
        // Plonky3's AIR only decomposes 32-bit operands. The other adapters
        // also support the u64 and u128 relations.
        self == Self::U32 || !matches!(backend, Backend::Plonky3Fri | Backend::Plonky3Whir)
    }
    /// Whether the operands are 128-bit values (the `u128` workload) rather
    /// than `u64` values.
    fn is_wide(self) -> bool {
        self == Self::U128
    }
    /// The exact integer output of one gate (`a*b`, or `a*b mod p`) of a
    /// 64-bit-or-narrower workload.
    fn output(self, a: u64, b: u64) -> u128 {
        let product = u128::from(a) * u128::from(b);
        match self {
            Self::U32 => product & u128::from(u32::MAX),
            Self::U64 => product,
            Self::U128 => panic!("the u128 workload has 128-bit operands"),
        }
    }
    /// The four native witness values of one `u128` gate: operands, then the
    /// low and high 128-bit halves of the exact 256-bit product.
    fn wide_row(self, x: u128, y: u128) -> [u128; 4] {
        assert!(self.is_wide(), "{} operands are u64 values", self.slug());
        let (lo, hi) = bitz::piop::spartan::mul_u128_full(x, y);
        [x, y, lo, hi]
    }
    /// The four native witness values of one gate: operands, then the
    /// output (`c` or `z_lo`) and the auxiliary value (`k`, `z_hi`, or 0).
    fn native_row(self, a: u64, b: u64) -> [u64; 4] {
        let output = self.output(a, b);
        match self {
            Self::U32 => [a, b, output as u64, (a * b) >> 32],
            Self::U64 => [a, b, output as u64, (output >> 64) as u64],
            Self::U128 => panic!("the u128 workload has 128-bit operands"),
        }
    }
}

/// The operand pairs of one corpus: `u64` values for the 64-bit-or-narrower
/// workloads, `u128` values for the `u128` workload.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Operands {
    Narrow(Vec<(u64, u64)>),
    Wide(Vec<(u128, u128)>),
}

struct Corpus {
    workload: Workload,
    operands: Operands,
    digest: String,
}
impl Corpus {
    fn new(workload: Workload, exponent: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        if workload.is_wide() {
            let inputs = (0..1usize << exponent)
                .map(|_| (rng.random::<u128>(), rng.random::<u128>()))
                .collect();
            return Self::from_wide_inputs(workload, inputs);
        }
        let inputs = if workload == Workload::U32 {
            mod32::inputs(exponent, seed)
        } else {
            (0..1usize << exponent)
                .map(|_| (rng.random::<u64>(), rng.random::<u64>()))
                .collect()
        };
        Self::from_inputs(workload, inputs)
    }
    /// Number of multiplications.
    fn len(&self) -> usize {
        match &self.operands {
            Operands::Narrow(inputs) => inputs.len(),
            Operands::Wide(inputs) => inputs.len(),
        }
    }
    /// The operand pairs of a 64-bit-or-narrower workload.
    fn inputs(&self) -> &[(u64, u64)] {
        match &self.operands {
            Operands::Narrow(inputs) => inputs,
            Operands::Wide(_) => {
                panic!("the {} workload has 128-bit operands", self.workload.slug())
            }
        }
    }
    /// The operand pairs of the `u128` workload.
    fn wide_inputs(&self) -> &[(u128, u128)] {
        match &self.operands {
            Operands::Wide(inputs) => inputs,
            Operands::Narrow(_) => {
                panic!("the {} workload has 64-bit operands", self.workload.slug())
            }
        }
    }
    /// The operand pairs of a 32-bit workload as `u32` values.
    fn narrow_inputs(&self) -> Vec<(u32, u32)> {
        narrow(self.inputs())
    }
    fn from_wide_inputs(workload: Workload, inputs: Vec<(u128, u128)>) -> Self {
        assert!(
            workload.is_wide(),
            "{} operands are u64 values",
            workload.slug()
        );
        let digest = common::mul_witness::u128_digest(
            &MulWitness::<u128>::from_inputs(&inputs).expect("canonical u128 witness"),
        );
        Self {
            workload,
            operands: Operands::Wide(inputs),
            digest,
        }
    }
    fn from_inputs(workload: Workload, inputs: Vec<(u64, u64)>) -> Self {
        let digest = match workload {
            Workload::U32 => mod32::digest_rows(
                inputs.iter().map(|&(a, b)| {
                    let a = u32::try_from(a).expect("u32 operand");
                    let b = u32::try_from(b).expect("u32 operand");
                    [u64::from(a), u64::from(b), u64::from(a.wrapping_mul(b))]
                }),
                inputs.len(),
            ),
            Workload::U64 => common::mul_witness::u64_digest(
                &MulWitness::<u64>::from_inputs(&inputs).expect("canonical u64 witness"),
            ),
            Workload::U128 => panic!("the u128 workload has 128-bit operands"),
        };
        Self {
            workload,
            operands: Operands::Narrow(inputs),
            digest,
        }
    }
}

fn narrow(inputs: &[(u64, u64)]) -> Vec<(u32, u32)> {
    inputs
        .iter()
        .map(|&(a, b)| {
            (
                u32::try_from(a).expect("32-bit workload operand"),
                u32::try_from(b).expect("32-bit workload operand"),
            )
        })
        .collect()
}

#[derive(Clone, Debug)]
struct Phase {
    name: &'static str,
    tag: &'static str,
    start: u64,
    end: u64,
}
#[derive(Clone, Debug)]
struct Timing {
    phases: Vec<Phase>,
    proof_bytes: usize,
}
impl Timing {
    fn from_trial(trial: &trace_capture::TrialScopes<'_>, proof_bytes: usize) -> Self {
        let mut t = Timing {
            phases: vec![],
            proof_bytes,
        };
        for (name, tag, span) in [
            ("verified_trial", "end-to-end", trial.verified),
            ("witness_to_proof", "proving", trial.witness_to_proof),
        ] {
            t.add(name, tag, span.start_ns, span.end_ns);
        }
        t.add(
            "online_prover",
            "proving",
            trial.witness.end_ns,
            trial.witness_to_proof.end_ns,
        );
        t.add(
            "witness",
            "witness-generation",
            trial.witness.start_ns,
            trial.witness.end_ns,
        );
        t.add(
            "verify",
            "verification",
            trial.verification.start_ns,
            trial.verification.end_ns,
        );
        t.add(
            "post_proof",
            "proof-accounting",
            trial.witness_to_proof.end_ns,
            trial.verification.start_ns,
        );
        t
    }

    fn new(
        start: u64,
        witness_end: u64,
        ready: u64,
        verify_start: u64,
        end: u64,
        proof_bytes: usize,
    ) -> Self {
        let mut result = Self {
            phases: vec![],
            proof_bytes,
        };
        result.add("verified_trial", "end-to-end", start, end);
        result.add("witness_to_proof", "proving", start, ready);
        result.add("online_prover", "proving", witness_end, ready);
        result.add("witness", "witness-generation", start, witness_end);
        result.add("verify", "verification", verify_start, end);
        result.add("post_proof", "proof-accounting", ready, verify_start);
        result
    }
    fn add(&mut self, name: &'static str, tag: &'static str, start: u64, end: u64) {
        assert!(end >= start, "reversed {name} interval");
        self.phases.push(Phase {
            name,
            tag,
            start,
            end,
        });
    }
    fn union_ms(&self, pred: impl Fn(&Phase) -> bool) -> f64 {
        let mut intervals: Vec<_> = self
            .phases
            .iter()
            .filter(|p| pred(p))
            .map(|p| (p.start, p.end))
            .collect();
        intervals.sort_unstable();
        let (mut total, mut end) = (0, 0);
        for (lo, hi) in intervals {
            if hi > end {
                total += hi - lo.max(end);
                end = hi;
            }
        }
        total as f64 / 1e6
    }
    fn metrics(&self) -> Metrics {
        Metrics {
            witness_ms: self.union_ms(|p| p.tag == "witness-generation"),
            commit_ms: self.union_ms(|p| p.tag == "commit"),
            piop_ms: self.union_ms(|p| p.tag == "constraint-proof"),
            opening_ms: self.union_ms(|p| p.tag == "opening-proof"),
            pcs_ms: self.union_ms(|p| matches!(p.tag, "commit" | "opening-proof")),
            online_prover_ms: self.union_ms(|p| p.name == "online_prover"),
            witness_to_proof_ms: self.union_ms(|p| p.name == "witness_to_proof"),
            verify_ms: self.union_ms(|p| p.tag == "verification"),
            verified_trial_ms: self.union_ms(|p| p.name == "verified_trial"),
            post_proof_ms: self.union_ms(|p| p.name == "post_proof"),
            proof_bytes: self.proof_bytes,
        }
    }
    fn validate(&self) {
        assert!(self.proof_bytes > 0, "missing proof size");
        let root = &self.phases[0];
        for p in &self.phases {
            assert!(p.start >= root.start && p.end <= root.end);
        }
        for tag in [
            "witness-generation",
            "commit",
            "constraint-proof",
            "opening-proof",
            "verification",
        ] {
            assert!(
                self.phases.iter().any(|p| p.tag == tag && p.end > p.start),
                "missing measured {tag} phase"
            );
        }
    }
}

fn captured<'a>(raw: &'a [CapturedSpan], name: &str, lo: u64, hi: u64) -> &'a CapturedSpan {
    raw.iter()
        .filter(|s| {
            (s.name == name || s.component.as_deref() == Some(name))
                && s.start_ns >= lo
                && s.end_ns <= hi
        })
        .min_by_key(|s| s.start_ns)
        .unwrap_or_else(|| panic!("missing native prover span {name}"))
}

enum Context {
    Binius(binius::Context),
    BiniusLigerito(binius_ligerito::Context),
    Plonky3Fri(plonky3::Context),
    Plonky3Whir(plonky3_whir::Context),
    Limber(limber::Context),
}
impl Context {
    fn setup(
        backend: Backend,
        corpus: Arc<Corpus>,
        rate: usize,
        accounting: bitz::binius_ligerito::Accounting,
    ) -> anyhow::Result<Self> {
        Ok(match backend {
            Backend::Bitz | Backend::Plonky3Whir => unreachable!("prepared separately"),
            Backend::Binius => Self::Binius(binius::Context::setup_at_rate(corpus, rate)),
            Backend::BiniusLigerito => Self::BiniusLigerito(
                binius_ligerito::Context::setup_at_rate(corpus, rate, accounting).map_err(
                    |error| match error {
                        bitz::binius_ligerito::Error::Config(reason) => {
                            anyhow::Error::new(super::Unsupported(reason))
                        }
                        error => anyhow::Error::new(error),
                    },
                )?,
            ),
            Backend::Plonky3Fri => Self::Plonky3Fri(plonky3::Context::setup_at_rate(corpus, rate)),
            Backend::Limber => Self::Limber(limber::Context::setup(corpus)),
        })
    }
    fn run(&self) -> Timing {
        match self {
            Self::Binius(c) => c.run(),
            Self::BiniusLigerito(c) => c.run(),
            Self::Plonky3Fri(c) => c.run(),
            Self::Plonky3Whir(c) => c.run(),
            Self::Limber(c) => c.run(),
        }
    }
    fn config(&self) -> Value {
        match self {
            Self::Binius(c) => c.config(),
            Self::BiniusLigerito(c) => c.config(),
            Self::Plonky3Fri(c) => c.config(),
            Self::Plonky3Whir(c) => c.config(),
            Self::Limber(c) => c.config(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Backend {
    Bitz,
    #[value(name = "binius64")]
    Binius,
    #[value(name = "binius64-ligerito")]
    BiniusLigerito,
    Plonky3Fri,
    Plonky3Whir,
    Limber,
}

impl Backend {
    fn slug(self) -> &'static str {
        match self {
            Self::Bitz => "bitz",
            Self::Binius => "binius64",
            Self::BiniusLigerito => "binius64-ligerito",
            Self::Plonky3Fri => "plonky3-fri",
            Self::Plonky3Whir => "plonky3-whir",
            Self::Limber => "limber",
        }
    }
}

struct WitnessAudit {
    digest: String,
    generation_ms: f64,
    representation: &'static str,
    quotient_reconstructed: bool,
}
impl WitnessAudit {
    /// Audits the native rows `[x, y, z_lo, z_hi]` of the `u128` workload
    /// against the corpus digest (32-byte entries, see `u128_digest`).
    fn check_wide(
        corpus: &Corpus,
        rows: Vec<[u128; 4]>,
        generation_ms: f64,
        representation: &'static str,
    ) -> Self {
        let inputs = corpus.wide_inputs();
        assert_eq!(rows.len(), inputs.len(), "native witness row count");
        let layout = MulLayout::<u128>::new(rows.len()).expect("canonical u128 layout");
        let capacity = layout.capacity();
        let mut hash = blake3::Hasher::new();
        hash.update(b"bitz/u128-mul-compare/integer-witness/v1");
        hash.update(&(layout.assignment_len() as u64).to_le_bytes());
        let mut entries = vec![(0_u128, 0_u128); layout.assignment_len()];
        entries[0] = (1, 0);
        for (i, (&[x, y, lo, hi], &(expected_x, expected_y))) in rows.iter().zip(inputs).enumerate()
        {
            assert_eq!(
                [x, y, lo, hi],
                corpus.workload.wide_row(expected_x, expected_y),
                "native witness mismatch at row {i} ({representation})"
            );
            entries[capacity + i] = (x, 0);
            entries[2 * capacity + i] = (y, 0);
            entries[3 * capacity + i] = (lo, hi);
        }
        for (lo, hi) in entries {
            hash.update(&lo.to_le_bytes());
            hash.update(&hi.to_le_bytes());
        }
        let digest = hash.finalize().to_hex().to_string();
        assert_eq!(digest, corpus.digest, "recovered native assignment digest");
        Self {
            digest,
            generation_ms,
            representation,
            quotient_reconstructed: false,
        }
    }
    fn check(
        corpus: &Corpus,
        rows: Vec<[u64; 4]>,
        generation_ms: f64,
        representation: &'static str,
        quotient_reconstructed: bool,
    ) -> Self {
        let inputs = corpus.inputs();
        assert_eq!(rows.len(), inputs.len(), "native witness row count");
        let n = rows.len();
        if corpus.workload == Workload::U32 {
            for (i, (row, &(x, y))) in rows.iter().zip(inputs).enumerate() {
                assert_eq!(
                    &row[..3],
                    &corpus.workload.native_row(x, y)[..3],
                    "native modular witness mismatch at row {i} ({representation})"
                );
            }
            let digest = mod32::digest_rows(rows.iter().map(|r| [r[0], r[1], r[2]]), n);
            assert_eq!(digest, corpus.digest, "recovered native assignment digest");
            return Self {
                digest,
                generation_ms,
                representation,
                quotient_reconstructed,
            };
        }
        assert_eq!(corpus.workload, Workload::U64);
        let layout = MulLayout::<u64>::new(n).expect("canonical u64 layout");
        let (capacity, assignment_len) = (layout.capacity(), layout.assignment_len());
        let mut assignment = vec![0u64; assignment_len];
        assignment[0] = 1;
        let has_fourth_block = true;
        for (i, (&[a, b, c, q], &(expected_a, expected_b))) in rows.iter().zip(inputs).enumerate() {
            assert_eq!(
                [a, b, c, q],
                corpus.workload.native_row(expected_a, expected_b),
                "native witness mismatch at row {i} ({representation})"
            );
            assignment[capacity + i] = a;
            assignment[2 * capacity + i] = b;
            assignment[3 * capacity + i] = c;
            if has_fourth_block {
                assignment[4 * capacity + i] = q;
            }
        }
        let domain: &[u8] = b"bitz/u64-mul-compare/integer-witness/v1";
        let mut hash = blake3::Hasher::new();
        hash.update(domain);
        hash.update(&(assignment.len() as u64).to_le_bytes());
        for value in assignment {
            hash.update(&value.to_le_bytes());
        }
        let digest = hash.finalize().to_hex().to_string();
        assert_eq!(digest, corpus.digest, "recovered native assignment digest");
        Self {
            digest,
            generation_ms,
            representation,
            quotient_reconstructed,
        }
    }
}
fn audit_backend(backend: Backend, corpus: &Corpus) -> WitnessAudit {
    match backend {
        Backend::Bitz => unreachable!("BitZ witnesses use the shared witness loop"),
        // The same Binius64 circuit and witness filler; only the opener differs.
        Backend::Binius | Backend::BiniusLigerito => binius::audit(corpus),
        Backend::Plonky3Fri | Backend::Plonky3Whir => mod32_air::audit(corpus),
        Backend::Limber => limber::audit(corpus),
    }
}

/// Sixteen boundary gates of a workload: zero, one, and the largest
/// operand in every combination.
#[cfg(test)]
#[allow(dead_code)] // `cargo bench` sets cfg(test) without running the #[test] callers.
fn edge_corpus(workload: Workload) -> Corpus {
    if workload.is_wide() {
        let max = u128::MAX;
        return Corpus::from_wide_inputs(
            workload,
            [(0, 0), (0, max), (1, max), (max, max)].repeat(4),
        );
    }
    let max = match workload {
        Workload::U32 => u64::from(u32::MAX),
        Workload::U64 => u64::MAX,
        Workload::U128 => unreachable!(),
    };
    Corpus::from_inputs(workload, [(0, 0), (0, max), (1, max), (max, max)].repeat(4))
}

#[cfg(test)]
#[allow(unused_imports)]
mod witness_tests {
    use super::*;
    #[test]
    fn all_native_witnesses_recover_the_same_assignment() {
        let _trace = common::test_tracing();
        for workload in [Workload::U32, Workload::U64, Workload::U128] {
            let corpus = edge_corpus(workload);
            for backend in [
                Backend::Binius,
                Backend::BiniusLigerito,
                Backend::Plonky3Fri,
                Backend::Plonky3Whir,
            ] {
                if !workload.supports(backend) {
                    continue;
                }
                assert_eq!(audit_backend(backend, &corpus).digest, corpus.digest);
            }
        }
    }
    #[test]
    fn witness_check_rejects_a_changed_native_row() {
        let corpus = Corpus::new(Workload::U32, 4, 7);
        let mut rows: Vec<_> = corpus
            .inputs()
            .iter()
            .map(|&(a, b)| corpus.workload.native_row(a, b))
            .collect();
        rows[0][2] ^= 1;
        assert!(
            std::panic::catch_unwind(|| WitnessAudit::check(
                &corpus,
                rows,
                0.0,
                "corrupted native witness",
                false
            ))
            .is_err()
        );
    }
}

pub(super) fn run(run: &mut Run) -> anyhow::Result<()> {
    let case = &run.job.case;
    let workload = match case.workload {
        config::Workload::U32Mod32 => Workload::U32,
        config::Workload::U64 => Workload::U64,
        config::Workload::U128 => Workload::U128,
        _ => unreachable!("validated native workload"),
    };
    let backend =
        <Backend as clap::ValueEnum>::from_str(&case.backend, false).map_err(anyhow::Error::msg)?;
    let corpus = Arc::new(Corpus::new(
        workload,
        case.log_n,
        super::proof::shape_seed(case),
    ));
    if case.mode == config::Mode::Witness {
        for i in 0..run.trials() {
            run.begin_memory();
            let audit = audit_backend(backend, &corpus);
            run.end_memory();
            run.sample(
                i,
                RawMetrics::from([("witness_ms".into(), audit.generation_ms)]),
            );
            run.effective = json!({"corpus_digest":audit.digest,"boundary":"witness-generation","representation":audit.representation,"quotient_reconstructed":audit.quotient_reconstructed});
        }
        return Ok(());
    }
    let params = if backend == Backend::Plonky3Whir {
        let selected = if let Some(p) = case.whir {
            common::whir_tuning::Params {
                extension_degree: p.degree,
                folding: p.folding,
                starting_log_inv_rate: p.log_inv_rate,
                max_pow_bits: p.max_pow_bits,
                max_round_log_inv_rate: p.max_round_log_inv_rate,
            }
        } else {
            anyhow::ensure!(
                run.latency(),
                "heap-only WHIR requires explicit --whir-degree/--whir-folding/--whir-pow settings from the latency run"
            );
            let (params, report) = common::whir_tuning::tune_with_reps(
                &[2, 5],
                None,
                run.job.tuning_reps,
                |params| {
                    if case
                        .log_inv_rate
                        .is_some_and(|rate| params.starting_log_inv_rate != rate as usize)
                    {
                        return Err("rate outside requested configuration".into());
                    }
                    plonky3_whir::Context::setup_with_params(Arc::clone(&corpus), params)
                },
                |c| c.run().metrics().witness_to_proof_ms,
                plonky3_whir::Context::security,
            )
            .map_err(|error| {
                if error.starts_with("no eligible WHIR configuration:") {
                    anyhow::Error::new(super::Unsupported(error))
                } else {
                    anyhow::Error::msg(error)
                }
            })?;
            run.tuning = Some(serde_json::to_value(report)?);
            params
        };
        run.job.case.whir = Some(config::WhirConfig {
            degree: selected.extension_degree,
            folding: selected.folding,
            log_inv_rate: selected.starting_log_inv_rate,
            max_pow_bits: selected.max_pow_bits,
            max_round_log_inv_rate: selected.max_round_log_inv_rate,
        });
        Some(selected)
    } else {
        None
    };
    let started = std::time::Instant::now();
    let context = if let Some(params) = params {
        Context::Plonky3Whir(
            plonky3_whir::Context::setup_with_params(Arc::clone(&corpus), params)
                .map_err(super::Unsupported)?,
        )
    } else {
        Context::setup(
            backend,
            Arc::clone(&corpus),
            run.job.case.log_inv_rate.unwrap_or(1) as usize,
            if run.job.case.binius_ligerito_accounting.as_deref() == Some("rbr") {
                bitz::binius_ligerito::Accounting::RoundByRound
            } else {
                bitz::binius_ligerito::Accounting::UnionBound
            },
        )?
    };
    let setup_ms = started.elapsed().as_secs_f64() * 1000.;
    run.effective = json!({"config":context.config(),"corpus_digest":corpus.digest,"boundary":"witness-to-proof","whir_params":params});
    for i in 0..run.trials() {
        run.begin_memory();
        if !run.latency() {
            let bytes = match &context {
                Context::Binius(c) => c.prove_and_verify(),
                Context::BiniusLigerito(c) => c.prove_and_verify(),
                Context::Plonky3Fri(c) => c.prove_and_verify(),
                Context::Plonky3Whir(c) => c.prove_and_verify(),
                Context::Limber(c) => c.prove_and_verify(),
            };
            run.end_memory();
            anyhow::ensure!(bytes > 0, "missing verified proof size");
        } else {
            let timing = context.run();
            timing.validate();
            let mut metrics: RawMetrics =
                serde_json::from_value(serde_json::to_value(timing.metrics())?)?;
            metrics.insert("setup_ms".into(), setup_ms);
            run.sample(i, metrics);
        }
    }
    Ok(())
}
