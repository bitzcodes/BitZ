//! Matched raw SHA-256 chains: public blocks and final state, standard IV,
//! private intermediate states, no padding or ECDSA. One case per process.
#[path = "common/cli.rs"]
mod cli;
#[allow(dead_code)]
#[path = "common/trace_capture.rs"]
mod trace_capture;

use binius_circuits::sha256::compress::{State, ref_compress, sha256_compress_2x_seq};
use binius_core::{constraint_system::ValueVec, word::Word};
use binius_frontend::{Circuit, CircuitBuilder, Wire};
use binius_hash::sha256::Sha256HashSuite;
use binius_prover::{OptimalPackedB128, Prover};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{Verifier, config::StdChallenger};
use bitz::{
    binius_ligerito::{Accounting, Prepared},
    ligerito_flock::LigeritoSelection,
    observability::{self, Interval, Recording},
    piop::spartan::{
        Sha256ChainStatement, commit_sha256_chain_witness_with_config,
        generate_sha256_chain_witnesses, prepare_sha256_chain_batch,
        prove_sha256_chain_with_config, sha256_chain_configs, sha256_compress,
        verify_sha256_chain_with_config,
    },
    transcript::Blake3Transcript,
};
use serde_json::{Value, json};
use std::error::Error;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SCHEMA: &str = "bitz/sha256-chain-compare/v1";
const FIXTURE: &str = "sha256-chain/public-blocks-standard-iv/v1";
const IV: [u32; 8] = circuit::sha256::INITIAL_STATE;

#[derive(clap::Parser)]
struct Args {
    #[command(flatten)]
    cargo: cli::CargoArgs,
    #[arg(long, default_value = "bitz", value_parser = ["bitz", "binius64", "binius64-ligerito"])]
    method: String,
    #[arg(long, value_parser = clap::value_parser!(u8).range(7..=16))]
    exponent: u8,
    #[arg(long, default_value = "1", value_parser = cli::positive)]
    threads: usize,
    #[arg(long, default_value = "5", value_parser = cli::positive)]
    reps: usize,
    #[arg(long, default_value = "0")]
    seed: u64,
    #[arg(long, default_value = "custom:1:4", value_parser = ["custom:1:4", "custom:3:4"])]
    bitz_profile: String,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=3))]
    log_inv_rate: u8,
    /// Describe the deterministic public statement without setup or proving.
    #[arg(long)]
    fixture_only: bool,
    /// Check statement/proof mutations on the warmup; intended for correctness tests.
    #[arg(long)]
    self_test: bool,
}

struct Fixture {
    statement: Sha256ChainStatement,
    id: String,
}

impl Fixture {
    fn generate(exponent: u8, seed: u64) -> Result<Self> {
        let mut state = seed;
        let blocks: Vec<[u32; 16]> = (0..1usize << exponent)
            .map(|_| {
                std::array::from_fn(|_| {
                    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    (z ^ (z >> 31)) as u32
                })
            })
            .collect();
        let mut digest = IV;
        for block in &blocks {
            let next = sha256_compress(digest, *block);
            if next != ref_compress(digest, *block) {
                return Err("native SHA implementations disagree".into());
            }
            digest = next;
        }
        let statement = Sha256ChainStatement { blocks, digest };
        // Canonical fixture identity: domain, exponent, seed LE, then SHA words BE.
        let mut hash = blake3::Hasher::new();
        hash.update(FIXTURE.as_bytes());
        hash.update(&[exponent]);
        hash.update(&seed.to_le_bytes());
        for word in statement.blocks.iter().flatten().chain(&statement.digest) {
            hash.update(&word.to_be_bytes());
        }
        Ok(Self {
            statement,
            id: hash.finalize().to_hex().to_string(),
        })
    }

    fn description(&self) -> Value {
        json!({"fixture_id":self.id, "fixture_profile":FIXTURE,
            "compressions":self.statement.blocks.len(), "digest":self.statement.digest,
            "message_bytes":64*self.statement.blocks.len(),
            "statement_bytes":64*self.statement.blocks.len()+32,
            "statement":"public-blocks-and-final-state", "padding":false, "signatures":0})
    }

    fn public_words(&self) -> Vec<Word> {
        self.statement
            .blocks
            .iter()
            .flatten()
            .chain(&self.statement.digest)
            .map(|&v| Word(u64::from(v)))
            .collect()
    }
}

struct BiniusChain {
    circuit: Circuit,
    blocks: Vec<[Wire; 16]>,
    digest: [Wire; 8],
}

impl BiniusChain {
    fn new(count: usize) -> Result<Self> {
        if count < 2 || !count.is_power_of_two() {
            return Err("chain length must be an even power of two".into());
        }
        let builder = CircuitBuilder::new();
        let blocks: Vec<[Wire; 16]> = (0..count)
            .map(|_| std::array::from_fn(|_| builder.add_inout()))
            .collect();
        let digest = std::array::from_fn(|_| builder.add_inout());
        let mut state = State::iv(&builder);
        for pair in blocks.chunks_exact(2) {
            state = sha256_compress_2x_seq(&builder, state, [pair[0], pair[1]]);
        }
        let mask = builder.add_constant(Word(u32::MAX as u64));
        for (actual, expected) in state.0.into_iter().zip(digest) {
            builder.assert_eq(
                "final_sha_chaining_value",
                builder.band(actual, mask),
                expected,
            );
        }
        let circuit = builder.build();
        circuit.constraint_system().validate()?;
        Ok(Self {
            circuit,
            blocks,
            digest,
        })
    }

    fn populate(&self, fixture: &Fixture) -> Result<ValueVec> {
        if fixture.statement.blocks.len() != self.blocks.len() {
            return Err("chain length mismatch".into());
        }
        let mut filler = self.circuit.new_witness_filler();
        for (&wire, &value) in self.blocks.iter().flatten().chain(&self.digest).zip(
            fixture
                .statement
                .blocks
                .iter()
                .flatten()
                .chain(&fixture.statement.digest),
        ) {
            filler[wire] = Word(u64::from(value));
        }
        self.circuit.populate_wire_witness(&mut filler)?;
        Ok(filler.into_value_vec())
    }
}

enum Backend {
    Basefold {
        prover: Prover<OptimalPackedB128, Sha256HashSuite>,
        verifier: Verifier<Sha256HashSuite>,
    },
    Ligerito(Prepared),
}

impl Backend {
    fn verify(&self, public: &[Word], bytes: &[u8]) -> Result<()> {
        match self {
            Self::Basefold { verifier, .. } => {
                let mut transcript =
                    VerifierTranscript::new(StdChallenger::default(), bytes.to_vec());
                verifier.verify(public, &mut transcript)?;
                transcript.finalize()?;
            }
            Self::Ligerito(prepared) => {
                prepared.verify(public, &prepared.proof_from_bytes(bytes)?)?
            }
        }
        Ok(())
    }
}

fn ms(intervals: &[Interval], label: &str) -> Result<f64> {
    Ok(observability::duration(intervals, label)?.as_secs_f64() * 1000.)
}

fn emit(
    args: &Args,
    fixture: &Fixture,
    trial: usize,
    security: &Value,
    setup_ms: f64,
    intervals: &[Interval],
    packing_ms: f64,
    commit_ms: f64,
    proof_bytes: usize,
    proof_size_kind: &str,
) -> Result<()> {
    let mut row = fixture.description();
    row.as_object_mut().unwrap().extend(json!({
        "schema":SCHEMA, "method":args.method, "log_compressions":args.exponent,
        "threads":args.threads, "seed":args.seed, "security_target":100,
        "ligerito_profile":if args.method == "bitz" {Some(args.bitz_profile.as_str())} else {None},
        "log_inv_rate":if args.method == "bitz" {args.bitz_profile.split(':').nth(1).unwrap().parse::<u8>()?} else {args.log_inv_rate},
        "trial":if trial == 0 {"warmup"} else {"sample"}, "sample":trial,
        "verified":true, "zk":false, "timing":"perfetto", "security":security,
        "setup_ms":setup_ms, "witness_ms":ms(intervals,"chain-compare:witness")?+packing_ms,
        "commit_ms":commit_ms, "prove_ms":ms(intervals,"chain-compare:prove")?-packing_ms,
        "e2e_prover_ms":ms(intervals,"chain-compare:e2e")?,
        "verify_ms":ms(intervals,"chain-compare:verify")?,
        "proof_bytes":proof_bytes, "proof_size_kind":proof_size_kind,
        "phases_ms":observability::phase_totals(intervals,"chain-compare:prove")?
            .into_iter().map(|(k,v)|(k,v*1000.)).collect::<std::collections::BTreeMap<_,_>>(),
    }).as_object().unwrap().clone());
    println!("{row}");
    Ok(())
}

fn run_bitz(args: &Args, fixture: &Fixture) -> Result<()> {
    let setup_recording = Recording::start(Vec::new())?;
    let setup = tracing::info_span!("chain-compare:setup").entered();
    let prepared = prepare_sha256_chain_batch(args.exponent.into())?
        .with_ligerito(LigeritoSelection::parse(&args.bitz_profile, 100)?)?;
    let (pc, vc) = sha256_chain_configs(&prepared)?;
    drop(setup);
    let setup_ms = ms(&setup_recording.intervals()?, "chain-compare:setup")?;
    let security = json!({"model":"BitZ per-check economic accounting",
        "economic_bits":prepared.security().accounting.achieved_bits(),
        "ligerito":prepared.ligerito_configuration()?.report(&args.bitz_profile, prepared.security().ood)});
    for trial in 0..=args.reps {
        let recording = Recording::start(Vec::new())?;
        let e2e = tracing::info_span!("chain-compare:e2e").entered();
        let witness = tracing::info_span!("chain-compare:witness")
            .in_scope(|| generate_sha256_chain_witnesses(&prepared, &fixture.statement.blocks))?;
        let proving = tracing::info_span!("chain-compare:prove").entered();
        let hint = tracing::info_span!("chain-compare:commit")
            .in_scope(|| commit_sha256_chain_witness_with_config(&prepared, &witness, &pc))?;
        let mut proof = prove_sha256_chain_with_config(
            &mut Blake3Transcript::new(),
            &prepared,
            &fixture.statement,
            &witness,
            &hint,
            &pc,
        )?;
        drop(proving);
        drop(e2e);
        tracing::info_span!("chain-compare:verify").in_scope(|| {
            verify_sha256_chain_with_config(
                &mut Blake3Transcript::new(),
                &prepared,
                &fixture.statement,
                &hint.commitment,
                &proof,
                &vc,
            )
        })?;
        let intervals = recording.intervals()?;
        // PIOP is an analytical payload count; PCS and commitment are serialized.
        let proof_bytes = proof.piop_bytes()
            + proof.bitz().to_bytes().len()
            + bincode::serialized_size(&hint.commitment)? as usize;
        if args.self_test && trial == 0 {
            for position in [
                0,
                fixture.statement.blocks.len() - 1,
                fixture.statement.blocks.len(),
            ] {
                let mut wrong = fixture.statement.clone();
                if position == wrong.blocks.len() {
                    wrong.digest[0] ^= 1;
                } else {
                    wrong.blocks[position][0] ^= 1;
                }
                if verify_sha256_chain_with_config(
                    &mut Blake3Transcript::new(),
                    &prepared,
                    &wrong,
                    &hint.commitment,
                    &proof,
                    &vc,
                )
                .is_ok()
                {
                    return Err("BitZ accepted changed public chain statement".into());
                }
            }
            *proof.initial_nonce_mut() ^= 1;
            if verify_sha256_chain_with_config(
                &mut Blake3Transcript::new(),
                &prepared,
                &fixture.statement,
                &hint.commitment,
                &proof,
                &vc,
            )
            .is_ok()
            {
                return Err("BitZ accepted changed proof".into());
            }
        }
        emit(
            args,
            fixture,
            trial,
            &security,
            setup_ms,
            &intervals,
            0.,
            ms(&intervals, "chain-compare:commit")?,
            proof_bytes,
            "analytical-piop-plus-serialized-pcs-and-commitment",
        )?;
    }
    Ok(())
}

fn run_binius(args: &Args, fixture: &Fixture) -> Result<()> {
    let setup_recording = Recording::start(Vec::new())?;
    let setup = tracing::info_span!("chain-compare:setup").entered();
    let relation = BiniusChain::new(fixture.statement.blocks.len())?;
    let (backend, security) = if args.method == "binius64" {
        let verifier = Verifier::<Sha256HashSuite>::setup_with_security_bits(
            relation.circuit.constraint_system().clone(),
            args.log_inv_rate.into(),
            100,
        )?;
        let security = json!({"model":"Binius BaseFold query target; not whole-proof accounting",
            "pcs":"BaseFold", "fri_query_target_bits":100,
            "log_inv_rate":verifier.fri_params().rs_code().log_inv_rate(),
            "fri_queries":verifier.fri_params().n_test_queries(), "merkle_hash":"SHA-256"});
        let prover = Prover::setup(verifier.clone())?;
        (Backend::Basefold { prover, verifier }, security)
    } else {
        let prepared = Prepared::with_options(
            relation.circuit.constraint_system(),
            args.log_inv_rate.into(),
            Accounting::RoundByRound,
        )?;
        let s = prepared.security();
        let security = json!({"model":"Binius64 PIOP with BitZ-Ligerito; per-round algebraic accounting",
            "pcs":"BitZ-Ligerito", "accounting":s.accounting.name(), "target_bits":s.target_bits,
            "algebraic_bits":s.algebraic_bits, "log_inv_rate":args.log_inv_rate,
            "union_bound_bits":s.union_bound_bits, "round_by_round_bits":s.round_by_round_bits,
            "merkle_hash":"BLAKE3"});
        (Backend::Ligerito(prepared), security)
    };
    drop(setup);
    let setup_ms = ms(&setup_recording.intervals()?, "chain-compare:setup")?;
    let public = fixture.public_words();
    for trial in 0..=args.reps {
        let recording = Recording::start(Vec::new())?;
        let e2e = tracing::info_span!("chain-compare:e2e").entered();
        let witness =
            tracing::info_span!("chain-compare:witness").in_scope(|| relation.populate(fixture))?;
        let proving = tracing::info_span!("chain-compare:prove").entered();
        let mut lig_proof = None;
        let mut bytes = Vec::new();
        match &backend {
            Backend::Basefold { prover, .. } => {
                let mut transcript = ProverTranscript::new(StdChallenger::default());
                prover.prove(&witness, &mut transcript)?;
                bytes = transcript.finalize();
            }
            Backend::Ligerito(prepared) => lig_proof = Some(prepared.prove(&witness)?),
        }
        drop(proving);
        drop(e2e);
        if witness
            .inout()
            .iter()
            .map(|w| w.0)
            .ne(public.iter().map(|w| w.0))
        {
            return Err("Binius public inputs disagree with the fixture".into());
        }
        if let Some(proof) = lig_proof {
            bytes = proof.to_bytes();
        }
        tracing::info_span!("chain-compare:verify").in_scope(|| backend.verify(&public, &bytes))?;
        let intervals = recording.intervals()?;
        let (packing_ms, commit_ms) = match &backend {
            Backend::Basefold { .. } => (
                ms(&intervals, "prepare_witness")?,
                ms(&intervals, "commit_witness")?,
            ),
            Backend::Ligerito(_) => {
                let phases = trace_capture::BiniusLigeritoPhases::from_spans(&intervals);
                (
                    ms(&intervals, "prepare_witness")?,
                    (phases.commit.1 - phases.commit.0) as f64 / 1e6,
                )
            }
        };
        if args.self_test && trial == 0 {
            for position in [0, public.len() - 9, public.len() - 1] {
                let mut wrong = public.clone();
                wrong[position].0 ^= 1;
                if backend.verify(&wrong, &bytes).is_ok() {
                    return Err("Binius accepted changed public chain statement".into());
                }
            }
            let mut corrupt = bytes.clone();
            corrupt[0] ^= 1;
            let mut extended = bytes.clone();
            extended.push(0);
            for wrong in [&corrupt[..], &extended[..], &bytes[..bytes.len() - 1]] {
                if backend.verify(&public, wrong).is_ok() {
                    return Err("Binius accepted corrupt/trailing/truncated proof".into());
                }
            }
        }
        emit(
            args,
            fixture,
            trial,
            &security,
            setup_ms,
            &intervals,
            packing_ms,
            commit_ms,
            bytes.len(),
            "serialized-proof-including-commitments",
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    use clap::Parser;
    let args = Args::parse();
    let fixture = Fixture::generate(args.exponent, args.seed)?;
    if args.fixture_only {
        println!("{}", fixture.description());
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    observability::install()?;
    if args.method == "bitz" {
        run_bitz(&args, &fixture)
    } else {
        run_binius(&args, &fixture)
    }
}

#[cfg(test)]
mod tests {
    // harness=false bench builds enable cfg(test) without collecting #[test] functions.
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use clap::Parser;
    #[test]
    fn fixture_matches_sha256_standard_vector() {
        let mut block = [0u32; 16];
        block[0] = 0x61626380;
        block[15] = 24;
        let expected = [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
            0xf20015ad,
        ];
        assert_eq!(sha256_compress(IV, block), expected);
        assert_eq!(ref_compress(IV, block), expected);
        assert_eq!(
            Fixture::generate(7, 0).unwrap().id,
            Fixture::generate(7, 0).unwrap().id
        );
        assert_ne!(
            Fixture::generate(7, 0).unwrap().id,
            Fixture::generate(7, 1).unwrap().id
        );
    }
    #[test]
    fn matched_chain_proofs_reject_statement_and_proof_mutations() {
        observability::install().unwrap();
        let fixture = Fixture::generate(7, 7).unwrap();
        for rate in [1, 3] {
            for method in ["bitz", "binius64", "binius64-ligerito"] {
                let mut args = Args::parse_from([
                    "test",
                    "--exponent",
                    "7",
                    "--method",
                    method,
                    "--self-test",
                ]);
                args.reps = 0; // One correctness proof per backend/rate, no measured samples.
                args.seed = 7;
                args.threads = rayon::current_num_threads();
                args.log_inv_rate = rate;
                args.bitz_profile = format!("custom:{rate}:4");
                if method == "bitz" {
                    run_bitz(&args, &fixture)
                } else {
                    run_binius(&args, &fixture)
                }
                .unwrap();
            }
        }
    }
}
