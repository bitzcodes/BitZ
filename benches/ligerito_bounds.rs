//! Controlled decoding-bound experiment within BitZ. No competing backend configuration is read.
use ::bitz::ligerito_flock::IntEvalRsLigModQProof;
use ::bitz::ligerito_flock::IntEvalRsLigVirtProof;

mod common;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
macro_rules! bail {
    ($message:expr) => {
        return Err($message.into())
    };
}
use ::bitz::observability::{self, Recording};
use ::bitz::{
    ligerito_flock::{LigeritoSelection, ResolvedLigerito},
    piop::spartan::*,
    transcript::Blake3Transcript,
};
use serde_json::json;
const SEED: u64 = 0x5533_3250_4353_0064;
type SetupCapture = (Recording<Vec<u8>>, tracing::span::EnteredSpan);

struct Experiment {
    case: String,
    selection: LigeritoSelection,
    memory: bool,
}
impl Experiment {
    fn run<W, H, P>(
        &self,
        setup: SetupCapture,
        resolved: &ResolvedLigerito,
        ood: Option<::bitz::ligerito_flock::OodRoundParams>,
        corpus: &[u8],
        witness: impl Fn() -> Result<W>,
        commit: impl Fn(&W) -> Result<H>,
        prove: impl Fn(&W, &H) -> Result<P>,
        verify: impl Fn(&W, &H, &P) -> Result<()>,
        size: impl Fn(&H, &P) -> Result<ProofSize>,
    ) -> Result<()> {
        drop(setup.1);
        let setup_ms =
            observability::duration(&setup.0.intervals()?, "bounds:setup")?.as_secs_f64() * 1000.;
        let mut config = common::ligerito_report(resolved, ood);
        config["requested_profile"] = json!(self.selection.name());
        let corpus_digest = blake3::hash(corpus).to_hex().to_string();
        let trials = if self.memory { 1 } else { 6 };
        for trial in 0..trials {
            let recording = Recording::start(Vec::new())?;
            let total = tracing::info_span!("bounds:witness_to_proof", trial, warmup = trial == 0)
                .entered();
            let w = tracing::info_span!("bounds:witness").in_scope(&witness)?;
            let online = tracing::info_span!("bounds:online").entered();
            let h = tracing::info_span!("bounds:commit").in_scope(|| commit(&w))?;
            let p = prove(&w, &h)?;
            drop(online);
            drop(total);
            // Codecs and byte accounting are outside all proof/verification timers.
            let proof_size = size(&h, &p)?;
            tracing::info_span!("bounds:verification").in_scope(|| verify(&w, &h, &p))?;
            let intervals = recording.intervals()?;
            let millis =
                |name| observability::duration(&intervals, name).map(|d| d.as_secs_f64() * 1000.);
            let witness_ms = millis("bounds:witness")?;
            let commit_ms = millis("bounds:commit")?;
            let online_prover_ms = millis("bounds:online")?;
            let witness_to_proof_ms = millis("bounds:witness_to_proof")?;
            let verify_ms = millis("bounds:verification")?;
            println!(
                "{}",
                json!({"schema":"bitz-ligerito-bound-comparison/v1", "case":self.case,
                "trial":if self.memory {"memory"} else if trial==0 {"warmup"} else {"sample"}, "index":trial,
                "seed":SEED,"corpus_digest":corpus_digest,"threads":rayon::current_num_threads(),"measurement_policy":"one-setup/one-warmup/five-proofs/v1",
                "ligerito":config,"setup_ms":setup_ms,"witness_ms":witness_ms,"commit_ms":commit_ms,
                "online_prover_ms":online_prover_ms,"witness_to_proof_ms":witness_to_proof_ms,
                "verify_ms":verify_ms,"proof_size":proof_size,"verified":true})
            );
        }
        Ok(())
    }
}
#[derive(serde::Serialize)]
struct ProofSize {
    total_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    commitment_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opening_codec_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    analytical_piop_bytes: Option<usize>,
    accounting: &'static str,
}

fn bytes(root: usize, opening: &[u8], analytic: usize) -> ProofSize {
    ProofSize {
        total_bytes: root + opening.len() + analytic,
        commitment_bytes: Some(root),
        opening_codec_bytes: Some(opening.len()),
        analytical_piop_bytes: Some(analytic),
        accounting: "initial commitment + versioned opening codec + explicitly counted PIOP payload",
    }
}
fn inputs() -> Vec<(u128, u128)> {
    let mut h = blake3::Hasher::new();
    h.update(b"bitz/ligerito-bound-inputs/v1");
    h.update(&SEED.to_le_bytes());
    let mut r = h.finalize_xof();
    (0..1 << 15)
        .map(|_| {
            let mut b = [0; 32];
            r.fill(&mut b);
            (
                u128::from_le_bytes(b[..16].try_into().unwrap()),
                u128::from_le_bytes(b[16..].try_into().unwrap()),
            )
        })
        .collect()
}
#[derive(clap::Parser)]
struct Args {
    #[command(flatten)]
    cargo: common::cli::CargoArgs,
    #[arg(value_parser = ["sha-compression", "sha-chain", "ecdsa-split", "ecdsa-all", "hybrid-15-7", "hybrid-15-2", "pcs-22"])]
    case: String,
    profile: String,
    #[arg(long)]
    memory: bool,
}

fn main() -> Result<()> {
    let args = <Args as clap::Parser>::parse();
    let e = Experiment {
        selection: LigeritoSelection::parse(
            &args.profile,
            if args.case.starts_with("hybrid") {
                106
            } else {
                100
            },
        )?,
        case: args.case,
        memory: args.memory,
    };
    ::bitz::observability::install().expect("install Perfetto subscriber");
    if rayon::current_num_threads() != 8 {
        bail!("controlled comparison requires RAYON_NUM_THREADS=8");
    }
    let input = inputs();
    let setup = (
        Recording::start(Vec::new())?,
        tracing::info_span!("bounds:setup").entered(),
    );
    match e.case.as_str() {
        "sha-compression" => {
            let p = sha256::prepare_sha256_compression_batch(7)?.with_ligerito(e.selection)?;
            let source: Vec<_> = (0..128)
                .map(|i| ([i as u32; 8], std::array::from_fn(|j| (i * 16 + j) as u32)))
                .collect();
            e.run(
                setup,
                p.ligerito_configuration()?,
                p.security().ood,
                &bincode::serialize(&source)?,
                || {
                    let w = sha256::generate_sha256_compression_witnesses(&p, &source)?;
                    let statement: Vec<_> = source
                        .iter()
                        .copied()
                        .zip(w.outputs().iter().copied())
                        .map(|(i, o)| sha256::Sha256CompressionStatement::new(i, o))
                        .collect();
                    Ok((w, statement))
                },
                |w| Ok(sha256::commit_sha256_compression_witness(&p, &w.0)?),
                |w, h| {
                    Ok(sha256::prove_sha256_compressions(
                        &mut Blake3Transcript::new(),
                        &p,
                        &w.1,
                        &w.0,
                        h,
                    )?)
                },
                |w, h, proof| {
                    Ok(sha256::verify_sha256_compressions(
                        &mut Blake3Transcript::new(),
                        &p,
                        &w.1,
                        &h.commitment,
                        proof,
                    )?)
                },
                |h, proof| {
                    let b = proof.bitz().to_bytes();
                    let decoded = IntEvalRsLigVirtProof::from_bytes(&b)?;
                    assert_eq!(decoded.to_bytes(), b);
                    Ok(bytes(
                        h.commitment.root.len(),
                        &b,
                        16 + proof.inner_nonces().len() * 8
                            + proof.inner().round_polynomials.len() * 3 * 16,
                    ))
                },
            )
        }
        "sha-chain" => {
            let p = sha256::prepare_sha256_chain_batch(7)?.with_ligerito(e.selection)?;
            let source: Vec<_> = (0..128)
                .map(|i| std::array::from_fn(|j| (i * 16 + j) as u32))
                .collect();
            e.run(
                setup,
                p.ligerito_configuration()?,
                p.security().ood,
                &bincode::serialize(&source)?,
                || Ok(sha256::generate_sha256_chain_witnesses(&p, &source)?),
                |w| Ok(sha256::commit_sha256_chain_witness(&p, w)?),
                |w, h| {
                    Ok(sha256::prove_sha256_chain(
                        &mut Blake3Transcript::new(),
                        &p,
                        &w.statement(),
                        w,
                        h,
                    )?)
                },
                |w, h, proof| {
                    Ok(sha256::verify_sha256_chain(
                        &mut Blake3Transcript::new(),
                        &p,
                        &w.statement(),
                        &h.commitment,
                        proof,
                    )?)
                },
                |h, proof| {
                    let b = proof.bitz().to_bytes();
                    let decoded = IntEvalRsLigVirtProof::from_bytes(&b)?;
                    assert_eq!(decoded.to_bytes(), b);
                    Ok(bytes(h.commitment.root.len(), &b, proof.piop_bytes()))
                },
            )
        }
        "ecdsa-split" | "ecdsa-all" => ecdsa(&e, setup),
        "hybrid-15-7" | "hybrid-15-2" => {
            use ::bitz::hybrid::*;
            let sha_log = if e.case.ends_with('7') { 7 } else { 2 };
            let p = PreparedHybrid::new_with_ligerito(
                Parameters {
                    multiplications: 1 << 15,
                    sha_compressions: 1 << sha_log,
                },
                e.selection,
            )?;
            let blocks: Vec<_> = (0..1 << sha_log)
                .map(|i| std::array::from_fn(|j| (i * 16 + j) as u32))
                .collect();
            e.run(setup,p.ligerito_configuration(),p.ood_round(),&bincode::serialize(&(&input,&blocks))?,
                || Ok(input.iter().map(|&(x,y)|MulRow::<u32>::new(x as u32,y as u32)).collect::<Vec<_>>()),
                |rows| Ok(p.commit_mod32(rows,&blocks)?), |_,h| Ok(p.prove(h)?),
                |_,h,proof| { let b=proof.to_bytes(); let decoded=p.proof_from_bytes(h.statement(),&b)?; Ok(p.verify(h.statement(),&decoded)?) },
                |_,proof| Ok(ProofSize { total_bytes: proof.to_bytes().len(), commitment_bytes: None,
                    opening_codec_bytes: None, analytical_piop_bytes: None,
                    accounting: "complete hybrid codec, including initial roots; witness synthesis is fused into commit_ms" }))
        }
        "pcs-22" => pcs(&e, setup),
        _ => bail!("unknown case"),
    }
}
fn ecdsa(e: &Experiment, setup: SetupCapture) -> Result<()> {
    use ::bitz::piop::spartan::ecdsa_sha256::*;
    use p256::ecdsa::{Signature, SigningKey, signature::Signer};
    let mode = if e.case.ends_with("split") {
        OuterMode::Split
    } else {
        OuterMode::AllRows
    };
    let p = prepare_sha256_ecdsa(3, 100, mode)?.with_ligerito(e.selection)?;
    let message: Vec<_> = (0..p.message_bytes()).map(|i| i as u8).collect();
    let key = SigningKey::from_bytes((&[7u8; 32]).into())?;
    let sig: Signature = key.sign(&message);
    let point = key.verifying_key().to_encoded_point(false);
    let (r, s) = sig.split_bytes();
    let statement = Sha256EcdsaStatement {
        log_compressions: 3,
        qx: point.x().unwrap().as_slice().try_into()?,
        qy: point.y().unwrap().as_slice().try_into()?,
        r: r.into(),
        s: s.into(),
    };
    e.run(
        setup,
        p.ligerito_configuration(),
        p.ligerito_configuration().round0(100)?,
        &bincode::serialize(&(
            &message,
            statement.qx,
            statement.qy,
            statement.r,
            statement.s,
        ))?,
        || Ok(generate_sha256_ecdsa_witness(&p, &statement, &message)?),
        |w| Ok(commit_sha256_ecdsa(&p, w)?),
        |w, h| {
            Ok(prove_sha256_ecdsa(
                &mut Blake3Transcript::new(),
                &p,
                &statement,
                w,
                h,
                4,
            )?)
        },
        |_, h, proof| {
            let decoded = Sha256EcdsaProof::from_bytes(&proof.to_bytes())?;
            Ok(verify_sha256_ecdsa(
                &mut Blake3Transcript::new(),
                &p,
                &statement,
                &h.commitment,
                &decoded,
            )?)
        },
        |h, proof| {
            Ok(ProofSize {
                total_bytes: h.commitment.root.len() + proof.to_bytes().len(),
                commitment_bytes: Some(h.commitment.root.len()),
                opening_codec_bytes: None,
                analytical_piop_bytes: None,
                accounting: "initial root + complete SHA+ECDSA codec",
            })
        },
    )
}
fn pcs(e: &Experiment, setup: SetupCapture) -> Result<()> {
    use ::bitz::{
        ext_proj::*,
        ligerito_flock::*,
        pcs::{IntegerMatrixLayout, smallest_generator},
    };
    let p = IntegerMatrixLayout {
        row_vars: 11,
        col_vars: 11,
        word_bits: 1,
    };
    let q_bits = 113;
    let alpha = smallest_generator();
    let resolved = e.selection.resolve(15, 100)?;
    let ood = resolved.round0(100)?;
    let sample = |t: &mut Blake3Transcript| {
        let q = sample_proj_prime(
            t,
            &ExtProjParams {
                prime_bits: q_bits,
                ..Default::default()
            },
        )
        .expect("bounded benchmark prime search");
        let a = field::FpCtx::from_prime_u128(q);
        let eq = |r: Vec<u128>| {
            let mut table = vec![1];
            for x in r {
                let factor = a.prepare_multiplier_u128(x);
                let mut next = Vec::with_capacity(table.len() * 2);
                for v in table {
                    let v1 = a.mul_canonical_u128(v, &factor);
                    next.push(a.sub_canonical_u128(v, v1));
                    next.push(v1);
                }
                table = next;
            }
            table
        };
        let rows = eq((0..p.row_vars).map(|_| sample_proj_point(t, q)).collect());
        let cols = eq((0..p.col_vars).map(|_| sample_proj_point(t, q)).collect());
        (q, rows, cols)
    };
    e.run(
        setup,
        &resolved,
        ood,
        b"BLAKE3-XOF:bitz/ligerito-bound-pcs/v1:seed=0x5533325043530064:t11:s11:w1",
        || {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"bitz/ligerito-bound-pcs/v1");
            hasher.update(&SEED.to_le_bytes());
            let mut r = hasher.finalize_xof();
            Ok((0..p.cols())
                .map(|_| {
                    (0..p.rows() / 64)
                        .map(|_| {
                            let mut b = [0; 8];
                            r.fill(&mut b);
                            u64::from_le_bytes(b)
                        })
                        .collect()
                })
                .collect::<Vec<Vec<u64>>>())
        },
        |w| Ok(commit_rs_ligerito_rows(&p, w.clone(), resolved.prover())),
        |_, h| {
            let mut t = Blake3Transcript::new();
            absorb_standalone_mod_q_statement(
                &mut t,
                &h.commitment,
                &p,
                alpha,
                q_bits,
                ood,
                resolved.verifier(),
            );
            resolved.bind(&mut t);
            let bound = bind_prover_ood(&mut t, h, ood);
            let (q, rows, cols) = sample(&mut t);
            let a = field::FpCtx::from_prime_u128(q);
            let mut y = 0;
            for (c, w) in h.rows().iter().enumerate() {
                let mut acc = 0;
                for (wi, &word) in w.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        acc = a.add_u128(acc, rows[wi * 64 + bit]);
                    }
                }
                y = a.add_u128(y, a.mul_u128(cols[c], acc));
            }
            absorb_standalone_mod_q_claim(&mut t, q, y);
            Ok((
                prove_mle_eval_mod_q_ligerito_with_ood(
                    &mut t,
                    h,
                    &p,
                    &rows,
                    q_bits,
                    alpha,
                    bound,
                    resolved.prover(),
                ),
                y,
            ))
        },
        |_, h, (proof, y)| {
            let proof = IntEvalRsLigModQProof::from_bytes(&proof.to_bytes())?;
            let mut t = Blake3Transcript::new();
            absorb_standalone_mod_q_statement(
                &mut t,
                &h.commitment,
                &p,
                alpha,
                q_bits,
                ood,
                resolved.verifier(),
            );
            resolved.bind(&mut t);
            let bound = bind_verifier_ood(&mut t, 15, ood, proof.ood.as_ref())
                .map_err(|e| format!("{e:?}"))?;
            let (q, rows, cols) = sample(&mut t);
            absorb_standalone_mod_q_claim(&mut t, q, *y);
            Ok(verify_mle_eval_mod_q_ligerito_runtime(
                &mut t,
                &h.commitment,
                &proof,
                &p,
                &rows,
                &cols,
                alpha,
                *y,
                q,
                q_bits,
                bound,
                resolved.verifier(),
            )
            .map_err(|e| format!("{e:?}"))?)
        },
        |h, (proof, _)| Ok(bytes(h.commitment.root.len(), &proof.to_bytes(), 16)),
    )
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    #[test]
    fn proof_sizes_omit_unavailable_components() {
        assert_eq!(
            serde_json::to_value(ProofSize {
                total_bytes: 9,
                commitment_bytes: None,
                opening_codec_bytes: None,
                analytical_piop_bytes: None,
                accounting: "opaque"
            })
            .unwrap(),
            json!({"total_bytes":9,"accounting":"opaque"})
        );
        let value = serde_json::to_value(bytes(2, &[0; 3], 4)).unwrap();
        assert_eq!(value["total_bytes"], 9);
        assert_eq!(value["commitment_bytes"], 2);
        assert_eq!(value["opening_codec_bytes"], 3);
        assert_eq!(value["analytical_piop_bytes"], 4);
    }
}

#[cfg(test)]
mod cli_tests {
    use super::Args;
    use clap::{CommandFactory, Parser, error::ErrorKind};

    #[test]
    fn positional_cases_memory_and_cargo_flag() {
        Args::command().debug_assert();
        let latency =
            Args::try_parse_from(["bounds", "sha-compression", "custom:1:4", "--bench"]).unwrap();
        assert_eq!(
            (
                latency.case.as_str(),
                latency.profile.as_str(),
                latency.memory
            ),
            ("sha-compression", "custom:1:4", false)
        );
        let memory =
            Args::try_parse_from(["bounds", "hybrid-15-7", "udrg:1:4", "--memory"]).unwrap();
        assert_eq!(
            (memory.case.as_str(), memory.profile.as_str(), memory.memory),
            ("hybrid-15-7", "udrg:1:4", true)
        );
        for argv in [
            &["bounds"][..],
            &["bounds", "u64"],
            &["bounds", "unknown", "custom:1:4"],
            &["bounds", "u64", "custom:1:4", "--unknown"],
        ] {
            assert!(Args::try_parse_from(argv).is_err(), "accepted {argv:?}");
        }
        assert_eq!(
            Args::try_parse_from(["bounds", "--help"])
                .err()
                .unwrap()
                .kind(),
            ErrorKind::DisplayHelp
        );
    }
}
