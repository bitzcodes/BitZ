use crate::sumcheck::inner::{
    packed::{PackedInput, Sha256InnerGrinding},
    prove_inner_sumcheck,
};
use field::{RingOps, Uint};

use flock_core::pcs::commit::Commitment;

use super::{
    Config, Result, error,
    inner_reduction::{InnerSumcheckClaim, ModQCoefficients},
    relation::{OuterMode, PreparedSha256Ecdsa, Sha256EcdsaStatement},
    security::Sha256EcdsaSecurity,
    witness::Sha256EcdsaWitness,
};
use {
    crate::{
        ext_proj::sample_prime_in_interval,
        ligerito::packed_vars,
        ligerito_flock::{
            FlockCommitHint, IntEvalRsLigVirtProof, commit_rs_ligerito_shared_rows,
            grinding::{GrindingContext, GrindingNonces},
            prove_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_and_modulus_with_security,
            validate_ligerito_commitment,
            verify_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_and_read_off_with_security,
        },
        pcs::ModQWeightChunks,
        piop::spartan::{
            SpartanField, absorb_spartan_message,
            bitz::SpartanBitzField as F,
            grinding::GrindingDomain,
            matrix::eq_table,
            protocol::{check_boundary, bitz_generator, grind_boundary},
            sha256::inner_sumcheck::ColumnMajorPackedBits,
            squeeze_field,
            sumcheck::{OuterSumcheckProof, ProverGrindingRoundBoundary, SumcheckProof},
        },
        transcript::traits::Transcript,
    },
    circuit::linear_map::binary::VirtualMap,
};

enum OuterGrinding {}
impl GrindingDomain for OuterGrinding {
    const DOMAIN: &'static [u8] = b"bitz/sha256-ecdsa/outer/v1";
}

/// A single source commitment supports both PIOP reductions and the final opening.
#[derive(Clone)]
pub struct Sha256EcdsaProof {
    pub(crate) modulus: u128,
    pub(crate) initial_nonce: u64,
    pub(crate) batch_nonce: u64,
    pub(crate) flock_nonces: Vec<u64>,
    pub(crate) outer: OuterSumcheckProof<F>,
    pub(crate) outer_nonces: Vec<u64>,
    pub(crate) inner: SumcheckProof<F, 3>,
    pub(crate) inner_nonces: Vec<u64>,
    pub(crate) opening: IntEvalRsLigVirtProof,
}

pub fn commit_sha256_ecdsa(
    prepared: &PreparedSha256Ecdsa,
    witness: &Sha256EcdsaWitness,
) -> Result<FlockCommitHint> {
    let pc = prepared.ligerito.prover();
    if witness.statement.log_compressions as usize != prepared.log_n {
        return Err(error("witness layout mismatch"));
    }
    Ok(commit_rs_ligerito_shared_rows(
        &prepared.f_layout,
        witness.f_rows.clone(),
        pc,
    ))
}

fn bind_statement<T: Transcript>(
    t: &mut T,
    prepared: &PreparedSha256Ecdsa,
    statement: &Sha256EcdsaStatement,
    commitment: &Commitment,
    security: &Sha256EcdsaSecurity,
) -> Result<()> {
    if statement.log_compressions as usize != prepared.log_n {
        return Err(error("statement layout mismatch"));
    }
    absorb_spartan_message(t, b"protocol", b"bitz/sha256-ecdsa/split-inner/early-ood/v2");
    absorb_spartan_message(t, b"relation", &prepared.local.digest);
    absorb_spartan_message(t, b"map", &prepared.map.digest());
    absorb_spartan_message(t, b"statement", &statement.bytes());
    absorb_spartan_message(
        t,
        b"mode",
        &[match prepared.mode {
            OuterMode::Split => 0,
            OuterMode::AllRows => 1,
        }],
    );
    absorb_spartan_message(t, b"security-target", &prepared.lambda.to_le_bytes());
    for block in &security.blocks {
        absorb_spartan_message(t, b"challenge-block", block.label.as_bytes());
        absorb_spartan_message(t, b"block-work", &block.grinding_bits.to_le_bytes());
        absorb_spartan_message(
            t,
            b"block-count",
            &(block.max_occurrences as u64).to_le_bytes(),
        );
    }
    absorb_spartan_message(
        t,
        b"commitment",
        &bincode::serialize(commitment).map_err(error)?,
    );
    prepared.ligerito.bind(t);
    Ok(())
}

struct InitialSpartanChallenges {
    modulus: u128,
    field_config: Config,
    outer_eq_challenges: Vec<F>,
    grinding_nonce: u64,
}

/// Prove or verify initial grinding, then derive the field and outer equality challenges.
fn derive_initial_challenges<T: Transcript>(
    t: &mut T,
    prepared: &PreparedSha256Ecdsa,
    security: &Sha256EcdsaSecurity,
    nonce: Option<u64>,
) -> Result<InitialSpartanChallenges> {
    let grinding_nonce = boundary(t, INITIAL_GRINDING_DOMAIN, security.initial, nonce)?;
    let modulus = sample_prime_in_interval(t, 1u128 << 112, (1u128 << 113) - 1).map_err(error)?;
    absorb_spartan_message(t, b"q", &modulus.to_le_bytes());
    let field_config =
        F::make_cfg(&Uint::from(modulus)).map_err(|_| error("invalid sampled modulus"))?;
    let outer_eq_challenges = (0..prepared.outer_sumcheck_num_vars())
        .map(|_| squeeze_field(t, &field_config))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(InitialSpartanChallenges {
        modulus,
        field_config,
        outer_eq_challenges,
        grinding_nonce,
    })
}

fn sample_inner_batch_challenges<T: Transcript>(
    t: &mut T,
    prepared: &PreparedSha256Ecdsa,
    cfg: &Config,
) -> Result<(F, Vec<F>, F)> {
    // Sample only after the outer terminal triple has been absorbed.
    absorb_spartan_message(t, b"shared-inner", b"rho-sigma-gamma/v1");
    let matrix_batch_challenge = squeeze_field(t, cfg)?;
    let linear_row_point = (0..prepared.linear_vars())
        .map(|_| squeeze_field(t, cfg))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let linear_batch_weight = squeeze_field(t, cfg)?;
    Ok((
        matrix_batch_challenge,
        linear_row_point,
        linear_batch_weight,
    ))
}

fn bind_opening<T: Transcript>(
    t: &mut T,
    point: &[F],
    scale: &F,
    value: &F,
    field_config: &Config,
) {
    absorb_spartan_message(t, b"scaled-assignment-claim", b"v1");
    for x in point.iter().chain([scale, value]) {
        absorb_spartan_message(t, b"field", &x.canonical_element_encoding(field_config));
    }
}

/// Proves one standard SHA-256 message followed by P-256 ECDSA verification.
/// `prefix_vars` (0..=4) changes only the packed inner prover's implementation.
pub fn prove_sha256_ecdsa<T: Transcript + Send>(
    t: &mut T,
    prepared: &PreparedSha256Ecdsa,
    statement: &Sha256EcdsaStatement,
    witness: &Sha256EcdsaWitness,
    hint: &FlockCommitHint,
    prefix_vars: usize,
) -> Result<Sha256EcdsaProof> {
    let pc = prepared.ligerito.prover();
    if &witness.statement != statement || !hint.matches_rows(&witness.f_rows) {
        return Err(error("statement or commitment witness mismatch"));
    }
    if prefix_vars > 4 {
        return Err(error("inner prefix must be in 0..=4"));
    }
    validate_ligerito_commitment(&hint.commitment, pc).map_err(|e| error(format!("{e:?}")))?;
    let security = prepared.security()?;
    bind_statement(t, prepared, statement, &hint.commitment, &security)?;
    let ood = crate::ligerito_flock::bind_prover_ood(t, hint, security.ood);
    let InitialSpartanChallenges {
        modulus,
        field_config: cfg,
        outer_eq_challenges,
        grinding_nonce: initial_nonce,
    } = derive_initial_challenges(t, prepared, &security, None)?;
    crate::utils::delayed_reduction::prepare_field(&cfg).map_err(error)?;
    let mut mod_q_coefficients = ModQCoefficients::from_relation(prepared, &cfg);
    let (outer, outer_nonces) = {
        let _scope = tracing::info_span!("ecdsa:outer_prove").entered();
        let rows = witness.outer_integer_rows(prepared);
        let mut round_boundary =
            ProverGrindingRoundBoundary::<OuterGrinding>::with_round_offset(security.outer, 0);
        let outer: crate::sumcheck::proof::OuterSumcheckOutput<F> =
            crate::sumcheck::outer::prove_outer_sumcheck(
                &cfg,
                t,
                crate::sumcheck::outer::OuterClaim::RowwiseZero,
                &outer_eq_challenges,
                &rows,
                None,
                &mut round_boundary,
            )
            .map_err(error)?
            .into();
        (outer, round_boundary.into_nonces())
    };
    let batch_nonce = boundary(t, BATCH_GRINDING_DOMAIN, security.batch, None)?;
    let (matrix_batch_challenge, linear_row_point, linear_batch_weight) =
        sample_inner_batch_challenges(t, prepared, &cfg)?;
    let inner_claim = InnerSumcheckClaim::from_outer_claims(
        prepared,
        statement,
        &outer.proof,
        outer.eval_points,
        matrix_batch_challenge,
        linear_row_point,
        linear_batch_weight,
        &cfg,
    )?;
    let batched_matrix_mle =
        mod_q_coefficients.build_batched_matrix_mle(prepared, &inner_claim, &cfg)?;
    drop(mod_q_coefficients);
    let (inner, inner_nonces) = {
        let _scope = tracing::info_span!("ecdsa:shared_inner_prove").entered();
        {
            let coefficients = &batched_matrix_mle.as_mle(&cfg)?;
            let mut boundary =
                ProverGrindingRoundBoundary::<Sha256InnerGrinding>::with_round_offset(
                    security.inner,
                    0,
                );
            prove_inner_sumcheck(
                &cfg,
                t,
                inner_claim.claimed_sum().clone(),
                PackedInput::new(
                    coefficients,
                    &ColumnMajorPackedBits::new(&witness.h_rows, prepared.h_layout.row_vars),
                    prepared.h_layout.row_vars + prepared.h_layout.col_vars,
                    coefficients.live_len(),
                    prefix_vars,
                ),
                (),
                &mut boundary,
            )
            .map(|out| (out, boundary.into_nonces()))
        }
        .map_err(error)?
    };
    if batched_matrix_mle.evaluate(&inner.point, &cfg)? != inner.terminal_evaluations[0] {
        return Err(error("batched matrix MLE evaluation mismatch"));
    }
    drop(batched_matrix_mle);
    if cfg.mul(
        &(inner.terminal_evaluations[0].clone()),
        &(&inner.terminal_evaluations[1]),
    ) != inner.final_claim
    {
        return Err(error("witness does not satisfy the shared inner claim"));
    }
    bind_opening(
        t,
        &inner.point,
        &inner.terminal_evaluations[0],
        &inner.final_claim,
        &cfg,
    );
    let rows: Vec<_> = eq_table(&inner.point[..prepared.h_layout.row_vars], &cfg)
        .map_err(error)?
        .into_iter()
        .map(|mut x| {
            x = cfg.mul(&(x), &(&inner.terminal_evaluations[0]));
            u128::from(cfg.to_integer(&x))
        })
        .collect();
    let mut flock_nonces = Vec::new();
    let chunks = ModQWeightChunks::from_single_chunk(&prepared.h_layout, 113, rows)
        .map_err(|_| error("invalid row weights"))?;
    let mut grinding = GrindingContext {
        plan: &security.flock,
        nonces: GrindingNonces::Prove(&mut flock_nonces),
    };
    let opening = {
        let _scope = tracing::info_span!("ecdsa:bitz_prove").entered();
        prove_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_and_modulus_with_security(
            t,
            hint,
            &witness.h_rows,
            &prepared.h_layout,
            &prepared.f_layout,
            &prepared.map,
            &chunks,
            modulus,
            113,
            bitz_generator(),
            security.forest,
            ood,
            pc,
            Some(&mut grinding),
        )
    };
    Ok(Sha256EcdsaProof {
        modulus,
        initial_nonce,
        batch_nonce,
        flock_nonces,
        outer: outer.proof,
        outer_nonces,
        inner: inner.proof,
        inner_nonces,
        opening,
    })
}

pub fn verify_sha256_ecdsa<T: Transcript + Send>(
    transcript: &mut T,
    prepared: &PreparedSha256Ecdsa,
    statement: &Sha256EcdsaStatement,
    commitment: &Commitment,
    proof: &Sha256EcdsaProof,
) -> Result<()> {
    let vc = prepared.ligerito.verifier();
    validate_ligerito_commitment(commitment, vc).map_err(|e| error(format!("{e:?}")))?;
    let security = prepared.security()?;
    bind_statement(transcript, prepared, statement, commitment, &security)?;
    let ood = crate::ligerito_flock::bind_verifier_ood(
        transcript,
        packed_vars(&prepared.f_layout),
        security.ood,
        proof.opening.ood.as_ref(),
    )
    .map_err(|e| error(format!("{e:?}")))?;
    let InitialSpartanChallenges {
        modulus,
        field_config: cfg,
        outer_eq_challenges,
        grinding_nonce: _,
    } = derive_initial_challenges(transcript, prepared, &security, Some(proof.initial_nonce))?;
    if proof.modulus != modulus {
        return Err(error("proof modulus does not match transcript replay"));
    }
    let outer_row_point = proof
        .outer
        .verify_grinded::<OuterGrinding>(
            transcript,
            F::zero_with_cfg(&cfg),
            &outer_eq_challenges,
            &cfg,
            &proof.outer_nonces,
            security.outer,
        )
        .map_err(error)?
        .eval_points;
    boundary(
        transcript,
        BATCH_GRINDING_DOMAIN,
        security.batch,
        Some(proof.batch_nonce),
    )?;
    let (matrix_batch_challenge, linear_row_point, linear_batch_weight) =
        sample_inner_batch_challenges(transcript, prepared, &cfg)?;
    let mut mod_q_coefficients = ModQCoefficients::from_relation(prepared, &cfg);
    let inner_claim = InnerSumcheckClaim::from_outer_claims(
        prepared,
        statement,
        &proof.outer,
        outer_row_point,
        matrix_batch_challenge,
        linear_row_point,
        linear_batch_weight,
        &cfg,
    )?;
    let (inner_eval_point, inner_final_claim) = proof
        .inner
        .verify_grinded::<Sha256InnerGrinding>(
            transcript,
            inner_claim.claimed_sum().clone(),
            prepared.h_layout.row_vars + prepared.h_layout.col_vars,
            &cfg,
            &proof.inner_nonces,
            security.inner,
        )
        .map_err(error)?;
    // inner_final_claim ≡ scale · h(inner_eval_point) (mod q).
    let scale = mod_q_coefficients.evaluate_batched_matrix_mle(
        prepared,
        &inner_claim,
        &inner_eval_point,
        &cfg,
    )?;
    drop(mod_q_coefficients);
    bind_opening(
        transcript,
        &inner_eval_point,
        &scale,
        &inner_final_claim,
        &cfg,
    );
    // rows[b] = (scale · eq(b, inner_eval_point[..row_vars])) mod q ∈ [0, q).
    let rows: Vec<_> = eq_table(&inner_eval_point[..prepared.h_layout.row_vars], &cfg)
        .map_err(error)?
        .into_iter()
        .map(|mut x| {
            x = cfg.mul(&(x), &(&scale));
            u128::from(cfg.to_integer(&x))
        })
        .collect();
    // cols[c] = eq(c, inner_eval_point[row_vars..]) mod q ∈ [0, q).
    // Σ_{b,c} rows[b] · h[b,c] · cols[c] ≡ inner_final_claim (mod q).
    let cols: Vec<_> = eq_table(&inner_eval_point[prepared.h_layout.row_vars..], &cfg)
        .map_err(error)?
        .iter()
        .map(|x| u128::from(cfg.to_integer(x)))
        .collect();
    // R = 2^row_vars, C = 2^col_vars; h: R × C.
    // w = 127 - row_vars - 1 ≥ 113 ⇒ L = ⌈113 / w⌉ = 1.
    // chunks: L × R; folds = chunks · h: L × C.
    // folds[ℓ][c] = Σ_b chunks[ℓ][b] · h[b,c].
    // chunks[0][b] = rows[b].
    let chunks = ModQWeightChunks::from_single_chunk(&prepared.h_layout, 113, rows)
        .map_err(|_| error("invalid row weights"))?;
    let mut grinding = GrindingContext {
        plan: &security.flock,
        nonces: GrindingNonces::Verify {
            values: &proof.flock_nonces,
            cursor: 0,
        },
    };
    let arithmetic = field::FpCtx::from_prime_u128(modulus);
    verify_mle_eval_mod_q_ligerito_virtual_with_weight_chunks_and_read_off_with_security(
        transcript,
        commitment,
        &proof.opening,
        &prepared.h_layout,
        &prepared.f_layout,
        &prepared.map,
        &chunks,
        modulus,
        113,
        bitz_generator(),
        security.forest,
        ood,
        vc,
        cols.len(),
        |values, width, count| {
            // This profile has one 113-bit chunk; the BitZ preflight enforces the
            // fold magnitudes before this canonical mod-q read-off is accepted.
            if count != 1 || values.len() < cols.len() || width < 113 {
                return false;
            }
            let sum = cols.iter().zip(values).fold(0, |sum, (&c, &v)| {
                arithmetic.add_u128(sum, arithmetic.mul_u128(c, arithmetic.reduce_u128(v)))
            });
            sum == u128::from(cfg.to_integer(&inner_final_claim))
        },
        Some(&mut grinding),
    )
    .map_err(|e| error(format!("{e:?}")))
}

const INITIAL_GRINDING_DOMAIN: &[u8] = b"bitz/sha256-ecdsa/initial/v1";
const BATCH_GRINDING_DOMAIN: &[u8] = b"bitz/sha256-ecdsa/batch/v1";

/// The shared protocol boundary (skipped at difficulty 0 with the canonical
/// zero nonce): ground by the prover, checked by the verifier.
fn boundary<T: Transcript>(t: &mut T, domain: &[u8], bits: u32, nonce: Option<u64>) -> Result<u64> {
    let _scope = tracing::info_span!(
        "boundary_grinding",
        component = if nonce.is_some() {
            "ecdsa:boundary_grinding_verify"
        } else {
            "ecdsa:boundary_grinding_prove"
        }
    )
    .entered();
    match nonce {
        Some(n) => {
            check_boundary(t, domain, bits, n).map_err(error)?;
            Ok(n)
        }
        None => grind_boundary(t, domain, bits).map_err(error),
    }
}
