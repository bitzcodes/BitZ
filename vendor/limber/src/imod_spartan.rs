//! Single-field (p = q) Integer Mod-R1CS SNARK driver — the prototype
//! that `imod_spartan_modp` generalizes.
//!
//! Mirrors `spartan::SpartanSNARK` but proves the IntMod-R1CS relation
//!   `Az ∘ Bz = Cz + m ∘ q`
//! over the curve scalar field (i.e. p = q; the paper's separate
//! random prime p is what `imod_spartan_modp` adds). Single witness segment (no
//! shared/precommitted/rest split), no limb decomposition, no range checks,
//! no BDDT first-round optimization in the inner sumcheck — this is the
//! simplest version that exercises every structural change.

use crate::{
  Blind, CommitmentKey, VerifierKey,
  digest::{DigestComputer, Digestible},
  errors::SpartanError,
  imod_r1cs::{IntModR1CSInstance, IntModR1CSShape, IntModR1CSWitness},
  math::Math,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    pcs::PCSEngineTrait,
    snark::SpartanDigest,
    transcript::{ByteTranscript, TranscriptEngineTrait},
  },
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Prover key for the IntMod-R1CS SNARK.
#[derive(Clone)]
pub struct IntModSpartanProverKey<E: Engine> {
  pub(crate) ck: CommitmentKey<E>,
  pub(crate) ck_s: CommitmentKey<E>,
  pub(crate) shape: IntModR1CSShape<E>,
  pub(crate) vk_digest: SpartanDigest,
}

impl<E: Engine> IntModSpartanProverKey<E> {
  /// Commitment key used for the witness and quotient polynomials. External
  /// callers need this to construct `IntModR1CSWitness` instances.
  pub fn ck(&self) -> &CommitmentKey<E> {
    &self.ck
  }
}

/// Verifier key for the IntMod-R1CS SNARK.
#[derive(Clone)]
pub struct IntModSpartanVerifierKey<E: Engine> {
  pub(crate) vk_ee: VerifierKey<E>,
  pub(crate) ck_s: CommitmentKey<E>,
  pub(crate) shape: IntModR1CSShape<E>,
  pub(crate) digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> Digestible for IntModSpartanVerifierKey<E> {
  fn write_bytes<W: Sized + std::io::Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
    use bincode::Options;
    let config = bincode::DefaultOptions::new()
      .with_little_endian()
      .with_fixint_encoding();
    config
      .serialize_into(&mut *w, &self.vk_ee)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.ck_s)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    self.shape.write_bytes(w)?;
    Ok(())
  }
}

impl<E: Engine> IntModSpartanVerifierKey<E> {
  /// Returns the (cached) digest binding the verifier key's public parameters.
  pub fn digest(&self) -> Result<SpartanDigest, SpartanError> {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .map_err(|_| SpartanError::DigestError {
        reason: "Unable to compute digest for IntModSpartanVerifierKey".to_string(),
      })
  }
}

/// IntMod-R1CS SNARK proof.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IntModSpartanSNARK<E: Engine> {
  // outer sumcheck
  sc_outer: SumcheckProof<E>,
  // (v_a, v_b, v_c, v_m, v_q) at end of outer SC
  v_a: E::Scalar,
  v_b: E::Scalar,
  v_c: E::Scalar,
  v_m: E::Scalar,
  v_q: E::Scalar,
  // inner sumcheck (for w)
  sc_inner: SumcheckProof<E>,
  // eval of W at r_y[1..]
  eval_w: E::Scalar,
  blind_eval_w: Blind<E>,
  eval_arg_w: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
  // PCS opening for q at r_x
  blind_eval_q: Blind<E>,
  eval_arg_q: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
}

impl<E: Engine> IntModSpartanSNARK<E> {
  /// Setup: builds prover and verifier keys from the shape.
  pub fn setup(
    shape: IntModR1CSShape<E>,
  ) -> Result<(IntModSpartanProverKey<E>, IntModSpartanVerifierKey<E>), SpartanError> {
    let (ck, vk_ee) = shape.commitment_key();
    E::PCS::precompute_ck(&ck);
    let (ck_s, _) = E::PCS::setup(b"ck_s_imod", 1, 1);
    E::PCS::precompute_ck(&ck_s);

    let vk = IntModSpartanVerifierKey {
      vk_ee,
      ck_s: ck_s.clone(),
      shape: shape.clone(),
      digest: OnceCell::new(),
    };
    let vk_digest = vk.digest()?;
    let pk = IntModSpartanProverKey {
      ck,
      ck_s,
      shape,
      vk_digest,
    };
    Ok((pk, vk))
  }

  /// Prove satisfaction of the IntMod-R1CS instance.
  pub fn prove(
    pk: &IntModSpartanProverKey<E>,
    U: &IntModR1CSInstance<E>,
    W: &IntModR1CSWitness<E>,
  ) -> Result<Self, SpartanError> {
    let (_prove_span, prove_t) = start_span!("imod_spartan_prove");
    let mut transcript = E::TE::new(b"IntModSpartanSNARK");

    // Bind the public state to the transcript.
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"comm_w", &U.comm_w);
    transcript.absorb(b"comm_q", &U.comm_q);
    transcript.absorb(b"x", &U.x.as_slice());

    let shape = &pk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1; // z lives in 2*num_vars

    // z = (W, 1, X), padded with zeros to 2*num_vars for the MLE.
    let mut z = Vec::with_capacity(2 * num_vars);
    z.extend_from_slice(&W.w);
    z.push(E::Scalar::ONE);
    z.extend_from_slice(&U.x);
    z.resize(2 * num_vars, E::Scalar::ZERO);

    // SpMV: (Az, Bz, Cz) sliced to the matrix's column dimension.
    let z_for_spmv = &z[..num_vars + 1 + shape.num_io];
    let (az, bz, cz) = shape.multiply_vec(z_for_spmv)?;

    // M polynomial: extend mods from length num_cons (already a power of two
    // by construction). m is public and preprocessed.
    let m: Vec<E::Scalar> = shape.mods.clone();
    // Q polynomial: clone witness q so the SC can bind in place while we
    // retain the original for the PCS opening below.
    let q_for_open = W.q.clone();

    // Outer sumcheck: prove sum_i eq(i, tau) * (Az·Bz − Cz − M·Q) = 0.
    let tau: Vec<E::Scalar> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let mut poly_az = MultilinearPolynomial::new(az);
    let mut poly_bz = MultilinearPolynomial::new(bz);
    let mut poly_cz = MultilinearPolynomial::new(cz);
    let mut poly_m = MultilinearPolynomial::new(m);
    let mut poly_q = MultilinearPolynomial::new(W.q.clone());

    let (_so_span, so_t) = start_span!("imod_outer_sumcheck");
    let (sc_outer, r_x, outer_claims) = SumcheckProof::prove_cubic_with_five_inputs(
      &E::Scalar::ZERO,
      tau,
      &mut poly_az,
      &mut poly_bz,
      &mut poly_cz,
      &mut poly_m,
      &mut poly_q,
      &mut transcript,
    )?;
    info!(elapsed_ms = %so_t.elapsed().as_millis(), "imod_outer_sumcheck");

    let v_a = outer_claims[0];
    let v_b = outer_claims[1];
    let v_c = outer_claims[2];
    let v_m = outer_claims[3];
    let v_q = outer_claims[4];

    transcript.absorb(b"outer_claims", &[v_a, v_b, v_c, v_m, v_q].as_slice());

    // Inner sumcheck: prove sum_y (A(r_x,y) + r·B(r_x,y) + r²·C(r_x,y)) · z(y)
    //                    = v_a + r·v_b + r²·v_c
    let r = transcript.squeeze(b"r")?;
    let claim_inner = v_a + r * v_b + r * r * v_c;

    // Bind row variable r_x in A, B, C and form ABC[y].
    let eq_rx = EqPolynomial::evals_from_points(&r_x);
    let abc = bind_abc::<E>(shape, &eq_rx, &r);

    debug_assert_eq!(abc.len(), 2 * num_vars);
    debug_assert_eq!(z.len(), 2 * num_vars);

    let mut poly_abc = MultilinearPolynomial::new(abc);
    let mut poly_z = MultilinearPolynomial::new(z);

    let (_si_span, si_t) = start_span!("imod_inner_sumcheck");
    let (sc_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner,
      num_rounds_y,
      &mut poly_abc,
      &mut poly_z,
      &mut transcript,
    )?;
    info!(elapsed_ms = %si_t.elapsed().as_millis(), "imod_inner_sumcheck");

    // eval_W is recovered from eval_Z via Z = (W, 1, X, …) and
    //   Z(r_y) = (1 - r_y[0]) · W(r_y[1..]) + r_y[0] · pub(r_y[1..]).
    let eval_z = poly_z[0];
    let eval_x = eval_public_at::<E>(num_rounds_y - 1, &U.x, &r_y[1..]);
    let one_minus_r0 = E::Scalar::ONE - r_y[0];
    let inv: Option<E::Scalar> = one_minus_r0.invert().into();
    let eval_w = (eval_z - r_y[0] * eval_x) * inv.ok_or(SpartanError::DivisionByZero)?;

    // PCS open W at r_y[1..]
    let blind_eval_w = E::PCS::blind(&pk.ck_s, 1);
    let comm_eval_w = E::PCS::commit(&pk.ck_s, &[eval_w], &blind_eval_w, false)?;
    let eval_arg_w = E::PCS::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U.comm_w,
      &W.w,
      &W.r_w,
      &r_y[1..],
      &comm_eval_w,
      &blind_eval_w,
    )?;

    // PCS open Q at r_x
    let blind_eval_q = E::PCS::blind(&pk.ck_s, 1);
    let comm_eval_q = E::PCS::commit(&pk.ck_s, &[v_q], &blind_eval_q, false)?;
    let eval_arg_q = E::PCS::prove(
      &pk.ck,
      &pk.ck_s,
      &mut transcript,
      &U.comm_q,
      &q_for_open,
      &W.r_q,
      &r_x,
      &comm_eval_q,
      &blind_eval_q,
    )?;

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "imod_spartan_prove");
    Ok(Self {
      sc_outer,
      v_a,
      v_b,
      v_c,
      v_m,
      v_q,
      sc_inner,
      eval_w,
      blind_eval_w,
      eval_arg_w,
      blind_eval_q,
      eval_arg_q,
    })
  }

  /// Verify the SNARK against an instance.
  pub fn verify(
    &self,
    vk: &IntModSpartanVerifierKey<E>,
    U: &IntModR1CSInstance<E>,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("imod_spartan_verify");
    let mut transcript = E::TE::new(b"IntModSpartanSNARK");

    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"comm_w", &U.comm_w);
    transcript.absorb(b"comm_q", &U.comm_q);
    transcript.absorb(b"x", &U.x.as_slice());

    let shape = &vk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    // Outer SC verification.
    let tau: Vec<E::Scalar> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    // Verify v_m matches the public mods MLE at r_x.
    let v_m_expected = dense_evaluate::<E>(&shape.mods, &r_x);
    if v_m_expected != self.v_m {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // Reconstruct outer-SC final claim and check it.
    let eq_tau_rx = EqPolynomial::new(tau).evaluate(&r_x);
    let outer_final_expected = eq_tau_rx * (self.v_a * self.v_b - self.v_c - self.v_m * self.v_q);
    if claim_outer_final != outer_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"outer_claims",
      &[self.v_a, self.v_b, self.v_c, self.v_m, self.v_q].as_slice(),
    );

    // Inner SC verification.
    let r = transcript.squeeze(b"r")?;
    let claim_inner = self.v_a + r * self.v_b + r * r * self.v_c;

    let (claim_inner_final, r_y) =
      self
        .sc_inner
        .verify(claim_inner, num_rounds_y, 2, &mut transcript)?;

    // Reconstruct eval_Z from eval_W and public IO.
    let eval_x = eval_public_at::<E>(num_rounds_y - 1, &U.x, &r_y[1..]);
    let eval_z = (E::Scalar::ONE - r_y[0]) * self.eval_w + r_y[0] * eval_x;

    // Evaluate A, B, C MLEs at (r_x, r_y) via full eq tables.
    let t_x = EqPolynomial::evals_from_points(&r_x);
    let t_y = EqPolynomial::evals_from_points(&r_y);
    let (eval_a, eval_b, eval_c) = evaluate_matrices::<E>(shape, &t_x, &t_y);

    let inner_final_expected = (eval_a + r * eval_b + r * r * eval_c) * eval_z;
    if claim_inner_final != inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // PCS verification for W at r_y[1..].
    let comm_eval_w = E::PCS::commit(&vk.ck_s, &[self.eval_w], &self.blind_eval_w, false)?;
    E::PCS::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U.comm_w,
      &r_y[1..],
      &comm_eval_w,
      &self.eval_arg_w,
    )?;

    // PCS verification for Q at r_x.
    let comm_eval_q = E::PCS::commit(&vk.ck_s, &[self.v_q], &self.blind_eval_q, false)?;
    E::PCS::verify(
      &vk.vk_ee,
      &vk.ck_s,
      &mut transcript,
      &U.comm_q,
      &r_x,
      &comm_eval_q,
      &self.eval_arg_q,
    )?;

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "imod_spartan_verify");
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// helpers

/// Compute ABC[y] = sum_i eq_rx[i] · (A[i,y] + r·B[i,y] + r²·C[i,y]),
/// then right-pad to length 2·num_vars so it fits the inner-SC layout.
fn bind_abc<E: Engine>(
  shape: &IntModR1CSShape<E>,
  eq_rx: &[E::Scalar],
  r: &E::Scalar,
) -> Vec<E::Scalar> {
  let num_cols = shape.num_vars + 1 + shape.num_io;
  let mut abc = vec![E::Scalar::ZERO; num_cols];
  let r_sq = *r * r;

  for (i, j, val) in shape.A.iter() {
    abc[j] += eq_rx[i] * val;
  }
  for (i, j, val) in shape.B.iter() {
    abc[j] += *r * eq_rx[i] * val;
  }
  for (i, j, val) in shape.C.iter() {
    abc[j] += r_sq * eq_rx[i] * val;
  }
  abc.resize(2 * shape.num_vars, E::Scalar::ZERO);
  abc
}

/// Evaluate the multilinear extension of the public side `(1, x, 0, …, 0)`
/// (length num_vars, treated as a polynomial in `num_rounds_y - 1` vars) at
/// the point `r`.
fn eval_public_at<E: Engine>(num_vars_pub: usize, x: &[E::Scalar], r: &[E::Scalar]) -> E::Scalar {
  debug_assert_eq!(r.len(), num_vars_pub);
  // Build the public vector: [1, x_0, x_1, …, 0, 0, …] padded to 2^num_vars_pub.
  let mut pub_vec = Vec::with_capacity(1 << num_vars_pub);
  pub_vec.push(E::Scalar::ONE);
  pub_vec.extend_from_slice(x);
  pub_vec.resize(1 << num_vars_pub, E::Scalar::ZERO);
  dense_evaluate::<E>(&pub_vec, r)
}

/// Evaluate a dense multilinear (given as a power-of-two evaluation table)
/// at point `r`.
fn dense_evaluate<E: Engine>(z: &[E::Scalar], r: &[E::Scalar]) -> E::Scalar {
  let chis = EqPolynomial::evals_from_points(r);
  debug_assert_eq!(chis.len(), z.len());
  chis
    .par_iter()
    .zip(z.par_iter())
    .map(|(c, v)| *c * *v)
    .sum()
}

/// Evaluate the MLEs of A, B, C as bivariate polynomials at (r_x, r_y) using
/// the precomputed eq-tables `t_x` and `t_y`.
fn evaluate_matrices<E: Engine>(
  shape: &IntModR1CSShape<E>,
  t_x: &[E::Scalar],
  t_y: &[E::Scalar],
) -> (E::Scalar, E::Scalar, E::Scalar) {
  let eval_one = |m: &crate::r1cs::SparseMatrix<E::Scalar>| -> E::Scalar {
    m.iter()
      .map(|(i, j, v)| t_x[i] * t_y[j] * v)
      .fold(E::Scalar::ZERO, |a, b| a + b)
  };
  let (eval_a, (eval_b, eval_c)) = rayon::join(
    || eval_one(&shape.A),
    || rayon::join(|| eval_one(&shape.B), || eval_one(&shape.C)),
  );
  (eval_a, eval_b, eval_c)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::r1cs::SparseMatrix;

  /// Toy circuit: prove `a · b ≡ c (mod N)` for a, b, c, q ∈ Z with no public input.
  ///
  /// One real constraint:  a · b = c + N · q
  /// One trivial padding row to make num_cons a power of two.
  fn build_shape_and_witness<E: Engine>(
    a: u64,
    b: u64,
    c: u64,
    n: u64,
    q: u64,
  ) -> (IntModR1CSShape<E>, Vec<E::Scalar>, Vec<E::Scalar>) {
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 0usize;
    let num_cols = num_vars + 1 + num_io;

    // w = [a, b, c, 0]; column layout: 0..num_vars = w, num_vars = 1, …
    // Single constraint: A[0, 0] = 1 selects a; B[0, 1] = 1 selects b;
    //                    C[0, 2] = 1 selects c; mods[0] = N selects q[0].
    let one = E::Scalar::ONE;
    let mat_a = SparseMatrix::<E::Scalar>::new(&[(0, 0, one)], num_cons, num_cols);
    let mat_b = SparseMatrix::<E::Scalar>::new(&[(0, 1, one)], num_cons, num_cols);
    let mat_c = SparseMatrix::<E::Scalar>::new(&[(0, 2, one)], num_cons, num_cols);

    let mods = vec![E::Scalar::from(n), E::Scalar::ZERO];

    let shape =
      IntModR1CSShape::<E>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods).unwrap();

    let w = vec![
      E::Scalar::from(a),
      E::Scalar::from(b),
      E::Scalar::from(c),
      E::Scalar::ZERO,
    ];
    let q_vec = vec![E::Scalar::from(q), E::Scalar::ZERO];
    (shape, w, q_vec)
  }

  fn run_toy_prove_verify_with<E: Engine>() {
    // 3 · 5 = 15 = 1 + 14 · 1
    let (shape, w, q) = build_shape_and_witness::<E>(3, 5, 1, 14, 1);

    let (pk, vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();

    // Build instance/witness
    let (W, U) = IntModR1CSWitness::<E>::new(&shape, &pk.ck, w, q, vec![]).unwrap();

    // sanity: shape is satisfied
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    // prove + verify
    let proof = IntModSpartanSNARK::<E>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  #[test]
  fn test_imod_spartan_toy() {
    use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
    run_toy_prove_verify_with::<PallasHyraxEngine>();
    run_toy_prove_verify_with::<T256HyraxEngine>();
  }

  /// A bad witness must be rejected by `is_sat`. Picks q = 0 for a relation
  /// whose true quotient is 1.
  fn run_bad_witness_with<E: Engine>() {
    let (shape, w, _) = build_shape_and_witness::<E>(3, 5, 1, 14, 1);
    // Inject the wrong quotient: q = 0 (truth is 1).
    let bad_q = vec![E::Scalar::ZERO, E::Scalar::ZERO];

    let (pk, _vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitness::<E>::new(&shape, &pk.ck, w, bad_q, vec![]).unwrap();
    assert!(shape.is_sat(&pk.ck, &U, &W).is_err());
  }

  /// Two non-trivial constraints, both modular: a₁·b₁ ≡ c₁ (mod N₁) and
  /// a₂·b₂ ≡ c₂ (mod N₂). Exercises the outer SC with more than one active row.
  fn run_two_row_with<E: Engine>() {
    let one = E::Scalar::ONE;
    let num_cons = 2usize;
    let num_vars = 8usize;
    let num_io = 0usize;
    let num_cols = num_vars + 1 + num_io;

    // Layout w = [a1, b1, c1, a2, b2, c2, 0, 0]
    let mat_a = SparseMatrix::<E::Scalar>::new(&[(0, 0, one), (1, 3, one)], num_cons, num_cols);
    let mat_b = SparseMatrix::<E::Scalar>::new(&[(0, 1, one), (1, 4, one)], num_cons, num_cols);
    let mat_c = SparseMatrix::<E::Scalar>::new(&[(0, 2, one), (1, 5, one)], num_cons, num_cols);

    // Row 0: 3·5 ≡ 1 (mod 14), q₀ = 1
    // Row 1: 7·9 ≡ 3 (mod 20), since 63 = 3 + 20·3, q₁ = 3
    let n0: u64 = 14;
    let n1: u64 = 20;
    let mods = vec![E::Scalar::from(n0), E::Scalar::from(n1)];

    let shape =
      IntModR1CSShape::<E>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods).unwrap();

    let w: Vec<E::Scalar> = [3u64, 5, 1, 7, 9, 3, 0, 0]
      .iter()
      .map(|x| E::Scalar::from(*x))
      .collect();
    let q: Vec<E::Scalar> = [1u64, 3].iter().map(|x| E::Scalar::from(*x)).collect();

    let (pk, vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitness::<E>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();
    let proof = IntModSpartanSNARK::<E>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// Tamper with the proof's claimed v_q and confirm verify rejects.
  fn run_verify_rejects_tampering_with<E: Engine>() {
    let (shape, w, q) = build_shape_and_witness::<E>(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanSNARK::<E>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitness::<E>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let mut proof = IntModSpartanSNARK::<E>::prove(&pk, &U, &W).unwrap();
    // Flip v_q to something the prover did not commit to.
    proof.v_q += E::Scalar::ONE;
    assert!(proof.verify(&vk, &U).is_err());
  }

  #[test]
  fn test_imod_spartan_bad_witness() {
    use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
    run_bad_witness_with::<PallasHyraxEngine>();
    run_bad_witness_with::<T256HyraxEngine>();
  }

  #[test]
  fn test_imod_spartan_two_row() {
    use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
    run_two_row_with::<PallasHyraxEngine>();
    run_two_row_with::<T256HyraxEngine>();
  }

  #[test]
  fn test_imod_spartan_verify_rejects_tampering() {
    use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
    run_verify_rejects_tampering_with::<PallasHyraxEngine>();
    run_verify_rejects_tampering_with::<T256HyraxEngine>();
  }

  /// Two shapes with the same dimensions and mods but a different `A` entry
  /// must produce distinct verifier-key digests, and a proof produced for one
  /// must be rejected under the other's vk.
  fn run_digest_binds_matrices_with<E: Engine>() {
    let one = E::Scalar::ONE;
    let two = E::Scalar::from(2);
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 0usize;
    let num_cols = num_vars + 1 + num_io;
    let mat_a1 = SparseMatrix::<E::Scalar>::new(&[(0, 0, one)], num_cons, num_cols);
    let mat_a2 = SparseMatrix::<E::Scalar>::new(&[(0, 0, two)], num_cons, num_cols);
    let mat_b = SparseMatrix::<E::Scalar>::new(&[(0, 1, one)], num_cons, num_cols);
    let mat_c = SparseMatrix::<E::Scalar>::new(&[(0, 2, one)], num_cons, num_cols);
    let mods = vec![E::Scalar::from(14u64), E::Scalar::ZERO];

    let shape1 = IntModR1CSShape::<E>::new(
      num_cons,
      num_vars,
      num_io,
      mat_a1,
      mat_b.clone(),
      mat_c.clone(),
      mods.clone(),
    )
    .unwrap();
    let shape2 =
      IntModR1CSShape::<E>::new(num_cons, num_vars, num_io, mat_a2, mat_b, mat_c, mods).unwrap();
    let (pk1, vk1) = IntModSpartanSNARK::<E>::setup(shape1.clone()).unwrap();
    let (_, vk2) = IntModSpartanSNARK::<E>::setup(shape2).unwrap();
    assert_ne!(vk1.digest().unwrap(), vk2.digest().unwrap());

    // End-to-end: a valid proof for shape1 must be rejected by shape2's vk.
    let w = vec![
      E::Scalar::from(3u64),
      E::Scalar::from(5u64),
      E::Scalar::from(1u64),
      E::Scalar::ZERO,
    ];
    let q = vec![E::Scalar::from(1u64), E::Scalar::ZERO];
    let (W, U) = IntModR1CSWitness::<E>::new(&shape1, &pk1.ck, w, q, vec![]).unwrap();
    let proof = IntModSpartanSNARK::<E>::prove(&pk1, &U, &W).unwrap();
    proof.verify(&vk1, &U).unwrap();
    assert!(proof.verify(&vk2, &U).is_err());
  }

  #[test]
  fn test_imod_spartan_digest_binds_matrices() {
    use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
    run_digest_binds_matrices_with::<PallasHyraxEngine>();
    run_digest_binds_matrices_with::<T256HyraxEngine>();
  }
}
