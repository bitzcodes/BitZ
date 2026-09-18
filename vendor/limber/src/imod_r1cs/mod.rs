//! Integer Mod-R1CS relation types (paper Def 5.4) — the single-field
//! prototype; `imod_r1cs_modp` is the dual-field relation.
//!
//! Simplifications relative to `imod_r1cs_modp`:
//!   - single field: we use `E::Scalar` (= `Fq`, the curve scalar field) as
//!     both the sumcheck field and the PCS field. The paper's separate prime
//!     `p` is what `imod_r1cs_modp` / `imod_spartan_modp` add.
//!   - no limb split (ν_z = ν_q = 1).
//!   - no range checks (norm bounds are metadata only).
//!
//! Convention for z layout mirrors `r1cs::R1CSShape`:
//!     z = (W, 1, X)             length num_vars + 1 + num_io
//! Matrix columns:
//!     [0, num_vars)             witness w
//!     num_vars                  constant 1
//!     [num_vars + 1, …)         public input x
//!
//! q is a separate witness polynomial of length num_cons, indexed by row.

use crate::{
  Blind, Commitment, CommitmentKey, DEFAULT_COMMITMENT_WIDTH, PCS, VerifierKey,
  digest::SimpleDigestible,
  errors::SpartanError,
  r1cs::SparseMatrix,
  traits::{Engine, pcs::PCSEngineTrait},
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Shape of an Integer Mod-R1CS instance: matrices A, B, C plus the per-row
/// modulus vector `mods` (paper's `m`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IntModR1CSShape<E: Engine> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,
  pub(crate) A: SparseMatrix<E::Scalar>,
  pub(crate) B: SparseMatrix<E::Scalar>,
  pub(crate) C: SparseMatrix<E::Scalar>,
  pub(crate) mods: Vec<E::Scalar>,
}

impl<E: Engine> SimpleDigestible for IntModR1CSShape<E> {}

/// Witness for an Integer Mod-R1CS instance: the satisfying assignment `w`
/// and the per-row quotients `q`, plus PCS blinds for both commitments.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IntModR1CSWitness<E: Engine> {
  // r1cs witness has a is_small bool, could be a future optimization
  pub(crate) w: Vec<E::Scalar>,
  pub(crate) q: Vec<E::Scalar>,
  pub(crate) r_w: Blind<E>,
  pub(crate) r_q: Blind<E>,
}

/// Public instance: the two witness commitments plus the public input `x`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IntModR1CSInstance<E: Engine> {
  pub(crate) comm_w: Commitment<E>,
  pub(crate) comm_q: Commitment<E>,
  pub(crate) x: Vec<E::Scalar>,
}

impl<E: Engine> IntModR1CSShape<E> {
  /// Build a new shape. Invariants: `num_vars` and `num_cons` are
  /// powers of two, and `num_vars >= 1 + num_io` so the public side fits
  /// inside the upper half of the MLE.
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
    mods: Vec<E::Scalar>,
  ) -> Result<Self, SpartanError> {
    if !num_vars.is_power_of_two() || !num_cons.is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntModR1CSShape requires power-of-two sizes (got num_vars={num_vars}, num_cons={num_cons})"
        ),
      });
    }
    if num_vars < 1 + num_io {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "num_vars ({num_vars}) must be at least 1 + num_io ({})",
          1 + num_io
        ),
      });
    }
    if mods.len() != num_cons {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "mods length ({}) must equal num_cons ({num_cons})",
          mods.len()
        ),
      });
    }
    let num_cols = num_vars + 1 + num_io;
    for m in [&A, &B, &C] {
      for (row, col, _) in m.iter() {
        if row >= num_cons || col >= num_cols {
          return Err(SpartanError::InvalidIndex);
        }
      }
    }
    Ok(Self {
      num_cons,
      num_vars,
      num_io,
      A,
      B,
      C,
      mods,
    })
  }

  /// Convenience constructor that takes raw `(row, col, val)` entries for
  /// each matrix. Builds CSR matrices via the same incremental pattern
  /// bellpepper uses (`SparseMatrix::empty()` + push to public fields),
  /// so external callers (benchmarks, future frontends) don't need access
  /// to the test-only `SparseMatrix::new` COO constructor.
  ///
  /// Entries within each row are sorted by column before insertion.
  pub fn from_entries(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    a_entries: &[(usize, usize, E::Scalar)],
    b_entries: &[(usize, usize, E::Scalar)],
    c_entries: &[(usize, usize, E::Scalar)],
    mods: Vec<E::Scalar>,
  ) -> Result<Self, SpartanError> {
    let num_cols = num_vars + 1 + num_io;
    let to_csr = |entries: &[(usize, usize, E::Scalar)]| -> SparseMatrix<E::Scalar> {
      let mut by_row: Vec<Vec<(usize, E::Scalar)>> = vec![Vec::new(); num_cons];
      for &(r, c, v) in entries {
        by_row[r].push((c, v));
      }
      for row in &mut by_row {
        row.sort_by_key(|x| x.0);
      }
      let mut m = SparseMatrix::<E::Scalar>::empty();
      m.cols = num_cols;
      for row in by_row {
        for (c, v) in row {
          m.data.push(v);
          m.indices.push(c);
        }
        m.indptr.push(m.data.len());
      }
      m
    };
    let mat_a = to_csr(a_entries);
    let mat_b = to_csr(b_entries);
    let mat_c = to_csr(c_entries);
    Self::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
  }

  /// PCS setup. The same commitment key is reused for both `w` (length
  /// `num_vars`) and `q` (length `num_cons`); we size it to the larger of
  /// the two so a single key suffices.
  pub fn commitment_key(&self) -> (CommitmentKey<E>, VerifierKey<E>) {
    let n = self.num_vars.max(self.num_cons);
    E::PCS::setup(b"ck_imod", n, DEFAULT_COMMITMENT_WIDTH)
  }

  /// Returns (Az, Bz, Cz) for z = (w, 1, x).
  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len() != self.num_vars + 1 + self.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let (az, (bz, cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );
    Ok((az?, bz?, cz?))
  }

  /// Check that (instance, witness) satisfies the Integer Mod-R1CS equation
  /// over E::Scalar (p = q). Also verifies that the commitments
  /// open to the claimed w and q.
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &IntModR1CSInstance<E>,
    W: &IntModR1CSWitness<E>,
  ) -> Result<(), SpartanError> {
    if W.w.len() != self.num_vars || W.q.len() != self.num_cons || U.x.len() != self.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let z = [W.w.as_slice(), &[E::Scalar::ONE], U.x.as_slice()].concat();
    let (az, bz, cz) = self.multiply_vec(&z)?;

    let ok_eq = (0..self.num_cons)
      .into_par_iter()
      .all(|i| az[i] * bz[i] == cz[i] + self.mods[i] * W.q[i]);

    let (comm_w_ok, comm_q_ok) = rayon::join(
      || -> Result<bool, SpartanError> {
        let cw = PCS::<E>::commit(ck, &W.w, &W.r_w, false)?;
        Ok(cw == U.comm_w)
      },
      || -> Result<bool, SpartanError> {
        let cq = PCS::<E>::commit(ck, &W.q, &W.r_q, false)?;
        Ok(cq == U.comm_q)
      },
    );
    let comm_w_ok = comm_w_ok?;
    let comm_q_ok = comm_q_ok?;

    if !ok_eq {
      return Err(SpartanError::UnSat {
        reason: "IntMod-R1CS equation does not hold".to_string(),
      });
    }
    if !(comm_w_ok && comm_q_ok) {
      return Err(SpartanError::UnSat {
        reason: "IntMod-R1CS commitment mismatch".to_string(),
      });
    }
    Ok(())
  }
}

impl<E: Engine> IntModR1CSWitness<E> {
  /// Commits to (w, q) under the supplied key and returns the witness/instance pair.
  pub fn new(
    shape: &IntModR1CSShape<E>,
    ck: &CommitmentKey<E>,
    w: Vec<E::Scalar>,
    q: Vec<E::Scalar>,
    x: Vec<E::Scalar>,
  ) -> Result<(Self, IntModR1CSInstance<E>), SpartanError> {
    if w.len() != shape.num_vars || q.len() != shape.num_cons || x.len() != shape.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let r_w = PCS::<E>::blind(ck, shape.num_vars);
    let r_q = PCS::<E>::blind(ck, shape.num_cons);
    let (comm_w, comm_q) = rayon::join(
      || PCS::<E>::commit(ck, &w, &r_w, false),
      || PCS::<E>::commit(ck, &q, &r_q, false),
    );
    let comm_w = comm_w?;
    let comm_q = comm_q?;
    Ok((
      Self { w, q, r_w, r_q },
      IntModR1CSInstance { comm_w, comm_q, x },
    ))
  }
}
