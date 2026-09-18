//! Integer Mod-R1CS relation over arbitrary-precision integers,
//! parameterized over a `ModEngine`.
//!
//! In contrast to `imod_r1cs` (which uses the curve scalar field directly),
//! this relation stores the matrices, mods, and witness as `BigUint` integers —
//! they're chosen at shape-construction time, *before* the verifier samples
//! the runtime prime `p`. The SNARK driver reduces the integer data into
//! `M::Scalar` (the dynamic-prime field `Z_p`) once `p` is sampled inside
//! `prove`/`verify`.
//!
//! Matrices are stored as raw COO `Vec<(row, col, value)>` triples (we can't
//! reuse `SparseMatrix<F: PrimeField>` since `BigUint` isn't a field).
//! `is_sat` checks the relation over Z: `A·z ∘ B·z = C·z + m ∘ q`.
//!
//! Invariants: `num_vars`, `num_cons` are powers of two,
//! `num_vars ≥ 1 + num_io`, and `mods.len() == num_cons`.

use crate::{
  errors::SpartanError,
  start_span,
  traits::mod_engine::{ModEngine, ModPCSEngineTrait, SmallValueBlock},
};
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use tracing::info;

/// An aligned witness segment committed at its own value-width bound
/// (`log_t_f`), for width-grouped commitment. `[start, start+2^log_len)`
/// must be aligned (`start % 2^log_len == 0`); the segments tile
/// `[0, num_vars)`. A narrow segment (small `log_t_f`) commits at fewer
/// limbs, so its Mod-PCS commit + range check + opening are cheaper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WidthSegment {
  /// Aligned start column.
  pub start: usize,
  /// log2 of the segment length.
  pub log_len: usize,
  /// Commitment norm bound (bits) for this segment's values.
  pub log_t_f: usize,
}

impl WidthSegment {
  /// Segment length `2^log_len`.
  pub fn size(&self) -> usize {
    1usize << self.log_len
  }
}

type ModPCS<M> = <M as ModEngine>::ModPCS;
type ModCK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::CommitmentKey;
type ModVK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::VerifierKey;
type ModComm<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::Commitment;
type ModBlind<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::Blind;

/// IntMod-R1CS shape over `M: ModEngine`. Integer-valued
/// matrices/mods; `p`-independent so the same shape can be used with any
/// verifier-sampled prime.
///
/// Per-row modulus semantics (`mods[r]` in `Az∘Bz = Cz + m∘q`):
/// - `m ≥ 2`: ordinary modular row, `LC_A·LC_B ≡ LC_C (mod m)` with the
///   prover's quotient `q_r` as advice.
/// - `m = 0`: **exact integer row** — the `m·q` term vanishes, so the row
///   enforces `LC_A·LC_B = LC_C` over ℤ (congruence mod 0 is equality).
///   Used for e.g. bit constraints `b·b = b`. The `q_r` slot is dead
///   weight (multiplied by zero); conventionally set to 0.
/// - `m = 1` is degenerate (everything is ≡ mod 1, so the quotient can
///   absorb up to `T_f` of residual): avoid.
#[derive(Clone, Debug)]
pub struct IntModR1CSShapeModp<M: ModEngine> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,
  pub(crate) A: Vec<(usize, usize, BigUint)>,
  pub(crate) B: Vec<(usize, usize, BigUint)>,
  pub(crate) C: Vec<(usize, usize, BigUint)>,
  pub(crate) mods: Vec<BigUint>,
  /// Aligned witness blocks asserted `< 2^16` by the Mod-PCS (no rows).
  pub(crate) small_blocks: Vec<SmallValueBlock>,
  /// Width-grouped commitment segments tiling `[0, num_vars)`; empty means
  /// a single uniform segment (the default).
  pub(crate) width_segments: Vec<WidthSegment>,
  pub(crate) _phantom: core::marker::PhantomData<M>,
}

/// Witness: integer assignment `w`, integer quotients `q`, and the Mod-PCS
/// blinds for the (integer-valued) commitments.
#[derive(Clone, Debug)]
pub struct IntModR1CSWitnessModp<M: ModEngine> {
  pub(crate) w: Vec<BigUint>,
  pub(crate) q: Vec<BigUint>,
  pub(crate) r_w: Vec<ModBlind<M>>,
  pub(crate) r_q: ModBlind<M>,
}

/// Public instance: integer-valued IO and the integer-valued commitments.
#[derive(Clone, Debug)]
pub struct IntModR1CSInstanceModp<M: ModEngine> {
  pub(crate) comm_w: Vec<ModComm<M>>,
  pub(crate) comm_q: ModComm<M>,
  pub(crate) x: Vec<BigUint>,
}

impl<M: ModEngine> IntModR1CSShapeModp<M> {
  /// Build a new shape. Invariants: power-of-two sizes,
  /// `num_vars ≥ 1 + num_io`, `mods.len() == num_cons`.
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: Vec<(usize, usize, BigUint)>,
    B: Vec<(usize, usize, BigUint)>,
    C: Vec<(usize, usize, BigUint)>,
    mods: Vec<BigUint>,
  ) -> Result<Self, SpartanError> {
    if !num_vars.is_power_of_two() || !num_cons.is_power_of_two() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "IntModR1CSShapeModp requires power-of-two sizes (got num_vars={num_vars}, num_cons={num_cons})"
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
    for entries in [&A, &B, &C] {
      for (row, col, _) in entries {
        if *row >= num_cons || *col >= num_cols {
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
      small_blocks: Vec::new(),
      width_segments: Vec::new(),
      _phantom: core::marker::PhantomData,
    })
  }

  /// Declare aligned witness blocks whose values the Mod-PCS asserts to
  /// be `< 2^16` (see [`SmallValueBlock`]) — the SNARK's range lookup for
  /// chunk decompositions. Blocks enter the shape digest.
  pub fn with_small_value_blocks(
    mut self,
    blocks: Vec<SmallValueBlock>,
  ) -> Result<Self, SpartanError> {
    let n = self.num_vars.trailing_zeros() as usize;
    for b in &blocks {
      b.validate(n)?;
    }
    self.small_blocks = blocks;
    Ok(self)
  }

  /// The declared small-value blocks.
  pub fn small_value_blocks(&self) -> &[SmallValueBlock] {
    &self.small_blocks
  }

  /// Declare width-grouped commitment segments. They must tile
  /// `[0, num_vars)` with aligned starts (each `start % size == 0`), be
  /// sorted and contiguous, and every column must be covered exactly
  /// once. Segments enter the shape digest.
  pub fn with_width_segments(mut self, segments: Vec<WidthSegment>) -> Result<Self, SpartanError> {
    let mut cursor = 0usize;
    for s in &segments {
      if s.start != cursor
        || s.start % s.size() != 0
        || s.log_len > self.num_vars.trailing_zeros() as usize
      {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "WidthSegment {{start:{}, log_len:{}}} not aligned/contiguous at cursor {cursor}",
            s.start, s.log_len
          ),
        });
      }
      cursor += s.size();
    }
    if !segments.is_empty() && cursor != self.num_vars {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "width segments cover {cursor} columns, expected num_vars={}",
          self.num_vars
        ),
      });
    }
    self.width_segments = segments;
    Ok(self)
  }

  /// The declared width segments (empty = single uniform segment).
  pub fn width_segments(&self) -> &[WidthSegment] {
    &self.width_segments
  }

  /// Number of witness columns (`|w|`), a power of two.
  pub fn num_vars(&self) -> usize {
    self.num_vars
  }

  /// Number of constraint rows, a power of two.
  pub fn num_cons(&self) -> usize {
    self.num_cons
  }

  /// Borrow the integer constraint data for benchmark provenance hashing.
  ///
  /// This is intentionally a read-only support hook: benchmark code can hash
  /// the fully validated shape after construction without cloning its sparse
  /// matrices or extending the measured witness-generation interval.
  #[doc(hidden)]
  pub fn benchmark_constraint_parts(
    &self,
  ) -> (
    &[(usize, usize, BigUint)],
    &[(usize, usize, BigUint)],
    &[(usize, usize, BigUint)],
    &[BigUint],
  ) {
    (&self.A, &self.B, &self.C, &self.mods)
  }

  /// Mod-PCS commitment-key setup. Sized to the larger of `num_vars` /
  /// `num_cons` so a single key can commit either `w` or `q`.
  pub fn commitment_key(&self) -> (ModCK<M>, ModVK<M>) {
    let n = self.num_vars.max(self.num_cons);
    <ModPCS<M> as ModPCSEngineTrait<M>>::setup(b"ck_imod_modp", n, crate::DEFAULT_COMMITMENT_WIDTH)
  }

  /// Integer SpMV: returns `(A·z, B·z, C·z)` for `z` of length
  /// `num_vars + 1 + num_io`. All values are non-negative `BigUint`s.
  pub fn multiply_vec(
    &self,
    z: &[BigUint],
  ) -> Result<(Vec<BigUint>, Vec<BigUint>, Vec<BigUint>), SpartanError> {
    if z.len() != self.num_vars + 1 + self.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let multiply = |entries: &Vec<(usize, usize, BigUint)>| -> Vec<BigUint> {
      let mut out = vec![BigUint::zero(); self.num_cons];
      for (i, j, v) in entries {
        out[*i] += v * &z[*j];
      }
      out
    };
    let (az, (bz, cz)) = rayon::join(
      || multiply(&self.A),
      || rayon::join(|| multiply(&self.B), || multiply(&self.C)),
    );
    Ok((az, bz, cz))
  }

  /// Check the integer relation `A·z ∘ B·z = C·z + m ∘ q`, and that the
  /// commitments open to the claimed `w` and `q`. All arithmetic is over
  /// the non-negative integers.
  pub fn is_sat(
    &self,
    ck: &ModCK<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
  ) -> Result<(), SpartanError> {
    if W.w.len() != self.num_vars || W.q.len() != self.num_cons || U.x.len() != self.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let z = [W.w.as_slice(), &[BigUint::from(1u32)], U.x.as_slice()].concat();
    let (az, bz, cz) = self.multiply_vec(&z)?;

    let ok_eq = (0..self.num_cons)
      .into_par_iter()
      .all(|i| &az[i] * &bz[i] == &cz[i] + &self.mods[i] * &W.q[i]);
    // Small-value blocks are asserted by the Mod-PCS, not by rows; check
    // them here so an out-of-range witness is caught before proving.
    let ok_blocks = self.small_blocks.iter().all(|b| {
      W.w[b.start..b.start + b.size()]
        .iter()
        .all(|v| v.bits() <= 16)
    });

    let (comm_w_ok, comm_q_ok) = rayon::join(
      || -> Result<bool, SpartanError> {
        // Re-commit the witness the same way `new` did: per width segment,
        // or as one commitment when the shape declares none.
        let segs = self.width_segments();
        if segs.is_empty() {
          let cw = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &W.w, &W.r_w[0])?;
          Ok(U.comm_w.len() == 1 && cw == U.comm_w[0])
        } else {
          if U.comm_w.len() != segs.len() || W.r_w.len() != segs.len() {
            return Ok(false);
          }
          for (i, seg) in segs.iter().enumerate() {
            let slice = &W.w[seg.start..seg.start + seg.size()];
            let cw =
              <ModPCS<M> as ModPCSEngineTrait<M>>::commit_at(ck, slice, &W.r_w[i], seg.log_t_f)?;
            if cw != U.comm_w[i] {
              return Ok(false);
            }
          }
          Ok(true)
        }
      },
      || -> Result<bool, SpartanError> {
        let cq = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &W.q, &W.r_q)?;
        Ok(cq == U.comm_q)
      },
    );
    let comm_w_ok = comm_w_ok?;
    let comm_q_ok = comm_q_ok?;

    if !ok_eq {
      return Err(SpartanError::UnSat {
        reason: "IntMod-R1CS equation does not hold over Z".to_string(),
      });
    }
    if !ok_blocks {
      return Err(SpartanError::UnSat {
        reason: "IntMod-R1CS small-value block holds a value >= 2^16".to_string(),
      });
    }
    if !(comm_w_ok && comm_q_ok) {
      return Err(SpartanError::UnSat {
        reason: "IntMod-R1CS commitment mismatch".to_string(),
      });
    }
    Ok(())
  }

  /// Hash the shape's public data to a 32-byte digest for transcript
  /// binding. `imod_r1cs` uses `bincode`-based `Digestible`; here we hash raw
  /// bytes since `BigUint` is serializable but we want a stable, simple
  /// byte layout independent of `serde`'s encoding choices.
  pub fn digest(&self) -> [u8; 32] {
    use sha3::{Digest, Keccak256};
    let mut h = Keccak256::new();
    h.update(b"IntModR1CSShapeModp");
    h.update((self.num_cons as u64).to_le_bytes());
    h.update((self.num_vars as u64).to_le_bytes());
    h.update((self.num_io as u64).to_le_bytes());
    for entries in [&self.A, &self.B, &self.C] {
      h.update((entries.len() as u64).to_le_bytes());
      for (i, j, v) in entries {
        h.update((*i as u64).to_le_bytes());
        h.update((*j as u64).to_le_bytes());
        let bytes = v.to_bytes_le();
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
      }
    }
    h.update((self.mods.len() as u64).to_le_bytes());
    for m in &self.mods {
      let bytes = m.to_bytes_le();
      h.update((bytes.len() as u64).to_le_bytes());
      h.update(&bytes);
    }
    h.update((self.small_blocks.len() as u64).to_le_bytes());
    for b in &self.small_blocks {
      h.update((b.start as u64).to_le_bytes());
      h.update((b.log_len as u64).to_le_bytes());
    }
    h.update((self.width_segments.len() as u64).to_le_bytes());
    for s in &self.width_segments {
      h.update((s.start as u64).to_le_bytes());
      h.update((s.log_len as u64).to_le_bytes());
      h.update((s.log_t_f as u64).to_le_bytes());
    }
    h.finalize().into()
  }
}

impl IntModR1CSShapeModp<crate::provider::T256DynPrimeEngine> {
  /// Mod-PCS key setup with explicit IntEval params, overriding the
  /// default `DEFAULT_LOG_T_F`. Used by callers that commit wide-operand
  /// witnesses (e.g. MultiSwap's ~2048-bit `mod N` values). Sized to the
  /// larger of `num_vars` / `num_cons`, matching `commitment_key`.
  pub fn commitment_key_with_params(
    &self,
    params: crate::provider::pcs::integer_modpcs::IntEvalParams,
  ) -> Result<
    (
      ModCK<crate::provider::T256DynPrimeEngine>,
      ModVK<crate::provider::T256DynPrimeEngine>,
    ),
    SpartanError,
  > {
    let n = self.num_vars.max(self.num_cons);
    crate::provider::pcs::integer_modpcs::IntegerModPCS::setup_with_params(
      b"ck_imod_modp",
      n,
      crate::DEFAULT_COMMITMENT_WIDTH,
      params,
    )
  }
}

impl<M: ModEngine> IntModR1CSInstanceModp<M> {
  /// Exact canonical bytes of the per-proof input-commitment pair, as the
  /// single tuple `(comm_w, comm_q)` under the crate's pinned bincode
  /// configuration (little-endian, fixed-int). Bincode adds no outer
  /// tuple tag or length — the elements are consecutive, though their own
  /// representations may contain length prefixes — so this is one
  /// canonical encoding, not the sum of two independently chosen ones.
  /// Proof-size accounting: these commitments are freshly created per
  /// proof and transmitted with it.
  pub fn commitment_bytes(&self) -> Result<Vec<u8>, SpartanError> {
    crate::imod_spartan_modp::to_canonical_bytes(&(&self.comm_w, &self.comm_q))
  }
}

impl<M: ModEngine> IntModR1CSWitnessModp<M> {
  /// Read-only integer assignment for untimed benchmark provenance checks.
  pub fn benchmark_assignment(&self) -> (&[BigUint], &[BigUint]) {
    (&self.w, &self.q)
  }

  /// Commit to integer `(w, q)` and return the witness/instance pair.
  pub fn new(
    shape: &IntModR1CSShapeModp<M>,
    ck: &ModCK<M>,
    w: Vec<BigUint>,
    q: Vec<BigUint>,
    x: Vec<BigUint>,
  ) -> Result<(Self, IntModR1CSInstanceModp<M>), SpartanError> {
    if w.len() != shape.num_vars || q.len() != shape.num_cons || x.len() != shape.num_io {
      return Err(SpartanError::InvalidWitnessLength);
    }
    let r_q = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(ck, shape.num_cons);
    let (_wq_span, wq_t) = start_span!("imod_modp_wq_commit");
    // Witness commitment: one commitment per width-grouped segment (each at
    // its own value-width bound), or a single commitment over the whole
    // witness when the shape declares no segments.
    let segs = shape.width_segments();
    let (r_w, comm_w): (Vec<ModBlind<M>>, Vec<ModComm<M>>) = if segs.is_empty() {
      let r = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(ck, shape.num_vars);
      let c = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &w, &r)?;
      (vec![r], vec![c])
    } else {
      let mut rs = Vec::with_capacity(segs.len());
      let mut cs = Vec::with_capacity(segs.len());
      for seg in segs {
        let slice = &w[seg.start..seg.start + seg.size()];
        let r = <ModPCS<M> as ModPCSEngineTrait<M>>::blind(ck, seg.size());
        let c = <ModPCS<M> as ModPCSEngineTrait<M>>::commit_at(ck, slice, &r, seg.log_t_f)?;
        rs.push(r);
        cs.push(c);
      }
      (rs, cs)
    };
    let comm_q = <ModPCS<M> as ModPCSEngineTrait<M>>::commit(ck, &q, &r_q)?;
    info!(elapsed_ms = %wq_t.elapsed().as_millis(), "imod_modp_wq_commit");
    Ok((
      Self { w, q, r_w, r_q },
      IntModR1CSInstanceModp { comm_w, comm_q, x },
    ))
  }
}
