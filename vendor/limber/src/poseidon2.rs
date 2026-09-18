//! Poseidon2 benchmark workload over three non-native prime fields in one
//! mixed-modulus circuit.
//!
//! Thirty Poseidon2 compressions (`t = 3`, `α = 5`, `R_F = 8`, `R_P = 56`;
//! ten per field at the default `H = 10`) encoded as ONE Integer Mod-R1CS
//! shape over `imod_r1cs_modp`: three independent field blocks — BN254-Fr,
//! BLS12-381-Fr, secp256k1-Fr, in the fixed [`FIELD_ORDER`] — each a
//! `433H`-row / `(434H − 1)`-column chain segment with its own row modulus,
//! concatenated unpadded and padded once to a single power-of-two domain.
//!
//! **This permutation is a benchmark workload, not a security-reviewed
//! production hash.** The round constants are custom BLAKE3-derived values
//! (see [`build_params`]); the construction is cost- and structure-faithful
//! to Poseidon2 ([eprint 2023/323](https://eprint.iacr.org/2023/323)) but is
//! not the published interoperable instance for any of these fields.
//!
//! Each block proves `h_{f,i} = P_f([h_{f,i-1}, m_{f,i}, 0])[0]` for
//! `i = 1..=H` from the same fixed public IV, with `3H` distinct witness
//! message columns and three public digests in [`FIELD_ORDER`]
//! (`num_io = 3`). The proof establishes that, per field, *there exist*
//! witness messages producing that digest; it neither binds them to the
//! deterministic benchmark messages of [`build_inputs`] nor asserts the
//! three private message vectors are equal (the benchmark duplicates the
//! same pinned values across blocks purely as a fixture choice).
//!
//! No zero-knowledge claim is made for this driver: Hyrax commitments are
//! hiding, Brakedown commitments are not, and the sumcheck transcript
//! carries unmasked witness-dependent data regardless of backend.
//! "Messages are private" means *not public IO*, not confidential.
//!
//! Verifiers must go through [`verify_poseidon_chain`], which enforces the
//! ordered three-digest canonicality policy ([`check_canonical_io`]) that
//! the generic `IntModSpartanModpSNARK::verify` cannot know about;
//! bypassing it forfeits the three-digest canonicality guarantee.

use crate::{
  errors::SpartanError,
  imod_r1cs_modp::{IntModR1CSInstanceModp, IntModR1CSShapeModp},
  imod_spartan_modp::{IntModSpartanModpSNARK, IntModSpartanModpVerifierKey},
  provider::keccak::Keccak256Transcript,
  traits::mod_engine::ModEngine,
};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

/// State width of the permutation.
const T: usize = 3;
/// S-box exponent.
const ALPHA: usize = 5;
/// Number of full (external) rounds.
const R_F: usize = 8;
/// Number of partial (internal) rounds.
const R_P: usize = 56;
/// Total rounds.
const ROUNDS: usize = R_F + R_P;
/// Rows contributed by one permutation: 64 linear layers of 3 reduce rows,
/// one lane-0-only terminal reduce row, and 80 S-boxes of 3 rows each.
const ROWS_PER_PERM: usize = 433;
/// Fresh columns contributed by one permutation *plus* its message column;
/// each block's last permutation materializes one column fewer (its public
/// digest).
const COLS_PER_PERM: usize = 434;
/// Number of field blocks in the combined circuit.
const NUM_FIELDS: usize = 3;

/// External (full-round) linear-layer matrix `M_E = circ(2, 1, 1)`.
const M_E: [[u64; 3]; 3] = [[2, 1, 1], [1, 2, 1], [1, 1, 2]];
/// Internal (partial-round) linear-layer matrix `M_I = J + diag(1, 1, 2)`.
/// Distinct from `M_E` per the official Poseidon2 generator's `t = 3` pair;
/// `J + I` fails the §5.3 subspace-trail checks (see
/// [`validate_internal_matrix`]'s regression test).
const M_I: [[u64; 3]; 3] = [[2, 1, 1], [1, 2, 1], [1, 1, 3]];

/// Domain-separation prefix for the round-constant XOF.
const RC_DOMAIN: &[u8] = b"limber-poseidon2-v1/rc";
/// Domain-separation prefix for the benchmark message XOF.
const MSG_DOMAIN: &[u8] = b"limber-poseidon2-v1/msg";

/// Target scalar field for the Poseidon2 workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
  /// BN254 scalar field.
  Bn254Fr,
  /// BLS12-381 scalar field.
  Bls12381Fr,
  /// secp256k1 scalar field.
  Secp256k1Fr,
}

/// Canonical field-block and public-IO order for the combined circuit.
pub const FIELD_ORDER: [Field; 3] = [Field::Bn254Fr, Field::Bls12381Fr, Field::Secp256k1Fr];

impl Field {
  /// Short lowercase identifier used in benchmark IDs and artifacts.
  pub fn name(&self) -> &'static str {
    match self {
      Field::Bn254Fr => "bn254",
      Field::Bls12381Fr => "bls12_381",
      Field::Secp256k1Fr => "secp256k1",
    }
  }

  /// The field's prime modulus.
  fn modulus(&self) -> BigUint {
    let hex: &[u8] = match self {
      Field::Bn254Fr => b"30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
      Field::Bls12381Fr => b"73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
      Field::Secp256k1Fr => b"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
    };
    BigUint::parse_bytes(hex, 16).expect("hard-coded field modulus parses")
  }
}

/// Parameters of the Poseidon2 workload for one target field: the modulus,
/// both linear-layer matrices, and the 80 BLAKE3-derived round constants.
/// Fields are private; construct via [`build_params`] and inspect through
/// [`Poseidon2Params::modulus`].
#[derive(Clone, Debug)]
pub struct Poseidon2Params {
  /// Target field modulus `p`.
  modulus: BigUint,
  /// External matrix (validated MDS).
  m_e: [[u64; 3]; 3],
  /// Internal matrix (validated MDS + subspace-trail Algorithms 1–3).
  m_i: [[u64; 3]; 3],
  /// Round constants: `rc[r]` holds round `r + 1`'s constants — three for
  /// full rounds (`r + 1 ∈ 1..=4 ∪ 61..=64`), one (lane 0) for partial.
  rc: Vec<Vec<BigUint>>,
}

impl Poseidon2Params {
  /// The target field's prime modulus.
  pub fn modulus(&self) -> &BigUint {
    &self.modulus
  }
}

/// Fixed-order parameters for all three field blocks. Fields and
/// constructor are private: only [`build_all_params`] can create one, and
/// it fills [`FIELD_ORDER`], so a caller cannot swap a modulus into another
/// block while retaining a self-consistent but mislabeled circuit.
pub struct Poseidon2ParamsSet {
  /// One parameter set per block, in [`FIELD_ORDER`].
  params: [Poseidon2Params; NUM_FIELDS],
}

impl Poseidon2ParamsSet {
  /// The parameters for one field.
  pub fn get(&self, field: Field) -> &Poseidon2Params {
    let idx = FIELD_ORDER
      .iter()
      .position(|f| *f == field)
      .expect("FIELD_ORDER contains every Field variant");
    &self.params[idx]
  }

  /// The three ordered block moduli.
  fn ordered_moduli(&self) -> [BigUint; NUM_FIELDS] {
    core::array::from_fn(|f| self.params[f].modulus.clone())
  }
}

/// Row/column bookkeeping of the combined shape: real (unpadded)
/// dimensions, the per-block boundaries, the padded power-of-two
/// dimensions, the three ordered target moduli, and the completed shape
/// digest. Private fields; construct via [`build_shape`].
#[derive(Clone, Debug)]
pub struct Layout {
  /// Number of chained compressions per field block.
  hashes_per_field: usize,
  /// Total compressions across the three blocks: `3H`.
  total_hashes: usize,
  /// Rows per block: `433H`.
  block_rows: usize,
  /// Real witness columns per block: `434H − 1`.
  block_cols: usize,
  /// Real constraint rows: `1299H`.
  real_rows: usize,
  /// Real witness columns: `1302H − 3`.
  real_cols: usize,
  /// Padded row count (power of two).
  num_cons: usize,
  /// Padded witness-column count (power of two).
  num_vars: usize,
  /// `log₂(max(num_cons, num_vars))`.
  log_n: usize,
  /// The three ordered target moduli the row segments reduce by.
  moduli: [BigUint; NUM_FIELDS],
  /// `IntModR1CSShapeModp::digest()` of the completed shape.
  shape_digest: [u8; 32],
}

impl Layout {
  /// Number of chained compressions per field block (`H`).
  pub fn hashes_per_field(&self) -> usize {
    self.hashes_per_field
  }

  /// Total compressions across the three blocks (`3H`).
  pub fn total_hashes(&self) -> usize {
    self.total_hashes
  }

  /// Real (unpadded) constraint-row count, `1299H`.
  pub fn real_rows(&self) -> usize {
    self.real_rows
  }

  /// Real (unpadded) witness-column count, `1302H − 3`.
  pub fn real_cols(&self) -> usize {
    self.real_cols
  }

  /// Padded constraint-row count (power of two).
  pub fn num_cons(&self) -> usize {
    self.num_cons
  }

  /// Padded witness-column count (power of two).
  pub fn num_vars(&self) -> usize {
    self.num_vars
  }

  /// `log₂(max(num_cons, num_vars))` — the multilinear variable count the
  /// Mod-PCS parameters are derived for.
  pub fn log_n(&self) -> usize {
    self.log_n
  }

  /// Row range of block `f` (crate-side: verifier-key predicates and
  /// structural tests).
  fn block_row_range(&self, f: usize) -> core::ops::Range<usize> {
    f * self.block_rows..(f + 1) * self.block_rows
  }
}

/// Reject invalid per-field chain lengths before any dimension arithmetic:
/// `H = 0` (would evaluate `434H − 1` as an underflow) and `H > u32::MAX`
/// (the message derivation encodes `j` as a `u32`). Shared by every public
/// chain entry point.
fn checked_hashes(hashes: usize) -> Result<u32, SpartanError> {
  if hashes == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "poseidon2: per-field hash count H must be at least 1".to_string(),
    });
  }
  u32::try_from(hashes).map_err(|_| SpartanError::InvalidInputLength {
    reason: format!("poseidon2: per-field hash count H = {hashes} exceeds u32::MAX"),
  })
}

/// Real per-block, real combined, and padded dimensions for `H`
/// compressions per field, all arithmetic checked — `H = 0` is rejected
/// before `3(434H − 1)` is evaluated, and every multiplication by three is
/// checked. Returns
/// `(block_rows, block_cols, real_rows, real_cols, num_cons, num_vars, log_n)`.
#[allow(clippy::type_complexity)]
fn checked_dims(
  hashes: usize,
) -> Result<(usize, usize, usize, usize, usize, usize, usize), SpartanError> {
  checked_hashes(hashes)?;
  let overflow = || SpartanError::InvalidInputLength {
    reason: format!("poseidon2: dimension arithmetic overflows for H = {hashes}"),
  };
  let block_rows = ROWS_PER_PERM.checked_mul(hashes).ok_or_else(overflow)?;
  let block_cols = COLS_PER_PERM
    .checked_mul(hashes)
    .and_then(|c| c.checked_sub(1))
    .ok_or_else(overflow)?;
  let real_rows = block_rows.checked_mul(NUM_FIELDS).ok_or_else(overflow)?;
  let real_cols = block_cols.checked_mul(NUM_FIELDS).ok_or_else(overflow)?;
  let num_cons = real_rows.checked_next_power_of_two().ok_or_else(overflow)?;
  let num_vars = real_cols.checked_next_power_of_two().ok_or_else(overflow)?;
  let log_n = num_cons.max(num_vars).ilog2() as usize;
  Ok((
    block_rows, block_cols, real_rows, real_cols, num_cons, num_vars, log_n,
  ))
}

/// The fixed public chain IV `h₀ = 2^64 + 0x9e3779b97f4a7c15`, embedded
/// separately in each field block's `A` entries on the constant column.
fn chain_iv() -> BigUint {
  (BigUint::one() << 64u32) + BigUint::from(0x9e37_79b9_7f4a_7c15u64)
}

/// One BLAKE3-XOF draw of the round-constant derivation: 32 output bytes,
/// interpreted big-endian. The counter increments per draw (including
/// rejected draws).
fn rc_draw(p_be32: &[u8; 32], counter: u32) -> BigUint {
  let mut hasher = blake3::Hasher::new();
  hasher.update(RC_DOMAIN);
  hasher.update(p_be32);
  hasher.update(&[T as u8, R_F as u8, R_P as u8, ALPHA as u8]);
  hasher.update(&counter.to_be_bytes());
  let mut out = [0u8; 32];
  hasher.finalize_xof().fill(&mut out);
  BigUint::from_bytes_be(&out)
}

/// Derive the 80 round constants for modulus `p` by rejection sampling the
/// BLAKE3 XOF (unbiased), and assert every constant is nonzero (the
/// structural COO-entry count of [`build_shape`] depends on it).
fn derive_round_constants(p: &BigUint) -> Result<Vec<Vec<BigUint>>, SpartanError> {
  let p_bytes = p.to_bytes_be();
  if p_bytes.len() > 32 {
    return Err(SpartanError::InvalidInputLength {
      reason: "poseidon2: modulus wider than 32 bytes".to_string(),
    });
  }
  let mut p_be32 = [0u8; 32];
  p_be32[32 - p_bytes.len()..].copy_from_slice(&p_bytes);

  let mut counter: u32 = 0;
  let draw = |counter: &mut u32| -> Result<BigUint, SpartanError> {
    loop {
      let v = rc_draw(&p_be32, *counter);
      *counter = counter.checked_add(1).ok_or(SpartanError::InternalError {
        reason: "poseidon2: round-constant counter overflow".to_string(),
      })?;
      if &v < p {
        return Ok(v);
      }
    }
  };

  let mut rc = Vec::with_capacity(ROUNDS);
  for round in 1..=ROUNDS {
    let lanes = if is_full_round(round) { T } else { 1 };
    let mut per_round = Vec::with_capacity(lanes);
    for _ in 0..lanes {
      let c = draw(&mut counter)?;
      if c.is_zero() {
        return Err(SpartanError::InternalError {
          reason: format!("poseidon2: zero round constant drawn for round {round}"),
        });
      }
      per_round.push(c);
    }
    rc.push(per_round);
  }
  Ok(rc)
}

/// Whether round `r` (1-based, `1..=64`) is a full (external) round.
fn is_full_round(r: usize) -> bool {
  r <= R_F / 2 || r > R_F / 2 + R_P
}

/// Whether linear layer `l` (`0..=64`; 0 is the initial layer, `l ≥ 1` ends
/// round `l`) applies the external matrix `M_E`.
fn layer_is_external(l: usize) -> bool {
  l == 0 || is_full_round(l)
}

// ---------------------------------------------------------------------------
// Matrix security validation (build_params)

/// 3×3 matrix–vector product over `Z_p`, entries as canonical residues.
fn mat_vec(m: &[[BigUint; 3]; 3], v: &[BigUint; 3], p: &BigUint) -> [BigUint; 3] {
  core::array::from_fn(|i| {
    let mut acc = BigUint::zero();
    for j in 0..3 {
      acc += &m[i][j] * &v[j];
    }
    acc % p
  })
}

/// 3×3 matrix product over `Z_p`.
fn mat_mul(a: &[[BigUint; 3]; 3], b: &[[BigUint; 3]; 3], p: &BigUint) -> [[BigUint; 3]; 3] {
  core::array::from_fn(|i| {
    core::array::from_fn(|j| {
      let mut acc = BigUint::zero();
      for k in 0..3 {
        acc += &a[i][k] * &b[k][j];
      }
      acc % p
    })
  })
}

/// Lift a small unsigned integer matrix into canonical `Z_p` residues.
fn mat_mod_p(m: &[[u64; 3]; 3], p: &BigUint) -> [[BigUint; 3]; 3] {
  core::array::from_fn(|i| core::array::from_fn(|j| BigUint::from(m[i][j]) % p))
}

/// `a − b mod p` on canonical residues.
fn sub_mod(a: &BigUint, b: &BigUint, p: &BigUint) -> BigUint {
  ((a + p) - b) % p
}

/// Whether a matrix is scalar (`λ·I`) mod `p`.
fn is_scalar_matrix(m: &[[BigUint; 3]; 3], p: &BigUint) -> bool {
  for (i, row) in m.iter().enumerate() {
    for (j, v) in row.iter().enumerate() {
      if i != j && !(v % p).is_zero() {
        return false;
      }
    }
  }
  m[0][0] == m[1][1] && m[1][1] == m[2][2]
}

/// Whether two 3-vectors over `Z_p` are proportional (all 2×2 minors zero).
/// The zero vector is proportional to everything.
fn proportional(a: &[BigUint; 3], b: &[BigUint; 3], p: &BigUint) -> bool {
  for i in 0..3 {
    for j in (i + 1)..3 {
      let d = sub_mod(&(&a[i] * &b[j] % p), &(&a[j] * &b[i] % p), p);
      if !d.is_zero() {
        return false;
      }
    }
  }
  true
}

/// Determinant of a 3×3 matrix over `Z_p` (canonical residues).
fn det3(m: &[[BigUint; 3]; 3], p: &BigUint) -> BigUint {
  let pos = &m[0][0] * &m[1][1] * &m[2][2]
    + &m[0][1] * &m[1][2] * &m[2][0]
    + &m[0][2] * &m[1][0] * &m[2][1];
  let neg = &m[0][2] * &m[1][1] * &m[2][0]
    + &m[0][0] * &m[1][2] * &m[2][1]
    + &m[0][1] * &m[1][0] * &m[2][2];
  sub_mod(&(pos % p), &(neg % p), p)
}

/// Reject an internal-matrix candidate that admits the subspace trails of
/// [eprint 2023/323] §5.3, via the official generator's `t = 3, s = 1`
/// Algorithms 1–3 specialized to exact modular rank/proportionality checks:
///
/// 1. `M` and `M²` are non-scalar;
/// 2. the inactive subspace `S₁ = span(e₁, e₂)` satisfies `S₁ ≠ M S₁`;
/// 3. `S₂ = span((0, 1, −1))` differs from both `M S₂` and `M² S₂`;
/// 4. no eigenline of `M` lies inside `S₁` (the invariant-line rejection);
/// 5. Algorithms 2–3 reduce to `D_r = det[e₀, M^r e₀, M^(2r) e₀] ≠ 0` for
///    `r = 1..=12` (`= 4t`).
fn validate_internal_matrix(m_int: &[[u64; 3]; 3], p: &BigUint) -> Result<(), SpartanError> {
  let fail = |reason: String| SpartanError::InternalError {
    reason: format!("poseidon2 internal-matrix validation: {reason}"),
  };
  let m = mat_mod_p(m_int, p);
  let m2 = mat_mul(&m, &m, p);

  // 1. Non-scalar M and M².
  if is_scalar_matrix(&m, p) {
    return Err(fail("M is a scalar matrix".to_string()));
  }
  if is_scalar_matrix(&m2, p) {
    return Err(fail("M^2 is a scalar matrix".to_string()));
  }

  // 2. S₁ = span(e₁, e₂) must not be M-invariant: M e₁ and M e₂ must not
  // both have zero first coordinate.
  if (&m[0][1] % p).is_zero() && (&m[0][2] % p).is_zero() {
    return Err(fail(
      "inactive subspace span(e1, e2) is M-invariant".to_string(),
    ));
  }

  // 3. S₂ = span((0, 1, −1)) must differ from M S₂ and M² S₂.
  let v = [BigUint::zero(), BigUint::one(), p - BigUint::one()];
  let mv = mat_vec(&m, &v, p);
  let m2v = mat_vec(&m2, &v, p);
  if proportional(&mv, &v, p) {
    return Err(fail("span((0,1,-1)) is M-invariant".to_string()));
  }
  if proportional(&m2v, &v, p) {
    return Err(fail("span((0,1,-1)) is M^2-invariant".to_string()));
  }

  // 4. No eigenline of M inside S₁. A vector v = (0, a, b) with Mv ∈ S₁
  // forces M[0][1]·a + M[0][2]·b ≡ 0; up to scale the unique such line is
  // (a, b) = (M[0][2], −M[0][1]) (nontrivial by check 2). Reject if Mv is
  // proportional to v on the last two coordinates.
  let a = m[0][2].clone();
  let b = sub_mod(&BigUint::zero(), &m[0][1], p);
  let u = mat_vec(&m, &[BigUint::zero(), a.clone(), b.clone()], p);
  debug_assert!(u[0].is_zero(), "eigenline candidate must stay in S1");
  let d = sub_mod(&(&a * &u[2] % p), &(&b * &u[1] % p), p);
  if d.is_zero() {
    return Err(fail(
      "an eigenline of M lies inside span(e1, e2)".to_string(),
    ));
  }

  // 5. Algorithms 2–3: D_r = det[e₀, M^r e₀, M^(2r) e₀] ≠ 0 for r = 1..=12.
  let mut m_r = m.clone();
  for r in 1..=12usize {
    let m_2r = mat_mul(&m_r, &m_r, p);
    let e0 = [BigUint::one(), BigUint::zero(), BigUint::zero()];
    let c1 = mat_vec(&m_r, &e0, p);
    let c2 = mat_vec(&m_2r, &e0, p);
    let d_mat: [[BigUint; 3]; 3] =
      core::array::from_fn(|i| [e0[i].clone(), c1[i].clone(), c2[i].clone()]);
    if det3(&d_mat, p).is_zero() {
      return Err(fail(format!(
        "subspace-trail determinant D_{r} vanishes mod p"
      )));
    }
    m_r = mat_mul(&m_r, &m, p);
  }
  Ok(())
}

/// MDS check for a matrix of the form `M = J + diag(μ₀−1, μ₁−1, μ₂−1)`:
/// requires `μᵢ ≠ 0`, `μᵢ ≠ 1`, `μᵢ·μⱼ ≠ 1`, and
/// `μ₀μ₁μ₂ − μ₀ − μ₁ − μ₂ + 2 ≠ 0`, each checked over the integers *and*
/// mod `p` (an integer bound on `μ` does not bound the residue of `μᵢμⱼ`).
fn validate_j_plus_diag_mds(m: &[[u64; 3]; 3], p: &BigUint) -> Result<(), SpartanError> {
  let fail = |reason: String| SpartanError::InternalError {
    reason: format!("poseidon2 MDS validation: {reason}"),
  };
  for (i, row) in m.iter().enumerate() {
    for (j, v) in row.iter().enumerate() {
      if i != j && *v != 1 {
        return Err(fail("matrix is not of the form J + diag".to_string()));
      }
    }
  }
  let mu: [BigUint; 3] = core::array::from_fn(|i| BigUint::from(m[i][i]));
  let one = BigUint::one();
  for (i, mi) in mu.iter().enumerate() {
    if mi.is_zero() || (mi % p).is_zero() {
      return Err(fail(format!("mu_{i} is 0")));
    }
    if *mi == one || (mi % p) == (&one % p) {
      return Err(fail(format!("mu_{i} is 1")));
    }
  }
  for i in 0..3 {
    for j in (i + 1)..3 {
      let prod = &mu[i] * &mu[j];
      if prod == one || (&prod % p) == one {
        return Err(fail(format!("mu_{i} * mu_{j} = 1")));
      }
    }
  }
  // μ₀μ₁μ₂ + 2 ≠ μ₀ + μ₁ + μ₂, over ℤ and mod p.
  let lhs = &mu[0] * &mu[1] * &mu[2] + 2u32;
  let rhs = &mu[0] + &mu[1] + &mu[2];
  if lhs == rhs || (&lhs % p) == (&rhs % p) {
    return Err(fail("mu0*mu1*mu2 - mu0 - mu1 - mu2 + 2 = 0".to_string()));
  }
  Ok(())
}

/// Build the Poseidon2 parameters for one target field: derive the 80
/// BLAKE3 round constants (rejection-sampled, all asserted nonzero) and
/// validate both matrices — MDS over ℤ and mod `p` for `M_E` and `M_I`,
/// plus the subspace-trail Algorithms 1–3 for `M_I` — before returning.
pub fn build_params(field: Field) -> Result<Poseidon2Params, SpartanError> {
  let p = field.modulus();
  // α = 5 must be a valid S-box exponent: gcd(5, p − 1) = 1.
  let p_minus_1 = &p - BigUint::one();
  if !p_minus_1.gcd(&BigUint::from(ALPHA)).is_one() {
    return Err(SpartanError::InternalError {
      reason: format!("poseidon2: gcd(5, p-1) != 1 for {}", field.name()),
    });
  }
  validate_j_plus_diag_mds(&M_E, &p)?;
  validate_j_plus_diag_mds(&M_I, &p)?;
  validate_internal_matrix(&M_I, &p)?;
  let rc = derive_round_constants(&p)?;
  Ok(Poseidon2Params {
    modulus: p,
    m_e: M_E,
    m_i: M_I,
    rc,
  })
}

/// Build the fixed-order parameter set for all three field blocks,
/// invoking the full matrix validation for every block before any shape
/// construction. The ONLY constructor of [`Poseidon2ParamsSet`].
pub fn build_all_params() -> Result<Poseidon2ParamsSet, SpartanError> {
  Ok(Poseidon2ParamsSet {
    params: [
      build_params(FIELD_ORDER[0])?,
      build_params(FIELD_ORDER[1])?,
      build_params(FIELD_ORDER[2])?,
    ],
  })
}

/// Deterministic benchmark messages `m_1..m_H`: BLAKE3 XOF of
/// `"limber-poseidon2-v1/msg" ‖ j_be4`, first 32 bytes big-endian, masked
/// to the low 250 bits — canonical for all three fields (`< 2^250 < min p`).
/// The combined builder copies these `H` values into each block's distinct
/// message columns (`3H` slots total): a fixture choice for value-for-value
/// comparability, not a cross-field equality constraint.
pub fn build_inputs(hashes_per_field: usize) -> Result<Vec<BigUint>, SpartanError> {
  checked_hashes(hashes_per_field)?;
  let mask = (BigUint::one() << 250u32) - BigUint::one();
  let mut out = Vec::with_capacity(hashes_per_field);
  for j in 1..=hashes_per_field {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MSG_DOMAIN);
    hasher.update(&(j as u32).to_be_bytes());
    let mut buf = [0u8; 32];
    hasher.finalize_xof().fill(&mut buf);
    out.push(BigUint::from_bytes_be(&buf) & &mask);
  }
  Ok(out)
}

/// Reference Poseidon2 permutation on canonical field elements. Rejects a
/// state containing any lane `≥ p` rather than silently reducing it.
pub fn permute(
  params: &Poseidon2Params,
  state: [BigUint; 3],
) -> Result<[BigUint; 3], SpartanError> {
  let p = &params.modulus;
  for (i, lane) in state.iter().enumerate() {
    if lane >= p {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("poseidon2 permute: state lane {i} is not a canonical residue (>= p)"),
      });
    }
  }
  let apply = |m: &[[u64; 3]; 3], s: &[BigUint; 3]| -> [BigUint; 3] {
    core::array::from_fn(|i| {
      let mut acc = BigUint::zero();
      for j in 0..3 {
        acc += BigUint::from(m[i][j]) * &s[j];
      }
      acc % p
    })
  };
  let sbox = |x: &BigUint| -> BigUint {
    let x2 = (x * x) % p;
    let x4 = (&x2 * &x2) % p;
    (&x4 * x) % p
  };
  // Initial external layer, then each round: ARC, S-box, linear layer.
  let mut s = apply(&params.m_e, &state);
  for round in 1..=ROUNDS {
    let rc = &params.rc[round - 1];
    if is_full_round(round) {
      for lane in 0..T {
        s[lane] = (&s[lane] + &rc[lane]) % p;
        s[lane] = sbox(&s[lane]);
      }
      s = apply(&params.m_e, &s);
    } else {
      s[0] = (&s[0] + &rc[0]) % p;
      s[0] = sbox(&s[0]);
      s = apply(&params.m_i, &s);
    }
  }
  Ok(s)
}

/// Reference chain `h_i = P([h_{i-1}, m_i, 0])[0]` from the fixed IV, for
/// one field. Returns all of `h_1..h_H`; rejects a message `≥ p`.
pub fn expected_chain(
  params: &Poseidon2Params,
  messages: &[BigUint],
) -> Result<Vec<BigUint>, SpartanError> {
  checked_hashes(messages.len())?;
  let p = &params.modulus;
  let mut h = chain_iv();
  debug_assert!(h < *p, "chain IV must be canonical for every target field");
  let mut out = Vec::with_capacity(messages.len());
  for (j, m) in messages.iter().enumerate() {
    if m >= p {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "poseidon2 chain: message {} is not a canonical residue (>= p)",
          j + 1
        ),
      });
    }
    let s = permute(params, [h, m.clone(), BigUint::zero()])?;
    h = s[0].clone();
    out.push(h.clone());
  }
  Ok(out)
}

// ---------------------------------------------------------------------------
// The shared row schedule

/// Source of a linear-layer input lane.
#[derive(Clone, Copy, Debug)]
enum Src {
  /// The fixed public IV (each block's first permutation, lane 0), folded
  /// into the constant column of the shape.
  Iv,
  /// The constant zero (capacity lane of every initial layer).
  Zero,
  /// A witness column.
  Col(usize),
}

/// Where a row's output value lands.
#[derive(Clone, Copy, Debug)]
enum Out {
  /// A fresh witness column.
  Wit(usize),
  /// Public-IO slot `f` in [`FIELD_ORDER`] (block `f`'s final digest).
  Digest(usize),
}

/// One row of the circuit, as seen by the shared schedule walker.
enum RowKind<'a> {
  /// `(Σ_j coeffs[j]·srcs[j] + rc) · 1 ≡ out (mod p_f)`. `q_bound` is the
  /// exact quotient bound: `LC < (Σ_{srcs[j] ≠ 0} coeffs[j] + [rc]) · p_f`.
  Reduce {
    /// Matrix-row coefficients for the three lanes.
    coeffs: &'a [u64; 3],
    /// Input sources for the three lanes.
    srcs: [Src; 3],
    /// Round constant folded into this row, if any.
    rc: Option<&'a BigUint>,
    /// Inclusive upper bound on the quotient.
    q_bound: u64,
  },
  /// `a · b ≡ out (mod p_f)` — one of the three S-box rows.
  Sbox {
    /// Column of the left operand.
    a: usize,
    /// Column of the right operand.
    b: usize,
  },
}

/// One visited row: its global index, block, kind, and output target.
struct RowInfo<'a> {
  /// Global row index (`0..1299H`).
  row: usize,
  /// Row kind and operands.
  kind: RowKind<'a>,
  /// Output target.
  out: Out,
}

/// Walk every real row of one field block in row order, invoking `visit`
/// once per row. `row_base`/`col_base` are the block's global offsets; the
/// block's messages occupy columns `col_base..col_base + H` and its final
/// terminal reduce row targets [`Out::Digest`]`(block)`.
fn walk_block_rows<F>(
  params: &Poseidon2Params,
  hashes: usize,
  block: usize,
  row_base: usize,
  col_base: usize,
  mut visit: F,
) -> Result<(), SpartanError>
where
  F: FnMut(RowInfo<'_>) -> Result<(), SpartanError>,
{
  let (block_rows, block_cols, ..) = checked_dims(hashes)?;
  let mut row = row_base;
  let mut next_col = col_base + hashes; // block columns start with the messages
  let mut prev_out = Src::Iv;

  for perm in 0..hashes {
    let mut state: [Src; 3] = [prev_out, Src::Col(col_base + perm), Src::Zero];
    for layer in 0..=ROUNDS {
      let m: &[[u64; 3]; 3] = if layer_is_external(layer) {
        &params.m_e
      } else {
        &params.m_i
      };
      let lanes = if layer == ROUNDS { 1 } else { T };
      let mut layer_out = [usize::MAX; 3];
      for (lane, m_row) in m.iter().enumerate().take(lanes) {
        // Layer l < 64 feeds round l + 1; the terminal layer has no rc.
        let rc: Option<&BigUint> = if layer == ROUNDS {
          None
        } else {
          let round = layer + 1;
          if is_full_round(round) {
            Some(&params.rc[round - 1][lane])
          } else if lane == 0 {
            Some(&params.rc[round - 1][0])
          } else {
            None
          }
        };
        let q_bound = state
          .iter()
          .zip(m_row.iter())
          .map(|(s, c)| if matches!(s, Src::Zero) { 0 } else { *c })
          .sum::<u64>()
          + u64::from(rc.is_some())
          - 1;
        let is_last_row = perm == hashes - 1 && layer == ROUNDS;
        let out = if is_last_row {
          Out::Digest(block)
        } else {
          let c = next_col;
          next_col += 1;
          Out::Wit(c)
        };
        visit(RowInfo {
          row,
          kind: RowKind::Reduce {
            coeffs: m_row,
            srcs: state,
            rc,
            q_bound,
          },
          out,
        })?;
        if let Out::Wit(c) = out {
          layer_out[lane] = c;
        }
        row += 1;
      }
      if layer < ROUNDS {
        let round = layer + 1;
        let sbox_lanes = if is_full_round(round) { T } else { 1 };
        let mut new_state: [Src; 3] = core::array::from_fn(|l| Src::Col(layer_out[l]));
        for (lane, slot) in new_state.iter_mut().enumerate().take(sbox_lanes) {
          let y = layer_out[lane];
          let x2 = next_col;
          next_col += 1;
          visit(RowInfo {
            row,
            kind: RowKind::Sbox { a: y, b: y },
            out: Out::Wit(x2),
          })?;
          row += 1;
          let x4 = next_col;
          next_col += 1;
          visit(RowInfo {
            row,
            kind: RowKind::Sbox { a: x2, b: x2 },
            out: Out::Wit(x4),
          })?;
          row += 1;
          let x5 = next_col;
          next_col += 1;
          visit(RowInfo {
            row,
            kind: RowKind::Sbox { a: x4, b: y },
            out: Out::Wit(x5),
          })?;
          row += 1;
          *slot = Src::Col(x5);
        }
        state = new_state;
      } else if perm < hashes - 1 {
        // Each block restarts its own chain: the terminal lane-0 output
        // feeds the next permutation of THIS block only. A digest never
        // feeds the next field's chain.
        prev_out = Src::Col(layer_out[0]);
      }
    }
  }
  // Unconditional per-block row/column bookkeeping (§3).
  assert_eq!(
    row,
    row_base + block_rows,
    "poseidon2 block schedule row count mismatch"
  );
  assert_eq!(
    next_col,
    col_base + block_cols,
    "poseidon2 block schedule column count mismatch"
  );
  Ok(())
}

/// Walk every real row of the combined `3H`-compression circuit — the
/// three field blocks in [`FIELD_ORDER`], each in row order over its
/// disjoint row/column ranges. The single source of truth for the row and
/// column schedule, shared by [`build_shape`], [`compute_advice`], and
/// [`validate_advice`].
fn walk_rows<F>(set: &Poseidon2ParamsSet, hashes: usize, mut visit: F) -> Result<(), SpartanError>
where
  F: FnMut(&Poseidon2Params, RowInfo<'_>) -> Result<(), SpartanError>,
{
  let (block_rows, block_cols, ..) = checked_dims(hashes)?;
  for (f, params) in set.params.iter().enumerate() {
    walk_block_rows(params, hashes, f, f * block_rows, f * block_cols, |info| {
      visit(params, info)
    })?;
  }
  Ok(())
}

/// Build the single padded mixed-modulus IntMod-R1CS shape and its
/// [`Layout`] for `H` chained compressions per field: the three real
/// (unpadded) field blocks concatenated in [`FIELD_ORDER`], then padded
/// once (concatenating three already-padded shapes would force a wasteful
/// doubled domain). Performs unconditional bookkeeping asserts (row/column
/// schedule, non-zero counts, COO uniqueness) and exact padding (three
/// contiguous modulus segments, `mods[real_rows..] == 2`, no matrix entry
/// at a padding row).
pub fn build_shape<M: ModEngine>(
  set: &Poseidon2ParamsSet,
  hashes_per_field: usize,
) -> Result<(IntModR1CSShapeModp<M>, Layout), SpartanError> {
  let (block_rows, block_cols, real_rows, real_cols, num_cons, num_vars, log_n) =
    checked_dims(hashes_per_field)?;
  let total_hashes =
    hashes_per_field
      .checked_mul(NUM_FIELDS)
      .ok_or_else(|| SpartanError::InvalidInputLength {
        reason: "poseidon2: 3H overflows usize".to_string(),
      })?;
  let const_col = num_vars;
  let one = BigUint::one();
  let iv = chain_iv();

  let mut a_entries: Vec<(usize, usize, BigUint)> = Vec::new();
  let mut b_entries: Vec<(usize, usize, BigUint)> = Vec::with_capacity(real_rows);
  let mut c_entries: Vec<(usize, usize, BigUint)> = Vec::with_capacity(real_rows);
  let mut mods: Vec<BigUint> = Vec::with_capacity(num_cons);

  walk_rows(set, hashes_per_field, |params, info| {
    let out_col = match info.out {
      Out::Wit(c) => c,
      // z = (w, 1, x): public-IO slot f sits at column num_vars + 1 + f.
      Out::Digest(f) => num_vars + 1 + f,
    };
    match info.kind {
      RowKind::Reduce {
        coeffs, srcs, rc, ..
      } => {
        let mut const_acc = BigUint::zero();
        for (src, coeff) in srcs.iter().zip(coeffs.iter()) {
          match src {
            Src::Zero => {}
            Src::Iv => const_acc += BigUint::from(*coeff) * &iv,
            Src::Col(c) => a_entries.push((info.row, *c, BigUint::from(*coeff))),
          }
        }
        if let Some(rc) = rc {
          const_acc += rc;
        }
        if !const_acc.is_zero() {
          a_entries.push((info.row, const_col, const_acc));
        }
        b_entries.push((info.row, const_col, one.clone()));
        c_entries.push((info.row, out_col, one.clone()));
      }
      RowKind::Sbox { a, b } => {
        a_entries.push((info.row, a, one.clone()));
        b_entries.push((info.row, b, one.clone()));
        c_entries.push((info.row, out_col, one.clone()));
      }
    }
    mods.push(params.modulus.clone());
    Ok(())
  })?;

  // Unconditional structural checks: non-zero counts (§3: A = 2688H − 9,
  // B = C = 1299H) and COO uniqueness. The A count assumes every round
  // constant is nonzero, which `build_params` asserts at generation.
  let overflow = || SpartanError::InvalidInputLength {
    reason: format!("poseidon2: NNZ arithmetic overflows for H = {hashes_per_field}"),
  };
  let expected_a = 2688usize
    .checked_mul(hashes_per_field)
    .and_then(|v| v.checked_sub(9))
    .ok_or_else(overflow)?;
  assert_eq!(a_entries.len(), expected_a, "A non-zero count mismatch");
  assert_eq!(b_entries.len(), real_rows, "B non-zero count mismatch");
  assert_eq!(c_entries.len(), real_rows, "C non-zero count mismatch");
  for (name, entries) in [("A", &a_entries), ("B", &b_entries), ("C", &c_entries)] {
    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    for (r, c, _) in entries {
      assert!(
        seen.insert((*r, *c)),
        "duplicate COO entry in {name} at ({r}, {c})"
      );
    }
  }
  assert!(
    a_entries
      .iter()
      .chain(&b_entries)
      .chain(&c_entries)
      .all(|(r, _, _)| *r < real_rows),
    "matrix entry at a padding row"
  );

  // Exact padding: padding rows use the nondegenerate modulus 2 (m = 1 is
  // degenerate, m = 0 is an exact-integer row). The three real modulus
  // segments precede it, contiguous and in FIELD_ORDER.
  mods.resize(num_cons, BigUint::from(2u32));

  let shape = IntModR1CSShapeModp::<M>::new(
    num_cons, num_vars, NUM_FIELDS, a_entries, b_entries, c_entries, mods,
  )?;
  let shape_digest = shape.digest();
  let layout = Layout {
    hashes_per_field,
    total_hashes,
    block_rows,
    block_cols,
    real_rows,
    real_cols,
    num_cons,
    num_vars,
    log_n,
    moduli: set.ordered_moduli(),
    shape_digest,
  };
  Ok((shape, layout))
}

/// Compute the combined witness `w`, quotients `q`, and the three ordered
/// public digests for the given per-field messages by walking the full row
/// schedule. The `H` message values are copied into each block's distinct
/// message columns. Rejects a `messages` slice whose length is not
/// `layout.hashes_per_field()` and any message `≥` the modulus of a block
/// in which it is used.
pub fn compute_advice(
  set: &Poseidon2ParamsSet,
  layout: &Layout,
  messages: &[BigUint],
) -> Result<(Vec<BigUint>, Vec<BigUint>, [BigUint; 3]), SpartanError> {
  if messages.len() != layout.hashes_per_field {
    return Err(SpartanError::InvalidInputLength {
      reason: format!(
        "poseidon2 advice: expected {} messages per field, got {}",
        layout.hashes_per_field,
        messages.len()
      ),
    });
  }
  let iv = chain_iv();
  let mut w = vec![BigUint::zero(); layout.num_vars];
  let mut q = vec![BigUint::zero(); layout.num_cons];
  for (f, params) in set.params.iter().enumerate() {
    let p = &params.modulus;
    let col_base = f * layout.block_cols;
    for (j, m) in messages.iter().enumerate() {
      if m >= p {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "poseidon2 advice: message {} is not a canonical residue for block {}",
            j + 1,
            FIELD_ORDER[f].name()
          ),
        });
      }
      w[col_base + j] = m.clone();
    }
  }
  let mut digests: [Option<BigUint>; NUM_FIELDS] = [const { None }; NUM_FIELDS];
  walk_rows(set, layout.hashes_per_field, |params, info| {
    let p = &params.modulus;
    let (value, quotient) = match info.kind {
      RowKind::Reduce {
        coeffs, srcs, rc, ..
      } => {
        let mut lc = BigUint::zero();
        for (src, coeff) in srcs.iter().zip(coeffs.iter()) {
          match src {
            Src::Zero => {}
            Src::Iv => lc += BigUint::from(*coeff) * &iv,
            Src::Col(c) => lc += BigUint::from(*coeff) * &w[*c],
          }
        }
        if let Some(rc) = rc {
          lc += rc;
        }
        let (qq, y) = lc.div_rem(p);
        (y, qq)
      }
      RowKind::Sbox { a, b } => {
        let prod = &w[a] * &w[b];
        let (qq, y) = prod.div_rem(p);
        (y, qq)
      }
    };
    q[info.row] = quotient;
    match info.out {
      Out::Wit(c) => w[c] = value,
      Out::Digest(f) => digests[f] = Some(value),
    }
    Ok(())
  })?;
  let digests: [BigUint; 3] = core::array::from_fn(|f| {
    digests[f]
      .take()
      .expect("schedule produces one digest per block")
  });
  Ok((w, q, digests))
}

/// Full witness/quotient bound scan (benchmark-preflight and test work,
/// deliberately separate from [`compute_advice`]): padding zeros, per-block
/// canonical witness values against the OWNING block's modulus (never a
/// global min/max modulus), exact per-row quotient bounds (`q ≤ 4` on all
/// reduce rows except one `q ≤ 5` row per permutation, `q < p_f` on S-box
/// rows), and three canonical digests.
pub fn validate_advice(
  set: &Poseidon2ParamsSet,
  layout: &Layout,
  w: &[BigUint],
  q: &[BigUint],
  digests: &[BigUint; 3],
) -> Result<(), SpartanError> {
  let fail = |reason: String| SpartanError::InternalError {
    reason: format!("poseidon2 validate_advice: {reason}"),
  };
  if w.len() != layout.num_vars || q.len() != layout.num_cons {
    return Err(fail(format!(
      "length mismatch: w {} (want {}), q {} (want {})",
      w.len(),
      layout.num_vars,
      q.len(),
      layout.num_cons
    )));
  }
  if !w[layout.real_cols..].iter().all(BigUint::is_zero) {
    return Err(fail("witness padding is not all-zero".to_string()));
  }
  if !q[layout.real_rows..].iter().all(BigUint::is_zero) {
    return Err(fail("quotient padding is not all-zero".to_string()));
  }
  // Every real committed value is canonical for its OWNING block
  // (< p_f < 2^log_t_f = 2^256).
  for (f, params) in set.params.iter().enumerate() {
    let p = &params.modulus;
    let col_base = f * layout.block_cols;
    for (off, v) in w[col_base..col_base + layout.block_cols].iter().enumerate() {
      if v >= p {
        return Err(fail(format!(
          "witness column {} is not canonical for its block (>= p_{})",
          col_base + off,
          FIELD_ORDER[f].name()
        )));
      }
    }
    if &digests[f] >= p {
      return Err(fail(format!(
        "digest {} is not canonical (>= p)",
        FIELD_ORDER[f].name()
      )));
    }
  }
  walk_rows(set, layout.hashes_per_field, |params, info| {
    match info.kind {
      RowKind::Reduce { q_bound, .. } => {
        if q[info.row] > BigUint::from(q_bound) {
          return Err(fail(format!(
            "reduce-row quotient at row {} exceeds its bound {}",
            info.row, q_bound
          )));
        }
      }
      RowKind::Sbox { .. } => {
        if q[info.row] >= params.modulus {
          return Err(fail(format!(
            "S-box quotient at row {} is >= its block modulus",
            info.row
          )));
        }
      }
    }
    Ok(())
  })
}

// ---------------------------------------------------------------------------
// Verification wrapper: ordered three-digest canonicality bound to the key

/// The shared canonicality predicate over trusted ordered moduli. Private:
/// there is no public helper that accepts caller-supplied moduli —
/// [`check_canonical_io`] obtains them from the private-constructor
/// parameter set, [`verify_poseidon_chain`] from the verifier key.
fn check_canonical_io_inner(
  x: &[BigUint],
  moduli: &[BigUint; NUM_FIELDS],
) -> Result<(), SpartanError> {
  if x.len() != NUM_FIELDS {
    return Err(SpartanError::InvalidInputLength {
      reason: format!(
        "poseidon2: expected exactly three public IO values, got {}",
        x.len()
      ),
    });
  }
  for (f, (xi, p)) in x.iter().zip(moduli.iter()).enumerate() {
    if xi >= p {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "poseidon2: public digest {} ({}) is not a canonical residue (>= p)",
          f,
          FIELD_ORDER[f].name()
        ),
      });
    }
  }
  Ok(())
}

/// Canonicality policy alone — no proof argument, directly testable.
/// Requires EXACTLY THREE public values in [`FIELD_ORDER`], each a
/// canonical residue of its own block's modulus. The moduli come from the
/// private-constructor parameter set; no caller-supplied modulus is
/// accepted.
pub fn check_canonical_io(x: &[BigUint], set: &Poseidon2ParamsSet) -> Result<(), SpartanError> {
  check_canonical_io_inner(x, &set.ordered_moduli())
}

/// A Poseidon-chain verifier key: the generic SNARK verifier key plus the
/// three ordered target-field moduli the shape blocks were built for,
/// bound together at construction so they cannot disagree. The stored
/// moduli feed the per-slot canonical-digest check of
/// [`verify_poseidon_chain`]; caller-supplied moduli are never trusted.
pub struct PoseidonVerifierKey<M: ModEngine> {
  /// The generic SNARK verifier key.
  inner: IntModSpartanModpVerifierKey<M>,
  /// The three block moduli, in [`FIELD_ORDER`].
  moduli: [BigUint; NUM_FIELDS],
}

impl<M: ModEngine> PoseidonVerifierKey<M> {
  /// The ONLY constructor. `set`/`layout` have private fields and come
  /// from [`build_all_params`]/[`build_shape`]; no caller-supplied
  /// `BigUint` is trusted. Unconditionally checks that:
  ///
  /// 1. the shape has exactly three public IO values and its padded
  ///    dimensions equal the layout's;
  /// 2. the key digest equals the layout's shape digest and the
  ///    fixed-order parameter-set modulus equals the layout's at every
  ///    [`FIELD_ORDER`] index;
  /// 3. each recorded row block has exactly `433H` rows carrying only its
  ///    matching modulus, the three blocks are contiguous and cover
  ///    `mods[..real_rows]`, and every padding modulus is 2;
  /// 4. the checked identities `real_rows = 1299H` and
  ///    `real_cols = 1302H − 3` hold, together with each block's
  ///    row/column boundaries and its distinct ordered terminal public-IO
  ///    column.
  pub fn new(
    inner: IntModSpartanModpVerifierKey<M>,
    set: &Poseidon2ParamsSet,
    layout: &Layout,
  ) -> Result<Self, SpartanError> {
    let fail = |reason: String| SpartanError::ProofVerifyError {
      reason: format!("PoseidonVerifierKey: {reason}"),
    };
    let shape = &inner.shape;
    // 1. Exactly three IO values; padded dimensions match.
    if shape.num_io != NUM_FIELDS {
      return Err(fail(format!(
        "shape has num_io = {}, want {NUM_FIELDS}",
        shape.num_io
      )));
    }
    if shape.num_cons != layout.num_cons || shape.num_vars != layout.num_vars {
      return Err(fail("shape dimensions do not match the layout".to_string()));
    }
    // 2. Digest and per-index modulus binding.
    if inner.digest() != layout.shape_digest {
      return Err(fail("shape digest does not match the layout".to_string()));
    }
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      if set.get(*field).modulus() != &layout.moduli[f] {
        return Err(fail(format!(
          "parameter-set modulus for {} does not match the layout",
          field.name()
        )));
      }
    }
    // 3. Three contiguous per-block modulus segments covering the real
    // rows; padding moduli are 2.
    if layout.real_rows == 0 || layout.real_rows >= shape.mods.len() {
      return Err(fail("real_rows outside (0, num_cons)".to_string()));
    }
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let range = layout.block_row_range(f);
      if !shape.mods[range].iter().all(|m| m == &layout.moduli[f]) {
        return Err(fail(format!(
          "row block {f} carries a modulus other than {}",
          field.name()
        )));
      }
    }
    let two = BigUint::from(2u32);
    if !shape.mods[layout.real_rows..].iter().all(|m| m == &two) {
      return Err(fail("a padding-row modulus is not 2".to_string()));
    }
    // 4. Row/column identities, block boundaries, and distinct ordered
    // terminal public-IO columns.
    let (block_rows, block_cols, real_rows, real_cols, ..) = checked_dims(layout.hashes_per_field)?;
    if layout.block_rows != block_rows
      || layout.block_cols != block_cols
      || layout.real_rows != real_rows
      || layout.real_cols != real_cols
    {
      return Err(fail("layout row/column identities do not hold".to_string()));
    }
    for f in 0..NUM_FIELDS {
      let terminal_row = (f + 1) * block_rows - 1;
      let io_col = shape.num_vars + 1 + f;
      let ok = shape
        .C
        .iter()
        .any(|(r, c, _)| *r == terminal_row && *c == io_col);
      if !ok {
        return Err(fail(format!(
          "block {f}'s terminal row does not target public-IO slot {f}"
        )));
      }
    }
    Ok(Self {
      inner,
      moduli: layout.moduli.clone(),
    })
  }

  /// The three ordered target-field moduli this key's shape blocks were
  /// built for.
  pub fn moduli(&self) -> &[BigUint; 3] {
    &self.moduli
  }
}

/// Verify a combined Poseidon-chain proof: the ordered three-digest
/// canonicality policy (against the key-bound moduli) followed by the
/// generic SNARK verification. The benchmark, tests, and documented
/// Poseidon2 API must verify through this function; calling the generic
/// `IntModSpartanModpSNARK::verify` directly forfeits the three-digest
/// canonicality guarantee.
pub fn verify_poseidon_chain<M>(
  vk: &PoseidonVerifierKey<M>,
  instance: &IntModR1CSInstanceModp<M>,
  proof: &IntModSpartanModpSNARK<M>,
) -> Result<(), SpartanError>
where
  M: ModEngine<TE = Keccak256Transcript<M>>,
{
  check_canonical_io_inner(&instance.x, &vk.moduli)?;
  proof.verify(&vk.inner, instance)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256DynPrimeEngine;

  type ME = T256DynPrimeEngine;

  /// Fixed integer oracle for the specialized Algorithms 2–3 determinants
  /// `D_r = det[e₀, M^r e₀, M^(2r) e₀]`, `r = 1..=12`, of the selected
  /// `M_I`. `validate_internal_matrix` recomputes each `D_r` mod the
  /// target prime; this list pins the integer values.
  const D_ORACLE: [u64; 12] = [
    1,
    84,
    3683,
    133056,
    4453429,
    143857980,
    4562087447,
    143190609408,
    4467253670377,
    138861853051236,
    4306780344808523,
    133388643001078080,
  ];

  #[test]
  fn dims_match_the_plan() {
    // H = 10 per field: 12,990 rows, 13,017 real witness columns, both
    // padded to 2^14.
    let (br, bc, rr, rc, nc, nv, log_n) = checked_dims(10).unwrap();
    assert_eq!(
      (br, bc, rr, rc, nc, nv, log_n),
      (4330, 4339, 12990, 13017, 16384, 16384, 14)
    );
    // H = 1 per field (CI proof size): 1,299 rows -> 2^11.
    let (_, _, rr, rc, nc, nv, log_n) = checked_dims(1).unwrap();
    assert_eq!((rr, rc, nc, nv, log_n), (1299, 1299, 2048, 2048, 11));
  }

  #[test]
  fn field_order_is_the_literal_sequence() {
    // FIELD_ORDER is the literal BN254 / BLS12-381 / secp256k1 sequence,
    // and the private set maps each enum to its matching modulus.
    assert_eq!(
      FIELD_ORDER,
      [Field::Bn254Fr, Field::Bls12381Fr, Field::Secp256k1Fr]
    );
    let set = build_all_params().unwrap();
    for field in FIELD_ORDER {
      assert_eq!(set.get(field).modulus(), &field.modulus());
    }
    let moduli = set.ordered_moduli();
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      assert_eq!(moduli[f], field.modulus());
    }
  }

  #[test]
  fn hash_count_bounds_are_enforced() {
    assert!(checked_hashes(0).is_err());
    assert!(checked_hashes(1).is_ok());
    assert!(checked_hashes(u32::MAX as usize).is_ok());
    assert!(checked_hashes(u32::MAX as usize + 1).is_err());
    assert!(build_inputs(0).is_err());
    assert!(build_inputs(u32::MAX as usize + 1).is_err());
    let set = build_all_params().unwrap();
    assert!(build_shape::<ME>(&set, 0).is_err());
    assert!(expected_chain(set.get(Field::Bn254Fr), &[]).is_err());
    // Combined ×3 arithmetic is checked, not wrapped.
    assert!(checked_dims(usize::MAX / 500).is_err());
  }

  #[test]
  fn padding_boundary_is_never_exact() {
    // 1299H and 1302H − 3 are multiples of 3 greater than 1, hence never
    // powers of two: every combined shape has padding rows and columns.
    for h in [1usize, 2, 3, 9, 10, 11, 255, 256] {
      let (_, _, rr, rc, nc, nv, _) = checked_dims(h).unwrap();
      assert_eq!(rr % 3, 0);
      assert_eq!(rc % 3, 0);
      assert!(!rr.is_power_of_two());
      assert!(!rc.is_power_of_two());
      assert!(rr < nc, "H={h} must leave padding rows");
      assert!(rc < nv, "H={h} must leave padding columns");
    }
  }

  /// Synthetic-dimension check for the exact power-of-two case the real
  /// schedule can never hit: a hypothetical real row count that IS a power
  /// of two pads to itself, which `PoseidonVerifierKey::new`'s
  /// `real_rows < mods.len()` predicate would reject (no padding rows).
  #[test]
  fn synthetic_power_of_two_dimension_pads_to_itself() {
    let real = 2048usize;
    assert_eq!(real.next_power_of_two(), real);
  }

  #[test]
  fn subspace_oracle_matches_the_pinned_integers() {
    // Recompute D_r = det[e0, M^r e0, M^(2r) e0] over the integers (signed)
    // and compare with the fixed oracle list.
    let m = [[2i128, 1, 1], [1, 2, 1], [1, 1, 3]];
    let mat_mul = |a: &[[i128; 3]; 3], b: &[[i128; 3]; 3]| -> [[i128; 3]; 3] {
      core::array::from_fn(|i| core::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    };
    let mut m_r = m;
    for (r, expected) in D_ORACLE.iter().enumerate() {
      let m_2r = mat_mul(&m_r, &m_r);
      // det[e0, M^r e0, M^2r e0] with e0 = (1,0,0): expands to the 2x2
      // minor of rows 1..2 of columns (M^r e0, M^2r e0).
      let d = m_r[1][0] * m_2r[2][0] - m_r[2][0] * m_2r[1][0];
      assert_eq!(d, *expected as i128, "D_{} mismatch", r + 1);
      m_r = mat_mul(&m_r, &m);
    }
  }

  #[test]
  fn official_internal_matrix_passes_all_fields() {
    for field in FIELD_ORDER {
      let p = field.modulus();
      validate_internal_matrix(&M_I, &p).unwrap();
      validate_j_plus_diag_mds(&M_I, &p).unwrap();
      validate_j_plus_diag_mds(&M_E, &p).unwrap();
    }
  }

  #[test]
  fn j_plus_i_fails_the_subspace_checks() {
    // J + I fixes span((0, a, −a)) across all partial rounds — the exact
    // subspace trail §5.3 exists to prevent. It must fail for every field
    // even though it IS MDS.
    let j_plus_i = [[2u64, 1, 1], [1, 2, 1], [1, 1, 2]];
    for field in FIELD_ORDER {
      let p = field.modulus();
      validate_j_plus_diag_mds(&j_plus_i, &p).unwrap();
      assert!(validate_internal_matrix(&j_plus_i, &p).is_err());
    }
  }

  #[test]
  fn internal_matrix_witnesses_are_pinned() {
    // The plan's pinned witnesses for the selected M_I, as p-residues.
    let p = Field::Bn254Fr.modulus();
    let m = mat_mod_p(&M_I, &p);
    let e1 = [BigUint::zero(), BigUint::one(), BigUint::zero()];
    let e2 = [BigUint::zero(), BigUint::zero(), BigUint::one()];
    let to = |v: [i64; 3]| -> [BigUint; 3] {
      core::array::from_fn(|i| {
        if v[i] >= 0 {
          BigUint::from(v[i] as u64)
        } else {
          &p - BigUint::from((-v[i]) as u64)
        }
      })
    };
    assert_eq!(mat_vec(&m, &e1, &p), to([1, 2, 1]));
    assert_eq!(mat_vec(&m, &e2, &p), to([1, 1, 3]));
    let v = to([0, 1, -1]);
    assert_eq!(mat_vec(&m, &v, &p), to([0, 1, -2]));
    let m2 = mat_mul(&m, &m, &p);
    assert_eq!(mat_vec(&m2, &v, &p), to([-1, 0, -5]));
  }

  #[test]
  fn params_build_for_all_fields() {
    for field in FIELD_ORDER {
      let params = build_params(field).unwrap();
      assert_eq!(params.rc.len(), ROUNDS);
      let mut total = 0usize;
      for (r, per_round) in params.rc.iter().enumerate() {
        let want = if is_full_round(r + 1) { T } else { 1 };
        assert_eq!(per_round.len(), want, "round {} lane count", r + 1);
        for c in per_round {
          assert!(!c.is_zero(), "zero round constant");
          assert!(c < params.modulus());
        }
        total += per_round.len();
      }
      assert_eq!(total, 80);
      assert!(chain_iv() < *params.modulus());
    }
  }

  #[test]
  fn inputs_are_canonical_for_all_fields() {
    let msgs = build_inputs(10).unwrap();
    assert_eq!(msgs.len(), 10);
    let bound = BigUint::one() << 250u32;
    for m in &msgs {
      assert!(*m < bound);
    }
    // Deterministic: same call, same values.
    assert_eq!(msgs, build_inputs(10).unwrap());
  }

  #[test]
  fn permute_rejects_noncanonical_lanes() {
    let params = build_params(Field::Bn254Fr).unwrap();
    let p = params.modulus().clone();
    assert!(permute(&params, [p.clone(), BigUint::zero(), BigUint::zero()]).is_err());
    assert!(permute(&params, [BigUint::zero(), BigUint::zero(), BigUint::zero()]).is_ok());
    assert!(expected_chain(&params, std::slice::from_ref(&p)).is_err());
  }

  #[test]
  fn advice_matches_reference_chain_in_every_block() {
    // Structure check at H = 2: the terminal lane-0 columns materialized
    // in EACH block equal that field's reference chain states, and each
    // digest equals its field's last state. (The proof round-trip lives
    // in tests/poseidon_modp.rs.)
    let set = build_all_params().unwrap();
    let h = 2usize;
    let (_shape, layout) = build_shape::<ME>(&set, h).unwrap();
    let messages = build_inputs(h).unwrap();
    let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();
    validate_advice(&set, &layout, &w, &q, &digests).unwrap();
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let chain = expected_chain(set.get(*field), &messages).unwrap();
      let col_base = f * layout.block_cols;
      for (i, expected) in chain.iter().enumerate().take(h - 1) {
        // Terminal reduce row of the block's permutation i: local row
        // 433·(i+1) − 1; its output column is col_base + H + local_row.
        let col = col_base + h + ROWS_PER_PERM * (i + 1) - 1;
        assert_eq!(&w[col], expected, "chain state {} in block {f}", i + 1);
      }
      assert_eq!(&digests[f], chain.last().unwrap(), "digest of block {f}");
    }
  }

  #[test]
  fn advice_rejects_bad_inputs() {
    let set = build_all_params().unwrap();
    let (_shape, layout) = build_shape::<ME>(&set, 2).unwrap();
    // Wrong message count.
    assert!(compute_advice(&set, &layout, &build_inputs(3).unwrap()).is_err());
    // A message canonical for secp256k1 but not for the BN254 block that
    // also uses it: rejected against the owning block's modulus.
    let mut msgs = build_inputs(2).unwrap();
    msgs[1] = set.get(Field::Bn254Fr).modulus().clone();
    assert!(msgs[1] < *set.get(Field::Secp256k1Fr).modulus());
    assert!(compute_advice(&set, &layout, &msgs).is_err());
  }

  #[test]
  fn validate_advice_rejects_tampering() {
    let set = build_all_params().unwrap();
    let (_shape, layout) = build_shape::<ME>(&set, 1).unwrap();
    let messages = build_inputs(1).unwrap();
    let (w, q, digests) = compute_advice(&set, &layout, &messages).unwrap();

    // Padding tamper.
    let mut w_bad = w.clone();
    w_bad[layout.real_cols()] = BigUint::one();
    assert!(validate_advice(&set, &layout, &w_bad, &q, &digests).is_err());
    let mut q_bad = q.clone();
    q_bad[layout.real_rows()] = BigUint::one();
    assert!(validate_advice(&set, &layout, &w, &q_bad, &digests).is_err());

    // A value canonical for a LARGER field placed in a smaller field's
    // block: caught against the owning block's modulus (never a global
    // min/max). Column 0 belongs to the BN254 block.
    let mut w_bad = w.clone();
    w_bad[0] = set.get(Field::Bn254Fr).modulus().clone();
    assert!(w_bad[0] < *set.get(Field::Secp256k1Fr).modulus());
    assert!(validate_advice(&set, &layout, &w_bad, &q, &digests).is_err());

    // Reduce-row quotient above its bound (row 0 is a reduce row).
    let mut q_bad = q.clone();
    q_bad[0] = BigUint::from(6u32);
    assert!(validate_advice(&set, &layout, &w, &q_bad, &digests).is_err());

    // A noncanonical digest at each ordered slot.
    for f in 0..3 {
      let mut d_bad = digests.clone();
      d_bad[f] = set.params[f].modulus().clone();
      assert!(validate_advice(&set, &layout, &w, &q, &d_bad).is_err());
    }

    // The untampered advice passes.
    validate_advice(&set, &layout, &w, &q, &digests).unwrap();
  }

  #[test]
  fn quotient_bound_distribution_matches_the_plan() {
    // Per permutation: 192 reduce rows at q_bound ≤ 4, exactly one at 5
    // (lane 2 of the M_I layer feeding the first terminal full round) —
    // so the combined circuit has 3H rows at bound 5 (30 at the default).
    let set = build_all_params().unwrap();
    let h = 2usize;
    let mut bound5 = 0usize;
    let mut reduce = 0usize;
    let mut sbox = 0usize;
    walk_rows(&set, h, |_params, info| {
      match info.kind {
        RowKind::Reduce { q_bound, .. } => {
          reduce += 1;
          assert!(q_bound <= 5);
          if q_bound == 5 {
            bound5 += 1;
          }
        }
        RowKind::Sbox { .. } => sbox += 1,
      }
      Ok(())
    })
    .unwrap();
    assert_eq!(reduce, 3 * 193 * h);
    assert_eq!(sbox, 3 * 240 * h);
    assert_eq!(bound5, 3 * h, "exactly one q ≤ 5 row per permutation");
  }

  #[test]
  fn check_canonical_io_policy() {
    let set = build_all_params().unwrap();
    let ok: Vec<BigUint> = FIELD_ORDER
      .iter()
      .map(|f| set.get(*f).modulus() - BigUint::one())
      .collect();
    // p_f − 1 accepted at every ordered slot.
    check_canonical_io(&ok, &set).unwrap();
    // Lengths other than 3 rejected.
    assert!(check_canonical_io(&[], &set).is_err());
    assert!(check_canonical_io(&ok[..2], &set).is_err());
    let four: Vec<BigUint> = ok.iter().cloned().chain([BigUint::zero()]).collect();
    assert!(check_canonical_io(&four, &set).is_err());
    // p_f rejected independently at every ordered slot.
    for f in 0..3 {
      let mut bad = ok.clone();
      bad[f] = set.params[f].modulus().clone();
      assert!(check_canonical_io(&bad, &set).is_err(), "slot {f}");
    }
    // The BLS modulus is canonical for secp256k1 but not at the BN254
    // slot: order matters.
    let mut swapped = ok.clone();
    swapped[0] = set.get(Field::Bls12381Fr).modulus().clone();
    assert!(check_canonical_io(&swapped, &set).is_err());
  }

  #[test]
  fn shape_padding_structure_is_exact() {
    // Crate-internal: A/B/C/mods are pub(crate), so the structural padding
    // checks live here rather than in an integration test.
    let set = build_all_params().unwrap();
    let h = 2usize;
    let (shape, layout) = build_shape::<ME>(&set, h).unwrap();
    assert_eq!(shape.mods.len(), layout.num_cons());
    // Exact three modulus segments in FIELD_ORDER, then modulus-2 padding.
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let range = layout.block_row_range(f);
      assert!(
        shape.mods[range].iter().all(|m| m == &field.modulus()),
        "block {f} modulus segment"
      );
    }
    let two = BigUint::from(2u32);
    assert!(shape.mods[layout.real_rows()..].iter().all(|m| m == &two));
    for entries in [&shape.A, &shape.B, &shape.C] {
      assert!(entries.iter().all(|(r, _, _)| *r < layout.real_rows()));
    }
    // No witness-column reference crosses a recorded field-block boundary
    // (the constant and public-IO columns are the only shared ones).
    for entries in [&shape.A, &shape.B, &shape.C] {
      for (r, c, _) in entries.iter() {
        if *c >= layout.num_vars() {
          continue; // constant or public-IO column
        }
        assert_eq!(
          r / layout.block_rows,
          c / layout.block_cols,
          "row {r} references column {c} across a block boundary"
        );
      }
    }
    // NNZ identities: A = 2688H − 9, B = C = 1299H.
    assert_eq!(shape.A.len(), 2688 * h - 9);
    assert_eq!(shape.B.len(), 1299 * h);
    assert_eq!(shape.C.len(), 1299 * h);
    // Each block's terminal row targets its own ordered public-IO column.
    for f in 0..3 {
      let terminal_row = (f + 1) * layout.block_rows - 1;
      let io_col = layout.num_vars() + 1 + f;
      assert!(
        shape
          .C
          .iter()
          .any(|(r, c, _)| *r == terminal_row && *c == io_col)
      );
    }
  }

  #[test]
  fn ksweep_derive_table_is_pinned() {
    // §6: at combined log_n = 14, the full k = 7..=13 sweep derives these
    // (log_p, s) pairs.
    use crate::provider::pcs::integer_modpcs::IntEvalParams;
    let expected_log_p = [26, 22, 20, 18, 16, 14, 13];
    let expected_s = [9, 13, 16, 20, 29, 53, 90];
    for (i, k) in (7..=13usize).enumerate() {
      let p = IntEvalParams::derive(256, 64, k, 14).unwrap();
      assert_eq!((p.log_p, p.s), (expected_log_p[i], expected_s[i]), "k={k}");
      assert_eq!((p.numlimb, p.numlimb_var), (4, 2));
    }
  }

  #[test]
  fn verifier_key_constructor_enforces_every_predicate() {
    use crate::imod_spartan_modp::IntModSpartanModpSNARK;
    let set = build_all_params().unwrap();
    let (shape, layout) = build_shape::<ME>(&set, 1).unwrap();
    let (_pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape).unwrap();

    // The well-formed case passes.
    PoseidonVerifierKey::new(vk.clone(), &set, &layout).unwrap();

    // A reordered parameter set cannot be produced by the public API
    // (private constructor fills FIELD_ORDER); a hand-built reorder is
    // rejected by the per-index modulus binding.
    let reordered = Poseidon2ParamsSet {
      params: [
        set.params[1].clone(),
        set.params[0].clone(),
        set.params[2].clone(),
      ],
    };
    assert!(PoseidonVerifierKey::new(vk.clone(), &reordered, &layout).is_err());

    // Wrong-layout digest: a different H's layout.
    let (_shape2, layout2) = build_shape::<ME>(&set, 2).unwrap();
    assert!(PoseidonVerifierKey::new(vk.clone(), &set, &layout2).is_err());

    // A key whose shape has the wrong IO arity fails predicate 1.
    let one = BigUint::one();
    let toy = crate::imod_r1cs_modp::IntModR1CSShapeModp::<ME>::new(
      2,
      4,
      1,
      vec![(0, 0, one.clone())],
      vec![(0, 1, one.clone())],
      vec![(0, 2, one)],
      vec![BigUint::from(14u32), BigUint::from(2u32)],
    )
    .unwrap();
    let (_tpk, tvk) = IntModSpartanModpSNARK::<ME>::setup(toy).unwrap();
    assert!(PoseidonVerifierKey::new(tvk, &set, &layout).is_err());
  }

  /// Direct fixture comparison against the private parameter fields
  /// (modulus, `M_E`, `M_I`, all 80 round constants in draw order), plus
  /// the fixture's explicit `field_order` array against [`FIELD_ORDER`].
  /// Lives in this descendant module because `Poseidon2Params` exposes
  /// only `modulus()`; the standalone/chain/composition KAT comparisons
  /// need only the public API and live in `tests/poseidon_modp.rs`. The
  /// fixture is produced by the independent Python generator
  /// (`scripts/gen_poseidon_kat.py`); neither implementation consumes the
  /// other's result.
  #[test]
  fn kat_fixture_matches_private_parameters() {
    let path =
      std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/poseidon2_kat_v1.json");
    let raw = std::fs::read_to_string(&path).expect("KAT fixture readable");
    let fixture: serde_json::Value = serde_json::from_str(&raw).expect("KAT fixture parses");
    assert_eq!(fixture["domain"], "limber-poseidon2-v1");
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["t"], T as u64);
    assert_eq!(fixture["alpha"], ALPHA as u64);
    assert_eq!(fixture["r_f"], R_F as u64);
    assert_eq!(fixture["r_p"], R_P as u64);
    // The explicit order array must equal FIELD_ORDER — a sorted JSON
    // object key order is NOT the circuit's semantic order.
    let order: Vec<&str> = fixture["field_order"]
      .as_array()
      .expect("field_order array")
      .iter()
      .map(|v| v.as_str().expect("field name"))
      .collect();
    assert_eq!(order, FIELD_ORDER.map(|f| f.name()).to_vec());

    let from_hex = |v: &serde_json::Value| -> BigUint {
      let s = v.as_str().expect("hex string");
      assert_eq!(s.len(), 64, "32-byte lowercase hex");
      BigUint::parse_bytes(s.as_bytes(), 16).expect("valid hex")
    };
    let matrix = |v: &serde_json::Value| -> [[u64; 3]; 3] {
      core::array::from_fn(|i| core::array::from_fn(|j| v[i][j].as_u64().expect("matrix entry")))
    };

    for field in FIELD_ORDER {
      let params = build_params(field).unwrap();
      let entry = &fixture["fields"][field.name()];
      assert!(!entry.is_null(), "fixture has field {}", field.name());
      assert_eq!(&from_hex(&entry["modulus"]), params.modulus());
      assert_eq!(matrix(&entry["m_e"]), params.m_e);
      assert_eq!(matrix(&entry["m_i"]), params.m_i);
      // Flat draw-order constants: rounds 1..=64, full rounds 3 lanes.
      let flat: Vec<BigUint> = entry["round_constants"]
        .as_array()
        .expect("round_constants array")
        .iter()
        .map(&from_hex)
        .collect();
      assert_eq!(flat.len(), 80);
      let rust_flat: Vec<BigUint> = params.rc.iter().flatten().cloned().collect();
      assert_eq!(flat, rust_flat, "round constants for {}", field.name());
      assert_eq!(from_hex(&entry["iv"]), chain_iv());
    }
  }
}
