//! A row builder for Integer Mod-R1CS circuits over symbolic linear
//! combinations, with the gadgets the MultiSwap statement needs.
//!
//! Every row is `⟨A, z⟩ · ⟨B, z⟩ = ⟨C, z⟩ + m · q` over the non-negative
//! integers with a public modulus `m` (`m = 0` means exact). All matrix
//! coefficients are non-negative, so a subtraction is always expressed
//! as a witnessed difference (`d + b = a`, exact), and "`x ≥ 0`" for a
//! witness is free — the Mod-PCS bounds every committed value. Values
//! asserted below `2^16` live in a dedicated [`SmallValueBlock`] at the
//! start of the witness and cost no rows.
//!
//! The builder evaluates every row as it is emitted and panics on an
//! unsatisfied one, so a circuit that finalizes is satisfied by
//! construction; `IntModR1CSShapeModp::is_sat` re-checks independently.

use crate::{
  errors::SpartanError,
  imod_r1cs_modp::{IntModR1CSShapeModp, WidthSegment},
  traits::mod_engine::{ModEngine, SmallValueBlock},
};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};

/// A symbolic variable of `z = (w, 1, x)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Var {
  /// Witness column (regular region).
  W(usize),
  /// Witness column in the small-value block.
  Small(usize),
  /// The constant-one column.
  Const,
  /// Public input `x[i]`.
  Io(usize),
}

/// A linear combination `Σ coeff · var` with non-negative coefficients.
#[derive(Clone, Debug, Default)]
pub struct Lc(pub Vec<(Var, BigUint)>);

impl Lc {
  /// The constant `c`.
  pub fn constant(c: BigUint) -> Self {
    if c.is_zero() {
      Self::default()
    } else {
      Self(vec![(Var::Const, c)])
    }
  }
  /// A single variable.
  pub fn var(v: Var) -> Self {
    Self(vec![(v, BigUint::one())])
  }
  /// `self + coeff · v`.
  pub fn add_term(mut self, v: Var, coeff: BigUint) -> Self {
    if !coeff.is_zero() {
      self.0.push((v, coeff));
    }
    self
  }
  /// `self + other`.
  pub fn plus(mut self, other: &Lc) -> Self {
    self.0.extend(other.0.iter().cloned());
    self
  }
  /// `self + c`.
  pub fn add_const(self, c: &BigUint) -> Self {
    self.add_term(Var::Const, c.clone())
  }
  /// `k · self`.
  pub fn scale(&self, k: &BigUint) -> Self {
    Self(self.0.iter().map(|(v, c)| (*v, c * k)).collect())
  }
  /// Merge duplicate variables and reduce coefficients modulo `p` (only
  /// valid for combinations consumed by rows with modulus `p`).
  pub fn normalize_mod(&self, p: &BigUint) -> Self {
    let mut acc: Vec<(Var, BigUint)> = Vec::new();
    for (v, c) in &self.0 {
      if let Some(slot) = acc.iter_mut().find(|(u, _)| u == v) {
        slot.1 = (&slot.1 + c) % p;
      } else {
        acc.push((*v, c % p));
      }
    }
    acc.retain(|(_, c)| !c.is_zero());
    Self(acc)
  }
  /// Number of terms.
  pub fn len(&self) -> usize {
    self.0.len()
  }
  /// Whether the combination is empty (identically zero).
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }
}

/// Which modulus a row reduces by.
#[derive(Clone, Debug)]
pub enum Modulus {
  /// Exact integer equation.
  Exact,
  /// Public constant modulus.
  Public(BigUint),
}

/// The builder state: rows, column values, and the small-value block.
#[derive(Debug, Default)]
pub struct Builder {
  /// Regular witness values.
  pub w: Vec<BigUint>,
  /// Per-`w` value bound (upper bound on the value for any satisfying
  /// assignment, from the modulus that produced it). Drives width-grouped
  /// commitment: a value bounded by `p` (254-bit) need not commit at the
  /// wide 2048-bit chunk count. Defaults to the global `WIDE_BOUND`.
  pub bounds: Vec<BigUint>,
  /// Small-value block values (each `< 2^16`).
  pub small: Vec<BigUint>,
  /// Public inputs.
  pub io: Vec<BigUint>,
  a: Vec<(usize, Lc)>,
  b: Vec<(usize, Lc)>,
  c: Vec<(usize, Lc)>,
  mods: Vec<BigUint>,
  q: Vec<BigUint>,
  /// Non-zero matrix entries emitted so far (after LC merging).
  pub nnz: usize,
  /// When set, `finalize` reorders the witness into aligned width-grouped
  /// commitment segments (by tracked value bound) instead of one uniform
  /// block.
  pub segment_widths: bool,
}

/// A finalized circuit: shape plus its satisfying assignment.
pub struct Built<M: ModEngine> {
  /// The shape (with the small-value block declared).
  pub shape: IntModR1CSShapeModp<M>,
  /// Witness `w` (padded).
  pub w: Vec<BigUint>,
  /// Per-column value bound aligned with `w` (for width-grouped commit).
  pub bounds: Vec<BigUint>,
  /// Quotients `q` (padded).
  pub q: Vec<BigUint>,
  /// Public inputs.
  pub io: Vec<BigUint>,
  /// Real (unpadded) row count.
  pub real_rows: usize,
  /// Real (unpadded) column count, block included.
  pub real_cols: usize,
  /// Size of the small-value block (a power of two, possibly zero).
  pub block_len: usize,
}

/// Global commitment norm bound (2^2048): every witness value is `<`
/// this, so it is the sound default per-value bound and the wide
/// segment's `log_t_f`.
pub fn wide_bound() -> BigUint {
  BigUint::one() << 2048
}

/// Planned width-grouped column layout produced by `plan_segments`.
struct SegLayout {
  num_vars: usize,
  small_col: Vec<usize>,
  w_col: Vec<usize>,
  segments: Vec<WidthSegment>,
  small_block: Option<SmallValueBlock>,
}

impl Builder {
  /// Empty builder.
  pub fn new() -> Self {
    Self::default()
  }

  /// Rows emitted so far.
  pub fn num_rows(&self) -> usize {
    self.mods.len()
  }

  /// Value of a variable.
  pub fn value(&self, v: Var) -> BigUint {
    match v {
      Var::W(i) => self.w[i].clone(),
      Var::Small(i) => self.small[i].clone(),
      Var::Const => BigUint::one(),
      Var::Io(i) => self.io[i].clone(),
    }
  }

  /// Sound upper bound on a variable's value (any satisfying assignment).
  fn var_bound(&self, v: Var) -> BigUint {
    match v {
      Var::W(i) => self.bounds[i].clone(),
      Var::Small(_) => BigUint::one() << 16u32,
      Var::Const => BigUint::from(2u32),
      Var::Io(_) => wide_bound(),
    }
  }

  /// Sound upper bound on `|eval(lc)|` = sum of `coeff * var_bound`,
  /// capped at `WIDE_BOUND` (every witness value is `< 2^2048` by circuit
  /// soundness, so the cap never loses a real value — it only avoids an
  /// over-wide segment for a combination whose naive term-sum exceeds it).
  pub fn lc_bound(&self, lc: &Lc) -> BigUint {
    let mut acc = BigUint::zero();
    for (v, c) in &lc.0 {
      acc += c * self.var_bound(*v);
    }
    acc.min(wide_bound())
  }

  /// Value of a linear combination.
  pub fn eval(&self, lc: &Lc) -> BigUint {
    let mut acc = BigUint::zero();
    for (v, c) in &lc.0 {
      acc += c * self.value(*v);
    }
    acc
  }

  /// Allocate a regular witness at the global `WIDE_BOUND` (2^2048 — the
  /// commitment norm bound; sound for any value the circuit produces).
  pub fn alloc(&mut self, v: BigUint) -> Var {
    self.alloc_bounded(v, wide_bound())
  }

  /// Allocate a regular witness known to satisfy `v < bound` for every
  /// satisfying assignment (e.g. the output of a `mod m` row is `< m`).
  /// The bound feeds width-grouped commitment; it must be a real upper
  /// bound, not the observed value, or the commitment is unsound.
  pub fn alloc_bounded(&mut self, v: BigUint, bound: BigUint) -> Var {
    debug_assert!(
      v < bound,
      "alloc_bounded: value {} >= bound {}",
      v.bits(),
      bound.bits()
    );
    self.w.push(v);
    self.bounds.push(bound);
    Var::W(self.w.len() - 1)
  }

  /// Allocate a small-value witness (`v < 2^16`, enforced by the block).
  pub fn alloc_small(&mut self, v: BigUint) -> Var {
    assert!(v.bits() <= 16, "alloc_small: value has {} bits", v.bits());
    self.small.push(v);
    Var::Small(self.small.len() - 1)
  }

  /// Allocate a public input.
  pub fn alloc_io(&mut self, v: BigUint) -> Var {
    self.io.push(v);
    Var::Io(self.io.len() - 1)
  }

  /// Emit a row `a · b = c (mod m)`, computing and checking the quotient.
  pub fn row(&mut self, a: Lc, b: Lc, c: Lc, m: Modulus) {
    let av = self.eval(&a);
    let bv = self.eval(&b);
    let cv = self.eval(&c);
    let lhs = &av * &bv;
    let (m_val, q) = match &m {
      Modulus::Exact => {
        assert!(
          lhs == cv,
          "row {}: exact row unsatisfied ({lhs} != {cv})",
          self.num_rows()
        );
        (BigUint::zero(), BigUint::zero())
      }
      Modulus::Public(m) => {
        assert!(
          lhs >= cv,
          "row {}: lhs {lhs} < rhs {cv} (quotient would be negative)",
          self.num_rows()
        );
        let diff = &lhs - &cv;
        let (q, r) = diff.div_rem(m);
        assert!(
          r.is_zero(),
          "row {}: lhs − rhs not divisible by the modulus",
          self.num_rows()
        );
        (m.clone(), q)
      }
    };
    let i = self.num_rows();
    self.nnz += a.len() + b.len() + c.len();
    self.a.push((i, a));
    self.b.push((i, b));
    self.c.push((i, c));
    self.mods.push(m_val);
    self.q.push(q);
  }

  // ------------------------------------------------------------ gadgets

  /// `c = a · b mod m` (public modulus); returns the new witness.
  pub fn mul_mod(&mut self, a: &Lc, b: &Lc, m: &BigUint) -> Var {
    let v = (self.eval(a) * self.eval(b)) % m;
    let c = self.alloc_bounded(v, m.clone());
    self.row(a.clone(), b.clone(), Lc::var(c), Modulus::Public(m.clone()));
    c
  }

  /// `c = a · b` exactly; returns the new witness.
  pub fn mul_exact(&mut self, a: &Lc, b: &Lc) -> Var {
    let v = self.eval(a) * self.eval(b);
    let bd = (self.lc_bound(a) * self.lc_bound(b)).min(wide_bound());
    let c = self.alloc_bounded(v, bd);
    self.row(a.clone(), b.clone(), Lc::var(c), Modulus::Exact);
    c
  }

  /// Assert `a · b = c` exactly.
  pub fn assert_exact(&mut self, a: &Lc, b: &Lc, c: &Lc) {
    self.row(a.clone(), b.clone(), c.clone(), Modulus::Exact);
  }

  /// Assert `lc = v mod m` for a fresh witness `v` (materialize an LC
  /// reduced modulo a public modulus).
  pub fn reduce_mod(&mut self, lc: &Lc, m: &BigUint) -> Var {
    let v = self.eval(lc) % m;
    let out = self.alloc_bounded(v, m.clone());
    self.row(
      lc.clone(),
      Lc::constant(BigUint::one()),
      Lc::var(out),
      Modulus::Public(m.clone()),
    );
    out
  }

  /// Materialize an exact LC as a witness (`v = lc`).
  pub fn materialize(&mut self, lc: &Lc) -> Var {
    let v = self.eval(lc);
    let bd = self.lc_bound(lc);
    let out = self.alloc_bounded(v, bd);
    self.assert_exact(lc, &Lc::constant(BigUint::one()), &Lc::var(out));
    out
  }

  /// `d = a − b` (requires `a ≥ b`): witness `d` with `d + b = a`.
  pub fn sub(&mut self, a: &Lc, b: &Lc) -> Var {
    let av = self.eval(a);
    let bv = self.eval(b);
    assert!(av >= bv, "sub: negative difference");
    let bd = self.lc_bound(a);
    let d = self.alloc_bounded(&av - &bv, bd);
    self.assert_exact(&Lc::var(d).plus(b), &Lc::constant(BigUint::one()), a);
    d
  }

  /// Assert `a < b`: witness `e` with `e + a + 1 = b`.
  pub fn assert_lt(&mut self, a: &Lc, b: &Lc) {
    let av = self.eval(a);
    let bv = self.eval(b);
    assert!(av < bv, "assert_lt: {av} >= {bv}");
    let ebd = self.lc_bound(b);
    let e = self.alloc_bounded(&bv - &av - BigUint::one(), ebd);
    self.assert_exact(
      &Lc::var(e).plus(a).add_const(&BigUint::one()),
      &Lc::constant(BigUint::one()),
      b,
    );
  }

  /// `c = a · b mod n` for a WITNESS modulus `n`: `t = n · k`, then
  /// `a · b = c + t` exactly. Two rows.
  pub fn mul_mod_witness(&mut self, a: &Lc, b: &Lc, n: &Lc) -> Var {
    let nv = self.eval(n);
    let prod = self.eval(a) * self.eval(b);
    let (k, c) = prod.div_rem(&nv);
    let k = self.alloc(k);
    let t = self.mul_exact(n, &Lc::var(k));
    let c = self.alloc(c);
    self.row(
      a.clone(),
      b.clone(),
      Lc::var(c).add_term(t, BigUint::one()),
      Modulus::Exact,
    );
    c
  }

  /// Allocate a bit with its exact `b · b = b` row.
  pub fn alloc_bit(&mut self, bit: bool) -> Var {
    let b = self.alloc_bounded(BigUint::from(bit as u8), BigUint::from(2u32));
    self.assert_exact(&Lc::var(b), &Lc::var(b), &Lc::var(b));
    b
  }

  /// Decompose `lc` into `nbits` bits (LSB first) with bit rows and one
  /// exact reconstruction row. Panics if the value does not fit.
  pub fn bits_of(&mut self, lc: &Lc, nbits: usize) -> Vec<Var> {
    let v = self.eval(lc);
    assert!(v.bits() as usize <= nbits, "bits_of: value does not fit");
    let bits: Vec<Var> = (0..nbits as u64)
      .map(|i| self.alloc_bit(v.bit(i)))
      .collect();
    let mut recon = Lc::default();
    for (i, b) in bits.iter().enumerate() {
      recon = recon.add_term(*b, BigUint::one() << i);
    }
    self.assert_exact(&recon, &Lc::constant(BigUint::one()), lc);
    bits
  }

  /// Decompose `lc` into `nchunks` 16-bit chunks (LSB first) in the
  /// small-value block, with one exact reconstruction row.
  pub fn chunks_of(&mut self, lc: &Lc, nchunks: usize) -> Vec<Var> {
    let v = self.eval(lc);
    assert!(
      v.bits() as usize <= 16 * nchunks,
      "chunks_of: value does not fit"
    );
    let mask = BigUint::from(0xffffu32);
    let chunks: Vec<Var> = (0..nchunks)
      .map(|i| self.alloc_small((&v >> (16 * i)) & &mask))
      .collect();
    let mut recon = Lc::default();
    for (i, c) in chunks.iter().enumerate() {
      recon = recon.add_term(*c, BigUint::one() << (16 * i));
    }
    self.assert_exact(&recon, &Lc::constant(BigUint::one()), lc);
    chunks
  }

  /// `Σ 2^i · bits[i]` as an LC (LSB first).
  pub fn lc_of_bits(bits: &[Var]) -> Lc {
    let mut lc = Lc::default();
    for (i, b) in bits.iter().enumerate() {
      lc = lc.add_term(*b, BigUint::one() << i);
    }
    lc
  }

  /// Square-and-multiply `base^e mod m` with a VARIABLE base, `e` given
  /// as bits MSB first (each bit an LC — a witness bit or the constant
  /// 1). Per bit: `sq = acc² mod m`, `t = bit · (base − 1)`,
  /// `acc' = sq · (t + 1) mod m`. `base_minus_one` is the witnessed
  /// `base − 1` (from [`Self::sub`]). For a witness modulus pass
  /// `mod_lc = Some(n)`; each modular product then costs two rows.
  pub fn exp_var_base(
    &mut self,
    base: &Lc,
    base_minus_one: &Lc,
    exp_bits_msb: &[Lc],
    m: &BigUint,
    mod_lc: Option<&Lc>,
  ) -> Var {
    let mut acc = Lc::constant(BigUint::one());
    let mut acc_var: Option<Var> = None;
    for bit in exp_bits_msb {
      let sq = match mod_lc {
        Some(n) => self.mul_mod_witness(&acc, &acc, n),
        None => self.mul_mod(&acc, &acc, m),
      };
      let t = self.mul_exact(bit, base_minus_one);
      let mult = Lc::var(t).add_const(&BigUint::one());
      let next = match mod_lc {
        Some(n) => self.mul_mod_witness(&Lc::var(sq), &mult, n),
        None => self.mul_mod(&Lc::var(sq), &mult, m),
      };
      acc = Lc::var(next);
      acc_var = Some(next);
    }
    let _ = base;
    acc_var.expect("at least one exponent bit")
  }

  /// Square-and-multiply with a CONSTANT base `g` (the multiplier
  /// `bit · (g − 1) + 1` is a linear combination): 2 rows per bit.
  pub fn exp_const_base(&mut self, g: &BigUint, exp_bits_msb: &[Lc], m: &BigUint) -> Var {
    let g_minus_1 = g - BigUint::one();
    let mut acc = Lc::constant(BigUint::one());
    let mut acc_var: Option<Var> = None;
    for bit in exp_bits_msb {
      let sq = self.mul_mod(&acc, &acc, m);
      let mult = bit.scale(&g_minus_1).add_const(&BigUint::one());
      let next = self.mul_mod(&Lc::var(sq), &mult, m);
      acc = Lc::var(next);
      acc_var = Some(next);
    }
    acc_var.expect("at least one exponent bit")
  }

  /// Constant-base square-and-multiply modulo a WITNESS modulus `n`
  /// (four rows per bit: two witnessed-modulus products).
  pub fn exp_const_base_witness_mod(&mut self, g: &BigUint, exp_bits_msb: &[Lc], n: &Lc) -> Var {
    let g_minus_1 = g - BigUint::one();
    let mut acc = Lc::constant(BigUint::one());
    let mut acc_var: Option<Var> = None;
    for bit in exp_bits_msb {
      let sq = self.mul_mod_witness(&acc, &acc, n);
      let mult = bit.scale(&g_minus_1).add_const(&BigUint::one());
      let next = self.mul_mod_witness(&Lc::var(sq), &mult, n);
      acc = Lc::var(next);
      acc_var = Some(next);
    }
    acc_var.expect("at least one exponent bit")
  }

  /// Canonical representative `min(x, N − x)` of the quotient group
  /// `(Z/N)^*/{±1}`: `y = N − x`; selector bit `s = [y < x]`;
  /// `r = s·y + (1−s)·x` via two products and one exact row; `x − r ≥ 0`
  /// and `y − r ≥ 0` as witnessed differences. Six rows.
  pub fn canon(&mut self, x: &Lc, n: &BigUint) -> Var {
    let xv = self.eval(x);
    let y = self.sub(&Lc::constant(n.clone()), x);
    let yv = n - &xv;
    let s = self.alloc_bit(yv < xv);
    let t1 = self.mul_exact(&Lc::var(s), &Lc::var(y));
    let t2 = self.mul_exact(&Lc::var(s), x);
    let rv = if yv < xv { yv.clone() } else { xv.clone() };
    let r = self.alloc(rv);
    self.assert_exact(
      &Lc::var(t1).plus(x),
      &Lc::constant(BigUint::one()),
      &Lc::var(r).add_term(t2, BigUint::one()),
    );
    let _dx = self.sub(x, &Lc::var(r));
    let _dy = self.sub(&Lc::var(y), &Lc::var(r));
    r
  }

  /// Finalize into a shape and padded witness. `block_len_hint` lets the
  /// caller reserve a larger block than needed (kept a power of two).
  /// Enable width-grouped commitment segmentation in `finalize`.
  pub fn with_width_segments(mut self) -> Self {
    self.segment_widths = true;
    self
  }

  /// Plan the width-grouped column layout: classify every witness value
  /// (small block + regular `w`) by its tracked bound into log_t_f classes
  /// {64, 256, 2048}, lay each class as an aligned power-of-two block
  /// (largest first, so offsets stay aligned), fill the tail to a power of
  /// two with tiny zero blocks, and return the column map + segments.
  fn plan_segments(&self) -> SegLayout {
    // Commit-width class of a value bound (bits) -> log_t_f in {64,256,2048}.
    let class_of = |bits: usize| -> usize {
      if bits <= 64 {
        64
      } else if bits <= 256 {
        256
      } else {
        2048
      }
    };
    // Regular w sorted by class DESC, so the wide (32-limb) values sit at
    // offset 0 and form a few large aligned blocks (fragmentation comes
    // from the run START alignment, so wide must start at 0).
    let mut reg: Vec<(usize, usize)> = self
      .bounds
      .iter()
      .enumerate()
      .map(|(i, b)| (class_of(b.bits() as usize), i))
      .collect();
    reg.sort_by_key(|x| core::cmp::Reverse(x.0));

    // Small block is a single aligned region at the END (all values < 2^16,
    // one range-check block); putting it last keeps the wide run at 0.
    let s_small = self.small.len().next_power_of_two() * (!self.small.is_empty() as usize);
    let total = reg.len() + s_small;
    let num_vars = total.next_power_of_two().max(2);
    let small_start = num_vars - s_small;

    let small_col: Vec<usize> = (0..self.small.len()).map(|i| small_start + i).collect();
    let mut w_col = vec![usize::MAX; self.w.len()];
    for (pos, (_, idx)) in reg.iter().enumerate() {
      w_col[*idx] = pos;
    }

    // Class at absolute position: regular region by sorted class, the small
    // block and any gap are tiny (64).
    let class_at = |p: usize| -> usize { if p < reg.len() { reg[p].0 } else { 64 } };
    // Quantize class labels to G-aligned granules so a value just past a
    // power-of-two boundary does not spawn a chain of 1-column segments;
    // a granule takes the MAX class of its members (a little over-wide at
    // class boundaries, far fewer segments). Positions never move.
    // 4096-column granule by default (the swept optimum: recovers the
    // verify/proof penalty while keeping the prover win); SEGGLOG
    // overrides (benching knob).
    let g_log: usize = std::env::var("SEGGLOG")
      .ok()
      .and_then(|v| v.parse::<usize>().ok())
      .unwrap_or(12);
    let g = 1usize << g_log;
    let granule_class = |gk: usize| -> usize {
      let lo = gk * g;
      let hi = (lo + g).min(num_vars);
      (lo..hi).map(class_at).max().unwrap_or(64)
    };

    // Tile [0, num_vars) into aligned dyadic blocks that never cross a
    // granule-class boundary; each block is one commitment segment.
    let mut segments: Vec<WidthSegment> = Vec::new();
    let ngran = num_vars.div_ceil(g);
    let mut off = 0usize;
    while off < num_vars {
      let gk = off / g;
      let class = if num_vars <= g {
        (0..num_vars).map(class_at).max().unwrap_or(64)
      } else {
        granule_class(gk)
      };
      // extent of this granule-class run
      let mut end_g = gk + 1;
      while end_g < ngran && granule_class(end_g) == class {
        end_g += 1;
      }
      let run_end = (end_g * g).min(num_vars);
      let by_align = if off == 0 {
        num_vars
      } else {
        off.isolate_lowest_one()
      };
      let avail = run_end - off;
      let by_size = 1usize << (usize::BITS - 1 - avail.leading_zeros());
      let b = by_align.min(by_size);
      segments.push(WidthSegment {
        start: off,
        log_len: b.trailing_zeros() as usize,
        log_t_f: class,
      });
      off += b;
    }

    // Range-check block over the aligned small region (every value < 2^16).
    let small_block = (s_small > 0).then(|| SmallValueBlock {
      start: small_start,
      log_len: s_small.trailing_zeros() as usize,
    });

    SegLayout {
      num_vars,
      small_col,
      w_col,
      segments,
      small_block,
    }
  }

  /// Finalize into a shape and padded witness, laying the witness out
  /// either as width-grouped commitment segments (when enabled) or one
  /// uniform block.
  pub fn finalize<M: ModEngine>(self) -> Result<Built<M>, SpartanError> {
    let num_cons = self.num_rows().next_power_of_two().max(2);
    let num_io = self.io.len();

    // Column layout for the witness values: either width-grouped aligned
    // segments, or one uniform block (small block prefix + regular w).
    let (num_vars, small_col, w_col, width_segments, small_block, block_len) =
      if self.segment_widths {
        let plan = self.plan_segments();
        let bl = plan.small_block.map(|b| b.size()).unwrap_or(0);
        (
          plan.num_vars,
          plan.small_col,
          plan.w_col,
          plan.segments,
          plan.small_block,
          bl,
        )
      } else {
        let block_len = if self.small.is_empty() {
          0
        } else {
          self.small.len().next_power_of_two()
        };
        let real_cols = block_len + self.w.len();
        let nv = real_cols.next_power_of_two().max(2);
        let sc: Vec<usize> = (0..self.small.len()).collect();
        let wc: Vec<usize> = (0..self.w.len()).map(|i| block_len + i).collect();
        let sb = if block_len > 0 {
          Some(SmallValueBlock {
            start: 0,
            log_len: block_len.trailing_zeros() as usize,
          })
        } else {
          None
        };
        (nv, sc, wc, Vec::<WidthSegment>::new(), sb, block_len)
      };
    let real_cols = block_len + self.w.len();
    let const_col = num_vars;
    let resolve = |v: Var| -> usize {
      match v {
        Var::Small(i) => small_col[i],
        Var::W(i) => w_col[i],
        Var::Const => const_col,
        Var::Io(i) => const_col + 1 + i,
      }
    };
    let to_entries = |rows: &[(usize, Lc)]| -> Vec<(usize, usize, BigUint)> {
      let mut out = Vec::new();
      for (r, lc) in rows {
        // Merge duplicate variables within an LC (exact sums).
        let mut acc: Vec<(usize, BigUint)> = Vec::new();
        for (v, c) in &lc.0 {
          let col = resolve(*v);
          if let Some(slot) = acc.iter_mut().find(|(u, _)| *u == col) {
            slot.1 += c;
          } else {
            acc.push((col, c.clone()));
          }
        }
        for (col, c) in acc {
          if !c.is_zero() {
            out.push((*r, col, c));
          }
        }
      }
      out
    };
    let a = to_entries(&self.a);
    let b = to_entries(&self.b);
    let c = to_entries(&self.c);
    let mut mods = self.mods;
    mods.resize(num_cons, BigUint::from(2u32));
    // Scatter values + per-column bounds through the column map.
    let two16 = BigUint::from(1u32) << 16usize;
    let mut w = vec![BigUint::zero(); num_vars];
    let mut bounds = vec![BigUint::from(2u32); num_vars];
    for (i, v) in self.small.iter().enumerate() {
      w[small_col[i]] = v.clone();
      bounds[small_col[i]] = two16.clone();
    }
    for (i, v) in self.w.iter().enumerate() {
      w[w_col[i]] = v.clone();
      bounds[w_col[i]] = self.bounds[i].clone();
    }
    let mut q = self.q;
    q.resize(num_cons, BigUint::zero());
    let mut shape = IntModR1CSShapeModp::<M>::new(num_cons, num_vars, num_io, a, b, c, mods)?;
    if let Some(sb) = small_block {
      shape = shape.with_small_value_blocks(vec![sb])?;
    }
    if !width_segments.is_empty() {
      shape = shape.with_width_segments(width_segments)?;
    }
    Ok(Built {
      shape,
      w,
      bounds,
      q,
      io: self.io,
      real_rows: self.a.len(),
      real_cols,
      block_len,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256DynPrimeEngine;

  #[test]
  fn exp_gadgets_match_modpow() {
    let m = BigUint::from(1_000_003u64);
    let mut b = Builder::new();
    let g = BigUint::from(12345u64);
    let e = BigUint::from(0xdeadbeefu64);
    let e_lc = Lc::var(b.alloc(e.clone()));
    let bits = b.bits_of(&e_lc, 32);
    let bits_msb: Vec<Lc> = bits.iter().rev().map(|v| Lc::var(*v)).collect();
    let r1 = b.exp_const_base(&g, &bits_msb, &m);
    let gv = b.alloc(g.clone());
    let gm1 = b.sub(&Lc::var(gv), &Lc::constant(BigUint::one()));
    let r2 = b.exp_var_base(&Lc::var(gv), &Lc::var(gm1), &bits_msb, &m, None);
    let n = b.alloc(m.clone());
    let r3 = b.exp_var_base(
      &Lc::var(gv),
      &Lc::var(gm1),
      &bits_msb,
      &m,
      Some(&Lc::var(n)),
    );
    let expect = g.modpow(&e, &m);
    assert_eq!(b.value(r1), expect);
    assert_eq!(b.value(r2), expect);
    assert_eq!(b.value(r3), expect);
    let c = b.canon(&Lc::var(r2), &m);
    let cv = b.value(c);
    assert!(cv == expect || cv == &m - &expect);
    let built = b.finalize::<T256DynPrimeEngine>().unwrap();
    assert_eq!(built.io.len(), 0);
    assert!(built.real_rows > 32 * 5);
  }

  #[test]
  fn chunks_and_small_block_finalize() {
    let mut b = Builder::new();
    let v = b.alloc(BigUint::from(0x1234_5678_9abcu64));
    let ch = b.chunks_of(&Lc::var(v), 3);
    assert_eq!(b.value(ch[0]), BigUint::from(0x9abcu32));
    assert_eq!(b.value(ch[2]), BigUint::from(0x1234u32));
    let io = b.alloc_io(BigUint::from(7u32));
    let _ = b.materialize(&Lc::var(io).add_const(&BigUint::from(3u32)));
    let built = b.finalize::<T256DynPrimeEngine>().unwrap();
    assert_eq!(built.block_len, 4);
    assert_eq!(built.shape.small_value_blocks().len(), 1);
    assert_eq!(built.io, vec![BigUint::from(7u32)]);
  }
}
