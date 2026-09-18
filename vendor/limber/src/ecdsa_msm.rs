//! ECDSA-verification MSM circuit gadgets for integer Mod-R1CS (secp256k1).
//!
//! Builds the elliptic-curve arithmetic of `R = u1·G + u2·Q` as rows of the
//! relation `A·z ∘ B·z = C·z + m∘q` (per-row modulus `m = p`, the secp256k1
//! base prime). EC coordinates are affine; divisions use prover-advice (one
//! row). Subtractions use **difference witnesses** to stay non-negative (a
//! `(p−1)`-coefficient negation would produce negative quotients — unsound
//! here). All witness/quotient values are `< p`.
//!
//! This is the gadget layer; the Shamir MSM loop and the row-count /
//! prove-time experiments in the test module build on it. Test-only.

use num_bigint::BigUint;
use num_integer::Integer;

/// secp256k1 base field prime `p = 2^256 − 2^32 − 977`.
fn secp256k1_p() -> BigUint {
  BigUint::parse_bytes(
    b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
    16,
  )
  .unwrap()
}

/// Modular inverse via Fermat (`p` prime): `a^{p−2} mod p`.
fn mod_inv(a: &BigUint, p: &BigUint) -> BigUint {
  a.modpow(&(p - 2u32), p)
}

/// Reference affine EC addition over secp256k1 for distinct, non-identity
/// points `P1 + P2` (no doubling, no identity). Returns `(x3, y3)`.
fn ref_ec_add(
  x1: &BigUint,
  y1: &BigUint,
  x2: &BigUint,
  y2: &BigUint,
  p: &BigUint,
) -> (BigUint, BigUint) {
  let dx = (x2 + p - x1) % p;
  let dy = (y2 + p - y1) % p;
  let lam = (&dy * mod_inv(&dx, p)) % p;
  let t = (&lam * &lam) % p;
  let x3 = (&t + p + p - x1 - x2) % p;
  let dx3 = (x1 + p - &x3) % p;
  let u = (&lam * &dx3) % p;
  let y3 = (&u + p - y1) % p;
  (x3, y3)
}

/// Reference affine EC doubling over secp256k1 (`a = 0`) for a non-identity,
/// non-2-torsion point `2·P`. Returns `(x3, y3)`.
fn ref_ec_double(x: &BigUint, y: &BigUint, p: &BigUint) -> (BigUint, BigUint) {
  let xsq = (x * x) % p;
  let r3 = (BigUint::from(3u32) * &xsq) % p;
  let two_y = (BigUint::from(2u32) * y) % p;
  let lam = (&r3 * mod_inv(&two_y, p)) % p;
  let t = (&lam * &lam) % p;
  let x3 = (&t + p + p - x - x) % p;
  let dx3 = (x + p - &x3) % p;
  let u = (&lam * &dx3) % p;
  let y3 = (&u + p - y) % p;
  (x3, y3)
}

/// A single (row, col, coeff) triple for one of the A/B/C matrices.
type Triple = (usize, usize, BigUint);

/// Accumulates the rows of an integer Mod-R1CS circuit as it is wired.
struct CircuitBuilder {
  a: Vec<Triple>,
  b: Vec<Triple>,
  c: Vec<Triple>,
  mods: Vec<BigUint>,
  w: Vec<BigUint>,
  q: Vec<BigUint>,
  /// next free witness column
  next_col: usize,
  /// the constant-1 column index (set when finalized); during building we
  /// record references and patch them — but here we pass it in explicitly.
  const_col: usize,
  p: BigUint,
}

impl CircuitBuilder {
  fn new(const_col: usize, p: BigUint) -> Self {
    let mut w = vec![BigUint::ZERO; const_col + 1];
    w[const_col] = BigUint::from(1u32); // not stored in w[] in the real shape, but
    // we keep a local z-style vector for the relation self-check in tests.
    Self {
      a: Vec::new(),
      b: Vec::new(),
      c: Vec::new(),
      mods: Vec::new(),
      w,
      q: Vec::new(),
      next_col: 0,
      const_col,
      p,
    }
  }

  /// Allocate a fresh witness column holding `val`.
  fn alloc(&mut self, val: BigUint) -> usize {
    let col = self.next_col;
    self.next_col += 1;
    if col >= self.w.len() {
      self.w.resize(col + 1, BigUint::ZERO);
    }
    self.w[col] = val;
    col
  }

  /// Push a row `(Σa)·(Σb) = (Σc) + m·q` with modulus `m`, computing and
  /// storing the quotient. `m = 0` is an *exact* integer row (`q = 0`). Panics
  /// (in tests) if `q` would be negative — the soundness guard.
  fn push_row_m(
    &mut self,
    row: usize,
    a: &[(usize, u32)],
    b: &[(usize, u32)],
    c: &[(usize, u32)],
    m: BigUint,
  ) {
    let lc = |terms: &[(usize, u32)], this: &Self| -> BigUint {
      terms
        .iter()
        .map(|(col, k)| BigUint::from(*k) * &this.w[*col])
        .sum()
    };
    let az = lc(a, self);
    let bz = lc(b, self);
    let cz = lc(c, self);
    let lhs = &az * &bz;
    assert!(lhs >= cz, "negative quotient at row {row} (unsound wiring)");
    let qv = if m == BigUint::ZERO {
      assert!(lhs == cz, "exact row {row} not satisfied over Z");
      BigUint::ZERO
    } else {
      let (qv, rem) = (&lhs - &cz).div_rem(&m);
      assert!(rem == BigUint::ZERO, "row {row} not satisfied mod m");
      qv
    };
    for (col, k) in a {
      self.a.push((row, *col, BigUint::from(*k)));
    }
    for (col, k) in b {
      self.b.push((row, *col, BigUint::from(*k)));
    }
    for (col, k) in c {
      self.c.push((row, *col, BigUint::from(*k)));
    }
    self.mods.push(m);
    if row >= self.q.len() {
      self.q.resize(row + 1, BigUint::ZERO);
    }
    self.q[row] = qv;
  }

  /// Push a modular row with modulus `p` (the EC base field).
  fn push_row(&mut self, row: usize, a: &[(usize, u32)], b: &[(usize, u32)], c: &[(usize, u32)]) {
    self.push_row_m(row, a, b, c, self.p.clone());
  }

  /// Difference witness `d = (a − b) mod p` with the row `d + b ≡ a (mod p)`.
  /// Returns the column of `d`.
  fn diff(&mut self, row: usize, a_col: usize, b_col: usize) -> usize {
    let a_val = self.w[a_col].clone();
    let b_val = self.w[b_col].clone();
    let d = (&a_val + &self.p - &b_val) % &self.p;
    let d_col = self.alloc(d);
    // (d + b)·1 = a + p·q
    self.push_row(
      row,
      &[(d_col, 1), (b_col, 1)],
      &[(self.const_col, 1)],
      &[(a_col, 1)],
    );
    d_col
  }

  /// Modular product witness `m = (x·y) mod p` with row `x·y = m + p·q`.
  fn mul(&mut self, row: usize, x_col: usize, y_col: usize) -> usize {
    let m = (&self.w[x_col] * &self.w[y_col]) % &self.p;
    let m_col = self.alloc(m);
    self.push_row(row, &[(x_col, 1)], &[(y_col, 1)], &[(m_col, 1)]);
    m_col
  }

  /// Affine EC add of points at columns `(x1,y1)`,`(x2,y2)`. Returns `(x3,y3)`
  /// columns. Uses 9 rows starting at `row`; returns the next free row.
  fn ec_add(
    &mut self,
    mut row: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
  ) -> (usize, usize, usize) {
    let dx = self.diff(row, x2, x1);
    row += 1;
    let dy = self.diff(row, y2, y1);
    row += 1;
    // λ = dy / dx  ⇒  λ·dx = dy : prover supplies λ, row asserts λ·dx ≡ dy.
    let lam_val = (&self.w[dy] * mod_inv(&self.w[dx], &self.p)) % &self.p;
    let lam = self.alloc(lam_val);
    self.push_row(row, &[(lam, 1)], &[(dx, 1)], &[(dy, 1)]);
    row += 1;
    let t = self.mul(row, lam, lam); // t = λ²
    row += 1;
    let t1 = self.diff(row, t, x1); // t1 = t − x1
    row += 1;
    let x3 = self.diff(row, t1, x2); // x3 = t1 − x2 = λ² − x1 − x2
    row += 1;
    let dx3 = self.diff(row, x1, x3); // dx3 = x1 − x3
    row += 1;
    let u = self.mul(row, lam, dx3); // u = λ·(x1 − x3)
    row += 1;
    let y3 = self.diff(row, u, y1); // y3 = u − y1
    row += 1;
    (x3, y3, row)
  }

  /// Affine EC doubling `2·P` for `P` at columns `(x,y)` (secp256k1, `a=0`).
  /// Returns `(x3,y3)` columns and the next free row. 9 rows.
  fn ec_double(&mut self, mut row: usize, x: usize, y: usize) -> (usize, usize, usize) {
    let cc = self.const_col;
    let xsq = self.mul(row, x, x); // xsq = x²
    row += 1;
    // r3 = 3·xsq mod p ; row: 3·xsq ≡ r3 (mod p)
    let r3_val = (BigUint::from(3u32) * &self.w[xsq]) % &self.p;
    let r3 = self.alloc(r3_val);
    self.push_row(row, &[(xsq, 3)], &[(cc, 1)], &[(r3, 1)]);
    row += 1;
    // two_y = 2y mod p (reduce first, so the slope product stays < p² ⇒ q < p;
    // a `B=[(y,2)]` row would give λ·2y < 2p² ⇒ q up to 2p ≥ 2^256, exceeding
    // log_t_f and truncating the quotient commitment).
    let two_y_val = (BigUint::from(2u32) * &self.w[y]) % &self.p;
    let two_y = self.alloc(two_y_val);
    self.push_row(row, &[(y, 2)], &[(cc, 1)], &[(two_y, 1)]);
    row += 1;
    // λ = r3 / (2y) ⇒ λ·two_y ≡ r3 (mod p)
    let lam_val = (&self.w[r3] * mod_inv(&self.w[two_y], &self.p)) % &self.p;
    let lam = self.alloc(lam_val);
    self.push_row(row, &[(lam, 1)], &[(two_y, 1)], &[(r3, 1)]);
    row += 1;
    let t = self.mul(row, lam, lam); // t = λ²
    row += 1;
    // x3 = t − 2x ; row: (x3 + 2x) ≡ t (mod p)
    let x3_val = (&self.w[t] + &self.p + &self.p - &self.w[x] - &self.w[x]) % &self.p;
    let x3 = self.alloc(x3_val);
    self.push_row(row, &[(x3, 1), (x, 2)], &[(cc, 1)], &[(t, 1)]);
    row += 1;
    let dx3 = self.diff(row, x, x3); // dx3 = x − x3
    row += 1;
    let u = self.mul(row, lam, dx3); // u = λ·(x − x3)
    row += 1;
    let y3 = self.diff(row, u, y); // y3 = u − y
    row += 1;
    (x3, y3, row)
  }

  /// Shamir joint MSM `R = u1·G + u2·Q` for `n`-bit scalars (MSB-first bit
  /// vectors `b1`,`b2`), given the precomputed table
  /// `[SEED, SEED+G, SEED+Q, SEED+G+Q]` (indexed `b1 + 2·b2`) and the
  /// `neg_correction` point `−(2ⁿ−1)·SEED`. The SEED offset keeps the
  /// accumulator off the point at infinity. Per-round addend is a constant
  /// point (public bits ⇒ addend selection done off-circuit, matching Zinc+'s
  /// public `u1,u2`). Returns `(Rx,Ry)` columns and the next free row.
  ///
  /// Cost: round 0 = select only; rounds 1..n = double (8) + add (9); plus one
  /// final add for the correction ⇒ `17·(n−1) + 9` EC rows.
  fn shamir_msm(
    &mut self,
    mut row: usize,
    b1: &[u8],
    b2: &[u8],
    table: &[(BigUint, BigUint); 4],
    neg_correction: &(BigUint, BigUint),
  ) -> (usize, usize, usize) {
    let n = b1.len();
    let idx0 = (b1[0] + 2 * b2[0]) as usize;
    let mut ax = self.alloc(table[idx0].0.clone());
    let mut ay = self.alloc(table[idx0].1.clone());
    for i in 1..n {
      let (dx, dy, r) = self.ec_double(row, ax, ay); // acc ← 2·acc
      row = r;
      let idx = (b1[i] + 2 * b2[i]) as usize;
      let tx = self.alloc(table[idx].0.clone());
      let ty = self.alloc(table[idx].1.clone());
      let (sx, sy, r) = self.ec_add(row, dx, dy, tx, ty); // acc ← acc + T_i
      row = r;
      ax = sx;
      ay = sy;
    }
    // R = acc − (2ⁿ−1)·SEED.
    let cx = self.alloc(neg_correction.0.clone());
    let cy = self.alloc(neg_correction.1.clone());
    let (rx, ry, r) = self.ec_add(row, ax, ay, cx, cy);
    row = r;
    (rx, ry, row)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A small generator-ish point on secp256k1 and its double, for testing.
  /// G = standard secp256k1 generator.
  fn g() -> (BigUint, BigUint) {
    let gx = BigUint::parse_bytes(
      b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
      16,
    )
    .unwrap();
    let gy = BigUint::parse_bytes(
      b"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
      16,
    )
    .unwrap();
    (gx, gy)
  }

  /// 2G (precomputed), for an add of two distinct points G + 2G = 3G.
  fn g2() -> (BigUint, BigUint) {
    let p = secp256k1_p();
    let (gx, gy) = g();
    // double G via affine doubling: λ = 3x²/(2y)
    let lam =
      (BigUint::from(3u32) * &gx * &gx % &p) * mod_inv(&(BigUint::from(2u32) * &gy % &p), &p) % &p;
    let x2 = (&lam * &lam + &p + &p - &gx - &gx) % &p;
    let y2 = (&lam * (&gx + &p - &x2) % &p + &p - &gy) % &p;
    (x2, y2)
  }

  #[test]
  fn ec_add_gadget_is_satisfied_and_correct() {
    let p = secp256k1_p();
    let (gx, gy) = g();
    let (g2x, g2y) = g2();
    let expected = ref_ec_add(&gx, &gy, &g2x, &g2y, &p); // 3G

    // Reserve 4 input columns + room; const col placed after a power-of-two pad.
    let const_col = 64usize; // generous; > all columns we'll allocate
    let mut cb = CircuitBuilder::new(const_col, p.clone());
    let x1 = cb.alloc(gx);
    let y1 = cb.alloc(gy);
    let x2 = cb.alloc(g2x);
    let y2 = cb.alloc(g2y);

    let (x3, y3, n_rows) = cb.ec_add(0, x1, y1, x2, y2);
    assert_eq!(n_rows, 9, "affine add should be 9 rows");

    // Output matches the reference 3G.
    assert_eq!(cb.w[x3], expected.0, "x3 mismatch");
    assert_eq!(cb.w[y3], expected.1, "y3 mismatch");

    // Self-check the full relation A·z ∘ B·z = C·z + m∘q on every row.
    let z = &cb.w; // z includes the const-1 at const_col
    for row in 0..cb.mods.len() {
      let lc = |entries: &[Triple]| -> BigUint {
        entries
          .iter()
          .filter(|(r, _, _)| *r == row)
          .map(|(_, col, k)| k * &z[*col])
          .sum()
      };
      let az = lc(&cb.a);
      let bz = lc(&cb.b);
      let cz = lc(&cb.c);
      assert_eq!(
        &az * &bz,
        &cz + &cb.mods[row] * &cb.q[row],
        "row {row} unsatisfied"
      );
    }
  }

  #[test]
  fn ec_double_gadget_is_satisfied_and_correct() {
    let p = secp256k1_p();
    let (gx, gy) = g();
    let expected = ref_ec_double(&gx, &gy, &p); // 2G

    let const_col = 64usize;
    let mut cb = CircuitBuilder::new(const_col, p.clone());
    let x = cb.alloc(gx);
    let y = cb.alloc(gy);
    let (x3, y3, n_rows) = cb.ec_double(0, x, y);
    assert_eq!(n_rows, 9, "affine double should be 9 rows");
    assert_eq!(cb.w[x3], expected.0, "x3 mismatch");
    assert_eq!(cb.w[y3], expected.1, "y3 mismatch");

    let z = &cb.w;
    for row in 0..cb.mods.len() {
      let lc = |entries: &[Triple]| -> BigUint {
        entries
          .iter()
          .filter(|(r, _, _)| *r == row)
          .map(|(_, col, k)| k * &z[*col])
          .sum()
      };
      assert_eq!(
        lc(&cb.a) * lc(&cb.b),
        lc(&cb.c) + &cb.mods[row] * &cb.q[row],
        "row {row} unsatisfied"
      );
    }

    // Sanity: 2G via the gadget equals 2G via direct doubling (g2()).
    assert_eq!((cb.w[x3].clone(), cb.w[y3].clone()), g2());
  }

  /// Generic point add over secp256k1 (handles identity/doubling). `None` = O.
  fn pt_add(
    a: &Option<(BigUint, BigUint)>,
    b: &Option<(BigUint, BigUint)>,
    p: &BigUint,
  ) -> Option<(BigUint, BigUint)> {
    match (a, b) {
      (None, _) => b.clone(),
      (_, None) => a.clone(),
      (Some(a), Some(b)) => {
        if a.0 == b.0 {
          if (&a.1 + &b.1) % p == BigUint::ZERO {
            None // a = −b
          } else {
            Some(ref_ec_double(&a.0, &a.1, p)) // a = b
          }
        } else {
          Some(ref_ec_add(&a.0, &a.1, &b.0, &b.1, p))
        }
      }
    }
  }

  /// Reference scalar multiplication `k·P` (double-and-add). `None` = O.
  fn scalar_mul(k: &BigUint, pt: &(BigUint, BigUint), p: &BigUint) -> Option<(BigUint, BigUint)> {
    let mut acc: Option<(BigUint, BigUint)> = None;
    for i in (0..k.bits()).rev() {
      acc = acc.map(|a| ref_ec_double(&a.0, &a.1, p));
      if (k >> i) & BigUint::from(1u32) == BigUint::from(1u32) {
        acc = pt_add(&acc, &Some(pt.clone()), p);
      }
    }
    acc
  }

  /// MSB-first `n`-bit decomposition of `k`.
  fn bits_msb(k: &BigUint, n: usize) -> Vec<u8> {
    (0..n)
      .rev()
      .map(|i| ((k >> i) & BigUint::from(1u32) == BigUint::from(1u32)) as u8)
      .collect()
  }

  /// Table `[SEED, SEED+G, SEED+Q, SEED+G+Q]` and `−(2ⁿ−1)·SEED`.
  fn msm_setup(
    g: &(BigUint, BigUint),
    q: &(BigUint, BigUint),
    seed: &(BigUint, BigUint),
    n: usize,
    p: &BigUint,
  ) -> ([(BigUint, BigUint); 4], (BigUint, BigUint)) {
    let unwrap = |o: Option<(BigUint, BigUint)>| o.unwrap();
    let p00 = seed.clone();
    let p10 = unwrap(pt_add(&Some(seed.clone()), &Some(g.clone()), p));
    let p01 = unwrap(pt_add(&Some(seed.clone()), &Some(q.clone()), p));
    let gq = unwrap(pt_add(&Some(g.clone()), &Some(q.clone()), p));
    let p11 = unwrap(pt_add(&Some(seed.clone()), &Some(gq), p));
    let mult = (BigUint::from(1u32) << n) - BigUint::from(1u32);
    let corr = scalar_mul(&mult, seed, p).unwrap();
    let neg = (corr.0.clone(), (p - &corr.1) % p);
    ([p00, p10, p01, p11], neg)
  }

  /// O(entries+rows) self-check of `A·z ∘ B·z = C·z + m∘q`.
  fn assert_relation(cb: &CircuitBuilder) {
    let nrows = cb.mods.len();
    let acc = |entries: &[Triple]| -> Vec<BigUint> {
      let mut v = vec![BigUint::ZERO; nrows];
      for (r, col, k) in entries {
        v[*r] += k * &cb.w[*col];
      }
      v
    };
    let (az, bz, cz) = (acc(&cb.a), acc(&cb.b), acc(&cb.c));
    for row in 0..nrows {
      assert_eq!(
        &az[row] * &bz[row],
        &cz[row] + &cb.mods[row] * &cb.q[row],
        "row {row} unsatisfied"
      );
    }
  }

  #[test]
  fn shamir_msm_8bit_correct_and_satisfied() {
    let p = secp256k1_p();
    let g_pt = g();
    let q_pt = scalar_mul(&BigUint::from(7u32), &g_pt, &p).unwrap(); // Q = 7G
    let seed = scalar_mul(&BigUint::from(11u32), &g_pt, &p).unwrap(); // SEED = 11G
    let n = 8;
    let u1 = BigUint::from(181u32);
    let u2 = BigUint::from(108u32);
    let (table, neg) = msm_setup(&g_pt, &q_pt, &seed, n, &p);
    let expected = pt_add(&scalar_mul(&u1, &g_pt, &p), &scalar_mul(&u2, &q_pt, &p), &p).unwrap();

    let mut cb = CircuitBuilder::new(1 << 12, p.clone());
    let (rx, ry, _) = cb.shamir_msm(0, &bits_msb(&u1, n), &bits_msb(&u2, n), &table, &neg);
    assert_eq!(cb.w[rx], expected.0, "Rx mismatch");
    assert_eq!(cb.w[ry], expected.1, "Ry mismatch");
    assert_relation(&cb);
  }

  /// Build the `n`-bit 2-scalar MSM circuit used by the benchmarks.
  fn build_msm_circuit(n: usize) -> CircuitBuilder {
    let p = secp256k1_p();
    let g_pt = g();
    let q_pt = scalar_mul(&BigUint::from(7u32), &g_pt, &p).unwrap();
    let seed = scalar_mul(&BigUint::from(11u32), &g_pt, &p).unwrap();
    let u1 = BigUint::parse_bytes(
      b"A1B2C3D4E5F60718293A4B5C6D7E8F90A1B2C3D4E5F60718293A4B5C6D7E8F90",
      16,
    )
    .unwrap();
    let u2 = BigUint::parse_bytes(
      b"0123456789ABCDEFFEDCBA98765432100123456789ABCDEFFEDCBA9876543210",
      16,
    )
    .unwrap();
    let (table, neg) = msm_setup(&g_pt, &q_pt, &seed, n, &p);
    let mut cb = CircuitBuilder::new(1 << 16, p);
    cb.shamir_msm(0, &bits_msb(&u1, n), &bits_msb(&u2, n), &table, &neg);
    cb
  }

  type ME = crate::provider::T256DynPrimeEngine;

  /// Convert a built circuit into an `IntModR1CSShapeModp` + witness `(w, q)`,
  /// remapping the constant column to `num_vars` and padding to powers of two.
  fn to_shape(
    cb: &CircuitBuilder,
  ) -> (
    crate::imod_r1cs_modp::IntModR1CSShapeModp<ME>,
    Vec<BigUint>,
    Vec<BigUint>,
  ) {
    let num_vars = cb.next_col.next_power_of_two();
    let num_cons = cb.mods.len().next_power_of_two();
    let cc = cb.const_col;
    let rm = |c: usize| if c == cc { num_vars } else { c };
    let map =
      |e: &[Triple]| -> Vec<Triple> { e.iter().map(|(r, c, v)| (*r, rm(*c), v.clone())).collect() };
    let (a, b, c) = (map(&cb.a), map(&cb.b), map(&cb.c));
    let mut w = cb.w[..cb.next_col].to_vec();
    w.resize(num_vars, BigUint::ZERO);
    let mut q = cb.q.clone();
    q.resize(num_cons, BigUint::ZERO);
    let mut mods = cb.mods.clone();
    mods.resize(num_cons, BigUint::from(2u32)); // pad rows: m=2, empty LCs ⇒ 0=0
    let shape =
      crate::imod_r1cs_modp::IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, 0, a, b, c, mods)
        .unwrap();
    (shape, w, q)
  }

  /// Isolation: a minimal hand-built mod-p shape (one `w0·w1 = w2 + p·q` row)
  /// through the same SNARK. If this verifies, the harness bug is in `to_shape`.
  #[test]
  #[ignore = "diagnostic"]
  fn snark_minimal_modp() {
    use crate::imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp};
    use crate::imod_spartan_modp::IntModSpartanModpSNARK;
    use crate::provider::pcs::integer_modpcs::IntEvalParams;
    let p = secp256k1_p();
    let num_vars = 8usize;
    let num_cons = 4usize;
    // row 0: w0·w1 = w2 (+ p·0). w0=3, w1=5, w2=15.
    let w: Vec<BigUint> = vec![
      BigUint::from(3u32),
      BigUint::from(5u32),
      BigUint::from(8u32), // w0+w1 = 8 (const-column diff-style row)
      BigUint::ZERO,
      BigUint::ZERO,
      BigUint::ZERO,
      BigUint::ZERO,
      BigUint::ZERO,
    ];
    let q = vec![BigUint::ZERO; num_cons];
    // diff-style row using the CONST column (z[num_vars]=1): (w0+w1)·1 = w2.
    // w2 set to 8 = 3+5 below via the witness override.
    let a = vec![
      (0usize, 0usize, BigUint::from(1u32)),
      (0usize, 1usize, BigUint::from(1u32)),
    ];
    let b = vec![(0usize, num_vars, BigUint::from(1u32))]; // const column
    let c = vec![(0usize, 2usize, BigUint::from(1u32))];
    let mut mods = vec![p.clone()];
    mods.resize(num_cons, BigUint::from(2u32));
    let shape = IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, 0, a, b, c, mods).unwrap();
    let params = IntEvalParams::derive(256, 64, 7, 3).unwrap();
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup_with_params(shape.clone(), params).unwrap();
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
    shape.is_sat(pk.ck(), &instance, &witness).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();
    println!("minimal mod-p SNARK: OK");
  }

  /// Isolation: ONE `ec_add` built via `CircuitBuilder` + `to_shape` → SNARK.
  /// If this fails (but `snark_minimal_modp` passes), the bug is in `to_shape`.
  #[test]
  #[ignore = "diagnostic"]
  fn snark_one_ec_add() {
    use crate::imod_r1cs_modp::IntModR1CSWitnessModp;
    use crate::imod_spartan_modp::IntModSpartanModpSNARK;
    use crate::provider::pcs::integer_modpcs::IntEvalParams;
    let p = secp256k1_p();
    let g_pt = g();
    let acc0 = scalar_mul(&BigUint::from(100u32), &g_pt, &p).unwrap(); // 100G
    let t = scalar_mul(&BigUint::from(7u32), &g_pt, &p).unwrap(); // 7G
    let mut cb = CircuitBuilder::new(1 << 16, p.clone());
    // 7 CHAINED rounds of (double; add T) with safe large-multiple points
    // (acc grows 100G→207G→…, never collides with 7G) — MSM depth, no correction.
    let mut ax = cb.alloc(acc0.0);
    let mut ay = cb.alloc(acc0.1);
    let mut row = 0;
    for _ in 0..2 {
      let (dx, dy, r) = cb.ec_double(row, ax, ay);
      row = r;
      let tx = cb.alloc(t.0.clone());
      let ty = cb.alloc(t.1.clone());
      let (sx, sy, r) = cb.ec_add(row, dx, dy, tx, ty);
      row = r;
      ax = sx;
      ay = sy;
    }
    let (shape, w, q) = to_shape(&cb);
    let log_n = (shape.num_vars() as u64).ilog2() as usize;
    let params = IntEvalParams::derive(256, 64, 9, log_n).unwrap();
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup_with_params(shape.clone(), params).unwrap();
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, pk.ck(), w, q, vec![]).unwrap();
    shape.is_sat(pk.ck(), &instance, &witness).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();
    println!("one ec_add via to_shape SNARK: OK");
  }

  /// Prover-time benchmark on the plain-Shamir MSM (Hyrax Mod-PCS baseline).
  /// `RAYON_NUM_THREADS=1 cargo test --release --lib -- --ignored --nocapture ecdsa_msm_prove_time`
  #[test]
  #[ignore = "benchmark; run with --release --ignored --nocapture"]
  fn ecdsa_msm_prove_time() {
    use crate::imod_r1cs_modp::IntModR1CSWitnessModp;
    use crate::imod_spartan_modp::IntModSpartanModpSNARK;
    use crate::provider::pcs::integer_modpcs::IntEvalParams;
    use std::time::Instant;

    let cb = build_msm_circuit(256); // diagnostic size; 256 for the real bench
    let real_rows = cb.mods.len();
    let (shape, w, q) = to_shape(&cb);
    let nv = shape.num_vars();
    let log_n = (nv.max(real_rows.next_power_of_two()) as u64).ilog2() as usize;
    println!(
      "\nECDSA 2-scalar MSM (secp256k1, affine Shamir, {real_rows} rows → 2^{}, threads={}):",
      (nv.max(real_rows.next_power_of_two()) as u64).ilog2(),
      rayon::current_num_threads()
    );
    for k in [7usize, 8, 9, 10, 11, 12] {
      let params = IntEvalParams::derive(256, 64, k, log_n).unwrap();
      let (sval, lpval, nl) = (params.s, params.log_p, params.numlimb);
      let (pk, vk) =
        IntModSpartanModpSNARK::<ME>::setup_with_params(shape.clone(), params).unwrap();
      // Witness commitment is part of the prover's work — cross-system
      // comparisons (e.g. Zinc+, whose prove commits internally) must
      // use commit + prove.
      let tc = Instant::now();
      let (witness, instance) =
        IntModR1CSWitnessModp::<ME>::new(&shape, pk.ck(), w.clone(), q.clone(), vec![]).unwrap();
      let commit_ms = tc.elapsed().as_secs_f64() * 1e3;
      let t0 = Instant::now();
      let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
      let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
      let t1 = Instant::now();
      proof.verify(&vk, &instance).unwrap();
      let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
      println!(
        "  k={k:<2} (s={sval}, log_p={lpval}, numlimb={nl}): commit {commit_ms:6.1} ms, \
         prove {prove_ms:7.1} ms, commit+prove {:7.1} ms, verify {verify_ms:.1} ms",
        commit_ms + prove_ms
      );
    }
  }

  #[test]
  fn shamir_msm_256bit_row_count() {
    let p = secp256k1_p();
    let g_pt = g();
    let q_pt = scalar_mul(&BigUint::from(7u32), &g_pt, &p).unwrap();
    let seed = scalar_mul(&BigUint::from(11u32), &g_pt, &p).unwrap();
    let n = 256;
    let u1 = BigUint::parse_bytes(
      b"A1B2C3D4E5F60718293A4B5C6D7E8F90A1B2C3D4E5F60718293A4B5C6D7E8F90",
      16,
    )
    .unwrap();
    let u2 = BigUint::parse_bytes(
      b"0123456789ABCDEFFEDCBA98765432100123456789ABCDEFFEDCBA9876543210",
      16,
    )
    .unwrap();
    let (table, neg) = msm_setup(&g_pt, &q_pt, &seed, n, &p);
    let expected = pt_add(&scalar_mul(&u1, &g_pt, &p), &scalar_mul(&u2, &q_pt, &p), &p).unwrap();

    let mut cb = CircuitBuilder::new(1 << 16, p.clone());
    let (rx, ry, _) = cb.shamir_msm(0, &bits_msb(&u1, n), &bits_msb(&u2, n), &table, &neg);
    assert_eq!(cb.w[rx], expected.0);
    assert_eq!(cb.w[ry], expected.1);
    assert_relation(&cb);

    let num_cons = cb.mods.len();
    println!(
      "\nECDSA 2-scalar MSM (256-bit secp256k1, affine, public addends): \
       {num_cons} constraint rows → pad to {}",
      num_cons.next_power_of_two()
    );
  }
}
