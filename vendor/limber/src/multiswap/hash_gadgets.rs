//! Poseidon and MiMC-7 gadgets over the row builder. Linear layers
//! (round-key additions, the MDS matrix, sponge absorption) are folded
//! into linear combinations and cost nothing; every S-box multiplication
//! is one `mod p` row. A partial round's S-box input is a combination
//! over every earlier S-box output, so it is materialized (one `mod p`
//! row) once it grows past a few terms to keep the matrices sparse.

use super::circuit::{Builder, Lc, Var};
use super::mimc;
use super::poseidon::PoseidonParams;
use num_bigint::BigUint;
use num_traits::Zero;

/// Materialize `lc` (mod `p`) when it has more than `max_terms` terms.
fn compact(b: &mut Builder, lc: Lc, p: &BigUint, max_terms: usize) -> Lc {
  if lc.len() > max_terms {
    Lc::var(b.reduce_mod(&lc, p))
  } else {
    lc
  }
}

/// `x^5 mod p` as three rows; returns the output variable.
fn sbox(b: &mut Builder, x: &Lc, p: &BigUint) -> Var {
  let x2 = b.mul_mod(x, x, p);
  let x4 = b.mul_mod(&Lc::var(x2), &Lc::var(x2), p);
  b.mul_mod(&Lc::var(x4), x, p)
}

/// One Poseidon permutation over a symbolic state (`t` combinations,
/// reduced mod `p`); returns the output state as combinations.
pub fn poseidon_permute(b: &mut Builder, params: &PoseidonParams, state: Vec<Lc>) -> Vec<Lc> {
  let p = &params.p;
  let t = params.t;
  assert_eq!(state.len(), t);
  let pre = params.pre_full_rounds();
  let mut state = state;
  let round = |b: &mut Builder, state: &mut Vec<Lc>, key: &[BigUint], full: bool, last: bool| {
    // Add round keys.
    let with_keys: Vec<Lc> = state
      .iter()
      .zip(key)
      .map(|(s, k)| s.clone().add_const(k).normalize_mod(p))
      .collect();
    // S-boxes.
    let outs: Vec<Lc> = with_keys
      .into_iter()
      .enumerate()
      .map(|(i, x)| {
        if full || i == 0 {
          let x = compact(b, x, p, 8);
          Lc::var(sbox(b, &x, p))
        } else {
          x
        }
      })
      .collect();
    if last {
      *state = outs;
      return;
    }
    // MDS layer as combinations.
    *state = (0..t)
      .map(|i| {
        let mut acc = Lc::default();
        for (m, o) in params.mds_row(i).iter().zip(&outs) {
          acc = acc.plus(&o.scale(m));
        }
        acc.normalize_mod(p)
      })
      .collect();
  };
  for r in 0..pre {
    round(b, &mut state, params.full_key(r), true, false);
  }
  for r in 0..params.r_p {
    round(b, &mut state, params.partial_key(r), false, false);
  }
  for r in pre..params.r_f {
    round(b, &mut state, params.full_key(r), true, r == params.r_f - 1);
  }
  state
}

/// The Poseidon sponge hash of `inputs` (combinations); returns the
/// output as a witness reduced mod `p`. Mirrors
/// [`PoseidonParams::hash`]: initial zero-state permutation (a constant
/// here), absorb `rate` inputs per cycle, output word 0.
pub fn poseidon_hash(b: &mut Builder, params: &PoseidonParams, inputs: &[Lc]) -> Var {
  let rate = params.rate();
  let cycles = inputs.len().div_ceil(rate).max(1);
  let mut init = vec![BigUint::zero(); params.t];
  params.permute(&mut init);
  let mut state: Vec<Lc> = init.into_iter().map(Lc::constant).collect();
  for c in 0..cycles {
    for (i, slot) in state.iter_mut().enumerate().take(rate) {
      if let Some(x) = inputs.get(c * rate + i) {
        *slot = slot.clone().plus(x).normalize_mod(&params.p);
      }
    }
    state = poseidon_permute(b, params, state);
  }
  let out = b.reduce_mod(&state[0], &params.p);
  debug_assert_eq!(
    b.value(out),
    params.hash(
      &inputs
        .iter()
        .map(|x| b.eval(x) % &params.p)
        .collect::<Vec<_>>()
    )
  );
  out
}

/// The MiMC-7 permutation `x ← (x + k_i)^7` (four `mod p` rows per
/// round); returns the output witness.
pub fn mimc_permute(b: &mut Builder, keys: &[BigUint], x: &Lc, p: &BigUint) -> Var {
  let mut x = x.clone();
  let mut out = None;
  for k in keys.iter().take(mimc::ROUNDS) {
    let y = x.add_const(k).normalize_mod(p);
    let y2 = b.mul_mod(&y, &y, p);
    let y4 = b.mul_mod(&Lc::var(y2), &Lc::var(y2), p);
    let y6 = b.mul_mod(&Lc::var(y4), &Lc::var(y2), p);
    let y7 = b.mul_mod(&Lc::var(y6), &y, p);
    x = Lc::var(y7);
    out = Some(y7);
  }
  out.expect("rounds > 0")
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::multiswap::poseidon::bls12_381_r;

  #[test]
  fn poseidon_gadget_matches_native() {
    let params = PoseidonParams::bls12_381_owwb20();
    let mut b = Builder::new();
    let inputs: Vec<BigUint> = (1..=7u32).map(BigUint::from).collect();
    let lcs: Vec<Lc> = inputs.iter().map(|v| Lc::var(b.alloc(v.clone()))).collect();
    let rows0 = b.num_rows();
    let out = poseidon_hash(&mut b, &params, &lcs);
    assert_eq!(b.value(out), params.hash(&inputs));
    // Two permutations: 2 × 105 S-boxes × 3 rows, plus materializations
    // and the output reduction.
    let rows = b.num_rows() - rows0;
    assert!(
      (2 * 105 * 3..=2 * 105 * 3 + 2 * 64 + 1).contains(&rows),
      "rows = {rows}"
    );
  }

  #[test]
  fn mimc_gadget_matches_native() {
    let p = bls12_381_r();
    let keys = mimc::round_keys(&p);
    let mut b = Builder::new();
    let x = b.alloc(BigUint::from(42u32));
    let out = mimc_permute(&mut b, &keys, &Lc::var(x), &p);
    assert_eq!(
      b.value(out),
      mimc::permute(&BigUint::from(42u32), &keys, &p)
    );
    assert_eq!(b.num_rows(), 4 * 91);
  }
}
