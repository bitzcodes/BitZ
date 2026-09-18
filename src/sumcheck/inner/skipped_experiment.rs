//! Test-only experiment: full degree-(2*2^K-2) prefix, quadratic tail, and
//! two shared-challenge bridges back to ordinary multilinear openings.
//! This deliberately retains the original tables so the bridge cost is counted.
use super::*;
use crate::sumcheck::UngrindedRoundBoundary;
use field::{Fp, FpCtx, IntegerEmbedding};

type E = Fp<2>;
type Ctx = FpCtx<2>;
struct Proof {
    prefix: Vec<E>,
    tail: SumcheckProof<E, 3>,
    tail_values: [E; 2],
    bridge: [SumcheckProof<E, 3>; 2],
    terminal: [E; 2],
}

fn evaluate(f: &Ctx, p: &[E], x: E) -> E {
    p.iter()
        .rev()
        .fold(f.zero(), |a, c| f.add(&f.mul(&a, &x), c))
}
fn basis(f: &Ctx, n: usize, x: E) -> Vec<E> {
    let nodes: Vec<_> = (0..n).map(|i| f.from_integer(&(i as u64))).collect();
    (0..n)
        .map(|i| {
            let (a, b) = (0..n)
                .filter(|&j| j != i)
                .fold((f.one(), f.one()), |(a, b), j| {
                    (
                        f.mul(&a, &f.sub(&x, &nodes[j])),
                        f.mul(&b, &f.sub(&nodes[i], &nodes[j])),
                    )
                });
            f.mul(&a, f.inverse_ct(&b).value())
        })
        .collect()
}
fn interpolate(f: &Ctx, ys: &[E]) -> Vec<E> {
    let n = ys.len();
    let nodes: Vec<_> = (0..n).map(|i| f.from_integer(&(i as u64))).collect();
    let mut out = vec![f.zero(); n];
    for i in 0..n {
        let mut p = vec![f.one()];
        let mut d = f.one();
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut next = vec![f.zero(); p.len() + 1];
            for (k, c) in p.iter().enumerate() {
                next[k] = f.sub(&next[k], &f.mul(c, &nodes[j]));
                next[k + 1] = f.add(&next[k + 1], c);
            }
            p = next;
            d = f.mul(&d, &f.sub(&nodes[i], &nodes[j]));
        }
        let scale = f.mul(&ys[i], f.inverse_ct(&d).value());
        for (o, c) in out.iter_mut().zip(p) {
            *o = f.add(o, &f.mul(&scale, &c));
        }
    }
    out
}
fn fold_prefix(f: &Ctx, table: &[E], weights: &[E]) -> Vec<E> {
    table
        .chunks_exact(weights.len())
        .map(|row| field::Reduce::reduce(f, f.batch_mul_acc(weights, row)))
        .collect()
}
fn suffix_evaluations(f: &Ctx, table: &[E], width: usize, point: &[E]) -> Vec<E> {
    let mut table = table.to_vec();
    for r in point {
        let next = table.len() / 2;
        for i in 0..next {
            let block = i / width;
            let col = i % width;
            let a = table[2 * block * width + col];
            let b = table[(2 * block + 1) * width + col];
            table[i] = f.add(&a, &f.mul(r, &f.sub(&b, &a)));
        }
        table.truncate(next);
    }
    table
}
fn frame(f: &Ctx, t: &mut impl Transcript, n: usize, k: usize, claim: E) {
    t.absorb_slice(b"bitz/inner/skip/experimental/v1");
    t.absorb_slice(&(n as u64).to_le_bytes());
    t.absorb_slice(&(k as u64).to_le_bytes());
    absorb_field_elements(t, &[claim], f);
}
fn prove(
    f: &Ctx,
    t: &mut impl Transcript,
    claim: E,
    values: &[E],
    weights: &[E],
    k: usize,
) -> Result<(Proof, Vec<E>), SumcheckError> {
    let n = values.len();
    if !(1..=4).contains(&k) || !n.is_power_of_two() || weights.len() != n || 1usize << k > n {
        return Err(SumcheckError::InvalidProductDimensions);
    }
    let m = 1 << k;
    // q(z)=sum_u W_z(u)*V_z(u). The prefix is not a known-zero polynomial.
    let evaluations: Vec<_> = (0..2 * m - 1)
        .map(|i| {
            let l = basis(f, m, f.from_integer(&(i as u64)));
            let w = fold_prefix(f, weights, &l);
            let v = fold_prefix(f, values, &l);
            field::Reduce::reduce(f, f.batch_mul_acc(&w, &v))
        })
        .collect();
    if evaluations[..m].iter().fold(f.zero(), |s, x| f.add(&s, x)) != claim {
        return Err(SumcheckError::InvalidTerminalClaim);
    }
    let prefix = interpolate(f, &evaluations);
    frame(f, t, n, k, claim);
    absorb_field_elements(t, &prefix, f);
    let z = squeeze_field(t, f)?;
    let l = basis(f, m, z);
    let tail = prove_inner_sumcheck(
        f,
        t,
        evaluate(f, &prefix, z),
        fold_prefix(f, values, &l),
        fold_prefix(f, weights, &l),
        &mut UngrindedRoundBoundary,
    )?;
    let tail_values = tail.terminal_evaluations;
    absorb_field_elements(t, &tail_values, f);
    // Each tail terminal is sum_s L_s(z)*original(s, r_tail). A second
    // ordinary sumcheck proves these two equalities at one Boolean-prefix point.
    let bridge = prove_batched_inner_sumcheck(
        f,
        t,
        &tail_values,
        [
            suffix_evaluations(f, weights, m, &tail.point),
            suffix_evaluations(f, values, m, &tail.point),
        ],
        [l.clone(), l],
        &mut UngrindedRoundBoundary,
    )?;
    let terminal = bridge.terminal_evaluations.map(|[_, v]| v);
    let point = bridge.point.into_iter().chain(tail.point).collect();
    Ok((
        Proof {
            prefix,
            tail: tail.proof,
            tail_values,
            bridge: bridge.proofs,
            terminal,
        },
        point,
    ))
}
impl Proof {
    fn verify(
        &self,
        f: &Ctx,
        t: &mut impl Transcript,
        claim: E,
        n: usize,
        k: usize,
    ) -> Result<(Vec<E>, [E; 2]), SumcheckError> {
        if !(1..=4).contains(&k)
            || !n.is_power_of_two()
            || 1usize << k > n
            || self.prefix.len() != 2 * (1usize << k) - 1
        {
            return Err(SumcheckError::InvalidProductDimensions);
        }
        let m = 1 << k;
        let sum = (0..m).fold(f.zero(), |a, i| {
            f.add(&a, &evaluate(f, &self.prefix, f.from_integer(&(i as u64))))
        });
        if sum != claim {
            return Err(SumcheckError::InvalidRoundClaim { round: 0 });
        }
        frame(f, t, n, k, claim);
        absorb_field_elements(t, &self.prefix, f);
        let z = squeeze_field(t, f)?;
        let (tail_point, tail_claim) =
            self.tail
                .verify(t, evaluate(f, &self.prefix, z), n.ilog2() as usize - k, f)?;
        if tail_claim != f.mul(&self.tail_values[0], &self.tail_values[1]) {
            return Err(SumcheckError::InvalidTerminalClaim);
        }
        absorb_field_elements(t, &self.tail_values, f);
        let (prefix_point, claims) = SumcheckProof::verify_batch_with_round_boundary(
            [&self.bridge[0], &self.bridge[1]],
            t,
            &self.tail_values,
            k,
            f,
            &mut UngrindedRoundBoundary,
        )?;
        let mut l = basis(f, m, z);
        for r in &prefix_point {
            let next = l.len() / 2;
            for i in 0..next {
                l[i] = f.add(&l[2 * i], &f.mul(r, &f.sub(&l[2 * i + 1], &l[2 * i])));
            }
            l.truncate(next);
        }
        for i in 0..2 {
            if claims[i] != f.mul(&l[0], &self.terminal[i]) {
                return Err(SumcheckError::InvalidTerminalClaim);
            }
        }
        // The caller must open both original MLEs at this returned point.
        Ok((
            prefix_point.into_iter().chain(tail_point).collect(),
            self.terminal,
        ))
    }
}

#[test]
fn full_prefix_bridge_checks_original_multilinear_openings_and_tampering() {
    let f = super::tests::field();
    for k in 1..=4 {
        for extra in [0, 2] {
            let n = 1 << (k + extra);
            let w: Vec<_> = (0..n)
                .map(|i| f.from_integer(&(i as u64 * 17 + 1)))
                .collect();
            let v: Vec<_> = (0..n).map(|i| f.from_integer(&((i % 7) as u64))).collect();
            let claim = field::Reduce::reduce(&f, f.batch_mul_acc(&w, &v));
            let mut pt = crate::transcript::Blake3Transcript::new();
            let (mut proof, point) = prove(&f, &mut pt, claim, &v, &w, k).unwrap();
            let mut vt = crate::transcript::Blake3Transcript::new();
            let verified = proof.verify(&f, &mut vt, claim, n, k).unwrap();
            assert_eq!(verified.0, point);
            assert_eq!(
                verified.1,
                [
                    suffix_evaluations(&f, &w, 1, &point)[0],
                    suffix_evaluations(&f, &v, 1, &point)[0]
                ]
            );
            assert_eq!(
                squeeze_field::<E, _>(&mut pt, &f),
                squeeze_field::<E, _>(&mut vt, &f)
            );
            proof.terminal[0] = f.add(&proof.terminal[0], &f.one());
            assert!(
                proof
                    .verify(
                        &f,
                        &mut crate::transcript::Blake3Transcript::new(),
                        claim,
                        n,
                        k
                    )
                    .is_err()
            );
            proof.terminal[0] = f.sub(&proof.terminal[0], &f.one());
            proof.prefix[0] = f.add(&proof.prefix[0], &f.one());
            assert!(
                proof
                    .verify(
                        &f,
                        &mut crate::transcript::Blake3Transcript::new(),
                        claim,
                        n,
                        k
                    )
                    .is_err()
            );
        }
    }
}

#[test]
#[ignore = "full skipped-inner experiment including prefix preparation and opening bridge"]
fn full_path_measurement() {
    let f = super::tests::field();
    let n = 1 << 15;
    let w: Vec<_> = (0..n)
        .map(|i| f.from_integer(&(i as u64 * 17 + 1)))
        .collect();
    let v: Vec<_> = (0..n).map(|i| f.from_integer(&((i % 7) as u64))).collect();
    let claim = field::Reduce::reduce(&f, f.batch_mul_acc(&w, &v));
    for k in 0..=4 {
        let start = std::time::Instant::now();
        let mut t = crate::transcript::Blake3Transcript::new();
        let mut verify_t = crate::transcript::Blake3Transcript::new();
        if k == 0 {
            let out = prove_inner_sumcheck(
                &f,
                &mut t,
                claim,
                v.clone(),
                w.clone(),
                &mut UngrindedRoundBoundary,
            )
            .unwrap();
            let elapsed = start.elapsed();
            out.proof.verify(&mut verify_t, claim, 15, &f).unwrap();
            println!(
                "INNER_EXPERIMENT k={k} full_prove_ms={:.3} verified=true",
                elapsed.as_secs_f64() * 1000.
            );
        } else {
            let (p, point) = prove(&f, &mut t, claim, &v, &w, k).unwrap();
            let elapsed = start.elapsed();
            let (_, terminal) = p.verify(&f, &mut verify_t, claim, n, k).unwrap();
            assert_eq!(
                terminal,
                [
                    suffix_evaluations(&f, &w, 1, &point)[0],
                    suffix_evaluations(&f, &v, 1, &point)[0]
                ]
            );
            println!(
                "INNER_EXPERIMENT k={k} full_prove_ms={:.3} verified=true",
                elapsed.as_secs_f64() * 1000.
            );
        }
    }
}
