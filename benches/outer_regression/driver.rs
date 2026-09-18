//! Same inputs, timing boundary and verification on both sides of the refactor.
//! Only adapter.rs differs between the historical and current builds.
use super::{
    OuterSumcheckProof, SpartanField,
    raw_monty::{NativeProducts, NativeWideProducts},
    univariate_skip::UnivariateSkipOuterSumcheckProof,
};
use crate::transcript::Blake3Transcript;
use field::{Fp, FpCtx, IntegerEmbedding, RingOps, Uint, WideMul};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{hint::black_box, time::Instant};
mod adapter;
type Elem = Fp<2>;
type Field = FpCtx<2>;

pub(super) enum Inputs {
    U32 {
        a: Vec<u64>,
        b: Vec<u64>,
        c: Vec<u64>,
        narrow_a: Vec<u32>,
        narrow_b: Vec<u32>,
    },
    U64 {
        a: Vec<u64>,
        b: Vec<u64>,
        lo: Vec<u64>,
        hi: Vec<u64>,
        c: Vec<u128>,
    },
    U128 {
        a: Vec<u128>,
        b: Vec<u128>,
        lo: Vec<u128>,
        hi: Vec<u128>,
        c: Vec<Uint<4>>,
    },
}
impl Inputs {
    fn new(bits: u32, n: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed ^ n as u64 ^ u64::from(bits));
        // Include boundaries among otherwise full-width uniform operands.
        let a: Vec<u128> = (0..1usize << n)
            .map(|i| match i {
                0 => 0,
                1 => 1,
                2 => u128::MAX,
                _ => rng.random(),
            })
            .collect();
        let b: Vec<u128> = (0..a.len())
            .map(|i| if i < 3 { u128::MAX } else { rng.random() })
            .collect();
        match bits {
            32 => {
                let a: Vec<u64> = a.iter().map(|&v| v as u32 as u64).collect();
                let b: Vec<u64> = b.iter().map(|&v| v as u32 as u64).collect();
                Self::U32 {
                    c: a.iter().zip(&b).map(|(a, b)| a * b).collect(),
                    narrow_a: a.iter().map(|&v| v as u32).collect(),
                    narrow_b: b.iter().map(|&v| v as u32).collect(),
                    a,
                    b,
                }
            }
            64 => {
                let a: Vec<u64> = a.iter().map(|&v| v as u64).collect();
                let b: Vec<u64> = b.iter().map(|&v| v as u64).collect();
                let c: Vec<u128> = a
                    .iter()
                    .zip(&b)
                    .map(|(&a, &b)| a as u128 * b as u128)
                    .collect();
                Self::U64 {
                    lo: c.iter().map(|&v| v as u64).collect(),
                    hi: c.iter().map(|&v| (v >> 64) as u64).collect(),
                    a,
                    b,
                    c,
                }
            }
            128 => {
                let c: Vec<Uint<4>> = a
                    .iter()
                    .zip(&b)
                    .map(|(&a, &b)| {
                        *field::IntegerOps
                            .mul_wide(&Uint::<2>::from(a), &Uint::<2>::from(b))
                            .checked_resize_ct::<4>()
                            .value()
                    })
                    .collect();
                let lo = c
                    .iter()
                    .map(|v| v.as_words()[0] as u128 | ((v.as_words()[1] as u128) << 64))
                    .collect();
                let hi = c
                    .iter()
                    .map(|v| v.as_words()[2] as u128 | ((v.as_words()[3] as u128) << 64))
                    .collect();
                Self::U128 { a, b, c, lo, hi }
            }
            _ => panic!("unsupported width {bits}"),
        }
    }
    fn fixture_digest(&self) -> String {
        let mut hash = blake3::Hasher::new();
        let mut row = |a: u128, b: u128, c: [u64; 4]| {
            hash.update(&a.to_le_bytes());
            hash.update(&b.to_le_bytes());
            for word in c {
                hash.update(&word.to_le_bytes());
            }
        };
        match self {
            Self::U32 { a, b, c, .. } => {
                for i in 0..a.len() {
                    row(a[i] as u128, b[i] as u128, [c[i], 0, 0, 0]);
                }
            }
            Self::U64 { a, b, c, .. } => {
                for i in 0..a.len() {
                    row(
                        a[i] as u128,
                        b[i] as u128,
                        [c[i] as u64, (c[i] >> 64) as u64, 0, 0],
                    );
                }
            }
            Self::U128 { a, b, c, .. } => {
                for i in 0..a.len() {
                    row(a[i], b[i], *c[i].as_words());
                }
            }
        }
        hash.finalize().to_hex().to_string()
    }
}

pub(super) enum Proof {
    Ordinary(OuterSumcheckProof<Elem>),
    Skip(UnivariateSkipOuterSumcheckProof<Elem>),
}
impl Proof {
    fn verify_and_hash(&self, field: &Field, tau: &[Elem], n: usize, digest: [u8; 32]) -> String {
        let mut verifier = Blake3Transcript::new();
        let mut hasher = blake3::Hasher::new();
        let mut hash = |x: &Elem| {
            hasher.update(&x.canonical_element_encoding(field));
        };
        let outer = match self {
            Self::Ordinary(p) => {
                p.verify(&mut verifier, field.zero(), tau, field).unwrap();
                p
            }
            Self::Skip(p) => {
                p.verify(&mut verifier, tau, n, field).unwrap();
                for x in p
                    .skip
                    .finite_q_evaluations
                    .iter()
                    .chain(std::iter::once(&p.skip.q_at_infinity))
                {
                    hash(x)
                }
                &p.tail
            }
        };
        assert_eq!(digest, verifier.state_digest());
        for x in outer.sumcheck.round_polynomials.iter().flatten().chain([
            &outer.az_mle_claim,
            &outer.bz_mle_claim,
            &outer.cz_mle_claim,
        ]) {
            hash(x)
        }
        hasher.update(&digest);
        hasher.finalize().to_hex().to_string()
    }
}

/// Paired trials preserve the distinct preparation boundaries and verify identical proofs.
pub fn samples(
    bits: u32,
    n: usize,
    seed: u64,
    k: usize,
    trials: usize,
) -> (String, Vec<(f64, f64)>) {
    let field = Fp::<2>::make_cfg(&Uint::from((1u128 << 100) - 15)).unwrap();
    let input = Inputs::new(bits, n, seed);
    let fixture = input.fixture_digest();
    let tau: Vec<_> = (0..n - k)
        .map(|i| field.from_integer(&(i as u64 + 7)))
        .collect();
    let prepared = adapter::prepare(&field, k);
    let mut expected = None;
    let mut samples = Vec::new();
    for sample in 0..trials {
        let mut times = [0.; 2];
        for offset in 0..2 {
            let index = (sample + offset) % 2;
            let mut transcript = Blake3Transcript::new();
            adapter::reset_measurements();
            let start = Instant::now();
            let proof = if index == 0 {
                adapter::production(&input, &field, &mut transcript, &tau, n, k)
            } else {
                adapter::generic(&input, &field, &mut transcript, &tau, k, &prepared)
            };
            times[index] = start.elapsed().as_secs_f64() * 1000.;
            let fingerprint = proof.verify_and_hash(&field, &tau, n, transcript.state_digest());
            if let Some(expected) = &expected {
                assert_eq!(&fingerprint, expected);
            } else {
                expected = Some(fingerprint);
            }
            black_box(proof);
        }
        samples.push((times[0], times[1]));
    }
    (fixture, samples)
}
