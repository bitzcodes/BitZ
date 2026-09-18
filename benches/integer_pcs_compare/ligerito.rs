//! BitZ's opener as a binary-field PCS on the same packed rows the Binius64
//! BaseFold adapter commits: one `GF(2^128)` row per gate, the same bit-MLE
//! claim at a transcript-derived point, opened by Round 0 + ring switching +
//! Johnson-regime Ligerito at rate 1/2 with fold and query grinding. The
//! opener's own union bound (Round 0, ring switch, every Ligerito level) is
//! gated at 100 bits, so the row is the PCS-only counterpart of the
//! `binius64-ligerito` end-to-end scheme and directly comparable with the
//! `binius64-basefold` row, which opens the identical claim.

#![allow(dead_code)]

use std::error::Error;

use bitz::binary_pcs::{BinaryPcs, LOG_INV_RATE, MIN_PACKED_LOG, write_round0};
use bitz::ligerito::LOG_PACKING;
use bitz::poly::univariate::binary_gf128::Gf128 as Gf;
use bitz::proof_codec::Writer;
use bitz::transcript::Blake3Transcript;
use bitz::transcript::traits::Transcript;
use flock_core::field::Gf128;

/// The whole-opener gate.
pub const SECURITY_BITS: usize = 100;
pub const COMMITMENT_BYTES: usize = 32;
pub const PACKING_BITS: usize = 128;
const MIN_COMPONENT_BITS: usize = 100;
const MAX_COMPONENT_BITS: usize = 112;

const DOMAIN_TAG: &[u8] = b"bitz/integer-pcs-compare/ligerito-binary/v1";
const CLAIM_TAG: &[u8] = b"bitz/integer-pcs-compare/ligerito-binary/claim/v1";

const ROOT_SCOPE: &str = "pcs-compare:verified_trial";
const MATERIALIZE_SCOPE: &str = "pcs-compare:materialize";
const COMMIT_SCOPE: &str = "pcs-compare:commit";
const CLAIM_SCOPE: &str = "pcs-compare:claim_setup";
const OPENING_SCOPE: &str = "pcs-compare:opening";
const VERIFY_SCOPE: &str = "pcs-compare:verification";

pub const COMMIT_CODEWORD_SCOPE: &str = "ligerito:commit_rs_codeword";
pub const ROUND0_SCOPE: &str = "ligerito:round0_ood_sample";
pub const SAMPLE_POINT_SCOPE: &str = "ligerito:sample_opening_point";
pub const EVALUATE_CLAIM_SCOPE: &str = "ligerito:evaluate_bit_mle";
pub const OPEN_SCOPE: &str = "ligerito:ring_switch_and_open";
pub const VERIFY_ROUND0_SCOPE: &str = "ligerito:verify_round0";
pub const VERIFY_OPEN_SCOPE: &str = "ligerito:verify_ring_switch_and_ligerito";

pub struct LigeritoBackend {
    pcs: BinaryPcs,
    log_rows: usize,
    soundness_bits: f64,
}

pub struct TrialOutput {
    pub proof_bytes: usize,
    pub commitment_bytes: usize,
    pub public_claim_bytes: usize,
}

impl TrialOutput {
    pub fn opening_proof_bytes(&self) -> usize {
        self.proof_bytes.saturating_sub(self.commitment_bytes)
    }
    pub fn total_wire_bytes(&self) -> usize {
        self.proof_bytes + self.public_claim_bytes
    }
}

impl LigeritoBackend {
    /// Solve the opener at the smallest round-by-round target whose own
    /// union bound (Round 0, ring switch, every level) clears the gate.
    pub fn setup(log_rows: usize) -> Result<Self, Box<dyn Error>> {
        if log_rows < MIN_PACKED_LOG {
            return Err(format!(
                "the Ligerito opener needs at least 2^{MIN_PACKED_LOG} rows, got 2^{log_rows}"
            )
            .into());
        }
        let mut last = None;
        for target in MIN_COMPONENT_BITS..=MAX_COMPONENT_BITS {
            let pcs = match BinaryPcs::new(log_rows, target) {
                Ok(pcs) => pcs,
                Err(error) => {
                    last = Some(error.to_string());
                    continue;
                }
            };
            let total = pcs.error_sum() + BinaryPcs::ring_switch_term().error_bound;
            let bits = -total.log2();
            if bits >= SECURITY_BITS as f64 {
                return Ok(Self {
                    pcs,
                    log_rows,
                    soundness_bits: bits,
                });
            }
        }
        Err(format!(
            "no opener target in {MIN_COMPONENT_BITS}..={MAX_COMPONENT_BITS} reaches {SECURITY_BITS} bits{}",
            last.map(|e| format!(" ({e})")).unwrap_or_default()
        )
        .into())
    }

    pub const fn log_rows(&self) -> usize {
        self.log_rows
    }
    pub const fn log_inv_rate(&self) -> usize {
        LOG_INV_RATE
    }
    pub fn component_bits(&self) -> usize {
        self.pcs.component_bits()
    }
    /// Level-0 codeword queries (the counterpart of BaseFold's test queries).
    pub fn n_test_queries(&self) -> usize {
        self.pcs.level0_queries()
    }
    pub fn level0_fold_grinding_bits(&self) -> usize {
        self.pcs.level0_fold_grinding_bits()
    }
    pub fn level0_query_grinding_bits(&self) -> usize {
        self.pcs.level0_query_grinding_bits()
    }
    pub fn ood_grinding_bits(&self) -> u32 {
        self.pcs.ood_grinding_bits()
    }
    /// `-log2` of the opener's whole union bound.
    pub fn soundness_bits(&self) -> f64 {
        self.soundness_bits
    }

    /// One independently verified opening with the canonical comparison
    /// scopes. `packed_rows` must contain exactly `2^log_rows` rows.
    pub fn run_trial(
        &self,
        materialize_rows: impl FnOnce() -> Vec<u128>,
        trial_seed: u64,
    ) -> Result<TrialOutput, Box<dyn Error>> {
        let root = tracing::info_span!(ROOT_SCOPE).entered();
        let packed = {
            let _phase = tracing::info_span!(MATERIALIZE_SCOPE).entered();
            let rows = materialize_rows();
            if rows.len() != 1usize << self.log_rows {
                return Err(format!(
                    "Ligerito row count {} does not match setup shape 2^{}",
                    rows.len(),
                    self.log_rows
                )
                .into());
            }
            rows.into_iter()
                .map(|row| Gf128 {
                    lo: row as u64,
                    hi: (row >> 64) as u64,
                })
                .collect::<Vec<_>>()
        };

        let (commitment, data, round0, mut pt) = {
            let _phase = tracing::info_span!(COMMIT_SCOPE).entered();
            let (commitment, data) = {
                let _procedure = tracing::info_span!(COMMIT_CODEWORD_SCOPE).entered();
                self.pcs.commit(&packed)?
            };
            let mut pt = seed_transcript(&commitment.root, trial_seed);
            let round0 = {
                let _procedure = tracing::info_span!(ROUND0_SCOPE).entered();
                self.pcs.prove_round0(&mut pt, &packed)
            };
            (commitment, data, round0, pt)
        };

        let (point, value) = {
            let _phase = tracing::info_span!(CLAIM_SCOPE).entered();
            let point = {
                let _procedure = tracing::info_span!(SAMPLE_POINT_SCOPE).entered();
                sample_point(&mut pt, self.log_rows)
            };
            let value = {
                let _procedure = tracing::info_span!(EVALUATE_CLAIM_SCOPE).entered();
                evaluate_bit_mle(&packed, &point)
            };
            absorb_claim(&mut pt, value);
            (point, value)
        };

        let (opening, bytes) = {
            let _phase = tracing::info_span!(OPENING_SCOPE).entered();
            let opening = {
                let _procedure = tracing::info_span!(OPEN_SCOPE).entered();
                self.pcs
                    .open_bit_mle(&mut pt, &packed, &data, &round0, &point)
            };
            let mut w = Writer::new();
            w.bytes(&commitment.root);
            write_round0(&mut w, &round0.round());
            opening.write(&mut w);
            (opening, w.into_vec())
        };
        drop(data);

        {
            let _phase = tracing::info_span!(VERIFY_SCOPE).entered();
            let mut vt = seed_transcript(&commitment.root, trial_seed);
            let round0 = {
                let _procedure = tracing::info_span!(VERIFY_ROUND0_SCOPE).entered();
                self.pcs.verify_round0(&mut vt, &round0.round())?
            };
            let verifier_point = sample_point(&mut vt, self.log_rows);
            if verifier_point != point {
                return Err("Ligerito verifier did not replay the opening point".into());
            }
            absorb_claim(&mut vt, value);
            let _procedure = tracing::info_span!(VERIFY_OPEN_SCOPE).entered();
            self.pcs.verify_bit_mle(
                &mut vt,
                &commitment.root,
                &round0,
                &verifier_point,
                value,
                &opening,
            )?;
        }
        std::hint::black_box(value);
        drop(root);

        let public_claim_bytes = (self.log_rows + LOG_PACKING + 1) * 16;
        Ok(TrialOutput {
            proof_bytes: bytes.len(),
            commitment_bytes: COMMITMENT_BYTES,
            public_claim_bytes,
        })
    }
}

fn seed_transcript(root: &[u8; 32], seed: u64) -> Blake3Transcript {
    let mut t = Blake3Transcript::new();
    t.absorb_slice(DOMAIN_TAG);
    t.absorb_slice(&seed.to_le_bytes());
    t.absorb_slice(root);
    t
}

fn sample_point(t: &mut Blake3Transcript, log_rows: usize) -> Vec<Gf> {
    (0..log_rows + LOG_PACKING)
        .map(|_| t.get_field_challenge::<Gf>(&()))
        .collect()
}

fn absorb_claim(t: &mut Blake3Transcript, value: Gf) {
    t.absorb_slice(CLAIM_TAG);
    let words = value.as_words();
    t.absorb_slice(&words[0].to_le_bytes());
    t.absorb_slice(&words[1].to_le_bytes());
}

fn eq_table(point: &[Gf]) -> Vec<Gf> {
    let mut out = vec![Gf::one()];
    for &r in point {
        let n = out.len();
        out.resize(2 * n, Gf::zero());
        for i in 0..n {
            let high = out[i] * r;
            out[n + i] = high;
            out[i] = out[i] + high;
        }
    }
    out
}

/// `MLE[bits(packed)](point)`: bit `j` of row `i` sits at index `128·i + j`.
/// The seven bit coordinates are folded through a byte lookup table (sixteen
/// lookups per row instead of 128 multiplications), the row coordinates
/// through an `eq` table.
pub fn evaluate_bit_mle(packed: &[Gf128], point: &[Gf]) -> Gf {
    let low = eq_table(&point[..LOG_PACKING]);
    let high = eq_table(&point[LOG_PACKING..]);
    let mut table = vec![Gf::zero(); 16 * 256];
    for byte in 0..16 {
        for val in 1usize..256 {
            table[byte * 256 + val] = table[byte * 256 + (val & (val - 1))]
                + low[byte * 8 + val.trailing_zeros() as usize];
        }
    }
    let mut acc = Gf::zero();
    for (word, &h) in packed.iter().zip(&high) {
        let bytes = ((word.lo as u128) | ((word.hi as u128) << 64)).to_le_bytes();
        let folded = bytes
            .iter()
            .enumerate()
            .fold(Gf::zero(), |s, (i, &b)| s + table[i * 256 + b as usize]);
        acc = acc + folded * h;
    }
    acc
}

/// Same claim, evaluated the slow way; the fast path must agree.
pub fn evaluate_bit_mle_reference(packed: &[Gf128], point: &[Gf]) -> Gf {
    let low = eq_table(&point[..LOG_PACKING]);
    let high = eq_table(&point[LOG_PACKING..]);
    let mut acc = Gf::zero();
    for (i, word) in packed.iter().enumerate() {
        let bits = word.lo as u128 | ((word.hi as u128) << 64);
        for (j, &l) in low.iter().enumerate() {
            if bits >> j & 1 == 1 {
                acc = acc + l * high[i];
            }
        }
    }
    acc
}

pub fn commitment_root_gf(root: &[u8; 32]) -> Gf {
    let lo = u64::from_le_bytes(root[..8].try_into().expect("8 bytes"));
    let hi = u64::from_le_bytes(root[8..16].try_into().expect("8 bytes"));
    (Gf128 { lo, hi })
}
