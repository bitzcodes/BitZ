use super::{PreparedSha256Ecdsa, Result, error};
use crate::{
    ligerito_flock::{OodRoundParams, grinding::GrindingPlan},
    piop::spartan::profile::log2_prime_count_lower_bound,
};

/// Soundness bound and grinding difficulty for a Fiat–Shamir challenge block.
/// A block may draw several field elements; repeated rounds with the same
/// bound and difficulty share one entry (for example, `outer-round`).
#[derive(Clone, Debug)]
pub struct ChallengeSecurity {
    /// Block identifier, such as `outer-round` or `prime+tau`, absorbed into
    /// the transcript together with its grinding difficulty and occurrence bound.
    pub label: String,
    /// Probability bound `e` for any soundness error in one occurrence,
    /// summing the block's bad events before grinding.
    pub failure_probability_bound: f64,
    /// Proof-of-work difficulty `g` per occurrence: about `2^g` nonce trials.
    /// The modeled economic security is `-log2(e) + g` bits; zero disables grinding.
    pub grinding_bits: u32,
    /// Occurrence bound `n` per proof, e.g. the number of outer sumcheck rounds.
    /// This entry contributes `n * e` to the statistical failure bound.
    pub max_occurrences: usize,
}

/// Security accounting and grinding settings derived from the prepared relation
/// and Ligerito configuration. The prover and verifier use the same settings.
/// Economic security includes grinding and is capped at 128 bits for BLAKE3;
/// statistical security uses the union bound without grinding.
#[derive(Clone, Debug)]
pub struct Sha256EcdsaSecurity {
    /// Requested economic security `prepared.lambda`, in bits, used to size grinding.
    /// The statistical security of the complete proof is computed separately.
    pub target: u32,
    /// Bounds, difficulties, and occurrence counts used by both security metrics.
    /// Repeated rounds share entries; vector order is the transcript's accounting
    /// order, not the order in which challenges are sampled.
    pub blocks: Vec<ChallengeSecurity>,
    /// Grinding bits before sampling the projection modulus `q` and the outer
    /// sumcheck's equality-test point. Corresponds to `prime+tau` in `blocks`.
    pub(crate) initial: u32,
    /// Grinding bits after the outer terminal claims, before sampling the matrix
    /// batching scalar, linear-equation row point, and linear batching weight.
    /// Corresponds to `rho+sigma+gamma` in `blocks`.
    pub(crate) batch: u32,
    /// Grinding bits before each outer sumcheck round challenge, after absorbing
    /// its round polynomial. Corresponds to `outer-round` in `blocks`.
    pub(crate) outer: u32,
    /// Grinding bits before each inner sumcheck round challenge, after absorbing
    /// its round polynomial. Corresponds to `inner-round` in `blocks`.
    pub(crate) inner: u32,
    /// Grinding bits per forest/bridge challenge in the final polynomial opening.
    /// Corresponds to `forest-and-bridge` in `blocks`.
    pub(crate) forest: u32,
    /// Per-block grinding settings for Ligerito's folding, introduction, and query
    /// challenges. Their security entries appear in `blocks` with a `flock/` prefix.
    pub(crate) flock: GrindingPlan,
    /// Grinding settings for the initial out-of-domain evaluation in Johnson mode.
    /// Its claim is bound just after the commitment and checked at the final opening.
    /// `None` in unique-decoding mode; corresponds to `step0:ood-draw` in `blocks`.
    pub(crate) ood: Option<OodRoundParams>,
}

impl Sha256EcdsaSecurity {
    /// Computes modeled attack work in bits: `min(128, min_i(-log2(e_i) + g_i))`.
    /// Here `e_i = blocks[i].failure_probability_bound` and `g_i = blocks[i].grinding_bits`.
    /// Repetition counts do not enter this per-block economic model.
    pub fn compute_economic_security_bits(&self) -> f64 {
        self.blocks
            .iter()
            .map(|b| -b.failure_probability_bound.log2() + f64::from(b.grinding_bits))
            .fold(128., f64::min)
    }
    /// Computes statistical security in bits: `-log2(sum_i n_i * e_i)`.
    /// Here `e_i = blocks[i].failure_probability_bound` and `n_i = blocks[i].max_occurrences`.
    /// The sum is a union bound on failure probability, excluding grinding.
    pub fn compute_statistical_security_bits(&self) -> f64 {
        -self
            .blocks
            .iter()
            .map(|b| b.failure_probability_bound * b.max_occurrences as f64)
            .sum::<f64>()
            .log2()
    }
    /// Builds failure bounds and grinding settings for projection, batching,
    /// sumcheck, and opening challenges from the relation and Ligerito config.
    /// For target `T = prepared.lambda` and raw failure bound `e_i`, each block gets
    /// `g_i = max(native_i, 0, ceil(T + log2(e_i)))` grinding bits, giving
    /// `-log2(e_i) + g_i >= T` in the per-block economic model.
    /// `native_i` is the configured Ligerito grinding, or zero for other blocks.
    /// Rejects requirements above 32 bits, or 24 bits for the initial OOD round.
    /// Repetition bounds are recorded for the separate statistical union bound.
    pub(crate) fn allocate_grinding_budget(prepared: &PreparedSha256Ecdsa) -> Result<Self> {
        let mut blocks = Vec::new();
        let mut add = |label: &str, error: f64, count: usize| -> Result<u32> {
            let bits = (f64::from(prepared.lambda) + error.log2()).ceil().max(0.) as u32;
            if bits > 32 {
                return Err(super::error(format!("{label} exceeds 32-bit grinding cap")));
            }
            blocks.push(ChallengeSecurity {
                label: label.into(),
                failure_probability_bound: error,
                grinding_bits: bits,
                max_occurrences: count,
            });
            Ok(bits)
        };
        let q_inv = 2f64.powi(-112);
        let divisors = prepared.local.defect_bits / 112;
        let bad_prime = f64::from(divisors) * 2f64.powf(-log2_prime_count_lower_bound(113));
        // The maximum all-row arity keeps the two benchmark modes on one schedule.
        let max_outer = (256 * prepared.compressions() + prepared.local.rows())
            .next_power_of_two()
            .ilog2() as usize;
        let initial = add("prime+tau", bad_prime + max_outer as f64 * q_inv, 1)?;
        let batch = add(
            "rho+sigma+gamma",
            (prepared.linear_vars() + 3) as f64 * q_inv,
            1,
        )?;
        let outer = add(
            "outer-round",
            3. * q_inv,
            prepared.outer_sumcheck_num_vars(),
        )?;
        let inner = add(
            "inner-round",
            2. * q_inv,
            prepared.h_layout.row_vars + prepared.h_layout.col_vars,
        )?;
        // Each host forest/bridge draw has degree at most the assignment arity
        // plus seven ring coordinates. 4096 bounds the number of draws for the
        // supported <=31-variable, single-chunk shapes (deliberately conservative).
        let forest = add(
            "forest-and-bridge",
            (prepared.h_layout.row_vars + prepared.h_layout.col_vars + 7) as f64 * 2f64.powi(-128),
            4096,
        )?;
        let ood = prepared
            .ligerito
            .ood_bits()
            .map(|bits| {
                let work = (f64::from(prepared.lambda) - bits).ceil().max(0.) as u32;
                if work > 24 {
                    return Err(error("Round-0 exceeds the 24-bit derived cap"));
                }
                blocks.push(ChallengeSecurity {
                    label: "step0:ood-draw".into(),
                    failure_probability_bound: 2f64.powf(-bits),
                    grinding_bits: work,
                    max_occurrences: 1,
                });
                Ok(OodRoundParams {
                    grinding_bits: work,
                })
            })
            .transpose()?;
        let flock =
            GrindingPlan::resolve(prepared.ligerito.security(), prepared.lambda).map_err(error)?;
        for b in &flock.blocks {
            blocks.push(ChallengeSecurity {
                label: format!("flock/{}", b.label),
                failure_probability_bound: b.raw_error,
                grinding_bits: b.bits,
                max_occurrences: 1,
            });
        }
        Ok(Self {
            target: prepared.lambda,
            blocks,
            initial,
            batch,
            outer,
            inner,
            forest,
            flock,
            ood,
        })
    }
}

impl PreparedSha256Ecdsa {
    pub fn security(&self) -> Result<Sha256EcdsaSecurity> {
        Sha256EcdsaSecurity::allocate_grinding_budget(self)
    }
}
