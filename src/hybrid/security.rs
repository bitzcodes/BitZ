//! Union-bound accounting for the supported composition shapes.
use crate::piop::spartan::mul::MulLayout;
use crate::piop::spartan::protocol::PreparedRelationPrefix;

use super::{Error, opening::Geometry};
use crate::piop::spartan::profile::{IopSecurityProfile, PrimePolicy};

/// Slack for the complete composition: the standalone 100-bit per-round
/// preset cannot be reused as a 100-bit whole-protocol guarantee.
///
/// `LIGERITO_TARGET_BITS` governs the multiplication relation's own
/// standalone (unique-decoding, fold-grinded) opener, which the `separate`
/// mode runs; the SHARED opener of the hybrid proof is configured by
/// [`LIGERITO_COMPONENT_BITS`] instead.
pub struct CompositionProfile;
impl IopSecurityProfile for CompositionProfile {
    const NAME: &'static str = "hybrid100-components108";
    const LAMBDA: u32 = 108;
    const PRIME_POLICY: PrimePolicy = PrimePolicy::SingleDerived;
    const LIGERITO_TARGET_BITS: usize = 112;
    const FOREST_ROUND_GRINDING_BITS: u32 = 0;
    const RING_SWITCH_GRINDING_BITS: u32 = 0;
}

/// Round-by-round target of the SHARED opener at the default rate 1/2: the
/// Johnson-regime Ligerito configuration of the virtual geometry (see
/// [`Geometry::security`]; the proximity-gap figures quoted below were
/// derived at the earlier rate 1/8 — at rate 1/2 the same solver re-derives
/// the ladder at this target). It is deliberately below the 108-bit
/// component profile: in the Johnson regime the level-0 proximity-gap
/// bound sits near 86 bits for 2^19 multiplications (84 at 2^21), so the
/// fold-challenge grinding that tops it up is exponential in the target —
/// about 2^21 hash evaluations per level-0 fold at 106 bits and 2^27 at
/// 112 — where the unique-decoding opener the profile constant was set for
/// pays a few thousand. 106 is the smallest target at which the measured
/// shapes clear the 100-bit composition gate with margin: the opener's
/// terms sum to about 22·2^-target at five levels, so 104 fails the gate
/// and 105 leaves it less than half a bit.
///
/// This constant governs the RATE-1/2 opener only (keeping its transcripts
/// byte-identical). A rate-1/8 selection solves the smallest component
/// target in 100..=112 whose whole-protocol union bound clears the 100-bit
/// gate instead (`PreparedHybrid::new_with_ligerito`), the same
/// smallest-clearing rule `src/binius_ligerito` applies.
pub const LIGERITO_COMPONENT_BITS: usize = 106;

/// The whole-protocol union-bound gate every prepared composition must clear.
pub(super) const GATE_BITS: f64 = 100.0;

#[derive(Clone, Debug)]
pub struct SecurityTerm {
    pub name: &'static str,
    pub error_bound: f64,
}

/// Algebraic/IOP error accounting, using the pinned implementations' soundness
/// analyses and grinding model. Fiat–Shamir uses BLAKE3, and both Merkle trees
/// use its 256-bit output (128-bit generic collision resistance). This is not
/// a claim of an unconditional 100-bit Fiat–Shamir security theorem.
#[derive(Clone, Debug)]
pub struct SecurityReport {
    pub target_bits: u32,
    pub algebraic_bits: f64,
    pub terms: Vec<SecurityTerm>,
}

impl SecurityReport {
    /// The term with the largest error bound (the binding term).
    pub fn binding_term(&self) -> Option<&SecurityTerm> {
        self.terms
            .iter()
            .max_by(|a, b| a.error_bound.total_cmp(&b.error_bound))
    }
}

pub(super) fn account(
    mul: &PreparedRelationPrefix<MulLayout<u32>>,
    sha: &binius_verifier::IOPVerifier,
    geometry: &Geometry,
    resolved: &crate::ligerito_flock::ResolvedLigerito,
) -> Result<SecurityReport, Error> {
    let report = account_terms(mul, sha, geometry, resolved)?;
    if report.algebraic_bits < GATE_BITS {
        return Err(Error::Invalid("composition does not reach 100 bits"));
    }
    Ok(report)
}

/// The union-bound accounting WITHOUT the 100-bit gate, so a caller solving
/// the smallest clearing component target can probe candidates.
pub(super) fn account_terms(
    mul: &PreparedRelationPrefix<MulLayout<u32>>,
    sha: &binius_verifier::IOPVerifier,
    geometry: &Geometry,
    resolved: &crate::ligerito_flock::ResolvedLigerito,
) -> Result<SecurityReport, Error> {
    let mut terms = Vec::new();
    let mut add = |name, error_bound| terms.push(SecurityTerm { name, error_bound });
    // Round 0 of the shared opening: the same event, bound and grinding
    // rule as `IopSecurityParams::adopt_ood_round` uses for the standalone
    // relations (`step0:ood-draw`, `grinding = ceil(λ − bits)` under
    // `MAX_DERIVED_GRINDING_BITS`). In flock's own analysis level 0 is
    // pinned to one list element only by the final opening's evaluation
    // claim; that is too late for the forest and PIOP challenges drawn in
    // between, so this explicit sample replaces that implicit binding.
    if let Some((ood_bits, ood)) = super::opening::ood_parameters(resolved)? {
        add(
            "step0:ood-draw",
            2f64.powf(-(ood_bits + f64::from(ood.grinding_bits))),
        );
    }
    let gate_log = mul.layout().gate_vars();
    for term in &mul.security().accounting.terms {
        // These are precisely the integer-prefix stages retained here.
        let count = match term.name {
            "step2:projection-draw" | "step3:tau-draw" | "step4:terminal-draw" => 1,
            "step3:piop-round" => 2 * gate_log + 8,
            _ if term.name.starts_with("step2:") => 1,
            _ => continue,
        };
        add(term.name, count as f64 * 2f64.powf(-term.bits));
    }
    let p = mul.params();
    let depth = p.row_vars + p.word_bits.trailing_zeros() as usize;
    let k_inv = 2f64.powi(-128);
    // Two sumchecks per GKR layer, with degrees at most three, plus the
    // closing child randomization. Overcount all rounds by depth+s+4.
    add(
        "multiplication GKR",
        (4 * depth * (depth + p.col_vars + 4)) as f64 * k_inv,
    );
    let cs = sha.constraint_system();
    let sha_dims = sha.log_witness_words()
        + cs.log_and_constraints().unwrap_or(0)
        + cs.log_zero_constraints().unwrap_or(0)
        + 64;
    // AND's univariate degree is <=126 (64-bit word domain), with three
    // skipped Bit coordinates. The subsequent zerocheck/shift rounds
    // have degree <=3. 4096 per coordinate bounds the initial identity
    // tests, skipped rounds, all operand batches and their sumchecks.
    // This also covers public-segment ring switching and its degree-two
    // sumcheck: its packed coordinate count is bounded by log_witness_words.
    // The wiring polynomial is checked natively, adding no new soundness term.
    // No IMUL/BMUL auxiliary protocols are allowed in this SHA circuit.
    add("SHA PIOP", (4096 * sha_dims) as f64 * k_inv);
    // The joint sumcheck's rounds plus its batching draw, plus the
    // Round-0 batching draw `η_ood` into the opening basis.
    add(
        "joint sumcheck and batching",
        (2 * geometry.bit_log() + 2) as f64 * k_inv,
    );
    add("ring switching", 128.0 * k_inv);
    add(
        "logical support padding",
        (geometry.packed_log() + 1) as f64 * k_inv,
    );
    let config = resolved.security();
    config.validate().map_err(Error::Config)?;
    for (index, level) in config.levels.iter().enumerate() {
        let (pg, query) = level.paper_predicted_bits();
        add(
            "Ligerito proximity folds",
            level.k_recursive as f64 * 2f64.powf(-pg - level.fold_grinding_bits as f64),
        );
        add(
            "Ligerito queries",
            2f64.powf(-query - level.grinding_bits as f64),
        );
        // Deeper levels bind their list element with explicit out-of-domain
        // samples inside the opener; level 0's binding is Round 0 above.
        if index > 0 {
            if let Some(bits) = level.paper_predicted_ood_bits() {
                add("Ligerito OOD samples", 2f64.powf(-bits));
            }
        }
    }
    add(
        "Ligerito field rounds",
        (16 * geometry.packed_log() + 128) as f64 * k_inv,
    );
    let algebraic_bits = -terms
        .iter()
        .map(|term| term.error_bound)
        .sum::<f64>()
        .log2();
    Ok(SecurityReport {
        target_bits: 100,
        algebraic_bits,
        terms,
    })
}
