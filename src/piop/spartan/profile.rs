//! The configurable IOP security profile: one compile-time policy type per
//! security target, one runtime instantiation per shape, one derivation for
//! every interval width and grinding difficulty, and a per-term soundness
//! accounting that names the binding term.
//!
//! # Two layers
//!
//! - **Compile-time policy** ([`IopSecurityProfile`], zero-sized types):
//!   the target `LAMBDA`, the prime strategy (one transcript prime vs the
//!   two-prime Strategy 2), the Ligerito/WHIR target, and the forest/GKR +
//!   ring-switch grinding difficulties.
//! - **Runtime instantiation** ([`IopSecurityProfile::instantiate`] against
//!   [`IopInstanceFacts`]): the concrete interval endpoints and per-boundary
//!   grinding bits, *derived* — not tabulated — from the policy plus the
//!   shape facts, validated against the hard geometric constraints, and
//!   accompanied by the full [`SoundnessAccounting`]. Construction fails
//!   when any term this crate controls cannot reach `LAMBDA`.
//!
//! # Derivation rules (paper §Instantiation + the former `prime.rs` docs)
//!
//! With a `b`-bit prime interval `[2^(b-1), 2^b)` (so `q >= 2^(b-1)`):
//!
//! - a degree-`d` PIOP round message carries error `d/q`; grinding tops the
//!   term up to `λ`: `round_grind = ceil(λ + log2(d) - (b-1))⁺`.
//! - the τ/terminal batching draws carry error `arity/q`:
//!   `initial_grind = ceil(λ + log2(arity) - (b-1))⁺`, `terminal = round`.
//! - a projection draw against a nonzero integer defect `< 2^D` hits a bad
//!   prime with probability `floor(D/(b-1)) / |P_b|`, with `|P_b|` the
//!   conservative prime-count lower bound below.
//! - the interval width is `b = min(113, 126, 128 - a, c_w)` — the paper's
//!   112–113-bit Strategy-1 field policy, the transcript sampler's `2^126`
//!   operating cap, the no-wrap lift bound `(2^a + 1)(q-1) <= 2^128 - 1`,
//!   and (for a direct opening) the one-chunk exponent-fold width
//!   `c_w = 127 - t - W`. The full-width `[2^127, 2^128)` fingerprint of
//!   Strategy 2 bypasses the width rule and uses its dedicated sampler.
//! - each forest/GKR sumcheck round over `K = GF(2^128)` carries `3/|K|`
//!   (`~2^-126.4`): `forest_grind = ceil(λ - 126.415)⁺`; the ring-switch
//!   round carries `1/|K|`: `rs_grind = ceil(λ + 1 - 128)⁺` (the paper's
//!   "two bits per round / one bit" rule at λ = 128).
//!
//! These rules **reproduce the historical parameter tables exactly** — the SHA
//! `initial 20|21|22 / outer 18|19` schedule and MultiSwap's
//! `[2^112, 2^113)` + 10-bit reduction grind. Transcript domains bind the
//! selected profile explicitly, so matching parameters do not imply proof-byte
//! compatibility with an older protocol version.
//!
//! # The GF(2^128) floor
//!
//! Flock-internal field-limited rounds (each degree-`d` sumcheck message,
//! error `~d/2^128`) sit at ~126.4 bits and no parameter reaches past
//! them. The accounting carries them as an explicit `floor` term:
//! validation ignores floor terms (nothing this crate configures can move
//! them), but [`SoundnessAccounting::achieved_bits`] includes them, so a
//! `Lambda128` build reports **achieved ≈ 126.4, binding: the GF(2^128)
//! floor** rather than silently claiming 128.

use field::RingOps;
use thiserror::Error;

use crate::ligerito_flock::OodRoundParams;

/// Number of transcript primes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimePolicy {
    /// One transcript prime carries the PIOP and the opening (Strategy 1).
    SingleDerived,
    /// Strategy 2: a full-width `[2^127, 2^128)` fingerprint prime for the
    /// PIOP plus a Step-5.0 reduction prime for the opening.
    TwoFullWidthFingerprint,
}

/// Per-instance facts the derivation consumes. All are public statement
/// quantities.
#[derive(Clone, Copy, Debug)]
pub struct IopInstanceFacts {
    /// `⌈log₂⌉` upper bound on any nonzero integer defect the Step-2
    /// projection draw must catch (0 = the defect is smaller than any
    /// sampled prime, so no bad prime exists).
    pub defect_log2_bound: u32,
    /// `log₂` of the Step-5.1 lift arity `n₁`: the no-wrap bound is
    /// `(2^a + 1)(q - 1) <= 2^128 - 1`.
    pub lift_arity_log2: u32,
    /// Row variables `t` of the tensor the BitZ opening runs against.
    pub opening_t: u32,
    /// Word width `W` of that tensor.
    pub opening_word_bits: u32,
    /// Whether the opening prime feeds the DIRECT exponent-fold opener
    /// (then `q_bits <= c_w = 127 - t - W` is enforced). The virtual
    /// opening path caps its fold width from `q_bits` instead.
    pub direct_opening: bool,
    /// Worst union arity of one q-sized challenge draw (the zerocheck τ
    /// point; per-coordinate error `1/q`, `arity` coordinates).
    pub tau_arity: u32,
    /// Maximum degree of a PIOP sumcheck round message.
    pub piop_degree: u32,
    /// `⌈log₂⌉` magnitude bound of the Step-5.0 lift difference
    /// (`d·Q²`-class); only read under Strategy 2.
    pub step50_magnitude_log2: u32,
}

/// Step-5.0 reduction-prime parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReductionPrimeParams {
    /// Inclusive interval endpoints.
    pub min: u128,
    pub max: u128,
    /// Proof-of-work bits immediately before the draw.
    pub grinding_bits: u32,
}

/// One soundness term of the round-by-round accounting.
#[derive(Clone, Debug)]
pub struct SoundnessTerm {
    /// Step-keyed name (`docs/bench-schema.md` vocabulary).
    pub name: &'static str,
    /// `-log₂` of the term's error, including its grinding.
    pub bits: f64,
    /// Grinding bits spent on this term.
    pub grinding_bits: u32,
    /// `true` = outside this crate's control (the GF(2^128) floor);
    /// excluded from target validation, included in `achieved_bits`.
    pub floor: bool,
}

/// The complete per-term accounting of one instantiated profile.
#[derive(Clone, Debug)]
pub struct SoundnessAccounting {
    /// The profile's target.
    pub target: u32,
    pub terms: Vec<SoundnessTerm>,
}

impl SoundnessAccounting {
    /// `-log₂` of the largest error term — the honest end-to-end figure,
    /// floors included.
    pub fn achieved_bits(&self) -> f64 {
        self.terms
            .iter()
            .map(|term| term.bits)
            .fold(f64::INFINITY, f64::min)
    }

    /// The term with the largest error.
    pub fn binding_term(&self) -> &SoundnessTerm {
        self.terms
            .iter()
            .min_by(|left, right| left.bits.total_cmp(&right.bits))
            .expect("an accounting always carries terms")
    }

    /// The weakest term this crate's parameters control (floors excluded).
    pub fn controllable_bits(&self) -> f64 {
        self.terms
            .iter()
            .filter(|term| !term.floor)
            .map(|term| term.bits)
            .fold(f64::INFINITY, f64::min)
    }
}

/// The validated runtime parameter set of one profile at one shape.
#[derive(Clone, Debug)]
pub struct IopSecurityParams {
    /// The originating profile's name (bound into statements).
    pub profile_name: &'static str,
    /// The target λ.
    pub lambda: u32,
    /// Inclusive Step-2 projection/fingerprint prime interval.
    pub projection_min: u128,
    pub projection_max: u128,
    /// `true` = the `[2^127, 2^128)` fingerprint (dedicated sampler).
    pub projection_full_width: bool,
    /// Proof-of-work bits before the projection draw (0 = no boundary).
    pub initial_grinding_bits: u32,
    /// Proof-of-work bits before each PIOP outer-round challenge.
    pub piop_round_grinding_bits: u32,
    /// Proof-of-work bits before the terminal opening challenges.
    pub terminal_grinding_bits: u32,
    /// Step-5.0 reduction draw; `None` = the step does not fire.
    pub reduction: Option<ReductionPrimeParams>,
    /// Proof-of-work bits at every forest/GKR sumcheck round boundary.
    pub forest_round_grinding_bits: u32,
    /// Proof-of-work bits at the ring-switch round.
    pub ring_switch_grinding_bits: u32,
    /// Ligerito/WHIR round-by-round target.
    pub ligerito_target_bits: usize,
    /// Round 0 (the out-of-domain sample) of the BitZ opening: `Some` when
    /// the selected opener runs beyond unique decoding (its list must be
    /// pinned before the first forest challenge), `None` in the
    /// unique-decoding regime. Set by [`IopSecurityParams::adopt_ood_round`]
    /// once the relation has chosen its opener.
    pub ood: Option<OodRoundParams>,
    /// The full per-term accounting.
    pub accounting: SoundnessAccounting,
}

impl IopSecurityParams {
    /// Accounts for Round 0 of the BitZ opening once the opener is known.
    /// `ood_bits` is the theorem's collision bound in bits
    /// ([`crate::ligerito_flock::ood_round_bits`]): `None` (unique
    /// decoding) leaves the round off; otherwise the grinding topping it up
    /// to the target is derived under the usual economic cap and the term
    /// `step0:ood-draw` joins the accounting.
    pub fn adopt_ood_round(&mut self, ood_bits: Option<f64>) -> Result<(), ProfileError> {
        self.accounting
            .terms
            .retain(|term| term.name != "step0:ood-draw");
        let Some(bits) = ood_bits else {
            self.ood = None;
            return Ok(());
        };
        let grinding_bits = ceil_pos(f64::from(self.lambda) - bits);
        if grinding_bits > MAX_DERIVED_GRINDING_BITS {
            return Err(ProfileError::GrindingTooExpensive {
                profile: self.profile_name,
                term: "step0:ood-draw",
                bits: grinding_bits,
                cap: MAX_DERIVED_GRINDING_BITS,
            });
        }
        self.accounting.terms.push(SoundnessTerm {
            name: "step0:ood-draw",
            bits: bits + f64::from(grinding_bits),
            grinding_bits,
            floor: false,
        });
        self.ood = Some(OodRoundParams { grinding_bits });
        Ok(())
    }
}

/// Compile-time IOP security policy. Zero-sized marker types implement
/// this; everything runtime lives in [`IopSecurityParams`].
pub trait IopSecurityProfile: 'static {
    /// Name bound into statements and reports.
    const NAME: &'static str;
    /// Target security bits for every term this crate controls.
    const LAMBDA: u32;
    /// One transcript prime or the two-prime Strategy 2.
    const PRIME_POLICY: PrimePolicy;
    /// Ligerito/WHIR round-by-round target (`>= LAMBDA`; 128 is the
    /// audited production configuration).
    const LIGERITO_TARGET_BITS: usize;
    /// Per-round forest/GKR grinding (`derive_forest_grinding(LAMBDA)` for
    /// the paper rule; 0 disables this grinding boundary).
    const FOREST_ROUND_GRINDING_BITS: u32;
    /// Ring-switch round grinding (`derive_ring_switch_grinding(LAMBDA)`).
    const RING_SWITCH_GRINDING_BITS: u32;
    /// Reference-schedule escape hatch: historical design targets may skip
    /// strict `>= LAMBDA` validation while still reporting full accounting.
    /// Never set this for a production profile.
    const DESIGN_ONLY: bool = false;

    /// Derives and validates the runtime parameters for one shape.
    fn instantiate(facts: &IopInstanceFacts) -> Result<IopSecurityParams, ProfileError> {
        derive_params(
            Self::NAME,
            Self::LAMBDA,
            Self::PRIME_POLICY,
            Self::LIGERITO_TARGET_BITS,
            Self::FOREST_ROUND_GRINDING_BITS,
            Self::RING_SWITCH_GRINDING_BITS,
            Self::DESIGN_ONLY,
            facts,
        )
    }
}

/// The default target: near-zero grinding, every controllable term
/// `>= 100` bits (paper: "achieving λ = 100 is relatively straightforward").
pub struct Lambda100;

impl IopSecurityProfile for Lambda100 {
    const NAME: &'static str = "lambda100";
    const LAMBDA: u32 = 100;
    const PRIME_POLICY: PrimePolicy = PrimePolicy::SingleDerived;
    const LIGERITO_TARGET_BITS: usize = 100;
    const FOREST_ROUND_GRINDING_BITS: u32 = derive_forest_grinding(100);
    const RING_SWITCH_GRINDING_BITS: u32 = derive_ring_switch_grinding(100);
}

/// Every term this crate controls `>= 128` bits: the SHA-path grinding
/// schedule, two bits per forest round, one bit at the ring switch, the
/// 128-bit Ligerito target. The GF(2^128) floor (~126.4, flock-internal
/// untracked rounds) still binds end to end and is reported as such.
pub struct Lambda128;

impl IopSecurityProfile for Lambda128 {
    const NAME: &'static str = "lambda128";
    const LAMBDA: u32 = 128;
    const PRIME_POLICY: PrimePolicy = PrimePolicy::SingleDerived;
    const LIGERITO_TARGET_BITS: usize = 128;
    const FOREST_ROUND_GRINDING_BITS: u32 = derive_forest_grinding(128);
    const RING_SWITCH_GRINDING_BITS: u32 = derive_ring_switch_grinding(128);
}

/// Matched MultiSwap target, with a full-width fingerprint and a
/// shape-derived reduction grind. Fixed field checks can exceed this target.
pub struct Limber112;

impl IopSecurityProfile for Limber112 {
    const NAME: &'static str = "limber112";
    const LAMBDA: u32 = 112;
    const PRIME_POLICY: PrimePolicy = PrimePolicy::TwoFullWidthFingerprint;
    const LIGERITO_TARGET_BITS: usize = 112;
    const FOREST_ROUND_GRINDING_BITS: u32 = derive_forest_grinding(112);
    const RING_SWITCH_GRINDING_BITS: u32 = derive_ring_switch_grinding(112);
}

/// The MultiSwap/Limber comparison target: Strategy 2 with the full-width
/// fingerprint and the 10-bit Step-5.0 grind, matching the floors Limber's
/// own implementation accepts (`LAMBDA_BOUND2 = 117`, fingerprint
/// `~2^-114`), and a 114-bit Ligerito target (the MultiSwap opener is the
/// validator-gated `udrg:3:4:114`, see `multiswap_lig_configs`). Pinned by
/// the published comparison table.
pub struct Limber114;

impl IopSecurityProfile for Limber114 {
    const NAME: &'static str = "limber114";
    const LAMBDA: u32 = 114;
    const PRIME_POLICY: PrimePolicy = PrimePolicy::TwoFullWidthFingerprint;
    const LIGERITO_TARGET_BITS: usize = 114;
    const FOREST_ROUND_GRINDING_BITS: u32 = derive_forest_grinding(114);
    const RING_SWITCH_GRINDING_BITS: u32 = derive_ring_switch_grinding(114);
}

/// Historical SHA-256 reference schedule: 128-bit projection, Spartan, and
/// Ligerito targets, but no forest or ring-switch grinding. It is retained as
/// an explicit comparison schedule and reaches the ~126.4-bit GKR-round floor;
/// it is not the default security profile.
pub struct Sha128ReferenceSchedule;

impl IopSecurityProfile for Sha128ReferenceSchedule {
    const NAME: &'static str = "sha128-reference-schedule";
    const LAMBDA: u32 = 128;
    const PRIME_POLICY: PrimePolicy = PrimePolicy::SingleDerived;
    const LIGERITO_TARGET_BITS: usize = 128;
    const FOREST_ROUND_GRINDING_BITS: u32 = 0;
    const RING_SWITCH_GRINDING_BITS: u32 = 0;
    const DESIGN_ONLY: bool = true;
}

/// `ceil(λ - 126.415)⁺`: the paper's two-bits-per-GKR-round rule at 128.
pub const fn derive_forest_grinding(lambda: u32) -> u32 {
    // Each GKR round: 3/2^128 = 2^-126.415…; integer-safe ceiling of
    // λ - (128 - log2(3)) via λ - 126 adjusted for the 0.415 fraction.
    if lambda <= 126 {
        0
    } else {
        // λ = 127 → need 0.585 → 1; λ = 128 → need 1.585 → 2.
        lambda - 126
    }
}

/// `ceil(λ + 1 - 128)⁺`: the paper's one-bit ring-switch rule at 128
/// (`1/|K| = 2^-128` needs strict clearance at λ = 128).
pub const fn derive_ring_switch_grinding(lambda: u32) -> u32 {
    if lambda >= 128 { lambda - 127 } else { 0 }
}

/// Failures of profile instantiation.
#[derive(Debug, Error)]
pub enum ProfileError {
    /// The exponent-fold geometry rejects the shape outright.
    #[error("exponent-fold geometry needs t + W <= 126, got t={t}, W={word_bits}")]
    ShapeTooWide { t: u32, word_bits: u32 },

    /// No prime interval satisfies every width constraint.
    #[error("no admissible prime interval: width would be {width_bits} bits")]
    EmptyPrimeInterval { width_bits: i64 },

    /// A term this crate controls cannot reach the target.
    #[error(
        "profile {profile}: term `{term}` reaches only {bits:.1} bits, \
         below the λ = {lambda} target"
    )]
    TargetUnreachable {
        profile: &'static str,
        term: &'static str,
        bits: f64,
        lambda: u32,
    },

    /// A grinding requirement above this bound means the configuration is
    /// economically wrong, not merely slow.
    #[error(
        "profile {profile}: term `{term}` would need {bits} grinding bits \
         (cap {cap}); use a larger field or Strategy 2"
    )]
    GrindingTooExpensive {
        profile: &'static str,
        term: &'static str,
        bits: u32,
        cap: u32,
    },
}

/// Largest single grinding difficulty a profile may derive (2^24 hashes
/// ≈ 150 ms multi-threaded — the paper's own economic cutoff).
pub const MAX_DERIVED_GRINDING_BITS: u32 = 24;

/// Conservative lower bound on `log₂ |{primes in [2^(b-1), 2^b)}|`:
/// `π(x) > x/ln x` above and `π(x) < 1.26·x/ln x` below. Reproduces the
/// former doc-comment floors (b=128 → 120.07; b=113 → 105.25).
pub fn log2_prime_count_lower_bound(interval_bits: u32) -> f64 {
    let b = f64::from(interval_bits);
    let ln2 = core::f64::consts::LN_2;
    // count >= 2^b/(b·ln2) - 1.26·2^(b-1)/((b-1)·ln2), factored as
    // 2^b · (1/(b·ln2) - 0.63/((b-1)·ln2)).
    let density = 1.0 / (b * ln2) - 0.63 / ((b - 1.0) * ln2);
    assert!(density > 0.0, "prime-count bound needs b >= 5");
    b + density.log2()
}

fn ceil_pos(value: f64) -> u32 {
    if value <= 0.0 { 0 } else { value.ceil() as u32 }
}

/// `-log₂` error of a projection draw against a `< 2^defect` defect from a
/// `b`-bit interval; `None` when no bad prime can exist.
fn projection_draw_bits(defect_log2: u32, interval_bits: u32) -> Option<f64> {
    let divisors = defect_log2 / (interval_bits - 1);
    if divisors == 0 {
        return None;
    }
    Some(log2_prime_count_lower_bound(interval_bits) - f64::from(divisors).log2())
}

#[allow(clippy::too_many_arguments)]
fn derive_params(
    profile_name: &'static str,
    lambda: u32,
    policy: PrimePolicy,
    ligerito_target_bits: usize,
    forest_round_grinding_bits: u32,
    ring_switch_grinding_bits: u32,
    design_only: bool,
    facts: &IopInstanceFacts,
) -> Result<IopSecurityParams, ProfileError> {
    let lambda_f = f64::from(lambda);
    if facts.opening_t + facts.opening_word_bits > 126 {
        return Err(ProfileError::ShapeTooWide {
            t: facts.opening_t,
            word_bits: facts.opening_word_bits,
        });
    }
    let c_w = 127 - facts.opening_t - facts.opening_word_bits;
    let mut terms = Vec::new();

    // The width of a derived (non-fingerprint) interval: the Strategy-1
    // field policy (113), the sampler cap (126), the no-wrap lift bound,
    // and — for a direct opening — the one-chunk fold width.
    let derived_width = |direct: bool| -> Result<u32, ProfileError> {
        let mut width = 113.min(126).min(128 - i64::from(facts.lift_arity_log2));
        if direct {
            width = width.min(i64::from(c_w));
        }
        if width < 8 {
            return Err(ProfileError::EmptyPrimeInterval { width_bits: width });
        }
        Ok(u32::try_from(width).expect("width in [8, 126]"))
    };
    // Endpoints: dyadic interval, top capped by the exact no-wrap bound
    // (2^a + 1)(q - 1) <= 2^128 - 1 (the former Sha256PrimeProfile rule).
    let interval_endpoints = |width: u32| -> (u128, u128) {
        let min = 1u128 << (width - 1);
        let no_wrap_max = 1 + u128::MAX / ((1u128 << facts.lift_arity_log2) + 1);
        let bit_max = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        (min, bit_max.min(no_wrap_max))
    };
    let grind = |raw_bits: f64, term: &'static str| -> Result<u32, ProfileError> {
        let bits = ceil_pos(lambda_f - raw_bits);
        if bits > MAX_DERIVED_GRINDING_BITS && !design_only {
            return Err(ProfileError::GrindingTooExpensive {
                profile: profile_name,
                term,
                bits,
                cap: MAX_DERIVED_GRINDING_BITS,
            });
        }
        Ok(bits)
    };

    // ── Step 2: the projection/fingerprint prime ─────────────────────
    let (projection_min, projection_max, projection_full_width, projection_bits_raw) = match policy
    {
        PrimePolicy::SingleDerived => {
            let width = derived_width(facts.direct_opening)?;
            let (min, max) = interval_endpoints(width);
            let raw = projection_draw_bits(facts.defect_log2_bound, width);
            (min, max, false, (width, raw))
        }
        PrimePolicy::TwoFullWidthFingerprint => {
            let raw = projection_draw_bits(facts.defect_log2_bound, 128);
            (1u128 << 127, u128::MAX, true, (128, raw))
        }
    };
    let (projection_width, projection_raw) = projection_bits_raw;
    // The initial grinding boundary protects the whole pre-PIOP draw block:
    // the projection prime AND the τ point (error tau_arity/q).
    let tau_raw = f64::from(projection_width - 1) - f64::from(facts.tau_arity.max(1)).log2();
    let draw_raw = match projection_raw {
        Some(bits) => bits.min(tau_raw),
        None => tau_raw,
    };
    let initial_grinding_bits = grind(draw_raw, "step2:projection-draw")?;
    if let Some(bits) = projection_raw {
        terms.push(SoundnessTerm {
            name: "step2:projection-draw",
            bits: bits + f64::from(initial_grinding_bits),
            grinding_bits: initial_grinding_bits,
            floor: false,
        });
    }
    terms.push(SoundnessTerm {
        name: "step3:tau-draw",
        bits: tau_raw + f64::from(initial_grinding_bits),
        grinding_bits: initial_grinding_bits,
        floor: false,
    });

    // ── Step 3: PIOP rounds and terminal draws over the projection field ─
    let round_raw = f64::from(projection_width - 1) - f64::from(facts.piop_degree.max(1)).log2();
    let piop_round_grinding_bits = grind(round_raw, "step3:piop-round")?;
    terms.push(SoundnessTerm {
        name: "step3:piop-round",
        bits: round_raw + f64::from(piop_round_grinding_bits),
        grinding_bits: piop_round_grinding_bits,
        floor: false,
    });
    let terminal_grinding_bits = piop_round_grinding_bits;
    terms.push(SoundnessTerm {
        name: "step4:terminal-draw",
        bits: round_raw + f64::from(terminal_grinding_bits),
        grinding_bits: terminal_grinding_bits,
        floor: false,
    });

    // ── Step 5.0: the reduction draw (Strategy 2 only) ───────────────
    let reduction = match policy {
        PrimePolicy::SingleDerived => None,
        PrimePolicy::TwoFullWidthFingerprint => {
            let width = derived_width(true)?;
            let (min, max) = interval_endpoints(width);
            let raw =
                projection_draw_bits(facts.step50_magnitude_log2, width).unwrap_or(f64::INFINITY);
            let grinding_bits = grind(raw, "step5_0:reduction-draw")?;
            terms.push(SoundnessTerm {
                name: "step5_0:reduction-draw",
                bits: raw + f64::from(grinding_bits),
                grinding_bits,
                floor: false,
            });
            Some(ReductionPrimeParams {
                min,
                max,
                grinding_bits,
            })
        }
    };

    // ── Steps 5.2/5.3: the forest and the opener over K = GF(2^128) ──
    terms.push(SoundnessTerm {
        name: "step5_2:gkr-round",
        bits: 128.0 - 3f64.log2() + f64::from(forest_round_grinding_bits),
        grinding_bits: forest_round_grinding_bits,
        floor: false,
    });
    terms.push(SoundnessTerm {
        name: "step5_3:ring-switch",
        bits: 128.0 + f64::from(ring_switch_grinding_bits),
        grinding_bits: ring_switch_grinding_bits,
        floor: false,
    });
    terms.push(SoundnessTerm {
        name: "step5_3:ligerito-tracked",
        bits: ligerito_target_bits as f64,
        grinding_bits: 0,
        floor: false,
    });
    // Flock-internal field-limited rounds: ~d/2^128 per sumcheck message
    // with nothing to grind them — the GF(2^128) floor.
    terms.push(SoundnessTerm {
        name: "step5_3:gf128-floor-untracked",
        bits: 128.0 - 3f64.log2(),
        grinding_bits: 0,
        floor: true,
    });

    let accounting = SoundnessAccounting {
        target: lambda,
        terms,
    };
    if !design_only {
        for term in &accounting.terms {
            if !term.floor && term.bits + 1e-9 < lambda_f {
                return Err(ProfileError::TargetUnreachable {
                    profile: profile_name,
                    term: term.name,
                    bits: term.bits,
                    lambda,
                });
            }
        }
    }

    Ok(IopSecurityParams {
        profile_name,
        lambda,
        projection_min,
        projection_max,
        projection_full_width,
        initial_grinding_bits,
        piop_round_grinding_bits,
        terminal_grinding_bits,
        reduction,
        forest_round_grinding_bits,
        ring_switch_grinding_bits,
        ligerito_target_bits,
        ood: None,
        accounting,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The facts of the wired MultiSwap k=0 instance (former
    /// `multiswap/prime.rs` doc accounting: values `< 2^2048`, coefficients
    /// `< 2^2048`, `<= 353` row entries → defect `< 2^8210`; lift magnitude
    /// `< d·Q² <= 2^(25+256)`; t=13, W=1, gate_vars+2 = 15 τ coordinates).
    fn multiswap_facts() -> IopInstanceFacts {
        IopInstanceFacts {
            defect_log2_bound: 8210,
            lift_arity_log2: 13,
            opening_t: 13,
            opening_word_bits: 1,
            direct_opening: true,
            tau_arity: 15,
            piop_degree: 3,
            step50_magnitude_log2: 282,
        }
    }

    fn sha_facts(log_compressions: u32) -> IopInstanceFacts {
        IopInstanceFacts {
            // Per-row SHA defects are < 2^96 « q, so no bad prime exists.
            defect_log2_bound: 96,
            lift_arity_log2: log_compressions,
            // The SHA opening is virtual (fold width capped from q_bits).
            opening_t: log_compressions,
            opening_word_bits: 1,
            direct_opening: false,
            tau_arity: 8 + log_compressions,
            piop_degree: 3,
            step50_magnitude_log2: 0,
        }
    }

    #[test]
    fn limber114_reproduces_the_multiswap_constants_exactly() {
        let params = Limber114::instantiate(&multiswap_facts()).unwrap();
        assert_eq!(params.projection_min, 1u128 << 127);
        assert_eq!(params.projection_max, u128::MAX);
        assert!(params.projection_full_width);
        assert_eq!(params.initial_grinding_bits, 0);
        assert_eq!(params.piop_round_grinding_bits, 0);
        assert_eq!(params.terminal_grinding_bits, 0);
        let reduction = params.reduction.unwrap();
        assert_eq!(reduction.min, 1u128 << 112);
        assert_eq!(reduction.max, (1u128 << 113) - 1);
        assert_eq!(reduction.grinding_bits, 10);
        assert_eq!(params.forest_round_grinding_bits, 0);
        assert_eq!(params.ring_switch_grinding_bits, 0);
        assert_eq!(params.ligerito_target_bits, 114);
        // The former doc-comment floors: fingerprint 2^-114.0, reduction
        // 2^-104.2 + 10 = 2^-114.2.
        let fingerprint = params
            .accounting
            .terms
            .iter()
            .find(|term| term.name == "step2:projection-draw")
            .unwrap();
        assert!(
            (fingerprint.bits - 114.07).abs() < 0.1,
            "{}",
            fingerprint.bits
        );
        let reduction_term = params
            .accounting
            .terms
            .iter()
            .find(|term| term.name == "step5_0:reduction-draw")
            .unwrap();
        assert!(
            (reduction_term.bits - 114.25).abs() < 0.1,
            "{}",
            reduction_term.bits
        );
        assert!(params.accounting.controllable_bits() >= 114.0);
    }

    #[test]
    fn sha128_reference_reproduces_the_grinding_tables_exactly() {
        for t in 7..=16u32 {
            let params = Sha128ReferenceSchedule::instantiate(&sha_facts(t)).unwrap();
            let expected_initial = match t {
                7 | 8 => 20,
                9..=15 => 21,
                16 => 22,
                _ => unreachable!(),
            };
            let expected_outer = if t == 16 { 19 } else { 18 };
            assert_eq!(
                params.initial_grinding_bits, expected_initial,
                "initial at t={t}"
            );
            assert_eq!(
                params.piop_round_grinding_bits, expected_outer,
                "outer at t={t}"
            );
            assert_eq!(params.terminal_grinding_bits, expected_outer);
            // Interval: 113-bit window, 112 at t=16; the exact former
            // endpoints including the no-wrap top cap.
            let width = 113.min(128 - t);
            assert_eq!(params.projection_min, 1u128 << (width - 1), "min at t={t}");
            let no_wrap = 1 + u128::MAX / ((1u128 << t) + 1);
            assert_eq!(
                params.projection_max,
                ((1u128 << width) - 1).min(no_wrap),
                "max at t={t}"
            );
            assert!(!params.projection_full_width);
            assert!(params.reduction.is_none());
        }
    }

    #[test]
    fn lambda100_zeroes_every_grind_on_the_sha_shape() {
        for t in 4..=16u32 {
            let params = Lambda100::instantiate(&sha_facts(t)).unwrap();
            assert_eq!(params.initial_grinding_bits, 0);
            assert_eq!(params.piop_round_grinding_bits, 0);
            assert_eq!(params.terminal_grinding_bits, 0);
            assert_eq!(params.forest_round_grinding_bits, 0);
            assert_eq!(params.ring_switch_grinding_bits, 0);
            assert_eq!(params.ligerito_target_bits, 100);
            // Same prime interval as the reference schedule: λ moves grinding,
            // never the field, so the chunk geometry is λ-independent.
            let reference = Sha128ReferenceSchedule::instantiate(&sha_facts(t)).unwrap();
            assert_eq!(params.projection_min, reference.projection_min);
            assert_eq!(params.projection_max, reference.projection_max);
            assert!(params.accounting.controllable_bits() >= 100.0);
            assert!(params.accounting.achieved_bits() >= 100.0);
        }
    }

    #[test]
    fn lambda128_grinds_the_forest_and_reports_the_floor() {
        let params = Lambda128::instantiate(&sha_facts(14)).unwrap();
        assert_eq!(params.forest_round_grinding_bits, 2);
        assert_eq!(params.ring_switch_grinding_bits, 1);
        assert_eq!(params.initial_grinding_bits, 21);
        assert_eq!(params.piop_round_grinding_bits, 18);
        assert!(params.accounting.controllable_bits() >= 128.0);
        // Honest end-to-end figure: the GF(2^128) floor binds at ~126.4.
        let achieved = params.accounting.achieved_bits();
        assert!((achieved - 126.415).abs() < 0.01, "{achieved}");
        assert_eq!(
            params.accounting.binding_term().name,
            "step5_3:gf128-floor-untracked"
        );
        assert!(params.accounting.binding_term().floor);
    }

    #[test]
    fn sha128_reference_documents_its_floor_instead_of_refusing() {
        // DESIGN_ONLY: the 126.4-bit un-grinded GKR round would fail a
        // strict λ=128 validation; the reference schedule documents it.
        let params = Sha128ReferenceSchedule::instantiate(&sha_facts(14)).unwrap();
        let gkr = params
            .accounting
            .terms
            .iter()
            .find(|term| term.name == "step5_2:gkr-round")
            .unwrap();
        assert!(gkr.bits < 128.0);
        assert!(params.accounting.controllable_bits() < 128.0);
    }

    #[test]
    fn multiswap_at_lambda128_needs_a_fingerprint_grind() {
        // Strategy 2 at λ=128: the 114.07-bit fingerprint draw picks up a
        // 14-bit grind, the reduction draw ~24 — derivable, near the cap.
        struct Limber128;
        impl IopSecurityProfile for Limber128 {
            const NAME: &'static str = "limber128-test";
            const LAMBDA: u32 = 128;
            const PRIME_POLICY: PrimePolicy = PrimePolicy::TwoFullWidthFingerprint;
            const LIGERITO_TARGET_BITS: usize = 128;
            const FOREST_ROUND_GRINDING_BITS: u32 = derive_forest_grinding(128);
            const RING_SWITCH_GRINDING_BITS: u32 = derive_ring_switch_grinding(128);
        }
        let params = Limber128::instantiate(&multiswap_facts()).unwrap();
        assert_eq!(params.initial_grinding_bits, 14);
        assert_eq!(params.reduction.unwrap().grinding_bits, 24);
        assert!(params.accounting.controllable_bits() >= 128.0);
    }

    #[test]
    fn oversized_shapes_are_rejected() {
        let mut facts = multiswap_facts();
        facts.opening_t = 120;
        facts.opening_word_bits = 8;
        assert!(matches!(
            Limber114::instantiate(&facts),
            Err(ProfileError::ShapeTooWide { .. })
        ));
    }

    #[test]
    fn prime_count_bound_matches_the_former_doc_floors() {
        assert!((log2_prime_count_lower_bound(128) - 120.07).abs() < 0.05);
        assert!((log2_prime_count_lower_bound(113) - 105.25).abs() < 0.05);
    }
}
