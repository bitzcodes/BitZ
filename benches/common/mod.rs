//! Shared harness for the protocol benches: one accounting model, one
//! printer, one machine-readable `RESULT` line (`schema=bitz/1`).
//!
//! Multiplication campaigns use the separate manifest/sample format described
//! in `docs/native-mul-compare.md`. This module retains the other benches’ console output.
//!
//! Summary of the semantics implemented here:
//! - `prove_ms` is the **end-to-end prover**: everything after the prover
//!   holds a witness (bit-packing, commitment, projection, prime sampling +
//!   grinding, PIOP, bitification, Step 5.0, the BitZ opening).
//! - Witness generation and one-time public preprocessing are excluded and
//!   reported separately (`witness_ms`, `setup_ms`).
//! - Per-step splits come from the crate's umbrella tracing spans
//!   (`step2:*` … `step5:*`); a signed residual makes each split sum to its
//!   total exactly.
//! - Steps that do not run in a path print `na`, never `0.00`.

#![allow(dead_code)] // each bench uses a subset of the harness

pub mod cli;
pub mod environment;
pub mod proof_fingerprint;

/// Record actual forest choices for the application benchmarks as well as multiplication.
pub fn start_gkr_recording() {
    #[cfg(feature = "bench-internals")]
    bitz::merged_forest::schedule::start_recording();
}

pub fn print_gkr_schedules() {
    #[cfg(feature = "bench-internals")]
    println!(
        "GKR_SCHEDULES {}",
        serde_json::json!({
            "requested": std::env::var("F2_FOREST_SCHEDULE").unwrap_or_else(|_| "auto".into()),
            "resolved": bitz::merged_forest::schedule::take_records(),
        })
    );
}

/// Serialize native SDK sessions in tests and explicitly supply their subscriber.
#[cfg(all(test, feature = "span-metrics"))]
pub fn test_tracing() -> (
    tracing::subscriber::DefaultGuard,
    std::sync::MutexGuard<'static, ()>,
) {
    use tracing_subscriber::prelude::*;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let lock = LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let subscriber = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(bitz::observability::layer()),
    );
    (subscriber, lock)
}
#[cfg(feature = "bench-peak-memory")]
pub mod heap_run;
pub mod mul_witness;
pub mod output;
pub mod pcs_console;
#[cfg(feature = "bench-peak-memory")]
pub mod peak_memory;
#[cfg(feature = "span-metrics")]
pub mod perfetto;
#[cfg(feature = "plonky3-whir-bench")]
pub mod plonky3;
#[cfg(any(feature = "native-mul-compare", feature = "plonky3-sha256-bench"))]
pub mod whir_tuning;

use clap::ValueEnum;
use bitz::piop::spartan::{
    IopSecurityProfile, Lambda100, Lambda128, Limber112, Limber114, PrimePolicy,
    Sha128ReferenceSchedule,
};

/// Actual local dependency commit recorded when this benchmark was built.
/// Path dependencies have no Git revision in Cargo.lock.
pub fn local_vendor_revision(package: &str) -> String {
    let revision = if package.starts_with("p3-") {
        env!("PLONKY3_REVISION")
    } else if package.starts_with("binius-") {
        env!("BINIUS64_REVISION")
    } else if package == "limber" {
        env!("LIMBER_REVISION")
    } else {
        panic!("unknown benchmark dependency {package}");
    };
    revision.to_owned()
}

// ---------------------------------------------------------------------
// Environment: canonical knobs, deprecated aliases, strict unknown check
// ---------------------------------------------------------------------

/// Every `BITZ_*` variable any binary in this repo understands. An exported
/// `BITZ_*` variable outside this list aborts the bench so a typo'd knob can
/// never silently do nothing. Keep sorted; add new knobs here.
pub const KNOWN_BITZ_ENV: &[&str] = &[
    // Build metadata from scripts/build_metadata.py. Cargo also exports
    // rustc-env values when launching benchmark executables.
    "BITZ_DIRTY",
    "BITZ_REVISION",
    // A/B example harness knobs (examples/taps_ab.rs, examples/rlc_ab.rs).
    "BITZ_AB_B3FAM",
    "BITZ_AB_B3OPEN",
    "BITZ_AB_BLAKE3",
    "BITZ_AB_COLLAPSE",
    "BITZ_AB_COLS4",
    "BITZ_AB_COLS4_FAM",
    "BITZ_AB_FAM_DELTA",
    "BITZ_AB_J3",
    "BITZ_AB_MIX6",
    "BITZ_AB_N",
    "BITZ_AB_NO_FAMILY",
    "BITZ_AB_OPEN8",
    "BITZ_AB_OPEN_DELTA",
    "BITZ_AB_REPS",
    "BITZ_AB_ROUNDS",
    "BITZ_AB_SCHED",
    "BITZ_AB_SHARED",
    "BITZ_AB_SINGLES",
    "BITZ_AB_STMTS",
    // Deprecated aliases (kept working; see `reps`/`shapes`/`seed`).
    // Canonical bench knobs.
    "BITZ_BENCH_EXT",
    "BITZ_BENCH_FILL",
    "BITZ_BENCH_LAMBDA",
    "BITZ_MULTISWAP_BATCH_COUNT",
    "BITZ_MULTISWAP_CHECK_ONLY",
    "BITZ_BENCH_ORDER",
    "BITZ_BENCH_PASS",
    "BITZ_BENCH_QUIET",
    "BITZ_BENCH_PHASE_SAMPLES",
    "BITZ_BENCH_PROOF_FINGERPRINT",
    "BITZ_BENCH_REPS",
    "BITZ_BENCH_SEED",
    "BITZ_BENCH_SHAPES",
    "BITZ_CM_EXPONENTS",
    "BITZ_CM_SEED",
    // Prover-path toggles (transcript-preserving optimization knobs).
    "BITZ_COL_ELIDE",
    "BITZ_EQF_DOUBLE",
    "BITZ_EQF_DOUBLE_MIN",
    "BITZ_EQF_FUSE",
    "BITZ_EQF_NOKERNEL",
    "BITZ_GKR_DIRECT_CLOSE",
    "BITZ_GKR_RECOVER",
    "BITZ_EQ_TABLE_SAMPLES",
    "BITZ_FIXED_SCALAR",
    "BITZ_FLAT_FOREST",
    "BITZ_FOLDV_LUT",
    "BITZ_INNER_FIELD_ACCUM",
    "BITZ_INNER_NATIVE_FOLD",
    "BITZ_JIT_GRID",
    "BITZ_JIT_R1",
    "BITZ_LEAF8",
    "BITZ_LEAF_A2_FACTORED",
    "BITZ_LEAF_TILE",
    "BITZ_LIG_PROFILE",
    "BITZ_LUT3",
    "BITZ_LUT4",
    "BITZ_LUT_PRFM",
    "BITZ_MATS_PRE",
    "BITZ_MATS_TILE",
    "BITZ_MATS_TILE_B",
    "BITZ_MAT_GRID",
    // Matched MultiSwap/Mod-R1CS campaign trace metadata.
    "BITZ_MULTISWAP_BUILD_PROFILE",
    "BITZ_MULTISWAP_CAMPAIGN_ID",
    "BITZ_MULTISWAP_CPU",
    "BITZ_MULTISWAP_EXPECTED_CONSTRAINT_DIGEST",
    "BITZ_MULTISWAP_GIT_REV",
    // Deprecated alias.
    "BITZ_MULTISWAP_REPS",
    "BITZ_MULTISWAP_TRACE_PATH",
    "BITZ_PAIR2_FACTORED",
    "BITZ_PAR_CHUNK",
    "BITZ_BINIUS_LOG_INV_RATE",
    "BITZ_BINIUS_LIGERITO_LOG_INV_RATE",
    "BITZ_PLONKY3_LOG_INV_RATE",
    // Binius64-with-BitZ-opener rows: the 100-bit gate's accounting model
    // (`union` = union bound over every term, `rbr` = round-by-round minimum).
    "BITZ_BINIUS_LIGERITO_ACCOUNTING",
    "BITZ_WHIR_FOLDING",
    "BITZ_WHIR_LOG_INV_RATE",
    "BITZ_WHIR_MAX_POW_BITS",
    "BITZ_WHIR_CONFIG",
    "BITZ_WHIR_TUNING_REPS",
    "BITZ_QUAD",
    "BITZ_QUAD_KERNEL",
    "BITZ_RLC_EAGER",
    "BITZ_RLC_J34_LAZY",
    "BITZ_RS_FAST",
    // SHA trace-writer knobs.
    "BITZ_SHA_BUILD_PROFILE",
    "BITZ_SHA_CPU",
    "BITZ_SHA_GIT_REV",
    "BITZ_SHA_INNER_PREFIX_VARS",
    "BITZ_SHA_LOG2S",
    "BITZ_SHA_MNUMROWS_LOG2S",
    "BITZ_SHA_OPENING_LAYOUT",
    "BITZ_SHA_OPENING_T",
    "BITZ_SHA_PRODUCT_TS",
    "BITZ_SHA_REPS",
    "BITZ_SHA_RESULT_PATH",
    "BITZ_SHA_SEED",
    "BITZ_SHA_TRACE_PATH",
    "BITZ_T4_FACTORED",
    "BITZ_T4_PRFM",
    "BITZ_TAPS_DELTA",
    "BITZ_TAPS_GRP",
    "BITZ_TAPS_SEED",
    "BITZ_VIRT_ID_FAST",
    "BITZ_VIRT_PLANES",
];

/// Aborts on any exported `BITZ_*` variable the repo does not know.
pub fn enforce_known_env() {
    let mut unknown: Vec<String> = std::env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .filter(|key| key.starts_with("BITZ_") && !KNOWN_BITZ_ENV.contains(&key.as_str()))
        .collect();
    if unknown.is_empty() {
        return;
    }
    unknown.sort();
    eprintln!(
        "error: unknown BITZ_* environment variable(s): {}",
        unknown.join(", ")
    );
    eprintln!("       known benchmark knobs:");
    for chunk in KNOWN_BITZ_ENV.chunks(4) {
        eprintln!("         {}", chunk.join(" "));
    }
    std::process::exit(2);
}

/// `BITZ_BENCH_QUIET=1` mutes the harness's advisory `warning:` lines
/// (deprecated-alias notices, ignored-knob notices, build-configuration
/// hints). Errors that abort a run are never muted.
pub fn quiet() -> bool {
    std::env::var("BITZ_BENCH_QUIET").is_ok_and(|v| v != "0")
}

/// Prints `warning: <msg>` on stderr unless [`quiet`] is set. Every
/// advisory warning a bench emits goes through here so one knob mutes
/// them all.
pub fn warn(msg: impl std::fmt::Display) {
    if !quiet() {
        eprintln!("warning: {msg}");
    }
}

/// Reads `canonical`, falling back to `alias` with a deprecation warning.
/// Setting both to different values is an error.
fn env_with_alias(canonical: &str, alias: Option<&str>) -> Option<String> {
    let canonical_value = std::env::var(canonical).ok();
    let alias_value = alias.and_then(|name| std::env::var(name).ok());
    match (canonical_value, alias_value) {
        (Some(main), Some(old)) => {
            if main != old {
                eprintln!(
                    "error: {canonical}={main} and deprecated alias {}={old} disagree",
                    alias.unwrap_or_default()
                );
                std::process::exit(2);
            }
            Some(main)
        }
        (Some(main), None) => Some(main),
        (None, Some(old)) => {
            warn(format_args!(
                "{} is deprecated; use {canonical}",
                alias.unwrap_or_default()
            ));
            Some(old)
        }
        (None, None) => None,
    }
}

/// Measured repetitions (one extra untimed warm-up is always run).
pub fn reps(alias: Option<&str>, default: usize) -> usize {
    env_with_alias("BITZ_BENCH_REPS", alias).map_or(default, |value| {
        cli::value("BITZ_BENCH_REPS", &value, cli::positive)
    })
}

/// Bench-specific shape list (meaning documented per bench).
pub fn shape_values<T, P>(alias: Option<&str>, parser: P) -> Option<Vec<T>>
where
    T: Clone + Send + Sync + 'static,
    P: clap::builder::TypedValueParser<Value = T>,
{
    env_with_alias("BITZ_BENCH_SHAPES", alias)
        .map(|value| cli::values("BITZ_BENCH_SHAPES", &value, parser))
}

/// Root seed (decimal or 0x-hex).
pub fn seed(alias: Option<&str>, default: u64) -> u64 {
    env_with_alias("BITZ_BENCH_SEED", alias).map_or(default, |value| {
        cli::value("BITZ_BENCH_SEED", &value, cli::seed)
    })
}

// ---------------------------------------------------------------------
// Security profile selection (`BITZ_BENCH_LAMBDA`)
// ---------------------------------------------------------------------

/// The IOP security profile a run measures at — one of the compile-time
/// policy types of `src/piop/spartan/profile.rs`, chosen at runtime by
/// `BITZ_BENCH_LAMBDA` and dispatched to the monomorphized bench body by
/// [`with_profile!`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum SecurityProfile {
    #[value(name = "100", alias = "lambda100")]
    Lambda100,
    #[value(name = "128", alias = "lambda128")]
    Lambda128,
    #[value(name = "112", alias = "limber112")]
    Limber112,
    #[value(name = "114", alias = "limber114")]
    Limber114,
    Sha128ReferenceSchedule,
}

impl SecurityProfile {
    /// The profile's `NAME` (the `profile=` RESULT key).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Lambda100 => Lambda100::NAME,
            Self::Lambda128 => Lambda128::NAME,
            Self::Limber112 => Limber112::NAME,
            Self::Limber114 => Limber114::NAME,
            Self::Sha128ReferenceSchedule => Sha128ReferenceSchedule::NAME,
        }
    }

    /// The profile's target λ.
    pub const fn lambda(self) -> u32 {
        match self {
            Self::Lambda100 => Lambda100::LAMBDA,
            Self::Lambda128 => Lambda128::LAMBDA,
            Self::Limber112 => Limber112::LAMBDA,
            Self::Limber114 => Limber114::LAMBDA,
            Self::Sha128ReferenceSchedule => Sha128ReferenceSchedule::LAMBDA,
        }
    }

    /// One transcript prime or the two-prime Strategy 2 — what decides
    /// which relations can instantiate the profile.
    pub const fn prime_policy(self) -> PrimePolicy {
        match self {
            Self::Lambda100 => Lambda100::PRIME_POLICY,
            Self::Lambda128 => Lambda128::PRIME_POLICY,
            Self::Limber112 => Limber112::PRIME_POLICY,
            Self::Limber114 => Limber114::PRIME_POLICY,
            Self::Sha128ReferenceSchedule => Sha128ReferenceSchedule::PRIME_POLICY,
        }
    }

    /// The shortest `BITZ_BENCH_LAMBDA` spelling of the profile: the target
    /// bits where that is unambiguous, the full name otherwise.
    pub fn knob_value(self) -> String {
        self.to_possible_value()
            .expect("selectable profile")
            .get_name()
            .to_owned()
    }

    fn admissible(policy: PrimePolicy) -> String {
        Self::value_variants()
            .iter()
            .filter(|profile| profile.prime_policy() == policy)
            .map(|profile| {
                if profile.knob_value() == profile.name() {
                    profile.name().to_owned()
                } else {
                    format!("{} ({})", profile.knob_value(), profile.name())
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

const fn describe_policy(policy: PrimePolicy) -> &'static str {
    match policy {
        PrimePolicy::SingleDerived => "single-prime",
        PrimePolicy::TwoFullWidthFingerprint => "two-prime (Strategy 2)",
    }
}

/// Reads `BITZ_BENCH_LAMBDA`: `100`, `128`, `114`, or a profile name
/// (`lambda100`, `lambda128`, `limber114`, `sha128-reference-schedule`).
/// `None` = unset, and the bench keeps its own default. `policy` is the
/// prime strategy the calling bench's relation instantiates; selecting a
/// profile of the other strategy aborts here, with the admissible list,
/// instead of failing later inside relation preparation. A value that
/// names no profile aborts too (a typo can never silently do nothing).
pub fn security_profile(policy: PrimePolicy) -> Option<SecurityProfile> {
    let value = std::env::var("BITZ_BENCH_LAMBDA").ok()?;
    let profile = cli::value("BITZ_BENCH_LAMBDA", &value, |value: &str| {
        SecurityProfile::from_str(value.trim(), true)
    });
    if profile.prime_policy() != policy {
        eprintln!(
            "error: BITZ_BENCH_LAMBDA={value} selects {}, a {} profile, but this bench's \
             relation instantiates {} profiles",
            profile.name(),
            describe_policy(profile.prime_policy()),
            describe_policy(policy),
        );
        eprintln!(
            "       admissible here: {}",
            SecurityProfile::admissible(policy)
        );
        std::process::exit(2);
    }
    Some(profile)
}

/// Banner fragment naming the profile a run measures at and where the
/// choice came from.
pub fn profile_banner(selected: Option<SecurityProfile>, default: SecurityProfile) -> String {
    match selected {
        Some(profile) => format!(
            "{} (λ={}, BITZ_BENCH_LAMBDA={})",
            profile.name(),
            profile.lambda(),
            profile.knob_value()
        ),
        None => format!(
            "{} (λ={}, the default; BITZ_BENCH_LAMBDA selects another)",
            default.name(),
            default.lambda()
        ),
    }
}

/// Expands to `$f::<P>($args…)` with `P` the profile type `$profile`
/// names. `$f` is a local function generic over exactly one
/// `P: IopSecurityProfile` parameter — the relation-preparation seam every
/// protocol bench has. All five bodies are compiled; the env knob only
/// picks which one runs.
#[allow(unused_macros)]
macro_rules! with_profile {
    ($profile:expr, $f:ident ( $($arg:expr),* $(,)? )) => {
        match $profile {
            $crate::common::SecurityProfile::Lambda100 => {
                $f::<::bitz::piop::spartan::Lambda100>($($arg),*)
            }
            $crate::common::SecurityProfile::Lambda128 => {
                $f::<::bitz::piop::spartan::Lambda128>($($arg),*)
            }
            $crate::common::SecurityProfile::Limber112 => {
                $f::<::bitz::piop::spartan::Limber112>($($arg),*)
            }
            $crate::common::SecurityProfile::Limber114 => {
                $f::<::bitz::piop::spartan::Limber114>($($arg),*)
            }
            $crate::common::SecurityProfile::Sha128ReferenceSchedule => {
                $f::<::bitz::piop::spartan::Sha128ReferenceSchedule>($($arg),*)
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use with_profile;

/// Validate the benchmark environment and size the performance thread pool.
/// Subscriber installation belongs to the executable. Returns the thread count.
pub fn init() -> usize {
    enforce_known_env();
    let _ = flock_core::init_perf_thread_pool();
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Rayon thread count without the full [`init`] (for benches that keep the
/// phase profiler opt-in, like `pcs`).
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

// ---------------------------------------------------------------------
// Step taxonomy (paper §2.1, code order)
// ---------------------------------------------------------------------

/// Umbrella scope labels: the step totals. Step 1 (commit) is bench-timed
/// wall clock, not a scope.
pub const STEP2_PROVE: &str = "step2:project_prove";
pub const STEP3_PROVE: &str = "step3:piop_prove";
pub const STEP4_PROVE: &str = "step4:bitify_prove";
pub const STEP5_0_PROVE: &str = "step5_0:reduce_prove";
pub const STEP5_PROVE: &str = "step5:open_prove";
pub const STEP2_VERIFY: &str = "step2:project_verify";
pub const STEP3_VERIFY: &str = "step3:piop_verify";
pub const STEP4_VERIFY: &str = "step4:bitify_verify";
pub const STEP5_0_VERIFY: &str = "step5_0:reduce_verify";
pub const STEP5_VERIFY: &str = "step5:open_verify";

/// Shared detail-label table (nested under the umbrellas). Sums of these
/// feed the optional detail keys; they never enter the step totals.
const S3_OUTER: &[&str] = &[
    "spartan:outer_sumcheck",
    "spartan:outer_univariate_skip",
    "sha256:spartan_outer_prove",
    "sha256:spartan_outer_verify",
];
const S3_BIND: &[&str] = &["spartan:bind_and_batch"];
const S3_INNER: &[&str] = &[
    "spartan:inner_sumcheck",
    "sha256:spartan_inner_prove",
    "sha256:spartan_inner_verify",
];
/// Step 5.2: exponent tables + merged GKR forest + presum discharge.
const S5_FOREST: &[&str] = &[
    "mc:pack",
    "mc:pow2",
    "mc:forest",
    "mc:fold_v",
    "mc:presum_tbls",
    "mc:presum_run",
    "mqv:pack",
];
/// Step 5.3: ring switch + recursive Ligerito (including the virtual
/// batching machinery: derived weights, the h_i fold, the a′ build).
const S5_OPENER: &[&str] = &[
    "mq:rings",
    "mq:bcomb",
    "mq:lig",
    "mq:rings_main",
    "mq:bcomb_main",
    "mqv:wprep",
    "mqv:planes",
    "mqv:hs",
    "mqv:aprime",
    "mqv:lig",
    "mqv:vwprep",
    "mqv:vaprime",
    "mqv:vlig",
];

fn label_sum_ms(phases: &[(String, f64)], labels: &[&str]) -> Option<f64> {
    let mut sum = 0.0;
    let mut seen = false;
    for (label, seconds) in phases {
        if labels.contains(&label.as_str()) {
            sum += seconds * 1e3;
            seen = true;
        }
    }
    seen.then_some(sum)
}

fn label_ms(phases: &[(String, f64)], label: &str) -> Option<f64> {
    label_sum_ms(phases, &[label])
}

/// One side's per-step samples across reps. Every value is `Some` only when
/// the corresponding scope fired in **every** rep (otherwise `na`).
#[derive(Default)]
pub struct StepSamples {
    total: Vec<f64>,
    commit: Vec<Option<f64>>,
    project: Vec<Option<f64>>,
    piop: Vec<Option<f64>>,
    bitify: Vec<Option<f64>>,
    reduce: Vec<Option<f64>>,
    open: Vec<Option<f64>>,
    outer: Vec<Option<f64>>,
    bind: Vec<Option<f64>>,
    inner: Vec<Option<f64>>,
    forest: Vec<Option<f64>>,
    opener: Vec<Option<f64>>,
}

impl StepSamples {
    /// Records one prover rep: the end-to-end wall time, the bench-timed
    /// Step 1 (bit-pack + commit) wall time, and the profiler totals drained
    /// after the prove call.
    pub fn record_prove(&mut self, total_ms: f64, commit_ms: f64, phases: &[(String, f64)]) {
        if std::env::var("BITZ_BENCH_PHASE_SAMPLES").is_ok_and(|v| v == "1") {
            println!(
                "PHASE_SAMPLE {}",
                serde_json::json!({"kind":"prove", "total_ms":total_ms,
                "commit_ms":commit_ms, "phases_seconds":phases})
            );
        }
        self.total.push(total_ms);
        self.commit.push(Some(commit_ms));
        self.record_scopes(
            phases,
            [
                STEP2_PROVE,
                STEP3_PROVE,
                STEP4_PROVE,
                STEP5_0_PROVE,
                STEP5_PROVE,
            ],
        );
    }

    /// Records one verifier rep (no Step 1: the verifier holds a commitment).
    pub fn record_verify(&mut self, total_ms: f64, phases: &[(String, f64)]) {
        if std::env::var("BITZ_BENCH_PHASE_SAMPLES").is_ok_and(|v| v == "1") {
            println!(
                "PHASE_SAMPLE {}",
                serde_json::json!({"kind":"verify", "total_ms":total_ms,
                "phases_seconds":phases})
            );
        }
        self.total.push(total_ms);
        self.commit.push(None);
        self.record_scopes(
            phases,
            [
                STEP2_VERIFY,
                STEP3_VERIFY,
                STEP4_VERIFY,
                STEP5_0_VERIFY,
                STEP5_VERIFY,
            ],
        );
    }

    fn record_scopes(&mut self, phases: &[(String, f64)], steps: [&str; 5]) {
        self.project.push(label_ms(phases, steps[0]));
        self.piop.push(label_ms(phases, steps[1]));
        self.bitify.push(label_ms(phases, steps[2]));
        self.reduce.push(label_ms(phases, steps[3]));
        self.open.push(label_ms(phases, steps[4]));
        self.outer.push(label_sum_ms(phases, S3_OUTER));
        self.bind.push(label_sum_ms(phases, S3_BIND));
        self.inner.push(label_sum_ms(phases, S3_INNER));
        self.forest.push(label_sum_ms(phases, S5_FOREST));
        self.opener.push(label_sum_ms(phases, S5_OPENER));
    }

    /// Medians. The residual is the signed difference between the total
    /// median and the sum of the present step medians, so the printed split
    /// sums to the printed total exactly.
    pub fn medians(&self) -> StepMedians {
        let total = median(&self.total);
        let commit = optional_median(&self.commit);
        let project = optional_median(&self.project);
        let piop = optional_median(&self.piop);
        let bitify = optional_median(&self.bitify);
        let reduce = optional_median(&self.reduce);
        let open = optional_median(&self.open);
        let steps_sum: f64 = [commit, project, piop, bitify, reduce, open]
            .iter()
            .flatten()
            .sum();
        StepMedians {
            total,
            commit,
            project,
            piop,
            bitify,
            reduce,
            open,
            residual: total - steps_sum,
            outer: optional_median(&self.outer),
            bind: optional_median(&self.bind),
            inner: optional_median(&self.inner),
            forest: optional_median(&self.forest),
            opener: optional_median(&self.opener),
        }
    }
}

/// Median step split of one side, in milliseconds. `None` = `na`.
#[derive(Clone, Copy, Debug)]
pub struct StepMedians {
    pub total: f64,
    pub commit: Option<f64>,
    pub project: Option<f64>,
    pub piop: Option<f64>,
    pub bitify: Option<f64>,
    pub reduce: Option<f64>,
    pub open: Option<f64>,
    pub residual: f64,
    pub outer: Option<f64>,
    pub bind: Option<f64>,
    pub inner: Option<f64>,
    pub forest: Option<f64>,
    pub opener: Option<f64>,
}

// ---------------------------------------------------------------------
// Report + printer
// ---------------------------------------------------------------------

/// Transmitted proof size split.
#[derive(Clone, Copy, Debug)]
pub struct ProofBytes {
    /// Spartan payload + grinding nonces + the Step 5.0 integer lift.
    pub piop: usize,
    /// The serialized BitZ opening (`to_bytes` of the codec proof).
    pub open: usize,
}

impl ProofBytes {
    pub const fn total(&self) -> usize {
        self.piop + self.open
    }
}

/// Everything the uniform printer needs for one measured configuration.
pub struct BenchReport {
    /// `bench=` value: `multiswap`, `sha256`, `u32_mul`, `pcs`, …
    pub bench: &'static str,
    /// `shape=` value: one compact token unique within the bench.
    pub shape: String,
    /// Bench-specific keys, printed between `shape=` and `lambda=`.
    pub extra: Vec<(String, String)>,
    /// Security target the run was measured at; `None` prints `na`.
    pub lambda: Option<u32>,
    /// Achieved bits (min over every soundness term, floors included).
    pub lambda_achieved: Option<f64>,
    /// Name of the binding soundness term.
    pub lambda_bind: Option<String>,
    pub threads: usize,
    pub reps: usize,
    /// Root seed; `None` prints `na` (deterministic benches).
    pub seed: Option<u64>,
    /// One-time, excluded from `prove_ms`.
    pub witness_ms: f64,
    /// One-time public preprocessing, excluded from `prove_ms`.
    pub setup_ms: f64,
    pub prover: StepMedians,
    pub verifier: StepMedians,
    pub proof: ProofBytes,
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map_or_else(|| "na".to_owned(), |value| format!("{value:.3}"))
}

fn fmt_row(value: Option<f64>) -> String {
    value.map_or_else(
        || "     n/a   ".to_owned(),
        |value| format!("{value:9.2} ms"),
    )
}

impl BenchReport {
    /// The uniform human block (step rows sum to the totals exactly).
    pub fn print_human(&self) {
        self.print_human_with_commitment(0);
    }

    /// Include a separately transmitted commitment in every proof-size total.
    pub fn print_human_with_commitment(&self, commitment_bytes: usize) {
        if let (Some(lambda), Some(achieved)) = (self.lambda, self.lambda_achieved) {
            println!(
                "  security: target λ={lambda} | achieved {achieved:.1} bits (binding term: {})",
                self.lambda_bind.as_deref().unwrap_or("unknown")
            );
        }
        println!(
            "  one-time (excluded from prove): witness {:.1} ms | setup {:.1} ms",
            self.witness_ms, self.setup_ms
        );
        println!(
            "  prove (end-to-end, median of {}): {:.2} ms",
            self.reps, self.prover.total
        );
        let p = &self.prover;
        println!("    step 1   commit        {}", fmt_row(p.commit));
        let step2_label = if self.bench == "sha256" {
            "field setup"
        } else {
            "project"
        };
        println!("    step 2   {step2_label:<13}{}", fmt_row(p.project));
        println!(
            "    step 3   piop          {}   (outer {} | bind {} | inner {})",
            fmt_row(p.piop),
            fmt_opt(p.outer),
            fmt_opt(p.bind),
            fmt_opt(p.inner)
        );
        println!("    step 4   bitify        {}", fmt_row(p.bitify));
        println!("    step 5.0 reduce        {}", fmt_row(p.reduce));
        println!(
            "    step 5.* open          {}   (5.2 forest {} | 5.3 ring-switch+Ligerito {})",
            fmt_row(p.open),
            fmt_opt(p.forest),
            fmt_opt(p.opener)
        );
        println!("    residual               {:9.2} ms", p.residual);
        println!("  verify (median): {:.2} ms", self.verifier.total);
        let v = &self.verifier;
        println!("    step 2   {step2_label:<13}{}", fmt_row(v.project));
        println!(
            "    step 3   piop          {}   (outer {})",
            fmt_row(v.piop),
            fmt_opt(v.outer)
        );
        println!("    step 4   bitify        {}", fmt_row(v.bitify));
        println!("    step 5.0 reduce        {}", fmt_row(v.reduce));
        println!("    step 5.* open          {}", fmt_row(v.open));
        println!("    residual               {:9.2} ms", v.residual);
        println!(
            "  proof: {} B ({:.1} KB) = piop {} B + open {} B + commitment {} B",
            self.proof.total() + commitment_bytes,
            (self.proof.total() + commitment_bytes) as f64 / 1e3,
            self.proof.piop,
            self.proof.open,
            commitment_bytes,
        );
        println!("  {}", self.result_line_with_commitment(commitment_bytes));
    }

    /// The machine-readable console line.
    pub fn result_line(&self) -> String {
        self.result_line_with_commitment(0)
    }

    fn result_line_with_commitment(&self, commitment_bytes: usize) -> String {
        let schema = if self.extra.iter().any(|(key, _)| key == "ligerito_hex") {
            "bitz/2"
        } else {
            "bitz/1"
        };
        let mut line = format!(
            "RESULT schema={schema} bench={} shape={}",
            self.bench, self.shape
        );
        for (key, value) in &self.extra {
            line.push_str(&format!(" {key}={value}"));
        }
        let lambda = self
            .lambda
            .map_or_else(|| "na".to_owned(), |bits| bits.to_string());
        let lambda_achieved = self
            .lambda_achieved
            .map_or_else(|| "na".to_owned(), |bits| format!("{bits:.1}"));
        let lambda_bind = self.lambda_bind.clone().unwrap_or_else(|| "na".to_owned());
        let seed = self
            .seed
            .map_or_else(|| "na".to_owned(), |seed| format!("{seed:#018x}"));
        let p = &self.prover;
        let v = &self.verifier;
        line.push_str(&format!(
            " lambda={lambda} lambda_achieved={lambda_achieved} lambda_bind={lambda_bind} \
             threads={} reps={} warmups=1 seed={seed} \
             witness_ms={:.3} setup_ms={:.3} prove_ms={:.3} s1_commit_ms={} \
             s2_project_ms={} s3_piop_ms={} s4_bitify_ms={} s5_0_reduce_ms={} \
             s5_open_ms={} prove_residual_ms={:.3} s3_outer_ms={} s3_bind_ms={} \
             s3_inner_ms={} s5_forest_ms={} s5_opener_ms={} verify_ms={:.3} \
             v2_project_ms={} v3_piop_ms={} v4_bitify_ms={} v5_0_reduce_ms={} \
             v5_open_ms={} verify_residual_ms={:.3} proof_bytes={} \
             proof_piop_bytes={} proof_open_bytes={} proof_commitment_bytes={} verified_samples={}",
            self.threads,
            self.reps,
            self.witness_ms,
            self.setup_ms,
            p.total,
            fmt_opt(p.commit),
            fmt_opt(p.project),
            fmt_opt(p.piop),
            fmt_opt(p.bitify),
            fmt_opt(p.reduce),
            fmt_opt(p.open),
            p.residual,
            fmt_opt(p.outer),
            fmt_opt(p.bind),
            fmt_opt(p.inner),
            fmt_opt(p.forest),
            fmt_opt(p.opener),
            v.total,
            fmt_opt(v.project),
            fmt_opt(v.piop),
            fmt_opt(v.bitify),
            fmt_opt(v.reduce),
            fmt_opt(v.open),
            v.residual,
            self.proof.total() + commitment_bytes,
            self.proof.piop,
            self.proof.open,
            commitment_bytes,
            self.reps,
        ));
        line
    }
}

// ---------------------------------------------------------------------
// Small shared utilities
// ---------------------------------------------------------------------

pub fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty(), "median of an empty sample set");
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    sorted[sorted.len() / 2]
}

/// Median over reps of an optional per-rep value; `Some` only when the value
/// was present in every rep.
fn optional_median(samples: &[Option<f64>]) -> Option<f64> {
    let values: Option<Vec<f64>> = samples.iter().copied().collect();
    values
        .filter(|values| !values.is_empty())
        .map(|values| median(&values))
}

/// Milliseconds elapsed since `start`.
/// Milliseconds from a completed, uniquely named Perfetto operation.
#[cfg(feature = "span-metrics")]
pub fn span_ms(intervals: &[bitz::observability::Interval], label: &str) -> f64 {
    bitz::observability::duration(intervals, label)
        .unwrap_or_else(|error| panic!("invalid benchmark measurement: {error}"))
        .as_secs_f64()
        * 1e3
}

/// Only BitZ callers consult this selector. Competing PCS configurations do not.
pub fn ligerito_selection(target: usize) -> bitz::ligerito_flock::LigeritoSelection {
    ligerito_selection_or(
        target,
        bitz::ligerito_flock::LigeritoSelection::for_target(target),
    )
}

pub fn ligerito_selection_or(
    target: usize,
    default: bitz::ligerito_flock::LigeritoSelection,
) -> bitz::ligerito_flock::LigeritoSelection {
    match std::env::var("BITZ_LIG_PROFILE") {
        Ok(request) => bitz::ligerito_flock::LigeritoSelection::parse(&request, target)
            .expect("invalid BITZ_LIG_PROFILE"),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("invalid BITZ_LIG_PROFILE: {error}"),
    }
}

pub fn ligerito_report(
    resolved: &bitz::ligerito_flock::ResolvedLigerito,
    ood: Option<bitz::ligerito_flock::OodRoundParams>,
) -> serde_json::Value {
    let request = std::env::var("BITZ_LIG_PROFILE").unwrap_or_else(|_| resolved.selection().name());
    resolved.report(&request, ood)
}

pub fn ligerito_identity(
    resolved: &bitz::ligerito_flock::ResolvedLigerito,
    ood: Option<bitz::ligerito_flock::OodRoundParams>,
) -> (String, String) {
    (
        "ligerito_hex".into(),
        bitz::ligerito_flock::ResolvedLigerito::encode_report(&ligerito_report(resolved, ood)),
    )
}

/// Per-trial detail timings, emitted after the captured proof/verification scopes.
/// Retain each label separately so improvements cannot hide a slower inner sumcheck.
pub fn print_regression_phases(phases: &[(String, f64)]) {
    println!(
        "REGRESSION_PHASES {}",
        serde_json::to_string(phases).expect("phase JSON")
    );
}
