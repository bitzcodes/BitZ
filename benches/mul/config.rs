use anyhow::{Result, ensure};
type List<T> = Vec<T>;
use clap::{Parser, ValueEnum};
use bitz::merged_forest::schedule::SchedulePolicy;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Proof,
    Witness,
    Pcs,
    Piop,
    Outer,
    Bounds,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Workload {
    U32Full,
    U32Mod32,
    U64,
    U128,
    BabyBear,
    Field,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Memory {
    None,
    Rss,
    Heap,
}

/// A scalar, comma-separated list, or inclusive range; duplicates are mistakes.
pub fn integers(s: &str) -> Result<Vec<i64>, String> {
    let mut values = Vec::new();
    for item in s.split(',') {
        let parse = |s: &str| s.parse::<i64>().map_err(|e| e.to_string());
        let (lo, hi) = if let Some((a, b)) = item.split_once("..=") {
            (parse(a)?, parse(b)?)
        } else {
            let n = parse(item)?;
            (n, n)
        };
        if lo > hi || hi.saturating_sub(lo) > 4096 {
            return Err("invalid or excessive range".into());
        }
        for n in lo..=hi {
            if values.contains(&n) {
                return Err(format!("duplicate value {n}"));
            }
            values.push(n);
        }
    }
    Ok(values)
}
fn workloads(s: &str) -> Result<Vec<Workload>, String> {
    let mut values = Vec::new();
    for value in s.split(',') {
        let value = Workload::from_str(value, false)?;
        if values.contains(&value) {
            return Err("duplicate workload".into());
        }
        values.push(value);
    }
    Ok(values)
}
fn schedules(value: &str) -> Result<Vec<SchedulePolicy>, String> {
    let mut result = Vec::new();
    for part in value.split(',') {
        let policy = part.parse()?;
        if result.contains(&policy) {
            return Err(format!("duplicate GKR schedule {part}"));
        }
        result.push(policy);
    }
    Ok(result)
}
#[derive(Parser, Debug)]
#[command(about = "Verified multiplication benchmarks; all experiment settings are flags")]
pub struct Args {
    #[arg(value_enum, default_value = "proof")]
    pub mode: Mode,
    #[arg(long, value_parser = workloads)]
    pub workload: Option<List<Workload>>,
    #[arg(long, value_parser = integers)]
    pub log_n: Option<List<i64>>,
    #[arg(long, value_parser = integers)]
    pub w: Option<List<i64>>,
    #[arg(long, value_parser = integers, allow_hyphen_values = true)]
    pub split: Option<List<i64>>,
    #[arg(long, value_parser = integers)]
    pub threads: Option<List<i64>>,
    #[arg(long, value_parser = integers)]
    pub bitz_profile: Option<List<i64>>,
    #[arg(long)]
    pub backends: Option<String>,
    #[arg(long, value_parser = crate::common::cli::seed)]
    pub seed: Option<u64>,
    #[arg(long, value_parser = crate::common::cli::positive)]
    pub reps: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub warmups: usize,
    #[arg(
        long,
        help = "BitZ opener selections, e.g. custom:1:4,udrg:1:4 (comma-separated)"
    )]
    pub ligerito: Option<String>,
    #[arg(long, default_value = "johnson")]
    pub bound: String,
    #[arg(long, default_value = "current", value_parser = ["current", "regression"])]
    pub preset: String,
    #[arg(long)]
    pub variants: Option<String>,
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=3))]
    pub log_inv_rate: Option<u8>,
    #[arg(long, default_value = "union", value_parser = ["union", "rbr"])]
    pub binius_ligerito_accounting: String,
    #[arg(long)]
    pub whir_degree: Option<usize>,
    #[arg(long)]
    pub whir_folding: Option<usize>,
    #[arg(long)]
    pub whir_pow: Option<usize>,
    #[arg(long)]
    pub whir_rate_cap: Option<usize>,
    #[arg(long, default_value_t = 5, value_parser = crate::common::cli::positive)]
    pub tuning_reps: usize,
    #[arg(long, default_value_t = 100)]
    pub limber_bits: usize,
    #[arg(long, value_enum, default_value = "none")]
    pub memory: Memory,
    #[arg(long, default_value = "auto", value_parser = schedules)]
    pub gkr_schedule: Option<List<SchedulePolicy>>,
    #[arg(long)]
    pub proof_fingerprints: bool,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub skip_unsupported: bool,
    #[arg(long, hide = true)]
    pub bench: bool,
    #[arg(long, hide = true)]
    pub worker: Option<PathBuf>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitzConfig {
    pub gkr_schedule: Option<SchedulePolicy>,
    pub w: usize,
    pub split: i8,
    pub profile: Option<usize>,
    pub bound: Option<String>,
    pub ligerito: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirConfig {
    pub degree: usize,
    pub folding: usize,
    pub log_inv_rate: usize,
    pub max_pow_bits: usize,
    pub max_round_log_inv_rate: Option<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub mode: Mode,
    pub workload: Workload,
    pub backend: String,
    pub log_n: usize,
    pub seed: u64,
    pub threads: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitz: Option<BitzConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_inv_rate: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binius_ligerito_accounting: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limber_bits: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub whir: Option<WhirConfig>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub proof_fingerprints: bool,
    pub id: String,
    pub case: Case,
    pub reps: usize,
    pub warmups: usize,
    pub memory: Memory,
    pub skip: Option<String>,
    pub skip_unsupported: bool,
    pub tuning_reps: usize,
}
const PROOF_BACKENDS: &[&str] = &[
    "bitz",
    "binius64",
    "binius64-ligerito",
    "plonky3-fri",
    "plonky3-whir",
    "limber",
];
const PCS_BACKENDS: &[&str] = &[
    "bitz",
    "plonky3-whir",
    "binius64-basefold",
    "bitz-ligerito-binary",
];
impl Args {
    pub fn expand(&self, compare: bool) -> Result<Vec<Job>> {
        ensure!(
            compare || self.backends.as_deref().is_none_or(|s| s == "bitz"),
            "mul_bitz only accepts backend bitz"
        );
        ensure!(
            !compare || matches!(self.mode, Mode::Proof | Mode::Witness | Mode::Pcs),
            "this experiment belongs to mul_bitz"
        );
        ensure!(
            self.limber_bits > 0 && self.limber_bits <= 128,
            "invalid Limber target"
        );
        ensure!(
            !self.proof_fingerprints || matches!(self.mode, Mode::Proof | Mode::Bounds),
            "--proof-fingerprints requires proof or bounds"
        );
        let bounds: Vec<_> = self.bound.split(',').collect();
        ensure!(
            bounds
                .iter()
                .all(|b| matches!(*b, "johnson" | "unique" | "unique-ungrinded")),
            "unknown decoding bound"
        );
        ensure!(
            self.mode == Mode::Bounds || bounds == ["johnson"],
            "--bound is only applicable to bounds"
        );
        ensure!(
            self.mode == Mode::Outer || self.preset == "current",
            "--preset is only applicable to outer"
        );
        ensure!(
            self.ligerito.is_none() || self.bound == "johnson",
            "use --bound or --ligerito, not both"
        );
        let selections: Vec<Option<&str>> = self
            .ligerito
            .as_deref()
            .map_or_else(|| vec![None], |s| s.split(',').map(Some).collect());
        for request in selections.iter().flatten() {
            bitz::ligerito_flock::LigeritoSelection::parse(request, 100)
                .map_err(anyhow::Error::msg)?;
        }
        let variants: Vec<_> = self
            .variants
            .as_deref()
            .unwrap_or(if self.mode == Mode::Outer && self.preset == "current" {
                "standard,zero,skip1,skip2,skip3,skip4"
            } else {
                "standard,skip1,skip2,skip3,skip4"
            })
            .split(',')
            .collect();
        ensure!(
            variants.iter().all(|v| matches!(
                *v,
                "standard" | "zero" | "skip1" | "skip2" | "skip3" | "skip4"
            )),
            "unknown variant"
        );
        let default_workload = if compare && self.mode != Mode::Pcs {
            Workload::U32Mod32
        } else {
            Workload::U32Full
        };
        let workloads = self.workload.clone().unwrap_or_else(|| {
            if self.mode == Mode::Outer {
                let mut v = vec![Workload::U32Full, Workload::U64, Workload::U128];
                if self.preset == "current" {
                    v.push(Workload::Field);
                }
                v
            } else {
                vec![default_workload]
            }
        });
        let all = if self.mode == Mode::Pcs {
            PCS_BACKENDS
        } else {
            PROOF_BACKENDS
        };
        let backends: Vec<&str> = match self.backends.as_deref() {
            Some("all") => all.to_vec(),
            Some(s) => s.split(',').collect(),
            None if !compare => vec!["bitz"],
            None if self.mode == Mode::Pcs => all.to_vec(),
            None => PROOF_BACKENDS[..4].to_vec(),
        };
        ensure!(
            backends
                .iter()
                .all(|b| PROOF_BACKENDS.contains(b) || PCS_BACKENDS.contains(b)),
            "unknown backend"
        );
        let threads = self.threads.clone().unwrap_or_else(|| {
            vec![if self.mode == Mode::Bounds {
                8
            } else {
                if cfg!(feature = "parallel") {
                    std::thread::available_parallelism().map_or(1, |n| n.get()) as i64
                } else {
                    1
                }
            }]
        });
        ensure!(
            threads.iter().all(|&n| n > 0 && n <= 65536),
            "invalid thread count"
        );
        ensure!(
            cfg!(feature = "parallel") || threads == [1],
            "serial builds require --threads 1"
        );
        ensure!(
            self.whir_degree.is_none_or(|n| [2, 4, 5].contains(&n)),
            "WHIR degree must be 2, 4 or 5"
        );
        ensure!(
            self.whir_folding.is_none_or(|n| (2..=12).contains(&n)),
            "invalid WHIR folding"
        );
        ensure!(
            self.whir_pow.is_none_or(|n| n <= 32),
            "invalid WHIR PoW cap"
        );
        ensure!(
            self.whir_rate_cap.is_none_or(|n| (1..=8).contains(&n)),
            "invalid WHIR rate cap"
        );
        let ws = self.w.clone().unwrap_or_else(|| vec![1]);
        let splits = self.split.clone().unwrap_or_else(|| vec![0]);
        ensure!(
            ws.iter().all(|&w| (1..=126).contains(&w)),
            "W must be in 1..=126 (further shape bounds apply)"
        );
        ensure!(
            splits.iter().all(|&s| i8::try_from(s).is_ok()),
            "split must fit i8"
        );
        if let Some(ps) = &self.bitz_profile {
            ensure!(
                ps.iter().all(|p| [100, 128].contains(p)),
                "profile must be 100 or 128"
            );
        }
        if let Some(ns) = &self.log_n {
            ensure!(
                ns.iter()
                    .all(|&n| (4..=(usize::BITS as i64 - 11)).contains(&n)),
                "log-n is out of range"
            );
        }
        let mut jobs = Vec::new();
        for workload in workloads {
            let ns = self.log_n.clone().unwrap_or_else(|| match self.mode {
                Mode::Witness => vec![10],
                Mode::Outer => vec![12, 15, 17, 19],
                Mode::Proof if compare => vec![15],
                Mode::Piop | Mode::Bounds => vec![15],
                Mode::Pcs if workload == Workload::BabyBear => (15..=24).collect(),
                _ => (15..=25).collect(),
            });
            let profiles = if self.mode == Mode::Witness {
                vec![100]
            } else {
                self.bitz_profile.clone().unwrap_or_else(|| {
                    if workload == Workload::BabyBear && self.mode == Mode::Proof && !compare {
                        vec![100, 128]
                    } else {
                        vec![100]
                    }
                })
            };
            let seed = self.seed.unwrap_or(if self.mode == Mode::Bounds {
                crate::common::mul_witness::U32_SEED
            } else if self.mode == Mode::Outer {
                if self.preset == "regression" {
                    0xa345_0385
                } else {
                    0x5533_326d_756c_0073
                }
            } else if self.mode == Mode::Proof && !compare && workload == Workload::U32Full {
                0x5533_326d_756c_0064
            } else if self.mode == Mode::Proof && !compare && workload == Workload::BabyBear {
                0x6262_6d75_6c5f_0031
            } else {
                match workload {
                    Workload::U32Full | Workload::U32Mod32 => crate::common::mul_witness::U32_SEED,
                    Workload::U64 => crate::common::mul_witness::U64_SEED,
                    Workload::U128 => crate::common::mul_witness::U128_SEED,
                    Workload::BabyBear => crate::common::mul_witness::BABY_BEAR_SEED,
                    Workload::Field => 0x5533_326d_756c_0073,
                }
            });
            for &log_n in &ns {
                for &threads in &threads {
                    for &backend in &backends {
                        let configurations = if backend == "bitz"
                            && !matches!(self.mode, Mode::Outer | Mode::Piop)
                        {
                            let mut c = Vec::new();
                            for &w in &ws {
                                for &split in &splits {
                                    for &profile in &profiles {
                                        for bound in &bounds {
                                            for request in &selections {
                                                let selected = if let Some(request) = request {
                                                    bitz::ligerito_flock::LigeritoSelection::parse(
                                                        request,
                                                        profile as usize,
                                                    )
                                                    .map_err(anyhow::Error::msg)?
                                                } else if self.mode != Mode::Bounds {
                                                    bitz::ligerito_flock::LigeritoSelection::for_target(profile as usize)
                                                } else {
                                                    match *bound {"johnson"=>bitz::ligerito_flock::LigeritoSelection::JOHNSON,"unique"=>bitz::ligerito_flock::LigeritoSelection::MATCHED_UDR,_=>bitz::ligerito_flock::LigeritoSelection::CustomUdr{log_inv_rate:1,initial_k:4,fold_grinding:false}}
                                                };
                                                let f = Some(BitzConfig {
                                                    gkr_schedule: None,
                                                    w: w as usize,
                                                    split: split as i8,
                                                    profile: (self.mode != Mode::Witness)
                                                        .then_some(profile as usize),
                                                    bound: (self.mode != Mode::Witness).then(
                                                        || {
                                                            if selected
                                                                .name()
                                                                .starts_with("custom:")
                                                            {
                                                                "johnson"
                                                            } else {
                                                                "unique"
                                                            }
                                                            .into()
                                                        },
                                                    ),
                                                    ligerito: (self.mode != Mode::Witness)
                                                        .then(|| selected.name()),
                                                });
                                                if !c.contains(&f) {
                                                    c.push(f);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            c
                        } else {
                            vec![None]
                        };
                        let configurations: Vec<_> = configurations
                            .into_iter()
                            .flat_map(|f| {
                                if let Some(f) = &f {
                                    if self.mode != Mode::Witness {
                                        return self
                                            .gkr_schedule
                                            .as_deref()
                                            .unwrap_or(&[SchedulePolicy::Auto])
                                            .iter()
                                            .map(|&policy| {
                                                let mut f = f.clone();
                                                f.gkr_schedule = Some(policy);
                                                Some(f)
                                            })
                                            .collect::<Vec<_>>();
                                    }
                                }
                                vec![f]
                            })
                            .collect();
                        for bitz in configurations {
                            let variants: Vec<_> = if matches!(self.mode, Mode::Outer | Mode::Piop)
                            {
                                variants.iter().map(|v| Some((*v).to_string())).collect()
                            } else {
                                vec![None]
                            };
                            for variant in variants {
                                let case = Case {
                                    mode: self.mode,
                                    workload,
                                    backend: backend.into(),
                                    log_n: log_n as usize,
                                    seed,
                                    threads: threads as usize,
                                    bitz: bitz.clone(),
                                    variant,
                                    preset: (self.mode == Mode::Outer).then(|| self.preset.clone()),
                                    log_inv_rate: if self.mode == Mode::Witness {
                                        None
                                    } else if backend == "plonky3-whir" && self.mode == Mode::Proof
                                    {
                                        self.log_inv_rate
                                    } else {
                                        matches!(
                                            backend,
                                            "binius64"
                                                | "binius64-ligerito"
                                                | "binius64-basefold"
                                                | "plonky3-fri"
                                                | "plonky3-whir"
                                        )
                                        .then_some(self.log_inv_rate.unwrap_or(1))
                                    },
                                    binius_ligerito_accounting: (backend == "binius64-ligerito"
                                        && self.mode == Mode::Proof)
                                        .then(|| self.binius_ligerito_accounting.clone()),
                                    limber_bits: (backend == "limber" && self.mode == Mode::Proof)
                                        .then_some(self.limber_bits),
                                    whir: if backend == "plonky3-whir"
                                        && self.mode != Mode::Witness
                                        && (self.mode == Mode::Pcs
                                            || self.whir_degree.is_some()
                                            || self.whir_folding.is_some()
                                            || self.whir_pow.is_some()
                                            || self.whir_rate_cap.is_some())
                                    {
                                        Some(WhirConfig {
                                            degree: self.whir_degree.unwrap_or(
                                                if self.mode == Mode::Pcs {
                                                    pcs_whir_degree(workload)
                                                } else {
                                                    5
                                                },
                                            ),
                                            folding: self.whir_folding.unwrap_or(
                                                if self.mode == Mode::Pcs {
                                                    ((log_n as usize * 3 / 4).saturating_sub(5))
                                                        .clamp(2, 12)
                                                } else {
                                                    4
                                                },
                                            ),
                                            log_inv_rate: self.log_inv_rate.unwrap_or(1) as usize,
                                            max_pow_bits: self.whir_pow.unwrap_or(12),
                                            max_round_log_inv_rate: self.whir_rate_cap,
                                        })
                                    } else {
                                        None
                                    },
                                };
                                let skip = case.unsupported(compare);
                                if let Some(reason) = &skip {
                                    ensure!(
                                        self.skip_unsupported,
                                        "unsupported {case:?}: {reason}; use --skip-unsupported to record this case"
                                    );
                                }
                                ensure!(
                                    !jobs.iter().any(|j: &Job| j.case == case),
                                    "duplicate case: {case:?}"
                                );
                                jobs.push(Job {
                                    proof_fingerprints: self.proof_fingerprints
                                        && backend == "bitz"
                                        && matches!(self.mode, Mode::Proof | Mode::Bounds),
                                    id: format!("case-{:05}", jobs.len()),
                                    case,
                                    reps: self.reps.unwrap_or(if self.mode == Mode::Pcs {
                                        21
                                    } else {
                                        5
                                    }),
                                    warmups: self.warmups,
                                    memory: self.memory,
                                    skip,
                                    skip_unsupported: self.skip_unsupported,
                                    tuning_reps: self.tuning_reps,
                                });
                            }
                        }
                    }
                }
            }
        }
        ensure!(jobs.iter().any(|j| j.skip.is_none()), "no runnable cases");
        Ok(jobs)
    }
}
fn pcs_whir_degree(workload: Workload) -> usize {
    if workload == Workload::BabyBear && cfg!(feature = "plonky3-whir-degree4-bench") {
        4
    } else if workload == Workload::U32Full
        && cfg!(feature = "plonky3-whir-goldilocks-degree2-bench")
    {
        2
    } else {
        5
    }
}
impl Case {
    pub fn unsupported(&self, compare: bool) -> Option<String> {
        let reason = if self.workload == Workload::Field
            && (self.mode != Mode::Outer || self.preset.as_deref() == Some("regression"))
        {
            "field input belongs to the current outer kernel experiment"
        } else if self.mode == Mode::Outer
            && self.preset.as_deref() == Some("regression")
            && self.variant.as_deref() == Some("zero")
        {
            "regression uses standard or a prefix skip"
        } else if self.mode == Mode::Pcs
            && !matches!(self.workload, Workload::U32Full | Workload::BabyBear)
        {
            "PCS supports u32-full and baby-bear"
        } else if self.mode == Mode::Pcs && !PCS_BACKENDS.contains(&self.backend.as_str()) {
            "backend has no PCS adapter"
        } else if self.backend == "plonky3-whir"
            && self.mode == Mode::Pcs
            && self.whir.is_some_and(|p| {
                p.degree != pcs_whir_degree(self.workload) || p.max_round_log_inv_rate.is_some()
            })
        {
            "PCS WHIR uses its compiled extension degree and does not support --whir-rate-cap"
        } else if self.backend == "plonky3-whir"
            && self.mode == Mode::Proof
            && self.whir.is_some_and(|p| p.degree == 4)
        {
            "native WHIR supports extension degrees 2 and 5"
        } else if self.mode == Mode::Piop && self.workload != Workload::U32Full {
            "whole PIOP experiment supports u32-full"
        } else if self.mode == Mode::Piop && self.variant.as_deref() == Some("zero") {
            "whole PIOP uses standard or a prefix skip"
        } else if self.mode == Mode::Outer && self.workload == Workload::BabyBear {
            "outer experiment supports integer workloads"
        } else if self.backend != "bitz"
            && self.mode != Mode::Pcs
            && matches!(self.workload, Workload::BabyBear | Workload::U32Full)
        {
            "native comparisons use u32-mod32, u64, or u128"
        } else if self.mode != Mode::Pcs
            && self.workload != Workload::U32Mod32
            && matches!(self.backend.as_str(), "plonky3-fri" | "plonky3-whir")
        {
            "Plonky3 native multiplication supports u32-mod32"
        } else if self.mode != Mode::Pcs && !PROOF_BACKENDS.contains(&self.backend.as_str()) {
            "backend has no native proof adapter"
        } else if !compare && self.mode != Mode::Bounds && self.workload == Workload::U32Mod32 {
            "standalone BitZ proves the full u32 product; use u32-full"
        } else if self.log_n < 15
            && matches!(self.mode, Mode::Proof | Mode::Pcs | Mode::Bounds)
            && self.backend == "bitz"
        {
            "BitZ opening needs log-n >= 15"
        } else if let Some(f) = &self.bitz {
            if (self.mode == Mode::Pcs || self.workload == Workload::BabyBear)
                && (f.w != 1 || f.split != 0)
            {
                "this adapter requires W=1 and split=0"
            } else if self.mode == Mode::Pcs && f.profile != Some(100) {
                "PCS uses the fixed Lambda100 terminal opening"
            } else {
                return self.layout_error();
            }
        } else {
            return None;
        };
        Some(reason.into())
    }
    fn layout_error(&self) -> Option<String> {
        use bitz::piop::spartan::mul::MulLayout;
        let f = self.bitz.as_ref()?;
        let n = 1usize << self.log_n;
        fn check<S: bitz::piop::spartan::protocol::RelationSpec>(
            spec: S,
            mode: Mode,
            config: &BitzConfig,
            threads: usize,
        ) -> Result<(), String> {
            use bitz::piop::spartan::{Lambda100, Lambda128, protocol::instantiate_profile};
            if mode == Mode::Witness {
                return Ok(());
            }
            if let Some(policy) = config.gkr_schedule {
                use bitz::merged_forest::schedule::{ForestPath, resolve_schedule};
                resolve_schedule(policy, &spec.opening_layout(), ForestPath::Single, threads)
                    .map_err(|e| e.to_string())?;
            }
            if config.profile == Some(128) {
                instantiate_profile::<Lambda128, _>(&spec)
            } else {
                instantiate_profile::<Lambda100, _>(&spec)
            }
            .map(|_| ())
            .map_err(|e| e.to_string())
        }
        let result = match self.workload {
            Workload::U32Full | Workload::U32Mod32 => MulLayout::<u32>::new_with_word_bits(n, f.w)
                .and_then(|l| l.with_split_shift(f.split))
                .map_err(|e| e.to_string())
                .and_then(|l| check(l, self.mode, f, self.threads)),
            Workload::U64 => MulLayout::<u64>::new_with_word_bits(n, f.w)
                .and_then(|l| l.with_split_shift(f.split))
                .map_err(|e| e.to_string())
                .and_then(|l| check(l, self.mode, f, self.threads)),
            Workload::U128 => MulLayout::<u128>::new_with_word_bits(n, f.w)
                .and_then(|l| l.with_split_shift(f.split))
                .map_err(|e| e.to_string())
                .and_then(|l| check(l, self.mode, f, self.threads)),
            Workload::BabyBear => bitz::piop::spartan::baby_bear_mul::BabyBearMulLayout::new(n)
                .map_err(|e| e.to_string())
                .and_then(|l| check(l, self.mode, f, self.threads)),
            Workload::Field => return None,
        };
        if let Err(error) = result {
            return Some(error.to_string());
        }
        if self.mode != Mode::Witness {
            let packed = match self.workload {
                Workload::U32Full | Workload::U32Mod32 | Workload::BabyBear => self.log_n,
                Workload::U64 => self.log_n + 1,
                Workload::U128 => self.log_n + 2,
                _ => return None,
            };
            let selection = bitz::ligerito_flock::LigeritoSelection::parse(
                f.ligerito.as_deref().expect("proof selection"),
                f.profile.expect("profile"),
            )
            .expect("validated selection");
            return selection.resolve(packed, f.profile.expect("profile")).err();
        }
        None
    }
}
