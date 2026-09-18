//! Deterministic storage scheduling. Decisions use public geometry only.
use super::ForestSchedule;
use crate::pcs::IntegerMatrixLayout;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchedulePolicy {
    #[default]
    Auto,
    L2,
    L4,
    L8,
}
impl std::str::FromStr for SchedulePolicy {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "l2" => Ok(Self::L2),
            "l4" => Ok(Self::L4),
            "l8" => Ok(Self::L8),
            _ => Err(format!(
                "unknown GKR schedule {value:?}; expected auto, l2, l4, or l8"
            )),
        }
    }
}
impl SchedulePolicy {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::L2 => "l2",
            Self::L4 => "l4",
            Self::L8 => "l8",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum ForestPath {
    Single,
    Multi,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnsupportedSchedule {
    #[error("GKR L2 is unsupported by the multi-claim forest; use auto, l4, or l8")]
    MultiL2,
    #[error("GKR L8 requires forest depth >= 5; got {depth}")]
    ShallowL8 { depth: usize },
}

/// A single shared policy for all forest entry points. The automatic fallback
/// is confirmed by the complete-prover comparisons in docs/gkr-schedule-validation.md.
pub fn resolve_schedule(
    policy: SchedulePolicy,
    layout: &IntegerMatrixLayout,
    path: ForestPath,
    threads: usize,
) -> Result<ForestSchedule, UnsupportedSchedule> {
    let depth = layout.row_vars + layout.word_bits.ilog2() as usize;
    let size = depth + layout.col_vars;
    match (policy, path) {
        (SchedulePolicy::L2, ForestPath::Multi) => Err(UnsupportedSchedule::MultiL2),
        (SchedulePolicy::L8, _) if depth < 5 => Err(UnsupportedSchedule::ShallowL8 { depth }),
        (SchedulePolicy::L2, _) => Ok(ForestSchedule::L2),
        (SchedulePolicy::L8, _) => Ok(ForestSchedule::L8),
        // The AMD-qualified L8 crossover does not hold on Apple Silicon with
        // one worker at these measured geometries. Keep the faster stored
        // levels there; explicit requests and other geometries are unchanged.
        (SchedulePolicy::Auto, ForestPath::Single)
            if cfg!(all(target_arch = "aarch64", target_os = "macos"))
                && threads == 1
                && depth == 13
                && (12..=14).contains(&layout.col_vars) =>
        {
            Ok(ForestSchedule::L2)
        }
        // Nor does it hold on Apple Silicon above four workers: there L8 costs
        // large single-claim forests 19-37% of prover time for ~19% less peak
        // RSS. Keep L4; the multi-claim path and explicit requests are unchanged.
        (SchedulePolicy::Auto, ForestPath::Single)
            if cfg!(all(target_arch = "aarch64", target_os = "macos"))
                && threads > 4
                && depth >= 13
                && size >= 25 =>
        {
            Ok(ForestSchedule::L4)
        }
        // Large, sufficiently deep forests benefit from the smaller stored chain.
        // With few workers, tall forests instead benefit from storing more levels.
        (SchedulePolicy::Auto, _)
            if depth >= 13 && size >= 25 && (threads > 4 || depth <= layout.col_vars + 1) =>
        {
            Ok(ForestSchedule::L8)
        }
        (SchedulePolicy::Auto, ForestPath::Single) if threads <= 4 && depth >= layout.col_vars => {
            Ok(ForestSchedule::L2)
        }
        (SchedulePolicy::Auto | SchedulePolicy::L4, _) => Ok(ForestSchedule::L4),
    }
}
pub(super) fn configured(
    p: &IntegerMatrixLayout,
    path: ForestPath,
) -> Result<ForestSchedule, UnsupportedSchedule> {
    let policy = std::env::var("F2_FOREST_SCHEDULE")
        .unwrap_or_else(|_| "auto".into())
        .parse()
        .expect("valid F2_FOREST_SCHEDULE");
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    let schedule = resolve_schedule(policy, p, path, threads)?;
    #[cfg(feature = "bench-internals")]
    {
        if let Some(records) = RECORDS.lock().expect("schedule recorder lock").as_mut() {
            let record = ScheduleRecord {
                path,
                row_vars: p.row_vars,
                col_vars: p.col_vars,
                word_bits: p.word_bits,
                threads,
                schedule,
            };
            if !records.contains(&record) {
                records.push(record);
            }
        }
    }
    Ok(schedule)
}
impl ForestSchedule {
    pub const fn name(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::L4 => "l4",
            Self::L8 => "l8",
        }
    }
}
#[cfg(feature = "bench-internals")]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScheduleRecord {
    pub path: ForestPath,
    pub row_vars: usize,
    pub col_vars: usize,
    pub word_bits: usize,
    pub threads: usize,
    pub schedule: ForestSchedule,
}
#[cfg(feature = "bench-internals")]
static RECORDS: std::sync::Mutex<Option<Vec<ScheduleRecord>>> = std::sync::Mutex::new(None);
#[cfg(feature = "bench-internals")]
pub fn start_recording() {
    *RECORDS.lock().expect("schedule recorder lock") = Some(Vec::with_capacity(32));
}
#[cfg(feature = "bench-internals")]
pub fn take_records() -> Vec<ScheduleRecord> {
    let mut records = RECORDS
        .lock()
        .expect("schedule recorder lock")
        .take()
        .unwrap_or_default();
    records.sort();
    records
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_policy_covers_measured_crossovers_and_multi_eligibility() {
        // Apple Silicon keeps L4 where other platforms take the many-worker L8.
        let many_worker_l8 = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            ForestSchedule::L4
        } else {
            ForestSchedule::L8
        };
        for (row_vars, col_vars, word_bits, threads, path, expected) in [
            (15, 7, 1, 1, ForestPath::Single, ForestSchedule::L2),
            (12, 7, 8, 1, ForestPath::Single, ForestSchedule::L2),
            (17, 9, 1, 4, ForestPath::Single, ForestSchedule::L2),
            (17, 9, 1, 10, ForestPath::Single, many_worker_l8),
            (15, 7, 4, 8, ForestPath::Single, ForestSchedule::L4),
            (7, 15, 1, 1, ForestPath::Single, ForestSchedule::L4),
            (12, 15, 1, 10, ForestPath::Single, ForestSchedule::L4),
            (13, 12, 1, 8, ForestPath::Single, many_worker_l8),
            (13, 14, 1, 10, ForestPath::Single, many_worker_l8),
            (15, 7, 1, 1, ForestPath::Multi, ForestSchedule::L4),
        ] {
            let layout = IntegerMatrixLayout {
                row_vars,
                col_vars,
                word_bits,
            };
            assert_eq!(
                resolve_schedule(SchedulePolicy::Auto, &layout, path, threads),
                Ok(expected)
            );
            // An explicit supported request always overrides the automatic choice.
            assert_eq!(
                resolve_schedule(SchedulePolicy::L4, &layout, path, threads),
                Ok(ForestSchedule::L4)
            );
        }
    }

    #[test]
    fn apple_single_worker_crossover_is_limited_to_measured_geometry() {
        let apple = cfg!(all(target_arch = "aarch64", target_os = "macos"));
        for (rows, cols, bits, threads, path, apple_schedule) in [
            (13, 12, 1, 1, ForestPath::Single, ForestSchedule::L2),
            (13, 13, 1, 1, ForestPath::Single, ForestSchedule::L2),
            (13, 14, 1, 1, ForestPath::Single, ForestSchedule::L2),
            (10, 13, 8, 1, ForestPath::Single, ForestSchedule::L2),
            (13, 15, 1, 1, ForestPath::Single, ForestSchedule::L8),
            (14, 13, 1, 1, ForestPath::Single, ForestSchedule::L8),
            (13, 13, 1, 2, ForestPath::Single, ForestSchedule::L8),
            (13, 13, 1, 8, ForestPath::Single, ForestSchedule::L4),
            (13, 13, 1, 1, ForestPath::Multi, ForestSchedule::L8),
        ] {
            let layout = IntegerMatrixLayout {
                row_vars: rows,
                col_vars: cols,
                word_bits: bits,
            };
            assert_eq!(
                resolve_schedule(SchedulePolicy::Auto, &layout, path, threads),
                Ok(if apple {
                    apple_schedule
                } else {
                    ForestSchedule::L8
                })
            );
            for (policy, expected) in [
                (SchedulePolicy::L4, ForestSchedule::L4),
                (SchedulePolicy::L8, ForestSchedule::L8),
            ] {
                assert_eq!(
                    resolve_schedule(policy, &layout, path, threads),
                    Ok(expected)
                );
            }
        }
    }

    #[test]
    fn apple_many_worker_crossover_falls_back_to_l4() {
        let apple = cfg!(all(target_arch = "aarch64", target_os = "macos"));
        for (rows, cols, bits, threads, path, apple_schedule) in [
            (13, 12, 1, 5, ForestPath::Single, ForestSchedule::L4),
            (13, 13, 1, 10, ForestPath::Single, ForestSchedule::L4),
            (17, 9, 1, 10, ForestPath::Single, ForestSchedule::L4),
            (21, 4, 1, 16, ForestPath::Single, ForestSchedule::L4),
            // Four workers and the multi-claim path keep the L8 crossover.
            (13, 13, 1, 4, ForestPath::Single, ForestSchedule::L8),
            (13, 13, 1, 10, ForestPath::Multi, ForestSchedule::L8),
            (17, 9, 1, 10, ForestPath::Multi, ForestSchedule::L8),
        ] {
            let layout = IntegerMatrixLayout {
                row_vars: rows,
                col_vars: cols,
                word_bits: bits,
            };
            assert_eq!(
                resolve_schedule(SchedulePolicy::Auto, &layout, path, threads),
                Ok(if apple {
                    apple_schedule
                } else {
                    ForestSchedule::L8
                })
            );
            for (policy, expected) in [
                (SchedulePolicy::L4, ForestSchedule::L4),
                (SchedulePolicy::L8, ForestSchedule::L8),
            ] {
                assert_eq!(
                    resolve_schedule(policy, &layout, path, threads),
                    Ok(expected)
                );
            }
        }
    }

    #[test]
    fn explicit_multi_schedule_is_never_silently_substituted() {
        let p = IntegerMatrixLayout {
            row_vars: 15,
            col_vars: 2,
            word_bits: 1,
        };
        assert_eq!(
            resolve_schedule(SchedulePolicy::L2, &p, ForestPath::Multi, 8),
            Err(UnsupportedSchedule::MultiL2)
        );
        for policy in [SchedulePolicy::Auto, SchedulePolicy::L4, SchedulePolicy::L8] {
            assert!(resolve_schedule(policy, &p, ForestPath::Multi, 8).is_ok());
        }
        for (policy, expected) in [
            (SchedulePolicy::L2, ForestSchedule::L2),
            (SchedulePolicy::L4, ForestSchedule::L4),
            (SchedulePolicy::L8, ForestSchedule::L8),
        ] {
            assert_eq!(
                resolve_schedule(policy, &p, ForestPath::Single, 8),
                Ok(expected)
            );
        }
        let shallow = IntegerMatrixLayout { row_vars: 4, ..p };
        assert_eq!(
            resolve_schedule(SchedulePolicy::L8, &shallow, ForestPath::Single, 1),
            Err(UnsupportedSchedule::ShallowL8 { depth: 4 })
        );
        assert!("typo".parse::<SchedulePolicy>().is_err());
    }
}
