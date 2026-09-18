#![allow(dead_code)]
#[path = "../benches/common/mod.rs"]
mod common;
#[path = "../benches/mul/mod.rs"]
mod mul;
use clap::Parser;
use mul::config::{Args, Mode, integers};

fn parse(args: &[&str]) -> Args {
    Args::try_parse_from(std::iter::once("mul").chain(args.iter().copied())).unwrap()
}
#[test]
fn inclusive_lists_are_strict() {
    assert_eq!(integers("1,3..=5,-2..=-1").unwrap(), [1, 3, 4, 5, -2, -1]);
    for invalid in ["", "1,", "1,1", "3..=2", "2..4", "1..=5000", "one"] {
        assert!(integers(invalid).is_err(), "{invalid}");
    }
}
#[test]
fn comprehensive_bitz_expansion_and_worker_roundtrip() {
    let args = parse(&[
        "proof",
        "--workload",
        "u32-full,u64,u128",
        "--log-n",
        "15..=20",
        "--w",
        "1,3,8",
        "--split=0,1",
        "--threads",
        "1,8",
        "--bitz-profile",
        "100,128",
        "--reps",
        "5",
        "--dry-run",
        "--skip-unsupported",
        "--bench",
    ]);
    let jobs = args.expand(false).unwrap();
    assert_eq!(jobs.len(), 3 * 6 * 3 * 2 * 2 * 2);
    for job in jobs {
        assert_eq!(job.case.mode, Mode::Proof);
        if job.case.workload == mul::config::Workload::U32Full
            && job.case.log_n == 15
            && job
                .case
                .bitz
                .as_ref()
                .is_some_and(|f| f.w == 8 && f.profile == Some(128))
        {
            assert!(
                job.skip
                    .as_ref()
                    .is_some_and(|s| s.contains("projection-draw"))
            );
        }
        let decoded: mul::config::Job =
            serde_json::from_str(&serde_json::to_string(&job).unwrap()).unwrap();
        assert_eq!(decoded.case, job.case);
    }
}
#[test]
fn bitz_axes_do_not_duplicate_competitors() {
    let jobs = parse(&[
        "proof",
        "--skip-unsupported",
        "--workload",
        "u64",
        "--backends",
        "bitz,binius64",
        "--log-n",
        "15",
        "--w",
        "1,3,8",
        "--split=0,1",
        "--bitz-profile",
        "100,128",
        "--threads",
        "1",
    ])
    .expand(true)
    .unwrap();
    assert_eq!(jobs.len(), 13);
    let competitor = jobs.iter().find(|j| j.case.backend == "binius64").unwrap();
    assert!(competitor.case.bitz.is_none());
}
#[test]
fn witness_omits_unused_security_axes() {
    let jobs = parse(&[
        "witness",
        "--workload",
        "u64",
        "--w",
        "1,3",
        "--bitz-profile",
        "100,128",
        "--threads",
        "1",
    ])
    .expand(false)
    .unwrap();
    assert_eq!(jobs.len(), 2);
    assert_eq!(jobs[0].case.log_n, 10);
    assert!(
        jobs.iter()
            .all(|j| j.case.bitz.as_ref().unwrap().profile.is_none())
    );
}
#[test]
fn capabilities_reject_or_record_without_hiding_malformed_arguments() {
    let arguments = [
        "proof",
        "--workload",
        "u64,u128",
        "--backends",
        "all",
        "--log-n",
        "15",
        "--threads",
        "1",
    ];
    assert!(parse(&arguments).expand(true).is_err());
    let mut args = parse(&arguments);
    args.skip_unsupported = true;
    let jobs = args.expand(true).unwrap();
    assert_eq!(jobs.len(), 12);
    assert_eq!(jobs.iter().filter(|j| j.skip.is_some()).count(), 4);
    for flags in [
        vec!["--w", "0"],
        vec!["--bitz-profile", "101"],
        vec!["--backends", "typo"],
        vec!["--threads", "0"],
    ] {
        let mut a = parse(&flags);
        a.skip_unsupported = true;
        assert!(a.expand(true).is_err());
    }
}
#[test]
fn mode_defaults_and_packing_constraints() {
    let jobs = parse(&[
        "proof",
        "--workload",
        "baby-bear",
        "--threads",
        "1",
        "--skip-unsupported",
    ])
    .expand(false)
    .unwrap();
    assert_eq!(jobs.len(), 22);
    assert_eq!(jobs[0].case.log_n, 15);
    assert_eq!(jobs.last().unwrap().case.log_n, 25);
    let outer = parse(&[
        "outer",
        "--workload",
        "u32-full",
        "--log-n",
        "12",
        "--threads",
        "1",
    ])
    .expand(false)
    .unwrap();
    assert_eq!(outer.len(), 6);
    assert!(
        outer
            .iter()
            .any(|j| j.case.variant.as_deref() == Some("zero"))
    );
    let bounds = parse(&[
        "bounds",
        "--workload",
        "u32-mod32,u32-full,u64,u128,baby-bear",
        "--threads",
        "1",
    ])
    .expand(false)
    .unwrap();
    assert_eq!(bounds.len(), 5);
    assert!(bounds.iter().all(|j| j.case.seed == bounds[0].case.seed));
    assert!(
        parse(&["proof", "--workload", "u64", "--w", "126", "--threads", "1"])
            .expand(false)
            .is_err()
    );
    let profiles = parse(&[
        "proof",
        "--log-n",
        "15",
        "--bitz-profile",
        "128",
        "--threads",
        "1",
    ])
    .expand(false)
    .unwrap();
    assert_eq!(
        profiles[0].case.bitz.as_ref().unwrap().bound.as_deref(),
        Some("unique")
    );
}

#[test]
fn whir_configuration_is_applied_or_rejected() {
    let args = [
        "proof",
        "--workload",
        "u32-mod32",
        "--backends",
        "plonky3-whir",
        "--log-n",
        "15",
        "--threads",
        "1",
    ];
    let auto = parse(&args).expand(true).unwrap();
    assert!(auto[0].case.log_inv_rate.is_none());
    assert!(auto[0].case.whir.is_none());
    let mut explicit = parse(&args);
    explicit.log_inv_rate = Some(3);
    assert_eq!(explicit.expand(true).unwrap()[0].case.log_inv_rate, Some(3));
    explicit.whir_degree = Some(4);
    assert!(explicit.expand(true).is_err());
    let mut pcs = parse(&[
        "pcs",
        "--backends",
        "plonky3-whir",
        "--log-n",
        "15",
        "--threads",
        "1",
    ]);
    let default = pcs.expand(true).unwrap()[0].case.whir.unwrap();
    pcs.whir_degree = Some(if default.degree == 5 { 2 } else { 5 });
    assert!(pcs.expand(true).is_err());
    pcs.whir_degree = None;
    pcs.whir_rate_cap = Some(2);
    assert!(pcs.expand(true).is_err());
}

#[test]
fn phase_records_use_milliseconds_and_union_overlaps() {
    use bitz::observability::Interval;
    let interval = |id, name: &str, start_ns, end_ns| Interval {
        id,
        parent: None,
        track_id: id,
        depth: 0,
        name: name.into(),
        component: None,
        start_ns,
        end_ns,
    };
    let intervals = [
        interval(1, "prove", 0, 10_000_000),
        interval(2, "opening", 1_000_000, 3_000_000),
        interval(3, "opening", 2_000_000, 4_000_000),
        interval(4, "bitify", 5_000_000, 7_000_000),
    ];
    let phases: std::collections::BTreeMap<_, _> = mul::phase_milliseconds(&intervals, "prove")
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(phases["opening"], 3.);
    assert_eq!(phases["bitify"], 2.);
    let commit_ms = 4.;
    assert_eq!(commit_ms + phases["opening"] + phases["bitify"], 9.);
}

#[test]
fn accounting_is_part_of_only_the_applicable_case() {
    let flags = [
        "proof",
        "--workload",
        "u32-mod32",
        "--backends",
        "binius64,binius64-ligerito",
        "--log-n",
        "15",
        "--threads",
        "1",
        "--binius-ligerito-accounting",
        "rbr",
    ];
    let args = parse(&flags);
    let jobs = args.expand(true).unwrap();
    assert!(jobs[0].case.binius_ligerito_accounting.is_none());
    assert_eq!(
        jobs[1].case.binius_ligerito_accounting.as_deref(),
        Some("rbr")
    );
    let mut witness = args;
    witness.mode = Mode::Witness;
    assert!(
        witness
            .expand(true)
            .unwrap()
            .iter()
            .all(|j| j.case.binius_ligerito_accounting.is_none())
    );
}

#[test]
fn schedules_expand_only_where_gkr_runs() {
    use bitz::merged_forest::schedule::SchedulePolicy;
    let jobs = parse(&[
        "proof",
        "--workload",
        "u64",
        "--log-n",
        "15",
        "--threads",
        "1",
        "--backends",
        "bitz,binius64",
        "--gkr-schedule",
        "auto,l2,l4,l8",
        "--proof-fingerprints",
    ])
    .expand(true)
    .unwrap();
    assert_eq!(jobs.len(), 5);
    assert_eq!(jobs.iter().filter(|j| j.proof_fingerprints).count(), 4);
    assert_eq!(
        jobs[0].case.bitz.as_ref().unwrap().gkr_schedule,
        Some(SchedulePolicy::Auto)
    );
    let witness = parse(&["witness", "--threads", "1", "--gkr-schedule", "l2,l4,l8"])
        .expand(false)
        .unwrap();
    assert_eq!(witness.len(), 1);
    assert_eq!(witness[0].case.bitz.as_ref().unwrap().gkr_schedule, None);
    for value in ["", "l2,l2", "fastest"] {
        assert!(Args::try_parse_from(["mul", "--gkr-schedule", value]).is_err());
    }
}

#[test]
fn unachievable_security_profile_is_a_preflight_skip_only_when_requested() {
    let flags = [
        "proof",
        "--workload",
        "u32-full",
        "--log-n",
        "15",
        "--threads",
        "1",
        "--w",
        "1,8",
        "--bitz-profile",
        "128",
    ];
    assert!(parse(&flags).expand(false).is_err());
    let mut args = parse(&flags);
    args.skip_unsupported = true;
    let jobs = args.expand(false).unwrap();
    assert_eq!(jobs.len(), 2);
    assert!(jobs[0].skip.is_none());
    assert!(jobs[1].skip.as_ref().unwrap().contains("projection-draw"));
}
