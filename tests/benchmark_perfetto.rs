#[path = "../benches/common/output.rs"]
mod output;
#[path = "../src/observability.rs"]
mod perfetto;

use output::{BenchmarkOutput, FileMode};
use perfetto::Recording;
use std::{
    io::{self, Write},
    path::Path,
    process::Command,
    sync::{Arc, Barrier, Mutex},
};
use tracing_subscriber::prelude::*;

// The native producer is process-global. Keep test sessions isolated.
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn typed_duration_queries_reject_ambiguity_and_union_parallel_occurrences() {
    let interval = |name: &str, start_ns, end_ns| perfetto::Interval {
        id: 0,
        parent: None,
        track_id: 0,
        depth: 0,
        name: name.into(),
        component: None,
        start_ns,
        end_ns,
    };
    let spans = vec![
        interval("trial", 0, 100),
        interval("work", 10, 30),
        interval("work", 20, 40),
        interval("work", 60, 70),
    ];
    assert_eq!(perfetto::duration(&spans, "trial").unwrap().as_nanos(), 100);
    assert!(perfetto::duration(&spans, "work").is_err());
    assert!(perfetto::duration(&spans, "absent").is_err());
    assert_eq!(
        perfetto::totals(&spans),
        vec![("trial".into(), 100.0 / 1e9), ("work".into(), 40.0 / 1e9)]
    );
}

#[test]
#[ignore = "requires PERFETTO_TRACE_PROCESSOR; exercises the real native query engine"]
fn measured_operation_returns_value_and_exact_span_duration() {
    let _lock = TEST_LOCK.lock().unwrap();
    tracing::subscriber::with_default(
        tracing_subscriber::registry().with(perfetto::layer()),
        || {
            let (result, elapsed) = perfetto::measure(tracing::info_span!("setup"), || {
                tracing::info_span!("child").in_scope(|| std::hint::black_box(42))
            })
            .unwrap();
            assert_eq!(result, 42);
            assert!(!elapsed.is_zero());
        },
    );
    tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
        assert!(
            perfetto::measure(tracing::info_span!("disabled"), || panic!(
                "must not execute"
            ))
            .is_err()
        );
    });
}

#[test]
fn shared_output_preserves_creation_policy() {
    let _lock = TEST_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let output = BenchmarkOutput::new(dir.path());
    let subscriber = tracing_subscriber::registry().with(perfetto::layer());
    tracing::subscriber::with_default(subscriber, || {
        for trial in 0..2 {
            let name = format!("trial-{trial}.pftrace");
            let recording =
                Recording::start(output.buffered(&name, FileMode::CreateNew).unwrap()).unwrap();
            tracing::info_span!("trial", trial).in_scope(|| std::hint::black_box(42));
            recording.finish().unwrap();
            let bytes = std::fs::read(dir.path().join(&name)).unwrap();
            let fields = perfetto_sdk::pb_decoder::PbDecoder::new(&bytes)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert!(!fields.is_empty(), "SDK must emit valid protobuf packets");
            assert_eq!(
                output.file(&name, FileMode::CreateNew).unwrap_err().kind(),
                io::ErrorKind::AlreadyExists
            );
        }
    });
}

#[test]
#[ignore = "requires PERFETTO_TRACE_PROCESSOR; exercises the real native query engine"]
fn shared_protocol_spans_preserve_relation_labels_and_parentage() {
    let _lock = TEST_LOCK.lock().unwrap();
    let u32_scopes = bitz::protocol_scopes!("u32-spartan-bitz");
    let u64_scopes = bitz::protocol_scopes!("u64-spartan-bitz");
    let intervals = tracing::subscriber::with_default(
        tracing_subscriber::registry().with(perfetto::layer()),
        || {
            let recording = Recording::start(Vec::new()).unwrap();
            tracing::info_span!("step3:piop_prove").in_scope(|| {
                (u32_scopes.spartan_prove)().in_scope(|| std::hint::black_box(32));
                (u64_scopes.spartan_prove)().in_scope(|| std::hint::black_box(64));
            });
            recording.intervals().unwrap()
        },
    );
    let parent = perfetto::span(&intervals, "step3:piop_prove").unwrap();
    for label in [
        "u32-spartan-bitz:spartan_prove",
        "u64-spartan-bitz:spartan_prove",
    ] {
        let child = perfetto::span(&intervals, label).unwrap();
        assert_eq!(child.parent, Some(parent.id));
        assert!(child.start_ns >= parent.start_ns && child.end_ns <= parent.end_ns);
        assert!(child.end_ns > child.start_ns);
    }
}

struct FailingWriter {
    fail_write: bool,
}

#[test]
fn profile_projection_preserves_union_totals_memory_and_output_errors() {
    let interval = |id, parent, depth, name: &str, start_ns, end_ns| perfetto::Interval {
        id,
        parent,
        track_id: id,
        depth,
        name: name.into(),
        component: None,
        start_ns,
        end_ns,
    };
    let spans = vec![
        interval(1, None, 0, "trial", 0, 100_000_000),
        interval(2, Some(1), 1, "work", 10_000_000, 40_000_000),
        interval(3, Some(1), 1, "work", 30_000_000, 60_000_000),
        interval(4, Some(2), 2, "child", 20_000_000, 30_000_000),
    ];
    let memory = std::collections::BTreeMap::from([(
        "trial".into(),
        perfetto::memory::PeakGrowth {
            bytes: 1048576,
            peak_bytes: 2097152,
        },
    )]);
    let mut output = Vec::new();
    perfetto::write_profile(&mut output, "fixture", &spans, Some(&memory)).unwrap();
    let output = String::from_utf8(output).unwrap();
    assert!(
        output.contains("trial: 100.000ms (100.0%) self=50.000ms Δrss=1.0 MiB peak=2.0 MiB"),
        "{output}"
    );
    assert!(
        output.contains("work: 50.000ms (50.0%) n=2 self=40.000ms"),
        "{output}"
    );
    let recursive = vec![
        interval(1, None, 0, "work", 0, 100_000_000),
        interval(2, Some(1), 1, "work", 20_000_000, 60_000_000),
        interval(3, Some(2), 2, "child", 30_000_000, 40_000_000),
    ];
    let mut output = Vec::new();
    perfetto::write_profile(&mut output, "recursive", &recursive, None).unwrap();
    let output = String::from_utf8(output).unwrap();
    assert!(
        output.contains("work: 100.000ms (100.0%) n=2 self=90.000ms"),
        "{output}"
    );
    for fail_write in [true, false] {
        let error = perfetto::write_profile(FailingWriter { fail_write }, "fixture", &spans, None)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            if fail_write {
                "injected write failure"
            } else {
                "injected flush failure"
            }
        );
    }
}
impl Write for FailingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.fail_write {
            Err(io::Error::other("injected write failure"))
        } else {
            Ok(bytes.len())
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("injected flush failure"))
    }
}

#[test]
fn write_and_flush_errors_are_returned() {
    let _lock = TEST_LOCK.lock().unwrap();
    for fail_write in [true, false] {
        let recording = Recording::start(FailingWriter { fail_write }).unwrap();
        let error = recording.finish().unwrap_err();
        assert_eq!(
            error.to_string(),
            if fail_write {
                "injected write failure"
            } else {
                "injected flush failure"
            }
        );
    }
}

fn query(trace: &Path, sql: &str) -> Vec<csv::StringRecord> {
    let executable = std::env::var_os("PERFETTO_TRACE_PROCESSOR")
        .expect("set PERFETTO_TRACE_PROCESSOR to the native trace_processor_shell binary");
    let output = Command::new(executable)
        .arg(trace)
        .args(["-Q", sql])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    csv::Reader::from_reader(output.stdout.as_slice())
        .records()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn scalar(trace: &Path, sql: &str) -> i64 {
    let rows = query(trace, sql);
    assert_eq!(rows.len(), 1);
    rows[0][0].parse().unwrap()
}

#[test]
#[ignore = "requires PERFETTO_TRACE_PROCESSOR; exercises the real native query engine"]
fn native_processor_validates_intervals_trials_parallelism_and_tuning_sessions() {
    let _lock = TEST_LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let output = BenchmarkOutput::new(dir.path());
    let subscriber = tracing_subscriber::registry().with(perfetto::layer());
    let dispatch = tracing::Dispatch::new(subscriber);
    tracing::dispatcher::with_default(&dispatch, || {
        let outer = Recording::start(
            output
                .buffered("tuning.pftrace", FileMode::CreateNew)
                .unwrap(),
        )
        .unwrap();
        let tuning = tracing::info_span!("tuning").entered();
        for trial in 0..2 {
            let path = dir.path().join(format!("candidate-{trial}.pftrace"));
            let recording =
                Recording::start(output.buffered(&path, FileMode::CreateNew).unwrap()).unwrap();
            tracing::info_span!("trial", trial, warmup = trial == 0).in_scope(|| {
                // The same span may be entered repeatedly and on parallel threads.
                let operation = tracing::info_span!(
                    "operation",
                    component = "test.operation",
                    result = tracing::field::Empty
                );
                operation.in_scope(|| std::hint::black_box(1));
                operation.record("result", 42_u64);
                operation.in_scope(|| std::hint::black_box(2));
                let barrier = Arc::new(Barrier::new(2));
                std::thread::scope(|scope| {
                    for _ in 0..2 {
                        let barrier = Arc::clone(&barrier);
                        let operation = &operation;
                        let dispatch = &dispatch;
                        scope.spawn(move || {
                            tracing::dispatcher::with_default(dispatch, || {
                                operation.in_scope(|| {
                                    barrier.wait();
                                });
                            })
                        });
                    }
                });
                let failed: Result<(), &str> =
                    tracing::info_span!("failed_operation").in_scope(|| Err("expected"));
                assert!(failed.is_err());
                // Creation/lifetime must not become an execution interval.
                let _unused = tracing::info_span!("never_entered");
                tracing::info!(answer = 42_u64, "candidate completed");
            });
            recording.finish().unwrap();
            // A completed candidate is queryable while the outer tuning span
            // and outer recording remain open. No application clock is read.
            let intervals = perfetto::TraceProcessor::from_env()
                .intervals(&std::fs::read(&path).unwrap())
                .unwrap();
            let operations: Vec<_> = intervals
                .iter()
                .filter(|span| span.component.as_deref() == Some("test.operation"))
                .collect();
            assert_eq!(operations.len(), 4);
            assert_eq!(
                operations
                    .iter()
                    .map(|span| span.id)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                4
            );
            assert_eq!(
                operations
                    .iter()
                    .map(|span| span.track_id)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                3
            );
            assert!(scalar(&path, "SELECT dur FROM slice WHERE name = 'trial'") > 0);
            assert_eq!(
                scalar(&path, "SELECT COUNT(*) FROM slice WHERE name = 'operation'"),
                4
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT COUNT(DISTINCT track_id) FROM slice WHERE name = 'operation'"
                ),
                3
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT COUNT(*) FROM slice WHERE name = 'never_entered' OR dur < 0"
                ),
                0
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT EXTRACT_ARG(arg_set_id, 'debug.trial') FROM slice WHERE name = 'trial'"
                ),
                trial
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT EXTRACT_ARG(arg_set_id, 'debug.warmup') FROM slice WHERE name = 'trial'"
                ),
                i64::from(trial == 0)
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT COUNT(*) FROM slice WHERE name = 'operation' AND EXTRACT_ARG(arg_set_id, 'debug.result') = 42"
                ),
                3
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT COUNT(*) FROM slice WHERE name = 'operation' AND EXTRACT_ARG(arg_set_id, 'debug.component') = 'test.operation'"
                ),
                4
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT COUNT(*) FROM slice WHERE name = 'failed_operation' AND dur >= 0"
                ),
                1
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT COUNT(*) FROM slice c JOIN slice p ON c.parent_id = p.id WHERE c.name = 'operation' AND (c.ts < p.ts OR c.ts + c.dur > p.ts + p.dur)"
                ),
                0
            );
            let work = scalar(&path, "SELECT SUM(dur) FROM slice WHERE name = 'operation'");
            let union = scalar(
                &path,
                "WITH intervals AS (SELECT ts AS lo, ts + dur AS hi FROM slice WHERE name = 'operation'), covered AS (SELECT *, MAX(hi) OVER (ORDER BY lo, hi ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_hi FROM intervals) SELECT SUM(MAX(0, hi - MAX(lo, COALESCE(previous_hi, lo)))) FROM covered",
            );
            assert!(
                union > 0 && union < work,
                "parallel wall time must exclude overlap"
            );
            assert_eq!(
                scalar(
                    &path,
                    "SELECT COUNT(*) FROM stats WHERE value != 0 AND (severity IN ('error', 'data_loss'))"
                ),
                0
            );
        }
        drop(tuning);
        outer.finish().unwrap();
        let path = dir.path().join("tuning.pftrace");
        assert_eq!(
            scalar(&path, "SELECT COUNT(*) FROM slice WHERE name = 'trial'"),
            2
        );
        assert!(scalar(&path, "SELECT dur FROM slice WHERE name = 'tuning'") > 0);
        assert_eq!(scalar(&path, "SELECT COUNT(*) FROM slice WHERE dur < 0"), 0);
    });
}

#[test]
#[ignore = "requires PERFETTO_TRACE_PROCESSOR; exercises the native measurement backend"]
fn native_query_rejects_an_unclosed_trial() {
    let _lock = TEST_LOCK.lock().unwrap();
    let subscriber = tracing_subscriber::registry().with(perfetto::layer());
    tracing::subscriber::with_default(subscriber, || {
        let recording = Recording::start(Vec::new()).unwrap();
        let trial = tracing::info_span!("unfinished_trial").entered();
        let result = recording.intervals();
        drop(trial);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("incomplete or lossy")
        );
    });
}
