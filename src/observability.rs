//! Perfetto recording and typed interval queries. The SDK owns all clocks,
//! thread tracks, and buffering; the native trace processor reconstructs slices.

#[path = "observability/memory.rs"]
pub mod memory;

use perfetto_sdk::{
    heap_buffer::HeapBuffer,
    pb_msg::{PbMsg, PbMsgWriter},
    protos::config::{
        data_source_config::DataSourceConfig,
        trace_config::{
            BufferConfigFillPolicy, TraceConfig, TraceConfigBufferConfig, TraceConfigDataSource,
        },
        track_event::track_event_config::TrackEventConfig,
    },
    tracing_session::TracingSession,
};
use std::{
    collections::HashMap,
    ffi::OsString,
    io::{self, Write},
    process::{Command, Stdio},
    sync::{Arc, Mutex, Once, mpsc},
    time::Duration,
};

const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// Install once at an executable boundary. Libraries should compose `layer()`
/// with their caller's subscriber instead of replacing it.
pub fn install() -> io::Result<()> {
    use tracing_subscriber::prelude::*;
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer()))
        .map_err(io::Error::other)
}

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(tracing_perfetto_sdk::init_in_process);
}

/// Compose with the application's existing subscriber; never install a second one.
pub fn layer() -> tracing_perfetto_sdk::PerfettoLayer {
    init();
    tracing_perfetto_sdk::PerfettoLayer::new()
}

#[must_use = "close all spans, then call finish to flush and save the trace"]
pub struct Recording<W> {
    session: TracingSession,
    writer: W,
}

impl<W: Write + Send + 'static> Recording<W> {
    /// Supply a writer opened through `BenchmarkOutput`. Session startup is
    /// outside the measured region. Keep each recording bounded to one trial.
    pub fn start(writer: W) -> io::Result<Self> {
        init();
        let mut session = TracingSession::in_process().map_err(io::Error::other)?;
        session.setup(&trace_config());
        session.start_blocking();
        Ok(Self { session, writer })
    }

    /// Flush producers before stopping, then save through the supplied writer.
    /// Use the asynchronous flush API because the blocking SDK API hides failure.
    pub fn finish(self) -> io::Result<()> {
        self.into_inner().map(|_| ())
    }

    /// Complete capture and recover the flushed writer (including an in-memory sink).
    pub fn into_inner(mut self) -> io::Result<W> {
        let (send, receive) = mpsc::sync_channel(1);
        self.session.flush_async(FLUSH_TIMEOUT, move |success| {
            let _ = send.send(success);
        });
        let flushed = receive.recv_timeout(FLUSH_TIMEOUT + Duration::from_secs(1));
        self.session.stop_blocking();
        if !matches!(flushed, Ok(true)) {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Perfetto producer flush failed",
            ));
        }

        // The SDK invokes its callback on an internal thread and cannot return
        // I/O errors. Retain the first error and report it after the blocking read.
        let result = Arc::new(Mutex::new(Ok(self.writer)));
        let sink = Arc::clone(&result);
        self.session.read_trace_blocking(move |data, _has_more| {
            if let Ok(mut result) = sink.lock()
                && let Ok(writer) = &mut *result
                && let Err(error) = writer.write_all(data)
            {
                *result = Err(error);
            }
        });
        let result = Arc::try_unwrap(result)
            .map_err(|_| io::Error::other("Perfetto retained the completed read callback"))?;
        let mut writer = result
            .into_inner()
            .map_err(|_| io::Error::other("Perfetto sink poisoned"))??;
        writer.flush()?;
        Ok(writer)
    }
}

impl Recording<Vec<u8>> {
    /// Query only after the measured scopes have exited. No temporary trace file
    /// or Python process is needed on the supported Unix benchmark platforms.
    pub fn intervals(self) -> io::Result<Vec<Interval>> {
        TraceProcessor::from_env().intervals(&self.into_inner()?)
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Interval {
    pub id: u64,
    pub parent: Option<u64>,
    pub track_id: u64,
    pub depth: usize,
    pub name: String,
    pub component: Option<String>,
    pub start_ns: u64,
    pub end_ns: u64,
}

impl Interval {
    /// Dynamic operation identities use `component`; ordinary spans use their name.
    pub fn label(&self) -> &str {
        self.component.as_deref().unwrap_or(&self.name)
    }

    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.end_ns - self.start_ns)
    }
}

/// Resolve one completed operation. Ambiguity or missing instrumentation is an
/// error, never a fabricated zero-duration sample.
pub fn span<'a>(intervals: &'a [Interval], label: &str) -> io::Result<&'a Interval> {
    let mut matching = intervals.iter().filter(|span| span.label() == label);
    let span = matching.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("missing span {label}"))
    })?;
    if matching.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("multiple spans for {label}; select a trial or aggregate intervals"),
        ));
    }
    Ok(span)
}

pub fn duration(intervals: &[Interval], label: &str) -> io::Result<Duration> {
    span(intervals, label).map(Interval::duration)
}

/// Inclusive operation totals inside a completed phase, excluding the reporting
/// envelope itself. Recordings must isolate one trial; cross-track work is
/// included by its real endpoints and repeated labels are unioned.
pub fn phase_totals(intervals: &[Interval], phase: &str) -> io::Result<Vec<(String, f64)>> {
    let parent = span(intervals, phase)?;
    Ok(totals(intervals.iter().filter(|s| {
        s.id != parent.id && s.start_ns >= parent.start_ns && s.end_ns <= parent.end_ns
    })))
}

/// A small executable-boundary convenience for one operation. The ordinary
/// tracing span defines its boundaries; Perfetto supplies its elapsed time.
/// Setup, flushing and querying are outside the operation's span.
pub fn measure<T>(span: tracing::Span, operation: impl FnOnce() -> T) -> io::Result<(T, Duration)> {
    let name = span
        .metadata()
        .ok_or_else(|| {
            io::Error::other("measurement span is disabled; install the Perfetto layer first")
        })?
        .name();
    let recording = Recording::start(Vec::new())?;
    let value = span.in_scope(operation);
    let intervals = recording.intervals()?;
    // Use the supplied span's static name, not an optional component override.
    let mut matching = intervals.iter().filter(|s| s.name == name);
    let root = matching
        .next()
        .ok_or_else(|| io::Error::other(format!("missing measurement {name}")))?;
    if matching.next().is_some() {
        return Err(io::Error::other(format!("ambiguous measurement {name}")));
    }
    Ok((value, root.duration()))
}

/// Per-label inclusive wall seconds in first-observed order. Union repeated or
/// parallel occurrences of the *same* operation; never sum nested labels into
/// a total. Callers select the trial/phase before aggregation.
pub fn totals<'a>(intervals: impl IntoIterator<Item = &'a Interval>) -> Vec<(String, f64)> {
    let mut groups: Vec<(String, Vec<(u64, u64)>)> = Vec::new();
    let mut indices = HashMap::new();
    for span in intervals {
        let index = *indices.entry(span.label()).or_insert_with(|| {
            groups.push((span.label().to_owned(), Vec::new()));
            groups.len() - 1
        });
        groups[index].1.push((span.start_ns, span.end_ns));
    }
    groups
        .into_iter()
        .map(|(label, ranges)| (label, union_ns(ranges) as f64 / 1e9))
        .collect()
}

fn union_ns(ranges: impl IntoIterator<Item = (u64, u64)>) -> u64 {
    let mut ranges: Vec<_> = ranges.into_iter().collect();
    ranges.sort_unstable();
    let mut end = 0;
    let mut ns = 0;
    for (start, next_end) in ranges {
        ns += next_end.saturating_sub(end.max(start));
        end = end.max(next_end);
    }
    ns
}

/// Render completed intervals, never measure them. Inclusive and exclusive
/// wall durations use interval unions; labels and repeated occurrences remain
/// in first-observed order. Optional RSS growth is a separate observation.
pub fn write_profile(
    mut writer: impl Write,
    header: &str,
    intervals: &[Interval],
    memory: Option<&std::collections::BTreeMap<String, memory::PeakGrowth>>,
) -> io::Result<()> {
    let denominator = union_ns(intervals.iter().map(|s| (s.start_ns, s.end_ns)));
    writeln!(writer, "┌─ prove profile: {header}")?;
    for (label, seconds) in totals(intervals) {
        let occurrences: Vec<_> = intervals.iter().filter(|s| s.label() == label).collect();
        let inclusive = union_ns(occurrences.iter().map(|s| (s.start_ns, s.end_ns)));
        // Subtract children per occurrence BEFORE unioning the same label.
        // A child's time can overlap another occurrence's own work (including
        // recursive occurrences of this label); subtracting group unions loses it.
        let mut self_ranges = Vec::new();
        for occurrence in &occurrences {
            let mut children: Vec<_> = intervals
                .iter()
                .filter(|s| s.parent == Some(occurrence.id))
                .map(|s| (s.start_ns, s.end_ns))
                .collect();
            children.sort_unstable();
            let mut cursor = occurrence.start_ns;
            for (start, end) in children {
                if cursor < start {
                    self_ranges.push((cursor, start));
                }
                cursor = cursor.max(end);
            }
            if cursor < occurrence.end_ns {
                self_ranges.push((cursor, occurrence.end_ns));
            }
        }
        let exclusive = union_ns(self_ranges);
        let share = if denominator == 0 {
            0.0
        } else {
            inclusive as f64 * 100.0 / denominator as f64
        };
        let indent = "  ".repeat(occurrences[0].depth);
        write!(
            writer,
            "│ {indent}{label}: {:.3?} ({share:.1}%)",
            Duration::from_secs_f64(seconds)
        )?;
        if occurrences.len() > 1 {
            write!(writer, " n={}", occurrences.len())?;
        }
        if exclusive < inclusive {
            write!(writer, " self={:.3?}", Duration::from_nanos(exclusive))?;
        }
        if let Some(row) = memory.and_then(|rows| rows.get(&label)) {
            write!(
                writer,
                " Δrss={:.1} MiB peak={:.1} MiB",
                row.bytes as f64 / 1048576.0,
                row.peak_bytes as f64 / 1048576.0
            )?;
        }
        writeln!(writer)?;
    }
    writeln!(writer, "└─")?;
    writer.flush()
}

/// A local native executable, never a downloader or an external tracing service.
pub struct TraceProcessor {
    executable: OsString,
}

impl TraceProcessor {
    pub fn new(executable: impl Into<OsString>) -> Self {
        Self {
            executable: executable.into(),
        }
    }

    pub fn from_env() -> Self {
        Self::new(
            std::env::var_os("PERFETTO_TRACE_PROCESSOR")
                .unwrap_or_else(|| "trace_processor_shell".into()),
        )
    }

    pub fn intervals(&self, trace: &[u8]) -> io::Result<Vec<Interval>> {
        if !cfg!(unix) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "in-memory Perfetto queries currently require Unix /dev/stdin",
            ));
        }
        let mut child = Command::new(&self.executable)
            .args(["/dev/stdin", "-Q", INTERVAL_QUERY])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().map_err(|error| io::Error::new(error.kind(), format!(
                "cannot start native Perfetto processor {:?}: {error}; set PERFETTO_TRACE_PROCESSOR",
                self.executable)))?;
        let mut input = child.stdin.take().expect("piped processor stdin");
        // Drain stdout/stderr while supplying the trace: either pipe can exceed
        // the OS buffer for large traces, including parser diagnostics.
        std::thread::scope(|scope| {
            let write = scope.spawn(move || input.write_all(trace));
            let output = child.wait_with_output();
            let written = write
                .join()
                .map_err(|_| io::Error::other("Perfetto input thread panicked"))?;
            let output = output?;
            if !output.status.success() {
                return Err(io::Error::other(format!(
                    "Perfetto query failed ({}): {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
            written?;
            decode_intervals(&output.stdout)
        })
    }
}

// One result set. JSON is the query's typed wire format inside CSV, preserving
// nulls and arbitrary span names without depending on the CLI's NULL sentinel.
const INTERVAL_QUERY: &str = r#"
WITH records AS (
SELECT json_object('record', 'status',
  'incomplete', (SELECT count(*) FROM slice WHERE dur < 0),
  'errors', (SELECT count(*) FROM stats WHERE value != 0 AND severity IN ('error', 'data_loss'))
) AS record
UNION ALL
SELECT json_object('record', 'span', 'id', id, 'parent', parent_id,
  'track_id', track_id, 'depth', depth, 'name', name,
  'component', EXTRACT_ARG(arg_set_id, 'debug.component'),
  'start_ns', ts - (SELECT min(ts) FROM slice),
  'end_ns', ts + dur - (SELECT min(ts) FROM slice))
FROM slice
)
SELECT replace(record, '"', '""') AS record FROM records"#;

#[derive(serde::Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum QueryRecord {
    Status { incomplete: u64, errors: u64 },
    Span(Interval),
}

fn decode_intervals(bytes: &[u8]) -> io::Result<Vec<Interval>> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    let mut reader = csv::Reader::from_reader(bytes);
    if reader.headers().map_err(io::Error::other)? != &csv::StringRecord::from(vec!["record"]) {
        return Err(invalid("unexpected Perfetto query columns"));
    }
    let mut status = false;
    let mut spans = Vec::new();
    for row in reader.records() {
        let row = row.map_err(io::Error::other)?;
        match serde_json::from_str::<QueryRecord>(&row[0]).map_err(io::Error::other)? {
            QueryRecord::Status { incomplete, errors } if !status => {
                if incomplete != 0 || errors != 0 {
                    return Err(invalid(
                        "incomplete or lossy Perfetto capture; refusing timing metrics",
                    ));
                }
                status = true;
            }
            QueryRecord::Span(span) if status => spans.push(span),
            _ => return Err(invalid("missing or duplicate Perfetto capture status")),
        }
    }
    if !status || spans.is_empty() {
        return Err(invalid(
            "empty Perfetto capture; is its tracing layer installed?",
        ));
    }
    let by_id: HashMap<_, _> = spans.iter().map(|span| (span.id, span)).collect();
    if by_id.len() != spans.len() {
        return Err(invalid("duplicate Perfetto slice ID"));
    }
    for span in &spans {
        if span.end_ns < span.start_ns {
            return Err(invalid("reversed Perfetto interval"));
        }
        if let Some(id) = span.parent {
            let parent = by_id
                .get(&id)
                .ok_or_else(|| invalid("missing Perfetto parent"))?;
            if parent.id >= span.id
                || parent.track_id != span.track_id
                || parent.start_ns > span.start_ns
                || span.end_ns > parent.end_ns
            {
                return Err(invalid("invalid Perfetto parent interval"));
            }
        }
    }
    spans.sort_by_key(|span| (span.start_ns, span.end_ns, span.id));
    Ok(spans)
}

fn trace_config() -> Vec<u8> {
    let writer = PbMsgWriter::new();
    let buffer = HeapBuffer::new(writer.stream_writer());
    let mut message = PbMsg::new(&writer).expect("allocate Perfetto configuration");
    {
        let mut config = TraceConfig { msg: &mut message };
        config.set_buffers(|buffer: &mut TraceConfigBufferConfig| {
            buffer.set_size_kb(64 * 1024);
            buffer.set_fill_policy(BufferConfigFillPolicy::Discard);
        });
        config.set_data_sources(|source: &mut TraceConfigDataSource| {
            source.set_config(|source: &mut DataSourceConfig| {
                source.set_name("track_event");
                source.set_track_event_config(|events: &mut TrackEventConfig| {
                    events.set_disabled_categories("*");
                    events.set_enabled_categories("tracing");
                });
            });
        });
    }
    message.finalize();
    let mut bytes = vec![0; writer.stream_writer().get_written_size()];
    buffer.copy_into(&mut bytes);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn decode(rows: &[Value]) -> io::Result<Vec<Interval>> {
        let mut csv = csv::Writer::from_writer(Vec::new());
        csv.write_record(["record"]).unwrap();
        for row in rows {
            csv.write_record([row.to_string()]).unwrap();
        }
        decode_intervals(&csv.into_inner().unwrap())
    }

    fn fixture() -> Vec<Value> {
        vec![
            json!({"record":"status", "incomplete":0, "errors":0}),
            json!({"record":"span", "id":0, "parent":null, "track_id":10, "depth":0,
                "name":"comma, quote\"\nnewline", "component":null,
                "start_ns":0, "end_ns":9007199254740993_u64}),
            json!({"record":"span", "id":1, "parent":0, "track_id":10, "depth":1,
                "name":"child", "component":"test.child", "start_ns":1, "end_ns":2}),
        ]
    }

    #[test]
    fn query_preserves_exact_nanoseconds_nulls_and_escaped_names() {
        let spans = decode(&fixture()).unwrap();
        assert_eq!(spans[0].end_ns, 9007199254740993);
        assert_eq!(spans[0].name, "comma, quote\"\nnewline");
        assert_eq!(spans[0].component, None);
        assert_eq!(spans[1].parent, Some(0));
        assert_eq!(spans[1].component.as_deref(), Some("test.child"));
    }

    #[test]
    fn query_rejects_incomplete_lossy_empty_or_malformed_output() {
        for field in ["incomplete", "errors"] {
            let mut rows = fixture();
            rows[0][field] = json!(1);
            assert!(decode(&rows).is_err());
        }
        assert!(decode(&fixture()[..1]).is_err());
        assert!(decode(&fixture()[1..]).is_err());
        assert!(decode_intervals(b"wrong\nheader\n").is_err());
        assert!(decode_intervals(b"record\nnot-json\n").is_err());
        for (field, value) in [
            ("end_ns", json!(0)),
            ("parent", json!(3)),
            ("parent", json!(1)),
            ("id", json!(0)),
            ("track_id", json!(11)),
        ] {
            let mut rows = fixture();
            rows[2][field] = value;
            assert!(decode(&rows).is_err(), "accepted invalid {field}");
        }
    }

    #[test]
    fn missing_processor_is_an_error_not_a_clock_fallback() {
        let processor = TraceProcessor::new("/nonexistent-bitz-perfetto-test/trace_processor_shell");
        assert_eq!(
            processor.intervals(&[]).unwrap_err().kind(),
            if cfg!(unix) {
                io::ErrorKind::NotFound
            } else {
                io::ErrorKind::Unsupported
            }
        );
    }
}
