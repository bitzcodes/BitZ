#[path = "../benches/common/output.rs"]
mod output;

use output::{BenchmarkOutput, FileMode::*, JsonStyle::*, JsonlWriter};
use serde_json::json;
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bitz-output-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn output(&self) -> BenchmarkOutput {
        BenchmarkOutput::new(&self.0)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn creation_replacement_append_and_directory_policies() {
    let temp = Temp::new();
    let out = temp.output();
    out.write_text("result", "first", CreateNew).unwrap();
    assert!(out.file("result", CreateNew).is_err());
    assert!(out.file("missing", AppendExisting).is_err());
    out.write_text("result", "+tail", AppendExisting).unwrap();
    assert_eq!(fs::read(temp.0.join("result")).unwrap(), b"first+tail");
    out.write_text("result", "new", Replace).unwrap();
    assert_eq!(fs::read(temp.0.join("result")).unwrap(), b"new");
    assert!(out.file("nested/result", CreateNew).is_err());
    out.create_parents("nested/result").unwrap();
    out.file("nested/result", CreateNew).unwrap();
    assert!(out.create_new_dir().is_err());
    out.create_dir_all().unwrap();
}

#[test]
fn json_styles_jsonl_and_binary_fidelity() {
    let temp = Temp::new();
    let out = temp.output();
    let value = json!({"bytes": 2048, "median": 2048.0, "optional": null});
    out.write_json("compact", &value, CreateNew, Compact)
        .unwrap();
    out.write_json("pretty", &value, CreateNew, Pretty).unwrap();
    assert_eq!(
        fs::read(temp.0.join("compact")).unwrap(),
        serde_json::to_vec(&value).unwrap()
    );
    assert_eq!(
        fs::read(temp.0.join("pretty")).unwrap(),
        serde_json::to_vec_pretty(&value).unwrap()
    );
    let mut lines = out.jsonl("lines", CreateNew).unwrap();
    lines.write(&value).unwrap();
    lines.write(&json!({"text": "line\nquote\""})).unwrap();
    lines.finish().unwrap();
    let data = fs::read_to_string(temp.0.join("lines")).unwrap();
    assert!(data.ends_with('\n'));
    assert_eq!(data.lines().count(), 2);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(data.lines().next().unwrap()).unwrap(),
        value
    );
    let bytes = [0, 255, 10, 128];
    out.write_bytes("staged", &bytes, Replace).unwrap();
    out.rename("staged", "proof.bin").unwrap();
    assert_eq!(fs::read(temp.0.join("proof.bin")).unwrap(), bytes);
}

struct Failing {
    fail_write: bool,
}
impl Write for Failing {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.fail_write {
            Err(io::Error::other("write failed"))
        } else {
            Ok(bytes.len())
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("flush failed"))
    }
}

#[test]
fn stream_errors_are_not_lost_on_drop() {
    assert!(
        JsonlWriter::new(Failing { fail_write: true })
            .write(&json!({}))
            .is_err()
    );
    let mut lines = JsonlWriter::new(Failing { fail_write: false });
    lines.write(&json!({})).unwrap();
    assert!(lines.finish().is_err());
    let mut csv = output::csv_writer(Failing { fail_write: true });
    csv.write_record(["header"]).unwrap();
    assert!(csv.flush().is_err());
    let mut csv = output::csv_writer(Failing { fail_write: false });
    csv.write_record(["header"]).unwrap();
    assert!(csv.flush().is_err());
}

#[test]
fn csv_headers_empty_tables_and_escaping() {
    let mut csv = output::csv_writer(Vec::new());
    csv.write_record(["name", "value"]).unwrap();
    csv.flush().unwrap();
    assert_eq!(csv.get_ref(), b"name,value\n");
    csv.serialize(("comma,quote\"\nline", 42)).unwrap();
    let bytes = csv.into_inner().unwrap();
    assert_eq!(bytes, b"name,value\n\"comma,quote\"\"\nline\",42\n");
    let temp = Temp::new();
    let mut csv = temp.output().csv("stream.csv", CreateNew).unwrap();
    csv.write_record(["name", "value"]).unwrap();
    csv.flush().unwrap();
    assert_eq!(
        fs::read(temp.0.join("stream.csv")).unwrap(),
        b"name,value\n"
    );
    csv.serialize(("sample", 1)).unwrap();
    csv.flush().unwrap();
    assert_eq!(
        fs::read(temp.0.join("stream.csv")).unwrap(),
        b"name,value\nsample,1\n"
    );
}

#[test]
fn stdout_sink() {
    if std::env::var_os("BITZ_OUTPUT_TEST_CHILD").is_some() {
        let mut stdout = JsonlWriter::new(std::io::stdout().lock());
        stdout.write(&json!({"stdout_test": true})).unwrap();
        stdout.finish().unwrap();
        return;
    }
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "stdout_sink", "--nocapture"])
        .env("BITZ_OUTPUT_TEST_CHILD", "1")
        .output()
        .unwrap();
    assert!(child.status.success());
    assert!(
        String::from_utf8(child.stdout)
            .unwrap()
            .lines()
            .any(|line| line == "{\"stdout_test\":true}")
    );
}
