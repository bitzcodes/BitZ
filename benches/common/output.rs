//! Benchmark artifact I/O, independent of any prover or benchmark schema.
#![allow(dead_code)] // Also included by isolated workers with fewer output formats.

use serde::Serialize;
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy)]
pub enum FileMode {
    CreateNew,
    Replace,
    AppendExisting,
}

#[derive(Clone, Copy)]
pub enum JsonStyle {
    Compact,
    Pretty,
}

pub struct BenchmarkOutput {
    root: PathBuf,
}

impl BenchmarkOutput {
    /// Resolves relative artifact paths without creating or clearing anything.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_owned(),
        }
    }

    pub fn create_dir_all(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    pub fn create_new_dir(&self) -> io::Result<()> {
        fs::create_dir(&self.root)
    }

    pub fn create_parents(&self, path: impl AsRef<Path>) -> io::Result<()> {
        if let Some(parent) = self
            .root
            .join(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    pub fn file(&self, path: impl AsRef<Path>, mode: FileMode) -> io::Result<File> {
        let mut options = OpenOptions::new();
        options.write(true);
        match mode {
            FileMode::CreateNew => {
                options.create_new(true);
            }
            FileMode::Replace => {
                options.create(true).truncate(true);
            }
            FileMode::AppendExisting => {
                options.append(true);
            }
        }
        options.open(self.root.join(path))
    }

    pub fn buffered(&self, path: impl AsRef<Path>, mode: FileMode) -> io::Result<BufWriter<File>> {
        self.file(path, mode).map(BufWriter::new)
    }

    pub fn jsonl(
        &self,
        path: impl AsRef<Path>,
        mode: FileMode,
    ) -> io::Result<JsonlWriter<BufWriter<File>>> {
        self.buffered(path, mode).map(JsonlWriter::new)
    }

    pub fn csv(&self, path: impl AsRef<Path>, mode: FileMode) -> io::Result<csv::Writer<File>> {
        self.file(path, mode).map(csv_writer)
    }

    pub fn write_json(
        &self,
        path: impl AsRef<Path>,
        value: &impl Serialize,
        mode: FileMode,
        style: JsonStyle,
    ) -> Result<()> {
        let mut writer = self.buffered(path, mode)?;
        match style {
            JsonStyle::Compact => serde_json::to_writer(&mut writer, value)?,
            JsonStyle::Pretty => serde_json::to_writer_pretty(&mut writer, value)?,
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_bytes(
        &self,
        path: impl AsRef<Path>,
        bytes: &[u8],
        mode: FileMode,
    ) -> io::Result<()> {
        let mut writer = self.file(path, mode)?;
        writer.write_all(bytes)?;
        writer.flush()
    }

    pub fn write_text(&self, path: impl AsRef<Path>, text: &str, mode: FileMode) -> io::Result<()> {
        self.write_bytes(path, text.as_bytes(), mode)
    }

    /// Preserve callers' existing staged-publication policy (for example SRS caches).
    pub fn rename(&self, from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
        fs::rename(self.root.join(from), self.root.join(to))
    }
}

/// Headers are explicit so even an empty report retains its schema.
pub fn csv_writer<W: Write>(writer: W) -> csv::Writer<W> {
    csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(writer)
}

/// Presentation only: JSON records keep their numeric fields unchanged.
pub mod csv_format {
    use serde::{Serializer, ser::Error};

    pub fn three_decimals<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("{value:.3}"))
    }

    pub fn nine_decimals<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("{value:.9}"))
    }

    pub fn display<T: std::fmt::Display, S: Serializer>(
        value: &T,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_str(value)
    }

    /// Native multiplication historically printed JSON-number spellings in CSV.
    pub fn json_number<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serde_json::to_string(value).map_err(S::Error::custom)?)
    }
}

pub struct JsonlWriter<W: Write> {
    writer: W,
}

impl<W: Write> JsonlWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn write(&mut self, record: &impl Serialize) -> Result<()> {
        serde_json::to_writer(&mut self.writer, record)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn finish(mut self) -> io::Result<W> {
        self.flush()?;
        Ok(self.writer)
    }
}
