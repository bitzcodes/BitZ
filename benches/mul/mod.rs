//! Case expansion and worker transport are shared; experiments own their timing boundaries.
pub mod config;
#[cfg(feature = "native-mul-compare")]
mod native;
mod outer;
mod pcs;
mod piop;
mod proof;
use anyhow::{Result, ensure};
use clap::Parser;
use config::{Args, Job, Memory, Mode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::Path,
    process::Command,
};

pub type Metrics = BTreeMap<String, f64>;

/// Observability totals are seconds; benchmark records consistently use milliseconds.
pub fn phase_milliseconds(
    intervals: &[bitz::observability::Interval],
    scope: &str,
) -> std::io::Result<Vec<(String, f64)>> {
    Ok(bitz::observability::phase_totals(intervals, scope)?
        .into_iter()
        .map(|(name, seconds)| (name, seconds * 1000.))
        .collect())
}

#[derive(Serialize, Deserialize)]
pub struct Sample {
    pub case_id: String,
    pub kind: String,
    pub index: usize,
    pub verified: bool,
    pub metrics: Metrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<crate::common::proof_fingerprint::ProofFingerprint>,
}
#[derive(Serialize, Deserialize)]
pub struct Run {
    pub job: Job,
    pub effective: Value,
    pub samples: Vec<Sample>,
    pub tuning: Option<Value>,
    pub heap_peak: Option<usize>,
}
impl Run {
    fn new(job: Job) -> Self {
        Self {
            job,
            effective: json!({}),
            samples: Vec::new(),
            tuning: None,
            heap_peak: None,
        }
    }
    pub fn trials(&self) -> usize {
        if self.job.memory == Memory::None {
            self.job.reps + self.job.warmups
        } else {
            1
        }
    }
    pub fn sample(&mut self, trial: usize, metrics: Metrics) {
        let warmup = trial < self.job.warmups;
        self.samples.push(Sample {
            case_id: self.job.id.clone(),
            kind: if warmup { "warmup" } else { "sample" }.into(),
            index: if warmup {
                trial
            } else {
                trial - self.job.warmups
            },
            verified: true,
            metrics,
            fingerprint: None,
        });
    }
    pub fn begin_memory(&self) {
        #[cfg(feature = "bench-peak-memory")]
        if self.job.memory == Memory::Heap {
            crate::common::peak_memory::reset_peak();
        }
    }
    pub fn end_memory(&mut self) {
        #[cfg(feature = "bench-peak-memory")]
        if self.job.memory == Memory::Heap {
            self.heap_peak = Some(crate::common::peak_memory::peak_bytes());
        }
    }
    pub fn latency(&self) -> bool {
        self.job.memory == Memory::None
    }
}
#[derive(Debug, thiserror::Error)]
#[error("unsupported configuration: {0}")]
pub struct Unsupported(pub String);
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn execute(run: &mut Run, compare: bool) -> Result<()> {
    match run.job.case.mode {
        Mode::Proof | Mode::Witness if compare && run.job.case.backend != "bitz" => {
            #[cfg(feature = "native-mul-compare")]
            {
                native::run(run)?;
            }
            #[cfg(not(feature = "native-mul-compare"))]
            anyhow::bail!("comparison backend requires native-mul-compare");
        }
        Mode::Proof | Mode::Witness | Mode::Bounds => proof::run(run, compare)?,
        Mode::Pcs => pcs::run(run)?,
        Mode::Piop => piop::run(run)?,
        Mode::Outer => outer::run(run)?,
    }
    Ok(())
}
fn worker(path: &Path, compare: bool) -> Result<()> {
    let job: Job = serde_json::from_reader(File::open(path)?)?;
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(job.case.threads)
        .build_global()?;
    crate::common::init();
    if job.memory == Memory::None {
        bitz::observability::install()?;
    }
    bitz::merged_forest::schedule::start_recording();
    let mut run = Run::new(job);
    if let Err(error) = execute(&mut run, compare) {
        if run.job.skip_unsupported && error.is::<Unsupported>() {
            run.job.skip = Some(error.to_string());
            return write_json(&path.with_extension("result.json"), &run);
        }
        return Err(error);
    }
    let schedules = bitz::merged_forest::schedule::take_records();
    if run
        .job
        .case
        .bitz
        .as_ref()
        .is_some_and(|f| f.gkr_schedule.is_some())
    {
        ensure!(!schedules.is_empty(), "missing resolved GKR schedules");
        run.effective["gkr_schedules"] = serde_json::to_value(schedules)?;
    }
    if !run.latency() {
        let bytes = match run.job.memory {
            Memory::Rss => peak_rss()? as f64,
            Memory::Heap => {
                #[cfg(feature = "bench-peak-memory")]
                {
                    run.heap_peak.expect("heap boundary measured") as f64
                }
                #[cfg(not(feature = "bench-peak-memory"))]
                anyhow::bail!("heap requires bench-peak-memory");
            }
            Memory::None => unreachable!(),
        };
        run.samples = vec![Sample {
            case_id: run.job.id.clone(),
            kind: if run.job.memory == Memory::Rss {
                "rss"
            } else {
                "heap"
            }
            .into(),
            index: 0,
            verified: true,
            fingerprint: None,
            metrics: Metrics::from([(
                if run.job.memory == Memory::Rss {
                    "peak_rss_bytes"
                } else {
                    "peak_heap_bytes"
                }
                .into(),
                bytes,
            )]),
        }];
    }
    write_json(&path.with_extension("result.json"), &run)
}
fn peak_rss() -> Result<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes this valid output pointer on success.
    ensure!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
        "getrusage failed"
    );
    let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    Ok(if cfg!(target_os = "macos") {
        rss
    } else {
        rss * 1024
    })
}
fn child(job: &Job, dir: &Path) -> Result<Run> {
    let name = format!("{}-{:?}", job.id, job.memory).to_lowercase();
    let request = dir.join(format!("{name}.json"));
    write_json(&request, job)?;
    let log = File::create(dir.join(format!("{name}.log")))?;
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("--worker")
        .arg(&request)
        .env("RAYON_NUM_THREADS", job.case.threads.to_string())
        .env("HARDWARE_CONCURRENCY", job.case.threads.to_string());
    // Only CLI settings control experiments, including dependency environment knobs.
    for (key, _) in std::env::vars_os() {
        let s = key.to_string_lossy();
        if s.starts_with("BITZ_") || s.starts_with("F2_") || s.starts_with("BD") {
            command.env_remove(key);
        }
    }
    if let Some(schedule) = job.case.bitz.as_ref().and_then(|f| f.gkr_schedule) {
        command.env("F2_FOREST_SCHEDULE", schedule.name());
    }
    if let Some(bits) = job.case.limber_bits {
        command.env("BDLAMBDA", bits.to_string());
    }
    let status = command.stdout(log.try_clone()?).stderr(log).status()?;
    ensure!(
        status.success(),
        "{} worker failed ({status}); see {}",
        job.id,
        dir.join(format!("{name}.log")).display()
    );
    let result = request.with_extension("result.json");
    let run: Run = serde_json::from_reader(File::open(&result)?)?;
    let mut expected = job.case.clone();
    if expected.whir.is_none() {
        expected.whir = run.job.case.whir;
    }
    ensure!(run.job.case == expected, "worker changed case identity");
    fs::remove_file(request)?;
    fs::remove_file(result)?;
    Ok(run)
}
fn provenance() -> Result<Value> {
    let mut value = crate::common::environment::metadata(0);
    for key in ["threads", "requested_threads"] {
        value.as_object_mut().expect("metadata object").remove(key);
    }
    let executable = fs::read(std::env::current_exe()?)?;
    value["executable_blake3"] = json!(blake3::hash(&executable).to_hex().to_string());
    value["compiled_features"] = json!({"parallel":cfg!(feature="parallel"),"span-metrics":cfg!(feature="span-metrics"),"bench-internals":cfg!(feature="bench-internals"),"native-mul-compare":cfg!(feature="native-mul-compare"),"bench-peak-memory":cfg!(feature="bench-peak-memory"),"unchecked":cfg!(feature="unchecked"),"bench-perfetto":cfg!(feature="bench-perfetto"),"plonky3-whir-degree4-bench":cfg!(feature="plonky3-whir-degree4-bench"),"plonky3-whir-goldilocks-degree2-bench":cfg!(feature="plonky3-whir-goldilocks-degree2-bench")});
    value["debug_assertions"] = json!(cfg!(debug_assertions));
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<String> = if root.join(".git").exists() {
        let paths = Command::new("git")
            .args([
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ])
            .current_dir(root)
            .output()?;
        ensure!(paths.status.success(), "cannot fingerprint source tree");
        std::str::from_utf8(&paths.stdout)?
            .split('\0')
            .filter(|p| !p.is_empty())
            .map(str::to_owned)
            .collect()
    } else {
        fs::read_to_string(root.join("release-source-files.txt"))?
            .lines()
            .map(str::to_owned)
            .collect()
    };
    files.sort();
    files.dedup();
    let mut hash = blake3::Hasher::new();
    for name in files {
        let relative = Path::new(&name);
        ensure!(
            !relative.is_absolute()
                && !relative
                    .components()
                    .any(|p| matches!(p, std::path::Component::ParentDir)),
            "invalid release source path"
        );
        let path = root.join(relative);
        if path.is_file()
            && (name.ends_with(".rs")
                || name.ends_with(".toml")
                || name.ends_with(".lock")
                || name.ends_with(".py")
                || name.ends_with(".sh"))
        {
            let bytes = fs::read(path)?;
            hash.update(&(name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(&(bytes.len() as u64).to_le_bytes());
            hash.update(&bytes);
        }
    }
    value["source_blake3"] = json!(hash.finalize().to_hex().to_string());
    value["heap_boundary"] = json!(
        "verified trial after preparation; includes live baseline allocations; excludes reporting"
    );
    value["rss_boundary"] = json!(
        "isolated worker lifetime including corpus, preparation, one verified trial and artifact codecs; no profiler recording"
    );
    Ok(value)
}
pub fn main(compare: bool) -> Result<()> {
    let args = Args::parse();
    if let Some(path) = &args.worker {
        return worker(path, compare);
    }
    let jobs = args.expand(compare)?;
    ensure!(
        !cfg!(feature = "bench-peak-memory") || args.memory == Memory::Heap,
        "instrumented binaries only accept --memory heap; use a separate latency build"
    );
    ensure!(
        cfg!(feature = "bench-peak-memory") || args.memory != Memory::Heap,
        "--memory heap requires bench-peak-memory"
    );
    if args.dry_run {
        println!("{}", serde_json::to_string_pretty(&jobs)?);
        return Ok(());
    }
    let out = args.out.unwrap_or_else(|| {
        std::path::PathBuf::from(format!(
            "results/mul-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ))
    });
    fs::create_dir_all(&out)?;
    let manifest_path = out.join("manifest.json");
    let mut manifest = json!({"schema":"mul-bench/v2","target":if compare {"mul_compare"}else{"mul_bitz"},"status":"running","provenance":provenance()?,"cases":[]});
    // Reserve artifacts before any measurement; never overwrite a prior campaign.
    write_json(&manifest_path, &manifest)?;
    let mut samples = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("samples.jsonl"))?;
    let workers = out.join("workers");
    fs::create_dir(&workers)?;
    for job in jobs {
        eprintln!("{}: {:?}", job.id, job.case);
        if let Some(reason) = &job.skip {
            manifest["cases"]
                .as_array_mut()
                .expect("case list")
                .push(json!({"job":job,"status":"skipped","reason":reason}));
        } else {
            let mut latency_job = job.clone();
            if job.memory == Memory::Rss {
                latency_job.memory = Memory::None;
            }
            let mut run = child(&latency_job, &workers)?;
            let mut resolved_job = job.clone();
            resolved_job.case = run.job.case.clone();
            if let Some(reason) = &run.job.skip {
                manifest["cases"]
                    .as_array_mut()
                    .expect("case list")
                    .push(json!({"job":resolved_job,"status":"skipped","reason":reason}));
                fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
                continue;
            }
            if job.memory == Memory::Rss {
                let memory = child(&resolved_job, &workers)?;
                ensure!(
                    memory.job.skip.is_none(),
                    "memory replay became unsupported"
                );
                ensure!(
                    run.effective == memory.effective,
                    "memory configuration differs from latency configuration for {}",
                    job.id
                );
                run.samples.extend(memory.samples);
            }
            for sample in &run.samples {
                serde_json::to_writer(&mut samples, sample)?;
                samples.write_all(b"\n")?;
            }
            samples.flush()?;
            manifest["cases"].as_array_mut().expect("case list").push(json!({"job":resolved_job,"status":"measured","effective":run.effective,"tuning":run.tuning}));
        }
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    }
    manifest["status"] = json!("complete");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    eprintln!("Results: {}", out.display());
    Ok(())
}
