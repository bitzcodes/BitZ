//! Run each workload/backend in a separate process to isolate peak RSS.
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use super::AnyError;
use super::output::{BenchmarkOutput, FileMode, JsonStyle};
use super::report::{HybridRow, NativeRow, SummaryRow};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    mul_log: u32,
    sha_log: u32,
}

fn child_identity(log: &str, mode: &str) -> Result<Option<serde_json::Value>, AnyError> {
    let reports: Vec<_> = log
        .lines()
        .filter_map(|line| line.strip_prefix("LIGERITO_CONFIG "))
        .collect();
    if mode == "all-binius" {
        if !reports.is_empty() {
            return Err("all-Binius output unexpectedly carries Ligerito configuration".into());
        }
        return Ok(None);
    }
    if reports.len() != 1 {
        return Err("missing or duplicated child Ligerito identity".into());
    }
    let report: serde_json::Value = serde_json::from_str(reports[0])?;
    match mode {
        "binius-ligerito" => {
            serde_json::from_value::<super::report::BiniusLigeritoIdentity>(report.clone())?
                .validate()?;
        }
        "hybrid" | "separate" => {
            bitz::ligerito_flock::ResolvedLigerito::validate_report(&report)?;
            let rate = report["configuration"]["levels"][0]["log_inv_rate"].as_u64();
            let target = report["target_bits"].as_u64().unwrap_or(0);
            let valid = if mode == "separate" {
                target == 112
            } else {
                match rate {
                    Some(1) => target == 106,
                    Some(3) => (100..=112).contains(&target),
                    _ => false,
                }
            };
            if !valid {
                return Err("incorrect Ligerito component budget".into());
            }
        }
        _ => return Err("unknown hybrid benchmark mode".into()),
    }
    Ok(Some(report))
}

pub fn equal_witness_shapes() -> Vec<Shape> {
    (15..=20)
        .map(|mul_log| Shape {
            mul_log,
            sha_log: mul_log - 8,
        })
        .collect()
}

pub fn parse_shapes(value: &str) -> Result<Vec<Shape>, AnyError> {
    let mut shapes = Vec::new();
    for pair in value.split(',') {
        let (mul, sha) = pair
            .trim()
            .split_once(':')
            .ok_or("--shapes expects MUL_LOG:SHA_LOG pairs, e.g. 15:7,16:8")?;
        let shape = Shape {
            mul_log: mul.trim().parse()?,
            sha_log: sha.trim().parse()?,
        };
        if !(9..=22).contains(&shape.mul_log) || !(1..=16).contains(&shape.sha_log) {
            return Err("--shapes requires multiplication logs 9..22 and SHA logs 1..16".into());
        }
        if shapes.contains(&shape) {
            return Err(format!("duplicate --shapes pair {pair}").into());
        }
        shapes.push(shape);
    }
    Ok(shapes)
}

fn grouped_count(value: usize) -> String {
    let digits = value.to_string();
    let mut result = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            result.push(',');
        }
        result.push(digit);
    }
    result
}

/// Produce a combined CSV record alongside a labelled terminal sample.
fn formatted_rows(
    csv: &str,
    mode: &str,
    shape: Shape,
    iterations: usize,
) -> Result<Vec<(SummaryRow, String)>, AnyError> {
    let mut reader = csv::Reader::from_reader(csv.as_bytes());
    let expected: Vec<_> = if mode == "hybrid" {
        HybridRow::HEADER.to_vec()
    } else {
        NativeRow::header(mode).to_vec()
    };
    if reader.headers()?.iter().ne(expected.iter().copied()) {
        return Err(format!("unexpected {mode} child CSV header").into());
    }
    let mut rows = Vec::new();
    // Reader rejects records with a different width from the header.
    for (iteration, record) in reader.deserialize::<SummaryRow>().enumerate() {
        let mut row = record?;
        if row.mode != mode || row.iteration.parse::<usize>()? != iteration {
            return Err(format!("invalid {mode} child CSV row {iteration}").into());
        }
        row.multiplication_relation = "u32_mod_2_32".into();
        row.mul_log = shape.mul_log;
        row.sha_log = shape.sha_log;
        row.multiplications = 1usize << shape.mul_log;
        row.sha_compressions = 1usize << shape.sha_log;
        let source = format!("{mode} [mul 2^{}, SHA 2^{}]", shape.mul_log, shape.sha_log);
        let mut display = String::new();
        if iteration == 0 {
            display.push_str(&format!(
                "{source}: setup {} ms (once, excluded from prover time)\n",
                row.setup_ms
            ));
        }
        display.push_str(&format!(
            "{source}: sample {}/{} — prover {} ms, verifier {} ms — VERIFIED\n",
            iteration + 1,
            iterations,
            row.total_prover_ms,
            row.verify_ms,
        ));
        if mode == "hybrid" {
            display.push_str(&format!(
                "  Witness + commitments: {} ms; PIOP + IOP continuation: {} ms\n",
                row.witness_commit_ms, row.continuation_ms,
            ));
            display.push_str(&format!(
                "  PIOP: {} ms (multiplication / Spartan: {} ms; SHA: {} ms)\n  IOP / PCS opening: {} ms (multiplication BitZ/GKR: {} ms; joint sumcheck: {} ms; shared opening: {} ms, of which Round 0: {} ms)\n",
                row.piop_ms,
                row.mul_piop_ms,
                row.sha_piop_ms,
                row.iop_ms,
                row.mul_opening_ms,
                row.joint_sumcheck_ms,
                row.shared_opening_ms,
                row.ood_round_ms,
            ));
        }
        let (size, size_label) = if mode == "separate" {
            (&row.proof_payload_bytes_estimate, "Proof payload estimate")
        } else {
            (&row.proof_bytes, "Proof")
        };
        let bytes: usize = size.parse()?;
        let peak_kib: u64 = row.peak_rss_kib.parse()?;
        let peak = if peak_kib == 0 {
            "unavailable".to_string()
        } else {
            format!("{:.2} MiB", peak_kib as f64 / 1024.0)
        };
        display.push_str(&format!(
            "  {size_label}: {} B ({:.2} KiB); peak RSS: {peak}",
            grouped_count(bytes),
            bytes as f64 / 1024.0,
        ));
        rows.push((row, display));
    }
    if rows.len() != iterations {
        return Err(format!("expected {iterations} verified rows, got {}", rows.len()).into());
    }
    Ok(rows)
}

pub fn run(
    shapes: Vec<Shape>,
    mode: &str,
    iterations: usize,
    results_dir: Option<PathBuf>,
    profile: Option<&str>,
) -> Result<(), AnyError> {
    let modes = if mode == "all" {
        vec!["hybrid", "separate", "all-binius", "binius-ligerito"]
    } else {
        vec![mode]
    };
    let executable = std::env::current_exe()?;
    let results_dir = if let Some(directory) = results_dir {
        directory
    } else {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("benches/results/hybrid-u32-sha256")
            .join(format!("sweep-{stamp}-{}", std::process::id()))
    };
    if let Some(parent) = results_dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        BenchmarkOutput::new(parent).create_dir_all()?;
    }
    BenchmarkOutput::new(&results_dir)
        .create_new_dir()
        .map_err(|error| {
            format!(
                "cannot create new results directory {}: {error}",
                results_dir.display()
            )
        })?;
    let results_dir = results_dir.canonicalize()?;
    let binius = super::cli::environment::<super::BiniusConfig>();
    let output = BenchmarkOutput::new(&results_dir);
    output.write_text(
        "run.txt",
        &format!(
            "executable={}\nprotocol=hybrid-u32-mod32-sha256-v5\nmultiplication_relation=xy=z+2^32*w (x,y,z,w are u32)\nshapes={shapes:?}\nmodes={modes:?}\niterations={iterations}\nRAYON_NUM_THREADS={}\nnon_zk=true\nsecurity_target_bits=100\nprofile={}\nBITZ_HYBRID_BINIUS_LOG_INV_RATE={}\nBITZ_HYBRID_BINIUS_SECURITY_BITS={}\nBITZ_BINIUS_LOG_INV_RATE={}\nBITZ_BINIUS_LIGERITO_ACCOUNTING={}\n",
            executable.display(),
            std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
            profile
                .map(str::to_owned)
                .or_else(|| std::env::var("BITZ_LIG_PROFILE").ok())
                .unwrap_or_else(|| "custom:1:4".into()),
            binius.log_inv_rate,
            binius.security_bits,
            super::binius_ligerito_log_inv_rate(),
            super::binius_ligerito_accounting().name(),
        ), FileMode::Replace,
    )?;
    let mut summary = output.csv("summary.csv", FileMode::Replace)?;
    summary.write_record(SummaryRow::HEADER)?;
    summary.flush()?;
    println!("SHA-256 chain + multiplication modulo 2^32 | non-ZK | 100-bit security target");
    println!(
        "Prover time includes witness generation, commitments and proof encoding; setup is separate."
    );
    println!("Peak RSS includes setup and is cumulative within each workload process.");
    println!("CSV results and detailed logs: {}", results_dir.display());
    let total = shapes.len() * modes.len();
    let mut completed = 0;
    for shape in shapes {
        for mode in &modes {
            let stem = format!("{mode}-m{}-s{}", shape.mul_log, shape.sha_log);
            let csv_path = results_dir.join(format!("{stem}.csv"));
            let log_path = results_dir.join(format!("{stem}.log"));
            let backend = match *mode {
                "hybrid" => "BitZ multiplication + Binius SHA, shared opening",
                "separate" => "BitZ multiplication + Binius SHA, separate proofs",
                "binius-ligerito" => "Binius multiplication + SHA, BitZ opener",
                _ => "Binius multiplication + SHA",
            };
            println!(
                "\n[{}/{total}] {mode} — {backend}\n  {} modular multiplications + {} chained SHA-256 compressions ({iterations} samples)",
                completed + 1,
                grouped_count(1usize << shape.mul_log),
                grouped_count(1usize << shape.sha_log),
            );
            std::io::stdout().flush()?;
            let status = Command::new(&executable)
                .args(profile.map(|p| vec!["--profile", p]).unwrap_or_default())
                .args([
                    "--mode",
                    mode,
                    "--mul-log",
                    &shape.mul_log.to_string(),
                    "--sha-log",
                    &shape.sha_log.to_string(),
                    "--iterations",
                    &iterations.to_string(),
                ])
                .stdout(output.file(&csv_path, FileMode::Replace)?)
                .stderr(output.file(&log_path, FileMode::Replace)?)
                .status()?;
            if !status.success() {
                return Err(format!(
                    "{stem} failed ({status}); see {}. Completed results are preserved in {}",
                    log_path.display(),
                    results_dir.display(),
                )
                .into());
            }
            let log = fs::read_to_string(&log_path)?;
            let identity = if let Some(report) = child_identity(&log, mode)? {
                output.write_json(
                    format!("{stem}.ligerito.json"),
                    &report,
                    FileMode::Replace,
                    JsonStyle::Pretty,
                )?;
                bitz::ligerito_flock::ResolvedLigerito::encode_report(&report)
            } else {
                String::new()
            };
            for (mut row, display) in
                formatted_rows(&fs::read_to_string(&csv_path)?, mode, shape, iterations)?
            {
                row.ligerito_hex = identity.clone();
                summary.serialize(row)?;
                println!("{display}");
            }
            summary.flush()?;
            completed += 1;
        }
    }
    println!(
        "\nCompleted {completed} workloads; all {} samples verified. CSV summary: {}",
        completed * iterations,
        results_dir.join("summary.csv").display()
    );
    Ok(())
}

#[cfg(test)]
mod reporting_tests {
    use super::*;

    #[test]
    fn binius_identity_roundtrips_and_rejects_wrong_modes_and_tampering() {
        // Both witness and relation oracles must satisfy BinaryPcs's log-13 floor.
        let native = super::super::Native::new(4096, 2, true, None).unwrap();
        let super::super::NativeBackend::Ligerito(prepared) = &native.backend else {
            unreachable!();
        };
        let identity = super::super::report::BiniusLigeritoIdentity::new(prepared).unwrap();
        let value = serde_json::to_value(&identity).unwrap();
        let log = format!("LIGERITO_CONFIG {value}\n");
        assert_eq!(
            child_identity(&log, "binius-ligerito").unwrap(),
            Some(value.clone())
        );
        assert_eq!(value["target_bits"], 100);
        assert_eq!(
            value["oracles"].as_array().unwrap().len(),
            prepared.oracle_specs().len()
        );
        for mode in ["hybrid", "separate", "all-binius", "unknown"] {
            assert!(child_identity(&log, mode).is_err(), "{mode}");
        }
        for bad in [
            "".to_string(),
            format!("{log}{log}"),
            "LIGERITO_CONFIG {}\n".into(),
        ] {
            assert!(child_identity(&bad, "binius-ligerito").is_err());
        }
        for (key, replacement) in [
            ("schema", serde_json::json!("old")),
            ("target_bits", serde_json::json!(112)),
            ("component_bits", serde_json::json!(99)),
            ("oracles", serde_json::json!([])),
            ("extra", serde_json::json!(true)),
        ] {
            let mut bad = value.clone();
            bad[key] = replacement;
            assert!(child_identity(&format!("LIGERITO_CONFIG {bad}"), "binius-ligerito").is_err());
        }
        for pointer in [
            "/oracles/0/configuration/levels/0/queries",
            "/oracles/0/ood_grinding_bits",
        ] {
            let mut bad = value.clone();
            *bad.pointer_mut(pointer).unwrap() = serde_json::json!(999);
            assert!(child_identity(&format!("LIGERITO_CONFIG {bad}"), "binius-ligerito").is_err());
        }
    }

    #[test]
    fn binius_identity_accepts_selected_rates_and_accounting() {
        use bitz::binius_ligerito::{Accounting, Prepared};
        let native = super::super::Native::new(4096, 2, true, None).unwrap();
        for rate in 1..=3 {
            for accounting in [Accounting::UnionBound, Accounting::RoundByRound] {
                let prepared = Prepared::with_options(native.circuit.constraint_system(), rate, accounting).unwrap();
                let identity = super::super::report::BiniusLigeritoIdentity::new(&prepared).unwrap();
                let value = serde_json::to_value(identity).unwrap();
                let log = format!("LIGERITO_CONFIG {value}");
                assert_eq!(child_identity(&log, "binius-ligerito").unwrap(), Some(value.clone()));
                assert_eq!(value["log_inv_rate"], rate);
                assert_eq!(value["accounting"], accounting.name());
                let mut bad = value.clone();
                bad["log_inv_rate"] = serde_json::json!(if rate == 1 { 3 } else { 1 });
                assert!(child_identity(&format!("LIGERITO_CONFIG {bad}"), "binius-ligerito").is_err());
            }
        }
    }

    #[test]
    fn single_opener_modes_keep_their_original_identity_and_budget_validation() {
        use bitz::ligerito_flock::{LigeritoSelection, OodRoundParams};
        assert!(child_identity("", "all-binius").unwrap().is_none());
        for (mode, bits) in [("hybrid", 106), ("separate", 112)] {
            let resolved = LigeritoSelection::JOHNSON.resolve(22, bits).unwrap();
            let value = resolved.report("custom:1:4", Some(OodRoundParams { grinding_bits: 0 }));
            let log = format!("LIGERITO_CONFIG {value}");
            assert_eq!(child_identity(&log, mode).unwrap(), Some(value));
            let other = if mode == "hybrid" {
                "separate"
            } else {
                "hybrid"
            };
            assert!(child_identity(&log, other).is_err());
            assert!(child_identity(&log, "binius-ligerito").is_err());
        }
    }
    const NATIVE_HEADER: &str =
        "mode,iteration,setup_ms,witness_ms,total_prover_ms,verify_ms,proof_bytes,peak_rss_kib\n";
    const ROW: &str = "all-binius,0,1.230,2.000,3.000,4.000,1024,0\n";
    const SHAPE: Shape = Shape {
        mul_log: 9,
        sha_log: 1,
    };

    #[test]
    fn csv_contract_sweep_preserves_child_lexemes_and_missing_columns() {
        let source = format!("{NATIVE_HEADER}{ROW}");
        let rows = formatted_rows(&source, "all-binius", SHAPE, 1).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(
            rows[0]
                .1
                .contains("Proof: 1,024 B (1.00 KiB); peak RSS: unavailable")
        );
        let mut csv = super::super::output::csv_writer(Vec::new());
        csv.write_record(SummaryRow::HEADER).unwrap();
        csv.flush().unwrap();
        let header = "mode,multiplication_relation,mul_log,sha_log,multiplications,sha_compressions,iteration,setup_ms,witness_ms,witness_commit_ms,continuation_ms,total_prover_ms,verify_ms,proof_bytes,proof_payload_bytes_estimate,peak_rss_kib,piop_ms,iop_ms,mul_piop_ms,sha_piop_ms,mul_opening_ms,joint_sumcheck_ms,shared_opening_ms,ood_round_ms,ligerito_hex\n";
        assert_eq!(csv.get_ref(), header.as_bytes());
        csv.serialize(&rows[0].0).unwrap();
        assert_eq!(
            String::from_utf8(csv.into_inner().unwrap()).unwrap(),
            format!(
                "{header}all-binius,u32_mod_2_32,9,1,512,2,0,1.230,2.000,,,3.000,4.000,1024,,0,,,,,,,,,\n"
            )
        );
        assert!(
            formatted_rows(NATIVE_HEADER, "all-binius", SHAPE, 0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn csv_contract_reader_accepts_quoted_fields_and_embedded_newlines() {
        let source = format!(
            "{NATIVE_HEADER}all-binius,\"0\",\"1,2\n\"\"3\"\"\",2.000,3.000,4.000,1024,0\n"
        );
        let rows = formatted_rows(&source, "all-binius", SHAPE, 1).unwrap();
        assert_eq!(rows[0].0.setup_ms, "1,2\n\"3\"");
        let mut csv = super::super::output::csv_writer(Vec::new());
        csv.write_record(SummaryRow::HEADER).unwrap();
        csv.serialize(&rows[0].0).unwrap();
        let bytes = csv.into_inner().unwrap();
        let mut reader = csv::Reader::from_reader(bytes.as_slice());
        let roundtrip: SummaryRow = reader.deserialize().next().unwrap().unwrap();
        assert_eq!(roundtrip.setup_ms, rows[0].0.setup_ms);
    }

    #[test]
    fn csv_contract_reader_rejects_wrong_header_mode_iteration_width_and_count() {
        let valid = format!("{NATIVE_HEADER}{ROW}");
        for bad in [
            String::new(),
            valid.replacen("proof_bytes", "wrong_bytes", 1),
            valid.replacen("all-binius,0", "separate,0", 1),
            valid.replacen("all-binius,0", "all-binius,1", 1),
            valid.replacen("1024,0", "1024,0,extra", 1),
            valid.replacen("1024,0", "1024", 1),
            valid.replacen("1024", "not-an-integer", 1),
            format!("{valid}{ROW}"),
            NATIVE_HEADER.into(),
        ] {
            assert!(
                formatted_rows(&bad, "all-binius", SHAPE, 1).is_err(),
                "{bad}"
            );
        }
        assert!(formatted_rows(&valid, "all-binius", SHAPE, 2).is_err());
    }

    #[test]
    fn csv_contract_reader_maps_hybrid_and_separate_proof_sizes() {
        let hybrid = "mode,iteration,setup_ms,witness_ms,witness_commit_ms,continuation_ms,total_prover_ms,verify_ms,proof_bytes,peak_rss_kib,piop_ms,iop_ms,mul_piop_ms,sha_piop_ms,mul_opening_ms,joint_sumcheck_ms,shared_opening_ms,ood_round_ms\nhybrid,0,1.000,2.000,3.000,4.000,5.000,6.000,1024,0,7.000,8.000,9.000,10.000,11.000,12.000,13.000,14.000\n";
        let rows = formatted_rows(hybrid, "hybrid", SHAPE, 1).unwrap();
        assert_eq!(rows[0].0.ood_round_ms, "14.000");
        assert_eq!(rows[0].0.proof_bytes, "1024");
        assert!(rows[0].0.proof_payload_bytes_estimate.is_empty());
        let separate = format!("{NATIVE_HEADER}{ROW}")
            .replace("proof_bytes", "proof_payload_bytes_estimate")
            .replace("all-binius", "separate");
        let rows = formatted_rows(&separate, "separate", SHAPE, 1).unwrap();
        assert!(rows[0].0.proof_bytes.is_empty());
        assert_eq!(rows[0].0.proof_payload_bytes_estimate, "1024");
        assert!(rows[0].1.contains("Proof payload estimate:"));
    }
}
