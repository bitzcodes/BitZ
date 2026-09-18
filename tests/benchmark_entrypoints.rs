//! Smoke the actual harness=false executables without running benchmark campaigns.
use std::{
    collections::BTreeMap,
    path::Path,
    process::{Command, Output, Stdio},
};

fn run(executable: &Path, directory: &Path, args: &[&str], env: &[(&str, &str)]) -> Output {
    Command::new(executable)
        .args(args)
        .current_dir(directory)
        .env_clear()
        .envs(env.iter().copied())
        .output()
        .unwrap()
}

#[test]
#[ignore = "builds all benchmark executables; run with cargo test --test benchmark_entrypoints -- --ignored"]
fn every_entrypoint_handles_help_and_errors_before_work() {
    let build = Command::new(env!("CARGO")).current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["build", "--benches", "--bin", "hybrid-u32-sha256", "--profile", "dev",
            "--features", "span-metrics,bench-internals,native-mul-compare,native-sha256-compare,sha256-ecdsa-compare,hybrid,bench-peak-memory",
            "--message-format=json", "--offline"])
        .stderr(Stdio::inherit()).output().unwrap();
    assert!(build.status.success(), "benchmark build failed");
    let executables: BTreeMap<String, std::path::PathBuf> = String::from_utf8(build.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| v["reason"] == "compiler-artifact")
        .filter(|v| {
            v["target"]["kind"][0] == "bench"
                || (v["target"]["name"] == "hybrid-u32-sha256" && v["profile"]["test"] == false)
        })
        .filter_map(|v| {
            Some((
                v["target"]["name"].as_str()?.to_owned(),
                v["executable"].as_str()?.into(),
            ))
        })
        .collect();
    assert!(executables.contains_key("mul_bitz") && executables.contains_key("mul_compare"));
    assert!(!executables.contains_key("mul_e2e_compare"));
    let directory = tempfile::tempdir().unwrap();
    for (name, executable) in &executables {
        let args = match name.as_str() {
            "sha256_ecdsa" => vec!["3", "split", "100", "1", "--bench", "--help"],
            "sha256_ecdsa_compare" => vec![
                "--r",
                "1",
                "--c",
                "2",
                "--export-fixture",
                "fixture.json",
                "--bench",
                "--help",
            ],
            "ligerito_bounds" => vec!["pcs-22", "custom:1:4", "--memory", "--bench", "--help"],
            _ => vec!["--bench", "--help"],
        };
        // Invalid environment values must not stop help or initialize the profiler.
        let out = run(
            executable,
            directory.path(),
            &args,
            &[
                ("BITZ_BENCH_REPS", "invalid"),
                ("BITZ_SHA_COMPARE_REPS", "invalid"),
                ("BITZ_HYBRID_BINIUS_LOG_INV_RATE", "invalid"),
                ("PERFETTO_TRACE_PROCESSOR", "/missing"),
            ],
        );
        assert!(
            out.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let help = String::from_utf8(out.stdout).unwrap();
        assert!(
            help.contains("Usage:") && !help.contains("--bench"),
            "{name}: {help}"
        );
        if !matches!(
            name.as_str(),
            "sha256_ecdsa"
                | "sha256_ecdsa_compare"
                | "ligerito_bounds"
                | "hybrid_u32_sha256"
                | "hybrid-u32-sha256"
                | "mul_bitz"
                | "mul_compare"
        ) {
            assert!(
                !help.contains("--reps") && !help.contains("--shapes"),
                "{name}: {help}"
            );
        }
        let unknown = run(executable, directory.path(), &["--unknown-option"], &[]);
        assert_eq!(unknown.status.code(), Some(2), "{name}");
        assert_eq!(
            std::fs::read_dir(directory.path()).unwrap().count(),
            0,
            "{name} created output before parsing"
        );
    }
    for (name, variable) in [
        ("pcs", "BITZ_BENCH_REPS"),
        ("field", "BITZ_BENCH_REPS"),
        ("eq_tables", "BITZ_EQ_TABLE_SAMPLES"),
        ("cm_and", "BITZ_BENCH_REPS"),
        ("multiswap", "BITZ_BENCH_REPS"),
        ("lambda_sweep", "BITZ_BENCH_REPS"),
        ("sha256_compressions", "BITZ_BENCH_REPS"),
        ("sha256_chain", "BITZ_BENCH_REPS"),
        ("sha256_product_layout", "BITZ_BENCH_REPS"),
        ("sha256_e2e_compare", "BITZ_SHA_COMPARE_REPS"),
    ] {
        let out = run(
            &executables[name],
            directory.path(),
            &["--bench"],
            &[(variable, "invalid")],
        );
        assert_eq!(
            out.status.code(),
            Some(2),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(variable),
            "{name}"
        );
    }
    for (name, variable, value) in [
        ("cm_and", "BITZ_CM_EXPONENTS", "14"),
        ("sha256_chain", "BITZ_BENCH_SHAPES", "6"),
        ("lambda_sweep", "BITZ_BENCH_SHAPES", "17"),
        ("pcs", "BITZ_BENCH_FILL", "0"),
        ("pcs", "BITZ_BENCH_FILL", "1.1"),
        ("pcs", "BITZ_BENCH_FILL", "NaN"),
        ("pcs", "BITZ_BENCH_FILL", "inf"),
        ("sha256_compressions", "BITZ_SHA_INNER_PREFIX_VARS", "5"),
        ("sha256_compressions", "BITZ_SHA_PRODUCT_TS", "29"),
        ("sha256_compressions", "BITZ_SHA_MNUMROWS_LOG2S", "17"),
        ("sha256_compressions", "BITZ_SHA_LOG2S", "3"),
        ("sha256_e2e_compare", "BITZ_SHA_COMPARE_EXPONENTS", "6"),
        ("sha256_e2e_compare", "BITZ_BENCH_SHAPES", "17"),
    ] {
        let out = run(
            &executables[name],
            directory.path(),
            &["--bench"],
            &[(variable, value), ("PERFETTO_TRACE_PROCESSOR", "/missing")],
        );
        let error = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{name} {variable}={value}: {error}"
        );
        assert!(
            error.contains("BITZ_BENCH_SHAPES") || error.contains(variable),
            "{name}: {error}"
        );
        assert!(!error.contains("panicked"), "{name}: {error}");
        assert_eq!(
            std::fs::read_dir(directory.path()).unwrap().count(),
            0,
            "{name}"
        );
    }
    for name in ["hybrid_u32_sha256", "hybrid-u32-sha256"] {
        for mode in ["hybrid", "separate", "all-binius", "binius-ligerito", "all"] {
            assert!(
                run(
                    &executables[name],
                    directory.path(),
                    &["--mode", mode, "--sweep", "--help"],
                    &[]
                )
                .status
                .success()
            );
        }
        assert!(
            !run(
                &executables[name],
                directory.path(),
                &["--mode", "all"],
                &[]
            )
            .status
            .success()
        );
        assert!(
            !run(
                &executables[name],
                directory.path(),
                &["--mode", "hybrid", "--profile", "invalid"],
                &[]
            )
            .status
            .success()
        );
    }
    assert!(
        !run(
            &executables["mul_compare"],
            directory.path(),
            &["proof", "--workload", "u64", "--w", "0"],
            &[]
        )
        .status
        .success()
    );
    assert!(
        !run(
            &executables["sha256_e2e_compare"],
            directory.path(),
            &[],
            &[("BITZ_SHA_COMPARE_PREFLIGHT_CHILD", "unknown")]
        )
        .status
        .success()
    );
    for (name, args) in [
        ("sha256_ecdsa", vec!["3", "split", "100", "0"]),
        (
            "sha256_ecdsa_compare",
            vec!["--method", "bitz-split", "--r", "1", "--c", "1"],
        ),
        ("ligerito_bounds", vec!["unknown", "custom:1:4"]),
    ] {
        assert!(
            !run(&executables[name], directory.path(), &args, &[])
                .status
                .success()
        );
    }
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    let export = run(
        &executables["sha256_ecdsa_compare"],
        directory.path(),
        &[
            "--method",
            "bitz-split",
            "--r",
            "1",
            "--c",
            "2",
            "--seed",
            "7",
            "--export-fixture",
            "fixture.json",
        ],
        &[],
    );
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("fixture.json")).unwrap())
            .unwrap();
    assert_eq!(fixture["log_compressions"], 3);
    assert_eq!(fixture["seed"], 7);
}
