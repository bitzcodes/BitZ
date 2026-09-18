#![cfg(feature = "span-metrics")]

#[cfg(not(feature = "bench-internals"))]
#[allow(dead_code)]
#[path = "../benches/common/mod.rs"]
mod common;
#[cfg(feature = "bench-internals")]
#[allow(dead_code)]
#[path = "../benches/sha256_compressions.rs"]
mod sha256_compressions;
use common::cli;
#[cfg(feature = "bench-internals")]
use sha256_compressions::common;

use clap::{CommandFactory, Parser, ValueEnum, error::ErrorKind};

#[test]
fn build_metadata_is_accepted_but_unknown_knobs_are_rejected() {
    if std::env::var_os("BENCH_KNOWN_ENV_PROBE").is_some() {
        common::enforce_known_env();
        return;
    }
    for typo in [false, true] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "build_metadata_is_accepted_but_unknown_knobs_are_rejected",
                "--nocapture",
            ])
            .env_clear()
            .env("BENCH_KNOWN_ENV_PROBE", "1")
            .env("BITZ_REVISION", env!("BITZ_REVISION"))
            .env("BITZ_DIRTY", env!("BITZ_DIRTY"))
            .env("BITZ_BENCH_REPS", "1");
        if typo {
            command.env("BITZ_BENCH_REPZ", "1");
        }
        let output = command.output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            output.status.code(),
            Some(if typo { 2 } else { 0 }),
            "{stderr}"
        );
        if typo {
            assert!(
                stderr.contains("unknown BITZ_* environment variable(s): BITZ_BENCH_REPZ"),
                "{stderr}"
            );
        }
    }
}

#[test]
fn cargo_flag_help_and_unknown_options() {
    cli::EnvironmentCli::command().debug_assert();
    assert!(cli::EnvironmentCli::try_parse_from(["bench", "--bench"]).is_ok());
    assert_eq!(
        cli::EnvironmentCli::try_parse_from(["bench", "--unknown"])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::UnknownArgument
    );
    assert_eq!(
        cli::EnvironmentCli::try_parse_from(["bench", "--help"])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::DisplayHelp
    );
    assert!(
        !cli::EnvironmentCli::command()
            .render_help()
            .to_string()
            .contains("--bench")
    );
}

#[test]
fn counts_seeds_lists_and_passes_preserve_supported_values() {
    assert_eq!(cli::positive("3").unwrap(), 3);
    for text in ["0", "-1", "oops", "18446744073709551616"] {
        assert!(cli::positive(text).is_err());
    }
    for text in ["255", "0xff", "0XFF"] {
        assert_eq!(cli::seed(text).unwrap(), 255);
    }
    for text in ["", "-1", "0x", "18446744073709551616"] {
        assert!(cli::seed(text).is_err());
    }
    assert_eq!(cli::list::<usize>("15, 16  17").unwrap(), vec![15, 16, 17]);
    for text in ["", " , ", "15,nope"] {
        assert!(cli::list::<usize>(text).is_err());
    }
    assert_eq!(cli::value("SEED", "0XFF", cli::seed), 255);
    assert_eq!(cli::value("COUNT", "3", cli::positive), 3);
    assert_eq!(
        cli::value(
            "LABEL",
            "--literal",
            clap::builder::StringValueParser::new()
        ),
        "--literal"
    );
    for (text, latency, memory) in [
        ("latency", true, false),
        ("memory", false, true),
        ("both", true, true),
    ] {
        let pass = cli::BenchmarkPass::from_str(text, false).unwrap();
        assert_eq!(
            (
                pass.as_str(),
                pass.measures_latency(),
                pass.measures_memory()
            ),
            (text, latency, memory)
        );
    }
    assert!(cli::BenchmarkPass::from_str("unknown", false).is_err());
}

#[derive(Parser)]
struct Environment {
    #[arg(long, env = "BITZ_CLI_TEST_REPS", default_value = "5", value_parser = cli::positive)]
    reps: usize,
    #[arg(long, env = "BITZ_CLI_TEST_SHAPES", default_value = "15 16", value_parser = cli::list::<u32>)]
    shapes: cli::List<u32>,
}

#[test]
fn environment_probe() {
    let Ok(mode) = std::env::var("BITZ_CLI_TEST_CHILD") else {
        return;
    };
    if mode == "common" {
        let reps = common::reps(Some("BITZ_SHA_REPS"), 3);
        let seed = common::seed(Some("BITZ_SHA_SEED"), 7);
        let shapes = common::shape_values(Some("BITZ_SHA_LOG2S"), str::parse::<usize>);
        let profile = common::security_profile(bitz::piop::spartan::PrimePolicy::SingleDerived);
        println!(
            "COMMON {reps} {seed} {shapes:?} {:?}",
            profile.map(|p| p.name())
        );
    } else if mode == "value" {
        let raw = std::env::var("BITZ_CLI_TEST_VALUE").unwrap();
        let seed = cli::value("BITZ_CLI_TEST_VALUE", &raw, cli::seed);
        println!("CLI_VALUE {seed}");
    } else if mode == "pass" {
        println!("CLI_PASS {}", cli::BenchmarkPass::from_env().as_str());
    } else if mode == "shapes" {
        let shapes = common::shape_values(Some("BITZ_SHA_LOG2S"),
            clap::builder::RangedU64ValueParser::<usize>::new().range(8..=25));
        println!("CLI_SHAPES {shapes:?}");
    } else {
        let scalar = cli::env::<u32>("BITZ_CLI_TEST_SCALAR");
        let config = cli::environment::<Environment>();
        println!("CLI_ENV {scalar:?} {} {:?}", config.reps, config.shapes);
    }
}

#[test]
fn pass_feature_gate_and_shape_bounds_use_clap_errors() {
    for (value, accepted) in [
        ("latency", true),
        ("memory", cfg!(feature = "bench-peak-memory")),
        ("both", cfg!(feature = "bench-peak-memory")),
        ("invalid", false),
    ] {
        let out = child(&[("BITZ_CLI_TEST_CHILD", "pass"), ("BITZ_BENCH_PASS", value)]);
        assert_eq!(out.status.code(), Some(if accepted { 0 } else { 2 }));
        if !accepted {
            let error = String::from_utf8_lossy(&out.stderr);
            assert!(error.contains("BITZ_BENCH_PASS"), "{error}");
            assert!(!error.contains("panicked"), "{error}");
            if value != "invalid" { assert!(error.contains("bench-peak-memory"), "{error}"); }
        }
    }
    for variable in ["BITZ_BENCH_SHAPES", "BITZ_SHA_LOG2S"] {
        let out = child(&[("BITZ_CLI_TEST_CHILD", "shapes"), (variable, "8, 25 +08")]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert!(String::from_utf8_lossy(&out.stdout).contains("Some([8, 25, 8])"));
        for value in ["", " , ", "7", "26", "8 nope", "-1", "18446744073709551616"] {
            let out = child(&[("BITZ_CLI_TEST_CHILD", "shapes"), (variable, value)]);
            assert_eq!(out.status.code(), Some(2), "accepted {variable}={value}");
            assert!(String::from_utf8_lossy(&out.stderr).contains("BITZ_BENCH_SHAPES"));
        }
    }
}

fn child(env: &[(&str, &str)]) -> std::process::Output {
    std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "environment_probe", "--nocapture"])
        .env_clear()
        .env("BITZ_CLI_TEST_CHILD", "environment")
        .envs(env.iter().copied())
        .output()
        .unwrap()
}

#[test]
fn environment_defaults_values_and_errors_are_isolated() {
    Environment::command().debug_assert();
    for (env, expected) in [
        (vec![], "CLI_ENV None 5 [15, 16]"),
        (
            vec![
                ("BITZ_CLI_TEST_SCALAR", "7"),
                ("BITZ_CLI_TEST_REPS", "3"),
                ("BITZ_CLI_TEST_SHAPES", "15, 17 19"),
            ],
            "CLI_ENV Some(7) 3 [15, 17, 19]",
        ),
        (
            vec![
                ("BITZ_CLI_TEST_CHILD", "value"),
                ("BITZ_CLI_TEST_VALUE", "0xff"),
            ],
            "CLI_VALUE 255",
        ),
    ] {
        let output = child(&env);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains(expected));
    }
    for (name, value, diagnostic) in [
        ("BITZ_CLI_TEST_SCALAR", "oops", "BITZ_CLI_TEST_SCALAR"),
        ("BITZ_CLI_TEST_REPS", "0", "BITZ_CLI_TEST_REPS"),
        ("BITZ_CLI_TEST_SHAPES", "15 nope", "BITZ_CLI_TEST_SHAPES"),
    ] {
        let output = child(&[(name, value)]);
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains(diagnostic));
    }
    let output = child(&[
        ("BITZ_CLI_TEST_CHILD", "value"),
        ("BITZ_CLI_TEST_VALUE", "0x"),
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("BITZ_CLI_TEST_VALUE"));
}

#[test]
fn legacy_aliases_keep_precedence_conflicts_and_warnings() {
    let child = |settings: &[(&str, &str)]| {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "environment_probe", "--nocapture"])
            .env_clear()
            .env("BITZ_CLI_TEST_CHILD", "common")
            .envs(settings.iter().copied())
            .output()
            .unwrap()
    };
    let default = child(&[]);
    assert!(default.status.success());
    assert!(String::from_utf8_lossy(&default.stdout).contains("COMMON 3 7 None None"));
    let aliases = child(&[
        ("BITZ_SHA_REPS", "4"),
        ("BITZ_SHA_SEED", "0Xff"),
        ("BITZ_SHA_LOG2S", "15, 17 19"),
        ("BITZ_BENCH_LAMBDA", " LaMbDa128 "),
    ]);
    assert!(
        aliases.status.success(),
        "{}",
        String::from_utf8_lossy(&aliases.stderr)
    );
    assert!(
        String::from_utf8_lossy(&aliases.stdout)
            .contains("COMMON 4 255 Some([15, 17, 19]) Some(\"lambda128\")")
    );
    assert!(String::from_utf8_lossy(&aliases.stderr).contains("deprecated"));
    let equal = child(&[("BITZ_SHA_REPS", "4"), ("BITZ_BENCH_REPS", "4")]);
    assert!(equal.status.success());
    assert!(!String::from_utf8_lossy(&equal.stderr).contains("deprecated"));
    for settings in [
        vec![("BITZ_SHA_REPS", "4"), ("BITZ_BENCH_REPS", "04")],
        vec![("BITZ_SHA_SEED", "0xff"), ("BITZ_BENCH_SEED", "255")],
        vec![("BITZ_SHA_LOG2S", "15 17"), ("BITZ_BENCH_SHAPES", "15,17")],
        vec![("BITZ_BENCH_REPS", "0")],
        vec![("BITZ_BENCH_LAMBDA", "114")],
    ] {
        assert_eq!(
            child(&settings).status.code(),
            Some(2),
            "accepted {settings:?}"
        );
    }
}
