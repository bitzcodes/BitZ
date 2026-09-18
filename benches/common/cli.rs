//! CLI support shared by the standalone benchmark executables.
#![allow(dead_code)] // Each executable uses a subset of the shared parsers.
use clap::{Arg, Command, Parser, builder::TypedValueParser};
use std::{fmt::Display, str::FromStr};

#[derive(clap::Args, Default)]
pub struct CargoArgs {
    // Cargo passes this even when harness = false.
    #[arg(long, hide = true)]
    bench: bool,
}

#[derive(Parser)]
pub struct EnvironmentCli {
    #[command(flatten)]
    cargo: CargoArgs,
}

impl EnvironmentCli {
    pub fn parse() {
        <Self as Parser>::parse();
    }
}

/// Private schemas never see argv: their options stay environment-only.
pub fn environment<T: Parser>() -> T {
    let mut command = T::command().disable_help_flag(true).mut_args(|arg| {
        // Private schemas use the environment name in clap's own diagnostics.
        match arg
            .get_env()
            .map(|name| name.to_string_lossy().into_owned())
        {
            Some(name) => arg.long(None::<&str>).short(None::<char>).value_name(name),
            None => arg,
        }
    });
    let matches = command
        .try_get_matches_from_mut(["environment"])
        .unwrap_or_else(|mut error| {
            error.remove(clap::error::ContextKind::Usage);
            error.exit()
        });
    T::from_arg_matches(&matches).unwrap_or_else(|error| error.exit())
}

/// For isolated settings where a separate options struct would add plumbing.
pub fn env<T>(name: &'static str) -> Option<T>
where
    T: FromStr + Clone + Send + Sync + 'static,
    T::Err: Display,
{
    std::env::var_os(name).map(|input| {
        value(name, input, |s: &str| {
            s.parse::<T>().map_err(|error| error.to_string())
        })
    })
}

/// Parse a reconciled environment value with the same diagnostics as argv.
pub fn value<T, P>(name: &'static str, value: impl AsRef<std::ffi::OsStr>, parser: P) -> T
where
    T: Clone + Send + Sync + 'static,
    P: clap::builder::TypedValueParser<Value = T>,
{
    Command::new("environment")
        .disable_help_flag(true)
        .arg(
            Arg::new(name)
                .required(true)
                .allow_hyphen_values(true)
                .value_parser(parser),
        )
        .get_matches_from([std::ffi::OsStr::new("environment"), value.as_ref()])
        .remove_one(name)
        .expect("required environment value")
}

pub fn positive(value: &str) -> Result<usize, String> {
    value
        .parse::<std::num::NonZeroUsize>()
        .map(usize::from)
        .map_err(|error| error.to_string())
}

/// Parse a mixed comma/space list with clap's per-element value parser.
pub fn values<T, P>(name: &'static str, value: &str, parser: P) -> Vec<T>
where
    T: Clone + Send + Sync + 'static,
    P: TypedValueParser<Value = T>,
{
    Command::new("environment")
        .disable_help_flag(true)
        .arg(Arg::new(name).required(true).num_args(1..).allow_hyphen_values(true).value_parser(parser))
        .get_matches_from(std::iter::once("environment").chain(
            value.split([',', ' ']).filter(|value| !value.is_empty())))
        .remove_many(name)
        .expect("required list values")
        .collect()
}

pub fn ecdsa_target(value: &str) -> Result<u32, String> {
    let bits = value.parse::<u32>().map_err(|error| error.to_string())?;
    match bits {
        100 | 128 => Ok(bits),
        _ => Err("expected a security target of 100 or 128 bits".into()),
    }
}

pub fn seed(value: &str) -> Result<u64, String> {
    value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .map_or_else(|| value.parse(), |hex| u64::from_str_radix(hex, 16))
        .map_err(|error| error.to_string())
}

/// Opaque to clap derive, so one environment value parses into one list.
pub type List<T> = Vec<T>;

pub fn list<T: FromStr>(value: &str) -> Result<Vec<T>, String>
where
    T::Err: Display,
{
    let values: Vec<T> = value
        .split([',', ' '])
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|error: T::Err| error.to_string()))
        .collect::<Result<_, _>>()?;
    if values.is_empty() {
        return Err("list must not be empty".into());
    }
    Ok(values)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum BenchmarkPass {
    Latency,
    Memory,
    Both,
}

impl BenchmarkPass {
    pub fn from_env() -> Self {
        #[derive(Parser)]
        struct Env {
            #[arg(long, env = "BITZ_BENCH_PASS", default_value = "latency",
                value_parser = clap::value_parser!(BenchmarkPass).try_map(|pass| {
                    if pass.measures_memory() && !cfg!(feature = "bench-peak-memory") {
                        Err("memory and both require --features bench-peak-memory")
                    } else {
                        Ok(pass)
                    }
                }))]
            pass: BenchmarkPass,
        }
        environment::<Env>().pass
    }
    pub const fn measures_latency(self) -> bool {
        matches!(self, Self::Latency | Self::Both)
    }
    pub const fn measures_memory(self) -> bool {
        matches!(self, Self::Memory | Self::Both)
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Latency => "latency",
            Self::Memory => "memory",
            Self::Both => "both",
        }
    }
}

pub fn enum_list<T: clap::ValueEnum>(value: &str) -> Result<Vec<T>, String> {
    list::<String>(value)?
        .into_iter()
        .map(|value| T::from_str(&value, false))
        .collect()
}
