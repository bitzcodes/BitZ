// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.

mod utils;

use std::{alloc::System, env};

use binius_examples::circuits::sha256sign::{Instance, Params, Sha256SignExample};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use peakmem_alloc::PeakMemAlloc;
use utils::{ExampleBenchmark, SignBenchConfig, print_benchmark_header, run_cs_benchmark};

// Global allocator that tracks peak memory usage
#[global_allocator]
static SHA256SIGN_PEAK_ALLOC: PeakMemAlloc<System> = PeakMemAlloc::new(System);

struct Sha256SignBenchmark {
	config: SignBenchConfig,
	max_msg_len_bytes: u32,
}

impl Sha256SignBenchmark {
	fn new() -> Self {
		let config = SignBenchConfig::from_env(1); // default: 1 signature

		// Parse message size from environment variable
		let max_msg_len_bytes = env::var("MESSAGE_MAX_BYTES")
			.ok()
			.map(|s| s.parse::<u32>().expect("MESSAGE_MAX_BYTES must be a u32"))
			.unwrap_or(320);
		assert!(max_msg_len_bytes > 0, "MESSAGE_MAX_BYTES must be positive");

		Self {
			config,
			max_msg_len_bytes,
		}
	}
}

impl ExampleBenchmark for Sha256SignBenchmark {
	type Params = Params;
	type Instance = Instance;
	type Example = Sha256SignExample;

	fn create_params(&self) -> Self::Params {
		Params {
			n_signatures: self.config.n_signatures,
			max_msg_len_bytes: self.max_msg_len_bytes,
		}
	}

	fn create_instance(&self) -> Self::Instance {
		Instance {}
	}

	fn bench_name(&self) -> String {
		format!("sig_{}_msg_{}", self.config.n_signatures, self.max_msg_len_bytes)
	}

	fn throughput(&self) -> Throughput {
		Throughput::Elements(self.config.n_signatures as u64)
	}

	fn proof_description(&self) -> String {
		format!("{} signatures, {} max bytes", self.config.n_signatures, self.max_msg_len_bytes)
	}

	fn log_inv_rate(&self) -> usize {
		self.config.log_inv_rate
	}

	fn print_params(&self) {
		let params_list = vec![
			("Signatures".to_string(), self.config.n_signatures.to_string()),
			("Max message length".to_string(), format!("{} bytes", self.max_msg_len_bytes)),
			("Log inverse rate".to_string(), self.config.log_inv_rate.to_string()),
		];
		print_benchmark_header("Sha256Sign", &params_list);
	}
}

fn bench_sha256sign_signatures(c: &mut Criterion) {
	let benchmark = Sha256SignBenchmark::new();
	run_cs_benchmark(c, &benchmark, "sha256sign", &SHA256SIGN_PEAK_ALLOC);
}

criterion_group!(sha256sign, bench_sha256sign_signatures);
criterion_main!(sha256sign);
