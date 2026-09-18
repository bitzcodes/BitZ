// Copyright 2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
use anyhow::Result;
use binius_examples::{Cli, circuits::sha256sign::Sha256SignExample};

fn main() -> Result<()> {
	Cli::new("sha256sign")
		.circuit::<Sha256SignExample>("sha256sign", "secp256k1 recovery with SHA-256")
		.run()
}
