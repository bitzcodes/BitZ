# Binius64 SHA-chain/P-256 worker

This isolated worker compares Binius64 BaseFold with the BitZ Ligerito opener
over the same SHA-chain and standard P-256 circuit. Its source dependency is
the local `../../vendor/binius64` snapshot recorded in `../../provenance.toml`.
Build metadata includes that snapshot and the root BitZ commit.

Compile without executing the worker:

```sh
cargo +1.98.1 build --release --locked --manifest-path benchmarks/binius64/Cargo.toml
```

The campaign supports `--methods bitz-split binius64 binius64-ligerito`,
`--exponents`, `--bitz-profiles custom:1:4 custom:3:4`, and `--binius-rates 1 3`.
Running `build.py` also executes the worker's `--build-info` command and writes
a binary provenance sidecar used by the campaign runner.

The public statement is exponent and canonical P-256 Qx, Qy, r, s;
the private message contains `64 * (2^i - 1)` bytes. The proofs are non-ZK.
Fixture generation is outside the recorded proof stages. Benchmarks report
their actual security model and target with each sample.
