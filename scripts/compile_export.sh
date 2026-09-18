#!/usr/bin/env bash
# Compile the retained campaign targets and affected tests without running them.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 scripts/materialize_vendors.py --check
export RUSTFLAGS="-C target-cpu=native"
unset BITZ_LIG_PROFILE CARGO_ENCODED_RUSTFLAGS
LOG_DIR="${BITZ_COMPILE_LOG_DIR:-$ROOT/outputs/compile}"
mkdir -p "$LOG_DIR"

compile() {
    local label="$1"; shift
    printf 'Compiling %s\n' "$label"
    "$@" 2>&1 | tee "$LOG_DIR/$label.log"
}

compile sha256-p256 cargo +1.98.1 build --release --locked \
    --features sha256-ecdsa-compare --bench sha256_ecdsa_compare
compile binius-worker cargo +1.98.1 build --release --locked \
    --manifest-path benchmarks/binius64/Cargo.toml
compile sha256-chain cargo +1.98.1 build --release --locked \
    --features unchecked,span-metrics,binius64-bench --bench sha256_chain_compare
compile multiplication cargo +1.98.1 bench --no-run --locked \
    --features bench-internals,native-mul-compare --bench mul_compare --bench mul_bitz
compile full-product cargo +1.98.1 build --release --locked \
    --features unchecked,span-metrics --bin bitz
compile bitz-multiswap cargo +1.98.1 bench --no-run --locked \
    --features unchecked,span-metrics --bench multiswap
compile limber-multiswap cargo +nightly-2026-07-01 bench --no-run --locked \
    --manifest-path vendor/limber/Cargo.toml --bench multiswap_modp
compile hybrid cargo +1.98.1 bench --no-run --locked \
    --features hybrid --bench hybrid_u32_sha256
compile sha256-layout cargo +1.98.1 bench --no-run --locked \
    --features unchecked,span-metrics,bench-internals --bench sha256_product_layout
compile comparison-tests cargo +1.98.1 test --release --locked --no-run \
    --features bench-internals,native-mul-compare,sha256-ecdsa-compare,hybrid \
    --test sha256_ecdsa_comparison --test sha256_chain_comparison \
    --test native_mul_compare --test mul_bench --test benchmark_reporting
compile binius-worker-tests cargo +1.98.1 test --release --locked --no-run \
    --manifest-path benchmarks/binius64/Cargo.toml
compile field-tests cargo +1.98.1 test --release --locked --no-run \
    --manifest-path vendor/field/Cargo.toml --features serde
compile plonky3-bn254-tests cargo +1.98.1 test --release --locked --no-run \
    --manifest-path vendor/plonky3/Cargo.toml -p p3-bn254
printf 'Compilation completed. No benchmark or test executable was run.\n'
