#!/usr/bin/env bash
# Run the seven README campaigns sequentially, under one benchmark lock.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all_benchmarks.sh [--smoke] [--dry-run] [--output DIR]
                                        [--no-gate] [--swap-grow-gb N]

Full: all seven campaigns with the README sizes, rates and thread counts.
Smoke: one size and one measured sample per campaign, keeping every listed
backend/rate/thread count; also checks the equal-count hybrid table.
Smoke still builds and runs real proofs, warmups and memory trials.
Dry-run: print commands without downloads, builds, locks or output files.

Requires Python 3.11+, rustup, and the materialized release workspace.
Vendor source is verified first. Rust toolchains and Perfetto are installed
if needed. Existing output directories are never reused. CARGO_TARGET_DIR
is preserved so an existing build cache can be used.
Cargo's BITZ_REVISION and BITZ_DIRTY build metadata are accepted by the shared
benchmark validator; unknown BITZ_* knobs still fail validation.
The default outer swap-growth guard is 34 GiB. Multiplication uses --no-gate
inside this wrapper to avoid acquiring the same lock twice.
EOF
}

ORIGINAL_ARGS=("$@")
SMOKE=0 DRY_RUN=0 NO_GATE=0 INSIDE_GATE=0
OUTPUT="" SWAP_GROW_GB=34
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-gate) NO_GATE=1; shift ;;
    --inside-gate) INSIDE_GATE=1; shift ;;
    --output|--swap-grow-gb)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      if [[ "$1" == --output ]]; then OUTPUT="$2"; else SWAP_GROW_GB="$2"; fi
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! "$SWAP_GROW_GB" =~ ^[0-9]+([.][0-9]+)?$ || ! "$SWAP_GROW_GB" =~ [1-9] ]]; then
  echo '--swap-grow-gb must be a positive finite number' >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "$OUTPUT" && "$OUTPUT" != /* ]]; then OUTPUT="$ROOT/$OUTPUT"; fi
if [[ -n "$OUTPUT" && ( -e "$OUTPUT" || -L "$OUTPUT" ) ]]; then
  echo "Refusing existing output directory: $OUTPUT" >&2
  exit 1
fi

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }
run() {
  if [[ "$DRY_RUN" == 1 ]]; then print_command "$@"; else "$@"; fi
}
logged() {
  local log="$1"; shift
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command "$@"
  else
    "$@" 2>&1 | tee "$log"
  fi
}

if [[ "$NO_GATE" == 0 && "$INSIDE_GATE" == 0 ]]; then
  gate=(python3 scripts/bench_gate.py run --label all-benchmarks --swap-grow-gb "$SWAP_GROW_GB"
        -- bash "$ROOT/scripts/run_all_benchmarks.sh" --inside-gate ${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"})
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Outer lock:\n'; print_command "${gate[@]}"
  else
    exec "${gate[@]}"
  fi
fi

# These seven campaigns set their own experiment controls. Inherited SHA shape
# aliases, for example, can conflict with the layout sweep in step 6.
# Cargo re-exports BITZ_REVISION and BITZ_DIRTY from the build script after this
# cleanup. benches/common/mod.rs allows both metadata variables, so the fix
# covers every Cargo-launched benchmark, including standalone README commands.
while IFS= read -r benchmark_env_name; do
  case "$benchmark_env_name" in
    BITZ_BENCH_LOCK) ;;
    BITZ_*|F2_*|F2Z_*|RAYON_*|BD*|HARDWARE_CONCURRENCY) unset "$benchmark_env_name" ;;
  esac
done < <(compgen -e)

# Verification is mandatory, including for smoke runs. Bundles are not needed.
run python3 -c 'import sys; sys.exit("Python 3.11+ is required" if sys.version_info < (3, 11) else 0)'
run python3 scripts/materialize_vendors.py --check
run rustup toolchain install 1.98.1
run rustup toolchain install nightly-2026-07-01
run bash scripts/install_trace_processor.sh

if [[ "$DRY_RUN" == 1 ]]; then
  RUN_DIR="${OUTPUT:-$ROOT/bench_results/<new-all-benchmarks-directory>}"
elif [[ -n "$OUTPUT" ]]; then
  mkdir -p -- "$(dirname -- "$OUTPUT")"
  mkdir -- "$OUTPUT"
  RUN_DIR="$OUTPUT"
else
  mkdir -p bench_results
  RUN_DIR="$(mktemp -d "$ROOT/bench_results/all-benchmarks-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
fi
export RUN_DIR
printf 'Results: %s\n' "$RUN_DIR"
export RUSTFLAGS="-C target-cpu=native"
export PERFETTO_TRACE_PROCESSOR="$ROOT/.tools/perfetto/trace_processor_shell"
unset BITZ_LIG_PROFILE CARGO_ENCODED_RUSTFLAGS

REPS=5
P256_EXPONENTS=(4 5 6 7)
CHAIN_EXPONENTS=(7 8 9 10 11 12 13 14 15 16)
MUL_SIZE_ARGS=()
FULL_PRODUCT_RANGE=15-22
FULL_PRODUCT_COOLDOWN=20
MULTISWAP_BATCHES=1,2,4,8,16
MULTISWAP_SAMPLES=10
HYBRID_SHAPES="15:7,16:8,17:9,18:10,19:11,20:12"
LAYOUT_ENV=()
if [[ "$SMOKE" == 1 ]]; then
  REPS=1
  P256_EXPONENTS=(4)
  CHAIN_EXPONENTS=(7)
  MUL_SIZE_ARGS=(--exponents 15)
  FULL_PRODUCT_RANGE=15-15
  FULL_PRODUCT_COOLDOWN=0
  MULTISWAP_BATCHES=1
  MULTISWAP_SAMPLES=1
  HYBRID_SHAPES="15:7"
  LAYOUT_ENV=(BITZ_SHA_PRODUCT_TS=13 BITZ_BENCH_REPS=1)
fi

printf '\n[1/7] SHA-256 + P-256\n'
logged "$RUN_DIR/sha256-p256.log" python3 scripts/run_sha256_ecdsa_compare.py \
  --output "$RUN_DIR/sha256-p256" --methods bitz-split binius64 binius64-ligerito \
  --exponents "${P256_EXPONENTS[@]}" --targets 100 --threads 1 10 --reps "$REPS" \
  --bitz-profiles custom:1:4 custom:3:4 --binius-rates 1 3 --timing perfetto

printf '\n[2/7] SHA-256 chains\n'
logged "$RUN_DIR/sha256-chain.log" python3 scripts/run_sha256_chain_compare.py \
  --methods bitz binius64 binius64-ligerito --exponents "${CHAIN_EXPONENTS[@]}" \
  --threads 1 10 --reps "$REPS" --bitz-profiles custom:1:4 custom:3:4 --binius-rates 1 3 \
  --output "$RUN_DIR/sha256-chain"

printf '\n[3/7] Multiplication comparisons\n'
logged "$RUN_DIR/multiplication.log" python3 scripts/run_multiplication_benchmarks.py \
  --no-gate --workloads u32 u64 u128 --backends bitz binius64 binius64-ligerito plonky3-fri limber \
  --threads 1 10 --reps "$REPS" --bitz-profiles custom:1:4 custom:3:4 --binius-rates 1 3 \
  ${MUL_SIZE_ARGS[@]+"${MUL_SIZE_ARGS[@]}"} --output "$RUN_DIR/multiplication"

printf '\n[4/7] BitZ full-product multiplication\n'
logged "$RUN_DIR/u32-full-product.log" cargo +1.98.1 run --release --locked --bin bitz \
  --features unchecked,span-metrics -- --mul-sweep "$FULL_PRODUCT_RANGE" --threads 10 \
  --reps "$REPS" --profile custom:1:4 --cooldown "$FULL_PRODUCT_COOLDOWN" \
  --latex "$RUN_DIR/u32-full-product.tex"

printf '\n[5/7] MultiSwap\n'
# The BitZ harness emits Limber-compatible canonical comparison digests while
# preserving its v2 proof transcript hashes. Keep the runner's digest checks on.
logged "$RUN_DIR/multiswap.log" python3 scripts/run_matched_multiswap_campaign.py \
  --draft --limber-root "$ROOT/vendor/limber" --security-bits 114 --batch-counts "$MULTISWAP_BATCHES" \
  --all-threads 10 --warmups 1 --samples "$MULTISWAP_SAMPLES" --rustflags="-C target-cpu=native" \
  --output-dir "$RUN_DIR/multiswap"

printf '\n[6/7] SHA-256 layout parameter sweep\n'
logged "$RUN_DIR/sha256-layout.log" env ${LAYOUT_ENV[@]+"${LAYOUT_ENV[@]}"} \
  RAYON_NUM_THREADS=10 BITZ_SHA_RESULT_PATH="$RUN_DIR/sha256-layout.csv" \
  cargo +1.98.1 bench --locked --bench sha256_product_layout --features unchecked,span-metrics,bench-internals

printf '\n[7/7] Hybrid SHA-256 + multiplication\n'
HYBRID_BUILD=(cargo +1.98.1 bench --locked --no-run --bench hybrid_u32_sha256 --features hybrid --message-format=json)
if [[ "$DRY_RUN" == 1 ]]; then
  print_command "${HYBRID_BUILD[@]}"
  HYBRID_BIN='<Cargo executable: hybrid_u32_sha256>'
else
  "${HYBRID_BUILD[@]}" > "$RUN_DIR/hybrid-build.jsonl" 2> "$RUN_DIR/hybrid-build.log" || {
    code=$?; cat "$RUN_DIR/hybrid-build.log" >&2; exit "$code"
  }
  cat "$RUN_DIR/hybrid-build.log" >&2
  HYBRID_BIN="$(python3 - "$RUN_DIR/hybrid-build.jsonl" "$ROOT" <<'PY'
import json
from pathlib import Path
import sys
source = Path(sys.argv[2]) / 'benches/hybrid_u32_sha256.rs'
with open(sys.argv[1]) as stream:
    artifacts = [json.loads(line) for line in stream if line.startswith('{')]
executables = {entry['executable'] for entry in artifacts
               if entry.get('reason') == 'compiler-artifact'
               and 'bench' in entry.get('target', {}).get('kind', [])
               and Path(entry['target']['src_path']).resolve() == source.resolve()
               and entry.get('executable')}
if len(executables) != 1:
    raise SystemExit('expected exactly one hybrid benchmark executable')
print(executables.pop())
PY
)"
fi

hybrid_sweep() {
  local mode="$1" rate="$2" threads="$3"
  local dir="$HYBRID_ROOT/$mode-rate$rate-t$threads"
  local tsv="$dir-peak-rss-and-swap.tsv"
  local command=("$HYBRID_BIN" --sweep --mode "$mode" --shapes "$HYBRID_SHAPES"
                 --iterations "$REPS" --results-dir "$dir")
  if [[ "$mode" == hybrid ]]; then command+=(--profile "custom:$rate:4"); fi
  if [[ "$(uname -s)" == Darwin ]]; then
    command=(python3 scripts/rss_sampler.py --output "$tsv" -- "${command[@]}")
  fi
  logged "$dir.log" env RAYON_NUM_THREADS="$threads" BITZ_HYBRID_BINIUS_LOG_INV_RATE="$rate" \
    BITZ_HYBRID_BINIUS_SECURITY_BITS=100 BITZ_BINIUS_LOG_INV_RATE="$rate" \
    BITZ_BINIUS_LIGERITO_ACCOUNTING=rbr "${command[@]}"
  if [[ "$DRY_RUN" == 0 && -f "$tsv" ]]; then mv "$tsv" "$dir/peak-rss-and-swap.tsv"; fi
}

hybrid_campaign() {
  local variant="$1" threads rate mode
  local rows=()
  HYBRID_ROOT="$RUN_DIR/hybrid-$variant"
  run mkdir -p "$HYBRID_ROOT"
  for threads in 1 10; do
    for rate in 1 3; do
      for mode in hybrid all-binius binius-ligerito; do
        hybrid_sweep "$mode" "$rate" "$threads"
        rows+=(--row "$mode@$rate:$threads=$HYBRID_ROOT/$mode-rate$rate-t$threads")
      done
    done
  done
  run python3 scripts/hybrid_table.py --variant "$variant" "${rows[@]}" --output "$RUN_DIR/hybrid-$variant.tex"
}
hybrid_campaign witness
if [[ "$SMOKE" == 1 ]]; then HYBRID_SHAPES="9:9"; hybrid_campaign counts; fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf '\nDry-run complete; no campaigns were executed.\n'
else
  printf '\nAll seven campaigns completed. Results: %s\n' "$RUN_DIR"
fi
