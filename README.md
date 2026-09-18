# BitZ benchmark workspace

This workspace contains BitZ and materialized source for the retained benchmark
dependencies. Cargo uses relative paths; no submodule initialization, patch
application, or nested Git repository is needed to build.

`provenance.toml` records each vendor's official upstream, original and normalized
upstream revisions, final snapshot commit/tree, and review artifacts. Each vendor
has one customization commit above its normalized upstream history. The release
includes its patch; self-contained Git history bundles are distributed separately.
The root development history is retained internally and excluded from the release.
See [PROVENANCE_CHANGES.md](PROVENANCE_CHANGES.md) for the public source pins,
history normalization, and deliberately retained attribution.

Python 3.11 or newer is required by the campaign and release tools. Cargo may
fetch registry packages and the exact official upstream `halo2curves` revision.
Install `rustup` before running the campaigns. Git is needed to reconstruct
missing vendors or create a ZIP from a development checkout, but not to verify
the source already included in a ZIP.
The field reference microbenchmark and retired external comparison integrations
are omitted; the seven retained campaigns are documented below.

## Run all seven campaigns

From the repository root, or the `bitz/` directory extracted from the ZIP:

```sh
python3 scripts/materialize_vendors.py --check
bash scripts/run_all_benchmarks.sh --dry-run
bash scripts/run_all_benchmarks.sh --smoke
bash scripts/run_all_benchmarks.sh
```

The wrapper runs the seven campaigns below sequentially. It checks vendor source,
installs the pinned Rust toolchains and Perfetto if needed, and creates a fresh
results directory under `bench_results/`. Use `--output DIR` to choose another
new directory. `CARGO_TARGET_DIR` is preserved for build-cache reuse.

Cargo exports `BITZ_REVISION` and `BITZ_DIRTY` from the build script when it
launches benchmarks. The shared validator in `benches/common/mod.rs` accepts
these metadata variables while still rejecting unknown `BITZ_*` knobs. This
fix applies to both the wrapper and the individual commands below. If an older
build reports `unknown BITZ_* environment variable(s): BITZ_DIRTY, BITZ_REVISION`,
rerun with the updated harness; unsetting them in the shell alone cannot fix it
because Cargo adds them again.

`--dry-run` only prints commands; it does not compile, verify dependencies, or
execute benchmarks. `--smoke` uses one size and one measured sample per campaign,
retaining the listed backends, rates, and thread counts. It also exercises the
equal-count hybrid table. Warmups and separate memory trials still run.

One outer benchmark lock protects the complete workflow; before the first
campaign, the gate waits for at least 88% CPU idle held for 120 s. Multiplication
receives `--no-gate` internally to avoid taking that lock twice. The default
swap-growth guard is 34 GiB, configurable with `--swap-grow-gb`. Failures retain
their logs and stop the workflow without claiming completion. The wrapper waits for
idle only once; see [Measurement conditions](#measurement-conditions-of-the-published-numbers)
for how the published numbers were gated.

## Materialize vendors and create the source ZIP

From a development checkout, with Python 3.11+ and Git installed:

```sh
python3 scripts/materialize_vendors.py
python3 scripts/materialize_vendors.py --check
python3 scripts/package_release.py --dry-run
python3 scripts/package_release.py --output outputs/bitz-source.zip
```

The materializer fetches pinned official upstream commits and applies the bundled
patches. It verifies final source trees, skips matching existing vendors without
network access, and refuses to overwrite local changes. Preserve and move aside
any mismatched vendor directory before reconstructing it. `--check` verifies
source and patches without downloading or changing files.

The packager requires included source to be committed and clean. It preserves
source, patches, licenses, build configuration, and fixtures while excluding Git
metadata, editor/agent settings, caches, generated results, root documentation,
and manuscript files. Root-relative directory exclusions do not remove similarly
named vendor inputs. The ZIP has a single `bitz/` directory, reproducible ordering
and timestamps, and a default limit of **20,000,000 bytes**. An existing output
is never overwritten. It prints the exact byte count and SHA-256 after success.

ZIP recipients already have materialized vendors; `--check` works without Git.
No history bundle is required for compilation. Cargo may download registry
packages and the pinned official `halo2curves` revision. Install Perfetto with
`scripts/install_trace_processor.sh` before campaigns that require it.
In a Git checkout, `git log -- vendor/limber` shows the root repository's vendor
updates. Original upstream history is in the separately distributed vendor
bundle; there are no submodules to initialize in the source ZIP.

## Compile without running benchmarks

Install Rust `1.98.1` and `nightly-2026-07-01`, then run:

```sh
bash scripts/compile_export.sh
```

This builds all seven campaigns below and their affected Rust tests.
It uses native CPU code generation and locked dependencies. Registry packages
and the pinned upstream `halo2curves` source may be downloaded. No benchmark,
proof, test executable, or report generator is run by this script. Cargo's
normal build scripts and procedural macros run
as part of compilation. Build outputs and logs are ignored by Git.

To compile the hybrid SHA-256/multiplication benchmark without executing it:

```sh
RUSTFLAGS="-C target-cpu=native" cargo +1.98.1 bench --locked --no-run \
  --bench hybrid_u32_sha256 --features hybrid
```

## Measurement conditions of the published numbers

The paper's numbers were measured on an Apple M5 (24 GB) under the conditions
below. Departing from them moves the numbers by more than most of the effects the
tables report.

- **Idle gate.** Every timed campaign started only after `scripts/bench_gate.py`
  saw at least 88% CPU idle held for 120 s, sampled every 20 s (`--min-idle`,
  `--hold-seconds`, `--poll-seconds`). Run back-to-back on a warm machine, an
  unchanged binary measured a 21% slower prover and a 47% slower verifier (SHA-256,
  `2^14` compressions, 10 threads). `--hold-seconds 60` is acceptable; do not drop
  the wait.
- **Campaign granularity.** One gated invocation per workload, backend, rate and
  thread count, with the sizes running inside it; the SHA-256 tables were gated per
  rate and thread group. The multiplication launcher (campaign 3, without
  `--no-gate`) gates each of its campaigns itself. To reproduce a SHA-256 table
  group, run it separately under the gate, for example:

  ```bash
  python3 scripts/bench_gate.py run --label sha256-p256-rate2-t10 -- \
    python3 scripts/run_sha256_ecdsa_compare.py \
    --output "$RUN_DIR/sha256-p256-rate2-t10" \
    --methods bitz-split binius64 binius64-ligerito --exponents 4 5 6 7 \
    --targets 100 --threads 10 --reps 5 --bitz-profiles custom:1:4 \
    --binius-rates 1 --timing perfetto
  ```

- **GKR forest schedule.** On Apple Silicon, large single-claim forests keep the
  L/4 storage schedule above four worker threads (`src/merged_forest/schedule.rs`).
  The L/8 rule used elsewhere costs 19–37% of prover time there, for about 19% less
  peak RSS. The paper's 10-thread numbers were measured with this rule. Each
  multiplication result records the resolved schedule under
  `effective.gkr_schedules`; expect `l4` at ten threads. Explicit `--gkr-schedule`
  requests are never substituted.
- **Builds.** `cargo +1.98.1`, fat LTO and one codegen unit (the release and bench
  profiles), and `RUSTFLAGS="-C target-cpu=native"` for every scheme, including the
  competitors. Resolve benchmark executables from Cargo's `--message-format=json`
  output, as the runners do, never by listing an existing `target/` directory. A
  stale binary measures old code, and can also reject shapes that the current
  source accepts.
- **Fixed inputs.** The u32 corpus seed is `0x5533_3250_4353_0064`
  (6139306037344403556). Each multiplication result records its per-exponent
  corpus digest (`effective.corpus_digest`); rows measured at another seed are not
  comparable. Transcript domain strings determine the proof bytes, so changing them
  changes the proof-size columns.

Not reproducible from this artifact:

- the Fields-Witch comparison (rates 1/2 and 1/8); its runner is not included;
- the Zinc+ rows; the external Zinc+ comparison is omitted (see `NOTICE.md`).

## Benchmark campaigns

Run the following commands from the repository root in **Bash**. These commands
execute benchmarks and verify generated proofs. Use a fresh `RUN_DIR` for each
campaign and run the workloads sequentially on an otherwise idle machine.

### Setup

```bash
set -euo pipefail

rustup toolchain install 1.98.1
rustup toolchain install nightly-2026-07-01
python3 scripts/materialize_vendors.py --check
bash scripts/install_trace_processor.sh

export RUSTFLAGS="-C target-cpu=native"
export PERFETTO_TRACE_PROCESSOR="$PWD/.tools/perfetto/trace_processor_shell"
unset BITZ_LIG_PROFILE CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_DIR

mkdir -p bench_results
export RUN_DIR="$(mktemp -d "$PWD/bench_results/all-benchmarks-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
LIMBER_DIR="$PWD/vendor/limber"
echo "Results: $RUN_DIR"
```

### 1. SHA-256 + P-256: BitZ, Binius64, Binius64-Ligerito

```bash
python3 scripts/run_sha256_ecdsa_compare.py \
  --output "$RUN_DIR/sha256-p256" \
  --methods bitz-split binius64 binius64-ligerito \
  --exponents 4 5 6 7 \
  --targets 100 \
  --threads 1 10 \
  --reps 5 \
  --bitz-profiles custom:1:4 custom:3:4 \
  --binius-rates 1 3 \
  --timing perfetto \
  2>&1 | tee "$RUN_DIR/sha256-p256.log"
```

### 2. SHA-256 chains: BitZ, Binius64, Binius64-Ligerito

```bash
python3 scripts/run_sha256_chain_compare.py \
  --methods bitz binius64 binius64-ligerito \
  --exponents 7 8 9 10 11 12 13 14 15 16 \
  --threads 1 10 \
  --reps 5 \
  --bitz-profiles custom:1:4 custom:3:4 \
  --binius-rates 1 3 \
  --output "$RUN_DIR/sha256-chain" \
  2>&1 | tee "$RUN_DIR/sha256-chain.log"
```

### 3. Multiplication comparisons

The launcher builds once, validates every selection with the Rust case planner,
runs sequentially, and generates combined reports in `<output>/reports`. Without
a `bitz` or `compare` target, it uses the retained per-backend size limits:
180 configurations across u32-mod32, u64, and u128, with 1 and 10 threads, both
BitZ/Binius rates, one warmup, five samples, and a separate RSS trial. Plonky3-FRI
runs only u32-mod32. Limber stops at exponent 19; other defaults range through
21 or 23 depending on workload and backend.

```bash
python3 scripts/run_multiplication_benchmarks.py \
  --output "$RUN_DIR/multiplication" \
  2>&1 | tee "$RUN_DIR/multiplication.log"
```

Use `--dry-run` to preview this matrix or `--exponents 15 --reps 1 --threads 1`
for a smaller run. Direct experiments, including WHIR, remain available with
`compare -- proof --backends all --log-n 15 --threads 1` or `bitz -- ...`.
For direct experiments, launcher flags precede `--` and Rust options follow it;
benchmark `--dry-run` after the separator compiles and validates case selection.
The reporter consumes `mul-bench/v2` results; historical formats are unsupported.

### 4. BitZ full-product u32 × u32 → u64, with component breakdown

```bash
cargo +1.98.1 run --release --locked --bin bitz \
  --features unchecked,span-metrics -- \
  --mul-sweep 15-22 \
  --threads 10 \
  --reps 5 \
  --profile custom:1:4 \
  --cooldown 20 \
  --latex "$RUN_DIR/u32-full-product.tex" \
  2>&1 | tee "$RUN_DIR/u32-full-product.log"
```

### 5. MultiSwap: BitZ, Limber-Hyrax, Limber-Brakedown

This uses the local `vendor/limber` snapshot. `MSCFG=paper` is a workload name
and does not require a manuscript directory. `--draft` runs proofs and the
local comparison checks while marking canonical trace validation as pending.

Matched reports use the same minimal-byte v1 circuit digest and batch statement
contract for BitZ and Limber. BitZ's fixed-width v2 proof digest is kept separate;
the benchmark computes comparison digests from the actual public matrices and
moduli. This fixes the `canonical digest mismatch` caused by reporting the v2
hash as v1. Rerun affected campaigns into a fresh directory to regenerate traces.

```bash
python3 scripts/run_matched_multiswap_campaign.py \
  --draft \
  --limber-root "$LIMBER_DIR" \
  --security-bits 114 \
  --batch-counts 1,2,4,8,16 \
  --all-threads 10 \
  --warmups 1 \
  --samples 10 \
  --rustflags="-C target-cpu=native" \
  --output-dir "$RUN_DIR/multiswap" \
  2>&1 | tee "$RUN_DIR/multiswap.log"
```

### 6. SHA-256 layout parameter sweep over s and t

The `sha256_product_layout` benchmark holds the workload at `2^14` SHA-256
compressions and sweeps the layout split with `s + t = 29`. By default, it runs
`t = 7..27` (`s = 29 - t`), with one warmup and 21 measured samples per split.
The `t = 28` case is skipped because its projected peak memory exceeds 60 GiB.
This is a controlled fixed-prime layout experiment.

```bash
RUSTFLAGS="-C target-cpu=native" \
RAYON_NUM_THREADS=10 \
cargo +1.98.1 bench --locked \
  --bench sha256_product_layout \
  --features unchecked,span-metrics,bench-internals
```

To select particular splits and change the sample count, prepend
`BITZ_SHA_PRODUCT_TS="13 17" BITZ_BENCH_REPS=5` to the command. This selects
`(t, s) = (13, 16)` and `(17, 12)`, with five measured samples per split.
Add `--no-run` to the Cargo command to compile without executing the sweep.

### 7. Hybrid SHA-256 chain + multiplication modulo 2^32

This is the paper's **“Modular multiplications and bit operations”** experiment
(table label `tab:hybrid-sha256-mul`). It proves `N` relations
`x*y = z + 2^32*w`, with four u32 limbs, together with `M = N/256` chained SHA-256
compressions. The two branches have equal packed witness sizes; their witness
values are independent. Shape `15:7`, for example, means `2^15` multiplications
and `2^7` compressions.

| CLI mode | Multiplication and SHA proof |
|---|---|
| `hybrid` | BitZ multiplication PIOP + Binius64 SHA PIOP, with one shared BitZ opening |
| `all-binius` | Both relations in Binius64, using BaseFold/FRI (paper: Binius UDR) |
| `binius-ligerito` | Both relations in Binius64, using the BitZ/Ligerito opener (paper: Binius Johnson) |

The sweep below runs all three modes at rates 1/2 and 1/8, with 1 and 10
threads, one warmup and five measured iterations per shape. The BitZ hybrid
checks a 100-bit whole-protocol union bound; the Binius/Ligerito mode uses
100-bit round-by-round accounting. The BaseFold query target is explicitly
set to 100 below, overriding the hybrid CLI's default of 112.

Build the benchmark once and obtain its executable path from Cargo's artifact
record, then run each configuration in a separate sweep. On macOS, the sampler
also records per-case peak RSS and swap-outs for the table's memory column.

```bash
cargo +1.98.1 bench --locked --no-run \
  --bench hybrid_u32_sha256 --features hybrid --message-format=json \
  > "$RUN_DIR/hybrid-build.jsonl"

HYBRID_BIN="$(python3 - "$RUN_DIR/hybrid-build.jsonl" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    artifacts = [json.loads(line) for line in stream]
executables = [entry["executable"] for entry in artifacts
               if entry.get("reason") == "compiler-artifact"
               and entry.get("target", {}).get("name") == "hybrid_u32_sha256"
               and entry.get("executable")]
if len(executables) != 1:
    raise SystemExit("expected exactly one hybrid benchmark executable")
print(executables[0])
PY
)"

HYBRID_SHAPES="15:7,16:8,17:9,18:10,19:11,20:12"
HYBRID_ROOT="$RUN_DIR/hybrid-witness"
mkdir -p "$HYBRID_ROOT"

hybrid_sweep() {
  local mode="$1" rate="$2" threads="$3"
  local dir="$HYBRID_ROOT/$mode-rate$rate-t$threads"
  local tsv="$dir-peak-rss-and-swap.tsv"
  local command=("$HYBRID_BIN" --sweep --mode "$mode"
    --shapes "$HYBRID_SHAPES" --iterations 5 --results-dir "$dir")
  if [[ "$mode" == hybrid ]]; then
    command+=(--profile "custom:$rate:4")
  fi
  if [[ "$(uname -s)" == Darwin ]]; then
    command=(python3 scripts/rss_sampler.py --output "$tsv" -- "${command[@]}")
  fi
  env RAYON_NUM_THREADS="$threads" \
    BITZ_HYBRID_BINIUS_LOG_INV_RATE="$rate" \
    BITZ_HYBRID_BINIUS_SECURITY_BITS=100 \
    BITZ_BINIUS_LOG_INV_RATE="$rate" \
    BITZ_BINIUS_LIGERITO_ACCOUNTING=rbr \
    "${command[@]}" 2>&1 | tee "$dir.log"
  if [[ -f "$tsv" ]]; then
    mv "$tsv" "$dir/peak-rss-and-swap.tsv"
  fi
}

for threads in 1 10; do
  for rate in 1 3; do
    for mode in hybrid all-binius binius-ligerito; do
      hybrid_sweep "$mode" "$rate" "$threads"
    done
  done
done

echo "Completed. Results: $RUN_DIR"
```

For the optional **equal-operation-count** experiment (`N = M`), set the
following variables, then repeat the three nested `for` loops above:

```bash
HYBRID_SHAPES="9:9,10:10,11:11,12:12,13:13,14:14"
HYBRID_ROOT="$RUN_DIR/hybrid-counts"
mkdir -p "$HYBRID_ROOT"
```

This is a separate workload from the paper's equal-witness table. Each sweep's
result directory must not already exist. It will contain `summary.csv`,
`run.txt`, and per-case CSV/log files.

### Tables and figures

The bundled `scripts/zk_trace.py` provides trace report tooling. Tables default
to `outputs/tables/` and figures to `outputs/figures/`; the root `paper/` directory
is not needed. After collecting the hybrid witness sweeps above, its table can
be generated with:

```bash
hybrid_rows=()
for threads in 1 10; do
  for rate in 1 3; do
    for mode in hybrid all-binius binius-ligerito; do
      hybrid_rows+=(--row "$mode@$rate:$threads=$RUN_DIR/hybrid-witness/$mode-rate$rate-t$threads")
    done
  done
done
python3 scripts/hybrid_table.py --variant witness "${hybrid_rows[@]}" \
  --output "$RUN_DIR/hybrid-witness.tex"
```

For equal-count results, use `--variant counts` and the `hybrid-counts`
directories instead. Generation of a table does not rerun the proofs.

Source attribution and licenses are retained alongside the incorporated code.
Historical upstream citations may identify their original contributors;
metadata normalization does not prevent recognizing previously published code.

## Reviewing and validating the release

Use `python3 scripts/materialize_vendors.py --check` to verify vendor source and
patch checksums. Inspect the customization patches at the paths recorded in
`provenance.toml`. Vendor history bundles are separate artifacts; clone one to
inspect its normalized upstream history and single customization commit. Verify
the separately downloaded bundles with:

```sh
python3 scripts/verify_vendor_histories.py --bundle-dir /path/to/bundles
```

Run the release tooling tests with:

```sh
python3 -m unittest discover -s scripts -p test_release_tooling.py -v
```

The root development history is retained internally and excluded from the ZIP.
License notices and scholarly attribution are retained alongside the code.
