#!/usr/bin/env bash
# The 2026-09-13 bench-suite campaign queue (README.md, "Integer
# multiplication"): integer-mult tables at odd exponents, every scheme at 1 and 10
# threads, Binius suite under round-by-round accounting, Limber at the pinned
# 100-bit Brakedown target, Plonky3-FRI at rate 1/2.
#
# Every campaign is one fresh runner process, serialized through
# scripts/bench_gate.py (machine lock + swap guard). Run
# this ONLY when no other session is measuring (the gate waits, but a campaign
# started elsewhere without the lock will still collide) and with the source
# tree frozen — the runners reject tracked-source edits mid-campaign.
#
#   bash scripts/run_suite_2026_09_13.sh [u32] [u64] [u128] [multiswap]
#
# No arguments = all phases in that order. Paging policy: cells known to page
# are excluded, except the Binius64-family rows, which run under a raised swap
# guard (the guard aborts a runaway; an aborted campaign exits 86 and the
# queue stops there). Output: PerfRuns/suite-<label>. A label that already
# exists fails the runner: remove the directory or rename before re-running.
set -euo pipefail
cd "$(dirname "$0")/.."

REPS="${BITZ_SUITE_REPS:-5}"
phases="$*"
[ -z "$phases" ] && phases="sha-ecdsa hybrid-counts hybrid-witness u32 u64 u128 multiswap"
has() { case " $phases " in *" $1 "*) return 0;; *) return 1;; esac; }

# One hybrid sweep = one (mode, rate) at one thread count over the variant's
# shapes; the table script joins them per (row, threads). Leading KEY=VAL
# arguments become environment for the bench; the rest are bench arguments.
# The bench binary is built once, before any measurement, and each sweep runs
# it under scripts/rss_sampler.py: the sampler writes the per-case peak
# resident set and swap-out counts the table's Peak mem. column reads.
hybrid_binary() {
  RUSTFLAGS="-C target-cpu=native" cargo +1.98.1 bench --bench hybrid_u32_sha256 \
    --features hybrid --no-run --message-format=json | \
    PYTHONPATH=scripts python3 -c 'import sys; from bench_support import cargo_executables; print(cargo_executables(sys.stdin.read(), ["hybrid_u32_sha256"])["hybrid_u32_sha256"])'
}

hybrid_sweep() { # label swap_gb threads shapes [KEY=VAL...] --mode ...
  local label=$1 guard=$2 threads=$3 shapes=$4; shift 4
  local envs=()
  while [ $# -gt 0 ] && [[ $1 == *=* && $1 != --* ]]; do envs+=("$1"); shift; done
  local dir="PerfRuns/suite-$label"
  # The bench refuses an existing results directory, so the sampler writes its
  # TSV beside it and the file is moved in once the sweep has created the dir.
  local tsv="PerfRuns/$label-peak-rss-and-swap.tsv"
  python3 scripts/bench_gate.py run --label "$label" --swap-grow-gb "$guard" -- \
    env RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS="$threads" \
      ${envs[@]+"${envs[@]}"} \
      python3 scripts/rss_sampler.py --output "$tsv" -- \
        "$HYBRID_BIN" --sweep --shapes "$shapes" --results-dir "$dir" "$@"
  local code=$?
  [ -d "$dir" ] && [ -f "$tsv" ] && mv "$tsv" "$dir/peak-rss-and-swap.tsv"
  return $code
}
HY_COUNTS="9:9,10:10,11:11,12:12,13:13,14:14"
HY_WITNESS="15:7,16:8,17:9,18:10,19:11,20:12"

hybrid_phase() { # phase-name shapes
  local name=$1 shapes=$2
  for T in 10 1; do
    hybrid_sweep "$name-bitz-r2-t$T"  12 "$T" "$shapes" --mode hybrid
    hybrid_sweep "$name-bitz-r8-t$T"  12 "$T" "$shapes" --mode hybrid --profile custom:3:4
    hybrid_sweep "$name-bin-r1-t$T"  30 "$T" "$shapes" BITZ_HYBRID_BINIUS_LOG_INV_RATE=1 --mode all-binius
    hybrid_sweep "$name-bin-r3-t$T"  30 "$T" "$shapes" BITZ_HYBRID_BINIUS_LOG_INV_RATE=3 --mode all-binius
    hybrid_sweep "$name-lig-r1-t$T"  30 "$T" "$shapes" BITZ_BINIUS_LOG_INV_RATE=1 BITZ_BINIUS_LIGERITO_ACCOUNTING=rbr --mode binius-ligerito
    hybrid_sweep "$name-lig-r3-t$T"  30 "$T" "$shapes" BITZ_BINIUS_LOG_INV_RATE=3 BITZ_BINIUS_LIGERITO_ACCOUNTING=rbr --mode binius-ligerito
  done
}

if has sha-ecdsa; then
  # The complete SHA+ECDSA matrix (BitZ rho=1/2,1/8; Binius64 rho=1/2,1/8;
  # opener rho=1/2,1/8 rbr) at threads 1 and 10, over the message sizes the
  # paper table groups by (2^4..2^7 compressions), one runner invocation.
  python3 scripts/bench_gate.py run --label sha-ecdsa --swap-grow-gb 12 -- \
    python3 scripts/run_sha256_ecdsa_compare.py \
      --output "bench_results/suite-sha256-ecdsa-$(date +%Y%m%d)" --exponents 4 5 6 7
fi
if has hybrid-counts || has hybrid-witness; then
  HYBRID_BIN=$(hybrid_binary) || { echo "hybrid bench build failed" >&2; exit 1; }
  [ -x "$HYBRID_BIN" ] || { echo "hybrid bench binary not found" >&2; exit 1; }
fi
if has hybrid-counts;  then hybrid_phase hy-counts  "$HY_COUNTS";  fi
if has hybrid-witness; then hybrid_phase hy-witness "$HY_WITNESS"; fi

if has u32 || has u64 || has u128; then
  workloads=()
  has u32 && workloads+=(u32-mod32)
  has u64 && workloads+=(u64)
  has u128 && workloads+=(u128)
  workload_list=$(IFS=,; echo "${workloads[*]}")
  python3 scripts/bench_gate.py run --label suite-mul --swap-grow-gb 30 -- \
    env RUSTFLAGS="-C target-cpu=native" \
    cargo bench --bench mul_compare --features native-mul-compare,bench-internals -- \
    proof --workload "$workload_list" --backends all --log-n 15,17,19 \
    --threads 1,10 --reps "$REPS" --skip-unsupported --memory rss --out PerfRuns/suite-mul
fi

if has multiswap; then
  # MultiSwap re-measure at 10 threads (was 8): the matched campaign as in
  # README "MultiSwap", with --all-threads 10. Prepare the Limber checkout
  # first (scripts/prepare_matched_limber.py). The campaign script measures
  # single-threaded and all-threads modes itself.
  python3 scripts/bench_gate.py run --label multiswap-t10 --swap-grow-gb 10 -- \
    python3 scripts/run_matched_multiswap_campaign.py \
      --draft \
      --limber-root vendor/limber \
      --security-bits 114 \
      --batch-counts 1,2,4,8,16 \
      --all-threads 10 \
      --warmups 1 \
      --samples 10 \
      --rustflags="-C target-cpu=native"
fi

# Export multiplication rows: python3 scripts/mul_report.py PerfRuns/suite-mul --out reports/suite-mul
echo "suite queue done"
