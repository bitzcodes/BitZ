#!/usr/bin/env bash
# Profile one criterion benchmark with samply and open an interactive flame
# graph (Firefox Profiler) in the browser. No sudo required — samply uses the
# macOS sampling API directly (unlike `cargo flamegraph`, which drives dtrace
# and is normally blocked by SIP on Apple Silicon).
#
# Usage:
#   ./scripts/flamegraph.sh                                  # prove/msshape_c14, 20s
#   ./scripts/flamegraph.sh <bench> <filter-regex> <seconds>
#   ./scripts/flamegraph.sh imod_spartan_modp 'verify/msshape_c14' 15
#
# Env knobs:
#   RAYON_NUM_THREADS=1   profile single-threaded (frames not split across the
#                         rayon pool — usually far more readable)
#   SECS / FILTER / BENCH override the positional args if you prefer
#
# Builds with the `profiling` profile (release opt level + full debug symbols +
# thin LTO; see Cargo.toml). criterion's --profile-time runs the routine in a
# bare loop for <seconds> with no statistical machinery, which is exactly what
# an external sampler wants.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH="${BENCH:-${1:-imod_spartan_modp}}"
FILTER="${FILTER:-${2:-prove/msshape_c14}}"
SECS="${SECS:-${3:-20}}"

echo "==> building $BENCH (profiling profile)…" >&2
BIN=$(RUSTFLAGS="-C target-cpu=native" \
  cargo bench --bench "$BENCH" --profile profiling --no-run --message-format=json 2>/dev/null \
  | jq -r 'select(.executable != null and (.target.kind[]? == "bench") and .target.name == "'"$BENCH"'") | .executable' \
  | tail -1)

if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
  echo "error: could not locate the built bench binary for '$BENCH'" >&2
  exit 1
fi

echo "==> binary: $BIN" >&2
echo "==> profiling '$FILTER' for ${SECS}s (RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-<all cores>})…" >&2

# samply record runs the binary, samples it, then serves the interactive
# profile. --profile-time keeps criterion in plain-loop mode.
exec samply record -- "$BIN" --bench --profile-time "$SECS" "$FILTER"
