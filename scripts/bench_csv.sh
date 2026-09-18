#!/bin/bash
# Historical human-output CSV helper (no production identity validation).
# Use the versioned RESULT/LIGERITO_CONFIG output for new production tables.
# Run the bitz PCS bench suite (benches/pcs.rs) one shape per process — the
# measurement protocol — and write one CSV row per (profile, shape).
#
# Usage:
#   scripts/bench_csv.sh [-o out.csv] [-p profiles] [-s "t:s:W ..."]
#                        [-r reps] [-j threads] [-g gap_seconds] [--big] [--phases]
#
#   -o        output CSV (default bench_results/bitz-<timestamp>.csv)
#   -p        comma-separated profile list for BITZ_LIG_PROFILE
#             (default: "" = the bench default; e.g. "fast,slim,slim3,custom:4:4")
#   -s        space-separated t:s:W shapes (default: the validated n=20..28 list)
#   -r        timing reps per shape (default 3)
#   -j        RAYON_NUM_THREADS (default: all cores; 1 = single-threaded)
#   -g        cooldown seconds between shapes (default 20; use 45-90 for big shapes)
#   --big     append the n=30/31/32 reference shapes (31/32 under
#             F2_FOREST_SCHEDULE=l8); check `vm_stat` free pages first — the
#             5-12 GB shapes are only quotable from a memory-healthy box
#   --phases  set OBLONG_PROFILE=1 → forest/open phase columns (adds ~µs-scale
#             scope overhead to the timed medians; leave off for headline runs)
#
# CSV columns:
#   timestamp,profile_arg,lig_geometry,n,t,s,W,chunks,threads,reps,
#   commit_ms,commit_peak_mb,forest_ms,open_ms,prove_ms,prove_peak_mb,
#   verify_ms,proof_bytes,forest_side_kib,s_v_kib,lig_kib,serialize_us,deserialize_us
# (forest_ms/open_ms empty unless --phases; every prove is verified by the bench.)
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=""
PROFILES=""
SHAPES="13:7:1 14:8:1 15:9:1 16:10:1 17:11:1"
REPS=3
THREADS=""
GAP=20
BIG=0
PHASES=0
while [ $# -gt 0 ]; do
  case "$1" in
    -o) OUT="$2"; shift 2 ;;
    -p) PROFILES="$2"; shift 2 ;;
    -s) SHAPES="$2"; shift 2 ;;
    -r) REPS="$2"; shift 2 ;;
    -j) THREADS="$2"; shift 2 ;;
    -g) GAP="$2"; shift 2 ;;
    --big) BIG=1; shift ;;
    --phases) PHASES=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ "$BIG" = 1 ] && SHAPES="$SHAPES 18:12:1 19:12:1@l8 19:13:1@l8"
[ -z "$OUT" ] && OUT="bench_results/bitz-$(date '+%Y%m%d-%H%M%S').csv"
mkdir -p "$(dirname "$OUT")"

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
[ "$PHASES" = 1 ] && export OBLONG_PROFILE=1
[ -n "$THREADS" ] && export RAYON_NUM_THREADS="$THREADS"

# Build once up front so per-shape runs are measurement-only.
cargo bench --bench pcs --features unchecked,span-metrics --no-run >/dev/null 2>&1 || {
  echo "build failed" >&2; exit 1; }

echo "timestamp,profile_arg,lig_geometry,n,t,s,W,chunks,threads,reps,commit_ms,commit_peak_mb,forest_ms,open_ms,prove_ms,prove_peak_mb,verify_ms,proof_bytes,forest_side_kib,s_v_kib,lig_kib,serialize_us,deserialize_us" > "$OUT"

threads_label="${RAYON_NUM_THREADS:-all}"
IFS=',' read -ra PROFS <<< "${PROFILES:-__default__}"
for prof in "${PROFS[@]}"; do
  penv=""
  plabel="default"
  if [ "$prof" != "__default__" ]; then penv="$prof"; plabel="$prof"; fi
  for shape in $SHAPES; do
    sched=""
    s="$shape"
    case "$shape" in *@l8) sched="l8"; s="${shape%@l8}" ;; esac
    echo ">> profile=$plabel shape=$s ${sched:+sched=l8}" >&2
    out=$(env ${penv:+BITZ_LIG_PROFILE="$penv"} ${sched:+F2_FOREST_SCHEDULE=l8} \
      BITZ_BENCH_SHAPES="$s" BITZ_BENCH_REPS="$REPS" \
      cargo bench --bench pcs --features unchecked,span-metrics 2>/dev/null)
    echo "$out" | awk -v ts="$(date '+%Y-%m-%dT%H:%M:%S')" -v prof="$plabel" \
        -v threads="$threads_label" -v reps="$REPS" '
      function num(x) { gsub(/[^0-9.]/, "", x); return x }
      # === n=28 (t=17, s=11, W=1, m_p=21, chunks=1, lig=slim@r1/4k4, data=... ===
      /^=== n=/ {
        n = num($2); t = num($3); sv = num($4); w = num($5); ch = num($7)
        geo = $8; sub(/^lig=/, "", geo); sub(/,$/, "", geo)
      }
      # "  commit:     43.24 ms   peak   260.16 MB   live-after ..."
      /^  commit:/ { cm = $2; cpk = $5 }
      /^  prove:/  { pm = $2; ppk = $5 }
      # "  phases:  forest+presum   42.45 ms | ligerito open   21.45 ms ..."
      /^  phases:/ { fm = $3; om = $8 }
      /^  verify:/ { vm = $2 }
      # "  proof:  182596 B (178.3 KiB)  serialize 68 µs / deserialize 54 µs"
      /^  proof:/ {
        pb = $2; ser = ""; de = ""
        for (i = 1; i <= NF; i++) {
          if ($i == "serialize") ser = $(i + 1)
          if ($i == "deserialize") de = $(i + 1)
        }
      }
      # "  split:  forest-side  50.7 KiB | open-side  125.2 KiB (s_v  2.0 + lig  123.2)"
      /^  split:/ {
        fs = $3; sv2 = ""; lg = ""
        for (i = 1; i <= NF; i++) {
          if ($i == "(s_v") sv2 = $(i + 1)
          if ($i == "lig") lg = num($(i + 1))
        }
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
          ts, prof, geo, n, t, sv, w, ch, threads, reps,
          cm, cpk, fm, om, pm, ppk, vm, pb, fs, sv2, lg, ser, de
        fm = ""; om = ""
      }
    ' >> "$OUT"
    sleep "$GAP"
  done
done
echo "wrote $OUT" >&2
n_rows=$(($(wc -l < "$OUT") - 1))
echo "$n_rows data rows" >&2
