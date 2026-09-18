#!/usr/bin/env bash
# Canonical runner for the Poseidon2 benchmark (plan §12).
#
# Two-phase protocol: stage an immutable run-config.json (emitted by the
# crate-side poseidon_bench_config helper — this script never interprets
# benchmark flags or backend defaults itself), hash it, move the staging
# directory to its run id, write manifest.running.json, launch the
# benchmark with the config handshake, and on success finalize
# manifest.json with artifact hashes. On failure manifest.failed.json is
# retained; an interrupted run keeps a failed/running manifest rather
# than being labelled complete.
#
# Ordinary run output stays under target/poseidon-bench/ (ignored);
# publication into bench-results/ is a separate explicit step
# (scripts/publish_poseidon_bench.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCH_ROOT="target/poseidon-bench"
RUN_DIR=""
STATUS_WRITTEN=0

finalize_failure() {
  # Exit trap: tee cannot mask the benchmark's status (pipefail), and an
  # interrupted run retains a failed manifest.
  local code=$?
  if [[ $code -ne 0 && -n "$RUN_DIR" && -d "$RUN_DIR" && $STATUS_WRITTEN -eq 0 ]]; then
    python3 - "$RUN_DIR" "$code" <<'EOF'
import json, sys, time
run_dir, code = sys.argv[1], int(sys.argv[2])
manifest = {
    "status": "failed",
    "exit_code": code,
    "publishable": False,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(f"{run_dir}/manifest.failed.json", "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
EOF
    echo "run_poseidon_bench: FAILED (exit $code); manifest.failed.json written in $RUN_DIR" >&2
  fi
}
trap finalize_failure EXIT

echo "== building the config helper and benchmark =="
cargo build --release --bin poseidon_bench_config
cargo bench --bench poseidon_modp --no-run
HELPER="target/release/poseidon_bench_config"

# Phase 1: immutable config in a staging directory. The helper — not
# shell — enforces the default clean-tree gate via the shared parsed
# POSEIDON_ALLOW_DIRTY value.
mkdir -p "$BENCH_ROOT"
STAGING="$(mktemp -d "$BENCH_ROOT/staging.XXXXXX")"
"$HELPER" > "$STAGING/run-config.json"

# Phase 2: hash the config's exact bytes (the JSON cannot contain its own
# hash), pick the run id, and move the staging directory into place.
CONFIG_SHA="$(shasum -a 256 "$STAGING/run-config.json" | cut -d' ' -f1)"
echo "$CONFIG_SHA" > "$STAGING/run-config.sha256"
CONFIG12="${CONFIG_SHA:0:12}"
START_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${START_UTC}-cfg-${CONFIG12}"
RUN_DIR="$BENCH_ROOT/$RUN_ID"
mv "$STAGING" "$RUN_DIR"
echo "== run id: $RUN_ID =="

# Phase 3: running manifest, immutability re-check, then launch with the
# handshake. The benchmark re-parses the environment, recomputes the file
# hash, and byte-compares its canonical protocol subsection.
python3 - "$RUN_DIR" "$CONFIG_SHA" "$START_UTC" <<'EOF'
import json, sys
run_dir, sha, start = sys.argv[1:4]
with open(f"{run_dir}/manifest.running.json", "w") as f:
    json.dump({"status": "running", "config_sha256": sha, "started_utc": start},
              f, indent=2, sort_keys=True)
    f.write("\n")
EOF
"$HELPER" --check "$RUN_DIR/run-config.json"

mkdir -p "$RUN_DIR/criterion"
set +e
POSEIDON_RUN_DIR="$REPO_ROOT/$RUN_DIR" \
POSEIDON_CONFIG_PATH="$REPO_ROOT/$RUN_DIR/run-config.json" \
POSEIDON_CONFIG_SHA256="$CONFIG_SHA" \
  cargo bench --bench poseidon_modp 2>&1 | tee "$RUN_DIR/bench.log"
BENCH_CODE=${PIPESTATUS[0]}
set -e
if [[ $BENCH_CODE -ne 0 ]]; then
  echo "run_poseidon_bench: benchmark exited with $BENCH_CODE" >&2
  exit "$BENCH_CODE"
fi

# Phase 4: post-run validation. Re-check the immutable config to catch
# source/toolchain changes during the measurement, require the expected
# audit entries, run the K-sweep summarizer when applicable, hash every
# artifact, and atomically write the complete manifest.
"$HELPER" --check "$RUN_DIR/run-config.json"

python3 - "$RUN_DIR" <<'EOF'
import json, sys
run_dir = sys.argv[1]
with open(f"{run_dir}/run-config.json") as f:
    cfg = json.load(f)
proto = cfg["protocol"]
mode, backend = proto["mode"], proto["backend"]
if mode == "proof_size":
    with open(f"{run_dir}/proof-size.json") as f:
        ps = json.load(f)
    # One combined-circuit block, never three fictitious per-field proofs.
    for key in ("input_commitments", "eval_arg", "sumcheck_remainder"):
        if not isinstance(ps["combined"].get(key), int):
            raise SystemExit(f"proof-size.json combined block is missing {key}")
else:
    with open(f"{run_dir}/cache-audit.json") as f:
        audit = json.load(f)
    if backend == "brakedown":
        want = (["commit_witness/", "prove_after_input_commit/", "prove_e2e/"]
                if mode == "normal" else ["prove_e2e/"])
        for prefix in want:
            if not any(k.startswith(prefix) for k in audit):
                raise SystemExit(f"cache-audit.json has no {prefix} entries")
if mode == "ksweep":
    with open(f"{run_dir}/ksweep-metadata.json"):
        pass
print("post-run sidecar validation OK")
EOF

with_mode="$(python3 -c "import json;print(json.load(open('$RUN_DIR/run-config.json'))['protocol']['mode'])")"
if [[ "$with_mode" == "ksweep" ]]; then
  echo "== summarizing the k sweep =="
  python3 scripts/summarize_poseidon_ksweep.py "$RUN_DIR"
fi

python3 - "$RUN_DIR" "$CONFIG_SHA" "$START_UTC" <<'EOF'
import hashlib, json, os, sys, time
run_dir, sha, start = sys.argv[1:4]
with open(f"{run_dir}/run-config.json") as f:
    cfg = json.load(f)
artifacts = {}
for root, _dirs, files in os.walk(run_dir):
    for name in files:
        path = os.path.join(root, name)
        rel = os.path.relpath(path, run_dir)
        if rel.startswith("manifest"):
            continue
        with open(path, "rb") as f:
            artifacts[rel] = hashlib.sha256(f.read()).hexdigest()
audit = None
if os.path.exists(f"{run_dir}/cache-audit.json"):
    with open(f"{run_dir}/cache-audit.json") as f:
        audit = json.load(f)
manifest = {
    "status": "complete",
    "run_id": os.path.basename(run_dir),
    "config_sha256": sha,
    "publishable": cfg["environment"]["publishable"],
    "started_utc": start,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "exit_code": 0,
    "cache_audit": audit,
    "artifacts": artifacts,
}
tmp = f"{run_dir}/manifest.json.tmp"
with open(tmp, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, f"{run_dir}/manifest.json")
os.remove(f"{run_dir}/manifest.running.json")
print(f"manifest.json written (publishable={manifest['publishable']})")
EOF
STATUS_WRITTEN=1
echo "== complete: $RUN_DIR =="
