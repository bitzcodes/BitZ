#!/usr/bin/env bash
# Explicit publication step for completed Poseidon2 benchmark runs
# (plan §12): validates every input run directory, enforces the
# combined-table compatibility gate, then copies immutable
# configs/manifests/logs/raw Criterion data and generated tables into
# bench-results/poseidon2/<run-id>/ and updates the README section.
#
# Usage: scripts/publish_poseidon_bench.sh <run-dir> [<run-dir> ...]
#
# Ordinary run output must never be written to bench-results/ directly —
# that directory is a committed publication target, and untracked output
# there would dirty the tree and fail the next canonical run's clean-tree
# gate. Do not run another canonical benchmark after publication dirties
# the tree unless these artifacts are committed first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <completed-run-dir> [<completed-run-dir> ...]" >&2
  exit 2
fi

python3 - "$@" <<'EOF'
import hashlib, json, os, shutil, sys

run_dirs = sys.argv[1:]

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Validate every input run directory; abort on the first failure.
manifests, configs = {}, {}
for run_dir in run_dirs:
    mpath = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"{run_dir}: no manifest.json (incomplete run)")
    with open(mpath) as f:
        manifest = json.load(f)
    if manifest.get("status") != "complete":
        raise SystemExit(f"{run_dir}: manifest status is not complete")
    if not manifest.get("publishable"):
        raise SystemExit(f"{run_dir}: manifest is marked non-publishable")
    cfg_path = os.path.join(run_dir, "run-config.json")
    cfg_sha = sha256_file(cfg_path)
    with open(os.path.join(run_dir, "run-config.sha256")) as f:
        recorded = f.read().strip()
    if cfg_sha != recorded:
        raise SystemExit(f"{run_dir}: run-config.json does not re-hash to run-config.sha256")
    if cfg_sha != manifest["config_sha256"]:
        raise SystemExit(f"{run_dir}: run-config hash differs from the manifest's")
    for rel, expected in manifest["artifacts"].items():
        path = os.path.join(run_dir, rel)
        if not os.path.exists(path):
            raise SystemExit(f"{run_dir}: artifact {rel} is missing")
        if sha256_file(path) != expected:
            raise SystemExit(f"{run_dir}: artifact {rel} does not re-hash to its manifest value")
    with open(cfg_path) as f:
        configs[run_dir] = json.load(f)
    manifests[run_dir] = manifest

# Compatibility gate for a combined Hyrax/Brakedown table: every
# comparison-controlled field must agree across the published set; only
# backend, backend-specific k, mode-specific registration order, and the
# derived config hashes may differ.
def controlled(cfg):
    env, proto = cfg["environment"], cfg["protocol"]
    return {
        "git_sha": env["git_sha"],
        "cargo_lock_sha256": env["cargo_lock_sha256"],
        "cpu_model": env["cpu_model"],
        "os": env["os"], "kernel": env["kernel"], "arch": env["arch"],
        "rustc_vV": env["rustc_vV"], "cargo_version": env["cargo_version"],
        "criterion_version": env["criterion_version"],
        "allocator": env["allocator"], "jem_feature": env["jem_feature"],
        "rayon_num_threads": env["rayon_num_threads"],
        "rustflags": env["rustflags"],
        "target_features": env["target_features"],
        "hashes_per_field": proto["hashes_per_field"],
        "total_hashes": proto["total_hashes"],
        "circuit": proto["circuit"],
        "dims": proto["dims"],
        "num_io": proto["num_io"],
        "field_blocks": proto["field_blocks"],
        "criterion": proto["criterion"],
        "allow_knobs": proto["allow_knobs"], "knobs": proto["knobs"],
        "workload": proto["workload"],
        "log_t_f": proto["log_t_f"], "log_t": proto["log_t"],
    }

if len(run_dirs) > 1:
    base = controlled(configs[run_dirs[0]])
    for run_dir in run_dirs[1:]:
        other = controlled(configs[run_dir])
        diffs = [k for k in base if base[k] != other[k]]
        if diffs:
            raise SystemExit(
                f"compatibility gate: {run_dirs[0]} and {run_dir} differ on "
                f"comparison-controlled fields {diffs}")

# Copy into the publication target.
published = []
for run_dir in run_dirs:
    run_id = os.path.basename(os.path.normpath(run_dir))
    dest = os.path.join("bench-results", "poseidon2", run_id)
    if os.path.exists(dest):
        raise SystemExit(f"{dest} already exists; refusing to overwrite a publication")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(run_dir, dest)
    published.append((run_id, configs[run_dir], manifests[run_dir]))
    print(f"published {run_dir} -> {dest}")

# Update the README section between the markers.
BEGIN = "<!-- poseidon2-bench:begin -->"
END = "<!-- poseidon2-bench:end -->"
lines = [BEGIN, "## Poseidon2 non-native-field benchmark", ""]
lines.append(
    "Thirty Poseidon2 compressions (t = 3, α = 5, R_F = 8, R_P = 56) proven "
    "in ONE mixed-modulus circuit: three independent ten-compression chains — "
    "one per field block, BN254-Fr, BLS12-381-Fr, secp256k1-Fr, in that fixed "
    "order — each restarting from the same fixed IV and ending at its own "
    "ordered public digest (num_io = 3; 12,990 real rows padded once to "
    "2^14 × 2^14). **This permutation is a benchmark workload, not a "
    "security-reviewed production hash** (custom BLAKE3-derived constants). "
    "**No zero-knowledge claim is made for this driver**: Hyrax commitments "
    "are hiding, Brakedown commitments are not, and the sumcheck transcript "
    "carries unmasked witness-dependent data regardless of backend; "
    "\"messages are private\" means *not public IO*, not confidential. "
    "Verification must go through "
    "`limber::poseidon2::verify_poseidon_chain` — bypassing it forfeits the "
    "three-digest canonicality guarantee. Published Brakedown timings are "
    "layout-warm steady state with an empty retained cache per measured "
    "sample.")
lines.append("")
lines.append("| run id | backend | mode | H/field (total) | k | git | config |")
lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
for run_id, cfg, manifest in published:
    proto = cfg["protocol"]
    lines.append(
        f"| [`{run_id}`](bench-results/poseidon2/{run_id}/) "
        f"| {proto['backend']} | {proto['mode']} "
        f"| {proto['hashes_per_field']} ({proto['total_hashes']}) "
        f"| {proto['k']} | `{cfg['environment']['git_sha'][:12]}` "
        f"| `cfg-{manifest['config_sha256'][:12]}` |")
lines.append("")
lines.append(
    "Raw Criterion data, immutable run configs, manifests, and proof-size / "
    "k-sweep sidecars live in each run directory. Reproduce with the exact "
    "commands in `scripts/run_poseidon_bench.sh` (see plan §12).")
lines.append(END)
section = "\n".join(lines) + "\n"

with open("README.md", encoding="utf-8") as f:
    readme = f.read()
if BEGIN in readme and END in readme:
    head, rest = readme.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    readme = head + section.rstrip("\n") + tail
else:
    readme = readme.rstrip("\n") + "\n\n" + section
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("README.md section updated")
EOF
