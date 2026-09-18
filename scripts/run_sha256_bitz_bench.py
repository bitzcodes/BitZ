#!/usr/bin/env python3
"""Build and measure only the two BitZ SHA-256 workloads."""

from __future__ import annotations

import argparse
from local_provenance import root_metadata, vendor_snapshots
import json
import os
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from ligerito_results import validate_result_fields
from bench_support import command_text, cpu_name, write_csv, run_logged, cargo_executables, environment


ROOT = Path(__file__).resolve().parents[1]
BENCHES = {"compressions": "sha256_compressions", "chain": "sha256_chain"}


def fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=[*BENCHES, "both"], default="both")
    parser.add_argument("--shapes", type=int, nargs="+", default=[7, 8, 10], metavar="K")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--lambda", dest="security", choices=["100", "128", "sha128-reference-schedule"], default="100")
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x46325A5F53484132)
    parser.add_argument("--checked", action="store_true", help="omit the usual unchecked feature")
    parser.add_argument("--offline", action="store_true", help="require Cargo's cached dependencies")
    parser.add_argument("--out-dir", type=Path, help="new directory; an existing path is rejected")
    args = parser.parse_args()
    if args.threads < 1 or args.reps < 1:
        parser.error("--threads and --reps must be positive")
    min_shape = 7
    if any(shape not in range(min_shape, 17) for shape in args.shapes):
        parser.error(f"--shapes for {args.workload} must contain exponents in {min_shape}..16")
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("--shapes must not contain duplicates")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must fit in an unsigned 64-bit integer")
    return args


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    out = (args.out_dir or ROOT / "bench_results" / started.strftime("sha256-bitz-%Y%m%dT%H%M%S.%fZ")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    # Select the standard workload/layout despite aliases or a previous trace run.
    removed = {
        key: env.pop(key) for key in list(env)
        if key.startswith("BITZ_SHA_") or key in {"BITZ_BENCH_SHAPES", "BITZ_BENCH_PASS", "OBLONG_PROFILE_INTERVALS"}
    }
    env.update(RAYON_NUM_THREADS=str(args.threads), BITZ_BENCH_REPS=str(args.reps),
               BITZ_BENCH_SEED=hex(args.seed), BITZ_BENCH_LAMBDA=args.security)
    features = ["span-metrics"] if args.checked else ["unchecked", "span-metrics"]
    build = ["cargo", "bench", "--locked", "--no-run", "--message-format=json-render-diagnostics"]
    if args.offline:
        build.append("--offline")
    for bench in BENCHES.values():
        build.extend(["--bench", bench])
    if features:
        build.extend(["--features", ",".join(features)])
    source = root_metadata(ROOT)
    metadata = {
        "started_utc": started.isoformat(), "status": "running", "git_revision": source["revision"],
        "git_status": source["git_status"], "vendor_snapshots": vendor_snapshots(ROOT),
        "rustc": command_text("rustc", "-Vv"), "cargo": command_text("cargo", "-V"),
        "cpu": cpu_name(), "os": platform.platform(), "architecture": platform.machine(),
        "workload": args.workload, "shapes": args.shapes, "threads": args.threads,
        "reps": args.reps, "warmups": 1, "lambda": args.security, "seed": hex(args.seed),
        "features": features, "default_features": True, "build_command": build,
        "environment": environment(env),
        "cleared_environment": removed, "runs": [],
    }
    metadata_path = out / "metadata.json"
    summaries: list[dict[str, str]] = []
    samples: list[dict[str, str]] = []
    try:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        build_log = run_logged(build, env, out / "build.log")
        executables = cargo_executables(build_log, BENCHES.values())
        workloads = list(BENCHES) if args.workload == "both" else [args.workload]
        for workload in workloads:
            bench = BENCHES[workload]
            for shape in args.shapes:
                run_id = f"{workload}-2p{shape}"
                run_env = dict(env, BITZ_BENCH_SHAPES=str(shape))
                log = run_logged([executables[bench]], run_env, out / f"{run_id}.log")
                results = [fields(line) for line in log.splitlines() if line.startswith("RESULT ")]
                raw = [fields(line.strip()) for line in log.splitlines() if line.strip().startswith("SAMPLE ")]
                if len(results) != 1 or results[0].get("verified_samples") != str(args.reps):
                    raise RuntimeError(f"{run_id}: expected one RESULT with {args.reps} verified samples")
                if len(raw) != args.reps or any(row.get("verified") != "true" for row in raw):
                    raise RuntimeError(f"{run_id}: expected {args.reps} verified SAMPLE rows")
                validate_result_fields(results[0])
                row = dict(run_id=run_id, workload=workload, **results[0])
                row["witness_to_proof_ms"] = f"{statistics.median(float(sample['witness_ms']) + float(sample['prove_ms']) for sample in raw):.6f}"
                summaries.append(row)
                samples.extend(dict(run_id=run_id, workload=workload, **sample) for sample in raw)
                metadata["runs"].append({"run_id": run_id, "executable": executables[bench], "shape": shape})
                write_csv(out / "summary.csv", summaries)
                write_csv(out / "samples.csv", samples)
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        metadata["status"] = "complete"
    except (OSError, ValueError, RuntimeError) as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        print(f"Benchmark failed: {error}\nPartial results: {out}", flush=True)
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results: {out}", flush=True)
    return 0 if metadata["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
