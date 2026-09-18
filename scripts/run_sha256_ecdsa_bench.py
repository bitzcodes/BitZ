#!/usr/bin/env python3
"""Persist a fresh-process SHA-256/ECDSA campaign, including failed configurations."""
from bench_support import source_metadata, file_hash, run_process, address_space_limit, write_json
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import time


def write_summary(directory):
    """Keep a compact comparison beside the full proof and phase records."""
    fields = ["exponent", "mode", "target", "threads", "status", "samples",
              "compressions", "message_bytes", "witness_ms", "prove_ms",
              "verify_ms", "proof_bytes", "peak_rss_bytes",
              "outer_ms", "shared_inner_ms"]
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        records = [json.loads(path.read_text()) for path in directory.glob("*.result.json")]
        records.sort(key=lambda r: (r["exponent"], r["target"], r["threads"], r["mode"]))
        for result in records:
            row = {key: result[key] for key in fields[:5]}
            samples = [r for r in result["rows"] if r["trial"] == "sample"]
            row.update(samples=len(samples), peak_rss_bytes=result["peak_rss_bytes"])
            if samples:
                for key in fields[6:12]:
                    row[key] = statistics.median(r[key] for r in samples)
                for key, phase in [("outer_ms", "ecdsa:outer_prove"),
                                   ("shared_inner_ms", "ecdsa:shared_inner_prove")]:
                    row[key] = 1000 * statistics.median(
                        dict(r["prove_phases_seconds"])[phase] for r in samples)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exponents", type=int, nargs="+", default=list(range(3, 17)))
    parser.add_argument("--targets", type=int, nargs="+", choices=[100, 128], default=[100, 128])
    parser.add_argument("--threads", type=int, nargs="+", default=sorted({1, os.cpu_count() or 1}))
    parser.add_argument("--modes", nargs="+", choices=["split", "all"], default=["split", "all"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--memory-gib", type=int, default=48)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if (args.reps < 1 or args.memory_gib < 1 or args.timeout <= 0
            or any(n < 1 for n in args.threads)
            or any(i not in range(3, 17) for i in args.exponents)):
        parser.error("positive reps/threads and exponents 3..16 required")
    args.output.mkdir(parents=True, exist_ok=True)
    binary_hash = file_hash(binary)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["binary_sha256"] != binary_hash:
        parser.error("output contains results from a different binary; choose a new output directory")
    if not manifest_path.exists():
        root = Path(__file__).resolve().parents[1]
        fingerprint = (binary.parent.parent / ".fingerprint"
                       / binary.name.replace("sha256_ecdsa-", "bitz-", 1)
                       / "test-bench-sha256_ecdsa.json")
        build = json.loads(fingerprint.read_text()) if fingerprint.exists() else {}
        build = {key: build[key] for key in ["rustc", "features", "rustflags", "compile_kind"] if key in build}
        write_json(manifest_path, dict(binary=str(binary), binary_sha256=binary_hash,
                                       **source_metadata(root), build=build))
    failed = False
    for exponent in args.exponents:
        for target in args.targets:
            for threads in args.threads:
                for mode in args.modes:
                    name = f"n{exponent}-{mode}-s{target}-t{threads}-r{args.reps}"
                    result_path = args.output / f"{name}.result.json"
                    if result_path.exists() and (not args.retry_failed or json.loads(result_path.read_text())["status"] == "complete"):
                        failed |= json.loads(result_path.read_text())["status"] != "complete"
                        print(f"skip recorded {name}", flush=True)
                        continue
                    command = [str(binary), str(exponent), mode, str(target), str(args.reps)]
                    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
                    started = time.monotonic()
                    # Linux time reports this worker's peak RSS, not a cumulative
                    # high-water mark inherited from a previous configuration.
                    rss_path = args.output / f"{name}.rss-kib"
                    if Path("/usr/bin/time").exists():
                        command = ["/usr/bin/time", "-f", "%M", "-o", str(rss_path), *command]
                    status = "failed"
                    returncode = None
                    with (args.output / f"{name}.stdout").open("w") as stdout, (args.output / f"{name}.stderr").open("w") as stderr:
                        returncode, timed_out = run_process(command, env=env, stdout=stdout, stderr=stderr,
                                                            preexec_fn=address_space_limit(args.memory_gib), timeout=args.timeout)
                        if timed_out:
                            status = "timeout"
                    rows = []
                    for line in (args.output / f"{name}.stdout").read_text().splitlines():
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(row, dict) and row.get("schema") == "bitz/sha256-ecdsa/v2":
                            from ligerito_results import validate_ligerito
                            validate_ligerito(row.get("ligerito"), target)
                            rows.append(row)
                    samples = [r for r in rows if r.get("trial") == "sample"]
                    if returncode == 0 and len(samples) == args.reps and all(r.get("verified") for r in rows):
                        status = "complete"
                    rss = None
                    if rss_path.exists():
                        lines = rss_path.read_text().splitlines()
                        if lines and lines[-1].isdigit():
                            rss = int(lines[-1]) * 1024
                    result = dict(status=status, returncode=returncode, seconds=time.monotonic()-started,
                                  exponent=exponent, mode=mode, target=target, threads=threads,
                                  peak_rss_bytes=rss, rows=rows)
                    write_json(result_path, result)
                    failed |= status != "complete"
                    write_summary(args.output)
                    print(f"{name}: {status} ({result['seconds']:.1f}s)", flush=True)
    write_summary(args.output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
