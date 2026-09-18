#!/usr/bin/env python3
"""Compare BitZ, Binius64 and Binius64-Ligerito on identical raw SHA-256 chains."""
from __future__ import annotations

import argparse
from local_provenance import root_metadata, source_patch, vendor_snapshots
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

from ligerito_results import validate_ligerito
from run_sha256_ecdsa_compare import address_space_limit, peak_rss_bytes
from run_sha256_bitz_bench import command_text, cpu_name

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bitz/sha256-chain-compare/v1"
FIXTURE = "sha256-chain/public-blocks-standard-iv/v1"
METHODS = ("bitz", "binius64", "binius64-ligerito")
METRICS = ("setup_ms", "witness_ms", "commit_ms", "prove_ms", "e2e_prover_ms", "verify_ms", "proof_bytes")
CASE_FIELDS = ("method", "log_compressions", "threads", "seed", "ligerito_profile", "log_inv_rate")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--exponents", nargs="+", type=int, choices=range(7, 17), default=list(range(7, 17)))
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bitz-profiles", nargs="+", choices=["custom:1:4", "custom:3:4"],
                        default=["custom:1:4", "custom:3:4"])
    parser.add_argument("--binius-rates", nargs="+", type=int, choices=[1, 3], default=[1, 3])
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per worker, including setup and warmup")
    parser.add_argument("--memory-gib", type=int, default=48, help="worker address-space limit on Linux; unenforced on macOS")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--binary", type=Path, help="use an already compiled sha256_chain_compare worker")
    parser.add_argument("--dry-run", action="store_true", help="print all cases without building, proving or writing files")
    args = parser.parse_args(argv)
    for name in ("methods", "exponents", "threads", "bitz_profiles", "binius_rates"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if min(args.threads) < 1 or args.reps < 1 or args.memory_gib < 1:
        parser.error("threads, reps and memory-gib must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 <= args.seed < 2**64:
        parser.error("require a positive finite timeout and an unsigned 64-bit seed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    args.output = (args.output or ROOT / "bench_results" / f"sha256-chain-{stamp}").resolve()
    return args


def cases(args):
    for exponent, threads, method in itertools.product(args.exponents, args.threads, args.methods):
        profiles = args.bitz_profiles if method == "bitz" else [None]
        for profile in profiles:
            rates = [int(profile.split(":")[1])] if profile else args.binius_rates
            for rate in rates:
                yield dict(method=method, log_compressions=exponent, threads=threads, seed=args.seed,
                           ligerito_profile=profile, log_inv_rate=rate)


def worker_command(binary, case, reps):
    command = [str(binary), "--method", case["method"], "--exponent", str(case["log_compressions"]),
               "--threads", str(case["threads"]), "--reps", str(reps), "--seed", str(case["seed"]),
               "--log-inv-rate", str(case["log_inv_rate"])]
    if case["ligerito_profile"]:
        command += ["--bitz-profile", case["ligerito_profile"]]
    return command


def environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("BITZ_", "F2_FOREST", "OBLONG_", "RAYON_")) and k != "CARGO_ENCODED_RUSTFLAGS"}
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    local = ROOT / ".tools/perfetto/trace_processor_shell"
    if "PERFETTO_TRACE_PROCESSOR" not in env and local.is_file():
        env["PERFETTO_TRACE_PROCESSOR"] = str(local)
    return env


def build(args, env):
    command = ["cargo", "build", "--release", "--locked", "--features", "unchecked,span-metrics,binius64-bench",
               "--bench", "sha256_chain_compare", "--message-format=json-render-diagnostics"]
    if args.offline:
        command.append("--offline")
    print(f"Building worker; log: {args.output / 'build.log'}", flush=True)
    with (args.output / "build.jsonl").open("w") as stdout, (args.output / "build.log").open("w") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
    for line in reversed((args.output / "build.jsonl").read_text().splitlines()):
        try:
            artifact = json.loads(line)
        except ValueError:
            continue
        if artifact.get("target", {}).get("name") == "sha256_chain_compare" and artifact.get("executable"):
            return Path(artifact["executable"]).resolve()
    raise RuntimeError("Cargo did not report the chain comparison executable")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def validate_rows(rows, case, reps, fixture):
    if len(rows) != reps + 1:
        raise ValueError(f"expected one warmup and {reps} samples, got {len(rows)} rows")
    for i, row in enumerate(rows):
        if row.get("schema") != SCHEMA or any(row.get(k) != v for k, v in case.items()):
            raise ValueError("result schema/configuration mismatch")
        if any(row.get(k) != v for k, v in fixture.items()):
            raise ValueError("public chain fixture mismatch")
        if (row.get("verified") is not True or row.get("zk") is not False
                or row.get("sample") != i or row.get("trial") != ("warmup" if i == 0 else "sample")
                or row.get("security_target") != 100 or row.get("timing") != "perfetto"):
            raise ValueError("missing verified sample or incorrect measurement policy")
        if any(type(row.get(k)) not in (float, int) or not math.isfinite(row[k]) or row[k] < 0 for k in METRICS):
            raise ValueError("missing, negative or nonfinite metric")
        if (row["commit_ms"] > row["prove_ms"] + 0.005
                or row["witness_ms"] + row["prove_ms"] > row["e2e_prover_ms"] + 0.005
                or type(row["proof_bytes"]) is not int or row["proof_bytes"] <= 0):
            raise ValueError("inconsistent timing boundaries or proof size")
        security = row.get("security", {})
        size_kind = ("analytical-piop-plus-serialized-pcs-and-commitment" if case["method"] == "bitz"
                     else "serialized-proof-including-commitments")
        phases = row.get("phases_ms")
        if (not isinstance(security, dict) or row.get("proof_size_kind") != size_kind
                or not isinstance(phases, dict)
                or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in phases.values())):
            raise ValueError("missing or invalid security, size or phase metadata")
        if security != rows[0].get("security"):
            raise ValueError("security parameters changed between trials")
        if case["method"] == "bitz":
            identity = validate_ligerito(security.get("ligerito"), 100)
            if (identity["requested_profile"] != case["ligerito_profile"]
                    or identity["configuration"]["levels"][0]["log_inv_rate"] != case["log_inv_rate"]
                    or not math.isfinite(security.get("economic_bits", float("nan")))
                    or security["economic_bits"] < 100 - 1e-6):
                raise ValueError("BitZ profile/target mismatch")
        elif security.get("log_inv_rate") != case["log_inv_rate"]:
            raise ValueError("Binius rate mismatch")
        elif case["method"] == "binius64":
            if security.get("pcs") != "BaseFold" or security.get("fri_query_target_bits") != 100:
                raise ValueError("BaseFold target mismatch")
        elif (security.get("pcs") != "BitZ-Ligerito" or security.get("accounting") != "round-by-round"
              or security.get("target_bits") != 100
              or not math.isfinite(security.get("algebraic_bits", float("nan")))
              or security["algebraic_bits"] < 100 - 1e-6):
            raise ValueError("Binius-Ligerito accounting mismatch")


def stop_worker(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_case(args, binary, case, fixture, env):
    name = f"{case['method']}-i{case['log_compressions']}-rate{case['log_inv_rate']}-t{case['threads']}"
    command = worker_command(binary, case, args.reps)
    rss_file, stderr_file = args.output / f"{name}.rss-kib", args.output / f"{name}.stderr"
    stdout_file = args.output / f"{name}.stdout"
    if sys.platform == "darwin":
        command = ["/usr/bin/time", "-l", *command]
    elif sys.platform.startswith("linux"):
        command = ["/usr/bin/time", "-f", "%M", "-o", str(rss_file), *command]
    else:
        raise RuntimeError("peak RSS collection supports macOS and Linux")
    result = dict(case=case, command=command, status="failed", rows=[], peak_rss_bytes=None, returncode=None)
    print(f"Running {name}; log: {stdout_file}", flush=True)
    started = time.monotonic()
    interrupted = False
    try:
        with stdout_file.open("w") as stdout, stderr_file.open("w") as stderr:
            with subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True,
                                  env=env | {"RAYON_NUM_THREADS":str(case["threads"]),
                                             "HARDWARE_CONCURRENCY":str(case["threads"])},
                                  preexec_fn=address_space_limit(args.memory_gib)) as process:
                try:
                    result["returncode"] = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    stop_worker(process)
                    result["status"] = "timeout"
                except KeyboardInterrupt:
                    stop_worker(process)
                    interrupted = True
                    result["status"] = "interrupted"
        result["peak_rss_bytes"] = peak_rss_bytes(rss_file, stderr_file)
        for line in stdout_file.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("schema") == SCHEMA:
                result["rows"].append(row)
        if result["returncode"] == 0:
            validate_rows(result["rows"], case, args.reps, fixture)
            if not result["peak_rss_bytes"] or result["peak_rss_bytes"] < 0:
                raise ValueError("missing peak RSS measurement")
            result["status"] = "complete"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output / f"{name}.result.json", result)
    print(f"{name}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return result


def summarize(directory, results):
    fields = [*CASE_FIELDS, "status", "samples", "peak_rss_bytes", "fixture_id", "proof_size_kind", *METRICS]
    with (directory / "summary.csv").open("w", newline="") as stream, \
            (directory / "samples.csv").open("w", newline="") as sample_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        sample_writer = csv.DictWriter(sample_stream, fieldnames=[*CASE_FIELDS, "trial", "sample", "verified", "fixture_id", *METRICS])
        writer.writeheader()
        sample_writer.writeheader()
        for result in results:
            row = dict(result["case"], status=result["status"], samples=0, peak_rss_bytes=result["peak_rss_bytes"])
            for sample in result["rows"]:
                sample_writer.writerow({k:sample.get(k) for k in sample_writer.fieldnames})
            if result["status"] == "complete":
                samples = [s for s in result["rows"] if s["trial"] == "sample"]
                row.update(samples=len(samples), fixture_id=samples[0]["fixture_id"], proof_size_kind=samples[0]["proof_size_kind"])
                row.update({k:statistics.median(s[k] for s in samples) for k in METRICS})
            writer.writerow(row)


def main(argv=None):
    args = parse_args(argv)
    planned = list(cases(args))
    print(f"SHA-256 chains: {len(planned)} configurations; one warmup + {args.reps} measured proofs each.")
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    env = environment()
    results = []
    manifest = dict(schema=SCHEMA, status="building", cases=planned, reps=args.reps, warmups=1,
                    platform=platform.platform(), cpu=cpu_name(), rustflags=env["RUSTFLAGS"],
                    rustc=command_text("rustc", "-Vv"), cargo=command_text("cargo", "-V"),
                    features=["unchecked", "span-metrics", "binius64-bench"],
                    trace_processor=env.get("PERFETTO_TRACE_PROCESSOR", "trace_processor_shell"),
                    security_note="100-bit BitZ economic, Binius FRI query, and Binius-Ligerito per-round targets use different accounting; inspect each row.",
                    memory_note="Whole-worker peak RSS includes fixture construction, setup, warmup, proofs and verification; excludes compilation.",
                    timing_note="Prove includes commitment. E2E runs from witness generation through proof completion, excluding reusable setup and verification. Both Binius backends attribute witness packing to witness_ms.",
                    memory_cap_enforced=address_space_limit(args.memory_gib) is not None,
                    memory_gib=args.memory_gib, timeout_seconds=args.timeout)
    try:
        write_json(args.output / "manifest.json", manifest)
        binary = args.binary.resolve(strict=True) if args.binary else build(args, env)
        manifest["binary"] = str(binary)
        manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        manifest["revision"] = root_metadata(ROOT)["revision"]
        manifest["vendor_snapshots"] = vendor_snapshots(ROOT)
        manifest["cargo_lock_sha256"] = hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest()
        (args.output / "source.patch").write_bytes(source_patch(ROOT))
        sources = args.output / "source"
        sources.mkdir()
        for relative in ["scripts/run_sha256_chain_compare.py", "benches/sha256_chain_compare.rs", "Cargo.lock", "Cargo.toml"]:
            (sources / Path(relative).name).write_bytes((ROOT / relative).read_bytes())
        fixtures = {}
        for exponent in args.exponents:
            description = subprocess.check_output([str(binary), "--fixture-only", "--exponent", str(exponent),
                                                   "--seed", str(args.seed)], cwd=ROOT, env=env, text=True)
            fixture = json.loads(description)
            if (fixture.get("fixture_profile") != FIXTURE or fixture.get("compressions") != 1 << exponent
                    or fixture.get("signatures") != 0 or fixture.get("padding") is not False):
                raise ValueError("worker does not implement the expected raw chain fixture")
            fixtures[exponent] = fixture
        write_json(args.output / "fixtures.json", fixtures)
        manifest["status"] = "running"
        write_json(args.output / "manifest.json", manifest)
        for case in planned:
            results.append(run_case(args, binary, case, fixtures[case["log_compressions"]], env))
            summarize(args.output, results)
        manifest["status"] = "complete" if all(r["status"] == "complete" for r in results) else "failed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest.update(status="failed", error=str(error))
        print(f"error: {error}", file=sys.stderr)
    finally:
        write_json(args.output / "manifest.json", manifest)
    print("\nMedian measured timings; warmup excluded:")
    if (args.output / "summary.csv").exists():
        with (args.output / "summary.csv").open() as stream:
            print(f"{'Method':20} {'k':>2} {'Threads':>7} {'Rate':>5} {'Prove ms':>12} {'E2E ms':>12} {'Peak RSS MiB':>13}  Status")
            for row in csv.DictReader(stream):
                fmt = lambda key: f"{float(row[key]):.3f}" if row[key] else "-"
                rss = f"{int(row['peak_rss_bytes'])/1024**2:.1f}" if row["peak_rss_bytes"] else "-"
                print(f"{row['method']:20} {row['log_compressions']:>2} {row['threads']:>7} {'1/'+str(2**int(row['log_inv_rate'])):>5} "
                      f"{fmt('prove_ms'):>12} {fmt('e2e_prover_ms'):>12} {rss:>13}  {row['status']}")
    print(f"Peak RSS is the whole-worker maximum, including setup and warmup.\nResults: {args.output}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
