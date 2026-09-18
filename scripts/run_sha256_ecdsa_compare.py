#!/usr/bin/env python3
"""Compare signed SHA-256 chains in fresh processes; retain failures and full samples."""
from bench_support import cargo_executables, source_metadata, environment, file_hash, run_process, address_space_limit, write_json
import argparse
from local_provenance import source_patch, vendor_snapshots
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bitz/sha256-ecdsa-compare/v1"
# The 2026-09-13 suite: BitZ at rates 1/2 and 1/8 (Ligerito profiles), Binius64
# and Binius64-with-BitZ-opener each at rates 1/2 and 1/8 (round-by-round gate).
DEFAULT_METHODS = ["bitz-split", "binius64", "binius64-ligerito"]
METHODS = ["bitz-split", "bitz-all", "binius64", "binius64-ligerito"]
# Ligerito profile per BitZ case: custom:1:4 = Johnson rate 1/2, custom:3:4 = rate 1/8.
DEFAULT_BITZ_PROFILES = ["custom:1:4", "custom:3:4"]
PROFILE_RATES = {"custom:1:4": 1, "custom:3:4": 3}
DEFAULT_BINIUS_RATES = [1, 3]
FIXTURE_SCHEMA = "bitz/sha256-ecdsa-fixture/standard-p256/v1"
METRICS = ["setup_ms", "witness_ms", "commit_ms", "protocol_ms", "prove_ms",
           "witness_to_proof_ms", "e2e_prover_ms", "verify_ms", "codec_ms", "outer_ms", "inner_ms",
           "opening_ms", "folding_ms", "piop_ms", "iop_ms", "proof_object_bytes", "proof_material_bytes"]


def sample_metrics(row):
    """Derive disjoint protocol stages from the original measured sample.

    PIOP includes projection, preparation, folding, sumchecks and transcript
    work outside the timed opening. IOP/PCS is the complete opening path.
    Keep the raw worker records unchanged; derive before taking medians.
    """
    row = dict(row)
    protocol, opening = row.get("protocol_ms"), row.get("opening_ms")
    valid = (type(protocol) in (int, float) and type(opening) in (int, float)
             and math.isfinite(protocol) and math.isfinite(opening) and 0 <= opening <= protocol)
    row["piop_ms"] = protocol - opening if valid else None
    row["iop_ms"] = opening if valid else None
    return row


def cases(shapes, methods, targets, threads, seeds,
          bitz_profiles=DEFAULT_BITZ_PROFILES, binius_rates=DEFAULT_BINIUS_RATES):
    """BitZ depends only on total work; emit it once for each exponent/target.

    Every case carries its scheme configuration: BitZ methods a `ligerito_profile`
    (the per-case `BITZ_LIG_PROFILE`), Binius-family methods a `log_inv_rate`.
    The opener (`binius64-ligerito`) exists only at its fixed 100-bit gate.
    """
    work = sorted({r + c for r, c in shapes})
    for method, workers, seed in itertools.product(methods, threads, seeds):
        if method.startswith("bitz"):
            for exponent, target, profile in itertools.product(work, targets, bitz_profiles):
                yield dict(method=method, log_compressions=exponent, r=None, c=None,
                           security_target=target, threads=workers, seed=seed,
                           ligerito_profile=profile)
        else:
            method_targets = [100] if method == "binius64-ligerito" else targets
            for exponent, target, rate in itertools.product(work, method_targets, binius_rates):
                yield dict(method=method, log_compressions=exponent, r=None, c=None,
                           security_target=target, threads=workers, seed=seed,
                           log_inv_rate=rate)


def validate_rows(rows, case, reps, binius_log_inv_rate=None):
    """Reject incomplete, mislabelled, unverified or fixture-changing workers."""
    expected_rate = case.get("log_inv_rate", binius_log_inv_rate if binius_log_inv_rate is not None else 1)
    if binius_log_inv_rate is not None and "log_inv_rate" in case and expected_rate != binius_log_inv_rate:
        return False
    if len(rows) != reps + 1:
        return False
    for sample, row in enumerate(rows):
        row = sample_metrics(row)
        timing = row.get("timing", "perfetto")
        if timing not in ("perfetto", "wall-clock") or timing != case.get("timing", "perfetto"):
            return False
        if timing == "wall-clock":
            if case["method"].startswith("binius64"):
                return False
            if any(row.get(k) is not None for k in
                   ("outer_ms", "inner_ms", "opening_ms", "folding_ms")):
                return False
        if row.get("schema") != SCHEMA or row.get("verified") is not True:
            return False
        if any(row.get(k) != v for k, v in case.items()):
            return False
        if row.get("sample") != sample or row.get("trial") != ("warmup" if sample == 0 else "sample"):
            return False
        if row.get("compressions") != 1 << case["log_compressions"]:
            return False
        if row.get("message_bytes") != 64 * ((1 << case["log_compressions"]) - 1):
            return False
        if row.get("signatures") != 1 or row.get("statement_bytes") != 129:
            return False
        if not isinstance(row.get("fixture_id"), str) or len(row["fixture_id"]) != 64:
            return False
        if not isinstance(row.get("security"), dict) or "model" not in row["security"]:
            return False
        if row.get("zk") is not False or row.get("fixture_profile") != FIXTURE_SCHEMA:
            return False
        if case["method"].startswith("bitz"):
            from ligerito_results import validate_ligerito
            try:
                validate_ligerito(row["security"].get("ligerito"), case["security_target"])
                if row["security"]["ligerito"] != rows[0]["security"].get("ligerito"):
                    return False
            except ValueError:
                return False
            if "ligerito_profile" in case:
                # The case pins the opener profile; the recorded request and
                # its resolved level-0 rate must both agree.
                if row["security"]["ligerito"].get("requested_profile") != case["ligerito_profile"]:
                    return False
                expected_rate = PROFILE_RATES.get(case["ligerito_profile"])
                config = row["security"]["ligerito"].get("configuration", {})
                levels = config.get("levels") or [{}]
                if expected_rate is not None and levels[0].get("log_inv_rate", 1) != expected_rate:
                    return False
        binius = case["method"].startswith("binius64")
        revision = row.get("binius_revision" if binius else "bitz_revision")
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            return False
        if binius and (row["security"].get("log_inv_rate") != expected_rate
                       or row.get("circuit_profile") != "sha256-chain-p256/standard/v1"):
            return False
        if case["method"] == "binius64" and (row["security"].get("fri_query_target_bits") != case["security_target"]
                                             or row["security"].get("pcs") != "BaseFold"):
            return False
        if case["method"] == "binius64-ligerito":
            s = row["security"]
            if (s.get("pcs") != "BitZ-Ligerito" or s.get("accounting") != "round-by-round"
                    or s.get("target_bits") != 100 or case["security_target"] != 100
                    or type(s.get("round_by_round_bits")) not in (int, float)
                    or s["round_by_round_bits"] < 100):
                return False
        for key in METRICS:
            value = row.get(key)
            optional = ["folding_ms", "outer_ms", "inner_ms", "opening_ms"]
            if timing == "wall-clock":
                optional += ["piop_ms", "iop_ms"]
            if value is None and key in optional:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                return False
        if not math.isclose(row["prove_ms"], row["commit_ms"] + row["protocol_ms"], abs_tol=1e-6):
            return False
        if not math.isclose(row["witness_to_proof_ms"], row["witness_ms"] + row["prove_ms"], abs_tol=1e-6):
            return False
    return len({row["fixture_id"] for row in rows}) == 1


def summarize(directory):
    paths = sorted(directory.glob("*.result.json"))
    results = [json.loads(p.read_text()) for p in paths]
    fields = ["method", "log_compressions", "r", "c", "security_target", "threads", "seed",
              "status", "samples", "peak_rss_bytes", "fixture_id", "security_model",
              "economic_bits", "statistical_bits_lower_bound", *METRICS,
              "log_inv_rate", "ligerito_profile", "timing"]
    fixture_ids = {}
    sample_fields = [*fields[:8], "source_file", "trial", "sample", "verified", "peak_rss_bytes",
                     "fixture_id", "bitz_revision", "binius_revision", "artifact_id", "security_model", "economic_bits",
                     "statistical_bits_lower_bound", *METRICS, "log_inv_rate", "ligerito_profile", "timing"]
    with (directory / "summary.csv").open("w", newline="") as stream, \
            (directory / "samples.csv").open("w", newline="") as sample_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        sample_writer = csv.DictWriter(sample_stream, fieldnames=sample_fields)
        sample_writer.writeheader()
        for path, result in zip(paths, results):
            row = {k: result["case"][k] for k in fields[:7]}
            row.update(status=result["status"], peak_rss_bytes=result["peak_rss_bytes"],
                       timing=result["case"].get("timing", "perfetto"),
                       log_inv_rate=result["case"].get("log_inv_rate"),
                       ligerito_profile=result["case"].get("ligerito_profile"))
            normalized = [sample_metrics(s) for s in result["rows"]]
            for sample in normalized:
                record = dict(row, source_file=path.name)
                record.update({k: sample.get(k) for k in ["trial", "sample", "verified", "fixture_id",
                                                         "bitz_revision", "binius_revision", "artifact_id", *METRICS,
                                                         "log_inv_rate", "ligerito_profile"]})
                security = sample.get("security")
                if isinstance(security, dict):
                    record.update(security_model=security.get("model"), economic_bits=security.get("economic_bits"),
                                  statistical_bits_lower_bound=security.get("statistical_bits_lower_bound"))
                sample_writer.writerow(record)
            samples = [s for s in normalized if s.get("trial") == "sample"]
            row["samples"] = len(samples)
            if samples:
                row["fixture_id"] = samples[0].get("fixture_id")
                security = samples[0].get("security", {})
                if not isinstance(security, dict):
                    security = {}
                row.update(security_model=security.get("model"), economic_bits=security.get("economic_bits"),
                           statistical_bits_lower_bound=security.get("statistical_bits_lower_bound"))
                for key in METRICS:
                    values = [s[key] for s in samples if type(s.get(key)) in (int, float) and math.isfinite(s[key])]
                    row[key] = statistics.median(values) if values else None
            if result["status"] == "complete":
                key = (row["log_compressions"], row["seed"])
                fixture_ids.setdefault(key, set()).add(row["fixture_id"])
            writer.writerow(row)
    mismatches = [list(key) for key, ids in fixture_ids.items() if len(ids) != 1]
    (directory / "comparison.json").write_text(json.dumps({
        "all_recorded_cases_complete": all(r["status"] == "complete" for r in results),
        "matched_fixtures": not mismatches, "mismatched_exponent_seed_pairs": mismatches,
        "security_note": "BitZ economic targets, Spartan group security, and Binius FRI query targets use different accounting; see each row.",
        "proof_size_note": "proof_material_bytes includes commitments and auxiliary inputs; statement bytes are separate.",
        "timing_note": "Per sample: IOP/PCS = opening_ms; PIOP/preparation = protocol_ms - opening_ms. e2e_prover_ms is independently measured from witness generation through proof completion, excluding reusable setup and codec. Historical unavailable measurements remain null.",
        "memory_note": "peak_rss_bytes is the whole worker maximum, including setup and all trials; repeated in samples.csv, not measured per proof.",
    }, indent=2) + "\n")
    return not mismatches


def print_summary(directory):
    """Print median timings and whole-worker peak memory, including resumed cases."""
    columns = [("method", "Method"), ("status", "Status"), ("timing", "Timing"),
               ("log_compressions", "Exponent"), ("threads", "Threads"),
               ("ligerito_profile", "Profile"), ("setup_ms", "Setup ms"),
               ("witness_ms", "Witness ms"), ("prove_ms", "Prove ms"),
               ("e2e_prover_ms", "E2E ms"), ("verify_ms", "Verify ms"),
               ("proof_material_bytes", "Proof bytes"), ("peak_rss_bytes", "Peak RSS MiB")]
    with (directory / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    table = [[label for _, label in columns]]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key) or "-"
            if row["status"] != "complete" and key in METRICS:
                value = "-"
            elif value != "-" and key.endswith("_ms"):
                value = f"{float(value):.3f}"
            elif value != "-" and key == "peak_rss_bytes":
                value = f"{int(value) / 1024**2:.1f}"
            values.append(value)
        table.append(values)
    widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
    print("\nMedian timings (warmup excluded; E2E excludes setup and codec):")
    for row in table:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    print("\nPeak RSS is the whole-worker maximum, including setup, warmup, measured proofs "
          "and verification; it is not a per-proof median. MiB = 2^20 bytes.")
    print(f"\nFull results: {directory / 'summary.csv'}\nIndividual trials: {directory / 'samples.csv'}")


def build(args, directory):
    command = ["cargo", "build", "--release", "--locked", "--features", "sha256-ecdsa-compare",
               "--bench", "sha256_ecdsa_compare", "--message-format=json-render-diagnostics"]
    if args.offline:
        command.append("--offline")
    env = dict(os.environ)
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    (directory / "build.jsonl").write_text(result.stdout)
    (directory / "build.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"build failed; see {directory / 'build.log'}")
    return Path(cargo_executables(result.stdout, ["sha256_ecdsa_compare"])["sha256_ecdsa_compare"]).resolve()


def metadata(binary):
    fingerprint = (binary.parent.parent / ".fingerprint" /
                   binary.name.replace("sha256_ecdsa_compare-", "bitz-", 1) /
                   "test-bench-sha256_ecdsa_compare.json")
    build_info = json.loads(fingerprint.read_text()) if fingerprint.exists() else None
    return dict(vendor_snapshots=vendor_snapshots(ROOT), binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                runner_sha256=file_hash(Path(__file__)), memory_limit_enforced=sys.platform.startswith("linux"),
                **source_metadata(ROOT), build=build_info, runtime_env=environment(os.environ))


def case_config_token(case):
    """Filesystem token for the case's scheme configuration, '' if none."""
    if "ligerito_profile" in case:
        return "-lig" + case["ligerito_profile"].replace(":", "_")
    if "log_inv_rate" in case:
        return f"-rate{case['log_inv_rate']}"
    return ""


def run_case(binary, case, args, directory):
    shape = f"r{case['r']}-c{case['c']}" if case["r"] is not None else "total"
    name = (f"{case['method']}-i{case['log_compressions']}-{shape}-s{case['security_target']}"
            f"{case_config_token(case)}-t{case['threads']}-seed{case['seed']}-reps{args.reps}")
    path = directory / f"{name}.result.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["status"] == "complete" or not args.retry_failed:
            print(f"skip {name}: {existing['status']}", flush=True)
            return existing["status"] == "complete"
    r = case["r"] if case["r"] is not None else case["log_compressions"]
    c = case["c"] if case["c"] is not None else 0
    binius = case["method"].startswith("binius64")
    worker = args.binius64_worker if binius else binary
    command = [str(worker), "--method", case["method"], "--r", str(r), "--c", str(c),
               "--threads", str(case["threads"]), "--reps", str(args.reps), "--seed", str(case["seed"])]
    if not binius:
        command.extend(["--timing", args.timing])
    if case["security_target"] is not None:
        command.extend(["--target", str(case["security_target"])])
    if "log_inv_rate" in case:
        command.extend(["--log-inv-rate", str(case["log_inv_rate"])])
    fixture = directory / "fixtures" / f"i{case['log_compressions']}-seed{case['seed']}.json"
    expected_fixture = json.loads(fixture.read_text())["id"]
    command.extend(["--fixture", str(fixture)])
    rss_path = directory / f"{name}.rss-kib"
    rss_path.unlink(missing_ok=True)
    if Path("/usr/bin/time").exists() and sys.platform.startswith("linux"):
        command = ["/usr/bin/time", "-f", "%M", "-o", str(rss_path), *command]
    elif sys.platform == "darwin":
        command = ["/usr/bin/time", "-l", *command]

    started = time.monotonic()
    status, returncode = "failed", None
    with (directory / f"{name}.stdout").open("w") as stdout, (directory / f"{name}.stderr").open("w") as stderr:
        env = dict(os.environ, RAYON_NUM_THREADS=str(case["threads"]),
                   HARDWARE_CONCURRENCY=str(case["threads"]))
        # BitZ cases pin their Ligerito profile per case; other methods must
        # never see an ambient profile.
        env.pop("BITZ_LIG_PROFILE", None)
        if "ligerito_profile" in case:
            env["BITZ_LIG_PROFILE"] = case["ligerito_profile"]
        try:
            returncode, timed_out = run_process(command, stdout=stdout, stderr=stderr, cwd=ROOT,
                                                preexec_fn=address_space_limit(args.memory_gib),
                                                env=env, timeout=args.timeout)
            if timed_out:
                status = "timeout"
        except (OSError, subprocess.SubprocessError) as exc:
            stderr.write(str(exc) + "\n")
    rows = []
    for line in (directory / f"{name}.stdout").read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("schema") == SCHEMA:
            rows.append(row)
    if (returncode == 0 and validate_rows(rows, case, args.reps, args.binius_log_inv_rate)
            and all(row["fixture_id"] == expected_fixture for row in rows)
            and (not case["method"].startswith("binius64")
                 or all(row["binius_revision"] == args.binius64_info["binius_revision"] for row in rows))):
        status = "complete"
    rss = peak_rss_bytes(rss_path, directory / f"{name}.stderr")
    result = dict(case=case, status=status, returncode=returncode, rows=rows,
                  peak_rss_bytes=rss, elapsed_seconds=time.monotonic()-started)
    write_json(path, result)
    summarize(directory)
    print(f"{name}: {status} ({result['elapsed_seconds']:.1f}s)", flush=True)
    return status == "complete"


def peak_rss_bytes(linux_rss_path, stderr_path):
    if sys.platform == "darwin":
        for line in stderr_path.read_text().splitlines():
            fields = line.split()
            if fields[1:] == ["maximum", "resident", "set", "size"] and fields[0].isdigit():
                return int(fields[0])  # BSD time reports bytes; GNU time reports KiB.
    elif linux_rss_path.exists():
        lines = linux_rss_path.read_text().splitlines()
        if lines and lines[-1].isdigit():
            return int(lines[-1]) * 1024
    return None


def prepare_fixtures(args, directory, binary, splits):
    fixtures = directory / "fixtures"
    fixtures.mkdir(exist_ok=True)
    hashes = {}
    for exponent, seed in itertools.product(sorted({r+c for r,c in splits}), sorted(set(args.seeds))):
        path = fixtures / f"i{exponent}-seed{seed}.json"
        if not path.exists():
            subprocess.run([str(binary), "--method", "bitz-split", "--r", str(exponent), "--c", "0",
                            "--seed", str(seed), "--export-fixture", str(path)], check=True)
        if json.loads(path.read_text()).get("schema") != FIXTURE_SCHEMA:
            raise ValueError("Legacy fixture profile: regenerate fixtures in a new output directory")
        hashes[path.name] = file_hash(path)
    return dict(profile=FIXTURE_SCHEMA, files=hashes)


def prepare_binius(args, directory):
    worker_root = ROOT / "benchmarks/binius64"
    if args.binius64_binary:
        worker = args.binius64_binary.resolve(strict=True)
        metadata_path = worker.with_suffix(".build.json")
        if not metadata_path.is_file():
            raise ValueError("Prebuilt Binius worker requires its .build.json provenance sidecar")
        info = json.loads(metadata_path.read_text())
        if info["binary_sha256"] != file_hash(worker):
            raise ValueError("Binius binary does not match its build provenance")
    else:
        command = [sys.executable, str(worker_root / "build.py")]
        if args.offline:
            command.append("--offline")
        with (directory / "binius64-build.log").open("w") as log:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True, check=True)
        info = json.loads(result.stdout.splitlines()[-1])
        worker = Path(info["binary"])
    embedded = json.loads(subprocess.check_output([str(worker), "--build-info"], text=True))
    for key in ["binius_revision", "lock_sha256", "source_sha256", "rustc", "rustflags"]:
        if embedded[key] != info[key]:
            raise ValueError(f"Binius embedded provenance mismatch: {key}")
    if len(info["binius_revision"]) != 40 or any(c not in "0123456789abcdef" for c in info["binius_revision"]):
        raise ValueError("Binius worker must be built from a pinned published commit")
    args.binius64_worker, args.binius64_info = worker, info
    return info


def compatible_manifest(previous, current):
    return all(previous.get(key) == current.get(key) for key in
               ["binary_sha256", "runner_sha256", "fixtures", "binius64", "ligerito_profile", "binius_log_inv_rate", "timing"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binius64-binary", type=Path, help="Prebuilt worker with its .build.json sidecar")
    parser.add_argument("--binius-log-inv-rate", type=int, choices=[1, 2, 3], default=None,
                        help="Binius64 initial code rate: 1=1/2, 2=1/4, 3=1/8")
    parser.add_argument("--exponents", nargs="+", type=int, default=[3, 5, 7],
                        help="Total compression exponents i (default: 3 5 7)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--targets", nargs="+", type=int, choices=[100, 128], default=[100],
                        help="security targets; the binius64-ligerito gate is fixed at 100, so its cases exist only there")
    parser.add_argument("--bitz-profiles", nargs="+", choices=sorted(PROFILE_RATES), default=DEFAULT_BITZ_PROFILES,
                        help="Ligerito profile per BitZ case: custom:1:4 = rate 1/2, custom:3:4 = rate 1/8")
    parser.add_argument("--binius-rates", nargs="+", type=int, choices=[1, 2, 3], default=None,
                        help="log inverse rate per Binius-family case: 1 = rate 1/2, 3 = rate 1/8")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--memory-gib", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timing", choices=["perfetto", "wall-clock"], default="perfetto",
                        help="wall-clock uses Rust timers without Perfetto; supported for BitZ, with internal phase timings unavailable")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Regenerate summary.csv and samples.csv from existing raw records")
    args = parser.parse_args()
    if args.binius_log_inv_rate is not None and args.binius_rates is not None:
        parser.error("select either --binius-log-inv-rate or --binius-rates")
    args.binius_rates = args.binius_rates or ([args.binius_log_inv_rate] if args.binius_log_inv_rate is not None else DEFAULT_BINIUS_RATES)
    args.with_binius64 = any(method.startswith("binius64") for method in args.methods)
    if args.summarize_only:
        if not args.output.is_dir() or not any(args.output.glob("*.result.json")):
            parser.error("--summarize-only requires an output directory with recorded cases")
        matched = summarize(args.output.resolve())
        print_summary(args.output.resolve())
        return 0 if matched else 1
    if os.environ.get("BITZ_LIG_PROFILE"):
        parser.error("unset BITZ_LIG_PROFILE: the runner selects it per case (--bitz-profiles)")
    if args.timing == "wall-clock" and args.with_binius64:
        parser.error("--timing wall-clock supports --methods bitz-split bitz-all")
    if args.exponents and any(i not in range(3, 17) for i in args.exponents):
        parser.error("compression exponents must be in 3..16")
    shapes = [(i, 0) for i in args.exponents]
    if (args.reps < 1 or args.memory_gib < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or min(args.threads) < 1
            or any(not 0 <= seed < 2**64 for seed in args.seeds)):
        parser.error("positive reps/threads/limits and unsigned 64-bit seeds required")
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.output.resolve()
    binary = args.binary.resolve(strict=True) if args.binary else build(args, directory)
    manifest = metadata(binary)
    manifest["timing"] = args.timing
    # Profiles are pinned per case (recorded in each case and row); the
    # manifest-level value only guards resumption against runner-policy drift.
    manifest["ligerito_profile"] = "per-case:" + ",".join(args.bitz_profiles)
    manifest["fixtures"] = prepare_fixtures(args, directory, binary, shapes)
    if args.with_binius64:
        manifest["binius64"] = prepare_binius(args, directory)
        manifest["binius_log_inv_rate"] = args.binius_log_inv_rate
    manifest["campaign"] = dict(methods=args.methods, targets=args.targets, threads=args.threads,
                                bitz_profiles=args.bitz_profiles, binius_rates=args.binius_rates,
                                seeds=args.seeds, reps=args.reps, timeout=args.timeout, memory_gib=args.memory_gib,
                                memory_cap_enforced=address_space_limit(args.memory_gib) is not None)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not compatible_manifest(previous, manifest):
            parser.error("output belongs to different binaries, circuits, or fixtures; choose a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        patch = source_patch(ROOT)
        (directory / "source.patch").write_bytes(patch)
        # Preserve this task's new source files, which git diff does not include.
        sources = ["benches/sha256_ecdsa_compare.rs", "scripts/run_sha256_ecdsa_compare.py",
                   "benches/support/sha256_ecdsa_fixture.rs"]
        if args.with_binius64:
            sources += ["benchmarks/binius64/" + path for path in
                        ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "build.py", "build.rs", "src/main.rs"]]
        for relative in sources:
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
    jobs = list(cases(shapes, args.methods, args.targets, sorted(set(args.threads)), sorted(set(args.seeds)),
                      bitz_profiles=args.bitz_profiles, binius_rates=sorted(set(args.binius_rates))))
    for case in jobs:
        if not case["method"].startswith("binius64"):
            case["timing"] = args.timing
    (directory / "requested_cases.json").write_text(json.dumps(jobs, indent=2)+"\n")
    complete = True
    for case in jobs:
        complete = run_case(binary, case, args, directory) and complete
    matched = summarize(directory)
    print_summary(directory)
    if not matched:
        print("ERROR: completed methods used different fixtures; see comparison.json", file=sys.stderr)
    return 0 if complete and matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
