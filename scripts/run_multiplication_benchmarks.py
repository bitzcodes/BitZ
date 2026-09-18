#!/usr/bin/env python3
"""Run the multiplication comparison matrix, or a unified experiment directly.

Without a target, run the archived suite's u32/u64/u128 matrix using the current
mul_compare executable: 180 configurations, one warmup and five measured proofs
plus a separate RSS trial per configuration. Build once and run sequentially.
  run_multiplication_benchmarks.py --dry-run
  run_multiplication_benchmarks.py --exponents 15 --reps 1 --threads 1

Choose bitz (proof/witness/pcs/piop/outer/bounds) or compare (proof/witness/pcs).
Put launcher options before -- and forward benchmark options after it:
  run_multiplication_benchmarks.py bitz -- proof --workload u64 --w 3
  run_multiplication_benchmarks.py compare -- pcs --workload baby-bear
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

from bench_support import cargo_executables, clean_environment, write_json
from local_provenance import vendor_snapshots
from mul_report import main as report
from mul_results import load

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"bitz": "mul_bitz", "compare": "mul_compare"}
BACKENDS = ("bitz", "binius64", "binius64-ligerito", "plonky3-fri", "limber")
# Retain the ZIP's campaign limits; these are workload choices, not RAM limits.
# WHIR and other experiments remain available through the explicit compare CLI.
MAX_EXPONENT = {
    "u32-mod32": dict(zip(BACKENDS, (23, 23, 23, 23, 19))),
    "u64": {"bitz": 21, "binius64": 21, "binius64-ligerito": 21, "limber": 19},
    "u128": {"bitz": 21, "binius64": 21, "binius64-ligerito": 19, "limber": 19},
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in TARGETS:
        return matrix_main(argv)
    args = parse_args(argv)
    env = benchmark_environment(os.environ)
    build = build_command(args)
    inspection = any(flag in args.experiment for flag in ("--help", "-h", "--dry-run"))
    if args.dry_run:
        print("Build:", shlex.join(build))
        command = experiment_command(args, f"<Cargo executable: {TARGETS[args.target]}>", inspection)
        print("Run:", shlex.join(command))
        if not inspection:
            print("Report:", shlex.join([sys.executable, str(ROOT / "scripts/mul_report.py"),
                                        str(args.output), "--out", str(args.output / "reports")]))
        return 0
    try:
        # Help and Rust's case-planning mode may build, but never reserve results,
        # require Perfetto, acquire the measurement lock, or generate reports.
        if inspection:
            with tempfile.TemporaryDirectory(prefix="bitz-mul-build-") as directory:
                executable, code = build_executable(args, env, Path(directory) / "build.log")
                if code:
                    return code
                return run_campaign(experiment_command(args, executable, True), env, None, gated=False)
        if args.output.exists():
            raise ValueError(f"results directory already exists: {args.output}")
        vendor_snapshots(ROOT)
        args.output.mkdir(parents=True, exist_ok=False)
        print(f"Results: {args.output}", flush=True)
        executable, code = build_executable(args, env, args.output / "build.log")
        if code:
            return code
        command = experiment_command(args, executable, False)
        log = args.output / "run.log"
        print(f"Running {shlex.join(command)}\nLog: {log}", flush=True)
        with log.open("x") as stream:
            code = run_campaign(command, env, stream, gated=not args.no_gate)
        if code:
            print(f"Benchmark exited {code}; partial artifacts retained in {args.output}", file=sys.stderr)
            return code
        # The shared loader checks completion, verified trials, and configuration
        # identities before any tables are emitted.
        return report([str(args.output), "--out", str(args.output / "reports")])
    except KeyboardInterrupt as error:
        print("Interrupted; any partial results and logs are retained.", file=sys.stderr)
        return getattr(error, "exit_code", 130)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parse_matrix_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", nargs="+", choices=("u32", *MAX_EXPONENT),
                        default=list(MAX_EXPONENT), help="u32 aliases multiplication modulo 2^32")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS),
                        help="Plonky3-FRI supports u32 only; use compare -- for WHIR")
    parser.add_argument("--threads", nargs="+", type=positive_integer, default=[1, 10])
    parser.add_argument("--reps", type=positive_integer, default=5)
    parser.add_argument("--warmups", type=nonnegative_integer, default=1)
    parser.add_argument("--exponents", nargs="+", type=positive_integer,
                        help="explicit sizes 2^N; otherwise use each backend's archived size limits")
    parser.add_argument("--bitz-profiles", nargs="+", choices=("custom:1:4", "custom:3:4"),
                        default=["custom:1:4", "custom:3:4"])
    parser.add_argument("--binius-rates", nargs="+", type=int, choices=(1, 3), default=[1, 3],
                        help="log inverse rates for both Binius backends; Ligerito uses rbr accounting")
    parser.add_argument("--memory", choices=("rss", "none"), default="rss")
    parser.add_argument("--output", type=Path, help="new suite directory; reports go in <output>/reports")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--swap-grow-gb", type=positive_number,
                        help="override the archived per-backend swap growth guards")
    parser.add_argument("--features", action="append", default=[], metavar="FEATURES")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the matrix and commands without building, running, or writing files")
    args = parser.parse_args(argv)
    args.workloads = ["u32-mod32" if w == "u32" else w for w in args.workloads]
    for name in ("workloads", "backends", "threads", "exponents", "bitz_profiles", "binius_rates"):
        values = getattr(args, name)
        if values and len(values) != len(set(values)):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if any(t > 65536 for t in args.threads):
        parser.error("--threads must be in 1..65536")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    args.output = (args.output or ROOT / "PerfRuns" / f"{stamp}-multiplication").resolve()
    args.target = "compare"
    try:
        matrix_plan(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def matrix_plan(args):
    campaigns, skipped = [], []
    for workload in args.workloads:
        for backend in args.backends:
            if backend not in MAX_EXPONENT[workload]:
                skipped.append(f"{workload}/{backend}: Plonky3-FRI supports u32-mod32 only")
    for threads in args.threads:
        for workload in args.workloads:
            for backend in args.backends:
                if backend not in MAX_EXPONENT[workload]:
                    continue
                exponents = args.exponents or list(range(15, MAX_EXPONENT[workload][backend] + 1, 2))
                minimum = 15 if backend == "bitz" else 4
                maximum = {"plonky3-fri": 29, "limber": 24}.get(backend, sys.maxsize.bit_length() - 10)
                if any(not minimum <= n <= maximum for n in exponents):
                    raise ValueError(f"{workload}/{backend}: exponents must be in {minimum}..{maximum}")
                variants = (args.bitz_profiles if backend == "bitz" else
                            args.binius_rates if backend.startswith("binius64") else [None])
                for variant in variants:
                    profile = variant if backend == "bitz" else None
                    rate = variant if backend.startswith("binius64") else None
                    label = f"{'u32' if workload == 'u32-mod32' else workload}-{backend}"
                    if profile or rate:
                        label += f"-rate{2 ** (int(profile.split(':')[1]) if profile else rate)}"
                    label += f"-t{threads}"
                    guard = args.swap_grow_gb or (34 if workload == "u128" and backend == "binius64" else
                                                 30 if backend.startswith("binius64") else 10)
                    experiment = ["proof", "--workload", workload, "--backends", backend,
                                  "--log-n", ",".join(map(str, exponents)), "--threads", str(threads),
                                  "--reps", str(args.reps), "--warmups", str(args.warmups),
                                  "--memory", args.memory]
                    if profile:
                        experiment += ["--w", "1", "--split", "0", "--bitz-profile", "100",
                                       "--ligerito", profile]
                    if rate:
                        experiment += ["--log-inv-rate", str(rate)]
                    if backend == "binius64-ligerito":
                        experiment += ["--binius-ligerito-accounting", "rbr"]
                    if backend == "plonky3-fri":
                        experiment += ["--log-inv-rate", "1"]
                    if backend == "limber":
                        experiment += ["--limber-bits", "100"]
                    campaigns.append(dict(id=label, workload=workload, backend=backend, threads=threads,
                                          exponents=exponents, bitz_profile=profile, binius_log_inv_rate=rate,
                                          swap_guard_gb=None if args.no_gate else guard,
                                          experiment=experiment, status="pending"))
    if not campaigns:
        raise ValueError("no supported workload/backend combinations selected")
    return campaigns, skipped


def campaign_args(args, campaign):
    return argparse.Namespace(**(vars(args) | dict(output=args.output / campaign["id"],
                             experiment=campaign["experiment"], swap_grow_gb=campaign["swap_guard_gb"])))


def matrix_main(argv):
    args = parse_matrix_args(argv)
    campaigns, skipped = matrix_plan(args)
    cells = sum(len(c["exponents"]) for c in campaigns)
    print(f"Multiplication benchmarks: {len(campaigns)} campaigns, {cells} configurations, "
          f"{cells * args.reps} measured proofs; {args.warmups} warmups per configuration; memory={args.memory}")
    print(f"Results: {args.output}")
    for reason in skipped:
        print(f"Skipped: {reason}")
    if args.dry_run:
        print("Build:", shlex.join(build_command(args)))
        for campaign in campaigns:
            command = experiment_command(campaign_args(args, campaign), "<Cargo executable: mul_compare>", False)
            print(f"{campaign['id']}:", shlex.join(command))
        print("Report:", shlex.join([sys.executable, str(ROOT / "scripts/mul_report.py"),
                                    *(str(args.output / c["id"]) for c in campaigns),
                                    "--out", str(args.output / "reports")]))
        return 0
    try:
        if args.output.exists():
            raise ValueError(f"results directory already exists: {args.output}")
        vendor_snapshots(ROOT)
        env = benchmark_environment(os.environ)
        processor = env.get("PERFETTO_TRACE_PROCESSOR", "trace_processor_shell")
        resolved = shutil.which(processor, path=env.get("PATH", os.defpath))
        if not resolved:
            raise ValueError("Perfetto processor not executable; run bash scripts/install_trace_processor.sh "
                             "or set PERFETTO_TRACE_PROCESSOR")
        env["PERFETTO_TRACE_PROCESSOR"] = str(Path(resolved).resolve())
        return execute_matrix(args, campaigns, skipped, env)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def execute_matrix(args, campaigns, skipped, env):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="multiplication-benchmarks/v2", status="running", phase="build",
                    created_utc=datetime.now(timezone.utc).isoformat(), campaigns=campaigns, skipped=skipped,
                    repetitions=args.reps, warmups=args.warmups, memory=args.memory,
                    build_command=build_command(args), trace_processor=env["PERFETTO_TRACE_PROCESSOR"])
    def save():
        write_json(args.output / "suite.json", manifest)
    save()
    try:
        executable, code = build_executable(args, env, args.output / "build.log")
        if code:
            manifest.update(status="failed", returncode=code)
            save()
            return code
        # Ask the actual Rust executable to validate every selection before any
        # proofs start. Keep its jobs to check that results match the plan.
        manifest["phase"] = "preflight"
        save()
        for campaign in campaigns:
            plan_log = args.output / f"{campaign['id']}.plan.json"
            with plan_log.open("x") as stream:
                code = run_campaign([executable, *campaign["experiment"], "--dry-run"], env, stream, gated=False)
            if code:
                raise ValueError(f"invalid campaign {campaign['id']}; inspect {plan_log}")
            jobs = json.loads(plan_log.read_text())
            if not jobs or any(job.get("skip") for job in jobs):
                raise ValueError(f"{campaign['id']}: preflight did not select runnable cases")
            campaign["planned_jobs"] = jobs
        completed = []
        for index, campaign in enumerate(campaigns, 1):
            child = campaign_args(args, campaign)
            command = experiment_command(child, executable, False)
            campaign.update(status="running", command=command)
            manifest["phase"] = "proofs"
            save()
            log = args.output / f"{campaign['id']}.log"
            print(f"[{index}/{len(campaigns)}] {campaign['id']} — log: {log}", flush=True)
            with log.open("x") as stream:
                code = run_campaign(command, env, stream, gated=not args.no_gate)
            campaign["returncode"] = code
            if code:
                campaign["status"] = manifest["status"] = "failed"
                save()
                print(f"Failed: {campaign['id']}. See {log}", file=sys.stderr)
                return code
            cases = load(child.output)
            if [case.job for case in cases] != campaign["planned_jobs"]:
                raise ValueError(f"{campaign['id']}: results differ from the preflight plan")
            completed.append(str(child.output))
            manifest["phase"] = "report"
            save()
            code = report([*completed, "--out", str(args.output / "reports")])
            if code:
                raise RuntimeError(f"report generation exited {code}")
            campaign["status"] = "complete"
            save()
    except (KeyboardInterrupt, OSError, ValueError, RuntimeError, KeyError) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error)
        for campaign in campaigns:
            if campaign["status"] == "running":
                campaign["status"] = manifest["status"]
        save()
        print(f"{manifest['status']}: {error}. Results retained in {args.output}", file=sys.stderr)
        return getattr(error, "exit_code", 130) if isinstance(error, KeyboardInterrupt) else 1
    manifest.update(status="complete", phase="complete")
    save()
    print(f"Completed. Combined results: {args.output / 'reports/summary.csv'}")
    return 0


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    separator = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("--output", type=Path,
                        help="new results directory (default: PerfRuns/<UTC timestamp>-multiplication)")
    parser.add_argument("--no-gate", action="store_true", help="run without the machine lock or swap guard")
    parser.add_argument("--swap-grow-gb", type=positive_number, default=12.0,
                        help="allowed swap growth while gated (default: 12 GiB)")
    parser.add_argument("--features", action="append", default=[], metavar="FEATURES",
                        help="additional comma-separated Cargo features; repeatable (e.g. bench-peak-memory)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print build/run/report commands without building, running, or writing files")
    args = parser.parse_args(argv[:separator])
    args.experiment = argv[separator + 1:]
    # Output is orchestration state; worker transport is internal to Rust.
    for flag in args.experiment:
        if flag.split("=", 1)[0] in ("--out", "--worker"):
            parser.error("use launcher --output before --; --out and --worker are reserved")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    args.output = (args.output or ROOT / "PerfRuns" / f"{stamp}-multiplication").resolve()
    return args


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def benchmark_environment(inherited):
    env = clean_environment(inherited)
    # Preserve a caller's custom lock while dropping pre-rename controls too.
    env = {key: value for key, value in env.items() if not key.startswith("F2Z_")}
    if "BITZ_BENCH_LOCK" in inherited:
        env["BITZ_BENCH_LOCK"] = inherited["BITZ_BENCH_LOCK"]
    if "RUSTFLAGS" not in env and "CARGO_ENCODED_RUSTFLAGS" not in env:
        env["RUSTFLAGS"] = "-C target-cpu=native"
    local = ROOT / ".tools/perfetto/trace_processor_shell"
    if "PERFETTO_TRACE_PROCESSOR" not in env and os.access(local, os.X_OK):
        env["PERFETTO_TRACE_PROCESSOR"] = str(local)
    return env


def build_command(args):
    features = ["span-metrics", "bench-internals"]
    if args.target == "compare":
        features.append("native-mul-compare")
    features.extend(feature for group in args.features for feature in group.split(",") if feature)
    features = list(dict.fromkeys(features))
    return ["cargo", "bench", "--no-run", "--locked", "--bench", TARGETS[args.target],
            "--features", ",".join(features), "--message-format=json"]


def build_executable(args, env, log):
    command = build_command(args)
    print("Building:", shlex.join(command), flush=True)
    with log.open("x") as stream:
        code = run_campaign(command, env, stream, gated=False)
    if code:
        # Inspection logs live in a temporary directory, so always show failures.
        print(log.read_text(), file=sys.stderr)
        return None, code
    target = TARGETS[args.target]
    source = (ROOT / "benches" / f"{target}.rs").resolve()
    artifacts = []
    for line in log.read_text().splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        metadata = event.get("target", {})
        if (event.get("reason") == "compiler-artifact"
                and "bench" in metadata.get("kind", [])
                and Path(metadata.get("src_path", "")).resolve() == source):
            artifacts.append(line)
    executable = cargo_executables("\n".join(artifacts), [target])[target]
    return executable, 0


def experiment_command(args, executable, inspection):
    command = [str(executable), *args.experiment]
    if inspection:
        return command
    command += ["--out", str(args.output)]
    if not args.no_gate:
        command = [sys.executable, str(ROOT / "scripts/bench_gate.py"), "run",
                   "--label", args.output.name, "--swap-grow-gb", str(args.swap_grow_gb), "--", *command]
    return command


def run_campaign(command, environment, stream, *, gated):
    with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                          stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            code = process.wait()
        except BaseException:
            # Finish cleanup even if the user presses Ctrl-C again.
            handlers = {sig: signal.signal(sig, signal.SIG_IGN)
                        for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                # The gate owns a separate worker group and releases its lock
                # only after those workers have stopped.
                try:
                    if gated:
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            finally:
                if not gated:
                    kill_remaining_workers(process.pid)
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            raise
        finally:
            if not gated and process.poll() is not None:
                kill_remaining_workers(process.pid)
    return code if code >= 0 else 128 - code


def kill_remaining_workers(group):
    # A campaign leader can exit before a worker that ignores SIGTERM.
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def terminated(signum, _frame):
    error = KeyboardInterrupt()
    error.exit_code = 128 + signum
    raise error


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminated)
    sys.exit(main())
