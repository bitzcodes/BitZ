"""Ordinary process and artifact helpers shared by benchmark scripts."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

from local_provenance import root_metadata, source_patch

ROOT = Path(__file__).resolve().parents[1]
BUILD_ENV = {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_HOME", "CARGO_TARGET_DIR", "RAYON_NUM_THREADS"}


def command_text(*command, cwd=ROOT, missing="unavailable"):
    try:
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return missing


def cpu_name():
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    if sys.platform == "darwin":
        return command_text("sysctl", "-n", "machdep.cpu.brand_string")
    return platform.processor() or platform.machine()


def environment(env, prefixes=("BITZ_", "F2_", "OBLONG_"), keys=BUILD_ENV):
    return {key: value for key, value in env.items() if key.startswith(prefixes) or key in keys}


def file_hash(path):
    # Chunked rather than hashlib.file_digest (Python 3.11+); same digest.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_metadata(root=ROOT):
    diff = source_patch(root)
    return dict(**root_metadata(root),
                tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                rustc=command_text("rustc", "-Vv", cwd=root), cpu=cpu_name(),
                platform=platform.platform(), logical_cpus=os.cpu_count())


def cargo_executables(log, required=()):
    """Use Cargo's structured artifact events, never human-readable path text."""
    executables = {}
    for line in log.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("reason") == "compiler-artifact" and event.get("executable"):
            name = event["target"]["name"]
            executable = event["executable"]
            if name in executables and executables[name] != executable:
                raise ValueError(f"Cargo returned multiple executables for {name}")
            executables[name] = executable
    missing = set(required) - executables.keys()
    if missing:
        raise RuntimeError(f"Cargo did not report executables for {sorted(missing)}")
    return executables


def run_process(command, *, env=None, cwd=None, stdout=None, stderr=None,
                timeout=None, preexec_fn=None, start_new_session=True):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                               start_new_session=start_new_session, preexec_fn=preexec_fn)
    def stop():
        try:
            if start_new_session:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass  # The child may finish between wait and cancellation.
        process.wait()

    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        stop()
        return None, True
    except BaseException:
        stop()
        raise


def run_logged(command, env, path, cwd=ROOT):
    print(f"Running {' '.join(map(str, command))}; log: {path}", flush=True)
    with Path(path).open("w") as stream:
        code, _ = run_process(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"exit {code}; inspect {path}")
    return Path(path).read_text()


def address_space_limit(memory_gib):
    if not sys.platform.startswith("linux"):
        return None
    def limit():
        import resource
        size = memory_gib * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (size, size))
    return limit


def write_json(path, value):
    """Replace a complete artifact atomically, including on interrupted campaigns."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def filtered_environment(env, prefixes=(), keys=(), overrides=None):
    result = {k: v for k, v in env.items() if not k.startswith(prefixes) and k not in keys}
    result.update(overrides or {})
    return result


def stream_logged(command, *, cwd, env, log, output=sys.stdout):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        output.write(line)
        output.flush()
        log.write(line)
    return process.wait()


def clean_environment(env):
    """Keep build settings; discard inherited experiment and thread controls."""
    return {k:v for k,v in env.items() if not k.startswith(("BITZ_", "F2_", "RAYON_", "BD"))
            and k != "HARDWARE_CONCURRENCY"}


def run_checked(command, **kwargs):
    code, timed_out = run_process(command, **kwargs)
    if timed_out:
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
    if code:
        raise subprocess.CalledProcessError(code, command)
