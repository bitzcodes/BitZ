#!/usr/bin/env python3
"""Cooperative gate for timed benchmark campaigns on a shared box.

Serializes campaigns across sessions with a machine-wide lock, waits for
sustained CPU idle before starting (so another session's campaign or build
never overlaps a timed run), and aborts the campaign if swap growth passes a
limit (the runaway-paging guard; Binius paging cells run with a higher limit).

    python3 scripts/bench_gate.py run --label u32-bitz-r2-t10 \
        [--min-idle 88] [--hold-seconds 120] [--swap-grow-gb 12] -- command...

Exit code: the command's, or 86 if the swap guard aborted it. The lock is
`BITZ_BENCH_LOCK` (default /tmp/bitz-bench.lock), a mkdir lock holding the
owner pid; a lock whose owner is dead is reclaimed. Other sessions running
timed work should take the same lock.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

LOCK = Path(os.environ.get("BITZ_BENCH_LOCK", "/tmp/bitz-bench.lock"))
ABORTED = 86


def idle_percent() -> float:
    if sys.platform.startswith("linux"):
        def counters():
            values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
            return sum(values), values[3] + values[4]
        total0, idle0 = counters()
        time.sleep(1)
        total1, idle1 = counters()
        return 100.0 * (idle1 - idle0) / max(1, total1 - total0)
    if sys.platform != "darwin":
        raise SystemExit("bench_gate.py knows only the macOS idle probe")
    out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"],
                         capture_output=True, text=True, check=True).stdout
    lines = [line for line in out.splitlines() if line.startswith("CPU usage")]
    match = re.search(r"([\d.]+)% idle", lines[-1])
    if not match:
        raise SystemExit(f"unparsable top output: {lines[-1]!r}")
    return float(match.group(1))


def swap_used_gb() -> float:
    if sys.platform.startswith("linux"):
        values = {line.split(":", 1)[0]: int(line.split()[1])
                  for line in Path("/proc/meminfo").read_text().splitlines()}
        return (values["SwapTotal"] - values["SwapFree"]) / (1024 * 1024)
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True, check=True).stdout
    match = re.search(r"used = ([\d.]+)([MG])", out)
    if not match:
        raise SystemExit(f"unparsable vm.swapusage: {out!r}")
    value = float(match.group(1))
    return value / 1024 if match.group(2) == "M" else value


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire(label: str, poll_seconds: float) -> None:
    while True:
        try:
            LOCK.mkdir()
            (LOCK / "owner").write_text(f"{os.getpid()} {label}\n")
            return
        except FileExistsError:
            try:
                pid = int((LOCK / "owner").read_text().split()[0])
            except (OSError, ValueError, IndexError):
                pid = None
            if pid is not None and not pid_alive(pid):
                print(f"[bench_gate] reclaiming stale lock of dead pid {pid}", flush=True)
                shutil.rmtree(LOCK, ignore_errors=True)
                continue
            print(f"[bench_gate] {label}: waiting for {LOCK} (held by {pid})", flush=True)
            time.sleep(poll_seconds)


def release() -> None:
    shutil.rmtree(LOCK, ignore_errors=True)


def wait_idle(label: str, min_idle: float, hold_seconds: float, poll_seconds: float,
              max_wait_seconds: float | None = None) -> None:
    """Require `hold_seconds` of consecutive samples at or above `min_idle`."""
    needed = max(1, round(hold_seconds / poll_seconds))
    streak = 0
    started = time.monotonic()
    while streak < needed:
        if max_wait_seconds is not None and time.monotonic() - started >= max_wait_seconds:
            raise TimeoutError(f"{label}: host did not sustain {min_idle}% idle within {max_wait_seconds}s")
        idle = idle_percent()
        streak = streak + 1 if idle >= min_idle else 0
        print(f"[bench_gate] {label}: idle {idle:.0f}% (streak {streak}/{needed})", flush=True)
        if streak < needed:
            time.sleep(poll_seconds)


def terminate(signum, _frame):
    # Run the finally block before relinquishing the measurement lock.
    raise SystemExit(128 + signum)


def stop_process_group(process: subprocess.Popen) -> None:
    handlers = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # Reap the whole group even when the leader exits before its workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run", help="gate and run one campaign command")
    run.add_argument("--label", required=True)
    run.add_argument("--min-idle", type=float, default=88.0)
    run.add_argument("--hold-seconds", type=float, default=120.0)
    run.add_argument("--poll-seconds", type=float, default=20.0,
                     help="interval between idle samples and between lock attempts")
    run.add_argument("--max-idle-wait-seconds", type=float,
                     help="fail without running the campaign if the host cannot qualify in time")
    run.add_argument("--swap-grow-gb", type=float, default=12.0,
                     help="abort the campaign if swap grows past this over its baseline")
    run.add_argument("command", nargs=argparse.REMAINDER,
                     help="-- followed by the campaign command")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given after --")
    signal.signal(signal.SIGTERM, terminate)
    acquire(args.label, args.poll_seconds)
    process = None
    try:
        wait_idle(args.label, args.min_idle, args.hold_seconds, args.poll_seconds,
                  args.max_idle_wait_seconds)
        baseline = swap_used_gb()
        print(f"[bench_gate] {args.label}: starting (swap baseline {baseline:.2f} GB, "
              f"guard +{args.swap_grow_gb:.0f} GB)", flush=True)
        process = subprocess.Popen(command, start_new_session=True)
        aborted = threading.Event()

        def guard():
            while process.poll() is None:
                time.sleep(15)
                try:
                    grown = swap_used_gb() - baseline
                except SystemExit:
                    continue
                if grown > args.swap_grow_gb and process.poll() is None:
                    aborted.set()
                    print(f"[bench_gate] {args.label}: swap grew {grown:.1f} GB — aborting campaign",
                          flush=True)
                    os.killpg(process.pid, signal.SIGTERM)
                    time.sleep(20)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    return

        watchdog = threading.Thread(target=guard, daemon=True)
        watchdog.start()
        code = process.wait()
        return ABORTED if aborted.is_set() else (code if code >= 0 else 128 - code)
    finally:
        if process is not None:
            stop_process_group(process)
        release()


if __name__ == "__main__":
    sys.exit(main())
