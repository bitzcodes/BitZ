#!/usr/bin/env python3
"""Peak resident set and swap-out accounting for a sweep's child processes (macOS).

Runs COMMAND (typically the hybrid bench's `--sweep`) and samples `ps` while it
runs. Every child process whose arguments carry `--mode M --mul-log A
--sha-log B` (one case of the sweep, one process per case) gets one row in the
output TSV once it has exited:

    mode  mul_log  sha_log  pid  peak_mib  swapouts  swapouts_before  swapouts_after  compressions

`swapouts` is the growth of `vm_stat`'s Swapouts counter while the case ran (a
nonzero value marks a paged case); `compressions` is the growth of its
Compressions counter (pages the kernel compressed to make room — memory
pressure that inflates timings before any swap-out). Example:

    python3 scripts/rss_sampler.py --output PerfRuns/<stamp>/peak-rss-and-swap.tsv -- \
        target/release/deps/hybrid_u32_sha256-<hash> --sweep --mode hybrid --iterations 11 \
        --results-dir PerfRuns/<stamp>/hybrid
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

ARGS_RE = re.compile(r"--mode\s+(\S+).*--mul-log\s+(\d+).*--sha-log\s+(\d+)")


def counters() -> tuple[int, int]:
    """(Swapouts, Compressions) of `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    swap = re.search(r"Swapouts:\s+(\d+)", out)
    comp = re.search(r"Compressions:\s+(\d+)", out)
    return (int(swap.group(1)) if swap else 0, int(comp.group(1)) if comp else 0)


def discover(exe_name: str) -> dict[int, tuple[str, int, int]]:
    """pid -> (mode, mul_log, sha_log) of the sweep's case processes."""
    found = {}
    out = subprocess.run(["pgrep", "-f", f"{exe_name}.*--mul-log"], capture_output=True, text=True).stdout
    for token in out.split():
        pid = int(token)
        args = subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True).stdout
        match = ARGS_RE.search(args)
        if match is not None:
            found[pid] = (match.group(1), int(match.group(2)), int(match.group(3)))
    return found


def rss_kib(pids: list[int]) -> dict[int, int]:
    """Resident set (KiB) of the given pids; missing pids have exited."""
    if not pids:
        return {}
    out = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True).stdout
    sizes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sizes[int(parts[0])] = int(parts[1])
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="TSV to write (one row per finished case)")
    parser.add_argument("--interval", type=float, default=0.2, help="sampling period in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the sweep command (after `--`)")
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command")
    exe_name = command[0].rsplit("/", 1)[-1]
    child = subprocess.Popen(command)
    live: dict[int, dict] = {}
    with open(args.output, "w") as out:
        out.write("mode\tmul_log\tsha_log\tpid\tpeak_mib\tswapouts\tswapouts_before\tswapouts_after\tcompressions\n")
        out.flush()
        while True:
            # Discovery (a process-table scan) only while no case is live —
            # one case runs at a time — then cheap per-pid RSS reads, so the
            # sampler does not contend with a multi-gigabyte child.
            if not live:
                for pid, (mode, mul_log, sha_log) in discover(exe_name).items():
                    live[pid] = {"mode": mode, "mul_log": mul_log, "sha_log": sha_log, "peak": 0, "before": counters()}
            current = rss_kib(list(live))
            for pid, rss in current.items():
                live[pid]["peak"] = max(live[pid]["peak"], rss)
            for pid in [p for p in live if p not in current]:
                entry = live.pop(pid)
                after = counters()
                before = entry["before"]
                out.write(
                    f"{entry['mode']}\t{entry['mul_log']}\t{entry['sha_log']}\t{pid}\t{entry['peak'] // 1024}\t"
                    f"{after[0] - before[0]}\t{before[0]}\t{after[0]}\t{after[1] - before[1]}\n"
                )
                out.flush()
            if child.poll() is not None and not live:
                break
            time.sleep(args.interval)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
