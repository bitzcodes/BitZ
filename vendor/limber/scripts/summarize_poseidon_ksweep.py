#!/usr/bin/env python3
"""Summarize a Poseidon2 KSWEEP run (plan §6, combined circuit).

Reads Criterion's `estimates.json` (median point estimate with its 95%
interval — there is no separate timing loop), REQUIRES and joins the
`ksweep-metadata.json` sidecar the benchmark wrote, and stages a CSV plus
a tie-rule decision JSON into the run directory. The sweep measures
`(backend, k)` on the ONE combined mixed-modulus circuit:

  score(k)  = the combined prove_e2e median
  min_score = min_k score(k)
  tie_band  = {k | score(k) <= 1.03 * min_score}
  winner    = min(tie_band)

Failed candidates stay in the CSV (admissible = false, no timings).
Invoked automatically by scripts/run_poseidon_bench.sh after a successful
sweep; usage: summarize_poseidon_ksweep.py <run-dir>.
"""

from __future__ import annotations

import csv
import json
import os
import sys


def collect_estimates(run_dir: str) -> dict[str, tuple[float, float, float]]:
    """Map original `function_id` -> (median, ci_lo, ci_hi) in ns.

    Criterion sanitizes IDs into flat directory names, so directories are
    matched through each benchmark.json's recorded `function_id` rather
    than by reconstructing the sanitization.
    """
    out: dict[str, tuple[float, float, float]] = {}
    base = os.path.join(run_dir, "criterion", "prove_e2e")
    for entry in sorted(os.listdir(base)):
        bench_path = os.path.join(base, entry, "new", "benchmark.json")
        est_path = os.path.join(base, entry, "new", "estimates.json")
        if not (os.path.exists(bench_path) and os.path.exists(est_path)):
            continue
        with open(bench_path, encoding="utf-8") as f:
            function_id = json.load(f)["function_id"]
        with open(est_path, encoding="utf-8") as f:
            med = json.load(f)["median"]
        out[function_id] = (
            med["point_estimate"],
            med["confidence_interval"]["lower_bound"],
            med["confidence_interval"]["upper_bound"],
        )
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = sys.argv[1]

    # The metadata sidecar is required, not optional.
    with open(os.path.join(run_dir, "ksweep-metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(run_dir, "run-config.json"), encoding="utf-8") as f:
        cfg = json.load(f)["protocol"]
    backend = meta["backend"]
    hashes = meta["hashes_per_field"]
    total = meta["total_hashes"]
    dims = cfg["dims"]
    lc = dims["num_cons"].bit_length() - 1
    lv = dims["num_vars"].bit_length() - 1
    with open(os.path.join(run_dir, "run-config.sha256"), encoding="utf-8") as f:
        config12 = f.read().strip()[:12]

    estimates = collect_estimates(run_dir)
    rows = []
    scores: dict[int, float] = {}
    for k in meta["k_order"]:
        entry = meta["candidates"][f"k{k}"]
        median = lo = hi = None
        if entry["admissible"]:
            function_id = (
                f"{backend}/mixed3/Hpf{hashes}-total{total}/c2^{lc}v2^{lv}/k{k}/cfg-{config12}"
            )
            if function_id not in estimates:
                raise SystemExit(f"no Criterion estimates for {function_id}")
            median, lo, hi = estimates[function_id]
            scores[k] = median
        ps = entry["proof_size"]
        rows.append(
            {
                "backend": backend,
                "circuit": "mixed3",
                "k": k,
                "admissible": entry["admissible"],
                "error": entry["error"] or "",
                "log_p": entry["log_p"],
                "s": entry["s"],
                "median_ns": median,
                "ci95_lower_ns": lo,
                "ci95_upper_ns": hi,
                "input_commitments_bytes": ps["input_commitments"],
                "eval_arg_bytes": ps["eval_arg"],
                "sumcheck_remainder_bytes": ps["sumcheck_remainder"],
            }
        )

    if not scores:
        print("summarize_poseidon_ksweep: no admissible k", file=sys.stderr)
        return 1

    min_score = min(scores.values())
    tie_band = sorted(k for k, s in scores.items() if s <= 1.03 * min_score)
    winner = tie_band[0]

    csv_path = os.path.join(run_dir, "ksweep-summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    decision = {
        "backend": backend,
        "circuit": "mixed3",
        "rule": "combined median; tie band 1.03x; winner = min k in band",
        "scores_ns": {str(k): scores[k] for k in sorted(scores)},
        "min_score_ns": min_score,
        "tie_band": tie_band,
        "winner_k": winner,
    }
    with open(os.path.join(run_dir, "ksweep-decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"ksweep summary: {csv_path}")
    print(f"ksweep decision: winner k = {winner} (tie band {tie_band})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
