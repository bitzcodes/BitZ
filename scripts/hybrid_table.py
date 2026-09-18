#!/usr/bin/env python3
"""LaTeX table for the hybrid mod-2^32 multiplication + chained SHA-256 benchmark.

2026-09-13 bench-suite methodology: six schemes — the BitZ hybrid at opener
rates 1/2 and 1/8, all-Binius64 at FRI rates 1/2 and 1/8, and the all-Binius
circuit with the BitZ opener (round-by-round accounting) at rates 1/2 and 1/8 —
each measured at 1 and 10 threads. One sweep directory per (scheme, threads):

    python3 scripts/hybrid_table.py --variant counts \
        --row hybrid@1:1=PerfRuns/<dir> --row hybrid@1:10=PerfRuns/<dir> \
        --row all-binius@3:10=PerfRuns/<dir> ...

Row keys are `<mode>@<log_inv_rate>:<threads>` with mode one of `hybrid`,
`all-binius`, `binius-ligerito` and log_inv_rate 1 (rate 1/2) or 3 (rate 1/8).
Each directory must be a `benches/hybrid_u32_sha256` sweep result
(`summary.csv`, per-case logs, `run.txt` with the recorded knobs, optionally
`peak-rss-and-swap.tsv` from scripts/rss_sampler.py); the recorded mode, rate,
accounting and RAYON_NUM_THREADS are validated against the key, so runs made
before the knobs were recorded are rejected rather than mislabeled.

Output: one row group per shape (N multiplications, M compressions),
sub-grouped by thread count, one row per scheme; bold = best of the schemes in
that (shape, threads) group and column; dagger = the case paged.
`--variant witness` (M = N/256, outputs/tables/hybrid-table.tex) or
`--variant counts` (M = N, outputs/tables/hybrid-table-equal-counts.tex).
"""
from __future__ import annotations

import argparse
from local_provenance import root_metadata
import csv
import datetime
import json
import platform
import statistics
import subprocess
from pathlib import Path

# Table rows in display order: (mode, log_inv_rate) -> LaTeX label.
# Naming rules (user directive 2026-09-13): the BitZ rows are \ftwoz-SNARK
# (never "(this work)"), and no \cite{...} after a scheme name in the table.
ROWS = [
    (("hybrid", 1), r"\ftwoz-SNARK, rate $1/2$"),
    (("hybrid", 3), r"\ftwoz-SNARK, rate $1/8$"),
    (("all-binius", 1), r"Binius (UDR), rate $1/2$"),
    (("all-binius", 3), r"Binius (UDR), rate $1/8$"),
    (("binius-ligerito", 1), r"Binius (Johnson), rate $1/2$"),
    (("binius-ligerito", 3), r"Binius (Johnson), rate $1/8$"),
]

SCHEMES = (
    r" \ftwoz-SNARK proves the multiplications with Spartan over a transcript-sampled prime and the SHA-256 chain with the Binius64 PIOP, and discharges both through one shared \ftwoz\ opening (Johnson-regime Ligerito with the out-of-domain Round~0, at the row's rate); its whole-protocol union bound is gated at $\lambda = 100$. Binius (UDR) proves the same SHA-256 chain and a native four-limb multiplication gadget in one Binius64 proof with ring switching and FRI at the row's rate and a $100$-bit query-phase target. Binius (Johnson) proves the same all-Binius circuit with Binius64's PIOP and the \ftwoz\ opener at the row's rate, every round-by-round error term gated at $100$ bits on its own."
    r" \emph{Prover} includes witness synthesis and the commitments; \emph{verifier} includes decoding; \emph{peak mem.} is the high-water resident set of the proving process ($1$\,GB $= 2^{30}$ bytes); $^{\dagger}$ marks cases that paged. Medians of the verified runs."
)

VARIANTS = {
    # Equal packed witnesses: one packed 128-bit word per multiplication,
    # 256 per compression, so M = N/256.
    "witness": {
        "output": "outputs/tables/hybrid-table.tex",
        "label": "tab:hybrid-sha256-mul",
        "title": "Hybrid u32-multiplication + chained SHA-256 comparison",
        "caption": r"End-to-end proofs of $N$ multiplications $x \cdot y = z + 2^{32} w$ of $32$-bit integers together with $M = N/256$ chained SHA-256 compressions (equal packed witnesses for the two branches)."
        + SCHEMES,
    },
    # Equal operation counts: N = M, the SHA-256 branch's packed witness is
    # 256 times the multiplication branch's.
    "counts": {
        "output": "outputs/tables/hybrid-table-equal-counts.tex",
        "label": "tab:hybrid-sha256-mul-equal-counts",
        "title": "Hybrid u32-multiplication + chained SHA-256 comparison at equal operation counts N = M",
        "caption": r"End-to-end proofs of $N$ multiplications $x \cdot y = z + 2^{32} w$ of $32$-bit integers together with $M = N$ chained SHA-256 compressions (equal operation counts; the packed SHA-256 witness is $256\times$ the multiplication witness, so the workload is dominated by the compressions)."
        + SCHEMES,
    },
}

KEYS = {f"{mode}@{rate}": (mode, rate) for (mode, rate), _ in ROWS}


class RunError(SystemExit):
    pass


def parse_run_txt(base: Path) -> dict[str, str]:
    run = base / "run.txt"
    if not run.exists():
        raise RunError(f"{base}: missing run.txt (not a sweep results directory)")
    out = {}
    for line in run.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def decode_hex_json(encoded: str) -> dict:
    return json.loads(bytes.fromhex(encoded))


def knob(meta: dict[str, str], name: str) -> str | None:
    """A recorded knob; sweeps measured before the BitZ rename record F2Z_* names."""
    value = meta.get(f"BITZ_{name}")
    return value if value is not None else meta.get(f"F2Z_{name}")


def validate(base: Path, mode: str, rate: int, threads: int, meta: dict[str, str]) -> None:
    """The recorded knobs must match the row key; unrecorded runs are rejected."""
    if mode not in meta.get("modes", ""):
        raise RunError(f"{base}: run.txt modes {meta.get('modes')!r} do not include {mode!r}")
    recorded_threads = meta.get("RAYON_NUM_THREADS", "default")
    if recorded_threads != str(threads):
        raise RunError(
            f"{base}: RAYON_NUM_THREADS={recorded_threads!r} but the row declares {threads} "
            "threads (runs must record an explicit thread count)"
        )
    if mode == "all-binius":
        recorded = knob(meta, "HYBRID_BINIUS_LOG_INV_RATE")
        if recorded is None:
            raise RunError(
                f"{base}: run.txt does not record BITZ_HYBRID_BINIUS_LOG_INV_RATE; "
                "re-measure with the updated sweep"
            )
        if recorded != str(rate):
            raise RunError(f"{base}: all-Binius rate {recorded} does not match key rate {rate}")
    elif mode == "binius-ligerito":
        recorded = knob(meta, "BINIUS_LOG_INV_RATE")
        accounting = knob(meta, "BINIUS_LIGERITO_ACCOUNTING")
        if recorded is None or accounting is None:
            raise RunError(
                f"{base}: run.txt does not record the BitZ-opener knobs; re-measure with the updated sweep"
            )
        if recorded != str(rate):
            raise RunError(f"{base}: BitZ-opener rate {recorded} does not match key rate {rate}")
        if accounting != "round-by-round":
            raise RunError(
                f"{base}: BitZ-opener accounting {accounting!r}; the suite requires round-by-round (rbr)"
            )
    else:  # hybrid
        profile = meta.get("profile")
        expected = f"custom:{rate}:4"
        if profile != expected:
            raise RunError(f"{base}: hybrid profile {profile!r} does not match {expected!r}")


def validate_identity(row: dict, base: Path, mode: str, rate: int) -> None:
    """Per-sample cross-check of the identity recorded in summary.csv."""
    encoded = (row.get("ligerito_hex") or "").strip()
    if mode == "all-binius":
        if encoded:
            raise RunError(f"{base}: all-Binius row unexpectedly carries a Ligerito identity")
        return
    if not encoded:
        raise RunError(f"{base}: {mode} row is missing its Ligerito identity")
    report = decode_hex_json(encoded)
    if mode == "hybrid":
        resolved = report.get("resolved_profile", "")
        if resolved != f"custom:{rate}:4":
            raise RunError(f"{base}: resolved opener profile {resolved!r} does not match rate {rate}")
    else:  # binius-ligerito
        if report.get("log_inv_rate") != rate:
            raise RunError(f"{base}: opener identity rate {report.get('log_inv_rate')} != {rate}")
        if report.get("accounting") != "round-by-round":
            raise RunError(f"{base}: opener identity accounting {report.get('accounting')!r}; require round-by-round")


def medians(base: Path, mode: str, rate: int) -> dict:
    """Per-shape medians of one sweep directory, sampler peaks included."""
    if not (base / "summary.csv").exists() and (base / mode / "summary.csv").exists():
        base = base / mode
    summary = base / "summary.csv"
    if not summary.exists():
        raise RunError(f"{base}: missing summary.csv")
    by_shape: dict[tuple[int, int], list[dict]] = {}
    with open(summary) as f:
        for row in csv.DictReader(f):
            if row["mode"] != mode:
                continue
            validate_identity(row, base, mode, rate)
            by_shape.setdefault((int(row["mul_log"]), int(row["sha_log"])), []).append(row)
    if not by_shape:
        raise RunError(f"{base}: no {mode} rows in summary.csv")
    peaks = {}
    tsv = base / "peak-rss-and-swap.tsv"
    if tsv.exists():
        with open(tsv) as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) < 6 or parts[0] != mode or not parts[1].isdigit():
                    continue
                compressions = int(parts[8]) if len(parts) > 8 and parts[8].strip().isdigit() else 0
                peaks[(int(parts[1]), int(parts[2]))] = (int(parts[4]), int(parts[5]), compressions)
    out = {}
    for shape, rows in by_shape.items():
        bits = None
        log = base / f"{mode}-m{shape[0]}-s{shape[1]}.log"
        if log.exists():
            for line in log.read_text().splitlines():
                if "algebraic_security_bits=" in line:
                    bits = float(line.split("algebraic_security_bits=")[1].split()[0])
        peak, swapouts, compressions = peaks.get(shape, (None, 0, 0))
        out[shape] = {
            "prover": statistics.median(float(r["total_prover_ms"]) for r in rows),
            "verify": statistics.median(float(r["verify_ms"]) for r in rows),
            "proof": int(statistics.median(int(r["proof_bytes"]) for r in rows)),
            "peak_mib": peak,
            "swapouts": swapouts,
            "compressions": compressions,
            # The dagger keeps the paper's definition (the case paged); page
            # compressions are reported in the header for transparency (they
            # can happen during setup without touching the timed proves).
            "pressured": swapouts > 0,
            "bits": bits,
            "samples": len(rows),
        }
    return out


def sig3(x: float) -> str:
    """Three significant figures, as the paper's tables print them."""
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def parse_row(value: str) -> tuple[tuple[str, int], int, Path]:
    head, _, directory = value.partition("=")
    key, _, threads = head.partition(":")
    if key not in KEYS or not threads.isdigit() or not directory:
        raise RunError(
            f"--row {value!r}: expected <mode>@<rate>:<threads>=<dir> with mode@rate one of "
            + ", ".join(sorted(KEYS))
        )
    return KEYS[key], int(threads), Path(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        metavar="MODE@RATE:THREADS=DIR",
        help="one sweep directory per (scheme, threads), e.g. hybrid@1:10=PerfRuns/run-a",
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="witness", help="which table: equal packed witnesses (M = N/256) or equal operation counts (N = M)")
    parser.add_argument("--output", type=Path, help="defaults to the variant's file under outputs/tables/")
    args = parser.parse_args()
    if not args.row:
        parser.error("at least one --row MODE@RATE:THREADS=DIR is required")
    variant = VARIANTS[args.variant]
    output = args.output or Path(variant["output"])
    # data[(mode, rate)][threads][shape] = medians
    data: dict[tuple[str, int], dict[int, dict]] = {}
    sources: list[str] = []
    for value in args.row:
        (mode, rate), threads, directory = parse_row(value)
        if threads in data.get((mode, rate), {}):
            raise RunError(f"duplicate --row for {mode}@{rate}:{threads}")
        meta = parse_run_txt(directory)
        validate(directory, mode, rate, threads, meta)
        data.setdefault((mode, rate), {})[threads] = medians(directory, mode, rate)
        sources.append(f"{mode}@{rate}:{threads}={directory}")
    thread_counts = sorted({t for by in data.values() for t in by})
    shapes = sorted({shape for by in data.values() for rows in by.values() for shape in rows})
    source = root_metadata()
    rev, dirty = source["revision"], source["git_dirty"]
    lines = [
        f"% {variant['title']} — GENERATED FILE, do not edit by hand.",
        f"% Generated by scripts/hybrid_table.py on {datetime.date.today()} at {rev}{'-dirty' if dirty else ''} from:",
    ]
    lines += [f"%   {source}" for source in sources]
    lines += [
        "%   benches/hybrid_u32_sha256 medians of the verified iterations, one process per case; recorded",
        "%   mode/rate/accounting/threads validated against each row key (run.txt + summary identities).",
        "% Rows: \\ftwoz-SNARK = hybrid (shared Johnson opener at the row's rate, Round 0, 100-bit whole-protocol union bound;",
        "%   rate 1/2 keeps the 106-bit component target, rate 1/8 solves the smallest in 100..=112);",
        "%   Binius64 = all-Binius circuit with ring switching + FRI at the row's rate (100-bit query-phase target);",
        "%   Binius64 + BitZ opener = the same circuit, every oracle committed/opened by the BitZ opener at the row's rate,",
        "%   round-by-round accounting gated at 100 bits.",
        f"% Machine: {platform.machine()} {platform.platform()}. Columns: prover = total_prover_ms (witness synthesis, commitments, PIOPs, opening, encoding; setup excluded);",
        "%   verifier = verify_ms; proof = proof_bytes, KB = 1000 bytes; peak mem. = external RSS sample of the child process, GB = 2^30 bytes.",
        "% Bold = best of the schemes for that shape and column (each thread count is its own column); dagger = the case paged (swap-outs > 0). Raw medians:",
    ]
    for shape in shapes:
        for threads in thread_counts:
            for (mode, rate), _ in ROWS:
                row = data.get((mode, rate), {}).get(threads, {}).get(shape)
                if row is None:
                    continue
                bits = f" bits={row['bits']:.2f}" if row["bits"] is not None else ""
                peak = row["peak_mib"] if row["peak_mib"] is not None else "?"
                lines.append(
                    f"%   2^{shape[0]}:2^{shape[1]:<3} t={threads:<3} {mode}@{rate:<2} prover={row['prover']:<8.1f} verify={row['verify']:<7.2f} "
                    f"proof_bytes={row['proof']} peak_mib={peak} swapouts={row['swapouts']} compressions={row['compressions']} samples={row['samples']}{bits}"
                )
    lines += [
        "",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}rrl" + "r" * (2 * len(thread_counts) + 2) + "@{}}",
        r"    \toprule",
        r"     & & & \multicolumn{%d}{c}{Prover (ms)} & \multicolumn{%d}{c}{Verifier (ms)} & Proof & Peak mem. \\" % (len(thread_counts), len(thread_counts)),
        r"    \cmidrule(lr){4-%d} \cmidrule(lr){%d-%d}" % (3 + len(thread_counts), 4 + len(thread_counts), 3 + 2 * len(thread_counts)),
        r"    $N$ & $M$ & Scheme & " + " & ".join(f"{t} thr" for t in thread_counts) + " & " + " & ".join(f"{t} thr" for t in thread_counts) + r" & (KB) & (GB) \\",
        r"    \midrule",
    ]
    ref_t = max(thread_counts)
    order_t = [ref_t] + [t for t in thread_counts if t != ref_t]
    cols = ([("prover", t, sig3) for t in thread_counts] + [("verify", t, sig3) for t in thread_counts]
            + [("proof", None, lambda v: f"{v / 1000:.0f}"), ("peak_mib", None, lambda v: sig3(v / 1024))])
    first_group = True
    for shape in shapes:
        present = [((mode, rate), label) for (mode, rate), label in ROWS
                   if any(shape in data.get((mode, rate), {}).get(t, {}) for t in thread_counts)]
        if not present:
            continue
        if not first_group:
            lines.append(r"    \addlinespace")
        first_group = False
        def pick(key, t):
            for u in (order_t if t is None else [t]):
                r = data.get(key, {}).get(u, {}).get(shape)
                if r is not None:
                    return r
            return None
        grid = {}
        for key, _ in present:
            row_cells = []
            for metric, t, _ in cols:
                r = pick(key, t)
                row_cells.append((None if r is None else r[metric], bool(r is not None and r["pressured"])))
            grid[key] = row_cells
        best = []
        for ci, (_, _, fmt) in enumerate(cols):
            vals = [grid[k][ci][0] for k, _ in present if grid[k][ci][0] is not None]
            best.append(fmt(min(vals)) if len(vals) > 1 else None)
        for j, (key, label) in enumerate(present):
            out_cells = []
            for ci, (metric, t, fmt) in enumerate(cols):
                v, pressured = grid[key][ci]
                if v is None:
                    out_cells.append("--")
                    continue
                text = fmt(v)
                cell = rf"\textbf{{{text}}}" if best[ci] is not None and text == best[ci] else text
                if pressured and metric in ("prover", "verify", "peak_mib"):
                    cell += r"$^{\dagger}$"
                out_cells.append(cell)
            head = rf"    $2^{{{shape[0]}}}$ & $2^{{{shape[1]}}}$" if j == 0 else "     &"
            lines.append(f"{head} & {label} & " + " & ".join(out_cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{variant['caption']} Prover and verifier times are given for {' and '.join(map(str, thread_counts))} threads; proof size does not depend on the thread count, and peak memory is from the {max(thread_counts)}-thread runs.}}",
        rf"  \label{{{variant['label']}}}",
        r"\end{table}",
    ]
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {output} ({len(shapes)} shapes x {len(thread_counts)} thread counts)")


if __name__ == "__main__":
    main()
