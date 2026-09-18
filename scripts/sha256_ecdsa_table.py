#!/usr/bin/env python3
"""LaTeX table of absolute numbers from a SHA-256 + P-256 ECDSA campaign (scripts/run_sha256_ecdsa_compare.py).

Reads the `*.result.json` records of one or more campaign output directories
(later directories override earlier ones for the same case) and writes a
self-documenting `outputs/tables/sha256-ecdsa-table.tex`: one row group per
(security target, thread count), one row per scheme, with the medians of
witness generation, the complete prover (commitment included) and the complete
verifier, plus the complete proof material and the worker's peak memory.

    python3 scripts/sha256_ecdsa_table.py bench_results/sha256-ecdsa-i7-bitz-binius
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import subprocess
from pathlib import Path

from local_provenance import root_metadata

SCHEMA = "bitz/sha256-ecdsa-compare/v1"
# Table rows are (scheme id, LaTeX label), in display order. A scheme id is
# `method@log_inv_rate`: the BitZ rows split by their Ligerito profile's
# level-0 rate, the Binius-family rows by their commitment rate. Naming
# follows the 2026-09-13 directive: no `(this work)`, no \cite after names.
SCHEMES = [
    ("bitz-split@1", "\\ftwoz-SNARK, rate $1/2$"),
    ("bitz-split@3", "\\ftwoz-SNARK, rate $1/8$"),
    ("binius64@1", "Binius (UDR), rate $1/2$"),
    ("binius64@3", "Binius (UDR), rate $1/8$"),
    ("binius64-ligerito@1", "Binius (Johnson), rate $1/2$"),
    ("binius64-ligerito@3", "Binius (Johnson), rate $1/8$"),
]
PLACEHOLDER = "--"


def scheme_id(case: dict, sample: dict) -> str:
    """`method@rate` for suite methods; the bare method otherwise."""
    method = case["method"]
    security = sample.get("security") or {}
    if method.startswith("bitz"):
        lig = security.get("ligerito") or {}
        levels = (lig.get("configuration") or lig).get("levels") or [{}]
        return f"{method}@{int(levels[0].get('log_inv_rate', 1))}"
    if method.startswith("binius64"):
        rate = case.get("log_inv_rate", security.get("log_inv_rate", 1))
        return f"{method}@{int(rate)}"
    return method


def fmt_ms(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_gb(peak_rss_bytes: float) -> str:
    v = peak_rss_bytes / (1 << 30)
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def fmt_kb(b: float) -> str:
    return f"{b / 1000:.0f}"


def probe(cmd: list[str], default: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip() or default
    except Exception:
        return default


def load_cases(directory: Path) -> dict:
    """Complete cases keyed by (log_compressions, security_target, threads, method)."""
    result = {}
    for path in sorted(directory.glob("*.result.json")):
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            continue
        case = record["case"]
        samples = [r for r in record["rows"] if r.get("trial") == "sample"]
        if not samples or any(r.get("schema") != SCHEMA or r.get("verified") is not True for r in samples):
            raise ValueError(f"{path}: complete case without verified samples")
        key = (case["log_compressions"], case["security_target"], case["threads"], scheme_id(case, samples[0]))
        if key in result:
            raise ValueError(f"duplicate case {key} in {directory}")
        med = lambda k: statistics.median(r[k] for r in samples)  # noqa: E731
        result[key] = dict(
            key=key, run_dir=str(directory), file=path.name, samples=len(samples),
            seed=case["seed"], compressions=samples[0]["compressions"], message_bytes=samples[0]["message_bytes"],
            fixture_id=samples[0]["fixture_id"], security=samples[0].get("security", {}),
            circuit=samples[0].get("circuit"), peak_rss_bytes=record["peak_rss_bytes"],
            witness_ms=med("witness_ms"), prove_ms=med("prove_ms"), e2e_prover_ms=med("e2e_prover_ms"),
            verify_ms=med("verify_ms"), proof_bytes=med("proof_material_bytes"), setup_ms=med("setup_ms"),
            opening_ms=med("opening_ms") if all(isinstance(r.get("opening_ms"), (int, float)) for r in samples) else None,
        )
    return result


def bitz_security(row: dict) -> str:
    """One caption clause for an BitZ row's Ligerito policy at level 0."""
    lig = row["security"].get("ligerito", {})
    levels = (lig.get("configuration") or lig).get("levels") or []
    if not levels:
        return "Lambda%d" % row["key"][1]
    top = levels[0]
    regime = "Johnson" if str(top.get("regime", "")).startswith("johnson") else "unique decoding radius"
    ood = "early Round-0 OOD" if str(top.get("regime", "")).endswith("ood") else "no OOD"
    return (f"target {row['key'][1]} bits: Ligerito {regime}, rate $1/{1 << int(top.get('log_inv_rate', 1))}$, "
            f"{top.get('queries')} level-0 queries, {top.get('grinding_bits', 0)} bits of grinding, "
            f"{top.get('fold_grinding_bits', 0)} bits of fold grinding, {ood}; {row['security'].get('economic_bits', 0):.1f} bits achieved")


def binius_security(row: dict) -> str:
    s = row["security"]
    return (f"target {s.get('fri_query_target_bits')} bits: {s.get('fri_queries')} queries at rate "
            f"$1/{1 << int(s.get('log_inv_rate', 1))}$")


def opener_security(row: dict) -> str:
    """One caption clause per binius64-ligerito rate."""
    s = row["security"]
    return (f"rate $1/{1 << int(s.get('log_inv_rate', 1))}$: opener component target {s.get('component_bits')} bits, "
            f"{s.get('round_by_round_bits', 0):.1f} bits achieved")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    ap.add_argument("--out", type=Path, default=Path("outputs/tables/sha256-ecdsa-table.tex"))
    ap.add_argument("--label", default="tab:sha256-ecdsa")
    args = ap.parse_args()

    by = {}
    for run_dir in args.run_dirs:
        by.update(load_cases(run_dir))
    if not by:
        raise SystemExit("no complete cases found")
    fixtures = {}
    for row in by.values():
        fixtures.setdefault((row["key"][0], row["seed"]), set()).add(row["fixture_id"])
    if any(len(ids) != 1 for ids in fixtures.values()):
        raise SystemExit("cases at one size and seed used different fixtures")

    exponents = sorted({k[0] for k in by})
    targets = sorted({k[1] for k in by if k[1] is not None})
    threads = sorted({k[2] for k in by})
    methods = [m for m, _ in SCHEMES if any(k[3] == m for k in by)]
    show_n, show_t, show_th = len(exponents) > 1, len(targets) > 1, len(threads) > 1
    lead = ([("$N$", "r")] if show_n else []) + ([("$\\lambda$ (bits)", "r")] if show_t else []) + ([("Threads", "r")] if show_th else [])
    metrics = [("witness_ms", "Witgen (ms)", fmt_ms), ("prove_ms", "Prover (ms)", fmt_ms), ("verify_ms", "Verifier (ms)", fmt_ms),
               ("proof_bytes", "Proof (KB)", fmt_kb), ("peak_rss_bytes", "Peak mem.\\ (GB)", fmt_gb)]

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    source = root_metadata()
    rev = source["revision"]
    dirty = "+dirty" if source["git_dirty"] else ""
    machine = probe(["sysctl", "-n", "machdep.cpu.brand_string"], platform.processor() or platform.machine())
    for run_dir in args.run_dirs:
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            cpu = json.loads(manifest.read_text()).get("cpu")
            if cpu and cpu not in ("arm", "arm64", "x86_64", "i386"):
                machine = cpu
    dirs = " ".join(str(d) for d in args.run_dirs)
    header = [
        "% SHA-256 chain + P-256 ECDSA comparison (BitZ vs Binius64 vs Binius64 with the BitZ opener) — GENERATED FILE, do not edit by hand.",
        f"% Generated by scripts/sha256_ecdsa_table.py on {stamp} (UTC) at {rev}{dirty} from {dirs}",
        "%   (benches/sha256_ecdsa_compare.rs + benchmarks/binius64, driven by scripts/run_sha256_ecdsa_compare.py;",
        "%   see README.md; later run directories override earlier ones per case).",
        "% Regenerate (from the repo root; this file is overwritten):",
        f"%   python3 scripts/sha256_ecdsa_table.py {dirs}",
        f"% Machine: {machine}; medians of the measured samples after one warm-up; one worker process per case.",
        "% Columns: witgen = witness_ms (witness generation from the message and signature; the binius64 rows count their",
        "%   witness packing here, while the binius64-ligerito rows pack inside their prover, so their witgen is the wire",
        "%   assignment alone); prover = prove_ms = commit_ms + protocol_ms (the complete prover after witness generation:",
        "%   commitment + PIOP + PCS opening); verifier = verify_ms (statement validation + verification of the decoded proof);",
        "%   proof = proof_material_bytes (complete proof material incl. commitments and auxiliary public inputs, statement",
        "%   excluded), KB = 1000 bytes; peak mem. = peak_rss_bytes of the whole worker process (setup, warm-up and all",
        "%   samples included), GB = 2^30 bytes.",
        "% Scheme ids are method@log_inv_rate (@1 = rate 1/2, @3 = rate 1/8); binius64-ligerito rows are gated round-by-round at 100 bits.",
        "% Bold = best of the schemes for that (size, target) group and column (each thread count is its own column).",
        "% Medians as recorded (ms unless noted):",
    ]
    for key in sorted(by, key=lambda k: (k[0], k[1] if k[1] is not None else -1, k[2], methods.index(k[3]))):
        r = by[key]
        header.append(
            f"%   2^{key[0]} target={key[1]} threads={key[2]} {key[3]:<22} witness={r['witness_ms']:.3f} prove={r['prove_ms']:.3f} "
            f"e2e_prover={r['e2e_prover_ms']:.3f} verify={r['verify_ms']:.3f} setup={r['setup_ms']:.3f} "
            f"opening={r['opening_ms'] if r['opening_ms'] is None else round(r['opening_ms'], 3)} proof_bytes={int(r['proof_bytes'])} "
            f"peak_rss={r['peak_rss_bytes']} samples={r['samples']} file={r['file']}")
    circuits = {}
    for key in sorted(by):
        circuits.setdefault(key[3], json.dumps(by[key].get("circuit"), sort_keys=True))
    for method, circuit in circuits.items():
        header.append(f"%   circuit {method}: {circuit}")

    ref_th = max(threads)
    order_th = [ref_th] + [x for x in threads if x != ref_th]
    lead = ([("$N$", "r")] if show_n else []) + ([("$\\lambda$ (bits)", "r")] if show_t else [])
    span = len(threads)
    thr = " & ".join(f"{x} thr" for x in threads)
    c0 = len(lead) + 3
    lines = ["", "\\begin{table}[H]", "  \\centering", "  \\small", "  \\setlength{\\tabcolsep}{4pt}",
             "  \\begin{tabular}{@{}" + "".join(a for _, a in lead) + "l" + "r" * (1 + 2 * span + 2) + "@{}}", "    \\toprule",
             "    " + " & ".join([""] * len(lead) + ["", "Witgen"]) + f" & \\multicolumn{{{span}}}{{c}}{{Prover (ms)}} & \\multicolumn{{{span}}}{{c}}{{Verifier (ms)}} & Proof & Peak mem. \\\\",
             f"    \\cmidrule(lr){{{c0}-{c0 + span - 1}}} \\cmidrule(lr){{{c0 + span}-{c0 + 2 * span - 1}}}",
             "    " + " & ".join([h for h, _ in lead] + ["Scheme", "(ms)"]) + f" & {thr} & {thr} & (KB) & (GB) \\\\", "    \\midrule"]
    cols = ([("witness_ms", None, fmt_ms)] + [("prove_ms", x, fmt_ms) for x in threads]
            + [("verify_ms", x, fmt_ms) for x in threads] + [("proof_bytes", None, fmt_kb), ("peak_rss_bytes", None, fmt_gb)])
    first_group = True
    for n in exponents:
        first_n = True
        for t in targets + ([None] if any(k[1] is None for k in by) else []):
            present = [m for m in methods if any((n, t, x, m) in by for x in threads)]
            if not present:
                continue
            if not first_group:
                lines.append("    \\addlinespace")
            first_group = False
            def value(m, k, x):
                for y in (order_th if x is None else [x]):
                    r = by.get((n, t, y, m))
                    if r is not None:
                        return r.get(k)
                return None
            grid = {m: [value(m, k, x) for k, x, _ in cols] for m in present}
            best = []
            for ci, (_, _, f) in enumerate(cols):
                vals = [grid[m][ci] for m in present if grid[m][ci] is not None]
                best.append(f(min(vals)) if len(vals) > 1 else None)
            first_t = True
            for m in present:
                cells = []
                if show_n:
                    cells.append(f"$2^{{{n}}}$" if first_n else "")
                if show_t:
                    cells.append(("$%d$" % t if t is not None else "n/a") if first_t else "")
                first_n = first_t = False
                cells.append(dict(SCHEMES)[m])
                for ci, (_, _, f) in enumerate(cols):
                    v = grid[m][ci]
                    txt = PLACEHOLDER if v is None else f(v)
                    cells.append(f"\\textbf{{{txt}}}" if v is not None and best[ci] is not None and txt == best[ci] else txt)
                lines.append("    " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}"]

    any_row = next(iter(by.values()))
    bitz_rows = [r for r in by.values() if r["key"][3].startswith("bitz")]
    binius_rows = [r for r in by.values() if r["key"][3].split("@")[0] == "binius64"]
    opener_rows = [r for r in by.values() if r["key"][3].split("@")[0] == "binius64-ligerito"]
    clauses = []
    if bitz_rows:
        policies = []
        for r in sorted(bitz_rows, key=lambda r: r["key"]):
            clause = bitz_security(r)
            if clause not in policies:
                policies.append(clause)
        clauses.append("\\ftwoz-SNARK (integer R1CS with $\\FF_2$-virtualization; round-by-round economic security model; " + "; ".join(policies) + ")")
    if binius_rows:
        policies = []
        for r in sorted(binius_rows, key=lambda r: r["key"]):
            clause = binius_security(r)
            if clause not in policies:
                policies.append(clause)
        s = binius_rows[0]["security"]
        clauses.append(f"Binius (UDR) (Binius64's fixed SHA-256 circuit and complete-arithmetic P-256 gadget, ring switching and BaseFold with "
                       f"{s.get('merkle_hash', 'SHA-256')} Merkle hashing; FRI query target only, " + "; ".join(policies) + ")")
    if opener_rows:
        policies = []
        for r in sorted(opener_rows, key=lambda r: r["key"]):
            clause = opener_security(r)
            if clause not in policies:
                policies.append(clause)
        clauses.append("Binius (Johnson) (the same Binius64 circuit and PIOP, every oracle committed and opened by the "
                       "\\ftwoz\\ opener --- Johnson regime, early Round-0 OOD, fold and query grinding --- gated at $100$ bits "
                       "under the round-by-round model, every error term at most $2^{-100}$ on its own; " + "; ".join(policies) + ")")
    compressions = any_row["compressions"]
    n_desc = f"$2^{{{exponents[0]}}} = {compressions}$" if len(exponents) == 1 else "$N$"
    msg = f"{any_row['message_bytes']:,}".replace(",", "{,}") + " bytes" if len(exponents) == 1 else "$64(N-1)$ bytes"
    threads_desc = " and ".join(str(t) for t in threads)
    reps = sorted({r["samples"] for r in by.values()})
    caption = (f"Proving one SHA-256 hash of a message of {msg} ({n_desc} compressions, padding block included) followed by one P-256 ECDSA "
               f"signature verification of the digest, non-ZK: " + "; ".join(clauses) + ". "
               "Native security targets are reported separately; these are not a uniform complete-protocol bound. "
               "\\emph{Witgen} is the witness generation from the message and signature; \\emph{prover} is the complete prover after "
               "witness generation, commitment included; \\emph{verifier} is the complete verification of the decoded proof; "
               "\\emph{proof} is the complete proof material, commitments included ($1$\\,KB $= 1000$ bytes); \\emph{peak mem.} is the "
               "high-water resident set of the worker process, setup and warm-up included ($1$\\,GB $= 2^{30}$ bytes). "
               f"Apple M5; threads per run: {threads_desc}; medians of {' or '.join(map(str, reps))} runs after one warm-up.")
    lines += ["  \\caption{" + caption + (" Prover and verifier times are given for %s threads; witgen and peak memory are from the %d-thread runs, and proof size does not depend on the thread count." % (" and ".join(map(str, threads)), max(threads))) + "}", f"  \\label{{{args.label}}}", "\\end{table}", ""]
    args.out.write_text("\n".join(header + lines))
    print(f"wrote {args.out} ({len(by)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
