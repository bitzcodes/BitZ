#!/usr/bin/env python3
"""Tables and figures from verified multiplication campaigns (mul-bench/v2 only)."""
from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path
from mul_results import aggregate, load, select


def label(row):
    case = row["case"]
    config = case.get("bitz", {})
    pieces = [case["mode"], case["workload"], case["backend"], f"2^{case['log_n']}",
              f"t={case['threads']}", f"seed={case['seed']}"]
    pieces += [f"{key}={value}" for key, value in config.items() if value is not None]
    pieces += [f"{key}={case[key]}" for key in ("variant", "preset", "log_inv_rate", "binius_ligerito_accounting", "limber_bits") if key in case]
    pieces += [f"whir.{key}={value}" for key, value in case.get("whir", {}).items() if value is not None]
    return " / ".join(pieces)


def table(rows, out):
    metrics = sorted({key for row in rows for key in row["metrics"]})
    memory = sorted({key for row in rows for key in row["memory"]})
    columns = ["source", "case_id", "case", "status", "reason", *metrics, *memory]
    data = [dict(source=row["source"], case_id=row["case_id"], case=label(row), status=row["status"], reason=row["reason"] or "",
                 **{key: value["median"] for key, value in row["metrics"].items()},
                 **row["memory"]) for row in rows]
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
    def fmt(value):
        return f"{value:.6g}" if isinstance(value, (float, int)) else str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(fmt(row.get(c, "—")).replace("|", "\\|") for c in columns) + " |" for row in data]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    escape = lambda s: str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("^", r"\textasciicircum{}")
    tex = [r"\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(map(escape, columns)) + r" \\"]
    tex += [" & ".join(escape(fmt(row.get(c, "—"))) for c in columns) + r" \\" for row in data]
    (out / "table.tex").write_text("\n".join([*tex, r"\end{tabular}"]) + "\n")


def figure(rows, metric, out):
    """One bar per complete case; incompatible configurations never share a bar."""
    values = [(label(row), row["metrics"][metric]) for row in rows if metric in row["metrics"]]
    if not values:
        raise ValueError(f"no measured cases contain {metric}")
    width, left, plot_width = 1600, 950, 580
    height = 70 + 30 * len(values)
    maximum = max(stats["p95"] for _, stats in values) or 1
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="15" y="25" font-family="sans-serif" font-size="16">{html.escape(metric)} — median; 5th–95th percentiles</text>']
    for i, (name, stats) in enumerate(values):
        y = 50 + i * 30
        x = lambda value: left + plot_width * value / maximum
        elements += [f'<text x="15" y="{y + 14}" font-family="monospace" font-size="11">{html.escape(name)}</text>',
                     f'<rect x="{left}" y="{y}" width="{x(stats["median"]) - left}" height="18" fill="#2563eb"/>',
                     f'<line x1="{x(stats["p05"])}" x2="{x(stats["p95"])}" y1="{y + 9}" y2="{y + 9}" stroke="#111827"/>',
                     f'<text x="{x(stats["p95"]) + 5}" y="{y + 14}" font-family="sans-serif" font-size="11">{stats["median"]:.4g}</text>']
    (out / (metric.replace("/", "-") + ".svg")).write_text("\n".join([*elements, "</svg>"]) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--select", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--metric", action="append", help="draw this metric (repeatable)")
    args = parser.parse_args(argv)
    filters = {}
    for item in args.select:
        if "=" not in item:
            parser.error("--select requires KEY=VALUE")
        key, value = item.split("=", 1)
        if key in filters:
            parser.error(f"duplicate selection {key}")
        filters[key] = value
    cases = [case for directory in args.campaign for case in load(directory)]
    rows = aggregate(select(cases, filters))
    if not rows:
        parser.error("no selected cases")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    table(rows, args.out)
    metrics = args.metric or [key for key in ("witness_to_proof_ms", "online_prover_ms", "witness_ms", "pcs_ms", "piop_ms", "outer_ms", "production_ms", "generic_ms", "proof_bytes") if any(key in row["metrics"] for row in rows)]
    for metric in metrics:
        figure(rows, metric, args.out)
    print(f"{len(rows)} cases: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
