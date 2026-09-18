#!/usr/bin/env python3
"""Build BitZ interval reports with protocol- and witness-component rows."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


DEFAULT_PROFILER = Path(__file__).resolve().with_name("zk_trace.py")


def load_profiler() -> Any:
    path = Path(os.environ.get("ZK_PROOF_PROFILER_SCRIPT", DEFAULT_PROFILER))
    spec = importlib.util.spec_from_file_location("zk_trace", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load zk-proof-profiler reporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zk_trace = load_profiler()


MATH_BY_OPERATION: dict[str, tuple[str, ...]] = {
    "sha256-trace.witness_generation": (
        r"\bar{\mathbf h}=M\bar{\mathbf f}",
        r"C\bar{\mathbf h}=\mathbf 0\quad(\text{over }\mathbb Z)",
    ),
    "sha256-witness.circuit_synthesis": (
        r"(\bar f_i,\bar h_i,O_i)=\operatorname{SHA256Compress}(S_i,B_i)",
    ),
    "sha256-witness.source_packing": (
        r"\bar{\mathbf f}=(1\Vert f_0\Vert\cdots\Vert f_{N-1}\Vert 0^{*})",
        r"j=r+2^t c",
    ),
    "sha256-witness.flat_assignment_packing": (
        r"\bar{\mathbf h}=(1\Vert h_0\Vert\cdots\Vert h_{N-1}\Vert 0^{*})",
    ),
    "sha256-witness.product_layout_transpose": (
        r"D_{\ell,i}=\bar h_i[\ell]",
        r"D\in\mathbb F_2^{L\times N}",
    ),
    "sha256-trace.commit": (
        r"C_f=\operatorname{Com}_{\mathbb F_{2^{128}}}(\bar{\mathbf f})",
    ),
    "sha256.local_relation_collapse_prover": (
        r"\beta_c=\sum_{r=0}^{183}\widetilde{\operatorname{eq}}(\boldsymbol\xi,r)C_{r,c}",
    ),
    "sha256.product_batch_prepare_prover": (
        r"d_c=\beta_c+\alpha_0\mathbf 1_{\{c=0\}}+\alpha_{\mathrm{pub}}\sum_{p:c_p=c}\lambda_p",
        r"V_{i,c}=u_i d_c",
    ),
    "sha256.bitz_prove": (
        r"\widetilde{\bar{\mathbf h}}(\mathbf r)=v",
        r"\bar{\mathbf h}=M\bar{\mathbf f}",
    ),
    "mqv.wprep": (
        r"E_r=\sum_{\ell}\eta_{\ell}\widetilde{\operatorname{eq}}(\mathbf r_{\ell},r)",
        r"W_j=\sum_{r:M_{rj}=1}E_r",
    ),
    "mc.presum": (
        r"\sum_{x\in\{0,1\}^{t_w}}\widetilde R(x)\,\widetilde m_{\mathbf z_c}(x)=Y_{\mathbf z_c}",
        r"\widetilde M(\mathbf r^{\star},\mathbf z_c)=\mu",
    ),
    "mq.reduction": (
        r"s_v=\widetilde M(\mathbf r_{\mathrm{hi}},v)",
        r"\sum_{v\in\{0,1\}^{7}}\widetilde{\operatorname{eq}}(\mathbf r_{\mathrm{lo}},v)\,s_v=\mu",
    ),
    "mqv.hs": (
        r"W_j=\sum_{r:M_{rj}=1}E_r",
        r"h_k=\sum_j\operatorname{bit}_k(W_j)\,p_{\lfloor j/128\rfloor}\,A(e_{j\bmod 128})",
    ),
    "mqv.aprime": (
        r"a'_p=\sum_{v=0}^{127}\Phi_{\boldsymbol\rho}(W_{128p+v})\,A(e_v)",
    ),
    "mq.lig": (r"\langle a',f\rangle=\sum_p a'_p f_p",),
}


KATEX_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
"""


def operation_math(operation: str, existing: list[str] | None = None) -> list[str]:
    return list(MATH_BY_OPERATION.get(operation, tuple(existing or ())))


def normalize_summary_math(summary: dict[str, Any]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            operation = value.get("operation")
            if isinstance(operation, str) and (
                operation in MATH_BY_OPERATION or "mathLatex" in value
            ):
                value["mathLatex"] = operation_math(
                    operation, value.get("mathLatex")
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(summary)


@dataclass(frozen=True)
class ComponentRow:
    key: str
    label: str
    row_tag: str
    alternatives: tuple[tuple[str, ...], ...]
    domain: str = "prover"
    ancestor_operations: tuple[str, ...] = ()
    parent_key: str | None = None
    relationship: str = "partition"


COMPONENT_ROWS = (
    ComponentRow(
        "sha-circuit-synthesis",
        "Actual SHA circuit synthesis",
        "witness-generation",
        (("sha256-witness.circuit_synthesis",),),
        domain="witness",
        parent_key="witness-generation",
        relationship="nested",
    ),
    ComponentRow(
        "source-packing",
        "Source packing",
        "witness-generation",
        (("sha256-witness.source_packing",),),
        domain="witness",
        parent_key="witness-generation",
        relationship="nested",
    ),
    ComponentRow(
        "flat-assignment-packing",
        "Flat assignment packing",
        "witness-generation",
        (("sha256-witness.flat_assignment_packing",),),
        domain="witness",
        parent_key="witness-generation",
        relationship="nested",
    ),
    ComponentRow(
        "product-layout-transpose",
        "Product-layout transpose",
        "witness-generation",
        (("sha256-witness.product_layout_transpose",),),
        domain="witness",
        parent_key="witness-generation",
        relationship="nested",
    ),
    ComponentRow(
        "proof",
        "Proof",
        "proving",
        (("sha256-trace.proof",),),
        parent_key="proving",
        relationship="nested",
    ),
    ComponentRow(
        "statement-binding",
        "Statement binding",
        "proving",
        (("sha256.statement_bind_prover",),),
        parent_key="proof",
        relationship="nested",
    ),
    ComponentRow(
        "runtime-field-setup",
        "Runtime-field setup / relation binding",
        "proving",
        (("step2.project_prove",),),
        parent_key="proof",
        relationship="nested",
    ),
    ComponentRow(
        "piop",
        "PIOP / relation reduction",
        "constraint-proof",
        (("step3.piop_prove",),),
        parent_key="proof",
        relationship="nested",
    ),
    ComponentRow(
        "bitify",
        "Bitify / terminal opening-claim bridge",
        "preparation",
        (("step4.bitify_prove",),),
        parent_key="proof",
        relationship="nested",
    ),
    ComponentRow(
        "virtual-bitz",
        "Virtual BitZ",
        "opening-proof",
        (("sha256.bitz_prove",),),
        parent_key="proof",
        relationship="nested",
    ),
    ComponentRow(
        "virtual-statement-binding",
        "Virtual-statement binding",
        "opening-proof",
        (("mqv.stmt",),),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
    ComponentRow(
        "derived-row-packing",
        "Derived-row packing",
        "opening-proof",
        (("mqv.pack",),),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
    ComponentRow(
        "power-table-construction",
        "Power-table construction",
        "opening-proof",
        (("mc.pow2",),),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
    ComponentRow(
        "gkr",
        "Merged-forest GKR",
        "opening-proof",
        (("mc.forest",),),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
    ComponentRow(
        "gkr-sumchecks",
        "GKR Sumchecks",
        "sumcheck",
        (("eqf.rounds",),),
        ancestor_operations=("mc.forest",),
        parent_key="gkr",
        relationship="nested",
    ),
    ComponentRow(
        "integer-fold",
        "Integer fold / v-message",
        "opening-proof",
        (("mc.fold_v",),),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
    ComponentRow(
        "pre-sumcheck",
        "Pre-sumcheck",
        "sumcheck",
        (("mc.presum",), ("mc.presum_tbls", "mc.presum_run")),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
    ComponentRow(
        "pre-sumcheck-tables",
        "Pre-sumcheck table construction",
        "opening-proof",
        (("mc.presum_tbls",),),
        parent_key="pre-sumcheck",
        relationship="nested",
    ),
    ComponentRow(
        "pre-sumcheck-rounds",
        "Pre-sumcheck protocol rounds",
        "sumcheck",
        (("mc.presum_run",),),
        parent_key="pre-sumcheck",
        relationship="nested",
    ),
    ComponentRow(
        "bitz-setup-packing",
        "BitZ setup / packing",
        "opening-proof",
        (("mqv.stmt", "mqv.pack", "mc.pack", "mc.pow2"),),
        parent_key="virtual-bitz",
        relationship="cross-cutting",
    ),
    ComponentRow(
        "ring-switching",
        "Ring switching",
        "opening-proof",
        (
            ("mq.reduction",),
            ("mq.rings", "mq.bcomb", "mqv.wprep", "mqv.hs", "mqv.aprime"),
        ),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
    ComponentRow(
        "virtual-weight-preparation",
        "Virtual BitZ weight preparation",
        "opening-proof",
        (("mqv.wprep",),),
        parent_key="ring-switching",
        relationship="nested",
    ),
    ComponentRow(
        "virtual-h-fold",
        "Virtual BitZ h fold",
        "opening-proof",
        (("mqv.hs",),),
        parent_key="ring-switching",
        relationship="nested",
    ),
    ComponentRow(
        "virtual-a-prime",
        "Virtual BitZ a-prime construction",
        "opening-proof",
        (("mqv.aprime",),),
        parent_key="ring-switching",
        relationship="nested",
    ),
    ComponentRow(
        "ligerito",
        "Recursive Ligerito",
        "opening-proof",
        (("mq.lig",),),
        parent_key="virtual-bitz",
        relationship="nested",
    ),
)


ROW_ORDER = (
    "proving",
    "pcs",
    "commit",
    "opening-proof",
    "proof",
    "witness-generation",
    "sha-circuit-synthesis",
    "source-packing",
    "flat-assignment-packing",
    "product-layout-transpose",
    "preparation",
    "constraint-proof",
    "statement-binding",
    "runtime-field-setup",
    "piop",
    "bitify",
    "sumcheck",
    "virtual-bitz",
    "virtual-statement-binding",
    "derived-row-packing",
    "power-table-construction",
    "gkr",
    "gkr-sumchecks",
    "integer-fold",
    "pre-sumcheck",
    "pre-sumcheck-tables",
    "pre-sumcheck-rounds",
    "bitz-setup-packing",
    "ring-switching",
    "virtual-weight-preparation",
    "virtual-h-fold",
    "virtual-a-prime",
    "ligerito",
    "verification",
)


def in_domain(span: dict[str, Any], domain: str) -> bool:
    tags = set(span.get("phase_tags", ()))
    if domain == "witness":
        return "witness-generation" in tags
    if domain == "prover":
        return "proving" in tags and "verification" not in tags
    raise AssertionError(f"unknown component-row domain: {domain}")


def select_spans(
    spans: list[dict[str, Any]], component: ComponentRow
) -> list[dict[str, Any]]:
    eligible = [span for span in spans if in_domain(span, component.domain)]
    if component.ancestor_operations:
        by_id = {span["span_id"]: span for span in spans}

        def has_requested_ancestor(span: dict[str, Any]) -> bool:
            parent_id = span.get("parent_span_id")
            while parent_id is not None:
                parent = by_id.get(parent_id)
                if parent is None:
                    return False
                if parent["operation"] in component.ancestor_operations:
                    return True
                parent_id = parent.get("parent_span_id")
            return False

        eligible = [span for span in eligible if has_requested_ancestor(span)]
    for operations in component.alternatives:
        selected = [span for span in eligible if span["operation"] in operations]
        if selected:
            return selected
    return []


def metric(values: list[int]) -> dict[str, Any]:
    stats = zk_trace._distribution(values)
    return {
        "medianNs": zk_trace._decimal_string(stats["median"]),
        "medianMsExact": zk_trace._exact_ms(stats["median"]),
        "n": stats["n"],
        "decilesMsExact": {
            key: zk_trace._exact_ms(value)
            for key, value in stats["deciles"].items()
        },
    }


def add_component_rows(trace: dict[str, Any], summary: dict[str, Any]) -> None:
    runs = trace["runs"]
    spans_by_run = trace["spans"]

    for series in summary["series"]:
        series_id = series["seriesId"]
        measured = [
            run
            for run in runs.values()
            if run["series_id"] == series_id
            and run["trial"]["kind"] == "sample"
            and run["status"] == "ok"
            and run["trace_complete"]
        ]
        measured.sort(key=lambda run: run["trial"].get("sample_index", 0))
        representative_id = series["representativeRunId"]
        representative = next(run for run in measured if run["run_id"] == representative_id)
        rep_spans = spans_by_run[representative_id]
        rep_by_id = {span["span_id"]: span for span in rep_spans}
        rep_root = rep_by_id[representative["root_span_id"]]
        geometry_scale = Decimal(series["geometryScaleExact"])
        derive_pcs = series.get("pcsDerived", False)

        operation_values: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        operation_keys: dict[tuple[str, str], tuple[Any, ...]] = {}
        for run in measured:
            run_spans = spans_by_run[run["run_id"]]
            keys = zk_trace._operation_instance_keys(run_spans)
            for span in run_spans:
                key = keys[span["span_id"]]
                operation_keys[(run["run_id"], span["span_id"])] = key
                operation_values[key].append(span["_duration"])
        operation_stats = {
            key: zk_trace._distribution(values)
            for key, values in operation_values.items()
        }

        def interval(span: dict[str, Any], component: ComponentRow) -> dict[str, Any]:
            key = operation_keys[(representative_id, span["span_id"])]
            stats = operation_stats[key]
            start = (
                Decimal(span["_start"] - rep_root["_start"])
                * geometry_scale
                / Decimal(1_000_000)
            )
            end = (
                Decimal(span["_end"] - rep_root["_start"])
                * geometry_scale
                / Decimal(1_000_000)
            )
            coordinate = span.get("coordinate", {})
            attributes = span.get("attributes", {})
            tags = [
                tag
                for tag in zk_trace._span_tags(span, derive_pcs)
                if tag != "fri"
            ]
            return {
                "id": f"{component.key}:{span['span_id']}",
                "spanId": span["span_id"],
                "operation": span["operation"],
                "name": span["name"],
                "shortName": attributes.get("short_name") or span["name"],
                "primaryPhase": span["primary_phase"],
                "rowTag": component.row_tag,
                "tags": tags,
                "scopeKind": attributes.get("scope_kind", "operation"),
                "startMs": float(start),
                "endMs": float(end),
                "medianNs": zk_trace._decimal_string(stats["median"]),
                "medianMsExact": zk_trace._exact_ms(stats["median"]),
                "medianN": stats["n"],
                "decilesMsExact": {
                    key: zk_trace._exact_ms(value)
                    for key, value in stats["deciles"].items()
                },
                "mathLatex": operation_math(
                    span["operation"], attributes.get("math_latex", [])
                ),
                "representativeNs": str(span["_duration"]),
                "representativeMsExact": zk_trace._exact_ms(
                    Decimal(span["_duration"])
                ),
                "occurrenceIndex": coordinate.get("occurrence_index"),
                "occurrenceCount": coordinate.get(
                    "occurrence_count", attributes.get("occurrence_count")
                ),
                "roundIndex": coordinate.get("round_index"),
                "roundCount": coordinate.get(
                    "round_count", attributes.get("round_count")
                ),
                "recursionDepth": coordinate.get("recursion_depth"),
                "recursionInstanceIndex": coordinate.get(
                    "recursion_instance_index"
                ),
                "overlay": attributes.get("overlay") is True,
            }

        custom_rows = []
        for component in COMPONENT_ROWS:
            values = []
            for run in measured:
                chosen = select_spans(spans_by_run[run["run_id"]], component)
                if chosen:
                    values.append(
                        zk_trace._union_duration(
                            [(span["_start"], span["_end"]) for span in chosen]
                        )
                    )
            if not values:
                continue
            if len(values) != len(measured):
                summary["warnings"].append(
                    f"{series_id} {component.key}: present in "
                    f"{len(values)}/{len(measured)} measured runs"
                )
            component_metric = metric(values)
            series["metrics"][component.key] = component_metric
            chosen_rep = select_spans(rep_spans, component)
            custom_rows.append(
                {
                    "tag": component.key,
                    "label": component.label,
                    "metric": component_metric,
                    "intervals": [
                        interval(span, component) for span in chosen_rep
                    ],
                    "section": "overlap",
                    "parentTag": component.parent_key,
                    "relationship": component.relationship,
                }
            )

        series["metrics"].pop("fri", None)
        existing_rows = [row for row in series["rows"] if row["tag"] != "fri"]
        by_key = {row["tag"]: row for row in (*existing_rows, *custom_rows)}
        series["rows"] = [by_key[key] for key in ROW_ORDER if key in by_key]
        for row in series["rows"]:
            for item in row["intervals"]:
                item["tags"] = [tag for tag in item["tags"] if tag != "fri"]
        for item in (series["root"], *series["primary"]):
            item["tags"] = [tag for tag in item["tags"] if tag != "fri"]


def write_metrics_csv(summary: dict[str, Any], output: Path) -> None:
    columns = [
        "series_id",
        "label",
        "metric",
        "median_ns",
        "median_ms",
        *(f"p{percentile}_ms" for percentile in range(10, 100, 10)),
        "n",
        "warmups",
        "excluded_samples",
        "representative_run_id",
        "geometry_scale",
        "root_boundary",
        "pcs_derived",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for series in summary["series"]:
            ordered = ["end-to-end", *(row["tag"] for row in series["rows"])]
            ordered.extend(key for key in series["metrics"] if key not in ordered)
            for key in ordered:
                value = series["metrics"].get(key)
                if value is None:
                    continue
                writer.writerow(
                    [
                        series["seriesId"],
                        series["label"],
                        key,
                        value["medianNs"],
                        value["medianMsExact"],
                        *(
                            value["decilesMsExact"][f"p{percentile}"]
                            for percentile in range(10, 100, 10)
                        ),
                        value["n"],
                        series["warmupN"],
                        series["excludedN"],
                        series["representativeRunId"],
                        series["geometryScaleExact"],
                        series["rootBoundary"],
                        str(series["pcsDerived"]).lower(),
                    ]
                )


def render_html(summary: dict[str, Any], fragment: bool) -> str:
    rendered = zk_trace.render_html(summary, fragment)
    rendered = rendered.replace(
        "y1: 30, y2: 520, class: \"guide\"",
        "y1: 30, y2: Math.max(520, 170 + rowInfo(series).length * 39), class: \"guide\"",
    )
    rendered = rendered.replace(
        'code.textContent = "\\\\(" + expression + "\\\\)";',
        "code.textContent = expression;",
    )
    if not fragment:
        rendered = rendered.replace("</head>", KATEX_HEAD + "</head>", 1)
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="BitZ component interval profile")
    parser.add_argument("--fragment", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        trace = zk_trace.load_and_validate(args.input)
        summary = zk_trace.build_summary(trace, args.title)
        add_component_rows(trace, summary)
        normalize_summary_math(summary)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.out_dir / "summary.json"
        csv_path = args.out_dir / "metrics.csv"
        html_path = args.out_dir / (
            "intervals.fragment.html" if args.fragment else "intervals.html"
        )
        for path in (summary_path, csv_path, html_path):
            zk_trace._ensure_new_output(path, args.force)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_metrics_csv(summary, csv_path)
        html_path.write_text(render_html(summary, args.fragment), encoding="utf-8")
        print(summary_path)
        print(csv_path)
        print(html_path)
        return 0
    except (OSError, RuntimeError, zk_trace.TraceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
