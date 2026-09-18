#!/usr/bin/env python3
"""Validate, import, summarize, and render tagged ZK prover performance traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


TRACE_SCHEMA = "zkperf.trace/v1"
REPORT_SCHEMA = "zkperf.report/v1"
PHASE_TAGS = (
    "end-to-end",
    "witness-generation",
    "preparation",
    "proving",
    "commit",
    "pcs",
    "opening-proof",
    "constraint-proof",
    "sumcheck",
    "fri",
    "verification",
)
TAG_LABELS = {
    "end-to-end": "End-to-end",
    "witness-generation": "Witness generation",
    "preparation": "Preparation",
    "proving": "Proving",
    "commit": "Commit",
    "pcs": "PCS total",
    "opening-proof": "Opening proof",
    "constraint-proof": "Constraint proof",
    "sumcheck": "Sumcheck",
    "fri": "FRI",
    "verification": "Verification",
}
PRIMARY_ROWS = ("end-to-end", "proving", "pcs", "commit", "opening-proof")
ROW_ORDER = (
    "proving",
    "pcs",
    "commit",
    "opening-proof",
    "witness-generation",
    "preparation",
    "constraint-proof",
    "sumcheck",
    "fri",
    "verification",
)
OPERATION_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
NS_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")


class TraceError(Exception):
    """User-facing trace validation or conversion failure."""


def _reject_constant(value: str) -> None:
    raise TraceError(f"non-finite JSON number is forbidden: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TraceError(f"non-finite JSON number is forbidden: {value}")
    return number


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceError(f"{where} must be a JSON object")
    return value


def _require_string(record: dict[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TraceError(f"{where}.{key} must be a non-empty string")
    return value


def _parse_ns(value: Any, where: str) -> int:
    if not isinstance(value, str) or not NS_RE.fullmatch(value):
        raise TraceError(f"{where} must be an exact unsigned decimal string")
    return int(value)


def _median(values: Iterable[int]) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise TraceError("cannot compute a median from an empty sequence")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return Decimal(ordered[middle])
    return (Decimal(ordered[middle - 1]) + Decimal(ordered[middle])) / 2


def _type7_quantile(values: Iterable[int], percentile: int) -> Decimal:
    """Return a Hyndman-Fan Type 7 sample quantile using exact arithmetic."""
    ordered = sorted(values)
    if not ordered:
        raise TraceError("cannot compute a quantile from an empty sequence")
    if percentile < 0 or percentile > 100:
        raise TraceError("percentile must be between 0 and 100")
    position = Decimal(len(ordered) - 1) * Decimal(percentile) / Decimal(100)
    lower = int(position)
    fraction = position - Decimal(lower)
    upper = min(lower + 1, len(ordered) - 1)
    return Decimal(ordered[lower]) + fraction * Decimal(
        ordered[upper] - ordered[lower]
    )


def _distribution(values: Iterable[int]) -> dict[str, Any]:
    samples = list(values)
    median = _median(samples)
    return {
        "median": median,
        "n": len(samples),
        "deciles": {
            f"p{percentile}": _type7_quantile(samples, percentile)
            for percentile in range(10, 100, 10)
        },
    }


def _decimal_string(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _exact_ms(value_ns: Decimal) -> str:
    rendered = format(value_ns / Decimal(1_000_000), "f")
    whole, separator, fractional = rendered.partition(".")
    fractional = fractional.rstrip("0") if separator else ""
    return f"{whole}.{fractional.ljust(6, '0')}"


def _union_duration(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _metadata_fingerprint(run: dict[str, Any]) -> str:
    stable = {
        "benchmark": run.get("benchmark", {}),
        "environment": run.get("environment", {}),
        "parameters": run.get("parameters", {}),
        "tags": run.get("tags", {}),
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_and_validate(paths: list[Path]) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    spans_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_span_ids: dict[str, set[str]] = defaultdict(set)
    source_hashes: list[dict[str, str]] = []

    for path in paths:
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise TraceError(f"cannot read UTF-8 trace {path}: {exc}") from exc
        source_hashes.append({"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                raise TraceError(f"{path}:{line_number}: blank JSONL lines are forbidden")
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_no_duplicate_keys,
                    parse_constant=_reject_constant,
                    parse_float=_finite_float,
                )
            except (json.JSONDecodeError, TraceError) as exc:
                raise TraceError(f"{path}:{line_number}: {exc}") from exc
            record = _require_object(record, f"{path}:{line_number}")
            if record.get("schema") != TRACE_SCHEMA:
                raise TraceError(
                    f"{path}:{line_number}: schema must be {TRACE_SCHEMA!r}"
                )
            kind = record.get("record")
            if kind == "run":
                run_id = _require_string(record, "run_id", f"{path}:{line_number}")
                if run_id in runs:
                    raise TraceError(f"duplicate run_id: {run_id}")
                record["_source"] = f"{path}:{line_number}"
                runs[run_id] = record
            elif kind == "span":
                run_id = _require_string(record, "run_id", f"{path}:{line_number}")
                span_id = _require_string(record, "span_id", f"{path}:{line_number}")
                if span_id in seen_span_ids[run_id]:
                    raise TraceError(f"{path}:{line_number}: duplicate span_id {span_id!r}")
                seen_span_ids[run_id].add(span_id)
                record["_source"] = f"{path}:{line_number}"
                spans_by_run[run_id].append(record)
            else:
                raise TraceError(
                    f"{path}:{line_number}: record must be 'run' or 'span'"
                )

    if not runs:
        raise TraceError("trace contains no run records")
    orphan_runs = sorted(set(spans_by_run) - set(runs))
    if orphan_runs:
        raise TraceError(f"span records reference missing runs: {', '.join(orphan_runs)}")

    trial_indices: set[tuple[str, str, int]] = set()
    series_fingerprints: dict[str, str] = {}
    warnings: list[str] = []

    for run_id, run in runs.items():
        where = run["_source"]
        series_id = _require_string(run, "series_id", where)
        root_span_id = _require_string(run, "root_span_id", where)
        benchmark = _require_object(run.get("benchmark"), f"{where}.benchmark")
        _require_string(benchmark, "name", f"{where}.benchmark")
        for key in ("environment", "parameters", "tags"):
            if key in run:
                _require_object(run[key], f"{where}.{key}")
        trial = _require_object(run.get("trial"), f"{where}.trial")
        trial_kind = trial.get("kind")
        if trial_kind == "sample":
            index_key = "sample_index"
        elif trial_kind == "warmup":
            index_key = "warmup_index"
        else:
            raise TraceError(f"{where}.trial.kind must be 'sample' or 'warmup'")
        index = trial.get(index_key)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TraceError(f"{where}.trial.{index_key} must be a non-negative integer")
        if ("warmup_index" if index_key == "sample_index" else "sample_index") in trial:
            raise TraceError(f"{where}.trial mixes sample and warmup indices")
        trial_key = (series_id, trial_kind, index)
        if trial_key in trial_indices:
            raise TraceError(f"duplicate {trial_kind} index {index} in series {series_id}")
        trial_indices.add(trial_key)

        clock = _require_object(run.get("clock"), f"{where}.clock")
        if clock.get("kind") != "monotonic" or clock.get("unit") != "ns":
            raise TraceError(f"{where}.clock must use monotonic nanoseconds")
        if run.get("status") not in {"ok", "error", "panic", "cancelled"}:
            raise TraceError(f"{where}.status is invalid")
        if not isinstance(run.get("trace_complete"), bool):
            raise TraceError(f"{where}.trace_complete must be boolean")

        fingerprint = _metadata_fingerprint(run)
        previous = series_fingerprints.setdefault(series_id, fingerprint)
        if previous != fingerprint:
            raise TraceError(f"metadata drift inside series {series_id!r}")

        spans = spans_by_run.get(run_id, [])
        if not spans:
            raise TraceError(f"run {run_id!r} contains no spans")
        by_id = {span["span_id"]: span for span in spans}
        root = by_id.get(root_span_id)
        if root is None:
            raise TraceError(f"run {run_id!r} is missing root span {root_span_id!r}")

        for span in spans:
            span_where = span["_source"]
            operation = _require_string(span, "operation", span_where)
            if not OPERATION_RE.fullmatch(operation):
                raise TraceError(
                    f"{span_where}.operation must be stable lower-case dotted/hyphenated identity"
                )
            _require_string(span, "name", span_where)
            primary = _require_string(span, "primary_phase", span_where)
            tags = span.get("phase_tags")
            if not isinstance(tags, list) or not tags:
                raise TraceError(f"{span_where}.phase_tags must be a non-empty array")
            if any(not isinstance(tag, str) for tag in tags):
                raise TraceError(f"{span_where}.phase_tags must contain strings")
            if len(tags) != len(set(tags)):
                raise TraceError(f"{span_where}.phase_tags contains duplicates")
            unknown = sorted(set(tags) - set(PHASE_TAGS))
            if unknown:
                raise TraceError(f"{span_where}.phase_tags has unknown tags: {unknown}")
            if primary not in tags:
                raise TraceError(f"{span_where}.primary_phase must appear in phase_tags")
            for key in ("lane", "coordinate", "attributes"):
                if key in span:
                    _require_object(span[key], f"{span_where}.{key}")
            coordinate = span.get("coordinate", {})
            for key in (
                "occurrence_index",
                "occurrence_count",
                "round_index",
                "round_count",
                "recursion_depth",
                "recursion_instance_index",
            ):
                value = coordinate.get(key)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise TraceError(
                        f"{span_where}.coordinate.{key} must be a non-negative integer"
                    )
            attributes = span.get("attributes", {})
            scope_kind = attributes.get("scope_kind")
            if scope_kind is not None and scope_kind not in {
                "scope",
                "operation",
                "phase",
                "procedure",
                "round",
            }:
                raise TraceError(f"{span_where}.attributes.scope_kind is invalid")
            scope_tag = attributes.get("scope_tag")
            if scope_tag is not None and scope_tag not in PHASE_TAGS:
                raise TraceError(f"{span_where}.attributes.scope_tag is invalid")
            if scope_tag is not None and scope_tag not in tags:
                raise TraceError(
                    f"{span_where}.attributes.scope_tag must appear in phase_tags"
                )
            short_name = attributes.get("short_name")
            if short_name is not None and not isinstance(short_name, str):
                raise TraceError(f"{span_where}.attributes.short_name must be a string")
            math_latex = attributes.get("math_latex")
            if math_latex is not None:
                if (
                    not isinstance(math_latex, list)
                    or not 1 <= len(math_latex) <= 2
                    or any(not isinstance(item, str) or not item.strip() for item in math_latex)
                ):
                    raise TraceError(
                        f"{span_where}.attributes.math_latex must contain one or two non-empty strings"
                    )
            for key in ("primary_sequence", "overlay", "repeated"):
                value = attributes.get(key)
                if value is not None and not isinstance(value, bool):
                    raise TraceError(f"{span_where}.attributes.{key} must be boolean")
            for index_key, count_key in (
                ("occurrence_index", "occurrence_count"),
                ("round_index", "round_count"),
            ):
                index_value = coordinate.get(index_key)
                count_value = coordinate.get(count_key)
                if (
                    index_value is not None
                    and count_value is not None
                    and index_value >= count_value
                ):
                    raise TraceError(
                        f"{span_where}.coordinate.{index_key} must be less than "
                        f"{count_key}"
                    )

            start = _parse_ns(span.get("start_ns"), f"{span_where}.start_ns")
            end = _parse_ns(span.get("end_ns"), f"{span_where}.end_ns")
            duration = _parse_ns(span.get("duration_ns"), f"{span_where}.duration_ns")
            if end < start:
                raise TraceError(f"{span_where}: end_ns precedes start_ns")
            if duration != end - start:
                raise TraceError(f"{span_where}: duration_ns != end_ns - start_ns")
            span["_start"] = start
            span["_end"] = end
            span["_duration"] = duration
            parent_id = span.get("parent_span_id")
            if parent_id is not None and not isinstance(parent_id, str):
                raise TraceError(f"{span_where}.parent_span_id must be string or null")

        if root.get("parent_span_id") is not None:
            raise TraceError(f"run {run_id!r} root span must have no parent")
        for span in spans:
            if span is root:
                continue
            parent_id = span.get("parent_span_id")
            parent = by_id.get(parent_id)
            if parent is None:
                raise TraceError(
                    f"{span['_source']}: missing parent span {parent_id!r}"
                )
            if not (
                parent["_start"] <= span["_start"]
                and span["_end"] <= parent["_end"]
            ):
                raise TraceError(f"{span['_source']}: parent does not contain child interval")
            if not (
                root["_start"] <= span["_start"] and span["_end"] <= root["_end"]
            ):
                raise TraceError(f"{span['_source']}: root does not contain span")

        for span in spans:
            visited: set[str] = set()
            cursor = span
            while cursor.get("parent_span_id") is not None:
                cursor_id = cursor["span_id"]
                if cursor_id in visited:
                    raise TraceError(f"run {run_id!r} has a parent cycle")
                visited.add(cursor_id)
                cursor = by_id[cursor["parent_span_id"]]

    for series_id in sorted({run["series_id"] for run in runs.values()}):
        eligible = [
            run
            for run in runs.values()
            if run["series_id"] == series_id
            and run["trial"]["kind"] == "sample"
            and run["status"] == "ok"
            and run["trace_complete"]
        ]
        if not eligible:
            warnings.append(f"series {series_id!r} has no eligible measured samples")
        elif len(eligible) < 5:
            warnings.append(
                f"series {series_id!r} has {len(eligible)} eligible samples; five are preferred"
            )

    return {
        "runs": runs,
        "spans": spans_by_run,
        "warnings": warnings,
        "source_hashes": sorted(source_hashes, key=lambda item: item["name"]),
    }


def _span_tags(span: dict[str, Any], derive_pcs: bool) -> list[str]:
    tags = list(span["phase_tags"])
    if derive_pcs and "pcs" not in tags and (
        "commit" in tags or "opening-proof" in tags
    ):
        tags.append("pcs")
    return [tag for tag in PHASE_TAGS if tag in tags]


def _is_descendant(
    inner: dict[str, Any],
    outer: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> bool:
    parent_id = inner.get("parent_span_id")
    while parent_id is not None:
        if parent_id == outer["span_id"]:
            return True
        parent = by_id.get(parent_id)
        if parent is None:
            return False
        parent_id = parent.get("parent_span_id")
    return False


def _choose_primary(spans: list[dict[str, Any]], root: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = [
        span
        for span in spans
        if span is not root
        and _require_object(span.get("attributes", {}), "span.attributes").get(
            "primary_sequence"
        )
        is True
    ]
    if explicit:
        return sorted(explicit, key=lambda span: (span["_start"], span["_end"]))
    candidates = []
    for span in spans:
        if span is root:
            continue
        attrs = _require_object(span.get("attributes", {}), "span.attributes")
        scope_kind = attrs.get("scope_kind")
        if scope_kind == "phase" or (
            scope_kind == "operation"
            and "witness-generation" in span["phase_tags"]
            and span.get("parent_span_id") == root["span_id"]
        ):
            candidates.append(span)
    if not candidates:
        candidates = [
            span
            for span in spans
            if span.get("parent_span_id") == root["span_id"]
        ]
    return sorted(candidates, key=lambda span: (span["_start"], span["_end"]))


def _choose_row_spans(
    spans: list[dict[str, Any]],
    root: dict[str, Any],
    tag: str,
    derive_pcs: bool,
) -> list[dict[str, Any]]:
    by_id = {span["span_id"]: span for span in spans}
    candidates = [
        span
        for span in spans
        if span is not root and tag in _span_tags(span, derive_pcs)
    ]
    explicit = [
        span
        for span in candidates
        if span.get("attributes", {}).get("scope_tag") == tag
    ]
    if explicit:
        return sorted(explicit, key=lambda span: (span["_start"], span["_end"]))

    phase_preferred = tag in {
        "proving",
        "pcs",
        "preparation",
        "constraint-proof",
        "verification",
    }
    phases = [
        span
        for span in candidates
        if span.get("attributes", {}).get("scope_kind") == "phase"
    ]
    if phase_preferred and phases:
        selected = list(phases)
        for span in candidates:
            if span in phases:
                continue
            if span.get("attributes", {}).get("overlay") is True:
                selected.append(span)
                continue
            if not any(_is_descendant(span, phase, by_id) for phase in phases):
                selected.append(span)
        return sorted(
            {span["span_id"]: span for span in selected}.values(),
            key=lambda span: (span["_start"], span["_end"]),
        )

    selected: list[dict[str, Any]] = []
    for span in candidates:
        attrs = span.get("attributes", {})
        if attrs.get("overlay") is True:
            selected.append(span)
            continue
        repeated_ancestor = any(
            _is_descendant(span, other, by_id)
            and other.get("attributes", {}).get("repeated") is True
            for other in candidates
        )
        if repeated_ancestor:
            continue
        if attrs.get("repeated") is True:
            selected.append(span)
            continue
        descendants = [
            other
            for other in candidates
            if _is_descendant(other, span, by_id)
            and other.get("attributes", {}).get("overlay") is not True
        ]
        if not descendants:
            selected.append(span)
    return sorted(
        {span["span_id"]: span for span in selected}.values(),
        key=lambda span: (span["_start"], span["_end"]),
    )


def _series_label(run: dict[str, Any]) -> str:
    benchmark = run.get("benchmark", {})
    if benchmark.get("label"):
        return str(benchmark["label"])
    inputs = run.get("parameters", {}).get("input", {})
    compressions = inputs.get("sha256_compressions")
    if isinstance(compressions, int) and compressions > 0 and compressions & (compressions - 1) == 0:
        return f"2^{compressions.bit_length() - 1}"
    return str(run["series_id"])


def _algorithm_name(run: dict[str, Any]) -> str:
    benchmark = run.get("benchmark", {})
    parameters = run.get("parameters", {})
    parameter_algorithm = parameters.get("algorithm")
    candidates = [
        benchmark.get("algorithm"),
        (
            parameter_algorithm.get("name")
            if isinstance(parameter_algorithm, dict)
            else parameter_algorithm
        ),
        benchmark.get("label"),
        benchmark.get("name"),
        benchmark.get("implementation"),
        run.get("series_id"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "ZK proof"


def _comparison_title(report_title: str, algorithm: str) -> str:
    title = report_title.strip()
    if not title:
        return algorithm
    if algorithm.casefold() in title.casefold():
        return title
    return f"{algorithm} — {title}"


def _metadata_badges(run: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    inputs = run.get("parameters", {}).get("input", {})
    security = run.get("parameters", {}).get("security", {})
    environment = run.get("environment", {})
    fields = (
        ("SHA compressions", inputs.get("sha256_compressions")),
        ("SHA internal rounds", inputs.get("sha256_internal_rounds")),
        ("Witness bits", inputs.get("witness_bits")),
        ("Rows", inputs.get("num_rows")),
        ("Columns", inputs.get("num_cols")),
        ("Constraints", inputs.get("constraints")),
        ("Security", security.get("target_bits")),
        ("Queries", security.get("query_count")),
        ("Threads", environment.get("threads")),
    )
    for label, value in fields:
        if value is None:
            continue
        suffix = " bits" if label == "Security" else ""
        rendered = f"{value:,}" if isinstance(value, int) else str(value)
        result.append({"label": label, "value": f"{rendered}{suffix}"})
    return result


def _operation_instance_keys(
    spans: list[dict[str, Any]],
) -> dict[str, tuple[Any, ...]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    identity_fields = (
        "occurrence_index",
        "round_index",
        "recursion_depth",
        "recursion_instance_index",
    )
    for span in spans:
        coordinate = span.get("coordinate", {})
        identity = tuple(
            (field, coordinate[field])
            for field in identity_fields
            if field in coordinate
        )
        grouped[(span["operation"], span["primary_phase"], identity)].append(span)

    result: dict[str, tuple[Any, ...]] = {}
    for base_key, instances in grouped.items():
        ordered = sorted(
            instances,
            key=lambda span: (span["_start"], span["_end"], span["span_id"]),
        )
        for ordinal, span in enumerate(ordered):
            result[span["span_id"]] = (*base_key, ordinal)
    return result


def build_summary(trace: dict[str, Any], title: str) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = trace["runs"]
    spans_by_run: dict[str, list[dict[str, Any]]] = trace["spans"]
    report_series: list[dict[str, Any]] = []

    for series_id in sorted({run["series_id"] for run in runs.values()}):
        series_runs = sorted(
            [run for run in runs.values() if run["series_id"] == series_id],
            key=lambda run: (
                run["trial"]["kind"] != "warmup",
                run["trial"].get("warmup_index", run["trial"].get("sample_index", 0)),
            ),
        )
        measured = [
            run
            for run in series_runs
            if run["trial"]["kind"] == "sample"
            and run["status"] == "ok"
            and run["trace_complete"]
        ]
        if not measured:
            continue
        exemplar = measured[0]
        explicit_pcs = any(
            "pcs" in span["phase_tags"]
            for run in measured
            for span in spans_by_run[run["run_id"]]
        )
        derive_pcs = not explicit_pcs

        root_durations: dict[str, int] = {}
        for run in measured:
            by_id = {span["span_id"]: span for span in spans_by_run[run["run_id"]]}
            root_durations[run["run_id"]] = by_id[run["root_span_id"]]["_duration"]
        median_root = _median(root_durations.values())
        representative_pool = (
            measured
            if median_root == 0
            else [
                run for run in measured if root_durations[run["run_id"]] > 0
            ]
        )
        representative = min(
            representative_pool,
            key=lambda run: (
                abs(Decimal(root_durations[run["run_id"]]) - median_root),
                run["run_id"],
            ),
        )
        rep_spans = spans_by_run[representative["run_id"]]
        rep_by_id = {span["span_id"]: span for span in rep_spans}
        rep_root = rep_by_id[representative["root_span_id"]]
        geometry_scale = (
            Decimal(1)
            if rep_root["_duration"] == 0
            else median_root / Decimal(rep_root["_duration"])
        )

        root_distribution = _distribution(root_durations.values())
        metrics: dict[str, dict[str, Any]] = {
            "end-to-end": {
                "medianNs": _decimal_string(median_root),
                "medianMsExact": _exact_ms(median_root),
                "n": len(measured),
                "decilesMsExact": {
                    key: _exact_ms(value)
                    for key, value in root_distribution["deciles"].items()
                },
            }
        }
        for tag in ROW_ORDER:
            values = []
            for run in measured:
                tagged = [
                    (span["_start"], span["_end"])
                    for span in spans_by_run[run["run_id"]]
                    if tag in _span_tags(span, derive_pcs)
                ]
                if tagged:
                    values.append(_union_duration(tagged))
            if values:
                distribution = _distribution(values)
                median_value = distribution["median"]
                metrics[tag] = {
                    "medianNs": _decimal_string(median_value),
                    "medianMsExact": _exact_ms(median_value),
                    "n": len(values),
                    "decilesMsExact": {
                        key: _exact_ms(value)
                        for key, value in distribution["deciles"].items()
                    },
                }

        operation_values: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        operation_keys: dict[tuple[str, str], tuple[Any, ...]] = {}
        for run in measured:
            run_spans = spans_by_run[run["run_id"]]
            keys = _operation_instance_keys(run_spans)
            for span in run_spans:
                key = keys[span["span_id"]]
                operation_keys[(run["run_id"], span["span_id"])] = key
                operation_values[key].append(span["_duration"])
        operation_stats = {
            key: _distribution(values) for key, values in operation_values.items()
        }

        def interval(span: dict[str, Any], row_tag: str) -> dict[str, Any]:
            key = operation_keys[(representative["run_id"], span["span_id"])]
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
            attrs = span.get("attributes", {})
            return {
                "id": f"{row_tag}:{span['span_id']}",
                "spanId": span["span_id"],
                "operation": span["operation"],
                "name": span["name"],
                "shortName": attrs.get("short_name") or span["name"],
                "primaryPhase": span["primary_phase"],
                "rowTag": row_tag,
                "tags": _span_tags(span, derive_pcs),
                "scopeKind": attrs.get("scope_kind", "operation"),
                "startMs": float(start),
                "endMs": float(end),
                "medianNs": _decimal_string(stats["median"]),
                "medianMsExact": _exact_ms(stats["median"]),
                "medianN": stats["n"],
                "decilesMsExact": {
                    key: _exact_ms(value)
                    for key, value in stats["deciles"].items()
                },
                "mathLatex": attrs.get("math_latex", []),
                "representativeNs": str(span["_duration"]),
                "representativeMsExact": _exact_ms(Decimal(span["_duration"])),
                "occurrenceIndex": coordinate.get("occurrence_index"),
                "occurrenceCount": coordinate.get(
                    "occurrence_count", attrs.get("occurrence_count")
                ),
                "roundIndex": coordinate.get("round_index"),
                "roundCount": coordinate.get("round_count", attrs.get("round_count")),
                "recursionDepth": coordinate.get("recursion_depth"),
                "recursionInstanceIndex": coordinate.get(
                    "recursion_instance_index"
                ),
                "overlay": attrs.get("overlay") is True,
            }

        root_interval = interval(rep_root, "end-to-end")
        primary = [
            interval(span, "phase-sequence")
            for span in _choose_primary(rep_spans, rep_root)
        ]
        rows = []
        for tag in ROW_ORDER:
            if tag not in metrics:
                continue
            chosen = _choose_row_spans(rep_spans, rep_root, tag, derive_pcs)
            if not chosen:
                continue
            rows.append(
                {
                    "tag": tag,
                    "label": TAG_LABELS[tag],
                    "metric": metrics[tag],
                    "intervals": [interval(span, tag) for span in chosen],
                    "section": "primary" if tag in PRIMARY_ROWS else "overlap",
                }
            )

        report_series.append(
            {
                "seriesId": series_id,
                "label": _series_label(exemplar),
                "algorithm": _algorithm_name(exemplar),
                "displayTitle": _comparison_title(title, _algorithm_name(exemplar)),
                "benchmark": exemplar.get("benchmark", {}),
                "environment": exemplar.get("environment", {}),
                "parameters": exemplar.get("parameters", {}),
                "badges": _metadata_badges(exemplar),
                "measuredN": len(measured),
                "warmupN": sum(
                    run["trial"]["kind"] == "warmup" for run in series_runs
                ),
                "excludedN": sum(
                    run["trial"]["kind"] == "sample" and run not in measured
                    for run in series_runs
                ),
                "representativeRunId": representative["run_id"],
                "medianTotalMs": float(median_root / Decimal(1_000_000)),
                "medianTotalMsExact": _exact_ms(median_root),
                "geometryScale": float(geometry_scale),
                "geometryScaleExact": _decimal_string(geometry_scale),
                "rootBoundary": exemplar.get("tags", {}).get(
                    "root_boundary", "declared-root"
                ),
                "pcsDerived": derive_pcs and "pcs" in metrics,
                "metrics": metrics,
                "root": root_interval,
                "primary": primary,
                "rows": rows,
            }
        )

    if not report_series:
        raise TraceError("no eligible measured series can be summarized")
    return {
        "schema": REPORT_SCHEMA,
        "title": title,
        "sourceHashes": trace["source_hashes"],
        "warnings": trace["warnings"],
        "series": report_series,
    }


def write_metrics_csv(summary: dict[str, Any], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "series_id",
                "label",
                "metric",
                "median_ns",
                "median_ms",
                "p10_ms",
                "p20_ms",
                "p30_ms",
                "p40_ms",
                "p50_ms",
                "p60_ms",
                "p70_ms",
                "p80_ms",
                "p90_ms",
                "n",
                "warmups",
                "excluded_samples",
                "representative_run_id",
                "geometry_scale",
                "root_boundary",
                "pcs_derived",
            ]
        )
        for series in summary["series"]:
            for tag in PHASE_TAGS:
                metric = series["metrics"].get(tag)
                if metric is None:
                    continue
                writer.writerow(
                    [
                        series["seriesId"],
                        series["label"],
                        tag,
                        metric["medianNs"],
                        metric["medianMsExact"],
                        *[
                            metric["decilesMsExact"][f"p{percentile}"]
                            for percentile in range(10, 100, 10)
                        ],
                        metric["n"],
                        series["warmupN"],
                        series["excludedN"],
                        series["representativeRunId"],
                        series["geometryScaleExact"],
                        series["rootBoundary"],
                        str(series["pcsDerived"]).lower(),
                    ]
                )


def _safe_embedded_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _ensure_new_output(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise TraceError(f"output already exists: {path}; pass --force to replace it")


def render_html(summary: dict[str, Any], fragment: bool) -> str:
    root_id = "zk-proof-profiler-" + hashlib.sha256(
        _safe_embedded_json(summary).encode("utf-8")
    ).hexdigest()[:10]
    body = HTML_FRAGMENT.replace("__ROOT_ID__", root_id).replace(
        "__REPORT_DATA__", _safe_embedded_json(summary)
    )
    if fragment:
        return body
    title = html.escape(
        str(summary["series"][0].get("displayTitle") or summary["title"])
    )
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{title}</title><style>{STANDALONE_CSS}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _sanitize_operation(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "unknown_operation"


def _perfetto_integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    integer = int(number)
    return integer if integer >= 0 else None


def import_perfetto(args: argparse.Namespace) -> None:
    _ensure_new_output(args.output, args.force)
    binary = args.trace_processor or shutil.which("trace_processor_shell") or shutil.which(
        "trace_processor"
    )
    if not binary:
        raise TraceError(
            "Perfetto Trace Processor is required; install its official wrapper or pass "
            "--trace-processor PATH"
        )
    sql = PERFETTO_SQL
    command = [binary, "query", str(args.trace), sql]
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise TraceError(f"Perfetto query failed: {detail}") from exc
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if not rows:
        raise TraceError("Perfetto trace contains no tagged completed slices")

    all_rows = []
    for row in rows:
        start = int(row["ts"])
        duration = int(row["dur"])
        if duration < 0:
            continue
        tags = []
        for source, target in PERFETTO_TAG_FIELDS:
            if _truthy(row.get(source)):
                tags.append(target)
        scope_tag = row.get("scope_tag") or None
        if scope_tag in PHASE_TAGS and scope_tag not in tags:
            tags.append(scope_tag)
        scope_kind = row.get("scope_kind") or "procedure"
        raw_component = row.get("component") or ""
        component = (
            raw_component
            if OPERATION_RE.fullmatch(raw_component)
            else _sanitize_operation(raw_component or row["name"])
        )
        priority = (
            "end-to-end",
            "witness-generation",
            "preparation",
            "commit",
            "constraint-proof",
            "opening-proof",
            "verification",
            "sumcheck",
            "fri",
            "proving",
            "pcs",
        )
        primary = next((tag for tag in priority if tag in tags), None)
        all_rows.append(
            {
                "raw_id": str(row["id"]),
                "raw_parent": str(row.get("parent_id") or ""),
                "name": row["name"],
                "component": component,
                "scope_kind": scope_kind,
                "scope_tag": scope_tag,
                "short_name": row.get("short_name") or None,
                "math_latex": [
                    value
                    for value in (
                        row.get("math_latex_1"),
                        row.get("math_latex_2"),
                    )
                    if value
                ],
                "start": start,
                "end": start + duration,
                "duration": duration,
                "track_id": str(row.get("track_id") or ""),
                "tags": [tag for tag in PHASE_TAGS if tag in tags],
                "primary": primary,
                "repeated": _truthy(row.get("tag_repeated")),
                "primary_sequence": _truthy(row.get("primary_sequence")),
                "overlay": _truthy(row.get("overlay")),
                "occurrence_index": row.get("occurrence_index"),
                "occurrence_count": row.get("occurrence_count"),
                "round_index": row.get("round_index"),
                "round_count": row.get("round_count"),
                "recursion_depth": row.get("recursion_depth"),
                "recursion_instance_index": row.get(
                    "recursion_instance_index"
                ),
            }
        )
    parsed_rows = [row for row in all_rows if row["tags"]]
    if not parsed_rows:
        raise TraceError("Perfetto trace has slices, but none carry recognized ZK tags")

    boundary = None
    if args.root_component:
        matches = [
            row for row in all_rows if row["component"] == args.root_component
        ]
        if not matches:
            raise TraceError(
                f"Perfetto trace has no component {args.root_component!r}"
            )
        boundary = max(matches, key=lambda row: (row["duration"], row["raw_id"]))
    else:
        matches = [
            row
            for row in parsed_rows
            if "end-to-end" in row["tags"] or row.get("scope_tag") == "end-to-end"
        ]
        if matches:
            boundary = max(matches, key=lambda row: (row["duration"], row["raw_id"]))

    if boundary is not None:
        root_start = boundary["start"]
        root_end = boundary["end"]
        parsed_rows = [
            row
            for row in parsed_rows
            if root_start <= row["start"] and row["end"] <= root_end
        ]
        if not parsed_rows:
            raise TraceError(
                f"root component {boundary['component']!r} contains no tagged slices"
            )
        boundary_basis = f"component:{boundary['component']}"
    else:
        root_start = min(row["start"] for row in parsed_rows)
        root_end = max(row["end"] for row in parsed_rows)
        boundary_basis = "tagged-envelope"
        print(
            "warning: no explicit end-to-end/root component; using the envelope of all "
            "tagged slices",
            file=sys.stderr,
        )
    trial = (
        {"kind": "warmup", "warmup_index": args.warmup_index}
        if args.warmup_index is not None
        else {"kind": "sample", "sample_index": args.sample_index}
    )
    parameters: dict[str, Any] = {}
    for item in args.parameter:
        if "=" not in item:
            raise TraceError(f"--parameter requires KEY=JSON, got {item!r}")
        key, raw_value = item.split("=", 1)
        try:
            value = json.loads(
                raw_value,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except (json.JSONDecodeError, TraceError) as exc:
            raise TraceError(f"invalid JSON value for --parameter {key!r}: {exc}") from exc
        parameters[key] = value

    run = {
        "schema": TRACE_SCHEMA,
        "record": "run",
        "run_id": args.run_id,
        "series_id": args.series_id,
        "root_span_id": "perfetto-root",
        "benchmark": {
            "suite": args.suite,
            "name": args.benchmark_name,
            "label": args.label or args.series_id,
            "algorithm": args.algorithm or args.label or args.benchmark_name,
            "implementation": args.implementation,
            "git_rev": args.git_rev,
            "build_profile": args.build_profile,
        },
        "trial": trial,
        "clock": {
            "id": f"perfetto:{args.trace.name}",
            "kind": "monotonic",
            "unit": "ns",
            "source": "Perfetto trace clock",
        },
        "status": args.status,
        "trace_complete": not args.incomplete,
        "environment": {"threads": args.threads} if args.threads else {},
        "parameters": parameters,
        "tags": {"root_boundary": boundary_basis},
    }
    root = {
        "schema": TRACE_SCHEMA,
        "record": "span",
        "run_id": args.run_id,
        "span_id": "perfetto-root",
        "parent_span_id": None,
        "operation": "proof.end-to-end",
        "name": args.label or args.benchmark_name,
        "primary_phase": "end-to-end",
        "phase_tags": ["end-to-end"],
        "start_ns": str(root_start),
        "end_ns": str(root_end),
        "duration_ns": str(root_end - root_start),
        "lane": {"process": args.implementation},
        "attributes": {"scope_kind": "scope", "scope_tag": "end-to-end"},
    }
    selected_ids = {row["raw_id"] for row in parsed_rows}
    selected_by_id = {row["raw_id"]: row for row in parsed_rows}
    all_by_id = {row["raw_id"]: row for row in all_rows}
    spans = []
    for row in parsed_rows:
        parent = row["raw_parent"]
        visited = set()
        while parent and parent not in selected_ids:
            if parent in visited:
                parent = ""
                break
            visited.add(parent)
            ancestor = all_by_id.get(parent)
            if ancestor is None:
                parent = ""
                break
            parent = ancestor["raw_parent"]
        parent = parent if parent in selected_ids else "perfetto-root"
        parent_is_boundary = parent == "perfetto-root" or (
            "end-to-end" in selected_by_id[parent]["tags"]
        )
        coordinate = {}
        for key in (
            "occurrence_index",
            "occurrence_count",
            "round_index",
            "round_count",
            "recursion_depth",
            "recursion_instance_index",
        ):
            parsed_value = _perfetto_integer(row.get(key))
            if parsed_value is not None:
                coordinate[key] = parsed_value
        spans.append(
            {
                "schema": TRACE_SCHEMA,
                "record": "span",
                "run_id": args.run_id,
                "span_id": f"perfetto-{row['raw_id']}",
                "parent_span_id": (
                    f"perfetto-{parent}" if parent != "perfetto-root" else parent
                ),
                "operation": row["component"],
                "name": row["name"],
                "primary_phase": row["primary"],
                "phase_tags": row["tags"],
                "start_ns": str(row["start"]),
                "end_ns": str(row["end"]),
                "duration_ns": str(row["duration"]),
                "lane": {"thread": row["track_id"]},
                "coordinate": coordinate,
                "attributes": {
                    "scope_kind": row["scope_kind"],
                    "scope_tag": row["scope_tag"],
                    "short_name": row["short_name"],
                    **(
                        {"math_latex": row["math_latex"]}
                        if row["math_latex"]
                        else {}
                    ),
                    "primary_sequence": (
                        row["primary_sequence"]
                        or row["scope_kind"] == "phase"
                        or (
                            row["scope_kind"] == "operation"
                            and "witness-generation" in row["tags"]
                            and parent_is_boundary
                        )
                    ),
                    "overlay": row["overlay"],
                    "repeated": row["repeated"],
                },
            }
        )
    output_lines = [run, root, *spans]
    serialized = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for item in output_lines
    )
    with tempfile.TemporaryDirectory(prefix="zkperf-import-") as temporary:
        validation_path = Path(temporary) / "import.jsonl"
        validation_path.write_text(serialized, encoding="utf-8")
        load_and_validate([validation_path])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(args.output)


def command_validate(args: argparse.Namespace) -> None:
    trace = load_and_validate(args.input)
    print(
        f"valid: {len(trace['runs'])} runs, "
        f"{sum(len(spans) for spans in trace['spans'].values())} spans"
    )
    for warning in trace["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    if args.warnings_as_errors and trace["warnings"]:
        raise TraceError("warnings treated as errors")


def command_summarize(args: argparse.Namespace) -> None:
    trace = load_and_validate(args.input)
    summary = build_summary(trace, args.title)
    _ensure_new_output(args.output, args.force)
    if args.csv:
        _ensure_new_output(args.csv, args.force)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_metrics_csv(summary, args.csv)
    print(args.output)


def command_render(args: argparse.Namespace) -> None:
    trace = load_and_validate(args.input)
    summary = build_summary(trace, args.title)
    _ensure_new_output(args.output, args.force)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(summary, args.fragment), encoding="utf-8")
    print(args.output)


def command_report(args: argparse.Namespace) -> None:
    trace = load_and_validate(args.input)
    summary = build_summary(trace, args.title)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "metrics.csv"
    html_path = args.out_dir / (
        "intervals.fragment.html" if args.fragment else "intervals.html"
    )
    for path in (summary_path, csv_path, html_path):
        _ensure_new_output(path, args.force)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_metrics_csv(summary, csv_path)
    html_path.write_text(render_html(summary, args.fragment), encoding="utf-8")
    print(summary_path)
    print(csv_path)
    print(html_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate canonical JSONL")
    validate.add_argument("input", nargs="+", type=Path)
    validate.add_argument("--warnings-as-errors", action="store_true")
    validate.set_defaults(func=command_validate)

    summarize = subparsers.add_parser("summarize", help="write normalized summary")
    summarize.add_argument("input", nargs="+", type=Path)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--csv", type=Path)
    summarize.add_argument("--title", default="ZK prover performance")
    summarize.add_argument("--force", action="store_true")
    summarize.set_defaults(func=command_summarize)

    render = subparsers.add_parser("render", help="render interactive interval HTML")
    render.add_argument("input", nargs="+", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--title", default="ZK prover performance")
    render.add_argument("--fragment", action="store_true")
    render.add_argument("--force", action="store_true")
    render.set_defaults(func=command_render)

    report = subparsers.add_parser("report", help="write summary, CSV, and HTML")
    report.add_argument("input", nargs="+", type=Path)
    report.add_argument("--out-dir", type=Path, required=True)
    report.add_argument("--title", default="ZK prover performance")
    report.add_argument("--fragment", action="store_true")
    report.add_argument("--force", action="store_true")
    report.set_defaults(func=command_report)

    importer = subparsers.add_parser(
        "import-perfetto", help="convert tagged Perfetto slices to canonical JSONL"
    )
    importer.add_argument("trace", type=Path)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--force", action="store_true")
    importer.add_argument("--trace-processor")
    importer.add_argument("--series-id", required=True)
    importer.add_argument("--run-id", required=True)
    trial = importer.add_mutually_exclusive_group(required=True)
    trial.add_argument("--sample-index", type=int)
    trial.add_argument("--warmup-index", type=int)
    importer.add_argument("--benchmark-name", required=True)
    importer.add_argument("--label")
    importer.add_argument(
        "--algorithm",
        help="human-readable algorithm placed in the comparison title",
    )
    importer.add_argument("--suite", default="zk")
    importer.add_argument("--implementation", required=True)
    importer.add_argument("--git-rev", default="unknown")
    importer.add_argument("--build-profile", default="release")
    importer.add_argument("--threads", type=int)
    importer.add_argument(
        "--root-component",
        help="stable component whose interval defines the imported run boundary",
    )
    importer.add_argument(
        "--status",
        choices=("ok", "error", "panic", "cancelled"),
        default="ok",
    )
    importer.add_argument("--incomplete", action="store_true")
    importer.add_argument("--parameter", action="append", default=[], metavar="KEY=JSON")
    importer.set_defaults(func=import_perfetto)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


PERFETTO_TAG_FIELDS = (
    ("tag_end_to_end", "end-to-end"),
    ("tag_witness_generation", "witness-generation"),
    ("tag_preparation", "preparation"),
    ("tag_proving", "proving"),
    ("tag_commit", "commit"),
    ("tag_pcs", "pcs"),
    ("tag_opening_proof", "opening-proof"),
    ("tag_constraint_proof", "constraint-proof"),
    ("tag_sumcheck", "sumcheck"),
    ("tag_fri", "fri"),
    ("tag_verification", "verification"),
)


PERFETTO_SQL = r"""
WITH tagged AS (
  SELECT
    s.id,
    s.parent_id,
    s.name,
    s.ts,
    s.dur,
    s.track_id,
    EXTRACT_ARG(s.arg_set_id, 'debug.component') AS component,
    EXTRACT_ARG(s.arg_set_id, 'debug.scope_kind') AS scope_kind,
    EXTRACT_ARG(s.arg_set_id, 'debug.scope_tag') AS scope_tag,
    EXTRACT_ARG(s.arg_set_id, 'debug.short_name') AS short_name,
    EXTRACT_ARG(s.arg_set_id, 'debug.math_latex_1') AS math_latex_1,
    EXTRACT_ARG(s.arg_set_id, 'debug.math_latex_2') AS math_latex_2,
    EXTRACT_ARG(s.arg_set_id, 'debug.primary_sequence') AS primary_sequence,
    EXTRACT_ARG(s.arg_set_id, 'debug.overlay') AS overlay,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_end_to_end') AS tag_end_to_end,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_witness_generation') AS tag_witness_generation,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_preparation') AS tag_preparation,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_proving') AS tag_proving,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_commit') AS tag_commit,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_pcs') AS tag_pcs,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_opening_proof') AS tag_opening_proof,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_constraint_proof') AS tag_constraint_proof,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_sumcheck') AS tag_sumcheck,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_fri') AS tag_fri,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_verification') AS tag_verification,
    EXTRACT_ARG(s.arg_set_id, 'debug.tag_repeated') AS tag_repeated,
    EXTRACT_ARG(s.arg_set_id, 'debug.occurrence_index') AS occurrence_index,
    EXTRACT_ARG(s.arg_set_id, 'debug.occurrence_count') AS occurrence_count,
    EXTRACT_ARG(s.arg_set_id, 'debug.round_index') AS round_index,
    EXTRACT_ARG(s.arg_set_id, 'debug.round_count') AS round_count,
    EXTRACT_ARG(s.arg_set_id, 'debug.recursion_depth') AS recursion_depth,
    EXTRACT_ARG(s.arg_set_id, 'debug.recursion_instance_index') AS recursion_instance_index
  FROM slice s
  WHERE s.dur >= 0
)
SELECT *
FROM tagged
WHERE component IS NOT NULL
   OR tag_end_to_end IS NOT NULL
   OR tag_witness_generation IS NOT NULL
   OR tag_preparation IS NOT NULL
   OR tag_proving IS NOT NULL
   OR tag_commit IS NOT NULL
   OR tag_pcs IS NOT NULL
   OR tag_opening_proof IS NOT NULL
   OR tag_constraint_proof IS NOT NULL
   OR tag_sumcheck IS NOT NULL
   OR tag_fri IS NOT NULL
   OR tag_verification IS NOT NULL
ORDER BY ts, id;
"""


STANDALONE_CSS = r"""
:root {
  color-scheme: light dark;
  --foreground: light-dark(#18181b, #f4f4f5);
  --muted-foreground: light-dark(#71717a, #a1a1aa);
  --muted: light-dark(#f4f4f5, #27272a);
  --border: light-dark(#d4d4d8, #3f3f46);
  --card: light-dark(#fafafa, #202023);
  --viz-series-1: #7c3aed;
  --viz-series-2: #f97316;
  --viz-series-3: #16a34a;
  --viz-series-4: #ec4899;
  --viz-series-5: #2563eb;
  --viz-series-6: #0891b2;
  font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
body { margin: 0; padding: 16px; color: var(--foreground); background: light-dark(white, #18181b); }
.tooltip { padding: 9px 11px; border: 1px solid var(--border); border-radius: 9px; background: var(--card); box-shadow: 0 8px 24px rgb(0 0 0 / .12); }
.btn, .form-select { font: inherit; border: 1px solid var(--border); border-radius: 8px; background: var(--card); color: var(--foreground); }
.btn { padding: 5px 9px; cursor: pointer; }
.btn-primary { color: white; background: #18181b; }
.form-select { min-height: 32px; padding: 4px 28px 4px 8px; }
.text-small { font-size: 12px; }
.text-muted { color: var(--muted-foreground); }
.viz-badge { display: inline-flex; padding: 3px 8px; border-radius: 999px; background: color-mix(in srgb, var(--foreground) 8%, transparent); color: var(--foreground); font-size: 12px; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; }
"""


HTML_FRAGMENT = r"""
<div id="__ROOT_ID__" class="zkpp-root">
  <header class="zkpp-heading">
    <h2 data-role="algorithm-title"></h2>
  </header>
  <div class="zkpp-toolbar">
    <div class="zkpp-controls" data-role="series-buttons" role="group" aria-label="Benchmark series"></div>
    <span class="text-small text-muted" data-role="run-meta"></span>
  </div>
  <div class="zkpp-kpis" data-role="kpis" role="group" aria-label="Selected timing totals"></div>
  <svg class="zkpp-chart" viewBox="0 0 1180 560" role="img" aria-labelledby="__ROOT_ID__-title __ROOT_ID__-desc">
    <title id="__ROOT_ID__-title" data-role="svg-title">ZK prover runtime intervals</title>
    <desc id="__ROOT_ID__-desc">Measured phase sequence and overlapping tagged protocol rows.</desc>
    <g data-role="chart-content"></g>
  </svg>
  <div class="tooltip zkpp-tooltip" data-role="tooltip" role="tooltip" aria-hidden="true">
    <strong data-role="tooltip-name"></strong>
    <div class="zkpp-math" data-role="tooltip-math"></div>
    <span class="text-small" data-role="tooltip-time"></span>
    <div class="zkpp-distribution" data-role="tooltip-distribution"></div>
    <span class="text-small text-muted" data-role="tooltip-meta"></span>
  </div>
  <div class="zkpp-inspector" role="group" aria-label="Interval details">
    <label class="zkpp-inspector-label" for="__ROOT_ID__-interval-select">
      <span>Inspect interval</span>
      <select class="form-select" id="__ROOT_ID__-interval-select" data-role="interval-select"></select>
    </label>
    <div class="zkpp-detail" aria-live="polite">
      <span class="text-small text-muted">Pinned interval</span>
      <strong data-role="detail-name"></strong>
      <div class="zkpp-math" data-role="detail-math"></div>
      <span data-role="detail-time"></span>
      <div class="zkpp-distribution" data-role="detail-distribution"></div>
      <span class="text-small text-muted" data-role="detail-meta"></span>
    </div>
  </div>
  <div class="zkpp-mobile" data-role="mobile-list" role="group" aria-label="Measured interval rows"></div>
  <div class="zkpp-badges" data-role="badges"></div>
  <div class="text-small text-muted zkpp-basis" data-role="basis"></div>
  <div class="sr-only" data-role="accessible-summary"></div>
</div>

<style>
  #__ROOT_ID__ { position: relative; width: 100%; color: var(--foreground, #18181b); }
  #__ROOT_ID__ .zkpp-heading h2 { margin: 0 0 8px; font-size: clamp(1.25rem, 2.4vw, 1.8rem); line-height: 1.2; text-wrap: balance; }
  #__ROOT_ID__ .zkpp-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  #__ROOT_ID__ .zkpp-controls { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
  #__ROOT_ID__ .zkpp-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-block: 12px 6px; }
  #__ROOT_ID__ .zkpp-card { display: grid; gap: 2px; min-width: 0; padding: 12px; border-radius: 12px; background: var(--card, var(--muted, #f4f4f5)); }
  #__ROOT_ID__ .zkpp-card strong { font-size: 1.35em; }
  #__ROOT_ID__ .zkpp-chart { display: block; width: 100%; height: auto; }
  #__ROOT_ID__ .zkpp-chart text { fill: var(--foreground, #18181b); font-size: 12px; pointer-events: none; }
  #__ROOT_ID__ .zkpp-chart .muted { fill: var(--muted-foreground, #71717a); font-size: 10px; }
  #__ROOT_ID__ .zkpp-chart .lane-label { font-weight: 600; }
  #__ROOT_ID__ .zkpp-chart .section { fill: var(--muted-foreground, #71717a); font-size: 10px; font-weight: 600; letter-spacing: .08em; }
  #__ROOT_ID__ .lane-bg { fill: color-mix(in srgb, var(--muted, #f4f4f5) 62%, transparent); }
  #__ROOT_ID__ .lane-bg-alt { fill: color-mix(in srgb, var(--muted, #f4f4f5) 38%, transparent); }
  #__ROOT_ID__ .guide { stroke: color-mix(in srgb, var(--border, #d4d4d8) 65%, transparent); stroke-width: 1; stroke-dasharray: 2 5; }
  #__ROOT_ID__ .axis, #__ROOT_ID__ .divider { stroke: var(--border, #d4d4d8); stroke-width: 1; }
  #__ROOT_ID__ .segment { stroke-width: 1; cursor: help; }
  #__ROOT_ID__ .seg-end-to-end, #__ROOT_ID__ .seg-proving { fill: color-mix(in srgb, var(--foreground, #18181b) 7%, transparent); stroke: color-mix(in srgb, var(--foreground, #18181b) 38%, transparent); }
  #__ROOT_ID__ .seg-witness-generation { fill: color-mix(in srgb, var(--viz-series-5, #2563eb) 23%, transparent); stroke: var(--viz-series-5, #2563eb); }
  #__ROOT_ID__ .seg-preparation { fill: color-mix(in srgb, var(--viz-series-1, #7c3aed) 23%, transparent); stroke: var(--viz-series-1, #7c3aed); }
  #__ROOT_ID__ .seg-commit { fill: color-mix(in srgb, var(--viz-series-2, #f97316) 23%, transparent); stroke: var(--viz-series-2, #f97316); }
  #__ROOT_ID__ .seg-pcs, #__ROOT_ID__ .seg-opening-proof, #__ROOT_ID__ .seg-fri { fill: color-mix(in srgb, var(--viz-series-4, #ec4899) 22%, transparent); stroke: var(--viz-series-4, #ec4899); }
  #__ROOT_ID__ .seg-constraint-proof { fill: color-mix(in srgb, var(--viz-series-3, #16a34a) 22%, transparent); stroke: var(--viz-series-3, #16a34a); }
  #__ROOT_ID__ .seg-sumcheck { fill: color-mix(in srgb, var(--viz-series-6, #0891b2) 20%, transparent); stroke: var(--viz-series-6, #0891b2); }
  #__ROOT_ID__ .seg-verification { fill: color-mix(in srgb, var(--viz-series-1, #7c3aed) 18%, transparent); stroke: var(--viz-series-1, #7c3aed); }
  #__ROOT_ID__ .overlay { fill-opacity: .35; stroke-dasharray: 3 2; }
  #__ROOT_ID__ .micro { stroke: var(--foreground, #18181b); stroke-width: 1.2; }
  #__ROOT_ID__ .zkpp-tooltip { position: absolute; z-index: 20; width: min(640px, calc(100% - 16px)); max-width: min(640px, calc(100% - 16px)); visibility: hidden; opacity: 0; pointer-events: none; }
  #__ROOT_ID__ .zkpp-tooltip.is-visible { visibility: visible; opacity: 1; }
  #__ROOT_ID__ .zkpp-tooltip > * { display: block; }
  #__ROOT_ID__ .zkpp-math { display: grid; gap: 3px; margin-block: 5px; min-height: 0; overflow-x: auto; }
  #__ROOT_ID__ .zkpp-math:empty { display: none; }
  #__ROOT_ID__ .zkpp-math-line { font: 16px/1.35 ui-serif, Georgia, "Times New Roman", serif; white-space: nowrap; }
  #__ROOT_ID__ .zkpp-math-line code { font: inherit; }
  #__ROOT_ID__ .zkpp-distribution { display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 0; margin-block: 8px 5px; border-top: 1px solid var(--border, #d4d4d8); }
  #__ROOT_ID__ .zkpp-decile { position: relative; display: grid; gap: 1px; padding-top: 9px; text-align: center; color: var(--muted-foreground, #71717a); }
  #__ROOT_ID__ .zkpp-decile::before { content: ""; position: absolute; top: -5px; left: 50%; height: 9px; border-left: 1px solid currentColor; }
  #__ROOT_ID__ .zkpp-decile.is-median { color: var(--foreground, #18181b); font-weight: 650; }
  #__ROOT_ID__ .zkpp-decile.is-median::before { border-left-width: 3px; border-color: var(--viz-series-5, #2563eb); }
  #__ROOT_ID__ .zkpp-decile-label, #__ROOT_ID__ .zkpp-decile-value { font-size: 10px; white-space: nowrap; }
  #__ROOT_ID__ .zkpp-inspector { display: grid; grid-template-columns: minmax(220px, .7fr) minmax(0, 1.3fr); gap: 12px; align-items: end; margin-top: 8px; }
  #__ROOT_ID__ .zkpp-inspector-label, #__ROOT_ID__ .zkpp-detail { display: grid; gap: 4px; min-width: 0; }
  #__ROOT_ID__ .zkpp-inspector .form-select { width: 100%; max-width: 100%; }
  #__ROOT_ID__ .zkpp-detail [data-role="detail-meta"] { overflow-wrap: anywhere; }
  #__ROOT_ID__ .zkpp-badges { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 9px; }
  #__ROOT_ID__ .viz-badge { display: inline-flex; padding: 3px 8px; border-radius: 999px; background: color-mix(in srgb, var(--foreground, #18181b) 8%, transparent); color: var(--foreground, #18181b); font-size: 12px; }
  #__ROOT_ID__ .zkpp-basis { margin-top: 7px; }
  #__ROOT_ID__ .zkpp-mobile { display: none; }
  @media (max-width: 720px) {
    #__ROOT_ID__ .zkpp-kpis { grid-template-columns: 1fr 1fr; }
    #__ROOT_ID__ .zkpp-chart { display: none; }
    #__ROOT_ID__ .zkpp-tooltip { display: none; }
    #__ROOT_ID__ .zkpp-inspector { grid-template-columns: 1fr; }
    #__ROOT_ID__ .zkpp-mobile { display: grid; gap: 14px; margin-top: 14px; }
    #__ROOT_ID__ .mobile-group { display: grid; gap: 7px; }
    #__ROOT_ID__ .mobile-heading { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; }
    #__ROOT_ID__ .mobile-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 8px; }
    #__ROOT_ID__ .mobile-track { grid-column: 1 / -1; position: relative; height: 9px; background: color-mix(in srgb, var(--muted, #f4f4f5) 70%, transparent); }
    #__ROOT_ID__ .mobile-bar { position: absolute; inset-block: 0; border: 1px solid currentColor; background: color-mix(in srgb, currentColor 22%, transparent); box-sizing: border-box; }
  }
</style>

<script>
(() => {
  const report = __REPORT_DATA__;
  const root = document.getElementById("__ROOT_ID__");
  const chart = root.querySelector('[data-role="chart-content"]');
  const svg = root.querySelector(".zkpp-chart");
  const tooltip = root.querySelector('[data-role="tooltip"]');
  const intervalSelect = root.querySelector('[data-role="interval-select"]');
  const NS = "http://www.w3.org/2000/svg";
  const X0 = 210, X1 = 1160, WIDTH = X1 - X0;
  let options = [];
  let pinnedIntervalId = "";
  let pinnedOperation = "";

  function el(name, attrs = {}, text) {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function add(parent, name, attrs = {}, text) {
    const node = el(name, attrs, text);
    parent.appendChild(node);
    return node;
  }
  function formatTime(ms) {
    if (ms >= 1000) return `${(ms / 1000).toFixed(3)} s`;
    if (ms >= 100) return `${ms.toFixed(1)} ms`;
    if (ms >= 10) return `${ms.toFixed(2)} ms`;
    if (ms >= 1) return `${ms.toFixed(3)} ms`;
    if (ms >= .01) return `${(ms * 1000).toFixed(1)} µs`;
    return `${(ms * 1000).toFixed(2)} µs`;
  }
  function renderMath(container, expressions) {
    container.replaceChildren();
    (expressions || []).slice(0, 2).forEach(expression => {
      const line = document.createElement("div");
      line.className = "zkpp-math-line";
      if (globalThis.katex && typeof globalThis.katex.render === "function") {
        globalThis.katex.render(expression, line, { throwOnError: false, displayMode: false });
      } else {
        const code = document.createElement("code");
        code.textContent = "\\(" + expression + "\\)";
        line.appendChild(code);
      }
      container.appendChild(line);
    });
  }
  function renderDistribution(container, info) {
    container.replaceChildren();
    const values = info.decilesMsExact || {};
    for (let percentile = 10; percentile < 100; percentile += 10) {
      const cell = document.createElement("span");
      cell.className = "zkpp-decile" + (percentile === 50 ? " is-median" : "");
      const label = document.createElement("span");
      label.className = "zkpp-decile-label";
      label.textContent = "P" + percentile;
      const value = document.createElement("span");
      value.className = "zkpp-decile-value";
      value.textContent = (values["p" + percentile] || "—") + " ms";
      cell.append(label, value);
      container.appendChild(cell);
    }
  }
  function rowInfo(series) {
    return [
      { tag: "phase-sequence", label: "Measured phase sequence", metric: series.metrics["end-to-end"], intervals: series.primary, section: "sequence" },
      { tag: "end-to-end", label: "End-to-end", metric: series.metrics["end-to-end"], intervals: [series.root], section: "primary" },
      ...series.rows
    ];
  }
  function infoMeta(info) {
    const recurrence = [];
    if (info.occurrenceIndex != null) recurrence.push(`occurrence ${info.occurrenceIndex}`);
    if (info.occurrenceCount != null) recurrence.push(`${info.occurrenceCount} occurrences`);
    if (info.roundIndex != null) recurrence.push(`round ${info.roundIndex}`);
    if (info.roundCount != null) recurrence.push(`${info.roundCount} rounds`);
    if (info.recursionDepth != null) recurrence.push(`depth ${info.recursionDepth}`);
    if (info.recursionInstanceIndex != null) recurrence.push(`recursion instance ${info.recursionInstanceIndex}`);
    return [info.rowLabel, info.scopeKind, ...info.tags, ...recurrence].join(" · ");
  }
  function showTooltip(mark, info) {
    root.querySelector('[data-role="tooltip-name"]').textContent = info.name;
    renderMath(root.querySelector('[data-role="tooltip-math"]'), info.mathLatex);
    root.querySelector('[data-role="tooltip-time"]').textContent =
      `Raw median: ${info.medianMsExact} ms (n=${info.medianN}) · representative interval: ${info.representativeMsExact} ms`;
    root.querySelector('[data-role="tooltip-meta"]').textContent = infoMeta(info);
    renderDistribution(root.querySelector('[data-role="tooltip-distribution"]'), info);
    tooltip.classList.add("is-visible");
    tooltip.setAttribute("aria-hidden", "false");
    tooltip.style.visibility = "hidden";
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    const rootBox = root.getBoundingClientRect();
    const markBox = mark.getBoundingClientRect();
    const tipBox = tooltip.getBoundingClientRect();
    const centered = markBox.left - rootBox.left + markBox.width / 2 - tipBox.width / 2;
    const left = Math.max(8, Math.min(centered, rootBox.width - tipBox.width - 8));
    let top = markBox.top - rootBox.top - tipBox.height - 8;
    if (top < 8) top = markBox.bottom - rootBox.top + 8;
    top = Math.max(8, Math.min(top, rootBox.height - tipBox.height - 8));
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.style.visibility = "visible";
  }
  function hideTooltip() {
    tooltip.classList.remove("is-visible");
    tooltip.setAttribute("aria-hidden", "true");
    tooltip.style.visibility = "hidden";
  }
  function pin(info) {
    if (!info) return;
    pinnedIntervalId = info.id;
    pinnedOperation = info.operation;
    root.querySelector('[data-role="detail-name"]').textContent = info.name;
    renderMath(root.querySelector('[data-role="detail-math"]'), info.mathLatex);
    root.querySelector('[data-role="detail-time"]').textContent =
      `Raw median: ${info.medianMsExact} ms · representative interval: ${info.representativeMsExact} ms`;
    root.querySelector('[data-role="detail-meta"]').textContent = infoMeta(info);
    renderDistribution(root.querySelector('[data-role="detail-distribution"]'), info);
    const index = options.findIndex(candidate => candidate.key === info.key);
    if (index >= 0) intervalSelect.value = String(index);
  }
  function register(interval, rowLabel) {
    const info = { ...interval, key: interval.id, rowLabel };
    options.push(info);
    return info;
  }
  function drawSegment(series, interval, rowLabel, y, height = 27) {
    const totalMs = series.medianTotalMs > 0 ? series.medianTotalMs : 1;
    const x = X0 + (interval.startMs / totalMs) * WIDTH;
    const trueWidth = ((interval.endMs - interval.startMs) / totalMs) * WIDTH;
    const width = Math.max(.65, trueWidth);
    const tag = interval.rowTag === "phase-sequence" ? interval.primaryPhase || interval.tags.find(tag => tag !== "proving") || "proving" : interval.rowTag;
    const rect = add(chart, "rect", {
      x, y, width, height, rx: 3,
      class: `segment seg-${tag}${interval.overlay ? " overlay" : ""}`
    });
    const info = register(interval, rowLabel);
    add(rect, "title", {}, `${interval.name}: ${interval.medianMsExact} ms`);
    rect.addEventListener("pointerenter", () => showTooltip(rect, info));
    rect.addEventListener("pointerleave", hideTooltip);
    rect.addEventListener("click", () => pin(info));
    const label = interval.shortName || interval.name;
    if (width >= label.length * 6 + 12) {
      add(chart, "text", { x: x + width / 2, y: y + height / 2 + 4, "text-anchor": "middle" }, label);
    }
    if (trueWidth < 1.5) {
      add(chart, "line", { x1: x + trueWidth / 2, x2: x + trueWidth / 2, y1: y - 2, y2: y + height + 2, class: "micro" });
    }
  }
  function laneLabel(y, label, metric) {
    add(chart, "text", { x: 10, y: y + 18, class: "lane-label" }, label);
    add(chart, "text", { x: 198, y: y + 18, "text-anchor": "end", class: "muted" }, `${Number(metric.medianMsExact).toFixed(3)} ms`);
  }
  function renderChart(series) {
    hideTooltip();
    options = [];
    chart.replaceChildren();
    add(chart, "text", { x: 10, y: 25, class: "section" }, "MEASURED PHASE SEQUENCE");
    series.primary.forEach(interval => drawSegment(series, interval, "Measured phase sequence", 43, 45));
    [0, .25, .5, .75, 1].forEach(ratio => {
      const x = X0 + WIDTH * ratio;
      add(chart, "line", { x1: x, x2: x, y1: 30, y2: 520, class: "guide" });
      add(chart, "text", { x, y: 119, "text-anchor": ratio === 0 ? "start" : ratio === 1 ? "end" : "middle", class: "muted" }, formatTime(series.medianTotalMs * ratio));
    });
    add(chart, "line", { x1: X0, x2: X1, y1: 105, y2: 105, class: "axis" });
    let y = 128;
    const rows = rowInfo(series).slice(1);
    let overlapStarted = false;
    rows.forEach((row, index) => {
      if (row.section === "overlap" && !overlapStarted) {
        y += 22;
        add(chart, "line", { x1: 0, x2: 1172, y1: y - 9, y2: y - 9, class: "divider" });
        add(chart, "text", { x: 10, y: y - 14, class: "section" }, "OVERLAPPING TAGGED SCOPES");
        overlapStarted = true;
      }
      add(chart, "rect", { x: 0, y: y - 2, width: 1172, height: 31, rx: 4, class: index % 2 ? "lane-bg-alt" : "lane-bg" });
      laneLabel(y, row.label, row.metric);
      row.intervals.forEach(interval => drawSegment(series, interval, row.label, y));
      y += 39;
    });
    add(chart, "line", { x1: X0, x2: X1, y1: y + 3, y2: y + 3, class: "axis" });
    add(chart, "text", { x: X0, y: y + 22, class: "muted" }, "proof start");
    add(chart, "text", { x: X1, y: y + 22, "text-anchor": "end", class: "muted" }, "measured boundary complete");
    svg.setAttribute("viewBox", `0 0 1180 ${y + 42}`);

    intervalSelect.replaceChildren();
    options.forEach((info, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${info.rowLabel}: ${info.name} — ${info.medianMsExact} ms`;
      intervalSelect.appendChild(option);
    });
    const preferred =
      options.find(info => info.id === pinnedIntervalId) ||
      options.find(info => info.operation === pinnedOperation) ||
      options[0];
    pin(preferred);
  }
  function renderKpis(series) {
    const container = root.querySelector('[data-role="kpis"]');
    container.replaceChildren();
    [
      ["End-to-end", "end-to-end"],
      ["Total proving", "proving"],
      ["Total commit", "commit"],
      ["Total PCS", "pcs"]
    ].forEach(([label, tag]) => {
      const metric = series.metrics[tag];
      if (!metric) return;
      const card = document.createElement("div");
      card.className = "zkpp-card";
      const small = document.createElement("span");
      small.className = "text-small text-muted";
      small.textContent = label;
      const value = document.createElement("strong");
      value.textContent = formatTime(Number(metric.medianMsExact));
      const note = document.createElement("span");
      note.className = "text-small text-muted";
      note.textContent = `raw union median · n=${metric.n}`;
      card.append(small, value, note);
      container.appendChild(card);
    });
  }
  function renderMobile(series) {
    const mobile = root.querySelector('[data-role="mobile-list"]');
    mobile.replaceChildren();
    rowInfo(series).forEach(row => {
      const group = document.createElement("section");
      group.className = "mobile-group";
      const heading = document.createElement("div");
      heading.className = "mobile-heading";
      const label = document.createElement("span");
      label.textContent = row.label;
      const total = document.createElement("span");
      total.className = "text-small text-muted";
      total.textContent = `${row.metric.medianMsExact} ms`;
      heading.append(label, total);
      group.appendChild(heading);
      row.intervals.forEach(interval => {
        const item = document.createElement("div");
        item.className = "mobile-row";
        const name = document.createElement("span");
        name.textContent = interval.name;
        const value = document.createElement("span");
        value.className = "text-small text-muted";
        value.textContent = `${interval.medianMsExact} ms`;
        const track = document.createElement("div");
        track.className = "mobile-track";
        const bar = document.createElement("span");
        bar.className = "mobile-bar";
        bar.style.color = "var(--viz-series-5, #2563eb)";
        const totalMs = series.medianTotalMs > 0 ? series.medianTotalMs : 1;
        bar.style.left = `${Math.max(0, interval.startMs / totalMs * 100)}%`;
        bar.style.width = `${Math.max(.15, (interval.endMs - interval.startMs) / totalMs * 100)}%`;
        track.appendChild(bar);
        item.append(name, value, track);
        group.appendChild(item);
      });
      mobile.appendChild(group);
    });
  }
  function renderBadges(series) {
    const container = root.querySelector('[data-role="badges"]');
    container.replaceChildren();
    series.badges.forEach(item => {
      const badge = document.createElement("span");
      badge.className = "viz-badge";
      badge.textContent = `${item.label}: ${item.value}`;
      container.appendChild(badge);
    });
  }
  function update(seriesId) {
    const series = report.series.find(item => item.seriesId === seriesId) || report.series[0];
    root.querySelector('[data-role="algorithm-title"]').textContent = series.displayTitle;
    root.querySelector('[data-role="svg-title"]').textContent = series.displayTitle + " runtime intervals";
    root.querySelectorAll("[data-series]").forEach(button => {
      const active = button.dataset.series === series.seriesId;
      button.classList.toggle("btn-primary", active);
      button.setAttribute("aria-pressed", String(active));
    });
    root.querySelector('[data-role="run-meta"]').textContent =
      `${series.warmupN} warmup · median of ${series.measuredN} · representative ${series.representativeRunId}`;
    root.querySelector('[data-role="basis"]').textContent =
      `Geometry uses representative run ${series.representativeRunId}, scaled ×${series.geometryScale.toFixed(6)} to the median end-to-end. Root boundary: ${series.rootBoundary}. Tooltips show warmup-excluded per-operation P10–P90 sample deciles using Hyndman–Fan Type 7 (not confidence intervals). Tagged rows are interval unions and are not additive.${series.pcsDerived ? " PCS is derived as commit ∪ opening proof." : ""}${series.rootBoundary === "tagged-envelope" ? " End-to-end is inferred from the tagged envelope; leading or trailing untagged work may be excluded." : ""}`;
    root.querySelector('[data-role="accessible-summary"]').textContent =
      `${series.displayTitle}: end-to-end ${series.medianTotalMsExact} milliseconds, ${series.measuredN} measured samples.`;
    renderKpis(series);
    renderChart(series);
    renderMobile(series);
    renderBadges(series);
  }

  const buttons = root.querySelector('[data-role="series-buttons"]');
  report.series.forEach((series, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `btn${index === 0 ? " btn-primary" : ""}`;
    button.dataset.series = series.seriesId;
    button.setAttribute("aria-pressed", String(index === 0));
    button.textContent = series.label;
    button.addEventListener("click", () => update(series.seriesId));
    buttons.appendChild(button);
  });
  intervalSelect.addEventListener("change", () => pin(options[Number(intervalSelect.value)]));
  update(report.series[0].seriesId);
})();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
