"""Validate the current multiplication format and aggregate raw, verified trials."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from bench_statistics import percentile, sample_statistics

SCHEMA = "mul-bench/v2"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}: {value!r}")
    return value




@dataclass
class Case:
    source: Path
    job: dict
    effective: dict
    provenance: dict
    status: str
    reason: str | None
    records: list[dict]

    @property
    def identity(self):
        return canonical(dict(case=self.job["case"], effective=self.effective,
                              provenance=self.provenance))


def load(directory: Path) -> list[Case]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError(f"{directory}: expected a complete {SCHEMA} campaign")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("missing campaign provenance")
    cases = {}
    identities = set()
    for entry in manifest["cases"]:
        job = entry["job"]
        identifier = job["id"]
        descriptor = job["case"]
        if "f2z" in descriptor or str(descriptor.get("backend", "")).startswith("f2z"):
            raise ValueError("legacy F2Z results are unsupported; run a new BitZ campaign")
        if identifier in cases or canonical(descriptor) in identities:
            raise ValueError(f"duplicate case {identifier}")
        for name, minimum in [("log_n", 4), ("seed", 0), ("threads", 1)]:
            integer(descriptor.get(name), name, minimum)
        if descriptor["seed"] >= 2**64 or descriptor["log_n"] > 63:
            raise ValueError("invalid seed or size")
        if not all(descriptor.get(k) for k in ("mode", "workload", "backend")):
            raise ValueError("incomplete case identity")
        if descriptor["backend"] == "bitz" and descriptor["mode"] not in ("outer", "piop"):
            config = descriptor.get("bitz", {})
            if not all(k in config for k in ("w", "split", "profile", "bound", "ligerito", "gkr_schedule")):
                raise ValueError("incomplete BitZ configuration")
            if descriptor["mode"] != "witness" and not all(config.get(k) for k in ("profile", "bound", "ligerito")):
                raise ValueError("missing BitZ proof configuration")
        if config := descriptor.get("bitz"):
            schedule = config.get("gkr_schedule")
            if descriptor["mode"] == "witness":
                if schedule is not None:
                    raise ValueError("witness generation cannot have a GKR schedule")
            elif schedule not in ("auto", "l2", "l4", "l8"):
                raise ValueError("missing or invalid GKR schedule policy")
        if type(job.get("proof_fingerprints")) is not bool:
            raise ValueError("missing fingerprint policy")
        integer(job.get("reps"), "reps", 1)
        integer(job.get("warmups"), "warmups")
        if job.get("memory") not in ("none", "rss", "heap"):
            raise ValueError("unknown memory mode")
        status = entry.get("status")
        if status not in ("measured", "skipped"):
            raise ValueError(f"invalid case status {status}")
        if status == "skipped" and not entry.get("reason"):
            raise ValueError("skip has no reason")
        effective = entry.get("effective", {})
        if status == "measured" and (not effective.get("boundary") or not
                (effective.get("corpus_digest") or effective.get("fixture_digest"))):
            raise ValueError("missing measured configuration or corpus identity")
        if status == "measured" and descriptor.get("bitz", {}).get("gkr_schedule"):
            schedules = effective.get("gkr_schedules")
            if not isinstance(schedules, list) or not schedules:
                raise ValueError("missing resolved GKR schedules")
            keys = set()
            for resolved in schedules:
                key = canonical({k:v for k,v in resolved.items() if k != "schedule"})
                if key in keys or resolved.get("schedule") not in ("l2", "l4", "l8") or resolved.get("path") not in ("single", "multi"):
                    raise ValueError("invalid or duplicate resolved GKR schedule")
                keys.add(key)
                if resolved["threads"] != descriptor["threads"] or (resolved["path"] == "multi" and resolved["schedule"] == "l2"):
                    raise ValueError("inconsistent resolved GKR schedule")
                for name in ("row_vars", "col_vars", "word_bits", "threads"):
                    integer(resolved.get(name), name, 1 if name in ("word_bits", "threads") else 0)
                width = resolved["word_bits"]
                if width & (width - 1):
                    raise ValueError("physical GKR word width must be a power of two")
                if config["gkr_schedule"] != "auto" and resolved["schedule"] != config["gkr_schedule"]:
                    raise ValueError("explicit GKR schedule was substituted")
        cases[identifier] = Case(directory, job, effective, provenance, status,
                                 entry.get("reason"), [])
        identities.add(canonical(descriptor))
    seen = set()
    for number, line in enumerate((directory / "samples.jsonl").read_text().splitlines(), 1):
        record = json.loads(line)
        identifier = record["case_id"]
        if identifier not in cases or cases[identifier].status != "measured":
            raise ValueError(f"sample {number} references an absent/skipped case")
        index = integer(record.get("index"), "sample index")
        kind = record.get("kind")
        key = (identifier, kind, index)
        if key in seen:
            raise ValueError(f"duplicate sample {key}")
        if record.get("verified") is not True:
            raise ValueError(f"unverified sample {key}")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"missing metrics {key}")
        for name, value in metrics.items():
            finite_number(value, name)
        seen.add(key)
        cases[identifier].records.append(record)
    for identifier, case in cases.items():
        job = case.job
        expected = set()
        if case.status == "measured":
            if job["memory"] != "heap":
                expected |= {(identifier, "sample", i) for i in range(job["reps"])}
                expected |= {(identifier, "warmup", i) for i in range(job["warmups"])}
            if job["memory"] != "none":
                expected.add((identifier, job["memory"], 0))
        actual = {key for key in seen if key[0] == identifier}
        if actual != expected:
            raise ValueError(f"{identifier}: missing or unexpected samples: {actual ^ expected}")
        measured = [r for r in case.records if r["kind"] == "sample"]
        if measured and any(set(r["metrics"]) != set(measured[0]["metrics"]) for r in measured):
            raise ValueError(f"{identifier}: inconsistent sample metrics")
        required = {
            "standalone-proving": {"commit_ms", "online_prover_ms", "verify_ms", "proof_bytes"},
            "witness-to-proof": {"witness_ms", "commit_ms", "online_prover_ms", "witness_to_proof_ms", "verify_ms", "verified_trial_ms", "proof_bytes"},
            "witness-generation": {"witness_ms"},
            "pcs-opening": {"materialize_ms", "commit_ms", "claim_ms", "opening_ms", "verify_ms", "verified_trial_ms", "pcs_ms", "commitment_bytes", "claim_bytes", "opening_bytes", "proof_bytes"},
            "whole-piop": {"piop_ms", "verify_ms", "analytical_piop_bytes"},
            "outer-kernel": {"outer_ms"},
            "outer-regression": {"production_ms", "generic_ms"},
        }.get(case.effective.get("boundary"))
        if case.status == "measured" and required is None:
            raise ValueError(f"{identifier}: unknown measurement boundary")
        for record in case.records:
            if record["kind"] in ("sample", "warmup"):
                fingerprint = record.get("fingerprint")
                if job["proof_fingerprints"]:
                    if not isinstance(fingerprint, dict) or set(fingerprint) != {"proof", "transcript"} or any(
                        not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                        for value in fingerprint.values()
                    ):
                        raise ValueError("missing or invalid proof fingerprint")
                elif fingerprint is not None:
                    raise ValueError("unexpected proof fingerprint")
                if not required <= record["metrics"].keys():
                    raise ValueError(f"{identifier}: missing required metrics: {required - record['metrics'].keys()}")
                if "proof_bytes" in required and record["metrics"]["proof_bytes"] <= 0:
                    raise ValueError(f"{identifier}: missing verified proof size")
            if record["kind"] in ("rss", "heap"):
                if record.get("fingerprint") is not None:
                    raise ValueError("memory workers cannot generate fingerprints")
                metric = "peak_rss_bytes" if record["kind"] == "rss" else "peak_heap_bytes"
                if set(record["metrics"]) != {metric} or record["metrics"][metric] <= 0:
                    raise ValueError(f"{identifier}: invalid memory record")
    return list(cases.values())


def select(cases, filters):
    for case in cases:
        values = {**case.job["case"], **case.job["case"].get("bitz", {})}
        if all(str(values.get(key)) == value for key, value in filters.items()):
            yield case


def aggregate(cases):
    rows = []
    seen = set()
    for case in cases:
        # Re-reading the same campaign is an error, not another repetition.
        key = (str(case.source.resolve()), case.job["id"])
        if key in seen:
            raise ValueError(f"duplicate campaign case {key}")
        seen.add(key)
        measurements = [r["metrics"] for r in case.records if r["kind"] == "sample"]
        stats = {}
        if measurements:
            for metric in measurements[0]:
                values = [m[metric] for m in measurements]
                stats[metric] = sample_statistics(values)
        memory = {k: v for r in case.records if r["kind"] in ("rss", "heap")
                  for k, v in r["metrics"].items()}
        rows.append(dict(source=str(case.source), case_id=case.job["id"],
                         case=case.job["case"], effective=case.effective,
                         provenance=case.provenance, status=case.status,
                         reason=case.reason, metrics=stats, memory=memory))
    return rows
