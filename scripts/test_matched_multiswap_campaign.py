#!/usr/bin/env python3
"""Fixture-only tests for the matched MultiSwap campaign tooling."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matched_multiswap_report as report
import run_matched_multiswap_campaign as runner


class BenchmarkDomainMigrationTests(unittest.TestCase):
    def test_updates_dependency_hash_tags_and_records_source_identity(self):
        from prepare_matched_limber import migrate_multiswap_domains
        domains = (
            "f2z/multiswap/circuit-digest/v1",
            "f2z/multiswap/integer-assignment/v1",
            "f2z-limber/multiswap-statement/v2",
        )
        original = "// Keep the proof implementation untouched.\n" + "\n".join(
            f'hasher.update(b"{domain}"); let label = "{domain}";' for domain in domains)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "benches/multiswap_modp.rs"
            source.parent.mkdir()
            source.write_text(original)
            result = migrate_multiswap_domains(root)
            self.assertTrue(result["changed"])
            self.assertEqual(source.read_text(), original.replace("f2z", "bitz"))
            self.assertEqual(result["input_sha256"], hashlib.sha256(original.encode()).hexdigest())
            self.assertEqual(result["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertFalse(migrate_multiswap_domains(root)["changed"])

    def test_rejects_incompatible_source_without_changing_it(self):
        from prepare_matched_limber import migrate_multiswap_domains
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "benches/multiswap_modp.rs"
            source.parent.mkdir()
            source.write_text('hasher.update(b"f2z/multiswap/circuit-digest/v1");')
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "lacks matched digest domain"):
                migrate_multiswap_domains(root)
            self.assertEqual(source.read_bytes(), original)


DOMAIN = "bitz/multiswap/circuit-digest/v1"
DIGEST = "ab" * 32
ASSIGNMENT = "cd" * 32


def _span(
    run_id: str,
    span_id: str,
    operation: str,
    name: str,
    start: int,
    end: int,
    phase: str,
    parent: str | None,
    math: list[str] | None = None,
) -> dict[str, object]:
    attributes: dict[str, object] = {"scope_kind": "operation"}
    if math:
        attributes["math_latex"] = math
    return {
        "schema": report.TRACE_SCHEMA,
        "record": "span",
        "run_id": run_id,
        "span_id": span_id,
        "parent_span_id": parent,
        "operation": operation,
        "name": name,
        "primary_phase": phase,
        "phase_tags": [phase],
        "start_ns": str(start),
        "end_ns": str(end),
        "duration_ns": str(end - start),
        "attributes": attributes,
    }


def _records(
    cell_id: str,
    implementation: str,
    digest: str = DIGEST,
    threads: int = 1,
    workload_k: int = 0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    trials = [("warmup", 0), *(('sample', index) for index in range(5))]
    for kind, index in trials:
        run_id = f"{cell_id}-{kind}-{index}"
        root_id = f"{run_id}-root"
        scale = 1 if kind == "warmup" else index + 1
        statement = {"domain": DOMAIN, "digest_blake3": digest}
        parameters: dict[str, object]
        if implementation == "bitz-ligerito":
            parameters = {
                "input": {
                    "workload_id": "multiswap-rsa-wired-cost-model-v1",
                    "limber_k": workload_k,
                    "constraint_digest_domain": DOMAIN,
                    "constraint_digest_blake3": digest,
                    "live_rows": 100 + workload_k,
                    "live_columns": 96 + workload_k,
                    "padded_rows": 128,
                    "padded_columns": 128,
                    "nnz_a": 200 + workload_k,
                    "nnz_b": 210 + workload_k,
                    "nnz_c": 220 + workload_k,
                    "witness_stats": {
                        "assignment_digest_domain": report.ASSIGNMENT_DOMAIN,
                        "assignment_digest_blake3": ASSIGNMENT,
                    },
                }
            }
            operations = {
                "root": "multiswap-trace.verified_trial",
                "witness": "multiswap-trace.witness_generation",
                "commit": "multiswap-trace.commit",
                "prover": "multiswap-trace.end_to_end_prove",
                "projection": "step2.project_prove",
                "piop": "step3.piop_prove",
                "open": "step5.open_prove",
                "verify": "multiswap-trace.verification",
            }
        else:
            parameters = {
                "input": {
                    "workload_id": "multiswap-rsa-wired-cost-model-v1",
                    "limber_k": workload_k,
                    "live_rows": 100 + workload_k,
                    "live_cols": 96 + workload_k,
                    "num_cons": 128,
                    "num_vars": 128,
                    "a_nnz": 200 + workload_k,
                    "b_nnz": 210 + workload_k,
                    "c_nnz": 220 + workload_k,
                },
                "statement": {
                    **statement,
                    "assignment_digest_domain": report.ASSIGNMENT_DOMAIN,
                    "assignment_digest_blake3": ASSIGNMENT,
                }
            }
            operations = {
                "root": "multiswap.trial",
                "witness": "multiswap.witness_generation",
                "commit": "multiswap.commit",
                "prover": "multiswap.prover",
                "projection": "limber.projection",
                "piop": "limber.piop",
                "open": "limber.pcs.opening",
                "verify": "multiswap.verify",
            }
        trial = {"kind": kind, f"{kind}_index": index}
        records.append(
            {
                "schema": report.TRACE_SCHEMA,
                "record": "run",
                "run_id": run_id,
                "series_id": cell_id,
                "root_span_id": root_id,
                "benchmark": {"name": "multiswap-rsa-matched", "implementation": implementation},
                "trial": trial,
                "clock": {"kind": "monotonic", "unit": "ns"},
                "status": "ok",
                "trace_complete": True,
                "environment": {"threads": threads},
                "parameters": parameters,
                "validation": {
                    "proof_verified": True,
                    "relation_valid": True,
                    "expected_digest_supplied": False,
                },
            }
        )
        end = 12_000_000 * scale
        records.extend(
            [
                _span(run_id, root_id, operations["root"], "Verified trial", 0, end, "end-to-end", None),
                _span(run_id, f"{run_id}-w", operations["witness"], "Witness generation", 0, 1_000_000 * scale, "witness-generation", root_id, [r"\mathbf z=(\mathbf W,1,\mathbf x)"]),
                _span(run_id, f"{run_id}-c", operations["commit"], "Commit", 1_000_000 * scale, 2_000_000 * scale, "commit", root_id),
                _span(run_id, f"{run_id}-p", operations["prover"], "Prover", 2_000_000 * scale, 9_000_000 * scale, "proving", root_id),
                _span(run_id, f"{run_id}-r", operations["projection"], "Projection", 2_000_000 * scale, 3_000_000 * scale, "preparation", f"{run_id}-p"),
                _span(run_id, f"{run_id}-s", operations["piop"], "PIOP", 3_000_000 * scale, 5_000_000 * scale, "constraint-proof", f"{run_id}-p"),
                _span(run_id, f"{run_id}-o", operations["open"], "PCS opening", 5_000_000 * scale, 8_000_000 * scale, "opening-proof", f"{run_id}-p", [r"\sum_i\lambda^i C(z_i)=\sum_i\lambda^i v_i"]),
                _span(run_id, f"{run_id}-v", operations["verify"], "Verify", 9_000_000 * scale, 10_000_000 * scale, "verification", root_id),
            ]
        )
    return records


def _write_fixture(
    root: Path,
    *,
    mismatch_cell: str | None = None,
    k_values: tuple[int, ...] = (0,),
) -> Path:
    raw_dir = root / "raw"
    metadata_dir = root / "metadata"
    raw_dir.mkdir()
    metadata_dir.mkdir()
    cells = []
    for workload_k in k_values:
        specs = (
            (f"k{workload_k}-bitz-single", "bitz-ligerito", "virtual-bitz", 1),
            (f"k{workload_k}-bitz-performance", "bitz-ligerito", "virtual-bitz", 8),
            (f"k{workload_k}-limber-hyrax-single", "limber-hyrax", "hyrax", 1),
            (f"k{workload_k}-limber-hyrax-performance", "limber-hyrax", "hyrax", 8),
            (f"k{workload_k}-limber-brakedown-single", "limber-brakedown", "brakedown", 1),
            (f"k{workload_k}-limber-brakedown-performance", "limber-brakedown", "brakedown", 8),
        )
        group_digest = DIGEST if workload_k == 0 else f"{workload_k:02x}" * 32
        for cell_id, implementation, backend, threads in specs:
            digest = "ef" * 32 if cell_id == mismatch_cell else group_digest
            path = raw_dir / f"{cell_id}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(record, separators=(",", ":")) + "\n"
                    for record in _records(
                        cell_id, implementation, digest, threads, workload_k
                    )
                ),
                encoding="utf-8",
            )
            trace_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            cells.append(
                {
                    "cell_id": cell_id,
                    "label": cell_id,
                    "implementation": implementation,
                    "backend": backend,
                    "workload_k": workload_k,
                    "thread_mode": "single" if threads == 1 else "performance",
                    "rayon_threads": threads,
                    "trace": f"../raw/{path.name}",
                    "trace_sha256": trace_sha256,
                    "status": "ok",
                }
            )
    for execution_index, cell in enumerate(cells):
        cell["execution_index"] = execution_index
    manifest = {
        "schema": report.CAMPAIGN_SCHEMA,
        "trace_schema": report.TRACE_SCHEMA,
        "campaign_id": "fixture",
        "sampling": {"warmups": 1, "samples": 5},
        "workload": {
            "name": "wired MultiSwap/RSA cost-model",
            "workload_k_values": list(k_values),
        },
        "execution_order": {"cell_ids": [cell["cell_id"] for cell in cells]},
        "cells": cells,
    }
    path = metadata_dir / "campaign.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class Matched112Tests(unittest.TestCase):
    security_bits = 112

    def test_batch_sweep_has_thirty_explicit_security_cells(self) -> None:
        cells = runner.build_cells(bitz_root=Path("/bitz"), limber_root=Path("/limber"),
            run_dir=Path("/run"), campaign_id=f"matched{self.security_bits}", samples=10, warmups=1,
            all_threads=16, rustflags="-Ctarget-cpu=native", expected_digests={},
            k_values=(0,), security_bits=self.security_bits, batch_counts=(1,2,4,8,16))
        self.assertEqual(len(cells), 30)
        self.assertEqual(len({cell["trace"] for cell in cells}),30)
        self.assertEqual({runner.report.cell_group(c) for c in cells},{"b1","b2","b4","b8","b16"})
        for cell in cells:
            env=cell["environment"]
            self.assertEqual(cell["security_bits"],self.security_bits)
            if cell["implementation"]=="bitz-ligerito":
                self.assertEqual(env["BITZ_BENCH_LAMBDA"],str(self.security_bits))
                self.assertEqual(env["BITZ_MULTISWAP_BATCH_COUNT"],str(cell["batch_count"]))
            else:
                self.assertEqual(env["MATCHED_SECURITY_BITS"],str(self.security_bits))
                self.assertEqual(env["BDLAMBDA"],str(self.security_bits))
                self.assertEqual(env["MATCHED_INTEGER_SECURITY_BITS"], "128" if self.security_bits == 114 else "112")
                self.assertEqual(env["MSCFG"],"paper")
        self.assertEqual([c["execution_index"] for c in cells],list(range(30)))

    def test_batch_sweep_rejects_mixed_k_and_duplicate_batches(self) -> None:
        args=dict(bitz_root=Path("/bitz"),limber_root=Path("/limber"),run_dir=Path("/run"),campaign_id="x",samples=10,warmups=1,all_threads=16,rustflags="",expected_digests={},security_bits=self.security_bits)
        for ks,bs in [((1,),(1,2)),((0,),(1,1)),((0,),(3,)),((0,),())]:
            with self.assertRaises(report.CampaignError):
                runner.build_cells(**args,k_values=ks,batch_counts=bs)

    def test_missing_profiler_is_a_preflight_failure(self) -> None:
        with self.assertRaisesRegex(report.CampaignError,"required dependency"):
            runner.preflight_profiler(None)

    @classmethod
    def valid_run(cls, implementation="bitz-ligerito", batch=1):
        captured = json.loads((Path(__file__).parent / f"fixtures/multiswap{cls.security_bits}-preflight.json").read_text())
        metadata = next(row for row in captured if row["backend"] == implementation and row["batch_count"] == batch)
        run = _records("fixture", implementation, DIGEST, 1, 0)[0]
        cell={"implementation":implementation,"workload_k":0,"batch_count":batch,"security_bits":cls.security_bits}
        inp=run["parameters"]["input"]
        inp.update(live_rows=6209*batch,live_columns=6204*batch,padded_rows=8192*batch,padded_columns=8192*batch,batch_count=batch,public_input_count=0,public_inputs=[])
        # Reuse fixture security parameters and domain, with a synthetic statement digest.
        inp["statement_contract"] = dict(metadata["statement_contract"],
            digest_blake3=DIGEST)
        run["parameters"]["security"]=metadata["security"]
        run["artifacts"]={"proof_bytes":100,"commitment_bytes":10,"piop_and_bridge_bytes":30,"pcs_opening_bytes":60,"peak_rss_bytes":1000,"proof_size_kind":"serialized commitment/opening plus analytical PIOP and bridge estimate","memory_boundary":"process high-water RSS including setup and warmups; compiler excluded"}
        if implementation != "bitz-ligerito":
            run["artifacts"].update(opening_argument_bytes=60,dynamic_sumcheck_bytes_estimate=30,proof_size_kind="serialized commitment/opening plus analytical sumcheck estimate")
        return run,cell

    def test_all_captured_parameter_bounds_and_repetition_drift(self) -> None:
        for batch in (1,2,4,8,16):
            for implementation in ("bitz-ligerito","limber-hyrax","limber-brakedown"):
                with self.subTest(batch=batch,implementation=implementation):
                    run,cell=self.valid_run(implementation,batch)
                    report.validate_matched_parameters(run,cell)
                    if implementation != "bitz-ligerito":
                        run["parameters"]["security"]["small_primes"]-=1
                        with self.assertRaisesRegex(report.CampaignError,"repetitions"):
                            report.validate_matched_parameters(run,cell)

    def test_inherited_settings_are_overridden(self) -> None:
        from unittest.mock import patch
        with patch.dict("os.environ",{"GKRSKIP":"0","MSCFG":"full","BITZ_BENCH_LAMBDA":"100","BDLAMBDA":"80","CHAIN_BITS":"1","CARGO_ENCODED_RUSTFLAGS":"-Ctarget-cpu=generic"}):
            env=runner.benchmark_environment({"MSCFG":"paper","BITZ_BENCH_LAMBDA":str(self.security_bits)})
            self.assertEqual(env["MSCFG"],"paper")
            self.assertEqual(env["BITZ_BENCH_LAMBDA"],str(self.security_bits))
            self.assertTrue({"GKRSKIP","BDLAMBDA","CHAIN_BITS","CARGO_ENCODED_RUSTFLAGS"}.isdisjoint(env))

    def test_batch_report_and_missing_security_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root / "release-metadata.toml").write_text('root_revision = "' + '1' * 40 + '"\n')
            cells=runner.build_cells(bitz_root=root,limber_root=root,run_dir=root,campaign_id="fixture",samples=5,warmups=1,all_threads=16,rustflags="",expected_digests={},k_values=(0,),security_bits=self.security_bits,batch_counts=(1,2,4,8,16))
            manifest=runner.build_manifest(campaign_id="fixture",run_dir=root,bitz_root=root,limber_root=root,samples=5,warmups=1,all_threads=16,core_detection="fixture",cells=cells,k_values=(0,),expected_digests={})
            manifest["workload"]["batch_counts"]=[1,2,4,8,16]
            manifest["security"]={"target_bits":self.security_bits,"model":"per-check-round-minimum/v1"}
            (root/"raw").mkdir(); (root/"metadata").mkdir()
            for cell in cells:
                prototype,_=self.valid_run(cell["implementation"],cell["batch_count"])
                records=_records(cell["cell_id"],cell["implementation"],DIGEST,cell["rayon_threads"])
                for record in records:
                    if record["record"]=="run":
                        record["parameters"]=prototype["parameters"]
                        record["artifacts"]=prototype["artifacts"]
                path=(root/"metadata"/cell["trace"]).resolve()
                path.write_text("".join(json.dumps(record)+"\n" for record in records))
                cell.update(status="ok",trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            path=root/"metadata/campaign.json"
            runner._write_manifest(path,manifest)
            validated,loaded=report.validate_campaign(path)
            report.write_report(report.build_summary(validated,loaded),root/"reports")
            self.assertIn("batch=16",(root/"reports/intervals.html").read_text())
            self.assertIn("batched reference copies",(root/"reports/metrics.csv").read_text())
            duplicate = next(cell for cell in cells if cell["cell_id"] == "b2-bitz-single")
            duplicate_path = (path.parent / duplicate["trace"]).resolve()
            original_bytes = duplicate_path.read_bytes()
            records = [json.loads(line) for line in original_bytes.splitlines()]
            for record in records:
                record["run_id"] = record["run_id"].replace("b2-bitz-single", "b1-bitz-single")
            duplicate_path.write_text("".join(json.dumps(record) + "\n" for record in records))
            duplicate["trace_sha256"] = hashlib.sha256(duplicate_path.read_bytes()).hexdigest()
            runner._write_manifest(path, manifest)
            with self.assertRaisesRegex(report.CampaignError, "duplicate run_id across"):
                report.validate_campaign(path)
            duplicate_path.write_bytes(original_bytes)
            duplicate["trace_sha256"] = hashlib.sha256(original_bytes).hexdigest()
            changed = next(cell for cell in cells if cell["cell_id"] == "b1-bitz-performance")
            trace = (path.parent / changed["trace"]).resolve()
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            for record in records:
                if record["record"] == "run":
                    record["parameters"]["security"]["ligerito_config_digest"] = "00" * 32
            trace.write_text("".join(json.dumps(record) + "\n" for record in records))
            changed["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            runner._write_manifest(path,manifest)
            with self.assertRaisesRegex(report.CampaignError,"between thread counts"):
                report.validate_campaign(path)
            manifest.pop("security")
            runner._write_manifest(path,manifest)
            with self.assertRaisesRegex(report.CampaignError,"campaign security"):
                report.validate_campaign(path)

    def test_rejects_missing_security_weaker_bounds_and_statement_drift(self) -> None:
        import copy
        valid,cell=self.valid_run()
        report.validate_matched_parameters(valid,cell)
        mutations=[
            lambda r:r["parameters"].pop("security"),
            lambda r:r["parameters"]["security"].update(target_bits=100),
            lambda r:r["parameters"]["security"]["terms"][0].update(bits=111.9),
            lambda r:r["parameters"]["security"]["terms"][0].update(bits=float("nan")),
            lambda r:r["parameters"]["security"]["terms"].pop(),
            lambda r:r["parameters"]["security"].update(ligerito_target_bits=100),
            lambda r:r["parameters"]["input"].update(public_inputs=[123]),
            lambda r:r["parameters"]["input"]["statement_contract"].update(value_bits=1024),
            lambda r:r["parameters"]["input"].update(batch_count=2),
            lambda r:r["artifacts"].update(proof_bytes=90),
            lambda r:r["artifacts"].update(proof_size_kind="opening only"),
            lambda r:r["artifacts"].update(memory_boundary="per-trial heap"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                run=copy.deepcopy(valid); mutate(run)
                with self.assertRaises(report.CampaignError): report.validate_matched_parameters(run,cell)


class Matched114Tests(Matched112Tests):
    security_bits = 114

    def test_default_target_and_native_limber_parameters(self) -> None:
        self.assertEqual(runner.parse_args([]).security_bits, 114)
        self.assertEqual(runner.parse_args(["--security-bits", "112"]).security_bits, 112)
        for implementation in ("limber-hyrax", "limber-brakedown"):
            for batch in (1, 2, 4, 8, 16):
                run, cell = self.valid_run(implementation, batch)
                security = run["parameters"]["security"]
                self.assertEqual(security["integer_target_bits"], 128)
                self.assertEqual(security["integer_challenge_target_bits"], 117)
                self.assertGreaterEqual(next(t["bits"] for t in security["terms"] if t["name"] == "integer-crt"), 128)
                for field, weak_value in (("integer_target_bits", 114), ("integer_challenge_target_bits", 114), ("challenge_bits", 114)):
                    original = security[field]
                    security[field] = weak_value
                    with self.assertRaisesRegex(report.CampaignError, "Limber"):
                        report.validate_matched_parameters(run, cell)
                    security[field] = original

    def test_rejects_112_bit_trace_replay(self) -> None:
        for implementation in ("bitz-ligerito", "limber-hyrax", "limber-brakedown"):
            for batch in (1, 2, 4, 8, 16):
                run, cell = Matched112Tests.valid_run(implementation, batch)
                cell["security_bits"] = 114
                with self.assertRaisesRegex(report.CampaignError, "target differs"):
                    report.validate_matched_parameters(run, cell)

    def test_inherited_integer_target_cannot_lower_campaign_target(self) -> None:
        from unittest.mock import patch
        with patch.dict("os.environ", {"MATCHED_SECURITY_BITS": "112", "MATCHED_INTEGER_SECURITY_BITS": "112"}):
            env = runner.benchmark_environment({"MATCHED_SECURITY_BITS": "114", "MATCHED_INTEGER_SECURITY_BITS": "128"})
        self.assertEqual(env["MATCHED_SECURITY_BITS"], "114")
        self.assertEqual(env["MATCHED_INTEGER_SECURITY_BITS"], "128")


class CampaignFixtureTests(unittest.TestCase):
    def test_draft_runs_local_checks_and_marks_every_report(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_fixture(root)
            manifest = json.loads(path.read_text())
            manifest["validation"] = {"mode": "draft", "canonical_validation": "pending"}
            for cell in manifest["cells"]:
                cell.update(cwd=str(root), command=["fixture"], environment={})
            (root / "traces").mkdir()
            # The fixture supplies traces; all production acceptance and
            # aggregation checks still run. Canonical validation must not run.
            with patch.object(runner, "_run_logged"), patch.object(
                runner, "_canonical_validate", side_effect=AssertionError("canonical validation invoked")
            ), redirect_stdout(StringIO()):
                runner.execute_campaign(manifest, path, None, root / "reports", root / "canonical")
            recorded = json.loads(path.read_text())
            self.assertEqual(recorded["status"], "draft")
            self.assertEqual(recorded["validation"], {
                "mode": "draft", "canonical_validation": "pending", "comparison_checks": "passed"
            })
            summary = json.loads((root / "reports/summary.json").read_text())
            self.assertEqual(summary["validation"], recorded["validation"])
            self.assertIn("Draft — canonical validation pending", (root / "reports/intervals.html").read_text())
            self.assertIn("canonical_validation", (root / "reports/metrics.csv").read_text())
            self.assertFalse((root / "canonical").exists())

    def test_draft_still_rejects_failed_proofs_and_canonical_mode_needs_profiler(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_fixture(root)
            manifest = json.loads(path.read_text())
            with self.assertRaisesRegex(report.CampaignError, "required dependency"):
                runner.execute_campaign(manifest, path, None, root / "reports", root / "canonical")
            manifest["validation"] = {"mode": "draft"}
            for cell in manifest["cells"]:
                cell.update(cwd=str(root), command=["fixture"], environment={})
            trace = (path.parent / manifest["cells"][0]["trace"]).resolve()
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            next(record for record in records if record["record"] == "run")["validation"]["proof_verified"] = False
            trace.write_text("".join(json.dumps(record) + "\n" for record in records))
            with patch.object(runner, "_run_logged"), redirect_stdout(StringIO()):
                with self.assertRaisesRegex(report.CampaignError, "proof_verified"):
                    runner.execute_campaign(manifest, path, None, root / "reports", root / "canonical")
            self.assertEqual(json.loads(path.read_text())["status"], "failed")

    def test_plan_has_six_isolated_trace_files_per_k(self) -> None:
        root = Path("/tmp/bitz")
        cells = runner.build_cells(
            bitz_root=root,
            limber_root=Path("/tmp/limber"),
            run_dir=Path("/tmp/campaign"),
            campaign_id="fixture",
            samples=5,
            warmups=1,
            all_threads=8,
            rustflags="-Ctarget-cpu=native",
            expected_digests={},
            k_values=(0, 1, 2, 4, 8),
        )
        self.assertEqual(len(cells), 30)
        self.assertEqual({cell["rayon_threads"] for cell in cells}, {1, 8})
        self.assertEqual(len({cell["environment"].get("BITZ_MULTISWAP_TRACE_PATH") or cell["environment"].get("MATCHED_TRACE_PATH") for cell in cells}), 30)
        self.assertEqual(sum(cell["backend"] == "hyrax" for cell in cells), 10)
        self.assertEqual(sum(cell["backend"] == "brakedown" for cell in cells), 10)
        self.assertEqual({cell["workload_k"] for cell in cells}, {0, 1, 2, 4, 8})
        limber_commands = [cell["command"] for cell in cells if cell["implementation"].startswith("limber")]
        self.assertTrue(
            all(command[:4] == ["rustup", "run", "nightly-2026-07-01", "cargo"] for command in limber_commands)
        )
        limber_cells = [cell for cell in cells if cell["implementation"].startswith("limber")]
        self.assertTrue(
            all(cell["environment"]["MATCHED_K"] == str(cell["workload_k"]) for cell in limber_cells)
        )
        self.assertEqual(
            {cell["environment"]["CARGO_TARGET_DIR"] for cell in limber_cells},
            {"/tmp/campaign/build/limber-target"},
        )
        self.assertEqual([cell["execution_index"] for cell in cells], list(range(30)))
        first_by_k = [cells[index] for index in range(0, 30, 6)]
        self.assertEqual(
            [cell["implementation"] for cell in first_by_k],
            [
                "bitz-ligerito",
                "limber-hyrax",
                "limber-brakedown",
                "bitz-ligerito",
                "limber-hyrax",
            ],
        )
        self.assertEqual(
            [cell["thread_mode"] for cell in first_by_k],
            ["single", "performance", "single", "performance", "single"],
        )

    def test_valid_fixture_aggregates_type7_and_renders_math(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_fixture(root)
            manifest, cells = report.validate_campaign(manifest_path)
            self.assertEqual({cell.assignment_digest for cell in cells}, {ASSIGNMENT})
            summary = report.build_summary(manifest, cells)
            first = summary["cells"][0]
            self.assertEqual(first["headline"]["witness"]["median_ns"], "3000000")
            self.assertEqual(first["headline"]["witness"]["p10_ns"], "1400000")
            self.assertEqual(first["headline"]["witness"]["p90_ns"], "4600000")
            self.assertEqual(first["headline"]["pcs_total"]["median_ns"], "12000000")
            self.assertEqual(
                first["headline"]["application_total"]["median_ns"], "24000000"
            )
            out_dir = root / "report"
            report.write_report(summary, out_dir)
            page = (out_dir / "intervals.html").read_text(encoding="utf-8")
            self.assertIn("Matched BitZ / Limber", page)
            self.assertIn(r"\mathbf z", page)
            self.assertTrue((out_dir / "metrics.csv").is_file())

    def test_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_fixture(Path(temporary), mismatch_cell="k0-limber-hyrax-single")
            with self.assertRaisesRegex(report.CampaignError, "digest mismatch"):
                report.validate_campaign(manifest_path)

    def test_distinct_k_groups_may_have_distinct_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_fixture(Path(temporary), k_values=(0, 2))
            manifest, cells = report.validate_campaign(manifest_path)
            summary = report.build_summary(manifest, cells)
            self.assertEqual(len(cells), 12)
            self.assertEqual(set(summary["canonical_statements"]), {"0", "2"})
            self.assertNotEqual(
                summary["canonical_statements"]["0"]["digest_blake3"],
                summary["canonical_statements"]["2"]["digest_blake3"],
            )

    def test_invalid_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            trace_path = (manifest_path.parent / manifest["cells"][0]["trace"]).resolve()
            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            next(record for record in records if record["record"] == "run")["validation"]["proof_verified"] = False
            trace_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            with self.assertRaisesRegex(report.CampaignError, "proof_verified"):
                report.validate_campaign(manifest_path)

    def test_relation_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cell = next(
                cell for cell in manifest["cells"] if cell["cell_id"] == "k0-limber-hyrax-single"
            )
            trace_path = (manifest_path.parent / cell["trace"]).resolve()
            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            for record in records:
                if record["record"] == "run":
                    record["parameters"]["input"]["a_nnz"] += 1
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            cell["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(report.CampaignError, "dimensions/nnz mismatch"):
                report.validate_campaign(manifest_path)

    def test_manifest_trace_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cell = manifest["cells"][0]
            trace_path = (manifest_path.parent / cell["trace"]).resolve()
            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            trace_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(report.CampaignError, "trace SHA-256"):
                report.validate_campaign(manifest_path)

    def test_noncanonical_statement_domain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = _write_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cell = manifest["cells"][0]
            trace_path = (manifest_path.parent / cell["trace"]).resolve()
            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            first_run = next(record for record in records if record["record"] == "run")
            first_run["parameters"]["input"]["constraint_digest_domain"] = "wrong/domain"
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            cell["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(report.CampaignError, "statement domain"):
                report.validate_campaign(manifest_path)


if __name__ == "__main__":
    unittest.main()
