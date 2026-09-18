"""Campaign compatibility and failure checks without running performance sweeps."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_multiplication_benchmarks as launcher
from test_mul_report import fixture


class Matrix(unittest.TestCase):
    def setUp(self):
        check = patch.object(launcher, "vendor_snapshots", return_value={})
        check.start()
        self.addCleanup(check.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results"
        self.arguments = ["--workloads", "u64", "--backends", "bitz", "--threads", "1",
                          "--exponents", "15", "--reps", "2", "--no-gate",
                          "--output", str(self.output)]
        sample = self.root / "fixture"
        sample.mkdir()
        self.manifest, self.records = fixture(sample)

    def test_default_matrix_keeps_all_180_configurations_and_size_caps(self):
        campaigns, skipped = launcher.matrix_plan(launcher.parse_matrix_args([]))
        self.assertEqual(len(campaigns), 44)
        self.assertEqual(sum(len(c["exponents"]) for c in campaigns), 180)
        self.assertEqual(len({c["id"] for c in campaigns}), 44)
        self.assertEqual(len(skipped), 2)
        for c in campaigns:
            self.assertEqual(max(c["exponents"]), launcher.MAX_EXPONENT[c["workload"]][c["backend"]])
            self.assertEqual(c["exponents"][0], 15)
            self.assertEqual(c["exponents"][1] - c["exponents"][0], 2)
        wide = next(c for c in campaigns if c["workload"] == "u128" and c["backend"] == "binius64")
        self.assertEqual(wide["swap_guard_gb"], 34)

    def test_old_shell_flags_become_current_rust_flags(self):
        args = launcher.parse_matrix_args([
            "--no-gate", "--workloads", "u32", "u64", "u128", "--backends", *launcher.BACKENDS,
            "--threads", "1", "10", "--reps", "5", "--bitz-profiles", "custom:1:4", "custom:3:4",
            "--binius-rates", "1", "3", "--exponents", "15", "--output", str(self.output)])
        campaigns, _ = launcher.matrix_plan(args)
        for c in campaigns:
            command = launcher.experiment_command(launcher.campaign_args(args, c), "/bench", False)
            self.assertEqual(command[:2], ["/bench", "proof"])
            self.assertIn("--log-n", command)
            self.assertNotIn("--exponents", command)
            self.assertNotIn("--workloads", command)
            self.assertNotIn("--bitz-profiles", command)
            self.assertEqual(command[-2:], ["--out", str(self.output / c["id"])])
            if c["backend"] == "bitz":
                self.assertEqual(command[command.index("--ligerito") + 1], c["bitz_profile"])
                self.assertEqual(command[command.index("--w") + 1], "1")
                self.assertEqual(command[command.index("--bitz-profile") + 1], "100")
            elif c["backend"].startswith("binius64"):
                self.assertEqual(command[command.index("--log-inv-rate") + 1], str(c["binius_log_inv_rate"]))
                if c["backend"].endswith("ligerito"):
                    self.assertEqual(command[command.index("--binius-ligerito-accounting") + 1], "rbr")
            elif c["backend"] == "limber":
                self.assertEqual(command[command.index("--limber-bits") + 1], "100")
        self.assertEqual({c["workload"] for c in campaigns}, {"u32-mod32", "u64", "u128"})

    def test_invalid_selection_is_rejected_before_building(self):
        for flags in [["--workloads", "u32", "u32-mod32"], ["--threads", "1", "1"],
                      ["--threads", "65537"], ["--reps", "0"], ["--warmups", "-1"],
                      ["--swap-grow-gb", "nan"], ["--backends", "bitz", "--exponents", "14"],
                      ["--backends", "limber", "--exponents", "25"],
                      ["--backends", "plonky3-fri", "--workloads", "u64"]]:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_matrix_args(flags)

    def test_dry_run_has_no_build_network_or_output_side_effects(self):
        with patch.object(launcher, "build_executable") as build, patch.object(launcher, "run_campaign") as run, \
             patch.object(launcher.shutil, "which") as processor, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launcher.main([*self.arguments, "--dry-run"]), 0)
        build.assert_not_called()
        run.assert_not_called()
        processor.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertIn("mul_compare", output.getvalue())
        self.assertIn("--ligerito custom:3:4", output.getvalue())
        self.assertIn("mul_report.py", output.getvalue())

    def complete(self, command, env, stream, **kwargs):
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["job"]["case"]["bitz"]["ligerito"] = command[command.index("--ligerito") + 1]
        if "--dry-run" in command:
            self.assertFalse(any(self.output.glob("*/manifest.json")))
            stream.write(json.dumps([c["job"] for c in manifest["cases"]]))
        else:
            directory = Path(command[command.index("--out") + 1])
            directory.mkdir()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))
        return 0

    def run_suite(self, worker=None, build_result=("/bench", 0)):
        with patch.object(launcher, "build_executable", return_value=build_result) as build, \
             patch.object(launcher, "run_campaign", side_effect=worker or self.complete) as run, \
             patch.object(launcher.shutil, "which", return_value="/processor"), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = launcher.main(self.arguments)
        return code, build, run

    def test_build_once_preflight_all_then_prove_and_report_current_artifacts(self):
        code, build, run = self.run_suite()
        self.assertEqual(code, 0)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(["--dry-run" in c.args[0] for c in run.call_args_list], [True, True, False, False])
        suite = json.loads((self.output / "suite.json").read_text())
        self.assertEqual(suite["status"], "complete")
        self.assertEqual([c["status"] for c in suite["campaigns"]], ["complete", "complete"])
        rows = json.loads((self.output / "reports/summary.json").read_text())
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["case"]["bitz"]["ligerito"] for r in rows}, {"custom:1:4", "custom:3:4"})
        self.assertTrue((self.output / "reports/summary.csv").is_file())
        self.assertFalse((self.output / "metrics.csv").exists())

    def test_failed_preflight_stops_all_proofs(self):
        def fail_second(command, env, stream, **kwargs):
            if "custom:3:4" in command:
                return 2
            return self.complete(command, env, stream, **kwargs)
        code, _, run = self.run_suite(fail_second)
        self.assertEqual(code, 1)
        self.assertTrue(all("--dry-run" in call.args[0] for call in run.call_args_list))
        suite = json.loads((self.output / "suite.json").read_text())
        self.assertEqual((suite["status"], suite["phase"]), ("failed", "preflight"))

    def test_failed_proof_preserves_partial_reports_and_exit_status(self):
        def fail_second(command, env, stream, **kwargs):
            if "--dry-run" not in command and "custom:3:4" in command:
                return 86
            return self.complete(command, env, stream, **kwargs)
        code, _, _ = self.run_suite(fail_second)
        self.assertEqual(code, 86)
        suite = json.loads((self.output / "suite.json").read_text())
        self.assertEqual(suite["status"], "failed")
        self.assertEqual([c["status"] for c in suite["campaigns"]], ["complete", "failed"])
        self.assertEqual(len(json.loads((self.output / "reports/summary.json").read_text())), 1)

    def test_different_results_cannot_be_reported_as_the_requested_matrix(self):
        def changed(command, env, stream, **kwargs):
            code = self.complete(command, env, stream, **kwargs)
            if "--dry-run" not in command:
                path = Path(command[command.index("--out") + 1]) / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["cases"][0]["job"]["case"]["seed"] += 1
                path.write_text(json.dumps(manifest))
            return code
        code, _, _ = self.run_suite(changed)
        self.assertEqual(code, 1)
        self.assertFalse((self.output / "reports").exists())
        self.assertIn("differ from the preflight plan", json.loads((self.output / "suite.json").read_text())["error"])

    def test_interruption_records_state_and_leaves_pending_campaigns_unrun(self):
        def interrupt(command, env, stream, **kwargs):
            if "--dry-run" not in command:
                raise KeyboardInterrupt()
            return self.complete(command, env, stream, **kwargs)
        code, _, _ = self.run_suite(interrupt)
        self.assertEqual(code, 130)
        suite = json.loads((self.output / "suite.json").read_text())
        self.assertEqual(suite["status"], "interrupted")
        self.assertEqual([c["status"] for c in suite["campaigns"]], ["interrupted", "pending"])

    def test_build_failure_and_existing_output_do_not_start_campaigns(self):
        code, _, run = self.run_suite(build_result=(None, 101))
        self.assertEqual(code, 101)
        run.assert_not_called()
        original = (self.output / "suite.json").read_bytes()
        code, build, run = self.run_suite()
        self.assertEqual(code, 1)
        build.assert_not_called()
        run.assert_not_called()
        self.assertEqual((self.output / "suite.json").read_bytes(), original)

    def test_vendor_verification_failure_prevents_build_and_output(self):
        with patch.object(launcher, "vendor_snapshots", side_effect=ValueError("vendor mismatch")):
            code, build, run = self.run_suite()
        self.assertEqual(code, 1)
        build.assert_not_called()
        run.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
