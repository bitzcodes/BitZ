"""Focused orchestration checks; no real benchmark campaign is run."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_multiplication_benchmarks as launcher
from test_mul_report import fixture


class Launcher(unittest.TestCase):
    def setUp(self):
        check = patch.object(launcher, "vendor_snapshots", return_value={})
        check.start()
        self.addCleanup(check.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = (Path(self.temp.name) / "results").resolve()

    def arguments(self, target="bitz", *options, experiment=()):
        return [target, "--output", str(self.output), *options, "--", *experiment]

    def test_flags_are_forwarded_without_python_expansion(self):
        for target, modes in [("bitz", ["proof", "witness", "pcs", "piop", "outer", "bounds"]),
                              ("compare", ["proof", "witness", "pcs"])]:
            for mode in modes:
                flags = [mode, "--workload", "u32-full,u64", "--split=-1..=1", "--unknown-future-flag", "a b"]
                args = launcher.parse_args(self.arguments(target, experiment=flags))
                self.assertEqual(args.experiment, flags)
                command = launcher.experiment_command(args, "/binary with spaces", False)
                marker = command.index("--")
                self.assertEqual(command[marker + 1:], ["/binary with spaces", *flags, "--out", str(self.output)])
                self.assertIn("12.0", command)

    def test_features_and_builds_use_selected_target(self):
        for target in ["bitz", "compare"]:
            args = launcher.parse_args(self.arguments(target, "--features", "bench-peak-memory,bench-internals"))
            build = launcher.build_command(args)
            self.assertIn("mul_" + target, build)
            features = build[build.index("--features") + 1].split(",")
            self.assertEqual(features.count("bench-internals"), 1)
            self.assertIn("bench-peak-memory", features)
            self.assertEqual("native-mul-compare" in features, target == "compare")
            self.assertIn("--no-run", build)
            self.assertIn("--locked", build)

    def test_orchestration_options_cannot_be_overridden_after_separator(self):
        for flags in [["--out=x"], ["--out", "x"], ["--worker=x"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(self.arguments(experiment=flags))
        for number in ["nan", "inf", "0", "-1"]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(self.arguments("bitz", "--swap-grow-gb", number))

    def test_environment_retains_build_and_lock_not_experiment_controls(self):
        env = launcher.benchmark_environment(dict(
            PATH="/bin", BITZ_BENCH_LOCK="/custom/lock", BITZ_BENCH_REPS="99",
            F2Z_BENCH_REPS="99", F2_FOREST_SCHEDULE="l8", RAYON_NUM_THREADS="17",
            BDLAMBDA="117", HARDWARE_CONCURRENCY="17", RUSTFLAGS="-C opt-level=2",
            CARGO_ENCODED_RUSTFLAGS="--cfg\x1fcustom", CARGO_TARGET_DIR="/tmp/build-cache"))
        self.assertEqual(env["BITZ_BENCH_LOCK"], "/custom/lock")
        self.assertEqual(env["RUSTFLAGS"], "-C opt-level=2")
        self.assertEqual(env["CARGO_TARGET_DIR"], "/tmp/build-cache")
        self.assertEqual(env["CARGO_ENCODED_RUSTFLAGS"], "--cfg\x1fcustom")
        for key in ["BITZ_BENCH_REPS", "F2Z_BENCH_REPS", "F2_FOREST_SCHEDULE", "RAYON_NUM_THREADS", "BDLAMBDA", "HARDWARE_CONCURRENCY"]:
            self.assertNotIn(key, env)

    def test_launcher_dry_run_does_not_build_or_create_output(self):
        with patch.object(launcher, "run_campaign") as run, contextlib.redirect_stdout(io.StringIO()) as out:
            code = launcher.main(self.arguments("compare", "--dry-run", experiment=["proof", "--backends", "all"]))
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertIn("mul_compare", out.getvalue())
        self.assertIn("mul_report.py", out.getvalue())

    def test_cargo_structured_output_selects_executable(self):
        args = launcher.parse_args(self.arguments())
        def build(command, env, stream, **kwargs):
            stream.write('a diagnostic mentioning a different path\n')
            stream.write(json.dumps(dict(reason="compiler-artifact", target=dict(name="mul_bitz", kind=["bin"]), executable="/wrong/bin")) + "\n")
            stream.write(json.dumps(dict(reason="compiler-artifact", target=dict(name="mul_bitz", kind=["bench"], src_path="/dependency/benches/mul_bitz.rs"), executable="/wrong/dependency")) + "\n")
            stream.write(json.dumps(dict(reason="compiler-artifact", target=dict(name="mul_bitz", kind=["bench"], src_path=str(launcher.ROOT / "benches/mul_bitz.rs")), executable="/real/bench")) + "\n")
            return 0
        with patch.object(launcher, "run_campaign", side_effect=build) as run:
            executable, code = launcher.build_executable(args, {}, Path(self.temp.name) / "build.log")
        self.assertEqual((executable, code), ("/real/bench", 0))
        self.assertEqual(run.call_count, 1)

    def test_forwarded_inspection_never_gates_or_reports(self):
        for flag in ["--help", "-h", "--dry-run"]:
            with patch.object(launcher, "build_executable", return_value=("/bench", 0)) as build, \
                 patch.object(launcher, "run_campaign", return_value=0) as run, \
                 patch.object(launcher, "report") as report:
                self.assertEqual(launcher.main(self.arguments(experiment=[flag])), 0)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(run.call_args.args[0], ["/bench", flag])
            self.assertFalse(run.call_args.kwargs["gated"])
            report.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_success_uses_shared_report_and_current_artifacts(self):
        def complete(command, env, stream, **kwargs):
            fixture(self.output)
            return 0
        with patch.object(launcher, "build_executable", return_value=("/bench", 0)) as build, \
             patch.object(launcher, "run_campaign", side_effect=complete):
            self.assertEqual(launcher.main(self.arguments()), 0)
        self.assertEqual(build.call_count, 1)
        self.assertTrue((self.output / "reports/summary.csv").is_file())
        self.assertTrue((self.output / "samples.jsonl").is_file())
        self.assertFalse((self.output / "campaign.json").exists())
        self.assertFalse((self.output / "metrics.csv").exists())
        self.assertFalse((self.output / "suite.json").exists())

    def test_failed_worker_retains_logs_and_exit_code_without_report(self):
        with patch.object(launcher, "build_executable", return_value=("/bench", 0)), \
             patch.object(launcher, "run_campaign", return_value=86), \
             patch.object(launcher, "report") as report:
            self.assertEqual(launcher.main(self.arguments()), 86)
        self.assertTrue((self.output / "run.log").exists())
        report.assert_not_called()

    def test_build_failure_and_existing_output_never_start_workers(self):
        with patch.object(launcher, "build_executable", return_value=(None, 101)), \
             patch.object(launcher, "run_campaign") as run:
            self.assertEqual(launcher.main(self.arguments()), 101)
        run.assert_not_called()
        with patch.object(launcher, "build_executable") as build:
            self.assertEqual(launcher.main(self.arguments()), 1)
        build.assert_not_called()

    def test_success_exit_without_complete_results_cannot_report(self):
        with patch.object(launcher, "build_executable", return_value=("/bench", 0)), \
             patch.object(launcher, "run_campaign", return_value=0):
            self.assertEqual(launcher.main(self.arguments()), 1)
        self.assertFalse((self.output / "reports").exists())

    def test_cancellation_waits_for_gate_cleanup(self):
        marker = Path(self.temp.name) / "ready"
        stopped = Path(self.temp.name) / "stopped"
        fake_gate = Path(self.temp.name) / "gate.py"
        fake_gate.write_text('''import signal, sys, time\nfrom pathlib import Path\ndef stop(*_):\n    Path(sys.argv[2]).write_text("cleaned")\n    raise SystemExit(143)\nsignal.signal(signal.SIGTERM, stop)\nPath(sys.argv[1]).write_text("ready")\nwhile True: time.sleep(0.1)\n''')
        harness = 'import sys; import run_multiplication_benchmarks as r; r.run_campaign(sys.argv[1:], None, None, gated=True)'
        env = dict(os.environ, PYTHONPATH=str(launcher.ROOT / "scripts"))
        with subprocess.Popen([sys.executable, "-c", harness, sys.executable, str(fake_gate), str(marker), str(stopped)],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists())
                process.send_signal(signal.SIGINT)
                self.assertNotEqual(process.wait(timeout=10), 0)
                self.assertEqual(stopped.read_text(), "cleaned")
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=20)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX process groups")
    def test_exited_leader_does_not_leave_a_worker_holding_output_open(self):
        leader = Path(self.temp.name) / "leader.py"
        group_file = Path(self.temp.name) / "group"
        ready = Path(self.temp.name) / "worker-ready"
        leader.write_text("""import os, signal, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpgrp()))
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(sys.argv[2]).touch()
    time.sleep(60)
else:
    while not Path(sys.argv[2]).exists(): time.sleep(0.01)
""")
        env = dict(os.environ, PYTHONPATH=str(launcher.ROOT / "scripts"),
                   BITZ_BENCH_LOCK=str(Path(self.temp.name) / "lock"))
        harnesses = [
            ('import sys; import run_multiplication_benchmarks as r; '
             'sys.exit(r.run_campaign(sys.argv[1:], None, None, gated=False))', []),
            ('import bench_gate as g; g.swap_used_gb = lambda: 0; g.wait_idle = lambda *a: None; '
             'raise SystemExit(g.main())', ["run", "--label", "test", "--"]),
        ]
        for harness, flags in harnesses:
            with self.subTest(gated=bool(flags)):
                ready.unlink(missing_ok=True)
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", harness, *flags, sys.executable,
                         str(leader), str(group_file), str(ready)],
                        env=env, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                finally:
                    if group_file.exists():
                        try:
                            os.killpg(int(group_file.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        self.assertFalse(Path(env["BITZ_BENCH_LOCK"]).exists())

    def test_gate_preserves_child_signal_exit(self):
        env = dict(os.environ, PYTHONPATH=str(launcher.ROOT / "scripts"),
                   BITZ_BENCH_LOCK=str(Path(self.temp.name) / "lock"))
        result = subprocess.run(
            [sys.executable, "-c", "import bench_gate as g; g.swap_used_gb = lambda: 0; "
             "g.wait_idle = lambda *a: None; raise SystemExit(g.main())",
             "run", "--label", "test", "--", sys.executable, "-c",
             "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
            env=env, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 143)
        self.assertFalse(Path(env["BITZ_BENCH_LOCK"]).exists())

    def test_gate_requires_sustained_idle_before_starting(self):
        import bench_gate
        samples = iter([95.0, 50.0, 95.0, 95.0])
        with patch.object(bench_gate, "idle_percent", side_effect=lambda: next(samples)), \
             patch.object(bench_gate.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            # Two consecutive qualifying samples; the busy sample resets the streak.
            bench_gate.wait_idle("test", 88.0, 40.0, 20.0)
        self.assertIsNone(next(samples, None))
        with patch.object(bench_gate, "idle_percent", return_value=10.0), \
             patch.object(bench_gate.time, "sleep"), contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(TimeoutError):
            bench_gate.wait_idle("test", 88.0, 40.0, 20.0, max_wait_seconds=0)

    def test_gate_cleans_workers_after_leader_exits(self):
        import bench_gate
        from unittest.mock import Mock
        process = Mock(pid=123456)
        process.wait.return_value = 0
        with patch.object(bench_gate.os, "killpg") as kill:
            bench_gate.stop_process_group(process)
        self.assertEqual(kill.call_args_list[0].args, (123456, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[-1].args, (123456, signal.SIGKILL))

    def test_ungated_exit_cleans_remaining_workers(self):
        from unittest.mock import Mock
        process = Mock(pid=123456)
        process.wait.return_value = 0
        process.poll.return_value = 0
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        with patch.object(launcher.subprocess, "Popen", return_value=process), \
             patch.object(launcher.os, "killpg") as kill:
            self.assertEqual(launcher.run_campaign(["bench"], {}, None, gated=False), 0)
        kill.assert_called_once_with(123456, signal.SIGKILL)

    def test_ungated_child_signal_exit_is_preserved(self):
        code = launcher.run_campaign([sys.executable, "-c", "import os,signal; os.kill(os.getpid(),signal.SIGTERM)"],
                                     None, None, gated=False)
        self.assertEqual(code, 143)


if __name__ == "__main__":
    unittest.main()
