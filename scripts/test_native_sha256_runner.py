"""Check backend selection before the SHA runner launches Cargo."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().with_name("run_native_sha256_compare.sh")


class RunnerTests(unittest.TestCase):
    def invoke(self, overrides, expected_code=0):
        with tempfile.TemporaryDirectory(prefix="sha-runner-test-") as directory:
            root = Path(directory)
            cargo = root / "cargo"
            cargo.write_text(
                '#!/bin/sh\n'
                'printf "%s\\n" "$BITZ_SHA_COMPARE_BACKENDS" '
                '"$BITZ_SHA_COMPARE_LIMBER_ENGINE" "${BITZ_SHA_COMPARE_LIMBER_K-unset}" '
                '"$BITZ_SHA_COMPARE_EXPONENTS" > "$SHA_TEST_CAPTURE"\n'
            )
            cargo.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith("BITZ_SHA_COMPARE_")}
            env.update(
                PATH=str(root) + os.pathsep + env["PATH"],
                SHA_TEST_CAPTURE=str(root / "capture"),
                ZK_TRACE_SCRIPT=str(root / "absent-profiler"),
                BITZ_SHA_COMPARE_RUN_DIR=str(root / "run"),
            )
            env.update(overrides)
            result = subprocess.run(
                ["bash", str(RUNNER)], env=env, capture_output=True, text=True, timeout=15
            )
            self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
            if expected_code:
                self.assertFalse((root / "capture").exists(), "invalid settings launched Cargo")
                self.assertFalse((root / "run").exists(), "invalid settings created run artifacts")
                return result.stderr
            return (root / "capture").read_text().splitlines()

    def test_defaults_use_exactly_four_backends_and_brakedown(self):
        self.assertEqual(self.invoke({}), [
            "bitz plonky3-whir binius64 limber", "brakedown", "unset",
            "7 8 10 11 12 13 14 15 16",
        ])

    def test_explicit_subset_and_k_are_preserved(self):
        settings = dict(BITZ_SHA_COMPARE_BACKENDS="binius64,limber",
                        BITZ_SHA_COMPARE_LIMBER_ENGINE="brakedown",
                        BITZ_SHA_COMPARE_LIMBER_K="11", BITZ_SHA_COMPARE_EXPONENTS="7")
        self.assertEqual(self.invoke(settings), ["binius64,limber", "brakedown", "11", "7"])

    def test_hyrax_and_unknown_selections_fail_before_build(self):
        for settings in [
            dict(BITZ_SHA_COMPARE_LIMBER_ENGINE="hyrax"),
            dict(BITZ_SHA_COMPARE_LIMBER_ENGINE="unknown"),
            dict(BITZ_SHA_COMPARE_BACKENDS="bitz spartan-hyrax limber"),
            dict(BITZ_SHA_COMPARE_BACKENDS="unknown"),
            dict(BITZ_SHA_COMPARE_BACKENDS=" "),
        ]:
            with self.subTest(settings=settings):
                self.invoke(settings, expected_code=2)


if __name__ == "__main__":
    unittest.main()
