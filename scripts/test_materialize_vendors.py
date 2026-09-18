"""Materialization and idempotence checks using tiny local upstream repositories."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import local_provenance as provenance
import materialize_vendors as materializer

def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)

def init(root):
    root.mkdir(parents=True)
    git(root, "init", "--quiet")
    git(root, "config", "user.name", "Release Test")
    git(root, "config", "user.email", "release@example.invalid")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "core.filemode", "true")

def commit(root):
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "-m", "fixture")
    return git(root, "rev-parse", "HEAD").decode().strip()


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.root = self.work / "root"
        init(self.root)
        (self.root / "README.md").write_text("Fixture release\n")
        (self.root / "Cargo.toml").write_text('[package]\nname = "fixture"\n')
        (self.root / "patches").mkdir()
        self.records = {}
        for name in ("alpha", "beta"):
            upstream = self.work / (name + "-upstream")
            init(upstream)
            (upstream / "src").mkdir()
            (upstream / "src/lib.rs").write_text("pub fn value() -> u32 { 1 }\n")
            (upstream / "README.md").write_text("Retained upstream README\n")
            (upstream / "run.sh").write_text("#!/bin/sh\nexit 0\n")
            (upstream / "run.sh").chmod(0o755)
            (upstream / "readme-link").symlink_to("README.md")
            base = commit(upstream)
            (upstream / "src/lib.rs").write_text("pub fn value() -> u32 { 2 }\n")
            (upstream / "fixture.bin").write_bytes(bytes(range(256)))
            snapshot = commit(upstream)
            patch = self.root / "patches" / (name + ".patch")
            patch.write_bytes(git(upstream, "diff", "--binary", base, snapshot))
            self.records[name] = dict(git=upstream.as_uri(), path="vendor/" + name,
                                      original_upstream_commit=base, upstream_commit=base,
                                      snapshot_commit=snapshot,
                                      snapshot_tree=git(upstream, "rev-parse", "HEAD^{tree}").decode().strip(),
                                      patch="patches/" + name + ".patch", patch_sha256=provenance.sha256(patch),
                                      bundle="history/" + name + ".bundle", bundle_sha256="0" * 64)
        self.write_provenance()

    def write_provenance(self):
        text = "\n".join("[" + name + "]\n" + "\n".join(
            key + " = " + json.dumps(value) for key, value in record.items())
                         for name, record in self.records.items())
        (self.root / "provenance.toml").write_text(text + "\n")

    def ready(self):
        materializer.materialize(self.root)
        commit(self.root)

    def test_materialize_then_check_without_git_or_network(self):
        materializer.materialize(self.root)
        for name, record in self.records.items():
            path = self.root / record["path"]
            self.assertFalse((path / ".git").exists())
            self.assertEqual(provenance.source_tree(path), record["snapshot_tree"])
            self.assertTrue((path / "run.sh").stat().st_mode & 0o111)
            self.assertTrue((path / "readme-link").is_symlink())
        with mock.patch("subprocess.check_output", side_effect=AssertionError("unexpected Git invocation")):
            materializer.materialize(self.root)
            materializer.materialize(self.root, check=True)

    def test_check_missing_is_read_only(self):
        with self.assertRaisesRegex(ValueError, "missing vendors"):
            materializer.materialize(self.root, check=True)
        self.assertFalse((self.root / "vendor").exists())

    def test_refuse_modified_existing_vendor_before_any_fetch(self):
        self.ready()
        path = self.root / "vendor/alpha/src/lib.rs"
        path.write_text("local work")
        shutil.rmtree(self.root / "vendor/beta")
        with mock.patch.object(materializer, "prepare", side_effect=AssertionError("unexpected fetch")):
            with self.assertRaisesRegex(ValueError, "alpha: materialized source differs"):
                materializer.materialize(self.root)
        self.assertEqual(path.read_text(), "local work")

    def test_patch_checksum_failure(self):
        (self.root / "patches/alpha.patch").write_text("damaged")
        with self.assertRaisesRegex(ValueError, "patch checksum"):
            materializer.materialize(self.root)
        self.assertFalse((self.root / "vendor").exists())

    def test_invalid_patch_does_not_install_other_vendor(self):
        patch = self.root / "patches/beta.patch"
        patch.write_text("not a patch")
        self.records["beta"]["patch_sha256"] = provenance.sha256(patch)
        self.write_provenance()
        with self.assertRaises(subprocess.CalledProcessError):
            materializer.materialize(self.root)
        self.assertFalse((self.root / "vendor/alpha").exists())
        self.assertEqual(list((self.root / "vendor").iterdir()), [])

    def test_missing_upstream_commit_leaves_no_vendor(self):
        self.records["beta"]["original_upstream_commit"] = "0" * 40
        self.write_provenance()
        with self.assertRaises(subprocess.CalledProcessError):
            materializer.materialize(self.root)
        self.assertEqual(list((self.root / "vendor").iterdir()), [])

    def test_snapshot_mismatch_leaves_no_vendor(self):
        self.records["beta"]["snapshot_tree"] = "0" * 40
        self.write_provenance()
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            materializer.materialize(self.root)
        self.assertEqual(list((self.root / "vendor").iterdir()), [])

    def test_interrupted_preparation_leaves_existing_vendor_untouched(self):
        materializer.materialize(self.root)
        shutil.rmtree(self.root / "vendor/beta")
        with mock.patch.object(materializer, "prepare", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                materializer.materialize(self.root)
        provenance.verify_vendor(self.root, "alpha", self.records["alpha"])
        self.assertFalse((self.root / "vendor/beta").exists())
        self.assertFalse((self.root / "vendor/.materialize-lock").exists())


if __name__ == "__main__":
    unittest.main()
