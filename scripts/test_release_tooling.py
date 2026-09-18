"""End-to-end release checks using small local upstream repositories."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import local_provenance as provenance
import materialize_vendors as materializer
import package_release as packager
import build_metadata


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


class ReleaseTests(unittest.TestCase):
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

    def test_exclusions_are_root_anchored_and_preserve_fixtures(self):
        for path in ["docs/design.md", "output/result", ".claude/settings.json",
                     "src/nested/__pycache__/x.pyc", "vendor/alpha/target/foo",
                     "vendor/alpha/.git", "x/.DS_Store", "x/._resource", "x/y.pyo"]:
            self.assertTrue(packager.excluded(path), path)
        for path in ["vendor/alpha/docs/input", "vendor/alpha/output/fixture",
                     "fixtures/test.out", "fixtures/test.bin", "patches/a.patch",
                     ".cargo/config.toml", "vendor/alpha/README.md"]:
            self.assertFalse(packager.excluded(path), path)

    def test_zip_is_deterministic_and_preserves_modes_and_patches(self):
        self.ready()
        (self.root / "docs").mkdir()
        (self.root / "docs/local.md").write_text("not shipped")
        (self.root / "tests").mkdir()
        (self.root / "tests/expected.out").write_text("fixture")
        commit(self.root)
        # Excluded dirtiness must not prevent packaging or leak into the ZIP.
        (self.root / "docs/local.md").write_text("private local changes")
        first, second = self.work / "one.zip", self.work / "two.zip"
        a = packager.package(self.root, first)
        b = packager.package(self.root, second)
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            names = archive.namelist()
            self.assertFalse(any("/docs/" in name or "/.git/" in name for name in names))
            self.assertIn("bitz/patches/alpha.patch", names)
            self.assertIn("bitz/tests/expected.out", names)
            self.assertTrue(archive.getinfo("bitz/vendor/alpha/run.sh").external_attr >> 16 & 0o111)
            self.assertEqual(archive.getinfo("bitz/vendor/alpha/readme-link").external_attr >> 16 & 0o170000, 0o120000)
            self.assertIn("bitz/release-metadata.toml", names)
        # Bundles were never created and are not needed for packaging.
        self.assertFalse((self.root / "history").exists())

    def test_dry_run_and_size_limit_do_not_write_output(self):
        self.ready()
        output = self.work / "release.zip"
        result = packager.package(self.root, output, dry_run=True)
        self.assertIn("README.md", result["included"])
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "exceeding"):
            packager.package(self.root, output, max_bytes=10)
        self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        self.ready()
        output = self.work / "release.zip"
        output.write_bytes(b"user artifact")
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            packager.package(self.root, output)
        self.assertEqual(output.read_bytes(), b"user artifact")

    def test_output_appearing_during_publication_is_preserved(self):
        self.ready()
        output = self.work / "release.zip"
        real_link = os.link

        def competing_writer(source, destination):
            output.write_bytes(b"concurrent user artifact")
            return real_link(source, destination)

        with mock.patch.object(packager.os, "link", side_effect=competing_writer):
            with self.assertRaises(FileExistsError):
                packager.package(self.root, output)
        self.assertEqual(output.read_bytes(), b"concurrent user artifact")
        self.assertEqual(list(self.work.glob(".bitz-zip-*")), [])

    def test_root_changes_and_missing_tracked_files_are_rejected(self):
        self.ready()
        readme = self.root / "README.md"
        readme.write_text("uncommitted")
        with self.assertRaisesRegex(ValueError, "not committed and clean"):
            packager.package(self.root, self.work / "x.zip")
        readme.unlink()
        with self.assertRaisesRegex(ValueError, "tracked file is missing"):
            packager.package(self.root, self.work / "x.zip")

    def test_escaping_symlink_is_rejected(self):
        self.ready()
        (self.root / "escape").symlink_to("../private")
        commit(self.root)
        with self.assertRaisesRegex(ValueError, "symlink escapes"):
            packager.package(self.root, self.work / "x.zip")

    def test_personal_fork_dependency_is_rejected(self):
        self.ready()
        (self.root / "Cargo.toml").write_text('[dependencies]\nx = {git="https://github.com/fork-owner/private"}\n')
        commit(self.root)
        with mock.patch.dict(os.environ, {"RELEASE_FORBIDDEN_FORKS": "fork-owner|other-owner"}), \
             self.assertRaisesRegex(ValueError, "personal fork dependency"):
            packager.package(self.root, self.work / "x.zip")

    def test_archive_checks_and_build_metadata_need_no_git(self):
        self.ready()
        output = self.work / "release.zip"
        packager.package(self.root, output)
        extracted = self.work / "extracted"
        with zipfile.ZipFile(output) as archive:
            for info in archive.infolist():
                path = extracted / info.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                mode = info.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    path.symlink_to(os.fsdecode(archive.read(info)))
                else:
                    path.write_bytes(archive.read(info))
                    path.chmod(mode & 0o777)
        archive_root = extracted / "bitz"
        with mock.patch("subprocess.check_output", side_effect=AssertionError("unexpected Git invocation")):
            materializer.materialize(archive_root, check=True)
            values, _ = build_metadata.metadata(archive_root)
            self.assertEqual(values["BITZ_DIRTY"], "unknown")
            self.assertEqual(provenance.root_metadata(archive_root)["source_kind"], "archive")
            self.assertEqual(provenance.source_patch(archive_root), b"")
        self.assertEqual(values["BITZ_REVISION"], git(self.root, "rev-parse", "HEAD").decode().strip())
        # Compile and run the actual Rust build-script bridge against this archive.
        if shutil.which("rustc"):
            script_root = Path(__file__).resolve().parent
            (archive_root / "scripts").mkdir(exist_ok=True)
            for name in ["local_provenance.py", "build_metadata.py"]:
                shutil.copy2(script_root / name, archive_root / "scripts" / name)
            executable = self.work / "build-script"
            subprocess.run(["rustc", "--edition=2024", str(script_root.parent / "build.rs"),
                            "-o", str(executable)], check=True, capture_output=True)
            output = subprocess.check_output([str(executable)], env=dict(os.environ,
                                               CARGO_MANIFEST_DIR=str(archive_root)), text=True)
            self.assertIn("cargo:rustc-env=BITZ_REVISION=" + values["BITZ_REVISION"], output)
            self.assertIn("cargo:rustc-env=BITZ_DIRTY=unknown", output)

    def make_bundles(self):
        directory = self.work / "bundles"
        directory.mkdir()
        for name, record in self.records.items():
            upstream = self.work / (name + "-upstream")
            git(upstream, "update-ref", "refs/heads/upstream", record["upstream_commit"])
            git(upstream, "update-ref", "refs/heads/snapshot", record["snapshot_commit"])
            bundle = directory / (name + ".bundle")
            git(upstream, "bundle", "create", str(bundle), "refs/heads/upstream", "refs/heads/snapshot")
            record["bundle_sha256"] = provenance.sha256(bundle)
        self.write_provenance()
        return directory

    def test_history_bundles_and_patch_trees_match(self):
        directory = self.make_bundles()
        self.assertEqual(provenance.verify_histories(self.root, directory), ["alpha", "beta"])

    def test_corrupted_bundle_is_rejected(self):
        directory = self.make_bundles()
        (directory / "alpha.bundle").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "bundle checksum"):
            provenance.verify_histories(self.root, directory)

    def test_history_bundle_rejects_extra_refs(self):
        directory = self.make_bundles()
        upstream = self.work / "alpha-upstream"
        bundle = directory / "alpha.bundle"
        git(upstream, "bundle", "create", str(bundle), "--all")
        self.records["alpha"]["bundle_sha256"] = provenance.sha256(bundle)
        self.write_provenance()
        with self.assertRaisesRegex(ValueError, "only upstream and snapshot refs"):
            provenance.verify_histories(self.root, directory)

    def test_untracked_vendor_source_is_not_silently_omitted(self):
        self.ready()
        git(self.root, "rm", "--cached", "vendor/alpha/README.md")
        git(self.root, "commit", "--quiet", "-m", "missing materialization in root inventory")
        with self.assertRaisesRegex(ValueError, "vendor file missing from committed release inventory"):
            packager.package(self.root, self.work / "x.zip")


if __name__ == "__main__":
    unittest.main()
