#!/usr/bin/env python3
"""Reconstruct pinned official upstream source plus the reviewed vendor patches."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from local_provenance import ROOT, git, inside, source_tree, vendors, verify_patch, verify_vendor


def prepare(root: Path, name: str, record: dict, staging: Path) -> Path:
    patch = verify_patch(root, name, record).resolve()
    repository = staging / name
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    git(repository, "config", "core.autocrlf", "false")
    git(repository, "config", "core.filemode", "true")
    # URLs and revisions are arguments, never shell code. No personal fork or
    # branch fallback: the official source and full revision are in provenance.
    git(repository, "fetch", "--quiet", "--no-tags", "--depth=1", "--",
        record["git"], record["original_upstream_commit"])
    git(repository, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    git(repository, "-c", "core.autocrlf=false", "apply", "--index", "--whitespace=nowarn", str(patch))
    actual = git(repository, "write-tree").decode().strip()
    if actual != record["snapshot_tree"] or source_tree(repository) != actual:
        raise ValueError(f"{name}: patched source does not reproduce the recorded snapshot")
    shutil.rmtree(repository / ".git")
    return repository


def materialize(root: Path, *, check: bool = False) -> list[str]:
    records = vendors(root)
    missing = []
    for name, record in records.items():
        verify_patch(root, name, record)
        destination = inside(root, record["path"])
        if destination.exists() or destination.is_symlink():
            verify_vendor(root, name, record)
        else:
            missing.append(name)
    if check and missing:
        raise ValueError("missing vendors: " + ", ".join(missing) + "; run materialize_vendors.py")
    if not missing:
        return list(records)
    vendor_root = root / "vendor"
    vendor_root.mkdir(exist_ok=True)
    # A directory lock prevents concurrent materializers from installing into
    # each other's destinations. A leftover lock is never removed automatically.
    lock = vendor_root / ".materialize-lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f"materialization lock exists: {lock}; check for another running process") from None
    installed = []
    try:
        with tempfile.TemporaryDirectory(prefix=".materialize-", dir=vendor_root) as temporary:
            staged = {name: prepare(root, name, records[name], Path(temporary)) for name in missing}
            # Validate all preconditions again before installing any result.
            for name, record in records.items():
                destination = inside(root, record["path"])
                if name in staged:
                    if destination.exists() or destination.is_symlink():
                        raise ValueError(f"destination appeared during preparation: {destination}")
                else:
                    verify_vendor(root, name, record)
            try:
                for name, source in staged.items():
                    destination = inside(root, records[name]["path"])
                    # mkdir is exclusive; rename alone could replace an empty directory.
                    destination.mkdir()
                    installed.append(destination)
                    for child in source.iterdir():
                        os.rename(child, destination / child.name)
            except BaseException:
                for destination in reversed(installed):
                    shutil.rmtree(destination)
                raise
    finally:
        lock.rmdir()
    return list(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="verify only; never download or modify source")
    args = parser.parse_args()
    try:
        names = materialize(args.root.resolve(), check=args.check)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"vendor verification failed: {error}\n")
    print("Verified vendor source: " + ", ".join(names))


if __name__ == "__main__":
    main()
