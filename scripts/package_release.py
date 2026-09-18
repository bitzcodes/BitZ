#!/usr/bin/env python3
"""Create a verified, reproducible source ZIP without local development state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile

from local_provenance import (ROOT, file_entry, generated, git, object_id,
                              sha256, source_inventory, vendor_snapshots, vendors)

EXCLUDED = (
    ".claude", ".history", ".tmp", ".tools", ".vscode", ".codex", ".agents",
    "bench_results", "benchmark-results", "docs", "experiments", "flock-zkperf-traces",
    "output", "outputs", "paper", "PerfRuns", "tmp", "benches/results",
    "benchmarks/results", "vendor/field/validation", ".olcli.json", ".olignore.local",
    ".env", "src/sumcheck/performance.csv",
)
METADATA_FILES = {"release-metadata.toml", "release-source-files.txt", "release-files.json"}


def excluded(name: str, bundles=()) -> bool:
    return (any(name == p or name.startswith(p + "/") for p in (*EXCLUDED, *bundles))
            or generated(Path(name).parts))


def tracked_entries(root: Path):
    for entry in git(root, "ls-tree", "--full-tree", "-rz", "HEAD").split(b"\0"):
        if entry:
            header, name = entry.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            yield os.fsdecode(name), mode, kind, oid


def collect(root: Path):
    vendor_snapshots(root)
    records = vendors(root)
    bundles = tuple(r["bundle"] for r in records.values() if "bundle" in r)
    if not (root / ".git").exists():
        raise ValueError("packaging requires the development checkout, not an extracted release")
    if git(root, "ls-files", "--unmerged").strip():
        raise ValueError("resolve and commit the cherry-pick before packaging")
    included, omitted = {}, []
    for name, committed_mode, kind, oid in tracked_entries(root):
        if excluded(name, bundles):
            omitted.append(name)
            continue
        if name in METADATA_FILES:
            raise ValueError(f"generated archive metadata must not be tracked: {name}")
        if kind != "blob":
            raise ValueError(f"source must be materialized, not a submodule: {name}")
        path = root / name
        if not path.exists() and not path.is_symlink():
            raise ValueError(f"included tracked file is missing: {name}")
        mode, data = file_entry(path)
        if mode != committed_mode or object_id("blob", data).hex() != oid:
            raise ValueError(f"included source is not committed and clean: {name}")
        included[name] = (mode, data)
    # A staged change may differ from HEAD even when the working copy was restored.
    staged = git(root, "diff", "--cached", "--name-only", "-z").decode().split("\0")
    if any(name and not excluded(name, bundles) for name in staged):
        raise ValueError("included source has staged changes; commit before packaging")
    # All verified vendor files must actually be in the root source inventory.
    for record in records.values():
        prefix = record["path"]
        for name in source_inventory(root / prefix):
            if prefix + "/" + name not in included:
                raise ValueError(f"vendor file missing from committed release inventory: {prefix}/{name}")
        if record["patch"] not in included:
            raise ValueError(f"review patch missing from release inventory: {record['patch']}")
    for name, (mode, data) in included.items():
        if mode == "120000":
            link = Path(os.fsdecode(data))
            if link.is_absolute():
                raise ValueError(f"absolute source symlink: {name}")
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(f"symlink escapes exported workspace: {name}")
            relative = target.relative_to(root.resolve()).as_posix()
            if relative not in included and not any(n.startswith(relative + "/") for n in included):
                raise ValueError(f"symlink target is missing from archive: {name}")
        # Fork owners to reject are supplied by the packager, not embedded here.
        forks = os.environ.get("RELEASE_FORBIDDEN_FORKS")
        if forks and Path(name).name in {"Cargo.toml", "Cargo.lock"} and re.search(
                rb"github\.com[/:](?:" + forks.encode() + rb")/", data, re.I):
            raise ValueError(f"personal fork dependency remains: {name}")
    if "README.md" not in included or "provenance.toml" not in included:
        raise ValueError("release must include README.md and provenance.toml")
    return included, sorted(omitted)


def zip_entry(name, mode):
    info = zipfile.ZipInfo("bitz/" + name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    unix_mode = stat.S_IFLNK | 0o777 if mode == "120000" else stat.S_IFREG | (0o755 if mode == "100755" else 0o644)
    info.external_attr = unix_mode << 16
    return info


def package(root: Path, output: Path | None, *, dry_run=False, max_bytes=20_000_000):
    root = root.resolve()
    files, omitted = collect(root)
    if dry_run:
        return dict(included=sorted(files), excluded=omitted, max_bytes=max_bytes)
    if output is None:
        raise ValueError("--output is required unless --dry-run is used")
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to replace existing output: {output}")
    if max_bytes <= 0:
        raise ValueError("maximum ZIP size must be positive")
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    listing = "\n".join(sorted(files)) + "\n"
    files["release-source-files.txt"] = ("100644", listing.encode())
    files["release-metadata.toml"] = ("100644", f'root_revision = "{revision}"\n'.encode())
    inventory = {name: {"mode": mode, "sha256": hashlib.sha256(data).hexdigest()}
                 for name, (mode, data) in sorted(files.items())}
    files["release-files.json"] = ("100644", (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode())
    with tempfile.TemporaryDirectory(prefix="bitz-zip-") as temporary:
        pending = Path(temporary) / "release.zip"
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, (mode, data) in sorted(files.items()):
                archive.writestr(zip_entry(name, mode), data, compresslevel=9)
        size = pending.stat().st_size
        if size > max_bytes:
            raise ValueError(f"ZIP is {size:,} bytes, exceeding the {max_bytes:,}-byte limit")
        with zipfile.ZipFile(pending) as archive:
            expected = {"bitz/" + name for name in files}
            if set(archive.namelist()) != expected or len(archive.infolist()) != len(expected):
                raise ValueError("ZIP inventory mismatch")
            for name, (mode, data) in files.items():
                info = archive.getinfo("bitz/" + name)
                if archive.read(info) != data or info.external_attr != zip_entry(name, mode).external_attr:
                    raise ValueError(f"ZIP content or mode mismatch: {name}")
        digest = sha256(pending)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Stage on the destination filesystem, then publish atomically without
        # replacing any existing path. Interruptions cannot leave a partial ZIP
        # at the requested output name.
        with tempfile.NamedTemporaryFile(prefix=".bitz-zip-", dir=output.parent) as staged:
            with pending.open("rb") as source:
                shutil.copyfileobj(source, staged)
            staged.flush()
            os.fsync(staged.fileno())
            os.link(staged.name, output)
    return dict(archive=str(output), bytes=size, sha256=digest, root_revision=revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    args = parser.parse_args()
    try:
        result = package(args.root, args.output, dry_run=args.dry_run, max_bytes=args.max_bytes)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"release packaging failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
