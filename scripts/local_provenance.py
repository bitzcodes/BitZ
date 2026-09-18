"""Source verification shared by release scripts and benchmark runners."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = {".git", "target", "__pycache__", ".pytest_cache"}


def generated(parts: tuple[str, ...]) -> bool:
    return (bool(GENERATED_DIRS.intersection(parts))
            or any(p == ".DS_Store" or p.startswith("._") for p in parts)
            or parts[-1].endswith((".pyc", ".pyo")))


def inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"invalid workspace-relative path: {relative!r}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes workspace: {relative}")
    return result


def sha256(path: Path) -> str:
    # Chunked rather than hashlib.file_digest (Python 3.11+); same digest.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_id(kind: str, content: bytes) -> bytes:
    return hashlib.sha1(kind.encode() + b" " + str(len(content)).encode()
                        + b"\0" + content).digest()


def file_entry(path: Path) -> tuple[str, bytes]:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return "120000", os.fsencode(os.readlink(path))
    if stat.S_ISREG(mode):
        return ("100755" if mode & 0o111 else "100644"), path.read_bytes()
    raise ValueError(f"not a source file or symlink: {path}")


def source_inventory(directory: Path) -> list[str]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"missing materialized vendor directory: {directory}")
    result = []
    def visit(path, relative):
        for child in sorted(path.iterdir()):
            name = relative / child.name
            if generated(name.parts):
                continue
            if child.is_symlink() or not child.is_dir():
                file_entry(child)  # Reject sockets/devices and other unsupported files.
                result.append(name.as_posix())
            else:
                visit(child, name)
    visit(directory, Path())
    return result


def source_tree(directory: Path) -> str:
    """Compute a Git tree ID without Git or a repository, preserving file modes."""
    tree = {}
    for name in source_inventory(directory):
        node = tree
        parts = Path(name).parts
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        mode, content = file_entry(directory / name)
        node[parts[-1]] = (mode, object_id("blob", content))
    def encode(node):
        entries = []
        for name, value in node.items():
            directory = isinstance(value, dict)
            mode, oid = ("40000", encode(value)) if directory else value
            key = os.fsencode(name) + (b"/" if directory else b"")
            entries.append((key, mode.encode() + b" " + os.fsencode(name) + b"\0" + oid))
        return object_id("tree", b"".join(data for _, data in sorted(entries)))
    return encode(tree).hex()


def records(root: Path = ROOT) -> dict:
    return tomllib.loads((root / "provenance.toml").read_text())


def vendors(root: Path = ROOT) -> dict:
    result = {name: record for name, record in records(root).items() if "path" in record}
    if not result:
        raise ValueError("provenance.toml contains no vendor snapshots")
    paths = []
    for name, record in result.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError(f"invalid vendor name: {name!r}")
        required = {"git", "path", "original_upstream_commit", "upstream_commit",
                    "snapshot_commit", "snapshot_tree", "patch", "patch_sha256"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{name}: incomplete provenance ({', '.join(sorted(missing))}); "
                             "reconstruct the authoritative vendor snapshot first")
        for key in ("original_upstream_commit", "upstream_commit", "snapshot_commit", "snapshot_tree"):
            if not re.fullmatch(r"[0-9a-f]{40}", record[key]):
                raise ValueError(f"{name}: invalid {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", record["patch_sha256"]):
            raise ValueError(f"{name}: invalid patch checksum")
        path = inside(root, record["path"])
        if len(Path(record["path"]).parts) != 2 or Path(record["path"]).parts[0] != "vendor":
            raise ValueError(f"{name}: vendor path must be vendor/<name>")
        if path in paths:
            raise ValueError(f"duplicate vendor path: {path}")
        paths.append(path)
        inside(root, record["patch"])
    return result


def verify_patch(root: Path, name: str, record: dict) -> Path:
    path = inside(root, record["patch"])
    if sha256(path) != record["patch_sha256"]:
        raise ValueError(f"{name}: patch checksum mismatch")
    return path


def verify_vendor(root: Path, name: str, record: dict) -> None:
    directory = inside(root, record["path"])
    actual = source_tree(directory)
    if actual != record["snapshot_tree"]:
        raise ValueError(f"{name}: materialized source differs from snapshot "
                         f"({actual} != {record['snapshot_tree']}); preserve local changes "
                         "and move this directory aside before reconstructing")


def vendor_snapshots(root: Path = ROOT) -> dict:
    result = records(root)
    for name, record in vendors(root).items():
        verify_patch(root, name, record)
        verify_vendor(root, name, record)
        result[name].update(expected_snapshot_commit=record["snapshot_commit"], dirty=False)
    return result


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), *args])


def root_metadata(root: Path = ROOT) -> dict:
    # Require metadata at this root, never accidentally report an enclosing repo.
    if (root / ".git").exists():
        revision = git(root, "rev-parse", "HEAD").decode().strip()
        status = git(root, "status", "--porcelain", "--untracked-files=normal").decode().strip()
        return dict(revision=revision, git_status=status, git_dirty=bool(status), source_kind="checkout")
    release = tomllib.loads((root / "release-metadata.toml").read_text())
    return dict(revision=release["root_revision"], git_status=None,
                git_dirty=None, source_kind="archive")


def source_patch(root: Path = ROOT) -> bytes:
    if (root / ".git").exists():
        return git(root, "diff", "--binary", "HEAD")
    root_metadata(root)  # Missing release metadata must not be mistaken for clean source.
    return b""


def repository_metadata(root: Path) -> dict:
    root = root.resolve()
    if (root / ".git").exists() or (root / "release-metadata.toml").is_file() or root == ROOT.resolve():
        return root_metadata(root)
    for record in vendor_snapshots(ROOT).values():
        if record.get("path") and inside(ROOT, record["path"]).resolve() == root:
            return dict(revision=record["snapshot_commit"], git_status=None,
                        git_dirty=False, source_kind="vendor")
    raise ValueError(f"no provenance for source directory: {root}")


def verify_histories(root: Path, bundle_directory: Path) -> list[str]:
    """Validate separately distributed histories; not required for source builds."""
    verified = []
    for name, record in vendors(root).items():
        patch = verify_patch(root, name, record).resolve()
        bundle = bundle_directory / Path(record["bundle"]).name
        if sha256(bundle) != record["bundle_sha256"]:
            raise ValueError(f"{name}: bundle checksum mismatch")
        with tempfile.TemporaryDirectory(prefix="bitz-history-") as temporary:
            repository = Path(temporary)
            git(repository, "init", "--quiet")
            git(repository, "bundle", "verify", str(bundle.resolve()))
            advertised = dict(line.split()[::-1] for line in
                              git(repository, "bundle", "list-heads", str(bundle.resolve())).decode().splitlines())
            expected = {"refs/heads/upstream": record["upstream_commit"],
                        "refs/heads/snapshot": record["snapshot_commit"]}
            if advertised != expected:
                raise ValueError(f"{name}: bundle must contain only upstream and snapshot refs")
            git(repository, "fetch", "--quiet", str(bundle.resolve()),
                "refs/heads/upstream:refs/heads/upstream", "refs/heads/snapshot:refs/heads/snapshot")
            snapshot, base = record["snapshot_commit"], record["upstream_commit"]
            if git(repository, "show", "-s", "--format=%P", snapshot).decode().split() != [base]:
                raise ValueError(f"{name}: expected one customization commit directly above upstream")
            if git(repository, "rev-parse", snapshot + "^{tree}").decode().strip() != record["snapshot_tree"]:
                raise ValueError(f"{name}: bundle snapshot tree mismatch")
            identities = git(repository, "log", snapshot, "--format=%an <%ae>%n%cn <%ce>").decode()
            # The identities to reject are supplied by the packager, not embedded here.
            forbidden = os.environ.get("RELEASE_FORBIDDEN_IDENTITIES")
            if forbidden and re.search(forbidden, identities, re.I):
                raise ValueError(f"{name}: identifying author/committer remains in bundle")
            git(repository, "checkout", "--quiet", "--detach", base)
            git(repository, "apply", "--index", "--whitespace=nowarn", str(patch))
            if git(repository, "write-tree").decode().strip() != record["snapshot_tree"]:
                raise ValueError(f"{name}: patch does not reproduce bundled snapshot")
        verified.append(name)
    return verified
