"""Emit Cargo build metadata for development checkouts and source archives."""
from pathlib import Path
import sys
import subprocess

from local_provenance import git, root_metadata, source_inventory, vendor_snapshots


def metadata(root: Path):
    snapshots = vendor_snapshots(root)
    source = root_metadata(root)
    dirty = source["git_dirty"]
    values = {"BITZ_REVISION": source["revision"],
              "BITZ_DIRTY": "unknown" if dirty is None else "dirty" if dirty else "clean"}
    for name, variable in [("binius64", "BINIUS64_REVISION"), ("plonky3", "PLONKY3_REVISION"),
                           ("limber", "LIMBER_REVISION"), ("flock-mod", "FLOCK_REVISION")]:
        if name in snapshots:
            values[variable] = snapshots[name]["snapshot_commit"]
    watched = [root / "provenance.toml", root / "release-metadata.toml",
               root / "scripts/local_provenance.py", Path(__file__).resolve()]
    for record in snapshots.values():
        if "path" in record:
            watched += [root / record["patch"]]
            directory = root / record["path"]
            files = [directory / p for p in source_inventory(directory)]
            watched += files
            # Directory mtimes detect added files as well as edits to known files.
            directories = {directory}
            for file in files:
                directories.update(parent for parent in file.parents
                                   if parent.is_relative_to(directory))
            watched += sorted(directories)
    if (root / ".git").exists():
        for name in ["HEAD", "index", "packed-refs"]:
            watched.append(root / git(root, "rev-parse", "--git-path", name).decode().strip())
        head_path = root / git(root, "rev-parse", "--git-path", "HEAD").decode().strip()
        head = head_path.read_text().strip()
        if head.startswith("ref: "):
            watched.append(root / git(root, "rev-parse", "--git-path", head[5:]).decode().strip())
    return values, watched


if __name__ == "__main__":
    try:
        values, watched = metadata(Path(sys.argv[1]).resolve())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"source verification failed: {error}")
    for path in watched:
        print(f"cargo:rerun-if-changed={path}")
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"invalid multiline build metadata: {key}")
        print(f"cargo:rustc-env={key}={value}")
