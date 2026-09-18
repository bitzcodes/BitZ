#!/usr/bin/env python3
"""Build the pinned, isolated Binius SHA+ECDSA worker and retain its provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))
from local_provenance import vendor_snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target"))
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    command = ["cargo", "build", "--release", "--locked"]
    if args.offline:
        command.append("--offline")
    subprocess.run(command, cwd=ROOT, env=env, stdout=sys.stderr, check=True)
    binary = ROOT / "target/release/binius64-sha256-ecdsa"
    info = json.loads(subprocess.check_output([str(binary), "--build-info"], text=True))
    info.update(vendor_snapshots=vendor_snapshots(ROOT.parents[1]), binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    # Resolved sha2 backends, one line per version: the bitz path dependency
    # pulls flock-core, whose `asm` feature accelerates sha2 0.10 (host-side
    # p256 digests only); binius-hash's BaseFold Merkle hashing is sha2 0.11,
    # a separate version that feature unification cannot touch. Recorded so a
    # campaign's provenance pins what actually hashed.
    tree = subprocess.check_output(
        ["cargo", "tree", "-f", "{p} {f}", "--edges", "normal,build", "--prefix", "none"],
        cwd=ROOT, env=env, text=True)
    info["sha2_features"] = sorted({line.strip().removesuffix(" (*)") for line in tree.splitlines()
                                    if line.strip().startswith("sha2 v")})
    binary.with_suffix(".build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
