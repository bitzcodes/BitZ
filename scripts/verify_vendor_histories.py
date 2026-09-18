#!/usr/bin/env python3
"""Verify separately downloaded vendor Git bundles against release provenance."""
import argparse
from pathlib import Path
import subprocess

from local_provenance import ROOT, verify_histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        verified = verify_histories(args.root.resolve(), args.bundle_dir.resolve())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"history verification failed: {error}\n")
    print("Verified vendor histories: " + ", ".join(verified))


if __name__ == "__main__":
    main()
