# Release source provenance

The release is reconstructed from the public dependency commits pinned by root
revision `8446d64fc011b4b10b6ae852621fd1be4461afaa`, together with the root-tracked
Flock library integration and the packaging intent of `02701f81`. This release
does not claim source equivalence with earlier local snapshots.

| Vendor | Original upstream |
| --- | --- |
| Limber | `959575409f38a0894dab2f2a7b7125c8aa424b78` |
| Plonky3 | `59be31386d5ab81b87dbceb0b83bf797f9eefaec` |
| Binius64 | `c28940ae693c3999fd225cc6142f43d07d1100bb` |
| Flock | `e636760f8dae78306f804554fb4244993758b011` |

Each final `snapshot_tree` includes the release adaptations, described in the
manifest and represented in a binary-capable patch. Official upstream URLs are recorded
for reconstruction; builds have no personal-fork dependency.

## History normalization

Only the agreed, verified author/committer identities are replaced by
`bitzcodes <bitzcodes@fastmail.com>`; signatures invalidated by that rewrite are
removed. Every upstream source tree, parent relationship, commit message,
timestamp, and other contributor identity is preserved. The original and
normalized tips are recorded separately.

Limber's upstream `main` executable remains in its historical trees. Its removal
is now in the single customization commit, together with removal of tracked
Python bytecode. This replaces the previous approach of removing the executable
throughout history. Every history bundle advertises only `upstream`
and `snapshot`; `snapshot` has exactly one parent, `upstream`. Bundles contain
full history and are distributed separately from the source ZIP.

## Current-source adaptations

- Limber retains its published benchmark instrumentation, aligns the three
  MultiSwap digest domains with the root launcher, reports its verified snapshot
  revision without Git metadata, and keeps the exact official `halo2curves` pin.
- Plonky3 retains its published benchmarks. The optional external `zkhash`
  development dependency and its reference-comparison test are omitted; the
  remaining BN254 tests and library implementation are retained.
- Binius64 retains its published implementation and instrumentation. An incidental
  personal attribution in the SHA/P-256 example introduction is removed.
- Flock retains the root's shared-field and interleaved-oracle implementation.
  The snapshot contains the core and prover libraries, licenses and upstream
  README. Standalone upstream benchmarks, CUDA sources and development targets
  remain omitted, and the workspace manifest lists only retained crates.
- Lockfiles are present for every retained workspace. Existing lockfiles keep
  their dependency versions; newly materialized workspaces receive lockfiles.

## Retained attribution

- License and copyright notices in the retained sources are unchanged.
- Source comments crediting third-party designs and implementations are kept.
- Third-party documentation keeps its citations of prior work.
- Patches retain upstream text in deletion/context lines. Upstream historical
  contents and commit messages retain their original attributions. These are
  necessary for exact reconstruction.

The root Git history is retained internally. It is neither normalized nor
included in the source ZIP.

## Changes on 2026-09-18

These edits restore the state the paper's numbers were measured on:

- `src/merged_forest/schedule.rs`: on Apple Silicon, large single-claim forests
  keep the L/4 schedule above four workers; tests updated and one added.
- `scripts/bench_gate.py`: restores the sustained-idle wait before each gated
  campaign; tested in `scripts/test_multiplication_launcher.py`.
- `scripts/bench_support.py`, `scripts/local_provenance.py`: file checksums no
  longer use `hashlib.file_digest`; the digests are unchanged.
- `scripts/hybrid_table.py`: also accepts sweeps that recorded the pre-rename
  `F2Z_*` knob names.
- `README.md`: measurement conditions, and the comparisons this artifact does not
  reproduce.

The release identity checks now read their patterns from the environment
(`RELEASE_FORBIDDEN_IDENTITIES`, `RELEASE_FORBIDDEN_FORKS`) instead of embedding
them, and the published-fork pins are omitted from this anonymous release.
