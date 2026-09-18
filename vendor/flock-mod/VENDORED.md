# Flock benchmark snapshot

The retained library source is materialized here. See the root provenance.toml
for the upstream and customization commits; full upstream history is supplied
in a separate bundle. Licenses and the upstream README are retained. The
upstream README describes the full project: its standalone benchmark scripts,
CUDA sources, examples, and development targets are outside this library
snapshot. Use the root README for the supported release campaigns.

Local changes let Ligerito authenticate its initial interleaved oracle using
two existing Merkle trees. Recursive levels retain the ordinary single-root
protocol. The initial layout fixes the lane placement, zero padding, leaf
widths, position domain, and hash before any challenges are sampled.

`flock-prover/src` retains the root-tracked field integration from the selected source snapshot for the SHA-256
R1CS witness generator and chain-shift reduction. Its manifest omits upstream
benches, examples, and development dependencies; the library uses the same
local `flock-core`. This source is preparation for a future selectable Flock
SHA prover. It is not connected to the hybrid runner yet; the current hybrid
uses Binius64 for SHA constraint proving and Flock/Ligerito for the shared
commitment and opening.
