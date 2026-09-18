#!/usr/bin/env bash
# Regenerate the MultiSwap-shaped comparison plots from scratch:
# runs both msshape bench sweeps (criterion, 10 samples each) and then
# renders docs/plots/msshape_{prove,verify}.{pdf,png}.
#
# Usage:
#   ./scripts/regen_msshape_plots.sh           # full re-bench + plots (~15-20 min)
#   ./scripts/regen_msshape_plots.sh --plots   # plots only, from existing criterion data
#
# Knobs:
#   IMOD_K=<k>   override IntEval's per-iteration variable count for the
#                imod side (default DEFAULT_K = 9); recorded in the plot? No —
#                note the k you used alongside the figure.
#
# The bench configs live in benches/imod_spartan_modp.rs (MSSHAPE_GATES)
# and benches/spartan_synthetic.rs (the msshape loop); keep the two gate
# lists in sync — the plot script pairs series by the `msshape_c{log2 cons}`
# tag and skips sizes missing on either side.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" != "--plots" ]]; then
  RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan_modp -- "msshape"
  RUSTFLAGS="-C target-cpu=native" cargo bench --bench spartan_synthetic -- "msshape"
fi

python3 scripts/plot_msshape.py
