#!/usr/bin/env bash
# Copy the TikZ sources and data used by the paper into the manuscript repository: ./sync_paper.sh /path/to/manuscript
set -euo pipefail
cd "$(dirname "$0")"
dest="${1:?usage: sync_paper.sh MANUSCRIPT_DIR}/figures/tikz"
mkdir -p "$dest/data"
cp preamble.tex fig3_body.tex fig4_body.tex fig6_body.tex toy_render_common.tex toy_square_body.tex toy_warp_body.tex "$dest"/
cp data/fig3_*.csv data/fig4_mdf_wide.csv data/fig4_null.csv data/fig6_*.csv data/toy_*_*_*_*.csv data/toy_outline_*.csv data/toy_render_macros.tex "$dest/data/"
echo "synced to $dest"
