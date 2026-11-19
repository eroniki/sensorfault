#!/usr/bin/env bash
# Build the manuscript and refresh the committed PDF (depthcal-gs.pdf), as papers/sensorfault keeps its PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >build.log 2>&1 || { tail -30 build.log; rm -f build.log; echo "build failed" >&2; exit 1; }
rm -f build.log
cp build/main.pdf depthcal-gs.pdf
echo "wrote depthcal-gs.pdf"
