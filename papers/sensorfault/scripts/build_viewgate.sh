#!/usr/bin/env bash
# Build the earlier view-gate letter (build/main_viewgate.pdf) and its supplement (build/supplement.pdf).
# The two documents cross-reference each other through xr, so each is built twice.
set -euo pipefail
cd "$(dirname "$0")/.."
for pass in 1 2; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error main_viewgate.tex
done
