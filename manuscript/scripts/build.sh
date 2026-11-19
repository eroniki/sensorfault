#!/usr/bin/env bash
# Build the RA-L letter (build/main.pdf). RA-L accepts no supplementary PDF, so the letter stands alone.
# The earlier view-gate letter and its supplement are built by scripts/build_viewgate.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
