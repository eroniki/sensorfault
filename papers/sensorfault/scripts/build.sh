#!/usr/bin/env bash
# Build the RA-L letter and refresh the committed PDF (sensorfault.pdf). RA-L accepts no supplementary PDF, so the letter stands alone.
set -euo pipefail
cd "$(dirname "$0")/.."
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cp build/main.pdf sensorfault.pdf
echo "wrote sensorfault.pdf"
