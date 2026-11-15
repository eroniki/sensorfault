#!/usr/bin/env bash
# Build the standalone PDFs (and PNG previews) of the TikZ/PGFPlots figures: ./build.sh [fig3 fig4 ...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
names=("$@")
if [ ${#names[@]} -eq 0 ]; then names=($(ls *_body.tex | sed 's/_body.tex//')); fi
for n in "${names[@]}"; do
  for pass in 1 2; do   # two passes: legends placed with \ref need the second
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build -jobname="$n" "\\def\\figname{$n}\\input{wrapper}" >build/"$n".log 2>&1 \
      || { tail -25 build/"$n".log; echo "FAILED $n" >&2; exit 1; }
  done
  pdftoppm -r 200 -png -singlefile build/"$n".pdf build/"$n"
  echo "built build/$n.pdf"
done
