# TikZ / PGFPlots figures

Style: `preamble.tex` (after the figures in Ambarkutuk & Plassmann, Sensors 2024; see `../summary.md`). Each `*_body.tex` is one `tikzpicture` that reads CSV from `data/` and
images from `img/`; `build.sh [name ...]` compiles standalone PDFs and PNG previews into `build/` (two pdflatex passes; `\figw` = 16 cm there). In a paper the body is `\input` with
`\figw` set to the text width. Data and images are produced from saved results by `scripts/export_tikz_data.py` (CPU) and `scripts/export_render_images.py` (GPU, from saved surfels; never trains).
`sync_paper.sh DIR` copies the sources and data used by the manuscript. Needs TeX Live with `pgfplots`, `DejaVuSans`, `sansmath`, `xstring`, `standalone`, `adjustbox`.
