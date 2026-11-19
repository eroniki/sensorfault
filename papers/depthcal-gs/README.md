# depthcal-gs-manuscript

Draft skeleton of *Calibrated Multi-View-Consistent Depth Sampling with a Diffusion Model* in the Elsevier `elsarticle`
template, single-column preprint layout. Companion code and plan: `eroniki/depthcal-gs` (`PLAN.md`).

**Status: skeleton.** Section structure follows the plan; no experiment has been run. Every result table and figure is a
placeholder marked `TODO` in red.

## Build
```
scripts/build.sh             # latexmk -pdf main.tex, then copies build/main.pdf to depthcal-gs.pdf
git config core.hooksPath .githooks   # once per clone: pre-commit rebuilds and stages depthcal-gs.pdf
```
Class `elsarticle` with option `preprint,12pt`, bibliography style `elsarticle-num`, both from TeX Live.
`depthcal-gs.pdf` is the last committed build; the pre-commit hook rebuilds it whenever `main.tex`, `sections/`,
`figures/`, `refs/references.bib` or `latexmkrc` is part of a commit, so it always matches the sources in the same commit.

## Layout
- `main.tex` front matter, abstract, declarations, section includes
- `sections/` `intro`, `related`, `method`, `setup`, `results`, `conclusion`, `macros`
- `refs/references.bib` cited entries only; `refs/EVIDENCE.md` how each was checked
- `figures/` PDF figures (toy results; made by `scripts/make_figures.py --paper` in `eroniki/depthcal-gs`)
- `depthcal-gs.pdf` committed build; `scripts/build.sh` builds it; `.githooks/pre-commit` keeps it current

## Licence note
`elsarticle` is distributed with TeX Live (LPPL). Keep this repository private. Original text is the author's.
