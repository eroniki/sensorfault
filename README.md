# biasfield-gs-manuscript

Draft skeleton of *Calibrated Multi-View-Consistent Depth Sampling with a Diffusion Model* in the Elsevier `elsarticle`
template, single-column preprint layout. Companion code and plan: `eroniki/biasfield-gs` (`PLAN.md`).

**Status: skeleton.** Section structure follows the plan; no experiment has been run. Every result table and figure is a
placeholder marked `TODO` in red.

## Build
```
latexmk -pdf main.tex        # output: build/main.pdf (TeX Live; uses ./latexmkrc)
```
Class `elsarticle` with option `preprint,12pt`, bibliography style `elsarticle-num`, both from TeX Live.

## Layout
- `main.tex` front matter, abstract, declarations, section includes
- `sections/` `intro`, `related`, `method`, `setup`, `results`, `conclusion`, `macros`
- `refs/references.bib` cited entries only; `refs/EVIDENCE.md` how each was checked

## Licence note
`elsarticle` is distributed with TeX Live (LPPL). Keep this repository private. Original text is the author's.
