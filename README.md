# sensorfault-manuscript

Draft v0 of *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface*, in the
Elsevier `elsarticle` template, single-column preprint layout (preprint first; no pay-to-publish venue). Companion code
and plan: `eroniki/sensorfault`; reference design: `docs/design-3dgs-bayes.md` in that repo.

The paper proposes a depth-only view gate for depth-supervised Gaussian splatting. Each 2DGS surfel keeps its shape for
rendering and a separate 10x10 uncertainty over mean, rotation error, log scales and opacity (opacity frozen after a
warm-up). In the fuse step each view sends the surfels an information-form message (alpha-weighted J^T R^-1 J); leaving
a view out is a subtraction plus one Newton step (the cavity step of expectation propagation, used once, without moment
matching). The tested view is compared with the other views' surface rendered into it, with one free noise variable
(a scene temperature) and a shared distance bias.

**Status: pre-experiment draft.** No experiment has been run. Every result table and figure is a placeholder marked
`TODO` in red; hypotheses H1-H6 (Sec. 4) are pre-registered predictions, not findings.

## Build
```
latexmk -pdf main.tex        # output: build/main.pdf (TeX Live; uses ./latexmkrc)
```
Class `elsarticle` with option `preprint,12pt` and bibliography style `elsarticle-num`, both from TeX Live.
`draft_v0.pdf` is the last committed build: main text pp. 1-12, declarations and references from p. 13, then the
Supplementary Material (S1-S4) in the same PDF.

## Outline
Main text: Abstract, Keywords; 1 Introduction; 2 Related Work (short); 3 Method (setting, measurement model and faults;
surfels with an uncertainty that renders; fusion on rendering, leave-view-out and the message-passing reading; the view
gate with Algorithm 1 "Train, freeze, fuse, test"); 4 Experimental Design and Hypotheses (H1-H6, breakdown);
5 Results (status); 6 Conclusions, Limitations, Future Work. Declarations; References.
Supplement: S1 related work; S2 derivations (ray-surfel and opacity Jacobians, information update and Newton leave-out,
solution separation, rendered variance, minimum detectable bias, scene temperature and heterogeneity); S3 experimental
setup; S4 planned tables and figures.

## Layout
- `main.tex` front matter, abstract, declarations, section includes
- `sections/` main text: `intro`, `related`, `method`, `setup`, `results`, `conclusion`; `supplement.tex` includes `supp_*.tex`; `macros.tex`
- `refs/references.bib` cited entries only; `refs/EVIDENCE.md` how each was checked
- placeholder figures are drawn by `\todofig` in `sections/macros.tex`; there are no figure files yet

## Licence note
`elsarticle` is distributed with TeX Live (LPPL). Keep this repository private until the preprint is released.
Original text in `sections/` and `main.tex` is the author's.

## Open items before the preprint
AI-use declaration and the code/data statement (TODO in `main.tex`); Holm's step-down procedure is named but not cited;
final literature pass; results, figures and tables from the experiment code; choice of a subscription (no-APC)
Elsevier journal.
