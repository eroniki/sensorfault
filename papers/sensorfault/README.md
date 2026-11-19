# Manuscript

Draft of *Auditing Monocular Depth with Per-Source Information on Gaussian Surfels*, in the IEEE Robotics and
Automation Letters (RA-L) template (`IEEEtran`, journal mode), for the RA-L submission path with the IROS
presentation option. Code and plan are in the same repository under `code/sensorfault/` (`PLAN.md`, design v4.0; method
`docs/learned-indicator.md`; protocol `docs/learned-indicator/experiment-design.md`; paper design
`docs/learned-indicator/paper-plan.md`). Run the commands below from `papers/sensorfault/`.

The paper stores, on each planar Gaussian surfel patch, a Fisher information record on its placement (an offset and
two tilts) separately for the monocular depth prior and for the pixels. To first order the image Jacobian factors
through the depth Jacobian, so both records are quadratic forms on the same coordinates and their disagreement is a
statistic with a null distribution. It claims that a depth-only map is overconfident by the share of depth error that
all frames repeat, that the pixel-against-depth test detects repeated faults that depth consensus cannot see, and
that the pixel information can be calibrated on the scene without labels.

**Status: only a preliminary Stage A run exists (Fig. 3).** Every number is a red placeholder (`\ph{...}`, `\TODO{...}`,
`\todocell`), and the letter builds to 5 pages of text plus references in that state. The four figures are TikZ, written by
`code/sensorfault/experiments/letter_figures_tikz.py` into `figures/letter/`: the teaser and overview are schematic, the detection
figure is an analytic prediction, and the Stage A figure is a **preliminary** run (numbers in `figures/letter/stage_a_snapshot.json`;
regenerate with `python3 experiments/letter_figures_tikz.py ../../papers/sensorfault --stage-a results/stage_a` from `code/sensorfault`). Statements
about the closest prior work rest on abstracts (`refs/EVIDENCE.md`).

## Build
```
scripts/build.sh             # output: build/main.pdf (TeX Live; uses ./latexmkrc)
scripts/build_viewgate.sh    # the earlier view-gate letter and its supplement (see below)
```
`sensorfault.pdf` is the committed build of this letter (`scripts/build.sh` refreshes it); versions live in the git history.
Class and bibliography style `IEEEtran`, from TeX Live (`texlive-publishers`). RA-L allows 6 pages plus up to 2 paid
extra pages, references included, and **no supplementary PDF** (video only), so the letter stands alone. The abstract
limit is 1,200 characters in the submission form; the current text is 1,181 with one-letter placeholders. Set
`\anonymoustrue` in `main.tex` for double-anonymous review.

## Outline
Abstract, Index Terms; I Introduction; II Related Work; III Method (setting; two records on one placement; honest
information; tension and the score test; fusion and scoring); IV Experimental Design (data, faults, baselines,
hypotheses H1-H7); V Results (overconfidence law; pixel information; detection; fused map; real scenes);
VI Limitations and Conclusion; Acknowledgment; References. Four figures (teaser, overview, Stage A law,
detection) and three tables (detection, maps, real scenes).

The letter numbers its hypotheses consecutively. They map to the protocol's IDs as: letter H1, H2, H3, H4 = protocol
H1, H2, H3, H4; letter H5 = protocol H9; letter H6 = protocol H10; letter H7 = protocol H7.

## The earlier view-gate letter
`main_viewgate.tex` with `sections/vg_*.tex`, and `supplement.tex` with `sections/supp_*.tex`, are the previous
paper, *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface*, kept intact.
Its depth-only view gate is the depth-consensus baseline of the new letter, and its controlled experiments (E1, the
pose check) are the source of the figures under `figures/`. Build it with `scripts/build_viewgate.sh`;
its last committed PDFs are in the git history. The two documents cross-reference each other with `xr` (prefix `S-` in the letter, `M-` in the
supplement); run `scripts/xref_prefixes.py` after moving text between them.

## Double-anonymous review
Set `\anonymoustrue` near the top of `main.tex` and `supplement.tex` to replace the author block and running
head with "Anonymous". The repository URL TODO in the Acknowledgment must stay anonymous too.

## Layout
- `main.tex` RA-L letter: title, abstract, index terms, section includes, acknowledgment; `supplement.tex` supplementary material; `preamble.tex` shared packages and macros
- `sections/` new letter: `intro`, `related`, `method`, `setup`, `results`, `conclusion`; view-gate letter: `vg_*.tex`; its supplement: `supp_*.tex`; `macros.tex`
- `refs/references.bib` cited entries only; `refs/EVIDENCE.md` how each was checked
- `figures/e1/` E1 figures (pgfplots) and summary table, with their CSV data, copied from `code/sensorfault/results/e1/figures/` (regenerate there with `experiments/e1_figures.py`); `figures/e1q/` E1 qualitative checks (PNG panels and LaTeX), copied from `results/e1/qualitative/` (regenerate with `experiments/e1_qualitative.py`); remaining placeholders are drawn by `\todofig` in `sections/macros.tex`

## License note
`IEEEtran` is distributed with TeX Live (LPPL). Keep this repository private until the preprint is released.
Original text in `sections/` and `main.tex` is the author's.

## Open items before the preprint
AI-use declaration and the code/data statement (TODO in `main.tex`); Holm's step-down procedure is named but not cited;
final literature pass; results, figures and tables from the experiment code; resize the TikZ generators for the 252 pt column (main-text figures span both columns for now).
