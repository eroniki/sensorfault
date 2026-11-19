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
The previous paper, *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface*,
with its supplement and figures, is in the git history (last present in commit 42a2fc0). Its depth-only view gate
is the depth-consensus baseline of this letter; the code is `code/sensorfault/src/sensorfault/gs/gate.py`.

## Double-anonymous review
Set `\anonymoustrue` near the top of `main.tex` to replace the author block and running
head with "Anonymous". The repository URL TODO in the Acknowledgment must stay anonymous too.

## Layout
- `main.tex` RA-L letter: title, abstract, index terms, section includes, acknowledgment; `preamble.tex` packages and macros
- `sections/` `intro`, `related`, `method`, `setup`, `results`, `conclusion`; `macros.tex` (placeholders `\ph`, `\TODO`, `\todocell`, `\todofig`)
- `refs/references.bib` cited entries, plus `stat_holm` and the UseGeo, H3D and UrbanScene3D entries that the text names but does not cite yet; `refs/EVIDENCE.md` how each was checked
- `figures/letter/` the four letter figures and `stage_a_snapshot.json`, written by `code/sensorfault/experiments/letter_figures_tikz.py`; `figures/figstyle.tex` shared PGFPlots style

## License note
`IEEEtran` is distributed with TeX Live (LPPL). Make the repository private during double-anonymous review.
Original text in `sections/` and `main.tex` is the author's.

## Open items before the preprint
AI-use declaration and the code/data statement (TODO in `main.tex`); Holm's step-down procedure is named but not cited;
final literature pass; results, figures and tables from the experiment code; resize the TikZ generators for the 252 pt column (main-text figures span both columns for now).
