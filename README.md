# sensorfault-manuscript

Draft v0 of *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface* (aerial survey
setting; UseGeo, MatrixCity, ETH3D as a sanity check; off-the-shelf depth models only; all datasets non-commercial,
see the data statement), in the
IEEE Robotics and Automation Letters (RA-L) template (`IEEEtran`, journal mode), for the RA-L submission path with the IROS
presentation option. The supplement is a separate document. Companion code
and plan: `eroniki/sensorfault`; reference design: `docs/design-3dgs-bayes.md` in that repo.

The paper proposes a depth-only view gate for depth-supervised Gaussian splatting. Each 2DGS surfel keeps its shape for
rendering and a separate 10x10 uncertainty over mean, rotation error, log scales and opacity (opacity frozen after a
warm-up). In the fuse step each view sends the surfels an information-form message (alpha-weighted J^T R^-1 J); leaving
a view, or a block of neighbouring strip frames, out is a subtraction plus one Newton step (the cavity step of expectation propagation, used once, without moment
matching). The tested view is compared with the other views' surface rendered into it, with one free noise variable
(a scene temperature) and a shared distance bias. Training, freezing, fusing and testing repeat in rounds that drop
flagged views (one per round by default), with a Holm step-down schedule across rounds.

**Status: pre-experiment draft.** No experiment has been run. Every result table and figure is a placeholder marked
`TODO` in red; hypotheses H1-H6 (Sec. 4) are pre-registered predictions, not findings.

## Build
```
scripts/build.sh             # output: build/main.pdf (letter) and build/supplement.pdf (TeX Live; uses ./latexmkrc)
```
Class `IEEEtran` (`journal`, two-column letter; the supplement uses `onecolumn`) and bibliography style `IEEEtran`, both
from TeX Live (`texlive-publishers`). `preamble.tex` is shared by both documents. They cross-reference each other with
`xr`: in the letter, labels from the supplement carry the prefix `S-` (e.g. `\ref{S-sup:h7}`); in the supplement, labels
from the letter carry `M-`. The build script runs each document twice so these resolve.
`draft_v0.pdf` (letter) and `draft_v0_supplement.pdf` are the last committed builds.
RA-L allows 6 pages plus up to 2 paid extra pages, references included; the letter is 8 pages. Material cut from the letter lives in the supplement (method details, results in full, hypotheses and limitations in full). After moving text between the two documents, run `scripts/xref_prefixes.py` to re-derive the S-/M- prefixes.

## Outline
Main text: Abstract, Keywords; 1 Introduction; 2 Related Work (short); 3 Method (setting, measurement model and faults;
poses as a trajectory spline refined by render-and-compare; surfels with an uncertainty that renders; fusion on rendering, leave-view-out and the message-passing reading; the view
gate; leave-block-out; the outer loop with Algorithm 1 "Train, freeze, fuse, test" in rounds, Holm across rounds); 4 Experimental Design and Hypotheses (H1-H6, breakdown);
5 Results (status); 6 Conclusions, Limitations, Future Work. Acknowledgment (data/code and AI-use statements); References.
Supplement (`supplement.tex`, separate PDF): S1 related work; S2 derivations (ray-surfel and opacity Jacobians, information update and Newton leave-out,
solution separation, rendered variance, minimum detectable bias, scene temperature and heterogeneity); S3 experimental
setup; S4 planned tables and figures.

## Layout
- `main.tex` RA-L letter: title, abstract, index terms, section includes, acknowledgment; `supplement.tex` supplementary material; `preamble.tex` shared packages and macros
- `sections/` main text: `intro`, `related`, `method`, `setup`, `results`, `conclusion`; `supp_*.tex` for the supplement; `macros.tex`
- `refs/references.bib` cited entries only; `refs/EVIDENCE.md` how each was checked
- `figures/e1/` E1 figures (pgfplots) and summary table, with their CSV data, copied from `eroniki/sensorfault` `results/e1/figures/` (regenerate there with `experiments/e1_figures.py`); `figures/e1q/` E1 qualitative checks (PNG panels and LaTeX), copied from `results/e1/qualitative/` (regenerate with `experiments/e1_qualitative.py`); remaining placeholders are drawn by `\todofig` in `sections/macros.tex`

## Licence note
`IEEEtran` is distributed with TeX Live (LPPL). Keep this repository private until the preprint is released.
Original text in `sections/` and `main.tex` is the author's.

## Open items before the preprint
AI-use declaration and the code/data statement (TODO in `main.tex`); Holm's step-down procedure is named but not cited;
final literature pass; results, figures and tables from the experiment code; cut the letter to the RA-L page limit and
the abstract to IEEE length; resize the TikZ generators for the 252 pt column (main-text figures span both columns for now).
