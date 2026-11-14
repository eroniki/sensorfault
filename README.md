# sensorfault-manuscript

Draft v0 of *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian
Surface*, for IEEE Access (official LaTeX template). Companion code and plan: `eroniki/sensorfault`.
The gate is the contribution; the monocular-depth pipeline is the test bed.

**Status: pre-experiment draft.** No experiment has been run. Every result table and figure is a
placeholder marked `TODO` in red; the hypotheses H1-H5 in Sec. IV are pre-registered *predictions*,
not findings. Do not submit before the TODOs are resolved.

## Build
```
latexmk -pdf main.tex        # output: build/main.pdf (needs TeX Live; uses ./latexmkrc)
```
`latexmkrc` points TeX at `template/` for the class, fonts and `IEEEtran.bst`.
`draft_v0.pdf` is the last committed build.

## Outline (follows the section pattern of the author's Sensors 23(23):9309 and 24(23):7744; IEEE Access mandates none)
Abstract + Index Terms; I Introduction (A Relevant Literature, B Challenges (C1-C6), C Baseline Study and
Fundamental Differences, D Summary of the Contributions, E Organization); II Method (A Definitions and
Notation, B Forward Problem, C Inverse Problem, D View-Level Gate (shared-surface consensus by cross-fitting), E Test Bed); III Experiments
(A Experimental Setup, B Hypotheses and Breakdown Scenarios, C Implementation); IV Results;
V Conclusions and Future Work (with Limitations); Appendices A-B.

## Layout
- `main.tex` front matter, abstract, section includes
- `sections/` intro, method, setup (Experiments), results (placeholders), conclusion, appendix
- `refs/references.bib` merged bibliography (`refs/part_*.bib` + `part_*_evidence.md` are the sources and the evidence log)
- `template/` IEEE Access template files (see licence note)
- `figures/` (empty; placeholders are drawn by `\todofig` in `sections/macros.tex`)

## Citation verification
Each entry in `references.bib` was checked against a Crossref/DOI or arXiv record (publisher pages
were often unreachable by automated fetch); sources are logged per key in `refs/part_*_evidence.md`.
Known limits: initials-only authors for a few older papers; some one-line descriptions come from the
record, not an abstract; `fus_zach08` (Zach 2008) could not be verified and is not cited;
`gs_bayesiangsslam` has a Crossref RA-L record but an arXiv page without a venue; `gs_yogo` is
arXiv-only and only its abstract was read. Re-check all entries before submission.

## Licence note
`template/` is IEEE's distributed template. Only `IEEEtran.cls` states a licence (LPPL 1.3); the
other class files and the embedded fonts state none, and some fonts are commercial. **Keep this
repository private**; do not publish `template/` or `draft_v0.pdf` without checking IEEE's terms.
Original text in `sections/` and `main.tex` is the author's.

## Open items before submission
Author/affiliation/funding; AI-use disclosure in the Acknowledgment (IEEE Access requires it);
confirm Algorithm 1 inequality directions against the typeset equations of the cited source paper;
read YOGO in full; final literature pass; results, figures and tables from the experiment code;
satellite section decision.
