# sensorfault-manuscript

Draft v0 of *Sensor-Level Fault Elimination for Multi-Sensor Depth Fusion on a Fixed-Budget
Gaussian-Splat Surface*, for IEEE Access (official LaTeX template). Companion code and plan:
`eroniki/sensorfault`.

**Status: pre-experiment draft.** No experiment has been run. Every result table and figure is a
placeholder marked `TODO` in red; the hypotheses H1-H5 in Sec. IV are pre-registered *predictions*,
not findings. Do not submit before the TODOs are resolved.

## Build
```
latexmk -pdf main.tex        # output: build/main.pdf (needs TeX Live; uses ./latexmkrc)
```
`latexmkrc` points TeX at `template/` for the class, fonts and `IEEEtran.bst`.
`draft_v0.pdf` is the last committed build.

## Outline (follows the section order requested; IEEE Access mandates none)
Abstract + Index Terms; I Introduction; II Related Work; III Problem Formulation; IV Proposed Method;
V Experimental Setup; VI Results; VII Discussion and Limitations; VIII Conclusion.

## Layout
- `main.tex` front matter, abstract, section includes
- `sections/` intro, related, formulation, method, setup, results (placeholders), discussion, conclusion
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
