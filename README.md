# sensorfault-manuscript

Draft v0 of *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian
Surface*, in the Elsevier `elsarticle` template (preprint first; no pay-to-publish venue). Companion code and plan: `eroniki/sensorfault`.
Design v3: a depth-only view gate for depth-supervised Gaussian splatting, built on surfels whose 10x10 uncertainty
(opacity frozen after a warm-up) renders into depth variance; leave-view-out is a subtraction and one Newton step;
one scene temperature is the only free noise variable; reference design: `eroniki/sensorfault` `docs/design-3dgs-bayes.md`.

**Status: pre-experiment draft.** No experiment has been run. Every result table and figure is a
placeholder marked `TODO` in red; the hypotheses H1-H6 in Sec. 4 are pre-registered *predictions*,
not findings. Do not submit before the TODOs are resolved.

## Build
```
latexmk -pdf main.tex        # output: build/main.pdf (needs TeX Live; uses ./latexmkrc)
```
The class is `elsarticle` (TeX Live) with `elsarticle-num`; the layout is `5p,twocolumn`, and changing the class option to `preprint` gives the single-column preprint format.
The main text (title to Future Work) is limited to 8 pages (currently about 5); the declarations and references follow, then the Supplementary Material (S1--S4, same PDF).
`draft_v0.pdf` is the last committed build.

## Outline
Main text: Abstract, Keywords; 1 Introduction (problem, approach, what is and is not new, contributions); 2 Related Work
(short); 3 Method (setting and faults, surfels with an uncertainty that renders, fusion on rendering and leave-view-out,
the view gate); 4 Experimental Design and Hypotheses (H1-H6, breakdown); 5 Results (status, Figure 2 placeholder);
6 Conclusions, Limitations, Future Work. Then Declarations, References.
Supplementary Material: S1 related work; S2 derivations (ray-surfel Jacobian, information update and leave-view-out,
solution separation = leave-one-out innovation, rendered variance, minimum detectable bias); S3 experimental setup;
S4 planned tables and figures.

## Layout
- `main.tex` front matter, abstract, section includes
- `sections/` intro, related, method, setup, results, conclusion (main text); `supplement.tex` includes `supp_*.tex`
- `refs/references.bib` merged bibliography (`refs/part_*.bib` + `part_*_evidence.md` are the sources and the evidence log)
- `figures/` (empty; placeholders are drawn by `\todofig` in `sections/macros.tex`)

## Citation verification
Each entry in `references.bib` was checked against a Crossref/DOI or arXiv record (publisher pages
were often unreachable by automated fetch); sources are logged per key in `refs/part_*_evidence.md`.
Known limits: initials-only authors for a few older papers; some one-line descriptions come from the
record, not an abstract; `fus_zach08` (Zach 2008) could not be verified and is not cited;
`gs_bayesiangsslam` has a Crossref RA-L record but an arXiv page without a venue; `gs_yogo` is
arXiv-only and only its abstract was read. Re-check all entries before submission.

## Licence note
`elsarticle` is distributed with TeX Live (LPPL). Keep this repository private until the preprint is released.
Original text in `sections/` and `main.tex` is the author's.

## Open items before the preprint
AI-use declaration and the code/data statement (marked TODO in `main.tex`); read YOGO in full; final literature pass;
results, figures and tables from the experiment code; satellite section decision; choice of a subscription (no-APC) Elsevier journal.
