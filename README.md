# sensorfault-manuscript

Draft v0 of *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian
Surface*, in the Elsevier `elsarticle` template (preprint first; no pay-to-publish venue). Companion code and plan: `eroniki/sensorfault`.
The gate is the contribution; the monocular-depth pipeline is the test bed.

**Status: pre-experiment draft.** No experiment has been run. Every result table and figure is a
placeholder marked `TODO` in red; the hypotheses H1-H10 in Sec. 3 are pre-registered *predictions*,
not findings. Do not submit before the TODOs are resolved.

## Build
```
latexmk -pdf main.tex        # output: build/main.pdf (needs TeX Live; uses ./latexmkrc)
```
The class is `elsarticle` (TeX Live) with `elsarticle-num`; the layout is `5p,twocolumn`, and changing the class option to `preprint` gives the single-column preprint format.
The main text (title to Future Work) is 8 pages; the declarations and references follow, then the Supplementary Material (S1--S10, same PDF).
`draft_v0.pdf` is the last committed build.

## Outline
Main text (8 pages): Abstract, Keywords; 1 Introduction (core argument, approach, related work in brief, contributions);
2 Method (notation, measurement model, view-level gate: shared Gaussian surface, statistics, detection theory, admission,
cost, exclusion and alarms, conditional protection level); 3 Experimental Design and Hypotheses (H1-H10, breakdown
scenarios); 4 Results (status); 5 Conclusions, Limitations, Future Work. Then Declarations, References.
Supplementary Material: S1 related work, challenges, baseline study; S2 gate details, test bed, ensemble, cost, MDB table,
leave-one-out proposition; S3 experimental setup in full and hypotheses in full; S4 planned tables and figures;
S5-S8 derivations (pose Jacobian and variance, proofs including the protection level, pairwise gate, detection analysis).

## Layout
- `main.tex` front matter, abstract, section includes
- `sections/` intro, method, setup, results, conclusion (main text); `supplement.tex` includes `supp_*.tex`
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
