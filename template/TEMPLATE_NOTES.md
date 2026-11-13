# IEEE Access LaTeX template notes

- Source: https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip
  (linked from https://ieeeaccess.ieee.org/authors/preparing-your-article/; HTTP 200, 1,942,796 bytes, Last-Modified 2026-05-27)
- Retrieved: 2026-11-13. Zip folder name: ACCESS_latex_template_20260513 (sample comment: "Adjusted 05/13/2026").
- Version: ieeeaccess.cls carries NO version string (no \ProvidesClass); it is a thin wrapper: `\LoadClass{IEEEtran}`.
  Bundled IEEEtran.cls is "2015/08/26 version V1.8b" (Michael Shell).
- Files in template/ (flat): ieeeaccess.cls, IEEEtran.cls, IEEEtran.bst, spotcolor.sty, logo.png, notaglinelogo.png,
  bullet.png, fig1.png, equation3.png, author1-3.png, access.pdf (official rendered sample), sample.tex (= access.tex),
  and Formata/Times/Giovanni/Helvetica font files (t1-*.pfb/.tfm/.map, t1*.fd). The class loads the .map files
  via \pdfmapfile, so ALL t1-* font files must sit next to the .tex (or be on TEXINPUTS).
  Removed from the zip: access.aux, access.log.
- Licence: only IEEEtran.cls header (LPPL 1.3, "may be freely used, distributed and modified"; modified files must be
  renamed and flagged; retain notices; as-is, no warranty). ieeeaccess.cls, spotcolor.sty and the fonts state no licence
  text in the files (fonts: Formata/Giovanni are commercial typefaces embedded as Type1 by IEEE; redistribution terms
  unstated -> do not publish the template/ folder in a public repo without checking). Keep template/ out of any public release.

## Compile result
Scratch copy, `latexmk -pdf -interaction=nonstopmode sample.tex` (pdflatex): exit 0, 8-page sample.pdf produced.
No errors. Only warnings: `LaTeX Font Warning: Font shape T1/formata/m/sl undefined` (x8) and `T1/times/m/sc undefined`,
then "Some font shapes were not available, defaults substituted" -- harmless (no slanted Formata / Times small caps).
No missing packages. Note sample uses a manual thebibliography, so bibtex was not exercised.

## Front matter (exact commands, in order, after \begin{document})
    \documentclass{ieeeaccess}
    \usepackage{cite}, amsmath,amssymb,amsfonts, algorithmic, graphicx, textcomp, bm   % + the \makeatletter bold math-version block in sample.tex lines 9-20 (copy it verbatim) and \def\BibTeX
    \history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
    \doi{10.1109/ACCESS.2024.0429000}
    \title{...}
    \author{\uppercase{First A. Author}\authorrefmark{1}, \IEEEmembership{Fellow, IEEE}, \uppercase{Second B. Author}\authorrefmark{2}, AND \uppercase{Third C. Author}\authorrefmark{3}, \IEEEmembership{Member, IEEE}}
    \address[1]{Dept, Institution, City, Country (e-mail: ...)}     % one \address[n] per \authorrefmark{n}
    \tfootnote{This work was supported in part by ...}               % funding / unnumbered first-page footnote
    \markboth{Author \headeretal: Short Title}{Author \headeretal: Short Title}   % running heads (two args)
    \corresp{Corresponding author: Name (e-mail: ...).}
    \begin{abstract} ... \end{abstract}
    \begin{keywords} kw1, kw2, ... \end{keywords}                    % NOTE: environment is `keywords`, not IEEEkeywords
    \titlepgskip=-21pt
    \maketitle                                                        % check sample.tex around line 83; sample calls it after titlepgskip
Document ends: `\section*{Acknowledgment}` (unnumbered; sponsor/financial info belongs in \tfootnote instead),
bibliography, biographies, then `\EOD` before `\end{document}`.

## Bibliography
Sample uses a hand-written `\begin{thebibliography}{00}` with `\bibitem{b1}` (IEEE style, `\cite{b1}`; cite package).
For BibTeX use `\bibliographystyle{IEEEtran}` + `\bibliography{refs}` (IEEEtran.bst is bundled); sample.tex has no
\bibliographystyle line -- unverified by compile.

## Figures / tables
- Figure: `\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt){file.png}{\textbf{Caption text.}\label{fig1}}` (custom macro; caption bold, full width of one column). Normal figure env presumably also works (unverified).
- Table: standard `\begin{table}\caption{\textbf{Title}}\label{table} \begin{tabular}{|p{25pt}|...}` ... `\end{table}`; caption ABOVE, bold.
- Equations: standard amsmath equation/align.

## Biographies / photos
`\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author1.png}}]{Name} text \end{IEEEbiography}`;
photo-less variant `\begin{IEEEbiographynophoto}{Name} text \end{IEEEbiographynophoto}` is in the template. Biographies are
placed after the bibliography and before \EOD. Sample says biographies are often omitted for conference papers; for a v0
the IEEEbiographynophoto form (or omitting the biographies entirely) works; whether IEEE Access requires them at
submission is not verified (the template text says "all authors may include biographies").

## Required packages (all found by system TeX Live)
ifpdf, xkeyval, color, hyperref (loaded by class), cite, amsmath/amssymb/amsfonts, algorithmic, graphicx, textcomp, bm;
spotcolor.sty (bundled, needed by class), ltxutil/ltxgrid only in revtex-like branch (not triggered).

## Gotchas
- Class is not drop-in with other style files; keep the \AtBeginDocument bold-math block.
- Sample's template-only text/refs/biographies must be stripped from the manuscript.
- Placeholder DOI/dates in \history, \doi are template values; IEEE fills them at publication.
