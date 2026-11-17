"""TikZ/PGFPlots versions of all E1 figures, in the shared style of experiments/figstyle.tex.

Uses only saved outputs: the CSVs of experiments/e1_figures.py (results/e1/figures) and the PNG panels
and meta.json of experiments/e1_qualitative.py (results/e1/qualitative). Writes fig_*.tex next to them
and, if pdflatex is available, standalone PDFs. Raster maps are embedded images inside PGFPlots axes,
with PGFPlots colorbars (colormaps defined in figstyle.tex match the PNG colormaps).
Run: python experiments/e1_figures_tikz.py results/e1
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, header, rows):
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------
# comparison figures (results/e1/figures)
# ------------------------------------------------------------------------------------------------


def grid(fig: Path) -> str:
    """Grouped bars in the style of the clean-scene figure: detection by magnitude, one bar per area."""
    rows = read_csv(fig / "e1_grid_detection.csv")
    areas = sorted({float(r["area"]) for r in rows})
    deltas = sorted({int(float(r["delta"])) for r in rows})
    val = {
        (m, int(float(r["delta"])), float(r["area"])): float(r[m])
        for r in rows
        for m in ("gate", "rms")
    }
    out = []
    for i, d in enumerate(deltas):
        row = [i, d]
        for m in ("gate", "rms"):
            for a in areas:
                v = val[(m, d, a)]
                row += [f"{v:g}", "" if v >= 99.5 else f"{v:.0f}"]
        out.append(row)
    head = ["x", "delta"] + [
        f"{m}{k}{s}" for m in ("gate", "rms") for k in range(len(areas)) for s in ("", "lab")
    ]
    write_csv(fig / "e1_grid_bars.csv", head, out)
    shades = ["sfgrey!45", "sfgrey", "black!65", "black"]
    panels = []
    for sub, m, name in (("a", "gate", "Gate (this paper)"), ("b", "rms", "Aligned RMS")):
        plots = ("\\addlegendimage{empty legend}\n" if sub == "b" else "") + "\n".join(
            rf"\addplot[fill={c}, draw=black, point meta=explicit symbolic, nodes near coords] table[x=x, y={m}{k}, meta={m}{k}lab, col sep=comma] {{\efigdir/e1_grid_bars.csv}};"
            for k, c in enumerate(shades)
        )
        leg = (
            r"\legend{Fault area (share of the view):, "
            + ", ".join(f"{100 * a:g}\\%" for a in areas)
            + "}"
            if sub == "b"
            else ""
        )
        yax = ", ylabel={Detected [\\%]}" if sub == "a" else ", yticklabels={}"
        panels.append(
            rf"""\nextgroupplot[title={{\sfsub{{{sub}}} {name}}}{yax}]
{plots}
{leg}"""
        )
    xtl = ",".join(f"${d}\\sigma$" for d in deltas)
    return rf"""% E1, one faulty view: detection [%] at matched 5% per-view false alarm (10 seeds per cell)
\begin{{tikzpicture}}
\begin{{groupplot}}[sfaxis, sfbar, /pgf/bar width=5.5pt, group style={{group size=2 by 1, horizontal sep=0.35cm}},
  width=5.6cm, height=3.0cm, scale only axis, ymin=0, ymax=118, ytick={{0,25,50,75,100}}, xmin=-0.55, xmax={len(deltas) - 0.45},
  xtick={{0,...,{len(deltas) - 1}}}, xticklabels={{{xtl}}}, xlabel={{Fault magnitude}},
  every node near coord/.append style={{font=\sffamily\sansmath\tiny, anchor=south, inner sep=1pt}},
  legend style={{at={{(-0.03,-0.36)}}, anchor=north, legend columns=5}}]
{chr(10).join(panels)}
\end{{groupplot}}
\end{{tikzpicture}}
"""


def clean(fig: Path) -> str:
    rows = read_csv(fig / "e1_clean_false_alarms.csv")
    by = {r["label"]: r for r in rows}

    def g(noise, on, key):
        lab = f"{noise}x / {'T on' if on else 'T off'}"
        return float(by[lab][key])

    out = []
    for i, noise in enumerate(("1.0", "1.3")):
        out.append(
            [
                i,
                f"{noise}",
                g(noise, True, "per_view_fa"),
                g(noise, False, "per_view_fa"),
                g(noise, True, "loop_any_pct"),
                g(noise, False, "loop_any_pct"),
            ]
        )
    write_csv(
        fig / "e1_clean_by_noise.csv", ["x", "noise", "fa_on", "fa_off", "loop_on", "loop_off"], out
    )
    panels = []
    for sub, on, off, ylab in (
        ("a", "fa_on", "fa_off", "Views flagged, single test [\\%]"),
        ("b", "loop_on", "loop_off", "Seeds with an exclusion [\\%]"),
    ):
        panels.append(
            rf"""\nextgroupplot[ylabel={{{ylab}}}, title={{\sfsub{{{sub}}}}}]
\addplot[sfbarred] table[x=x, y={on}, col sep=comma] {{\efigdir/e1_clean_by_noise.csv}};
\addplot[sfbarblack] table[x=x, y={off}, col sep=comma] {{\efigdir/e1_clean_by_noise.csv}};"""
            + (r"\legend{Temperature on, Temperature off}" if sub == "a" else "")
        )
    return rf"""% E1, clean scenes (20 seeds per condition): single-test false alarms and Holm-loop exclusions
\begin{{tikzpicture}}
\begin{{groupplot}}[sfaxis, sfbar, /pgf/bar width=9pt, group style={{group size=2 by 1, horizontal sep=1.6cm}},
  width=4.6cm, height=3.6cm, scale only axis, ymin=0, ymax=112, xmin=-0.6, xmax=1.6,
  xtick={{0,1}}, xticklabels={{Nominal noise, Noise $1.3\times$}},
  nodes near coords, every node near coord/.append style={{font=\sffamily\sansmath\tiny, /pgf/number format/fixed,
    /pgf/number format/precision=1}},
  legend style={{at={{(0.03,0.97)}}, anchor=north west}}]
{chr(10).join(panels)}
\end{{groupplot}}
\end{{tikzpicture}}
"""


def repeated(fig: Path) -> str:
    rows = read_csv(fig / "e1_repeated_ablations.csv")
    labs = []
    for r in rows:
        if r["label"] not in labs:
            labs.append(r["label"])
    out = []
    for i, lab in enumerate(labs):
        d = {r["noise_scale"]: r for r in rows if r["label"] == lab}
        out.append(
            [
                i,
                d["1.0"]["detected_pct"],
                d["1.3"]["detected_pct"],
                d["1.0"]["clean_excluded_pct"],
                d["1.3"]["clean_excluded_pct"],
            ]
        )
    write_csv(fig / "e1_repeated_by_noise.csv", ["x", "det_10", "det_13", "fa_10", "fa_13"], out)
    names = {
        "B=3 / loop": r"$B{=}3$, loop (the gate)",
        "B=1 / loop": r"$B{=}1$, loop",
        "B=3 / single pass": r"$B{=}3$, single pass",
        "B=1 / single pass": r"$B{=}1$, single pass",
        "B=3 / loop / no temp.": r"$B{=}3$, loop, no temperature",
        "B=3 / loop / unwhitened": r"$B{=}3$, loop, not whitened",
    }
    ticks = ",".join("{" + names.get(lab, lab) + "}" for lab in labs[::-1])
    panels = []
    for sub, a, b, title, xmax in (
        ("a", "det_10", "det_13", "Faulty frames excluded [\\%]", 122),
        ("b", "fa_10", "fa_13", "Clean views excluded [\\%]", 12.2),
    ):
        yax = rf", yticklabels={{{ticks}}}" if sub == "a" else ", yticklabels={}"
        leg = r"\legend{Nominal noise, Noise $1.3\times$}" if sub == "b" else ""
        panels.append(
            rf"""\nextgroupplot[title={{\sfsub{{{sub}}} {title}}}, xmax={xmax}{yax}]
\addplot[sfbarblack, bar shift=2.6pt] table[y expr={len(labs) - 1}-\thisrow{{x}}, x={a}, col sep=comma] {{\efigdir/e1_repeated_by_noise.csv}};
\addplot[sfbargrey, bar shift=-2.6pt] table[y expr={len(labs) - 1}-\thisrow{{x}}, x={b}, col sep=comma] {{\efigdir/e1_repeated_by_noise.csv}};
{leg}"""
        )
    return rf"""% E1, the same 8-sigma error on three neighbouring frames (10 seeds per bar)
\begin{{tikzpicture}}
\begin{{groupplot}}[sfaxis, xbar, every axis plot/.append style={{line width=0.4pt}}, point meta=x, /pgf/bar width=4.5pt,
  legend image code/.code={{\draw[#1, line width=0.4pt] (0cm,-0.07cm) rectangle (0.22cm,0.09cm);}}, group style={{group size=2 by 1, horizontal sep=0.35cm}},
  width=3.9cm, height=4.4cm, scale only axis, xmin=0, ymin=-0.6, ymax={len(labs) - 0.4},
  ytick={{0,...,{len(labs) - 1}}}, ytick style={{draw=none}},
  nodes near coords, every node near coord/.append style={{font=\sffamily\sansmath\tiny, /pgf/number format/fixed,
    /pgf/number format/precision=1, anchor=west, inner sep=1.5pt}},
  legend style={{at={{(0.97,0.03)}}, anchor=south east}}]
{chr(10).join(panels)}
\end{{groupplot}}
\end{{tikzpicture}}
"""


def runtime(fig: Path) -> str:
    rows = read_csv(fig / "e1_runtime.csv")
    ticks = ",".join(
        "{"
        + f"{r['stage'].capitalize()}\\\\{r['threads']} threads\\\\{r['s_per_pass']}\\,s per pass"
        + "}"
        for r in rows
    )
    return rf"""% E1 runtime: CPU wall-clock per stage; seconds per gate pass (one cavity test of every active view)
\begin{{tikzpicture}}
\begin{{axis}}[sfaxis, sfbar, /pgf/bar width=14pt, width=6.0cm, height=3.4cm, scale only axis, ymin=0, ymax=68,
  xtick=data, xticklabels={{{ticks}}}, xticklabel style={{align=center}}, ylabel={{Wall-clock [min]}},
  enlarge x limits=0.25, nodes near coords, every node near coord/.append style={{font=\sffamily\sansmath\tiny}}]
\addplot[sfbargrey] table[x=i, y=minutes, col sep=comma] {{\efigdir/e1_runtime.csv}};
\end{{axis}}
\end{{tikzpicture}}
"""


# ------------------------------------------------------------------------------------------------
# qualitative figures (results/e1/qualitative)
# ------------------------------------------------------------------------------------------------

PW, PH = 2.3, 2.3 * 54 / 96  # panel size [cm] for a 96x54 view


def image_grid(rows, cols, colorbars, row_labels, col_titles, w=PW, h=PH, hsep=0.12, vsep=0.12):
    """rows x cols of embedded PNGs in PGFPlots frames, titles, row labels, one colorbar per column."""
    nrow, ncol = len(rows), len(cols)
    body = []
    for ri, rtag in enumerate(rows):
        for ci, ctag in enumerate(cols):
            opts = []
            if ri == 0:
                opts.append(f"title={{{col_titles[ci]}}}")
            if ci == 0:
                opts.append(f"ylabel={{{row_labels[ri]}}}")
            body.append(
                rf"\nextgroupplot[{', '.join(opts)}]"
                + "\n"
                + rf"\addplot graphics[xmin=0, xmax=1, ymin=0, ymax=1] {{\qfigdir/{rtag}_{ctag}.png}};"
            )
    cbs = []
    for ci, (cmap, lo, hi, lab) in enumerate(colorbars):
        cbs.append(
            rf"""\begin{{axis}}[sfcolorbar={w}cm, colormap name={cmap}, at={{($(group c{ci + 1}r{nrow}.south)+(0,-0.12cm)$)}},
  anchor=north, point meta min={lo}, point meta max={hi}, colorbar style={{xlabel={{{lab}}}}}]
\addplot[draw=none] coordinates {{(0,0)}};
\end{{axis}}"""
        )
    return rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[sfimage, group style={{group size={ncol} by {nrow}, horizontal sep={hsep}cm, vertical sep={vsep}cm}},
  width={w}cm, height={h}cm, xmin=0, xmax=1, ymin=0, ymax=1,
  ylabel style={{align=center, font=\sffamily\tiny}}, title style={{font=\sffamily\scriptsize, yshift=-2pt}}]
{chr(10).join(body)}
\end{{groupplot}}
{chr(10).join(cbs)}
\end{{tikzpicture}}
"""


def qual_views(q: Path) -> str:
    m = json.loads((q / "meta.json").read_text())
    V, sc = m["views"], m["scales"]
    lab = {
        "clean": "Clean",
        "faulty": "Faulty",
        "swamped": "Swamped\\\\neighbour",
        "swamped_after": "Neighbour,\\\\faulty dropped",
    }
    rows = ["clean", "faulty", "swamped", "swamped_after"]
    labels = [f"{lab[t]}\\\\view {V[t]['view']}" for t in rows]
    titles = ["True depth", "Cavity render", "Raw residual", "Standardised", "Region scan $|z|$"]
    cbs = [
        ("sfviridis", f"{sc['depth_m'][0]:.0f}", f"{sc['depth_m'][1]:.0f}", "Depth [m]"),
        ("sfviridis", f"{sc['depth_m'][0]:.0f}", f"{sc['depth_m'][1]:.0f}", "Depth [m]"),
        ("sfrdbu", f"{-sc['raw_residual_m']:.1f}", f"{sc['raw_residual_m']:.1f}", "Residual [m]"),
        ("sfrdbu", "-5", "5", "Residual [$\\sigma$]"),
        ("sfviridis", "0", "20", "$|z|$ per tile ($\\geq$20 clipped)"),
    ]
    return (
        "% E1 qualitative check of the test (generated by experiments/e1_figures_tikz.py)\n"
        + image_grid(rows, ["gt", "render", "raw", "std", "scan"], cbs, labels, titles)
    )


def qual_uncertainty(q: Path) -> str:
    m = json.loads((q / "meta.json").read_text())
    U, sc = m["uncertainty"], m["scales"]
    rows = ["unc_weak", "unc_strong"]
    labels = [
        f"Weak support\\\\view {U['weak']['view']}, {100 * U['weak']['testable']:.0f}\\% testable",
        f"Strong support\\\\view {U['strong']['view']}, {100 * U['strong']['testable']:.0f}\\% testable",
    ]
    titles = [
        "$\\sqrt{(\\Sigma_r)_{uu}}$",
        "Surface part $\\sqrt{V_{-j}(u)}$",
        "Tile variance: full / diagonal",
    ]
    cbs = [
        ("sfviridis", "0", f"{sc['sd_m']:.1f}", "Standard deviation [m]"),
        ("sfviridis", "0", f"{sc['surface_sd_m']:.1f}", "Standard deviation [m]"),
        ("sfviridis", "1", f"{sc['tile_ratio_max']:.1f}", "Variance ratio [--]"),
    ]
    return (
        "% E1 residual uncertainty and the within-surfel term (generated by experiments/e1_figures_tikz.py)\n"
        + image_grid(rows, ["sd", "surf", "ratio"], cbs, labels, titles, w=3.6, h=3.6 * 54 / 96)
    )


def qual_topdown(q: Path) -> str:
    m = json.loads((q / "meta.json").read_text())
    T = m["topdown"]
    x0, x1, y0, y1 = T["extent_m"]
    panels = []
    import math

    slo, shi = T["offset_sd_m"]
    logticks = [t for t in (0.3, 1, 3, 10, 30) if slo <= t <= shi * 1.0001]
    logopts = (
        ", xtick={"
        + ",".join(f"{math.log10(t):.4f}" for t in logticks)
        + "}, xticklabels={"
        + ",".join(f"{t:g}" for t in logticks)
        + "}"
    )
    cbars = []
    for ci, (sub, img, lo, hi, lab, cmapopts) in enumerate(
        (
            ("a", "top_height.png", T["height_m"][0], T["height_m"][1], "Surfel height [m]", ""),
            (
                "b",
                "top_sd.png",
                math.log10(slo),
                math.log10(shi),
                "Normal-offset s.d. [m], log scale",
                logopts,
            ),
        )
    ):
        panels.append(
            rf"""\nextgroupplot[title={{\sfsub{{{sub}}}}}]
\addplot[forget plot] graphics[xmin={x0}, xmax={x1}, ymin={y0}, ymax={y1}] {{\qfigdir/{img}}};
\addplot[only marks, mark=*, mark size=0.7pt, black] table[x=x, y=y, col sep=comma] {{\qfigdir/cameras.csv}};
\addplot[only marks, mark=x, mark size=2.4pt, sfred, line width=1pt, nodes near coords, point meta=explicit symbolic,
  every node near coord/.append style={{font=\sffamily\tiny, text=sfred, anchor=west, xshift=1pt, fill=white, fill opacity=0.8,
    text opacity=1, inner sep=0.8pt}}] table[x=x, y=y, meta=label, col sep=comma] {{\qfigdir/tested.csv}};"""
            + (r"\legend{Camera centre, Tested view}" if sub == "b" else "")
        )
        cbars.append(
            rf"""\begin{{axis}}[sfcolorbar=4.9cm, at={{($(group c{ci + 1}r1.south)+(0,-1.05cm)$)}}, anchor=north,
  point meta min={lo:.4g}, point meta max={hi:.4g}, colorbar style={{xlabel={{{lab}}}{cmapopts}}}]
\addplot[draw=none] coordinates {{(0,0)}};
\end{{axis}}"""
        )
    return rf"""% E1 top-down surfel map of MatrixCity block 3 (generated by experiments/e1_figures_tikz.py)
\begin{{tikzpicture}}
\begin{{groupplot}}[sfaxis, group style={{group size=2 by 1, horizontal sep=1.6cm}}, width=4.9cm, height=4.9cm,
  scale only axis, axis equal image, enlargelimits=false, axis on top, xlabel={{X [m]}}, ylabel={{Y [m]}},
  legend style={{at={{(0.98,0.98)}}, anchor=north east}}]
{chr(10).join(panels)}
\end{{groupplot}}
{chr(10).join(cbars)}
\end{{tikzpicture}}
"""


def compile_pdfs(d: Path, figs: list[str], efig: str, qfig: str) -> None:
    if not shutil.which("pdflatex"):
        return
    style = (HERE / "figstyle.tex").read_text()
    for f in figs:
        wrap = d / f"_sa_{f}.tex"
        wrap.write_text(
            "\\documentclass[border=4pt]{standalone}\n\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots}\n"
            "\\usetikzlibrary{calc}\n\\usepackage{sansmath}\n\\pgfplotsset{compat=1.16}\n"
            + style
            + f"\n\\newcommand{{\\efigdir}}{{{efig}}}\n"
            f"\\newcommand{{\\qfigdir}}{{{qfig}}}\n\\begin{{document}}\n\\input{{{f}.tex}}\n\\end{{document}}\n"
        )
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", wrap.name],
            cwd=d,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"{f}: "
                + "\n".join(line for line in r.stdout.splitlines() if line.startswith("!"))[:600]
            )
        (d / f"{wrap.stem}.pdf").rename(d / f"{f}.pdf")
        for ext in ("aux", "log", "tex"):
            (d / f"{wrap.stem}.{ext}").unlink(missing_ok=True)


def main(root: str) -> None:
    root = Path(root)
    fig, q = root / "figures", root / "qualitative"
    for name, fn in (
        ("fig_e1_grid", grid),
        ("fig_e1_clean", clean),
        ("fig_e1_repeated", repeated),
        ("fig_e1_runtime", runtime),
    ):
        (fig / f"{name}.tex").write_text(fn(fig))
    for name, fn in (
        ("fig_e1_qual_views", qual_views),
        ("fig_e1_qual_uncertainty", qual_uncertainty),
        ("fig_e1_qual_topdown", qual_topdown),
    ):
        (q / f"{name}.tex").write_text(fn(q))
    compile_pdfs(
        fig,
        ["fig_e1_grid", "fig_e1_clean", "fig_e1_repeated", "fig_e1_runtime"],
        ".",
        "../qualitative",
    )
    compile_pdfs(
        q,
        ["fig_e1_qual_views", "fig_e1_qual_uncertainty", "fig_e1_qual_topdown"],
        "../figures",
        ".",
    )
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/e1")
