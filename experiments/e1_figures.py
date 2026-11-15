"""Comparison figures and a summary table from the saved E1 results (no recomputation).

Reads results/e1/{clean,grid,repeat}.json and writes to results/e1/figures/:
  *.csv       plotted data
  fig_*.tex   pgfplots figures (input-able; standalone wrappers compile them to PDF)
  fig_*.pdf   compiled figures (if pdflatex is available)
  e1_summary_table.{csv,tex}
Run: python experiments/e1_figures.py results/e1
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

N_VIEWS = 185


def load(root: Path):
    C = json.loads((root / "clean.json").read_text())
    G = json.loads((root / "grid.json").read_text())
    P = json.loads((root / "repeat.json").read_text())
    return C, G, P


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fig_grid(G, out: Path):
    g = G["single_view_grid"]
    deltas = sorted({r["delta"] for r in g})
    areas = sorted({r["area"] for r in g})
    rows = []
    for di, d in enumerate(deltas):
        for ai, a in enumerate(areas):
            c = [r for r in g if r["delta"] == d and r["area"] == a]
            gate = 100 * np.mean([r["gate_detect_matched"] for r in c])
            rms = 100 * np.mean([r["rms_detect_matched"] for r in c])
            rows.append([ai, di, a, d, len(c), f"{gate:.0f}", f"{rms:.0f}"])
    write_csv(
        out / "e1_grid_detection.csv", ["ai", "di", "area", "delta", "n", "gate", "rms"], rows
    )
    xt = ",".join(str(a) for a in areas)
    yt = ",".join(str(d) for d in deltas)
    panels = []
    for col, title in (("gate", "Gate"), ("rms", "Aligned RMS")):
        panels.append(
            rf"""\nextgroupplot[title={{{title}}}]
\addplot[matrix plot*, point meta=explicit, mesh/cols={len(areas)}, nodes near coords={{\pgfmathprintnumber{{\pgfplotspointmeta}}}}, nodes near coords style={{font=\scriptsize, anchor=center}}]
  table[x=ai, y=di, meta={col}, col sep=comma]{{\efigdir/e1_grid_detection.csv}};"""
        )
    tex = rf"""% E1: detection (%) at matched 5% per-view false alarm, one faulty view, 10 seeds per cell
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=1.6cm}},
  width=5.6cm, height=5.0cm, enlargelimits=false,
  colormap={{det}}{{rgb255=(255,255,255) rgb255=(33,102,172)}}, point meta min=0, point meta max=100,
  xtick={{0,...,{len(areas) - 1}}}, xticklabels={{{xt}}}, ytick={{0,...,{len(deltas) - 1}}}, yticklabels={{{yt}}},
  xlabel={{area fraction}}, ylabel={{magnitude ($\sigma$)}}, title style={{font=\small}}, tick label style={{font=\scriptsize}}, label style={{font=\small}}]
{chr(10).join(panels)}
\end{{groupplot}}
\end{{tikzpicture}}
"""
    (out / "fig_e1_grid.tex").write_text(tex)


def fig_clean(C, out: Path):
    rows = []
    labels = []
    for scale in ("1.0", "1.3"):
        for temp in (True, False):
            rr = [r for r in C["clean"][scale] if r["temperature"] == temp]
            fa = 100 * np.mean([r["per_view_fa"] for r in rr])
            fw = sum(r["loop_excluded"] > 0 for r in rr)
            lo, hi = wilson(fw, len(rr))
            lab = f"{scale}x, {'T on' if temp else 'T off'}"
            labels.append(lab)
            rows.append(
                [
                    len(labels) - 1,
                    lab.replace(", ", " / "),
                    f"{fa:.1f}",
                    f"{100 * fw / len(rr):.0f}",
                    f"{lo:.1f}",
                    f"{hi:.1f}",
                    fw,
                    len(rr),
                ]
            )
    write_csv(
        out / "e1_clean_false_alarms.csv",
        ["i", "label", "per_view_fa", "loop_any_pct", "loop_lo", "loop_hi", "loop_k", "n"],
        rows,
    )
    xl = ",".join(
        "{" + lab.replace("T on", "temp.\\ on").replace("T off", "temp.\\ off") + "}"
        for lab in labels
    )
    tex = rf"""% E1: clean scenes, 20 seeds per condition (noise at the calibrated level and 1.3x)
\begin{{tikzpicture}}
\begin{{axis}}[ybar, bar width=9pt, width=10cm, height=5.2cm, ymin=0, ymax=125, enlarge x limits=0.18,
  xtick=data, xticklabels={{{xl}}}, tick label style={{font=\scriptsize}}, ylabel={{\%}},
  legend style={{font=\scriptsize, at={{(0.02,0.98)}}, anchor=north west}}, nodes near coords, nodes near coords style={{font=\tiny}},
  every node near coord/.append style={{/pgf/number format/precision=1}}]
\addplot[fill=gray!40] table[x=i, y=per_view_fa, col sep=comma]{{\efigdir/e1_clean_false_alarms.csv}};
\addplot[fill=blue!50] table[x=i, y=loop_any_pct, col sep=comma]{{\efigdir/e1_clean_false_alarms.csv}};
\legend{{single test: views flagged at $p<0.05$, loop with Holm: seeds with any exclusion}}
\end{{axis}}
\end{{tikzpicture}}
"""
    (out / "fig_e1_clean.tex").write_text(tex)


def fig_repeat(P, out: Path):
    rp = P["repeated"]
    confs = [
        ("B=3, loop", 3, True, 10, True),
        ("B=1, loop", 1, True, 10, True),
        ("B=3, single pass", 3, True, 1, True),
        ("B=1, single pass", 1, True, 1, True),
        ("B=3, loop, no temp.", 3, False, 10, True),
        ("B=3, loop, unwhitened", 3, True, 10, False),
    ]
    rows = []
    for scale in (1.0, 1.3):
        for i, (lab, B, temp, cap, wh) in enumerate(confs):
            rr = [
                r
                for r in rp
                if (r["noise_scale"], r["B"], r["temperature"], r["round_cap"], r["whiten"])
                == (scale, B, temp, cap, wh)
            ]
            if not rr:
                continue
            det = 100 * np.mean([r["detected"] / len(r["faulty"]) for r in rr])
            fa = 100 * np.mean([r["false"] / (N_VIEWS - len(r["faulty"])) for r in rr])
            rows.append([scale, i, lab.replace(", ", " / "), f"{det:.1f}", f"{fa:.1f}", len(rr)])
    write_csv(
        out / "e1_repeated_ablations.csv",
        ["noise_scale", "i", "label", "detected_pct", "clean_excluded_pct", "n"],
        rows,
    )
    for scale in ("1.0", "1.3"):
        sub = [r for r in rows if str(r[0]) == scale]
        write_csv(
            out / f"e1_repeated_ablations_{scale}.csv",
            ["i", "label", "detected_pct", "clean_excluded_pct"],
            [[r[1], r[2], r[3], r[4]] for r in sub],
        )
    xl = ",".join(
        "{" + c[0].replace(", ", "\\\\").replace("no temp.", "no temp.") + "}" for c in confs
    )
    panels = []
    for scale in ("1.0", "1.3"):
        panels.append(
            rf"""\nextgroupplot[title={{noise {scale}$\times$ calibrated}}]
\addplot[fill=blue!50] table[x=i, y=detected_pct, col sep=comma]{{\efigdir/e1_repeated_ablations_{scale}.csv}};
\addplot[fill=red!45] table[x=i, y=clean_excluded_pct, col sep=comma]{{\efigdir/e1_repeated_ablations_{scale}.csv}};"""
        )
    tex = rf"""% E1: same 8-sigma error on three neighbouring frames of a strip, 10 seeds per bar
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=1 by 2, vertical sep=2.5cm}}, ybar, /pgf/bar width=7pt,
  width=12cm, height=4.6cm, ymin=0, ymax=105, enlarge x limits=0.1, xtick={{0,...,{len(confs) - 1}}},
  xticklabels={{{xl}}}, xticklabel style={{align=center}}, tick label style={{font=\scriptsize}}, ylabel={{\%}}, title style={{font=\small}},
  nodes near coords, nodes near coords style={{font=\tiny}}, every node near coord/.append style={{/pgf/number format/precision=1}},
  legend style={{font=\scriptsize, at={{(0.5,-0.42)}}, anchor=north, legend columns=2}}, ymax=115]
{chr(10).join(panels)}
\legend{{faulty frames excluded, clean views excluded}}
\end{{groupplot}}
\end{{tikzpicture}}
"""
    (out / "fig_e1_repeated.tex").write_text(tex)


def fig_runtime(C, G, P, out: Path):
    # gate passes (one leave-out test of every active view) per stage, from the saved outcomes
    clean_passes = sum(
        1 + min(r["loop_excluded"] + 1, 10) for rows in C["clean"].values() for r in rows
    )
    grid_passes = sum(1 + r["rounds"] for r in G["single_view_grid"])
    rep_passes = sum(r["rounds"] for r in P["repeated"])
    stages = [
        ("clean", C["runtime_s"], clean_passes, 4),
        ("grid", G["runtime_s"], grid_passes, 2),
        ("repeated", P["runtime_s"], rep_passes, 2),
    ]
    rows = [
        [i, s, f"{t / 60:.1f}", p, f"{t / p:.1f}", th] for i, (s, t, p, th) in enumerate(stages)
    ]
    write_csv(
        out / "e1_runtime.csv",
        ["i", "stage", "minutes", "gate_passes", "s_per_pass", "threads"],
        rows,
    )
    labels = ",".join("{" + f"{r[1]}\\\\{r[5]} threads\\\\{r[4]} s per pass" + "}" for r in rows)
    tex = r"""% E1: wall-clock per stage (CPU only), threads, and seconds per gate pass (one cavity test of every active view)
\begin{tikzpicture}
\begin{axis}[ybar, bar width=12pt, width=7.5cm, height=5cm, ymin=0, ymax=70, xtick=data,
  xticklabels={LABELS}, xticklabel style={align=center}, ylabel={minutes}, tick label style={font=\scriptsize},
  nodes near coords, nodes near coords style={font=\tiny}, enlarge x limits=0.25]
\addplot[fill=gray!50] table[x=i, y=minutes, col sep=comma]{\efigdir/e1_runtime.csv};
\end{axis}
\end{tikzpicture}
""".replace("LABELS", labels)
    (out / "fig_e1_runtime.tex").write_text(tex)
    return stages, rows


def summary_table(C, G, P, runtime_rows, out: Path):
    g = G["single_view_grid"]
    rp = P["repeated"]

    def rep(scale, B, temp, cap, wh=True):
        rr = [
            r
            for r in rp
            if (r["noise_scale"], r["B"], r["temperature"], r["round_cap"], r["whiten"])
            == (scale, B, temp, cap, wh)
        ]
        det = 100 * np.mean([r["detected"] / len(r["faulty"]) for r in rr])
        fa = 100 * np.mean([r["false"] / (N_VIEWS - len(r["faulty"])) for r in rr])
        return f"{det:.1f}", f"{fa:.1f}"

    def clean(scale, temp):
        rr = [r for r in C["clean"][scale] if r["temperature"] == temp]
        return (
            f"{100 * np.mean([r['per_view_fa'] for r in rr]):.1f}",
            f"{sum(r['loop_excluded'] > 0 for r in rr)}/{len(rr)}",
        )

    rows = [
        [
            "Clean, nominal noise",
            "temperature on",
            "per-view FA (single test) " + clean("1.0", True)[0] + "%",
            "loop: " + clean("1.0", True)[1] + " seeds with an exclusion",
        ],
        [
            "Clean, nominal noise",
            "temperature off",
            "per-view FA " + clean("1.0", False)[0] + "%",
            "loop: " + clean("1.0", False)[1],
        ],
        [
            "Clean, noise 1.3x",
            "temperature on",
            "per-view FA " + clean("1.3", True)[0] + "%",
            "loop: " + clean("1.3", True)[1],
        ],
        [
            "Clean, noise 1.3x",
            "temperature off",
            "per-view FA " + clean("1.3", False)[0] + "%",
            "loop: " + clean("1.3", False)[1],
        ],
        [
            "One faulty view (160 scenes)",
            "gate vs aligned RMS, matched 5% FA",
            f"gate {sum(r['gate_detect_matched'] for r in g)}/160",
            f"aligned RMS {sum(r['rms_detect_matched'] for r in g)}/160",
        ],
        [
            "One faulty view (160 scenes)",
            "loop, Holm",
            f"detected {sum(r['loop_detect'] for r in g)}/160",
            f"scenes with a clean exclusion {sum(r['loop_false'] > 0 for r in g)}/160",
        ],
    ]
    for scale in (1.0, 1.3):
        for lab, B, temp, cap, wh in (
            ("B=3, loop", 3, True, 10, True),
            ("B=1, loop", 1, True, 10, True),
            ("B=3, single pass", 3, True, 1, True),
            ("B=1, single pass", 1, True, 1, True),
            ("B=3, loop, no temperature", 3, False, 10, True),
            ("B=3, loop, unwhitened", 3, True, 10, False),
        ):
            d, f = rep(scale, B, temp, cap, wh)
            rows.append(
                [
                    f"3 neighbouring faulty frames, noise {scale}x",
                    lab,
                    f"detected {d}%",
                    f"clean excluded {f}%",
                ]
            )
    for r in runtime_rows:
        rows.append(
            [
                "Runtime",
                f"stage {r[1]} ({r[5]} threads)",
                f"{r[2]} min",
                f"{r[4]} s per gate pass ({r[3]} passes)",
            ]
        )
    write_csv(out / "e1_summary_table.csv", ["scenario", "variant", "outcome_1", "outcome_2"], rows)
    tex = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}p{2.8cm}>{\raggedright\arraybackslash}p{3.6cm}}",
        r"\toprule",
        r"Scenario & Variant & \multicolumn{2}{l}{Outcome} \\",
        r"\midrule",
    ]
    prev = None
    for sc, var, o1, o2 in rows:
        s = sc if sc != prev else ""
        prev = sc
        esc = lambda x: (
            x.replace("%", r"\%")
            .replace("x,", r"$\times$,")
            .replace("1.3x", r"1.3$\times$")
            .replace("1.0x", r"1.0$\times$")
        )
        tex.append(f"{esc(s)} & {esc(var)} & {esc(o1)} & {esc(o2)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "e1_summary_table.tex").write_text("\n".join(tex) + "\n")


def compile_pdfs(out: Path):
    if not shutil.which("pdflatex"):
        return
    for f in sorted(out.glob("fig_*.tex")):
        wrap = out / f"_standalone_{f.stem}.tex"
        wrap.write_text(
            "\\documentclass[border=4pt]{standalone}\n\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots}\n"
            "\\pgfplotsset{compat=1.16}\n\\newcommand{\\efigdir}{.}\n\\begin{document}\n"
            f"\\input{{{f.name}}}\n\\end{{document}}\n"
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", wrap.name],
            cwd=out,
            check=True,
            capture_output=True,
        )
        (out / f"{wrap.stem}.pdf").rename(out / f"{f.stem}.pdf")
        for ext in ("aux", "log", "tex"):
            (out / f"{wrap.stem}.{ext}").unlink(missing_ok=True)


def main(path: str) -> None:
    root = Path(path)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    C, G, P = load(root)
    fig_grid(G, out)
    fig_clean(C, out)
    fig_repeat(P, out)
    _, rt_rows = fig_runtime(C, G, P, out)
    summary_table(C, G, P, rt_rows, out)
    compile_pdfs(out)
    print("\n".join(sorted(str(p) for p in out.iterdir())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/e1")
