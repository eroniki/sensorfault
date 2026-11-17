"""H7 summary, TikZ/PGFPlots figures and tables (style: figstyle.tex) from results/h7.

Writes results/h7/summary.json (criteria (a) and (b) per level) and, into <manuscript>/figures/h7/:
  fig_h7.tex          main text: detection rate (top) and recovered fraction (bottom) vs perturbation size,
                      rotation / translation / combined sweeps; predicted ranges shaded, 80% line
  fig_h7_scatter.tex  supplement: pose error after vs before refinement, per view and component
  tab_h7.tex          main text: per level counts, detection, recovery, induced error, swamping
  *.csv               plotted data
Run: python3 experiments/h7_figures_tikz.py ../sensorfault-manuscript
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/h7"


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def q(x, p):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.percentile(x, p)) if x else float("nan")


def summarise(rows, configs, meta):
    rr, ta = meta["settings"]["recover_rule"], meta["settings"]["detect_rule"]
    out = []
    keys = []
    for r in rows:
        k = (r["sweep"], r["rot_deg"], r["trans_gsd"])
        if k not in keys:
            keys.append(k)
    for sweep, rot, tg in keys:
        R = [r for r in rows if (r["sweep"], r["rot_deg"], r["trans_gsd"]) == (sweep, rot, tg)]
        clean = [r for r in R if r["type"] != "flagged"]
        flg = [r for r in R if r["type"] == "flagged"]
        C = [
            c
            for c in configs
            if (c["sweep"], c["rot_deg"], c["trans_gsd"]) == (sweep, rot, tg)
            and c["type"] != "flagged"
        ]
        comp = []
        if rot > 0:
            comp.append("rot")
        if tg > 0:
            comp.append("trans")
        rec = {c: med([r[f"{c}_recovered"] for r in clean]) for c in comp}
        rec_np = {c: med([r[f"noprior_{c}_recovered"] for r in clean]) for c in comp}
        row = {
            "sweep": sweep,
            "rot_deg": rot,
            "trans_gsd": tg,
            "n_clean": len(clean),
            "n_isolated": sum(r["type"] == "isolated" for r in clean),
            "n_block": sum(r["type"] == "block" for r in clean),
            "detected": float(np.mean([r["flagged"] for r in clean])) if clean else float("nan"),
            "detected_isolated": float(
                np.mean([r["flagged"] for r in clean if r["type"] == "isolated"] or [np.nan])
            ),
            "detected_block": float(
                np.mean([r["flagged"] for r in clean if r["type"] == "block"] or [np.nan])
            ),
            "flagged_view_detected": float(np.mean([r["flagged"] for r in flg]))
            if flg
            else float("nan"),
            "recovered_median": rec,
            "recovered_q25": {c: q([r[f"{c}_recovered"] for r in clean], 25) for c in comp},
            "recovered_q75": {c: q([r[f"{c}_recovered"] for r in clean], 75) for c in comp},
            "recovered_noprior_median": rec_np,
            "rot_err_after_deg_median": med([r["rot_err_after_deg"] for r in clean]),
            "trans_err_after_gsd_median": med([r["trans_err_after_gsd"] for r in clean]),
            "noprior_rot_err_after_deg_median": med(
                [r["noprior_rot_err_after_deg"] for r in clean]
            ),
            "noprior_trans_err_after_gsd_median": med(
                [r["noprior_trans_err_after_gsd"] for r in clean]
            ),
            "flagged_view_recovered_median": {
                c: med([r[f"{c}_recovered"] for r in flg]) for c in comp
            },
            "unperturbed_flagged_per_group_mean": float(
                np.mean([c["unperturbed_flagged"] for c in C])
            )
            if C
            else float("nan"),
            "unperturbed_flagged_per_group_max": int(
                max([c["unperturbed_flagged"] for c in C] or [0])
            ),
            "in_detect_range": sweep != "control"
            and (rot >= ta["rot_deg"] or tg >= ta["trans_gsd"]),
            "in_recover_range": sweep != "control"
            and rot <= rr["rot_deg"]
            and tg <= rr["trans_gsd"],
        }
        row["a_pass"] = (row["detected"] >= 0.8) if row["in_detect_range"] else None
        row["b_pass"] = (min(rec.values()) >= 0.8) if (row["in_recover_range"] and rec) else None
        out.append(row)
    a = [r["a_pass"] for r in out if r["a_pass"] is not None]
    b = [r["b_pass"] for r in out if r["b_pass"] is not None]
    ctrl = [r for r in out if r["sweep"] == "control"]
    return {
        "levels": out,
        "a_holds": all(a),
        "a_levels_passed": f"{sum(a)}/{len(a)}",
        "b_holds": all(b),
        "b_levels_passed": f"{sum(b)}/{len(b)}",
        "b_noprior_levels_passed": f"{sum(min(r['recovered_noprior_median'].values()) >= 0.8 for r in out if r['in_recover_range'] and r['recovered_noprior_median'])}/{len(b)}",
        "control_false_flags": {
            "perturbed_group_views": ctrl[0]["detected"] if ctrl else None,
            "unperturbed_per_group": ctrl[0]["unperturbed_flagged_per_group_mean"]
            if ctrl
            else None,
        },
        "gsd_m": meta["gsd_m"],
        "wallclock_s": meta.get("wallclock_s"),
    }


def write_csvs(S, d: Path):
    for sweep, xk in (
        ("rotation", "rot_deg"),
        ("translation", "trans_gsd"),
        ("combined", "rot_deg"),
    ):
        L = [r for r in S["levels"] if r["sweep"] == sweep]
        comp = {"rotation": ["rot"], "translation": ["trans"], "combined": ["rot", "trans"]}[sweep]
        lines = ["x,det,det_iso,det_blk," + ",".join(f"rec_{c},lo_{c},hi_{c},np_{c}" for c in comp)]
        for r in L:
            vals = [
                r[xk],
                100 * r["detected"],
                100 * r["detected_isolated"],
                100 * r["detected_block"],
            ]
            for c in comp:
                vals += [
                    100 * r["recovered_median"][c],
                    100 * r["recovered_q25"][c],
                    100 * r["recovered_q75"][c],
                    100 * r["recovered_noprior_median"][c],
                ]
            lines.append(",".join(f"{v:.4g}" for v in vals))
        (d / f"h7_{sweep}.csv").write_text("\n".join(lines) + "\n")


def fig_main(S) -> str:
    rr, ta = (1.0, 25), (0.5, 10)
    w, h = 3.55, 2.6
    axes = []
    spec = [
        (
            "rotation",
            r"Rotation [$^{\circ}$]",
            "0.05,0.1,0.25,0.5,1,2",
            "0.04",
            "2.5",
            ta[0],
            rr[0],
            ["rot"],
        ),
        (
            "translation",
            "Translation [GSD]",
            "1,2,5,10,25,50",
            "0.8",
            "62",
            ta[1],
            rr[1],
            ["trans"],
        ),
        (
            "combined",
            r"Rotation [$^{\circ}$] / \textcolor{sfgrey}{translation [GSD]}",
            "0.05,0.1,0.25,0.5,1,2",
            "0.04",
            "2.5",
            ta[0],
            rr[0],
            ["rot", "trans"],
        ),
    ]
    titles = {
        "rotation": "Rotation only",
        "translation": "Translation only",
        "combined": "Both (paired levels)",
    }
    for row in ("det", "rec"):
        for ci, (sweep, xl, ticks, xmin, xmax, tdet, trec, comp) in enumerate(spec):
            o = [
                f"xmode=log, xmin={xmin}, xmax={xmax}, xtick={{{ticks}}}, xticklabels={{{ticks}}}, log ticks with fixed point",
                "ymin=-5, ymax=105, ytick={0,20,40,60,80,100}",
            ]
            if row == "det":
                o.append(f"title={{\\sfsub{{{'abc'[ci]}}} {titles[sweep]}}}, xticklabels={{}}")
                if ci == 0:
                    o.append("ylabel={Flagged [\\%]}")
            else:
                o.append(f"title={{\\sfsub{{{'def'[ci]}}}}}, xlabel={{{xl}}}")
                if ci == 0:
                    o.append("ylabel={Error removed [\\%]}")
            if ci > 0:
                o.append("yticklabels={}")
            if sweep == "combined":
                o.append(
                    "extra x ticks={0.05,0.1,0.25,0.5,1,2}, extra x tick labels={1,2,5,10,25,50}, "
                    "extra x tick style={tick label style={yshift=-7pt, font=\\sffamily\\sansmath\\tiny, text=sfgrey}, major tick length=0pt}"
                    if row == "rec"
                    else ""
                )
            body = [f"\\nextgroupplot[{', '.join(x for x in o if x)}]"]
            if row == "det":
                body.append(
                    f"\\fill[sflightgrey] (axis cs:{tdet},-5) rectangle (axis cs:{xmax},105);"
                )
                body.append(
                    f"\\addplot[sfred, mark=*, mark size=1.4pt] table[x=x, y=det, col sep=comma] {{\\hfigdir/h7_{sweep}.csv}};"
                )
                body.append(
                    f"\\addplot[black, dashed, mark=+, mark size=2pt, mark options={{solid}}] table[x=x, y=det_blk, col sep=comma] {{\\hfigdir/h7_{sweep}.csv}};"
                )
            else:
                body.append(
                    f"\\fill[sflightgrey] (axis cs:{xmin},-5) rectangle (axis cs:{trec},105);"
                )
                styles = {"rot": ("sfred", "*"), "trans": ("sforange", "square*")}
                for c in comp:
                    col, mk = styles[c]
                    body.append(
                        f"\\addplot[{col}, mark={mk}, mark size=1.4pt] table[x=x, y=rec_{c}, col sep=comma] {{\\hfigdir/h7_{sweep}.csv}};"
                    )
                    body.append(
                        f"\\addplot[{col}, dashed, mark={mk}, mark size=1.4pt, mark options={{solid, fill=white}}] table[x=x, y=np_{c}, col sep=comma] {{\\hfigdir/h7_{sweep}.csv}};"
                    )
            body.append(
                f"\\addplot[sfgrey, densely dotted, line width=0.7pt, domain={xmin}:{xmax}, samples=2] {{80}};"
            )
            axes.append("\n".join(body))
    ln = r"\tikz[baseline=-0.6ex]{\draw[%s, line width=0.9pt%s] (0,0) -- (0.4,0); %s}"
    red_o = ln % ("sfred", "", r"\fill[sfred] (0.2,0) circle (1.4pt);")
    blk = ln % (
        "black",
        ", dashed",
        r"\draw[black, line width=0.6pt] (0.2,0) ++(-2pt,0) -- ++(4pt,0) ++(-2pt,-2pt) -- ++(0,4pt);",
    )
    org = ln % (
        "sforange",
        "",
        r"\fill[sforange] (0.2,0) ++(-1.4pt,-1.4pt) rectangle ++(2.8pt,2.8pt);",
    )
    dash = ln % (
        "sfred",
        ", dashed",
        r"\filldraw[sfred, fill=white, line width=0.6pt] (0.2,0) circle (1.4pt);",
    )
    rng = r"\tikz[baseline=-0.6ex]{\fill[sflightgrey] (0,-0.08) rectangle (0.4,0.12);}"
    tgt = (
        r"\tikz[baseline=-0.6ex]{\draw[sfgrey, densely dotted, line width=0.9pt] (0,0) -- (0.4,0);}"
    )
    legend = (
        r"\node[anchor=north, draw=sfgrey, rounded corners=1.5pt, inner sep=2pt, font=\sffamily\tiny] at ($(group c2r2.south)+(0,-1.45cm)$) {"
        r"\begin{tabular}{@{}l@{\hspace{1.6em}}l@{}}"
        r"\textbf{(a--c)} " + red_o + r" all perturbed clean views & " + blk + r" blocks of three\\"
        r"\textbf{(d--f)} "
        + red_o
        + r" rotation, "
        + org
        + r" translation, as designed & "
        + dash
        + r" exploratory: the frame's GNSS/IMU prior dropped\\"
        + rng
        + r" predicted range & "
        + tgt
        + r" 80\% target"
        r"\end{tabular}};"
    )
    return (
        r"""% H7 pose perturbation check (generated by experiments/h7_figures_tikz.py)
\begin{tikzpicture}
\begin{groupplot}[sfaxis, group style={group size=3 by 2, horizontal sep=0.22cm, vertical sep=0.95cm}, scale only axis,
  width="""
        + f"{w}cm, height={h}cm"
        + r""", title style={font=\sffamily\scriptsize, yshift=-3pt}]
"""
        + "\n".join(axes)
        + "\n\\end{groupplot}\n"
        + legend
        + "\n\\end{tikzpicture}\n"
    )


def fig_scatter(rows) -> str:
    """Error after refinement vs before, per view and component (as designed and exploratory)."""
    d = SRC / "scatter"
    d.mkdir(exist_ok=True)
    lines = {"rot": ["b,a,np"], "trans": ["b,a,np"]}
    for r in rows:
        if r["type"] == "flagged" or r["sweep"] == "control":
            continue
        if r["rot_err_before_deg"] > 1e-9:
            lines["rot"].append(
                f"{r['rot_err_before_deg']:.4g},{max(r['rot_err_after_deg'], 1e-3):.4g},{max(r['noprior_rot_err_after_deg'], 1e-3):.4g}"
            )
        if r["trans_err_before_gsd"] > 1e-9:
            lines["trans"].append(
                f"{r['trans_err_before_gsd']:.4g},{max(r['trans_err_after_gsd'], 1e-2):.4g},{max(r['noprior_trans_err_after_gsd'], 1e-2):.4g}"
            )
    return lines


def fig_scatter_tex() -> str:
    w = 4.9
    ax = []
    for ci, (c, lab, lo, hi) in enumerate(
        (("rot", r"$^{\circ}$", 0.005, 4), ("trans", "GSD", 0.1, 100))
    ):
        o = (
            f"xmode=log, ymode=log, xmin={lo}, xmax={hi}, ymin={lo}, ymax={hi}, xlabel={{Error before [{lab}]}}, "
            f"ylabel={{Error after [{lab}]}}, title={{\\sfsub{{{'ab'[ci]}}} {'Rotation' if c == 'rot' else 'Translation'}}}"
        )
        if ci == 0:
            o += ", legend style={at={(0.03,0.97)}, anchor=north west}"
        ax.append(
            f"\\nextgroupplot[{o}]\n"
            f"\\addplot[black, line width=0.5pt, domain={lo}:{hi}, samples=2] {{x}};\n"
            f"\\addplot[sfgrey, densely dotted, line width=0.7pt, domain={lo}:{hi}, samples=2] {{0.2*x}};\n"
            f"\\addplot[only marks, mark=x, mark size=1.6pt, sfred, line width=0.6pt] table[x=b, y=a, col sep=comma] {{\\hfigdir/h7_scatter_{c}.csv}};\n"
            f"\\addplot[only marks, mark=+, mark size=1.6pt, black, line width=0.5pt] table[x=b, y=np, col sep=comma] {{\\hfigdir/h7_scatter_{c}.csv}};"
            + (
                "\n\\legend{No change, 80\\% removed, As designed, Exploratory: prior dropped}"
                if ci == 0
                else ""
            )
        )
    return (
        r"""% H7 errors before and after refinement (generated by experiments/h7_figures_tikz.py)
\begin{tikzpicture}
\begin{groupplot}[sfaxis, group style={group size=2 by 1, horizontal sep=1.75cm}, scale only axis, width="""
        + f"{w}cm, height={w}cm]\n"
        + "\n".join(ax)
        + "\n\\end{groupplot}\n\\end{tikzpicture}\n"
    )


def table(S) -> str:
    def pct(v):
        if v is None or not np.isfinite(v):
            return "--"
        return (
            f"{100 * v:.1f}" if 79.5 <= 100 * v < 80.5 else f"{100 * v:.0f}"
        )  # do not round across the 80% target

    rows = []
    last = None
    for r in S["levels"]:
        if last is not None and r["sweep"] != last:
            rows.append(r"\midrule")
        last = r["sweep"]
        name = {
            "control": "Control",
            "rotation": "Rotation",
            "translation": "Translation",
            "combined": "Both",
        }[r["sweep"]]
        size = (
            (f"{r['rot_deg']:g}$^{{\\circ}}$" if r["rot_deg"] > 0 else "")
            + (" + " if r["rot_deg"] > 0 and r["trans_gsd"] > 0 else "")
            + (f"{r['trans_gsd']:g} GSD" if r["trans_gsd"] > 0 else "")
        )
        size = size or "0"
        rec = "/".join(pct(v) for v in r["recovered_median"].values()) or "--"
        recn = "/".join(pct(v) for v in r["recovered_noprior_median"].values()) or "--"
        da = r"\,\textsuperscript{a}" if r["in_detect_range"] else ""
        db = r"\,\textsuperscript{b}" if r["in_recover_range"] else ""
        rows.append(
            f"{name} & {size} & {r['n_clean']} & {pct(r['detected'])}{da} & {rec}{db} & {recn} & "
            f"{r['rot_err_after_deg_median']:.3f} & {r['trans_err_after_gsd_median']:.1f} & "
            f"{r['unperturbed_flagged_per_group_mean']:.1f} \\\\"
        )
    return (
        "% generated by experiments/h7_figures_tikz.py\n\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
        "Sweep & Size & $n$ & Flagged & Removed & Removed, & Rot.\\ after & Transl.\\ after & Others \\\\\n"
        " & & & [\\%] & [\\%] & expl.\\ [\\%] & [$^{\\circ}$] & [GSD] & flagged \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def compile_check(ms: Path, names, out: Path):
    style = (ms / "figures" / "figstyle.tex").read_text()
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        wrap = out / f"{n}_check.tex"
        wrap.write_text(
            "\\documentclass[12pt,border=4pt]{standalone}\n\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots}\n"
            "\\usetikzlibrary{calc,arrows.meta,positioning}\n\\usepackage{sansmath}\n\\pgfplotsset{compat=1.16}\n"
            + style
            + "\n\\newcommand{\\hfigdir}{figures/h7}\n\\begin{document}\n\\input{figures/h7/"
            + n
            + ".tex}\n\\end{document}\n"
        )
        r = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={out}",
                str(wrap),
            ],
            cwd=ms,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                n + ": " + "\n".join(x for x in r.stdout.splitlines() if x.startswith("!"))[:900]
            )
        info = subprocess.run(
            ["pdfinfo", str(out / f"{n}_check.pdf")], capture_output=True, text=True
        ).stdout
        wpt = float([x for x in info.splitlines() if x.startswith("Page size")][0].split()[2])
        print(f"{n}: {wpt - 8:.0f} pt content width (text width 390 pt)")
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                "150",
                "-singlefile",
                str(out / f"{n}_check.pdf"),
                str(out / f"{n}_preview"),
            ],
            check=True,
        )
        for ext in ("aux", "log", "tex"):
            (out / f"{n}_check.{ext}").unlink(missing_ok=True)


def main(manuscript: str) -> None:
    ms = Path(manuscript).resolve()
    d = ms / "figures" / "h7"
    d.mkdir(parents=True, exist_ok=True)
    rows = json.loads((SRC / "rows.json").read_text())
    configs = json.loads((SRC / "configs.json").read_text())
    meta = json.loads((SRC / "meta.json").read_text())
    S = summarise(rows, configs, meta)
    (SRC / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_csvs(S, d)
    sc = fig_scatter(rows)
    for c, L in sc.items():
        (d / f"h7_scatter_{c}.csv").write_text("\n".join(L) + "\n")
    (d / "fig_h7.tex").write_text(fig_main(S))
    (d / "fig_h7_scatter.tex").write_text(fig_scatter_tex())
    (d / "tab_h7.tex").write_text(table(S))
    for f in d.glob("*.csv"):
        shutil.copy(f, SRC / f.name)
    if shutil.which("pdflatex"):
        compile_check(ms, ["fig_h7", "fig_h7_scatter"], SRC / "check")
    print(json.dumps({k: v for k, v in S.items() if k != "levels"}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "sensorfault-manuscript"))
