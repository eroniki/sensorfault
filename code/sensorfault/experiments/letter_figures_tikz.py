"""TikZ/PGFPlots figures of the RA-L letter (papers/sensorfault/main.tex), in the style of figstyle.tex.

Writes into <paper>/figures/letter/:
  fig_teaser.tex     3D schematic (figure*): a depth error repeated in every frame makes the depth-only surfel
                     map confidently wrong; the per-source record flags it and keeps an honest band
  fig_overview.tex   method overview: rendering, depth and photometric residuals, two records on one placement,
                     honest information, score test, fusion; tags mark the experiment stage (0, A, B, C)
  fig_stage_a.tex    PRELIMINARY Stage A on MatrixCity block 3: s_z per depth model against the H1 bar,
                     repeated share rho_hat against its permutation null, and the overconfidence law
  fig_detection.tex  analytic prediction: tension against depth consensus for repeated and partial faults
  stage_a_snapshot.json  the numbers fig_stage_a.tex was drawn from, so the figure rebuilds without the runs

Stage A numbers are read from results/stage_a/<run>/summary.json (experiments/stage_a_honesty.py). When
that folder is absent (it is on the Stage A branch until that work merges) the snapshot is used instead.
Standalone check PDFs and PNG previews go to results/letter_figures/ when pdflatex is available.

Run from code/sensorfault:
  python3 experiments/letter_figures_tikz.py ../../papers/sensorfault [--stage-a results/stage_a]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

LIBS = "\\usetikzlibrary{calc,arrows.meta,positioning,patterns,decorations.pathreplacing,3d,backgrounds,fit,shapes.geometric}"

# runs shown in panel (a) of the Stage A figure: folder, label, converged alignment
STAGE_A_RUNS = [
    ("block3_depth_pro_s20", "Depth Pro", True),
    ("block3_depth_pro_s10", "Depth Pro, step 10", True),
    ("block3_metric3d_v2_s20", "Metric3D v2", True),
    ("block3_unidepth_v2_s20", "UniDepthV2", True),
    ("block3_depth_anything_v2_s20", "DAv2", False),
    ("block3_control_independent", "exact + indep.", True),
    ("block3_control_repeated", "exact + rep. ($\\varrho{=}0.3$)", False),
]
LAW_RUN = "block3_depth_pro_s20"
# representative n of each n bin of the law table (geometric center of the bin; 17+ taken as 17-32)
N_BIN_CENTRE = {
    "1": 1.0,
    "2": 2.0,
    "3-4": math.sqrt(12),
    "5-8": math.sqrt(40),
    "9-16": 12.0,
    "17+": math.sqrt(17 * 32),
}


def f(x: float, p: int = 4) -> str:
    return f"{x:.{p}f}"


# ------------------------------------------------------------------------------------------ teaser


def teaser() -> str:
    """3D schematic, three panels (figure*): frames over two buildings; every frame's depth places the roof
    of the larger one 2.5 m too low, the smaller one is placed right. (a) the scene and the frames; (b) the
    depth-only surfel map colored by its standardized error, with the stated 2-sigma slab; (c) the
    per-source map colored by the energy between depth prior and pixels, with the honest slab."""
    rng = np.random.default_rng(3)
    n, sig, rho = 12, 1.0, 0.65  # frames, single-frame sd [m], repeated share
    # buildings: x0, x1, y0, y1, true height [m], repeated depth error [m]
    blds = [(14.0, 30.0, 12.0, 26.0, 12.0, -2.5), (36.0, 44.0, 4.0, 12.0, 7.0, 0.0)]
    s, gap = 2.0, 0.3  # surfel spacing and gap [m]
    vd = sig**2 * (rho + (1 - rho) / n)  # honest prior variance
    stated = sig / np.sqrt(n)  # sd the summed depth record states
    vi_roof, vi_ground = 0.7**2, 0.8**2

    def cell(x, y, z0, bias, vi):
        ind = sig * np.sqrt(1 - rho) / np.sqrt(n) * rng.normal()
        depth = z0 + bias + 0.1 * rng.normal() + ind  # depth-only fused placement
        pix = z0 + np.sqrt(vi) * 0.3 * rng.normal()  # pixel placement
        t2 = (depth - pix) ** 2 / (vd + vi)
        ours = pix if t2 > 3.84 else (depth / vd + pix / vi) / (1 / vd + 1 / vi)
        return (x, y, depth, (depth - z0) / stated, ours, 0.5 + 0.5 * t2)

    ground, roofs = [], [[] for _ in blds]
    for x in np.arange(s / 2, 48, s):
        for y in np.arange(s / 2, 34, s):
            inside = [k for k, b in enumerate(blds) if b[0] < x < b[1] and b[2] < y < b[3]]
            if inside:
                b = blds[inside[0]]
                roofs[inside[0]].append(cell(x, y, b[4], b[5], vi_roof))
            else:
                ground.append(cell(x, y, 0.0, 0.0, vi_ground))

    def table(cs, zkey, ckey, clip):
        h = s / 2 - gap / 2
        rows = []
        for c in cs:
            x, y, z, m = c[0], c[1], c[zkey], float(np.clip(c[ckey], -clip, clip))
            rows += [
                f"{x + dx:.2f} {y + dy:.2f} {z:.2f} {m:.2f}"
                for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))
            ]
        return "x y z c\n" + "\n".join(rows) + "\n"

    ps = "patch, patch type=rectangle, shader=flat, z buffer=sort, point meta=explicit, draw=black!35, line width=0.08pt"

    def plot(cs, zkey, ckey, cmap):
        if cmap == "grey":
            opt, clip = ps + ", colormap={gr}{gray(0cm)=(0.66); gray(1cm)=(0.66)}", 0
        elif cmap == "z":
            opt, clip = ps + ", colormap name=sfrdbu, point meta min=-8, point meta max=8", 8
        else:
            opt, clip = ps + ", colormap name=sfviridis, point meta min=0, point meta max=6", 6
        return f"\\addplot3[{opt}] table[meta=c] {{{table(cs, zkey, ckey, clip)}}};"

    def quad(pts, style):
        return (
            rf"\draw[{style}] "
            + " -- ".join(f"(axis cs:{a:.2f},{b:.2f},{c:.2f})" for a, b, c in pts)
            + " -- cycle;"
        )

    def walls(b, zt):
        x0, x1, y0, y1 = b[:4]
        st = "fill=black!7, draw=black!45, line width=0.3pt"
        # back faces first, then the faces toward the viewer (view from -x, -y)
        return "\n".join(
            [
                quad([(x1, y0, 0), (x1, y1, 0), (x1, y1, zt), (x1, y0, zt)], st),
                quad([(x0, y1, 0), (x1, y1, 0), (x1, y1, zt), (x0, y1, zt)], st),
                quad(
                    [(x0, y0, 0), (x1, y0, 0), (x1, y0, zt), (x0, y0, zt)], st + ", fill=black!12"
                ),
                quad([(x0, y0, 0), (x0, y1, 0), (x0, y1, zt), (x0, y0, zt)], st + ", fill=black!4"),
            ]
        )

    def slab(b, zc, half, col, op):
        x0, x1, y0, y1 = b[:4]
        z0, z1 = zc - half, zc + half
        st = f"fill={col}, fill opacity={op}, draw={col}, draw opacity=0.9, line width=0.4pt"
        return "\n".join(
            [
                quad([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], st),
                quad([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], st),
                quad(
                    [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                    f"draw={col}, line width=0.4pt",
                ),
            ]
        )

    def true_roof(b):
        x0, x1, y0, y1, h = b[:5]
        return quad(
            [(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)],
            "sforange, line width=1.0pt, densely dashed",
        )

    def scene(zkey, ckey, cmap, honest):
        out = [plot(ground, zkey, ckey, cmap)]
        for k in (0, 1):  # far building first
            b = blds[k]
            roof_z = float(np.median([c[zkey] for c in roofs[k]]))
            out.append(walls(b, roof_z))
            out.append(plot(roofs[k], zkey, ckey, cmap))
            if honest is not None:
                half = 2 * (np.sqrt(vi_roof) if honest else stated)
                out.append(
                    slab(b, roof_z, half, "sfred" if honest else "black", 0.16 if honest else 0.3)
                )
            out.append(true_roof(b))
        return "\n".join(out)

    # frames: one strip at 30 m height, looking down and forward (oblique, as in the survey)
    cams = [(4.0 + 10.0 * k, 4.0, 34.0) for k in range(5)]

    def frustum(cx, cy, cz):
        c = [
            (cx - 2.0, cy + 1.2, cz - 2.0),
            (cx + 2.0, cy + 1.2, cz - 2.0),
            (cx + 2.0, cy + 3.0, cz - 0.8),
            (cx - 2.0, cy + 3.0, cz - 0.8),
        ]
        st = "black!75, line width=0.4pt"
        out = [
            rf"\draw[{st}] (axis cs:{cx},{cy},{cz}) -- (axis cs:{a:.1f},{b:.1f},{d:.1f});"
            for a, b, d in c
        ]
        out.append(quad(c, st + ", fill=white, fill opacity=0.9"))
        return "\n".join(out)

    b = blds[0]
    targets = [(17.0, 16.0), (20.0, 20.0), (24.0, 22.0), (27.0, 17.0), (26.0, 23.0)]
    zt = b[4] + b[5]
    rays, hits = [], []
    for (cx, cy, cz), (tx, ty) in zip(cams, targets, strict=True):
        lam = (cz - zt) / (
            cz - b[4]
        )  # ray toward the true roof point, stopped at the depth-placed height
        hx, hy = cx + lam * (tx - cx), cy + lam * (ty - cy)
        rays.append(
            rf"\draw[sfred, line width=0.45pt, densely dashed] (axis cs:{cx},{cy},{cz}) -- (axis cs:{hx:.2f},{hy:.2f},{zt:.2f});"
        )
        hits.append(f"({hx:.2f},{hy:.2f},{zt:.2f})")

    common = (
        r"view={-35}{34}, hide axis, scale only axis, width=6.0cm, height=6.0cm, xmin=0, xmax=48, ymin=0, ymax=34, zmin=0,"
        r" clip=false, unit vector ratio*=1 1 1"
    )
    cb = (
        r"colorbar horizontal, colorbar style={{width=3.4cm, height=0.12cm, at={{(0.5,-0.03)}}, anchor=north, {ticks},"
        r" xticklabel style={{font=\sffamily\sansmath\tiny}}, xlabel={{{label}}}, xlabel style={{font=\sffamily\tiny, yshift=3pt}}}}"
    )
    lab = r"\node[font=\sffamily\tiny, align=center, inner sep=1pt]"
    scale_bar = r"\draw[black, line width=0.6pt] (axis cs:30,-1.5,0) -- (axis cs:40,-1.5,0) node[midway, below, font=\sffamily\tiny, inner sep=2pt] {10 m};"
    return rf"""% Teaser, 3D schematic (generated by experiments/letter_figures_tikz.py; analytic, no data)
% n={n} frames, single-frame sd {sig} m, repeated share {rho}; large roof: repeated error {blds[0][5]} m
\begin{{tikzpicture}}
\begin{{axis}}[{common}, zmax=14, name=a]
{scene(2, 3, "grey", None)}
{"".join(frustum(*c) for c in cams)}
{"".join(rays)}
\addplot3[only marks, mark=*, mark size=1.0pt, sfred] coordinates {{{" ".join(hits)}}};
{lab}[anchor=south] at (axis cs:24,4,35.5) {{{n} frames, one depth map each}};
{lab}[anchor=south east, text=sforange!80!black] at (axis cs:14,26,12.3) {{true roof}};
{scale_bar}
\end{{axis}}
\begin{{axis}}[{common}, at={{(a.south east)}}, anchor=south west, xshift=0.1cm, zmax=14, name=b, colormap name=sfrdbu, point meta min=-8, point meta max=8,
  {cb.format(ticks="xtick={-8,-4,0,4,8}", label="standardized error $z$")}]
{scene(2, 3, "z", False)}
\end{{axis}}
\begin{{axis}}[{common}, at={{(b.south east)}}, anchor=south west, xshift=0.1cm, zmax=14, name=c, colormap name=sfviridis, point meta min=0, point meta max=6,
  {cb.format(ticks="xtick={0,2,4,6}, xticklabels={0,2,4,6+}", label="energy $E_q=T_q/2$ [nat]")}]
{scene(4, 5, "E", True)}
\end{{axis}}
\coordinate (bot) at ([yshift=-1pt]current bounding box.south);
\foreach \p/\t in {{a/{{\sfsub{{a}} the frames agree on a wrong roof}}, b/{{\sfsub{{b}} depth only, stated $2\sigma$}},
  c/{{\sfsub{{c}} per source, honest $2\sigma$}}}}
  \node[font=\sffamily\scriptsize, anchor=north] at (\p.south |- bot) {{\t}};
\end{{tikzpicture}}
"""


# ------------------------------------------------------------------------------------------ overview


def overview() -> str:
    """Rendering of shared surfels gives two residuals, depth (left, black) and photometric (right, red);
    each yields an information record on the entity placement; the records are made honest, tested
    against each other and fused. Stage tags (0, A, B, C) mark which experiment stage measures each box."""
    return r"""% Method overview (generated by experiments/letter_figures_tikz.py)
\begin{tikzpicture}[font=\sffamily\fontsize{6.5}{8.8}\selectfont, >={Stealth[length=3.5pt]},
  box/.style={draw=#1, line width=0.6pt, fill=white, align=flush center, inner sep=2.5pt, minimum height=0.8cm, text width=3.5cm},
  box/.default=black!60, wide/.style={text width=8.25cm, minimum height=0.6cm}, arr/.style={->, line width=0.6pt},
  tag/.style={circle, fill=#1, text=white, font=\sffamily\bfseries\fontsize{5.5}{5.5}\selectfont, inner sep=0pt,
    minimum size=8.5pt}, tag/.default=black!70]
\node[box, wide] (in) at (2.375,0)
  {posed frames: images $\mathbf c_i$, one frozen-network depth map $\hat d_i$ each\\
   shared planar surfels $k$ (the joint space), placement $\boldsymbol\theta_k=(\nu,\tau_1,\tau_2)$};
\node[box, wide, fill=black!4] (r) at (2.375,-0.95)
  {\textbf{train depth-guided 2DGS, freeze; render} through $\mathrm H_{ik}$, weights $w_{uk}$\\
   depth $D_i(u)=\sum_k w_{uk}d_{uk}$,\ color $\hat{\mathbf c}_i(u)=\sum_k w_{uk}\mathbf a_k$};
\node[box] (dres) at (0,-1.95) {\textbf{depth residual}\\
  $r^{D}_i(u)=D_i(u)-a_i\hat d_i(u)-b_i$};
\node[box=sfred] (ires) at (4.75,-1.95) {\textbf{photometric residual}\\
  $\mathbf r^{I}_i(u)=\hat{\mathbf c}_i(u)-\mathbf c_i(u)$};
\node[box] (dr) at (0,-3.0) {\textbf{depth record} $\Lambda^{D}_k$, Gauss--Newton\\
  rows $\mathbf j_{uk}=\partial d_{uk}/\partial\boldsymbol\theta_k$};
\node[box=sfred] (ir) at (4.75,-3.0) {\textbf{image record} $\Lambda^{I}_k$, Gauss--Newton\\
  $\partial\hat{\mathbf c}_i/\partial\boldsymbol\theta_k=\mathbf g_{uk}\,\mathbf j_{uk}$};
\node[box] (dh) at (0,-4.05) {\textbf{cap the depth sensor} at its bias field\\
  $\tilde\Lambda^{D}_k=\bigl(\Sigma^{D}_\beta+(\Lambda^{D}_k)^{-1}\bigr)^{-1}$};
\node[box=sfred] (ih) at (4.75,-4.05) {\textbf{deflate the pixels}: $\hat\varphi$ from\\ frame replicates and injected $\boldsymbol\delta$};
\node[box=sfred, wide, fill=sfred!7] (t) at (2.375,-5.1)
  {\textbf{score test at the depth-placed surface}, segment $q$, fault basis $\mathrm B$: $T_q\sim\chi^2_r$};
\node[box, wide] (o) at (2.375,-6.0)
  {\textbf{fuse}: one Gauss--Newton step, $\Lambda_k=\tilde\Lambda^{D}_k+\hat\varphi^{-1}\Lambda^{I}_k$, depth left out where $T_q$ fires;\\
   pose, scale, offset marginalized; unsupported where the pixels have no stiffness};
\draw[arr] (in) -- (r);
\draw[arr] (r.south -| dres.north) -- (dres);
\draw[arr] (r.south -| ires.north) -- (ires);
\draw[arr] (dres) -- (dr);
\draw[arr] (ires) -- (ir);
\draw[arr] (dr) -- (dh);
\draw[arr] (ir) -- (ih);
\draw[arr] (dh.south) -- (dh.south |- t.north);
\draw[arr] (ih.south) -- (ih.south |- t.north);
\draw[arr] (t) -- (o);
\draw[<->, line width=0.5pt, black!60, densely dashed] (dr.east) -- node[above, inner sep=1pt, font=\sffamily\tiny, align=center] {same\\[-1pt] $\mathbf j_{uk}$} (ir.west);
% stage tags: 0 renderer and image Jacobian; A depth only; B pixels; C both sources
\node[tag] at (r.north west) {0};
\node[tag=sfred] at (ir.north east) {0};
\node[tag] at (dres.north west) {A};
\node[tag] at (dr.north west) {A};
\node[tag] at (dh.north west) {A};
\node[tag=sfred] at (ires.north east) {B};
\node[tag=sfred] at (ih.north east) {B};
\node[tag=sfmaroon] at (t.north west) {C};
\node[tag=sfmaroon] at (o.north west) {C};
\end{tikzpicture}
"""


# ------------------------------------------------------------------------------------------ Stage A (preliminary)


def load_stage_a(stage_a: Path | None, snap: Path) -> dict:
    if stage_a is None or not stage_a.is_dir():
        print(f"Stage A results not found; using the snapshot {snap}")
        return json.loads(snap.read_text())
    runs = []
    for folder, label, conv in STAGE_A_RUNS:
        s = json.loads((stage_a / folder / "summary.json").read_text())
        land = next(r for r in s["H1"]["rows"] if r["group"] == "land")
        sf = s["single_frame"]["s_z_heldout_land"]
        runs.append(
            {
                "folder": folder,
                "label": label,
                "converged": conv and bool(s["alignment"].get("converged_0.1pct", True)),
                "fused_sz": [land["ci"]["s_z"][k] for k in ("estimate", "lo", "hi")],
                "single_sz": [sf[k] for k in ("estimate", "lo", "hi")],
                "a8_pass": bool(s["single_frame"]["pass"]),
            }
        )
    s = json.loads((stage_a / LAW_RUN / "summary.json").read_text())
    rep = s["repetition"]
    bs = [b for b in rep["rho"] if not math.isnan(rep["rho"][b]["estimate"])]
    rho = [[int(b)] + [rep["rho"][b][k] for k in ("estimate", "lo", "hi")] for b in bs]
    null = [[int(b)] + [rep["rho_null"][b][k] for k in ("estimate", "lo", "hi")] for b in bs]
    law = [
        {
            "n_bin": r["n_bin"],
            "R_real": r["R_real"],
            "R_pred": r["R_pred"],
            "R_nominal": r["R_nominal"],
            "ratio": [r["ratio_ci"][k] for k in ("estimate", "lo", "hi")],
        }
        for r in s["H2"]["rows"]
    ]
    rho_law = s["H2"]["rows"][0]["rho"]
    readme = (stage_a / "README.md").read_text()
    commit = readme.split("commit `")[1].split("`")[0] if "commit `" in readme else "unknown"
    out = {
        "source": "results/stage_a (experiments/stage_a_honesty.py)",
        "code_commit": commit,
        "runs": runs,
        "law_run": LAW_RUN,
        "rho_hat": rho,
        "rho_null": null,
        "rho_law": rho_law,
        "law": law,
    }
    snap.write_text(json.dumps(out, indent=1))
    return out


def stage_a(d: dict) -> str:
    runs = d["runs"]
    xt = ",".join(str(i) for i in range(len(runs)))
    xl = ",".join("{" + r["label"] + "}" for r in runs)

    def ebar(key: str, conv: bool) -> str:
        pts = [(i, r[key]) for i, r in enumerate(runs) if r["converged"] == conv]
        dx = 0.17 if key == "fused_sz" else -0.17
        return " ".join(
            f"({i + dx:.2f},{v[0]:.3f}) += (0,{v[2] - v[0]:.3f}) -= (0,{v[0] - v[1]:.3f})"
            for i, v in pts
        )

    fused_c, fused_n = ebar("fused_sz", True), ebar("fused_sz", False)
    single_c, single_n = ebar("single_sz", True), ebar("single_sz", False)
    rho = " ".join(
        f"({b},{e:.3f}) += (0,{hi - e:.3f}) -= (0,{e - lo:.3f})" for b, e, lo, hi in d["rho_hat"]
    )
    nl = " ".join(
        f"({b + 0.12:.2f},{e:.3f}) += (0,{hi - e:.3f}) -= (0,{e - lo:.3f})"
        for b, e, lo, hi in d["rho_null"]
    )
    law = d["law"]
    nx = [N_BIN_CENTRE[r["n_bin"]] for r in law]
    real = " ".join(f"({n:.3f},{r['R_real']:.4f})" for n, r in zip(nx, law, strict=True))
    rho_l = d["rho_law"]
    ns = np.geomspace(1, 32, 40)
    pred = " ".join(f"({n:.3f},{rho_l + (1 - rho_l) / n:.4f})" for n in ns)
    nom = " ".join(f"({n:.3f},{1 / n:.4f})" for n in ns)
    ymax_a = max(max(r["fused_sz"][2], r["single_sz"][2]) for r in runs) * 1.25
    eb = "error bars/.cd, y dir=both, y explicit, error mark options={rotate=90, mark size=1.2pt, line width=0.5pt}, error bar style={line width=0.5pt}"
    return rf"""% Stage A, PRELIMINARY (generated by experiments/letter_figures_tikz.py from {d["source"]}, code {d["code_commit"]})
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep=1.35cm}}, sfaxis, height=3.1cm, scale only axis]
\nextgroupplot[width=6.6cm, ymode=log, ymin=0.7, ymax={ymax_a:.1f}, xmin=-0.6, xmax={len(runs) - 0.4},
  xtick={{{xt}}}, xticklabels={{{xl}}}, xticklabel style={{rotate=28, anchor=north east, inner sep=1pt, font=\sffamily\tiny}},
  ytick={{1,1.25,2,4,8}}, yticklabels={{1,1.25,2,4,8}}, ylabel={{$s_z$ on held-out land}},
  title={{\sfsub{{a}} depth-only map vs one frame}},
  legend style={{at={{(0.99,0.98)}}, anchor=north east}}]
\addplot[sfred, densely dashed, line width=0.6pt, forget plot] coordinates {{(-0.6,1.25) ({len(runs) - 0.4},1.25)}};
\addplot[sfgrey, line width=0.5pt, forget plot] coordinates {{(-0.6,1) ({len(runs) - 0.4},1)}};
\addplot[black, only marks, mark=square*, mark size=1.5pt, {eb}] coordinates {{{fused_c}}};
\addlegendentry{{fused, $n\ge5$}}
\addplot[sforange, only marks, mark=*, mark size=1.5pt, {eb}] coordinates {{{single_c}}};
\addlegendentry{{one frame (A8)}}
\addplot[black, only marks, mark=square, mark size=1.5pt, forget plot, {eb}] coordinates {{{fused_n}}};
\addplot[sforange, only marks, mark=o, mark size=1.5pt, forget plot, {eb}] coordinates {{{single_n}}};
\node[font=\sffamily\tiny, text=sfred, anchor=south east, inner sep=1pt] at (axis cs:{len(runs) - 0.45},1.25) {{H1 bar}};
\nextgroupplot[width=3.6cm, xmin=-0.5, xmax=5.6, ymin=-0.1, ymax=1.2, xtick={{0,1,2,3,5}}, xlabel={{gap $B$ [frames]}},
  ylabel={{repeated share $\hat\varrho$}}, title={{\sfsub{{b}} $\hat\varrho$ vs permutation null}},
  legend style={{at={{(0.02,0.02)}}, anchor=south west}}]
\addplot[sfred, mark=x, mark size=2pt, {eb}] coordinates {{{rho}}};
\addlegendentry{{Depth Pro}}
\addplot[sfgrey, mark=+, mark size=2pt, {eb}] coordinates {{{nl}}};
\addlegendentry{{null}}
\nextgroupplot[width=3.6cm, xmode=log, ymode=log, xmin=0.9, xmax=34, ymin=0.025, ymax=1.5, xtick={{1,2,4,8,16,32}},
  xticklabels={{1,2,4,8,16,32}}, ytick={{0.03,0.1,0.3,1}}, yticklabels={{0.03,0.1,0.3,1}},
  xlabel={{frames fused, $n$}}, ylabel={{fused / one-frame variance}},
  title={{\sfsub{{c}} the law, $\hat\varrho={rho_l:.2f}$}},
  legend style={{at={{(0.02,0.02)}}, anchor=south west}}]
\addplot[sfgrey, densely dashed, line width=0.7pt] coordinates {{{nom}}};
\addlegendentry{{stated $1/n$}}
\addplot[sfred, line width=0.8pt] coordinates {{{pred}}};
\addlegendentry{{$\hat\varrho+(1-\hat\varrho)/n$}}
\addplot[black, only marks, mark=square*, mark size=1.4pt] coordinates {{{real}}};
\addlegendentry{{realized}}
\end{{groupplot}}
\end{{tikzpicture}}
"""


# ------------------------------------------------------------------------------------------ detection (analytic)


def phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def power(lam: np.ndarray, zc: float = 1.959964) -> np.ndarray:
    return phi(lam - zc) + phi(-lam - zc)


def detection(rho: float, n: int = 12) -> str:
    """Detection at 5% false alarms under the Gaussian model of Proposition 1, single placement direction.
    Fault delta (in units of the single-frame sd sigma) in k of n frames. Consensus: frame j against the mean
    of the others, threshold calibrated on the true null (its sd is sigma*sqrt((1-rho) n/(n-1)), the repeated
    part cancels). Tension: depth-placed surface (honest prior variance sigma^2 (rho+(1-rho)/n)) against the
    pixels (information equal to one frame's depth information)."""
    vi = 1.0
    vd = rho + (1 - rho) / n
    sd_cons = math.sqrt((1 - rho) * n / (n - 1))

    def cons(delta, k):
        return power(delta * (1 - (k - 1) / (n - 1)) / sd_cons)

    def tens(delta, k):
        return power(delta * k / n / math.sqrt(vd + vi))

    ks = np.arange(1, n + 1)
    d0 = 3.0
    a_c = " ".join(f"({k / n:.3f},{p:.4f})" for k, p in zip(ks, cons(d0, ks), strict=True))
    a_t = " ".join(f"({k / n:.3f},{p:.4f})" for k, p in zip(ks, tens(d0, ks), strict=True))
    ds = np.linspace(0, 6, 61)
    b_c = " ".join(f"({x:.2f},{p:.4f})" for x, p in zip(ds, cons(ds, n), strict=True))
    b_t = " ".join(f"({x:.2f},{p:.4f})" for x, p in zip(ds, tens(ds, n), strict=True))
    b_c1 = " ".join(f"({x:.2f},{p:.4f})" for x, p in zip(ds, cons(ds, 1), strict=True))
    return rf"""% Detection, analytic prediction (generated by experiments/letter_figures_tikz.py; n={n}, rho={rho:.2f})
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=0.55cm, yticklabels at=edge left}}, sfaxis,
  width=3.3cm, height=2.6cm, scale only axis, ymin=0, ymax=1.02, ytick={{0.05,0.5,1}}, yticklabels={{.05,.5,1}}]
\nextgroupplot[xmin=0, xmax=1.04, xtick={{0.083,0.5,1}}, xticklabels={{$1/n$,0.5,1}}, xlabel={{share of frames with the fault}},
  ylabel={{detection at 5\% FA}}, title={{\sfsub{{a}} fault $3\sigma$}}]
\addplot[sfgrey, line width=0.4pt, forget plot] coordinates {{(0,0.05) (1.04,0.05)}};
\addplot[sfred, mark=x, mark size=1.6pt] coordinates {{{a_t}}};
\addplot[black, mark=+, mark size=1.6pt] coordinates {{{a_c}}};
\nextgroupplot[xmin=0, xmax=6, xtick={{0,2,4,6}}, xlabel={{fault size [$\sigma$]}}, title={{\sfsub{{b}} fault in every frame}},
  legend style={{at={{(0.98,0.12)}}, anchor=south east, font=\sffamily\tiny}}]
\addplot[sfgrey, line width=0.4pt, forget plot] coordinates {{(0,0.05) (6,0.05)}};
\addplot[sfred] coordinates {{{b_t}}};
\addlegendentry{{tension}}
\addplot[black] coordinates {{{b_c}}};
\addlegendentry{{depth consensus}}
\addplot[black, densely dotted] coordinates {{{b_c1}}};
\addlegendentry{{consensus, one frame}}
\end{{groupplot}}
\end{{tikzpicture}}
"""


# ------------------------------------------------------------------------------------------ check build


def compile_check(paper: Path, names: list[str], out: Path) -> None:
    style = (paper / "figures" / "figstyle.tex").read_text()
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        wrap = out / f"{n}_check.tex"
        wrap.write_text(
            "\\documentclass[10pt,border=4pt]{standalone}\n\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots,fillbetween}\n"
            + LIBS
            + "\n\\usepackage{amsmath,amssymb,bm}\n\\usepackage{sansmath}\n\\pgfplotsset{compat=1.16}\n"
            + style
            + "\n\\begin{document}\n\\input{figures/letter/"
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
            cwd=paper,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                n + ": " + "\n".join(x for x in r.stdout.splitlines() if x.startswith("!"))[:900]
            )
        pdf = out / f"{n}_check.pdf"
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        wpt = float([x for x in info.splitlines() if x.startswith("Page size")][0].split()[2])
        print(f"{n}: {wpt - 8:.0f} pt wide (column 252 pt, text 516 pt)")
        subprocess.run(
            ["pdftoppm", "-png", "-r", "200", "-singlefile", str(pdf), str(out / f"{n}_preview")],
            check=True,
        )
        for ext in ("aux", "log", "tex"):
            (out / f"{n}_check.{ext}").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paper", nargs="?", default=str(ROOT.parent.parent / "papers" / "sensorfault"))
    ap.add_argument("--stage-a", default=str(ROOT / "results" / "stage_a"))
    a = ap.parse_args()
    paper = Path(a.paper).resolve()
    d = paper / "figures" / "letter"
    d.mkdir(parents=True, exist_ok=True)
    sa = load_stage_a(Path(a.stage_a).resolve(), d / "stage_a_snapshot.json")
    figs = {
        "fig_teaser": teaser(),
        "fig_overview": overview(),
        "fig_stage_a": stage_a(sa),
        "fig_detection": detection(sa["rho_law"]),
    }
    for n, t in figs.items():
        (d / f"{n}.tex").write_text(t)
    if shutil.which("pdflatex"):
        compile_check(paper, list(figs), ROOT / "results" / "letter_figures")


if __name__ == "__main__":
    main()
