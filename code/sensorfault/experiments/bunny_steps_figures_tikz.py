"""Step-by-step figure of one view-gate pass on the Stanford bunny (style of figstyle.tex).

Reads results/bunny/steps.npz/json (experiments/bunny_steps.py) and writes
<manuscript>/figures/main/fig_bunny_steps.tex plus four PNGs (fig_bunny_steps_*.png):
  top row, the shared surface in 3D (axes in metres, viewed from the cameras' side):
   (a) step 1, the posed depth views back-projected (v3 red)
   (b) step 2, the fused surfels, coloured by the sd of their normal offset
   (c) step 3, the share of each surfel's normal-offset information that came from v3
   (d) step 4, the cavity without v3: sd of the normal offset; surfels below the support threshold
       are outlined in orange (the gate inflates their covariance instead of dropping them)
  bottom row, view v3 in image space:
   (e) step 5, the cavity rendered into v3
   (f) observed minus rendered depth, with the +5 cm window dashed
   (g) residual after the shared bias and affine nuisance, standardised
   (h) step 6, region-scan z per tile of the 4x4 grid and the resulting p-value
Run: python3 experiments/bunny_steps_figures_tikz.py ../../papers/sensorfault
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

RES = ROOT / "results" / "bunny"
VIEW = (-12.0, 16.0)  # pgfplots view={az}{el}: from the cameras' side (-y), slightly from the left
VIRIDIS = (
    np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]) / 255
)
RDBU = (
    np.array([(33, 102, 172), (146, 197, 222), (247, 247, 247), (244, 165, 130), (178, 24, 43)])
    / 255
)
W3 = 4.5  # 3D panel width (cm)
WI = 2.3  # image panel width (cm)


def _cmap(stops: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.clip(np.nan_to_num(t, nan=0.0), 0, 1) * (len(stops) - 1)
    i = np.minimum(np.floor(t).astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def png(a: np.ndarray, stops, lo, hi, path: Path, scale: int = 8) -> None:
    from PIL import Image

    rgb = _cmap(stops, (a - lo) / (hi - lo))
    rgb[~np.isfinite(a)] = 1.0
    img = Image.fromarray((255 * rgb).astype(np.uint8))
    img.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST).save(path, optimize=True)


def _view_dir() -> np.ndarray:
    az, el = np.deg2rad(VIEW[0]), np.deg2rad(VIEW[1])
    return np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])


def surfel_fills(
    s: dict, value: np.ndarray, stops: np.ndarray, lo: float, hi: float, edge=None, n_seg: int = 10
) -> str:
    """Each surfel as a filled 1-sigma ellipse in its tangent plane, coloured by ``value`` through
    ``stops`` (same 5-stop maps as figstyle), drawn back to front for the pgfplots viewer; surfels in
    ``edge`` (bool mask) get an orange outline."""
    mu, R, sc = s["mu"], s["R"], s["scale"]
    order = np.argsort(mu @ _view_dir())
    ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    col = _cmap(stops, (value - lo) / (hi - lo))
    out = []
    for k in order:
        ring = [
            mu[k] + sc[k, 0] * np.cos(a) * R[k][:, 0] + sc[k, 1] * np.sin(a) * R[k][:, 1]
            for a in ang
        ]
        path = " -- ".join(f"(axis cs:{p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in ring)
        r, g, b = col[k]
        dr = (
            "draw=sforange, line width=0.25pt"
            if edge is not None and edge[k]
            else "draw=black!40, line width=0.08pt"
        )
        out.append(
            rf"\filldraw[fill={{rgb,1:red,{r:.3f};green,{g:.3f};blue,{b:.3f}}}, {dr}] {path} -- cycle;"
        )
    return "\n".join(out)


def axis3d(
    name: str, at: str, title: str, body: str, cmap: str, lo: float, hi: float, labels: bool
) -> str:
    lab = (
        r"xlabel={$x$ [m]}, ylabel={$y$ [m]}, zlabel={$z$ [m]},"
        if labels
        else r"xlabel={}, ylabel={}, zlabel={},"
    )
    return rf"""\begin{{axis}}[sfaxis, name={name}, {at}, view={{{VIEW[0]:.0f}}}{{{VIEW[1]:.0f}}}, width={W3}cm, height={W3 * 0.95:.2f}cm,
  axis equal image, {lab} xmin=-0.85, xmax=0.85, ymin=-0.65, ymax=0.65, zmin=0, zmax=1.25,
  xtick={{-0.5,0,0.5}}, ytick=\empty, ztick={{0,0.5,1}}, tick label style={{font=\sffamily\sansmath\tiny}},
  label style={{font=\sffamily\tiny}}, xlabel style={{yshift=3pt}}, zlabel style={{yshift=-4pt}},
  title={{{title}}}, title style={{font=\sffamily\tiny, align=center, yshift=-3pt}}, colormap name={cmap},
  point meta min={lo}, point meta max={hi}, clip=false]
{body}
\end{{axis}}"""


def colorbar(at: str, width: float, cmap: str, lo, hi, label: str, ticks: str = "") -> str:
    xt = f", xtick={{{ticks}}}" if ticks else ""
    return rf"""\begin{{axis}}[sfcolorbar={width:.2f}cm, colormap name={cmap}, {at}, anchor=north,
  point meta min={lo}, point meta max={hi}, colorbar style={{xlabel={{{label}}}{xt}}}]
\addplot[draw=none] coordinates {{(0,0)}};
\end{{axis}}"""


def figure(s: dict, f: dict, d_lo: float, d_hi: float) -> str:
    pts, pv = s["points"], s["point_view"]
    rng = np.random.default_rng(0)
    sub = rng.random(len(pts)) < 0.5
    grey = " ".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in pts[sub & (pv != 3)])
    red = " ".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in pts[sub & (pv == 3)])
    a_body = (
        rf"\addplot3[only marks, mark=*, mark size=0.25pt, black!45] coordinates {{{grey}}};"
        + "\n"
        + rf"\addplot3[only marks, mark=*, mark size=0.3pt, sfred] coordinates {{{red}}};"
    )
    sd_hi = 8.0
    reds = np.array([(255, 245, 240), (252, 146, 114), (220, 20, 20)]) / 255
    b_body = surfel_fills(s, s["sd_fused"] * 1e3, VIRIDIS, 0, sd_hi)
    c_body = surfel_fills(s, 100 * s["share_j"], reds, 0, 50)
    d_body = surfel_fills(s, s["sd_cav"] * 1e3, VIRIDIS, 0, sd_hi, edge=~s["keep"])
    z_lim, r_lim = 4.0, 10.0
    hw = s["z"].shape[0]
    fr = np.argwhere(s["fault_mask"])
    (r0, c0), (r1, c1) = fr.min(0), fr.max(0) + 1
    win = rf"\draw[black, dashed, line width=0.6pt] (axis cs:{c0},{r0}) rectangle (axis cs:{c1},{r1});"
    tiles = []
    g = s["z_tile"].shape[0]
    for i in range(g):
        for j in range(g):
            z = s["z_tile"][i, j]
            if np.isfinite(z):
                y = (i + 0.5) * hw / g
                x = (j + 0.5) * hw / g
                tiles.append(
                    rf"\node[font=\sffamily\sansmath\tiny, text={'white' if abs(z) < 2.2 else 'black'}, inner sep=0pt] at (axis cs:{x:.1f},{y:.1f}) {{{z:.1f}}};"
                )
    p3 = f["pvals"]["v3"]
    mant, ex = f"{p3:.1e}".split("e")
    img = lambda name, at, title, file, extra="": (
        rf"""\begin{{axis}}[sfimage, name={name}, {at}, width={WI}cm, height={WI}cm,
  xmin=0, xmax={hw}, ymin=0, ymax={hw}, y dir=reverse, title={{{title}}},
  title style={{font=\sffamily\tiny, align=center, yshift=-3pt}}]
\addplot graphics[xmin=0, xmax={hw}, ymin=0, ymax={hw}] {{\mfigdir/{file}}};
{extra}
\end{{axis}}"""
    )
    gap = 1.3
    t = [
        axis3d(
            "sa",
            "at={(0,0)}",
            r"\sfsub{a} 1: posed depth views,\\back-projected (v3 red)",
            a_body,
            "sfviridis",
            0,
            1,
            True,
        ),
        axis3d(
            "sb",
            f"at={{($(sa.east)+({gap}cm,0)$)}}, anchor=west",
            r"\sfsub{b} 2: shared surface,\\all views fused",
            b_body,
            "sfviridis",
            0,
            sd_hi,
            True,
        ),
        axis3d(
            "sc",
            "at={($(sa.south)+(0,-2.8cm)$)}, anchor=north",
            r"\sfsub{c} 3: share of information\\from v3",
            c_body,
            "sfreds",
            0,
            50,
            True,
        ),
        axis3d(
            "sd",
            f"at={{($(sc.east)+({gap}cm,0)$)}}, anchor=west",
            r"\sfsub{d} 4: cavity, v3 subtracted\\(orange: below support)",
            d_body,
            "sfviridis",
            0,
            sd_hi,
            True,
        ),
    ]
    cb = [
        colorbar(
            "at={($(sb.south)+(0,-0.6cm)$)}",
            W3 * 0.75,
            "sfviridis",
            0,
            sd_hi,
            r"sd of normal offset [mm]",
            "0,4,8",
        ),
        colorbar(
            "at={($(sc.south)+(0,-0.6cm)$)}",
            W3 * 0.75,
            "sfreds",
            0,
            50,
            r"from v3 [\%]",
            "0,25,50",
        ),
        colorbar(
            "at={($(sd.south)+(0,-0.6cm)$)}",
            W3 * 0.75,
            "sfviridis",
            0,
            sd_hi,
            r"sd of normal offset [mm]",
            "0,4,8",
        ),
    ]
    row2_y = "-2.75cm"
    ims = [
        img(
            "ie",
            f"at={{($(sc.south west)+(0.1cm,{row2_y})$)}}, anchor=north west",
            r"\sfsub{e} 5: cavity render\\into v3",
            "fig_bunny_steps_render.png",
        ),
        img(
            "if",
            "at={($(ie.east)+(0.62cm,0)$)}, anchor=west",
            r"\sfsub{f} observed $-$\\rendered depth",
            "fig_bunny_steps_resid.png",
            win,
        ),
        img(
            "ig",
            "at={($(if.east)+(0.62cm,0)$)}, anchor=west",
            r"\sfsub{g} standardised\\residual",
            "fig_bunny_steps_z.png",
            win,
        ),
        img(
            "ih",
            "at={($(ig.east)+(0.62cm,0)$)}, anchor=west",
            r"\sfsub{h} 6: region scan,\\$z$ per tile",
            "fig_bunny_steps_scan.png",
            win + "\n" + "\n".join(tiles),
        ),
    ]
    cb2 = [
        colorbar(
            "at={($(ie.south)+(0,-0.08cm)$)}",
            WI,
            "sfviridis",
            f"{d_lo:.1f}",
            f"{d_hi:.1f}",
            "Depth [m]",
            f"{d_lo:.1f},{d_hi:.1f}",
        ),
        colorbar(
            "at={($(if.south)+(0,-0.08cm)$)}",
            WI,
            "sfrdbu",
            -r_lim,
            r_lim,
            "[cm]",
            f"{-r_lim:.0f},0,{r_lim:.0f}",
        ),
        colorbar(
            "at={($(ig.south)+(0,-0.08cm)$)}",
            WI,
            "sfrdbu",
            -z_lim,
            z_lim,
            r"[$\sigma$]",
            f"{-z_lim:.0f},0,{z_lim:.0f}",
        ),
        colorbar(
            "at={($(ih.south)+(0,-0.08cm)$)}",
            WI,
            "sfrdbu",
            -z_lim,
            z_lim,
            r"$z$ per tile",
            f"{-z_lim:.0f},0,{z_lim:.0f}",
        ),
    ]
    verdict = (
        r"\node[anchor=north east, align=right, font=\sffamily\tiny, text=sfred] at ($(ih.south east)+(0,-1.2cm)$) "
        + rf"{{\textbf{{v3 excluded}}, $p={mant}\times10^{{{int(ex)}}}$ (Holm, $M=6$)}};"
    )
    return (
        "% One view-gate pass on the Stanford bunny, step by step (generated by experiments/bunny_steps_figures_tikz.py)\n"
        + r"\begin{tikzpicture}"
        + "\n"
        + r"\pgfplotsset{colormap={sfreds}{rgb255=(255,245,240) rgb255=(252,146,114) rgb255=(220,20,20)}}"
        + "\n"
        + "\n".join(t + cb + ims + cb2 + [verdict])
        + "\n"
        + r"\end{tikzpicture}"
        + "\n"
    )


def main(manuscript: str) -> None:
    ms = Path(manuscript).resolve()
    d = ms / "figures" / "main"
    d.mkdir(parents=True, exist_ok=True)
    s = dict(np.load(RES / "steps.npz"))
    f = json.loads((RES / "steps.json").read_text())
    dep = s["cav_depth"]
    d_lo = float(np.floor(np.nanmin(dep) * 10) / 10)
    d_hi = float(np.ceil(np.nanmax(dep) * 10) / 10)
    png(dep, VIRIDIS, d_lo, d_hi, d / "fig_bunny_steps_render.png")
    png(100 * s["residual"], RDBU, -10, 10, d / "fig_bunny_steps_resid.png")
    png(s["z"], RDBU, -4, 4, d / "fig_bunny_steps_z.png")
    g = s["z_tile"].shape[0]
    hw = s["z"].shape[0]
    tile = np.kron(s["z_tile"], np.ones((hw // g, hw // g)))
    tile[~np.isfinite(s["z"])] = np.nan
    png(tile, RDBU, -4, 4, d / "fig_bunny_steps_scan.png")
    (d / "fig_bunny_steps.tex").write_text(figure(s, f, d_lo, d_hi))
    if shutil.which("pdflatex"):
        from main_figures_tikz import compile_check

        compile_check(ms, ["fig_bunny_steps"], RES)
    print(json.dumps({"n_unsupported": f["n_unsupported_without_v3"], "p_v3": f["pvals"]["v3"]}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent.parent / "papers" / "sensorfault"))
