"""Manuscript figure of the Stanford bunny check (non-planar scene), in the style of figstyle.tex.

Writes <manuscript>/figures/main/fig_bunny.tex and fig_bunny_v3_depth.png from results/bunny (made by
experiments/bunny_gate.py):
  (a) the scene: decimated bunny (flat-shaded patches) and the six cameras, axes in metres
  (b) depth of the faulty view v3 with the +5 cm window outlined
  (c) per-view -log10 p, fault on v3 vs clean (seed 0, the golden-test cases), first Holm threshold
  (d) the same for the two known failures: clean seed 1 and a fault on the end-of-arc view v5
Compiles a standalone check PDF + PNG preview into results/bunny/ when pdflatex is available.
Run: python3 experiments/bunny_figures_tikz.py ../../papers/sensorfault
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
sys.path.insert(0, str(ROOT / "src"))

RES = ROOT / "results" / "bunny"
VIEWER = np.array([-15.0, 28.0])  # pgfplots view={azimuth}{elevation}
VIRIDIS = (
    np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]) / 255
)


def _viridis(t: np.ndarray) -> np.ndarray:
    """figstyle's 5-stop sfviridis, linearly interpolated (so the PNG matches the colour bar)."""
    t = np.clip(t, 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(np.floor(t).astype(int), len(VIRIDIS) - 2)
    f = (t - i)[..., None]
    return VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f


def depth_png(npz: dict, view: int, path: Path, lo: float, hi: float, scale: int = 8) -> None:
    from PIL import Image

    d = npz["depth"][view].astype(float)
    rgb = _viridis((d - lo) / (hi - lo))
    rgb[d <= 0] = 1.0  # background: white
    img = Image.fromarray((255 * rgb).astype(np.uint8))
    img.resize((d.shape[1] * scale, d.shape[0] * scale), Image.NEAREST).save(path, optimize=True)


def _view_dir(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector from the scene towards the pgfplots viewer."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])


def scene_panel(facts: dict) -> str:
    """Flat-shaded bunny (faces sorted back to front for the viewer) and the six camera frusta."""
    from sensorfault.sim.bunny import bunny_mesh
    from sensorfault.sim.mesh import decimate_vertex_cluster, face_normals

    mesh = decimate_vertex_cluster(bunny_mesh(), 0.13)  # ~1k faces keep the .tex small
    tri = mesh.vertices[mesh.faces]
    n = face_normals(mesh)
    vd = _view_dir(*VIEWER)
    n = np.where((n @ vd)[:, None] < 0, -n, n)
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    order = np.argsort(tri.mean(1) @ vd)  # far first
    pts = []
    for k in order:
        for p in tri[k]:
            pts.append(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) [{shade[k]:.2f}]")
    patches = "\n".join(" ".join(pts[i : i + 6]) for i in range(0, len(pts), 6))

    cams = []
    npz = np.load(RES / "views_faulty_v3_seed0.npz")
    T, K = npz["T_wc"], npz["K"]
    hw = facts["hw"]
    s = 0.55  # frustum depth drawn [m]
    for i, Tw in enumerate(T):
        c = Tw[:3, 3]
        corners = []
        for u, v in ((0, 0), (hw - 1, 0), (hw - 1, hw - 1), (0, hw - 1)):
            ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
            corners.append(c + s * Tw[:3, :3] @ ray)
        col = "sfred" if i == 3 else "black"
        lw = "0.7pt" if i == 3 else "0.45pt"
        ring = " ".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in [*corners, corners[0]])
        spokes = "\n".join(
            rf"\addplot3[{col}, line width={lw}, forget plot] coordinates {{({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})}};"
            for p in corners
        )
        cams.append(
            rf"\addplot3[{col}, line width={lw}, forget plot] coordinates {{{ring}}};"
            + "\n"
            + spokes
            + "\n"
            + rf"\node[font=\sffamily\tiny, text={col}, inner sep=1pt, anchor=south] at (axis cs:{c[0]:.3f},{c[1]:.3f},{c[2] + 0.2:.3f}) {{v{i}}};"
        )
    return rf"""\begin{{axis}}[sfaxis, name=scene, view={{{VIEWER[0]:.0f}}}{{{VIEWER[1]:.0f}}}, width=7.6cm, height=5.6cm,
  axis equal image, xlabel={{$x$ [m]}}, ylabel={{$y$ [m]}}, zlabel={{$z$ [m]}},
  xmin=-1.6, xmax=1.6, ymin=-3.2, ymax=0.8, zmin=0, zmax=1.6, xtick={{-1,0,1}}, ytick={{-3,-2,-1,0}}, ztick={{0,1}},
  tick label style={{font=\sffamily\sansmath\tiny}}, label style={{font=\sffamily\tiny}},
  xlabel style={{sloped}}, ylabel style={{sloped}},
  colormap={{sfgreys}}{{gray(0)=(0.35) gray(1)=(0.97)}}, point meta min=0, point meta max=1,
  title={{\sfsub{{a}} Scene: bunny and six cameras}}, clip=false]
\addplot3[patch, patch type=triangle, shader=flat, point meta=explicit, draw=black!35, line width=0.05pt, forget plot]
  coordinates {{
{patches}
}};
{chr(10).join(cams)}
\end{{axis}}"""


def bars(
    case_a: dict,
    case_b: dict,
    lab_a: str,
    lab_b: str,
    thr: float,
    title: str,
    name: str,
    at: str,
    legend: str = "north west",
) -> str:
    views = list(case_a["neg_log10p"])
    xs = ",".join(views)
    a = " ".join(f"({v},{case_a['neg_log10p'][v]:.3f})" for v in views)
    b = " ".join(f"({v},{case_b['neg_log10p'][v]:.3f})" for v in views)
    ymax = max(16.0, max([*case_a["neg_log10p"].values(), *case_b["neg_log10p"].values()]) * 1.08)
    marks = []
    for case, shift in ((case_a, "-3.2pt"), (case_b, "3.2pt")):
        for v in case["excluded"]:
            y = case["neg_log10p"][v]
            col = "sfred" if case is case_b else "black"
            marks.append(
                rf"\node[font=\sffamily\tiny, text={col}, anchor=south, inner sep=1pt, xshift={shift}] at (axis cs:{v},{min(y, ymax * 0.93):.2f}) {{excl.}};"
            )
    return rf"""\begin{{axis}}[sfaxis, sfbar, name={name}, {at}, width=6.3cm, height=3.9cm, symbolic x coords={{{xs}}}, xtick=data,
  ymin=0, ymax={ymax:.1f}, ylabel={{$-\log_{{10}} p$}}, xlabel={{View}}, enlarge x limits=0.12,
  legend pos={legend}, legend columns=1, title={{{title}}}, clip=false]
\addplot[sfbarblack] coordinates {{{a}}};
\addplot[sfbarred] coordinates {{{b}}};
\draw[sforange, dashed, line width=0.7pt] ({{rel axis cs:0,0}}|-{{axis cs:{views[0]},{thr:.3f}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:{views[0]},{thr:.3f}}});
\legend{{{lab_a}, {lab_b}}}
{chr(10).join(marks)}
\end{{axis}}"""


def figure(g: dict) -> str:
    facts, C = g["scene"], g["cases"]
    M = C["faulty_v3_seed0"]["holm_family"]
    thr = -np.log10(0.05 / M)
    lo, hi = g["depth_range_m"]
    w = 3.3
    panel_c = bars(
        C["clean_seed0"],
        C["faulty_v3_seed0"],
        "Clean",
        "Fault on v3",
        thr,
        r"\sfsub{c} Golden cases (seed 0)",
        "pa",
        "at={(scene.south west)}, anchor=north west, yshift=-1.55cm, xshift=0.6cm",
    )
    panel_d = bars(
        C["clean_seed1"],
        C["faulty_v5_seed0"],
        "Clean{,} seed 1",
        "Fault on v5",
        thr,
        r"\sfsub{d} Failures at the ends of the arc",
        "pb",
        "at={(pa.east)}, anchor=west, xshift=1.2cm",
        "north east",
    )
    return rf"""% Stanford bunny check of the view gate (generated by experiments/bunny_figures_tikz.py)
\begin{{tikzpicture}}
{scene_panel(facts)}
\begin{{axis}}[sfimage, name=depth, at={{($(scene.east)+(0.5cm,0.35cm)$)}}, anchor=west, width={w}cm, height={w}cm,
  xmin=0, xmax={facts["hw"]}, ymin=0, ymax={facts["hw"]}, y dir=reverse,
  title={{\sfsub{{b}} View v3 depth, +{100 * facts["fault_bias_m"]:.0f}\,cm window}}]
\addplot graphics[xmin=0, xmax={facts["hw"]}, ymin=0, ymax={facts["hw"]}] {{\mfigdir/fig_bunny_v3_depth.png}};
\draw[sfred, dashed, line width=0.8pt] (axis cs:{facts["fault_region"][2]},{facts["fault_region"][0]}) rectangle (axis cs:{facts["fault_region"][3]},{facts["fault_region"][1]});
\end{{axis}}
\begin{{axis}}[sfcolorbar={w}cm, colormap name=sfviridis, at={{($(depth.south)+(0,-0.12cm)$)}}, anchor=north,
  point meta min={lo:.1f}, point meta max={hi:.1f}, colorbar style={{xlabel={{Depth [m]}}, xtick={{{lo:.1f},{(lo + hi) / 2:.2f},{hi:.1f}}}}}]
\addplot[draw=none] coordinates {{(0,0)}};
\end{{axis}}
{panel_c}
{panel_d}
\end{{tikzpicture}}
"""


def main(manuscript: str) -> None:
    ms = Path(manuscript).resolve()
    d = ms / "figures" / "main"
    d.mkdir(parents=True, exist_ok=True)
    g = json.loads((RES / "gate.json").read_text())
    npz = dict(np.load(RES / "views_faulty_v3_seed0.npz"))
    valid = npz["depth"][3] > 0
    lo = float(np.floor(npz["depth"][3][valid].min() * 10) / 10)
    hi = float(np.ceil(npz["depth"][3][valid].max() * 10) / 10)
    g["depth_range_m"] = [lo, hi]
    depth_png(npz, 3, d / "fig_bunny_v3_depth.png", lo, hi)
    (d / "fig_bunny.tex").write_text(figure(g))
    if shutil.which("pdflatex"):
        from main_figures_tikz import compile_check

        compile_check(ms, ["fig_bunny"], RES)
    print(
        json.dumps(
            {"depth_range_m": [lo, hi], "holm_family": g["cases"]["faulty_v3_seed0"]["holm_family"]}
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent.parent / "papers" / "sensorfault"))
