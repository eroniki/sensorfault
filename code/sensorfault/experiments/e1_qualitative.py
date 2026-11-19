"""Qualitative E1 figures from the MatrixCity block already on disk (light computation only).

One fault scene from the E1 grid is rebuilt from its seed (8 sigma on 40% of one view, seed 1): one
fusion pass of all 185 views, then the cavity test of a few views. Writes PNG panels (dependency-free
PNG writer) and LaTeX figure files to results/e1/qualitative/:
  1. per view (clean, faulty, swamped neighbour before and after the faulty view is dropped):
     ground-truth depth, cavity-rendered surfel depth, raw residual, standardised residual, region-scan
     statistic per tile with the flagged tiles outlined;
  2. per-pixel residual standard deviation (sqrt of the diagonal of Sigma_r) for a weakly and a strongly
     supported view, and the per-tile ratio of the full (U P U^T) to the diagonal-only variance of a
     tile sum;
  3. top-down map of the surfels coloured by height and by the fused normal-offset sd, with the flight
     path and the tested views.
Run: PYTHONPATH=src:experiments python experiments/e1_qualitative.py results/e1/qualitative
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import e1_matrixcity_injection as e1x
import numpy as np
from matrixcity_io import load_block
from scipy.stats import norm

VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
RDBU = [(33, 102, 172), (146, 197, 222), (247, 247, 247), (244, 165, 130), (178, 24, 43)]
GREY = (190, 190, 190)


def cmap(x: np.ndarray, anchors, lo: float, hi: float) -> np.ndarray:
    a = np.asarray(anchors, float)
    t = np.clip((x - lo) / (hi - lo), 0, 1) * (len(a) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(a) - 2)
    f = (t - i)[..., None]
    rgb = a[i] * (1 - f) + a[i + 1] * f
    rgb[~np.isfinite(x)] = GREY
    return rgb.astype(np.uint8)


def write_png(path: Path, rgb: np.ndarray, scale: int = 4) -> None:
    rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def colorbar(path: Path, anchors) -> None:
    write_png(path, cmap(np.tile(np.linspace(0, 1, 200), (8, 1)), anchors, 0, 1), scale=1)


def to_map(h, w, pix, values):
    m = np.full(h * w, np.nan)
    m[pix] = values
    return m.reshape(h, w)


def outline(rgb, mask):
    """Draw black borders of the True cells of a boolean (h, w) mask."""
    edge = mask & ~(
        np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    )
    edge[0, :] |= mask[0, :]
    edge[-1, :] |= mask[-1, :]
    edge[:, 0] |= mask[:, 0]
    edge[:, -1] |= mask[:, -1]
    rgb = rgb.copy()
    rgb[edge] = (0, 0, 0)
    return rgb


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(e1x.CFG)
    K, views, geo = e1x.load_cached(cfg)
    n, h, w = len(views), geo["h"], geo["w"]
    # the E1 grid scene: delta = 8, area = 0.4, seed = 1
    delta, area, seed = 8, 0.4, 1
    rng = np.random.default_rng(2000 + 97 * seed + int(10 * delta) + int(100 * area))
    e1x.measure(views, rng)
    j_fault = int(rng.integers(n))
    e1x.inject_region(views[j_fault], geo, rng, delta, area)
    g = e1x.Gate(views, geo, cfg, B=1)
    active = list(range(n))
    p0 = g.pvalues(active)
    ratio = {j: s["T"] / e1x.chi2.median(s["m"]) for j, s in g.last.items() if s is not None}
    S0 = g.last
    swamped = min((i for i in active if i != j_fault), key=lambda i: g.logp[i])
    clean = max(
        (i for i in active if S0[i] is not None and views[i]["strip"] != views[j_fault]["strip"]),
        key=lambda i: S0[i]["testable"],
    )
    weak = min((i for i in active if S0[i] is not None), key=lambda i: S0[i]["testable"])
    alpha, M = cfg["alpha"], n * cfg["round_cap"]

    def panels(j, act, tag):
        g.fuse(act)
        s = g.stats(j, act, return_maps=True)
        g2 = float(np.median([r for i, r in ratio.items() if i != j and i in act]))
        zthr = norm.isf(alpha / M / (2 * s["nR"])) * np.sqrt(
            g2
        )  # first Holm rejection, Bonferroni over tiles
        return s, zthr, g2

    cases = [
        ("clean", clean, active),
        ("faulty", j_fault, active),
        ("swamped", swamped, active),
        ("swamped_after", swamped, [i for i in active if i != j_fault]),
    ]
    gt = {}
    need = sorted({views[c[1]]["frame"] for c in cases} | {views[weak]["frame"]})
    _, raw = load_block(cfg["data_root"], cfg["transforms"], step=cfg["step"], frames=need)
    for v in raw:
        gt[v["frame"]] = v["depth"]
    depth_lo = min(np.nanpercentile(gt[f], 1) for f in gt)
    depth_hi = max(np.nanpercentile(gt[f], 99) for f in gt)
    meta = {
        "scene": {"delta_sigma": delta, "area": area, "seed": seed, "faulty_view": j_fault},
        "views": {},
    }
    rmax = 0.0
    store = {}
    for tag, j, act in cases:
        s, zthr, g2 = panels(j, act, tag)
        store[tag] = (j, s, zthr)
        rmax = max(rmax, np.nanpercentile(np.abs(s["r"]), 99))
    for tag, (j, s, zthr) in store.items():
        v = views[j]
        write_png(out / f"{tag}_gt.png", cmap(gt[v["frame"]], VIRIDIS, depth_lo, depth_hi))
        write_png(
            out / f"{tag}_render.png",
            cmap(to_map(h, w, s["pix"], s["D"]), VIRIDIS, depth_lo, depth_hi),
        )
        write_png(out / f"{tag}_raw.png", cmap(to_map(h, w, s["pix"], s["r"]), RDBU, -rmax, rmax))
        write_png(out / f"{tag}_std.png", cmap(to_map(h, w, s["pix"], s["z_std"]), RDBU, -5, 5))
        zt = np.abs(s["z_tile"])
        zt = np.where(s["tile_valid"], zt, np.nan)
        tile_map = zt[g.tile].reshape(h, w)
        flagged = (np.abs(s["z_tile"]) > zthr)[g.tile].reshape(h, w) & np.isfinite(tile_map)
        rgb = cmap(np.minimum(tile_map, 20.0), VIRIDIS, 0, 20)
        write_png(out / f"{tag}_scan.png", outline(rgb, flagged))
        meta["views"][tag] = {
            "view": j,
            "frame": v["frame"],
            "strip": list(v["strip"]),
            "p": p0[j] if tag != "swamped_after" else None,
            "zmax": s["zmax"],
            "z_threshold": float(zthr),
            "flagged_tiles": int((np.abs(s["z_tile"]) > zthr).sum()),
            "testable": s["testable"],
        }
    meta["scales"] = {
        "depth_m": [float(depth_lo), float(depth_hi)],
        "raw_residual_m": float(rmax),
        "std_residual": 5,
        "tile_z_max": 20,
    }

    # 2. uncertainty maps, weak vs strong support, clean data (same scene; neither is the faulty view)
    e1x.measure(views, np.random.default_rng(1000))
    g = e1x.Gate(views, geo, cfg, B=1)
    g.fuse(active)
    unc = {}
    for tag, j in (("weak", weak), ("strong", clean)):
        s = g.stats(j, active, return_maps=True)
        unc[tag] = (j, s)
    smax = max(np.nanpercentile(np.sqrt(s["s2"] + s["V"]), 99) for _, s in unc.values())
    rmax2 = max(np.nanmax(s["tile_var_ratio"]) for _, s in unc.values())
    for tag, (j, s) in unc.items():
        write_png(
            out / f"unc_{tag}_sd.png",
            cmap(to_map(h, w, s["pix"], np.sqrt(s["s2"] + s["V"])), VIRIDIS, 0, smax),
        )
        write_png(
            out / f"unc_{tag}_surf.png",
            cmap(
                to_map(h, w, s["pix"], np.sqrt(s["V"])),
                VIRIDIS,
                0,
                np.nanpercentile(np.sqrt(unc["weak"][1]["V"]), 99),
            ),
        )
        write_png(
            out / f"unc_{tag}_ratio.png",
            cmap(s["tile_var_ratio"][g.tile].reshape(h, w), VIRIDIS, 1.0, rmax2),
        )
        meta.setdefault("uncertainty", {})[tag] = {
            "view": j,
            "frame": views[j]["frame"],
            "testable": s["testable"],
            "median_sd_m": float(np.median(np.sqrt(s["s2"] + s["V"]))),
            "median_surface_sd_m": float(np.median(np.sqrt(s["V"]))),
            "tile_var_ratio_max": float(np.nanmax(s["tile_var_ratio"])),
            "tile_var_ratio_median": float(np.nanmedian(s["tile_var_ratio"])),
        }
    meta["scales"].update(
        sd_m=float(smax),
        surface_sd_m=float(np.nanpercentile(np.sqrt(unc["weak"][1]["V"]), 99)),
        tile_ratio_max=float(rmax2),
    )

    # 3. top-down surfel map coloured by height and by the fused normal-offset sd; flight path
    mu = geo["mu"]
    L = g.Lsum[:, 0, 0] + g.L0[0, 0]
    sd = 1.0 / np.sqrt(L)
    res = cfg["voxel_m"]  # one cell per surfel voxel
    x0, y0 = np.floor(mu[:, 0].min()), np.floor(mu[:, 1].min())
    ix = ((mu[:, 0] - x0) / res).astype(int)
    iy = ((mu[:, 1] - y0) / res).astype(int)
    W, H = ix.max() + 1, iy.max() + 1
    top = np.full((H, W), -np.inf)
    np.maximum.at(top, (iy, ix), mu[:, 2])
    top[~np.isfinite(top)] = np.nan
    sdm = np.full((H, W), np.inf)
    np.minimum.at(sdm, (iy, ix), sd)
    sdm[~np.isfinite(sdm)] = np.nan
    zlo, zhi = np.nanpercentile(top, 1), np.nanpercentile(top, 99)
    slo, shi = np.nanpercentile(sdm, 2), np.nanpercentile(sdm, 98)
    write_png(out / "top_height.png", cmap(top[::-1], VIRIDIS, zlo, zhi), scale=1)
    write_png(
        out / "top_sd.png",
        cmap(np.log10(sdm[::-1]), VIRIDIS, np.log10(slo), np.log10(shi)),
        scale=1,
    )
    cams = [[float(v["c"][0]), float(v["c"][1]), v["yaw"]] for v in views]
    tested = {
        t: [float(views[j]["c"][0]), float(views[j]["c"][1])]
        for t, j in (("clean", clean), ("faulty", j_fault), ("swamped", swamped), ("weak", weak))
    }
    meta["topdown"] = {
        "extent_m": [float(x0), float(x0 + W * res), float(y0), float(y0 + H * res)],
        "height_m": [float(zlo), float(zhi)],
        "offset_sd_m": [float(slo), float(shi)],
        "cell_m": res,
    }
    with (out / "cameras.csv").open("w") as f:
        f.write("x,y,yaw\n" + "\n".join(f"{c[0]},{c[1]},{c[2]}" for c in cams) + "\n")
    with (out / "tested.csv").open("w") as f:
        f.write("x,y,label\n" + "\n".join(f"{c[0]},{c[1]},{t}" for t, c in tested.items()) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    import e1_figures_tikz  # figures in the shared TikZ/PGFPlots style

    e1_figures_tikz.main(str(out.parent))
    print(json.dumps(meta, indent=1, default=float))


def write_tex(out: Path, meta: dict) -> None:
    sc = meta["scales"]
    V = meta["views"]
    rows = []
    names = {
        "clean": "clean",
        "faulty": "faulty",
        "swamped": "swamped neighbour",
        "swamped_after": "same, faulty view dropped",
    }
    for tag in ("clean", "faulty", "swamped", "swamped_after"):
        cells = " & ".join(
            rf"\includegraphics[width=0.18\linewidth]{{\qfigdir/{tag}_{k}.png}}"
            for k in ("gt", "render", "raw", "std", "scan")
        )
        lab = rf"\rotatebox{{90}}{{\scriptsize\parbox{{1.6cm}}{{\centering {names[tag]}\\ view {V[tag]['view']}}}}}"
        rows.append(f"{lab} & {cells} \\\\")
    ge20 = "$\\ge$20"
    cb = lambda f, lo, hi: (
        rf"\scriptsize {lo}\,\includegraphics[width=0.11\linewidth,height=0.6em]{{\qfigdir/{f}}}\,{hi}"
    )
    tex = rf"""% E1 qualitative: rendered vs true depth, residuals and the region scan (generated by experiments/e1_qualitative.py)
\setlength{{\tabcolsep}}{{1pt}}
\begin{{tabular}}{{ccccccc}}
 & \scriptsize true depth & \scriptsize cavity render & \scriptsize raw residual & \scriptsize standardised & \scriptsize region scan $|z|$ \\
{chr(10).join(rows)}
 & {cb("cb_viridis.png", f"{sc['depth_m'][0]:.0f}", f"{sc['depth_m'][1]:.0f} m")} & {cb("cb_viridis.png", f"{sc['depth_m'][0]:.0f}", f"{sc['depth_m'][1]:.0f} m")} & {cb("cb_rdbu.png", f"$-${sc['raw_residual_m']:.1f}", f"{sc['raw_residual_m']:.1f} m")} & {cb("cb_rdbu.png", "$-5$", "5")} & {cb("cb_viridis.png", "0", ge20)} \\
\end{{tabular}}
"""
    (out / "fig_e1_qual_views.tex").write_text(tex)
    U = meta["uncertainty"]
    rows = []
    for tag, lab in (("weak", "weakly supported"), ("strong", "strongly supported")):
        cells = " & ".join(
            rf"\includegraphics[width=0.27\linewidth]{{\qfigdir/unc_{tag}_{k}.png}}"
            for k in ("sd", "surf", "ratio")
        )
        rows.append(
            rf"\rotatebox{{90}}{{\scriptsize\parbox{{1.9cm}}{{\centering {lab}\\ view {U[tag]['view']}, {100 * U[tag]['testable']:.0f}\% testable}}}} & {cells} \\"
        )
    tex = rf"""% E1 qualitative: residual uncertainty and why the within-surfel term matters
\setlength{{\tabcolsep}}{{1pt}}
\begin{{tabular}}{{cccc}}
 & \scriptsize $\sqrt{{(\Sigma_r)_{{uu}}}}$ & \scriptsize surface part $\sqrt{{V_{{-j}}(u)}}$ & \scriptsize tile variance, full / diagonal \\
{chr(10).join(rows)}
 & {cb("cb_viridis.png", "0", f"{sc['sd_m']:.2f} m")} & {cb("cb_viridis.png", "0", f"{sc['surface_sd_m']:.2f} m")} & {cb("cb_viridis.png", "1", f"{sc['tile_ratio_max']:.1f}")} \\
\end{{tabular}}
"""
    (out / "fig_e1_qual_uncertainty.tex").write_text(tex)
    T = meta["topdown"]
    ext = T["extent_m"]
    panels = []
    for img, title in (
        ("top_height.png", f"surfel height, {T['height_m'][0]:.0f}--{T['height_m'][1]:.0f} m"),
        (
            "top_sd.png",
            f"normal-offset sd (log), {T['offset_sd_m'][0]:.2f}--{T['offset_sd_m'][1]:.2f} m",
        ),
    ):
        panels.append(rf"""\nextgroupplot[title={{{title}}}]
\addplot graphics[xmin={ext[0]}, xmax={ext[1]}, ymin={ext[2]}, ymax={ext[3]}] {{\qfigdir/{img}}};
\addplot[only marks, mark=*, mark size=0.8pt, black] table[x=x, y=y, col sep=comma] {{\qfigdir/cameras.csv}};
\addplot[only marks, mark=square*, mark size=2.2pt, red, nodes near coords, point meta=explicit symbolic, nodes near coords style={{font=\tiny, anchor=west, fill=white, inner sep=1pt}}]
  table[x=x, y=y, meta=label, col sep=comma] {{\qfigdir/tested.csv}};""")
    tex = rf"""% E1 qualitative: top-down surfel map of MatrixCity block 3, camera centres (black) and tested views (red)
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=1.4cm}}, width=7.2cm, height=7.2cm, axis equal image,
  xlabel={{$x$ (m)}}, ylabel={{$y$ (m)}}, tick label style={{font=\tiny}}, label style={{font=\scriptsize}}, title style={{font=\scriptsize}}, enlargelimits=false]
{chr(10).join(panels)}
\end{{groupplot}}
\end{{tikzpicture}}
"""
    (out / "fig_e1_qual_topdown.tex").write_text(tex)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/e1/qualitative")
