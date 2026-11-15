"""Surface (DSM) comparison on MatrixCity block 3: reconstructions vs the exact-depth ground truth.

One fusion operator for every surface, so that differences come from the depth, not from the fusion:
each view's depth is back-projected to world points, each view keeps the highest point per grid cell,
and the DSM is the median of those per-view maxima over the views that see the cell (at least
MIN_VIEWS). Sources:
  gt        MatrixCity exact z-depth of the 185 training frames, 480x270 (point-sampled every 4 px);
  rgb2dgs   the trained RGB 2DGS model (rgb_block3_2dgs.pt), median depth rendered at 480x270 by
            experiments/surface_render_rgb.py;
  e1_*      the E1 linearised surfels (exact-geometry voxel surfels, 96x54 per view) after fusing the
            noisy depth of the E1 grid scene of Figure S5 (8 sigma on 40% of one view, seed 1): with all
            views (gate off), without the views that the gate's Holm loop excluded (gate on), and the same
            noise without the fault (reference). Surfel depth = dbar + J theta with theta the fused
            posterior offset (nu, tau1, tau2).
Residual = reconstruction - GT height [m] on cells where both exist. Also depth-derived normals of
one view (GT vs RGB 2DGS). No training; light CPU work (2 threads) after the GPU render.
Run: PYTHONPATH=src:experiments python3 experiments/surface_dsm.py results/surface
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
import e1_matrixcity_injection as e1x  # noqa: E402
from matrixcity_io import load_block, pixel_rays_world  # noqa: E402

DATA = Path("/home/eiki/data/matrixcity")
CELL = 2.0  # DSM cell [m] for GT and the RGB 2DGS model
CELL_E1 = 4.0  # E1 surfels are sampled at 96x54 per view (3-4 m per pixel): coarser cell
WATER_Z = -2.0  # GT DSM below this is the flat water plane at -3.4 m (land and roads start at +0.5 m)
EXTENT = (-402.0, 402.0, -926.0, -220.0)  # x0, x1, y0, y1 [m] (block 3 surfel extent)
MIN_VIEWS = 3
NORMAL_FRAME = 88  # the clean view 69 of Figure S5
VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], float)
RDBU = np.array([(33, 102, 172), (146, 197, 222), (247, 247, 247), (244, 165, 130), (178, 24, 43)], float)


def grid_shape(cell=CELL):
    x0, x1, y0, y1 = EXTENT
    return int(np.ceil((y1 - y0) / cell)), int(np.ceil((x1 - x0) / cell))


class DSM:
    """Median over views of the per-view highest point in each cell."""

    def __init__(self, cell=CELL):
        self.cell = cell
        self.ny, self.nx = grid_shape(cell)
        self.per_view = []

    def add(self, X: np.ndarray) -> None:
        X = X[np.all(np.isfinite(X), 1)]
        ix = ((X[:, 0] - EXTENT[0]) / self.cell).astype(int)
        iy = ((X[:, 1] - EXTENT[2]) / self.cell).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        top = np.full(self.ny * self.nx, -np.inf, np.float32)
        np.maximum.at(top, iy[ok] * self.nx + ix[ok], X[ok, 2].astype(np.float32))
        self.per_view.append(top)

    def result(self):
        A = np.stack(self.per_view)
        A[~np.isfinite(A)] = np.nan
        n = np.sum(np.isfinite(A), 0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(A, 0)
        med[n < MIN_VIEWS] = np.nan
        return med.reshape(self.ny, self.nx), n.reshape(self.ny, self.nx)


def depth_normals(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    du, dv = np.gradient(X, axis=1), np.gradient(X, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n[np.sum(n * (c - X), -1) < 0] *= -1
    return n


def stats(res: np.ndarray, mask: np.ndarray | None = None) -> dict:
    r = res[np.isfinite(res) & (True if mask is None else mask)]
    if r.size == 0:
        return {"n_cells": 0}
    q = np.percentile(r, [5, 25, 50, 75, 95])
    return {"n_cells": int(r.size), "median_m": float(q[2]), "mean_m": float(r.mean()), "rmse_m": float(np.sqrt(np.mean(r**2))),
            "mae_m": float(np.mean(np.abs(r))), "p05_m": float(q[0]), "p25_m": float(q[1]), "p75_m": float(q[3]),
            "p95_m": float(q[4]), "abs_p90_m": float(np.percentile(np.abs(r), 90)), "within_1m": float(np.mean(np.abs(r) < 1.0))}


def cmap(x, anchors, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1) * (len(anchors) - 1)
    t = np.nan_to_num(t)
    i = np.minimum(t.astype(int), len(anchors) - 2)
    f = (t - i)[..., None]
    return (1 - f) * anchors[i] + f * anchors[i + 1]


def hillshade(z: np.ndarray, cell=CELL, az=315.0, alt=45.0) -> np.ndarray:
    zf = np.where(np.isfinite(z), z, np.nanmedian(z))
    gy, gx = np.gradient(zf, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = np.deg2rad(az), np.deg2rad(alt)
    return np.clip(np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def save_png(path: Path, rgb: np.ndarray, nan_mask: np.ndarray | None = None, north_up=True) -> None:
    from PIL import Image

    rgb = rgb.copy()
    if nan_mask is not None:
        rgb[nan_mask] = (200, 200, 200)
    img = rgb.astype(np.uint8)
    Image.fromarray(img[::-1] if north_up else img).save(path, optimize=True)


def e1_surfaces(log):
    """Fused E1 surfel depths per view for: reference (no fault), gate off, gate on."""
    cfg = dict(e1x.CFG)
    K, views, geo = e1x.load_cached(cfg)
    n, h, w = len(views), geo["h"], geo["w"]
    delta, area, seed = 8, 0.4, 1  # E1 grid scene of Figure S5 (same seeding as e1_qualitative.py)

    def scene(fault: bool):
        rng = np.random.default_rng(2000 + 97 * seed + int(10 * delta) + int(100 * area))
        e1x.measure(views, rng)
        j = int(rng.integers(n))
        if fault:
            e1x.inject_region(views[j], geo, rng, delta, area)
        return j

    def depths(g, active):
        g.fuse(active)
        L = g.Lsum + g.L0[None]
        theta = np.linalg.solve(L, g.Esum[..., None])[..., 0]
        out = []
        for v in views:
            ray = pixel_rays_world(K, v["R"], h, w).reshape(-1, 3)[v["pix"]]
            d = v["dbar"] + np.sum(v["J"] * theta[v["sid"]], 1)
            out.append(v["c"] + d[:, None] * ray)
        return out

    t0 = time.time()
    scene(False)
    ref = depths(e1x.Gate(views, geo, cfg, B=1), list(range(n)))
    j_fault = scene(True)
    fault_pts = None
    v = views[j_fault]
    ray = pixel_rays_world(K, v["R"], h, w).reshape(-1, 3)[v["pix"]]
    fault_pts = (v["c"] + v["dbar"][:, None] * ray)[v["fault"]]  # true positions of the faulty pixels
    off = depths(e1x.Gate(views, geo, cfg, B=1), list(range(n)))
    loop = e1x.run_loop(e1x.Gate(views, geo, cfg, B=1), n, cfg["round_cap"])
    excluded = [int(i) for i in loop.excluded]
    on = depths(e1x.Gate(views, geo, cfg, B=1), [i for i in range(n) if i not in excluded])
    log(f"E1 fusion + gate loop: {time.time() - t0:.0f} s; faulty view {j_fault}, excluded {excluded}")
    meta = {"faulty_view": j_fault, "faulty_frame": int(views[j_fault]["frame"]), "excluded_views": excluded,
            "excluded_frames": [int(views[i]["frame"]) for i in excluded], "rounds": int(loop.rounds) if hasattr(loop, "rounds") else None}
    return ref, off, on, fault_pts, meta


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "log.txt").open("w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    t_start = time.time()
    cfg = dict(e1x.CFG)
    # ground truth: exact z-depth at 480x270
    K4, gtv = load_block(cfg["data_root"], cfg["transforms"], step=4)
    gt, gt4 = DSM(), DSM(CELL_E1)
    gt_pts = {}
    for v in gtv:
        X = v["c"] + v["depth"][..., None] * pixel_rays_world(K4, v["R"], *v["depth"].shape)
        gt.add(X.reshape(-1, 3))
        gt4.add(X.reshape(-1, 3))
        if v["frame"] == NORMAL_FRAME:
            gt_pts[NORMAL_FRAME] = (X, v["c"])
    gt_z, gt_n = gt.result()
    gt4_z = gt4.result()[0]
    log(f"GT DSM: {np.isfinite(gt_z).mean():.3f} of cells, {time.time() - t_start:.0f} s")

    # RGB 2DGS: median depth rendered at the same 480x270 pixels
    R = np.load(DATA / "rgb_block3_2dgs_depth480.npz")
    by_frame = {v["frame"]: v for v in gtv}
    rgb = DSM()
    rgb_pts = {}
    for d, a, fr in zip(R["depth_median"], R["alpha"], R["frames"]):
        v = by_frame[int(fr)]
        dd = np.where((a.astype(np.float32) > 0.5) & (d > 1.0), d, np.nan)
        X = v["c"] + dd[..., None] * pixel_rays_world(K4, v["R"], *dd.shape)
        rgb.add(X.reshape(-1, 3))
        if int(fr) == NORMAL_FRAME:
            rgb_pts[NORMAL_FRAME] = (X, v["c"])
    rgb_z, rgb_n = rgb.result()
    log(f"RGB 2DGS DSM: {np.isfinite(rgb_z).mean():.3f} of cells, {time.time() - t_start:.0f} s")

    # E1 surfels: reference, gate off, gate on
    ref, off, on, fault_pts, e1meta = e1_surfaces(log)
    surf = {}
    for tag, pts in (("e1_ref", ref), ("e1_off", off), ("e1_on", on)):
        dsm = DSM(CELL_E1)
        for X in pts:
            dsm.add(X)
        surf[tag] = dsm.result()[0]
    # cells under the injected fault (true positions of the faulty pixels)
    ny, nx = grid_shape(CELL_E1)
    fmask = np.zeros((ny, nx), bool)
    ix = ((fault_pts[:, 0] - EXTENT[0]) / CELL_E1).astype(int)
    iy = ((fault_pts[:, 1] - EXTENT[2]) / CELL_E1).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    fmask[iy[ok], ix[ok]] = True

    res = {"rgb2dgs": rgb_z - gt_z, **{k: z - gt4_z for k, z in surf.items()}}
    water = {"rgb2dgs": gt_z < WATER_Z, **{k: gt4_z < WATER_Z for k in surf}}
    summary = {
        "cell_m": {"gt_rgb2dgs": CELL, "e1": CELL_E1}, "extent_m": list(EXTENT), "min_views": MIN_VIEWS, "water_below_m": WATER_Z,
        "coverage": {"gt": float(np.isfinite(gt_z).mean()), "gt_4m": float(np.isfinite(gt4_z).mean()), "rgb2dgs": float(np.isfinite(rgb_z).mean()),
                     **{k: float(np.isfinite(z).mean()) for k, z in surf.items()}},
        "water_share_of_gt": float(np.mean(gt_z[np.isfinite(gt_z)] < WATER_Z)),
        "residual_all": {k: stats(r) for k, r in res.items()},
        "residual_land": {k: stats(r, ~water[k]) for k, r in res.items()},
        "residual_water": {k: stats(r, water[k]) for k, r in res.items()},
        "residual_fault_region": {k: stats(r, fmask) for k, r in res.items() if k.startswith("e1")},
        "gate_effect_fault_region": stats(surf["e1_on"] - surf["e1_off"], fmask),
        "fault_region_cells": int(fmask.sum()),
        "e1_scene": e1meta,
        "height_range_m": [float(np.nanpercentile(gt_z, 1)), float(np.nanpercentile(gt_z, 99))],
    }

    # depth-derived normals of one view, GT vs RGB 2DGS (world frame)
    Xg, c = gt_pts[NORMAL_FRAME]
    Xr, _ = rgb_pts[NORMAL_FRAME]
    ng, nr = depth_normals(Xg, c), depth_normals(Xr, c)
    ang = np.degrees(np.arccos(np.clip(np.sum(ng * nr, -1), -1, 1)))
    ang[~np.all(np.isfinite(Xr), -1)] = np.nan
    land_px = Xg[..., 2] >= WATER_Z
    summary["normals_frame"] = NORMAL_FRAME

    def angstats(a):
        share = float(np.mean(np.isfinite(a)))
        a = a[np.isfinite(a)]
        return {"median": float(np.nanmedian(a)), "mean": float(np.nanmean(a)), "p90": float(np.nanpercentile(a, 90)),
                "within_10deg": float(np.mean(a < 10)), "share_of_pixels": share}

    summary["normal_angle_deg"] = {"all": angstats(ang), "land": angstats(np.where(land_px, ang, np.nan)),
                                   "water": angstats(np.where(~land_px, ang, np.nan))}

    # images (north up), height colour with hillshade; residuals on a diverging scale
    zlo, zhi = summary["height_range_m"]
    rmax = 5.0
    gmax = 5.0
    for tag, z, cell in (("gt", gt_z, CELL), ("gt_4m", gt4_z, CELL_E1), ("rgb2dgs", rgb_z, CELL), *((k, z, CELL_E1) for k, z in surf.items())):
        shade = hillshade(z, cell)[..., None]
        save_png(out / f"dsm_{tag}.png", cmap(z, VIRIDIS, zlo, zhi) * (0.45 + 0.55 * shade), ~np.isfinite(z))
    for tag, r in res.items():
        save_png(out / f"res_{tag}.png", cmap(r, RDBU, -rmax, rmax), ~np.isfinite(r))
    gdiff = surf["e1_on"] - surf["e1_off"]
    save_png(out / "gate_on_minus_off.png", cmap(gdiff, RDBU, -gmax, gmax), ~np.isfinite(gdiff))
    # fault-region outline as a CSV polygon set (cell centres of the boundary) for the TikZ overlay
    yy, xx = np.nonzero(fmask)
    with (out / "fault_cells.csv").open("w") as f:
        f.write("x,y\n" + "\n".join(f"{EXTENT[0] + (i + 0.5) * CELL_E1:.1f},{EXTENT[2] + (j + 0.5) * CELL_E1:.1f}" for j, i in zip(yy, xx)) + "\n")
    save_png(out / "normals_gt.png", np.nan_to_num((ng * 0.5 + 0.5) * 255), None, north_up=False)
    save_png(out / "normals_rgb2dgs.png", np.nan_to_num((nr * 0.5 + 0.5) * 255), ~np.all(np.isfinite(nr), -1), north_up=False)
    save_png(out / "normals_angle.png", cmap(np.minimum(ang, 45), VIRIDIS, 0, 45), ~np.isfinite(ang), north_up=False)
    summary["scales"] = {"height_m": [zlo, zhi], "residual_m": rmax, "gate_diff_m": gmax, "normal_angle_deg": 45}
    summary["wallclock_s"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps({k: summary[k] for k in ("coverage", "water_share_of_gt", "residual_fault_region", "gate_effect_fault_region",
                                            "e1_scene", "normal_angle_deg", "wallclock_s")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/surface")
