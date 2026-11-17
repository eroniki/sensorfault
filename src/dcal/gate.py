"""ETH3D gate (PLAN sec. 10/11): does the unguided model's sample spread make the fault statistic work on real views?

CPU-only part. Inputs per view j (produced on the GPU by ``scripts/gate_eth3d.py``, same square centre crop as
``scripts/stage1_sample.py``):

- ``samples`` (M, s, s): unguided affine-invariant samples of view j (RGB of view j only);
- ``render`` / ``alpha`` (s, s): depth and opacity of a leave-j-out 2DGS built from the *other* views' sparse GT depth
  (a stand-in for a surface built from model predictions, as in ``scripts/fit_2dgs_pipes.py``);
- ``gt`` (s, s): view j's sparse laser depth (NaN = none), used only for the calibration metrics, never by the statistic.

Each crop is cut into ``tiles x tiles`` windows and each window is binned to a ``g x g`` grid (dense matrices, as in the toy).
Windows of the same view are correlated, so the empirical null from them is optimistic about its own size; it is a gate,
not a result.

Three questions, each answered against a trivial baseline:

1. **Is the spread informative about the error?** Spearman correlation of the per-bin sample spread with the per-bin
   absolute error against GT (after a scale/shift fit of the mean to GT), and z-score coverage at 1 and 2 sd.
2. **Is the clean-view statistic well behaved?** T/(n-2) on clean windows (1 for a calibrated posterior).
3. **Does the statistic detect injected depth-space faults** (plan sec. 6, family 1) at the empirical-null threshold of the
   identical pipeline? Compared with ``cov="iso"``: the same fit and test with an isotropic covariance, which ignores the
   model's spread. If the model's spread does not beat ``iso``, the spread adds nothing to the test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from .stat import align_samples, empirical_threshold, fault_statistic, fit_stationary_cov, shrunk_sample_cov

COVS = ("kernel", "shrunk", "iso")
FAULTS = ("square", "edge", "warp")


def bin_mean(img: np.ndarray, g: int, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``img`` over a g x g grid of near-equal bins (last two axes), ignoring invalid pixels; returns (mean, count)."""
    h, w = img.shape[-2:]
    iy = np.minimum(np.arange(h) * g // h, g - 1)
    ix = np.minimum(np.arange(w) * g // w, g - 1)
    lab = (iy[:, None] * g + ix[None, :]).ravel()
    ok = np.ones((h, w), bool) if valid is None else np.asarray(valid, bool)
    ok = ok.ravel()
    cnt = np.bincount(lab[ok], minlength=g * g).astype(float)
    flat = img.reshape(-1, h * w)
    out = np.stack([np.bincount(lab[ok], weights=np.where(np.isfinite(f[ok]), f[ok], 0.0), minlength=g * g) for f in flat])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, out / cnt, np.nan)
    return out.reshape(img.shape[:-2] + (g, g)), cnt.reshape(g, g)


@dataclass
class Window:
    view: int
    tile: tuple[int, int]
    samples: np.ndarray   # (M, n) binned samples
    render: np.ndarray    # (n,) binned render depth (NaN where uncovered)
    keep: np.ndarray      # (n,) bool: render covers the bin
    gt: np.ndarray        # (n,) binned GT (NaN where too few GT pixels)
    grid: tuple[int, int]


def make_windows(view: int, samples, render, alpha, gt, *, g: int = 24, tiles: int = 2, min_alpha_frac: float = 0.8,
                 min_gt_px: int = 4) -> list[Window]:
    S = np.asarray(samples, float)
    s = S.shape[-1]
    out = []
    edges = np.linspace(0, s, tiles + 1).astype(int)
    for ty in range(tiles):
        for tx in range(tiles):
            sl = (slice(edges[ty], edges[ty + 1]), slice(edges[tx], edges[tx + 1]))
            cov_ok = np.asarray(alpha)[sl] > 0.5
            Sb, _ = bin_mean(S[(slice(None),) + sl], g)
            Rb, rc = bin_mean(np.asarray(render, float)[sl], g, cov_ok)
            px = (edges[ty + 1] - edges[ty]) * (edges[tx + 1] - edges[tx]) / (g * g)
            keep = (rc / px >= min_alpha_frac) & np.isfinite(Rb)
            gv = np.isfinite(np.asarray(gt)[sl])
            Gb, gc = bin_mean(np.asarray(gt, float)[sl], g, gv)
            Gb = np.where(gc >= min_gt_px, Gb, np.nan)
            out.append(Window(view, (ty, tx), Sb.reshape(len(S), -1), Rb.ravel(), keep.ravel(), Gb.ravel(), (g, g)))
    return out


def sigma_u(A: np.ndarray, grid, cov: str) -> np.ndarray:
    if cov == "kernel":
        return fit_stationary_cov(A, grid)
    if cov == "shrunk":
        return shrunk_sample_cov(A)
    if cov == "iso":
        return float(A.var(0, ddof=1).mean()) * np.eye(A.shape[1])
    raise ValueError(cov)


def fault_pattern(kind: str, grid, mean: np.ndarray, area: float = 0.12, rng=None) -> np.ndarray:
    """Unit-amplitude fault shape on the grid (toy shapes; ``edge`` follows the level set of the sample mean)."""
    h, w = grid
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w]).reshape(2, 1, 1)
    if kind == "square":
        side = np.sqrt(area)
        oy, ox = (0.3, 0.3) if rng is None else rng.uniform(0.05, 0.95 - side, 2)
        f = (xx >= ox) & (xx < ox + side) & (yy >= oy) & (yy < oy + side)
    elif kind == "edge":
        f = (mean >= np.quantile(mean, 1 - area)).reshape(h, w)
    elif kind == "warp":
        f = np.sin(2 * np.pi * 1.5 * xx) * np.cos(2 * np.pi * yy)
        f = f / np.abs(f).max()
    else:
        raise ValueError(kind)
    return np.asarray(f, float).ravel()


def window_T(win: Window, cov: str, *, rel_sigma_r: float = 0.005, fault: str | None = None, kappa: float = 0.0,
             area: float = 0.12, rng=None, robust: bool = True) -> dict:
    """Statistic on one window; a fault of ``kappa`` x the median per-bin total sd is added to every sample (depth space)."""
    A = align_samples(win.samples)
    Su_full = sigma_u(A, win.grid, cov)
    k = win.keep
    S = win.samples[:, k].copy()
    D = win.render[k]
    Sr = np.diag((rel_sigma_r * D) ** 2 + 1e-12)
    Su = Su_full[np.ix_(k, k)]
    if fault is not None and kappa:
        M = S.shape[0]
        clean = fault_statistic(S, D, Sr, Sigma_u=Su, robust=robust)
        sd_tot = np.sqrt(np.median((1 + 1 / M) * np.diag(Su) + np.diag(Sr) / clean["s"] ** 2))   # in sample units
        f = fault_pattern(fault, win.grid, A.mean(0), area, rng)[k]
        S = S + kappa * sd_tot * f[None]
    r = fault_statistic(S, D, Sr, Sigma_u=Su, robust=robust)
    r["n"] = int(k.sum())
    return r


def calibration(win: Window) -> dict | None:
    """Spread vs error against GT on the bins that have GT (mean fitted to GT by least squares)."""
    A = align_samples(win.samples)
    m, sd = A.mean(0), A.std(0, ddof=1)
    ok = np.isfinite(win.gt)
    if ok.sum() < 10:
        return None
    G = np.stack([m[ok], np.ones(ok.sum())], 1)
    a, b = np.linalg.lstsq(G, win.gt[ok], rcond=None)[0]
    err = a * m[ok] + b - win.gt[ok]
    z = err / np.maximum(abs(a) * sd[ok] * np.sqrt(1 + 1 / len(A)), 1e-12)
    return {"abs_err": np.abs(err), "sd": abs(a) * sd[ok], "z": z, "gt": win.gt[ok]}


def run_gate(windows: list[Window], *, covs=COVS, faults=FAULTS, kappas=(0.5, 1.0, 2.0, 4.0), alpha: float = 0.05,
             rel_sigma_r: float = 0.005, area: float = 0.12, seed: int = 0, min_keep: float = 0.5) -> dict:
    wins = [w for w in windows if w.keep.mean() >= min_keep]
    rng = np.random.default_rng(seed)
    res = {"n_windows": len(wins), "n_views": len({w.view for w in wins}), "alpha": alpha, "rel_sigma_r": rel_sigma_r}
    cal = [c for c in (calibration(w) for w in wins) if c is not None]
    if cal:
        ae = np.concatenate([c["abs_err"] / c["gt"] for c in cal])
        sd = np.concatenate([c["sd"] / c["gt"] for c in cal])
        z = np.concatenate([c["z"] for c in cal])
        res["calibration"] = {"bins": int(len(z)), "spearman_spread_vs_abs_err": float(spearmanr(sd, ae)[0]),
                              "per_window_spearman_median": float(np.median([spearmanr(c["sd"], c["abs_err"])[0] for c in cal])),
                              "coverage_1sd": float((np.abs(z) < 1).mean()), "coverage_2sd": float((np.abs(z) < 2).mean()),
                              "mean_z2": float((z ** 2).mean()), "median_rel_err": float(np.median(ae)),
                              "median_rel_spread": float(np.median(sd))}
    res["covs"] = {}
    for cov in covs:
        null = [window_T(w, cov, rel_sigma_r=rel_sigma_r) for w in wins]
        nT = np.array([r["T"] for r in null])
        ratio = np.array([r["T"] / r["dof"] for r in null])
        # leave-one-VIEW-out threshold: a window is never tested against a null that contains its own view
        views = np.array([w.view for w in wins])
        thr = {v: empirical_threshold(nT[views != v], alpha) for v in np.unique(views)}
        fa = float(np.mean([t > thr[v] for t, v in zip(nT, views)]))
        out = {"null_T_over_dof_median": float(np.median(ratio)), "null_T_over_dof_mean": float(ratio.mean()),
               "false_alarm_loo": fa, "power": {}}
        for fault in faults:
            row = {}
            for kap in kappas:
                hits = [window_T(w, cov, rel_sigma_r=rel_sigma_r, fault=fault, kappa=kap, area=area, rng=rng)["T"] > thr[w.view]
                        for w in wins]
                row[str(kap)] = float(np.mean(hits))
            out["power"][fault] = row
        res["covs"][cov] = out
    return res


def verdict(res: dict, min_power: float = 0.8, kappa: str = "2.0") -> list[str]:
    """Plain-language reading of ``run_gate``'s output against the gate rule (beat the trivial ``iso`` baseline)."""
    lines = []
    c = res.get("calibration")
    if c:
        lines.append(f"spread vs error: Spearman {c['spearman_spread_vs_abs_err']:.2f} pooled "
                     f"(median per window {c['per_window_spearman_median']:.2f}); coverage {c['coverage_1sd']:.2f} at 1 sd "
                     f"(0.68 if calibrated), {c['coverage_2sd']:.2f} at 2 sd (0.95); mean z^2 {c['mean_z2']:.2f} (1)")
    iso = res["covs"].get("iso")
    for cov, o in res["covs"].items():
        p = {f: o["power"][f].get(kappa) for f in o["power"]}
        lines.append(f"{cov}: null T/(n-2) median {o['null_T_over_dof_median']:.2f}, false alarms {o['false_alarm_loo']:.2f}, "
                     f"power at kappa {kappa}: " + ", ".join(f"{f} {v:.2f}" for f, v in p.items()))
    if iso is not None:
        for cov in (k for k in res["covs"] if k != "iso"):
            o = res["covs"][cov]
            beats = [f for f in o["power"] if o["power"][f][kappa] > iso["power"][f][kappa] + 0.1]
            ok = beats and max(o["power"][f][kappa] for f in o["power"]) >= min_power
            lines.append(f"GATE {cov}: {'PASS' if ok else 'FAIL'} (beats iso by >0.1 on: {', '.join(beats) or 'none'})")
    return lines
