"""C5: TV-L1 range-image integration (Zach, Pock, Bischof 2007; Zach 2008) on a voxel grid.

Per view i a truncated signed distance f_i(x) = clamp(d_i(pi_i(x)) - z_i(x), -t, t) is
computed for every voxel centre x (positive in front of the surface); voxels that do not
project onto a valid depth pixel contribute nothing. We solve
    min_u  TV(u) + lambda * sum_i |u - f_i|
with Chambolle-Pock primal-dual iterations. The prox of theta*sum_i|u - f_i| at v has the
closed form (Zach 2008): the median of {f_i} U {v - theta(2k - n)}_{k=0..n}. Views that
are invalid at a voxel are encoded as the pair (+B, -B), which has zero sub-gradient.
Surface points = linear zero crossings of u along the grid axes. numpy only.
sensor_scores = mean |u* - f_k| over near-surface voxels (mean per-view L1 residual).
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from sensorfault.baselines._common import (
    ParamFuser,
    finish,
    frame_world_points,
    project,
    valid_mask,
    world_to_cam,
)
from sensorfault.baselines.base import register
from sensorfault.types import DepthFrame, FusionResult


def prox_l1_sum(v: np.ndarray, F: np.ndarray, theta: float) -> np.ndarray:
    """argmin_u 0.5 (u - v)^2 + theta * sum_e |u - F_e|, per voxel. F: (E, ...) entries."""
    n = F.shape[0]
    ks = np.arange(n + 1).reshape((n + 1,) + (1,) * v.ndim)
    cand = v[None] - theta * (2 * ks - n)
    return np.median(np.concatenate([F, cand], 0), axis=0)


def _grad(u):
    g = np.zeros((3,) + u.shape)
    g[0, :-1] = u[1:] - u[:-1]
    g[1, :, :-1] = u[:, 1:] - u[:, :-1]
    g[2, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    return g


def _div(p):  # negative adjoint of _grad
    d = np.zeros(p.shape[1:])
    d[:-1] += p[0, :-1]
    d[1:] -= p[0, :-1]
    d[:, :-1] += p[1, :, :-1]
    d[:, 1:] -= p[1, :, :-1]
    d[:, :, :-1] += p[2, :, :, :-1]
    d[:, :, 1:] -= p[2, :, :, :-1]
    return d


def tvl1_solve(f: np.ndarray, lam: float, iters: int, big: float) -> np.ndarray:
    """f: (V, X, Y, Z) truncated SDFs with NaN = invalid. Returns u (X, Y, Z)."""
    valid = np.isfinite(f)
    # each view -> two entries each of weight theta/2 so that invalid views cancel
    E = np.concatenate([np.where(valid, f, big), np.where(valid, f, -big)], 0)
    tau = sigma = 1.0 / np.sqrt(12.0)
    cnt = valid.sum(0)
    u = np.where(cnt > 0, np.nansum(np.where(valid, f, 0), 0) / np.maximum(cnt, 1), 0.0)
    ub = u.copy()
    p = np.zeros((3,) + u.shape)
    theta = tau * lam / 2.0
    for _ in range(iters):
        p += sigma * _grad(ub)
        p /= np.maximum(1.0, np.sqrt((p**2).sum(0)))[None]
        v = u + tau * _div(p)
        un = prox_l1_sum(v, E, theta)
        ub = 2 * un - u
        u = un
    return u


def zero_crossings(u: np.ndarray, origin: np.ndarray, voxel: float, mask: np.ndarray) -> np.ndarray:
    pts = []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(0, -1)
        b[ax] = slice(1, None)
        ua, ub = u[tuple(a)], u[tuple(b)]
        m = (((ua < 0) & (ub >= 0)) | ((ua > 0) & (ub <= 0))) & mask[tuple(a)] & mask[tuple(b)]
        idx = np.argwhere(m).astype(float)
        t = ua[m] / (ua[m] - ub[m])
        idx[:, ax] += t
        pts.append(origin + idx * voxel)
    return np.concatenate(pts, 0) if pts else np.zeros((0, 3))


@register("tvl1_range")
class TVL1Range(ParamFuser):
    """Params: voxel (0.02 m), trunc (3*voxel), lam (data weight, 0.5), iters (100),
    bounds ((min xyz),(max xyz)) or None -> bounding box of all valid points + margin."""

    DEFAULTS = dict(voxel=0.02, trunc=None, lam=0.5, iters=100, bounds=None, margin=3)

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        p = self.params(cfg)
        vox = float(p["voxel"])
        trunc = float(p["trunc"] or 3 * vox)
        if p["bounds"] is None:
            allp = np.concatenate([P[m] for P, m in (frame_world_points(f) for f in frames)], 0)
            lo = allp.min(0) - p["margin"] * vox
            hi = allp.max(0) + p["margin"] * vox
        else:
            lo, hi = (np.asarray(b, float) for b in p["bounds"])
        shape = tuple(np.maximum(np.ceil((hi - lo) / vox).astype(int) + 1, 2))
        gx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
        X = lo + gx.reshape(-1, 3) * vox
        sids = list(dict.fromkeys(f.sensor_id for f in frames))
        fs = np.full((len(frames),) + (len(X),), np.nan)
        for i, f in enumerate(frames):
            pc = world_to_cam(X, f.T_wc)
            u, v, z = project(pc, f.K)
            h, w = f.depth.shape
            ui = np.rint(np.nan_to_num(u, nan=-1)).astype(int)
            vi = np.rint(np.nan_to_num(v, nan=-1)).astype(int)
            ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            d = np.full(len(X), np.nan)
            d[ok] = f.depth[vi[ok], ui[ok]]
            ok &= valid_mask(d)
            fs[i, ok] = np.clip(d[ok] - z[ok], -trunc, trunc)
        fs = fs.reshape((len(frames),) + shape)
        u = tvl1_solve(fs, p["lam"], int(p["iters"]), 10 * trunc)
        has = np.isfinite(fs).any(0)
        pts = zero_crossings(u, lo, vox, has).astype(np.float32)
        band = has & (np.abs(u) < 0.9 * trunc)
        scores = {}
        for s in sids:
            idx = [i for i, f in enumerate(frames) if f.sensor_id == s]
            r = np.abs(fs[idx] - u[None])[:, band]
            r = r[np.isfinite(r)]
            scores[s] = float(r.mean()) if r.size else 0.0
        return finish(
            FusionResult(
                points=pts,
                excluded_sensors=dict.fromkeys(sids, False),
                sensor_scores=scores,
                meta=dict(grid=shape),
            ),
            t0,
        )
