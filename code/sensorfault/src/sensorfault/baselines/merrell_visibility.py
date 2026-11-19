"""C4: own reimplementation of Merrell et al. 2007 (real-time visibility-based fusion)
stability test; the mesh/volumetric stage is NOT implemented (point output only).

For a depth sample X of reference view i and every other view j (X projected to j's
nearest pixel, d_j the depth there, eps = rel_eps):
  * support   : |z_X - d_j| <= eps * z_X                         (j agrees with X)
  * conflict  : d_j >  z_X (1 + eps)   -> X lies in j's free space (j sees *through* X)
  * conflict  : d_j <  z_X (1 - eps) and j's surface point Y, projected back into view i,
                lands within one pixel with depth < d_i (1 - eps)   -> Y occludes X in i,
                i.e. i's depth is too far (violation of free space of i).
  * otherwise (d_j nearer but not in i's line of sight): X occluded in j, no evidence.
X is kept iff support >= min_support and support >= conflict (stability >= 0).
sensor_scores = mean conflict / (support + conflict + 1) per sensor. No exclusion.
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
    project_into,
    voxel_dedup,
    world_to_cam,
)
from sensorfault.baselines.base import register
from sensorfault.types import DepthFrame, FusionResult


def stability_counts(frames: Sequence[DepthFrame], eps: float, pix_tol: float = 1.0):
    Pw, valid = zip(*[frame_world_points(f) for f in frames])
    out = []
    for i, fi in enumerate(frames):
        h, w = fi.depth.shape
        sup = np.zeros((h, w), int)
        con = np.zeros((h, w), int)
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        for j, fj in enumerate(frames):
            if i == j:
                continue
            inb, uc, vc, zx, dj = project_into(Pw[i], fj)
            inb &= valid[i]
            agree = inb & (np.abs(zx - dj) <= eps * zx)
            far = inb & (dj > zx * (1 + eps))
            near = inb & (dj < zx * (1 - eps))
            # occlusion case: does j's nearer surface point lie on i's line of sight?
            Y = Pw[j][vc, uc]
            fin = near & np.isfinite(Y[..., 2])
            pc = np.full_like(Y, np.nan)
            pc[fin] = world_to_cam(Y[fin], fi.T_wc)
            u, v, zy = project(pc, fi.K)
            di = fi.depth.astype(np.float64)
            with np.errstate(invalid="ignore"):
                viol = fin & (np.hypot(u - uu, v - vv) <= pix_tol) & (zy < di * (1 - eps))
            sup += agree
            con += far | viol
        out.append((sup, con))
    return out, Pw, valid


@register("merrell_visibility")
class MerrellVisibility(ParamFuser):
    """Params: rel_eps (0.02), min_support (1), pix_tol (1.0), voxel (None)."""

    DEFAULTS = dict(rel_eps=0.02, min_support=1, pix_tol=1.0, voxel=None)

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        p = self.params(cfg)
        sc, Pw, valid = stability_counts(frames, p["rel_eps"], p["pix_tol"])
        pts = []
        acc: dict = {}
        for f, (sup, con), P, m in zip(frames, sc, Pw, valid):
            keep = m & (sup >= p["min_support"]) & (sup >= con)
            pts.append(P[keep])
            frac = (con / (sup + con + 1.0))[m]
            a = acc.setdefault(f.sensor_id, [0.0, 0])
            a[0] += float(frac.sum())
            a[1] += int(m.sum())
        pts = np.concatenate(pts, 0) if pts else np.zeros((0, 3))
        pts = voxel_dedup(pts, p["voxel"]).astype(np.float32)
        scores = {s: (a[0] / a[1] if a[1] else 0.0) for s, a in acc.items()}
        return finish(
            FusionResult(
                points=pts, excluded_sensors=dict.fromkeys(acc, False), sensor_scores=scores
            ),
            t0,
        )
