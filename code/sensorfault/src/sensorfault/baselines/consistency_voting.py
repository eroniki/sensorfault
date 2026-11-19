"""C3: own reimplementation of COLMAP-stereo-fusion-style geometric consistency.

NOT COLMAP code. For each pixel of each frame (reference), back-project to world and
project into every other frame (neighbour). A neighbour *supports* the pixel if
  (a) relative depth difference |z_proj - d_nb| / d_nb < max_depth_error,
  (b) the neighbour pixel re-projected into the reference lands within max_reproj_error px,
  (c) the normals (estimated from depth by central differences) differ by less than
      max_normal_error degrees.
A pixel is kept if >= min_consistent neighbours support it. Default thresholds are
placeholders (1 % depth, 2 px, 25 deg, 2 views) -- TODO(verify) against COLMAP's
StereoFusionOptions; tune on dev scenes only.
sensor_scores = fraction of that sensor's valid pixels rejected. No sensor is excluded.
"""

from __future__ import annotations

import time
from typing import List, Mapping, Sequence

import numpy as np

from sensorfault.baselines._common import (
    ParamFuser,
    depth_to_normals,
    finish,
    frame_world_points,
    project,
    project_into,
    voxel_dedup,
    world_to_cam,
)
from sensorfault.baselines.base import register
from sensorfault.types import DepthFrame, FusionResult


def support_counts(frames: Sequence[DepthFrame], p: Mapping):
    """Per-frame (H,W) support counts, valid masks and world points/normals."""
    Pw, Nw, valid = [], [], []
    for f in frames:
        P, m = frame_world_points(f)
        n = depth_to_normals(f.depth, f.K)
        nw = np.full_like(n, np.nan)
        mm = np.isfinite(n[..., 0])
        nw[mm] = n[mm] @ f.T_wc[:3, :3].T
        Pw.append(P)
        Nw.append(nw)
        valid.append(m)
    cos_t = np.cos(np.deg2rad(p["max_normal_error"]))
    counts: List[np.ndarray] = []
    for i, fi in enumerate(frames):
        cnt = np.zeros(fi.depth.shape, int)
        for j, fj in enumerate(frames):
            if i == j or (p["cross_sensor_only"] and fi.sensor_id == fj.sensor_id):
                continue
            inb, uc, vc, zx, dj = project_into(Pw[i], fj)
            ok = inb & valid[i]
            rel = np.abs(zx - dj) / np.maximum(dj, 1e-9)
            ok &= rel < p["max_depth_error"]
            # reprojection error of the neighbour sample back into the reference frame
            Xj = Pw[j][vc, uc]
            fin = np.isfinite(Xj[..., 2]) & ok
            pc = np.full_like(Xj, np.nan)
            pc[fin] = world_to_cam(Xj[fin], fi.T_wc)
            u, v, _ = project(pc, fi.K)
            uu, vv = np.meshgrid(np.arange(fi.depth.shape[1]), np.arange(fi.depth.shape[0]))
            err = np.hypot(u - uu, v - vv)
            ok &= fin & (err < p["max_reproj_error"])
            nj = Nw[j][vc, uc]
            dots = np.sum(Nw[i] * nj, axis=-1)
            ok &= np.isfinite(dots) & (dots > cos_t)
            cnt += ok
        counts.append(cnt)
    return counts, Pw, valid


@register("consistency_voting")
class ConsistencyVoting(ParamFuser):
    """Params: max_depth_error (0.01), max_reproj_error px (2.0), max_normal_error deg (25),
    min_consistent (2), cross_sensor_only (False), voxel (None: dedup voxel size)."""

    DEFAULTS = dict(
        max_depth_error=0.01,
        max_reproj_error=2.0,
        max_normal_error=25.0,
        min_consistent=2,
        cross_sensor_only=False,
        voxel=None,
    )

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        p = self.params(cfg)
        counts, Pw, valid = support_counts(frames, p)
        pts = []
        rej: dict = {}
        tot: dict = {}
        for f, c, P, m in zip(frames, counts, Pw, valid):
            keep = m & (c >= p["min_consistent"])
            pts.append(P[keep])
            rej[f.sensor_id] = rej.get(f.sensor_id, 0) + int(m.sum() - keep.sum())
            tot[f.sensor_id] = tot.get(f.sensor_id, 0) + int(m.sum())
        pts = np.concatenate(pts, 0) if pts else np.zeros((0, 3))
        pts = voxel_dedup(pts, p["voxel"]).astype(np.float32)
        scores = {s: (rej[s] / tot[s] if tot[s] else 0.0) for s in tot}
        return finish(
            FusionResult(
                points=pts, excluded_sensors=dict.fromkeys(tot, False), sensor_scores=scores
            ),
            t0,
        )
