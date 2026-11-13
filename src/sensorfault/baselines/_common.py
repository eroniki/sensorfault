"""Shared numpy helpers for the baselines (no heavy deps).

Conventions (match sensorfault.types): depth is z-depth in metres along the camera
optical axis, pixel (u, v) = (column, row) with pixel centres at integer coordinates,
T_wc is camera-to-world, K is a 3x3 pinhole matrix.
"""

from __future__ import annotations

import time
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from sensorfault.baselines.base import BaseFuser
from sensorfault.types import DepthFrame, FusionResult


class ParamFuser(BaseFuser):
    """BaseFuser with layered parameters: DEFAULTS < ctor kwargs < cfg < cfg[name]."""

    DEFAULTS: dict = {}

    def __init__(self, **kwargs):
        self._init = dict(kwargs)

    def params(self, cfg: Mapping | None) -> dict:
        p = dict(self.DEFAULTS)
        p.update(self._init)
        cfg = cfg or {}
        p.update({k: v for k, v in cfg.items() if not isinstance(v, Mapping)})
        sub = cfg.get(self.name)
        if isinstance(sub, Mapping):
            p.update(sub)
        return p


# ----------------------------------------------------------------------------- geometry
def valid_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def backproject_cam(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W, 3) camera-space points, NaN where depth is invalid."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.where(valid_mask(depth), depth.astype(np.float64), np.nan)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def cam_to_world(pts: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    return pts @ T_wc[:3, :3].T + T_wc[:3, 3]


def world_to_cam(pts: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    return (pts - T_wc[:3, 3]) @ T_wc[:3, :3]


def project(pts_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, z); u, v are NaN for z <= 0."""
    z = pts_cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pts_cam[..., 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[..., 1] / z + K[1, 2]
    bad = ~(z > 0)
    u = np.where(bad, np.nan, u)
    v = np.where(bad, np.nan, v)
    return u, v, z


def frame_world_points(f: DepthFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) world points (NaN invalid) and (H, W) validity."""
    P = backproject_cam(f.depth, f.K)
    m = np.isfinite(P[..., 2])
    Pw = np.full_like(P, np.nan)
    Pw[m] = cam_to_world(P[m], f.T_wc)
    return Pw, m


def depth_to_normals(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W, 3) unit normals in camera space, oriented towards the camera; NaN if the
    central-difference stencil touches an invalid pixel. Edges are NaN."""
    P = backproject_cam(depth, K)
    n = np.full_like(P, np.nan)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    c = np.cross(dx, dy)
    nrm = np.linalg.norm(c, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = c / np.where(nrm > 0, nrm, np.nan)
    ctr = P[1:-1, 1:-1]
    flip = np.sum(c * ctr, axis=-1, keepdims=True) > 0  # normal pointing away from camera
    c = np.where(flip, -c, c)
    n[1:-1, 1:-1] = c
    return n


def depth_to_world_points(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    P = backproject_cam(depth, K).reshape(-1, 3)
    m = np.isfinite(P[:, 2])
    return cam_to_world(P[m], T_wc).astype(np.float32)


def voxel_dedup(points: np.ndarray, voxel: float | None) -> np.ndarray:
    """Average points falling in the same voxel (cheap de-duplication)."""
    if not voxel or len(points) == 0:
        return points
    key = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, points)
    cnt = np.bincount(inv, minlength=n)[:, None]
    return (out / cnt).astype(np.float32)


# ------------------------------------------------------------- common reference stack
def same_camera(a: DepthFrame, ref: DepthFrame) -> bool:
    return (
        a.depth.shape == ref.depth.shape
        and np.allclose(a.K, ref.K)
        and np.allclose(a.T_wc, ref.T_wc)
    )


def reproject_to_ref(f: DepthFrame, ref: DepthFrame) -> np.ndarray:
    """Forward-splat frame ``f`` into the reference camera; z-buffer (nearest wins).
    Nearest-pixel splatting leaves small holes under strong viewpoint change
    (documented limitation; use co-located or near-co-located sensors for C2/D)."""
    h, w = ref.depth.shape
    if same_camera(f, ref):
        return np.where(valid_mask(f.depth), f.depth, np.nan).astype(np.float64)
    Pw, m = frame_world_points(f)
    pc = world_to_cam(Pw[m], ref.T_wc)
    u, v, z = project(pc, ref.K)
    ui = np.rint(u)
    vi = np.rint(v)
    ok = np.isfinite(ui) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    flat = vi[ok].astype(np.int64) * w + ui[ok].astype(np.int64)
    buf = np.full(h * w, np.inf)
    np.minimum.at(buf, flat, z[ok])
    buf[np.isinf(buf)] = np.nan
    return buf.reshape(h, w)


def sensor_stack(
    frames: Sequence[DepthFrame], ref_sensor: str | None = None
) -> Tuple[List[str], np.ndarray, DepthFrame]:
    """Stack all sensors onto the reference camera (first frame of ``ref_sensor`` or of
    the first sensor). Returns (sensor_ids, (S,H,W) depths with NaN, ref_frame).
    Several frames of one sensor are merged by nearest-depth z-buffer."""
    sids: List[str] = []
    for f in frames:
        if f.sensor_id not in sids:
            sids.append(f.sensor_id)
    ref_sensor = ref_sensor or sids[0]
    ref = next(f for f in frames if f.sensor_id == ref_sensor)
    h, w = ref.depth.shape
    stack = np.full((len(sids), h, w), np.nan)
    for f in frames:
        d = reproject_to_ref(f, ref)
        i = sids.index(f.sensor_id)
        stack[i] = np.fmin(stack[i], d)  # fmin ignores NaN
    return sids, stack, ref


def sigma_per_sensor(p: Mapping, sids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """sigma0_k and sigma_z2_k arrays; ``sigma`` may be float or {sensor: float}."""
    s = p.get("sigma", 0.01)
    c = p.get("sigma_z2", 0.0)
    s0 = np.array([s[k] if isinstance(s, Mapping) else s for k in sids], float)
    c0 = np.array([c[k] if isinstance(c, Mapping) else c for k in sids], float)
    return s0, c0


def sensor_var(
    stack: np.ndarray, s0: np.ndarray, c0: np.ndarray, floor: float = 1e-6
) -> np.ndarray:
    """(S,H,W) variance sigma_k(z)^2 with sigma_k(z) = s0_k + c_k z^2."""
    z = np.where(np.isfinite(stack), stack, 0.0)
    sig = s0[:, None, None] + c0[:, None, None] * z**2
    return np.maximum(sig, floor) ** 2


def weighted_fit(
    stack: np.ndarray, var: np.ndarray, member: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance mean over member sensors, NaN-aware.
    Returns (mu, v, count); mu/v are NaN where no member is valid."""
    ok = np.isfinite(stack) & member[:, None, None]
    w = np.where(ok, 1.0 / var, 0.0)
    sw = w.sum(0)
    num = (w * np.where(ok, stack, 0.0)).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(sw > 0, num / sw, np.nan)
        v = np.where(sw > 0, 1.0 / sw, np.nan)
    return mu, v, ok.sum(0)


def finish(res: FusionResult, t0: float) -> FusionResult:
    res.runtime_s = time.perf_counter() - t0
    return res


def project_into(Pw: np.ndarray, tgt: DepthFrame):
    """Project (H,W,3) world points into ``tgt`` (nearest pixel).

    Returns (inb, ui, vi, zx, dj): inb (H,W) bool = lands inside the image with positive
    depth *and* the target depth there is valid; ui/vi int (clipped); zx = depth of the
    point in the target camera; dj = target depth sampled at (vi, ui)."""
    h, w = tgt.depth.shape
    fin = np.isfinite(Pw[..., 2])
    pc = np.full_like(Pw, np.nan)
    pc[fin] = world_to_cam(Pw[fin], tgt.T_wc)
    u, v, z = project(pc, tgt.K)
    ui = np.rint(np.nan_to_num(u, nan=-1)).astype(np.int64)
    vi = np.rint(np.nan_to_num(v, nan=-1)).astype(np.int64)
    inb = fin & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    uc = np.clip(ui, 0, w - 1)
    vc = np.clip(vi, 0, h - 1)
    dj = tgt.depth[vc, uc].astype(np.float64)
    inb &= valid_mask(dj)
    return inb, uc, vc, z, dj
