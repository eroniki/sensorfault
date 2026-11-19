"""2DGS (gsplat) surface behind the leave-view-out interface of ``surface.py``.

Structure-only surfels initialised from depth: one oriented disk per (strided) pixel of every view except the excluded one;
mean = back-projected point, rendering frame (t1, t2, n) from depth gradients, radius from the pixel footprint, no appearance, no
optimisation yet. Each source view is rasterised separately with gsplat's ``rasterization_2dgs`` ("RGB+ED": the colour channel
carries the per-surfel depth variance, ED is the expected z-depth) and fused in image space like the toy point surface:
nearest layer within ``z_tol``, mean depth, mean variance / support, support = number of source views with alpha > 0.5.
Requires torch with CUDA and gsplat (``pip install dcal[gpu]``); nothing here is imported by the CPU code paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
from gsplat import rasterization_2dgs
from scipy.spatial.transform import Rotation

from .surface import ViewData


@dataclass
class SurfaceGS:
    per_view: dict = field(default_factory=dict)   # view id -> dict(means, quats, scales, opac, var) as cuda tensors

    @property
    def source_ids(self):
        return sorted(self.per_view)


def _surfels_from_view(view: ViewData, stride: int, sigma_rel: float, edge: float, device):
    d = np.asarray(view.depth, np.float64)[::stride, ::stride]
    h, w = d.shape
    jj, ii = np.mgrid[0:h, 0:w]
    u = (ii * stride + 0.5 - view.K[0, 2]) / view.K[0, 0]
    v = (jj * stride + 0.5 - view.K[1, 2]) / view.K[1, 1]
    pc = np.stack([u * d, v * d, d], -1)                              # camera frame (h, w, 3)
    tu = np.gradient(pc, axis=1)
    tv = np.gradient(pc, axis=0)
    n = np.cross(tu, tv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    flip = (n * pc).sum(-1, keepdims=True) > 0                        # normals face the camera
    n = np.where(flip, -n, n)
    t1 = tu - (tu * n).sum(-1, keepdims=True) * n
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    R_cam = np.stack([t1, t2, n], -1)                                 # columns t1, t2, n
    T_wc = np.linalg.inv(view.T_cw)
    R_w = np.einsum("ij,hwjk->hwik", T_wc[:3, :3], R_cam)
    means = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    # drop depth discontinuities and degenerate normals
    dz = np.maximum(np.abs(np.gradient(d, axis=0)), np.abs(np.gradient(d, axis=1))) / np.maximum(d, 1e-9)
    keep = (dz < edge) & (nn[..., 0] > 1e-12) & np.isfinite(d) & (d > 0)
    flat = keep.ravel()
    q_xyzw = Rotation.from_matrix(R_w.reshape(-1, 3, 3)[flat]).as_quat()
    quats = np.concatenate([q_xyzw[:, 3:], q_xyzw[:, :3]], 1)         # gsplat wants (w, x, y, z)
    foot = (stride * d / view.K[0, 0]).reshape(-1)[flat]
    scales = np.stack([0.7 * foot, 0.7 * foot, 1e-4 * np.ones_like(foot)], 1)
    var = ((sigma_rel * d) ** 2).reshape(-1)[flat]
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)  # noqa: E731
    return {"means": t(means.reshape(-1, 3)[flat]), "quats": t(quats), "scales": t(scales),
            "opac": torch.full((int(flat.sum()),), 0.99, device=device), "var": t(var[:, None])}


def fit_surface_2dgs(views: Mapping[int, ViewData], exclude: int, stride: int = 2, sigma_rel: float = 0.02,
                     edge: float = 0.05, device: str = "cuda") -> SurfaceGS:
    """The excluded entry is skipped by key before any attribute access (same guarantee as ``surface.fit_surface``)."""
    s = SurfaceGS()
    for vid in sorted(views):
        if vid == exclude:
            continue
        s.per_view[vid] = _surfels_from_view(views[vid], stride, sigma_rel, edge, device)
    return s


@torch.no_grad()
def render_2dgs(surface: SurfaceGS, K: np.ndarray, T_cw: np.ndarray, shape: tuple[int, int], z_tol: float = 0.03,
                alpha_thresh: float = 0.5):
    """Returns numpy (depth, variance, support) with NaN where no source view has alpha > ``alpha_thresh``."""
    h, w = shape
    dev = next(iter(surface.per_view.values()))["means"].device
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)[None]
    Vt = torch.as_tensor(T_cw, dtype=torch.float32, device=dev)[None]
    depths, vars_, valid = [], [], []
    for vid in surface.source_ids:
        g = surface.per_view[vid]
        out = rasterization_2dgs(g["means"], g["quats"], g["scales"], g["opac"], g["var"], Vt, Kt, w, h,
                                 render_mode="RGB+ED")
        col, alpha = out[0][0], out[1][0, ..., 0]
        vmask = alpha > alpha_thresh
        depths.append(torch.where(vmask, col[..., 1], torch.full_like(alpha, float("inf"))))
        vars_.append(col[..., 0])
        valid.append(vmask)
    D = torch.stack(depths)
    V = torch.stack(vars_)
    M = torch.stack(valid)
    zmin = D.min(0).values
    keep = M & (D <= zmin * (1 + z_tol))
    cnt = keep.sum(0)
    depth = torch.where(cnt > 0, torch.where(keep, D, torch.zeros_like(D)).sum(0) / cnt.clamp(min=1),
                        torch.full_like(zmin, float("nan")))
    var = torch.where(cnt > 0, torch.where(keep, V, torch.zeros_like(V)).sum(0) / cnt.clamp(min=1) ** 2,
                      torch.full_like(zmin, float("nan")))
    return depth.cpu().numpy(), var.cpu().numpy(), M.sum(0).cpu().numpy()
