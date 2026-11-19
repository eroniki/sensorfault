"""Optimised 2DGS surface from sparse depth + RGB of every view except the excluded one (gsplat 2DGS, structure + colour).

Used on real ETH3D views whose ground-truth depth is sparse (about 9% of pixels): surfels are initialised at the valid depth pixels
as camera-facing disks, then trained with an RGB L1 loss on the full image and an L1 depth loss on the valid pixels. No densification,
no pruning. The excluded view is skipped by key before any attribute access, as in ``surface.fit_surface``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from gsplat import rasterization_2dgs
from scipy.spatial.transform import Rotation

from .surface import ViewData


@dataclass
class GSParams:
    means: torch.Tensor
    quats: torch.Tensor
    log_scales: torch.Tensor
    logit_opac: torch.Tensor
    colors: torch.Tensor


def _sparse_init(view: ViewData, device, footprint: float = 1.2):
    d = np.asarray(view.depth, np.float64)
    h, w = d.shape
    jj, ii = np.mgrid[0:h, 0:w]
    ok = np.isfinite(d) & (d > 0)
    u = (ii[ok] + 0.5 - view.K[0, 2]) / view.K[0, 0]
    v = (jj[ok] + 0.5 - view.K[1, 2]) / view.K[1, 1]
    z = d[ok]
    pc = np.stack([u * z, v * z, z], 1)
    T_wc = np.linalg.inv(view.T_cw)
    means = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    q = Rotation.from_matrix(T_wc[:3, :3]).as_quat()                 # disks face the camera (normal = camera z axis)
    quats = np.tile(np.array([q[3], q[0], q[1], q[2]]), (len(z), 1))
    s = footprint * z / view.K[0, 0]
    scales = np.stack([s, s, 0.1 * s], 1)
    cols = np.asarray(view.rgb, np.float64)[ok]
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)  # noqa: E731
    return t(means), t(quats), t(np.log(scales)), t(cols)


def render_view(p: GSParams, K, T_cw, shape, device="cuda"):
    h, w = shape
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)[None]
    Vt = torch.as_tensor(T_cw, dtype=torch.float32, device=device)[None]
    out = rasterization_2dgs(p.means, p.quats, torch.exp(p.log_scales), torch.sigmoid(p.logit_opac), p.colors, Vt, Kt, w, h,
                             render_mode="RGB+ED")
    return out[0][0, ..., :3], out[0][0, ..., 3], out[1][0, ..., 0]      # rgb, expected depth, alpha


def fit_2dgs(views: Mapping[int, ViewData], exclude: int, iters: int = 2000, seed: int = 0, lam_depth: float = 1.0,
             device: str = "cuda", log_every: int = 0) -> tuple[GSParams, dict]:
    parts = {k: _sparse_init(views[k], device) for k in sorted(views) if k != exclude}   # skips the excluded key first
    means = torch.cat([p[0] for p in parts.values()])
    quats = torch.cat([p[1] for p in parts.values()])
    logs = torch.cat([p[2] for p in parts.values()])
    cols = torch.cat([p[3] for p in parts.values()])
    P = GSParams(means.requires_grad_(), quats.requires_grad_(), logs.requires_grad_(),
                 torch.full((len(means),), 2.0, device=device).requires_grad_(), cols.clone().requires_grad_())
    opt = torch.optim.Adam([{"params": [P.means], "lr": 2e-4}, {"params": [P.quats], "lr": 1e-3},
                            {"params": [P.log_scales], "lr": 5e-3}, {"params": [P.logit_opac], "lr": 5e-2},
                            {"params": [P.colors], "lr": 2.5e-3}])
    gen = np.random.default_rng(seed)
    ids = [k for k in sorted(views) if k != exclude]
    cache = {k: (torch.as_tensor(np.asarray(views[k].rgb), dtype=torch.float32, device=device),
                 torch.as_tensor(np.asarray(views[k].depth), dtype=torch.float32, device=device)) for k in ids}
    hist = []
    for it in range(iters):
        k = ids[int(gen.integers(len(ids)))]
        rgb_t, dep_t = cache[k]
        rgb, depth, alpha = render_view(P, views[k].K, views[k].T_cw, rgb_t.shape[:2], device)
        valid = torch.isfinite(dep_t)
        l_rgb = (rgb - rgb_t).abs().mean()
        l_dep = (depth[valid] - dep_t[valid]).abs().mean()
        loss = l_rgb + lam_depth * l_dep
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            hist.append((it, float(l_rgb), float(l_dep)))
            print(f"  it {it:5d}  rgb L1 {float(l_rgb):.4f}  depth L1 {float(l_dep):.4f}  surfels {len(means)}", flush=True)
    return P, {"history": hist, "n_surfels": len(means)}
