"""Toy leave-view-out surface: the interface the real 2DGS pipeline must satisfy.

The hard rule of the method is that the render used to guide or test view j is built from the *other*
views only. This module fixes the interface (``fit_surface(views, exclude)`` and ``render``) and a
small point-based stand-in so the rule can be tested before any 2DGS exists. ``check_leave_out`` is
implementation-agnostic and is meant to be reused on the real pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np


class LeakError(RuntimeError):
    """Raised when excluded-view data is touched."""


@dataclass(frozen=True)
class ViewData:
    view_id: int
    rgb: np.ndarray      # (H, W, 3)
    depth: np.ndarray    # (H, W) depth predicted for this view (what the surface is built from)
    K: np.ndarray        # (3, 3)
    T_cw: np.ndarray     # (4, 4) world -> camera


class PoisonedView:
    """Stands in for ``ViewData`` and raises on any attribute access (view_id is the only exception)."""

    def __init__(self, view_id: int) -> None:
        object.__setattr__(self, "view_id", view_id)

    def __getattr__(self, name: str):
        raise LeakError(f"excluded view {self.view_id} was accessed: .{name}")


@dataclass(frozen=True)
class Surface:
    points: np.ndarray    # (N, 3) world
    source: np.ndarray    # (N,) source view id (kept for auditing and support counting)
    sigma2: np.ndarray    # (N,) per-point depth variance


def backproject(view: ViewData, sigma_rel: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = view.depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = view.depth.ravel()
    x = (u.ravel() + 0.5 - view.K[0, 2]) / view.K[0, 0] * z
    y = (v.ravel() + 0.5 - view.K[1, 2]) / view.K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    T_wc = np.linalg.inv(view.T_cw)
    pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    return pw, (sigma_rel * z) ** 2


def fit_surface(views: Mapping[int, ViewData], exclude: int, sigma_rel: float = 0.02) -> Surface:
    """Fuse all views except ``exclude``. The excluded entry is skipped by key, before any attribute access."""
    pts, src, s2 = [], [], []
    for vid in sorted(views):
        if vid == exclude:
            continue
        p, s = backproject(views[vid], sigma_rel)
        pts.append(p)
        src.append(np.full(len(p), vid))
        s2.append(s)
    return Surface(np.concatenate(pts), np.concatenate(src), np.concatenate(s2))


def render(surface: Surface, K: np.ndarray, T_cw: np.ndarray, shape: tuple[int, int], z_tol: float = 0.03):
    """Z-buffer render. Returns (depth, variance, support) with NaN where no point lands.

    support = number of distinct source views within ``z_tol`` (relative) of the nearest point in the pixel.
    """
    h, w = shape
    pc = surface.points @ T_cw[:3, :3].T + T_cw[:3, 3]
    ok = pc[:, 2] > 1e-6
    pc, src, s2 = pc[ok], surface.source[ok], surface.sigma2[ok]
    u = np.floor(pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = np.floor(pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, src, s2 = u[ins], v[ins], pc[ins, 2], src[ins], s2[ins]
    pix = v * w + u
    order = np.lexsort((z, pix))
    pix, z, src, s2 = pix[order], z[order], src[order], s2[order]
    first = np.r_[True, pix[1:] != pix[:-1]]
    zmin = np.full(h * w, np.nan)
    zmin[pix[first]] = z[first]
    keep = z <= zmin[pix] * (1 + z_tol)
    depth = np.full(h * w, np.nan)
    var = np.full(h * w, np.nan)
    support = np.zeros(h * w, int)
    d_sum = np.zeros(h * w)
    cnt = np.bincount(pix[keep], minlength=h * w)
    d_sum = np.bincount(pix[keep], weights=z[keep], minlength=h * w)
    s_sum = np.bincount(pix[keep], weights=s2[keep], minlength=h * w)
    m = cnt > 0
    depth[m] = d_sum[m] / cnt[m]
    var[m] = s_sum[m] / cnt[m] ** 2
    # distinct sources per pixel
    key = np.unique(np.stack([pix[keep], src[keep]], 1), axis=0)
    support = np.bincount(key[:, 0], minlength=h * w)
    return depth.reshape(h, w), var.reshape(h, w), support.reshape(h, w)


FitFn = Callable[[Mapping[int, "ViewData"], int], object]
RenderFn = Callable[[object, ViewData], tuple]


def perturb_view(view: ViewData, rng: np.random.Generator) -> ViewData:
    """Replace everything a view carries with different values (same shapes)."""
    t = view.T_cw.copy()
    t[:3, 3] += rng.normal(0, 0.5, 3)
    return replace(
        view,
        rgb=rng.random(view.rgb.shape),
        depth=view.depth * rng.uniform(0.5, 2.0) + rng.normal(0, 0.3, view.depth.shape),
        T_cw=t,
    )


def check_leave_out(fit: FitFn, render_fn: RenderFn, views: Mapping[int, ViewData], j: int,
                    rng: np.random.Generator, atol: float = 0.0) -> dict:
    """Empirical no-leak check for view ``j``. Returns booleans; all must be True for a compliant pipeline.

    ``atol`` > 0 allows a numerical tolerance for non-bit-reproducible (GPU) renders; 0 demands bit-identical output.

    1. ``poisoned``: the pipeline runs with view j replaced by an object that raises on access.
    2. ``invariant``: the render for j is bit-identical when j's data is arbitrarily perturbed or removed.
    3. ``sensitive``: perturbing a *different* view does change the render (guards against a vacuous test).
    """
    target = views[j]
    same = (lambda a, b: np.array_equal(a, b, equal_nan=True)) if atol == 0 else \
        (lambda a, b: np.allclose(a, b, atol=atol, rtol=0.0, equal_nan=True))
    ref = render_fn(fit(views, j), target)

    poisoned_views = {k: (PoisonedView(k) if k == j else v) for k, v in views.items()}
    try:
        got = render_fn(fit(poisoned_views, j), target)
        poisoned = all(same(a, b) for a, b in zip(ref, got))
    except LeakError:
        poisoned = False

    perturbed = dict(views)
    perturbed[j] = perturb_view(target, rng)
    got = render_fn(fit(perturbed, j), target)
    invariant = all(same(a, b) for a, b in zip(ref, got))

    other = next(k for k in views if k != j)
    perturbed = dict(views)
    perturbed[other] = perturb_view(views[other], rng)
    got = render_fn(fit(perturbed, j), target)
    sensitive = not all(same(a, b) for a, b in zip(ref, got))
    return {"poisoned": poisoned, "invariant": invariant, "sensitive": sensitive}
