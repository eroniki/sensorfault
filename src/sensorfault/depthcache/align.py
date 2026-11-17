"""Per-view alignment of cached depth to a reference (sparse or dense).

Metric models (Depth Pro, Metric3D v2, UniDepthV2) can be off by a large global scale on aerial
views (block 3 of MatrixCity: Depth Pro ~0.01x, Metric3D ~0.15x). Relative models (Depth Anything V2)
output inverse depth up to an affine map. Both are fitted per view against reference depth that is
valid at some pixels (SfM/triangulated points, LiDAR, GNSS-derived ground height, or exact depth in
synthetic tests). The fit is robust and the parameters are returned so callers can log or gate on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorfault.depthcache.base import DepthResult
from sensorfault.depthcache.cache import load_depth

_EPS = 1e-6


@dataclass
class Alignment:
    """depth_out = scale * depth (metric) or 1 / (a * raw + b) (relative inverse depth)."""

    mode: str  # "scale" | "affine_inverse"
    scale: float | None = None
    a: float | None = None
    b: float | None = None
    n_used: int = 0
    rel_resid_median: float = float("nan")  # median |aligned - ref| / ref on used pixels


def _valid_pairs(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None, min_depth: float):
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: depth {pred.shape} vs reference {ref.shape}")
    ok = np.isfinite(pred) & np.isfinite(ref) & (ref > min_depth)
    if mask is not None:
        ok &= mask.astype(bool)
    return ok


def fit_scale(pred: np.ndarray, ref: np.ndarray, ok: np.ndarray) -> float:
    """Median of ref/pred over valid pixels (robust to outliers, scale-only)."""
    p, r = pred[ok & (pred > _EPS)], ref[ok & (pred > _EPS)]
    if p.size == 0:
        raise ValueError("no valid pixels to fit scale")
    return float(np.median(r / p))


def fit_affine_inverse(
    raw: np.ndarray, ref: np.ndarray, ok: np.ndarray, iters: int = 5, trim: float = 0.9
) -> tuple[float, float]:
    """Fit 1/ref = a * raw + b by trimmed least squares (drops the worst 1 - trim each pass)."""
    x, y = raw[ok].astype(np.float64), 1.0 / ref[ok].astype(np.float64)
    if x.size < 2:
        raise ValueError("need at least 2 valid pixels for an affine fit")
    keep = np.ones(x.size, bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        xa, ya = x[keep], y[keep]
        if xa.size < 2 or np.ptp(xa) == 0:
            break
        a, b = np.polyfit(xa, ya, 1)
        r = np.abs(a * x + b - y)
        keep = r <= np.quantile(r, trim)
    return float(a), float(b)


def align_depth(
    result: DepthResult,
    ref_depth: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 0.0,
    max_depth: float | None = None,
) -> tuple[DepthResult, Alignment]:
    """Align one cached view to reference depth; returns a new DepthResult (kind metric_depth).

    ref_depth: (H, W) metres, NaN or <= min_depth where unknown. mask: optional extra validity.
    max_depth: clip the aligned depth (relative models can blow up near b ~ 0).
    """
    ref = np.asarray(ref_depth, dtype=np.float64)
    raw = result.depth.astype(np.float64)
    ok = _valid_pairs(raw, ref, mask, min_depth)
    if result.kind == "metric_depth":
        s = fit_scale(raw, ref, ok)
        out = raw * s
        al = Alignment("scale", scale=s)
    elif result.kind == "relative_inverse_depth":
        a, b = fit_affine_inverse(raw, ref, ok)
        inv = a * raw + b
        out = np.where(inv > _EPS, 1.0 / np.maximum(inv, _EPS), np.nan)
        al = Alignment("affine_inverse", a=a, b=b)
    else:
        raise ValueError(f"unknown depth kind {result.kind!r}")
    if max_depth is not None:
        out = np.where(out > max_depth, max_depth, out)
    u = ok & np.isfinite(out)
    al.n_used = int(u.sum())
    if al.n_used:
        al.rel_resid_median = float(np.median(np.abs(out[u] - ref[u]) / ref[u]))
    conf = result.confidence
    aligned = DepthResult(
        depth=out.astype(np.float32),
        kind="metric_depth",
        confidence=conf,
        focal_px=result.focal_px,
        extra={**result.extra, "aligned": True, "alignment_mode": al.mode},
    )
    return aligned, al


def load_depth_aligned(
    root,
    dataset: str,
    scene: str,
    model_id: str,
    view_id: str,
    ref_depth: np.ndarray,
    mask: np.ndarray | None = None,
    **kw,
) -> tuple[DepthResult, Alignment]:
    """load_depth + align_depth; the cache on disk stays unaligned (raw model output)."""
    return align_depth(load_depth(root, dataset, scene, model_id, view_id), ref_depth, mask, **kw)
