"""Per-view alignment of cached depth to a reference (sparse or dense).

Metric models (Depth Pro, Metric3D v2, UniDepthV2) can be off by a large global scale on aerial
views (block 3 of MatrixCity: Depth Pro ~0.01x, Metric3D ~0.15x). Relative models (Depth Anything V2)
output inverse depth up to an affine map. Both are fitted per view against reference depth that is
valid at some pixels (SfM/triangulated points, LiDAR, GNSS-derived ground height, or exact depth in
synthetic tests). Fits are robust to outlying reference pixels; the parameters and fit quality are
returned so callers can gate on them.

Scope: alignment removes a per-view scale (and, for relative models, offset) error, so a fault that
is only a scale/offset error cannot be detected after it. If the reference is the same ground truth
the view is later tested against, the fit absorbs part of the very fault under test: for experiments
take the reference from something available at test time and leave the tested view out of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorfault.depthcache.base import DepthResult
from sensorfault.depthcache.cache import MANIFEST, load_depth, model_dir

_EPS = 1e-6
MIN_PIXELS = 200


class AlignmentError(ValueError):
    """The reference does not determine a usable alignment for this view."""


@dataclass
class Alignment:
    """depth_out = scale * depth (metric) or 1 / (a * raw + b) (relative inverse depth)."""

    mode: str  # "scale" | "affine_inverse"
    scale: float | None = None
    a: float | None = None
    b: float | None = None
    n_valid: int = 0  # reference pixels that were valid (finite, > min_depth, in mask, not clamped)
    n_inliers: int = 0  # pixels the final fit actually used (all valid ones for the scale fit)
    rel_resid_median: float = float("nan")  # median |aligned - ref| / ref over ALL valid pixels


def _valid_pairs(pred, ref, mask, min_depth, clamp_m):
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: depth {pred.shape} vs reference {ref.shape}")
    ok = np.isfinite(pred) & np.isfinite(ref) & (ref > min_depth)
    if mask is not None:
        ok &= mask.astype(bool)
    if clamp_m is not None:  # saturated by the model's clamp: not a measurement
        ok &= pred < clamp_m * (1 - 1e-6)
    return ok


def fit_scale(pred: np.ndarray, ref: np.ndarray, ok: np.ndarray, min_pixels: int = MIN_PIXELS):
    """Median of ref/pred over valid pixels (robust to outliers, scale-only)."""
    sel = ok & (pred > _EPS)
    n = int(sel.sum())
    if n < min_pixels:
        raise AlignmentError(f"only {n} valid reference pixels (min_pixels={min_pixels})")
    return float(np.median(ref[sel] / pred[sel])), n


def fit_affine_inverse(
    raw: np.ndarray,
    ref: np.ndarray,
    ok: np.ndarray,
    min_pixels: int = MIN_PIXELS,
    tol: float = 0.1,
    n_trials: int = 500,
    seed: int = 0,
):
    """Fit 1/ref = a * raw + b, robust to a large fraction of bad reference pixels.

    RANSAC over pixel pairs (inlier: relative inverse-depth error <= tol), then least squares on the
    inliers, repeated until the inlier set settles. Raises AlignmentError for too few pixels,
    constant ``raw``, no consensus, or a non-positive slope (inverse depth must grow with raw).
    """
    x_all, y_all = raw[ok].astype(np.float64), 1.0 / ref[ok].astype(np.float64)
    n = x_all.size
    if n < min_pixels:
        raise AlignmentError(f"only {n} valid reference pixels (min_pixels={min_pixels})")
    if np.ptp(x_all) <= 1e-12 * max(1.0, np.abs(x_all).max()):
        raise AlignmentError("constant raw depth over the valid pixels: affine fit undefined")
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, size=min(n, 20000), replace=False)
    xs, ys = x_all[sub], y_all[sub]
    best, best_cnt = None, -1
    for _ in range(n_trials):
        i, j = rng.integers(0, xs.size, 2)
        if xs[i] == xs[j]:
            continue
        a = (ys[i] - ys[j]) / (xs[i] - xs[j])
        if a <= 0:
            continue
        b = ys[i] - a * xs[i]
        cnt = int((np.abs(a * xs + b - ys) <= tol * ys).sum())
        if cnt > best_cnt:
            best, best_cnt = (a, b), cnt
    if best is None or best_cnt < max(2, min_pixels // 10, int(0.1 * xs.size)):
        raise AlignmentError("no consistent affine relation between raw depth and the reference")
    a, b = best
    inl = np.abs(a * x_all + b - y_all) <= tol * y_all
    for _ in range(5):
        if inl.sum() < 2 or np.ptp(x_all[inl]) == 0:
            raise AlignmentError("inlier set degenerated during refinement")
        a, b = np.polyfit(x_all[inl], y_all[inl], 1)
        new = np.abs(a * x_all + b - y_all) <= tol * y_all
        if np.array_equal(new, inl):
            break
        inl = new
    if a <= 0:
        raise AlignmentError(
            f"non-positive slope a={a:.3g}: raw is not increasing with inverse depth"
        )
    return float(a), float(b), int(inl.sum()), n


def align_depth(
    result: DepthResult,
    ref_depth: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth: float = 0.0,
    max_depth: float | None = None,
    min_pixels: int = MIN_PIXELS,
    clamp_m: float | None = None,
    keep_confidence: bool = False,
) -> tuple[DepthResult, Alignment]:
    """Align one cached view to reference depth; returns a new DepthResult (kind metric_depth).

    ref_depth: (H, W) metres, NaN or <= min_depth where unknown. mask: optional extra validity.
    max_depth: clip the aligned depth (relative models can blow up near b ~ 0).
    clamp_m: the model's own depth clamp (Metric3D ``max_depth_m``); pixels at it are saturated, not
        measured, so they are excluded from the fit and set to NaN in the output.
    keep_confidence: confidence is dropped by default, because its semantics differ per model (see
        docs) and are not converted here; pass True to carry the raw map through unchanged.
    Raises AlignmentError when fewer than ``min_pixels`` reference pixels are usable or the fit is
    degenerate. ``rel_resid_median`` is in-sample: gate on ``n_inliers`` too.
    """
    ref = np.asarray(ref_depth, dtype=np.float64)
    raw = result.depth.astype(np.float64)
    ok = _valid_pairs(raw, ref, mask, min_depth, clamp_m)
    if result.kind == "metric_depth":
        s, n_in = fit_scale(raw, ref, ok, min_pixels)
        out = raw * s
        al = Alignment("scale", scale=s, n_inliers=n_in)
    elif result.kind == "relative_inverse_depth":
        a, b, n_in, _ = fit_affine_inverse(raw, ref, ok, min_pixels)
        inv = a * raw + b
        out = np.where(inv > _EPS, 1.0 / np.maximum(inv, _EPS), np.nan)
        al = Alignment("affine_inverse", a=a, b=b, n_inliers=n_in)
    else:
        raise ValueError(f"unknown depth kind {result.kind!r}")
    if clamp_m is not None:
        out = np.where(raw >= clamp_m * (1 - 1e-6), np.nan, out)
    if max_depth is not None:
        out = np.where(out > max_depth, max_depth, out)
    u = ok & np.isfinite(out)
    al.n_valid = int(ok.sum())
    if u.any():
        al.rel_resid_median = float(np.median(np.abs(out[u] - ref[u]) / ref[u]))
    aligned = DepthResult(
        depth=out.astype(np.float32),
        kind="metric_depth",
        confidence=result.confidence if keep_confidence else None,
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
    """load_depth + align_depth; the cache on disk stays unaligned (raw model output).

    If the model's manifest records a ``max_depth_m`` clamp (Metric3D), it is used as ``clamp_m``.
    """
    if "clamp_m" not in kw:
        import json

        mpath = model_dir(root, dataset, scene, model_id) / MANIFEST
        if mpath.is_file():
            kw["clamp_m"] = json.loads(mpath.read_text()).get("model_info", {}).get("max_depth_m")
    return align_depth(load_depth(root, dataset, scene, model_id, view_id), ref_depth, mask, **kw)
