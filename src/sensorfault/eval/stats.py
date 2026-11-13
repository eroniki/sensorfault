"""Scene-level resampling statistics."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    stat: Callable = np.mean,
) -> Dict[str, float]:
    """Percentile bootstrap CI resampling SCENES (one value per scene; average seeds within a
    scene first so seeds are not treated as independent scenes)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB007]))
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(1)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(stat(v)), "lo": float(lo), "hi": float(hi), "n": len(v)}


def paired_wilcoxon(
    a: Sequence[float], b: Sequence[float], alternative: str = "two-sided"
) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank on per-scene metrics a vs b. Returns p=1.0 when all
    differences are zero (tie) instead of raising."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    nz = int(np.sum(d != 0))
    if nz == 0:
        return {"statistic": 0.0, "p": 1.0, "n": len(d), "n_nonzero": 0, "median_diff": 0.0}
    r = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return {
        "statistic": float(r.statistic),
        "p": float(r.pvalue),
        "n": len(d),
        "n_nonzero": nz,
        "median_diff": float(np.median(d)),
    }
