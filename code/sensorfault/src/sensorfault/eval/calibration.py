"""Calibration of stated placement information against realised placement error.

An entity carries a stated information matrix Lambda (3x3, m^-2) on its placement (offset and two
tilts times the footprint half-width) and a realised error epsilon (3,) in metres. If Lambda is
honest, z^2 = epsilon^T Lambda epsilon is chi-square with three degrees of freedom. Everything here
works on z^2 (and log det Lambda), so it does not care how the records were produced.

Definitions follow the protocol (docs/learned-indicator/experiment-design.md, Sections 1 and 4):

- robust scale s_z = sqrt(median(z^2) / 2.366) (2.366 is the chi2_3 median),
- tail scale sqrt(mean(z^2) / 3),
- coverage: share of entities with z^2 <= 3.53 (0.683) and <= 7.815 (0.95),
- ECE: mean absolute deviation of the empirical CDF of u = F_chi2_3(z^2) from the diagonal at the
  19 levels 0.05 ... 0.95,
- surface log-score: sum_m A_m (z^2/2 - log det(Lambda_m)/2) / sum_m A_m, nats (lower is better),
  without the constant (3/2) log 2 pi,
- calibration loss: (3/2)(c - 1 - log c) with c = mean(z^2)/3, the part of the log-score that a
  global rescaling of Lambda by 1/c removes.

Intervals come from a percentile bootstrap over split tiles: tiles are resampled with replacement
and the statistic is recomputed on the pooled entities of the drawn tiles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

DOF = 3
CHI2_MEDIAN = float(stats.chi2.ppf(0.5, DOF))  # 2.366
COVERAGE_LEVELS = (0.683, 0.95)
ECE_LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def z2(eps: np.ndarray, Lam: np.ndarray) -> np.ndarray:
    """(N, 3) errors and (N, 3, 3) information -> (N,) squared standardised error."""
    eps = np.asarray(eps, float)
    return np.einsum("ni,nij,nj->n", eps, np.asarray(Lam, float), eps)


def logdet(Lam: np.ndarray) -> np.ndarray:
    sign, ld = np.linalg.slogdet(np.asarray(Lam, float))
    return np.where(sign > 0, ld, np.nan)


def robust_scale(z2v: np.ndarray) -> float:
    z2v = np.asarray(z2v, float)
    return float(np.sqrt(np.median(z2v) / CHI2_MEDIAN)) if z2v.size else float("nan")


def tail_scale(z2v: np.ndarray) -> float:
    z2v = np.asarray(z2v, float)
    return float(np.sqrt(np.mean(z2v) / DOF)) if z2v.size else float("nan")


def coverage(z2v: np.ndarray, level: float) -> float:
    z2v = np.asarray(z2v, float)
    return float(np.mean(z2v <= stats.chi2.ppf(level, DOF))) if z2v.size else float("nan")


def ece(z2v: np.ndarray, levels: Sequence[float] = ECE_LEVELS) -> float:
    z2v = np.asarray(z2v, float)
    if not z2v.size:
        return float("nan")
    u = stats.chi2.cdf(z2v, DOF)
    emp = np.array([np.mean(u <= q) for q in levels])
    return float(np.mean(np.abs(emp - np.asarray(levels))))


def log_score(z2v: np.ndarray, logdet_v: np.ndarray, area: np.ndarray | None = None) -> float:
    """Area-weighted mean of z^2/2 - log det(Lambda)/2, in nats per entity (per m^2 with ``area``)."""
    z2v, ld = np.asarray(z2v, float), np.asarray(logdet_v, float)
    if not z2v.size:
        return float("nan")
    w = np.ones_like(z2v) if area is None else np.asarray(area, float)
    return float(np.sum(w * (0.5 * z2v - 0.5 * ld)) / np.sum(w))


def calibration_loss(z2v: np.ndarray) -> float:
    z2v = np.asarray(z2v, float)
    if not z2v.size:
        return float("nan")
    c = float(np.mean(z2v)) / DOF
    return 1.5 * (c - 1.0 - np.log(c))


def summary(z2v: np.ndarray, logdet_v: np.ndarray | None = None) -> dict[str, float]:
    """All point metrics of one set of entities."""
    out = {
        "n": int(np.size(z2v)),
        "s_z": robust_scale(z2v),
        "s_z_tail": tail_scale(z2v),
        "cov_0.683": coverage(z2v, 0.683),
        "cov_0.95": coverage(z2v, 0.95),
        "ece": ece(z2v),
        "calibration_loss": calibration_loss(z2v),
    }
    if logdet_v is not None:
        out["log_score"] = log_score(z2v, logdet_v)
    return out


def tile_bootstrap(
    stat: Callable[[np.ndarray], float],
    tiles: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap of ``stat(index_array)`` over tiles.

    ``tiles`` is the (N,) tile id of each entity; ``stat`` receives an index array into the entities
    (with repeats) and returns a scalar. Returns estimate (on all entities), lo, hi, n_tiles.
    """
    tiles = np.asarray(tiles)
    uniq, inv = np.unique(tiles, return_inverse=True)
    groups = [np.flatnonzero(inv == t) for t in range(len(uniq))]
    est = float(stat(np.arange(len(tiles))))
    if len(uniq) < 2:
        return {"estimate": est, "lo": float("nan"), "hi": float("nan"), "n_tiles": len(uniq)}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5A]))
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    bs = np.empty(n_boot)
    for b in range(n_boot):
        bs[b] = stat(np.concatenate([groups[t] for t in draws[b]]))
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "n_tiles": len(uniq)}


def bootstrap_summary(
    z2v: np.ndarray,
    logdet_v: np.ndarray,
    tiles: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    metrics: Sequence[str] = (
        "s_z",
        "cov_0.683",
        "cov_0.95",
        "ece",
        "log_score",
        "calibration_loss",
    ),
) -> dict[str, Mapping[str, float]]:
    """Tile-bootstrap interval of each metric of :func:`summary`."""
    z2v, logdet_v = np.asarray(z2v, float), np.asarray(logdet_v, float)
    fns: dict[str, Callable[[np.ndarray], float]] = {
        "s_z": lambda i: robust_scale(z2v[i]),
        "s_z_tail": lambda i: tail_scale(z2v[i]),
        "cov_0.683": lambda i: coverage(z2v[i], 0.683),
        "cov_0.95": lambda i: coverage(z2v[i], 0.95),
        "ece": lambda i: ece(z2v[i]),
        "log_score": lambda i: log_score(z2v[i], logdet_v[i]),
        "calibration_loss": lambda i: calibration_loss(z2v[i]),
    }
    return {
        k: tile_bootstrap(fns[k], tiles, n_boot=n_boot, seed=seed + j)
        for j, k in enumerate(metrics)
    }
