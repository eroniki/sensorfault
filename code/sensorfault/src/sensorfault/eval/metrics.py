"""Point-cloud and detection metrics (numpy/scipy only)."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

_trapz = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201 (numpy<2 compat)


def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cKDTree(b).query(a, k=1)[0]


def accuracy(pred: np.ndarray, gt: np.ndarray, percentile: Optional[float] = None) -> float:
    """Mean (or percentile) distance pred -> gt. inf if pred is empty."""
    if len(pred) == 0 or len(gt) == 0:
        return float("inf")
    d = _nn_dist(pred, gt)
    return float(np.mean(d) if percentile is None else np.percentile(d, percentile))


def completeness(pred: np.ndarray, gt: np.ndarray, percentile: Optional[float] = None) -> float:
    """Mean (or percentile) distance gt -> pred (lower = more complete)."""
    if len(pred) == 0 or len(gt) == 0:
        return float("inf")
    d = _nn_dist(gt, pred)
    return float(np.mean(d) if percentile is None else np.percentile(d, percentile))


def chamfer(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Chamfer = 0.5*(accuracy + completeness), mean (L1) distances, metres."""
    return 0.5 * (accuracy(pred, gt) + completeness(pred, gt))


def fscore(pred: np.ndarray, gt: np.ndarray, tau: float) -> Dict[str, float]:
    """Precision (pred within tau of gt), recall (gt within tau of pred), F1."""
    if len(pred) == 0 or len(gt) == 0:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    p = float(np.mean(_nn_dist(pred, gt) <= tau))
    r = float(np.mean(_nn_dist(gt, pred) <= tau))
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f": f}


def pointcloud_metrics(pred: np.ndarray, gt: np.ndarray, taus: Sequence[float]) -> Dict[str, float]:
    out = {
        "accuracy": accuracy(pred, gt),
        "completeness": completeness(pred, gt),
        "chamfer": chamfer(pred, gt),
    }
    for t in taus:
        out[f"fscore@{t:g}"] = fscore(pred, gt, t)["f"]
    return out


def detection_pr(
    truth_faulty: Mapping[str, bool], predicted_excluded: Mapping[str, bool]
) -> Dict[str, float]:
    """Per-sensor detection precision/recall. precision=1 if nothing predicted (convention:
    no alarms -> no false alarms); recall=1 if nothing is faulty."""
    ids = list(truth_faulty)
    tp = sum(1 for i in ids if truth_faulty[i] and predicted_excluded.get(i, False))
    fp = sum(1 for i in ids if not truth_faulty[i] and predicted_excluded.get(i, False))
    fn = sum(1 for i in ids if truth_faulty[i] and not predicted_excluded.get(i, False))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def time_to_detect(
    onsets: Sequence[int], detections: Sequence[Optional[int]], horizon: int
) -> Dict[str, float]:
    """Time-to-detect (frames) with right-censoring.

    ``detections[i]`` is the first frame at which fault i was flagged (None = never).
    Delay = detection - onset (floored at 0; detections before onset count as 0 and are
    false alarms to be counted separately). Never-detected faults are censored at
    ``horizon - onset``. Median / 90th percentile come from the Kaplan-Meier estimate of the
    survival function S(t) = P(not yet detected); a quantile q is the smallest t with
    S(t) <= 1-q, or ``inf`` if S never gets that low (too much censoring).
    """
    T, E = [], []
    for o, d in zip(onsets, detections):
        if d is None:
            T.append(max(horizon - o, 0))
            E.append(0)
        else:
            T.append(max(d - o, 0))
            E.append(1)
    T, E = np.asarray(T, float), np.asarray(E, int)
    n = len(T)
    if n == 0:
        return {"median": float("nan"), "p90": float("nan"), "n": 0, "n_censored": 0}
    S, times = 1.0, np.unique(T[E == 1])
    curve = []
    for t in times:
        at_risk = np.sum(t <= T)
        ev = np.sum((t == T) & (E == 1))
        S *= 1 - ev / at_risk
        curve.append((t, S))

    def q(p):
        for t, s in curve:
            if s <= 1 - p + 1e-12:
                return float(t)
        return float("inf")

    return {"median": q(0.5), "p90": q(0.9), "n": n, "n_censored": int(np.sum(E == 0))}


def false_alarm_rate_per_100(n_alarms: int, n_clean_frames: int) -> float:
    """Alarms per 100 clean (fault-free) frames."""
    return 100.0 * n_alarms / n_clean_frames if n_clean_frames > 0 else float("nan")


def elimination_curve_auc(
    scores_faulty: Sequence[float], scores_clean: Sequence[float], fer_max: float = 0.2
) -> float:
    """Normalised partial area under the power-vs-false-elimination curve.

    DEFINITION (non-standard, PLAN.md section 6). A method assigns each sensor a fault score
    (higher = more suspicious). Sweep a threshold t over all observed scores:
        FER(t)   = fraction of sensors in CLEAN runs (no fault injected) with score > t
        power(t) = fraction of injected-FAULTY sensors with score > t
    The elimination curve is power as a function of FER. The AUC is
        (1 / fer_max) * integral_0^{fer_max} power(FER) dFER ,
    i.e. mean power over the operating region FER <= fer_max (default 0.2), computed by the
    trapezoid rule on the step ROC curve (power is right-continuous in FER). Range [0,1];
    1 = every faulty sensor outranks every clean one; 0 = none within the region. With
    fer_max=1 it equals the usual ROC AUC.
    """
    f = np.asarray(scores_faulty, float)
    c = np.asarray(scores_clean, float)
    if len(f) == 0 or len(c) == 0:
        return float("nan")
    thr = np.unique(np.concatenate([f, c, [-np.inf, np.inf]]))[::-1]
    fer = np.array([np.mean(c > t) for t in thr])
    pw = np.array([np.mean(f > t) for t in thr])
    # power at FER=x is the max power achievable with FER <= x
    xs = np.linspace(0, fer_max, 2001)
    py = np.array([pw[fer <= x + 1e-12].max() if np.any(fer <= x + 1e-12) else 0.0 for x in xs])
    return float(_trapz(py, xs) / fer_max)
