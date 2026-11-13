"""D / B3: iterated leave-one-sensor-out normalised-residual test (RAIM-style).

Per pixel and active sensor k: consensus of the *other* active sensors
  mu_{-k} = sum_j w_j d_j / sum_j w_j,  w_j = 1/sigma_j^2,  v_{-k} = 1/sum_j w_j,
residual r = d_k - mu_{-k}, s^2 = sigma_k^2 + v_{-k}. Statistic
  T_k = mean over pixels of r^2 / s^2   (E[T_k] = 1 under H0, Gaussian noise).
If max_k T_k > tau and more than ``min_sensors`` remain: exclude argmax, refit, repeat.
tau is calibrated on clean data (``calibrate_tau``). See PLAN.md sections 2 and 4.
"""

from __future__ import annotations

import time
from typing import List, Mapping, Optional, Sequence

import numpy as np

from sensorfault.baselines._common import (
    ParamFuser,
    depth_to_world_points,
    finish,
    sensor_stack,
    sensor_var,
    sigma_per_sensor,
    weighted_fit,
)
from sensorfault.baselines.base import register
from sensorfault.types import DepthFrame, FusionResult


def loo_statistics(
    stack: np.ndarray,
    var: np.ndarray,
    active: np.ndarray,
    min_others: int = 2,
    min_pixels: int = 50,
) -> np.ndarray:
    """T_k for every sensor in ``active`` (NaN for inactive / too few pixels)."""
    T = np.full(stack.shape[0], np.nan)
    for k in np.flatnonzero(active):
        others = active.copy()
        others[k] = False
        mu, v, cnt = weighted_fit(stack, var, others)
        m = np.isfinite(stack[k]) & (cnt >= min_others)
        if m.sum() < min_pixels:
            continue
        r = stack[k][m] - mu[m]
        s2 = var[k][m] + v[m]
        T[k] = float(np.mean(r * r / s2))
    return T


@register("loo_chi2")
class LooChi2(ParamFuser):
    """Params: sigma (float or {sensor: float}), sigma_z2, tau (default 1.5; replace by
    ``calibrate_tau`` output), min_sensors (3), min_others (2), min_pixels (50), ref_sensor."""

    DEFAULTS = dict(
        sigma=0.01,
        sigma_z2=0.0,
        tau=1.5,
        min_sensors=3,
        min_others=2,
        min_pixels=50,
        ref_sensor=None,
    )

    @staticmethod
    def calibrate_tau(clean_max_T: Sequence[float], alpha: float = 0.05) -> float:
        """tau = (1-alpha) empirical quantile of max_k T_k over clean runs/permutations."""
        return float(np.quantile(np.asarray(clean_max_T, float), 1.0 - alpha))

    def max_statistic(self, frames: Sequence[DepthFrame], cfg: Optional[Mapping] = None) -> float:
        """max_k T_k on the full sensor set (use on clean data to calibrate tau)."""
        p = self.params(cfg)
        sids, stack, _ = sensor_stack(frames, p["ref_sensor"])
        s0, c0 = sigma_per_sensor(p, sids)
        T = loo_statistics(
            stack,
            sensor_var(stack, s0, c0),
            np.ones(len(sids), bool),
            p["min_others"],
            p["min_pixels"],
        )
        return float(np.nanmax(T))

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        p = self.params(cfg)
        sids, stack, ref = sensor_stack(frames, p["ref_sensor"])
        s0, c0 = sigma_per_sensor(p, sids)
        var = sensor_var(stack, s0, c0)
        active = np.ones(len(sids), bool)
        first_T = None
        history: List[dict] = []
        while True:
            T = loo_statistics(stack, var, active, p["min_others"], p["min_pixels"])
            if first_T is None:
                first_T = T.copy()
            if not np.any(np.isfinite(T)):
                break
            k = int(np.nanargmax(T))
            history.append(dict(T={sids[i]: float(T[i]) for i in np.flatnonzero(active)}))
            if T[k] > p["tau"] and active.sum() > p["min_sensors"]:
                active[k] = False
                history[-1]["excluded"] = sids[k]
            else:
                break
        mu, _, cnt = weighted_fit(stack, var, active)
        pts = depth_to_world_points(mu, ref.K, ref.T_wc)
        scores = {
            s: (float(first_T[i]) if np.isfinite(first_T[i]) else 0.0) for i, s in enumerate(sids)
        }
        ex = {s: bool(not active[i]) for i, s in enumerate(sids)}
        return finish(
            FusionResult(
                points=pts,
                excluded_sensors=ex,
                sensor_scores=scores,
                meta=dict(tau=p["tau"], history=history),
            ),
            t0,
        )
