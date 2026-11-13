"""B3b: enumerated subset consensus scored with MSAC (K <= 6 sensors).

For every subset C (|C| >= min_size) of sensors: fit the per-pixel inverse-variance
consensus of C, then cost(C) = sum over ALL sensors and valid pixels of
min(r^2 / s^2, trunc) (MSAC truncated quadratic), s^2 = sigma_k^2 + v_C. The subset with
minimum cost is retained; sensors outside it are excluded. Ties broken towards the
larger subset. Exhaustive: 2^K subsets, hence the K <= 6 limit.
"""

from __future__ import annotations

import itertools
import time
from typing import Mapping, Sequence

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


@register("subset_msac")
class SubsetMSAC(ParamFuser):
    """Params: sigma, sigma_z2, trunc (9.0 = 3 sigma squared), min_size (2),
    max_excluded (None = K - min_size), ref_sensor, max_sensors (6)."""

    DEFAULTS = dict(
        sigma=0.01,
        sigma_z2=0.0,
        trunc=9.0,
        min_size=2,
        max_excluded=None,
        ref_sensor=None,
        max_sensors=6,
    )

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        p = self.params(cfg)
        sids, stack, ref = sensor_stack(frames, p["ref_sensor"])
        K = len(sids)
        if p["max_sensors"] < K:
            raise ValueError(f"subset_msac enumerates 2^K subsets; K={K} > {p['max_sensors']}")
        s0, c0 = sigma_per_sensor(p, sids)
        var = sensor_var(stack, s0, c0)
        max_ex = K - p["min_size"] if p["max_excluded"] is None else p["max_excluded"]
        valid = np.isfinite(stack)
        best = (np.inf, None, None)
        for size in range(
            K, max(p["min_size"], K - max_ex) - 1, -1
        ):  # large first -> ties favour large
            for sub in itertools.combinations(range(K), size):
                member = np.zeros(K, bool)
                member[list(sub)] = True
                mu, v, _ = weighted_fit(stack, var, member)
                r = stack - mu[None]
                with np.errstate(invalid="ignore"):
                    c = np.minimum(r * r / (var + v[None]), p["trunc"])
                m = valid & np.isfinite(mu)[None]
                cost = float(np.sum(np.where(m, c, 0.0)))
                if cost < best[0] - 1e-9:
                    best = (cost, member, c)
        cost, member, c = best
        m = valid & np.isfinite(weighted_fit(stack, var, member)[0])[None]
        scores = {s: (float(c[i][m[i]].mean()) if m[i].any() else 0.0) for i, s in enumerate(sids)}
        ex = {s: bool(not member[i]) for i, s in enumerate(sids)}
        mu, _, _ = weighted_fit(stack, var, member)
        pts = depth_to_world_points(mu, ref.K, ref.T_wc)
        return finish(
            FusionResult(
                points=pts,
                excluded_sensors=ex,
                sensor_scores=scores,
                meta=dict(msac_cost=cost, kept=[s for s, e in ex.items() if not e]),
            ),
            t0,
        )
