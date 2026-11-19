"""C2: per-pixel median across sensors with MAD outlier fraction (pure numpy)."""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from sensorfault.baselines._common import ParamFuser, depth_to_world_points, finish, sensor_stack
from sensorfault.baselines.base import register
from sensorfault.types import DepthFrame, FusionResult


@register("median_fusion")
class MedianFusion(ParamFuser):
    """All sensors are splatted onto a common reference camera grid; the fused depth is
    the per-pixel median over sensors (50 % breakdown). A sensor's score is the fraction
    of its valid pixels with |d - median| > k * max(1.4826*MAD, mad_floor).
    No sensor is excluded (excluded_sensors all False) unless ``exclude_frac`` is set.

    Params: ref_sensor, mad_k (3.0), mad_floor (metres, 0.005), min_sensors (2:
    pixels with fewer valid sensors are dropped), exclude_frac (None)."""

    DEFAULTS = dict(ref_sensor=None, mad_k=3.0, mad_floor=0.005, min_sensors=2, exclude_frac=None)

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        p = self.params(cfg)
        sids, stack, ref = sensor_stack(frames, p["ref_sensor"])
        n_valid = np.isfinite(stack).sum(0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(stack, axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(stack - med), axis=0)
        med = np.where(n_valid >= p["min_sensors"], med, np.nan)
        scale = np.maximum(mad, p["mad_floor"])
        scores = {}
        for i, s in enumerate(sids):
            m = np.isfinite(stack[i]) & np.isfinite(med)
            scores[s] = (
                float(np.mean(np.abs(stack[i][m] - med[m]) > p["mad_k"] * scale[m]))
                if m.any()
                else 0.0
            )
        ex = {s: (p["exclude_frac"] is not None and scores[s] > p["exclude_frac"]) for s in sids}
        pts = depth_to_world_points(med, ref.K, ref.T_wc)
        return finish(
            FusionResult(
                points=pts,
                excluded_sensors=ex,
                sensor_scores=scores,
                meta=dict(ref_sensor=ref.sensor_id),
            ),
            t0,
        )
