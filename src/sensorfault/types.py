"""Shared data contracts. Every module (sim, faults, baselines, elim, eval) uses these."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    """One depth image from one sensor.

    depth: (H, W) float32 metres, 0 or NaN = invalid.
    K: (3, 3) pinhole intrinsics.
    T_wc: (4, 4) camera-to-world pose (the *believed* pose; faults may corrupt it).
    conf: optional (H, W) float32 in [0, 1], higher = better.
    """

    sensor_id: str
    frame_id: int
    depth: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    conf: Optional[np.ndarray] = None


@dataclass
class FusionResult:
    """Output of any fuser. Points are in world frame."""

    points: np.ndarray  # (N, 3) float32
    normals: Optional[np.ndarray] = None  # (N, 3)
    # sensor_id -> True if the method excluded this sensor (sensor-level methods only)
    excluded_sensors: dict[str, bool] = field(default_factory=dict)
    # sensor_id -> scalar fault score (higher = more suspicious), if the method has one
    sensor_scores: dict[str, float] = field(default_factory=dict)
    runtime_s: float = 0.0
    meta: dict = field(default_factory=dict)


class Fuser(Protocol):
    """Common interface for the proposed method and every baseline."""

    name: str

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult: ...
