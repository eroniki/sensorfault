"""Interface shared by all depth-cache model wrappers.

Wrappers import their heavy dependencies (torch, model packages) lazily inside ``load`` so that this
package imports on a CPU-only machine without them. Nothing here downloads weights at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class DepthResult:
    """One view's prediction, at the input image resolution.

    depth: (H, W) float32, metres for metric models. For DAv2 (relative disparity/inverse depth)
        ``kind`` says so and ``depth`` holds the raw model output, unmodified.
    confidence: optional (H, W) float32 (models with a confidence output only), else None.
    focal_px: focal length in pixels used or predicted by the model, if any.
    """

    depth: np.ndarray
    kind: str  # "metric_depth" | "relative_inverse_depth"
    confidence: np.ndarray | None = None
    focal_px: float | None = None
    extra: dict = field(default_factory=dict)  # small JSON-serialisable items


class DepthEstimator(Protocol):
    model_id: str  # one of: depth_pro, metric3d_v2, unidepth_v2, depth_anything_v2
    kind: str
    has_confidence: bool

    def load(self, device: str = "cuda", **kw) -> None:
        """Load weights (from a local path / HF cache; never at import)."""

    def predict(self, rgb: np.ndarray, focal_px: float | None = None) -> DepthResult:
        """rgb: (H, W, 3) uint8. focal_px: dataset-calibrated focal in pixels if known."""

    def info(self) -> dict:
        """JSON-serialisable: checkpoint name/variant, package versions, weights licence string."""
