"""Depth Anything V2 wrapper (HF transformers ``AutoModelForDepthEstimation``).

Default: relative checkpoints, whose ``predicted_depth`` is affine-invariant disparity (inverse
depth up to scale and shift); returned unmodified with kind ``relative_inverse_depth``. It ignores
focal_px. Metric fine-tunes (``...-Metric-Indoor/Outdoor-hf``, Hypersim/VKITTI) output metres and
are selected by passing a checkpoint name containing ``Metric`` (kind becomes ``metric_depth``).
Licences: Small is Apache-2.0; Base, Large and Giant are CC BY-NC 4.0 (non-commercial).
"""

from __future__ import annotations

import os

import numpy as np

from sensorfault.depthcache.base import DepthEstimator, DepthResult
from sensorfault.depthcache.depth_pro import resize_bilinear

DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Large-hf"


def weights_licence(checkpoint: str) -> str:
    name = checkpoint.lower()
    if "small" in name:
        return "Apache-2.0"
    if any(s in name for s in ("large", "base", "giant")):
        return "CC-BY-NC-4.0 (non-commercial)"
    return "unknown (local path); Small is Apache-2.0, Base/Large/Giant CC-BY-NC-4.0"


class DepthAnythingV2Estimator:
    model_id = "depth_anything_v2"
    has_confidence = False

    def __init__(self, checkpoint: str | None = None, half: bool = True) -> None:
        self.checkpoint = checkpoint or os.environ.get("SENSORFAULT_DAV2_CKPT", DEFAULT_CHECKPOINT)
        self.half = half
        self.kind = (
            "metric_depth" if "metric" in self.checkpoint.lower() else "relative_inverse_depth"
        )
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self, device: str = "cuda", **kw) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._device = device
        dtype = torch.float16 if (self.half and device.startswith("cuda")) else torch.float32
        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint, **kw)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(self.checkpoint, torch_dtype=dtype, **kw)
            .to(device)
            .eval()
        )

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        """Run the network (batch 1); returns the raw (h, w) output at network resolution."""
        import torch

        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        inputs = {
            k: v.to(self._model.dtype) if v.is_floating_point() else v for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self._model(**inputs)
        return out.predicted_depth[0].float().cpu().numpy()

    def predict(self, rgb: np.ndarray, focal_px: float | None = None) -> DepthResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be (H, W, 3) uint8")
        h, w = rgb.shape[:2]
        depth = resize_bilinear(np.squeeze(self._infer(rgb)), (h, w))
        # focal_px is not used by DAv2 (no camera model); recorded only.
        extra = {"focal_ignored_px": float(focal_px)} if focal_px is not None else {}
        return DepthResult(depth=depth, kind=self.kind, confidence=None, focal_px=None, extra=extra)

    def info(self) -> dict:
        versions = {}
        for pkg in ("torch", "transformers"):
            try:
                from importlib.metadata import version

                versions[pkg] = version(pkg)
            except Exception:
                versions[pkg] = None
        return {
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "kind": self.kind,
            "fp16": self.half,
            "batch_size": 1,
            "weights_licence": weights_licence(self.checkpoint),
            "versions": versions,
        }


def build(checkpoint: str | None = None, half: bool = True) -> DepthEstimator:
    return DepthAnythingV2Estimator(checkpoint=checkpoint, half=half)
