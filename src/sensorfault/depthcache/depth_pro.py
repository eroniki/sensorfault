"""Depth Pro (Apple) wrapper, via HF transformers ``DepthProForDepthEstimation``.

Metric depth, no confidence output. The HF port (checkpoint ``apple/DepthPro-hf``) is used rather
than the ``depth_pro`` pip package so that one dependency (transformers) serves all wrappers.
Weights licence: the Hugging Face card shows ``apple-amlr`` / "Apple-ASCL" (Apple ML Research
licence; research use) and is not an OSI licence; read the licence text before any other use.
All heavy imports are lazy; nothing is downloaded at import time.
"""

from __future__ import annotations

import math
import os

import numpy as np

from sensorfault.depthcache.base import DepthEstimator, DepthResult

DEFAULT_CHECKPOINT = "apple/DepthPro-hf"
WEIGHTS_LICENCE = (
    "apple-amlr (Apple ML Research licence; HF card also says Apple-ASCL); research use"
)


def resize_bilinear(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2D array to ``shape`` (H, W); float32 out. Pure numpy."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape == tuple(shape):
        return a
    h, w = a.shape
    H, W = shape
    ys = (np.arange(H, dtype=np.float32) + 0.5) * h / H - 0.5
    xs = (np.arange(W, dtype=np.float32) + 0.5) * w / W - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


class DepthProEstimator:
    model_id = "depth_pro"
    kind = "metric_depth"
    has_confidence = False

    def __init__(self, checkpoint: str | None = None, half: bool = True) -> None:
        # Name on the HF hub or a local directory; env var lets configs override without code.
        self.checkpoint = checkpoint or os.environ.get(
            "SENSORFAULT_DEPTH_PRO_CKPT", DEFAULT_CHECKPOINT
        )
        self.half = half
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self, device: str = "cuda", **kw) -> None:
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor

        self._device = device
        dtype = torch.float16 if (self.half and device.startswith("cuda")) else torch.float32
        self._processor = DepthProImageProcessor.from_pretrained(self.checkpoint, **kw)
        self._model = (
            DepthProForDepthEstimation.from_pretrained(self.checkpoint, torch_dtype=dtype, **kw)
            .to(device)
            .eval()
        )

    def _infer(self, rgb: np.ndarray) -> tuple[np.ndarray, float | None]:
        """Run the network (batch 1). Returns (depth in metres at any size, predicted focal px)."""
        import torch

        h, w = rgb.shape[:2]
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        inputs = {
            k: v.to(self._model.dtype) if v.is_floating_point() else v for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self._model(**inputs)
        post = self._processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        depth = post["predicted_depth"].float().cpu().numpy()
        f = post.get("focal_length")
        return depth, (None if f is None else float(f))

    def predict(self, rgb: np.ndarray, focal_px: float | None = None) -> DepthResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be (H, W, 3) uint8")
        h, w = rgb.shape[:2]
        depth, f_pred = self._infer(rgb)
        depth = resize_bilinear(np.squeeze(depth), (h, w))
        extra: dict = {"focal_source": "predicted"}
        f_used = f_pred
        if focal_px is not None:
            if not (focal_px > 0 and math.isfinite(focal_px)):
                raise ValueError("focal_px must be positive and finite")
            # Depth scales linearly with the focal length assumed (canonical inverse depth * W/f).
            if f_pred is not None and f_pred > 0:
                depth = depth * np.float32(focal_px / f_pred)
                extra["focal_pred_px"] = f_pred
                extra["focal_source"] = "given_rescaled"
            else:
                extra["focal_source"] = "given_not_applied"  # backend gave no focal to rescale from
            f_used = float(focal_px)
        return DepthResult(
            depth=depth.astype(np.float32),
            kind=self.kind,
            confidence=None,
            focal_px=f_used,
            extra=extra,
        )

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
            "weights_licence": WEIGHTS_LICENCE,
            "versions": versions,
        }


def build(checkpoint: str | None = None, half: bool = True) -> DepthEstimator:
    return DepthProEstimator(checkpoint=checkpoint, half=half)
