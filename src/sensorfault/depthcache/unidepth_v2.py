"""UniDepthV2 (lpiccinelli-eth/UniDepth) wrapper: metric depth, confidence and intrinsics.

Backend: ``unidepth.models.UniDepthV2.from_pretrained`` (Hugging Face ids
``lpiccinelli/unidepth-v2-vits14|vitb14|vitl14``) or a local directory/hub id given as
``checkpoint``. Imports and downloads happen only in ``load``.

``model.infer(rgb_uint8_chw, camera)`` returns depth (1,1,H,W) at the input resolution, a
confidence map, ``intrinsics`` (1,3,3) and points. If a focal is known, a pinhole camera
``K = [[f,0,W/2],[0,f,H/2],[0,0,1]]`` is passed (principal point assumed at the image centre; only
the focal is dataset-calibrated here) and the model then uses it instead of its own estimate;
the focal it would predict on its own is not computed in that case (set ``use_known_focal=False``
to ignore the dataset focal and record the predicted one in ``focal_px``).
``extra['focal_pred_px']`` is the intrinsics returned by the model (equal to the given one when a
camera was passed).

Confidence semantics (NOTE: differs from the name). In the released code the output is
``exp(clip(log_conf, -8, 8))`` and is trained with ``unidepth/ops/losses/confidence.py`` to make
``log(confidence)`` regress the norm of the (median-aligned) depth error, so it reads as a
predicted error magnitude, i.e. HIGHER = LESS reliable. The README only lists "Confidence output"
without semantics. Returned unmodified; downstream code must not assume higher is better.
UNVERIFIED empirically: monotone relation to real error and its units.

Licence: code CC BY-NC 4.0 (repository README/LICENSE). The Hugging Face weights cards carry no
licence field (checked 2026-11-17), so the weights are treated as CC BY-NC 4.0 (non-commercial),
which is unconfirmed.
"""

from __future__ import annotations

import math
import os

import numpy as np

from sensorfault.depthcache.base import DepthEstimator, DepthResult
from sensorfault.depthcache.depth_pro import resize_bilinear

VARIANTS = ("vits14", "vitb14", "vitl14")
DEFAULT_VARIANT = "vitl14"
WEIGHTS_LICENCE = (
    "code CC BY-NC 4.0; HF weights cards state no licence, assumed CC BY-NC 4.0 (unconfirmed)"
)
CONFIDENCE_SEMANTICS = (
    "exp(log-confidence) trained to regress depth-error magnitude: higher = LESS reliable "
    "(per training loss in repo); empirical relation unverified"
)


def pinhole_K(focal_px: float, h: int, w: int) -> np.ndarray:
    return np.array(
        [[focal_px, 0.0, w / 2.0], [0.0, focal_px, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


class UniDepthV2Estimator:
    model_id = "unidepth_v2"
    kind = "metric_depth"
    has_confidence = True

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        checkpoint: str | None = None,
        use_known_focal: bool = True,
        resolution_level: int | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.checkpoint = (
            checkpoint
            or os.environ.get("SENSORFAULT_UNIDEPTH_CKPT")
            or f"lpiccinelli/unidepth-v2-{variant}"
        )
        self.use_known_focal = use_known_focal
        self.resolution_level = resolution_level
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", **kw) -> None:
        from unidepth.models import UniDepthV2

        model = UniDepthV2.from_pretrained(self.checkpoint, **kw)
        if self.resolution_level is not None:
            model.resolution_level = self.resolution_level
        self._device = device
        self._model = model.to(device).eval()

    def _run(self, rgb: np.ndarray, K: np.ndarray | None):
        """Network call -> (depth (h,w), confidence (h,w) or None, intrinsics 3x3)."""
        import torch

        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        cam = None if K is None else torch.from_numpy(K)
        with torch.no_grad():
            out = self._model.infer(t, cam)
        d = out["depth"].squeeze().float().cpu().numpy()
        c = out.get("confidence")
        c = None if c is None else c.squeeze().float().cpu().numpy()
        k = out["intrinsics"].reshape(3, 3).float().cpu().numpy()
        return d, c, k

    def predict(self, rgb: np.ndarray, focal_px: float | None = None) -> DepthResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be (H, W, 3) uint8")
        h, w = rgb.shape[:2]
        K = None
        if focal_px is not None and self.use_known_focal:
            if not (focal_px > 0 and math.isfinite(focal_px)):
                raise ValueError("focal_px must be positive and finite")
            K = pinhole_K(float(focal_px), h, w)
        d, c, k = self._run(rgb, K)
        depth = resize_bilinear(d, (h, w))
        conf = None if c is None else resize_bilinear(c, (h, w))
        f_pred = float(k[0, 0])
        extra = {
            "focal_source": "given" if K is not None else "predicted",
            "focal_pred_px": f_pred,
            "focal_pred_fy_px": float(k[1, 1]),
            "principal_point_px": [float(k[0, 2]), float(k[1, 2])],
        }
        f_used = float(focal_px) if K is not None else f_pred
        return DepthResult(
            depth=depth, kind=self.kind, confidence=conf, focal_px=f_used, extra=extra
        )

    def info(self) -> dict:
        from importlib.metadata import version

        versions = {}
        for pkg in ("torch", "unidepth", "huggingface_hub"):
            try:
                versions[pkg] = version(pkg)
            except Exception:
                versions[pkg] = None
        return {
            "model_id": self.model_id,
            "variant": self.variant,
            "checkpoint": self.checkpoint,
            "kind": self.kind,
            "use_known_focal": self.use_known_focal,
            "resolution_level": self.resolution_level,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "weights_licence": WEIGHTS_LICENCE,
            "versions": versions,
        }


def build(
    variant: str = DEFAULT_VARIANT,
    checkpoint: str | None = None,
    use_known_focal: bool = True,
    resolution_level: int | None = None,
) -> DepthEstimator:
    return UniDepthV2Estimator(variant, checkpoint, use_known_focal, resolution_level)
