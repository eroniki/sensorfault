"""Metric3D v2 (YvanYin/Metric3D) wrapper: metric depth plus a raw confidence map.

Backend: the repository's ``torch.hub`` entry points (``metric3d_vit_small`` / ``_large`` /
``_giant2``; the ViT models are the v2 models). The hub code is loaded from a local checkout
(``hub_source``, recommended; no network) or from GitHub (``yvanyin/metric3d``). Weights come from
a local ``.pth`` (``checkpoint_path``) or, if none is given, from the hub's Hugging Face URL
(``JUGGHM/Metric3D``) on first ``load`` only. Nothing is imported or downloaded at import time.

Canonical camera space and de-canonicalisation
----------------------------------------------
The network predicts depth for a canonical camera of focal length ``F_CANON = 1000`` px. For an
image whose focal length is ``f`` px (after any resize, the focal scales with the resize factor)
the metric depth is ``d_metric = d_canonical * f / F_CANON``, which is what the repository's
``hubconf.py`` demo does (``canonical_to_real_scale = intrinsic[0] / 1000.0``). Here the image is
resized (aspect ratio kept) by ``s = min(616 / H, 1064 / W)`` to fit the ViT input size, so the
focal used is ``f_net = s * focal_px`` and ``d_metric = d_net * f_net / 1000``. The result is
clamped to [0, 300] m as in the demo. A dataset focal is therefore required; without it the metric
scale is undefined, so ``predict`` raises unless ``assumed_focal_px`` was set (then recorded as
``focal_source = "assumed"``).

Confidence
----------
``model.inference`` returns ``(pred_depth, confidence, output_dict)``. The README documents no
semantics for ``confidence`` (it only documents the normal confidence). The training code
(``training/mono/model/losses/ConfidenceLoss.py``) regresses ``1 - |pred - gt| / gt`` where the
relative error is below 1, so higher = more reliable and the value is roughly ``1 - AbsRel``. It is
returned unmodified (no de-canonicalisation: it is dimensionless), resampled to the input size.
UNVERIFIED: that the released checkpoints were trained with that loss, and its calibration.

Licence: code BSD-2-Clause (repository README/LICENSE). The Hugging Face weights repo
``JUGGHM/Metric3D`` carries no model card or licence text (checked 2026-11-17); the weights
terms are therefore unstated and should be treated as unconfirmed (the README points commercial
enquiries to the authors).
"""

from __future__ import annotations

import math
import os

import numpy as np

from sensorfault.depthcache.base import DepthEstimator, DepthResult
from sensorfault.depthcache.depth_pro import resize_bilinear

F_CANON = 1000.0  # focal length of the canonical camera (px)
INPUT_HW = (616, 1064)  # ViT input size (H, W); the ConvNeXt models use (544, 1216)
PAD_VALUE = (123.675, 116.28, 103.53)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
MAX_DEPTH_M = 300.0
HUB_ENTRY = {
    "vit_small": "metric3d_vit_small",
    "vit_large": "metric3d_vit_large",
    "vit_giant2": "metric3d_vit_giant2",
}
DEFAULT_VARIANT = "vit_large"
DEFAULT_HUB_SOURCE = "yvanyin/metric3d"
WEIGHTS_LICENCE = (
    "code BSD-2-Clause; weights licence unstated (HF JUGGHM/Metric3D has no model card), "
    "treat as unconfirmed"
)
CONFIDENCE_SEMANTICS = (
    "raw head output; training loss targets 1-|pred-gt|/gt (higher = more reliable); "
    "calibration unverified"
)


def fit_geometry(
    h: int, w: int, input_hw: tuple[int, int] = INPUT_HW
) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """Keep-ratio resize factor, resized (H, W) and padding (top, bottom, left, right)."""
    s = min(input_hw[0] / h, input_hw[1] / w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    nh, nw = min(nh, input_hw[0]), min(nw, input_hw[1])
    ph, pw = input_hw[0] - nh, input_hw[1] - nw
    return s, (nh, nw), (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2)


def decanonicalise(
    depth_canon: np.ndarray, focal_net_px: float, max_depth_m: float = MAX_DEPTH_M
) -> np.ndarray:
    """Canonical-camera depth -> metric depth for a camera of focal ``focal_net_px`` (net input px).

    Clamped to [0, max_depth_m]; the upstream demo uses 300 m, which is below most of a 300 m-altitude
    oblique aerial view (MatrixCity block 3: median GT depth ~340 m), so aerial runs may raise it.
    """
    return np.clip(depth_canon * np.float32(focal_net_px / F_CANON), 0.0, max_depth_m).astype(
        np.float32
    )


def preprocess(rgb: np.ndarray, input_hw: tuple[int, int] = INPUT_HW):
    """uint8 (H, W, 3) -> normalised float32 (3, Hin, Win), scale s, pad (t, b, l, r)."""
    h, w = rgb.shape[:2]
    s, (nh, nw), pad = fit_geometry(h, w, input_hw)
    chans = [resize_bilinear(rgb[..., c].astype(np.float32), (nh, nw)) for c in range(3)]
    x = np.stack(chans, axis=0)
    out = np.empty((3, *input_hw), dtype=np.float32)
    for c in range(3):
        out[c] = PAD_VALUE[c]
    out[:, pad[0] : pad[0] + nh, pad[2] : pad[2] + nw] = x
    out = (out - MEAN[:, None, None]) / STD[:, None, None]
    return out.astype(np.float32), s, pad


def unpad(a: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
    return a[pad[0] : a.shape[0] - pad[1], pad[2] : a.shape[1] - pad[3]]


class Metric3DV2Estimator:
    model_id = "metric3d_v2"
    kind = "metric_depth"
    has_confidence = True

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        checkpoint_path: str | None = None,
        hub_source: str | None = None,
        assumed_focal_px: float | None = None,
        max_depth_m: float = MAX_DEPTH_M,
    ) -> None:
        if variant not in HUB_ENTRY:
            raise ValueError(f"variant must be one of {sorted(HUB_ENTRY)}")
        self.variant = variant
        self.checkpoint_path = checkpoint_path or os.environ.get("SENSORFAULT_METRIC3D_CKPT")
        self.hub_source = hub_source or os.environ.get(
            "SENSORFAULT_METRIC3D_HUB", DEFAULT_HUB_SOURCE
        )
        self.assumed_focal_px = assumed_focal_px
        self.max_depth_m = float(max_depth_m)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", **kw) -> None:
        import torch

        source = "local" if os.path.isdir(self.hub_source) else "github"
        own_weights = self.checkpoint_path is not None
        model = torch.hub.load(
            self.hub_source,
            HUB_ENTRY[self.variant],
            source=source,
            pretrain=not own_weights,
            trust_repo=True,
            **kw,
        )
        if own_weights:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(state.get("model_state_dict", state), strict=False)
        self._device = device
        self._model = model.to(device).eval()

    def _run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Network call on a normalised (3, Hin, Win) array -> canonical depth, confidence."""
        import torch

        t = torch.from_numpy(x)[None].to(self._device)
        with torch.no_grad():
            depth, conf, _ = self._model.inference({"input": t})
        d = depth.squeeze().float().cpu().numpy()
        c = None if conf is None else conf.squeeze().float().cpu().numpy()
        return d, c

    def predict(self, rgb: np.ndarray, focal_px: float | None = None) -> DepthResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be (H, W, 3) uint8")
        extra: dict = {"focal_source": "given"}
        f = focal_px
        if f is None:
            f = self.assumed_focal_px
            extra["focal_source"] = "assumed"
        if f is None:
            raise ValueError("Metric3D v2 needs focal_px (canonical-camera rescaling)")
        if not (f > 0 and math.isfinite(f)):
            raise ValueError("focal_px must be positive and finite")
        h, w = rgb.shape[:2]
        x, s, pad = preprocess(rgb)
        d_net, c_net = self._run(x)
        depth = resize_bilinear(unpad(np.squeeze(d_net), pad), (h, w))
        depth = decanonicalise(depth, s * f, self.max_depth_m)
        conf = None
        if c_net is not None:
            conf = resize_bilinear(unpad(np.squeeze(c_net), pad), (h, w))
        extra.update(resize_scale=float(s), focal_net_px=float(s * f), pad=list(pad))
        return DepthResult(
            depth=depth, kind=self.kind, confidence=conf, focal_px=float(f), extra=extra
        )

    def info(self) -> dict:
        from importlib.metadata import version

        versions = {}
        for pkg in ("torch", "mmengine", "mmcv"):
            try:
                versions[pkg] = version(pkg)
            except Exception:
                versions[pkg] = None
        return {
            "model_id": self.model_id,
            "variant": self.variant,
            "checkpoint": self.checkpoint_path or f"hub:{HUB_ENTRY[self.variant]}",
            "hub_source": self.hub_source,
            "kind": self.kind,
            "canonical_focal_px": F_CANON,
            "input_hw": list(INPUT_HW),
            "max_depth_m": self.max_depth_m,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "weights_licence": WEIGHTS_LICENCE,
            "versions": versions,
        }


def build(
    variant: str = DEFAULT_VARIANT,
    checkpoint_path: str | None = None,
    hub_source: str | None = None,
    assumed_focal_px: float | None = None,
    max_depth_m: float = MAX_DEPTH_M,
) -> DepthEstimator:
    return Metric3DV2Estimator(variant, checkpoint_path, hub_source, assumed_focal_px, max_depth_m)
