"""Minimal directory-cache ``DepthSource``.

Layout (the real cache may differ; adapt here)::

    <root>/<scene>/<model>/<view_id>.npz  (depthcache layout: pass scene='<dataset>/<scene>');
    arrays: depth (H, W) metres; confidence (H, W) optional (legacy key 'conf' also read)
    <root>/<scene>/transforms.json            {"views": {<view_id>: {"K": 3x3, "T_wc": 4x4}}}
                                              or {"K": 3x3 (shared), "views": {<id>: {"T_wc": ...}}}

``T_wc`` is camera-to-world. Views are ordered by id.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensorfault.types import DepthFrame


class DirectoryDepthSource:
    def __init__(self, root: str | Path, transforms_name: str = "transforms.json"):
        self.root = Path(root)
        self.transforms_name = transforms_name

    def frames(self, scene: str, model: str) -> list[DepthFrame]:
        sdir = self.root / scene
        meta = json.loads((sdir / self.transforms_name).read_text())
        shared_K = meta.get("K")
        views = meta["views"]
        out: list[DepthFrame] = []
        for i, f in enumerate(sorted((sdir / model).glob("*.npz"))):
            vid = f.stem
            if vid not in views:
                raise KeyError(f"{vid}: no pose/intrinsics in {sdir / self.transforms_name}")
            v = views[vid]
            with np.load(f) as z:
                depth = np.asarray(z["depth"], dtype=np.float32)
                ck = (
                    "confidence"
                    if "confidence" in z.files
                    else ("conf" if "conf" in z.files else None)
                )
                conf = np.asarray(z[ck], dtype=np.float32) if ck else None
            out.append(
                DepthFrame(
                    sensor_id=vid,
                    frame_id=i,
                    depth=depth,
                    K=np.asarray(v.get("K", shared_K), dtype=np.float64),
                    T_wc=np.asarray(v["T_wc"], dtype=np.float64),
                    conf=conf,
                )
            )
        return out


def load_depth_frames(root: str | Path, scene: str, model: str) -> list[DepthFrame]:
    """Convenience: ``DirectoryDepthSource(root).frames(scene, model)``."""
    return DirectoryDepthSource(root).frames(scene, model)
