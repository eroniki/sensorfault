"""Load a MatrixCity aerial block (float32 depth + transforms.json) at reduced resolution.

Conventions (MatrixCity README): NeRF camera frame (x right, y up, z backward), Z-depth in cm,
positions in metres; rot_mat holds a rotation scaled by 0.01 plus the camera centre.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sensorfault.data.exr_zip import read_exr

W_FULL, H_FULL = 1920, 1080


def load_block(
    root: str | Path, transforms: str | Path, step: int = 20, frames: list | None = None
):
    """Returns dict with K (3x3, reduced), and per view: R (3x3 world-from-camera), c (3,),
    depth (h, w) in metres (point-sampled every ``step`` pixels), frame_index, yaw_deg, strip key."""
    root = Path(root)
    T = json.loads(Path(transforms).read_text())
    fov = T["camera_angle_x"]
    f_full = 0.5 * W_FULL / np.tan(0.5 * fov)
    w, h = W_FULL // step, H_FULL // step
    K = np.array(
        [[f_full / step, 0, (w - 1) / 2.0], [0, f_full / step, (h - 1) / 2.0], [0, 0, 1.0]]
    )
    views = []
    for fr in T["frames"]:
        if frames is not None and fr["frame_index"] not in frames:
            continue
        M = np.array(fr["rot_mat"], dtype=float)
        R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0, keepdims=True)
        c = M[:3, 3]
        d = read_exr(root / f"{fr['frame_index']:04d}.exr", "R")
        d = d[step // 2 :: step, step // 2 :: step][:h, :w].astype(np.float64) / 100.0
        yaw = float(np.round(np.rad2deg(fr["euler"][2])))
        axis = 0 if abs(yaw % 180) == 90 else 1  # track direction varies along y for yaw -90/-270
        views.append(
            {
                "frame": fr["frame_index"],
                "R": R,
                "c": c,
                "depth": d,
                "yaw": yaw,
                "strip": (yaw, round(float(c[axis]), 1)),
                "along": float(c[1 - axis]),
            }
        )
    return K, views


def pixel_rays_world(K: np.ndarray, R: np.ndarray, h: int, w: int) -> np.ndarray:
    """World-frame ray per pixel, scaled so that the z-depth along the camera's -z axis is 1."""
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    cam = np.stack([(uu - K[0, 2]) / K[0, 0], -(vv - K[1, 2]) / K[1, 1], -np.ones_like(uu)], -1)
    return cam @ R.T


def project(K, R, c, X):
    """World points -> (u, v, z-depth) in a view."""
    Xc = (X - c) @ R
    z = -Xc[..., 2]
    u = K[0, 0] * Xc[..., 0] / z + K[0, 2]
    v = -K[1, 1] * Xc[..., 1] / z + K[1, 2]
    return u, v, z
