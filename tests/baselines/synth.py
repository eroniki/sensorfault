"""Synthetic planar scene helpers."""

import numpy as np

from sensorfault.types import DepthFrame

H = W = 64
K = np.array([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]])


def pose(tx=0.0, ty=0.0, tz=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T[:3, 3] = [tx, ty, tz]
    return T


def plane_depth(T_wc, n=(0, 0, 1.0), d=2.0, shape=(H, W), Kc=K):
    """z-depth of the plane n.x = d (world) seen from camera T_wc."""
    h, w = shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dirs = np.stack(
        [(u - Kc[0, 2]) / Kc[0, 0], (v - Kc[1, 2]) / Kc[1, 1], np.ones_like(u, float)], -1
    )
    n = np.asarray(n, float)
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    den = (dirs @ R.T) @ n
    z = (d - n @ t) / den
    return np.where(z > 0, z, np.nan)


def make_frames(
    biases, sigma=0.005, seed=0, poses=None, frame_id=0, n=(0, 0, 1.0), d=2.0, extra=None
):
    """One frame per sensor; sensor s{i} has constant depth bias biases[i]."""
    rng = np.random.default_rng(seed)
    out = []
    for i, b in enumerate(biases):
        T = poses[i] if poses is not None else np.eye(4)
        z = plane_depth(T, n, d)
        z = z + b + rng.normal(0, sigma, z.shape)
        out.append(DepthFrame(f"s{i}", frame_id, z.astype(np.float32), K.copy(), T))
    return out
