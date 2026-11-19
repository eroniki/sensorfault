"""Camera geometry in the MatrixCity convention (NeRF camera frame: x right, y up, z backward).

R is world-from-camera, c the camera centre; depth is z-depth along the camera's -z axis.
Same formulas as ``experiments/matrixcity_io.py``, kept here so the package does not import from
``experiments``.
"""

from __future__ import annotations

import numpy as np


def pixel_rays_world(K: np.ndarray, R: np.ndarray, h: int, w: int) -> np.ndarray:
    """(h, w, 3) world-frame ray per pixel, scaled so that the z-depth along the camera's -z is 1."""
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    cam = np.stack([(uu - K[0, 2]) / K[0, 0], -(vv - K[1, 2]) / K[1, 1], -np.ones_like(uu)], -1)
    return cam @ R.T


def backproject(K, R, c, depth: np.ndarray) -> np.ndarray:
    """(h, w) z-depth -> (h, w, 3) world points (NaN where depth is not finite)."""
    h, w = depth.shape
    return c + depth[..., None] * pixel_rays_world(K, R, h, w)


def project(K, R, c, X: np.ndarray):
    """World points (..., 3) -> (u, v, z-depth) in a view."""
    Xc = (X - c) @ R
    z = -Xc[..., 2]
    u = K[0, 0] * Xc[..., 0] / z + K[0, 2]
    v = -K[1, 1] * Xc[..., 1] / z + K[1, 2]
    return u, v, z


def normals_from_points(P: np.ndarray) -> np.ndarray:
    """(h, w, 3) back-projected points -> (h, w, 3) unit normals from central differences,
    oriented to +z (towards a nadir camera). NaN at the border and where a neighbour is missing."""
    n = np.full_like(P, np.nan)
    du = P[1:-1, 2:] - P[1:-1, :-2]
    dv = P[2:, 1:-1] - P[:-2, 1:-1]
    c = np.cross(du, dv)
    nrm = np.linalg.norm(c, axis=-1, keepdims=True)
    c = c / np.where(nrm > 0, nrm, np.nan)
    c = np.where(c[..., 2:3] < 0, -c, c)
    n[1:-1, 1:-1] = c
    return n
