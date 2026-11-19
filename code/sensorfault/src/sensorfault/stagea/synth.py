"""Synthetic aerial block: a height field with buildings and water, nadir cameras in strips.

Views use the conventions of ``experiments/matrixcity_io.py`` (NeRF camera frame: x right, y up,
z backward; R world-from-camera; depth is z-depth in metres), so the Stage A pipeline runs on them
unchanged. Used by the tests and by the pipeline control without monocular depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorfault.stagea.geometry import pixel_rays_world


@dataclass
class HeightField:
    z: np.ndarray  # (ny, nx) heights in metres
    x0: float
    y0: float
    res: float

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        i = np.clip(np.floor((y - self.y0) / self.res).astype(int), 0, self.z.shape[0] - 1)
        j = np.clip(np.floor((x - self.x0) / self.res).astype(int), 0, self.z.shape[1] - 1)
        return self.z[i, j]


def make_heightfield(
    size: float = 600.0, res: float = 0.5, seed: int = 0, n_buildings: int = 60
) -> HeightField:
    """Gently rolling ground, box buildings 8-30 m, and a water band at -5 m."""
    rng = np.random.default_rng(seed)
    n = int(size / res)
    x0 = y0 = -size / 2
    xs = x0 + res * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(xs, xs)
    z = 1.5 * np.sin(X / 90.0) * np.cos(Y / 70.0) + 0.5 * np.sin(X / 23.0 + Y / 31.0)
    water = np.abs(Y - 0.35 * X - 120.0) < 40.0
    z[water] = -5.0
    for _ in range(n_buildings):
        cx, cy = rng.uniform(x0 + 20, -x0 - 20, 2)
        if np.abs(cy - 0.35 * cx - 120.0) < 60.0:
            continue
        wx, wy = rng.uniform(10, 40, 2)
        h = rng.uniform(8, 30)
        m = (np.abs(X - cx) < wx / 2) & (np.abs(Y - cy) < wy / 2)
        z[m] = np.maximum(z[m], h + rng.uniform(-0.3, 0.3))
    return HeightField(z, x0, y0, res)


def yaw_R(yaw_deg: float) -> np.ndarray:
    """Nadir camera (looking down world -z) rotated by yaw about world z."""
    a = np.deg2rad(yaw_deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def raycast(
    hf: HeightField, c: np.ndarray, rays: np.ndarray, zmin: float, zmax: float, dt: float = 0.25
) -> np.ndarray:
    """z-depth t with c + t r on the height field (rays scaled so that r_z = -1 for nadir)."""
    shp = rays.shape[:-1]
    r = rays.reshape(-1, 3)
    rz = -r[:, 2]
    t0 = (c[2] - zmax) / rz
    t1 = (c[2] - zmin) / rz
    t = t0.copy()
    out = np.full(len(r), np.nan)
    prev_gap = None
    nsteps = int(np.ceil(np.max(t1 - t0) / dt)) + 2
    for _ in range(nsteps):
        X = c + t[:, None] * r
        gap = X[:, 2] - hf(X[:, 0], X[:, 1])
        if prev_gap is not None:
            hit = np.isnan(out) & (gap <= 0) & (prev_gap > 0)
            # linear refinement between t - dt and t
            frac = prev_gap[hit] / (prev_gap[hit] - gap[hit])
            out[hit] = t[hit] - dt + frac * dt
        prev_gap = gap
        t = t + dt
    return out.reshape(shp)


def make_block(
    seed: int = 0,
    altitude: float = 200.0,
    fov_x_deg: float = 60.0,
    w: int = 96,
    h: int = 54,
    spacing: float = 40.0,
    strip_gap: float = 70.0,
    n_strips: int = 3,
    frames_per_strip: int = 9,
    cross: bool = True,
):
    """Returns (K, views, hf) with views as in matrixcity_io.load_block (plus exact ``depth``)."""
    hf = make_heightfield(seed=seed)
    f = 0.5 * w / np.tan(0.5 * np.deg2rad(fov_x_deg))
    K = np.array([[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]])
    views = []
    frame = 0
    plans = [(0.0, 1)] + ([(90.0, 0)] if cross else [])
    for yaw, axis in plans:
        for s in range(n_strips):
            cross_c = (s - (n_strips - 1) / 2) * strip_gap
            for k in range(frames_per_strip):
                along = (k - (frames_per_strip - 1) / 2) * spacing
                c = np.zeros(3)
                c[axis] = cross_c
                c[1 - axis] = along
                c[2] = altitude
                R = yaw_R(yaw)
                rays = pixel_rays_world(K, R, h, w)
                d = raycast(hf, c, rays, float(hf.z.min()) - 1, float(hf.z.max()) + 1)
                views.append(
                    {
                        "frame": frame,
                        "R": R,
                        "c": c,
                        "depth": d,
                        "yaw": yaw,
                        "strip": (yaw, round(float(c[axis]), 1)),
                        "along": float(c[1 - axis]),
                    }
                )
                frame += 1
    return K, views, hf


def tilt_R(yaw_deg: float, tilt_deg: float) -> np.ndarray:
    """Camera tilted ``tilt_deg`` off nadir about its own x axis, then rotated by yaw."""
    t = np.deg2rad(tilt_deg)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    return yaw_R(yaw_deg) @ Rx


def make_oblique_block(
    seed: int = 0,
    altitude: float = 200.0,
    tilt_deg: float = 30.0,
    headings=(-90.0, -180.0, -270.0, -360.0),
    fov_x_deg: float = 60.0,
    w: int = 96,
    h: int = 54,
    spacing: float = 35.0,
    nx: int = 5,
    ny: int = 4,
):
    """Block-3-like geometry: a grid of camera centres, one oblique frame per heading at each."""
    hf = make_heightfield(seed=seed)
    f = 0.5 * w / np.tan(0.5 * np.deg2rad(fov_x_deg))
    K = np.array([[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]])
    views = []
    frame = 0
    for yaw in headings:
        axis = 0 if abs(yaw % 180) == 90 else 1
        for ix in range(nx):
            for iy in range(ny):
                c = np.array(
                    [(ix - (nx - 1) / 2) * spacing, (iy - (ny - 1) / 2) * spacing, altitude]
                )
                R = tilt_R(yaw, tilt_deg)
                d = raycast(
                    hf,
                    c,
                    pixel_rays_world(K, R, h, w),
                    float(hf.z.min()) - 1,
                    float(hf.z.max()) + 1,
                )
                views.append(
                    {
                        "frame": frame,
                        "R": R,
                        "c": c,
                        "depth": d,
                        "yaw": yaw,
                        "strip": (yaw, round(float(c[axis]), 1)),
                        "along": float(c[1 - axis]),
                    }
                )
                frame += 1
    return K, views, hf
