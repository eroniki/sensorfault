"""Synthetic box-room scene + multi-sensor depth rig. Dependency-free (numpy only).

Camera convention: x right, y down, z forward (OpenCV). ``depth`` is z-depth. T_wc maps
camera -> world. The room is an axis-aligned box [0,L]; optional axis-aligned obstacle
boxes create convex edges and (with the walls) concave corners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from sensorfault.types import DepthFrame

from .noise import SensorModel


@dataclass
class BoxRoom:
    size: tuple = (5.0, 4.0, 2.5)
    obstacles: list = field(
        default_factory=lambda: [((1.8, 1.4, 0.0), (2.8, 2.4, 0.9))]
    )  # (lo, hi)

    def raycast(self, o: np.ndarray, d: np.ndarray):
        """o (3,), d (N,3) unnormalised (z-depth parametrisation). Returns t (N,), n (N,3)."""
        L = np.asarray(self.size, float)
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        bound = np.where(d > 0, L[None, :], 0.0)
        t_ax = (bound - o[None, :]) * inv
        ax = np.argmin(t_ax, axis=1)
        t = t_ax[np.arange(len(d)), ax]
        n = np.zeros_like(d)
        n[np.arange(len(d)), ax] = -np.sign(d[np.arange(len(d)), ax])
        for lo, hi in self.obstacles:
            lo = np.asarray(lo, float)
            hi = np.asarray(hi, float)
            t1 = (lo[None, :] - o[None, :]) * inv
            t2 = (hi[None, :] - o[None, :]) * inv
            tmin_ax = np.minimum(t1, t2)
            tmax_ax = np.maximum(t1, t2)
            ent_ax = np.argmax(tmin_ax, axis=1)
            tent = tmin_ax[np.arange(len(d)), ent_ax]
            tex = tmax_ax.min(axis=1)
            hit = (tex >= tent) & (tent > 1e-6) & (tent < t)
            t = np.where(hit, tent, t)
            nn = np.zeros_like(d)
            nn[np.arange(len(d)), ent_ax] = -np.sign(d[np.arange(len(d)), ent_ax])
            n = np.where(hit[:, None], nn, n)
        return t, n


def look_at(pos: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, f, pos
    return T


def intrinsics(h: int, w: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * w / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def render(room: BoxRoom, K: np.ndarray, T_wc: np.ndarray, h: int, w: int):
    """Returns clean z-depth (H,W) float32, cos(incidence) (H,W)."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dc = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1
    ).reshape(-1, 3)
    dw = dc @ T_wc[:3, :3].T
    t, n = room.raycast(T_wc[:3, 3], dw)
    cos = np.abs(np.sum(n * dw, 1)) / np.linalg.norm(dw, axis=1)
    return t.reshape(h, w).astype(np.float32), cos.reshape(h, w)


def backproject(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Valid pixels -> world points (N,3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    m = np.isfinite(depth) & (depth > 0)
    z = depth[m].astype(np.float64)
    p = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z, (v[m] - K[1, 2]) / K[1, 1] * z, z], 1)
    return p @ T_wc[:3, :3].T + T_wc[:3, 3]


@dataclass
class SyntheticConfig:
    n_sensors: int = 4
    n_frames: int = 8
    height: int = 48
    width: int = 64
    fov_deg: float = 70.0
    aim: str = "center"  # "center" (high overlap) | "outward" (no overlap)
    sensor_kinds: Sequence[str] = ("kinect_sl", "tof", "lidar", "stereo")
    sensor_params: dict = field(default_factory=dict)  # kind -> params
    pose_jitter_m: float = 0.05
    aim_jitter_deg: float = 4.0
    room_size: tuple = (5.0, 4.0, 2.5)
    gt_max_points: int = 20000

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticConfig":
        keys = cls.__dataclass_fields__.keys()
        kw = {k: v for k, v in (d or {}).items() if k in keys}
        if "room_size" in kw:
            kw["room_size"] = tuple(kw["room_size"])
        return cls(**kw)


@dataclass
class SyntheticData:
    frames: List[DepthFrame]  # noisy, true poses (faults not yet applied)
    clean_frames: List[DepthFrame]
    gt_points: np.ndarray  # visible-surface GT cloud (N,3)
    room: BoxRoom
    sensor_ids: List[str]
    meta: dict


def make_rig(cfg: SyntheticConfig, rng: np.random.Generator) -> List[np.ndarray]:
    """K base poses on an ellipse near the walls, aimed at the room centre or outward."""
    L = np.asarray(cfg.room_size, float)
    c = L / 2
    poses = []
    for k in range(cfg.n_sensors):
        a = 2 * np.pi * (k + 0.37) / cfg.n_sensors
        pos = np.array([c[0] + 0.38 * L[0] * np.cos(a), c[1] + 0.38 * L[1] * np.sin(a), 1.2])
        tgt = c.copy() if cfg.aim == "center" else pos + np.array([np.cos(a), np.sin(a), 0.0])
        tgt = tgt + np.array([0, 0, -0.3]) if cfg.aim == "center" else tgt
        poses.append(look_at(pos, tgt))
    return poses


def generate_synthetic(cfg: SyntheticConfig, seed: int) -> SyntheticData:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xA11CE]))
    room = BoxRoom(size=tuple(cfg.room_size))
    K = intrinsics(cfg.height, cfg.width, cfg.fov_deg)
    base = make_rig(cfg, rng)
    frames, clean, gt = [], [], []
    ids = [f"s{k}" for k in range(cfg.n_sensors)]
    models = []
    for k in range(cfg.n_sensors):
        kind = cfg.sensor_kinds[k % len(cfg.sensor_kinds)]
        models.append(SensorModel(kind, **cfg.sensor_params.get(kind, {})))
    for f in range(cfg.n_frames):
        for k in range(cfg.n_sensors):
            T = base[k].copy()
            T[:3, 3] += rng.normal(0, cfg.pose_jitter_m, 3)
            ang = np.deg2rad(cfg.aim_jitter_deg) * rng.normal(size=3)
            from scipy.spatial.transform import Rotation as R

            T[:3, :3] = T[:3, :3] @ R.from_rotvec(ang).as_matrix()
            d, cos = render(room, K, T, cfg.height, cfg.width)
            # noise params are specified at a 640-px-wide reference sensor; rescale focal length
            noisy = models[k](d, cos, rng, K[0, 0] * 640.0 / cfg.width)
            frames.append(DepthFrame(ids[k], f, noisy, K.copy(), T))
            clean.append(DepthFrame(ids[k], f, d, K.copy(), T))
            gt.append(backproject(d, K, T))
    gt = np.concatenate(gt)
    if len(gt) > cfg.gt_max_points:
        gt = gt[rng.choice(len(gt), cfg.gt_max_points, replace=False)]
    return SyntheticData(
        frames,
        clean,
        gt.astype(np.float32),
        room,
        ids,
        {"seed": int(seed), "kinds": [m.kind for m in models], "aim": cfg.aim},
    )
