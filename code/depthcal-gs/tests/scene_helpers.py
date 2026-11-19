"""Shared synthetic scene for the leave-view-out tests: a room (floor y=1, back wall z=5) seen from 5 cameras on a line."""
import numpy as np

from dcal.surface import ViewData


def look_at(eye, target=np.array([0.0, 0.0, 3.0])):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, -1, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])           # rows: x right, y down, z forward
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


def scene_depth(K, T_cw, H, W):
    """Analytic z-depth of the room."""
    v, u = np.mgrid[0:H, 0:W]
    dirs = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    T_wc = np.linalg.inv(T_cw)
    o = T_wc[:3, 3]
    dw = dirs @ T_wc[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dw[..., 1] > 1e-9, (1.0 - o[1]) / dw[..., 1], np.inf)
        t_wall = np.where(dw[..., 2] > 1e-9, (5.0 - o[2]) / dw[..., 2], np.inf)
    t = np.minimum(t_floor, t_wall)
    return np.where(np.isfinite(t), t, 10.0)     # camera z-depth equals the ray parameter (unit z direction)


def make_views(H, W, focal=None, n=5, seed=0):
    focal = focal or float(W)
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
    rng = np.random.default_rng(seed)
    out = {}
    for i, x in enumerate(np.linspace(-1.5, 1.5, n)):
        T = look_at(np.array([x, 0.0, 0.0]))
        out[i] = ViewData(i, rng.random((H, W, 3)), scene_depth(K, T, H, W), K, T)
    return out
