"""Trajectory prior for posed survey images: a uniform cubic B-spline in time (design v3.3).

Aerial survey flights follow lawnmower patterns: long straight strips joined by turns. The camera
trajectory is a uniform cubic B-spline over time whose control points are fitted to GNSS/IMU (or the
dataset poses) and then refined by differentiable pose-graph optimisation with render-and-compare
depth terms (correspondence-free). Positions use the standard basis; rotations use the cumulative
form on SO(3) (Kim et al.; Sommer et al., CVPR 2020). The tested view's pose never uses its own
depth: its depth term is subtracted from the control-point information before Sigma_q is formed,
and Sigma_q = B(t) Sigma_ctrl B(t)^T carries the pose uncertainty into the test.
"""

from __future__ import annotations

import numpy as np

from sensorfault.eval.gaussian_uncertainty import so3_exp

_M = (
    np.array(
        [[1.0, 4.0, 1.0, 0.0], [-3.0, 0.0, 3.0, 0.0], [3.0, -6.0, 3.0, 0.0], [-1.0, 3.0, -3.0, 1.0]]
    )
    / 6.0
)


def basis(u: float) -> np.ndarray:
    """Uniform cubic B-spline weights of the four active control points at local time u in [0,1)."""
    return np.array([1.0, u, u * u, u**3]) @ _M


def cumulative_basis(u: float) -> np.ndarray:
    """Cumulative weights B~_m(u) = sum_{l>=m} B_l(u), m = 0..3 (B~_0 = 1)."""
    return np.cumsum(basis(u)[::-1])[::-1]


def segment(t: float, t0: float, dt: float, n_ctrl: int) -> tuple[int, float]:
    """Index of the first active control point and the local time u for time t."""
    s = (t - t0) / dt
    i = int(np.clip(np.floor(s), 0, n_ctrl - 4))
    return i, float(s - i)


def position_row(t: float, t0: float, dt: float, n_ctrl: int) -> np.ndarray:
    """Row of the linear map control points -> position at time t (length n_ctrl)."""
    i, u = segment(t, t0, dt, n_ctrl)
    row = np.zeros(n_ctrl)
    row[i : i + 4] = basis(u)
    return row


def eval_position(ctrl: np.ndarray, t: float, t0: float, dt: float) -> np.ndarray:
    return position_row(t, t0, dt, len(ctrl)) @ ctrl


def second_difference(n_ctrl: int) -> np.ndarray:
    """Second-difference operator on the control points (smoothness prior)."""
    D = np.zeros((n_ctrl - 2, n_ctrl))
    for k in range(n_ctrl - 2):
        D[k, k : k + 3] = [1.0, -2.0, 1.0]
    return D


def fit_positions(
    times: np.ndarray,
    obs: np.ndarray,
    sigma_obs: float,
    n_ctrl: int,
    t0: float,
    dt: float,
    smooth: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares of control points to GNSS positions with a smoothness prior.

    Returns (ctrl (n_ctrl, 3), information (n_ctrl, n_ctrl), shared by the three axes)."""
    A = np.stack([position_row(t, t0, dt, n_ctrl) for t in times])
    D = second_difference(n_ctrl)
    L = A.T @ A / sigma_obs**2 + smooth * D.T @ D
    ctrl = np.linalg.solve(L, A.T @ obs / sigma_obs**2)
    return ctrl, L


def position_cov(L_ctrl: np.ndarray, t: float, t0: float, dt: float) -> float:
    """Per-axis variance of the position at time t: B(t) L^-1 B(t)^T."""
    row = position_row(t, t0, dt, len(L_ctrl))
    return float(row @ np.linalg.solve(L_ctrl, row))


def log_so3(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return w / 2.0
    return w * th / (2.0 * np.sin(th))


def eval_rotation(Rctrl: list, t: float, t0: float, dt: float) -> np.ndarray:
    """Cumulative cubic B-spline on SO(3): R = R_i prod_{m=1..3} exp(B~_m(u) log(R_{i+m-1}^T R_{i+m}))."""
    i, u = segment(t, t0, dt, len(Rctrl))
    cb = cumulative_basis(u)
    R = Rctrl[i].copy()
    for m in range(1, 4):
        R = R @ so3_exp(cb[m] * log_so3(Rctrl[i + m - 1].T @ Rctrl[i + m]))
    return R


# ---------------------------------------------------------------------------------------------
# Pose perturbation check (design v3.4): known perturbations, detection and recovery error
# ---------------------------------------------------------------------------------------------


def perturb_pose(
    R: np.ndarray, t: np.ndarray, rot_deg: float, trans_m: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a known perturbation: a rotation of exactly ``rot_deg`` about a random axis
    (right-multiplied, body frame) and a translation of exactly ``trans_m`` in a random direction."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    return R @ so3_exp(np.deg2rad(rot_deg) * axis), t + trans_m * d


def pose_error(
    R_est: np.ndarray, t_est: np.ndarray, R_true: np.ndarray, t_true: np.ndarray
) -> tuple[float, float]:
    """Recovery error: rotation angle (degrees) of R_true^T R_est and translation distance."""
    ang = float(np.rad2deg(np.linalg.norm(log_so3(R_true.T @ R_est))))
    return ang, float(np.linalg.norm(t_est - t_true))


def recovery_curve(sizes: list, errors_before: list, errors_after: list) -> list[dict]:
    """Rows of the report: perturbation size, error before and after refinement, and the
    fraction of the perturbation removed (1 = fully recovered)."""
    rows = []
    for s, b, a in zip(sizes, errors_before, errors_after):
        rows.append(
            {"size": s, "before": b, "after": a, "recovered": 1.0 - a / b if b > 0 else 1.0}
        )
    return rows
