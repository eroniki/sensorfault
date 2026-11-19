"""Public sensor noise models (pure functions, explicit RNG, no global state).

All functions are vectorised over numpy arrays. Sources (to verify before citing):
Khoshelham & Elberink 2012 (structured light), Nguyen/Izadi/Lovell 2012,
Sarbolandi 2015 (ToF), Glennie & Lichti 2010 / Soudarissanane 2011 (LiDAR / incidence).

Variance conventions (PLAN.md section 3)
----------------------------------------
* ``depth_residual_variance`` is the variance of the depth residual *along the ray*
  (use it to WEIGHT depth residuals). Exact linearisation: d r / d theta = r tan(theta).
* ``normal_direction_variance`` is the variance of the point error projected on the
  surface normal (use it to SET splat covariance); bounded as theta -> 90 deg.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EPS = 0.17  # tan^2 -> sin^2/(cos^2+eps^2) clamp, PLAN.md section 3
DEFAULT_GATE_DEG = 85.0
DEFAULT_VAR_FLOOR = 1e-8  # (0.1 mm)^2
DEFAULT_VAR_CAP = 0.25  # (0.5 m)^2


def grazing_gate(theta, gate_deg: float = DEFAULT_GATE_DEG):
    """True where the incidence angle (rad, to the surface normal) is usable (<= gate)."""
    return np.abs(np.asarray(theta, dtype=np.float64)) <= np.deg2rad(gate_deg)


def _phi2(sigma_phi, kappa: float, gamma):
    """Effective angular variance incl. beam footprint term (kappa*gamma)^2."""
    return (
        np.asarray(sigma_phi, dtype=np.float64) ** 2
        + (kappa * np.asarray(gamma, dtype=np.float64)) ** 2
    )


def depth_residual_variance(
    r,
    theta,
    sigma_r,
    sigma_phi,
    *,
    eps: float = DEFAULT_EPS,
    gate_deg: float = DEFAULT_GATE_DEG,
    kappa: float = 0.0,
    gamma=0.0,
    var_floor: float = DEFAULT_VAR_FLOOR,
    var_cap: float = DEFAULT_VAR_CAP,
):
    """Regularised variance of the depth residual along the ray.

    var = sigma_r^2 + r^2 * (sigma_phi^2 + (kappa*gamma)^2) * sin^2(t) / (cos^2(t) + eps^2)

    With eps -> 0 this is exactly sigma_r^2 + (r sigma_phi tan t)^2. The clamp keeps the
    value finite near grazing incidence where the linearisation fails
    (sigma_phi tan t >~ 0.3). Beyond ``gate_deg`` the value is set to ``var_cap`` (the
    caller should additionally drop those pixels via :func:`grazing_gate`).
    Result is clipped to [var_floor, var_cap].
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    s2 = np.sin(theta) ** 2
    c2 = np.cos(theta) ** 2
    amp = s2 / (c2 + eps**2)
    var = np.asarray(sigma_r, dtype=np.float64) ** 2 + r**2 * _phi2(sigma_phi, kappa, gamma) * amp
    var = np.clip(var, var_floor, var_cap)
    return np.where(grazing_gate(theta, gate_deg), var, var_cap)


def normal_direction_variance(
    r,
    theta,
    sigma_r,
    sigma_phi,
    *,
    kappa: float = 0.0,
    gamma=0.0,
    var_floor: float = DEFAULT_VAR_FLOOR,
    var_cap: float = DEFAULT_VAR_CAP,
):
    """Variance of the 3-D point error projected on the surface normal.

    sigma_n^2 = sigma_r^2 cos^2(t) + r^2 sigma_phi,eff^2 sin^2(t). Bounded as t -> 90 deg.
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    var = (
        np.asarray(sigma_r, dtype=np.float64) ** 2 * np.cos(theta) ** 2
        + r**2 * _phi2(sigma_phi, kappa, gamma) * np.sin(theta) ** 2
    )
    return np.clip(var, var_floor, var_cap)


# ----------------------------------------------------------------- sigma_z models
def structured_light_sigma_z(z, sigma_d_px: float, f_px: float, baseline_m: float):
    """Kinect-style structured light axial std: sigma_z = z^2 * sigma_d / (f * b)."""
    return np.asarray(z, dtype=np.float64) ** 2 * sigma_d_px / (f_px * baseline_m)


def stereo_sigma_z(z, sigma_d_px: float, f_px: float, baseline_m: float):
    """Stereo triangulation axial std; same first-order form as structured light."""
    return structured_light_sigma_z(z, sigma_d_px, f_px, baseline_m)


def tof_sigma_z(z, cos_theta, sigma0_m: float, z_ref_m: float = 1.0, min_cos: float = 0.2):
    """Amplitude-modulated ToF: noise ~ 1/amplitude, amplitude ~ cos(theta)/z^2.

    sigma = sigma0 * (z/z_ref)^2 / max(cos theta, min_cos). Heuristic proxy [UNVERIFIED constants].
    """
    z = np.asarray(z, dtype=np.float64)
    c = np.maximum(np.asarray(cos_theta, dtype=np.float64), min_cos)
    return sigma0_m * (z / z_ref_m) ** 2 / c


def lidar_sigma_r(r, sigma0_m: float, sigma_prop: float = 0.0):
    """LiDAR range noise: constant + range-proportional part."""
    return sigma0_m + sigma_prop * np.asarray(r, dtype=np.float64)


# --------------------------------------------------------------- noise application
def add_gaussian_depth_noise(
    depth: np.ndarray, sigma: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    out = depth.astype(np.float64) + rng.standard_normal(depth.shape) * sigma
    return out


def add_flying_pixels(
    depth: np.ndarray, rng: np.random.Generator, prob: float = 0.5, jump_frac: float = 0.05
) -> np.ndarray:
    """ToF flying pixels: at depth edges (local range > jump_frac * z) with probability
    ``prob`` replace z by a random convex mix of the local min/max of the 3x3 window."""
    from scipy.ndimage import maximum_filter, minimum_filter

    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    big = np.where(d > 0, d, np.inf)
    lo = minimum_filter(big, size=3, mode="nearest")
    hi = maximum_filter(d, size=3, mode="nearest")
    edge = (d > 0) & np.isfinite(lo) & ((hi - lo) > jump_frac * np.maximum(d, 1e-6))
    pick = edge & (rng.random(d.shape) < prob)
    mix = rng.random(d.shape)
    fly = lo + mix * (hi - lo)
    return np.where(pick, fly, depth)


def apply_dropout(
    depth: np.ndarray, rng: np.random.Generator, p: float, grazing_cos=None, min_cos: float = 0.0
):
    """Random pixel dropout (to 0) plus optional dropout where cos(theta) < min_cos."""
    out = depth.copy()
    kill = rng.random(depth.shape) < p
    if grazing_cos is not None and min_cos > 0:
        kill |= grazing_cos < min_cos
    out[kill] = 0.0
    return out


class SensorModel:
    """Callable noise model: ``model(depth, cos_theta, rng, f_px) -> noisy depth (float32)``."""

    def __init__(self, kind: str, **params):
        self.kind = kind
        self.params = dict(params)
        if kind not in ("kinect_sl", "tof", "lidar", "stereo"):
            raise ValueError(f"unknown sensor kind {kind!r}")

    def sigma(self, depth: np.ndarray, cos_theta: np.ndarray, f_px: float) -> np.ndarray:
        p = self.params
        if self.kind == "kinect_sl":
            return structured_light_sigma_z(
                depth, p.get("sigma_d_px", 0.5), f_px, p.get("baseline_m", 0.075)
            )
        if self.kind == "stereo":
            return stereo_sigma_z(depth, p.get("sigma_d_px", 0.25), f_px, p.get("baseline_m", 0.05))
        if self.kind == "tof":
            return tof_sigma_z(depth, cos_theta, p.get("sigma0_m", 0.005), p.get("z_ref_m", 1.0))
        return lidar_sigma_r(depth, p.get("sigma0_m", 0.01), p.get("sigma_prop", 0.0))

    def __call__(self, depth, cos_theta, rng: np.random.Generator, f_px: float) -> np.ndarray:
        p = self.params
        valid = np.isfinite(depth) & (depth > 0)
        noisy = add_gaussian_depth_noise(
            np.where(valid, depth, 0.0), self.sigma(depth, cos_theta, f_px), rng
        )
        if self.kind == "tof":
            noisy = add_flying_pixels(
                noisy, rng, p.get("flying_prob", 0.5), p.get("jump_frac", 0.05)
            )
        noisy = apply_dropout(noisy, rng, p.get("dropout", 0.0), cos_theta, p.get("min_cos", 0.0))
        noisy = np.where(valid & (noisy > 0), noisy, 0.0)
        return noisy.astype(np.float32)
