"""Closed-form Gaussian-primitive surface built from depth maps only (no colour, no optimisation).

Each primitive is a planar Gaussian with

* centre ``mu`` and unit normal ``n``;
* tangent half-widths ``(s_par, s_perp)``: the *extent*, used for rendering weights only;
* normal offset ``h = n^T mu`` with variance ``sigma_h^2`` (information ``lam = 1 / sigma_h^2``):
  the *uncertainty*, used for fusion and for the residual variance only;
* opacity ``alpha``.

Extent and uncertainty are separate on purpose. With a single covariance for both, a confident
primitive (small variance) would shrink in the render and vanish from the composite; here the
render weight depends on the extent only, and confidence enters only through variances.

Pipeline implemented here (all closed form):

1. forward: back-project a depth map to primitives and propagate depth, pose and intrinsics
   uncertainty to ``sigma_h^2 = n^T P n`` (``primitives_from_depth``);
2. fusion in information form, ``lam = sum lam_i``, ``h = sum lam_i h_i / lam``, with exact
   leave-view-out by subtraction (``fuse_information``, ``leave_one_out``), which equals the RAIM
   solution separation (``solution_separation_stat`` = ``innovation_stat``);
3. backward: ray-plane depth of a primitive in another view and its variance
   (``ray_plane_depth``, ``ray_plane_depth_variance``), composite with extent-only weights
   (``composite``), and the gate residual (``gate_residual``);
4. rendering-informed fusion: association gate (``associates``) and soft weight;
5. per-region outputs: posterior, protection level and view count (``region_outputs``).

Assumptions that are stated, not hidden: observations are independent (correlated depth error
within a view and the shared pose error of a view across its primitives make the information
overconfident; see the manuscript limitations); the normal is treated as known when computing
``sigma_h^2``; the composite variance is diagonal over pixels.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------------------------
# 1. forward: depth map -> primitives
# ---------------------------------------------------------------------------------------------


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix [v]_x of a 3-vector."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def pixel_rays(K: np.ndarray, h: int, w: int) -> np.ndarray:
    """r = K^-1 [u v 1]^T for every pixel, shape (h, w, 3); the third component is 1 (z-depth)."""
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    return pix @ np.linalg.inv(K).T


def backproject(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World points mu = R d r + t for a depth map (z-depth) and a camera-to-world pose (R, t)."""
    r = pixel_rays(K, *depth.shape)
    return (depth[..., None] * r) @ R.T + t


def normals_from_depth(X: np.ndarray, cam_centre: np.ndarray) -> np.ndarray:
    """Unit normals from the depth gradient: cross product of the image-axis tangents of X.

    ``X`` is (h, w, 3) world points. Normals are oriented towards the camera centre. The border
    uses one-sided differences.
    """
    du = np.gradient(X, axis=1)
    dv = np.gradient(X, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    to_cam = cam_centre - X
    flip = np.sum(n * to_cam, axis=-1) < 0
    n[flip] *= -1.0
    return n


def extent_halfwidths(
    d: np.ndarray, focal: float, cos_theta: np.ndarray, footprint: float = 1.0, cos_min: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Tangent half-widths: one pixel subtends d / f; stretched by 1 / cos(theta) along the tilt.

    Returns ``(s_par, s_perp)``: along the direction of the projected ray (stretched) and across it.
    ``footprint`` scales the half-width (e.g. 0.5 for half a pixel, 1 for one pixel).
    """
    s_perp = footprint * d / focal
    s_par = s_perp / np.maximum(cos_theta, cos_min)
    return s_par, s_perp


def pose_jacobian_point(p_cam_world: np.ndarray) -> np.ndarray:
    """Jacobian (3, 6) of the world point mu = R d r + t with respect to a world-frame pose
    perturbation [dt, dw]: d mu = dt - [R d r]_x dw. ``p_cam_world`` is R d r = mu - t."""
    return np.concatenate([np.eye(3), -skew(p_cam_world)], axis=1)


def intrinsics_jacobian_point(
    d: float, R: np.ndarray, u: float, v: float, K: np.ndarray
) -> np.ndarray:
    """Jacobian (3, 3) of mu = R d r + t with respect to (f, cx, cy) for a pinhole K with equal
    focal lengths: r = [(u - cx)/f, (v - cy)/f, 1]."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    dr_df = np.array([-(u - cx) / f**2, -(v - cy) / f**2, 0.0])
    dr_dcx = np.array([-1.0 / f, 0.0, 0.0])
    dr_dcy = np.array([0.0, -1.0 / f, 0.0])
    return d * R @ np.stack([dr_df, dr_dcx, dr_dcy], axis=1)


def point_covariance(
    J_d: np.ndarray,
    sigma_d: float,
    J_q: np.ndarray | None = None,
    Sigma_q: np.ndarray | None = None,
    J_K: np.ndarray | None = None,
    Sigma_K: np.ndarray | None = None,
) -> np.ndarray:
    """P = J_d sigma_d^2 J_d^T + J_q Sigma_q J_q^T + J_K Sigma_K J_K^T  (3 x 3)."""
    P = np.outer(J_d, J_d) * sigma_d**2
    if J_q is not None and Sigma_q is not None:
        P = P + J_q @ Sigma_q @ J_q.T
    if J_K is not None and Sigma_K is not None:
        P = P + J_K @ Sigma_K @ J_K.T
    return P


def offset_variance(n: np.ndarray, P: np.ndarray) -> float:
    """sigma_h^2 = n^T P n: variance of the normal offset h = n^T mu."""
    return float(n @ P @ n)


def primitives_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    sigma_d: float | np.ndarray,
    Sigma_q: np.ndarray | None = None,
    Sigma_K: np.ndarray | None = None,
    footprint: float = 1.0,
    cos_min: float = 0.1,
) -> dict[str, np.ndarray]:
    """Closed-form primitives for every interior pixel of a depth map (no optimisation).

    ``sigma_d`` is the depth standard deviation (scalar or (h, w)), e.g. sigma(range, texture,
    grazing) of the measurement model. Returns arrays over the (h-2)(w-2) interior pixels.
    """
    h, w = depth.shape
    X = backproject(depth, K, R, t)
    centre = t
    n = normals_from_depth(X, centre)
    r = pixel_rays(K, h, w)
    ray_world = r @ R.T
    ray_unit = ray_world / np.linalg.norm(ray_world, axis=-1, keepdims=True)
    cos_theta = np.abs(np.sum(n * ray_unit, axis=-1))
    s_par, s_perp = extent_halfwidths(depth, K[0, 0], cos_theta, footprint, cos_min)
    sig = np.broadcast_to(np.asarray(sigma_d, dtype=float), depth.shape)
    out = {k: [] for k in ("mu", "n", "h", "sigma_h2", "lam", "s_par", "s_perp", "t_par", "cos_theta")}
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            d = depth[i, j]
            Rdr = R @ (d * r[i, j])
            J_d = R @ r[i, j]
            Jq = pose_jacobian_point(Rdr) if Sigma_q is not None else None
            JK = intrinsics_jacobian_point(d, R, j, i, K) if Sigma_K is not None else None
            P = point_covariance(J_d, sig[i, j], Jq, Sigma_q, JK, Sigma_K)
            nn = n[i, j]
            s2 = offset_variance(nn, P)
            tp = ray_unit[i, j] - np.dot(ray_unit[i, j], nn) * nn
            tp = tp / max(np.linalg.norm(tp), 1e-12)
            out["mu"].append(X[i, j])
            out["n"].append(nn)
            out["h"].append(float(nn @ X[i, j]))
            out["sigma_h2"].append(s2)
            out["lam"].append(1.0 / s2)
            out["s_par"].append(s_par[i, j])
            out["s_perp"].append(s_perp[i, j])
            out["t_par"].append(tp)
            out["cos_theta"].append(cos_theta[i, j])
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# 2. information-form fusion and exact leave-view-out
# ---------------------------------------------------------------------------------------------


def fuse_information(lam: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """lam = sum lam_i, h = sum lam_i h_i / lam. Posterior variance of h is 1 / lam."""
    lam = np.asarray(lam, dtype=float)
    h = np.asarray(h, dtype=float)
    tot = float(lam.sum())
    return tot, float((lam * h).sum() / tot)


def leave_one_out(
    lam_tot: float, h_tot: float, lam_j: float, h_j: float, eps: float = 1e-12
) -> tuple[float, float]:
    """Exact leave-view-j-out by subtracting its information: no refit.

    Returns (lam_{-j}, h_{-j}); lam_{-j} = 0 and h = nan if view j was the only contributor.
    """
    lam_m = lam_tot - lam_j
    if lam_m <= eps * max(lam_tot, 1.0):
        return 0.0, float("nan")
    return lam_m, (lam_tot * h_tot - lam_j * h_j) / lam_m


def solution_separation_stat(lam_tot: float, h_tot: float, lam_j: float, h_j: float) -> float:
    """RAIM solution separation: d = h_all - h_{-j}; statistic d^2 / Cov(d) with
    Cov(d) = 1/lam_{-j} - 1/lam_all."""
    lam_m, h_m = leave_one_out(lam_tot, h_tot, lam_j, h_j)
    d = h_tot - h_m
    return float(d * d / (1.0 / lam_m - 1.0 / lam_tot))


def innovation_stat(lam_tot: float, h_tot: float, lam_j: float, h_j: float) -> float:
    """Leave-one-out innovation: nu = h_j - h_{-j}; statistic nu^2 / (1/lam_j + 1/lam_{-j})."""
    lam_m, h_m = leave_one_out(lam_tot, h_tot, lam_j, h_j)
    nu = h_j - h_m
    return float(nu * nu / (1.0 / lam_j + 1.0 / lam_m))


# ---------------------------------------------------------------------------------------------
# 3. backward: primitives -> depth in another view; composite; gate residual
# ---------------------------------------------------------------------------------------------


def ray_plane_depth(
    h: np.ndarray, n: np.ndarray, c: np.ndarray, R: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """z-depth along the pixel ray of view j (centre ``c``, rotation ``R``, ray ``r`` with
    r_z = 1) to the plane n^T x = h:  d_k = (h - n^T c) / (n^T R r)."""
    return (h - n @ c) / (n @ (R @ r))


def ray_plane_pose_jacobian(
    h: float, n: np.ndarray, c: np.ndarray, R: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """(6,) derivative of d_k with respect to a world-frame pose perturbation [dt, dw] of view j:
    d d_k / d dt = -n / (n^T R r),  d d_k / d dw = -d_k (R r x n) / (n^T R r)."""
    Rr = R @ r
    den = float(n @ Rr)
    d = (h - float(n @ c)) / den
    return np.concatenate([-n / den, -d * np.cross(Rr, n) / den])


def ray_plane_depth_variance(
    sigma_h2: float,
    n: np.ndarray,
    R: np.ndarray,
    r: np.ndarray,
    J_pose: np.ndarray | None = None,
    Sigma_q: np.ndarray | None = None,
) -> float:
    """sigma_k^2 = sigma_h^2 / (n^T R r)^2 + J_pose Sigma_q J_pose^T.

    For a unit-ray range the first term is sigma_h^2 / cos^2(theta_j); with z-depth and r_z = 1
    the factor is 1 / (n^T R r)^2.
    """
    den = float(n @ (R @ r))
    var = sigma_h2 / den**2
    if J_pose is not None and Sigma_q is not None:
        var += float(J_pose @ Sigma_q @ J_pose)
    return var


def extent_weight(
    delta: np.ndarray, t_par: np.ndarray, t_perp: np.ndarray, s_par: float, s_perp: float
) -> float:
    """Gaussian footprint G(u; Sigma_ext) at the in-plane displacement ``delta`` from the centre.

    Depends on the extent only; the uncertainty of the primitive does not enter.
    """
    a = float(delta @ t_par) / s_par
    b = float(delta @ t_perp) / s_perp
    return math.exp(-0.5 * (a * a + b * b))


def alpha_composite_weights(alpha: np.ndarray, G: np.ndarray, order: np.ndarray) -> np.ndarray:
    """w_k = alpha_k G_k T_k with the transmittance T_k = prod over nearer primitives (1 - alpha G).

    ``order`` is the indices sorted from nearest to farthest along the ray.
    """
    w = np.zeros_like(alpha, dtype=float)
    T = 1.0
    for k in order:
        a = alpha[k] * G[k]
        w[k] = a * T
        T *= 1.0 - a
    return w


def composite(
    w: np.ndarray, d_k: np.ndarray, sigma2_k: np.ndarray, include_spread: bool = False
) -> tuple[float, float, float]:
    """Normalised composite depth d(u) = sum w d_k / W and variance sigma^2(u) = sum w sigma_k^2 / W.

    Returns (d, sigma2, W) with W = sum w the accumulated opacity (support). With
    ``include_spread`` the between-primitive spread sum w (d_k - d)^2 / W is added (law of total
    variance), which inflates the variance at depth discontinuities.
    """
    W = float(w.sum())
    if W <= 0:
        return float("nan"), float("nan"), 0.0
    d = float((w * d_k).sum() / W)
    var = float((w * sigma2_k).sum() / W)
    if include_spread:
        var += float((w * (d_k - d) ** 2).sum() / W)
    return d, var, W


def gate_residual(
    d_hat: float, a: float, b: float, d_comp: float, sigma2_comp: float, sigma_d2: float
) -> tuple[float, float]:
    """Residual r = d_hat - (a d(u) + b) and its variance sigma_d^2 + a^2 sigma^2(u)."""
    return d_hat - (a * d_comp + b), sigma_d2 + a * a * sigma2_comp


# ---------------------------------------------------------------------------------------------
# 4. rendering-informed fusion
# ---------------------------------------------------------------------------------------------


def associates(
    mu_new: np.ndarray,
    n_new: np.ndarray,
    sigma_h2_new: float,
    mu_k: np.ndarray,
    n_k: np.ndarray,
    h_k: float,
    sigma_h2_k: float,
    t_par_k: np.ndarray,
    s_par_k: float,
    s_perp_k: float,
    tau_h: float = 3.0,
    tau_ext: float = 3.0,
    cos_min: float = 0.8,
) -> bool:
    """Projective association: a new observation updates primitive k iff its normal agrees, its
    offset is within ``tau_h`` combined standard deviations of the plane, and it lies within
    ``tau_ext`` extents of the centre; otherwise a new primitive is spawned."""
    if float(n_new @ n_k) < cos_min:
        return False
    z = abs(float(n_k @ mu_new) - h_k) / math.sqrt(sigma_h2_new + sigma_h2_k)
    if z > tau_h:
        return False
    delta = mu_new - mu_k
    delta = delta - float(delta @ n_k) * n_k
    t_perp = np.cross(n_k, t_par_k)
    a = float(delta @ t_par_k) / s_par_k
    b = float(delta @ t_perp) / s_perp_k
    return math.sqrt(a * a + b * b) <= tau_ext


# ---------------------------------------------------------------------------------------------
# 5. per-region outputs: posterior, protection level, view count
# ---------------------------------------------------------------------------------------------


def mdb_offset(
    lam_j: float, lam_m: float, alpha: float = 0.05, beta: float = 0.1
) -> float:
    """Smallest offset error of view j detected with probability 1 - beta at false-alarm alpha:
    (z_alpha + z_beta) * sqrt(1/lam_j + 1/lam_{-j}) = (z_alpha + z_beta) * sqrt(lam / (lam_j lam_{-j}))."""
    za = float(norm.ppf(1.0 - alpha / 2.0))
    zb = float(norm.ppf(1.0 - beta))
    return (za + zb) * math.sqrt(1.0 / lam_j + 1.0 / lam_m)


def primitive_protection_level(
    lams: np.ndarray, alpha: float = 0.05, beta: float = 0.1, z: float = 1.96
) -> float:
    """Conditional protection level of the fused offset of one region (single faulty view).

    An undetected fault of view j is at most its minimum detectable offset error, which moves the
    fused offset by (lam_j / lam) times that error: slope x MDB, as in RAIM solution separation.
    PL = max_j (lam_j / lam) * mdb_offset_j + z / sqrt(lam). Views whose removal leaves no
    information cannot be tested and are excluded (the region is then not testable).
    """
    lams = np.asarray(lams, dtype=float)
    lam = float(lams.sum())
    terms = [
        (lj / lam) * mdb_offset(lj, lam - lj, alpha, beta) for lj in lams if lam - lj > 0
    ]
    if not terms:
        return float("inf")
    return max(terms) + z / math.sqrt(lam)


def region_outputs(
    lams: np.ndarray,
    hs: np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.1,
    min_views: int = 3,
) -> dict[str, float | int | bool]:
    """Posterior, protection level and view count (availability) of a region."""
    lam, h = fuse_information(lams, hs)
    m = len(lams)
    return {
        "posterior_mean": h,
        "posterior_var": 1.0 / lam,
        "protection_level": primitive_protection_level(lams, alpha, beta),
        "n_views": m,
        "testable": m >= min_views,
    }
