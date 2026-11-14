"""Gaussian surfels whose uncertainty renders, and the depth-only view gate built on it (design v3).

Each surfel (2DGS-style planar Gaussian) has

* a *shape* used for rendering only: centre ``mu``, rotation ``R = [t1 t2 n]`` (normal ``n``),
  scales ``s`` and opacity; rasterised (e.g. with gsplat) into depth and alpha weights;
* a separate 9 x 9 *uncertainty* over the perturbation ``theta = [d_mu (3, world), d_phi (3, body
  frame rotation error, R <- R exp([d_phi]x)), d_log_s (3)]``, kept in information form.

Depth observes only the normal offset and the tilt of a surfel. The gate renders that 3-dim
sub-block ``(nu, tau1, tau2)`` with ``nu = n^T d_mu``, ``tau1 = d_phi_2``, ``tau2 = -d_phi_1``
(the tangent components of ``d_n``). Scales do not change the depth along a ray to first order
(they change the alpha weights, which are held fixed), so their information comes from the prior.

Fusion on rendering: view i adds ``sum_u a^2 (w_uk J_uk)^T (w_uk J_uk) / sigma_u^2`` to surfel k,
where ``w_uk`` is the normalised alpha-compositing weight of surfel k at pixel u and ``J_uk`` the
1 x 9 Jacobian of its ray-surfel depth. Leave-view-out is a subtraction. The other views' surface
is rendered into view j (surfels without enough information from the other views are dropped,
so the tested view spawns nothing), and the whitened depth residuals of view j are tested.
Pixels with no supported surfel are untestable. No correspondences are used.

Stated approximation: each surfel has its own block (cross-surfel correlations, the correlation of
depth error within a view and the shared pose error of a view are ignored), so the rendered
variance is overconfident. ``information_deflation`` scales a view's information down (e.g. by
n_eff / n) as an ablation of this limitation.
"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix [v]_x."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues' formula."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * k @ k


def perturb_surfel(
    mu: np.ndarray, R: np.ndarray, s: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply theta = [d_mu, d_phi (body), d_log_s] to a surfel."""
    return mu + theta[:3], R @ so3_exp(theta[3:6]), s * np.exp(theta[6:9])


def subblock_map(R: np.ndarray) -> np.ndarray:
    """G (3 x 9): theta -> (nu, tau1, tau2), normal offset and tilt."""
    n = R[:, 2]
    G = np.zeros((3, 9))
    G[0, :3] = n
    G[1, 4] = 1.0  # tau1 = t1^T d_n = d_phi_2
    G[2, 3] = -1.0  # tau2 = t2^T d_n = -d_phi_1
    return G


def ray_surfel_depth(
    mu: np.ndarray, R: np.ndarray, c: np.ndarray, Rc: np.ndarray, r: np.ndarray
) -> tuple[float, np.ndarray]:
    """z-depth of the ray of camera (centre c, rotation Rc) with pixel ray r (r_z = 1) to the
    surfel plane n^T (x - mu) = 0, and the hit point x."""
    n = R[:, 2]
    w = Rc @ r
    d = float(n @ (mu - c)) / float(n @ w)
    return d, c + d * w


def depth_jacobian_subblock(
    mu: np.ndarray, R: np.ndarray, c: np.ndarray, Rc: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """d depth / d (nu, tau1, tau2) = [1, -t1^T (x - mu), -t2^T (x - mu)] / (n^T Rc r)."""
    _, x = ray_surfel_depth(mu, R, c, Rc, r)
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]
    den = float(n @ (Rc @ r))
    dx = x - mu
    return np.array([1.0, -float(t1 @ dx), -float(t2 @ dx)]) / den


def depth_jacobian(
    mu: np.ndarray, R: np.ndarray, c: np.ndarray, Rc: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Full 1 x 9 Jacobian of the ray-surfel depth with respect to theta (scale columns are 0)."""
    return depth_jacobian_subblock(mu, R, c, Rc, r) @ subblock_map(R)


def composite_weights(alpha_g: np.ndarray, order: np.ndarray, keep: np.ndarray | None = None):
    """Alpha-compositing weights w_k = alpha_k G_k T_k along one ray (front to back ``order``),
    optionally with some surfels removed (``keep`` False), and their normalised form.

    Returns (w, w_hat, W) with W = sum w the accumulated opacity (support of the pixel)."""
    keep = np.ones(len(alpha_g), bool) if keep is None else keep
    w = np.zeros(len(alpha_g))
    T = 1.0
    for k in order:
        if not keep[k]:
            continue
        a = float(alpha_g[k])
        w[k] = a * T
        T *= 1.0 - a
    W = float(w.sum())
    return w, (w / W if W > 0 else w), W


def view_information(
    pix: np.ndarray,
    gid: np.ndarray,
    w_hat: np.ndarray,
    J: np.ndarray,
    e: np.ndarray,
    sigma2: np.ndarray,
    n_gauss: int,
    a: float = 1.0,
    information_deflation: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Information (Lambda_i, eta_i) that one view adds to every surfel.

    Contributions are listed per (pixel, surfel) pair: ``pix`` pixel index, ``gid`` surfel index,
    ``w_hat`` normalised weight, ``J`` (m, 9) Jacobians. ``e`` is the per-pixel residual
    y_u - (a D_bar_u + b) at the linearisation point and ``sigma2`` its noise variance.
    Lambda_k += c (a w J)^T (a w J) / sigma^2, eta_k += c (a w J)^T e / sigma^2, c = deflation.
    """
    L = np.zeros((n_gauss, 9, 9))
    eta = np.zeros((n_gauss, 9))
    Jw = a * w_hat[:, None] * J
    inv = information_deflation / sigma2[pix]
    np.add.at(L, gid, inv[:, None, None] * Jw[:, :, None] * Jw[:, None, :])
    np.add.at(eta, gid, (inv * e[pix])[:, None] * Jw)
    return L, eta


def fuse(
    prior: np.ndarray, views: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Lambda = Lambda_0 + sum_i Lambda_i, eta = sum_i eta_i (prior centred at the linearisation point)."""
    L = np.broadcast_to(prior, views[0][0].shape).copy()
    eta = np.zeros_like(views[0][1])
    for Li, ei in views:
        L += Li
        eta += ei
    return L, eta


def leave_view_out(L: np.ndarray, eta: np.ndarray, Lj: np.ndarray, ej: np.ndarray):
    """Exact (for the linearised model) removal of view j: subtract its information."""
    return L - Lj, eta - ej


def posterior(L: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean perturbation and covariance per surfel."""
    cov = np.linalg.inv(L)
    return np.einsum("kij,kj->ki", cov, eta), cov


def offset_tilt_cov(cov: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Marginal 3 x 3 covariance of (nu, tau1, tau2) that the gate renders."""
    G = subblock_map(R)
    return G @ cov @ G.T


def supported(L_other: np.ndarray, Rs: np.ndarray, prior: np.ndarray, lam_min: float) -> np.ndarray:
    """A surfel is kept in the leave-j-out render iff the other views' information on its normal
    offset exceeds ``lam_min`` beyond the prior. Surfels only view j supports are dropped."""
    out = np.zeros(len(L_other), bool)
    for k in range(len(L_other)):
        g = subblock_map(Rs[k])[0]
        out[k] = float(g @ (L_other[k] - prior) @ g) >= lam_min
    return out


def render_depth_and_variance(
    d_bar: np.ndarray, w_hat: np.ndarray, J: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> tuple[float, float]:
    """Rendered depth sum w (d_bar + J mean) and variance sum w^2 J cov J^T at one pixel
    (independent per-surfel blocks)."""
    D = float(np.sum(w_hat * (d_bar + np.einsum("ki,ki->k", J, mean))))
    var = float(np.sum(w_hat**2 * np.einsum("ki,kij,kj->k", J, cov, J)))
    return D, var


def whitened_residual(
    d_hat: float, a: float, b: float, D: float, var: float, sigma_d2: float
) -> float:
    """z = (d_hat - (a D + b)) / sqrt(sigma_d^2 + a^2 var)."""
    return (d_hat - (a * D + b)) / np.sqrt(sigma_d2 + a * a * var)
