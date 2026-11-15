"""Gaussian surfels whose uncertainty renders, and the depth-only view gate built on it (design v3.1).

Reference design: ``docs/design-3dgs-bayes.md`` (sections 1-4, 9 and 10; section 8 is background for a
separate paper and is not implemented here).

Each surfel (2DGS-style planar Gaussian) has

* a *shape* used for rendering only: centre ``mu``, rotation ``R = [t1 t2 n]`` (normal ``n``),
  scales ``s`` and opacity ``alpha``; rasterised (e.g. with gsplat) into depth and alpha weights;
* a separate 10 x 10 *uncertainty* over ``theta = [d_mu (3, world), d_phi (3, body-frame rotation
  error, R <- R exp([d_phi]x)), d_log_s (3), d_logit_alpha (1)]``, kept in information form.

Depth observes the normal offset and tilt of a surfel, ``(nu, tau1, tau2) = G theta`` with
``nu = n^T d_mu``, ``tau1 = d_phi_2``, ``tau2 = -d_phi_1``, through the ray-surfel depth, and its
opacity through the compositing weights (``composite_depth_opacity_jacobian``). Scales do not change
the depth along a ray to first order and keep their prior. Opacity is trained during a warm-up and
then frozen: after the freeze it is not re-estimated, and its information is used only for the
support rule (``opacity_supported``).

Fusion on rendering: view i adds ``sum_u a^2 J_u^T J_u / sigma_u^2`` to surfel k, with ``J_u`` the
alpha-weighted rasteriser Jacobian of the rendered depth. At the MAP, leaving view j out is the
subtraction ``Lambda_{-j} = Lambda - Lambda_j`` plus one Newton step on the mean,
``mu_{-j} = mu + Lambda_{-j}^{-1} grad L_j`` (infinitesimal jackknife; exact for the linear-Gaussian
case), on the 9 geometric coordinates. The other views' surface is rendered into view j (surfels
supported only by view j are dropped, so the tested view spawns nothing) and the whitened depth
residuals of view j are tested. No correspondences are used.

Noise: one free variable, a scene temperature gamma (``detection_theory.temperature_median_ratio``);
the shape of the noise is calibrated offline. Shared bias: a distance term d -> d + beta d^2 shared by
all views, fitted by least squares with leave-view-out by subtraction (``distance_bias_*``).

Stated approximation: each surfel has its own block (cross-surfel correlations, the correlation of
depth error within a view and the shared pose error of a view are ignored), so the rendered
variance is overconfident. ``information_deflation`` (n_eff / n) and a jackknife inflation are
ablations of this limitation; no deeper fix is attempted.
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


DIM = 10  # [d_mu (3), d_phi (3), d_log_s (3), d_logit_alpha (1)]
GEO = slice(0, 9)  # geometric coordinates (re-estimated by the leave-out Newton step)
OPA = 9  # opacity coordinate (frozen after warm-up)


def perturb_surfel(
    mu: np.ndarray, R: np.ndarray, s: np.ndarray, theta: np.ndarray, alpha: float | None = None
):
    """Apply theta = [d_mu, d_phi (body), d_log_s, d_logit_alpha] to a surfel.

    Returns (mu, R, s) or, if ``alpha`` is given, (mu, R, s, alpha)."""
    out = (mu + theta[:3], R @ so3_exp(theta[3:6]), s * np.exp(theta[6:9]))
    if alpha is None:
        return out
    logit = np.log(alpha / (1.0 - alpha)) + (theta[OPA] if len(theta) > OPA else 0.0)
    return (*out, float(1.0 / (1.0 + np.exp(-logit))))


def subblock_map(R: np.ndarray) -> np.ndarray:
    """G (3 x 10): theta -> (nu, tau1, tau2), normal offset and tilt."""
    n = R[:, 2]
    G = np.zeros((3, DIM))
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
    """Full 1 x 10 Jacobian of the ray-surfel depth with respect to theta (scale and opacity
    columns are 0: opacity acts through the compositing weights, see
    ``composite_depth_opacity_jacobian``)."""
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


def composite_depth_opacity_jacobian(
    alpha: np.ndarray, g: np.ndarray, d: np.ndarray, order: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Normalised composited depth D = sum w d / sum w along one ray, with w_k = alpha_k g_k T_k,
    and its derivative with respect to logit(alpha_k) for every surfel (including the effect on the
    transmittance of the surfels behind k).

    Returns (D, w_hat, dD_dlogit_alpha)."""
    a = alpha * g
    w = np.zeros(len(alpha))
    T = 1.0
    for k in order:
        w[k] = a[k] * T
        T *= 1.0 - a[k]
    W = float(w.sum())
    A = float((w * d).sum())
    D = A / W
    dD = np.zeros(len(alpha))
    for pos, k in enumerate(order):
        behind = order[pos + 1 :]
        dA = g[k] * (w[k] / a[k]) * d[k] - g[k] / (1.0 - a[k]) * float(
            (w[behind] * d[behind]).sum()
        )
        dW = g[k] * (w[k] / a[k]) - g[k] / (1.0 - a[k]) * float(w[behind].sum())
        dD[k] = alpha[k] * (1.0 - alpha[k]) * (dA - D * dW) / W
    return D, w / W, dD


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
    j_opacity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Information (Lambda_i, eta_i) that one view adds to every surfel (10 x 10 blocks).

    Contributions are listed per (pixel, surfel) pair: ``pix`` pixel index, ``gid`` surfel index,
    ``w_hat`` normalised weight, ``J`` (m, 10) ray-surfel Jacobians, ``j_opacity`` (m,) optional
    dD/dlogit(alpha). ``e`` is the per-pixel residual y_u - (a D_bar_u + b) at the linearisation
    point and ``sigma2`` its noise variance. Lambda_k += c a^2 g^T g / sigma^2,
    eta_k += c a g^T e / sigma^2, g = [w J_geo, dD/dlogit alpha], c = deflation.
    """
    L = np.zeros((n_gauss, DIM, DIM))
    eta = np.zeros((n_gauss, DIM))
    Jw = a * w_hat[:, None] * J
    if j_opacity is not None:
        Jw[:, OPA] = a * j_opacity
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


def posterior(
    L: np.ndarray, eta: np.ndarray, frozen_opacity: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Mean perturbation and covariance per surfel.

    With ``frozen_opacity`` the opacity is held at its frozen value: the geometric coordinates are
    estimated conditionally (their 9 x 9 information block), and the returned arrays are 9-dim."""
    if frozen_opacity:
        L = L[:, GEO, GEO]
        eta = eta[:, GEO]
    cov = np.linalg.inv(L)
    return np.einsum("kij,kj->ki", cov, eta), cov


def view_gradient(eta_j: np.ndarray) -> np.ndarray:
    """Gradient of view j's negative log-likelihood at the linearisation point: -eta_j."""
    return -eta_j


def leave_view_out_newton(
    mean: np.ndarray, L: np.ndarray, Lj: np.ndarray, grad_j: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-view-out at the MAP: Lambda_{-j} = Lambda - Lambda_j and one Newton step
    mu_{-j} = mu + Lambda_{-j}^{-1} grad L_j(mu) on the 9 geometric coordinates (opacity frozen).

    Exact for the linear-Gaussian model (infinitesimal jackknife). Returns (mean_{-j}, cov_{-j})."""
    Lm = (L - Lj)[:, GEO, GEO]
    cov = np.linalg.inv(Lm)
    return mean + np.einsum("kij,kj->ki", cov, grad_j[:, GEO]), cov


def opacity_supported(L_data: np.ndarray, Lj: np.ndarray, rho: float = 0.5) -> np.ndarray:
    """Support rule for the frozen opacity: drop surfel k from view j's leave-out render if view j
    supplied more than a fraction rho of its opacity information, Lambda_j[a,a] / Lambda[a,a] > rho.
    ``L_data`` is the data information (prior excluded)."""
    tot = L_data[:, OPA, OPA]
    share = np.divide(Lj[:, OPA, OPA], tot, out=np.zeros_like(tot), where=tot > 0)
    return share <= rho


def offset_tilt_cov(cov: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Marginal 3 x 3 covariance of (nu, tau1, tau2) that the gate renders."""
    G = subblock_map(R)[:, GEO] if cov.shape[-1] == 9 else subblock_map(R)
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
    J = J[:, : mean.shape[1]]
    D = float(np.sum(w_hat * (d_bar + np.einsum("ki,ki->k", J, mean))))
    var = float(np.sum(w_hat**2 * np.einsum("ki,kij,kj->k", J, cov, J)))
    return D, var


def whitened_residual(
    d_hat: float, a: float, b: float, D: float, var: float, sigma_d2: float
) -> float:
    """z = (d_hat - (a D + b)) / sqrt(sigma_d^2 + a^2 var)."""
    return (d_hat - (a * D + b)) / np.sqrt(sigma_d2 + a * a * var)


def distance_bias_normal_equations(
    d_render: np.ndarray, a: float, e: np.ndarray, sigma2: np.ndarray
) -> tuple[float, float]:
    """Contribution of one view to the shared distance-bias coefficient beta in
    d_hat = a (d + beta d^2) + b + eps: returns (h, g) with h = sum a^2 d^4 / sigma^2 and
    g = sum a d^2 e / sigma^2, e the residual at beta = 0."""
    x = a * d_render**2
    return float(np.sum(x * x / sigma2)), float(np.sum(x * e / sigma2))


def distance_bias_fit(
    h: np.ndarray, g: np.ndarray, exclude: int | None = None, prior_info: float = 0.0
) -> float:
    """Shared beta from per-view normal equations; leave-view-out by subtracting view ``exclude``."""
    H = float(np.sum(h)) + prior_info
    Gs = float(np.sum(g))
    if exclude is not None:
        H -= float(h[exclude])
        Gs -= float(g[exclude])
    return Gs / H


# ---------------------------------------------------------------------------------------------
# Leave-N-out: a block of neighbouring frames (design v3.3)
# ---------------------------------------------------------------------------------------------


def block_for_view(order: list, strip: dict, j, n_block: int) -> list:
    """The block left out when view ``j`` is tested: ``n_block`` neighbouring frames of the same
    flight strip (or image sequence), centred on ``j`` and shifted inwards at the strip ends; the
    whole strip if it is shorter than ``n_block``. ``order`` lists the views in acquisition order
    and ``strip`` maps a view to its strip id. ``n_block = 1`` is single-view leave-out."""
    same = [v for v in order if strip[v] == strip[j]]
    if n_block >= len(same):
        return same
    i = same.index(j)
    lo = min(max(i - (n_block - 1) // 2, 0), len(same) - n_block)
    return same[lo : lo + n_block]


def block_information(infos: dict, block: list) -> tuple[np.ndarray, np.ndarray]:
    """Summed information (Lambda_B, eta_B) of the views in a block; ``infos`` maps view ->
    (Lambda_i, eta_i). Messages add, so the block's message is the sum of its views' messages."""
    L = sum(infos[v][0] for v in block)
    eta = sum(infos[v][1] for v in block)
    return L, eta


def leave_block_out_newton(
    mean: np.ndarray, L: np.ndarray, L_block: np.ndarray, grad_block: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cavity without a whole block: subtract the block's summed information and take one Newton
    step with the block's summed gradient (exact for the linear-Gaussian model)."""
    return leave_view_out_newton(mean, L, L_block, grad_block)


# ---------------------------------------------------------------------------------------------
# Residual precision with the block-sparse surfel term (frozen equation, design v3.4)
# ---------------------------------------------------------------------------------------------


def residual_precision_apply(
    wn: np.ndarray, U: np.ndarray, inv: np.ndarray, Lam_cav: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Apply Sigma_r^{-1} to X for Sigma_r = W^{-1} + U P U^T, by Woodbury.

    ``wn`` (n,) is the diagonal noise precision W, ``U`` (n, 3) holds each pixel's row
    w_uk J_uk on the (nu, tau1, tau2) block of its surfel ``inv[u]``, and ``Lam_cav`` (K, 3, 3) the
    cavity information P^{-1} of each surfel. Pixels rendered by the same surfel are correlated
    through its error; different surfels are independent. Cost: one pass over the pixels.
        Sigma_r^{-1} X = W X - W U (Lam_cav + U^T W U)^{-1} U^T W X
    """
    X = X if X.ndim == 2 else X[:, None]
    S = Lam_cav.copy()
    np.add.at(S, inv, wn[:, None, None] * U[:, :, None] * U[:, None, :])
    Sinv = np.linalg.inv(S)
    WX = wn[:, None] * X
    UtWX = np.zeros((len(S), X.shape[1], 3))
    np.add.at(UtWX, inv, WX[:, :, None] * U[:, None, :])
    corr = np.einsum("kij,kcj->kci", Sinv, UtWX)
    return WX - wn[:, None] * np.einsum("nci,ni->nc", corr[inv], U)
