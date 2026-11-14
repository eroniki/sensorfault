"""Detection theory for view-level fault detection and exclusion (integrity monitoring).

Two layers live here.

1. Whitened, nuisance-projected theory derived from the explicit measurement model (the layer the
   manuscript uses): residual covariance ``Sigma_r`` (heteroscedastic, spatially correlated
   depth noise + pose + intrinsics + surface spread), the projected precision that removes the
   monocular scale/offset nuisance, the detectability index ``kappa``, minimum detectable biases
   for the matched, energy and scan statistics, ``n_eff = tr(C)^2 / tr(C^2)``, the pose Jacobian
   of rendered depth, and the ellipsoid protection level. See the "Measurement model" and
   "Detection statistics" sections of the manuscript.
2. The earlier iid special case (``mean_shift_*``, ``pixel_*``, ``count_*``), kept because the
   pixel-filter comparison still uses it. NOTE: with a free per-view offset the signed mean is
   absorbed by the alignment, so ``mdb_mean_shift`` applies only when no offset is fitted; a
   uniform bias over a whole view is then benign (``kappa_projected`` = 0).

Original docstring of the iid layer follows.

A coherent fault adds a bias of ``delta`` (in units of the per-pixel noise sigma) to a fraction
``f`` of the ``n_eff`` effectively independent pixels of a view. This module gives the closed
forms used in the manuscript (Section "Detection statistics and minimum detectable bias" and
Appendix "Detection analysis") so that the numbers in the paper are reproducible:

* the mean-shift statistic and its minimum detectable bias (MDB),
* the energy (chi-square-like) statistic and its MDB,
* the per-pixel flagging probability of a threshold-``k`` filter and its second-order excess,
* the aggregated pixel-count test and the crossover bias above which it beats the mean shift.

Assumptions (stated, not hidden): independent unit-variance residuals over ``n_eff`` pixels, a
Gaussian approximation of the sums, and a two-sided test at per-view false-alarm probability
``alpha``. Real residuals of learned depth are correlated and heavy-tailed; ``n_eff`` and the
threshold must be calibrated on clean data.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2, norm


def z_alpha(alpha: float) -> float:
    """Two-sided Gaussian quantile for false-alarm probability ``alpha`` per view."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def z_beta(beta: float) -> float:
    """One-sided Gaussian quantile for missed-detection probability ``beta``."""
    return float(norm.ppf(1.0 - beta))


def mean_shift_power(delta: float, f: float, n_eff: float, alpha: float = 0.05) -> float:
    """Detection power of the two-sided mean-shift test for a bias ``delta`` on a fraction ``f``."""
    mu = f * delta * math.sqrt(n_eff)
    za = z_alpha(alpha)
    return float(norm.sf(za - mu) + norm.cdf(-za - mu))


def mdb_mean_shift(f: float, n_eff: float, alpha: float = 0.05, beta: float = 0.1) -> float:
    """Minimum detectable bias (units of sigma) of the mean-shift statistic: (za+zb)/(f sqrt(n_eff))."""
    return (z_alpha(alpha) + z_beta(beta)) / (f * math.sqrt(n_eff))


def mdb_energy(f: float, n_eff: float, alpha: float = 0.05, beta: float = 0.1) -> float:
    """Leading-order MDB (rms amplitude, sigma units) of the energy statistic for a zero-mean field.

    Scales as n_eff**-1/4: detects a zero-mean warp that the mean-shift statistic cannot see.
    """
    return math.sqrt(math.sqrt(2.0) * (z_alpha(alpha) + z_beta(beta)) / (f * math.sqrt(n_eff)))


def pixel_flag_prob(delta: float, k: float = 3.0) -> float:
    """Probability that a threshold-``k`` pixel filter flags a pixel carrying a bias ``delta``."""
    return float(norm.cdf(delta - k) + norm.cdf(-delta - k))


def pixel_false_alarm(k: float = 3.0) -> float:
    return float(2.0 * norm.cdf(-k))


def pixel_excess_second_order(delta: float, k: float = 3.0) -> float:
    """Second-order excess over the null: k * phi(k) * delta**2."""
    return float(k * norm.pdf(k) * delta**2)


def count_test_snr(delta: float, f: float, n_eff: float, k: float = 3.0) -> float:
    """Shift (in standard errors) of the aggregated flagged-pixel count test."""
    pf = pixel_false_alarm(k)
    return f * (pixel_flag_prob(delta, k) - pf) * math.sqrt(n_eff) / math.sqrt(pf * (1.0 - pf))


def mean_shift_snr(delta: float, f: float, n_eff: float) -> float:
    """Shift (in standard errors) of the mean-shift statistic: f * delta * sqrt(n_eff)."""
    return f * delta * math.sqrt(n_eff)


def count_over_mean_ratio(delta: float, k: float = 3.0) -> float:
    """R(delta) = count-test SNR / mean-shift SNR (independent of f and n_eff)."""
    pf = pixel_false_alarm(k)
    return (pixel_flag_prob(delta, k) - pf) / (delta * math.sqrt(pf * (1.0 - pf)))


def crossover_bias(k: float = 3.0, lo: float = 0.5, hi: float = 4.0) -> float:
    """Bias (sigma units) above which the aggregated count beats the mean shift; root of R=1."""
    return float(brentq(lambda d: count_over_mean_ratio(d, k) - 1.0, lo, hi))


def protection_level(sensitivities, mdbs, sigma: float, sigma_q: float, z_q: float = 1.96) -> float:
    """Conditional protection level PL = max_i |s_i| * MDB_i * sigma + z_q * sigma_q (single fault)."""
    worst = max(abs(s) * m for s, m in zip(sensitivities, mdbs, strict=True))
    return worst * sigma + z_q * sigma_q


# ---------------------------------------------------------------------------------------------
# Layer 1: whitened, nuisance-projected theory from the explicit measurement model
# ---------------------------------------------------------------------------------------------


def n_eff_from_correlation(C: np.ndarray) -> float:
    """Effective number of independent pixels of the energy statistic: tr(C)^2 / tr(C^2).

    ``C`` is the correlation matrix of the (diagonally standardised) residuals. It is the
    participation ratio of the eigenvalues of ``C`` and equals ``Var``-inverse of the energy
    statistic of standardised residuals: Var[rho^T rho / n] = 2 / n_eff. n for C = I, 1 for C = 11^T.
    """
    tr = float(np.trace(C))
    return tr * tr / float(np.sum(C * C))  # tr(C^2) = sum_ij C_ij^2 for symmetric C


def n_mean_from_correlation(C: np.ndarray) -> float:
    """Effective number of independent pixels of a *mean* statistic: n^2 / (1^T C 1)."""
    n = C.shape[0]
    return n * n / float(C.sum())


def gaussian_correlation_grid(h: int, w: int, ell: float, nugget: float = 0.0) -> np.ndarray:
    """Correlation matrix C = (1 - nugget) * K_ell + nugget * I of an (h, w) pixel grid.

    K_ell is the Gaussian kernel exp(-|u - v|^2 / (2 ell^2)) (ell in pixels). The nugget is the
    fraction of the noise variance that is independent from pixel to pixel; without it the
    covariance is numerically singular and any sharp-edged fault pattern looks infinitely
    detectable, which no real depth noise supports.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    P = np.stack([yy.ravel(), xx.ravel()], 1).astype(float)
    K = np.exp(-(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)) / (2.0 * ell**2))
    return (1.0 - nugget) * K + nugget * np.eye(h * w)


def n_eff_gaussian_kernel_2d(n: float, ell: float, nugget: float = 0.0) -> float:
    """Closed form n_eff ~ n / [(1-nu)^2 pi ell^2 + 2 nu (1-nu) + nu^2] (2-D Gaussian kernel + nugget nu)."""
    nu = nugget
    return n / ((1 - nu) ** 2 * math.pi * ell**2 + 2 * nu * (1 - nu) + nu**2)


def n_mean_gaussian_kernel_2d(n: float, ell: float, nugget: float = 0.0) -> float:
    """Closed form n_mean ~ n / [(1-nu) 2 pi ell^2 + nu] for a 2-D Gaussian kernel + nugget nu."""
    return n / ((1 - nugget) * 2.0 * math.pi * ell**2 + nugget)


def delete_group_jackknife_cov(surfaces: np.ndarray) -> np.ndarray:
    """Delete-a-group jackknife covariance of the surface prediction over K cross-fitted surfaces.

    ``surfaces`` is (K, n): the depth rendered at the view by each of the K surfaces. Returns
    ((K-1)/K) * sum_k (s_k - mean)(s_k - mean)^T, an (n, n) matrix of rank <= K-1.
    """
    k = surfaces.shape[0]
    dev = surfaces - surfaces.mean(0, keepdims=True)
    return (k - 1) / k * dev.T @ dev


def residual_covariance(
    sigma: np.ndarray,
    C: np.ndarray,
    J_pose: np.ndarray | None = None,
    Sigma_q: np.ndarray | None = None,
    J_K: np.ndarray | None = None,
    Sigma_K: np.ndarray | None = None,
    Sigma_surf: np.ndarray | None = None,
    floor: float = 0.0,
) -> np.ndarray:
    """Sigma_r = D C D + J_pose Sigma_q J_pose^T + J_K Sigma_K J_K^T + Sigma_surf (+ floor * I).

    ``sigma`` is the per-pixel standard deviation sigma(range, texture, grazing) (the diagonal of
    D); ``C`` the spatial correlation matrix of the depth noise (length scale ell, with a nugget: see
    ``gaussian_correlation_grid``).
    """
    Sig = (sigma[:, None] * C) * sigma[None, :]
    if J_pose is not None and Sigma_q is not None:
        Sig = Sig + J_pose @ Sigma_q @ J_pose.T
    if J_K is not None and Sigma_K is not None:
        Sig = Sig + J_K @ Sigma_K @ J_K.T
    if Sigma_surf is not None:
        Sig = Sig + Sigma_surf
    if floor:
        Sig = Sig + floor * np.eye(Sig.shape[0])
    return 0.5 * (Sig + Sig.T)


def projected_precision(Sigma: np.ndarray, A: np.ndarray) -> np.ndarray:
    """M = S^-1 - S^-1 A (A^T S^-1 A)^-1 A^T S^-1: whitened precision with the nuisance removed.

    ``A`` (n, p) spans the explained nuisance (monocular scale and offset: columns [d, 1]). M
    annihilates A (M A = 0), so a fault pattern inside span(A) is invisible and benign.
    """
    Si = np.linalg.inv(Sigma)
    SiA = Si @ A
    return Si - SiA @ np.linalg.solve(A.T @ SiA, SiA.T)


def kappa_projected(g: np.ndarray, Sigma: np.ndarray, A: np.ndarray) -> float:
    """Detectability index kappa = sqrt(g^T M g) of a fault pattern ``g`` (units: per unit beta)."""
    M = projected_precision(Sigma, A)
    return float(math.sqrt(max(float(g @ M @ g), 0.0)))


def whitened_energy(r0: np.ndarray, Sigma: np.ndarray, A: np.ndarray) -> tuple[float, int]:
    """Energy statistic T = r0^T M r0 (chi-square with n - rank(A) dof under H0) and its dof."""
    M = projected_precision(Sigma, A)
    return float(r0 @ M @ r0), int(r0.shape[0] - np.linalg.matrix_rank(A))


def noncentrality_for_power(dof: int, alpha: float = 0.05, beta: float = 0.1) -> float:
    """Noncentrality lambda_0 at which the one-sided energy test has missed-detection prob. beta."""
    thr = chi2.isf(alpha, dof)
    return float(brentq(lambda lam: ncx2.sf(thr, dof, lam) - (1.0 - beta), 1e-6, 50.0 * dof))


def mdb_matched_projected(kappa: float, alpha: float = 0.05, beta: float = 0.1) -> float:
    """MDB (sigma units) of the matched statistic for a known pattern: (z_a + z_b) / kappa."""
    return float("inf") if kappa <= 0 else (z_alpha(alpha) + z_beta(beta)) / kappa


def mdb_energy_projected(kappa: float, dof: int, alpha: float = 0.05, beta: float = 0.1) -> float:
    """MDB of the pattern-agnostic energy statistic: sqrt(lambda_0) / kappa."""
    return (
        float("inf") if kappa <= 0 else math.sqrt(noncentrality_for_power(dof, alpha, beta)) / kappa
    )


def mdb_scan(kappa: float, n_regions: int, alpha: float = 0.05, beta: float = 0.1) -> float:
    """MDB of a matched statistic scanned over ``n_regions`` candidate patterns (Bonferroni)."""
    if kappa <= 0:
        return float("inf")
    za = float(norm.ppf(1.0 - alpha / (2.0 * n_regions)))
    return (za + z_beta(beta)) / kappa


def pose_jacobian_depth(ray: np.ndarray, normal: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Jacobian (n, 6) of rendered depth with respect to a camera pose perturbation [dt, dw].

    ``ray`` (n, 3) unit pixel rays in the camera frame, ``normal`` (n, 3) unit surface normals
    oriented so that normal . ray > 0, ``r`` (n,) depth along the ray. First-order ray-plane result:
    d r / d t = -nu^T / (nu . l),  d r / d w = -r (l x nu)^T / (nu . l). The rotation row has norm
    r tan(theta) with theta the incidence angle: the pointing-error term of the residual variance.
    """
    c = np.sum(normal * ray, axis=1)
    jt = -normal / c[:, None]
    jw = -r[:, None] * np.cross(ray, normal) / c[:, None]
    return np.concatenate([jt, jw], axis=1)


def protection_level_ellipsoid(
    s: np.ndarray,
    Sigma: np.ndarray,
    A: np.ndarray,
    lam0: float,
    sigma_q: float = 0.0,
    z_q: float = 1.96,
) -> float:
    """Conditional protection level for a linear output q = q0 + s^T phi (single faulty view).

    Faults that the energy test misses (noncentrality below ``lam0``) lie in the ellipsoid
    phi^T M phi <= lam0, so the worst-case error is sqrt(lam0 * s^T Sigma_proj s) with
    Sigma_proj = Sigma - A (A^T Sigma^-1 A)^-1 A^T; plus the noise term z_q * sigma_q.
    """
    Si = np.linalg.inv(Sigma)
    Sproj = Sigma - A @ np.linalg.solve(A.T @ Si @ A, A.T)
    return float(math.sqrt(max(lam0 * float(s @ Sproj @ s), 0.0)) + z_q * sigma_q)


def sigma_surf_diag(a: float, sigma2_comp: np.ndarray) -> np.ndarray:
    """Diagonal Sigma_surf = a^2 diag(sigma^2(u)) from the composite variance of the closed-form
    surface (leave-view-out), replacing the cross-fitted jackknife covariance.

    Diagonal by assumption: spatial correlation of the surface error is ignored (stated limitation).
    Pass the result as ``Sigma_surf`` to :func:`residual_covariance`.
    """
    return np.diag(a * a * np.asarray(sigma2_comp, dtype=float))
