"""Tests for the whitened, nuisance-projected detection theory built on the measurement model."""

import numpy as np
import pytest
from scipy.stats import chi2, norm

from sensorfault.eval import detection_theory as dt


def corr_grid(h, w, ell, nugget=0.0):
    return dt.gaussian_correlation_grid(h, w, ell, nugget)


def test_n_eff_identities_and_closed_forms():
    assert dt.n_eff_from_correlation(np.eye(50)) == pytest.approx(50.0)
    assert dt.n_eff_from_correlation(np.ones((50, 50))) == pytest.approx(1.0)
    assert dt.n_mean_from_correlation(np.eye(50)) == pytest.approx(50.0)
    for ell in (1.5, 2.0, 3.0):
        C = corr_grid(40, 40, ell)
        n = C.shape[0]
        # closed forms ignore the grid boundary: within ~15%
        assert dt.n_eff_from_correlation(C) == pytest.approx(
            dt.n_eff_gaussian_kernel_2d(n, ell), rel=0.15
        )
        assert dt.n_mean_from_correlation(C) == pytest.approx(
            dt.n_mean_gaussian_kernel_2d(n, ell), rel=0.2
        )
        # a mean statistic is hurt twice as much by correlation as the energy statistic
        assert dt.n_mean_from_correlation(C) < dt.n_eff_from_correlation(C)
    # with a nugget the closed forms still hold and n_eff is larger (more independent noise)
    Cn = corr_grid(40, 40, 3.0, nugget=0.2)
    assert dt.n_eff_from_correlation(Cn) == pytest.approx(
        dt.n_eff_gaussian_kernel_2d(1600, 3.0, 0.2), rel=0.15
    )
    assert dt.n_eff_from_correlation(Cn) > dt.n_eff_from_correlation(corr_grid(40, 40, 3.0))


def test_jackknife_surface_covariance_is_low_rank_psd():
    rng = np.random.default_rng(0)
    K, n = 4, 60
    S = dt.delete_group_jackknife_cov(rng.standard_normal((K, n)))
    assert np.linalg.matrix_rank(S) <= K - 1
    assert np.min(np.linalg.eigvalsh(S)) > -1e-9


def test_residual_covariance_has_the_stated_structure():
    rng = np.random.default_rng(1)
    n = 80
    sigma = 0.02 + 0.01 * rng.random(n)
    C = corr_grid(8, 10, 1.5, 0.2)
    J = rng.standard_normal((n, 6))
    Sq = np.diag([1e-6] * 3 + [1e-8] * 3)
    S0 = dt.residual_covariance(sigma, C)
    S1 = dt.residual_covariance(sigma, C, J_pose=J, Sigma_q=Sq)
    assert np.allclose(S0, (sigma[:, None] * C) * sigma[None, :])
    assert np.linalg.matrix_rank(S1 - S0, tol=1e-12) <= 6  # pose adds a rank-6 term
    assert np.min(np.linalg.eigvalsh(S1)) > 0


def test_projection_annihilates_the_nuisance_and_energy_is_chi_square():
    rng = np.random.default_rng(2)
    n = 120
    d = 1.0 + rng.random(n) * 4.0
    A = np.stack([d, np.ones(n)], 1)
    sigma = 0.02 + 0.01 * d
    C = corr_grid(10, 12, 2.0, 0.2)
    Sig = dt.residual_covariance(sigma, C)
    M = dt.projected_precision(Sig, A)
    assert np.abs(M @ A).max() < 1e-6 * np.abs(np.linalg.inv(Sig)).max()
    L = np.linalg.cholesky(Sig)
    trials = 4000
    T = []
    for _ in range(trials):
        r0 = L @ rng.standard_normal(n) + A @ np.array([0.03, -0.2])  # nuisance is arbitrary
        t, dof = dt.whitened_energy(r0, Sig, A)
        T.append(t)
    T = np.array(T)
    assert dof == n - 2
    assert T.mean() == pytest.approx(dof, rel=0.03)
    assert np.mean(chi2.isf(0.05, dof) < T) == pytest.approx(0.05, abs=0.012)


def test_uniform_bias_over_a_whole_view_is_benign_but_a_region_is_not():
    n = 1000
    A = np.ones((n, 1))
    Sig = np.eye(n)
    f = 0.5
    mask = np.zeros(n)
    mask[: int(f * n)] = 1.0
    assert dt.kappa_projected(mask, Sig, A) ** 2 == pytest.approx(n * f * (1 - f), rel=1e-6)
    assert dt.kappa_projected(np.ones(n), Sig, A) == pytest.approx(0.0, abs=1e-6)
    # sigma-proportional bias with sigma affine in depth lies in span([d, 1]) -> absorbed
    d = np.linspace(1, 5, n)
    A2 = np.stack([d, np.ones(n)], 1)
    g = 0.02 + 0.01 * d
    assert dt.kappa_projected(g, np.diag(g**2), A2) == pytest.approx(0.0, abs=1e-6)


def test_detectability_peaks_at_half_and_correlation_lowers_it():
    n = 400
    A = np.ones((n, 1))
    ks = []
    for f in (0.05, 0.2, 0.5, 0.8, 0.95):
        m = np.zeros(n)
        m[: int(f * n)] = 1.0
        ks.append(dt.kappa_projected(m, np.eye(n), A))
    assert ks[2] == max(ks)
    # a contiguous half-region of a correlated field is harder to detect than iid noise
    C = corr_grid(20, 20, 3.0, 0.2)
    m = np.zeros(n)
    m[:200] = 1.0
    assert dt.kappa_projected(m, C, A) < dt.kappa_projected(m, np.eye(n), A)


def test_mdb_values_for_the_paper_table():
    n = 1000
    kap = lambda f: float(np.sqrt(n * f * (1 - f)))
    assert dt.mdb_matched_projected(kap(0.5)) == pytest.approx(0.205, abs=2e-3)
    assert dt.mdb_matched_projected(kap(0.2)) == pytest.approx(0.256, abs=2e-3)
    assert dt.mdb_matched_projected(kap(0.05)) == pytest.approx(0.470, abs=3e-3)
    assert dt.mdb_matched_projected(0.0) == float("inf")  # f = 1: benign
    assert dt.mdb_energy_projected(kap(0.5), n - 2) == pytest.approx(0.745, abs=0.01)
    assert dt.mdb_energy_projected(kap(0.2), n - 2) == pytest.approx(0.932, abs=0.01)
    assert dt.mdb_scan(kap(0.5), 100) == pytest.approx(0.301, abs=3e-3)
    assert dt.mdb_scan(kap(0.2), 100) == pytest.approx(0.376, abs=3e-3)
    assert dt.noncentrality_for_power(998) == pytest.approx(138.8, abs=0.5)


def test_matched_statistic_power_matches_theory_under_correlated_noise():
    rng = np.random.default_rng(3)
    n = 150
    A = np.ones((n, 1))
    C = corr_grid(10, 15, 2.0, 0.2)
    Sig = dt.residual_covariance(np.full(n, 0.05), C)
    M = dt.projected_precision(Sig, A)
    g = np.zeros(n)
    g[:60] = 0.05  # region bias pattern, per unit beta (units of sigma)
    kap = float(np.sqrt(g @ M @ g))
    beta = 0.9 * (z := (dt.z_alpha(0.05) + dt.z_beta(0.1))) / kap  # near but above the MDB
    L = np.linalg.cholesky(Sig)
    Mg = M @ g
    zs0, zs1 = [], []
    for _ in range(6000):
        e = L @ rng.standard_normal(n)
        zs0.append(Mg @ e / kap)
        zs1.append(Mg @ (e + beta * g) / kap)
    za = dt.z_alpha(0.05)
    assert np.mean(np.abs(zs0) > za) == pytest.approx(0.05, abs=0.012)
    theory = norm.sf(za - beta * kap) + norm.cdf(-za - beta * kap)
    assert np.mean(np.abs(zs1) > za) == pytest.approx(theory, abs=0.02)
    assert z > 0


def test_pose_jacobian_matches_exact_ray_plane_and_rotation_row_is_r_tan_theta():
    rng = np.random.default_rng(4)
    n = 5
    ray = rng.standard_normal((n, 3))
    ray[:, 2] = np.abs(ray[:, 2]) + 1.0
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    nu = rng.standard_normal((n, 3))
    nu[:, 2] = -np.abs(nu[:, 2]) - 1.0
    nu /= np.linalg.norm(nu, axis=1, keepdims=True)
    nu = np.where(np.sum(nu * ray, 1, keepdims=True) < 0, -nu, nu)
    r = 1.0 + 4.0 * rng.random(n)
    c = r * np.sum(nu * ray, 1)
    J = dt.pose_jacobian_depth(ray, nu, r)

    def depth(dt_, dw):
        return (c - nu @ dt_) / (np.sum(nu * ray, 1) + np.cross(ray, nu) @ dw)

    eps = 1e-7
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        assert np.allclose((depth(e, np.zeros(3)) - r) / eps, J[:, i], atol=1e-5)
        assert np.allclose((depth(np.zeros(3), e) - r) / eps, J[:, 3 + i], atol=1e-5)
    theta = np.arccos(np.sum(nu * ray, 1))
    assert np.allclose(np.linalg.norm(J[:, 3:], axis=1), r * np.tan(theta), rtol=1e-6)


def test_protection_level_ellipsoid_is_tight_and_never_exceeded():
    rng = np.random.default_rng(5)
    n = 40
    A = np.ones((n, 1))
    Sig = 0.04 * np.eye(n)
    s = rng.standard_normal(n)
    lam0 = 25.0
    pl = dt.protection_level_ellipsoid(s, Sig, A, lam0)
    # closed form for iid noise with an offset nuisance: sqrt(lam0) * sigma * ||s - mean(s)||
    assert pl == pytest.approx(np.sqrt(lam0) * 0.2 * np.linalg.norm(s - s.mean()), rel=1e-9)
    M = dt.projected_precision(Sig, A)
    worst = 0.0
    for _ in range(3000):
        phi = rng.standard_normal(n)
        phi *= np.sqrt(lam0 / (phi @ M @ phi))  # on the boundary of the undetected ellipsoid
        worst = max(worst, float(s @ phi))
    assert worst <= pl * (1 + 1e-9)
    # the maximiser attains the bound
    phi_star = (Sig - A @ np.linalg.solve(A.T @ np.linalg.inv(Sig) @ A, A.T)) @ s
    phi_star *= np.sqrt(lam0 / (phi_star @ M @ phi_star))
    assert float(s @ phi_star) == pytest.approx(pl, rel=1e-6)
