"""Tests for the closed-form Gaussian-primitive surface (depth maps only)."""

import numpy as np
import pytest

from sensorfault.eval import surface_model as sm

K = np.array([[100.0, 0, 16], [0, 100, 12], [0, 0, 1]])


def _rot(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = sm.skew(w / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def test_plane_backprojection_normals_and_extent():
    d = np.full((24, 32), 2.0)
    p = sm.primitives_from_depth(d, K, np.eye(3), np.zeros(3), 0.01)
    assert np.allclose(np.abs(p["n"][:, 2]), 1.0, atol=1e-6)
    assert np.allclose(p["sigma_h2"][0], 1e-4)
    # tilted plane z = 2 + 0.5 x: extent stretched by 1/cos(theta)
    d2 = np.empty((24, 32))
    uu = (np.arange(32) - 16) / 100.0
    # solve z = 2 + 0.5 z*uu -> z = 2 / (1 - 0.5 uu)
    d2[:] = 2.0 / (1 - 0.5 * uu)[None, :]
    p2 = sm.primitives_from_depth(d2, K, np.eye(3), np.zeros(3), 0.01)
    ratio = p2["s_par"] / p2["s_perp"]
    assert np.allclose(ratio, 1 / np.maximum(p2["cos_theta"], 0.1))
    assert ratio.max() > 1.0


def test_jacobians_finite_difference():
    R = _rot(np.array([0.1, -0.2, 0.3]))
    t = np.array([0.3, -0.1, 0.2])
    d, u, v = 1.7, 10.0, 7.0
    r = np.linalg.inv(K) @ np.array([u, v, 1.0])
    mu = R @ (d * r) + t
    J = sm.pose_jacobian_point(R @ (d * r))
    eps = 1e-6
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = eps
        Rn = _rot(dq[3:]) @ R
        mu_n = Rn @ (d * r) + t + dq[:3]
        assert np.allclose((mu_n - mu) / eps, J[:, k], atol=1e-4)
    JK = sm.intrinsics_jacobian_point(d, R, u, v, K)
    for k, (i, j) in enumerate([(0, 0), (0, 2), (1, 2)]):
        K2 = K.copy()
        K2[i, j] += eps
        if k == 0:
            K2[1, 1] += eps
        r2 = np.linalg.inv(K2) @ np.array([u, v, 1.0])
        assert np.allclose((R @ (d * r2) + t - mu) / eps, JK[:, k], atol=1e-4)


def test_offset_variance_monte_carlo():
    rng = np.random.default_rng(0)
    R, t = _rot(np.array([0.0, 0.1, 0.0])), np.array([0.1, 0.0, 0.0])
    d, u, v, sd = 2.0, 20.0, 9.0, 0.02
    r = np.linalg.inv(K) @ np.array([u, v, 1.0])
    n = np.array([0.1, 0.2, -1.0])
    n /= np.linalg.norm(n)
    Sq = np.diag([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5])
    Jq = sm.pose_jacobian_point(R @ (d * r))
    P = sm.point_covariance(R @ r, sd, Jq, Sq)
    s2 = sm.offset_variance(n, P)
    N = 200000
    dd = d + sd * rng.standard_normal(N)
    dq = rng.multivariate_normal(np.zeros(6), Sq, N)
    mu = (dd[:, None] * (R @ r)[None]) + t + dq[:, :3] - np.cross(R @ (d * r), dq[:, 3:]) * 1.0
    # sign: d mu = dt - [R d r]x dw = dt + dw x (R d r)  -> cross(dw, p)
    mu = (dd[:, None] * (R @ r)[None]) + t + dq[:, :3] + np.cross(dq[:, 3:], R @ (d * r))
    assert np.var(mu @ n) == pytest.approx(s2, rel=0.03)


def test_information_fusion_variance_and_loo_exact():
    rng = np.random.default_rng(1)
    lam = np.array([4.0, 9.0, 16.0, 25.0])
    h = np.array([1.0, 1.1, 0.9, 1.05])
    lt, ht = sm.fuse_information(lam, h)
    assert lt == lam.sum()
    N = 100000
    obs = 1.0 + rng.standard_normal((N, 4)) / np.sqrt(lam)
    est = (obs * lam).sum(1) / lt
    assert np.var(est) == pytest.approx(1 / lt, rel=0.03)
    for j in range(4):
        keep = np.arange(4) != j
        lm, hm = sm.leave_one_out(lt, ht, lam[j], h[j])
        lr, hr = sm.fuse_information(lam[keep], h[keep])
        assert lm == pytest.approx(lr) and hm == pytest.approx(hr)
    lm, hm = sm.leave_one_out(5.0, 1.0, 5.0, 1.0)
    assert lm == 0.0 and np.isnan(hm)


def test_solution_separation_equals_innovation():
    rng = np.random.default_rng(2)
    for _ in range(20):
        lam = rng.uniform(1, 30, 5)
        h = rng.normal(size=5)
        lt, ht = sm.fuse_information(lam, h)
        j = rng.integers(5)
        assert sm.solution_separation_stat(lt, ht, lam[j], h[j]) == pytest.approx(
            sm.innovation_stat(lt, ht, lam[j], h[j])
        )


def test_loo_statistic_is_chi2_1_under_null_and_has_power():
    rng = np.random.default_rng(3)
    lam = np.array([10.0, 20.0, 30.0, 40.0])
    N = 20000
    stats, stats_b = [], []
    for _ in range(N):
        h = 1.0 + rng.standard_normal(4) / np.sqrt(lam)
        lt, ht = sm.fuse_information(lam, h)
        stats.append(sm.innovation_stat(lt, ht, lam[0], h[0]))
        hb = h.copy()
        hb[0] += 0.9 * sm.mdb_offset(lam[0], lam[1:].sum(), 0.05, 0.1) / 1.0 / 1.0
        lt, ht = sm.fuse_information(lam, hb)
        stats_b.append(sm.innovation_stat(lt, ht, lam[0], hb[0]))
    assert np.mean(stats) == pytest.approx(1.0, rel=0.05)
    assert np.mean(np.array(stats) > 3.84) == pytest.approx(0.05, abs=0.01)
    assert np.mean(np.array(stats_b) > 3.84) > 0.6


def test_ray_plane_depth_and_jacobian():
    R = _rot(np.array([0.05, 0.1, -0.05]))
    c = np.array([0.2, -0.1, 0.0])
    n = np.array([0.1, 0.0, -1.0])
    n /= np.linalg.norm(n)
    h = float(n @ np.array([0.0, 0.0, 2.0]))
    r = np.linalg.inv(K) @ np.array([14.0, 12.0, 1.0])
    d = sm.ray_plane_depth(h, n, c, R, r)
    x = c + d * (R @ r)
    assert n @ x == pytest.approx(h)
    J = sm.ray_plane_pose_jacobian(h, n, c, R, r)
    eps = 1e-6
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = eps
        Rn = _rot(dq[3:]) @ R
        dn = sm.ray_plane_depth(h, n, c + dq[:3], Rn, r)
        assert (dn - d) / eps == pytest.approx(J[k], abs=1e-4)
    Sq = np.diag([1e-4] * 3 + [1e-5] * 3)
    v = sm.ray_plane_depth_variance(1e-4, n, R, r, J, Sq)
    assert v == pytest.approx(1e-4 / (n @ (R @ r)) ** 2 + J @ Sq @ J)


def test_extent_weights_ignore_uncertainty_and_composite():
    t1, t2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    g = sm.extent_weight(np.array([0.01, 0, 0]), t1, t2, 0.02, 0.01)
    assert g == pytest.approx(np.exp(-0.125))
    # no sigma argument exists: confident and uncertain primitives of equal extent weigh equally
    alpha = np.array([0.8, 0.8])
    G = np.array([g, g])
    w = sm.alpha_composite_weights(alpha, G, np.array([0, 1]))
    assert w[0] == pytest.approx(0.8 * g)
    assert w[1] == pytest.approx(0.8 * g * (1 - 0.8 * g))
    d, var, W = sm.composite(w, np.array([2.0, 2.1]), np.array([1e-4, 4e-4]))
    assert d == pytest.approx((w * [2.0, 2.1]).sum() / w.sum())
    assert var == pytest.approx((w * [1e-4, 4e-4]).sum() / w.sum())
    _, var2, _ = sm.composite(w, np.array([2.0, 2.1]), np.array([1e-4, 4e-4]), True)
    assert var2 > var
    assert np.isnan(sm.composite(np.zeros(2), np.ones(2), np.ones(2))[0])


def test_gate_residual_calibration_and_power():
    rng = np.random.default_rng(4)
    a, b, sd, s2 = 1.05, 0.02, 0.01, 4e-4
    N = 50000
    dtrue = rng.uniform(1, 3, N)
    dcomp = dtrue + np.sqrt(s2) * rng.standard_normal(N)  # composite estimate with variance s2
    dhat = a * dtrue + b + sd * rng.standard_normal(N)
    r, v = sm.gate_residual(dhat, a, b, dcomp, s2, sd**2)
    z = r / np.sqrt(v)
    assert np.std(z) == pytest.approx(1.0, rel=0.03)
    zb = (r + 3 * np.sqrt(v)) / np.sqrt(v)
    assert np.mean(np.abs(zb) > 1.96) > 0.8


def test_protection_level_closed_form_and_equal_info():
    lam = np.full(5, 100.0)
    pl = sm.primitive_protection_level(lam, 0.05, 0.1, 1.96)
    za, zb = 1.959964, 1.281552
    assert pl == pytest.approx((za + zb) / np.sqrt(500 * 4) + 1.96 / np.sqrt(500), rel=1e-4)
    assert sm.primitive_protection_level(np.array([5.0]), 0.05, 0.1) == np.inf
    # Monte Carlo: undetected single-view fault stays below the PL at rate >= 1 - beta-ish
    rng = np.random.default_rng(5)
    lt = lam.sum()
    bias = sm.mdb_offset(100.0, 400.0)
    N = 20000
    err, det = [], []
    for _ in range(N):
        h = rng.standard_normal(5) / 10
        h[0] += bias
        lt_, ht = sm.fuse_information(lam, h)
        det.append(sm.innovation_stat(lt_, ht, lam[0], h[0]) > 3.84)
        err.append(abs(ht))
    assert 1 - np.mean(det) == pytest.approx(0.1, abs=0.02)
    assert np.mean(np.array(err)[~np.array(det)] < pl) > 0.9
    assert lt == 500.0


def test_region_outputs_and_testable():
    out = sm.region_outputs(np.array([10.0, 10.0]), np.array([1.0, 1.0]))
    assert out["n_views"] == 2 and out["testable"] is False
    out = sm.region_outputs(np.array([10.0, 10.0, 10.0]), np.array([1.0, 1.0, 1.0]))
    assert out["testable"] and out["posterior_var"] == pytest.approx(1 / 30)


def test_association_gate():
    n = np.array([0.0, 0.0, -1.0])
    mu = np.array([0.0, 0.0, 2.0])
    tp = np.array([1.0, 0, 0])
    args = dict(sigma_h2_new=1e-4, mu_k=mu, n_k=n, h_k=-2.0, sigma_h2_k=1e-4, t_par_k=tp,
                s_par_k=0.02, s_perp_k=0.02)
    assert sm.associates(mu + [0.01, 0, 0], n, **args)
    assert not sm.associates(mu + [0, 0, 0.2], n, **args)  # offset far outside the gate
    assert not sm.associates(mu + [0.5, 0, 0], n, **args)  # outside extent
    n2 = np.array([0.8, 0.0, -0.6])
    assert not sm.associates(mu, n2, **args)  # normals disagree
