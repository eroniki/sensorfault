"""Tests for the cubic B-spline trajectory prior of survey flights."""

import numpy as np
import pytest

from sensorfault.eval import trajectory_spline as ts
from sensorfault.eval.gaussian_uncertainty import so3_exp


def test_basis_is_a_partition_of_unity_and_reproduces_lines():
    for u in np.linspace(0, 0.999, 7):
        b = ts.basis(u)
        assert b.sum() == pytest.approx(1.0) and (b >= -1e-12).all()
        assert b @ np.arange(4) == pytest.approx(u + 1.0)  # linear precision
        assert ts.cumulative_basis(u)[0] == pytest.approx(1.0)
    ctrl = np.stack([np.arange(10.0) * 2.0, np.ones(10), np.full(10, 100.0)], axis=1)
    for t in (0.5, 3.2, 6.9):
        p = ts.eval_position(ctrl, t, t0=-1.0, dt=1.0)
        # straight, constant-speed strip: positions are exactly linear in time (s = t - t0 = t + 1)
        assert p == pytest.approx([2.0 * (t + 2.0), 1.0, 100.0], abs=1e-9)


def _lawnmower(times):
    """Two strips at 10 m/s and 120 m altitude joined by a semicircular turn of radius 25 m."""
    tt = 10.0 + np.pi * 2.5  # end of the turn
    phi = np.clip((times - 10.0) * 10.0 / 25.0, 0.0, np.pi)
    x = np.where(
        times < 10,
        10 * times,
        np.where(times < tt, 100 + 25 * np.sin(phi), 100 - 10 * (times - tt)),
    )
    y = np.where(times < 10, 0.0, np.where(times < tt, 25 - 25 * np.cos(phi), 50.0))
    return np.stack([x, y, np.full_like(times, 120.0)], axis=1)


def test_fit_to_noisy_gnss_reduces_error_on_a_lawnmower_flight():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 27.5, 276)
    truth = _lawnmower(times)
    gnss = truth + rng.normal(0, 0.5, truth.shape)
    n_ctrl, t0, dt = 31, -1.0, 1.0
    ctrl, L = ts.fit_positions(times, gnss, 0.5, n_ctrl, t0, dt, smooth=1e-3)
    fit = np.stack([ts.eval_position(ctrl, t, t0, dt) for t in times])
    assert np.sqrt(np.mean((fit - truth) ** 2)) < 0.7 * np.sqrt(np.mean((gnss - truth) ** 2))


def test_position_covariance_matches_monte_carlo():
    rng = np.random.default_rng(1)
    times = np.linspace(0, 9.9, 100)
    truth = _lawnmower(times)
    n_ctrl, t0, dt, s = 13, -1.0, 1.0, 0.5
    _, L = ts.fit_positions(times, truth, s, n_ctrl, t0, dt, smooth=1e-3)
    t = 4.3
    v = ts.position_cov(L, t, t0, dt)
    est = []
    for _ in range(400):
        ctrl, _ = ts.fit_positions(
            times, truth + rng.normal(0, s, truth.shape), s, n_ctrl, t0, dt, 1e-3
        )
        est.append(ts.eval_position(ctrl, t, t0, dt)[0])
    assert np.var(est) == pytest.approx(v, rel=0.2)


def test_tested_view_information_is_removed_by_subtraction():
    rng = np.random.default_rng(2)
    times = np.linspace(0, 9.9, 50)
    n_ctrl, t0, dt = 13, -1.0, 1.0
    rows = np.stack([ts.position_row(t, t0, dt, n_ctrl) for t in times])
    L_prior = 1e-3 * ts.second_difference(n_ctrl).T @ ts.second_difference(n_ctrl) + 1e-6 * np.eye(
        n_ctrl
    )
    L_views = [np.outer(r, r) / 0.1**2 for r in rows]  # one render-and-compare term per view
    L_all = L_prior + sum(L_views)
    j = 20
    L_wo = L_all - L_views[j]
    L_ref = L_prior + sum(L for i, L in enumerate(L_views) if i != j)
    assert np.allclose(L_wo, L_ref)
    # the tested view's pose is less certain without its own depth term
    assert ts.position_cov(L_wo, times[j], t0, dt) > ts.position_cov(L_all, times[j], t0, dt)
    assert rng is not None


def test_cumulative_rotation_spline_is_constant_and_interpolates_constant_rate_yaw():
    R0 = so3_exp(np.array([0.1, -0.2, 0.3]))
    const = [R0] * 8
    assert np.allclose(ts.eval_rotation(const, 2.4, -1.0, 1.0), R0)
    step = 0.2
    yaw = [so3_exp(np.array([0.0, 0.0, step * k])) for k in range(8)]
    for t in (0.3, 1.7, 3.5):
        R = ts.eval_rotation(yaw, t, -1.0, 1.0)
        assert ts.log_so3(R)[2] == pytest.approx(step * (t + 2.0), abs=1e-9)
