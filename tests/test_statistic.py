"""CPU toy checks of the scale/shift-fitted fault statistic (synthetic Gaussian world, small grid)."""
import numpy as np
import pytest
from scipy.linalg import solve_triangular

from dcal.stat import _whiten, align_samples, empirical_threshold, fit_scale_shift, fault_statistic
from dcal.toy import ToyConfig, World

G = 12


@pytest.fixture(scope="module")
def world():
    return World(ToyConfig(g=G), 0)


def _T(world, fault=None, kappa=0.0, area=0.12, robust=True, n=100, seed=0):
    rng = np.random.default_rng(seed)
    out, ab = [], []
    for _ in range(n):
        S, D, Kr, Ku, (a, b) = world.simulate(rng, fault, kappa, area)
        r = fault_statistic(S, D, Kr, Sigma_u=Ku, robust=robust)
        out.append(r["T"])
        ab.append((r["s"] - a, r["t"] - b))
    return np.array(out), np.array(ab)


def test_fit_cost_is_two_dof_when_covariance_is_fixed(world):
    """With Sigma held fixed the GLS fit of (s, t) lowers T by ~chi2_2 (mean 2), the two-parameter cost."""
    rng = np.random.default_rng(3)
    M, n = world.cfg.M, world.n
    diffs = []
    for _ in range(300):
        a, b = rng.uniform(0.8, 1.25), rng.uniform(-1, 1)
        dtrue = world.mu + world.Lu @ rng.standard_normal(n)
        D = a * dtrue + b + world.Lr @ rng.standard_normal(n)
        m = (world.mu[None] + (world.Lu @ rng.standard_normal((n, M))).T).mean(0)
        L = _whiten(a * a * (1 + 1 / M) * world.Ku + world.Kr)
        G_ = solve_triangular(L, np.stack([m, np.ones(n)], 1), lower=True)
        y = solve_triangular(L, D, lower=True)
        th = np.linalg.lstsq(G_, y, rcond=None)[0]
        diffs.append(((y - G_ @ [a, b]) ** 2).sum() - ((y - G_ @ th) ** 2).sum())
    assert 1.6 < np.mean(diffs) < 3.2   # 2 plus a small errors-in-variables excess


def test_null_mean_is_near_dof_with_oracle_covariance(world):
    T, _ = _T(world, n=120, seed=1)
    ratio = T.mean() / (world.n - 2)
    assert 0.88 < ratio < 1.08, ratio


def test_empirical_threshold_controls_false_alarm(world):
    null, _ = _T(world, n=150, seed=2)
    thr = empirical_threshold(null, 0.05)
    fresh, _ = _T(world, n=150, seed=99)
    assert (fresh > thr).mean() < 0.13


def test_local_fault_is_detected(world):
    null, _ = _T(world, n=100, seed=4)
    thr = empirical_threshold(null, 0.05)
    T, _ = _T(world, "square", 4.0, n=60, seed=5)
    assert (T > thr).mean() > 0.9


def test_constant_shift_fault_is_not_detected(world):
    """Blind spot by design: a constant offset is absorbed exactly by the fitted shift."""
    null, _ = _T(world, n=100, seed=4)
    thr = empirical_threshold(null, 0.05)
    T, ab = _T(world, "shift", 8.0, n=60, seed=6)
    assert (T > thr).mean() < 0.2
    assert abs(ab[:, 1].mean()) > 2.0   # the fit moved t instead (render units)


def test_robust_fit_recovers_scale_shift_better_under_a_large_fault(world):
    _, ab_r = _T(world, "square", 4.0, area=0.30, robust=True, n=60, seed=7)
    _, ab_n = _T(world, "square", 4.0, area=0.30, robust=False, n=60, seed=7)
    err = lambda ab: np.sqrt((ab ** 2).sum(1)).mean()   # noqa: E731
    assert err(ab_r) < err(ab_n)


def test_alignment_removes_per_sample_affine_wobble(world):
    rng = np.random.default_rng(8)
    S, *_ = world.simulate(rng)
    A = align_samples(S)
    spread_before = (S - S.mean(0)).std()
    spread_after = (A - A.mean(0)).std()
    assert spread_after < spread_before
