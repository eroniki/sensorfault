import numpy as np
import pytest

from sensorfault.eval import calibration as cal


def _honest(n, rng, scale=1.0):
    A = rng.normal(size=(n, 3, 3))
    Lam = A @ np.transpose(A, (0, 2, 1)) + 0.5 * np.eye(3)
    L = np.linalg.cholesky(np.linalg.inv(Lam))
    eps = scale * np.einsum("nij,nj->ni", L, rng.normal(size=(n, 3)))
    return eps, Lam


def test_honest_placements_recover_unit_scale_and_nominal_coverage():
    rng = np.random.default_rng(1)
    eps, Lam = _honest(5000, rng)
    z2 = cal.z2(eps, Lam)
    assert abs(cal.robust_scale(z2) - 1) < 0.03
    assert abs(cal.tail_scale(z2) - 1) < 0.03
    assert abs(cal.coverage(z2, 0.683) - 0.683) < 0.01
    assert abs(cal.coverage(z2, 0.95) - 0.95) < 0.01
    assert cal.ece(z2) < 0.02
    assert cal.calibration_loss(z2) < 0.01


def test_overconfident_placements_show_their_factor():
    rng = np.random.default_rng(2)
    eps, Lam = _honest(5000, rng, scale=1.6)
    z2 = cal.z2(eps, Lam)
    assert abs(cal.robust_scale(z2) - 1.6) < 0.05
    assert cal.coverage(z2, 0.95) < 0.8
    assert cal.calibration_loss(z2) > 0.5


def test_log_score_matches_gaussian_density():
    rng = np.random.default_rng(3)
    eps, Lam = _honest(200, rng)
    z2, ld = cal.z2(eps, Lam), cal.logdet(Lam)
    from scipy import stats

    ref = -np.mean(
        [
            stats.multivariate_normal(np.zeros(3), np.linalg.inv(L)).logpdf(e)
            for e, L in zip(eps, Lam)
        ]
    )
    assert cal.log_score(z2, ld) == pytest.approx(ref - 1.5 * np.log(2 * np.pi), rel=1e-9)


def test_rescaling_by_calibration_constant_removes_calibration_loss():
    rng = np.random.default_rng(4)
    eps, Lam = _honest(4000, rng, scale=1.4)
    z2, ld = cal.z2(eps, Lam), cal.logdet(Lam)
    c = np.mean(z2) / 3
    z2c, ldc = z2 / c, ld - 3 * np.log(c)
    assert cal.log_score(z2, ld) - cal.log_score(z2c, ldc) == pytest.approx(
        cal.calibration_loss(z2), rel=1e-9
    )


def test_tile_bootstrap_interval_covers_truth_about_95_percent():
    rng = np.random.default_rng(5)
    n_tiles, per = 100, 50
    tiles = np.repeat(np.arange(n_tiles), per)
    hits = 0
    reps = 150
    for r in range(reps):
        eps, Lam = _honest(n_tiles * per, rng)
        z2 = cal.z2(eps, Lam)
        ci = cal.tile_bootstrap(lambda i, z2=z2: cal.robust_scale(z2[i]), tiles, n_boot=300, seed=r)
        hits += ci["lo"] <= 1.0 <= ci["hi"]
    assert 0.88 <= hits / reps <= 0.99


def test_bootstrap_summary_keys():
    rng = np.random.default_rng(6)
    eps, Lam = _honest(600, rng)
    out = cal.bootstrap_summary(cal.z2(eps, Lam), cal.logdet(Lam), np.arange(600) // 20, n_boot=50)
    assert set(out) >= {"s_z", "cov_0.95", "ece", "log_score"}
    assert out["s_z"]["lo"] <= out["s_z"]["estimate"] <= out["s_z"]["hi"]
