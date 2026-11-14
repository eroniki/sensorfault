"""The numbers quoted in the manuscript's detection analysis, and a Monte Carlo cross-check."""

import numpy as np
import pytest

from sensorfault.eval import detection_theory as dt


def test_quantiles_for_5pct_false_alarm_and_10pct_miss():
    assert dt.z_alpha(0.05) == pytest.approx(1.96, abs=1e-3)
    assert dt.z_beta(0.1) == pytest.approx(1.2816, abs=1e-3)
    assert dt.z_alpha(0.05) + dt.z_beta(0.1) == pytest.approx(3.242, abs=1e-3)


def test_mdb_table_from_the_paper():
    assert dt.mdb_mean_shift(1.0, 1000) == pytest.approx(0.103, abs=1e-3)
    assert dt.mdb_mean_shift(0.2, 1000) == pytest.approx(0.513, abs=1e-3)
    assert dt.mdb_mean_shift(0.05, 1000) == pytest.approx(2.05, abs=1e-2)
    # MDB falls as 1/sqrt(n_eff) and 1/f
    assert dt.mdb_mean_shift(1.0, 10000) == pytest.approx(dt.mdb_mean_shift(1.0, 1000) / 10**0.5)


def test_energy_mdb_scales_as_quarter_power():
    r = dt.mdb_energy(1.0, 10000) / dt.mdb_energy(1.0, 1000)
    assert r == pytest.approx(10**-0.25, rel=1e-6)
    assert dt.mdb_energy(1.0, 1000) == pytest.approx(0.381, abs=2e-3)


def test_pixel_filter_is_blind_at_subnoise_bias():
    pf = dt.pixel_false_alarm(3.0)
    assert pf == pytest.approx(0.0027, abs=1e-4)
    assert dt.pixel_flag_prob(0.3, 3.0) == pytest.approx(0.00395, abs=2e-5)
    # second-order excess k*phi(k)*delta^2 is accurate at small delta
    ex = dt.pixel_flag_prob(0.3, 3.0) - pf
    assert ex == pytest.approx(dt.pixel_excess_second_order(0.3, 3.0), rel=0.1)
    assert dt.pixel_flag_prob(4.0, 3.0) > 0.8  # large bias: filters work


def test_aggregated_mean_shift_is_about_12x_stronger_at_0p3_sigma():
    zm = dt.mean_shift_snr(0.3, 1.0, 1000)
    zc = dt.count_test_snr(0.3, 1.0, 1000)
    assert zm == pytest.approx(9.49, abs=0.01) and zc == pytest.approx(0.76, abs=0.01)
    assert zm / zc == pytest.approx(12.4, abs=0.1)
    # ratio is independent of f and n_eff
    assert dt.count_over_mean_ratio(0.3) == pytest.approx(zc / zm, rel=1e-6)


def test_crossover_bias_is_about_1p65_sigma():
    assert dt.crossover_bias(3.0) == pytest.approx(1.65, abs=0.01)
    assert dt.count_over_mean_ratio(1.0) < 1.0 < dt.count_over_mean_ratio(2.0)


def test_protection_level_is_slope_times_mdb_plus_noise_term():
    pl = dt.protection_level([0.5, 2.0, 1.0], [0.2, 0.1, 0.3], sigma=0.01, sigma_q=0.002)
    assert pl == pytest.approx(0.3 * 0.01 + 1.96 * 0.002)


def test_mean_shift_power_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n, delta, f, trials = 1000, 0.3, 0.5, 20000
    mask = np.zeros(n)
    mask[: int(f * n)] = delta
    z0 = rng.standard_normal((trials, n)).sum(1) / np.sqrt(n)
    z1 = (rng.standard_normal((trials, n)) + mask).sum(1) / np.sqrt(n)
    thr = dt.z_alpha(0.05)
    assert np.mean(np.abs(z0) > thr) == pytest.approx(0.05, abs=0.01)
    assert np.mean(np.abs(z1) > thr) == pytest.approx(dt.mean_shift_power(delta, f, n), abs=0.01)


def test_energy_detects_zero_mean_warp_that_mean_shift_cannot():
    rng = np.random.default_rng(1)
    n, d, trials = 1000, 0.5, 4000
    x0 = rng.standard_normal((trials, n))
    x1 = x0 + rng.standard_normal(n) * d
    thr = np.quantile((x0**2).mean(1), 0.95)
    energy_power = np.mean((x1**2).mean(1) > thr)
    mean_power = np.mean(np.abs(x1.sum(1) / np.sqrt(n)) > dt.z_alpha(0.05))
    assert energy_power > 0.95 and mean_power < 0.3
