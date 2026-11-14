"""Detection theory for view-level fault detection and exclusion (integrity monitoring).

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

from scipy.optimize import brentq
from scipy.stats import norm


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
