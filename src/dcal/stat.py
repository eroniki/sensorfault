"""Fault statistic on unguided samples with a per-view scale and shift fitted to the leave-j-out render.

Model of the null for view j (affine-invariant samples vs a render in scene units)::

    D_render = s * m + t * 1 + r,    r ~ N(0, s^2 (1 + 1/M) Sigma_U + Sigma_r)

m is the mean of M samples aligned to each other, Sigma_U their local-shape covariance, Sigma_r the render
variance (from other views only). (s, t) are the only quantities fitted with view j's data. The (1 + 1/M)
factor is the predictive variance of the sample mean for a calibrated posterior. Dense (n, n) matrices are
used here for the CPU toy; production would use stationary kernel + diagonal + low rank with Woodbury/CG.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def align_samples(S: np.ndarray) -> np.ndarray:
    """Fit each sample's scale/shift to the raw mean sample (fixed reference, so no scale drift)."""
    M, n = S.shape
    ref = S.mean(0)
    out = np.empty_like(S)
    one = np.ones(n)
    for k in range(M):
        G = np.stack([S[k], one], 1)
        s, t = np.linalg.lstsq(G, ref, rcond=None)[0]
        out[k] = s * S[k] + t
    return out


def _dist(grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    yy, xx = np.mgrid[0:h, 0:w]
    p = np.stack([yy.ravel(), xx.ravel()], 1).astype(float)
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))


KERNELS = {
    "exp": lambda d, ell: np.exp(-d / ell),
    "matern32": lambda d, ell: (1 + np.sqrt(3.0) * d / ell) * np.exp(-np.sqrt(3.0) * d / ell),
    "gauss": lambda d, ell: np.exp(-0.5 * (d / ell) ** 2),
}


def exp_kernel(grid_shape: tuple[int, int], length: float) -> np.ndarray:
    return KERNELS["exp"](_dist(grid_shape), length)


def fit_stationary_cov(A: np.ndarray, grid_shape: tuple[int, int], max_lag: int = 5,
                       families: tuple[str, ...] = ("exp", "matern32", "gauss"), return_info: bool = False):
    """Stationary kernel + nugget fitted to the pooled empirical autocovariance of the aligned sample deviations.

    Pooling over M samples and all pixel pairs gives far more data than the rank-(M-1) sample covariance.
    The kernel family (roughness) is chosen by the lag-fit residual: too rough a family inflates variance in
    the high-frequency directions where the true residual has little energy and makes the test conservative.
    Short lags only: the per-sample affine alignment biases long-lag autocovariance downwards.
    """
    M, n = A.shape
    h, w = grid_shape
    X = (A - A.mean(0)) * np.sqrt(M / (M - 1.0))
    X = X.reshape(M, h, w)
    lags = np.arange(0, min(max_lag, min(h, w) // 3) + 1)
    c = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L == 0:
            c[i] = (X * X).mean()
        else:
            c[i] = 0.5 * ((X[:, :, L:] * X[:, :, :-L]).mean() + (X[:, L:, :] * X[:, :-L, :]).mean())
    best = None
    for fam in families:
        for ell in np.exp(np.linspace(np.log(0.7), np.log(30), 50)):
            B = np.stack([KERNELS[fam](lags.astype(float), ell), (lags == 0).astype(float)], 1)
            coef, *_ = np.linalg.lstsq(B, c, rcond=None)
            coef = np.maximum(coef, [1e-12, 0.0])
            r = ((B @ coef - c) ** 2).sum()
            if best is None or r < best[0]:
                best = (r, fam, ell, coef)
    _, fam, ell, (sig2, nug) = best
    C = sig2 * KERNELS[fam](_dist(grid_shape), ell) + nug * np.eye(n)
    return (C, {"family": fam, "ell": ell, "sigma2": sig2, "nugget": nug}) if return_info else C


def shrunk_sample_cov(A: np.ndarray, lam: float = 0.3) -> np.ndarray:
    """Baseline estimator: rank-(M-1) sample covariance shrunk to its diagonal (expected to be inadequate)."""
    M = A.shape[0]
    X = A - A.mean(0)
    S = X.T @ X / (M - 1.0)
    return (1 - lam) * S + lam * np.diag(np.diag(S)) + 1e-9 * np.eye(S.shape[0]) * np.trace(S) / S.shape[0]


def _whiten(Sigma: np.ndarray):
    L = cholesky(Sigma + 1e-10 * np.trace(Sigma) / len(Sigma) * np.eye(len(Sigma)), lower=True)
    return L


def fit_scale_shift(m, D, Sigma_u_pred, Sigma_r, *, robust=True, huber_c=1.345, outer=3, inner=8):
    """GLS (optionally Huber-IRLS on the whitened residual) for (s, t); Sigma depends on s, so iterate."""
    n = len(m)
    G = np.stack([m, np.ones(n)], 1)
    s, t = np.linalg.lstsq(G, D, rcond=None)[0]
    for _ in range(outer):
        L = _whiten(s * s * Sigma_u_pred + Sigma_r)
        Gw = solve_triangular(L, G, lower=True)
        yw = solve_triangular(L, D, lower=True)
        w = np.ones(n)
        for _ in range(inner if robust else 1):
            sw = np.sqrt(w)[:, None]
            theta = np.linalg.lstsq(Gw * sw, yw * sw[:, 0], rcond=None)[0]
            e = yw - Gw @ theta
            if not robust:
                break
            a = np.abs(e)
            w = np.where(a <= huber_c, 1.0, huber_c / np.maximum(a, 1e-12))
        s, t = theta
    return float(s), float(t)


def fault_statistic(samples, render, Sigma_r, *, grid_shape=None, Sigma_u=None, cov="kernel",
                   robust=True, huber_c=1.345):
    """Return dict(T, s, t, dof, e). ``Sigma_u`` (sample units) overrides the estimator, e.g. the oracle in tests."""
    S = np.asarray(samples, float)
    M, n = S.shape
    A = align_samples(S)
    m = A.mean(0)
    if Sigma_u is None:
        Sigma_u = fit_stationary_cov(A, grid_shape) if cov == "kernel" else shrunk_sample_cov(A)
    Sp = (1.0 + 1.0 / M) * Sigma_u
    s, t = fit_scale_shift(m, render, Sp, Sigma_r, robust=robust, huber_c=huber_c)
    r = s * m + t - render
    L = _whiten(s * s * Sp + Sigma_r)
    e = solve_triangular(L, r, lower=True)
    return {"T": float(e @ e), "s": s, "t": t, "dof": n - 2, "e": e}


def empirical_threshold(null_T: np.ndarray, alpha: float = 0.05) -> float:
    """Conformal-style upper quantile with the (k+1) correction."""
    k = int(np.ceil((len(null_T) + 1) * (1 - alpha)))
    return float(np.sort(null_T)[min(k, len(null_T)) - 1])
