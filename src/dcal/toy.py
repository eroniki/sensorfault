"""Toy simulator for the fault statistic. Grid g x g, Matern-3/2 truth kernels (estimator uses exponential)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stat import _dist  # noqa: F401


def matern32(grid_shape, length):
    h, w = grid_shape
    yy, xx = np.mgrid[0:h, 0:w]
    p = np.stack([yy.ravel(), xx.ravel()], 1).astype(float)
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)) * np.sqrt(3.0) / length
    return (1 + d) * np.exp(-d)


@dataclass
class ToyConfig:
    g: int = 24
    M: int = 10
    sigma_u: float = 1.0       # local-shape posterior sd (sample units)
    ell_u: float = 3.0
    sigma_r: float = 0.7       # render sd (render units)
    ell_r: float = 6.0
    wobble: float = 0.05       # per-sample scale/shift wobble from percentile normalisation
    a_range: tuple = (0.8, 1.25)
    b_range: tuple = (-1.0, 1.0)


class World:
    def __init__(self, cfg: ToyConfig, seed: int = 0):
        self.cfg = cfg
        g = cfg.g
        self.shape = (g, g)
        self.n = g * g
        self.Ku = cfg.sigma_u ** 2 * matern32(self.shape, cfg.ell_u)
        self.Kr = cfg.sigma_r ** 2 * matern32(self.shape, cfg.ell_r)
        j = 1e-8 * np.eye(self.n)
        self.Lu = np.linalg.cholesky(self.Ku + j)
        self.Lr = np.linalg.cholesky(self.Kr + j)
        yy, xx = np.mgrid[0:g, 0:g] / g
        # smooth relief plus a step edge, large vs sigma so m and 1 are not collinear
        self.mu = (4 * np.sin(2.5 * xx) * np.cos(2 * yy) + 3 * (xx > 0.55) + 2 * yy).ravel() * 2.0

    def fault_pattern(self, kind: str, area: float = 0.12) -> np.ndarray:
        g = self.cfg.g
        yy, xx = np.mgrid[0:g, 0:g] / g
        if kind == "square":
            side = np.sqrt(area)
            f = ((xx > 0.3) & (xx < 0.3 + side) & (yy > 0.4) & (yy < 0.4 + side)).astype(float)
        elif kind == "warp":
            f = np.sin(2 * np.pi * 1.5 * xx) * np.cos(2 * np.pi * yy)
            f = f / np.abs(f).max()
        elif kind == "edge":   # region following the truth level set
            f = (self.mu >= np.quantile(self.mu, 1 - area)).astype(float).reshape(g, g)
        elif kind == "shift":   # constant offset: the only fault the (s, t) fit absorbs exactly
            f = np.ones((g, g))
        elif kind == "affine":  # global scale/shift fault: lies in span{m, 1}
            f = (self.mu - self.mu.mean()).reshape(g, g)
            f = f / np.abs(f).max()
        else:
            raise ValueError(kind)
        return f.ravel()

    def sd_tot(self, a: float = 1.0) -> float:
        return float(np.sqrt(np.median(np.diag(a * a * (1 + 1 / self.cfg.M) * self.Ku + self.Kr))))

    def simulate(self, rng, fault: str | None = None, kappa: float = 0.0, area: float = 0.12):
        """One view: returns samples (M, n), render (n,), Sigma_r, oracle Sigma_u (sample units), truth (a, b)."""
        c = self.cfg
        a = rng.uniform(*c.a_range)
        b = rng.uniform(*c.b_range)
        n, M = self.n, c.M
        d_true = self.mu + self.Lu @ rng.standard_normal(n)             # truth is one more draw of the posterior
        render = a * d_true + b + self.Lr @ rng.standard_normal(n)
        S = self.mu[None] + (self.Lu @ rng.standard_normal((n, M))).T
        alpha = 1 + c.wobble * rng.standard_normal(M)
        beta = c.wobble * rng.standard_normal(M)
        S = alpha[:, None] * S + beta[:, None]
        if fault is not None and kappa != 0:
            amp = kappa * self.sd_tot(a)
            S = S + (amp / a) * self.fault_pattern(fault, area)[None]
        return S, render, self.Kr, self.Ku, (a, b)
