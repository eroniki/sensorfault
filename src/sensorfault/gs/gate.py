"""Leave-view-out test of the depth-only view gate (design v3.x), on the abstract ``SurfelBackend``.

``ViewGateTest`` is the ``test`` callback of ``eval.gate_loop.run_gate_loop``: called with a fitted
``SurfelState`` and view ids it returns ``(p_values, log_p_values)`` so that p-values that underflow
to 0.0 are still ranked. It only uses numpy/scipy and the backend protocol (never torch).

Pipeline per call
1. every fused view i: ``ViewInfo`` from ``backend.view_information`` (scaled by its fit weight and
   the information deflation); fused information ``Lambda = prior + sum_i Lambda_i``.
2. per tested view j: cavity = subtract ``Lambda_j`` (or the block's) and one Newton step on the nine
   geometric coordinates (``leave_view_out_newton``); keep only supported surfels (normal-offset
   information from the other views, and the opacity-share rule); ``backend.render_cavity`` into j.
3. shared distance bias beta (``d_hat = D + beta D^2``), fitted by weighted least squares over all
   fused views with view j (its block) removed by subtraction of normal equations.
4. one scene temperature gamma^2 (median ratio of the other views' energies to the chi2 median);
   no per-view noise parameter.
5. statistic on whitened residuals ``z = (d_hat - D_eff) / sqrt(sigma^2 + var)``: energy
   ``T = r^T M r`` with the per-view scale/offset nuisance projected out (chi2, ``dof = n - rank``),
   and a region scan (matched mean over grid blocks, exact variance with the projection), combined
   by a factor-2 Bonferroni.

Approximations (see also ``gaussian_uncertainty`` module docstring): diagonal pixel noise (no spatial
correlation; ``n_eff_fraction`` rescales the chi2 dof, and the temperature absorbs only the overall
scale); the cavity mean is linearised at the MAP (mean perturbation 0); beta used for the *other*
views' energies in the temperature fit is the all-view beta, not the per-j leave-out beta;
``information_deflation="n_eff_over_n"`` estimates the ratio from the lag-1 autocorrelation of a
first-pass whitened residual; ``overconfidence="jackknife_inflation"`` adds a per-surfel
delete-one-view jackknife covariance of the cavity means.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2, norm

from sensorfault.eval.detection_theory import temperature_median_ratio
from sensorfault.eval.gaussian_uncertainty import (
    DIM,
    GEO,
    distance_bias_fit,
    distance_bias_normal_equations,
    leave_view_out_newton,
    opacity_supported,
    supported,
)
from sensorfault.gs.interfaces import CavityRender, SurfelBackend, SurfelState, ViewInfo
from sensorfault.types import DepthFrame

_ALLOWED = {
    "uncertainty": {"rendered_10x10_opacity_frozen", "rendered_9x9", "constant_sigma"},
    "leave_view_out": {"subtraction", "none_joint"},
    "spawn": {"none", "keep_view_j_only_surfels"},
    "overconfidence": {None, "none", "jackknife_inflation"},
    "noise": {"scene_temperature", "dev_calibrated_level_only"},
    "shared_bias": {None, "none", "distance"},
    "statistic": {"combined_region_scan_energy", "energy", "region_scan"},
}
_NOT_IMPLEMENTED = {
    "fisherrf_style_diagonal_fisher": (
        "uncertainty='fisherrf_style_diagonal_fisher': the diagonal-Fisher surfel uncertainty is a "
        "separate backend/uncertainty implementation and is not part of ViewGateTest."
    ),
    "bayes_rays_style_perturbation_field": (
        "uncertainty='bayes_rays_style_perturbation_field': the perturbation-field uncertainty is a "
        "separate backend/uncertainty implementation and is not part of ViewGateTest."
    ),
}
# Parameters of the config that belong to the fit / outer loop, not to the test.
_IGNORED = {
    "opacity_warmup",
    "outer_loop",
    "outer_loop_policy",
    "outer_loop_exclusion",
    "outer_loop_readmission",
    "downweight",
    "poses",
    "leave_block_out_B",
}


class ViewGateTest:
    """Callable ``(state, views) -> (pvals, logps)``; see the module docstring."""

    def __init__(
        self,
        backend: SurfelBackend,
        frames: Sequence[DepthFrame],
        sigma: float = 0.01,
        params: Mapping | None = None,
        *,
        prior_precision: float = 1e3,
        rho: float = 0.5,
        lam_min: float | None = None,
        scan_grids: Sequence[int] = (2, 4),
        min_region_pixels: int = 8,
        fit_affine: bool = True,
        gamma: float = 1.0,
        heterogeneity_omega: float | str = "estimate",
        n_eff_fraction: float = 1.0,
        blocks: Mapping[str, Sequence[str]] | None = None,
        active: Sequence[str] | None = None,
    ) -> None:
        """``sigma``: depth noise (m) passed to ``view_information`` and added to the rendered
        variance. ``params``: the config's ``view_gate`` params (defaults give the main gate).
        ``blocks`` maps a view to the block left out with it (leave-N-out); ``active`` lists the
        views the state was fitted on (default: ``state.meta['weights']`` if a mapping, else all)."""
        p = dict(params or {})
        self.uncertainty = p.pop("uncertainty", "rendered_10x10_opacity_frozen")
        if self.uncertainty in _NOT_IMPLEMENTED:
            raise NotImplementedError(_NOT_IMPLEMENTED[self.uncertainty])
        self.leave_view_out = p.pop("leave_view_out", "subtraction")
        self.spawn = p.pop("spawn", "none")
        self.deflation = p.pop("information_deflation", 1.0)
        self.overconfidence = p.pop("overconfidence", None)
        self.noise = p.pop("noise", "scene_temperature")
        # scale of the region-scan z-scores: fitted from the other views ("empirical", default; one more
        # scene-level scale, no per-view parameters) or tied to the energy temperature ("energy")
        self.scan_noise = p.pop("scan_noise", "empirical")
        if self.scan_noise not in ("empirical", "energy"):
            raise ValueError(f"scan_noise={self.scan_noise!r}")
        self.shared_bias = p.pop("shared_bias", "distance")
        self.statistic = p.pop("statistic", "combined_region_scan_energy")
        for key in ("uncertainty", "leave_view_out", "spawn", "overconfidence", "noise"):
            self._check(key, getattr(self, key))
        self._check("shared_bias", self.shared_bias)
        self._check("statistic", self.statistic)
        if self.deflation is None:
            self.deflation = 1.0
        if not (self.deflation == "n_eff_over_n" or isinstance(self.deflation, (int, float))):
            raise ValueError(f"information_deflation={self.deflation!r}")
        for k in list(p):
            if k in _IGNORED:
                p.pop(k)
        if p:
            raise ValueError(f"unknown view_gate params: {sorted(p)}")

        self.backend = backend
        self.frames = {f.sensor_id: f for f in frames}
        self.ids = [f.sensor_id for f in frames]
        self.sigma = float(sigma)
        self.prior = prior_precision * np.eye(DIM)
        self.rho = rho
        self.lam_min = 25.0 / self.sigma**2 if lam_min is None else lam_min
        self.scan_grids = tuple(scan_grids)
        self.min_region_pixels = min_region_pixels
        self.fit_affine = fit_affine
        self.gamma = gamma
        if not (heterogeneity_omega == "estimate" or isinstance(heterogeneity_omega, (int, float))):
            raise ValueError(f"heterogeneity_omega={heterogeneity_omega!r}")
        self.omega = heterogeneity_omega  # float, or "estimate": log-scale spread of T/dof over the other views
        self.n_eff_fraction = n_eff_fraction
        self.blocks = {k: list(v) for k, v in (blocks or {}).items()}
        self.active = None if active is None else list(active)
        self.last: dict = {}

    def _check(self, key: str, value) -> None:
        if value not in _ALLOWED[key]:
            raise ValueError(f"{key}={value!r}; allowed {sorted(map(str, _ALLOWED[key]))}")

    @classmethod
    def from_params(cls, backend, frames, params: Mapping | None = None, **kw) -> ViewGateTest:
        return cls(backend, frames, params=params, **kw)

    # ------------------------------------------------------------------ helpers
    def _fused(self, state: SurfelState) -> dict[str, float]:
        w = state.meta.get("weights") if isinstance(state.meta, dict) else None
        if isinstance(w, Mapping):
            return {v: float(w.get(v, 0.0)) for v in self.ids if w.get(v, 0.0) > 0}
        ids = self.ids if self.active is None else self.active
        return dict.fromkeys(ids, 1.0)

    def _informations(self, state, fused: Mapping[str, float], c: float) -> dict[str, ViewInfo]:
        out = {}
        for v, wt in fused.items():
            vi = self.backend.view_information(state, self.frames[v], self.sigma)
            s = wt * c
            out[v] = ViewInfo(vi.Lambda * s, vi.eta * s, vi.n_pixels)
        return out

    def _jackknife(self, L, infos) -> np.ndarray | None:
        n = len(infos)
        if n < 2:
            return None
        means = []
        for vi in infos.values():
            cov = np.linalg.inv((L - vi.Lambda)[:, GEO, GEO])
            means.append(-np.einsum("kij,kj->ki", cov, vi.eta[:, GEO]))
        m = np.stack(means)
        dev = m - m.mean(0, keepdims=True)
        return (n - 1) / n * np.einsum("nki,nkj->kij", dev, dev)

    def _cavity(self, state, j, infos, L, L_data, jk) -> CavityRender:
        frame = self.frames[j]
        K = len(state.mu)
        if self.leave_view_out == "none_joint":
            cov = np.linalg.inv(L[:, GEO, GEO])
            mean = np.zeros((K, 9))
            keep = np.ones(K, bool)
        else:
            block = [v for v in self.blocks.get(j, [j]) if v in infos]
            Lj = sum((infos[v].Lambda for v in block), np.zeros_like(L))
            ej = sum((infos[v].eta for v in block), np.zeros((K, DIM)))
            mean, cov = leave_view_out_newton(np.zeros((K, 9)), L, Lj, -ej)
            if self.spawn == "keep_view_j_only_surfels":
                keep = np.ones(K, bool)
            else:
                keep = supported(L - Lj, state.R, self.prior, self.lam_min) & opacity_supported(
                    L_data, Lj, self.rho
                )
        if jk is not None:
            cov = cov + jk
        cav = self.backend.render_cavity(state, frame, mean, cov, keep)
        var = np.zeros_like(cav.var) if self.uncertainty == "constant_sigma" else cav.var
        valid = cav.valid & np.isfinite(cav.depth) & np.isfinite(frame.depth) & (frame.depth > 0)
        valid &= np.isfinite(var)
        return CavityRender(cav.depth, var, valid)

    def _pixels(self, j: str, cav: CavityRender):
        m = cav.valid
        return cav.depth[m], cav.var[m] + self.sigma**2, self.frames[j].depth[m], m

    def _stats(self, j: str, cav: CavityRender, beta: float):
        """Energy T (gamma=1), dof and standardised region-scan values at gamma = 1."""
        D, s2, d, m = self._pixels(j, cav)
        n = D.size
        if n < 3:
            return 0.0, 0, np.zeros(0)
        r = d - (D + beta * D * D)
        w = 1.0 / s2
        A = np.stack([D, np.ones(n)], 1) if self.fit_affine else np.zeros((n, 0))
        if A.shape[1]:
            AtWA = (A * w[:, None]).T @ A
            Ginv = np.linalg.pinv(AtWA, rcond=1e-10)
            rank = int(np.linalg.matrix_rank(AtWA, tol=1e-10 * max(np.trace(AtWA), 1e-30)))
            Atwr = A.T @ (w * r)
            T = float(np.sum(w * r * r) - Atwr @ Ginv @ Atwr)
        else:
            Ginv, rank, Atwr = np.zeros((0, 0)), 0, np.zeros(0)
            T = float(np.sum(w * r * r))
        dof = n - rank
        scan = []
        H, W = cav.valid.shape
        rows, cols = np.nonzero(m)
        for g in self.scan_grids:
            rid = (rows * g // H) * g + (cols * g // W)
            for q in np.unique(rid):
                sel = rid == q
                if sel.sum() < self.min_region_pixels:
                    continue
                sw = float(w[sel].sum())
                num = float((w[sel] * r[sel]).sum())
                if A.shape[1]:
                    b = (A[sel] * w[sel, None]).sum(0)
                    num -= float(b @ Ginv @ Atwr)
                    sw -= float(b @ Ginv @ b)
                if sw > 1e-9 * float(w[sel].sum()):
                    scan.append(num / math.sqrt(sw))
        return max(T, 0.0), dof, np.asarray(scan)

    def _bias_equations(self, cache):
        h, g = {}, {}
        for v, cav in cache.items():
            D, s2, d, _ = self._pixels(v, cav)
            h[v], g[v] = distance_bias_normal_equations(D, 1.0, d - D, s2)
        return h, g

    def _autocorr_fraction(self, cache) -> float:
        """Scene n_eff / n from lag-1 autocorrelation of whitened residuals (AR(1) product form)."""
        fr = []
        for v, cav in cache.items():
            z = np.where(
                cav.valid, (self.frames[v].depth - cav.depth) / np.sqrt(cav.var + self.sigma**2), 0
            )
            f = 1.0
            for ax in (0, 1):
                a = np.moveaxis(z, ax, 0)
                ok = np.moveaxis(cav.valid, ax, 0)
                pair = ok[1:] & ok[:-1]
                if pair.sum() < 10:
                    continue
                den = float(np.mean(z[cav.valid] ** 2)) or 1.0
                rho = max(float(np.mean(a[1:][pair] * a[:-1][pair])) / den, 0.0)
                f *= (1 - rho) / (1 + rho)
            fr.append(f)
        return float(np.clip(np.median(fr), 0.05, 1.0)) if fr else 1.0

    # --------------------------------------------------------------------- call
    def __call__(
        self, state: SurfelState, views: Sequence[str] | None = None
    ) -> tuple[dict[str, float], dict[str, float]]:
        tested = list(self.ids if views is None else views)
        fused = self._fused(state)
        if self.deflation == "n_eff_over_n":
            c = 1.0
            first = self._pass(state, fused, tested, c, want_cache_only=True)
            c = self._autocorr_fraction(first)
        else:
            c = float(self.deflation)
        return self._pass(state, fused, tested, c)

    def _pass(self, state, fused, tested, c, want_cache_only: bool = False):
        infos = self._informations(state, fused, c)
        K = len(state.mu)
        L_data = sum((vi.Lambda for vi in infos.values()), np.zeros((K, DIM, DIM)))
        L = L_data + self.prior
        jk = self._jackknife(L, infos) if self.overconfidence == "jackknife_inflation" else None
        need = list(dict.fromkeys([*fused, *tested]))
        cache = {v: self._cavity(state, v, infos, L, L_data, jk) for v in need}
        if want_cache_only:
            return cache

        fused_ids = list(fused)
        use_bias = self.shared_bias == "distance"
        h, g = self._bias_equations({v: cache[v] for v in fused_ids}) if use_bias else ({}, {})
        h_arr = np.array([h[v] for v in fused_ids]) if use_bias else np.zeros(0)
        g_arr = np.array([g[v] for v in fused_ids]) if use_bias else np.zeros(0)
        eps = 1e-12

        def beta_without(excl: Sequence[str]) -> float:
            if not use_bias or not len(h_arr):
                return 0.0
            hm, gm = h_arr.copy(), g_arr.copy()
            for v in excl:
                if v in fused:
                    hm[fused_ids.index(v)] = 0.0
                    gm[fused_ids.index(v)] = 0.0
            if hm.sum() <= 0:
                return 0.0
            return distance_bias_fit(hm, gm, None, prior_info=eps)

        beta_all = beta_without([])
        base = {v: self._stats(v, cache[v], beta_all) for v in fused_ids}
        frac = self.n_eff_fraction
        pvals, logps = {}, {}
        diag = {}
        for j in tested:
            excl = self.blocks.get(j, [j])
            beta_j = beta_without(excl)
            T, dof, scan = self._stats(j, cache[j], beta_j)
            if self.noise == "scene_temperature":
                oth = [v for v in fused_ids if v not in excl and base[v][1] > 0]
                if len(oth) >= 2:
                    Ts = np.array([base[v][0] * frac for v in oth])
                    ms = np.array([max(base[v][1] * frac, 1.0) for v in oth])
                    g2 = max(temperature_median_ratio(Ts, ms), 1e-12)
                else:
                    g2 = 1.0
            else:
                g2 = self.gamma**2
            g2_scan = g2
            if self.scan_noise == "empirical" and self.noise == "scene_temperature":
                pool = [np.abs(base[v][2]) for v in fused_ids if v not in excl and len(base[v][2])]
                if pool:  # robust scale of the other views' region z-scores (MAD), floored at the nominal 1
                    z = np.concatenate(pool)
                    g2_scan = max((float(np.median(z)) / 0.6745) ** 2, 1e-12)
            om = self.omega
            if om == "estimate":
                lr = [
                    0.5 * math.log(max(base[v][0], 1e-300) / (base[v][1] * g2))
                    for v in fused_ids
                    if v not in excl and base[v][1] > 0 and base[v][0] > 0
                ]
                # robust (MAD) spread of the per-view level; needs >= 3 other views
                om = (
                    1.4826 * float(np.median(np.abs(np.asarray(lr) - np.median(lr))))
                    if len(lr) >= 3
                    else 0.0
                )
            lp_e, lp_s = self._log_p(T, dof, scan, g2, g2_scan, om)
            parts = {
                "combined_region_scan_energy": [lp_e, lp_s],
                "energy": [lp_e],
                "region_scan": [lp_s],
            }[self.statistic]
            lp = min(0.0, min(parts) + (math.log(len(parts)) if len(parts) > 1 else 0.0))
            logps[j], pvals[j] = lp, float(math.exp(lp))
            diag[j] = {
                "beta": beta_j,
                "gamma2": g2,
                "gamma2_scan": g2_scan,
                "omega": om,
                "T": T,
                "dof": dof,
                "logp_energy": lp_e,
                "logp_scan": lp_s,
                "n_regions": len(scan),
                "deflation": c,
            }
        self.last = diag
        return pvals, logps

    def _log_p(
        self,
        T: float,
        dof: int,
        scan: np.ndarray,
        g2: float,
        g2_scan: float | None = None,
        omega: float | None = None,
    ) -> tuple[float, float]:
        g2s = g2 if g2_scan is None else g2_scan
        frac = self.n_eff_fraction
        m = dof * frac
        if dof <= 0 or m < 1e-9:
            lp_e = 0.0
        else:
            te = T * frac / g2
            om = self.omega if omega is None else omega
            if isinstance(om, (int, float)) and om > 0:
                x, wq = np.polynomial.hermite_e.hermegauss(40)
                wq = wq / wq.sum()
                r2 = np.exp(2.0 * om * x)
                lp_e = float(logsumexp(np.log(wq) + _chi2_logsf(te / r2, m)))
            else:
                lp_e = float(_chi2_logsf(te, m))
        if len(scan):
            lp_s = float(
                math.log(2.0)
                + norm.logsf(np.max(np.abs(scan)) / math.sqrt(g2s))
                + math.log(len(scan))
            )
        else:
            lp_s = 0.0
        return min(lp_e, 0.0), min(lp_s, 0.0)


def _chi2_logsf(x, m: float):
    """log P(chi2_m > x), with an asymptotic expansion where scipy's logsf underflows to -inf."""
    x = np.asarray(x, dtype=float)
    out = np.asarray(chi2.logsf(x, m), dtype=float)
    bad = ~np.isfinite(out) & (x > 0)
    if np.any(bad):
        a, xh = m / 2.0, x[bad] / 2.0
        term, series = np.ones_like(xh), np.ones_like(xh)
        for i in range(1, 60):
            term = term * (a - i) / xh
            series = series + term
            if np.all(np.abs(term) < 1e-12):
                break
        out[bad] = -xh + (a - 1.0) * np.log(xh) - gammaln(a) + np.log(np.maximum(series, 1e-300))
    return out if out.ndim else float(out)
