"""STUB -- planned interface of the proposed method (NO implementation; PLAN.md section 4).

Planned pipeline (``SplatConsensus.fuse``):
 1. Joint fit of a fixed-N MCMC Gaussian-splat model to all sensors' depth (depth-only,
    accumulated-alpha mask loss, variance-weighted residuals).
 2. Visibility-aware residuals r = (D_s - D_hat) / sigma_s(z, theta) where rendered alpha > tau
    and the first-return gate matches; grazing handled via sim.depth_residual_variance.
 3. Per-sensor statistics: leave-one-out chi^2 (warm-started ~200-iter refit), delta-entropy
    (mixture-of-per-sensor-posteriors), Moran's I (diagnostic only).
 4. Exclusion through elim.eliminate (corrected criterion) / threshold calibrated on clean-run
    permutations (5% false elimination); refit; iterate; keep >= 2-3 sensors.
 5. Persistence logic (CUSUM) separates sensor faults from dynamic objects.
Expected cfg keys: n_gaussians, iters, loo_iters, alpha_tau, theta_gate_deg, eps, F_max,
    null_threshold (from calibrate_threshold), variant in {"corrected", "literal_alg1"}.
Returns FusionResult with excluded_sensors, sensor_scores (LOO chi2), meta (chain, Lambda).

Deliberately NOT registered in the fuser registry so the runner logs "unavailable (stub)".
"""

from __future__ import annotations

from typing import Mapping, Sequence

from sensorfault.types import DepthFrame, FusionResult


class SplatConsensus:
    name = "splat_consensus"

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        raise NotImplementedError("splat_consensus is a documented stub")

    @classmethod
    def available(cls) -> tuple:
        return False, "stub: not implemented"
