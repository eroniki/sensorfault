"""Contracts between the three parts of the gate test bed. Keep this file small and stable.

Parts: ``fit.py`` + ``blocks.py`` (backends: fit and information blocks), ``gate.py`` (leave-view-out
test), ``fusers.py`` (registered fusers + outer loop). Everything in ``gate.py`` and ``fusers.py``
must run on CPU/numpy against a fake ``SurfelBackend``; only ``fit.py``'s gsplat path needs a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sensorfault.types import DepthFrame

THETA_DIM = 10  # [d_mu(3, world), d_phi(3, body), d_log_s(3), d_logit_alpha(1)]
GEO_DIM = 9  # geometric coordinates (opacity frozen after warm-up)


@dataclass
class SurfelState:
    """K 2DGS surfels after the fit (opacity frozen after warm-up)."""

    mu: np.ndarray  # (K, 3) world
    R: np.ndarray  # (K, 3, 3) columns [t1 t2 n]
    log_s: np.ndarray  # (K, 3) (third = fixed thin axis for 2DGS)
    logit_alpha: np.ndarray  # (K,)
    meta: dict = field(default_factory=dict)  # e.g. warm-up info; backend-private payload allowed


@dataclass
class ViewInfo:
    """Information one view adds to every surfel it touches (dense over K for simplicity)."""

    Lambda: np.ndarray  # (K, 10, 10), alpha-weighted J^T R^-1 J, zeros for untouched surfels
    eta: np.ndarray  # (K, 10) J^T R^-1 r (negative gradient of this view's NLL at the MAP)
    n_pixels: int = 0


@dataclass
class CavityRender:
    """View j rendered from the surface fitted without view j."""

    depth: np.ndarray  # (H, W) rendered depth, NaN where unsupported / no coverage
    var: np.ndarray  # (H, W) rendered depth variance from the cavity posterior (before temperature)
    valid: np.ndarray  # (H, W) bool: supported by other views AND the observed depth is valid


class SurfelBackend(Protocol):
    """What the gate needs from a renderer/fitter. Implemented by the gsplat backend (GPU) and by a
    numpy reference backend (tests, tiny scenes)."""

    def fit(
        self,
        frames: Sequence[DepthFrame],
        weights: Mapping[str, float],
        init: SurfelState | None = None,
    ) -> SurfelState:
        """Depth-supervised 2DGS fit; a view's weight scales its depth loss (0 = absent).
        Opacity trains in a warm-up then freezes. ``frames[i].sensor_id`` is the view key."""

    def view_information(self, state: SurfelState, frame: DepthFrame, sigma: float) -> ViewInfo:
        """10x10 information blocks of one view at the MAP (alpha-weighted rasteriser J^T R^-1 J)."""

    def render_cavity(
        self,
        state: SurfelState,
        frame: DepthFrame,
        mean_shift: np.ndarray,
        cov: np.ndarray,
        keep: np.ndarray,
    ) -> CavityRender:
        """Render ``frame``'s camera from surfels with posterior ``mean_shift`` (K, 9) and
        marginal covariance ``cov`` (K, 9, 9), dropping surfels where ``keep`` (K,) is False."""


class DepthSource(Protocol):
    """Depth-cache interface (built by the parallel depth-cache thread; adapt here, not there)."""

    def frames(self, scene: str, model: str) -> Sequence[DepthFrame]:
        """All views of ``scene`` with depth from ``model`` in {depth_pro, metric3d_v2,
        unidepth_v2, depth_anything_v2}; ``conf`` filled where the model provides one."""
