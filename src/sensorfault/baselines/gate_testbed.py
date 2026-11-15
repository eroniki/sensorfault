"""Stub registrations for the design-v3 test bed: depth-supervised Gaussian splatting with a
depth-only view gate whose uncertainty renders.

None of these is implemented yet. Each registers a name so that configs can reference it and the
runner can skip it with a clear reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sensorfault.baselines.base import BaseFuser, register
from sensorfault.types import DepthFrame, FusionResult

_STUBS: dict[str, str] = {
    "no_gate": "Depth-supervised 2DGS surfel splatting (gsplat) with every view's depth trusted.",
    "oracle_gate": "Upper bound: drops exactly the faulty views (manifest or laser reference).",
    "view_gate": (
        "The contribution: depth-only view gate. Each surfel carries a 10x10 information block over "
        "(d_mu, d_phi, d_log_s, d_logit_alpha; opacity frozen after warm-up); views add alpha-weighted "
        "rasteriser J^T R^-1 J; leave-view-out is a subtraction plus one Newton step; noise has one free "
        "scene temperature; shared distance bias; the other views' surface (normal offset + tilt) is rendered into view j with its "
        "variance and the whitened depth residuals are tested. Params: uncertainty in {rendered_9x9, "
        "constant_sigma, fisherrf_style_diagonal_fisher, bayes_rays_style_perturbation_field}, "
        "leave_view_out in {subtraction, none_joint}, spawn in {none, keep_view_j_only_surfels}, "
        "information_deflation, overconfidence (jackknife_inflation), noise, shared_bias, opacity_warmup, statistic, "
        "outer_loop (round_cap, policy, exclusion, readmission; see src/sensorfault/eval/gate_loop.py). Theory: src/sensorfault/eval/gaussian_uncertainty.py and "
        "src/sensorfault/eval/detection_theory.py."
    ),
    "aligned_rms_view_test": (
        "Cheap baseline: RMS of the per-view affine-aligned depth residual against the other views' "
        "render, no whitening, thresholded at the same false-alarm rate."
    ),
    "depth_confidence_score": (
        "Cheap baseline: view score from the depth model's own confidence (Metric3D v2, UniDepthV2)."
    ),
    "spotless_splats": "Baseline: SpotLessSplats run as released, with depth supervision added.",
    "robustnerf_loss": "Baseline: RobustNeRF trimmed robust loss applied to splatting.",
    "robust_depth_loss": "Baseline: robust (Huber / Geman-McClure) depth loss in the splatting fit.",
}


def _make(name: str, doc: str) -> type[BaseFuser]:
    class _Stub(BaseFuser):
        __doc__ = doc

        @classmethod
        def available(cls) -> tuple[bool, str]:
            return False, f"not implemented yet (gate test bed stub): {doc}"

        def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
            raise NotImplementedError(f"{name}: {doc}")

    _Stub.__name__ = f"Stub_{name}"
    return register(name)(_Stub)


for _n, _d in _STUBS.items():
    _make(_n, _d)

GATE_TESTBED_NAMES = frozenset(_STUBS)
