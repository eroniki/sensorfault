"""Stub registrations for the design-v3 test bed: depth-supervised Gaussian splatting with a
depth-only view gate whose uncertainty renders.

no_gate, oracle_gate and view_gate are implemented in ``sensorfault.gs.fusers`` (imported at the
bottom); the others are stubs. Each stub registers a name so that configs can reference it and the
runner can skip it with a clear reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sensorfault.baselines.base import BaseFuser, register
from sensorfault.types import DepthFrame, FusionResult

# Implemented in sensorfault.gs.fusers (kept here so config coverage checks see every name).
IMPLEMENTED = frozenset({"no_gate", "oracle_gate", "view_gate"})

_STUBS: dict[str, str] = {
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

import sensorfault.gs.fusers  # noqa: E402,F401  (registers no_gate, oracle_gate, view_gate)

STUB_NAMES = frozenset(_STUBS)
GATE_TESTBED_NAMES = STUB_NAMES | IMPLEMENTED
