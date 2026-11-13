"""Stub registrations for the gate test bed (design v2: view-level gate + monocular-depth pipeline).

None of these is implemented yet. Each registers a name so that configs can reference it and the
runner can skip it with a clear reason. The implemented, still-useful pieces from design v1 live in
the sibling modules (e.g. ``consistency_voting`` is the COLMAP-style pixel filter, ``median_fusion``
is the per-pixel median voting baseline).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sensorfault.baselines.base import BaseFuser, register
from sensorfault.types import DepthFrame, FusionResult

_STUBS: dict[str, str] = {
    "no_gate": "Test bed with every depth source trusted and a quadratic pose-graph cost.",
    "oracle_gate": "Upper bound: excludes exactly the faulty sources (manifest or laser reference).",
    "view_gate": (
        "The contribution: view-level gate; params: statistic in {chi2, evidence}. "
        "Admission test per PLAN.md section 2 and src/sensorfault/elim/admission.py (grid form)."
    ),
    "robust_kernel_pgo": (
        "Robust kernels in the pose graph, no gate; params: kernel in "
        "{huber, cauchy, geman_mcclure_gnc, switchable, dcs}."
    ),
    "colmap_consistency_filter": (
        "Pixel-wise COLMAP geometric consistency filter on the test bed's depth sources "
        "(see consistency_voting for the implemented numpy test)."
    ),
    "plain_voting": "Per-pixel median / majority across the ensemble and neighbouring views.",
    "estimate_and_correct": "Refit per-source alignment without excluding any source.",
    "mast3r_sfm": "Reference point (not a rival): MASt3R-SfM run as released on the same images.",
    "vggt": "Reference point (not a rival): VGGT run as released on the same images.",
    "colmap_sfm": "Classical SfM reference run on the same images.",
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
