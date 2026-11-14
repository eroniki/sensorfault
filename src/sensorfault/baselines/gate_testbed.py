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
    "no_gate": "Test bed with every view's depth trusted and a quadratic pose-graph cost.",
    "oracle_gate": "Upper bound: drops exactly the faulty views (manifest or laser reference).",
    "view_gate": (
        "The contribution: view-level fault detection and exclusion (integrity monitoring) derived from an explicit "
        "measurement model (Sigma_r, whitened nuisance-projected statistics); params: "
        "consensus in {shared_surface, pairwise}, statistic in {energy_whitened_projected, "
        "region_scan_matched, combined_region_scan_energy, evidence}, cross_fit in {True, False, exact_leave_one_view_out}, "
        "densification in {fixed_N, adaptive_matched_final_N}, sources in {one_model_per_view, "
        "ensemble_M_models_per_view}. Admission test per PLAN.md section 2 and "
        "src/sensorfault/elim/admission.py (grid form); detection theory in "
        "src/sensorfault/eval/detection_theory.py."
    ),
    "robust_kernel_pgo": (
        "Robust kernels in the pose graph, no gate; params: kernel in "
        "{huber, cauchy, geman_mcclure_gnc, switchable, dcs, max_mixture}."
    ),
    "colmap_consistency_filter": (
        "Pixel-wise COLMAP geometric consistency filter on the test bed's depth sources "
        "(see consistency_voting for the implemented numpy test)."
    ),
    "soft_view_weights_pgo": (
        "Baseline: soft per-view down-weighting inside the differentiable pose graph "
        "(Student-t or learned-scale weight); down-weights but neither excludes nor alarms."
    ),
    "pixel_count_test": (
        "Baseline: pixel-wise flags (threshold k sigma) aggregated over the view into a count test "
        "at the same false-alarm rate; strongest pixel-based competitor (see eval/detection_theory.py)."
    ),
    "plain_voting": (
        "Per-pixel median / majority; param rule in {cross_view_median, cross_view_majority} "
        "(baseline) or cross_model_median_or_majority (ensemble ablation H7)."
    ),
    "estimate_and_correct": "Refit per-view alignment without dropping any view.",
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
