"""Schema checks for the design-v2 (view-level gate) configs. No experiment is run."""

from pathlib import Path

import pytest
import yaml

import sensorfault.baselines  # noqa: F401  (registers the stubs)
from sensorfault.baselines.base import get_fuser, list_fusers
from sensorfault.baselines.gate_testbed import GATE_TESTBED_NAMES
from sensorfault.runner import load_config

CFG = Path(__file__).resolve().parents[2] / "configs"
HYP = [
    "h1_view_level_gate_vs_baselines",
    "h2_evidence_vs_chi2",
    "h3_cross_fitting_alignment_support",
    "h4_gate_placement",
    "h5_clean_cost_reference_points",
    "h6_shared_surface_vs_pairwise",
]
BREAK = [
    "common_mode_estimators",
    "m2",
    "sub_noise",
    "sparse_outliers_control",
    "low_overlap",
    "faulty_views_dominate_surface",
    "poor_provisional_poses",
]


def baseline_ids():
    ids = {}
    for f in (CFG / "baselines").glob("*.yaml"):
        for b in yaml.safe_load(f.read_text())["baselines"]:
            ids[b["id"]] = b["fuser"]
    return ids


@pytest.mark.parametrize("name", HYP + [f"breakdown_{b}" for b in BREAK] + ["ablations_gate"])
def test_gate_experiment_configs_complete(name):
    c = load_config(CFG / f"{name}.yaml")
    assert c["status"] == "preregistered_draft" and c["design"] == "v2_view_level_gate"
    h = c["hypothesis"]
    assert h["prediction"] and h["falsification_criterion"]
    assert h["independent_variables"] and h["controlled_variables"]
    assert len(c["seeds"]["master"]) >= 5
    assert c["scenes"]["dev"] and c["scenes"]["test"]
    assert not set(c["scenes"]["dev"]) & set(c["scenes"]["test"])
    ids = baseline_ids()
    assert c["baselines"] and set(c["baselines"]) <= set(ids), set(c["baselines"]) - set(ids)
    assert c["metrics"]
    assert c["implemented"] is False  # honest flag until the test bed exists


def test_baseline_files_use_registered_names_and_cover_all_stubs():
    used = set(baseline_ids().values())
    assert used == set(GATE_TESTBED_NAMES)
    assert used <= set(list_fusers())
    for n in used:
        ok, reason = type(get_fuser(n)).available()
        assert not ok and "not implemented" in reason


def test_hypotheses_match_the_manuscript_ids():
    assert [load_config(CFG / f"{n}.yaml")["hypothesis"]["id"] for n in HYP] == [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    ]


def test_h6_covers_efficiency_robustness_and_fixed_budget():
    c = load_config(CFG / "h6_shared_surface_vs_pairwise.yaml")
    assert {"gate_pairwise_evidence", "gate_surface_evidence", "gate_surface_adaptive"} <= set(
        c["baselines"]
    )
    assert c["metrics"]["speedup_threshold"] == 2.0
    ids = baseline_ids()
    assert all(ids[b] == "view_gate" for b in c["baselines"])


def test_shared_surface_params_are_declared():
    c = load_config(CFG / "base.yaml")
    ss = c["shared_surface"]
    assert ss["fixed_budget"] is True and ss["depth_only"] is True
    assert ss["cross_fit"]["folds_K"] == [3, 4, 6]


def test_fault_grid_and_ensemble_files_exist():
    g = yaml.safe_load((CFG / "faults" / "view_fault_grid.yaml").read_text())
    assert set(g["classes"]) == {"smooth_warp", "region_failure", "nonaffine_distortion"}
    e = yaml.safe_load((CFG / "estimators" / "ensemble.yaml").read_text())
    assert set(e["ensemble_sizes"]) == {"M3", "M4", "M5"}
    for d in ("eth3d", "scannetpp", "replica_render", "trajectory_benchmarks"):
        assert (CFG / "datasets" / f"{d}.yaml").exists()
