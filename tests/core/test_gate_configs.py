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
    "h7_ensemble_ablation",
]
BREAK = [
    "scenewide_model_failure",
    "few_views",
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
        "H7",
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


def test_default_is_one_metric_model_per_image_and_ensemble_is_ablation_only():
    c = load_config(CFG / "base.yaml")
    assert c["depth_model"]["default"] == "depth_pro" and c["depth_model"]["one_model_per_image"]
    assert c["ensemble_ablation"]["only_used_by"] == ["h7_ensemble_ablation"]
    m = yaml.safe_load((CFG / "depth_models" / "depth_models.yaml").read_text())
    assert m["default"]["id"] == "depth_pro" and m["default"]["output"] == "metric"
    assert set(m["ensemble_sizes"]) == {"M2", "M3", "M4"}
    assert all(m["default"]["id"] in v for v in m["ensemble_sizes"].values())
    # no design-v2 experiment config other than H7 may use an ensemble baseline
    for name in HYP + [f"breakdown_{b}" for b in BREAK] + ["ablations_gate"]:
        if name == "h7_ensemble_ablation":
            continue
        cfg = load_config(CFG / f"{name}.yaml")
        assert not {"cross_model_voting", "gate_ensemble"} & set(cfg["baselines"]), name


def test_h7_covers_shared_failures_voting_and_ensemble_value():
    c = load_config(CFG / "h7_ensemble_ablation.yaml")
    assert set(c["baselines"]) == {"cross_model_voting", "gate_surface_evidence", "gate_ensemble"}
    assert "joint_failure_lift" in c["metrics"]["primary"]
    ids = baseline_ids()
    assert ids["cross_model_voting"] == "plain_voting" and ids["gate_ensemble"] == "view_gate"


def test_fault_grid_targets_view_depth_not_estimators():
    g = yaml.safe_load((CFG / "faults" / "view_fault_grid.yaml").read_text())
    assert set(g["classes"]) == {"smooth_warp", "region_failure", "nonaffine_distortion"}
    assert "ensemble_ablation_only" in g and "faulted_estimators_per_view" not in g
    for d in ("eth3d", "scannetpp", "replica_render", "trajectory_benchmarks"):
        assert (CFG / "datasets" / f"{d}.yaml").exists()
