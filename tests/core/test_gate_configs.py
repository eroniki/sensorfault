"""Schema checks for the design-v2.3 (integrity-monitoring, view-level gate) configs. No experiment is run."""

from pathlib import Path

import pytest
import yaml

import sensorfault.baselines  # noqa: F401  (registers the stubs)
from sensorfault.baselines.base import get_fuser, list_fusers
from sensorfault.baselines.gate_testbed import GATE_TESTBED_NAMES
from sensorfault.runner import load_config

CFG = Path(__file__).resolve().parents[2] / "configs"
HYP = [
    "h1_regime_map_subnoise_bias",
    "h2_consequence_vs_baselines",
    "h3_statistic_choice",
    "h4_leave_view_out_alignment_support",
    "h5_placement_soft_downweighting",
    "h6_price_of_monitoring_reference_points",
    "h7_shared_surface_vs_pairwise",
    "h8_pose_faults",
    "h9_protection_level",
    "h10_ensemble_ablation",
]
BREAK = [
    "scenewide_model_failure",
    "misspecified_measurement_model",
    "few_views",
    "sub_noise",
    "small_n_eff",
    "sparse_outliers_control",
    "low_overlap",
    "faulty_views_dominate_surface",
    "association_failure",
    "large_false_loop_closures",
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
        f"H{i}" for i in range(1, 11)
    ]


def test_h1_is_the_regime_map_with_the_affine_control_and_no_uniform_fault():
    c = load_config(CFG / "h1_regime_map_subnoise_bias.yaml")
    lv = c["hypothesis"]["levels"]
    assert 0.3 in lv["delta_sigma"] and 0.5 in lv["fraction_f"] and 0.02 in lv["fraction_f"]
    # a bias on ALL pixels is an offset: it must not be in the fault grid, only in the control
    assert 1 not in lv["fraction_f"] and max(lv["fraction_f"]) < 1
    assert "uniform_bias_f1" in lv["affine_control"]
    assert {"colmap_geom_consistency", "pixel_count_test", "soft_view_weights"} <= set(
        c["baselines"]
    )
    rm = load_config(CFG / "base.yaml")["regime_map"]
    assert rm["regimes"]["sub_noise"] == {"delta_le": 0.5, "f_ge": 0.2, "f_le": 0.8}
    assert rm["regimes"]["sparse_outlier"]["delta_ge"] == 4
    assert rm["affine_control"]["fraction_f"] == 1.0
    assert c["metrics"]["subnoise_gap_points"] == 30 and c["metrics"]["mdb_tolerance_factor"] == 2
    g = yaml.safe_load((CFG / "faults" / "view_fault_grid.yaml").read_text())
    assert max(g["depth_faults"]["coherent_bias"]["fraction_f"]) < 1
    assert g["depth_faults"]["affine_control"]["uniform_bias_f1"] is True


def test_measurement_model_block_lists_the_six_uncertainty_sources_and_matches_theory():
    import numpy as np

    from sensorfault.eval import detection_theory as dt

    mm = load_config(CFG / "base.yaml")["measurement_model"]
    assert [u.split()[0] for u in mm["sources_of_uncertainty"]] == [f"U{i}" for i in range(1, 7)]
    assert "Sigma_surf" in mm["residual_covariance"] and "J_q" in mm["residual_covariance"]
    assert "NOT faults" in mm["fault_definition"]
    assert mm["n_eff"].startswith("tr(C_r)^2 / tr(C_r^2)")
    assert mm["calibration"]["fit_once_per_depth_model_on_clean_dev"][0].startswith("sigma")
    conf = mm["calibration"]["confidence_outputs_per_public_readme_checked_2026_11_14"]
    assert conf["depth_pro"].startswith("no") and conf["metric3d_v2"].startswith("yes")
    # the configured regimes are consistent with the theory: with n_eff=1000 the gate can
    # detect the sub-noise corner and cannot detect the affine control
    n, d = 1000, 0.5
    kap = float(np.sqrt(n * 0.5 * 0.5))
    assert dt.mdb_scan(kap, 100) < d  # detectable at delta = 0.5, f = 0.5
    assert dt.mdb_matched_projected(0.0) == float("inf")  # f = 1 control: undetectable


def test_integrity_block_matches_the_detection_theory_defaults():
    from sensorfault.eval import detection_theory as dt

    ig = load_config(CFG / "base.yaml")["integrity"]
    assert ig["false_alarm_per_view"] == 0.05 and ig["missed_detection"] == 0.1
    assert ig["z_alpha"] == pytest.approx(dt.z_alpha(ig["false_alarm_per_view"]), abs=1e-2)
    assert ig["z_beta"] == pytest.approx(dt.z_beta(ig["missed_detection"]), abs=1e-3)
    assert "no_consensus_per_view" in ig["alarms"]
    assert "region_scan_matched" in ig["statistics"] and "mean_shift" not in ig["statistics"]


def test_h5_soft_downweighting_is_a_baseline_and_h8_has_loop_closure_baselines():
    h5 = load_config(CFG / "h5_placement_soft_downweighting.yaml")
    assert "soft_view_weights" in h5["baselines"]
    h8 = load_config(CFG / "h8_pose_faults.yaml")
    assert {"rk_switchable", "rk_dcs", "rk_max_mixture"} <= set(h8["baselines"])
    g = yaml.safe_load((CFG / "faults" / "view_fault_grid.yaml").read_text())
    assert set(g["pose_faults"]) == {"spoofed_prior", "intermittent_glitch", "false_loop_closure"}


def test_h9_protection_level_is_optional_and_conditional():
    c = load_config(CFG / "h9_protection_level.yaml")
    assert c["optional"] is True
    assert "pl_exceedance_rate" in c["metrics"]["primary"]
    assert "A1-A4" in c["hypothesis"]["prediction"]


def test_h7_covers_efficiency_robustness_and_mcmc_ablation():
    c = load_config(CFG / "h7_shared_surface_vs_pairwise.yaml")
    assert {
        "gate_pairwise_evidence",
        "gate_surface_evidence",
        "gate_surface_mcmc_fixed_budget",
    } <= set(
        c["baselines"]
    )
    assert c["metrics"]["speedup_threshold"] == 2.0
    ids = baseline_ids()
    assert all(ids[b] == "view_gate" for b in c["baselines"])


def test_shared_surface_params_are_declared():
    c = load_config(CFG / "base.yaml")
    ss = c["shared_surface"]
    assert ss["inputs"] == "depth_maps_only" and ss["colour"] == "out_of_scope"
    assert ss["fusion"]["form"] == "information" and ss["fusion"]["folds"] == 0
    assert ss["fusion"]["refits"] == 0
    assert ss["densification"].startswith("none")
    assert "fixed_budget" not in ss and "cross_fit" not in ss
    assert ss["primitive"]["extent"]["used_for"] == "rendering_weights_only"
    assert set(ss["region_outputs"]) == {
        "posterior_mean_var",
        "protection_level",
        "n_views_availability",
    }
    assert c["testbed"]["soft_downweighting_is_baseline"] is True


def test_default_is_one_metric_model_per_image_and_ensemble_is_ablation_only():
    c = load_config(CFG / "base.yaml")
    assert c["depth_model"]["default"] == "depth_pro" and c["depth_model"]["one_model_per_image"]
    assert c["ensemble_ablation"]["only_used_by"] == ["h10_ensemble_ablation"]
    m = yaml.safe_load((CFG / "depth_models" / "depth_models.yaml").read_text())
    assert m["default"]["id"] == "depth_pro" and m["default"]["output"] == "metric"
    assert set(m["ensemble_sizes"]) == {"M2", "M3", "M4"}
    assert all(m["default"]["id"] in v for v in m["ensemble_sizes"].values())
    for name in HYP + [f"breakdown_{b}" for b in BREAK] + ["ablations_gate"]:
        if name == "h10_ensemble_ablation":
            continue
        cfg = load_config(CFG / f"{name}.yaml")
        assert not {"cross_model_voting", "gate_ensemble"} & set(cfg["baselines"]), name


def test_h10_covers_shared_failures_voting_and_ensemble_value():
    c = load_config(CFG / "h10_ensemble_ablation.yaml")
    assert set(c["baselines"]) == {"cross_model_voting", "gate_surface_evidence", "gate_ensemble"}
    assert "joint_failure_lift" in c["metrics"]["primary"]
    ids = baseline_ids()
    assert ids["cross_model_voting"] == "plain_voting" and ids["gate_ensemble"] == "view_gate"


def test_fault_grid_has_the_regime_map_and_targets_views_not_estimators():
    g = yaml.safe_load((CFG / "faults" / "view_fault_grid.yaml").read_text())
    assert set(g["depth_faults"]) == {
        "coherent_bias",
        "affine_control",
        "smooth_warp",
        "region_failure",
        "nonaffine_distortion",
    }
    assert "ensemble_ablation_only" in g and "faulted_estimators_per_view" not in g
    for d in ("eth3d", "scannetpp", "replica_render", "trajectory_benchmarks"):
        assert (CFG / "datasets" / f"{d}.yaml").exists()
