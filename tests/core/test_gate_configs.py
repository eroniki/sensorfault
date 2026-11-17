"""Schema checks for the design-v3 configs (depth-only gate on surfels whose uncertainty renders).

No experiment is run.
"""

from pathlib import Path

import pytest
import yaml

import sensorfault.baselines  # noqa: F401  (registers the stubs)
from sensorfault.baselines.base import get_fuser, list_fusers
from sensorfault.baselines.gate_testbed import GATE_TESTBED_NAMES, IMPLEMENTED
from sensorfault.runner import load_config

CFG = Path(__file__).resolve().parents[2] / "configs"
HYP = [
    "h1_residual_measurement",
    "h2_cross_view_repetition",
    "h3_detection_vs_cheap_baselines",
    "h4_reconstruction_with_and_without_gate",
    "h5_rendered_uncertainty_calibration",
    "h6_gate_design_ablations",
    "h7_pose_perturbation_check",
]
BREAK = [
    "common_mode_repetition",
    "low_overlap_untestable",
    "faulty_views_dominate_region",
    "misspecified_measurement_model",
    "linearisation_limit",
    "loop_masking_swamping",
]
ALL = HYP + [f"breakdown_{b}" for b in BREAK] + ["ablations_gate"]


def baseline_ids():
    ids = {}
    for f in (CFG / "baselines").glob("*.yaml"):
        for b in yaml.safe_load(f.read_text())["baselines"]:
            ids[b["id"]] = b["fuser"]
    return ids


@pytest.mark.parametrize("name", ALL)
def test_experiment_configs_complete(name):
    c = load_config(CFG / f"{name}.yaml")
    assert c["status"] == "preregistered_draft" and c["design"] == "v3_rendered_uncertainty_gate"
    h = c["hypothesis"]
    assert h["prediction"] and h["falsification_criterion"]
    assert h["independent_variables"] and h["controlled_variables"]
    assert len(c["seeds"]["master"]) >= 5
    assert c["scenes"]["dev"] and c["scenes"]["test"]
    assert not set(c["scenes"]["dev"]) & set(c["scenes"]["test"])
    ids = baseline_ids()
    assert c["baselines"] and set(c["baselines"]) <= set(ids), set(c["baselines"]) - set(ids)
    assert c["metrics"]
    if name == "h7_pose_perturbation_check":
        # H7 has a run (experiments/h7_pose_check.py); `implemented` names the setting it ran in.
        assert isinstance(c["implemented"], str) and c["implemented"]
    else:
        assert c["implemented"] is False


def test_baseline_files_use_registered_names_and_cover_all_stubs():
    used = set(baseline_ids().values())
    assert used == set(GATE_TESTBED_NAMES)
    assert used <= set(list_fusers())
    for n in used - IMPLEMENTED:
        ok, reason = type(get_fuser(n)).available()
        assert not ok and "not implemented" in reason


def test_hypothesis_ids_and_residual_measurement_first():
    assert [load_config(CFG / f"{n}.yaml")["hypothesis"]["id"] for n in HYP] == [
        f"H{i}" for i in range(1, 8)
    ]
    h1 = load_config(CFG / "h1_residual_measurement.yaml")
    assert h1["first_task"] is True
    assert load_config(CFG / "base.yaml")["measurement_model"]["calibration"]["first_task"] is True


def test_no_sub_noise_claims_and_faults_are_regional():
    g = yaml.safe_load((CFG / "faults" / "view_fault_grid.yaml").read_text())
    assert g["no_sub_noise_claims"] is True
    assert min(g["injected_regional"]["magnitude_sigma"]) >= 2
    assert max(g["injected_regional"]["area_fraction"]) < 1
    assert {"specular_glass_water", "textureless_roof_road_wall"} <= set(
        g["natural_failures"]["kinds"]
    )
    assert "repeated_within_block_of_neighbouring_frames" in g["injected_regional"]["across_views"]
    assert "repeated_same_error_across_views" in g["injected_regional"]["across_views"]
    assert g["controls"]["whole_view_affine"]["absorbed"] is True
    for name in ALL:
        assert "sub_noise" not in (CFG / f"{name}.yaml").read_text()


def test_surface_appearance_and_uncertainty_blocks():
    b = load_config(CFG / "base.yaml")
    assert b["surface"]["representation"] == "2dgs_surfels" and b["surface"]["renderer"] == "gsplat"
    ap = b["appearance"]
    assert ap["updates_map_geometry"] is False and ap["updates_opacity"] is False
    assert ap["refines_poses"] is True and ap["gate_inputs"] == "depth only"
    u = b["uncertainty"]
    assert u["dim"] == 10 and len(u["state"]) == 10 and u["state"][-1] == "d_logit_alpha"
    assert "frozen" in u["opacity"]["schedule"] and u["opacity"]["rho"] == 0.5
    assert "Newton" in u["leave_view_out"]
    assert set(u["overconfidence_ablations"]) == {
        "information_deflation_n_eff_over_n",
        "jackknife_inflation",
    }
    assert u["rendered_subblock"] == ["normal_offset_nu", "tilt_tau1", "tilt_tau2"]
    assert u["leave_view_out"].startswith("subtraction") and u["correspondences"] == "none"
    assert "overconfident" in u["limitation"]


def test_required_baselines_present():
    ids = baseline_ids()
    for b in (
        "spotless_splats",
        "robustnerf_loss",
        "robust_depth_huber",
        "aligned_rms",
        "depth_confidence",
        "gate",
        "no_gate",
    ):
        assert b in ids
    h4 = load_config(CFG / "h4_reconstruction_with_and_without_gate.yaml")
    assert {"no_gate", "gate", "spotless_splats", "robustnerf_loss"} <= set(h4["baselines"])
    assert h4["hypothesis"]["levels"]["dataset"] == ["usegeo", "matrixcity", "eth3d"]
    h6 = load_config(CFG / "h6_gate_design_ablations.yaml")
    assert {"gate_fisherrf_diag", "gate_bayesrays", "gate_keep_unsupported"} <= set(h6["baselines"])


def test_h2_tests_repetition_in_textureless_regions():
    h2 = load_config(CFG / "h2_cross_view_repetition.yaml")
    assert "textureless_roof_road_wall" in h2["hypothesis"]["levels"]["appearance_class"]
    assert h2["hypothesis"]["levels"]["frame_separation"][0] == 1
    assert h2["metrics"]["textureless_min"] == 0.5


def test_theory_modules_named_in_config_exist():
    b = load_config(CFG / "base.yaml")
    root = CFG.parent
    assert (root / b["uncertainty"]["theory_module"]).exists()
    assert (root / b["integrity"]["theory_module"]).exists()


def test_single_free_noise_variable_and_shared_distance_bias():
    mm = load_config(CFG / "base.yaml")["measurement_model"]
    npar = mm["noise_parameters"]
    assert npar["free"] == ["gamma (one scene temperature)"]
    assert npar["per_view"] == "none" and npar["per_pixel"] == "none"
    assert "leave-view-out" in npar["gamma_estimator"]
    sb = mm["shared_bias"]
    assert sb["term"] == "distance" and "beta d^2" in sb["model"]
    assert sb["per_view_bias_field"].startswith("not in this paper")
    assert "omega" in mm["calibration"]["fit"] and "beta" in mm["calibration"]["fit"]


def test_poses_use_rule_b_and_pose_faults_are_only_a_check():
    p = load_config(CFG / "base.yaml")["poses"]
    assert p["input"].startswith("posed images") and "B-spline" in p["trajectory"]
    assert "no feature correspondences" in p["refinement"] and "Sigma_q" in p["tested_pose"]
    assert p["ablation"] == "perturbed starting poses"
    assert p["pose_faults"].startswith("not a target of the gate")


def test_outer_loop_block_and_its_ablations():
    from sensorfault.eval import gate_loop as gl

    lp = load_config(CFG / "base.yaml")["outer_loop"]
    assert lp["enabled"] is True and lp["round_cap"] == 10 and lp["readmission"] is False
    assert lp["policy"].startswith("one_per_round") and lp["exclusion"].startswith("drop")
    assert "M = n_views * round_cap" in lp["false_alarm_control"]["family_size"]
    assert "train step" in lp["differentiability"]
    assert gl.holm_family_size(30, lp["round_cap"]) == 300
    ids = baseline_ids()
    for b in ("gate_single_pass", "gate_drop_all_rejected", "gate_downweight", "gate_readmit"):
        assert ids[b] == "view_gate"
    h6 = load_config(CFG / "h6_gate_design_ablations.yaml")
    assert "gate_single_pass" in h6["baselines"] and h6["hypothesis"]["levels"]["round_cap"] == [
        1,
        3,
        10,
    ]


def test_aerial_setting_datasets_and_leave_block_out():
    b = load_config(CFG / "base.yaml")
    ds = b["setting"]["datasets"]
    assert ds["main_real"] == "usegeo" and ds["synthetic"] == "matrixcity"
    assert ds["sanity_check"] == "eth3d" and "scannetpp" in ds["dropped"]
    for d in ("usegeo", "h3d", "matrixcity", "eth3d", "considered"):
        assert (CFG / "datasets" / f"{d}.yaml").exists()
    assert not (CFG / "datasets" / "scannetpp.yaml").exists()
    usegeo = yaml.safe_load((CFG / "datasets" / "usegeo.yaml").read_text())
    assert usegeo["licence"].startswith("CC BY-NC-SA 4.0")
    blk = b["leave_block_out"]
    assert blk["sweep"] == [1, 3, 5, 9] and blk["ablation"].startswith("B = 1")
    assert "M = n_views * round_cap" in blk["holm_family"]
    ids = baseline_ids()
    assert ids["gate_block_1"] == "view_gate" and ids["gate_perturbed_poses"] == "view_gate"
    h6 = load_config(CFG / "h6_gate_design_ablations.yaml")
    assert h6["hypothesis"]["levels"]["block_size_B"] == [1, 3, 5, 9]
    assert (CFG.parent / b["poses"]["theory_module"]).exists()


def test_off_the_shelf_depth_models_and_pose_check():
    b = load_config(CFG / "base.yaml")
    assert b["depth_model"]["off_the_shelf_only"] is True
    assert "no unreleased or in-house" in b["depth_model"]["policy"]
    assert set(b["depth_model"]["altitude_comparison"]["models"]) >= {"depth_pro", "metric3d_v2"}
    pc = b["pose_check"]
    assert set(pc["checks"]) == {"a", "b"} and "recovery error" in pc["report"]
    h7 = load_config(CFG / "h7_pose_perturbation_check.yaml")
    assert h7["hypothesis"]["id"] == "H7" and "rotation_deg" in h7["hypothesis"]["levels"]
    assert b["downloads"]["policy"].startswith("no large download")


def test_all_datasets_are_recorded_as_non_commercial():
    for d in ("usegeo", "matrixcity", "h3d", "urbanscene3d"):
        y = yaml.safe_load((CFG / "datasets" / f"{d}.yaml").read_text())
        assert y["non_commercial"] is True, d
