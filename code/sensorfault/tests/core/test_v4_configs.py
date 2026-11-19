"""Schema checks for the design-v4 stage configs (per-source information on surfels). No experiment is run."""

from pathlib import Path

import pytest

from sensorfault.runner import load_config

CFG = Path(__file__).resolve().parents[2] / "configs"
STAGES = [
    "stage0_entity",
    "stage_a_depth_only_honesty",
    "stage_b_image_information",
    "stage_c_two_source_map",
    "real_scenes_check",
]
FIRST_PAPER = {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H14"}
STOPS = {None, "S1", "S2", "S3", "S4", "S5", "S1 on real data"}


@pytest.mark.parametrize("name", STAGES)
def test_stage_config_complete(name):
    c = load_config(CFG / f"{name}.yaml")
    assert c["id"] == name
    assert c["status"] == "preregistered_draft" and c["design"] == "v4_per_source_information"
    assert c["implemented"] is False and c["paper_items"] is not None
    for h in c["hypotheses"]:
        assert h["id"] in FIRST_PAPER and h["stop"] in STOPS
        for key in ("prediction", "primary_metric", "supported_if", "refuted_if"):
            assert h[key], (name, h["id"], key)


def test_first_paper_hypotheses_all_have_a_stage():
    seen = {h["id"] for n in STAGES for h in load_config(CFG / f"{n}.yaml")["hypotheses"]}
    assert seen == FIRST_PAPER


def test_view_gate_configs_are_legacy():
    assert not list(CFG.glob("h[1-7]_*.yaml")) and (CFG / "legacy_view_gate").is_dir()
