import json
import sys
from pathlib import Path

import pytest
import yaml

from sensorfault.runner import load_config, main, resolve_fusers

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs"
HYP = [
    "h1_coherent_fault_regimes",
    "h2_entropy_vs_loo_chi2",
    "h3_visibility_grazing_false_elim",
    "h4_fixedN_vs_adc",
    "h5_learned_fusion_bias",
]
BREAK = [
    "majority_correlated",
    "common_mode",
    "k2",
    "sub_noise",
    "sparse_outlier_control",
    "no_overlap",
]
REGISTRY_NAMES = {
    "tsdf_open3d",
    "median_fusion",
    "consistency_voting",
    "merrell_visibility",
    "tvl1_range",
    "loo_chi2",
    "subset_msac",
    "openmvs",
    "senfunet",
    "routedfusion",
    "gs_mcmc_depth",
    "twodgs_depth",
    "splat_consensus",
}


@pytest.mark.parametrize("name", HYP + [f"breakdown_{b}" for b in BREAK] + ["ablations_A0_A11"])
def test_experiment_configs_complete(name):
    c = load_config(CFG / f"{name}.yaml")
    assert c["status"] == "preregistered_draft"
    h = c["hypothesis"]
    assert h["prediction"] and h["falsification_criterion"]
    assert h["independent_variables"] and h["controlled_variables"]
    assert len(c["seeds"]["master"]) >= 5 and c["scenes"]["dev"] and c["scenes"]["test"]
    assert not set(c["scenes"]["dev"]) & set(c["scenes"]["test"])
    assert c["baselines"] and c["metrics"]
    assert set(c["baselines"]) <= REGISTRY_NAMES


def test_baseline_files_cover_all_registry_names():
    names = set()
    for f in (CFG / "baselines").glob("*.yaml"):
        names |= {b["fuser"] for b in yaml.safe_load(f.read_text())["baselines"]}
    assert names == REGISTRY_NAMES


def test_ablations_and_datasets():
    a = yaml.safe_load((CFG / "ablations_A0_A11.yaml").read_text())
    assert [x["id"] for x in a["ablations"]] == [f"A{i}" for i in range(12)]
    for d in ("skoltech3d", "replica_multisensor", "us3d_optional"):
        assert (CFG / "datasets" / f"{d}.yaml").exists()


def test_overlay_merge(tmp_path):
    ov = tmp_path / "o.yaml"
    ov.write_text("synthetic: {n_frames: 3}\nseeds: {master: [1,2]}\n")
    c = load_config(CFG / "h1_coherent_fault_regimes.yaml", [ov])
    assert (
        c["synthetic"]["n_frames"] == 3
        and c["synthetic"]["width"] == 128
        and c["seeds"]["master"] == [1, 2]
    )


def test_dry_run_end_to_end(tmp_path):
    rc = main(
        ["--config", str(CFG / "h2_entropy_vs_loo_chi2.yaml"), "--dry-run", "--out", str(tmp_path)]
    )
    assert rc == 0
    run = next((tmp_path / "h2_entropy_vs_loo_chi2").iterdir())
    res = json.loads((run / "results.json").read_text())
    assert res["dry_run"] and res["n_rows"] > 0 and not any("error" in r for r in res["rows"])
    assert "builtin_oracle_exclusion" in res["fusers_run"]
    assert (run / "run_manifest.json").exists() and list((run / "fault_manifests").glob("*.json"))
    oracle = [r for r in res["rows"] if r["fuser"] == "builtin_oracle_exclusion"][0]
    assert oracle["detection"]["recall"] == 1.0


def test_runs_without_baselines_package(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "sensorfault.baselines", None
    )  # makes `import` raise ImportError
    f, skipped = resolve_fusers(["median_fusion", "openmvs"], {}, True)
    assert set(f) == {"builtin_naive_union", "builtin_oracle_exclusion"}
    assert all("not importable" in v for v in skipped.values())
    assert (
        main(["--config", str(CFG / "breakdown_k2.yaml"), "--dry-run", "--out", str(tmp_path)]) == 0
    )


def test_test_split_guard(tmp_path):
    # Run from a cwd without the prereg tag; guard must trigger (no git repo tags in tmp).
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(SystemExit):
            main(
                [
                    "--config",
                    str(CFG / "breakdown_k2.yaml"),
                    "--split",
                    "test",
                    "--out",
                    str(tmp_path),
                ]
            )
    finally:
        os.chdir(cwd)
