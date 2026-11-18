"""Golden regression test of the view gate on a non-planar scene: the Stanford bunny.

Six posed 32x32 RGB-D views of the decimated bunny (``sim.bunny``), 5 mm depth noise, and in the
faulty set a +5 cm bias on pixel window (8:24, 8:24) of view v3. ``view_gate`` with the numpy
reference backend (stride 2, ~270 surfels), one round (``round_cap=1``), ``sigma=0.01``.

Checked: the rendered views match the stored golden numbers; the gate excludes exactly {v3} on
the faulty set and nothing on the clean set; per-view log10 p-values and the surfel count match
the golden file. Known failures on other seeds / fault placements are strict xfails (marked slow),
so a fix shows up as XPASS.

Regenerate the golden file after an intended change:
    SENSORFAULT_UPDATE_GOLDEN=1 python -m pytest -q tests/gs/test_bunny_golden.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from sensorfault.gs.fusers import ViewGate
from sensorfault.gs.reference import NumpyReferenceBackend
from sensorfault.sim.bunny import ASSET, BunnyConfig, make_bunny_views

GOLDEN = Path(__file__).with_name("golden") / "bunny_gate.json"
UPDATE = os.environ.get("SENSORFAULT_UPDATE_GOLDEN") == "1"
GATE_CFG = {"round_cap": 1, "sigma": 0.01}
STRIDE = 2
CASES = {"faulty_v3_seed0": dict(fault_view=3, seed=0), "clean_seed0": dict(seed=0)}


def _render_summary(bv) -> dict:
    out = {}
    for f, rgb, d0, m in zip(bv.frames, bv.rgb, bv.depth_clean, bv.fault_mask):
        valid = d0 > 0
        out[f.sensor_id] = {
            "n_valid": int(valid.sum()),
            "depth_clean_sum": float(d0[valid].astype(np.float64).sum()),
            "depth_sum": float(f.depth[valid].astype(np.float64).sum()),
            "rgb_sum": int(rgb.astype(np.int64).sum()),
            "n_fault": int(m.sum()),
        }
    return out


def _run_gate(**kw) -> dict:
    bv = make_bunny_views(BunnyConfig(**kw))
    res = ViewGate(backend=NumpyReferenceBackend(stride=STRIDE)).fuse(bv.frames, GATE_CFG)
    pv = res.meta["history"][0]["pvals"]
    return {
        "excluded": sorted(v for v, flag in res.excluded_sensors.items() if flag),
        "n_surfels": len(res.points),
        "log10p": {v: math.log10(max(p, 1e-300)) for v, p in pv.items()},
    }


@pytest.fixture(scope="module")
def golden() -> dict:
    if UPDATE:
        g = {
            "asset_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
            "gate_cfg": {**GATE_CFG, "stride": STRIDE},
            "render": {
                k: _render_summary(make_bunny_views(BunnyConfig(**kw))) for k, kw in CASES.items()
            },
            "gate": {k: _run_gate(**kw) for k, kw in CASES.items()},
        }
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(g, indent=1, sort_keys=True) + "\n")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def gate_runs(golden) -> dict:
    return golden["gate"] if UPDATE else {k: _run_gate(**kw) for k, kw in CASES.items()}


def test_asset_unchanged(golden):
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == golden["asset_sha256"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_bunny_render_matches_golden(golden, case):
    got = _render_summary(make_bunny_views(BunnyConfig(**CASES[case])))
    exp = golden["render"][case]
    assert got.keys() == exp.keys()
    for v in exp:
        assert got[v]["n_valid"] == exp[v]["n_valid"], v
        assert got[v]["n_fault"] == exp[v]["n_fault"], v
        assert got[v]["rgb_sum"] == pytest.approx(exp[v]["rgb_sum"], rel=1e-3, abs=255), v
        for k in ("depth_clean_sum", "depth_sum"):
            assert got[v][k] == pytest.approx(exp[v][k], rel=1e-5), (v, k)


def test_faulty_view_rendered_with_bias(golden):
    exp = golden["render"]["faulty_v3_seed0"]
    assert [v for v, e in exp.items() if e["n_fault"]] == ["v3"]
    assert exp["v3"]["n_fault"] > 100  # most of the window lands on the bunny


def test_gate_flags_only_the_faulty_view(gate_runs):
    assert gate_runs["faulty_v3_seed0"]["excluded"] == ["v3"]


def test_gate_no_false_alarm_on_clean_bunny(gate_runs):
    assert gate_runs["clean_seed0"]["excluded"] == []


@pytest.mark.parametrize("case", sorted(CASES))
def test_gate_outputs_match_golden(golden, gate_runs, case):
    got, exp = gate_runs[case], golden["gate"][case]
    assert got["excluded"] == exp["excluded"]
    assert got["n_surfels"] == exp["n_surfels"]
    assert got["log10p"].keys() == exp["log10p"].keys()
    for v, lp in exp["log10p"].items():  # loose in log space: BLAS / platform differences
        assert got["log10p"][v] == pytest.approx(lp, abs=0.15 + 0.05 * abs(lp)), v


# ---------------------------------------------------------------------------------------------
# Known failures (found while building this test, 2026-11-18). Strict xfail: an XPASS means the
# gate got better here; then move the case into CASES and regenerate the golden file.
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="false alarm on end-of-arc view v0 (p ~ 2e-14), seed 1")
def test_known_false_alarm_clean_seed1():
    assert _run_gate(seed=1)["excluded"] == []


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="fault on end-of-arc view v5 missed (p ~ 0.23), seed 0")
def test_known_miss_fault_on_v5():
    assert _run_gate(fault_view=5, seed=0)["excluded"] == ["v5"]


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="stride-4 surfels (~65) too coarse for the bunny")
def test_known_coarse_surfels_fail():
    bv = make_bunny_views(BunnyConfig(fault_view=3, seed=0))
    res = ViewGate(backend=NumpyReferenceBackend(stride=4)).fuse(bv.frames, GATE_CFG)
    assert {v for v, f in res.excluded_sensors.items() if f} == {"v3"}


# ---------------------------------------------------------------------------------------------
# GPU variant (gsplat backend): same scenes, behaviour only (no golden numbers: different fitter)
# ---------------------------------------------------------------------------------------------


# TITAN RTX, gsplat 1.5.3, f92aff8 (2026-11-18): the clean set rejects v3 (log10 p -12.2; v2 -7.1)
# while the numpy backend passes it, so the faulty case passing here does not show detection (v3 is
# an outlier with or without the bias: log10 p -9.1 with it).
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("faulty_v3_seed0", ["v3"]),
        pytest.param(
            "clean_seed0",
            [],
            marks=pytest.mark.xfail(strict=True, reason="gsplat fit: clean v3 rejected (p~6e-13)"),
        ),
    ],
)
def test_gsplat_gate_on_bunny(case, expected):
    from sensorfault.gs.fit import GsplatBackend

    ok, why = GsplatBackend.available()
    if not ok:
        pytest.skip(why)
    bv = make_bunny_views(BunnyConfig(**CASES[case]))
    res = ViewGate(backend=GsplatBackend(stride=STRIDE)).fuse(bv.frames, GATE_CFG)
    assert sorted(v for v, f in res.excluded_sensors.items() if f) == expected
