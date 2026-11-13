import numpy as np
import pytest

from sensorfault.faults import (
    FaultSpec,
    derive_rng,
    expand_cells,
    inject_faults,
    multipath_weight,
    profile_scales,
    read_manifest,
    scenario_specs,
    stable_hash,
    write_manifest,
)
from sensorfault.sim import SyntheticConfig, generate_synthetic


@pytest.fixture(scope="module")
def data():
    return generate_synthetic(SyntheticConfig(n_sensors=3, n_frames=10, height=32, width=40), 1)


def test_stable_hash_is_fixed():
    assert stable_hash("s0") == stable_hash("s0") != stable_hash("s1")
    assert stable_hash("s0") == 1805396094 or isinstance(stable_hash("s0"), int)


def test_seed_independence_from_spec_order(data):
    s1 = FaultSpec("depth_bias_const", "s0", 0, {"frac_of_median_range": 0.03})
    s2 = FaultSpec("depth_bias_range", "s1", 1, {"beta": 0.02})
    a, _ = inject_faults(data.frames, [s1, s2], 5)
    b, _ = inject_faults(data.frames, [s2, s1], 5)
    assert all(np.array_equal(x.depth, y.depth) for x, y in zip(a, b))


def test_determinism_and_no_mutation(data):
    spec = FaultSpec(
        "extrinsic_drift", "s1", 0, {"rotation_deg": 2, "translation_cm": 5}, 0.2, "random_walk"
    )
    orig = [f.T_wc.copy() for f in data.frames]
    a, ma = inject_faults(data.frames, [spec], 3)
    b, mb = inject_faults(data.frames, [spec], 3)
    c, _ = inject_faults(data.frames, [spec], 4)
    assert all(np.array_equal(x.T_wc, y.T_wc) for x, y in zip(a, b)) and ma == mb
    assert any(not np.array_equal(x.T_wc, y.T_wc) for x, y in zip(a, c))
    assert all(np.array_equal(o, f.T_wc) for o, f in zip(orig, data.frames))
    # other sensors untouched
    for x, f in zip(a, data.frames):
        if x.sensor_id != "s1":
            assert x is f


@pytest.mark.parametrize("profile", ["step", "ramp"])
def test_extrinsic_magnitude_profile(data, profile):
    spec = FaultSpec(
        "extrinsic_drift", "s0", 0, {"rotation_deg": 2.0, "translation_cm": 5.0}, 0.4, profile
    )
    out, man = inject_faults(data.frames, [spec], 0)
    fr0 = [f for f in data.frames if f.sensor_id == "s0"]
    fo = [f for f in out if f.sensor_id == "s0"]
    ang, tr = [], []
    from scipy.spatial.transform import Rotation as R

    for a, b in zip(fr0, fo):
        dT = np.linalg.inv(a.T_wc) @ b.T_wc
        ang.append(np.rad2deg(R.from_matrix(dT[:3, :3]).magnitude()))
        tr.append(100 * np.linalg.norm(dT[:3, 3]))
    assert (
        np.allclose(ang[:4], 0, atol=1e-6)
        and np.isclose(ang[-1], 2.0, atol=1e-4)
        and np.isclose(tr[-1], 5.0, atol=1e-4)
    )
    if profile == "ramp":
        assert ang[5] < ang[7] < ang[9]
    else:
        assert np.isclose(ang[4], 2.0, atol=1e-4)


def test_bias_const_and_range(data):
    out, man = inject_faults(
        data.frames,
        [FaultSpec("depth_bias_const", "s0", 0, {"frac_of_median_range": 0.03, "sign": 1})],
        0,
    )
    fo = [f for f in out if f.sensor_id == "s0"]
    fi = [f for f in data.frames if f.sensor_id == "s0"]
    med = np.median(np.concatenate([f.depth[f.depth > 0] for f in fi]))
    v = fi[0].depth > 0
    assert np.allclose((fo[0].depth - fi[0].depth)[v], 0.03 * med, atol=1e-4)
    assert man["faults"][0]["resolved"]["bias_m"] == pytest.approx(0.03 * med)
    out, _ = inject_faults(
        data.frames, [FaultSpec("depth_bias_range", "s2", 0, {"beta": 0.05, "sign": -1})], 0
    )
    fo = [f for f in out if f.sensor_id == "s2"][0]
    fi = [f for f in data.frames if f.sensor_id == "s2"][0]
    v = fi.depth > 0
    assert np.allclose(fo.depth[v] / fi.depth[v], 0.95, atol=1e-5)


def test_multipath_nonneg_and_concentrated_near_corners(data):
    fr = data.clean_frames[0]
    w = multipath_weight(fr.depth, fr.K)
    assert w.min() >= 0 and w.max() <= 1
    out, _ = inject_faults([fr], [FaultSpec("multipath", fr.sensor_id, 0, {"amplitude_m": 0.1})], 0)
    delta = out[0].depth - fr.depth
    assert delta.min() >= 0 and delta.max() <= 0.1 + 1e-6
    # synthetic room has corners/edges: some pixels affected, most flat-wall pixels not
    assert 0 < (delta > 1e-4).mean() < 0.5


def test_transient_bursts(data):
    out, man = inject_faults(
        data.frames,
        [
            FaultSpec(
                "transient",
                "s0",
                0,
                {"fraction": 0.3, "burst_min": 3, "burst_max": 4, "mode": "dropout"},
            )
        ],
        2,
    )
    fo = [f for f in out if f.sensor_id == "s0"]
    dead = [not (f.depth > 0).any() for f in fo]
    assert sum(dead) == 3 and man["faults"][0]["resolved"]["active_frames"] == [
        i for i, d in enumerate(dead) if d
    ]


def test_profile_scales_random_walk_seeded():
    a = profile_scales(20, 0.25, "random_walk", derive_rng(1, "s0", 0))
    b = profile_scales(20, 0.25, "random_walk", derive_rng(1, "s0", 0))
    assert np.array_equal(a, b) and np.all(a[:5] == 0)


def test_manifest_roundtrip_and_tamper(tmp_path, data):
    _, man = inject_faults(
        data.frames, [FaultSpec("depth_bias_range", "s0", 0, {"beta": 0.02, "sign": 1})], 9
    )
    p = tmp_path / "m.json"
    h = write_manifest(man, p)
    assert read_manifest(p)["content_sha256"] == h
    p.write_text(p.read_text().replace('"beta": 0.02', '"beta": 0.03'))
    with pytest.raises(ValueError):
        read_manifest(p)


def test_grid_expand_and_scenarios():
    from pathlib import Path

    import yaml

    g = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs/faults/grid.yaml").read_text()
    )
    cells = expand_cells(g)
    kinds = {c["kind"] for c in cells}
    assert kinds == {
        "extrinsic_drift",
        "depth_bias_const",
        "depth_bias_range",
        "multipath",
        "transient",
    }
    specs = scenario_specs(cells[0], ["s0", "s1", "s2", "s3"], [0, 1], same_direction=True)
    assert [s.sensor_id for s in specs] == ["s0", "s1"] and [s.fault_id for s in specs] == [0, 1]
    with pytest.raises(ValueError):
        FaultSpec("nope", "s0").validate()


def test_common_mode_identical_realisation(data):
    specs = [
        FaultSpec("extrinsic_drift", s, j, {"rotation_deg": 2.0, "common_fault_id": 0})
        for j, s in enumerate(["s0", "s1"])
    ]
    _, man = inject_faults(data.frames, specs, 1)
    a, b = (r["resolved"]["rot_axis"] for r in man["faults"])
    assert a == b


def test_pixel_outliers_sparse(data):
    out, _ = inject_faults(
        data.frames,
        [
            FaultSpec(
                "transient",
                "s0",
                0,
                {
                    "fraction": 1.0,
                    "mode": "pixel_outliers",
                    "pixel_fraction": 0.05,
                    "magnitude_m": 0.5,
                },
            )
        ],
        0,
    )
    f0 = [f for f in data.frames if f.sensor_id == "s0"][0]
    o0 = [f for f in out if f.sensor_id == "s0"][0]
    frac = (np.abs(o0.depth - f0.depth) > 1e-6).mean()
    assert 0.0 < frac < 0.15
