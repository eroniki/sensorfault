"""CPU tests of the gate-testbed fusers with a fake backend and fake gate."""

import json

import numpy as np

import sensorfault.baselines  # noqa: F401
from sensorfault.baselines.base import get_fuser, list_fusers
from sensorfault.gs.depth_source import DirectoryDepthSource
from sensorfault.gs.fusers import NoGate, OracleGate, ViewGate
from sensorfault.gs.interfaces import SurfelState
from sensorfault.types import DepthFrame


def _frames(n=5):
    return [
        DepthFrame(f"v{i}", i, np.ones((4, 4), np.float32), np.eye(3), np.eye(4)) for i in range(n)
    ]


class FakeBackend:
    def __init__(self):
        self.calls = []

    def fit(self, frames, weights, init=None):
        self.calls.append(dict(weights))
        k = int(sum(w > 0 for w in weights.values()))
        R = np.tile(np.eye(3), (k, 1, 1))
        return SurfelState(np.zeros((k, 3)), R, np.zeros((k, 3)), np.zeros(k), {"w": dict(weights)})


class FakeGate:
    """v2 is faulty; significant whenever it carries weight in the state."""

    def test(self, state, frames, backend):
        ids = [f.sensor_id for f in frames]
        p = {v: (1e-9 if v == "v2" and state.meta["w"]["v2"] > 0 else 0.5) for v in ids}
        return p, {v: float(np.log(x)) for v, x in p.items()}


def test_no_gate_all_weights_one():
    b = FakeBackend()
    res = NoGate(backend=b).fuse(_frames(), {})
    assert b.calls == [dict.fromkeys(["v0", "v1", "v2", "v3", "v4"], 1.0)]
    assert not any(res.excluded_sensors.values()) and res.points.shape == (5, 3)
    assert res.normals.shape == (5, 3)


def test_oracle_gate_drops_listed_views():
    b = FakeBackend()
    res = OracleGate(backend=b).fuse(_frames(), {"oracle": {"faulty_sensors": ["v1"]}})
    assert b.calls[0]["v1"] == 0.0 and sum(b.calls[0].values()) == 4
    assert res.excluded_sensors["v1"] and not res.excluded_sensors["v0"]
    res2 = OracleGate(backend=FakeBackend()).fuse(_frames(), {"faulty_views": ["v3"]})
    assert res2.excluded_sensors["v3"]


def test_view_gate_excludes_faulty_view():
    b = FakeBackend()
    res = ViewGate(backend=b, gate=FakeGate()).fuse(_frames(), {"outer_loop": {"round_cap": 5}})
    assert [v for v, e in res.excluded_sensors.items() if e] == ["v2"]
    assert res.sensor_scores["v2"] > 10 > res.sensor_scores["v0"]
    assert res.meta["rounds"] == 2 and res.meta["alarm"] is None and res.meta["history"]
    assert b.calls[-1]["v2"] == 0.0 and res.points.shape == (4, 3)


def test_view_gate_single_pass_flat_params_refits_final():
    b = FakeBackend()
    res = ViewGate(backend=b, gate=FakeGate()).fuse(_frames(), {"round_cap": 1})
    assert res.excluded_sensors["v2"] and res.meta["final_refit"] and res.points.shape == (4, 3)


def test_registry_and_availability():
    for n in ("no_gate", "oracle_gate", "view_gate"):
        assert n in list_fusers()
        assert get_fuser(n, backend=FakeBackend()).available() == (True, "")
        ok, why = type(get_fuser(n)).available()
        assert ok or "torch/gsplat" in why
    assert "not implemented" in type(get_fuser("spotless_splats")).available()[1]


def test_directory_depth_source(tmp_path):
    d = tmp_path / "s" / "m"
    d.mkdir(parents=True)
    np.savez(d / "a.npz", depth=np.ones((2, 2)), conf=np.ones((2, 2)))
    np.savez(d / "b.npz", depth=np.ones((2, 2)))
    T = np.eye(4).tolist()
    (tmp_path / "s" / "transforms.json").write_text(
        json.dumps({"K": np.eye(3).tolist(), "views": {"a": {"T_wc": T}, "b": {"T_wc": T}}})
    )
    fr = DirectoryDepthSource(tmp_path).frames("s", "m")
    assert [f.sensor_id for f in fr] == ["a", "b"] and fr[0].conf is not None and fr[1].conf is None
