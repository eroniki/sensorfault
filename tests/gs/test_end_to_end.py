"""view_gate with the real ViewGateTest on the numpy reference backend (CPU, tiny scene)."""

import numpy as np

from sensorfault.gs.fusers import ViewGate
from sensorfault.gs.reference import NumpyReferenceBackend
from sensorfault.types import DepthFrame


def _scene(bias_view=None, n=5, hw=16, seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[20.0, 0, hw / 2], [0, 20.0, hw / 2], [0, 0, 1]])
    frames = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = 0.05 * (i - n // 2)
        d = 2.0 + 0.01 * rng.standard_normal((hw, hw))
        if i == bias_view:
            d[: hw // 2, : hw // 2] += 0.15
        frames.append(DepthFrame(f"v{i}", i, d.astype(np.float32), K, T))
    return frames


def test_view_gate_runs_end_to_end_on_reference_backend():
    be = NumpyReferenceBackend(stride=4)
    res = ViewGate(backend=be).fuse(_scene(bias_view=2), {"round_cap": 3, "sigma": 0.01})
    assert set(res.excluded_sensors) == {f"v{i}" for i in range(5)}
    assert "rounds" in res.meta
