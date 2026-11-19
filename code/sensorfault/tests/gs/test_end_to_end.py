"""view_gate with the real ViewGateTest on the numpy reference backend (CPU, tiny scene)."""

import numpy as np

from sensorfault.gs.fusers import ViewGate
from sensorfault.gs.reference import NumpyReferenceBackend
from sensorfault.types import DepthFrame


def _scene(
    bias_view=None, n=6, hw=32, f=40.0, noise=0.005, bias=0.05, seed=0, region=(0, 16, 0, 16)
):
    """Tilted plane seen from n cameras; optional bias on a (possibly off-grid) pixel region."""
    rng = np.random.default_rng(seed)
    K = np.array([[f, 0, hw / 2], [0, f, hw / 2], [0, 0, 1.0]])
    nrm = np.array([0.15, -0.1, 1.0])
    nrm /= np.linalg.norm(nrm)
    frames = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = 0.12 * (i - (n - 1) / 2)
        T[1, 3] = 0.03 * ((i % 2) - 0.5)
        u, v = np.meshgrid(np.arange(hw) + 0.5, np.arange(hw) + 0.5)
        rays = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u)], -1)
        d = ((np.array([0, 0, 2.0]) - T[:3, 3]) @ nrm) / (rays @ nrm)
        d = d + noise * rng.standard_normal(d.shape)
        if i == bias_view:
            r0, r1, c0, c1 = region
            d[r0:r1, c0:c1] += bias
        frames.append(DepthFrame(f"v{i}", i, d.astype(np.float32), K, T))
    return frames


def _fuse(frames):
    return ViewGate(backend=NumpyReferenceBackend(stride=4)).fuse(
        frames, {"round_cap": 3, "sigma": 0.01}
    )


def _excluded(res):
    return {v for v, flag in res.excluded_sensors.items() if flag}


def test_view_gate_excludes_only_the_biased_view():
    res = _fuse(_scene(bias_view=3))
    assert _excluded(res) == {"v3"}
    assert "rounds" in res.meta


def test_view_gate_excludes_off_grid_fault():
    res = _fuse(_scene(bias_view=3, region=(5, 17, 7, 19)))
    assert "v3" in _excluded(res)


def test_view_gate_clean_scene_excludes_nothing():
    assert _excluded(_fuse(_scene())) == set()
