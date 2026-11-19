"""Torch ray-plane compositor agrees with the numpy reference (skipped without torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sensorfault.gs.fit import _ray_plane_depth  # noqa: E402
from sensorfault.gs.reference import _composite_chunk, init_surfels  # noqa: E402
from sensorfault.types import DepthFrame  # noqa: E402


def test_ray_plane_depth_matches_numpy_reference():
    K = np.array([[30.0, 0, 8], [0, 30.0, 8], [0, 0, 1]])
    T = np.eye(4)
    d = np.full((16, 16), 2.0, np.float32)
    f = DepthFrame("v0", 0, d, K, T)
    st = init_surfels([f], 4)
    u, v = np.meshgrid(np.arange(16) + 0.5, np.arange(16) + 0.5)
    rays = np.stack([(u - 8) / 30, (v - 8) / 30, np.ones_like(u)], -1).reshape(-1, 3)
    ref = _composite_chunk(
        st.mu, st.R, st.log_s, st.logit_alpha, T[:3, 3], T[:3, :3], rays, np.ones(len(st.mu), bool)
    )
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    D, W = _ray_plane_depth(
        t(st.mu), t(st.R), t(st.log_s), t(st.logit_alpha), t(T[:3, 3]), t(T[:3, :3]), t(rays)
    )
    live = np.isfinite(ref["D"])
    np.testing.assert_allclose(D.numpy()[live], ref["D"][live], atol=1e-6)
    np.testing.assert_allclose(W.numpy(), ref["W"], atol=1e-6)
