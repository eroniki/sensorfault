"""CPU-only, offline tests for the Depth Pro and DAv2 wrappers (backend mocked)."""

from __future__ import annotations

import numpy as np
import pytest

from sensorfault.depthcache import depth_anything_v2 as dav2
from sensorfault.depthcache import depth_pro as dp


@pytest.fixture
def rgb():
    return np.zeros((12, 20, 3), dtype=np.uint8)


def test_depth_pro_resize_and_predicted_focal(monkeypatch, rgb):
    est = dp.build()
    monkeypatch.setattr(est, "_infer", lambda x: (np.full((6, 10), 5.0, np.float32), 800.0))
    r = est.predict(rgb)
    assert r.depth.shape == (12, 20) and r.depth.dtype == np.float32
    assert r.kind == "metric_depth" and r.confidence is None
    assert r.focal_px == 800.0 and np.allclose(r.depth, 5.0)


def test_depth_pro_given_focal_rescales(monkeypatch, rgb):
    est = dp.build()
    monkeypatch.setattr(est, "_infer", lambda x: (np.full((12, 20), 5.0, np.float32), 800.0))
    r = est.predict(rgb, focal_px=1600.0)
    assert np.allclose(r.depth, 10.0) and r.focal_px == 1600.0
    assert r.extra["focal_source"] == "given_rescaled"
    with pytest.raises(ValueError):
        est.predict(rgb, focal_px=-1.0)


def test_dav2_relative_unmodified_and_resized(monkeypatch, rgb):
    est = dav2.build()
    monkeypatch.setattr(est, "_infer", lambda x: np.full((7, 7), 3.0, np.float32))
    r = est.predict(rgb, focal_px=500.0)
    assert r.depth.shape == (12, 20) and r.depth.dtype == np.float32
    assert r.kind == "relative_inverse_depth" and r.focal_px is None
    assert np.allclose(r.depth, 3.0) and r.extra["focal_ignored_px"] == 500.0


def test_info_and_licences():
    assert "CC-BY-NC" in dav2.build().info()["weights_licence"]
    small = dav2.build("depth-anything/Depth-Anything-V2-Small-hf")
    assert small.info()["weights_licence"] == "Apache-2.0"
    assert dav2.build("x/Depth-Anything-V2-Metric-Outdoor-Large-hf").kind == "metric_depth"
    assert dp.build().info()["checkpoint"] == "apple/DepthPro-hf"
    assert dp.build("/local/dir").info()["checkpoint"] == "/local/dir"


def test_bad_input(rgb):
    with pytest.raises(ValueError):
        dp.build().predict(rgb.astype(np.float32))
