"""Metric3D v2 and UniDepthV2 wrappers: CPU-only, offline, backend mocked (no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from sensorfault.depthcache import metric3d_v2 as m3d
from sensorfault.depthcache import unidepth_v2 as ud


def _rgb(h=120, w=200):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_fit_geometry_and_preprocess():
    s, (nh, nw), pad = m3d.fit_geometry(120, 200)
    assert s == pytest.approx(min(616 / 120, 1064 / 200))
    assert nh + pad[0] + pad[1] == 616 and nw + pad[2] + pad[3] == 1064
    x, s2, pad2 = m3d.preprocess(_rgb())
    assert x.shape == (3, 616, 1064) and x.dtype == np.float32 and s2 == s and pad2 == pad


def test_decanonicalise_math():
    d = np.full((2, 2), 10.0, np.float32)
    assert np.allclose(m3d.decanonicalise(d, 500.0), 5.0)  # f/1000 scaling
    assert np.allclose(m3d.decanonicalise(d, 1000.0), 10.0)
    assert m3d.decanonicalise(d * 1e3, 1000.0).max() == m3d.MAX_DEPTH_M
    assert m3d.decanonicalise(d * 1e3, 1000.0, max_depth_m=1000.0).max() == 1000.0


def test_metric3d_predict_rescales_with_focal(monkeypatch):
    est = m3d.build()
    seen = {}

    def fake_run(x):
        seen["shape"] = x.shape
        return np.full((616, 1064), 20.0, np.float32), np.full((616, 1064), 0.7, np.float32)

    monkeypatch.setattr(est, "_run", fake_run)
    rgb = _rgb(120, 200)
    r = est.predict(rgb, focal_px=250.0)
    s = min(616 / 120, 1064 / 200)
    assert seen["shape"] == (3, 616, 1064)
    assert r.depth.shape == (120, 200) and r.depth.dtype == np.float32
    assert np.allclose(r.depth, 20.0 * s * 250.0 / 1000.0, rtol=1e-4)
    assert r.confidence.shape == (120, 200) and r.confidence.dtype == np.float32
    assert np.allclose(r.confidence, 0.7, atol=1e-5)
    assert r.focal_px == 250.0 and r.kind == "metric_depth" and est.has_confidence
    r2 = est.predict(rgb, focal_px=500.0)
    assert np.allclose(r2.depth, 2 * r.depth)  # linear in focal


def test_metric3d_requires_focal_or_assumed(monkeypatch):
    est = m3d.build()
    monkeypatch.setattr(est, "_run", lambda x: (np.ones((616, 1064), np.float32), None))
    with pytest.raises(ValueError):
        est.predict(_rgb())
    with pytest.raises(ValueError):
        est.predict(_rgb(), focal_px=-1.0)
    est2 = m3d.build(assumed_focal_px=1000.0)
    monkeypatch.setattr(est2, "_run", lambda x: (np.ones((616, 1064), np.float32), None))
    r = est2.predict(_rgb())
    assert r.extra["focal_source"] == "assumed" and r.confidence is None


def test_unidepth_given_and_predicted_focal(monkeypatch):
    est = ud.build()
    calls = []

    def fake_run(rgb, K):
        calls.append(K)
        h, w = rgb.shape[:2]
        k = (
            np.array([[333.0, 0, w / 2], [0, 334.0, h / 2], [0, 0, 1]], np.float32)
            if K is None
            else K
        )
        return np.full((h, w), 3.0, np.float32), np.full((h, w), 0.5, np.float32), k

    monkeypatch.setattr(est, "_run", fake_run)
    rgb = _rgb(48, 64)
    r = est.predict(rgb, focal_px=100.0)
    assert np.allclose(calls[0], [[100, 0, 32], [0, 100, 24], [0, 0, 1]])
    assert r.depth.shape == (48, 64) and r.depth.dtype == np.float32
    assert r.confidence.shape == (48, 64) and r.focal_px == 100.0
    assert r.extra["focal_source"] == "given"
    r = est.predict(rgb)
    assert calls[1] is None and r.focal_px == 333.0 and r.extra["focal_source"] == "predicted"
    est2 = ud.build(use_known_focal=False)
    monkeypatch.setattr(est2, "_run", fake_run)
    r = est2.predict(rgb, focal_px=100.0)
    assert calls[2] is None and r.focal_px == 333.0


def test_resolution_mismatch_is_resampled(monkeypatch):
    est = ud.build()
    monkeypatch.setattr(
        est,
        "_run",
        lambda rgb, K: (
            np.ones((24, 32), np.float32),
            np.ones((24, 32), np.float32),
            np.eye(3, dtype=np.float32),
        ),
    )
    assert est.predict(_rgb(48, 64)).depth.shape == (48, 64)


@pytest.mark.parametrize("mod", [m3d, ud])
def test_info_and_validation(mod):
    est = mod.build()
    info = est.info()
    assert info["weights_licence"] and info["model_id"] == est.model_id
    assert info["confidence_semantics"]
    with pytest.raises(ValueError):
        est.predict(np.zeros((4, 4, 3), np.float32), focal_px=10.0)
    with pytest.raises(ValueError):
        mod.build(variant="nope")
