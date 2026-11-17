"""Depth cache: layout, atomic/resumable writes, runner with a fake estimator. CPU, offline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sensorfault.depthcache import cache as dc
from sensorfault.depthcache.base import DepthResult
from sensorfault.depthcache.datasets import DatasetUnavailableError, View, enumerate_views
from sensorfault.depthcache.run import run


class FakeEstimator:
    kind = "metric_depth"
    has_confidence = True

    def __init__(self, model_id="depth_pro", fail_on=()):
        self.model_id = model_id
        self.fail_on = set(fail_on)
        self.calls = []
        self.loaded = None

    def load(self, device="cuda", **kw):
        self.loaded = device

    def predict(self, rgb, focal_px=None):
        self.calls.append(float(rgb[0, 0, 0]))
        if float(rgb[0, 0, 0]) in self.fail_on:
            raise RuntimeError("boom")
        d = np.full(rgb.shape[:2], 2.5, np.float32)
        return DepthResult(d, self.kind, confidence=d * 0 + 0.5, focal_px=focal_px)

    def info(self):
        return {"checkpoint": "fake", "license": "none"}


def make_views(n=4):
    return [
        View(f"{i:04d}", (lambda i=i: np.full((3, 5, 3), i, np.uint8)), 100.0 + i) for i in range(n)
    ]


def test_roundtrip_and_layout(tmp_path):
    r = DepthResult(np.full((2, 3), 1.5, np.float32), "metric_depth", np.ones((2, 3)), 12.5)
    p = dc.save_depth(tmp_path, "ds", "sc", "depth_pro", "0001", r, "float16")
    assert p == tmp_path / "ds" / "sc" / "depth_pro" / "0001.npz"
    out = dc.load_depth(tmp_path, "ds", "sc", "depth_pro", "0001")
    assert out.depth.dtype == np.float32 and out.kind == "metric_depth"
    assert out.focal_px == 12.5 and np.allclose(out.depth, 1.5)
    assert not list(p.parent.glob("*.tmp"))


def test_optional_fields_absent(tmp_path):
    dc.save_depth(
        tmp_path,
        "ds",
        "sc",
        "m",
        "v",
        DepthResult(np.ones((2, 2)), "relative_inverse_depth"),
        "float32",
    )
    out = dc.load_depth(tmp_path, "ds", "sc", "m", "v")
    assert out.confidence is None and out.focal_px is None and out.kind == "relative_inverse_depth"


def test_bad_names_and_missing(tmp_path):
    with pytest.raises(ValueError):
        dc.view_path(tmp_path, "ds", "../x", "m", "v")
    with pytest.raises(FileNotFoundError):
        dc.load_depth(tmp_path, "ds", "sc", "m", "nope")


def test_run_resumes_and_writes_manifest(tmp_path):
    views = make_views()
    est = FakeEstimator()
    out = run(
        "ds",
        "sc",
        ["depth_pro"],
        tmp_path,
        "cpu",
        views=views,
        builder=lambda m: est,
        log=lambda s: None,
    )
    assert out[0]["ok"] == 4 and est.loaded == "cpu"
    man = json.loads((tmp_path / "ds/sc/depth_pro/manifest.json").read_text())
    assert man["model_info"]["checkpoint"] == "fake" and man["environment"]["device"] == "cpu"
    assert "git_sha" in man["environment"] and set(man["views"]) == {f"{i:04d}" for i in range(4)}
    est2 = FakeEstimator()
    out = run(
        "ds",
        "sc",
        ["depth_pro"],
        tmp_path,
        "cpu",
        views=views,
        builder=lambda m: est2,
        log=lambda s: None,
    )
    assert est2.calls == [] and out[0]["skipped"] == 4


def test_failed_views_are_retried_and_limit(tmp_path):
    views = make_views()
    est = FakeEstimator(fail_on={1.0})
    out = run(
        "ds",
        "sc",
        ["depth_pro"],
        tmp_path,
        "cpu",
        views=views,
        builder=lambda m: est,
        log=lambda s: None,
    )
    assert out[0]["ok"] == 3 and out[0]["failed"] == 1
    est2 = FakeEstimator()
    run(
        "ds",
        "sc",
        ["depth_pro"],
        tmp_path,
        "cpu",
        views=views,
        builder=lambda m: est2,
        log=lambda s: None,
    )
    assert est2.calls == [1.0]
    lim = FakeEstimator("metric3d_v2")
    run(
        "ds",
        "sc",
        ["metric3d_v2"],
        tmp_path,
        "cpu",
        limit=2,
        views=views,
        builder=lambda m: lim,
        log=lambda s: None,
    )
    assert len(lim.calls) == 2


def test_deleted_file_is_recomputed(tmp_path):
    views = make_views(2)
    run(
        "ds",
        "sc",
        ["depth_pro"],
        tmp_path,
        "cpu",
        views=views,
        builder=lambda m: FakeEstimator(),
        log=lambda s: None,
    )
    (tmp_path / "ds/sc/depth_pro/0000.npz").unlink()
    est = FakeEstimator()
    run(
        "ds",
        "sc",
        ["depth_pro"],
        tmp_path,
        "cpu",
        views=views,
        builder=lambda m: est,
        log=lambda s: None,
    )
    assert est.calls == [0.0]


def test_dry_run_loads_nothing(tmp_path):
    def boom(m):
        raise AssertionError("builder must not be called")

    out = run(
        "ds",
        "sc",
        ["depth_pro", "unidepth_v2"],
        tmp_path,
        "cpu",
        dry_run=True,
        views=make_views(),
        builder=boom,
        log=lambda s: None,
    )
    assert [o["views"] for o in out] == [4, 4] and not (tmp_path / "ds").exists()


def test_unknown_model_and_missing_wrapper(tmp_path):
    with pytest.raises(ValueError):
        run("ds", "sc", ["nope"], tmp_path, views=make_views())
    from sensorfault.depthcache.run import default_builder

    est = default_builder("unidepth_v2")
    est.checkpoint = str(tmp_path / "no-such-checkpoint")  # never reach the hub from this test
    with pytest.raises((RuntimeError, ImportError, AttributeError, OSError, ValueError)):
        est.load(device="cpu")  # wrapper deps absent or weights absent


def test_stub_datasets_raise_clearly():
    for name in ("usegeo", "eth3d"):
        with pytest.raises(DatasetUnavailableError, match="approval"):
            enumerate_views(name, "x")
    with pytest.raises(ValueError):
        enumerate_views("nope", "x")


def test_matrixcity_views_from_synthetic_block(tmp_path):
    import math

    block = tmp_path / "matrixcity" / "b"
    (block / "images").mkdir(parents=True)
    fov = 1.0
    frames = [{"frame_index": 7, "file_path": "images/7"}, {"frame_index": 8}]
    (block / "transforms.json").write_text(json.dumps({"camera_angle_x": fov, "frames": frames}))
    views = enumerate_views("matrixcity", "b", data_root=tmp_path)
    assert [v.view_id for v in views] == ["0007", "0008"]
    assert views[0].focal_px == pytest.approx(0.5 * 1920 / math.tan(0.5))
    with pytest.raises(DatasetUnavailableError):
        enumerate_views("matrixcity", "missing", data_root=tmp_path)
    pytest.importorskip("PIL")
    from PIL import Image

    Image.fromarray(np.full((4, 6, 3), 9, np.uint8)).save(block / "images" / "7.png")
    assert views[0].load_rgb().shape == (4, 6, 3)


def test_manifest_keeps_result_extra(tmp_path):
    cache = dc.ModelCache(tmp_path, "ds", "sc", "depth_pro", "float32")
    r = DepthResult(
        np.ones((2, 2), np.float32),
        "metric_depth",
        focal_px=10.0,
        extra={"focal_source": "given_rescaled", "focal_pred_px": np.float32(8.0)},
    )
    cache.put("0001", r, 0.1)
    cache.flush()
    rec = dc.ModelCache(tmp_path, "ds", "sc", "depth_pro").manifest["views"]["0001"]
    assert rec["extra"] == {"focal_source": "given_rescaled", "focal_pred_px": 8.0}
