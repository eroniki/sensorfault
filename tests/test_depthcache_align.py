import numpy as np
import pytest

from sensorfault.depthcache.align import AlignmentError, align_depth, load_depth_aligned
from sensorfault.depthcache.base import DepthResult
from sensorfault.depthcache.cache import ModelCache


def _gt(shape=(60, 80), seed=0):
    return (50 + 300 * np.random.default_rng(seed).random(shape)).astype(np.float32)


def _sparse(gt, frac, seed=1):
    ref = np.full_like(gt, np.nan)
    idx = np.random.default_rng(seed).random(gt.shape) < frac
    ref[idx] = gt[idx]
    return ref, idx


def _corrupt(ref, idx, frac, value=5.0, seed=2):
    out = ref.copy()
    pos = np.flatnonzero(idx)
    bad = np.random.default_rng(seed).choice(pos, int(frac * pos.size), replace=False)
    out.flat[bad] = value
    return out


def test_scale_recovered_with_sparse_reference_and_20pct_outliers():
    gt = _gt()
    res = DepthResult(depth=gt * 0.013, kind="metric_depth")
    ref, idx = _sparse(gt, 0.1)
    out, al = align_depth(res, _corrupt(ref, idx, 0.2))
    assert al.mode == "scale" and al.scale == pytest.approx(1 / 0.013, rel=0.02)
    np.testing.assert_allclose(out.depth, gt, rtol=0.03)
    assert out.kind == "metric_depth" and out.extra["aligned"] is True
    assert out.confidence is None


@pytest.mark.parametrize("frac", [0.05, 0.15, 0.30, 0.40])
def test_affine_recovered_with_noise_and_outliers(frac):
    gt = _gt()
    rng = np.random.default_rng(3)
    raw = (0.7 / gt + 0.002) * (1 + 0.005 * rng.standard_normal(gt.shape))
    ref, idx = _sparse(gt, 0.2)
    out, al = align_depth(
        DepthResult(depth=raw.astype(np.float32), kind="relative_inverse_depth"),
        _corrupt(ref, idx, frac),
    )
    assert al.mode == "affine_inverse" and al.a == pytest.approx(1 / 0.7, rel=0.05)
    assert np.nanmedian(np.abs(out.depth - gt) / gt) < 0.02
    assert al.n_inliers < al.n_valid or frac == 0


def test_degenerate_cases_raise():
    gt = _gt()
    ref, _ = _sparse(gt, 0.2)
    rel = "relative_inverse_depth"
    with pytest.raises(AlignmentError, match="constant"):
        align_depth(DepthResult(depth=np.full_like(gt, 0.01), kind=rel), ref)
    with pytest.raises(AlignmentError, match=r"non-positive|no consistent"):
        align_depth(DepthResult(depth=-0.7 / gt + 0.05, kind=rel), ref)
    two = np.full_like(gt, np.nan)
    two[0, :2] = gt[0, :2]
    with pytest.raises(AlignmentError, match="valid reference pixels"):
        align_depth(DepthResult(depth=0.7 / gt, kind=rel), two)
    with pytest.raises(AlignmentError, match="valid reference pixels"):
        align_depth(DepthResult(depth=gt, kind="metric_depth"), two)
    with pytest.raises(ValueError, match="shape"):
        align_depth(DepthResult(depth=gt, kind="metric_depth"), gt[:5])
    with pytest.raises(AlignmentError):
        align_depth(DepthResult(depth=gt, kind="metric_depth"), np.full_like(gt, np.nan))
    with pytest.raises(ValueError, match="unknown depth kind"):
        align_depth(DepthResult(depth=gt, kind="nope"), gt)


def test_max_depth_clip_and_clamp_saturation_excluded():
    gt = _gt()
    ref = gt.copy()
    pred = np.minimum(gt * 0.5, 100.0)  # Metric3D-style clamp at 100 m
    out, al = align_depth(DepthResult(depth=pred, kind="metric_depth"), ref, clamp_m=100.0)
    assert al.scale == pytest.approx(2.0, rel=1e-3)  # saturated pixels would have biased it
    sat = gt * 0.5 >= 100.0
    assert np.isnan(out.depth[sat]).all() and np.isfinite(out.depth[~sat]).all()
    out2, _ = align_depth(DepthResult(depth=gt * 0.5, kind="metric_depth"), ref, max_depth=200.0)
    assert np.nanmax(out2.depth) == pytest.approx(200.0)


def test_load_depth_aligned_roundtrip_float16_uses_manifest_clamp(tmp_path):
    gt = _gt()
    pred = np.minimum(gt * 0.15, 40.0).astype(np.float32)
    c = ModelCache(tmp_path, "d", "s", "m", "float16")
    c.start({"max_depth_m": 40.0}, "cpu")
    c.put("v0", DepthResult(depth=pred, kind="metric_depth"), 0.1)
    c.flush()
    out, al = load_depth_aligned(tmp_path, "d", "s", "m", "v0", gt)
    assert al.scale == pytest.approx(1 / 0.15, rel=5e-3)
    assert np.isnan(out.depth[gt * 0.15 >= 40.0]).all()
    assert al.n_valid < gt.size
