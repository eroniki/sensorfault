import numpy as np
import pytest

from sensorfault.depthcache.align import align_depth
from sensorfault.depthcache.base import DepthResult


def _gt(shape=(40, 60)):
    rng = np.random.default_rng(0)
    return (50 + 300 * rng.random(shape)).astype(np.float32)


def test_metric_scale_recovered_with_sparse_reference_and_outliers():
    gt = _gt()
    res = DepthResult(depth=gt * 0.013, kind="metric_depth")
    ref = np.full_like(gt, np.nan)
    idx = np.random.default_rng(1).random(gt.shape) < 0.02
    ref[idx] = gt[idx]
    out, al = align_depth(res, ref)
    assert al.mode == "scale" and al.scale == pytest.approx(1 / 0.013, rel=1e-3)
    np.testing.assert_allclose(out.depth, gt, rtol=1e-3)
    assert out.kind == "metric_depth" and out.extra["aligned"] is True


def test_relative_inverse_depth_affine_recovered():
    gt = _gt()
    raw = 0.7 / gt + 0.002
    ref = gt.copy()
    ref[::2] = np.nan
    out, al = align_depth(
        DepthResult(depth=raw.astype(np.float32), kind="relative_inverse_depth"), ref
    )
    assert al.mode == "affine_inverse"
    np.testing.assert_allclose(out.depth, gt, rtol=1e-3)
    assert al.rel_resid_median < 1e-3


def test_errors():
    gt = _gt()
    with pytest.raises(ValueError, match="shape"):
        align_depth(DepthResult(depth=gt, kind="metric_depth"), gt[:5])
    with pytest.raises(ValueError, match="no valid"):
        align_depth(DepthResult(depth=gt, kind="metric_depth"), np.full_like(gt, np.nan))
    with pytest.raises(ValueError, match="unknown depth kind"):
        align_depth(DepthResult(depth=gt, kind="nope"), gt)
