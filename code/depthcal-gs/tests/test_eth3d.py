"""ETH3D interface: parsers, camera models, depth re-sampling, manifests. A real-data test runs if ETH3D_PIPES is set."""
import os
from pathlib import Path

import numpy as np
import pytest

from dcal.eth3d import (Camera, cam_from_img, img_from_cam, parse_cameras, parse_images, qvec_to_R, read_depth,
                        undistort_depth)
from dcal.manifest import build_manifest, verify_manifest

FISHEYE = Camera(0, "THIN_PRISM_FISHEYE", 6048, 4032,
                 (3430.27, 3429.23, 3032.95, 2003.59, 0.218983, 0.155576, -0.000274629, -0.000387159, -0.0364669, 0.302935,
                  0.00133613, 0.00101227))


def test_parse_documentation_example_lines(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n0 THIN_PRISM_FISHEYE 6048 4032 3408.35 3408.8 3033.92 2019.32 0.21167 0.20864 0.00053 -0.00015 "
        "-0.16568 0.4075 0.00048 0.00028\n")
    (tmp_path / "images.txt").write_text(
        "# Image list\n49 0.995 0.00592 -0.0882 0.0311 3.21 3.60 9.19 1 dslr_images/DSC_0323.JPG\n"
        "64.3 1948.0 60 237.1 897.6 -1\n"
        "50 1 0 0 0 0 0 0 1 dslr_images/DSC_0324.JPG\n\n")      # empty keypoint line
    cams, imgs = parse_cameras(tmp_path / "cameras.txt"), parse_images(tmp_path / "images.txt")
    assert cams[0].model == "THIN_PRISM_FISHEYE" and len(cams[0].params) == 12
    assert [i.name for i in imgs] == ["dslr_images/DSC_0323.JPG", "dslr_images/DSC_0324.JPG"]
    assert imgs[0].cam_id == 1 and imgs[0].t == (3.21, 3.60, 9.19)


def test_quaternion_is_hamilton_and_orthonormal():
    assert np.allclose(qvec_to_R((1, 0, 0, 0)), np.eye(3))
    c = np.sqrt(0.5)
    R = qvec_to_R((c, 0, 0, c))                       # +90 degrees about z maps x to y
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0]) and np.isclose(np.linalg.det(R), 1)
    R2 = qvec_to_R((0.995, 0.00592, -0.0882, 0.0311))
    assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-6)


def test_fisheye_projection_roundtrip():
    rng = np.random.default_rng(0)
    xy = np.stack([rng.uniform(100, FISHEYE.width - 100, 500), rng.uniform(100, FISHEYE.height - 100, 500)], 1)
    uv = cam_from_img(FISHEYE, xy)
    back, ok = img_from_cam(FISHEYE, np.concatenate([uv, np.ones((len(uv), 1))], 1))
    assert ok.all() and np.abs(back - xy).max() < 1e-3


def _synthetic_cameras():
    src = Camera(0, "THIN_PRISM_FISHEYE", 240, 160, (140.0, 140.0, 120.5, 80.5, 0.12, 0.05, 1e-4, -1e-4, -0.01, 0.02,
                                                    5e-4, 4e-4))
    dst = Camera(0, "PINHOLE", 250, 168, (140.0, 140.0, 125.0, 84.0))
    return src, dst


@pytest.mark.parametrize("kind", ["z", "range"])
def test_undistort_depth_recovers_a_tilted_plane(kind):
    """Plane z = a / (1 - b u): analytic z-depth per ray; the stored dump is z or range."""
    src, dst = _synthetic_cameras()
    a, b = 3.0, 0.3
    jj, ii = np.mgrid[0:src.height, 0:src.width]
    uv = cam_from_img(src, np.stack([ii.ravel() + 0.5, jj.ravel() + 0.5], 1))
    z = a / (1 - b * uv[:, 0])
    dump = z * (np.sqrt(1 + (uv ** 2).sum(1)) if kind == "range" else 1.0)
    out, K = undistort_depth(dump.reshape(src.height, src.width).astype(np.float32), src, dst, scale=1.0, depth_kind=kind)
    h, w = out.shape
    uu, vv = np.meshgrid((np.arange(w) + 0.5 - dst.params[2]) / dst.params[0], (np.arange(h) + 0.5 - dst.params[3]) / dst.params[1])
    expect = a / (1 - b * uu)
    ok = np.isfinite(out)
    assert ok.mean() > 0.5
    rel = np.abs(out[ok] - expect[ok]) / expect[ok]
    assert np.median(rel) < 2e-3 and np.percentile(rel, 99) < 2e-2
    assert np.allclose(K, dst.K)


def test_undistort_depth_scale_changes_resolution_and_intrinsics():
    src, dst = _synthetic_cameras()
    out, K = undistort_depth(np.full((src.height, src.width), 2.0, np.float32), src, dst, scale=2.0)
    assert out.shape == (84, 125) and np.allclose(K[0, 0], dst.params[0] / 2.0)
    assert np.nanmax(np.abs(out - 2.0)) < 1e-6


def test_read_depth_maps_inf_to_nan_and_checks_size(tmp_path):
    d = np.array([[1.0, np.inf], [2.0, 3.0]], "<f4")
    d.tofile(tmp_path / "x.JPG")
    r = read_depth(tmp_path / "x.JPG", 2, 2)
    assert np.isnan(r[0, 1]) and r[1, 0] == 2.0
    with pytest.raises(ValueError):
        read_depth(tmp_path / "x.JPG", 3, 2)


def test_manifest_roundtrip_detects_changes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    m = build_manifest(tmp_path, ["a.bin"], {"source": "test"})
    assert verify_manifest(tmp_path, m) == []
    (tmp_path / "a.bin").write_bytes(b"abd")
    assert verify_manifest(tmp_path, m) == ["changed: a.bin"]


@pytest.mark.skipif(not os.environ.get("ETH3D_PIPES"), reason="set ETH3D_PIPES=/path/to/eth3d/pipes/pipes to run on real data")
def test_real_scene_depth_matches_laser_scan_after_undistortion():
    from dcal.eth3d import load_scene
    base = Path(os.environ["ETH3D_PIPES"])
    from scripts_helpers import project_scan_depth  # defined below via conftest-free import
    views = load_scene(base.parent, base.name, scale=8.0, names=["DSC_0634.JPG"])
    v = views[0]
    assert v.rgb.shape[:2] == v.gt_depth.shape and np.isfinite(v.gt_depth).mean() > 0.05
    ref = project_scan_depth(base, v)
    both = np.isfinite(v.gt_depth) & np.isfinite(ref)
    assert both.sum() > 1000
    rel = np.abs(v.gt_depth[both] - ref[both]) / ref[both]
    assert np.median(rel) < 5e-3
