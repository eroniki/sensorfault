import numpy as np
import pytest
from synth import K, make_frames, pose

from sensorfault.baselines import get_fuser
from sensorfault.baselines._common import depth_to_normals
from sensorfault.baselines.tvl1_range import prox_l1_sum

POSES = [pose(tx=-0.1), pose(tx=0.0), pose(tx=0.1, yaw=0.0)]


def test_normals_fronto_parallel():
    n = depth_to_normals(np.full((16, 16), 2.0), K)
    assert np.allclose(n[5:-5, 5:-5], [0, 0, -1], atol=1e-6)


def test_consistency_voting_rejects_offset_sensor():
    fr = make_frames([0, 0, 0.3], sigma=0.0, poses=POSES)
    r = get_fuser("consistency_voting").fuse(fr, {"min_consistent": 1})
    assert (
        r.sensor_scores["s2"] > 0.9 and r.sensor_scores["s0"] < 0.3 and r.sensor_scores["s1"] < 0.3
    )
    assert len(r.points) > 0
    # kept points lie on the true plane z=2 (world)
    assert np.all(np.abs(r.points[:, 2] - 2.0) < 0.02)


def test_consistency_voting_min_views():
    fr = make_frames([0, 0, 0.3], sigma=0.0, poses=POSES)
    r1 = get_fuser("consistency_voting").fuse(fr, {"min_consistent": 1})
    r2 = get_fuser("consistency_voting").fuse(fr, {"min_consistent": 2})
    assert len(r2.points) < len(r1.points)


def test_merrell_conflicts_flag_outlier():
    fr = make_frames([0, 0, 0.3], sigma=0.0, poses=POSES)
    r = get_fuser("merrell_visibility").fuse(fr, {})
    assert r.sensor_scores["s2"] > r.sensor_scores["s0"] + 0.2
    assert len(r.points) > 0 and np.all(np.abs(r.points[:, 2] - 2.0) < 0.02)


def test_merrell_all_consistent_keeps_most():
    fr = make_frames([0, 0, 0], sigma=0.0, poses=POSES)
    r = get_fuser("merrell_visibility").fuse(fr, {})
    assert len(r.points) > 0.7 * 3 * 64 * 64


def test_tvl1_prox_matches_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = rng.normal(size=(5, 1))
        v = rng.normal(size=(1,))
        th = 0.3
        u = prox_l1_sum(v, f, th)[0]
        xs = np.linspace(-4, 4, 80001)
        cost = 0.5 * (xs - v[0]) ** 2 + th * np.abs(xs[:, None] - f[:, 0]).sum(1)
        assert abs(u - xs[np.argmin(cost)]) < 2e-3


def test_tvl1_surface_small_grid():
    fr = make_frames([0, 0, 0.1], sigma=0.0, poses=POSES)
    bounds = ((-0.3, -0.3, 1.8), (0.3, 0.3, 2.2))
    r = get_fuser("tvl1_range").fuse(fr, dict(voxel=0.02, trunc=0.06, iters=150, bounds=bounds))
    assert len(r.points) > 200
    assert abs(np.median(r.points[:, 2]) - 2.0) < 0.02  # median of 3 views ignores the +0.1 outlier
    assert r.sensor_scores["s2"] > r.sensor_scores["s0"]


def test_tsdf_open3d():
    pytest.importorskip("open3d")
    fr = make_frames([0, 0], sigma=0.0, poses=POSES[:2])
    r = get_fuser("tsdf_open3d").fuse(fr, dict(voxel_length=0.02))
    assert len(r.points) > 100
