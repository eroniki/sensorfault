import numpy as np
import pytest
from synth import K, make_frames, pose

from sensorfault.baselines import get_fuser
from sensorfault.baselines import openmvs as om
from sensorfault.baselines import openmvs_dmap as dm
from sensorfault.baselines._colmap import rotmat_to_quat_wxyz, write_colmap_model


def test_dmap_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 6.0, (12, 20)).astype(np.float32)
    d[3, 4] = 0
    n = rng.normal(size=(12, 20, 3)).astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n[3, 4] = 0
    conf = rng.uniform(0, 1, (12, 20)).astype(np.float32)
    R = pose(yaw=0.3)[:3, :3].T
    m = dm.DMap(
        d,
        K,
        R,
        np.array([1.0, 2, 3]),
        n,
        conf,
        ids=[7, 1, 2],
        image_name="a.png",
        dmin=0.5,
        dmax=6.0,
    )
    p = tmp_path / "depth0007.dmap"
    dm.write_dmap(str(p), m)
    assert p.read_bytes()[:2] == b"D2"
    r = dm.read_dmap(str(p))
    assert r.ids == [7, 1, 2] and r.image_name == "a.png"
    assert np.allclose(r.K, K) and np.allclose(r.R, R) and np.allclose(r.C, [1, 2, 3])
    assert r.depth[3, 4] == 0
    ok = d > 0
    assert np.max(np.abs(r.depth[ok] - d[ok]) / d[ok]) < 1 / 1024 + 1e-6
    assert np.all(r.normals[3, 4] == 0)
    dots = np.sum(r.normals[ok] * n[ok], -1)
    assert dots.min() > 0.9999
    assert np.max(np.abs(r.conf - conf)) < 0.5 / 255 * conf.max() + 1e-6


def test_write_dmaps_from_frames(tmp_path):
    fr = make_frames([0, 0], sigma=0.0, poses=[pose(), pose(tx=0.1)])
    paths = om.write_dmaps(fr, tmp_path)
    r = dm.read_dmap(str(paths[1]))
    assert np.allclose(r.C, [0.1, 0, 0]) and r.depth.shape == (64, 64)


def test_ply_reader(tmp_path):
    p = tmp_path / "a.ply"
    arr = np.zeros(3, [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    arr["x"] = [1, 2, 3]
    arr["z"] = [4, 5, 6]
    p.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\n"
        b"property float y\nproperty float z\nend_header\n" + arr.tobytes()
    )
    pts, n = om.read_ply_points(p)
    assert n is None and np.allclose(pts[:, 2], [4, 5, 6])


def test_command_and_availability(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENMVS_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    ok, why = get_fuser("openmvs").available()
    assert not ok and "DensifyPointCloud" in why
    cmd = om.OpenMVS.build_command(
        "D", tmp_path / "s.mvs", tmp_path / "n.txt", tmp_path / "o.ply", om.OpenMVS.DEFAULTS
    )
    for tok in [
        "--geometric-iters",
        "--fusion-filter",
        "--number-views-fuse",
        "--postprocess-dmaps",
        "--view-neighbors-file",
    ]:
        assert tok in cmd
    assert cmd[cmd.index("--geometric-iters") + 1] == "0"
    with pytest.raises(RuntimeError):
        get_fuser("openmvs").fuse(make_frames([0]), {})


def test_mvs_writer_is_stub(tmp_path):
    with pytest.raises(NotImplementedError):
        om.write_mvs_scene([], tmp_path / "x.mvs")


@pytest.mark.parametrize("name", ["senfunet", "routedfusion", "gs_mcmc_depth", "twodgs_depth"])
def test_external_unavailable_and_stub(name, monkeypatch, tmp_path):
    monkeypatch.setenv("SENSORFAULT_THIRD_PARTY", str(tmp_path))
    ok, why = get_fuser(name).available()
    assert not ok and why
    with pytest.raises(NotImplementedError):
        get_fuser(name).fuse(make_frames([0]), {})


def test_colmap_writer(tmp_path):
    fr = make_frames([0, 0], sigma=0.0, poses=[pose(), pose(tx=0.5, yaw=0.2)])
    write_colmap_model(fr, tmp_path)
    cams = (tmp_path / "sparse/0/cameras.txt").read_text().splitlines()
    assert cams[1].startswith("1 PINHOLE 64 64 60")
    line = [
        l for l in (tmp_path / "sparse/0/images.txt").read_text().splitlines() if l.startswith("2 ")
    ][0]
    q = np.array(line.split()[1:5], float)
    t = np.array(line.split()[5:8], float)
    assert abs(np.linalg.norm(q) - 1) < 1e-9
    R_cw = pose(yaw=0.2)[:3, :3].T
    assert np.allclose(rotmat_to_quat_wxyz(R_cw), q)
    assert np.allclose(t, -R_cw @ np.array([0.5, 0, 0]))
    assert (tmp_path / "images/img0001.png").read_bytes()[:4] == b"\x89PNG"
