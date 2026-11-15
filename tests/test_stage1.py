"""Stage-1 pipeline on tiny random models and synthetic data (CPU, deterministic): data, loss, checkpoint/resume, NaN guard."""
import json

import numpy as np
import pytest
import torch

from dcal.train import data as D
from dcal.train import loop as L

CFG = dict(tiny=True, res=64, micro_batch=2, accum=2, total_steps=6, lr=2e-3, lr_conv=1e-3, warmup=2, workers=0, seed=3, log_every=1,
           ckpt_every_steps=3, ckpt_every_min=1e9, val_samples=4, eval_every=6, val_sample_images=2, val_ddim_steps=3, fp32=True,
           datasets=[{"type": "synthetic", "n": 16, "size": 96}], grad_ckpt=False, rank=4, keep_last=5)


def test_pfm_roundtrip_and_flip(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    D.write_pfm(tmp_path / "x.pfm", a)
    assert np.array_equal(D.read_pfm(tmp_path / "x.pfm"), a)


def test_affine_invariant_target_ignores_scale_and_shift_and_hits_the_range():
    rng = np.random.RandomState(0)
    d = rng.uniform(1, 5, (50, 60)).astype(np.float32)
    valid = np.ones_like(d, bool)
    t1 = D.affine_invariant_target(d, valid)
    t2 = D.affine_invariant_target(3.7 * d + 11.0, valid)
    assert np.allclose(t1, t2, atol=1e-4)
    assert t1.min() >= -1 and t1.max() <= 1 and abs(np.percentile(t1, 2) + 1) < 1e-3 and abs(np.percentile(t1, 98) - 1) < 1e-3
    assert D.affine_invariant_target(np.full_like(d, 2.0), valid) is None


def test_fill_invalid_uses_the_nearest_valid_value():
    t = np.array([[0.0, 0.5, 0.9, 0.0]], np.float32)
    valid = np.array([[False, True, True, False]])
    f = D.fill_invalid(t, valid)
    assert np.isclose(f[0, 0], 0.5) and np.isclose(f[0, 3], 0.9)


def test_planar_from_euclid_centre_and_corner():
    dist = np.full((5, 7), 2.0, np.float32)
    p = D.planar_from_euclid(dist, focal=100.0)
    assert abs(p[2, 3] - 2.0) < 1e-6 and p[0, 0] < 2.0


def test_blendedmvs_dataset_reads_pairs_and_skips_masked_and_val_scenes(tmp_path):
    from PIL import Image
    for sc in ("s1", "s2", "s3"):
        (tmp_path / sc / "blended_images").mkdir(parents=True)
        (tmp_path / sc / "rendered_depth_maps").mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.random.randint(0, 255, (40, 50, 3), np.uint8)).save(tmp_path / sc / "blended_images" / f"{i:08d}.jpg")
            Image.fromarray(np.random.randint(0, 255, (40, 50, 3), np.uint8)).save(tmp_path / sc / "blended_images" / f"{i:08d}_masked.jpg")
            d = np.random.uniform(1, 5, (40, 50)).astype(np.float32)
            d[:5] = 0
            D.write_pfm(tmp_path / sc / "rendered_depth_maps" / f"{i:08d}.pfm", d)
    tr = D.BlendedMVSDataset(tmp_path, "train", val_scenes=1)
    va = D.BlendedMVSDataset(tmp_path, "val", val_scenes=1)
    assert len(tr) == 4 and len(va) == 2 and set(tr.scenes).isdisjoint(va.scenes)
    rgb, depth, valid = tr.load(0)
    assert rgb.shape == (40, 50, 3) and depth.shape == valid.shape == (40, 50) and not valid[:5].any()


def test_hypersim_dataset_reads_hdf5_and_converts_to_planar(tmp_path):
    import h5py
    from PIL import Image
    s = tmp_path / "ai_001_001" / "images"
    (s / "scene_cam_00_final_preview").mkdir(parents=True)
    (s / "scene_cam_00_geometry_hdf5").mkdir(parents=True)
    Image.fromarray(np.random.randint(0, 255, (30, 40, 3), np.uint8)).save(s / "scene_cam_00_final_preview" / "frame.0000.tonemap.jpg")
    with h5py.File(s / "scene_cam_00_geometry_hdf5" / "frame.0000.depth_meters.hdf5", "w") as f:
        f["dataset"] = np.full((30, 40), 3.0, np.float32)
    ds = D.HypersimDataset(tmp_path, "train")
    rgb, depth, valid = ds.load(0)
    assert len(ds) == 1 and valid.all() and depth.max() <= 3.0 + 1e-6 and depth.min() > 2.9


def test_mixed_dataset_is_a_pure_function_of_seed_and_index():
    a = D.MixedDepthDataset(CFG["datasets"], "train", 64, seed=5, length=100)
    b = D.MixedDepthDataset(CFG["datasets"], "train", 64, seed=5, length=100)
    x, y = a[17], b[17]
    assert torch.equal(x["rgb"], y["rgb"]) and torch.equal(x["depth"], y["depth"])
    assert not torch.equal(a[17]["depth"], a[18]["depth"])
    assert x["rgb"].shape == (3, 64, 64) and x["depth"].shape == (1, 64, 64) and x["valid"].dtype == torch.bool
    assert -1 <= float(x["depth"].min()) and float(x["depth"].max()) <= 1


def test_training_runs_logs_finite_losses_and_checkpoints(tmp_path):
    out = L.train(dict(CFG), str(tmp_path), device="cpu")
    assert out["step"] == 6
    recs = [json.loads(line) for line in open(tmp_path / "log.jsonl")]
    losses = [r["loss"] for r in recs if "loss" in r]
    assert len(losses) == 6 and all(np.isfinite(losses))
    assert any("val_loss" in r and np.isfinite(r["val_loss"]) and r.get("val_finite_samples") for r in recs)
    assert (tmp_path / "ckpt_0000003.pt").exists() and (tmp_path / "ckpt_0000006.pt").exists()


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    full = L.train(dict(CFG), str(tmp_path / "full"), device="cpu")
    L.train(dict(CFG), str(tmp_path / "split"), device="cpu", stop_at=3)
    assert (tmp_path / "split" / "ckpt_0000003.pt").exists()
    out = L.train(dict(CFG), str(tmp_path / "split"), device="cpu")          # resumes at step 3
    assert out["step"] == 6 and full["step"] == 6
    a = torch.load(tmp_path / "full" / "ckpt_0000006.pt", weights_only=False)["trainable"]
    b = torch.load(tmp_path / "split" / "ckpt_0000006.pt", weights_only=False)["trainable"]
    assert a.keys() == b.keys()
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst < 1e-6, worst


def test_nonfinite_losses_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "diffusion_loss", lambda *a, **k: torch.tensor(float("nan")))
    cfg = dict(CFG, max_consecutive_bad=2, val_samples=0)
    with pytest.raises(RuntimeError, match="non-finite"):
        L.train(cfg, str(tmp_path), device="cpu")


def test_loss_falls_on_a_learnable_synthetic_task(tmp_path):
    cfg = dict(CFG, total_steps=60, accum=1, micro_batch=4, lr=3e-3, lr_conv=3e-3, val_samples=0, ckpt_every_steps=0, log_every=5, warmup=3,
               datasets=[{"type": "synthetic", "n": 32, "size": 96}])
    L.train(cfg, str(tmp_path), device="cpu")
    losses = [json.loads(line)["loss"] for line in open(tmp_path / "log.jsonl")]
    assert np.mean(losses[-3:]) < np.mean(losses[:3]), losses


def test_blendedmvs_depth_convention_check_picks_z_depth_on_synthetic_views(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cbd", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "check_blendedmvs_depth.py"))
    cbd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cbd)
    h, w = 60, 80
    K = np.array([[80.0, 0, 40], [0, 80.0, 30], [0, 0, 1]])

    def view(x):                      # camera at (x,0,0) looking down +z; scene: plane z = 5 + 0.3 * X
        E = np.eye(4)
        E[0, 3] = -x
        v, u = np.mgrid[0:h, 0:w]
        ray = np.linalg.inv(K) @ np.stack([u.ravel() + 0.0, v.ravel() + 0.0, np.ones(u.size)])
        # point = o + t ray (z-depth t): world X = x + t*rx ; z = t = 5 + 0.3 * (x + t * rx)  ->  t = (5 + 0.3x) / (1 - 0.3 rx)
        t = (5 + 0.3 * x) / (1 - 0.3 * ray[0])
        return E, t.reshape(h, w).astype(np.float32)
    (EA, dA), (EB, dB) = view(0.0), view(0.5)
    e = cbd.hypothesis_errors(dA, EA, K, dB, EB, K)
    assert e["z"][0] < 0.01 < e["range"][0], e


def test_validation_is_skipped_when_a_dataset_has_no_val_frames(tmp_path, capsys):
    """Hypersim splits by scene: a root with only train scenes has no val frames; that must not crash the run."""
    import h5py
    from PIL import Image
    s = tmp_path / "ai_001_001" / "images"
    (s / "scene_cam_00_final_preview").mkdir(parents=True)
    (s / "scene_cam_00_geometry_hdf5").mkdir(parents=True)
    Image.fromarray(np.random.randint(0, 255, (30, 40, 3), np.uint8)).save(s / "scene_cam_00_final_preview" / "frame.0000.tonemap.jpg")
    with h5py.File(s / "scene_cam_00_geometry_hdf5" / "frame.0000.depth_meters.hdf5", "w") as f:
        f["dataset"] = np.random.uniform(1, 5, (30, 40)).astype(np.float32)
    csv_path = tmp_path / "split.csv"
    csv_path.write_text("scene_name,camera_name,frame_id,split_partition_name\nai_001_001,cam_00,0,train\n")
    specs = [{"type": "hypersim", "root": str(tmp_path), "split_csv": str(csv_path), "weight": 1}]
    val = D.MixedDepthDataset(specs, "val", 32, 0, 4)
    assert val.empty and "no val data" in capsys.readouterr().out
    tr = D.MixedDepthDataset(specs, "train", 32, 0, 4)
    assert not tr.empty and tr[0]["rgb"].shape == (3, 32, 32)
    with pytest.raises(FileNotFoundError):                      # but a missing TRAIN set is still fatal
        D.build_dataset({"type": "blendedmvs", "root": str(tmp_path)}, "train")
