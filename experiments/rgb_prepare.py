"""Prepare MatrixCity block 3 RGB for 2DGS training (system Python with Pillow; no GPU).

Reads the train/test image tars directly, resizes to half resolution (960x540, Lanczos) and writes
uint8 arrays with their poses; also exports initial surfel points (E1 voxel surfels from the float32
depth) with colours sampled from the training images.
Run: python3 experiments/rgb_prepare.py /home/eiki/data/matrixcity
"""

from __future__ import annotations

import io
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 960, 540


def load_split(tar_path: Path, transforms: Path):
    T = json.loads(transforms.read_text())
    want = {f["frame_index"]: f for f in T["frames"]}
    imgs, poses, idx = [], [], []
    with tarfile.open(tar_path) as tf:
        members = {
            Path(m.name).stem: m
            for m in tf.getmembers()
            if m.name.lower().endswith((".png", ".jpg"))
        }
        for k in sorted(want):
            m = members.get(f"{k:04d}")
            if m is None:
                continue
            data = tf.extractfile(m).read()
            try:
                im = Image.open(io.BytesIO(data)).convert("RGB").resize((W, H), Image.LANCZOS)
            except OSError as e:
                raise OSError(f"{tar_path.name}:{m.name} ({len(data)} bytes): {e}") from e
            imgs.append(np.asarray(im, dtype=np.uint8))
            M = np.array(want[k]["rot_mat"], dtype=np.float64)
            M[:3, :3] /= np.linalg.norm(M[:3, :3], axis=0, keepdims=True)
            poses.append(M)
            idx.append(k)
    return np.stack(imgs), np.stack(poses), np.array(idx), float(T["camera_angle_x"])


def main(root: str) -> None:
    root = Path(root)
    tr = load_split(root / "block_3_rgb_train.tar", root / "transforms_block3.json")
    te = load_split(root / "block_3_rgb_test.tar", root / "transforms_block3_test.json")
    fov = tr[3]
    np.savez(
        root / "rgb_block3_half.npz",
        train_images=tr[0],
        train_c2w=tr[1],
        train_frames=tr[2],
        test_images=te[0],
        test_c2w=te[1],
        test_frames=te[2],
        fov_x=fov,
        width=W,
        height=H,
    )
    # initial points: E1 surfels (exact-depth voxels), colours from the first training view that sees them
    K, views, geo = pickle.loads((root / "e1_cache_step20_vox3.0.pkl").read_bytes())
    mu = geo["mu"]
    f = 0.5 * W / np.tan(0.5 * fov)
    col = np.full((len(mu), 3), 128.0)
    seen = np.zeros(len(mu), bool)
    for img, c2w in zip(tr[0], tr[1]):
        Xc = (mu - c2w[:3, 3]) @ c2w[:3, :3]
        z = -Xc[:, 2]
        u = f * Xc[:, 0] / np.maximum(z, 1e-6) + (W - 1) / 2
        v = -f * Xc[:, 1] / np.maximum(z, 1e-6) + (H - 1) / 2
        ok = (z > 1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & ~seen
        col[ok] = img[v[ok].astype(int), u[ok].astype(int)]
        seen |= ok
    np.savez(
        root / "rgb_block3_init.npz",
        means=mu.astype(np.float32),
        colors=(col / 255.0).astype(np.float32),
        seen=seen,
    )
    print(
        json.dumps(
            {
                "train": int(len(tr[2])),
                "test": int(len(te[2])),
                "fov_x": fov,
                "init_points": int(len(mu)),
                "init_coloured": int(seen.sum()),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/eiki/data/matrixcity")
