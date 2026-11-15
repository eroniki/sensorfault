"""Training data for stage 1: image + dense depth pairs -> (RGB in [-1,1], affine-invariant depth target in [-1,1], validity mask).

Datasets: BlendedMVS/BlendedMVG (blended image + rendered .pfm depth), Hypersim (tonemapped preview + depth_meters hdf5, converted
from ray distance to planar depth), TartanAir (optional, minimal), and a synthetic set for tests. ETH3D is never read here (held out).

Determinism and resume: the sample for global index k is a pure function of (seed, k): which dataset, which item, which crop/flip.
So a run restarted at optimiser step s simply continues at k = s * batch_size.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


def read_pfm(path) -> np.ndarray:
    """Read a (single-channel) PFM as float32 (H, W), top row first."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"{path}: not a PFM ({header!r})")
        w, h = (int(v) for v in f.readline().decode("ascii").split())
        scale = float(f.readline().decode("ascii").strip())
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4", count=w * h * (3 if header == "PF" else 1))
    if header == "PF":
        data = data.reshape(h, w, 3)[..., 0]
    else:
        data = data.reshape(h, w)
    return np.flipud(data).astype(np.float32)


def write_pfm(path, arr: np.ndarray) -> None:
    arr = np.flipud(np.asarray(arr, "<f4"))
    with open(path, "wb") as f:
        f.write(b"Pf\n" + f"{arr.shape[1]} {arr.shape[0]}\n".encode() + b"-1.0\n")
        f.write(arr.tobytes())


def affine_invariant_target(depth: np.ndarray, valid: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    """Per-image percentile normalisation to [-1, 1] (Marigold Eq. 3), clipped. Returns None if the image has no usable range."""
    v = depth[valid]
    if v.size < 64:
        return None
    d_lo, d_hi = np.percentile(v, [lo, hi])
    if not np.isfinite(d_lo) or not np.isfinite(d_hi) or d_hi - d_lo < 1e-6 * max(abs(d_hi), 1.0):
        return None
    t = ((depth - d_lo) / (d_hi - d_lo) - 0.5) * 2.0
    return np.clip(t, -1.0, 1.0).astype(np.float32)


def fill_invalid(t: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid pixels by the nearest valid value (display of holes to the VAE; the loss ignores them via the mask)."""
    if valid.all():
        return t
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return t[tuple(idx)]


def planar_from_euclid(dist: np.ndarray, focal: float = 886.81) -> np.ndarray:
    """Hypersim depth_meters (distance to the optical centre) -> planar z-depth, nominal intrinsics (per-scene tilt-shift ignored)."""
    h, w = dist.shape
    xs = (np.arange(w) - (w - 1) / 2.0) / focal
    ys = (np.arange(h) - (h - 1) / 2.0) / focal
    return (dist / np.sqrt(1.0 + xs[None, :] ** 2 + ys[:, None] ** 2)).astype(np.float32)


def _augment(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, res: int, rng: np.random.RandomState, scale=(0.6, 1.0)):
    """Random crop (square, area fraction in `scale`), flip, resize to res x res: bilinear RGB, nearest depth/mask."""
    h, w = depth.shape
    side = int(min(h, w) * np.sqrt(rng.uniform(*scale)))
    side = max(8, min(side, min(h, w)))
    y0 = rng.randint(0, h - side + 1)
    x0 = rng.randint(0, w - side + 1)
    rgb, depth, valid = rgb[y0:y0 + side, x0:x0 + side], depth[y0:y0 + side, x0:x0 + side], valid[y0:y0 + side, x0:x0 + side]
    if rng.rand() < 0.5:
        rgb, depth, valid = rgb[:, ::-1], depth[:, ::-1], valid[:, ::-1]
    rgb = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((res, res), Image.BILINEAR))
    depth = np.asarray(Image.fromarray(np.ascontiguousarray(depth), mode="F").resize((res, res), Image.NEAREST))
    valid = np.asarray(Image.fromarray(np.ascontiguousarray(valid).astype(np.uint8)).resize((res, res), Image.NEAREST)).astype(bool)
    return rgb, depth, valid


@dataclass
class Item:
    rgb_path: str
    depth_path: str


class PairDataset:
    """Base: list of items + a loader returning (rgb uint8 HxWx3, depth float32 HxW, valid bool HxW)."""

    name = "pairs"

    def __init__(self, items: list[Item]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def load(self, i: int):
        raise NotImplementedError


class BlendedMVSDataset(PairDataset):
    """<root>/<scene>/blended_images/NNNNNNNN.jpg  +  <root>/<scene>/rendered_depth_maps/NNNNNNNN.pfm (depth <= 0 or non-finite = empty).

    `split`: "train" uses all scenes except the last `val_scenes` (sorted), "val" the last `val_scenes`. Masked images are skipped.
    """

    name = "blendedmvs"

    def __init__(self, root, split="train", val_scenes=6, max_depth_valid=None):
        root = Path(root)
        scenes = sorted(p for p in root.iterdir() if (p / "blended_images").is_dir())
        scenes = scenes[:-val_scenes] if split == "train" and val_scenes else (scenes[-val_scenes:] if val_scenes else scenes)
        items = []
        for s in scenes:
            for img in sorted((s / "blended_images").glob("*.jpg")):
                if img.stem.endswith("_masked"):
                    continue
                d = s / "rendered_depth_maps" / f"{img.stem}.pfm"
                if d.exists():
                    items.append(Item(str(img), str(d)))
        if not items:
            raise FileNotFoundError(f"no BlendedMVS pairs under {root}")
        super().__init__(items)
        self.scenes = [s.name for s in scenes]

    def load(self, i):
        it = self.items[i]
        rgb = np.asarray(Image.open(it.rgb_path).convert("RGB"))
        depth = read_pfm(it.depth_path)
        valid = np.isfinite(depth) & (depth > 0)
        return rgb, np.where(valid, depth, 0.0).astype(np.float32), valid


class HypersimDataset(PairDataset):
    """<root>/ai_VVV_NNN/images/scene_cam_XX_final_preview/frame.IIII.tonemap.jpg + scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5.

    `split_csv` (optional) is ml-hypersim's metadata_images_split_scene_v1.csv; rows with split_partition_name == `split` are kept.
    """

    name = "hypersim"

    def __init__(self, root, split="train", split_csv=None, focal=886.81, max_scenes=None):
        root = Path(root)
        keep = None
        if split_csv:
            keep = set()
            with open(split_csv) as f:
                for r in csv.DictReader(f):
                    if r.get("split_partition_name") == split:
                        keep.add((r["scene_name"], r["camera_name"], int(r["frame_id"])))
        items = []
        scenes = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("ai_"))
        if max_scenes:
            scenes = scenes[:max_scenes]
        for s in scenes:
            for prev in sorted((s / "images").glob("scene_cam_*_final_preview")):
                cam = prev.name.replace("_final_preview", "").replace("scene_", "")
                geo = s / "images" / prev.name.replace("final_preview", "geometry_hdf5")
                for img in sorted(prev.glob("frame.*.tonemap.jpg")):
                    fid = int(img.name.split(".")[1])
                    if keep is not None and (s.name, cam, fid) not in keep:
                        continue
                    d = geo / f"frame.{fid:04d}.depth_meters.hdf5"
                    if d.exists():
                        items.append(Item(str(img), str(d)))
        if not items:
            raise FileNotFoundError(f"no Hypersim pairs under {root}")
        super().__init__(items)
        self.focal = focal

    def load(self, i):
        import h5py
        it = self.items[i]
        rgb = np.asarray(Image.open(it.rgb_path).convert("RGB"))
        with h5py.File(it.depth_path, "r") as f:
            dist = np.asarray(f["dataset"], np.float32)
        valid = np.isfinite(dist) & (dist > 0)
        depth = planar_from_euclid(np.where(valid, dist, 0.0), self.focal)
        return rgb, depth, valid


class TartanAirDataset(PairDataset):
    """Optional and minimal: <root>/**/image_left/NNNNNN_left.png + depth_left/NNNNNN_left_depth.npy (z-depth, metres; > max_depth = sky = invalid)."""

    name = "tartanair"

    def __init__(self, root, split="train", max_depth=80.0, val_every=20):
        items = []
        for img in sorted(Path(root).glob("**/image_left/*_left.png")):
            d = img.parent.parent / "depth_left" / img.name.replace("_left.png", "_left_depth.npy")
            if d.exists():
                items.append(Item(str(img), str(d)))
        items = [it for k, it in enumerate(items) if (k % val_every == 0) == (split == "val")]
        if not items:
            raise FileNotFoundError(f"no TartanAir pairs under {root}")
        super().__init__(items)
        self.max_depth = max_depth

    def load(self, i):
        it = self.items[i]
        rgb = np.asarray(Image.open(it.rgb_path).convert("RGB"))
        depth = np.load(it.depth_path).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        return rgb, np.where(valid, depth, 0.0).astype(np.float32), valid


class SyntheticDataset(PairDataset):
    """Procedural scenes for tests: coloured planes whose RGB tint encodes depth, so a tiny model can learn something."""

    name = "synthetic"

    def __init__(self, n=64, size=96, seed=0):
        super().__init__([Item(str(i), "") for i in range(n)])
        self.size, self.seed = size, seed

    def load(self, i):
        r = np.random.RandomState(self.seed + i)
        h = w = self.size
        yy, xx = np.mgrid[0:h, 0:w] / float(h)
        depth = 1.0 + r.uniform(0.5, 2.0) * xx + r.uniform(0.0, 1.5) * yy + 0.3 * np.sin(6 * xx * r.uniform(0.5, 2))
        rgb = np.stack([255 * (1 - (depth - depth.min()) / (np.ptp(depth) + 1e-6)), 255 * yy, 255 * xx], -1).astype(np.uint8)
        valid = np.ones((h, w), bool)
        valid[: h // 8, : w // 4] = False
        return rgb, np.where(valid, depth, 0.0).astype(np.float32), valid


def build_dataset(spec: dict, split: str):
    t = spec["type"]
    if t == "blendedmvs":
        return BlendedMVSDataset(spec["root"], split, spec.get("val_scenes", 6))
    if t == "hypersim":
        return HypersimDataset(spec["root"], split, spec.get("split_csv"), spec.get("focal", 886.81), spec.get("max_scenes"))
    if t == "tartanair":
        return TartanAirDataset(spec["root"], split, spec.get("max_depth", 80.0))
    if t == "synthetic":
        return SyntheticDataset(spec.get("n", 64) if split == "train" else spec.get("n_val", 16), spec.get("size", 96),
                                seed=0 if split == "train" else 10_000)
    raise ValueError(t)


class MixedDepthDataset(torch.utils.data.Dataset):
    """Map-style dataset over the global sample index k (see module docstring)."""

    def __init__(self, specs: list[dict], split: str, res: int, seed: int, length: int):
        self.sets = [build_dataset(s, split) for s in specs]
        w = np.array([s.get("weight", 1.0) for s in specs], float)
        self.p = w / w.sum()
        self.res, self.seed, self.length = res, seed, length
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, k: int):
        rng = np.random.RandomState((self.seed * 1_000_003 + k) % (2 ** 32 - 1))
        for attempt in range(20):                      # resample (deterministically) if an image has no usable depth range
            ds = self.sets[rng.choice(len(self.sets), p=self.p)] if self.split == "train" else self.sets[k % len(self.sets)]
            i = rng.randint(len(ds)) if self.split == "train" else (k // len(self.sets)) % len(ds)
            rgb, depth, valid = ds.load(i)
            if self.split == "train":
                rgb, depth, valid = _augment(rgb, depth, valid, self.res, rng)
            else:
                rgb, depth, valid = _augment(rgb, depth, valid, self.res, rng, scale=(1.0, 1.0))
            t = affine_invariant_target(depth, valid)
            if t is None:
                continue
            t = fill_invalid(t, valid)
            return {"rgb": torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1),
                    "depth": torch.from_numpy(t)[None], "valid": torch.from_numpy(valid)[None], "source": ds.name}
        raise RuntimeError("no usable sample after 20 attempts")
