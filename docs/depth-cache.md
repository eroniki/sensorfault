# Depth cache

Monocular depth is computed once per view and stored, so depth-supervised fits never run a depth model.
Models: `depth_pro`, `metric3d_v2`, `unidepth_v2`, `depth_anything_v2` (wrappers in
`sensorfault.depthcache.<id>`, each with `build()`; imported lazily).

## Layout
`<root>/<dataset>/<scene>/<model_id>/<view_id>.npz` with `depth` (float16 default, float32 option),
`kind` (`metric_depth` or `relative_inverse_depth`; DAv2 stores the raw model output unmodified),
optional `confidence`, optional `focal_px`; and `manifest.json` (model `info()`, device, torch/cudnn
flags, git sha, per-view status, seconds). Writes are atomic; a rerun skips views whose status is `ok`
and file exists, and retries failed ones. float16 keeps about 3 significant digits (relative error ~1e-3, max 65504); use `--dtype float32` for fits that need more.

Load: `sensorfault.depthcache.load_depth(root, dataset, scene, model_id, view_id)` -> `DepthResult` (float32).

## CLI
    python scripts/depth_cache.py --dataset matrixcity --scene block3 \
        --models depth_pro,metric3d_v2,unidepth_v2,depth_anything_v2 --device cuda --root data/depth_cache
    # add --limit N for a smoke test, --dry-run to list work without loading any model

Models run one at a time; the estimator is deleted and the CUDA cache emptied between models
(8 GB target). Defaults live in `configs/depth_cache.yaml`.

## GPU environment (tested 2026-11-17, TITAN RTX 24 GB, driver 550)
Python 3.10 conda env with `PYTHONNOUSERSITE=1` (user site-packages otherwise leak in), then:

    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -e ".[dev]" transformers pillow huggingface_hub safetensors einops timm tqdm
    # UniDepth: clone lpiccinelli-eth/UniDepth, `pip install --no-deps -e <clone>`; it imports
    # wandb, h5py and tabulate at import time; xformers is optional (warning only)
    pip install wandb h5py tabulate
    # Metric3D: clone YvanYin/Metric3D, set SENSORFAULT_METRIC3D_HUB=<clone>. Its mono/utils/comm.py
    # imports mmcv.utils (mmcv 1.x); mmengine alone is not enough. Python-only build:
    pip install "setuptools<70" wheel mmengine opencv-python-headless
    MMCV_WITH_OPS=0 pip install --no-build-isolation mmcv==1.7.2

transformers >= 5 needs torch >= 2.6 (it disables torch silently below that).
MatrixCity block 3 test split (`small_city/aerial/test/block_3_test.tar`, its `transforms.json`,
19 frames of 25 PNGs, 1920x1080 RGBA) extracts to `data/matrixcity/block3/images/<frame:04d>.png`,
which is the reader's fallback name.

## Datasets
- MatrixCity: reads `<data_root>/matrixcity/<scene>/transforms.json` and `images/` (frame image,
  `frame['file_path']` or `<frame_index:04d>.png`); focal from `camera_angle_x` at 1920 px width as in
  `experiments/matrixcity_io.py`. RGB frames are not downloaded by this code (block 3 images: ~1 GB).
- UseGeo, ETH3D: stubs that raise; download needs M's approval, ETH3D licence unchecked.

## Tests
`pytest tests/test_depthcache_cache.py` (CPU, offline, fake estimator).

## Scale alignment

Cached depth is raw model output. On aerial MatrixCity block 3 the metric models are off by a large global
scale (Depth Pro ~0.01x, Metric3D v2 ~0.15x, UniDepthV2 ~0.87x), and Depth Anything V2 is relative inverse depth.
`sensorfault.depthcache.align` fits this per view against reference depth that is valid at some pixels
(SfM points, LiDAR, exact synthetic depth):

```python
from sensorfault.depthcache import load_depth_aligned

res, al = load_depth_aligned(root, "matrixcity", "block3", "depth_pro", "0000", ref_depth)
```

- Metric models: one scale, the median of ref/pred (robust to outliers).
- DAv2 (`relative_inverse_depth`): affine fit `1/ref = a*raw + b`, trimmed least squares.
- Returns a metric `DepthResult` plus an `Alignment` (parameters, pixels used, median relative residual).
  Gate on `n_used` and the residual before trusting a view. The cache on disk is never modified.
- Open question: where the reference comes from for real sets (UseGeo/ETH3D) is not decided here.
## First GPU run: MatrixCity block 3 (4 views: 0000, 0005, 0006, 0007; 2026-11-17, TITAN RTX)

| Model | Peak VRAM | s/view | Result vs exact depth (median GT 338 m) |
|---|---|---|---|
| Depth Pro (fp16) | 6.3 GB | 0.54 | pred/GT 0.009 (0.013-0.016 with given-focal rescale), AbsRel 0.98; reads the scene as a miniature (3-5 m) |
| Metric3D v2 ViT-L | 4.4 GB | 1.35 | median 0.15x GT (clamped at 300 m by default; 72% of GT is beyond that) |
| UniDepthV2 ViT-L | 4.6 GB | 0.22 | median 0.87x GT, AbsRel 0.12-0.15 (CUDA patch op not compiled: slow fallback) |
| DAv2 Large | 2.1 GB | 0.35 | relative; affine fit to 1/GT gives AbsRel 1.4-3.7% |

VRAM is from nvidia-smi and includes about 0.5 GB used by other processes. Environment pins
(`transformers`, `timm`, `einops`, the UniDepth and Metric3D checkouts) were not recorded in the run:
record `pip freeze` and the two clone commit hashes next time.

Settings: `model_options` in `configs/depth_cache.yaml` or `--opt metric3d_v2.max_depth_m=1000` pass
options to a wrapper's `build()`. Resuming a cache directory with different model settings or dtype is
refused (use a fresh `--root`). `load_depth` returns `extra` from the manifest (e.g. Depth Pro's
predicted focal, needed to undo the focal rescale).

**Storage precision.** The cache stores float16 by default: a 0.25 m step at 256-512 m, and a Depth Pro
map at 3-5 m scaled by about 70 lands on a 0.27 m step. Small next to the models' error on block 3, but not
next to a centimetre-level fault: use `--dtype float32` for synthetic fault experiments on cached depth.

**Confidence.** UniDepth's confidence reads as predicted error (higher = worse). Metric3D's direction
flips between views on block 3, so neither is a drop-in "higher = better" score.
