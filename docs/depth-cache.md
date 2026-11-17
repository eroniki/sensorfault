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
