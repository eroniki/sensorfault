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
- ETH3D (high-res multi-view training, 13 DSLR scenes; CC BY-NC-SA 4.0, eth3d.net home page,
  checked 2026-11-18): `multi_view_training_dslr_undistorted.7z` extracted under `<data_root>/eth3d`
  gives `<scene>/dslr_calibration_undistorted/{cameras,images}.txt` (COLMAP text, PINHOLE) and
  `<scene>/images/dslr_images_undistorted/*.JPG`. `eth3d_views` reads those: view_id = image stem,
  focal_px = fx of the image's camera, sorted by view_id; non-PINHOLE cameras are refused.
  Ground truth (`<scene>_dslr_depth.7z`) is `<scene>/ground_truth_depth/dslr_images/<NAME>`: raw
  float32 at full resolution (6048x4032), `.JPG` name but not a JPEG, inf = no depth. **It matches the
  distorted images** (6048x4032), not the undistorted ones the models see (about 6200x4135, size and
  PINHOLE intrinsics vary per camera): warp one into the other's geometry before comparing. The
  distorted calibration (`dslr_calibration_jpg`, THIN_PRISM_FISHEYE) ships with
  `multi_view_training_dslr_jpg.7z` (4.7 GB), not with the undistorted archive. 454 views in total
  (38/44/45/76/31/15/26/14/38/31/31/23/42 in the order of `configs/datasets/eth3d.yaml`); GT valid
  pixels about 10-12% at full resolution. Sizes in `configs/datasets/eth3d.yaml`.
- UseGeo: stub that raises (download server's TLS certificate expired 2024-08-30; waiting on M).

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
- DAv2 (`relative_inverse_depth`): affine fit `1/ref = a*raw + b`; RANSAC over pixel pairs
  (inlier = relative inverse-depth error <= 10%) then least squares on the inliers. Measured on synthetic
  data to hold up to 40% corrupted reference pixels.
- Raises `AlignmentError` for fewer than `min_pixels` (200) usable reference pixels, constant raw depth, no
  consensus, or a non-positive slope. Nothing is returned silently from a degenerate fit.
- Returns a metric `DepthResult` plus an `Alignment` (parameters, `n_valid`, `n_inliers`, median relative
  residual over all valid pixels). The residual is in-sample: gate on `n_inliers` as well.
- Metric3D's clamp (`max_depth_m`, read from the manifest) marks saturated pixels: they are excluded from the
  fit and NaN in the output.
- Confidence is dropped by default (its semantics differ per model and are not converted); `keep_confidence=True`
  passes the raw map through. The cache on disk is never modified.

**Scope.** Alignment removes a per-view scale (and offset) error, so a fault that is only a scale or offset
error cannot be detected after it; faults are deviations from "reference-aligned depth up to scale and
offset", consistent with the scale/offset the gate already projects out. If the reference is the exact
synthetic depth the view is later tested against, the fit absorbs part of the fault under test and clean-view
residuals are optimistic. For experiments take the reference from something available at test time (SfM
points, or the other views' fused surface), leave-view-out with respect to the tested view. Where the
reference comes from for real sets (UseGeo/ETH3D) is not decided here; decide it before any H-series result
depends on alignment.

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

### Alignment on the real block 3 cache (4 views, reference = exact GT, CPU, 2026-11-17)

Run a: dense exact depth. Run b: 1% of valid pixels (20,736) with 10% of those multiplied by 3, to mimic SfM
points; "err vs GT" in b is measured on the 99% held-out pixels. All 32 fits succeeded (no `AlignmentError`).
Scale is ref/pred; resid is the in-sample median relative residual (inflated in b by the x3 points).

| Model | Scale per view, a / b | Median rel. error vs GT after alignment, a / b |
|---|---|---|
| Depth Pro | 63.1-82.6 / 63.5-83.4 | 0.027-0.054 / 0.027-0.056 |
| Metric3D v2 | 5.4-6.9 / 5.5-7.0 | 0.095-0.141 / 0.084-0.136 |
| UniDepthV2 | 1.145-1.184 / 1.146-1.186 | 0.003-0.008 / 0.003-0.008 |
| DAv2 (affine, 1/ref = a*raw + b) | a 3.6-5.8e-6, b 2.1-2.3e-3 | 0.012-0.024 / 0.012-0.024 |

- The scale must be per view: it varies about +-13% across the 4 views for Depth Pro (63-83) and Metric3D (5.4-6.9).
- One scale per view is enough for Depth Pro (3-5% left), UniDepth (under 1%) and DAv2 with its affine fit
  (1-2%). Metric3D keeps 9.5-14% error after scaling, so its error has a shape component beyond scale.
- Robustness: sparse-run scales stay within 0.5-4% of the dense ones, held-out error is no worse than the dense
  in-sample error, and the DAv2 RANSAC rejected 10-17% of the sparse points (10% were planted).
- This cache was written before `max_depth_m` was recorded in the manifest, so the Metric3D clamp was not
  picked up; it does not matter here (largest prediction 92 m, below 300 m).
- The reference here is the exact GT, so this checks the premise (a per-view scale/affine fits), not the
  fault-detection setting: see Scope above. Tested environment: `docs/depth-cache-env.txt` on branch
  `depth-cache-env`.
