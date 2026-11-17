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

## Datasets
- MatrixCity: reads `<data_root>/matrixcity/<scene>/transforms.json` and `images/` (frame image,
  `frame['file_path']` or `<frame_index:04d>.png`); focal from `camera_angle_x` at 1920 px width as in
  `experiments/matrixcity_io.py`. RGB frames are not downloaded by this code (block 3 images: ~1 GB).
- UseGeo, ETH3D: stubs that raise; download needs M's approval, ETH3D licence unchecked.

## Tests
`pytest tests/test_depthcache_cache.py` (CPU, offline, fake estimator).
