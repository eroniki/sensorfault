"""Run depth estimators over a dataset's views and fill the cache, one model at a time."""

from __future__ import annotations

import gc
import importlib
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from sensorfault.depthcache.base import DepthEstimator
from sensorfault.depthcache.cache import ModelCache
from sensorfault.depthcache.datasets import View, enumerate_views

MODEL_IDS = ("depth_pro", "metric3d_v2", "unidepth_v2", "depth_anything_v2")


def default_builder(model_id: str) -> DepthEstimator:
    """Import ``sensorfault.depthcache.<model_id>`` lazily and call its ``build()``."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model {model_id!r}; known: {', '.join(MODEL_IDS)}")
    try:
        mod = importlib.import_module(f"sensorfault.depthcache.{model_id}")
    except ImportError as e:
        raise RuntimeError(f"wrapper for {model_id} is not available: {e}") from e
    return mod.build()


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_model(
    est: DepthEstimator,
    views: Sequence[View],
    root,
    dataset: str,
    scene: str,
    device: str,
    dtype: str = "float16",
    log: Callable[[str], None] = print,
) -> dict:
    """Fill the cache for one loaded estimator; skips views already done. Returns a summary."""
    cache = ModelCache(root, dataset, scene, est.model_id, dtype)
    cache.start(est.info(), device)
    todo = [v for v in views if not cache.is_done(v.view_id)]
    log(f"[{est.model_id}] {len(views) - len(todo)} cached, {len(todo)} to run")
    for i, v in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            res = est.predict(v.load_rgb(), focal_px=v.focal_px)
            cache.put(v.view_id, res, time.perf_counter() - t0)
        except Exception as e:  # keep going; the failure is recorded and retried on the next run
            cache.fail(v.view_id, f"{type(e).__name__}: {e}", time.perf_counter() - t0)
            log(f"[{est.model_id}] view {v.view_id} failed: {e}")
        if i % 10 == 0 or i == len(todo):
            cache.flush()
    cache.flush()
    return {"model_id": est.model_id, "skipped": len(views) - len(todo), **cache.summary()}


def run(
    dataset: str,
    scene: str,
    models: Sequence[str],
    root,
    device: str = "cuda",
    limit: int | None = None,
    dry_run: bool = False,
    dtype: str = "float16",
    builder: Callable[[str], DepthEstimator] = default_builder,
    views: Sequence[View] | None = None,
    log: Callable[[str], None] = print,
    **dataset_kw,
) -> list[dict]:
    """Process models sequentially; each is loaded, run over all views, then released."""
    for m in models:
        if m not in MODEL_IDS:
            raise ValueError(f"unknown model {m!r}; known: {', '.join(MODEL_IDS)}")
    if views is None:
        views = enumerate_views(dataset, scene, **dataset_kw)
    views = list(views)[:limit] if limit else list(views)
    out = []
    for m in models:
        if dry_run:
            cache = ModelCache(root, dataset, scene, m, dtype)
            n_done = sum(cache.is_done(v.view_id) for v in views)
            log(
                f"[dry-run] {m}: {len(views)} views, {n_done} cached, {len(views) - n_done} to run"
                f" -> {Path(root) / dataset / scene / m}"
            )
            out.append({"model_id": m, "views": len(views), "cached": n_done})
            continue
        est = builder(m)
        try:
            est.load(device=device)
            out.append(run_model(est, views, root, dataset, scene, device, dtype, log))
        finally:
            del est
            free_gpu()
    return out
