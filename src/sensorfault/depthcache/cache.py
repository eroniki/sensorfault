"""On-disk depth cache.

Layout::

    <root>/<dataset>/<scene>/<model_id>/<view_id>.npz
    <root>/<dataset>/<scene>/<model_id>/manifest.json

Each ``.npz`` holds ``depth`` (float16 or float32), ``kind`` (str) and optionally ``confidence``
and ``focal_px``. Writes are atomic (temp file + ``os.replace``), so an interrupted run never leaves
a half-written view and the cache is resumable: a view counts as done only if its status in the
manifest is ``ok`` and its file exists. The manifest records the model ``info()``, the device,
torch/cudnn flags, the git revision and per-view status and timing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from sensorfault.depthcache.base import DepthResult

MANIFEST = "manifest.json"
_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_name(label: str, name: str) -> str:
    name = str(name)
    if not _SAFE.match(name) or name in {".", ".."}:
        raise ValueError(f"{label} {name!r} must match [A-Za-z0-9._-]+")
    return name


def model_dir(root: str | Path, dataset: str, scene: str, model_id: str) -> Path:
    return (
        Path(root)
        / _check_name("dataset", dataset)
        / _check_name("scene", scene)
        / _check_name("model_id", model_id)
    )


def view_path(root, dataset: str, scene: str, model_id: str, view_id: str) -> Path:
    return model_dir(root, dataset, scene, model_id) / f"{_check_name('view_id', view_id)}.npz"


def _atomic_write_bytes(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_depth(
    root,
    dataset: str,
    scene: str,
    model_id: str,
    view_id: str,
    result: DepthResult,
    dtype="float16",
) -> Path:
    """Write one view atomically. ``dtype`` is "float16" or "float32" (depth and confidence)."""
    if dtype not in ("float16", "float32"):
        raise ValueError("dtype must be float16 or float32")
    path = view_path(root, dataset, scene, model_id, view_id)
    arrays = {
        "depth": np.asarray(result.depth).astype(dtype),
        "kind": np.array(result.kind),
    }
    if result.confidence is not None:
        arrays["confidence"] = np.asarray(result.confidence).astype(dtype)
    if result.focal_px is not None:
        arrays["focal_px"] = np.array(float(result.focal_px), dtype=np.float64)
    _atomic_write_bytes(path, lambda f: np.savez_compressed(f, **arrays))
    return path


def load_depth(root, dataset: str, scene: str, model_id: str, view_id: str) -> DepthResult:
    """Load one cached view. Depth and confidence are returned as float32.

    ``extra`` (e.g. Depth Pro's predicted focal and ``focal_source``, Metric3D's clamp) is read back
    from the model's manifest.json when present.
    """
    path = view_path(root, dataset, scene, model_id, view_id)
    if not path.is_file():
        raise FileNotFoundError(f"no cached depth at {path}")
    extra: dict = {}
    mpath = model_dir(root, dataset, scene, model_id) / MANIFEST
    if mpath.is_file():
        rec = json.loads(mpath.read_text()).get("views", {}).get(view_id, {})
        extra = dict(rec.get("extra") or {})
    with np.load(path, allow_pickle=False) as z:
        conf = z["confidence"].astype(np.float32) if "confidence" in z.files else None
        focal = float(z["focal_px"]) if "focal_px" in z.files else None
        return DepthResult(
            depth=z["depth"].astype(np.float32),
            kind=str(z["kind"]),
            confidence=conf,
            focal_px=focal,
            extra=extra,
        )


def git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def torch_flags(device: str) -> dict:
    """Environment record; works without torch installed."""
    flags: dict = {"device": device, "torch": None}
    try:
        import torch
    except Exception:
        return flags
    flags["torch"] = torch.__version__
    flags["cuda"] = getattr(torch.version, "cuda", None)
    try:
        flags["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        flags["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            flags["device_name"] = torch.cuda.get_device_name(torch.device(device))
    except Exception:
        pass
    return flags


class ModelCache:
    """Manifest + files for one (dataset, scene, model). Not thread-safe."""

    def __init__(self, root, dataset: str, scene: str, model_id: str, dtype: str = "float16"):
        self.root = Path(root)
        self.dataset, self.scene, self.model_id, self.dtype = dataset, scene, model_id, dtype
        self.dir = model_dir(root, dataset, scene, model_id)
        self.manifest_path = self.dir / MANIFEST
        self.manifest: dict = {"views": {}}
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text())
            self.manifest.setdefault("views", {})

    def start(self, info: dict, device: str, allow_settings_change: bool = False) -> None:
        """Record run metadata (overwrites model info/environment, keeps view records).

        Refuses to resume over existing views if the model settings or dtype differ from the ones
        recorded, because the manifest would then describe depth it did not produce.
        """
        old = self.manifest.get("model_info")
        if old is not None and self.manifest["views"] and not allow_settings_change:
            new = json.loads(json.dumps(info, default=float))
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            if self.manifest.get("dtype") not in (None, self.dtype):
                diff.append("dtype")
            if diff:
                raise RuntimeError(
                    f"{self.dir}: settings changed since the cached views were written "
                    f"({', '.join(diff)}); use a fresh --root or pass allow_settings_change=True"
                )
        self.manifest.update(
            dataset=self.dataset,
            scene=self.scene,
            model_id=self.model_id,
            dtype=self.dtype,
            model_info=info,
            environment={**torch_flags(device), "git_sha": git_sha()},
            updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        self.flush()

    def is_done(self, view_id: str) -> bool:
        rec = self.manifest["views"].get(view_id)
        return bool(
            rec
            and rec.get("status") == "ok"
            and view_path(self.root, self.dataset, self.scene, self.model_id, view_id).is_file()
        )

    def put(self, view_id: str, result: DepthResult, seconds: float) -> None:
        save_depth(self.root, self.dataset, self.scene, self.model_id, view_id, result, self.dtype)
        self.manifest["views"][view_id] = {
            "status": "ok",
            "seconds": round(float(seconds), 4),
            "shape": list(np.asarray(result.depth).shape),
            "has_confidence": result.confidence is not None,
            "focal_px": None if result.focal_px is None else float(result.focal_px),
            # e.g. Depth Pro's predicted focal and focal_source: needed to undo a focal rescale
            "extra": json.loads(json.dumps(result.extra or {}, default=float)),
        }

    def fail(self, view_id: str, error: str, seconds: float = 0.0) -> None:
        self.manifest["views"][view_id] = {
            "status": "failed",
            "error": error[:500],
            "seconds": round(float(seconds), 4),
        }

    def flush(self) -> None:
        self.manifest["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        data = json.dumps(self.manifest, indent=2, sort_keys=True).encode()
        _atomic_write_bytes(self.manifest_path, lambda f: f.write(data))

    def summary(self) -> dict:
        v = self.manifest["views"].values()
        return {
            "ok": sum(r.get("status") == "ok" for r in v),
            "failed": sum(r.get("status") == "failed" for r in v),
            "seconds": round(sum(r.get("seconds", 0.0) for r in v), 3),
        }
