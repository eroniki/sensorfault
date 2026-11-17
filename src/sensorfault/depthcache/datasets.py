"""View enumeration for the depth cache.

Each reader returns a list of :class:`View` (view_id, lazy RGB loader, focal length in pixels).
Nothing here downloads data. UseGeo and ETH3D are stubs until the download is approved.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class View:
    view_id: str
    load_rgb: Callable[[], np.ndarray]  # () -> (H, W, 3) uint8
    focal_px: float | None


class DatasetUnavailableError(RuntimeError):
    """The dataset is not on disk or not approved for download."""


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image  # lazy: only needed when images are actually read

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _matrixcity_size() -> tuple[int, int]:
    """(W_FULL, H_FULL) from experiments/matrixcity_io.py, with the documented fallback."""
    p = _REPO / "experiments" / "matrixcity_io.py"
    try:
        spec = importlib.util.spec_from_file_location("_matrixcity_io", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return int(mod.W_FULL), int(mod.H_FULL)
    except Exception:
        return 1920, 1080


def matrixcity_views(
    scene: str,
    data_root: str | Path | None = None,
    transforms: str | Path | None = None,
    images: str | Path | None = None,
) -> list[View]:
    """MatrixCity aerial block.

    Expects ``<data_root>/matrixcity/<scene>/transforms.json`` and RGB frames under
    ``<data_root>/matrixcity/<scene>/images`` (``frame['file_path']`` if present, else
    ``<frame_index:04d>.png``). Override with ``transforms`` / ``images``. Focal length follows
    matrixcity_io: f = 0.5 W / tan(camera_angle_x / 2) at the full image width. The image decode
    happens only when ``load_rgb()`` is called. view_id is the zero-padded frame index.
    """
    base = Path(data_root or os.environ.get("SENSORFAULT_DATA", "data")) / "matrixcity" / scene
    tpath = Path(transforms) if transforms else base / "transforms.json"
    ipath = Path(images) if images else base / "images"
    if not tpath.is_file():
        raise DatasetUnavailableError(
            f"MatrixCity transforms not found at {tpath}. Nothing is downloaded by this code; "
            "place the block there or pass --transforms/--images."
        )
    T = json.loads(tpath.read_text())
    w_full, _ = _matrixcity_size()
    focal = 0.5 * w_full / np.tan(0.5 * float(T["camera_angle_x"]))
    views = []
    for fr in T["frames"]:
        idx = int(fr["frame_index"])
        rel = fr.get("file_path")
        path = ipath / Path(rel).name if rel else ipath / f"{idx:04d}.png"
        if rel and not path.suffix:
            path = path.with_suffix(".png")
        views.append(View(f"{idx:04d}", (lambda p=path: _read_image(p)), float(focal)))
    return views


def usegeo_views(scene: str, **_kw) -> list[View]:
    raise DatasetUnavailableError(
        "UseGeo reader is a stub: the download needs M's approval (nothing is fetched by this "
        "code). Licence is CC BY-NC-SA 4.0 per the repo README, still to be confirmed."
    )


def eth3d_views(scene: str, **_kw) -> list[View]:
    raise DatasetUnavailableError(
        "ETH3D reader is a stub: the download needs M's approval and the licence is unchecked "
        "(configs/datasets/eth3d.yaml: TODO(verify)). Nothing is fetched by this code."
    )


READERS: dict[str, Callable[..., list[View]]] = {
    "matrixcity": matrixcity_views,
    "usegeo": usegeo_views,
    "eth3d": eth3d_views,
}


def enumerate_views(dataset: str, scene: str, **kw) -> list[View]:
    try:
        reader = READERS[dataset]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(READERS)}") from None
    return reader(scene, **kw)
