"""View enumeration for the depth cache.

Each reader returns a list of :class:`View` (view_id, lazy RGB loader, focal length in pixels).
Nothing here downloads data. UseGeo is a stub until its data is on disk.
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


def _colmap_cameras(path: Path) -> dict[int, tuple[str, int, int, list[float]]]:
    """COLMAP cameras.txt -> {camera_id: (model, width, height, params)}."""
    cams = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        cams[int(tok[0])] = (tok[1], int(tok[2]), int(tok[3]), [float(x) for x in tok[4:]])
    return cams


def _colmap_images(path: Path) -> list[tuple[int, int, str]]:
    """COLMAP images.txt -> [(image_id, camera_id, name)] in file order.

    Each image takes two lines (pose line, then the 2D points line, which may be empty), so lines
    are paired by position, not by skipping blanks.
    """
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    out = []
    for i in range(0, len(lines), 2):
        tok = lines[i].split()
        if not tok:
            continue
        if len(tok) < 10:
            raise ValueError(f"{path}: malformed image line {lines[i]!r}")
        out.append((int(tok[0]), int(tok[8]), " ".join(tok[9:])))
    return out


def eth3d_views(
    scene: str,
    data_root: str | Path | None = None,
    **_kw,
) -> list[View]:
    """ETH3D high-res multi-view scene, pre-undistorted DSLR images.

    Expects the layout of ``multi_view_training_dslr_undistorted.7z`` extracted under
    ``<data_root>/eth3d``: ``<scene>/dslr_calibration_undistorted/{cameras,images}.txt`` (COLMAP
    text, PINHOLE ``fx fy cx cy``) and ``<scene>/images/<NAME>`` where NAME comes from images.txt
    (``dslr_images_undistorted/DSC_0286.JPG``). view_id is the image stem, focal_px is fx; views
    are sorted by view_id. Other camera models are refused. Ground-truth depth
    (``<scene>_dslr_depth.7z``) matches the *distorted* images, not these; it is not read here.
    """
    base = Path(data_root or os.environ.get("SENSORFAULT_DATA", "data")) / "eth3d" / scene
    calib = base / "dslr_calibration_undistorted"
    if not (calib / "cameras.txt").is_file() or not (calib / "images.txt").is_file():
        raise DatasetUnavailableError(
            f"ETH3D calibration not found under {calib}. Nothing is downloaded by this code; "
            "extract multi_view_training_dslr_undistorted.7z into <data_root>/eth3d."
        )
    cams = _colmap_cameras(calib / "cameras.txt")
    views = []
    for _image_id, cam_id, name in _colmap_images(calib / "images.txt"):
        model, _w, _h, params = cams[cam_id]
        if model != "PINHOLE":
            raise ValueError(f"ETH3D {scene}: camera {cam_id} is {model}, expected PINHOLE")
        path = base / "images" / name
        views.append(View(Path(name).stem, (lambda p=path: _read_image(p)), float(params[0])))
    ids = [v.view_id for v in views]
    if len(set(ids)) != len(ids):
        raise ValueError(f"ETH3D {scene}: duplicate image stems in images.txt")
    return sorted(views, key=lambda v: v.view_id)


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
