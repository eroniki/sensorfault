"""COLMAP text-model writer (cameras.txt / images.txt / points3D.txt) + placeholder PNGs.
COLMAP text format is public and simple: images.txt pose = world->camera quaternion
(QW QX QY QZ) and translation. Used for the Gaussian-splat wrappers' data lists."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

from sensorfault.types import DepthFrame


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        v = [0.0] * 3
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = [(R[k, j] - R[j, k]) / s] + v
    q = np.array(q)
    return q / np.linalg.norm(q)


def write_gray_png(path: Path, h: int, w: int, value: int = 127) -> None:
    row = b"\x00" + bytes([value]) * (3 * w)
    raw = row * h

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_colmap_model(frames: Sequence[DepthFrame], root: Path, write_images: bool = True) -> None:
    """root/sparse/0/{cameras,images,points3D}.txt and root/images/imgNNNN.png (grey
    placeholder RGB -- the splat baselines need RGB; PLAN.md: placeholder-RGB + depth loss)."""
    sp = root / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    with open(sp / "cameras.txt", "w") as fc, open(sp / "images.txt", "w") as fi:
        fc.write("# CAMERA_ID MODEL W H fx fy cx cy\n")
        fi.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n#\n")
        for i, f in enumerate(frames, start=1):
            h, w = f.depth.shape
            fc.write(f"{i} PINHOLE {w} {h} {f.K[0, 0]} {f.K[1, 1]} {f.K[0, 2]} {f.K[1, 2]}\n")
            R_cw = f.T_wc[:3, :3].T
            t_cw = -R_cw @ f.T_wc[:3, 3]
            q = rotmat_to_quat_wxyz(R_cw)
            name = f"img{i - 1:04d}.png"
            fi.write(
                f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t_cw[0]} {t_cw[1]} {t_cw[2]} {i} {name}\n\n"
            )
            if write_images:
                write_gray_png(root / "images" / name, h, w)
    (sp / "points3D.txt").write_text(
        "# empty: TODO supply init points (3DGS-MCMC random/depth init)\n"
    )
