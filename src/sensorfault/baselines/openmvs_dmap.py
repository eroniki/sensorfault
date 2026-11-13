"""Writer/reader for OpenMVS ``.dmap`` depth-map files (header name "D2").

Implemented from the OpenMVS ``develop`` source: ``libs/MVS/Interface.h``
(``HeaderDepthDataRaw``, ``ExportDepthDataRaw``, ``DEPTHDATA::*``), fetched and read
2026-11-13. Layout (little endian, as written by ``ExportDepthDataRaw``):

  header (32 bytes, natural alignment):
    uint16 name = "D2" ; uint8 type (bit0 depth, bit1 normal, bit2 conf, bit3 views,
    bit4 CONF_ADJUSTED) ; int8 depthExp ; uint32 imageWidth, imageHeight, depthWidth,
    depthHeight ; float32 dMin, dMax, confScale
  uint16 nameLen + chars (image path, by convention relative to the dmap)
  uint32 nIDs + uint32[nIDs]  (reference view ID first, then neighbour IDs)
  double K[3][3], R[3][3] (world->camera), C[3] (camera centre in world), row-major
  half depth[H][W]  stored as depth * 2^-depthExp (depthExp = ilogb(max depth))
  int16 normal[H][W][2] octahedral (camera space; (-32768,-32768) = zero normal)
  uint8 conf[H][W] over [0, confScale]
  [uint8 views[H][W][4] if HAS_VIEWS -- not written here]

The upstream header comment in older versions listed the field order differently; this
module follows the code of the version read above. TODO(verify): re-check against the
exact OpenMVS tag you build (the format has changed between versions, "D1" -> "D2").
Round-trip tested here only against this module's own reader (no OpenMVS binary run).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

HAS_DEPTH, HAS_NORMAL, HAS_CONF, HAS_VIEWS, CONF_ADJUSTED = 1, 2, 4, 8, 16
_HDR = "<HBbIIIIfff"
_SENTINEL = -32768


@dataclass
class DMap:
    depth: np.ndarray  # (H, W) float32
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) world -> camera
    C: np.ndarray  # (3,) camera centre, world
    normals: Optional[np.ndarray] = None  # (H, W, 3) camera space, zeros = invalid
    conf: Optional[np.ndarray] = None  # (H, W)
    ids: List[int] = field(default_factory=lambda: [0])
    image_name: str = ""
    dmin: float = 0.0
    dmax: float = 0.0
    image_size: Optional[tuple] = None  # (W, H); defaults to depth size


def encode_normals(n: np.ndarray) -> np.ndarray:
    """(H,W,3) -> (H,W,2) int16 octahedral, port of DEPTHDATA::EncodeNormal."""
    n = np.nan_to_num(n.astype(np.float32))
    zero = np.all(n == 0, axis=-1)
    l1 = np.abs(n).sum(-1)
    l1 = np.where(l1 > 0, l1, 1.0)
    px = n[..., 0] / l1
    py = n[..., 1] / l1
    neg = n[..., 2] < 0
    nx = (1 - np.abs(py)) * np.where(px < 0, -1.0, 1.0)
    ny = (1 - np.abs(px)) * np.where(py < 0, -1.0, 1.0)
    px = np.where(neg, nx, px)
    py = np.where(neg, ny, py)
    ox = np.floor(np.clip(px, -1, 1) * 32767.0 + 0.5).astype(np.int16)
    oy = np.floor(np.clip(py, -1, 1) * 32767.0 + 0.5).astype(np.int16)
    ox[zero] = _SENTINEL
    oy[zero] = _SENTINEL
    return np.stack([ox, oy], -1)


def decode_normals(o: np.ndarray) -> np.ndarray:
    ox, oy = o[..., 0], o[..., 1]
    sent = (ox == _SENTINEL) & (oy == _SENTINEL)
    px = ox / 32767.0
    py = oy / 32767.0
    z = 1 - np.abs(px) - np.abs(py)
    x = np.where(z < 0, (1 - np.abs(py)) * np.where(px < 0, -1.0, 1.0), px)
    y = np.where(z < 0, (1 - np.abs(px)) * np.where(py < 0, -1.0, 1.0), py)
    v = np.stack([x, y, z], -1)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    v[sent] = 0.0
    return v.astype(np.float32)


def write_dmap(path: str, d: DMap) -> None:
    depth = np.ascontiguousarray(d.depth, dtype=np.float32)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
    h, w = depth.shape
    iw, ih = d.image_size or (w, h)
    mx = float(depth.max())
    exp = int(max(-100, min(100, math.frexp(mx)[1] - 1))) if mx > 0 else 0  # == ilogb
    typ = (
        HAS_DEPTH
        | (HAS_NORMAL if d.normals is not None else 0)
        | (HAS_CONF if d.conf is not None else 0)
    )
    conf_scale = 1.0
    if d.conf is not None:
        mc = float(np.max(d.conf))
        conf_scale = mc if (mc > 0 and math.isfinite(mc)) else 1.0
    dmin = float(d.dmin) * (1 - 1 / 1024.0)
    dmax = float(d.dmax) * (1 + 1 / 1024.0)
    name = int.from_bytes(b"D2", "little")
    nm = d.image_name.encode()
    with open(path, "wb") as f:
        f.write(struct.pack(_HDR, name, typ, exp, iw, ih, w, h, dmin, dmax, conf_scale))
        f.write(struct.pack("<H", len(nm)) + nm)
        f.write(struct.pack("<I", len(d.ids)) + np.asarray(d.ids, "<u4").tobytes())
        f.write(np.asarray(d.K, "<f8").tobytes())
        f.write(np.asarray(d.R, "<f8").tobytes())
        f.write(np.asarray(d.C, "<f8").tobytes())
        f.write((depth * np.float32(2.0**-exp)).astype("<f2").tobytes())
        if typ & HAS_NORMAL:
            f.write(encode_normals(d.normals).astype("<i2").tobytes())
        if typ & HAS_CONF:
            q = np.clip(np.rint(np.nan_to_num(d.conf) * (255.0 / conf_scale)), 0, 255)
            f.write(q.astype(np.uint8).tobytes())


def read_dmap(path: str) -> DMap:
    with open(path, "rb") as f:
        raw = f.read()
    o = 0
    sz = struct.calcsize(_HDR)
    name, typ, exp, iw, ih, w, h, dmin, dmax, cs = struct.unpack_from(_HDR, raw, o)
    o += sz
    if name != int.from_bytes(b"D2", "little") or not typ & HAS_DEPTH:
        raise ValueError("not a D2 dmap")
    (ln,) = struct.unpack_from("<H", raw, o)
    o += 2
    nm = raw[o : o + ln].decode()
    o += ln
    (nid,) = struct.unpack_from("<I", raw, o)
    o += 4
    ids = np.frombuffer(raw, "<u4", nid, o).tolist()
    o += 4 * nid
    K = np.frombuffer(raw, "<f8", 9, o).reshape(3, 3).copy()
    o += 72
    R = np.frombuffer(raw, "<f8", 9, o).reshape(3, 3).copy()
    o += 72
    C = np.frombuffer(raw, "<f8", 3, o).copy()
    o += 24
    depth = (np.frombuffer(raw, "<f2", w * h, o).astype(np.float32) * np.float32(2.0**exp)).reshape(
        h, w
    )
    o += 2 * w * h
    nrm = conf = None
    if typ & HAS_NORMAL:
        nrm = decode_normals(np.frombuffer(raw, "<i2", w * h * 2, o).reshape(h, w, 2))
        o += 4 * w * h
    if typ & HAS_CONF:
        conf = np.frombuffer(raw, np.uint8, w * h, o).reshape(h, w).astype(np.float32) * (
            cs / 255.0
        )
        o += w * h
    return DMap(depth, K, R, C, nrm, conf, ids, nm, dmin, dmax, (iw, ih))
