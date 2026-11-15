"""Minimal reader for scanline OpenEXR files with ZIP (zlib) compression and FLOAT/HALF channels.

Enough for MatrixCity's float32 depth maps without extra dependencies. Not a general EXR reader:
only compression 2 (ZIPS, 1 line) and 3 (ZIP, 16 lines), scanline images, increasing line order.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_PT = {1: (np.float16, 2), 2: (np.float32, 4)}  # pixel type -> dtype, bytes (0 = UINT not used)


def _header(b: bytes):
    if b[:4] != b"\x76\x2f\x31\x01":
        raise ValueError("not an OpenEXR file")
    p, attrs = 8, {}
    while b[p] != 0:
        e = b.index(b"\0", p)
        name = b[p:e].decode()
        p = e + 1
        e = b.index(b"\0", p)
        p = e + 1
        size = struct.unpack("<i", b[p : p + 4])[0]
        p += 4
        attrs[name] = b[p : p + size]
        p += size
    return attrs, p + 1


def _channels(raw: bytes):
    out, p = [], 0
    while raw[p] != 0:
        e = raw.index(b"\0", p)
        name = raw[p:e].decode()
        ptype = struct.unpack("<i", raw[e + 1 : e + 5])[0]
        out.append((name, ptype))
        p = e + 1 + 16
    return out


def _unpredict_deinterleave(t: bytes) -> bytes:
    # undo the predictor: b[i] = b[i-1] + t[i] - 128 (mod 256), b[0] = t[0]
    d = np.frombuffer(t, dtype=np.uint8).astype(np.int64)
    a = ((np.cumsum(np.concatenate([[d[0]], d[1:] - 128]))) % 256).astype(np.uint8)
    n = len(a)
    half = (n + 1) // 2
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = a[:half]
    out[1::2] = a[half:]
    return out.tobytes()


def read_exr(path: str | Path, channel: str | None = None) -> np.ndarray | dict:
    """Read a ZIP-compressed scanline EXR; returns one channel (h, w) or a dict of all channels."""
    b = Path(path).read_bytes()
    attrs, p = _header(b)
    comp = attrs["compression"][0]
    lines = {2: 1, 3: 16}.get(comp)
    if lines is None:
        raise ValueError(f"unsupported EXR compression {comp}")
    x0, y0, x1, y1 = struct.unpack("<4i", attrs["dataWindow"])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    chans = _channels(attrs["channels"])
    n_chunks = (h + lines - 1) // lines
    offsets = struct.unpack(f"<{n_chunks}Q", b[p : p + 8 * n_chunks])
    data = {c: np.empty((h, w), dtype=_PT[t][0]) for c, t in chans}
    for off in offsets:
        y, size = struct.unpack("<ii", b[off : off + 8])
        raw = b[off + 8 : off + 8 + size]
        nl = min(lines, y1 - y + 1)
        expected = nl * w * sum(_PT[t][1] for _, t in chans)
        buf = raw if size == expected else _unpredict_deinterleave(zlib.decompress(raw))
        q = 0
        for r in range(nl):
            for c, t in chans:
                dt, nb = _PT[t]
                data[c][y - y0 + r] = np.frombuffer(buf, dtype=dt, count=w, offset=q)
                q += w * nb
    if channel is not None:
        return data[channel]
    return data
