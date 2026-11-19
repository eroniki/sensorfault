"""Strip table, 8 m entities, 64 m split tiles and the calibration / held-out split.

Split rule: 64 m tiles are grouped in square groups of ``group`` x ``group`` tiles separated by
``buffer`` tiles in each direction. Groups alternate between calibration and held-out in a
checkerboard; buffer tiles are not scored. A held-out tile is therefore never 8-adjacent to a
calibration tile. With group=1 the usable share is 1/4, with group=2 (default) it is 4/9.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

CAL, HELD, BUFFER = 0, 1, 2
ROLE_NAMES = {CAL: "calibration", HELD: "held_out", BUFFER: "buffer"}


def cell_index(xy: np.ndarray, side: float, origin: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """(N, 2) ground coordinates -> (N, 2) integer cell indices of a square grid of ``side`` metres."""
    xy = np.asarray(xy, float)
    return np.floor((xy - np.asarray(origin, float)) / side).astype(np.int64)


def tile_role(ij: np.ndarray, group: int = 2, buffer: int = 1) -> np.ndarray:
    """(N, 2) split-tile indices -> (N,) role in {CAL, HELD, BUFFER}."""
    ij = np.asarray(ij, np.int64)
    p = group + buffer
    local = np.mod(ij, p)
    used = np.all(local < group, axis=-1)
    sup = np.floor_divide(ij, p)
    parity = np.mod(sup[..., 0] + sup[..., 1], 2)
    return np.where(used, np.where(parity == 0, CAL, HELD), BUFFER)


def tile_key(ij: np.ndarray) -> np.ndarray:
    """(N, 2) indices -> (N,) int64 keys, unique for |i|, |j| < 2**31."""
    ij = np.asarray(ij, np.int64)
    return (ij[..., 0] << 32) ^ (ij[..., 1] & 0xFFFFFFFF)


def touching_roles(ij_all: np.ndarray, roles: np.ndarray) -> bool:
    """True if any held-out tile is 8-adjacent to a calibration tile (the split is then invalid)."""
    pos = {tuple(t): r for t, r in zip(np.asarray(ij_all).tolist(), np.asarray(roles).tolist())}
    for (i, j), r in pos.items():
        if r != HELD:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if pos.get((i + di, j + dj)) == CAL:
                    return True
    return False


def assign_strips(views: Sequence[Mapping]) -> list[dict]:
    """Add ``strip_id`` (dense int) and ``strip_pos`` (0.. along the strip) to each view.

    Views need ``strip`` (hashable key, e.g. (yaw, cross-track)) and ``along`` (float)."""
    by = defaultdict(list)
    for k, v in enumerate(views):
        by[v["strip"]].append(k)
    keys = sorted(by, key=lambda s: (s[0], s[1]) if isinstance(s, tuple) else s)
    out = [dict(v) for v in views]
    for sid, key in enumerate(keys):
        members = sorted(by[key], key=lambda k: views[k]["along"])
        for pos, k in enumerate(members):
            out[k]["strip_id"] = sid
            out[k]["strip_pos"] = pos
    return out


def strip_table(
    views: Sequence[Mapping], footprint_along: Mapping[int, float] | None = None
) -> list[dict]:
    """One row per strip: key, frames, along-track spacing (median), overlap (1 - spacing/footprint).

    ``views`` as returned by :func:`assign_strips`; ``footprint_along`` maps frame -> along-track
    ground extent of that frame in metres (optional)."""
    by = defaultdict(list)
    for v in views:
        by[v["strip_id"]].append(v)
    rows = []
    for sid in sorted(by):
        vs = sorted(by[sid], key=lambda v: v["strip_pos"])
        al = np.array([v["along"] for v in vs])
        sp = float(np.median(np.abs(np.diff(al)))) if len(al) > 1 else float("nan")
        row = {
            "strip_id": sid,
            "key": vs[0]["strip"],
            "frames": len(vs),
            "first_frame": vs[0]["frame"],
            "last_frame": vs[-1]["frame"],
            "spacing_m": sp,
        }
        if footprint_along is not None and len(vs) > 1:
            fp = float(np.median([footprint_along[v["frame"]] for v in vs]))
            row["footprint_along_m"] = fp
            row["overlap"] = 1.0 - sp / fp if fp > 0 else float("nan")
        rows.append(row)
    return rows


def strip_table_markdown(rows: Sequence[Mapping]) -> str:
    cols = [
        "strip_id",
        "key",
        "frames",
        "first_frame",
        "last_frame",
        "spacing_m",
        "footprint_along_m",
        "overlap",
    ]
    cols = [c for c in cols if any(c in r for r in rows)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            x = r.get(c, "")
            cells.append(f"{x:.3g}" if isinstance(x, float) else str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
