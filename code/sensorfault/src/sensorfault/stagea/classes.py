"""Evaluation classes of entities (never used by the map).

From exact depth (protocol, Section 2.2): water = planar and below ``water_level`` (-3 m); roof =
upward normal and above ``roof_height`` (6 m); ground = other land. From the source images:
contrast quintile of land entities by mean gradient energy over the entity (quintile 1 = lowest).
A class with fewer than ``min_entities`` (50) entities carries no interval claim.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

WATER, ROOF, GROUND = 0, 1, 2
NAMES = {WATER: "water", ROOF: "roof", GROUND: "ground"}


def geometric_classes(
    ent, water_level: float = -3.0, roof_height: float = 6.0, planar_rms: float = 0.3
) -> np.ndarray:
    z = ent.p0[:, 2]
    up = ent.normal_class == 0
    cls = np.full(ent.M, GROUND)
    cls[up & (z > roof_height)] = ROOF
    cls[(ent.plane_rms < planar_rms) & (z < water_level)] = WATER
    return cls


def gradient_energy(gray: np.ndarray, step: int, h: int, w: int) -> np.ndarray:
    """Mean squared image gradient in the step x step block around each point-sampled pixel."""
    g = np.asarray(gray, float)
    gy, gx = np.gradient(g)
    e = gx * gx + gy * gy
    H, W = (h * step, w * step)
    e = e[:H, :W]
    return e.reshape(h, step, w, step).mean(axis=(1, 3))


def entity_contrast(ent, energy: Mapping[int, np.ndarray]) -> np.ndarray:
    """Per-entity mean gradient energy over all frames (NaN if no frame has an image)."""
    num = np.zeros(ent.M)
    cnt = np.zeros(ent.M)
    for fr, E in energy.items():
        m = ent.pixel_entity[fr]
        ok = (m >= 0) & np.isfinite(E)
        num += np.bincount(m[ok], E[ok], ent.M)
        cnt += np.bincount(m[ok], minlength=ent.M)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / cnt


def contrast_quintile(contrast: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """1..5 for land entities with a contrast value, 0 otherwise."""
    q = np.zeros(len(cls), np.int64)
    land = (cls != WATER) & np.isfinite(contrast)
    if land.sum() >= 5:
        edges = np.quantile(contrast[land], [0.2, 0.4, 0.6, 0.8])
        q[land] = 1 + np.searchsorted(edges, contrast[land], side="right")
    return q


def class_table(ent, cls: np.ndarray, quint: np.ndarray, min_entities: int = 50) -> list[dict]:
    rows = []
    for c, name in NAMES.items():
        m = cls == c
        rows.append(
            {
                "class": name,
                "entities": int(m.sum()),
                "area_m2": float(ent.area[m].sum()),
                "claim": bool(m.sum() >= min_entities),
            }
        )
    for k in range(1, 6):
        m = quint == k
        rows.append(
            {
                "class": f"contrast_q{k}",
                "entities": int(m.sum()),
                "area_m2": float(ent.area[m].sum()),
                "claim": bool(m.sum() >= min_entities),
            }
        )
    return rows
