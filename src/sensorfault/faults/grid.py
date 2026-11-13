"""Expand configs/faults/grid.yaml into fault cells and scenario FaultSpec lists."""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence

from .injectors import FaultSpec


def expand_cells(
    grid: dict, kinds: Optional[Sequence[str]] = None, limit_per_kind: Optional[int] = None
) -> List[dict]:
    """Return list of {"kind", "params", "profile"} cells. Extrinsic drift is expanded as
    rotation-only and translation-only cells (separate axes) x profiles."""
    cells: List[dict] = []
    f = grid["fault_types"]
    want = lambda k: kinds is None or k in kinds
    per: dict = {}
    if want("extrinsic_drift"):
        e = f["extrinsic_drift"]
        for prof in e["profiles"]:
            for r in e["rotation_deg"]:
                per.setdefault("extrinsic_drift", []).append(
                    {"kind": "extrinsic_drift", "params": {"rotation_deg": r}, "profile": prof}
                )
            for t in e["translation_cm"]:
                per.setdefault("extrinsic_drift", []).append(
                    {"kind": "extrinsic_drift", "params": {"translation_cm": t}, "profile": prof}
                )
    if want("depth_bias_const"):
        b = f["depth_bias_const"]
        for fr, s in itertools.product(b["frac_of_median_range"], b["sign"]):
            per.setdefault("depth_bias_const", []).append(
                {
                    "kind": "depth_bias_const",
                    "params": {"frac_of_median_range": fr, "sign": s},
                    "profile": "step",
                }
            )
    if want("depth_bias_range"):
        b = f["depth_bias_range"]
        for be, s in itertools.product(b["beta"], b["sign"]):
            per.setdefault("depth_bias_range", []).append(
                {"kind": "depth_bias_range", "params": {"beta": be, "sign": s}, "profile": "step"}
            )
    if want("multipath"):
        m = f["multipath"]
        for regime in ("tabletop", "room"):
            for A in m["amplitude_m"][regime]:
                per.setdefault("multipath", []).append(
                    {
                        "kind": "multipath",
                        "params": {"amplitude_m": A, "regime": regime},
                        "profile": "step",
                    }
                )
    if want("transient"):
        t = f["transient"]
        for fr in t["fraction"]:
            per.setdefault("transient", []).append(
                {
                    "kind": "transient",
                    "params": {
                        "fraction": fr,
                        "burst_min": t["burst_frames"][0],
                        "burst_max": t["burst_frames"][1],
                        "mode": "offset",
                        "magnitude_m": t.get("offset_m", 0.1),
                    },
                    "profile": "step",
                }
            )
    for v in per.values():
        cells.extend(v[:limit_per_kind] if limit_per_kind else v)
    return cells


def scenario_specs(
    cell: dict,
    sensor_ids: Sequence[str],
    faulty_idx: Sequence[int],
    *,
    onset_frac: float = 0.2,
    same_direction: bool = True,
) -> List[FaultSpec]:
    """Build FaultSpecs for one cell applied to the sensors at ``faulty_idx``.
    same_direction=True forces identical sign (correlated majority-fault case); False lets each
    faulty sensor draw its own sign (independent)."""
    specs = []
    for j, i in enumerate(faulty_idx):
        p = dict(cell["params"])
        if "sign" in p and not same_direction and j % 2 == 1:
            p["sign"] = -p["sign"]
        specs.append(
            FaultSpec(
                cell["kind"],
                sensor_ids[i],
                fault_id=j,
                params=p,
                onset_frac=onset_frac,
                profile=cell.get("profile", "step"),
            )
        )
    return specs
