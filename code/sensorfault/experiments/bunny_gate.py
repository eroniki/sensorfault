"""Stanford bunny check of the view gate (non-planar scene): runs the golden-test cases and writes
``results/bunny/gate.json`` plus small depth maps of the faulty view.

Same setup as ``tests/gs/test_bunny_golden.py``: six 32x32 views of the bunny on a 60 deg arc, 5 mm
depth noise, +5 cm bias on pixels (8:24, 8:24) of the faulty view, ``view_gate`` with the numpy
reference backend (stride 2), ``round_cap=1``, ``sigma=0.01``. Cases: fault on v3 and clean (seed 0,
the golden cases), clean seed 1 and fault on v5 (seed 0), the known failures. Runs the cases in
parallel processes (about 2-3 min on 4 cores).

Run: python3 experiments/bunny_gate.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bunny"
CASES = {
    "faulty_v3_seed0": dict(fault_view=3, seed=0),
    "clean_seed0": dict(seed=0),
    "clean_seed1": dict(seed=1),
    "faulty_v5_seed0": dict(fault_view=5, seed=0),
}
GATE_CFG = {"round_cap": 1, "sigma": 0.01}
STRIDE = 2


def run_case(name: str) -> tuple[str, dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from sensorfault.eval.gate_loop import holm_family_size
    from sensorfault.gs.fusers import ViewGate
    from sensorfault.gs.reference import NumpyReferenceBackend
    from sensorfault.sim.bunny import BunnyConfig, make_bunny_views

    bv = make_bunny_views(BunnyConfig(**CASES[name]))
    res = ViewGate(backend=NumpyReferenceBackend(stride=STRIDE)).fuse(bv.frames, GATE_CFG)
    pv = res.meta["history"][0]["pvals"]
    return name, {
        **CASES[name],
        "excluded": sorted(v for v, f in res.excluded_sensors.items() if f),
        "n_surfels": len(res.points),
        "neg_log10p": {v: -float(np.log10(max(p, 1e-300))) for v, p in pv.items()},
        "holm_family": int(holm_family_size(len(bv.frames), GATE_CFG["round_cap"])),
    }


def scene_facts() -> dict:
    from sensorfault.sim.bunny import BunnyConfig, make_bunny_views

    bv = make_bunny_views(BunnyConfig(fault_view=3, seed=0))
    cfg = bv.cfg
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "views_faulty_v3_seed0.npz",
        depth=np.stack([f.depth for f in bv.frames]),
        depth_clean=np.stack(bv.depth_clean),
        rgb=np.stack(bv.rgb),
        fault_mask=np.stack(bv.fault_mask),
        T_wc=np.stack([f.T_wc for f in bv.frames]),
        K=bv.frames[0].K,
    )
    lo, hi = bv.mesh.bounds
    return {
        "n_views": cfg.n_views,
        "hw": cfg.hw,
        "fov_deg": cfg.fov_deg,
        "radius_m": cfg.radius,
        "noise_m": cfg.noise,
        "fault_bias_m": cfg.fault_bias,
        "fault_region": list(cfg.fault_region),
        "bunny_extent_m": [float(x) for x in hi - lo],
        "n_fault_pixels_v3": int(bv.fault_mask[3].sum()),
        "n_valid_pixels": [int((d > 0).sum()) for d in bv.depth_clean],
        "camera_positions_m": [[float(x) for x in f.T_wc[:3, 3]] for f in bv.frames],
        "n_faces": len(bv.mesh.faces),
    }


def main() -> None:
    facts = scene_facts()
    with ProcessPoolExecutor(max_workers=min(4, len(CASES))) as ex:
        cases = dict(ex.map(run_case, CASES))
    (OUT / "gate.json").write_text(
        json.dumps(
            {"scene": facts, "gate_cfg": {**GATE_CFG, "stride": STRIDE}, "cases": cases}, indent=1
        )
        + "\n"
    )
    for k, c in cases.items():
        print(k, c["excluded"], {v: round(x, 2) for v, x in c["neg_log10p"].items()})


if __name__ == "__main__":
    main()
