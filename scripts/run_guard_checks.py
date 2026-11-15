"""Run the leave-view-out guard checks (small, seconds) and record pass/fail in figures/data/guards.json.

Pipelines: the CPU toy point surface, a deliberately leaky fit (must FAIL: this is the mutation check), and the gsplat 2DGS
surface on the synthetic room if torch+CUDA+gsplat are available. The ETH3D pipes guard results come from docs/pipes-2dgs-fit.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from dcal.surface import check_leave_out, fit_surface, render  # noqa: E402
from scene_helpers import make_views  # noqa: E402


def main():
    out = []
    H = W = 24
    views = make_views(H, W, focal=24.0)
    rf = lambda s, t: render(s, t.K, t.T_cw, (H, W))      # noqa: E731
    for j in views:
        out.append({"pipeline": "toy point surface (CPU, synthetic room)", "view": j, "atol": 0.0,
                    **check_leave_out(fit_surface, rf, views, j, np.random.default_rng(j))})
    leaky = lambda vs, ex: fit_surface(vs, exclude=-1)      # ignores `exclude` (reads and fuses view j)   # noqa: E731
    for j in (0, 2, 4):
        out.append({"pipeline": "deliberately leaky fit (must fail)", "view": j, "atol": 0.0,
                    **check_leave_out(leaky, rf, views, j, np.random.default_rng(j))})
    try:
        import torch
        from dcal.gs2d import fit_surface_2dgs, render_2dgs
        if torch.cuda.is_available():
            H2, W2 = 48, 64
            v2 = make_views(H2, W2, focal=64.0)
            rf2 = lambda s, t: render_2dgs(s, t.K, t.T_cw, (H2, W2))      # noqa: E731
            for j in v2:
                out.append({"pipeline": "gsplat 2DGS surface (GPU, synthetic room)", "view": j, "atol": 0.0,
                            **check_leave_out(lambda v, e: fit_surface_2dgs(v, e), rf2, v2, j, np.random.default_rng(j))})
    except Exception as e:   # noqa: BLE001
        out.append({"pipeline": "gsplat 2DGS surface (GPU, synthetic room)", "skipped": repr(e)})
    (ROOT / "figures" / "data" / "guards.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
