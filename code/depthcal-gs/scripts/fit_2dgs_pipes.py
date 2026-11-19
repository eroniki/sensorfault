"""Fit a leave-one-view-out 2DGS on the ETH3D `pipes` scene and test the held-out view against its laser-scan depth.

For each held-out view j: surfels from the sparse ground-truth depth of the other 13 views, optimised with RGB + depth L1
(no densification). Reports, for j only: coverage of j's valid ground-truth pixels (alpha > 0.5), relative depth error
(median / p90), and RGB PSNR, before (0 iterations) and after optimisation. Also runs the leave-out guards on this real pipeline:
a poisoned view j must not be touched, and perturbing j's data must change the render by no more than the run-to-run noise.
Needs torch + CUDA + gsplat. Usage: python scripts/fit_2dgs_pipes.py /path/to/eth3d/pipes --iters 2000 --held 2 6 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dcal.eth3d import load_scene  # noqa: E402
from dcal.gs2d_fit import fit_2dgs, render_view  # noqa: E402
from dcal.surface import PoisonedView, ViewData, perturb_view  # noqa: E402


def evaluate(P, view):
    with torch.no_grad():
        rgb, depth, alpha = render_view(P, view.K, view.T_cw, view.rgb.shape[:2])
    gt = torch.as_tensor(view.depth, device=depth.device)
    valid = torch.isfinite(gt)
    cov = (alpha[valid] > 0.5).float().mean().item()
    m = valid & (alpha > 0.5)
    rel = ((depth[m] - gt[m]).abs() / gt[m]).cpu().numpy()
    mse = ((rgb - torch.as_tensor(view.rgb, device=rgb.device)) ** 2).mean().item()
    return {"coverage_of_valid_gt": round(cov, 3), "rel_err_median": round(float(np.median(rel)), 4) if rel.size else None,
            "rel_err_p90": round(float(np.percentile(rel, 90)), 4) if rel.size else None,
            "psnr_db": round(-10 * np.log10(mse), 2), "depth_render": depth.cpu().numpy(), "alpha": alpha.cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--guard-iters", type=int, default=150)
    ap.add_argument("--held", type=int, nargs="+", default=[2, 6, 10])
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--guard-only", action="store_true", help="skip the long fits; run only the leave-out guards")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = Path(a.scene_dir)
    ev = load_scene(base.parent, base.name, scale=a.scale)
    views = {v.view_id: ViewData(v.view_id, v.rgb, v.gt_depth, v.K, v.T_cw) for v in ev}
    print(f"{len(views)} views, {views[0].rgb.shape[:2]}, valid depth {np.isfinite(views[0].depth).mean():.3f}", flush=True)
    results = {}
    for j in a.held:
        t0 = time.time()
        res = {}
        if not a.guard_only:
            P0, _ = fit_2dgs(views, j, iters=0)
            before = evaluate(P0, views[j])
            P, info = fit_2dgs(views, j, iters=a.iters, log_every=max(a.iters // 4, 1))
            after = evaluate(P, views[j])
            res = {"before": {k: v for k, v in before.items() if not isinstance(v, np.ndarray)},
                   "after": {k: v for k, v in after.items() if not isinstance(v, np.ndarray)}, "surfels": info["n_surfels"],
                   "seconds": round(time.time() - t0, 1)}
            del P, P0
        # guards on the real pipeline
        poisoned = {k: (PoisonedView(k) if k == j else v) for k, v in views.items()}
        try:
            Pp, _ = fit_2dgs(poisoned, j, iters=a.guard_iters, seed=1)
            ref = evaluate(Pp, views[j])["depth_render"]
            res["poisoned_view_untouched"] = True
        except Exception as e:     # noqa: BLE001
            res["poisoned_view_untouched"] = False
            res["poison_error"] = repr(e)
            ref = None
        if ref is not None:
            alt = dict(views)
            alt[j] = perturb_view(views[j], np.random.default_rng(0))
            Pa, _ = fit_2dgs(alt, j, iters=a.guard_iters, seed=1)
            d_pert = evaluate(Pa, views[j])["depth_render"]
            Pb, _ = fit_2dgs(views, j, iters=a.guard_iters, seed=1)
            d_ctrl = evaluate(Pb, views[j])["depth_render"]
            both = np.isfinite(ref) & np.isfinite(d_pert) & np.isfinite(d_ctrl)
            for name, d in (("run_to_run_noise", d_ctrl), ("perturbed_j", d_pert)):
                diff = np.abs(ref[both] - d[both])
                res[f"{name}_abs_diff"] = {"median": float(np.median(diff)), "p99": float(np.percentile(diff, 99)),
                                          "max": float(diff.max())}
        results[str(j)] = res
        print(json.dumps({str(j): res}), flush=True)
        torch.cuda.empty_cache()
    results["peak_gpu_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
