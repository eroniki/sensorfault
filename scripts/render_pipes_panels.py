"""Render image panels from the leave-one-view-out 2DGS on ETH3D pipes (GPU; re-fits only when parameters are not saved yet).

Per held-out view j: input RGB | rendered RGB | rendered depth | ground-truth depth (valid pixels only) | absolute depth error,
depth maps on one shared colour scale, errors on one shared scale, with colourbars. Guard cases (view 6): the fit with view j replaced by a
poisoned object (never touched) and a deliberately LEAKY fit that includes view j, same layout.

Surfel parameters are saved outside the repository (default ~/datasets/eth3d/pipes/surfels) so renders can be redone without training.
Usage: python scripts/render_pipes_panels.py /path/to/eth3d/pipes/pipes [--iters 2000] [--held 2 6 10] [--guard-view 6]
"""
from __future__ import annotations

import argparse
import json
import textwrap
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dcal.eth3d import load_scene  # noqa: E402
from dcal.gs2d_fit import GSParams, fit_2dgs, render_view  # noqa: E402
from dcal.surface import PoisonedView, ViewData  # noqa: E402

OUT = ROOT / "figures" / "renders"
DEPTH_RANGE = None
ERR_MAX = 0.03     # metres, shared error scale (clipped)


def save_params(P, path):
    torch.save({k: getattr(P, k).detach().cpu() for k in ("means", "quats", "log_scales", "logit_opac", "colors")}, path)


def load_params(path, device="cuda"):
    d = torch.load(path, map_location=device)
    return GSParams(**{k: v.to(device) for k, v in d.items()})


def dilate_nan(a, size=3):
    """Display only: grow valid pixels so that 9%-dense sparse depth is visible at figure scale."""
    valid = np.isfinite(a)
    filled = np.where(valid, a, -np.inf)
    g = maximum_filter(filled, size=size)
    return np.where(np.isfinite(g), g, np.nan)


def metrics(P, v):
    with torch.no_grad():
        rgb, depth, alpha = render_view(P, v.K, v.T_cw, v.rgb.shape[:2])
    gt = np.asarray(v.depth)
    d = depth.cpu().numpy()
    a = alpha.cpu().numpy()
    valid = np.isfinite(gt)
    cov = float((a[valid] > 0.5).mean())
    m = valid & (a > 0.5)
    rel = np.abs(d[m] - gt[m]) / gt[m]
    mse = float(((rgb.cpu().numpy() - np.asarray(v.rgb)) ** 2).mean())
    return {"coverage_of_valid_gt": round(cov, 3), "rel_err_median": round(float(np.median(rel)), 4),
            "rel_err_p90": round(float(np.percentile(rel, 90)), 4), "psnr_db": round(-10 * np.log10(mse), 2)}, \
        rgb.cpu().numpy(), d, a


def panel(v, rgb_r, depth_r, alpha_r, m, title, foot, path, dr):
    gt = np.asarray(v.depth)
    valid = np.isfinite(gt)
    show_alpha = alpha_r > 0.5
    err = np.where(valid & show_alpha, np.abs(depth_r - gt), np.nan)
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
    ax[0].imshow(np.clip(v.rgb, 0, 1))
    ax[0].set_title("input RGB (held-out view)" if "LEAKY" not in title else "input RGB (view j: used in this fit!)", fontsize=8)
    ax[1].imshow(np.clip(rgb_r, 0, 1))
    ax[1].set_title(f"rendered RGB (PSNR {m['psnr_db']} dB)", fontsize=8)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    im = ax[2].imshow(np.where(show_alpha, depth_r, np.nan), cmap=cmap, vmin=dr[0], vmax=dr[1])
    ax[2].set_title(f"rendered depth (alpha>0.5)\ncovers {100 * show_alpha.mean():.0f}% of pixels", fontsize=7.6)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02, label="depth (m)")
    im = ax[3].imshow(dilate_nan(np.where(valid, gt, np.nan)), cmap=cmap, vmin=dr[0], vmax=dr[1])
    ax[3].set_title(f"ground-truth depth (laser), valid pixels only\n{100 * valid.mean():.1f}% valid; grey = no GT; dots dilated for display", fontsize=7.6)
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.02, label="depth (m)")
    ecm = plt.get_cmap("magma").copy()
    ecm.set_bad("#d9d9d9")
    im = ax[4].imshow(dilate_nan(err), cmap=ecm, vmin=0, vmax=ERR_MAX)
    ax[4].set_title(f"|rendered - GT| at valid GT pixels\nmedian {100 * m['rel_err_median']:.2f}%, p90 {100 * m['rel_err_p90']:.2f}% (relative); "
                    f"covers {100 * m['coverage_of_valid_gt']:.1f}% of GT", fontsize=7.6)
    fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.02, label=f"abs error (m), clipped at {ERR_MAX}")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=10.5, y=0.99)
    lines = textwrap.wrap(foot, 235)
    fig.text(0.008, 0.008, "\n".join(lines), fontsize=6.6, color="#444", va="bottom")
    fig.tight_layout(rect=(0, 0.035 * len(lines) + 0.01, 1, 0.93), w_pad=1.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--held", type=int, nargs="+", default=[2, 6, 10])
    ap.add_argument("--guard-view", type=int, default=6)
    ap.add_argument("--surfels", default=str(Path.home() / "datasets" / "eth3d" / "pipes" / "surfels"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sd = Path(a.surfels)
    sd.mkdir(parents=True, exist_ok=True)
    base = Path(a.scene_dir)
    ev = load_scene(base.parent, base.name, scale=8.0)
    views = {v.view_id: ViewData(v.view_id, v.rgb, v.gt_depth, v.K, v.T_cw) for v in ev}
    allgt = np.concatenate([v.depth[np.isfinite(v.depth)] for v in views.values()])
    dr = (float(np.percentile(allgt, 1)), float(np.percentile(allgt, 99)))
    recorded = json.load(open(ROOT / "docs" / "pipes-2dgs-fit.json"))
    foot_base = ("DATA: REAL ETH3D 'pipes' scene (14 views, 1/8 scale 778x518). Ground-truth depth = laser scan, valid at 9-13% of pixels (per view). The surface is built from the OTHER 13 views' "
                 "sparse ground-truth depth + RGB (stand-in for model predictions; no depth network involved), {it} iterations of RGB+depth L1, no densification, seed 0. "
                 "Depth maps share one colour scale; errors share one scale. ")
    log = {}

    def get(j, case, iters, vs, exclude):
        p = sd / f"j{j}_{case}_{iters}.pt"
        if p.exists():
            return load_params(p), False
        t0 = time.time()
        P, _ = fit_2dgs(vs, exclude, iters=iters, seed=0 if case != "poisoned" else 0)
        save_params(P, p)
        log[f"{case}_{j}_seconds"] = round(time.time() - t0, 1)
        return P, True

    for j in a.held:
        P, fitted = get(j, "heldout", a.iters, views, j)
        m, rgb, d, al = metrics(P, views[j])
        rec = recorded[str(j)]["after"]
        log[f"heldout_{j}"] = {"this_run": m, "recorded_earlier_run": {k: rec[k] for k in m}}
        panel(views[j], rgb, d, al, m, f"ETH3D pipes: held-out view {j} (view {j} NOT used in the fit; {a.iters}-iteration optimised 2DGS)",
              foot_base.format(it=a.iters) + f"This run vs the earlier recorded run (non-deterministic CUDA): coverage {m['coverage_of_valid_gt']} vs {rec['coverage_of_valid_gt']}, "
              f"median error {m['rel_err_median']} vs {rec['rel_err_median']}. Source: scripts/render_pipes_panels.py.",
              OUT / f"pipes_view{j}.png", dr)
        print("held-out", j, m, flush=True)
        del P
        torch.cuda.empty_cache()
    g = a.guard_view
    poisoned = {k: (PoisonedView(k) if k == g else v) for k, v in views.items()}
    P, _ = get(g, "poisoned", a.iters, poisoned, g)
    m, rgb, d, al = metrics(P, views[g])
    log["poisoned"] = m
    panel(views[g], rgb, d, al, m, f"GUARD case 1, POISONED view {g}: the fit received an object that raises on any access to view {g}'s data (it ran, so view {g} was never read)",
          foot_base.format(it=a.iters) + "Compare with the held-out panel of the same view: the result should be the same up to CUDA non-determinism. Source: scripts/render_pipes_panels.py.",
          OUT / "pipes_guard_poisoned.png", dr)
    del P
    torch.cuda.empty_cache()
    P, _ = get(g, "leaky", a.iters, views, -1)      # exclude=-1: view g's depth and RGB ARE used
    m, rgb, d, al = metrics(P, views[g])
    log["leaky"] = m
    panel(views[g], rgb, d, al, m, f"GUARD case 2, LEAKY fit: view {g}'s own ground-truth depth and RGB are INCLUDED in the fit (what the guard exists to prevent)",
          foot_base.format(it=a.iters).replace("the OTHER 13 views'", "ALL 14 views' (including the evaluated one)") +
          "The near-zero error is circular: the surface was fitted to the very pixels it is scored on. Source: scripts/render_pipes_panels.py.",
          OUT / "pipes_guard_leaky.png", dr)
    (ROOT / "figures" / "data" / "pipes_panels_metrics.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
