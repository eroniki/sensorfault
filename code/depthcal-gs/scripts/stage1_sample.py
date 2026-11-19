"""Sample affine-invariant depth from a stage-1 checkpoint and compare with sparse laser ground truth (ETH3D pipes by default).

Loads the frozen SDXL base + LoRA structure exactly as training does, loads the trained parameters (LoRA + widened conv_in), draws
several DDIM samples per image, aligns them to each other (scale/shift), and shows RGB | median depth | sample spread | GT | error.
Metrics are in real depth units: the median prediction is fitted to the GT depth by least squares (scale and shift) on the valid GT
pixels; AbsRel = mean |fit - gt| / gt and delta1 = share of pixels with max(fit/gt, gt/fit) < 1.25. ETH3D is only evaluated here, never trained on.

  python scripts/stage1_sample.py --pretrained ~/datasets/sdxl-base --weights trained.pt --eth3d ~/datasets/eth3d/pipes --scene pipes --views 2 6 10 --out figures/peek
  --weights may be a full training checkpoint (ckpt_*.pt) or a small file {"trainable": ..., "step": N}; omit it to run the untrained base (a pipeline test).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dcal.eth3d import load_scene  # noqa: E402
from dcal.train.loop import sample_depth  # noqa: E402
from dcal.train.model import build_models, load_trainable_state, make_schedulers  # noqa: E402


def lsq_fit(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return sol[0], sol[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--embeds-cache", default=None)
    ap.add_argument("--eth3d", required=True, help="dir holding the extracted ETH3D scene (e.g. ~/datasets/eth3d/pipes)")
    ap.add_argument("--scene", default="pipes")
    ap.add_argument("--views", type=int, nargs="+", default=[2, 6, 10])
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10, help="DDIM steps of the figure panels")
    ap.add_argument("--steps-list", type=int, nargs="*", default=None, help="also report metrics for these step counts (for example 1 4 10 30)")
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    cfg = {"pretrained": a.pretrained, "rank": a.rank, "grad_ckpt": False, "vae": "fp32",
           "embeds_cache": a.embeds_cache or str(out / "empty_prompt_embeds.pt")}
    unet, vae, cond = build_models(cfg, dev)
    step = None
    if a.weights:
        d = torch.load(a.weights, map_location="cpu", weights_only=False)
        load_trainable_state(unet, d["trainable"])
        step = d.get("step")
        print(f"loaded trained parameters from {a.weights} (step {step})", flush=True)
    unet.eval()
    _, infer = make_schedulers({"prediction_type": "v_prediction", "zero_snr": True})
    views = load_scene(a.eth3d, a.scene, scale=8.0)
    summary = []
    step_counts = sorted(set((a.steps_list or []) + [a.steps]))
    for j in a.views:
        v = views[j]
        H, W = v.gt_depth.shape
        s = min(H, W)
        y0, x0 = (H - s) // 2, (W - s) // 2
        rgb = v.rgb[y0:y0 + s, x0:x0 + s]
        gt = v.gt_depth[y0:y0 + s, x0:x0 + s]
        valid = np.isfinite(gt)
        x = torch.from_numpy(rgb.astype(np.float32) * 2 - 1).permute(2, 0, 1)[None]
        x = F.interpolate(x, size=(a.res, a.res), mode="bilinear", align_corners=False)
        for nsteps in step_counts:
            preds = []
            for k in range(a.samples):
                p = sample_depth(unet, vae, infer, cond, x, dev, True, steps=nsteps, seed=k)
                preds.append(F.interpolate(p, size=(s, s), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
            preds = np.stack(preds)
            ref = preds[0]
            al = [ref]
            for p in preds[1:]:
                sc, sh = lsq_fit(p.ravel(), ref.ravel())
                al.append(sc * p + sh)
            al = np.stack(al)
            med, spread = np.median(al, 0), al.std(0)
            sc, sh = lsq_fit(med[valid], gt[valid])
            fit = sc * med + sh
            rel = np.abs(fit[valid] - gt[valid]) / gt[valid]
            ratio = np.maximum(fit[valid] / gt[valid], gt[valid] / np.maximum(fit[valid], 1e-6))
            single = [lsq_fit(q[valid], gt[valid]) for q in preds]
            single_absrel = float(np.mean([np.mean(np.abs(c1 * q[valid] + c0 - gt[valid]) / gt[valid]) for q, (c1, c0) in zip(preds, single)]))
            m = {"view": j, "step": step, "ddim_steps": nsteps, "absrel": float(rel.mean()), "median_rel_err": float(np.median(rel)),
                 "delta1": float((ratio < 1.25).mean()), "single_sample_absrel": single_absrel, "gt_valid_pct": float(100 * valid.mean()),
                 "fit_scale": float(sc), "fit_shift": float(sh), "mean_sample_spread": float(spread.mean())}
            summary.append(m)
            print(json.dumps(m), flush=True)
            if nsteps != a.steps:
                continue
            fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
            lo, hi = np.nanpercentile(gt, [1, 99])
            ax[0].imshow(np.clip(rgb, 0, 1)); ax[0].set_title(f"input RGB (ETH3D {a.scene} view {j}, centre crop)", fontsize=8)
            ax[1].imshow(fit, cmap="viridis", vmin=lo, vmax=hi)
            ax[1].set_title(f"predicted depth (median of {a.samples}, {nsteps} DDIM steps, scale/shift fitted to GT)", fontsize=8)
            ax[2].imshow(spread, cmap="magma"); ax[2].set_title("sample spread (std after alignment)", fontsize=8)
            from scipy.ndimage import maximum_filter
            gd = maximum_filter(np.where(valid, gt, -np.inf), size=3)
            gd = np.where(np.isfinite(gd), gd, np.nan)
            cm = plt.get_cmap("viridis").copy(); cm.set_bad("#d9d9d9")
            ax[3].imshow(gd, cmap=cm, vmin=lo, vmax=hi); ax[3].set_title(f"laser GT depth ({m['gt_valid_pct']:.1f}% valid; grey = none)", fontsize=8)
            err = np.where(valid, np.abs(fit - gt) / gt, -np.inf)
            ed = maximum_filter(err, size=3)
            ed = np.where(np.isfinite(ed), ed, np.nan)
            cm2 = plt.get_cmap("magma").copy(); cm2.set_bad("#d9d9d9")
            ax[4].imshow(ed, cmap=cm2, vmin=0, vmax=0.5); ax[4].set_title(f"relative error at GT pixels: AbsRel {m['absrel']:.3f}, d1 {m['delta1']:.2f}", fontsize=8)
            for q in ax:
                q.set_xticks([]); q.set_yticks([])
            fig.suptitle(f"Stage-1 checkpoint {'step ' + str(step) if step else '(UNTRAINED BASE, pipeline test)'}: held-out real scene, never used in training", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"peek_view{j}.png", dpi=130)
            plt.close(fig)
    (out / "peek_metrics.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
