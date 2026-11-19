"""ETH3D gate on the TITAN: unguided samples from a stage-1 checkpoint + leave-view-out 2DGS renders + the fault statistic.

Three steps sharing one cache directory, so the GPU steps can run while training is paused and the CPU step can be re-run freely:

  # 1. samples (GPU, SDXL; pause training first: it keeps about 18 GB of the card after its first validation)
  PYTHONPATH=src python scripts/gate_eth3d.py sample --pretrained $SDXL_DIR --weights ~/workspace/depthcal-runs/stage1/ckpt_0002000.pt \
      --eth3d ~/datasets/eth3d --scene pipes --cache runs/gate/pipes_s2000 --samples 10 --ddim-steps 1 4 10
  # 2. leave-j-out renders of every view from the other views' sparse GT depth (GPU, gsplat; about 2 min per view)
  PYTHONPATH=src python scripts/gate_eth3d.py render --eth3d ~/datasets/eth3d --scene pipes --cache runs/gate/pipes_s2000
  # 3. statistic, calibration and power (CPU only); --cache may be given several times (several scenes pool their nulls)
  PYTHONPATH=src python scripts/gate_eth3d.py stat --cache runs/gate/pipes_s2000 --ddim-steps 10 --out runs/gate/pipes_s2000/gate.json

Renders do not depend on the checkpoint: reuse them with ``--renders DIR`` (or copy the render_*.npz files) for a later checkpoint.
What the numbers mean and the pass rule are in ``src/dcal/gate.py``. ETH3D is only evaluated here, never trained on.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def crop_box(h: int, w: int):
    s = min(h, w)
    return (h - s) // 2, (w - s) // 2, s


def cmd_sample(a):
    import torch
    import torch.nn.functional as F

    from dcal.eth3d import load_scene
    from dcal.train.loop import sample_depth
    from dcal.train.model import build_models, load_trainable_state, make_schedulers

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    cfg = {"pretrained": a.pretrained, "rank": a.rank, "grad_ckpt": False, "vae": "fp32",
           "embeds_cache": a.embeds_cache or str(cache / "empty_prompt_embeds.pt")}
    unet, vae, cond = build_models(cfg, dev)
    d = torch.load(a.weights, map_location="cpu", weights_only=False)
    load_trainable_state(unet, d["trainable"])
    step = d.get("step")
    del d
    unet.eval()
    _, infer = make_schedulers({"prediction_type": "v_prediction", "zero_snr": True})
    views = load_scene(a.eth3d, a.scene, scale=a.scale)
    ids = a.views if a.views else [v.view_id for v in views]
    meta = {"scene": a.scene, "step": step, "weights": str(a.weights), "samples": a.samples, "res": a.res, "scale": a.scale}
    (cache / "sample_meta.json").write_text(json.dumps(meta, indent=1))
    for j in ids:
        v = views[j]
        y0, x0, s = crop_box(*v.gt_depth.shape)
        x = torch.from_numpy(v.rgb[y0:y0 + s, x0:x0 + s].astype(np.float32) * 2 - 1).permute(2, 0, 1)[None]
        x = F.interpolate(x, size=(a.res, a.res), mode="bilinear", align_corners=False)
        for n in a.ddim_steps:
            f = cache / f"samples_v{j:02d}_k{n:02d}.npz"
            if f.exists() and not a.overwrite:
                continue
            t0 = time.time()
            out = []
            with torch.no_grad():
                for k in range(a.samples):
                    p = sample_depth(unet, vae, infer, cond, x, dev, True, steps=n, seed=a.seed0 + k)
                    out.append(F.interpolate(p, size=(s, s), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
            np.savez_compressed(f, samples=np.stack(out).astype(np.float32), view=j, ddim_steps=n, step=step if step is not None else -1)
            print(f"view {j} ddim {n}: {a.samples} samples in {time.time() - t0:.0f} s", flush=True)


def cmd_render(a):
    import torch

    from dcal.eth3d import load_scene
    from dcal.gs2d_fit import fit_2dgs, render_view
    from dcal.surface import ViewData

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    ev = load_scene(a.eth3d, a.scene, scale=a.scale)
    views = {v.view_id: ViewData(v.view_id, v.rgb, v.gt_depth, v.K, v.T_cw) for v in ev}
    ids = a.views if a.views else sorted(views)
    for j in ids:
        f = cache / f"render_v{j:02d}.npz"
        if f.exists() and not a.overwrite:
            continue
        t0 = time.time()
        P, _ = fit_2dgs(views, j, iters=a.iters, seed=a.seed)          # never touches view j (guarded in tests/test_gs2d.py)
        with torch.no_grad():
            _, depth, alpha = render_view(P, views[j].K, views[j].T_cw, views[j].rgb.shape[:2])
        y0, x0, s = crop_box(*views[j].depth.shape)
        sl = (slice(y0, y0 + s), slice(x0, x0 + s))
        np.savez_compressed(f, render=depth.cpu().numpy()[sl].astype(np.float32), alpha=alpha.cpu().numpy()[sl].astype(np.float32),
                            gt=np.asarray(views[j].depth, np.float32)[sl], view=j)
        del P
        torch.cuda.empty_cache()
        print(f"view {j}: leave-out render in {time.time() - t0:.0f} s", flush=True)


def cmd_stat(a):
    from dcal.gate import make_windows, run_gate, verdict

    results = {}
    for n in a.ddim_steps:
        wins = []
        for ci, c in enumerate(a.cache):
            c = Path(c)
            rdir = Path(a.renders[ci]) if a.renders else c
            for f in sorted(c.glob(f"samples_v*_k{n:02d}.npz")):
                j = int(np.load(f)["view"])
                rf = rdir / f"render_v{j:02d}.npz"
                if not rf.exists():
                    print(f"no render for view {j} in {rdir}; skipped", flush=True)
                    continue
                S = np.load(f)["samples"][: a.samples]
                R = np.load(rf)
                # views of different scenes must not share an id (the null is leave-one-view-out)
                wins += make_windows(1000 * ci + j, S, R["render"], R["alpha"], R["gt"], g=a.grid, tiles=a.tiles)
        if not wins:
            print(f"ddim {n}: no windows", flush=True)
            continue
        t0 = time.time()
        res = run_gate(wins, kappas=tuple(a.kappas), alpha=a.alpha, rel_sigma_r=a.rel_sigma_r, area=a.area, seed=a.seed)
        res["ddim_steps"] = n
        res["seconds"] = round(time.time() - t0, 1)
        results[str(n)] = res
        print(f"\n== {n} DDIM steps: {res['n_windows']} windows from {res['n_views']} views ==")
        print("\n".join(verdict(res, kappa=str(float(a.verdict_kappa)))), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--weights", required=True, help="training checkpoint (ckpt_*.pt) or {'trainable': ..., 'step': N}")
    p.add_argument("--embeds-cache", default=None)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--ddim-steps", type=int, nargs="+", default=[10])
    p.add_argument("--seed0", type=int, default=0)
    q = sub.add_parser("render")
    q.add_argument("--iters", type=int, default=2000)
    q.add_argument("--seed", type=int, default=0)
    for x in (p, q):
        x.add_argument("--eth3d", required=True, help="dir holding extracted ETH3D scenes")
        x.add_argument("--scene", default="pipes")
        x.add_argument("--scale", type=float, default=8.0)
        x.add_argument("--views", type=int, nargs="*", default=None, help="default: every view of the scene")
        x.add_argument("--cache", required=True)
        x.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("stat")
    r.add_argument("--cache", action="append", required=True, help="sample cache dir; repeat to pool several scenes")
    r.add_argument("--renders", action="append", default=None, help="render dir per --cache (default: the cache itself)")
    r.add_argument("--ddim-steps", type=int, nargs="+", default=[10])
    r.add_argument("--samples", type=int, default=10)
    r.add_argument("--grid", type=int, default=24)
    r.add_argument("--tiles", type=int, default=2)
    r.add_argument("--kappas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    r.add_argument("--verdict-kappa", type=float, default=2.0)
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--rel-sigma-r", type=float, default=0.005, help="render sd as a fraction of depth (pipes fit: 0.1-0.6%%)")
    r.add_argument("--area", type=float, default=0.12)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "stat" and a.renders and len(a.renders) != len(a.cache):
        ap.error("give one --renders per --cache")
    {"sample": cmd_sample, "render": cmd_render, "stat": cmd_stat}[a.cmd](a)


if __name__ == "__main__":
    main()
