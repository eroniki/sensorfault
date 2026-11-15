"""Train one 2DGS model (gsplat) on MatrixCity block 3 RGB and evaluate on its test split.

Inputs (from experiments/rgb_prepare.py): rgb_block3_half.npz (960x540 images and NeRF-convention
camera-to-world poses) and rgb_block3_init.npz (initial points: the E1 voxel surfels with colours).
Photometric loss L1 + 0.2 D-SSIM, with the 2DGS normal-consistency and distortion regularisers;
default gsplat densification. A temperature guard queries nvidia-smi and lm-sensors every 100
iterations: it pauses while the GPU or the CPU (Tctl) is above the limit, resumes below the resume
level, and stops (saving what it has) if the total pause exceeds a budget.
Run: /home/eiki/.venvs/sensorfault-gs/bin/python experiments/rgb_train_2dgs.py --data /home/eiki/data/matrixcity --out results/rgb
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy

torch.set_num_threads(4)


# --------------------------------------------------------------------------------- temperatures


def gpu_temp() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def cpu_temp() -> float:
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Tctl:\s*\+?([\d.]+)", out)
        return float(m.group(1)) if m else float("nan")
    except Exception:
        return float("nan")


class TempGuard:
    def __init__(self, limit=80.0, resume=72.0, budget_s=900.0, log=None):
        self.limit, self.resume, self.budget, self.paused, self.log = limit, resume, budget_s, 0.0, log
        self.max_gpu, self.max_cpu, self.samples = 0.0, 0.0, []

    def check(self, it: int) -> bool:
        """Returns False if training must stop (pause budget exhausted)."""
        g, c = gpu_temp(), cpu_temp()
        self.max_gpu, self.max_cpu = max(self.max_gpu, g), max(self.max_cpu, c)
        self.samples.append((it, time.time(), g, c))
        if g > self.limit or c > self.limit:
            self.log(f"iter {it}: GPU {g:.0f} C, CPU {c:.0f} C above {self.limit:.0f} C, pausing")
            torch.cuda.synchronize()
            t0 = time.time()
            while True:
                time.sleep(20)
                g, c = gpu_temp(), cpu_temp()
                self.samples.append((it, time.time(), g, c))
                if g < self.resume and c < self.resume:
                    break
                if self.paused + (time.time() - t0) > self.budget:
                    self.paused += time.time() - t0
                    self.log(f"pause budget exhausted at iter {it} (GPU {g:.0f} C, CPU {c:.0f} C): stopping")
                    return False
            self.paused += time.time() - t0
            self.log(f"resumed at GPU {g:.0f} C, CPU {c:.0f} C after {time.time() - t0:.0f} s")
        return True


# --------------------------------------------------------------------------------- image metrics


def ssim(a: torch.Tensor, b: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM of two (B,3,H,W) images in [0,1], Gaussian window (Wang et al. 2004 constants)."""
    x = torch.arange(win, device=a.device, dtype=a.dtype) - win // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    k = g.expand(3, 1, win, win)
    mu_a = F.conv2d(a, k, padding=win // 2, groups=3)
    mu_b = F.conv2d(b, k, padding=win // 2, groups=3)
    s_aa = F.conv2d(a * a, k, padding=win // 2, groups=3) - mu_a**2
    s_bb = F.conv2d(b * b, k, padding=win // 2, groups=3) - mu_b**2
    s_ab = F.conv2d(a * b, k, padding=win // 2, groups=3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10 * torch.log10(F.mse_loss(a, b)))


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    path.write_bytes(png + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------------- model


def viewmats_from_c2w(c2w: np.ndarray) -> torch.Tensor:
    """NeRF (x right, y up, z back) camera-to-world -> OpenCV world-to-camera."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    cv = c2w @ flip
    return torch.tensor(np.linalg.inv(cv), dtype=torch.float32)


def quat_from_normal(n: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (w,x,y,z) rotating +z onto n."""
    z = torch.tensor([0.0, 0.0, 1.0], device=n.device).expand_as(n)
    axis = torch.cross(z, n, dim=-1)
    s = axis.norm(dim=-1, keepdim=True)
    c = (z * n).sum(-1, keepdim=True)
    half = torch.atan2(s, c) / 2
    axis = torch.where(s > 1e-8, axis / s.clamp_min(1e-8), torch.tensor([1.0, 0.0, 0.0], device=n.device).expand_as(n))
    return torch.cat([torch.cos(half), axis * torch.sin(half)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/eiki/data/matrixcity")
    ap.add_argument("--out", default="results/rgb")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--sh", type=int, default=3)
    ap.add_argument("--limit", type=float, default=80.0)
    ap.add_argument("--figure-frames", default="88,59,58")
    ap.add_argument("--max-gaussians", type=int, default=1_500_000, help="stop densifying above this count (8 GB GPU)")
    ap.add_argument("--grow-grad2d", type=float, default=4e-4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "train_log.txt").open("w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    dev = "cuda"
    D = np.load(Path(a.data) / "rgb_block3_half.npz")
    I = np.load(Path(a.data) / "rgb_block3_init.npz")
    W, H, fov = int(D["width"]), int(D["height"]), float(D["fov_x"])
    f = 0.5 * W / math.tan(0.5 * fov)
    K = torch.tensor([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], dtype=torch.float32, device=dev)
    train_img = torch.tensor(D["train_images"], device=dev)  # uint8 (N,H,W,3)
    train_vm = torch.stack([viewmats_from_c2w(c) for c in D["train_c2w"]]).to(dev)
    test_img = torch.tensor(D["test_images"], device=dev)
    test_vm = torch.stack([viewmats_from_c2w(c) for c in D["test_c2w"]]).to(dev)
    n_train = len(train_img)

    means = torch.tensor(I["means"], device=dev)
    # normals: towards the mean camera direction for initialisation; 2DGS refines them
    cam_c = torch.tensor(D["train_c2w"][:, :3, 3], dtype=torch.float32, device=dev).mean(0)
    nrm = F.normalize(cam_c - means, dim=-1)
    N = len(means)
    sh_k = (a.sh + 1) ** 2
    colors = torch.zeros(N, sh_k, 3, device=dev)
    colors[:, 0] = (torch.tensor(I["colors"], device=dev) - 0.5) / 0.28209479177387814
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.log(torch.full((N, 3), 1.5, device=dev))),
        "quats": torch.nn.Parameter(quat_from_normal(nrm)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(colors[:, :1].contiguous()),
        "shN": torch.nn.Parameter(colors[:, 1:].contiguous()),
    })
    scene_scale = float(torch.linalg.norm(torch.tensor(D["train_c2w"][:, :3, 3]) - torch.tensor(D["train_c2w"][:, :3, 3]).mean(0), dim=1).max()) * 1.1
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / a.iters))
    strategy = DefaultStrategy(verbose=False, key_for_gradient="gradient_2dgs", refine_stop_iter=int(0.75 * a.iters),
                               grow_grad2d=a.grow_grad2d)
    strategy.check_sanity(params, opt)
    state = strategy.initialize_state(scene_scale=scene_scale)
    guard = TempGuard(limit=a.limit, log=log)
    log(json.dumps({"train_views": n_train, "test_views": len(test_img), "W": W, "H": H, "init_gaussians": N, "iters": a.iters,
                    "scene_scale": scene_scale, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}))

    def render(vm, sh_deg):
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out_ = rasterization_2dgs(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=vm[None], Ks=K[None],
            width=W, height=H, sh_degree=sh_deg, render_mode="RGB+ED", packed=False,
            near_plane=1.0, far_plane=5000.0,
        )
        return out_

    t_start = time.time()
    rng = np.random.default_rng(0)
    order = rng.permutation(n_train)
    stopped = False
    for it in range(a.iters):
        if it % 100 == 0 and not guard.check(it):
            stopped = True
            break
        i = int(order[it % n_train])
        if it % n_train == n_train - 1:
            order = rng.permutation(n_train)
        sh_deg = min(it // 1000, a.sh)
        rc, ra, rn, sn, rd, rmed, info = render(train_vm[i], sh_deg)
        img = rc[0, ..., :3].clamp(0, 1)
        gt = train_img[i].float() / 255.0
        strategy.step_pre_backward(params, opt, state, it, info)
        l1 = (img - gt).abs().mean()
        ss = ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        if it > a.iters // 3:  # 2DGS regularisers after the first third
            nloss = (1 - (rn[0] * sn.reshape(rn[0].shape)).sum(-1)).mean()
            loss = loss + 0.05 * nloss + 1e-2 * rd.mean()
        loss.backward()
        # memory cap: above max_gaussians stop growing (pruning continues); resume growing below it
        strategy.grow_grad2d = float("inf") if len(params["means"]) > a.max_gaussians else a.grow_grad2d
        strategy.step_post_backward(params, opt, state, it, info, packed=False)
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if it % 1000 == 0:
            log(f"iter {it}: loss {loss.item():.4f}, psnr(train view) {psnr(img, gt):.2f}, gaussians {len(params['means'])}, "
                f"GPU {guard.samples[-1][2]:.0f} C, CPU {guard.samples[-1][3]:.0f} C, {time.time() - t_start:.0f} s")
    train_s = time.time() - t_start

    # evaluation on the test split, and renders for the figure views
    with torch.no_grad():
        ps, ss_ = [], []
        for k in range(len(test_img)):
            rc, *_ = render(test_vm[k], a.sh)
            img = rc[0, ..., :3].clamp(0, 1)
            gt = test_img[k].float() / 255.0
            ps.append(psnr(img, gt))
            ss_.append(float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
            if k < 3:
                write_png(out / f"test_{int(D['test_frames'][k]):04d}_render.png", (img.cpu().numpy() * 255).astype(np.uint8))
                write_png(out / f"test_{int(D['test_frames'][k]):04d}_gt.png", test_img[k].cpu().numpy())
        frames = [int(x) for x in a.figure_frames.split(",")]
        fig = {}
        for fr in frames:
            k = int(np.where(D["train_frames"] == fr)[0][0])
            rc, ra, rn, sn, rd, rmed, info = render(train_vm[k], a.sh)
            img = rc[0, ..., :3].clamp(0, 1)
            gt = train_img[k].float() / 255.0
            write_png(out / f"train_{fr:04d}_render.png", (img.cpu().numpy() * 255).astype(np.uint8))
            write_png(out / f"train_{fr:04d}_gt.png", train_img[k].cpu().numpy())
            err = (img - gt).abs().mean(-1).cpu().numpy()
            np.save(out / f"train_{fr:04d}_abserr.npy", err.astype(np.float32))
            fig[fr] = {"psnr": psnr(img, gt), "ssim": float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))}
    res = {
        "iterations_done": it + (0 if stopped else 1), "stopped_by_temperature_guard": stopped,
        "train_wallclock_s": round(train_s, 1), "paused_s": round(guard.paused, 1),
        "max_gpu_temp_c": guard.max_gpu, "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "max_cpu_temp_c": guard.max_cpu, "final_gaussians": int(len(params["means"])),
        "test": {"n": len(ps), "psnr_mean": float(np.mean(ps)), "psnr_sd": float(np.std(ps)), "ssim_mean": float(np.mean(ss_)),
                 "ssim_sd": float(np.std(ss_)), "per_view": [{"frame": int(fr), "psnr": p, "ssim": s} for fr, p, s in zip(D["test_frames"], ps, ss_)]},
        "figure_train_views": fig, "resolution": [W, H], "sh_degree": a.sh,
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    with (out / "temperatures.csv").open("w") as fh:
        fh.write("iter,t_s,gpu_c,cpu_c\n" + "\n".join(f"{i},{t - t_start:.1f},{g},{c}" for i, t, g, c in guard.samples) + "\n")
    torch.save({k: v.detach().cpu() for k, v in params.items()}, Path(a.data) / "rgb_block3_2dgs.pt")
    log(json.dumps({k: v for k, v in res.items() if k != "test"} | {"test_psnr": res["test"]["psnr_mean"], "test_ssim": res["test"]["ssim_mean"]}))


if __name__ == "__main__":
    main()
