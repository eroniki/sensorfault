"""Render saved 2DGS test views (rgb_block3_2dgs.pt) to PNG: python rgb_render_view.py FRAME [FRAME ...]."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gsplat import rasterization_2dgs  # noqa: E402
from rgb_train_2dgs import psnr, ssim, viewmats_from_c2w, write_png  # noqa: E402

DATA, OUT = Path("/home/eiki/data/matrixcity"), Path("results/rgb")
D = np.load(DATA / "rgb_block3_half.npz")
P = {k: v.cuda() for k, v in torch.load(DATA / "rgb_block3_2dgs.pt").items()}
W, H, fov = int(D["width"]), int(D["height"]), float(D["fov_x"])
f = 0.5 * W / np.tan(0.5 * fov)
K = torch.tensor([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
with torch.no_grad():
    for fr in map(int, sys.argv[1:]):
        k = int(np.where(D["test_frames"] == fr)[0][0])
        vm = viewmats_from_c2w(D["test_c2w"][k]).cuda()
        rc, *_ = rasterization_2dgs(
            means=P["means"], quats=P["quats"], scales=torch.exp(P["scales"]), opacities=torch.sigmoid(P["opacities"]),
            colors=torch.cat([P["sh0"], P["shN"]], 1), viewmats=vm[None], Ks=K[None], width=W, height=H, sh_degree=3,
            render_mode="RGB", near_plane=1.0, far_plane=5000.0)
        img = rc[0, ..., :3].clamp(0, 1)
        gt = torch.tensor(D["test_images"][k], device="cuda").float() / 255
        write_png(OUT / f"test_{fr:04d}_render.png", (img.cpu().numpy() * 255).astype(np.uint8))
        write_png(OUT / f"test_{fr:04d}_gt.png", D["test_images"][k])
        print(fr, round(psnr(img, gt), 2), round(float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])), 3))
