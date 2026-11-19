"""Render median depth of the trained RGB 2DGS model (rgb_block3_2dgs.pt) into all 185 training views
at 480x270 (z-depth, metres) for surface fusion; one depth-derived normal check view is fused later.
Run (gsplat venv): /home/eiki/.venvs/sensorfault-gs/bin/python experiments/surface_render_rgb.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization_2dgs

torch.set_num_threads(4)
DATA = Path("/home/eiki/data/matrixcity")
W, H = 480, 270
D = np.load(DATA / "rgb_block3_half.npz")
P = {k: v.cuda() for k, v in torch.load(DATA / "rgb_block3_2dgs.pt", weights_only=True).items()}
fov = float(D["fov_x"])
f = 0.5 * W / np.tan(0.5 * fov)
K = torch.tensor(
    [[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], dtype=torch.float32, device="cuda"
)
flip = np.diag([1.0, -1.0, -1.0, 1.0])
depth_med, depth_exp, alpha = [], [], []
t0 = time.time()
with torch.no_grad():
    for c2w in D["train_c2w"]:
        vm = torch.tensor(np.linalg.inv(c2w @ flip), dtype=torch.float32, device="cuda")
        rc, ra, rn, sn, rd, rmed, _ = rasterization_2dgs(
            means=P["means"],
            quats=P["quats"],
            scales=torch.exp(P["scales"]),
            opacities=torch.sigmoid(P["opacities"]),
            colors=torch.cat([P["sh0"], P["shN"]], 1),
            viewmats=vm[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=3,
            render_mode="RGB+ED",
            near_plane=1.0,
            far_plane=5000.0,
        )
        depth_med.append(rmed[0, ..., 0].cpu().numpy().astype(np.float32))
        depth_exp.append(rc[0, ..., 3].cpu().numpy().astype(np.float32))
        alpha.append(ra[0, ..., 0].cpu().numpy().astype(np.float16))
np.savez_compressed(
    DATA / "rgb_block3_2dgs_depth480.npz",
    depth_median=np.stack(depth_med),
    depth_expected=np.stack(depth_exp),
    alpha=np.stack(alpha),
    frames=D["train_frames"],
    c2w=D["train_c2w"],
    fov_x=fov,
    width=W,
    height=H,
)
print(f"rendered {len(depth_med)} views in {time.time() - t0:.1f} s")
