"""Test helper: laser-scan depth in an undistorted view (used only by the real-data ETH3D test)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eth3d_check_scene import read_mlp_matrix, read_ply_xyz  # noqa: E402


def project_scan_depth(base, view):
    pts = read_ply_xyz(Path(base) / "dslr_scan_eval" / "scan1.ply")
    M = read_mlp_matrix(Path(base) / "dslr_scan_eval" / "scan_alignment.mlp")
    pg = pts @ M[:3, :3].T + M[:3, 3]
    pc = pg @ view.T_cw[:3, :3].T + view.T_cw[:3, 3]
    h, w = view.gt_depth.shape
    ok = pc[:, 2] > 1e-6
    pc = pc[ok]
    x = np.floor(pc[:, 0] / pc[:, 2] * view.K[0, 0] + view.K[0, 2]).astype(int)
    y = np.floor(pc[:, 1] / pc[:, 2] * view.K[1, 1] + view.K[1, 2]).astype(int)
    ins = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    pix, z = y[ins] * w + x[ins], pc[ins, 2]
    order = np.lexsort((z, pix))
    pix, z = pix[order], z[order]
    first = np.r_[True, pix[1:] != pix[:-1]]
    out = np.full(h * w, np.nan)
    out[pix[first]] = z[first]
    return out.reshape(h, w)
