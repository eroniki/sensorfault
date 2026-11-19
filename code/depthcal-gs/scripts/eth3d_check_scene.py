"""Check an extracted ETH3D scene against its laser scan: depth-dump convention (z vs range), undistortion, alignment.

Usage: python scripts/eth3d_check_scene.py /path/to/eth3d/pipes/pipes [--images 2]
Reads: dslr_calibration_jpg, ground_truth_depth, dslr_scan_eval. Nothing is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dcal.eth3d import img_from_cam, parse_cameras, parse_images, read_depth  # noqa: E402


def read_ply_xyz(path):
    with open(path, "rb") as f:
        n = None
        props = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element"):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n = int(parts[2])
            elif line.startswith("property") and in_vertex:
                props.append((line.split()[2], "<f4" if line.split()[1] == "float" else "<f8"))
            elif line == "end_header":
                break
        arr = np.fromfile(f, dtype=np.dtype(props), count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)


def read_mlp_matrix(path):
    import re
    t = Path(path).read_text()
    m = re.search(r"<MLMatrix44>(.*?)</MLMatrix44>", t, re.S)
    return np.array(m.group(1).split(), float).reshape(4, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--images", type=int, default=2)
    a = ap.parse_args()
    b = Path(a.scene_dir)
    cams = parse_cameras(b / "dslr_calibration_jpg" / "cameras.txt")
    poses = parse_images(b / "dslr_calibration_jpg" / "images.txt")
    pts = read_ply_xyz(b / "dslr_scan_eval" / "scan1.ply")
    M = read_mlp_matrix(b / "dslr_scan_eval" / "scan_alignment.mlp")
    pg = pts @ M[:3, :3].T + M[:3, 3]
    print(f"scan points {len(pts)}")
    for p in poses[: a.images]:
        cam = cams[p.cam_id]
        depth = read_depth(b / "ground_truth_depth" / "dslr_images" / Path(p.name).name, cam.width, cam.height)
        T = p.T_cw
        pc = pg @ T[:3, :3].T + T[:3, 3]
        xy, ok = img_from_cam(cam, pc)
        xi, yi = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
        ins = ok & (xi >= 0) & (xi < cam.width) & (yi >= 0) & (yi < cam.height)
        pix = yi[ins] * cam.width + xi[ins]
        z = pc[ins, 2]
        rng = np.linalg.norm(pc[ins], axis=1)
        order = np.lexsort((z, pix))
        pix, z, rng = pix[order], z[order], rng[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, z, rng = pix[first], z[first], rng[first]
        d = depth.ravel()[pix]
        v = np.isfinite(d)
        rz, rr = d[v] / z[v], d[v] / rng[v]
        near = (np.abs(rz - 1) < 0.2) | (np.abs(rr - 1) < 0.2)
        print(f"{p.name}: depth valid {np.isfinite(depth).mean():.3f}, scan pixels with depth {v.sum()}; "
              f"median |d/z-1| {np.median(np.abs(rz[near] - 1)):.5f}, median |d/range-1| {np.median(np.abs(rr[near] - 1)):.5f}; "
              f"fraction within 2% of z {np.mean(np.abs(rz - 1) < 0.02):.3f}, of range {np.mean(np.abs(rr - 1) < 0.02):.3f}")


if __name__ == "__main__":
    main()
