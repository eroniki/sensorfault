"""Check, on real BlendedMVS data, whether the rendered depth maps are z-depth or ray distance (and sanity-check the camera files).

BlendedMVS follows MVSNet's file format: cams/NNNNNNNN_cam.txt = "extrinsic" (4x4 world->camera), "intrinsic" (3x3), then depth range.
For pairs of views, depth of view A is unprojected under each hypothesis (z-depth or range), projected into view B and compared with
B's own depth map; the right hypothesis gives a small median relative error.

  python scripts/check_blendedmvs_depth.py /path/to/BlendedMVS/<scene_id> [--pairs 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dcal.train.data import read_pfm  # noqa: E402


def read_cam(path):
    t = Path(path).read_text().split()
    i = t.index("extrinsic")
    E = np.array(t[i + 1:i + 17], float).reshape(4, 4)
    j = t.index("intrinsic")
    K = np.array(t[j + 1:j + 10], float).reshape(3, 3)
    return E, K


def hypothesis_errors(dA, EA, KA, dB, EB, KB, step=4):
    """Median relative error of B's depth against A's unprojected depth, under z-depth and under range hypotheses."""
    h, w = dA.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    d = dA[::step, ::step]
    ok = np.isfinite(d) & (d > 0)
    uu, vv, d = u[ok].astype(float), v[ok].astype(float), d[ok]
    ray = np.linalg.inv(KA) @ np.stack([uu, vv, np.ones_like(uu)])
    out = {}
    for hyp in ("z", "range"):
        pc = ray * (d if hyp == "z" else d / np.linalg.norm(ray, axis=0))
        pw = np.linalg.inv(EA) @ np.vstack([pc, np.ones(pc.shape[1])])
        pb = (EB @ pw)[:3]
        z = pb[2]
        uv = KB @ pb
        xb, yb = np.round(uv[0] / z).astype(int), np.round(uv[1] / z).astype(int)
        ins = (z > 0) & (xb >= 0) & (xb < dB.shape[1]) & (yb >= 0) & (yb < dB.shape[0])
        obs = dB[yb[ins], xb[ins]]
        good = np.isfinite(obs) & (obs > 0)
        pred = z[ins] if hyp == "z" else np.linalg.norm(pb[:, ins], axis=0)
        rel = np.abs(pred[good] - obs[good]) / obs[good]
        out[hyp] = (float(np.median(rel)) if rel.size else float("nan"), int(rel.size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--pairs", type=int, default=12)
    a = ap.parse_args()
    s = Path(a.scene_dir)
    ids = sorted(p.stem for p in (s / "rendered_depth_maps").glob("*.pfm"))
    rng = np.random.RandomState(0)
    res = {"z": [], "range": []}
    for _ in range(a.pairs):
        i, j = rng.choice(len(ids), 2, replace=False)
        dA, dB = read_pfm(s / "rendered_depth_maps" / f"{ids[i]}.pfm"), read_pfm(s / "rendered_depth_maps" / f"{ids[j]}.pfm")
        (EA, KA), (EB, KB) = read_cam(s / "cams" / f"{ids[i]}_cam.txt"), read_cam(s / "cams" / f"{ids[j]}_cam.txt")
        if dA.shape != dB.shape:
            continue
        e = hypothesis_errors(dA, EA, KA, dB, EB, KB)
        for k in res:
            if np.isfinite(e[k][0]) and e[k][1] > 200:
                res[k].append(e[k][0])
    for k, v in res.items():
        print(f"{k:>5}: median over {len(v)} pairs of the per-pair median relative error = {np.median(v) if v else float('nan'):.5f}")
    if res["z"] and res["range"]:
        print("verdict:", "z-depth" if np.median(res["z"]) < np.median(res["range"]) else "ray distance (range)")


if __name__ == "__main__":
    main()
