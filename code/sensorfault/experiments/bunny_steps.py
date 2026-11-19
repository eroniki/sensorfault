"""Intermediate states of one view-gate pass on the Stanford bunny, for the step-by-step figure.

Faulty case of the golden test (+5 cm on view v3, seed 0; numpy reference backend, stride 2, sigma
0.01), one round. Writes results/bunny/steps.npz and steps.json with:
  1. inputs: back-projected points of every view (world, m) and their view index
  2. shared surface: surfel centres, frames, scales; normal-offset information of the fused fit
  3. leave v3 out: share of each surfel's normal-offset information that came from v3; support mask
  4. cavity: posterior sd of each surfel's normal offset without v3 (m)
  5. render into v3: cavity depth and sd, observed depth, residual after the test's shared distance
     bias and per-view affine nuisance, standardised by sqrt(sigma^2 + var) and the scene temperature
  6. region scan: z per tile of the 4x4 grid (scan scale), p-values of all views
The residual and scan maps re-derive what ``ViewGateTest._stats`` computes, per pixel and per tile;
the script checks that its max |z| over tiles matches the gate's own scan.
Run: python3 experiments/bunny_steps.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sensorfault.eval.gaussian_uncertainty import (
    GEO,
    leave_view_out_newton,
    opacity_supported,
    subblock_map,
    supported,
)
from sensorfault.gs.gate import ViewGateTest
from sensorfault.gs.reference import NumpyReferenceBackend
from sensorfault.sim.bunny import BunnyConfig, make_bunny_views
from sensorfault.sim.scene import backproject

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bunny"
J = "v3"
GRID = 4


def main() -> None:
    bv = make_bunny_views(BunnyConfig(fault_view=3, seed=0))
    frames = bv.frames
    ids = [f.sensor_id for f in frames]
    backend = NumpyReferenceBackend(stride=2)
    weights = dict.fromkeys(ids, 1.0)
    state = backend.fit(frames, weights)
    state.meta["weights"] = dict(weights)
    gate = ViewGateTest(backend, frames, sigma=0.01)
    pvals, logps = gate(state, ids)
    diag = gate.last[J]

    # step 2-4: information and cavity, as in ViewGateTest._pass / _cavity
    infos = gate._informations(state, gate._fused(state), 1.0)
    K = len(state.mu)
    L_data = sum((vi.Lambda for vi in infos.values()), np.zeros((K, 10, 10)))
    L = L_data + gate.prior
    Lj = infos[J].Lambda
    g_cav = sum((vi.eta for v, vi in infos.items() if v != J), np.zeros((K, 10)))
    _, cov = leave_view_out_newton(np.zeros((K, 9)), L, Lj, g_cav)
    keep = supported(L - Lj, state.R, gate.prior, gate.lam_min) & opacity_supported(
        L_data, Lj, gate.rho
    )
    G = np.stack([subblock_map(R)[0] for R in state.R])  # (K, 10) normal-offset row
    info_all = np.einsum("ki,kij,kj->k", G, L_data, G)
    info_j = np.einsum("ki,kij,kj->k", G, Lj, G)
    share_j = np.divide(info_j, info_all, out=np.zeros(K), where=info_all > 0)
    sd_cav = np.sqrt(np.einsum("ki,kij,kj->k", G[:, GEO], cov, G[:, GEO]))
    sd_fused = np.sqrt(
        np.einsum("ki,kij,kj->k", G[:, GEO], np.linalg.inv(L[:, GEO, GEO]), G[:, GEO])
    )

    # step 5-6: the cavity render of v3 and the test's residuals
    cav = gate._cavity(state, J, infos, L, L_data, None)
    D, s2, d, m = gate._pixels(J, cav)
    beta = diag["beta"]
    r = d - (D + beta * D * D)
    w = 1.0 / s2
    A = np.stack([D, np.ones(D.size)], 1)
    AtWA = (A * w[:, None]).T @ A
    Ginv = np.linalg.pinv(AtWA, rcond=1e-10)
    Atwr = A.T @ (w * r)
    coef = Ginv @ Atwr
    r_aff = r - A @ coef
    hw = cav.valid.shape
    z_px = np.full(hw, np.nan)
    z_px[m] = r_aff / np.sqrt(s2 * diag["gamma2"])
    res_px = np.full(hw, np.nan)
    res_px[m] = d - D  # raw residual (m)
    rows, cols = np.nonzero(m)
    rid = (rows * GRID // hw[0]) * GRID + (cols * GRID // hw[1])
    z_tile = np.full((GRID, GRID), np.nan)
    for q in np.unique(rid):
        sel = rid == q
        if sel.sum() < gate.min_region_pixels:
            continue
        sw = float(w[sel].sum())
        num = float((w[sel] * r[sel]).sum())
        b = (A[sel] * w[sel, None]).sum(0)
        num -= float(b @ Ginv @ Atwr)
        sw -= float(b @ Ginv @ b)
        if sw > 1e-9 * float(w[sel].sum()):
            z_tile[q // GRID, q % GRID] = num / math.sqrt(sw) / math.sqrt(diag["gamma2_scan"])
    # check against the gate's own scan (max over the 2x2 and 4x4 grids)
    _, _, scan = gate._stats(J, cav, beta)
    scan_max = float(np.max(np.abs(scan)) / math.sqrt(diag["gamma2_scan"]))

    pts, vid = [], []
    for i, f in enumerate(frames):
        p = backproject(f.depth, f.K, f.T_wc)
        pts.append(p)
        vid.append(np.full(len(p), i))
    depth_obs = np.where(frames[3].depth > 0, frames[3].depth, np.nan)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "steps.npz",
        points=np.concatenate(pts),
        point_view=np.concatenate(vid),
        mu=state.mu,
        R=state.R,
        scale=np.exp(state.log_s[:, :2]),
        info_all=info_all,
        share_j=share_j,
        keep=keep,
        sd_cav=sd_cav,
        sd_fused=sd_fused,
        cav_depth=np.where(cav.valid, cav.depth, np.nan),
        cav_sd=np.where(cav.valid, np.sqrt(cav.var + gate.sigma**2), np.nan),
        depth_obs=depth_obs,
        residual=res_px,
        z=z_px,
        z_tile=z_tile,
        fault_mask=bv.fault_mask[3],
        T_wc=np.stack([f.T_wc for f in frames]),
        K=frames[0].K,
    )
    facts = {
        "view": J,
        "n_surfels": K,
        "n_unsupported_without_v3": int((~keep).sum()),
        "beta": beta,
        "gamma2": diag["gamma2"],
        "gamma2_scan": diag["gamma2_scan"],
        "affine": [float(x) for x in coef],
        "max_abs_z_tile_4x4": float(np.nanmax(np.abs(z_tile))),
        "gate_scan_max_abs_z": scan_max,
        "T": diag["T"],
        "dof": diag["dof"],
        "pvals": pvals,
        "neg_log10p": {v: -lp / math.log(10) for v, lp in logps.items()},
        "n_valid_pixels_v3": int(m.sum()),
    }
    (OUT / "steps.json").write_text(json.dumps(facts, indent=1) + "\n")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
