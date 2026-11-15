"""H7: pose perturbation check on MatrixCity block 3 (E1 setting: exact geometry, 1% depth noise, 96x54).

After the gate has run on the E1 grid scene of Figure S5 (8 sigma on 40% of view 59; the Holm loop
excluded view 59), the input (GNSS/IMU) poses of chosen views are perturbed by known amounts:
isolated clean views, blocks of three consecutive clean views of a strip (one shared perturbation),
and the flagged view. Sweeps: rotation only, translation only, and both (paired levels), plus a
size-0 control, two seeds each.

(a) Detection before refinement: every view is re-rendered at its (possibly perturbed) pose with one
    surfel renderer (z-buffered surfel discs, ray-plane depth); the map is fused from all active views
    at their poses; each view is tested against its cavity with the pose term J_q Sigma_q J_q^T in
    Sigma_r (Woodbury, rank 6), Sigma_q from the strip's spline with the GNSS/IMU prior and the
    smoothness prior, without the view's own depth. Flagged = rejected by Holm in the round
    (alpha = 0.05, M = N R_max = 1850), ranked by log p.
(b) Refinement: per strip, a cubic B-spline of pose corrections (one control per frame interval) with
    the GNSS/IMU prior at the input poses, a second-difference smoothness prior, and render-and-compare
    depth factors (Gauss-Newton on the 6-dof pose with the per-view scale and offset alignment
    marginalised); 8 iterations, re-rendering the perturbed frames. Error to the exact pose after
    refinement, as an angle and a distance; recovered fraction = 1 - after / before.
Prior settings are fixed before the run (configs/experiments/h7_matrixcity_pose.yaml).
Run: PYTHONPATH=src:experiments python3 experiments/h7_pose_check.py results/h7
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
from scipy.stats import chi2, norm  # noqa: E402

import e1_matrixcity_injection as e1x  # noqa: E402
from matrixcity_io import load_block, pixel_rays_world  # noqa: E402
from sensorfault.eval import gate_loop as gl  # noqa: E402
from sensorfault.eval.gaussian_uncertainty import so3_exp  # noqa: E402
from sensorfault.eval.trajectory_spline import log_so3, pose_error, position_row, second_difference  # noqa: E402

H7 = {
    "rot_levels_deg": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    "trans_levels_gsd": [1, 2, 5, 10, 25, 50],
    "seeds": [0, 1],
    "n_isolated": 5,  # per seed; each group is perturbed on its own
    "n_blocks": 1,
    "block_len": 3,
    "gnss_sd_m": 0.3,
    "gnss_sd_deg": 0.02,
    "smooth_sd_m": 0.3,
    "smooth_sd_deg": 0.02,
    "gn_iters": 8,
    "disc_radius_m": 2.1,  # surfel disc radius: covers 99.8% of the measured pixels at the true poses
    "detect_rule": {"rot_deg": 0.5, "trans_gsd": 10},
    "recover_rule": {"rot_deg": 1.0, "trans_gsd": 25},
    "target": 0.8,
}


# ------------------------------------------------------------------------------------------------ surfels


def surfel_frames(cfg, geo):
    """Surfel normals and tangents, rebuilt exactly as in e1x.build (the cache stores only centres)."""
    K, raw = load_block(cfg["data_root"], cfg["transforms"], step=cfg["step"])
    h, w = raw[0]["depth"].shape
    pts, nrm = [], []
    for v in raw:
        ray = pixel_rays_world(K, v["R"], h, w)
        X = v["c"] + v["depth"][..., None] * ray
        du, dv = np.gradient(X, axis=1), np.gradient(X, axis=0)
        n = np.cross(du, dv)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        n[np.sum(n * (v["c"] - X), -1) < 0] *= -1
        pts.append(X.reshape(-1, 3))
        nrm.append(n.reshape(-1, 3))
    P, N = np.concatenate(pts), np.concatenate(nrm)
    key = np.floor(P / cfg["voxel_m"]).astype(np.int64)
    _, sid, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    sid = sid.ravel()
    mu = np.zeros((sid.max() + 1, 3))
    np.add.at(mu, sid, P)
    mu /= counts[:, None]
    assert np.allclose(mu, geo["mu"]), "surfel order differs from the E1 cache"
    nn = np.zeros_like(mu)
    np.add.at(nn, sid, N)
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
    ref = np.where(np.abs(nn[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    t1 = np.cross(nn, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return nn, t1, np.cross(nn, t1)


class Renderer:
    """Z-buffered surfel discs at 96x54: the nearest surfel plane hit within its disc, per pixel."""

    def __init__(self, K, h, w, mu, nn, t1, t2, cfg):
        self.K, self.h, self.w, self.mu, self.nn, self.t1, self.t2 = K, h, w, mu, nn, t1, t2
        self.r = H7["disc_radius_m"]
        self.cos_min = cfg["cos_min"]

    def rays(self, R):
        return pixel_rays_world(self.K, R, self.h, self.w).reshape(-1, 3)

    def assign(self, R, c, offset=None):
        """Per pixel: surfel id (-1 if none). ``offset``: normal offsets of the fused map (optional)."""
        K, h, w = self.K, self.h, self.w
        Xc = (self.mu - c) @ R
        z = -Xc[:, 2]
        front = z > 1.0
        u = np.where(front, K[0, 0] * Xc[:, 0] / np.where(front, z, 1) + K[0, 2], -99)
        v = np.where(front, -K[1, 1] * Xc[:, 1] / np.where(front, z, 1) + K[1, 2], -99)
        idx = np.nonzero(front & (u > -4) & (u < w + 3) & (v > -4) & (v < h + 3))[0]
        ks = np.clip(np.ceil(self.r * K[0, 0] / z[idx] - 0.5), 0, 3).astype(int)
        ui, vi = np.rint(u[idx]).astype(int), np.rint(v[idx]).astype(int)
        cp, cs = [], []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                m = (np.abs(dx) <= ks) & (np.abs(dy) <= ks)
                pu, pv = ui[m] + dx, vi[m] + dy
                ok = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
                cp.append(pv[ok] * w + pu[ok])
                cs.append(idx[m][ok])
        pix, s = np.concatenate(cp), np.concatenate(cs)
        ray = self.rays(R)[pix]
        n = self.nn[s]
        p = self.mu[s] + (0.0 if offset is None else offset[s, None] * n)
        den = np.sum(n * ray, 1)
        d = np.sum(n * (p - c), 1) / den
        x = c + d[:, None] * ray
        ok = (d > 1.0) & (np.linalg.norm(x - p, axis=1) < self.r) & (np.abs(den) / np.linalg.norm(ray, axis=1) > self.cos_min)
        pix, s, d = pix[ok], s[ok], d[ok]
        o = np.lexsort((d, pix))
        pix, s = pix[o], s[o]
        first = np.r_[True, pix[1:] != pix[:-1]]
        sid = np.full(h * w, -1)
        sid[pix[first]] = s[first]
        return sid

    def depth(self, R, c, pix, s, offset=None):
        ray = self.rays(R)[pix]
        n = self.nn[s]
        p = self.mu[s] + (0.0 if offset is None else offset[s, None] * n)
        return np.sum(n * (p - c), 1) / np.sum(n * ray, 1), ray

    def pose_jacobian(self, R, c, pix, s, offset=None, er=1e-4, et=1e-2):
        """d depth / d (body rotation [rad], centre [m]) with the surfel assignment held fixed."""
        J = np.zeros((len(pix), 6))
        for k in range(3):
            e = np.zeros(3)
            e[k] = er
            J[:, k] = (self.depth(R @ so3_exp(e), c, pix, s, offset)[0] - self.depth(R @ so3_exp(-e), c, pix, s, offset)[0]) / (2 * er)
            e = np.zeros(3)
            e[k] = et
            J[:, 3 + k] = (self.depth(R, c + e, pix, s, offset)[0] - self.depth(R, c - e, pix, s, offset)[0]) / (2 * et)
        return J


# ------------------------------------------------------------------------------------------------ gate


class PoseGate(e1x.Gate):
    """E1 gate with the pose term J_q Sigma_q J_q^T added to Sigma_r (rank-6 Woodbury)."""

    def stats(self, j, active, return_maps=False):
        v = self.v[j]
        blk = [i for i in self.block_of(j) if i in active or i == j]
        us = v["us"]
        Lc, Ec = self.Lsum[us].copy(), self.Esum[us].copy()
        for i in blk:
            if i not in active:
                continue
            u = self.v[i]
            pos = np.clip(np.searchsorted(us, u["us"]), 0, len(us) - 1)
            hit = us[pos] == u["us"]
            np.subtract.at(Lc, pos[hit], u["L"][hit])
            np.subtract.at(Ec, pos[hit], self.eta[i][hit])
        supp = Lc[:, 0, 0] >= self.lam_min
        Lc = Lc + self.L0
        P = np.linalg.inv(Lc)
        th = np.einsum("kij,kj->ki", P, Ec)
        ok = supp[v["inv"]]
        if ok.mean() < self.cfg["min_testable_frac"]:
            return None
        J, inv = v["J"][ok], v["inv"][ok]
        D = v["dbar"][ok] + np.sum(J * th[inv], 1)
        s2 = v["sigma"][ok] ** 2
        y = v["y"][ok]
        A = np.stack([D, np.ones_like(D)], 1)
        tiles = self.tile[v["pix"][ok]]
        G = np.zeros((len(y), self.n_tiles))
        G[np.arange(len(y)), tiles] = 1.0
        wn = 1.0 / s2
        Sk = Lc.copy()
        np.add.at(Sk, inv, (wn[:, None, None] * J[:, :, None] * J[:, None, :]))
        Sinv = np.linalg.inv(Sk)

        def prec0(X):
            X = X if X.ndim == 2 else X[:, None]
            WX = wn[:, None] * X
            UtWX = np.zeros((len(Sk), X.shape[1], 3))
            np.add.at(UtWX, inv, WX[:, :, None] * J[:, None, :])
            corr = np.einsum("kij,kcj->kci", Sinv, UtWX)
            return WX - wn[:, None] * np.einsum("nci,ni->nc", corr[inv], J)

        Jq = v["Jq"][ok]
        PJ = prec0(Jq)
        Sm = np.linalg.inv(v["Sq"]) + Jq.T @ PJ

        def prec(X):
            PX = prec0(X)
            return PX - PJ @ np.linalg.solve(Sm, Jq.T @ PX)

        PA = prec(A)
        AtPA = A.T @ PA
        ab = np.linalg.solve(AtPA, PA.T @ y)
        r = y - A @ ab
        Pr = prec(r)[:, 0]
        T = float(r @ Pr)
        PG = prec(G)
        num = G.T @ Pr
        gMg = np.einsum("nr,nr->r", G, PG) - np.einsum("ra,ab,rb->r", G.T @ PA, np.linalg.inv(AtPA), G.T @ PA)
        valid = np.bincount(tiles, minlength=self.n_tiles) >= 10
        z = np.where(valid, num / np.sqrt(np.maximum(gMg, 1e-12)), 0.0)
        return {"T": T, "m": len(r) - 2, "zmax": float(np.max(np.abs(z))), "nR": int(valid.sum()), "rms": 0.0,
                "testable": float(ok.mean())}


def pvalues_from_stats(S):
    """p-values and log p from per-view statistics, with the leave-view-out scene temperature (as in
    e1x.Gate.pvalues)."""
    ratio = {j: s["T"] / chi2.median(s["m"]) for j, s in S.items() if s is not None}
    keys = list(ratio)
    vals = np.array([ratio[k] for k in keys])
    out, logp = {}, {}
    for j, s in S.items():
        if s is None:
            out[j], logp[j] = 1.0, 0.0
            continue
        others = vals[np.array([k != j for k in keys])]
        g2 = float(np.median(others))
        lpE = e1x.chi2_logsf(s["T"] / g2, s["m"])
        lpS = float(np.log(s["nR"] * 2) + norm.logsf(s["zmax"] / np.sqrt(g2)))
        lp = min(0.0, np.log(2.0) + min(lpE, lpS))
        out[j], logp[j] = float(np.exp(lp)), lp
    return out, logp


# ------------------------------------------------------------------------------------------------ spline


def strip_spline(n):
    """Rows B (n x n_ctrl) of a uniform cubic B-spline, one control per frame interval, and D2."""
    n_ctrl = n + 3
    B = np.stack([position_row(float(t), -1.0, 1.0, n_ctrl) for t in range(n)])
    return B, second_difference(n_ctrl)


def strip_info(B, D2):
    """Information of the control corrections from the GNSS/IMU and smoothness priors, per component."""
    sr, st = np.deg2rad(H7["gnss_sd_deg"]), H7["gnss_sd_m"]
    mr, mt = np.deg2rad(H7["smooth_sd_deg"]), H7["smooth_sd_m"]
    ridge = 1e-4 * np.eye(B.shape[1])  # keeps short strips (fewer frames than active controls) well posed
    Lr = B.T @ B / sr**2 + D2.T @ D2 / mr**2 + ridge / sr**2
    Lt = B.T @ B / st**2 + D2.T @ D2 / mt**2 + ridge / st**2
    return Lr, Lt


def sigma_q(B, D2, i):
    Lr, Lt = strip_info(B, D2)
    vr = float(B[i] @ np.linalg.solve(Lr, B[i]))
    vt = float(B[i] @ np.linalg.solve(Lt, B[i]))
    return np.diag([vr] * 3 + [vt] * 3)


HUBER_K = 3.0  # robust render-and-compare: Huber weights beyond 3 sigma (wrong surfel at edges, faults)


def huber_cost(z):
    a = np.abs(z)
    return float(np.sum(np.where(a <= HUBER_K, 0.5 * z**2, HUBER_K * a - 0.5 * HUBER_K**2)))


def rc_eval(rend, R, c, yimg, offset, sid=None):
    """Aligned, standardised residuals of one view at pose (R, c): (pix, s, D, y, w, ab, z)."""
    if sid is None:
        sid = rend.assign(R, c, offset)
    pix = np.nonzero((sid >= 0) & np.isfinite(yimg))[0]
    s = sid[pix]
    D, _ = rend.depth(R, c, pix, s, offset)
    y = yimg[pix]
    sig = 0.01 * D
    A = np.stack([D, np.ones_like(D)], 1)
    rw = np.ones_like(D)
    for _ in range(3):  # IRLS for the alignment
        w = rw / sig**2
        ab = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
        z = (y - A @ ab) / sig
        rw = np.minimum(1.0, HUBER_K / np.maximum(np.abs(z), 1e-12))
    return pix, s, D, y, rw / sig**2, ab, z


def rc_cost(rend, R, c, yimg, offset, sid=None):
    return huber_cost(rc_eval(rend, R, c, yimg, offset, sid)[-1])


def rc_factor(rend, R, c, yimg, offset, sid=None):
    """Render-and-compare factor of one view at pose (R, c): marginal information H (6x6) and GN step
    on the pose, with the view's scale and offset alignment marginalised; Huber-weighted (IRLS)."""
    pix, s, D, y, w, ab, z = rc_eval(rend, R, c, yimg, offset, sid)
    A = np.stack([D, np.ones_like(D)], 1)
    r = y - A @ ab
    Jq = ab[0] * rend.pose_jacobian(R, c, pix, s, offset)
    Jf = np.concatenate([Jq, A], 1)
    H = Jf.T @ (w[:, None] * Jf)
    g = Jf.T @ (w * r)
    Hm = H[:6, :6] - H[:6, 6:] @ np.linalg.solve(H[6:, 6:], H[6:, :6])
    gm = g[:6] - H[:6, 6:] @ np.linalg.solve(H[6:, 6:], g[6:])
    return Hm, np.linalg.solve(Hm, gm)


def refine_strip(rend, frames, poses_in, yimgs, offset, base_sid, perturbed, drop_prior=False):
    """Spline + pose-graph + render-and-compare refinement of one strip. ``poses_in``: input poses
    (R, c) per frame (perturbed where faulty). Gauss-Newton on the spline's control corrections with
    step halving on the total cost (priors + Huber render-and-compare). ``drop_prior``: exploratory
    variant (not preregistered) without the GNSS/IMU prior of the perturbed frames. Returns refined poses."""
    n = len(frames)
    B, D2 = strip_spline(n)
    nc = B.shape[1]
    sr, st = np.deg2rad(H7["gnss_sd_deg"]), H7["gnss_sd_m"]
    mr, mt = np.deg2rad(H7["smooth_sd_deg"]), H7["smooth_sd_m"]
    Wp = np.diag([1 / sr**2] * 3 + [1 / st**2] * 3)
    Ws = np.diag([1 / mr**2] * 3 + [1 / mt**2] * 3)
    A0 = np.kron(D2.T @ D2, Ws) + np.kron(1e-4 * np.eye(nc), Wp)
    Wpi = [np.zeros((6, 6)) if (drop_prior and perturbed[i]) else Wp for i in range(n)]

    # the spline is the trajectory: a smooth base fitted to the input (GNSS/IMU) poses with the same priors
    # (rotations as small angles about the strip's middle frame), and the corrections live on that base
    Rref = poses_in[n // 2][0]
    om = np.stack([log_so3(Rref.T @ R) for R, _ in poses_in])
    cin = np.stack([c for _, c in poses_in])
    Lr, Lt = strip_info(B, D2)
    om_b = B @ np.linalg.solve(Lr, B.T @ om / sr**2)
    c_b = B @ np.linalg.solve(Lt, B.T @ cin / st**2)
    base = [(Rref @ so3_exp(om_b[i]), c_b[i]) for i in range(n)]
    dprior = np.stack([np.r_[log_so3(base[i][0].T @ poses_in[i][0]), poses_in[i][1] - base[i][1]] for i in range(n)])

    def pose(i, xi):
        R0, c0 = base[i]
        return R0 @ so3_exp(xi[i, :3]), c0 + xi[i, 3:]

    def sid_for(i, xi):
        return None  # re-render every frame: the base trajectory moves all frames slightly

    def total(C):
        xi = B @ C
        prior = 0.5 * sum((xi[i] - dprior[i]) @ Wpi[i] @ (xi[i] - dprior[i]) for i in range(n)) + 0.5 * float(np.sum((D2 @ C) @ Ws * (D2 @ C)))
        return prior + sum(rc_cost(rend, *pose(i, xi), yimgs[frames[i]], offset, sid_for(i, xi)) for i in range(n))

    C = np.zeros((nc, 6))
    cost = total(C)
    for _ in range(H7["gn_iters"]):
        xi = B @ C
        A = A0.copy()
        b = np.zeros(nc * 6)
        for i in range(n):
            Hm, step = rc_factor(rend, *pose(i, xi), yimgs[frames[i]], offset, sid_for(i, xi))
            Bi = np.kron(B[i][None, :], np.eye(6))  # 6 x 6nc
            A += Bi.T @ (Wpi[i] + Hm) @ Bi
            b += Bi.T @ (Wpi[i] @ dprior[i] + Hm @ (xi[i] + step))
        Cn = np.linalg.solve(A, b).reshape(nc, 6)
        for t in (1.0, 0.5, 0.25, 0.125, 0.0625):
            Ct = C + t * (Cn - C)
            ct = total(Ct)
            if ct < cost:
                C, cost = Ct, ct
                break
        else:
            break
    xi = B @ C
    return [pose(i, xi) for i in range(n)]


# ------------------------------------------------------------------------------------------------ run


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "log.txt").open("w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    t0 = time.time()
    cfg = dict(e1x.CFG)
    K, views, geo = e1x.load_cached(cfg)
    n, h, w = len(views), geo["h"], geo["w"]
    nn, t1, t2 = surfel_frames(cfg, geo)
    rend = Renderer(K, h, w, geo["mu"], nn, t1, t2, cfg)
    gsd = float(np.median([np.median(v["dbar"]) for v in views]) / (K[0, 0] * cfg["step"]))
    log(f"surfels {len(nn)}, GSD {gsd:.4f} m, {time.time() - t0:.0f} s")

    # the E1 scene of Figure S5 (same seeding as e1_qualitative.py / surface_dsm.py)
    delta, area, seed = 8, 0.4, 1
    rng = np.random.default_rng(2000 + 97 * seed + int(10 * delta) + int(100 * area))
    e1x.measure(views, rng)
    j_fault = int(rng.integers(n))  # rng now sits where e1x.inject_region drew the Figure S5 fault rectangle
    excluded = [j_fault]  # the Holm loop's exclusion in this scene (results/surface: [59])
    true_pose = [(v["R"].copy(), v["c"].copy()) for v in views]

    strips = {}
    for i in sorted(range(n), key=lambda i: (views[i]["strip"], views[i]["along"])):
        strips.setdefault(views[i]["strip"], []).append(i)
    strip_of = {i: k for k, fr in strips.items() for i in fr}
    spl = {k: strip_spline(len(fr)) for k, fr in strips.items()}
    Sq = {}
    for k, fr in strips.items():
        B, D2 = spl[k]
        for pos, i in enumerate(fr):
            Sq[i] = sigma_q(B, D2, pos)

    base_sid = [rend.assign(R, c) for R, c in true_pose]
    agree = np.mean([np.mean(base_sid[i][v["pix"]] == v["sid"]) for i, v in enumerate(views)])
    log(f"renderer vs E1 assignment at the true poses: {100 * agree:.1f}% of measured pixels on the same surfel")
    # the world for H7 is the surfel-disc model itself: measured depth = render at the true pose + 1% noise,
    # so that the renderer used by the gate and the refinement is exact at the true poses (as E1's model was)
    nrng = np.random.default_rng(4242)
    yimgs = []
    for i, (R, c) in enumerate(true_pose):
        pix = np.nonzero(base_sid[i] >= 0)[0]
        d, _ = rend.depth(R, c, pix, base_sid[i][pix])
        yi = np.full(h * w, np.nan)
        yi[pix] = d + cfg["sigma_rel"] * d * nrng.standard_normal(len(d))
        yimgs.append(yi)

    def set_view(i, R, c, sid=None):
        v = views[i]
        sid = rend.assign(R, c) if sid is None else sid
        pix = np.nonzero((sid >= 0) & np.isfinite(yimgs[i]))[0]
        s = sid[pix]
        dbar, ray = rend.depth(R, c, pix, s)
        x = c + dbar[:, None] * ray
        dx = x - geo["mu"][s]
        den = np.sum(nn[s] * ray, 1)
        J = np.stack([np.ones_like(den), -np.sum(t1[s] * dx, 1), -np.sum(t2[s] * dx, 1)], 1) / den[:, None]
        sigma = cfg["sigma_rel"] * dbar
        Jw = J / sigma[:, None]
        us, inv = np.unique(s, return_inverse=True)
        L = np.zeros((len(us), 3, 3))
        np.add.at(L, inv, Jw[:, :, None] * Jw[:, None, :])
        Jq = rend.pose_jacobian(R, c, pix, s)
        v.update(sid=s, pix=pix, dbar=dbar, J=J, sigma=sigma, y=yimgs[i][pix], us=us, inv=inv, L=L, Jq=Jq, Sq=Sq[i])

    for i, (R, c) in enumerate(true_pose):
        set_view(i, R, c, base_sid[i])
    e1x.inject_region(views[j_fault], geo, rng, delta, area)  # same rectangle as Figure S5, on the re-rendered view
    yimgs[j_fault][views[j_fault]["pix"]] = views[j_fault]["y"]
    log(f"fault re-injected in view {j_fault}: {views[j_fault]['fault'].mean():.2f} of its pixels")
    base_state = {i: {k: views[i][k] for k in ("sid", "pix", "dbar", "J", "sigma", "y", "us", "inv", "L", "Jq", "Sq")} for i in range(n)}
    log(f"baseline views rendered, {time.time() - t0:.0f} s")

    long_strips = [k for k, fr in strips.items() if len(fr) >= 8 and k != views[j_fault]["strip"]]
    sweeps = [("control", [(0.0, 0.0)])]
    sweeps.append(("rotation", [(r, 0.0) for r in H7["rot_levels_deg"]]))
    sweeps.append(("translation", [(0.0, t) for t in H7["trans_levels_gsd"]]))
    sweeps.append(("combined", list(zip(H7["rot_levels_deg"], H7["trans_levels_gsd"]))))
    M = n * cfg["round_cap"]
    active = [i for i in range(n) if i not in excluded]
    seeds = H7["seeds"]
    if os.environ.get("H7_QUICK"):
        seeds = seeds[:1]
        sweeps = [("control", [(0.0, 0.0)]), ("combined", [(1.0, 25)])]

    # baseline statistics of every view (no perturbation); a perturbed group only changes the statistics
    # of views that share surfels with it, so only those are recomputed (exact, not an approximation)
    g0 = PoseGate(views, geo, cfg, B=1)
    g0.fuse(active)
    S0 = {j: g0.stats(j, active) for j in range(n)}
    pv0, lp0 = pvalues_from_stats(S0)
    rej0 = gl.holm_round_rejections(pv0, cfg["alpha"], M, 0, logp=lp0)
    log(f"baseline: Holm round rejects {sorted(rej0)} (view {j_fault} is the faulty one), {time.time() - t0:.0f} s")

    rows, configs = [], []
    for sd in seeds:
        prng = np.random.default_rng(700 + sd)
        ks = [long_strips[q] for q in prng.permutation(len(long_strips))]
        chosen = []
        for k in ks[: H7["n_isolated"]]:
            chosen.append(("isolated", [strips[k][int(prng.integers(1, len(strips[k]) - 1))]]))
        for k in ks[H7["n_isolated"] : H7["n_isolated"] + H7["n_blocks"]]:
            s0 = int(prng.integers(1, len(strips[k]) - H7["block_len"]))
            chosen.append(("block", strips[k][s0 : s0 + H7["block_len"]]))
        chosen.append(("flagged", [j_fault]))
        for sweep, levels in sweeps:
            for rot, tg in levels:
                for typ, grp in chosen:
                    tl = time.time()
                    axis = prng.normal(size=3)
                    axis /= np.linalg.norm(axis)
                    tdir = prng.normal(size=3)
                    tdir /= np.linalg.norm(tdir)
                    inp = {}
                    touched = set()
                    for i in grp:  # one shared perturbation per group (a block moves together)
                        R, c = true_pose[i]
                        touched |= set(views[i]["us"].tolist())
                        inp[i] = (R @ so3_exp(np.deg2rad(rot) * axis), c + tg * gsd * tdir)
                        set_view(i, *inp[i])
                        touched |= set(views[i]["us"].tolist())
                    # (a) gate before refinement
                    g = PoseGate(views, geo, cfg, B=1)
                    g.fuse(active)
                    T = np.fromiter(touched, int)
                    affected = [j for j in range(n) if j in grp or np.isin(views[j]["us"], T).any()]
                    S = dict(S0)
                    for j in affected:
                        S[j] = g.stats(j, active)
                    pv, lp = pvalues_from_stats(S)
                    rej = set(gl.holm_round_rejections(pv, cfg["alpha"], M, 0, logp=lp))
                    # (b) refinement of the group's strip against the fused map
                    Lm = g.Lsum + g.L0[None]
                    offset = np.linalg.solve(Lm, g.Esum[..., None])[:, 0, 0]
                    fr = strips[strip_of[grp[0]]]
                    poses_in = [inp.get(i, true_pose[i]) for i in fr]
                    flags = [i in inp for i in fr]
                    refined = dict(zip(fr, refine_strip(rend, fr, poses_in, yimgs, offset, base_sid, flags)))
                    refined_np = dict(zip(fr, refine_strip(rend, fr, poses_in, yimgs, offset, base_sid, flags, drop_prior=True)))
                    for i in grp:
                        Rt, ct = true_pose[i]
                        eb = pose_error(*inp[i], Rt, ct)
                        ea = pose_error(*refined[i], Rt, ct)
                        en = pose_error(*refined_np[i], Rt, ct)
                        rows.append({
                            "seed": sd, "sweep": sweep, "rot_deg": rot, "trans_gsd": tg, "type": typ, "view": i,
                            "frame": int(views[i]["frame"]), "p": pv[i], "logp": lp[i], "flagged": i in rej,
                            "rot_err_before_deg": eb[0], "trans_err_before_gsd": eb[1] / gsd,
                            "rot_err_after_deg": ea[0], "trans_err_after_gsd": ea[1] / gsd,
                            "trans_along_axis_frac": float(abs(tdir @ (-Rt[:, 2]))),
                            "rot_recovered": 1 - ea[0] / eb[0] if eb[0] > 1e-9 else None,
                            "trans_recovered": 1 - ea[1] / eb[1] if eb[1] > 1e-9 else None,
                            "noprior_rot_err_after_deg": en[0], "noprior_trans_err_after_gsd": en[1] / gsd,
                            "noprior_rot_recovered": 1 - en[0] / eb[0] if eb[0] > 1e-9 else None,
                            "noprior_trans_recovered": 1 - en[1] / eb[1] if eb[1] > 1e-9 else None,
                        })
                    others = [i for i in rej if i not in grp and i != j_fault]
                    configs.append({"seed": sd, "sweep": sweep, "rot_deg": rot, "trans_gsd": tg, "type": typ, "group": grp,
                                    "affected_views": len(affected), "unperturbed_flagged": len(others),
                                    "faulty_view_flagged": j_fault in rej, "seconds": round(time.time() - tl, 1)})
                    for i in grp:
                        views[i].update(base_state[i])
                done = [r for r in rows if r["seed"] == sd and r["sweep"] == sweep and r["rot_deg"] == rot and r["trans_gsd"] == tg]
                cl = [r for r in done if r["type"] != "flagged"]
                cf = [c for c in configs if c["seed"] == sd and c["sweep"] == sweep and c["rot_deg"] == rot and c["trans_gsd"] == tg]
                log(f"seed {sd} {sweep:11s} rot {rot:5.2f} deg trans {tg:4.0f} GSD: flagged {sum(r['flagged'] for r in cl)}/{len(cl)} "
                    f"clean perturbed; median recovered rot {np.median([r['rot_recovered'] for r in done if r['rot_recovered'] is not None] or [np.nan]):.2f} "
                    f"trans {np.median([r['trans_recovered'] for r in done if r['trans_recovered'] is not None] or [np.nan]):.2f}; "
                    f"unperturbed flagged {sum(c['unperturbed_flagged'] for c in cf)}; {sum(c['seconds'] for c in cf):.0f} s")
                (out / "rows.json").write_text(json.dumps(rows, indent=0, default=float))
                (out / "configs.json").write_text(json.dumps(configs, indent=0, default=float))
    meta = {"gsd_m": gsd, "renderer_agreement": float(agree), "M": M, "excluded_before": excluded, "settings": H7,
            "wallclock_s": round(time.time() - t0, 1)}
    (out / "rows.json").write_text(json.dumps(rows, indent=0, default=float))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    log(json.dumps(meta, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/h7")
