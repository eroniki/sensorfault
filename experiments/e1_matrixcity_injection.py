"""Experiment E1: controlled depth-fault injection on one MatrixCity aerial block (exact geometry).

What this run tests: the statistical gate (information-form fusion on rendering, leave-view-out and
leave-block-out, whitening, scene temperature, region scan, Holm outer loop) and the cheap aligned-RMS
baseline, with exact truth. What it does not test: gsplat training, opacity, densification, pose
refinement or a real depth model. Surfels are placed on the exact MatrixCity geometry (voxel-merged
back-projections of the rendered depth), and the "depth model" output is the surfel model's rendered
depth plus synthetic noise sigma_u = sigma_rel * depth, plus injected regional faults. The fusion is the
linearised Gaussian model at that surface (exact for this synthetic model).

Run (CPU, single process; set OMP_NUM_THREADS<=8):
  PYTHONPATH=src:experiments python experiments/e1_matrixcity_injection.py --out results/e1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from matrixcity_io import load_block, pixel_rays_world
from scipy.stats import chi2, norm

from sensorfault.eval import gate_loop as gl

CFG = {
    "data_root": "/home/eiki/data/matrixcity/block_3_depth",
    "transforms": "/home/eiki/data/matrixcity/transforms_block3.json",
    "block": "MatrixCity small_city aerial train block_3 (float32 depth)",
    "step": 20,  # 1920x1080 -> 96x54, point-sampled
    "voxel_m": 3.0,
    "cos_min": 0.2,
    "plane_tol_m": 2.0,
    "sigma_rel": 0.01,  # depth noise sd = 1% of depth
    "prior_sd": {"nu_m": 30.0, "tilt_rad": 1.0},
    "lam_min_obs": 2.0,  # support: >= 2 median-pixel observations from outside the block
    "min_testable_frac": 0.3,
    "tiles": [4, 6],  # region-scan candidate regions (image tiles; no appearance segments here)
    "alpha": 0.05,
    "round_cap": 10,
    "block_sizes": [1, 3],
    "seeds_clean": 20,
    "seeds_cell": 10,
    "deltas_sigma": [2, 4, 8, 16],
    "areas": [0.05, 0.1, 0.2, 0.4],
    "repeat": {"n_views": 3, "delta_sigma": 8, "radius_m": 25.0, "seeds": 10, "noise_scale": 1.3},
}


# ------------------------------------------------------------------------------------------------
# geometry: surfels on the exact surface, per-pixel assignment, rendered depth and Jacobians
# ------------------------------------------------------------------------------------------------


def build(K, views, cfg):
    h, w = views[0]["depth"].shape
    pts, nrm, vid, pix = [], [], [], []
    for i, v in enumerate(views):
        ray = pixel_rays_world(K, v["R"], h, w)
        X = v["c"] + v["depth"][..., None] * ray
        du, dv = np.gradient(X, axis=1), np.gradient(X, axis=0)
        n = np.cross(du, dv)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        flip = np.sum(n * (v["c"] - X), -1) < 0
        n[flip] *= -1
        v["ray"] = ray
        pts.append(X.reshape(-1, 3))
        nrm.append(n.reshape(-1, 3))
        vid.append(np.full(h * w, i))
        pix.append(np.arange(h * w))
    P, N, VID, PIX = map(np.concatenate, (pts, nrm, vid, pix))
    key = np.floor(P / cfg["voxel_m"]).astype(np.int64)
    _, sid, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    sid = sid.ravel()
    ns = sid.max() + 1
    mu = np.zeros((ns, 3))
    np.add.at(mu, sid, P)
    mu /= counts[:, None]
    nn = np.zeros((ns, 3))
    np.add.at(nn, sid, N)
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
    ref = np.where(np.abs(nn[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    t1 = np.cross(nn, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(nn, t1)
    # per view: surfel id per pixel, model depth D_bar, Jacobian J (3) wrt (nu, tau1, tau2)
    for i, v in enumerate(views):
        m = i == VID
        s = sid[m]
        ray = v["ray"].reshape(-1, 3)
        den = np.sum(nn[s] * ray, 1)
        dbar = np.sum(nn[s] * (mu[s] - v["c"]), 1) / den
        x = v["c"] + dbar[:, None] * ray
        dx = x - mu[s]
        J = (
            np.stack([np.ones_like(den), -np.sum(t1[s] * dx, 1), -np.sum(t2[s] * dx, 1)], 1)
            / den[:, None]
        )
        cosang = np.abs(den) / np.linalg.norm(ray, axis=1)
        ok = (cosang > cfg["cos_min"]) & (np.abs(dbar - v["depth"].ravel()) < cfg["plane_tol_m"])
        v.update(sid=s[ok], pix=PIX[m][ok], dbar=dbar[ok], J=J[ok])
        v["sigma"] = cfg["sigma_rel"] * v["dbar"]
        Jw = v["J"] / v["sigma"][:, None]
        us, inv = np.unique(v["sid"], return_inverse=True)
        L = np.zeros((len(us), 3, 3))
        np.add.at(L, inv, Jw[:, :, None] * Jw[:, None, :])
        v.update(us=us, inv=inv, L=L)
    return {"n_surfels": int(ns), "mu": mu, "h": h, "w": w}


# ------------------------------------------------------------------------------------------------
# measurements, fusion, cavity, test
# ------------------------------------------------------------------------------------------------


def measure(views, rng, noise_scale=1.0):
    for v in views:
        v["y"] = v["dbar"] + noise_scale * v["sigma"] * rng.standard_normal(len(v["dbar"]))
        v["fault"] = np.zeros(len(v["dbar"]), bool)


def eta_of(v):
    e = (v["y"] - v["dbar"]) / v["sigma"] ** 2
    eta = np.zeros((len(v["us"]), 3))
    np.add.at(eta, v["inv"], v["J"] * e[:, None])
    return eta


class Gate:
    def __init__(self, views, geo, cfg, B=1, whiten=True, temperature=True, prior=None):
        self.v, self.cfg, self.B, self.whiten, self.temperature = views, cfg, B, whiten, temperature
        self.ns = geo["n_surfels"]
        sd = cfg["prior_sd"]
        self.L0 = np.diag([1 / sd["nu_m"] ** 2, 1 / sd["tilt_rad"] ** 2, 1 / sd["tilt_rad"] ** 2])
        med_prec = np.median(np.concatenate([1 / v["sigma"] ** 2 for v in views]))
        self.lam_min = cfg["lam_min_obs"] * med_prec
        self.eta = [eta_of(v) for v in views]
        th, tw = cfg["tiles"]
        r, c = np.divmod(np.arange(geo["h"] * geo["w"]), geo["w"])
        self.tile = (r * th // geo["h"]) * tw + (c * tw // geo["w"])
        self.n_tiles = th * tw
        # strips: views with the same (yaw, track) ordered along the track
        self.order = sorted(range(len(views)), key=lambda i: (views[i]["strip"], views[i]["along"]))

    def block_of(self, j):
        same = [i for i in self.order if self.v[i]["strip"] == self.v[j]["strip"]]
        if len(same) <= self.B:
            return same
        k = same.index(j)
        lo = min(max(k - (self.B - 1) // 2, 0), len(same) - self.B)
        return same[lo : lo + self.B]

    def fuse(self, active):
        L = np.zeros((self.ns, 3, 3))
        E = np.zeros((self.ns, 3))
        for i in active:
            v = self.v[i]
            np.add.at(L, v["us"], v["L"])
            np.add.at(E, v["us"], self.eta[i])
        self.Lsum, self.Esum = L, E

    def stats(self, j, active):
        """Energy T, dof m, max |z| over tiles, aligned-RMS score, testable fraction for view j."""
        v = self.v[j]
        blk = [i for i in self.block_of(j) if i in active or i == j]
        us = v["us"]
        Lc = self.Lsum[us].copy()
        Ec = self.Esum[us].copy()
        for i in blk:
            if i not in active:
                continue
            u = self.v[i]
            pos = np.searchsorted(us, u["us"])
            pos = np.clip(pos, 0, len(us) - 1)
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
        V = np.einsum("ni,nij,nj->n", J, P[inv], J)
        s2 = v["sigma"][ok] ** 2
        y = v["y"][ok]
        A = np.stack([D, np.ones_like(D)], 1)
        tiles = self.tile[v["pix"][ok]]
        G = np.zeros((len(y), self.n_tiles))
        G[np.arange(len(y)), tiles] = 1.0
        if self.whiten:
            # Sigma_r = W^-1 + U P U^T with U the block-sparse pixel-to-surfel Jacobian: pixels on the
            # same surfel share its estimation error. Woodbury with one 3x3 block per surfel:
            # Sigma_r^-1 x = W x - W U (Lambda_c + U^T W U)^-1 U^T W x
            wn = 1.0 / s2
            Sk = Lc.copy()
            np.add.at(Sk, inv, (wn[:, None, None] * J[:, :, None] * J[:, None, :]))
            Sinv = np.linalg.inv(Sk)

            def prec(X):
                X = X if X.ndim == 2 else X[:, None]
                WX = wn[:, None] * X
                UtWX = np.zeros((len(Sk), X.shape[1], 3))
                np.add.at(UtWX, inv, WX[:, :, None] * J[:, None, :])
                corr = np.einsum("kij,kcj->kci", Sinv, UtWX)
                return WX - wn[:, None] * np.einsum("nci,ni->nc", corr[inv], J)
        else:
            wconst = 1.0 / np.median(s2 + V)

            def prec(X):
                X = X if X.ndim == 2 else X[:, None]
                return wconst * X

        PA = prec(A)
        AtPA = A.T @ PA
        ab = np.linalg.solve(AtPA, PA.T @ y)
        r = y - A @ ab
        Pr = prec(r)[:, 0]
        T = float(r @ Pr)
        m = len(r) - 2
        PG = prec(G)
        num = G.T @ Pr
        gMg = np.einsum("nr,nr->r", G, PG) - np.einsum(
            "ra,ab,rb->r", G.T @ PA, np.linalg.inv(AtPA), G.T @ PA
        )
        den = np.sqrt(np.maximum(gMg, 1e-12))
        valid = np.bincount(tiles, minlength=self.n_tiles) >= 10
        z = np.where(valid, num / den, 0.0)
        # aligned RMS (baseline): OLS alignment, unweighted, in units of the median sigma
        ab0 = np.linalg.lstsq(A, y, rcond=None)[0]
        rms = float(np.sqrt(np.mean((y - A @ ab0) ** 2)) / np.sqrt(np.median(s2)))
        return {
            "T": T,
            "m": m,
            "zmax": float(np.max(np.abs(z))),
            "nR": int(valid.sum()),
            "rms": rms,
            "testable": float(ok.mean()),
        }

    def pvalues(self, active, views_to_test=None):
        self.fuse(active)
        tested = list(active) if views_to_test is None else views_to_test
        S = {j: self.stats(j, active) for j in tested}
        ratio = {j: s["T"] / chi2.median(s["m"]) for j, s in S.items() if s is not None}
        out, logp = {}, {}
        for j, s in S.items():
            if s is None:
                out[j] = 1.0  # untestable
                logp[j] = 0.0
                continue
            if self.temperature:
                others = [r for i, r in ratio.items() if i != j]
                g2 = float(np.median(others)) if others else 1.0
            else:
                g2 = 1.0
            lpE = float(chi2.logsf(s["T"] / g2, s["m"]))
            lpS = float(np.log(s["nR"] * 2) + norm.logsf(s["zmax"] / np.sqrt(g2)))
            lp = min(0.0, np.log(2.0) + min(lpE, lpS))
            out[j] = float(np.exp(lp))
            logp[j] = lp
        self.last = S
        self.logp = logp
        return out


# ------------------------------------------------------------------------------------------------
# faults
# ------------------------------------------------------------------------------------------------


def inject_region(v, geo, rng, delta, area):
    h, w = geo["h"], geo["w"]
    rh = max(1, int(round(h * np.sqrt(area))))
    rw = max(1, int(round(w * np.sqrt(area))))
    r0 = rng.integers(0, h - rh + 1)
    c0 = rng.integers(0, w - rw + 1)
    r, c = np.divmod(v["pix"], w)
    m = (r >= r0) & (r < r0 + rh) & (c >= c0) & (c < c0 + rw)
    v["y"] = v["y"] + delta * v["sigma"] * m
    v["fault"] = m


def inject_world_patch(views, idx, geo, centre, radius, delta):
    for i in idx:
        v = views[i]
        d = np.linalg.norm(geo["mu"][v["sid"], :2] - centre[:2], axis=1)
        m = d < radius
        v["y"] = v["y"] + delta * v["sigma"] * m
        v["fault"] = m


# ------------------------------------------------------------------------------------------------
# runs
# ------------------------------------------------------------------------------------------------


def load_cached(cfg):
    """Load the block and build the surfels once; cache next to the data (not in the repo)."""
    import pickle

    cache = Path(cfg["data_root"]).parent / f"e1_cache_step{cfg['step']}_vox{cfg['voxel_m']}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    K, views = load_block(cfg["data_root"], cfg["transforms"], step=cfg["step"])
    geo = build(K, views, cfg)
    for v in views:
        v.pop("ray", None)
        v.pop("depth", None)
    cache.write_bytes(pickle.dumps((K, views, geo)))
    return K, views, geo


def run_loop(gate, n_views, round_cap):
    ids = list(range(n_views))

    def train(weights):
        return [i for i, wt in weights.items() if wt > 0]

    def test(state, views):
        p = gate.pvalues(state, views)
        return p, gate.logp

    # loop off (single pass): every Holm-rejected view is excluded at once; loop on: one per round
    policy = "all_rejected" if round_cap == 1 else "one_per_round"
    return gl.run_gate_loop(
        ids, train, test, alpha=CFG["alpha"], round_cap=round_cap, policy=policy
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["clean", "grid", "repeat"], required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(CFG)
    if a.quick:
        cfg.update(seeds_clean=2, seeds_cell=1, deltas_sigma=[8], areas=[0.2])
        cfg["repeat"] = dict(cfg["repeat"], seeds=1)
    t0 = time.time()
    K, views, geo = load_cached(cfg)
    n = len(views)
    info = {
        "n_views": n,
        "n_surfels": geo["n_surfels"],
        "pixels_per_view_valid_median": float(np.median([len(v["dbar"]) for v in views])),
        "build_s": round(time.time() - t0, 1),
    }
    print(json.dumps(info), flush=True)
    res = {"config": cfg, "info": info}

    stage = a.stage
    if stage != "clean":
        prev = json.loads((out / "clean.json").read_text())
        res["matched_thresholds"] = prev["matched_thresholds"]
        p_thr = prev["matched_thresholds"]["gate_p"]
        rms_thr = prev["matched_thresholds"]["aligned_rms"]
        cfg["seeds_clean"] = 0
    # --- C0: clean scenes, nominal noise and noise 1.3x the offline level ---------------------------
    clean = {}
    for scale in (1.0, cfg["repeat"]["noise_scale"]):
        rows = []
        for seed in range(cfg["seeds_clean"]):
            rng = np.random.default_rng(1000 + seed)
            measure(views, rng, scale)
            for temp in (True, False):
                g = Gate(views, geo, cfg, B=1, temperature=temp)
                p = g.pvalues(list(range(n)))
                S = [s for s in g.last.values() if s is not None]
                Tm = float(np.mean([s["T"] / s["m"] for s in S]))
                loop = run_loop(Gate(views, geo, cfg, B=1, temperature=temp), n, cfg["round_cap"])
                rows.append(
                    {
                        "seed": seed,
                        "temperature": temp,
                        "per_view_fa": float(np.mean(np.array(list(p.values())) < cfg["alpha"])),
                        "mean_T_over_m": Tm,
                        "untestable": float(np.mean([s is None for s in g.last.values()])),
                        "loop_excluded": len(loop.excluded),
                        "rms": [s["rms"] for s in S],
                        "p": list(p.values()),
                    }
                )
            print(f"clean scale={scale} seed={seed} t={time.time() - t0:.0f}s", flush=True)
        clean[str(scale)] = rows
    if stage == "clean":
        res["clean"] = clean
        nom = [r for r in clean["1.0"] if r["temperature"]]
        p_thr = float(np.quantile(np.concatenate([r["p"] for r in nom]), cfg["alpha"]))
        rms_thr = float(np.quantile(np.concatenate([r["rms"] for r in nom]), 1 - cfg["alpha"]))
        res["matched_thresholds"] = {"gate_p": p_thr, "aligned_rms": rms_thr}
        res["runtime_s"] = round(time.time() - t0, 1)
        (out / "clean.json").write_text(json.dumps(res, indent=1, default=float))
        print("done clean", res["runtime_s"], flush=True)
        return

    # --- C1: one faulty view, region fault grid ---------------------------------------------------
    grid = []
    for delta in cfg["deltas_sigma"] if stage == "grid" else []:
        for area in cfg["areas"]:
            for seed in range(cfg["seeds_cell"]):
                rng = np.random.default_rng(2000 + 97 * seed + int(10 * delta) + int(100 * area))
                measure(views, rng)
                j = int(rng.integers(n))
                inject_region(views[j], geo, rng, delta, area)
                g = Gate(views, geo, cfg, B=1)
                p = g.pvalues(list(range(n)))
                sj = g.last[j]
                loop = run_loop(Gate(views, geo, cfg, B=1), n, cfg["round_cap"])
                grid.append(
                    {
                        "delta": delta,
                        "area": area,
                        "seed": seed,
                        "view": j,
                        "testable": sj is not None,
                        "gate_detect_matched": bool(p[j] <= p_thr),
                        "rms_detect_matched": bool(sj is not None and sj["rms"] > rms_thr),
                        "clean_flag_matched": float(
                            np.mean([p[i] <= p_thr for i in range(n) if i != j])
                        ),
                        "loop_detect": j in loop.excluded,
                        "loop_false": len([i for i in loop.excluded if i != j]),
                        "rounds": loop.rounds,
                    }
                )
            print(f"grid delta={delta} area={area} t={time.time() - t0:.0f}s", flush=True)
    if stage == "grid":
        res["single_view_grid"] = grid
        res["runtime_s"] = round(time.time() - t0, 1)
        (out / "grid.json").write_text(json.dumps(res, indent=1, default=float))
        print("done grid", res["runtime_s"], flush=True)
        return

    # --- C2: same error repeated on neighbouring frames of a strip (masking) -----------------------
    rep = []
    rc = cfg["repeat"]
    g_order = Gate(views, geo, cfg, B=1)
    strips = {}
    for i in g_order.order:
        strips.setdefault(views[i]["strip"], []).append(i)
    long_strips = [s for s in strips.values() if len(s) >= rc["n_views"] + 2]
    for seed in range(rc["seeds"]):
        rng = np.random.default_rng(3000 + seed)
        s = long_strips[rng.integers(len(long_strips))]
        k = int(rng.integers(1, len(s) - rc["n_views"]))
        idx = s[k : k + rc["n_views"]]
        mid = views[idx[len(idx) // 2]]
        centre = geo["mu"][mid["sid"][len(mid["sid"]) // 2]]
        for scale in (1.0, rc["noise_scale"]):
            measure(views, rng, scale)
            inject_world_patch(views, idx, geo, centre, rc["radius_m"], rc["delta_sigma"])
            combos = [
                (B, True, cap, True) for B in cfg["block_sizes"] for cap in (1, cfg["round_cap"])
            ]
            combos += [(3, False, cfg["round_cap"], True), (3, True, cfg["round_cap"], False)]
            for B, temp, cap, whiten in combos:
                loop = run_loop(Gate(views, geo, cfg, B=B, whiten=whiten, temperature=temp), n, cap)
                rep.append(
                    {
                        "seed": seed,
                        "noise_scale": scale,
                        "B": B,
                        "temperature": temp,
                        "round_cap": cap,
                        "whiten": whiten,
                        "faulty": idx,
                        "detected": len([i for i in loop.excluded if i in idx]),
                        "false": len([i for i in loop.excluded if i not in idx]),
                        "rounds": loop.rounds,
                        "alarm": loop.alarm,
                    }
                )
        print(f"repeat seed={seed} t={time.time() - t0:.0f}s", flush=True)
    res["repeated"] = rep
    res["runtime_s"] = round(time.time() - t0, 1)
    (out / "repeat.json").write_text(json.dumps(res, indent=1, default=float))
    print("done repeat", res["runtime_s"], flush=True)


if __name__ == "__main__":
    main()
