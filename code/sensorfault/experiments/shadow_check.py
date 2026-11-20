"""Shadow check: are monocular depth errors on flat ground tied to cast shadows, repeated across
frames, and do the pixels (multi-view parallax) pull a shadow-induced height step back?

Diagnostic for the sensorfault letter on MatrixCity block 3. No renderer: Part 2 warps real source
images into the judge view through plane-induced homographies (ray-plane intersection).

Alignment is Stage A's: the label-free pair-graph per-frame scale (``stagea.align.align_block`` on
the step-20 grid, no offsets, no beta, no alternations) followed by the calibration-tile block
datum, applied to the model output on the finer analysis grid. The oracle variant is a per-frame
scale, median(exact / raw) over calibration-tile land pixels.

Part 1 (analysis grid, ``--step``; default 4 = 480 x 270):
  luminance L (linear sRGB luminance, S x S block mean); shadow mask L < thr * median_win(L)
  (median over a win x win grid-pixel window; primary win 31, thr 0.6; alternatives (61, 0.6) and
  (31, 0.5)), cleaned by a 3 x 3 opening; boundary band = within 3 px of a mask edge; generic edge
  strength = |grad log L| (Gaussian sigma 1 before and after the gradient).
  Flat ground: exact normal within 10 deg of vertical, exact height range < 0.5 m in the 7 x 7
  window (no real step within 3 px), -3 m < z < 6 m (Stage A's ground: not water, not roof).
  Height error dh = (aligned - exact z-depth) * r_z, the world-z error of the back-projected point
  (equal to the error along the exact normal within 1.5% for normals within 10 deg).
  (a) step = median dh on the dark side minus on the bright side within 5 px (11 x 11 window),
      at shadow-boundary pixels (sides = mask), non-shadow luminance edges and no-edge ground
      (sides = log L below / above its window median); non-shadow edges are reweighted to the
      shadow edges' contrast histogram.
  (b) per 8 m entity and frame: mean dh in shadow minus mean dh lit, averaged over frames.
  (c) Spearman over entities of error magnitude vs boundary-band share.
  Repetition: 1 m cells, >= 5 frames, rho = pooled pairwise product / pooled mean square (Stage A's
  r), boundary-band cells vs other cells, permutation null (values shuffled across cells within a
  frame and class), tile bootstrap.

Part 2 (full resolution): plane sweep nu in [-6, 6] m (0.1 m) along the exact entity normal,
cost = sum over same-heading pairs of 1 - ZNCC over the entity's judge pixels.

Example:
  python experiments/shadow_check.py --model depth_pro --step 4 --out results/shadow_check/depth_pro
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sensorfault.data.exr_zip import read_exr  # noqa: E402
from sensorfault.eval import calibration as cal  # noqa: E402
from sensorfault.stagea import align as alg  # noqa: E402
from sensorfault.stagea import records as rc  # noqa: E402
from sensorfault.stagea import tiles as tl  # noqa: E402
from sensorfault.stagea.geometry import backproject, normals_from_points, pixel_rays_world, project  # noqa: E402

W_FULL, H_FULL = 1920, 1080
SHADOW_CFGS = [(31, 0.6), (61, 0.6), (31, 0.5)]  # (window in grid px, threshold); first = primary
FLAT_DEG = 10.0
FLAT_RANGE = 0.5  # m, exact height range in 7 x 7
STEP_EDGE = 1.0  # m, exact height range in 3 x 3 that counts as a real step (for the distance)
BAND = 3
RAD = 5
MIN_SIDE = 10
NU = np.round(np.arange(-6.0, 6.0001, 0.1), 2)


# ----------------------------------------------------------------------------------------- inputs
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(y):
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(np.maximum(y, 0), 1 / 2.4) - 0.055)


def load_rgb(images, frame):
    with Image.open(images / f"{frame:04d}.png") as im:
        return np.asarray(im.convert("RGB"))


def block_mean(a, S, h, w):
    return a[: h * S, : w * S].reshape(h, S, w, S, *a.shape[2:]).mean(axis=(1, 3))


def cache_full(a, frame):
    from sensorfault.depthcache.cache import load_depth

    r = load_depth(a.cache_root, "matrixcity", a.scene, a.model, f"{frame:04d}")
    if r.kind == "relative_inverse_depth":
        raise SystemExit("this diagnostic supports metric-depth models only")
    return r.depth


def sub(full, S, h, w):
    return full[S // 2 :: S, S // 2 :: S][:h, :w].astype(float)


def block_datum(views, depths, role_fn):
    num = []
    for v, d in zip(views, depths):
        ok = np.isfinite(d) & np.isfinite(v["depth"]) & (role_fn(v) == tl.CAL)
        num.append(v["depth"][ok] / d[ok])
    r = np.concatenate(num)
    return float(np.median(r)) if r.size else 1.0


# ----------------------------------------------------------------------------------------- helpers
def windows(a, vv, uu, r=RAD, fill=np.nan):
    P = np.pad(a, r, mode="constant", constant_values=fill)
    W = np.lib.stride_tricks.sliding_window_view(P, (2 * r + 1, 2 * r + 1))
    return W[vv, uu]


def side_steps(dh, flat, A, B):
    """dh windows (N, k, k), side masks -> median(A) - median(B) where both sides have MIN_SIDE."""
    a = np.where(A & flat, dh, np.nan)
    b = np.where(B & flat, dh, np.nan)
    na = np.isfinite(a).sum((1, 2))
    nb = np.isfinite(b).sum((1, 2))
    ok = (na >= MIN_SIDE) & (nb >= MIN_SIDE)
    out = np.full(len(dh), np.nan)
    if ok.any():
        out[ok] = np.nanmedian(a[ok], (1, 2)) - np.nanmedian(b[ok], (1, 2))
    return out


def tile_of(xy):
    return tl.tile_key(tl.cell_index(xy, 64.0))


def boot(stat, tiles, n_boot, seed=0):
    return cal.tile_bootstrap(stat, tiles, n_boot=n_boot, seed=seed)


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}g}"


def _ci(c, nd=3):
    return f"{_f(c['estimate'], nd)} [{_f(c['lo'], nd)}, {_f(c['hi'], nd)}]" if c else "n/a"


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------------------- part 1
def shadow_maps(Yg, win, thr):
    g8 = np.clip(np.round(lin_to_srgb(Yg) * 255), 0, 255).astype(np.uint8)
    med = srgb_to_lin(cv2.medianBlur(g8, win).astype(float) / 255.0)
    mask = Yg < thr * med
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    er = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    di = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    inner = mask & ~er
    edge = inner | (di & ~mask)
    dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    return mask, inner, dist


def frame_geometry(K, v, raw, s_lf, s_or):
    d = v["depth"]
    h, w = d.shape
    P = backproject(K, v["R"], v["c"], d)
    N = normals_from_points(P)
    rays = pixel_rays_world(K, v["R"], h, w)
    Z = P[..., 2]
    ok = np.isfinite(P).all(-1) & np.isfinite(N).all(-1) & np.isfinite(raw) & (raw > 0)
    Zf = np.where(np.isfinite(Z), Z, np.nanmedian(Z))
    rng7 = ndimage.maximum_filter(Zf, 7) - ndimage.minimum_filter(Zf, 7)
    rng3 = ndimage.maximum_filter(Zf, 3) - ndimage.minimum_filter(Zf, 3)
    step = rng3 > STEP_EDGE
    dist_step = cv2.distanceTransform((~step).astype(np.uint8), cv2.DIST_L2, 3)
    flat = ok & (N[..., 2] > np.cos(np.deg2rad(FLAT_DEG))) & (rng7 < FLAT_RANGE) & (Z > -3) & (Z < 6)
    dh_lf = (s_lf * raw - d) * rays[..., 2]
    dh_or = (s_or * raw - d) * rays[..., 2]
    dh_lf[~flat] = np.nan
    dh_or[~flat] = np.nan
    xy = P[..., :2]
    tile = np.where(flat, tile_of(np.nan_to_num(xy)), 0)
    # local error: frame's flat-ground median of its 64 m tile removed (takes out the per-frame
    # scale error, keeps structure below the tile scale)
    dh_loc = np.full_like(dh_lf, np.nan)
    for t in np.unique(tile[flat]):
        m = flat & (tile == t)
        if m.sum() >= 50:
            dh_loc[m] = dh_lf[m] - np.median(dh_lf[m])
    return dict(P=P, flat=flat, dh_lf=dh_lf, dh_or=dh_or, dh_loc=dh_loc, tile=tile, dist_step=dist_step)


def part1_frame(geo, Yg, tex, cfg, rng, max_centers=3000):
    flat = geo["flat"]
    mask, inner, dist = shadow_maps(Yg, *cfg)
    logL = np.log(Yg + 1e-3)
    gs = ndimage.gaussian_filter(logL, 1.0)
    gy, gx = np.gradient(gs)
    grad = ndimage.gaussian_filter(np.hypot(gx, gy), 1.0)
    band = dist <= BAND
    out = {"mask": mask, "band": band}
    cents = {}
    shd = inner & flat
    cents["shadow"] = shd
    if shd.sum() >= 20:
        g_thr = np.percentile(grad[shd], 25)
    else:
        g_thr = np.percentile(grad[flat], 90) if flat.any() else np.inf
    far = dist > 8
    cents["edge"] = flat & far & (grad >= g_thr)
    g_lo = np.percentile(grad[flat], 50) if flat.any() else 0
    cents["none"] = flat & far & (grad < g_lo)
    rows = {}
    Wdh = {}
    for name, cm in cents.items():
        vv, uu = np.nonzero(cm)
        if len(vv) > max_centers:
            sel = rng.choice(len(vv), max_centers, replace=False)
            vv, uu = vv[sel], uu[sel]
        if not len(vv):
            continue
        fl = windows(flat.astype(float), vv, uu, fill=0) > 0
        L = windows(logL, vv, uu)
        mk = windows(mask.astype(float), vv, uu, fill=0) > 0
        Lm = np.where(fl, L, np.nan)
        with np.errstate(all="ignore"):
            medL = np.nanmedian(Lm.reshape(len(vv), -1), 1)[:, None, None]
        dark_l, bright_l = (L < medL), (L >= medL)
        r = {"tile": geo["tile"][vv, uu], "dist_step": geo["dist_step"][vv, uu]}
        for key in ("dh_lf", "dh_or", "dh_loc"):
            Wd = windows(geo[key], vv, uu)
            if name == "shadow":
                r[key + "_mask"] = side_steps(Wd, fl, mk, ~mk)
            r[key] = side_steps(Wd, fl, dark_l, bright_l)
        a = np.where(fl & dark_l, L, np.nan)
        b = np.where(fl & bright_l, L, np.nan)
        with np.errstate(all="ignore"):
            r["contrast"] = np.nanmedian(b.reshape(len(vv), -1), 1) - np.nanmedian(
                a.reshape(len(vv), -1), 1
            )
            if name == "shadow":
                am = np.where(fl & mk, L, np.nan)
                bm = np.where(fl & ~mk, L, np.nan)
                r["contrast_mask"] = np.nanmedian(bm.reshape(len(vv), -1), 1) - np.nanmedian(
                    am.reshape(len(vv), -1), 1
                )
        rows[name] = r
    out["steps"] = rows
    out["tex"] = tex
    return out


def ent_ids(P, flat, side=8.0):
    keys = np.zeros(flat.shape, np.int64)
    keys[flat] = tl.tile_key(tl.cell_index(P[..., :2][flat], side))
    return keys


def entity_frame_table(frame, geo, p1, ekeys):
    """Rows per (8 m entity, frame) on flat ground."""
    flat = geo["flat"]
    k = ekeys[flat]
    u, inv = np.unique(k, return_inverse=True)
    n = len(u)
    msk = p1["mask"][flat]
    bnd = p1["band"][flat]
    out = {"key": u, "frame": np.full(n, frame), "n": np.bincount(inv, minlength=n)}
    out["n_sh"] = np.bincount(inv, msk, n)
    out["n_band"] = np.bincount(inv, bnd, n)
    out["tex"] = np.bincount(inv, p1["tex"][flat], n)
    for key in ("dh_lf", "dh_or", "dh_loc"):
        d = geo[key][flat]
        ok = np.isfinite(d)
        d0 = np.where(ok, d, 0)
        out[key + "_cnt"] = np.bincount(inv, ok, n)
        out[key + "_sum"] = np.bincount(inv, d0, n)
        out[key + "_abs"] = np.bincount(inv, np.abs(d0), n)
        out[key + "_sh"] = np.bincount(inv, d0 * (msk & ok), n)
        out[key + "_sh_cnt"] = np.bincount(inv, msk & ok, n)
        out[key + "_lit"] = np.bincount(inv, d0 * (~msk & ok), n)
        out[key + "_lit_cnt"] = np.bincount(inv, ~msk & ok, n)
    xy = geo["P"][..., :2][flat]
    out["x"] = np.bincount(inv, xy[:, 0], n) / out["n"]
    out["y"] = np.bincount(inv, xy[:, 1], n) / out["n"]
    return out


def cell_table(frame, yawc, geo, p1):
    flat = geo["flat"]
    xy = geo["P"][..., :2][flat]
    ij = np.floor(xy).astype(np.int64)
    k = tl.tile_key(ij)
    u, inv = np.unique(k, return_inverse=True)
    n = len(u)
    cnt = np.bincount(inv, minlength=n).astype(float)
    out = {"key": u, "frame": np.full(n, frame), "yawc": np.full(n, yawc)}
    for key in ("dh_lf", "dh_or", "dh_loc"):
        d = geo[key][flat]
        ok = np.isfinite(d)
        c = np.bincount(inv, ok, n)
        out[key] = np.where(c > 0, np.bincount(inv, np.where(ok, d, 0), n) / np.maximum(c, 1), np.nan)
    out["band"] = np.bincount(inv, p1["band"][flat], n) / cnt
    out["shadow"] = np.bincount(inv, p1["mask"][flat], n) / cnt
    out["tile"] = tile_of(np.c_[np.bincount(inv, xy[:, 0], n), np.bincount(inv, xy[:, 1], n)] / cnt[:, None])
    return out


def concat_tables(ts):
    return {k: np.concatenate([t[k] for t in ts]) for k in ts[0]}


# ----------------------------------------------------------------------------------------- stats
def summarize_steps(steps, n_boot):
    """steps: name -> dict of arrays (pooled over frames)."""
    res = {}
    sh = steps.get("shadow")
    if sh is None:
        return res
    for variant in ("dh_lf", "dh_or", "dh_loc"):
        rv = {}
        for name, r in steps.items():
            for split in (["_mask", ""] if name == "shadow" else [""]):
                x = r[variant + split]
                c = r["contrast" + (split if name == "shadow" else "")]
                ok = np.isfinite(x) & np.isfinite(c)
                lab = name + ("_masksides" if split else "")
                if ok.sum() < 30:
                    rv[lab] = {"n": int(ok.sum())}
                    continue
                xx, tt = x[ok], r["tile"][ok]
                rv[lab] = {
                    "n": int(ok.sum()),
                    "mean_abs": boot(lambda i: np.mean(np.abs(xx[i])), tt, n_boot),
                    "mean_signed": boot(lambda i: np.mean(xx[i]), tt, n_boot),
                    "median_contrast": float(np.median(c[ok])),
                }
                if variant == "dh_lf":
                    far = ok & (r["dist_step"] > 10)
                    if far.sum() >= 30:
                        xf, tf = x[far], r["tile"][far]
                        rv[lab]["far_from_steps"] = {
                            "n": int(far.sum()),
                            "mean_abs": boot(lambda i: np.mean(np.abs(xf[i])), tf, n_boot),
                            "mean_signed": boot(lambda i: np.mean(xf[i]), tf, n_boot),
                        }
        # contrast-matched non-shadow edges (weights equalise the shadow contrast histogram)
        e = steps.get("edge")
        if e is not None:
            cs = sh["contrast_mask"][np.isfinite(sh[variant + "_mask"]) & np.isfinite(sh["contrast_mask"])]
            okE = np.isfinite(e[variant]) & np.isfinite(e["contrast"])
            if len(cs) >= 30 and okE.sum() >= 30:
                edges = np.quantile(cs, np.linspace(0, 1, 11))
                edges[0], edges[-1] = -np.inf, np.inf
                bs = np.clip(np.searchsorted(edges, cs, side="right") - 1, 0, 9)
                be = np.clip(np.searchsorted(edges, e["contrast"][okE], side="right") - 1, 0, 9)
                ts = np.bincount(bs, minlength=10) / len(bs)
                te = np.bincount(be, minlength=10) / len(be)
                wgt = np.where(te[be] > 0, ts[be] / np.maximum(te[be], 1e-12), 0)
                xe, tte = e[variant][okE], e["tile"][okE]

                def wm(i, f=np.abs):
                    ww = wgt[i]
                    return np.sum(ww * f(xe[i])) / max(ww.sum(), 1e-12)

                rv["edge_contrast_matched"] = {
                    "n": int(okE.sum()),
                    "effective_n": float(wgt.sum() ** 2 / np.sum(wgt**2)),
                    "mean_abs": boot(wm, tte, n_boot),
                    "mean_signed": boot(lambda i: wm(i, lambda z: z), tte, n_boot),
                    "empty_contrast_bins": int((te == 0).sum()),
                }
        res[variant] = rv
    return res


def summarize_entities(E, n_boot, min_px=5):
    """(b) shadow minus lit per entity, (c) Spearman. E: concatenated entity-frame table."""
    u, inv = np.unique(E["key"], return_inverse=True)
    M = len(u)
    res = {"entities_flat": int(M)}
    tile = np.zeros(M, np.int64)
    xs = np.bincount(inv, E["x"], M) / np.bincount(inv, minlength=M)
    ys = np.bincount(inv, E["y"], M) / np.bincount(inv, minlength=M)
    tile = tile_of(np.c_[xs, ys])
    out_b = {}
    for key in ("dh_lf", "dh_or", "dh_loc"):
        sh_c, li_c = E[key + "_sh_cnt"], E[key + "_lit_cnt"]
        both = (sh_c >= min_px) & (li_c >= min_px)
        diff = np.where(both, E[key + "_sh"] / np.maximum(sh_c, 1) - E[key + "_lit"] / np.maximum(li_c, 1), 0)
        s = np.bincount(inv, diff, M)
        c = np.bincount(inv, both, M)
        ok = c >= 1
        de = s[ok] / c[ok]
        te = tile[ok]
        out_b[key] = {
            "entities": int(ok.sum()),
            "entity_frames": int(both.sum()),
            "mean_shadow_minus_lit_m": boot(lambda i: np.mean(de[i]), te, n_boot),
            "median_shadow_minus_lit_m": float(np.median(de)) if len(de) else float("nan"),
            "share_lowered": float(np.mean(de < 0)) if len(de) else float("nan"),
        }
    res["b"] = out_b
    # (c)
    n = np.bincount(inv, E["n"], M)
    bandshare = np.bincount(inv, E["n_band"], M) / n
    out_c = {}
    for key in ("dh_lf", "dh_or", "dh_loc"):
        cnt = np.bincount(inv, E[key + "_cnt"], M)
        mag = np.bincount(inv, E[key + "_abs"], M) / np.maximum(cnt, 1)
        ok = cnt >= 20
        rr = stats.spearmanr(mag[ok], bandshare[ok]).statistic
        mm, bb, tt = mag[ok], bandshare[ok], tile[ok]
        ci = boot(lambda i: stats.spearmanr(mm[i], bb[i]).statistic, tt, min(n_boot, 500))
        out_c[key] = {"entities": int(ok.sum()), "spearman": float(rr), "ci": ci}
    res["c"] = out_c
    return res


def rho_cells(C, n_boot, seed=0, min_frames=5):
    """Repeated share on 1 m cells; band vs other; permutation null within frame and class."""
    res = {}
    u, inv = np.unique(C["key"], return_inverse=True)
    M = len(u)
    nobs = np.bincount(inv, minlength=M)
    band_c = np.bincount(inv, C["band"] > 0, M) / np.maximum(nobs, 1) >= 0.5
    tile = np.zeros(M, np.int64)
    tile[inv] = C["tile"]
    rng = np.random.default_rng(seed)
    for key in ("dh_lf", "dh_loc", "dh_or"):
        e = C[key]
        ok = np.isfinite(e)

        def sums(vals, okm):
            v = np.where(okm, vals, 0)
            n = np.bincount(inv, okm, M).astype(float)
            s1 = np.bincount(inv, v, M)
            s2 = np.bincount(inv, v * v, M)
            # same-heading pairs: sums per (cell, heading class)
            g = inv * 4 + C["yawc"]
            n_h = np.bincount(g, okm, 4 * M).reshape(M, 4)
            s1_h = np.bincount(g, v, 4 * M).reshape(M, 4)
            s2_h = np.bincount(g, v * v, 4 * M).reshape(M, 4)
            return dict(
                n=n,
                s2=s2,
                pp=(s1 * s1 - s2) / 2,
                np_=n * (n - 1) / 2,
                pp_h=((s1_h**2 - s2_h) / 2).sum(1),
                np_h=(n_h * (n_h - 1) / 2).sum(1),
            )

        def rho(S, idx, same=False):
            pp = S["pp_h" if same else "pp"][idx].sum()
            npp = S["np_h" if same else "np_"][idx].sum()
            ms = S["s2"][idx].sum() / max(S["n"][idx].sum(), 1)
            return (pp / npp) / ms if npp > 0 and ms > 0 else np.nan

        S = sums(e, ok)
        # null: shuffle values across cells within each frame and class
        en = e.copy()
        cls_obs = band_c[inv]
        for f in np.unique(C["frame"]):
            for b in (False, True):
                i = np.flatnonzero((C["frame"] == f) & (cls_obs == b) & ok)
                en[i] = e[rng.permutation(i)]
        Sn = sums(en, ok)
        rk = {}
        keep = S["n"] >= min_frames
        for lab, sel in (("band", keep & band_c), ("other", keep & ~band_c), ("all", keep)):
            idx = np.flatnonzero(sel)
            if len(idx) < 30:
                rk[lab] = {"cells": int(len(idx))}
                continue
            row = {"cells": int(len(idx)), "median_frames": float(np.median(S["n"][idx]))}
            for same in (False, True):
                tag = "same_heading" if same else "all_pairs"
                row[tag] = boot(lambda i, same=same: rho(S, idx[i], same), tile[idx], n_boot)
                row[tag + "_null"] = boot(lambda i, same=same: rho(Sn, idx[i], same), tile[idx], n_boot)
            row["rms_m"] = float(np.sqrt(S["s2"][idx].sum() / S["n"][idx].sum()))
            rk[lab] = row
        res[key] = rk
    return res


# ----------------------------------------------------------------------------------------- part 2
class FullRes:
    def __init__(self, a, images, views):
        self.a, self.images = a, images
        self.v = {v["frame"]: v for v in views}
        self.gray, self.depth = {}, {}
        f = 0.5 * W_FULL / np.tan(0.5 * a.fov)
        self.K = np.array([[f, 0, (W_FULL - 1) / 2], [0, f, (H_FULL - 1) / 2], [0, 0, 1.0]])

    def g(self, fr):
        if fr not in self.gray:
            rgb = load_rgb(self.images, fr).astype(np.float32) / 255
            self.gray[fr] = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
        return self.gray[fr]

    def d(self, fr):
        if fr not in self.depth:
            self.depth[fr] = read_exr(Path(self.a.data_root) / f"{fr:04d}.exr", "R").astype(np.float32) / 100.0
        return self.depth[fr]

    def rays(self, fr, u, v):
        R = self.v[fr]["R"]
        K = self.K
        cam = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -np.ones_like(u)], -1)
        return cam @ R.T


def judge_pixels(FR, fr, pts, cell_ij, side, n, d0, shadow_grid, S, want_shadow):
    """Full-res judge pixels of the entity region: inside the 8 m cell, on the exact plane (0.3 m),
    and (for shadow targets) inside the judge's grid shadow mask."""
    v = FR.v[fr]
    u, vv, z = project(FR.K, v["R"], v["c"], pts)
    ok = np.isfinite(u) & (z > 0)
    if ok.sum() < 5:
        return None
    u0, u1 = int(max(np.floor(u[ok].min()) - 4, 0)), int(min(np.ceil(u[ok].max()) + 4, W_FULL - 1))
    v0, v1 = int(max(np.floor(vv[ok].min()) - 4, 0)), int(min(np.ceil(vv[ok].max()) + 4, H_FULL - 1))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None
    UU, VV = np.meshgrid(np.arange(u0, u1 + 1, dtype=float), np.arange(v0, v1 + 1, dtype=float))
    D = FR.d(fr)[v0 : v1 + 1, u0 : u1 + 1].astype(float)
    r = FR.rays(fr, UU, VV)
    X = v["c"] + D[..., None] * r
    ij = np.floor(X[..., :2] / side).astype(np.int64)
    m = (ij[..., 0] == cell_ij[0]) & (ij[..., 1] == cell_ij[1]) & (np.abs(X @ n - d0) < 0.3)
    gi = np.clip(((VV - S // 2) / S).round().astype(int), 0, shadow_grid.shape[0] - 1)
    gj = np.clip(((UU - S // 2) / S).round().astype(int), 0, shadow_grid.shape[1] - 1)
    sm = shadow_grid[gi, gj]
    m &= sm if want_shadow else ~sm
    if m.sum() < 1:
        return None
    return UU[m], VV[m], X[m], r[m]


def sweep_pair(FR, j, s, jp, n, d0):
    uj, vj, Xe, r = jp
    vs = FR.v[s]
    vjv = FR.v[j]
    # occlusion: the judge's exact point must be the front surface in the source
    us, vss, zs = project(FR.K, vs["R"], vs["c"], Xe)
    inb = (us >= 1) & (us < W_FULL - 2) & (vss >= 1) & (vss < H_FULL - 2) & (zs > 0)
    if inb.sum() < 50:
        return None
    Ds = FR.d(s)
    ui = np.clip(np.round(us).astype(int), 0, W_FULL - 1)
    vi = np.clip(np.round(vss).astype(int), 0, H_FULL - 1)
    vis = inb & (np.abs(Ds[vi, ui] - zs) < 0.01 * zs)
    if vis.sum() < 50:
        return None
    r = r[vis]
    a = FR.g(j)[vj[vis].astype(int), uj[vis].astype(int)].astype(float)
    nc = n @ vjv["c"]
    nr = r @ n
    t = (d0 + NU[:, None] - nc) / nr[None, :]  # (Nnu, N)
    X = vjv["c"] + t[..., None] * r[None]
    u2, v2, z2 = project(FR.K, vs["R"], vs["c"], X)
    okall = np.all((u2 >= 0) & (u2 < W_FULL - 1) & (v2 >= 0) & (v2 < H_FULL - 1) & (z2 > 0), 0)
    if okall.sum() < 50:
        return None
    a = a[okall]
    B = cv2.remap(
        FR.g(s), u2[:, okall].astype(np.float32), v2[:, okall].astype(np.float32), cv2.INTER_LINEAR
    ).astype(float)
    az = (a - a.mean()) / (a.std() + 1e-6)
    Bz = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-6)
    zncc = (Bz * az[None]).mean(1)
    base = np.linalg.norm(vs["c"] - vjv["c"])
    return 1 - zncc, int(okall.sum()), float(a.std()), base


def curvature(cost, i0, half=5):
    lo, hi = max(i0 - half, 0), min(i0 + half + 1, len(NU))
    if hi - lo < 5:
        return np.nan
    c = np.polyfit(NU[lo:hi], cost[lo:hi], 2)
    return 2 * c[0]


def part2(a, FR, views, grids, E, rng, S, max_pairs=8):
    """Plane sweep on faulty shadow entities and textured controls."""
    side = 8.0
    u, inv = np.unique(E["key"], return_inverse=True)
    M = len(u)
    cnt_f = np.bincount(inv, minlength=M)
    n = np.bincount(inv, E["n"], M)
    shfrac = np.bincount(inv, E["n_sh"], M) / n
    bandfrac = np.bincount(inv, E["n_band"], M) / n
    tex = np.bincount(inv, E["tex"], M) / n
    # per entity-frame means of dh in shadow and overall
    sc = E["dh_lf_sh_cnt"]
    lc = E["dh_lf_lit_cnt"]
    ac = E["dh_lf_cnt"]
    m_sh = np.where(sc >= 10, E["dh_lf_sh"] / np.maximum(sc, 1), np.nan)
    m_li = np.where(lc >= 10, E["dh_lf_lit"] / np.maximum(lc, 1), np.nan)
    m_all = np.where(ac >= 10, E["dh_lf_sum"] / np.maximum(ac, 1), np.nan)

    def per_ent(x):
        ok = np.isfinite(x)
        c = np.bincount(inv, ok, M)
        s1 = np.bincount(inv, np.where(ok, x, 0), M)
        s2 = np.bincount(inv, np.where(ok, x * x, 0), M)
        mu = np.where(c > 0, s1 / np.maximum(c, 1), np.nan)
        sd = np.sqrt(np.maximum(s2 / np.maximum(c, 1) - mu**2, 0) * c / np.maximum(c - 1, 1))
        return mu, sd, c

    mu_sh, sd_sh, c_sh = per_ent(m_sh)
    mu_li, _, c_li = per_ent(m_li)
    mu_all, sd_all, c_all = per_ent(m_all)
    scl, acl = E["dh_loc_sh_cnt"], E["dh_loc_cnt"]
    mu_sh_loc = per_ent(np.where(scl >= 10, E["dh_loc_sh"] / np.maximum(scl, 1), np.nan))[0]
    mu_all_loc = per_ent(np.where(acl >= 10, E["dh_loc_sum"] / np.maximum(acl, 1), np.nan))[0]
    step = mu_sh - mu_li
    faulty_pool = (
        (shfrac >= 0.25) & (shfrac <= 0.85) & (c_sh >= 3) & (c_li >= 3)
        & (np.abs(mu_sh) > 2 * sd_sh) & np.isfinite(step)
    )
    order = np.argsort(-np.abs(np.where(faulty_pool, step, 0)))
    faulty = order[: min(a.n_entities, int(faulty_pool.sum()))]
    ctrl_pool = np.flatnonzero((shfrac < 0.02) & (bandfrac < 0.02) & (c_all >= 3))
    # texture-matched controls: nearest log-texture without replacement
    ctrl = []
    avail = set(ctrl_pool.tolist())
    lt = np.log(tex + 1e-9)
    cp = ctrl_pool[np.argsort(lt[ctrl_pool])]
    for e in faulty:
        if not avail:
            break
        cand = cp[np.isin(cp, list(avail))]
        k = cand[np.argmin(np.abs(lt[cand] - lt[e]))]
        ctrl.append(k)
        avail.discard(k)
    info = {
        "faulty_pool": int(faulty_pool.sum()),
        "faulty_selected": len(faulty),
        "control_pool": int(len(ctrl_pool)),
        "control_selected": len(ctrl),
    }
    # region points per entity (exact world points of the grid pixels of the target region)
    want = {int(e): True for e in faulty} | {int(e): False for e in ctrl}
    key_to_m = {int(k): i for i, k in enumerate(u)}
    pts = {e: [] for e in want}
    seen = {e: [] for e in want}
    wkeys = np.array([u[e] for e in want])
    wsh = np.array([want[e] for e in want])
    wlist = list(want)
    for fr, g in grids.items():
        fl = g["flat"]
        k = g["ekeys"][fl]
        P = g["P"][fl]
        sm = g["mask"][fl]
        pos = np.searchsorted(np.sort(wkeys), k)
        srt = np.argsort(wkeys)
        pos = np.clip(pos, 0, len(wkeys) - 1)
        hit = np.sort(wkeys)[pos] == k
        wi = np.full(len(k), -1)
        wi[hit] = srt[pos[hit]]
        ok = wi >= 0
        ok[ok] &= sm[ok] == wsh[wi[ok]]
        for w_ in np.unique(wi[ok]):
            m = ok & (wi == w_)
            if m.sum() >= 3:
                e = wlist[w_]
                pts[e].append(P[m])
                seen[e].append((fr, int(m.sum())))
    del key_to_m
    yaw = {v["frame"]: float(v["yaw"]) % 360 for v in views}
    C = {v["frame"]: v["c"] for v in views}
    rows = []
    t0 = time.time()
    for e, is_f in want.items():
        if not pts[e]:
            continue
        Pp = np.concatenate(pts[e])
        p0 = Pp.mean(0)
        _, _, Vt = np.linalg.svd(Pp - p0, full_matrices=False)
        nrm = Vt[2] * np.sign(Vt[2][2])
        d0 = float(nrm @ p0)
        cell_ij = tl.cell_index(p0[None, :2], side)[0]
        judges = sorted(seen[e], key=lambda t: -t[1])
        pairs = []
        for j, _ in judges:
            for s, _ in judges:
                if s == j or abs(yaw[s] - yaw[j]) > 1:
                    continue
                b = np.linalg.norm(C[s] - C[j])
                if 20 < b < 160:
                    pairs.append((b, j, s))
        # prefer the longest baselines among judges with most pixels, at most max_pairs
        pairs.sort(key=lambda t: -t[0])
        cost = np.zeros(len(NU))
        used = 0
        npx = []
        stds = []
        jcache = {}
        for b, j, s in pairs:
            if used >= max_pairs:
                break
            if j not in jcache:
                jcache[j] = judge_pixels(FR, j, Pp, cell_ij, side, nrm, d0, grids[j]["mask"], S, is_f)
            jp = jcache[j]
            if jp is None or len(jp[0]) < 80:
                continue
            res = sweep_pair(FR, j, s, jp, nrm, d0)
            if res is None:
                continue
            cost += res[0]
            used += 1
            npx.append(res[1])
            stds.append(res[2])
        if used == 0:
            continue
        # Depth Pro plane offset along the exact normal (normal ~ vertical, so dh ~ offset)
        nu_d = float(mu_sh[e] if is_f else mu_all[e])
        i_ph = int(np.argmin(cost))
        nu_p = float(NU[i_ph])
        gcost = np.gradient(cost, NU)
        if abs(nu_d) <= NU[-1]:
            gd = float(np.interp(nu_d, NU, gcost))
        else:
            gd = float(gcost[-1] if nu_d > 0 else gcost[0])
        nu_l = float(mu_sh_loc[e] if is_f else mu_all_loc[e])
        gl = float(np.interp(np.clip(nu_l, NU[0], NU[-1]), NU, gcost))
        rows.append(
            {
                "nu_depth_loc": nu_l,
                "grad_at_nu_depth_loc": gl,
                "toward_truth_loc": bool(gl * nu_l > 0),
                "entity": int(u[e]),
                "faulty": bool(is_f),
                "pairs": used,
                "pixels_median": float(np.median(npx)),
                "judge_std": float(np.median(stds)),
                "nu_depth": nu_d,
                "nu_photo": nu_p,
                "grad_at_nu_depth": gd,
                "toward_truth": bool(gd * nu_d > 0),
                "stiffness": float(curvature(cost / used, i_ph)),
                "cost_min": float(cost[i_ph] / used),
                "cost_at_0": float(cost[np.argmin(np.abs(NU))] / used),
                "shadow_step": float(step[e]) if np.isfinite(step[e]) else None,
                "shadow_frac": float(shfrac[e]),
                "texture": float(tex[e]),
                "sd_frames": float(sd_sh[e] if is_f else sd_all[e]),
                "cost": (cost / used).tolist(),
                "p0": p0.tolist(),
                "judge": judges[0][0],
            }
        )
        if a.max_seconds_part2 and time.time() - t0 > a.max_seconds_part2:
            info["stopped_at_time_budget"] = True
            break
    info["seconds"] = time.time() - t0
    return rows, info


def summarize_part2(rows, n_boot):
    out = {}
    for lab, f in (("faulty", True), ("control", False)):
        R = [r for r in rows if r["faulty"] == f]
        if not R:
            out[lab] = {"entities": 0}
            continue
        nd = np.array([r["nu_depth"] for r in R])
        ndl = np.array([r["nu_depth_loc"] for r in R])
        nph = np.array([r["nu_photo"] for r in R])
        tw = np.array([r["toward_truth"] for r in R])
        twl = np.array([r["toward_truth_loc"] for r in R])
        st = np.array([r["stiffness"] for r in R])
        tx = np.array([r["texture"] for r in R])
        inr = np.abs(nd) <= 6
        inl = np.abs(ndl) <= 6
        big = inl & (np.abs(ndl) >= 0.5)
        o = {
            "entities": len(R),
            "nu_depth_outside_sweep": int((~inr).sum()),
            "median_abs_nu_depth": float(np.median(np.abs(nd))),
            "median_abs_nu_photo": float(np.median(np.abs(nph))),
            "share_photo_closer": float(np.mean(np.abs(nph) < np.abs(nd))),
            "share_gradient_toward_truth": float(np.mean(tw)),
            "inside_sweep": {
                "entities": int(inr.sum()),
                "share_photo_closer": float(np.mean((np.abs(nph) < np.abs(nd))[inr])) if inr.any() else None,
                "share_gradient_toward_truth": float(np.mean(tw[inr])) if inr.any() else None,
            },
            "local": {
                "median_abs_nu_depth_loc": float(np.median(np.abs(ndl))),
                "outside_sweep": int((~inl).sum()),
                "share_photo_closer": float(np.mean(np.abs(nph) < np.abs(ndl))),
                "share_gradient_toward_truth": float(np.mean(twl[inl])) if inl.any() else None,
                "abs_ge_0.5_entities": int(big.sum()),
                "abs_ge_0.5_share_photo_closer": float(np.mean((np.abs(nph) < np.abs(ndl))[big])) if big.any() else None,
                "abs_ge_0.5_share_gradient_toward_truth": float(np.mean(twl[big])) if big.any() else None,
            },
            "share_abs_nu_photo_le_0.3": float(np.mean(np.abs(nph) <= 0.3)),
            "share_abs_nu_photo_le_1": float(np.mean(np.abs(nph) <= 1.0)),
            "median_stiffness": float(np.nanmedian(st)),
            "spearman_stiffness_texture": float(stats.spearmanr(st, tx, nan_policy="omit").statistic)
            if len(R) > 5
            else None,
            "median_pairs": float(np.median([r["pairs"] for r in R])),
        }
        # shares split by stiffness: stiff half vs soft half
        if len(R) >= 20:
            hi = st >= np.nanmedian(st)
            o["share_photo_closer_stiff_half"] = float(np.mean((np.abs(nph) < np.abs(nd))[hi]))
            o["share_photo_closer_soft_half"] = float(np.mean((np.abs(nph) < np.abs(nd))[~hi]))
            o["median_abs_nu_photo_stiff_half"] = float(np.median(np.abs(nph[hi])))
        out[lab] = o
    return out


# ----------------------------------------------------------------------------------------- figures
def figures(out, rows, FR, grids, views, S, dhmaps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = [r for r in rows if r["faulty"]]
    F = sorted(F, key=lambda r: -abs(r["shadow_step"] or 0))
    picks = []
    for r in F:
        if len(picks) >= 3:
            break
        picks.append(r)
    ctrl = [r for r in rows if not r["faulty"]]
    if ctrl:
        picks.append(sorted(ctrl, key=lambda r: -r["stiffness"])[len(ctrl) // 2])
    if not picks:
        return []
    fig, ax = plt.subplots(len(picks), 3, figsize=(10, 2.9 * len(picks)))
    ax = np.atleast_2d(ax)
    for k, r in enumerate(picks):
        j = r["judge"]
        v = FR.v[j]
        p0 = np.array(r["p0"])
        u, vv, _ = project(FR.K, v["R"], v["c"], p0[None])
        cu, cv = int(u[0]), int(vv[0])
        hw = 90
        u0, u1 = max(cu - hw, 0), min(cu + hw, W_FULL)
        v0, v1 = max(cv - hw, 0), min(cv + hw, H_FULL)
        rgb = load_rgb(FR.images, j)[v0:v1, u0:u1]
        ax[k, 0].imshow(rgb)
        g = grids[j]
        gm = g["mask"]
        gu0, gu1, gv0, gv1 = u0 // S, u1 // S, v0 // S, v1 // S
        ax[k, 0].contour(
            np.kron(gm[gv0:gv1, gu0:gu1].astype(float), np.ones((S, S))), [0.5], colors="c", linewidths=0.6
        )
        # entity outline (8 m cell)
        ek = g["ekeys"][gv0:gv1, gu0:gu1] == r["entity"]
        ax[k, 0].contour(np.kron(ek.astype(float), np.ones((S, S))), [0.5], colors="y", linewidths=0.8)
        ax[k, 0].set_title(f"frame {j}: shadow (cyan), entity (yellow)", fontsize=8)
        ax[k, 0].axis("off")
        dm = dhmaps[j][gv0:gv1, gu0:gu1]
        lim = np.nanpercentile(np.abs(dm), 98) if np.isfinite(dm).any() else 1
        im = ax[k, 1].imshow(dm, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[k, 1].contour(gm[gv0:gv1, gu0:gu1].astype(float), [0.5], colors="k", linewidths=0.5)
        ax[k, 1].set_title("height error dh (m), flat ground", fontsize=8)
        ax[k, 1].axis("off")
        plt.colorbar(im, ax=ax[k, 1], fraction=0.046)
        ax[k, 2].plot(NU, r["cost"], "k-", lw=1)
        ax[k, 2].axvline(0, color="g", lw=1, label="truth")
        if abs(r["nu_depth"]) <= 6:
            ax[k, 2].axvline(r["nu_depth"], color="r", lw=1, ls="--", label="depth")
        ax[k, 2].axvline(r["nu_photo"], color="b", lw=1, ls=":", label="photo")
        ax[k, 2].set_xlabel("plane offset nu (m)", fontsize=8)
        ax[k, 2].set_ylabel("1 - ZNCC", fontsize=8)
        tag = "shadow-faulty" if r["faulty"] else "control"
        ax[k, 2].set_title(
            f"{tag}: nu_depth {r['nu_depth']:.2f}, nu_photo {r['nu_photo']:.1f}, {r['pairs']} pairs", fontsize=8
        )
        ax[k, 2].legend(fontsize=7)
        ax[k, 2].tick_params(labelsize=7)
    fig.tight_layout()
    p = out / "panels_part2.png"
    fig.savefig(p, dpi=80)
    plt.close(fig)
    return [p.name]


def fig_part1(out, steps_primary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, col in (("shadow", "C3"), ("edge", "C0"), ("none", "C7")):
        r = steps_primary.get(name)
        if r is None:
            continue
        x = r["dh_lf_mask"] if name == "shadow" else r["dh_lf"]
        x = x[np.isfinite(x)]
        ax[0].hist(np.clip(x, -4, 4), bins=80, histtype="step", density=True, color=col, label=name)
        c = r["contrast_mask"] if name == "shadow" else r["contrast"]
        c = c[np.isfinite(c)]
        ax[1].hist(np.clip(c, 0, 3), bins=60, histtype="step", density=True, color=col, label=name)
    ax[0].set_xlabel("height-error step, dark minus bright side (m)")
    ax[1].set_xlabel("log-luminance contrast across the window")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    p = out / "panels_part1_steps.png"
    fig.savefig(p, dpi=80)
    plt.close(fig)
    return [p.name]


# ----------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/eiki/data/matrixcity/block_3_depth")
    ap.add_argument("--transforms", default="/home/eiki/data/matrixcity/transforms_block3.json")
    ap.add_argument("--images", default="/home/eiki/data/matrixcity/block_3_rgb_train/block_3")
    ap.add_argument("--cache-root", default="/home/eiki/data/depth_cache")
    ap.add_argument("--scene", default="block3")
    ap.add_argument("--model", default="depth_pro")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: first N frames only")
    ap.add_argument("--max-seconds-part2", type=float, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    from matrixcity_io import load_block

    a.fov = json.loads(Path(a.transforms).read_text())["camera_angle_x"]
    sel = None
    if a.max_frames:
        T = json.loads(Path(a.transforms).read_text())
        sel = [f["frame_index"] for f in T["frames"][: a.max_frames]]
    K20, v20 = load_block(a.data_root, a.transforms, step=20, frames=sel)
    KS, vS = load_block(a.data_root, a.transforms, step=a.step, frames=sel)
    v20 = tl.assign_strips(v20)
    vS = tl.assign_strips(vS)
    log = {"args": {k: v for k, v in vars(a).items()}, "frames": len(vS)}
    print(f"[{time.time() - t0:.0f}s] loaded {len(vS)} frames", flush=True)

    # --- Stage A alignment on the step-20 grid
    full = {v["frame"]: cache_full(a, v["frame"]) for v in vS}
    h20, w20 = v20[0]["depth"].shape
    raw20 = [sub(full[v["frame"]], 20, h20, w20) for v in v20]
    al = alg.align_block(K20, v20, raw20, mode="metric", n_alt=0, huber=0.05, fit_beta=False)
    ent20 = rc.build_entities(K20, v20, side=8.0, tile=64.0, group=2)

    def role20(v):
        eid = ent20.pixel_entity[v["frame"]]
        r = np.full(eid.shape, -1)
        r[eid >= 0] = ent20.role[eid[eid >= 0]]
        return r

    datum = block_datum(v20, al.depth, role20)
    s_lf = {v["frame"]: float(al.params[i, 0] * datum) for i, v in enumerate(v20)}
    assert np.allclose(al.params[:, 1], 0) and al.beta == 0
    # oracle: per-frame median exact/raw on calibration-tile land (step-20 grid, Stage A's roles)
    s_or = {}
    for v, x in zip(v20, raw20):
        r = role20(v)
        ok = (r == tl.CAL) & np.isfinite(x) & (x > 0) & np.isfinite(v["depth"])
        s_or[v["frame"]] = float(np.median(v["depth"][ok] / x[ok])) if ok.sum() > 20 else np.nan
    gl = np.nanmedian(list(s_or.values()))
    s_or = {k: (x if np.isfinite(x) else gl) for k, x in s_or.items()}
    rel = [abs(s_lf[f] / s_or[f] - 1) for f in s_lf]
    log["alignment"] = {
        "history_last": al.history[-1],
        "converged_0.1pct": bool(al.history[-1] < 1e-3),
        "datum_factor": datum,
        "median_abs_scale_diff_labelfree_vs_oracle": float(np.median(rel)),
        "p90_abs_scale_diff_labelfree_vs_oracle": float(np.percentile(rel, 90)),
    }
    print(f"[{time.time() - t0:.0f}s] aligned; {log['alignment']}", flush=True)

    # --- Part 1 per frame
    hS, wS = vS[0]["depth"].shape
    S = a.step
    steps = {c: {} for c in SHADOW_CFGS}
    Etab = {c: [] for c in SHADOW_CFGS}
    Ctab = {c: [] for c in SHADOW_CFGS}
    grids, dhmaps = {}, {}
    shadow_share = {c: [] for c in SHADOW_CFGS}
    for i, v in enumerate(vS):
        fr = v["frame"]
        raw = sub(full[fr], S, hS, wS)
        geo = frame_geometry(KS, v, raw, s_lf[fr], s_or[fr])
        rgb = load_rgb(Path(a.images), fr).astype(float) / 255
        lin = srgb_to_lin(rgb)
        Y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        Yg = block_mean(Y, S, hS, wS)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        gy, gx = np.gradient(gray)
        tex = block_mean(gx * gx + gy * gy, S, hS, wS)
        ek = ent_ids(geo["P"], geo["flat"])
        yawc = int(np.round((float(v["yaw"]) % 360) / 90)) % 4
        for c in SHADOW_CFGS:
            p1 = part1_frame(geo, Yg, tex, c, rng)
            for name, r in p1["steps"].items():
                steps[c].setdefault(name, []).append(r)
            Etab[c].append(entity_frame_table(fr, geo, p1, ek))
            Ctab[c].append(cell_table(fr, yawc, geo, p1))
            if geo["flat"].any():
                shadow_share[c].append(float(p1["mask"][geo["flat"]].mean()))
            if c == SHADOW_CFGS[0]:
                grids[fr] = {"mask": p1["mask"], "flat": geo["flat"], "ekeys": ek, "P": geo["P"]}
                dhmaps[fr] = geo["dh_lf"].astype(np.float32)
        if i % 20 == 0:
            print(f"[{time.time() - t0:.0f}s] part 1 frame {i}/{len(vS)}", flush=True)
    del full

    P1 = {}
    for c in SHADOW_CFGS:
        st = {n: concat_tables(rs) for n, rs in steps[c].items()}
        E = concat_tables(Etab[c])
        Cc = concat_tables(Ctab[c])
        tag = f"win{c[0]}_thr{c[1]}"
        nb = a.n_boot if c == SHADOW_CFGS[0] else max(a.n_boot // 4, 200)
        P1[tag] = {
            "flat_ground_shadow_share_median": float(np.median(shadow_share[c])),
            "a_steps": summarize_steps(st, nb),
            "bc_entities": summarize_entities(E, nb),
            "repetition": rho_cells(Cc, nb, seed=a.seed),
            "centers": {n: int(len(r["tile"])) for n, r in st.items()},
        }
        if c == SHADOW_CFGS[0]:
            E_primary = E
            st_primary = st
        print(f"[{time.time() - t0:.0f}s] part 1 stats {tag}", flush=True)
    log["part1"] = P1
    figs = fig_part1(out, st_primary)

    # --- Part 2
    FR = FullRes(a, Path(a.images), vS)
    rows, info = part2(a, FR, vS, grids, E_primary, rng, S)
    log["part2"] = {"info": info, "summary": summarize_part2(rows, a.n_boot)}
    print(f"[{time.time() - t0:.0f}s] part 2 done {info}", flush=True)
    figs += figures(out, rows, FR, grids, vS, S, dhmaps)
    (out / "part2_entities.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "cost"} for r in rows], indent=0, default=_json)
    )
    np.savez_compressed(out / "part2_costs.npz", nu=NU, cost=np.array([r["cost"] for r in rows]),
                        faulty=np.array([r["faulty"] for r in rows]))
    log["figures"] = figs
    log["runtime_s"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(log, indent=1, default=_json))
    (out / "summary.md").write_text(render(log))
    print((out / "summary.md").read_text())


def render(log):
    a = log["args"]
    L = [f"# Shadow check: MatrixCity {a['scene']}, {a['model']}, step {a['step']}", ""]
    al = log["alignment"]
    L += [
        f"{log['frames']} frames. Alignment: Stage A label-free pair graph + calibration datum "
        f"(factor {_f(al['datum_factor'], 5)}; converged {al['converged_0.1pct']}); per-frame scale differs from the "
        f"oracle (median exact/raw on calibration tiles) by median {_f(al['median_abs_scale_diff_labelfree_vs_oracle'])}, "
        f"p90 {_f(al['p90_abs_scale_diff_labelfree_vs_oracle'])}.",
        "",
        "Error variants: `dh_lf` label-free aligned; `dh_or` oracle scale; `dh_loc` label-free minus the frame's flat-ground "
        "median of its 64 m tile (removes the per-frame scale error). Height errors in metres, + = raised. "
        "CIs: 95% tile bootstrap over 64 m tiles.",
        "",
    ]
    for tag, P in log["part1"].items():
        L += [f"## Part 1, shadow mask {tag}", "",
              f"Median shadow share of flat ground per frame {_f(P['flat_ground_shadow_share_median'])}; centers {P['centers']}.", ""]
        L += ["### (a) Height-error step across edges (dark minus bright side, 5 px)", "",
              "| variant | group | n | mean abs step | mean signed step | median contrast | mean abs, >10 px from real steps |",
              "|---|---|---|---|---|---|---|"]
        for var, rv in P["a_steps"].items():
            for g, r in rv.items():
                far = r.get("far_from_steps", {})
                L.append(f"| {var} | {g} | {r['n']} | {_ci(r.get('mean_abs'))} | {_ci(r.get('mean_signed'))} | "
                         f"{_f(r.get('median_contrast'))} | {_ci(far.get('mean_abs')) if far else ''} |")
        L += ["", "### (b) Shadow minus lit on the same 8 m entity", "",
              "| variant | entities | entity-frames | mean (m) | median (m) | share lowered |", "|---|---|---|---|---|---|"]
        for var, r in P["bc_entities"]["b"].items():
            L.append(f"| {var} | {r['entities']} | {r['entity_frames']} | {_ci(r['mean_shadow_minus_lit_m'])} | "
                     f"{_f(r['median_shadow_minus_lit_m'])} | {_f(r['share_lowered'])} |")
        L += ["", "### (c) Spearman: entity mean |dh| vs boundary-band share", "",
              "| variant | entities | rho_s [CI] |", "|---|---|---|"]
        for var, r in P["bc_entities"]["c"].items():
            L.append(f"| {var} | {r['entities']} | {_ci(r['ci'])} |")
        L += ["", "### Repetition on 1 m cells (>= 5 frames)", "",
              "| variant | cells | n cells | median frames | rms (m) | rho all pairs | null | rho same heading | null |",
              "|---|---|---|---|---|---|---|---|---|"]
        for var, rk in P["repetition"].items():
            for lab, r in rk.items():
                if "all_pairs" not in r:
                    L.append(f"| {var} | {lab} | {r['cells']} | | | | | | |")
                    continue
                L.append(f"| {var} | {lab} | {r['cells']} | {_f(r['median_frames'])} | {_f(r['rms_m'])} | "
                         f"{_ci(r['all_pairs'])} | {_ci(r['all_pairs_null'])} | {_ci(r['same_heading'])} | {_ci(r['same_heading_null'])} |")
        L.append("")
    p2 = log["part2"]
    L += ["## Part 2: plane sweep (full resolution, same-heading pairs)", "", f"Selection: {p2['info']}", "",
          "| group | entities | median abs nu_depth | median abs nu_photo | share photo closer | share gradient toward truth | "
          "share abs nu_photo <= 0.3 m | median stiffness | Spearman(stiffness, texture) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g, o in p2["summary"].items():
        if not o.get("entities"):
            L.append(f"| {g} | 0 | | | | | | | |")
            continue
        L.append(f"| {g} | {o['entities']} | {_f(o['median_abs_nu_depth'])} | {_f(o['median_abs_nu_photo'])} | "
                 f"{_f(o['share_photo_closer'])} | {_f(o['share_gradient_toward_truth'])} | {_f(o['share_abs_nu_photo_le_0.3'])} | "
                 f"{_f(o['median_stiffness'])} | {_f(o['spearman_stiffness_texture'])} |")
    L += ["", "nu_depth outside the +-6 m sweep and the local variant (nu_depth_loc = shadow-region mean of dh_loc, i.e. "
          "the frame's 64 m tile median removed; the gradient is taken at the clipped value):", ""]
    for g, o in p2["summary"].items():
        if o.get("entities"):
            L.append(f"- {g}: outside sweep {o['nu_depth_outside_sweep']}/{o['entities']}; inside sweep {o['inside_sweep']}; "
                     f"local {o['local']}; stiff/soft halves: closer {_f(o.get('share_photo_closer_stiff_half'))} / "
                     f"{_f(o.get('share_photo_closer_soft_half'))}")
    L += ["", f"Figures: {', '.join(log['figures'])}. Runtime {log['runtime_s'] / 60:.1f} min.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
