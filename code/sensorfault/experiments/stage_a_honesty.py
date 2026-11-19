"""Stage A: is a depth-only information map overconfident, and by the predicted amount? (H1, H2)

Runs tasks A1 and A3 to A15 of docs/stage-a-tasks.md from exact depth, poses and the depth cache,
and writes results/stage_a/<run>/{strips.md, classes.md, records.npz, summary.md, summary.json,
law_figure.json}.

Depth sources (``--source``):
  cache            monocular depth from the depth cache (``--cache-root``, ``--model``)
  exact_independent  exact depth plus independent noise (pipeline control, A13)
  exact_repeated   exact depth plus entity-level noise with a planted repeated share (A13)

Data (``--block``): ``matrixcity`` reads ``--data-root`` (exact depth EXR) and ``--transforms`` as
experiments/matrixcity_io.py; ``synthetic`` uses sensorfault.stagea.synth (no data, for smoke runs).

Example:
  python experiments/stage_a_honesty.py --block matrixcity \
      --data-root /data/matrixcity/block_3_depth --transforms /data/matrixcity/transforms_block3.json \
      --images /data/matrixcity/block_3_rgb --source cache --cache-root /data/depth_cache \
      --model depth_pro --step 20 --out results/stage_a/depth_pro_s20
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sensorfault.eval import calibration as cal  # noqa: E402
from sensorfault.stagea import align as alg  # noqa: E402
from sensorfault.stagea import classes as clm  # noqa: E402
from sensorfault.stagea import consensus as cg  # noqa: E402
from sensorfault.stagea import records as rc  # noqa: E402
from sensorfault.stagea import repetition as rp  # noqa: E402
from sensorfault.stagea import tiles as tl  # noqa: E402
from sensorfault.stagea.geometry import backproject, pixel_rays_world, project  # noqa: E402

B_LIST = (0, 1, 2, 3, 5, 9)
B_PRIMARY = 3
MIN_N_H1 = 5


# ----------------------------------------------------------------------------------------- inputs
def load_inputs(a):
    if a.block == "synthetic":
        from sensorfault.stagea.synth import make_block

        K, views, _ = make_block(seed=a.seed)
        return K, views, None
    from matrixcity_io import load_block

    K, views = load_block(a.data_root, a.transforms, step=a.step)
    return K, views, (Path(a.images) if a.images else None)


def cache_raw(a, views):
    from sensorfault.depthcache.cache import load_depth

    raw, kinds = [], set()
    for v in views:
        r = load_depth(a.cache_root, "matrixcity", a.scene, a.model, f"{v['frame']:04d}")
        h, w = v["depth"].shape
        d = r.depth[a.step // 2 :: a.step, a.step // 2 :: a.step][:h, :w].astype(float)
        raw.append(d)
        kinds.add(r.kind)
    if len(kinds) != 1:
        raise ValueError(f"mixed depth kinds {kinds}")
    mode = "inverse" if kinds.pop() == "relative_inverse_depth" else "metric"
    return raw, mode


def synthetic_raw(a, K, views, ent, rng):
    """A13: exact depth with independent noise, or with a planted repeated share at entity level."""
    raw = []
    rho, tau = a.planted_rho, a.planted_tau
    common = rng.normal(size=(ent.M, 3)) * tau * np.sqrt(rho)
    for v in views:
        d = v["depth"].copy()
        white = 1 + a.white_noise * rng.normal(size=d.shape)
        if a.source == "exact_independent":
            raw.append(d * white)
            continue
        own = rng.normal(size=(ent.M, 3)) * tau * np.sqrt(1 - rho)
        eid = ent.pixel_entity[v["frame"]]
        rays = pixel_rays_world(K, v["R"], *d.shape)
        ok = eid >= 0
        e = eid[ok]
        X = v["c"] + d[ok][:, None] * rays[ok]
        L = 0.5 * ent.side
        u = np.sum((X - ent.p0[e]) * ent.t1[e], 1) / L
        w = np.sum((X - ent.p0[e]) * ent.t2[e], 1) / L
        phi = np.stack([np.ones_like(u), u, w], 1)
        dh = np.sum(phi * (common[e] + own[e]), 1)
        nr = np.sum(ent.n[e] * rays[ok], 1)
        nr = np.where(
            np.abs(nr) < 0.3, np.copysign(0.3, nr), nr
        )  # grazing pixels: cap the depth change
        dd = np.zeros_like(d)
        dd[ok] = dh / nr
        raw.append((d + dd) * white)
    return raw, "metric"


def contrast_maps(a, views, images_dir):
    if images_dir is None:
        return {}
    from PIL import Image

    out = {}
    for v in views:
        p = images_dir / f"{v['frame']:04d}.png"
        if not p.is_file():
            continue
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"), float) / 255.0
        h, w = v["depth"].shape
        off = a.step // 2
        g = g[max(off - a.step // 2, 0) :, :]
        out[v["frame"]] = clm.gradient_energy(g, a.step, h, w)
    return out


# ----------------------------------------------------------------------------------------- helpers
def footprint_along(K, views):
    out = {}
    for v in views:
        P = backproject(K, v["R"], v["c"], v["depth"]).reshape(-1, 3)
        P = P[np.isfinite(P).all(1)]
        ax = 0 if abs(v["yaw"] % 180) == 90 else 1
        coord = P[:, 1 - ax]
        out[v["frame"]] = (
            float(np.percentile(coord, 99) - np.percentile(coord, 1))
            if len(coord)
            else float("nan")
        )
    return out


def block_datum(views, depths, ent):
    """One factor for the whole block: median exact / aligned depth on calibration-tile land pixels.

    Fixes the block datum (common scale) that parallax leaves uncertain; ground truth from
    calibration tiles only, like the single-frame noise scale."""
    num = []
    for v, d in zip(views, depths):
        eid = ent.pixel_entity[v["frame"]]
        ok = (eid >= 0) & np.isfinite(d) & np.isfinite(v["depth"])
        ok[ok] &= ent.role[eid[ok]] == tl.CAL
        num.append(v["depth"][ok] / d[ok])
    r = np.concatenate(num)
    return float(np.median(r)) if r.size else 1.0


def depth_residual(depths, views):
    rel = [np.nanmedian(np.abs(d - v["depth"]) / v["depth"]) for d, v in zip(depths, views)]
    return {"median_rel": float(np.median(rel)), "p90_rel": float(np.percentile(rel, 90))}


def calibrate_sigma(rec, ent, land, min_rec=30):
    """A8: per-frame sigma so that single-frame s_z = 1 on calibration tiles (land)."""
    q = np.einsum("ri,rij,rj->r", rec.eps, rec.Lam0, rec.eps)
    use = (ent.role[rec.entity] == tl.CAL) & land[rec.entity]
    glob = float(np.sqrt(np.median(q[use]) / cal.CHI2_MEDIAN)) if use.any() else 1.0
    sig = {}
    n_own = 0
    for f in np.unique(rec.frame):
        m = use & (rec.frame == f)
        if m.sum() >= min_rec:
            sig[int(f)] = float(np.sqrt(np.median(q[m]) / cal.CHI2_MEDIAN))
            n_own += 1
        else:
            sig[int(f)] = glob
    return sig, glob, n_own


def single_frame_z2(rec, sigma):
    s = rc._sigma_of(rec.frame, sigma)
    return np.einsum("ri,rij,rj->r", rec.eps, rec.Lam0, rec.eps) / s**2


def metric_rows(z2v, ld, tiles, groups, n_boot, seed):
    rows = []
    for name, sel in groups:
        idx = np.flatnonzero(sel)
        row = {"group": name, **cal.summary(z2v[idx], ld[idx])}
        if len(idx) >= 50 and len(np.unique(tiles[idx])) >= 2:
            ci = cal.bootstrap_summary(z2v[idx], ld[idx], tiles[idx], n_boot=n_boot, seed=seed)
            row["ci"] = ci
        rows.append(row)
    return rows


def class_groups(cls, quint, base):
    g = [("all", base), ("land", base & (cls != clm.WATER))]
    g += [(clm.NAMES[c], base & (cls == c)) for c in (clm.WATER, clm.ROOF, clm.GROUND)]
    g += [(f"contrast_q{k}", base & (quint == k)) for k in range(1, 6)]
    return g


def decide(lo, hi, tau, supported_if="above"):
    if supported_if == "above":
        return "supported" if lo > tau else ("refuted" if hi <= tau else "inconclusive")
    raise ValueError(supported_if)


def law_table(terms, ent, pool, sel_base, rho_fn, n_boot, seed):
    """A11 rows per n-bin. rho comes from the entities ``pool`` (those the repetition sums cover);
    the tile bootstrap resamples the tiles of ``pool``, recomputes rho from the resample and the
    ratio from the resampled entities of the bin, so rho's uncertainty is carried."""
    rows = []
    tiles = ent.tile_id
    rho_all = rho_fn(None)
    for (a_, b_), lab in zip(rp.N_BINS, rp.n_bin_labels()):
        in_bin = sel_base & (terms["n"] >= a_) & (terms["n"] <= b_)
        sel = np.flatnonzero(in_bin)
        if not len(sel):
            continue
        row = {"n_bin": lab, **rp.law_ratio(terms, rho_all, sel), "rho": rho_all}
        if len(sel) >= 50 and len(np.unique(tiles[sel])) >= 2:

            def stat(i, in_bin=in_bin):
                ii = pool[i]
                jj = ii[in_bin[ii]]
                if not len(jj):
                    return np.nan
                return rp.law_ratio(terms, rho_fn(ii), jj)["ratio"]

            row["ratio_ci"] = cal.tile_bootstrap(stat, tiles[pool], n_boot=n_boot, seed=seed)
        rows.append(row)
    return rows


def parallax(K, views, depths, ent, textured, step, scales=(1, 2, 4, 8)):
    """A15: image displacement between neighbouring frames left by aligned depth against exact depth."""
    by = {}
    for i, v in enumerate(views):
        by.setdefault(v["strip_id"], []).append(i)
    disp = []
    for idx in by.values():
        idx = sorted(idx, key=lambda i: views[i]["strip_pos"])
        for i, j in itertools.pairwise(idx):
            vi, vj = views[i], views[j]
            eid = ent.pixel_entity[vi["frame"]]
            ok = (eid >= 0) & np.isfinite(depths[i]) & np.isfinite(vi["depth"])
            ok &= textured[np.where(eid >= 0, eid, 0)] & (eid >= 0)
            if not ok.any():
                continue
            rays = pixel_rays_world(K, vi["R"], *vi["depth"].shape)[ok]
            Xh = vi["c"] + depths[i][ok][:, None] * rays
            Xe = vi["c"] + vi["depth"][ok][:, None] * rays
            uh, vh, _ = project(K, vj["R"], vj["c"], Xh)
            ue, ve, _ = project(K, vj["R"], vj["c"], Xe)
            disp.append(np.hypot(uh - ue, vh - ve) * step)  # full-resolution pixels
    if not disp:
        return {}
    d = np.concatenate(disp)
    d = d[np.isfinite(d)]
    return {f"p90_px_scale{s}": float(np.percentile(d, 90) / s) for s in scales} | {
        "n_pixels": int(d.size)
    }


# ----------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["matrixcity", "synthetic"], default="matrixcity")
    ap.add_argument("--data-root")
    ap.add_argument("--transforms")
    ap.add_argument("--images")
    ap.add_argument("--scene", default="block3")
    ap.add_argument(
        "--source", choices=["cache", "exact_independent", "exact_repeated"], default="cache"
    )
    ap.add_argument("--cache-root")
    ap.add_argument("--model", default="depth_pro")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--entity", type=float, default=8.0)
    ap.add_argument("--tile", type=float, default=64.0)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--huber", type=float, default=0.05)
    ap.add_argument("--alternations", type=int, default=0)
    ap.add_argument("--beta", choices=["on", "off"], default="off")
    ap.add_argument("--datum", choices=["parallax", "calibration"], default="calibration")
    ap.add_argument("--water-level", type=float, default=-3.0)
    ap.add_argument("--roof-height", type=float, default=6.0)
    ap.add_argument("--white-noise", type=float, default=0.002)
    ap.add_argument("--planted-rho", type=float, default=0.3)
    ap.add_argument("--planted-tau", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.source == "exact_independent":
        a.planted_rho = 0.0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([a.seed, 0xA]))
    t0 = time.time()
    log = {"args": vars(a)}

    # A1: strips and tiles
    K, views, images = load_inputs(a)
    views = tl.assign_strips(views)
    rows = tl.strip_table(views, footprint_along(K, views))
    (out / "strips.md").write_text(
        f"# Strip table ({a.block} {a.scene}, step {a.step})\n\n{len(views)} frames, {len(rows)} strips.\n\n"
        + tl.strip_table_markdown(rows)
        + "\n"
    )
    log["strips"] = {"n_frames": len(views), "n_strips": len(rows)}
    finfo = {v["frame"]: v for v in views}

    # entities, split and A3 classes (exact depth: evaluation only)
    ent = rc.build_entities(K, views, side=a.entity, tile=a.tile, group=a.group)
    cls = clm.geometric_classes(ent, a.water_level, a.roof_height)
    contrast = (
        clm.entity_contrast(ent, contrast_maps(a, views, images))
        if images
        else np.full(ent.M, np.nan)
    )
    quint = clm.contrast_quintile(contrast, cls)
    ctab = clm.class_table(ent, cls, quint)
    (out / "classes.md").write_text(
        "# Entity classes\n\n| class | entities | area (m2) | interval claim |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {r['class']} | {r['entities']} | {r['area_m2']:.0f} | {'yes' if r['claim'] else 'no (<50)'} |"
            for r in ctab
        )
        + "\n"
    )
    log["entities"] = {
        "M": ent.M,
        "roles": {tl.ROLE_NAMES[k]: int((ent.role == k).sum()) for k in tl.ROLE_NAMES},
        "classes": ctab,
    }

    # A2 input / A13 control
    if a.source == "cache":
        raw, mode = cache_raw(a, views)
    else:
        raw, mode = synthetic_raw(a, K, views, ent, rng)

    # A5: label-free alignment, and the oracle for comparison
    al = alg.align_block(
        K, views, raw, mode=mode, n_alt=a.alternations, huber=a.huber, fit_beta=a.beta == "on"
    )
    resid_parallax = depth_residual(al.depth, views)
    datum_q = 1.0
    if a.datum == "calibration":
        datum_q = block_datum(views, al.depth, ent)
        al.depth = [d * datum_q for d in al.depth]
    po = alg.oracle_align(mode, raw, [v["depth"] for v in views], a.huber)
    d_or = [alg.aligned_depth(mode, raw[i], po[i], 0.0) for i in range(len(views))]
    log["alignment"] = {
        "mode": mode,
        "gauge_k": al.k,
        "beta": al.beta,
        "history": al.history,
        "converged_0.1pct": bool(al.history[-1] < 1e-3),
        "label_free_parallax_datum": resid_parallax,
        "datum": a.datum,
        "datum_factor": datum_q,
        "label_free": depth_residual(al.depth, views),
        "oracle": depth_residual(d_or, views),
    }

    # A7: records; additivity
    rec = rc.concat([rc.frame_records(K, v, al.depth[i], ent) for i, v in enumerate(views)])
    land = cls != clm.WATER
    sigma, s_glob, n_own = calibrate_sigma(rec, ent, land)  # A8
    LamF, epsF, nF = rc.fuse(rec, sigma)
    fr_sorted = np.unique(rec.frame)
    half = np.isin(rec.frame, fr_sorted[::2])
    m2 = rc.merge_fused(rc.fuse(rec.subset(half), sigma), rc.fuse(rec.subset(~half), sigma))
    seen = nF > 0
    add_err = float(
        np.max(
            np.linalg.norm(m2[0][seen] - LamF[seen], axis=(1, 2))
            / np.linalg.norm(LamF[seen], axis=(1, 2))
        )
    )
    rec.save(
        out / "records.npz",
        sigma=np.array([sigma[int(f)] for f in fr_sorted]),
        sigma_frames=fr_sorted,
        fused_Lam=LamF,
        fused_eps=epsF,
        fused_n=nF,
        cls=cls,
        quint=quint,
        role=ent.role,
        tile=ent.split_tile,
        p0=ent.p0,
        normal=ent.n,
    )
    log["records"] = {
        "n_records": int(len(rec.entity)),
        "entities_seen": int(seen.sum()),
        "median_n": float(np.median(nF[seen])),
        "additivity_rel": add_err,
        "median_pixels_per_record": float(np.median(rec.npix)),
        "median_shape_rms_m": float(np.median(rec.shape_rms)),
    }

    # A8 check on held-out tiles
    z1 = single_frame_z2(rec, sigma)
    held_r = ent.role[rec.entity] == tl.HELD
    lr = held_r & land[rec.entity]
    sz1 = cal.tile_bootstrap(
        lambda i: cal.robust_scale(z1[lr][i]),
        ent.tile_id[rec.entity][lr],
        n_boot=min(a.n_boot, 2000),
        seed=a.seed,
    )
    log["single_frame"] = {
        "sigma_global": s_glob,
        "frames_with_own_sigma": n_own,
        "s_z_heldout_land": sz1,
        "pass": bool(0.9 <= sz1["estimate"] <= 1.1),
    }

    # A9: H1
    held = ent.role == tl.HELD
    base = held & (nF >= MIN_N_H1)
    z2F = np.full(ent.M, np.nan)
    ldF = np.full(ent.M, np.nan)
    z2F[seen] = cal.z2(epsF[seen], LamF[seen])
    ldF[seen] = cal.logdet(LamF[seen])
    h1 = metric_rows(z2F, ldF, ent.tile_id, class_groups(cls, quint, base), a.n_boot, a.seed)
    land_row = next(r for r in h1 if r["group"] == "land")
    ci = land_row.get("ci", {}).get("s_z")
    log["H1"] = {"rows": h1, "decision": decide(ci["lo"], ci["hi"], 1.25) if ci else "no interval"}

    # A10: repetition
    ps = rp.pair_sums(rec.subset(held_r), finfo)
    psn = rp.pair_sums(rec.subset(held_r), finfo, rng=np.random.default_rng(a.seed + 1))
    rs = rp.r_of_s(ps)
    rep = {
        "r_s": rs["r"][1:].tolist(),
        "pairs_s": rs["n_pairs"][1:].tolist(),
        "r_cross_strip": rs["r_cross_strip"],
        "r_cross_heading": rs["r_cross_heading"],
        "rho": {},
        "rho_null": {},
    }
    ents_held = np.flatnonzero(held)
    for B in B_LIST:
        est = rp.rho_hat(ps, B)
        bs = cal.tile_bootstrap(
            lambda i, B=B: rp.rho_hat(ps, B, ents_held[i]),
            ent.tile_id[ents_held],
            n_boot=min(a.n_boot, 2000),
            seed=a.seed + B,
        )
        nul = cal.tile_bootstrap(
            lambda i, B=B: rp.rho_hat(psn, B, ents_held[i]),
            ent.tile_id[ents_held],
            n_boot=min(a.n_boot, 2000),
            seed=a.seed + 50 + B,
        )
        rep["rho"][B] = {**bs, "estimate": est}
        rep["rho_null"][B] = nul
    log["repetition"] = rep

    # A11: H2 law (rho beyond the primary block, recomputed inside the bootstrap)
    terms = rp.law_terms(rec, epsF, nF)
    rho_fn = lambda ii: rp.rho_hat(ps, B_PRIMARY, ii)
    law = law_table(terms, ent, ents_held, held & seen, rho_fn, min(a.n_boot, 2000), a.seed)
    rho_ci = rep["rho"][B_PRIMARY]
    ratios = [r["ratio_ci"] for r in law if "ratio_ci" in r and r["n_bin"] not in ("1",)]
    if rho_ci["hi"] < 0.1 or any(r["lo"] > 2 or r["hi"] < 0.5 for r in ratios):
        h2 = "refuted"
    elif ratios and all(r["lo"] >= 0.67 and r["hi"] <= 1.5 for r in ratios):
        h2 = "supported"
    else:
        h2 = "inconclusive"
    log["H2"] = {"rows": law, "rho_B3": rho_ci, "decision": h2}

    # A12: consensus gate, re-fused without flagged frames
    gate_rows = {}
    for B in (1, 3, 5, 9):
        st = cg.frame_statistics(rec, finfo, sigma, B=B)
        g = cg.gate(st)
        keep = ~np.isin(rec.frame, g["flagged"])
        Lg, eg, ng = rc.fuse(rec.subset(keep), sigma)
        sg = held & (ng >= MIN_N_H1) & land
        z2g = cal.z2(np.nan_to_num(eg[sg]), Lg[sg])
        row = {
            "removed": len(g["flagged"]),
            "temperature": g["temperature"],
            **cal.summary(z2g, cal.logdet(Lg[sg])),
        }
        if sg.sum() >= 50:
            row["s_z_ci"] = cal.tile_bootstrap(
                lambda i, z=z2g: cal.robust_scale(z[i]),
                ent.tile_id[sg],
                n_boot=min(a.n_boot, 2000),
                seed=a.seed,
            )
        gate_rows[B] = row
    log["consensus_gate"] = gate_rows
    gB = gate_rows[B_PRIMARY].get("s_z_ci")
    if log["H1"]["decision"] == "refuted" or (gB and gB["hi"] <= 1.25):
        log["H1"]["after_gate"] = "honest after gate" if gB and gB["hi"] <= 1.25 else "n/a"

    # A15: residual parallax on textured land
    textured = land & ((quint >= 3) | (quint == 0) & ~np.isfinite(contrast))
    log["parallax"] = parallax(K, views, al.depth, ent, textured, a.step)

    log["runtime_s"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(log, indent=1, default=_json))
    (out / "law_figure.json").write_text(
        json.dumps(
            {
                "n_bin": [r["n_bin"] for r in law],
                "R_real": [r["R_real"] for r in law],
                "R_pred": [r["R_pred"] for r in law],
                "R_nominal": [r["R_nominal"] for r in law],
                "rho_by_class": _rho_by_class(rec, held_r, finfo, cls),
            },
            indent=1,
            default=_json,
        )
    )
    (out / "summary.md").write_text(render_summary(log))
    print((out / "summary.md").read_text())
    return log


def _rho_by_class(rec, held_r, finfo, cls):
    out = {}
    for c, name in clm.NAMES.items():
        m = held_r & (cls[rec.entity] == c)
        if m.sum() < 50:
            continue
        out[name] = rp.rho_hat(rp.pair_sums(rec.subset(m), finfo), B_PRIMARY)
    return out


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}g}"


def _ci(c, nd=3):
    return f"{_f(c['estimate'], nd)} [{_f(c['lo'], nd)}, {_f(c['hi'], nd)}]" if c else "n/a"


def render_summary(log) -> str:
    a = log["args"]
    L = [
        f"# Stage A summary: {a['block']} {a['scene']}, source {a['source']}"
        + (f" ({a['model']})" if a["source"] == "cache" else "")
        + f", step {a['step']}",
        "",
    ]
    L.append(
        f"Frames {log['strips']['n_frames']} in {log['strips']['n_strips']} strips; entities {log['entities']['M']} "
        f"(side {a['entity']} m; split tiles {a['tile']} m, groups of {a['group']}): {log['entities']['roles']}."
    )
    L.append(
        "Tables use one block only and are marked **same scene**."
        if a["block"] == "matrixcity"
        else "Synthetic block."
    )
    L += [
        "",
        "## Decisions",
        "",
        f"- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **{log['H1']['decision']}**.",
        f"- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **{log['H2']['decision']}**.",
        f"- A8 single-frame calibration on held-out land: s_z = {_ci(log['single_frame']['s_z_heldout_land'])}, "
        f"{'pass' if log['single_frame']['pass'] else 'FAIL (noise model at fault, H1 undefined)'}.",
        "",
    ]
    al = log["alignment"]
    L += [
        "## Alignment (A5)",
        "",
        f"Mode {al['mode']}, gauge k = {_f(al['gauge_k'], 4)}, beta = {_f(al['beta'])}, last change {_f(al['history'][-1])} "
        f"({'converged' if al['converged_0.1pct'] else 'not below 0.1%'}).",
        f"Block datum: {al['datum']} (factor {_f(al['datum_factor'], 5)}).",
        f"Median relative depth residual: label-free with parallax datum {_f(al['label_free_parallax_datum']['median_rel'])}, "
        f"after datum {_f(al['label_free']['median_rel'])}, oracle per-frame fit {_f(al['oracle']['median_rel'])}.",
        "",
    ]
    r = log["records"]
    L += [
        "## Records (A7)",
        "",
        f"{r['n_records']} records, {r['entities_seen']} entities seen, median n = {_f(r['median_n'])}, "
        f"median pixels per record {_f(r['median_pixels_per_record'])}, median shape remainder {_f(r['median_shape_rms_m'])} m; "
        f"additivity max relative difference {_f(r['additivity_rel'])}.",
        "",
    ]
    L += [
        "## H1: depth-only map on held-out tiles, n >= 5 (A9)",
        "",
        "| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in log["H1"]["rows"]:
        ci = row.get("ci", {})
        L.append(
            f"| {row['group']} | {row['n']} | {_ci(ci.get('s_z')) if ci else _f(row['s_z'])} | {_f(row['s_z_tail'])} | "
            f"{_f(row['cov_0.683'])} | {_f(row['cov_0.95'])} | {_f(row['ece'])} | {_f(row.get('log_score'))} | {_f(row['calibration_loss'])} |"
        )
    L += [
        "",
        "Consensus gate (A12), land, after removing flagged frames:",
        "",
        "| B | frames removed | temperature | s_z | cov 0.95 |",
        "|---|---|---|---|---|",
    ]
    for B, row in log["consensus_gate"].items():
        L.append(
            f"| {B} | {row['removed']} | {_f(row['temperature'])} | {_ci(row.get('s_z_ci')) if row.get('s_z_ci') else _f(row['s_z'])} | {_f(row['cov_0.95'])} |"
        )
    rep = log["repetition"]
    L += [
        "",
        "## Repetition (A10)",
        "",
        "r(s), same strip: "
        + ", ".join(f"s={i + 1}: {_f(x)}" for i, x in enumerate(rep["r_s"][:10])),
        f"Cross strip r = {_f(rep['r_cross_strip'])}; cross heading r = {_f(rep['r_cross_heading'])}.",
        "",
        "| B | rho_hat(B) | permutation null |",
        "|---|---|---|",
    ]
    for B in rep["rho"]:
        L.append(f"| {B} | {_ci(rep['rho'][B])} | {_ci(rep['rho_null'][B])} |")
    L += [
        "",
        f"## H2: the law (A11), rho_hat(B={B_PRIMARY}) = {_ci(log['H2']['rho_B3'])}",
        "",
        "| n bin | entities | R_real | R_pred | 1/n | real/pred |",
        "|---|---|---|---|---|---|",
    ]
    for row in log["H2"]["rows"]:
        L.append(
            f"| {row['n_bin']} | {row['n_ent']} | {_f(row['R_real'])} | {_f(row['R_pred'])} | {_f(row['R_nominal'])} | "
            f"{_ci(row['ratio_ci']) if 'ratio_ci' in row else _f(row['ratio'])} |"
        )
    px = log.get("parallax", {})
    L += [
        "",
        "## Residual parallax on textured land (A15)",
        "",
        ", ".join(f"{k}: {_f(v)}" for k, v in px.items()) or "n/a",
        "",
        f"Runtime {log['runtime_s'] / 60:.1f} min.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    main()
