"""Repetition of single-frame placement errors and the fused-variance law.

Pair correlation at same-strip separation s (protocol, Section 4):

    r(s) = <eps_mj, eps_mk> / <||eps_mj||^2>,

pooled over entities and over pairs (j, k) of one strip with |pos_j - pos_k| = s. Cross-strip
(same heading, other strip) and cross-heading pairs are pooled separately. The repeated share
rho_hat(B) is the mean of r(s) over the separations s > B for which pairs exist.

Law: per bin of n (frames that see the entity), R_real = mean ||eps_bar_m||^2 / mean ||eps_mj||^2
against R_pred = mean over entities of rho + (1 - rho)/n_m, and the nominal 1/n.

All sums are kept per entity, so a tile bootstrap recomputes r(s), rho_hat and the ratio from
resampled entities without touching the records again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

N_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10_000))


def _yaw_class(yaw: float) -> int:
    return int(np.round((yaw % 180.0) / 90.0)) % 2


def pair_sums(
    rec, frame_info: Mapping[int, Mapping], s_max: int = 12, rng: np.random.Generator | None = None
):
    """Per-entity sums for r(s).

    Returns dict with ``self`` (M,) sum ||eps||^2, ``nself`` (M,), ``same`` (M, s_max+1) inner-product
    sums at separations 0..s_max (0 unused), ``nsame`` counts, and ``cross_strip`` / ``cross_heading``
    sums and counts (M,). ``frame_info[f]`` needs strip_id, strip_pos, yaw. With ``rng`` the records
    of each frame are shuffled across entities first (the permutation null)."""
    ent = rec.entity.copy()
    if rng is not None:
        for f in np.unique(rec.frame):
            i = np.flatnonzero(rec.frame == f)
            ent[i] = ent[rng.permutation(i)]
    M = rec.M
    eps = rec.eps
    out = {
        "self": np.bincount(ent, np.sum(eps * eps, 1), M),
        "nself": np.bincount(ent, minlength=M).astype(float),
        "same": np.zeros((M, s_max + 1)),
        "nsame": np.zeros((M, s_max + 1)),
        "cross_strip": np.zeros(M),
        "ncross_strip": np.zeros(M),
        "cross_heading": np.zeros(M),
        "ncross_heading": np.zeros(M),
    }
    sid = np.array([frame_info[int(f)]["strip_id"] for f in rec.frame])
    pos = np.array([frame_info[int(f)]["strip_pos"] for f in rec.frame])
    yc = np.array([_yaw_class(frame_info[int(f)]["yaw"]) for f in rec.frame])
    order = np.argsort(ent, kind="stable")
    e_sorted = ent[order]
    starts = np.flatnonzero(np.r_[True, e_sorted[1:] != e_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        idx = order[a:b]
        m = ent[idx[0]]
        E = eps[idx]
        G = E @ E.T
        iu = np.triu_indices(len(idx), 1)
        g = G[iu]
        same = sid[idx][iu[0]] == sid[idx][iu[1]]
        sep = np.abs(pos[idx][iu[0]] - pos[idx][iu[1]])
        sameh = yc[idx][iu[0]] == yc[idx][iu[1]]
        sel = same & (sep <= s_max)
        np.add.at(out["same"][m], sep[sel], g[sel])
        np.add.at(out["nsame"][m], sep[sel], 1)
        cs = (~same) & sameh
        out["cross_strip"][m] += g[cs].sum()
        out["ncross_strip"][m] += cs.sum()
        ch = ~sameh
        out["cross_heading"][m] += g[ch].sum()
        out["ncross_heading"][m] += ch.sum()
    return out


def r_of_s(ps: Mapping, idx: np.ndarray | None = None) -> dict:
    """r(s) for s = 1..s_max, cross-strip r and cross-heading r, from (resampled) entities ``idx``."""
    if idx is None:
        idx = np.arange(len(ps["self"]))
    denom = ps["self"][idx].sum() / max(ps["nself"][idx].sum(), 1)
    num = ps["same"][idx].sum(0)
    cnt = ps["nsame"][idx].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (num / cnt) / denom
        rcs = (ps["cross_strip"][idx].sum() / ps["ncross_strip"][idx].sum()) / denom
        rch = (ps["cross_heading"][idx].sum() / ps["ncross_heading"][idx].sum()) / denom
    return {"r": r, "n_pairs": cnt, "r_cross_strip": float(rcs), "r_cross_heading": float(rch)}


def rho_hat(ps: Mapping, B: int, idx: np.ndarray | None = None) -> float:
    rr = r_of_s(ps, idx)
    r, n = rr["r"], rr["n_pairs"]
    s = np.arange(len(r))
    sel = (s > max(B, 0)) & (n > 0)
    return float(np.mean(r[sel])) if sel.any() else float("nan")


def law_terms(rec, fused_eps: np.ndarray, n_frames: np.ndarray) -> dict:
    """Per-entity terms of the law: single-frame sum ||eps_mj||^2 and count, fused ||eps_bar||^2."""
    M = rec.M
    s1 = np.bincount(rec.entity, np.sum(rec.eps**2, 1), M)
    c1 = np.bincount(rec.entity, minlength=M).astype(float)
    fz = np.where(np.isfinite(fused_eps).all(1), np.sum(np.nan_to_num(fused_eps) ** 2, 1), 0.0)
    return {"single_sum": s1, "single_n": c1, "fused_sq": fz, "n": n_frames}


def law_ratio(terms: Mapping, rho: float, sel: np.ndarray) -> dict:
    """R_real, R_pred, nominal 1/n and real/pred for the entities ``sel`` (index array)."""
    if len(sel) == 0:
        return {
            "R_real": np.nan,
            "R_pred": np.nan,
            "R_nominal": np.nan,
            "ratio": np.nan,
            "n_ent": 0,
        }
    single = terms["single_sum"][sel].sum() / terms["single_n"][sel].sum()
    R_real = terms["fused_sq"][sel].mean() / single
    n = terms["n"][sel].astype(float)
    rho = (
        float(np.clip(rho, 0.0, 1.0)) if np.isfinite(rho) else rho
    )  # a share: negative estimates read as 0
    R_pred = float(np.mean(rho + (1 - rho) / n))
    return {
        "R_real": float(R_real),
        "R_pred": R_pred,
        "R_nominal": float(np.mean(1 / n)),
        "ratio": float(R_real / R_pred),
        "n_ent": int(len(sel)),
    }


def n_bin_labels(bins: Sequence[tuple[int, int]] = N_BINS) -> list[str]:
    return [f"{a}" if a == b else (f"{a}+" if b >= 10_000 else f"{a}-{b}") for a, b in bins]
