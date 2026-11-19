"""Depth-consensus gate on placement records (the baseline of the letter).

Frame j is tested against the map built without it and without the ``B`` frames on either side of
it in its strip (frames that share its error would otherwise vouch for it). Per entity the
discrepancy d = eps_mj - eps_m(-j) has covariance Lambda_mj^-1 + Lambda_m(-j)^-1; the frame's
statistic is T_j = sum_m d^T S^-1 d with 3 x (entities) degrees of freedom. The scene temperature
(median of T_j / dof over frames) is divided out, as in the view gate, so that only frames that
disagree more than the typical frame are flagged; flags use Holm at level ``alpha``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats

from sensorfault.stagea.records import _sigma_of


def frame_statistics(rec, frame_info: Mapping[int, Mapping], sigma, B: int = 3) -> dict[int, dict]:
    s = _sigma_of(rec.frame, sigma)
    Lam = rec.Lam0 / (s**2)[:, None, None]
    eta = np.einsum("rij,rj->ri", Lam, rec.eps)
    M = rec.M
    LamF = np.zeros((M, 3, 3))
    etaF = np.zeros((M, 3))
    np.add.at(LamF, rec.entity, Lam)
    np.add.at(etaF, rec.entity, eta)
    sid = np.array([frame_info[int(f)]["strip_id"] for f in rec.frame])
    pos = np.array([frame_info[int(f)]["strip_pos"] for f in rec.frame])
    out = {}
    for f in np.unique(rec.frame):
        fi = frame_info[int(f)]
        mine = rec.frame == f
        # records of frames inside the block around f (same strip, |pos - pos_f| <= B), f included
        blk = (sid == fi["strip_id"]) & (np.abs(pos - fi["strip_pos"]) <= B)
        Lb = np.zeros((M, 3, 3))
        eb = np.zeros((M, 3))
        np.add.at(Lb, rec.entity[blk], Lam[blk])
        np.add.at(eb, rec.entity[blk], eta[blk])
        Lr, er = LamF - Lb, etaF - eb
        ents = rec.entity[mine]
        Lo = Lr[ents]
        ok = np.linalg.det(Lo) > 0
        if not ok.any():
            out[int(f)] = {"T": 0.0, "dof": 0}
            continue
        ents, idx = ents[ok], np.flatnonzero(mine)[ok]
        eo = np.linalg.solve(Lr[ents], er[ents][..., None])[..., 0]
        d = rec.eps[idx] - eo
        S = np.linalg.inv(Lam[idx]) + np.linalg.inv(Lr[ents])
        T = float(np.einsum("ni,nij,nj->", d, np.linalg.inv(S), d))
        out[int(f)] = {"T": T, "dof": 3 * len(ents)}
    return out


def gate(stats_by_frame: Mapping[int, Mapping], alpha: float = 0.01) -> dict:
    fr = [f for f, s in stats_by_frame.items() if s["dof"] > 0]
    ratio = np.array([stats_by_frame[f]["T"] / stats_by_frame[f]["dof"] for f in fr])
    temp = float(np.median(ratio)) if len(ratio) else 1.0
    p = np.array(
        [stats.chi2.sf(stats_by_frame[f]["T"] / temp, stats_by_frame[f]["dof"]) for f in fr]
    )
    order = np.argsort(p)
    m = len(p)
    flagged = []
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            flagged.append(fr[i])
        else:
            break
    return {"temperature": temp, "flagged": sorted(flagged), "p": dict(zip(fr, p.tolist()))}
