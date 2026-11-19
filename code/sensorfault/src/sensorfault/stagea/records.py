"""Entities on the exact surface and per-entity placement records of each frame.

Entity. Exact depth of every frame is back-projected; each point gets an 8 m ground cell, a normal
class (up if the normal is within ``up_deg`` of vertical, else steep) and a height layer (points of
one cell and class sorted by height and split where consecutive heights differ by more than
``layer_gap`` metres). One (cell, class, layer) is one entity. Its frame is the least-squares plane
of its pooled exact points: centre p0, normal n (towards +z), tangents t1, t2.

Placement. Within an entity, a frame's depth error dd = d_hat - d_exact at pixel k moves the surface
along n by dh = (n . r_k) dd, with r_k the pixel ray scaled to unit z-depth. The frame's placement
error is the weighted least-squares plane dh = o + a u + b v, with (u, v) the tangent coordinates
of the exact point divided by the half-width L = side / 2, so all three coefficients are metres.
With per-pixel depth noise sigma_i * d_hat_k (relative noise of frame i), the frame's information is

    Lambda_mi = sum_k phi_k phi_k^T / (sigma_i d_hat_k |n . r_k|)^2,   phi_k = (1, u_k, v_k),

and eps_mi = Lambda_mi^-1 sum_k phi_k dh_k / (...)^2. Records store Lambda at sigma_i = 1
(``Lam0``); the calibrated information is Lam0 / sigma_i^2. The fused depth-only entity is the
information-weighted combination, which is exactly the least-squares fit of all frames' pixels
pooled, so records are additive by construction.

Correlated pixel errors. Monocular depth errors are smooth: within one record the pixel errors
share most of their size, so the information sum above grows with the pixel count while the actual
error does not. A record-level floor models the shared part: with h = (mean of 1 / (d_hat |n.r|)^2
over the record's pixels)^-1/2 the metres per unit relative depth error,

    Sigma_mi = sigma_i^2 Lambda0_mi^-1 + (tau_i h_mi)^2 I_3,

where sigma_i is the within-record (white) relative noise and tau_i the relative error shared by
the record's pixels. A record-specific multiplier g_mi of tau_i (``floor_scale``) lets the shared
error depend on record covariates such as surface class, incidence, depth and pixel count. ``record_information`` returns Sigma_mi^-1; tau = 0 gives Lam0 / sigma^2.
Records also keep the white noise estimate ``white`` (weighted residual sum of squares about the
plane per degree of freedom, at sigma = 1) and ``h``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from sensorfault.stagea import tiles as tl
from sensorfault.stagea.geometry import backproject, normals_from_points, pixel_rays_world


@dataclass
class Entities:
    side: float
    p0: np.ndarray  # (M, 3)
    n: np.ndarray  # (M, 3)
    t1: np.ndarray  # (M, 3)
    t2: np.ndarray  # (M, 3)
    cell: np.ndarray  # (M, 2) int
    normal_class: np.ndarray  # (M,) 0 up, 1 steep
    area: np.ndarray  # (M,) m^2 (1 m occupancy on the plane)
    plane_rms: np.ndarray  # (M,) rms of exact points about the plane
    n_points: np.ndarray  # (M,)
    split_tile: np.ndarray  # (M, 2) int, 64 m tiles
    role: np.ndarray  # (M,) tiles.CAL / HELD / BUFFER
    pixel_entity: dict = field(default_factory=dict)  # frame -> (h, w) entity id, -1 = none

    @property
    def M(self) -> int:
        return len(self.p0)

    @property
    def tile_id(self) -> np.ndarray:
        return tl.tile_key(self.split_tile)


def build_entities(
    K: np.ndarray,
    views: Sequence[Mapping],
    side: float = 8.0,
    tile: float = 64.0,
    up_deg: float = 30.0,
    layer_gap: float = 2.0,
    min_points: int = 10,
    group: int = 2,
    buffer: int = 1,
    origin: Sequence[float] = (0.0, 0.0),
) -> Entities:
    """Entities from the exact depth (``view['depth']``) of all views."""
    P_all, N_all, F_all, I_all = [], [], [], []
    shapes = {}
    for v in views:
        P = backproject(K, v["R"], v["c"], v["depth"])
        N = normals_from_points(P)
        h, w = v["depth"].shape
        shapes[v["frame"]] = (h, w)
        ok = np.isfinite(P).all(-1) & np.isfinite(N).all(-1)
        idx = np.flatnonzero(ok.ravel())
        P_all.append(P.reshape(-1, 3)[idx])
        N_all.append(N.reshape(-1, 3)[idx])
        F_all.append(np.full(len(idx), v["frame"]))
        I_all.append(idx)
    P = np.concatenate(P_all)
    N = np.concatenate(N_all)
    F = np.concatenate(F_all)
    I = np.concatenate(I_all)
    cell = tl.cell_index(P[:, :2], side, origin)
    ncls = (N[:, 2] < np.cos(np.deg2rad(up_deg))).astype(np.int64)
    ckey = tl.tile_key(cell) * 2 + ncls
    order = np.lexsort((P[:, 2], ckey))
    ck, z = ckey[order], P[order, 2]
    brk = np.ones(len(order), bool)
    brk[1:] = (ck[1:] != ck[:-1]) | (np.diff(z) > layer_gap)
    gid_sorted = np.cumsum(brk) - 1
    gid = np.empty_like(gid_sorted)
    gid[order] = gid_sorted
    counts = np.bincount(gid)
    keep_g = counts >= min_points
    remap = -np.ones(len(counts), np.int64)
    remap[keep_g] = np.arange(int(keep_g.sum()))
    eid = remap[gid]
    sel = eid >= 0
    P, N, F, I, eid, cell, ncls = P[sel], N[sel], F[sel], I[sel], eid[sel], cell[sel], ncls[sel]
    M = int(keep_g.sum())

    cnt = np.bincount(eid, minlength=M).astype(float)
    p0 = np.stack([np.bincount(eid, P[:, k], M) for k in range(3)], 1) / cnt[:, None]
    D = P - p0[eid]
    C = np.zeros((M, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            C[:, a, b] = C[:, b, a] = np.bincount(eid, D[:, a] * D[:, b], M) / cnt
    evals, evecs = np.linalg.eigh(C)
    n = evecs[:, :, 0]
    n = np.where(n[:, 2:3] < 0, -n, n)
    # tangent 1: world x projected on the plane (world y if the plane is near x-normal)
    ref = np.where(np.abs(n[:, :1]) > 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    t1 = ref - np.sum(ref * n, 1, keepdims=True) * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    plane_rms = np.sqrt(np.maximum(evals[:, 0], 0))
    # area: occupied 1 m cells on the plane
    uu = np.floor(np.sum(D * t1[eid], 1)).astype(np.int64)
    vv = np.floor(np.sum(D * t2[eid], 1)).astype(np.int64)
    occ = np.unique(np.stack([eid, uu, vv], 1), axis=0)
    area = np.bincount(occ[:, 0], minlength=M).astype(float)
    ecell = np.zeros((M, 2), np.int64)
    ecell[eid] = cell
    encls = np.zeros(M, np.int64)
    encls[eid] = ncls
    stile = tl.cell_index(p0[:, :2], tile, origin)
    role = tl.tile_role(stile, group=group, buffer=buffer)

    pix = {}
    for fr, (h, w) in shapes.items():
        m = -np.ones(h * w, np.int64)
        s = fr == F
        m[I[s]] = eid[s]
        pix[fr] = m.reshape(h, w)
    return Entities(
        side, p0, n, t1, t2, ecell, encls, area, plane_rms, cnt.astype(np.int64), stile, role, pix
    )


@dataclass
class Records:
    """Sparse per-(entity, frame) placement records at sigma = 1."""

    entity: np.ndarray  # (R,)
    frame: np.ndarray  # (R,)
    Lam0: np.ndarray  # (R, 3, 3) information at sigma_i = 1
    eps: np.ndarray  # (R, 3) metres
    npix: np.ndarray  # (R,)
    shape_rms: np.ndarray  # (R,) weighted rms of dh about the fitted plane (shape remainder)
    M: int = 0
    white: np.ndarray | None = None  # (R,) weighted rss / (npix - 3): within-record sigma^2
    h: np.ndarray | None = None  # (R,) metres per unit relative depth error
    nr: np.ndarray | None = None  # (R,) mean |n . r| of the record's pixels (incidence)
    floor_scale: np.ndarray | None = None  # (R,) record-specific multiplier of tau (default 1)

    def __post_init__(self):
        R = len(self.entity)
        for name, val in (("white", np.nan), ("h", np.nan), ("nr", np.nan), ("floor_scale", 1.0)):
            if getattr(self, name) is None:
                setattr(self, name, np.full(R, val))

    def subset(self, mask: np.ndarray) -> Records:
        return Records(
            self.entity[mask],
            self.frame[mask],
            self.Lam0[mask],
            self.eps[mask],
            self.npix[mask],
            self.shape_rms[mask],
            self.M,
            self.white[mask],
            self.h[mask],
            self.nr[mask],
            self.floor_scale[mask],
        )

    def save(self, path, **extra) -> None:
        np.savez_compressed(
            path,
            entity=self.entity,
            frame=self.frame,
            Lam0=self.Lam0,
            eps=self.eps,
            npix=self.npix,
            shape_rms=self.shape_rms,
            M=self.M,
            white=self.white,
            h=self.h,
            nr=self.nr,
            floor_scale=self.floor_scale,
            **extra,
        )


def neighbour_median(d: np.ndarray, k: int = 5) -> np.ndarray:
    """Median of the k x k neighbourhood of each pixel, the pixel itself left out (NaN-aware)."""
    h = k // 2
    P = np.pad(np.asarray(d, float), h, mode="constant", constant_values=np.nan)
    W = np.lib.stride_tricks.sliding_window_view(P, (k, k)).reshape(*d.shape, k * k)
    W = np.delete(W, (k * k) // 2, axis=-1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(W, axis=-1)


def frame_records(
    K: np.ndarray,
    view: Mapping,
    pred_depth: np.ndarray,
    ent: Entities,
    min_pixels: int = 4,
    max_cond: float = 1e6,
    weight_window: int = 5,
) -> Records:
    """Placement records of one frame (pred_depth aligned, same grid as view['depth'])."""
    d_ex = view["depth"]
    eidmap = ent.pixel_entity[view["frame"]]
    rays = pixel_rays_world(K, view["R"], *d_ex.shape)
    ok = (eidmap >= 0) & np.isfinite(pred_depth) & (pred_depth > 0) & np.isfinite(d_ex)
    e = eidmap[ok]
    r = rays[ok]
    X = view["c"] + d_ex[ok][:, None] * r
    dd = pred_depth[ok] - d_ex[ok]
    nr = np.sum(ent.n[e] * r, 1)
    dh = nr * dd
    L = 0.5 * ent.side
    D = X - ent.p0[e]
    u = np.sum(D * ent.t1[e], 1) / L
    v = np.sum(D * ent.t2[e], 1) / L
    # the weight must not depend on the pixel's own error: with multiplicative noise, weights from
    # the pixel's own predicted depth favour pixels whose noise pulls them closer, a bias that
    # grows linearly under fusion (s_z 1.17 on the block 3 independent control at sigma 0.044)
    d_w = neighbour_median(pred_depth, weight_window)
    d_w = np.where(np.isfinite(d_w) & (d_w > 0), d_w, pred_depth)
    wgt = 1.0 / (d_w[ok] * np.abs(nr)) ** 2
    phi = np.stack([np.ones_like(u), u, v], 1)
    M = ent.M
    A = np.zeros((M, 3, 3))
    b = np.zeros((M, 3))
    for i in range(3):
        b[:, i] = np.bincount(e, wgt * phi[:, i] * dh, M)
        for j in range(i, 3):
            A[:, i, j] = A[:, j, i] = np.bincount(e, wgt * phi[:, i] * phi[:, j], M)
    cnt = np.bincount(e, minlength=M)
    sw = np.bincount(e, wgt, M)
    swr2 = np.bincount(e, wgt * dh * dh, M)
    good = cnt >= min_pixels
    if good.any():
        cond = np.full(M, np.inf)
        cond[good] = np.linalg.cond(A[good])
        good &= cond < max_cond
    idx = np.flatnonzero(good)
    eps = np.linalg.solve(A[idx], b[idx][..., None])[..., 0]
    # weighted residual sum of squares = sum w dh^2 - eps^T b
    rss = np.maximum(swr2[idx] - np.sum(eps * b[idx], 1), 0)
    shape = np.sqrt(rss / sw[idx])
    white = rss / np.maximum(cnt[idx] - 3, 1)
    h = np.sqrt(cnt[idx] / sw[idx])
    nr_mean = np.bincount(e, np.abs(nr), M)[idx] / cnt[idx]
    return Records(
        idx, np.full(len(idx), view["frame"]), A[idx], eps, cnt[idx], shape, M, white, h, nr_mean
    )


def concat(recs: Sequence[Records]) -> Records:
    recs = [r for r in recs if len(r.entity)]
    M = max(r.M for r in recs)
    return Records(
        np.concatenate([r.entity for r in recs]),
        np.concatenate([r.frame for r in recs]),
        np.concatenate([r.Lam0 for r in recs]),
        np.concatenate([r.eps for r in recs]),
        np.concatenate([r.npix for r in recs]),
        np.concatenate([r.shape_rms for r in recs]),
        M,
        np.concatenate([r.white for r in recs]),
        np.concatenate([r.h for r in recs]),
        np.concatenate([r.nr for r in recs]),
        np.concatenate([r.floor_scale for r in recs]),
    )


def record_information(rec: Records, sigma=1.0, tau=None) -> np.ndarray:
    """Per-record information (R, 3, 3): (sigma^2 Lam0^-1 + (tau g h)^2 I)^-1, or Lam0 / sigma^2.

    ``sigma`` and ``tau`` are each a scalar, a frame -> value mapping or an array by frame; g is
    the record's ``floor_scale`` (1 unless a covariate model of the floor set it)."""
    s = _sigma_of(rec.frame, sigma)
    if tau is None:
        return rec.Lam0 / (s**2)[:, None, None]
    t = _sigma_of(rec.frame, tau) * rec.floor_scale
    if not np.any(t > 0):
        return rec.Lam0 / (s**2)[:, None, None]
    C = (s**2)[:, None, None] * np.linalg.inv(rec.Lam0)
    C = C + ((t * rec.h) ** 2)[:, None, None] * np.eye(3)
    return np.linalg.inv(C)


def record_z2(rec: Records, sigma=1.0, tau=None) -> np.ndarray:
    """Single-frame z^2 of each record under the noise model (sigma, tau)."""
    return np.einsum("ri,rij,rj->r", rec.eps, record_information(rec, sigma, tau), rec.eps)


def fuse(rec: Records, sigma: Mapping[int, float] | np.ndarray | float = 1.0, tau=None):
    """Depth-only fused entities: (Lam (M, 3, 3), eps (M, 3), n_frames (M,)).

    ``sigma`` (and the optional record floor ``tau``) is a scalar, a frame -> value mapping, or an
    array indexed by frame."""
    Lam = record_information(rec, sigma, tau)
    M = rec.M
    LamF = np.zeros((M, 3, 3))
    eta = np.zeros((M, 3))
    np.add.at(LamF, rec.entity, Lam)
    np.add.at(eta, rec.entity, np.einsum("rij,rj->ri", Lam, rec.eps))
    nF = np.bincount(rec.entity, minlength=M)
    eps = np.full((M, 3), np.nan)
    ok = nF > 0
    eps[ok] = np.linalg.solve(LamF[ok], eta[ok][..., None])[..., 0]
    return LamF, eps, nF


def merge_fused(a, b):
    """Merge two fused records (Lam, eps, n) of disjoint frame sets."""
    La, ea, na = a
    Lb, eb, nb = b
    L = La + Lb
    eta = np.einsum("mij,mj->mi", La, np.nan_to_num(ea)) + np.einsum(
        "mij,mj->mi", Lb, np.nan_to_num(eb)
    )
    e = np.full_like(ea, np.nan)
    ok = (na + nb) > 0
    e[ok] = np.linalg.solve(L[ok], eta[ok][..., None])[..., 0]
    return L, e, na + nb


def _sigma_of(frames: np.ndarray, sigma) -> np.ndarray:
    if np.isscalar(sigma):
        return np.full(len(frames), float(sigma))
    if isinstance(sigma, Mapping):
        return np.array([sigma[int(f)] for f in frames], float)
    return np.asarray(sigma, float)[frames]
