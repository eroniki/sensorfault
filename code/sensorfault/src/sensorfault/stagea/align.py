"""Label-free per-frame alignment of monocular depth to the fused depth-only map.

Model. Frame i's aligned depth is ``base_i(x) + beta * base_i(x)^2 / D0`` with raw model output x,
``base_i = s_i x + o_i`` for metric models and ``base_i = 1 / (a_i x + b_i)`` for relative
inverse-depth models (Depth Anything V2), and one distance coefficient beta shared by all frames
(D0 = 100 m only sets its unit).

Procedure (no ground truth anywhere):

1. Global gauge. One factor k for all frames, chosen so that frames agree with each other through
   the known poses: frame i's points at depth k x_i, projected into an overlapping frame j, must
   land at the depth k x_j that j reports there. The median relative disagreement over pairs is
   scanned over k (log grid, then refined). Parallax between frames is what fixes k.
2. Relative scales. Every overlapping pair (i, j), of any heading, gives the log median ratio of
   frame i's depth seen from j to the depth j reports there. With frame i too large by a_i the
   ratio is a_i / a_j, so the log a are one robust weighted least-squares solve on the pair graph
   (median pinned to 0, so the common factor stays at the gauge). A few rounds, since rescaling
   a frame also moves its points through parallax; the largest change of each round is reported.
   A global solve does not inherit the errors of the neighbours a frame is compared with, which
   matters when per-frame scale errors are tens of percent (Depth Pro on MatrixCity block 3).
3. Alternations (optional, ``n_alt``; off by default). Render the current fused map into each
   frame (all other frames' aligned points, front surface, per-pixel median), refit the frame's
   scale, then beta from all frames. After each alternation the common factor is reset to the
   gauge of step 1 (the median over frames of the aligned median depth is held), since
   frame-to-frame fits cannot see it. The rendering of frame i leaves frame i out, so a frame
   does not anchor itself. Without beta, these refits move the scales away from the pair-graph
   solution when frames also carry shape errors, and do not bring them closer to the truth.

Offsets. With ``fit_offset=False`` (default) each frame gets a scale only (metric: o = 0;
relative: b = 0). In nadir strips the label-free target is itself noisy and the depth spread
within a frame is small, so a free per-frame offset is not identified: the scale-offset pair
drifts along a valley that rescales relief (errors in both variables). The synthetic control
of Stage A shows the drift; ``fit_offset=True`` is kept for comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from sensorfault.stagea.geometry import pixel_rays_world, project

D0 = 100.0
# std of the per-frame offset prior, as a share of the frame's median depth
OFFSET_PRIOR = {"metric": 0.02, "inverse": 0.2}
# least share of a frame's points that must land in a neighbour's view for a pair to count
MIN_OVERLAP = 0.2
# sd of the prior on a frame's log scale about the gauge (Depth Pro per-frame errors are ~0.3)
PRIOR_SD = 0.5


@dataclass
class AlignResult:
    mode: str
    k: float
    params: np.ndarray  # (F, 2): (s, o) or (a, b) per frame, in the frame order given
    beta: float
    depth: list  # aligned depth per frame
    history: list = field(
        default_factory=list
    )  # max relative change per graph round, then alternation
    neighbours: dict = field(default_factory=dict)


def _base(mode, x, p):
    if mode == "metric":
        return p[0] * x + p[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / (p[0] * x + p[1])
    return np.where(d > 0, d, np.nan)


def aligned_depth(mode: str, x: np.ndarray, p: Sequence[float], beta: float) -> np.ndarray:
    b = _base(mode, x, p)
    return b + beta * b * b / D0


def _huber_w(r, delta, cut=6.0):
    """Huber weights, set to zero beyond ``cut`` thresholds: far outliers with large raw values
    would otherwise act as leverage points and pull the scale towards zero."""
    a = np.abs(r)
    w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
    return np.where(a <= cut * delta, w, 0.0)


def _fit_frame(mode, x, t, beta, p_prev, delta, iters=10, offset_prior=None, fit_offset=True):
    """Huber fit of one frame's (s, o) or (a, b) to target depth ``t``.

    The raw output is regressed on the target, x = A u + C with u = base target (metric) or its
    inverse (relative), and the parameters are recovered as (1/A, -C/A). Regressing the target on
    the raw output instead would suffer regression dilution: in nadir frames the spread of depth
    within a frame is small against the model's error, which biases the scale towards zero.
    """
    ok = np.isfinite(x) & np.isfinite(t) & (t > 0)
    if mode == "inverse":
        ok &= x > 0
    x, t = x[ok], t[ok]
    if x.size < 20:
        return p_prev
    u = _invert_distance(t, beta)
    if mode == "inverse":
        u = 1.0 / u
    A_, C_ = 1.0 / p_prev[0], -p_prev[1] / p_prev[0]
    # robust restart of the slope when the previous parameters are far off: the redescending
    # weights below would otherwise reject every pixel
    ratio = np.median(x / u)
    if (
        np.isfinite(ratio)
        and ratio > 0
        and abs(np.median((A_ * u + C_ - x) / (A_ * u))) > 2 * delta
    ):
        A_, C_ = float(ratio), 0.0
    prior = OFFSET_PRIOR[mode] if offset_prior is None else offset_prior
    um = float(np.median(u))
    for _ in range(iters):
        pred = A_ * u + C_
        r = (pred - x) / np.abs(A_ * u)
        w = _huber_w(r, delta) / (A_ * u) ** 2
        if not np.any(w > 0):
            w = _huber_w(r, delta, cut=np.inf) / (A_ * u) ** 2
        D = np.stack([u, np.ones_like(u)], 1)
        DtD = D.T @ (D * w[:, None])
        Dtx = D.T @ (w * x)
        # prior: the offset, as a share of the median depth (or inverse depth), is zero with std
        # ``prior``, weighed against the robust scale of the relative residuals. It breaks the
        # near-collinearity of scale and offset in nadir frames.
        if prior:
            sd = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-3)
            DtD[1, 1] += (sd / prior) ** 2 / (A_ * um) ** 2
        if not fit_offset:
            A_, C_ = Dtx[0] / DtD[0, 0], 0.0
        else:
            try:
                A_, C_ = np.linalg.solve(DtD, Dtx)
            except np.linalg.LinAlgError:
                break
        if not np.isfinite(A_) or A_ == 0:
            return p_prev
    return np.array([1.0 / A_, -C_ / A_])


def _invert_distance(t, beta):
    """Base depth b with b + beta b^2 / D0 = t."""
    if abs(beta) < 1e-12:
        return t
    return (-1 + np.sqrt(np.maximum(1 + 4 * beta * t / D0, 1e-12))) / (2 * beta / D0)


def _fit_beta(bases, targets, delta, beta_prev, ridge=1e-3, iters=10):
    b = np.concatenate([x.ravel() for x in bases])
    t = np.concatenate([x.ravel() for x in targets])
    ok = np.isfinite(b) & np.isfinite(t) & (t > 0)
    b, t = b[ok], t[ok]
    g = b * b / D0
    beta = beta_prev
    for _ in range(iters):
        r = (b + beta * g - t) / t
        w = _huber_w(r, delta) / t**2
        beta = float(np.sum(w * g * (t - b)) / (np.sum(w * g * g) * (1 + ridge) + 1e-30))
    return beta


def neighbours(views: Sequence[Mapping], radius: float) -> dict[int, list[int]]:
    """Frames (by position in ``views``) whose camera centres lie within ``radius`` (xy) of each other."""
    C = np.array([v["c"][:2] for v in views])
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    return {i: [j for j in np.flatnonzero(D[i] < radius) if j != i] for i in range(len(views))}


def render_into(K, view_i, views, depths, cand, tol=0.02):
    """Median front-surface depth of the frames ``cand`` rendered into ``view_i`` (NaN = no data)."""
    h, w = view_i["depth"].shape if "depth" in view_i else depths[0].shape
    us, vs, zs = [], [], []
    for j in cand:
        d = depths[j]
        X = views[j]["c"] + d[..., None] * pixel_rays_world(K, views[j]["R"], *d.shape)
        X = X.reshape(-1, 3)
        X = X[np.isfinite(X).all(1)]
        u, v, z = project(K, view_i["R"], view_i["c"], X)
        us.append(u)
        vs.append(v)
        zs.append(z)
    out = np.full(h * w, np.nan)
    if not us:
        return out.reshape(h, w)
    u = np.rint(np.concatenate(us)).astype(np.int64)
    v = np.rint(np.concatenate(vs)).astype(np.int64)
    z = np.concatenate(zs)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    pix = v[ok] * w + u[ok]
    z = z[ok]
    if not len(z):
        return out.reshape(h, w)
    o = np.lexsort((z, pix))
    pix, z = pix[o], z[o]
    start = np.r_[True, pix[1:] != pix[:-1]]
    gstart = np.maximum.accumulate(np.where(start, np.arange(len(pix)), 0))
    keep = z <= z[gstart] * (1 + tol)
    pix, z = pix[keep], z[keep]
    start = np.flatnonzero(np.r_[True, pix[1:] != pix[:-1]])
    cnt = np.diff(np.r_[start, len(pix)])
    med = 0.5 * (z[start + (cnt - 1) // 2] + z[start + cnt // 2])
    out[pix[start]] = med
    return out.reshape(h, w)


def _pair_disagreement(K, vi, vj, xi, xj, k):
    X = vi["c"] + (k * xi)[..., None] * pixel_rays_world(K, vi["R"], *xi.shape)
    X = X.reshape(-1, 3)
    X = X[np.isfinite(X).all(1)]
    u, v, z = project(K, vj["R"], vj["c"], X)
    h, w = xj.shape
    ui, vi_ = np.rint(u).astype(int), np.rint(v).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi_ >= 0) & (vi_ < h) & (z > 0)
    # a factor that throws the frame's points out of the neighbour's view is not a candidate: a
    # handful of stray points would otherwise give a spuriously small disagreement (tiny k
    # puts all points next to camera i)
    if ok.sum() < MIN_OVERLAP * len(X):
        return np.array([])
    dj = k * xj[vi_[ok], ui[ok]]
    zz = z[ok]
    fin = np.isfinite(dj) & (dj > 0)
    if fin.sum() < 20:
        return np.array([])
    # remove the pair's relative scale (per-frame scale errors), keep the misregistration that a
    # wrong common factor causes through parallax
    rho = np.median(zz[fin] / dj[fin])
    return np.abs(zz[fin] - rho * dj[fin]) / zz[fin]


def _apply_common(mode, params, g):
    p = np.array(params, float)
    if mode == "metric":
        return p * g
    return p / g


def global_gauge(
    K, views, xs, pairs, lo=1e-3, hi=1e4, n=80, refine=1.15
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scan the common factor k; returns (k, grid, cost)."""

    def cost(k):
        rs = [_pair_disagreement(K, views[i], views[j], xs[i], xs[j], k) for i, j in pairs]
        rs = [r for r in rs if r.size]
        if len(rs) < 0.5 * len(pairs):
            return np.inf
        return float(np.median(np.concatenate(rs)))

    grid = np.geomspace(lo, hi, n)
    c = np.array([cost(k) for k in grid])
    k0 = grid[int(np.nanargmin(c))]
    fine = np.geomspace(k0 / refine, k0 * refine, 41)
    cf = np.array([cost(k) for k in fine])
    k1 = fine[int(np.nanargmin(cf))]
    finer = np.geomspace(k1 / 1.012, k1 * 1.012, 25)
    cff = np.array([cost(k) for k in finer])
    # parabola through the minimum and its neighbours (in log k) for sub-grid precision
    j = int(np.nanargmin(cff))
    kb = finer[j]
    if 0 < j < len(finer) - 1 and np.all(np.isfinite(cff[j - 1 : j + 2])):
        x = np.log(finer[j - 1 : j + 2])
        a2, a1, _ = np.polyfit(x, cff[j - 1 : j + 2], 2)
        if a2 > 0:
            kb = float(np.exp(np.clip(-a1 / (2 * a2), x[0], x[2])))
    return float(kb), np.r_[grid, fine, finer], np.r_[c, cf, cff]


def _pair_log_ratio(K, vi, vj, di, dj, step=2):
    """log median of (frame i's depth seen from j) / (frame j's depth there), and the point count.

    Pixels of i are subsampled by ``step``. Returns (nan, 0) when fewer than ``MIN_OVERLAP`` of
    i's points land in j's view."""
    d = di[::step, ::step]
    rays = pixel_rays_world(K, vi["R"], *di.shape)[::step, ::step]
    X = (vi["c"] + d[..., None] * rays).reshape(-1, 3)
    X = X[np.isfinite(X).all(1)]
    if not len(X):
        return np.nan, 0
    u, v, z = project(K, vj["R"], vj["c"], X)
    h, w = dj.shape
    ui, vi_ = np.rint(u).astype(int), np.rint(v).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi_ >= 0) & (vi_ < h) & (z > 0)
    if ok.sum() < max(MIN_OVERLAP * len(X), 20):
        return np.nan, 0
    zj = dj[vi_[ok], ui[ok]]
    zz = z[ok]
    fin = np.isfinite(zj) & (zj > 0)
    if fin.sum() < 20:
        return np.nan, 0
    return float(np.median(np.log(zz[fin] / zj[fin]))), int(fin.sum())


def relative_scales(K, views, depths, nb, iters=20, step=2, prior=None, prior_sd=PRIOR_SD):
    """Per-frame scale corrections g (median 1) that make overlapping frames agree.

    Each overlapping pair (i, j), in both directions, gives l_ij = log median(z_i->j / z_j); with
    frame i too large by a_i, l_ij = log a_i - log a_j. The log a are a robust (Huber IRLS)
    weighted least-squares solve on the pair graph with the median pinned to 0; frames outside the
    largest connected component keep g = 1. Frames are then divided by g. This is a global solve,
    so it does not inherit the bias of a median target drawn from neighbours with large errors.

    Pairs are weighted by sqrt(points) / sigma_pair, with sigma_pair the residual MAD of a first
    pass, and down-weighted beyond 1.5 sigma_pair (cut at 9). ``prior`` (F,) is each frame's
    current log scale relative to the gauge; a Gaussian prior of sd ``prior_sd`` on the corrected
    log scale keeps frames near the gauge, which only matters for frames with few or inconsistent
    pairs (a weakly linked frame would otherwise follow one bad pair)."""
    F = len(views)
    rows = []
    for i in range(F):
        for j in nb[i]:
            lij, n = _pair_log_ratio(K, views[i], views[j], depths[i], depths[j], step)
            if n:
                rows.append((i, j, lij, n))
    if not rows:
        return np.ones(F), 0
    I = np.array([r[0] for r in rows])
    J = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows])
    W0 = np.sqrt(np.array([r[3] for r in rows], float))
    W0 = W0 / W0.mean()
    # largest connected component of the pair graph
    lab = np.arange(F)
    for _ in range(F):
        a = np.minimum(lab[I], lab[J])
        new = lab.copy()
        np.minimum.at(new, I, a)
        np.minimum.at(new, J, a)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    comp = lab == np.bincount(lab[np.unique(np.r_[I, J])]).argmax()
    keep = comp[I] & comp[J]
    I, J, L, W0 = I[keep], J[keep], L[keep], W0[keep]
    idx = np.flatnonzero(comp)
    nF = len(idx)
    pos = -np.ones(F, int)
    pos[idx] = np.arange(nF)
    nP = len(L)
    A = np.zeros((nP + nF + 1, nF))
    A[np.arange(nP), pos[I]] = 1.0
    A[np.arange(nP), pos[J]] = -1.0
    A[nP + np.arange(nF), np.arange(nF)] = 1.0
    A[-1] = 1.0  # mean of log a = 0 (re-centred to the median below)
    p0 = np.zeros(F) if prior is None else np.asarray(prior, float)
    # log a_i = p0_i would bring frame i back to the gauge
    b = np.r_[L, p0[idx], 0.0]
    sig = 1.0
    hw = np.ones(nP)
    for it in range(iters):
        w = np.r_[W0 * hw / sig, np.full(nF, 1.0 / prior_sd), 10.0 * nF / sig]
        x = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        r = A[:nP] @ x - L
        if it == 0:
            sig = max(1.4826 * float(np.median(np.abs(r))), 1e-3)
        hw = _huber_w(r / sig, 1.5)
    loga = np.zeros(F)
    loga[idx] = x - np.median(x)
    return np.exp(loga), nP


def align_block(
    K: np.ndarray,
    views: Sequence[Mapping],
    raw: Sequence[np.ndarray],
    mode: str = "metric",
    n_alt: int = 0,
    huber: float = 0.05,
    radius: float | None = None,
    fit_beta: bool = True,
    tol: float = 0.02,
    fit_offset: bool = False,
    n_graph: int = 6,
) -> AlignResult:
    """Label-free alignment of ``raw`` (one array per view, model output on the view grid)."""
    F = len(views)
    xs = [np.asarray(r, float) for r in raw]
    xs_depthlike = [np.where(x > 0, 1.0 / x, np.nan) for x in xs] if mode == "inverse" else xs
    C = np.array([v["c"][:2] for v in views])
    # several frames may share a camera centre (different headings or tilts); spacing is taken
    # over distinct centres
    U = np.unique(np.round(C, 3), axis=0)
    if len(U) < 2:
        raise ValueError("label-free alignment needs at least two distinct camera centres")
    Du = np.linalg.norm(U[:, None] - U[None], axis=-1)
    np.fill_diagonal(Du, np.inf)
    spacing = float(np.median(Du.min(1)))
    if radius is None:
        radius = 3.5 * spacing
    nb = neighbours(views, radius)
    # gauge pairs need a baseline (parallax; co-located frames carry none) and a shared view:
    # oblique frames of different headings hardly overlap, so neighbours of the same heading
    # come first
    yaw = np.array([float(v.get("yaw", 0.0)) % 360.0 for v in views])
    pairs = []
    for i in range(F):
        cand = [j for j in nb[i] if np.linalg.norm(C[j] - C[i]) > 0.25 * spacing]
        same = [j for j in cand if abs(yaw[j] - yaw[i]) < 1.0]
        cand = same if same else cand
        cand.sort(key=lambda j: np.linalg.norm(C[j] - C[i]))
        pairs += [(i, j) for j in cand[:4]]
    if not pairs:
        raise ValueError("no frame pairs with a baseline inside the neighbour radius")
    k, _, _ = global_gauge(K, views, xs_depthlike, pairs)
    params = np.tile([k, 0.0] if mode == "metric" else [1.0 / k, 0.0], (F, 1))
    beta = 0.0
    depths = [aligned_depth(mode, xs[i], params[i], beta) for i in range(F)]
    # global relative scales from the pair graph (a few rounds, since moving a frame's scale also
    # moves its points through parallax), before any frame is fitted to a rendered target
    hist = []
    logs = np.zeros(F)  # each frame's log scale relative to the gauge
    for _ in range(n_graph):
        g, _ = relative_scales(K, views, depths, nb, prior=logs)
        logs = logs - np.log(g)
        params = np.array([_apply_common(mode, params[i], 1.0 / g[i]) for i in range(F)])
        depths = [aligned_depth(mode, xs[i], params[i], beta) for i in range(F)]
        hist.append(float(np.max(np.abs(g - 1))))
        if hist[-1] < 1e-3:
            break
    eff_gauge = _effective_scale(mode, xs, params)
    for _ in range(n_alt):
        targets = [render_into(K, views[i], views, depths, nb[i], tol) for i in range(F)]
        new = np.array(
            [
                _fit_frame(mode, xs[i], targets[i], beta, params[i], huber, fit_offset=fit_offset)
                for i in range(F)
            ]
        )
        if fit_beta:
            bases = [_base(mode, xs[i], new[i]) for i in range(F)]
            beta = _fit_beta(bases, targets, huber, beta)
        eff_old = _effective_scale(mode, xs, params)
        params = new
        depths = [aligned_depth(mode, xs[i], params[i], beta) for i in range(F)]
        # keep the common factor at the parallax gauge: frame-to-frame fits cannot see it, so
        # without this the whole block drifts along the scale-offset valley
        q = float(np.median(_effective_scale(mode, xs, params) / eff_gauge))
        params = _apply_common(mode, params, 1.0 / q)
        depths = [aligned_depth(mode, xs[i], params[i], beta) for i in range(F)]
        change = float(np.nanmax(np.abs(_effective_scale(mode, xs, params) / eff_old - 1)))
        hist.append(change)
    return AlignResult(mode, k, params, beta, depths, hist, nb)


def _effective_scale(mode, xs, params):
    """Median aligned depth per frame (a mode-independent size of the alignment)."""
    return np.array([np.nanmedian(_base(mode, x, p)) for x, p in zip(xs, params)])


def oracle_align(
    mode: str,
    raw: Sequence[np.ndarray],
    exact: Sequence[np.ndarray],
    huber: float = 0.05,
    fit_offset: bool = False,
):
    """Per-frame fit against exact depth (the cost-of-no-labels reference; never used in the map)."""
    out = []
    for x, d in zip(raw, exact):
        p0 = (
            np.array([np.nanmedian(d / x), 0.0])
            if mode == "metric"
            else np.array([np.nanmedian(1 / (d * x)), 0.0])
        )
        out.append(_fit_frame(mode, np.asarray(x, float), d, 0.0, p0, huber, fit_offset=fit_offset))
    return np.array(out)
