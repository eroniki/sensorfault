"""Pure-numpy reference ``SurfelBackend`` (tiny scenes, CPU).

Dense (surfel x pixel) renderer of planar Gaussian surfels with the same alpha compositing as 2DGS
(front-to-back, normalised depth ``D = sum w d / sum w``) and the closed-form Jacobians of
``sensorfault.eval.gaussian_uncertainty``. Cost is O(K * H * W): use it for tests and tiny scenes only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from sensorfault.eval import gaussian_uncertainty as gu
from sensorfault.gs.interfaces import (
    GEO_DIM,
    THETA_DIM,
    CavityRender,
    SurfelState,
    ViewInfo,
)
from sensorfault.types import DepthFrame

W_MIN = 0.05  # accumulated opacity below which a pixel counts as unsupported
THIN_SCALE = 1e-3  # fixed thin axis of a 2DGS surfel


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _orthonormal_frame(n: np.ndarray) -> np.ndarray:
    """R = [t1 t2 n] right-handed with third column ``n``."""
    a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(a, n)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n], axis=1)


def _frame_rays(frame: DepthFrame) -> np.ndarray:
    """(H*W, 3) camera-frame pixel rays with z = 1, row-major pixel order."""
    h, w = frame.depth.shape
    v, u = np.mgrid[0:h, 0:w]
    pix = np.stack([u.ravel(), v.ravel(), np.ones(h * w)], axis=0).astype(float)
    return (np.linalg.inv(frame.K) @ pix).T


def _valid_depth(frame: DepthFrame) -> np.ndarray:
    d = np.asarray(frame.depth, dtype=float).ravel()
    return np.isfinite(d) & (d > 0)


def init_surfels(
    frames: Sequence[DepthFrame],
    stride: int = 4,
    voxel: float | None = None,
    alpha0: float = 0.5,
) -> SurfelState:
    """One surfel per ``stride``-th valid pixel, back-projected, facing its camera; surfels from
    different frames closer than ``voxel`` (default: pixel footprint * stride) are merged."""
    mus, normals, scales, seen = [], [], [], set()
    for f in frames:
        h, w = f.depth.shape
        rays = _frame_rays(f)
        Rc, c = f.T_wc[:3, :3], f.T_wc[:3, 3]
        valid = _valid_depth(f).reshape(h, w)
        fx = float(f.K[0, 0])
        for v in range(stride // 2, h, stride):
            for u in range(stride // 2, w, stride):
                if not valid[v, u]:
                    continue
                d = float(f.depth[v, u])
                ray_w = Rc @ rays[v * w + u]
                x = c + d * ray_w
                vox = voxel if voxel is not None else stride * d / fx
                key = tuple(np.floor(x / vox).astype(int))
                if key in seen:
                    continue
                seen.add(key)
                mus.append(x)
                normals.append(-ray_w / np.linalg.norm(ray_w))
                scales.append(0.7 * stride * d / fx)
    if not mus:
        raise ValueError("no valid depth pixels to initialise surfels from")
    K = len(mus)
    R = np.stack([_orthonormal_frame(n) for n in normals])
    s = np.asarray(scales)
    log_s = np.stack([np.log(s), np.log(s), np.full(K, np.log(THIN_SCALE))], axis=1)
    logit = np.full(K, np.log(alpha0 / (1.0 - alpha0)))
    return SurfelState(mu=np.asarray(mus), R=R, log_s=log_s, logit_alpha=logit)


def _composite_chunk(
    mu, R, log_s, logit, c: np.ndarray, Rc: np.ndarray, rays: np.ndarray, keep: np.ndarray
) -> dict:
    """Render a chunk of m pixels from K surfels. All outputs are in the original surfel order.

    d (K,m) hit depth, jsub (K,m,3) d depth / d (nu, tau1, tau2), w_hat (K,m), D (m,) (NaN where the
    accumulated opacity is zero), W (m,), dD (K,m) dD / dlogit(alpha_k)."""
    K, m = len(mu), len(rays)
    n, t1, t2 = R[:, :, 2], R[:, :, 0], R[:, :, 1]
    w = rays @ Rc.T  # (m, 3) world ray directions (z-depth parameterisation)
    den = n @ w.T  # (K, m)
    num = np.einsum("ki,ki->k", mu - c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = num[:, None] / den
    hit = c + d[:, :, None] * w[None]
    dx = hit - mu[:, None, :]
    u = np.einsum("kmi,ki->km", dx, t1)
    v = np.einsum("kmi,ki->km", dx, t2)
    s = np.exp(log_s)
    with np.errstate(over="ignore", invalid="ignore"):
        g = np.exp(-0.5 * ((u / s[:, :1]) ** 2 + (v / s[:, 1:2]) ** 2))
    ok = np.isfinite(d) & (d > 1e-6) & (np.abs(den) > 1e-9) & keep[:, None]
    g = np.where(ok, g, 0.0)
    d = np.where(ok, d, 0.0)
    alpha = 1.0 / (1.0 + np.exp(-logit))
    a = np.minimum(alpha[:, None] * g, 0.999)
    key = np.where(g > 0, d, np.inf)
    order = np.argsort(key, axis=0, kind="stable")
    take = lambda x: np.take_along_axis(x, order, axis=0)
    a_s, d_s, g_s, al_s = take(a), take(d), take(g), take(np.broadcast_to(alpha[:, None], (K, m)))
    T = np.concatenate([np.ones((1, m)), np.cumprod(1.0 - a_s, axis=0)[:-1]], axis=0)
    w_s = a_s * T
    W = w_s.sum(0)
    wd = w_s * d_s
    live = W > 1e-12
    Ws = np.where(live, W, 1.0)
    D = np.where(live, wd.sum(0) / Ws, np.nan)
    suf_wd = np.cumsum(wd[::-1], axis=0)[::-1] - wd  # strictly behind
    suf_w = np.cumsum(w_s[::-1], axis=0)[::-1] - w_s
    dA = g_s * (T * d_s - suf_wd / (1.0 - a_s))
    dW = g_s * (T - suf_w / (1.0 - a_s))
    dD_s = al_s * (1.0 - al_s) * (dA - np.where(live, D, 0.0) * dW) / Ws
    w_hat_s = w_s / Ws
    dD_s = np.where(live, dD_s, 0.0)
    inv = np.argsort(order, axis=0)
    w_hat = np.take_along_axis(w_hat_s, inv, axis=0)
    dD = np.take_along_axis(dD_s, inv, axis=0)
    safe_den = np.where(np.abs(den) > 1e-9, den, 1.0)
    jsub = np.stack([np.ones_like(d), -u, -v], axis=-1) / safe_den[:, :, None]
    return {"d": d, "jsub": jsub, "w_hat": w_hat, "D": D, "W": W, "dD": dD, "g": g}


def _chunks(P: int, chunk: int):
    for i in range(0, P, chunk):
        yield slice(i, min(i + chunk, P))


def _geometry(state: SurfelState, shift: np.ndarray | None = None):
    mu, R, log_s = state.mu.copy(), state.R.copy(), state.log_s.copy()
    if shift is not None:
        mu = mu + shift[:, :3]
        R = np.stack([R[k] @ gu.so3_exp(shift[k, 3:6]) for k in range(len(R))])
        log_s = log_s + shift[:, 6:9]
    return mu, R, log_s


def _apply_normal_tilt(mu, R, delta):
    """delta (K, 3) = (nu, tau1, tau2): mu += nu n, R <- R exp([-tau2, tau1, 0])."""
    mu = mu + delta[:, :1] * R[:, :, 2]
    R = np.stack(
        [R[k] @ gu.so3_exp(np.array([-delta[k, 2], delta[k, 1], 0.0])) for k in range(len(R))]
    )
    return mu, R


# --------------------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------------------


class NumpyReferenceBackend:
    """Numpy ``SurfelBackend``. Parameters: ``stride`` (init surfel spacing in pixels), ``n_warmup``
    (iterations with opacity trainable), ``n_iter`` (further geometry-only iterations), ``chunk``
    (pixels per rendering chunk)."""

    def __init__(
        self,
        stride: int = 4,
        n_warmup: int = 15,
        n_iter: int = 25,
        opacity_lr: float = 0.2,
        chunk: int = 512,
    ):
        self.stride, self.n_warmup, self.n_iter = stride, n_warmup, n_iter
        self.opacity_lr, self.chunk = opacity_lr, chunk

    @staticmethod
    def available() -> tuple[bool, str]:
        return True, ""

    # -- rendering -----------------------------------------------------------------------
    def _render_frame(self, mu, R, log_s, logit, frame, keep, need_pairs=False):
        rays = _frame_rays(frame)
        Rc, c = frame.T_wc[:3, :3], frame.T_wc[:3, 3]
        out = []
        for sl in _chunks(len(rays), self.chunk):
            out.append((sl, _composite_chunk(mu, R, log_s, logit, c, Rc, rays[sl], keep)))
        return out

    def _stack(self, mu, R, log_s, logit, act, need_jac):
        """Weighted residuals (and Jacobian wrt (nu, tau1, tau2) per surfel) over all active views."""
        K = len(mu)
        keep = np.ones(K, bool)
        res, rows, dDs = [], [], []
        for frame, wt in act:
            y = np.asarray(frame.depth, dtype=float).ravel()
            ok_depth = _valid_depth(frame)
            sw = np.sqrt(wt)
            for sl, comp in self._render_frame(mu, R, log_s, logit, frame, keep):
                m = ok_depth[sl] & (comp["W"] >= W_MIN) & np.isfinite(comp["D"])
                if not m.any():
                    continue
                res.append(sw * (y[sl][m] - comp["D"][m]))
                if need_jac:
                    J = comp["w_hat"][:, m, None] * comp["jsub"][:, m]  # (K, mm, 3)
                    rows.append(sw * np.transpose(J, (1, 0, 2)).reshape(m.sum(), 3 * K))
                    dDs.append(sw * comp["dD"][:, m])
        r = np.concatenate(res) if res else np.zeros(0)
        if not need_jac:
            return r, None, None
        J = np.concatenate(rows) if rows else np.zeros((0, 3 * K))
        dD = np.concatenate(dDs, axis=1) if dDs else np.zeros((K, 0))
        return r, J, dD

    # -- SurfelBackend -------------------------------------------------------------------
    def fit(
        self,
        frames: Sequence[DepthFrame],
        weights: Mapping[str, float],
        init: SurfelState | None = None,
    ) -> SurfelState:
        act = [(f, float(weights.get(f.sensor_id, 1.0))) for f in frames]
        act = [(f, wt) for f, wt in act if wt > 0]
        if not act:
            raise ValueError("all views have weight 0")
        st = init if init is not None else init_surfels([f for f, _ in act], self.stride)
        mu, R, log_s, logit = st.mu.copy(), st.R.copy(), st.log_s.copy(), st.logit_alpha.copy()
        K = len(mu)
        lam = 1e-2
        m_ad, v_ad = np.zeros(K), np.zeros(K)
        loss = np.inf
        for it in range(self.n_warmup + self.n_iter):
            r, J, dD = self._stack(mu, R, log_s, logit, act, True)
            if len(r) == 0:
                break
            if it < self.n_warmup:  # opacity warm-up (Adam on logit alpha); frozen afterwards
                grad = -2.0 * dD @ r
                m_ad = 0.9 * m_ad + 0.1 * grad
                v_ad = 0.99 * v_ad + 0.01 * grad**2
                step = (
                    self.opacity_lr
                    * (m_ad / (1 - 0.9 ** (it + 1)))
                    / (np.sqrt(v_ad / (1 - 0.99 ** (it + 1))) + 1e-12)
                )
                logit = np.clip(logit - step, -6.0, 6.0)
                r, J, _ = self._stack(mu, R, log_s, logit, act, True)
            base = float(r @ r)
            H, gvec = J.T @ J, J.T @ r
            for _ in range(8):  # Levenberg-Marquardt on the normal offset and tilt
                delta = np.linalg.solve(H + lam * np.eye(3 * K), gvec).reshape(K, 3)
                mu2, R2 = _apply_normal_tilt(mu, R, delta)
                r2, _, _ = self._stack(mu2, R2, log_s, logit, act, False)
                if len(r2) == len(r) and float(r2 @ r2) < base:
                    mu, R, base, lam = mu2, R2, float(r2 @ r2), max(lam * 0.5, 1e-6)
                    break
                lam *= 4.0
            loss = base
        meta = {
            "backend": "numpy_reference",
            "warmup_iters": self.n_warmup,
            "n_iter": self.n_iter,
            "final_loss": loss,
            "opacity_frozen": True,
            "sensor_ids": [f.sensor_id for f, _ in act],
        }
        return SurfelState(mu=mu, R=R, log_s=log_s, logit_alpha=logit, meta=meta)

    def view_information(self, state: SurfelState, frame: DepthFrame, sigma: float) -> ViewInfo:
        K = len(state.mu)
        mu, R, log_s = _geometry(state)
        y = np.asarray(frame.depth, dtype=float).ravel()
        ok_depth = _valid_depth(frame)
        keep = np.ones(K, bool)
        Lam, eta, n_pix = np.zeros((K, THETA_DIM, THETA_DIM)), np.zeros((K, THETA_DIM)), 0
        for sl, comp in self._render_frame(mu, R, log_s, state.logit_alpha, frame, keep):
            m = ok_depth[sl] & (comp["W"] >= W_MIN) & np.isfinite(comp["D"])
            if not m.any():
                continue
            kk, pp = np.nonzero((comp["w_hat"] > 1e-6) & m[None, :])
            if len(kk) == 0:
                continue
            Js = comp["jsub"][kk, pp]  # (n, 3)
            J = np.zeros((len(kk), THETA_DIM))
            for i in range(len(kk)):
                J[i] = Js[i] @ gu.subblock_map(R[kk[i]])
            e = np.where(m, y[sl] - np.where(m, comp["D"], 0.0), 0.0)
            Li, ei = gu.view_information(
                pix=pp,
                gid=kk,
                w_hat=comp["w_hat"][kk, pp],
                J=J,
                e=e,
                sigma2=np.full(len(e), sigma**2),
                n_gauss=K,
                j_opacity=comp["dD"][kk, pp],
            )
            Lam += Li
            eta += ei
            n_pix += int(m.sum())
        return ViewInfo(Lambda=Lam, eta=eta, n_pixels=n_pix)

    def render_cavity(
        self,
        state: SurfelState,
        frame: DepthFrame,
        mean_shift: np.ndarray,
        cov: np.ndarray,
        keep: np.ndarray,
    ) -> CavityRender:
        h, w = frame.depth.shape
        mu, R, log_s = _geometry(state, mean_shift)
        keep = np.asarray(keep, bool)
        depth = np.full(h * w, np.nan)
        var = np.full(h * w, np.nan)
        supported = np.zeros(h * w, bool)
        for sl, comp in self._render_frame(mu, R, log_s, state.logit_alpha, frame, keep):
            Js = comp["jsub"]  # (K, m, 3)
            K = len(mu)
            G = np.stack([gu.subblock_map(R[k])[:, :GEO_DIM] for k in range(K)])  # (K, 3, 9)
            Jgeo = np.einsum("kmi,kij->kmj", Js, G)  # (K, m, 9)
            v = np.einsum("km,kmi,kij,kmj->m", comp["w_hat"] ** 2, Jgeo, cov, Jgeo)
            sup = (comp["W"] >= W_MIN) & np.isfinite(comp["D"])
            supported[sl] = sup
            depth[sl] = np.where(sup, comp["D"], np.nan)
            var[sl] = np.where(sup, v, np.nan)
        valid = supported & _valid_depth(frame)
        return CavityRender(
            depth=depth.reshape(h, w), var=var.reshape(h, w), valid=valid.reshape(h, w)
        )
