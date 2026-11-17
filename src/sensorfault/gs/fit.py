"""gsplat (2DGS) ``SurfelBackend``: depth-supervised Adam fit on GPU.

``torch`` and ``gsplat`` are imported lazily inside methods, so importing this module is always safe;
``GsplatBackend.available()`` reports whether the fit can run. The fit uses
``depth_mode='ray_plane'`` (default): a torch port of the reference compositor, the same per-pixel ray-plane
depth ``D = sum w d / sum w`` as ``render_cavity`` and the closed-form Jacobians. ``depth_mode='gsplat'``
uses ``gsplat.rasterization_2dgs`` RGB+D, which blends per-surfel centre depth and so does not see surfel
tilt (found on the TITAN smoke test; kept for comparison only). ``view_information`` and ``render_cavity`` use
the closed-form alpha-weighted Jacobian of ``gaussian_uncertainty`` through the chunked dense
implementation of ``reference.py`` (memory O(K * chunk)); a rasteriser-side per-(pixel, surfel) weight
dump would replace it for large K.

The ray-plane torch path is verified against the numpy reference only by tests/gs/test_torch_port.py
(needs torch) and has not been run on the GPU yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from sensorfault.gs.interfaces import CavityRender, SurfelState, ViewInfo
from sensorfault.gs.reference import NumpyReferenceBackend, _frame_rays, init_surfels
from sensorfault.types import DepthFrame


def _rot_to_quat_wxyz(R):
    """(K, 3, 3) torch rotation matrices -> (K, 4) unit quaternions (w, x, y, z)."""
    import torch

    K = R.shape[0]
    m = R
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(K, 4, dtype=R.dtype, device=R.device)
    c0 = tr > 0
    s = torch.sqrt(torch.clamp(tr + 1.0, min=1e-12)) * 2
    q0 = torch.stack(
        [
            0.25 * s,
            (m[:, 2, 1] - m[:, 1, 2]) / s,
            (m[:, 0, 2] - m[:, 2, 0]) / s,
            (m[:, 1, 0] - m[:, 0, 1]) / s,
        ],
        1,
    )
    s1 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-12)) * 2
    q1 = torch.stack(
        [
            (m[:, 2, 1] - m[:, 1, 2]) / s1,
            0.25 * s1,
            (m[:, 0, 1] + m[:, 1, 0]) / s1,
            (m[:, 0, 2] + m[:, 2, 0]) / s1,
        ],
        1,
    )
    s2 = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-12)) * 2
    q2 = torch.stack(
        [
            (m[:, 0, 2] - m[:, 2, 0]) / s2,
            (m[:, 0, 1] + m[:, 1, 0]) / s2,
            0.25 * s2,
            (m[:, 1, 2] + m[:, 2, 1]) / s2,
        ],
        1,
    )
    s3 = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-12)) * 2
    q3 = torch.stack(
        [
            (m[:, 1, 0] - m[:, 0, 1]) / s3,
            (m[:, 0, 2] + m[:, 2, 0]) / s3,
            (m[:, 1, 2] + m[:, 2, 1]) / s3,
            0.25 * s3,
        ],
        1,
    )
    d0, d1, d2 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    c1 = (~c0) & (d0 >= d1) & (d0 >= d2)
    c2 = (~c0) & (~c1) & (d1 >= d2)
    q = torch.where(c0[:, None], q0, q3)
    q = torch.where(c1[:, None], q1, q)
    q = torch.where(c2[:, None], q2, q)
    return q / q.norm(dim=1, keepdim=True)


def _skew(w):
    import torch

    z = torch.zeros_like(w[:, 0])
    return torch.stack(
        [z, -w[:, 2], w[:, 1], w[:, 2], z, -w[:, 0], -w[:, 1], w[:, 0], z], dim=1
    ).reshape(-1, 3, 3)


def _ray_plane_depth(mu, R, log_s, logit, c, Rc, rays):
    """Torch port of ``reference._composite_chunk`` (differentiable): alpha-composited, per-pixel
    ray-plane hit depth ``D = sum w d / sum w`` and accumulated opacity ``W``.

    mu (K, 3), R (K, 3, 3) columns [t1 t2 n], log_s (K, 3), logit (K,), camera centre c (3,),
    camera-to-world rotation Rc (3, 3), camera-frame rays (m, 3) with z = 1. Returns D (m,), W (m,)."""
    import torch

    n, t1, t2 = R[:, :, 2], R[:, :, 0], R[:, :, 1]
    w = rays @ Rc.T
    den = n @ w.T
    num = ((mu - c) * n).sum(1)
    safe = torch.where(den.abs() > 1e-9, den, torch.ones_like(den))
    d = num[:, None] / safe
    dx = c + d[:, :, None] * w[None] - mu[:, None, :]
    u = (dx * t1[:, None, :]).sum(-1)
    v = (dx * t2[:, None, :]).sum(-1)
    sc = torch.exp(log_s)
    g = torch.exp(-0.5 * ((u / sc[:, :1]) ** 2 + (v / sc[:, 1:2]) ** 2))
    ok = (den.abs() > 1e-9) & (d > 1e-6)
    g = torch.where(ok, g, torch.zeros_like(g))
    d = torch.where(ok, d, torch.zeros_like(d))
    a = (torch.sigmoid(logit)[:, None] * g).clamp(max=0.999)
    order = torch.argsort(torch.where(g > 0, d, torch.full_like(d, float("inf"))), dim=0)
    a_s, d_s = torch.gather(a, 0, order), torch.gather(d, 0, order)
    trans = torch.cat([torch.ones_like(a_s[:1]), torch.cumprod(1.0 - a_s, 0)[:-1]], 0)
    w_s = a_s * trans
    W = w_s.sum(0)
    return (w_s * d_s).sum(0) / W.clamp(min=1e-12), W


class GsplatBackend:
    """gsplat 2DGS ``SurfelBackend``.

    Surfel scales are frozen by default (lr_scale=0): the gate's information model has no information on
    them, and learned scales widen the footprints, which breaks its independent-block approximation
    (the cavity step overshoots by roughly the overlap factor).

    n_warmup steps train geometry + opacity, then opacity is frozen for n_steps more geometry steps
    (Adam; separate learning rates). ``sigma_fit`` is unused: a view's weight scales its squared
    depth loss (mean over its valid pixels)."""

    def __init__(
        self,
        n_warmup: int = 200,
        n_steps: int = 600,
        lr_mu: float = 2e-3,
        lr_rot: float = 5e-3,
        lr_scale: float = 0.0,
        lr_opacity: float = 5e-2,
        stride: int = 4,
        device: str = "cuda",
        chunk: int = 2048,
        depth_mode: str = "ray_plane",
        px_stride: int = 2,
    ):
        if depth_mode not in ("ray_plane", "gsplat"):
            raise ValueError(depth_mode)
        # ray_plane: supervise the same per-pixel ray-plane depth that render_cavity and the Jacobians
        # use (gsplat's RGB+D blends per-surfel centre depth, which never sees surfel tilt);
        # gsplat: the stock rasteriser depth (kept for comparison)
        self.depth_mode, self.px_stride = depth_mode, px_stride
        self.n_warmup, self.n_steps = n_warmup, n_steps
        self.lrs = (lr_mu, lr_rot, lr_scale, lr_opacity)
        self.stride, self.device = stride, device
        self._info = NumpyReferenceBackend(stride=stride, chunk=chunk)

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import torch
        except Exception as e:
            return False, f"torch not importable: {e}"
        try:
            import gsplat  # noqa: F401
        except Exception as e:
            return False, f"gsplat not importable: {e}"
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        return True, ""

    # -- fit -------------------------------------------------------------------------------
    def fit(
        self,
        frames: Sequence[DepthFrame],
        weights: Mapping[str, float],
        init: SurfelState | None = None,
    ) -> SurfelState:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"GsplatBackend unavailable: {why}")
        import torch
        from gsplat import rasterization_2dgs

        act = [(f, float(weights.get(f.sensor_id, 1.0))) for f in frames]
        act = [(f, w) for f, w in act if w > 0]
        if not act:
            raise ValueError("all views have weight 0")
        st = init if init is not None else init_surfels([f for f, _ in act], self.stride)
        dev, dt = self.device, torch.float32
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        mu = T(st.mu).clone().requires_grad_(True)
        phi = torch.zeros(len(st.mu), 3, dtype=dt, device=dev, requires_grad=True)
        log_s = T(st.log_s).clone().requires_grad_(True)
        logit = T(st.logit_alpha).clone().requires_grad_(True)
        R0 = T(st.R)
        lr_mu, lr_rot, lr_scale, lr_op = self.lrs
        opt = torch.optim.Adam(
            [
                {"params": [mu], "lr": lr_mu},
                {"params": [phi], "lr": lr_rot},
                {"params": [logit], "lr": lr_op},
            ]
        )
        if (
            lr_scale > 0
        ):  # scales are not observed by depth to first order (design v3.1): frozen by default
            opt.add_param_group({"params": [log_s], "lr": lr_scale})
        log_s0 = log_s.detach().clone()
        log_s.requires_grad_(lr_scale > 0)
        views = []
        for f, w in act:
            h, wd = f.depth.shape
            d = np.asarray(f.depth, dtype=np.float32)
            valid = np.isfinite(d) & (d > 0)
            sel = np.zeros(h * wd, bool)
            sel[:: self.px_stride] = True
            sel &= valid.reshape(-1)
            rays_np = _frame_rays(f)
            views.append(
                (
                    w,
                    T(np.linalg.inv(f.T_wc))[None],
                    T(f.K)[None],
                    h,
                    wd,
                    T(np.where(valid, d, 0.0)),
                    torch.as_tensor(valid, device=dev),
                    (
                        torch.as_tensor(np.nonzero(sel)[0], device=dev),
                        T(rays_np[sel]),
                        T(f.T_wc[:3, 3]),
                        T(f.T_wc[:3, :3]),
                    ),
                )
            )
        loss_v = float("nan")
        for step in range(self.n_warmup + self.n_steps):
            if step == self.n_warmup:  # freeze opacity
                logit.requires_grad_(False)
                logit.grad = None
            R = R0 @ torch.linalg.matrix_exp(_skew(phi))
            quats = _rot_to_quat_wxyz(R)
            loss = mu.new_zeros(())
            for w, view, Kmat, h, wd, y, valid, rp in views:
                if self.depth_mode == "ray_plane":
                    idx, rays_t, c_t, Rc_t = rp
                    D, Wacc = _ray_plane_depth(mu, R, log_s, logit, c_t, Rc_t, rays_t)
                    m = Wacc > 0.05
                    if m.any():
                        loss = loss + w * ((D - y.reshape(-1)[idx])[m] ** 2).mean()
                    continue
                out = rasterization_2dgs(
                    means=mu,
                    quats=quats,
                    scales=torch.exp(log_s),
                    opacities=torch.sigmoid(logit),
                    colors=torch.ones(1, len(mu), 3, dtype=dt, device=dev),  # [C, N, 3]
                    viewmats=view,
                    Ks=Kmat,
                    width=wd,
                    height=h,
                    render_mode="RGB+D",
                )
                colors, alphas = out[0][0], out[1][0, ..., 0]
                depth = colors[..., -1] / alphas.clamp(min=1e-6)
                m = valid & (alphas > 0.05)
                if m.any():
                    loss = loss + w * ((depth - y)[m] ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_v = float(loss.detach())
        with torch.no_grad():
            R = R0 @ torch.linalg.matrix_exp(_skew(phi))
        meta = {
            "backend": "gsplat_2dgs",
            "warmup_iters": self.n_warmup,
            "n_iter": self.n_steps,
            "final_loss": loss_v,
            "opacity_frozen": True,
            "log_s_rms_change": float((log_s.detach() - log_s0).pow(2).mean().sqrt()),
            "sensor_ids": [f.sensor_id for f, _ in act],
        }
        return SurfelState(
            mu=mu.detach().cpu().numpy().astype(np.float64),
            R=R.detach().cpu().numpy().astype(np.float64),
            log_s=log_s.detach().cpu().numpy().astype(np.float64),
            logit_alpha=logit.detach().cpu().numpy().astype(np.float64),
            meta=meta,
        )

    # -- information and cavity (closed form, shared with the reference backend) ---------------
    def view_information(self, state: SurfelState, frame: DepthFrame, sigma: float) -> ViewInfo:
        return self._info.view_information(state, frame, sigma)

    def render_cavity(
        self,
        state: SurfelState,
        frame: DepthFrame,
        mean_shift: np.ndarray,
        cov: np.ndarray,
        keep: np.ndarray,
    ) -> CavityRender:
        return self._info.render_cavity(state, frame, mean_shift, cov, keep)
