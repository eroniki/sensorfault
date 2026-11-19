"""CPU tests of the numpy reference SurfelBackend."""

from __future__ import annotations

import numpy as np

from sensorfault.eval import gaussian_uncertainty as gu
from sensorfault.gs import reference as ref
from sensorfault.gs.interfaces import SurfelBackend, SurfelState
from sensorfault.types import DepthFrame

H = W = 16
KMAT = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]])
PLANE_N = np.array([0.0, 0.2, -1.0]) / np.linalg.norm([0.0, 0.2, -1.0])  # faces the cameras
PLANE_P = np.array([0.0, 0.0, 2.0])


def _frame(sid: str, x: float, noise: float = 0.0, seed: int = 0) -> DepthFrame:
    T = np.eye(4)
    T[0, 3] = x
    rays = ref._frame_rays(DepthFrame(sid, 0, np.zeros((H, W), np.float32), KMAT, T))
    d = (PLANE_N @ (PLANE_P - T[:3, 3])) / (rays @ PLANE_N)  # z-depth since rays have z = 1
    d = d + np.random.default_rng(seed).normal(0, noise, d.shape)
    return DepthFrame(sid, 0, d.reshape(H, W).astype(np.float32), KMAT, T)


def _plane_depth(frame: DepthFrame) -> np.ndarray:
    return _frame(frame.sensor_id, frame.T_wc[0, 3]).depth


def _backend() -> ref.NumpyReferenceBackend:
    return ref.NumpyReferenceBackend(stride=4, n_warmup=10, n_iter=20)


def test_satisfies_protocol_and_available():
    b = _backend()
    assert b.available() == (True, "")
    for name in ("fit", "view_information", "render_cavity"):
        assert callable(getattr(b, name))
    _: SurfelBackend = b  # structural typing


def test_fit_recovers_planar_patch_from_perturbed_init():
    frames = [_frame("a", -0.1), _frame("b", 0.1)]
    b = _backend()
    st0 = ref.init_surfels(frames, stride=4)
    bad = SurfelState(
        mu=st0.mu + 0.08 * st0.R[:, :, 2],
        R=st0.R,
        log_s=st0.log_s,
        logit_alpha=st0.logit_alpha,
    )
    st = b.fit(frames, {}, init=bad)
    assert st.meta["opacity_frozen"]
    # surfels end on the plane
    off = np.abs((st.mu - PLANE_P) @ PLANE_N)
    assert off.max() < 5e-3
    cav = b.render_cavity(
        st,
        frames[0],
        np.zeros((len(st.mu), 9)),
        np.zeros((len(st.mu), 9, 9)),
        np.ones(len(st.mu), bool),
    )
    m = cav.valid
    assert m.sum() > 50
    assert np.nanmax(np.abs(cav.depth[m] - frames[0].depth[m])) < 1e-2


def test_fit_from_default_init_with_noise():
    frames = [_frame("a", -0.1, 0.01, 1), _frame("b", 0.1, 0.01, 2)]
    b = _backend()
    st = b.fit(frames, {})
    K = len(st.mu)
    cav = b.render_cavity(st, frames[0], np.zeros((K, 9)), np.zeros((K, 9, 9)), np.ones(K, bool))
    assert cav.valid.sum() > 50
    # rendered surface is closer to the truth than the 1 cm per-pixel noise
    err = np.abs(cav.depth[cav.valid] - _plane_depth(frames[0])[cav.valid])
    assert err.mean() < 1e-2


def test_weight_zero_removes_view():
    fa = _frame("a", -0.1)
    fb = _frame("b", 0.1)
    fb_bad = DepthFrame("b", 0, fb.depth + 0.5, KMAT, fb.T_wc)  # grossly biased view
    b = _backend()
    st0 = b.fit([fa, fb_bad], {"b": 0.0})
    assert st0.meta["sensor_ids"] == ["a"]
    assert np.abs((st0.mu - PLANE_P) @ PLANE_N).max() < 5e-3
    # with weight 1 the biased view pulls the surface away
    st1 = b.fit([fa, fb_bad], {"b": 1.0})
    assert np.abs((st1.mu - PLANE_P) @ PLANE_N).mean() > 2e-2


def test_view_information_matches_gaussian_uncertainty_loop():
    frames = [_frame("a", -0.1), _frame("b", 0.1)]
    b = _backend()
    st = b.fit(frames, {})
    f = frames[1]
    f = DepthFrame("b", 0, f.depth + 0.01, KMAT, f.T_wc)  # nonzero residuals
    sigma = 0.05
    info = b.view_information(st, f, sigma)
    K = len(st.mu)
    assert info.Lambda.shape == (K, 10, 10) and info.eta.shape == (K, 10)

    rays = ref._frame_rays(f)
    Rc, c = f.T_wc[:3, :3], f.T_wc[:3, 3]
    alpha = 1 / (1 + np.exp(-st.logit_alpha))
    s = np.exp(st.log_s)
    L = np.zeros((K, 10, 10))
    eta = np.zeros((K, 10))
    for p in range(H * W):
        d = np.zeros(K)
        g = np.zeros(K)
        for k in range(K):
            dk, x = gu.ray_surfel_depth(st.mu[k], st.R[k], c, Rc, rays[p])
            dx = x - st.mu[k]
            if dk <= 0:
                continue
            d[k] = dk
            g[k] = np.exp(
                -0.5 * ((st.R[k][:, 0] @ dx / s[k, 0]) ** 2 + (st.R[k][:, 1] @ dx / s[k, 1]) ** 2)
            )
        if not (g > 0).any():
            continue
        order = np.argsort(np.where(g > 0, d, np.inf))
        D, w_hat, dD = gu.composite_depth_opacity_jacobian(alpha, g, d, order)
        W_acc = gu.composite_weights(alpha * g, order)[2]
        if W_acc < ref.W_MIN:
            continue
        J = np.array([gu.depth_jacobian(st.mu[k], st.R[k], c, Rc, rays[p]) for k in range(K)])
        sel = w_hat > 1e-6
        kk = np.nonzero(sel)[0]
        Li, ei = gu.view_information(
            pix=np.zeros(len(kk), int),
            gid=kk,
            w_hat=w_hat[kk],
            J=J[kk],
            e=np.array([float(f.depth.ravel()[p]) - D]),
            sigma2=np.array([sigma**2]),
            n_gauss=K,
            j_opacity=dD[kk],
            lumped=True,
        )
        L += Li
        eta += ei
    np.testing.assert_allclose(info.Lambda, L, rtol=1e-6, atol=1e-6 * np.abs(L).max())
    np.testing.assert_allclose(info.eta, eta, rtol=1e-6, atol=1e-6 * np.abs(eta).max())
    assert np.abs(info.Lambda).max() > 0
    # symmetric PSD blocks
    assert np.allclose(info.Lambda, info.Lambda.transpose(0, 2, 1))
    assert np.linalg.eigvalsh(info.Lambda).min() > -1e-6 * np.abs(info.Lambda).max()


def test_cavity_render_shift_and_variance():
    frame = _frame("a", 0.0)
    b = _backend()
    st = b.fit([frame], {})
    K = len(st.mu)
    # shifting every surfel by nu along its own normal moves the plane toward the camera by ~nu / cos
    shift = np.zeros((K, 9))
    shift[:, :3] = 0.05 * st.R[:, :, 2]
    cov = np.broadcast_to(np.eye(9) * 1e-4, (K, 9, 9)).copy()
    cav = b.render_cavity(st, frame, shift, cov, np.ones(K, bool))
    base = b.render_cavity(st, frame, np.zeros((K, 9)), cov, np.ones(K, bool))
    m = cav.valid & base.valid
    assert m.sum() > 50
    dz = base.depth[m] - cav.depth[m]
    assert np.all((dz > 0.04) & (dz < 0.06))  # nu / |n . ray|, rays within ~25 deg
    assert np.all(cav.var[m] > 0)
    # surfels dropped by `keep` stop supporting pixels
    none = b.render_cavity(st, frame, np.zeros((K, 9)), cov, np.zeros(K, bool))
    assert not none.valid.any() and np.isnan(none.depth).all()


def test_leave_view_out_cavity_matches_other_views_surface():
    fa, fb = _frame("a", -0.1), _frame("b", 0.1)
    b = _backend()
    st_all = b.fit([fa, fb], {})
    st_wo = b.fit([fa, fb], {"b": 0.0})
    K_all = len(st_all.mu)
    Z = (np.zeros((K_all, 9)), np.broadcast_to(np.eye(9) * 1e-6, (K_all, 9, 9)).copy())
    # information-form leave-out: subtracting view b's information keeps surfels that a supports
    ia, ib = b.view_information(st_all, fa, 0.02), b.view_information(st_all, fb, 0.02)
    keep = gu.opacity_supported(ia.Lambda + ib.Lambda, ib.Lambda, rho=0.5)
    cav = b.render_cavity(st_all, fb, Z[0], Z[1], keep)
    m = cav.valid
    assert m.sum() > 20
    assert np.nanmax(np.abs(cav.depth[m] - _plane_depth(fb).ravel().reshape(H, W)[m])) < 1e-2
    # and agrees with the surface fitted directly without view b, rendered into b
    cav2 = b.render_cavity(
        st_wo,
        fb,
        np.zeros((len(st_wo.mu), 9)),
        np.zeros((len(st_wo.mu), 9, 9)),
        np.ones(len(st_wo.mu), bool),
    )
    both = m & cav2.valid
    assert both.sum() > 20
    assert np.nanmax(np.abs(cav.depth[both] - cav2.depth[both])) < 1e-2


def test_gsplat_backend_import_safe_and_reports_unavailable():
    from sensorfault.gs.fit import GsplatBackend

    ok, why = GsplatBackend.available()
    assert isinstance(ok, bool) and (ok or why)


import pytest  # noqa: E402


@pytest.mark.gpu
def test_gsplat_fit_recovers_plane():
    from sensorfault.gs.fit import GsplatBackend

    ok, why = GsplatBackend.available()
    if not ok:
        pytest.skip(why)
    frames = [_frame("a", -0.1), _frame("b", 0.1)]
    b = GsplatBackend(n_warmup=100, n_steps=300)
    st = b.fit(frames, {})
    K = len(st.mu)
    cav = b.render_cavity(st, frames[0], np.zeros((K, 9)), np.zeros((K, 9, 9)), np.ones(K, bool))
    err = np.abs(cav.depth[cav.valid] - _plane_depth(frames[0])[cav.valid])
    assert err.mean() < 2e-2


def _bias_scene(n=6, hw=32, f=40.0, noise=0.005, bias=0.05, bias_view=3):
    rng = np.random.default_rng(0)
    K = np.array([[f, 0, hw / 2], [0, f, hw / 2], [0, 0, 1.0]])
    nrm = np.array([0.15, -0.1, 1.0])
    nrm /= np.linalg.norm(nrm)
    frames = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = 0.12 * (i - (n - 1) / 2)
        T[1, 3] = 0.03 * ((i % 2) - 0.5)
        u, v = np.meshgrid(np.arange(hw) + 0.5, np.arange(hw) + 0.5)
        rays = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u)], -1)
        d = ((np.array([0, 0, 2.0]) - T[:3, 3]) @ nrm) / (rays @ nrm)
        d = d + noise * rng.standard_normal(d.shape)
        if i == bias_view:
            d[: hw // 2, : hw // 2] += bias
        frames.append(DepthFrame(f"v{i}", i, d.astype(np.float32), K, T))
    return frames


@pytest.mark.slow
def test_newton_cavity_is_closer_to_a_refit_than_no_step():
    """Regression for the overlap overshoot: with w^2 blocks the leave-view-out step is several times
    too long and worse than no step; the lumped (w) blocks are a majoriser and must beat no step."""
    from sensorfault.gs.gate import ViewGateTest

    frames = _bias_scene()
    be = ref.NumpyReferenceBackend(stride=4)
    w = {f.sensor_id: 1.0 for f in frames}
    st = be.fit(frames, w)
    st.meta["weights"] = dict(w)
    st2 = be.fit(frames, {**w, "v3": 0.0}, init=st)
    K = len(st.mu)
    z, cov, keep = np.zeros((K, 9)), np.zeros((K, 9, 9)), np.ones(K, bool)
    exact = be.render_cavity(st2, frames[3], z, cov, keep)
    nostep = be.render_cavity(st, frames[3], z, cov, keep)
    gate = ViewGateTest(be, frames, sigma=0.01)
    cav = gate._pass(st, gate._fused(st), [], 1.0, want_cache_only=True)["v3"]
    m = exact.valid & nostep.valid & cav.valid
    rms = lambda a: float(np.sqrt(np.mean(a[m] ** 2)))
    assert rms(cav.depth - exact.depth) < 0.7 * rms(nostep.depth - exact.depth)


def test_full_jacobian_matches_finite_differences():
    """dD/d(nu, tau1, tau2) with the footprint term equals central differences of the renderer."""
    from sensorfault.gs.reference import _apply_normal_tilt, _composite_chunk, _frame_rays

    frames = [_frame(f"v{i}", x) for i, x in enumerate((-0.2, 0.0, 0.2))]
    st = ref.init_surfels(frames, stride=4)
    rng = np.random.default_rng(3)
    st = SurfelState(
        mu=st.mu + 0.01 * rng.standard_normal(st.mu.shape),
        R=st.R,
        log_s=st.log_s,
        logit_alpha=rng.uniform(-1.0, 1.0, len(st.mu)),
    )
    f = frames[1]
    rays, c, Rc = _frame_rays(f), f.T_wc[:3, 3], f.T_wc[:3, :3]
    K = len(st.mu)
    keep = np.ones(K, bool)
    base = _composite_chunk(st.mu, st.R, st.log_s, st.logit_alpha, c, Rc, rays, keep)
    ok = np.isfinite(base["D"]) & (base["W"] > 0.05)
    h = 1e-6
    for k in rng.choice(K, 5, replace=False):
        for i in range(3):
            dl = np.zeros((K, 3))
            dl[k, i] = h
            p = _composite_chunk(
                *_apply_normal_tilt(st.mu, st.R, dl), st.log_s, st.logit_alpha, c, Rc, rays, keep
            )
            m = _apply_normal_tilt(st.mu, st.R, -dl)
            q = _composite_chunk(*m, st.log_s, st.logit_alpha, c, Rc, rays, keep)
            fd = ((p["D"] - q["D"]) / (2 * h))[ok]
            jac = (base["w_hat"][k] * base["jsub"][k, :, i] + base["jfoot"][k, :, i])[ok]
            assert np.allclose(jac, fd, rtol=1e-4, atol=1e-6 * max(1.0, np.abs(fd).max()))
