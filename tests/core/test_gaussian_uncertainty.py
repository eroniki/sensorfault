"""Tests for Gaussian surfels with a rendered 9x9 uncertainty and the depth-only view gate."""

import numpy as np
import pytest

from sensorfault.eval import gaussian_uncertainty as gu

F = 50.0
H = W = 12
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
KI = np.linalg.inv(K)
R_S = np.column_stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # t1, t2, n = -z


def _scene(spacing=0.1, n=9):
    xs = (np.arange(n) - (n - 1) / 2) * spacing
    mus = np.array([[x, y, 2.0] for y in xs for x in xs])
    Rs = np.repeat(R_S[None], len(mus), 0)
    return mus, Rs


def _view(mus, Rs, c, scale=0.1, alpha=0.6, keep=None, ntop=6):
    """Per (pixel, surfel) contributions of a camera at c looking along +z."""
    pix, gid, wh, J, dbar = [], [], [], [], []
    Dbar = np.zeros(H * W)
    Wsum = np.zeros(H * W)
    for v in range(H):
        for u in range(W):
            r = KI @ np.array([u, v, 1.0])
            ag, ds, Js = [], [], []
            for k in range(len(mus)):
                d, x = gu.ray_surfel_depth(mus[k], Rs[k], c, np.eye(3), r)
                g = np.exp(-0.5 * np.sum((x - mus[k]) ** 2) / scale**2)
                ag.append(alpha * g)
                ds.append(d)
                Js.append(gu.depth_jacobian(mus[k], Rs[k], c, np.eye(3), r))
            ag = np.array(ag)
            order = np.argsort(-ag)[:ntop]
            _, w_hat, Wp = gu.composite_weights(ag, order, keep)
            p = v * W + u
            Wsum[p] = Wp
            for k in order:
                if w_hat[k] > 0:
                    pix.append(p)
                    gid.append(k)
                    wh.append(w_hat[k])
                    J.append(Js[k])
                    dbar.append(ds[k])
                    Dbar[p] += w_hat[k] * ds[k]
    return dict(
        pix=np.array(pix),
        gid=np.array(gid),
        w=np.array(wh),
        J=np.array(J),
        dbar=np.array(dbar),
        D=Dbar,
        W=Wsum,
    )


CENTRES = [np.array([x, y, 0.0]) for x in (-0.15, 0.0, 0.15) for y in (-0.1, 0.1)]
MUS, RS = _scene()
VIEWS = [_view(MUS, RS, c) for c in CENTRES]
VIEWS1 = [_view(MUS, RS, c, ntop=1) for c in CENTRES]  # one surfel per pixel
PRIOR = np.eye(9) * 1e-6


def _measure(view, theta, a, b, sd, rng, extra=None):
    lin = np.zeros(H * W)
    np.add.at(lin, view["pix"], view["w"] * np.einsum("mi,mi->m", view["J"], theta[view["gid"]]))
    y = a * (view["D"] + lin) + b + sd * rng.standard_normal(H * W)
    if extra is not None:
        y += extra
    return y


def _info(view, y, a, b, sd, defl=1.0):
    e = y - (a * view["D"] + b)
    return gu.view_information(
        view["pix"], view["gid"], view["w"], view["J"], e, np.full(H * W, sd**2), len(MUS), a, defl
    )


def _render(view, mean, cov, keep=None):
    D = np.zeros(H * W)
    V = np.zeros(H * W)
    for p in range(H * W):
        m = view["pix"] == p
        if keep is not None:
            m &= keep[view["gid"]]
        if not m.any():
            D[p] = V[p] = np.nan
            continue
        w = view["w"][m] / view["w"][m].sum()
        g = view["gid"][m]
        D[p], V[p] = gu.render_depth_and_variance(view["dbar"][m], w, view["J"][m], mean[g], cov[g])
    return D, V


def test_depth_jacobian_finite_difference_and_scale_columns():
    rng = np.random.default_rng(0)
    R = gu.so3_exp(np.array([0.2, -0.1, 0.3])) @ R_S
    mu = np.array([0.05, -0.02, 2.0])
    s = np.array([0.1, 0.1, 0.01])
    c = np.array([0.1, 0.05, 0.0])
    Rc = gu.so3_exp(np.array([0.02, 0.03, -0.01]))
    r = KI @ np.array([5.0, 7.0, 1.0])
    J = gu.depth_jacobian(mu, R, c, Rc, r)
    d0, _ = gu.ray_surfel_depth(mu, R, c, Rc, r)
    eps = 1e-6
    for i in range(9):
        th = np.zeros(9)
        th[i] = eps
        m2, R2, _ = gu.perturb_surfel(mu, R, s, th)
        d1, _ = gu.ray_surfel_depth(m2, R2, c, Rc, r)
        assert (d1 - d0) / eps == pytest.approx(J[i], abs=1e-4)
    assert np.allclose(J[6:], 0.0)
    assert rng is not None


def test_subblock_map_matches_normal_perturbation():
    R = gu.so3_exp(np.array([0.3, 0.2, -0.4]))
    phi = np.array([1e-6, -2e-6, 3e-6])
    dn = (R @ gu.so3_exp(phi))[:, 2] - R[:, 2]
    G = gu.subblock_map(R)
    th = np.concatenate([np.zeros(3), phi, np.zeros(3)])
    tau = G @ th
    assert R[:, 0] @ dn == pytest.approx(tau[1], abs=1e-10)
    assert R[:, 1] @ dn == pytest.approx(tau[2], abs=1e-10)


def test_leave_view_out_is_exact_subtraction():
    rng = np.random.default_rng(1)
    theta = np.zeros((len(MUS), 9))
    infos = [_info(v, _measure(v, theta, 1.0, 0.0, 0.01, rng), 1.0, 0.0, 0.01) for v in VIEWS]
    L, eta = gu.fuse(PRIOR, infos)
    for j in (0, 3):
        Lm, em = gu.leave_view_out(L, eta, *infos[j])
        Lr, er = gu.fuse(PRIOR, [x for i, x in enumerate(infos) if i != j])
        assert np.allclose(Lm, Lr) and np.allclose(em, er)


def test_posterior_covariance_matches_monte_carlo():
    rng = np.random.default_rng(2)
    theta = np.zeros((len(MUS), 9))
    theta[:, :3] = rng.normal(0, 0.005, (len(MUS), 3))
    k = 40  # centre surfel
    ests = []
    for _ in range(300):
        infos = [_info(v, _measure(v, theta, 1.0, 0.0, 0.01, rng), 1.0, 0.0, 0.01) for v in VIEWS]
        L, eta = gu.fuse(PRIOR, infos)
        mean, cov = gu.posterior(L, eta)
        ests.append(gu.subblock_map(RS[k]) @ mean[k])
    ests = np.array(ests)
    C = gu.offset_tilt_cov(cov[k], RS[k])
    assert np.var(ests[:, 0]) == pytest.approx(C[0, 0], rel=0.2)
    assert np.mean(ests[:, 0]) == pytest.approx(
        RS[k][:, 2] @ theta[k, :3], abs=3 * np.sqrt(C[0, 0])
    )


def _gate_z(rng, extra=None, defl=1.0, corr=None, j=2, reps=1, views=None):
    views = VIEWS1 if views is None else views
    zs = []
    theta = np.zeros((len(MUS), 9))
    for _ in range(reps):
        ys = []
        for i, v in enumerate(views):
            add = None
            if corr is not None:
                add = corr * np.sin(np.arange(H * W) / 7.0 + rng.uniform(0, 6))
            if i == j and extra is not None:
                add = extra if add is None else add + extra
            ys.append(_measure(v, theta, 1.0, 0.0, 0.01, rng, add))
        infos = [_info(v, y, 1.0, 0.0, 0.01, defl) for v, y in zip(views, ys)]
        L, eta = gu.fuse(PRIOR, infos)
        Lm, em = gu.leave_view_out(L, eta, *infos[j])
        mean, cov = gu.posterior(Lm, em)
        keep = gu.supported(Lm, RS, PRIOR, 1.0)
        D, V = _render(views[j], mean, cov, keep)
        ok = ~np.isnan(D)
        zs.append(
            [gu.whitened_residual(ys[j][p], 1.0, 0.0, D[p], V[p], 1e-4) for p in np.where(ok)[0]]
        )
    return np.concatenate(zs)


def test_whitened_residual_is_calibrated_when_blocks_are_exact():
    # one surfel per pixel: the per-surfel blocks are the exact posterior, so z ~ N(0, 1)
    z = _gate_z(np.random.default_rng(3), reps=20)
    assert np.std(z) == pytest.approx(1.0, rel=0.06)
    assert abs(np.mean(z)) < 0.05


def test_overlapping_footprints_make_independent_blocks_overconfident():
    # several surfels per pixel: cross-surfel correlations are dropped (stated limitation)
    z = _gate_z(np.random.default_rng(3), reps=20, views=VIEWS)
    assert np.std(z) > 1.03


def test_regional_fault_is_detected():
    rng = np.random.default_rng(4)
    mask = np.zeros(H * W)
    mask[: H * W // 2] = 1.0
    z = _gate_z(rng, extra=0.05 * mask, views=VIEWS)  # 5 sigma on half the view
    assert np.mean(z**2) > 5.0


def test_tested_view_spawns_nothing_and_unseen_is_untestable():
    ag = np.array([0.9, 0.6])  # floater in front (k=0), surface behind (k=1)
    order = np.array([0, 1])
    _, w_all, _ = gu.composite_weights(ag, order)
    _, w_kept, W_kept = gu.composite_weights(ag, order, keep=np.array([False, True]))
    assert w_all[0] > w_all[1]
    assert w_kept[0] == 0 and w_kept[1] == 1.0 and W_kept == pytest.approx(0.6)
    L_other = np.stack([PRIOR, PRIOR + np.diag([0, 0, 50.0] + [0] * 6)])
    keep = gu.supported(L_other, RS[:2], PRIOR, 1.0)
    assert keep.tolist() == [False, True]
    _, _, W_none = gu.composite_weights(ag, order, keep=np.array([False, False]))
    assert W_none == 0.0


def test_independent_blocks_are_overconfident_under_correlated_noise():
    z = _gate_z(np.random.default_rng(5), corr=0.02, reps=10)
    z_def = _gate_z(np.random.default_rng(5), corr=0.02, defl=0.05, reps=10)
    assert np.std(z) > 1.3
    assert np.std(z_def) < np.std(z)
