"""ViewGateTest against a small fake SurfelBackend: a plane scene tiled by planar surfels."""

from __future__ import annotations

import numpy as np
import pytest

from sensorfault.eval.detection_theory import whitened_energy
from sensorfault.eval.gaussian_uncertainty import depth_jacobian, perturb_surfel
from sensorfault.gs.gate import ViewGateTest
from sensorfault.gs.interfaces import CavityRender, SurfelState, ViewInfo
from sensorfault.types import DepthFrame

Z0 = 2.0
TILE = 1.0
GRID = 3  # 3 x 3 tiles over [-1.5, 1.5]^2
HW = 12
FOCAL = 12.0
SIGMA = 0.01
OFFSETS = [(-0.3, -0.2), (0.3, -0.2), (0.0, 0.3), (-0.2, 0.25), (0.25, 0.2), (0.0, 0.0)]
IDS = [f"v{i}" for i in range(len(OFFSETS))]


def _rays() -> np.ndarray:
    v, u = np.mgrid[0:HW, 0:HW]
    return np.stack([(u - HW / 2 + 0.5) / FOCAL, (v - HW / 2 + 0.5) / FOCAL, np.ones(u.shape)], -1)


RAYS = _rays()


def _tile_of(c: np.ndarray) -> np.ndarray:
    """Index of the surfel tile hit by each pixel ray (plane z = Z0), -1 outside the grid."""
    x = c[0] + Z0 * RAYS[..., 0]
    y = c[1] + Z0 * RAYS[..., 1]
    ix = np.floor((x + GRID * TILE / 2) / TILE).astype(int)
    iy = np.floor((y + GRID * TILE / 2) / TILE).astype(int)
    ok = (ix >= 0) & (ix < GRID) & (iy >= 0) & (iy < GRID)
    return np.where(ok, iy * GRID + ix, -1)


def _ray_depth(mu, R, c, Rc, r):
    n = R[:, 2]
    w = Rc @ r
    return float(n @ (mu - c)) / float(n @ w)


class FakeBackend:
    """Plane scene; surfel k owns tile k. Exactly linear-Gaussian up to the tiny tilt nonlinearity."""

    def __init__(self, frames):
        self.frames = {f.sensor_id: f for f in frames}

    def initial_state(self) -> SurfelState:
        mu = np.array(
            [
                [(i + 0.5) * TILE - GRID * TILE / 2, (j + 0.5) * TILE - GRID * TILE / 2, Z0]
                for j in range(GRID)
                for i in range(GRID)
            ]
        )
        n = len(mu)
        return SurfelState(mu, np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), np.zeros(n))

    def _rows(self, state, frame):
        c, Rc = frame.T_wc[:3, 3], frame.T_wc[:3, :3]
        tile = _tile_of(c)
        D = np.full((HW, HW), np.nan)
        J = np.zeros((HW, HW, 10))
        for y in range(HW):
            for x in range(HW):
                k = tile[y, x]
                if k >= 0:
                    D[y, x] = _ray_depth(state.mu[k], state.R[k], c, Rc, RAYS[y, x])
                    J[y, x] = depth_jacobian(state.mu[k], state.R[k], c, Rc, RAYS[y, x])
        return tile, D, J

    def view_information(self, state, frame, sigma):
        tile, D, J = self._rows(state, frame)
        K = len(state.mu)
        Lam, eta = np.zeros((K, 10, 10)), np.zeros((K, 10))
        ok = (tile >= 0) & np.isfinite(frame.depth)
        for y, x in zip(*np.nonzero(ok), strict=True):
            k = tile[y, x]
            Lam[k] += np.outer(J[y, x], J[y, x]) / sigma**2
            eta[k] += J[y, x] * (frame.depth[y, x] - D[y, x]) / sigma**2
        return ViewInfo(Lam, eta, int(ok.sum()))

    def fit(self, frames, weights, init=None):
        state = init or self.initial_state()
        for _ in range(2):
            K = len(state.mu)
            L = np.tile(1e-6 * np.eye(9), (K, 1, 1))
            g = np.zeros((K, 9))
            for f in frames:
                w = weights.get(f.sensor_id, 1.0)
                if w > 0:
                    vi = self.view_information(state, f, SIGMA)
                    L += w * vi.Lambda[:, :9, :9]
                    g += w * vi.eta[:, :9]
            th = np.einsum("kij,kj->ki", np.linalg.inv(L), g)
            mu, R = state.mu.copy(), state.R.copy()
            for k in range(K):
                mu[k], R[k], _ = perturb_surfel(mu[k], R[k], np.ones(3), np.r_[th[k], 0.0])
            state = SurfelState(mu, R, state.log_s, state.logit_alpha)
        return state

    def render_cavity(self, state, frame, mean_shift, cov, keep):
        c, Rc = frame.T_wc[:3, 3], frame.T_wc[:3, :3]
        tile = _tile_of(c)
        depth = np.full((HW, HW), np.nan)
        var = np.full((HW, HW), np.nan)
        valid = np.zeros((HW, HW), bool)
        for y, x in zip(*np.nonzero(tile >= 0), strict=True):
            k = tile[y, x]
            if not keep[k]:
                continue
            # a backend renders only the observable part G theta = (nu, tau1, tau2) of the shift:
            # the tangential part of d_mu slides the surfel along its own plane
            th = np.r_[mean_shift[k], 0.0]
            n = state.R[k][:, 2]
            th[:3] = n * (n @ th[:3])
            mu, R, _ = perturb_surfel(state.mu[k], state.R[k], np.ones(3), th)
            depth[y, x] = _ray_depth(mu, R, c, Rc, RAYS[y, x])
            Jk = depth_jacobian(state.mu[k], state.R[k], c, Rc, RAYS[y, x])[:9]  # at the MAP
            var[y, x] = float(Jk @ cov[k] @ Jk)
            valid[y, x] = np.isfinite(frame.depth[y, x])
        return CavityRender(depth, var, valid)


def make_frames(seed, noise_scale=1.0, bias_view=None, bias=0.0):
    rng = np.random.default_rng(seed)
    frames = []
    for vid, (ox, oy) in zip(IDS, OFFSETS, strict=True):
        d = Z0 + noise_scale * SIGMA * rng.standard_normal((HW, HW))
        if vid == bias_view:
            d[:6, :6] += bias
        T = np.eye(4)
        T[:2, 3] = (ox, oy)
        K = np.array([[FOCAL, 0, HW / 2], [0, FOCAL, HW / 2], [0, 0, 1.0]])
        frames.append(DepthFrame(vid, 0, d.astype(np.float64), K, T))
    return frames


def run(seed, params=None, views=None, **kw):
    frames = make_frames(seed, **kw.pop("scene", {}))
    be = FakeBackend(frames)
    state = be.fit(frames, {})
    gate = ViewGateTest(be, frames, sigma=SIGMA, params=params, **kw)
    return gate(state, views), gate


def test_returns_pvalues_and_log_pvalues():
    (p, lp), _ = run(0)
    assert set(p) == set(lp) == set(IDS)
    for v in IDS:
        assert 0.0 <= p[v] <= 1.0
        assert lp[v] <= 0.0
        assert np.isclose(np.log(max(p[v], 1e-300)), lp[v]) or p[v] == 0.0


def test_clean_views_not_flagged_at_nominal_rate():
    alpha, n_seed = 0.05, 30
    ps = []
    for seed in range(n_seed):
        (p, _), _ = run(seed)
        ps.extend(p.values())
    ps = np.array(ps)
    assert np.mean(ps < alpha) < 0.15  # nominal 5%, allow slack for correlated pixels
    assert np.mean(ps < 0.001) < 0.03


def test_biased_view_gets_tiny_p_and_log_p_ranks():
    (p, lp), _ = run(1, scene={"bias_view": "v2", "bias": 0.06})
    assert p["v2"] < 1e-8
    assert lp["v2"] < np.log(1e-8)
    assert lp["v2"] == min(lp.values())
    others = [lp[v] for v in IDS if v != "v2"]
    assert max(others) > lp["v2"] + 10  # the biased view is the most significant by far


def test_log_p_stays_finite_when_p_underflows():
    (p, lp), _ = run(1, scene={"bias_view": "v2", "bias": 5.0})
    assert np.isfinite(lp["v2"]) and lp["v2"] < -700
    assert p["v2"] == 0.0


def test_joint_vs_subtraction_differ():
    sc = {"bias_view": "v2", "bias": 0.06}
    (_, lp_sub), _ = run(2, scene=sc)
    (_, lp_joint), _ = run(2, params={"leave_view_out": "none_joint"}, scene=sc)
    # the joint fit lets the biased view pull the surface towards itself: far less significant
    assert lp_joint["v2"] > lp_sub["v2"] + 5.0


def test_temperature_absorbs_global_noise_scale():
    alpha, n_seed = 0.05, 12
    p_t, p_fixed, g2 = [], [], []
    for seed in range(n_seed):
        (p, _), gate = run(seed, scene={"noise_scale": 3.0})
        p_t.extend(p.values())
        g2.extend(d["gamma2"] for d in gate.last.values())
        (p, _), _ = run(
            seed, params={"noise": "dev_calibrated_level_only"}, scene={"noise_scale": 3.0}
        )
        p_fixed.extend(p.values())
    assert np.mean(np.array(p_fixed) < alpha) > 0.5  # without temperature everything fires
    assert np.mean(np.array(p_t) < alpha) < 0.2
    assert 4.0 < np.median(g2) < 16.0  # gamma^2 ~ 9 for a 3x noise scale


def test_temperature_does_not_hide_a_biased_view():
    (p, _), _ = run(3, scene={"noise_scale": 3.0, "bias_view": "v4", "bias": 0.3})
    assert p["v4"] < 1e-4


def test_unsupported_surfels_dropped_unless_spawn_kept():
    (_, _), g1 = run(0, views=["v0"])
    (_, _), g2 = run(0, params={"spawn": "keep_view_j_only_surfels"}, views=["v0"])
    assert g1.last["v0"]["dof"] < g2.last["v0"]["dof"]


def test_shared_bias_fitted_and_switch_off():
    _, g = run(0, views=["v0"])
    assert "beta" in g.last["v0"]
    _, g0 = run(0, params={"shared_bias": "none"}, views=["v0"])
    assert g0.last["v0"]["beta"] == 0.0


def test_ablation_switches_run():
    for params in (
        {"uncertainty": "constant_sigma"},
        {"information_deflation": 0.5},
        {"information_deflation": "n_eff_over_n"},
        {"overconfidence": "jackknife_inflation"},
        {"statistic": "energy"},
        {"statistic": "region_scan"},
    ):
        (p, lp), _ = run(0, params=params, views=["v1"])
        assert 0.0 <= p["v1"] <= 1.0 and lp["v1"] <= 0.0, params


def test_jackknife_inflation_is_less_significant():
    sc = {"bias_view": "v2", "bias": 0.03}
    # isolate the inflation: fixed omega and the energy-tied scan scale (the data-driven scales
    # re-normalise the other views and can mask the effect of a single ablation)
    base = {"scan_noise": "energy"}
    (_, lp0), _ = run(4, params=base, scene=sc, views=["v2"], heterogeneity_omega=0.0)
    (_, lp1), _ = run(
        4,
        params={**base, "overconfidence": "jackknife_inflation"},
        scene=sc,
        views=["v2"],
        heterogeneity_omega=0.0,
    )
    assert lp1["v2"] >= lp0["v2"] - 1e-9


def test_unimplemented_and_bad_params():
    be = FakeBackend(make_frames(0))
    for u in ("fisherrf_style_diagonal_fisher", "bayes_rays_style_perturbation_field"):
        with pytest.raises(NotImplementedError, match=u):
            ViewGateTest(be, make_frames(0), params={"uncertainty": u})
    with pytest.raises(ValueError):
        ViewGateTest(be, make_frames(0), params={"leave_view_out": "bogus"})
    with pytest.raises(ValueError):
        ViewGateTest(be, make_frames(0), params={"nonsense": 1})
    # fit/outer-loop params of the config are accepted and ignored
    ViewGateTest(
        be, make_frames(0), params={"opacity_warmup": "fold_wise_K5", "leave_block_out_B": 1}
    )


def test_inactive_view_is_not_subtracted():
    frames = make_frames(0)
    be = FakeBackend(frames)
    weights = dict.fromkeys(IDS, 1.0) | {"v5": 0.0}
    state = be.fit(frames, weights)
    state.meta["weights"] = weights
    gate = ViewGateTest(be, frames, sigma=SIGMA)
    p, lp = gate(state, ["v5"])  # readmission test of a dropped view
    assert 0.0 <= p["v5"] <= 1.0


def test_energy_matches_whitened_energy_helper():
    (_, _), gate = run(0, params={"shared_bias": "none"}, views=["v1"])
    frames = make_frames(0)
    be = FakeBackend(frames)
    state = be.fit(frames, {})
    gate2 = ViewGateTest(be, frames, sigma=SIGMA, params={"shared_bias": "none"})
    infos = gate2._informations(state, dict.fromkeys(IDS, 1.0), 1.0)
    K = len(state.mu)
    L_data = sum((vi.Lambda for vi in infos.values()), np.zeros((K, 10, 10)))
    cav = gate2._cavity(state, "v1", infos, L_data + gate2.prior, L_data, None)
    D, s2, d, _ = gate2._pixels("v1", cav)
    T, dof, _ = gate2._stats("v1", cav, 0.0)
    A = np.stack([D, np.ones_like(D)], 1)
    T_ref, dof_ref = whitened_energy(d - D, np.diag(s2), A)
    assert np.isclose(T, T_ref, rtol=1e-6, atol=1e-8)
    assert dof == dof_ref or dof == dof_ref + 1  # plane scene: D nearly constant, rank 1 vs 2
    assert gate.last["v1"]["dof"] == dof


def test_scan_noise_and_omega_options_validate_and_run():
    with pytest.raises(ValueError):
        run(0, params={"scan_noise": "bogus"})
    with pytest.raises(ValueError):
        run(0, heterogeneity_omega="bogus")
    (p, lp), g = run(0, views=["v1"])  # defaults: empirical scan scale, estimated omega
    assert lp["v1"] <= 0.0 and "gamma2_scan" in g.last["v1"] and g.last["v1"]["omega"] >= 0.0
