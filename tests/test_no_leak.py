"""Leave-view-out guard: view j's render must use nothing from view j."""
import numpy as np
import pytest

from dcal.surface import (LeakError, PoisonedView, ViewData, check_leave_out, fit_surface, render)

H = W = 24


def _look_at(eye, target=np.array([0.0, 0.0, 3.0])):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, -1, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])           # rows: x right, y down, z forward
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


def _scene_depth(K, T_cw):
    """Analytic depth of a room made of a floor (y = 1) and a back wall (z = 5)."""
    v, u = np.mgrid[0:H, 0:W]
    dirs = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    T_wc = np.linalg.inv(T_cw)
    o = T_wc[:3, 3]
    dw = dirs @ T_wc[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dw[..., 1] > 1e-9, (1.0 - o[1]) / dw[..., 1], np.inf)
        t_wall = np.where(dw[..., 2] > 1e-9, (5.0 - o[2]) / dw[..., 2], np.inf)
    t = np.minimum(t_floor, t_wall)
    # camera z-depth: dirs have unit z in camera frame, so ray parameter equals z-depth
    return np.where(np.isfinite(t), t, 10.0)


@pytest.fixture(scope="module")
def views():
    K = np.array([[24.0, 0, W / 2], [0, 24.0, H / 2], [0, 0, 1]])
    rng = np.random.default_rng(0)
    out = {}
    for i, x in enumerate(np.linspace(-1.5, 1.5, 5)):
        T = _look_at(np.array([x, 0.0, 0.0]))
        out[i] = ViewData(i, rng.random((H, W, 3)), _scene_depth(K, T), K, T)
    return out


def _render_fn(surface, target):
    return render(surface, target.K, target.T_cw, (H, W))


def test_pipeline_passes_all_checks(views):
    for j in views:
        res = check_leave_out(fit_surface, _render_fn, views, j, np.random.default_rng(j))
        assert res == {"poisoned": True, "invariant": True, "sensitive": True}, (j, res)


def test_render_has_support_to_test_against(views):
    s = fit_surface(views, 2)
    depth, var, support = _render_fn(s, views[2])
    assert np.isfinite(depth).mean() > 0.5
    assert (support >= 2).mean() > 0.3


def test_leaky_pipeline_is_caught(views):
    """Mutation test: a fit that also fuses the excluded view must fail the check."""
    def leaky_fit(vs, exclude):
        # ignores `exclude`, so it reads (and fuses) view j whenever it is accessible
        return fit_surface(vs, exclude=-1)

    res = check_leave_out(leaky_fit, _render_fn, views, 2, np.random.default_rng(1))
    assert not (res["poisoned"] and res["invariant"])


def test_poisoned_view_raises_on_any_access():
    p = PoisonedView(3)
    assert p.view_id == 3
    with pytest.raises(LeakError):
        p.depth
    with pytest.raises(LeakError):
        p.rgb
