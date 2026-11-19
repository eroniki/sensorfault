"""Leave-view-out guard: view j's render must use nothing from view j."""
import numpy as np
import pytest

from dcal.surface import LeakError, PoisonedView, check_leave_out, fit_surface, render
from scene_helpers import make_views

H = W = 24


@pytest.fixture(scope="module")
def views():
    return make_views(H, W, focal=24.0)


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
