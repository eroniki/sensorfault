"""Real 2DGS rasteriser behind the leave-view-out interface (needs torch + CUDA + gsplat; skipped otherwise)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from dcal.gs2d import fit_surface_2dgs, render_2dgs  # noqa: E402
from dcal.surface import check_leave_out  # noqa: E402
from scene_helpers import make_views  # noqa: E402

H, W = 48, 64


@pytest.fixture(scope="module")
def views():
    return make_views(H, W, focal=64.0)


def _render_fn(surface, target):
    return render_2dgs(surface, target.K, target.T_cw, (H, W))


def test_2dgs_surface_passes_leave_out_checks(views):
    for j in (0, 2, 4):
        res = check_leave_out(lambda v, e: fit_surface_2dgs(v, e), _render_fn, views, j, np.random.default_rng(j), atol=1e-5)
        assert res == {"poisoned": True, "invariant": True, "sensitive": True}, (j, res)


def test_2dgs_render_matches_the_analytic_room(views):
    j = 2
    depth, var, support = _render_fn(fit_surface_2dgs(views, j), views[j])
    gt = views[j].depth
    assert np.array_equal(np.isfinite(depth), support >= 1)      # depth exists exactly where some view supports it
    ok = np.isfinite(depth) & (support >= 2)
    assert ok.mean() > 0.3
    rel = np.abs(depth[ok] - gt[ok]) / gt[ok]
    assert np.median(rel) < 0.03, np.median(rel)
    assert np.all(var[ok] > 0)


def test_optimised_fit_never_touches_the_excluded_view_and_handles_sparse_depth():
    """fit_2dgs on sparse depth (NaN holes) with the held-out view poisoned: must run and render with finite depth."""
    import dataclasses

    from dcal.gs2d_fit import fit_2dgs, render_view
    from dcal.surface import PoisonedView

    base = make_views(H, W, focal=64.0)
    rng = np.random.default_rng(0)
    sparse = {k: dataclasses.replace(v, depth=np.where(rng.random(v.depth.shape) < 0.25, v.depth, np.nan)) for k, v in base.items()}
    j = 2
    poisoned = {k: (PoisonedView(k) if k == j else v) for k, v in sparse.items()}
    P, info = fit_2dgs(poisoned, j, iters=15)
    rgb, depth, alpha = render_view(P, base[j].K, base[j].T_cw, (H, W))
    assert info["n_surfels"] > 100 and torch.isfinite(depth).all() and torch.isfinite(rgb).all()
    assert (alpha > 0.5).float().mean() > 0.2
