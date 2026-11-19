"""CPU checks of the ETH3D gate glue (binning, windows, statistic on synthetic windows from the toy world)."""
import numpy as np
import pytest

from dcal.gate import Window, bin_mean, make_windows, run_gate, verdict, window_T
from dcal.toy import ToyConfig, World

G = 10


def test_bin_mean_ignores_invalid_and_counts():
    img = np.arange(16, dtype=float).reshape(4, 4)
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    m, c = bin_mean(img, 2, valid)
    assert c.tolist() == [[3, 4], [4, 4]]
    assert m[0, 0] == pytest.approx((1 + 4 + 5) / 3)
    assert m[1, 1] == pytest.approx((10 + 11 + 14 + 15) / 4)
    S, _ = bin_mean(np.stack([img, 2 * img]), 2)
    assert S.shape == (2, 2, 2) and S[1, 0, 1] == pytest.approx(2 * S[0, 0, 1])


def test_make_windows_masks_uncovered_bins_and_sparse_gt():
    rng = np.random.default_rng(0)
    s = 40
    samples = rng.standard_normal((4, s, s))
    render = np.full((s, s), 5.0)
    alpha = np.ones((s, s))
    alpha[:, :10] = 0.0                          # left quarter uncovered
    gt = np.full((s, s), np.nan)
    gt[::3, ::3] = 5.0
    wins = make_windows(0, samples, render, alpha, gt, g=4, tiles=1, min_gt_px=2)
    assert len(wins) == 1
    w = wins[0]
    assert w.samples.shape == (4, 16)
    keep = w.keep.reshape(4, 4)
    assert not keep[:, 0].any() and keep[:, 1:].all()
    assert np.isfinite(w.gt).all()
    assert len(make_windows(0, samples, render, alpha, gt, g=4, tiles=2)) == 4


def _toy_windows(n, seed=0, miscal=1.0):
    world = World(ToyConfig(g=G, miscal=miscal), seed)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        S, _, _, _, (a, b) = world.simulate(rng)
        d_true = world.mu + world.Lu @ rng.standard_normal(world.n)
        S = world.mu[None] + miscal * (world.Lu @ rng.standard_normal((world.n, world.cfg.M))).T
        depth = a * d_true + 40.0
        render = depth * (1 + 1e-3 * rng.standard_normal(world.n))
        out.append(Window(i % 6, (0, i), S, render, np.ones(world.n, bool), depth.copy(), (G, G)))
    return out


def test_fault_raises_T():
    w = _toy_windows(1)[0]
    t0 = window_T(w, "kernel", rel_sigma_r=1e-3)["T"]
    t1 = window_T(w, "kernel", rel_sigma_r=1e-3, fault="square", kappa=4.0)["T"]
    assert t1 > t0 + 20


def test_run_gate_calibrated_toy():
    res = run_gate(_toy_windows(30), covs=("kernel", "iso"), faults=("square",), kappas=(1.0, 4.0), rel_sigma_r=1e-3)
    c = res["calibration"]
    assert 0.55 < c["coverage_1sd"] < 0.8 and 0.88 < c["coverage_2sd"] < 0.99
    k = res["covs"]["kernel"]
    assert k["false_alarm_loo"] <= 0.2
    assert k["power"]["square"]["4.0"] >= 0.8
    assert k["power"]["square"]["4.0"] >= k["power"]["square"]["1.0"]
    lines = verdict(res, kappa="4.0")
    assert any(line.startswith("GATE kernel") for line in lines)


def test_overconfident_spread_shows_in_coverage():
    c = run_gate(_toy_windows(20, miscal=0.5), covs=("iso",), faults=("square",), kappas=(1.0,), rel_sigma_r=1e-3)["calibration"]
    assert c["coverage_1sd"] < 0.5 and c["mean_z2"] > 2


def test_stat_command_end_to_end(tmp_path):
    """The CPU step of scripts/gate_eth3d.py on fake cached samples and renders (two 'scenes' pooled)."""
    import subprocess
    import sys
    from pathlib import Path

    rng = np.random.default_rng(1)
    s = 48
    yy, xx = np.mgrid[0:s, 0:s] / s
    for scene in ("a", "b"):
        c = tmp_path / scene
        c.mkdir()
        for j in range(4):
            depth = 3 + np.sin(3 * xx + j) + yy
            S = (depth - 3.5)[None] + 0.05 * rng.standard_normal((6, s, s))
            gt = np.where(rng.random((s, s)) < 0.3, depth, np.nan)
            np.savez(c / f"samples_v{j:02d}_k10.npz", samples=S.astype(np.float32), view=j, ddim_steps=10, step=2000)
            np.savez(c / f"render_v{j:02d}.npz", render=depth.astype(np.float32), alpha=np.ones((s, s), np.float32),
                     gt=gt.astype(np.float32), view=j)
    out = tmp_path / "gate.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "gate_eth3d.py"
    p = subprocess.run([sys.executable, str(script), "stat", "--cache", str(tmp_path / "a"), "--cache", str(tmp_path / "b"),
                        "--grid", "8", "--tiles", "1", "--samples", "6", "--kappas", "2", "--out", str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "GATE kernel" in p.stdout
    import json
    res = json.loads(out.read_text())["10"]
    assert res["n_windows"] == 8 and res["n_views"] == 8
