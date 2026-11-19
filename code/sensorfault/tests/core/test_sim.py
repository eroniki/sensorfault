import numpy as np
import pytest

from sensorfault.sim import (
    SensorModel,
    SyntheticConfig,
    depth_residual_variance,
    generate_synthetic,
    grazing_gate,
    normal_direction_variance,
    structured_light_sigma_z,
)


def test_sl_sigma_formula():
    z, sd, f, b = 2.0, 0.5, 580.0, 0.075
    assert np.isclose(structured_light_sigma_z(z, sd, f, b), z**2 * sd / (f * b))


def test_variance_matches_monte_carlo_on_plane():
    """Plane at incidence theta: ray direction perturbed by phi in the incidence plane."""
    rng = np.random.default_rng(1)
    r, sr, sphi = 3.0, 0.004, 0.002
    for theta_deg in (0.0, 30.0, 60.0):
        th = np.deg2rad(theta_deg)
        d = r * np.cos(th)  # perpendicular distance to plane
        phi = rng.normal(0, sphi, 400_000)
        r_meas = d / np.cos(th + phi) + rng.normal(0, sr, phi.shape)
        mc = np.var(r_meas - r)
        model = depth_residual_variance(r, th, sr, sphi, eps=0.0, var_floor=0, var_cap=10)
        assert abs(mc / float(model) - 1) < 0.03, (theta_deg, mc, float(model))


def test_variance_regularised_bounded_and_ordered():
    th = np.deg2rad(np.array([10.0, 60.0, 80.0, 84.9, 89.0]))
    v = depth_residual_variance(3.0, th, 0.005, 0.003)
    v0 = depth_residual_variance(3.0, th, 0.005, 0.003, eps=1e-6, var_cap=1e9)
    assert np.all(np.isfinite(v)) and np.all(v <= 0.25 + 1e-12) and np.all(v >= 1e-8)
    assert np.all(v[:4] <= v0[:4] + 1e-12)  # clamp never inflates
    assert v[4] == 0.25  # beyond 85 deg gate -> cap
    assert grazing_gate(np.deg2rad([84.0, 86.0])).tolist() == [True, False]


def test_normal_variance_bounded_at_grazing():
    v = normal_direction_variance(3.0, np.deg2rad(89.9), 0.005, 0.003)
    assert np.isclose(
        v, 0.005**2 * np.cos(np.deg2rad(89.9)) ** 2 + 9 * 0.003**2 * np.sin(np.deg2rad(89.9)) ** 2
    )


@pytest.mark.parametrize("kind", ["kinect_sl", "tof", "lidar", "stereo"])
def test_sensor_models_run_and_are_seeded(kind):
    z = np.full((20, 30), 2.0, np.float32)
    cos = np.full_like(z, 0.8, dtype=float)
    m = SensorModel(kind)
    a = m(z, cos, np.random.default_rng(3), 500.0)
    b = m(z, cos, np.random.default_rng(3), 500.0)
    assert np.array_equal(a, b) and a.dtype == np.float32 and abs(a[a > 0].mean() - 2.0) < 0.05


def test_synthetic_generator_deterministic_and_geometry():
    cfg = SyntheticConfig(n_sensors=3, n_frames=2, height=24, width=32)
    a, b = generate_synthetic(cfg, 7), generate_synthetic(cfg, 7)
    assert len(a.frames) == 6 and np.array_equal(a.frames[0].depth, b.frames[0].depth)
    assert np.array_equal(a.gt_points, b.gt_points)
    # GT points lie inside the room box
    L = np.array(cfg.room_size)
    assert np.all(a.gt_points >= -1e-3) and np.all(a.gt_points <= L + 1e-3)
    assert (a.frames[0].depth > 0).mean() > 0.95
    c = generate_synthetic(cfg, 8)
    assert not np.array_equal(a.frames[0].depth, c.frames[0].depth)
