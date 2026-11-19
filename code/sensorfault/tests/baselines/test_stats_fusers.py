import numpy as np
import pytest
from synth import make_frames, pose

from sensorfault.baselines import get_fuser, list_fusers
from sensorfault.baselines.loo_chi2 import LooChi2


def test_registry_all_names():
    for n in [
        "tsdf_open3d",
        "median_fusion",
        "consistency_voting",
        "merrell_visibility",
        "tvl1_range",
        "loo_chi2",
        "subset_msac",
        "openmvs",
        "senfunet",
        "routedfusion",
        "gs_mcmc_depth",
        "twodgs_depth",
    ]:
        assert n in list_fusers()
        ok, why = get_fuser(n).available()
        assert isinstance(ok, bool) and isinstance(why, str)


def test_median_robust_and_scores():
    fr = make_frames([0, 0, 0, 0, 0.3])
    r = get_fuser("median_fusion").fuse(fr, {})
    assert abs(np.median(r.points[:, 2]) - 2.0) < 0.01
    assert r.sensor_scores["s4"] > 0.9
    assert all(r.sensor_scores[f"s{i}"] < 0.05 for i in range(4))


def test_median_shifted_poses():
    poses = [pose(tx=0.002 * i) for i in range(4)]
    fr = make_frames([0, 0, 0, 0.3], poses=poses)
    r = get_fuser("median_fusion").fuse(fr, {})
    assert len(r.points) > 0.8 * 64 * 64
    assert abs(np.median(r.points[:, 2]) - 2.0) < 0.01
    assert r.sensor_scores["s3"] > 0.8


def test_loo_flags_bias_sensor():
    fr = make_frames([0, 0, 0, 0, 0.05], sigma=0.005)
    r = get_fuser("loo_chi2").fuse(fr, {"sigma": 0.005, "tau": 1.5})
    assert r.excluded_sensors["s4"] is True
    assert sum(r.excluded_sensors.values()) == 1
    assert r.sensor_scores["s4"] == max(r.sensor_scores.values())
    assert abs(np.median(r.points[:, 2]) - 2.0) < 0.005


def test_loo_clean_no_exclusion_and_calibration():
    maxT = [
        LooChi2().max_statistic(make_frames([0] * 5, seed=s), {"sigma": 0.005}) for s in range(10)
    ]
    tau = LooChi2.calibrate_tau(maxT, alpha=0.1)
    assert 0.8 < tau < 1.3
    r = get_fuser("loo_chi2").fuse(
        make_frames([0] * 5, seed=99), {"sigma": 0.005, "tau": tau * 1.1}
    )
    assert not any(r.excluded_sensors.values())


def test_loo_min_sensors_respected():
    fr = make_frames([0, 0.05, 0.1])
    r = get_fuser("loo_chi2").fuse(fr, {"sigma": 0.005, "tau": 1.1})
    assert sum(not e for e in r.excluded_sensors.values()) >= 3


def test_msac_flags_bias_sensor():
    fr = make_frames([0, 0, 0, 0.05, 0], sigma=0.005)
    r = get_fuser("subset_msac").fuse(fr, {"sigma": 0.005})
    assert r.excluded_sensors == {"s0": False, "s1": False, "s2": False, "s3": True, "s4": False}


def test_msac_two_biased_and_limit():
    r = get_fuser("subset_msac").fuse(make_frames([0, 0, 0, 0, 0.05, -0.06]), {"sigma": 0.005})
    assert r.excluded_sensors["s4"] and r.excluded_sensors["s5"]
    assert not any(r.excluded_sensors[f"s{i}"] for i in range(4))
    with pytest.raises(ValueError):
        get_fuser("subset_msac").fuse(make_frames([0] * 7), {})
