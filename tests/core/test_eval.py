import numpy as np

from sensorfault.eval import (
    accuracy,
    bootstrap_ci,
    chamfer,
    completeness,
    detection_pr,
    elimination_curve_auc,
    false_alarm_rate_per_100,
    fscore,
    paired_wilcoxon,
    pointcloud_metrics,
    time_to_detect,
)


def test_pointcloud_metrics_known_offset():
    gt = np.random.default_rng(0).random((2000, 3))
    pred = gt + np.array([0.01, 0, 0])
    assert chamfer(pred, gt) <= 0.01 + 1e-9
    assert fscore(pred, gt, 0.02)["f"] == 1.0 and fscore(pred, gt, 0.001)["f"] < 0.5
    half = gt[:1000]
    assert completeness(half, gt) > accuracy(half, gt) == 0.0
    assert pointcloud_metrics(pred, gt, [0.02])["fscore@0.02"] == 1.0
    assert accuracy(np.zeros((0, 3)), gt) == float("inf")


def test_detection_pr():
    r = detection_pr({"a": True, "b": False, "c": False}, {"a": True, "b": True})
    assert r["precision"] == 0.5 and r["recall"] == 1.0
    assert detection_pr({"a": False}, {})["precision"] == 1.0


def test_time_to_detect_censoring():
    r = time_to_detect([10, 10, 10, 10], [12, 14, None, None], horizon=100)
    assert r["n_censored"] == 2
    assert r["median"] == 4.0  # KM: S(2)=.75, S(4)=.5 -> median 4
    assert r["p90"] == float("inf")  # censoring prevents estimating the 90th percentile
    assert time_to_detect([0, 0], [1, 3], 10)["p90"] == 3.0


def test_false_alarm_rate():
    assert false_alarm_rate_per_100(3, 200) == 1.5


def test_elimination_auc_definition():
    f = np.array([5.0, 6, 7, 8])
    c = np.array([0.0, 1, 2, 3])
    assert elimination_curve_auc(f, c) == 1.0  # perfect separation
    assert elimination_curve_auc(c, f) < 0.05  # reversed
    rng = np.random.default_rng(0)
    a = elimination_curve_auc(rng.normal(size=2000), rng.normal(size=2000), fer_max=1.0)
    assert abs(a - 0.5) < 0.05  # fer_max=1 -> ROC AUC


def test_bootstrap_and_wilcoxon():
    v = np.arange(10.0)
    ci = bootstrap_ci(v, n_boot=2000, seed=1)
    assert ci["lo"] < 4.5 < ci["hi"] and bootstrap_ci(v, 2000, seed=1) == ci
    a = np.arange(1.0, 13.0)
    w = paired_wilcoxon(a + 1, a)
    assert w["p"] < 0.01 and w["median_diff"] == 1.0
    assert paired_wilcoxon(a, a)["p"] == 1.0
