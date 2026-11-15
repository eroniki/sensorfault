"""Tests for the outer loop of train, freeze, fuse, test (logic only; train/test are synthetic)."""

import numpy as np
import pytest

from sensorfault.eval import gate_loop as gl

VIEWS = [f"v{i}" for i in range(10)]
SIG = 1e-9  # p-value of a clearly faulty view


def _train(weights):
    return {v for v, w in weights.items() if w > 0}  # state = set of views shaping the surface


def _null_test(state, views):
    return dict.fromkeys(views, 0.5)


def test_holm_counts_every_view_in_every_round():
    assert gl.holm_family_size(10, 5) == 50
    assert gl.holm_threshold(0.05, 50, 0) == pytest.approx(0.001)
    assert gl.holm_threshold(0.05, 50, 3) == pytest.approx(0.05 / 47)
    rej = gl.holm_round_rejections({"a": 1e-5, "b": 0.0012, "c": 0.5}, 0.05, 50, 0)
    assert rej == ["a"]  # b fails alpha/49 = 0.00102: step-down stops


def test_no_fault_stops_after_one_round():
    res = gl.run_gate_loop(VIEWS, _train, _null_test, round_cap=5)
    assert res.excluded == [] and res.rounds == 1 and res.tests_run == 10 and res.alarm is None


def test_loop_off_is_a_single_pass():
    def test(state, views):
        return {v: (SIG if v in ("v1", "v2") else 0.5) for v in views}

    res = gl.run_gate_loop(VIEWS, _train, test, round_cap=1)
    assert res.rounds == 1 and res.excluded == ["v1"]  # one per round, cap reached
    assert res.alarm == "round_cap_reached"


def test_masking_is_resolved_by_retraining():
    # v2 is faulty but masked while v1 (also faulty) shapes the surface
    def test(state, views):
        out = {}
        for v in views:
            if v == "v1":
                out[v] = SIG
            elif v == "v2":
                out[v] = 0.3 if "v1" in state else SIG
            else:
                out[v] = 0.5
        return out

    single = gl.run_gate_loop(VIEWS, _train, test, round_cap=1)
    looped = gl.run_gate_loop(VIEWS, _train, test, round_cap=5)
    assert "v2" not in single.excluded
    assert looped.excluded == ["v1", "v2"] and looped.rounds == 3 and looped.alarm is None


def test_one_per_round_protects_against_swamping():
    # v3 is clean but looks faulty while the faulty v1 shapes the surface
    def test(state, views):
        out = {}
        for v in views:
            if v == "v1":
                out[v] = 1e-12
            elif v == "v3":
                out[v] = 1e-6 if "v1" in state else 0.5
            else:
                out[v] = 0.5
        return out

    default = gl.run_gate_loop(VIEWS, _train, test, round_cap=5)
    greedy = gl.run_gate_loop(VIEWS, _train, test, round_cap=5, policy="all_rejected")
    assert default.excluded == ["v1"]
    assert set(greedy.excluded) == {"v1", "v3"}  # the clean view is swamped


def test_readmission_recovers_a_swamped_view_once():
    def test(state, views):
        out = {}
        for v in views:
            if v == "v1":
                out[v] = 1e-12
            elif v == "v3":
                out[v] = 1e-6 if "v1" in state else 0.5
            else:
                out[v] = 0.5
        return out

    res = gl.run_gate_loop(
        VIEWS, _train, test, round_cap=5, policy="all_rejected", readmission=True
    )
    assert res.excluded == ["v1"] and res.readmitted == ["v3"]
    assert res.tests_run == 10 * res.rounds  # readmission tests are counted


def test_downweight_keeps_the_view_with_a_small_weight():
    def test(state, views):
        return {v: (SIG if v == "v4" else 0.5) for v in views}

    res = gl.run_gate_loop(VIEWS, _train, test, exclusion="downweight", downweight=0.1)
    assert res.weights["v4"] == 0.1 and res.excluded == ["v4"]


def test_scene_level_alarm_when_more_than_half_excluded():
    def test(state, views):
        return dict.fromkeys(views, SIG)

    res = gl.run_gate_loop(VIEWS, _train, test, round_cap=20, policy="all_rejected")
    assert res.alarm == "scene_level_more_than_half_excluded"


def test_family_wise_false_alarm_under_the_null_is_at_most_alpha():
    rng = np.random.default_rng(0)
    alpha, trials, false_alarms = 0.05, 4000, 0

    def test(state, views):
        return {v: float(rng.uniform()) for v in views}

    for _ in range(trials):
        res = gl.run_gate_loop(VIEWS, _train, test, alpha=alpha, round_cap=5)
        false_alarms += bool(res.excluded)
    assert false_alarms / trials <= alpha + 0.01


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        gl.run_gate_loop(VIEWS, _train, _null_test, policy="x")


def test_holm_count_under_leave_n_out():
    # per-view tests: the block only changes the cavity, the family does not depend on its size
    assert (
        gl.holm_family_size(120, 10, "view", 1) == gl.holm_family_size(120, 10, "view", 9) == 1200
    )
    # block-level tests (alternative): one test per block per round
    assert gl.holm_family_size(120, 10, "block", 5) == 240
    assert gl.holm_family_size(121, 10, "block", 5) == 250
    with pytest.raises(ValueError):
        gl.holm_family_size(10, 1, "strip")


def test_log_p_ranks_views_whose_p_values_underflow():
    # the faulty view v1 and a swamped clean neighbour v3 both have p = 0.0 in floating point;
    # log p ranks v1 first, so the default policy drops v1 and v3 recovers next round
    def test(state, views):
        p, lp = {}, {}
        for v in views:
            if v == "v1":
                p[v], lp[v] = 0.0, -5000.0
            elif v == "v3":
                p[v], lp[v] = (0.0, -800.0) if "v1" in state else (0.5, np.log(0.5))
            else:
                p[v], lp[v] = 0.5, np.log(0.5)
        return p, lp

    res = gl.run_gate_loop(["v3"] + [v for v in VIEWS if v != "v3"], _train, test, round_cap=5)
    assert res.excluded == ["v1"]
    rej = gl.holm_round_rejections({"a": 0.0, "b": 0.0}, 0.05, 10, 0, {"a": -10.0, "b": -99.0})
    assert rej == ["b", "a"]
