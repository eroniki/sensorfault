import numpy as np
import pytest

from sensorfault.elim import (
    eliminate,
    entropy_from_logF,
    literal_algorithm1,
    log_lambda,
    normalise_logpdf,
)

X = np.linspace(-20, 20, 40001)
DV = X[1] - X[0]


def gauss(mu, s):
    return normalise_logpdf(-0.5 * ((X - mu) / s) ** 2, DV)


def toy(extra_good=0):
    d = {"g1": gauss(0, 1), "g2": gauss(0, 1), "g3": gauss(0, 1), "bad": gauss(8, 0.3)}
    return d


def test_paper_review_finding_entropy_drops_but_lambda_rejects():
    d = toy()
    H3 = entropy_from_logF(d["g1"] + d["g2"] + d["g3"], DV)
    H4 = entropy_from_logF(d["g1"] + d["g2"] + d["g3"] + d["bad"], DV)
    assert pytest.approx(0.87, abs=0.01) == H3 and pytest.approx(0.095, abs=0.01) == H4
    assert H4 < H3  # entropy ALONE rewards adding the faulty sensor
    log_S = np.log(len(X) * DV)
    ll_bad = log_lambda(d, ["g1", "g2", "g3"], "bad", DV, log_S)
    assert ll_bad < 0  # Lambda < 1 -> rejected
    r = eliminate(d, DV)
    assert r.status == "ok" and r.excluded == ["bad"] and r.kept == ["g1", "g2", "g3"]
    assert r.log_lambdas["bad"] < 0


def test_order_independence():
    d = toy()
    names = list(d)
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = list(rng.permutation(names))
        r = eliminate({k: d[k] for k in p}, DV)
        assert sorted(r.excluded) == ["bad"]


def test_no_fault_keeps_all():
    d = {f"g{i}": gauss(0.1 * i, 1) for i in range(4)}
    r = eliminate(d, DV)
    assert r.excluded == [] and r.status == "ok"


def test_two_faulty_with_budget_and_no_consensus_cases():
    d = {f"g{i}": gauss(0, 1) for i in range(4)}
    d["b1"], d["b2"] = gauss(7, 0.4), gauss(-9, 0.4)
    r = eliminate(d, DV)  # m=6 -> F=2
    assert sorted(r.excluded) == ["b1", "b2"]
    r1 = eliminate(d, DV, max_faulty=1)  # budget too small: cannot remove both
    assert r1.status == "no_consensus" and r1.excluded == []
    two = {"a": gauss(0, 1), "b": gauss(8, 1)}
    assert eliminate(two, DV).status == "no_consensus"  # K=2 unidentifiable
    # majority faulty with same-sign bias: 2 good, 3 bad cluster -> cannot identify the good pair
    maj = {
        "g1": gauss(0, 1),
        "g2": gauss(0, 1),
        "b1": gauss(6, 0.3),
        "b2": gauss(6, 0.3),
        "b3": gauss(6, 0.3),
    }
    rm = eliminate(maj, DV)
    assert (
        "g1" in rm.excluded or rm.status == "no_consensus"
    )  # honest failure: does not keep the good minority


def test_seed_is_max_Z_not_entropy():
    d = toy()
    r = eliminate(d, DV)
    assert set(r.chain[0]) <= {"g1", "g2", "g3"}


def test_literal_alg1_is_degenerate_and_excludes_nobody():
    d = toy()
    r = literal_algorithm1(d, DV)
    assert r.status == "degenerate_literal" and r.excluded == []  # bug B2: h*=-inf never seeds
    r2 = literal_algorithm1(d, DV, fix_seed_init=True)
    r3 = literal_algorithm1(d, DV, order=["bad", "g3", "g2", "g1"], fix_seed_init=True)
    assert r2.status == "ok"
    assert isinstance(
        r3.kept, list
    )  # order-dependence (B3) is exercised; outcome documented, not asserted
