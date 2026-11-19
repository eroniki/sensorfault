"""Outer loop of "train, freeze, fuse, test" (design v3.2).

Each round trains on the active views (opacity unfrozen again, densification and pruning back on),
freezes opacity, fuses (one round of information-form messages from views to surfels), and tests
every active view against its cavity (leave-view-out) render. Flagged views are dropped (default) or
down-weighted, and the next round trains without them. The loop stops when no view is flagged, when
the round cap is reached, or when more than half of the views are excluded (scene-level alarm).

This is Baarda-style iterative data snooping and hard EM (the gate is the E-step that sets the fault
indicators, training is the M-step); both are known ideas.

False-alarm control across rounds (Holm step-down). The family counts every test the loop may run:
``M = n_views * round_cap`` (each view once per round, including re-tests of the same view in later
rounds and readmission tests of dropped views). The (k+1)-th rejection overall uses the threshold
``alpha / (M - k)``. Within a round, p-values are sorted and compared step-down against
``alpha / (M - k), alpha / (M - k - 1), ...``; the first failure stops the round's rejections. With the
fixed family M, the union bound over at most M tests gives family-wise false alarm <= alpha for the
Bonferroni part; the step-down relaxation is Holm's and is exact for a fixed set of p-values, but
p-values change between rounds as the surface is retrained, so the overall level is calibrated by
clean-run permutation, as the single-pass threshold is.

Default policy: drop only the most significant Holm-rejected view per round (protects against
swamping: a clean view that looks faulty only because a faulty view still shapes the surface), and
no readmission (dropped views stay dropped, so the loop is monotone and terminates). Readmission and
dropping all rejected views per round are ablations.

Gradients run only inside the train step; fusion and testing are not differentiated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


def holm_family_size(
    n_views: int, round_cap: int, test_unit: str = "view", block_size: int = 1
) -> int:
    """Number of tests counted across rounds.

    With per-view tests (the default, also under leave-N-out, where the block only changes the
    cavity) every view is tested in every round: M = N * R_max, independent of the block size.
    With block-level tests (an alternative) each of ceil(N / block_size) blocks is tested per round."""
    if test_unit == "view":
        return n_views * round_cap
    if test_unit == "block":
        return -(-n_views // block_size) * round_cap
    raise ValueError(test_unit)


def holm_threshold(alpha: float, family: int, k_rejected: int) -> float:
    """Threshold for the (k+1)-th rejection overall: alpha / (M - k)."""
    return alpha / max(family - k_rejected, 1)


def holm_round_rejections(
    pvals: Mapping[str, float],
    alpha: float,
    family: int,
    k_rejected: int,
    logp: Mapping[str, float] | None = None,
) -> list[str]:
    """Step-down Holm within one round, continuing the global rejection count ``k_rejected``.

    Returns the rejected views, most significant first. Views are ordered by ``logp`` when given
    (log p-values do not underflow, so they also order views whose p-values are all 0.0), else by p.
    """
    out = []
    key = (lambda kv: logp[kv[0]]) if logp is not None else (lambda kv: kv[1])
    for i, (v, p) in enumerate(sorted(pvals.items(), key=key)):
        if p <= holm_threshold(alpha, family, k_rejected + i):
            out.append(v)
        else:
            break
    return out


@dataclass
class LoopResult:
    excluded: list[str]
    weights: dict[str, float]
    rounds: int
    tests_run: int
    family: int
    alarm: str | None
    history: list[dict] = field(default_factory=list)
    readmitted: list[str] = field(default_factory=list)


def run_gate_loop(
    views: Sequence[str],
    train: Callable[[Mapping[str, float]], object],
    test: Callable[[object, Sequence[str]], Mapping[str, float]],
    alpha: float = 0.05,
    round_cap: int = 10,
    policy: str = "one_per_round",
    exclusion: str = "drop",
    downweight: float = 0.1,
    readmission: bool = False,
) -> LoopResult:
    """Run the outer loop.

    ``train(weights)`` fits the surfels on the views with weight > 0 (weights scale a view's depth
    loss) and returns a state; ``test(state, views)`` returns a p-value per tested view from the
    cavity test, or a pair (p-values, log p-values) so that views whose p-values underflow to 0 are
    still ranked correctly (the most significant view is dropped first). ``policy`` is "one_per_round" (default) or "all_rejected"; ``exclusion`` is "drop"
    (weight 0, default) or "downweight" (weight ``downweight``). With ``round_cap=1`` the loop is
    off (single pass). With ``readmission``, excluded views are re-tested each round and readmitted
    (weight 1) if their test is not rejected; each view is readmitted at most once.
    """
    if policy not in ("one_per_round", "all_rejected"):
        raise ValueError(policy)
    if exclusion not in ("drop", "downweight"):
        raise ValueError(exclusion)
    n = len(views)
    family = holm_family_size(n, round_cap)
    weights = dict.fromkeys(views, 1.0)
    excluded: list[str] = []
    readmitted: list[str] = []
    k = 0
    tests_run = 0
    history: list[dict] = []
    alarm = None
    r = 0
    for r in range(1, round_cap + 1):
        state = train(dict(weights))
        active = [v for v in views if v not in excluded]
        to_test = list(views) if readmission else active
        res = test(state, to_test)
        pvals, logp = (dict(res[0]), dict(res[1])) if isinstance(res, tuple) else (dict(res), None)
        tests_run += len(to_test)
        changed = False
        if readmission:
            for v in list(excluded):
                if v in readmitted:
                    continue
                if pvals[v] > holm_threshold(alpha, family, k):
                    excluded.remove(v)
                    readmitted.append(v)
                    weights[v] = 1.0
                    changed = True
        rej = holm_round_rejections({v: pvals[v] for v in active}, alpha, family, k, logp)
        if policy == "one_per_round":
            rej = rej[:1]
        for v in rej:
            excluded.append(v)
            weights[v] = 0.0 if exclusion == "drop" else downweight
        k += len(rej)
        history.append(
            {"round": r, "pvals": pvals, "rejected": rej, "readmitted": list(readmitted)}
        )
        if len(excluded) > n / 2:
            alarm = "scene_level_more_than_half_excluded"
            break
        if not rej and not changed:
            break
    else:
        if history and history[-1]["rejected"]:
            alarm = "round_cap_reached"
    return LoopResult(excluded, weights, r, tests_run, family, alarm, history, readmitted)
