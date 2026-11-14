"""Corrected Byzantine admission criterion (PLAN.md section 2) on gridded per-sensor PDFs.

All PDFs live on a common regular grid (flattened, any dimension) with cell volume ``dv``.
Everything is computed in log space; ``logpdfs[name]`` holds log f_i(x) per cell.

Definitions
-----------
F_C = prod_{i in C} f_i;  Z_C = sum F_C * dv;  p_C = F_C / Z_C;  H(C) = -sum p_C log p_C dv.
Bayes factor of sensor k against a uniform null on the domain S (|S| = volume):
    Lambda_k(C) = |S| * Z_{C u k} / Z_C   (>= 1  <=> k is at least as well explained by the
                                             consensus of C as by "no information")
Requires each f_i to be a normalised density (``normalise_logpdf``).

Corrected algorithm (``eliminate``)
-----------------------------------
1. Seed: pair (i,j) maximising Z_ij (NOT entropy). Seed must itself be admissible: Lambda >= tau.
2. Expand: repeatedly add the feasible sensor with the largest Lambda_k(C) >= tau
   (order independent; ties broken by name order).
3. Stop when no remaining sensor is admissible.
4. Select: among the nested chain of admissible subsets with at most F removals
   (|C| >= m - F), return the one minimising H(C).  If no chain member satisfies the
   removal budget, report ``status="no_consensus"`` and exclude nobody (attribution
   unidentifiable: e.g. K=2 or majority-faulty).
Entropy measures precision of the surviving consensus; Lambda is the fault test.  Entropy
alone cannot flag a sharp, biased sensor (see tests/core/test_elim.py toy).

Literal Alg. 1 (``literal_algorithm1``, legacy ablation A8) reproduces the printed Algorithm 1 of
Ambarkutuk et al. (Sensors 23(23):9309). The author has confirmed that its inequality is a typo and
that Eq. 14 is intended (PLAN.md section 2); the corrected variant is the one used, and the literal
reading is kept only to document the typo (legacy v1 code and tests).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.special import logsumexp


def normalise_logpdf(logf: np.ndarray, dv: float) -> np.ndarray:
    """Normalise so that sum(exp(logf)) * dv = 1."""
    return logf - (logsumexp(logf) + np.log(dv))


def log_Z(logF: np.ndarray, dv: float) -> float:
    return float(logsumexp(logF) + np.log(dv))


def entropy_from_logF(logF: np.ndarray, dv: float) -> float:
    """Differential entropy of p = F/Z of the normalised product (grid quadrature)."""
    lz = log_Z(logF, dv)
    logp = logF - lz
    p = np.exp(logp)
    return float(-np.sum(np.where(p > 0, p * logp, 0.0)) * dv)


def _logF(logpdfs: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.sum([logpdfs[n] for n in names], axis=0)


def log_lambda(logpdfs, C: Sequence[str], k: str, dv: float, log_S: float) -> float:
    """log Lambda_k(C) = log|S| + log Z_{C u k} - log Z_C."""
    return float(log_S + log_Z(_logF(logpdfs, list(C) + [k]), dv) - log_Z(_logF(logpdfs, C), dv))


@dataclass
class EliminationResult:
    kept: List[str]
    excluded: List[str]
    status: str  # "ok" | "no_consensus" | "degenerate_literal"
    chain: List[List[str]] = field(default_factory=list)
    log_lambdas: Dict[str, float] = field(
        default_factory=dict
    )  # final Lambda of each sensor vs kept set
    entropies: List[float] = field(default_factory=list)  # H of each chain member
    meta: dict = field(default_factory=dict)

    def excluded_flags(self, names: Sequence[str]) -> Dict[str, bool]:
        return {n: n in self.excluded for n in names}


def eliminate(
    logpdfs: Mapping[str, np.ndarray],
    dv: float,
    max_faulty: Optional[int] = None,
    tau: float = 1.0,
    log_S: Optional[float] = None,
) -> EliminationResult:
    """Corrected admission + selection. ``tau`` is the Lambda threshold (log taken internally).

    ``max_faulty`` (F) defaults to ceil(m/2) - 1 (strict minority). Requires m >= 3 to be
    identifiable; for m < 3 returns no_consensus. ``log_S`` defaults to log(n_cells*dv).
    """
    names = list(logpdfs)
    m = len(names)
    n_cells = len(next(iter(logpdfs.values())))
    log_S = float(np.log(n_cells * dv)) if log_S is None else float(log_S)
    F = (int(np.ceil(m / 2)) - 1) if max_faulty is None else int(max_faulty)
    if m < 3 or m / 2 <= F:
        return EliminationResult(
            names,
            [],
            "no_consensus",
            chain=[names],
            meta={"reason": "need m>=3 and F<m/2", "m": m, "F": F},
        )
    log_tau = float(np.log(tau))
    # Seed: max Z_ij
    best, bestZ = None, -np.inf
    for a, b in itertools.combinations(names, 2):
        lz = log_Z(_logF(logpdfs, [a, b]), dv)
        if lz > bestZ + 1e-12:
            best, bestZ = (a, b), lz
    C = list(best)
    chain = [list(C)]
    # seed admissibility: Bayes factor of j given i (symmetric in Z_ij / (Z_i Z_j) form is not needed)
    seed_ll = log_lambda(logpdfs, [C[0]], C[1], dv, log_S)
    if seed_ll < log_tau:
        return EliminationResult(
            names,
            [],
            "no_consensus",
            chain=[C],
            meta={"reason": "seed pair not admissible", "seed": C, "F": F},
        )
    while True:
        cand = [(log_lambda(logpdfs, C, k, dv, log_S), k) for k in names if k not in C]
        cand = [(l, k) for l, k in cand if l >= log_tau]
        if not cand:
            break
        l, k = max(cand, key=lambda t: (t[0], -names.index(t[1])))
        C.append(k)
        chain.append(list(C))
    ents = [entropy_from_logF(_logF(logpdfs, c), dv) for c in chain]
    feas = [i for i, c in enumerate(chain) if m - len(c) <= F]
    final_ll = {
        k: log_lambda(logpdfs, chain[-1], k, dv, log_S) for k in names if k not in chain[-1]
    }
    if not feas:
        return EliminationResult(
            names,
            [],
            "no_consensus",
            chain=chain,
            log_lambdas=final_ll,
            entropies=ents,
            meta={"reason": "removal budget exceeded", "F": F},
        )
    i_best = min(feas, key=lambda i: (ents[i], -len(chain[i])))
    kept = sorted(chain[i_best], key=names.index)
    return EliminationResult(
        kept,
        [n for n in names if n not in kept],
        "ok",
        chain=chain,
        log_lambdas=final_ll,
        entropies=ents,
        meta={"F": F, "log_S": log_S, "tau": tau, "selected_chain_index": i_best},
    )


def literal_algorithm1(
    logpdfs: Mapping[str, np.ndarray],
    dv: float,
    order: Optional[Sequence[str]] = None,
    fix_seed_init: bool = False,
) -> EliminationResult:
    """Literal reading of Alg. 1 for ablation A8, with documented bugs (PLAN.md section 2).

    Here h(C) = -log Z_C, the unnormalised-product (conditional) surprisal that Eq. 14 appears
    to define (bug 4: it is not the differential entropy of a normalised PDF).

    Flagged bugs reproduced when ``fix_seed_init=False``:
      B1  Eq.14/prose seed = MAX entropy but line 5 tests ``h_ij <= h*`` (a MIN test).
      B2  Line 2 sets h* = -inf so line 5 never fires -> C stays empty. Then line 11
          ``h_i >= h*`` is true for every sensor: everything is accepted, nobody excluded.
      B3  Line 11 accepts a sensor when entropy RISES (opposite of 'sharpen the joint PDF').
          Single pass, result depends on ``order``.
      B5  No maximum-number-of-faulty-sensors bound; independence assumed.
    ``fix_seed_init=True`` initialises h* = +inf so the seed pair is the MIN-h pair (B2 fixed,
    B1/B3 retained), which exposes B3 order dependence and entropy-only blindness.
    """
    names = list(logpdfs)
    order = list(order) if order is not None else names
    h = lambda C: -log_Z(_logF(logpdfs, C), dv)
    h_star = np.inf if fix_seed_init else -np.inf
    C: List[str] = []
    for a, b in itertools.combinations(names, 2):
        hij = h([a, b])
        if hij <= h_star:
            h_star, C = hij, [a, b]
    degenerate = not C
    for k in order:
        if k in C:
            continue
        hk = h(C + [k]) if C else h([k])
        if hk >= h_star:  # B3: accepts when surprisal rises
            C.append(k)
            h_star = hk if not degenerate else h_star
    kept = [n for n in names if n in C] if not degenerate else list(names)
    return EliminationResult(
        kept,
        [n for n in names if n not in kept],
        "degenerate_literal" if degenerate else "ok",
        chain=[list(kept)],
        meta={"variant": "literal_alg1", "fix_seed_init": fix_seed_init, "order": order},
    )
