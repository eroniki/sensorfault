"""Registered fusers of the gate test bed: ``no_gate``, ``oracle_gate``, ``view_gate``.

All three fit a depth-supervised 2DGS with a ``SurfelBackend`` (injectable; default lazily builds
``gs.fit.GsplatBackend``). ``view_gate`` additionally runs the outer loop of
``eval.gate_loop.run_gate_loop`` with ``gate.test(state, frames, backend) -> (pvals, logp)``.
Everything here is numpy only and runs against a fake backend / gate on CPU.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np

from sensorfault.baselines.base import BaseFuser, register
from sensorfault.eval.gate_loop import run_gate_loop
from sensorfault.gs.interfaces import SurfelBackend, SurfelState
from sensorfault.types import DepthFrame, FusionResult

# keys of a fuser cfg that never belong to the gate's own parameters
_RUNNER_KEYS = {"oracle", "seed", "faulty_views", "faulty_sensors"}
_LOOP_KEYS = ("round_cap", "policy", "exclusion", "readmission", "alpha", "downweight")


def _torch_gsplat_available() -> tuple[bool, str]:
    try:
        import gsplat  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        return False, f"torch/gsplat missing: {e}"
    return True, ""


def _default_backend():
    from sensorfault.gs.fit import GsplatBackend  # lazy: needs torch + gsplat

    return GsplatBackend()


def outer_loop_params(cfg: Mapping) -> dict:
    """round_cap/policy/exclusion/readmission/alpha/downweight from ``cfg['outer_loop']`` (dict) or
    the flat keys ``outer_loop_<name>`` / ``<name>`` (gate_variants.yaml uses the flat form)."""
    nested = cfg.get("outer_loop")
    out: dict = dict(nested) if isinstance(nested, Mapping) else {}
    for k in _LOOP_KEYS:
        for key in (f"outer_loop_{k}", k):
            if key in cfg and k not in out:
                out[k] = cfg[key]
    if isinstance(nested, str) and "round_cap 1" in nested:  # "off (round_cap 1)" in the yaml
        out.setdefault("round_cap", 1)
    if cfg.get("round_cap") is not None:
        out["round_cap"] = cfg["round_cap"]
    return {k: v for k, v in out.items() if k in _LOOP_KEYS}


def faulty_views(cfg: Mapping) -> list[str]:
    """Faulty view ids from the manifest key ``faulty_sensors`` (also ``faulty_views``), either at
    the top level of cfg or under ``cfg['oracle']`` (as the runner passes it)."""
    for src in (cfg.get("oracle") or {}, cfg):
        for key in ("faulty_views", "faulty_sensors"):
            if src.get(key) is not None:
                return [str(v) for v in src[key]]
    return []


def _result(
    state: SurfelState,
    excluded: Sequence[str],
    views: Sequence[str],
    scores: Mapping[str, float],
    meta: dict,
    t0: float,
) -> FusionResult:
    normals = np.asarray(state.R)[:, :, 2]
    ex = set(excluded)
    return FusionResult(
        points=np.asarray(state.mu, dtype=np.float32),
        normals=normals.astype(np.float32),
        excluded_sensors={v: v in ex for v in views},
        sensor_scores={v: float(scores.get(v, 0.0)) for v in views},
        runtime_s=time.perf_counter() - t0,
        meta=meta,
    )


class _GsFuser(BaseFuser):
    def __init__(self, backend: SurfelBackend | None = None, **_ignored):
        self._backend = backend

    @property
    def backend(self) -> SurfelBackend:
        if self._backend is None:
            self._backend = _default_backend()
        return self._backend


class _Avail:
    """Descriptor so ``available()`` works on both the class (runner) and an instance with an
    injected backend."""

    def __get__(self, inst, owner):
        if inst is not None and getattr(inst, "_backend", None) is not None:
            return lambda: (True, "")
        return _torch_gsplat_available


_GsFuser.available = _Avail()  # type: ignore[assignment]


@register("no_gate")
class NoGate(_GsFuser):
    """Single 2DGS fit with every view's depth trusted (weight 1)."""

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        views = [f.sensor_id for f in frames]
        state = self.backend.fit(frames, dict.fromkeys(views, 1.0))
        return _result(state, [], views, {}, {"rounds": 1, "tests_run": 0, "alarm": None}, t0)


@register("oracle_gate")
class OracleGate(_GsFuser):
    """Single fit that drops exactly the views listed faulty in the manifest (weight 0)."""

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        views = [f.sensor_id for f in frames]
        bad = set(faulty_views(cfg)) & set(views)
        weights = {v: 0.0 if v in bad else 1.0 for v in views}
        state = self.backend.fit(frames, weights)
        scores = {v: float(v in bad) for v in views}
        meta = {"rounds": 1, "tests_run": 0, "alarm": None, "oracle_dropped": sorted(bad)}
        return _result(state, sorted(bad), views, scores, meta, t0)


@register("view_gate")
class ViewGate(_GsFuser):
    """Depth-only view gate: leave-view-out cavity test inside the Holm outer loop."""

    def __init__(self, backend: SurfelBackend | None = None, gate=None, **_ignored):
        super().__init__(backend)
        self._gate = gate

    def _make_gate(self, cfg: Mapping, frames: Sequence[DepthFrame]):
        if self._gate is not None:
            return self._gate
        from sensorfault.gs.gate import ViewGateTest

        loop_keys = set(_LOOP_KEYS) | {f"outer_loop_{k}" for k in _LOOP_KEYS} | {"outer_loop"}
        params = {k: v for k, v in cfg.items() if k not in _RUNNER_KEYS | loop_keys}
        sigma = params.pop("sigma", 0.01)
        return ViewGateTest(self.backend, frames, sigma=sigma, params=params)

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        t0 = time.perf_counter()
        views = [f.sensor_id for f in frames]
        by_id = {f.sensor_id: f for f in frames}
        backend, gate = self.backend, self._make_gate(cfg, frames)
        loop = outer_loop_params(cfg)
        last: dict = {}
        logps: dict[str, float] = {}

        def train(weights: Mapping[str, float]) -> SurfelState:
            last["w"] = dict(weights)
            last["state"] = backend.fit(frames, weights)
            last["state"].meta["weights"] = dict(weights)  # the gate subtracts only fused views
            return last["state"]

        def test(state, tested: Sequence[str]):
            if hasattr(gate, "test"):
                pv, lp = gate.test(state, [by_id[v] for v in tested], backend)
            else:  # ViewGateTest: callable on view ids
                pv, lp = gate(state, list(tested))
            for v in tested:  # keep the most significant evidence per view across rounds
                logps[v] = min(logps.get(v, 0.0), float(lp[v]) if lp and v in lp else _log(pv[v]))
            return pv, lp

        res = run_gate_loop(views, train, test, **loop)
        state = last["state"]
        meta = {
            "rounds": res.rounds,
            "tests_run": res.tests_run,
            "alarm": res.alarm,
            "history": res.history,
            "family": res.family,
            "readmitted": res.readmitted,
            "weights": dict(res.weights),
            "outer_loop": loop,
        }
        if res.weights != last["w"]:  # the last round changed weights after the last fit
            state = backend.fit(frames, res.weights)
            state.meta["weights"] = dict(res.weights)
            meta["final_refit"] = True
        scores = {v: -logps.get(v, 0.0) for v in views}
        return _result(state, res.excluded, views, scores, meta, t0)


def _log(p: float) -> float:
    return float(np.log(max(float(p), 1e-300)))
