"""Experiment runner.

    python -m sensorfault.runner --config configs/legacy_sensor_faults/h1_coherent_fault_regimes.yaml --dry-run

Pipeline: load config (+ ``extends`` chain + overlays) -> build frames (synthetic generator;
dataset loaders belong to the data scripts) -> inject faults -> run listed fusers from the
registry (unavailable ones are skipped with a logged reason) -> evaluate -> write
``outputs/<exp>/<timestamp>/{results.json,run_manifest.json,fault_manifests/*.json}``.

Only numpy/scipy/pyyaml are needed for ``--dry-run``; two built-in reference fusers
(``builtin_naive_union``, ``builtin_oracle_exclusion``) always run so the evaluation path is
exercised even when no baseline is installed. They are NOT paper baselines.

Test-split guard: ``--split test`` refuses to run unless git tag ``prereg-v4`` exists
(see prereg/TAGGING.md); override with ``--allow-untagged`` (results then marked untagged).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from sensorfault.eval import (
    bootstrap_ci,
    detection_pr,
    pointcloud_metrics,
)
from sensorfault.faults import (
    FaultSpec,
    expand_cells,
    inject_faults,
    scenario_specs,
    write_manifest,
)
from sensorfault.sim import SyntheticConfig, backproject, generate_synthetic
from sensorfault.types import DepthFrame, FusionResult

log = logging.getLogger("sensorfault.runner")


# ------------------------------------------------------------------ config handling
def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = (
            deep_merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else copy.deepcopy(v)
        )
    return out


def load_config(path, overlays: Sequence = (), _seen=None) -> dict:
    """Load YAML; resolve ``extends: <file or list>`` (relative to the file) depth-first,
    then merge ``overlays`` (files) last. Later wins; dicts merge recursively, lists replace."""
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"cyclic extends at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    base: dict = {}
    ext = raw.pop("extends", [])
    for e in [ext] if isinstance(ext, str) else ext:
        base = deep_merge(base, load_config(path.parent / e, (), _seen | {path}))
    cfg = deep_merge(base, raw)
    for o in overlays:
        cfg = deep_merge(cfg, load_config(o, (), _seen | {path}))
    cfg["_config_path"] = str(path)
    return cfg


def load_baseline_table(config_dir: Path) -> Dict[str, dict]:
    """fuser name -> {id, family, params, ...} from configs/baselines/*.yaml."""
    table: Dict[str, dict] = {}
    for f in sorted((config_dir / "baselines").glob("*.yaml")):
        for b in (yaml.safe_load(f.read_text()) or {}).get("baselines", []):
            table[b["fuser"]] = b
    return table


# -------------------------------------------------------------------- built-in fusers
def _points(frames: Sequence[DepthFrame], stride: int = 2) -> np.ndarray:
    pts = [
        backproject(f.depth[::stride, ::stride], _strided_K(f.K, stride), f.T_wc) for f in frames
    ]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _strided_K(K: np.ndarray, s: int) -> np.ndarray:
    K2 = K.copy().astype(float)
    K2[0, :] /= s
    K2[1, :] /= s
    return K2


def builtin_naive_union(frames, cfg) -> FusionResult:
    t = time.perf_counter()
    return FusionResult(
        _points(frames).astype(np.float32),
        runtime_s=time.perf_counter() - t,
        meta={"builtin": True},
    )


def builtin_oracle_exclusion(frames, cfg) -> FusionResult:
    t = time.perf_counter()
    bad = set(cfg.get("oracle", {}).get("faulty_sensors", []))
    keep = [f for f in frames if f.sensor_id not in bad]
    ids = sorted({f.sensor_id for f in frames})
    return FusionResult(
        _points(keep).astype(np.float32),
        excluded_sensors={i: i in bad for i in ids},
        sensor_scores={i: float(i in bad) for i in ids},
        runtime_s=time.perf_counter() - t,
        meta={"builtin": True, "oracle": True},
    )


BUILTINS = {
    "builtin_naive_union": builtin_naive_union,
    "builtin_oracle_exclusion": builtin_oracle_exclusion,
}


class _Callable:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def fuse(self, frames, cfg):
        return self.fn(frames, cfg)


def resolve_fusers(names: Sequence[str], baseline_table: Dict[str, dict], include_builtin: bool):
    """Returns (fusers: {name: (obj, params)}, skipped: {name: reason})."""
    registry_ok, list_fn, get_fn = True, None, None
    try:
        import sensorfault.baselines  # noqa: F401  (registers fusers; may be absent/partial)
        from sensorfault.baselines.base import get_fuser, list_fusers

        list_fn, get_fn = list_fusers, get_fuser
    except Exception as e:
        registry_ok = False
        reg_reason = f"baselines package not importable: {type(e).__name__}: {e}"
    fusers, skipped = {}, {}
    known = set(list_fn()) if registry_ok else set()
    for n in names:
        if not registry_ok:
            skipped[n] = reg_reason
        elif n not in known:
            skipped[n] = "not registered in fuser registry (stub or not yet implemented)"
        else:
            try:
                from sensorfault.baselines.base import _REGISTRY

                ok, why = _REGISTRY[n].available()
                if not ok:
                    skipped[n] = f"unavailable: {why}"
                    continue
                fusers[n] = (get_fn(n), dict(baseline_table.get(n, {}).get("params", {})))
            except Exception as e:
                skipped[n] = f"construction failed: {type(e).__name__}: {e}"
    if include_builtin:
        for n, fn in BUILTINS.items():
            fusers[n] = (_Callable(n, fn), {})
    return fusers, skipped


# ------------------------------------------------------------------------ scenarios
def build_scenarios(cfg: dict) -> List[dict]:
    sc = [copy.deepcopy(s) for s in cfg.get("scenarios", [])]
    sg = cfg.get("scenario_grid")
    if sg:
        grid_path = Path(cfg["_config_path"]).parent / sg.get("grid", "faults/grid.yaml")
        grid = yaml.safe_load(grid_path.read_text())
        for cell in expand_cells(grid, sg.get("kinds"), sg.get("limit_per_kind")):
            for fr in sg.get("fractions", [{"faulty": 1, "total": 4}]):
                sc.append(
                    {
                        "name": f"{cell['kind']}:{json.dumps(cell['params'], sort_keys=True)}:{cell['profile']}"
                        f":{fr['faulty']}of{fr['total']}",
                        "cell": cell,
                        "n_sensors": fr["total"],
                        "faulty": list(range(fr["faulty"])),
                        "same_direction": fr.get("direction", "same") != "independent",
                    }
                )
    return sc


def scenario_fault_specs(scn: dict, sensor_ids: List[str]) -> List[FaultSpec]:
    onset = float(scn.get("onset_frac", 0.2))
    if "cell" in scn:
        return scenario_specs(
            scn["cell"],
            sensor_ids,
            scn.get("faulty", [0]),
            onset_frac=onset,
            same_direction=scn.get("same_direction", True),
        )
    specs = []
    for j, f in enumerate(scn.get("faults", [])):
        specs.append(
            FaultSpec(
                f["kind"],
                sensor_ids[int(f["sensor"])],
                f.get("fault_id", j),
                f.get("params", {}),
                f.get("onset_frac", onset),
                f.get("profile", "step"),
            )
        )
    return specs


def apply_dry_run(cfg: dict) -> dict:
    d = cfg.get("dry_run", {})
    cfg = copy.deepcopy(cfg)
    cfg["scenes"] = {**cfg.get("scenes", {}), "source": "synthetic"}
    cfg["synthetic"] = deep_merge(cfg.get("synthetic", {}), d.get("synthetic", {}))
    cfg["_dry_n_scenes"], cfg["_dry_n_seeds"] = d.get("n_scenes", 1), d.get("n_seeds", 1)
    cfg["_dry_max_scenarios"] = d.get("max_scenarios", 2)
    cfg["bootstrap_n"] = 200
    return cfg


# ---------------------------------------------------------------------------- main
def _git(*args) -> Optional[str]:
    try:
        return (
            subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def run_experiment(
    cfg: dict,
    split: str,
    dry_run: bool,
    out_root: Path,
    config_dir: Path,
    allow_untagged: bool = False,
) -> Path:
    if (
        split == "test"
        and not dry_run
        and not allow_untagged
        and not _git("tag", "-l", "prereg-v4")
    ):
        raise SystemExit(
            "refusing --split test: git tag prereg-v4 not found (see prereg/TAGGING.md)"
        )
    if dry_run:
        cfg = apply_dry_run(cfg)
        split = "dev"
    if cfg.get("scenes", {}).get("source", "synthetic") != "synthetic":
        raise NotImplementedError(
            "dataset-backed scenes are loaded via the data scripts; use source: synthetic"
        )
    exp = cfg.get("id", Path(cfg["_config_path"]).stem)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = out_root / exp / stamp
    (out / "fault_manifests").mkdir(parents=True, exist_ok=True)

    table = load_baseline_table(config_dir)
    fusers, skipped = resolve_fusers(
        cfg.get("baselines", []), table, cfg.get("include_builtin_reference", True)
    )
    for n, why in skipped.items():
        log.warning("skip fuser %s: %s", n, why)
    scenes = list(cfg["scenes"].get(split, []))
    seeds = list(cfg["seeds"]["master"])
    scenarios = build_scenarios(cfg) or [{"name": "control", "faulty": [], "faults": []}]
    if dry_run:
        scenes, seeds = scenes[: cfg["_dry_n_scenes"]], seeds[: cfg["_dry_n_seeds"]]
        scenarios = scenarios[: cfg["_dry_max_scenarios"]]
    taus = cfg["metrics"]["fscore_thresholds_m"]
    rows: List[dict] = []
    for scn in scenarios:
        syn = SyntheticConfig.from_dict(
            deep_merge(
                cfg.get("synthetic", {}),
                {
                    **scn.get("synthetic", {}),
                    **({"n_sensors": scn["n_sensors"]} if "n_sensors" in scn else {}),
                },
            )
        )
        for scene in scenes:
            data = generate_synthetic(syn, int(scene))
            for ms in seeds:
                specs = scenario_fault_specs(scn, data.sensor_ids)
                frames, man = inject_faults(data.frames, specs, ms)
                man.update({"scenario": scn["name"], "scene": scene})
                tag = hashlib.sha1(f"{scn['name']}|{scene}|{ms}".encode()).hexdigest()[:12]
                write_manifest(man, out / "fault_manifests" / f"{tag}.json")
                truth = {s: s in man["faulty_sensors"] for s in data.sensor_ids}
                for name, (fuser, params) in fusers.items():
                    fcfg = {
                        **cfg.get("fuser_defaults", {}),
                        **params,
                        "oracle": {"faulty_sensors": man["faulty_sensors"]},
                        "seed": int(ms),
                    }
                    row = {
                        "scenario": scn["name"],
                        "scene": scene,
                        "master_seed": ms,
                        "fuser": name,
                        "manifest": f"fault_manifests/{tag}.json",
                        "truth_faulty": truth,
                    }
                    try:
                        t0 = time.perf_counter()
                        res = fuser.fuse(frames, fcfg)
                        wall = time.perf_counter() - t0
                        pts = np.asarray(res.points)
                        if len(pts) > 60000:
                            pts = pts[
                                np.random.default_rng(0).choice(len(pts), 60000, replace=False)
                            ]
                        row.update(
                            metrics=pointcloud_metrics(pts, data.gt_points, taus),
                            excluded=dict(res.excluded_sensors),
                            scores=dict(res.sensor_scores),
                            runtime_s=float(res.runtime_s or wall),
                            wall_s=wall,
                        )
                        if res.excluded_sensors:
                            row["detection"] = detection_pr(truth, res.excluded_sensors)
                            row["alarms_on_clean_sensors"] = int(
                                sum(
                                    1 for s, e in res.excluded_sensors.items() if e and not truth[s]
                                )
                            )
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {e}"
                        log.error("fuser %s failed: %s", name, row["error"])
                    rows.append(row)
    summary = summarise(rows, cfg.get("bootstrap_n", 5000))
    results = {
        "experiment": exp,
        "status": cfg.get("status"),
        "split": split,
        "dry_run": dry_run,
        "hypothesis": cfg.get("hypothesis", {}),
        "fusers_run": list(fusers),
        "fusers_skipped": skipped,
        "n_rows": len(rows),
        "summary": summary,
        "rows": rows,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "config_path": cfg["_config_path"],
                "config_sha256": hashlib.sha256(
                    json.dumps(cfg_clean, sort_keys=True, default=str).encode()
                ).hexdigest(),
                "git_commit": _git("rev-parse", "HEAD"),
                "prereg_tag_present": bool(_git("tag", "-l", "prereg-v4")),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "split": split,
                "dry_run": dry_run,
                "scenes": scenes,
                "master_seeds": seeds,
                "fusers_run": list(fusers),
                "fusers_skipped": skipped,
                "config": cfg_clean,
            },
            indent=1,
            default=str,
        )
    )
    log.info("wrote %s (%d rows)", out, len(rows))
    return out


def summarise(rows: List[dict], n_boot: int) -> List[dict]:
    """Per (scenario, fuser): scene-level bootstrap CI of Chamfer (seeds averaged within scene)."""
    out = []
    keys = sorted({(r["scenario"], r["fuser"]) for r in rows})
    for scn, fz in keys:
        rs = [r for r in rows if r["scenario"] == scn and r["fuser"] == fz and "metrics" in r]
        by_scene: Dict[Any, List[float]] = {}
        for r in rs:
            by_scene.setdefault(r["scene"], []).append(r["metrics"]["chamfer"])
        per_scene = [float(np.mean(v)) for v in by_scene.values()]
        ent = {
            "scenario": scn,
            "fuser": fz,
            "n_runs": len(rs),
            "n_errors": sum(
                1 for r in rows if r["scenario"] == scn and r["fuser"] == fz and "error" in r
            ),
        }
        if per_scene:
            ent["chamfer"] = bootstrap_ci(per_scene, n_boot=n_boot)
            ent["runtime_s_median"] = float(np.median([r["runtime_s"] for r in rs]))
        out.append(ent)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny synthetic run, dev split, built-ins only needed",
    )
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--allow-untagged", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    cfg = load_config(a.config, a.overlay)
    out = run_experiment(
        cfg, a.split, a.dry_run, Path(a.out), Path(a.config).resolve().parent, a.allow_untagged
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
