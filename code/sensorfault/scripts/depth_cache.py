#!/usr/bin/env python3
"""Fill the monocular depth cache. Example:

    python scripts/depth_cache.py --dataset matrixcity --scene block3 \
        --models depth_pro,depth_anything_v2 --device cuda --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sensorfault.depthcache.datasets import DatasetUnavailableError  # noqa: E402
from sensorfault.depthcache.run import MODEL_IDS, default_builder, run  # noqa: E402


def main(argv=None) -> int:
    cfg_path = REPO / "configs" / "depth_cache.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["matrixcity", "usegeo", "eth3d"])
    p.add_argument("--scene", required=True)
    p.add_argument(
        "--models",
        default=",".join(cfg.get("models", MODEL_IDS)),
        help="comma-separated: " + ", ".join(MODEL_IDS),
    )
    p.add_argument("--device", default=cfg.get("device", "cuda"))
    p.add_argument("--limit", type=int, default=None, help="first N views only")
    p.add_argument("--root", default=cfg.get("root", "data/depth_cache"))
    p.add_argument("--dtype", default=cfg.get("dtype", "float16"), choices=["float16", "float32"])
    p.add_argument("--data-root", default=None, help="dataset root (default $SENSORFAULT_DATA)")
    p.add_argument("--transforms", default=None, help="MatrixCity transforms.json override")
    p.add_argument("--images", default=None, help="MatrixCity RGB directory override")
    p.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="MODEL.KEY=VALUE",
        help="wrapper build() option, e.g. metric3d_v2.max_depth_m=1000 (repeatable; adds to "
        "model_options in configs/depth_cache.yaml)",
    )
    p.add_argument("--dry-run", action="store_true", help="list work, load no model")
    a = p.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    kw = {
        k: v
        for k, v in [("data_root", a.data_root), ("transforms", a.transforms), ("images", a.images)]
        if v
    }
    opts = {m: dict(o) for m, o in (cfg.get("model_options") or {}).items()}
    try:
        for item in a.opt:
            key, sep, val = item.partition("=")
            model, dot, name = key.partition(".")
            if not (sep and dot and name):
                raise ValueError(f"--opt expects MODEL.KEY=VALUE, got {item!r}")
            opts.setdefault(model, {})[name] = yaml.safe_load(val)
        run(
            a.dataset,
            a.scene,
            models,
            a.root,
            a.device,
            a.limit,
            a.dry_run,
            a.dtype,
            builder=lambda m: default_builder(m, **opts.get(m, {})),
            **kw,
        )
    except (DatasetUnavailableError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
