"""Stage-1 training CLI. Usage:
  python scripts/stage1_train.py --config configs/stage1_sdxl_lora.json --workdir runs/stage1 [--set key=value ...] [--max-steps N] [--resume auto|PATH|none]
Resumes from the newest checkpoint in --workdir by default; SIGTERM/SIGINT write a checkpoint and exit cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dcal.train.loop import train  # noqa: E402


def expand(o):
    """Expand ${VAR} in every string of the config (data roots, model path)."""
    if isinstance(o, str):
        r = os.path.expandvars(o)
        if "${" in r:
            raise SystemExit(f"environment variable not set in config value {o!r}")
        return r
    if isinstance(o, list):
        return [expand(x) for x in o]
    if isinstance(o, dict):
        return {k: expand(v) for k, v in o.items()}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (JSON values), e.g. total_steps=300 datasets=[...]")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--stop-at", type=int, default=None, help="interrupt after this step (checkpoint written); used to test resume")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    cfg = expand(cfg)
    print(json.dumps({"config": cfg}), flush=True)
    out = train(cfg, a.workdir, resume=None if a.resume == "none" else a.resume, max_steps=a.max_steps, stop_at=a.stop_at)
    print(json.dumps({"done": out}), flush=True)


if __name__ == "__main__":
    main()
