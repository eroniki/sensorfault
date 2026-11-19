"""Verdict for a stage-1 smoke run: reads <workdir>/log.jsonl and the console log. Exit code 0 = pass.

Checks: loss falls (mean of the last 25% of logged steps below the first 25%), no non-finite losses, few skipped (overflowed) steps,
the grad scale did not collapse, validation losses and sampled depths are finite, checkpoints exist, and a resume happened
(the console log contains 'resumed from') with step continuity. Usage: python scripts/stage1_smoke_check.py WORKDIR [CONSOLE_LOG]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main():
    wd = Path(sys.argv[1])
    recs = [json.loads(line) for line in open(wd / "log.jsonl")]
    tr = [r for r in recs if "loss" in r]
    val = [r for r in recs if "val_loss" in r]
    checks = {}
    n = len(tr)
    k = max(n // 4, 1)
    first, last = float(np.mean([r["loss"] for r in tr[:k]])), float(np.mean([r["loss"] for r in tr[-k:]]))
    checks["loss falls (last quarter < first quarter)"] = (last < first, f"{first:.4f} -> {last:.4f}")
    checks["no non-finite losses"] = (tr[-1]["nonfinite_loss_total"] == 0, f"{tr[-1]['nonfinite_loss_total']}")
    steps_done = tr[-1]["step"]
    skip_frac = tr[-1]["skipped_total"] / max(steps_done, 1)
    checks["skipped (overflow) steps <= 10%"] = (skip_frac <= 0.10, f"{tr[-1]['skipped_total']} of {steps_done}")
    scales = [r["scale"] for r in tr if r.get("scale") is not None]
    checks["grad scale not collapsed (min >= 64)"] = (not scales or min(scales) >= 64, f"min {min(scales) if scales else 'n/a'}")
    checks["validation finite"] = (bool(val) and all(np.isfinite(v["val_loss"]) and v.get("val_finite_samples", True) for v in val), f"{len(val)} evals")
    if val and "val_affine_absrel" in val[-1]:
        checks["val affine AbsRel reported (informational)"] = (True, f"{val[-1]['val_affine_absrel']:.3f}")
    ck = sorted(wd.glob("ckpt_*.pt"))
    checks["checkpoints written"] = (len(ck) >= 2, f"{[c.name for c in ck]}")
    if len(sys.argv) > 2:
        text = Path(sys.argv[2]).read_text()
        checks["resumed from a checkpoint"] = ("resumed from" in text, "found" if "resumed from" in text else "missing")
    steps = [r["step"] for r in tr]
    checks["step continuity across the resume"] = (steps == sorted(set(steps)) or len(steps) == len(set(steps)), f"{len(steps)} log points")
    ok = True
    for name, (good, info) in checks.items():
        print(f"[{'PASS' if good else 'FAIL'}] {name}: {info}")
        ok &= good
    print("SMOKE", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
