"""Turn stage-0 grid results (scripts/stage0_grid.sh -> results.jsonl) into a markdown table.

  python scripts/stage0_table.py <results.jsonl> [--gpu-mib 24576] [--frac 0.8]

"Fits" compares the device-level peak (nvidia-smi, includes CUDA context and allocator cache) with frac x GPU memory.
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--gpu-mib", type=float, default=24576)
    ap.add_argument("--frac", type=float, default=0.8)
    a = ap.parse_args()
    budget = a.frac * a.gpu_mib * 1048576 / 1e9
    rows = [json.loads(line) for line in open(a.results) if line.strip()]
    print(f"| res | batch | rank | ckpt | precision | trainable params | peak alloc GB | peak device GB | ms/step | fits {a.frac:.0%} ({budget:.1f} GB) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            args = r["args"].split()
            kv = {args[i].lstrip("-"): args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--") and not args[i + 1].startswith("--")}
            ck = "yes" if "--ckpt" in args else "no"
            prec = ("fp32" if "--fp32" in args else "fp16 autocast") + (", base fp16" if "--base-fp16" in args else "")
            print(f"| {kv.get('res')} | {kv.get('batch')} | {kv.get('rank')} | {ck} | {prec} | - | - | {r['smi_peak_GB']:.1f} | {r['error']} | no |")
            continue
        prec = ("fp16 autocast" if r["fp16"] else "fp32") + (", base fp16" if r.get("base_fp16") else "")
        fits = "yes" if r["smi_peak_GB"] <= budget else "no"
        print(f"| {r['res']} | {r['batch']} | {r['rank']} | {'yes' if r['ckpt'] else 'no'} | {prec} | {r['params_trainable'] / 1e6:.1f} M"
              f" | {r['peak_mem_GB']:.2f} | {r['smi_peak_GB']:.2f} | {r['ms_per_step_median']:.0f} | {fits} |")


if __name__ == "__main__":
    main()
