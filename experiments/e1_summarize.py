"""Summarise experiment E1 (results/e1/results.json) into tables and verdicts against the predictions.

Only predictions that this synthetic, linearised run can address are judged; the rest are listed as
not addressed. Run: python experiments/e1_summarize.py results/e1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main(path: str) -> None:
    root = Path(path)
    R = json.loads((root / "clean.json").read_text())
    rt = R["runtime_s"]
    for stage, key in (("grid", "single_view_grid"), ("repeat", "repeated")):
        f = root / f"{stage}.json"
        if f.exists():
            d = json.loads(f.read_text())
            R[key] = d[key]
            rt += d["runtime_s"]
        else:
            R[key] = []
    R["runtime_s"] = round(rt, 1)
    n = R["info"]["n_views"]
    lines = [
        f"# E1 summary: {R['config']['block']}",
        "",
        f"Views: {n}; surfels: {R['info']['n_surfels']}; runtime {R['runtime_s']} s.",
        "",
    ]
    S = {}

    # clean
    lines += [
        "## Clean scenes",
        "",
        "| noise scale | temperature | per-view FA (%) | mean T/m | loop: seeds with any exclusion / seeds | untestable views (%) |",
        "|---|---|---|---|---|---|",
    ]
    for scale, rows in R["clean"].items():
        for temp in (True, False):
            rr = [r for r in rows if r["temperature"] == temp]
            fa = float(np.mean([r["per_view_fa"] for r in rr]))
            tm = float(np.mean([r["mean_T_over_m"] for r in rr]))
            fw = sum(r["loop_excluded"] > 0 for r in rr)
            ut = float(np.mean([r["untestable"] for r in rr]))
            S[f"clean_{scale}_{temp}"] = {"fa": fa, "tm": tm, "fw": fw, "n": len(rr), "ut": ut}
            lines.append(
                f"| {scale} | {'on' if temp else 'off'} | {pct(fa)} | {tm:.3f} | {fw} / {len(rr)} | {pct(ut)} |"
            )
    lines.append("")

    # single-view grid
    G = R["single_view_grid"]
    if not G:
        G = [{"delta": 0, "area": 0, "gate_detect_matched": False, "rms_detect_matched": False, "loop_detect": False, "loop_false": 0, "clean_flag_matched": 0.0}]
    deltas = sorted({g["delta"] for g in G})
    areas = sorted({g["area"] for g in G})
    lines += [
        "## One faulty view: detection at matched 5% per-view false alarm (single pass)",
        "",
        "Gate / aligned RMS detection (%) per cell; rows: magnitude (sigma), columns: area fraction.",
        "",
        "| delta \\ area | " + " | ".join(str(a) for a in areas) + " |",
        "|---|" + "---|" * len(areas),
    ]
    cells = {}
    for d in deltas:
        row = []
        for a in areas:
            c = [g for g in G if g["delta"] == d and g["area"] == a]
            gd = float(np.mean([g["gate_detect_matched"] for g in c]))
            rd = float(np.mean([g["rms_detect_matched"] for g in c]))
            cells[(d, a)] = (gd, rd, len(c))
            row.append(f"{pct(gd)} / {pct(rd)}")
        lines.append(f"| {d} | " + " | ".join(row) + " |")
    lines += [
        "",
        "Outer loop with Holm across rounds: faulty view excluded (%) and clean views excluded per scene.",
        "",
        "| delta \\ area | " + " | ".join(str(a) for a in areas) + " |",
        "|---|" + "---|" * len(areas),
    ]
    for d in deltas:
        row = []
        for a in areas:
            c = [g for g in G if g["delta"] == d and g["area"] == a]
            row.append(
                f"{pct(np.mean([g['loop_detect'] for g in c]))} ({np.mean([g['loop_false'] for g in c]):.2f})"
            )
        lines.append(f"| {d} | " + " | ".join(row) + " |")
    allg = len(G)
    lines.append("")
    gate_all = float(np.mean([g["gate_detect_matched"] for g in G]))
    rms_all = float(np.mean([g["rms_detect_matched"] for g in G]))
    S["grid"] = {
        "gate": gate_all,
        "rms": rms_all,
        "n": allg,
        "loop_false_mean": float(np.mean([g["loop_false"] for g in G])),
        "loop_detect": float(np.mean([g["loop_detect"] for g in G])),
        "clean_flag_matched": float(np.mean([g["clean_flag_matched"] for g in G])),
    }

    # repeated
    P = R["repeated"]
    lines += [
        "## Same error on three neighbouring frames of a strip (8 sigma, 25 m patch)",
        "",
        "| noise scale | B | temperature | loop | whitened | faulty views detected (%) | clean views excluded per scene | Holm alarm (cap) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    combos = sorted(
        {(r["noise_scale"], r["B"], r["temperature"], r["round_cap"], r["whiten"]) for r in P}
    )
    for key in combos:
        rr = [
            r
            for r in P
            if (r["noise_scale"], r["B"], r["temperature"], r["round_cap"], r["whiten"]) == key
        ]
        det = float(np.mean([r["detected"] / len(r["faulty"]) for r in rr]))
        fal = float(np.mean([r["false"] for r in rr]))
        cap = sum(r["alarm"] == "round_cap_reached" for r in rr)
        S[f"rep_{key}"] = {"det": det, "false": fal, "n": len(rr)}
        lines.append(
            f"| {key[0]} | {key[1]} | {'on' if key[2] else 'off'} | {'on' if key[3] > 1 else 'off'} | {'yes' if key[4] else 'no'} | {pct(det)} | {fal:.2f} | {cap} |"
        )
    lines.append("")

    # verdicts
    def rep(scale, B, temp, cap, wh=True):
        return S.get(f"rep_({scale}, {B}, {temp}, {cap}, {wh})")

    v = ["## Against the predictions (only what this run can address)", ""]
    c0 = S["clean_1.0_True"]
    v.append(
        f"- Holm family-wise false alarm, clean scenes, nominal noise: {c0['fw']}/{c0['n']} seeds with any exclusion (alpha = 0.05)."
    )
    c13 = S["clean_1.3_True"]
    c13off = S["clean_1.3_False"]
    v.append(
        f"- Scene temperature with noise 1.3x the offline level (H6, 'no temperature raises clean FA'): per-view FA {pct(c13['fa'])}% with temperature vs {pct(c13off['fa'])}% without; loop exclusions in {c13['fw']}/{c13['n']} vs {c13off['fw']}/{c13off['n']} seeds."
    )
    v.append(
        f"- H5 ('raw T/m > 1.2 from independent blocks'): raw mean T/m = {c0['tm']:.3f} here. Not addressed: this run has one surfel per pixel and independent noise, the case in which the blocks are exact; H5 concerns real, correlated depth error."
    )
    big = [cells[(d, a)] for d in deltas for a in areas if d >= 4]
    gain = float(np.mean([g - r for g, r, _ in big])) if big else float("nan")
    v.append(
        f"- H3 (gate vs cheap baselines, >= 15 points): only the aligned-RMS baseline is run, on injected regional faults rather than specular masks. Mean gain over cells with delta >= 4: {100 * gain:.1f} points; overall {pct(gate_all)}% vs {pct(rms_all)}% at matched 5% per-view FA. Partial evidence only."
    )
    for scale in ("1.0", "1.3"):
        b3 = rep(float(scale), 3, True, 10)
        b1 = rep(float(scale), 1, True, 10)
        sp = rep(float(scale), 3, True, 1)
        if b3 and b1:
            v.append(
                f"- H6 leave-block-out (noise {scale}x): B=3 detects {pct(b3['det'])}% vs B=1 {pct(b1['det'])}% of repeated faulty views (prediction: >= 10 points) -> {'supported' if b3['det'] - b1['det'] >= 0.10 else 'not supported'} in this run."
            )
        if b3 and sp:
            v.append(
                f"- H6 outer loop (noise {scale}x, B=3): loop {pct(b3['det'])}% vs single pass {pct(sp['det'])}% (prediction: >= 10 points) -> {'supported' if b3['det'] - sp['det'] >= 0.10 else 'not supported'} in this run."
            )
    v.append(
        "- Not addressed by this run: H1, H2 (need a real depth model), H4 (needs training), H7 (pose check), and the remaining H6 ablations."
    )
    lines += v
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    (root / "summary.json").write_text(
        json.dumps(
            {k if isinstance(k, str) else str(k): val for k, val in S.items()},
            indent=1,
            default=float,
        )
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/e1")
