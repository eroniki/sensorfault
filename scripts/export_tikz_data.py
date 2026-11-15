"""Export the numbers behind figures/ as small CSV files for PGFPlots (figures/tikz/data/). CPU only, numpy + scipy.

Reads the saved results (docs/*.json, docs/toy-followups.md, figures/data/*.json) and, for the toy residual-map panels, re-runs the cheap
toy code with the same seeds as scripts/make_renders.py (seconds). Nothing here trains or fits a model.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "figures" / "tikz" / "data"
D.mkdir(parents=True, exist_ok=True)
VARS = ["known", "oracle", "kernel", "kernel_nr", "shrunk"]


def w(name, header, rows):
    with open(D / name, "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(header)
        c.writerows(rows)


def mdf(curve):
    ks = sorted(curve)
    p = [curve[k] for k in ks]
    for i, pi in enumerate(p):
        if pi >= 0.8:
            if i == 0:
                return ks[0], 0
            k0, k1, p0, p1 = ks[i - 1], ks[i], p[i - 1], pi
            return k0 + (0.8 - p0) * (k1 - k0) / max(p1 - p0, 1e-9), 0
    return ks[-1], 1


def main():
    pipes = json.load(open(ROOT / "docs" / "pipes-2dgs-fit.json"))
    toy = json.load(open(ROOT / "docs" / "toy-results.json"))
    guards = json.load(open(ROOT / "figures" / "data" / "guards.json"))
    n1 = toy["meta"]["n1"]

    # fig1
    rows = []
    for v in ("2", "6", "10"):
        b, a = pipes[v]["before"], pipes[v]["after"]
        rows.append([v, 100 * b["coverage_of_valid_gt"], 100 * a["coverage_of_valid_gt"], 100 * b["rel_err_median"], 100 * a["rel_err_median"],
                     100 * b["rel_err_p90"], 100 * a["rel_err_p90"], b["psnr_db"], a["psnr_db"]])
    w("fig1_pipes.csv", ["view", "cov_b", "cov_a", "med_b", "med_a", "p90_b", "p90_a", "psnr_b", "psnr_a"], rows)

    # fig2
    rows = []
    for key, name in (("toy point surface (CPU, synthetic room)", "toy"), ("gsplat 2DGS surface (GPU, synthetic room)", "gs"),
                      ("deliberately leaky fit (must fail)", "leaky")):
        rs = [g for g in guards if g.get("pipeline") == key and "poisoned" in g]
        rows.append([name, len(rs), sum(g["poisoned"] for g in rs), sum(g["invariant"] for g in rs), sum(g["sensitive"] for g in rs)])
    w("fig2_matrix.csv", ["pipeline", "n", "poisoned_ok", "invariant_ok", "sensitive_ok"], rows)
    rows = []
    for i, v in enumerate(("2", "6", "10")):
        p, c = pipes[v]["perturbed_j_abs_diff"], pipes[v]["run_to_run_noise_abs_diff"]
        rows.append([v, p["median"], p["p99"], c["median"], c["p99"]])
    w("fig2_floor.csv", ["view", "pert_med", "pert_p99", "ctrl_med", "ctrl_p99"], rows)

    # fig3 / fig4
    ks = sorted(float(k) for k in toy["power"]["square"])
    for fault in ("square", "warp", "edge"):
        rows = []
        for k in ks:
            r = [k]
            for v in VARS:
                p = toy["power"][fault][str(k)][v]
                r += [p, float(np.sqrt(max(p * (1 - p), 1e-4) / n1))]
            rows.append(r)
        w(f"fig3_{fault}.csv", ["kappa"] + [f"{v}{s}" for v in VARS for s in ("", "_se")], rows)
    rows = []
    for vi, v in enumerate(VARS):
        for fi, fault in enumerate(("square", "warp", "edge")):
            m, c = mdf({float(k): toy["power"][fault][k][v] for k in toy["power"][fault]})
            rows.append([v, vi, fault, fi, m, c])
    w("fig4_mdf.csv", ["variant", "vi", "fault", "fi", "mdf", "censored"], rows)
    for fi, fault in enumerate(("square", "warp", "edge")):
        rows = [[vi, m, c] for vi, (v, m, c) in enumerate([(v,) + mdf({float(k): toy["power"][fault][k][v] for k in toy["power"][fault]}) for v in VARS])]
        w(f"fig4_mdf_{fault}.csv", ["vi", "mdf", "censored"], rows)
    wide = []
    for fi, fault in enumerate(("square", "warp", "edge")):
        row = [fault, fi]
        labs = []
        for v in VARS:
            m, c = mdf({float(k): toy["power"][fault][k][v] for k in toy["power"][fault]})
            row += [m, c]
            labs.append(f">{m:g}" if c else f"{m:.2f}")
        wide.append(row + labs)
    w("fig4_mdf_wide.csv", ["fault", "fi"] + [f"{v}{s}" for v in VARS for s in ("", "_c")] + [f"{v}_lab" for v in VARS], wide)
    w("fig4_null.csv", ["vi", "variant", "far", "t_over_dof"], [[i, v, toy["null"][v]["far_at_chi2"], toy["null"][v]["mean_T_over_dof"]] for i, v in enumerate(VARS)])

    # fig5
    from make_figures import parse_followups
    miscal, area = parse_followups(ROOT / "docs" / "toy-followups.md")
    sw = toy["area_sweep_kappa2.5"]
    w("fig5_area_a.csv", ["area"] + VARS, [[float(a)] + [sw[a][v] for v in VARS] for a in sorted(sw, key=float)])
    for kappa in (2.5, 4.0):
        w(f"fig5_area_b_k{kappa:g}.csv", ["area", "oracle", "kernel", "kernel_nr"],
          [[a, area[kappa][a]["oracle"], area[kappa][a]["kernel"], area[kappa][a]["kernel_nr"]] for a in sorted(area[kappa])])
    w("fig5_global.csv", ["vi", "variant", "shift", "affine"], [[i, v, toy["shift_kappa8"][v], toy["affine_kappa8"][v]] for i, v in enumerate(VARS)])

    # fig6
    for v in ("oracle", "kernel", "kernel_nr"):
        rows = []
        for r in sorted((r for r in miscal if r["variant"] == v), key=lambda r: r["miscal"]):
            ms, cs = (float(r["mdf_square"].lstrip(">")), int(r["mdf_square"].startswith(">")))
            mw, cw = (float(r["mdf_warp"].lstrip(">")), int(r["mdf_warp"].startswith(">")))
            rows.append([r["miscal"], r["T_over_dof"], r["far_chi2"], ms, cs, mw, cw])
        w(f"fig6_{v}.csv", ["miscal", "t_over_dof", "far", "mdf_sq", "cens_sq", "mdf_warp", "cens_warp"], rows)

    # toy residual maps (same seeds as scripts/make_renders.py toy)
    from dcal.stat import align_samples, empirical_threshold, fault_statistic
    from dcal.toy import ToyConfig, World

    wd = World(ToyConfig(g=16), 0)
    g = wd.cfg.g
    kappa, seed = 3.0, 7

    def run(sd, fault, variant):
        S, Dd, Kr, Ku, _ = wd.simulate(np.random.default_rng(sd), fault, kappa if fault else 0.0, 0.12)
        kw = dict(Sigma_u=Ku) if variant == "oracle" else dict(grid_shape=wd.shape, cov="kernel")
        res = fault_statistic(S, Dd, Kr, **kw)
        r = res["s"] * align_samples(S).mean(0) + res["t"] - Dd
        return res, r

    thr = {v: empirical_threshold(np.array([run(10_000 + i, None, v)[0]["T"] for i in range(150)]), 0.05) for v in ("oracle", "kernel")}
    stats = []
    for fault in ("square", "warp", "edge"):
        pat = wd.fault_pattern(fault, 0.12).reshape(g, g)
        mask = (np.abs(pat) > 0.5) if fault == "warp" else (pat > 0.5)
        segs = []                     # outline of the fault support (cell edges between inside and outside)
        for i in range(g):
            for j in range(g):
                if mask[i, j]:
                    if i == 0 or not mask[i - 1, j]:
                        segs.append((j, g - i, j + 1, g - i))
                    if i == g - 1 or not mask[i + 1, j]:
                        segs.append((j, g - i - 1, j + 1, g - i - 1))
                    if j == 0 or not mask[i, j - 1]:
                        segs.append((j, g - i - 1, j, g - i))
                    if j == g - 1 or not mask[i, j + 1]:
                        segs.append((j + 1, g - i - 1, j + 1, g - i))
        with open(D / f"toy_outline_{fault}.csv", "w") as f:
            f.write("x y\n")
            for s_ in segs:
                f.write(f"{s_[0]} {s_[1]}\n{s_[2]} {s_[3]}\nnan nan\n")
        for variant in ("oracle", "kernel"):
            rc, r_c = run(seed, None, variant)
            rf, r_f = run(seed, fault, variant)
            for state, res, r in (("clean", rc, r_c), ("fault", rf, r_f)):
                for kind, arr in (("r", r), ("e", res["e"])):
                    a = arr.reshape(g, g)
                    with open(D / f"toy_{fault}_{variant}_{state}_{kind}.csv", "w") as f:
                        f.write("x y z\n")
                        for i in range(g):
                            for j in range(g):
                                f.write(f"{j} {g - 1 - i} {a[i, j]:.5f}\n")
                stats.append([fault, variant, state, res["T"], thr[variant], int(res["T"] > thr[variant])])
    w("toy_render_stats.csv", ["fault", "variant", "state", "T", "thr", "flagged"], stats)
    with open(D / "toy_render_macros.tex", "w") as f:
        f.write("% generated by scripts/export_tikz_data.py: T, threshold and the symmetric colour limits of the toy residual-map panels\n")
        for fault, variant, state, T, thr_, fl in stats:
            f.write(f"\\expandafter\\def\\csname ToyT@{fault}@{variant}@{state}\\endcsname{{{T:.0f}}}%\n")
            f.write(f"\\expandafter\\def\\csname ToyThr@{fault}@{variant}@{state}\\endcsname{{{thr_:.0f}}}%\n")
            f.write(f"\\expandafter\\def\\csname ToyFlag@{fault}@{variant}@{state}\\endcsname{{{'flagged' if fl else 'not flagged'}}}%\n")
        for fault in ("square", "warp", "edge"):
            for variant in ("oracle", "kernel"):
                vr = max(np.abs(np.loadtxt(D / f"toy_{fault}_{variant}_{st}_r.csv", skiprows=1)[:, 2]).max() for st in ("clean", "fault"))
                ve = max(np.abs(np.loadtxt(D / f"toy_{fault}_{variant}_{st}_e.csv", skiprows=1)[:, 2]).max() for st in ("clean", "fault"))
                f.write(f"\\expandafter\\def\\csname ToyVr@{fault}@{variant}\\endcsname{{{vr:.3f}}}%\n")
                f.write(f"\\expandafter\\def\\csname ToyVe@{fault}@{variant}\\endcsname{{{ve:.3f}}}%\n")
    print("wrote", len(list(D.glob("*.csv"))), "csv files to", D)


if __name__ == "__main__":
    main()
