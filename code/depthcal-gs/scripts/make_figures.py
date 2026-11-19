"""Build the comparison figures and figures/summary.md from results already on disk (no new experiments).

Inputs : docs/pipes-2dgs-fit.json, docs/toy-results.json, docs/toy-followups.md, figures/data/guards.json
         (the last is written by scripts/run_guard_checks.py; the others by scripts/fit_2dgs_pipes.py, toy_power.py, toy_followups.py)
Outputs: figures/fig1..fig6 (.png and .pdf), figures/summary.md.
Usage  : python scripts/make_figures.py            (needs numpy + matplotlib; CPU only)
         python scripts/make_figures.py --paper    (fig3, fig4, fig6 only: narrower, no footer, to figures/paper/ for the manuscript)
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PAPER = False
SZ = {"fig3": (11, 3.9), "fig4": (11, 3.8), "fig6": (11, 3.7)}
PAPER_SZ = {"fig3": (7.2, 3.1), "fig4": (7.2, 3.0), "fig6": (7.2, 2.9)}
VARS = ["known", "oracle", "kernel", "kernel_nr", "shrunk"]
COL = {"known": "#6e6e6e", "oracle": "#1f77b4", "kernel": "#e07b00", "kernel_nr": "#2a9d4b", "shrunk": "#8e44ad"}
MK = {"known": "s", "oracle": "o", "kernel": "^", "kernel_nr": "v", "shrunk": "D"}
LAB = {"known": "known (true a,b, true cov.; unattainable)", "oracle": "oracle cov., fitted (s,t)",
       "kernel": "estimated kernel cov., robust fit", "kernel_nr": "estimated kernel cov., plain fit",
       "shrunk": "shrunk sample cov., robust fit"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 110})

FOOT_TOY = ("DATA: synthetic Gaussian toy (not real images), 16x16 grid, M=10 samples, 1 seed, 200 clean trials for the null and 80 per cell; "
            "threshold = empirical 95% quantile of the clean-view null of the same pipeline; power s.e. about +-0.05.")
FOOT_FOLLOW = ("DATA: synthetic Gaussian toy (not real images), 16x16 grid, M=10 samples, 1 seed, 150 clean trials and 60 per cell; "
               "empirical-null threshold of the same pipeline; power s.e. about +-0.06.")


def sup(fig, text):
    if not PAPER:
        fig.suptitle(text, fontsize=10, y=0.985)


def save(fig, name, footer, top=0.93):
    if PAPER:
        fig.tight_layout(rect=(0, 0, 1, 1))
        out = FIG / "paper"
        out.mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)
        return
    lines = textwrap.wrap(footer, 170)
    fig.text(0.01, 0.006, "\n".join(lines), fontsize=6.6, color="#444", va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.035 * len(lines) + 0.02, 1, top))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)


def mdf(curve: dict) -> tuple[float, bool]:
    """Interpolated 80%-power magnitude; (value, censored). Censored values are the largest tested magnitude."""
    ks = sorted(curve)
    p = [curve[k] for k in ks]
    for i, pi in enumerate(p):
        if pi >= 0.8:
            if i == 0:
                return ks[0], False
            k0, k1, p0, p1 = ks[i - 1], ks[i], p[i - 1], pi
            return k0 + (0.8 - p0) * (k1 - k0) / max(p1 - p0, 1e-9), False
    return ks[-1], True


def parse_followups(path):
    t = Path(path).read_text().splitlines()
    miscal, area = [], {}
    sec = kappa = None
    for ln in t:
        if ln.startswith("## 1."):
            sec = 1
        elif ln.startswith("## 2."):
            sec = 2
        m = re.match(r"kappa = ([\d.]+)", ln)
        if m:
            kappa = float(m.group(1))
        if ln.startswith("|") and not ln.startswith("|---"):
            c = [x.strip() for x in ln.strip("|").split("|")]
            if sec == 1 and re.match(r"^[\d.]+$", c[0]):
                miscal.append({"miscal": float(c[0]), "variant": c[1], "T_over_dof": float(c[2]), "far_chi2": float(c[3]),
                               "mdf_square": c[4], "mdf_warp": c[5]})
            elif sec == 2 and kappa and re.match(r"^[\d.]+$", c[0]):
                area.setdefault(kappa, {})[float(c[0])] = {v: float(x) for v, x in zip(["oracle", "kernel", "kernel_nr"], c[1:4])}
    return miscal, area


def mdf_num(s):
    return (float(s.lstrip(">")), s.startswith(">"))


# ---------------------------------------------------------------------------------------------------------------- fig 1
def fig1(pipes):
    views = ["2", "6", "10"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
    spec = [("coverage_of_valid_gt", "Held-out GT pixels covered by the render (alpha>0.5)", 100, "%"),
            ("rel_err_median", "Median relative depth error", 100, "%"),
            ("rel_err_p90", "90th-percentile relative depth error", 100, "%")]
    x = np.arange(3)
    for a, (key, title, scale, unit) in zip(ax, spec):
        b = [pipes[v]["before"][key] * scale for v in views]
        o = [pipes[v]["after"][key] * scale for v in views]
        r1 = a.bar(x - 0.2, b, 0.38, color="#bbbbbb", label="initialised surfels, NOT optimised (0 iterations)")
        r2 = a.bar(x + 0.2, o, 0.38, color="#1f77b4", label="optimised (2000 its, RGB+depth L1, no densification)")
        for r in list(r1) + list(r2):
            a.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():.2f}" if key != "coverage_of_valid_gt" else f"{r.get_height():.1f}",
                   ha="center", va="bottom", fontsize=7)
        a.set_xticks(x, [f"view {v}" for v in views])
        a.set_title(title, fontsize=8.5)
        a.set_ylabel(unit)
        if key == "coverage_of_valid_gt":
            a.set_ylim(70, 103)
        else:
            a.set_ylim(0, max(b) * 1.25)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, fontsize=7.5, frameon=False)
    sup(fig, "Leave-one-view-out 2DGS on ETH3D 'pipes': held-out view vs its laser-scan depth")
    save(fig, "fig1_pipes_2dgs", "DATA: REAL ETH3D 'pipes' scene (14 views, 1/8 scale 778x518; laser-scan depth, valid at about 9-10% of pixels), held-out views 2, 6, 10, "
         "1 run each. The surface is built from the OTHER 13 views' sparse GROUND-TRUTH depth (stand-in for model predictions): this tests the "
         "surface and leave-out machinery, not any depth model. 'before' = freshly initialised surfels, not optimised. Source: docs/pipes-2dgs-fit.json.", top=0.84)


# ---------------------------------------------------------------------------------------------------------------- fig 2
def fig2(guards, pipes):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    rows = [("toy point surface\n(CPU, synthetic room)", "toy point surface (CPU, synthetic room)", True),
            ("gsplat 2DGS surface\n(GPU, synthetic room)", "gsplat 2DGS surface (GPU, synthetic room)", True),
            ("deliberately leaky fit\n(mutation: must FAIL)", "deliberately leaky fit (must fail)", False)]
    cols = [("poisoned", "view j poisoned:\nnever touched"), ("invariant", "j perturbed:\nrender unchanged\n(bit-identical)"),
            ("sensitive", "other view perturbed:\nrender changes")]
    a.set_xlim(-0.6, 3)
    a.set_ylim(0, len(rows) + 1.6)
    a.axis("off")
    for ci, (_, ct) in enumerate(cols):
        a.text(ci + 0.5, len(rows) + 0.6, ct, ha="center", va="center", fontsize=6.8, fontweight="bold")
    for ri, (rt, key, expect_pass) in enumerate(rows):
        y = len(rows) - ri - 0.5
        a.text(-0.03, y, rt, ha="right", va="center", fontsize=8)
        rs = [g for g in guards if g.get("pipeline") == key and "poisoned" in g]
        for ci, (ck, _) in enumerate(cols):
            n_ok = sum(1 for g in rs if g[ck])
            good = (n_ok == len(rs)) if (expect_pass or ck == "sensitive") else (n_ok == 0)
            color = "#cfe8d5" if good else "#f3c6c6"
            a.add_patch(plt.Rectangle((ci + 0.03, y - 0.42), 0.94, 0.84, color=color))
            txt = f"{n_ok}/{len(rs)} pass" if (expect_pass or ck == "sensitive") else f"{len(rs) - n_ok}/{len(rs)} caught"
            a.text(ci + 0.5, y, txt, ha="center", va="center", fontsize=8.5)
    a.text(-0.55, len(rows) + 1.45, "Leave-view-out guard checks (green = as required; the leaky fit must be caught)", fontsize=8.5, ha="left", va="center")
    views = ["2", "6", "10"]
    x = np.arange(3)
    w = 0.2
    for i, (key, lab, c) in enumerate([("perturbed_j_abs_diff", "j's data perturbed (RGB, depth, pose)", "#d62728"),
                                       ("run_to_run_noise_abs_diff", "control: identical data re-run (noise floor)", "#7f7f7f")]):
        for j, (st, hatch) in enumerate([("median", None), ("p99", "//")]):
            vals = [pipes[v][key][st] for v in views]
            b.bar(x + (i * 2 + j - 1.5) * w, vals, w, color=c, hatch=hatch, alpha=0.9, label=f"{lab}, {st}" if True else None)
    b.set_yscale("log")
    b.set_xticks(x, [f"held-out view {v}" for v in views])
    b.set_ylabel("|depth render difference| (m)")
    b.set_title("REAL pipeline (ETH3D pipes, 300-it fits): poisoned runs untouched;\nperturbation effect = noise floor", fontsize=8)
    b.set_ylim(1.5e-5, 4e-1)
    b.legend(fontsize=6.0, loc="upper left", ncol=2)
    sup(fig, "Leave-out guards: synthetic checks (left) and the optimised 2DGS on real ETH3D data (right)")
    save(fig, "fig2_leave_out_guards", "DATA: left = synthetic room (5 views, toy point surface 24x24 and gsplat 2DGS 48x64), every view held out once, deliberately leaky fit on views 0/2/4; "
         "source figures/data/guards.json. Right = REAL ETH3D pipes, 3 held-out views, single runs; CUDA backward is non-deterministic, so 'unchanged' there means equal to the "
         "run-to-run floor; (max differences of 2-3 m from floaters occur in the control too). Source: docs/pipes-2dgs-fit.json.")


# ---------------------------------------------------------------------------------------------------------------- fig 3
def fig3(toy):
    fig, ax = plt.subplots(1, 3, figsize=PAPER_SZ["fig3"] if PAPER else SZ["fig3"], sharey=True)
    n1 = toy["meta"]["n1"]
    for a, fault in zip(ax, ["square", "warp", "edge"]):
        ks = sorted(float(k) for k in toy["power"][fault])
        for v in VARS:
            p = np.array([toy["power"][fault][str(k)][v] for k in ks])
            se = np.sqrt(np.maximum(p * (1 - p), 1e-4) / n1)
            a.errorbar(ks, p, yerr=se, color=COL[v], marker=MK[v], ms=4, lw=1.3, capsize=2, ls="--" if v == "known" else "-", label=LAB[v])
        a.axhline(0.8, color="k", lw=0.6, ls=":")
        a.axhline(0.05, color="k", lw=0.6, ls=":")
        a.set_title({"square": "square region (12% of the view)", "warp": "smooth warp (whole view)",
                     "edge": "edge-aligned region (12%)"}[fault], fontsize=9)
        a.set_xlabel("fault magnitude (sd units)")
    ax[0].set_ylabel("power at 5% false-alarm\n(empirical null)")
    ax[0].set_ylim(0, 1.5)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0].legend(fontsize=6.3, loc="upper center", ncol=1, frameon=False)
    sup(fig, "Toy statistic: power versus fault magnitude (dotted lines: 80% power and 5% level)")
    save(fig, "fig3_toy_power_vs_magnitude", FOOT_TOY + " Source: docs/toy-results.json.")


# ---------------------------------------------------------------------------------------------------------------- fig 4
def fig4(toy, null_tbl_out):
    fig, ax = plt.subplots(1, 2, figsize=PAPER_SZ["fig4"] if PAPER else SZ["fig4"], gridspec_kw={"width_ratios": [1.6, 1]})
    faults = ["square", "warp", "edge"]
    x = np.arange(3)
    w = 0.16
    res = {}
    for i, v in enumerate(VARS):
        vals, cens = [], []
        for f in faults:
            m, c = mdf({float(k): toy["power"][f][k][v] for k in toy["power"][f]})
            vals.append(m)
            cens.append(c)
        res[v] = list(zip(vals, cens))
        bars = ax[0].bar(x + (i - 2) * w, vals, w, color=COL[v], label=LAB[v], hatch=None)
        for r, c in zip(bars, cens):
            if c:
                r.set_hatch("///")
                r.set_alpha(0.5)
                ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.04, ">4", ha="center", fontsize=7)
            else:
                ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.04, f"{r.get_height():.2f}", ha="center", fontsize=6.5)
    ax[0].set_xticks(x, ["square", "smooth warp", "edge-aligned"])
    ax[0].set_ylabel("minimum detectable fault at 80% power\n(sd units; hatched = not reached by 4)")
    ax[0].set_ylim(0, 5.2)
    ax[0].legend(fontsize=6.3, loc="upper left", ncol=2, frameon=False)
    ax[0].set_title("Minimum detectable fault per covariance variant", fontsize=9)
    nl = toy["null"]
    xs = np.arange(len(VARS))
    ax[1].bar(xs, [nl[v]["far_at_chi2"] for v in VARS], color=[COL[v] for v in VARS])
    ax[1].axhline(0.05, color="k", ls=":", lw=0.8)
    ax[1].set_xticks(xs, VARS, rotation=25, fontsize=7.5)
    ax[1].set_ylabel("false-alarm rate on clean views")
    ax[1].set_title("False alarms at the chi2 95% threshold\n(dotted = nominal 5%)", fontsize=8.5)
    for xx, v in zip(xs, VARS):
        ax[1].text(xx, nl[v]["far_at_chi2"] + 0.005, f"{nl[v]['far_at_chi2']:.2f}", ha="center", fontsize=7)
    sup(fig, "Toy statistic: detectability and null calibration by covariance variant")
    save(fig, "fig4_toy_mdf_by_variant", FOOT_TOY + " Source: docs/toy-results.json.")
    return res


# ---------------------------------------------------------------------------------------------------------------- fig 5
def fig5(toy, area):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1, 1.1, 0.8]})
    sw = toy["area_sweep_kappa2.5"]
    ar = sorted(float(a) for a in sw)
    for v in VARS:
        ax[0].plot(ar, [sw[str(a)][v] for a in ar], color=COL[v], marker=MK[v], ms=4, ls="--" if v == "known" else "-", label=LAB[v])
    ax[0].set_title("kappa = 2.5, band fault (known, oracle, shrunk overlap at 1.0)", fontsize=8)
    ax[0].set_ylim(0, 1.75)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0].set_xlabel("fraction of the view covered by the fault")
    ax[0].set_ylabel("power")
    ax[0].legend(fontsize=5.8, loc="upper center", ncol=1, frameon=False)
    for kappa, ls in [(2.5, "-"), (4.0, ":")]:
        for v in ["oracle", "kernel", "kernel_nr"]:
            a2 = sorted(area[kappa])
            ax[1].plot(a2, [area[kappa][a][v] for a in a2], color=COL[v], ls=ls, marker=MK[v], ms=3.5,
                       label=f"{v}, kappa {kappa}" if True else None)
    ax[1].set_title("Larger areas to 0.95 (kappa 2.5 solid, 4 dotted)", fontsize=8)
    ax[1].set_ylim(0, 1.45)
    ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_xlabel("fraction of the view covered by the fault")
    ax[1].legend(fontsize=5.8, loc="upper center", ncol=3, frameon=False)
    ax[1].annotate("fault ~ constant offset:\nabsorbed by the fitted shift", xy=(0.95, 0.04), xytext=(0.55, 0.3), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=0.7))
    x = np.arange(len(VARS))
    ax[2].bar(x - 0.2, [toy["shift_kappa8"][v] for v in VARS], 0.38, color="#bbbbbb", label="constant shift")
    ax[2].bar(x + 0.2, [toy["affine_kappa8"][v] for v in VARS], 0.38, color="#1f77b4", label="scale error (affine)")
    ax[2].set_xticks(x, VARS, rotation=30, fontsize=7)
    ax[2].set_title("Global faults, kappa = 8", fontsize=8.5)
    ax[2].set_ylim(0, 1.3)
    ax[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[2].legend(fontsize=6.5, loc="upper center", ncol=2, frameon=False)
    sup(fig, "Toy statistic: power versus fault area, and the two global-fault cases")
    save(fig, "fig5_toy_area_and_global", FOOT_TOY + " Middle panel: " + FOOT_FOLLOW + " Sources: docs/toy-results.json, docs/toy-followups.md.")


# ---------------------------------------------------------------------------------------------------------------- fig 6
def fig6(miscal):
    fig, ax = plt.subplots(1, 3, figsize=PAPER_SZ["fig6"] if PAPER else SZ["fig6"])
    vs = ["oracle", "kernel", "kernel_nr"]
    ms = sorted({r["miscal"] for r in miscal})
    for v in vs:
        rows = [next(r for r in miscal if r["variant"] == v and r["miscal"] == m) for m in ms]
        ax[0].plot(ms, [r["T_over_dof"] for r in rows], color=COL[v], marker=MK[v], label=LAB[v])
        for a, key in ((ax[1], "mdf_square"), (ax[2], "mdf_warp")):
            vals = [mdf_num(r[key]) for r in rows]
            a.plot(ms, [x for x, _ in vals], color=COL[v], marker=MK[v], label=LAB[v])
            for m, (x, c) in zip(ms, vals):
                if c:
                    a.annotate(">4", (m, x), textcoords="offset points", xytext=(0, 5), fontsize=7, ha="center", color=COL[v])
    ax[0].axhline(1, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel("mean T / (n-2) on clean views")
    ax[0].set_title("Null statistic vs chi2 reference (1 = nominal)", fontsize=8.5)
    ax[0].legend(fontsize=6, loc="upper right")
    ax[1].set_ylabel("MDF (sd units)")
    ax[1].set_title("Detectability, square region", fontsize=8.5)
    ax[2].set_ylabel("MDF (sd units; >4 = not reached)")
    ax[2].set_title("Detectability, smooth warp", fontsize=8.5)
    for a in ax:
        a.set_xlabel("spread / true spread\n(<1 over-confident, >1 under)")
        a.set_xscale("log")
        a.set_xticks(ms, [str(m) for m in ms])
    sup(fig, "Toy statistic under a miscalibrated posterior")
    save(fig, "fig6_toy_miscalibration", FOOT_FOLLOW + " Oracle = the covariance the model would report (miscal^2 x true). Source: docs/toy-followups.md.")


# ---------------------------------------------------------------------------------------------------------------- summary
def summary(pipes, guards, toy, mdf_res, miscal, area):
    L = ["# Summary of results on disk", "",
         "All numbers come from results already produced (no new runs for this table). Figures are in this folder; each carries its data provenance in its footer. "
         "Everything marked **toy** is synthetic, one seed, with power standard errors of about 0.05; nothing here involves a trained depth model.", "",
         "| # | Experiment | Data and provenance | Result | Status / caveat | Figure |", "|---|---|---|---|---|---|"]
    r = pipes
    def rowp(v):
        b, a = r[v]["before"], r[v]["after"]
        return (f"view {v}: coverage {100*b['coverage_of_valid_gt']:.1f}->{100*a['coverage_of_valid_gt']:.1f}%, median error "
                f"{100*b['rel_err_median']:.2f}->{100*a['rel_err_median']:.2f}%, p90 {100*b['rel_err_p90']:.2f}->{100*a['rel_err_p90']:.2f}%")
    L.append("| 1 | Leave-one-view-out 2DGS, held-out view vs laser depth | REAL ETH3D pipes, 3 held-out views, 1 run each; surface from the other 13 views' sparse ground-truth depth "
             "(`docs/pipes-2dgs-fit.json`) | " + "; ".join(rowp(v) for v in ["2", "6", "10"]) +
             " | 'before' = initialised surfels, **not optimised**; 'after' = 2000 its (RGB+depth L1, no densification). Uses ground-truth depth as a stand-in for model predictions: tests the machinery, not a depth model | fig1 |")
    pg = lambda k: [g for g in guards if g.get("pipeline") == k and "poisoned" in g]    # noqa: E731
    def cnt(k, key):
        rs = pg(k)
        return f"{sum(1 for g in rs if g[key])}/{len(rs)}"
    L.append("| 2 | Leave-out guards, synthetic room | toy point surface (5 views, CPU) and gsplat 2DGS (5 views, GPU), every view held out once (`figures/data/guards.json`) | "
             f"toy surface: poisoned {cnt('toy point surface (CPU, synthetic room)','poisoned')}, invariant {cnt('toy point surface (CPU, synthetic room)','invariant')}, sensitive {cnt('toy point surface (CPU, synthetic room)','sensitive')}; "
             f"2DGS: poisoned {cnt('gsplat 2DGS surface (GPU, synthetic room)','poisoned')}, bit-identical {cnt('gsplat 2DGS surface (GPU, synthetic room)','invariant')}, sensitive {cnt('gsplat 2DGS surface (GPU, synthetic room)','sensitive')} | pass | fig2 |")
    lk = pg("deliberately leaky fit (must fail)")
    L.append("| 3 | Mutation check: a fit that ignores the exclusion | deliberately leaky fit, views 0/2/4 | "
             f"caught in {sum(1 for g in lk if not (g['poisoned'] and g['invariant']))}/{len(lk)} views (poisoned-view check fails {sum(1 for g in lk if not g['poisoned'])}/{len(lk)}, perturbation check fails {sum(1 for g in lk if not g['invariant'])}/{len(lk)}) | the guard can detect a leak | fig2 |")
    fl = [pipes[v] for v in ["2", "6", "10"]]
    L.append("| 4 | Leave-out guards, real optimised 2DGS | REAL ETH3D pipes, 3 held-out views, 300-iteration fits | "
             f"poisoned view untouched {sum(1 for x in fl if x['poisoned_view_untouched'])}/3; median absolute difference with j perturbed "
             f"{', '.join(f'{x['perturbed_j_abs_diff']['median']:.0e}' for x in fl)} m vs control {', '.join(f'{x['run_to_run_noise_abs_diff']['median']:.0e}' for x in fl)} m | "
             "perturbation effect equals the run-to-run noise floor (CUDA non-determinism); floaters give 2-3 m max differences in the control as well | fig2 |")
    def md(f):
        out = []
        for v in VARS:
            val, cens = mdf_res[v][["square", "warp", "edge"].index(f)]
            out.append(f"{v} >{val:g}" if cens else f"{v} {val:.2f}")
        return "; ".join(out)
    L.append("| 5 | Toy statistic: minimum detectable fault (80% power) | **toy**, 16x16, M=10, 1 seed, 80 trials/cell (`docs/toy-results.json`) | "
             f"square: {md('square')}. warp: {md('warp')}. edge: {md('edge')} (sd units; '>' = not reached by 4) | fitted (s,t) costs ~0 with the true covariance (0.88 vs 0.89); the estimated covariance dominates the loss | fig3, fig4 |")
    nl = toy["null"]
    L.append("| 6 | Toy statistic: false alarms at the chi2(n-2) threshold | **toy**, 200 clean trials | " + "; ".join(f"{v} {nl[v]['far_at_chi2']:.2f}" for v in VARS) +
             f" (nominal 0.05; mean T/(n-2): " + ", ".join(f"{v} {nl[v]['mean_T_over_dof']:.2f}" for v in VARS) + ") | chi2 is unusable for estimated covariances; the empirical null controls the level by construction (assumes exchangeable clean views, untested) | fig4 |")
    L.append("| 7 | Toy statistic: global faults, kappa=8 | **toy** | constant shift power " + ", ".join(f"{v} {toy['shift_kappa8'][v]:.2f}" for v in VARS) +
             "; scale error power " + ", ".join(f"{v} {toy['affine_kappa8'][v]:.2f}" for v in VARS) + " | only a constant offset is invisible; a scale error is flagged because the fitted scale also rescales the sample covariance | fig5 |")
    a25, a4 = area[2.5], area[4.0]
    L.append("| 8 | Toy statistic: power vs fault area (band) | **toy**; kappa 2.5 and 4; areas to 0.95 (`docs/toy-followups.md`) | kappa 2.5, oracle: " +
             ", ".join(f"{a}:{a25[a]['oracle']:.2f}" for a in sorted(a25)) + "; estimated kernel: " + ", ".join(f"{a}:{a25[a]['kernel']:.2f}" for a in sorted(a25)) +
             " | collapses only at 0.95 (near-constant offset); the statistic detects inconsistency but cannot say which side is wrong | fig5 |")
    def mrow(m):
        rs = {r["variant"]: r for r in miscal if r["miscal"] == m}
        return f"x{m}: " + ", ".join(f"{v} {rs[v]['mdf_square']}/{rs[v]['mdf_warp']}" for v in ["oracle", "kernel", "kernel_nr"])
    L.append("| 9 | Toy statistic: miscalibrated posterior | **toy**, spread = 0.6x / 1.0x / 1.6x the truth's (`docs/toy-followups.md`) | MDF square/warp: " + "; ".join(mrow(m) for m in [0.6, 1.0, 1.6]) +
             " | scale error of the spread is absorbed by the empirical threshold; a wrong covariance *shape* (estimated kernel) costs power | fig6 |")
    L += ["", "## Not in this table (not run yet)", "",
          "- Stage 0 (SDXL LoRA sizing) on the TITAN RTX: not run; no SDXL memory or speed number exists.",
          "- Any trained or fine-tuned diffusion model, any calibration against ground truth for a model, any fault-injection result on real model samples.",
          "- Real-data 2DGS fed with predicted (not ground-truth) depth.", "",
          "## Files", "", "| Figure | File (png and pdf) | Data |", "|---|---|---|",
          "| fig1 | `fig1_pipes_2dgs` | real ETH3D pipes, 2DGS before/after |",
          "| fig2 | `fig2_leave_out_guards` | synthetic guards + real pipes guards |",
          "| fig3 | `fig3_toy_power_vs_magnitude` | synthetic toy |",
          "| fig4 | `fig4_toy_mdf_by_variant` | synthetic toy |",
          "| fig5 | `fig5_toy_area_and_global` | synthetic toy |",
          "| fig6 | `fig6_toy_miscalibration` | synthetic toy |", "",
          "Regenerate: `python scripts/run_guard_checks.py` (seconds; GPU optional), then `python scripts/make_figures.py`.", "",
          "## Renders (image panels, `figures/renders/`)", "", "| File | Shows | Data |", "|---|---|---|"]
    desc = {"toy_square": "clean vs faulted residual maps with T and threshold, square fault, true and estimated covariance",
            "toy_warp": "same, smooth warp (the estimated-kernel statistic does NOT flag it)",
            "toy_edge": "same, edge-aligned region"}
    for name, d in desc.items():
        if (FIG / "renders" / f"{name}.png").exists():
            L.append(f"| `{name}.png` | {d} | synthetic Gaussian toy, 1 seed (7) for the paired view; threshold from 150 clean views |")
    for j in (2, 6, 10):
        f = FIG / "renders" / f"pipes_view{j}.png"
        if f.exists():
            L.append(f"| `pipes_view{j}.png` | held-out view {j}: input RGB, rendered RGB, rendered depth, GT depth with validity mask, abs. error | REAL ETH3D pipes |")
    for name, d in (("pipes_guard_poisoned", "poisoned-view guard case"), ("pipes_guard_leaky", "deliberately leaky fit")):
        f = FIG / "renders" / f"{name}.png"
        if f.exists():
            L.append(f"| `{name}.png` | {d}, same panel layout | REAL ETH3D pipes |")
    if not (FIG / "renders" / "pipes_view2.png").exists():
        L += ["", "The ETH3D pipes render panels (held-out views and guard cases) are **not made yet**: no 2DGS parameters were saved by the earlier runs, so they cannot be rendered without re-running the fits."]
    L += ["", "## TikZ / PGFPlots versions (`figures/tikz/`)", "",
          "The six figures and the render panels were redone as PGFPlots/TikZ from the saved data: sources `figures/tikz/*_body.tex`, shared style `preamble.tex`, "
          "data `figures/tikz/data/*.csv` and `*_macros.tex` (made by `scripts/export_tikz_data.py` and `scripts/export_render_images.py`), render images "
          "`figures/tikz/img/`, built PDFs and PNG previews in `figures/tikz/build/` (`figures/tikz/build.sh`). The matplotlib PNG/PDF versions above are kept for comparison.", "",
          "| TikZ figure (`figures/tikz/build/`) | Corresponds to | Data |", "|---|---|---|",
          "| `fig1` | fig1, 2DGS before/after on ETH3D pipes | real ETH3D |",
          "| `fig2` | fig2, leave-out guards | synthetic + real |",
          "| `fig3`, `fig4`, `fig5`, `fig6` | fig3-fig6, toy statistic | synthetic toy, 1 seed |",
          "| `toy_square`, `toy_warp`, `toy_edge` | toy residual-map panels (matrix plots with shared colourbars; T and threshold printed from the data) | synthetic toy, seed 7 |",
          "| `pipes_view2`, `pipes_view6`, `pipes_view10`, `pipes_guard_poisoned`, `pipes_guard_leaky` | ETH3D render panels (images inside TikZ frames, shared depth and error colourbars) | real ETH3D |", "",
          "In the manuscript (`papers/depthcal-gs`) the PGFPlots sources of fig3, fig4, fig6 and the two toy residual-map figures (`toy_square`, `toy_warp`) are included directly "
          "(`figures/tikz/sync_paper.sh` copies them); the ETH3D render panels are not in the paper because they use ground-truth depth as a stand-in for model predictions.", "",
          "### Style sources", "",
          "Figure style (fonts, colours, line weights, axis and legend layout) was taken from your papers:", "",
          "1. M. Ambarkutuk and P. E. Plassmann, *Modeling and analysis of dispersive propagation of structural waves for vibro-localization*, Sensors 24(23):7744, 2024 "
          "(doi 10.3390/s24237744). **Read in full** from your local copy (`~/Desktop/sensors-24-07744.pdf`); figures 5-8 inspected visually.",
          "2. M. Ambarkutuk, S. Alajlouni, P. A. Tarazaga and P. E. Plassmann, *A multi-sensor stochastic energy-based vibro-localization technique with byzantine sensor elimination*, "
          "Sensors 23(23):9309, 2023 (doi 10.3390/s23239309). **Only the text and figure captions were read** (Europe PMC full-text XML): the PDF and figure images could not be fetched "
          "(mdpi.com returned 403, PMC and Europe PMC PDF endpoints sit behind a bot check). Its captions state that proposed and baseline PDFs/CDFs are drawn in blue and brown "
          "and that the occupants are red and black; I used that colour pairing, but I could not see its figure layout. If you drop the PDF in `~/Desktop`, I can check it against this style.", "",
          "What was taken: boxed axes with outward ticks on the left and bottom; DejaVu Sans text (the matplotlib default of the 2024 figures) with sans-serif maths; thin lines with small markers; "
          "red for the proposed (here: fitted scale and shift with the true covariance), black (dashed) for the baseline/bound, blue and brown for the further pair, plus green; "
          "axis labels as \"Quantity [unit]\"; a framed legend with a light border; `(a)`, `(b)`, `(c)` panel labels below each panel; viridis for depth maps; red x / black + markers; "
          "bold best values in tables (not used here). Differences from the papers: the figures here are vector PGFPlots, not matplotlib bitmaps, and legends are shared above the panels where five "
          "series would not fit inside a small panel.", ""]
    (FIG / "summary.md").write_text("\n".join(L) + "\n")


def main():
    global PAPER
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    pipes = json.load(open(ROOT / "docs" / "pipes-2dgs-fit.json"))
    toy = json.load(open(ROOT / "docs" / "toy-results.json"))
    guards = json.load(open(FIG / "data" / "guards.json"))
    miscal, area = parse_followups(ROOT / "docs" / "toy-followups.md")
    if args.paper:
        PAPER = True
        plt.rcParams.update({"font.size": 7, "legend.fontsize": 5.5, "axes.titlesize": 7.5})
        fig3(toy)
        fig4(toy, None)
        fig6(miscal)
        print("written (paper):", sorted(p.name for p in (FIG / "paper").iterdir()))
        return
    fig1(pipes)
    fig2(guards, pipes)
    fig3(toy)
    res = fig4(toy, None)
    fig5(toy, area)
    fig6(miscal)
    summary(pipes, guards, toy, res, miscal, area)
    print("written:", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
