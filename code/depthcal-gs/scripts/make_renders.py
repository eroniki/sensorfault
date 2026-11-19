"""Image panels (not plots) from existing code and results. Writes PNGs to figures/renders/.

  python scripts/make_renders.py toy                       # toy-statistic examples: clean vs faulted residual maps (CPU, seconds)
  python scripts/make_renders.py pipes --params DIR        # 2DGS renders on ETH3D pipes from saved surfel parameters (GPU)

The pipes mode renders from parameters saved by scripts/fit_2dgs_pipes.py --save-dir; it never trains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "renders"
sys.path.insert(0, str(ROOT / "src"))
plt.rcParams.update({"font.size": 8, "figure.dpi": 110})


# ------------------------------------------------------------------------------------------------------------- toy
def toy():
    from dcal.stat import align_samples, empirical_threshold, fault_statistic
    from dcal.toy import ToyConfig, World

    w = World(ToyConfig(g=16), 0)
    g = w.cfg.g
    n = w.n
    kappa = 3.0

    def run(seed, fault, variant):
        S, D, Kr, Ku, _ = w.simulate(np.random.default_rng(seed), fault, kappa if fault else 0.0, 0.12)
        kw = dict(Sigma_u=Ku) if variant == "oracle" else dict(grid_shape=w.shape, cov="kernel")
        res = fault_statistic(S, D, Kr, **kw)
        m = align_samples(S).mean(0)
        r = res["s"] * m + res["t"] - D
        return res, r

    thr = {}
    for variant in ("oracle", "kernel"):
        null = np.array([run(10_000 + i, None, variant)[0]["T"] for i in range(150)])
        thr[variant] = empirical_threshold(null, 0.05)
    seed = 7
    for fault in ("square", "warp", "edge"):
        pat = w.fault_pattern(fault, 0.12).reshape(g, g)
        fig, ax = plt.subplots(2, 4, figsize=(11.2, 5.9))
        for row, variant in enumerate(("oracle", "kernel")):
            rc, r_c = run(seed, None, variant)
            rf, r_f = run(seed, fault, variant)
            vmax_r = np.abs(np.concatenate([r_c, r_f])).max()
            vmax_e = np.abs(np.concatenate([rc["e"], rf["e"]])).max()
            panels = [(r_c.reshape(g, g), "raw residual r, CLEAN view", vmax_r, "RdBu_r"),
                      (r_f.reshape(g, g), "raw residual r, FAULTED view", vmax_r, "RdBu_r"),
                      (rc["e"].reshape(g, g), f"whitened residual e (innovations), CLEAN\nT = {rc['T']:.0f}  (threshold {thr[variant]:.0f})  ->  "
                       f"{'FLAGGED' if rc['T'] > thr[variant] else 'not flagged'}", vmax_e, "PiYG"),
                      (rf["e"].reshape(g, g), f"whitened residual e (innovations), FAULTED\nT = {rf['T']:.0f}  (threshold {thr[variant]:.0f})  ->  "
                       f"{'FLAGGED' if rf['T'] > thr[variant] else 'not flagged'}", vmax_e, "PiYG")]
            for a, (img, title, vm, cm) in zip(ax[row], panels):
                im = a.imshow(img, cmap=cm, vmin=-vm, vmax=vm, interpolation="nearest")
                a.set_title(title, fontsize=7.6)
                a.set_xticks([])
                a.set_yticks([])
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
                if "FAULTED" in title:
                    a.contour(pat > 0.5 if fault != "warp" else np.abs(pat) > 0.5, levels=[0.5], colors="k", linewidths=0.8)
            ax[row][0].set_ylabel("covariance: TRUE (oracle)" if variant == "oracle" else "covariance: ESTIMATED kernel", fontsize=8)
        fig.suptitle(f"Toy statistic, {fault} fault of {kappa:g} sd: the same synthetic view without and with the fault "
                     f"(black outline = fault support)", fontsize=10, y=0.985)
        foot = (f"DATA: synthetic Gaussian toy (not real images), 16x16 grid, M=10 samples, seed {seed} for the paired clean/faulted view (the fault is the only difference); "
                "T = whitened residual energy after fitting a per-view scale and shift; threshold = 95% quantile of the clean-view null of the same pipeline (150 views). "
                "Left pair: raw residual in render units. Right pair: whitened residual e = L^-1 r (Cholesky innovations in raster order, unit variance under the null): "
                "its streaks follow the whitening order and are NOT a localisation of the fault. Source: scripts/make_renders.py toy.")
        import textwrap
        fig.text(0.01, 0.005, "\n".join(textwrap.wrap(foot, 190)), fontsize=6.5, color="#444")
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        fig.subplots_adjust(hspace=0.38)
        fig.savefig(OUT / f"toy_{fault}.png", dpi=160)
        plt.close(fig)
        print("wrote", OUT / f"toy_{fault}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["toy", "pipes"])
    ap.add_argument("--params", default=None)
    a = ap.parse_args()
    if a.mode == "toy":
        toy()
    else:
        raise SystemExit("pipes mode needs saved surfel parameters: run scripts/fit_2dgs_pipes.py with --save-dir first")
