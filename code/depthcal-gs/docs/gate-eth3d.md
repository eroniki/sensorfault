# ETH3D gate (PLAN sec. 10 and 11.1)

Status: **script written and CPU-tested (2026-11-17), not yet run on real samples.** Code: `src/dcal/gate.py` (CPU), `scripts/gate_eth3d.py` (GPU steps + CLI), tests in `tests/test_gate.py`.

## What it asks

On every view of a held-out ETH3D scene, with unguided samples from a stage-1 checkpoint:

1. **Is the sample spread informative about the error?** Spearman correlation of per-bin spread with per-bin absolute error against the laser GT (mean fitted to GT by scale/shift), and z-score coverage at 1 and 2 sd (0.68 / 0.95 if calibrated).
2. **Is the clean-view statistic well behaved?** T/(n-2) on clean windows, and the false-alarm rate at a leave-one-view-out empirical-null threshold.
3. **Does the statistic catch injected depth-space faults** (square, edge-aligned, smooth warp; kappa x the median per-bin total sd) **better than a trivial baseline?** The baseline is `iso`: the same fit and test with an isotropic covariance that ignores the model's spread. The script prints `GATE <cov>: PASS` if that covariance beats `iso` by more than 0.1 in power on some fault family and reaches 0.8 power at the verdict kappa (default 2).

The leave-j-out render is built from the **other views' sparse GT depth** (as in `scripts/fit_2dgs_pipes.py`), a stand-in for a surface from model predictions. So this tests the model's spread and the statistic, not the surface.

## Running it on the TITAN

Pause training first (`kill -TERM $(cat $STAGE1_WORK/pid)` checkpoints and exits; `scripts/stage1_run.sh` resumes the same run).

```
CK=~/workspace/depthcal-runs/stage1/ckpt_0002000.pt
PYTHONPATH=src python scripts/gate_eth3d.py sample --pretrained $SDXL_DIR --weights $CK --eth3d ~/datasets/eth3d --scene pipes \
    --cache runs/gate/pipes_s2000 --samples 10 --ddim-steps 1 4 10
PYTHONPATH=src python scripts/gate_eth3d.py render --eth3d ~/datasets/eth3d --scene pipes --cache runs/gate/pipes_s2000
PYTHONPATH=src python scripts/gate_eth3d.py stat --cache runs/gate/pipes_s2000 --ddim-steps 1 4 10 --out runs/gate/pipes_s2000/gate.json
```

Estimated GPU time [EST]: sampling 14 views x 10 samples x (1 + 4 + 10) steps, a few minutes per view. Renders about 2 min per view (the `pipes` fits took 100-130 s), so about 30 min. The `stat` step is CPU only and can be re-run with other `--grid`, `--tiles` or `--rel-sigma-r` values. Renders do not depend on the checkpoint, so for a later checkpoint pass `--renders runs/gate/pipes_s2000`.

## Limits to state with any result

- `pipes` has 14 views. With 2 x 2 windows per view that is 56 null values, but windows of one view are correlated, so the null is thinner than it looks. Add more ETH3D scenes with a repeated `--cache` (each scene's renders and samples in its own directory) before reading small differences.
- The render noise is a diagonal `rel_sigma_r x depth` (default 0.5%; the `pipes` fits measured 0.1-0.6% relative error). It has no per-surfel variance.
- The 1-step vs 10-step question from `docs/stage1-titan.md` is answered here only for this scene: run `--ddim-steps 1 4 10` and compare.

## Results

None yet.
