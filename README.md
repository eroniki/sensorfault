# sensorfault

Research scaffold for **view-level fault detection and exclusion (FDE), in the spirit of GNSS RAIM, for
learned-depth reconstruction** (design v2.4). One metric depth model is applied to each image; the gate tests each
depth map against a shared Gaussian-splat surface fitted from the other views (fixed budget, cross-fitted), excludes
faulty views before pose graph optimisation and fusion, and raises a no-consensus alarm. Core argument: a coherent
view-level fault that the monocular scale/offset cannot absorb is sub-noise at each pixel (invisible to pixel filters
and robust kernels) but large when aggregated over the view and whitened by an explicit residual covariance
(minimum detectable bias ~ 1/sqrt(n_eff f (1-f)); a uniform bias over a whole view is an offset, not a fault). The gate is the contribution; a
differentiable, renderable pose graph is the test bed; soft down-weighting is a baseline; a multi-model ensemble
is an ablation; MASt3R-SfM and VGGT are reference points. See `PLAN.md`.

**Status:** configs, registry stubs, the measurement-model and detection theory (`src/sensorfault/eval/detection_theory.py`:
Sigma_r, nuisance projection, kappa, MDBs, n_eff, pose Jacobian, ellipsoid protection level; tested, Monte Carlo cross-checked) and config tests exist; the test bed, view-level fault injectors and
the gate itself are **not implemented yet**. Design v1 (sensor-level elimination on a splat surface)
is kept in `configs/legacy_sensor_faults/`, with its implemented pieces still under `src/`.

## Layout

```
sensorfault/
  PLAN.md                 research plan
  pyproject.toml          package metadata, extras, ruff/pytest config
  environment.yml         conda env (Python 3.10)
  Makefile                env / install / test / lint / data-help / clean
  configs/                design-v2 experiment configs (H1-H5, breakdown, ablations, datasets, baselines)
  configs/legacy_sensor_faults/   design-v1 configs (physical-sensor faults); still exercised by tests
  scripts/                data download helpers, third_party setup (nothing runs by default)
  src/sensorfault/
    types.py              DepthFrame, FusionResult, Fuser protocol
    baselines/            registry (base.py) + wrappers for external fusers
    sim/ faults/ elim/ eval/   simulator, fault injection, elimination, metrics
  tests/                  CPU-only pytest suite
  docs/DEVELOPMENT.md     conventions: seeds, determinism, dev/test split
  data/ outputs/ third_party/   git-ignored
```

## Quickstart

```bash
conda env create -f environment.yml && conda activate sensorfault   # or: make env
make install        # editable install, CPU only
make test
make lint
make data-help      # prints dataset instructions; downloads nothing
```

Optional extras: `pip install -e ".[tsdf]"` (Open3D), `".[gs]"` (torch, gsplat),
`".[learned]"`. Install the torch build matching your CUDA driver first.

## Hypotheses and configs

| Hypothesis | Claim (short) | Config |
|---|---|---|
| H1 | Regime map: per-pixel bias vs fraction (f<=0.8) of the view, plus the affine control (f=1: absorbed, not a fault); gate wins sub-noise/coherent, pixel filters win sparse/large | `configs/h1_regime_map_subnoise_bias.yaml` |
| H2 | Consequence for trajectory and fused surface vs robust kernels, soft weights, COLMAP filter, voting | `configs/h2_consequence_vs_baselines.yaml` |
| H3 | Evidence vs chi-square only where Gaussian assumptions break; region scan vs energy by fault shape | `configs/h3_statistic_choice.yaml` |
| H4 | Cross-fitting, alignment and a support mask keep clean false alarm <=5% | `configs/h4_cross_fitting_alignment_support.yaml` |
| H5 | Gate before PGO beats gating after and soft down-weighting; gate + kernel >= either | `configs/h5_placement_soft_downweighting.yaml` |
| H6 | Adequacy of the measurement model; price of monitoring (false alarm <=5%, availability >=90%, error change <=5%); reference points are context | `configs/h6_price_of_monitoring_reference_points.yaml` |
| H7 | Shared splat surface vs pairwise checks: cost, non-inferior power, fixed budget beats adaptive densification | `configs/h7_shared_surface_vs_pairwise.yaml` |
| H8 | Pose faults (spoofed prior, glitches, false loop closures); loop-closure baselines expected to match the gate on large ones | `configs/h8_pose_faults.yaml` |
| H9 | Conditional protection level coverage (optional) | `configs/h9_protection_level.yaml` |
| H10 | Ensemble ablation: shared failures; where cross-model voting breaks; value of the ensemble | `configs/h10_ensemble_ablation.yaml` |

Breakdown scenarios: `configs/breakdown_*.yaml`; ablations: `configs/ablations_gate.yaml`. All are
`preregistered_draft` and flagged `implemented: false`. The manuscript is in the private repo
`sensorfault-manuscript`.

## Licence caveats

The original code is MIT (see `LICENSE`). `NOTICE`: no third-party code is vendored.

| Component | Licence / terms | Consequence here |
|---|---|---|
| OpenMVS | AGPL-3.0 | Not vendored; run as external binary via subprocess only; do not link or copy |
| 3DGS-family code (original 3DGS and derivatives) | Research / non-commercial | Install into `third_party/`, not redistributed; no commercial use |
| Skoltech3D dataset | CC BY-NC 4.0 | Non-commercial, attribution; do not commit data |
| Replica dataset | Research-only, no redistribution | Never publish renderings; release generator + seeds |
| DFC2019 / US3D | Source conflict (CC BY 4.0 listing vs contest terms banning redistribution); publication needs case-by-case approval | Optional section; obtain approval before publishing results |
| IARPA MVS3DM | Availability to be verified | Secondary |

Check each upstream licence yourself before use; this table is a summary of PLAN.md section 6,
not legal advice.

## Data hygiene

Public datasets and public sensor models only. Nothing under `data/`, `outputs/`,
`third_party/` or checkpoints is committed.
