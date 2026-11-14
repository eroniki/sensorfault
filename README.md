# sensorfault

Research scaffold for **view-level fault rejection in depth-supervised Gaussian splatting** (design v3). One
metric depth model is applied to each image; 2DGS-style surfels are fitted by depth supervision and rendered with gsplat;
colour is a separate appearance block that never updates geometry. Each surfel keeps its shape for rendering plus a
separate 9x9 uncertainty whose normal-offset and tilt block renders into depth variance. Views add alpha-weighted
J^T R^-1 J information; leaving a view out is a subtraction; the other views' surface is rendered into the tested view
and its whitened depth residuals are tested (depth only, no correspondences, unseen pixels untestable, the tested view
spawns nothing). Faults are regional and appearance-driven (reflections, textureless walls); the gate must beat cheap
baselines (aligned RMS, depth confidence, robust depth losses, RobustNeRF, SpotLessSplats). See `PLAN.md`.

**Status:** configs, registry stubs, the detection theory (`src/sensorfault/eval/detection_theory.py`) and the rendered-
uncertainty theory (`src/sensorfault/eval/gaussian_uncertainty.py`), both tested, exist. The residual measurement on
ETH3D (the first task), the splatting pipeline and the gate on real data are **not implemented yet**. Design v1
(sensor-level elimination) is kept in `configs/legacy_sensor_faults/`, with its implemented pieces still under `src/`.

## Layout

```
sensorfault/
  PLAN.md                 research plan
  pyproject.toml          package metadata, extras, ruff/pytest config
  environment.yml         conda env (Python 3.10)
  Makefile                env / install / test / lint / data-help / clean
  configs/                design-v3 experiment configs (H1-H6, breakdown, ablations, datasets, baselines)
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
| H1 | Residuals first (go/no-go): fit sigma, ell, nu, tails of Depth Pro on ETH3D; characterise natural regional failures and their harm | `configs/h1_residual_measurement.yaml` |
| H2 | The depth model repeats its error across views in textureless regions (gate blind there) but not on reflections | `configs/h2_cross_view_repetition.yaml` |
| H3 | At 5% false alarm the gate beats the best cheap/robust baseline by >=15 points on reflective-region failures | `configs/h3_detection_vs_cheap_baselines.yaml` |
| H4 | Depth-supervised splatting with the gate improves AbsRel and F-score on ETH3D and ScanNet++ | `configs/h4_reconstruction_with_and_without_gate.yaml` |
| H5 | Independent blocks are overconfident; calibrated false alarm stays in [0.5, 1.5] alpha | `configs/h5_rendered_uncertainty_calibration.yaml` |
| H6 | Ablations: leave-view-out, no spawning, whitening, rendered 9x9 vs FisherRF-style and Bayes' Rays-style | `configs/h6_gate_design_ablations.yaml` |

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
