# sensorfault

Research scaffold for a **view-level gate** that decides which monocular depth sources to trust
before pose graph optimisation and fusion (design v2). The gate is the contribution; the pipeline
(RGB images, an ensemble of monocular depth estimators, a partly end-to-end pose graph, fusion) is
the test bed. The gate tests each source once against a single shared Gaussian-splat surface
(cross-fitted, fixed budget) that also serves as the fused output. Baselines the gate must beat: robust kernels in the pose graph, COLMAP's geometric
consistency filter, plain voting; MASt3R-SfM and VGGT are reference points. See `PLAN.md`.

**Status:** configs, registry stubs and tests exist; the test bed, view-level fault injectors and
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
| H1 | The view-level gate beats robust kernels, the COLMAP filter and plain voting under coherent view-level faults | `configs/h1_view_level_gate_vs_baselines.yaml` |
| H2 | The evidence criterion beats leave-one-source-out chi-square only where Gaussian assumptions break | `configs/h2_evidence_vs_chi2.yaml` |
| H3 | Surface fitted without the source's view (cross-fitting), alignment and a support mask keep clean false exclusion <=5% | `configs/h3_cross_fitting_alignment_support.yaml` |
| H4 | Gating before pose graph optimisation beats gating after; gate + robust kernel >= either alone | `configs/h4_gate_placement.yaml` |
| H5 | Price of gating on clean data <=5%; reference points (MASt3R-SfM, VGGT) are context only | `configs/h5_clean_cost_reference_points.yaml` |
| H6 | Shared splat surface vs pairwise checks: cost grows with views not view pairs (dense overlap), non-inferior power, fixed budget beats adaptive densification | `configs/h6_shared_surface_vs_pairwise.yaml` |

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
