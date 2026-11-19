# sensorfault

Research code for **auditing monocular depth with per-source information on Gaussian surfels** (design v4.0;
plan `PLAN.md`; method `docs/learned-indicator.md`; protocol `docs/learned-indicator/experiment-design.md`; paper
design `docs/learned-indicator/paper-plan.md`). Setting: posed aerial video in strips with one off-the-shelf
monocular depth map per frame (MatrixCity for exact depth, ETH3D as a real-scene check).

A depth-supervised surfel map looks complete and can be confidently wrong over whole surfaces, identically in every
frame, because averaging depth maps removes only the part of their error that differs between frames. Each surface
entity therefore carries two information records on the same three placement coordinates (normal offset and two
tilts): one from the depth prior, one from the pixels, the second computed through the differentiable renderer. Their
disagreement is a statistic with a null distribution; the pixel information is calibrated on the scene by frame
replicates and by displacements injected through the renderer; the prior information saturates. No features are
matched and no fault labels are used. The earlier depth-only view gate (design v3.4,
`docs/plan-v3-view-gate.md`) is kept as the depth-consensus baseline.

**Status:** nothing of design v4 is measured yet. Existing and tested: the depth-only surfel fit
(`src/sensorfault/gs/`), the reference renderer with compositing weights and depth Jacobians, the information blocks
and the consensus gate (`src/sensorfault/eval/`), the depth cache for four models (`src/sensorfault/depthcache/`),
and the controlled experiments E1 and the pose check on one MatrixCity block (`experiments/`, `results/`). To build:
the image record, injection through the renderer, the two-source fusion and test (work packages in `PLAN.md` §6).
Design v1 (sensor-level elimination) is kept in `configs/legacy_sensor_faults/`.

## Layout

```
sensorfault/
  PLAN.md                 research plan
  pyproject.toml          package metadata, extras, ruff/pytest config
  environment.yml         conda env (Python 3.10)
  Makefile                env / install / test / lint / data-help / clean
  configs/                design-v4 stage configs (stage0, stage A-C, real scenes), datasets, baselines
  configs/legacy_view_gate/       design-v3 configs (depth-only view gate: H1-H7, breakdown, ablations); schema-tested
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

| Stage | Question | Hypotheses | Config |
|---|---|---|---|
| 0 | Is the image record a measurement on a synthetic entity? Are records additive? | H3, H14 | `configs/stage0_entity.yaml` |
| A | Is a depth-only information map overconfident, and by rho + (1 - rho)/n? | H1, H2 | `configs/stage_a_depth_only_honesty.yaml` |
| B | Is the pixel information honest, and how far does it reach? | H3-H6 | `configs/stage_b_image_information.yaml` |
| C | Are repeated depth faults detected where consensus is blind, and is the fused map calibrated? | H7-H10 | `configs/stage_c_two_source_map.yaml` |
| real | Do the exact-depth findings reappear on scenes with laser truth? | H1, H2 | `configs/real_scenes_check.yaml` |

All are `preregistered_draft` and flagged `implemented: false`; thresholds are proposals until the
pre-registration tag (`prereg/`). The manuscript is in `papers/sensorfault/` at the repository root.

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
| Stanford Bunny (Stanford 3D Scanning Repository) | Research use and free redistribution with acknowledgement; no commercial use | The one committed third-party data file: a ~70 KB decimated copy, `src/sensorfault/sim/assets/stanford_bunny_8mm.obj`, for offline tests (regenerate with `scripts/data/bunny.py`); not MIT |

Check each upstream licence yourself before use; this table is a summary of PLAN.md section 6,
not legal advice.

## Data hygiene

Public datasets and public sensor models only. Nothing under `data/`, `outputs/`,
`third_party/` or checkpoints is committed. The one exception is the small test mesh above.
