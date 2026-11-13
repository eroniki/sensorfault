# sensorfault

Research scaffold for sensor-level fault elimination in a shared Gaussian-splat surface.
Several depth sensors observe one scene; a consensus render of the shared surface is used to
test whole sensors (or whole frames) against the rest and exclude coherent faults (calibration
drift, range bias, multipath-like errors) before dense fusion. The repo also provides a
multi-sensor depth fusion benchmark with deterministic fault injection and public baselines.
See `PLAN.md` for the research plan.

## Layout

```
sensorfault/
  PLAN.md                 research plan
  pyproject.toml          package metadata, extras, ruff/pytest config
  environment.yml         conda env (Python 3.10)
  Makefile                env / install / test / lint / data-help / clean
  configs/                experiment configs, one family per hypothesis (H1-H5)
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

| Hypothesis | Claim (short) | Configs |
|---|---|---|
| H1 | Sensor-level elimination beats voting/filtering baselines under one coherent fault, K>=3 | `configs/h1_*.yaml` |
| H2 | Entropy-based criterion beats leave-one-out chi-square only where Gaussian assumptions break | `configs/h2_*.yaml` |
| H3 | Visibility + grazing weighting keep clean-run false elimination <=5% | `configs/h3_*.yaml` |
| H4 | Fixed-N splat budget resists absorbing the faulty sensor vs adaptive densification | `configs/h4_*.yaml` |
| H5 | Pretrained learned fusion degrades under smooth bias/drift; fine-tuning narrows, not closes, the gap | `configs/h5_*.yaml` |

Exact config file names are owned by the configs/ directory; see PLAN.md section 5 for the full
statements and the pre-registration.

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
