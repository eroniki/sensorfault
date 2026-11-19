# Papers and code: sensorfault and depthcal-gs

One repository for two papers and their code. Each of the four workspaces was a repository of its own and was
merged here with its full commit history. The imported commits keep their original paths, so they appear in
`git log` and on the second parent of each import merge, not under `git log -- <directory>`.

| Workspace | Content | Start here |
|---|---|---|
| `code/sensorfault/` | Auditing monocular depth with per-source information on Gaussian surfels: surfel fit, information records, baselines, experiments | `code/sensorfault/PLAN.md`, `code/sensorfault/README.md` |
| `papers/sensorfault/` | The RA-L letter for it (LaTeX), and the earlier view-gate letter with its supplement | `papers/sensorfault/README.md` |
| `code/depthcal-gs/` | depthcal-gs code, configs, figures and training notes | `code/depthcal-gs/PLAN.md` |
| `papers/depthcal-gs/` | The depthcal-gs manuscript (LaTeX) | `papers/depthcal-gs/README.md` |

## Working in a workspace

Each code workspace is its own Python project with its own `pyproject.toml`; install and test from inside it:

```bash
cd code/sensorfault && pip install -e ".[dev]" && pytest -q tests/core && ruff check .
cd code/depthcal-gs && pip install -e . && pytest -q
```

Each paper builds from inside its directory with `scripts/build.sh` (TeX Live with `latexmk`). The sensorfault figure
generators (`code/sensorfault/experiments/*_figures_tikz.py`) write into `papers/sensorfault/figures/` by default.
`papers/depthcal-gs/.githooks/pre-commit` rebuilds its committed PDF; enable it with
`git config core.hooksPath papers/depthcal-gs/.githooks` if wanted.

## Licences and data

Code under `code/sensorfault/` is MIT (`code/sensorfault/LICENSE`, `NOTICE`); see each workspace for the terms of the
datasets and models it uses. No dataset, checkpoint or third-party code is committed. The repository is private; the
letters are under double-anonymous review rules, so check each venue before making it public.
